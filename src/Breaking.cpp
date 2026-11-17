// Destruction in the renderer (step 1.3): breakable boxes as pieces and the
// bodies that carry them. The cutting and the graph are in heliora_core
// (Fracture.h, BreakGraph.h); this is where they meet the scene, the physics
// world and the nodes that draw the pieces.
#include "VulkanRenderer.h"

#include "ScenePhysics.h"

#include <glm/gtc/matrix_transform.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <format>
#include <unordered_set>

namespace {

// Faces this close count as touching (objects linked across) or resting on
// something (an anchor).
constexpr float kTouchTolerance = 0.01f; // m
constexpr float kAnchorReach = 0.05f;    // m
// How much of a hit carries on into the pieces it frees, and the fastest it
// throws them: the ball bounced off the object before it broke, so without
// this the freed pieces would sit where they were, stacked like dry stone.
constexpr float kHitCarry = 1.0f;
constexpr float kFastestPiece = 18.0f; // m/s
// Cut faces: the object's colour, darker, like the raw inside of things.
glm::vec3 rawColour(glm::vec3 c) { return c * 0.55f; }

glm::mat4 poseMatrix(const PhysicsPose &pose) {
  return glm::translate(glm::mat4(1.0f), pose.position) * glm::mat4_cast(pose.rotation);
}

PhysicsPose entryPose(const SceneModelEntry &entry) {
  return {entry.transform.position, rotationFromDegrees(entry.transform.rotationDeg)};
}

} // namespace

int VulkanRenderer::breakableOf(uint64_t userData) const {
  if (!(userData & kBreakBodyFlag))
    return -1;
  const uint64_t body = userData & ~kBreakBodyFlag;
  if (body >= breakBodies.size() || breakBodies[body].pieces.empty())
    return -1;
  return breakPieces[breakBodies[body].pieces.front()].object;
}

void VulkanRenderer::clearBreakables() {
  if (physics)
    for (const BreakBody &body : breakBodies)
      if (body.id.isValid())
        physics->removeBody(body.id);
  std::unordered_set<const SceneNode *> nodes;
  for (const BreakPiece &piece : breakPieces)
    nodes.insert(piece.node);
  if (!nodes.empty())
    rootNode.removeChildren([&](const SceneNode &node) { return nodes.contains(&node); });
  breakables.clear();
  breakPieces.clear();
  breakBodies.clear();
  breakLinks.clear();
  breakLinkPoints.clear();
  pieceLinks.clear();
  breakGraphReady = false;
  breakablesDirty = false;
}

void VulkanRenderer::rebuildBreakables() {
  clearBreakables();
  if (!physics)
    return;
  for (int i = 0; i < static_cast<int>(currentScene.models.size()); ++i) {
    const SceneModelEntry &entry = currentScene.models[i];
    if (!entry.breakPieces || !entry.body || entry.shape != PrimitiveShape::Box)
      continue;
    Breakable object;
    object.entry = {SceneEntryRef::Kind::Model, i};
    const glm::vec3 size = entry.size * entry.transform.scale;
    object.key = std::format("{} {} {}|{}|{}", size.x, size.y, size.z, *entry.breakPieces,
                             entry.breakSeed);
    std::shared_ptr<const Fracture> &cut = fractures[object.key];
    if (!cut)
      cut = std::make_shared<const Fracture>(fractureBox(size, *entry.breakPieces, entry.breakSeed));
    object.fracture = cut;
    object.firstPiece = static_cast<int>(breakPieces.size());
    object.strength = entry.breakStrength.value_or(kDefaultBreakStrength);
    object.moving = *entry.body != PhysicsMotion::Static;
    const int objectIndex = static_cast<int>(breakables.size());
    std::vector<int> pieces;
    std::vector<PhysicsPose> worlds;
    const PhysicsPose pose = entryPose(entry);
    for (int k = 0; k < static_cast<int>(cut->pieces.size()); ++k) {
      // Its own model: a piece is a shape of its own. Kept in the model
      // cache, so a rebuild (Stop, an edit) makes nothing new.
      const glm::vec3 c = entry.color;
      const std::string modelKey =
          std::format("#piece|{}|{}|{} {} {}", object.key, k, c.r, c.g, c.b);
      int modelId = modelManager.findModel(modelKey);
      if (modelId < 0) {
        modelId = modelManager.createModel(modelKey,
                                           makePieceMesh(cut->pieces[k], c, rawColour(c)), false,
                                           false, device, textureManager, descriptorManager);
        gpuDrivenGBufferPass.registerModelGeometry(modelId, modelManager);
      }
      auto node = std::make_unique<SceneNode>();
      node->setModelId(modelId);
      node->setEntry(object.entry);
      node->setPose(pose.position, pose.rotation);
      BreakPiece piece;
      piece.object = objectIndex;
      piece.index = k;
      piece.node = node.get();
      rootNode.addChild(std::move(node));
      pieces.push_back(static_cast<int>(breakPieces.size()));
      worlds.push_back(pose);
      breakPieces.push_back(piece);
    }
    breakables.push_back(std::move(object));
    addBreakBody(pieces, worlds, !breakables.back().moving, glm::vec3(0.0f), glm::vec3(0.0f));
  }
  physics->optimizeBroadPhase();
  if (playing)
    prepareBreakGraph();
}

PhysicsPose VulkanRenderer::pieceWorld(const BreakPiece &piece) const {
  if (piece.body >= 0 && breakBodies[piece.body].id.isValid()) {
    const PhysicsPose body = physics->pose(breakBodies[piece.body].id);
    return {body.position + body.rotation * piece.localPosition,
            body.rotation * piece.localRotation};
  }
  return entryPose(currentScene.models[breakables[piece.object].entry.index]);
}

// A body for these pieces where they are now: its frame is the first
// piece's, and each piece's mesh frame is kept inside it, so nothing moves
// on screen when bodies change.
int VulkanRenderer::addBreakBody(const std::vector<int> &pieces,
                                 const std::vector<PhysicsPose> &worlds, bool fixed,
                                 glm::vec3 velocity, glm::vec3 spin,
                                 const std::vector<BreakHit> &hits) {
  const PhysicsPose frame = worlds.front();
  const glm::quat toLocal = glm::conjugate(frame.rotation);
  auto parts = std::make_shared<PhysicsMeshParts>();
  std::vector<std::pair<glm::vec3, glm::quat>> locals;
  float volume = 0.0f;
  glm::vec3 middle(0.0f); // world, volume weighted
  for (size_t i = 0; i < pieces.size(); ++i) {
    const BreakPiece &piece = breakPieces[pieces[i]];
    const FracturePiece &shape = breakables[piece.object].fracture->pieces[piece.index];
    const glm::vec3 lp = toLocal * (worlds[i].position - frame.position);
    const glm::quat lr = toLocal * worlds[i].rotation;
    PhysicsMeshPart part;
    for (const FractureFace &face : shape.faces)
      for (const glm::vec3 &q : face.points)
        part.positions.push_back(lp + lr * q);
    parts->parts.push_back(std::move(part));
    locals.push_back({lp, lr});
    volume += shape.volume;
    middle += (worlds[i].position + worlds[i].rotation * shape.centre) * shape.volume;
  }
  middle /= std::max(volume, 1e-6f);
  const Breakable &object = breakables[breakPieces[pieces.front()].object];
  const SceneModelEntry &entry = currentScene.models[object.entry.index];
  PhysicsBodyDesc desc = bodyDescFor(entry, nullptr, physicsPanel.defaultFriction);
  desc.shape = PhysicsShape::Hulls;
  desc.parts = parts;
  desc.scale = glm::vec3(1.0f);
  desc.motion = fixed ? PhysicsMotion::Static : PhysicsMotion::Dynamic;
  desc.position = frame.position;
  desc.rotation = frame.rotation;
  if (entry.mass) {
    // The object's mass spread over its volume, so a chunk weighs its share.
    const glm::vec3 size = entry.size * entry.transform.scale;
    desc.density = *entry.mass / (size.x * size.y * size.z);
    desc.mass = 0.0f;
  }
  // This frame's hits push what they freed: the share of each hit that
  // reaches it (as for breaking links), over its mass, and a tumble from
  // being hit off its middle.
  if (!fixed) {
    const float mass = desc.mass > 0.0f ? desc.mass : volume * desc.density;
    for (const BreakHit &hit : hits) {
      const float d = glm::distance(middle, hit.point) / kBreakSpread;
      glm::vec3 kick = hit.push * (kHitCarry * hit.impulse / (1.0f + d * d) / std::max(mass, 1e-3f));
      if (glm::length(kick) > kFastestPiece)
        kick = glm::normalize(kick) * kFastestPiece;
      velocity += kick;
      spin += glm::cross(middle - hit.point, kick) * 1.5f;
    }
  }
  desc.linearVelocity = velocity;
  desc.angularVelocity = spin;
  desc.reportImpacts = true;
  const int index = static_cast<int>(breakBodies.size());
  desc.userData = kBreakBodyFlag | static_cast<uint64_t>(index);
  BreakBody body;
  body.id = physics->addBody(desc);
  body.pieces = pieces;
  body.fixed = fixed;
  breakBodies.push_back(std::move(body));
  for (size_t i = 0; i < pieces.size(); ++i) {
    BreakPiece &piece = breakPieces[pieces[i]];
    piece.body = index;
    piece.localPosition = locals[i].first;
    piece.localRotation = locals[i].second;
  }
  return index;
}

void VulkanRenderer::poseBreakBody(const BreakBody &body, const PhysicsPose &pose) {
  for (int p : body.pieces) {
    const BreakPiece &piece = breakPieces[p];
    piece.node->setPose(pose.position + pose.rotation * piece.localPosition,
                        pose.rotation * piece.localRotation);
  }
}

// On Play: the links inside each object from its cut, links between pieces
// of objects that touch, and which pieces rest on something that never
// breaks. Everything is still where the layout put it.
void VulkanRenderer::prepareBreakGraph() {
  breakLinks.clear();
  breakLinkPoints.clear();
  breakGraphReady = false;
  if (breakables.empty() || !physics)
    return;
  const auto start = std::chrono::steady_clock::now();
  for (const Breakable &object : breakables) {
    const std::vector<BreakLink> cut = cutLinks(*object.fracture, object.strength, object.firstPiece);
    for (size_t i = 0; i < cut.size(); ++i) {
      breakLinks.push_back(cut[i]);
      breakLinkPoints.push_back(object.fracture->links[i].centre);
    }
  }
  std::vector<PlacedPiece> placed;
  std::vector<glm::mat4> worlds;
  for (const BreakPiece &piece : breakPieces) {
    worlds.push_back(poseMatrix(pieceWorld(piece)));
    placed.push_back(placePiece(breakables[piece.object].fracture->pieces[piece.index],
                                worlds.back(), piece.object));
  }
  const size_t inside = breakLinks.size();
  for (const PieceTouch &touch : touchingPieces(placed, kTouchTolerance)) {
    // Only things that don't move hold onto each other: a building. Crates
    // stacked on each other stay crates.
    if (breakables[breakPieces[touch.a].object].moving ||
        breakables[breakPieces[touch.b].object].moving)
      continue;
    const float strength = touch.area * std::min(breakables[breakPieces[touch.a].object].strength,
                                                 breakables[breakPieces[touch.b].object].strength);
    breakLinks.push_back({touch.a, touch.b, strength});
    breakLinkPoints.push_back(glm::vec3(glm::inverse(worlds[touch.a]) * glm::vec4(touch.centre, 1.0f)));
  }
  pieceLinks.assign(breakPieces.size(), {});
  for (int i = 0; i < static_cast<int>(breakLinks.size()); ++i) {
    pieceLinks[breakLinks[i].a].push_back(i);
    pieceLinks[breakLinks[i].b].push_back(i);
  }
  // Anchors: a piece of a static object with an outside face against
  // something that isn't breakable and doesn't move (the ground, a rock, a
  // cliff for a bridge). Short rays out of the face, from just outside it.
  int anchors = 0;
  for (size_t p = 0; p < breakPieces.size(); ++p) {
    BreakPiece &piece = breakPieces[p];
    piece.anchor = false;
    if (breakables[piece.object].moving)
      continue;
    for (const FractureFace &face : placed[p].faces) {
      if (face.neighbour >= 0 || piece.anchor)
        continue;
      glm::vec3 middle(0.0f);
      for (const glm::vec3 &q : face.points)
        middle += q;
      middle /= static_cast<float>(face.points.size());
      std::vector<glm::vec3> from = {middle};
      for (const glm::vec3 &q : face.points)
        from.push_back(glm::mix(middle, q, 0.6f));
      for (const glm::vec3 &origin : from) {
        const std::optional<PhysicsRayHit> hit =
            physics->castRay(origin + face.normal * 0.002f, face.normal, kAnchorReach);
        if (hit && !(hit->userData & kBreakBodyFlag) && physics->mass(hit->body) == 0.0f) {
          piece.anchor = true;
          ++anchors;
          break;
        }
      }
    }
  }
  breakGraphReady = true;
  spdlog::info("[break] {} objects, {} pieces, {} links ({} between objects), {} anchors, {:.2f} ms",
               breakables.size(), breakPieces.size(), breakLinks.size(),
               breakLinks.size() - inside, anchors,
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                   .count());
}

void VulkanRenderer::handleImpacts() {
  const std::vector<PhysicsImpact> impacts = physics->takeImpacts();
  if (!breakGraphReady || impacts.empty())
    return;
  std::vector<BreakHit> hits;
  for (const PhysicsImpact &impact : impacts) {
    if (!(impact.userData & kBreakBodyFlag))
      continue;
    const uint64_t bodyIndex = impact.userData & ~kBreakBodyFlag;
    if (bodyIndex >= breakBodies.size() || !breakBodies[bodyIndex].id.isValid() ||
        breakBodies[bodyIndex].id.value != impact.body.value)
      continue;
    // The links this body's pieces have, where they are now.
    std::vector<LinkAt> candidates;
    std::unordered_set<int> seen;
    for (int p : breakBodies[bodyIndex].pieces)
      for (int link : pieceLinks[p]) {
        if (breakLinks[link].broken || !seen.insert(link).second)
          continue;
        const PhysicsPose a = pieceWorld(breakPieces[breakLinks[link].a]);
        candidates.push_back({link, a.position + a.rotation * breakLinkPoints[link]});
      }
    const std::vector<int> broken = linksBrokenBy(breakLinks, candidates, impact.point, impact.impulse);
    for (int link : broken)
      breakLinks[link].broken = true;
    if (!broken.empty())
      hits.push_back({impact.point, -impact.normal, impact.impulse});
  }
  if (!hits.empty())
    regroupBreakables(hits);
}

// The graph split into groups again. A body whose pieces are still exactly
// one group, staying put or falling as before, is kept; the rest are
// replaced: anchored groups by static bodies (pieces in kept static bodies
// stay there), loose groups by one moving body each.
void VulkanRenderer::regroupBreakables(const std::vector<BreakHit> &hits) {
  const auto start = std::chrono::steady_clock::now();
  std::vector<bool> anchors(breakPieces.size());
  for (size_t p = 0; p < breakPieces.size(); ++p)
    anchors[p] = breakPieces[p].anchor;
  const std::vector<BreakGroup> groups =
      splitGroups(static_cast<int>(breakPieces.size()), breakLinks, anchors);
  std::vector<int> groupOf(breakPieces.size());
  for (int g = 0; g < static_cast<int>(groups.size()); ++g)
    for (int p : groups[g].pieces)
      groupOf[p] = g;

  std::vector<int> dirty;
  for (int b = 0; b < static_cast<int>(breakBodies.size()); ++b) {
    const BreakBody &body = breakBodies[b];
    if (!body.id.isValid())
      continue;
    const int g = groupOf[body.pieces.front()];
    const bool one = std::all_of(body.pieces.begin(), body.pieces.end(),
                                 [&](int p) { return groupOf[p] == g; });
    const bool keep = one && (groups[g].anchored ? body.fixed
                                                 : !body.fixed && groups[g].pieces.size() == body.pieces.size());
    if (!keep)
      dirty.push_back(b);
  }
  if (dirty.empty())
    return;

  // Where every piece of a replaced body is now, and how its body moved.
  std::vector<PhysicsPose> world(breakPieces.size());
  std::vector<glm::vec3> velocity(breakPieces.size(), glm::vec3(0.0f));
  std::vector<glm::vec3> spin(breakPieces.size(), glm::vec3(0.0f));
  std::vector<bool> moving(breakPieces.size(), false);
  for (int b : dirty) {
    const BreakBody &body = breakBodies[b];
    const glm::vec3 v = physics->linearVelocity(body.id), w = physics->angularVelocity(body.id);
    for (int p : body.pieces) {
      world[p] = pieceWorld(breakPieces[p]);
      velocity[p] = v;
      spin[p] = w;
      moving[p] = true;
    }
  }
  std::unordered_set<int> affected;
  for (int b : dirty) {
    for (int p : breakBodies[b].pieces)
      affected.insert(groupOf[p]);
    physics->removeBody(breakBodies[b].id);
    breakBodies[b].id = {};
    breakBodies[b].pieces.clear();
  }
  int made = 0;
  for (int g : affected) {
    std::vector<int> pieces;
    std::vector<PhysicsPose> worlds;
    for (int p : groups[g].pieces)
      if (moving[p]) {
        pieces.push_back(p);
        worlds.push_back(world[p]);
      }
    if (pieces.empty())
      continue;
    addBreakBody(pieces, worlds, groups[g].anchored, velocity[pieces.front()], spin[pieces.front()],
                 hits);
    ++made;
  }
  spdlog::info("[break] {} bodies replaced by {}, {:.2f} ms", dirty.size(), made,
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                   .count());
}

// Edit mode: a breakable moved in the Inspector or with the gizmo. A new
// size or scale means a new cut, made at the top of the next frame.
void VulkanRenderer::placeBreakable(SceneEntryRef ref) {
  const SceneModelEntry &entry = currentScene.models[ref.index];
  for (const Breakable &object : breakables) {
    if (object.entry != ref)
      continue;
    const glm::vec3 size = entry.size * entry.transform.scale;
    if (!object.key.starts_with(std::format("{} {} {}|", size.x, size.y, size.z))) {
      breakablesDirty = true;
      return;
    }
    const PhysicsPose pose = entryPose(entry);
    for (int k = 0; k < static_cast<int>(object.fracture->pieces.size()); ++k) {
      const BreakPiece &piece = breakPieces[object.firstPiece + k];
      piece.node->setPose(pose.position, pose.rotation);
      if (piece.body >= 0 && breakBodies[piece.body].id.isValid())
        physics->setPose(breakBodies[piece.body].id, pose);
    }
  }
}

// Show pieces: the selected breakable's cuts, as lines.
void VulkanRenderer::appendPieceLines(SceneEntryRef ref) {
  // The UI's warm yellow (kGizmoHot, 0xf5d36b), as for the skeleton.
  constexpr uint32_t kCutColor = 0xf5u | 0xd3u << 8 | 0x6bu << 16 | 0xffu << 24;
  for (const Breakable &object : breakables) {
    if (object.entry != ref)
      continue;
    for (int k = 0; k < static_cast<int>(object.fracture->pieces.size()); ++k) {
      const glm::mat4 where = poseMatrix(pieceWorld(breakPieces[object.firstPiece + k]));
      for (const FractureFace &face : object.fracture->pieces[k].faces)
        for (size_t i = 0; i < face.points.size(); ++i) {
          editorLines.push_back({glm::vec3(where * glm::vec4(face.points[i], 1.0f)), kCutColor});
          editorLines.push_back(
              {glm::vec3(where * glm::vec4(face.points[(i + 1) % face.points.size()], 1.0f)),
               kCutColor});
        }
    }
  }
}
