// Destruction in the renderer (steps 1.3 and 1.3b): breakable boxes as
// pieces and the bodies that carry them, pieces cut again where they're
// hit, and rubble. The cutting and the graph are in heliora_core
// (Fracture.h, BreakGraph.h); this is where they meet the scene, the physics
// world and the nodes that draw the pieces.
#include "VulkanRenderer.h"

#include "ScenePhysics.h"

#include <glm/gtc/matrix_transform.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <format>
#include <random>
#include <unordered_set>

namespace {

// Faces this close count as touching (objects linked across) or resting on
// something (an anchor).
constexpr float kTouchTolerance = 0.01f; // m
constexpr float kAnchorReach = 0.05f;    // m
// How much of a hit carries on into what it frees (they share it); what hit
// keeps the rest and carries on through.
constexpr float kHitCarry = 0.5f;
constexpr float kShareReach = 2.0f; // m: freed groups further than this get none
// Cutting where hit: no piece smaller than this (smaller ones are rubble
// straight away), and pieces smaller than kPickUp settle into rubble.
constexpr float kSmallestPiece = 0.15f; // m, the cube root of its volume
constexpr float kPickUp = 0.2f;         // m
constexpr float kSettleSeconds = 4.0f;  // a small piece still moving after this is rubble too
constexpr float kRubblePatch = 0.5f;    // m
constexpr float kLargestHeap = 1.6f;    // m of radius
// The budget: cuts a frame (the rest wait), and pieces cut in a whole Play.
constexpr int kSplitsPerFrame = 4;
constexpr int kRuntimePieceBudget = 2000;
constexpr float kSplinterStretch = 4.0f;
constexpr float kFastestSpin = 20.0f; // rad/s
// No hit softer than this cuts a piece: debris knocking about cracks links
// but doesn't set off cut after cut. A heavy ball (about 1,500 N s) and big
// chunks landing do.
constexpr float kMinSplitImpulse = 200.0f; // N s

// While a frame's cuts run, new models' geometry waits here and goes into
// the GPU-driven pass's shared buffer with one upload at the end.
std::vector<int> *gDeferredGeometry = nullptr;

// Cut faces: the object's colour, darker, like the raw inside of things.
glm::vec3 rawColour(glm::vec3 c) { return c * 0.55f; }

glm::mat4 poseMatrix(const PhysicsPose &pose) {
  return glm::translate(glm::mat4(1.0f), pose.position) * glm::mat4_cast(pose.rotation);
}

PhysicsPose entryPose(const SceneModelEntry &entry) {
  return {entry.transform.position, rotationFromDegrees(entry.transform.rotationDeg)};
}

float pieceSize(const FracturePiece &shape) { return std::cbrt(std::max(shape.volume, 0.0f)); }

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
  if (physics) {
    for (const BreakBody &body : breakBodies)
      if (body.id.isValid())
        physics->removeBody(body.id);
    for (const auto &[patch, heap] : rubble)
      if (heap.body.isValid())
        physics->removeBody(heap.body);
  }
  std::unordered_set<const SceneNode *> nodes;
  for (const BreakPiece &piece : breakPieces)
    nodes.insert(piece.node);
  for (const auto &[patch, heap] : rubble)
    nodes.insert(heap.node);
  if (!nodes.empty())
    rootNode.removeChildren([&](const SceneNode &node) { return nodes.contains(&node); });
  breakables.clear();
  breakPieces.clear();
  breakBodies.clear();
  breakLinks.clear();
  breakLinkPoints.clear();
  pieceLinks.clear();
  rubble.clear();
  pendingSplits.clear();
  uploadsInFlight.clear(); // each waits for its fence first
  runtimePieces = 0;
  breakGraphReady = false;
  breakablesDirty = false;
}

// A piece's own model: kept in the model cache under key, so a rebuild
// (Stop, an edit) makes nothing new for the first cut's pieces.
int VulkanRenderer::pieceModel(const Breakable &object, const FracturePiece &shape,
                               const std::string &key) {
  int modelId = modelManager.findModel(key);
  if (modelId < 0) {
    modelId = modelManager.createModel(key,
                                       makePieceMesh(shape, object.colour, rawColour(object.colour)),
                                       false, false, device, textureManager, descriptorManager);
    if (gDeferredGeometry)
      gDeferredGeometry->push_back(modelId);
    else
      gpuDrivenGBufferPass.registerModelGeometry(modelId, modelManager);
  }
  return modelId;
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
    // Its material says how it breaks; break_strength in the file beats it.
    const std::optional<PhysicsMaterial> material =
        entry.material ? findMaterial(*entry.material) : std::nullopt;
    object.strength = entry.breakStrength.value_or(material ? material->breakStrength
                                                            : kDefaultBreakStrength);
    object.pattern = material ? material->pattern : FracturePattern::Clustered;
    object.moving = *entry.body != PhysicsMotion::Static;
    object.volume = size.x * size.y * size.z;
    const PhysicsBodyDesc desc = bodyDescFor(entry, nullptr, physicsPanel.defaultFriction);
    object.density = desc.mass > 0.0f ? desc.mass / object.volume : desc.density;
    object.grain = size.x >= size.y && size.x >= size.z ? glm::vec3(1, 0, 0)
                   : size.y >= size.z                   ? glm::vec3(0, 1, 0)
                                                        : glm::vec3(0, 0, 1);
    object.colour = entry.color;
    const int objectIndex = static_cast<int>(breakables.size());
    breakables.push_back(object);
    std::vector<int> pieces;
    std::vector<PhysicsPose> worlds;
    const PhysicsPose pose = entryPose(entry);
    for (int k = 0; k < static_cast<int>(cut->pieces.size()); ++k) {
      const glm::vec3 c = entry.color;
      const int modelId = pieceModel(breakables.back(), cut->pieces[k],
                                     std::format("#piece|{}|{}|{} {} {}", object.key, k, c.r, c.g, c.b));
      auto node = std::make_unique<SceneNode>();
      node->setModelId(modelId);
      node->setEntry(object.entry);
      node->setPose(pose.position, pose.rotation);
      BreakPiece piece;
      piece.object = objectIndex;
      piece.shape = std::shared_ptr<const FracturePiece>(cut, &cut->pieces[k]);
      piece.node = node.get();
      rootNode.addChild(std::move(node));
      pieces.push_back(static_cast<int>(breakPieces.size()));
      worlds.push_back(pose);
      breakPieces.push_back(piece);
    }
    addBreakBody(pieces, worlds, !object.moving, glm::vec3(0.0f), glm::vec3(0.0f));
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
                                 glm::vec3 velocity, glm::vec3 spin) {
  const PhysicsPose frame = worlds.front();
  const glm::quat toLocal = glm::conjugate(frame.rotation);
  auto parts = std::make_shared<PhysicsMeshParts>();
  std::vector<std::pair<glm::vec3, glm::quat>> locals;
  for (size_t i = 0; i < pieces.size(); ++i) {
    const BreakPiece &piece = breakPieces[pieces[i]];
    const glm::vec3 lp = toLocal * (worlds[i].position - frame.position);
    const glm::quat lr = toLocal * worlds[i].rotation;
    PhysicsMeshPart part;
    for (const FractureFace &face : piece.shape->faces)
      for (const glm::vec3 &q : face.points)
        part.positions.push_back(lp + lr * q);
    parts->parts.push_back(std::move(part));
    locals.push_back({lp, lr});
  }
  const Breakable &object = breakables[breakPieces[pieces.front()].object];
  const SceneModelEntry &entry = currentScene.models[object.entry.index];
  PhysicsBodyDesc desc = bodyDescFor(entry, nullptr, physicsPanel.defaultFriction);
  desc.shape = PhysicsShape::Hulls;
  desc.parts = parts;
  desc.scale = glm::vec3(1.0f);
  desc.motion = fixed ? PhysicsMotion::Static : PhysicsMotion::Dynamic;
  desc.position = frame.position;
  desc.rotation = frame.rotation;
  desc.mass = 0.0f;
  desc.density = object.density; // so a chunk weighs its share
  desc.linearVelocity = velocity;
  // Jolt refuses spins past about 47 rad/s; a tumble that fast reads as a
  // blur anyway.
  if (glm::length(spin) > kFastestSpin)
    spin = glm::normalize(spin) * kFastestSpin;
  desc.angularVelocity = spin;
  desc.reportImpacts = true;
  const int index = static_cast<int>(breakBodies.size());
  desc.userData = kBreakBodyFlag | static_cast<uint64_t>(index);
  BreakBody body;
  body.id = physics->addBody(desc);
  body.pieces = pieces;
  body.fixed = fixed;
  body.madeAt = playSeconds;
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
    if (piece.alive)
      piece.node->setPose(pose.position + pose.rotation * piece.localPosition,
                          pose.rotation * piece.localRotation);
  }
}

// An anchor: an outside face against something that isn't breakable and
// doesn't move (the ground, a rock, a cliff for a bridge). Short rays out of
// the face, from just outside it.
bool VulkanRenderer::restsOnFixed(const PlacedPiece &placed) const {
  for (const FractureFace &face : placed.faces) {
    if (face.neighbour != kOutsideFace)
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
      if (hit && !(hit->userData & kBreakBodyFlag) && physics->mass(hit->body) == 0.0f)
        return true;
    }
  }
  return false;
}

// On Play: the links inside each object from its cut (each Play's seed
// varies their strengths), links between pieces of objects that touch, and
// which pieces rest on something that never breaks.
void VulkanRenderer::prepareBreakGraph() {
  breakLinks.clear();
  breakLinkPoints.clear();
  breakGraphReady = false;
  if (breakables.empty() || !physics)
    return;
  const auto start = std::chrono::steady_clock::now();
  playSeed = std::random_device{}();
  splitsMade = 0;
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
    placed.push_back(placePiece(*piece.shape, worlds.back(), piece.object));
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
  varyStrengths(breakLinks, playSeed, 0.25f);
  pieceLinks.assign(breakPieces.size(), {});
  for (int i = 0; i < static_cast<int>(breakLinks.size()); ++i) {
    pieceLinks[breakLinks[i].a].push_back(i);
    pieceLinks[breakLinks[i].b].push_back(i);
  }
  int anchors = 0;
  for (size_t p = 0; p < breakPieces.size(); ++p) {
    BreakPiece &piece = breakPieces[p];
    piece.anchor = !breakables[piece.object].moving && restsOnFixed(placed[p]);
    anchors += piece.anchor;
  }
  if (!rubbleParts) {
    // The unit mound's points, the hull every heap's body is scaled from.
    rubbleParts = std::make_shared<PhysicsMeshParts>();
    PhysicsMeshPart part;
    for (const Vertex &v : makeRubbleMound(0, glm::vec3(1.0f)).vertices)
      part.positions.push_back(v.pos);
    rubbleParts->parts.push_back(std::move(part));
  }
  breakGraphReady = true;
  spdlog::info("[break] {} objects, {} pieces, {} links ({} between objects), {} anchors, {:.2f} ms",
               breakables.size(), breakPieces.size(), breakLinks.size(), breakLinks.size() - inside,
               anchors,
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                   .count());
}

void VulkanRenderer::handleImpacts() {
  const std::vector<PhysicsImpact> impacts = physics->takeImpacts();
  if (!breakGraphReady)
    return;
  std::vector<BreakHit> hits;
  std::unordered_set<int> queued;
  for (const PendingSplit &split : pendingSplits)
    queued.insert(split.piece);
  for (const PhysicsImpact &impact : impacts) {
    if (!(impact.userData & kBreakBodyFlag))
      continue;
    const uint64_t bodyIndex = impact.userData & ~kBreakBodyFlag;
    if (bodyIndex >= breakBodies.size() || !breakBodies[bodyIndex].id.isValid() ||
        breakBodies[bodyIndex].id.value != impact.body.value)
      continue;
    const glm::vec3 push = -impact.normal;
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
    // Pieces it reaches harder than they can hold are cut where it hit:
    // the nearest few, in their own space, done over the next frames.
    std::vector<std::pair<float, int>> reached;
    for (int p : breakBodies[bodyIndex].pieces) {
      const BreakPiece &piece = breakPieces[p];
      const Breakable &object = breakables[piece.object];
      if (!piece.alive || queued.contains(p) || object.pattern == FracturePattern::None ||
          pieceSize(*piece.shape) < 2.0f * kSmallestPiece || runtimePieces >= kRuntimePieceBudget)
        continue;
      const PhysicsPose world = pieceWorld(piece);
      const float d =
          std::max(0.0f, glm::distance(world.position + world.rotation * piece.shape->centre,
                                       impact.point) -
                             0.5f * pieceSize(*piece.shape)) /
          kBreakSpread;
      const float arriving = impact.impulse / (1.0f + d * d);
      if (arriving > std::max(kMinSplitImpulse, splitImpulse(piece.shape->volume, object.strength)))
        reached.push_back({d, p});
    }
    std::sort(reached.begin(), reached.end());
    for (size_t i = 0; i < reached.size() && i < 2; ++i) {
      const BreakPiece &piece = breakPieces[reached[i].second];
      const PhysicsPose world = pieceWorld(piece);
      const glm::quat toLocal = glm::conjugate(world.rotation);
      pendingSplits.push_back({reached[i].second, toLocal * (impact.point - world.position),
                               toLocal * push, impact.impulse});
      queued.insert(reached[i].second);
    }
    if (!broken.empty() || !reached.empty())
      hits.push_back({impact.point, push, impact.impulse, impact.other, impact.otherVelocity});
  }
  // A few cuts a frame, their new models uploaded together: one wait for
  // the meshes' own buffers and one for the shared buffer.
  int made = 0;
  const auto cutting = std::chrono::steady_clock::now();
  std::erase_if(uploadsInFlight, [](const std::unique_ptr<UploadBatch> &b) { return b->finished(); });
  if (!pendingSplits.empty()) {
    auto batch = std::make_unique<UploadBatch>(
        device.getLogicalDevice(), device.getGraphicsCommandPool(), device.getGraphicsQueue());
    std::vector<int> geometry;
    gUploadBatch = batch.get();
    gDeferredGeometry = &geometry;
    while (!pendingSplits.empty() && made < kSplitsPerFrame) {
      const PendingSplit split = pendingSplits.front();
      pendingSplits.erase(pendingSplits.begin());
      if (breakPieces[split.piece].alive && runtimePieces < kRuntimePieceBudget) {
        splitPiece(split);
        ++made;
      }
    }
    gDeferredGeometry = nullptr;
    gpuDrivenGBufferPass.registerModelGeometries(geometry, modelManager); // into the batch
    gUploadBatch = nullptr;
    batch->submit();
    uploadsInFlight.push_back(std::move(batch));
  }
  if (made > 0)
    spdlog::info("[break] {} cuts, {:.2f} ms ({} pieces cut so far, {} rubble heaps)", made,
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cutting)
                     .count(),
                 runtimePieces, rubble.size());
  if (!hits.empty() || made > 0)
    regroupBreakables(hits);
  settleRubble();
}

// The piece cut into pieces around where it was hit, in its material's
// pattern. The new pieces link to each other and, where they touch them, to
// the old piece's neighbours; the smallest go straight to rubble.
void VulkanRenderer::splitPiece(const PendingSplit &split) {
  const BreakPiece old = breakPieces[split.piece]; // a copy: the vector grows below
  const Breakable &object = breakables[old.object];
  const float holds = splitImpulse(old.shape->volume, object.strength);
  const int count =
      std::clamp(static_cast<int>(4.0f * std::sqrt(split.impulse / std::max(holds, 1e-3f))), 4, 16);
  const uint32_t seed = playSeed ^ (++splitsMade * 0x9e3779b9u) ^ static_cast<uint32_t>(split.piece);
  const std::vector<glm::vec3> points =
      pointsAround(*old.shape, split.hit, split.push, count, object.pattern, seed);
  if (points.size() < 2)
    return;
  const bool splinters = object.pattern == FracturePattern::Splinters;
  const Fracture cut = fractureConvex(*old.shape, points, splinters ? object.grain : glm::vec3(0.0f),
                                      splinters ? kSplinterStretch : 1.0f);
  const PhysicsPose world = pieceWorld(old);
  const glm::mat4 where = poseMatrix(world);

  // The new pieces: big enough to stay pieces, or rubble now.
  std::vector<int> made(cut.pieces.size(), -1);
  for (size_t k = 0; k < cut.pieces.size(); ++k) {
    const FracturePiece &shape = cut.pieces[k];
    if (pieceSize(shape) < kSmallestPiece) {
      addRubble(glm::vec3(where * glm::vec4(shape.centre, 1.0f)), shape.volume,
                rawColour(object.colour));
      continue;
    }
    BreakPiece piece;
    piece.object = old.object;
    piece.shape = std::make_shared<const FracturePiece>(shape);
    auto node = std::make_unique<SceneNode>();
    node->setModelId(pieceModel(object, shape, std::format("#piece|cut|{}", seed + k * 7919u)));
    node->setEntry(object.entry);
    node->setPose(world.position, world.rotation);
    piece.node = node.get();
    rootNode.addChild(std::move(node));
    // In the old piece's body for now, where the old piece was; the regroup
    // that follows gives them bodies of their own.
    piece.body = old.body;
    piece.localPosition = old.localPosition;
    piece.localRotation = old.localRotation;
    piece.anchor = old.anchor && restsOnFixed(placePiece(shape, where, 0));
    made[k] = static_cast<int>(breakPieces.size());
    breakPieces.push_back(std::move(piece));
    pieceLinks.emplace_back();
    if (old.body >= 0)
      breakBodies[old.body].pieces.push_back(made[k]);
    ++runtimePieces;
  }
  breakPieces[split.piece].alive = false;
  SceneNode *goneNode = old.node;
  rootNode.removeChildren([&](const SceneNode &node) { return &node == goneNode; });

  auto addLink = [&](int a, int b, float strength, glm::vec3 pointInA) {
    const int index = static_cast<int>(breakLinks.size());
    // The point is kept in the frame of the link's first (lower) piece.
    glm::vec3 point = pointInA;
    if (a > b) {
      const PhysicsPose pa = pieceWorld(breakPieces[a]), pb = pieceWorld(breakPieces[b]);
      point = glm::conjugate(pb.rotation) * (pa.position + pa.rotation * pointInA - pb.position);
    }
    breakLinks.push_back({std::min(a, b), std::max(a, b), strength});
    breakLinkPoints.push_back(point);
    pieceLinks[a].push_back(index);
    pieceLinks[b].push_back(index);
  };
  // Between the new pieces, through their new faces.
  std::vector<BreakLink> inner = cutLinks(cut, object.strength, 0);
  varyStrengths(inner, seed, 0.25f);
  for (size_t i = 0; i < cut.links.size(); ++i)
    if (made[cut.links[i].a] >= 0 && made[cut.links[i].b] >= 0)
      addLink(made[cut.links[i].a], made[cut.links[i].b], inner[i].strength, cut.links[i].centre);
  // To the old piece's neighbours, where they touch.
  const std::vector<int> oldLinks = pieceLinks[split.piece];
  for (int link : oldLinks) {
    if (breakLinks[link].broken)
      continue;
    breakLinks[link].broken = true;
    const int other = breakLinks[link].a == split.piece ? breakLinks[link].b : breakLinks[link].a;
    if (!breakPieces[other].alive)
      continue;
    const PlacedPiece placedOther =
        placePiece(*breakPieces[other].shape, poseMatrix(pieceWorld(breakPieces[other])), 1);
    const float strength = std::min(object.strength, breakables[breakPieces[other].object].strength);
    for (size_t k = 0; k < made.size(); ++k) {
      if (made[k] < 0)
        continue;
      glm::vec3 centre;
      const float area =
          touchArea(placePiece(cut.pieces[k], where, 0), placedOther, kTouchTolerance, centre);
      if (area > 1e-5f)
        addLink(made[k], other, area * strength,
                glm::vec3(glm::inverse(where) * glm::vec4(centre, 1.0f)));
    }
  }
}

// The graph split into groups again. A body whose pieces are still exactly
// one group, staying put or moving as before, is kept; the rest are
// replaced: groups that are still the structure by static bodies (pieces in
// kept static bodies stay there), the rest by one moving body each. This
// frame's hits are shared between the moving groups they freed, and what
// hit carries on with what's left.
void VulkanRenderer::regroupBreakables(const std::vector<BreakHit> &hits) {
  const auto start = std::chrono::steady_clock::now();
  const int count = static_cast<int>(breakPieces.size());
  std::vector<bool> anchors(count);
  std::vector<int> objectOf(count);
  std::vector<float> volume(count);
  for (int p = 0; p < count; ++p) {
    anchors[p] = breakPieces[p].anchor && breakPieces[p].alive;
    objectOf[p] = breakPieces[p].object;
    volume[p] = breakPieces[p].alive ? breakPieces[p].shape->volume : 0.0f;
  }
  std::vector<float> objectVolume;
  for (const Breakable &object : breakables)
    objectVolume.push_back(object.volume);
  const std::vector<BreakGroup> groups = splitGroups(count, breakLinks, anchors);
  std::vector<int> groupOf(count);
  std::vector<bool> structural(groups.size());
  for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
    for (int p : groups[g].pieces)
      groupOf[p] = g;
    structural[g] = staysPut(groups[g], objectOf, volume, objectVolume);
  }

  std::vector<int> dirty;
  for (int b = 0; b < static_cast<int>(breakBodies.size()); ++b) {
    const BreakBody &body = breakBodies[b];
    if (!body.id.isValid())
      continue;
    const int g = groupOf[body.pieces.front()];
    const bool one = std::all_of(body.pieces.begin(), body.pieces.end(), [&](int p) {
      return breakPieces[p].alive && groupOf[p] == g;
    });
    int aliveInGroup = 0;
    for (int p : groups[g].pieces)
      aliveInGroup += breakPieces[p].alive;
    const bool keep = one && (structural[g] ? body.fixed
                                            : !body.fixed &&
                                                  aliveInGroup == static_cast<int>(body.pieces.size()));
    if (!keep)
      dirty.push_back(b);
  }
  if (dirty.empty())
    return;

  // Where every piece of a replaced body is now, and how its body moved.
  std::vector<PhysicsPose> world(count);
  std::vector<glm::vec3> velocity(count, glm::vec3(0.0f));
  std::vector<glm::vec3> spin(count, glm::vec3(0.0f));
  std::vector<bool> loose(count, false);
  for (int b : dirty) {
    const BreakBody &body = breakBodies[b];
    const glm::vec3 v = physics->linearVelocity(body.id), w = physics->angularVelocity(body.id);
    for (int p : body.pieces) {
      if (!breakPieces[p].alive)
        continue;
      world[p] = pieceWorld(breakPieces[p]);
      velocity[p] = v;
      spin[p] = w;
      loose[p] = true;
    }
  }
  std::unordered_set<int> affected;
  for (int b : dirty) {
    for (int p : breakBodies[b].pieces)
      if (breakPieces[p].alive)
        affected.insert(groupOf[p]);
    physics->removeBody(breakBodies[b].id);
    breakBodies[b].id = {};
    breakBodies[b].pieces.clear();
  }
  // The new bodies, and each moving one's middle and mass, for sharing hits.
  struct NewBody {
    std::vector<int> pieces;
    std::vector<PhysicsPose> worlds;
    bool fixed = false;
    glm::vec3 velocity{0.0f}, spin{0.0f}, middle{0.0f};
    float mass = 0.0f;
  };
  std::vector<NewBody> made;
  for (int g : affected) {
    NewBody nb;
    float volumeSum = 0.0f;
    for (int p : groups[g].pieces)
      if (loose[p]) {
        nb.pieces.push_back(p);
        nb.worlds.push_back(world[p]);
        const float v = breakPieces[p].shape->volume;
        nb.middle += (world[p].position + world[p].rotation * breakPieces[p].shape->centre) * v;
        volumeSum += v;
        nb.mass += v * breakables[breakPieces[p].object].density;
      }
    if (nb.pieces.empty())
      continue;
    nb.middle /= std::max(volumeSum, 1e-9f);
    nb.fixed = structural[g];
    nb.velocity = velocity[nb.pieces.front()];
    nb.spin = spin[nb.pieces.front()];
    made.push_back(std::move(nb));
  }
  for (const BreakHit &hit : hits) {
    std::vector<int> near;
    std::vector<float> distances;
    for (int i = 0; i < static_cast<int>(made.size()); ++i)
      if (!made[i].fixed && glm::distance(made[i].middle, hit.point) < kShareReach) {
        near.push_back(i);
        distances.push_back(glm::distance(made[i].middle, hit.point));
      }
    const float closing = std::max(0.0f, glm::dot(hit.otherVelocity, hit.push));
    const std::vector<float> shares = shareHit(kHitCarry * hit.impulse, distances);
    float given = 0.0f;
    for (size_t k = 0; k < near.size(); ++k) {
      NewBody &nb = made[near[k]];
      // Never faster than half what hit it was coming: fragments trail the
      // ball rather than outrun it.
      const float speed = std::min(shares[k] / std::max(nb.mass, 1e-3f), 0.5f * closing);
      nb.velocity += hit.push * speed;
      nb.spin += glm::cross(nb.middle - hit.point, hit.push * speed) * 1.5f;
      given += speed * nb.mass;
    }
    // What hit carries on through with what it didn't give, instead of
    // bouncing off as if the object were whole. Breakable bodies that hit
    // keep what the solver gave them.
    const float hitterMass = hit.other.isValid() ? physics->mass(hit.other) : 0.0f;
    bool hitterBreaks = false;
    for (const BreakBody &body : breakBodies)
      hitterBreaks = hitterBreaks || (body.id.isValid() && body.id.value == hit.other.value);
    if (given > 0.0f && hitterMass > 0.0f && !hitterBreaks) {
      const glm::vec3 along = hit.push * std::max(0.0f, closing - given / hitterMass);
      const glm::vec3 sideways = hit.otherVelocity - hit.push * glm::dot(hit.otherVelocity, hit.push);
      physics->setLinearVelocity(hit.other, sideways + along);
    }
  }
  for (const NewBody &nb : made)
    addBreakBody(nb.pieces, nb.worlds, nb.fixed, nb.velocity, nb.spin);
  spdlog::info("[break] {} bodies replaced by {}, {:.2f} ms", dirty.size(), made.size(),
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                   .count());
}

// Small pieces that have settled (or are still rolling a few seconds on)
// on something that never breaks turn into rubble there.
void VulkanRenderer::settleRubble() {
  for (int b = 0; b < static_cast<int>(breakBodies.size()); ++b) {
    BreakBody &body = breakBodies[b];
    if (!body.id.isValid() || body.fixed || body.pieces.size() != 1)
      continue;
    const int index = body.pieces.front();
    BreakPiece &piece = breakPieces[index];
    if (!piece.alive || pieceSize(*piece.shape) >= kPickUp)
      continue;
    const bool settled = !physics->isActive(body.id) ||
                         (playSeconds - body.madeAt > kSettleSeconds &&
                          glm::length(physics->linearVelocity(body.id)) < 0.5f);
    if (!settled)
      continue;
    const PhysicsPose world = pieceWorld(piece);
    const glm::vec3 middle = world.position + world.rotation * piece.shape->centre;
    // Under it: the ground or something else that doesn't break or move.
    // Its own body is in the way of a ray from its middle, so it's ignored
    // by starting just under its lowest point.
    float lowest = middle.y;
    for (const FractureFace &face : piece.shape->faces)
      for (const glm::vec3 &q : face.points)
        lowest = std::min(lowest, (world.position + world.rotation * q).y);
    const std::optional<PhysicsRayHit> below = physics->castRay(
        glm::vec3(middle.x, lowest - 0.005f, middle.z), glm::vec3(0.0f, -1.0f, 0.0f), 0.5f);
    if (!below || (below->userData & kBreakBodyFlag) || physics->mass(below->body) > 0.0f)
      continue; // on another piece, or a moving body: not yet
    physics->removeBody(body.id);
    body.id = {};
    body.pieces.clear();
    piece.alive = false;
    SceneNode *node = piece.node;
    rootNode.removeChildren([&](const SceneNode &n) { return &n == node; });
    for (int link : pieceLinks[index])
      breakLinks[link].broken = true;
    addRubble(middle, piece.shape->volume, rawColour(breakables[piece.object].colour));
  }
}

// Rubble joins the heap on the half-metre patch of ground under where.
void VulkanRenderer::addRubble(glm::vec3 where, float volume, glm::vec3 colour) {
  if (!rubbleParts || volume <= 0.0f)
    return;
  std::optional<PhysicsRayHit> ground;
  // The first thing below that isn't a breakable piece (pieces may be in
  // the way while a wall comes down).
  glm::vec3 from = where + glm::vec3(0.0f, 0.05f, 0.0f);
  for (int tries = 0; tries < 8; ++tries) {
    ground = physics->castRay(from, glm::vec3(0.0f, -1.0f, 0.0f), 40.0f);
    if (!ground || !(ground->userData & kBreakBodyFlag))
      break;
    from.y -= ground->distance + 0.02f;
    ground.reset();
  }
  if (!ground)
    return; // over nothing: dust
  const glm::vec3 at(from.x, from.y - ground->distance, from.z);
  const std::pair<int, int> patch{static_cast<int>(std::floor(at.x / kRubblePatch)),
                                  static_cast<int>(std::floor(at.z / kRubblePatch))};
  RubbleHeap &heap = rubble[patch];
  if (heap.volume == 0.0f) {
    heap.base = glm::vec3((patch.first + 0.5f) * kRubblePatch, at.y,
                          (patch.second + 0.5f) * kRubblePatch);
    heap.seed = static_cast<uint32_t>(patch.first * 73856093) ^
                static_cast<uint32_t>(patch.second * 19349663);
  }
  heap.colour = (heap.colour * heap.volume + colour * volume) / (heap.volume + volume);
  heap.volume += volume;
  const float radius = std::min(rubbleRadius(heap.volume), kLargestHeap);
  // Its look: a few rough mounds and colours, shared between heaps.
  const glm::vec3 c = glm::round(heap.colour * 16.0f) / 16.0f;
  const uint32_t variant = heap.seed % 4;
  const std::string key = std::format("#rubble|{}|{} {} {}", variant, c.r, c.g, c.b);
  int modelId = modelManager.findModel(key);
  if (modelId < 0) {
    modelId = modelManager.createModel(key, makeRubbleMound(variant, c), false, false, device,
                                       textureManager, descriptorManager);
    if (gDeferredGeometry)
      gDeferredGeometry->push_back(modelId);
    else
      gpuDrivenGBufferPass.registerModelGeometry(modelId, modelManager);
  }
  if (!heap.node) {
    auto node = std::make_unique<SceneNode>();
    heap.node = node.get();
    rootNode.addChild(std::move(node));
  }
  heap.node->setModelId(modelId);
  NodeTransform t;
  t.position = heap.base;
  t.rotationDeg.y = static_cast<float>(heap.seed % 360);
  t.scale = radius;
  heap.node->setTransform(t);
  // Its body, made again once it has grown by a fifth.
  if (!heap.body.isValid() || radius > heap.bodyRadius * 1.2f) {
    if (heap.body.isValid())
      physics->removeBody(heap.body);
    PhysicsBodyDesc desc;
    desc.shape = PhysicsShape::Hulls;
    desc.parts = rubbleParts;
    desc.scale = glm::vec3(radius);
    desc.motion = PhysicsMotion::Static;
    desc.position = heap.base;
    desc.rotation = rotationFromDegrees(t.rotationDeg);
    desc.friction = 0.8f;
    heap.body = physics->addBody(desc);
    heap.bodyRadius = radius;
  }
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
  for (const BreakPiece &piece : breakPieces) {
    if (!piece.alive || breakables[piece.object].entry != ref)
      continue;
    const glm::mat4 where = poseMatrix(pieceWorld(piece));
    for (const FractureFace &face : piece.shape->faces)
      for (size_t i = 0; i < face.points.size(); ++i) {
        editorLines.push_back({glm::vec3(where * glm::vec4(face.points[i], 1.0f)), kCutColor});
        editorLines.push_back(
            {glm::vec3(where * glm::vec4(face.points[(i + 1) % face.points.size()], 1.0f)),
             kCutColor});
      }
  }
}
