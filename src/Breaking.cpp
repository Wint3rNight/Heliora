// Destruction in the renderer (steps 1.3, 1.3b and 1.3c): breakable boxes
// as pieces and the bodies that carry them, pieces cut again where they're
// hit, and rubble as a layer over the ground. The cutting and the graph are in heliora_core
// (Fracture.h, BreakGraph.h); this is where they meet the scene, the physics
// world and the nodes that draw the pieces.
#include "VulkanRenderer.h"

#include "ScenePhysics.h"

#include <glm/gtc/matrix_transform.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <numeric>
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
// Rubble (Rubble.h): broken stuff takes about half again its solid volume;
// every cut sheds a share of the piece cut as fines (dust and gravel our
// pieces don't have) on the ground under the hit; a settled piece's volume
// goes around it. All of it fed in over kFeedSeconds.
constexpr float kBulking = 1.5f;
constexpr float kFines = 0.15f;
constexpr float kFinesReach = 0.5f;   // m
constexpr float kDepositReach = 0.25f; // m, at least
constexpr float kFeedSeconds = 0.5f;
constexpr float kCastAbove = 2.0f;    // m: ground rays start this far over the rubble
constexpr int kSlumpPasses = 8;       // a frame
constexpr int kTilesPerFrame = 6;     // re-meshed at a time, the rest wait
constexpr float kRemeshSeconds = 0.05f; // tiles re-meshed at most 20 times a second (each time a submit)
constexpr float kRemeshMove = 0.002f; // m: a tile changing less isn't re-meshed yet
// Weight (Stress.h): iterations of the solve a frame until it settles, and
// links it breaks a frame, the worst first, so a collapse runs instead of
// happening all at once.
constexpr int kStressIterations = 40;
// A solve that hasn't settled after this many is judged as it is: a piece
// tipping off its support is a nearly free mode the solver crawls along.
constexpr int kStressMostIterations = 2000;
constexpr int kStressBreaksPerFrame = 3;
// Live loads: smoothed this much a frame, handed to the solve once anything
// moved more than kLiveLoadStep, and a link breaks only after being over its
// limits this long, so one noisy frame doesn't break it.
constexpr float kLiveSmoothing = 0.3f;
constexpr float kLiveLoadStep = 50.0f; // N, or N m
constexpr float kOverloadSeconds = 0.1f;
constexpr float kSupportFacing = 0.3f; // a contact facing at least this much up holds weight
// Room kept in the GPU-driven pass's shared buffers at Play for what's made
// while playing (runtime pieces and rubble tiles), so it never grows, and
// waits for the GPU, mid-game.
constexpr size_t kReservedVertices = 256 * 1024;
constexpr size_t kReservedIndices = 768 * 1024;

// True if any height moved more than kRemeshMove, or rubble came or went.
bool movedMuch(const std::vector<float> &before, const std::vector<float> &after) {
  if (before.size() != after.size())
    return true;
  for (size_t k = 0; k < after.size(); ++k) {
    if (std::isnan(before[k]) != std::isnan(after[k]))
      return true;
    if (!std::isnan(after[k]) && std::fabs(after[k] - before[k]) > kRemeshMove)
      return true;
  }
  return false;
}
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
  clearEffects(); // Play, Stop and a new scene all come through here
  if (physics) {
    for (const BreakBody &body : breakBodies)
      if (body.id.isValid())
        physics->removeBody(body.id);
    for (const auto &[key, tile] : rubbleTiles)
      if (tile.body.isValid())
        physics->removeBody(tile.body);
  }
  std::unordered_set<const SceneNode *> nodes;
  for (const BreakPiece &piece : breakPieces)
    nodes.insert(piece.node); // the stones among them
  for (const auto &[key, tile] : rubbleTiles)
    nodes.insert(tile.node);
  if (!nodes.empty())
    rootNode.removeChildren([&](const SceneNode &node) { return nodes.contains(&node); });
  breakables.clear();
  breakPieces.clear();
  breakBodies.clear();
  breakLinks.clear();
  breakLinkPoints.clear();
  breakLinkFaces.clear();
  breakLinkContact.clear();
  groundLinks.clear();
  pieceGroundLinks.clear();
  pieceShift.clear();
  pieceTurn.clear();
  solvedLinks.clear();
  solvedLoad.clear();
  loadLines.clear();
  restsOn.clear();
  liveForce.clear();
  liveTorque.clear();
  liveAppliedForce.clear();
  liveAppliedTorque.clear();
  overloadSeconds.clear();
  stressDirty = false;
  pieceLinks.clear();
  rubbleTiles.clear();
  rubbleWaiting.clear();
  rubbleFeed.clear();
  rubbleStones.clear();
  if (rubbleLayer)
    rubbleLayer->clear();
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
    // Weight: its material's strengths (stone's if it has none), scaled by
    // its break_strength against that material's default.
    const PhysicsMaterial base = material ? *material : *findMaterial("stone");
    const float scale = object.strength / base.breakStrength;
    object.limits = {base.pullStrength * scale, base.pushStrength * scale,
                     base.slideStrength * scale, false, desc.friction};
    object.glued = entry.glued;
    object.surface = surfaceOf(entry.material);
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
std::vector<int> VulkanRenderer::restingFaces(const PlacedPiece &placed) const {
  std::vector<int> resting;
  for (int f = 0; f < static_cast<int>(placed.faces.size()); ++f) {
    const FractureFace &face = placed.faces[f];
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
      if (hit && !(hit->userData & kBreakBodyFlag) && physics->mass(hit->body) == 0.0f) {
        resting.push_back(f);
        break;
      }
    }
  }
  return resting;
}

// A ground link for each of the piece's faces resting on something fixed
// (the whole face, for now); with one it's an anchor.
void VulkanRenderer::addGroundLinks(int piece, const PlacedPiece &placed) {
  for (int f : restingFaces(placed)) {
    const FractureFace &face = breakPieces[piece].shape->faces[f];
    pieceGroundLinks[piece].push_back(static_cast<int>(groundLinks.size()));
    groundLinks.push_back({piece, linkFace(face.points, face.normal), false});
  }
  breakPieces[piece].anchor = !pieceGroundLinks[piece].empty();
}

LinkLimits VulkanRenderer::linkLimits(int link) const {
  const Breakable &a = breakables[breakPieces[breakLinks[link].a].object];
  const Breakable &b = breakables[breakPieces[breakLinks[link].b].object];
  LinkLimits limits = a.limits;
  if (&a != &b) {
    // Between objects: the weaker of the two, and only resting unless glued.
    limits.pull = std::min(a.limits.pull, b.limits.pull);
    limits.push = std::min(a.limits.push, b.limits.push);
    limits.slide = std::min(a.limits.slide, b.limits.slide);
    limits.friction = std::min(a.limits.friction, b.limits.friction);
  }
  limits.contact = breakLinkContact[link] != 0;
  // Bullets' damage weakens it: the more battered of its two pieces.
  return weakened(limits, linkStrengthLeft(breakPieces[breakLinks[link].a].damage,
                                           breakPieces[breakLinks[link].b].damage));
}

LinkLimits VulkanRenderer::groundLimits(int piece) const {
  LinkLimits limits = breakables[breakPieces[piece].object].limits;
  limits.contact = true;
  return weakened(limits, breakPieces[piece].damage.strengthLeft());
}

// On Play: the links inside each object from its cut (each Play's seed
// varies their strengths), links between pieces of objects that touch, and
// which pieces rest on something that never breaks.
void VulkanRenderer::prepareBreakGraph() {
  breakLinks.clear();
  breakLinkPoints.clear();
  breakLinkFaces.clear();
  breakLinkContact.clear();
  groundLinks.clear();
  breakGraphReady = false;
  if (breakables.empty() || !physics)
    return;
  const auto start = std::chrono::steady_clock::now();
  playSeed = std::random_device{}();
  splitsMade = 0;
  for (const Breakable &object : breakables) {
    const std::vector<BreakLink> cut = cutLinks(*object.fracture, object.strength, object.firstPiece);
    const std::vector<LinkFace> faces = cutFaces(*object.fracture);
    for (size_t i = 0; i < cut.size(); ++i) {
      breakLinks.push_back(cut[i]);
      breakLinkPoints.push_back(object.fracture->links[i].centre);
      breakLinkFaces.push_back(faces[i]);
      breakLinkContact.push_back(0);
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
    // Things rest on each other unless one is glued (cast together).
    breakLinkFaces.push_back(movedFace(touch.face, glm::inverse(worlds[touch.a])));
    breakLinkContact.push_back(!breakables[breakPieces[touch.a].object].glued &&
                               !breakables[breakPieces[touch.b].object].glued);
  }
  varyStrengths(breakLinks, playSeed, 0.25f);
  pieceLinks.assign(breakPieces.size(), {});
  for (int i = 0; i < static_cast<int>(breakLinks.size()); ++i) {
    pieceLinks[breakLinks[i].a].push_back(i);
    pieceLinks[breakLinks[i].b].push_back(i);
  }
  int anchors = 0;
  pieceGroundLinks.assign(breakPieces.size(), {});
  for (size_t p = 0; p < breakPieces.size(); ++p) {
    breakPieces[p].anchor = false;
    if (!breakables[breakPieces[p].object].moving)
      addGroundLinks(static_cast<int>(p), placed[p]);
    anchors += breakPieces[p].anchor;
  }
  pieceShift.assign(breakPieces.size(), glm::vec3(0.0f));
  pieceTurn.assign(breakPieces.size(), glm::vec3(0.0f));
  for (std::vector<glm::vec3> *loads : {&liveForce, &liveTorque, &liveAppliedForce, &liveAppliedTorque})
    loads->assign(breakPieces.size(), glm::vec3(0.0f));
  restsOn.clear();
  stressDirty = true;
  stressFirst = true;
  gpuDrivenGBufferPass.reserveGeometry(kReservedVertices, kReservedIndices);
  if (!rubbleLayer)
    // The ground under rubble: the first thing below that doesn't break,
    // move or belong to the rubble itself.
    rubbleLayer = std::make_unique<RubbleLayer>([this](glm::vec2 p) -> std::optional<float> {
      glm::vec3 from(p.x, rubbleCastTop, p.y);
      for (int tries = 0; tries < 8; ++tries) {
        const std::optional<PhysicsRayHit> hit =
            physics->castRay(from, glm::vec3(0.0f, -1.0f, 0.0f), 100.0f);
        if (!hit)
          return std::nullopt;
        if (hit->distance < 1e-3f) {
          from.y += 10.0f; // started inside something: look again from above it
          continue;
        }
        if (!(hit->userData & (kBreakBodyFlag | kRubbleBodyFlag)) && physics->mass(hit->body) == 0.0f)
          return from.y - hit->distance;
        from.y -= hit->distance + 0.01f;
      }
      return std::nullopt;
    });
  breakGraphReady = true;
  spdlog::info("[break] {} objects, {} pieces, {} links ({} between objects), {} anchors, {:.2f} ms",
               breakables.size(), breakPieces.size(), breakLinks.size(), breakLinks.size() - inside,
               anchors,
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                   .count());
}

void VulkanRenderer::handleImpacts() {
  std::vector<PhysicsImpact> impacts = physics->takeImpacts();
  if (playing)
    soundImpacts(impacts);
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
    spdlog::info("[break] {} cuts, {:.2f} ms ({} pieces cut so far)", made,
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cutting)
                     .count(),
                 runtimePieces);
  if (!hits.empty() || made > 0) {
    stressFirst = false; // damage: from now on what's overloaded breaks
    regroupBreakables(hits);
  }
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
  // Fines: what a real break grinds to dust and gravel, on the ground under
  // the hit.
  addRubble(glm::vec3(where * glm::vec4(split.hit, 1.0f)), kFines * old.shape->volume * kBulking,
            kFinesReach, rawColour(object.colour));

  // The new pieces: big enough to stay pieces, or rubble now.
  std::vector<int> made(cut.pieces.size(), -1);
  for (size_t k = 0; k < cut.pieces.size(); ++k) {
    const FracturePiece &shape = cut.pieces[k];
    if (pieceSize(shape) < kSmallestPiece) {
      addRubble(glm::vec3(where * glm::vec4(shape.centre, 1.0f)), shape.volume * kBulking,
                kDepositReach, rawColour(object.colour));
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
    made[k] = static_cast<int>(breakPieces.size());
    breakPieces.push_back(std::move(piece));
    pieceLinks.emplace_back();
    pieceGroundLinks.emplace_back();
    pieceShift.push_back(glm::vec3(0.0f));
    pieceTurn.push_back(glm::vec3(0.0f));
    for (std::vector<glm::vec3> *loads : {&liveForce, &liveTorque, &liveAppliedForce, &liveAppliedTorque})
      loads->push_back(glm::vec3(0.0f));
    // Resting where the old piece rested, if it still did: ground links of
    // its own (a piece of something already falling gets none).
    if (old.anchor)
      addGroundLinks(made[k], placePiece(shape, where, 0));
    if (old.body >= 0)
      breakBodies[old.body].pieces.push_back(made[k]);
    ++runtimePieces;
  }
  breakPieces[split.piece].alive = false;
  for (int g : pieceGroundLinks[split.piece])
    groundLinks[g].broken = true;
  stressDirty = true;
  SceneNode *goneNode = old.node;
  rootNode.removeChildren([&](const SceneNode &node) { return &node == goneNode; });

  // faceWorld faces from a to b; it's kept, like the point, in the frame of
  // the link's first (lower) piece.
  auto addLink = [&](int a, int b, float strength, glm::vec3 pointInA, LinkFace faceWorld,
                     bool contact) {
    const int index = static_cast<int>(breakLinks.size());
    const int lower = std::min(a, b);
    if (a != lower)
      faceWorld.normal = -faceWorld.normal;
    breakLinkFaces.push_back(
        movedFace(faceWorld, glm::inverse(poseMatrix(pieceWorld(breakPieces[lower])))));
    breakLinkContact.push_back(contact);
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
  const std::vector<LinkFace> innerFaces = cutFaces(cut);
  for (size_t i = 0; i < cut.links.size(); ++i)
    if (made[cut.links[i].a] >= 0 && made[cut.links[i].b] >= 0)
      addLink(made[cut.links[i].a], made[cut.links[i].b], inner[i].strength, cut.links[i].centre,
              movedFace(innerFaces[i], where), false);
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
    const bool contact = breakLinkContact[link] != 0;
    for (size_t k = 0; k < made.size(); ++k) {
      if (made[k] < 0)
        continue;
      glm::vec3 centre;
      const PlacedPiece placedNew = placePiece(cut.pieces[k], where, 0);
      const float area = touchArea(placedNew, placedOther, kTouchTolerance, centre);
      if (area > 1e-5f)
        addLink(made[k], other, area * strength,
                glm::vec3(glm::inverse(where) * glm::vec4(centre, 1.0f)),
                touchFace(placedNew, placedOther, kTouchTolerance), contact);
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
  stressDirty = true; // links broke: the weight goes other ways now
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
  // Where each replaced body was: what sleeps on it is woken once the new
  // bodies are in, or it would hang in the air (Jolt doesn't wake what
  // rests on a removed body) and its weight would rest on nothing.
  std::vector<std::pair<glm::vec3, glm::vec3>> wake;
  for (int b : dirty) {
    glm::vec3 lo(std::numeric_limits<float>::max()), hi(-std::numeric_limits<float>::max());
    for (int p : breakBodies[b].pieces) {
      if (!breakPieces[p].alive)
        continue;
      for (const FractureFace &face : breakPieces[p].shape->faces)
        for (const glm::vec3 &q : face.points) {
          const glm::vec3 w = world[p].position + world[p].rotation * q;
          lo = glm::min(lo, w);
          hi = glm::max(hi, w);
        }
    }
    if (lo.x <= hi.x)
      wake.push_back({lo - glm::vec3(0.1f), hi + glm::vec3(0.1f)});
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
  for (const auto &[lo, hi] : wake)
    physics->activateBodiesIn(lo, hi);
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
    // Under it: the ground, the rubble or something else that doesn't break
    // or move. Its own body is in the way of a ray from its middle, so it's
    // ignored by starting just under its lowest point.
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
    for (int link : pieceLinks[index])
      breakLinks[link].broken = true;
    // It stays where it lies, a stone on the pile, which rises around it.
    piece.node->setPose(world.position, world.rotation);
    rubbleStones.push_back({piece.node, glm::vec2(middle.x, middle.z), world.position, world.rotation,
                            world.position.y - lowest});
    addRubble(middle, piece.shape->volume * kBulking,
              std::max(kDepositReach, 1.5f * pieceSize(*piece.shape)),
              rawColour(breakables[piece.object].colour));
  }
}

void VulkanRenderer::addRubble(glm::vec3 where, float volume, float radius, glm::vec3 colour) {
  if (!rubbleLayer || !(volume > 0.0f))
    return;
  rubbleFeed.push_back({glm::vec2(where.x, where.z), volume, volume / kFeedSeconds, radius,
                        where.y + kCastAbove, colour});
}

void VulkanRenderer::updateRubble(float seconds) {
  if (!rubbleLayer)
    return;
  const auto start = std::chrono::steady_clock::now();
  for (RubbleFeed &feed : rubbleFeed) {
    const float amount = std::min(feed.volume, feed.rate * seconds);
    rubbleCastTop = feed.castTop;
    rubbleLayer->deposit(feed.at, amount, feed.radius, feed.colour);
    feed.volume -= amount;
  }
  std::erase_if(rubbleFeed, [](const RubbleFeed &feed) { return feed.volume <= 1e-7f; });
  rubbleLayer->settle(kSlumpPasses);
  for (RubbleKey key : rubbleLayer->takeChanged())
    if (std::find(rubbleWaiting.begin(), rubbleWaiting.end(), key) == rubbleWaiting.end())
      rubbleWaiting.push_back(key);
  rubbleRemeshWait -= seconds;
  if (rubbleWaiting.empty() || rubbleRemeshWait > 0.0f)
    return;
  rubbleRemeshWait = kRemeshSeconds;

  // A few tiles at a time, skipping those that barely changed (slumping's
  // last fractions of a millimetre).
  std::vector<RubbleKey> keys;
  std::vector<std::vector<float>> newHeights;
  size_t looked = 0;
  while (looked < rubbleWaiting.size() && keys.size() < static_cast<size_t>(kTilesPerFrame)) {
    const RubbleKey key = rubbleWaiting[looked++];
    std::vector<float> heights = rubbleLayer->tileHeights(key);
    const auto drawn = rubbleTiles.find(key);
    if (drawn != rubbleTiles.end() && drawn->second.modelId >= 0 &&
        !movedMuch(drawn->second.heights, heights))
      continue;
    keys.push_back(key);
    newHeights.push_back(std::move(heights));
  }
  rubbleWaiting.erase(rubbleWaiting.begin(), rubbleWaiting.begin() + looked);
  if (keys.empty())
    return;
  std::erase_if(uploadsInFlight, [](const std::unique_ptr<UploadBatch> &b) { return b->finished(); });
  auto batch = std::make_unique<UploadBatch>(device.getLogicalDevice(), device.getGraphicsCommandPool(),
                                             device.getGraphicsQueue());
  gUploadBatch = batch.get();
  // New tiles' models first: registering them may grow the shared buffer,
  // which replaces it, so the rewrites in place come after.
  std::vector<int> fresh;
  std::vector<std::pair<int, std::vector<Vertex>>> rewrites;
  for (size_t n = 0; n < keys.size(); ++n) {
    const RubbleKey key = keys[n];
    MeshData mesh = rubbleLayer->tileMesh(key);
    if (mesh.vertices.empty())
      continue;
    RubbleTileDraw &tile = rubbleTiles[key];
    const glm::vec2 corner = rubbleTileCorner(key);
    if (tile.modelId < 0) {
      // A tile's model is kept under its place, so another Play rewrites it
      // instead of making a new one.
      const std::string name = std::format("#rubble|{}|{}", key.first, key.second);
      tile.modelId = modelManager.findModel(name);
      float middle = 0.0f;
      for (const Vertex &v : mesh.vertices)
        middle += v.pos.y;
      middle /= static_cast<float>(mesh.vertices.size());
      if (tile.modelId < 0) {
        tile.modelId = modelManager.createModel(name, mesh, false, false, device, textureManager,
                                                descriptorManager);
        fresh.push_back(tile.modelId);
      } else {
        rewrites.emplace_back(tile.modelId, std::move(mesh.vertices));
      }
      // Culling: a sphere over the whole tile and a pile up to about 2.5 m.
      if (MeshModel *model = modelManager.getModel(tile.modelId))
        model->setBoundingSphere(glm::vec3(0.5f * kRubbleTile, middle + 1.0f, 0.5f * kRubbleTile),
                                 3.2f);
      auto node = std::make_unique<SceneNode>();
      node->setModelId(tile.modelId);
      node->setPose(glm::vec3(corner.x, 0.0f, corner.y), glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
      tile.node = node.get();
      rootNode.addChild(std::move(node));
    } else {
      rewrites.emplace_back(tile.modelId, std::move(mesh.vertices));
    }

    // Its height field, and whatever rests on it woken, whichever way it
    // moved: risen, so it's lifted a little at a time instead of left inside
    // once the field passes its middle (a height field is one-sided);
    // dropped, so it doesn't hover.
    std::vector<float> heights = std::move(newHeights[n]);
    float low = std::numeric_limits<float>::max(), high = -low;
    for (const std::vector<float> *list : {&heights, &tile.heights})
      for (float h : *list)
        if (!std::isnan(h)) {
          low = std::min(low, h);
          high = std::max(high, h);
        }
    if (!tile.body.isValid()) {
      PhysicsBodyDesc desc;
      desc.shape = PhysicsShape::HeightField;
      desc.motion = PhysicsMotion::Static;
      desc.heights = heights;
      desc.heightSamples = kRubbleSamples;
      desc.heightSpacing = kRubbleSpacing;
      desc.position = glm::vec3(corner.x, 0.0f, corner.y);
      desc.friction = 0.8f;
      desc.userData = kRubbleBodyFlag;
      tile.body = physics->addBody(desc); // invalid while it's all too thin: made next time
    } else {
      physics->setHeights(tile.body, heights, kRubbleSamples, kRubbleSpacing);
    }
    if (high >= low)
      physics->activateBodiesIn(glm::vec3(corner.x, low - 1.0f, corner.y),
                                glm::vec3(corner.x + kRubbleTile, high + 1.0f, corner.y + kRubbleTile));
    tile.heights = std::move(heights);
  }
  if (!fresh.empty())
    gpuDrivenGBufferPass.registerModelGeometries(fresh, modelManager);
  for (auto &[modelId, vertices] : rewrites) {
    modelManager.updateModelVertices(modelId, vertices);
    gpuDrivenGBufferPass.updateModelVertices(modelId, modelManager);
  }
  gUploadBatch = nullptr;
  batch->submit();
  uploadsInFlight.push_back(std::move(batch));

  // Stones follow the surface down where the pile slumped away under them.
  for (RubbleStone &stone : rubbleStones) {
    if (std::find(keys.begin(), keys.end(), rubbleKeyAt(stone.at)) == keys.end())
      continue;
    const float surface = rubbleLayer->surface(stone.at);
    if (!std::isnan(surface) && stone.position.y - stone.below > surface + 0.02f) {
      stone.position.y = surface + stone.below;
      stone.node->setPose(stone.position, stone.rotation);
    }
  }
  spdlog::debug("[rubble] {} tiles, {:.3f} ms ({} in all, {:.2f} m^3)", keys.size(),
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count(),
                rubbleTiles.size(), rubbleLayer->volume());
}

// Weight: the pieces that stay put, balanced under gravity by the springs
// at their links (Stress.h). Built again when the structure changes, solved
// a few dozen iterations a frame until it settles, then the links over
// their limits break, the worst few a frame, and what that frees falls.
void VulkanRenderer::updateStress(float seconds) {
  if (!breakGraphReady)
    return;
  const auto start = std::chrono::steady_clock::now();
  sinceJudged += seconds;
  stressFrame = seconds;
  gatherLiveLoads();
  if (stressDirty) {
    stressDirty = false;
    // What stays put now, and the links among it; if that's what's being
    // solved already (a regroup of things that move), it carries on.
    std::vector<int> pieceOf(breakPieces.size(), -1);
    std::vector<int> order;
    for (int p = 0; p < static_cast<int>(breakPieces.size()); ++p) {
      const BreakPiece &piece = breakPieces[p];
      if (!piece.alive || piece.body < 0 || !breakBodies[piece.body].fixed ||
          !breakBodies[piece.body].id.isValid())
        continue;
      pieceOf[p] = static_cast<int>(order.size());
      order.push_back(p);
    }
    std::vector<SolvedLink> solved;
    for (int l = 0; l < static_cast<int>(breakLinks.size()); ++l)
      if (!breakLinks[l].broken && pieceOf[breakLinks[l].a] >= 0 && pieceOf[breakLinks[l].b] >= 0)
        solved.push_back({l, false});
    for (int g = 0; g < static_cast<int>(groundLinks.size()); ++g)
      if (!groundLinks[g].broken && pieceOf[groundLinks[g].piece] >= 0)
        solved.push_back({g, true});
    const bool same = order == stressOrder && solved.size() == solvedLinks.size() &&
                      std::equal(solved.begin(), solved.end(), solvedLinks.begin(),
                                 [](const SolvedLink &x, const SolvedLink &y) {
                                   return x.index == y.index && x.ground == y.ground;
                                 });
    // The mapping goes with breakPieces' size even when nothing that stays
    // put changed (a moving chunk was cut).
    stressPieceOf = std::move(pieceOf);
    if (!same) {
      stressOrder = std::move(order);
      solvedLinks = std::move(solved);
      std::vector<StressPiece> pieces;
      std::vector<glm::mat4> frames;
      for (int p : stressOrder) {
        const PhysicsPose world = pieceWorld(breakPieces[p]);
        frames.push_back(poseMatrix(world));
        pieces.push_back({world.position + world.rotation * breakPieces[p].shape->centre,
                          breakPieces[p].shape->volume * breakables[breakPieces[p].object].density});
      }
      std::vector<StressLink> links;
      for (const SolvedLink &link : solvedLinks) {
        if (link.ground) {
          const int a = stressPieceOf[groundLinks[link.index].piece];
          links.push_back({a, -1, movedFace(groundLinks[link.index].face, frames[a]), true});
        } else {
          const int a = stressPieceOf[breakLinks[link.index].a];
          const int b = stressPieceOf[breakLinks[link.index].b];
          links.push_back({a, b, movedFace(breakLinkFaces[link.index], frames[a]),
                           breakLinkContact[link.index] != 0});
        }
      }
      stressSolver.build(pieces, links);
      std::vector<glm::vec3> shifts(pieces.size()), turns(pieces.size());
      for (size_t k = 0; k < stressOrder.size(); ++k) {
        shifts[k] = pieceShift[stressOrder[k]];
        turns[k] = pieceTurn[stressOrder[k]];
      }
      stressSolver.setStart(shifts, turns);
      solvedLoad.clear();
      loadLines.clear();
      overloadSeconds.assign(solvedLinks.size(), 0.0f);
      stressStarted = 0.0;
      stressJudged = false;
      // What rests on it carries over into the new solve.
      std::fill(liveAppliedForce.begin(), liveAppliedForce.end(), glm::vec3(0.0f));
      std::fill(liveAppliedTorque.begin(), liveAppliedTorque.end(), glm::vec3(0.0f));
    }
  }
  applyLiveLoads();
  if (stressJudged)
    return; // nothing changed since it settled
  const bool settled = stressSolver.solve(kStressIterations);
  stressStarted +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  if (!settled && stressSolver.iterations() < kStressMostIterations)
    return;
  if (!settled) {
    // It crawls when something is tipping off its support: a nearly free
    // mode. Its unfinished answer is no good for judging the rest, so only
    // the contacts tipping open (they're why), and it's solved again.
    std::vector<int> tipping;
    for (size_t i = 0; i < solvedLinks.size(); ++i) {
      const SolvedLink &solved = solvedLinks[i];
      const LinkLimits limits =
          solved.ground ? groundLimits(groundLinks[solved.index].piece) : linkLimits(solved.index);
      if (::tipping(stressSolver.stress(static_cast<int>(i)), limits) > 1.0f)
        tipping.push_back(static_cast<int>(i));
    }
    if (tipping.empty()) {
      // Nothing tipping: just slow. It carries on, up to a few times the
      // cap, then gives up on this structure until it changes again.
      if (stressSolver.iterations() < 4 * kStressMostIterations)
        return;
      spdlog::warn("[weight] {} pieces didn't settle in {} iterations; not judged",
                   stressOrder.size(), stressSolver.iterations());
      stressJudged = true;
      return;
    }
    spdlog::info("[weight] {} pieces didn't settle in {} iterations, {:.2f} ms; {} contacts tipping",
                 stressOrder.size(), stressSolver.iterations(), stressStarted, tipping.size());
    stressJudged = true;
    for (int i : tipping) {
      const SolvedLink &solved = solvedLinks[i];
      if (solved.ground) {
        GroundLink &ground = groundLinks[solved.index];
        ground.broken = true;
        bool holds = false;
        for (int g : pieceGroundLinks[ground.piece])
          holds = holds || !groundLinks[g].broken;
        breakPieces[ground.piece].anchor = holds;
      } else {
        breakLinks[solved.index].broken = true;
      }
    }
    regroupBreakables({});
    return;
  }
  const std::vector<glm::vec3> shifts = stressSolver.shifts(), turns = stressSolver.turns();
  for (size_t i = 0; i < stressOrder.size(); ++i) {
    pieceShift[stressOrder[i]] = shifts[i];
    pieceTurn[stressOrder[i]] = turns[i];
  }

  // How close every link is to breaking, and the load view's lines.
  stressJudged = true;
  solvedLoad.assign(solvedLinks.size(), 0.0f);
  std::vector<std::pair<float, int>> over;
  std::vector<int> open; // contacts that lifted off: they hold nothing
  // Structures (pieces joined by the solved links) carrying live loads: a
  // contact lifted there stays lifted rather than opening for good, since
  // what lifted it may move off and it should press back down.
  std::vector<int> joined(stressOrder.size());
  std::iota(joined.begin(), joined.end(), 0);
  auto root = [&](int x) {
    while (joined[x] != x)
      x = joined[x] = joined[joined[x]];
    return x;
  };
  for (const SolvedLink &solved : solvedLinks)
    if (!solved.ground)
      joined[root(stressPieceOf[breakLinks[solved.index].a])] = root(stressPieceOf[breakLinks[solved.index].b]);
  std::vector<char> loadedRoot(stressOrder.size(), 0);
  for (size_t k = 0; k < stressOrder.size(); ++k) {
    const int p = stressOrder[k];
    if (glm::length(liveAppliedForce[p]) > 1.0f || glm::length(liveAppliedTorque[p]) > 1.0f)
      loadedRoot[root(static_cast<int>(k))] = 1;
  }
  std::vector<char> loaded(stressOrder.size(), 0);
  for (size_t k = 0; k < stressOrder.size(); ++k)
    loaded[k] = loadedRoot[root(static_cast<int>(k))];
  for (size_t i = 0; i < solvedLinks.size(); ++i) {
    const SolvedLink &solved = solvedLinks[i];
    const LinkLimits limits =
        solved.ground ? groundLimits(groundLinks[solved.index].piece) : linkLimits(solved.index);
    const LinkStress stress = stressSolver.stress(static_cast<int>(i));
    if (limits.contact && stress.liftedShare >= 1.0f) {
      if (!loaded[stressPieceOf[solved.ground ? groundLinks[solved.index].piece
                                              : breakLinks[solved.index].a]])
        open.push_back(static_cast<int>(i));
      continue;
    }
    solvedLoad[i] = overload(stress, limits);
    if (solvedLoad[i] > 1.0f)
      over.push_back({solvedLoad[i], static_cast<int>(i)});
  }
  loadLines.clear();
  for (size_t i = 0; i < solvedLinks.size(); ++i) {
    const SolvedLink &solved = solvedLinks[i];
    const float t = std::clamp(solvedLoad[i], 0.0f, 1.0f);
    const uint32_t r = static_cast<uint32_t>(255.0f * std::min(1.0f, 2.0f * t));
    const uint32_t g = static_cast<uint32_t>(255.0f * std::min(1.0f, 2.0f * (1.0f - t)));
    const uint32_t colour = r | g << 8 | 0x20u << 16 | 0xffu << 24;
    auto middle = [&](int piece) {
      const PhysicsPose w = pieceWorld(breakPieces[piece]);
      return w.position + w.rotation * breakPieces[piece].shape->centre;
    };
    const int a = solved.ground ? groundLinks[solved.index].piece : breakLinks[solved.index].a;
    const PhysicsPose wa = pieceWorld(breakPieces[a]);
    const LinkFace face =
        movedFace(solved.ground ? groundLinks[solved.index].face : breakLinkFaces[solved.index],
                  poseMatrix(wa));
    const glm::vec3 far = solved.ground ? face.centre + face.normal * 0.2f
                                        : middle(breakLinks[solved.index].b);
    for (const glm::vec3 &q : {middle(a), face.centre, face.centre, far})
      loadLines.push_back({q, colour});
  }

  std::sort(over.begin(), over.end(), std::greater<>());
  // How long each has been over: a link breaks only once it has been for
  // kOverloadSeconds (live loads flicker).
  // A link newly over counts from this frame, not from the last judging
  // (the solve may have sat idle for seconds in between).
  for (size_t i = 0; i < solvedLinks.size(); ++i)
    overloadSeconds[i] = overloadTime(overloadSeconds[i], solvedLoad[i],
                                      overloadSeconds[i] > 0.0f ? sinceJudged : stressFrame);
  sinceJudged = 0.0f;
  std::vector<std::pair<float, int>> due;
  for (const auto &[load, i] : over)
    if (overloadSeconds[i] >= kOverloadSeconds)
      due.push_back({load, i});
  if (stressSolver.iterations() != loggedIterations || !open.empty() || !due.empty())
    spdlog::info("[weight] {} pieces, {} links settled in {} iterations, {:.2f} ms; {} open, {} over "
                 "their limits{}",
                 stressOrder.size(), solvedLinks.size(), stressSolver.iterations(), stressStarted,
                 open.size(), over.size(),
                 over.empty() ? std::string() : std::format(" (the worst {:.2f})", over.front().first));
  stressStarted = 0.0;
  loggedIterations = stressSolver.iterations();
  auto breakSolved = [&](int i) {
    const SolvedLink &solved = solvedLinks[i];
    if (solved.ground) {
      GroundLink &ground = groundLinks[solved.index];
      ground.broken = true;
      bool holds = false;
      for (int g : pieceGroundLinks[ground.piece])
        holds = holds || !groundLinks[g].broken;
      breakPieces[ground.piece].anchor = holds;
    } else {
      breakLinks[solved.index].broken = true;
    }
  };
  // Open contacts aren't failures: things resting side by side, a roof off
  // a wall it doesn't press. They open now, even on Play, and the rest is
  // solved again without them (not where live loads are on, above).
  if (!open.empty()) {
    for (int i : open)
      breakSolved(i);
    regroupBreakables({});
    return;
  }
  if (stressFirst) {
    stressFirst = false;
    if (!over.empty())
      spdlog::warn("[weight] {} links can't hold this scene up as it is; they break once anything "
                   "changes",
                   over.size());
    return;
  }
  if (over.empty())
    return;
  if (due.empty()) {
    stressJudged = false; // over, not for long enough yet: judged again next frame
    return;
  }
  for (int k = 0; k < std::min<int>(kStressBreaksPerFrame, static_cast<int>(due.size())); ++k)
    breakSolved(due[k].second);
  regroupBreakables({});
}

// What rests on the structure: moving bodies' weight shared among the
// contacts holding them up, and every character's where it stands, on the
// piece under each point; summed per piece and smoothed.
// The body's piece the point is least outside of: the one it's on, or in.
int VulkanRenderer::pieceAt(const BreakBody &body, glm::vec3 point) const {
  int on = -1;
  float least = std::numeric_limits<float>::max();
  for (int p : body.pieces) {
    if (!breakPieces[p].alive)
      continue;
    const PhysicsPose world = pieceWorld(breakPieces[p]);
    float outside = -std::numeric_limits<float>::max();
    for (const FractureFace &face : breakPieces[p].shape->faces)
      outside = std::max(outside, glm::dot(world.rotation * face.normal,
                                           point - (world.position + world.rotation * face.points[0])));
    if (outside < least) {
      least = outside;
      on = p;
    }
  }
  return on;
}

void VulkanRenderer::gatherLiveLoads() {
  if (liveForce.size() != breakPieces.size())
    return;
  std::unordered_map<uint32_t, std::vector<PhysicsRest>> fresh;
  for (const PhysicsRest &rest : physics->takeRests())
    fresh[rest.on.value].push_back(rest);
  // A body that said nothing is still resting if it sleeps; awake and
  // silent, or gone, it has lifted off.
  std::erase_if(restsOn, [&](const auto &entry) {
    if (fresh.contains(entry.first))
      return false;
    const PhysicsBodyId body{entry.first};
    return physics->mass(body) <= 0.0f || physics->isActive(body);
  });
  for (auto &[id, list] : fresh)
    restsOn[id] = std::move(list);

  std::vector<glm::vec3> force(breakPieces.size(), glm::vec3(0.0f));
  std::vector<glm::vec3> torque(breakPieces.size(), glm::vec3(0.0f));
  auto put = [&](uint64_t userData, glm::vec3 point, glm::vec3 weight) {
    if (!(userData & kBreakBodyFlag))
      return;
    const uint64_t index = userData & ~kBreakBodyFlag;
    if (index >= breakBodies.size() || !breakBodies[index].fixed || !breakBodies[index].id.isValid())
      return;
    const int on = pieceAt(breakBodies[index], point);
    if (on < 0)
      return;
    const PhysicsPose world = pieceWorld(breakPieces[on]);
    force[on] += weight;
    torque[on] += glm::cross(point - (world.position + world.rotation * breakPieces[on].shape->centre),
                             weight);
  };
  const glm::vec3 gravity = physics->gravity();
  for (const auto &[id, rests] : restsOn) {
    const float mass = physics->mass(PhysicsBodyId{id});
    int holding = 0;
    for (const PhysicsRest &rest : rests)
      holding += rest.normal.y >= kSupportFacing;
    if (mass <= 0.0f || holding == 0)
      continue;
    for (const PhysicsRest &rest : rests)
      if (rest.normal.y >= kSupportFacing)
        put(rest.userData, rest.point, gravity * (mass / static_cast<float>(holding)));
  }
  if (roster)
    for (size_t i = 0; i < roster->size(); ++i) {
      const PhysicsCharacterState state = physics->characterState(roster->body(i).physicsId());
      if (state.groundBody.isValid())
        put(state.groundUserData, state.groundPoint, gravity * roster->body(i).tuning().mass);
    }
  for (size_t p = 0; p < breakPieces.size(); ++p) {
    liveForce[p] = glm::mix(liveForce[p], force[p], kLiveSmoothing);
    liveTorque[p] = glm::mix(liveTorque[p], torque[p], kLiveSmoothing);
  }
}

// The smoothed live loads into the solve, once they've moved enough to
// matter (a person walking on a slab: every few frames, not every one).
void VulkanRenderer::applyLiveLoads() {
  if (liveForce.size() != breakPieces.size() || stressPieceOf.size() != breakPieces.size())
    return;
  float moved = 0.0f;
  for (size_t p = 0; p < breakPieces.size(); ++p)
    moved = std::max({moved, glm::length(liveForce[p] - liveAppliedForce[p]),
                      glm::length(liveTorque[p] - liveAppliedTorque[p])});
  if (moved <= kLiveLoadStep)
    return;
  liveAppliedForce = liveForce;
  liveAppliedTorque = liveTorque;
  std::vector<LiveLoad> loads;
  for (size_t p = 0; p < breakPieces.size(); ++p)
    if (stressPieceOf[p] >= 0 && (glm::length(liveForce[p]) > 1.0f || glm::length(liveTorque[p]) > 1.0f))
      loads.push_back({stressPieceOf[p], liveForce[p], liveTorque[p]});
  stressSolver.setLiveLoads(loads);
  stressJudged = false;
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

void VulkanRenderer::queueBulletSplit(int p, glm::vec3 point, glm::vec3 push) {
  const BreakPiece &piece = breakPieces[p];
  const Breakable &object = breakables[piece.object];
  if (!piece.alive || object.pattern == FracturePattern::None ||
      pieceSize(*piece.shape) < 2.0f * kSmallestPiece || runtimePieces >= kRuntimePieceBudget)
    return;
  for (const PendingSplit &queued : pendingSplits)
    if (queued.piece == p)
      return;
  // splitPiece cuts 4 sqrt(impulse / what it holds) pieces: about six.
  const float holds = std::max(kMinSplitImpulse, splitImpulse(piece.shape->volume, object.strength));
  const PhysicsPose world = pieceWorld(piece);
  const glm::quat toLocal = glm::conjugate(world.rotation);
  pendingSplits.push_back({p, toLocal * (point - world.position), toLocal * push, 2.25f * holds});
}
