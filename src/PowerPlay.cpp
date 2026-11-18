#include "VulkanRenderer.h"

#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>

// The powers (step 1.4a): the renderer's side of Powers.h. Each frame the
// pickable things near the player are gathered, the one under the crosshair
// found, and the controller told what the player does; what it says to do
// happens here: the held body pulled toward the hold point every physics
// step, thrown, dropped, a piece torn out link by link (each one through the
// regroup, so the weight solver sees it), the burst pushing bodies and
// hitting breakables through handleImpacts. Drawn as lines until the shader
// pass (step 4).

namespace {

// 0xRRGGBB, as bright as `brightness`, in the lines' RGBA8 (red first).
uint32_t lineColour(uint32_t rgb, float brightness = 1.0f) {
  const float k = std::clamp(brightness, 0.0f, 1.0f);
  const uint32_t r = static_cast<uint32_t>(((rgb >> 16) & 0xffu) * k);
  const uint32_t g = static_cast<uint32_t>(((rgb >> 8) & 0xffu) * k);
  const uint32_t b = static_cast<uint32_t>((rgb & 0xffu) * k);
  return r | g << 8 | b << 16 | 0xffu << 24;
}

constexpr uint32_t kRevealRgb = 0x8fd0ff; // pale blue
constexpr uint32_t kTargetRgb = 0xc0ecff;
constexpr uint32_t kHoldRgb = 0xffe4a0; // warm gold-white
constexpr uint32_t kTearRgb = 0xffb050;
constexpr uint32_t kStrainRgb = 0xff6040;
constexpr uint32_t kBurstRgb = 0x9ad8ff;
constexpr float kTrailSeconds = 0.6f;
constexpr float kBurstShown = 0.25f; // s the burst's cone shows
constexpr float kRingFade = 0.5f;    // s the pulse's ring fades once it's out
constexpr int kSightChecks = 4;      // candidates checked for a clear view, nearest the line first
constexpr int kRibbonSegments = 12;
constexpr float kWheelStep = 0.5f;   // m a wheel notch moves what's held

void addLine(std::vector<PhysicsDebugVertex> &lines, glm::vec3 a, glm::vec3 b, uint32_t c) {
  lines.push_back({a, c});
  lines.push_back({b, c});
}

void addBox(std::vector<PhysicsDebugVertex> &lines, glm::vec3 lo, glm::vec3 hi, uint32_t c) {
  const auto corner = [&](int i) {
    return glm::vec3(i & 1 ? hi.x : lo.x, i & 2 ? hi.y : lo.y, i & 4 ? hi.z : lo.z);
  };
  for (int i = 0; i < 8; ++i)
    for (int bit : {1, 2, 4})
      if (!(i & bit))
        addLine(lines, corner(i), corner(i | bit), c);
}

void addCircle(std::vector<PhysicsDebugVertex> &lines, glm::vec3 centre, float radius, uint32_t c) {
  constexpr int kSegments = 48;
  for (int i = 0; i < kSegments; ++i) {
    const float a0 = glm::two_pi<float>() * i / kSegments;
    const float a1 = glm::two_pi<float>() * (i + 1) / kSegments;
    addLine(lines, centre + radius * glm::vec3(std::cos(a0), 0.0f, std::sin(a0)),
            centre + radius * glm::vec3(std::cos(a1), 0.0f, std::sin(a1)), c);
  }
}

// A bowed, swaying line from the hand to what it holds: the ribbon's
// stand-in.
void addRibbon(std::vector<PhysicsDebugVertex> &lines, glm::vec3 from, glm::vec3 to, float time,
               uint32_t c) {
  const float length = glm::distance(from, to);
  const glm::vec3 bow = 0.5f * (from + to) + glm::vec3(0.0f, 0.25f * length, 0.0f) +
                        glm::vec3(std::sin(3.0f * time), 0.0f, std::cos(2.3f * time)) * 0.15f * length;
  glm::vec3 last = from;
  for (int i = 1; i <= kRibbonSegments; ++i) {
    const float u = static_cast<float>(i) / kRibbonSegments;
    const glm::vec3 p = (1.0f - u) * (1.0f - u) * from + 2.0f * u * (1.0f - u) * bow + u * u * to;
    addLine(lines, last, p, c);
    last = p;
  }
}

// A number from 0 to 1 for each n (a hash), for tumbles and shakes.
float hash01(uint32_t n) {
  n = (n ^ 61u) ^ (n >> 16);
  n *= 9u;
  n ^= n >> 4;
  n *= 0x27d4eb2du;
  n ^= n >> 15;
  return static_cast<float>(n & 0xffffffu) / static_cast<float>(0x1000000);
}

enum TargetKind : int { kPieceTarget = 0, kChunkTarget = 1, kSceneTarget = 2 };
int targetKind(int id) { return id & 3; }
int targetIndex(int id) { return id >> 2; }

} // namespace

glm::vec3 VulkanRenderer::randomAxis() {
  const float z = 2.0f * hash01(++powerDraws * 2654435761u) - 1.0f;
  const float phi = glm::two_pi<float>() * hash01(++powerDraws * 2246822519u);
  const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
  return glm::vec3(r * std::cos(phi), r * std::sin(phi), z);
}

const PowerTuning &VulkanRenderer::powerTuning() const {
  if (playedCharacter >= 0 && playedCharacter < static_cast<int>(characters.size()))
    return characters[playedCharacter].tuning.powers;
  return characterPanel.character.powers;
}

float VulkanRenderer::pieceMass(int piece) const {
  const BreakPiece &p = breakPieces[piece];
  return p.shape->volume * breakables[p.object].density;
}

glm::vec3 VulkanRenderer::pieceCentre(int piece) const {
  const BreakPiece &p = breakPieces[piece];
  const PhysicsPose world = pieceWorld(p);
  return world.position + world.rotation * p.shape->centre;
}

glm::vec3 VulkanRenderer::bodyCentre(PhysicsBodyId body) const {
  const auto [lo, hi] = physics->bounds(body);
  return 0.5f * (lo + hi);
}

// The middle of the held body's mass: its pieces' middles weighted by their
// volume for a breakable's chunk (one material, one density), the body's own
// place for a scene object (their shapes are centred on it).
glm::vec3 VulkanRenderer::heldCentre() const {
  if (targetKind(powerGrabbed.id) == kSceneTarget || heldPiece < 0)
    return physics->pose(heldBody).position;
  const int b = breakPieces[heldPiece].body;
  glm::vec3 sum(0.0f);
  float volume = 0.0f;
  for (int p : breakBodies[b].pieces)
    if (breakPieces[p].alive) {
      sum += pieceCentre(p) * breakPieces[p].shape->volume;
      volume += breakPieces[p].shape->volume;
    }
  return volume > 0.0f ? sum / volume : physics->pose(heldBody).position;
}

// The look (step 4) ---------------------------------------------------------

namespace {

// Linear HDR colours: the light's warm core and blue edge, the reveal's and
// the target's pale blue, the tear's orange, the strain's red.
const glm::vec3 kGoldCore(1.0f, 0.62f, 0.22f);
const glm::vec3 kBlueEdge(0.20f, 0.45f, 1.0f);
const glm::vec3 kPaleBlue(0.25f, 0.55f, 1.0f);
const glm::vec3 kTearOrange(1.0f, 0.45f, 0.12f);
const glm::vec3 kStrainRed(1.0f, 0.18f, 0.08f);
// How bright each is: multipliers on those colours, above bloom's threshold.
constexpr float kRevealBright = 1.2f;
constexpr float kTargetBright = 1.0f;
constexpr float kHeldBright = 1.4f;
constexpr float kTearBright = 1.3f;
constexpr float kRibbonCore = 1.4f;
constexpr float kRibbonEdge = 0.9f;
constexpr float kTrailBright = 2.0f;
constexpr float kBurstBright = 1.2f;
constexpr float kRingBright = 1.6f;
// The even light over a target's, a held or a torn thing's faces (the
// reveal's glow is the rim alone).
constexpr float kGlowFloor = 0.15f;
constexpr int kStrands = 3;
constexpr int kStrandSegments = 24;
constexpr int kRingSegments = 96;
constexpr int kConeSegments = 40;

} // namespace

void VulkanRenderer::addPowerGlow(const PowerTarget &target, glm::vec3 colour, float strength,
                                  float floor) {
  const auto add = [&](int piece) {
    if (piece >= 0 && piece < static_cast<int>(breakPieces.size()) && breakPieces[piece].alive &&
        breakPieces[piece].node)
      powerGlows.push_back({breakPieces[piece].node, colour, strength, floor});
  };
  switch (targetKind(target.id)) {
  case kPieceTarget:
    add(target.piece);
    break;
  case kChunkTarget:
    if (const size_t b = targetIndex(target.id); b < breakBodies.size())
      for (int p : breakBodies[b].pieces)
        add(p);
    break;
  case kSceneTarget:
    if (const size_t i = targetIndex(target.id); i < physicsLinks.size() && physicsLinks[i].node)
      powerGlows.push_back({physicsLinks[i].node, colour, strength, floor});
    break;
  default:
    break;
  }
}

// What glows and the light's strips this frame, from the state the debug
// lines draw: what the reveal's pulse passed, what the crosshair is on, what
// the power has hold of and the ribbons to it, the throw's trail, the
// burst's cone and the reveal's ring.
void VulkanRenderer::buildPowerEffects() {
  powerGlows.clear();
  powerStrips.clear();
  if (!physics || !playerController || !playerController->body())
    return;
  const PowerTuning &t = powerTuning();
  const float time = static_cast<float>(playSeconds); // a float stays fine for hours
  const glm::vec3 eye = playerController->camera().position();

  if (revealSince >= 0.0f) {
    const float out = t.revealReach / t.revealSpeed;
    const float fade = 1.0f - std::clamp((revealSince - out) / kRingFade, 0.0f, 1.0f);
    const float radius = std::min(revealSince * t.revealSpeed, t.revealReach);
    if (fade > 0.0f && radius > 0.1f)
      addGroundRing(powerStrips, revealFrom + glm::vec3(0.0f, 0.04f, 0.0f), radius, 0.35f,
                    glm::vec4(kPaleBlue * kRingBright, fade), glm::vec4(kPaleBlue, 1.0f),
                    kRingSegments);
    for (const PowerTarget &c : powerCandidates) {
      const float glow = revealGlow(glm::distance(c.centre, revealFrom), revealSince, t);
      if (glow > 0.02f)
        addPowerGlow(c, kPaleBlue, kRevealBright * glow);
    }
  }
  if (powerTarget) {
    const glm::vec3 colour = powerTarget->kind == GrabKind::Lift   ? kPaleBlue
                             : powerTarget->kind == GrabKind::Tear ? kTearOrange
                                                                   : kStrainRed;
    addPowerGlow(*powerTarget, colour, kTargetBright * (0.85f + 0.15f * std::sin(6.0f * time)),
                 kGlowFloor);
  }

  const PowerState state = powerController.state();
  if (const std::optional<glm::vec3> end = powerReach()) {
    glm::vec3 core = kGoldCore;
    if (state == PowerState::Tearing || state == PowerState::Straining) {
      core = state == PowerState::Tearing ? kTearOrange : kStrainRed;
      // Brighter as the tear goes on, flickering while it pulls.
      const float progress =
          powerTear.total > 0.0f ? std::clamp(powerTear.seconds / powerTear.total, 0.0f, 1.0f) : 0.0f;
      addPowerGlow(powerGrabbed, core,
                   kTearBright * (0.7f + 0.6f * progress) * (0.85f + 0.15f * std::sin(14.0f * time)),
                   kGlowFloor);
    } else if (!heldPieces.empty()) {
      for (int p : heldPieces)
        if (breakPieces[p].alive && breakPieces[p].node)
          powerGlows.push_back({breakPieces[p].node, kGoldCore, kHeldBright, kGlowFloor});
    } else {
      for (const PhysicsLink &link : physicsLinks)
        if (link.body.value == heldBody.value && link.node)
          powerGlows.push_back({link.node, kGoldCore, kHeldBright, kGlowFloor});
    }
    for (int s = 0; s < kStrands; ++s)
      addFacingStrip(powerStrips, ribbonStrand(handPosition, *end, s, time, kStrandSegments), eye,
                     0.09f, 0.05f, glm::vec4(core * kRibbonCore, 1.0f),
                     glm::vec4(core * kRibbonCore, 0.7f), glm::vec4(kBlueEdge * kRibbonEdge, 1.0f));
  }
  if (powerTrail.body.isValid() && powerTrail.points.size() >= 2) {
    const float fade = 1.0f - std::clamp(powerTrail.seconds / kTrailSeconds, 0.0f, 1.0f);
    addFacingStrip(powerStrips, powerTrail.points, eye, 0.0f, 0.35f,
                   glm::vec4(kGoldCore * kTrailBright, 0.0f),
                   glm::vec4(kGoldCore * kTrailBright, fade), glm::vec4(kBlueEdge * kRibbonEdge, 1.0f));
  }
  if (burstSince >= 0.0f) {
    const float u = std::clamp(burstSince / kBurstShown, 0.0f, 1.0f);
    addCone(powerStrips, burstFrom, burstAim, std::max(0.3f, t.burstReach * u),
            glm::radians(0.5f * t.burstWidth), glm::vec4(kPaleBlue * kBurstBright, 1.0f - u),
            glm::vec4(kBlueEdge, 1.0f), kConeSegments);
  }
}

// After the transparent meshes in the lit pass: the glow's meshes at where
// their nodes are now, and this frame's strips. Nothing outside play.
void VulkanRenderer::recordPowerEffects(VkCommandBuffer cmd, uint32_t currentImage,
                                        const VkViewport &viewport, const VkRect2D &scissor) {
  powerGlowDraws.clear();
  static const std::vector<EffectVertex> kNone;
  if (playing)
    for (const PowerGlow &g : powerGlows) {
      MeshModel *model = g.node ? modelManager.getModel(g.node->getModelId()) : nullptr;
      if (!model)
        continue;
      for (size_t k = 0; k < model->getMeshCount(); ++k)
        powerGlowDraws.push_back({model->getMesh(k), g.node->getGlobalTransform(),
                                  g.node->getNormalMatrix(), g.colour, g.strength, g.floor});
    }
  powerEffectsPass.upload(device, currentImage, playing ? powerStrips : kNone);
  powerEffectsPass.record(cmd, currentImage, descriptorManager.getVPSet(currentImage), viewport,
                          scissor, powerGlowDraws, static_cast<float>(playSeconds));
}

void VulkanRenderer::clearPowers() {
  powerController = {};
  powerCandidates.clear();
  powerTarget.reset();
  powerGrabbed = {};
  heldBody = {};
  heldPiece = -1;
  heldPieces.clear();
  heldAway = 0.0f;
  powerTear = {};
  powerTrail = {};

  revealSince = -1.0f;
  burstSince = -1.0f;
  powerImpacts.clear();
  shakenPieces.clear();
  shakenLink = -1;
  powerLines.clear();
  powerGlows.clear();
  powerStrips.clear();
  characterPanel.powerState = nullptr;
}

// Everything the power could take within radius of around: moving bodies
// whole if they're light enough, and the pieces of breakables (torn out of
// what stays put, or out of a chunk too heavy to lift whole).
void VulkanRenderer::gatherPowerCandidates(glm::vec3 around, float radius) {
  powerCandidates.clear();
  const PowerTuning &t = powerTuning();
  const auto wholeBody = [&](int id, GrabKind kind, PhysicsBodyId body, int piece) {
    const auto [lo, hi] = physics->bounds(body);
    powerCandidates.push_back({id, kind, body, piece, 0.5f * (lo + hi), 0.5f * glm::length(hi - lo)});
  };
  for (const PhysicsBodyFound &found : physics->bodiesInSphere(around, radius)) {
    if (found.userData & kRubbleBodyFlag)
      continue;
    if (found.userData & kBreakBodyFlag) {
      const uint64_t b = found.userData & ~kBreakBodyFlag;
      if (b >= breakBodies.size() || breakBodies[b].id.value != found.body.value)
        continue;
      const BreakBody &body = breakBodies[b];
      if (!body.fixed && liftable(physics->mass(body.id), t)) {
        wholeBody(static_cast<int>(b) * 4 + kChunkTarget, GrabKind::Lift, body.id, body.pieces.front());
        continue;
      }
      for (int p : body.pieces) {
        const BreakPiece &piece = breakPieces[p];
        if (!piece.alive)
          continue;
        float reach = 0.0f;
        for (const FractureFace &face : piece.shape->faces)
          for (const glm::vec3 &q : face.points)
            reach = std::max(reach, glm::distance(q, piece.shape->centre));
        powerCandidates.push_back({p * 4 + kPieceTarget,
                                   liftable(pieceMass(p), t) ? GrabKind::Tear : GrabKind::Strain,
                                   body.id, p, pieceCentre(p), reach});
      }
      continue;
    }
    if (found.moving && found.userData < physicsLinks.size() &&
        physicsLinks[found.userData].body.value == found.body.value)
      wholeBody(static_cast<int>(found.userData) * 4 + kSceneTarget,
                liftable(physics->mass(found.body), t) ? GrabKind::Lift : GrabKind::Strain,
                found.body, -1);
  }
}

// The held body, or the one now carrying its piece if a regroup replaced
// it (a chunk that broke in the hand: the part with the piece stays held).
bool VulkanRenderer::followHeld() {
  if (targetKind(powerGrabbed.id) == kSceneTarget) {
    const int link = targetIndex(powerGrabbed.id);
    return link < static_cast<int>(physicsLinks.size()) &&
           physicsLinks[link].body.value == heldBody.value && physics->mass(heldBody) > 0.0f;
  }
  if (heldPiece < 0 || heldPiece >= static_cast<int>(breakPieces.size()) ||
      !breakPieces[heldPiece].alive) {
    // Its piece was cut (a hit while held): follow the biggest one left.
    int biggest = -1;
    for (int p : heldPieces)
      if (p >= 0 && p < static_cast<int>(breakPieces.size()) && breakPieces[p].alive &&
          (biggest < 0 || breakPieces[p].shape->volume > breakPieces[biggest].shape->volume))
        biggest = p;
    if (biggest < 0)
      return false;
    heldPiece = biggest;
  }
  const int b = breakPieces[heldPiece].body;
  if (b < 0 || !breakBodies[b].id.isValid() || breakBodies[b].fixed)
    return false;
  if (breakBodies[b].id.value != heldBody.value) {
    if (!liftable(physics->mass(breakBodies[b].id), powerTuning()))
      return false;
    heldBody = breakBodies[b].id;
  }
  heldPieces = breakBodies[b].pieces;
  return true;
}

// The piece's unbroken links and ground links, and when each gives way:
// weakest first, a ground link as strong as its face's area times its
// object's strength, like the links of its cut.
void VulkanRenderer::startTear(int piece) {
  powerTear = {};
  std::vector<float> strengths;
  for (int link : pieceLinks[piece])
    if (!breakLinks[link].broken) {
      powerTear.links.push_back(link);
      strengths.push_back(breakLinks[link].strength);
    }
  if (piece < static_cast<int>(pieceGroundLinks.size()))
    for (int g : pieceGroundLinks[piece])
      if (!groundLinks[g].broken) {
        powerTear.grounds.push_back(g);
        strengths.push_back(groundLinks[g].face.area * breakables[breakPieces[piece].object].strength);
      }
  const std::vector<float> times = tearTimes(strengths, powerTuning());
  powerTear.linkTimes.assign(times.begin(), times.begin() + powerTear.links.size());
  powerTear.groundTimes.assign(times.begin() + powerTear.links.size(), times.end());
  powerTear.total = times.empty() ? 0.0f : *std::max_element(times.begin(), times.end());
}

bool VulkanRenderer::advanceTear(float seconds) {
  const int piece = powerGrabbed.piece;
  powerTear.seconds += seconds;
  bool broke = false;
  for (size_t i = 0; i < powerTear.links.size(); ++i) {
    BreakLink &link = breakLinks[powerTear.links[i]];
    if (!link.broken && powerTear.linkTimes[i] <= powerTear.seconds) {
      link.broken = true;
      broke = true;
    }
  }
  for (size_t i = 0; i < powerTear.grounds.size(); ++i) {
    GroundLink &ground = groundLinks[powerTear.grounds[i]];
    if (!ground.broken && powerTear.groundTimes[i] <= powerTear.seconds) {
      ground.broken = true;
      broke = true;
    }
  }
  // Its time is up: whatever links it still has go too (a neighbour cut
  // while it tore links to it afresh, and those weren't on the schedule).
  if (powerTear.seconds >= powerTear.total) {
    for (int l : pieceLinks[piece])
      if (!breakLinks[l].broken) {
        breakLinks[l].broken = true;
        broke = true;
      }
    if (piece < static_cast<int>(pieceGroundLinks.size()))
      for (int g : pieceGroundLinks[piece])
        if (!groundLinks[g].broken) {
          groundLinks[g].broken = true;
          broke = true;
        }
  }
  if (broke) {
    if (piece < static_cast<int>(pieceGroundLinks.size())) {
      bool holds = false;
      for (int g : pieceGroundLinks[piece])
        holds = holds || !groundLinks[g].broken;
      breakPieces[piece].anchor = holds;
    }
    stressFirst = false; // damage: from now on what's overloaded breaks
    regroupBreakables({});
  }
  // Loose, and light enough with whatever came out with it: held from here.
  const int b = breakPieces[piece].body;
  if (b < 0 || !breakBodies[b].id.isValid() || breakBodies[b].fixed ||
      !liftable(physics->mass(breakBodies[b].id), powerTuning())) {
    // Free of everything and still too heavy (Jolt's mass a hair over the
    // cap): nothing more a tear can do.
    powerTear.done = powerTear.seconds >= powerTear.total;
    return false;
  }
  heldBody = breakBodies[b].id;
  heldPiece = piece;
  heldPieces = breakBodies[b].pieces;
  heldFrom = heldCentre();
  heldSeconds = 0.0f;
  heldAxis = randomAxis();
  heldAway = 0.0f;
  return true;
}

void VulkanRenderer::updatePowers(float seconds) {
  if (!physics || !roster || !playerController || !playerController->body() || playedCharacter < 0)
    return;
  const PowerTuning &t = powerTuning();
  const ThirdPersonCamera &view = playerController->camera();
  const glm::vec3 eye = view.position(), aim = view.forward();
  const CharacterMovement &me = roster->body(characters[playedCharacter].body);
  const glm::vec3 chest = me.position() + glm::vec3(0.0f, 0.75f * me.height(), 0.0f);

  if (playerControls.reveal) {
    revealSince = 0.0f;
    revealFrom = me.position();
  } else if (revealSince >= 0.0f) {
    revealSince += seconds;
    if (revealSince > t.revealReach / t.revealSpeed + t.revealSeconds)
      revealSince = -1.0f;
  }
  if (burstSince >= 0.0f && (burstSince += seconds) > kBurstShown)
    burstSince = -1.0f;
  if (powerTrail.body.isValid()) {
    powerTrail.seconds += seconds;
    if (powerTrail.seconds > kTrailSeconds || physics->mass(powerTrail.body) <= 0.0f)
      powerTrail = {};
    else
      powerTrail.points.push_back(bodyCentre(powerTrail.body));
  }

  PowerInput in;
  in.aim = playerControls.aim;
  in.cast = playerControls.cast;
  in.burst = playerControls.burst;
  if (heldBody.isValid() && (!followHeld() || heldAway >= t.lostSeconds))
    in.lost = true;
  const PowerState before = powerController.state();
  if (before == PowerState::Tearing) {
    if (!breakPieces[powerGrabbed.piece].alive)
      in.lost = true;
    else
      in.torn = advanceTear(seconds);
    in.lost = in.lost || powerTear.done;
  }
  if (before == PowerState::Straining && powerGrabbed.body.isValid() &&
      powerGrabbed.piece >= 0 && !breakPieces[powerGrabbed.piece].alive)
    in.lost = true;

  // What's near (for the crosshair and the reveal), and what the crosshair
  // is on: nearest its line, with nothing in the way.
  powerTarget.reset();
  const bool picking = in.aim && !powerController.holding();
  if (picking || revealSince >= 0.0f)
    gatherPowerCandidates(chest, std::max(t.pickReach, t.revealReach) + 2.0f);
  else
    powerCandidates.clear();
  // What the crosshair's own ray hits comes first, and on a structure the
  // piece it hits; the nearest to its line only when it misses.
  if (picking) {
    if (const std::optional<PhysicsRayHit> hit = physics->castRay(eye, aim, t.pickReach)) {
      const glm::vec3 point = eye + aim * (hit->distance + 0.01f);
      int on = -1;
      if (hit->userData & kBreakBodyFlag) {
        const uint64_t b = hit->userData & ~kBreakBodyFlag;
        if (b < breakBodies.size() && breakBodies[b].id.value == hit->body.value)
          on = pieceAt(breakBodies[b], point);
      }
      for (const PowerTarget &c : powerCandidates)
        if (c.body.value == hit->body.value &&
            (targetKind(c.id) != kPieceTarget || c.piece == on)) {
          powerTarget = c;
          break;
        }
    }
  }
  if (picking && !powerTarget) {
    std::vector<PickCandidate> picks;
    picks.reserve(powerCandidates.size());
    for (const PowerTarget &c : powerCandidates)
      picks.push_back({c.centre, c.radius});
    const std::vector<int> order = pickOrder(picks, eye, aim, t);
    for (int k = 0; k < static_cast<int>(order.size()) && k < kSightChecks; ++k) {
      const PowerTarget &c = powerCandidates[order[k]];
      const glm::vec3 to = c.centre - eye;
      const float distance = glm::length(to);
      if (distance < 1e-3f)
        continue;
      const std::optional<PhysicsRayHit> hit = physics->castRay(eye, to, distance + c.radius);
      if (hit && hit->body.value == c.body.value) {
        powerTarget = c;
        break;
      }
    }
  }
  if (powerTarget) {
    in.target = powerTarget->id;
    in.targetKind = powerTarget->kind;
  }

  const PowerEvents e = powerController.update(in, seconds, t);
  if (e.grabbed >= 0 && powerTarget && powerTarget->id == e.grabbed) {
    powerGrabbed = *powerTarget;
    spdlog::info("[power] grabbed {} ({:.0f} kg, {:.1f} m away): {}",
                 targetKind(powerGrabbed.id) == kPieceTarget   ? "a piece"
                 : targetKind(powerGrabbed.id) == kChunkTarget ? "a chunk"
                                                               : "a body",
                 targetKind(powerGrabbed.id) == kPieceTarget ? pieceMass(powerGrabbed.piece)
                                                             : physics->mass(powerGrabbed.body),
                 glm::distance(powerGrabbed.centre, eye) ,
                 powerGrabbed.kind == GrabKind::Lift   ? "lifting"
                 : powerGrabbed.kind == GrabKind::Tear ? "tearing it out"
                                                       : "too heavy");
    heldDistance = t.holdDistance;
    heldAway = 0.0f;
    if (powerGrabbed.kind == GrabKind::Lift) {
      heldBody = powerGrabbed.body;
      heldPiece = powerGrabbed.piece;
      heldPieces.clear();
      if (targetKind(powerGrabbed.id) == kChunkTarget)
        heldPieces = breakBodies[targetIndex(powerGrabbed.id)].pieces;
      heldFrom = heldCentre();
      heldSeconds = 0.0f;
      heldAxis = randomAxis();
    } else if (powerGrabbed.kind == GrabKind::Tear) {
      startTear(powerGrabbed.piece);
    }
  }
  if (in.torn && powerController.state() == PowerState::Lifting)
    spdlog::info("[power] torn out after {:.2f} s ({:.0f} kg with what came with it)",
                 powerTear.seconds, physics->mass(heldBody));
  if (e.dropped)
    spdlog::info("[power] {}", powerTear.done ? "still too heavy once torn free: let go"
                               : in.lost      ? "lost it"
                                              : "dropped");
  if (e.thrown)
    throwHeld();
  if (e.dropped || powerController.state() == PowerState::Idle) {
    heldBody = {};
    heldPiece = -1;
    heldPieces.clear();
  }
  if (powerController.state() != PowerState::Tearing)
    powerTear = {};
  if (e.burst)
    burstPower();

  // The wheel brings what's held in or pushes it out (it doesn't zoom then).
  if (powerController.holding())
    heldDistance =
        std::clamp(heldDistance + playerControls.wheel * kWheelStep, t.holdNearest, t.holdFarthest);
  const float characterDistance = std::max(0.0f, glm::dot(chest - eye, aim));
  holdTarget = holdPoint(eye, aim, characterDistance, heldDistance, t.holdRaise);
  characterPanel.powerState = powerStateName(powerController.state());
}

std::optional<glm::vec3> VulkanRenderer::powerReach() const {
  const bool pieceAlive = powerGrabbed.piece >= 0 && breakPieces[powerGrabbed.piece].alive;
  switch (powerController.state()) {
  case PowerState::Lifting:
  case PowerState::Holding:
  case PowerState::Throwing: // until it leaves the hand
    if (heldBody.isValid())
      return heldCentre();
    break;
  case PowerState::Tearing:
    if (pieceAlive)
      return pieceCentre(powerGrabbed.piece);
    break;
  case PowerState::Straining:
    return pieceAlive ? pieceCentre(powerGrabbed.piece) : bodyCentre(powerGrabbed.body);
  default:
    break;
  }
  return std::nullopt;
}

void VulkanRenderer::holdPowers(float stepSeconds) {
  const PowerState state = powerController.state();
  if (!heldBody.isValid() ||
      (state != PowerState::Lifting && state != PowerState::Holding && state != PowerState::Throwing))
    return;
  const float mass = physics->mass(heldBody);
  if (mass <= 0.0f)
    return;
  const PowerTuning &t = powerTuning();
  heldSeconds += stepSeconds;
  const glm::vec3 position = heldCentre();
  const glm::vec3 wanted = liftPoint(heldFrom, holdTarget, heldSeconds, t);
  // Short of anything that doesn't move on its way there, and along it, or a
  // hold point behind a wall rams what's held into the wall every step. What
  // moves is shoved aside.
  const glm::vec3 target =
      slideHoldPoint(position, wanted, [&](glm::vec3 start, glm::vec3 move) -> std::optional<HoldContact> {
        const std::optional<PhysicsSweepHit> hit = physics->sweepBody(heldBody, move, start - position);
        if (!hit)
          return std::nullopt;
        return HoldContact{hit->fraction, hit->normal};
      });
  physics->setLinearVelocity(heldBody, holdVelocity(position, physics->linearVelocity(heldBody), target,
                                                    mass, physics->gravity(), stepSeconds, t));
  physics->setAngularVelocity(heldBody,
                              holdSpin(physics->angularVelocity(heldBody), heldAxis, stepSeconds, t));
  // Lost by where it's wanted: stuck behind something long enough, it's let go.
  heldAway = glm::distance(position, wanted) > t.lostDistance ? heldAway + stepSeconds : 0.0f;
}

// At what the crosshair is on (past the held thing), on the arc that lands
// there, with a tumble.
void VulkanRenderer::throwHeld() {
  if (!heldBody.isValid() || !followHeld())
    return;
  const PowerTuning &t = powerTuning();
  const ThirdPersonCamera &view = playerController->camera();
  const glm::vec3 eye = view.position(), aim = view.forward();
  const std::optional<PhysicsRayHit> hit = physics->castRay(eye, aim, 500.0f, {heldBody});
  // No further than its reach, or anything far would be lobbed at 45 degrees
  // while the open sky is thrown flat.
  const glm::vec3 to = eye + aim * std::min(hit ? hit->distance : t.throwReach, t.throwReach);
  const glm::vec3 from = heldCentre();
  const float mass = physics->mass(heldBody);
  physics->setLinearVelocity(heldBody, throwVelocity(from, to, throwSpeed(mass, t), physics->gravity()));
  spdlog::info("[power] thrown at {:.1f} m/s toward a point {:.1f} m away", throwSpeed(mass, t),
               glm::distance(from, to));
  physics->setAngularVelocity(heldBody,
                              randomAxis() * t.throwTumble * (0.5f + 0.5f * hash01(++powerDraws)));
  powerTrail = {heldBody, {from}, 0.0f};
  heldBody = {};
  heldPiece = -1;
  heldPieces.clear();
}

// Everything in the cone thrown away from the hand, and every breakable in
// it hit at its piece the burst reaches hardest, through handleImpacts like
// any other hit (links break, pieces are cut, chunks fly).
void VulkanRenderer::burstPower() {
  const PowerTuning &t = powerTuning();
  const glm::vec3 origin = handPosition, aim = playerController->camera().forward();
  burstFrom = origin;
  burstAim = aim;
  burstSince = 0.0f;
  int pushed = 0, hit = 0;
  const auto kick = [&](PhysicsBodyId body) {
    const float mass = physics->mass(body);
    if (mass <= 0.0f)
      return;
    const glm::vec3 k = burstKick(origin, aim, bodyCentre(body), mass, t);
    if (glm::length(k) <= 0.0f)
      return;
    ++pushed;
    physics->setLinearVelocity(body, physics->linearVelocity(body) + k);
    physics->setAngularVelocity(body, physics->angularVelocity(body) + randomAxis() * 0.3f * glm::length(k));
  };
  for (const PhysicsBodyFound &found : physics->bodiesInSphere(origin, t.burstReach)) {
    if (found.userData & kRubbleBodyFlag)
      continue;
    if (found.userData & kBreakBodyFlag) {
      const uint64_t b = found.userData & ~kBreakBodyFlag;
      if (b >= breakBodies.size() || breakBodies[b].id.value != found.body.value)
        continue;
      float hardest = 0.0f;
      int at = -1;
      for (int p : breakBodies[b].pieces) {
        if (!breakPieces[p].alive)
          continue;
        const float f = burstFalloff(origin, aim, pieceCentre(p), t);
        if (f > hardest) {
          hardest = f;
          at = p;
        }
      }
      if (at >= 0) {
        const glm::vec3 point = pieceCentre(at);
        const glm::vec3 d = point - origin;
        const glm::vec3 away = glm::length(d) > 1e-4f ? glm::normalize(d) : aim;
        PhysicsImpact impact;
        impact.body = found.body;
        impact.userData = found.userData;
        impact.point = point;
        impact.normal = -away; // out of the body, back toward the hand
        impact.impulse = t.burstImpulse * hardest;
        impact.otherVelocity = away * (t.burstSpeed * hardest);
        powerImpacts.push_back(impact);
        ++hit;
      }
      if (!breakBodies[b].fixed)
        kick(found.body);
      continue;
    }
    if (found.moving && found.userData < physicsLinks.size() &&
        physicsLinks[found.userData].body.value == found.body.value)
      kick(found.body);
  }
  spdlog::info("[power] burst: {} bodies pushed, {} breakables hit", pushed, hit);
}

void VulkanRenderer::shakePowers() {
  // Last frame's shaken nodes back where their bodies are.
  for (int p : shakenPieces)
    if (p < static_cast<int>(breakPieces.size()) && breakPieces[p].alive) {
      const PhysicsPose world = pieceWorld(breakPieces[p]);
      breakPieces[p].node->setPose(world.position, world.rotation);
    }
  if (shakenLink >= 0 && shakenLink < static_cast<int>(physicsLinks.size()) &&
      physicsLinks[shakenLink].node && physicsLinks[shakenLink].body.isValid()) {
    const PhysicsPose pose = physics->pose(physicsLinks[shakenLink].body);
    physicsLinks[shakenLink].node->setPose(pose.position, pose.rotation);
  }
  shakenPieces.clear();
  shakenLink = -1;

  const PowerTuning &t = powerTuning();
  const PowerState state = powerController.state();
  float amount = 0.0f;
  if (state == PowerState::Tearing && powerTear.total > 0.0f)
    amount = t.tearShake * std::clamp(powerTear.seconds / powerTear.total, 0.2f, 1.0f);
  else if (state == PowerState::Straining)
    amount = t.tearShake;
  if (amount <= 0.0f)
    return;
  glm::vec3 offset;
  for (int k = 0; k < 3; ++k)
    offset[k] = amount * (2.0f * hash01(++powerDraws) - 1.0f);
  if (targetKind(powerGrabbed.id) == kPieceTarget && powerGrabbed.piece >= 0 &&
      breakPieces[powerGrabbed.piece].alive) {
    const PhysicsPose world = pieceWorld(breakPieces[powerGrabbed.piece]);
    breakPieces[powerGrabbed.piece].node->setPose(world.position + offset, world.rotation);
    shakenPieces.push_back(powerGrabbed.piece);
  } else if (targetKind(powerGrabbed.id) == kSceneTarget) {
    const int link = targetIndex(powerGrabbed.id);
    if (link < static_cast<int>(physicsLinks.size()) && physicsLinks[link].node &&
        physicsLinks[link].body.value == powerGrabbed.body.value) {
      const PhysicsPose pose = physics->pose(physicsLinks[link].body);
      physicsLinks[link].node->setPose(pose.position + offset, pose.rotation);
      shakenLink = link;
    }
  }
}

void VulkanRenderer::drawPowerLines() {
  buildPowerEffects();
  powerLines.clear();
  if (!physics || !playerController || !playerController->body() || !characterPanel.powerLines)
    return;
  const PowerTuning &t = powerTuning();
  const float time = static_cast<float>(playSeconds);
  const auto outline = [&](const PowerTarget &c, uint32_t colour) {
    if (targetKind(c.id) == kPieceTarget && c.piece >= 0 && breakPieces[c.piece].alive) {
      const BreakPiece &piece = breakPieces[c.piece];
      const PhysicsPose world = pieceWorld(piece);
      for (const FractureFace &face : piece.shape->faces)
        for (size_t i = 0; i < face.points.size(); ++i)
          addLine(powerLines, world.position + world.rotation * face.points[i],
                  world.position + world.rotation * face.points[(i + 1) % face.points.size()], colour);
    } else if (c.body.isValid()) {
      const auto [lo, hi] = physics->bounds(c.body);
      addBox(powerLines, lo, hi, colour);
    }
  };

  // The reveal: its pulse going out along the ground, and what it passed
  // glowing.
  if (revealSince >= 0.0f) {
    const float out = t.revealReach / t.revealSpeed;
    const float fade = 1.0f - std::clamp((revealSince - out) / kRingFade, 0.0f, 1.0f);
    if (fade > 0.0f)
      addCircle(powerLines, revealFrom + glm::vec3(0.0f, 0.05f, 0.0f),
                std::min(revealSince * t.revealSpeed, t.revealReach), lineColour(kRevealRgb, fade));
    for (const PowerTarget &c : powerCandidates) {
      const float glow = revealGlow(glm::distance(c.centre, revealFrom), revealSince, t);
      if (glow > 0.02f)
        outline(c, lineColour(kRevealRgb, glow));
    }
  }
  if (powerTarget)
    outline(*powerTarget, lineColour(powerTarget->kind == GrabKind::Lift   ? kTargetRgb
                                     : powerTarget->kind == GrabKind::Tear ? kTearRgb
                                                                           : kStrainRgb));
  // What's held and the light from the hand to it.
  if (const std::optional<glm::vec3> end = powerReach()) {
    const PowerState state = powerController.state();
    const uint32_t rgb = state == PowerState::Tearing     ? kTearRgb
                         : state == PowerState::Straining ? kStrainRgb
                                                          : kHoldRgb;
    addRibbon(powerLines, handPosition, *end, time, lineColour(rgb));
    if (state == PowerState::Tearing || state == PowerState::Straining) {
      outline(powerGrabbed, lineColour(rgb));
    } else {
      const auto [lo, hi] = physics->bounds(heldBody);
      addBox(powerLines, lo, hi, lineColour(kHoldRgb, 0.7f));
    }
  }
  // The burst's cone opening from the hand.
  if (burstSince >= 0.0f) {
    const float u = std::clamp(burstSince / kBurstShown, 0.0f, 1.0f);
    const float length = t.burstReach * u;
    const float half = glm::radians(0.5f * t.burstWidth);
    const glm::vec3 side = glm::normalize(std::fabs(burstAim.y) > 0.9f
                                              ? glm::cross(burstAim, glm::vec3(1, 0, 0))
                                              : glm::cross(burstAim, glm::vec3(0, 1, 0)));
    const glm::vec3 up = glm::cross(side, burstAim);
    const uint32_t colour = lineColour(kBurstRgb, 1.0f - u);
    glm::vec3 last{};
    for (int i = 0; i <= 16; ++i) {
      const float a = glm::two_pi<float>() * i / 16;
      const glm::vec3 dir = std::cos(half) * burstAim +
                            std::sin(half) * (std::cos(a) * side + std::sin(a) * up);
      const glm::vec3 end = burstFrom + dir * length;
      if (i % 2 == 0)
        addLine(powerLines, burstFrom, end, colour);
      if (i > 0)
        addLine(powerLines, last, end, colour);
      last = end;
    }
  }
  // The last throw's trail.
  for (size_t i = 1; i < powerTrail.points.size(); ++i)
    addLine(powerLines, powerTrail.points[i - 1], powerTrail.points[i],
            lineColour(kHoldRgb, 1.0f - powerTrail.seconds / kTrailSeconds));
}
