// VulkanRenderer's explosions (step 1.7; the grenades, charges and rockets
// went with the shooter, tag shooter-final). The blast (Blast.h) pushes
// bodies, hits breakable pieces through the destruction's own impacts and
// hurts people close by; a cased charge's fragments are worked out for
// people (every fragment of the fitted spread, as expected hits on each
// body part with cover); a fireball, a light, dust, smoke and a boom.

#include "VulkanRenderer.h"

#include "Particles.h"
#include "ScenePhysics.h"

#include <glm/gtc/quaternion.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace {

// A blast's push: the drag coefficient of a body (est., between a front
// face's 1.0 and a beam's 1.5); behind something solid, a quarter (est.:
// blast bends round corners but loses most of its push).
constexpr float kBlastDrag = 1.2f, kShadowed = 0.25f;
// Each fragment hit takes at least this many points (est.: a casualty hit;
// 1.9 brings real wounding), and blast past lung damage this many.
constexpr float kFragmentHitPoints = 20.0f, kLungDamagePoints = 60.0f;
// The fireball: at most 3.6 W^(1/3) m across, reached after about
// 7 W^(1/3) ms (one Southampton measurement, scaled); its light (est.).
constexpr float kFireballWidth = 3.6f, kBlastLightSeconds = 0.06f, kBlastLightPower = 3000.0f;
// The smoke cloud's puffs (est.).
constexpr int kCloudPuffs = 60;

float dice(uint32_t &state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return static_cast<float>(state % 100000u) / 100000.0f;
}

} // namespace

void VulkanRenderer::updateBlasts(float seconds) {
  if (blastLight > 0.0f)
    blastLight -= seconds;
  // A blast lights its surroundings for a few frames (no shadow).
  sceneUbo.flashColour = glm::vec4(0.0f);
  if (blastLight > 0.0f) {
    sceneUbo.flashPosition = glm::vec4(blastLightAt, 1.0f);
    sceneUbo.flashColour = glm::vec4(1.0f, 0.7f, 0.4f, blastLightPower);
  }
}

void VulkanRenderer::explode(glm::vec3 at, const CasedCharge &charge, int owner) {
  const float tnt = charge.charge * tntEquivalent(charge.explosive, 1e5f, false);
  const float cube = std::cbrt(tnt);
  spdlog::info("Explosion at {:.1f}, {:.1f}, {:.1f}: {:.0f} g of TNT", at.x, at.y, at.z, tnt * 1000.0f);
  // Blocked by something solid on the way?
  auto shadowed = [&](glm::vec3 to, PhysicsBodyId self) {
    const glm::vec3 d = to - at;
    const float length = glm::length(d);
    if (length < 1e-3f)
      return false;
    const std::optional<PhysicsRayHit> hit = physics->castRay(at, d / length, length - 0.05f, {self});
    return hit.has_value();
  };

  // Bodies that move: the blast's impulse as a push, Cd A I / m, out from
  // the blast; less behind cover. Out to where it's down to a fraction of a
  // pascal-second (a scaled distance of 20).
  const float pushReach = 20.0f * cube;
  for (const PhysicsBodyFound &found : physics->bodiesInSphere(at, pushReach)) {
    const float mass = physics->mass(found.body);
    if (!(mass > 0.0f) || (found.userData & kBreakBodyFlag))
      continue;
    const glm::vec3 where = physics->pose(found.body).position;
    const float distance = std::max(glm::distance(where, at), 0.1f);
    const float impulse = positiveImpulse(scaledDistance(distance, tnt), tnt);
    // The area facing it: a box's average face, else from its mass (est.
    // 600 kg/m^3).
    float area = std::pow(mass / 600.0f, 2.0f / 3.0f);
    if (found.userData < physicsLinks.size() && physicsLinks[found.userData].node) {
      const SceneEntryRef ref = physicsLinks[found.userData].node->entry();
      if (ref.kind == SceneEntryRef::Kind::Model && ref.index < currentScene.models.size()) {
        const SceneModelEntry &e = currentScene.models[ref.index];
        if (e.shape == PrimitiveShape::Box)
          area = (e.size.x * e.size.y + e.size.y * e.size.z + e.size.z * e.size.x) / 3.0f;
      }
    }
    float push = blastPush(kBlastDrag, area, impulse, mass);
    if (shadowed(where, found.body))
      push *= kShadowed;
    const glm::vec3 out = glm::normalize(where - at + glm::vec3(0.0f, 0.05f, 0.0f));
    physics->setLinearVelocity(found.body, physics->linearVelocity(found.body) + out * push);
    physics->setAngularVelocity(found.body, physics->angularVelocity(found.body) +
                                                glm::cross(out, glm::vec3(0.0f, 1.0f, 0.0f)) * (0.5f * push));
  }

  // Breakable pieces facing it: the reflected impulse over each piece's
  // face, as a hit the destruction takes (links break, pieces are cut).
  if (breakGraphReady)
    for (size_t b = 0; b < breakBodies.size(); ++b) {
      const BreakBody &body = breakBodies[b];
      if (!body.id.isValid())
        continue;
      for (int p : body.pieces) {
        const BreakPiece &piece = breakPieces[p];
        if (!piece.alive)
          continue;
        const PhysicsPose world = pieceWorld(piece);
        const glm::vec3 centre = world.position + world.rotation * piece.shape->centre;
        // From the blast to the piece's near side (half its size short of its
        // middle): a charge set on a wall is at its face, not half a metre
        // off.
        const float size = std::cbrt(piece.shape->volume);
        const float toCentre = glm::distance(centre, at);
        const float distance = std::max(toCentre - 0.5f * size, 0.05f);
        if (distance > pushReach)
          continue;
        const float z = scaledDistance(distance, tnt);
        const float pressure = peakOverpressure(z);
        const float impulse = positiveImpulse(z, tnt) * reflectionFactor(pressure) * size * size;
        if (impulse < 1.0f || shadowed(centre, body.id))
          continue;
        const glm::vec3 back = toCentre > 1e-4f ? (at - centre) / toCentre : glm::vec3(0.0f, 1.0f, 0.0f);
        PhysicsImpact hit;
        hit.body = body.id;
        hit.userData = kBreakBodyFlag | static_cast<uint64_t>(b);
        hit.point = centre + back * std::min(0.5f * size, toCentre);
        hit.normal = back; // the push goes against it
        hit.impulse = impulse;
        blastImpacts.push_back(hit);
      }
    }

  // People: the blast itself close by, and every fragment of the spread as
  // expected hits on each body part they can see (none without a case).
  const bool cased = charge.caseMass > 0.0f;
  const FragmentSpread spread = cased ? fitFragments(charge) : FragmentSpread{};
  for (int i = 0; i < static_cast<int>(characters.size()); ++i) {
    LiveCharacter &c = characters[i];
    if (c.capsules.empty() || c.health.dead())
      continue;
    // At his chest (the capsules start with the head).
    const HitCapsule *trunk = &c.capsules.front();
    for (const HitCapsule &part : c.capsules)
      if (part.zone == HitZone::Chest)
        trunk = &part;
    const glm::vec3 middle = 0.5f * (trunk->a + trunk->b);
    const float pressure = peakOverpressure(scaledDistance(std::max(glm::distance(middle, at), 0.1f), tnt));
    float points = 0.0f;
    if (pressure > kBlastLethal)
      points += c.health.points;
    else if (pressure > kLungDamage)
      points += kLungDamagePoints;
    if (pressure > kEardrumRupture)
      spdlog::info("Character {}: {:.0f} kPa of blast, eardrums burst", i, pressure / 1000.0f);
    int hits = 0;
    for (const HitCapsule &part : c.capsules) {
      if (!cased)
        break;
      const glm::vec3 mid = 0.5f * (part.a + part.b);
      const float distance = std::max(glm::distance(mid, at), 0.2f);
      if (shadowed(mid, PhysicsBodyId{}))
        continue;
      const float area = 2.0f * part.radius * glm::distance(part.a, part.b) + 3.14159f * part.radius * part.radius;
      const float expected = static_cast<float>(spread.count) * area / (4.0f * 3.14159f * distance * distance);
      // Poisson: how many of them hit this time.
      float l = std::exp(-expected), product = dice(fireDice);
      int k = 0;
      while (product > l && k < 50) {
        product *= dice(fireDice);
        ++k;
      }
      const float v = fragmentSpeed(spread.speed, spread.mass, distance);
      const float energy = 0.5f * spread.mass * v * v;
      for (int h = 0; h < k; ++h)
        points += std::max(kFragmentHitPoints, woundPoints(part.zone, energy));
      hits += k;
    }
    if (points <= 0.0f)
      continue;
    c.hitThisFrame = true;
    if (c.health.wound(points)) {
      const glm::vec3 from = glm::length(middle - at) > 1e-4f ? glm::normalize(middle - at) : c.facing;
      c.deathClip = c.crouched                         ? ClipSlot::DeathCrouched
                    : glm::dot(from, c.facing) > 0.0f ? ClipSlot::DeathBack
                                                       : ClipSlot::DeathFront;
      spdlog::info("Character {} killed by the blast ({} fragment hits, {:.0f} kPa)", i, hits,
                   pressure / 1000.0f);
    } else {
      spdlog::info("Character {} wounded by the blast: {} fragment hits, {:.0f} left", i, hits,
                   c.health.points);
    }
  }

  // What you see and hear: the fireball for a frame or two, its light,
  // dust thrown along the ground, grit, a cloud of the explosive's smoke
  // (TNT's near black, C-4's grey) spreading and rising; the boom; the
  // camera shaken by the pressure where it is.
  static ParticleRandom random(0x7a3d1e55u);
  ParticleSpawn fire;
  fire.position = at;
  fire.colour = glm::vec3(30.0f, 14.0f, 4.0f);
  fire.size = 0.35f * kFireballWidth * cube;
  fire.life = 0.09f;
  fire.drag = 12.0f;
  emitInCone(particleSpawns, at, glm::vec3(0.0f, 1.0f, 0.0f), 3.14159f, 160, 4.0f, 18.0f, fire, random);
  ParticleSpawn dust;
  dust.colour = glm::vec3(0.4f, 0.37f, 0.32f);
  dust.opacity = 0.7f; // laid over, not glowing (est.)
  dust.size = 0.9f;
  dust.spread = 4.0f;
  dust.life = 3.5f;
  dust.drag = 2.5f;
  dust.swirl = 1.0f;
  dust.gravity = -0.02f; // it rises a little
  for (int k = 0; k < 60; ++k) {
    const float a = 6.2831853f * static_cast<float>(k) / 60.0f;
    dust.position = at + glm::vec3(0.0f, 0.1f, 0.0f);
    dust.velocity = glm::vec3(std::cos(a), 0.15f, std::sin(a)) * (6.0f + 4.0f * random.next());
    particleSpawns.push_back(dust);
  }
  ParticleSpawn grit;
  grit.colour = glm::vec3(0.3f, 0.27f, 0.22f);
  grit.opacity = 0.9f;
  grit.size = 0.03f;
  grit.life = 1.5f;
  grit.drag = 0.5f;
  grit.gravity = 1.0f;
  emitInCone(particleSpawns, at, glm::vec3(0.0f, 1.0f, 0.0f), glm::radians(70.0f), 120, 4.0f, 14.0f, grit, random);
  // The cloud billows out of where the fireball was: puffs through its
  // volume, thrown out and up, some lighter (the sky on top) and some darker,
  // a third of it dust off what it hit (est.).
  const float fireball = 0.5f * kFireballWidth * cube; // m, its radius
  const glm::vec3 dusty(0.4f, 0.37f, 0.32f);
  ParticleSpawn cloud;
  cloud.opacity = 0.5f;
  cloud.spread = 2.5f;
  cloud.life = 8.0f;
  cloud.drag = 1.5f;
  cloud.swirl = 0.8f;
  cloud.gravity = -0.05f;
  for (int k = 0; k < kCloudPuffs; ++k) {
    const glm::vec3 out = random.onSphere();
    const glm::vec3 up(out.x, std::abs(out.y) * 0.8f + 0.2f, out.z);
    cloud.position = at + out * (fireball * 0.4f * random.next());
    cloud.velocity = glm::normalize(up) * (1.0f + 3.0f * random.next());
    cloud.size = fireball * (0.6f + 0.5f * random.next());
    const float shade = 0.75f + 0.5f * random.next() + 0.25f * up.y;
    cloud.colour = glm::mix(glm::vec3(smokeShade(charge.explosive)), dusty, 0.3f) * shade;
    particleSpawns.push_back(cloud);
  }
  blastLight = kBlastLightSeconds;
  blastLightAt = at + glm::vec3(0.0f, 0.5f, 0.0f);
  blastLightPower = kBlastLightPower * cube;
  if (audio)
    if (const SoundClip *boom = audio->load("Resources/Sounds/guns/akm_shot_near.ogg")) {
      SoundParams params;
      params.position = at;
      params.loudness = 180.0f; // dB at 1 m (est.)
      params.pitch = 0.4f;      // a deeper, longer report (a stand-in until a recorded blast)
      audio->play(boom, params);
    }
  if (playedCharacter >= 0 && playedCharacter < static_cast<int>(characters.size())) {
    LiveCharacter &player = characters[playedCharacter];
    const float pressure = peakOverpressure(
        scaledDistance(std::max(glm::distance(glm::vec3(sceneUbo.cameraPosition), at), 0.1f), tnt));
    const float strength = std::clamp(pressure / 30e3f, 0.0f, 1.0f); // est.
    player.shake.kick({0.0f, glm::radians(3.0f) * strength, glm::radians(2.0f) * strength * (2.0f * dice(fireDice) - 1.0f)},
                      0.05f);
  }
}

void VulkanRenderer::clearBlasts() {
  blastImpacts.clear();
  blastLight = 0.0f;
}
