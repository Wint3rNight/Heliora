// VulkanRenderer's rockets (step 1.7): a launcher's shot leaves its muzzle
// as a rocket that flies on every frame (Rocket.h: the booster's kick, the
// motor from 10 m to 500 m, drag, the wind, gravity), seen as its model
// with a flame, its light and a smoke trail while the motor burns. What it hits takes
// its warhead: the shaped charge's jet through what's in line, then the
// blast. If it hits nothing it blows itself up when its fuze runs out.

#include "VulkanRenderer.h"

#include "EffectGeometry.h"
#include "Particles.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace {

// A jet breaks up in the air: it goes at most this far past the hit (est.).
constexpr float kJetReach = 2.0f; // m
// The smoke trail while the motor burns: a puff every so far, starting this
// wide and spreading to kTrailSpread times that, lasting kTrailLife (all
// est., against footage: a thin white line that widens and hangs for
// seconds). Puffs under a third of their width apart make a solid line;
// half their width apart they read as beads.
constexpr float kTrailEvery = 0.1f;   // m
constexpr float kTrailWidth = 0.35f;  // m
constexpr float kTrailSpread = 3.5f;
constexpr float kTrailLife = 6.0f;    // s
// Its flame: a cone this long out of the back (est.), and so many
// short-lived glows a second behind it.
constexpr float kFlameLength = 0.7f;  // m
constexpr float kFlamesPerSecond = 150.0f;
// The back blast's smoke and the dust it raises off the ground behind, if
// the ground is that close under the blast (est.).
constexpr int kBackSmoke = 90;
constexpr int kBackDust = 50;
constexpr float kBackDustReach = 3.0f; // m
const glm::vec3 kGravity(0.0f, -9.81f, 0.0f);

// A frame along dir, its up as near the world's as it gets.
glm::quat facingAlong(glm::vec3 dir) {
  const glm::vec3 z = glm::normalize(dir);
  glm::vec3 x = glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), z);
  x = glm::length(x) > 1e-4f ? glm::normalize(x) : glm::vec3(1.0f, 0.0f, 0.0f);
  return glm::quat_cast(glm::mat3(x, glm::cross(z, x), z));
}

} // namespace

void VulkanRenderer::launchRocket(int index, CarriedGun &gun, const Shot &shot) {
  const WeaponAsset &asset = *gun.asset;
  if (!asset.spec.rocket)
    return;
  const RocketSpec &spec = *asset.spec.rocket;
  auto socketAt = [&](const char *name, glm::vec3 otherwise) {
    const std::optional<WeaponSocket> s = asset.parts.socket(name);
    return glm::vec3(gun.aimedFrom * glm::vec4(s ? s->position : otherwise, 1.0f));
  };
  const glm::vec3 muzzle = socketAt("muzzle", glm::vec3(0.0f));
  const glm::vec3 nose = socketAt("nose", glm::vec3(0.0f));
  const glm::vec3 venturi = socketAt("venturi", glm::vec3(0.0f, 0.0f, -0.8f));
  const glm::vec3 tube = glm::normalize(glm::vec3(gun.aimedFrom[2]));
  // Along the tube, the player's lifted for the range to what the camera
  // looks at (the sight's range marks set right), then off by the shot's
  // spread.
  glm::vec3 dir = tube;
  if (index == playedCharacter) {
    const glm::vec3 target = aimPoint(characters[index]);
    const glm::vec3 flat(target.x - nose.x, 0.0f, target.z - nose.z);
    const float lift = rocketLift(spec, asset.rocketFit, glm::length(flat), kGravity);
    const glm::vec3 right = glm::cross(tube, glm::vec3(0.0f, 1.0f, 0.0f));
    if (glm::length(right) > 1e-4f)
      dir = glm::angleAxis(lift, glm::normalize(right)) * tube;
  }
  dir = shotDirection(glm::mat3_cast(facingAlong(dir)), 0.0f, shot.deviation);

  LiveRocket rocket;
  rocket.flight = ::launchRocket(nose, dir, spec);
  rocket.asset = &asset;
  rocket.owner = index;
  if (gun.rocketPart >= 0) {
    auto node = std::make_unique<SceneNode>();
    node->setModelId(asset.partModels[gun.rocketPart]);
    rocket.node = node.get();
    rootNode.addChild(std::move(node));
  }
  rockets.push_back(rocket);
  gun.sinceShot = 0.0f;
  spdlog::info("Rocket away at {:.0f} m/s", glm::length(rocket.flight.velocity));

  // The booster's flash out of both ends; the back blast (the recoilless
  // launcher's push goes out behind): a jet of flame, a cloud of the
  // propellant's grey-white smoke, and dust off the ground if it's near; a
  // smaller puff at the muzzle.
  flashes.push_back({muzzle, tube, 0.45f, true});
  flashes.push_back({venturi, -tube, 1.4f, true});
  static ParticleRandom random(0x2c9e6b31u);
  ParticleSpawn fire;
  fire.colour = glm::vec3(24.0f, 11.0f, 3.0f);
  fire.size = 0.14f;
  fire.life = 0.02f; // the booster's burnt out in the tube: a flash
  fire.drag = 12.0f;
  emitInCone(particleSpawns, venturi, -tube, glm::radians(15.0f), 220, 10.0f, 45.0f, fire, random);
  ParticleSpawn smoke;
  smoke.colour = glm::vec3(0.72f, 0.71f, 0.68f);
  smoke.opacity = 0.55f; // laid over (est.)
  smoke.size = 0.5f;
  smoke.spread = 5.0f;
  smoke.life = 5.0f;
  smoke.drag = 3.0f;
  smoke.swirl = 0.6f;
  smoke.gravity = -0.03f; // warm, it rises a little
  emitInCone(particleSpawns, venturi, -tube, glm::radians(25.0f), kBackSmoke, 3.0f, 25.0f, smoke, random);
  if (physics) {
    // Where the blast meets the ground behind: down from a metre back.
    const glm::vec3 behind = venturi - tube * 1.0f;
    if (const std::optional<PhysicsRayHit> ground =
            physics->castRay(behind, glm::vec3(0.0f, -1.0f, 0.0f), kBackDustReach);
        ground && ground->normal.y > 0.5f) {
      const glm::vec3 at = behind - glm::vec3(0.0f, ground->distance - 0.15f, 0.0f);
      glm::vec3 back(-tube.x, 0.0f, -tube.z);
      back = glm::length(back) > 1e-3f ? glm::normalize(back) : glm::vec3(0.0f, 0.0f, -1.0f);
      ParticleSpawn dust;
      dust.colour = glm::vec3(0.45f, 0.41f, 0.35f);
      dust.opacity = 0.6f;
      dust.size = 0.4f;
      dust.spread = 6.0f;
      dust.life = 4.0f;
      dust.drag = 2.5f;
      dust.swirl = 0.8f;
      dust.gravity = -0.01f;
      // Thrown out along the ground, mostly away from the launcher.
      const glm::vec3 side = glm::cross(back, glm::vec3(0.0f, 1.0f, 0.0f));
      for (int k = 0; k < kBackDust; ++k) {
        const float a = glm::radians(80.0f) * (2.0f * random.next() - 1.0f);
        dust.position = at + (random.onSphere() * 0.2f);
        dust.velocity = (back * std::cos(a) + side * std::sin(a) + glm::vec3(0.0f, 0.15f, 0.0f)) *
                        (3.0f + 9.0f * random.next());
        particleSpawns.push_back(dust);
      }
    }
  }
  smoke.size = 0.3f;
  smoke.spread = 4.0f;
  smoke.life = 2.0f;
  emitInCone(particleSpawns, muzzle, tube, glm::radians(30.0f), 16, 1.0f, 5.0f, smoke, random);
}

void VulkanRenderer::updateRockets(float seconds) {
  for (size_t i = 0; i < rockets.size();) {
    LiveRocket &r = rockets[i];
    const RocketSpec &spec = *r.asset->spec.rocket;
    const glm::vec3 before = r.flight.position;
    stepRocket(r.flight, spec, r.asset->rocketFit, seconds, currentScene.wind, kGravity);
    const glm::vec3 moved = r.flight.position - before;
    bool gone = false;
    // What it meets on the way (not who fired it): the warhead goes off.
    if (glm::length(moved) > 1e-6f)
      if (const std::optional<BulletContact> hit = bulletContact(before, r.flight.position, r.owner)) {
        spdlog::info("Rocket hit at {:.1f} m after {:.2f} s", r.flight.flown, r.flight.time);
        rocketHit(hit->point, glm::normalize(moved), *r.asset, r.owner);
        gone = true;
      }
    if (!gone && rocketSpent(r.flight, spec)) {
      spdlog::info("Rocket self-destructs at {:.0f} m", r.flight.flown);
      explode(r.flight.position, CasedCharge{Explosive::Tnt, spec.warheadTnt, 0.0f, 0.0f}, r.owner);
      gone = true;
    }
    if (gone) {
      if (r.node)
        rootNode.removeChildren([&](const SceneNode &n) { return &n == r.node; });
      rockets.erase(rockets.begin() + static_cast<std::ptrdiff_t>(i));
      continue;
    }
    // The model: its nose at the flight's point, along its axis.
    const glm::quat turn = facingAlong(r.flight.axis);
    glm::vec3 noseInPart(0.0f, 0.0f, 0.7f);
    const WeaponParts &parts = r.asset->parts;
    if (const std::optional<WeaponSocket> nose = parts.socket("nose"))
      for (const WeaponPart &part : parts.parts)
        if (part.name == "rocket")
          noseInPart = nose->position - part.pivot;
    if (r.node)
      r.node->setPose(r.flight.position - turn * noseInPart, turn);
    // The motor: a flame out of its tail, its light (stepShots), and a
    // trail of smoke behind.
    r.tail = r.flight.position - r.flight.axis * (noseInPart.z + 0.25f);
    if (r.flight.motorOn) {
      const glm::vec3 tail = r.tail;
      addCone(effectStrips, tail, -r.flight.axis, kFlameLength, glm::radians(9.0f),
              glm::vec4(20.0f, 12.0f, 5.0f, 1.0f), glm::vec4(4.0f, 1.5f, 0.3f, 0.0f), 10);
      ParticleSpawn flame;
      flame.velocity = -r.flight.axis * 20.0f + r.flight.velocity * 0.6f;
      flame.colour = glm::vec3(30.0f, 15.0f, 5.0f);
      flame.size = 0.25f;
      flame.life = 0.04f;
      flame.drag = 8.0f;
      // So many a second whatever the frame rate, spread back along the
      // frame's path.
      const int flames = spawnCount(kFlamesPerSecond, seconds, r.flameCarry);
      for (int k = 0; k < flames; ++k) {
        flame.position = tail - moved * (static_cast<float>(k) / static_cast<float>(flames));
        particleSpawns.push_back(flame);
      }
      ParticleSpawn smoke;
      smoke.colour = glm::vec3(0.8f, 0.79f, 0.76f);
      smoke.opacity = 0.9f; // dense at first: it hangs on as it spreads (smokeCover)
      smoke.size = kTrailWidth;
      smoke.spread = kTrailSpread;
      smoke.life = kTrailLife;
      smoke.drag = 2.0f;
      smoke.swirl = 0.3f;
      smoke.gravity = -0.01f;
      // A puff every so far along the tail's path this frame, left still:
      // the swirl and the wind break the line up as it ages (a random kick
      // each, even 0.2 m/s, scatters puffs past their few-centimetre bright
      // middles and the young line reads as beads).
      emitTrail(particleSpawns, tail - moved, tail, kTrailEvery, r.trail, smoke);
    }
    ++i;
  }
}

void VulkanRenderer::rocketHit(glm::vec3 at, glm::vec3 dir, const WeaponAsset &asset, int owner) {
  const RocketSpec &spec = *asset.spec.rocket;
  heatJet(at, dir, spec.jetSteel, owner);
  // The blast where the warhead was, a little back out of what it hit.
  explode(at - dir * 0.1f, CasedCharge{Explosive::Tnt, spec.warheadTnt, 0.0f, 0.0f}, owner);
}

void VulkanRenderer::heatJet(glm::vec3 at, glm::vec3 dir, float steel, int owner) {
  // On through what's in line until its steel's worth runs out: each piece
  // it goes through broken loose, anyone in it killed.
  std::vector<BreakHit> broke;
  std::unordered_set<int> brokenNow;
  glm::vec3 from = at - dir * 0.05f;
  float travelled = 0.0f;
  int passed = 0;
  while (steel > 0.0f && travelled < kJetReach && passed < 16) {
    const std::optional<BulletContact> hit = bulletContact(from, from + dir * (kJetReach - travelled), owner);
    if (!hit)
      break;
    ++passed;
    const float density = matterDensity(hit->matter);
    const float through = std::min(hit->thickness, jetDepth(steel, density));
    if (hit->body & kCharacterHitFlag) {
      const size_t c = (hit->body >> 16) & 0xffffu;
      if (c < characters.size() && !characters[c].health.dead()) {
        LiveCharacter &victim = characters[c];
        victim.hitThisFrame = true;
        victim.health.wound(1e6f);
        // Thrown the way the jet went: from behind he falls forward.
        victim.deathClip = victim.crouched                        ? ClipSlot::DeathCrouched
                           : glm::dot(dir, victim.facing) > 0.0f ? ClipSlot::DeathBack
                                                                 : ClipSlot::DeathFront;
        spdlog::info("Character {} killed by the jet", c);
      }
    } else if (const auto known = bulletBodies.find(hit->body);
               known != bulletBodies.end() && (known->second & kBreakBodyFlag)) {
      const uint64_t b = known->second & ~kBreakBodyFlag;
      if (b < breakBodies.size() && breakBodies[b].id.value == static_cast<uint32_t>(hit->body))
        if (const int p = pieceAt(breakBodies[b], hit->point);
            p >= 0 && breakPieces[p].alive && !brokenNow.contains(p)) {
          // Gone right through it: its full toughness, whatever damage it
          // held. Stopped inside, it's left to the blast.
          if (through >= hit->thickness - 0.01f && damagePiece(p, -1.0f, hit->point, dir, 50.0f, broke))
            brokenNow.insert(p);
        }
    }
    static const char *kMatterNames[] = {"stone", "wood", "metal", "rubber", "ice", "dirt", "flesh"};
    spdlog::info("The jet went {:.2f} m into {} ({:.0f} kg/m^3), {:.0f} mm of steel's worth left", through,
                 kMatterNames[std::min(static_cast<size_t>(hit->matter), std::size(kMatterNames) - 1)], density,
                 1000.0f * std::max(0.0f, steel - jetUsed(through, density)));
    steel -= jetUsed(through, density);
    travelled += hit->distance + through;
    from = hit->point + dir * (through + 0.01f);
  }
  if (!broke.empty())
    regroupBreakables(broke);
}

void VulkanRenderer::clearRockets() {
  if (!rockets.empty()) {
    std::unordered_set<const SceneNode *> nodes;
    for (const LiveRocket &r : rockets)
      nodes.insert(r.node);
    rootNode.removeChildren([&](const SceneNode &n) { return nodes.contains(&n); });
  }
  rockets.clear();
}
