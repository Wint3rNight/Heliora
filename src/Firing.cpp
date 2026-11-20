// VulkanRenderer's firing (step 1.6, part 7): a gun's shots leave its muzzle
// as bullets (Bullets.h), which fly on every frame against the physics
// world and the characters' hit capsules; what they hit takes their
// momentum, a breakable piece their energy (PieceDamage.h), a character a
// wound. Tracers and the muzzle flash are drawn as light effects, and the
// shot is heard near or far. The maths is heliora_core's.

#include "VulkanRenderer.h"

#include "EffectGeometry.h"
#include "Particles.h"
#include "ScenePhysics.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <unordered_set>
#include <cmath>

namespace {

// How thick a thing a bullet looks for, back from beyond where it went in
// (deeper than any wall in the yard; a bullet stops long before).
constexpr float kDeepest = 1.5f; // m
// A tracer lights about 50 m out and burns to about 800 m (NSD pp. 51-52):
// at the AK's speed, 0.07 s and about 2 s in the air. Seen, it's a streak
// as long as the eye holds the light (est. 1/40 s).
constexpr float kTracerLights = 0.07f, kTracerBurns = 2.0f; // s
constexpr float kTracerStreak = 1.0f / 40.0f;              // s
// Burning, it's a point of light: seen at any range, so drawn no thinner
// than this angle (about three pixels across the view), however far.
constexpr float kTracerNarrowest = 0.003f; // radians
// The flash: a cone from the muzzle, this long for a pistol and a rifle
// (est.), and an outer flash half the time (Dye & Davis 2016).
constexpr float kPistolFlash = 0.12f, kRifleFlash = 0.25f; // m
// Its light's intensity (est., by eye in the night yard).
constexpr float kRifleFlashLight = 40.0f, kPistolFlashLight = 25.0f;
// Past this, a shot sounds as its far recording (est.).
constexpr float kFarShot = 60.0f; // m
// A bullet's hit, heard at 1 m (est.).
constexpr float kBulletHitLoudness = 95.0f; // dB
constexpr int kHitSoundsPerFrame = 4;

// What a hit throws off, back out of the surface (est., by eye; the
// particles only glow, so dust and splinters are dim and short): sparks
// off metal and stone, grit off stone and dirt, splinters off wood.
void hitDebris(std::vector<ParticleSpawn> &out, ParticleRandom &random, glm::vec3 point,
               glm::vec3 normal, Matter matter, float energy) {
  const float scale = std::clamp(energy / 1500.0f, 0.3f, 1.5f); // an AK round's 1.5 kJ is 1
  ParticleSpawn kind;
  kind.position = point + normal * 0.01f;
  auto burst = [&](int count, float slow, float fast, float spread) {
    emitInCone(out, kind.position, normal, spread, std::max(1, static_cast<int>(count * scale)), slow,
               fast, kind, random);
  };
  if (matter == Matter::Metal || matter == Matter::Stone) {
    kind.colour = matter == Matter::Metal ? glm::vec3(9.0f, 5.5f, 1.8f) : glm::vec3(5.0f, 3.0f, 1.0f);
    kind.size = 0.012f;
    kind.life = 0.25f;
    kind.drag = 2.0f;
    kind.gravity = 1.0f;
    burst(matter == Matter::Metal ? 18 : 6, 3.0f, 9.0f, glm::radians(55.0f));
  }
  if (matter == Matter::Stone || matter == Matter::Dirt || matter == Matter::Ice) {
    kind.colour = matter == Matter::Ice ? glm::vec3(0.7f, 0.75f, 0.8f) : glm::vec3(0.45f, 0.42f, 0.37f);
    kind.opacity = 0.5f; // a puff laid over, not glowing (est.)
    kind.size = 0.06f;
    kind.life = 0.9f;
    kind.drag = 4.0f;
    kind.swirl = 0.6f;
    kind.gravity = 0.15f;
    burst(14, 0.8f, 3.0f, glm::radians(40.0f));
    kind.size = 0.015f; // grit
    kind.opacity = 0.9f;
    kind.life = 0.6f;
    kind.drag = 1.0f;
    kind.swirl = 0.0f;
    kind.gravity = 1.0f;
    burst(10, 2.0f, 6.0f, glm::radians(35.0f));
  }
  if (matter == Matter::Wood) {
    kind.colour = glm::vec3(0.45f, 0.32f, 0.18f);
    kind.opacity = 0.9f; // splinters
    kind.size = 0.025f;
    kind.life = 0.8f;
    kind.drag = 1.5f;
    kind.gravity = 1.0f;
    burst(14, 2.0f, 6.0f, glm::radians(45.0f));
    kind.size = 0.05f; // a puff of wood dust
    kind.opacity = 0.4f;
    kind.colour = glm::vec3(0.5f, 0.42f, 0.3f);
    kind.life = 0.6f;
    kind.drag = 5.0f;
    kind.swirl = 0.4f;
    kind.gravity = 0.1f;
    burst(6, 0.5f, 1.5f, glm::radians(40.0f));
  }
  if (matter == Matter::Flesh) {
    kind.colour = glm::vec3(0.35f, 0.02f, 0.02f);
    kind.opacity = 0.9f;
    kind.size = 0.03f;
    kind.life = 0.4f;
    kind.drag = 3.0f;
    kind.gravity = 1.0f;
    burst(10, 1.0f, 3.0f, glm::radians(35.0f));
  }
}

} // namespace

void VulkanRenderer::fireShots(int index, CarriedGun &gun) {
  const WeaponAsset &asset = *gun.asset;
  const std::optional<WeaponSocket> muzzle = asset.parts.socket("muzzle");
  // From the gun as it was aimed: its visible kick is the recoil you see,
  // and the bullets' own recoil is already in their deviation.
  const glm::vec3 from =
      glm::vec3(gun.aimedFrom * glm::vec4(muzzle ? muzzle->position : glm::vec3(0.0f), 1.0f));
  const glm::mat3 frame(gun.aimedFrom);
  const bool rifle = !gun.atSide;
  for (const Shot &shot : gun.shots) {
    Bullet bullet;
    bullet.round = &asset.round;
    bullet.owner = index;
    bullet.tracer = shot.tracer;
    bullet.flight.position = from;
    bullet.flight.velocity = shotDirection(frame, asset.zero, shot.deviation) * asset.spec.muzzleSpeed;
    bullets.push_back(bullet);
    fireDice ^= fireDice << 13;
    fireDice ^= fireDice >> 17;
    fireDice ^= fireDice << 5;
    flashes.push_back({from, glm::vec3(frame[2]), rifle ? kRifleFlash : kPistolFlash, (fireDice & 1u) != 0});
    ejectCasing(gun);
    gun.sinceShot = 0.0f;
    // The kick you see: back, up (a little more or less each shot) and
    // aside either way.
    const float roll = static_cast<float>(fireDice % 10000u) / 10000.0f; // 0 to 1
    const WeaponSpec &spec = asset.spec;
    gun.kick.kick({spec.kickBack, glm::radians(spec.kickUp) * (0.85f + 0.3f * roll),
                   glm::radians(spec.kickAside) * (2.0f * std::fmod(roll * 7.0f, 1.0f) - 1.0f)},
                  spec.kickPeak);
    // The shot, its near recording close by and its far one further off.
    if (audio) {
      const float away = glm::distance(glm::vec3(sceneUbo.cameraPosition), from);
      const std::string &file =
          away > kFarShot && !asset.spec.farShotSound.empty() ? asset.spec.farShotSound : asset.spec.shotSound;
      const SoundClip *clip = file.empty() ? nullptr : audio->load(file); // loaded once, then kept
      if (clip) {
        SoundParams params;
        params.position = from;
        params.loudness = asset.spec.loudness;
        audio->play(clip, params);
      }
    }
  }
  gun.shots.clear();
}

std::optional<BulletContact> VulkanRenderer::bulletContact(glm::vec3 from, glm::vec3 to, int owner) {
  const glm::vec3 segment = to - from;
  const float length = glm::length(segment);
  if (length < 1e-6f || !physics)
    return std::nullopt;
  const glm::vec3 dir = segment / length;
  std::optional<BulletContact> best;
  if (const std::optional<PhysicsRayHit> hit = physics->castRay(from, dir, length)) {
    BulletContact contact;
    contact.distance = hit->distance;
    contact.point = from + dir * hit->distance;
    contact.normal = hit->normal;
    contact.body = hit->body.value;
    switch (surfaceUnder(hit->userData)) {
    case SoundSurface::Wood: contact.matter = Matter::Wood; break;
    case SoundSurface::Metal: contact.matter = Matter::Metal; break;
    case SoundSurface::Rubber: contact.matter = Matter::Rubber; break;
    case SoundSurface::Ice: contact.matter = Matter::Ice; break;
    case SoundSurface::Rubble: contact.matter = Matter::Dirt; break;
    default: contact.matter = Matter::Stone; break;
    }
    // How thick it is along the path: a ray back from beyond finds where
    // it comes out (past anything else in the way).
    contact.thickness = kDeepest;
    const glm::vec3 beyond = contact.point + dir * kDeepest;
    std::vector<PhysicsBodyId> skip;
    for (int k = 0; k < 4; ++k) {
      const std::optional<PhysicsRayHit> back = physics->castRay(beyond, -dir, kDeepest, skip);
      if (!back)
        break;
      if (back->body.value == hit->body.value) {
        contact.thickness = std::max(0.0f, kDeepest - back->distance);
        break;
      }
      skip.push_back(back->body);
    }
    bulletBodies[contact.body] = hit->userData;
    best = contact;
  }
  // The characters (the physics rays don't see them), not the shooter.
  for (int i = 0; i < static_cast<int>(characters.size()); ++i) {
    const LiveCharacter &c = characters[i];
    if (i == owner || owner == kFragmentOwner || c.capsules.empty() || c.health.dead())
      continue;
    if (const std::optional<CharacterHit> hit = firstHit(c.capsules, from, to);
        hit && (!best || hit->distance < best->distance)) {
      BulletContact contact;
      contact.distance = hit->distance;
      contact.point = hit->point;
      contact.normal = hit->normal;
      contact.body = kCharacterHitFlag | (static_cast<uint64_t>(i) << 16) |
                     static_cast<uint64_t>(hit->capsule);
      contact.matter = Matter::Flesh;
      contact.thickness = hit->thickness;
      best = contact;
    }
  }
  return best;
}

void VulkanRenderer::bulletHits(const std::vector<BulletImpact> &impacts) {
  std::vector<BreakHit> broke;
  std::unordered_set<int> brokenNow; // pieces this frame's bullets have already broken
  int sounded = 0;
  static ParticleRandom debrisRandom(0x51ed2701u);
  for (const BulletImpact &impact : impacts) {
    // What it throws off, where it went in (and where it came out).
    hitDebris(particleSpawns, debrisRandom, impact.point, impact.normal, impact.matter, impact.energy());
    if (!impact.stopped)
      hitDebris(particleSpawns, debrisRandom, impact.point + impact.direction * impact.thickness,
                impact.direction, impact.matter, 0.5f * impact.energy());
    if (impact.body & kCharacterHitFlag) {
      const size_t i = (impact.body >> 16) & 0xffffu;
      const size_t capsule = impact.body & 0xffffu;
      if (i >= characters.size() || capsule >= characters[i].capsules.size())
        continue;
      LiveCharacter &c = characters[i];
      if (c.health.dead())
        continue;
      const HitZone zone = c.capsules[capsule].zone;
      const float points = woundPoints(zone, impact.energy());
      c.hitThisFrame = true;
      if (c.health.wound(points)) {
        // Which way he goes down: by where the shot came from against how
        // he faces, a head shot from the front falling back, crouched
        // crumpling where he is (the right's clip for either side).
        const glm::vec3 left = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), c.facing));
        const float along = glm::dot(impact.direction, c.facing);
        c.deathClip = c.crouched                         ? ClipSlot::DeathCrouched
                      : along > 0.5f                     ? ClipSlot::DeathBack
                      : zone == HitZone::Head            ? ClipSlot::DeathHead
                      : along < -0.5f                    ? ClipSlot::DeathFront
                      : glm::dot(impact.direction, left) > 0.0f ? ClipSlot::DeathRight
                                                         : ClipSlot::DeathFront;
        spdlog::info("Character {} killed ({:.0f} J, {})", i, impact.energy(), clipSlotName(c.deathClip));
      } else {
        spdlog::info("Character {} hit: {:.0f} points, {:.0f} left", i, points, c.health.points);
      }
      continue;
    }
    const auto known = bulletBodies.find(impact.body);
    if (known == bulletBodies.end())
      continue;
    const uint64_t userData = known->second;
    PhysicsBodyId id;
    id.value = static_cast<uint32_t>(impact.body);
    // Its momentum, m v given up, to anything that moves (no spin: the
    // physics wrapper has no impulse at a point yet).
    if (const float mass = physics->mass(id); mass > 0.0f)
      physics->setLinearVelocity(id, physics->linearVelocity(id) + impact.direction * (impact.impulse() / mass));
    // Its energy to the breakable piece it went into.
    if (userData & kBreakBodyFlag) {
      const uint64_t b = userData & ~kBreakBodyFlag;
      if (b < breakBodies.size() && breakBodies[b].id.value == id.value) {
        if (const int p = pieceAt(breakBodies[b], impact.point);
            p >= 0 && breakPieces[p].alive && !brokenNow.contains(p)) {
          BreakPiece &piece = breakPieces[p];
          if (!(piece.damage.toughness > 0.0f)) {
            const SceneModelEntry &entry = currentScene.models[breakables[piece.object].entry.index];
            const std::optional<PhysicsMaterial> material =
                findMaterial(entry.material.value_or("stone"));
            piece.damage.toughness =
                pieceToughness(piece.shape->volume, material ? material->damageEnergy : 0.0f);
          }
          if (piece.damage.toughness > 0.0f) {
            stressDirty = true; // its links are weaker now
            if (piece.damage.add(impact.energy())) {
              spdlog::info("Bullets broke a piece ({:.3f} m^3) after {:.0f} J", piece.shape->volume,
                           piece.damage.energy);
              for (int link : pieceLinks[p])
                breakLinks[link].broken = true;
              for (int link : pieceGroundLinks[p])
                groundLinks[link].broken = true;
              piece.anchor = false; // shot loose: nothing holds it in place now
              brokenNow.insert(p);
              queueBulletSplit(p, impact.point, impact.direction);
              broke.push_back({impact.point, impact.direction, impact.impulse(), PhysicsBodyId{},
                               glm::vec3(0.0f)});
            }
          }
        }
      }
    }
    if (audio && sounded < kHitSoundsPerFrame) {
      playSound(knockSounds[static_cast<size_t>(surfaceUnder(userData))], impact.point,
                kBulletHitLoudness);
      ++sounded;
    }
  }
  if (!broke.empty())
    regroupBreakables(broke);
}

void VulkanRenderer::stepShots(float seconds) {
  const glm::vec3 eye(sceneUbo.cameraPosition);
  // The flash lights its surroundings for the frame of the shot; a blast's
  // for a few.
  sceneUbo.flashColour = glm::vec4(0.0f);
  if (blastLight > 0.0f) {
    sceneUbo.flashPosition = glm::vec4(blastLightAt, 1.0f);
    sceneUbo.flashColour = glm::vec4(1.0f, 0.7f, 0.4f, blastLightPower);
  } else if (!flashes.empty()) {
    const MuzzleFlash &flash = flashes.front();
    sceneUbo.flashPosition = glm::vec4(flash.at + flash.along * (0.5f * flash.size), 1.0f);
    sceneUbo.flashColour = glm::vec4(1.0f, 0.62f, 0.3f, flash.size >= kRifleFlash ? kRifleFlashLight
                                                                                  : kPistolFlashLight);
  }
  // The muzzle flashes of this frame's shots: one frame each (the real one
  // is over in 3 ms).
  for (const MuzzleFlash &flash : flashes) {
    addCone(effectStrips, flash.at, flash.along, flash.size, 0.35f, glm::vec4(14.0f, 9.0f, 3.5f, 1.0f),
            glm::vec4(3.0f, 1.2f, 0.2f, 0.0f), 10);
    if (flash.outer)
      addCone(effectStrips, flash.at + flash.along * (0.3f * flash.size), flash.along,
              1.8f * flash.size, 0.6f, glm::vec4(4.0f, 2.2f, 0.6f, 1.0f),
              glm::vec4(0.8f, 0.3f, 0.05f, 0.0f), 10);
  }
  flashes.clear();
  if (bullets.empty() || !(seconds > 0.0f) || !physics)
    return;

  // Each shooter's bullets on their own, so none hits its own shooter.
  std::vector<BulletImpact> impacts;
  bulletBodies.clear();
  std::vector<Bullet> flying;
  std::vector<int> owners;
  for (const Bullet &b : bullets)
    if (std::find(owners.begin(), owners.end(), b.owner) == owners.end())
      owners.push_back(b.owner);
  const Air air;
  const glm::vec3 gravity(0.0f, -9.81f, 0.0f);
  for (int owner : owners) {
    std::vector<Bullet> theirs;
    for (const Bullet &b : bullets)
      if (b.owner == owner)
        theirs.push_back(b);
    const BulletWorld world = [&](glm::vec3 from, glm::vec3 to) {
      return bulletContact(from, to, owner);
    };
    stepBullets(theirs, seconds, air, gravity, world, impacts);
    flying.insert(flying.end(), theirs.begin(), theirs.end());
  }
  bullets = std::move(flying);
  bulletHits(impacts);

  // Tracers: a red streak behind each lit one.
  for (const Bullet &b : bullets) {
    if (!b.tracer || b.age < kTracerLights || b.age > kTracerBurns)
      continue;
    const float speed = glm::length(b.flight.velocity);
    if (speed < 1.0f)
      continue;
    const glm::vec3 dir = b.flight.velocity / speed;
    const float streak = std::min(speed * kTracerStreak, speed * (b.age - kTracerLights));
    const glm::vec3 tail = b.flight.position - dir * streak, head = b.flight.position;
    const float tailWidth = std::max(0.01f, glm::distance(eye, tail) * kTracerNarrowest);
    const float headWidth = std::max(0.03f, glm::distance(eye, head) * kTracerNarrowest);
    addFacingStrip(effectStrips, {tail, head}, eye, tailWidth, headWidth,
                   glm::vec4(2.0f, 0.1f, 0.05f, 0.0f), glm::vec4(12.0f, 1.2f, 0.5f, 1.0f),
                   glm::vec4(3.0f, 0.1f, 0.05f, 0.0f));
    // Its burning head, facing the eye: a streak seen end on (flying
    // straight away from behind the shooter) has no length to show.
    const glm::vec3 view = glm::normalize(head - eye);
    const glm::vec3 across = glm::normalize(
        glm::cross(view, std::abs(view.y) < 0.9f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f)));
    addFacingStrip(effectStrips, {head - across * (0.5f * headWidth), head + across * (0.5f * headWidth)},
                   eye, headWidth, headWidth, glm::vec4(40.0f, 4.0f, 1.5f, 1.0f),
                   glm::vec4(40.0f, 4.0f, 1.5f, 1.0f), glm::vec4(8.0f, 0.3f, 0.1f, 0.0f));
  }
}

void VulkanRenderer::ejectCasing(const CarriedGun &gun) {
  constexpr size_t kMostCasings = 64;
  const bool rifle = !gun.atSide;
  const int model = casingModels[rifle ? 0 : 1];
  if (!physics || model < 0)
    return;
  // Out of the ejection port: the AK's to the right and a little forward;
  // the P38's to the left and up (its port is on the left). Gun space: +x
  // its left, +y up, +z ahead.
  const std::optional<WeaponSocket> port = gun.asset->parts.socket("ejection");
  const glm::vec3 inGun = port ? port->position : glm::vec3(0.01f, 0.03f, 0.03f);
  const glm::vec3 throwInGun = rifle ? glm::vec3(-1.0f, 0.35f, 0.3f) : glm::vec3(0.8f, 0.8f, -0.1f);
  fireDice ^= fireDice << 13;
  fireDice ^= fireDice >> 17;
  fireDice ^= fireDice << 5;
  const float jitter = static_cast<float>(fireDice % 1000u) / 1000.0f;
  const glm::mat3 frame(gun.at);
  SceneModelEntry casing;
  casing.shape = PrimitiveShape::Capsule;
  casing.radius = rifle ? 0.0057f : 0.005f;
  casing.height = rifle ? 0.0387f : 0.0192f;
  casing.body = PhysicsMotion::Dynamic;
  casing.mass = rifle ? 0.0065f : 0.0038f; // kg (est.: a steel 7.62x39 case, a brass 9 mm)
  casing.restitution = 0.4f;
  casing.transform.position = glm::vec3(gun.at * glm::vec4(inGun, 1.0f));
  PhysicsBodyDesc body = bodyDescFor(casing, nullptr, 0.4f);
  // Lying across the gun, its length (the capsule's y) along the gun's x.
  body.rotation = glm::normalize(glm::quat_cast(frame)) *
                  glm::angleAxis(glm::half_pi<float>(), glm::vec3(0.0f, 0.0f, 1.0f));
  body.linearVelocity = frame * glm::normalize(throwInGun) * (3.5f + 2.0f * jitter); // m/s (est.)
  body.angularVelocity = frame * glm::vec3(0.0f, 25.0f + 20.0f * jitter, 0.0f);
  // The oldest goes first, so its link slot can be used again.
  if (casings.size() >= kMostCasings) {
    const Casing old = casings.front();
    casings.erase(casings.begin());
    physics->removeBody(old.body);
    physicsLinks[old.link] = {};
    freeCasingLinks.push_back(old.link);
    rootNode.removeChildren([&](const SceneNode &n) { return &n == old.node; });
  }
  int link = static_cast<int>(physicsLinks.size());
  if (!freeCasingLinks.empty()) {
    link = freeCasingLinks.back();
    freeCasingLinks.pop_back();
  } else {
    physicsLinks.emplace_back();
  }
  body.userData = static_cast<uint64_t>(link);
  const PhysicsBodyId id = physics->addBody(body);
  if (!id.isValid()) {
    freeCasingLinks.push_back(link);
    return;
  }
  auto node = std::make_unique<SceneNode>();
  node->setModelId(model);
  node->setPose(body.position, body.rotation);
  node->setPhysicsLink(link);
  casings.push_back({node.get(), id, link});
  physicsLinks[link] = {node.get(), id, SoundSurface::Metal};
  rootNode.addChild(std::move(node));
}

void VulkanRenderer::clearCasings() {
  if (casings.empty())
    return;
  std::unordered_set<const SceneNode *> nodes;
  for (const Casing &c : casings) {
    if (physics && c.body.isValid())
      physics->removeBody(c.body);
    if (c.link >= 0 && c.link < static_cast<int>(physicsLinks.size()))
      physicsLinks[c.link] = {};
    nodes.insert(c.node);
  }
  rootNode.removeChildren([&](const SceneNode &n) { return nodes.contains(&n); });
  casings.clear();
  freeCasingLinks.clear();
}
