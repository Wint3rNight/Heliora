// VulkanRenderer's melee (step F3): the played character's blows from the
// keys (Melee.h decides which and when), and where they land: the striking
// hand or foot, swept from last frame's place to this one's, against the
// other characters' body parts and the breakable pieces. A blow that lands
// hurts, flinches and knocks back, earns focus, stops every animation for
// a moment (hit stop), throws what a hit throws off and is heard.

#include "VulkanRenderer.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <string_view>

namespace {

// How big a fist and a foot are, m (est.).
constexpr float kFistRadius = 0.05f, kFootRadius = 0.06f;
// Hit stop: a light blow freezes everything this long, a heavy one longer
// (est., the usual tenth of a second or less).
constexpr float kLightPause = 0.06f, kHeavyPause = 0.11f;
// What a blow's energy pushes with: p = sqrt(2 m E), the fist or foot and
// the arm or leg behind it moving as about this mass (est.).
constexpr float kBlowMass = 4.0f; // kg
// A blow landing, heard at 1 m (est.).
constexpr float kPunchLoudness = 90.0f; // dB
// Stepping in: a blow pulls him toward someone this close in front (within
// this angle of where he faces), to this far apart, the step dying away
// like a knock (the usual melee "magnetism"; est.).
constexpr float kStepInReach = 2.0f, kStepInAngle = 1.05f; // m, radians (60 degrees)
constexpr float kStrikeDistance = 0.7f;                    // m, middle to middle
constexpr float kStepFade = 0.15f, kFastestStep = 6.0f;    // s, m/s

// The bone named this after Mixamo's prefix ("mixamorig:RightHand"), or -1.
int boneEnding(const Skeleton &skeleton, std::string_view name) {
  for (int i = 0; i < static_cast<int>(skeleton.bones.size()); ++i) {
    const std::string &bone = skeleton.bones[i].name;
    if (bone.size() >= name.size() && bone.compare(bone.size() - name.size(), name.size(), name) == 0)
      return i;
  }
  return -1;
}

glm::vec3 flat(glm::vec3 v) { return glm::vec3(v.x, 0.0f, v.z); }

} // namespace

void VulkanRenderer::timeBlows(const LiveCharacter &c) {
  meleeTuning = MeleeTuning{};
  if (c.copy < 0 || !c.file || !c.file->asset)
    return;
  const AnimationLibrary &library = c.file->asset->library;
  const std::pair<ClipSlot, MeleeAttack *> blows[4] = {{ClipSlot::Attack1, &meleeTuning.light[0]},
                                                       {ClipSlot::Attack2, &meleeTuning.light[1]},
                                                       {ClipSlot::Attack3, &meleeTuning.light[2]},
                                                       {ClipSlot::AttackHeavy, &meleeTuning.heavy}};
  for (const auto &[slot, blow] : blows)
    if (const CharacterClipData *clip = library.clip(slot); clip && clip->hitFrom >= 0.0f)
      timeBlowFromClip(*blow, clip->start, clip->endTime(), clip->hitFrom, clip->hitTo);
}

void VulkanRenderer::updateMelee(float seconds) {
  const float dt = hitPause > 0.0f ? 0.0f : seconds;
  hitPause = std::max(0.0f, hitPause - seconds);
  if (playedCharacter < 0 || playedCharacter >= static_cast<int>(characters.size()) ||
      !playerController)
    return;
  LiveCharacter &c = characters[playedCharacter];
  if (c.health.dead())
    return;
  MeleeInput in;
  in.attack = playerControls.fire;
  in.block = playerControls.block;
  in.dodge = playerControls.dodge;
  // With something held (or a power going off) the button throws it.
  in.busy = powerController.holding() || powerController.state() != PowerState::Idle;
  const MeleeEvents e = c.melee.update(in, dt, meleeTuning);
  if (e.started >= 0) {
    c.struck.clear();
    c.struckPiece = false;
    spdlog::info("[melee] blow {}", e.started);
    // Step in toward whoever is close in front, where the camera looks.
    if (c.body >= 0 && roster) {
      CharacterMovement &me = roster->body(c.body);
      const glm::vec3 ahead = flat(cameraForward(playerController->camera().yaw(), 0.0f));
      float nearest = kStepInReach;
      glm::vec3 to(0.0f);
      for (int j = 0; j < static_cast<int>(characters.size()); ++j) {
        if (j == playedCharacter || characters[j].body < 0 || characters[j].health.dead())
          continue;
        const glm::vec3 d = flat(roster->body(characters[j].body).position() - me.position());
        const float distance = glm::length(d);
        if (distance < 1e-3f || distance >= nearest ||
            std::acos(std::clamp(glm::dot(d / distance, glm::normalize(ahead)), -1.0f, 1.0f)) > kStepInAngle)
          continue;
        nearest = distance;
        to = d;
      }
      if (glm::length(to) > kStrikeDistance)
        me.knock(glm::normalize(to) * std::min(kFastestStep, (nearest - kStrikeDistance) / kStepFade));
    }
  }
  // A blow turns him to where the camera looks and plants his feet until
  // it has struck; its recovery lets him move again.
  const bool attacking = c.melee.state() == MeleeState::Attacking;
  const MeleeAttack &blow = c.melee.blow(c.melee.attack(), meleeTuning);
  playerController->setThrowing(attacking);
  playerController->setRooted(attacking && c.melee.attackTime() < blow.windup + blow.active);
}

void VulkanRenderer::strikeWith(int index, const glm::mat4 &placement,
                                const std::vector<glm::mat4> &model) {
  LiveCharacter &c = characters[index];
  const CharacterAsset *asset = c.copy >= 0 ? c.file->asset : nullptr;
  if (!asset)
    return;
  if (c.limbBones[0] < 0 && c.limbBones[1] < 0) {
    static constexpr std::string_view kLimbs[4] = {"RightHand", "LeftHand", "RightFoot", "LeftFoot"};
    for (int k = 0; k < 4; ++k)
      c.limbBones[k] = boneEnding(asset->skeleton, kLimbs[k]);
  }
  std::array<glm::vec3, 4> now{};
  for (int k = 0; k < 4; ++k)
    now[k] = c.limbBones[k] >= 0 ? glm::vec3(placement * model[c.limbBones[k]][3]) : glm::vec3(0.0f);
  const bool wasPlaced = c.limbsPlaced;
  const std::array<glm::vec3, 4> before = c.limbAt;
  c.limbAt = now;
  c.limbsPlaced = true;
  if (!wasPlaced || !c.melee.striking() || c.body < 0 || !roster)
    return;
  const int which = c.melee.attack();
  const MeleeAttack &blow = c.melee.blow(which, meleeTuning);
  const int limb = static_cast<int>(blow.limb);
  if (c.limbBones[limb] < 0)
    return;
  const glm::vec3 from = before[limb], to = now[limb];
  const bool foot = blow.limb == Limb::RightFoot || blow.limb == Limb::LeftFoot;
  const float radius = foot ? kFootRadius : kFistRadius;
  const bool heavy = which == kHeavyAttack || which == 2;
  const glm::vec3 me = roster->body(c.body).position();

  // The others: the first body part the fist or foot passes, once a blow each.
  for (int j = 0; j < static_cast<int>(characters.size()); ++j) {
    LiveCharacter &other = characters[j];
    if (j == index || other.capsules.empty() || other.health.dead() ||
        std::find(c.struck.begin(), c.struck.end(), j) != c.struck.end())
      continue;
    const std::optional<CharacterHit> hit = sweptHit(other.capsules, from, to, radius);
    if (!hit)
      continue;
    c.struck.push_back(j);
    const glm::vec3 them = other.body >= 0 ? roster->body(other.body).position() : hit->point;
    glm::vec3 away = flat(them - me);
    away = glm::length(away) > 1e-3f ? glm::normalize(away) : c.facing;
    const bool killed = hurtCharacter(j, hit->zone, blow.damage, away);
    if (other.body >= 0)
      roster->body(other.body).knock(away * blow.push);
    if (index == playedCharacter)
      powerController.focus().gain(killed ? powerTuning().focusKill : powerTuning().focusHit,
                                   powerTuning());
    hitPause = std::max(hitPause, heavy ? kHeavyPause : kLightPause);
    hitEffects(hit->point, hit->normal, Matter::Flesh, blow.energy);
    if (audio)
      playSound(punchSounds[heavy ? 1 : 0], hit->point, kPunchLoudness + (heavy ? 6.0f : 0.0f));
    c.shake.kick({0.0f, glm::radians(heavy ? 1.2f : 0.5f), 0.0f}, 0.05f);
    spdlog::info("[melee] blow {} landed on character {} ({}): {:.0f} points{}", which, j,
                 static_cast<int>(hit->zone), blow.damage, killed ? ", killed" : "");
  }

  // A breakable piece (or anything loose) in its way, once a blow.
  if (c.struckPiece || !physics)
    return;
  const glm::vec3 path = to - from;
  const float length = glm::length(path);
  if (length < 1e-4f)
    return;
  const glm::vec3 dir = path / length;
  const std::optional<PhysicsRayHit> hit = physics->castRay(from, dir, length + radius);
  if (!hit)
    return;
  c.struckPiece = true;
  const glm::vec3 point = from + dir * hit->distance;
  const float impulse = std::sqrt(2.0f * kBlowMass * blow.energy);
  if (const float mass = physics->mass(hit->body); mass > 0.0f)
    physics->setLinearVelocity(hit->body, physics->linearVelocity(hit->body) + dir * (impulse / mass));
  std::vector<BreakHit> broke;
  if (hit->userData & kBreakBodyFlag) {
    const uint64_t b = hit->userData & ~kBreakBodyFlag;
    if (b < breakBodies.size() && breakBodies[b].id.value == hit->body.value)
      if (const int p = pieceAt(breakBodies[b], point); p >= 0 && breakPieces[p].alive)
        damagePiece(p, blow.energy, point, dir, impulse, broke);
  }
  const SoundSurface surface = surfaceUnder(hit->userData);
  hitEffects(point, -dir,
             surface == SoundSurface::Wood    ? Matter::Wood
             : surface == SoundSurface::Metal ? Matter::Metal
                                              : Matter::Stone,
             blow.energy);
  if (audio)
    playSound(knockSounds[static_cast<size_t>(surface)], point, kPunchLoudness);
  hitPause = std::max(hitPause, kLightPause);
  if (!broke.empty())
    regroupBreakables(broke);
}
