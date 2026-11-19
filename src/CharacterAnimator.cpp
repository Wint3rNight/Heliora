#include "CharacterAnimator.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string_view>
#include <vector>

namespace {

// The bone named this after Mixamo's prefix ("mixamorig:Spine"), or -1.
int boneNamed(const Skeleton &skeleton, std::string_view name) {
  for (int i = 0; i < static_cast<int>(skeleton.bones.size()); ++i) {
    const std::string_view bone = skeleton.bones[i].name;
    if (bone.size() < name.size() || bone.substr(bone.size() - name.size()) != name)
      continue;
    if (bone.size() == name.size() ||
        !std::isalnum(static_cast<unsigned char>(bone[bone.size() - name.size() - 1])))
      return i;
  }
  return -1;
}

size_t index(ClipSlot slot) { return static_cast<size_t>(slot); }

// Where a missing clip's move goes instead, ending at idle: a sideways run
// becomes a run, a run a walk, a crouch walk a crouch, a dodge the roll.
ClipSlot fallback(ClipSlot slot) {
  switch (slot) {
  case ClipSlot::WalkBack:
  case ClipSlot::WalkLeft:
  case ClipSlot::WalkRight:
    return ClipSlot::WalkForward;
  case ClipSlot::RunBack:
  case ClipSlot::RunLeft:
  case ClipSlot::RunRight:
  case ClipSlot::Sprint:
    return ClipSlot::RunForward;
  case ClipSlot::RunForward:
    return ClipSlot::WalkForward;
  case ClipSlot::CrouchBack:
  case ClipSlot::CrouchLeft:
  case ClipSlot::CrouchRight:
    return ClipSlot::CrouchForward;
  case ClipSlot::CrouchForward:
    return ClipSlot::CrouchIdle;
  case ClipSlot::Jump:
    return ClipSlot::Fall;
  case ClipSlot::LandHard:
    return ClipSlot::Land;
  case ClipSlot::DodgeForward:
  case ClipSlot::DodgeBack:
  case ClipSlot::DodgeLeft:
  case ClipSlot::DodgeRight:
    return ClipSlot::Roll;
  default:
    return ClipSlot::Idle;
  }
}

// Standing and moving: what a gun stance falls back on within itself, so
// the arms keep holding the gun.
bool isLocomotion(ClipSlot slot) {
  switch (slot) {
  case ClipSlot::Idle:
  case ClipSlot::AimIdle:
  case ClipSlot::WalkForward:
  case ClipSlot::WalkBack:
  case ClipSlot::WalkLeft:
  case ClipSlot::WalkRight:
  case ClipSlot::RunForward:
  case ClipSlot::RunBack:
  case ClipSlot::RunLeft:
  case ClipSlot::RunRight:
  case ClipSlot::Sprint:
  case ClipSlot::CrouchIdle:
  case ClipSlot::CrouchForward:
  case ClipSlot::CrouchBack:
  case ClipSlot::CrouchLeft:
  case ClipSlot::CrouchRight:
    return true;
  default:
    return false;
  }
}

bool isCrouch(ClipSlot slot) {
  return slot == ClipSlot::CrouchIdle || slot == ClipSlot::CrouchForward ||
         slot == ClipSlot::CrouchBack || slot == ClipSlot::CrouchLeft ||
         slot == ClipSlot::CrouchRight;
}

// A reload eases in over its first tenth and out over its last (est.).
constexpr float kReloadEase = 0.1f;

// A pose built up from weighted clips.
struct Mix {
  Pose pose;
  float total = 0.0f;
  void add(const Pose &p, float weight) {
    if (weight <= 0.0f)
      return;
    total += weight;
    if (total == weight)
      pose = p;
    else
      blendPoses(pose, p, weight / total, pose);
  }
};

// A moving clip's share of the blend.
struct Moving {
  ClipSlot slot;
  float weight;
};

// Each running stride at most this much longer or shorter than the clip's
// (est.): past it the clip plays faster or slower instead.
constexpr float kShortestStride = 0.7f;
constexpr float kLongestStride = 1.3f;

} // namespace

float runningCadence(float speed) {
  struct Point {
    float speed, strides;
  };
  static constexpr Point kTable[] = {{3.2f, 1.33f}, {3.8f, 1.37f}, {4.5f, 1.45f},
                                     {5.1f, 1.52f}, {5.7f, 1.63f}, {6.4f, 1.73f}};
  constexpr int n = sizeof(kTable) / sizeof(kTable[0]);
  if (!(speed > kTable[0].speed))
    return kTable[0].strides;
  int i = 0;
  while (i + 2 < n && speed > kTable[i + 1].speed)
    ++i;
  const Point a = kTable[i], b = kTable[i + 1];
  const float along = (speed - a.speed) / (b.speed - a.speed);
  return std::min(a.strides + (b.strides - a.strides) * along, 2.5f);
}

const char *animStateName(AnimState state) {
  switch (state) {
  case AnimState::Locomotion: return "Locomotion";
  case AnimState::Crouch: return "Crouch";
  case AnimState::Jump: return "Jump";
  case AnimState::Fall: return "Fall";
  case AnimState::Land: return "Land";
  case AnimState::Dodge: return "Dodge";
  }
  return "";
}

CharacterAnimator::CharacterAnimator(const Skeleton &skeleton,
                                     const AnimationLibrary &clips,
                                     float unitsToMetres, const AnimatorTuning &tuning)
    : skeleton(skeleton), library(clips), unitsToMetres(unitsToMetres),
      tune(tuning), current(bindPose(skeleton)) {
  base = current;
  lastBase = current;
  beforeLastBase = current;
  for (const char *name : {"Spine", "Spine1", "Spine2"})
    if (const int bone = boneNamed(skeleton, name); bone >= 0)
      spine.push_back(bone);
  if (!spine.empty())
    upperBody = upperBodyMask(skeleton, spine.front());
  for (const char *name : {"RightShoulder", "LeftShoulder"})
    if (const int bone = boneNamed(skeleton, name); bone >= 0) {
      const std::vector<float> side = upperBodyMask(skeleton, bone);
      if (rightArm.empty() && std::string_view(name) == "RightShoulder")
        rightArm = side;
      if (arms.empty())
        arms = side;
      else
        for (size_t i = 0; i < arms.size(); ++i)
          arms[i] = std::max(arms[i], side[i]);
    }
}

float CharacterAnimator::clipWeight(ClipSlot slot) const {
  return weights[index(slot)];
}

float CharacterAnimator::clipTime(ClipSlot slot) const {
  return times[index(slot)];
}

void CharacterAnimator::update(const AnimatorInput &input, float dt) {
  // Taking out or putting away a gun: its stance's clips, eased in.
  if (input.stance != activeStance) {
    activeStance = input.stance;
    blendPending = true;
    fadeTime = tune.stanceFade;
  }
  // The clip a move plays: the stance's own (for standing and moving, its
  // nearest own move), else the unarmed one, or the nearest there is.
  auto resolve = [&](ClipSlot slot) -> std::pair<ClipSlot, const CharacterClipData *> {
    if (activeStance != Stance::Unarmed) {
      ClipSlot own = slot;
      for (int guard = 0; guard < 6; ++guard) {
        if (const CharacterClipData *clip = library.clip(own, activeStance))
          return {own, clip};
        if (!isLocomotion(slot) || own == ClipSlot::Idle)
          break;
        const ClipSlot next = own == ClipSlot::AimIdle ? ClipSlot::Idle : fallback(own);
        // Crouched, only its own crouch moves: standing ones would stand
        // the legs up, so the unarmed crouch it is.
        if (isCrouch(slot) && !isCrouch(next))
          break;
        own = next;
      }
    }
    if (slot == ClipSlot::AimIdle)
      slot = ClipSlot::Idle; // unarmed, aiming still is the idle
    for (int guard = 0; guard < 6; ++guard) {
      if (const CharacterClipData *clip = library.clip(slot))
        return {slot, clip};
      if (slot == ClipSlot::Idle)
        break;
      slot = fallback(slot);
    }
    return {slot, nullptr};
  };
  auto clipSpeed = [&](const CharacterClipData &clip) {
    return clip.rootMotion.speed() * unitsToMetres;
  };

  // --- Time in the air, and whether this time began with a jump.
  const bool leftGround = wasGrounded && !input.grounded;
  if (input.grounded) {
    if (!wasGrounded)
      lastAirTime = airTime;
    airTime = 0.0f;
  } else {
    if (leftGround) {
      jumped = input.verticalSpeed > 0.5f;
      takeoffSpeed = input.verticalSpeed;
    }
    airTime += dt;
  }
  wasGrounded = input.grounded;

  // --- Which state.
  const AnimState groundState = input.crouched ? AnimState::Crouch
                                               : AnimState::Locomotion;
  // A jump stays in the jump clip, up and down; only falling fast is a fall.
  // Walking off a ledge goes to the same clip after a moment.
  const bool fallingFast = -input.verticalSpeed >= tune.fallSpeed;
  // A real fall lands hard, anything else softly.
  const ClipSlot landing = active == AnimState::Fall && library.clip(ClipSlot::LandHard)
                               ? ClipSlot::LandHard
                               : ClipSlot::Land;
  AnimState next = active;
  if (!input.grounded) {
    switch (active) {
    case AnimState::Jump:
      if (fallingFast)
        next = AnimState::Fall;
      break;
    case AnimState::Fall:
      break;
    default:
      if (leftGround && jumped)
        next = AnimState::Jump;
      else if (airTime >= tune.fallDelay && !(active == AnimState::Dodge && input.dodging))
        next = fallingFast ? AnimState::Fall : AnimState::Jump;
      break;
    }
  } else if (input.dodging && active != AnimState::Dodge) {
    next = AnimState::Dodge;
  } else {
    switch (active) {
    case AnimState::Dodge:
      if (!input.dodging)
        next = groundState;
      break;
    case AnimState::Jump:
    case AnimState::Fall:
      next = lastAirTime >= tune.landAirTime && library.clip(landing)
                 ? AnimState::Land
                 : groundState;
      break;
    case AnimState::Land: {
      const CharacterClipData *land = library.clip(oneShot);
      const float length = land ? land->endTime() - land->start : 0.0f;
      if (stateTime >= length ||
          (stateTime >= tune.landMinTime && input.speed > 0.5f))
        next = groundState;
      break;
    }
    default:
      next = groundState;
      break;
    }
  }
  if (next != active) {
    // The blend starts once the new state's pose is known, from the state's
    // own poses: the aim's lean fades by itself, and baked in here it would
    // be laid on twice.
    blendPending = true;
    switch (next) {
    case AnimState::Locomotion: fadeTime = tune.locomotionFade; break;
    case AnimState::Crouch: fadeTime = tune.crouchFade; break;
    case AnimState::Jump: fadeTime = tune.jumpFade; break;
    case AnimState::Fall: fadeTime = tune.fallFade; break;
    case AnimState::Land: fadeTime = tune.landFade; break;
    case AnimState::Dodge: fadeTime = tune.dodgeFade; break;
    }
    if (next == AnimState::Jump)
      oneShot = ClipSlot::Jump;
    else if (next == AnimState::Land)
      oneShot = landing;
    else if (next == AnimState::Dodge)
      oneShot = input.dodgeClip; // the controller picked it
    active = next;
    stateTime = 0.0f;
  }
  stateTime += dt;
  idleTime += dt;

  // --- The state's pose.
  weights.fill(0.0f);
  stride = 1.0f;
  // Where it moves, in the skeleton's space: it faces +z with its right
  // along -x (glTF's and Mixamo's way round).
  axis = input.aiming && glm::length(input.direction) > 1e-4f
             ? glm::normalize(glm::vec3(-input.direction.x, 0.0f, input.direction.y))
             : glm::vec3(0.0f, 0.0f, 1.0f);
  Mix mix;
  auto sample = [&](ClipSlot slot, float time, float weight) {
    if (weight <= 0.0f)
      return;
    const auto [used, clip] = resolve(slot);
    if (!clip)
      return;
    Pose pose = bindPose(skeleton);
    sampleClip(clip->clip, skeleton, time, clip->loop, pose);
    mix.add(pose, weight);
    weights[index(used)] += weight;
    times[index(used)] = time;
  };

  if (active == AnimState::Locomotion || active == AnimState::Crouch) {
    // Rows of clips by speed: standing still, then each moving level at the
    // controller's speed for that gait, or the clips' own (scaled to the
    // character) without one. While aiming each moving level is the four
    // directions mixed by where it moves.
    const bool crouch = active == AnimState::Crouch;
    const DirectionWeights dir = input.aiming ? directionBlend(input.direction)
                                              : DirectionWeights{1.0f, 0.0f, 0.0f, 0.0f};
    struct Level {
      std::array<ClipSlot, 4> slots; // forward, back, left, right
      float speed = 0.0f;     // where its row sits
      float clipSpeed = 0.0f; // what its clips cover at their own rate
      bool running = false;   // played at a runner's cadence
    };
    const std::array<float, 4> dirWeights = {dir.forward, dir.back, dir.left,
                                             dir.right};
    std::vector<Level> levels;
    auto addLevel = [&](std::array<ClipSlot, 4> slots, float gaitSpeed, bool running) {
      float own = 0.0f;
      for (int k = 0; k < 4; ++k)
        if (dirWeights[k] > 0.0f)
          if (const auto [used, clip] = resolve(slots[k]); clip)
            own += dirWeights[k] * clipSpeed(*clip);
      if (own <= 1e-4f)
        return;
      const float speed = gaitSpeed > 0.0f ? gaitSpeed : own;
      // Only levels faster than the one below make a usable row.
      const float below = levels.empty() ? 0.0f : levels.back().speed;
      if (speed > below + 1e-4f)
        levels.push_back({slots, speed, own, running});
    };
    if (crouch) {
      addLevel({ClipSlot::CrouchForward, ClipSlot::CrouchBack, ClipSlot::CrouchLeft,
                ClipSlot::CrouchRight},
               input.crouchSpeed, false);
    } else {
      addLevel({ClipSlot::WalkForward, ClipSlot::WalkBack, ClipSlot::WalkLeft,
                ClipSlot::WalkRight},
               input.walkSpeed, false);
      addLevel({ClipSlot::RunForward, ClipSlot::RunBack, ClipSlot::RunLeft,
                ClipSlot::RunRight},
               input.runSpeed, true);
      if (!input.aiming)
        addLevel({ClipSlot::Sprint, ClipSlot::Sprint, ClipSlot::Sprint,
                  ClipSlot::Sprint},
                 input.sprintSpeed, true);
    }
    std::vector<float> speeds = {0.0f};
    for (const Level &level : levels)
      speeds.push_back(level.speed);
    const std::vector<float> rows = speedBlend(speeds, input.speed);

    // The moving clips share one phase, so their feet stay in step. It moves
    // on by the blended length, at the capsule's speed over what the blended
    // clips cover, so the feet keep up whatever speed a row sits at.
    std::vector<Moving> moving;
    float blendedLength = 0.0f, movingWeight = 0.0f, naturalSpeed = 0.0f;
    float runningShare = 0.0f; // of the moving rows, running ahead
    for (size_t r = 0; r < levels.size(); ++r) {
      if (rows[r + 1] <= 0.0f)
        continue;
      naturalSpeed += rows[r + 1] * levels[r].clipSpeed;
      if (levels[r].running)
        runningShare += rows[r + 1] * dir.forward;
      for (int k = 0; k < 4; ++k) {
        const float w = rows[r + 1] * dirWeights[k];
        if (w <= 0.0f)
          continue;
        if (const auto [used, clip] = resolve(levels[r].slots[k]); clip) {
          moving.push_back({levels[r].slots[k], w});
          blendedLength += w * clip->clip.duration;
          movingWeight += w;
        }
      }
    }
    if (movingWeight > 0.0f && blendedLength > 0.0f) {
      blendedLength /= movingWeight;
      float rate = naturalSpeed > 1e-4f ? input.speed / naturalSpeed : 1.0f;
      // Running ahead, a stride a loop at the cadence people really run at
      // that speed, the stride stretched or shortened to cover the rest:
      // Mixamo's sprint takes a full sprint's quick steps at a jog's speed.
      if (tune.footIk.enabled && tune.footIk.strideWarp && naturalSpeed > 1e-4f &&
          runningShare > 0.0f) {
        const float share = runningShare / movingWeight;
        const float cadenceRate = runningCadence(input.speed) * blendedLength;
        const float wanted = rate + (cadenceRate - rate) * share;
        stride = std::clamp(input.speed / (naturalSpeed * wanted), kShortestStride,
                            kLongestStride);
        rate = input.speed / (naturalSpeed * stride);
      }
      rate = std::clamp(rate, 0.25f, 3.0f);
      phase = std::fmod(phase + dt * rate * tune.playbackRate / blendedLength, 1.0f);
    }
    const ClipSlot still = crouch ? ClipSlot::CrouchIdle
                           : input.aiming && activeStance != Stance::Unarmed ? ClipSlot::AimIdle
                                                                             : ClipSlot::Idle;
    sample(still, idleTime, rows[0]);
    for (const Moving &m : moving)
      if (const auto [used, clip] = resolve(m.slot); clip)
        sample(m.slot, phase * clip->clip.duration, m.weight);
  } else if (active == AnimState::Fall) {
    sample(ClipSlot::Fall, stateTime, 1.0f);
  } else if (active == AnimState::Jump) {
    // The clip follows the arc rather than the clock: its start at takeoff,
    // halfway at the top (no speed up), its end back at takeoff height (as
    // fast down as it went up), then held. Walking off a ledge goes
    // straight to the end.
    if (const auto [used, clip] = resolve(ClipSlot::Jump); clip) {
      const float lift = std::max(takeoffSpeed, 1.0f);
      const float progress =
          std::clamp(0.5f - input.verticalSpeed / (2.0f * lift), 0.0f, 1.0f);
      const float span = std::max(clip->endTime() - clip->start, 0.0f);
      sample(ClipSlot::Jump, clip->start + span * progress, 1.0f);
    }
  } else {
    const auto [used, clip] = resolve(oneShot);
    const float time = clip ? std::min(clip->start + stateTime, clip->endTime())
                            : stateTime;
    sample(oneShot, time, 1.0f);
  }
  Pose statePose = mix.total > 0.0f ? mix.pose : bindPose(skeleton);
  // With a gun but in a move its stance has no clip for (a crouch, a jump,
  // a fall, a landing, a dodge): the unarmed move, with the stance's arms
  // (its aim idle while aiming, else its idle) over it, so the gun is held
  // without the stance's standing spine going over the move's.
  bool fellBack = false;
  if (activeStance != Stance::Unarmed && !arms.empty()) {
    switch (active) {
    case AnimState::Crouch: fellBack = !library.clip(ClipSlot::CrouchIdle, activeStance); break;
    case AnimState::Jump: fellBack = !library.clip(ClipSlot::Jump, activeStance); break;
    case AnimState::Fall: fellBack = !library.clip(ClipSlot::Fall, activeStance); break;
    case AnimState::Land:
    case AnimState::Dodge: fellBack = !library.clip(oneShot, activeStance); break;
    case AnimState::Locomotion: break;
    }
  }
  if (fellBack) {
    const CharacterClipData *held =
        input.aiming ? library.clip(ClipSlot::AimIdle, activeStance) : nullptr;
    if (!held)
      held = library.clip(ClipSlot::Idle, activeStance);
    if (held) {
      Pose holding = bindPose(skeleton);
      sampleClip(held->clip, skeleton, idleTime, held->loop, holding);
      blendPosesMasked(statePose, holding, 1.0f, arms, statePose);
    }
  }
  // A gun carried relaxed in the right hand: off the ground or dodging,
  // that arm keeps the unarmed idle's hang.
  if (input.gunInRightHand && activeStance == Stance::Unarmed && !rightArm.empty() &&
      active != AnimState::Locomotion && active != AnimState::Crouch) {
    if (const CharacterClipData *idle = library.clip(ClipSlot::Idle)) {
      Pose hanging = bindPose(skeleton);
      sampleClip(idle->clip, skeleton, idleTime, idle->loop, hanging);
      blendPosesMasked(statePose, hanging, 1.0f, rightArm, statePose);
    }
  }

  // --- Inertialization (Bollo, GDC 2018): the new state plays in full and
  // the difference to what was showing, and how fast it moved, eases away.
  // Its offsets were taken at the last frame, so this frame is dt in.
  if (blendPending) {
    blendPending = false;
    if (framesShown > 0)
      blend.start(lastBase, framesShown > 1 ? beforeLastBase : lastBase,
                  framesShown > 1 ? lastDt : 0.0f, statePose, fadeTime);
    blendTime = 0.0f;
  }
  blendTime += dt;
  base = statePose;
  blend.apply(blendTime, base);
  // A reload on the upper body, over whatever the legs do, eased in and out.
  if (input.reload >= 0.0f && !upperBody.empty())
    if (const CharacterClipData *clip = library.clip(
            ClipSlot::Reload,
            input.reloadStance != Stance::Unarmed ? input.reloadStance : activeStance)) {
      const float progress = std::clamp(input.reload, 0.0f, 1.0f);
      const float weight =
          std::clamp(std::min(progress, 1.0f - progress) / kReloadEase, 0.0f, 1.0f);
      if (weight > 0.0f) {
        Pose reload = bindPose(skeleton);
        const float span = std::max(clip->endTime() - clip->start, 0.0f);
        sampleClip(clip->clip, skeleton, clip->start + span * progress, false, reload);
        blendPosesMasked(base, reload, weight, upperBody, base);
      }
    }
  beforeLastBase = lastBase;
  lastBase = base;
  lastDt = dt;
  framesShown = std::min(framesShown + 1, 2);
  current = base;
  layerAim(input, dt);
}

void CharacterAnimator::layerAim(const AnimatorInput &input, float dt) {
  const float step = tune.aimFade > 0.0f ? dt / tune.aimFade : 1.0f;
  // Holding a gun, the game aims the gun itself (AimIk.h) instead of leaning.
  const bool leaning = input.aiming && activeStance == Stance::Unarmed;
  leanWeight = std::clamp(leanWeight + (leaning ? step : -step), 0.0f, 1.0f);

  // The spine leans with the camera while aiming. The model faces +z, so
  // looking up turns its chest back about x.
  if (leanWeight > 0.0f && !spine.empty()) {
    const float lean = std::clamp(input.aimPitch, -tune.aimLean, tune.aimLean) * leanWeight;
    turnBones(skeleton, current, spine, glm::vec3(1.0f, 0.0f, 0.0f), -lean);
  }
}
