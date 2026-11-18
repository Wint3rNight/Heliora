#include "CharacterAnimator.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string_view>
#include <vector>

namespace {

// The hand reaches no further than this share of the arm's length, so the
// elbow stays a little bent pointing at something far off.
constexpr float kReachBend = 0.92f;

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

} // namespace

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
      tune(tuning), current(bindPose(skeleton)), fadeFrom(current) {
  base = current;
  powerPose = current;
  powerFrom = current;
  if (const int first = boneNamed(skeleton, "Spine"); first >= 0) {
    upperBody = upperBodyMask(skeleton, first);
    for (const char *name : {"Spine", "Spine1", "Spine2"})
      if (const int bone = boneNamed(skeleton, name); bone >= 0)
        spine.push_back(bone);
  }
  arm = boneNamed(skeleton, "RightArm");
  foreArm = boneNamed(skeleton, "RightForeArm");
  hand = boneNamed(skeleton, "RightHand");
}

float CharacterAnimator::clipWeight(ClipSlot slot) const {
  return weights[index(slot)];
}

float CharacterAnimator::clipTime(ClipSlot slot) const {
  return times[index(slot)];
}

void CharacterAnimator::update(const AnimatorInput &input, float dt) {
  // The clip a move plays: its own, or the nearest one there is.
  auto resolve = [&](ClipSlot slot) -> std::pair<ClipSlot, const CharacterClipData *> {
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
    // From the state's own pose: the powers' layers fade by themselves, and
    // baked in here they'd be laid on twice through the cross-fade.
    fadeFrom = base;
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
    };
    const std::array<float, 4> dirWeights = {dir.forward, dir.back, dir.left,
                                             dir.right};
    std::vector<Level> levels;
    auto addLevel = [&](std::array<ClipSlot, 4> slots, float gaitSpeed) {
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
        levels.push_back({slots, speed, own});
    };
    if (crouch) {
      addLevel({ClipSlot::CrouchForward, ClipSlot::CrouchBack, ClipSlot::CrouchLeft,
                ClipSlot::CrouchRight},
               input.crouchSpeed);
    } else {
      addLevel({ClipSlot::WalkForward, ClipSlot::WalkBack, ClipSlot::WalkLeft,
                ClipSlot::WalkRight},
               0.0f);
      addLevel({ClipSlot::RunForward, ClipSlot::RunBack, ClipSlot::RunLeft,
                ClipSlot::RunRight},
               input.runSpeed);
      if (!input.aiming)
        addLevel({ClipSlot::Sprint, ClipSlot::Sprint, ClipSlot::Sprint,
                  ClipSlot::Sprint},
                 input.sprintSpeed);
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
    for (size_t r = 0; r < levels.size(); ++r) {
      if (rows[r + 1] <= 0.0f)
        continue;
      naturalSpeed += rows[r + 1] * levels[r].clipSpeed;
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
      const float rate = naturalSpeed > 1e-4f
                             ? std::clamp(input.speed / naturalSpeed, 0.25f, 3.0f)
                             : 1.0f;
      phase = std::fmod(phase + dt * rate * tune.playbackRate / blendedLength, 1.0f);
    }
    sample(crouch ? ClipSlot::CrouchIdle : ClipSlot::Idle, idleTime, rows[0]);
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
  const Pose statePose = mix.total > 0.0f ? mix.pose : bindPose(skeleton);

  // --- Cross-fade from the pose the last change left.
  const float t = fadeTime > 0.0f ? std::min(stateTime / fadeTime, 1.0f) : 1.0f;
  if (t < 1.0f) {
    blendPoses(fadeFrom, statePose, t, base);
    for (float &w : weights)
      w *= t;
  } else {
    base = statePose;
  }
  current = base;
  layerPowers(input, dt);
}

void CharacterAnimator::startPower(ClipSlot slot) {
  // Without its clip the layer fades out on what it was playing.
  const CharacterClipData *clip = library.clip(slot);
  powerMissing = !clip;
  if (!clip)
    return;
  // From what the layer shows, if it shows anything.
  powerFromSlot = powerWeight > 0.0f ? powerSlot : ClipSlot::Count;
  powerFrom = powerPose;
  powerSwitch = powerFromSlot == ClipSlot::Count ? tune.powerFade : 0.0f;
  powerSlot = slot;
  powerTime = clip->start;
}

void CharacterAnimator::layerPowers(const AnimatorInput &input, float dt) {
  const float step = tune.powerFade > 0.0f ? dt / tune.powerFade : 1.0f;
  const auto toward = [&](float value, bool on) {
    return std::clamp(value + (on ? step : -step), 0.0f, 1.0f);
  };

  // --- The upper body's clip: the lift running into the hold, a throw, a
  // burst. Only their own clips: a stand-in like the idle would lay a second
  // copy of the legs' pose over the top.
  if (input.power != powerWas) {
    switch (input.power) {
    case PowerAnim::Hold:
      startPower(library.clip(ClipSlot::PowerLift) ? ClipSlot::PowerLift : ClipSlot::PowerHold);
      break;
    case PowerAnim::Throw: startPower(ClipSlot::PowerThrow); break;
    case PowerAnim::Burst: startPower(ClipSlot::PowerBurst); break;
    case PowerAnim::None: break; // the last clip plays on while it fades out
    }
    powerWas = input.power;
  } else {
    powerTime += dt;
  }
  powerSwitch += dt;
  if (powerSlot == ClipSlot::PowerLift && input.power == PowerAnim::Hold) {
    const CharacterClipData *lift = library.clip(ClipSlot::PowerLift);
    if (!lift || powerTime >= lift->endTime())
      startPower(ClipSlot::PowerHold);
  }
  const CharacterClipData *clip =
      powerSlot != ClipSlot::Count ? library.clip(powerSlot) : nullptr;
  powerWeight = toward(powerWeight, input.power != PowerAnim::None && !powerMissing && clip &&
                                        !upperBody.empty());
  if (clip && powerWeight > 0.0f) {
    if (clip->loop && clip->clip.duration > 0.0f)
      powerTime = std::fmod(powerTime, clip->clip.duration);
    const float time = clip->loop ? powerTime : std::min(powerTime, clip->endTime());
    Pose pose = bindPose(skeleton);
    sampleClip(clip->clip, skeleton, time, clip->loop, pose);
    const float t = tune.powerFade > 0.0f ? std::min(powerSwitch / tune.powerFade, 1.0f) : 1.0f;
    if (t < 1.0f)
      blendPoses(powerFrom, pose, t, pose);
    powerPose = pose;
    blendPosesMasked(current, powerPose, powerWeight, upperBody, current);
    weights[index(powerSlot)] = powerWeight * t;
    times[index(powerSlot)] = time;
    if (t < 1.0f && powerFromSlot != ClipSlot::Count && powerFromSlot != powerSlot)
      weights[index(powerFromSlot)] = powerWeight * (1.0f - t);
  }

  // --- The spine leans with the camera while aiming. The model faces +z, so
  // looking up turns its chest back about x.
  leanWeight = toward(leanWeight, input.aiming);
  if (leanWeight > 0.0f && !spine.empty()) {
    const float lean = std::clamp(input.aimPitch, -tune.aimLean, tune.aimLean) * leanWeight;
    turnBones(skeleton, current, spine, glm::vec3(1.0f, 0.0f, 0.0f), -lean);
  }

  // --- The right arm reaches toward what the power has hold of.
  // Only while something is held out: a throw's arm follows its clip from
  // the click, fading from the reach as the layer fades into the throw.
  const bool reaching = input.reaching && input.power == PowerAnim::Hold;
  reachWeight = toward(reachWeight, reaching);
  if (reaching)
    reachTarget = input.reachTarget;
  if (reachWeight > 0.0f && arm >= 0 && foreArm >= 0 && hand >= 0) {
    const std::vector<glm::mat4> model = modelSpace(skeleton, current);
    const glm::vec3 shoulder(model[arm][3]), elbow(model[foreArm][3]), wrist(model[hand][3]);
    const float far =
        kReachBend * (glm::distance(elbow, shoulder) + glm::distance(wrist, elbow));
    glm::vec3 target = reachTarget;
    if (glm::distance(target, shoulder) > far)
      target = shoulder + glm::normalize(target - shoulder) * far;
    Pose reached = current;
    // The elbow down and out: the model faces +z, so its right is -x.
    twoBoneIk(skeleton, reached, arm, foreArm, hand, target, glm::vec3(-0.7f, -0.7f, 0.0f));
    blendPoses(current, reached, reachWeight, current);
  }
}
