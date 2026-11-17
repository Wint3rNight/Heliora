#include "CharacterAnimator.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

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

CharacterAnimator::CharacterAnimator(const Skeleton &skeleton,
                                     const AnimationLibrary &clips,
                                     float unitsToMetres, const AnimatorTuning &tuning)
    : skeleton(skeleton), library(clips), unitsToMetres(unitsToMetres),
      tune(tuning), current(bindPose(skeleton)), fadeFrom(current) {}

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
    fadeFrom = current;
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
    else if (next == AnimState::Dodge) {
      // The roll; while aiming, a step the way it's pressed.
      const glm::vec2 d = input.direction;
      oneShot = !input.aiming ? ClipSlot::Roll
                : std::abs(d.x) > std::abs(d.y)
                    ? (d.x > 0.0f ? ClipSlot::DodgeRight : ClipSlot::DodgeLeft)
                    : (d.y < 0.0f ? ClipSlot::DodgeBack : ClipSlot::DodgeForward);
    }
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
      phase = std::fmod(phase + dt * rate / blendedLength, 1.0f);
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
    blendPoses(fadeFrom, statePose, t, current);
    for (float &w : weights)
      w *= t;
  } else {
    current = statePose;
  }
}
