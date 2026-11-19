#pragma once

#include "AnimationClip.h"
#include "CharacterMovement.h"
#include "CharacterFile.h"
#include "FootIk.h"
#include "Inertialize.h"
#include "PoseBlend.h"
#include "Skeleton.h"

#include <algorithm>
#include <array>
#include <optional>

// The character's animation state machine: what the controller is doing in,
// a pose out. No Vulkan, no files, tested like the rest of heliora_core.

struct CharacterClipData {
  AnimationClip clip;
  RootMotion rootMotion; // what came out of the hips
  float start = 0.0f;    // seconds into the clip to start at
  float end = 0.0f;      // and to stop at; 0 is the clip's end
  bool loop = true;
  // Where it stops, in the clip: end, or the clip's length without one.
  float endTime() const {
    return end > start ? std::min(end, clip.duration) : clip.duration;
  }
};

// A clip for each move, or none.
struct AnimationLibrary {
  std::array<std::optional<CharacterClipData>, static_cast<size_t>(ClipSlot::Count)>
      clips;
  const CharacterClipData *clip(ClipSlot slot) const {
    const auto &c = clips[static_cast<size_t>(slot)];
    return c ? &*c : nullptr;
  }
};

// What the controller is doing this frame.
struct AnimatorInput {
  float speed = 0.0f; // m/s along the ground
  // Where it moves in its own frame (x to its right, y ahead); only the
  // direction matters. Steers the directional clips while aiming and
  // picks the dodge.
  glm::vec2 direction{0.0f, 1.0f};
  bool grounded = true;
  bool crouched = false;
  bool aiming = false;
  bool dodging = false;
  float verticalSpeed = 0.0f; // m/s, up is positive
  // The controller's gaits, m/s: the run, sprint and crouch walk play in
  // full at these, faster or slower than their own speed so the feet keep
  // up. 0 leaves a gait at its clip's own speed.
  // The dodge the controller picked (it moves along that clip's path).
  ClipSlot dodgeClip = ClipSlot::Roll;
  float runSpeed = 0.0f;
  float sprintSpeed = 0.0f;
  float crouchSpeed = 0.0f;
  float aimPitch = 0.0f; // radians the camera looks up; the spine leans with it while aiming
};

enum class AnimState { Locomotion, Crouch, Jump, Fall, Land, Dodge };

// Strides a second (two steps each) people run at, by speed in m/s:
// Mercer, Vance, Hreljac & Hamill 2002 (Eur J Appl Physiol 87, Table 1),
// eight men on a treadmill from 3.2 to 6.4 m/s, joined in straight lines.
// Slower holds the slowest (the stride shortens instead); faster carries on
// the last line's rise, up to 2.5 (about an elite sprinter's, est.).
float runningCadence(float speed);

// The state's name, for the editor: "Locomotion", "Crouch" and so on.
const char *animStateName(AnimState state);

// The clip each of the controller's dodges plays.
inline ClipSlot dodgeClip(DodgeKind kind) {
  switch (kind) {
  case DodgeKind::Forward: return ClipSlot::DodgeForward;
  case DodgeKind::Back: return ClipSlot::DodgeBack;
  case DodgeKind::Left: return ClipSlot::DodgeLeft;
  case DodgeKind::Right: return ClipSlot::DodgeRight;
  default: return ClipSlot::Roll;
  }
}

struct AnimatorTuning {
  // Blend times into each state, seconds: how long the pose that was
  // showing takes to ease into the new one (inertialization).
  float locomotionFade = 0.2f;
  float crouchFade = 0.25f;
  float jumpFade = 0.08f;
  float fallFade = 0.25f;
  float landFade = 0.08f;
  float dodgeFade = 0.06f;
  // In the air without a jump (walking off a ledge) this long before the
  // air pose, so a step down doesn't flicker into it.
  float fallDelay = 0.2f;
  // Falling faster than this, m/s, is a real fall: the falling loop, then
  // the hard landing. A jump on flat ground never gets there (1.2 m up
  // comes back down at about 4.9 m/s); a drop of about 3 m does.
  float fallSpeed = 7.5f;
  // Landings after less time in the air than this skip the landing clip.
  float landAirTime = 0.35f;
  // Moving cuts a landing short, but not before this.
  float landMinTime = 0.12f;
  // The moving clips play this much faster than the speed calls for (1 has
  // the feet keep up with the ground).
  float playbackRate = 1.0f;
  // The aim's lean fades in and out over this, seconds.
  float aimFade = 0.12f;
  // Radians the spine leans with the camera at most, up or down (40 degrees).
  float aimLean = 0.7f;
  // Feet on uneven ground, laid on by the game after the animator (it casts
  // the rays).
  FootIkTuning footIk;
  bool operator==(const AnimatorTuning &) const = default;
};

class CharacterAnimator {
public:
  // unitsToMetres turns the clips' units into the character's metres (the
  // model's scale to the capsule), so clip speeds compare with the capsule's.
  CharacterAnimator(const Skeleton &skeleton, const AnimationLibrary &clips,
                    float unitsToMetres, const AnimatorTuning &tuning = {});

  void update(const AnimatorInput &input, float dt);

  const Pose &pose() const { return current; }
  AnimState state() const { return active; }
  // A clip's share of the state's pose, 0 to 1. A new state's clips play in
  // full from its first frame; the blend eases the difference away on top.
  float clipWeight(ClipSlot slot) const;
  // How far into a clip it is, seconds (the walk, run and sprint stay in
  // step, so their times are the same fraction of their lengths).
  float clipTime(ClipSlot slot) const;
  // Running, the clips play at runningCadence and each stride is this many
  // times as long as the clip's, along strideAxis (the skeleton's space),
  // so the feet still keep up with the ground; foot IK does the stretching.
  // 1 when walking, crouching, in the air, or with stride warping off.
  float strideScale() const { return stride; }
  glm::vec3 strideAxis() const { return axis; }
  void setTuning(const AnimatorTuning &t) { tune = t; }

private:
  // On top of the state's pose: the spine's lean with the aim.
  void layerAim(const AnimatorInput &input, float dt);

  const Skeleton &skeleton;
  const AnimationLibrary &library;
  float unitsToMetres;
  AnimatorTuning tune;

  AnimState active = AnimState::Locomotion;
  float stateTime = 0.0f; // seconds in the active state
  float airTime = 0.0f;   // seconds off the ground
  float lastAirTime = 0.0f; // how long the last time in the air lasted
  bool wasGrounded = true;
  bool jumped = false;    // this time in the air began with a jump
  float takeoffSpeed = 0.0f; // m/s up when it left the ground
  float idleTime = 0.0f;  // idle and crouch idle run on their own
  float phase = 0.0f;     // 0 to 1, shared by the moving loops
  float stride = 1.0f;    // see strideScale
  glm::vec3 axis{0.0f, 0.0f, 1.0f};
  ClipSlot oneShot = ClipSlot::Jump; // the clip a Jump, Land or Dodge plays

  Pose current;           // the state's pose with the aim's lean on top
  Pose base;              // the state's pose alone, its blend included
  // The last change's inertialization: the offsets from the pose that was
  // showing, eased away over fadeTime.
  PoseInertialization blend;
  float fadeTime = 0.0f;   // how long that change's blend takes
  float blendTime = 0.0f;  // seconds since it started
  bool blendPending = false; // a change this frame, its blend to start
  Pose lastBase;           // base one frame ago
  Pose beforeLastBase;     // and two
  float lastDt = 0.0f;     // seconds between those two
  int framesShown = 0;     // up to 2: how many of those there are
  std::array<float, static_cast<size_t>(ClipSlot::Count)> weights{};
  std::array<float, static_cast<size_t>(ClipSlot::Count)> times{};

  std::vector<int> spine;  // the spine bones the aim leans
  float leanWeight = 0.0f; // the lean's share, faded in while aiming
};
