#include "CharacterAnimator.h"
#include "TestRunner.h"

#include <cmath>
#include <string>

namespace {

constexpr float kDt = 1.0f / 60.0f;
const glm::vec3 kUp(0.0f, 1.0f, 0.0f);

Skeleton rig() {
  Skeleton s;
  Bone root;
  root.name = "root";
  Bone hips;
  hips.name = "hips";
  hips.parent = 0;
  hips.bind.translation = glm::vec3(0.0f, 1.0f, 0.0f);
  s.bones = {root, hips};
  return s;
}

// A clip that turns the hips by its own angle about y, so the pose says
// which clips are playing, and moves them forward at speed (m/s).
CharacterClipData marker(float degrees, float duration, float speed, bool loop) {
  CharacterClipData d;
  d.loop = loop;
  d.clip.duration = duration;
  d.clip.tracks.resize(2);
  const glm::quat turn = glm::angleAxis(glm::radians(degrees), kUp);
  d.clip.tracks[1].rotations = {{0.0f, turn}, {duration, turn}};
  d.clip.tracks[1].positions = {{0.0f, glm::vec3(0, 1, 0)},
                                {duration, glm::vec3(0, 1, speed * duration)}};
  const Skeleton s = rig();
  d.rootMotion = removeRootMotion(
      d.clip, s, 1, loop ? RootMotionMode::Progression : RootMotionMode::Path);
  return d;
}

void put(AnimationLibrary &library, ClipSlot slot, CharacterClipData data) {
  library.clips[static_cast<size_t>(slot)] = std::move(data);
}

AnimationLibrary fullLibrary() {
  AnimationLibrary l;
  put(l, ClipSlot::Idle, marker(0, 2.0f, 0.0f, true));
  put(l, ClipSlot::WalkForward, marker(20, 1.0f, 1.5f, true));
  put(l, ClipSlot::RunForward, marker(40, 0.6f, 4.0f, true));
  put(l, ClipSlot::Sprint, marker(60, 0.5f, 7.0f, true));
  put(l, ClipSlot::WalkRight, marker(100, 1.0f, 1.5f, true));
  put(l, ClipSlot::WalkLeft, marker(105, 1.0f, 1.5f, true));
  put(l, ClipSlot::WalkBack, marker(110, 1.0f, 1.5f, true));
  put(l, ClipSlot::Jump, marker(140, 1.0f, 0.0f, false));
  put(l, ClipSlot::Fall, marker(150, 0.7f, 0.0f, true));
  put(l, ClipSlot::Land, marker(160, 0.5f, 0.0f, false));
  put(l, ClipSlot::LandHard, marker(165, 0.5f, 0.0f, false));
  put(l, ClipSlot::CrouchIdle, marker(170, 1.0f, 0.0f, true));
  put(l, ClipSlot::CrouchForward, marker(175, 1.0f, 1.0f, true));
  put(l, ClipSlot::Roll, marker(120, 1.0f, 5.0f, false));
  put(l, ClipSlot::DodgeLeft, marker(125, 1.0f, 3.0f, false));
  return l;
}

float hipsDegrees(const CharacterAnimator &a) {
  const glm::quat q = a.pose().local[1].rotation;
  return glm::degrees(2.0f * std::acos(std::min(1.0f, std::abs(q.w))));
}

void runFor(CharacterAnimator &a, const AnimatorInput &input, float seconds) {
  for (int i = 0; i < int(std::lround(seconds / kDt)); ++i)
    a.update(input, kDt);
}

AnimatorInput moving(float speed) {
  AnimatorInput in;
  in.speed = speed;
  return in;
}

AnimatorInput inTheAir(float verticalSpeed) {
  AnimatorInput in;
  in.grounded = false;
  in.verticalSpeed = verticalSpeed;
  return in;
}

float fraction(float time, float duration) {
  const float f = std::fmod(time / duration, 1.0f);
  return f < 0.0f ? f + 1.0f : f;
}

} // namespace

TEST(standing_still_plays_idle) {
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, moving(0.0f), 1.0f);
  CHECK(a.state() == AnimState::Locomotion);
  CHECK_NEAR(a.clipWeight(ClipSlot::Idle), 1.0, 1e-5);
  CHECK_NEAR(hipsDegrees(a), 0.0, 0.1);
}

TEST(speeds_between_clips_blend_them) {
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, moving(2.75f), 1.0f); // halfway from walk (1.5) to run (4)
  CHECK_NEAR(a.clipWeight(ClipSlot::WalkForward), 0.5, 1e-3);
  CHECK_NEAR(a.clipWeight(ClipSlot::RunForward), 0.5, 1e-3);
  CHECK_NEAR(hipsDegrees(a), 30.0, 0.5);
}

TEST(clip_speeds_are_scaled_to_the_character) {
  // At half size the walk clip covers 0.75 m/s, so that's a pure walk.
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  CharacterAnimator a(s, l, 0.5f);
  runFor(a, moving(0.75f), 1.0f);
  CHECK_NEAR(a.clipWeight(ClipSlot::WalkForward), 1.0, 1e-3);
}

TEST(walk_and_run_stay_in_step) {
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, moving(2.75f), 1.3f);
  CHECK_NEAR(fraction(a.clipTime(ClipSlot::WalkForward), 1.0f),
             fraction(a.clipTime(ClipSlot::RunForward), 0.6f), 1e-3);
}

TEST(faster_than_the_sprint_clip_plays_it_faster) {
  // Twice the sprint clip's speed: it plays at twice its rate.
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, moving(14.0f), 0.2f);
  CHECK_NEAR(a.clipWeight(ClipSlot::Sprint), 1.0, 1e-3);
  CHECK_NEAR(a.clipTime(ClipSlot::Sprint), 0.4, 0.02);
}

TEST(the_controllers_run_speed_plays_only_the_run) {
  // The run clip covers 4 m/s and the sprint 7, but the controller runs at
  // 5 and sprints at 8: at 5 it's all run, played 5/4 as fast so the feet
  // keep up; at 8 all sprint, 8/7 as fast; halfway, half of each.
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  AnimatorInput run = moving(5.0f);
  run.runSpeed = 5.0f;
  run.sprintSpeed = 8.0f;
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, run, 0.3f);
  CHECK_NEAR(a.clipWeight(ClipSlot::RunForward), 1.0, 1e-3);
  CHECK_NEAR(a.clipWeight(ClipSlot::Sprint), 0.0, 1e-3);
  CHECK_NEAR(a.clipTime(ClipSlot::RunForward), 0.3 * 5.0 / 4.0, 0.01);

  AnimatorInput sprint = run;
  sprint.speed = 8.0f;
  CharacterAnimator b(s, l, 1.0f);
  runFor(b, sprint, 0.3f);
  CHECK_NEAR(b.clipWeight(ClipSlot::Sprint), 1.0, 1e-3);
  CHECK_NEAR(b.clipTime(ClipSlot::Sprint), 0.3 * 8.0 / 7.0, 0.01);

  AnimatorInput between = run;
  between.speed = 6.5f;
  CharacterAnimator c(s, l, 1.0f);
  runFor(c, between, 0.3f);
  CHECK_NEAR(c.clipWeight(ClipSlot::RunForward), 0.5, 1e-3);
  CHECK_NEAR(c.clipWeight(ClipSlot::Sprint), 0.5, 1e-3);
}

TEST(the_controllers_crouch_speed_plays_the_crouch_walk_in_step) {
  // The crouch walk covers 1 m/s; at the controller's 2 it plays twice as
  // fast, and at 1 (half the crouch speed) it's half crouch idle.
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  AnimatorInput crouched = moving(2.0f);
  crouched.crouched = true;
  crouched.crouchSpeed = 2.0f;
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, crouched, 0.3f);
  CHECK_NEAR(a.clipWeight(ClipSlot::CrouchForward), 1.0, 1e-3);
  CHECK_NEAR(a.clipTime(ClipSlot::CrouchForward), 0.3 * 2.0, 0.02);
  crouched.speed = 1.0f;
  runFor(a, crouched, 0.5f);
  CHECK_NEAR(a.clipWeight(ClipSlot::CrouchForward), 0.5, 1e-3);
}

TEST(a_jump_keeps_its_clip_until_it_falls_fast) {
  // A 1.2 m jump on flat ground leaves at about 4.85 m/s and comes down at
  // the same speed: all of it in the jump clip. Only a real fall, faster
  // than fallSpeed, turns into the falling loop.
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, moving(0.0f), 0.5f);
  a.update(inTheAir(4.85f), kDt);
  CHECK(a.state() == AnimState::Jump);
  runFor(a, inTheAir(0.0f), 0.5f);
  CHECK(a.state() == AnimState::Jump);
  runFor(a, inTheAir(-4.85f), 0.5f);
  CHECK(a.state() == AnimState::Jump); // the way down is still the jump
  a.update(inTheAir(-9.0f), kDt);
  CHECK(a.state() == AnimState::Fall);
}

TEST(the_jump_clip_follows_the_arc) {
  // The clip's part from start to end spans the jump: taking off at its
  // start, the top halfway, back at takeoff height at its end, then held.
  const Skeleton s = rig();
  AnimationLibrary l = fullLibrary();
  CharacterClipData jump = marker(140, 1.0f, 0.0f, false);
  jump.start = 0.2f;
  jump.end = 0.8f;
  put(l, ClipSlot::Jump, jump);
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, moving(0.0f), 0.5f);
  a.update(inTheAir(4.0f), kDt);
  CHECK_NEAR(a.clipTime(ClipSlot::Jump), 0.2, 1e-4);
  a.update(inTheAir(0.0f), kDt);
  CHECK_NEAR(a.clipTime(ClipSlot::Jump), 0.5, 1e-4);
  a.update(inTheAir(-4.0f), kDt);
  CHECK_NEAR(a.clipTime(ClipSlot::Jump), 0.8, 1e-4);
  a.update(inTheAir(-6.0f), kDt);
  CHECK_NEAR(a.clipTime(ClipSlot::Jump), 0.8, 1e-4); // held, not falling yet
}

TEST(walking_off_a_ledge_waits_then_holds_the_air_pose) {
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, moving(1.5f), 0.5f);
  runFor(a, inTheAir(-1.0f), 0.1f);
  CHECK(a.state() == AnimState::Locomotion); // a step down
  runFor(a, inTheAir(-3.0f), 0.2f);
  CHECK(a.state() == AnimState::Jump);
  CHECK_NEAR(a.clipTime(ClipSlot::Jump), 1.0, 1e-4); // the jump's last frame
  a.update(inTheAir(-9.0f), kDt);
  CHECK(a.state() == AnimState::Fall);
}

TEST(a_jump_lands_softly_and_a_fall_lands_hard) {
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, moving(0.0f), 0.5f);
  a.update(inTheAir(4.85f), kDt);
  runFor(a, inTheAir(0.0f), 0.5f);
  runFor(a, inTheAir(-4.85f), 0.4f);
  a.update(moving(0.0f), kDt);
  CHECK(a.state() == AnimState::Land);
  runFor(a, moving(0.0f), 0.1f);
  CHECK_NEAR(a.clipWeight(ClipSlot::Land), 1.0, 1e-3);
  CHECK_NEAR(a.clipWeight(ClipSlot::LandHard), 0.0, 1e-3);

  CharacterAnimator b(s, l, 1.0f);
  runFor(b, inTheAir(-9.0f), 1.0f);
  CHECK(b.state() == AnimState::Fall);
  b.update(moving(0.0f), kDt);
  CHECK(b.state() == AnimState::Land);
  runFor(b, moving(0.0f), 0.1f);
  CHECK_NEAR(b.clipWeight(ClipSlot::LandHard), 1.0, 1e-3);
}

TEST(a_hard_landing_without_its_clip_lands_softly) {
  const Skeleton s = rig();
  AnimationLibrary l = fullLibrary();
  l.clips[static_cast<size_t>(ClipSlot::LandHard)].reset();
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, inTheAir(-9.0f), 1.0f);
  a.update(moving(0.0f), kDt);
  CHECK(a.state() == AnimState::Land);
  runFor(a, moving(0.0f), 0.1f);
  CHECK_NEAR(a.clipWeight(ClipSlot::Land), 1.0, 1e-3);
}

TEST(a_landing_plays_from_its_start_to_its_end) {
  // Landing at 0.1 s into the clip and done at 0.3 s: 0.2 s of landing.
  const Skeleton s = rig();
  AnimationLibrary l = fullLibrary();
  CharacterClipData land = marker(160, 0.5f, 0.0f, false);
  land.start = 0.1f;
  land.end = 0.3f;
  put(l, ClipSlot::Land, land);
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, inTheAir(-5.0f), 1.0f);
  a.update(moving(0.0f), kDt);
  CHECK(a.state() == AnimState::Land);
  CHECK_NEAR(a.clipTime(ClipSlot::Land), 0.1 + kDt, 1e-4);
  runFor(a, moving(0.0f), 0.15f);
  CHECK(a.state() == AnimState::Land);
  CHECK(a.clipTime(ClipSlot::Land) <= 0.3f + 1e-4f);
  runFor(a, moving(0.0f), 0.1f);
  CHECK(a.state() == AnimState::Locomotion);
}

TEST(a_landing_plays_then_moves_on) {
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, inTheAir(-5.0f), 1.0f);
  a.update(moving(0.0f), kDt);
  CHECK(a.state() == AnimState::Land);
  runFor(a, moving(0.0f), 0.6f); // the landing clip is 0.5 s
  CHECK(a.state() == AnimState::Locomotion);
}

TEST(a_short_hop_skips_the_landing) {
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, moving(0.0f), 0.5f);
  runFor(a, inTheAir(2.0f), 0.2f);
  a.update(moving(0.0f), kDt);
  CHECK(a.state() == AnimState::Locomotion);
}

TEST(moving_cuts_a_landing_short) {
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, inTheAir(-5.0f), 1.0f);
  a.update(moving(0.0f), kDt);
  CHECK(a.state() == AnimState::Land);
  runFor(a, moving(3.0f), 0.05f);
  CHECK(a.state() == AnimState::Land); // not before landMinTime
  runFor(a, moving(3.0f), 0.1f);
  CHECK(a.state() == AnimState::Locomotion);
}

TEST(crouching_plays_the_crouch_clips) {
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  CharacterAnimator a(s, l, 1.0f);
  AnimatorInput crouched = moving(0.0f);
  crouched.crouched = true;
  runFor(a, crouched, 1.0f);
  CHECK(a.state() == AnimState::Crouch);
  CHECK_NEAR(a.clipWeight(ClipSlot::CrouchIdle), 1.0, 1e-3);
  crouched.speed = 1.0f;
  runFor(a, crouched, 0.5f);
  CHECK_NEAR(a.clipWeight(ClipSlot::CrouchForward), 1.0, 1e-3);
}

TEST(a_dodge_plays_the_clip_the_controller_picked) {
  // The controller picks the dodge (it moves along that clip's path), so
  // the animator plays what it's told, whatever the stick says.
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  CharacterAnimator a(s, l, 1.0f);
  AnimatorInput dodge = moving(5.0f);
  dodge.dodging = true;
  runFor(a, dodge, 0.2f);
  CHECK(a.state() == AnimState::Dodge);
  CHECK_NEAR(a.clipWeight(ClipSlot::Roll), 1.0, 1e-3);
  CharacterAnimator b(s, l, 1.0f);
  dodge.aiming = true;
  dodge.direction = glm::vec2(1.0f, 0.0f);
  dodge.dodgeClip = ClipSlot::DodgeLeft;
  runFor(b, dodge, 0.2f);
  CHECK_NEAR(b.clipWeight(ClipSlot::DodgeLeft), 1.0, 1e-3);
}

TEST(aiming_blends_the_directions) {
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  CharacterAnimator a(s, l, 1.0f);
  AnimatorInput aim = moving(1.5f);
  aim.aiming = true;
  aim.direction = glm::vec2(1.0f, 0.0f);
  runFor(a, aim, 0.5f);
  CHECK_NEAR(a.clipWeight(ClipSlot::WalkRight), 1.0, 1e-3);
  aim.direction = glm::vec2(0.7f, 0.7f);
  runFor(a, aim, 0.5f);
  CHECK_NEAR(a.clipWeight(ClipSlot::WalkRight), 0.5, 1e-3);
  CHECK_NEAR(a.clipWeight(ClipSlot::WalkForward), 0.5, 1e-3);
}

TEST(changing_state_cross_fades) {
  // Idle (0 degrees) to crouch (170) with a 0.2 s fade: 6 steps in, halfway
  // through it, the hips are halfway round.
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  AnimatorTuning tuning;
  tuning.crouchFade = 0.2f;
  CharacterAnimator a(s, l, 1.0f, tuning);
  runFor(a, moving(0.0f), 1.0f);
  AnimatorInput crouched = moving(0.0f);
  crouched.crouched = true;
  runFor(a, crouched, 0.1f);
  CHECK_NEAR(hipsDegrees(a), 85.0, 0.5);
  CHECK_NEAR(a.clipWeight(ClipSlot::CrouchIdle), 0.5, 1e-3);
  runFor(a, crouched, 0.2f);
  CHECK_NEAR(hipsDegrees(a), 170.0, 0.5);
}

TEST(missing_clips_fall_back_without_breaking) {
  // Only an idle (turned 10 degrees, unlike the bind pose): every move
  // still gives a pose, the idle one.
  const Skeleton s = rig();
  AnimationLibrary l;
  put(l, ClipSlot::Idle, marker(10, 2.0f, 0.0f, true));
  CharacterAnimator a(s, l, 1.0f);
  AnimatorInput in = moving(6.0f);
  in.aiming = true;
  in.direction = glm::vec2(-1, 0);
  runFor(a, in, 0.5f);
  in.crouched = true;
  runFor(a, in, 0.5f);
  in.dodging = true;
  runFor(a, in, 0.5f);
  runFor(a, inTheAir(3.0f), 1.0f);
  CHECK_NEAR(hipsDegrees(a), 10.0, 0.1); // falling, on the idle
  runFor(a, moving(0.0f), 1.0f);
  CHECK_NEAR(hipsDegrees(a), 10.0, 0.1);
  // No clips at all: the bind pose.
  const AnimationLibrary empty;
  CharacterAnimator b(s, empty, 1.0f);
  runFor(b, moving(3.0f), 0.5f);
  CHECK_NEAR(hipsDegrees(b), 0.0, 1e-4);
}

TEST(playback_rate_speeds_up_the_moving_clips) {
  // At the walk clip's own speed it plays at its own rate; a playback rate
  // of 2 plays it twice as fast (the feet then outpace the ground).
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  AnimatorTuning tuning;
  tuning.playbackRate = 2.0f;
  CharacterAnimator a(s, l, 1.0f, tuning);
  runFor(a, moving(1.5f), 0.3f);
  CHECK_NEAR(a.clipWeight(ClipSlot::WalkForward), 1.0, 1e-3);
  CHECK_NEAR(a.clipTime(ClipSlot::WalkForward), 0.6, 0.01);
}

TEST(every_state_has_its_own_name) {
  const AnimState states[] = {AnimState::Locomotion, AnimState::Crouch,
                              AnimState::Jump,       AnimState::Fall,
                              AnimState::Land,       AnimState::Dodge};
  for (AnimState a : states) {
    CHECK(std::string(animStateName(a)).size() > 0);
    for (AnimState b : states)
      if (a != b)
        CHECK(std::string(animStateName(a)) != animStateName(b));
  }
}

namespace {

// A Mixamo-named body: hips, a leg, three spine bones and a right arm.
enum PowerBone { kRoot, kHips, kLeg, kSpine, kSpine1, kSpine2, kArm, kForeArm, kHand, kBones };
Skeleton powerRig() {
  Skeleton s;
  auto add = [&](const char *name, int parent, glm::vec3 at) {
    Bone b;
    b.name = name;
    b.parent = parent;
    b.bind.translation = at;
    s.bones.push_back(b);
  };
  add("root", -1, {0, 0, 0});
  add("mixamorig:Hips", kRoot, {0, 1, 0});
  add("mixamorig:LeftUpLeg", kHips, {0.1f, -0.1f, 0});
  add("mixamorig:Spine", kHips, {0, 0.1f, 0});
  add("mixamorig:Spine1", kSpine, {0, 0.15f, 0});
  add("mixamorig:Spine2", kSpine1, {0, 0.15f, 0});
  add("mixamorig:RightArm", kSpine2, {-0.2f, 0.1f, 0});
  add("mixamorig:RightForeArm", kArm, {-0.3f, 0, 0});
  add("mixamorig:RightHand", kForeArm, {-0.25f, 0, 0});
  return s;
}

// A clip that turns one bone of the power rig about x by its angle (and a
// second one, if asked).
CharacterClipData turning(int bone, float degrees, float duration, bool loop, int other = -1,
                          float otherDegrees = 0.0f) {
  CharacterClipData d;
  d.loop = loop;
  d.clip.duration = duration;
  d.clip.tracks.resize(kBones);
  const glm::quat turn = glm::angleAxis(glm::radians(degrees), glm::vec3(1, 0, 0));
  d.clip.tracks[bone].rotations = {{0.0f, turn}, {duration, turn}};
  if (other >= 0) {
    const glm::quat t2 = glm::angleAxis(glm::radians(otherDegrees), glm::vec3(1, 0, 0));
    d.clip.tracks[other].rotations = {{0.0f, t2}, {duration, t2}};
  }
  return d;
}

float degreesOf(const CharacterAnimator &a, int bone) {
  const glm::quat q = a.pose().local[bone].rotation;
  return glm::degrees(2.0f * std::acos(std::min(1.0f, std::abs(q.w))));
}

AnimationLibrary powerLibrary() {
  AnimationLibrary l;
  put(l, ClipSlot::Idle, turning(kLeg, 10, 2.0f, true));
  // The hold turns the spine, and the leg too, which the mask keeps out.
  put(l, ClipSlot::PowerHold, turning(kSpine, 30, 1.0f, true, kLeg, 50));
  return l;
}

AnimatorInput powering(PowerAnim power) {
  AnimatorInput in;
  in.power = power;
  return in;
}

glm::vec3 boneAt(const Skeleton &s, const CharacterAnimator &a, int bone) {
  return glm::vec3(modelSpace(s, a.pose())[bone][3]);
}

} // namespace

TEST(the_powers_play_on_the_upper_body_and_leave_the_legs) {
  const Skeleton s = powerRig();
  const AnimationLibrary l = powerLibrary();
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, powering(PowerAnim::Hold), 1.0f);
  CHECK_NEAR(degreesOf(a, kSpine), 30.0, 0.5);
  CHECK_NEAR(degreesOf(a, kLeg), 10.0, 0.5); // still the idle's
  CHECK_NEAR(degreesOf(a, kHips), 0.0, 0.5);
  CHECK(a.clipWeight(ClipSlot::PowerHold) > 0.99f);
}

TEST(the_power_layer_fades_in_and_out_in_its_fade_time) {
  const Skeleton s = powerRig();
  const AnimationLibrary l = powerLibrary();
  AnimatorTuning t;
  t.powerFade = 0.2f;
  CharacterAnimator a(s, l, 1.0f, t);
  runFor(a, powering(PowerAnim::Hold), 0.1f);
  CHECK_NEAR(degreesOf(a, kSpine), 15.0, 1.5);
  runFor(a, powering(PowerAnim::Hold), 0.1f);
  CHECK_NEAR(degreesOf(a, kSpine), 30.0, 0.5);
  runFor(a, powering(PowerAnim::None), 0.1f);
  CHECK_NEAR(degreesOf(a, kSpine), 15.0, 1.5);
  runFor(a, powering(PowerAnim::None), 0.1f);
  CHECK_NEAR(degreesOf(a, kSpine), 0.0, 0.5);
  CHECK(a.clipWeight(ClipSlot::PowerHold) < 0.01f);
}

TEST(the_lift_runs_into_the_hold) {
  const Skeleton s = powerRig();
  AnimationLibrary l = powerLibrary();
  put(l, ClipSlot::PowerLift, turning(kSpine, 60, 0.5f, false));
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, powering(PowerAnim::Hold), 0.3f);
  CHECK(a.clipWeight(ClipSlot::PowerLift) > 0.99f);
  CHECK_NEAR(degreesOf(a, kSpine), 60.0, 0.5);
  runFor(a, powering(PowerAnim::Hold), 0.5f); // past the lift's end and the fade
  CHECK(a.clipWeight(ClipSlot::PowerHold) > 0.99f);
  CHECK(a.clipWeight(ClipSlot::PowerLift) < 0.01f);
  CHECK_NEAR(degreesOf(a, kSpine), 30.0, 0.5);
}

TEST(the_throw_starts_from_its_offset_and_holds_its_end) {
  const Skeleton s = powerRig();
  AnimationLibrary l = powerLibrary();
  CharacterClipData throwing = turning(kSpine, 45, 2.0f, false);
  throwing.start = 0.5f;
  throwing.end = 1.5f;
  put(l, ClipSlot::PowerThrow, throwing);
  CharacterAnimator a(s, l, 1.0f);
  a.update(powering(PowerAnim::Throw), kDt);
  CHECK(std::fabs(a.clipTime(ClipSlot::PowerThrow) - 0.5f) <= kDt + 1e-4f);
  runFor(a, powering(PowerAnim::Throw), 3.0f);
  CHECK_NEAR(a.clipTime(ClipSlot::PowerThrow), 1.5, 1e-4);
}

TEST(a_power_without_its_clip_plays_nothing) {
  const Skeleton s = powerRig();
  AnimationLibrary l;
  put(l, ClipSlot::Idle, turning(kLeg, 10, 2.0f, true));
  // A walk that turns the spine, its hips going forward at 1.5 m/s.
  CharacterClipData walk = turning(kSpine, 40, 1.0f, true);
  walk.clip.tracks[kHips].positions = {{0.0f, glm::vec3(0, 1, 0)}, {1.0f, glm::vec3(0, 1, 1.5f)}};
  walk.rootMotion = removeRootMotion(walk.clip, s, kHips, RootMotionMode::Progression);
  put(l, ClipSlot::WalkForward, walk);
  CharacterAnimator a(s, l, 1.0f);
  AnimatorInput in = moving(1.5f);
  in.power = PowerAnim::Burst;
  runFor(a, in, 0.5f);
  // The walk's spine, not the idle's laid over it as a stand-in.
  CHECK_NEAR(degreesOf(a, kSpine), 40.0, 0.5);
  CHECK(a.clipWeight(ClipSlot::PowerBurst) == 0.0f);
}

TEST(aiming_leans_the_spine_with_the_camera) {
  const Skeleton s = powerRig();
  const AnimationLibrary l = powerLibrary();
  CharacterAnimator a(s, l, 1.0f);
  AnimatorInput up;
  up.aiming = true;
  up.aimPitch = 0.3f;
  runFor(a, up, 0.5f);
  // The chest's up leans back (the model faces +z) by the camera's pitch.
  const glm::vec3 chestUp = glm::normalize(glm::vec3(modelSpace(s, a.pose())[kSpine2][1]));
  CHECK_NEAR(std::acos(chestUp.y), 0.3, 1e-3);
  CHECK(chestUp.z < 0.0f);
  CHECK_NEAR(degreesOf(a, kLeg), 10.0, 0.5);
  up.aimPitch = 1.4f; // looking far up: no more than aimLean
  runFor(a, up, 0.5f);
  const glm::vec3 capped = glm::normalize(glm::vec3(modelSpace(s, a.pose())[kSpine2][1]));
  CHECK_NEAR(std::acos(capped.y), AnimatorTuning{}.aimLean, 1e-3);
  AnimatorInput relaxed;
  relaxed.aimPitch = 0.3f; // not aiming: upright
  runFor(a, relaxed, 0.5f);
  const glm::vec3 upright = glm::normalize(glm::vec3(modelSpace(s, a.pose())[kSpine2][1]));
  CHECK(upright.y > 0.9999f);
}

TEST(reaching_puts_the_right_hand_toward_the_target) {
  const Skeleton s = powerRig();
  const AnimationLibrary l = powerLibrary();
  CharacterAnimator a(s, l, 1.0f);
  a.update(AnimatorInput{}, kDt);
  const glm::vec3 rest = boneAt(s, a, kHand);
  // Something held out: the hold's clip turns the spine, and the shoulder with it.
  runFor(a, powering(PowerAnim::Hold), 0.5f);
  const glm::vec3 shoulder = boneAt(s, a, kArm);
  AnimatorInput reach = powering(PowerAnim::Hold);
  reach.reaching = true;
  reach.reachTarget = shoulder + glm::vec3(-0.2f, -0.1f, 0.3f); // well in reach
  runFor(a, reach, 0.5f);
  CHECK(glm::distance(boneAt(s, a, kHand), reach.reachTarget) < 1e-3f);
  // Far off it points at it, the elbow a little bent.
  reach.reachTarget = shoulder + glm::vec3(0, 0, 5);
  runFor(a, reach, 0.5f);
  const glm::vec3 toHand = boneAt(s, a, kHand) - shoulder;
  CHECK(glm::dot(glm::normalize(toHand), glm::vec3(0, 0, 1)) > 0.999f);
  CHECK(glm::length(toHand) < 0.55f - 1e-3f); // the arm is 0.55 m: not locked straight
  // Let go and it comes back.
  runFor(a, AnimatorInput{}, 0.5f);
  CHECK(glm::distance(boneAt(s, a, kHand), rest) < 1e-3f);
}

TEST(aiming_through_a_change_of_state_keeps_the_lean_steady) {
  const Skeleton s = powerRig();
  AnimationLibrary l = powerLibrary();
  put(l, ClipSlot::CrouchIdle, turning(kLeg, 20, 1.0f, true));
  CharacterAnimator a(s, l, 1.0f);
  AnimatorInput aim;
  aim.aiming = true;
  aim.aimPitch = 0.5f;
  runFor(a, aim, 0.5f);
  aim.crouched = true; // a new state, its cross-fade starting
  a.update(aim, kDt);
  const glm::vec3 chestUp = glm::normalize(glm::vec3(modelSpace(s, a.pose())[kSpine2][1]));
  CHECK_NEAR(std::acos(chestUp.y), 0.5, 0.01);
}

TEST(a_throw_lets_the_arm_go_with_its_clip) {
  const Skeleton s = powerRig();
  const AnimationLibrary l = powerLibrary();
  AnimatorTuning t;
  t.powerFade = 0.2f;
  CharacterAnimator a(s, l, 1.0f, t);
  a.update(AnimatorInput{}, kDt);
  const glm::vec3 rest = boneAt(s, a, kHand);
  AnimatorInput hold = powering(PowerAnim::Hold);
  hold.reaching = true;
  hold.reachTarget = boneAt(s, a, kArm) + glm::vec3(-0.2f, -0.1f, 0.3f);
  runFor(a, hold, 0.5f);
  CHECK(glm::distance(boneAt(s, a, kHand), hold.reachTarget) < 1e-3f);
  // Thrown: the arm follows the throw's clip from the click, not the spot
  // the thing was held at until it leaves.
  AnimatorInput thrown = hold;
  thrown.power = PowerAnim::Throw;
  runFor(a, thrown, 0.2f);
  CHECK(glm::distance(boneAt(s, a, kHand), rest) < 1e-3f);
}

TEST(a_power_whose_clip_is_missing_fades_the_layer_out) {
  const Skeleton s = powerRig();
  const AnimationLibrary l = powerLibrary(); // a hold, no throw
  AnimatorTuning t;
  t.powerFade = 0.2f;
  CharacterAnimator a(s, l, 1.0f, t);
  runFor(a, powering(PowerAnim::Hold), 0.5f);
  a.update(powering(PowerAnim::Throw), kDt);
  CHECK(degreesOf(a, kSpine) > 25.0f); // still mostly the hold, not dropped
  runFor(a, powering(PowerAnim::Throw), 0.2f);
  CHECK_NEAR(degreesOf(a, kSpine), 0.0, 0.5);
}
