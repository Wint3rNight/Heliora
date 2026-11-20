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

// Moving clips played at the speed alone, as without foot IK: no stride
// warping, so no running cadence either.
AnimatorTuning speedMatched() {
  AnimatorTuning t;
  t.footIk.strideWarp = false;
  return t;
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
  CharacterAnimator a(s, l, 1.0f, speedMatched());
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
  CharacterAnimator a(s, l, 1.0f, speedMatched());
  runFor(a, run, 0.3f);
  CHECK_NEAR(a.clipWeight(ClipSlot::RunForward), 1.0, 1e-3);
  CHECK_NEAR(a.clipWeight(ClipSlot::Sprint), 0.0, 1e-3);
  CHECK_NEAR(a.clipTime(ClipSlot::RunForward), 0.3 * 5.0 / 4.0, 0.01);

  AnimatorInput sprint = run;
  sprint.speed = 8.0f;
  CharacterAnimator b(s, l, 1.0f, speedMatched());
  runFor(b, sprint, 0.3f);
  CHECK_NEAR(b.clipWeight(ClipSlot::Sprint), 1.0, 1e-3);
  CHECK_NEAR(b.clipTime(ClipSlot::Sprint), 0.3 * 8.0 / 7.0, 0.01);

  AnimatorInput between = run;
  between.speed = 6.5f;
  CharacterAnimator c(s, l, 1.0f, speedMatched());
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

TEST(changing_state_inertializes) {
  // Idle (0 degrees) to crouch (170) with a 0.2 s blend: the crouch plays
  // in full at once and the hips ease round from where they were, with no
  // jump on the first frame, half way at 0.1 s (from standing still, the
  // ease is 1 - smootherstep), never past 170, and there at 0.2 s.
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  AnimatorTuning tuning;
  tuning.crouchFade = 0.2f;
  CharacterAnimator a(s, l, 1.0f, tuning);
  runFor(a, moving(0.0f), 1.0f);
  AnimatorInput crouched = moving(0.0f);
  crouched.crouched = true;
  a.update(crouched, kDt);
  CHECK(hipsDegrees(a) < 2.0f);
  CHECK_NEAR(a.clipWeight(ClipSlot::CrouchIdle), 1.0, 1e-5);
  CHECK_NEAR(a.clipWeight(ClipSlot::Idle), 0.0, 1e-5);
  float last = hipsDegrees(a);
  for (int i = 1; i < 6; ++i) {
    a.update(crouched, kDt);
    CHECK(hipsDegrees(a) >= last);
    last = hipsDegrees(a);
  }
  CHECK_NEAR(hipsDegrees(a), 85.0, 0.5);
  for (int i = 0; i < 6; ++i) {
    a.update(crouched, kDt);
    CHECK(hipsDegrees(a) <= 170.0f + 1e-3f);
  }
  CHECK_NEAR(hipsDegrees(a), 170.0, 0.05);
}

TEST(a_change_of_state_mid_blend_carries_on_smoothly) {
  // Crouch, then stand again half way through: the hips turn back from
  // where they are, with no jump. They were turning away from standing, and
  // moving away counts as still (Bollo), so the first frame barely moves.
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  AnimatorTuning tuning;
  tuning.crouchFade = 0.2f;
  tuning.locomotionFade = 0.2f;
  CharacterAnimator a(s, l, 1.0f, tuning);
  runFor(a, moving(0.0f), 1.0f);
  AnimatorInput crouched = moving(0.0f);
  crouched.crouched = true;
  runFor(a, crouched, 0.1f);
  float last = hipsDegrees(a);
  a.update(moving(0.0f), kDt);
  CHECK(std::abs(hipsDegrees(a) - last) < 2.0f);
  for (int i = 0; i < 20; ++i) {
    last = hipsDegrees(a);
    a.update(moving(0.0f), kDt);
    CHECK(std::abs(hipsDegrees(a) - last) < 30.0f);
  }
  CHECK_NEAR(hipsDegrees(a), 0.0, 0.05);
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

// A Mixamo-named body: hips, a leg and three spine bones.
enum RigBone { kRoot, kHips, kLeg, kSpine, kSpine1, kSpine2, kBones };
Skeleton leanRig() {
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
  return s;
}

// A looping clip that turns one bone of the rig about x by its angle.
CharacterClipData turning(int bone, float degrees, float duration) {
  CharacterClipData d;
  d.loop = true;
  d.clip.duration = duration;
  d.clip.tracks.resize(kBones);
  const glm::quat turn = glm::angleAxis(glm::radians(degrees), glm::vec3(1, 0, 0));
  d.clip.tracks[bone].rotations = {{0.0f, turn}, {duration, turn}};
  return d;
}

float degreesOf(const CharacterAnimator &a, int bone) {
  const glm::quat q = a.pose().local[bone].rotation;
  return glm::degrees(2.0f * std::acos(std::min(1.0f, std::abs(q.w))));
}

AnimationLibrary leanLibrary() {
  AnimationLibrary l;
  put(l, ClipSlot::Idle, turning(kLeg, 10, 2.0f));
  return l;
}

// How far the chest leans back from upright, radians.
float chestLean(const Skeleton &s, const CharacterAnimator &a) {
  const glm::vec3 chestUp = glm::normalize(glm::vec3(modelSpace(s, a.pose())[kSpine2][1]));
  return std::acos(std::min(1.0f, chestUp.y));
}

} // namespace

TEST(aiming_leans_the_spine_with_the_camera) {
  const Skeleton s = leanRig();
  const AnimationLibrary l = leanLibrary();
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
  CHECK_NEAR(chestLean(s, a), AnimatorTuning{}.aimLean, 1e-3);
  AnimatorInput relaxed;
  relaxed.aimPitch = 0.3f; // not aiming: upright
  runFor(a, relaxed, 0.5f);
  CHECK(chestLean(s, a) < 1e-3f);
}

TEST(the_aim_lean_fades_in_and_out_in_its_fade_time) {
  const Skeleton s = leanRig();
  const AnimationLibrary l = leanLibrary();
  AnimatorTuning t;
  t.aimFade = 0.2f;
  CharacterAnimator a(s, l, 1.0f, t);
  AnimatorInput aim;
  aim.aiming = true;
  aim.aimPitch = 0.4f;
  runFor(a, aim, 0.1f); // half way in
  CHECK_NEAR(chestLean(s, a), 0.2, 0.02);
  runFor(a, aim, 0.15f);
  CHECK_NEAR(chestLean(s, a), 0.4, 1e-3);
  AnimatorInput relaxed;
  relaxed.aimPitch = 0.4f;
  runFor(a, relaxed, 0.1f); // half way out
  CHECK_NEAR(chestLean(s, a), 0.2, 0.02);
  runFor(a, relaxed, 0.15f);
  CHECK(chestLean(s, a) < 1e-3f);
}

TEST(aiming_through_a_change_of_state_keeps_the_lean_steady) {
  const Skeleton s = leanRig();
  AnimationLibrary l = leanLibrary();
  put(l, ClipSlot::CrouchIdle, turning(kLeg, 20, 1.0f));
  CharacterAnimator a(s, l, 1.0f);
  AnimatorInput aim;
  aim.aiming = true;
  aim.aimPitch = 0.5f;
  runFor(a, aim, 0.5f);
  aim.crouched = true; // a new state, its cross-fade starting
  a.update(aim, kDt);
  CHECK_NEAR(chestLean(s, a), 0.5, 0.01);
}

TEST(running_cadence_follows_the_measured_table) {
  // Mercer et al. 2002, Table 1: strides a second against m/s, treadmill.
  CHECK_NEAR(runningCadence(3.2f), 1.33, 1e-4);
  CHECK_NEAR(runningCadence(4.5f), 1.45, 1e-4);
  CHECK_NEAR(runningCadence(5.7f), 1.63, 1e-4);
  CHECK_NEAR(runningCadence(6.4f), 1.73, 1e-4);
  CHECK_NEAR(runningCadence(5.4f), 1.575, 1e-4); // half way from 5.1 to 5.7
  // A slow jog keeps the slowest cadence (the stride shortens); past the
  // table it carries on rising, up to an elite sprinter's.
  CHECK_NEAR(runningCadence(2.0f), 1.33, 1e-4);
  CHECK(runningCadence(7.5f) > 1.73f);
  CHECK_NEAR(runningCadence(30.0f), 2.5, 1e-4);
}

TEST(a_run_plays_at_the_cadence_people_run_at_and_the_stride_makes_up_the_rest) {
  // A sprint clip that covers 6 m/s in a 0.5 s stride, run at 5.7 m/s:
  // 1.63 strides a second (the table), not the 1.9 its own speed would
  // give, and each stride 5.7 / (6 x 1.63 x 0.5) longer.
  const Skeleton s = rig();
  AnimationLibrary l = fullLibrary();
  put(l, ClipSlot::Sprint, marker(60, 0.5f, 6.0f, true));
  AnimatorInput sprint = moving(5.7f);
  sprint.runSpeed = 4.0f;
  sprint.sprintSpeed = 5.7f;
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, sprint, 0.3f);
  CHECK_NEAR(a.clipWeight(ClipSlot::Sprint), 1.0, 1e-3);
  CHECK_NEAR(a.clipTime(ClipSlot::Sprint), 0.3 * 1.63 * 0.5, 0.005);
  CHECK_NEAR(a.strideScale(), 5.7 / (6.0 * 1.63 * 0.5), 2e-3);
  CHECK(glm::distance(a.strideAxis(), glm::vec3(0.0f, 0.0f, 1.0f)) < 1e-5f);
}

TEST(without_stride_warping_a_run_plays_at_its_speed) {
  // Off, or with foot IK off: the clip keeps up with the ground by its rate.
  const Skeleton s = rig();
  AnimationLibrary l = fullLibrary();
  put(l, ClipSlot::Sprint, marker(60, 0.5f, 6.0f, true));
  AnimatorInput sprint = moving(5.7f);
  sprint.runSpeed = 4.0f;
  sprint.sprintSpeed = 5.7f;
  AnimatorTuning noIk;
  noIk.footIk.enabled = false;
  for (const AnimatorTuning &t : {speedMatched(), noIk}) {
    CharacterAnimator a(s, l, 1.0f, t);
    runFor(a, sprint, 0.3f);
    CHECK_NEAR(a.clipTime(ClipSlot::Sprint), 0.3 * 5.7 / 6.0, 0.005);
    CHECK_NEAR(a.strideScale(), 1.0, 1e-5);
  }
}

TEST(walking_crouching_and_running_sideways_keep_their_own_stride) {
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  CharacterAnimator walk(s, l, 1.0f);
  runFor(walk, moving(1.5f), 0.3f);
  CHECK_NEAR(walk.strideScale(), 1.0, 1e-5);
  AnimatorInput crouched = moving(1.0f);
  crouched.crouched = true;
  CharacterAnimator crouch(s, l, 1.0f);
  runFor(crouch, crouched, 0.5f);
  CHECK_NEAR(crouch.strideScale(), 1.0, 1e-5);
  // Aiming, running to the right: no cadence data for that.
  AnimatorInput strafe = moving(4.0f);
  strafe.aiming = true;
  strafe.direction = glm::vec2(1.0f, 0.0f);
  CharacterAnimator side(s, l, 1.0f);
  runFor(side, strafe, 0.3f);
  CHECK_NEAR(side.strideScale(), 1.0, 1e-5);
  CHECK(glm::distance(side.strideAxis(), glm::vec3(-1.0f, 0.0f, 0.0f)) < 1e-5f);
}

TEST(the_stride_stretches_no_further_than_its_limit) {
  // 14 m/s on a 7 m/s sprint: 2.5 strides a second would need each 1.6
  // times as long; it stops at 1.3 and the clip plays faster for the rest.
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, moving(14.0f), 0.2f);
  CHECK_NEAR(a.strideScale(), 1.3, 1e-4);
  CHECK_NEAR(a.clipTime(ClipSlot::Sprint), 0.2 * 14.0 / (7.0 * 1.3), 0.01);
}

TEST(the_controllers_walk_speed_plays_only_the_walk) {
  // The walk clip covers 1.5 m/s; walking at the controller's 1.2 it's all
  // walk, played 1.2 / 1.5 as fast.
  const Skeleton s = rig();
  const AnimationLibrary l = fullLibrary();
  AnimatorInput walk = moving(1.2f);
  walk.walkSpeed = 1.2f;
  walk.runSpeed = 4.0f;
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, walk, 0.5f);
  CHECK_NEAR(a.clipWeight(ClipSlot::WalkForward), 1.0, 1e-3);
  CHECK_NEAR(a.clipTime(ClipSlot::WalkForward), 0.5 * 1.2 / 1.5, 0.01);
}

namespace {

void putIn(AnimationLibrary &library, Stance stance, ClipSlot slot, CharacterClipData data) {
  library.stanceClips[static_cast<size_t>(stance) - 1][static_cast<size_t>(slot)] = std::move(data);
}

AnimatorInput inStance(Stance stance, float speed = 0.0f) {
  AnimatorInput in = moving(speed);
  in.stance = stance;
  return in;
}

} // namespace

TEST(a_gun_stance_plays_its_own_clips) {
  const Skeleton s = rig();
  AnimationLibrary l = fullLibrary();
  putIn(l, Stance::Rifle, ClipSlot::Idle, marker(30, 2.0f, 0.0f, true));
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, inStance(Stance::Rifle), 1.0f);
  CHECK_NEAR(hipsDegrees(a), 30.0, 0.5);
  runFor(a, inStance(Stance::Unarmed), 1.0f);
  CHECK_NEAR(hipsDegrees(a), 0.0, 0.5);
}

TEST(a_stance_missing_a_move_keeps_to_its_own_moves_then_the_unarmed_ones) {
  // The rifle has an idle and a walk: walking backwards it walks forward
  // with the rifle (the arms keep holding it); a jump is the unarmed jump,
  // not the rifle's idle.
  const Skeleton s = rig();
  AnimationLibrary l = fullLibrary();
  putIn(l, Stance::Rifle, ClipSlot::Idle, marker(30, 2.0f, 0.0f, true));
  putIn(l, Stance::Rifle, ClipSlot::WalkForward, marker(50, 1.0f, 1.5f, true));
  CharacterAnimator a(s, l, 1.0f);
  AnimatorInput back = inStance(Stance::Rifle, 1.5f);
  back.aiming = true;
  back.direction = glm::vec2(0.0f, -1.0f);
  runFor(a, back, 1.0f);
  CHECK_NEAR(hipsDegrees(a), 50.0, 0.5);
  AnimatorInput up = inTheAir(3.0f);
  up.stance = Stance::Rifle;
  CharacterAnimator b(s, l, 1.0f);
  runFor(b, inStance(Stance::Rifle), 0.5f);
  runFor(b, up, 0.5f);
  CHECK(b.state() == AnimState::Jump);
  CHECK_NEAR(hipsDegrees(b), 140.0, 0.5);
}

TEST(changing_stance_eases_from_the_old_pose) {
  const Skeleton s = rig();
  AnimationLibrary l = fullLibrary();
  putIn(l, Stance::Pistol, ClipSlot::Idle, marker(40, 2.0f, 0.0f, true));
  AnimatorTuning tuning;
  tuning.stanceFade = 0.3f;
  CharacterAnimator a(s, l, 1.0f, tuning);
  runFor(a, inStance(Stance::Unarmed), 1.0f);
  a.update(inStance(Stance::Pistol), kDt);
  CHECK(hipsDegrees(a) < 5.0f); // still nearly the old pose
  runFor(a, inStance(Stance::Pistol), 0.3f);
  CHECK_NEAR(hipsDegrees(a), 40.0, 0.5);
}

TEST(aiming_still_in_a_stance_plays_its_aim_idle) {
  const Skeleton s = rig();
  AnimationLibrary l = fullLibrary();
  putIn(l, Stance::Rifle, ClipSlot::Idle, marker(30, 2.0f, 0.0f, true));
  putIn(l, Stance::Rifle, ClipSlot::AimIdle, marker(35, 2.0f, 0.0f, true));
  CharacterAnimator a(s, l, 1.0f);
  AnimatorInput aim = inStance(Stance::Rifle);
  aim.aiming = true;
  runFor(a, aim, 1.0f);
  CHECK_NEAR(hipsDegrees(a), 35.0, 0.5);
  // Without one, the plain idle.
  AnimationLibrary plain = fullLibrary();
  putIn(plain, Stance::Rifle, ClipSlot::Idle, marker(30, 2.0f, 0.0f, true));
  CharacterAnimator b(s, plain, 1.0f);
  runFor(b, aim, 1.0f);
  CHECK_NEAR(hipsDegrees(b), 30.0, 0.5);
}

namespace {

// The test rig with a spine above the hips and a right shoulder on it, for
// what plays on the upper body and the arms.
Skeleton rigWithSpine() {
  Skeleton s = rig();
  Bone spine;
  spine.name = "mixamorig:Spine";
  spine.parent = 1;
  spine.bind.translation = glm::vec3(0.0f, 0.1f, 0.0f);
  s.bones.push_back(spine);
  Bone shoulder;
  shoulder.name = "mixamorig:RightShoulder";
  shoulder.parent = 2;
  shoulder.bind.translation = glm::vec3(-0.1f, 0.3f, 0.0f);
  s.bones.push_back(shoulder);
  return s;
}

// Turns the spine and the shoulder by their own angles about x, the hips
// not at all: a gun stance's pose.
CharacterClipData armMarker(float spineDegrees, float shoulderDegrees) {
  CharacterClipData d;
  d.loop = true;
  d.clip.duration = 2.0f;
  d.clip.tracks.resize(4);
  const glm::vec3 x(1, 0, 0);
  const glm::quat spine = glm::angleAxis(glm::radians(spineDegrees), x);
  const glm::quat shoulder = glm::angleAxis(glm::radians(shoulderDegrees), x);
  d.clip.tracks[2].rotations = {{0.0f, spine}, {2.0f, spine}};
  d.clip.tracks[3].rotations = {{0.0f, shoulder}, {2.0f, shoulder}};
  return d;
}

float shoulderDegrees(const CharacterAnimator &a) {
  const glm::quat q = a.pose().local[3].rotation;
  return glm::degrees(2.0f * std::acos(std::min(1.0f, std::abs(q.w))));
}

// Turns the spine by its own angle about x, the hips not at all.
CharacterClipData spineMarker(float degrees, float duration) {
  CharacterClipData d;
  d.loop = false;
  d.clip.duration = duration;
  d.clip.tracks.resize(3);
  const glm::quat turn = glm::angleAxis(glm::radians(degrees), glm::vec3(1, 0, 0));
  d.clip.tracks[2].rotations = {{0.0f, turn}, {duration, turn}};
  return d;
}

float spineDegrees(const CharacterAnimator &a) {
  const glm::quat q = a.pose().local[2].rotation;
  return glm::degrees(2.0f * std::acos(std::min(1.0f, std::abs(q.w))));
}

} // namespace

TEST(a_reload_plays_on_the_upper_body_and_eases_in_and_out) {
  const Skeleton s = rigWithSpine();
  AnimationLibrary l = fullLibrary();
  putIn(l, Stance::Rifle, ClipSlot::Idle, marker(30, 2.0f, 0.0f, true));
  putIn(l, Stance::Rifle, ClipSlot::Reload, spineMarker(45, 2.0f));
  CharacterAnimator a(s, l, 1.0f);
  AnimatorInput in = inStance(Stance::Rifle);
  runFor(a, in, 1.0f);
  CHECK_NEAR(spineDegrees(a), 0.0, 0.5);
  in.reload = 0.5f; // half way through
  a.update(in, kDt);
  CHECK_NEAR(spineDegrees(a), 45.0, 0.5);
  CHECK_NEAR(hipsDegrees(a), 30.0, 0.5); // the legs and hips keep the idle
  in.reload = 0.02f; // just starting: easing in
  a.update(in, kDt);
  CHECK(spineDegrees(a) > 1.0f && spineDegrees(a) < 44.0f);
  in.reload = -1.0f; // done
  a.update(in, kDt);
  CHECK_NEAR(spineDegrees(a), 0.0, 0.5);
}

TEST(crouching_in_a_stance_without_crouch_clips_uses_the_unarmed_crouch) {
  // The pistol pack has no crouch: crouched with the pistol, the legs are
  // the unarmed crouch's, not the pistol's standing idle.
  const Skeleton s = rig();
  AnimationLibrary l = fullLibrary();
  putIn(l, Stance::Pistol, ClipSlot::Idle, marker(40, 2.0f, 0.0f, true));
  CharacterAnimator a(s, l, 1.0f);
  AnimatorInput crouched = inStance(Stance::Pistol);
  crouched.crouched = true;
  runFor(a, crouched, 1.0f);
  CHECK(a.state() == AnimState::Crouch);
  CHECK_NEAR(hipsDegrees(a), 170.0, 0.5);
  // Walking crouched too.
  crouched.speed = 1.0f;
  runFor(a, crouched, 1.0f);
  CHECK_NEAR(hipsDegrees(a), 175.0, 0.5);
}

TEST(crouching_with_a_gun_but_no_gun_crouch_takes_only_the_arms_from_the_gun) {
  // The legs, hips and spine the unarmed crouch's, the arms the pistol's
  // (its idle), so the gun stays up without the standing spine hunching
  // over crouched hips.
  const Skeleton s = rigWithSpine();
  AnimationLibrary l = fullLibrary();
  putIn(l, Stance::Pistol, ClipSlot::Idle, armMarker(30, 45));
  CharacterAnimator a(s, l, 1.0f);
  AnimatorInput crouched = inStance(Stance::Pistol);
  crouched.crouched = true;
  runFor(a, crouched, 1.0f);
  CHECK_NEAR(hipsDegrees(a), 170.0, 0.5);
  CHECK_NEAR(spineDegrees(a), 0.0, 0.5);
  CHECK_NEAR(shoulderDegrees(a), 45.0, 0.5);
  // A stance with its own crouch plays it whole.
  AnimationLibrary own = l;
  putIn(own, Stance::Pistol, ClipSlot::CrouchIdle, marker(60, 1.0f, 0.0f, true));
  CharacterAnimator b(s, own, 1.0f);
  runFor(b, crouched, 1.0f);
  CHECK_NEAR(hipsDegrees(b), 60.0, 0.5);
  CHECK_NEAR(shoulderDegrees(b), 0.0, 0.5);
}

TEST(in_the_air_with_a_gun_the_arms_keep_holding_it) {
  // No rifle jump: the unarmed jump, the rifle's arms (its aim idle while
  // aiming) over it.
  const Skeleton s = rigWithSpine();
  AnimationLibrary l = fullLibrary();
  putIn(l, Stance::Rifle, ClipSlot::Idle, armMarker(10, 20));
  putIn(l, Stance::Rifle, ClipSlot::AimIdle, armMarker(15, 50));
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, inStance(Stance::Rifle), 0.5f);
  AnimatorInput up = inTheAir(3.0f);
  up.stance = Stance::Rifle;
  up.aiming = true;
  runFor(a, up, 0.5f);
  CHECK(a.state() == AnimState::Jump);
  CHECK_NEAR(hipsDegrees(a), 140.0, 0.5);
  CHECK_NEAR(spineDegrees(a), 0.0, 0.5);
  CHECK_NEAR(shoulderDegrees(a), 50.0, 0.5);
}

TEST(a_reload_plays_its_guns_clip_while_the_body_moves_unarmed) {
  // Carrying the rifle relaxed (the unarmed moves), a reload still plays
  // the rifle's reload on the upper body.
  const Skeleton s = rigWithSpine();
  AnimationLibrary l = fullLibrary();
  putIn(l, Stance::Rifle, ClipSlot::Reload, spineMarker(45, 2.0f));
  CharacterAnimator a(s, l, 1.0f);
  AnimatorInput in = moving(0.0f);
  runFor(a, in, 0.5f);
  in.reload = 0.5f;
  in.reloadStance = Stance::Rifle;
  a.update(in, kDt);
  CHECK_NEAR(spineDegrees(a), 45.0, 0.5);
  CHECK_NEAR(hipsDegrees(a), 0.0, 0.5); // the unarmed idle's hips
}

TEST(carrying_a_pistol_relaxed_a_jump_keeps_the_gun_arm_hanging) {
  // The unarmed jump flings the arms; with a gun in the right hand that arm
  // keeps the unarmed idle's hang, the rest jumps as it would.
  const Skeleton s = rigWithSpine();
  AnimationLibrary l = fullLibrary();
  l.slotFor(ClipSlot::Idle, Stance::Unarmed) = armMarker(0, 10);
  l.slotFor(ClipSlot::Jump, Stance::Unarmed) = armMarker(20, 80);
  CharacterAnimator a(s, l, 1.0f);
  AnimatorInput standing = moving(0.0f);
  standing.gunInRightHand = true;
  runFor(a, standing, 0.5f);
  AnimatorInput up = inTheAir(3.0f);
  up.gunInRightHand = true;
  runFor(a, up, 0.5f);
  CHECK(a.state() == AnimState::Jump);
  CHECK_NEAR(spineDegrees(a), 20.0, 0.5);
  CHECK_NEAR(shoulderDegrees(a), 10.0, 0.5);
  // Empty-handed, the arm goes with the jump.
  CharacterAnimator b(s, l, 1.0f);
  runFor(b, moving(0.0f), 0.5f);
  runFor(b, inTheAir(3.0f), 0.5f);
  CHECK_NEAR(shoulderDegrees(b), 80.0, 0.5);
}

TEST(a_stance_with_only_an_idle_walks_and_runs_with_the_unarmed_moves) {
  // A pistol carried at the side: its own idle standing, but walking it
  // takes the unarmed walk (its idle would slide the feet), and a carry
  // with its own walk plays that.
  const Skeleton s = rigWithSpine();
  AnimationLibrary l = fullLibrary();
  putIn(l, Stance::PistolCarry, ClipSlot::Idle, armMarker(5, 25));
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, inStance(Stance::PistolCarry), 1.0f);
  CHECK_NEAR(shoulderDegrees(a), 25.0, 0.5);
  runFor(a, inStance(Stance::PistolCarry, 1.5f), 1.0f);
  CHECK(a.clipWeight(ClipSlot::WalkForward) > 0.5f);
  CHECK_NEAR(shoulderDegrees(a), 0.0, 0.5); // the unarmed walk's arm, not the idle's
  putIn(l, Stance::RifleCarry, ClipSlot::Idle, armMarker(5, 25));
  CharacterClipData walk = armMarker(10, 40);
  walk.rootMotion = marker(0, 2.0f, 1.5f, true).rootMotion; // walking at 1.5 m/s
  putIn(l, Stance::RifleCarry, ClipSlot::WalkForward, walk);
  CharacterAnimator b(s, l, 1.0f);
  runFor(b, inStance(Stance::RifleCarry, 1.5f), 1.0f);
  CHECK_NEAR(shoulderDegrees(b), 40.0, 1.0);
}

TEST(a_looping_clip_trimmed_to_a_stretch_loops_inside_it) {
  // The pistol's relaxed idle is the end of a longer clip: aiming for its
  // first second, lowered for its second. Trimmed to the second, it never
  // shows the first, however long it loops.
  const Skeleton s = rigWithSpine();
  AnimationLibrary l = fullLibrary();
  CharacterClipData idle;
  idle.loop = true;
  idle.clip.duration = 2.0f;
  idle.clip.tracks.resize(4);
  const glm::quat up = glm::angleAxis(glm::radians(80.0f), glm::vec3(1, 0, 0));
  const glm::quat down = glm::angleAxis(glm::radians(10.0f), glm::vec3(1, 0, 0));
  idle.clip.tracks[3].rotations = {{0.0f, up}, {0.99f, up}, {1.0f, down}, {2.0f, down}};
  idle.start = 1.0f;
  idle.end = 2.0f;
  putIn(l, Stance::PistolCarry, ClipSlot::Idle, idle);
  CharacterAnimator a(s, l, 1.0f);
  const AnimatorInput in = inStance(Stance::PistolCarry);
  runFor(a, in, 0.5f);
  for (int i = 0; i < 300; ++i) { // 2.5 s of looping, every frame checked
    a.update(in, kDt);
    CHECK(shoulderDegrees(a) < 12.0f);
  }
}

TEST(dying_plays_the_death_clip_once_and_stays_down) {
  const Skeleton s = rigWithSpine();
  AnimationLibrary l = fullLibrary();
  CharacterClipData death;
  death.loop = false;
  death.clip.duration = 1.0f;
  death.clip.tracks.resize(2);
  const glm::quat lying = glm::angleAxis(glm::radians(80.0f), glm::vec3(1, 0, 0));
  death.clip.tracks[1].rotations = {{0.0f, glm::quat(1, 0, 0, 0)}, {1.0f, lying}};
  put(l, ClipSlot::DeathFront, death);
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, moving(0.0f), 0.5f);
  AnimatorInput dead = moving(2.0f); // whatever the body does, he's down
  dead.dead = true;
  dead.deathClip = ClipSlot::DeathFront;
  runFor(a, dead, 3.0f);
  CHECK(a.state() == AnimState::Dead);
  CHECK_NEAR(hipsDegrees(a), 80.0, 0.5); // the clip's end, held
  // Brought back (a respawn): standing again.
  runFor(a, moving(0.0f), 1.0f);
  CHECK(a.state() == AnimState::Locomotion);
  CHECK_NEAR(hipsDegrees(a), 0.0, 0.5);
}

TEST(a_hit_flinches_the_upper_body_and_fades) {
  const Skeleton s = rigWithSpine();
  AnimationLibrary l = fullLibrary();
  put(l, ClipSlot::Hit, spineMarker(30, 0.5f));
  CharacterAnimator a(s, l, 1.0f);
  runFor(a, moving(0.0f), 0.5f);
  AnimatorInput hit = moving(0.0f);
  hit.hit = true;
  a.update(hit, kDt);
  runFor(a, moving(0.0f), 0.2f); // mid-flinch
  CHECK(spineDegrees(a) > 10.0f);
  CHECK_NEAR(hipsDegrees(a), 0.0, 0.5); // the legs keep the idle
  runFor(a, moving(0.0f), 1.0f);
  CHECK_NEAR(spineDegrees(a), 0.0, 0.5);
}

TEST(a_throw_plays_on_the_upper_body_by_its_progress) {
  const Skeleton s = rigWithSpine();
  AnimationLibrary l = fullLibrary();
  put(l, ClipSlot::Throw, spineMarker(50, 2.0f));
  CharacterAnimator a(s, l, 1.0f);
  AnimatorInput in = moving(0.0f);
  runFor(a, in, 0.5f);
  in.throwing = 0.5f;
  a.update(in, kDt);
  CHECK_NEAR(spineDegrees(a), 50.0, 0.5);
  CHECK_NEAR(hipsDegrees(a), 0.0, 0.5); // the legs keep the idle
  in.throwing = -1.0f;
  a.update(in, kDt);
  CHECK_NEAR(spineDegrees(a), 0.0, 0.5);
}
