#include "AnimationClip.h"
#include "TestRunner.h"

#include <cmath>

namespace {

constexpr float kHalfPi = 1.5707963f;

Skeleton twoBones() {
  Skeleton s;
  Bone root;
  root.name = "root";
  Bone child;
  child.name = "child";
  child.parent = 0;
  child.bind.translation = glm::vec3(0.0f, 0.5f, 0.0f);
  s.bones = {root, child};
  return s;
}

// A one-second clip moving the root from x 0 to x 2.
AnimationClip slide() {
  AnimationClip clip;
  clip.name = "slide";
  clip.duration = 1.0f;
  clip.tracks.resize(1);
  clip.tracks[0].positions = {{0.0f, glm::vec3(0.0f)}, {1.0f, glm::vec3(2, 0, 0)}};
  return clip;
}

glm::vec3 turned(const glm::quat &q) { return q * glm::vec3(1, 0, 0); }

} // namespace

TEST(positions_are_linear_between_keys) {
  const Skeleton s = twoBones();
  Pose pose = bindPose(s);
  sampleClip(slide(), s, 0.25f, false, pose);
  CHECK_NEAR(glm::distance(pose.local[0].translation, glm::vec3(0.5f, 0, 0)), 0.0, 1e-5);
}

TEST(rotations_turn_the_short_way_between_keys) {
  // q and -q are the same rotation; halfway must be 45 degrees, not the long
  // way round through 225.
  const Skeleton s = twoBones();
  AnimationClip clip;
  clip.duration = 1.0f;
  clip.tracks.resize(1);
  clip.tracks[0].rotations = {
      {0.0f, glm::quat(1, 0, 0, 0)},
      {1.0f, -glm::angleAxis(kHalfPi, glm::vec3(0, 1, 0))}};
  Pose pose = bindPose(s);
  sampleClip(clip, s, 0.5f, false, pose);
  const float c = std::cos(kHalfPi * 0.5f);
  CHECK_NEAR(glm::distance(turned(pose.local[0].rotation), glm::vec3(c, 0, -c)), 0.0, 1e-4);
}

TEST(outside_the_keys_the_ends_hold) {
  const Skeleton s = twoBones();
  Pose pose = bindPose(s);
  sampleClip(slide(), s, -1.0f, false, pose);
  CHECK_NEAR(pose.local[0].translation.x, 0.0, 1e-6);
  sampleClip(slide(), s, 5.0f, false, pose);
  CHECK_NEAR(pose.local[0].translation.x, 2.0, 1e-6);
}

TEST(looping_clips_wrap_around) {
  const Skeleton s = twoBones();
  Pose pose = bindPose(s);
  sampleClip(slide(), s, 1.25f, true, pose);
  CHECK_NEAR(pose.local[0].translation.x, 0.5, 1e-5);
  sampleClip(slide(), s, -0.25f, true, pose);
  CHECK_NEAR(pose.local[0].translation.x, 1.5, 1e-5);
}

TEST(bones_and_channels_a_clip_leaves_out_keep_the_bind_pose) {
  // The X Bot clips have no skirt bones, and many tracks only rotate.
  const Skeleton s = twoBones();
  AnimationClip clip = slide(); // a track for the root only
  clip.tracks[0].rotations = {{0.0f, glm::angleAxis(kHalfPi, glm::vec3(0, 0, 1))}};
  Pose pose = bindPose(s);
  pose.local[1].translation = glm::vec3(9.0f); // left over from another clip
  sampleClip(clip, s, 0.5f, false, pose);
  CHECK_NEAR(glm::distance(pose.local[1].translation, glm::vec3(0, 0.5f, 0)), 0.0, 1e-6);
  CHECK_NEAR(glm::distance(pose.local[0].scale, glm::vec3(1.0f)), 0.0, 1e-6);
  CHECK_NEAR(glm::distance(turned(pose.local[0].rotation), glm::vec3(0, 1, 0)), 0.0, 1e-5);
  // A bone that only turns keeps its bind position.
  AnimationClip turning;
  turning.duration = 1.0f;
  turning.tracks.resize(2);
  turning.tracks[1].rotations = {{0.0f, glm::quat(1, 0, 0, 0)}};
  sampleClip(turning, s, 0.5f, false, pose);
  CHECK_NEAR(glm::distance(pose.local[1].translation, glm::vec3(0, 0.5f, 0)), 0.0, 1e-6);
}

namespace {

// Hips under the root, moving 2 m forward in a second with a sway in the
// middle: 0.6 m along and 0.1 m to the side at the half, a little higher.
Skeleton hipsUnderRoot() {
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

AnimationClip run() {
  AnimationClip clip;
  clip.duration = 1.0f;
  clip.tracks.resize(2);
  clip.tracks[1].positions = {{0.0f, glm::vec3(0, 1, 0)},
                              {0.5f, glm::vec3(0.6f, 1.1f, 0.1f)},
                              {1.0f, glm::vec3(2, 1, 0)}};
  return clip;
}

glm::vec3 hipsAt(const AnimationClip &clip, const Skeleton &s, float t) {
  Pose pose = bindPose(s);
  sampleClip(clip, s, t, false, pose);
  return pose.local[1].translation;
}

} // namespace

TEST(taking_out_the_progression_keeps_the_sway) {
  const Skeleton s = hipsUnderRoot();
  AnimationClip clip = run();
  const RootMotion motion =
      removeRootMotion(clip, s, 1, RootMotionMode::Progression);
  CHECK_NEAR(motion.speed(), 2.0, 1e-5);
  // Halfway the straight line has done 1 m: 0.4 m behind it, 0.1 m aside,
  // the height untouched.
  CHECK_NEAR(glm::distance(hipsAt(clip, s, 0.5f), glm::vec3(-0.4f, 1.1f, 0.1f)), 0.0, 1e-5);
  CHECK_NEAR(glm::distance(hipsAt(clip, s, 1.0f), glm::vec3(0, 1, 0)), 0.0, 1e-5);
}

TEST(taking_out_the_path_keeps_only_the_height) {
  const Skeleton s = hipsUnderRoot();
  AnimationClip clip = run();
  const RootMotion motion = removeRootMotion(clip, s, 1, RootMotionMode::Path);
  CHECK_NEAR(glm::distance(hipsAt(clip, s, 0.5f), glm::vec3(0, 1.1f, 0)), 0.0, 1e-5);
  // What came out is the path along the ground, from the start.
  CHECK_NEAR(glm::distance(rootDisplacement(motion, 0.0f, 0.5f, false),
                           glm::vec2(0.6f, 0.1f)), 0.0, 1e-5);
  CHECK_NEAR(glm::distance(motion.total(), glm::vec2(2, 0)), 0.0, 1e-5);
}

TEST(root_motion_adds_up_across_a_loop) {
  const Skeleton s = hipsUnderRoot();
  AnimationClip clip = run();
  const RootMotion motion =
      removeRootMotion(clip, s, 1, RootMotionMode::Progression);
  // A quarter before the seam to a quarter after: half a loop, 1 m.
  CHECK_NEAR(glm::distance(rootDisplacement(motion, 0.75f, 1.25f, true),
                           glm::vec2(1, 0)), 0.0, 1e-5);
  // Two and a half loops.
  CHECK_NEAR(glm::distance(rootDisplacement(motion, 0.0f, 2.5f, true),
                           glm::vec2(5, 0)), 0.0, 1e-5);
  // Without looping the ends hold.
  CHECK_NEAR(glm::distance(rootDisplacement(motion, 0.5f, 3.0f, false),
                           glm::vec2(1, 0)), 0.0, 1e-5);
}

TEST(the_ground_is_found_under_a_turned_parent) {
  // Mixamo files often sit under a node turned for z-up: there the hips'
  // own z is the height and their y runs along the ground (towards -z).
  Skeleton s = hipsUnderRoot();
  s.bones[0].bind.rotation = glm::angleAxis(-1.5707963f, glm::vec3(1, 0, 0));
  AnimationClip clip;
  clip.duration = 1.0f;
  clip.tracks.resize(2);
  clip.tracks[1].positions = {{0.0f, glm::vec3(0, 0, 1)},
                              {1.0f, glm::vec3(0, 2, 1.2f)}};
  const RootMotion motion = removeRootMotion(clip, s, 1, RootMotionMode::Path);
  CHECK_NEAR(glm::distance(hipsAt(clip, s, 1.0f), glm::vec3(0, 0, 1.2f)), 0.0, 1e-5);
  CHECK_NEAR(glm::distance(motion.total(), glm::vec2(0, -2)), 0.0, 1e-5);
}

TEST(a_clip_whose_root_never_moves_has_no_root_motion) {
  const Skeleton s = hipsUnderRoot();
  AnimationClip clip;
  clip.duration = 1.0f;
  clip.tracks.resize(2); // the hips only rotate, if anything
  const RootMotion motion =
      removeRootMotion(clip, s, 1, RootMotionMode::Progression);
  CHECK_NEAR(motion.speed(), 0.0, 1e-6);
  CHECK_NEAR(glm::length(rootDisplacement(motion, 0.0f, 3.0f, true)), 0.0, 1e-6);
}
