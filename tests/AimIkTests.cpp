#include "AimIk.h"
#include "TestRunner.h"

#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>

namespace {

enum : int { kRoot, kSpine, kSpine1, kSpine2, kHand };

// A spine of three bones and a right hand held out in front of the chest,
// named like a Mixamo rig; the gun runs half a metre on from the hand
// along its +z.
Skeleton torso() {
  Skeleton s;
  auto add = [&](const char *name, int parent, glm::vec3 t) {
    Bone b;
    b.name = name;
    b.parent = parent;
    b.bind.translation = t;
    s.bones.push_back(b);
  };
  add("root", -1, glm::vec3(0.0f));
  add("mixamorig:Spine", kRoot, {0.0f, 1.0f, 0.0f});
  add("mixamorig:Spine1", kSpine, {0.0f, 0.12f, 0.0f});
  add("mixamorig:Spine2", kSpine1, {0.0f, 0.12f, 0.0f});
  add("mixamorig:RightHand", kSpine2, {-0.2f, 0.1f, 0.3f});
  return s;
}

Barrel barrelOf(const std::vector<glm::mat4> &model) {
  const glm::mat4 &hand = model[kHand];
  const glm::vec3 forward = glm::normalize(glm::vec3(hand[2]));
  return {glm::vec3(hand[3]) + 0.5f * forward, forward};
}

float degreesOff(const Skeleton &s, const Pose &pose, glm::vec3 target) {
  const Barrel b = barrelOf(modelSpace(s, pose));
  const float c = glm::dot(b.forward, glm::normalize(target - b.muzzle));
  return glm::degrees(std::acos(std::clamp(c, -1.0f, 1.0f)));
}

float turned(const Pose &pose, int bone) {
  return glm::degrees(glm::angle(pose.local[bone].rotation));
}

} // namespace

TEST(the_aim_rig_is_the_three_spine_bones_sharing_20_30_50) {
  const AimRig rig = findAimRig(torso());
  CHECK(rig.bones == (std::vector<int>{kSpine, kSpine1, kSpine2}));
  CHECK(rig.shares.size() == 3);
  if (rig.shares.size() == 3) {
    CHECK_NEAR(rig.shares[0], 0.2, 1e-6);
    CHECK_NEAR(rig.shares[2], 0.5, 1e-6);
  }
}

TEST(aiming_turns_the_spine_until_the_barrel_points_at_the_target) {
  const Skeleton s = torso();
  const AimRig rig = findAimRig(s);
  const glm::vec3 target(1.0f, 2.5f, 6.0f);
  Pose pose = bindPose(s);
  CHECK(degreesOff(s, pose, target) > 10.0f);
  aimAt(s, rig, pose, barrelOf, target, 1.0f);
  CHECK(degreesOff(s, pose, target) < 0.5f);
  // The upper spine does most of it.
  CHECK(turned(pose, kSpine2) > 2.0f * turned(pose, kSpine));
}

TEST(half_the_weight_turns_about_half_as_far) {
  const Skeleton s = torso();
  const AimRig rig = findAimRig(s);
  const glm::vec3 target(1.0f, 2.5f, 6.0f);
  Pose full = bindPose(s), half = bindPose(s), none = bindPose(s);
  const float all = aimAt(s, rig, full, barrelOf, target, 1.0f);
  const float some = aimAt(s, rig, half, barrelOf, target, 0.5f);
  CHECK_NEAR(some, 0.5 * all, 0.1 * all);
  CHECK_NEAR(aimAt(s, rig, none, barrelOf, target, 0.0f), 0.0, 1e-6);
  CHECK(glm::angle(none.local[kSpine2].rotation) < 1e-6f);
}

TEST(a_target_behind_turns_no_further_than_the_limit) {
  const Skeleton s = torso();
  const AimRig rig = findAimRig(s);
  Pose pose = bindPose(s);
  const float angle = aimAt(s, rig, pose, barrelOf, glm::vec3(0.0f, 1.3f, -5.0f), 1.0f, 1.0f);
  CHECK(angle <= 1.0f + 1e-3f);
  CHECK(angle > 0.9f);
}

TEST(aiming_works_under_a_scaled_armature_as_mixamo_rigs_come) {
  // The armature scaled to a hundredth with its bones in centimetres: the
  // same aim, in the same (scaled) space.
  Skeleton s = torso();
  s.bones[kRoot].bind.scale = glm::vec3(0.01f);
  for (int b = kSpine; b <= kHand; ++b)
    s.bones[b].bind.translation *= 100.0f;
  const AimRig rig = findAimRig(s);
  Pose pose = bindPose(s);
  const glm::vec3 target(1.0f, 2.5f, 6.0f);
  auto barrel = [](const std::vector<glm::mat4> &model) {
    const glm::mat4 &hand = model[kHand];
    const glm::vec3 forward = glm::normalize(glm::vec3(hand[2]));
    return Barrel{glm::vec3(hand[3]) + 0.5f * forward, forward};
  };
  aimAt(s, rig, pose, barrel, target, 1.0f);
  const Barrel b = barrel(modelSpace(s, pose));
  const float off = glm::degrees(
      std::acos(std::clamp(glm::dot(b.forward, glm::normalize(target - b.muzzle)), -1.0f, 1.0f)));
  CHECK(off < 0.5f);
}

TEST(a_bone_turns_toward_a_point_up_to_its_limit) {
  // The hand looking along its +z; a point up and to the side: turned to it.
  const Skeleton s = torso();
  Pose pose = bindPose(s);
  const glm::vec3 target(1.0f, 3.0f, 4.0f);
  const float turned = turnToward(s, pose, kHand, glm::vec3(0, 0, 1), target, 1.0f, 1.5f);
  CHECK(turned > 0.1f);
  const glm::mat4 hand = modelSpace(s, pose)[kHand];
  const glm::vec3 facing = glm::normalize(glm::vec3(hand[2]));
  CHECK(glm::dot(facing, glm::normalize(target - glm::vec3(hand[3]))) > 0.9999f);
  // Held to a limit, and none at no weight.
  Pose held = bindPose(s), still = bindPose(s);
  CHECK_NEAR(turnToward(s, held, kHand, glm::vec3(0, 0, 1), target, 1.0f, 0.2f), 0.2, 1e-4);
  CHECK_NEAR(turnToward(s, still, kHand, glm::vec3(0, 0, 1), target, 0.0f, 1.0f), 0.0, 1e-6);
}
