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

namespace {

enum : int { kLeftShoulder = kHand + 1, kRightShoulder };

// The torso with its shoulder joints, the chest turned `bladed` degrees to
// its right (the left shoulder forward, as a rifle stance stands).
Skeleton shouldered() {
  Skeleton s = torso();
  auto add = [&](const char *name, glm::vec3 t) {
    Bone b;
    b.name = name;
    b.parent = kSpine2;
    b.bind.translation = t;
    s.bones.push_back(b);
  };
  add("mixamorig:LeftArm", {0.18f, 0.15f, 0.0f});
  add("mixamorig:RightArm", {-0.18f, 0.15f, 0.0f});
  return s;
}

Pose bladed(const Skeleton &s, float degrees) {
  Pose pose = bindPose(s);
  pose.local[kSpine].rotation = glm::angleAxis(glm::radians(-degrees), glm::vec3(0, 1, 0));
  return pose;
}

// Degrees the chest faces to the right of +z, and the shoulder line's tilt
// (its left end up).
float chestRight(const Skeleton &s, const Pose &pose) {
  const std::vector<glm::mat4> m = modelSpace(s, pose);
  const glm::vec3 f = chestFacing(glm::vec3(m[kLeftShoulder][3]), glm::vec3(m[kRightShoulder][3]),
                                  glm::vec3(0, 1, 0));
  return glm::degrees(std::atan2(-f.x, f.z));
}
float shoulderTilt(const Skeleton &s, const Pose &pose) {
  const std::vector<glm::mat4> m = modelSpace(s, pose);
  const glm::vec3 line = glm::normalize(glm::vec3(m[kLeftShoulder][3]) - glm::vec3(m[kRightShoulder][3]));
  return glm::degrees(std::asin(line.y));
}

} // namespace

TEST(the_chest_faces_square_to_the_shoulder_line) {
  const glm::vec3 f = chestFacing({0.2f, 1.4f, 0.0f}, {-0.2f, 1.4f, 0.0f}, {0, 1, 0});
  CHECK_NEAR(f.z, 1.0, 1e-5);
  CHECK_NEAR(f.y, 0.0, 1e-5);
}

TEST(squaring_turns_a_bladed_chest_to_face_the_aim_less_its_blade) {
  // 55 degrees side-on, squared to 10 off the aim (+z), the spine sharing it.
  const Skeleton s = shouldered();
  const AimRig rig = findAimRig(s);
  Pose pose = bladed(s, 55.0f);
  CHECK_NEAR(chestRight(s, pose), 55.0, 0.5);
  const float angle = squareChest(s, rig, pose, kLeftShoulder, kRightShoulder, {0, 0, 1},
                                  glm::radians(10.0f), 1.0f, glm::radians(60.0f));
  CHECK_NEAR(glm::degrees(angle), 45.0, 0.5);
  CHECK_NEAR(chestRight(s, pose), 10.0, 0.5);
  CHECK(turned(pose, kSpine2) > 0.0f && turned(pose, kSpine1) > 0.0f);
}

TEST(squaring_goes_weight_of_the_way_and_no_further_than_its_limit) {
  const Skeleton s = shouldered();
  const AimRig rig = findAimRig(s);
  Pose half = bladed(s, 55.0f), capped = bladed(s, 55.0f);
  squareChest(s, rig, half, kLeftShoulder, kRightShoulder, {0, 0, 1}, glm::radians(10.0f), 0.5f,
              glm::radians(60.0f));
  CHECK_NEAR(chestRight(s, half), 32.5, 0.5);
  squareChest(s, rig, capped, kLeftShoulder, kRightShoulder, {0, 0, 1}, glm::radians(10.0f), 1.0f,
              glm::radians(30.0f));
  CHECK_NEAR(chestRight(s, capped), 25.0, 0.5);
}

TEST(bending_the_chest_about_its_shoulder_line_keeps_the_shoulders_level) {
  // Bladed, a lean back about the chest's own sideways line tilts the chest
  // up without tipping one shoulder over the other.
  const Skeleton s = shouldered();
  const AimRig rig = findAimRig(s);
  Pose pose = bladed(s, 55.0f);
  bendChest(s, rig, pose, kLeftShoulder, kRightShoulder, glm::radians(30.0f));
  CHECK_NEAR(shoulderTilt(s, pose), 0.0, 0.5);
  // The chest's top went back: Spine2's up tips away from its facing.
  const std::vector<glm::mat4> m = modelSpace(s, pose);
  const glm::vec3 up = glm::normalize(glm::vec3(m[kSpine2][1]));
  CHECK_NEAR(glm::degrees(std::acos(up.y)), 30.0, 0.5);
  CHECK(glm::dot(up, glm::vec3(-std::sin(glm::radians(55.0f)), 0.0f, std::cos(glm::radians(55.0f)))) <
        0.0f);
}

TEST(an_aimed_gun_lies_along_the_aim_level_with_its_anchor_where_asked) {
  const glm::vec3 aim = glm::normalize(glm::vec3(0.3f, 0.4f, 1.0f));
  const glm::vec3 butt(0.0f, -0.02f, -0.45f);
  const glm::mat4 g = aimedGun(aim, {0, 1, 0}, butt, {0.1f, 1.45f, 0.05f});
  CHECK_NEAR(glm::distance(glm::vec3(g * glm::vec4(butt, 1.0f)), glm::vec3(0.1f, 1.45f, 0.05f)), 0.0,
             1e-5);
  CHECK_NEAR(glm::dot(glm::vec3(g[2]), aim), 1.0, 1e-5);
  CHECK_NEAR(glm::vec3(g[0]).y, 0.0, 1e-5); // not canted
  CHECK(glm::vec3(g[1]).y > 0.0f);
}

TEST(squaring_works_under_a_scaled_armature_as_mixamo_rigs_come) {
  // The armature scaled to a hundredth with its bones in centimetres: the
  // same squaring, in the same (scaled) space.
  Skeleton s = shouldered();
  s.bones[kRoot].bind.scale = glm::vec3(0.01f);
  for (int b = kSpine; b < static_cast<int>(s.bones.size()); ++b)
    s.bones[b].bind.translation *= 100.0f;
  const AimRig rig = findAimRig(s);
  Pose pose = bladed(s, 55.0f);
  squareChest(s, rig, pose, kLeftShoulder, kRightShoulder, {0, 0, 1}, glm::radians(10.0f), 1.0f,
              glm::radians(60.0f));
  CHECK_NEAR(chestRight(s, pose), 10.0, 0.5);
  bendChest(s, rig, pose, kLeftShoulder, kRightShoulder, glm::radians(30.0f));
  CHECK_NEAR(shoulderTilt(s, pose), 0.0, 0.5);
}

TEST(squaring_a_forward_leaning_chest_twists_it_without_swinging_it_sideways) {
  // A rifle stance leans forward. Turned about the world's up, the top of a
  // leaning spine swings sideways (the back bows); turned about the trunk's
  // own line, the shoulders stay over the hips.
  const Skeleton s = shouldered();
  const AimRig rig = findAimRig(s);
  Pose pose = bladed(s, 55.0f);
  pose.local[kSpine].rotation =
      pose.local[kSpine].rotation * glm::angleAxis(glm::radians(25.0f), glm::vec3(1, 0, 0));
  auto middle = [&](const Pose &p) {
    const std::vector<glm::mat4> m = modelSpace(s, p);
    return 0.5f * (glm::vec3(m[kLeftShoulder][3]) + glm::vec3(m[kRightShoulder][3]));
  };
  const glm::vec3 before = middle(pose);
  squareChest(s, rig, pose, kLeftShoulder, kRightShoulder, {0, 0, 1}, glm::radians(10.0f), 1.0f,
              glm::radians(60.0f));
  CHECK_NEAR(chestRight(s, pose), 10.0, 1.0);
  CHECK(glm::distance(middle(pose), before) < 0.01f);
}
