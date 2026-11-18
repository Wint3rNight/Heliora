#include "PoseBlend.h"
#include "TestRunner.h"

#include <cmath>
#include <numeric>

namespace {

constexpr float kHalfPi = 1.5707963f;

Pose onePose(glm::vec3 translation, glm::quat rotation, glm::vec3 scale) {
  Pose pose;
  pose.local.push_back({translation, rotation, scale});
  return pose;
}

// root, hips, a leg under the hips, and spine, chest and arm up from the
// hips, each 0.5 m above or beside its parent.
Skeleton body() {
  Skeleton s;
  auto add = [&](const char *name, int parent, glm::vec3 at) {
    Bone b;
    b.name = name;
    b.parent = parent;
    b.bind.translation = at;
    s.bones.push_back(b);
  };
  add("root", -1, {0, 0, 0});
  add("hips", 0, {0, 1, 0});
  add("leg", 1, {0.2f, -0.5f, 0});
  add("spine", 1, {0, 0.5f, 0});
  add("chest", 3, {0, 0.5f, 0});
  add("arm", 4, {-0.5f, 0, 0});
  return s;
}

glm::vec3 at(const std::vector<glm::mat4> &model, int bone) { return glm::vec3(model[bone][3]); }

float turnedBy(glm::quat q) { return 2.0f * std::acos(std::min(1.0f, std::abs(q.w))); }

glm::quat rotationOf(const glm::mat4 &m) {
  return glm::quat_cast(glm::mat3(glm::normalize(glm::vec3(m[0])), glm::normalize(glm::vec3(m[1])),
                                  glm::normalize(glm::vec3(m[2]))));
}

// A shoulder under a root turned a quarter about z, then an upper arm and a
// forearm 1 m each.
Skeleton arm() {
  Skeleton s;
  Bone root;
  root.name = "root";
  root.bind.rotation = glm::angleAxis(kHalfPi, glm::vec3(0, 0, 1));
  Bone upper;
  upper.name = "upper";
  upper.parent = 0;
  upper.bind.translation = {0, 1, 0};
  Bone lower;
  lower.name = "lower";
  lower.parent = 1;
  lower.bind.translation = {1, 0, 0};
  Bone end;
  end.name = "end";
  end.parent = 2;
  end.bind.translation = {1, 0, 0};
  s.bones = {root, upper, lower, end};
  return s;
}

float sum(const std::vector<float> &weights) {
  return std::accumulate(weights.begin(), weights.end(), 0.0f);
}

} // namespace

TEST(blending_halfway_mixes_every_part) {
  const Pose a = onePose(glm::vec3(0.0f), glm::quat(1, 0, 0, 0), glm::vec3(1.0f));
  const Pose b = onePose(glm::vec3(2, 0, 0), glm::angleAxis(kHalfPi, glm::vec3(0, 1, 0)),
                         glm::vec3(3.0f));
  Pose out;
  blendPoses(a, b, 0.5f, out);
  CHECK_NEAR(glm::distance(out.local[0].translation, glm::vec3(1, 0, 0)), 0.0, 1e-5);
  CHECK_NEAR(glm::distance(out.local[0].scale, glm::vec3(2.0f)), 0.0, 1e-5);
  CHECK_NEAR(glm::length(out.local[0].rotation), 1.0, 1e-5); // still a rotation
  const glm::vec3 x = out.local[0].rotation * glm::vec3(1, 0, 0);
  const float c = std::cos(kHalfPi * 0.5f);
  CHECK_NEAR(glm::distance(x, glm::vec3(c, 0, -c)), 0.0, 1e-4);
}

TEST(blending_the_ends_gives_each_pose) {
  const Pose a = onePose(glm::vec3(1, 2, 3), glm::quat(1, 0, 0, 0), glm::vec3(1.0f));
  const Pose b = onePose(glm::vec3(4, 5, 6), glm::quat(1, 0, 0, 0), glm::vec3(1.0f));
  Pose out;
  blendPoses(a, b, 0.0f, out);
  CHECK_NEAR(glm::distance(out.local[0].translation, glm::vec3(1, 2, 3)), 0.0, 1e-6);
  blendPoses(a, b, 1.0f, out);
  CHECK_NEAR(glm::distance(out.local[0].translation, glm::vec3(4, 5, 6)), 0.0, 1e-6);
}

TEST(blending_rotations_takes_the_short_way) {
  // -q is the same rotation as q, so blending towards it is no change.
  const glm::quat q = glm::angleAxis(0.7f, glm::vec3(0, 1, 0));
  const Pose a = onePose(glm::vec3(0.0f), q, glm::vec3(1.0f));
  const Pose b = onePose(glm::vec3(0.0f), -q, glm::vec3(1.0f));
  Pose out;
  blendPoses(a, b, 0.5f, out);
  CHECK_NEAR(std::abs(glm::dot(out.local[0].rotation, q)), 1.0, 1e-5);
}

TEST(speed_blend_shares_between_the_two_nearest) {
  // Idle, walk, run and sprint at their own speeds.
  const std::vector<float> speeds = {0.0f, 1.5f, 5.0f, 8.0f};
  std::vector<float> w = speedBlend(speeds, 3.25f);
  CHECK(w.size() == 4);
  CHECK_NEAR(w[0], 0.0, 1e-6);
  CHECK_NEAR(w[1], 0.5, 1e-5);
  CHECK_NEAR(w[2], 0.5, 1e-5);
  CHECK_NEAR(w[3], 0.0, 1e-6);
  w = speedBlend(speeds, 1.5f);
  CHECK_NEAR(w[1], 1.0, 1e-6);
  w = speedBlend(speeds, 12.0f); // faster than sprint: still sprint
  CHECK_NEAR(w[3], 1.0, 1e-6);
  w = speedBlend(speeds, -1.0f);
  CHECK_NEAR(w[0], 1.0, 1e-6);
  for (float speed : {0.3f, 2.0f, 6.9f})
    CHECK_NEAR(sum(speedBlend(speeds, speed)), 1.0, 1e-5);
}

TEST(direction_blend_splits_diagonals) {
  // x is to the character's right, y ahead.
  DirectionWeights w = directionBlend(glm::vec2(0, 1));
  CHECK_NEAR(w.forward, 1.0, 1e-6);
  w = directionBlend(glm::vec2(0.6f, 0.6f));
  CHECK_NEAR(w.forward, 0.5, 1e-5);
  CHECK_NEAR(w.right, 0.5, 1e-5);
  w = directionBlend(glm::vec2(-1, -1));
  CHECK_NEAR(w.back, 0.5, 1e-5);
  CHECK_NEAR(w.left, 0.5, 1e-5);
  w = directionBlend(glm::vec2(0.0f)); // standing still faces forward
  CHECK_NEAR(w.forward, 1.0, 1e-6);
  w = directionBlend(glm::vec2(-0.3f, 0.9f));
  CHECK_NEAR(w.forward + w.back + w.left + w.right, 1.0, 1e-5);
}

TEST(the_upper_body_mask_is_its_first_bone_and_everything_under_it) {
  const std::vector<float> mask = upperBodyMask(body(), 3);
  CHECK(mask == std::vector<float>({0, 0, 0, 1, 1, 1}));
}

TEST(a_masked_blend_leaves_the_bones_outside_the_mask) {
  const Skeleton s = body();
  const Pose rest = bindPose(s);
  Pose turned = rest;
  for (BoneTransform &b : turned.local)
    b.rotation = glm::angleAxis(kHalfPi, glm::vec3(0, 1, 0));
  const std::vector<float> mask = upperBodyMask(s, 3);
  Pose out;
  blendPosesMasked(rest, turned, 1.0f, mask, out);
  CHECK(out.local.size() == 6);
  if (out.local.size() != 6)
    return;
  CHECK_NEAR(turnedBy(out.local[2].rotation), 0.0, 1e-5); // the leg
  CHECK_NEAR(turnedBy(out.local[4].rotation), kHalfPi, 1e-4);
  blendPosesMasked(rest, turned, 0.5f, mask, out);
  CHECK_NEAR(turnedBy(out.local[4].rotation), 0.5 * kHalfPi, 1e-3);
  CHECK_NEAR(turnedBy(out.local[1].rotation), 0.0, 1e-5);
}

TEST(turning_bones_shares_the_turn_down_the_chain) {
  const Skeleton s = body();
  Pose pose = bindPose(s);
  turnBones(s, pose, {3, 4}, glm::vec3(1, 0, 0), 0.4f);
  const std::vector<glm::mat4> model = modelSpace(s, pose);
  CHECK_NEAR(turnedBy(rotationOf(model[3])), 0.2, 1e-4);
  CHECK_NEAR(turnedBy(rotationOf(model[4])), 0.4, 1e-4);
  CHECK_NEAR(turnedBy(rotationOf(model[2])), 0.0, 1e-5); // the leg isn't under them
  // The arm is carried round: it hangs off the turned chest, turned 0.4 too.
  CHECK_NEAR(turnedBy(rotationOf(model[5])), 0.4, 1e-4);
  const glm::vec3 chest = at(model, 4);
  CHECK(std::fabs(chest.z - 0.5f * std::sin(0.2f)) < 1e-4f);
  CHECK(glm::distance(at(model, 5), chest + glm::angleAxis(0.4f, glm::vec3(1, 0, 0)) *
                                                glm::vec3(-0.5f, 0, 0)) < 1e-4f);
}

TEST(two_bone_ik_puts_the_end_on_a_target_in_reach) {
  // As built, and under a root scaled to a hundredth, the way Mixamo's
  // armatures come.
  for (const float scale : {1.0f, 0.01f}) {
    Skeleton s = arm();
    s.bones[0].bind.scale = glm::vec3(scale);
    Pose pose = bindPose(s);
    const glm::vec3 shoulder = at(modelSpace(s, pose), 1);
    const glm::vec3 target = shoulder + scale * glm::vec3(0.8f, -0.6f, 0.9f);
    const glm::vec3 pole(0, -1, 0);
    twoBoneIk(s, pose, 1, 2, 3, target, pole);
    const std::vector<glm::mat4> model = modelSpace(s, pose);
    CHECK(glm::distance(at(model, 3), target) < 1e-3f * scale);
    CHECK_NEAR(glm::distance(at(model, 2), at(model, 1)), scale, 1e-4 * scale); // lengths kept
    CHECK_NEAR(glm::distance(at(model, 3), at(model, 2)), scale, 1e-4 * scale);
    CHECK(glm::distance(at(model, 1), shoulder) < 1e-5f);                     // the shoulder stays
    // The elbow bends toward the pole, off the line to the target.
    const glm::vec3 u = glm::normalize(target - shoulder);
    const glm::vec3 elbow = at(model, 2) - shoulder;
    CHECK(glm::dot(elbow - u * glm::dot(elbow, u), pole) > 0.1f * scale);
  }
}

TEST(two_bone_ik_points_straight_at_a_target_out_of_reach) {
  const Skeleton s = arm();
  Pose pose = bindPose(s);
  const glm::vec3 shoulder = at(modelSpace(s, pose), 1);
  const glm::vec3 target = shoulder + glm::vec3(3, 4, -2);
  twoBoneIk(s, pose, 1, 2, 3, target, glm::vec3(0, -1, 0));
  const std::vector<glm::mat4> model = modelSpace(s, pose);
  const glm::vec3 u = glm::normalize(target - shoulder);
  CHECK(glm::dot(glm::normalize(at(model, 3) - shoulder), u) > 0.9999f);
  CHECK_NEAR(glm::distance(at(model, 3), shoulder), 2.0, 1e-2);
  const glm::vec3 elbow = at(model, 2) - shoulder;
  CHECK(glm::length(elbow - u * glm::dot(elbow, u)) < 2e-2f); // on the line
}
