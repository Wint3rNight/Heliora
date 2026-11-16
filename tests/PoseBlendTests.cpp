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
