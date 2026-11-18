#include "PoseBlend.h"

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>

namespace {

BoneTransform blendBone(const BoneTransform &from, const BoneTransform &to, float weight) {
  // q and -q are the same rotation; mixing towards the nearer one is the
  // short way.
  const glm::quat target = glm::dot(from.rotation, to.rotation) < 0.0f ? -to.rotation : to.rotation;
  BoneTransform mixed;
  mixed.translation = glm::mix(from.translation, to.translation, weight);
  mixed.scale = glm::mix(from.scale, to.scale, weight);
  mixed.rotation = glm::normalize(from.rotation * (1.0f - weight) + target * weight);
  return mixed;
}

} // namespace

void blendPoses(const Pose &a, const Pose &b, float weight, Pose &out) {
  const size_t count = std::min(a.local.size(), b.local.size());
  std::vector<BoneTransform> mixed(a.local.begin(), a.local.end());
  for (size_t i = 0; i < count; ++i)
    mixed[i] = blendBone(a.local[i], b.local[i], weight);
  // Built apart and moved in last, so out can be a or b.
  out.local = std::move(mixed);
}

std::vector<float> upperBodyMask(const Skeleton &skeleton, int firstBone) {
  std::vector<float> mask(skeleton.bones.size(), 0.0f);
  if (firstBone < 0 || firstBone >= static_cast<int>(mask.size()))
    return mask;
  mask[firstBone] = 1.0f;
  // Parents come before their children, so one pass down the list does it.
  for (size_t i = firstBone + 1; i < mask.size(); ++i)
    if (const int parent = skeleton.bones[i].parent; parent >= 0)
      mask[i] = mask[parent];
  return mask;
}

void blendPosesMasked(const Pose &a, const Pose &b, float weight, const std::vector<float> &mask,
                      Pose &out) {
  const size_t count = std::min(a.local.size(), b.local.size());
  std::vector<BoneTransform> mixed(a.local.begin(), a.local.end());
  for (size_t i = 0; i < count && i < mask.size(); ++i) {
    if (const float w = weight * mask[i]; w > 0.0f)
      mixed[i] = blendBone(a.local[i], b.local[i], w);
  }
  out.local = std::move(mixed);
}

namespace {

// A bone's rotation in the skeleton's space, without its scale.
glm::quat rotationOf(const glm::mat4 &m) {
  return glm::normalize(glm::quat_cast(glm::mat3(glm::normalize(glm::vec3(m[0])),
                                                 glm::normalize(glm::vec3(m[1])),
                                                 glm::normalize(glm::vec3(m[2])))));
}

// The shortest turn taking direction from to direction to.
glm::quat turnBetween(glm::vec3 from, glm::vec3 to) {
  from = glm::normalize(from);
  to = glm::normalize(to);
  const float d = glm::dot(from, to);
  if (d < -0.99999f) {
    glm::vec3 side = glm::cross(glm::vec3(1, 0, 0), from);
    if (glm::length(side) < 1e-4f)
      side = glm::cross(glm::vec3(0, 1, 0), from);
    return glm::angleAxis(glm::pi<float>(), glm::normalize(side));
  }
  const glm::vec3 c = glm::cross(from, to);
  return glm::normalize(glm::quat(1.0f + d, c.x, c.y, c.z));
}

} // namespace

void turnBones(const Skeleton &skeleton, Pose &pose, const std::vector<int> &bones, glm::vec3 axis,
               float angle) {
  if (bones.empty())
    return;
  const std::vector<glm::mat4> model = modelSpace(skeleton, pose);
  const float share = angle / static_cast<float>(bones.size());
  // Turning a bone's own frame about its axis in its own terms turns it about
  // the skeleton's axis; every turn is about the same axis, so the ones above
  // don't change what that axis is in its terms.
  for (int bone : bones) {
    if (bone < 0 || bone >= static_cast<int>(pose.local.size()))
      continue;
    const glm::vec3 own = glm::inverse(rotationOf(model[bone])) * axis;
    pose.local[bone].rotation = glm::normalize(pose.local[bone].rotation * glm::angleAxis(share, own));
  }
}

void twoBoneIk(const Skeleton &skeleton, Pose &pose, int upper, int lower, int end,
               glm::vec3 target, glm::vec3 pole) {
  const int count = static_cast<int>(pose.local.size());
  if (upper < 0 || lower < 0 || end < 0 || upper >= count || lower >= count || end >= count)
    return;
  const std::vector<glm::mat4> model = modelSpace(skeleton, pose);
  const glm::vec3 a(model[upper][3]), b(model[lower][3]), c(model[end][3]);
  const float l1 = glm::length(b - a), l2 = glm::length(c - b);
  const float reach = glm::length(target - a);
  if (l1 < 1e-6f || l2 < 1e-6f || reach < 1e-6f)
    return;
  // Where the elbow and the hand go: the hand on the line to the target, as
  // far as the arm reaches; the elbow off that line toward the pole, by the
  // triangle the two bones make.
  const glm::vec3 u = (target - a) / reach;
  const float d = std::clamp(reach, std::abs(l1 - l2), l1 + l2);
  const float cosine = std::clamp((l1 * l1 + d * d - l2 * l2) / (2.0f * l1 * d), -1.0f, 1.0f);
  glm::vec3 bend = pole - u * glm::dot(pole, u);
  if (glm::length(bend) < 1e-5f)
    bend = (b - a) - u * glm::dot(b - a, u);
  if (glm::length(bend) < 1e-5f)
    bend = glm::cross(u, glm::abs(u.y) < 0.9f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0));
  bend = glm::normalize(bend);
  const glm::vec3 elbow = a + l1 * (cosine * u + std::sqrt(1.0f - cosine * cosine) * bend);
  const glm::vec3 hand = a + u * d;
  // Turn the upper bone so its child lands on the elbow, then the lower one
  // so its child lands on the hand: turns in the skeleton's space, taken
  // into each bone's own.
  const glm::quat first = turnBetween(b - a, elbow - a);
  const glm::quat second = turnBetween(a + first * (c - a) - elbow, hand - elbow);
  const glm::quat parent =
      skeleton.bones[upper].parent >= 0 ? rotationOf(model[skeleton.bones[upper].parent])
                                        : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  const glm::quat upperWorld = rotationOf(model[upper]);
  pose.local[upper].rotation =
      glm::normalize(glm::inverse(parent) * first * parent * pose.local[upper].rotation);
  pose.local[lower].rotation = glm::normalize(glm::inverse(upperWorld) * glm::inverse(first) *
                                              second * first * upperWorld *
                                              pose.local[lower].rotation);
}

std::vector<float> speedBlend(const std::vector<float> &speeds, float speed) {
  std::vector<float> weights(speeds.size(), 0.0f);
  if (speeds.empty())
    return weights;
  if (speed <= speeds.front()) {
    weights.front() = 1.0f;
    return weights;
  }
  if (speed >= speeds.back()) {
    weights.back() = 1.0f;
    return weights;
  }
  for (size_t i = 0; i + 1 < speeds.size(); ++i) {
    if (speed < speeds[i + 1]) {
      const float span = speeds[i + 1] - speeds[i];
      const float along = span > 0.0f ? (speed - speeds[i]) / span : 0.0f;
      weights[i] = 1.0f - along;
      weights[i + 1] = along;
      break;
    }
  }
  return weights;
}

DirectionWeights directionBlend(glm::vec2 direction) {
  DirectionWeights w;
  const float total = std::abs(direction.x) + std::abs(direction.y);
  if (total < 1e-6f) {
    w.forward = 1.0f;
    return w;
  }
  w.forward = std::max(direction.y, 0.0f) / total;
  w.back = std::max(-direction.y, 0.0f) / total;
  w.right = std::max(direction.x, 0.0f) / total;
  w.left = std::max(-direction.x, 0.0f) / total;
  return w;
}
