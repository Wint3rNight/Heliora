#include "PoseBlend.h"

#include <algorithm>
#include <cmath>

void blendPoses(const Pose &a, const Pose &b, float weight, Pose &out) {
  const size_t count = std::min(a.local.size(), b.local.size());
  std::vector<BoneTransform> mixed(a.local.begin(), a.local.end());
  for (size_t i = 0; i < count; ++i) {
    const BoneTransform &from = a.local[i];
    const BoneTransform &to = b.local[i];
    // q and -q are the same rotation; mixing towards the nearer one is the
    // short way.
    const glm::quat target =
        glm::dot(from.rotation, to.rotation) < 0.0f ? -to.rotation : to.rotation;
    mixed[i].translation = glm::mix(from.translation, to.translation, weight);
    mixed[i].scale = glm::mix(from.scale, to.scale, weight);
    mixed[i].rotation = glm::normalize(from.rotation * (1.0f - weight) +
                                       target * weight);
  }
  // Built apart and moved in last, so out can be a or b.
  out.local = std::move(mixed);
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
