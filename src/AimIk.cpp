#include "AimIk.h"

#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string_view>

namespace {

int boneNamed(const Skeleton &skeleton, std::string_view name) {
  for (int i = 0; i < static_cast<int>(skeleton.bones.size()); ++i) {
    const std::string_view bone = skeleton.bones[i].name;
    if (bone.size() < name.size() || bone.substr(bone.size() - name.size()) != name)
      continue;
    if (bone.size() == name.size() ||
        !std::isalnum(static_cast<unsigned char>(bone[bone.size() - name.size() - 1])))
      return i;
  }
  return -1;
}

glm::quat rotationOf(const glm::mat4 &m) {
  return glm::normalize(glm::quat_cast(glm::mat3(glm::normalize(glm::vec3(m[0])),
                                                 glm::normalize(glm::vec3(m[1])),
                                                 glm::normalize(glm::vec3(m[2])))));
}

// The shortest turn from a to b (unit).
glm::quat between(glm::vec3 a, glm::vec3 b) {
  const float c = std::clamp(glm::dot(a, b), -1.0f, 1.0f);
  glm::vec3 axis = glm::cross(a, b);
  if (glm::length(axis) < 1e-6f) {
    if (c > 0.0f)
      return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    axis = std::abs(a.y) < 0.9f ? glm::cross(a, glm::vec3(0, 1, 0)) : glm::cross(a, glm::vec3(1, 0, 0));
  }
  return glm::angleAxis(std::acos(c), glm::normalize(axis));
}

glm::quat part(const glm::quat &q, float share) {
  return glm::slerp(glm::quat(1.0f, 0.0f, 0.0f, 0.0f), q, share);
}

} // namespace

AimRig findAimRig(const Skeleton &skeleton) {
  AimRig rig;
  const float shares[] = {0.2f, 0.3f, 0.5f};
  const char *names[] = {"Spine", "Spine1", "Spine2"};
  for (int i = 0; i < 3; ++i)
    if (const int bone = boneNamed(skeleton, names[i]); bone >= 0) {
      rig.bones.push_back(bone);
      rig.shares.push_back(shares[i]);
    }
  // Shares of what's there, still adding up to 1.
  float total = 0.0f;
  for (float s : rig.shares)
    total += s;
  for (float &s : rig.shares)
    s /= std::max(total, 1e-6f);
  return rig;
}

float turnToward(const Skeleton &skeleton, Pose &pose, int bone, glm::vec3 axis, glm::vec3 target,
                 float weight, float maxAngle) {
  if (!(weight > 0.0f) || bone < 0 || bone >= static_cast<int>(skeleton.bones.size()))
    return 0.0f;
  const std::vector<glm::mat4> model = modelSpace(skeleton, pose);
  const glm::quat turn = rotationOf(model[bone]);
  const glm::vec3 to = target - glm::vec3(model[bone][3]);
  if (glm::length(to) < 1e-5f || glm::length(axis) < 1e-6f)
    return 0.0f;
  glm::quat q = between(glm::normalize(turn * glm::normalize(axis)), glm::normalize(to));
  const float angle = glm::angle(q);
  if (angle > maxAngle)
    q = part(q, maxAngle / angle);
  q = part(q, std::min(weight, 1.0f));
  const int parent = skeleton.bones[bone].parent;
  const glm::quat parentTurn = parent >= 0 ? rotationOf(model[parent]) : glm::quat(1, 0, 0, 0);
  pose.local[bone].rotation = glm::normalize(glm::inverse(parentTurn) * (q * turn));
  return glm::angle(q);
}

float aimAt(const Skeleton &skeleton, const AimRig &rig, Pose &pose, const BarrelFinder &barrel,
            glm::vec3 target, float weight, float maxAngle, int passes) {
  if (!(weight > 0.0f) || !rig.valid() || !barrel)
    return 0.0f;
  // Solved in full on a copy, then laid on by weight.
  Pose solved = pose;
  glm::quat total(1.0f, 0.0f, 0.0f, 0.0f);
  for (int pass = 0; pass < passes; ++pass) {
    const Barrel b = barrel(modelSpace(skeleton, solved));
    const glm::vec3 toTarget = target - b.muzzle;
    if (glm::length(toTarget) < 1e-4f || glm::length(b.forward) < 1e-6f)
      break;
    glm::quat q = between(glm::normalize(b.forward), glm::normalize(toTarget));
    // Held to the limit, over all the passes.
    const float sofar = glm::angle(total);
    const float more = glm::angle(q);
    if (sofar + more > maxAngle)
      q = part(q, more > 1e-6f ? std::max(0.0f, maxAngle - sofar) / more : 0.0f);
    if (glm::angle(q) < 1e-6f)
      break;
    total = q * total;
    // Each spine bone its share, about its own joint in the skeleton's space.
    for (size_t i = 0; i < rig.bones.size(); ++i) {
      const int bone = rig.bones[i];
      const std::vector<glm::mat4> model = modelSpace(skeleton, solved);
      const int parent = skeleton.bones[bone].parent;
      const glm::quat parentTurn = parent >= 0 ? rotationOf(model[parent]) : glm::quat(1, 0, 0, 0);
      const glm::quat turned = part(q, rig.shares[i]) * rotationOf(model[bone]);
      solved.local[bone].rotation = glm::normalize(glm::inverse(parentTurn) * turned);
    }
  }
  const float w = std::min(weight, 1.0f);
  for (int bone : rig.bones)
    pose.local[bone].rotation =
        glm::normalize(glm::slerp(pose.local[bone].rotation, solved.local[bone].rotation, w));
  return glm::angle(total) * w;
}
