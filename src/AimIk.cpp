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

namespace {

// Turns the rig's bones about one axis in the skeleton's space, each its
// share of angle, so the top one ends up turned by all of it.
void turnRig(const Skeleton &skeleton, const AimRig &rig, Pose &pose, glm::vec3 axis, float angle) {
  const std::vector<glm::mat4> model = modelSpace(skeleton, pose);
  // Every turn is about the same axis, so turning one bone doesn't change
  // what that axis is in the terms of the ones above it.
  for (size_t i = 0; i < rig.bones.size(); ++i) {
    const int bone = rig.bones[i];
    const glm::vec3 own = glm::inverse(rotationOf(model[bone])) * axis;
    pose.local[bone].rotation =
        glm::normalize(pose.local[bone].rotation * glm::angleAxis(angle * rig.shares[i], own));
  }
}

glm::vec3 level(glm::vec3 v, glm::vec3 up) { return v - glm::dot(v, up) * up; }

} // namespace

glm::vec3 chestFacing(glm::vec3 leftShoulder, glm::vec3 rightShoulder, glm::vec3 up) {
  const glm::vec3 facing = level(glm::cross(leftShoulder - rightShoulder, up), up);
  return glm::length(facing) > 1e-6f ? glm::normalize(facing) : glm::vec3(0.0f);
}

float squareChest(const Skeleton &skeleton, const AimRig &rig, Pose &pose, int leftShoulder,
                  int rightShoulder, glm::vec3 aim, float blade, float weight, float maxAngle,
                  glm::vec3 up) {
  if (!(weight > 0.0f) || !rig.valid() || leftShoulder < 0 || rightShoulder < 0)
    return 0.0f;
  up = glm::normalize(up);
  const std::vector<glm::mat4> model = modelSpace(skeleton, pose);
  const glm::vec3 facing =
      chestFacing(glm::vec3(model[leftShoulder][3]), glm::vec3(model[rightShoulder][3]), up);
  const glm::vec3 ahead = level(aim, up);
  if (glm::length(facing) < 1e-6f || glm::length(ahead) < 1e-6f)
    return 0.0f;
  // Signed about up: positive turns toward the left.
  const glm::vec3 a = glm::normalize(ahead);
  auto heading = [&](glm::vec3 f) { return std::atan2(glm::dot(up, glm::cross(a, f)), glm::dot(a, f)); };
  const float turn = std::clamp(-blade - heading(facing), -maxAngle, maxAngle) * std::min(weight, 1.0f);
  // About the trunk's own line (the spine's base to between the shoulders),
  // so a leaning spine twists in place instead of its top swinging round
  // the vertical; a leaning axis turns the facing a little less than the
  // angle, so a couple of passes make it up.
  float done = 0.0f;
  for (int pass = 0; pass < 3; ++pass) {
    const std::vector<glm::mat4> m = modelSpace(skeleton, pose);
    const glm::vec3 base = glm::vec3(m[rig.bones.front()][3]);
    const glm::vec3 top = 0.5f * (glm::vec3(m[leftShoulder][3]) + glm::vec3(m[rightShoulder][3]));
    glm::vec3 trunk = top - base;
    trunk = glm::length(trunk) > 1e-6f && glm::dot(trunk, up) > 0.0f ? glm::normalize(trunk) : up;
    const float left = turn - done;
    if (std::abs(left) < 1e-4f)
      break;
    const float before = heading(chestFacing(glm::vec3(m[leftShoulder][3]), glm::vec3(m[rightShoulder][3]), up));
    turnRig(skeleton, rig, pose, trunk, left);
    const std::vector<glm::mat4> after = modelSpace(skeleton, pose);
    done += heading(chestFacing(glm::vec3(after[leftShoulder][3]), glm::vec3(after[rightShoulder][3]), up)) - before;
  }
  return done;
}

void bendChest(const Skeleton &skeleton, const AimRig &rig, Pose &pose, int leftShoulder,
               int rightShoulder, float angle, glm::vec3 up) {
  if (angle == 0.0f || !rig.valid() || leftShoulder < 0 || rightShoulder < 0)
    return;
  up = glm::normalize(up);
  const std::vector<glm::mat4> model = modelSpace(skeleton, pose);
  const glm::vec3 right =
      level(glm::vec3(model[rightShoulder][3]) - glm::vec3(model[leftShoulder][3]), up);
  if (glm::length(right) < 1e-6f)
    return;
  // About the line toward the right shoulder, a positive turn tips the
  // chest's front up.
  turnRig(skeleton, rig, pose, glm::normalize(right), angle);
}

glm::mat4 aimedGun(glm::vec3 aim, glm::vec3 up, glm::vec3 anchorInGun, glm::vec3 anchor) {
  const glm::vec3 z = glm::normalize(aim);
  glm::vec3 x = glm::cross(up, z);
  if (glm::length(x) < 1e-6f)
    x = glm::cross(glm::vec3(0.0f, 0.0f, 1.0f), z);
  x = glm::normalize(x);
  const glm::vec3 y = glm::cross(z, x);
  glm::mat4 g(glm::vec4(x, 0.0f), glm::vec4(y, 0.0f), glm::vec4(z, 0.0f), glm::vec4(0, 0, 0, 1));
  g[3] = glm::vec4(anchor - glm::mat3(g) * anchorInGun, 1.0f);
  return g;
}
