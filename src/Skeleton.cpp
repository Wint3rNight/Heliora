#include "Skeleton.h"

#include <glm/gtc/matrix_transform.hpp>

glm::mat4 BoneTransform::matrix() const {
  return glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation) *
         glm::scale(glm::mat4(1.0f), scale);
}

bool sameBoneName(std::string_view a, std::string_view b) {
  if (a == b)
    return true;
  const size_t colonA = a.rfind(':'), colonB = b.rfind(':');
  return colonA != std::string_view::npos && colonB != std::string_view::npos &&
         a.substr(colonA + 1) == b.substr(colonB + 1);
}

int Skeleton::find(std::string_view name) const {
  for (size_t i = 0; i < bones.size(); ++i)
    if (bones[i].name == name)
      return static_cast<int>(i);
  for (size_t i = 0; i < bones.size(); ++i)
    if (sameBoneName(bones[i].name, name))
      return static_cast<int>(i);
  return -1;
}

std::vector<glm::quat> bindModelRotations(const Skeleton &skeleton) {
  std::vector<glm::quat> model(skeleton.bones.size(), glm::quat(1, 0, 0, 0));
  for (size_t i = 0; i < skeleton.bones.size(); ++i) {
    const Bone &bone = skeleton.bones[i];
    const glm::quat parent = bone.parent >= 0 ? model[bone.parent] : glm::quat(1, 0, 0, 0);
    model[i] = glm::normalize(parent * bone.bind.rotation);
  }
  return model;
}

glm::quat retargetRotation(glm::quat key, glm::quat sourceBind, glm::quat targetBind,
                           glm::quat sourceParent, glm::quat targetParent) {
  // In the skeleton's space the source bone turned by
  //   turn = (parent * key) * inverse(parent rest * bind rest),
  // and the target bone must turn the same: target = turn * its own rest.
  // Its parent, retargeted the same way, has turned by the same amount, so
  // the parent rotations cancel down to `change`, the difference between
  // the two parents' rests.
  const glm::quat change = glm::inverse(targetParent) * sourceParent;
  return glm::normalize(change * key * glm::inverse(sourceBind) * glm::inverse(change) *
                        targetBind);
}

glm::vec3 retargetPosition(glm::vec3 key, glm::quat sourceParent, glm::quat targetParent) {
  return glm::inverse(targetParent) * sourceParent * key;
}

Pose bindPose(const Skeleton &skeleton) {
  Pose pose;
  pose.local.reserve(skeleton.bones.size());
  for (const Bone &bone : skeleton.bones)
    pose.local.push_back(bone.bind);
  return pose;
}

std::vector<glm::mat4> modelSpace(const Skeleton &skeleton, const Pose &pose) {
  // Parents come first, so each parent is done before its children need it.
  std::vector<glm::mat4> model(skeleton.bones.size());
  for (size_t i = 0; i < skeleton.bones.size(); ++i) {
    const glm::mat4 local = pose.local[i].matrix();
    const int parent = skeleton.bones[i].parent;
    model[i] = parent >= 0 ? model[parent] * local : local;
  }
  return model;
}

std::vector<glm::mat4> skinningMatrices(const Skeleton &skeleton,
                                        const std::vector<glm::mat4> &model) {
  std::vector<glm::mat4> skin(skeleton.bones.size());
  for (size_t i = 0; i < skeleton.bones.size(); ++i)
    skin[i] = model[i] * skeleton.bones[i].inverseBind;
  return skin;
}
