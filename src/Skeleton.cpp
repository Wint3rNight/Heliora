#include "Skeleton.h"

#include <glm/gtc/matrix_transform.hpp>

glm::mat4 BoneTransform::matrix() const {
  return glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation) *
         glm::scale(glm::mat4(1.0f), scale);
}

int Skeleton::find(std::string_view name) const {
  for (size_t i = 0; i < bones.size(); ++i)
    if (bones[i].name == name)
      return static_cast<int>(i);
  return -1;
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
