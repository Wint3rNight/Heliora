#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <string>
#include <string_view>
#include <vector>

// A character's skeleton and its poses, with no Vulkan and no file format,
// so the maths is tested like the rest of heliora_core.

// A bone's place relative to its parent: scale first, then rotation, then
// translation.
struct BoneTransform {
  glm::vec3 translation{0.0f};
  glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
  glm::vec3 scale{1.0f};
  glm::mat4 matrix() const;
};

struct Bone {
  std::string name;
  int parent = -1;     // lower than this bone's own index; -1 for a root
  BoneTransform bind;  // the bind pose, relative to the parent
  // From the mesh's space into this bone's, in the bind pose. The identity
  // for nodes no vertex follows.
  glm::mat4 inverseBind{1.0f};
};

struct Skeleton {
  std::vector<Bone> bones; // parents before their children
  int find(std::string_view name) const; // -1 when there's none
};

// Every bone's local transform at one moment.
struct Pose {
  std::vector<BoneTransform> local;
};

Pose bindPose(const Skeleton &skeleton);
// Each bone's transform in the skeleton's space, down the parent chain.
std::vector<glm::mat4> modelSpace(const Skeleton &skeleton, const Pose &pose);
// What each bone does to the vertices it carries: model space times inverse
// bind. The identity for every bone in the bind pose.
std::vector<glm::mat4> skinningMatrices(const Skeleton &skeleton,
                                        const std::vector<glm::mat4> &model);
