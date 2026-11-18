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

// Whether two bone names are the same bone: the same name, or the same
// after a rig prefix both have ("mixamorig:Hips" and "mixamorig7:Hips";
// Mixamo numbers the rig per download). A bare name is another rig.
bool sameBoneName(std::string_view a, std::string_view b);

struct Skeleton {
  std::vector<Bone> bones; // parents before their children
  // The bone with this name (sameBoneName), or -1 when there's none.
  int find(std::string_view name) const;
};

// Each bone's rotation in the skeleton's space in the bind pose: its
// parents' and its own, scale left out.
std::vector<glm::quat> bindModelRotations(const Skeleton &skeleton);

// Retargeting a clip made on one rig onto another with the same bones but a
// different rest pose (two Mixamo characters): a key keeps how far the bone
// turned away from its rest, in the skeleton's space, instead of its raw
// rotation, which on bones resting differently points them elsewhere.
// sourceBind and targetBind are the bone's bind local rotations on the two
// rigs, sourceParent and targetParent its parent's bind rotations in the
// skeleton's space (bindModelRotations; identity for a root). Works key by
// key once every bone is retargeted, since parents then turn alike too.
glm::quat retargetRotation(glm::quat key, glm::quat sourceBind, glm::quat targetBind,
                           glm::quat sourceParent, glm::quat targetParent);
// A key's position in its parent's space, onto the target's parent so it
// lands in the same place in the skeleton's space (the hips).
glm::vec3 retargetPosition(glm::vec3 key, glm::quat sourceParent, glm::quat targetParent);

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
