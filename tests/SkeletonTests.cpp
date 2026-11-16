#include "Skeleton.h"
#include "TestRunner.h"

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

namespace {

constexpr float kHalfPi = 1.5707963f;

glm::vec3 origin(const glm::mat4 &m) { return glm::vec3(m[3]); }

// A root 1 m up, turned 90 degrees about z, and a child 1 m along the
// root's x.
Skeleton twoBoneChain() {
  Skeleton s;
  Bone root;
  root.name = "root";
  root.bind.translation = glm::vec3(0.0f, 1.0f, 0.0f);
  root.bind.rotation = glm::angleAxis(kHalfPi, glm::vec3(0, 0, 1));
  Bone child;
  child.name = "child";
  child.parent = 0;
  child.bind.translation = glm::vec3(1.0f, 0.0f, 0.0f);
  s.bones = {root, child};
  return s;
}

} // namespace

TEST(children_are_placed_through_their_parents) {
  const Skeleton s = twoBoneChain();
  const std::vector<glm::mat4> model = modelSpace(s, bindPose(s));
  CHECK(model.size() == 2);
  CHECK_NEAR(glm::distance(origin(model[0]), glm::vec3(0, 1, 0)), 0.0, 1e-5);
  // The root's x points up after its turn, so the child sits 1 m above it.
  CHECK_NEAR(glm::distance(origin(model[1]), glm::vec3(0, 2, 0)), 0.0, 1e-5);
}

TEST(a_parents_scale_stretches_its_children) {
  Skeleton s = twoBoneChain();
  s.bones[0].bind.scale = glm::vec3(2.0f);
  const std::vector<glm::mat4> model = modelSpace(s, bindPose(s));
  CHECK_NEAR(glm::distance(origin(model[1]), glm::vec3(0, 3, 0)), 0.0, 1e-5);
}

TEST(skinning_matrices_are_the_identity_in_the_bind_pose) {
  Skeleton s = twoBoneChain();
  const std::vector<glm::mat4> bind = modelSpace(s, bindPose(s));
  for (size_t i = 0; i < s.bones.size(); ++i)
    s.bones[i].inverseBind = glm::inverse(bind[i]);
  const std::vector<glm::mat4> skin = skinningMatrices(s, bind);
  for (const glm::mat4 &m : skin)
    for (int c = 0; c < 4; ++c)
      for (int r = 0; r < 4; ++r)
        CHECK_NEAR(m[c][r], c == r ? 1.0 : 0.0, 1e-5);
}

TEST(a_bent_pose_moves_what_the_bone_carries) {
  // Turning the child a quarter turn about z swings a point 1 m along its x
  // to 1 m along its y, both seen from the child.
  Skeleton s = twoBoneChain();
  const std::vector<glm::mat4> bind = modelSpace(s, bindPose(s));
  for (size_t i = 0; i < s.bones.size(); ++i)
    s.bones[i].inverseBind = glm::inverse(bind[i]);
  Pose bent = bindPose(s);
  bent.local[1].rotation = glm::angleAxis(kHalfPi, glm::vec3(0, 0, 1));
  const std::vector<glm::mat4> skin = skinningMatrices(s, modelSpace(s, bent));
  // In the bind pose the point is 1 m past the child: (0, 3, 0).
  const glm::vec3 moved = glm::vec3(skin[1] * glm::vec4(0, 3, 0, 1));
  // The child's x points up, so its y points -x: the point swings to
  // (-1, 2, 0).
  CHECK_NEAR(glm::distance(moved, glm::vec3(-1, 2, 0)), 0.0, 1e-5);
}

TEST(bones_are_found_by_name) {
  const Skeleton s = twoBoneChain();
  CHECK(s.find("child") == 1);
  CHECK(s.find("root") == 0);
  CHECK(s.find("tail") == -1);
}

TEST(a_bones_scale_comes_before_its_rotation) {
  // Stretched along its own x, then turned: the child 1 m along x ends up
  // 2 m up. Turned first, it would only be 1 m up.
  Skeleton s = twoBoneChain();
  s.bones[0].bind.scale = glm::vec3(2.0f, 1.0f, 1.0f);
  const std::vector<glm::mat4> model = modelSpace(s, bindPose(s));
  CHECK_NEAR(glm::distance(origin(model[1]), glm::vec3(0, 3, 0)), 0.0, 1e-5);
}
