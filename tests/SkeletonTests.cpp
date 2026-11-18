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

TEST(mixamo_bones_match_whatever_number_their_rig_prefix_has) {
  // Mixamo numbers the rig per download: "mixamorig:", "mixamorig7:".
  CHECK(sameBoneName("mixamorig:Hips", "mixamorig:Hips"));
  CHECK(sameBoneName("mixamorig7:Hips", "mixamorig:Hips"));
  CHECK(sameBoneName("mixamorig:LeftUpLeg", "mixamorig12:LeftUpLeg"));
  CHECK(!sameBoneName("mixamorig7:Hips", "mixamorig:Spine"));
  // A bare name is another rig (MoCap Online's), posed differently: no match.
  CHECK(!sameBoneName("Hips", "mixamorig:Hips"));
  CHECK(!sameBoneName("mixamorig:Hips", "Hips"));

  Skeleton s;
  for (const char *name : {"mixamorig:Hips", "mixamorig:Spine"}) {
    Bone b;
    b.name = name;
    s.bones.push_back(b);
  }
  CHECK(s.find("mixamorig7:Spine") == 1);
  CHECK(s.find("mixamorig7:Hips") == 0);
  CHECK(s.find("Spine") == -1);
  CHECK(s.find("mixamorig7:Head") == -1);
}

TEST(a_bones_scale_comes_before_its_rotation) {
  // Stretched along its own x, then turned: the child 1 m along x ends up
  // 2 m up. Turned first, it would only be 1 m up.
  Skeleton s = twoBoneChain();
  s.bones[0].bind.scale = glm::vec3(2.0f, 1.0f, 1.0f);
  const std::vector<glm::mat4> model = modelSpace(s, bindPose(s));
  CHECK_NEAR(glm::distance(origin(model[1]), glm::vec3(0, 3, 0)), 0.0, 1e-5);
}

namespace {

// How far apart two rotations are, radians (q and -q are the same).
float apart(glm::quat a, glm::quat b) {
  return 2.0f * std::acos(std::min(1.0f, std::abs(glm::dot(glm::normalize(a), glm::normalize(b)))));
}

const glm::vec3 kX(1, 0, 0), kY(0, 1, 0), kZ(0, 0, 1);

} // namespace

TEST(bind_rotations_in_the_skeletons_space_add_up_down_the_chain) {
  Skeleton s = twoBoneChain();
  s.bones[0].bind.scale = glm::vec3(0.01f); // a Mixamo armature's scale doesn't count
  s.bones[1].bind.rotation = glm::angleAxis(kHalfPi, kX);
  const std::vector<glm::quat> model = bindModelRotations(s);
  CHECK(model.size() == 2);
  if (model.size() != 2)
    return;
  CHECK(apart(model[0], glm::angleAxis(kHalfPi, kZ)) < 1e-5f);
  CHECK(apart(model[1], glm::angleAxis(kHalfPi, kZ) * glm::angleAxis(kHalfPi, kX)) < 1e-5f);
}

TEST(retargeting_between_rigs_with_the_same_rest_changes_nothing) {
  const glm::quat key = glm::angleAxis(0.7f, glm::normalize(glm::vec3(1, 2, 3)));
  const glm::quat bind = glm::angleAxis(0.3f, kY), parent = glm::angleAxis(-0.4f, kX);
  CHECK(apart(retargetRotation(key, bind, bind, parent, parent), key) < 1e-5f);
}

TEST(retargeting_puts_the_rest_pose_on_the_targets_rest_pose) {
  const glm::quat sourceBind = glm::angleAxis(0.6f, kZ), targetBind = glm::angleAxis(-0.2f, kX);
  const glm::quat sourceParent = glm::angleAxis(0.5f, kY), targetParent = glm::angleAxis(0.1f, kZ);
  CHECK(apart(retargetRotation(sourceBind, sourceBind, targetBind, sourceParent, targetParent),
              targetBind) < 1e-5f);
}

TEST(retargeting_turns_a_bone_the_same_way_in_the_skeletons_space) {
  // Two rigs whose shoulder rests differ (one rolled 33 degrees about the
  // bone, like Remy's against the clips' character) and whose arm rests
  // differ too. Every key of the clip, retargeted, must turn the target's
  // arm away from its rest by the same rotation, in the skeleton's space, as
  // the source's arm turned away from its own.
  const glm::quat sourceShoulder = glm::angleAxis(0.4f, kZ);
  const glm::quat targetShoulder = glm::angleAxis(0.4f, kZ) * glm::angleAxis(0.58f, kX);
  const glm::quat sourceArm = glm::angleAxis(-0.5f, kX), targetArm = glm::angleAxis(0.15f, kZ);
  // The clip: the shoulder shrugs, the arm swings down.
  const glm::quat shoulderKey = sourceShoulder * glm::angleAxis(0.2f, kY);
  const glm::quat armKey = sourceArm * glm::angleAxis(1.2f, kZ);

  const glm::quat identity(1, 0, 0, 0);
  const glm::quat shoulderOut =
      retargetRotation(shoulderKey, sourceShoulder, targetShoulder, identity, identity);
  const glm::quat armOut =
      retargetRotation(armKey, sourceArm, targetArm, sourceShoulder, targetShoulder);

  const glm::quat sourceTurn = (shoulderKey * armKey) * glm::inverse(sourceShoulder * sourceArm);
  const glm::quat targetTurn = (shoulderOut * armOut) * glm::inverse(targetShoulder * targetArm);
  CHECK(apart(sourceTurn, targetTurn) < 1e-4f);
}

TEST(a_retargeted_root_position_lands_in_the_same_place) {
  // The hips' parent (the armature) rests turned differently on the two
  // rigs: the hips must still end up at the same place in the skeleton's
  // space.
  const glm::quat sourceParent = glm::angleAxis(-kHalfPi, kX), targetParent(1, 0, 0, 0);
  const glm::vec3 key(0.1f, 0.2f, 1.0f);
  const glm::vec3 out = retargetPosition(key, sourceParent, targetParent);
  CHECK(glm::length(targetParent * out - sourceParent * key) < 1e-5f);
}
