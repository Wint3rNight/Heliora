#include "SceneNode.h"
#include "TestRunner.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <memory>

namespace {

float maxDifference(const glm::mat4 &a, const glm::mat4 &b) {
  float worst = 0.0f;
  for (int c = 0; c < 4; ++c)
    for (int r = 0; r < 4; ++r)
      worst = std::fmax(worst, std::fabs(a[c][r] - b[c][r]));
  return worst;
}

const glm::vec3 kAngles[] = {
    {0, 0, 0}, {0, 90, 0}, {30, 0, 0}, {0, 0, 45}, {10, 20, 30}, {-70, 135, 15}};

} // namespace

TEST(rotation_from_degrees_matches_the_rotate_chain) {
  // The order scene files have always used: Y, then X, then Z.
  for (glm::vec3 d : kAngles) {
    glm::mat4 chain(1.0f);
    chain = glm::rotate(chain, glm::radians(d.y), glm::vec3(0, 1, 0));
    chain = glm::rotate(chain, glm::radians(d.x), glm::vec3(1, 0, 0));
    chain = glm::rotate(chain, glm::radians(d.z), glm::vec3(0, 0, 1));
    CHECK(maxDifference(glm::mat4_cast(rotationFromDegrees(d)), chain) < 1e-5f);
  }
}

TEST(pose_gives_the_same_matrix_as_the_scene_transform) {
  // When physics takes over a node, the first pose it writes is the body's
  // spawn pose. If the two paths disagreed, every object would jump.
  for (glm::vec3 d : kAngles) {
    NodeTransform t;
    t.position = glm::vec3(1.0f, 2.0f, -3.0f);
    t.rotationDeg = d;
    t.scale = 1.5f;

    // fromPose starts at the origin with no rotation and only gets the
    // position and rotation through setPose. Scale comes from its transform.
    NodeTransform scaleOnly;
    scaleOnly.scale = t.scale;
    SceneNode fromScene, fromPose;
    fromScene.setTransform(t);
    fromPose.setTransform(scaleOnly);
    fromPose.setPose(t.position, rotationFromDegrees(d));
    fromScene.update(glm::mat4(1.0f));
    fromPose.update(glm::mat4(1.0f));

    CHECK(maxDifference(fromScene.getGlobalTransform(),
                        fromPose.getGlobalTransform()) < 1e-5f);
  }
}

TEST(pose_keeps_the_scene_scale) {
  NodeTransform t;
  t.scale = 2.0f;
  SceneNode node;
  node.setTransform(t);
  const glm::quat q = glm::angleAxis(0.7f, glm::vec3(0, 1, 0));
  node.setPose(glm::vec3(4.0f, 5.0f, 6.0f), q);
  node.update(glm::mat4(1.0f));

  const glm::mat4 expected = glm::translate(glm::mat4(1.0f), glm::vec3(4, 5, 6)) *
                             glm::mat4_cast(q) *
                             glm::scale(glm::mat4(1.0f), glm::vec3(2.0f));
  CHECK(maxDifference(node.getGlobalTransform(), expected) < 1e-5f);
}

TEST(node_change_after_update_is_picked_up) {
  SceneNode node;
  node.update(glm::mat4(1.0f));
  node.setPose(glm::vec3(0.0f, 3.0f, 0.0f), glm::quat(1, 0, 0, 0));
  node.update(glm::mat4(1.0f));
  CHECK_NEAR(node.getGlobalTransform()[3].y, 3.0, 1e-6);

  node.setLocalTransform(glm::translate(glm::mat4(1.0f), glm::vec3(0, 7, 0)));
  node.update(glm::mat4(1.0f));
  CHECK_NEAR(node.getGlobalTransform()[3].y, 7.0, 1e-6);
}

TEST(child_follows_a_moved_parent) {
  SceneNode parent;
  auto child = std::make_unique<SceneNode>();
  child->setLocalTransform(glm::translate(glm::mat4(1.0f), glm::vec3(1, 0, 0)));
  SceneNode *childPtr = child.get();
  parent.addChild(std::move(child));
  parent.update(glm::mat4(1.0f));

  // Only the parent changes. The child must still pick up the new parent.
  parent.setPose(glm::vec3(0.0f, 2.0f, 0.0f), glm::quat(1, 0, 0, 0));
  parent.update(glm::mat4(1.0f));
  CHECK_NEAR(childPtr->getGlobalTransform()[3].x, 1.0, 1e-6);
  CHECK_NEAR(childPtr->getGlobalTransform()[3].y, 2.0, 1e-6);
}
