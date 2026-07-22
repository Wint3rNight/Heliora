#include "SceneNode.h"

#include <cmath>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace {

glm::mat4 composeTransform(const NodeTransform &t) {
  glm::mat4 m = glm::translate(glm::mat4(1.0f), t.position);
  m = glm::rotate(m, glm::radians(t.rotationDeg.y), glm::vec3(0, 1, 0));
  m = glm::rotate(m, glm::radians(t.rotationDeg.x), glm::vec3(1, 0, 0));
  m = glm::rotate(m, glm::radians(t.rotationDeg.z), glm::vec3(0, 0, 1));
  m = glm::scale(m, glm::vec3(t.scale));
  return m;
}

} // namespace

SceneNode::SceneNode()
    : localTransform(glm::mat4(1.0f)), globalTransform(glm::mat4(1.0f)),
      cachedNormalMatrix(glm::mat4(1.0f)), modelId(-1) {}

void SceneNode::setLocalTransform(const glm::mat4 &transform) {
  localTransform = transform;
  hasAnimation = false;
}

void SceneNode::setTransform(const NodeTransform &transform) {
  baseTransform = transform;
  localTransform = composeTransform(transform);
}

void SceneNode::setAnimation(const NodeAnimation &anim) {
  animation = anim;
  hasAnimation = anim.active();
}

glm::mat4 SceneNode::getLocalTransform() const { return localTransform; }

glm::mat4 SceneNode::getGlobalTransform() const { return globalTransform; }

glm::mat4 SceneNode::getNormalMatrix() const { return cachedNormalMatrix; }

glm::mat4 SceneNode::composeAnimatedLocal(float timeSeconds) const {
  NodeTransform t = baseTransform;

  if (animation.orbitDegPerSec != 0.0f && animation.orbitRadius != 0.0f) {
    float angle = glm::radians(animation.orbitDegPerSec) * timeSeconds;
    t.position += glm::vec3(std::cos(angle), 0.0f, std::sin(angle)) *
                  animation.orbitRadius;
  }
  if (animation.bobAmplitude != 0.0f && animation.bobFrequency != 0.0f) {
    t.position.y += animation.bobAmplitude *
                    std::sin(glm::two_pi<float>() * animation.bobFrequency *
                             timeSeconds);
  }

  glm::mat4 m = composeTransform(t);
  glm::vec3 spin = glm::radians(animation.spinDegPerSec) * timeSeconds;
  if (spin != glm::vec3(0.0f)) {
    // Spin rotates the model in place, after base orientation. Scale is
    // uniform, so applying spin inside the scaled basis introduces no skew.
    m = glm::rotate(m, spin.y, glm::vec3(0, 1, 0));
    m = glm::rotate(m, spin.x, glm::vec3(1, 0, 0));
    m = glm::rotate(m, spin.z, glm::vec3(0, 0, 1));
  }
  return m;
}

void SceneNode::update(const glm::mat4 &parentTransform, float timeSeconds) {
  if (hasAnimation)
    localTransform = composeAnimatedLocal(timeSeconds);

  globalTransform    = parentTransform * localTransform;
  cachedNormalMatrix = glm::transpose(glm::inverse(globalTransform));
  for (auto &child : children) {
    child->update(globalTransform, timeSeconds);
  }
}

void SceneNode::addChild(std::unique_ptr<SceneNode> child) {
  children.push_back(std::move(child));
}

const std::vector<std::unique_ptr<SceneNode>> &SceneNode::getChildren() const {
  return children;
}

void SceneNode::clearChildren() { children.clear(); }

void SceneNode::setModelId(int id) { modelId = id; }

int SceneNode::getModelId() const { return modelId; }
