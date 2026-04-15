#include "SceneNode.h"

SceneNode::SceneNode()
    : localTransform(glm::mat4(1.0f)), globalTransform(glm::mat4(1.0f)),
      modelId(-1) {}

void SceneNode::setLocalTransform(const glm::mat4 &transform) {
  localTransform = transform;
}

glm::mat4 SceneNode::getLocalTransform() const { return localTransform; }

glm::mat4 SceneNode::getGlobalTransform() const { return globalTransform; }

void SceneNode::update(const glm::mat4 &parentTransform) {
  globalTransform = parentTransform * localTransform;
  for (auto &child : children) {
    child->update(globalTransform);
  }
}

void SceneNode::addChild(std::unique_ptr<SceneNode> child) {
  children.push_back(std::move(child));
}

const std::vector<std::unique_ptr<SceneNode>> &SceneNode::getChildren() const {
  return children;
}

void SceneNode::setModelId(int id) { modelId = id; }

int SceneNode::getModelId() const { return modelId; }
