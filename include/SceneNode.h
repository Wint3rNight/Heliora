#pragma once

#include <glm/glm.hpp>
#include <memory>
#include <vector>

class MeshModel;

class SceneNode {
public:
  SceneNode();
  ~SceneNode() = default;

  // Transform management
  void setLocalTransform(const glm::mat4 &transform);
  glm::mat4 getLocalTransform() const;
  glm::mat4 getGlobalTransform() const;
  // Cached transpose(inverse(globalTransform)) — recomputed in update()
  // when the local transform changes.
  glm::mat4 getNormalMatrix() const;

  void update(const glm::mat4 &parentTransform);

  // Hierarchy
  void addChild(std::unique_ptr<SceneNode> child);
  const std::vector<std::unique_ptr<SceneNode>> &getChildren() const;

  // Rendering
  void setModelId(int id);
  int getModelId() const;

private:
  glm::mat4 localTransform;
  glm::mat4 globalTransform;
  glm::mat4 cachedNormalMatrix;

  int modelId = -1; // References a model in the ModelManager

  std::vector<std::unique_ptr<SceneNode>> children;
};
