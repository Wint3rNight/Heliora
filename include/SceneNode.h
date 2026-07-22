#pragma once

#include "Scene.h"

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
  // Decomposed transform + optional animation. When an animation is set the
  // local transform is recomposed from these components every update().
  void setTransform(const NodeTransform &transform);
  void setAnimation(const NodeAnimation &animation);
  glm::mat4 getLocalTransform() const;
  glm::mat4 getGlobalTransform() const;
  // Cached transpose(inverse(globalTransform)) — recomputed in update()
  // when the local transform changes.
  glm::mat4 getNormalMatrix() const;

  void update(const glm::mat4 &parentTransform, float timeSeconds = 0.0f);

  // Hierarchy
  void addChild(std::unique_ptr<SceneNode> child);
  const std::vector<std::unique_ptr<SceneNode>> &getChildren() const;
  void clearChildren();

  // Rendering
  void setModelId(int id);
  int getModelId() const;

private:
  glm::mat4 composeAnimatedLocal(float timeSeconds) const;

  glm::mat4 localTransform;
  glm::mat4 globalTransform;
  glm::mat4 cachedNormalMatrix;

  NodeTransform baseTransform;
  NodeAnimation animation;
  bool hasAnimation = false;

  int modelId = -1; // References a model in the ModelManager

  std::vector<std::unique_ptr<SceneNode>> children;
};
