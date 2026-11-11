#pragma once

#include "Scene.h"

#include <glm/glm.hpp>
#include <functional>
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
  // Position and rotation from physics. Scale stays the scene's.
  void setPose(glm::vec3 position, glm::quat rotation);
  glm::mat4 getLocalTransform() const;
  glm::mat4 getGlobalTransform() const;
  // Cached transpose(inverse(globalTransform)), recomputed in update()
  // when the node or its parent changed.
  glm::mat4 getNormalMatrix() const;

  // Recomputes world matrices, but only for nodes whose own transform or
  // parent changed since the last call, plus animated ones. A world full of
  // still or sleeping objects costs a walk of the tree, not a matrix inverse
  // per node.
  void update(const glm::mat4 &parentTransform, float timeSeconds = 0.0f);

  // Hierarchy
  void addChild(std::unique_ptr<SceneNode> child);
  const std::vector<std::unique_ptr<SceneNode>> &getChildren() const;
  void clearChildren();
  // Drops every child the test picks.
  void removeChildren(const std::function<bool(const SceneNode &)> &which);

  // Rendering
  void setModelId(int id);
  int getModelId() const;
  void setEntry(SceneEntryRef ref) { entryRef = ref; }
  SceneEntryRef entry() const { return entryRef; }
  // Where the owner keeps this node's physics body; -1 for none.
  void setPhysicsLink(int index) { physicsLinkIndex = index; }
  int physicsLink() const { return physicsLinkIndex; }

private:
  glm::mat4 composeAnimatedLocal(float timeSeconds) const;
  void updateTree(const glm::mat4 &parentTransform, float timeSeconds,
                  bool parentChanged);

  glm::mat4 localTransform;
  glm::mat4 globalTransform;
  glm::mat4 cachedNormalMatrix;
  glm::mat4 lastParentTransform; // what update() was last called with
  bool localDirty = true;        // local changed since the last update()

  NodeTransform baseTransform;
  NodeAnimation animation;
  bool hasAnimation = false;

  SceneEntryRef entryRef; // none for thrown balls
  int physicsLinkIndex = -1;
  int modelId = -1; // References a model in the ModelManager

  std::vector<std::unique_ptr<SceneNode>> children;
};
