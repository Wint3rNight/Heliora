#pragma once

#include "Model.h"
#include "PhysicsWorld.h"
#include "PrimitiveMesh.h"
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

class VulkanDevice;
class TextureManager;
class DescriptorManager;

class ModelManager {
public:
  ModelManager() = default;
  ~ModelManager() = default;

  // Non-copyable
  ModelManager(const ModelManager &) = delete;
  ModelManager &operator=(const ModelManager &) = delete;

  // Loads an OBJ file and returns its unique ID. Returns cached ID if already
  // loaded.
  int loadModel(const std::string &modelFile, const VulkanDevice &device,
                TextureManager &textureManager,
                DescriptorManager &descriptorManager);

  // A model from a generated mesh (PrimitiveMesh.h) with the default
  // material. key names the shape, size and colour; check findModel(key)
  // first so the mesh isn't generated again for a shape that's cached.
  int createModel(const std::string &key, MeshData mesh, bool withLods,
                  const VulkanDevice &device, TextureManager &textureManager,
                  DescriptorManager &descriptorManager);
  // -1 if nothing has been loaded or created under this key.
  int findModel(const std::string &key) const;

  void cleanup();

  MeshModel *getModel(int modelId);

  // The model's vertex positions and triangles, one part per mesh, for a
  // physics collider. Built the first time it's asked for, then shared.
  std::shared_ptr<const PhysicsMeshParts> collisionParts(int modelId);

  // The box around all of the model's vertices, in model space, for the
  // editor's selection outline. Built the first time it's asked for.
  struct Bounds {
    glm::vec3 min;
    glm::vec3 max;
  };
  std::optional<Bounds> localBounds(int modelId);

private:
  std::vector<MeshModel> models;
  std::unordered_map<std::string, int> modelMap;
  std::vector<std::shared_ptr<const PhysicsMeshParts>> collisionCache;
  std::vector<std::optional<Bounds>> boundsCache;
};
