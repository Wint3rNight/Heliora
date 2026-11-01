#pragma once

#include "Model.h"
#include "PrimitiveMesh.h"
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

private:
  std::vector<MeshModel> models;
  std::unordered_map<std::string, int> modelMap;
};
