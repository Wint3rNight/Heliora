#pragma once

#include "Model.h"
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

  void cleanup();

  MeshModel *getModel(int modelId);

private:
  std::vector<MeshModel> models;
  std::unordered_map<std::string, int> modelMap;
};
