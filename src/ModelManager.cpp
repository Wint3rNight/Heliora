#include "ModelManager.h"
#include "TextureManager.h"
#include "VulkanDevice.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <filesystem>
#include <stdexcept>

int ModelManager::loadModel(const std::string &modelFile,
                            const VulkanDevice &device,
                            TextureManager &textureManager,
                            DescriptorManager &descriptorManager) {
  // Check cache
  auto it = modelMap.find(modelFile);
  if (it != modelMap.end()) {
    return it->second;
  }

  Assimp::Importer importer;
  const aiScene *scene = nullptr;

  std::vector<std::string> candidatePaths;
  candidatePaths.push_back(modelFile);

  const std::filesystem::path inputPath(modelFile);
  if (!inputPath.is_absolute()) {
    candidatePaths.push_back(
        (std::filesystem::path("..") / inputPath).string());
  }

  for (const std::string &path : candidatePaths) {
    scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs |
                                        aiProcess_JoinIdenticalVertices);
    if (scene && !(scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) &&
        scene->mRootNode) {
      break;
    }
  }

  if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE ||
      !scene->mRootNode) {
    throw std::runtime_error("Failed to load model: " +
                             std::string(importer.GetErrorString()));
  }

  // get vector of all materials with 1:1 id placement
  std::vector<std::string> textureNames = MeshModel::LoadMaterials(scene);

  // conversion from material list id to Material
  std::vector<Material> materials(textureNames.size());

  for (size_t i = 0; i < textureNames.size(); i++) {
    if (textureNames[i].empty()) {
      materials[i].albedoTextureId = 0;
    } else {
      materials[i].albedoTextureId = textureManager.loadTexture(
          textureNames[i], device, descriptorManager);
    }
  }

  std::vector<Mesh> modelMeshes = MeshModel::LoadNode(
      device.getAllocator(), device.getLogicalDevice(),
      device.getGraphicsQueue(), device.getGraphicsCommandPool(),
      scene->mRootNode, scene, materials);

  MeshModel meshModel = MeshModel(std::move(modelMeshes));
  models.push_back(std::move(meshModel));

  int id = models.size() - 1;
  modelMap[modelFile] = id;
  return id;
}

void ModelManager::cleanup() {
  for (auto &model : models) {
    model.destroyMeshModel();
  }
  models.clear();
  modelMap.clear();
}

MeshModel *ModelManager::getModel(int modelId) {
  if (modelId < 0 || modelId >= models.size()) {
    return nullptr;
  }
  return &models[modelId];
}
