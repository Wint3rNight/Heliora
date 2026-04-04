#pragma once
#include "Mesh.h"
#include <assimp/scene.h>
#include <cstddef>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <vector>

class MeshModel {
public:
  MeshModel();
  MeshModel(std::vector<Mesh> meshList);

  size_t getMeshCount();
  Mesh *getMesh(size_t index);

  glm::mat4 getModel();
  void setModel(glm::mat4 newModel);

  void destroyMeshModel();

  static std::vector<std::string> LoadMaterials(const aiScene *scene);
  static std::vector<Mesh> LoadNode(VkPhysicalDevice newPhysicalDevice,
                                    VkDevice newDevice, VkQueue transferQueue,
                                    VkCommandPool transferCommandPool,
                                    aiNode *node, const aiScene *scene,
                                    std::vector<int> matToTex);
  static std::vector<Mesh> LoadMesh(VkPhysicalDevice newPhysicalDevice,
                                    VkDevice newDevice, VkQueue transferQueue,
                                    VkCommandPool transferCommandPool,
                                    aiMesh *mesh, const aiScene *scene,
                                    std::vector<int> matToTex);

  ~MeshModel();

private:
  std::vector<Mesh> mesheList;
  glm::mat4 model;
};
