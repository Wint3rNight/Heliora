#pragma once

#include "Material.h"
#include "Utilities.h"
#include <assimp/scene.h>
#include <cstddef>
#include <glm/glm.hpp>
#include <vector>
#include <vulkan/vulkan_core.h>

struct Model {
  glm::mat4 model;
};

class Mesh {
public:
  Mesh();
  Mesh(VkPhysicalDevice newPhysicalDevice, VkDevice newDevice,
       VkQueue transferQueue, VkCommandPool transferCommandPool,
       std::vector<Vertex> *vertices, std::vector<uint32_t> *indices,
       Material newMaterial);

  const Material &getMaterial() const;

  int getVertexCount() const;
  VkBuffer getVertexBuffer() const;

  int getIndexCount() const;
  VkBuffer getIndexBuffer() const;

  void destroyBuffers();

  ~Mesh();

private:
  Material material;

  int vertexCount;
  VkBuffer vertexBuffer;
  VkDeviceMemory vertexBufferMemory;

  int indexCount;
  VkBuffer indexBuffer;
  VkDeviceMemory indexBufferMemory;

  VkPhysicalDevice physicalDevice;
  VkDevice device;

  void createVertexBuffer(VkQueue transferQueue,
                          VkCommandPool transferCommandPool,
                          std::vector<Vertex> *vertices);
  void createIndexBuffer(VkQueue transferQueue,
                         VkCommandPool transferCommandPool,
                         std::vector<uint32_t> *indices);
};

class MeshModel {
public:
  MeshModel();
  MeshModel(std::vector<Mesh> meshList);

  size_t getMeshCount() const;
  const Mesh *getMesh(size_t index) const;

  void destroyMeshModel();

  static std::vector<std::string> LoadMaterials(const aiScene *scene);
  static std::vector<Mesh> LoadNode(VkPhysicalDevice newPhysicalDevice,
                                    VkDevice newDevice, VkQueue transferQueue,
                                    VkCommandPool transferCommandPool,
                                    aiNode *node, const aiScene *scene,
                                    const std::vector<Material> &materials);
  static std::vector<Mesh> LoadMesh(VkPhysicalDevice newPhysicalDevice,
                                    VkDevice newDevice, VkQueue transferQueue,
                                    VkCommandPool transferCommandPool,
                                    aiMesh *mesh, const aiScene *scene,
                                    const std::vector<Material> &materials);

  ~MeshModel();

private:
  std::vector<Mesh> meshList;
};
