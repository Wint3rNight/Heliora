#pragma once

#include "Material.h"
#include "Utilities.h"
#include <assimp/scene.h>
#include <cstddef>
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

struct Model {
  glm::mat4 model;
};

struct MaterialTextureNames {
  std::string albedo;
  std::string normal;
  std::string metallic;
  std::string roughness;
  std::string ao;
};

class Mesh {
public:
  Mesh();
  Mesh(VmaAllocator allocator, VkDevice device,
       VkQueue transferQueue, VkCommandPool transferCommandPool,
       std::vector<Vertex> *vertices, std::vector<uint32_t> *indices,
       Material newMaterial);

  Mesh(const Mesh &) = delete;
  Mesh &operator=(const Mesh &) = delete;
  Mesh(Mesh &&) noexcept = default;
  Mesh &operator=(Mesh &&) noexcept = default;

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
  AllocatedBuffer vertexBuffer;

  int indexCount;
  AllocatedBuffer indexBuffer;

  VmaAllocator allocator;
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

  MeshModel(const MeshModel &) = delete;
  MeshModel &operator=(const MeshModel &) = delete;
  MeshModel(MeshModel &&) noexcept = default;
  MeshModel &operator=(MeshModel &&) noexcept = default;

  size_t getMeshCount() const;
  const Mesh *getMesh(size_t index) const;

  void destroyMeshModel();

  static std::vector<MaterialTextureNames> LoadMaterials(const aiScene *scene);
  static std::vector<Mesh> LoadNode(VmaAllocator allocator,
                                    VkDevice newDevice, VkQueue transferQueue,
                                    VkCommandPool transferCommandPool,
                                    aiNode *node, const aiScene *scene,
                                    const std::vector<Material> &materials);
  static std::vector<Mesh> LoadMesh(VmaAllocator allocator,
                                    VkDevice newDevice, VkQueue transferQueue,
                                    VkCommandPool transferCommandPool,
                                    aiMesh *mesh, const aiScene *scene,
                                    const std::vector<Material> &materials);

  ~MeshModel();

private:
  std::vector<Mesh> meshList;
};
