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
  bool doubleSided = false;
  bool isCloth = false;
  bool alphaMasked = false;
  bool alphaBlended = false;
  float alphaCutoff = 0.5f;
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
  const std::vector<Vertex> &getCpuVertices() const { return cpuVertices; }

  // LOD 0 = full detail, LOD 1 = ~50%, LOD 2 = ~12%
  int      getIndexCount(int lod = 0) const;
  VkBuffer getIndexBuffer(int lod = 0) const;
  const std::vector<uint32_t> &getCpuIndices(int lod = 0) const;
  int      getLodCount() const;

  // Called after construction to append a reduced-index LOD level
  void addLod(VmaAllocator allocator, VkDevice device,
              VkQueue transferQueue, VkCommandPool transferCommandPool,
              std::vector<uint32_t> *indices);

  void destroyBuffers();

  ~Mesh();

  glm::vec3 boundingCenter = glm::vec3(0.0f);
  float     boundingRadius = 0.0f;

private:
  Material material;

  int vertexCount;
  AllocatedBuffer vertexBuffer;

  int indexCount;
  AllocatedBuffer indexBuffer;
  std::vector<Vertex> cpuVertices;
  std::vector<uint32_t> cpuIndices;

  // Extra LOD index buffers: [0] = LOD1, [1] = LOD2
  std::vector<AllocatedBuffer> extraLodIndexBuffers;
  std::vector<int>             extraLodIndexCounts;
  std::vector<std::vector<uint32_t>> extraLodCpuIndices;

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

  glm::vec3 boundingCenter = glm::vec3(0.0f);
  float     boundingRadius = 0.0f;

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
