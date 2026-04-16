#include "Model.h"
#include <stdexcept>
#include <cstring>
#include <iterator>

// =============================================================================
// Mesh Implementation
// =============================================================================

Mesh::Mesh()
    : allocator(VK_NULL_HANDLE), device(VK_NULL_HANDLE), vertexCount(0),
      indexCount(0) {}

Mesh::Mesh(VmaAllocator newAllocator, VkDevice newDevice,
           VkQueue transferQueue, VkCommandPool transferCommandPool,
           std::vector<Vertex> *vertices, std::vector<uint32_t> *indices,
           Material newMaterial) {
  vertexCount = vertices->size();
  indexCount = indices->size();
  allocator = newAllocator;
  device = newDevice;
  material = newMaterial;
  createVertexBuffer(transferQueue, transferCommandPool, vertices);
  createIndexBuffer(transferQueue, transferCommandPool, indices);
}

const Material &Mesh::getMaterial() const { return material; }
int Mesh::getVertexCount() const { return vertexCount; }
VkBuffer Mesh::getVertexBuffer() const { return vertexBuffer.get(); }
int Mesh::getIndexCount() const { return indexCount; }
VkBuffer Mesh::getIndexBuffer() const { return indexBuffer.get(); }

void Mesh::destroyBuffers() {
  vertexBuffer.reset();
  indexBuffer.reset();
}

Mesh::~Mesh() {}

void Mesh::createVertexBuffer(VkQueue transferQueue,
                              VkCommandPool transferCommandPool,
                              std::vector<Vertex> *vertices) {
  VkDeviceSize bufferSize = sizeof(Vertex) * vertices->size();

  AllocatedBuffer stagingBuffer;
  createBuffer(allocator, bufferSize,
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VMA_MEMORY_USAGE_AUTO,
               VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
               &stagingBuffer);

  void *data;
  vmaMapMemory(allocator, stagingBuffer.getAllocation(), &data);
  memcpy(data, vertices->data(), (size_t)bufferSize);
  vmaUnmapMemory(allocator, stagingBuffer.getAllocation());

  createBuffer(
      allocator, bufferSize,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
      0, &vertexBuffer);

  copyBuffer(device, transferQueue, transferCommandPool, stagingBuffer.get(),
             vertexBuffer.get(), bufferSize);
}

void Mesh::createIndexBuffer(VkQueue transferQueue,
                             VkCommandPool transferCommandPool,
                             std::vector<uint32_t> *indices) {
  VkDeviceSize bufferSize = sizeof(uint32_t) * indices->size();

  AllocatedBuffer stagingBuffer;
  createBuffer(allocator, bufferSize,
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VMA_MEMORY_USAGE_AUTO,
               VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
               &stagingBuffer);

  void *data;
  vmaMapMemory(allocator, stagingBuffer.getAllocation(), &data);
  memcpy(data, indices->data(), (size_t)bufferSize);
  vmaUnmapMemory(allocator, stagingBuffer.getAllocation());

  createBuffer(
      allocator, bufferSize,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
      0, &indexBuffer);

  copyBuffer(device, transferQueue, transferCommandPool, stagingBuffer.get(),
             indexBuffer.get(), bufferSize);
}

// =============================================================================
// MeshModel Implementation
// =============================================================================

MeshModel::MeshModel() {}

MeshModel::MeshModel(std::vector<Mesh> newMeshList) {
  meshList = std::move(newMeshList);
}

size_t MeshModel::getMeshCount() const { return meshList.size(); }

const Mesh *MeshModel::getMesh(size_t index) const {
  if (index >= meshList.size()) {
    throw std::runtime_error("Mesh index out of range");
  }
  return &meshList[index];
}

void MeshModel::destroyMeshModel() {
  for (auto &mesh : meshList) {
    mesh.destroyBuffers();
  }
}

std::vector<std::string> MeshModel::LoadMaterials(const aiScene *scene) {
  std::vector<std::string> textureList(scene->mNumMaterials);
  for (size_t i = 0; i < scene->mNumMaterials; i++) {
    aiMaterial *material = scene->mMaterials[i];
    textureList[i] = "";
    if (material->GetTextureCount(aiTextureType_DIFFUSE)) {
      aiString path; 
      if (material->GetTexture(aiTextureType_DIFFUSE, 0, &path) == AI_SUCCESS) {
        int idx = std::string(path.data).rfind('\\');
        std::string fileName = std::string(path.data).substr(idx + 1);
        textureList[i] = fileName;
      }
    }
  }
  return textureList;
}

std::vector<Mesh> MeshModel::LoadNode(VmaAllocator allocator,
                                      VkDevice newDevice, VkQueue transferQueue,
                                      VkCommandPool transferCommandPool,
                                      aiNode *node, const aiScene *scene,
                                      const std::vector<Material> &materials) {
  std::vector<Mesh> meshList;
  for (size_t i = 0; i < node->mNumMeshes; i++) {
    aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
    std::vector<Mesh> newMeshes =
        LoadMesh(allocator, newDevice, transferQueue,
                 transferCommandPool, mesh, scene, materials);
    meshList.insert(meshList.end(),
                    std::make_move_iterator(newMeshes.begin()),
                    std::make_move_iterator(newMeshes.end()));
  }

  for (size_t i = 0; i < node->mNumChildren; i++) {
    std::vector<Mesh> newList =
        LoadNode(allocator, newDevice, transferQueue,
                 transferCommandPool, node->mChildren[i], scene, materials);
    meshList.insert(meshList.end(),
                    std::make_move_iterator(newList.begin()),
                    std::make_move_iterator(newList.end()));
  }
  return meshList;
}

std::vector<Mesh> MeshModel::LoadMesh(VmaAllocator allocator,
                                      VkDevice newDevice, VkQueue transferQueue,
                                      VkCommandPool transferCommandPool,
                                      aiMesh *mesh, const aiScene *scene,
                                      const std::vector<Material> &materials) {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  vertices.resize(mesh->mNumVertices);
  
  for (size_t i = 0; i < mesh->mNumVertices; i++) {
    vertices[i].pos = {mesh->mVertices[i].x, mesh->mVertices[i].y,
                       mesh->mVertices[i].z};
    if (mesh->mTextureCoords[0]) {
      vertices[i].tex = {mesh->mTextureCoords[0][i].x,
                         mesh->mTextureCoords[0][i].y};
    } else {
      vertices[i].tex = {0.0f, 0.0f};
    }
    vertices[i].col = {1.0f, 1.0f, 1.0f};
  }
  
  for (size_t i = 0; i < mesh->mNumFaces; i++) {
    aiFace face = mesh->mFaces[i];
    for (size_t j = 0; j < face.mNumIndices; j++) {
      indices.push_back(face.mIndices[j]);
    }
  }
  
  Mesh newMesh =
      Mesh(allocator, newDevice, transferQueue, transferCommandPool,
           &vertices, &indices, materials[mesh->mMaterialIndex]);
  std::vector<Mesh> meshList;
  meshList.emplace_back(std::move(newMesh));
  return meshList;
}

MeshModel::~MeshModel() {}
