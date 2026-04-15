#include "Model.h"
#include <stdexcept>
#include <cstring>

// =============================================================================
// Mesh Implementation
// =============================================================================

Mesh::Mesh()
    : physicalDevice(VK_NULL_HANDLE), device(VK_NULL_HANDLE), vertexCount(0),
      vertexBuffer(VK_NULL_HANDLE), vertexBufferMemory(VK_NULL_HANDLE),
      indexCount(0), indexBuffer(VK_NULL_HANDLE),
      indexBufferMemory(VK_NULL_HANDLE) {}

Mesh::Mesh(VkPhysicalDevice newPhysicalDevice, VkDevice newDevice,
           VkQueue transferQueue, VkCommandPool transferCommandPool,
           std::vector<Vertex> *vertices, std::vector<uint32_t> *indices,
           Material newMaterial) {
  vertexCount = vertices->size();
  indexCount = indices->size();
  physicalDevice = newPhysicalDevice;
  device = newDevice;
  material = newMaterial;
  createVertexBuffer(transferQueue, transferCommandPool, vertices);
  createIndexBuffer(transferQueue, transferCommandPool, indices);
}

const Material &Mesh::getMaterial() const { return material; }
int Mesh::getVertexCount() const { return vertexCount; }
VkBuffer Mesh::getVertexBuffer() const { return vertexBuffer; }
int Mesh::getIndexCount() const { return indexCount; }
VkBuffer Mesh::getIndexBuffer() const { return indexBuffer; }

void Mesh::destroyBuffers() {
  vkDestroyBuffer(device, vertexBuffer, nullptr);
  vkFreeMemory(device, vertexBufferMemory, nullptr);
  vkDestroyBuffer(device, indexBuffer, nullptr);
  vkFreeMemory(device, indexBufferMemory, nullptr);
}

Mesh::~Mesh() {}

void Mesh::createVertexBuffer(VkQueue transferQueue,
                              VkCommandPool transferCommandPool,
                              std::vector<Vertex> *vertices) {
  VkDeviceSize bufferSize = sizeof(Vertex) * vertices->size();

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  createBuffer(physicalDevice, device, bufferSize,
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               &stagingBuffer, &stagingBufferMemory);

  void *data;
  vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
  memcpy(data, vertices->data(), (size_t)bufferSize);
  vkUnmapMemory(device, stagingBufferMemory);

  createBuffer(
      physicalDevice, device, bufferSize,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &vertexBuffer, &vertexBufferMemory);

  copyBuffer(device, transferQueue, transferCommandPool, stagingBuffer,
             vertexBuffer, bufferSize);

  vkDestroyBuffer(device, stagingBuffer, nullptr);
  vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void Mesh::createIndexBuffer(VkQueue transferQueue,
                             VkCommandPool transferCommandPool,
                             std::vector<uint32_t> *indices) {
  VkDeviceSize bufferSize = sizeof(uint32_t) * indices->size();

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  createBuffer(physicalDevice, device, bufferSize,
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               &stagingBuffer, &stagingBufferMemory);

  void *data;
  vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
  memcpy(data, indices->data(), (size_t)bufferSize);
  vkUnmapMemory(device, stagingBufferMemory);

  createBuffer(
      physicalDevice, device, bufferSize,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &indexBuffer, &indexBufferMemory);

  copyBuffer(device, transferQueue, transferCommandPool, stagingBuffer,
             indexBuffer, bufferSize);

  vkDestroyBuffer(device, stagingBuffer, nullptr);
  vkFreeMemory(device, stagingBufferMemory, nullptr);
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

std::vector<Mesh> MeshModel::LoadNode(VkPhysicalDevice newPhysicalDevice,
                                      VkDevice newDevice, VkQueue transferQueue,
                                      VkCommandPool transferCommandPool,
                                      aiNode *node, const aiScene *scene,
                                      const std::vector<Material> &materials) {
  std::vector<Mesh> meshList;
  for (size_t i = 0; i < node->mNumMeshes; i++) {
    aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
    std::vector<Mesh> newMeshes =
        LoadMesh(newPhysicalDevice, newDevice, transferQueue,
                 transferCommandPool, mesh, scene, materials);
    meshList.insert(meshList.end(), newMeshes.begin(), newMeshes.end());
  }

  for (size_t i = 0; i < node->mNumChildren; i++) {
    std::vector<Mesh> newList =
        LoadNode(newPhysicalDevice, newDevice, transferQueue,
                 transferCommandPool, node->mChildren[i], scene, materials);
    meshList.insert(meshList.end(), newList.begin(), newList.end());
  }
  return meshList;
}

std::vector<Mesh> MeshModel::LoadMesh(VkPhysicalDevice newPhysicalDevice,
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
      Mesh(newPhysicalDevice, newDevice, transferQueue, transferCommandPool,
           &vertices, &indices, materials[mesh->mMaterialIndex]);
  return {newMesh};
}

MeshModel::~MeshModel() {}
