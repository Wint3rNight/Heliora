#include "Model.h"
#include <assimp/GltfMaterial.h>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <glm/glm.hpp>
#include <iterator>
#include <meshoptimizer.h>
#include <stdexcept>

// =============================================================================
// Mesh Implementation
// =============================================================================

Mesh::Mesh()
    : allocator(VK_NULL_HANDLE), device(VK_NULL_HANDLE), vertexCount(0),
      indexCount(0) {}

Mesh::Mesh(VmaAllocator newAllocator, VkDevice newDevice, VkQueue transferQueue,
           VkCommandPool transferCommandPool, std::vector<Vertex> *vertices,
           std::vector<uint32_t> *indices, Material newMaterial) {
  vertexCount = vertices->size();
  indexCount = indices->size();
  cpuVertices = *vertices;
  cpuIndices = *indices;
  allocator = newAllocator;
  device = newDevice;
  material = newMaterial;

  if (!vertices->empty()) {
    glm::vec3 center(0.0f);
    for (const Vertex &v : *vertices)
      center += v.pos;
    center /= static_cast<float>(vertices->size());
    float radius = 0.0f;
    for (const Vertex &v : *vertices)
      radius = glm::max(radius, glm::length(v.pos - center));
    boundingCenter = center;
    boundingRadius = radius;
  }

  createVertexBuffer(transferQueue, transferCommandPool, vertices);
  createIndexBuffer(transferQueue, transferCommandPool, indices);
}

const Material &Mesh::getMaterial() const { return material; }
int Mesh::getVertexCount() const { return vertexCount; }
VkBuffer Mesh::getVertexBuffer() const { return vertexBuffer.get(); }

int Mesh::getIndexCount(int lod) const {
  if (lod <= 0 || extraLodIndexCounts.empty())
    return indexCount;
  int i = lod - 1;
  return (i < static_cast<int>(extraLodIndexCounts.size()))
             ? extraLodIndexCounts[i]
             : indexCount;
}

VkBuffer Mesh::getIndexBuffer(int lod) const {
  if (lod <= 0 || extraLodIndexBuffers.empty())
    return indexBuffer.get();
  int i = lod - 1;
  return (i < static_cast<int>(extraLodIndexBuffers.size()))
             ? extraLodIndexBuffers[i].get()
             : indexBuffer.get();
}

const std::vector<uint32_t> &Mesh::getCpuIndices(int lod) const {
  if (lod <= 0 || extraLodCpuIndices.empty())
    return cpuIndices;
  int i = lod - 1;
  return (i < static_cast<int>(extraLodCpuIndices.size()))
             ? extraLodCpuIndices[i]
             : cpuIndices;
}

int Mesh::getLodCount() const {
  return 1 + static_cast<int>(extraLodIndexBuffers.size());
}

void Mesh::addLod(VmaAllocator newAllocator, VkDevice newDevice,
                  VkQueue transferQueue, VkCommandPool transferCommandPool,
                  std::vector<uint32_t> *indices) {
  if (indices->empty())
    return;

  VkDeviceSize bufferSize = sizeof(uint32_t) * indices->size();
  AllocatedBuffer stagingBuffer;
  createBuffer(newAllocator, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VMA_MEMORY_USAGE_AUTO,
               VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
               &stagingBuffer);

  void *data;
  vmaMapMemory(newAllocator, stagingBuffer.getAllocation(), &data);
  memcpy(data, indices->data(), static_cast<size_t>(bufferSize));
  vmaUnmapMemory(newAllocator, stagingBuffer.getAllocation());

  AllocatedBuffer lodBuf;
  createBuffer(newAllocator, bufferSize,
               VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                   VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
               VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, &lodBuf);
  copyBuffer(newDevice, transferQueue, transferCommandPool, stagingBuffer.get(),
             lodBuf.get(), bufferSize);

  extraLodIndexBuffers.push_back(std::move(lodBuf));
  extraLodIndexCounts.push_back(static_cast<int>(indices->size()));
  extraLodCpuIndices.push_back(*indices);
}

void Mesh::destroyBuffers() {
  vertexBuffer.reset();
  indexBuffer.reset();
  for (auto &b : extraLodIndexBuffers)
    b.reset();
  extraLodIndexBuffers.clear();
  extraLodIndexCounts.clear();
  extraLodCpuIndices.clear();
}

Mesh::~Mesh() {}

void Mesh::createVertexBuffer(VkQueue transferQueue,
                              VkCommandPool transferCommandPool,
                              std::vector<Vertex> *vertices) {
  VkDeviceSize bufferSize = sizeof(Vertex) * vertices->size();

  AllocatedBuffer stagingBuffer;
  createBuffer(allocator, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VMA_MEMORY_USAGE_AUTO,
               VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
               &stagingBuffer);

  void *data;
  vmaMapMemory(allocator, stagingBuffer.getAllocation(), &data);
  memcpy(data, vertices->data(), (size_t)bufferSize);
  vmaUnmapMemory(allocator, stagingBuffer.getAllocation());

  createBuffer(allocator, bufferSize,
               VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
               VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, &vertexBuffer);

  copyBuffer(device, transferQueue, transferCommandPool, stagingBuffer.get(),
             vertexBuffer.get(), bufferSize);
}

void Mesh::createIndexBuffer(VkQueue transferQueue,
                             VkCommandPool transferCommandPool,
                             std::vector<uint32_t> *indices) {
  VkDeviceSize bufferSize = sizeof(uint32_t) * indices->size();

  AllocatedBuffer stagingBuffer;
  createBuffer(allocator, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VMA_MEMORY_USAGE_AUTO,
               VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
               &stagingBuffer);

  void *data;
  vmaMapMemory(allocator, stagingBuffer.getAllocation(), &data);
  memcpy(data, indices->data(), (size_t)bufferSize);
  vmaUnmapMemory(allocator, stagingBuffer.getAllocation());

  createBuffer(allocator, bufferSize,
               VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                   VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
               VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, &indexBuffer);

  copyBuffer(device, transferQueue, transferCommandPool, stagingBuffer.get(),
             indexBuffer.get(), bufferSize);
}

// =============================================================================
// MeshModel Implementation
// =============================================================================

MeshModel::MeshModel() {}

MeshModel::MeshModel(std::vector<Mesh> newMeshList) {
  meshList = std::move(newMeshList);

  if (!meshList.empty()) {
    glm::vec3 center(0.0f);
    for (const Mesh &m : meshList)
      center += m.boundingCenter;
    center /= static_cast<float>(meshList.size());
    float radius = 0.0f;
    for (const Mesh &m : meshList)
      radius = glm::max(radius, glm::length(m.boundingCenter - center) +
                                    m.boundingRadius);
    boundingCenter = center;
    boundingRadius = radius;
  }
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

namespace {
std::string filenameFromAssimpPath(const aiString &path) {
  std::string value(path.C_Str());
  size_t slash = value.find_last_of("\\/");
  if (slash != std::string::npos) {
    return value.substr(slash + 1);
  }
  return value;
}

std::string toLower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool looksLikeNormalMap(const std::string &filename) {
  std::string lower = toLower(filename);
  return lower.find("_nor") != std::string::npos ||
         lower.find("normal") != std::string::npos ||
         lower.find("_nrm") != std::string::npos;
}

// True if `name` contains any cloth/fabric keyword. Matches material names
// AND texture filenames so any of (named material, named texture) is
// enough to flag the material. Sponza glTF leaves materials unnamed and
// uses GUID texture filenames, so this fires for nothing on that scene —
// the shader-side chroma heuristic still catches its banners.
bool looksLikeCloth(const std::string &name) {
  static const char *kKeywords[] = {
      "fabric", "cloth", "curtain", "banner", "drape", "linen",
      "velvet", "silk",  "wool",    "carpet", "rug",   "tapestry",
  };
  std::string lower = toLower(name);
  for (const char *kw : kKeywords) {
    if (lower.find(kw) != std::string::npos)
      return true;
  }
  return false;
}

void replaceAll(std::string &value, const std::string &from,
                const std::string &to) {
  size_t pos = 0;
  while ((pos = value.find(from, pos)) != std::string::npos) {
    value.replace(pos, from.size(), to);
    pos += to.size();
  }
}

std::string colorMapForNormalNamedDiffuse(std::string filename) {
  replaceAll(filename, "_NOR", "_COL");
  replaceAll(filename, "_nor", "_col");
  replaceAll(filename, "_NRM", "_COL");
  replaceAll(filename, "_nrm", "_col");
  replaceAll(filename, "NormalGL", "Color");
  replaceAll(filename, "NormalDX", "Color");
  replaceAll(filename, "normalgl", "color");
  replaceAll(filename, "normaldx", "color");
  replaceAll(filename, "Normal", "Color");
  replaceAll(filename, "normal", "color");
  return filename;
}
} // namespace

std::vector<MaterialTextureNames>
MeshModel::LoadMaterials(const aiScene *scene) {
  std::vector<MaterialTextureNames> textureList(scene->mNumMaterials);
  for (size_t i = 0; i < scene->mNumMaterials; i++) {
    aiMaterial *mat = scene->mMaterials[i];
    aiString path;

    // Two-sided flag — glTF doubleSided lands here via Assimp.
    int twoSided = 0;
    if (mat->Get(AI_MATKEY_TWOSIDED, twoSided) == AI_SUCCESS && twoSided)
      textureList[i].doubleSided = true;

    aiString alphaMode;
    if (mat->Get(AI_MATKEY_GLTF_ALPHAMODE, alphaMode) == AI_SUCCESS) {
      const std::string mode = toLower(alphaMode.C_Str());
      if (mode == "mask") {
        textureList[i].alphaMasked = true;
        float cutoff = 0.5f;
        if (mat->Get(AI_MATKEY_GLTF_ALPHACUTOFF, cutoff) == AI_SUCCESS)
          textureList[i].alphaCutoff = std::clamp(cutoff, 0.0f, 1.0f);
      } else if (mode == "blend") {
        textureList[i].alphaBlended = true;
      }
    }

    // Cloth flag — check the material's own name first, since glTF v2
    // exporters often write descriptive names ("RedCurtain", "Banner.001")
    // even when Assimp doesn't surface them as the mesh material slot.
    aiString matName;
    if (mat->Get(AI_MATKEY_NAME, matName) == AI_SUCCESS &&
        looksLikeCloth(matName.C_Str())) {
      textureList[i].isCloth = true;
    }

    // Albedo
    if (mat->GetTextureCount(aiTextureType_DIFFUSE) &&
        mat->GetTexture(aiTextureType_DIFFUSE, 0, &path) == AI_SUCCESS) {
      textureList[i].albedo = filenameFromAssimpPath(path);
      if (looksLikeNormalMap(textureList[i].albedo))
        textureList[i].albedo =
            colorMapForNormalNamedDiffuse(textureList[i].albedo);
    } else if (mat->GetTextureCount(aiTextureType_BASE_COLOR) &&
               mat->GetTexture(aiTextureType_BASE_COLOR, 0, &path) ==
                   AI_SUCCESS) {
      textureList[i].albedo = filenameFromAssimpPath(path);
    }

    // Normal
    if (mat->GetTextureCount(aiTextureType_NORMALS) &&
        mat->GetTexture(aiTextureType_NORMALS, 0, &path) == AI_SUCCESS) {
      textureList[i].normal = filenameFromAssimpPath(path);
    } else if (mat->GetTextureCount(aiTextureType_HEIGHT) &&
               mat->GetTexture(aiTextureType_HEIGHT, 0, &path) == AI_SUCCESS) {
      textureList[i].normal = filenameFromAssimpPath(path);
    } else if (mat->GetTextureCount(aiTextureType_DISPLACEMENT) &&
               mat->GetTexture(aiTextureType_DISPLACEMENT, 0, &path) ==
                   AI_SUCCESS) {
      textureList[i].normal = filenameFromAssimpPath(path);
    }

    // Metallic
    if (mat->GetTextureCount(aiTextureType_METALNESS) &&
        mat->GetTexture(aiTextureType_METALNESS, 0, &path) == AI_SUCCESS) {
      textureList[i].metallic = filenameFromAssimpPath(path);
    } else if (mat->GetTextureCount(aiTextureType_UNKNOWN) &&
               mat->GetTexture(aiTextureType_UNKNOWN, 0, &path) == AI_SUCCESS) {
      // glTF ORM packed texture often lands here
      textureList[i].metallic = filenameFromAssimpPath(path);
    }

    // Roughness. glTF 2.0 packs metalness+roughness into one texture; Assimp
    // may surface it under DIFFUSE_ROUGHNESS, UNKNOWN, or only under METALNESS.
    // Fall through all three, then mirror the metallic path if nothing else
    // matched — the packed texture has both channels.
    if (mat->GetTextureCount(aiTextureType_DIFFUSE_ROUGHNESS) &&
        mat->GetTexture(aiTextureType_DIFFUSE_ROUGHNESS, 0, &path) ==
            AI_SUCCESS) {
      textureList[i].roughness = filenameFromAssimpPath(path);
    } else if (mat->GetTextureCount(aiTextureType_UNKNOWN) &&
               mat->GetTexture(aiTextureType_UNKNOWN, 0, &path) == AI_SUCCESS) {
      textureList[i].roughness = filenameFromAssimpPath(path);
    } else if (!textureList[i].metallic.empty()) {
      textureList[i].roughness = textureList[i].metallic;
    }

    // AO
    if (mat->GetTextureCount(aiTextureType_AMBIENT_OCCLUSION) &&
        mat->GetTexture(aiTextureType_AMBIENT_OCCLUSION, 0, &path) ==
            AI_SUCCESS) {
      textureList[i].ao = filenameFromAssimpPath(path);
    } else if (mat->GetTextureCount(aiTextureType_LIGHTMAP) &&
               mat->GetTexture(aiTextureType_LIGHTMAP, 0, &path) ==
                   AI_SUCCESS) {
      textureList[i].ao = filenameFromAssimpPath(path);
    }

    // Final cloth pass: any of the resolved texture filenames carrying a
    // cloth keyword is enough. Covers asset packs that name only their
    // textures (e.g. "red_fabric_BaseColor.png") but not the materials.
    if (!textureList[i].isCloth) {
      textureList[i].isCloth =
          looksLikeCloth(textureList[i].albedo) ||
          looksLikeCloth(textureList[i].normal) ||
          looksLikeCloth(textureList[i].roughness) ||
          looksLikeCloth(textureList[i].metallic);
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
        LoadMesh(allocator, newDevice, transferQueue, transferCommandPool, mesh,
                 scene, materials);
    meshList.insert(meshList.end(), std::make_move_iterator(newMeshes.begin()),
                    std::make_move_iterator(newMeshes.end()));
  }

  for (size_t i = 0; i < node->mNumChildren; i++) {
    std::vector<Mesh> newList =
        LoadNode(allocator, newDevice, transferQueue, transferCommandPool,
                 node->mChildren[i], scene, materials);
    meshList.insert(meshList.end(), std::make_move_iterator(newList.begin()),
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
    if (mesh->HasNormals()) {
      vertices[i].normal = {mesh->mNormals[i].x, mesh->mNormals[i].y,
                            mesh->mNormals[i].z};
    } else {
      vertices[i].normal = {0.0f, 1.0f, 0.0f};
    }
    if (mesh->HasTangentsAndBitangents()) {
      vertices[i].tangent = {mesh->mTangents[i].x, mesh->mTangents[i].y,
                             mesh->mTangents[i].z};
      vertices[i].bitangent = {mesh->mBitangents[i].x, mesh->mBitangents[i].y,
                               mesh->mBitangents[i].z};
    } else {
      vertices[i].tangent = {1.0f, 0.0f, 0.0f};
      vertices[i].bitangent = {0.0f, 0.0f, 1.0f};
    }
  }

  for (size_t i = 0; i < mesh->mNumFaces; i++) {
    aiFace face = mesh->mFaces[i];
    for (size_t j = 0; j < face.mNumIndices; j++) {
      indices.push_back(face.mIndices[j]);
    }
  }

  Mesh newMesh = Mesh(allocator, newDevice, transferQueue, transferCommandPool,
                      &vertices, &indices, materials[mesh->mMaterialIndex]);

  // Generate LOD1 (~50%) and LOD2 (~12%) using meshoptimizer
  if (indices.size() >= 12) {
    const float *positions = &vertices[0].pos.x;
    size_t stride = sizeof(Vertex);

    auto genLod = [&](size_t targetCount, float maxError) {
      std::vector<uint32_t> lodIdx(indices.size());
      size_t n = meshopt_simplify(lodIdx.data(), indices.data(), indices.size(),
                                  positions, vertices.size(), stride,
                                  targetCount, maxError);
      lodIdx.resize(n);
      return lodIdx;
    };

    auto lod1 = genLod(indices.size() / 2, 0.05f);
    newMesh.addLod(allocator, newDevice, transferQueue, transferCommandPool,
                   &lod1);

    auto lod2 = genLod(indices.size() / 8, 0.15f);
    newMesh.addLod(allocator, newDevice, transferQueue, transferCommandPool,
                   &lod2);
  }

  std::vector<Mesh> meshList;
  meshList.emplace_back(std::move(newMesh));
  return meshList;
}

MeshModel::~MeshModel() {}
