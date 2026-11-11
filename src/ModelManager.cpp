#include "ModelManager.h"

#include "ScenePhysics.h"
#include "TextureManager.h"
#include "VulkanDevice.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <filesystem>
#include <limits>
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
                                        aiProcess_JoinIdenticalVertices |
                                        aiProcess_GenSmoothNormals |
                                        aiProcess_CalcTangentSpace);
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

  // Tell TextureManager where this model lives so it can find textures
  // that sit next to the model file (common for glTF)
  for (const std::string &path : candidatePaths) {
    if (std::filesystem::exists(path)) {
      textureManager.setModelDirectory(
          std::filesystem::path(path).parent_path().string());
      break;
    }
  }

  // get vector of all materials with 1:1 id placement
  std::vector<MaterialTextureNames> textureNames =
      MeshModel::LoadMaterials(scene);

  // conversion from material list id to Material
  std::vector<Material> materials(textureNames.size());

  for (size_t i = 0; i < textureNames.size(); i++) {
    materials[i] = textureManager.loadMaterial(
        textureNames[i].albedo, textureNames[i].normal,
        textureNames[i].metallic, textureNames[i].roughness, textureNames[i].ao,
        device, descriptorManager);
    materials[i].doubleSided = textureNames[i].doubleSided;
    materials[i].isCloth     = textureNames[i].isCloth;
    materials[i].alphaMasked = textureNames[i].alphaMasked;
    materials[i].alphaBlended = textureNames[i].alphaBlended;
    materials[i].alphaCutoff = textureNames[i].alphaCutoff;
  }

  std::vector<Mesh> modelMeshes = MeshModel::LoadNode(
      device.getAllocator(), device.getLogicalDevice(),
      device.getGraphicsQueue(), device.getGraphicsCommandPool(),
      scene->mRootNode, scene, materials, aiMatrix4x4());

  MeshModel meshModel = MeshModel(std::move(modelMeshes));
  models.push_back(std::move(meshModel));

  int id = models.size() - 1;
  modelMap[modelFile] = id;
  return id;
}

int ModelManager::createModel(const std::string &key, MeshData mesh,
                              bool withLods, const VulkanDevice &device,
                              TextureManager &textureManager,
                              DescriptorManager &descriptorManager) {
  if (int id = findModel(key); id >= 0)
    return id;
  // Empty names give the default material: white albedo (the shape's vertex
  // colour tints it), flat normal, roughness 1, metallic 0.
  const Material material = textureManager.loadMaterial(
      "", "", "", "", "", device, descriptorManager);
  Mesh newMesh(device.getAllocator(), device.getLogicalDevice(),
               device.getGraphicsQueue(), device.getGraphicsCommandPool(),
               &mesh.vertices, &mesh.indices, material);
  if (withLods)
    newMesh.generateLods(device.getGraphicsQueue(),
                         device.getGraphicsCommandPool());
  std::vector<Mesh> meshes;
  meshes.push_back(std::move(newMesh));
  models.emplace_back(std::move(meshes));

  const int id = static_cast<int>(models.size()) - 1;
  modelMap[key] = id;
  return id;
}

std::shared_ptr<const PhysicsMeshParts>
ModelManager::collisionParts(int modelId) {
  const MeshModel *model = getModel(modelId);
  if (!model)
    return nullptr;
  if (collisionCache.size() < models.size())
    collisionCache.resize(models.size());
  std::shared_ptr<const PhysicsMeshParts> &cached = collisionCache[modelId];
  if (cached)
    return cached;
  // From full detail (LOD 0), in model space: loading already baked the
  // node transforms into the vertices. Cut down to 2,000 triangles a part:
  // the 134,000 of the Poly Haven boulder made each physics step take 5 to
  // 6 ms while a rock settled on it (2026-11-04).
  constexpr size_t kMaxCollisionTriangles = 2000;
  auto parts = std::make_shared<PhysicsMeshParts>();
  for (size_t i = 0; i < model->getMeshCount(); ++i) {
    const Mesh *mesh = model->getMesh(i);
    PhysicsMeshPart part;
    part.positions.reserve(mesh->getCpuVertices().size());
    for (const Vertex &vertex : mesh->getCpuVertices())
      part.positions.push_back(vertex.pos);
    part.indices = mesh->getCpuIndices(0);
    parts->parts.push_back(
        simplifiedForCollision(part, kMaxCollisionTriangles));
  }
  cached = parts;
  return cached;
}

std::optional<ModelManager::Bounds> ModelManager::localBounds(int modelId) {
  const MeshModel *model = getModel(modelId);
  if (!model)
    return std::nullopt;
  if (boundsCache.size() < models.size())
    boundsCache.resize(models.size());
  std::optional<Bounds> &cached = boundsCache[modelId];
  if (cached)
    return cached;
  Bounds bounds{glm::vec3(std::numeric_limits<float>::max()),
                glm::vec3(std::numeric_limits<float>::lowest())};
  for (size_t i = 0; i < model->getMeshCount(); ++i)
    for (const Vertex &vertex : model->getMesh(i)->getCpuVertices()) {
      bounds.min = glm::min(bounds.min, vertex.pos);
      bounds.max = glm::max(bounds.max, vertex.pos);
    }
  if (bounds.min.x > bounds.max.x)
    return std::nullopt; // no vertices
  cached = bounds;
  return cached;
}

int ModelManager::findModel(const std::string &key) const {
  auto it = modelMap.find(key);
  return it == modelMap.end() ? -1 : it->second;
}

void ModelManager::cleanup() {
  for (auto &model : models) {
    model.destroyMeshModel();
  }
  models.clear();
  modelMap.clear();
  collisionCache.clear();
  boundsCache.clear();
}

MeshModel *ModelManager::getModel(int modelId) {
  if (modelId < 0 || modelId >= models.size()) {
    return nullptr;
  }
  return &models[modelId];
}
