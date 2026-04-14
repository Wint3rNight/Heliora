#include "MeshModel.h"
#include "Utilities.h"
#include <cstddef>
#include <stdexcept>
#include <vector>

MeshModel::MeshModel() : model(glm::mat4(1.0f)) {}

MeshModel::MeshModel(std::vector<Mesh> newMeshList) {
  meshList = std::move(newMeshList);
  model = glm::mat4(1.0f);
}

size_t MeshModel::getMeshCount() const { return meshList.size(); }

const Mesh *MeshModel::getMesh(size_t index) const {
  if (index >= meshList.size()) {
    throw std::runtime_error("Mesh index out of range");
  }
  return &meshList[index];
}

glm::mat4 MeshModel::getModel() const { return model; }

void MeshModel::setModel(const glm::mat4 &newModel) { model = newModel; }

void MeshModel::destroyMeshModel() {
  for (auto &mesh : meshList) {
    mesh.destroyBuffers();
  }
}

std::vector<std::string> MeshModel::LoadMaterials(const aiScene *scene) {
  // create 1:1 sized vector of textures
  std::vector<std::string> textureList(scene->mNumMaterials);
  for (size_t i = 0; i < scene->mNumMaterials; i++) {
    // get material
    aiMaterial *material = scene->mMaterials[i];
    // initialize the texture to empty string
    textureList[i] = "";
    // check if material has diffuse texture
    if (material->GetTextureCount(aiTextureType_DIFFUSE)) {
      aiString path; // get texture path
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
                                      const std::vector<int> &matToTex) {
  std::vector<Mesh> meshList;
  // go throug all meshes in node and load them
  for (size_t i = 0; i < node->mNumMeshes; i++) {
    aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
    std::vector<Mesh> newMeshes =
        LoadMesh(newPhysicalDevice, newDevice, transferQueue,
                 transferCommandPool, mesh, scene, matToTex);
    meshList.insert(meshList.end(), newMeshes.begin(), newMeshes.end());
  }

  // go through each node attached to this node and load them
  for (size_t i = 0; i < node->mNumChildren; i++) {
    std::vector<Mesh> newList =
        LoadNode(newPhysicalDevice, newDevice, transferQueue,
                 transferCommandPool, node->mChildren[i], scene, matToTex);
    meshList.insert(meshList.end(), newList.begin(), newList.end());
  }
  return meshList;
}

std::vector<Mesh> MeshModel::LoadMesh(VkPhysicalDevice newPhysicalDevice,
                                      VkDevice newDevice, VkQueue transferQueue,
                                      VkCommandPool transferCommandPool,
                                      aiMesh *mesh, const aiScene *scene,
                                      const std::vector<int> &matToTex) {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  // resize the vertices vector to the number of vertices in the mesh
  vertices.resize(mesh->mNumVertices);
  // go through each vertex in mesh and load data into vertex struct
  for (size_t i = 0; i < mesh->mNumVertices; i++) {
    vertices[i].pos = {mesh->mVertices[i].x, mesh->mVertices[i].y,
                       mesh->mVertices[i].z}; // position data
    if (mesh->mTextureCoords[0]) { // check if mesh has texture coordinates
      vertices[i].tex = {mesh->mTextureCoords[0][i].x,
                         mesh->mTextureCoords[0][i].y};
    } else {
      vertices[i].tex = {0.0f, 0.0f};
    }
    vertices[i].col = {1.0f, 1.0f, 1.0f}; // set color
  }
  // go through each face in mesh and load indices
  for (size_t i = 0; i < mesh->mNumFaces; i++) {
    aiFace face = mesh->mFaces[i];
    for (size_t j = 0; j < face.mNumIndices; j++) {
      indices.push_back(face.mIndices[j]);
    }
  }
  // create mesh with loaded data and return it in a vector
  Mesh newMesh =
      Mesh(newPhysicalDevice, newDevice, transferQueue, transferCommandPool,
           &vertices, &indices, matToTex[mesh->mMaterialIndex]);
  return {newMesh};
}

MeshModel::~MeshModel() {}
