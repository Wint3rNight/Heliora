#pragma once
#include "Mesh.h"
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

  ~MeshModel();

private:
  std::vector<Mesh> mesheList;
  glm::mat4 model;
};
