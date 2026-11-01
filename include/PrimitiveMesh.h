#pragma once

#include "Vertex.h"

#include <cstdint>
#include <vector>

// Meshes generated in code, so scenes can use boxes, spheres, capsules and
// planes without a model file. Centred on the origin, counter-clockwise
// winding seen from outside (same as glTF), every vertex tinted by color.

enum class PrimitiveShape : uint8_t { Box, Sphere, Capsule, Plane };

struct MeshData {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
};

// size is the full extent on each axis.
MeshData makeBox(glm::vec3 size, glm::vec3 color);
MeshData makeSphere(float radius, glm::vec3 color);
// Along Y. height is end to end, round caps included.
MeshData makeCapsule(float radius, float height, glm::vec3 color);
// Flat in XZ, facing +Y.
MeshData makePlane(float sizeX, float sizeZ, glm::vec3 color);
