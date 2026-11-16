#pragma once

#include "Vertex.h"

#include <glm/glm.hpp>

#include <vector>

// Bending a mesh with its skeleton (linear blend skinning). The GPU does it
// every frame in Shaders/skinning.comp; this is the same maths on the CPU,
// tested, and used for bounds.

// The bones that move a vertex and how much: up to four, by skeleton bone,
// the weights adding up to 1. 32 bytes, the layout the shader reads.
struct VertexSkin {
  glm::uvec4 bones{0u};
  glm::vec4 weights{0.0f};
};

// The vertex moved by its bones: the skinning matrices mixed by weight,
// then applied to the position, and to the normal, tangent and bitangent
// (renormalised). Colour and texture coordinates are copied.
Vertex skinVertex(const Vertex &vertex, const VertexSkin &skin,
                  const std::vector<glm::mat4> &skinning);
