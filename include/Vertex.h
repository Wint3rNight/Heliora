#pragma once

#include <glm/glm.hpp>

// The vertex every mesh uses. Its own header, with no Vulkan, so code like
// the shape generator can build meshes inside heliora_core.
struct Vertex {
  glm::vec3 pos; // position attribute for the vertex
  glm::vec3 col; // color attribute for the vertex
  glm::vec2 tex; // texture coordinate attribute
  glm::vec3 normal;
  glm::vec3 tangent;
  glm::vec3 bitangent;
};
