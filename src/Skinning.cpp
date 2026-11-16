#include "Skinning.h"

#include <cmath>

namespace {

glm::vec3 unit(const glm::vec3 &v, const glm::vec3 &fallback) {
  const float length = glm::length(v);
  return length > 1e-12f ? v / length : fallback;
}

} // namespace

Vertex skinVertex(const Vertex &vertex, const VertexSkin &skin,
                  const std::vector<glm::mat4> &skinning) {
  glm::mat4 m(0.0f);
  for (int k = 0; k < 4; ++k)
    if (skin.weights[k] != 0.0f)
      m += skinning[skin.bones[k]] * skin.weights[k];
  const glm::mat3 turn(m);
  Vertex out = vertex;
  out.pos = glm::vec3(m * glm::vec4(vertex.pos, 1.0f));
  out.normal = unit(turn * vertex.normal, vertex.normal);
  out.tangent = unit(turn * vertex.tangent, vertex.tangent);
  out.bitangent = unit(turn * vertex.bitangent, vertex.bitangent);
  return out;
}
