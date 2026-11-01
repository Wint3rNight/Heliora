#include "PrimitiveMesh.h"

#include <glm/gtc/constants.hpp>

#include <cmath>

namespace {

// Sphere and capsule detail. 24 x 12 is about 550 triangles for a sphere:
// round enough up close, and the meshoptimizer LODs take over further away.
constexpr int kSegments = 24; // around Y
constexpr int kRings = 12;    // pole to pole, even so a capsule splits in half

// One flat quad: corners at center +- u*halfU +- v*halfV. Counter-clockwise
// seen from the normal side, because cross(u, v) = normal for every caller.
void addQuad(MeshData &mesh, glm::vec3 center, glm::vec3 normal, glm::vec3 u,
             glm::vec3 v, float halfU, float halfV, glm::vec3 color) {
  const uint32_t first = static_cast<uint32_t>(mesh.vertices.size());
  // v runs up the face but texture v runs down (images load top row first),
  // so the bitangent, which follows texture v, is -v.
  const glm::vec2 uvs[4] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
  const float su[4] = {-1, 1, 1, -1};
  const float sv[4] = {-1, -1, 1, 1};
  for (int i = 0; i < 4; ++i)
    mesh.vertices.push_back({center + u * (su[i] * halfU) + v * (sv[i] * halfV),
                             color, uvs[i], normal, u, -v});
  for (uint32_t i : {0u, 1u, 2u, 0u, 2u, 3u})
    mesh.indices.push_back(first + i);
}

// A grid of rows around the Y axis: row r sits at polar angle theta[r] (0 at
// the top pole) and is shifted by yOffset[r]. A sphere is one offset for every
// row; a capsule repeats the equator row with two offsets, which makes the
// straight band. The seam column is doubled so texture u can wrap 0 to 1.
MeshData makeRows(float radius, const std::vector<float> &theta,
                  const std::vector<float> &yOffset, glm::vec3 color) {
  MeshData mesh;
  const int rows = static_cast<int>(theta.size());
  for (int r = 0; r < rows; ++r) {
    const float st = std::sin(theta[r]), ct = std::cos(theta[r]);
    for (int s = 0; s <= kSegments; ++s) {
      const float phi = glm::two_pi<float>() * s / kSegments;
      const float sp = std::sin(phi), cp = std::cos(phi);
      const glm::vec3 normal(st * cp, ct, -st * sp);
      // Tangent follows texture u (around), bitangent follows texture v
      // (down from the top). Both come from the angles, not from neighbouring
      // vertices, so they stay valid at the poles where rows shrink to a point.
      const glm::vec3 tangent(-sp, 0.0f, -cp);
      const glm::vec3 bitangent(ct * cp, -st, -ct * sp);
      mesh.vertices.push_back({normal * radius + glm::vec3(0, yOffset[r], 0),
                               color,
                               {float(s) / kSegments, float(r) / (rows - 1)},
                               normal, tangent, bitangent});
    }
  }
  const uint32_t stride = kSegments + 1;
  for (int r = 0; r + 1 < rows; ++r) {
    for (int s = 0; s < kSegments; ++s) {
      const uint32_t a = r * stride + s, b = a + 1;
      const uint32_t c = a + stride, d = c + 1;
      // Rows at a pole have zero width, so only one triangle per quad there.
      if (theta[r] > 0.0f)
        for (uint32_t i : {a, c, b})
          mesh.indices.push_back(i);
      if (theta[r + 1] < glm::pi<float>())
        for (uint32_t i : {b, c, d})
          mesh.indices.push_back(i);
    }
  }
  return mesh;
}

} // namespace

MeshData makeBox(glm::vec3 size, glm::vec3 color) {
  const glm::vec3 h = size * 0.5f;
  MeshData mesh;
  mesh.vertices.reserve(24);
  mesh.indices.reserve(36);
  // normal, u; v = cross(normal, u) keeps every face counter-clockwise.
  const glm::vec3 faces[6][2] = {
      {{1, 0, 0}, {0, 0, -1}}, {{-1, 0, 0}, {0, 0, 1}},
      {{0, 1, 0}, {1, 0, 0}},  {{0, -1, 0}, {1, 0, 0}},
      {{0, 0, 1}, {1, 0, 0}},  {{0, 0, -1}, {-1, 0, 0}},
  };
  for (const auto &face : faces) {
    const glm::vec3 n = face[0], u = face[1], v = glm::cross(n, u);
    addQuad(mesh, n * glm::dot(glm::abs(n), h), n, u, v,
            glm::dot(glm::abs(u), h), glm::dot(glm::abs(v), h), color);
  }
  return mesh;
}

MeshData makeSphere(float radius, glm::vec3 color) {
  std::vector<float> theta, yOffset;
  for (int r = 0; r <= kRings; ++r) {
    theta.push_back(glm::pi<float>() * r / kRings);
    yOffset.push_back(0.0f);
  }
  return makeRows(radius, theta, yOffset, color);
}

MeshData makeCapsule(float radius, float height, glm::vec3 color) {
  // The top half-sphere sits halfSegment up, the bottom one halfSegment down,
  // and the equator row appears once for each.
  const float halfSegment = height * 0.5f - radius;
  std::vector<float> theta, yOffset;
  for (int r = 0; r <= kRings / 2; ++r) {
    theta.push_back(glm::pi<float>() * r / kRings);
    yOffset.push_back(halfSegment);
  }
  for (int r = kRings / 2; r <= kRings; ++r) {
    theta.push_back(glm::pi<float>() * r / kRings);
    yOffset.push_back(-halfSegment);
  }
  return makeRows(radius, theta, yOffset, color);
}

MeshData makePlane(float sizeX, float sizeZ, glm::vec3 color) {
  MeshData mesh;
  addQuad(mesh, glm::vec3(0.0f), glm::vec3(0, 1, 0), glm::vec3(1, 0, 0),
          glm::vec3(0, 0, -1), sizeX * 0.5f, sizeZ * 0.5f, color);
  return mesh;
}
