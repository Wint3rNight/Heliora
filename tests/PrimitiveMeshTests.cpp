#include "PrimitiveMesh.h"
#include "TestRunner.h"

#include <glm/glm.hpp>

namespace {

const glm::vec3 kColor(0.2f, 0.4f, 0.6f);

// What every generated mesh must get right. Counts bad vertices and
// triangles instead of checking each one, so a broken mesh prints one line.
void checkMesh(const MeshData &mesh) {
  CHECK(!mesh.vertices.empty());
  CHECK(mesh.indices.size() % 3 == 0);

  int badIndices = 0;
  for (uint32_t i : mesh.indices)
    if (i >= mesh.vertices.size())
      ++badIndices;
  CHECK(badIndices == 0);
  if (badIndices > 0)
    return;

  int badFrames = 0;
  int badColors = 0;
  for (const Vertex &v : mesh.vertices) {
    const bool unit = std::fabs(glm::length(v.normal) - 1.0f) < 1e-4f &&
                      std::fabs(glm::length(v.tangent) - 1.0f) < 1e-4f &&
                      std::fabs(glm::length(v.bitangent) - 1.0f) < 1e-4f;
    const bool orthogonal = std::fabs(glm::dot(v.normal, v.tangent)) < 1e-4f;
    if (!unit || !orthogonal)
      ++badFrames;
    if (v.col != kColor)
      ++badColors;
  }
  CHECK(badFrames == 0);
  CHECK(badColors == 0);

  // Counter-clockwise seen from outside, like glTF: the triangle's own normal
  // points the same way as its vertex normals. Zero-area triangles fail too.
  int badWinding = 0;
  for (size_t i = 0; i < mesh.indices.size(); i += 3) {
    const Vertex &a = mesh.vertices[mesh.indices[i]];
    const Vertex &b = mesh.vertices[mesh.indices[i + 1]];
    const Vertex &c = mesh.vertices[mesh.indices[i + 2]];
    const glm::vec3 faceNormal = glm::cross(b.pos - a.pos, c.pos - a.pos);
    const glm::vec3 vertexNormal = a.normal + b.normal + c.normal;
    if (glm::length(faceNormal) < 1e-8f ||
        glm::dot(faceNormal, vertexNormal) <= 0.0f)
      ++badWinding;
  }
  CHECK(badWinding == 0);
}

void bounds(const MeshData &mesh, glm::vec3 &lo, glm::vec3 &hi) {
  lo = hi = glm::vec3(0.0f);
  if (mesh.vertices.empty())
    return; // checkMesh already reported it
  lo = hi = mesh.vertices.front().pos;
  for (const Vertex &v : mesh.vertices) {
    lo = glm::min(lo, v.pos);
    hi = glm::max(hi, v.pos);
  }
}

} // namespace

TEST(box_has_24_vertices_and_its_size) {
  const MeshData box = makeBox(glm::vec3(2.0f, 1.0f, 3.0f), kColor);
  checkMesh(box);
  CHECK(box.vertices.size() == 24); // 4 per face, so each face has its own normal
  CHECK(box.indices.size() == 36);

  glm::vec3 lo, hi;
  bounds(box, lo, hi);
  CHECK(lo == glm::vec3(-1.0f, -0.5f, -1.5f));
  CHECK(hi == glm::vec3(1.0f, 0.5f, 1.5f));
}

TEST(sphere_vertices_sit_on_the_radius) {
  const MeshData sphere = makeSphere(0.7f, kColor);
  checkMesh(sphere);

  int off = 0;
  for (const Vertex &v : sphere.vertices)
    if (std::fabs(glm::length(v.pos) - 0.7f) > 1e-5f ||
        glm::length(v.normal - v.pos / 0.7f) > 1e-4f)
      ++off;
  CHECK(off == 0);
}

TEST(capsule_has_the_requested_height_and_radius) {
  const float radius = 0.5f, height = 2.0f;
  const MeshData capsule = makeCapsule(radius, height, kColor);
  checkMesh(capsule);

  glm::vec3 lo, hi;
  bounds(capsule, lo, hi);
  CHECK_NEAR(lo.y, -height / 2, 1e-5);
  CHECK_NEAR(hi.y, height / 2, 1e-5);

  // Every vertex is exactly one radius from the capsule's inner segment.
  const float halfSegment = height / 2 - radius;
  int off = 0;
  for (const Vertex &v : capsule.vertices) {
    const glm::vec3 axis(0.0f, glm::clamp(v.pos.y, -halfSegment, halfSegment),
                         0.0f);
    if (std::fabs(glm::length(v.pos - axis) - radius) > 1e-5f)
      ++off;
  }
  CHECK(off == 0);
}

TEST(plane_faces_up) {
  const MeshData plane = makePlane(4.0f, 6.0f, kColor);
  checkMesh(plane);
  CHECK(plane.vertices.size() == 4);
  CHECK(plane.indices.size() == 6);

  glm::vec3 lo, hi;
  bounds(plane, lo, hi);
  CHECK(lo == glm::vec3(-2.0f, 0.0f, -3.0f));
  CHECK(hi == glm::vec3(2.0f, 0.0f, 3.0f));
  for (const Vertex &v : plane.vertices)
    CHECK(v.normal == glm::vec3(0.0f, 1.0f, 0.0f));
}
