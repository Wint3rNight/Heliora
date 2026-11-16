#include "Skinning.h"
#include "TestRunner.h"

#include <glm/gtc/matrix_transform.hpp>

namespace {

constexpr float kHalfPi = 1.5707963f;

Vertex vertexAt(glm::vec3 p) {
  Vertex v{};
  v.pos = p;
  v.col = glm::vec3(0.5f);
  v.tex = glm::vec2(0.25f, 0.75f);
  v.normal = glm::vec3(1, 0, 0);
  v.tangent = glm::vec3(0, 1, 0);
  v.bitangent = glm::vec3(0, 0, 1);
  return v;
}

VertexSkin onBone(uint32_t bone) {
  VertexSkin s;
  s.bones = glm::uvec4(bone, 0, 0, 0);
  s.weights = glm::vec4(1, 0, 0, 0);
  return s;
}

} // namespace

TEST(skinning_in_the_bind_pose_changes_nothing) {
  const std::vector<glm::mat4> identity(2, glm::mat4(1.0f));
  const Vertex v = vertexAt(glm::vec3(1, 2, 3));
  const Vertex out = skinVertex(v, onBone(1), identity);
  CHECK_NEAR(glm::distance(out.pos, v.pos), 0.0, 1e-6);
  CHECK_NEAR(glm::distance(out.normal, v.normal), 0.0, 1e-6);
  CHECK(out.col == v.col);
  CHECK(out.tex == v.tex);
}

TEST(skinning_moves_and_turns_with_the_bone) {
  // The bone turns a quarter about z and moves 1 m up.
  const std::vector<glm::mat4> skin = {
      glm::translate(glm::mat4(1.0f), glm::vec3(0, 1, 0)) *
      glm::rotate(glm::mat4(1.0f), kHalfPi, glm::vec3(0, 0, 1))};
  const Vertex out = skinVertex(vertexAt(glm::vec3(1, 0, 0)), onBone(0), skin);
  CHECK_NEAR(glm::distance(out.pos, glm::vec3(0, 2, 0)), 0.0, 1e-5);
  CHECK_NEAR(glm::distance(out.normal, glm::vec3(0, 1, 0)), 0.0, 1e-5);
  CHECK_NEAR(glm::distance(out.tangent, glm::vec3(-1, 0, 0)), 0.0, 1e-5);
  CHECK_NEAR(glm::distance(out.bitangent, glm::vec3(0, 0, 1)), 0.0, 1e-5);
}

TEST(skinning_blends_bones_by_weight) {
  // Half on a bone that stays, half on one that moves 2 m along x.
  const std::vector<glm::mat4> skin = {
      glm::mat4(1.0f), glm::translate(glm::mat4(1.0f), glm::vec3(2, 0, 0))};
  VertexSkin halves;
  halves.bones = glm::uvec4(0, 1, 0, 0);
  halves.weights = glm::vec4(0.5f, 0.5f, 0, 0);
  const Vertex out = skinVertex(vertexAt(glm::vec3(0, 1, 0)), halves, skin);
  CHECK_NEAR(glm::distance(out.pos, glm::vec3(1, 1, 0)), 0.0, 1e-5);
}

TEST(skinned_normals_stay_unit_length_under_scale) {
  // A character scaled down to the capsule's size.
  const std::vector<glm::mat4> skin = {glm::scale(glm::mat4(1.0f), glm::vec3(0.01f))};
  const Vertex out = skinVertex(vertexAt(glm::vec3(100, 0, 0)), onBone(0), skin);
  CHECK_NEAR(glm::distance(out.pos, glm::vec3(1, 0, 0)), 0.0, 1e-5);
  CHECK_NEAR(glm::length(out.normal), 1.0, 1e-5);
  CHECK_NEAR(glm::length(out.tangent), 1.0, 1e-5);
}
