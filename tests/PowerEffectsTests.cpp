#include "PowerEffects.h"
#include "TestRunner.h"

#include <cmath>

namespace {

const glm::vec4 kCore(1.0f, 0.9f, 0.6f, 1.0f);
const glm::vec4 kEdge(0.4f, 0.7f, 1.0f, 1.0f);

float offLine(glm::vec3 p, glm::vec3 a, glm::vec3 b) {
  const glm::vec3 u = glm::normalize(b - a);
  const glm::vec3 d = p - a;
  return glm::length(d - u * glm::dot(d, u));
}

} // namespace

TEST(a_facing_strip_is_two_triangles_a_segment_turned_to_the_eye) {
  std::vector<EffectVertex> out;
  // Along x, seen from +z: its width goes along y.
  addFacingStrip(out, {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}, {1, 0, 5}, 0.2f, 0.1f, kCore, kCore, kEdge);
  CHECK(out.size() == 12);
  float widest = 0.0f, narrowest = 1.0f, along = 0.0f;
  for (const EffectVertex &v : out) {
    CHECK(std::fabs(v.position.z) < 1e-5f);
    CHECK(v.uv.y == 0.0f || v.uv.y == 1.0f);
    if (v.position.x < 1e-4f)
      widest = std::max(widest, std::fabs(v.position.y));
    if (v.position.x > 2.0f - 1e-4f)
      narrowest = std::min(narrowest, std::fabs(v.position.y));
    along = std::max(along, v.uv.x);
  }
  CHECK_NEAR(widest, 0.1, 1e-5);    // half of width0
  CHECK_NEAR(narrowest, 0.05, 1e-5); // half of width1
  CHECK_NEAR(along, 2.0, 1e-5);     // metres along it
  std::vector<EffectVertex> none;
  addFacingStrip(none, {{0, 0, 0}}, {0, 0, 5}, 0.2f, 0.2f, kCore, kCore, kEdge);
  CHECK(none.empty());
}

TEST(a_facing_strip_seen_end_on_does_not_twist) {
  // A trail flying away from the eye, wobbling across the line of sight: its
  // sides stay on the same side from one point to the next.
  std::vector<EffectVertex> out;
  addFacingStrip(out, {{0, 0, 0}, {0.01f, 0, -1}, {-0.01f, 0, -2}, {0.01f, 0, -3}, {-0.01f, 0, -4}},
                 {0, 0, 10}, 0.2f, 0.2f, kCore, kCore, kEdge);
  CHECK(out.size() == 24);
  // Each segment's first vertex is its left (uv.y 0) at its start point.
  for (size_t seg = 0; seg + 1 < out.size() / 6; ++seg) {
    const glm::vec3 a = out[seg * 6].position - out[seg * 6 + 1].position;       // left - right here
    const glm::vec3 b = out[(seg + 1) * 6].position - out[(seg + 1) * 6 + 1].position;
    CHECK(glm::dot(a, b) > 0.0f);
  }
}

TEST(a_ribbon_strand_runs_from_the_hand_to_what_is_held_and_bows) {
  const glm::vec3 hand(0, 1.4f, 0), held(0, 2, -4);
  const std::vector<glm::vec3> a = ribbonStrand(hand, held, 0, 0.0f, 24);
  CHECK(a.size() == 25);
  if (a.size() != 25)
    return;
  CHECK(glm::distance(a.front(), hand) < 1e-5f);
  CHECK(glm::distance(a.back(), held) < 1e-5f);
  const float length = glm::distance(hand, held);
  CHECK(offLine(a[12], hand, held) > 0.05f * length); // bowed
  const std::vector<glm::vec3> b = ribbonStrand(hand, held, 1, 0.0f, 24);
  CHECK(b.size() == 25 && glm::distance(a[12], b[12]) > 0.05f); // each strand its own way
  const std::vector<glm::vec3> later = ribbonStrand(hand, held, 0, 0.5f, 24);
  CHECK(later.size() == 25 && glm::distance(a[12], later[12]) > 0.01f); // swaying
  CHECK(later.size() == 25 && glm::distance(later.front(), hand) < 1e-5f);
}

TEST(a_ground_ring_lies_flat_round_its_centre) {
  std::vector<EffectVertex> out;
  const glm::vec3 centre(2, 0.05f, -3);
  addGroundRing(out, centre, 5.0f, 0.4f, kCore, kEdge, 32);
  CHECK(out.size() == 32 * 6);
  for (const EffectVertex &v : out) {
    CHECK(std::fabs(v.position.y - centre.y) < 1e-5f);
    const float r = glm::length(glm::vec2(v.position.x - centre.x, v.position.z - centre.z));
    CHECK(std::fabs(r - (v.uv.y < 0.5f ? 4.8f : 5.2f)) < 1e-3f); // inner side 0, outer 1
    // No seam where it closes: at angle 0 the flow coordinate is the same
    // from both sides.
    if (std::fabs(v.position.z - centre.z) < 1e-4f && v.position.x > centre.x)
      CHECK(std::fabs(v.uv.x) < 1e-3f);
  }
}

TEST(a_cone_opens_from_its_apex_to_its_rim_no_further_than_its_reach) {
  // The burst reaches 8 m in every direction it covers: the rim 8 m from the
  // apex along the side, not 8 m along the axis.
  for (const float degrees : {35.0f, 90.0f}) {
    std::vector<EffectVertex> out;
    const glm::vec3 apex(0, 1, 0), axis(0, 0, -1);
    const float half = glm::radians(degrees);
    addCone(out, apex, axis, 8.0f, half, kCore, kEdge, 24);
    CHECK(out.size() == 24 * 6);
    for (const EffectVertex &v : out) {
      const glm::vec3 d = v.position - apex;
      const float alongAxis = glm::dot(d, axis);
      const float across = glm::length(d - axis * alongAxis);
      if (v.uv.y < 0.5f) {
        CHECK(glm::length(d) < 1e-4f);
      } else {
        CHECK_NEAR(glm::length(d), 8.0, 1e-3);
        CHECK_NEAR(alongAxis, 8.0 * std::cos(half), 1e-3);
        CHECK_NEAR(across, 8.0 * std::sin(half), 1e-3);
      }
    }
  }
}
