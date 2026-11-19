#include "Ballistics.h"
#include "TestRunner.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr float kDt = 1.0f / 60.0f;
const glm::vec3 kGravity(0.0f, -9.81f, 0.0f);

// Where something launched from `from` at `velocity` is once it has gone as
// far across as `to`, stepped the way Jolt steps a body.
glm::vec3 landing(glm::vec3 from, glm::vec3 velocity, glm::vec3 to) {
  const float across = glm::length(glm::vec2(to.x - from.x, to.z - from.z));
  const auto gone = [&](glm::vec3 p) { return glm::length(glm::vec2(p.x - from.x, p.z - from.z)); };
  glm::vec3 p = from, before = from;
  for (int i = 0; i < 6000 && gone(p) < across; ++i) {
    before = p;
    velocity += kGravity * kDt;
    p += velocity * kDt;
  }
  const float u = (across - gone(before)) / std::max(gone(p) - gone(before), 1e-6f);
  return glm::mix(before, p, u);
}

} // namespace

TEST(a_throw_lands_where_it_was_aimed_on_the_low_arc) {
  const glm::vec3 from(0, 1, 0), to(20, 3, 5);
  const glm::vec3 v = throwVelocity(from, to, 30.0f, kGravity);
  CHECK_NEAR(glm::length(v), 30.0, 1e-3);
  CHECK_NEAR(glm::distance(landing(from, v, to), to), 0.0, 0.1); // stepping costs a few cm

  // The low arc: well under 45 degrees for a target this near.
  CHECK(v.y < 0.5f * glm::length(glm::vec2(v.x, v.z)));
  // Across, straight at it.
  CHECK_NEAR(v.x / v.z, 4.0, 1e-3);
}

TEST(a_throw_out_of_reach_goes_up_at_45_degrees_toward_it) {
  const glm::vec3 v = throwVelocity({0, 1, 0}, {500, 0, 0}, 30.0f, kGravity);
  CHECK_NEAR(glm::length(v), 30.0, 1e-3);
  CHECK_NEAR(v.y, v.x, 1e-3);
  CHECK_NEAR(v.z, 0.0, 1e-4);
}

TEST(a_throw_straight_up_or_down_goes_straight_at_it) {
  glm::vec3 v = throwVelocity({0, 1, 0}, {0, 11, 0}, 30.0f, kGravity);
  CHECK_NEAR(glm::distance(v, glm::vec3(0, 30, 0)), 0.0, 1e-3);
  v = throwVelocity({0, 1, 0}, {0, -5, 0}, 30.0f, kGravity);
  CHECK_NEAR(glm::distance(v, glm::vec3(0, -30, 0)), 0.0, 1e-3);
}
