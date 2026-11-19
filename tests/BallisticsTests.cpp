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

namespace {

// The AKM's ball round: G7 0.15, fitted to NSD's table (research), 715 m/s.
const Projectile kAkm{DragModel::G7, 0.15f};
constexpr float kAkmSpeed = 715.0f;
constexpr float kStep = 1.0f / 240.0f; // the bullets' sub-step in the game

struct Passing {
  float speed = 0.0f, time = 0.0f, height = 0.0f, side = 0.0f;
};

// Fired along +x from the origin, up by `angle`: the bullet as it passes
// `range` m downrange (between steps, by straight lines).
Passing passing(const Projectile &p, float speed, float range, const Air &air = {},
                float angle = 0.0f) {
  Flight f;
  f.velocity = glm::vec3(std::cos(angle), std::sin(angle), 0.0f) * speed;
  float t = 0.0f;
  for (int i = 0; i < 100000; ++i) {
    const Flight before = f;
    stepFlight(f, p, air, kGravity, kStep);
    if (f.position.x >= range) {
      const float u = (range - before.position.x) / (f.position.x - before.position.x);
      return {glm::length(glm::mix(before.velocity, f.velocity, u)), t + u * kStep,
              glm::mix(before.position.y, f.position.y, u),
              glm::mix(before.position.z, f.position.z, u)};
    }
    t += kStep;
  }
  return {};
}

} // namespace

TEST(the_standard_drag_tables_have_their_published_values) {
  CHECK_NEAR(standardDrag(DragModel::G1, 0.0f), 0.2629, 1e-5);
  CHECK_NEAR(standardDrag(DragModel::G1, 1.0f), 0.4805, 1e-5);
  CHECK_NEAR(standardDrag(DragModel::G7, 1.0f), 0.3803, 1e-5);
  CHECK_NEAR(standardDrag(DragModel::G7, 2.0f), 0.2980, 1e-5);
  // Between rows, a straight line; past the last, the last.
  CHECK_NEAR(standardDrag(DragModel::G7, 0.9125f), 0.5 * (0.1464 + 0.1660), 1e-5);
  CHECK_NEAR(standardDrag(DragModel::G7, 9.0f), 0.1618, 1e-5);
  CHECK_NEAR(standardDrag(DragModel::G1, -1.0f), 0.2629, 1e-5);
  CHECK_NEAR(standardDrag(DragModel::G1, std::nanf("")), 0.2629, 1e-5);
}

TEST(drag_slows_a_bullet_by_pi_rho_v_squared_cd_over_8_bc) {
  // At Mach 2 in sea-level air: G7's 0.2980, the coefficient in kg/m^2.
  const float v = 2.0f * 340.29f;
  const double expected =
      3.14159265 * 1.225 * v * v * 0.2980 / (8.0 * 0.15 * kBallisticCoefficientUnit);
  CHECK_NEAR(dragDeceleration(kAkm, v, Air{}), expected, expected * 1e-4);
  Air thin;
  thin.density = 0.0f;
  CHECK_NEAR(dragDeceleration(kAkm, v, thin), 0.0, 1e-6);
}

TEST(an_akm_bullet_matches_the_soviet_manuals_table) {
  // NSD p.159: speed and time of flight at 100, 400 and 800 m from 715 m/s.
  const Passing at100 = passing(kAkm, kAkmSpeed, 100.0f);
  CHECK_NEAR(at100.speed, 623.0, 5.0);
  CHECK_NEAR(at100.time, 0.15, 0.01);
  const Passing at400 = passing(kAkm, kAkmSpeed, 400.0f);
  CHECK_NEAR(at400.speed, 391.0, 8.0);
  CHECK_NEAR(at400.time, 0.76, 0.02);
  const Passing at800 = passing(kAkm, kAkmSpeed, 800.0f);
  CHECK_NEAR(at800.speed, 266.0, 10.0);
  CHECK_NEAR(at800.time, 2.05, 0.05);
}

TEST(zeroed_at_100_m_the_bullet_crosses_the_line_of_sight_there) {
  const float angle = zeroAngle(kAkm, kAkmSpeed, 100.0f, Air{}, kGravity);
  // About the drop over 0.15 s, a little over a milliradian.
  CHECK(angle > 0.0008f && angle < 0.0014f);
  CHECK_NEAR(passing(kAkm, kAkmSpeed, 100.0f, Air{}, angle).height, 0.0, 0.002);
  CHECK(passing(kAkm, kAkmSpeed, 50.0f, Air{}, angle).height > 0.01f);
  CHECK(passing(kAkm, kAkmSpeed, 200.0f, Air{}, angle).height < -0.05f);
  // Out of reach (or nothing to reach): no angle.
  CHECK_NEAR(zeroAngle(kAkm, kAkmSpeed, 0.0f, Air{}, kGravity), 0.0, 1e-6);
}

TEST(a_crosswind_drifts_a_bullet_by_the_time_drag_cost_it) {
  // The drift rule (Didion): wind x (time of flight - time in a vacuum).
  Air windy;
  windy.wind = glm::vec3(0.0f, 0.0f, 10.0f);
  const Passing calm = passing(kAkm, kAkmSpeed, 300.0f);
  const Passing drift = passing(kAkm, kAkmSpeed, 300.0f, windy);
  const double expected = 10.0 * (calm.time - 300.0 / kAkmSpeed);
  CHECK_NEAR(drift.side, expected, 0.03 * expected);
}

TEST(with_no_air_only_gravity_acts) {
  Air none;
  none.density = 0.0f;
  const Passing p = passing(kAkm, kAkmSpeed, 200.0f, none);
  const double t = 200.0 / kAkmSpeed;
  CHECK_NEAR(p.time, t, 1e-4);
  CHECK_NEAR(p.height, -0.5 * 9.81 * t * t, 1e-3);
}
