#include "FixedStepClock.h"
#include "PhysicsWorld.h"
#include "TestRunner.h"

#include <glm/gtc/quaternion.hpp>

namespace {

constexpr double kStep = 1.0 / 60.0;

// Rotation angle of a unit quaternion, 0 to pi. atan2 rather than acos(w):
// near w = 1 a float acos can't resolve anything under about 0.0007 rad.
double angleOf(glm::quat q) {
  const double sinHalf = glm::length(glm::dvec3(q.x, q.y, q.z));
  return 2.0 * std::atan2(sinHalf, std::fabs(double(q.w)));
}

double angleBetween(glm::quat a, glm::quat b) {
  return angleOf(glm::inverse(a) * b);
}

} // namespace

TEST(clock_runs_one_step_per_step_length) {
  FixedStepClock clock;
  CHECK(clock.advance(kStep + 1e-6) == 1);
  CHECK_NEAR(clock.alpha(), 0.0, 1e-3);
}

TEST(clock_carries_leftover_time_to_the_next_frame) {
  FixedStepClock clock;
  CHECK(clock.advance(0.010) == 0);
  CHECK_NEAR(clock.alpha(), 0.010 / kStep, 1e-6);
  CHECK(clock.advance(0.010) == 1);
  CHECK_NEAR(clock.alpha(), (0.020 - kStep) / kStep, 1e-6);
}

TEST(clock_caps_steps_after_a_hitch) {
  FixedStepClock clock;
  CHECK(clock.advance(1.003) == 4);
  CHECK(clock.alpha() >= 0.0f && clock.alpha() < 1.0f);
  // The missed steps are dropped, not run as a burst over the next frames.
  CHECK(clock.advance(0.005) == 0);
}

TEST(clock_keeps_time_over_many_frames) {
  FixedStepClock clock;
  int steps = 0;
  for (int i = 0; i < 1440; ++i) // 10 s at 144 fps
    steps += clock.advance(1.0 / 144.0);
  CHECK(steps >= 599 && steps <= 600);
}

TEST(blend_hits_both_ends_and_the_middle) {
  const PhysicsPose from{glm::vec3(0.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f)};
  const PhysicsPose to{glm::vec3(2.0f, 4.0f, 6.0f),
                       glm::angleAxis(glm::radians(90.0f), glm::vec3(0, 1, 0))};

  const PhysicsPose start = blendPoses(from, to, 0.0f);
  const PhysicsPose end = blendPoses(from, to, 1.0f);
  const PhysicsPose mid = blendPoses(from, to, 0.5f);

  CHECK(start.position == from.position);
  CHECK_NEAR(angleBetween(start.rotation, from.rotation), 0.0, 1e-3);
  CHECK(end.position == to.position);
  CHECK_NEAR(angleBetween(end.rotation, to.rotation), 0.0, 1e-3);
  CHECK_NEAR(mid.position.y, 2.0, 1e-6);
  CHECK_NEAR(angleOf(mid.rotation), glm::radians(45.0), 1e-3);
}

TEST(blend_takes_the_short_way_round) {
  // q and -q are the same rotation. Blending towards the negated one must
  // still turn 0.1 rad at the midpoint, not most of a full turn.
  const glm::quat small = glm::angleAxis(0.2f, glm::vec3(0, 1, 0));
  const PhysicsPose from{glm::vec3(0.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f)};
  const PhysicsPose to{glm::vec3(0.0f), -small};

  CHECK_NEAR(angleOf(blendPoses(from, to, 0.5f).rotation), 0.1, 1e-3);
}

TEST(blend_matches_slerp_for_one_step_of_spin) {
  // 0.5 rad between two steps is a body spinning at 30 rad/s.
  const glm::quat a(1.0f, 0.0f, 0.0f, 0.0f);
  const glm::quat b =
      glm::angleAxis(0.5f, glm::normalize(glm::vec3(1.0f, 2.0f, 3.0f)));
  double worst = 0.0;
  for (int i = 0; i <= 100; ++i) {
    const float t = i / 100.0f;
    const PhysicsPose p = blendPoses({glm::vec3(0.0f), a}, {glm::vec3(0.0f), b}, t);
    worst = std::fmax(worst, angleBetween(p.rotation, glm::slerp(a, b, t)));
  }
  CHECK(worst < 1e-3);
}
