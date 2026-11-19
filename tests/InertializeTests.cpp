#include "Inertialize.h"
#include "TestRunner.h"

#include <cmath>

namespace {

const glm::vec3 kUp(0.0f, 1.0f, 0.0f);

// Two bones; the second carries the offsets the tests look at.
Pose twoBones() {
  Pose p;
  p.local.resize(2);
  return p;
}

Pose turned(float degrees, glm::vec3 shift = glm::vec3(0.0f)) {
  Pose p = twoBones();
  p.local[1].rotation = glm::angleAxis(glm::radians(degrees), kUp);
  p.local[1].translation = shift;
  return p;
}

// The second bone's turn about y, degrees (signed).
float yaw(const Pose &p) {
  const glm::quat q = p.local[1].rotation;
  return glm::degrees(2.0f * std::atan2(q.y, q.w));
}

} // namespace

TEST(a_quintic_from_rest_is_one_minus_smootherstep) {
  // Bollo's check: x0 = 1, v0 = 0, t1 = 1 gives 1 - (6t^5 - 15t^4 + 10t^3).
  const QuinticDecay q = QuinticDecay::make(1.0f, 0.0f, 1.0f);
  for (float t : {0.0f, 0.1f, 0.25f, 0.5f, 0.8f, 1.0f}) {
    const float smooth = t * t * t * (t * (6.0f * t - 15.0f) + 10.0f);
    CHECK_NEAR(q.at(t), 1.0f - smooth, 1e-5);
  }
  CHECK_NEAR(q.at(2.0f), 0.0, 1e-6);
}

TEST(a_quintic_keeps_the_speed_it_starts_with) {
  const QuinticDecay q = QuinticDecay::make(1.0f, -2.0f, 1.0f);
  CHECK_NEAR(q.at(0.0f), 1.0, 1e-6);
  const float h = 1e-3f;
  CHECK_NEAR((q.at(h) - q.at(0.0f)) / h, -2.0, 0.02);
}

TEST(a_fast_approach_ends_sooner_and_never_overshoots) {
  // Closing at 10 per second from 1: t1 shrinks to -5 x0 / v0 = 0.5.
  const QuinticDecay q = QuinticDecay::make(1.0f, -10.0f, 1.0f);
  CHECK_NEAR(q.t1, 0.5, 1e-6);
  CHECK_NEAR(q.at(0.5f), 0.0, 1e-5);
  for (float t = 0.0f; t <= 1.0f; t += 0.01f)
    CHECK(q.at(t) >= -1e-5f);
}

TEST(moving_away_counts_as_standing_still) {
  const QuinticDecay away = QuinticDecay::make(1.0f, 3.0f, 1.0f);
  const QuinticDecay still = QuinticDecay::make(1.0f, 0.0f, 1.0f);
  for (float t : {0.1f, 0.4f, 0.7f})
    CHECK_NEAR(away.at(t), still.at(t), 1e-6);
}

TEST(an_inertialized_pose_starts_where_it_was_and_ends_on_the_target) {
  // The bone was turned 90 degrees and shifted, standing still; the new
  // pose has neither. Half way through, an offset from rest is half gone.
  const Pose last = turned(90.0f, glm::vec3(0.0f, 0.0f, 0.4f));
  const Pose target = turned(0.0f);
  PoseInertialization blend;
  blend.start(last, last, 1.0f / 60.0f, target, 0.2f);
  Pose p = target;
  blend.apply(0.0f, p);
  CHECK_NEAR(yaw(p), 90.0, 1e-3);
  CHECK_NEAR(p.local[1].translation.z, 0.4, 1e-5);
  p = target;
  blend.apply(0.1f, p);
  CHECK_NEAR(yaw(p), 45.0, 1e-2);
  CHECK_NEAR(p.local[1].translation.z, 0.2, 1e-4);
  p = target;
  blend.apply(0.2f, p);
  CHECK_NEAR(yaw(p), 0.0, 1e-4);
  CHECK_NEAR(p.local[1].translation.z, 0.0, 1e-6);
  CHECK(!blend.active(0.2f));
  CHECK(blend.active(0.1f));
}

TEST(an_inertialized_pose_carries_on_moving_the_way_it_was) {
  // Turning toward the target at 10 degrees a frame: the first frame after
  // the change goes on about 10 degrees, not stopping dead.
  const float dt = 1.0f / 60.0f;
  PoseInertialization blend;
  blend.start(turned(90.0f), turned(100.0f), dt, turned(0.0f), 0.2f);
  Pose p = turned(0.0f);
  blend.apply(dt, p);
  CHECK_NEAR(yaw(p), 80.0, 0.5);
}

TEST(an_inertialized_pose_never_overshoots_the_target) {
  const float dt = 1.0f / 60.0f;
  PoseInertialization blend;
  blend.start(turned(30.0f), turned(60.0f), dt, turned(0.0f), 0.3f);
  for (float t = 0.0f; t <= 0.35f; t += dt) {
    Pose p = turned(0.0f);
    blend.apply(t, p);
    CHECK(yaw(p) >= -1e-3f);
  }
}

TEST(a_change_in_the_middle_of_a_blend_has_no_jump) {
  // From 90 toward 0, then half way through toward -40: consecutive
  // frames stay close and the motion keeps its direction for a moment.
  const float dt = 1.0f / 60.0f;
  PoseInertialization blend;
  blend.start(turned(90.0f), turned(90.0f), dt, turned(0.0f), 0.2f);
  Pose before = turned(0.0f), last = turned(0.0f);
  float elapsed = 0.0f;
  for (int i = 0; i < 6; ++i) {
    elapsed += dt;
    before = last;
    last = turned(0.0f);
    blend.apply(elapsed, last);
  }
  const float step = yaw(before) - yaw(last); // degrees a frame, toward 0
  CHECK(step > 1.0f);
  blend.start(last, before, dt, turned(-40.0f), 0.2f);
  Pose next = turned(-40.0f);
  blend.apply(dt, next);
  CHECK_NEAR(yaw(last) - yaw(next), step, 0.5);
}

TEST(a_zero_length_blend_is_the_target_at_once) {
  PoseInertialization blend;
  blend.start(turned(90.0f), turned(90.0f), 1.0f / 60.0f, turned(0.0f), 0.0f);
  Pose p = turned(0.0f);
  blend.apply(0.0f, p);
  CHECK_NEAR(yaw(p), 0.0, 1e-5);
  CHECK(!blend.active(0.0f));
}

TEST(no_offset_stays_no_offset_however_fast_it_was_closing) {
  // Already on the target, having been moving toward it: nothing to ease.
  const QuinticDecay q = QuinticDecay::make(0.0f, -5.0f, 0.2f);
  for (float t : {0.0f, 0.05f, 0.1f, 0.19f})
    CHECK_NEAR(q.at(t), 0.0, 1e-9);
}

TEST(a_tiny_offset_closing_fast_gives_no_infinities) {
  // t1 = -5 x0 / v0 would be about 5e-10 s: the offset just goes.
  const QuinticDecay q = QuinticDecay::make(1e-9f, -10.0f, 0.2f);
  for (float t : {0.0f, 1e-12f, 1e-6f, 0.1f}) {
    CHECK(std::isfinite(q.at(t)));
    CHECK(std::abs(q.at(t)) <= 1e-9f);
  }
}
