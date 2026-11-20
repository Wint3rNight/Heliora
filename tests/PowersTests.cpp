#include "Powers.h"
#include "TestRunner.h"

#include <cmath>
#include <string>

namespace {

constexpr float kDt = 1.0f / 60.0f;
const glm::vec3 kGravity(0.0f, -9.81f, 0.0f);

// A held point mass over `seconds`, stepped the way Jolt steps a body: the
// velocity the hold sets, then gravity, then the move. Returns how far it
// ever went past the target along the way it came from, and when it first
// got within `near` of it (or -1).
struct Held {
  glm::vec3 position{0.0f};
  glm::vec3 velocity{0.0f};
  float overshoot = 0.0f;
  float reached = -1.0f;
  float fastest = 0.0f;
};
Held holdFor(glm::vec3 start, glm::vec3 target, float mass, float seconds, float near = 0.1f) {
  const PowerTuning t;
  Held h;
  h.position = start;
  const glm::vec3 way = glm::normalize(target - start);
  for (int i = 0; i * kDt < seconds; ++i) {
    h.velocity = holdVelocity(h.position, h.velocity, target, mass, kGravity, kDt, t);
    h.velocity += kGravity * kDt;
    h.position += h.velocity * kDt;
    h.overshoot = std::max(h.overshoot, glm::dot(h.position - target, way));
    h.fastest = std::max(h.fastest, glm::length(h.velocity));
    if (h.reached < 0.0f && glm::distance(h.position, target) < near)
      h.reached = (i + 1) * kDt;
  }
  return h;
}

// Where something thrown with this velocity from `from` is when it's gone
// as far across as `to`, stepped at 60 Hz the way Jolt steps it (gravity
// into the velocity, then the move; no damping).
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

PowerInput aiming(int target = -1, GrabKind kind = GrabKind::Lift) {
  PowerInput in;
  in.aim = true;
  in.target = target;
  in.targetKind = kind;
  return in;
}

// Runs the controller with this input for `seconds`, gathering the events.
PowerEvents runFor(PowerController &c, const PowerInput &in, float seconds) {
  const PowerTuning t;
  PowerEvents all;
  for (int i = 0; i * kDt < seconds - 1e-5f; ++i) {
    const PowerEvents e = c.update(in, kDt, t);
    if (e.grabbed >= 0)
      all.grabbed = e.grabbed;
    all.dropped = all.dropped || e.dropped;
    all.thrown = all.thrown || e.thrown;
    all.burst = all.burst || e.burst;
  }
  return all;
}

} // namespace

TEST(heavy_things_slow_down_by_the_root_of_their_mass) {
  CHECK_NEAR(heavySlowdown(100.0f, 300.0f), 1.0, 1e-6);
  CHECK_NEAR(heavySlowdown(300.0f, 300.0f), 1.0, 1e-6);
  CHECK_NEAR(heavySlowdown(1200.0f, 300.0f), 0.5, 1e-6);
}

TEST(the_power_lifts_moving_things_up_to_its_cap) {
  const PowerTuning t;
  CHECK(!liftable(0.0f, t)); // static
  CHECK(liftable(50.0f, t));
  CHECK(liftable(2500.0f, t));
  CHECK(!liftable(2501.0f, t));
}

TEST(the_hold_point_is_past_the_character_along_the_aim_and_raised) {
  const glm::vec3 p = holdPoint({0, 2, 5}, {0, 0, -1}, 3.0f, 4.0f, 0.5f);
  CHECK_NEAR(glm::distance(p, glm::vec3(0, 2.5f, -2)), 0.0, 1e-5);
}

TEST(a_grabbed_thing_rises_from_where_it_was_to_the_hold_point) {
  const PowerTuning t;
  const glm::vec3 from(0, 0, 0), hold(0, 2, -4);
  CHECK_NEAR(glm::distance(liftPoint(from, hold, 0.0f, t), from), 0.0, 1e-5);
  CHECK_NEAR(glm::distance(liftPoint(from, hold, t.liftSeconds, t), hold), 0.0, 1e-5);
  CHECK_NEAR(glm::distance(liftPoint(from, hold, 5.0f, t), hold), 0.0, 1e-5);
  const glm::vec3 half = liftPoint(from, hold, 0.5f * t.liftSeconds, t);
  CHECK_NEAR(glm::distance(half, 0.5f * (from + hold)), 0.0, 1e-4); // eased both ways: halfway at half time
  const glm::vec3 early = liftPoint(from, hold, 0.1f * t.liftSeconds, t);
  CHECK(glm::distance(early, from) < 0.1f * glm::distance(hold, from)); // eased in
}

namespace {

// A point swept among walls, each a point on it and the way it faces.
struct Wall {
  glm::vec3 on, facing;
};
HoldSweep sweepAmong(std::vector<Wall> walls) {
  return [walls](glm::vec3 start, glm::vec3 move) {
    std::optional<HoldContact> first;
    for (const Wall &w : walls) {
      const float gap = glm::dot(start - w.on, w.facing);
      const float closing = -glm::dot(move, w.facing);
      if (closing <= 1e-6f || gap < -1e-4f)
        continue; // along it, away from it, or behind it
      const float f = std::max(0.0f, gap) / closing;
      if (f <= 1.0f && (!first || f < first->fraction))
        first = HoldContact{f, w.facing};
    }
    return first;
  };
}

} // namespace

TEST(a_held_thing_stops_at_what_is_in_its_way_and_slides_along_it) {
  const glm::vec3 at(0, 1, 0), hold(2, 1, -4);
  const Wall ahead{{0, 0, -2}, {0, 0, 1}};
  CHECK_NEAR(glm::distance(slideHoldPoint(at, hold, sweepAmong({})), hold), 0.0, 1e-5);
  // A wall halfway: it gets there, then the rest of the way across, not in.
  CHECK_NEAR(glm::distance(slideHoldPoint(at, hold, sweepAmong({ahead})), glm::vec3(2, 1, -2)), 0.0, 1e-4);
  // Already against it.
  CHECK_NEAR(glm::distance(slideHoldPoint({0, 1, -2}, {2, 1, -4}, sweepAmong({ahead})),
                           glm::vec3(2, 1, -2)),
             0.0, 1e-4);
}

TEST(a_held_thing_pulled_into_a_corner_stays_in_it) {
  const Wall ahead{{0, 0, -2}, {0, 0, 1}}, side{{1, 0, 0}, {-1, 0, 0}};
  // It meets the side first, slides along it, then meets the wall ahead: not
  // through either.
  const glm::vec3 p = slideHoldPoint({0, 1, 0}, {3, 1, -4}, sweepAmong({ahead, side}));
  CHECK_NEAR(glm::distance(p, glm::vec3(1, 1, -2)), 0.0, 1e-4);
}

TEST(a_held_thing_reaches_its_point_without_going_past_it) {
  const Held h = holdFor({0, 0, 0}, {0, 1.5f, -4}, 50.0f, 2.0f);
  CHECK(h.reached > 0.0f);
  CHECK(h.overshoot < 0.02f);
  CHECK(glm::distance(h.position, glm::vec3(0, 1.5f, -4)) < 0.01f);
}

TEST(a_heavy_thing_gets_there_slower_than_a_light_one) {
  const Held light = holdFor({0, 0, 0}, {0, 1.5f, -4}, 50.0f, 4.0f);
  const Held heavy = holdFor({0, 0, 0}, {0, 1.5f, -4}, 2000.0f, 4.0f);
  CHECK(light.reached > 0.0f && heavy.reached > 0.0f);
  CHECK(heavy.reached > 1.5f * light.reached);
  CHECK(heavy.overshoot < 0.02f); // slower, not floppier
}

TEST(a_held_thing_hangs_still_against_gravity) {
  const Held h = holdFor({1, 2, 3}, {1, 2, 3.0001f}, 500.0f, 3.0f);
  CHECK(glm::distance(h.position, glm::vec3(1, 2, 3)) < 0.005f);
}

TEST(a_held_thing_is_no_faster_than_its_cap) {
  const PowerTuning t;
  const Held light = holdFor({0, 0, 0}, {0, 0, -100}, 50.0f, 1.0f);
  CHECK(light.fastest <= t.holdSpeed + 1e-3f);
  CHECK(light.fastest > 0.9f * t.holdSpeed);
  const Held heavy = holdFor({0, 0, 0}, {0, 0, -100}, 1200.0f, 2.0f);
  CHECK(heavy.fastest <= 0.5f * t.holdSpeed + 1e-3f);
}

TEST(a_held_things_spin_eases_to_a_slow_tumble) {
  const PowerTuning t;
  glm::vec3 spin(5.0f, 0.0f, 0.0f);
  for (int i = 0; i < 180; ++i)
    spin = holdSpin(spin, {0, 1, 0}, kDt, t);
  CHECK_NEAR(glm::distance(spin, glm::vec3(0, t.tumble, 0)), 0.0, 0.02);
}

TEST(heavier_things_are_thrown_slower) {
  const PowerTuning t;
  CHECK_NEAR(throwSpeed(25.0f, t), t.throwSpeed, 1e-4);         // a crate, full speed
  CHECK_NEAR(throwSpeed(100.0f, t), 0.5 * t.throwSpeed, 1e-4);  // a stone piece: one heavy ball's hit
  CHECK_NEAR(throwSpeed(400.0f, t), 0.25 * t.throwSpeed, 1e-4);
}

TEST(the_burst_falls_off_with_distance_and_misses_outside_its_cone) {
  const PowerTuning t;
  const glm::vec3 hand(0, 1, 0), aim(0, 0, -1);
  CHECK(burstFalloff(hand, aim, {0, 1, -0.01f}, t) > 0.99f);
  const float mid = burstFalloff(hand, aim, {0, 1, -4}, t);
  CHECK(mid > 0.0f && mid < 1.0f);
  CHECK(burstFalloff(hand, aim, {0, 1, -6}, t) < mid);
  CHECK(burstFalloff(hand, aim, {0, 1, -8.5f}, t) == 0.0f); // past its reach
  CHECK(burstFalloff(hand, aim, {0, 1, 3}, t) == 0.0f);     // behind
  // 40 degrees off the aim, outside a 70 degree cone; 30 inside it.
  const float off40 = glm::radians(40.0f), off30 = glm::radians(30.0f);
  CHECK(burstFalloff(hand, aim, hand + 3.0f * glm::vec3(std::sin(off40), 0, -std::cos(off40)), t) ==
        0.0f);
  CHECK(burstFalloff(hand, aim, hand + 3.0f * glm::vec3(std::sin(off30), 0, -std::cos(off30)), t) >
        0.0f);
}

TEST(the_burst_throws_things_away_from_the_hand_tilted_up_lighter_ones_faster) {
  const PowerTuning t;
  const glm::vec3 hand(0, 1, 0), aim(0, 0, -1);
  const glm::vec3 light = burstKick(hand, aim, {1, 1, -2}, 100.0f, t);
  CHECK(light.y > 0.0f);
  CHECK(light.x > 0.0f && light.z < 0.0f); // away from the hand
  CHECK_NEAR(light.x / light.z, -0.5, 1e-3);
  CHECK_NEAR(glm::length(light), t.burstSpeed * burstFalloff(hand, aim, {1, 1, -2}, t), 1e-3);
  const glm::vec3 heavy = burstKick(hand, aim, {1, 1, -2}, 2000.0f, t);
  CHECK_NEAR(glm::length(light) / glm::length(heavy), std::sqrt(2000.0 / 200.0), 1e-3);
  CHECK(glm::length(burstKick(hand, aim, {0, 1, 3}, 100.0f, t)) == 0.0f);
  // Right at the hand: along the aim.
  const glm::vec3 at = burstKick(hand, aim, hand, 100.0f, t);
  CHECK(at.z < 0.0f && glm::length(at) > 0.0f);
}

TEST(tearing_takes_longer_for_stronger_links_within_its_limits) {
  const PowerTuning t;
  CHECK_NEAR(tearSeconds(0.0f, t), t.tearShortest, 1e-6);
  CHECK(tearSeconds(100.0f, t) > t.tearShortest);
  CHECK(tearSeconds(400.0f, t) > tearSeconds(100.0f, t));
  CHECK_NEAR(tearSeconds(100.0f, t), 1.0, 1e-4); // a column piece's links, about a second
  CHECK_NEAR(tearSeconds(1e6f, t), t.tearLongest, 1e-6);
}

TEST(a_tear_breaks_the_weakest_link_first_and_the_last_at_its_end) {
  const PowerTuning t;
  const std::vector<float> times = tearTimes({40.0f, 10.0f, 25.0f}, t);
  const float total = tearSeconds(75.0f, t);
  CHECK(times.size() == 3);
  if (times.size() == 3) {
    CHECK_NEAR(times[1], total * 10.0f / 75.0f, 1e-5);
    CHECK_NEAR(times[2], total * 35.0f / 75.0f, 1e-5);
    CHECK_NEAR(times[0], total, 1e-5);
  }
  CHECK(tearTimes({}, t).empty());
  // Links of no strength go at once.
  const std::vector<float> zero = tearTimes({0.0f, 0.0f}, t);
  CHECK(zero.size() == 2 && zero[0] <= t.tearShortest && zero[1] <= t.tearShortest);
}

TEST(the_pick_takes_what_is_nearest_the_crosshair_not_nearest_the_player) {
  const PowerTuning t;
  const std::vector<PickCandidate> c = {
      {{0, 0, -10}, 0.5f},   // straight ahead
      {{0.3f, 0, -3}, 0.2f}, // nearer, a little off the line
      {{0, 0, 5}, 0.5f},     // behind
      {{0, 0, -40}, 0.5f},   // too far
      {{3, 0, -10}, 0.5f},   // too far off the line
  };
  const std::vector<int> order = pickOrder(c, {0, 0, 0}, {0, 0, -1}, t);
  CHECK(order.size() == 2);
  CHECK(order.size() == 2 && order[0] == 0 && order[1] == 1);
}

TEST(the_pick_skips_what_is_behind_even_from_inside_its_reach) {
  const PowerTuning t;
  // The camera pressed against a wall piece behind it: inside its sphere.
  std::vector<int> order = pickOrder({{{0, 0, 0.4f}, 0.8f}, {{0, 0, -5}, 0.3f}}, {0, 0, 0}, {0, 0, -1}, t);
  CHECK(order.size() == 1 && order[0] == 1);
  // Inside the reach of something big ahead: still picked.
  order = pickOrder({{{0, 0, -0.3f}, 1.0f}}, {0, 0, 0}, {0, 0, -1}, t);
  CHECK(order.size() == 1);
}

TEST(a_big_thing_is_picked_by_its_edge) {
  const PowerTuning t;
  // Its middle 11 degrees off, but 2 m across at 10 m: its edge is on the line.
  const std::vector<int> order = pickOrder({{{2, 0, -10}, 2.0f}}, {0, 0, 0}, {0, 0, -1}, t);
  CHECK(order.size() == 1);
}

TEST(the_reveal_glow_waits_for_the_pulse_then_fades) {
  const PowerTuning t;
  // 10 m away, the pulse arrives at 0.4 s.
  CHECK(revealGlow(10.0f, 0.3f, t) == 0.0f);
  CHECK(revealGlow(10.0f, 0.6f, t) > 0.9f);
  const float later = revealGlow(10.0f, 3.4f, t);
  CHECK(later > 0.0f && later < revealGlow(10.0f, 0.6f, t));
  CHECK(revealGlow(10.0f, 0.4f + t.revealSeconds, t) == 0.0f);
  CHECK(revealGlow(35.0f, 2.0f, t) == 0.0f); // past its reach
}

TEST(every_power_state_has_a_name) {
  CHECK(std::string(powerStateName(PowerState::Idle)) == "Idle");
  CHECK(std::string(powerStateName(PowerState::Bursting)) == "Bursting");
  CHECK(std::string(powerStateName(PowerState::Straining)) == "Straining");
}

TEST(right_mouse_going_down_on_a_target_grabs_it_and_lifts_it_into_the_hold) {
  PowerController c;
  const PowerTuning t;
  const PowerEvents e = c.update(aiming(7), kDt, t);
  CHECK(e.grabbed == 7);
  CHECK(c.state() == PowerState::Lifting && c.held() == 7 && c.holding());
  runFor(c, aiming(-1), t.liftSeconds + kDt);
  CHECK(c.state() == PowerState::Holding && c.held() == 7);
}

TEST(with_right_mouse_held_the_crosshair_resting_on_a_target_grabs_it) {
  PowerController c;
  const PowerTuning t;
  runFor(c, aiming(-1), 0.5f); // aiming at nothing
  CHECK(c.state() == PowerState::Idle);
  PowerEvents e = runFor(c, aiming(3), t.pickDwell - 2.0f * kDt);
  CHECK(e.grabbed < 0 && c.state() == PowerState::Idle);
  runFor(c, aiming(4), t.pickDwell - 2.0f * kDt); // moved onto another: counting again
  CHECK(c.state() == PowerState::Idle);
  e = runFor(c, aiming(4), 3.0f * kDt);
  CHECK(e.grabbed == 4 && c.state() == PowerState::Lifting);
}

TEST(nothing_is_grabbed_or_blasted_without_aiming) {
  PowerController c;
  PowerInput in;
  in.target = 2;
  in.cast = true;
  const PowerEvents e = runFor(c, in, 1.0f);
  CHECK(e.grabbed < 0 && !e.burst && c.state() == PowerState::Idle);
}

TEST(a_piece_still_part_of_something_is_torn_out_then_lifted) {
  PowerController c;
  const PowerTuning t;
  c.update(aiming(5, GrabKind::Tear), kDt, t);
  CHECK(c.state() == PowerState::Tearing && c.held() == 5);
  PowerInput torn = aiming(5);
  torn.torn = true;
  c.update(torn, kDt, t);
  CHECK(c.state() == PowerState::Lifting && c.held() == 5);
}

TEST(something_too_heavy_is_strained_against_and_nothing_more) {
  PowerController c;
  const PowerTuning t;
  c.update(aiming(9, GrabKind::Strain), kDt, t);
  CHECK(c.state() == PowerState::Straining);
  const PowerEvents let = c.update(PowerInput{}, kDt, t);
  CHECK(let.dropped && c.state() == PowerState::Idle && !c.holding());
}

TEST(the_burst_while_holding_tearing_or_straining_lets_go_and_bursts) {
  const PowerTuning t;
  for (GrabKind kind : {GrabKind::Lift, GrabKind::Tear, GrabKind::Strain}) {
    PowerController c;
    c.update(aiming(5, kind), kDt, t);
    runFor(c, aiming(5, kind), 0.5f);
    PowerInput burst = aiming(5, kind);
    burst.burst = true;
    PowerEvents e = c.update(burst, kDt, t);
    CHECK(e.dropped && !e.thrown && !c.holding());
    CHECK(c.state() == PowerState::Bursting);
    e = runFor(c, aiming(5, kind), t.burstDelay + kDt);
    CHECK(e.burst);
  }
}

TEST(left_mouse_with_nothing_lifted_does_nothing) {
  const PowerTuning t;
  PowerController idle;
  PowerInput cast = aiming(-1);
  cast.cast = true;
  PowerEvents e = runFor(idle, cast, 0.5f);
  CHECK(!e.burst && idle.state() == PowerState::Idle);
  for (GrabKind kind : {GrabKind::Tear, GrabKind::Strain}) {
    PowerController c;
    c.update(aiming(5, kind), kDt, t);
    PowerInput click = aiming(5, kind);
    click.cast = true;
    e = c.update(click, kDt, t);
    CHECK(!e.dropped && !e.burst && c.held() == 5);
  }
}

TEST(resting_the_crosshair_on_a_structure_or_something_too_heavy_takes_a_press) {
  PowerController c;
  const PowerTuning t;
  runFor(c, aiming(-1), 0.2f); // aiming at nothing first
  PowerEvents e = runFor(c, aiming(5, GrabKind::Tear), 1.0f);
  CHECK(e.grabbed < 0 && c.state() == PowerState::Idle);
  e = runFor(c, aiming(9, GrabKind::Strain), 1.0f);
  CHECK(e.grabbed < 0 && c.state() == PowerState::Idle);
  c.update(PowerInput{}, kDt, t);
  e = c.update(aiming(5, GrabKind::Tear), kDt, t); // a fresh press does
  CHECK(e.grabbed == 5 && c.state() == PowerState::Tearing);
}

TEST(letting_go_of_right_mouse_drops_what_is_held) {
  PowerController c;
  const PowerTuning t;
  c.update(aiming(1), kDt, t);
  runFor(c, aiming(1), 1.0f);
  const PowerEvents e = c.update(PowerInput{}, kDt, t);
  CHECK(e.dropped && c.state() == PowerState::Idle && c.held() < 0);
}

TEST(left_mouse_throws_what_is_held_once_after_its_delay) {
  PowerController c;
  const PowerTuning t;
  c.update(aiming(1), kDt, t);
  runFor(c, aiming(1), 1.0f);
  PowerInput cast = aiming(1);
  cast.cast = true;
  PowerEvents e = c.update(cast, kDt, t);
  CHECK(c.state() == PowerState::Throwing && !e.thrown && c.holding());
  e = runFor(c, aiming(1), t.throwDelay);
  CHECK(e.thrown && !c.holding());
  e = runFor(c, aiming(1), t.throwRecover - t.throwDelay - 2.0f * kDt);
  CHECK(!e.thrown && c.state() == PowerState::Throwing);
  runFor(c, aiming(-1), 3.0f * kDt);
  CHECK(c.state() == PowerState::Idle);
}

TEST(a_throw_goes_even_if_right_mouse_is_let_go_before_it_leaves) {
  PowerController c;
  const PowerTuning t;
  c.update(aiming(1), kDt, t);
  runFor(c, aiming(1), 1.0f);
  PowerInput cast = aiming(1);
  cast.cast = true;
  c.update(cast, kDt, t);
  const PowerEvents e = runFor(c, PowerInput{}, t.throwDelay + kDt);
  CHECK(e.thrown && !e.dropped);
}

TEST(the_burst_goes_aiming_or_not_at_most_once_per_cooldown) {
  PowerController c;
  const PowerTuning t;
  PowerInput cast; // not aiming
  cast.burst = true;
  PowerEvents e = c.update(cast, kDt, t);
  CHECK(c.state() == PowerState::Bursting && !e.burst);
  e = runFor(c, aiming(-1), t.burstDelay);
  CHECK(e.burst);
  runFor(c, aiming(-1), t.burstRecover - t.burstDelay);
  CHECK(c.state() == PowerState::Idle);
  // Back to idle but still cooling down: no burst.
  e = runFor(c, cast, kDt);
  e = runFor(c, aiming(-1), 0.2f);
  CHECK(!e.burst && c.state() == PowerState::Idle);
  runFor(c, aiming(-1), t.burstCooldown);
  PowerInput aimed = aiming(-1);
  aimed.burst = true;
  c.update(aimed, kDt, t);
  e = runFor(c, aiming(-1), t.burstDelay + kDt);
  CHECK(e.burst);
}

TEST(losing_the_held_thing_drops_it) {
  PowerController c;
  const PowerTuning t;
  c.update(aiming(1), kDt, t);
  runFor(c, aiming(1), 1.0f);
  PowerInput lost = aiming(1);
  lost.lost = true;
  const PowerEvents e = c.update(lost, kDt, t);
  CHECK(e.dropped && c.state() == PowerState::Idle && !c.holding());
}

TEST(right_after_a_throw_resting_on_anything_grabs_nothing) {
  PowerController c;
  const PowerTuning t;
  c.update(aiming(1), kDt, t);
  runFor(c, aiming(1), 1.0f);
  PowerInput cast = aiming(1);
  cast.cast = true;
  c.update(cast, kDt, t);
  // It flies along the crosshair's line and lands where it points, and so
  // does whatever it breaks into, under new ids.
  PowerEvents e = runFor(c, aiming(1), 0.5f);
  e = runFor(c, aiming(6), 0.8f);
  CHECK(e.grabbed < 0 && c.state() == PowerState::Idle);
  // Later, resting grabs again; a press always does.
  e = runFor(c, aiming(2), t.regrabSeconds);
  CHECK(e.grabbed == 2);
  PowerController d;
  d.update(aiming(1), kDt, t);
  runFor(d, aiming(1), 1.0f);
  d.update(cast, kDt, t);
  runFor(d, aiming(-1), 1.0f);
  d.update(PowerInput{}, kDt, t);
  e = d.update(aiming(1), kDt, t);
  CHECK(e.grabbed == 1);
}

TEST(right_after_a_burst_resting_on_flying_debris_grabs_nothing) {
  PowerController c;
  const PowerTuning t;
  PowerInput cast = aiming(-1);
  cast.burst = true;
  c.update(cast, kDt, t);
  runFor(c, aiming(-1), t.burstRecover + kDt);
  CHECK(c.state() == PowerState::Idle);
  // What it blasted flies past the crosshair.
  PowerEvents e = runFor(c, aiming(3), 1.0f);
  CHECK(e.grabbed < 0);
  // Later it does again, and a press always does.
  e = runFor(c, aiming(4), t.regrabSeconds);
  CHECK(e.grabbed == 4);
  PowerController d;
  d.update(cast, kDt, t);
  runFor(d, aiming(-1), t.burstRecover + kDt);
  d.update(PowerInput{}, kDt, t);
  e = d.update(aiming(3), kDt, t);
  CHECK(e.grabbed == 3);
}
