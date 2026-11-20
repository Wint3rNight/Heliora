#include "Rocket.h"
#include "TestRunner.h"

#include <cmath>

namespace {

const glm::vec3 kGravity(0.0f, -9.81f, 0.0f);

// Flies a rocket fired level along +z until it's past `range` along z (or
// its time is up); where it crossed that plane.
struct Crossing {
  bool reached = false;
  glm::vec3 at{0.0f};
  float time = 0.0f;
};
Crossing flyTo(const RocketSpec &spec, const RocketFit &fit, float range, glm::vec3 wind,
               glm::vec3 direction = {0.0f, 0.0f, 1.0f}, float dt = 1.0f / 60.0f) {
  RocketFlight f = launchRocket(glm::vec3(0.0f), direction, spec);
  while (!rocketSpent(f, spec)) {
    const glm::vec3 before = f.position;
    const float then = f.time;
    stepRocket(f, spec, fit, dt, wind, kGravity);
    if (f.position.z >= range) {
      const float t = (range - before.z) / (f.position.z - before.z);
      return {true, glm::mix(before, f.position, t), then + t * (f.time - then)};
    }
  }
  return {};
}

} // namespace

TEST(the_pg7v_is_fitted_to_its_documented_speeds_and_reach) {
  // Wikipedia's RPG-7: 115 m/s off the launcher, the motor lit at 10 m and
  // burning out to 500 m at 295 m/s; the fuze's usual 920 m at 4.5 s.
  const RocketSpec spec;
  const RocketFit fit = fitRocket(spec);
  CHECK(fit.thrust > 300.0f && fit.thrust < 700.0f);
  CHECK(fit.dragCoefficient > 0.8f && fit.dragCoefficient < 2.0f);
  RocketFlight f = launchRocket(glm::vec3(0.0f), glm::vec3(0, 0, 1), spec);
  CHECK_NEAR(glm::length(f.velocity), 115.0, 1e-3);
  float topSpeed = 0.0f, burnoutAt = -1.0f, litAt = -1.0f;
  while (!rocketSpent(f, spec)) {
    stepRocket(f, spec, fit, 1.0f / 60.0f, glm::vec3(0.0f), glm::vec3(0.0f));
    if (f.motorOn && litAt < 0.0f)
      litAt = f.flown;
    if (f.burntOut && burnoutAt < 0.0f)
      burnoutAt = f.flown;
    topSpeed = std::max(topSpeed, glm::length(f.velocity));
  }
  CHECK_NEAR(litAt, 10.0, 3.0); // the first frame past 10 m
  CHECK_NEAR(burnoutAt, 500.0, 6.0);
  CHECK_NEAR(topSpeed, 295.0, 2.0);
  CHECK_NEAR(f.time, 4.5, 1e-3);
  CHECK_NEAR(f.flown, 920.0, 8.0);
}

TEST(a_rocket_flies_the_same_at_any_frame_rate) {
  const RocketSpec spec;
  const RocketFit fit = fitRocket(spec);
  const Crossing slow = flyTo(spec, fit, 300.0f, glm::vec3(3.0f, 0.0f, 0.0f), {0, 0, 1}, 1.0f / 30.0f);
  const Crossing fast = flyTo(spec, fit, 300.0f, glm::vec3(3.0f, 0.0f, 0.0f), {0, 0, 1}, 1.0f / 240.0f);
  CHECK(slow.reached && fast.reached);
  CHECK(glm::distance(slow.at, fast.at) < 0.05f);
  CHECK_NEAR(slow.time, fast.time, 0.005);
}

TEST(a_crosswind_turns_the_burning_rocket_into_it_half_missing_a_tank_at_180_m) {
  // Wikipedia: "in an 11 km/h wind, firing at a stationary tank sized
  // target, the gunner cannot expect to get a first-round hit more than
  // 50% of the time at 180 m". The fins keep its nose along the air it
  // flies through, so the motor pushes it into the wind: by 180 m it's
  // about half a tank's front (3.6 m wide) upwind of the aim. A check,
  // nothing here is fitted to it.
  const RocketSpec spec;
  const RocketFit fit = fitRocket(spec);
  const glm::vec3 wind(11.0f / 3.6f, 0.0f, 0.0f); // blowing toward +x
  const Crossing c = flyTo(spec, fit, 180.0f, wind);
  CHECK(c.reached);
  CHECK(c.at.x < 0.0f); // upwind
  CHECK_NEAR(-c.at.x, 1.8, 0.4);
  // Still air: straight on.
  const Crossing still = flyTo(spec, fit, 180.0f, glm::vec3(0.0f));
  CHECK(std::abs(still.at.x) < 1e-3f);
}

TEST(the_launch_is_raised_so_the_rocket_drops_onto_the_aim_point) {
  const RocketSpec spec;
  const RocketFit fit = fitRocket(spec);
  for (const float range : {50.0f, 200.0f, 500.0f, 800.0f}) {
    const float lift = rocketLift(spec, fit, range, kGravity);
    CHECK(lift > 0.0f && lift < 0.1f);
    const glm::vec3 dir(0.0f, std::sin(lift), std::cos(lift));
    const Crossing c = flyTo(spec, fit, range, glm::vec3(0.0f), dir, 1.0f / 120.0f);
    CHECK(c.reached);
    CHECK(std::abs(c.at.y) < 0.1f);
  }
  // Further than it flies: its highest useful lift, not a guess past it.
  CHECK(rocketLift(spec, fit, 5000.0f, kGravity) <= rocketLift(spec, fit, 920.0f, kGravity) + 1e-4f);
}

TEST(a_rocket_is_spent_at_its_self_destruct_time) {
  const RocketSpec spec;
  const RocketFit fit = fitRocket(spec);
  RocketFlight f = launchRocket(glm::vec3(0.0f), glm::vec3(0, 0, 1), spec);
  stepRocket(f, spec, fit, 4.4f, glm::vec3(0.0f), kGravity);
  CHECK(!rocketSpent(f, spec));
  stepRocket(f, spec, fit, 0.5f, glm::vec3(0.0f), kGravity);
  CHECK(rocketSpent(f, spec));
  CHECK_NEAR(f.time, 4.5, 1e-4); // it stops there, not a frame later
}

TEST(a_heat_jet_goes_deeper_into_lighter_things_by_birkhoffs_rule) {
  // 260 mm of steel (the PG-7V); elsewhere by the square root of steel's
  // density over the target's.
  CHECK_NEAR(jetDepth(0.26f, kSteelDensity), 0.26, 1e-6);
  CHECK_NEAR(jetDepth(0.26f, 2400.0f), 0.26 * std::sqrt(7850.0 / 2400.0), 1e-4); // stone: 0.47 m
  CHECK_NEAR(jetDepth(0.26f, 600.0f), 0.26 * std::sqrt(7850.0 / 600.0), 1e-4);   // wood: 0.94 m
  // Through 0.3 m of stone it has the rest of its steel's worth left.
  const float used = jetUsed(0.3f, 2400.0f);
  CHECK_NEAR(used, 0.3 * std::sqrt(2400.0 / 7850.0), 1e-4);
  CHECK_NEAR(jetDepth(0.26f - used, 2400.0f), jetDepth(0.26f, 2400.0f) - 0.3f, 1e-4);
  // Every material the bullets know has a density.
  for (int m = 0; m < static_cast<int>(Matter::Count); ++m)
    CHECK(matterDensity(static_cast<Matter>(m)) > 500.0f);
  CHECK_NEAR(matterDensity(Matter::Metal), 7800.0, 1.0);
}
