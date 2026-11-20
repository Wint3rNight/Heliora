#include "Particles.h"
#include "TestRunner.h"

#include <cmath>

namespace {

float offPath(glm::vec3 p, const std::vector<glm::vec3> &path) {
  float best = 1e9f;
  for (size_t i = 0; i + 1 < path.size(); ++i) {
    const glm::vec3 d = path[i + 1] - path[i];
    const float t = glm::clamp(glm::dot(p - path[i], d) / glm::dot(d, d), 0.0f, 1.0f);
    best = std::min(best, glm::distance(p, path[i] + d * t));
  }
  return best;
}

} // namespace

TEST(the_particle_random_is_the_same_from_the_same_seed_and_stays_in_range) {
  ParticleRandom a(7), b(7), c(8);
  bool differs = false;
  for (int i = 0; i < 1000; ++i) {
    const float x = a.next();
    CHECK(x >= 0.0f && x < 1.0f);
    CHECK(x == b.next());
    differs = differs || x != c.next();
  }
  CHECK(differs);
  for (int i = 0; i < 100; ++i)
    CHECK_NEAR(glm::length(a.onSphere()), 1.0, 1e-4);
}

TEST(spawning_at_a_rate_carries_the_fraction_over) {
  float carry = 0.0f;
  int total = 0;
  for (int i = 0; i < 60; ++i)
    total += spawnCount(10.0f, 1.0f / 60.0f, carry); // a second at 60 frames
  CHECK(total == 10 || total == 9); // 10 in a second, give or take the last fraction
  CHECK(carry >= 0.0f && carry < 1.0f);
  float none = 0.0f;
  CHECK(spawnCount(0.0f, 1.0f, none) == 0);
}

TEST(particles_along_a_path_start_on_it_moving_along_it) {
  std::vector<ParticleSpawn> out;
  ParticleRandom random(3);
  const std::vector<glm::vec3> path = {{0, 0, 0}, {2, 0, 0}, {2, 0, -2}};
  ParticleSpawn kind;
  kind.life = 0.5f;
  emitAlongPath(out, path, 50, 3.0f, kind, random);
  CHECK(out.size() == 50);
  for (const ParticleSpawn &p : out) {
    CHECK(offPath(p.position, path) < 1e-4f);
    CHECK_NEAR(glm::length(p.velocity), 3.0, 1e-4);
    // Along the leg it's on, toward the end: +x on the first, -z on the second.
    const bool first = std::fabs(p.position.z) < 1e-4f && p.position.x < 2.0f - 1e-4f;
    CHECK(first ? p.velocity.x > 2.99f : p.velocity.z < -2.99f);
    CHECK(p.life == 0.5f);
  }
}

TEST(particles_from_a_box_start_in_it_moving_out) {
  std::vector<ParticleSpawn> out;
  ParticleRandom random(5);
  emitFromBox(out, {-1, 0, -1}, {1, 2, 1}, 40, 1.5f, ParticleSpawn{}, random);
  CHECK(out.size() == 40);
  for (const ParticleSpawn &p : out) {
    CHECK(p.position.x >= -1.0f && p.position.x <= 1.0f && p.position.y >= 0.0f &&
          p.position.y <= 2.0f && p.position.z >= -1.0f && p.position.z <= 1.0f);
    CHECK_NEAR(glm::length(p.velocity), 1.5, 1e-4);
    CHECK(glm::dot(p.velocity, p.position - glm::vec3(0, 1, 0)) >= 0.0f); // away from the middle
  }
}

TEST(particles_in_a_cone_leave_its_apex_within_its_angle) {
  std::vector<ParticleSpawn> out;
  ParticleRandom random(11);
  const glm::vec3 apex(0, 1, 0), axis(0, 0, -1);
  const float half = glm::radians(35.0f);
  emitInCone(out, apex, axis, half, 200, 6.0f, 16.0f, ParticleSpawn{}, random);
  CHECK(out.size() == 200);
  float widest = 0.0f;
  for (const ParticleSpawn &p : out) {
    CHECK(glm::distance(p.position, apex) < 1e-5f);
    const float speed = glm::length(p.velocity);
    CHECK(speed >= 6.0f - 1e-4f && speed <= 16.0f + 1e-4f);
    const float angle = std::acos(glm::clamp(glm::dot(p.velocity / speed, axis), -1.0f, 1.0f));
    CHECK(angle <= half + 1e-4f);
    widest = std::max(widest, angle);
  }
  CHECK(widest > 0.7f * half); // spread across it, not all down the middle
}

TEST(a_trail_lays_a_puff_every_so_far_whatever_the_frame_rate) {
  // 10.1 m flown in one frame or in 37 uneven ones: the same puffs, every
  // 0.25 m along the path, the distance since the last carried over.
  ParticleSpawn kind;
  kind.velocity = glm::vec3(0.0f, 0.3f, 0.0f);
  std::vector<ParticleSpawn> once, often;
  float carry = 0.0f;
  emitTrail(once, glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 10.1f), 0.25f, carry, kind);
  CHECK(once.size() == 40);
  CHECK_NEAR(carry, 0.1, 1e-4);
  carry = 0.0f;
  float z = 0.0f;
  for (int k = 0; k < 37; ++k) {
    const float next = k == 36 ? 10.1f : z + 10.0f / 37.0f * (0.5f + static_cast<float>(k % 3) * 0.5f);
    emitTrail(often, glm::vec3(0.0f, 0.0f, z), glm::vec3(0.0f, 0.0f, std::min(next, 10.1f)), 0.25f, carry,
              kind);
    z = std::min(next, 10.1f);
  }
  CHECK(often.size() == once.size());
  // Each on the path at a multiple of the spacing, moving as asked.
  for (const ParticleSpawn &p : often) {
    const float steps = p.position.z / 0.25f;
    CHECK_NEAR(steps, std::round(steps), 1e-3);
    CHECK(p.velocity == kind.velocity);
  }
  // Nothing for standing still.
  std::vector<ParticleSpawn> none;
  emitTrail(none, glm::vec3(1.0f), glm::vec3(1.0f), 0.25f, carry, kind);
  CHECK(none.empty());
}

TEST(smoke_thins_as_it_spreads_by_beer_lambert) {
  // A puff keeps its smoke as it spreads: its optical depth goes as one over
  // its area, so twice as wide lets (1 - cover) through to the 1/4 power.
  CHECK_NEAR(smokeCover(0.5f, 1.0f), 0.5, 1e-6);
  CHECK_NEAR(smokeCover(0.5f, 2.0f), 1.0 - std::pow(0.5, 0.25), 1e-5);
  const float depthThen = -std::log(1.0f - 0.6f);
  const float depthNow = -std::log(1.0f - smokeCover(0.6f, 3.0f));
  CHECK_NEAR(depthNow * 9.0f, depthThen, 1e-4);
  // Never thicker than it started, and a solid puff still thins.
  CHECK_NEAR(smokeCover(0.5f, 0.5f), 0.5, 1e-6);
  CHECK(smokeCover(1.0f, 2.0f) < 1.0f);
}

TEST(a_particle_grows_or_shrinks_to_its_spread_over_its_life) {
  ParticleSpawn smoke;
  smoke.size = 0.2f;
  smoke.spread = 6.0f;
  CHECK_NEAR(particleSizeAt(smoke, 0.0f), 0.2, 1e-6);
  CHECK_NEAR(particleSizeAt(smoke, 1.0f), 1.2, 1e-5);
  // Fast at first, slower later, as a puff mixes with the air.
  CHECK(particleSizeAt(smoke, 0.5f) - 0.2f > 0.5f * (1.2f - 0.2f));
  ParticleSpawn spark;
  spark.size = 0.1f; // the default spread shrinks it
  CHECK(particleSizeAt(spark, 1.0f) < 0.1f);
}

TEST(drag_slows_a_particle_to_the_air_not_to_a_stop) {
  // Smoke drifts with the wind: drag pulls it toward the air's speed.
  const glm::vec3 wind(4.0f, 0.0f, 0.0f);
  glm::vec3 v(0.0f, 10.0f, 0.0f);
  for (int k = 0; k < 600; ++k)
    v = airDrag(v, wind, 3.0f, 1.0f / 60.0f);
  CHECK(glm::distance(v, wind) < 1e-3f);
  // Stepped in one go or in pieces, the same.
  const glm::vec3 a = airDrag(glm::vec3(0.0f, 10.0f, 0.0f), wind, 3.0f, 0.1f);
  const glm::vec3 b = airDrag(airDrag(glm::vec3(0.0f, 10.0f, 0.0f), wind, 3.0f, 0.05f), wind, 3.0f, 0.05f);
  CHECK(glm::distance(a, b) < 1e-5f);
}

TEST(the_ring_draws_only_the_slots_that_can_still_be_alive) {
  // Each frame's new particles go in the ring's next slots, so the living
  // ones are the last few frames' worth: the oldest frame still alive on.
  LiveRing ring(1000);
  CHECK(ring.live() == 0);
  ring.step(0.1f, 100, 1.0f);  // 100 lasting up to 1 s
  ring.step(0.1f, 50, 0.3f);   // 50 lasting up to 0.3 s
  CHECK(ring.head() == 150);
  CHECK(ring.first() == 0 && ring.live() == 150);
  for (int k = 0; k < 6; ++k)
    ring.step(0.1f, 0, 0.0f);  // 0.7 s after the first frame: it lives on
  CHECK(ring.live() == 150);
  for (int k = 0; k < 4; ++k)
    ring.step(0.1f, 0, 0.0f);  // past it: all gone
  CHECK(ring.live() == 0);
  // Round the end: the range wraps, and never counts more than the ring.
  ring.step(0.1f, 900, 2.0f);
  ring.step(0.1f, 300, 2.0f);
  CHECK(ring.head() == (150 + 1200) % 1000);
  CHECK(ring.live() == 1000);
  CHECK(ring.first() == ring.head());
  ring.clear();
  CHECK(ring.live() == 0 && ring.head() == 0);
}
