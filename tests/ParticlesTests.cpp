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
