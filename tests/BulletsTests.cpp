#include "Bullets.h"
#include "TestRunner.h"

#include <cmath>

namespace {

constexpr float kDt = 1.0f / 60.0f;
const glm::vec3 kGravity(0.0f, -9.81f, 0.0f);
constexpr float kMass = 0.0079f; // the AKM's ball bullet, kg

// The AKM's speeds at 150 and 500 m (G7 0.15 from 715 m/s), where NSD p.158
// puts its depth in stacked dry pine at 30-40 cm and 25 cm.
constexpr float kAt150 = 580.1f, kAt500 = 325.8f;

Round akm() {
  Round r;
  r.projectile = {DragModel::G7, 0.15f};
  r.mass = kMass;
  r.resistance[static_cast<size_t>(Matter::Wood)] =
      ponceletFit(kMass, 0.35f, kAt150, 0.25f, kAt500);
  r.resistance[static_cast<size_t>(Matter::Stone)] =
      ponceletFrom(kMass, 0.135f, 623.0f, ratioOf(r.resistance[size_t(Matter::Wood)]));
  return r;
}

// Walls across +x: each at its x, this thick, of this matter.
struct Slab {
  float x, thickness;
  Matter matter;
  uint64_t body;
};

BulletWorld slabs(std::vector<Slab> walls) {
  return [walls](glm::vec3 from, glm::vec3 to) -> std::optional<BulletContact> {
    const glm::vec3 d = to - from;
    if (d.x <= 0.0f)
      return std::nullopt;
    const Slab *nearest = nullptr;
    for (const Slab &w : walls)
      if (w.x > from.x && w.x <= to.x && (!nearest || w.x < nearest->x))
        nearest = &w;
    if (!nearest)
      return std::nullopt;
    const float u = (nearest->x - from.x) / d.x;
    BulletContact c;
    c.distance = u * glm::length(d);
    c.point = from + d * u;
    c.normal = glm::vec3(-1.0f, 0.0f, 0.0f);
    c.body = nearest->body;
    c.matter = nearest->matter;
    c.thickness = nearest->thickness * glm::length(d) / d.x;
    return c;
  };
}

Bullet fired(const Round &round, float speed = 715.0f) {
  Bullet b;
  b.round = &round;
  b.flight.position = glm::vec3(0.0f, 1.0f, 0.0f);
  b.flight.velocity = glm::vec3(speed, 0.0f, 0.0f);
  return b;
}

void runFor(std::vector<Bullet> &bullets, const BulletWorld &world, float seconds,
            std::vector<BulletImpact> &impacts) {
  for (int i = 0; i < int(std::lround(seconds / kDt)); ++i)
    stepBullets(bullets, kDt, Air{}, kGravity, world, impacts);
}

} // namespace

TEST(poncelet_fitted_to_two_depths_reaches_both) {
  const Resistance pine = ponceletFit(kMass, 0.35f, kAt150, 0.25f, kAt500);
  CHECK_NEAR(stoppingDepth(pine, kMass, kAt150), 0.35, 1e-3);
  CHECK_NEAR(stoppingDepth(pine, kMass, kAt500), 0.25, 1e-3);
  CHECK_NEAR(ratioOf(pine), 1.417e-4, 0.01e-4);
  // From the muzzle a little deeper, still inside the manual's 30-40 cm.
  CHECK(stoppingDepth(pine, kMass, 715.0f) < 0.40f);
}

TEST(without_its_speed_term_poncelet_is_the_square_root_rule) {
  // A constant force: through t of a material it stops in t_max, a bullet
  // keeps v * sqrt(1 - t / t_max), the design's first rule.
  const Resistance r = ponceletFrom(kMass, 0.3f, 700.0f, 0.0f);
  CHECK_NEAR(r.b, 0.0, 1e-9);
  CHECK_NEAR(stoppingDepth(r, kMass, 700.0f), 0.3, 1e-4);
  CHECK_NEAR(exitSpeed(r, kMass, 700.0f, 0.1f), 700.0 * std::sqrt(1.0 - 0.1 / 0.3), 0.05);
  CHECK_NEAR(exitSpeed(r, kMass, 700.0f, 0.3f), 0.0, 1e-3);
}

TEST(two_boards_slow_a_bullet_as_much_as_one_of_both_thicknesses) {
  const Resistance pine = ponceletFit(kMass, 0.35f, kAt150, 0.25f, kAt500);
  const float once = exitSpeed(pine, kMass, 650.0f, 0.15f);
  const float twice = exitSpeed(pine, kMass, exitSpeed(pine, kMass, 650.0f, 0.05f), 0.10f);
  CHECK_NEAR(once, twice, 0.1);
  CHECK(once > 0.0f && once < 650.0f);
  // Thicker than it reaches: stopped.
  CHECK_NEAR(exitSpeed(pine, kMass, 650.0f, 0.5f), 0.0, 1e-6);
  // Nothing in the way, or a material that doesn't resist: no change.
  CHECK_NEAR(exitSpeed(pine, kMass, 650.0f, 0.0f), 650.0, 1e-3);
  CHECK_NEAR(exitSpeed(Resistance{}, kMass, 650.0f, 1.0f), 650.0, 1e-3);
}

TEST(a_bullet_in_the_open_flies_until_three_seconds_are_up) {
  const Round round = akm();
  std::vector<Bullet> bullets = {fired(round)};
  std::vector<BulletImpact> impacts;
  const BulletWorld open = slabs({});
  runFor(bullets, open, 2.9f, impacts);
  CHECK(bullets.size() == 1);
  CHECK(bullets[0].flight.position.x > 1000.0f);
  CHECK(bullets[0].flight.position.y < 1.0f); // it falls
  runFor(bullets, open, 0.2f, impacts);
  CHECK(bullets.empty());
  CHECK(impacts.empty());
  // One already under 50 m/s is spent at once.
  bullets = {fired(round, 40.0f)};
  runFor(bullets, open, kDt, impacts);
  CHECK(bullets.empty());
}

TEST(a_wall_deeper_than_its_reach_stops_it_and_takes_all_its_energy) {
  const Round round = akm();
  std::vector<Bullet> bullets = {fired(round)};
  std::vector<BulletImpact> impacts;
  runFor(bullets, slabs({{20.0f, 1.0f, Matter::Stone, 7}}), 0.1f, impacts);
  CHECK(bullets.empty());
  CHECK(impacts.size() == 1);
  if (impacts.empty())
    return;
  const BulletImpact &hit = impacts[0];
  CHECK(hit.stopped);
  CHECK(hit.body == 7);
  CHECK(hit.matter == Matter::Stone);
  CHECK_NEAR(hit.point.x, 20.0, 1e-3);
  CHECK(hit.speedIn > 690.0f && hit.speedIn < 715.0f);
  CHECK_NEAR(hit.speedOut, 0.0, 1e-6);
  CHECK_NEAR(hit.energy(), 0.5 * kMass * hit.speedIn * hit.speedIn, 1e-2);
  CHECK_NEAR(hit.impulse(), kMass * hit.speedIn, 1e-4);
  CHECK(glm::distance(hit.direction, glm::vec3(1.0f, 0.0f, 0.0f)) < 1e-3f);
}

TEST(a_board_slows_a_bullet_and_it_carries_on) {
  const Round round = akm();
  std::vector<Bullet> bullets = {fired(round)};
  std::vector<BulletImpact> impacts;
  runFor(bullets, slabs({{20.0f, 0.05f, Matter::Wood, 3}}), 0.1f, impacts);
  CHECK(impacts.size() == 1);
  if (impacts.empty())
    return;
  const BulletImpact &hit = impacts[0];
  CHECK(!hit.stopped);
  CHECK_NEAR(hit.speedOut,
             exitSpeed(round.resistance[size_t(Matter::Wood)], kMass, hit.speedIn, 0.05f), 0.5);
  CHECK(hit.speedOut < hit.speedIn - 50.0f);
  CHECK_NEAR(hit.energy(), 0.5 * kMass * (hit.speedIn * hit.speedIn - hit.speedOut * hit.speedOut),
             1e-2);
  CHECK(bullets.size() == 1);
  CHECK(bullets[0].flight.position.x > 40.0f);
  CHECK(glm::length(bullets[0].flight.velocity) < hit.speedOut);
}

TEST(two_boards_inside_one_sub_step_are_both_hit) {
  // At 715 m/s a 1/240 s sub-step covers 3 m; boards 1 m apart.
  const Round round = akm();
  std::vector<Bullet> bullets = {fired(round)};
  bullets[0].flight.position.x = 19.0f;
  std::vector<BulletImpact> impacts;
  stepBullets(bullets, 1.0f / 240.0f, Air{}, kGravity,
              slabs({{20.0f, 0.02f, Matter::Wood, 1}, {21.0f, 0.02f, Matter::Wood, 2}}), impacts);
  CHECK(impacts.size() == 2);
  if (impacts.size() == 2) {
    CHECK(impacts[0].body == 1 && impacts[1].body == 2);
    CHECK_NEAR(impacts[1].speedIn, impacts[0].speedOut, 2.0);
  }
}

TEST(a_bullet_slowed_under_50_m_s_is_spent_where_it_came_out) {
  // Just short of the board's full stopping depth: it would crawl out at a
  // few m/s, so it's spent there, and everything it had went into the board.
  const Round round = akm();
  const Resistance &pine = round.resistance[size_t(Matter::Wood)];
  const float depth = stoppingDepth(pine, kMass, 715.0f) - 0.01f;
  std::vector<Bullet> bullets = {fired(round)};
  bullets[0].flight.position.x = 19.9f;
  std::vector<BulletImpact> impacts;
  runFor(bullets, slabs({{20.0f, depth, Matter::Wood, 1}}), kDt, impacts);
  CHECK(impacts.size() == 1);
  if (impacts.empty())
    return;
  CHECK(impacts[0].stopped);
  CHECK_NEAR(impacts[0].speedOut, 0.0, 1e-6);
  CHECK(bullets.empty());
}
