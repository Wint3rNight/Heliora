#include "CharacterRoster.h"
#include "CharacterTestScenes.h"
#include "TestRunner.h"

#include <memory>

namespace {

using test::addFloor;

constexpr float kStep = 1.0f / 60.0f;

// Walks its body along -z, every step. The simplest brain there is.
class Walker : public Controller {
public:
  CharacterInput decide() override {
    CharacterInput input;
    input.move = glm::vec3(0.0f, 0.0f, -1.0f);
    return input;
  }
};

size_t addBody(CharacterRoster &roster, PhysicsWorld &world, glm::vec3 feet) {
  return roster.add(std::make_unique<CharacterMovement>(world, feet, 0.0f));
}

void steps(CharacterRoster &roster, PhysicsWorld &world, int count) {
  for (int i = 0; i < count; ++i) {
    roster.step(kStep);
    world.step(kStep);
  }
}

} // namespace

TEST(a_body_with_no_controller_stands_still) {
  PhysicsWorld world;
  addFloor(world);
  CharacterRoster roster;
  const size_t body = addBody(roster, world, glm::vec3(1.0f, 0.0f, 2.0f));
  steps(roster, world, 60);
  CHECK(roster.controllerOf(body) == nullptr);
  CHECK(roster.body(body).grounded());
  CHECK_NEAR(glm::distance(roster.body(body).position(), glm::vec3(1.0f, 0.0f, 2.0f)),
             0.0, 0.01);
}

TEST(a_body_with_no_controller_still_falls) {
  PhysicsWorld world;
  addFloor(world);
  CharacterRoster roster;
  const size_t body = addBody(roster, world, glm::vec3(0.0f, 2.0f, 0.0f));
  steps(roster, world, 90);
  CHECK(roster.body(body).grounded());
  CHECK(roster.body(body).position().y < 0.05f);
}

TEST(a_controller_moves_only_the_body_it_possesses) {
  PhysicsWorld world;
  addFloor(world);
  CharacterRoster roster;
  const size_t first = addBody(roster, world, glm::vec3(-3.0f, 0.0f, 0.0f));
  const size_t second = addBody(roster, world, glm::vec3(3.0f, 0.0f, 0.0f));
  Walker walker;
  roster.possess(walker, first);
  CHECK(walker.body() == &roster.body(first));
  CHECK(roster.controllerOf(first) == &walker);
  steps(roster, world, 60);
  CHECK(roster.body(first).position().z < -1.0f);
  CHECK_NEAR(roster.body(second).position().z, 0.0, 0.01);
}

TEST(possessing_another_body_lets_go_of_the_first) {
  PhysicsWorld world;
  addFloor(world);
  CharacterRoster roster;
  const size_t first = addBody(roster, world, glm::vec3(-3.0f, 0.0f, 0.0f));
  const size_t second = addBody(roster, world, glm::vec3(3.0f, 0.0f, 0.0f));
  Walker walker;
  roster.possess(walker, first);
  steps(roster, world, 60);
  roster.possess(walker, second);
  CHECK(roster.controllerOf(first) == nullptr);
  CHECK(walker.body() == &roster.body(second));
  steps(roster, world, 120); // long enough for the first to brake to a stop
  const float stoppedAt = roster.body(first).position().z;
  steps(roster, world, 30);
  CHECK_NEAR(roster.body(first).position().z, stoppedAt, 1e-3);
  CHECK(roster.body(second).position().z < -1.0f);
}

TEST(a_second_controller_takes_the_body_from_the_first) {
  PhysicsWorld world;
  addFloor(world);
  CharacterRoster roster;
  const size_t body = addBody(roster, world, glm::vec3(0.0f));
  Walker one, two;
  roster.possess(one, body);
  roster.possess(two, body);
  CHECK(one.body() == nullptr);
  CHECK(two.body() == &roster.body(body));
  CHECK(roster.controllerOf(body) == &two);
}

TEST(releasing_leaves_the_body_standing) {
  PhysicsWorld world;
  addFloor(world);
  CharacterRoster roster;
  const size_t body = addBody(roster, world, glm::vec3(0.0f));
  Walker walker;
  roster.possess(walker, body);
  roster.release(walker);
  CHECK(walker.body() == nullptr);
  CHECK(roster.controllerOf(body) == nullptr);
  steps(roster, world, 60);
  CHECK_NEAR(roster.body(body).position().z, 0.0, 0.01);
}

TEST(clearing_the_roster_lets_go_of_its_controllers) {
  PhysicsWorld world;
  addFloor(world);
  CharacterRoster roster;
  const size_t body = addBody(roster, world, glm::vec3(0.0f));
  Walker walker;
  roster.possess(walker, body);
  roster.clear();
  CHECK(roster.size() == 0);
  CHECK(walker.body() == nullptr);
}

TEST(a_walking_body_bumps_into_a_standing_one) {
  // Bodies nobody plays are still in the world: walking into one stops you
  // at its edge rather than passing through.
  PhysicsWorld world;
  addFloor(world);
  CharacterRoster roster;
  const size_t walking = addBody(roster, world, glm::vec3(0.0f));
  const size_t standing = addBody(roster, world, glm::vec3(0.0f, 0.0f, -3.0f));
  Walker walker;
  roster.possess(walker, walking);
  steps(roster, world, 180);
  CHECK(roster.body(walking).position().z > -3.0f + 0.5f); // two radii away, about
  CHECK(roster.body(walking).position().z < -1.5f);        // but it did walk up to it
  CHECK_NEAR(roster.body(standing).position().z, -3.0, 0.05);
}
