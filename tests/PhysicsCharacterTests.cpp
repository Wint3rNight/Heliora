#include "PhysicsWorld.h"
#include "TestRunner.h"

#include <glm/gtc/quaternion.hpp>

#include <cmath>
#include <optional>

namespace {

constexpr float kStep = 1.0f / 60.0f;
constexpr float kGravity = -9.81f;

PhysicsBodyId addStaticBox(PhysicsWorld &world, glm::vec3 position,
                           glm::vec3 halfExtents,
                           glm::quat rotation = glm::quat(1, 0, 0, 0)) {
  PhysicsBodyDesc box;
  box.shape = PhysicsShape::Box;
  box.halfExtents = halfExtents;
  box.motion = PhysicsMotion::Static;
  box.position = position;
  box.rotation = rotation;
  return world.addBody(box);
}

// A static floor whose top surface is at y = 0.
void addFloor(PhysicsWorld &world) {
  addStaticBox(world, glm::vec3(0.0f, -0.5f, 0.0f), glm::vec3(50.0f, 0.5f, 50.0f));
}

// A 8 m long, 20 cm thick ramp rising along +x at the angle. Its top surface
// meets the floor at about x = 4 - 0.1 / sin(angle).
void addRamp(PhysicsWorld &world, float degrees) {
  addStaticBox(world, glm::vec3(4.0f, 0.0f, 0.0f), glm::vec3(4.0f, 0.1f, 2.0f),
               glm::angleAxis(glm::radians(degrees), glm::vec3(0, 0, 1)));
}

float rampSurfaceY(float x, float degrees) {
  const float a = glm::radians(degrees);
  return (x - (4.0f - 0.1f / std::sin(a))) * std::tan(a);
}

PhysicsCharacterId addCharacter(PhysicsWorld &world, glm::vec3 feet) {
  PhysicsCharacterDesc desc;
  desc.position = feet;
  return world.addCharacter(desc);
}

// One step, driven the way a controller would: on walkable ground it keeps
// the ground's velocity, in the air it keeps its fall, and gravity pulls
// every step. horizontal is the walking velocity.
void stepCharacter(PhysicsWorld &world, PhysicsCharacterId character,
                   glm::vec3 horizontal, PhysicsCharacterMove move = {}) {
  const PhysicsCharacterState state = world.characterState(character);
  glm::vec3 velocity = state.ground == PhysicsGround::OnGround
                           ? state.groundVelocity
                           : glm::vec3(0.0f, state.velocity.y, 0.0f);
  velocity += glm::vec3(0.0f, kGravity * kStep, 0.0f) + horizontal;
  move.velocity = velocity;
  world.moveCharacter(character, move, kStep);
  world.step(kStep);
}

void walk(PhysicsWorld &world, PhysicsCharacterId character,
          glm::vec3 horizontal, int steps, PhysicsCharacterMove move = {}) {
  for (int i = 0; i < steps; ++i)
    stepCharacter(world, character, horizontal, move);
}

} // namespace

TEST(character_falls_onto_the_floor_and_stands) {
  PhysicsWorld world;
  addFloor(world);
  const PhysicsCharacterId character = addCharacter(world, glm::vec3(0, 2, 0));
  CHECK(character.isValid());
  walk(world, character, glm::vec3(0.0f), 120);
  const PhysicsCharacterState state = world.characterState(character);
  CHECK(state.ground == PhysicsGround::OnGround);
  CHECK_NEAR(state.position.y, 0.0, 0.03);
  CHECK_NEAR(state.height, 1.8, 1e-4);
}

TEST(character_walks_at_its_velocity) {
  PhysicsWorld world;
  addFloor(world);
  const PhysicsCharacterId character = addCharacter(world, glm::vec3(0.0f));
  // Spawned on the floor, it stands from the start.
  CHECK(world.characterState(character).ground == PhysicsGround::OnGround);
  walk(world, character, glm::vec3(3.0f, 0.0f, 0.0f), 60);
  const PhysicsCharacterState state = world.characterState(character);
  CHECK_NEAR(state.position.x, 3.0, 0.1);
  CHECK_NEAR(state.position.y, 0.0, 0.03);
}

TEST(wall_stops_the_character) {
  PhysicsWorld world;
  addFloor(world);
  addStaticBox(world, glm::vec3(3.0f, 2.0f, 0.0f), glm::vec3(0.5f, 2.0f, 5.0f));
  const PhysicsCharacterId character = addCharacter(world, glm::vec3(0.0f));
  walk(world, character, glm::vec3(5.0f, 0.0f, 0.0f), 120);
  const PhysicsCharacterState state = world.characterState(character);
  // The wall's face is at x = 2.5 and the capsule is 0.3 m wide each way.
  CHECK(state.position.x < 2.2f + 0.05f);
  CHECK(state.position.x > 2.2f - 0.1f);
  CHECK(std::fabs(state.velocity.x) < 0.5f); // the wall took it away
}

namespace {

// Walks at a step of the height from x = 0 and returns the height it ends
// at. The step runs from x = 1.5 to 3.5.
float heightAfterWalkingAtStep(float stepHeight, float stepUp) {
  PhysicsWorld world;
  addFloor(world);
  addStaticBox(world, glm::vec3(2.5f, 0.5f * stepHeight, 0.0f),
               glm::vec3(1.0f, 0.5f * stepHeight, 2.0f));
  const PhysicsCharacterId character = addCharacter(world, glm::vec3(0.0f));
  PhysicsCharacterMove move;
  move.stepUp = stepUp;
  walk(world, character, glm::vec3(3.0f, 0.0f, 0.0f), 50, move);
  return world.characterState(character).position.y;
}

} // namespace

TEST(step_up_decides_which_steps_are_climbed) {
  // Measured: the highest step climbed is stepUp + 10 cm.
  const float defaultStepUp = PhysicsCharacterMove{}.stepUp;
  CHECK_NEAR(heightAfterWalkingAtStep(0.35f, defaultStepUp), 0.35, 0.03);
  CHECK_NEAR(heightAfterWalkingAtStep(0.5f, defaultStepUp), 0.0, 0.03);
  CHECK_NEAR(heightAfterWalkingAtStep(0.35f, 0.2f), 0.0, 0.03);
  CHECK_NEAR(heightAfterWalkingAtStep(0.25f, 0.0f), 0.0, 0.03); // off
}

TEST(walkable_slope_is_climbed) {
  PhysicsWorld world;
  addFloor(world);
  addRamp(world, 30.0f);
  const PhysicsCharacterId character = addCharacter(world, glm::vec3(0.0f));
  walk(world, character, glm::vec3(3.0f, 0.0f, 0.0f), 120);
  const PhysicsCharacterState state = world.characterState(character);
  CHECK(state.position.x > 5.0f);
  CHECK_NEAR(state.position.y, rampSurfaceY(state.position.x, 30.0f), 0.1);
  CHECK(state.ground == PhysicsGround::OnGround);
}

TEST(too_steep_slope_is_not_climbed) {
  PhysicsWorld world;
  addFloor(world);
  addRamp(world, 60.0f);
  const PhysicsCharacterId character = addCharacter(world, glm::vec3(0.0f));
  walk(world, character, glm::vec3(3.0f, 0.0f, 0.0f), 180);
  const PhysicsCharacterState state = world.characterState(character);
  CHECK(state.position.y < 0.4f); // no higher than a step
  CHECK(state.position.x > 3.0f); // but it did walk up to the slope (x = 3.9)
}

TEST(max_slope_decides_what_is_walkable) {
  for (const float maxSlope : {45.0f, 30.0f}) {
    PhysicsWorld world;
    addFloor(world);
    addRamp(world, 40.0f);
    const PhysicsCharacterId character = addCharacter(world, glm::vec3(0.0f));
    PhysicsCharacterMove move;
    move.maxSlope = glm::radians(maxSlope);
    walk(world, character, glm::vec3(3.0f, 0.0f, 0.0f), 120, move);
    const float y = world.characterState(character).position.y;
    if (maxSlope > 40.0f)
      CHECK(y > 1.0f);
    else
      CHECK(y < 0.5f);
  }
}

TEST(character_sticks_to_a_slope_going_down) {
  // Walking down 30 degrees at 4 m/s drops 4.5 cm a step, faster than
  // gravity pulls at first, so without sticking it hops down the slope.
  for (const float stickDown : {0.5f, 0.0f}) {
    PhysicsWorld world;
    addFloor(world);
    addRamp(world, 30.0f);
    const PhysicsCharacterId character =
        addCharacter(world, glm::vec3(7.0f, rampSurfaceY(7.0f, 30.0f) + 0.05f, 0.0f));
    walk(world, character, glm::vec3(0.0f), 30); // settle
    PhysicsCharacterMove move;
    move.stickDown = stickDown;
    int stepsInAir = 0;
    for (int i = 0; i < 30; ++i) {
      stepCharacter(world, character, glm::vec3(-4.0f, 0.0f, 0.0f), move);
      if (world.characterState(character).ground != PhysicsGround::OnGround)
        ++stepsInAir;
    }
    if (stickDown > 0.0f)
      CHECK(stepsInAir == 0);
    else
      CHECK(stepsInAir > 5);
  }
}

TEST(crouched_character_cant_stand_under_a_ceiling) {
  PhysicsWorld world;
  addFloor(world);
  // The ceiling's underside is 1.4 m up, over x = -1 to 1.
  addStaticBox(world, glm::vec3(0.0f, 1.5f, 0.0f), glm::vec3(1.0f, 0.1f, 2.0f));
  const PhysicsCharacterId character = addCharacter(world, glm::vec3(-3, 0, 0));
  walk(world, character, glm::vec3(0.0f), 10);
  CHECK(world.setCharacterHeight(character, 1.1f));
  CHECK_NEAR(world.characterState(character).height, 1.1, 1e-4);
  walk(world, character, glm::vec3(3.0f, 0.0f, 0.0f), 60); // under it
  CHECK_NEAR(world.characterState(character).position.x, 0.0, 0.2);
  CHECK(!world.setCharacterHeight(character, 1.8f));
  CHECK_NEAR(world.characterState(character).height, 1.1, 1e-4);
  CHECK_NEAR(world.characterState(character).position.y, 0.0, 0.03);
  walk(world, character, glm::vec3(3.0f, 0.0f, 0.0f), 60); // out again
  CHECK(world.setCharacterHeight(character, 1.8f));
  CHECK_NEAR(world.characterState(character).height, 1.8, 1e-4);
}

TEST(thrown_ball_hits_the_character) {
  PhysicsWorld world;
  addFloor(world);
  const PhysicsCharacterId character = addCharacter(world, glm::vec3(0.0f));
  walk(world, character, glm::vec3(0.0f), 10);
  PhysicsBodyDesc ball;
  ball.shape = PhysicsShape::Sphere;
  ball.radius = 0.15f;
  ball.position = glm::vec3(-3.0f, 1.2f, 0.0f);
  ball.linearVelocity = glm::vec3(10.0f, 0.0f, 0.0f);
  const PhysicsBodyId id = world.addBody(ball);
  walk(world, character, glm::vec3(0.0f), 40);
  // Without a body in the capsule it would fly straight through to x > 3.
  CHECK(world.pose(id).position.x < -0.3f);
}

TEST(character_body_stays_out_of_moved_poses) {
  PhysicsWorld world;
  addFloor(world);
  const PhysicsCharacterId character = addCharacter(world, glm::vec3(0.0f));
  PhysicsBodyDesc box;
  box.position = glm::vec3(5.0f, 3.0f, 0.0f);
  box.userData = 5;
  world.addBody(box);
  CHECK(world.bodyCount() == 3); // the floor, the character's body, the box
  bool onlyTheBox = true;
  bool sawTheBox = false;
  for (int i = 0; i < 30; ++i) {
    stepCharacter(world, character, glm::vec3(1.0f, 0.0f, 0.0f));
    for (const PhysicsMovedPose &moved : world.movedPoses(1.0f)) {
      onlyTheBox = onlyTheBox && moved.userData == 5;
      sawTheBox = sawTheBox || moved.userData == 5;
    }
  }
  CHECK(onlyTheBox);
  CHECK(sawTheBox);
}

TEST(character_pushes_a_light_box) {
  for (const float strength : {100.0f, 0.0f}) {
    PhysicsWorld world;
    addFloor(world);
    PhysicsBodyDesc box;
    box.halfExtents = glm::vec3(0.3f);
    box.mass = 10.0f;
    box.position = glm::vec3(1.0f, 0.3f, 0.0f);
    const PhysicsBodyId id = world.addBody(box);
    const PhysicsCharacterId character = addCharacter(world, glm::vec3(0.0f));
    PhysicsCharacterMove move;
    move.pushStrength = strength;
    walk(world, character, glm::vec3(2.0f, 0.0f, 0.0f), 90, move);
    if (strength > 0.0f)
      CHECK(world.pose(id).position.x > 1.8f);
    else
      CHECK_NEAR(world.pose(id).position.x, 1.0, 0.05);
  }
}

TEST(character_rides_what_it_stands_on) {
  PhysicsWorld world;
  addFloor(world);
  // A heavy slab sliding along the floor with no friction.
  PhysicsBodyDesc slab;
  slab.halfExtents = glm::vec3(2.0f, 0.25f, 2.0f);
  slab.mass = 5000.0f;
  slab.friction = 0.0f;
  slab.position = glm::vec3(0.0f, 0.25f, 0.0f);
  slab.linearVelocity = glm::vec3(1.0f, 0.0f, 0.0f);
  const PhysicsBodyId id = world.addBody(slab);
  const PhysicsCharacterId character = addCharacter(world, glm::vec3(0, 0.55f, 0));
  walk(world, character, glm::vec3(0.0f), 60);
  const PhysicsCharacterState state = world.characterState(character);
  CHECK(state.ground == PhysicsGround::OnGround);
  CHECK_NEAR(state.groundVelocity.x, 1.0, 0.1);
  CHECK_NEAR(state.position.x - world.pose(id).position.x, 0.0, 0.1);
}

TEST(ground_velocity_is_up_to_date_after_the_step) {
  PhysicsWorld world;
  addFloor(world);
  // Friction slows this slab by about 4.4 m/s^2, 7 cm/s every step.
  PhysicsBodyDesc slab;
  slab.halfExtents = glm::vec3(2.0f, 0.25f, 2.0f);
  slab.mass = 5000.0f;
  slab.friction = 1.0f;
  slab.position = glm::vec3(0.0f, 0.25f, 0.0f);
  slab.linearVelocity = glm::vec3(3.0f, 0.0f, 0.0f);
  const PhysicsBodyId id = world.addBody(slab);
  const PhysicsCharacterId character = addCharacter(world, glm::vec3(0, 0.55f, 0));
  walk(world, character, glm::vec3(0.0f), 5);
  for (int i = 0; i < 10; ++i) {
    const float before = world.pose(id).position.x;
    stepCharacter(world, character, glm::vec3(0.0f));
    const float slabSpeed = (world.pose(id).position.x - before) / kStep;
    CHECK_NEAR(world.characterState(character).groundVelocity.x, slabSpeed, 0.03);
  }
}

TEST(wall_beside_a_falling_character_is_not_ground) {
  PhysicsWorld world;
  addFloor(world);
  addStaticBox(world, glm::vec3(1.0f, 3.0f, 0.0f), glm::vec3(0.5f, 3.0f, 5.0f));
  // Pressed against the wall's face at x = 0.5 while it falls.
  const PhysicsCharacterId character = addCharacter(world, glm::vec3(0.15f, 4, 0));
  bool wallWasGround = false;
  for (int i = 0; i < 80; ++i) { // the fall takes about 54 steps
    stepCharacter(world, character, glm::vec3(2.0f, 0.0f, 0.0f));
    const PhysicsCharacterState state = world.characterState(character);
    if (state.position.y > 0.5f)
      wallWasGround = wallWasGround || state.ground == PhysicsGround::OnSteepGround ||
                      state.ground == PhysicsGround::OnGround;
  }
  CHECK(!wallWasGround);
  CHECK(world.characterState(character).ground == PhysicsGround::OnGround);
}

TEST(place_character_moves_it_and_stops_it) {
  PhysicsWorld world;
  addFloor(world);
  const PhysicsCharacterId character = addCharacter(world, glm::vec3(0, 5, 0));
  walk(world, character, glm::vec3(1.0f, 0.0f, 0.0f), 20); // falling
  world.placeCharacter(character, glm::vec3(5.0f, 0.0f, 2.0f));
  const PhysicsCharacterState state = world.characterState(character);
  CHECK_NEAR(glm::distance(state.position, glm::vec3(5, 0, 2)), 0.0, 1e-4);
  CHECK_NEAR(glm::length(state.velocity), 0.0, 1e-6);
}

TEST(removed_character_is_gone) {
  PhysicsWorld world;
  addFloor(world);
  const PhysicsCharacterId character = addCharacter(world, glm::vec3(0.0f));
  CHECK(world.bodyCount() == 2);
  world.removeCharacter(character);
  CHECK(world.bodyCount() == 1);
  world.removeCharacter(character); // twice is harmless
  world.moveCharacter(character, PhysicsCharacterMove{}, kStep); // so is this
  CHECK(!world.setCharacterHeight(character, 1.1f));
  CHECK_NEAR(world.characterState(character).height, 0.0, 1e-6);
}

TEST(cast_sphere_stops_short_by_its_radius) {
  PhysicsWorld world;
  addStaticBox(world, glm::vec3(0.0f, 0.0f, -5.0f), glm::vec3(0.5f));
  const std::optional<PhysicsRayHit> hit = world.castSphere(
      glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -2.0f), 0.25f, 100.0f);
  CHECK(hit.has_value());
  if (hit) {
    CHECK_NEAR(hit->distance, 4.25, 1e-3); // the face is at 4.5
    CHECK_NEAR(glm::distance(hit->normal, glm::vec3(0, 0, 1)), 0.0, 1e-3);
  }
}

TEST(cast_sphere_misses_past_max_distance) {
  PhysicsWorld world;
  addStaticBox(world, glm::vec3(0.0f, 0.0f, -5.0f), glm::vec3(0.5f));
  CHECK(!world.castSphere(glm::vec3(0.0f), glm::vec3(0, 0, -1), 0.25f, 4.0f));
  CHECK(world.castSphere(glm::vec3(0.0f), glm::vec3(0, 0, -1), 0.25f, 4.5f));
}

TEST(cast_ray_passes_through_characters) {
  PhysicsWorld world;
  addFloor(world);
  addStaticBox(world, glm::vec3(0.0f, 1.0f, -5.0f), glm::vec3(0.5f));
  addCharacter(world, glm::vec3(0.0f, 0.0f, -2.5f));
  const std::optional<PhysicsRayHit> hit = world.castRay(
      glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0, 0, -1), 100.0f);
  CHECK_NEAR(hit ? hit->distance : -1.0f, 4.5, 1e-3);
}

TEST(cast_sphere_passes_through_characters) {
  PhysicsWorld world;
  addFloor(world);
  addStaticBox(world, glm::vec3(0.0f, 1.0f, -5.0f), glm::vec3(0.5f));
  addCharacter(world, glm::vec3(0.0f, 0.0f, -2.5f));
  // From inside a character too, like a camera's pivot.
  addCharacter(world, glm::vec3(0.0f, 0.0f, 0.0f));
  const std::optional<PhysicsRayHit> hit = world.castSphere(
      glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0, 0, -1), 0.25f, 100.0f);
  CHECK(hit.has_value());
  CHECK_NEAR(hit ? hit->distance : -1.0f, 4.25, 1e-3);
}
