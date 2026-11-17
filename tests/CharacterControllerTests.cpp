#include "CharacterController.h"
#include "CharacterTestScenes.h"
#include "TestRunner.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace {

using test::addFloor;
using test::addRamp;
using test::addStaticBox;

constexpr float kStep = 1.0f / 60.0f;
constexpr float kPi = 3.14159265f;

CharacterInput moving(glm::vec3 direction) {
  CharacterInput input;
  input.move = direction;
  return input;
}

CharacterInput pressing(bool jump, bool dodge, glm::vec3 direction = glm::vec3(0.0f)) {
  CharacterInput input = moving(direction);
  input.jump = jump;
  input.dodge = dodge;
  return input;
}

// Steps the controller and the world together, the way the game will. Jump
// and dodge count as pressed on the first step only.
void hold(PhysicsWorld &world, CharacterController &character,
          CharacterInput input, int steps) {
  for (int i = 0; i < steps; ++i) {
    character.update(input, kStep);
    world.step(kStep);
    input.jump = false;
    input.dodge = false;
  }
}

float flatSpeed(glm::vec3 v) { return std::sqrt(v.x * v.x + v.z * v.z); }

// The highest the feet get in the next steps, holding the input.
float highestOver(PhysicsWorld &world, CharacterController &character,
                  CharacterInput input, int steps) {
  float highest = character.position().y;
  for (int i = 0; i < steps; ++i) {
    hold(world, character, input, 1);
    input.jump = input.dodge = false;
    highest = std::max(highest, character.position().y);
  }
  return highest;
}

float facingError(float yaw, glm::vec3 direction) {
  return glm::distance(yawForward(yaw), direction);
}

} // namespace

TEST(yaw_zero_faces_minus_z) {
  CHECK_NEAR(glm::distance(yawForward(0.0f), glm::vec3(0, 0, -1)), 0.0, 1e-6);
  CHECK_NEAR(glm::distance(yawForward(-0.5f * kPi), glm::vec3(1, 0, 0)), 0.0, 1e-6);
}

TEST(controller_spawns_standing_where_placed) {
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(2.0f, 0.0f, 1.0f), 0.5f);
  CHECK_NEAR(glm::distance(character.position(), glm::vec3(2, 0, 1)), 0.0, 1e-4);
  CHECK_NEAR(character.yaw(), 0.5, 1e-6);
  CHECK_NEAR(character.height(), 1.8, 1e-5);
  hold(world, character, CharacterInput{}, 30);
  CHECK(character.grounded());
  CHECK_NEAR(glm::distance(character.position(), glm::vec3(2, 0, 1)), 0.0, 0.03);
  CHECK_NEAR(character.yaw(), 0.5, 1e-6); // nothing turned it
}

TEST(controller_leaves_the_world_with_its_character) {
  PhysicsWorld world;
  addFloor(world);
  {
    CharacterController character(world, glm::vec3(0.0f), 0.0f);
    CHECK(world.bodyCount() == 2);
  }
  CHECK(world.bodyCount() == 1);
}

TEST(speed_builds_from_a_walk_to_a_run) {
  // Moving at once, but the push fades near the run (5 m/s): walking pace
  // after 0.2 s, a jog by 0.6 s, nearly the run by 1.2 s.
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f), 0.0f);
  const CharacterInput ahead = moving(yawForward(0.0f));
  hold(world, character, ahead, 12);
  const float walking = flatSpeed(character.velocity());
  CHECK(walking > 1.0f && walking < 2.5f);
  hold(world, character, ahead, 24);
  const float jogging = flatSpeed(character.velocity());
  CHECK(jogging > 2.8f && jogging < 4.0f);
  hold(world, character, ahead, 36);
  const float nearly = flatSpeed(character.velocity());
  CHECK(nearly > 4.3f && nearly < 5.0f);
  hold(world, character, ahead, 108);
  CHECK_NEAR(flatSpeed(character.velocity()), 5.0, 0.05);
}

TEST(sprint_builds_on_from_the_run) {
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f), 0.0f);
  CharacterInput input = moving(yawForward(0.0f));
  hold(world, character, input, 180);
  input.sprint = true;
  hold(world, character, input, 30);
  const float building = flatSpeed(character.velocity());
  CHECK(building > 5.5f && building < 7.0f);
  hold(world, character, input, 150);
  CHECK_NEAR(flatSpeed(character.velocity()), 8.0, 0.1);
  // Letting go of sprint slows it back to the run at groundBraking: 1 m/s
  // slower after 0.1 s, the run by 0.3 s.
  input.sprint = false;
  hold(world, character, input, 6);
  CHECK_NEAR(flatSpeed(character.velocity()), 6.93, 0.1);
  hold(world, character, input, 24);
  CHECK_NEAR(flatSpeed(character.velocity()), 5.0, 0.05);
  // Half a stick is half the speed.
  input = moving(0.5f * yawForward(0.0f));
  hold(world, character, input, 60);
  CHECK_NEAR(flatSpeed(character.velocity()), 2.5, 0.05);
}

TEST(stopping_takes_a_moment) {
  // From the run at groundBraking (10 m/s^2): half the speed in 0.25 s,
  // stopped by 0.5 s, about 1.25 m on.
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f), 0.0f);
  hold(world, character, moving(yawForward(0.0f)), 180);
  const float before = character.position().z;
  hold(world, character, CharacterInput{}, 15);
  CHECK_NEAR(flatSpeed(character.velocity()), 2.5, 0.1);
  hold(world, character, CharacterInput{}, 20);
  CHECK_NEAR(flatSpeed(character.velocity()), 0.0, 1e-3);
  const float slid = before - character.position().z;
  CHECK(slid > 1.0f && slid < 1.5f);
}

TEST(turning_at_a_run_follows_an_arc) {
  // At 5 m/s with 16 m/s^2 of grip it turns at 3.2 rad/s: a quarter turn
  // takes about 0.5 s and swings about 1.5 m wide, keeping its speed.
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f), 0.0f);
  hold(world, character, moving(yawForward(0.0f)), 180);
  const glm::vec3 start = character.position();
  const glm::vec3 right(1.0f, 0.0f, 0.0f);
  hold(world, character, moving(right), 18);
  const float turned = std::acos(std::clamp(
      glm::dot(yawForward(character.yaw()), yawForward(0.0f)), -1.0f, 1.0f));
  CHECK(turned > glm::radians(45.0f) && turned < glm::radians(65.0f));
  hold(world, character, moving(right), 30);
  CHECK_NEAR(facingError(character.yaw(), right), 0.0, 1e-3);
  CHECK(flatSpeed(character.velocity()) > 4.5f);
  // The arc carried it on ahead before it went right.
  CHECK(start.z - character.position().z > 1.2f);
}

TEST(turning_on_the_spot_is_quick) {
  // Standing, it turns at turnRate first and only then sets off.
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f), 0.0f);
  hold(world, character, CharacterInput{}, 10);
  // Half way round it pushes only as much as it faces the way: about half
  // of a straight push's 0.67 m/s by now.
  hold(world, character, moving(glm::vec3(1, 0, 0)), 4);
  CHECK(flatSpeed(character.velocity()) < 0.45f);
  hold(world, character, moving(glm::vec3(1, 0, 0)), 4);
  CHECK_NEAR(facingError(character.yaw(), glm::vec3(1, 0, 0)), 0.0, 1e-3);
  CHECK(glm::length(character.position()) < 0.1f);
}

TEST(going_back_at_speed_plants_then_turns) {
  // Asked to go back at a run, it brakes along the way it was going, then
  // turns and sets off: no wide U-turn.
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f), 0.0f);
  hold(world, character, moving(yawForward(0.0f)), 180);
  const glm::vec3 back = -yawForward(0.0f);
  hold(world, character, moving(back), 18);
  CHECK_NEAR(facingError(character.yaw(), yawForward(0.0f)), 0.0, 1e-3);
  CHECK(flatSpeed(character.velocity()) < 3.0f);
  float widest = 0.0f;
  for (int i = 0; i < 72; ++i) {
    hold(world, character, moving(back), 1);
    widest = std::max(widest, std::fabs(character.position().x));
  }
  CHECK_NEAR(facingError(character.yaw(), back), 0.0, 1e-3);
  CHECK(glm::dot(character.velocity(), back) > 2.0f);
  CHECK(widest < 0.3f);
}

TEST(a_wall_takes_the_stored_speed) {
  // Stopped by a wall, it doesn't keep its run: going along the wall
  // starts again from standing.
  PhysicsWorld world;
  addFloor(world);
  addStaticBox(world, glm::vec3(0.0f, 1.0f, -3.0f), glm::vec3(5.0f, 1.0f, 0.25f));
  CharacterController character(world, glm::vec3(0.0f), 0.0f);
  hold(world, character, moving(yawForward(0.0f)), 90);
  CHECK(flatSpeed(character.velocity()) < 0.5f);
  // With the run still stored it would swing round at 5 m/s, over 3 m/s
  // along the wall by now.
  hold(world, character, moving(glm::vec3(1, 0, 0)), 18);
  CHECK(flatSpeed(character.velocity()) < 2.5f);
}

TEST(aiming_keeps_quick_footwork) {
  // While aiming it eases in a straight line at aimAcceleration (30 m/s^2):
  // the run in a sixth of a second, and a side step reversed as quickly.
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f), 0.0f);
  CharacterInput input = moving(glm::vec3(1, 0, 0));
  input.aim = true;
  hold(world, character, input, 12);
  CHECK_NEAR(character.velocity().x, 5.0, 0.05);
  input.move = glm::vec3(-1, 0, 0);
  hold(world, character, input, 24);
  CHECK_NEAR(character.velocity().x, -5.0, 0.05);
}

TEST(air_control_is_weaker_than_on_the_ground) {
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f, 5.0f, 0.0f), 0.0f);
  hold(world, character, moving(glm::vec3(1, 0, 0)), 12);
  CHECK(!character.grounded());
  CHECK_NEAR(flatSpeed(character.velocity()), 8.0 * 12.0 / 60.0, 0.05); // 8 m/s^2
}

TEST(jump_reaches_its_height) {
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f), 0.0f);
  hold(world, character, CharacterInput{}, 10);
  const float start = character.position().y;
  const float highest = highestOver(world, character, pressing(true, false), 90);
  CHECK_NEAR(highest - start, 1.2, 0.02);
  CHECK(character.grounded()); // and lands again
}

TEST(jump_height_holds_under_other_gravity) {
  PhysicsWorld world;
  PhysicsWorld::Config config;
  config.gravity = glm::vec3(0.0f, -4.0f, 0.0f);
  world.applySettings(config);
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f), 0.0f);
  hold(world, character, CharacterInput{}, 10);
  const float start = character.position().y;
  float highest = start;
  int stepsInAir = 0;
  hold(world, character, pressing(true, false), 1);
  while (!character.grounded() && stepsInAir < 200) {
    highest = std::max(highest, character.position().y);
    hold(world, character, CharacterInput{}, 1);
    ++stepsInAir;
  }
  CHECK_NEAR(highest - start, 1.2, 0.02);
  // Up and down takes 2 sqrt(2 h / g): 1.55 s here, 0.99 s on Earth.
  CHECK_NEAR(stepsInAir, 1.55 * 60.0, 4.0);
}

TEST(small_hop_isnt_cancelled_by_the_floor) {
  // A 10 cm hop rises 1.5 cm in its first step, still in reach of the
  // floor's contact, so the step after it must not count as standing.
  PhysicsWorld world;
  addFloor(world);
  CharacterTuning tuning;
  tuning.jumpHeight = 0.1f;
  CharacterController character(world, glm::vec3(0.0f), 0.0f, tuning);
  hold(world, character, CharacterInput{}, 10);
  const float start = character.position().y;
  const float highest = highestOver(world, character, pressing(true, false), 30);
  CHECK_NEAR(highest - start, 0.1, 0.02);
}

TEST(jump_stops_at_a_ceiling) {
  PhysicsWorld world;
  addFloor(world);
  // 50 cm of headroom over a 1.8 m character.
  addStaticBox(world, glm::vec3(0.0f, 2.4f, 0.0f), glm::vec3(3.0f, 0.1f, 3.0f));
  CharacterController character(world, glm::vec3(0.0f), 0.0f);
  hold(world, character, CharacterInput{}, 10);
  hold(world, character, pressing(true, false), 1);
  int stepsInAir = 1;
  while (!character.grounded() && stepsInAir < 120) {
    hold(world, character, CharacterInput{}, 1);
    ++stepsInAir;
  }
  // Up 50 cm and straight back down is about 26 steps. Pressing on into
  // the ceiling until gravity turned it round would be about 49.
  CHECK(stepsInAir < 35);
}

TEST(no_second_jump_in_the_air) {
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f), 0.0f);
  hold(world, character, CharacterInput{}, 10);
  const float start = character.position().y;
  hold(world, character, pressing(true, false), 20);
  const float highest = highestOver(world, character, pressing(true, false), 60);
  CHECK(highest - start < 1.25f);
}

namespace {

// Walks off a 1 m ledge and presses jump the given number of steps after
// leaving it. Returns how much higher than at the press the feet get.
float riseAfterJumpingLate(int stepsAfterLeaving) {
  PhysicsWorld world;
  addFloor(world);
  addStaticBox(world, glm::vec3(-2.5f, 0.5f, 0.0f), glm::vec3(2.5f, 0.5f, 2.0f));
  CharacterController character(world, glm::vec3(-1.0f, 1.0f, 0.0f), 0.0f);
  hold(world, character, CharacterInput{}, 5);
  const CharacterInput walk = moving(glm::vec3(0.6f, 0.0f, 0.0f));
  for (int i = 0; i < 120 && character.grounded(); ++i)
    hold(world, character, walk, 1);
  hold(world, character, walk, stepsAfterLeaving);
  const float atPress = character.position().y;
  CharacterInput jump = walk;
  jump.jump = true;
  return highestOver(world, character, jump, 40) - atPress;
}

// Drops the character onto the floor and presses jump the given number of
// steps before it lands. Returns the highest it gets after landing.
float heightAfterLandingWithJumpPressedEarly(int stepsBeforeLanding) {
  int landingStep = -1;
  {
    PhysicsWorld world;
    addFloor(world);
    CharacterController character(world, glm::vec3(0.0f, 2.0f, 0.0f), 0.0f);
    for (int i = 0; i < 120 && landingStep < 0; ++i) {
      hold(world, character, CharacterInput{}, 1);
      if (character.grounded())
        landingStep = i;
    }
  }
  CHECK(landingStep > 20);
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f, 2.0f, 0.0f), 0.0f);
  for (int i = 0; i <= landingStep; ++i)
    hold(world, character, pressing(i == landingStep - stepsBeforeLanding, false), 1);
  return highestOver(world, character, CharacterInput{}, 30);
}

} // namespace

TEST(coyote_time_lets_a_jump_off_a_ledge) {
  CHECK(riseAfterJumpingLate(3) > 0.5f);  // 0.05 s after leaving: jumps
  CHECK(riseAfterJumpingLate(9) < 0.01f); // 0.15 s: too late, just falls
}

TEST(jump_buffer_jumps_on_landing) {
  CHECK(heightAfterLandingWithJumpPressedEarly(3) > 0.5f);   // 0.05 s early
  CHECK(heightAfterLandingWithJumpPressedEarly(12) < 0.05f); // 0.2 s early
}

TEST(crouch_slows_and_waits_for_headroom) {
  PhysicsWorld world;
  addFloor(world);
  // The ceiling's underside is 1.4 m up, over x = -1 to 1.
  addStaticBox(world, glm::vec3(0.0f, 1.5f, 0.0f), glm::vec3(1.0f, 0.1f, 2.0f));
  CharacterController character(world, glm::vec3(-3.0f, 0.0f, 0.0f), 0.0f);
  CharacterInput crouchWalk = moving(glm::vec3(1, 0, 0));
  crouchWalk.crouch = true;
  hold(world, character, crouchWalk, 90);
  CHECK_NEAR(character.height(), 1.1, 1e-5);
  CHECK_NEAR(character.velocity().x, 2.0, 0.05);
  CHECK_NEAR(character.position().x, 0.0, 0.8); // under the ceiling
  hold(world, character, CharacterInput{}, 10); // let go of crouch under it
  CHECK_NEAR(character.height(), 1.1, 1e-5);
  hold(world, character, moving(glm::vec3(1, 0, 0)), 90); // walk out
  CHECK(character.position().x > 1.5f);
  CHECK_NEAR(character.height(), 1.8, 1e-5);
}

// A dodge path 2 m ahead by 0.5 s and 3 m by 1 s, slowing down.
RootMotion slowingPath(glm::vec2 way) {
  RootMotion path;
  path.duration = 1.0f;
  path.offsets = {{0.0f, glm::vec2(0.0f)}, {0.5f, 2.0f * way}, {1.0f, 3.0f * way}};
  return path;
}

TEST(a_roll_follows_its_clips_path) {
  // Supplied a roll's path, the dodge moves along it the way it's pressed
  // and lasts as long as it.
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f), 0.0f);
  std::array<RootMotion, 5> paths;
  paths[static_cast<size_t>(DodgeKind::Roll)] = slowingPath(glm::vec2(0.0f, 1.0f));
  character.setDodgePaths(paths);
  hold(world, character, CharacterInput{}, 10);
  hold(world, character, pressing(false, true, glm::vec3(1, 0, 0)), 1);
  CHECK(character.dodgeKind() == DodgeKind::Roll);
  hold(world, character, CharacterInput{}, 29); // 0.5 s in
  CHECK(character.dodging());
  CHECK_NEAR(character.position().x, 2.0, 0.05);
  hold(world, character, CharacterInput{}, 31);
  CHECK(!character.dodging());
  CHECK_NEAR(character.position().x, 3.0, 0.05);
  CHECK_NEAR(character.position().z, 0.0, 0.01);
  // Out of it at the path's last speed (2 m/s), braking: 0.2 m on.
  hold(world, character, CharacterInput{}, 20);
  CHECK_NEAR(character.position().x, 3.2, 0.05);
}

TEST(dodge_scale_stretches_the_path) {
  PhysicsWorld world;
  addFloor(world);
  CharacterTuning tuning;
  tuning.dodgeScale = 2.0f;
  CharacterController character(world, glm::vec3(0.0f), 0.0f, tuning);
  std::array<RootMotion, 5> paths;
  paths[static_cast<size_t>(DodgeKind::Roll)] = slowingPath(glm::vec2(0.0f, 1.0f));
  character.setDodgePaths(paths);
  hold(world, character, CharacterInput{}, 10);
  hold(world, character, pressing(false, true, glm::vec3(1, 0, 0)), 61);
  CHECK(!character.dodging());
  CHECK_NEAR(character.position().x, 6.0, 0.1);
}

TEST(an_aimed_dodge_steps_the_way_pressed) {
  // Aiming down -z and pressing right: the right dodge, its path laid out
  // from the way it faces, so its own "right" carries it to +x.
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f), 0.0f);
  std::array<RootMotion, 5> paths;
  paths[static_cast<size_t>(DodgeKind::Right)] = slowingPath(glm::vec2(0.5f, 0.0f));
  paths[static_cast<size_t>(DodgeKind::Back)] = slowingPath(glm::vec2(0.0f, -0.5f));
  character.setDodgePaths(paths);
  CharacterInput aim;
  aim.aim = true;
  hold(world, character, aim, 10);
  CharacterInput right = aim;
  right.move = glm::vec3(1, 0, 0);
  right.dodge = true;
  hold(world, character, right, 1);
  CHECK(character.dodgeKind() == DodgeKind::Right);
  hold(world, character, aim, 60);
  CHECK_NEAR(character.position().x, 1.5, 0.05);
  CHECK_NEAR(character.position().z, 0.0, 0.05);
  CHECK_NEAR(facingError(character.yaw(), glm::vec3(0, 0, -1)), 0.0, 1e-3);
  // Pressing back (+z, behind it): the back dodge.
  CharacterInput back = aim;
  back.move = glm::vec3(0, 0, 1);
  back.dodge = true;
  hold(world, character, back, 1);
  CHECK(character.dodgeKind() == DodgeKind::Back);
  hold(world, character, aim, 60);
  CHECK_NEAR(character.position().z, 1.5, 0.05);
}

TEST(dodge_covers_its_distance) {
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f), 0.0f);
  hold(world, character, CharacterInput{}, 10);
  hold(world, character, pressing(false, true, glm::vec3(1, 0, 0)), 1);
  CHECK(character.dodging());
  hold(world, character, CharacterInput{}, 13);
  CHECK(character.dodging());
  hold(world, character, CharacterInput{}, 1); // 15 steps = 0.25 s
  CHECK(!character.dodging());
  CHECK_NEAR(character.position().x, 5.0, 0.1);
  // Out of it at run speed at most, then no input brakes it like a run:
  // 1.25 m on.
  hold(world, character, CharacterInput{}, 1);
  CHECK_NEAR(flatSpeed(character.velocity()), 5.0 - 10.0 / 60.0, 0.05);
  hold(world, character, CharacterInput{}, 39);
  CHECK_NEAR(character.position().x, 6.25, 0.1);
}

TEST(dodge_without_input_goes_the_way_it_faces) {
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f), 0.0f);
  hold(world, character, CharacterInput{}, 10);
  hold(world, character, pressing(false, true), 15);
  CHECK_NEAR(character.position().z, -5.0, 0.1);
  CHECK_NEAR(character.position().x, 0.0, 0.01);
}

TEST(dodge_waits_for_its_cooldown) {
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f), 0.0f);
  hold(world, character, CharacterInput{}, 10);
  const CharacterInput dodgeRight = pressing(false, true, glm::vec3(1, 0, 0));
  hold(world, character, dodgeRight, 1);
  hold(world, character, CharacterInput{}, 23);
  hold(world, character, dodgeRight, 1); // 0.4 s after the first: ignored
  hold(world, character, CharacterInput{}, 20);
  CHECK_NEAR(character.position().x, 6.25, 0.1); // 5 m, then braking
  hold(world, character, dodgeRight, 1); // 0.75 s: allowed
  hold(world, character, CharacterInput{}, 14);
  CHECK_NEAR(character.position().x, 11.25, 0.15);
}

TEST(no_dodge_in_the_air) {
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f, 3.0f, 0.0f), 0.0f);
  hold(world, character, pressing(false, true, glm::vec3(1, 0, 0)), 15);
  CHECK(!character.dodging());
  CHECK(character.position().x < 1.0f);
}

TEST(character_turns_to_face_where_it_moves) {
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f), 0.0f);
  hold(world, character, moving(glm::vec3(1, 0, 0)), 1);
  CHECK_NEAR(character.yaw(), -12.566371 / 60.0, 1e-4); // one step of turning
  hold(world, character, moving(glm::vec3(1, 0, 0)), 30);
  CHECK_NEAR(facingError(character.yaw(), glm::vec3(1, 0, 0)), 0.0, 1e-4);
}

TEST(turn_takes_the_short_way) {
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f), 3.0f);
  // -3.0 is 0.28 rad away across the back, 6 rad away the long way.
  hold(world, character, moving(yawForward(-3.0f)), 2);
  CHECK_NEAR(facingError(character.yaw(), yawForward(-3.0f)), 0.0, 1e-4);
}

TEST(aiming_faces_the_camera_while_moving) {
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f), 0.0f);
  CharacterInput input = moving(glm::vec3(1, 0, 0));
  input.aim = true;
  input.aimYaw = kPi; // faces +z
  hold(world, character, input, 60);
  CHECK_NEAR(facingError(character.yaw(), glm::vec3(0, 0, 1)), 0.0, 1e-4);
  CHECK(character.position().x > 3.0f); // still walking sideways
}

TEST(controller_rides_a_sliding_slab) {
  PhysicsWorld world;
  addFloor(world);
  PhysicsBodyDesc slab;
  slab.halfExtents = glm::vec3(2.0f, 0.25f, 2.0f);
  slab.mass = 5000.0f;
  slab.friction = 0.0f;
  slab.position = glm::vec3(0.0f, 0.25f, 0.0f);
  slab.linearVelocity = glm::vec3(1.0f, 0.0f, 0.0f);
  const PhysicsBodyId id = world.addBody(slab);
  CharacterController character(world, glm::vec3(0.0f, 0.5f, 0.0f), 0.0f);
  hold(world, character, CharacterInput{}, 60);
  CHECK_NEAR(character.position().x - world.pose(id).position.x, 0.0, 0.1);
}

TEST(drawn_pose_blends_the_last_two_steps) {
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f), 0.0f);
  hold(world, character, moving(glm::vec3(1, 0, 0)), 29);
  const glm::vec3 before = character.position();
  const float yawBefore = character.yaw();
  hold(world, character, moving(glm::vec3(1, 0, 0)), 1);
  const glm::vec3 after = character.position();
  CHECK_NEAR(glm::distance(character.drawnPosition(0.0f), before), 0.0, 1e-5);
  CHECK_NEAR(glm::distance(character.drawnPosition(1.0f), after), 0.0, 1e-5);
  CHECK_NEAR(glm::distance(character.drawnPosition(0.5f), 0.5f * (before + after)), 0.0, 1e-5);
  CHECK_NEAR(character.drawnYaw(0.0f), yawBefore, 1e-5);
  // Placing it shows it there at once, not sliding in.
  character.place(glm::vec3(4.0f, 0.0f, 4.0f), 1.0f);
  CHECK_NEAR(glm::distance(character.drawnPosition(0.0f), glm::vec3(4, 0, 4)), 0.0, 1e-4);
  CHECK_NEAR(character.drawnYaw(0.3f), 1.0, 1e-5);
  CHECK_NEAR(glm::length(character.velocity()), 0.0, 1e-5);
}

TEST(drawn_yaw_blends_the_short_way) {
  PhysicsWorld world;
  addFloor(world);
  CharacterController character(world, glm::vec3(0.0f), 3.1f);
  hold(world, character, moving(yawForward(-3.1f)), 1); // crosses +-pi
  const float half = character.drawnYaw(0.5f);
  CHECK(std::fabs(std::fabs(half) - kPi) < 0.1f); // near the back, not the front
}

TEST(tuning_reaches_the_physics_character) {
  // stepUp: a 35 cm step stops it with no step up.
  for (const float stepUp : {0.3f, 0.0f}) {
    PhysicsWorld world;
    addFloor(world);
    addStaticBox(world, glm::vec3(2.5f, 0.175f, 0.0f), glm::vec3(1.0f, 0.175f, 2.0f));
    CharacterTuning tuning;
    tuning.stepUp = stepUp;
    CharacterController character(world, glm::vec3(0.0f), 0.0f, tuning);
    // Building up to 3 m/s, so it's on the 2 m long step at the end.
    hold(world, character, moving(glm::vec3(0.6f, 0, 0)), 80);
    CHECK((character.position().y > 0.3f) == (stepUp > 0.0f));
  }
  // maxSlope: a 40 degree ramp is too steep at 30.
  for (const float maxSlope : {45.0f, 30.0f}) {
    PhysicsWorld world;
    addFloor(world);
    addRamp(world, 40.0f);
    CharacterTuning tuning;
    tuning.maxSlope = glm::radians(maxSlope);
    CharacterController character(world, glm::vec3(0.0f), 0.0f, tuning);
    hold(world, character, moving(glm::vec3(0.6f, 0, 0)), 150);
    CHECK((character.position().y > 1.0f) == (maxSlope > 40.0f));
  }
  // pushStrength: with none it can't move a box.
  for (const float strength : {100.0f, 0.0f}) {
    PhysicsWorld world;
    addFloor(world);
    PhysicsBodyDesc box;
    box.halfExtents = glm::vec3(0.3f);
    box.mass = 10.0f;
    box.position = glm::vec3(1.0f, 0.3f, 0.0f);
    const PhysicsBodyId id = world.addBody(box);
    CharacterTuning tuning;
    tuning.pushStrength = strength;
    CharacterController character(world, glm::vec3(0.0f), 0.0f, tuning);
    hold(world, character, moving(glm::vec3(0.4f, 0, 0)), 90);
    CHECK((world.pose(id).position.x > 1.5f) == (strength > 0.0f));
  }
  // stickDown: running down 30 degrees, it leaves the ground without it.
  for (const float stickDown : {0.5f, 0.0f}) {
    PhysicsWorld world;
    addFloor(world);
    addRamp(world, 30.0f);
    CharacterTuning tuning;
    tuning.stickDown = stickDown;
    CharacterController character(
        world, glm::vec3(7.0f, test::rampSurfaceY(7.0f, 30.0f) + 0.05f, 0.0f), 0.0f, tuning);
    hold(world, character, CharacterInput{}, 30);
    int stepsInAir = 0;
    for (int i = 0; i < 20; ++i) {
      hold(world, character, moving(glm::vec3(-0.8f, 0, 0)), 1);
      stepsInAir += character.grounded() ? 0 : 1;
    }
    CHECK((stepsInAir == 0) == (stickDown > 0.0f));
  }
}

TEST(tuning_is_kept_consistent) {
  // A capsule needs more height than its two rounded ends, and crouching
  // can't make it taller. Values that already fit stay as they are.
  CharacterTuning fine;
  CHECK(consistent(fine).height == fine.height);
  CHECK(consistent(fine).crouchHeight == fine.crouchHeight);
  CharacterTuning squat;
  squat.radius = 0.5f;
  squat.height = 0.6f;
  squat.crouchHeight = 0.4f;
  const CharacterTuning fixed = consistent(squat);
  CHECK(fixed.height > 2.0f * fixed.radius);
  CHECK(fixed.crouchHeight > 2.0f * fixed.radius);
  CHECK(fixed.crouchHeight <= fixed.height);
  CharacterTuning tallCrouch;
  tallCrouch.crouchHeight = 2.5f;
  CHECK(consistent(tallCrouch).crouchHeight == tallCrouch.height);
}
