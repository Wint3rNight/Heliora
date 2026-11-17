#include "CharacterTestScenes.h"
#include "Player.h"
#include "TestRunner.h"

#include <cmath>

namespace {

using test::addFloor;

constexpr float kStep = 1.0f / 60.0f;
constexpr float kPi = 3.14159265f;

// One frame with exactly one physics step, the way the game runs at 60 fps.
void frame(PhysicsWorld &world, Player &player, const PlayerControls &controls,
           int steps = 1) {
  player.control(controls);
  for (int i = 0; i < steps; ++i) {
    player.step(kStep);
    world.step(kStep);
  }
  player.follow(1.0f, kStep);
}

void frames(PhysicsWorld &world, Player &player, PlayerControls controls,
            int count) {
  for (int i = 0; i < count; ++i) {
    frame(world, player, controls);
    controls.jump = controls.dodge = false;
    controls.mouse = glm::vec2(0.0f);
    controls.wheel = 0.0f;
  }
}

PlayerControls pressing(bool PlayerControls::*key) {
  PlayerControls controls;
  controls.*key = true;
  return controls;
}

// Pixels of mouse movement that turn the camera by this many radians, at
// the default sensitivity.
float pixelsFor(float radians) { return radians / 0.0025f; }

} // namespace

TEST(w_walks_where_the_camera_looks) {
  PhysicsWorld world;
  addFloor(world);
  Player player(world, glm::vec3(0.0f), 0.0f);
  PlayerControls turn;
  turn.mouse.x = pixelsFor(0.5f * kPi); // turn right, to look down +x
  frame(world, player, turn);
  frames(world, player, pressing(&PlayerControls::forward), 60);
  CHECK(player.character().position().x > 2.0f);
  CHECK(std::fabs(player.character().position().z) < 0.1f);
}

TEST(a_and_d_strafe_across_the_camera) {
  PhysicsWorld world;
  addFloor(world);
  Player player(world, glm::vec3(0.0f), 0.0f); // the camera looks down -z
  // Long enough to turn, build up speed and plant to go back.
  frames(world, player, pressing(&PlayerControls::right), 60);
  CHECK(player.character().position().x > 1.0f);
  frames(world, player, pressing(&PlayerControls::left), 120);
  CHECK(player.character().position().x < -1.0f);
  frames(world, player, pressing(&PlayerControls::back), 90);
  CHECK(player.character().position().z > 1.0f);
}

TEST(diagonal_is_no_faster) {
  PhysicsWorld world;
  addFloor(world);
  Player player(world, glm::vec3(0.0f), 0.0f);
  PlayerControls diagonal;
  diagonal.forward = diagonal.right = true;
  frames(world, player, diagonal, 240); // up to full speed
  const glm::vec3 v = player.character().velocity();
  CHECK_NEAR(std::sqrt(v.x * v.x + v.z * v.z), 5.0, 0.05);
}

TEST(jump_pressed_on_a_frame_without_a_step_still_jumps) {
  PhysicsWorld world;
  addFloor(world);
  Player player(world, glm::vec3(0.0f), 0.0f);
  frames(world, player, PlayerControls{}, 10);
  const float start = player.character().position().y;
  frame(world, player, pressing(&PlayerControls::jump), 0); // no step
  frame(world, player, PlayerControls{}, 1);
  frames(world, player, PlayerControls{}, 10);
  CHECK(player.character().position().y - start > 0.5f);
}

TEST(one_press_is_one_jump) {
  PhysicsWorld world;
  addFloor(world);
  Player player(world, glm::vec3(0.0f), 0.0f);
  frames(world, player, PlayerControls{}, 10);
  frame(world, player, pressing(&PlayerControls::jump), 3); // a slow frame
  frames(world, player, PlayerControls{}, 120);
  // Landed and stayed down: the press didn't fire again on landing.
  CHECK(player.character().grounded());
  CHECK(player.character().position().y < 0.05f);
}

TEST(shift_sprints_and_ctrl_crouches) {
  PhysicsWorld world;
  addFloor(world);
  Player player(world, glm::vec3(0.0f), 0.0f);
  PlayerControls sprint = pressing(&PlayerControls::forward);
  sprint.sprint = true;
  frames(world, player, sprint, 300); // the sprint builds for a while
  CHECK_NEAR(-player.character().velocity().z, 8.0, 0.05);
  PlayerControls crouch = pressing(&PlayerControls::forward);
  crouch.crouch = true;
  frames(world, player, crouch, 90);
  CHECK_NEAR(-player.character().velocity().z, 2.0, 0.05);
  CHECK_NEAR(player.character().height(), 1.1, 1e-5);
}

TEST(dodge_press_dodges) {
  PhysicsWorld world;
  addFloor(world);
  Player player(world, glm::vec3(0.0f), 0.0f);
  frames(world, player, PlayerControls{}, 10);
  frame(world, player, pressing(&PlayerControls::dodge), 0); // no step
  frames(world, player, PlayerControls{}, 15);
  CHECK_NEAR(player.character().position().z, -5.0, 0.1); // forward
}

TEST(aiming_turns_the_character_to_the_camera) {
  PhysicsWorld world;
  addFloor(world);
  Player player(world, glm::vec3(0.0f), 0.0f);
  PlayerControls turn;
  turn.mouse.x = pixelsFor(-kPi * 0.5f); // turn left, to look down -x
  frame(world, player, turn);
  frames(world, player, pressing(&PlayerControls::aim), 60);
  CHECK_NEAR(glm::distance(yawForward(player.character().yaw()),
                           glm::vec3(-1, 0, 0)), 0.0, 1e-3);
}

TEST(camera_follows_the_character) {
  PhysicsWorld world;
  addFloor(world);
  Player player(world, glm::vec3(0.0f), 0.0f);
  frames(world, player, pressing(&PlayerControls::forward), 60);
  frames(world, player, PlayerControls{}, 60); // stop and let it catch up
  const glm::vec3 feet = player.character().position();
  CHECK_NEAR(glm::distance(player.camera().pivot(), feet + glm::vec3(0, 1.5f, 0)),
             0.0, 0.01);
}

TEST(mouse_up_looks_up_and_the_wheel_zooms) {
  PhysicsWorld world;
  addFloor(world);
  Player player(world, glm::vec3(0.0f), 0.0f);
  const float pitch = player.camera().pitch();
  PlayerControls look;
  look.mouse.y = -40.0f; // up, screen style
  look.wheel = 1.0f;
  frame(world, player, look);
  CHECK_NEAR(player.camera().pitch() - pitch, 0.1, 1e-5);
  CHECK_NEAR(player.camera().distance(), 3.5, 1e-5);
}

TEST(camera_starts_behind_the_character) {
  PhysicsWorld world;
  addFloor(world);
  Player player(world, glm::vec3(2.0f, 0.0f, 0.0f), 0.5f * kPi); // faces -x
  CHECK_NEAR(player.camera().yaw(), 0.5 * kPi, 1e-5);
  CHECK(player.camera().position().x > 4.0f); // behind it, on the +x side
}

TEST(new_movement_tuning_applies_while_playing) {
  PhysicsWorld world;
  addFloor(world);
  Player player(world, glm::vec3(0.0f), 0.0f);
  CharacterTuning slower;
  slower.runSpeed = 3.0f;
  player.setTuning(slower, ThirdPersonCameraTuning{});
  frames(world, player, pressing(&PlayerControls::forward), 180);
  CHECK_NEAR(-player.character().velocity().z, 3.0, 0.05);
}

TEST(new_body_sizes_wait_for_the_next_spawn) {
  // The capsule in the physics world was made at the old size; changing it
  // under a moving character isn't safe, so Play picks it up next time.
  PhysicsWorld world;
  addFloor(world);
  Player player(world, glm::vec3(0.0f), 0.0f);
  CharacterTuning shorter;
  shorter.height = 1.2f;
  shorter.crouchHeight = 0.8f;
  shorter.radius = 0.2f;
  shorter.mass = 40.0f;
  shorter.jumpHeight = 2.0f; // this one does apply
  player.setTuning(shorter, ThirdPersonCameraTuning{});
  const CharacterTuning &now = player.character().tuning();
  CHECK_NEAR(now.height, 1.8, 1e-6);
  CHECK_NEAR(now.crouchHeight, 1.1, 1e-6);
  CHECK_NEAR(now.radius, 0.3, 1e-6);
  CHECK_NEAR(now.mass, 70.0, 1e-6);
  CHECK_NEAR(now.jumpHeight, 2.0, 1e-6);
  frames(world, player, PlayerControls{}, 10);
  CHECK_NEAR(player.character().height(), 1.8, 1e-5);
}

TEST(new_camera_tuning_applies_while_playing) {
  PhysicsWorld world;
  addFloor(world);
  Player player(world, glm::vec3(0.0f), 0.0f);
  ThirdPersonCameraTuning close;
  close.maxDistance = 3.0f;
  player.setTuning(CharacterTuning{}, close);
  CHECK_NEAR(player.camera().distance(), 3.0, 1e-5); // was 4, now the most
}

TEST(capture_scripts_hold_keys_or_press_them_on_a_frame) {
  PlayerControls c = scriptedControls("w shift", 5);
  CHECK(c.forward && c.sprint && !c.back && !c.jump);
  CHECK(!scriptedControls("space@120", 119).jump);
  CHECK(scriptedControls("space@120", 120).jump);
  CHECK(!scriptedControls("space@120", 121).jump);
  CHECK(!scriptedControls("w@10-20", 9).forward);
  CHECK(scriptedControls("w@10-20", 10).forward);
  CHECK(scriptedControls("w@10-20", 20).forward);
  CHECK(!scriptedControls("w@10-20", 21).forward);
  c = scriptedControls("rmb d ctrl alt@3 s a", 3);
  CHECK(c.aim && c.right && c.crouch && c.dodge && c.back && c.left);
  c = scriptedControls("q w@x", 1); // unknown keys and bad frames are skipped
  CHECK(!c.forward && !c.back && !c.left && !c.right && !c.jump);
}
