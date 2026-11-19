#include "CharacterRoster.h"
#include "CharacterTestScenes.h"
#include "PlayerController.h"
#include "TestRunner.h"

#include <cmath>
#include <memory>

namespace {

using test::addFloor;

constexpr float kStep = 1.0f / 60.0f;
constexpr float kPi = 3.14159265f;

// A body on a floor, possessed by the player's controller. The controller
// is made before the roster, so it outlives what it possesses.
struct Played {
  PhysicsWorld world;
  PlayerController player;
  CharacterRoster roster;
  explicit Played(glm::vec3 feet = glm::vec3(0.0f), float yaw = 0.0f) : player(world) {
    addFloor(world);
    roster.possess(player, roster.add(std::make_unique<CharacterMovement>(world, feet, yaw)));
  }
  const CharacterMovement &character() const { return *player.body(); }
};

// One frame with exactly one physics step, the way the game runs at 60 fps.
void frame(Played &p, const PlayerControls &controls, int steps = 1) {
  p.player.control(controls, kStep);
  for (int i = 0; i < steps; ++i) {
    p.roster.step(kStep);
    p.world.step(kStep);
  }
  p.player.follow(1.0f, kStep);
}

void frames(Played &p, PlayerControls controls, int count) {
  for (int i = 0; i < count; ++i) {
    frame(p, controls);
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

// The same with Shift held: running.
PlayerControls running(bool PlayerControls::*key) {
  PlayerControls controls = pressing(key);
  controls.run = true;
  return controls;
}

// Pixels of mouse movement that turn the camera by this many radians, at
// the default sensitivity.
float pixelsFor(float radians) { return radians / 0.0025f; }

} // namespace

TEST(w_moves_where_the_camera_looks) {
  Played p;
  PlayerControls turn;
  turn.mouse.x = pixelsFor(0.5f * kPi); // turn right, to look down +x
  frame(p, turn);
  frames(p, running(&PlayerControls::forward), 60);
  CHECK(p.character().position().x > 2.0f);
  CHECK(std::fabs(p.character().position().z) < 0.1f);
}

TEST(a_and_d_strafe_across_the_camera) {
  Played p; // the camera looks down -z
  // Long enough to turn, build up speed and plant to go back.
  frames(p, running(&PlayerControls::right), 60);
  CHECK(p.character().position().x > 1.0f);
  frames(p, running(&PlayerControls::left), 120);
  CHECK(p.character().position().x < -1.0f);
  frames(p, running(&PlayerControls::back), 90);
  CHECK(p.character().position().z > 1.0f);
}

TEST(diagonal_is_no_faster) {
  Played p;
  PlayerControls diagonal;
  diagonal.forward = diagonal.right = diagonal.run = true;
  frames(p, diagonal, 240); // up to full speed
  const glm::vec3 v = p.character().velocity();
  CHECK_NEAR(std::sqrt(v.x * v.x + v.z * v.z), 5.0, 0.05);
}

TEST(jump_pressed_on_a_frame_without_a_step_still_jumps) {
  Played p;
  frames(p, PlayerControls{}, 10);
  const float start = p.character().position().y;
  frame(p, pressing(&PlayerControls::jump), 0); // no step
  frame(p, PlayerControls{}, 1);
  frames(p, PlayerControls{}, 10);
  CHECK(p.character().position().y - start > 0.5f);
}

TEST(one_press_is_one_jump) {
  Played p;
  frames(p, PlayerControls{}, 10);
  frame(p, pressing(&PlayerControls::jump), 3); // a slow frame
  frames(p, PlayerControls{}, 120);
  // Landed and stayed down: the press didn't fire again on landing.
  CHECK(p.character().grounded());
  CHECK(p.character().position().y < 0.05f);
}

TEST(w_walks_shift_runs_a_quick_second_press_sprints_and_ctrl_crouches) {
  Played p;
  frames(p, pressing(&PlayerControls::forward), 120);
  CHECK_NEAR(-p.character().velocity().z, 1.4, 0.05);
  frames(p, running(&PlayerControls::forward), 180);
  CHECK_NEAR(-p.character().velocity().z, 5.0, 0.05);
  // Let go of Shift for a moment and press it again: a sprint, for as long
  // as it's held.
  frames(p, pressing(&PlayerControls::forward), 6);
  frames(p, running(&PlayerControls::forward), 300); // the sprint builds for a while
  CHECK_NEAR(-p.character().velocity().z, 8.0, 0.05);
  PlayerControls crouch = pressing(&PlayerControls::forward);
  crouch.crouch = true;
  frames(p, crouch, 90);
  CHECK_NEAR(-p.character().velocity().z, 2.0, 0.05);
  CHECK_NEAR(p.character().height(), 1.1, 1e-5);
}

TEST(dodge_press_dodges) {
  Played p;
  frames(p, PlayerControls{}, 10);
  frame(p, pressing(&PlayerControls::dodge), 0); // no step
  frames(p, PlayerControls{}, 15);
  CHECK_NEAR(p.character().position().z, -5.0, 0.1); // forward
}

TEST(aiming_turns_the_character_to_the_camera) {
  Played p;
  PlayerControls turn;
  turn.mouse.x = pixelsFor(-kPi * 0.5f); // turn left, to look down -x
  frame(p, turn);
  frames(p, pressing(&PlayerControls::aim), 60);
  CHECK_NEAR(glm::distance(yawForward(p.character().yaw()),
                           glm::vec3(-1, 0, 0)), 0.0, 1e-3);
}

TEST(camera_follows_the_character) {
  Played p;
  frames(p, pressing(&PlayerControls::forward), 60);
  frames(p, PlayerControls{}, 60); // stop and let it catch up
  const glm::vec3 feet = p.character().position();
  CHECK_NEAR(glm::distance(p.player.camera().pivot(), feet + glm::vec3(0, 1.5f, 0)),
             0.0, 0.01);
}

TEST(mouse_up_looks_up_and_the_wheel_zooms) {
  Played p;
  const float pitch = p.player.camera().pitch();
  PlayerControls look;
  look.mouse.y = -40.0f; // up, screen style
  look.wheel = 1.0f;
  frame(p, look);
  CHECK_NEAR(p.player.camera().pitch() - pitch, 0.1, 1e-5);
  CHECK_NEAR(p.player.camera().distance(), 3.5, 1e-5);
}

TEST(camera_starts_behind_the_character) {
  Played p(glm::vec3(2.0f, 0.0f, 0.0f), 0.5f * kPi); // faces -x
  CHECK_NEAR(p.player.camera().yaw(), 0.5 * kPi, 1e-5);
  CHECK(p.player.camera().position().x > 4.0f); // behind it, on the +x side
}

TEST(new_camera_tuning_applies_while_playing) {
  Played p;
  ThirdPersonCameraTuning close;
  close.maxDistance = 3.0f;
  p.player.setCameraTuning(close);
  CHECK_NEAR(p.player.camera().distance(), 3.0, 1e-5); // was 4, now the most
}

TEST(the_camera_goes_with_the_body_it_possesses) {
  Played p;
  const size_t other =
      p.roster.add(std::make_unique<CharacterMovement>(p.world, glm::vec3(6.0f, 0.0f, 0.0f), 0.0f));
  p.roster.possess(p.player, other);
  frames(p, pressing(&PlayerControls::forward), 60);
  CHECK(p.roster.body(other).position().z < -1.0f);
  CHECK_NEAR(p.roster.body(0).position().z, 0.0, 0.01); // the first stays
  frames(p, PlayerControls{}, 60); // stop and let the camera catch up
  CHECK_NEAR(glm::distance(p.player.camera().pivot(),
                           p.roster.body(other).position() + glm::vec3(0, 1.5f, 0)),
             0.0, 0.01);
}

TEST(with_no_body_the_controls_move_only_the_camera) {
  PhysicsWorld world;
  addFloor(world);
  PlayerController player(world);
  PlayerControls look;
  look.forward = true;
  look.mouse.x = 40.0f;
  player.control(look, kStep);
  player.follow(1.0f, kStep); // nothing to follow: no crash
  CHECK(player.body() == nullptr);
  CHECK(player.decide().move == glm::vec3(0.0f));
}

TEST(capture_scripts_hold_keys_or_press_them_on_a_frame) {
  PlayerControls c = scriptedControls("w shift", 5);
  CHECK(c.forward && c.run && !c.sprint && !c.back && !c.jump);
  CHECK(scriptedControls("w sprint", 5).sprint); // no double tap to script
  CHECK(!scriptedControls("space@120", 119).jump);
  CHECK(scriptedControls("space@120", 120).jump);
  CHECK(!scriptedControls("space@120", 121).jump);
  CHECK(!scriptedControls("w@10-20", 9).forward);
  CHECK(scriptedControls("w@10-20", 10).forward);
  CHECK(scriptedControls("w@10-20", 20).forward);
  CHECK(!scriptedControls("w@10-20", 21).forward);
  c = scriptedControls("rmb d ctrl alt@3 s a", 3);
  CHECK(c.aim && c.right && c.crouch && c.dodge && c.back && c.left);
  c = scriptedControls("lmb@4", 4); // fire, held like the aim
  CHECK(c.fire && !c.aim);
  CHECK(!scriptedControls("lmb@4", 5).fire);
  c = scriptedControls("mu@2-3 mr@3", 3); // the mouse: 8 px a frame up, down, left, right
  CHECK_NEAR(c.mouse.y, -8.0, 1e-6);
  CHECK_NEAR(c.mouse.x, 8.0, 1e-6);
  c = scriptedControls("md ml", 0);
  CHECK_NEAR(c.mouse.y, 8.0, 1e-6);
  CHECK_NEAR(c.mouse.x, -8.0, 1e-6);
  c = scriptedControls("z w@x", 1); // unknown keys and bad frames are skipped
  CHECK(!c.forward && !c.back && !c.left && !c.right && !c.jump);
}

TEST(shift_runs_and_a_press_soon_after_letting_go_sprints) {
  GaitKeys keys;
  CHECK(keys.update(false, kStep) == Gait::Walk);
  for (int i = 0; i < 30; ++i)
    CHECK(keys.update(true, kStep) == Gait::Run);
  // Let go for 0.2 s, then press again: a sprint until it's let go.
  for (int i = 0; i < 12; ++i)
    CHECK(keys.update(false, kStep) == Gait::Walk);
  for (int i = 0; i < 30; ++i)
    CHECK(keys.update(true, kStep) == Gait::Sprint);
  CHECK(keys.update(false, kStep) == Gait::Walk);
  // Let go for longer than the double tap: a run again.
  for (int i = 0; i < 30; ++i)
    keys.update(false, kStep);
  CHECK(keys.update(true, kStep) == Gait::Run);
  // A slow frame doesn't make a tap out of a long gap.
  keys.update(false, kStep);
  keys.update(false, 0.5f);
  CHECK(keys.update(true, kStep) == Gait::Run);
}

TEST(firing_from_the_hip_turns_him_to_the_camera_while_armed_and_a_second_after) {
  Played p;
  // Unarmed, the trigger does nothing to where he faces.
  frame(p, pressing(&PlayerControls::fire));
  CHECK(!p.player.decide().aim);
  p.player.setArmed(true);
  frame(p, pressing(&PlayerControls::fire));
  CHECK(p.player.decide().aim);
  CHECK(p.player.hipFiring());
  // Let go: still facing the camera for about a second, then not.
  frames(p, PlayerControls{}, 54);
  CHECK(p.player.decide().aim);
  frames(p, PlayerControls{}, 12);
  CHECK(!p.player.decide().aim);
  CHECK(!p.player.hipFiring());
  // Aiming with the right mouse isn't hip fire.
  frame(p, pressing(&PlayerControls::aim));
  CHECK(p.player.decide().aim);
  CHECK(!p.player.hipFiring());
}

TEST(capture_scripts_press_the_gun_keys) {
  const PlayerControls c = scriptedControls("1@3 2@4 h@5 r@6 b@7 e@8", 3);
  CHECK(c.pistol && !c.longGun);
  CHECK(scriptedControls("1@3 2@4 h@5 r@6 b@7 e@8", 4).longGun);
  CHECK(scriptedControls("1@3 2@4 h@5 r@6 b@7 e@8", 5).holster);
  CHECK(scriptedControls("1@3 2@4 h@5 r@6 b@7 e@8", 6).reload);
  CHECK(scriptedControls("1@3 2@4 h@5 r@6 b@7 e@8", 7).fireMode);
  CHECK(scriptedControls("1@3 2@4 h@5 r@6 b@7 e@8", 8).use);
}
