#include "CharacterTestScenes.h"
#include "TestRunner.h"
#include "ThirdPersonCamera.h"

#include <cmath>

namespace {

using test::addFloor;
using test::addStaticBox;

constexpr float kFrame = 1.0f / 60.0f;
constexpr float kPi = 3.14159265f;

void frames(ThirdPersonCamera &camera, const PhysicsWorld &world,
            glm::vec3 feet, bool aiming, int count) {
  for (int i = 0; i < count; ++i)
    camera.update(world, feet, aiming, kFrame);
}

float near(glm::vec3 a, glm::vec3 b) { return glm::distance(a, b); }

} // namespace

TEST(camera_forward_matches_the_character_yaw) {
  CHECK_NEAR(near(cameraForward(0.0f, 0.0f), glm::vec3(0, 0, -1)), 0.0, 1e-6);
  CHECK_NEAR(near(cameraForward(-0.5f * kPi, 0.0f), glm::vec3(1, 0, 0)), 0.0, 1e-6);
  CHECK_NEAR(near(cameraForward(0.0f, 0.5f * kPi), glm::vec3(0, 1, 0)), 0.0, 1e-6);
}

TEST(camera_sits_behind_the_pivot) {
  PhysicsWorld world;
  ThirdPersonCamera camera;
  camera.snapTo(world, glm::vec3(1.0f, 0.0f, 2.0f), 0.0f, -0.25f);
  const glm::vec3 pivot(1.0f, 1.5f, 2.0f);
  const glm::vec3 expected(1.0f, 1.5f + 4.0f * std::sin(0.25f),
                           2.0f + 4.0f * std::cos(0.25f));
  CHECK_NEAR(near(camera.pivot(), pivot), 0.0, 1e-5);
  CHECK_NEAR(near(camera.position(), expected), 0.0, 1e-4);
  frames(camera, world, glm::vec3(1.0f, 0.0f, 2.0f), false, 30);
  CHECK_NEAR(near(camera.position(), expected), 0.0, 1e-4);
}

TEST(orbit_turns_by_the_mouse) {
  PhysicsWorld world;
  ThirdPersonCamera camera;
  camera.snapTo(world, glm::vec3(0.0f), 0.0f, 0.0f);
  camera.orbit(100.0f, 0.0f); // right turns right: yaw goes down
  CHECK_NEAR(camera.yaw(), -0.25, 1e-6);
  camera.orbit(0.0f, -40.0f); // up looks up
  CHECK_NEAR(camera.pitch(), 0.1, 1e-6);
  camera.orbit(1e5f, 0.0f); // many turns stay in -pi to pi
  CHECK(std::fabs(camera.yaw()) <= kPi + 1e-6f);
}

TEST(camera_pitch_stays_inside_its_limits) {
  PhysicsWorld world;
  ThirdPersonCamera camera;
  camera.snapTo(world, glm::vec3(0.0f), 0.0f, 0.0f);
  camera.orbit(0.0f, -1e6f);
  CHECK_NEAR(camera.pitch(), glm::radians(75.0f), 1e-5);
  camera.orbit(0.0f, 1e6f);
  CHECK_NEAR(camera.pitch(), -glm::radians(75.0f), 1e-5);
  // Snapping can't sneak past them either.
  camera.snapTo(world, glm::vec3(0.0f), 0.0f, 3.0f);
  CHECK_NEAR(camera.pitch(), glm::radians(75.0f), 1e-5);
}

TEST(zoom_stays_between_its_limits) {
  ThirdPersonCamera camera;
  camera.zoom(1.0f);
  CHECK_NEAR(camera.distance(), 3.5, 1e-6);
  camera.zoom(100.0f);
  CHECK_NEAR(camera.distance(), 2.0, 1e-6);
  camera.zoom(-100.0f);
  CHECK_NEAR(camera.distance(), 8.0, 1e-6);
  ThirdPersonCameraTuning tighter;
  tighter.maxDistance = 3.0f;
  camera.setTuning(tighter);
  CHECK_NEAR(camera.distance(), 3.0, 1e-6);
}

TEST(wall_pulls_the_camera_in_at_once) {
  PhysicsWorld world;
  ThirdPersonCamera camera;
  camera.snapTo(world, glm::vec3(0.0f), 0.0f, 0.0f);
  CHECK_NEAR(camera.position().z, 4.0, 1e-4);
  // A wall whose face is 2 m behind the pivot.
  const PhysicsBodyId wall =
      addStaticBox(world, glm::vec3(0.0f, 1.5f, 2.5f), glm::vec3(3.0f, 3.0f, 0.5f));
  frames(camera, world, glm::vec3(0.0f), false, 1);
  CHECK_NEAR(camera.position().z, 2.0 - 0.2, 0.01); // the ball's radius short

  // And eases back out once it's gone.
  world.removeBody(wall);
  frames(camera, world, glm::vec3(0.0f), false, 1);
  CHECK(camera.position().z < 1.95f); // eases, so one frame barely moves
  // A closer wall while it's still easing out pulls it in at once too.
  frames(camera, world, glm::vec3(0.0f), false, 10);
  const PhysicsBodyId closer =
      addStaticBox(world, glm::vec3(0.0f, 1.5f, 1.5f), glm::vec3(3.0f, 3.0f, 0.5f));
  frames(camera, world, glm::vec3(0.0f), false, 1);
  CHECK_NEAR(camera.position().z, 1.0 - 0.2, 0.01);
  world.removeBody(closer);
  frames(camera, world, glm::vec3(0.0f), false, 180);
  CHECK_NEAR(camera.position().z, 4.0, 0.02);
}

TEST(snapping_in_front_of_a_wall_starts_pulled_in) {
  PhysicsWorld world;
  addStaticBox(world, glm::vec3(0.0f, 1.5f, 2.5f), glm::vec3(3.0f, 3.0f, 0.5f));
  ThirdPersonCamera camera;
  camera.snapTo(world, glm::vec3(0.0f), 0.0f, 0.0f);
  CHECK_NEAR(camera.position().z, 1.8, 0.01);
}

TEST(floor_keeps_the_camera_above_it) {
  PhysicsWorld world;
  addFloor(world);
  ThirdPersonCamera camera;
  // Looking up 52 degrees would put the camera 1.6 m under the floor.
  camera.snapTo(world, glm::vec3(0.0f), 0.0f, 0.9f);
  CHECK(camera.position().y > 0.19f);
  CHECK(camera.position().y < 0.25f);
}

TEST(camera_sees_past_the_character) {
  PhysicsWorld world;
  addFloor(world);
  PhysicsCharacterDesc character;
  world.addCharacter(character);
  ThirdPersonCamera camera;
  camera.snapTo(world, glm::vec3(0.0f), 0.0f, 0.0f);
  CHECK_NEAR(near(camera.position(), camera.pivot()), 4.0, 1e-3);
}

TEST(aiming_slides_over_the_right_shoulder) {
  PhysicsWorld world;
  ThirdPersonCamera camera;
  camera.snapTo(world, glm::vec3(0.0f), 0.0f, 0.0f);
  frames(camera, world, glm::vec3(0.0f), true, 3);
  CHECK(camera.position().x > 0.05f); // on its way, not there at once
  CHECK(camera.position().x < 0.5f);
  frames(camera, world, glm::vec3(0.0f), true, 27);
  CHECK_NEAR(near(camera.position(), glm::vec3(0.55f, 1.6f, 2.5f)), 0.0, 1e-3);
  CHECK_NEAR(near(camera.forward(), glm::vec3(0, 0, -1)), 0.0, 1e-6);
  frames(camera, world, glm::vec3(0.0f), false, 30); // and back, just as quick
  CHECK_NEAR(near(camera.position(), glm::vec3(0.0f, 1.5f, 4.0f)), 0.0, 1e-3);
}

TEST(pivot_follows_with_a_little_lag) {
  PhysicsWorld world;
  ThirdPersonCamera camera;
  camera.snapTo(world, glm::vec3(0.0f), 0.0f, 0.0f);
  frames(camera, world, glm::vec3(0.1f, 0.0f, 0.0f), false, 1);
  // A 0.08 s follow time closes 1 - e^(-(1/60)/0.08) = 19% of the gap.
  CHECK_NEAR(camera.pivot().x, 0.1 * (1.0 - std::exp(-kFrame / 0.08)), 1e-4);
  frames(camera, world, glm::vec3(0.1f, 0.0f, 0.0f), false, 60);
  CHECK_NEAR(camera.pivot().x, 0.1, 1e-4);
}

TEST(pivot_never_trails_further_than_max_lag) {
  PhysicsWorld world;
  ThirdPersonCamera camera;
  camera.snapTo(world, glm::vec3(0.0f), 0.0f, 0.0f);
  frames(camera, world, glm::vec3(10.0f, 0.0f, 0.0f), false, 1);
  CHECK(near(camera.pivot(), glm::vec3(10.0f, 1.5f, 0.0f)) <= 0.6f + 1e-4f);
}

TEST(view_looks_from_the_eye_along_forward) {
  PhysicsWorld world;
  ThirdPersonCamera camera;
  camera.snapTo(world, glm::vec3(3.0f, 0.0f, -2.0f), 0.7f, -0.3f);
  const glm::mat4 view = camera.view();
  const glm::vec3 eye = glm::vec3(view * glm::vec4(camera.position(), 1.0f));
  const glm::vec3 ahead =
      glm::vec3(view * glm::vec4(camera.position() + camera.forward(), 1.0f));
  CHECK_NEAR(glm::length(eye), 0.0, 1e-4);
  CHECK_NEAR(near(ahead, glm::vec3(0, 0, -1)), 0.0, 1e-4);
}
