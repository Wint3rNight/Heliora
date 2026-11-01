#include "PhysicsWorld.h"
#include "TestRunner.h"

#include <glm/gtc/quaternion.hpp>

#include <vector>

namespace {

constexpr float kStep = 1.0f / 60.0f;

// A static floor whose top surface is at y = 0.
PhysicsBodyId addFloor(PhysicsWorld &world) {
  PhysicsBodyDesc floor;
  floor.shape = PhysicsShape::Box;
  floor.halfExtents = glm::vec3(50.0f, 0.5f, 50.0f);
  floor.motion = PhysicsMotion::Static;
  floor.position = glm::vec3(0.0f, -0.5f, 0.0f);
  return world.addBody(floor);
}

PhysicsBodyId addBox(PhysicsWorld &world, glm::vec3 position,
                     glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f)) {
  PhysicsBodyDesc box;
  box.shape = PhysicsShape::Box;
  box.halfExtents = glm::vec3(0.5f);
  box.motion = PhysicsMotion::Dynamic;
  box.position = position;
  box.rotation = rotation;
  return world.addBody(box);
}

// Steps the world and returns how many steps dropped contacts.
uint32_t stepFor(PhysicsWorld &world, int steps) {
  const uint32_t before = world.droppedContactSteps();
  for (int i = 0; i < steps; ++i)
    world.step(kStep);
  return world.droppedContactSteps() - before;
}

// Twenty boxes dropped in a tilted heap, so they collide with each other.
std::vector<PhysicsPose> runHeap(PhysicsWorld &world) {
  addFloor(world);
  std::vector<PhysicsBodyId> boxes;
  for (int i = 0; i < 20; ++i) {
    const glm::vec3 position(0.3f * (i % 4), 1.0f + 1.2f * i,
                             0.2f * (i % 3));
    const glm::quat rotation =
        glm::angleAxis(0.4f * i, glm::normalize(glm::vec3(1.0f, 2.0f, 3.0f)));
    boxes.push_back(addBox(world, position, rotation));
  }
  world.optimizeBroadPhase();
  stepFor(world, 120);
  std::vector<PhysicsPose> poses;
  for (PhysicsBodyId id : boxes)
    poses.push_back(world.pose(id));
  return poses;
}

} // namespace

TEST(pose_round_trips_position_and_rotation) {
  PhysicsWorld world;
  PhysicsBodyDesc desc;
  desc.motion = PhysicsMotion::Static;
  desc.position = glm::vec3(1.0f, 2.0f, 3.0f);
  desc.rotation =
      glm::angleAxis(0.5f, glm::normalize(glm::vec3(1.0f, 2.0f, 3.0f)));
  const PhysicsBodyId id = world.addBody(desc);

  CHECK(id.isValid());
  const PhysicsPose pose = world.pose(id);
  CHECK_NEAR(pose.position.x, 1.0, 1e-5);
  CHECK_NEAR(pose.position.y, 2.0, 1e-5);
  CHECK_NEAR(pose.position.z, 3.0, 1e-5);
  // q and -q are the same rotation, so compare with |dot| instead of ==.
  CHECK_NEAR(std::fabs(glm::dot(pose.rotation, desc.rotation)), 1.0, 1e-5);
}

TEST(box_comes_to_rest_on_floor) {
  PhysicsWorld world;
  addFloor(world);
  const PhysicsBodyId box = addBox(world, glm::vec3(0.0f, 3.0f, 0.0f));

  CHECK(stepFor(world, 180) == 0); // 3 seconds

  // Resting on the floor: the centre is half a box above y = 0. Jolt lets
  // resting bodies sink up to its penetration slop (2 cm).
  CHECK_NEAR(world.pose(box).position.y, 0.5, 0.025);
  // At rest Jolt puts it to sleep, so it costs nothing per step.
  CHECK(!world.isActive(box));
}

TEST(five_box_stack_stands) {
  PhysicsWorld world;
  addFloor(world);
  PhysicsBodyId top;
  for (int i = 0; i < 5; ++i)
    top = addBox(world, glm::vec3(0.0f, 0.5f + i, 0.0f));

  CHECK(stepFor(world, 600) == 0); // 10 seconds

  const glm::vec3 p = world.pose(top).position;
  CHECK_NEAR(p.x, 0.0, 0.05);
  CHECK_NEAR(p.y, 4.5, 0.1);
  CHECK_NEAR(p.z, 0.0, 0.05);
  CHECK(world.activeBodyCount() == 0);
}

TEST(same_inputs_give_same_result) {
  PhysicsWorld first;
  PhysicsWorld second;
  const std::vector<PhysicsPose> a = runHeap(first);
  const std::vector<PhysicsPose> b = runHeap(second);

  CHECK(a.size() == b.size());
  // Exactly equal, not nearly: Jolt is deterministic for the same binary
  // and the same order of calls, even with several worker threads.
  bool identical = a.size() == b.size();
  for (size_t i = 0; identical && i < a.size(); ++i)
    identical = a[i].position == b[i].position && a[i].rotation == b[i].rotation;
  CHECK(identical);
  // The top box must have fallen but still be above the floor, or the
  // comparison above proves nothing.
  const float topStartY = 1.0f + 1.2f * 19;
  CHECK(a.back().position.y < topStartY - 1.0f);
  CHECK(a.back().position.y > 0.25f);
}

TEST(removed_body_is_gone) {
  PhysicsWorld world;
  addFloor(world);
  const uint32_t before = world.bodyCount();
  const PhysicsBodyId box = addBox(world, glm::vec3(0.0f, 1.0f, 0.0f));
  CHECK(world.bodyCount() == before + 1);

  world.removeBody(box);
  CHECK(world.bodyCount() == before);
}

TEST(interpolation_blends_between_the_last_two_steps) {
  PhysicsWorld world;
  addFloor(world);
  const PhysicsBodyId box = addBox(world, glm::vec3(0.0f, 5.0f, 0.0f));
  world.step(kStep);
  const PhysicsPose before = world.pose(box);
  world.step(kStep);
  const PhysicsPose after = world.pose(box);

  CHECK(before.position.y > after.position.y); // still falling
  CHECK(world.interpolatedPose(box, 0.0f).position == before.position);
  CHECK(world.interpolatedPose(box, 1.0f).position == after.position);
  CHECK_NEAR(world.interpolatedPose(box, 0.5f).position.y,
             (before.position.y + after.position.y) / 2.0, 1e-5);
}

TEST(new_body_interpolates_from_its_spawn_pose) {
  PhysicsWorld world;
  const PhysicsBodyId box = addBox(world, glm::vec3(1.0f, 5.0f, 2.0f));

  CHECK(world.interpolatedPose(box, 0.7f).position ==
        glm::vec3(1.0f, 5.0f, 2.0f));
}

TEST(resting_body_stops_interpolating) {
  PhysicsWorld world;
  addFloor(world);
  const PhysicsBodyId box = addBox(world, glm::vec3(0.0f, 1.0f, 0.0f));
  stepFor(world, 180);
  CHECK(!world.isActive(box));

  // The resting pose at any alpha, or the box would twitch while asleep as
  // alpha changes from frame to frame. 1e-6 is for float rounding.
  const PhysicsPose rest = world.pose(box);
  for (float alpha : {0.0f, 0.3f, 0.9f}) {
    const PhysicsPose drawn = world.interpolatedPose(box, alpha);
    CHECK(glm::length(drawn.position - rest.position) < 1e-6f);
    CHECK(std::fabs(glm::dot(drawn.rotation, rest.rotation)) > 1.0f - 1e-6f);
  }
}

TEST(body_is_listed_in_the_step_it_falls_asleep) {
  PhysicsWorld world;
  addFloor(world);
  const PhysicsBodyId box = addBox(world, glm::vec3(0.0f, 1.0f, 0.0f));
  for (int i = 0; i < 600 && world.isActive(box); ++i)
    world.step(kStep);
  CHECK(!world.isActive(box));

  // The step that put it to sleep still lists it, so its node gets the
  // final resting pose instead of staying where the last frame drew it.
  bool listed = false;
  for (PhysicsBodyId moved : world.movedBodies())
    listed = listed || moved.value == box.value;
  CHECK(listed);
}

TEST(moved_bodies_lists_only_what_moved) {
  PhysicsWorld world;
  const PhysicsBodyId floor = addFloor(world);
  const PhysicsBodyId box = addBox(world, glm::vec3(0.0f, 1.0f, 0.0f));
  world.step(kStep);

  auto listed = [&](PhysicsBodyId id) {
    for (PhysicsBodyId moved : world.movedBodies())
      if (moved.value == id.value)
        return true;
    return false;
  };
  CHECK(listed(box));
  CHECK(!listed(floor));

  // Once the box is asleep, one more step drops it from the list.
  stepFor(world, 180);
  CHECK(!world.isActive(box));
  world.step(kStep);
  CHECK(world.movedBodies().empty());
}

TEST(removed_body_leaves_the_moved_list) {
  PhysicsWorld world;
  const PhysicsBodyId box = addBox(world, glm::vec3(0.0f, 5.0f, 0.0f));
  world.step(kStep);
  world.removeBody(box);

  CHECK(world.movedBodies().empty());
}

TEST(moved_poses_carry_user_data_and_blend) {
  PhysicsWorld world;
  PhysicsBodyDesc desc;
  desc.position = glm::vec3(0.0f, 5.0f, 0.0f);
  desc.userData = 42;
  const PhysicsBodyId box = world.addBody(desc);
  world.step(kStep);
  world.step(kStep);

  const PhysicsPose expected = world.interpolatedPose(box, 0.5f);
  const std::vector<PhysicsMovedPose> &moved = world.movedPoses(0.5f);
  CHECK(moved.size() == 1);
  if (moved.size() == 1) {
    CHECK(moved[0].userData == 42);
    CHECK(moved[0].pose.position == expected.position);
  }
}

TEST(too_small_limits_are_reported) {
  // Room for only 4 contacts: the heap needs far more, and Jolt silently
  // drops the rest unless step() reports it.
  PhysicsWorld::Config config;
  config.maxBodyPairs = 4;
  config.maxContactConstraints = 4;
  PhysicsWorld world(config);
  runHeap(world);

  CHECK(world.droppedContactSteps() > 0);
}
