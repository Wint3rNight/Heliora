#include "PhysicsWorld.h"
#include "TestRunner.h"

#include <glm/gtc/quaternion.hpp>

#include <stdexcept>
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

TEST(debug_lines_outline_a_body) {
  PhysicsWorld world;
  PhysicsBodyDesc box;
  box.motion = PhysicsMotion::Static;
  box.halfExtents = glm::vec3(1.0f, 2.0f, 3.0f);
  box.position = glm::vec3(10.0f, 0.0f, 0.0f);
  world.addBody(box);

  const std::vector<PhysicsDebugVertex> &lines =
      world.debugLines(glm::vec3(0.0f, 5.0f, 20.0f));
  CHECK(!lines.empty());
  CHECK(lines.size() % 2 == 0); // pairs of end points
  int outside = 0;
  for (const PhysicsDebugVertex &v : lines)
    if (glm::any(glm::greaterThan(glm::abs(v.position - box.position),
                                  box.halfExtents + 0.01f)))
      ++outside;
  CHECK(outside == 0);
}

TEST(body_starts_with_its_velocity) {
  PhysicsWorld world;
  PhysicsBodyDesc ball;
  ball.shape = PhysicsShape::Sphere;
  ball.radius = 0.2f;
  ball.linearVelocity = glm::vec3(20.0f, 0.0f, 0.0f);
  const PhysicsBodyId id = world.addBody(ball);
  world.step(kStep);

  // 20 m/s for one step; gravity only pulls it down, so x is all velocity.
  CHECK_NEAR(world.pose(id).position.x, 20.0 * kStep, 1e-3);
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

TEST(density_sets_mass) {
  PhysicsWorld world;
  PhysicsBodyDesc wood; // a 1 m box
  wood.density = 600.0f;
  CHECK_NEAR(world.mass(world.addBody(wood)), 600.0, 6.0); // within 1%

  PhysicsBodyDesc plain; // the default, about water
  CHECK_NEAR(world.mass(world.addBody(plain)), 1000.0, 10.0);

  PhysicsBodyDesc floor;
  floor.motion = PhysicsMotion::Static;
  CHECK(world.mass(world.addBody(floor)) == 0.0f); // static: no mass
}

namespace {

// A box from lo to hi as a mesh part: 8 corners and 12 triangles, wound so
// they face outwards.
PhysicsMeshPart boxPart(glm::vec3 lo, glm::vec3 hi) {
  PhysicsMeshPart part;
  for (int i = 0; i < 8; ++i)
    part.positions.push_back(glm::vec3(i & 1 ? hi.x : lo.x,
                                       i & 2 ? hi.y : lo.y,
                                       i & 4 ? hi.z : lo.z));
  part.indices = {0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4,
                  2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5};
  return part;
}

std::shared_ptr<PhysicsMeshParts> partsOf(std::vector<PhysicsMeshPart> list) {
  auto parts = std::make_shared<PhysicsMeshParts>();
  parts->parts = std::move(list);
  return parts;
}

PhysicsBodyId dropBall(PhysicsWorld &world, glm::vec3 position) {
  PhysicsBodyDesc ball;
  ball.shape = PhysicsShape::Sphere;
  ball.radius = 0.2f;
  ball.position = position;
  return world.addBody(ball);
}

} // namespace

TEST(hull_mass_comes_from_its_volume) {
  PhysicsWorld world;
  PhysicsBodyDesc cube;
  cube.shape = PhysicsShape::Hulls;
  cube.parts = partsOf({boxPart(glm::vec3(-0.5f), glm::vec3(0.5f))});
  CHECK_NEAR(world.mass(world.addBody(cube)), 1000.0, 20.0); // 1 m^3 of water

  cube.scale = glm::vec3(2.0f); // 8 m^3
  CHECK_NEAR(world.mass(world.addBody(cube)), 8000.0, 160.0);
}

TEST(two_part_model_keeps_its_gap) {
  PhysicsWorld world;
  addFloor(world);
  // Two blocks 1 m apart, as one static body. A single hull around both
  // would fill the gap and catch the ball on top.
  PhysicsBodyDesc model;
  model.shape = PhysicsShape::Hulls;
  model.motion = PhysicsMotion::Static;
  model.parts = partsOf({boxPart({-2.0f, 0.0f, -1.0f}, {-0.5f, 1.0f, 1.0f}),
                         boxPart({0.5f, 0.0f, -1.0f}, {2.0f, 1.0f, 1.0f})});
  world.addBody(model);
  const PhysicsBodyId ball = dropBall(world, {0.0f, 3.0f, 0.0f});
  stepFor(world, 180);

  CHECK(world.pose(ball).position.y < 0.5f); // on the floor, not the blocks
}

TEST(static_mesh_collider_stops_a_ball) {
  PhysicsWorld world;
  // A 10 x 10 m ground made of two triangles facing up, and nothing else.
  PhysicsMeshPart ground;
  ground.positions = {{-5.0f, 0.0f, -5.0f},
                      {-5.0f, 0.0f, 5.0f},
                      {5.0f, 0.0f, 5.0f},
                      {5.0f, 0.0f, -5.0f}};
  ground.indices = {0, 1, 2, 0, 2, 3};
  PhysicsBodyDesc mesh;
  mesh.shape = PhysicsShape::TriangleMesh;
  mesh.motion = PhysicsMotion::Static;
  mesh.parts = partsOf({ground});
  world.addBody(mesh);
  const PhysicsBodyId ball = dropBall(world, {0.0f, 2.0f, 0.0f});
  stepFor(world, 180);

  const float y = world.pose(ball).position.y;
  CHECK(y > 0.1f && y < 0.3f); // resting on the triangles, radius 0.2
}

TEST(flat_points_fall_back_to_a_box) {
  PhysicsWorld world;
  PhysicsMeshPart flat; // a square with no thickness
  flat.positions = {{0.0f, 0.0f, 0.0f},
                    {1.0f, 0.0f, 0.0f},
                    {1.0f, 0.0f, 1.0f},
                    {0.0f, 0.0f, 1.0f}};
  PhysicsBodyDesc desc;
  desc.shape = PhysicsShape::Hulls;
  desc.parts = partsOf({flat});
  const PhysicsBodyId id = world.addBody(desc);
  CHECK(id.isValid());
  CHECK(world.mass(id) > 0.0f && std::isfinite(world.mass(id)));

  PhysicsMeshPart point; // one point, nothing to wrap
  point.positions = {{0.0f, 0.0f, 0.0f}};
  desc.parts = partsOf({point});
  CHECK(world.addBody(desc).isValid());
}

TEST(zero_gravity_body_stays_put) {
  PhysicsWorld::Config config;
  config.gravity = glm::vec3(0.0f);
  PhysicsWorld world(config);
  const PhysicsBodyId box = addBox(world, glm::vec3(0.0f, 5.0f, 0.0f));
  stepFor(world, 60);
  CHECK_NEAR(world.pose(box).position.y, 5.0, 1e-4);
}

TEST(longer_sleep_delay_keeps_a_body_awake_longer) {
  // With the default half second, a box resting on the floor soon sleeps.
  PhysicsWorld quick;
  addFloor(quick);
  const PhysicsBodyId a = addBox(quick, glm::vec3(0.0f, 0.5f, 0.0f));
  int steps = 0;
  for (; steps < 600 && quick.isActive(a); ++steps)
    quick.step(kStep);
  CHECK(!quick.isActive(a));

  PhysicsWorld::Config config;
  config.sleepDelay = 3.0f;
  PhysicsWorld slow(config);
  addFloor(slow);
  const PhysicsBodyId b = addBox(slow, glm::vec3(0.0f, 0.5f, 0.0f));
  stepFor(slow, steps);
  CHECK(slow.isActive(b));
}

TEST(gravity_change_wakes_sleeping_bodies) {
  PhysicsWorld world;
  addFloor(world);
  const PhysicsBodyId box = addBox(world, glm::vec3(0.0f, 0.5f, 0.0f));
  stepFor(world, 120);
  CHECK(!world.isActive(box));

  // Upwards: the box has to wake up to notice, then it lifts off.
  PhysicsWorld::Config config;
  config.gravity = glm::vec3(0.0f, 9.81f, 0.0f);
  world.applySettings(config);
  stepFor(world, 30);
  CHECK(world.pose(box).position.y > 1.0f);
}

TEST(big_contact_limit_gets_enough_scratch_memory) {
  // Raised in the Physics tab, past what the default scratch memory holds.
  PhysicsWorld::Config config;
  config.maxContactConstraints = 60000;
  bool threw = false;
  try {
    PhysicsWorld world(config);
    runHeap(world);
  } catch (const std::exception &) {
    threw = true;
  }
  CHECK(!threw);
}

TEST(cast_ray_finds_the_nearest_body) {
  PhysicsWorld world;
  PhysicsBodyDesc box;
  box.motion = PhysicsMotion::Static;
  box.position = glm::vec3(0.0f, 0.0f, -10.0f);
  box.userData = 9;
  world.addBody(box);
  box.position = glm::vec3(0.0f, 0.0f, -5.0f);
  box.userData = 7;
  world.addBody(box);

  const std::optional<PhysicsRayHit> hit =
      world.castRay(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f), 100.0f);
  CHECK(hit.has_value());
  if (hit) {
    CHECK(hit->userData == 7);
    CHECK_NEAR(hit->distance, 4.5, 1e-3); // the near face of a 1 m box
  }
  // The direction doesn't have to be unit length.
  const std::optional<PhysicsRayHit> longer =
      world.castRay(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -3.0f), 100.0f);
  CHECK_NEAR(longer ? longer->distance : -1.0f, 4.5, 1e-3);
}

TEST(cast_ray_misses_past_max_distance) {
  PhysicsWorld world;
  PhysicsBodyDesc box;
  box.motion = PhysicsMotion::Static;
  box.position = glm::vec3(0.0f, 0.0f, -10.0f);
  world.addBody(box);
  CHECK(!world.castRay(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f), 5.0f));
  CHECK(world.castRay(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f), 20.0f));
}

TEST(set_pose_moves_the_body_without_blending) {
  PhysicsWorld world;
  const PhysicsBodyId box = addBox(world, glm::vec3(0.0f, 5.0f, 0.0f));
  stepFor(world, 10);
  const PhysicsPose target{glm::vec3(3.0f, 2.0f, 1.0f),
                           glm::angleAxis(0.5f, glm::vec3(0.0f, 1.0f, 0.0f))};
  world.setPose(box, target);
  CHECK_NEAR(glm::distance(world.pose(box).position, target.position), 0.0,
             1e-5);
  // Drawn there straight away, not blended in from where it was.
  const PhysicsPose drawn = world.interpolatedPose(box, 0.5f);
  CHECK_NEAR(glm::distance(drawn.position, target.position), 0.0, 1e-5);
  CHECK_NEAR(std::fabs(glm::dot(drawn.rotation, target.rotation)), 1.0, 1e-5);
}
