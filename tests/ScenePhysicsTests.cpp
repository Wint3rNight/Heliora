#include "FixedStepClock.h"
#include "PhysicsWorld.h"
#include "ScenePhysics.h"
#include "SceneNode.h"
#include "TestRunner.h"

namespace {

SceneModelEntry entry(PrimitiveShape shape, PhysicsMotion body) {
  SceneModelEntry e;
  e.shape = shape;
  e.body = body;
  return e;
}

} // namespace

TEST(body_desc_matches_box_size_scale_and_pose) {
  SceneModelEntry box = entry(PrimitiveShape::Box, PhysicsMotion::Dynamic);
  box.size = glm::vec3(2.0f, 1.0f, 3.0f);
  box.transform.scale = 2.0f;
  box.transform.position = glm::vec3(1.0f, 2.0f, 3.0f);
  box.transform.rotationDeg = glm::vec3(10.0f, 20.0f, 30.0f);

  const PhysicsBodyDesc desc = bodyDescFor(box);
  CHECK(desc.shape == PhysicsShape::Box);
  CHECK(desc.motion == PhysicsMotion::Dynamic);
  CHECK(desc.halfExtents == glm::vec3(2.0f, 1.0f, 3.0f)); // size / 2 * scale
  CHECK(desc.position == box.transform.position);
  CHECK(desc.rotation == rotationFromDegrees(box.transform.rotationDeg));
}

TEST(body_desc_scales_sphere_and_capsule) {
  SceneModelEntry sphere = entry(PrimitiveShape::Sphere, PhysicsMotion::Dynamic);
  sphere.radius = 0.3f;
  sphere.transform.scale = 2.0f;
  CHECK(bodyDescFor(sphere).shape == PhysicsShape::Sphere);
  CHECK_NEAR(bodyDescFor(sphere).radius, 0.6, 1e-6);

  // Scene height is end to end; Jolt wants half the straight part.
  SceneModelEntry capsule = entry(PrimitiveShape::Capsule, PhysicsMotion::Dynamic);
  capsule.radius = 0.3f;
  capsule.height = 1.8f;
  capsule.transform.scale = 2.0f;
  const PhysicsBodyDesc desc = bodyDescFor(capsule);
  CHECK(desc.shape == PhysicsShape::Capsule);
  CHECK_NEAR(desc.radius, 0.6, 1e-6);
  CHECK_NEAR(desc.halfHeight, 1.2, 1e-6); // (1.8 / 2 - 0.3) * 2
}

TEST(plane_becomes_a_static_slab_under_its_surface) {
  SceneModelEntry plane = entry(PrimitiveShape::Plane, PhysicsMotion::Static);
  plane.size = glm::vec3(10.0f, 1.0f, 6.0f);
  plane.transform.position = glm::vec3(0.0f, 2.0f, 0.0f);

  const PhysicsBodyDesc flat = bodyDescFor(plane);
  CHECK(flat.shape == PhysicsShape::Box);
  CHECK(flat.motion == PhysicsMotion::Static);
  CHECK(flat.halfExtents == glm::vec3(5.0f, 0.5f, 3.0f));
  CHECK_NEAR(flat.position.y, 1.5, 1e-6); // top face exactly at y = 2

  // Tilted, the slab still sits behind the visible face.
  plane.transform.rotationDeg = glm::vec3(90.0f, 0.0f, 0.0f);
  const glm::vec3 up = rotationFromDegrees(plane.transform.rotationDeg) *
                       glm::vec3(0.0f, 1.0f, 0.0f);
  const PhysicsBodyDesc tilted = bodyDescFor(plane);
  CHECK(glm::length(tilted.position - (plane.transform.position - up * 0.5f)) <
        1e-5f);
}

TEST(scene_box_falls_onto_the_plane) {
  // What the game does each frame, minus the GPU: clock, steps, then the
  // interpolated poses of moved bodies go into their scene nodes.
  SceneModelEntry plane = entry(PrimitiveShape::Plane, PhysicsMotion::Static);
  plane.size = glm::vec3(20.0f, 1.0f, 20.0f);
  SceneModelEntry box = entry(PrimitiveShape::Box, PhysicsMotion::Dynamic);
  box.transform.position = glm::vec3(0.0f, 3.0f, 0.0f);

  PhysicsWorld world;
  SceneNode node;
  node.setTransform(box.transform);
  world.addBody(bodyDescFor(plane));
  PhysicsBodyDesc boxDesc = bodyDescFor(box);
  boxDesc.userData = 7;
  world.addBody(boxDesc);

  FixedStepClock clock;
  bool onlyTheBoxMoved = true;
  for (int frame = 0; frame < 3 * 144; ++frame) { // 3 s at 144 fps
    const int steps = clock.advance(1.0 / 144.0);
    for (int i = 0; i < steps; ++i)
      world.step(static_cast<float>(clock.stepSeconds()));
    for (const PhysicsMovedPose &moved : world.movedPoses(clock.alpha())) {
      onlyTheBoxMoved = onlyTheBoxMoved && moved.userData == 7;
      node.setPose(moved.pose.position, moved.pose.rotation);
    }
    node.update(glm::mat4(1.0f));
  }

  CHECK(onlyTheBoxMoved);
  CHECK_NEAR(node.getGlobalTransform()[3].y, 0.5, 0.025);
}
