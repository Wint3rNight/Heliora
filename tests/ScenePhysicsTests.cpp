#include "FixedStepClock.h"
#include "PhysicsWorld.h"
#include "ScenePhysics.h"
#include "SceneNode.h"
#include "TestRunner.h"

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>

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

TEST(body_desc_takes_mass_friction_and_restitution) {
  SceneModelEntry ball = entry(PrimitiveShape::Sphere, PhysicsMotion::Dynamic);
  ball.mass = 5.0f;
  ball.friction = 0.6f;
  ball.restitution = 0.4f;
  const PhysicsBodyDesc desc = bodyDescFor(ball);
  CHECK(desc.mass == 5.0f);
  CHECK(desc.friction == 0.6f);
  CHECK(desc.restitution == 0.4f);

  // Unset keys keep PhysicsBodyDesc's defaults.
  const PhysicsBodyDesc plain =
      bodyDescFor(entry(PrimitiveShape::Box, PhysicsMotion::Dynamic));
  CHECK(plain.mass == PhysicsBodyDesc{}.mass);
  CHECK(plain.friction == PhysicsBodyDesc{}.friction);
}

TEST(materials_have_real_densities) {
  auto density = [](const char *name) {
    const std::optional<PhysicsMaterial> material = findMaterial(name);
    return material ? material->density : -1.0f;
  };
  CHECK(density("wood") == 600.0f);
  CHECK(density("stone") == 2400.0f);
  CHECK(density("metal") == 7800.0f);
  CHECK(density("rubber") == 1100.0f);
  CHECK(density("ice") == 920.0f);
  CHECK(findMaterial("Stone").has_value()); // case doesn't matter
  CHECK(!findMaterial("cheese"));
}

TEST(mass_beats_density_beats_material) {
  SceneModelEntry box = entry(PrimitiveShape::Box, PhysicsMotion::Dynamic);
  box.material = "wood";
  box.density = 2000.0f;
  box.mass = 5.0f;
  CHECK(bodyDescFor(box).mass == 5.0f);

  box.mass.reset();
  CHECK(bodyDescFor(box).mass == 0.0f);
  CHECK(bodyDescFor(box).density == 2000.0f);

  box.density.reset();
  CHECK(bodyDescFor(box).density == 600.0f); // wood

  box.material.reset();
  CHECK(bodyDescFor(box).density == 1000.0f); // the default
}

TEST(material_sets_friction_and_restitution) {
  SceneModelEntry ball = entry(PrimitiveShape::Sphere, PhysicsMotion::Dynamic);
  ball.material = "rubber";
  CHECK(bodyDescFor(ball).friction == 0.9f);
  CHECK(bodyDescFor(ball).restitution == 0.7f);

  ball.friction = 0.1f; // a key in the file beats the material
  CHECK(bodyDescFor(ball).friction == 0.1f);
  CHECK(bodyDescFor(ball).restitution == 0.7f);
}

TEST(auto_collider_follows_motion) {
  auto parts = std::make_shared<PhysicsMeshParts>();
  PhysicsMeshPart part;
  part.positions = {{-1.0f, 0.0f, -1.0f}, {1.0f, 0.0f, -1.0f},
                    {0.0f, 2.0f, 0.0f},   {0.0f, 0.0f, 1.0f}};
  part.indices = {0, 2, 1, 0, 3, 2, 1, 2, 3, 0, 1, 3};
  parts->parts.push_back(part);

  SceneModelEntry rock;
  rock.path = "rock.glb";
  rock.body = PhysicsMotion::Static;
  CHECK(bodyDescFor(rock, parts).shape == PhysicsShape::TriangleMesh);

  rock.body = PhysicsMotion::Dynamic;
  rock.transform.scale = 2.0f;
  const PhysicsBodyDesc hull = bodyDescFor(rock, parts);
  CHECK(hull.shape == PhysicsShape::Hulls);
  CHECK(hull.parts == parts);
  CHECK(hull.scale == glm::vec3(2.0f));

  // A box: one hull from the 8 corners of the model's bounds.
  rock.collider = SceneCollider::Box;
  const PhysicsBodyDesc box = bodyDescFor(rock, parts);
  CHECK(box.shape == PhysicsShape::Hulls);
  CHECK(box.parts && box.parts->parts.size() == 1 &&
        box.parts->parts.front().positions.size() == 8);
}

namespace {

// A dense unit sphere: rings x segments quads, like a scanned rock's mesh.
PhysicsMeshPart denseSphere(int rings, int segments) {
  PhysicsMeshPart part;
  for (int r = 0; r <= rings; ++r)
    for (int s = 0; s <= segments; ++s) {
      const float theta = glm::pi<float>() * r / rings;
      const float phi = glm::two_pi<float>() * s / segments;
      part.positions.push_back({std::sin(theta) * std::cos(phi),
                                std::cos(theta),
                                std::sin(theta) * std::sin(phi)});
    }
  for (int r = 0; r < rings; ++r)
    for (int s = 0; s < segments; ++s) {
      const uint32_t a = r * (segments + 1) + s, b = a + segments + 1;
      part.indices.insert(part.indices.end(), {a, b, a + 1, a + 1, b, b + 1});
    }
  return part;
}

} // namespace

TEST(collision_mesh_is_simplified_to_the_limit) {
  const PhysicsMeshPart dense = denseSphere(100, 200); // 40,000 triangles
  const PhysicsMeshPart simple = simplifiedForCollision(dense, 2000);

  const size_t triangles = simple.indices.size() / 3;
  CHECK(triangles <= 2000);
  CHECK(triangles >= 200); // still a sphere, not a few slivers
  CHECK(simple.positions.size() < dense.positions.size() / 4); // unused gone
  bool inRange = true;
  for (uint32_t index : simple.indices)
    inRange = inRange && index < simple.positions.size();
  CHECK(inRange);
  float farthest = 0.0f;
  for (const glm::vec3 &p : simple.positions)
    farthest = std::max(farthest, glm::length(p));
  CHECK(farthest > 0.95f && farthest < 1.0001f); // the same size
}

TEST(small_collision_mesh_is_left_alone) {
  const PhysicsMeshPart small = denseSphere(4, 8); // 64 triangles
  const PhysicsMeshPart same = simplifiedForCollision(small, 2000);
  CHECK(same.positions == small.positions);
  CHECK(same.indices == small.indices);
}
