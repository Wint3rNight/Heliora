#include "ScenePhysics.h"

namespace {

// A plane has no thickness to collide with, so its body is a slab whose top
// face is the plane. 1 m thick so fast objects can't skip through it in one
// step.
constexpr float kSlabHalfThickness = 0.5f;

} // namespace

PhysicsBodyDesc bodyDescFor(const SceneModelEntry &entry) {
  const float scale = entry.transform.scale;
  PhysicsBodyDesc desc;
  desc.motion = *entry.body;
  desc.position = entry.transform.position;
  desc.rotation = rotationFromDegrees(entry.transform.rotationDeg);
  switch (*entry.shape) {
  case PrimitiveShape::Box:
    desc.shape = PhysicsShape::Box;
    desc.halfExtents = entry.size * 0.5f * scale;
    break;
  case PrimitiveShape::Sphere:
    desc.shape = PhysicsShape::Sphere;
    desc.radius = entry.radius * scale;
    break;
  case PrimitiveShape::Capsule:
    // Scene height is end to end, Jolt wants half the straight part.
    desc.shape = PhysicsShape::Capsule;
    desc.radius = entry.radius * scale;
    desc.halfHeight = (entry.height * 0.5f - entry.radius) * scale;
    break;
  case PrimitiveShape::Plane:
    desc.shape = PhysicsShape::Box;
    desc.halfExtents = glm::vec3(entry.size.x * 0.5f * scale,
                                 kSlabHalfThickness,
                                 entry.size.z * 0.5f * scale);
    desc.position -= desc.rotation * glm::vec3(0.0f, kSlabHalfThickness, 0.0f);
    break;
  }
  return desc;
}
