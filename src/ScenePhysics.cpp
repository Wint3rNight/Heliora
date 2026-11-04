#include "ScenePhysics.h"

#include <algorithm>
#include <cctype>

namespace {

// A plane has no thickness to collide with, so its body is a slab whose top
// face is the plane. 1 m thick so fast objects can't skip through it in one
// step.
constexpr float kSlabHalfThickness = 0.5f;

// Densities of the real, solid materials. Friction and bounce are picked to
// feel right next to each other, not measured.
constexpr PhysicsMaterial kMaterials[] = {
    {"wood", 600.0f, 0.5f, 0.2f},
    {"stone", 2400.0f, 0.6f, 0.1f},
    {"metal", 7800.0f, 0.4f, 0.1f},
    {"rubber", 1100.0f, 0.9f, 0.7f}, // grippy and bouncy
    {"ice", 920.0f, 0.05f, 0.05f},   // slippery
};

} // namespace

PhysicsBodyDesc bodyDescFor(const SceneModelEntry &entry) {
  const float scale = entry.transform.scale;
  PhysicsBodyDesc desc;
  desc.motion = *entry.body;
  desc.position = entry.transform.position;
  desc.rotation = rotationFromDegrees(entry.transform.rotationDeg);
  // A material sets everything at once; a key in the file beats it, and
  // mass beats density (PhysicsWorld only uses density when mass is 0).
  if (entry.material)
    if (const std::optional<PhysicsMaterial> material =
            findMaterial(*entry.material)) {
      desc.density = material->density;
      desc.friction = material->friction;
      desc.restitution = material->restitution;
    }
  if (entry.density)
    desc.density = *entry.density;
  if (entry.mass)
    desc.mass = *entry.mass;
  if (entry.friction)
    desc.friction = *entry.friction;
  if (entry.restitution)
    desc.restitution = *entry.restitution;
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

std::optional<PhysicsMaterial> findMaterial(std::string_view name) {
  const auto sameName = [name](const PhysicsMaterial &material) {
    return std::ranges::equal(name, std::string_view(material.name),
                              [](unsigned char a, unsigned char b) {
                                return std::tolower(a) == b;
                              });
  };
  const auto found = std::ranges::find_if(kMaterials, sameName);
  if (found == std::end(kMaterials))
    return {};
  return *found;
}
