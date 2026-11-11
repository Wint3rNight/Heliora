#include "ScenePhysics.h"

#include <algorithm>
#include <cctype>
#include <limits>

#include <meshoptimizer.h>

namespace {

// A plane has no thickness to collide with, so its body is a slab whose top
// face is the plane. 1 m thick so fast objects can't skip through it in one
// step.
constexpr float kSlabHalfThickness = 0.5f;

// One hull from the 8 corners of everything's bounds: collider = box.
std::shared_ptr<const PhysicsMeshParts>
boundsBox(const std::shared_ptr<const PhysicsMeshParts> &parts) {
  if (!parts)
    return nullptr;
  glm::vec3 lo(std::numeric_limits<float>::max());
  glm::vec3 hi(std::numeric_limits<float>::lowest());
  for (const PhysicsMeshPart &part : parts->parts)
    for (const glm::vec3 &p : part.positions) {
      lo = glm::min(lo, p);
      hi = glm::max(hi, p);
    }
  auto box = std::make_shared<PhysicsMeshParts>();
  PhysicsMeshPart &corners = box->parts.emplace_back();
  for (int i = 0; i < 8; ++i)
    corners.positions.push_back(
        glm::vec3(i & 1 ? hi.x : lo.x, i & 2 ? hi.y : lo.y,
                  i & 4 ? hi.z : lo.z));
  return box;
}

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

PhysicsBodyDesc bodyDescFor(const SceneModelEntry &entry,
                            std::shared_ptr<const PhysicsMeshParts> parts,
                            float defaultFriction) {
  const float scale = entry.transform.scale;
  PhysicsBodyDesc desc;
  desc.friction = defaultFriction;
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
  if (!entry.shape) {
    // A model file: the collider is built from its own parts.
    desc.scale = glm::vec3(scale);
    switch (entry.collider.value_or(SceneCollider::Auto)) {
    case SceneCollider::Auto:
      desc.shape = desc.motion == PhysicsMotion::Static
                       ? PhysicsShape::TriangleMesh
                       : PhysicsShape::Hulls;
      desc.parts = std::move(parts);
      break;
    case SceneCollider::Hull:
      desc.shape = PhysicsShape::Hulls;
      desc.parts = std::move(parts);
      break;
    case SceneCollider::Mesh:
      desc.shape = PhysicsShape::TriangleMesh;
      desc.parts = std::move(parts);
      break;
    case SceneCollider::Box:
      desc.shape = PhysicsShape::Hulls;
      desc.parts = boundsBox(parts);
      break;
    }
    return desc;
  }
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

PhysicsMeshPart simplifiedForCollision(const PhysicsMeshPart &part,
                                       size_t maxTriangles) {
  if (part.indices.size() / 3 <= maxTriangles || part.positions.empty())
    return part;
  const size_t target = maxTriangles * 3;
  const float *positions = &part.positions.front().x;
  std::vector<uint32_t> indices(part.indices.size());
  // Collapse edges until the count fits, at any error: this is a collider,
  // and the shape that's left is still the best one for that count.
  indices.resize(meshopt_simplify(indices.data(), part.indices.data(),
                                  part.indices.size(), positions,
                                  part.positions.size(), sizeof(glm::vec3),
                                  target, std::numeric_limits<float>::max()));
  // Edges on open borders can't collapse, so a scan with many holes can
  // stay over the limit. The sloppy version merges nearby vertices instead.
  if (indices.size() > target)
    indices.resize(meshopt_simplifySloppy(
        indices.data(), part.indices.data(), part.indices.size(), positions,
        part.positions.size(), sizeof(glm::vec3), target,
        std::numeric_limits<float>::max()));

  // Keep only the vertices the remaining triangles use.
  std::vector<uint32_t> remap(part.positions.size());
  const size_t used = meshopt_optimizeVertexFetchRemap(
      remap.data(), indices.data(), indices.size(), part.positions.size());
  PhysicsMeshPart simple;
  simple.positions.resize(used);
  meshopt_remapVertexBuffer(simple.positions.data(), part.positions.data(),
                            part.positions.size(), sizeof(glm::vec3),
                            remap.data());
  simple.indices.resize(indices.size());
  meshopt_remapIndexBuffer(simple.indices.data(), indices.data(),
                           indices.size(), remap.data());
  return simple;
}

std::span<const PhysicsMaterial> physicsMaterials() { return kMaterials; }

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
