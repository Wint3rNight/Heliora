#pragma once

#include "PhysicsWorld.h"
#include "Scene.h"

#include <optional>
#include <span>
#include <string_view>

// The physics body for a scene entry with a shape and a body. Sizes are
// scaled by the entry's scale, the same way its mesh is, and the pose
// matches its scene node. Needs entry.shape and entry.body set, which the
// scene parser guarantees for any entry with a body.
// A model file's body also needs its parts: the collider is built from them.
// defaultFriction is for entries with no friction or material of their own.
PhysicsBodyDesc bodyDescFor(const SceneModelEntry &entry,
                            std::shared_ptr<const PhysicsMeshParts> parts =
                                nullptr,
                            float defaultFriction = 0.2f);

// A named material: density, friction and restitution that go together.
struct PhysicsMaterial {
  const char *name;
  float density;     // kg/m^3
  float friction;    // 0 and up
  float restitution; // bounciness, 0 to 1
};

// The material called name (any case), or nothing if there isn't one.
std::optional<PhysicsMaterial> findMaterial(std::string_view name);
// Every material, for the Inspector's list.
std::span<const PhysicsMaterial> physicsMaterials();

// A model part cut down to at most maxTriangles for its collider, keeping
// only the vertices those triangles use. Colliders don't need a scanned
// rock's 2 cm triangles, and every triangle a moving body touches costs a
// test each step. Parts already under the limit come back unchanged.
PhysicsMeshPart simplifiedForCollision(const PhysicsMeshPart &part,
                                       size_t maxTriangles);
