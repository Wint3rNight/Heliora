#pragma once

#include "PhysicsWorld.h"
#include "Scene.h"

#include <optional>
#include <string_view>

// The physics body for a scene entry with a shape and a body. Sizes are
// scaled by the entry's scale, the same way its mesh is, and the pose
// matches its scene node. Needs entry.shape and entry.body set, which the
// scene parser guarantees for any entry with a body.
PhysicsBodyDesc bodyDescFor(const SceneModelEntry &entry);

// A named material: density, friction and restitution that go together.
struct PhysicsMaterial {
  const char *name;
  float density;     // kg/m^3
  float friction;    // 0 and up
  float restitution; // bounciness, 0 to 1
};

// The material called name (any case), or nothing if there isn't one.
std::optional<PhysicsMaterial> findMaterial(std::string_view name);
