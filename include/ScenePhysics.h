#pragma once

#include "PhysicsWorld.h"
#include "Scene.h"

// The physics body for a scene entry with a shape and a body. Sizes are
// scaled by the entry's scale, the same way its mesh is, and the pose
// matches its scene node. Needs entry.shape and entry.body set, which the
// scene parser guarantees for any entry with a body.
PhysicsBodyDesc bodyDescFor(const SceneModelEntry &entry);
