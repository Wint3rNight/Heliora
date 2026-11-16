#pragma once

#include "PhysicsWorld.h"
#include "Picking.h"
#include "Scene.h"

#include <cstdint>
#include <optional>
#include <vector>

// The scene's player start in the editor: clicking it, drawing it, and
// placing it from the spawn list. It has no mesh or body of its own, only
// the character's capsule (radius and height) standing on its position.

// How far along the ray it meets the capsule standing on the start, or
// nothing.
std::optional<float> rayHitsPlayerStart(const PickRay &ray,
                                        const ScenePlayerStart &start,
                                        float radius, float height);

// The capsule's outline and an arrow on the ground the way it faces, as
// pairs of line ends for DebugLinePass.
void appendPlayerStartLines(std::vector<PhysicsDebugVertex> &lines,
                            const ScenePlayerStart &start, float radius,
                            float height, uint32_t color);

// Where the spawn list puts it: standing on the surface the ray hit (or
// kSpawnFallback along the ray with nothing hit), off a wall by its radius,
// and facing along the ray, away from the camera.
ScenePlayerStart playerStartAt(const PickRay &ray,
                               const std::optional<PhysicsRayHit> &hit,
                               float radius, float height);
