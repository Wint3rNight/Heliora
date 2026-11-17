#pragma once

#include "Picking.h"
#include "Scene.h"

#include <glm/glm.hpp>

#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Adding objects from the editor's spawn list: what's on the list, what a
// new object starts as, and where it lands.

// How far ahead a new object looks for a surface to land on, and where it
// floats when there's none.
constexpr float kSpawnReach = 20.0f;   // metres
constexpr float kSpawnFallback = 5.0f; // metres

// Every .gltf and .glb under folder, sorted, as folder/sub/file.gltf: the
// form scene files use. Looks in ../folder when folder isn't there, as the
// game usually runs from a build folder.
std::vector<std::string> findModelFiles(const std::string &folder);
// The same for any extensions (".fbx" for a character's clips).
std::vector<std::string> findFiles(const std::string &folder,
                                   std::initializer_list<std::string_view> extensions);

// A new generated shape: default size and colour, and a dynamic body, apart
// from a plane, whose body must be static.
SceneModelEntry newShapeEntry(PrimitiveShape shape);
// A new model file: a dynamic body, its collider picked automatically.
SceneModelEntry newModelFileEntry(const std::string &path);

// Where a new object goes: against the surface the ray hit, pushed out
// along the surface's normal just far enough that its box (boundsMin and
// boundsMax around its origin, unrotated) touches it, so it sits on a floor
// and stands off a wall. With nothing hit, kSpawnFallback along the ray.
glm::vec3 spawnPosition(const PickRay &ray,
                        const std::optional<PhysicsRayHit> &hit,
                        glm::vec3 boundsMin, glm::vec3 boundsMax);

// Where Duplicate puts the copy, from the original: one width of its box
// (lo to hi, in the world) along the world axis, x or z, that points most
// the camera's right way, so the copy shows up beside it and lines up.
glm::vec3 duplicateOffset(glm::vec3 lo, glm::vec3 hi, glm::vec3 cameraRight);
