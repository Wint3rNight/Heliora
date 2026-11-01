#pragma once

#include "PhysicsWorld.h"
#include "PrimitiveMesh.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

// Procedural per-node animation evaluated from elapsed time each frame.
// This is the first dynamic-object path: transforms change per frame and flow
// through the existing per-frame draw-item build (push constants on the CPU
// path, per-frame transform records on the GPU-driven path), so no extra GPU
// synchronization is required.
struct NodeAnimation {
  glm::vec3 spinDegPerSec = glm::vec3(0.0f); // rotation around local X/Y/Z
  float orbitDegPerSec = 0.0f;               // circles the base position…
  float orbitRadius = 0.0f;                  // …at this radius (world units)
  float bobAmplitude = 0.0f;                 // vertical sine offset amplitude
  float bobFrequency = 0.0f;                 // in cycles per second

  bool active() const {
    return spinDegPerSec != glm::vec3(0.0f) ||
           (orbitDegPerSec != 0.0f && orbitRadius != 0.0f) ||
           (bobAmplitude != 0.0f && bobFrequency != 0.0f);
  }
};

// Decomposed transform kept alongside the matrix so animation can recompose
// position/rotation/scale against time without decomposing a mat4.
struct NodeTransform {
  glm::vec3 position = glm::vec3(0.0f);
  glm::vec3 rotationDeg = glm::vec3(0.0f); // applied Y, then X, then Z
  float scale = 1.0f;
};

// rotationDeg as a quaternion. Scene nodes and physics bodies both build
// their rotation through this, so a body spawns at exactly its node's angle.
glm::quat rotationFromDegrees(glm::vec3 degrees);

// A model is a file (path) or a shape generated in code (shape), never both.
struct SceneModelEntry {
  std::string path;
  std::optional<PrimitiveShape> shape;
  glm::vec3 size = glm::vec3(1.0f);  // box: full extents; plane: x and z
  float radius = 0.5f;               // sphere and capsule
  float height = 2.0f;               // capsule, end to end
  glm::vec3 color = glm::vec3(0.8f); // tint for generated shapes
  // Set for shapes that take part in physics. The collider is the shape
  // itself; a plane becomes a static slab under its surface.
  std::optional<PhysicsMotion> body;
  // Only with a body. Unset keeps PhysicsBodyDesc's defaults.
  std::optional<float> mass;        // kg
  std::optional<float> friction;    // 0 and up
  std::optional<float> restitution; // bounciness, 0 to 1
  NodeTransform transform;
  NodeAnimation animation;
};

// An instanced ring layout (the stress-test arrangement): `count` copies of
// one model placed on a circle, drawn with one indexed draw per mesh.
struct SceneRingEntry {
  std::string path;
  int count = 8;
  float radius = 5.0f;
  float height = 1.5f;
  float scale = 1.0f;
};

// Many copies of one model, laid out in a grid: centred on the model's
// position in x and z, stacking upwards from it in y. Each copy keeps the
// model's rotation.
struct SceneGridEntry {
  SceneModelEntry model;
  glm::ivec3 count = glm::ivec3(1);
  glm::vec3 spacing = glm::vec3(1.0f); // centre to centre
};

// One SceneModelEntry per copy of the grid.
std::vector<SceneModelEntry> expandGrid(const SceneGridEntry &grid);

struct SceneCameraPose {
  glm::vec3 position = glm::vec3(0.0f, 2.0f, 0.0f);
  float yaw = -90.0f;
  float pitch = 0.0f;
  float speed = 15.0f;
};

struct SceneDescription {
  std::string name;           // display name (defaults to file stem)
  std::string filePath;       // where this description was parsed from
  std::string environmentHdr; // HDRI path; empty = keep the current sky/IBL
  SceneCameraPose camera;
  std::vector<SceneModelEntry> models;
  std::vector<SceneGridEntry> grids;
  std::vector<SceneRingEntry> rings;
};

namespace SceneIO {

// Parses a .scene file (INI-style: `key = value` lines under [camera],
// [model], [grid] and [ring] sections; `#` starts a comment). Throws
// std::runtime_error with a line number on malformed input.
SceneDescription loadFromFile(const std::string &path);

// The same, from any stream. sourceName gives the default scene name and
// goes into error messages.
SceneDescription parse(std::istream &in, const std::string &sourceName);

// Loads every *.scene file in `directory`, sorted by filename. Tries the
// directory as given and prefixed with "../" (the binary runs from build/).
// Files that fail to parse are skipped with a logged warning.
std::vector<SceneDescription> discover(const std::string &directory);

} // namespace SceneIO
