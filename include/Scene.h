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
  bool operator==(const NodeAnimation &) const = default;
};

// Decomposed transform kept alongside the matrix so animation can recompose
// position/rotation/scale against time without decomposing a mat4.
struct NodeTransform {
  glm::vec3 position = glm::vec3(0.0f);
  glm::vec3 rotationDeg = glm::vec3(0.0f); // applied Y, then X, then Z
  float scale = 1.0f;
  bool operator==(const NodeTransform &) const = default;
};

// Where a section sat in its file and the comment lines right above it, so a
// saved scene keeps a person's notes and the order they wrote things in.
struct SceneSectionInfo {
  int order = 0;       // position among the file's [model], [grid], [ring]
  std::string comment; // the '#' lines above the section, as written
  bool operator==(const SceneSectionInfo &) const = default;
};

// rotationDeg as a quaternion. Scene nodes and physics bodies both build
// their rotation through this, so a body spawns at exactly its node's angle.
glm::quat rotationFromDegrees(glm::vec3 degrees);

// How a model file collides. Auto: its exact triangles if it never moves,
// one hull per part if it does. Box: one box around the whole model.
enum class SceneCollider : uint8_t { Auto, Hull, Mesh, Box };

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
  std::optional<float> density;     // kg/m^3
  std::optional<std::string> material; // wood, stone, metal, rubber, ice
  std::optional<SceneCollider> collider; // model files only; unset is Auto
  std::optional<float> friction;    // 0 and up
  std::optional<float> restitution; // bounciness, 0 to 1
  NodeTransform transform;
  NodeAnimation animation;
  SceneSectionInfo info;
  bool operator==(const SceneModelEntry &) const = default;
};

// An instanced ring layout (the stress-test arrangement): `count` copies of
// one model placed on a circle, drawn with one indexed draw per mesh.
struct SceneRingEntry {
  std::string path;
  int count = 8;
  float radius = 5.0f;
  float height = 1.5f;
  float scale = 1.0f;
  SceneSectionInfo info;
  bool operator==(const SceneRingEntry &) const = default;
};

// Many copies of one model, laid out in a grid: centred on the model's
// position in x and z, stacking upwards from it in y. Each copy keeps the
// model's rotation.
struct SceneGridEntry {
  SceneModelEntry model;
  glm::ivec3 count = glm::ivec3(1);
  glm::vec3 spacing = glm::vec3(1.0f); // centre to centre
  SceneSectionInfo info;
  bool operator==(const SceneGridEntry &) const = default;
};

// One SceneModelEntry per copy of the grid.
std::vector<SceneModelEntry> expandGrid(const SceneGridEntry &grid);

// Which entry of a SceneDescription a scene node was built from. Every copy
// of a grid points at the grid, so clicking one brick selects all of them.
struct SceneEntryRef {
  enum class Kind : uint8_t { None, Model, Grid };
  Kind kind = Kind::None;
  int index = -1; // into SceneDescription::models or ::grids
  bool operator==(const SceneEntryRef &) const = default;
};

struct SceneCameraPose {
  glm::vec3 position = glm::vec3(0.0f, 2.0f, 0.0f);
  float yaw = -90.0f;
  float pitch = 0.0f;
  float speed = 15.0f;
  SceneSectionInfo info; // only the comment is used
  bool operator==(const SceneCameraPose &) const = default;
};

// direction is the way the light travels, so it points down. It needn't be
// unit length in the file; the renderer normalizes it.
struct SceneSun {
  glm::vec3 direction = glm::normalize(glm::vec3(-0.3f, -1.0f, 0.2f));
  glm::vec3 color = glm::vec3(1.0f, 0.93f, 0.78f);
  // At 5.5 the tonemap crushed sunlit floors into a flat bright wash. 2.5
  // keeps their texture visible.
  float intensity = 2.5f;
  SceneSectionInfo info; // only the comment is used
  bool operator==(const SceneSun &) const = default;
};

// Height fog. Its colour is fixed in lit.frag and transparent.frag.
struct SceneFog {
  float density = 0.0f;  // how fast fog builds up per metre; 0 is no fog
  float falloff = 0.25f; // how fast it thins as the camera climbs
  float max = 1.0f;      // the most fog a pixel can get, 0 to 1
  SceneSectionInfo info; // only the comment is used
  bool operator==(const SceneFog &) const = default;
};

struct SceneDescription {
  std::string name;           // display name (defaults to file stem)
  std::string filePath;       // where this description was parsed from
  std::string environmentHdr; // HDRI path; empty = keep the current sky/IBL
  SceneCameraPose camera;
  SceneSun sun;
  SceneFog fog;
  std::vector<SceneModelEntry> models;
  std::vector<SceneGridEntry> grids;
  std::vector<SceneRingEntry> rings;
  std::string headerComment; // '#' lines at the top of the file
  bool operator==(const SceneDescription &) const = default;
};

namespace SceneIO {

// Parses a .scene file (INI-style: `key = value` lines under [camera], [sun],
// [fog], [model], [grid] and [ring] sections; `#` starts a comment). Throws
// std::runtime_error with a line number on malformed input.
SceneDescription loadFromFile(const std::string &path);

// The same, from any stream. sourceName gives the default scene name and
// goes into error messages.
SceneDescription parse(std::istream &in, const std::string &sourceName);

// Loads every *.scene file in `directory`, sorted by filename. Tries the
// directory as given and prefixed with "../" (the binary runs from build/).
// Files that fail to parse are skipped with a logged warning.
std::vector<SceneDescription> discover(const std::string &directory);

// The scene as .scene text: the header comment, the global keys, [camera],
// [sun], [fog], then every [model], [grid] and [ring] in their original
// order, each with the comment that sat above it. Only keys that differ from
// their defaults are written, so a saved file reads like a hand-written one.
std::string write(const SceneDescription &scene);

// Writes write(scene) to path through a temporary file next to it, so a
// failed save never leaves a half-written scene. Throws on failure.
void saveToFile(const SceneDescription &scene, const std::string &path);

} // namespace SceneIO
