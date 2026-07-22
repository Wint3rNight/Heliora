#pragma once

#include <glm/glm.hpp>
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

struct SceneModelEntry {
  std::string path;
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
  std::vector<SceneRingEntry> rings;
};

namespace SceneIO {

// Parses a .scene file (INI-style: `key = value` lines under [camera],
// [model], and [ring] sections; `#` starts a comment). Throws
// std::runtime_error with a line number on malformed input.
SceneDescription loadFromFile(const std::string &path);

// Loads every *.scene file in `directory`, sorted by filename. Tries the
// directory as given and prefixed with "../" (the binary runs from build/).
// Files that fail to parse are skipped with a logged warning.
std::vector<SceneDescription> discover(const std::string &directory);

} // namespace SceneIO
