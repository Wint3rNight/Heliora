#include "Scene.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace {

std::string trim(const std::string &s) {
  size_t begin = s.find_first_not_of(" \t\r");
  if (begin == std::string::npos)
    return "";
  size_t end = s.find_last_not_of(" \t\r");
  return s.substr(begin, end - begin + 1);
}

// Parses 1..3 floats; a single value is broadcast to all three components so
// `spin = 45` means "45°/s around every listed axis" the way `scale = 2`
// means uniform scale.
glm::vec3 parseVec3(const std::string &value, int lineNumber) {
  std::istringstream in(value);
  float x = 0.0f, y = 0.0f, z = 0.0f;
  if (!(in >> x))
    throw std::runtime_error("scene parse error, line " +
                             std::to_string(lineNumber) +
                             ": expected number(s), got '" + value + "'");
  if (!(in >> y))
    return glm::vec3(x);
  if (!(in >> z))
    z = 0.0f;
  return glm::vec3(x, y, z);
}

float parseFloat(const std::string &value, int lineNumber) {
  try {
    return std::stof(value);
  } catch (const std::exception &) {
    throw std::runtime_error("scene parse error, line " +
                             std::to_string(lineNumber) +
                             ": expected a number, got '" + value + "'");
  }
}

} // namespace

namespace SceneIO {

SceneDescription loadFromFile(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error("Failed to open scene file: " + path);

  SceneDescription scene;
  scene.filePath = path;
  scene.name = std::filesystem::path(path).stem().string();

  enum class Section { Global, Camera, Model, Ring };
  Section section = Section::Global;

  std::string line;
  int lineNumber = 0;
  while (std::getline(file, line)) {
    ++lineNumber;
    size_t comment = line.find('#');
    if (comment != std::string::npos)
      line = line.substr(0, comment);
    line = trim(line);
    if (line.empty())
      continue;

    if (line.front() == '[') {
      if (line.back() != ']')
        throw std::runtime_error("scene parse error, line " +
                                 std::to_string(lineNumber) +
                                 ": unterminated section header");
      std::string name = trim(line.substr(1, line.size() - 2));
      std::transform(name.begin(), name.end(), name.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (name == "camera") {
        section = Section::Camera;
      } else if (name == "model") {
        section = Section::Model;
        scene.models.emplace_back();
      } else if (name == "ring") {
        section = Section::Ring;
        scene.rings.emplace_back();
      } else {
        throw std::runtime_error("scene parse error, line " +
                                 std::to_string(lineNumber) +
                                 ": unknown section [" + name + "]");
      }
      continue;
    }

    size_t eq = line.find('=');
    if (eq == std::string::npos)
      throw std::runtime_error("scene parse error, line " +
                               std::to_string(lineNumber) +
                               ": expected 'key = value'");
    std::string key = trim(line.substr(0, eq));
    std::string value = trim(line.substr(eq + 1));
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    switch (section) {
    case Section::Global:
      if (key == "name")
        scene.name = value;
      else if (key == "environment")
        scene.environmentHdr = value;
      else
        throw std::runtime_error("scene parse error, line " +
                                 std::to_string(lineNumber) +
                                 ": unknown global key '" + key + "'");
      break;

    case Section::Camera:
      if (key == "position")
        scene.camera.position = parseVec3(value, lineNumber);
      else if (key == "yaw")
        scene.camera.yaw = parseFloat(value, lineNumber);
      else if (key == "pitch")
        scene.camera.pitch = parseFloat(value, lineNumber);
      else if (key == "speed")
        scene.camera.speed = parseFloat(value, lineNumber);
      else
        throw std::runtime_error("scene parse error, line " +
                                 std::to_string(lineNumber) +
                                 ": unknown camera key '" + key + "'");
      break;

    case Section::Model: {
      SceneModelEntry &model = scene.models.back();
      if (key == "path")
        model.path = value;
      else if (key == "position")
        model.transform.position = parseVec3(value, lineNumber);
      else if (key == "rotation")
        model.transform.rotationDeg = parseVec3(value, lineNumber);
      else if (key == "scale")
        model.transform.scale = parseFloat(value, lineNumber);
      else if (key == "spin")
        model.animation.spinDegPerSec = parseVec3(value, lineNumber);
      else if (key == "orbit_speed")
        model.animation.orbitDegPerSec = parseFloat(value, lineNumber);
      else if (key == "orbit_radius")
        model.animation.orbitRadius = parseFloat(value, lineNumber);
      else if (key == "bob_amplitude")
        model.animation.bobAmplitude = parseFloat(value, lineNumber);
      else if (key == "bob_frequency")
        model.animation.bobFrequency = parseFloat(value, lineNumber);
      else
        throw std::runtime_error("scene parse error, line " +
                                 std::to_string(lineNumber) +
                                 ": unknown model key '" + key + "'");
      break;
    }

    case Section::Ring: {
      SceneRingEntry &ring = scene.rings.back();
      if (key == "path")
        ring.path = value;
      else if (key == "count")
        ring.count = static_cast<int>(parseFloat(value, lineNumber));
      else if (key == "radius")
        ring.radius = parseFloat(value, lineNumber);
      else if (key == "height")
        ring.height = parseFloat(value, lineNumber);
      else if (key == "scale")
        ring.scale = parseFloat(value, lineNumber);
      else
        throw std::runtime_error("scene parse error, line " +
                                 std::to_string(lineNumber) +
                                 ": unknown ring key '" + key + "'");
      break;
    }
    }
  }

  for (const SceneModelEntry &model : scene.models)
    if (model.path.empty())
      throw std::runtime_error("scene file " + path +
                               ": [model] section is missing 'path'");
  for (const SceneRingEntry &ring : scene.rings)
    if (ring.path.empty())
      throw std::runtime_error("scene file " + path +
                               ": [ring] section is missing 'path'");

  return scene;
}

std::vector<SceneDescription> discover(const std::string &directory) {
  namespace fs = std::filesystem;

  fs::path dir(directory);
  if (!fs::is_directory(dir))
    dir = fs::path("..") / directory; // binary runs from build/
  if (!fs::is_directory(dir)) {
    spdlog::warn("Scene directory not found: {}", directory);
    return {};
  }

  std::vector<fs::path> files;
  for (const auto &entry : fs::directory_iterator(dir))
    if (entry.is_regular_file() && entry.path().extension() == ".scene")
      files.push_back(entry.path());
  std::sort(files.begin(), files.end());

  std::vector<SceneDescription> scenes;
  for (const fs::path &file : files) {
    try {
      scenes.push_back(loadFromFile(file.string()));
    } catch (const std::exception &e) {
      spdlog::warn("Skipping scene file {}: {}", file.string(), e.what());
    }
  }
  return scenes;
}

} // namespace SceneIO
