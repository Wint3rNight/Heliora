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

PrimitiveShape parseShape(const std::string &value, int lineNumber) {
  std::string name = value;
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (name == "box")
    return PrimitiveShape::Box;
  if (name == "sphere")
    return PrimitiveShape::Sphere;
  if (name == "capsule")
    return PrimitiveShape::Capsule;
  if (name == "plane")
    return PrimitiveShape::Plane;
  throw std::runtime_error("scene parse error, line " +
                           std::to_string(lineNumber) + ": unknown shape '" +
                           value + "' (box, sphere, capsule or plane)");
}

PhysicsMotion parseBody(const std::string &value, int lineNumber) {
  std::string name = value;
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (name == "static")
    return PhysicsMotion::Static;
  if (name == "dynamic")
    return PhysicsMotion::Dynamic;
  if (name == "kinematic")
    return PhysicsMotion::Kinematic;
  throw std::runtime_error("scene parse error, line " +
                           std::to_string(lineNumber) + ": unknown body '" +
                           value + "' (static, dynamic or kinematic)");
}

// A [model] needs a file or a shape, and a shape needs a size it can be
// built at. Checked after parsing, since the keys can come in any order.
void checkModel(const SceneModelEntry &model, int headerLine) {
  auto fail = [&](const std::string &what) {
    throw std::runtime_error("scene parse error, [model] at line " +
                             std::to_string(headerLine) + ": " + what);
  };
  if (model.path.empty() && !model.shape)
    fail("needs 'path' or 'shape'");
  if (!model.path.empty() && model.shape)
    fail("has both 'path' and 'shape', pick one");
  if (model.body && !model.shape)
    fail("'body' needs a 'shape' to collide with (model files have no "
         "collider yet)");
  if (model.body && model.animation.active())
    fail("a 'body' can't also have spin, orbit or bob animation, both would "
         "move it");
  if (model.body && model.shape == PrimitiveShape::Plane &&
      *model.body != PhysicsMotion::Static)
    fail("a plane's body must be static");
  if (!model.body && (model.mass || model.friction || model.restitution))
    fail("'mass', 'friction' and 'restitution' need a 'body'");
  if (model.mass && *model.mass <= 0.0f)
    fail("'mass' must be more than 0");
  if (model.friction && *model.friction < 0.0f)
    fail("'friction' must be 0 or more");
  if (model.restitution &&
      (*model.restitution < 0.0f || *model.restitution > 1.0f))
    fail("'restitution' must be between 0 and 1");
  if (!model.shape)
    return;
  switch (*model.shape) {
  case PrimitiveShape::Box:
    if (glm::any(glm::lessThanEqual(model.size, glm::vec3(0.0f))))
      fail("'size' must be positive on every axis");
    break;
  case PrimitiveShape::Plane:
    if (model.size.x <= 0.0f || model.size.z <= 0.0f)
      fail("'size' must be positive on x and z");
    break;
  case PrimitiveShape::Capsule:
    if (model.radius > 0.0f && model.height <= 2.0f * model.radius)
      fail("'height' must be more than twice the radius (it includes both "
           "round ends)");
    [[fallthrough]];
  case PrimitiveShape::Sphere:
    if (model.radius <= 0.0f)
      fail("'radius' must be positive");
    break;
  }
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

// Keys shared by [model] and [grid]. False if the key isn't one of them.
bool parseModelKey(SceneModelEntry &model, const std::string &key,
                   const std::string &value, int lineNumber) {
  if (key == "path")
    model.path = value;
  else if (key == "shape")
    model.shape = parseShape(value, lineNumber);
  else if (key == "size")
    model.size = parseVec3(value, lineNumber);
  else if (key == "radius")
    model.radius = parseFloat(value, lineNumber);
  else if (key == "height")
    model.height = parseFloat(value, lineNumber);
  else if (key == "color")
    model.color = parseVec3(value, lineNumber);
  else if (key == "body")
    model.body = parseBody(value, lineNumber);
  else if (key == "mass")
    model.mass = parseFloat(value, lineNumber);
  else if (key == "friction")
    model.friction = parseFloat(value, lineNumber);
  else if (key == "restitution")
    model.restitution = parseFloat(value, lineNumber);
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
    return false;
  return true;
}

} // namespace

glm::quat rotationFromDegrees(glm::vec3 degrees) {
  const glm::vec3 r = glm::radians(degrees);
  return glm::angleAxis(r.y, glm::vec3(0, 1, 0)) *
         glm::angleAxis(r.x, glm::vec3(1, 0, 0)) *
         glm::angleAxis(r.z, glm::vec3(0, 0, 1));
}

std::vector<SceneModelEntry> expandGrid(const SceneGridEntry &grid) {
  std::vector<SceneModelEntry> copies;
  copies.reserve(size_t(grid.count.x) * grid.count.y * grid.count.z);
  const glm::vec3 base = grid.model.transform.position;
  // Centred on x and z: copy i sits (i - (n - 1) / 2) spacings from base.
  const glm::vec2 half = (glm::vec2(grid.count.x, grid.count.z) - 1.0f) * 0.5f;
  for (int z = 0; z < grid.count.z; ++z)
    for (int y = 0; y < grid.count.y; ++y)
      for (int x = 0; x < grid.count.x; ++x) {
        SceneModelEntry copy = grid.model;
        copy.transform.position =
            base + glm::vec3((x - half.x) * grid.spacing.x, y * grid.spacing.y,
                             (z - half.y) * grid.spacing.z);
        copies.push_back(copy);
      }
  return copies;
}

namespace SceneIO {

SceneDescription loadFromFile(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error("Failed to open scene file: " + path);
  SceneDescription scene = parse(file, path);
  scene.filePath = path;
  return scene;
}

SceneDescription parse(std::istream &file, const std::string &path) {
  SceneDescription scene;
  scene.name = std::filesystem::path(path).stem().string();

  enum class Section { Global, Camera, Model, Grid, Ring };
  Section section = Section::Global;
  std::vector<int> modelLines; // header line of each [model], for errors
  std::vector<int> gridLines;  // and of each [grid]

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
        modelLines.push_back(lineNumber);
      } else if (name == "grid") {
        section = Section::Grid;
        scene.grids.emplace_back();
        gridLines.push_back(lineNumber);
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

    case Section::Model:
      if (!parseModelKey(scene.models.back(), key, value, lineNumber))
        throw std::runtime_error("scene parse error, line " +
                                 std::to_string(lineNumber) +
                                 ": unknown model key '" + key + "'");
      break;

    case Section::Grid: {
      SceneGridEntry &grid = scene.grids.back();
      if (key == "count") {
        const glm::vec3 count = parseVec3(value, lineNumber);
        grid.count = glm::ivec3(glm::round(count));
        if (glm::any(glm::notEqual(glm::vec3(grid.count), count)))
          throw std::runtime_error("scene parse error, line " +
                                   std::to_string(lineNumber) +
                                   ": 'count' takes whole numbers");
      } else if (key == "spacing") {
        grid.spacing = parseVec3(value, lineNumber);
      } else if (!parseModelKey(grid.model, key, value, lineNumber)) {
        throw std::runtime_error("scene parse error, line " +
                                 std::to_string(lineNumber) +
                                 ": unknown grid key '" + key + "'");
      }
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

  for (size_t i = 0; i < scene.models.size(); ++i)
    checkModel(scene.models[i], modelLines[i]);
  for (size_t i = 0; i < scene.grids.size(); ++i) {
    const SceneGridEntry &grid = scene.grids[i];
    checkModel(grid.model, gridLines[i]);
    const std::string where =
        "scene parse error, [grid] at line " + std::to_string(gridLines[i]);
    if (glm::any(glm::lessThan(grid.count, glm::ivec3(1))))
      throw std::runtime_error(where +
                               ": 'count' must be at least 1 on every axis");
    if (glm::any(glm::lessThanEqual(grid.spacing, glm::vec3(0.0f))))
      throw std::runtime_error(where +
                               ": 'spacing' must be positive on every axis");
  }
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
