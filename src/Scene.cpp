#include "Scene.h"

#include "ScenePhysics.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <utility>

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

SceneCollider parseCollider(const std::string &value, int lineNumber) {
  std::string name = value;
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (name == "auto")
    return SceneCollider::Auto;
  if (name == "hull")
    return SceneCollider::Hull;
  if (name == "mesh")
    return SceneCollider::Mesh;
  if (name == "box")
    return SceneCollider::Box;
  throw std::runtime_error("scene parse error, line " +
                           std::to_string(lineNumber) + ": unknown collider '" +
                           value + "' (auto, hull, mesh or box)");
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
  if (model.body && model.animation.active())
    fail("a 'body' can't also have spin, orbit or bob animation, both would "
         "move it");
  if (model.body && model.shape == PrimitiveShape::Plane &&
      *model.body != PhysicsMotion::Static)
    fail("a plane's body must be static");
  if (!model.body && (model.mass || model.density || model.material ||
                      model.friction || model.restitution || model.collider))
    fail("'mass', 'density', 'material', 'friction', 'restitution' and "
         "'collider' need a 'body'");
  if (model.collider && model.shape)
    fail("'collider' is for model files; generated shapes collide as "
         "themselves");
  if (model.body && *model.body != PhysicsMotion::Static &&
      model.collider == SceneCollider::Mesh)
    fail("a moving body can't use 'collider = mesh' (Jolt can't collide a "
         "mesh with another mesh or with terrain); use hull or box");
  if (model.mass && *model.mass <= 0.0f)
    fail("'mass' must be more than 0");
  if (model.density && *model.density <= 0.0f)
    fail("'density' must be more than 0");
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
  else if (key == "density")
    model.density = parseFloat(value, lineNumber);
  else if (key == "collider")
    model.collider = parseCollider(value, lineNumber);
  else if (key == "material") {
    const std::optional<PhysicsMaterial> material = findMaterial(value);
    if (!material)
      throw std::runtime_error("scene parse error, line " +
                               std::to_string(lineNumber) +
                               ": unknown material '" + value +
                               "' (wood, stone, metal, rubber or ice)");
    model.material = material->name;
  }
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

// Shortest text that reads back as exactly the same float.
std::string formatFloat(float value) { return std::format("{}", value); }

// One number when all three match, which parseVec3 spreads back out.
std::string formatVec3(glm::vec3 v) {
  if (v.x == v.y && v.y == v.z)
    return formatFloat(v.x);
  return std::format("{} {} {}", v.x, v.y, v.z);
}

const char *shapeName(PrimitiveShape shape) {
  switch (shape) {
  case PrimitiveShape::Box:
    return "box";
  case PrimitiveShape::Sphere:
    return "sphere";
  case PrimitiveShape::Capsule:
    return "capsule";
  case PrimitiveShape::Plane:
    return "plane";
  }
  return "box";
}

const char *colliderName(SceneCollider collider) {
  switch (collider) {
  case SceneCollider::Auto:
    return "auto";
  case SceneCollider::Hull:
    return "hull";
  case SceneCollider::Mesh:
    return "mesh";
  case SceneCollider::Box:
    return "box";
  }
  return "auto";
}

const char *bodyName(PhysicsMotion motion) {
  switch (motion) {
  case PhysicsMotion::Static:
    return "static";
  case PhysicsMotion::Dynamic:
    return "dynamic";
  case PhysicsMotion::Kinematic:
    return "kinematic";
  }
  return "static";
}

void writeKey(std::string &out, std::string_view key, std::string_view value) {
  out += std::format("{} = {}\n", key, value);
}

// The keys [model] and [grid] share, skipping any left at its default.
void writeModelKeys(std::string &out, const SceneModelEntry &model) {
  const SceneModelEntry d;
  if (!model.path.empty())
    writeKey(out, "path", model.path);
  if (model.shape)
    writeKey(out, "shape", shapeName(*model.shape));
  if (model.size != d.size)
    writeKey(out, "size", formatVec3(model.size));
  if (model.radius != d.radius)
    writeKey(out, "radius", formatFloat(model.radius));
  if (model.height != d.height)
    writeKey(out, "height", formatFloat(model.height));
  if (model.transform.position != d.transform.position)
    writeKey(out, "position", formatVec3(model.transform.position));
  if (model.transform.rotationDeg != d.transform.rotationDeg)
    writeKey(out, "rotation", formatVec3(model.transform.rotationDeg));
  if (model.transform.scale != d.transform.scale)
    writeKey(out, "scale", formatFloat(model.transform.scale));
  if (model.color != d.color)
    writeKey(out, "color", formatVec3(model.color));
  if (model.body)
    writeKey(out, "body", bodyName(*model.body));
  if (model.collider)
    writeKey(out, "collider", colliderName(*model.collider));
  if (model.material)
    writeKey(out, "material", *model.material);
  if (model.density)
    writeKey(out, "density", formatFloat(*model.density));
  if (model.mass)
    writeKey(out, "mass", formatFloat(*model.mass));
  if (model.friction)
    writeKey(out, "friction", formatFloat(*model.friction));
  if (model.restitution)
    writeKey(out, "restitution", formatFloat(*model.restitution));
  const NodeAnimation &a = model.animation;
  if (a.spinDegPerSec != d.animation.spinDegPerSec)
    writeKey(out, "spin", formatVec3(a.spinDegPerSec));
  if (a.orbitDegPerSec != d.animation.orbitDegPerSec)
    writeKey(out, "orbit_speed", formatFloat(a.orbitDegPerSec));
  if (a.orbitRadius != d.animation.orbitRadius)
    writeKey(out, "orbit_radius", formatFloat(a.orbitRadius));
  if (a.bobAmplitude != d.animation.bobAmplitude)
    writeKey(out, "bob_amplitude", formatFloat(a.bobAmplitude));
  if (a.bobFrequency != d.animation.bobFrequency)
    writeKey(out, "bob_frequency", formatFloat(a.bobFrequency));
}

// A blank line, the section's comment, then its header.
void writeHeader(std::string &out, const SceneSectionInfo &info,
                 std::string_view header) {
  out += '\n';
  if (!info.comment.empty())
    out += info.comment + '\n';
  out += std::format("[{}]\n", header);
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

  // Comment lines wait here for the section below them. The block at the
  // very top is the header instead, if a blank line or a key follows it.
  std::string pendingComment;
  bool headerDone = false;
  int nextOrder = 0;
  auto takeComment = [&] { return std::exchange(pendingComment, {}); };

  std::string line;
  int lineNumber = 0;
  while (std::getline(file, line)) {
    ++lineNumber;
    const std::string whole = trim(line);
    if (whole.starts_with('#')) {
      pendingComment += (pendingComment.empty() ? "" : "\n") + whole;
      continue;
    }
    size_t comment = line.find('#');
    if (comment != std::string::npos)
      line = line.substr(0, comment); // comments at the end of a line go
    line = trim(line);
    if (line.empty()) {
      if (!headerDone && !pendingComment.empty()) {
        scene.headerComment = takeComment();
        headerDone = true;
      }
      continue;
    }

    if (line.front() == '[') {
      headerDone = true;
      if (line.back() != ']')
        throw std::runtime_error("scene parse error, line " +
                                 std::to_string(lineNumber) +
                                 ": unterminated section header");
      std::string name = trim(line.substr(1, line.size() - 2));
      std::transform(name.begin(), name.end(), name.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (name == "camera") {
        section = Section::Camera;
        scene.camera.info.comment = takeComment();
      } else if (name == "model") {
        section = Section::Model;
        scene.models.emplace_back().info = {nextOrder++, takeComment()};
        modelLines.push_back(lineNumber);
      } else if (name == "grid") {
        section = Section::Grid;
        scene.grids.emplace_back().info = {nextOrder++, takeComment()};
        gridLines.push_back(lineNumber);
      } else if (name == "ring") {
        section = Section::Ring;
        scene.rings.emplace_back().info = {nextOrder++, takeComment()};
      } else {
        throw std::runtime_error("scene parse error, line " +
                                 std::to_string(lineNumber) +
                                 ": unknown section [" + name + "]");
      }
      continue;
    }

    // A comment between keys belongs to no section and goes.
    if (!headerDone)
      scene.headerComment = takeComment();
    headerDone = true;
    pendingComment.clear();

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

std::string write(const SceneDescription &scene) {
  std::string out;
  if (!scene.headerComment.empty())
    out += scene.headerComment + '\n';
  writeKey(out, "name", scene.name);
  if (!scene.environmentHdr.empty())
    writeKey(out, "environment", scene.environmentHdr);

  const SceneCameraPose &camera = scene.camera;
  const SceneCameraPose d;
  if (camera.position != d.position || camera.yaw != d.yaw ||
      camera.pitch != d.pitch || camera.speed != d.speed ||
      !camera.info.comment.empty()) {
    writeHeader(out, camera.info, "camera");
    if (camera.position != d.position)
      writeKey(out, "position", formatVec3(camera.position));
    if (camera.yaw != d.yaw)
      writeKey(out, "yaw", formatFloat(camera.yaw));
    if (camera.pitch != d.pitch)
      writeKey(out, "pitch", formatFloat(camera.pitch));
    if (camera.speed != d.speed)
      writeKey(out, "speed", formatFloat(camera.speed));
  }

  // Models, grids and rings live in separate lists; info.order puts them
  // back in the order the file had.
  enum class Kind { Model, Grid, Ring };
  struct Section {
    int order;
    Kind kind;
    size_t index;
  };
  std::vector<Section> sections;
  for (size_t i = 0; i < scene.models.size(); ++i)
    sections.push_back({scene.models[i].info.order, Kind::Model, i});
  for (size_t i = 0; i < scene.grids.size(); ++i)
    sections.push_back({scene.grids[i].info.order, Kind::Grid, i});
  for (size_t i = 0; i < scene.rings.size(); ++i)
    sections.push_back({scene.rings[i].info.order, Kind::Ring, i});
  std::stable_sort(sections.begin(), sections.end(),
                   [](const Section &a, const Section &b) {
                     return a.order < b.order;
                   });

  for (const Section &section : sections) {
    switch (section.kind) {
    case Kind::Model: {
      const SceneModelEntry &model = scene.models[section.index];
      writeHeader(out, model.info, "model");
      writeModelKeys(out, model);
      break;
    }
    case Kind::Grid: {
      const SceneGridEntry &grid = scene.grids[section.index];
      const SceneGridEntry gd;
      writeHeader(out, grid.info, "grid");
      writeModelKeys(out, grid.model);
      if (grid.count != gd.count) {
        const glm::ivec3 c = grid.count;
        writeKey(out, "count",
                 c.x == c.y && c.y == c.z
                     ? std::format("{}", c.x)
                     : std::format("{} {} {}", c.x, c.y, c.z));
      }
      if (grid.spacing != gd.spacing)
        writeKey(out, "spacing", formatVec3(grid.spacing));
      break;
    }
    case Kind::Ring: {
      const SceneRingEntry &ring = scene.rings[section.index];
      const SceneRingEntry rd;
      writeHeader(out, ring.info, "ring");
      writeKey(out, "path", ring.path);
      if (ring.count != rd.count)
        writeKey(out, "count", std::format("{}", ring.count));
      if (ring.radius != rd.radius)
        writeKey(out, "radius", formatFloat(ring.radius));
      if (ring.height != rd.height)
        writeKey(out, "height", formatFloat(ring.height));
      if (ring.scale != rd.scale)
        writeKey(out, "scale", formatFloat(ring.scale));
      break;
    }
    }
  }
  return out;
}

void saveToFile(const SceneDescription &scene, const std::string &path) {
  // Write next to the real file, then swap it in with one rename, so a
  // failure part way leaves the old file as it was.
  const std::string temporary = path + ".tmp";
  {
    std::ofstream file(temporary, std::ios::trunc);
    if (!file.is_open())
      throw std::runtime_error("Couldn't write " + temporary);
    file << write(scene);
    file.close();
    if (file.fail()) {
      std::filesystem::remove(temporary);
      throw std::runtime_error("Couldn't finish writing " + temporary);
    }
  }
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(temporary);
    throw std::runtime_error("Couldn't replace " + path + ": " +
                             error.message());
  }
}

} // namespace SceneIO
