#include "Scene.h"

#include "ScenePhysics.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
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

bool allFinite(glm::vec3 v) {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

// The rules behind problemWith, which a [model] has to pass.
std::optional<std::string> modelProblem(const SceneModelEntry &model) {
  // The parser can't read inf or nan back, so a scene mustn't hold one. NaN
  // also fails every comparison below, so it would pass them all.
  const NodeTransform &t = model.transform;
  if (!allFinite(t.position) || !allFinite(t.rotationDeg) ||
      !std::isfinite(t.scale))
    return "position, rotation and scale must be ordinary numbers";
  auto finite = [](const std::optional<float> &value) {
    return !value || std::isfinite(*value);
  };
  const NodeAnimation &a = model.animation;
  if (!allFinite(model.size) || !std::isfinite(model.radius) ||
      !std::isfinite(model.height) || !allFinite(model.color) ||
      !finite(model.mass) || !finite(model.density) ||
      !finite(model.friction) || !finite(model.restitution) ||
      !finite(model.breakStrength) ||
      !allFinite(a.spinDegPerSec) || !std::isfinite(a.orbitDegPerSec) ||
      !std::isfinite(a.orbitRadius) || !std::isfinite(a.bobAmplitude) ||
      !std::isfinite(a.bobFrequency))
    return "every number must be an ordinary number";
  if (t.scale <= 0.0f)
    return "'scale' must be more than 0";
  if (model.object.find('#') != std::string::npos)
    return "an object path can't contain '#'; rename the file";
  // A copy of an object asset before its object's values are in (just
  // read from a scene file): the rest is checked once they are.
  if (!model.object.empty() && model.path.empty() && !model.shape)
    return std::nullopt;
  // The parser reads # as the start of a comment.
  if (model.path.find('#') != std::string::npos)
    return "a model path can't contain '#' (a scene file reads it as a "
           "comment); rename the file";
  if (model.path.empty() && !model.shape)
    return "needs 'path' or 'shape'";
  if (!model.path.empty() && model.shape)
    return "has both 'path' and 'shape', pick one";
  if (model.body && model.animation.active())
    return "a 'body' can't also have spin, orbit or bob animation, both would "
           "move it";
  if (model.body && model.shape == PrimitiveShape::Plane &&
      *model.body != PhysicsMotion::Static)
    return "a plane's body must be static";
  if (!model.body && (model.mass || model.density || model.material ||
                      model.friction || model.restitution || model.collider))
    return "'mass', 'density', 'material', 'friction', 'restitution' and "
           "'collider' need a 'body'";
  if (model.collider && model.shape)
    return "'collider' is for model files; generated shapes collide as "
           "themselves";
  if (model.body && *model.body != PhysicsMotion::Static &&
      model.collider == SceneCollider::Mesh)
    return "a moving body can't use 'collider = mesh' (Jolt can't collide a "
           "mesh with another mesh or with terrain); use hull or box";
  if (model.mass && *model.mass <= 0.0f)
    return "'mass' must be more than 0";
  if (model.density && *model.density <= 0.0f)
    return "'density' must be more than 0";
  if (model.friction && *model.friction < 0.0f)
    return "'friction' must be 0 or more";
  if (model.restitution &&
      (*model.restitution < 0.0f || *model.restitution > 1.0f))
    return "'restitution' must be between 0 and 1";
  if (!model.breakPieces && (model.breakSeed != 0 || model.breakStrength))
    return "'break_seed' and 'break_strength' need 'break'";
  if (!model.breakPieces && model.glued)
    return "'glued' needs 'break'";
  if (model.breakPieces) {
    if (model.shape != PrimitiveShape::Box)
      return "only boxes break for now ('break' needs 'shape = box')";
    if (!model.body)
      return "'break' needs a 'body': hits break it";
    if (*model.breakPieces < 2 || *model.breakPieces > kMaxBreakPieces)
      return std::format("'break' is the number of pieces, 2 to {}", kMaxBreakPieces);
    if (model.breakStrength && *model.breakStrength <= 0.0f)
      return "'break_strength' must be more than 0";
  }
  if (!model.shape)
    return std::nullopt;
  switch (*model.shape) {
  case PrimitiveShape::Box:
    if (glm::any(glm::lessThanEqual(model.size, glm::vec3(0.0f))))
      return "'size' must be positive on every axis";
    break;
  case PrimitiveShape::Plane:
    if (model.size.x <= 0.0f || model.size.z <= 0.0f)
      return "'size' must be positive on x and z";
    break;
  case PrimitiveShape::Capsule:
    if (model.radius > 0.0f && model.height <= 2.0f * model.radius)
      return "'height' must be more than twice the radius (it includes both "
             "round ends)";
    [[fallthrough]];
  case PrimitiveShape::Sphere:
    if (model.radius <= 0.0f)
      return "'radius' must be positive";
    break;
  }
  return std::nullopt;
}

// Numbers the sections 0, 1, 2... in the order the writer puts them, so a
// scene that lost or gained entries reads back equal to itself.
void renumberSections(SceneDescription &scene) {
  struct Section {
    int order;
    SceneSectionInfo *info;
  };
  std::vector<Section> sections;
  for (SceneModelEntry &model : scene.models)
    sections.push_back({model.info.order, &model.info});
  for (SceneGridEntry &grid : scene.grids)
    sections.push_back({grid.info.order, &grid.info});
  for (SceneRingEntry &ring : scene.rings)
    sections.push_back({ring.info.order, &ring.info});
  for (SceneCharacterEntry &character : scene.characters)
    sections.push_back({character.info.order, &character.info});
  std::stable_sort(sections.begin(), sections.end(),
                   [](const Section &a, const Section &b) {
                     return a.order < b.order;
                   });
  for (size_t i = 0; i < sections.size(); ++i)
    sections[i].info->order = static_cast<int>(i);
}

// The rules behind problemWith for the sun and fog. Bad values here would
// turn the whole picture black or NaN.
std::optional<std::string> sunProblem(const SceneSun &sun) {
  if (!allFinite(sun.direction) || !allFinite(sun.color) ||
      !std::isfinite(sun.intensity))
    return "the sun's numbers must be ordinary numbers";
  if (sun.direction == glm::vec3(0.0f))
    return "'direction' can't be zero";
  if (glm::any(glm::lessThan(sun.color, glm::vec3(0.0f))))
    return "'color' can't be negative";
  if (sun.intensity < 0.0f)
    return "'intensity' must be 0 or more";
  return std::nullopt;
}

std::optional<std::string> fogProblem(const SceneFog &fog) {
  if (!std::isfinite(fog.density) || !std::isfinite(fog.falloff) ||
      !std::isfinite(fog.max))
    return "the fog's numbers must be ordinary numbers";
  if (fog.density < 0.0f)
    return "'density' must be 0 or more";
  if (fog.falloff < 0.0f)
    return "'falloff' must be 0 or more";
  if (fog.max < 0.0f || fog.max > 1.0f)
    return "'max' must be between 0 and 1";
  return std::nullopt;
}

// headerLine is 0 when the file has no such section, and then the defaults
// are fine anyway.
void checkSun(const SceneSun &sun, int headerLine) {
  if (const std::optional<std::string> problem = sunProblem(sun))
    throw std::runtime_error("scene parse error, [sun] at line " +
                             std::to_string(headerLine) + ": " + *problem);
}

void checkFog(const SceneFog &fog, int headerLine) {
  if (const std::optional<std::string> problem = fogProblem(fog))
    throw std::runtime_error("scene parse error, [fog] at line " +
                             std::to_string(headerLine) + ": " + *problem);
}

float parseFloat(const std::string &value, int lineNumber) {
  float number = 0.0f;
  try {
    number = std::stof(value);
  } catch (const std::exception &) {
    throw std::runtime_error("scene parse error, line " +
                             std::to_string(lineNumber) +
                             ": expected a number, got '" + value + "'");
  }
  // std::stof reads nan and inf, which would get past every range check.
  if (!std::isfinite(number))
    throw std::runtime_error("scene parse error, line " +
                             std::to_string(lineNumber) +
                             ": expected an ordinary number, got '" + value +
                             "'");
  return number;
}

// Keys shared by [model] and [grid]. False if the key isn't one of them.
// The keys saying what it is are noted, for a copy of an object asset,
// whose own keys are its overrides.
bool parseModelKey(SceneModelEntry &model, const std::string &key,
                   const std::string &value, int lineNumber) {
  if (isModelDataKey(key) &&
      std::find(model.overrides.begin(), model.overrides.end(), key) == model.overrides.end())
    model.overrides.push_back(key);
  if (key == "object")
    model.object = value;
  else if (key == "path")
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
  else if (key == "break" || key == "break_seed") {
    // Whole numbers only, checked as a float first: converting one past
    // int's range is undefined.
    const float whole = parseFloat(value, lineNumber);
    if (whole != std::round(whole) || whole < 0.0f || whole > 4294967295.0f)
      throw std::runtime_error("scene parse error, line " + std::to_string(lineNumber) +
                               ": '" + key + "' takes a whole number");
    if (key == "break")
      model.breakPieces = static_cast<int>(std::min(whole, 1e6f));
    else
      model.breakSeed = static_cast<uint32_t>(whole);
  }
  else if (key == "break_strength")
    model.breakStrength = parseFloat(value, lineNumber);
  else if (key == "glued") {
    if (value != "true" && value != "false")
      throw std::runtime_error("scene parse error, line " + std::to_string(lineNumber) +
                               ": 'glued' is true or false");
    model.glued = value == "true";
  }
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

bool isModelDataKey(std::string_view key) {
  for (std::string_view data : {"path", "shape", "size", "radius", "height", "color", "body",
                                "mass", "density", "collider", "material", "friction",
                                "restitution", "break", "break_seed", "break_strength", "glued"})
    if (key == data)
      return true;
  return false;
}

namespace {

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

// The keys [model] and [grid] share, skipping any left at its default. A
// copy of an object asset writes its object and, of what it is, only what
// it overrides (even a default: its object may hold something else); its
// place and movement are always its own.
void writeModelKeys(std::string &out, const SceneModelEntry &model) {
  const SceneModelEntry d;
  const bool copy = !model.object.empty();
  auto keep = [&](std::string_view key, bool differs) {
    if (!copy)
      return differs;
    return std::find(model.overrides.begin(), model.overrides.end(), key) !=
           model.overrides.end();
  };
  if (copy)
    writeKey(out, "object", model.object);
  if (keep("path", !model.path.empty()) && !model.path.empty())
    writeKey(out, "path", model.path);
  if (keep("shape", model.shape.has_value()) && model.shape)
    writeKey(out, "shape", shapeName(*model.shape));
  if (keep("size", model.size != d.size))
    writeKey(out, "size", formatVec3(model.size));
  if (keep("radius", model.radius != d.radius))
    writeKey(out, "radius", formatFloat(model.radius));
  if (keep("height", model.height != d.height))
    writeKey(out, "height", formatFloat(model.height));
  if (model.transform.position != d.transform.position)
    writeKey(out, "position", formatVec3(model.transform.position));
  if (model.transform.rotationDeg != d.transform.rotationDeg)
    writeKey(out, "rotation", formatVec3(model.transform.rotationDeg));
  if (model.transform.scale != d.transform.scale)
    writeKey(out, "scale", formatFloat(model.transform.scale));
  if (keep("color", model.color != d.color))
    writeKey(out, "color", formatVec3(model.color));
  if (keep("body", model.body.has_value()) && model.body)
    writeKey(out, "body", bodyName(*model.body));
  if (keep("collider", model.collider.has_value()) && model.collider)
    writeKey(out, "collider", colliderName(*model.collider));
  if (keep("material", model.material.has_value()) && model.material)
    writeKey(out, "material", *model.material);
  if (keep("density", model.density.has_value()) && model.density)
    writeKey(out, "density", formatFloat(*model.density));
  if (keep("mass", model.mass.has_value()) && model.mass)
    writeKey(out, "mass", formatFloat(*model.mass));
  if (keep("friction", model.friction.has_value()) && model.friction)
    writeKey(out, "friction", formatFloat(*model.friction));
  if (keep("restitution", model.restitution.has_value()) && model.restitution)
    writeKey(out, "restitution", formatFloat(*model.restitution));
  if (keep("break", model.breakPieces.has_value()) && model.breakPieces)
    writeKey(out, "break", std::format("{}", *model.breakPieces));
  if (keep("break_seed", model.breakSeed != d.breakSeed))
    writeKey(out, "break_seed", std::format("{}", model.breakSeed));
  if (keep("break_strength", model.breakStrength.has_value()) && model.breakStrength)
    writeKey(out, "break_strength", formatFloat(*model.breakStrength));
  if (keep("glued", model.glued) && model.glued)
    writeKey(out, "glued", "true");
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

glm::vec3 degreesFromRotation(const glm::quat &rotation) {
  // rotationFromDegrees builds Ry(y) * Rx(x) * Rz(z). Multiplied out, that
  // matrix has -sin(x) in row 1, column 2, cos(x) times sin(y) and cos(y) in
  // column 2's other rows, and cos(x) times sin(z) and cos(z) in row 1's
  // other columns. glm indexes [column][row].
  const glm::mat3 m = glm::mat3_cast(rotation);
  const float cosX = std::hypot(m[2][0], m[2][2]);
  const float x = std::atan2(-m[2][1], cosX);
  float y = 0.0f;
  float z = 0.0f;
  if (cosX > 1e-3f) {
    y = std::atan2(m[2][0], m[2][2]);
    z = std::atan2(m[0][1], m[1][1]);
  } else {
    // Straight up or down, y and z turn about the same axis and only their
    // sum or difference shows, so all of it goes in y.
    y = std::atan2(-m[0][2], m[0][0]);
  }
  return glm::degrees(glm::vec3(x, y, z));
}

std::vector<SceneModelEntry> expandGrid(const SceneGridEntry &grid) {
  std::vector<SceneModelEntry> copies;
  copies.reserve(size_t(grid.count.x) * grid.count.y * grid.count.z);
  const glm::vec3 base = grid.model.transform.position;
  // At 0 this is the identity, which leaves every position exactly as it
  // was, so grids without the key don't move by a rounding error.
  const glm::quat turn =
      rotationFromDegrees(glm::vec3(0.0f, grid.layoutRotation, 0.0f));
  // Centred on x and z: copy i sits (i - (n - 1) / 2) spacings from base.
  const glm::vec2 half = (glm::vec2(grid.count.x, grid.count.z) - 1.0f) * 0.5f;
  for (int z = 0; z < grid.count.z; ++z)
    for (int y = 0; y < grid.count.y; ++y)
      for (int x = 0; x < grid.count.x; ++x) {
        SceneModelEntry copy = grid.model;
        copy.transform.position =
            base + turn * glm::vec3((x - half.x) * grid.spacing.x,
                                    y * grid.spacing.y,
                                    (z - half.y) * grid.spacing.z);
        // Turning about world y first is the same as adding to the yaw,
        // since rotationFromDegrees applies y first.
        copy.transform.rotationDeg.y += grid.layoutRotation;
        copies.push_back(copy);
      }
  return copies;
}

std::optional<std::string> problemWith(const SceneModelEntry &model) {
  return modelProblem(model);
}

std::optional<std::string> problemWith(const SceneSun &sun) {
  return sunProblem(sun);
}

std::optional<std::string> problemWith(const SceneFog &fog) {
  return fogProblem(fog);
}

std::optional<std::string> problemWith(const ScenePlayerStart &start) {
  if (!allFinite(start.position) || !std::isfinite(start.yaw))
    return "'position' and 'yaw' must be ordinary numbers";
  return std::nullopt;
}

std::optional<std::string> problemWith(const SceneCharacterEntry &character) {
  if (character.file.empty())
    return "'file' needs a character file, like "
           "Resources/Characters/remy.character";
  if (!allFinite(character.position) || !std::isfinite(character.yaw))
    return "'position' and 'yaw' must be ordinary numbers";
  return std::nullopt;
}

std::optional<int> playerCharacter(const SceneDescription &scene) {
  for (size_t i = 0; i < scene.characters.size(); ++i)
    if (scene.characters[i].player)
      return static_cast<int>(i);
  return std::nullopt;
}

void setPlayerCharacter(SceneDescription &scene, int index) {
  for (size_t i = 0; i < scene.characters.size(); ++i)
    scene.characters[i].player = static_cast<int>(i) == index;
}

SceneEntryRef addCharacter(SceneDescription &scene, SceneCharacterEntry character) {
  character.info.order = std::numeric_limits<int>::max();
  scene.characters.push_back(std::move(character));
  if (scene.characters.back().player)
    setPlayerCharacter(scene, static_cast<int>(scene.characters.size()) - 1);
  renumberSections(scene);
  return {SceneEntryRef::Kind::Character, static_cast<int>(scene.characters.size()) - 1};
}

std::optional<std::string> problemWith(const SceneGridEntry &grid) {
  if (std::optional<std::string> problem = modelProblem(grid.model))
    return problem;
  if (grid.model.breakPieces)
    return "a [grid] can't break yet; split it into objects first";
  if (glm::any(glm::lessThan(grid.count, glm::ivec3(1))) ||
      glm::any(glm::greaterThan(grid.count, glm::ivec3(kMaxCountPerAxis))))
    return "'count' must be 1 to " + std::to_string(kMaxCountPerAxis) +
           " on every axis";
  if (int64_t(grid.count.x) * grid.count.y * grid.count.z > kMaxGridCopies)
    return "a grid can't have more than " + std::to_string(kMaxGridCopies) +
           " copies";
  if (!allFinite(grid.spacing) || !std::isfinite(grid.layoutRotation))
    return "'spacing' and 'layout_rotation' must be ordinary numbers";
  if (glm::any(glm::lessThanEqual(grid.spacing, glm::vec3(0.0f))))
    return "'spacing' must be positive on every axis";
  return std::nullopt;
}

bool showsWorldGrid(const SceneModelEntry &entry) {
  return entry.shape && entry.body == PhysicsMotion::Static;
}

void removeEntry(SceneDescription &scene, SceneEntryRef ref) {
  if (ref.kind == SceneEntryRef::Kind::Player) {
    scene.player.reset(); // not numbered with the others
    return;
  }
  if (ref.kind == SceneEntryRef::Kind::Model && ref.index >= 0 &&
      ref.index < static_cast<int>(scene.models.size()))
    scene.models.erase(scene.models.begin() + ref.index);
  else if (ref.kind == SceneEntryRef::Kind::Grid && ref.index >= 0 &&
           ref.index < static_cast<int>(scene.grids.size()))
    scene.grids.erase(scene.grids.begin() + ref.index);
  else if (ref.kind == SceneEntryRef::Kind::Character && ref.index >= 0 &&
           ref.index < static_cast<int>(scene.characters.size()))
    scene.characters.erase(scene.characters.begin() + ref.index);
  else
    return;
  renumberSections(scene);
}

std::optional<std::string> sceneFileNameFor(std::string_view name) {
  const std::string trimmed = trim(std::string(name));
  if (trimmed.empty())
    return std::nullopt;
  std::string file;
  for (const char c : trimmed) {
    // Only ASCII letters and digits: the byte cast keeps isalnum away from
    // negative values, and in the C locale bytes past 127 aren't letters.
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_')
      file += c;
    else if (c == ' ')
      file += '_';
    else
      return std::nullopt;
  }
  return file + ".scene";
}

SceneEntryRef addModel(SceneDescription &scene, SceneModelEntry model) {
  // Past every section, then numbered back into 0, 1, 2...
  model.info.order = std::numeric_limits<int>::max();
  scene.models.push_back(std::move(model));
  renumberSections(scene);
  return {SceneEntryRef::Kind::Model, static_cast<int>(scene.models.size()) - 1};
}

std::optional<SceneEntryRef> duplicateEntry(SceneDescription &scene,
                                            SceneEntryRef ref,
                                            glm::vec3 offset) {
  // The copy keeps no comment and goes past every section, then everything
  // is numbered back into 0, 1, 2...
  const SceneSectionInfo last{std::numeric_limits<int>::max(), ""};
  if (ref.kind == SceneEntryRef::Kind::Model && ref.index >= 0 &&
      ref.index < static_cast<int>(scene.models.size())) {
    SceneModelEntry copy = scene.models[ref.index];
    copy.transform.position += offset;
    copy.info = last;
    scene.models.push_back(std::move(copy));
    renumberSections(scene);
    return SceneEntryRef{SceneEntryRef::Kind::Model,
                         static_cast<int>(scene.models.size()) - 1};
  }
  if (ref.kind == SceneEntryRef::Kind::Grid && ref.index >= 0 &&
      ref.index < static_cast<int>(scene.grids.size())) {
    SceneGridEntry copy = scene.grids[ref.index];
    copy.model.transform.position += offset;
    copy.info = last;
    scene.grids.push_back(std::move(copy));
    renumberSections(scene);
    return SceneEntryRef{SceneEntryRef::Kind::Grid,
                         static_cast<int>(scene.grids.size()) - 1};
  }
  if (ref.kind == SceneEntryRef::Kind::Character && ref.index >= 0 &&
      ref.index < static_cast<int>(scene.characters.size())) {
    SceneCharacterEntry copy = scene.characters[ref.index];
    copy.position += offset;
    copy.player = false;
    copy.info = last;
    scene.characters.push_back(std::move(copy));
    renumberSections(scene);
    return SceneEntryRef{SceneEntryRef::Kind::Character,
                         static_cast<int>(scene.characters.size()) - 1};
  }
  return std::nullopt;
}

std::vector<SceneEntryRef> splitGrid(SceneDescription &scene, int gridIndex) {
  if (gridIndex < 0 || gridIndex >= static_cast<int>(scene.grids.size()))
    return {};
  const SceneGridEntry grid = scene.grids[gridIndex]; // erased just below
  scene.grids.erase(scene.grids.begin() + gridIndex);
  std::vector<SceneModelEntry> copies = expandGrid(grid);
  // The models list is in file order, so the copies go in after the models
  // that came before the grid. That keeps the scene equal to itself read
  // back from its saved file.
  const auto at = std::find_if(
      scene.models.begin(), scene.models.end(),
      [&](const SceneModelEntry &m) { return m.info.order > grid.info.order; });
  const int first = static_cast<int>(at - scene.models.begin());
  for (size_t i = 0; i < copies.size(); ++i)
    // All at the grid's place; renumbering spreads them out in copy order.
    copies[i].info = {grid.info.order, i == 0 ? grid.info.comment : ""};
  scene.models.insert(at, std::make_move_iterator(copies.begin()),
                      std::make_move_iterator(copies.end()));
  renumberSections(scene);
  std::vector<SceneEntryRef> refs;
  for (int i = 0; i < static_cast<int>(copies.size()); ++i)
    refs.push_back({SceneEntryRef::Kind::Model, first + i});
  return refs;
}

namespace {

[[noreturn]] void parseError(int lineNumber, const std::string &what) {
  throw std::runtime_error("scene parse error, line " +
                           std::to_string(lineNumber) + ": " + what);
}

std::string lowered(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return text;
}

enum class Section { Global, Camera, Sun, Fog, Player, Kit, Model, Grid, Ring, Character };

// What the parser carries from line to line.
struct SceneParser {
  SceneDescription scene;
  Section section = Section::Global;
  std::vector<int> modelLines; // header line of each [model], for errors
  std::vector<int> gridLines;  // and of each [grid]
  std::vector<int> characterLines; // and of each [character]
  int sunLine = 0;
  int fogLine = 0;
  // Comment lines wait here for the section below them. The block at the
  // very top is the header instead, if a blank line or a key follows it.
  std::string pendingComment;
  bool headerDone = false;
  int nextOrder = 0;
  std::string takeComment() { return std::exchange(pendingComment, {}); }
};

// A "[name]" line: the section the keys below it go to, which also gets
// the comment block above it.
void openSection(SceneParser &p, const std::string &line, int lineNumber) {
  p.headerDone = true;
  if (line.back() != ']')
    parseError(lineNumber, "unterminated section header");
  const std::string name = lowered(trim(line.substr(1, line.size() - 2)));
  SceneDescription &scene = p.scene;
  if (name == "camera") {
    p.section = Section::Camera;
    scene.camera.info.comment = p.takeComment();
  } else if (name == "sun") {
    p.section = Section::Sun;
    scene.sun.info.comment = p.takeComment();
    p.sunLine = lineNumber;
  } else if (name == "fog") {
    p.section = Section::Fog;
    scene.fog.info.comment = p.takeComment();
    p.fogLine = lineNumber;
  } else if (name == "player") {
    if (scene.player)
      parseError(lineNumber, "only one [player] per scene");
    p.section = Section::Player;
    scene.player.emplace().info.comment = p.takeComment();
  } else if (name == "kit") {
    if (scene.kit)
      parseError(lineNumber, "only one [kit] per scene");
    p.section = Section::Kit;
    scene.kit.emplace().info.comment = p.takeComment();
  } else if (name == "model") {
    p.section = Section::Model;
    scene.models.emplace_back().info = {p.nextOrder++, p.takeComment()};
    p.modelLines.push_back(lineNumber);
  } else if (name == "grid") {
    p.section = Section::Grid;
    scene.grids.emplace_back().info = {p.nextOrder++, p.takeComment()};
    p.gridLines.push_back(lineNumber);
  } else if (name == "ring") {
    p.section = Section::Ring;
    scene.rings.emplace_back().info = {p.nextOrder++, p.takeComment()};
  } else if (name == "character") {
    p.section = Section::Character;
    scene.characters.emplace_back().info = {p.nextOrder++, p.takeComment()};
    p.characterLines.push_back(lineNumber);
  } else {
    parseError(lineNumber, "unknown section [" + name + "]");
  }
}

void globalKey(SceneDescription &scene, const std::string &key,
               const std::string &value, int lineNumber) {
  if (key == "name")
    scene.name = value;
  else if (key == "environment")
    scene.environmentHdr = value;
  else
    parseError(lineNumber, "unknown global key '" + key + "'");
}

void cameraKey(SceneCameraPose &camera, const std::string &key,
               const std::string &value, int lineNumber) {
  if (key == "position")
    camera.position = parseVec3(value, lineNumber);
  else if (key == "yaw")
    camera.yaw = parseFloat(value, lineNumber);
  else if (key == "pitch")
    camera.pitch = parseFloat(value, lineNumber);
  else if (key == "speed")
    camera.speed = parseFloat(value, lineNumber);
  else
    parseError(lineNumber, "unknown camera key '" + key + "'");
}

void sunKey(SceneSun &sun, const std::string &key, const std::string &value,
            int lineNumber) {
  if (key == "direction")
    sun.direction = parseVec3(value, lineNumber);
  else if (key == "color")
    sun.color = parseVec3(value, lineNumber);
  else if (key == "intensity")
    sun.intensity = parseFloat(value, lineNumber);
  else
    parseError(lineNumber, "unknown sun key '" + key + "'");
}

void fogKey(SceneFog &fog, const std::string &key, const std::string &value,
            int lineNumber) {
  if (key == "density")
    fog.density = parseFloat(value, lineNumber);
  else if (key == "falloff")
    fog.falloff = parseFloat(value, lineNumber);
  else if (key == "max")
    fog.max = parseFloat(value, lineNumber);
  else
    parseError(lineNumber, "unknown fog key '" + key + "'");
}

void kitKey(SceneKit &kit, const std::string &key, const std::string &value, int lineNumber) {
  auto count = [&]() {
    const float n = parseFloat(value, lineNumber);
    if (n < 0.0f || n > 99.0f || n != std::floor(n))
      parseError(lineNumber, "'" + key + "' is a whole number from 0 to 99");
    return static_cast<int>(n);
  };
  if (key == "pistol") {
    if (value != "true" && value != "false")
      parseError(lineNumber, "'pistol' is true or false");
    kit.pistol = value == "true";
  } else if (key == "long") {
    // A gun's name, like "akm": letters, digits, - and _.
    if (value.empty() || value.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-_") !=
                             std::string::npos)
      parseError(lineNumber, "'long' is a gun's name, like akm or rpg");
    kit.longGun = value == "none" ? "" : value;
  } else if (key == "grenades") {
    kit.grenades = count();
  } else if (key == "charges") {
    kit.charges = count();
  } else if (key == "crate") {
    // A place: three numbers, nothing else.
    std::istringstream in(value);
    float x = 0.0f, y = 0.0f, z = 0.0f;
    std::string rest;
    if (!(in >> x >> y >> z) || (in >> rest))
      parseError(lineNumber, "'crate' is a place: three numbers, x y z");
    kit.crates.push_back(glm::vec3(x, y, z));
  } else {
    parseError(lineNumber, "unknown kit key '" + key + "'");
  }
}

void playerKey(ScenePlayerStart &player, const std::string &key,
               const std::string &value, int lineNumber) {
  if (key == "position")
    player.position = parseVec3(value, lineNumber);
  else if (key == "yaw")
    player.yaw = parseFloat(value, lineNumber);
  else if (key == "variant") {
    if (value.empty())
      parseError(lineNumber, "'variant' needs the name of one of the character's "
                             "variants");
    player.variant = value;
  } else
    parseError(lineNumber, "unknown player key '" + key + "'");
}

void characterKey(SceneCharacterEntry &character, const std::string &key,
                  const std::string &value, int lineNumber) {
  if (key == "file") {
    character.file = value;
  } else if (key == "position") {
    character.position = parseVec3(value, lineNumber);
  } else if (key == "yaw") {
    character.yaw = parseFloat(value, lineNumber);
  } else if (key == "variant") {
    if (value.empty())
      parseError(lineNumber, "'variant' needs the name of one of the character's "
                             "variants");
    character.variant = value;
  } else if (key == "player") {
    if (value != "true" && value != "false")
      parseError(lineNumber, "'player' is true or false");
    character.player = value == "true";
  } else {
    parseError(lineNumber, "unknown character key '" + key + "'");
  }
}

// A grid's own keys; anything else is its model's.
void gridKey(SceneGridEntry &grid, const std::string &key,
             const std::string &value, int lineNumber) {
  if (key == "object")
    parseError(lineNumber, "a [grid] can't be copies of a prefab yet; give it a "
                           "'shape' or 'path'");
  if (key == "count") {
    // Checked as floats first: converting one past int's range is
    // undefined.
    const glm::vec3 count = parseVec3(value, lineNumber);
    if (!allFinite(count) ||
        glm::any(glm::lessThan(count, glm::vec3(1.0f))) ||
        glm::any(glm::greaterThan(count, glm::vec3(kMaxCountPerAxis))))
      parseError(lineNumber, "'count' must be 1 to " +
                                 std::to_string(kMaxCountPerAxis) +
                                 " on every axis");
    grid.count = glm::ivec3(glm::round(count));
    if (glm::any(glm::notEqual(glm::vec3(grid.count), count)))
      parseError(lineNumber, "'count' takes whole numbers");
  } else if (key == "spacing") {
    grid.spacing = parseVec3(value, lineNumber);
  } else if (key == "layout_rotation") {
    grid.layoutRotation = parseFloat(value, lineNumber);
  } else if (!parseModelKey(grid.model, key, value, lineNumber)) {
    parseError(lineNumber, "unknown grid key '" + key + "'");
  }
}

void ringKey(SceneRingEntry &ring, const std::string &key,
             const std::string &value, int lineNumber) {
  if (key == "path") {
    ring.path = value;
  } else if (key == "count") {
    const float count = parseFloat(value, lineNumber);
    if (count < 1.0f || count > float(kMaxCountPerAxis) ||
        count != std::round(count))
      parseError(lineNumber,
                 "a ring's 'count' must be a whole number, 1 to " +
                     std::to_string(kMaxCountPerAxis));
    ring.count = static_cast<int>(count);
  } else if (key == "radius") {
    ring.radius = parseFloat(value, lineNumber);
  } else if (key == "height") {
    ring.height = parseFloat(value, lineNumber);
  } else if (key == "scale") {
    ring.scale = parseFloat(value, lineNumber);
  } else {
    parseError(lineNumber, "unknown ring key '" + key + "'");
  }
}

// A "key = value" line, for the section it's in.
void applyKey(SceneParser &p, const std::string &key, const std::string &value,
              int lineNumber) {
  SceneDescription &scene = p.scene;
  switch (p.section) {
  case Section::Global:
    globalKey(scene, key, value, lineNumber);
    break;
  case Section::Camera:
    cameraKey(scene.camera, key, value, lineNumber);
    break;
  case Section::Sun:
    sunKey(scene.sun, key, value, lineNumber);
    break;
  case Section::Fog:
    fogKey(scene.fog, key, value, lineNumber);
    break;
  case Section::Player:
    playerKey(*scene.player, key, value, lineNumber);
    break;
  case Section::Kit:
    kitKey(*scene.kit, key, value, lineNumber);
    break;
  case Section::Model:
    if (!parseModelKey(scene.models.back(), key, value, lineNumber))
      parseError(lineNumber, "unknown model key '" + key + "'");
    break;
  case Section::Grid:
    gridKey(scene.grids.back(), key, value, lineNumber);
    break;
  case Section::Ring:
    ringKey(scene.rings.back(), key, value, lineNumber);
    break;
  case Section::Character:
    characterKey(scene.characters.back(), key, value, lineNumber);
    break;
  }
}

// Checked once the whole file is read, since keys can come in any order.
void checkParsedScene(const SceneParser &p, const std::string &path) {
  const SceneDescription &scene = p.scene;
  checkSun(scene.sun, p.sunLine);
  checkFog(scene.fog, p.fogLine);
  for (size_t i = 0; i < scene.models.size(); ++i)
    if (const auto problem = problemWith(scene.models[i]))
      throw std::runtime_error("scene parse error, [model] at line " +
                               std::to_string(p.modelLines[i]) + ": " +
                               *problem);
  for (size_t i = 0; i < scene.grids.size(); ++i)
    if (const auto problem = problemWith(scene.grids[i]))
      throw std::runtime_error("scene parse error, [grid] at line " +
                               std::to_string(p.gridLines[i]) + ": " +
                               *problem);
  bool playerSeen = false;
  for (size_t i = 0; i < scene.characters.size(); ++i) {
    std::optional<std::string> problem = problemWith(scene.characters[i]);
    if (!problem && scene.characters[i].player && std::exchange(playerSeen, true))
      problem = "only one character can be the player";
    if (problem)
      throw std::runtime_error("scene parse error, [character] at line " +
                               std::to_string(p.characterLines[i]) + ": " +
                               *problem);
  }
  for (const SceneRingEntry &ring : scene.rings)
    if (ring.path.empty())
      throw std::runtime_error("scene file " + path +
                               ": [ring] section is missing 'path'");
}

} // namespace

bool readModelKey(SceneModelEntry &model, const std::string &key, const std::string &value,
                  int lineNumber) {
  return parseModelKey(model, key, value, lineNumber);
}

std::string modelKeysText(const SceneModelEntry &model) {
  std::string out;
  writeModelKeys(out, model);
  return out;
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
  SceneParser p;
  p.scene.name = std::filesystem::path(path).stem().string();

  std::string line;
  int lineNumber = 0;
  while (std::getline(file, line)) {
    ++lineNumber;
    const std::string whole = trim(line);
    if (whole.starts_with('#')) {
      p.pendingComment += (p.pendingComment.empty() ? "" : "\n") + whole;
      continue;
    }
    size_t comment = line.find('#');
    if (comment != std::string::npos)
      line = line.substr(0, comment); // comments at the end of a line go
    line = trim(line);
    if (line.empty()) {
      if (!p.headerDone && !p.pendingComment.empty()) {
        p.scene.headerComment = p.takeComment();
        p.headerDone = true;
      }
      continue;
    }
    if (line.front() == '[') {
      openSection(p, line, lineNumber);
      continue;
    }

    // A comment between keys belongs to no section and goes.
    if (!p.headerDone)
      p.scene.headerComment = p.takeComment();
    p.headerDone = true;
    p.pendingComment.clear();

    const size_t eq = line.find('=');
    if (eq == std::string::npos)
      parseError(lineNumber, "expected 'key = value'");
    applyKey(p, lowered(trim(line.substr(0, eq))), trim(line.substr(eq + 1)),
             lineNumber);
  }
  // Only a copy of an object asset needs to know which keys it set itself.
  for (SceneModelEntry &model : p.scene.models)
    if (model.object.empty())
      model.overrides.clear();
  for (SceneGridEntry &grid : p.scene.grids)
    grid.model.overrides.clear();
  checkParsedScene(p, path);
  return std::move(p.scene);
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

  // The error_code forms, so an unreadable folder or entry is a warning,
  // not an exception out of startup or the scene picker.
  std::vector<fs::path> files;
  std::error_code error;
  for (fs::directory_iterator it(dir, error);
       !error && it != fs::directory_iterator(); it.increment(error)) {
    std::error_code typeError;
    if (it->is_regular_file(typeError) && it->path().extension() == ".scene")
      files.push_back(it->path());
  }
  if (error)
    spdlog::warn("Scene scan of {} stopped early: {}", dir.string(),
                 error.message());
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

  // These compare the comment too, so a comment alone keeps its section.
  if (const SceneSun d; scene.sun != d) {
    writeHeader(out, scene.sun.info, "sun");
    if (scene.sun.direction != d.direction)
      writeKey(out, "direction", formatVec3(scene.sun.direction));
    if (scene.sun.color != d.color)
      writeKey(out, "color", formatVec3(scene.sun.color));
    if (scene.sun.intensity != d.intensity)
      writeKey(out, "intensity", formatFloat(scene.sun.intensity));
  }
  if (const SceneFog d; scene.fog != d) {
    writeHeader(out, scene.fog.info, "fog");
    if (scene.fog.density != d.density)
      writeKey(out, "density", formatFloat(scene.fog.density));
    if (scene.fog.falloff != d.falloff)
      writeKey(out, "falloff", formatFloat(scene.fog.falloff));
    if (scene.fog.max != d.max)
      writeKey(out, "max", formatFloat(scene.fog.max));
  }
  // Written whenever there is one: a start at the origin facing -z is still
  // a start.
  if (scene.player) {
    const ScenePlayerStart d;
    writeHeader(out, scene.player->info, "player");
    if (scene.player->position != d.position)
      writeKey(out, "position", formatVec3(scene.player->position));
    if (scene.player->yaw != d.yaw)
      writeKey(out, "yaw", formatFloat(scene.player->yaw));
    if (!scene.player->variant.empty())
      writeKey(out, "variant", scene.player->variant);
  }
  if (scene.kit) {
    writeHeader(out, scene.kit->info, "kit");
    if (scene.kit->pistol)
      writeKey(out, "pistol", "true");
    if (!scene.kit->longGun.empty())
      writeKey(out, "long", scene.kit->longGun);
    if (scene.kit->grenades != 0)
      writeKey(out, "grenades", std::to_string(scene.kit->grenades));
    if (scene.kit->charges != 0)
      writeKey(out, "charges", std::to_string(scene.kit->charges));
    for (const glm::vec3 &crate : scene.kit->crates)
      writeKey(out, "crate", formatVec3(crate));
  }

  // Models, grids and rings live in separate lists; info.order puts them
  // back in the order the file had.
  enum class Kind { Model, Grid, Ring, Character };
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
  for (size_t i = 0; i < scene.characters.size(); ++i)
    sections.push_back({scene.characters[i].info.order, Kind::Character, i});
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
      if (grid.layoutRotation != gd.layoutRotation)
        writeKey(out, "layout_rotation", formatFloat(grid.layoutRotation));
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
    case Kind::Character: {
      const SceneCharacterEntry &character = scene.characters[section.index];
      const SceneCharacterEntry cd;
      writeHeader(out, character.info, "character");
      writeKey(out, "file", character.file);
      if (character.position != cd.position)
        writeKey(out, "position", formatVec3(character.position));
      if (character.yaw != cd.yaw)
        writeKey(out, "yaw", formatFloat(character.yaw));
      if (!character.variant.empty())
        writeKey(out, "variant", character.variant);
      if (character.player)
        writeKey(out, "player", "true");
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
