#include "Prefab.h"

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>

namespace {

std::string trim(std::string_view s) {
  const size_t first = s.find_first_not_of(" \t\r");
  if (first == std::string_view::npos)
    return "";
  const size_t last = s.find_last_not_of(" \t\r");
  return std::string(s.substr(first, last - first + 1));
}

// One key's value copied from one entry to another, or compared.
void copyDataKey(SceneModelEntry &to, const SceneModelEntry &from, std::string_view key) {
  if (key == "path") to.path = from.path;
  else if (key == "shape") to.shape = from.shape;
  else if (key == "size") to.size = from.size;
  else if (key == "radius") to.radius = from.radius;
  else if (key == "height") to.height = from.height;
  else if (key == "color") to.color = from.color;
  else if (key == "body") to.body = from.body;
  else if (key == "mass") to.mass = from.mass;
  else if (key == "density") to.density = from.density;
  else if (key == "collider") to.collider = from.collider;
  else if (key == "material") to.material = from.material;
  else if (key == "friction") to.friction = from.friction;
  else if (key == "restitution") to.restitution = from.restitution;
  else if (key == "break") to.breakPieces = from.breakPieces;
  else if (key == "break_seed") to.breakSeed = from.breakSeed;
  else if (key == "break_strength") to.breakStrength = from.breakStrength;
  else if (key == "glued") to.glued = from.glued;
}

bool dataKeyDiffers(const SceneModelEntry &a, const SceneModelEntry &b, std::string_view key) {
  SceneModelEntry probe = a;
  copyDataKey(probe, b, key);
  return !(probe == a);
}

constexpr std::string_view kDataKeys[] = {"path",     "shape",       "size",       "radius",
                                          "height",   "color",       "body",       "mass",
                                          "density",  "collider",    "material",   "friction",
                                          "restitution", "break",    "break_seed", "break_strength",
                                          "glued"};

bool listed(const std::vector<std::string> &keys, std::string_view key) {
  return std::ranges::find(keys, key) != keys.end();
}

} // namespace

namespace ObjectIO {

ObjectAsset parse(std::istream &in, const std::string &sourceName) {
  ObjectAsset object;
  bool started = false; // past the comment at the top
  std::string line;
  int lineNumber = 0;
  auto fail = [&](int at, const std::string &message) {
    throw std::runtime_error(at > 0 ? std::format("{}, line {}: {}", sourceName, at, message)
                                    : std::format("{}: {}", sourceName, message));
  };
  while (std::getline(in, line)) {
    ++lineNumber;
    const std::string whole = trim(line);
    if (whole.empty())
      continue;
    if (whole[0] == '#') {
      if (!started)
        object.comment += whole + "\n";
      continue;
    }
    started = true;
    const std::string text = trim(std::string_view(whole).substr(0, whole.find('#')));
    const size_t equals = text.find('=');
    if (equals == std::string::npos)
      fail(lineNumber, "expected 'key = value', got '" + text + "'");
    std::string key = trim(std::string_view(text).substr(0, equals));
    std::ranges::transform(key, key.begin(), [](unsigned char c) { return std::tolower(c); });
    const std::string value = trim(std::string_view(text).substr(equals + 1));
    if (!isModelDataKey(key)) {
      SceneModelEntry probe;
      if (key != "object" && readModelKey(probe, key, value, lineNumber))
        fail(lineNumber, "'" + key + "' is where a copy stands or how it moves, which each "
                                     "scene sets, not the object");
      fail(lineNumber, "unknown key '" + key + "'");
    }
    readModelKey(object.model, key, value, lineNumber);
  }
  object.model.overrides.clear();
  if (const std::optional<std::string> problem = problemWith(object.model))
    fail(0, *problem);
  return object;
}

ObjectAsset loadFromFile(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("can't open " + path);
  return parse(in, path);
}

std::string write(const ObjectAsset &object) {
  SceneModelEntry model = object.model;
  model.transform = {};
  model.animation = {};
  model.object.clear();
  model.overrides.clear();
  return object.comment + modelKeysText(model);
}

void saveToFile(const ObjectAsset &object, const std::string &path) {
  namespace fs = std::filesystem;
  std::error_code error;
  fs::create_directories(fs::path(path).parent_path(), error);
  const std::string temporary = path + ".tmp";
  {
    std::ofstream out(temporary);
    out << write(object);
    if (!out)
      throw std::runtime_error("couldn't write " + temporary);
  }
  fs::rename(temporary, path, error);
  if (error) {
    fs::remove(temporary, error);
    throw std::runtime_error("couldn't replace " + path);
  }
}

} // namespace ObjectIO

SceneModelEntry resolvePrefab(const SceneModelEntry &copy, const SceneModelEntry &object) {
  SceneModelEntry resolved = object;
  resolved.transform = copy.transform;
  resolved.animation = copy.animation;
  resolved.info = copy.info;
  resolved.object = copy.object;
  resolved.overrides = copy.overrides;
  for (const std::string &key : copy.overrides)
    copyDataKey(resolved, copy, key);
  return resolved;
}

void noteOverrides(const SceneModelEntry &before, SceneModelEntry &edited) {
  if (edited.object.empty())
    return;
  for (const std::string_view key : kDataKeys)
    if (dataKeyDiffers(before, edited, key) && !listed(edited.overrides, key))
      edited.overrides.emplace_back(key);
}

SceneModelEntry revertOverrides(const SceneModelEntry &copy, const SceneModelEntry &object) {
  SceneModelEntry plain = copy;
  plain.overrides.clear();
  return resolvePrefab(plain, object);
}

SceneModelEntry unpackPrefab(const SceneModelEntry &copy) {
  SceneModelEntry plain = copy;
  plain.object.clear();
  plain.overrides.clear();
  return plain;
}

ObjectAsset prefabFrom(SceneModelEntry &entry, const std::string &path) {
  ObjectAsset object;
  object.model = unpackPrefab(entry);
  object.model.transform = {};
  object.model.animation = {};
  object.model.info = {};
  entry.object = path;
  entry.overrides.clear();
  return object;
}
