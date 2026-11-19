#include "Outliner.h"

#include "ScenePhysics.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <format>

namespace {

std::string lower(std::string_view text) {
  std::string out(text);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

std::string trimmed(std::string_view text) {
  size_t first = 0, last = text.size();
  while (first < last && std::isspace(static_cast<unsigned char>(text[first])))
    ++first;
  while (last > first && std::isspace(static_cast<unsigned char>(text[last - 1])))
    --last;
  return std::string(text.substr(first, last - first));
}

std::string stem(const std::string &path) { return std::filesystem::path(path).stem().string(); }

const char *shapeName(PrimitiveShape shape) {
  switch (shape) {
  case PrimitiveShape::Box: return "Box";
  case PrimitiveShape::Sphere: return "Sphere";
  case PrimitiveShape::Capsule: return "Capsule";
  case PrimitiveShape::Plane: return "Plane";
  }
  return "Shape";
}

const char *bodyName(const std::optional<PhysicsMotion> &body) {
  if (!body)
    return "";
  switch (*body) {
  case PhysicsMotion::Static: return "static";
  case PhysicsMotion::Dynamic: return "dynamic";
  case PhysicsMotion::Kinematic: return "kinematic";
  }
  return "";
}

std::string kindOf(const SceneModelEntry &e) {
  if (!e.object.empty())
    return "Prefab";
  if (e.shape)
    return shapeName(*e.shape);
  return "Model";
}

OutlinerRow modelRow(const SceneModelEntry &e, SceneEntryRef ref) {
  OutlinerRow row;
  row.ref = ref;
  row.category = e.shape && e.object.empty() ? OutlinerKind::Shape : OutlinerKind::Model;
  row.name = entryName(e.info.comment, describedName(e));
  row.kind = kindOf(e);
  row.material = e.material.value_or("");
  row.body = bodyName(e.body);
  row.breakPieces = e.breakPieces.value_or(0);
  row.mass = entryMass(e);
  row.position = e.transform.position;
  row.order = e.info.order;
  return row;
}

} // namespace

std::string entryName(std::string_view comment, const std::string &fallback) {
  // The first line with any text, without its '#'.
  std::string line;
  size_t at = 0;
  while (at <= comment.size() && line.empty()) {
    const size_t end = std::min(comment.find('\n', at), comment.size());
    std::string_view raw = comment.substr(at, end - at);
    while (!raw.empty() && (raw.front() == '#' || std::isspace(static_cast<unsigned char>(raw.front()))))
      raw.remove_prefix(1);
    line = trimmed(raw);
    at = end + 1;
  }
  if (line.empty())
    return fallback;
  // Up to a colon, or a full stop that ends a sentence (not one in "0.3").
  for (size_t i = 0; i < line.size(); ++i) {
    const bool stop = line[i] == ':' ||
                      (line[i] == '.' && (i + 1 == line.size() || line[i + 1] == ' '));
    if (stop) {
      line = trimmed(line.substr(0, i));
      break;
    }
  }
  constexpr size_t kLongest = 40;
  if (line.size() > kLongest) {
    size_t cut = line.rfind(' ', kLongest - 3);
    if (cut == std::string::npos || cut == 0)
      cut = kLongest - 3;
    line = trimmed(line.substr(0, cut)) + "...";
  }
  return line.empty() ? fallback : line;
}

std::string describedName(const SceneModelEntry &e) {
  if (!e.object.empty())
    return stem(e.object);
  if (!e.shape)
    return stem(e.path);
  switch (*e.shape) {
  case PrimitiveShape::Box:
    return std::format("Box {} x {} x {}", e.size.x, e.size.y, e.size.z);
  case PrimitiveShape::Sphere: return std::format("Sphere r {}", e.radius);
  case PrimitiveShape::Capsule: return std::format("Capsule r {} h {}", e.radius, e.height);
  case PrimitiveShape::Plane: return std::format("Plane {} x {}", e.size.x, e.size.z);
  }
  return "Shape";
}

std::optional<float> entryMass(const SceneModelEntry &e) {
  if (!e.body)
    return std::nullopt;
  if (e.mass)
    return *e.mass;
  if (!e.shape || *e.shape == PrimitiveShape::Plane)
    return std::nullopt;
  float density = 1000.0f; // PhysicsBodyDesc's
  if (e.material)
    if (const std::optional<PhysicsMaterial> m = findMaterial(*e.material))
      density = m->density;
  if (e.density)
    density = *e.density;
  const float s = e.transform.scale;
  constexpr float kPi = 3.14159265f;
  float volume = 0.0f;
  switch (*e.shape) {
  case PrimitiveShape::Box: volume = e.size.x * e.size.y * e.size.z; break;
  case PrimitiveShape::Sphere: volume = 4.0f / 3.0f * kPi * e.radius * e.radius * e.radius; break;
  case PrimitiveShape::Capsule: {
    const float r = e.radius, straight = std::max(e.height - 2.0f * r, 0.0f);
    volume = kPi * r * r * straight + 4.0f / 3.0f * kPi * r * r * r;
    break;
  }
  case PrimitiveShape::Plane: break;
  }
  return volume * s * s * s * density;
}

std::vector<OutlinerRow> outlinerRows(const SceneDescription &scene) {
  using Kind = SceneEntryRef::Kind;
  std::vector<OutlinerRow> rows;
  for (size_t i = 0; i < scene.models.size(); ++i)
    rows.push_back(modelRow(scene.models[i], {Kind::Model, static_cast<int>(i)}));
  for (size_t i = 0; i < scene.grids.size(); ++i) {
    const SceneGridEntry &g = scene.grids[i];
    OutlinerRow row = modelRow(g.model, {Kind::Grid, static_cast<int>(i)});
    row.category = OutlinerKind::Grid;
    row.kind = "Grid";
    // Copies of a shape are named for the shape, not its size.
    const std::string copies = g.model.shape && g.model.object.empty()
                                   ? shapeName(*g.model.shape)
                                   : describedName(g.model);
    row.name = entryName(g.info.comment, std::format("Grid {} x {} x {} of {}", g.count.x,
                                                      g.count.y, g.count.z, copies));
    row.order = g.info.order;
    rows.push_back(row);
  }
  for (size_t i = 0; i < scene.characters.size(); ++i) {
    const SceneCharacterEntry &c = scene.characters[i];
    OutlinerRow row;
    row.ref = {Kind::Character, static_cast<int>(i)};
    row.category = OutlinerKind::Character;
    row.name = entryName(c.info.comment, stem(c.file));
    row.kind = "Character";
    row.position = c.position;
    row.order = c.info.order;
    rows.push_back(row);
  }
  std::stable_sort(rows.begin(), rows.end(),
                   [](const OutlinerRow &a, const OutlinerRow &b) { return a.order < b.order; });
  if (scene.player) {
    OutlinerRow row;
    row.ref = {Kind::Player, 0};
    row.category = OutlinerKind::PlayerStart;
    row.name = "Player start";
    row.kind = "Player start";
    row.position = scene.player->position;
    row.order = -1;
    rows.insert(rows.begin(), row);
  }
  return rows;
}

bool rowMatches(const OutlinerRow &row, std::string_view text) {
  const std::string wanted = lower(trimmed(text));
  if (wanted.empty())
    return true;
  for (const std::string *field : {&row.name, &row.kind, &row.material})
    if (lower(*field).find(wanted) != std::string::npos)
      return true;
  return false;
}

bool rowPasses(const OutlinerRow &row, const OutlinerFilter &filter) {
  if (filter.breakableOnly && row.breakPieces <= 0)
    return false;
  switch (row.category) {
  case OutlinerKind::PlayerStart: return true;
  case OutlinerKind::Shape: return filter.shapes;
  case OutlinerKind::Model: return filter.models;
  case OutlinerKind::Grid: return filter.grids;
  case OutlinerKind::Character: return filter.characters;
  }
  return true;
}

void sortRows(std::vector<OutlinerRow> &rows, OutlinerColumn column, bool ascending) {
  auto key = [&](const OutlinerRow &a, const OutlinerRow &b) -> int {
    auto compare = [](const auto &x, const auto &y) { return x < y ? -1 : (y < x ? 1 : 0); };
    switch (column) {
    case OutlinerColumn::Order: return compare(a.order, b.order);
    case OutlinerColumn::Name: return compare(lower(a.name), lower(b.name));
    case OutlinerColumn::Kind: return compare(lower(a.kind), lower(b.kind));
    case OutlinerColumn::Material: return compare(a.material, b.material);
    case OutlinerColumn::Body: return compare(a.body, b.body);
    case OutlinerColumn::Breaks: return compare(a.breakPieces, b.breakPieces);
    case OutlinerColumn::Mass: return compare(a.mass.value_or(-1.0f), b.mass.value_or(-1.0f));
    case OutlinerColumn::Position:
      return compare(glm::length(a.position), glm::length(b.position));
    }
    return 0;
  };
  std::stable_sort(rows.begin(), rows.end(), [&](const OutlinerRow &a, const OutlinerRow &b) {
    const int k = key(a, b);
    if (k != 0)
      return ascending ? k < 0 : k > 0;
    return a.order < b.order;
  });
}
