#pragma once

#include "Scene.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

// The Outliner (the editor's list of everything in a scene): a row for each
// entry with its main settings, a name for it, search, filters and sorting.
// Unreal's World Outliner and Property Matrix in one table. No ImGui here,
// so it's tested like the rest of heliora_core.

enum class OutlinerKind { PlayerStart, Shape, Model, Grid, Character };

struct OutlinerRow {
  SceneEntryRef ref;
  OutlinerKind category = OutlinerKind::Shape;
  std::string name;
  std::string kind;     // "Box", "Model", "Prefab", "Grid", "Character", "Player start"
  std::string material; // empty without one
  std::string body;     // "static", "dynamic", "kinematic", or empty
  int breakPieces = 0;  // 0 if it doesn't break
  std::optional<float> mass; // kg, one copy for a grid; unknown without a body or a mesh's volume
  glm::vec3 position{0.0f};
  int order = 0; // its place in the file (the player start first)
};

// The player start, then every model, grid and character in file order.
std::vector<OutlinerRow> outlinerRows(const SceneDescription &scene);

// A name from the comment above an entry (its first line's text up to a
// colon or a full stop, at most 40 letters), or fallback without one.
std::string entryName(std::string_view comment, const std::string &fallback);
// A name from what it is: "Box 6 x 3 x 0.3", "Sphere r 0.25", a model or
// prefab file's name.
std::string describedName(const SceneModelEntry &entry);

// kg: its mass, or its volume times its density (its own, its material's,
// or water's, as bodies get) for a shape; nothing without a body, and for a
// model file without a mass (its volume needs its mesh).
std::optional<float> entryMass(const SceneModelEntry &entry);

// Whether its name, kind or material holds the text, in any case; anything
// does for no text.
bool rowMatches(const OutlinerRow &row, std::string_view text);

struct OutlinerFilter {
  bool shapes = true;
  bool models = true; // model files and prefabs
  bool grids = true;
  bool characters = true;
  bool breakableOnly = false;
};
bool rowPasses(const OutlinerRow &row, const OutlinerFilter &filter);

enum class OutlinerColumn { Order, Name, Kind, Material, Body, Breaks, Mass, Position };
// Sorted by a column (names in any case, positions by distance from the
// origin), rows that tie keeping their file order.
void sortRows(std::vector<OutlinerRow> &rows, OutlinerColumn column, bool ascending);
