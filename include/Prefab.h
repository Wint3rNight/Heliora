#pragma once

#include "Scene.h"

#include <istream>
#include <string>

// Object assets (prefabs): a .object file in Resources/Objects/ says what
// an object is (its model or shape, size, colour, material, body), and a
// scene's [model] with `object = ...` is a copy of it, with its own place
// and any values it overrides. Changing the file changes every copy.

struct ObjectAsset {
  std::string comment; // the # lines at the top
  SceneModelEntry model; // its place left at the defaults
  bool operator==(const ObjectAsset &) const = default;
};

namespace ObjectIO {
// The [model] keys that say what it is, as `key = value` lines; a place or
// movement key, an unknown key or nothing to draw is an error naming the
// line.
ObjectAsset parse(std::istream &in, const std::string &sourceName);
ObjectAsset loadFromFile(const std::string &path);
std::string write(const ObjectAsset &object);
// Through a temporary file and a rename, like scene saves. Throws.
void saveToFile(const ObjectAsset &object, const std::string &path);
} // namespace ObjectIO

// A copy with its object's values in: everything that says what it is from
// the object, apart from the keys the copy overrides; its place, movement,
// comment and link from the copy.
SceneModelEntry resolvePrefab(const SceneModelEntry &copy, const SceneModelEntry &object);
// After an edit of a copy (before to edited), the keys it changed that say
// what it is join its overrides.
void noteOverrides(const SceneModelEntry &before, SceneModelEntry &edited);
// The copy with no overrides: its object's values again, its place kept.
SceneModelEntry revertOverrides(const SceneModelEntry &copy, const SceneModelEntry &object);
// No longer a copy: an ordinary object with the same values.
SceneModelEntry unpackPrefab(const SceneModelEntry &copy);
// An object asset made from an object (without its place or movement), and
// the object turned into a copy of it at path.
ObjectAsset prefabFrom(SceneModelEntry &entry, const std::string &path);
