#pragma once

#include "SettingsFile.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

// A gun's model as tools/weapon_convert writes it (step 1.6): a .parts file
// in the settings format, naming each part's .gltf (next to it) and its
// pivot in gun space, and the sockets (muzzle, ejection port, magazine,
// sights) with their axes. Gun space: the trigger at the origin, +z along
// the barrel, +y up, +x to the gun's left, metres.

struct WeaponPart {
  std::string name; // "frame", "magazine", "bolt"...
  std::string file; // its .gltf, next to the .parts file
  glm::vec3 pivot{0.0f};
};

struct WeaponSocket {
  glm::vec3 position{0.0f};
  glm::vec3 x{1.0f, 0.0f, 0.0f}, y{0.0f, 1.0f, 0.0f}, z{0.0f, 0.0f, 1.0f};
};

struct WeaponParts {
  std::string name;
  float length = 0.0f; // m
  std::vector<WeaponPart> parts; // in file order, the frame first
  std::vector<std::pair<std::string, WeaponSocket>> sockets;
  const WeaponPart *part(std::string_view name) const;
  std::optional<WeaponSocket> socket(std::string_view name) const;
};

WeaponParts weaponPartsFrom(const SettingsFile &file);

// A socket as a transform in gun space: its axes and its place.
glm::mat4 socketMatrix(const WeaponSocket &socket);
