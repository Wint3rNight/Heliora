#include "WeaponParts.h"

const WeaponPart *WeaponParts::part(std::string_view wanted) const {
  for (const WeaponPart &p : parts)
    if (p.name == wanted)
      return &p;
  return nullptr;
}

std::optional<WeaponSocket> WeaponParts::socket(std::string_view wanted) const {
  for (const auto &[name, socket] : sockets)
    if (name == wanted)
      return socket;
  return std::nullopt;
}

WeaponParts weaponPartsFrom(const SettingsFile &file) {
  WeaponParts w;
  w.name = file.text("weapon", "name").value_or("");
  w.length = file.getFloat("weapon", "length").value_or(0.0f);
  for (const std::string &section : file.sectionNames()) {
    if (section.rfind("part ", 0) == 0) {
      WeaponPart part;
      part.name = section.substr(5);
      part.file = file.text(section, "file").value_or("");
      part.pivot = file.getVec3(section, "pivot").value_or(glm::vec3(0.0f));
      if (!part.file.empty())
        w.parts.push_back(part);
    } else if (section.rfind("socket ", 0) == 0) {
      WeaponSocket socket;
      socket.position = file.getVec3(section, "position").value_or(glm::vec3(0.0f));
      socket.x = file.getVec3(section, "x").value_or(socket.x);
      socket.y = file.getVec3(section, "y").value_or(socket.y);
      socket.z = file.getVec3(section, "z").value_or(socket.z);
      w.sockets.push_back({section.substr(7), socket});
    }
  }
  return w;
}

glm::mat4 socketMatrix(const WeaponSocket &s) {
  return glm::mat4(glm::vec4(s.x, 0.0f), glm::vec4(s.y, 0.0f), glm::vec4(s.z, 0.0f),
                   glm::vec4(s.position, 1.0f));
}
