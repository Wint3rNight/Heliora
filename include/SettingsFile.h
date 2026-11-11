#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

// Settings every scene shares, in the scene files' format: `key = value`
// lines under [sections], `#` starts a comment. Values stay as text until
// asked for, so a key the game doesn't know survives a save. Comments don't:
// the file is meant to be written by the game.
class SettingsFile {
public:
  // A missing file gives an empty one, so the code defaults stay.
  static SettingsFile load(const std::string &path);
  // A line that can't be read is skipped with a warning naming its line.
  static SettingsFile parse(std::istream &in, const std::string &sourceName);

  // Nothing if the key is missing, or if its value isn't that type (which
  // is logged). Numbers must be finite.
  std::optional<float> getFloat(std::string_view section,
                                std::string_view key) const;
  std::optional<int64_t> getInt(std::string_view section,
                                std::string_view key) const;
  std::optional<bool> getBool(std::string_view section,
                              std::string_view key) const;
  // One number for all three, or three.
  std::optional<glm::vec3> getVec3(std::string_view section,
                                   std::string_view key) const;
  // The value as written.
  std::optional<std::string> text(std::string_view section,
                                  std::string_view key) const;

  void setFloat(std::string_view section, std::string_view key, float value);
  void setInt(std::string_view section, std::string_view key, int64_t value);
  void setBool(std::string_view section, std::string_view key, bool value);
  void setVec3(std::string_view section, std::string_view key,
               glm::vec3 value);
  void remove(std::string_view section, std::string_view key);

  // The keys in a section, in file order.
  std::vector<std::string> keys(std::string_view section) const;

  std::string write() const;
  // Writes through a temporary file and a rename, like scene saves, so a
  // failure never leaves half a file. Throws on failure.
  void save(const std::string &path) const;

private:
  struct Entry {
    std::string key;
    std::string value;
  };
  struct Section {
    std::string name;
    std::vector<Entry> entries;
  };
  std::vector<Section> sections; // in file order, and new ones at the end

  void setText(std::string_view section, std::string_view key,
               std::string value);
};

// A setting the program keeps in a variable: where it sits in the file, the
// variable, and the range it must stay in. A loaded value outside the range
// is clamped, with a warning. Bools ignore the range.
struct SettingBinding {
  std::string_view section;
  std::string_view key;
  std::variant<float *, int *, uint32_t *, bool *, glm::vec3 *> value;
  double min = -1e30;
  double max = 1e30;
};

// Sets each bound variable in section ("" for every section) from file.
// A missing key leaves its variable alone. Keys nothing is bound to are
// logged, since that's usually a typo.
void readSettings(const SettingsFile &file,
                  std::span<const SettingBinding> bindings,
                  std::string_view section = {});

// Copies each bound variable in section ("" for every section) into file.
// A value that matches defaults is removed instead, so a saved file holds
// only what was tuned, and later changes to a code default still apply.
void writeSettings(SettingsFile &file, std::span<const SettingBinding> bindings,
                   std::string_view section = {},
                   const SettingsFile &defaults = {});
