#include "SettingsFile.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace {

std::string trim(std::string_view s) {
  const size_t begin = s.find_first_not_of(" \t\r");
  if (begin == std::string_view::npos)
    return "";
  const size_t end = s.find_last_not_of(" \t\r");
  return std::string(s.substr(begin, end - begin + 1));
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

// The whole text as one number, or nothing: "8.5" isn't an int, and NaN or
// infinity would get past every range check.
template <typename T> std::optional<T> parseNumber(std::string_view text) {
  T value{};
  const char *end = text.data() + text.size();
  const auto [stop, error] = std::from_chars(text.data(), end, value);
  if (error != std::errc() || stop != end)
    return std::nullopt;
  if constexpr (std::is_floating_point_v<T>)
    if (!std::isfinite(value))
      return std::nullopt;
  return value;
}

std::string formatFloat(float value) { return std::format("{}", value); }

template <class... Ts> struct Overloaded : Ts... {
  using Ts::operator()...;
};

} // namespace

SettingsFile SettingsFile::load(const std::string &path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    if (std::filesystem::exists(path))
      spdlog::warn("Couldn't read settings file {}, using the defaults", path);
    else
      spdlog::info("No settings file at {}, using the defaults", path);
    return {};
  }
  return parse(in, path);
}

SettingsFile SettingsFile::parse(std::istream &in,
                                 const std::string &sourceName) {
  SettingsFile file;
  std::string section;
  bool inSection = false;
  std::string line;
  int lineNumber = 0;
  auto skip = [&](std::string_view why) {
    spdlog::warn("{}:{}: {}, line skipped", sourceName, lineNumber, why);
  };
  while (std::getline(in, line)) {
    ++lineNumber;
    if (const size_t hash = line.find('#'); hash != std::string::npos)
      line.erase(hash);
    const std::string text = trim(line);
    if (text.empty())
      continue;
    if (text.front() == '[') {
      // A bad header could be any section, so its keys can't go anywhere.
      inSection = text.size() > 2 && text.back() == ']';
      if (!inSection) {
        skip("bad section header, the keys under it are skipped too");
        continue;
      }
      section = lower(trim(std::string_view(text).substr(1, text.size() - 2)));
      continue;
    }
    if (!inSection) {
      skip("key outside a [section]");
      continue;
    }
    const size_t eq = text.find('=');
    if (eq == std::string::npos) {
      skip("expected 'key = value'");
      continue;
    }
    std::string key = lower(trim(std::string_view(text).substr(0, eq)));
    if (key.empty()) {
      skip("no key before '='");
      continue;
    }
    file.setText(section, key, trim(std::string_view(text).substr(eq + 1)));
  }
  return file;
}

std::optional<std::string> SettingsFile::text(std::string_view section,
                                              std::string_view key) const {
  for (const Section &s : sections)
    if (s.name == section)
      for (const Entry &e : s.entries)
        if (e.key == key)
          return e.value;
  return std::nullopt;
}

std::optional<float> SettingsFile::getFloat(std::string_view section,
                                            std::string_view key) const {
  const std::optional<std::string> value = text(section, key);
  if (!value)
    return std::nullopt;
  const std::optional<float> number = parseNumber<float>(*value);
  if (!number)
    spdlog::warn("settings: [{}] {} = '{}' isn't a number, ignored", section,
                 key, *value);
  return number;
}

std::optional<int64_t> SettingsFile::getInt(std::string_view section,
                                            std::string_view key) const {
  const std::optional<std::string> value = text(section, key);
  if (!value)
    return std::nullopt;
  const std::optional<int64_t> number = parseNumber<int64_t>(*value);
  if (!number)
    spdlog::warn("settings: [{}] {} = '{}' isn't a whole number, ignored",
                 section, key, *value);
  return number;
}

std::optional<bool> SettingsFile::getBool(std::string_view section,
                                          std::string_view key) const {
  const std::optional<std::string> value = text(section, key);
  if (!value)
    return std::nullopt;
  const std::string word = lower(*value);
  if (word == "true" || word == "on" || word == "yes" || word == "1")
    return true;
  if (word == "false" || word == "off" || word == "no" || word == "0")
    return false;
  spdlog::warn("settings: [{}] {} = '{}' isn't true or false, ignored",
               section, key, *value);
  return std::nullopt;
}

std::optional<glm::vec3> SettingsFile::getVec3(std::string_view section,
                                               std::string_view key) const {
  const std::optional<std::string> value = text(section, key);
  if (!value)
    return std::nullopt;
  std::istringstream words(*value);
  std::vector<float> numbers;
  for (std::string word; words >> word;) {
    const std::optional<float> number = parseNumber<float>(word);
    if (!number) {
      numbers.clear();
      break;
    }
    numbers.push_back(*number);
  }
  if (numbers.size() == 1)
    return glm::vec3(numbers[0]);
  if (numbers.size() == 3)
    return glm::vec3(numbers[0], numbers[1], numbers[2]);
  spdlog::warn("settings: [{}] {} = '{}' needs one or three numbers, ignored",
               section, key, *value);
  return std::nullopt;
}

void SettingsFile::setText(std::string_view section, std::string_view key,
                           std::string value) {
  auto s = std::find_if(sections.begin(), sections.end(),
                        [&](const Section &s) { return s.name == section; });
  if (s == sections.end())
    s = sections.insert(sections.end(), Section{std::string(section), {}});
  for (Entry &e : s->entries)
    if (e.key == key) {
      e.value = std::move(value);
      return;
    }
  s->entries.push_back({std::string(key), std::move(value)});
}

void SettingsFile::setFloat(std::string_view section, std::string_view key,
                            float value) {
  setText(section, key, formatFloat(value));
}

void SettingsFile::setInt(std::string_view section, std::string_view key,
                          int64_t value) {
  setText(section, key, std::format("{}", value));
}

void SettingsFile::setBool(std::string_view section, std::string_view key,
                           bool value) {
  setText(section, key, value ? "true" : "false");
}

void SettingsFile::setVec3(std::string_view section, std::string_view key,
                           glm::vec3 value) {
  setText(section, key,
          std::format("{} {} {}", formatFloat(value.x), formatFloat(value.y),
                      formatFloat(value.z)));
}

void SettingsFile::remove(std::string_view section, std::string_view key) {
  for (Section &s : sections)
    if (s.name == section)
      std::erase_if(s.entries, [&](const Entry &e) { return e.key == key; });
}

std::vector<std::string> SettingsFile::keys(std::string_view section) const {
  std::vector<std::string> names;
  for (const Section &s : sections)
    if (s.name == section)
      for (const Entry &e : s.entries)
        names.push_back(e.key);
  return names;
}

std::string SettingsFile::write() const {
  std::string out =
      "# Heliora settings, shared by every scene. The Save buttons in the\n"
      "# Render and Physics tabs write this file, and comments aren't kept.\n";
  for (const Section &s : sections) {
    if (s.entries.empty())
      continue;
    out += std::format("\n[{}]\n", s.name);
    for (const Entry &e : s.entries)
      out += std::format("{} = {}\n", e.key, e.value);
  }
  return out;
}

void SettingsFile::save(const std::string &path) const {
  const std::string temporary = path + ".tmp";
  {
    std::ofstream file(temporary, std::ios::trunc);
    if (!file.is_open())
      throw std::runtime_error("Couldn't write " + temporary);
    file << write();
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

void readSettings(const SettingsFile &file,
                  std::span<const SettingBinding> bindings,
                  std::string_view section) {
  // Also held to the variable's own type, so a huge number in the file
  // can't overflow an int.
  auto clampTo = [](const SettingBinding &b, double value, double typeMin,
                    double typeMax) {
    const double lo = std::max(b.min, typeMin);
    const double hi = std::min(b.max, typeMax);
    const double clamped = std::clamp(value, lo, hi);
    if (clamped != value)
      spdlog::warn("settings: [{}] {} = {} is outside {} to {}, using {}",
                   b.section, b.key, value, lo, hi, clamped);
    return clamped;
  };
  constexpr double kFloatMax = std::numeric_limits<float>::max();
  for (const SettingBinding &b : bindings) {
    if (!section.empty() && b.section != section)
      continue;
    std::visit(
        Overloaded{
            [&](float *v) {
              if (const auto x = file.getFloat(b.section, b.key))
                *v = float(clampTo(b, *x, -kFloatMax, kFloatMax));
            },
            [&](int *v) {
              if (const auto x = file.getInt(b.section, b.key))
                *v = int(clampTo(b, double(*x),
                                 std::numeric_limits<int>::min(),
                                 std::numeric_limits<int>::max()));
            },
            [&](uint32_t *v) {
              if (const auto x = file.getInt(b.section, b.key))
                *v = uint32_t(clampTo(b, double(*x), 0.0,
                                      std::numeric_limits<uint32_t>::max()));
            },
            [&](bool *v) {
              if (const auto x = file.getBool(b.section, b.key))
                *v = *x;
            },
            [&](glm::vec3 *v) {
              if (const auto x = file.getVec3(b.section, b.key))
                for (int i = 0; i < 3; ++i)
                  (*v)[i] = float(clampTo(b, (*x)[i], -kFloatMax, kFloatMax));
            },
        },
        b.value);
  }

  // A key in a section we read that nothing reads is probably misspelt.
  std::vector<std::string_view> checked;
  for (const SettingBinding &b : bindings) {
    if ((!section.empty() && b.section != section) ||
        std::ranges::find(checked, b.section) != checked.end())
      continue;
    checked.push_back(b.section);
    for (const std::string &key : file.keys(b.section)) {
      const bool known = std::ranges::any_of(bindings, [&](const auto &other) {
        return other.section == b.section && other.key == key;
      });
      if (!known)
        spdlog::warn("settings: unknown key [{}] {}, ignored", b.section, key);
    }
  }
}

void writeSettings(SettingsFile &file, std::span<const SettingBinding> bindings,
                   std::string_view section, const SettingsFile &defaults) {
  for (const SettingBinding &b : bindings) {
    if (!section.empty() && b.section != section)
      continue;
    std::visit(Overloaded{
                   [&](float *v) { file.setFloat(b.section, b.key, *v); },
                   [&](int *v) { file.setInt(b.section, b.key, *v); },
                   [&](uint32_t *v) { file.setInt(b.section, b.key, *v); },
                   [&](bool *v) { file.setBool(b.section, b.key, *v); },
                   [&](glm::vec3 *v) { file.setVec3(b.section, b.key, *v); },
               },
               b.value);
    const std::optional<std::string> written = file.text(b.section, b.key);
    if (written == defaults.text(b.section, b.key))
      file.remove(b.section, b.key);
  }
}
