#include "CharacterFile.h"

#include <array>
#include <charconv>
#include <cmath>
#include <format>
#include <fstream>
#include <stdexcept>

namespace {

constexpr std::array<const char *, static_cast<size_t>(ClipSlot::Count)>
    kSlotNames = {"idle",           "walk_forward",  "walk_back",
                  "walk_left",      "walk_right",    "run_forward",
                  "run_back",       "run_left",      "run_right",
                  "sprint",         "jump",          "fall",
                  "land",           "land_hard",     "crouch_idle",
                  "crouch_forward", "crouch_back",   "crouch_left",
                  "crouch_right",   "roll",          "dodge_forward",
                  "dodge_back",     "dodge_left",    "dodge_right"};

std::string trim(std::string_view s) {
  const size_t first = s.find_first_not_of(" \t\r");
  if (first == std::string_view::npos)
    return "";
  const size_t last = s.find_last_not_of(" \t\r");
  return std::string(s.substr(first, last - first + 1));
}

// Reads one file, remembering where each clip started so a clip that's
// missing something can say where it is.
class Parser {
public:
  explicit Parser(std::string source) : source(std::move(source)) {}

  CharacterDescription run(std::istream &in) {
    std::string line;
    while (std::getline(in, line)) {
      ++lineNumber;
      // Whole-line comments are kept for the model line or the [clip] that
      // comes next, so a save doesn't lose the notes.
      if (const std::string whole = trim(line); !whole.empty() && whole[0] == '#') {
        notes += whole + "\n";
        continue;
      }
      if (const size_t hash = line.find('#'); hash != std::string::npos)
        line.erase(hash);
      const std::string text = trim(line);
      if (text.empty())
        continue;
      if (text == "[clip]") {
        finishClip();
        open = true;
        current = {};
        current.comment = std::move(notes);
        notes.clear();
        loopGiven = false;
        slotGiven = false;
        clipLine = lineNumber;
        continue;
      }
      const size_t equals = text.find('=');
      if (equals == std::string::npos)
        fail(lineNumber, "expected 'key = value' or [clip], got '" + text + "'");
      setKey(trim(std::string_view(text).substr(0, equals)),
             trim(std::string_view(text).substr(equals + 1)));
    }
    finishClip();
    if (result.model.empty())
      fail(0, "no 'model = ...' line");
    return result;
  }

private:
  std::string source;
  int lineNumber = 0;
  CharacterDescription result;
  CharacterClip current;
  bool open = false; // inside a [clip]
  std::string notes; // comment lines waiting for what they're above
  bool slotGiven = false;
  bool loopGiven = false;
  int clipLine = 0;

  [[noreturn]] void fail(int line, const std::string &message) const {
    throw std::runtime_error(
        line > 0 ? std::format("{}, line {}: {}", source, line, message)
                 : std::format("{}: {}", source, message));
  }

  void setKey(const std::string &key, const std::string &value) {
    if (!open) {
      if (key != "model")
        fail(lineNumber, "unknown key '" + key + "' (before any [clip], only "
                                                 "'model')");
      result.model = value;
      result.comment = std::move(notes);
      notes.clear();
      return;
    }
    if (key == "slot") {
      const std::optional<ClipSlot> slot = clipSlotFromName(value);
      if (!slot)
        fail(lineNumber, "unknown slot '" + value + "'");
      current.slot = *slot;
      slotGiven = true;
    } else if (key == "file") {
      current.file = value;
    } else if (key == "start") {
      float start = 0.0f;
      const auto [end, error] =
          std::from_chars(value.data(), value.data() + value.size(), start);
      if (error != std::errc() || end != value.data() + value.size() ||
          !std::isfinite(start) || start < 0.0f)
        fail(lineNumber, "'start' must be seconds, 0 or more, got '" + value + "'");
      current.start = start;
    } else if (key == "end") {
      float stop = 0.0f;
      const auto [end, error] =
          std::from_chars(value.data(), value.data() + value.size(), stop);
      if (error != std::errc() || end != value.data() + value.size() ||
          !std::isfinite(stop) || stop < 0.0f)
        fail(lineNumber, "'end' must be seconds, 0 or more, got '" + value + "'");
      current.end = stop;
    } else if (key == "loop") {
      if (value != "true" && value != "false")
        fail(lineNumber, "'loop' must be true or false, got '" + value + "'");
      current.loop = value == "true";
      loopGiven = true;
    } else {
      fail(lineNumber, "unknown key '" + key + "' in a [clip]");
    }
  }

  void finishClip() {
    if (!open)
      return;
    open = false;
    if (!slotGiven)
      fail(clipLine, "this [clip] has no 'slot'");
    if (current.file.empty())
      fail(clipLine, "this [clip] has no 'file'");
    if (result.clip(current.slot))
      fail(clipLine, std::string("slot '") + clipSlotName(current.slot) +
                         "' is given twice");
    if (current.end != 0.0f && current.end <= current.start)
      fail(clipLine, std::format("'end' ({}) must come after 'start' ({})",
                                 current.end, current.start));
    if (!loopGiven)
      current.loop = defaultLoop(current.slot);
    result.clips.push_back(current);
  }
};

} // namespace

const char *clipSlotName(ClipSlot slot) {
  const size_t i = static_cast<size_t>(slot);
  return i < kSlotNames.size() ? kSlotNames[i] : "";
}

std::optional<ClipSlot> clipSlotFromName(std::string_view name) {
  for (size_t i = 0; i < kSlotNames.size(); ++i)
    if (name == kSlotNames[i])
      return static_cast<ClipSlot>(i);
  return std::nullopt;
}

bool defaultLoop(ClipSlot slot) {
  switch (slot) {
  case ClipSlot::Jump:
  case ClipSlot::Land:
  case ClipSlot::LandHard:
  case ClipSlot::Roll:
  case ClipSlot::DodgeForward:
  case ClipSlot::DodgeBack:
  case ClipSlot::DodgeLeft:
  case ClipSlot::DodgeRight:
    return false;
  default:
    return true;
  }
}

const CharacterClip *CharacterDescription::clip(ClipSlot slot) const {
  for (const CharacterClip &c : clips)
    if (c.slot == slot)
      return &c;
  return nullptr;
}

void CharacterDescription::setClip(const CharacterClip &clip) {
  for (CharacterClip &c : clips)
    if (c.slot == clip.slot) {
      std::string notes = clip.comment.empty() ? c.comment : clip.comment;
      c = clip;
      c.comment = std::move(notes);
      return;
    }
  clips.push_back(clip);
}

void CharacterDescription::removeClip(ClipSlot slot) {
  std::erase_if(clips, [&](const CharacterClip &c) { return c.slot == slot; });
}

namespace CharacterIO {

CharacterDescription parse(std::istream &in, const std::string &sourceName) {
  return Parser(sourceName).run(in);
}

CharacterDescription loadFromFile(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("can't open " + path);
  return parse(in, path);
}

std::string write(const CharacterDescription &character) {
  std::string out = character.comment + "model = " + character.model + "\n";
  for (const CharacterClip &clip : character.clips) {
    out += std::format("\n{}[clip]\nslot = {}\nfile = {}\n", clip.comment,
                       clipSlotName(clip.slot), clip.file);
    if (clip.start != 0.0f)
      out += std::format("start = {}\n", clip.start);
    if (clip.end != 0.0f)
      out += std::format("end = {}\n", clip.end);
    if (clip.loop != defaultLoop(clip.slot))
      out += std::format("loop = {}\n", clip.loop ? "true" : "false");
  }
  return out;
}

} // namespace CharacterIO
