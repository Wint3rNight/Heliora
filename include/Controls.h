#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

// The player's controls: each action bound to one key or mouse button,
// rebound in the Controls tab and saved in settings.cfg by name ("jump =
// Space"). The codes are GLFW's, but nothing here calls GLFW.

enum class Action : uint8_t {
  Forward,
  Back,
  Left,
  Right,
  Sprint,
  Crouch,
  Aim,
  Jump,
  Dodge,
  Fire, // held, like the aim
  Count
};

const char *actionName(Action action);  // its key in the file: "jump"
const char *actionLabel(Action action); // for the tab: "Jump"

struct InputBinding {
  enum class Device : uint8_t { Key, Mouse };
  Device device = Device::Key;
  int code = 0; // a GLFW key, or a GLFW mouse button
  bool operator==(const InputBinding &) const = default;
};

// "W", "Space", "Left Shift", "Right mouse"; a key without a name of its
// own is "Key 300". bindingFromName reads them back, in any case.
std::string bindingName(InputBinding binding);
std::optional<InputBinding> bindingFromName(std::string_view name);

// Keys the editor keeps for itself (Escape, Tab, F1, F2, F5, F), so an
// action bound to one would fire both.
bool reservedForEditor(InputBinding binding);

class ControlBindings {
public:
  ControlBindings(); // the defaults: WASD, Left Shift, Left Ctrl, right mouse, Space, Left Alt, Q, left mouse, E
  const InputBinding &operator[](Action action) const {
    return keys[static_cast<size_t>(action)];
  }
  InputBinding &operator[](Action action) { return keys[static_cast<size_t>(action)]; }
  // Gives action this binding. An action that had it takes action's old
  // one, so no action is left without a key.
  void bind(Action action, InputBinding binding);

private:
  std::array<InputBinding, static_cast<size_t>(Action::Count)> keys;
};

// Which keys or buttons went down this frame. Asked any number of times in
// one frame it gives the same answer: the editor and the character both ask
// about the left mouse, and the first to ask mustn't take the press.
class PressEdges {
public:
  void nextFrame() { ++frame; }
  // Whether `code`, down or not right now, went down since the frame it was
  // last asked about.
  bool justPressed(int code, bool down);

private:
  struct Edge {
    bool down = false;    // as of the frame it was last asked about
    bool pressed = false; // that frame's answer
    uint64_t frame = ~uint64_t{0};
  };
  std::unordered_map<int, Edge> edges;
  uint64_t frame = 0;
};
