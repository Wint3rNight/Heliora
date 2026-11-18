#include "Controls.h"

// Only for the key and button codes; nothing here calls GLFW.
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <format>
#include <utility>
#include <vector>

namespace {

struct ActionInfo {
  const char *name;
  const char *label;
};
constexpr std::array<ActionInfo, static_cast<size_t>(Action::Count)> kActions = {{
    {"forward", "Forward"},
    {"back", "Back"},
    {"left", "Left"},
    {"right", "Right"},
    {"sprint", "Sprint"},
    {"crouch", "Crouch"},
    {"aim", "Aim"},
    {"jump", "Jump"},
    {"dodge", "Dodge"},
    {"reveal", "Reveal"},
    {"cast", "Cast"},
    {"burst", "Burst"},
}};

// Keys with names of their own; letters, digits, F keys and keypad digits
// are made up below.
const std::vector<std::pair<int, std::string>> &keyNames() {
  static const std::vector<std::pair<int, std::string>> names = [] {
    std::vector<std::pair<int, std::string>> n = {
        {GLFW_KEY_SPACE, "Space"},          {GLFW_KEY_APOSTROPHE, "'"},
        {GLFW_KEY_COMMA, ","},              {GLFW_KEY_MINUS, "-"},
        {GLFW_KEY_PERIOD, "."},             {GLFW_KEY_SLASH, "/"},
        {GLFW_KEY_SEMICOLON, ";"},          {GLFW_KEY_EQUAL, "="},
        {GLFW_KEY_LEFT_BRACKET, "["},       {GLFW_KEY_BACKSLASH, "\\"},
        {GLFW_KEY_RIGHT_BRACKET, "]"},      {GLFW_KEY_GRAVE_ACCENT, "`"},
        {GLFW_KEY_ESCAPE, "Escape"},        {GLFW_KEY_ENTER, "Enter"},
        {GLFW_KEY_TAB, "Tab"},              {GLFW_KEY_BACKSPACE, "Backspace"},
        {GLFW_KEY_INSERT, "Insert"},        {GLFW_KEY_DELETE, "Delete"},
        {GLFW_KEY_RIGHT, "Right"},          {GLFW_KEY_LEFT, "Left"},
        {GLFW_KEY_DOWN, "Down"},            {GLFW_KEY_UP, "Up"},
        {GLFW_KEY_PAGE_UP, "Page Up"},      {GLFW_KEY_PAGE_DOWN, "Page Down"},
        {GLFW_KEY_HOME, "Home"},            {GLFW_KEY_END, "End"},
        {GLFW_KEY_CAPS_LOCK, "Caps Lock"},  {GLFW_KEY_SCROLL_LOCK, "Scroll Lock"},
        {GLFW_KEY_NUM_LOCK, "Num Lock"},    {GLFW_KEY_PRINT_SCREEN, "Print Screen"},
        {GLFW_KEY_PAUSE, "Pause"},          {GLFW_KEY_KP_DECIMAL, "Keypad ."},
        {GLFW_KEY_KP_DIVIDE, "Keypad /"},   {GLFW_KEY_KP_MULTIPLY, "Keypad *"},
        {GLFW_KEY_KP_SUBTRACT, "Keypad -"}, {GLFW_KEY_KP_ADD, "Keypad +"},
        {GLFW_KEY_KP_ENTER, "Keypad Enter"}, {GLFW_KEY_KP_EQUAL, "Keypad ="},
        {GLFW_KEY_LEFT_SHIFT, "Left Shift"}, {GLFW_KEY_LEFT_CONTROL, "Left Ctrl"},
        {GLFW_KEY_LEFT_ALT, "Left Alt"},    {GLFW_KEY_LEFT_SUPER, "Left Super"},
        {GLFW_KEY_RIGHT_SHIFT, "Right Shift"}, {GLFW_KEY_RIGHT_CONTROL, "Right Ctrl"},
        {GLFW_KEY_RIGHT_ALT, "Right Alt"},  {GLFW_KEY_RIGHT_SUPER, "Right Super"},
        {GLFW_KEY_MENU, "Menu"},
    };
    for (int c = GLFW_KEY_A; c <= GLFW_KEY_Z; ++c)
      n.emplace_back(c, std::string(1, char(c)));
    for (int c = GLFW_KEY_0; c <= GLFW_KEY_9; ++c)
      n.emplace_back(c, std::string(1, char(c)));
    for (int i = 0; i < 25; ++i)
      n.emplace_back(GLFW_KEY_F1 + i, std::format("F{}", i + 1));
    for (int i = 0; i < 10; ++i)
      n.emplace_back(GLFW_KEY_KP_0 + i, std::format("Keypad {}", i));
    return n;
  }();
  return names;
}

const std::array<std::string, 3> kMouseNames = {"Left mouse", "Right mouse",
                                               "Middle mouse"};

bool sameText(std::string_view a, std::string_view b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
           return std::tolower(static_cast<unsigned char>(x)) ==
                  std::tolower(static_cast<unsigned char>(y));
         });
}

// The number after a prefix ("Key 300", "Mouse 4"), any case.
std::optional<int> numberAfter(std::string_view name, std::string_view prefix) {
  if (name.size() <= prefix.size() || !sameText(name.substr(0, prefix.size()), prefix))
    return std::nullopt;
  const std::string_view digits = name.substr(prefix.size());
  int value = 0;
  const auto [end, error] =
      std::from_chars(digits.data(), digits.data() + digits.size(), value);
  if (error != std::errc() || end != digits.data() + digits.size())
    return std::nullopt;
  return value;
}

InputBinding key(int code) { return {InputBinding::Device::Key, code}; }

} // namespace

const char *actionName(Action action) {
  const size_t i = static_cast<size_t>(action);
  return i < kActions.size() ? kActions[i].name : "";
}

const char *actionLabel(Action action) {
  const size_t i = static_cast<size_t>(action);
  return i < kActions.size() ? kActions[i].label : "";
}

std::string bindingName(InputBinding binding) {
  if (binding.device == InputBinding::Device::Mouse) {
    if (binding.code >= 0 && binding.code < static_cast<int>(kMouseNames.size()))
      return kMouseNames[binding.code];
    return std::format("Mouse {}", binding.code + 1);
  }
  for (const auto &[code, name] : keyNames())
    if (code == binding.code)
      return name;
  return std::format("Key {}", binding.code);
}

std::optional<InputBinding> bindingFromName(std::string_view name) {
  for (const auto &[code, known] : keyNames())
    if (sameText(name, known))
      return key(code);
  for (int i = 0; i < static_cast<int>(kMouseNames.size()); ++i)
    if (sameText(name, kMouseNames[i]))
      return InputBinding{InputBinding::Device::Mouse, i};
  if (const auto n = numberAfter(name, "Mouse "); n && *n >= 1 && *n <= GLFW_MOUSE_BUTTON_LAST + 1)
    return InputBinding{InputBinding::Device::Mouse, *n - 1};
  if (const auto n = numberAfter(name, "Key "); n && *n >= 0 && *n <= GLFW_KEY_LAST)
    return key(*n);
  return std::nullopt;
}

bool reservedForEditor(InputBinding binding) {
  if (binding.device != InputBinding::Device::Key)
    return false;
  for (const int code : {GLFW_KEY_ESCAPE, GLFW_KEY_TAB, GLFW_KEY_F1, GLFW_KEY_F2,
                         GLFW_KEY_F5, GLFW_KEY_F})
    if (binding.code == code)
      return true;
  return false;
}

ControlBindings::ControlBindings() {
  (*this)[Action::Forward] = key(GLFW_KEY_W);
  (*this)[Action::Back] = key(GLFW_KEY_S);
  (*this)[Action::Left] = key(GLFW_KEY_A);
  (*this)[Action::Right] = key(GLFW_KEY_D);
  (*this)[Action::Sprint] = key(GLFW_KEY_LEFT_SHIFT);
  (*this)[Action::Crouch] = key(GLFW_KEY_LEFT_CONTROL);
  (*this)[Action::Aim] = {InputBinding::Device::Mouse, GLFW_MOUSE_BUTTON_RIGHT};
  (*this)[Action::Jump] = key(GLFW_KEY_SPACE);
  (*this)[Action::Dodge] = key(GLFW_KEY_LEFT_ALT);
  (*this)[Action::Reveal] = key(GLFW_KEY_Q);
  (*this)[Action::Cast] = {InputBinding::Device::Mouse, GLFW_MOUSE_BUTTON_LEFT};
  (*this)[Action::Burst] = key(GLFW_KEY_E);
}

void ControlBindings::bind(Action action, InputBinding binding) {
  InputBinding &mine = (*this)[action];
  for (InputBinding &other : keys)
    if (&other != &mine && other == binding)
      other = mine;
  mine = binding;
}

bool PressEdges::justPressed(int code, bool down) {
  Edge &e = edges[code];
  if (e.frame != frame) {
    e.pressed = down && !e.down;
    e.down = down;
    e.frame = frame;
  }
  return e.pressed;
}
