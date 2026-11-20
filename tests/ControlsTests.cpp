#include "Controls.h"
#include "SettingsFile.h"
#include "TestRunner.h"

#include <set>
#include <sstream>
#include <string>

namespace {

InputBinding named(const char *name) {
  const std::optional<InputBinding> b = bindingFromName(name);
  CHECK(b.has_value());
  return b.value_or(InputBinding{});
}

SettingsFile parseText(const std::string &text) {
  std::istringstream in(text);
  return SettingsFile::parse(in, "test.cfg");
}

} // namespace

TEST(the_default_keys_are_the_old_ones) {
  const ControlBindings c;
  CHECK(bindingName(c[Action::Forward]) == "W");
  CHECK(bindingName(c[Action::Back]) == "S");
  CHECK(bindingName(c[Action::Left]) == "A");
  CHECK(bindingName(c[Action::Right]) == "D");
  CHECK(bindingName(c[Action::Run]) == "Left Shift");
  CHECK(bindingName(c[Action::Crouch]) == "Left Ctrl");
  CHECK(bindingName(c[Action::Aim]) == "Right mouse");
  CHECK(bindingName(c[Action::Jump]) == "Space");
  CHECK(bindingName(c[Action::Dodge]) == "Left Alt");
}

TEST(fire_is_on_the_left_mouse) {
  const ControlBindings c;
  CHECK(bindingName(c[Action::Fire]) == "Left mouse");
  CHECK(std::string(actionName(Action::Fire)) == "fire");
  CHECK(std::string(actionLabel(Action::Fire)).starts_with("Fire"));
}

TEST(every_action_has_its_own_name) {
  std::set<std::string> names, labels;
  for (int i = 0; i < static_cast<int>(Action::Count); ++i) {
    const Action a = static_cast<Action>(i);
    CHECK(std::string(actionName(a)).size() > 0);
    CHECK(std::string(actionLabel(a)).size() > 0);
    names.insert(actionName(a));
    labels.insert(actionLabel(a));
  }
  CHECK(names.size() == static_cast<size_t>(Action::Count));
  CHECK(labels.size() == static_cast<size_t>(Action::Count));
}

TEST(key_names_read_back) {
  for (const char *name : {"W", "Q", "7", "F12", "Space", "Left Shift", "Right Ctrl",
                           "Left Alt", "Enter", "Up", "Keypad 0", "Page Down", "/",
                           "Left mouse", "Right mouse", "Middle mouse", "Mouse 5"})
    CHECK(bindingName(named(name)) == name);
  CHECK(named("space") == named("Space")); // any case
  CHECK(!bindingFromName("Banana"));
  CHECK(!bindingFromName(""));
  // A key without a name of its own still reads back.
  InputBinding odd;
  odd.code = 300;
  CHECK(bindingFromName(bindingName(odd)) == odd);
}

TEST(binding_a_taken_key_swaps_the_two) {
  // Dodge's Left Alt given to jump: dodge takes jump's Space, so nothing is
  // left without a key.
  ControlBindings c;
  c.bind(Action::Jump, named("Left Alt"));
  CHECK(bindingName(c[Action::Jump]) == "Left Alt");
  CHECK(bindingName(c[Action::Dodge]) == "Space");
  c.bind(Action::Jump, named("E"));
  CHECK(bindingName(c[Action::Jump]) == "E");
  CHECK(bindingName(c[Action::Dodge]) == "Space");
}

TEST(the_editors_own_keys_are_kept_for_it) {
  for (const char *name : {"Escape", "Tab", "F1", "F2", "F5", "F9", "F"})
    CHECK(reservedForEditor(named(name)));
  CHECK(!reservedForEditor(named("E")));
  CHECK(!reservedForEditor(named("Right mouse")));
}

TEST(controls_are_saved_by_name) {
  InputBinding jump = named("Space");
  const SettingBinding bindings[] = {{"controls", "jump", &jump}};
  SettingsFile file;
  jump = named("Left Shift");
  writeSettings(file, bindings);
  CHECK(file.text("controls", "jump") == std::optional<std::string>("Left Shift"));
  const SettingsFile edited = parseText("[controls]\njump = mouse 4\n");
  readSettings(edited, bindings);
  CHECK(bindingName(jump) == "Mouse 4");
  CHECK(!hasUnsavedSettings(edited, bindings));
  readSettings(parseText("[controls]\njump = Banana\n"), bindings); // ignored
  CHECK(bindingName(jump) == "Mouse 4");
  CHECK(hasUnsavedSettings(file, bindings));
}

TEST(a_press_is_new_however_often_its_frame_asks) {
  PressEdges edges;
  edges.nextFrame();
  CHECK(edges.justPressed(1, true));
  CHECK(edges.justPressed(1, true)); // the editor asked first: still new for the character
  edges.nextFrame();
  CHECK(!edges.justPressed(1, true)); // held, no longer new
  edges.nextFrame();
  CHECK(!edges.justPressed(1, false));
  edges.nextFrame();
  CHECK(edges.justPressed(1, true));  // let go and pressed again
  CHECK(edges.justPressed(1, false)); // a tap GLFW hands back once: the frame's answer stands
  CHECK(edges.justPressed(2, true));  // each code on its own
}

TEST(the_powers_are_q_and_e) {
  const ControlBindings c;
  CHECK(bindingName(c[Action::Reveal]) == "Q");
  CHECK(bindingName(c[Action::Burst]) == "E");
  CHECK(std::string(actionName(Action::Burst)) == "burst");
}
