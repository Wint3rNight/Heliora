#pragma once

#include "Scene.h"

#include <cstddef>
#include <deque>
#include <optional>

// Undo and redo for the editor: a whole copy of the scene after every
// finished action (a drag, a typed value, a delete). Scenes are a few
// hundred entries at most, so whole copies are simpler and safer than
// recording each kind of change and how to reverse it.
class EditHistory {
public:
  explicit EditHistory(size_t maxSteps = 100);

  // The scene after a finished action. The first push after clear() is the
  // starting point, with nothing to undo. Anything that could have been
  // redone goes. A scene equal to the current one isn't a new step.
  void push(const SceneDescription &scene);
  // The scene one step back, or nothing at the start.
  std::optional<SceneDescription> undo();
  // The scene one step forward again, or nothing after the last step.
  std::optional<SceneDescription> redo();
  void clear();

private:
  size_t maxSteps;
  std::deque<SceneDescription> scenes; // oldest first
  size_t current = 0;                  // the one on screen
};
