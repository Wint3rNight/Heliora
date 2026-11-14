#include "EditHistory.h"

EditHistory::EditHistory(size_t maxSteps) : maxSteps(maxSteps) {}

void EditHistory::push(const SceneDescription &scene) {
  if (!scenes.empty() && scenes[current] == scene)
    return;
  // Whatever was undone can't be redone once something new happens.
  if (!scenes.empty())
    scenes.erase(scenes.begin() + current + 1, scenes.end());
  scenes.push_back(scene);
  // maxSteps undos need maxSteps + 1 scenes: the oldest is where they end.
  if (scenes.size() > maxSteps + 1)
    scenes.pop_front();
  current = scenes.size() - 1;
}

std::optional<SceneDescription> EditHistory::undo() {
  if (scenes.empty() || current == 0)
    return std::nullopt;
  return scenes[--current];
}

std::optional<SceneDescription> EditHistory::redo() {
  if (current + 1 >= scenes.size())
    return std::nullopt;
  return scenes[++current];
}

void EditHistory::clear() {
  scenes.clear();
  current = 0;
}
