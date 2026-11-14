#include "EditHistory.h"
#include "TestRunner.h"

namespace {

// A scene whose one box sits at x, so scenes are easy to tell apart.
SceneDescription sceneWithBoxAt(float x) {
  SceneDescription scene;
  SceneModelEntry box;
  box.shape = PrimitiveShape::Box;
  box.transform.position = glm::vec3(x, 0.0f, 0.0f);
  scene.models.push_back(box);
  return scene;
}

float boxX(const std::optional<SceneDescription> &scene) {
  return scene ? scene->models.front().transform.position.x : -1.0f;
}

} // namespace

TEST(undo_then_redo_returns_the_same_layout) {
  EditHistory history;
  history.push(sceneWithBoxAt(0.0f)); // as loaded
  history.push(sceneWithBoxAt(1.0f)); // moved
  history.push(sceneWithBoxAt(2.0f)); // moved again
  const std::optional<SceneDescription> back = history.undo();
  CHECK(back.has_value());
  CHECK(boxX(back) == 1.0f);
  CHECK(boxX(history.undo()) == 0.0f);
  CHECK(boxX(history.redo()) == 1.0f);
  const std::optional<SceneDescription> forward = history.redo();
  CHECK(forward == sceneWithBoxAt(2.0f));
}

TEST(nothing_to_undo_or_redo_at_the_ends) {
  EditHistory history;
  CHECK(!history.undo());
  history.push(sceneWithBoxAt(0.0f));
  CHECK(!history.undo()); // the scene as loaded
  CHECK(!history.redo());
  history.push(sceneWithBoxAt(1.0f));
  CHECK(boxX(history.undo()) == 0.0f);
  CHECK(!history.undo());
}

TEST(new_action_clears_redo) {
  EditHistory history;
  history.push(sceneWithBoxAt(0.0f));
  history.push(sceneWithBoxAt(1.0f));
  history.push(sceneWithBoxAt(2.0f));
  CHECK(boxX(history.undo()) == 1.0f);
  history.push(sceneWithBoxAt(5.0f)); // something else instead
  CHECK(!history.redo());
  CHECK(boxX(history.undo()) == 1.0f);
}

TEST(history_keeps_the_last_100) {
  EditHistory history;
  for (int i = 0; i <= 150; ++i) // the loaded scene, then 150 actions
    history.push(sceneWithBoxAt(float(i)));
  int undone = 0;
  float last = -1.0f;
  // Capped, so an undo that never runs out fails instead of hanging.
  for (int i = 0; i < 200; ++i) {
    const std::optional<SceneDescription> scene = history.undo();
    if (!scene)
      break;
    last = boxX(scene);
    ++undone;
  }
  CHECK(undone == 100);
  CHECK(last == 50.0f);
}

TEST(same_layout_is_not_a_new_step) {
  EditHistory history;
  history.push(sceneWithBoxAt(0.0f));
  history.push(sceneWithBoxAt(1.0f));
  history.push(sceneWithBoxAt(1.0f)); // a click that moved nothing
  CHECK(boxX(history.undo()) == 0.0f);
}

TEST(clear_starts_over) {
  EditHistory history;
  history.push(sceneWithBoxAt(0.0f));
  history.push(sceneWithBoxAt(1.0f));
  history.clear();
  CHECK(!history.undo());
  history.push(sceneWithBoxAt(7.0f)); // another scene loaded
  CHECK(!history.undo());
}
