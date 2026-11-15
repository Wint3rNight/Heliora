#include "SceneViewport.h"
#include "TestRunner.h"

#include <optional>

TEST(scene_viewport_maps_window_pixels_to_scene_pixels) {
  // A left panel 300 px wide and a 40 px bar on top.
  const SceneViewport area{glm::vec2(300.0f, 40.0f), glm::vec2(1000.0f, 700.0f)};
  CHECK(area.toScene(glm::vec2(300.0f, 40.0f)) == glm::vec2(0.0f, 0.0f));
  CHECK(area.toScene(glm::vec2(800.0f, 390.0f)) == glm::vec2(500.0f, 350.0f));
  CHECK(area.contains(glm::vec2(300.0f, 40.0f)));
  CHECK(area.contains(glm::vec2(1299.5f, 739.5f)));
  CHECK(!area.contains(glm::vec2(299.5f, 100.0f)));  // on the panel
  CHECK(!area.contains(glm::vec2(500.0f, 39.5f)));   // on the bar
  CHECK(!area.contains(glm::vec2(1300.0f, 100.0f))); // right edge
  CHECK(!area.contains(glm::vec2(500.0f, 740.0f)));  // bottom edge
}

namespace {

// How many frames of the same size it takes for update to ask for a rebuild,
// or 0 if it doesn't within 100.
int framesUntilRebuild(SceneResizeTracker &tracker, glm::ivec2 size,
                       bool mouseDown = false) {
  for (int frame = 1; frame <= 100; ++frame)
    if (tracker.update(size, mouseDown))
      return frame;
  return 0;
}

} // namespace

TEST(resize_waits_until_the_size_settles) {
  SceneResizeTracker tracker(glm::ivec2(1920, 1080));
  CHECK(!tracker.update(glm::ivec2(1920, 1080), false)); // as built
  // A divider moving: a new size every frame, so no rebuild.
  for (int width = 1900; width > 1400; width -= 20)
    CHECK(!tracker.update(glm::ivec2(width, 1080), false));
  // It stops at 1400.
  CHECK(framesUntilRebuild(tracker, glm::ivec2(1400, 1080)) ==
        SceneResizeTracker::kSettleFrames);
  CHECK(tracker.built() == glm::ivec2(1400, 1080));
  CHECK(framesUntilRebuild(tracker, glm::ivec2(1400, 1080)) == 0); // done
}

TEST(resize_waits_for_the_mouse_button) {
  SceneResizeTracker tracker(glm::ivec2(1920, 1080));
  // The divider held still at 1500 for a while: still dragging.
  for (int frame = 0; frame < 10; ++frame)
    CHECK(!tracker.update(glm::ivec2(1500, 1080), true));
  // Let go: the size has long settled, so it rebuilds at once.
  const std::optional<glm::ivec2> rebuild =
      tracker.update(glm::ivec2(1500, 1080), false);
  CHECK(rebuild == glm::ivec2(1500, 1080));
}

TEST(resize_back_to_the_built_size_needs_no_rebuild) {
  SceneResizeTracker tracker(glm::ivec2(1920, 1080));
  CHECK(!tracker.update(glm::ivec2(1700, 1080), false));
  CHECK(framesUntilRebuild(tracker, glm::ivec2(1920, 1080)) == 0);
  CHECK(tracker.built() == glm::ivec2(1920, 1080));
}

TEST(resize_never_goes_under_64_pixels) {
  SceneResizeTracker tracker(glm::ivec2(1920, 1080));
  CHECK(framesUntilRebuild(tracker, glm::ivec2(50, 700)) == 0);
  CHECK(framesUntilRebuild(tracker, glm::ivec2(800, 63)) == 0);
  CHECK(tracker.built() == glm::ivec2(1920, 1080));
  CHECK(framesUntilRebuild(tracker, glm::ivec2(64, 700)) ==
        SceneResizeTracker::kSettleFrames);
}
