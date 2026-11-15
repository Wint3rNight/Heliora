#include "SceneViewport.h"

glm::vec2 SceneViewport::toScene(glm::vec2 windowPixel) const {
  return windowPixel - origin;
}

bool SceneViewport::contains(glm::vec2 windowPixel) const {
  const glm::vec2 local = toScene(windowPixel);
  return local.x >= 0.0f && local.y >= 0.0f && local.x < size.x &&
         local.y < size.y;
}

SceneResizeTracker::SceneResizeTracker(glm::ivec2 built)
    : builtSize(built), lastSize(built) {}

std::optional<glm::ivec2> SceneResizeTracker::update(glm::ivec2 areaSize,
                                                     bool mouseDown) {
  if (areaSize == lastSize) {
    ++sameFrames;
  } else {
    lastSize = areaSize;
    sameFrames = 1;
  }
  if (areaSize == builtSize || mouseDown || sameFrames < kSettleFrames ||
      glm::any(glm::lessThan(areaSize, glm::ivec2(kMinSize))))
    return std::nullopt;
  builtSize = areaSize;
  return areaSize;
}
