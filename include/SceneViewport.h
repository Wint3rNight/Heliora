#pragma once

#include <glm/glm.hpp>

#include <optional>

// The part of the window the scene is drawn in: the space the editor's
// panels leave, or the whole window with them hidden. Mouse positions come
// in window pixels; picking and the gizmo want them on the scene.
struct SceneViewport {
  glm::vec2 origin{0.0f}; // top-left corner, window pixels
  glm::vec2 size{0.0f};   // width and height, window pixels

  // A window pixel as a pixel of the scene area, y down from its top. The
  // picture shown there can be a different size for a few frames while the
  // area is resized (it's stretched), so pair this with size, not with the
  // picture's own size.
  glm::vec2 toScene(glm::vec2 windowPixel) const;
  // Whether a window pixel falls on the scene. The right and bottom edges
  // are outside, as with pixels.
  bool contains(glm::vec2 windowPixel) const;
};

// When to rebuild the scene's images for a new area size. A rebuild waits
// for the GPU to go idle and makes a dozen images, too slow for every frame
// of a divider drag, so it waits until the size has stayed the same for
// kSettleFrames frames with no mouse button held. It never rebuilds to less
// than kMinSize pixels on a side; the scene just isn't drawn until it grows.
class SceneResizeTracker {
public:
  static constexpr int kSettleFrames = 3;
  static constexpr int kMinSize = 64;

  // built: the size the scene's images have now.
  explicit SceneResizeTracker(glm::ivec2 built);

  // Called once a frame with the area's size. Returns the size to rebuild
  // to on the one frame a rebuild is due, and takes it as built.
  std::optional<glm::ivec2> update(glm::ivec2 areaSize, bool mouseDown);
  glm::ivec2 built() const { return builtSize; }

private:
  glm::ivec2 builtSize;
  glm::ivec2 lastSize;
  int sameFrames = 0; // frames in a row lastSize was reported
};
