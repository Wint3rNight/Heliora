#include "InputManager.h"

void InputManager::init(GLFWwindow *newWindow) {
  window = newWindow;

  // Get initial window size for mouse centering
  int width, height;
  glfwGetWindowSize(window, &width, &height);
  lastX = static_cast<float>(width) / 2.0f;
  lastY = static_cast<float>(height) / 2.0f;

  // Store 'this' in the window user pointer so static callbacks can access it
  glfwSetWindowUserPointer(window, this);

  // Register callbacks
  glfwSetCursorPosCallback(window, mouseCallback);
  glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
  // Set before ImGui starts: its backend calls whatever callback was here,
  // so the panels still scroll.
  glfwSetScrollCallback(window, scrollCallback);
  glfwSetDropCallback(window, dropCallback);

  // Capture the cursor for FPS-style camera.
  // Raw mouse motion avoids XGrabPointer on X11, which can suppress touchpad
  // events while keyboard keys are held simultaneously.
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  if (glfwRawMouseMotionSupported())
    glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);

  // Sticky keys: a press+release that both happen within one (long) frame —
  // e.g. tapping a key during a scene-switch device-idle wait — still reads
  // as GLFW_PRESS on the next poll instead of being silently dropped.
  glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);
  // The same for mouse buttons: a touchpad tap can press and release
  // between two polls, and would otherwise never read as pressed.
  glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, GLFW_TRUE);
}

void InputManager::pollEvents() {
  // Reset per-frame deltas before polling
  deltaX = 0.0f;
  deltaY = 0.0f;
  scrollY = 0.0f;
  keyEdges.nextFrame();
  buttonEdges.nextFrame();
  glfwPollEvents();
}

bool InputManager::isKeyPressed(int key) const {
  return glfwGetKey(window, key) == GLFW_PRESS;
}

bool InputManager::wasKeyJustPressed(int key) {
  return keyEdges.justPressed(key, isKeyPressed(key));
}

glm::vec2 InputManager::getMouseDelta() const { return {deltaX, deltaY}; }

// Asked from GLFW, since the tracked position is stale right after the cursor
// is freed.
glm::vec2 InputManager::getMousePosition() const {
  double x = 0.0, y = 0.0;
  glfwGetCursorPos(window, &x, &y);
  return {static_cast<float>(x), static_cast<float>(y)};
}

void InputManager::resetMouseDelta() {
  deltaX = 0.0f;
  deltaY = 0.0f;
}

bool InputManager::isMouseButtonPressed(int button) const {
  return glfwGetMouseButton(window, button) == GLFW_PRESS;
}

bool InputManager::wasMouseButtonJustPressed(int button) {
  return buttonEdges.justPressed(button, isMouseButtonPressed(button));
}

void InputManager::setCursorCaptured(bool captured) {
  glfwSetInputMode(window, GLFW_CURSOR,
                   captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
  firstMouse = true;
  deltaX = 0.0f;
  deltaY = 0.0f;
}

bool InputManager::shouldClose() const { return glfwWindowShouldClose(window); }

bool InputManager::wasResized() const { return framebufferResized; }

void InputManager::resetResizedFlag() { framebufferResized = false; }

// --- Static GLFW callbacks ---

void InputManager::mouseCallback(GLFWwindow *w, double xposIn, double yposIn) {
  auto *self = static_cast<InputManager *>(glfwGetWindowUserPointer(w));
  if (!self)
    return;

  float xpos = static_cast<float>(xposIn);
  float ypos = static_cast<float>(yposIn);

  if (self->firstMouse) {
    self->lastX = xpos;
    self->lastY = ypos;
    self->firstMouse = false;
  }

  self->deltaX += xpos - self->lastX;
  self->deltaY += self->lastY - ypos; // reversed: y goes bottom to top

  self->lastX = xpos;
  self->lastY = ypos;
}

void InputManager::framebufferResizeCallback(GLFWwindow *w, int /*width*/,
                                             int /*height*/) {
  auto *self = static_cast<InputManager *>(glfwGetWindowUserPointer(w));
  if (!self)
    return;
  self->framebufferResized = true;
}

void InputManager::scrollCallback(GLFWwindow *w, double, double yoffset) {
  auto *self = static_cast<InputManager *>(glfwGetWindowUserPointer(w));
  if (self)
    self->scrollY += static_cast<float>(yoffset);
}

void InputManager::dropCallback(GLFWwindow *w, int count, const char **paths) {
  auto *self = static_cast<InputManager *>(glfwGetWindowUserPointer(w));
  for (int i = 0; i < count; ++i)
    self->droppedFiles.emplace_back(paths[i]);
}
