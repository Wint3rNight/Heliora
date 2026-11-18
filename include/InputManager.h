#pragma once
#include <vector>
#include <utility>
#include <string>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "Controls.h"

class InputManager {
public:
  InputManager() = default;
  ~InputManager() = default;

  void init(GLFWwindow *window);
  void pollEvents();

  // --- Key state ---
  bool isKeyPressed(int key) const;
  // True in the frame a key goes down, however many times that frame asks.
  bool wasKeyJustPressed(int key);

  // --- Mouse state ---
  glm::vec2 getMouseDelta() const;
  glm::vec2 getMousePosition() const;
  void resetMouseDelta();
  bool isMouseButtonPressed(int button) const;
  // True in the frame a button goes down, like wasKeyJustPressed.
  bool wasMouseButtonJustPressed(int button);
  // Files dropped on the window from the file manager since the last call.
  std::vector<std::string> takeDroppedFiles() { return std::exchange(droppedFiles, {}); }
  // Wheel notches this frame, positive away from you.
  float getScrollDelta() const { return scrollY; }
  // Hides the cursor and sends raw movement (flying), or frees it. The first
  // movement after a switch is dropped, since the cursor jumps.
  void setCursorCaptured(bool captured);

  // --- Window state ---
  bool shouldClose() const;
  bool wasResized() const;
  void resetResizedFlag();

private:
  GLFWwindow *window = nullptr;

  // Mouse tracking
  float lastX = 0.0f;
  float lastY = 0.0f;
  float deltaX = 0.0f;
  float deltaY = 0.0f;
  float scrollY = 0.0f;
  bool firstMouse = true;

  // Resize flag
  bool framebufferResized = false;

  // What went down this frame, for wasKeyJustPressed and
  // wasMouseButtonJustPressed.
  PressEdges keyEdges;
  PressEdges buttonEdges;

  // Static callbacks (GLFW requires free/static functions)
  static void mouseCallback(GLFWwindow *w, double xpos, double ypos);
  static void scrollCallback(GLFWwindow *w, double xoffset, double yoffset);
  static void framebufferResizeCallback(GLFWwindow *w, int width, int height);
  static void dropCallback(GLFWwindow *w, int count, const char **paths);
  std::vector<std::string> droppedFiles;
};
