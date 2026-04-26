#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include "VulkanRenderer.h"
#include "Camera.h"
#include "InputManager.h"
#include <GLFW/glfw3.h>
#include <glm/ext/matrix_transform.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

GLFWwindow *window;
VulkanRenderer vulkanRenderer;
InputManager inputManager;

// Camera settings — positioned inside Sponza hall, looking along the length
Camera camera(glm::vec3(0.0f, 2.0f, 0.0f));

// Timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

void processInput() {
  if (inputManager.isKeyPressed(GLFW_KEY_ESCAPE))
    glfwSetWindowShouldClose(window, true);

  if (inputManager.isKeyPressed(GLFW_KEY_W))
    camera.ProcessKeyboard(FORWARD, deltaTime);
  if (inputManager.isKeyPressed(GLFW_KEY_S))
    camera.ProcessKeyboard(BACKWARD, deltaTime);
  if (inputManager.isKeyPressed(GLFW_KEY_A))
    camera.ProcessKeyboard(LEFT, deltaTime);
  if (inputManager.isKeyPressed(GLFW_KEY_D))
    camera.ProcessKeyboard(RIGHT, deltaTime);
}

void initWindow(const std::string &wName = "Vulkan Renderer",
                const int width = 800, const int height = 600) {

  glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
  if (!glfwInit()) {
    throw std::runtime_error("Failed to init GLFW");
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); // Enable window resizing

  window = glfwCreateWindow(width, height, wName.c_str(), nullptr, nullptr);

  // InputManager handles cursor callback and resize callback
  inputManager.init(window);
}

int main() {
  initWindow("Vulkan Renderer", 1366, 768);

  if (vulkanRenderer.init(window) == EXIT_FAILURE) {
    return EXIT_FAILURE;
  }

  int sponzaId = vulkanRenderer.createMeshModel(
      "Resources/Models/Sponza/glTF/Sponza.gltf");
  auto sponzaNode = std::make_unique<SceneNode>();
  sponzaNode->setModelId(sponzaId);
  vulkanRenderer.getRootNode().addChild(std::move(sponzaNode));

  int helmetId = vulkanRenderer.createMeshModel(
      "Resources/Models/DamagedHelmet/glTF/DamagedHelmet.gltf");
  auto helmetNode = std::make_unique<SceneNode>();
  helmetNode->setModelId(helmetId);
  helmetNode->setLocalTransform(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 1.5f, 0.0f)));
  vulkanRenderer.getRootNode().addChild(std::move(helmetNode));

  // Initialize global transforms
  vulkanRenderer.getRootNode().update(glm::mat4(1.0f));

  while (!inputManager.shouldClose()) {
    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    inputManager.pollEvents();
    processInput();

    // Process mouse movement for camera
    glm::vec2 mouseDelta = inputManager.getMouseDelta();
    if (mouseDelta.x != 0.0f || mouseDelta.y != 0.0f) {
      camera.ProcessMouseMovement(mouseDelta.x, mouseDelta.y);
    }

    // Handle resize
    if (inputManager.wasResized()) {
      vulkanRenderer.notifyResize();
      inputManager.resetResizedFlag();
    }

    vulkanRenderer.getRootNode().update(glm::mat4(1.0f));
    
    vulkanRenderer.updateCameraView(camera.GetViewMatrix(), camera.Position);

    vulkanRenderer.draw();
  }

  vulkanRenderer.cleanup();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
