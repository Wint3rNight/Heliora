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

// Camera settings
Camera camera(glm::vec3(10.0f, 0.0f, 20.0f));

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

  float angle = 0.0f;

  int plantModelId = vulkanRenderer.createMeshModel("Models/indoor plant_02.obj");

  auto plantNode = std::make_unique<SceneNode>();
  plantNode->setModelId(plantModelId);
  SceneNode* plantNodePtr = plantNode.get(); // keep raw pointer for animation
  vulkanRenderer.getRootNode().addChild(std::move(plantNode));

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

    angle += 10.0f * deltaTime;
    if (angle > 360.0f) {
      angle -= 360.0f;
    }

    glm::mat4 testMat = glm::rotate(glm::mat4(1.0f), glm::radians(angle),
                                    glm::vec3(0.0f, 1.0f, 0.0f));
    testMat =
        glm::rotate(testMat, glm::radians(0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        
    plantNodePtr->setLocalTransform(testMat);
    vulkanRenderer.getRootNode().update(glm::mat4(1.0f));
    
    vulkanRenderer.updateCameraView(camera.GetViewMatrix());

    vulkanRenderer.draw();
  }

  vulkanRenderer.cleanup();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
