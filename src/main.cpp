#define STB_IMAGE_IMPLEMENTATION
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include "VulkanRenderer.h"
#include <GLFW/glfw3.h>
#include <glm/ext/matrix_transform.hpp>
#include "Camera.h"
#include <iostream>
#include <stdexcept>
#include <vector>

GLFWwindow *window;
VulkanRenderer vulkanRenderer;

// Camera settings
Camera camera(glm::vec3(10.0f, 0.0f, 20.0f));
float lastX = 1366.0f / 2.0;
float lastY = 768.0f / 2.0;
bool firstMouse = true;

// Timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

void initWindow(const std::string &wName = "Vulkan Renderer",
                const int width = 800, const int height = 600) {

  glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
  if (!glfwInit()) {
    throw std::runtime_error("Failed to init GLFW");
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  window = glfwCreateWindow(width, height, wName.c_str(), nullptr, nullptr);
  
  glfwSetCursorPosCallback(window, mouse_callback);
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

int main() {
  initWindow("Vulkan Renderer", 1366, 768);

  if (vulkanRenderer.init(window) == EXIT_FAILURE) {
    return EXIT_FAILURE;
  }

  float angle = 0.0f;
  
  int plant = vulkanRenderer.createMeshModel("Models/indoor plant_02.obj");

  while (!glfwWindowShouldClose(window)) {
    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    processInput(window);
    glfwPollEvents();
    
    angle += 10.0f * deltaTime;
    if (angle > 360.0f) {
      angle -= 360.0f;
    }

    glm::mat4 testMat = glm::rotate(glm::mat4(1.0f), glm::radians(angle),
                                    glm::vec3(0.0f, 1.0f, 0.0f));
    testMat =
        glm::rotate(testMat, glm::radians(0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    vulkanRenderer.updateModel(plant, testMat);
    vulkanRenderer.updateCameraView(camera.GetViewMatrix());

    vulkanRenderer.draw();
  }

  vulkanRenderer.cleanup();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
