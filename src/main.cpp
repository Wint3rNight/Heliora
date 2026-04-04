#define STB_IMAGE_IMPLEMENTATION
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include "VulkanRenderer.h"
#include <GLFW/glfw3.h>
#include <glm/ext/matrix_transform.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

GLFWwindow *window;
VulkanRenderer vulkanRenderer;

void initWindow(const std::string &wName = "Vulkan Renderer",
                const int width = 800, const int height = 600) {

  glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
  if (!glfwInit()) {
    throw std::runtime_error("Failed to init GLFW");
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  window = glfwCreateWindow(width, height, wName.c_str(), nullptr, nullptr);
}

int main() {
  initWindow("Vulkan Renderer", 1366, 768);

  if (vulkanRenderer.init(window) == EXIT_FAILURE) {
    return EXIT_FAILURE;
  }

  float angle = 0.0f;
  float deltaTime = 0.0f;
  float lastTime = 0.0f;
  int plant = vulkanRenderer.createMeshModel("Models/indoor plant_02.obj");

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    float now = glfwGetTime();
    deltaTime = now - lastTime;
    lastTime = now;

    angle += 10.0f * deltaTime;
    if (angle > 360.0f) {
      angle -= 360.0f;
    }

    glm::mat4 testMat = glm::rotate(glm::mat4(1.0f), glm::radians(angle),
                                    glm::vec3(0.0f, 1.0f, 0.0f));
    testMat =
        glm::rotate(testMat, glm::radians(0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    vulkanRenderer.updateModel(plant, testMat);

    vulkanRenderer.draw();
  }

  vulkanRenderer.cleanup();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
