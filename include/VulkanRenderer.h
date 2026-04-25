#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "DescriptorManager.h"
#include "PerformanceMetrics.h"
#include "Model.h"
#include "ModelManager.h"
#include "SceneNode.h"
#include "TextureManager.h"
#include "RenderPassManager.h"
#include "Utilities.h"
#include "VulkanDevice.h"
#include "VulkanPipeline.h"
#include "VulkanSwapchain.h"

class VulkanRenderer {
public:
  VulkanRenderer();

  int init(GLFWwindow *newWindow);
  void draw();
  int createMeshModel(const std::string &modelFile);
  SceneNode &getRootNode();
  void updateCameraView(const glm::mat4 &viewMatrix);
  void cleanup();

  // Called by main loop when framebuffer was resized
  void notifyResize();

  ~VulkanRenderer();

private:
  GLFWwindow *window = nullptr;

  int currentFrame = 0;

  // scene objects
  SceneNode rootNode;

  // scene settings
  struct UboViewProjection {
    glm::mat4 projection;
    glm::mat4 view;
  } uboViewProjection;

  // --- Subsystems ---
  VulkanDevice device;
  VulkanSwapchain swapchain;
  RenderPassManager renderPassManager;
  DescriptorManager descriptorManager;
  VulkanPipeline pipeline;

  // --- Resource Managers ---
  TextureManager textureManager;
  ModelManager modelManager;
  PerformanceMetrics metrics;

  // --- Synchronization ---
  std::vector<VkSemaphore> imageAvailable;
  std::vector<VkSemaphore> renderFinished;
  std::vector<VkFence> drawFences;
  std::vector<VkFence> imagesInFlight;

  bool framebufferResized = false;

  // --- Initialization helpers ---
  void createSynchronization();

  // --- Per-frame ---
  void recordCommands(uint32_t currentImage);
  void recreateSwapChain();

};
