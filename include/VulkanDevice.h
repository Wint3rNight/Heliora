#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector>

#include "Utilities.h"

class VulkanDevice {
public:
  VulkanDevice() = default;
  ~VulkanDevice() = default;

  // Non-copyable, non-movable for now (raw Vulkan handles)
  VulkanDevice(const VulkanDevice &) = delete;
  VulkanDevice &operator=(const VulkanDevice &) = delete;

  void init(GLFWwindow *window);
  void cleanup();

  // --- Accessors ---
  VkInstance getInstance() const { return instance; }
  VkPhysicalDevice getPhysicalDevice() const {
    return mainDevice.physicalDevice;
  }
  VkDevice getLogicalDevice() const { return mainDevice.logicalDevice; }
  VkSurfaceKHR getSurface() const { return surface; }
  VkQueue getGraphicsQueue() const { return graphicsQueue; }
  VkQueue getPresentationQueue() const { return presentationQueue; }
  VkCommandPool getGraphicsCommandPool() const { return graphicsCommandPool; }
  VmaAllocator getAllocator() const { return allocator; }
  QueueFamilyIndices getQueueFamilies() const;

private:
  // --- Vulkan handles ---
  VkInstance instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;

  struct {
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice logicalDevice = VK_NULL_HANDLE;
  } mainDevice;

  VkQueue graphicsQueue = VK_NULL_HANDLE;
  VkQueue presentationQueue = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkCommandPool graphicsCommandPool = VK_NULL_HANDLE;
  VmaAllocator allocator = VK_NULL_HANDLE;

  GLFWwindow *window = nullptr;

  // --- Creation functions ---
  void createInstance();
  void createSurface();
  void createLogicalDevice();
  void loadDebugFunctions();
  void createCommandPool();
  void selectPhysicalDevice();
  void createVmaAllocator();

  // --- Support functions ---
  bool checkInstanceExtensionSupport(
      std::vector<const char *> *checkExtensions);
  bool checkDeviceExtensionSupport(VkPhysicalDevice device);
  bool checkDeviceSuitable(VkPhysicalDevice device);

  QueueFamilyIndices getQueueFamilies(VkPhysicalDevice device) const;
  SwapChainDetails getSwapChainDetails(VkPhysicalDevice device) const;
  int rateDeviceSuitability(VkPhysicalDevice device);
};
