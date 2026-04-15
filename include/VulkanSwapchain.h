#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector>

#include "Utilities.h"
#include "VulkanDevice.h"

class VulkanSwapchain {
public:
  VulkanSwapchain() = default;
  ~VulkanSwapchain() = default;

  // Non-copyable
  VulkanSwapchain(const VulkanSwapchain &) = delete;
  VulkanSwapchain &operator=(const VulkanSwapchain &) = delete;

  void init(const VulkanDevice &device, VkRenderPass renderPass,
            GLFWwindow *window);
  void recreate(const VulkanDevice &device, VkRenderPass renderPass,
                GLFWwindow *window);
  void cleanup(VkDevice device);

  // --- Accessors ---
  VkSwapchainKHR getSwapchain() const { return swapchain; }
  VkFormat getImageFormat() const { return swapChainImageFormat; }
  VkExtent2D getExtent() const { return swapChainExtent; }
  const std::vector<SwapChainImage> &getImages() const {
    return swapChainImages;
  }
  size_t getImageCount() const { return swapChainImages.size(); }
  VkFramebuffer getFramebuffer(size_t index) const {
    return swapChainFramebuffers[index];
  }
  VkCommandBuffer getCommandBuffer(size_t index) const {
    return commandBuffers[index];
  }
  VkImageView getColorBufferView(size_t index) const {
    return colorBufferImageView[index];
  }
  VkImageView getDepthBufferView(size_t index) const {
    return depthBufferImageView[index];
  }

  // --- Format queries (no Vulkan objects created) ---
  VkFormat queryImageFormat(const VulkanDevice &device) const;
  VkFormat chooseSupportedFormat(VkPhysicalDevice physicalDevice,
                                 const std::vector<VkFormat> &formats,
                                 VkImageTiling tiling,
                                 VkFormatFeatureFlags featureFlags);

private:
  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  VkFormat swapChainImageFormat = VK_FORMAT_UNDEFINED;
  VkExtent2D swapChainExtent = {};

  std::vector<SwapChainImage> swapChainImages;
  std::vector<VkFramebuffer> swapChainFramebuffers;
  std::vector<VkCommandBuffer> commandBuffers;

  std::vector<VkImage> colorBufferImage;
  std::vector<VkDeviceMemory> colorBufferImageMemory;
  std::vector<VkImageView> colorBufferImageView;

  std::vector<VkImage> depthBufferImage;
  std::vector<VkDeviceMemory> depthBufferImageMemory;
  std::vector<VkImageView> depthBufferImageView;

  // --- Creation functions ---
  void createSwapChain(const VulkanDevice &device, GLFWwindow *window);
  void createColorBufferImage(const VulkanDevice &device);
  void createDepthBufferImage(const VulkanDevice &device);
  void createFramebuffers(VkDevice device, VkRenderPass renderPass);
  void createCommandBuffers(VkDevice device, VkCommandPool commandPool);
  void cleanupSwapChain(VkDevice device);

  // --- Support functions ---
  VkSurfaceFormatKHR
  chooseBestSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &formats);
  VkPresentModeKHR chooseBestPresentationMode(
      const std::vector<VkPresentModeKHR> &presentationModes);
  VkExtent2D
  chooseSwapExtent(const VkSurfaceCapabilitiesKHR &surfaceCapabilities,
                   GLFWwindow *window);

  // --- Image helpers ---
  VkImage createImage(VkPhysicalDevice physicalDevice, VkDevice device,
                      uint32_t width, uint32_t height, VkFormat format,
                      VkImageTiling tiling, VkImageUsageFlags useFlags,
                      VkMemoryPropertyFlags propFlags,
                      VkDeviceMemory *imageMemory);
  VkImageView createImageView(VkDevice device, VkImage image, VkFormat format,
                              VkImageAspectFlags aspectFlags);
};
