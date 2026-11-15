#pragma once

#define GLFW_INCLUDE_VULKAN
#include "Utilities.h"
#include "VulkanDevice.h"
#include <GLFW/glfw3.h>
#include <vector>

class VulkanSwapchain {
public:
  VulkanSwapchain() = default;
  ~VulkanSwapchain() = default;

  VulkanSwapchain(const VulkanSwapchain &) = delete;
  VulkanSwapchain &operator=(const VulkanSwapchain &) = delete;

  // Only what the window needs: the swapchain images and per-image command
  // buffers. The scene's own images (its HDR colour buffer among them) are
  // the renderer's, sized to the scene's area, not the window.
  void init(const VulkanDevice &device, GLFWwindow *window);
  void recreate(const VulkanDevice &device, GLFWwindow *window);
  void cleanup(VkDevice device, VmaAllocator allocator);

  VkSwapchainKHR getSwapchain() const { return swapchain; }
  VkFormat getImageFormat() const { return swapChainImageFormat; }
  VkExtent2D getExtent() const { return swapChainExtent; }
  const std::vector<SwapChainImage> &getImages() const {
    return swapChainImages;
  }
  size_t getImageCount() const { return swapChainImages.size(); }
  VkImageView getSwapImageView(size_t i) const {
    return swapChainImages[i].imageView.get();
  }
  VkImage getSwapImage(size_t i) const { return swapChainImages[i].image; }
  VkCommandBuffer getCommandBuffer(size_t i) const { return commandBuffers[i]; }
  void setPreferMailbox(bool prefer) { preferMailboxPresent = prefer; }
  bool getPreferMailbox() const { return preferMailboxPresent; }
  VkPresentModeKHR getActivePresentMode() const { return activePresentMode; }

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
  std::vector<VkCommandBuffer> commandBuffers;

  bool preferMailboxPresent = true;
  VkPresentModeKHR activePresentMode = VK_PRESENT_MODE_FIFO_KHR;

  void createSwapChain(const VulkanDevice &device, GLFWwindow *window);
  void createCommandBuffers(VkDevice device, VkCommandPool commandPool);
  void cleanupSwapChain(VkDevice device, VmaAllocator allocator);

  VkSurfaceFormatKHR
  chooseBestSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &formats);
  VkPresentModeKHR
  chooseBestPresentationMode(const std::vector<VkPresentModeKHR> &modes);
  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &caps,
                              GLFWwindow *window);

  VkImageView createImageView(VkDevice device, VkImage image, VkFormat format,
                              VkImageAspectFlags aspectFlags);
};
