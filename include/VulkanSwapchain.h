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

  // Composite framebuffers (3 attachments including TAA history)
  // are owned by CompositePass, not this class. init/recreate only own the
  // swapchain images + HDR colorBuffer + per-image command buffers.
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
  VkCommandBuffer getCommandBuffer(size_t i) const { return commandBuffers[i]; }
  VkImageView getColorBufferView(size_t i) const {
    return colorBufferImageView[i].get();
  }
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

  // HDR intermediate attachment (written by deferred pass, read as input
  // attachment in tone-mapping subpass).
  std::vector<AllocatedImage> colorBufferImage;
  std::vector<ImageViewHandle> colorBufferImageView;
  bool preferMailboxPresent = true;
  VkPresentModeKHR activePresentMode = VK_PRESENT_MODE_FIFO_KHR;

  void createSwapChain(const VulkanDevice &device, GLFWwindow *window);
  void createColorBufferImage(const VulkanDevice &device);
  void createCommandBuffers(VkDevice device, VkCommandPool commandPool);
  void cleanupSwapChain(VkDevice device, VmaAllocator allocator);

  VkSurfaceFormatKHR
  chooseBestSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &formats);
  VkPresentModeKHR
  chooseBestPresentationMode(const std::vector<VkPresentModeKHR> &modes);
  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &caps,
                              GLFWwindow *window);

  AllocatedImage createImage(VmaAllocator allocator, uint32_t width,
                             uint32_t height, VkFormat format,
                             VkImageTiling tiling, VkImageUsageFlags usage);
  VkImageView createImageView(VkDevice device, VkImage image, VkFormat format,
                              VkImageAspectFlags aspectFlags);
};
