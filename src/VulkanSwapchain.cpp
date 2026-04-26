#include "VulkanSwapchain.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Public
// ---------------------------------------------------------------------------

VkFormat VulkanSwapchain::queryImageFormat(const VulkanDevice &device) const {
  uint32_t count = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device.getPhysicalDevice(), device.getSurface(), &count, nullptr);
  std::vector<VkSurfaceFormatKHR> formats(count);
  if (count) vkGetPhysicalDeviceSurfaceFormatsKHR(device.getPhysicalDevice(), device.getSurface(), &count, formats.data());

  if (count == 1 && formats[0].format == VK_FORMAT_UNDEFINED) return VK_FORMAT_R8G8B8A8_UNORM;
  for (auto &f : formats)
    if ((f.format == VK_FORMAT_R8G8B8A8_UNORM || f.format == VK_FORMAT_B8G8R8A8_UNORM) &&
        f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) return f.format;
  return formats[0].format;
}

void VulkanSwapchain::init(const VulkanDevice &device, VkRenderPass compositionRenderPass,
                           GLFWwindow *window) {
  createSwapChain(device, window);
  createColorBufferImage(device);
  createFramebuffers(device.getLogicalDevice(), compositionRenderPass);
  createCommandBuffers(device.getLogicalDevice(), device.getGraphicsCommandPool());
}

void VulkanSwapchain::recreate(const VulkanDevice &device, VkRenderPass compositionRenderPass,
                               GLFWwindow *window) {
  int w = 0, h = 0;
  glfwGetFramebufferSize(window, &w, &h);
  while (w == 0 || h == 0) { glfwGetFramebufferSize(window, &w, &h); glfwWaitEvents(); }
  vkDeviceWaitIdle(device.getLogicalDevice());
  vkResetCommandPool(device.getLogicalDevice(), device.getGraphicsCommandPool(), 0);

  cleanupSwapChain(device.getLogicalDevice(), device.getAllocator());
  createSwapChain(device, window);
  createColorBufferImage(device);
  createFramebuffers(device.getLogicalDevice(), compositionRenderPass);
  createCommandBuffers(device.getLogicalDevice(), device.getGraphicsCommandPool());
}

void VulkanSwapchain::cleanup(VkDevice device, VmaAllocator allocator) {
  cleanupSwapChain(device, allocator);
}

// ---------------------------------------------------------------------------
// Private
// ---------------------------------------------------------------------------

void VulkanSwapchain::createSwapChain(const VulkanDevice &device, GLFWwindow *window) {
  VkSurfaceCapabilitiesKHR caps;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device.getPhysicalDevice(), device.getSurface(), &caps);

  uint32_t fmtCount = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device.getPhysicalDevice(), device.getSurface(), &fmtCount, nullptr);
  std::vector<VkSurfaceFormatKHR> fmts(fmtCount);
  if (fmtCount) vkGetPhysicalDeviceSurfaceFormatsKHR(device.getPhysicalDevice(), device.getSurface(), &fmtCount, fmts.data());

  uint32_t pmCount = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device.getPhysicalDevice(), device.getSurface(), &pmCount, nullptr);
  std::vector<VkPresentModeKHR> pms(pmCount);
  if (pmCount) vkGetPhysicalDeviceSurfacePresentModesKHR(device.getPhysicalDevice(), device.getSurface(), &pmCount, pms.data());

  VkSurfaceFormatKHR surfFmt = chooseBestSurfaceFormat(fmts);
  VkPresentModeKHR   presMode = chooseBestPresentationMode(pms);
  VkExtent2D         ext      = chooseSwapExtent(caps, window);

  uint32_t imageCount = caps.minImageCount + 1;
  if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
    imageCount = caps.maxImageCount;

  VkSwapchainCreateInfoKHR ci = {};
  ci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  ci.surface          = device.getSurface();
  ci.imageFormat      = surfFmt.format;
  ci.imageColorSpace  = surfFmt.colorSpace;
  ci.presentMode      = presMode;
  ci.imageExtent      = ext;
  ci.minImageCount    = imageCount;
  ci.imageArrayLayers = 1;
  ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  ci.preTransform     = caps.currentTransform;
  ci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  ci.clipped          = VK_TRUE;

  QueueFamilyIndices idx = device.getQueueFamilies();
  if (idx.graphicsFamily != idx.presentationFamily) {
    uint32_t qfi[] = { (uint32_t)idx.graphicsFamily, (uint32_t)idx.presentationFamily };
    ci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
    ci.queueFamilyIndexCount = 2;
    ci.pQueueFamilyIndices   = qfi;
  } else {
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }

  if (vkCreateSwapchainKHR(device.getLogicalDevice(), &ci, nullptr, &swapchain) != VK_SUCCESS)
    throw std::runtime_error("Failed to create swap chain");

  swapChainImageFormat = surfFmt.format;
  swapChainExtent      = ext;

  uint32_t imgCount = 0;
  vkGetSwapchainImagesKHR(device.getLogicalDevice(), swapchain, &imgCount, nullptr);
  std::vector<VkImage> images(imgCount);
  vkGetSwapchainImagesKHR(device.getLogicalDevice(), swapchain, &imgCount, images.data());

  for (VkImage img : images) {
    SwapChainImage sci = {};
    sci.image     = img;
    sci.imageView = ImageViewHandle(device.getLogicalDevice(),
                    createImageView(device.getLogicalDevice(), img, swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT));
    swapChainImages.push_back(std::move(sci));
  }
}

void VulkanSwapchain::cleanupSwapChain(VkDevice device, VmaAllocator /*allocator*/) {
  for (auto fb : swapChainFramebuffers) vkDestroyFramebuffer(device, fb, nullptr);
  swapChainFramebuffers.clear();
  swapChainImages.clear();
  colorBufferImage.clear();
  colorBufferImageView.clear();
  if (swapchain != VK_NULL_HANDLE) { vkDestroySwapchainKHR(device, swapchain, nullptr); swapchain = VK_NULL_HANDLE; }
}

void VulkanSwapchain::createColorBufferImage(const VulkanDevice &device) {
  colorBufferImage.resize(swapChainImages.size());
  colorBufferImageView.resize(swapChainImages.size());

  VkFormat fmt = chooseSupportedFormat(device.getPhysicalDevice(),
      { VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM },
      VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT);

  for (size_t i = 0; i < swapChainImages.size(); ++i) {
    colorBufferImage[i] = createImage(device.getAllocator(),
        swapChainExtent.width, swapChainExtent.height, fmt,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT);
    colorBufferImageView[i] = ImageViewHandle(device.getLogicalDevice(),
        createImageView(device.getLogicalDevice(), colorBufferImage[i].get(), fmt, VK_IMAGE_ASPECT_COLOR_BIT));
  }
}

void VulkanSwapchain::createFramebuffers(VkDevice device, VkRenderPass renderPass) {
  swapChainFramebuffers.resize(swapChainImages.size());
  for (size_t i = 0; i < swapChainImages.size(); ++i) {
    // Composition pass: [swapchain image, colorBuffer] — 2 attachments
    std::array<VkImageView, 2> attachments = {
        swapChainImages[i].imageView.get(),
        colorBufferImageView[i].get()
    };
    VkFramebufferCreateInfo ci = {};
    ci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    ci.renderPass      = renderPass;
    ci.attachmentCount = static_cast<uint32_t>(attachments.size());
    ci.pAttachments    = attachments.data();
    ci.width           = swapChainExtent.width;
    ci.height          = swapChainExtent.height;
    ci.layers          = 1;
    if (vkCreateFramebuffer(device, &ci, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
      throw std::runtime_error("Failed to create framebuffer");
  }
}

void VulkanSwapchain::createCommandBuffers(VkDevice device, VkCommandPool commandPool) {
  commandBuffers.resize(swapChainFramebuffers.size());
  VkCommandBufferAllocateInfo ai = {};
  ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool        = commandPool;
  ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
  if (vkAllocateCommandBuffers(device, &ai, commandBuffers.data()) != VK_SUCCESS)
    throw std::runtime_error("Failed to allocate command buffers");
}

// ---------------------------------------------------------------------------
// Support
// ---------------------------------------------------------------------------

VkSurfaceFormatKHR VulkanSwapchain::chooseBestSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &formats) {
  if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED)
    return { VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
  for (auto &f : formats)
    if ((f.format == VK_FORMAT_R8G8B8A8_UNORM || f.format == VK_FORMAT_B8G8R8A8_UNORM) &&
        f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) return f;
  return formats[0];
}

VkPresentModeKHR VulkanSwapchain::chooseBestPresentationMode(const std::vector<VkPresentModeKHR> &modes) {
  for (auto m : modes) if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D VulkanSwapchain::chooseSwapExtent(const VkSurfaceCapabilitiesKHR &caps, GLFWwindow *window) {
  if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max()) return caps.currentExtent;
  int w, h;
  glfwGetFramebufferSize(window, &w, &h);
  VkExtent2D e = { static_cast<uint32_t>(w), static_cast<uint32_t>(h) };
  e.width  = std::clamp(e.width,  caps.minImageExtent.width,  caps.maxImageExtent.width);
  e.height = std::clamp(e.height, caps.minImageExtent.height, caps.maxImageExtent.height);
  return e;
}

VkFormat VulkanSwapchain::chooseSupportedFormat(VkPhysicalDevice physicalDevice,
                                                  const std::vector<VkFormat> &formats,
                                                  VkImageTiling tiling,
                                                  VkFormatFeatureFlags features) {
  for (VkFormat fmt : formats) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, fmt, &props);
    if (tiling == VK_IMAGE_TILING_LINEAR  && (props.linearTilingFeatures  & features) == features) return fmt;
    if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) return fmt;
  }
  throw std::runtime_error("Failed to find supported format");
}

AllocatedImage VulkanSwapchain::createImage(VmaAllocator allocator, uint32_t width, uint32_t height,
                                             VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage) {
  VkImageCreateInfo ci = {};
  ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ci.imageType     = VK_IMAGE_TYPE_2D;
  ci.extent        = { width, height, 1 };
  ci.mipLevels     = 1;
  ci.arrayLayers   = 1;
  ci.format        = format;
  ci.tiling        = tiling;
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  ci.usage         = usage;
  ci.samples       = VK_SAMPLE_COUNT_1_BIT;
  ci.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo aci = {};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  VkImage img = VK_NULL_HANDLE;
  VmaAllocation alloc = VK_NULL_HANDLE;
  if (vmaCreateImage(allocator, &ci, &aci, &img, &alloc, nullptr) != VK_SUCCESS)
    throw std::runtime_error("Failed to create image");
  return AllocatedImage(allocator, img, alloc);
}

VkImageView VulkanSwapchain::createImageView(VkDevice device, VkImage image,
                                              VkFormat format, VkImageAspectFlags aspect) {
  VkImageViewCreateInfo ci = {};
  ci.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  ci.image                           = image;
  ci.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
  ci.format                          = format;
  ci.components                      = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                         VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
  ci.subresourceRange.aspectMask     = aspect;
  ci.subresourceRange.baseMipLevel   = 0;
  ci.subresourceRange.levelCount     = 1;
  ci.subresourceRange.baseArrayLayer = 0;
  ci.subresourceRange.layerCount     = 1;

  VkImageView view;
  if (vkCreateImageView(device, &ci, nullptr, &view) != VK_SUCCESS)
    throw std::runtime_error("Failed to create image view");
  return view;
}
