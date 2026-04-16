#include "VulkanSwapchain.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

// --- Public interface ---

VkFormat VulkanSwapchain::queryImageFormat(const VulkanDevice &device) const {
  uint32_t formatCount = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device.getPhysicalDevice(),
                                       device.getSurface(), &formatCount,
                                       nullptr);
  std::vector<VkSurfaceFormatKHR> formats(formatCount);
  if (formatCount != 0) {
    vkGetPhysicalDeviceSurfaceFormatsKHR(device.getPhysicalDevice(),
                                         device.getSurface(), &formatCount,
                                         formats.data());
  }

  // Same logic as chooseBestSurfaceFormat
  if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED) {
    return VK_FORMAT_R8G8B8A8_UNORM;
  }
  for (const auto &format : formats) {
    if ((format.format == VK_FORMAT_R8G8B8A8_UNORM ||
         format.format == VK_FORMAT_B8G8R8A8_UNORM) &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return format.format;
    }
  }
  return formats[0].format;
}

void VulkanSwapchain::init(const VulkanDevice &device, VkRenderPass renderPass,
                           GLFWwindow *window) {
  createSwapChain(device, window);
  createColorBufferImage(device);
  createDepthBufferImage(device);
  createFramebuffers(device.getLogicalDevice(), renderPass);
  createCommandBuffers(device.getLogicalDevice(),
                       device.getGraphicsCommandPool());
}

void VulkanSwapchain::recreate(const VulkanDevice &device,
                               VkRenderPass renderPass, GLFWwindow *window) {
  int width = 0, height = 0;
  glfwGetFramebufferSize(window, &width, &height);
  while (width == 0 || height == 0) {
    glfwGetFramebufferSize(window, &width, &height);
    glfwWaitEvents();
  }
  vkDeviceWaitIdle(device.getLogicalDevice());

  vkResetCommandPool(device.getLogicalDevice(),
                     device.getGraphicsCommandPool(), 0);

  cleanupSwapChain(device.getLogicalDevice(), device.getAllocator());
  createSwapChain(device, window);
  createColorBufferImage(device);
  createDepthBufferImage(device);
  createFramebuffers(device.getLogicalDevice(), renderPass);
  createCommandBuffers(device.getLogicalDevice(),
                       device.getGraphicsCommandPool());
}

void VulkanSwapchain::cleanup(VkDevice device, VmaAllocator allocator) { cleanupSwapChain(device, allocator); }

// --- Swapchain creation ---

void VulkanSwapchain::createSwapChain(const VulkanDevice &device,
                                      GLFWwindow *window) {
  // get swap chain details to pick best settings
  SwapChainDetails swapChainDetails;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
      device.getPhysicalDevice(), device.getSurface(),
      &swapChainDetails.surfaceCapabilities);

  uint32_t formatCount = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device.getPhysicalDevice(),
                                       device.getSurface(), &formatCount,
                                       nullptr);
  if (formatCount != 0) {
    swapChainDetails.formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device.getPhysicalDevice(),
                                         device.getSurface(), &formatCount,
                                         swapChainDetails.formats.data());
  }

  uint32_t presentModeCount = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device.getPhysicalDevice(),
                                            device.getSurface(),
                                            &presentModeCount, nullptr);
  if (presentModeCount != 0) {
    swapChainDetails.presentModes.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        device.getPhysicalDevice(), device.getSurface(), &presentModeCount,
        swapChainDetails.presentModes.data());
  }

  // find the best settings for the swap chain
  VkSurfaceFormatKHR surfaceFormat =
      chooseBestSurfaceFormat(swapChainDetails.formats);
  VkPresentModeKHR presentationMode =
      chooseBestPresentationMode(swapChainDetails.presentModes);
  VkExtent2D extent =
      chooseSwapExtent(swapChainDetails.surfaceCapabilities, window);

  // no of images in swap chain, ask for 1 more than minimum to allow triple
  // buffering
  uint32_t imageCount = swapChainDetails.surfaceCapabilities.minImageCount + 1;

  // min image count must be less than the max image count (0 is max for no max)
  //  if max image count is 0, there is no maximum
  if (swapChainDetails.surfaceCapabilities.maxImageCount > 0 &&
      imageCount > swapChainDetails.surfaceCapabilities.maxImageCount) {
    imageCount = swapChainDetails.surfaceCapabilities.maxImageCount;
  }

  // create info for swap chain creation
  VkSwapchainCreateInfoKHR swapChainCreateInfo = {};
  swapChainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  swapChainCreateInfo.surface = device.getSurface();
  swapChainCreateInfo.imageFormat = surfaceFormat.format;
  swapChainCreateInfo.imageColorSpace = surfaceFormat.colorSpace;
  swapChainCreateInfo.presentMode = presentationMode;
  swapChainCreateInfo.imageExtent = extent;
  swapChainCreateInfo.minImageCount = imageCount;
  swapChainCreateInfo.imageArrayLayers =
      1; // number of layers each image consists of
  swapChainCreateInfo.imageUsage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // what attachments images will be
                                           // used as

  swapChainCreateInfo.preTransform =
      swapChainDetails.surfaceCapabilities
          .currentTransform; // transform to perform on swapchain images
  swapChainCreateInfo.compositeAlpha =
      VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // alpha blending to use with other
                                         // windows
  swapChainCreateInfo.clipped =
      VK_TRUE; // whether to clip obscured pixels (by other windows)

  // get queue family indices to determine sharing mode
  QueueFamilyIndices indices = device.getQueueFamilies();

  // if graphics and presentation families are different, swapchain must be
  // shared between them which is not fast but can be done so why not
  if (indices.graphicsFamily != indices.presentationFamily) {
    // queue family indices to share between
    uint32_t queueFamilyIndices[] = {(uint32_t)indices.graphicsFamily,
                                     (uint32_t)indices.presentationFamily};
    swapChainCreateInfo.imageSharingMode =
        VK_SHARING_MODE_CONCURRENT; // images can be used across multiple queue
    swapChainCreateInfo.queueFamilyIndexCount =
        2; // number of queue families to share between
    swapChainCreateInfo.pQueueFamilyIndices =
        queueFamilyIndices; // list of queue families to share between

  } else {
    swapChainCreateInfo.imageSharingMode =
        VK_SHARING_MODE_EXCLUSIVE; // an image is owned by one queue family at a
                                   // time, best performance
    swapChainCreateInfo.queueFamilyIndexCount = 0;     // Optional
    swapChainCreateInfo.pQueueFamilyIndices = nullptr; // Optional
  }

  // if old swap chain is destroyed, we can use its resources for the new swap
  // chain
  swapChainCreateInfo.oldSwapchain =
      VK_NULL_HANDLE; // handle to old swap chain in case of recreation

  // create the swap chain
  VkResult result = vkCreateSwapchainKHR(
      device.getLogicalDevice(), &swapChainCreateInfo, nullptr, &swapchain);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create swap chain");
  }

  // store swap chain images
  swapChainImageFormat = surfaceFormat.format;
  swapChainExtent = extent;

  // get swap chain images
  uint32_t swapChainImageCount;
  vkGetSwapchainImagesKHR(device.getLogicalDevice(), swapchain,
                          &swapChainImageCount, nullptr);
  std::vector<VkImage> images(swapChainImageCount);
  vkGetSwapchainImagesKHR(device.getLogicalDevice(), swapchain,
                          &swapChainImageCount, images.data());

  for (VkImage image : images) {
    // store image handle
    SwapChainImage swapChainImage = {};
    swapChainImage.image = image;

    // create image view
    swapChainImage.imageView = ImageViewHandle(
        device.getLogicalDevice(),
        createImageView(device.getLogicalDevice(), image, swapChainImageFormat,
                        VK_IMAGE_ASPECT_COLOR_BIT));

    // add swap chain image to list
    swapChainImages.push_back(std::move(swapChainImage));
  }
}

// --- Swapchain cleanup ---

void VulkanSwapchain::cleanupSwapChain(VkDevice device, VmaAllocator allocator) {
  for (auto framebuffer : swapChainFramebuffers) {
    vkDestroyFramebuffer(device, framebuffer, nullptr);
  }
  swapChainFramebuffers.clear();

  swapChainImages.clear();

  depthBufferImage.clear();
  depthBufferImageView.clear();

  colorBufferImage.clear();
  colorBufferImageView.clear();

  if (swapchain != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(device, swapchain, nullptr);
    swapchain = VK_NULL_HANDLE;
  }
}

// --- Color buffer image creation ---

void VulkanSwapchain::createColorBufferImage(const VulkanDevice &device) {
  // resize supported format for color attachment
  colorBufferImage.resize(swapChainImages.size());
  colorBufferImageView.resize(swapChainImages.size());

  // get supported format for color attachment
  VkFormat colorFormat = chooseSupportedFormat(
      device.getPhysicalDevice(), {VK_FORMAT_R8G8B8A8_UNORM},
      VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

  for (size_t i = 0; i < swapChainImages.size(); i++) {
    // create the color buffer image and its memory
    colorBufferImage[i] = createImage(
        device.getAllocator(),
        swapChainExtent.width, swapChainExtent.height, colorFormat,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT);

    // create color buffer image view
    colorBufferImageView[i] = ImageViewHandle(
        device.getLogicalDevice(),
        createImageView(device.getLogicalDevice(), colorBufferImage[i].get(),
                        colorFormat, VK_IMAGE_ASPECT_COLOR_BIT));
  }
}

// --- Depth buffer image creation ---

void VulkanSwapchain::createDepthBufferImage(const VulkanDevice &device) {
  depthBufferImage.resize(swapChainImages.size());
  depthBufferImageView.resize(swapChainImages.size());

  // get supported depth format for depth buffer image
  VkFormat depthFormat = chooseSupportedFormat(
      device.getPhysicalDevice(),
      {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D32_SFLOAT,
       VK_FORMAT_D24_UNORM_S8_UINT},
      VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);

  for (size_t i = 0; i < swapChainImages.size(); i++) {

    // create depth buffer image and its memory
    depthBufferImage[i] = createImage(
        device.getAllocator(),
        swapChainExtent.width, swapChainExtent.height, depthFormat,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT);

    // create depth buffer image view
    depthBufferImageView[i] = ImageViewHandle(
        device.getLogicalDevice(),
        createImageView(device.getLogicalDevice(), depthBufferImage[i].get(),
                        depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT));
  }
}

// --- Framebuffer creation ---

void VulkanSwapchain::createFramebuffers(VkDevice device,
                                         VkRenderPass renderPass) {
  swapChainFramebuffers.resize(
      swapChainImages.size()); // resize fb list to no.of swapchain images
                               // create framebuffer for each swap chain image
  for (size_t i = 0; i < swapChainImages.size(); i++) {
    std::array<VkImageView, 3> attachments = {swapChainImages[i].imageView.get(),
                                              colorBufferImageView[i].get(),
                                              depthBufferImageView[i].get()};

    VkFramebufferCreateInfo framebufferCreateInfo = {};
    framebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferCreateInfo.renderPass = renderPass; // render pass to use
    framebufferCreateInfo.attachmentCount = static_cast<uint32_t>(
        attachments.size()); // number of attachments in framebuffer (only color
                             // attachment)
    framebufferCreateInfo.pAttachments =
        attachments.data(); // list of attachments(1:1 with render pass)
    framebufferCreateInfo.width = swapChainExtent.width;
    framebufferCreateInfo.height = swapChainExtent.height;
    framebufferCreateInfo.layers =
        1; // number of layers in framebuffer (1 unless doing stereoscopic 3D)

    VkResult result = vkCreateFramebuffer(device, &framebufferCreateInfo,
                                          nullptr, &swapChainFramebuffers[i]);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to create framebuffer");
    }
  }
}

// --- Command buffer creation ---

void VulkanSwapchain::createCommandBuffers(VkDevice device,
                                           VkCommandPool commandPool) {
  commandBuffers.resize(
      swapChainFramebuffers.size()); // resize cb list to no. of framebuffers

  VkCommandBufferAllocateInfo cbAllocInfo = {};
  cbAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbAllocInfo.commandPool = commandPool; // command pool to allocate from
  cbAllocInfo.level =
      VK_COMMAND_BUFFER_LEVEL_PRIMARY; // level of the command buffers (primary
                                       // can be submitted, secondary
                                       // cannot(primary executed by queues,
                                       // secondary executed by primary))
  cbAllocInfo.commandBufferCount = static_cast<uint32_t>(
      commandBuffers.size()); // number of
                              // command buffers to allocate
  // allocate command buffers and place handles in array of buffers
  VkResult result = vkAllocateCommandBuffers(device, &cbAllocInfo,
                                             commandBuffers.data());
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate command buffers");
  }
}

// --- Support functions ---

VkSurfaceFormatKHR VulkanSwapchain::chooseBestSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR> &formats) {
  // if only 1 format is available and is undefined, means all formats are
  // available
  if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED) {
    return {VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
  }

  // look for optimal surface format
  for (const auto &format : formats) {
    if ((format.format == VK_FORMAT_R8G8B8A8_UNORM ||
         format.format == VK_FORMAT_B8G8R8A8_UNORM) &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return format;
    }
  }

  // if optimal format isn't found, return first format available
  return formats[0];
}

VkPresentModeKHR VulkanSwapchain::chooseBestPresentationMode(
    const std::vector<VkPresentModeKHR> &presentationModes) {
  // look for mailbox presentation mode, best for triple buffering, low
  // latency
  for (const auto &presentationMode : presentationModes) {
    if (presentationMode == VK_PRESENT_MODE_MAILBOX_KHR) {
      return presentationMode;
    }
  }
  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D VulkanSwapchain::chooseSwapExtent(
    const VkSurfaceCapabilitiesKHR &surfaceCapabilities, GLFWwindow *window) {
  // if current extent is max, then extent can vary, so we set it to the
  // window size
  if (surfaceCapabilities.currentExtent.width !=
      std::numeric_limits<uint32_t>::max()) {
    return surfaceCapabilities.currentExtent;
  } else {
    // if value vary, set manually

    // get window size
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    // create new extent using window size
    VkExtent2D newExtent = {};
    newExtent.width = static_cast<uint32_t>(width);
    newExtent.height = static_cast<uint32_t>(height);

    // surface also defines min and max image extents so make sure the new
    // extent is within bounds
    newExtent.width = std::max(
        surfaceCapabilities.minImageExtent.width,
        std::min(surfaceCapabilities.maxImageExtent.width, newExtent.width));
    newExtent.height = std::max(
        surfaceCapabilities.minImageExtent.height,
        std::min(surfaceCapabilities.maxImageExtent.height, newExtent.height));
    return newExtent;
  }
}

VkFormat VulkanSwapchain::chooseSupportedFormat(
    VkPhysicalDevice physicalDevice, const std::vector<VkFormat> &formats,
    VkImageTiling tiling, VkFormatFeatureFlags featureFlags) {
  // loop through candidate formats and check if it supports the feature needed
  for (VkFormat format : formats) {
    // get format properties for format on this device
    VkFormatProperties properties;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
    // check if format supports features for given tiling option
    if (tiling == VK_IMAGE_TILING_LINEAR &&
        (properties.linearTilingFeatures & featureFlags) == featureFlags) {
      return format;
    } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
               (properties.optimalTilingFeatures & featureFlags) ==
                   featureFlags) {
      return format;
    }
  }
  throw std::runtime_error("Failed to find supported format");
}

// --- Image helpers ---

AllocatedImage VulkanSwapchain::createImage(VmaAllocator allocator,
                                     uint32_t width,
                                     uint32_t height, VkFormat format,
                                     VkImageTiling tiling,
                                     VkImageUsageFlags useFlags) {
  // CREATE IMAGE
  // image create info
  VkImageCreateInfo imageCreateInfo = {};
  imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageCreateInfo.imageType = VK_IMAGE_TYPE_2D; // type of image (1D, 2D, 3D)
  imageCreateInfo.extent.width = width;         // width of image
  imageCreateInfo.extent.height = height;       // height of image
  imageCreateInfo.extent.depth = 1;             // depth of image (for 3D
                                                // images)
  imageCreateInfo.mipLevels = 1;                // number of mipmap levels
  imageCreateInfo.arrayLayers = 1;              // number of array layers
  imageCreateInfo.format = format;              // format of image data
  imageCreateInfo.tiling =
      tiling; // how image data should be tiled in memory (linear or optimal)
  imageCreateInfo.initialLayout =
      VK_IMAGE_LAYOUT_UNDEFINED; // initial layout of image
  imageCreateInfo.usage =
      useFlags; // intended usage of image (color attachment, depth stencil
                // attachment, sampled image, etc.)
  imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT; // number of samples per
                                                   // pixel (for multisampling)
  imageCreateInfo.sharingMode =
      VK_SHARING_MODE_EXCLUSIVE; // how image will be shared between queues

  VmaAllocationCreateInfo allocCreateInfo = {};
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  // create image
  VkImage image = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  VkResult result =
      vmaCreateImage(allocator, &imageCreateInfo, &allocCreateInfo, &image,
                     &allocation, nullptr);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create image");
  }

  return AllocatedImage(allocator, image, allocation);
}

VkImageView VulkanSwapchain::createImageView(VkDevice device, VkImage image,
                                             VkFormat format,
                                             VkImageAspectFlags aspectFlags) {
  VkImageViewCreateInfo viewCreateInfo = {};
  viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewCreateInfo.image = image; // image to create view for
  viewCreateInfo.viewType =
      VK_IMAGE_VIEW_TYPE_2D;      // type of image (1D, 2D, 3D, cube)
  viewCreateInfo.format = format; // format of the image data
  viewCreateInfo.components.r =
      VK_COMPONENT_SWIZZLE_IDENTITY; // allows remapping of color to other
                                     // color
  viewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  // subresourceRange describes what the image's purpose is and which part
  // of the image to access
  viewCreateInfo.subresourceRange.aspectMask =
      aspectFlags; // which aspect of the image to view (color, depth,
                   // stencil)
  viewCreateInfo.subresourceRange.baseMipLevel =
      0; // start mipmap level to view from
  viewCreateInfo.subresourceRange.levelCount =
      1; // number of mipmap levels to view
  viewCreateInfo.subresourceRange.baseArrayLayer =
      0; // start array layer to view from
  viewCreateInfo.subresourceRange.layerCount =
      1; // number of array layers to view

  // create the image view
  VkImageView imageView;
  VkResult result = vkCreateImageView(device, &viewCreateInfo,
                                      nullptr, &imageView);

  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create image view");
  }

  return imageView;
}
