#include "TextureManager.h"
#include "DescriptorManager.h"
#include "Utilities.h"
#include "VulkanDevice.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdexcept>

void TextureManager::init(const VulkanDevice &device) {
  createTextureSampler(device.getLogicalDevice(), device.getPhysicalDevice());
}

void TextureManager::cleanup(VkDevice logicalDevice) {
  for (auto imageView : textureImageViews) {
    vkDestroyImageView(logicalDevice, imageView, nullptr);
  }
  for (auto image : textureImages) {
    vkDestroyImage(logicalDevice, image, nullptr);
  }
  for (auto memory : textureImageMemory) {
    vkFreeMemory(logicalDevice, memory, nullptr);
  }
  if (textureSampler != VK_NULL_HANDLE) {
    vkDestroySampler(logicalDevice, textureSampler, nullptr);
  }
}

int TextureManager::loadTexture(const std::string &filename,
                                const VulkanDevice &device,
                                DescriptorManager &descriptorManager) {
  // Check if already loaded
  auto it = textureMap.find(filename);
  if (it != textureMap.end()) {
    return it->second;
  }

  // Use the internal helper to parse avoiding duplication
  int textureImageLoc = createTextureImage(filename, device);

  // Create image view
  VkImageViewCreateInfo viewCreateInfo = {};
  viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewCreateInfo.image = textureImages[textureImageLoc];
  viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
  viewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewCreateInfo.subresourceRange.baseMipLevel = 0;
  viewCreateInfo.subresourceRange.levelCount = 1;
  viewCreateInfo.subresourceRange.baseArrayLayer = 0;
  viewCreateInfo.subresourceRange.layerCount = 1;

  VkImageView imageView;
  VkResult result = vkCreateImageView(device.getLogicalDevice(), &viewCreateInfo,
                                      nullptr, &imageView);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create texture image view");
  }
  textureImageViews.push_back(imageView);

  // Generate descriptor
  int descriptorLoc = descriptorManager.createTextureDescriptor(
      device.getLogicalDevice(), imageView, textureSampler);

  // Cache it
  textureMap[filename] = descriptorLoc;
  return descriptorLoc;
}

void TextureManager::createTextureSampler(VkDevice logicalDevice,
                                          VkPhysicalDevice physicalDevice) {
  VkSamplerCreateInfo samplerCreateInfo = {};
  samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
  samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
  samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerCreateInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
  samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerCreateInfo.mipLodBias = 0.0f;
  samplerCreateInfo.minLod = 0.0f;
  samplerCreateInfo.maxLod = 0.0f;
  
  VkPhysicalDeviceProperties deviceProperties;
  vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
  samplerCreateInfo.anisotropyEnable = VK_TRUE;
  samplerCreateInfo.maxAnisotropy = deviceProperties.limits.maxSamplerAnisotropy;

  if (vkCreateSampler(logicalDevice, &samplerCreateInfo, nullptr,
                      &textureSampler) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create texture sampler");
  }
}

int TextureManager::createTextureImage(const std::string &filename,
                                       const VulkanDevice &device) {
  int width, height;
  VkDeviceSize imageSize;
  unsigned char *imageData = loadTextureFile(filename, &width, &height, &imageSize);

  VkBuffer imageStagingBuffer;
  VkDeviceMemory imageStagingBufferMemory;
  createBuffer(device.getPhysicalDevice(), device.getLogicalDevice(), imageSize,
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               &imageStagingBuffer, &imageStagingBufferMemory);

  void *data;
  vkMapMemory(device.getLogicalDevice(), imageStagingBufferMemory, 0, imageSize,
              0, &data);
  memcpy(data, imageData, static_cast<size_t>(imageSize));
  vkUnmapMemory(device.getLogicalDevice(), imageStagingBufferMemory);

  stbi_image_free(imageData); // Now it's stbi_uc* equivalent

  VkImage texImage;
  VkDeviceMemory texImageMemory;
  
  VkImageCreateInfo imageCreateInfo = {};
  imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
  imageCreateInfo.extent.width = width;
  imageCreateInfo.extent.height = height;
  imageCreateInfo.extent.depth = 1;
  imageCreateInfo.mipLevels = 1;
  imageCreateInfo.arrayLayers = 1;
  imageCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
  imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateImage(device.getLogicalDevice(), &imageCreateInfo, nullptr, &texImage) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create image");
  }

  VkMemoryRequirements memoryRequirements;
  vkGetImageMemoryRequirements(device.getLogicalDevice(), texImage, &memoryRequirements);

  VkMemoryAllocateInfo memoryAllocInfo = {};
  memoryAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memoryAllocInfo.allocationSize = memoryRequirements.size;
  memoryAllocInfo.memoryTypeIndex = findMemoryTypeIndex(
      device.getPhysicalDevice(), memoryRequirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  if (vkAllocateMemory(device.getLogicalDevice(), &memoryAllocInfo, nullptr, &texImageMemory) != VK_SUCCESS) {
      throw std::runtime_error("Failed to allocate image memory");
  }

  vkBindImageMemory(device.getLogicalDevice(), texImage, texImageMemory, 0);

  transitionImageLayout(device.getLogicalDevice(), device.getGraphicsQueue(),
                        device.getGraphicsCommandPool(), texImage,
                        VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

  copyImageBuffer(device.getLogicalDevice(), device.getGraphicsQueue(),
                  device.getGraphicsCommandPool(), imageStagingBuffer, texImage,
                  width, height);

  transitionImageLayout(device.getLogicalDevice(), device.getGraphicsQueue(),
                        device.getGraphicsCommandPool(), texImage,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  textureImages.push_back(texImage);
  textureImageMemory.push_back(texImageMemory);

  vkDestroyBuffer(device.getLogicalDevice(), imageStagingBuffer, nullptr);
  vkFreeMemory(device.getLogicalDevice(), imageStagingBufferMemory, nullptr);

  return static_cast<int>(textureImages.size() - 1);
}

unsigned char *TextureManager::loadTextureFile(const std::string &filename,
                                               int *width, int *height,
                                               VkDeviceSize *imageSize) {
  int channels;
  std::string fileLoc = "Textures/" + filename;
  unsigned char *image =
      stbi_load(fileLoc.c_str(), width, height, &channels, STBI_rgb_alpha);
  if (!image) {
    fileLoc = "../Textures/" + filename;
    image =
        stbi_load(fileLoc.c_str(), width, height, &channels, STBI_rgb_alpha);
  }
  if (!image) {
    throw std::runtime_error("Failed to load texture image file!(" + fileLoc +
                             ")");
  }
  *imageSize = static_cast<VkDeviceSize>(*width) * (*height) * 4;
  return image;
}
