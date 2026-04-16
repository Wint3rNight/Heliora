#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>
#include "Utilities.h"

class VulkanDevice;
class DescriptorManager;

class TextureManager {
public:
  TextureManager() = default;
  ~TextureManager() = default;

  // Non-copyable
  TextureManager(const TextureManager &) = delete;
  TextureManager &operator=(const TextureManager &) = delete;

  void init(const VulkanDevice &device);
  void cleanup(VkDevice logicalDevice, VmaAllocator allocator);

  // Loads a texture or retrieves it if already loaded. Returns descriptor set index.
  int loadTexture(const std::string &filename, const VulkanDevice &device,
                  DescriptorManager &descriptorManager);

private:
  VkSampler textureSampler = VK_NULL_HANDLE;

  // Caches for textures we own
  std::vector<AllocatedImage> textureImages;
  std::vector<ImageViewHandle> textureImageViews;

  // Map to deduplicate textures (filename -> descriptor location ID)
  std::unordered_map<std::string, int> textureMap;

  // --- Helpers ---
  void createTextureSampler(VkDevice logicalDevice,
                            VkPhysicalDevice physicalDevice);
  int createTextureImage(const std::string &filename, const VulkanDevice &device);
  unsigned char *loadTextureFile(const std::string &filename, int *width,
                                 int *height, VkDeviceSize *imageSize);
};
