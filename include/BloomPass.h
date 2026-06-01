#pragma once

#include "Utilities.h"
#include "VulkanDevice.h"

#include <array>
#include <vector>
#include <vulkan/vulkan.h>

class BloomPass {
public:
  static constexpr uint32_t kMipCount = 6;

  void create(VulkanDevice &device, VkExtent2D extent, size_t swapCount,
              const std::vector<ImageViewHandle> &litViews);
  void cleanup();

  void record(VkCommandBuffer cmd, uint32_t currentImage,
              const std::vector<AllocatedImage> &litImages, bool enabled,
              int debugMode, float threshold, float radius, float intensity);

  std::vector<VkImageView> mip0Views() const;
  VkSampler sampler() const { return bloomSampler; }

private:
  struct MipResources {
    VkExtent2D extent = {};
    std::vector<AllocatedImage> images;
    std::vector<ImageViewHandle> views;
  };

  VulkanDevice *device = nullptr;
  VkExtent2D renderExtent = {};
  VkFormat bloomFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

  std::array<MipResources, kMipCount> mips;
  VkSampler bloomSampler = VK_NULL_HANDLE;
  VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> downsampleSets;
  std::vector<VkDescriptorSet> upsampleSets;
  VkPipelineLayout downsamplePipelineLayout = VK_NULL_HANDLE;
  VkPipelineLayout upsamplePipelineLayout = VK_NULL_HANDLE;
  VkPipeline downsamplePipeline = VK_NULL_HANDLE;
  VkPipeline upsamplePipeline = VK_NULL_HANDLE;
};
