#pragma once

#include "DescriptorManager.h"
#include "Utilities.h"
#include "VulkanDevice.h"

#include <vector>
#include <vulkan/vulkan.h>

class SsaoPass {
public:
  void create(VulkanDevice &device, VkExtent2D extent, size_t swapCount,
              const DescriptorManager &descriptorManager);
  void cleanup();

  void recordCommands(VkCommandBuffer cmd, uint32_t currentImage,
                      VkDescriptorSet vpSet, VkDescriptorSet gBufferSet);

  std::vector<VkImageView> views() const;
  VkSampler sampler() const { return resultSampler; }

private:
  VulkanDevice *device = nullptr;
  VkExtent2D renderExtent = {};
  VkFormat format = VK_FORMAT_R16_SFLOAT;

  std::vector<AllocatedImage> images;
  std::vector<ImageViewHandle> imageViews;
  VkSampler resultSampler = VK_NULL_HANDLE;
  VkDescriptorSetLayout outputSetLayout = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> outputSets;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
};
