#pragma once

#include "Utilities.h"
#include "VulkanDevice.h"

#include <vector>
#include <vulkan/vulkan.h>

class SsgiPass {
public:
  void create(VulkanDevice &device, VkExtent2D fullExtent, VkFormat format,
              VkRenderPass renderPass, size_t historyCount);
  void cleanup();

  void record(VkCommandBuffer cmd, uint32_t historyIndex, VkDescriptorSet vpSet,
              VkDescriptorSet gBufferSet, VkDescriptorSet prevSet,
              VkPipeline pipeline, VkPipelineLayout pipelineLayout);

  uint32_t historyIndex(uint32_t frameCounter) const;
  size_t historyCount() const { return historyViews.size(); }
  std::vector<VkImageView> views() const;
  VkSampler sampler() const { return historySampler; }

private:
  VulkanDevice *device = nullptr;
  VkExtent2D renderExtent = {};
  VkFormat historyFormat = VK_FORMAT_UNDEFINED;
  VkRenderPass targetRenderPass = VK_NULL_HANDLE;

  std::vector<AllocatedImage> historyImages;
  std::vector<ImageViewHandle> historyViews;
  std::vector<VkFramebuffer> framebuffers;
  VkSampler historySampler = VK_NULL_HANDLE;
};
