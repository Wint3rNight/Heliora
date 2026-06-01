#pragma once

#include <vector>
#include <vulkan/vulkan.h>

class CompositePass {
public:
  void create(VkDevice device, VkRenderPass renderPass, VkExtent2D extent,
              const std::vector<VkImageView> &swapViews,
              const std::vector<VkImageView> &colorViews,
              const std::vector<VkImageView> &historyViews);
  void cleanup();

  void record(VkCommandBuffer cmd, uint32_t currentImage,
              uint32_t historyIndex, VkDescriptorSet vpSet,
              VkDescriptorSet gBufferSet, VkDescriptorSet inputSet,
              VkDescriptorSet taaSet, VkPipeline deferredPipeline,
              VkPipelineLayout deferredLayout, VkPipeline secondPipeline,
              VkPipelineLayout secondLayout);

  size_t framebufferIndex(uint32_t historyIndex, uint32_t currentImage) const;

private:
  VkDevice device = VK_NULL_HANDLE;
  VkRenderPass renderPass = VK_NULL_HANDLE;
  VkExtent2D renderExtent = {};
  size_t swapCount = 0;
  std::vector<VkFramebuffer> framebuffers;
};
