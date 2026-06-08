#pragma once

#include "DescriptorManager.h"
#include "Utilities.h"
#include "VulkanDevice.h"

#include <vector>
#include <vulkan/vulkan.h>

class MaterialProbePass {
public:
  void create(VulkanDevice &device, uint32_t frameCount,
              const DescriptorManager &descriptorManager,
              VkPipelineCache pipelineCache);
  void cleanup();

  void record(VkCommandBuffer cmd, uint32_t frameIndex, VkDescriptorSet vpSet,
              VkDescriptorSet gBufferSet, glm::uvec2 pixel);
  MaterialProbeResult read(uint32_t frameIndex);

private:
  VulkanDevice *device = nullptr;

  VkDescriptorSetLayout outputSetLayout = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> outputSets;
  std::vector<AllocatedBuffer> outputBuffers;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
};
