#pragma once

#include <vector>
#include <vulkan/vulkan.h>

#include "DescriptorManager.h"

class VulkanPipeline {
public:
  VulkanPipeline() = default;
  ~VulkanPipeline() = default;

  // Non-copyable
  VulkanPipeline(const VulkanPipeline &) = delete;
  VulkanPipeline &operator=(const VulkanPipeline &) = delete;

  void createPipelines(VkDevice device, VkRenderPass renderPass,
                       VkExtent2D extent,
                       const DescriptorManager &descriptors);
  void cleanup(VkDevice device);

  VkPipeline getGraphicsPipeline() const { return graphicsPipeline; }
  VkPipelineLayout getGraphicsLayout() const { return pipelineLayout; }
  VkPipeline getSecondPipeline() const { return secondPipeline; }
  VkPipelineLayout getSecondLayout() const { return secondPipelineLayout; }

private:
  VkPipeline graphicsPipeline = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline secondPipeline = VK_NULL_HANDLE;
  VkPipelineLayout secondPipelineLayout = VK_NULL_HANDLE;
  VkPipelineCache pipelineCache = VK_NULL_HANDLE;

  VkShaderModule createShaderModule(VkDevice device,
                                    const std::vector<char> &code);
};
