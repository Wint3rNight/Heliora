#pragma once

#include "DescriptorManager.h"
#include <vector>
#include <vulkan/vulkan.h>

class VulkanPipeline {
public:
  VulkanPipeline() = default;
  ~VulkanPipeline() = default;

  VulkanPipeline(const VulkanPipeline &) = delete;
  VulkanPipeline &operator=(const VulkanPipeline &) = delete;

  // gBufferPass  : geometry pass (shader.vert + shader.frag → 3 MRT outputs)
  // compositionPass : 2-subpass tone-mapping pass (deferred.frag + second.frag)
  // shadowPass   : depth-only (unchanged)
  void createPipelines(VkDevice device, VkRenderPass gBufferPass,
                       VkRenderPass compositionPass, VkRenderPass shadowPass,
                       VkExtent2D extent, const DescriptorManager &descriptors);
  void cleanup(VkDevice device);

  VkPipeline getGraphicsPipeline() const { return graphicsPipeline; }
  VkPipelineLayout getGraphicsLayout() const { return pipelineLayout; }
  VkPipeline getInstancedPipeline() const { return instancedPipeline; }
  VkPipelineLayout getInstancedLayout() const {
    return instancedPipelineLayout;
  }
  VkPipeline getDeferredPipeline() const { return deferredPipeline; }
  VkPipelineLayout getDeferredLayout() const { return deferredPipelineLayout; }
  VkPipeline getSecondPipeline() const { return secondPipeline; }
  VkPipelineLayout getSecondLayout() const { return secondPipelineLayout; }
  VkPipeline getShadowPipeline() const { return shadowPipeline; }
  VkPipelineLayout getShadowLayout() const { return shadowPipelineLayout; }

private:
  VkPipeline graphicsPipeline = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline instancedPipeline = VK_NULL_HANDLE;
  VkPipelineLayout instancedPipelineLayout = VK_NULL_HANDLE;
  VkPipeline deferredPipeline = VK_NULL_HANDLE;
  VkPipelineLayout deferredPipelineLayout = VK_NULL_HANDLE;
  VkPipeline secondPipeline = VK_NULL_HANDLE;
  VkPipelineLayout secondPipelineLayout = VK_NULL_HANDLE;
  VkPipeline shadowPipeline = VK_NULL_HANDLE;
  VkPipelineLayout shadowPipelineLayout = VK_NULL_HANDLE;
  VkPipelineCache pipelineCache = VK_NULL_HANDLE;

  VkShaderModule createShaderModule(VkDevice device,
                                    const std::vector<char> &code);
};
