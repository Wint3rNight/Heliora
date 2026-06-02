#pragma once

#include <vulkan/vulkan.h>

class VulkanPipelineCache {
public:
  VulkanPipelineCache() = default;
  ~VulkanPipelineCache() = default;

  VulkanPipelineCache(const VulkanPipelineCache &) = delete;
  VulkanPipelineCache &operator=(const VulkanPipelineCache &) = delete;

  void init(VkDevice device);
  void cleanup(VkDevice device);

  VkPipelineCache get() const { return cache; }

private:
  VkPipelineCache cache = VK_NULL_HANDLE;
};
