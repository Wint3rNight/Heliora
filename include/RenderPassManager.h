#pragma once

#include <vulkan/vulkan.h>

class RenderPassManager {
public:
  RenderPassManager() = default;
  ~RenderPassManager() = default;

  // Non-copyable
  RenderPassManager(const RenderPassManager &) = delete;
  RenderPassManager &operator=(const RenderPassManager &) = delete;

  void createRenderPass(VkDevice device, VkFormat swapchainFormat,
                        VkFormat colorFormat, VkFormat depthFormat);
  void cleanup(VkDevice device);

  VkRenderPass getRenderPass() const { return renderPass; }

private:
  VkRenderPass renderPass = VK_NULL_HANDLE;
};
