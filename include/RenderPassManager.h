#pragma once

#include <vulkan/vulkan.h>

class RenderPassManager {
public:
  RenderPassManager() = default;
  ~RenderPassManager() = default;

  RenderPassManager(const RenderPassManager &) = delete;
  RenderPassManager &operator=(const RenderPassManager &) = delete;

  // Creates the G-buffer pass: 3 color MRTs + depth, outputs to
  // SHADER_READ_ONLY.
  void createGBufferRenderPass(VkDevice device, VkFormat gb0Format,
                               VkFormat gb1Format, VkFormat gb2Format,
                               VkFormat depthFormat);

  // Creates the composition pass: 2 subpasses.
  //   Subpass 0 (deferred PBR+IBL+SSAO+bloom+FXAA) → colorBuffer
  //   Subpass 1 (ACES+gamma)                        → swapchain
  void createRenderPass(VkDevice device, VkFormat swapchainFormat,
                        VkFormat colorFormat);

  // Depth-only shadow pass (directional + point, unchanged).
  void createShadowRenderPass(VkDevice device, VkFormat depthFormat);

  void cleanup(VkDevice device);

  VkRenderPass getGBufferRenderPass() const { return gBufferRenderPass; }
  VkRenderPass getRenderPass() const { return renderPass; }
  VkRenderPass getShadowRenderPass() const { return shadowRenderPass; }

private:
  VkRenderPass gBufferRenderPass = VK_NULL_HANDLE;
  VkRenderPass renderPass = VK_NULL_HANDLE;
  VkRenderPass shadowRenderPass = VK_NULL_HANDLE;
};
