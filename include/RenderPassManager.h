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

  // Creates the lit pass: single subpass that runs full PBR+IBL+SSAO+bloom+FXAA
  // and writes the lit HDR color into a sampleable image. SSR is NOT applied
  // here — it samples this image in the next pass.
  void createLitRenderPass(VkDevice device, VkFormat litFormat);

  // Creates the dedicated SSGI pass: single subpass, single color
  // attachment. Runs between G-buffer and lit so lit.frag can sample the
  // resulting screen-space-bounce image with a cross-bilateral filter.
  void createSsgiRenderPass(VkDevice device, VkFormat ssgiFormat);

  // Creates the composition pass: 2 subpasses, 3 attachments.
  //   Subpass 0 (SSR composite — samples litBuffer + G-buffer) → colorBuffer
  //   Subpass 1 (TAA + ACES+gamma) → swapchain (LDR) + history (HDR)
  // historyFormat is the HDR format used for the TAA ping-pong history images
  // (must match VulkanRenderer::litFormat).
  void createRenderPass(VkDevice device, VkFormat swapchainFormat,
                        VkFormat colorFormat, VkFormat historyFormat);

  // Depth-only shadow pass (directional + point, unchanged).
  void createShadowRenderPass(VkDevice device, VkFormat depthFormat);

  // Single-attachment pass that LOADs the swapchain image and transitions to
  // PRESENT_SRC_KHR — used to render ImGui on top of the ACES output.
  void createImGuiRenderPass(VkDevice device, VkFormat swapchainFormat);

  void cleanup(VkDevice device);

  VkRenderPass getGBufferRenderPass() const { return gBufferRenderPass; }
  VkRenderPass getLitRenderPass()     const { return litRenderPass; }
  VkRenderPass getSsgiRenderPass()    const { return ssgiRenderPass; }
  VkRenderPass getRenderPass()        const { return renderPass; }
  VkRenderPass getShadowRenderPass()  const { return shadowRenderPass; }
  VkRenderPass getImGuiRenderPass()   const { return imguiRenderPass; }

private:
  VkRenderPass gBufferRenderPass = VK_NULL_HANDLE;
  VkRenderPass litRenderPass     = VK_NULL_HANDLE;
  VkRenderPass ssgiRenderPass    = VK_NULL_HANDLE;
  VkRenderPass renderPass        = VK_NULL_HANDLE;
  VkRenderPass shadowRenderPass  = VK_NULL_HANDLE;
  VkRenderPass imguiRenderPass   = VK_NULL_HANDLE;
};
