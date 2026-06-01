#include "CompositePass.h"

#include "VulkanDebug.h"

#include <array>
#include <stdexcept>

void CompositePass::create(VkDevice newDevice, VkRenderPass newRenderPass,
                           VkExtent2D extent,
                           const std::vector<VkImageView> &swapViews,
                           const std::vector<VkImageView> &colorViews,
                           const std::vector<VkImageView> &historyViews) {
  cleanup();
  device = newDevice;
  renderPass = newRenderPass;
  renderExtent = extent;
  swapCount = swapViews.size();

  if (swapViews.size() != colorViews.size())
    throw std::runtime_error("Composite pass swap/color view count mismatch");

  framebuffers.assign(historyViews.size() * swapCount, VK_NULL_HANDLE);

  for (size_t historyIndex = 0; historyIndex < historyViews.size();
       ++historyIndex) {
    for (size_t imageIndex = 0; imageIndex < swapCount; ++imageIndex) {
      std::array<VkImageView, 3> attachments = {
          swapViews[imageIndex], colorViews[imageIndex],
          historyViews[historyIndex]};
      VkFramebufferCreateInfo framebufferCI{};
      framebufferCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      framebufferCI.renderPass = renderPass;
      framebufferCI.attachmentCount =
          static_cast<uint32_t>(attachments.size());
      framebufferCI.pAttachments = attachments.data();
      framebufferCI.width = renderExtent.width;
      framebufferCI.height = renderExtent.height;
      framebufferCI.layers = 1;
      VkFramebuffer framebuffer = VK_NULL_HANDLE;
      if (vkCreateFramebuffer(device, &framebufferCI, nullptr, &framebuffer) !=
          VK_SUCCESS) {
        throw std::runtime_error("Failed to create composite framebuffer");
      }
      framebuffers[framebufferIndex(static_cast<uint32_t>(historyIndex),
                                    static_cast<uint32_t>(imageIndex))] =
          framebuffer;
    }
  }
}

void CompositePass::cleanup() {
  if (device == VK_NULL_HANDLE)
    return;

  for (VkFramebuffer framebuffer : framebuffers) {
    if (framebuffer != VK_NULL_HANDLE)
      vkDestroyFramebuffer(device, framebuffer, nullptr);
  }
  framebuffers.clear();
  renderPass = VK_NULL_HANDLE;
  renderExtent = {};
  swapCount = 0;
  device = VK_NULL_HANDLE;
}

void CompositePass::record(VkCommandBuffer cmd, uint32_t currentImage,
                           uint32_t historyIndex, VkDescriptorSet vpSet,
                           VkDescriptorSet gBufferSet,
                           VkDescriptorSet inputSet, VkDescriptorSet taaSet,
                           VkPipeline deferredPipeline,
                           VkPipelineLayout deferredLayout,
                           VkPipeline secondPipeline,
                           VkPipelineLayout secondLayout) {
  const size_t fbIndex = framebufferIndex(historyIndex, currentImage);
  if (fbIndex >= framebuffers.size() ||
      framebuffers[fbIndex] == VK_NULL_HANDLE) {
    return;
  }

  std::array<VkClearValue, 3> clears{};
  clears[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
  clears[1].color = {0.0f, 0.0f, 0.0f, 1.0f};
  clears[2].color = {0.0f, 0.0f, 0.0f, 1.0f};

  VkRenderPassBeginInfo rpbi{};
  rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rpbi.renderPass = renderPass;
  rpbi.framebuffer = framebuffers[fbIndex];
  rpbi.renderArea = {{0, 0}, renderExtent};
  rpbi.clearValueCount = static_cast<uint32_t>(clears.size());
  rpbi.pClearValues = clears.data();

  VkViewport viewport{};
  viewport.width = static_cast<float>(renderExtent.width);
  viewport.height = static_cast<float>(renderExtent.height);
  viewport.maxDepth = 1.0f;
  VkRect2D scissor = {{0, 0}, renderExtent};

  vkdbgBeginLabel(cmd, "Composition (SSR + AgX Tonemap)", 0.6f, 0.2f, 0.8f);
  vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  vkCmdSetScissor(cmd, 0, 1, &scissor);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, deferredPipeline);
  std::array<VkDescriptorSet, 2> deferredSets = {vpSet, gBufferSet};
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          deferredLayout, 0,
                          static_cast<uint32_t>(deferredSets.size()),
                          deferredSets.data(), 0, nullptr);
  vkCmdDraw(cmd, 3, 1, 0, 0);

  vkCmdNextSubpass(cmd, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, secondPipeline);
  std::array<VkDescriptorSet, 3> secondSets = {vpSet, inputSet, taaSet};
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, secondLayout, 0,
                          static_cast<uint32_t>(secondSets.size()),
                          secondSets.data(), 0, nullptr);
  vkCmdDraw(cmd, 3, 1, 0, 0);

  vkCmdEndRenderPass(cmd);
  vkdbgEndLabel(cmd);
}

size_t CompositePass::framebufferIndex(uint32_t historyIndex,
                                       uint32_t currentImage) const {
  return static_cast<size_t>(historyIndex) * swapCount + currentImage;
}
