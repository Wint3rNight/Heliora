#include "SsgiPass.h"

#include "VulkanDebug.h"
#include "VulkanSync.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

void SsgiPass::create(VulkanDevice &newDevice, VkExtent2D fullExtent,
                      VkFormat format, VkRenderPass renderPass,
                      size_t historyCount) {
  cleanup();
  device = &newDevice;
  historyFormat = format;
  targetRenderPass = renderPass;
  renderExtent = {std::max(1u, fullExtent.width / 2),
                  std::max(1u, fullExtent.height / 2)};

  VkDevice dev = device->getLogicalDevice();
  historyImages.clear();
  historyViews.clear();
  framebuffers.assign(historyCount, VK_NULL_HANDLE);
  historyImages.reserve(historyCount);
  historyViews.reserve(historyCount);

  VmaAllocationCreateInfo allocCI{};
  allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  allocCI.flags = RENDER_DEVICE_ALLOCATION_FLAGS;

  for (size_t i = 0; i < historyCount; ++i) {
    VkImageCreateInfo imageCI{};
    imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCI.imageType = VK_IMAGE_TYPE_2D;
    imageCI.extent = {renderExtent.width, renderExtent.height, 1};
    imageCI.mipLevels = 1;
    imageCI.arrayLayers = 1;
    imageCI.format = historyFormat;
    imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage rawImage = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    if (vmaCreateImage(device->getAllocator(), &imageCI, &allocCI, &rawImage,
                       &allocation, nullptr) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create SSGI history image");
    }
    historyImages.emplace_back(device->getAllocator(), rawImage, allocation);

    VkImageViewCreateInfo viewCI{};
    viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image = historyImages.back().get();
    viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format = historyFormat;
    viewCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view = VK_NULL_HANDLE;
    if (vkCreateImageView(dev, &viewCI, nullptr, &view) != VK_SUCCESS)
      throw std::runtime_error("Failed to create SSGI history view");
    historyViews.emplace_back(dev, view);

    VkImageView attachment = view;
    VkFramebufferCreateInfo framebufferCI{};
    framebufferCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferCI.renderPass = targetRenderPass;
    framebufferCI.attachmentCount = 1;
    framebufferCI.pAttachments = &attachment;
    framebufferCI.width = renderExtent.width;
    framebufferCI.height = renderExtent.height;
    framebufferCI.layers = 1;
    if (vkCreateFramebuffer(dev, &framebufferCI, nullptr, &framebuffers[i]) !=
        VK_SUCCESS) {
      throw std::runtime_error("Failed to create SSGI framebuffer");
    }
  }

  VkCommandBuffer cmd =
      beginCommandBuffer(dev, device->getGraphicsCommandPool());
  std::vector<VkImageMemoryBarrier> barriers;
  barriers.reserve(historyImages.size());
  for (const AllocatedImage &image : historyImages) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.get();
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barriers.push_back(barrier);
  }
  if (!barriers.empty()) {
    recordImageBarriers2(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         static_cast<uint32_t>(barriers.size()),
                         barriers.data());
  }
  endAndSubmitCommandBuffer(dev, device->getGraphicsCommandPool(),
                            device->getGraphicsQueue(), cmd);

  VkSamplerCreateInfo samplerCI{};
  samplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerCI.magFilter = VK_FILTER_LINEAR;
  samplerCI.minFilter = VK_FILTER_LINEAR;
  samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerCI.maxLod = 0.0f;
  if (vkCreateSampler(dev, &samplerCI, nullptr, &historySampler) != VK_SUCCESS)
    throw std::runtime_error("Failed to create SSGI sampler");
}

void SsgiPass::cleanup() {
  if (!device)
    return;

  VkDevice dev = device->getLogicalDevice();
  for (VkFramebuffer framebuffer : framebuffers) {
    if (framebuffer != VK_NULL_HANDLE)
      vkDestroyFramebuffer(dev, framebuffer, nullptr);
  }
  if (historySampler != VK_NULL_HANDLE)
    vkDestroySampler(dev, historySampler, nullptr);

  framebuffers.clear();
  historyViews.clear();
  historyImages.clear();
  historySampler = VK_NULL_HANDLE;
  renderExtent = {};
  historyFormat = VK_FORMAT_UNDEFINED;
  targetRenderPass = VK_NULL_HANDLE;
  device = nullptr;
}

void SsgiPass::record(VkCommandBuffer cmd, uint32_t historyIndex,
                      VkDescriptorSet vpSet, VkDescriptorSet gBufferSet,
                      VkDescriptorSet prevSet, VkPipeline pipeline,
                      VkPipelineLayout pipelineLayout) {
  if (historyIndex >= framebuffers.size() || pipeline == VK_NULL_HANDLE)
    return;

  VkViewport viewport{};
  viewport.width = static_cast<float>(renderExtent.width);
  viewport.height = static_cast<float>(renderExtent.height);
  viewport.maxDepth = 1.0f;
  VkRect2D scissor = {{0, 0}, renderExtent};

  VkClearValue clear{};
  clear.color = {0.0f, 0.0f, 0.0f, 0.0f};

  VkRenderPassBeginInfo rpbi{};
  rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rpbi.renderPass = targetRenderPass;
  rpbi.framebuffer = framebuffers[historyIndex];
  rpbi.renderArea = {{0, 0}, renderExtent};
  rpbi.clearValueCount = 1;
  rpbi.pClearValues = &clear;

  vkdbgBeginLabel(cmd, "SSGI (reproject + blend)", 0.95f, 0.55f, 0.20f);
  vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  vkCmdSetScissor(cmd, 0, 1, &scissor);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
  std::array<VkDescriptorSet, 3> sets = {vpSet, gBufferSet, prevSet};
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                          0, static_cast<uint32_t>(sets.size()), sets.data(),
                          0, nullptr);
  vkCmdDraw(cmd, 3, 1, 0, 0);
  vkCmdEndRenderPass(cmd);
  vkdbgEndLabel(cmd);
}

uint32_t SsgiPass::historyIndex(uint32_t frameCounter) const {
  if (historyViews.empty())
    return 0;
  return static_cast<uint32_t>(frameCounter % historyViews.size());
}

std::vector<VkImageView> SsgiPass::views() const {
  std::vector<VkImageView> out;
  out.reserve(historyViews.size());
  for (const ImageViewHandle &view : historyViews)
    out.push_back(view.get());
  return out;
}
