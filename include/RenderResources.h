#pragma once

#include "Utilities.h"

#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

struct RenderQueueSharingInfo {
  VkSharingMode mode = VK_SHARING_MODE_EXCLUSIVE;
  uint32_t familyCount = 0;
  std::array<uint32_t, 2> families{};

  bool isConcurrent() const { return mode == VK_SHARING_MODE_CONCURRENT; }
};

RenderQueueSharingInfo
renderGraphicsComputeSharing(const QueueFamilyIndices &indices);
void applyRenderQueueSharing(VkImageCreateInfo &imageCI,
                             const RenderQueueSharingInfo &sharing);
VkImageAspectFlags renderImageAspectMask(VkFormat format);

struct FrameRenderTargets {
  VkImage swapchainImage = VK_NULL_HANDLE;
  VkImage colorBuffer = VK_NULL_HANDLE;
  VkImage gBuffer0 = VK_NULL_HANDLE;
  VkImage gBuffer1 = VK_NULL_HANDLE;
  VkImage gBuffer2 = VK_NULL_HANDLE;
  VkImage gBufferDepth = VK_NULL_HANDLE;
  VkImage lit = VK_NULL_HANDLE;
};

struct RenderImageDesc {
  std::string name;
  VkImage image = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkImageAspectFlags aspectMask = 0;
  uint32_t mipLevels = 1;
  uint32_t arrayLayers = 1;
  VkImageLayout initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  uint32_t queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
};

struct RenderImageState {
  std::string name;
  VkImage image = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkImageSubresourceRange range{};
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkPipelineStageFlags lastStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  VkAccessFlags lastAccess = 0;
  VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  uint32_t queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
};

class RenderResources {
public:
  void clear();

  void resizeFrames(size_t count);
  size_t frameCount() const { return frames.size(); }
  FrameRenderTargets &frame(size_t index);
  const FrameRenderTargets &frame(size_t index) const;

  void registerImage(const RenderImageDesc &desc);
  void unregisterImage(VkImage image);
  bool hasImage(VkImage image) const;
  const RenderImageState *findImage(VkImage image) const;

  void noteLayout(VkImage image, VkImageLayout layout,
                  VkPipelineStageFlags stage, VkAccessFlags access);

  void transition(VkCommandBuffer cmd, VkImage image, VkImageLayout newLayout,
                  VkPipelineStageFlags srcStage,
                  VkPipelineStageFlags dstStage, VkAccessFlags srcAccess,
                  VkAccessFlags dstAccess,
                  const VkImageSubresourceRange *rangeOverride = nullptr,
                  uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED);

  std::vector<RenderImageState> snapshot() const;

private:
  std::vector<FrameRenderTargets> frames;
  std::unordered_map<VkImage, RenderImageState> images;
};
