#include "RenderResources.h"
#include "VulkanSync.h"

#include <stdexcept>
#include <utility>

RenderQueueSharingInfo
renderGraphicsComputeSharing(const QueueFamilyIndices &indices) {
  RenderQueueSharingInfo sharing{};
  if (indices.hasDedicatedCompute()) {
    sharing.mode = VK_SHARING_MODE_CONCURRENT;
    sharing.families = {static_cast<uint32_t>(indices.graphicsFamily),
                        static_cast<uint32_t>(indices.computeFamily)};
    sharing.familyCount = 2;
  }
  return sharing;
}

void applyRenderQueueSharing(VkImageCreateInfo &imageCI,
                             const RenderQueueSharingInfo &sharing) {
  imageCI.sharingMode = sharing.mode;
  if (sharing.isConcurrent()) {
    imageCI.queueFamilyIndexCount = sharing.familyCount;
    imageCI.pQueueFamilyIndices = sharing.families.data();
  }
}

VkImageAspectFlags renderImageAspectMask(VkFormat format) {
  switch (format) {
  case VK_FORMAT_D16_UNORM:
  case VK_FORMAT_X8_D24_UNORM_PACK32:
  case VK_FORMAT_D32_SFLOAT:
    return VK_IMAGE_ASPECT_DEPTH_BIT;
  case VK_FORMAT_S8_UINT:
    return VK_IMAGE_ASPECT_STENCIL_BIT;
  case VK_FORMAT_D16_UNORM_S8_UINT:
  case VK_FORMAT_D24_UNORM_S8_UINT:
  case VK_FORMAT_D32_SFLOAT_S8_UINT:
    return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
  default:
    return VK_IMAGE_ASPECT_COLOR_BIT;
  }
}

void RenderResources::clear() {
  frames.clear();
  images.clear();
}

void RenderResources::resizeFrames(size_t count) {
  frames.assign(count, FrameRenderTargets{});
}

FrameRenderTargets &RenderResources::frame(size_t index) {
  if (index >= frames.size())
    throw std::out_of_range("Render frame target index is out of range");
  return frames[index];
}

const FrameRenderTargets &RenderResources::frame(size_t index) const {
  if (index >= frames.size())
    throw std::out_of_range("Render frame target index is out of range");
  return frames[index];
}

void RenderResources::registerImage(const RenderImageDesc &desc) {
  if (desc.image == VK_NULL_HANDLE)
    return;

  RenderImageState state{};
  state.name = desc.name;
  state.image = desc.image;
  state.format = desc.format;
  state.range.aspectMask =
      desc.aspectMask != 0 ? desc.aspectMask : renderImageAspectMask(desc.format);
  state.range.baseMipLevel = 0;
  state.range.levelCount = desc.mipLevels;
  state.range.baseArrayLayer = 0;
  state.range.layerCount = desc.arrayLayers;
  state.layout = desc.initialLayout;
  state.sharingMode = desc.sharingMode;
  state.queueFamilyIndex = desc.sharingMode == VK_SHARING_MODE_CONCURRENT
                               ? VK_QUEUE_FAMILY_IGNORED
                               : desc.queueFamilyIndex;
  images[desc.image] = std::move(state);
}

void RenderResources::unregisterImage(VkImage image) {
  images.erase(image);
}

bool RenderResources::hasImage(VkImage image) const {
  return images.find(image) != images.end();
}

const RenderImageState *RenderResources::findImage(VkImage image) const {
  auto it = images.find(image);
  return it == images.end() ? nullptr : &it->second;
}

void RenderResources::noteLayout(VkImage image, VkImageLayout layout,
                                 VkPipelineStageFlags stage,
                                 VkAccessFlags access) {
  auto it = images.find(image);
  if (it == images.end())
    return;
  it->second.layout = layout;
  it->second.lastStage = stage;
  it->second.lastAccess = access;
}

void RenderResources::transition(
    VkCommandBuffer cmd, VkImage image, VkImageLayout newLayout,
    VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
    VkAccessFlags srcAccess, VkAccessFlags dstAccess,
    const VkImageSubresourceRange *rangeOverride,
    uint32_t dstQueueFamilyIndex) {
  auto it = images.find(image);
  if (it == images.end())
    throw std::runtime_error("Transition requested for an unregistered image");

  RenderImageState &state = it->second;

  const bool needsQueueTransfer =
      state.sharingMode != VK_SHARING_MODE_CONCURRENT &&
      dstQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED &&
      dstQueueFamilyIndex != state.queueFamilyIndex;
  if (!needsQueueTransfer && state.layout == newLayout &&
      state.lastStage == dstStage && state.lastAccess == dstAccess) {
    return;
  }

  VkImageMemoryBarrier2 barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
  barrier.srcStageMask = vkSyncStage2(srcStage);
  barrier.dstStageMask = vkSyncStage2(dstStage);
  barrier.oldLayout = state.layout;
  barrier.newLayout = newLayout;
  barrier.srcAccessMask = vkSyncAccess2(srcAccess);
  barrier.dstAccessMask = vkSyncAccess2(dstAccess);
  barrier.image = image;
  barrier.subresourceRange = rangeOverride ? *rangeOverride : state.range;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

  if (needsQueueTransfer) {
    barrier.srcQueueFamilyIndex = state.queueFamilyIndex;
    barrier.dstQueueFamilyIndex = dstQueueFamilyIndex;
    state.queueFamilyIndex = dstQueueFamilyIndex;
  }

  VkDependencyInfo dependency{};
  dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
  dependency.imageMemoryBarrierCount = 1;
  dependency.pImageMemoryBarriers = &barrier;
  vkCmdPipelineBarrier2(cmd, &dependency);

  state.layout = newLayout;
  state.lastStage = dstStage;
  state.lastAccess = dstAccess;
}

std::vector<RenderImageState> RenderResources::snapshot() const {
  std::vector<RenderImageState> result;
  result.reserve(images.size());
  for (const auto &entry : images)
    result.push_back(entry.second);
  return result;
}
