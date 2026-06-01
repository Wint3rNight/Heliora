#pragma once

#include <array>
#include <vector>
#include <vulkan/vulkan.h>

inline VkPipelineStageFlags2 vkSyncStage2(VkPipelineStageFlags flags) {
  return static_cast<VkPipelineStageFlags2>(flags);
}

inline VkAccessFlags2 vkSyncAccess2(VkAccessFlags flags) {
  return static_cast<VkAccessFlags2>(flags);
}

inline VkImageMemoryBarrier2
vkSyncImageBarrier2(const VkImageMemoryBarrier &barrier,
                    VkPipelineStageFlags srcStage,
                    VkPipelineStageFlags dstStage) {
  VkImageMemoryBarrier2 out{};
  out.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
  out.srcStageMask = vkSyncStage2(srcStage);
  out.srcAccessMask = vkSyncAccess2(barrier.srcAccessMask);
  out.dstStageMask = vkSyncStage2(dstStage);
  out.dstAccessMask = vkSyncAccess2(barrier.dstAccessMask);
  out.oldLayout = barrier.oldLayout;
  out.newLayout = barrier.newLayout;
  out.srcQueueFamilyIndex = barrier.srcQueueFamilyIndex;
  out.dstQueueFamilyIndex = barrier.dstQueueFamilyIndex;
  out.image = barrier.image;
  out.subresourceRange = barrier.subresourceRange;
  return out;
}

inline VkBufferMemoryBarrier2
vkSyncBufferBarrier2(const VkBufferMemoryBarrier &barrier,
                     VkPipelineStageFlags srcStage,
                     VkPipelineStageFlags dstStage) {
  VkBufferMemoryBarrier2 out{};
  out.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
  out.srcStageMask = vkSyncStage2(srcStage);
  out.srcAccessMask = vkSyncAccess2(barrier.srcAccessMask);
  out.dstStageMask = vkSyncStage2(dstStage);
  out.dstAccessMask = vkSyncAccess2(barrier.dstAccessMask);
  out.srcQueueFamilyIndex = barrier.srcQueueFamilyIndex;
  out.dstQueueFamilyIndex = barrier.dstQueueFamilyIndex;
  out.buffer = barrier.buffer;
  out.offset = barrier.offset;
  out.size = barrier.size;
  return out;
}

inline void recordImageBarriers2(VkCommandBuffer cmd,
                                 VkPipelineStageFlags srcStage,
                                 VkPipelineStageFlags dstStage,
                                 uint32_t barrierCount,
                                 const VkImageMemoryBarrier *barriers) {
  if (barrierCount == 0)
    return;

  std::array<VkImageMemoryBarrier2, 16> stackBarriers{};
  std::vector<VkImageMemoryBarrier2> heapBarriers;
  VkImageMemoryBarrier2 *converted = stackBarriers.data();
  if (barrierCount > stackBarriers.size()) {
    heapBarriers.resize(barrierCount);
    converted = heapBarriers.data();
  }

  for (uint32_t i = 0; i < barrierCount; ++i)
    converted[i] = vkSyncImageBarrier2(barriers[i], srcStage, dstStage);

  VkDependencyInfo dependency{};
  dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
  dependency.imageMemoryBarrierCount = barrierCount;
  dependency.pImageMemoryBarriers = converted;
  vkCmdPipelineBarrier2(cmd, &dependency);
}

inline void recordImageBarrier2(VkCommandBuffer cmd,
                                VkPipelineStageFlags srcStage,
                                VkPipelineStageFlags dstStage,
                                const VkImageMemoryBarrier &barrier) {
  const VkImageMemoryBarrier2 converted =
      vkSyncImageBarrier2(barrier, srcStage, dstStage);

  VkDependencyInfo dependency{};
  dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
  dependency.imageMemoryBarrierCount = 1;
  dependency.pImageMemoryBarriers = &converted;
  vkCmdPipelineBarrier2(cmd, &dependency);
}

inline void recordBufferBarriers2(VkCommandBuffer cmd,
                                  VkPipelineStageFlags srcStage,
                                  VkPipelineStageFlags dstStage,
                                  uint32_t barrierCount,
                                  const VkBufferMemoryBarrier *barriers) {
  if (barrierCount == 0)
    return;

  std::array<VkBufferMemoryBarrier2, 16> stackBarriers{};
  std::vector<VkBufferMemoryBarrier2> heapBarriers;
  VkBufferMemoryBarrier2 *converted = stackBarriers.data();
  if (barrierCount > stackBarriers.size()) {
    heapBarriers.resize(barrierCount);
    converted = heapBarriers.data();
  }

  for (uint32_t i = 0; i < barrierCount; ++i)
    converted[i] = vkSyncBufferBarrier2(barriers[i], srcStage, dstStage);

  VkDependencyInfo dependency{};
  dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
  dependency.bufferMemoryBarrierCount = barrierCount;
  dependency.pBufferMemoryBarriers = converted;
  vkCmdPipelineBarrier2(cmd, &dependency);
}

inline void recordBufferBarrier2(VkCommandBuffer cmd,
                                 VkPipelineStageFlags srcStage,
                                 VkPipelineStageFlags dstStage,
                                 const VkBufferMemoryBarrier &barrier) {
  const VkBufferMemoryBarrier2 converted =
      vkSyncBufferBarrier2(barrier, srcStage, dstStage);

  VkDependencyInfo dependency{};
  dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
  dependency.bufferMemoryBarrierCount = 1;
  dependency.pBufferMemoryBarriers = &converted;
  vkCmdPipelineBarrier2(cmd, &dependency);
}

inline void recordMemoryBarrier2(VkCommandBuffer cmd,
                                 VkPipelineStageFlags srcStage,
                                 VkPipelineStageFlags dstStage,
                                 VkAccessFlags srcAccess,
                                 VkAccessFlags dstAccess) {
  VkMemoryBarrier2 barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
  barrier.srcStageMask = vkSyncStage2(srcStage);
  barrier.srcAccessMask = vkSyncAccess2(srcAccess);
  barrier.dstStageMask = vkSyncStage2(dstStage);
  barrier.dstAccessMask = vkSyncAccess2(dstAccess);

  VkDependencyInfo dependency{};
  dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
  dependency.memoryBarrierCount = 1;
  dependency.pMemoryBarriers = &barrier;
  vkCmdPipelineBarrier2(cmd, &dependency);
}
