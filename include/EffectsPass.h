#pragma once

#include "EffectGeometry.h"
#include "Utilities.h"

#include <vector>

class VulkanDevice;

// Light effects (tracers, muzzle flashes from step 1.6), drawn in the lit
// pass's transparent subpass after the transparent meshes: before bloom, so
// they glow, depth tested against the scene but not writing depth, added on
// top of what's there (no sorting). The strips are built on the CPU each
// frame (EffectGeometry.h) into a vertex buffer per swap image, like
// DebugLinePass. Nothing is drawn when there are none.
class EffectsPass {
public:
  // sceneLayout: the camera's set (set 0 of the meshes' pipelines).
  void create(const VulkanDevice &device, VkRenderPass litPass, VkDescriptorSetLayout sceneLayout,
              VkPipelineCache cache);
  void cleanup(VkDevice device);

  // Copies this frame's strips into the swap image's buffer. Safe once that
  // image's previous frame has finished.
  void upload(const VulkanDevice &device, uint32_t imageIndex,
              const std::vector<EffectVertex> &strips);
  // Inside the lit pass's subpass 1. time drives the flowing noise.
  void record(VkCommandBuffer cmd, uint32_t imageIndex, VkDescriptorSet sceneSet,
              const VkViewport &viewport, const VkRect2D &scissor, float time) const;

private:
  VkPipelineLayout stripLayout = VK_NULL_HANDLE;
  VkPipeline stripPipeline = VK_NULL_HANDLE;

  std::vector<AllocatedBuffer> vertexBuffers; // per swap image
  std::vector<uint32_t> capacities;           // vertices each buffer holds
  std::vector<uint32_t> vertexCounts;         // vertices uploaded this frame
};
