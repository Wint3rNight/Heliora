#pragma once

#include "PhysicsWorld.h"
#include "Utilities.h"

#include <glm/glm.hpp>
#include <vector>

class VulkanDevice;

// Draws debug lines (physics wireframes for now) over the finished frame,
// inside the ImGui render pass, so they sit under the panels. That pass has
// no depth attachment, so the fragment shader reads the scene depth as a
// texture: lines behind geometry are drawn faint instead of hidden, so a
// shape behind a wall can still be found.
class DebugLinePass {
public:
  void create(const VulkanDevice &device, VkRenderPass imguiRenderPass,
              VkPipelineCache cache);
  // The scene depth per swap image. Call again after the swapchain changes.
  void setDepthViews(VkDevice device, const std::vector<VkImageView> &views,
                     VkSampler sampler);
  void cleanup(VkDevice device);

  // Copies this frame's lines into the swap image's buffer. Safe once that
  // image's previous frame has finished, which draw() waits for first.
  void upload(const VulkanDevice &device, uint32_t imageIndex,
              const std::vector<PhysicsDebugVertex> &lines);
  // Inside the ImGui render pass. viewProj must be unjittered, or the lines
  // shimmer: they're drawn after TAA, so nothing averages the jitter out.
  void record(VkCommandBuffer cmd, uint32_t imageIndex, VkExtent2D extent,
              const glm::mat4 &viewProj, float nearPlane,
              float farPlane) const;

private:
  VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
  VkDescriptorPool pool = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> depthSets; // per swap image
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;

  std::vector<AllocatedBuffer> vertexBuffers; // per swap image
  std::vector<uint32_t> capacities;           // vertices each buffer holds
  std::vector<uint32_t> vertexCounts;         // vertices uploaded this frame
};
