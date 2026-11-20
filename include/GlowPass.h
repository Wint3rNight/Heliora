#pragma once

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include <vector>

class Mesh;
class VulkanDevice;

// A mesh drawn again with the powers' glow: a see-through shell, brighter
// where its surface turns away from the camera, slow noise flowing over it.
struct GlowDraw {
  const Mesh *mesh = nullptr;
  glm::mat4 model{1.0f};
  glm::mat4 normal{1.0f};
  glm::vec3 colour{1.0f}; // HDR
  float strength = 1.0f;
  // Light over the whole surface too, not only its rim: a piece's flat face
  // turned to the camera has no rim, so a target needs this to show.
  float floor = 0.0f;
};

// The powers' glow (step 1.4a; its light strips go through EffectsPass),
// drawn in the lit pass's transparent subpass: before bloom, so it glows,
// depth tested but not writing depth, added on top (no sorting).
class GlowPass {
public:
  // sceneLayout: the camera's set (set 0 of the meshes' pipelines).
  void create(const VulkanDevice &device, VkRenderPass litPass, VkDescriptorSetLayout sceneLayout,
              VkPipelineCache cache);
  void cleanup(VkDevice device);
  // Inside the lit pass's subpass 1. time drives the flowing noise.
  void record(VkCommandBuffer cmd, VkDescriptorSet sceneSet, const VkViewport &viewport,
              const VkRect2D &scissor, const std::vector<GlowDraw> &glows, float time) const;

private:
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
};
