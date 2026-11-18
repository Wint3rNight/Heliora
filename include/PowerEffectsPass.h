#pragma once

#include "PowerEffects.h"
#include "Utilities.h"

#include <glm/glm.hpp>
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

// The powers' light (step 1.4a, step 4), drawn in the lit pass's transparent
// subpass after the transparent meshes: before bloom, so it glows, depth
// tested against the scene but not writing depth, added on top of what's
// there (no sorting). The glow redraws meshes; the strips (ribbons, the
// throw's trail, the burst's cone, the reveal's ring) are built on the CPU
// each frame into a vertex buffer per swap image, like DebugLinePass.
class PowerEffectsPass {
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
              const VkViewport &viewport, const VkRect2D &scissor,
              const std::vector<GlowDraw> &glows, float time) const;

private:
  VkPipelineLayout glowLayout = VK_NULL_HANDLE;
  VkPipeline glowPipeline = VK_NULL_HANDLE;
  VkPipelineLayout stripLayout = VK_NULL_HANDLE;
  VkPipeline stripPipeline = VK_NULL_HANDLE;

  std::vector<AllocatedBuffer> vertexBuffers; // per swap image
  std::vector<uint32_t> capacities;           // vertices each buffer holds
  std::vector<uint32_t> vertexCounts;         // vertices uploaded this frame
};
