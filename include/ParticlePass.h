#pragma once

#include "Particles.h"
#include "Utilities.h"

#include <glm/glm.hpp>
#include <vector>

class VulkanDevice;

// The particles (step 1.4a, step 5): up to kCapacity at once in a ring of a
// GPU buffer. Each frame, before the lit pass, one compute dispatch
// (Shaders/particles.comp) writes the frame's new ones into the ring's next
// slots and moves the rest: drag, their share of gravity, and a curl field
// that swirls them without bunching them up. They're drawn in the lit pass's
// transparent subpass as soft camera-facing sprites, added on top (no
// sorting), before bloom: only the slots that can still be alive (LiveRing).
// When none are alive nothing is dispatched or drawn.
class ParticlePass {
public:
  static constexpr uint32_t kCapacity = 65536;
  static constexpr uint32_t kMostSpawns = 8192; // a frame; more are dropped

  void create(const VulkanDevice &device, VkRenderPass litPass, VkDescriptorSetLayout sceneLayout,
              VkPipelineCache cache);
  void cleanup(VkDevice device);

  // Outside a render pass, before the lit pass: this frame's new particles
  // in, everything moved on by dt (0 holds them still), drag pulling each
  // toward the wind.
  void simulate(VkCommandBuffer cmd, uint32_t imageIndex, const std::vector<ParticleSpawn> &spawns,
                float dt, float time, glm::vec3 gravity, glm::vec3 wind);
  // Every particle gone (Stop, a new scene): the ring is cleared again before
  // the next simulate reads it.
  void reset();
  // In the lit pass's subpass 1, after simulate this frame.
  void draw(VkCommandBuffer cmd, uint32_t imageIndex, VkDescriptorSet sceneSet,
            const VkViewport &viewport, const VkRect2D &scissor) const;

private:
  void ensureImage(uint32_t imageIndex);

  const VulkanDevice *device = nullptr;
  VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
  VkDescriptorPool pool = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> sets;          // per swap image
  std::vector<AllocatedBuffer> spawnBuffers;  // per swap image, mapped
  AllocatedBuffer particles;                  // the ring, on the GPU
  VkPipelineLayout simLayout = VK_NULL_HANDLE;
  VkPipeline simPipeline = VK_NULL_HANDLE;
  VkPipelineLayout drawLayout = VK_NULL_HANDLE;
  VkPipeline drawPipeline = VK_NULL_HANDLE;
  LiveRing ring{kCapacity}; // its head, and which slots can still be alive (drawn)
  bool cleared = false;   // the ring zeroed (every slot dead) on the GPU
  float aliveFor = 0.0f;  // s until the longest-lived particle dies
};
