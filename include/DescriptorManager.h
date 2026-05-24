#pragma once

#include "Utilities.h"
#include <glm/glm.hpp>
#include <vector>
#include <vulkan/vulkan.h>

class VulkanSwapchain;

class DescriptorManager {
public:
  DescriptorManager() = default;
  ~DescriptorManager() = default;

  DescriptorManager(const DescriptorManager &) = delete;
  DescriptorManager &operator=(const DescriptorManager &) = delete;

  void init(VkDevice device, VmaAllocator allocator,
            size_t swapchainImageCount);
  void cleanup(VkDevice device, VmaAllocator allocator,
               size_t swapchainImageCount);

  // --- Layout accessors ---
  VkDescriptorSetLayout getVPLayout() const { return descriptorSetLayout; }
  VkDescriptorSetLayout getSamplerLayout() const { return samplerSetLayout; }
  VkDescriptorSetLayout getBindlessLayout() const { return bindlessSetLayout; }
  VkDescriptorSetLayout getGBufferLayout() const { return gBufferSetLayout; }
  VkDescriptorSetLayout getInputLayout() const { return inputSetLayout; }
  // TAA history-prev sampler (binding 0) + G-buffer depth sampler (binding 1).
  // Bound at set 2 of the tonemap pipeline next session. Layout is created
  // in init() but no sets are allocated against it yet — the per-frame
  // ping-pong descriptors will be added in the wire-up step.
  VkDescriptorSetLayout getTaaLayout() const { return taaSetLayout; }
  VkPushConstantRange getPushConstantRange() const { return pushConstantRange; }

  // --- Set accessors ---
  VkDescriptorSet getVPSet(size_t i) const { return descriptorSets[i]; }
  VkDescriptorSet getSamplerSet(size_t i) const {
    return samplerDescriptorSets[i];
  }
  // Phase 7.2 — single global bindless texture array set. Contains up to
  // MAX_BINDLESS_TEXTURES combined-image-samplers at binding 0. Flagged
  // UPDATE_AFTER_BIND so textures can be registered after the set is bound.
  VkDescriptorSet getBindlessSet() const { return bindlessDescriptorSet; }
  VkDescriptorSet getGBufferSet(size_t i) const {
    return gBufferDescriptorSets[i]; // parity 0; Task 5 removes this overload
  }
  VkDescriptorSet getGBufferSet(size_t parity, size_t i) const {
    return gBufferDescriptorSets[parity * gBufferSwapCount + i];
  }
  VkDescriptorSet getInputSet(size_t i) const { return inputDescriptorSets[i]; }
  // TAA set indexed by (parity * swapCount + swapIdx). Binding 0 =
  // historyPrev view (the OTHER ping-pong image, written last frame),
  // binding 1 = G-buffer depth view for this swap index.
  VkDescriptorSet getTaaSet(size_t i) const { return taaDescriptorSets[i]; }

  // --- Data updates ---
  void updateUniformBuffer(VmaAllocator allocator, size_t imageIndex,
                           const void *data, size_t size);

  // Creates a 5-texture PBR material descriptor set (legacy, pre-bindless).
  int createTextureDescriptor(VkDevice device, VkImageView albedo,
                              VkImageView normal, VkImageView metallic,
                              VkImageView roughness, VkImageView ao,
                              VkSampler sampler);

  // Phase 7.2 — write a single texture into the global bindless array at
  // `index`. The set is flagged UPDATE_AFTER_BIND so this can be called
  // at any time (even after the set has been bound in a command buffer
  // that hasn't been submitted yet).
  void registerBindlessTexture(VkDevice device, uint32_t index,
                               VkImageView view, VkSampler sampler);

  // Writes shadow maps into every VP descriptor set (binding 1, 2). CSM uses
  // a compare-enabled sampler (hardware PCF) and the point cube uses a plain
  // sampler — they cannot share a sampler in GLSL.
  void updateShadowMapDescriptor(VkDevice device, VkImageView shadowView,
                                 VkImageView pointShadowView,
                                 VkSampler csmSampler,
                                 VkSampler pointSampler);

  // Writes IBL + SSAO noise + skybox into every VP descriptor set (bindings
  // 3-7).
  void updateIblDescriptors(VkDevice device, VkImageView irradianceView,
                            VkSampler iblSampler, VkImageView prefilteredView,
                            VkImageView brdfLutView, VkImageView ssaoNoiseView,
                            VkSampler noiseSampler, VkImageView skyboxView);

  // (Re)creates G-buffer descriptor sets. Binds gb0/gb1/gb2/depth +
  // the lit-pass output (for SSR composite) + the SSGI bounce image
  // (for lit.frag's cross-bilateral filter).
  // ssgiViews must have exactly 2 entries (ping-pong). Produces
  // 2 * swapCount G-buffer sets indexed (parity * swapCount + i).
  // Per parity P, binding 5 = ssgiViews[P].
  void recreateGBufferSets(VkDevice device,
                           const std::vector<VkImageView> &gb0Views,
                           const std::vector<VkImageView> &gb1Views,
                           const std::vector<VkImageView> &gb2Views,
                           const std::vector<VkImageView> &gbDepthViews,
                           const std::vector<VkImageView> &litViews,
                           const std::vector<VkImageView> &ssgiViews,
                           VkSampler sampler);

  // (Re)creates per-swapchain-image input attachment descriptor sets.
  void recreateInputSets(VkDevice device, const VulkanSwapchain &swapchain);

  // (Re)creates TAA descriptor sets — 2 * swapCount entries.
  // Layout for set s = parity * swapCount + swapIdx:
  //   binding 0 = historyPrevViews[parity] (image written last frame)
  //   binding 1 = gBufferDepthViews[swapIdx]
  //   binding 2 = colorBufferViews[swapIdx]  (HDR pre-tonemap, this frame)
  // All sampled with `sampler` (CLAMP_TO_EDGE recommended).
  void recreateTaaSets(VkDevice device,
                       const std::vector<VkImageView> &historyPrevViews,
                       const std::vector<VkImageView> &gBufferDepthViews,
                       const std::vector<VkImageView> &colorBufferViews,
                       VkSampler sampler);

  // --- SSGI prev-history set 2 accessors ---
  VkDescriptorSetLayout getSsgiPrevLayout() const { return ssgiPrevSetLayout; }
  VkDescriptorSet getSsgiPrevSet(size_t parity) const {
    return ssgiPrevDescriptorSets[parity];
  }
  // ssgiHistoryViews has size 2 (ping-pong). For parity P, the set
  // at index P binds historyViews[(P+1)&1] — last frame's SSGI.
  void recreateSsgiPrevSets(VkDevice device,
                            const std::vector<VkImageView> &ssgiHistoryViews,
                            VkSampler sampler);

private:
  // --- Layouts ---
  VkDescriptorSetLayout descriptorSetLayout =
      VK_NULL_HANDLE; // VP / scene (set 0)
  VkDescriptorSetLayout samplerSetLayout =
      VK_NULL_HANDLE; // 5 PBR textures (set 1, geometry) — legacy
  VkDescriptorSetLayout bindlessSetLayout =
      VK_NULL_HANDLE; // Phase 7.2: N-texture bindless array (set 1, geometry)
  VkDescriptorSetLayout gBufferSetLayout =
      VK_NULL_HANDLE; // 4 G-buffer samplers (set 1, deferred)
  VkDescriptorSetLayout inputSetLayout =
      VK_NULL_HANDLE; // 1 input attachment (set 0, post-process)
  VkDescriptorSetLayout taaSetLayout =
      VK_NULL_HANDLE; // TAA history + depth samplers (set 2, post-process)
  VkDescriptorSetLayout ssgiPrevSetLayout =
      VK_NULL_HANDLE; // set 2 on SSGI pipeline
  VkPushConstantRange pushConstantRange = {};

  // --- Pools ---
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  VkDescriptorPool samplerDescriptorPool = VK_NULL_HANDLE;
  VkDescriptorPool bindlessDescriptorPool = VK_NULL_HANDLE;
  VkDescriptorPool gBufferDescriptorPool = VK_NULL_HANDLE;
  VkDescriptorPool inputDescriptorPool = VK_NULL_HANDLE;
  VkDescriptorPool taaDescriptorPool = VK_NULL_HANDLE;
  VkDescriptorPool ssgiPrevDescriptorPool = VK_NULL_HANDLE;

  // --- Sets ---
  std::vector<VkDescriptorSet> descriptorSets;        // one per swapchain image
  std::vector<VkDescriptorSet> samplerDescriptorSets; // one per loaded material (legacy)
  VkDescriptorSet bindlessDescriptorSet = VK_NULL_HANDLE; // Phase 7.2: single global set
  std::vector<VkDescriptorSet> gBufferDescriptorSets; // 2 * swapCount (parity*swapCount + i)
  // Set by recreateGBufferSets so getGBufferSet(parity, i) can index correctly.
  size_t gBufferSwapCount = 0;
  std::vector<VkDescriptorSet> inputDescriptorSets;   // one per swapchain image
  std::vector<VkDescriptorSet> taaDescriptorSets;     // 2 * swapCount
  std::vector<VkDescriptorSet> ssgiPrevDescriptorSets; // size 2, indexed by parity

  // --- Uniform buffers ---
  std::vector<AllocatedBuffer> vpUniformBuffers;

  // --- Creation helpers ---
  void createDescriptorSetLayout(VkDevice device);
  void createPushConstantRange();
  void createUniformBuffers(VmaAllocator allocator, size_t swapchainImageCount);
  void createDescriptorPool(VkDevice device, size_t swapchainImageCount);
  void createDescriptorSets(VkDevice device, size_t swapchainImageCount);
  void createInputDescriptorSets(VkDevice device,
                                 const VulkanSwapchain &swapchain);
};
