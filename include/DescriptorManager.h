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
  VkDescriptorSetLayout getGBufferLayout() const { return gBufferSetLayout; }
  VkDescriptorSetLayout getInputLayout() const { return inputSetLayout; }
  VkPushConstantRange getPushConstantRange() const { return pushConstantRange; }

  // --- Set accessors ---
  VkDescriptorSet getVPSet(size_t i) const { return descriptorSets[i]; }
  VkDescriptorSet getSamplerSet(size_t i) const {
    return samplerDescriptorSets[i];
  }
  VkDescriptorSet getGBufferSet(size_t i) const {
    return gBufferDescriptorSets[i];
  }
  VkDescriptorSet getInputSet(size_t i) const { return inputDescriptorSets[i]; }

  // --- Data updates ---
  void updateUniformBuffer(VmaAllocator allocator, size_t imageIndex,
                           const void *data, size_t size);

  // Creates a 5-texture PBR material descriptor set.
  int createTextureDescriptor(VkDevice device, VkImageView albedo,
                              VkImageView normal, VkImageView metallic,
                              VkImageView roughness, VkImageView ao,
                              VkSampler sampler);

  // Writes shadow maps into every VP descriptor set (binding 1, 2).
  void updateShadowMapDescriptor(VkDevice device, VkImageView shadowView,
                                 VkImageView pointShadowView,
                                 VkSampler sampler);

  // Writes IBL + SSAO noise + skybox into every VP descriptor set (bindings
  // 3-7).
  void updateIblDescriptors(VkDevice device, VkImageView irradianceView,
                            VkSampler iblSampler, VkImageView prefilteredView,
                            VkImageView brdfLutView, VkImageView ssaoNoiseView,
                            VkSampler noiseSampler, VkImageView skyboxView);

  // (Re)creates per-swapchain-image G-buffer descriptor sets.
  void recreateGBufferSets(VkDevice device,
                           const std::vector<VkImageView> &gb0Views,
                           const std::vector<VkImageView> &gb1Views,
                           const std::vector<VkImageView> &gb2Views,
                           const std::vector<VkImageView> &gbDepthViews,
                           VkSampler sampler);

  // (Re)creates per-swapchain-image input attachment descriptor sets.
  void recreateInputSets(VkDevice device, const VulkanSwapchain &swapchain);

private:
  // --- Layouts ---
  VkDescriptorSetLayout descriptorSetLayout =
      VK_NULL_HANDLE; // VP / scene (set 0)
  VkDescriptorSetLayout samplerSetLayout =
      VK_NULL_HANDLE; // 5 PBR textures (set 1, geometry)
  VkDescriptorSetLayout gBufferSetLayout =
      VK_NULL_HANDLE; // 4 G-buffer samplers (set 1, deferred)
  VkDescriptorSetLayout inputSetLayout =
      VK_NULL_HANDLE; // 1 input attachment (set 0, post-process)
  VkPushConstantRange pushConstantRange = {};

  // --- Pools ---
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  VkDescriptorPool samplerDescriptorPool = VK_NULL_HANDLE;
  VkDescriptorPool gBufferDescriptorPool = VK_NULL_HANDLE;
  VkDescriptorPool inputDescriptorPool = VK_NULL_HANDLE;

  // --- Sets ---
  std::vector<VkDescriptorSet> descriptorSets;        // one per swapchain image
  std::vector<VkDescriptorSet> samplerDescriptorSets; // one per loaded material
  std::vector<VkDescriptorSet> gBufferDescriptorSets; // one per swapchain image
  std::vector<VkDescriptorSet> inputDescriptorSets;   // one per swapchain image

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
