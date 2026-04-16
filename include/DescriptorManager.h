#pragma once

#include <vector>
#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include "Utilities.h"

// Forward declarations
class VulkanSwapchain;

class DescriptorManager {
public:
  DescriptorManager() = default;
  ~DescriptorManager() = default;

  // Non-copyable
  DescriptorManager(const DescriptorManager &) = delete;
  DescriptorManager &operator=(const DescriptorManager &) = delete;

  void init(VkDevice device, VmaAllocator allocator,
            size_t swapchainImageCount);
  void cleanup(VkDevice device, VmaAllocator allocator, size_t swapchainImageCount);

  // --- Layout accessors ---
  VkDescriptorSetLayout getVPLayout() const { return descriptorSetLayout; }
  VkDescriptorSetLayout getSamplerLayout() const { return samplerSetLayout; }
  VkDescriptorSetLayout getInputLayout() const { return inputSetLayout; }
  VkPushConstantRange getPushConstantRange() const {
    return pushConstantRange;
  }

  // --- Set accessors ---
  VkDescriptorSet getVPSet(size_t imageIndex) const {
    return descriptorSets[imageIndex];
  }
  VkDescriptorSet getSamplerSet(size_t textureIndex) const {
    return samplerDescriptorSets[textureIndex];
  }
  VkDescriptorSet getInputSet(size_t imageIndex) const {
    return inputDescriptorSets[imageIndex];
  }

  // --- Operations ---
  void updateUniformBuffer(VmaAllocator allocator, size_t imageIndex,
                           const void *data, size_t size);
  int createTextureDescriptor(VkDevice device, VkImageView textureImageView,
                              VkSampler sampler);
  void recreateInputSets(VkDevice device,
                         const VulkanSwapchain &swapchain);

private:
  // --- Layouts ---
  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout samplerSetLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout inputSetLayout = VK_NULL_HANDLE;
  VkPushConstantRange pushConstantRange = {};

  // --- Pools ---
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  VkDescriptorPool samplerDescriptorPool = VK_NULL_HANDLE;
  VkDescriptorPool inputDescriptorPool = VK_NULL_HANDLE;

  // --- Sets ---
  std::vector<VkDescriptorSet> descriptorSets;
  std::vector<VkDescriptorSet> samplerDescriptorSets;
  std::vector<VkDescriptorSet> inputDescriptorSets;

  // --- Uniform buffers ---
  std::vector<AllocatedBuffer> vpUniformBuffers;

  // --- Creation functions ---
  void createDescriptorSetLayout(VkDevice device);
  void createPushConstantRange();
  void createUniformBuffers(VmaAllocator allocator,
                            size_t swapchainImageCount);
  void createDescriptorPool(VkDevice device, size_t swapchainImageCount);
  void createDescriptorSets(VkDevice device, size_t swapchainImageCount);
  void createInputDescriptorSets(VkDevice device,
                                 const VulkanSwapchain &swapchain);
};
