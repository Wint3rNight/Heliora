#include "DescriptorManager.h"
#include "Utilities.h"
#include "VulkanSwapchain.h"

#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

void DescriptorManager::init(VkDevice device, VmaAllocator allocator,
                             size_t swapchainImageCount) {
  createDescriptorSetLayout(device);
  createPushConstantRange();
  createUniformBuffers(allocator, swapchainImageCount);
  createDescriptorPool(device, swapchainImageCount);
  createDescriptorSets(device, swapchainImageCount);
}

void DescriptorManager::cleanup(VkDevice device, VmaAllocator /*allocator*/,
                                size_t /*swapchainImageCount*/) {
  vkDestroyDescriptorPool(device, inputDescriptorPool, nullptr);
  vkDestroyDescriptorPool(device, gBufferDescriptorPool, nullptr);
  vkDestroyDescriptorPool(device, samplerDescriptorPool, nullptr);
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);

  vkDestroyDescriptorSetLayout(device, inputSetLayout, nullptr);
  vkDestroyDescriptorSetLayout(device, gBufferSetLayout, nullptr);
  vkDestroyDescriptorSetLayout(device, samplerSetLayout, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

  vpUniformBuffers.clear();
}

void DescriptorManager::updateUniformBuffer(VmaAllocator allocator, size_t idx,
                                            const void *data, size_t size) {
  void *mapped;
  vmaMapMemory(allocator, vpUniformBuffers[idx].getAllocation(), &mapped);
  memcpy(mapped, data, size);
  vmaUnmapMemory(allocator, vpUniformBuffers[idx].getAllocation());
}

int DescriptorManager::createTextureDescriptor(
    VkDevice device, VkImageView albedo, VkImageView normal,
    VkImageView metallic, VkImageView roughness, VkImageView ao,
    VkSampler sampler) {
  VkDescriptorSet set;
  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = samplerDescriptorPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &samplerSetLayout;
  if (vkAllocateDescriptorSets(device, &allocInfo, &set) != VK_SUCCESS)
    throw std::runtime_error("Failed to allocate PBR material descriptor set");

  VkDescriptorImageInfo imgs[5] = {};
  VkImageView views[5] = {albedo, normal, metallic, roughness, ao};
  for (int i = 0; i < 5; ++i) {
    imgs[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imgs[i].imageView = views[i];
    imgs[i].sampler = sampler;
  }

  VkWriteDescriptorSet writes[5] = {};
  for (int i = 0; i < 5; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = set;
    writes[i].dstBinding = static_cast<uint32_t>(i);
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[i].descriptorCount = 1;
    writes[i].pImageInfo = &imgs[i];
  }
  vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);

  samplerDescriptorSets.push_back(set);
  return static_cast<int>(samplerDescriptorSets.size() - 1);
}

void DescriptorManager::updateShadowMapDescriptor(VkDevice device,
                                                  VkImageView shadowView,
                                                  VkImageView pointShadowView,
                                                  VkSampler sampler) {
  VkDescriptorImageInfo shadowInfo = {};
  shadowInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
  shadowInfo.imageView = shadowView;
  shadowInfo.sampler = sampler;

  VkDescriptorImageInfo pointInfo = shadowInfo;
  pointInfo.imageView = pointShadowView;

  std::vector<VkWriteDescriptorSet> writes;
  writes.reserve(descriptorSets.size() * 2);
  for (VkDescriptorSet s : descriptorSets) {
    VkWriteDescriptorSet w = {};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = s;
    w.dstBinding = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.descriptorCount = 1;
    w.pImageInfo = &shadowInfo;
    writes.push_back(w);

    w.dstBinding = 2;
    w.pImageInfo = &pointInfo;
    writes.push_back(w);
  }
  vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()),
                         writes.data(), 0, nullptr);
}

void DescriptorManager::updateIblDescriptors(
    VkDevice device, VkImageView irradianceView, VkSampler iblSampler,
    VkImageView prefilteredView, VkImageView brdfLutView,
    VkImageView ssaoNoiseView, VkSampler noiseSampler, VkImageView skyboxView) {
  // Bindings 3-7 in every VP descriptor set
  VkDescriptorImageInfo irrInfo = {iblSampler, irradianceView,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkDescriptorImageInfo prefInfo = {iblSampler, prefilteredView,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkDescriptorImageInfo brdfInfo = {iblSampler, brdfLutView,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkDescriptorImageInfo noiseInfo = {noiseSampler, ssaoNoiseView,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkDescriptorImageInfo skyboxInfo = {iblSampler, skyboxView,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

  std::vector<VkWriteDescriptorSet> writes;
  writes.reserve(descriptorSets.size() * 5);
  for (VkDescriptorSet s : descriptorSets) {
    VkDescriptorImageInfo *infos[5] = {&irrInfo, &prefInfo, &brdfInfo,
                                       &noiseInfo, &skyboxInfo};
    for (int b = 0; b < 5; ++b) {
      VkWriteDescriptorSet w = {};
      w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w.dstSet = s;
      w.dstBinding = static_cast<uint32_t>(3 + b);
      w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      w.descriptorCount = 1;
      w.pImageInfo = infos[b];
      writes.push_back(w);
    }
  }
  vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()),
                         writes.data(), 0, nullptr);
}

void DescriptorManager::recreateGBufferSets(
    VkDevice device, const std::vector<VkImageView> &gb0,
    const std::vector<VkImageView> &gb1, const std::vector<VkImageView> &gb2,
    const std::vector<VkImageView> &gbDepth, VkSampler sampler) {
  vkResetDescriptorPool(device, gBufferDescriptorPool, 0);

  size_t count = gb0.size();
  gBufferDescriptorSets.resize(count);
  std::vector<VkDescriptorSetLayout> layouts(count, gBufferSetLayout);

  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = gBufferDescriptorPool;
  allocInfo.descriptorSetCount = static_cast<uint32_t>(count);
  allocInfo.pSetLayouts = layouts.data();
  if (vkAllocateDescriptorSets(device, &allocInfo,
                               gBufferDescriptorSets.data()) != VK_SUCCESS)
    throw std::runtime_error("Failed to allocate G-buffer descriptor sets");

  for (size_t i = 0; i < count; ++i) {
    VkDescriptorImageInfo imgs[4] = {
        {sampler, gb0[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {sampler, gb1[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {sampler, gb2[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {sampler, gbDepth[i], VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL},
    };
    VkWriteDescriptorSet writes[4] = {};
    for (int b = 0; b < 4; ++b) {
      writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[b].dstSet = gBufferDescriptorSets[i];
      writes[b].dstBinding = static_cast<uint32_t>(b);
      writes[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      writes[b].descriptorCount = 1;
      writes[b].pImageInfo = &imgs[b];
    }
    vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
  }
}

void DescriptorManager::recreateInputSets(VkDevice device,
                                          const VulkanSwapchain &swapchain) {
  vkResetDescriptorPool(device, inputDescriptorPool, 0);
  createInputDescriptorSets(device, swapchain);
}

// ---------------------------------------------------------------------------
// Layout creation
// ---------------------------------------------------------------------------

void DescriptorManager::createDescriptorSetLayout(VkDevice device) {
  // --- VP / scene layout (set 0): 8 bindings ---
  VkDescriptorSetLayoutBinding bindings[8] = {};

  // 0: SceneUBO
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  // 1..7: samplers
  for (int i = 1; i <= 7; ++i) {
    bindings[i].binding = static_cast<uint32_t>(i);
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  }

  VkDescriptorSetLayoutCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  ci.bindingCount = 8;
  ci.pBindings = bindings;
  if (vkCreateDescriptorSetLayout(device, &ci, nullptr, &descriptorSetLayout) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create VP descriptor set layout");

  // --- PBR material sampler layout (set 1, geometry pass): 5 bindings ---
  VkDescriptorSetLayoutBinding samplerBindings[5] = {};
  for (int i = 0; i < 5; ++i) {
    samplerBindings[i].binding = static_cast<uint32_t>(i);
    samplerBindings[i].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerBindings[i].descriptorCount = 1;
    samplerBindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  }
  VkDescriptorSetLayoutCreateInfo samplerCI = {};
  samplerCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  samplerCI.bindingCount = 5;
  samplerCI.pBindings = samplerBindings;
  if (vkCreateDescriptorSetLayout(device, &samplerCI, nullptr,
                                  &samplerSetLayout) != VK_SUCCESS)
    throw std::runtime_error(
        "Failed to create PBR sampler descriptor set layout");

  // --- G-buffer sampler layout (set 1, deferred pass): 4 bindings ---
  VkDescriptorSetLayoutBinding gbBindings[4] = {};
  for (int i = 0; i < 4; ++i) {
    gbBindings[i].binding = static_cast<uint32_t>(i);
    gbBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    gbBindings[i].descriptorCount = 1;
    gbBindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  }
  VkDescriptorSetLayoutCreateInfo gbCI = {};
  gbCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  gbCI.bindingCount = 4;
  gbCI.pBindings = gbBindings;
  if (vkCreateDescriptorSetLayout(device, &gbCI, nullptr, &gBufferSetLayout) !=
      VK_SUCCESS)
    throw std::runtime_error(
        "Failed to create G-buffer sampler descriptor set layout");

  // --- Input attachment layout (set 0, tone-mapping subpass): 1 binding ---
  VkDescriptorSetLayoutBinding colorInput = {};
  colorInput.binding = 0;
  colorInput.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
  colorInput.descriptorCount = 1;
  colorInput.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo inputCI = {};
  inputCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  inputCI.bindingCount = 1;
  inputCI.pBindings = &colorInput;
  if (vkCreateDescriptorSetLayout(device, &inputCI, nullptr, &inputSetLayout) !=
      VK_SUCCESS)
    throw std::runtime_error(
        "Failed to create input attachment descriptor set layout");
}

// ---------------------------------------------------------------------------
// Push constant range
// ---------------------------------------------------------------------------

void DescriptorManager::createPushConstantRange() {
  pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = sizeof(ModelPushConstants);
}

// ---------------------------------------------------------------------------
// Uniform buffer creation
// ---------------------------------------------------------------------------

void DescriptorManager::createUniformBuffers(VmaAllocator allocator,
                                             size_t count) {
  vpUniformBuffers.resize(count);
  for (size_t i = 0; i < count; ++i) {
    createBuffer(allocator, sizeof(SceneUniformBuffer),
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT,
                 &vpUniformBuffers[i]);
  }
}

// ---------------------------------------------------------------------------
// Descriptor pool creation
// ---------------------------------------------------------------------------

void DescriptorManager::createDescriptorPool(VkDevice device, size_t count) {
  // VP pool: 1 UBO + 7 samplers per frame
  VkDescriptorPoolSize vpSizes[2] = {};
  vpSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  vpSizes[0].descriptorCount = static_cast<uint32_t>(count);
  vpSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  vpSizes[1].descriptorCount = static_cast<uint32_t>(count * 7);

  VkDescriptorPoolCreateInfo vpCI = {};
  vpCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  vpCI.maxSets = static_cast<uint32_t>(count);
  vpCI.poolSizeCount = 2;
  vpCI.pPoolSizes = vpSizes;
  if (vkCreateDescriptorPool(device, &vpCI, nullptr, &descriptorPool) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create VP descriptor pool");

  // Material sampler pool: 5 samplers per material, up to MAX_OBJECTS
  VkDescriptorPoolSize samplerSize = {};
  samplerSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  samplerSize.descriptorCount = MAX_OBJECTS * 5;

  VkDescriptorPoolCreateInfo samplerCI = {};
  samplerCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  samplerCI.maxSets = MAX_OBJECTS;
  samplerCI.poolSizeCount = 1;
  samplerCI.pPoolSizes = &samplerSize;
  if (vkCreateDescriptorPool(device, &samplerCI, nullptr,
                             &samplerDescriptorPool) != VK_SUCCESS)
    throw std::runtime_error(
        "Failed to create material sampler descriptor pool");

  // G-buffer sampler pool: 4 samplers per frame
  VkDescriptorPoolSize gbSize = {};
  gbSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  gbSize.descriptorCount = static_cast<uint32_t>(count * 4);

  VkDescriptorPoolCreateInfo gbCI = {};
  gbCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  gbCI.maxSets = static_cast<uint32_t>(count);
  gbCI.poolSizeCount = 1;
  gbCI.pPoolSizes = &gbSize;
  if (vkCreateDescriptorPool(device, &gbCI, nullptr, &gBufferDescriptorPool) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create G-buffer descriptor pool");

  // Input attachment pool: 1 input attachment per frame
  VkDescriptorPoolSize inputSize = {};
  inputSize.type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
  inputSize.descriptorCount = static_cast<uint32_t>(count);

  VkDescriptorPoolCreateInfo inputCI = {};
  inputCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  inputCI.maxSets = static_cast<uint32_t>(count);
  inputCI.poolSizeCount = 1;
  inputCI.pPoolSizes = &inputSize;
  if (vkCreateDescriptorPool(device, &inputCI, nullptr, &inputDescriptorPool) !=
      VK_SUCCESS)
    throw std::runtime_error(
        "Failed to create input attachment descriptor pool");
}

// ---------------------------------------------------------------------------
// Descriptor set creation
// ---------------------------------------------------------------------------

void DescriptorManager::createDescriptorSets(VkDevice device, size_t count) {
  descriptorSets.resize(count);
  std::vector<VkDescriptorSetLayout> layouts(count, descriptorSetLayout);

  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descriptorPool;
  allocInfo.descriptorSetCount = static_cast<uint32_t>(count);
  allocInfo.pSetLayouts = layouts.data();
  if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to allocate VP descriptor sets");

  for (size_t i = 0; i < count; ++i) {
    VkDescriptorBufferInfo bufInfo = {};
    bufInfo.buffer = vpUniformBuffers[i].get();
    bufInfo.offset = 0;
    bufInfo.range = sizeof(SceneUniformBuffer);

    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSets[i];
    write.dstBinding = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufInfo;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
  }
}

void DescriptorManager::createInputDescriptorSets(
    VkDevice device, const VulkanSwapchain &swapchain) {
  size_t count = swapchain.getImageCount();
  inputDescriptorSets.resize(count);
  std::vector<VkDescriptorSetLayout> layouts(count, inputSetLayout);

  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = inputDescriptorPool;
  allocInfo.descriptorSetCount = static_cast<uint32_t>(count);
  allocInfo.pSetLayouts = layouts.data();
  if (vkAllocateDescriptorSets(device, &allocInfo,
                               inputDescriptorSets.data()) != VK_SUCCESS)
    throw std::runtime_error(
        "Failed to allocate input attachment descriptor sets");

  for (size_t i = 0; i < count; ++i) {
    VkDescriptorImageInfo colorInfo = {};
    colorInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    colorInfo.imageView = swapchain.getColorBufferView(i);
    colorInfo.sampler = VK_NULL_HANDLE;

    VkWriteDescriptorSet w = {};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = inputDescriptorSets[i];
    w.dstBinding = 0;
    w.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    w.descriptorCount = 1;
    w.pImageInfo = &colorInfo;
    vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
  }
}
