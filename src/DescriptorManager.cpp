#include "DescriptorManager.h"
#include "Utilities.h"
#include "VulkanSwapchain.h"

#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

// Public interface


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
  if (ssgiPrevDescriptorPool != VK_NULL_HANDLE)
    vkDestroyDescriptorPool(device, ssgiPrevDescriptorPool, nullptr);
  if (taaDescriptorPool != VK_NULL_HANDLE)
    vkDestroyDescriptorPool(device, taaDescriptorPool, nullptr);
  vkDestroyDescriptorPool(device, inputDescriptorPool, nullptr);
  vkDestroyDescriptorPool(device, gBufferDescriptorPool, nullptr);
  if (bindlessDescriptorPool != VK_NULL_HANDLE)
    vkDestroyDescriptorPool(device, bindlessDescriptorPool, nullptr);
  vkDestroyDescriptorPool(device, samplerDescriptorPool, nullptr);
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);

  if (ssgiPrevSetLayout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(device, ssgiPrevSetLayout, nullptr);
  vkDestroyDescriptorSetLayout(device, taaSetLayout, nullptr);
  vkDestroyDescriptorSetLayout(device, inputSetLayout, nullptr);
  vkDestroyDescriptorSetLayout(device, gBufferSetLayout, nullptr);
  if (bindlessSetLayout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(device, bindlessSetLayout, nullptr);
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
                                                  VkSampler csmSampler,
                                                  VkSampler pointSampler) {
  VkDescriptorImageInfo shadowInfo = {};
  shadowInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
  shadowInfo.imageView = shadowView;
  shadowInfo.sampler = csmSampler;

  VkDescriptorImageInfo pointInfo = shadowInfo;
  pointInfo.imageView = pointShadowView;
  pointInfo.sampler = pointSampler;

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
    const std::vector<VkImageView> &gbDepth,
    const std::vector<VkImageView> &lit,
    const std::vector<VkImageView> &ssgi, VkSampler sampler) {
  // Produces historyCount * swapCount sets indexed
  // (historyIndex * swapCount + i).
  vkResetDescriptorPool(device, gBufferDescriptorPool, 0);

  size_t swapCount = gb0.size();
  size_t historyCount = ssgi.size();
  gBufferSwapCount = swapCount;
  size_t totalSets = historyCount * swapCount;
  gBufferDescriptorSets.assign(totalSets, VK_NULL_HANDLE);
  std::vector<VkDescriptorSetLayout> layouts(totalSets, gBufferSetLayout);

  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = gBufferDescriptorPool;
  allocInfo.descriptorSetCount = static_cast<uint32_t>(totalSets);
  allocInfo.pSetLayouts = layouts.data();
  if (vkAllocateDescriptorSets(device, &allocInfo,
                               gBufferDescriptorSets.data()) != VK_SUCCESS)
    throw std::runtime_error("Failed to allocate G-buffer descriptor sets");

  for (size_t historyIndex = 0; historyIndex < historyCount; ++historyIndex) {
    for (size_t i = 0; i < swapCount; ++i) {
      VkDescriptorImageInfo imgs[6] = {
          {sampler, gb0[i],     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
          {sampler, gb1[i],     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
          {sampler, gb2[i],     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
          {sampler, gbDepth[i], VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL},
          {sampler, lit[i],     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
          {sampler, ssgi[historyIndex], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
      };
      VkDescriptorSet set = gBufferDescriptorSets[historyIndex * swapCount + i];
      VkWriteDescriptorSet writes[6] = {};
      for (int b = 0; b < 6; ++b) {
        writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[b].dstSet = set;
        writes[b].dstBinding = static_cast<uint32_t>(b);
        writes[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[b].descriptorCount = 1;
        writes[b].pImageInfo = &imgs[b];
      }
      vkUpdateDescriptorSets(device, 6, writes, 0, nullptr);
    }
  }
}

void DescriptorManager::recreateInputSets(VkDevice device,
                                          const VulkanSwapchain &swapchain) {
  vkResetDescriptorPool(device, inputDescriptorPool, 0);
  createInputDescriptorSets(device, swapchain);
}

void DescriptorManager::recreateTaaSets(
    VkDevice device, const std::vector<VkImageView> &historyPrevViews,
    const std::vector<VkImageView> &gBufferDepthViews,
    const std::vector<VkImageView> &colorBufferViews, VkSampler sampler) {
  // historyPrevViews is the TAA history ring.
  // historyCount * swapCount sets, indexed
  // (historyIndex * swapCount + swapIdx).
  size_t historyCount = historyPrevViews.size();
  size_t swapCount = gBufferDepthViews.size();
  size_t totalSets = historyCount * swapCount;

  if (taaDescriptorPool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(device, taaDescriptorPool, nullptr);
    taaDescriptorPool = VK_NULL_HANDLE;
  }
  VkDescriptorPoolSize sz = {};
  sz.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  sz.descriptorCount = static_cast<uint32_t>(totalSets * 3);
  VkDescriptorPoolCreateInfo pci = {};
  pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pci.maxSets = static_cast<uint32_t>(totalSets);
  pci.poolSizeCount = 1;
  pci.pPoolSizes = &sz;
  if (vkCreateDescriptorPool(device, &pci, nullptr, &taaDescriptorPool) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create TAA descriptor pool");

  std::vector<VkDescriptorSetLayout> layouts(totalSets, taaSetLayout);
  taaDescriptorSets.assign(totalSets, VK_NULL_HANDLE);

  VkDescriptorSetAllocateInfo ai = {};
  ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  ai.descriptorPool = taaDescriptorPool;
  ai.descriptorSetCount = static_cast<uint32_t>(totalSets);
  ai.pSetLayouts = layouts.data();
  if (vkAllocateDescriptorSets(device, &ai, taaDescriptorSets.data()) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to allocate TAA descriptor sets");

  for (size_t historyIndex = 0; historyIndex < historyCount; ++historyIndex) {
    VkImageView historyPrev =
        historyPrevViews[(historyIndex + historyCount - 1) % historyCount];
    for (size_t i = 0; i < swapCount; ++i) {
      VkDescriptorImageInfo histInfo = {
          sampler, historyPrev, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      VkDescriptorImageInfo depthInfo = {
          sampler, gBufferDepthViews[i],
          VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
      VkDescriptorImageInfo colorInfo = {
          sampler, colorBufferViews[i],
          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      VkDescriptorSet set = taaDescriptorSets[historyIndex * swapCount + i];
      VkWriteDescriptorSet writes[3] = {};
      writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[0].dstSet = set;
      writes[0].dstBinding = 0;
      writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      writes[0].descriptorCount = 1;
      writes[0].pImageInfo = &histInfo;
      writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[1].dstSet = set;
      writes[1].dstBinding = 1;
      writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      writes[1].descriptorCount = 1;
      writes[1].pImageInfo = &depthInfo;
      writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[2].dstSet = set;
      writes[2].dstBinding = 2;
      writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      writes[2].descriptorCount = 1;
      writes[2].pImageInfo = &colorInfo;
      vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
    }
  }
}

void DescriptorManager::recreateSsgiPrevSets(
    VkDevice device, const std::vector<VkImageView> &ssgiHistoryViews,
    VkSampler sampler) {
  if (ssgiPrevDescriptorPool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(device, ssgiPrevDescriptorPool, nullptr);
    ssgiPrevDescriptorPool = VK_NULL_HANDLE;
  }
  size_t historyCount = ssgiHistoryViews.size();
  VkDescriptorPoolSize sz = {};
  sz.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  sz.descriptorCount = static_cast<uint32_t>(historyCount);
  VkDescriptorPoolCreateInfo pci = {};
  pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pci.maxSets = static_cast<uint32_t>(historyCount);
  pci.poolSizeCount = 1;
  pci.pPoolSizes = &sz;
  if (vkCreateDescriptorPool(device, &pci, nullptr,
                             &ssgiPrevDescriptorPool) != VK_SUCCESS)
    throw std::runtime_error("Failed to create SSGI prev descriptor pool");

  std::vector<VkDescriptorSetLayout> layouts(historyCount, ssgiPrevSetLayout);
  ssgiPrevDescriptorSets.assign(historyCount, VK_NULL_HANDLE);

  VkDescriptorSetAllocateInfo ai = {};
  ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  ai.descriptorPool = ssgiPrevDescriptorPool;
  ai.descriptorSetCount = static_cast<uint32_t>(historyCount);
  ai.pSetLayouts = layouts.data();
  if (vkAllocateDescriptorSets(device, &ai,
                               ssgiPrevDescriptorSets.data()) != VK_SUCCESS)
    throw std::runtime_error("Failed to allocate SSGI prev descriptor sets");

  for (size_t historyIndex = 0; historyIndex < historyCount; ++historyIndex) {
    VkDescriptorImageInfo info = {
        sampler,
        ssgiHistoryViews[(historyIndex + historyCount - 1) % historyCount],
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet w = {};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = ssgiPrevDescriptorSets[historyIndex];
    w.dstBinding = 0;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.descriptorCount = 1;
    w.pImageInfo = &info;
    vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
  }
}

// Layout creation

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
  // Legacy: kept for the shadow pipeline which still uses per-material
  // descriptor sets. The G-buffer pass now uses the bindless layout below.
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

  // --- Phase 7.2: Bindless texture array layout (set 1, geometry pass) ---
  // Single binding 0 with MAX_BINDLESS_TEXTURES combined-image-samplers.
  // Flags enable:
  //   PARTIALLY_BOUND: not all 4096 slots need to be populated
  //   UPDATE_AFTER_BIND: textures can be written after binding the set
  //   VARIABLE_DESCRIPTOR_COUNT: actual count can be less than max
  VkDescriptorSetLayoutBinding bindlessBinding = {};
  bindlessBinding.binding = 0;
  bindlessBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  bindlessBinding.descriptorCount = MAX_BINDLESS_TEXTURES;
  bindlessBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorBindingFlags bindlessFlags =
      VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
      VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
  VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsCI = {};
  bindingFlagsCI.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
  bindingFlagsCI.bindingCount = 1;
  bindingFlagsCI.pBindingFlags = &bindlessFlags;

  VkDescriptorSetLayoutCreateInfo bindlessCI = {};
  bindlessCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  bindlessCI.pNext = &bindingFlagsCI;
  bindlessCI.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
  bindlessCI.bindingCount = 1;
  bindlessCI.pBindings = &bindlessBinding;
  if (vkCreateDescriptorSetLayout(device, &bindlessCI, nullptr,
                                  &bindlessSetLayout) != VK_SUCCESS)
    throw std::runtime_error(
        "Failed to create bindless descriptor set layout");

  // --- G-buffer sampler layout (set 1, deferred/composite/SSGI passes): 6
  // bindings 0-3: gb0/gb1/gb2/depth
  // binding 4: lit (sampled by SSR composite)
  // binding 5: ssgi raw bounce (sampled by lit.frag with cross-bilateral)
  // ssgi.frag declares only bindings 0/1/3; unused bindings (4, 5) stay
  // bound but unread, which is legal in Vulkan.
  VkDescriptorSetLayoutBinding gbBindings[6] = {};
  for (int i = 0; i < 6; ++i) {
    gbBindings[i].binding = static_cast<uint32_t>(i);
    gbBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    gbBindings[i].descriptorCount = 1;
    gbBindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  }
  VkDescriptorSetLayoutCreateInfo gbCI = {};
  gbCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  gbCI.bindingCount = 6;
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

  // --- TAA layout (set 2, tonemap subpass): 3 sampled-image bindings ---
  // binding 0 = previous-frame TAA history (HDR R16G16B16A16_SFLOAT)
  // binding 1 = G-buffer depth (used to reconstruct world pos for
  //             reprojection through prevViewProj)
  // binding 2 = current-frame colorBuffer (HDR pre-tonemap) for the 3×3
  //             YCoCg neighborhood clamp. Same image as the input
  //             attachment at set=1,binding=0 — kept as a separate
  //             descriptor because subpassLoad does not accept offsets.
  std::array<VkDescriptorSetLayoutBinding, 3> taaBindings = {};
  for (uint32_t i = 0; i < 3; ++i) {
    taaBindings[i].binding = i;
    taaBindings[i].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    taaBindings[i].descriptorCount = 1;
    taaBindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  }
  VkDescriptorSetLayoutCreateInfo taaCI = {};
  taaCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  taaCI.bindingCount = static_cast<uint32_t>(taaBindings.size());
  taaCI.pBindings = taaBindings.data();
  if (vkCreateDescriptorSetLayout(device, &taaCI, nullptr, &taaSetLayout) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create TAA descriptor set layout");

  // --- SSGI prev-history layout (set 2 on the SSGI pipeline) ---
  // Single binding 0 = previous-frame SSGI bounce image. Read by
  // ssgi.frag for temporal reprojection + blend.
  VkDescriptorSetLayoutBinding ssgiPrevBinding = {};
  ssgiPrevBinding.binding = 0;
  ssgiPrevBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  ssgiPrevBinding.descriptorCount = 1;
  ssgiPrevBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  VkDescriptorSetLayoutCreateInfo ssgiPrevCI = {};
  ssgiPrevCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  ssgiPrevCI.bindingCount = 1;
  ssgiPrevCI.pBindings = &ssgiPrevBinding;
  if (vkCreateDescriptorSetLayout(device, &ssgiPrevCI, nullptr,
                                  &ssgiPrevSetLayout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create SSGI prev-history layout");
}

// Push constant range

void DescriptorManager::createPushConstantRange() {
  pushConstantRange.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = sizeof(ModelPushConstants);
}

// Uniform buffer creation

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

// Descriptor pool creation

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

  // G-buffer sampler pool: 6 samplers per frame, one set per SSGI history slot.
  //   gb0 / gb1 / gb2 / depth / lit (SSR comp) / ssgi (lit bilateral)
  // historyCount * count sets total (one per swap image, per history slot).
  VkDescriptorPoolSize gbSize = {};
  gbSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  gbSize.descriptorCount =
      static_cast<uint32_t>(count * 6 * (MAX_FRAMES_DRAWS + 1));

  VkDescriptorPoolCreateInfo gbCI = {};
  gbCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  gbCI.maxSets = static_cast<uint32_t>(count * (MAX_FRAMES_DRAWS + 1));
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

  // Phase 7.2: Bindless texture array pool. A single set with up to
  // MAX_BINDLESS_TEXTURES combined-image-sampler descriptors. The
  // UPDATE_AFTER_BIND flag is required because the set layout uses it.
  VkDescriptorPoolSize bindlessSize = {};
  bindlessSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  bindlessSize.descriptorCount = MAX_BINDLESS_TEXTURES;

  VkDescriptorPoolCreateInfo bindlessPoolCI = {};
  bindlessPoolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  bindlessPoolCI.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
  bindlessPoolCI.maxSets = 1;
  bindlessPoolCI.poolSizeCount = 1;
  bindlessPoolCI.pPoolSizes = &bindlessSize;
  if (vkCreateDescriptorPool(device, &bindlessPoolCI, nullptr,
                             &bindlessDescriptorPool) != VK_SUCCESS)
    throw std::runtime_error("Failed to create bindless descriptor pool");
}

// Descriptor set creation

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

  // Phase 7.2: Allocate the single global bindless descriptor set.
  // VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT lets us specify
  // the actual number of descriptors we intend to use (MAX_BINDLESS_TEXTURES)
  // rather than a fixed compile-time constant in the layout.
  uint32_t variableCount = MAX_BINDLESS_TEXTURES;
  VkDescriptorSetVariableDescriptorCountAllocateInfo varCountInfo = {};
  varCountInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
  varCountInfo.descriptorSetCount = 1;
  varCountInfo.pDescriptorCounts = &variableCount;

  VkDescriptorSetAllocateInfo bindlessAllocInfo = {};
  bindlessAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  bindlessAllocInfo.pNext = &varCountInfo;
  bindlessAllocInfo.descriptorPool = bindlessDescriptorPool;
  bindlessAllocInfo.descriptorSetCount = 1;
  bindlessAllocInfo.pSetLayouts = &bindlessSetLayout;
  if (vkAllocateDescriptorSets(device, &bindlessAllocInfo,
                               &bindlessDescriptorSet) != VK_SUCCESS)
    throw std::runtime_error("Failed to allocate bindless descriptor set");
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

void DescriptorManager::registerBindlessTexture(VkDevice device, uint32_t index,
                                                VkImageView view,
                                                VkSampler sampler) {
  VkDescriptorImageInfo imgInfo = {};
  imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  imgInfo.imageView = view;
  imgInfo.sampler = sampler;

  VkWriteDescriptorSet write = {};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = bindlessDescriptorSet;
  write.dstBinding = 0;
  write.dstArrayElement = index;
  write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  write.descriptorCount = 1;
  write.pImageInfo = &imgInfo;
  vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}
