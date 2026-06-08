#include "MaterialProbePass.h"

#include "VulkanDebug.h"
#include "VulkanSync.h"

#include <array>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace {
VkShaderModule loadProbeShader(VkDevice device) {
  const std::array<std::string, 2> candidates = {
      "Shaders/material_probe.comp.spv", "../Shaders/material_probe.comp.spv"};
  for (const std::string &candidate : candidates) {
    if (!std::filesystem::exists(candidate))
      continue;

    const std::vector<char> code = readFile(candidate);
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &ci, nullptr, &module) != VK_SUCCESS)
      throw std::runtime_error("Failed to create material probe shader module");
    return module;
  }
  throw std::runtime_error("Material probe shader not found");
}
} // namespace

void MaterialProbePass::create(VulkanDevice &newDevice, uint32_t frameCount,
                               const DescriptorManager &descriptorManager,
                               VkPipelineCache pipelineCache) {
  cleanup();
  device = &newDevice;
  VkDevice dev = device->getLogicalDevice();

  outputBuffers.resize(frameCount);
  for (AllocatedBuffer &buffer : outputBuffers) {
    createBuffer(device->getAllocator(), sizeof(MaterialProbeResult),
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT, &buffer);
    MaterialProbeResult zero{};
    void *mapped = nullptr;
    if (vmaMapMemory(device->getAllocator(), buffer.getAllocation(), &mapped) ==
            VK_SUCCESS &&
        mapped) {
      std::memcpy(mapped, &zero, sizeof(zero));
      vmaFlushAllocation(device->getAllocator(), buffer.getAllocation(), 0,
                         sizeof(zero));
      vmaUnmapMemory(device->getAllocator(), buffer.getAllocation());
    }
  }

  VkDescriptorSetLayoutBinding outputBinding{};
  outputBinding.binding = 0;
  outputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  outputBinding.descriptorCount = 1;
  outputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo layoutCI{};
  layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutCI.bindingCount = 1;
  layoutCI.pBindings = &outputBinding;
  if (vkCreateDescriptorSetLayout(dev, &layoutCI, nullptr, &outputSetLayout) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create material probe descriptor layout");

  VkDescriptorPoolSize poolSize{};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = frameCount;

  VkDescriptorPoolCreateInfo poolCI{};
  poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolCI.maxSets = frameCount;
  poolCI.poolSizeCount = 1;
  poolCI.pPoolSizes = &poolSize;
  if (vkCreateDescriptorPool(dev, &poolCI, nullptr, &descriptorPool) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create material probe descriptor pool");

  std::vector<VkDescriptorSetLayout> layouts(frameCount, outputSetLayout);
  outputSets.assign(frameCount, VK_NULL_HANDLE);
  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descriptorPool;
  allocInfo.descriptorSetCount = frameCount;
  allocInfo.pSetLayouts = layouts.data();
  if (vkAllocateDescriptorSets(dev, &allocInfo, outputSets.data()) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to allocate material probe descriptors");

  for (uint32_t i = 0; i < frameCount; ++i) {
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = outputBuffers[i].get();
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(MaterialProbeResult);

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = outputSets[i];
    write.dstBinding = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;
    vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);
  }

  std::array<VkDescriptorSetLayout, 3> setLayouts = {
      descriptorManager.getVPLayout(), descriptorManager.getGBufferLayout(),
      outputSetLayout};
  VkPushConstantRange pushRange{};
  pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushRange.offset = 0;
  pushRange.size = sizeof(MaterialProbePushConstants);

  VkPipelineLayoutCreateInfo pipelineLayoutCI{};
  pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCI.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
  pipelineLayoutCI.pSetLayouts = setLayouts.data();
  pipelineLayoutCI.pushConstantRangeCount = 1;
  pipelineLayoutCI.pPushConstantRanges = &pushRange;
  if (vkCreatePipelineLayout(dev, &pipelineLayoutCI, nullptr,
                             &pipelineLayout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create material probe pipeline layout");

  VkShaderModule shader = loadProbeShader(dev);
  VkPipelineShaderStageCreateInfo stage{};
  stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = shader;
  stage.pName = "main";

  VkComputePipelineCreateInfo pipelineCI{};
  pipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineCI.stage = stage;
  pipelineCI.layout = pipelineLayout;
  if (vkCreateComputePipelines(dev, pipelineCache, 1, &pipelineCI, nullptr,
                               &pipeline) != VK_SUCCESS) {
    vkDestroyShaderModule(dev, shader, nullptr);
    throw std::runtime_error("Failed to create material probe pipeline");
  }
  vkDestroyShaderModule(dev, shader, nullptr);
}

void MaterialProbePass::cleanup() {
  if (!device)
    return;

  VkDevice dev = device->getLogicalDevice();
  if (pipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(dev, pipeline, nullptr);
  if (pipelineLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(dev, pipelineLayout, nullptr);
  if (descriptorPool != VK_NULL_HANDLE)
    vkDestroyDescriptorPool(dev, descriptorPool, nullptr);
  if (outputSetLayout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(dev, outputSetLayout, nullptr);

  pipeline = VK_NULL_HANDLE;
  pipelineLayout = VK_NULL_HANDLE;
  descriptorPool = VK_NULL_HANDLE;
  outputSetLayout = VK_NULL_HANDLE;
  outputSets.clear();
  outputBuffers.clear();
  device = nullptr;
}

void MaterialProbePass::record(VkCommandBuffer cmd, uint32_t frameIndex,
                               VkDescriptorSet vpSet,
                               VkDescriptorSet gBufferSet,
                               glm::uvec2 pixel) {
  if (!device || pipeline == VK_NULL_HANDLE || frameIndex >= outputSets.size())
    return;

  vkdbgBeginLabel(cmd, "Material Probe", 0.9f, 0.25f, 0.95f);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  std::array<VkDescriptorSet, 3> sets = {vpSet, gBufferSet,
                                         outputSets[frameIndex]};
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
                          0, static_cast<uint32_t>(sets.size()), sets.data(),
                          0, nullptr);

  MaterialProbePushConstants pc{};
  pc.pixel = pixel;
  vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(pc), &pc);
  vkCmdDispatch(cmd, 1, 1, 1);

  recordMemoryBarrier2(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                       VK_ACCESS_HOST_READ_BIT);
  vkdbgEndLabel(cmd);
}

MaterialProbeResult MaterialProbePass::read(uint32_t frameIndex) {
  MaterialProbeResult result{};
  if (!device || frameIndex >= outputBuffers.size() || !outputBuffers[frameIndex])
    return result;

  VmaAllocator allocator = device->getAllocator();
  VmaAllocation allocation = outputBuffers[frameIndex].getAllocation();
  vmaInvalidateAllocation(allocator, allocation, 0, sizeof(result));

  void *mapped = nullptr;
  if (vmaMapMemory(allocator, allocation, &mapped) == VK_SUCCESS && mapped) {
    std::memcpy(&result, mapped, sizeof(result));
    vmaUnmapMemory(allocator, allocation);
  }
  return result;
}
