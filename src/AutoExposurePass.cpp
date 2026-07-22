#include "AutoExposurePass.h"

#include "VulkanDebug.h"
#include "VulkanSync.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace {
struct HistPC {
  float minLogLum;
  float invLogLumRange;
  uint32_t width;
  uint32_t height;
};

struct ExpPC {
  float minLogLum;
  float logLumRange;
  uint32_t numPixels;
};

VkShaderModule loadComputeSpv(VkDevice device, const std::string &relPath) {
  std::vector<std::string> candidates = {relPath, "../" + relPath};
  std::string found;
  for (const std::string &candidate : candidates) {
    if (std::filesystem::exists(candidate)) {
      found = candidate;
      break;
    }
  }
  if (found.empty())
    throw std::runtime_error("Compute shader not found: " + relPath);

  const std::vector<char> code = readFile(found);
  VkShaderModuleCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  ci.codeSize = code.size();
  ci.pCode = reinterpret_cast<const uint32_t *>(code.data());
  VkShaderModule module = VK_NULL_HANDLE;
  if (vkCreateShaderModule(device, &ci, nullptr, &module) != VK_SUCCESS)
    throw std::runtime_error("Failed to create shader module: " + found);
  return module;
}
} // namespace

void AutoExposurePass::create(VulkanDevice &newDevice, VkExtent2D extent,
                              const std::vector<ImageViewHandle> &litViews,
                              VkSampler litSampler,
                              VkPipelineCache pipelineCache) {
  cleanup();
  device = &newDevice;
  renderExtent = extent;

  VkDevice dev = device->getLogicalDevice();
  VmaAllocator allocator = device->getAllocator();
  const size_t swapCount = litViews.size();

  createBuffer(allocator, sizeof(uint32_t) * 256,
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, &histogramBuffer);

  {
    VkCommandBuffer cmd =
        beginCommandBuffer(dev, device->getGraphicsCommandPool());
    vkCmdFillBuffer(cmd, histogramBuffer.get(), 0, VK_WHOLE_SIZE, 0);
    endAndSubmitCommandBuffer(dev, device->getGraphicsCommandPool(),
                              device->getGraphicsQueue(), cmd);
  }

  resultBuffers.clear();
  resultMapped.clear();
  resultBuffers.reserve(MAX_FRAMES_DRAWS);
  resultMapped.reserve(MAX_FRAMES_DRAWS);
  for (int i = 0; i < MAX_FRAMES_DRAWS; ++i) {
    VkBufferCreateInfo bufferCI{};
    bufferCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCI.size = 16;
    bufferCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO;
    allocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer rawBuffer = VK_NULL_HANDLE;
    VmaAllocation rawAllocation = VK_NULL_HANDLE;
    VmaAllocationInfo rawInfo{};
    if (vmaCreateBuffer(allocator, &bufferCI, &allocCI, &rawBuffer,
                        &rawAllocation, &rawInfo) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create auto-exposure result buffer");
    }

    *reinterpret_cast<float *>(rawInfo.pMappedData) = 1.0f;
    vmaFlushAllocation(allocator, rawAllocation, 0, sizeof(float));
    resultBuffers.emplace_back(allocator, rawBuffer, rawAllocation);
    resultMapped.push_back(rawInfo.pMappedData);
  }

  {
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ci.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(dev, &ci, nullptr, &histogramSetLayout) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to create autoExp histogram set layout");
  }
  {
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ci.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(dev, &ci, nullptr, &exposureSetLayout) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to create autoExp exposure set layout");
  }

  {
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = sizeof(HistPC);
    VkPipelineLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ci.setLayoutCount = 1;
    ci.pSetLayouts = &histogramSetLayout;
    ci.pushConstantRangeCount = 1;
    ci.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(dev, &ci, nullptr,
                               &histogramPipelineLayout) != VK_SUCCESS)
      throw std::runtime_error(
          "Failed to create autoExp histogram pipeline layout");
  }
  {
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = sizeof(ExpPC);
    VkPipelineLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ci.setLayoutCount = 1;
    ci.pSetLayouts = &exposureSetLayout;
    ci.pushConstantRangeCount = 1;
    ci.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(dev, &ci, nullptr, &exposurePipelineLayout) !=
        VK_SUCCESS)
      throw std::runtime_error(
          "Failed to create autoExp exposure pipeline layout");
  }

  {
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(swapCount);
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount =
        static_cast<uint32_t>(swapCount + 2 * MAX_FRAMES_DRAWS);
    VkDescriptorPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.maxSets = static_cast<uint32_t>(swapCount + MAX_FRAMES_DRAWS);
    ci.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    ci.pPoolSizes = poolSizes.data();
    if (vkCreateDescriptorPool(dev, &ci, nullptr, &descriptorPool) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to create autoExp descriptor pool");
  }

  {
    std::vector<VkDescriptorSetLayout> layouts(swapCount, histogramSetLayout);
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = descriptorPool;
    ai.descriptorSetCount = static_cast<uint32_t>(swapCount);
    ai.pSetLayouts = layouts.data();
    histogramSets.resize(swapCount);
    if (vkAllocateDescriptorSets(dev, &ai, histogramSets.data()) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to alloc autoExp histogram sets");
  }
  {
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_DRAWS,
                                               exposureSetLayout);
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = descriptorPool;
    ai.descriptorSetCount = MAX_FRAMES_DRAWS;
    ai.pSetLayouts = layouts.data();
    exposureSets.resize(MAX_FRAMES_DRAWS);
    if (vkAllocateDescriptorSets(dev, &ai, exposureSets.data()) != VK_SUCCESS)
      throw std::runtime_error("Failed to alloc autoExp exposure sets");
  }

  for (size_t i = 0; i < swapCount; ++i) {
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = litViews[i].get();
    imageInfo.sampler = litSampler;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = histogramBuffer.get();
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;
    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = histogramSets[i];
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &imageInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = histogramSets[i];
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &bufferInfo;
    vkUpdateDescriptorSets(dev, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
  }

  for (int i = 0; i < MAX_FRAMES_DRAWS; ++i) {
    VkDescriptorBufferInfo histogramInfo{};
    histogramInfo.buffer = histogramBuffer.get();
    histogramInfo.offset = 0;
    histogramInfo.range = VK_WHOLE_SIZE;
    VkDescriptorBufferInfo resultInfo{};
    resultInfo.buffer = resultBuffers[i].get();
    resultInfo.offset = 0;
    resultInfo.range = VK_WHOLE_SIZE;
    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = exposureSets[i];
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &histogramInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = exposureSets[i];
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &resultInfo;
    vkUpdateDescriptorSets(dev, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
  }

  {
    VkShaderModule histModule = loadComputeSpv(dev, "Shaders/histogram.comp.spv");
    VkShaderModule expModule = loadComputeSpv(dev, "Shaders/exposure.comp.spv");
    auto buildPipeline = [&](VkShaderModule module, VkPipelineLayout layout) {
      VkPipelineShaderStageCreateInfo stage{};
      stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      stage.module = module;
      stage.pName = "main";
      VkComputePipelineCreateInfo ci{};
      ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
      ci.stage = stage;
      ci.layout = layout;
      VkPipeline pipeline = VK_NULL_HANDLE;
      if (vkCreateComputePipelines(dev, pipelineCache, 1, &ci, nullptr,
                                   &pipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create autoExp compute pipeline");
      return pipeline;
    };
    histogramPipeline = buildPipeline(histModule, histogramPipelineLayout);
    exposurePipeline = buildPipeline(expModule, exposurePipelineLayout);
    vkDestroyShaderModule(dev, histModule, nullptr);
    vkDestroyShaderModule(dev, expModule, nullptr);
  }
}

void AutoExposurePass::cleanup() {
  if (!device)
    return;

  VkDevice dev = device->getLogicalDevice();
  if (histogramPipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(dev, histogramPipeline, nullptr);
  if (exposurePipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(dev, exposurePipeline, nullptr);
  if (histogramPipelineLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(dev, histogramPipelineLayout, nullptr);
  if (exposurePipelineLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(dev, exposurePipelineLayout, nullptr);
  if (descriptorPool != VK_NULL_HANDLE)
    vkDestroyDescriptorPool(dev, descriptorPool, nullptr);
  if (histogramSetLayout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(dev, histogramSetLayout, nullptr);
  if (exposureSetLayout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(dev, exposureSetLayout, nullptr);

  histogramPipeline = VK_NULL_HANDLE;
  exposurePipeline = VK_NULL_HANDLE;
  histogramPipelineLayout = VK_NULL_HANDLE;
  exposurePipelineLayout = VK_NULL_HANDLE;
  descriptorPool = VK_NULL_HANDLE;
  histogramSetLayout = VK_NULL_HANDLE;
  exposureSetLayout = VK_NULL_HANDLE;
  histogramSets.clear();
  exposureSets.clear();
  resultBuffers.clear();
  resultMapped.clear();
  histogramBuffer.reset();
  renderExtent = {};
  device = nullptr;
}

void AutoExposurePass::record(VkCommandBuffer cmd, uint32_t currentImage,
                              int currentFrame,
                              const std::vector<AllocatedImage> &litImages,
                              bool enabled, bool litInputReadyForCompute) {
  if (!enabled || histogramPipeline == VK_NULL_HANDLE ||
      exposurePipeline == VK_NULL_HANDLE || currentImage >= histogramSets.size() ||
      currentImage >= litImages.size() ||
      currentFrame < 0 ||
      currentFrame >= static_cast<int>(exposureSets.size()))
    return;

  const float logLumRange = kMaxLogLum - kMinLogLum;
  const float invLogLumRange = 1.0f / logLumRange;

  vkdbgBeginLabel(cmd, "Auto-Exposure (histogram + reduce)", 0.95f, 0.85f,
                  0.2f);

  if (!litInputReadyForCompute) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = litImages[currentImage].get();
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    recordImageBarrier2(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, barrier);
  }

  {
    HistPC pc{kMinLogLum, invLogLumRange, renderExtent.width,
              renderExtent.height};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, histogramPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            histogramPipelineLayout, 0, 1,
                            &histogramSets[currentImage], 0, nullptr);
    vkCmdPushConstants(cmd, histogramPipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (renderExtent.width + 15) / 16,
                  (renderExtent.height + 15) / 16, 1);
  }

  {
    recordMemoryBarrier2(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_ACCESS_SHADER_WRITE_BIT,
                         VK_ACCESS_SHADER_READ_BIT |
                             VK_ACCESS_SHADER_WRITE_BIT);
  }

  {
    ExpPC pc{kMinLogLum, logLumRange, renderExtent.width * renderExtent.height};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, exposurePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            exposurePipelineLayout, 0, 1,
                            &exposureSets[currentFrame], 0, nullptr);
    vkCmdPushConstants(cmd, exposurePipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, 1, 1, 1);
  }

  {
    recordMemoryBarrier2(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         VK_ACCESS_SHADER_WRITE_BIT,
                         VK_ACCESS_HOST_READ_BIT);
  }

  vkdbgEndLabel(cmd);
}

float AutoExposurePass::updateExposureScale(int currentFrame, bool enabled,
                                            float exposureEV) {
  auto now = std::chrono::steady_clock::now();
  float dt = 0.0f;
  if (hasLastUpdate) {
    dt = std::chrono::duration<float>(now - lastUpdate).count();
  }
  lastUpdate = now;
  hasLastUpdate = true;
  dt = std::clamp(dt, 0.0f, 0.25f);

  if (enabled && currentFrame >= 0 &&
      currentFrame < static_cast<int>(resultMapped.size())) {
    // GPU-written readback: invalidate before the host read in case the
    // heap is host-visible but not host-coherent (no-op on coherent heaps).
    vmaInvalidateAllocation(device->getAllocator(),
                            resultBuffers[currentFrame].getAllocation(), 0,
                            sizeof(float));
    float target =
        *reinterpret_cast<const float *>(resultMapped[currentFrame]);
    if (!std::isfinite(target) || target <= 0.0f)
      target = 1.0f;
    target = std::clamp(target, 0.5f, 2.5f);
    const float alpha = 1.0f - std::exp(-dt / kTauSeconds);
    adaptedExposure =
        adaptedExposure + alpha * (target - adaptedExposure);
  } else {
    adaptedExposure = 1.0f;
  }

  return adaptedExposure * std::exp2(exposureEV);
}

void AutoExposurePass::resetAdaptation(float exposureScale) {
  adaptedExposure = std::clamp(exposureScale, 0.5f, 2.5f);
  hasLastUpdate = false;
}
