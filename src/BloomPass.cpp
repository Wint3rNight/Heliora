#include "BloomPass.h"

#include "VulkanDebug.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace {
struct BloomDownsamplePC {
  float srcTexelSize[2];
  float threshold;
  float knee;
  int prefilter;
};

struct BloomUpsamplePC {
  float srcTexelSize[2];
  float radius;
  float intensity;
  int finalPass;
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

void BloomPass::create(VulkanDevice &newDevice, VkExtent2D extent,
                       size_t swapCount,
                       const std::vector<ImageViewHandle> &litViews) {
  cleanup();
  device = &newDevice;
  renderExtent = extent;

  VkDevice dev = device->getLogicalDevice();
  if (litViews.size() < swapCount)
    throw std::runtime_error("Bloom requires one lit image view per swap image");
  {
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(device->getPhysicalDevice(),
                                        bloomFormat, &props);
    const VkFormatFeatureFlags required =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
        VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
    if ((props.optimalTilingFeatures & required) != required)
      throw std::runtime_error(
          "Bloom requires R16G16B16A16_SFLOAT sampled storage images");
  }

  VkExtent2D mipExtent = extent;
  for (uint32_t level = 0; level < kMipCount; ++level) {
    mips[level].extent = mipExtent;
    mips[level].images.clear();
    mips[level].views.clear();
    mips[level].images.reserve(swapCount);
    mips[level].views.reserve(swapCount);

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    for (size_t i = 0; i < swapCount; ++i) {
      VkImageCreateInfo ci{};
      ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      ci.imageType = VK_IMAGE_TYPE_2D;
      ci.extent = {mipExtent.width, mipExtent.height, 1};
      ci.mipLevels = 1;
      ci.arrayLayers = 1;
      ci.format = bloomFormat;
      ci.tiling = VK_IMAGE_TILING_OPTIMAL;
      ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
      ci.samples = VK_SAMPLE_COUNT_1_BIT;
      ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

      VkImage rawImage = VK_NULL_HANDLE;
      VmaAllocation allocation = VK_NULL_HANDLE;
      if (vmaCreateImage(device->getAllocator(), &ci, &aci, &rawImage,
                         &allocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create bloom image");
      }
      mips[level].images.emplace_back(device->getAllocator(), rawImage,
                                      allocation);

      VkImageViewCreateInfo vci{};
      vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      vci.image = mips[level].images.back().get();
      vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
      vci.format = bloomFormat;
      vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      VkImageView view = VK_NULL_HANDLE;
      if (vkCreateImageView(dev, &vci, nullptr, &view) != VK_SUCCESS)
        throw std::runtime_error("Failed to create bloom image view");
      mips[level].views.emplace_back(dev, view);
    }

    mipExtent.width = std::max(1u, mipExtent.width / 2);
    mipExtent.height = std::max(1u, mipExtent.height / 2);
  }

  {
    VkCommandBuffer cmd =
        beginCommandBuffer(dev, device->getGraphicsCommandPool());
    for (uint32_t level = 0; level < kMipCount; ++level) {
      for (size_t i = 0; i < swapCount; ++i) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = mips[level].images[i].get();
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);
      }
    }
    endAndSubmitCommandBuffer(dev, device->getGraphicsCommandPool(),
                              device->getGraphicsQueue(), cmd);
  }

  VkSamplerCreateInfo samplerCI{};
  samplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerCI.magFilter = VK_FILTER_LINEAR;
  samplerCI.minFilter = VK_FILTER_LINEAR;
  samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerCI.maxAnisotropy = 1.0f;
  samplerCI.minLod = 0.0f;
  samplerCI.maxLod = 0.0f;
  if (vkCreateSampler(dev, &samplerCI, nullptr, &bloomSampler) != VK_SUCCESS)
    throw std::runtime_error("Failed to create bloom sampler");

  {
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ci.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(dev, &ci, nullptr, &setLayout) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to create bloom descriptor layout");
  }

  auto createPipelineLayout = [&](uint32_t pcSize) {
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = pcSize;
    VkPipelineLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ci.setLayoutCount = 1;
    ci.pSetLayouts = &setLayout;
    ci.pushConstantRangeCount = 1;
    ci.pPushConstantRanges = &pc;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(dev, &ci, nullptr, &layout) != VK_SUCCESS)
      throw std::runtime_error("Failed to create bloom pipeline layout");
    return layout;
  };
  downsamplePipelineLayout =
      createPipelineLayout(sizeof(BloomDownsamplePC));
  upsamplePipelineLayout = createPipelineLayout(sizeof(BloomUpsamplePC));

  const uint32_t downCount = static_cast<uint32_t>(swapCount * kMipCount);
  const uint32_t upCount =
      static_cast<uint32_t>(swapCount * (kMipCount - 1));
  {
    std::array<VkDescriptorPoolSize, 2> sizes{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = downCount + upCount;
    sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[1].descriptorCount = downCount + upCount;
    VkDescriptorPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.maxSets = downCount + upCount;
    ci.poolSizeCount = static_cast<uint32_t>(sizes.size());
    ci.pPoolSizes = sizes.data();
    if (vkCreateDescriptorPool(dev, &ci, nullptr, &descriptorPool) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to create bloom descriptor pool");
  }

  auto allocateSets = [&](std::vector<VkDescriptorSet> &sets,
                          uint32_t count) {
    std::vector<VkDescriptorSetLayout> layouts(count, setLayout);
    sets.assign(count, VK_NULL_HANDLE);
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = descriptorPool;
    ai.descriptorSetCount = count;
    ai.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(dev, &ai, sets.data()) != VK_SUCCESS)
      throw std::runtime_error("Failed to allocate bloom descriptor sets");
  };
  allocateSets(downsampleSets, downCount);
  allocateSets(upsampleSets, upCount);

  for (size_t swapIdx = 0; swapIdx < swapCount; ++swapIdx) {
    for (uint32_t level = 0; level < kMipCount; ++level) {
      VkDescriptorImageInfo src{};
      src.sampler = bloomSampler;
      src.imageView =
          (level == 0) ? litViews[swapIdx].get()
                       : mips[level - 1].views[swapIdx].get();
      src.imageLayout = (level == 0) ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                     : VK_IMAGE_LAYOUT_GENERAL;

      VkDescriptorImageInfo dst{};
      dst.imageView = mips[level].views[swapIdx].get();
      dst.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

      VkDescriptorSet set = downsampleSets[swapIdx * kMipCount + level];
      std::array<VkWriteDescriptorSet, 2> writes{};
      writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[0].dstSet = set;
      writes[0].dstBinding = 0;
      writes[0].descriptorCount = 1;
      writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      writes[0].pImageInfo = &src;
      writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[1].dstSet = set;
      writes[1].dstBinding = 1;
      writes[1].descriptorCount = 1;
      writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      writes[1].pImageInfo = &dst;
      vkUpdateDescriptorSets(dev, static_cast<uint32_t>(writes.size()),
                             writes.data(), 0, nullptr);
    }

    for (uint32_t level = 1; level < kMipCount; ++level) {
      VkDescriptorImageInfo src{};
      src.sampler = bloomSampler;
      src.imageView = mips[level].views[swapIdx].get();
      src.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

      VkDescriptorImageInfo dst{};
      dst.imageView = mips[level - 1].views[swapIdx].get();
      dst.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

      VkDescriptorSet set =
          upsampleSets[swapIdx * (kMipCount - 1) + (level - 1)];
      std::array<VkWriteDescriptorSet, 2> writes{};
      writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[0].dstSet = set;
      writes[0].dstBinding = 0;
      writes[0].descriptorCount = 1;
      writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      writes[0].pImageInfo = &src;
      writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[1].dstSet = set;
      writes[1].dstBinding = 1;
      writes[1].descriptorCount = 1;
      writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      writes[1].pImageInfo = &dst;
      vkUpdateDescriptorSets(dev, static_cast<uint32_t>(writes.size()),
                             writes.data(), 0, nullptr);
    }
  }

  auto createComputePipeline = [&](const char *path, VkPipelineLayout layout) {
    VkShaderModule module = loadComputeSpv(dev, path);
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
    if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr,
                                 &pipeline) != VK_SUCCESS)
      throw std::runtime_error("Failed to create bloom compute pipeline");
    vkDestroyShaderModule(dev, module, nullptr);
    return pipeline;
  };
  downsamplePipeline = createComputePipeline(
      "Shaders/bloom_downsample.comp.spv", downsamplePipelineLayout);
  upsamplePipeline = createComputePipeline("Shaders/bloom_upsample.comp.spv",
                                           upsamplePipelineLayout);
}

void BloomPass::cleanup() {
  if (!device)
    return;

  VkDevice dev = device->getLogicalDevice();
  if (downsamplePipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(dev, downsamplePipeline, nullptr);
  if (upsamplePipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(dev, upsamplePipeline, nullptr);
  if (downsamplePipelineLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(dev, downsamplePipelineLayout, nullptr);
  if (upsamplePipelineLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(dev, upsamplePipelineLayout, nullptr);
  if (descriptorPool != VK_NULL_HANDLE)
    vkDestroyDescriptorPool(dev, descriptorPool, nullptr);
  if (setLayout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(dev, setLayout, nullptr);
  if (bloomSampler != VK_NULL_HANDLE)
    vkDestroySampler(dev, bloomSampler, nullptr);

  downsamplePipeline = VK_NULL_HANDLE;
  upsamplePipeline = VK_NULL_HANDLE;
  downsamplePipelineLayout = VK_NULL_HANDLE;
  upsamplePipelineLayout = VK_NULL_HANDLE;
  descriptorPool = VK_NULL_HANDLE;
  setLayout = VK_NULL_HANDLE;
  bloomSampler = VK_NULL_HANDLE;
  downsampleSets.clear();
  upsampleSets.clear();
  for (MipResources &mip : mips) {
    mip.views.clear();
    mip.images.clear();
    mip.extent = {};
  }
  renderExtent = {};
  device = nullptr;
}

void BloomPass::record(VkCommandBuffer cmd, uint32_t currentImage,
                       const std::vector<AllocatedImage> &litImages,
                       bool enabled, int debugMode, float threshold,
                       float radius, float intensity) {
  if ((!enabled && debugMode != 13) ||
      downsamplePipeline == VK_NULL_HANDLE ||
      upsamplePipeline == VK_NULL_HANDLE || currentImage >= litImages.size() ||
      currentImage >= mips[0].images.size())
    return;

  vkdbgBeginLabel(cmd, "Bloom Pyramid (downsample + upsample)", 1.0f, 0.55f,
                  0.2f);

  {
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
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &barrier);
  }

  std::array<VkImageMemoryBarrier, kMipCount> toGeneral{};
  for (uint32_t level = 0; level < kMipCount; ++level) {
    VkImageMemoryBarrier &barrier = toGeneral[level];
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = mips[level].images[currentImage].get();
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  }
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, static_cast<uint32_t>(toGeneral.size()),
                       toGeneral.data());

  auto barrierBloomMip = [&](uint32_t level) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = mips[level].images[currentImage].get();
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &barrier);
  };

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, downsamplePipeline);
  for (uint32_t level = 0; level < kMipCount; ++level) {
    VkExtent2D srcExtent = (level == 0) ? renderExtent : mips[level - 1].extent;
    VkExtent2D dstExtent = mips[level].extent;
    BloomDownsamplePC pc = {
        {1.0f / static_cast<float>(srcExtent.width),
         1.0f / static_cast<float>(srcExtent.height)},
        threshold, threshold * 0.5f, level == 0 ? 1 : 0};
    VkDescriptorSet set = downsampleSets[currentImage * kMipCount + level];
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            downsamplePipelineLayout, 0, 1, &set, 0,
                            nullptr);
    vkCmdPushConstants(cmd, downsamplePipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (dstExtent.width + 7) / 8,
                  (dstExtent.height + 7) / 8, 1);
    barrierBloomMip(level);
  }

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, upsamplePipeline);
  for (int level = static_cast<int>(kMipCount) - 1; level >= 1; --level) {
    VkExtent2D srcExtent = mips[level].extent;
    VkExtent2D dstExtent = mips[level - 1].extent;
    BloomUpsamplePC pc = {
        {1.0f / static_cast<float>(srcExtent.width),
         1.0f / static_cast<float>(srcExtent.height)},
        radius, intensity, level == 1 ? 1 : 0};
    VkDescriptorSet set =
        upsampleSets[currentImage * (kMipCount - 1) +
                     (static_cast<uint32_t>(level) - 1)];
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            upsamplePipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, upsamplePipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (dstExtent.width + 7) / 8,
                  (dstExtent.height + 7) / 8, 1);
    barrierBloomMip(static_cast<uint32_t>(level - 1));
  }

  std::array<VkImageMemoryBarrier, kMipCount> toRead{};
  for (uint32_t level = 0; level < kMipCount; ++level) {
    VkImageMemoryBarrier &barrier = toRead[level];
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = mips[level].images[currentImage].get();
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  }
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, static_cast<uint32_t>(toRead.size()),
                       toRead.data());

  vkdbgEndLabel(cmd);
}

std::vector<VkImageView> BloomPass::mip0Views() const {
  std::vector<VkImageView> views;
  views.reserve(mips[0].views.size());
  for (const ImageViewHandle &view : mips[0].views)
    views.push_back(view.get());
  return views;
}
