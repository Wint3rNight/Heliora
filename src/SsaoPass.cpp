#include "SsaoPass.h"

#include "VulkanDebug.h"

#include <array>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace {
struct QueueSharingInfo {
  VkSharingMode mode = VK_SHARING_MODE_EXCLUSIVE;
  uint32_t familyCount = 0;
  std::array<uint32_t, 2> families{};
};

QueueSharingInfo graphicsComputeSharing(const VulkanDevice &device) {
  QueueFamilyIndices indices = device.getQueueFamilies();
  QueueSharingInfo sharing{};
  if (indices.hasDedicatedCompute()) {
    sharing.mode = VK_SHARING_MODE_CONCURRENT;
    sharing.families = {static_cast<uint32_t>(indices.graphicsFamily),
                        static_cast<uint32_t>(indices.computeFamily)};
    sharing.familyCount = 2;
  }
  return sharing;
}

void applyQueueSharing(VkImageCreateInfo &ci, const QueueSharingInfo &sharing) {
  ci.sharingMode = sharing.mode;
  if (sharing.mode == VK_SHARING_MODE_CONCURRENT) {
    ci.queueFamilyIndexCount = sharing.familyCount;
    ci.pQueueFamilyIndices = sharing.families.data();
  }
}

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

void SsaoPass::create(VulkanDevice &newDevice, VkExtent2D extent,
                      size_t swapCount,
                      const DescriptorManager &descriptorManager) {
  cleanup();
  device = &newDevice;
  renderExtent = extent;

  VkDevice dev = device->getLogicalDevice();
  {
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(device->getPhysicalDevice(), format,
                                        &props);
    const VkFormatFeatureFlags required =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
        VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
    if ((props.optimalTilingFeatures & required) != required)
      throw std::runtime_error(
          "Async SSAO requires R16_SFLOAT sampled storage images");
  }

  const QueueSharingInfo queueSharing = graphicsComputeSharing(*device);
  images.clear();
  imageViews.clear();
  images.reserve(swapCount);
  imageViews.reserve(swapCount);

  VmaAllocationCreateInfo allocCI{};
  allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  for (size_t i = 0; i < swapCount; ++i) {
    VkImageCreateInfo imageCI{};
    imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCI.imageType = VK_IMAGE_TYPE_2D;
    imageCI.extent = {extent.width, extent.height, 1};
    imageCI.mipLevels = 1;
    imageCI.arrayLayers = 1;
    imageCI.format = format;
    imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
    applyQueueSharing(imageCI, queueSharing);

    VkImage rawImage = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    if (vmaCreateImage(device->getAllocator(), &imageCI, &allocCI, &rawImage,
                       &allocation, nullptr) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create SSAO image");
    }
    images.emplace_back(device->getAllocator(), rawImage, allocation);

    VkImageViewCreateInfo viewCI{};
    viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image = images.back().get();
    viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format = format;
    viewCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view = VK_NULL_HANDLE;
    if (vkCreateImageView(dev, &viewCI, nullptr, &view) != VK_SUCCESS)
      throw std::runtime_error("Failed to create SSAO image view");
    imageViews.emplace_back(dev, view);
  }

  VkSamplerCreateInfo samplerCI{};
  samplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerCI.magFilter = VK_FILTER_LINEAR;
  samplerCI.minFilter = VK_FILTER_LINEAR;
  samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerCI.maxAnisotropy = 1.0f;
  if (vkCreateSampler(dev, &samplerCI, nullptr, &resultSampler) != VK_SUCCESS)
    throw std::runtime_error("Failed to create SSAO sampler");

  VkDescriptorSetLayoutBinding outBinding{};
  outBinding.binding = 0;
  outBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  outBinding.descriptorCount = 1;
  outBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo dslCI{};
  dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslCI.bindingCount = 1;
  dslCI.pBindings = &outBinding;
  if (vkCreateDescriptorSetLayout(dev, &dslCI, nullptr, &outputSetLayout) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create SSAO output descriptor layout");
  }

  VkDescriptorPoolSize poolSize{};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  poolSize.descriptorCount = static_cast<uint32_t>(swapCount);
  VkDescriptorPoolCreateInfo poolCI{};
  poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolCI.maxSets = static_cast<uint32_t>(swapCount);
  poolCI.poolSizeCount = 1;
  poolCI.pPoolSizes = &poolSize;
  if (vkCreateDescriptorPool(dev, &poolCI, nullptr, &descriptorPool) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create SSAO descriptor pool");
  }

  std::vector<VkDescriptorSetLayout> layouts(swapCount, outputSetLayout);
  outputSets.assign(swapCount, VK_NULL_HANDLE);
  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descriptorPool;
  allocInfo.descriptorSetCount = static_cast<uint32_t>(swapCount);
  allocInfo.pSetLayouts = layouts.data();
  if (vkAllocateDescriptorSets(dev, &allocInfo, outputSets.data()) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate SSAO descriptor sets");
  }

  for (size_t i = 0; i < swapCount; ++i) {
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = imageViews[i].get();
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = outputSets[i];
    write.dstBinding = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.descriptorCount = 1;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);
  }

  std::array<VkDescriptorSetLayout, 3> setLayouts = {
      descriptorManager.getVPLayout(), descriptorManager.getGBufferLayout(),
      outputSetLayout};
  VkPipelineLayoutCreateInfo pipelineLayoutCI{};
  pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCI.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
  pipelineLayoutCI.pSetLayouts = setLayouts.data();
  if (vkCreatePipelineLayout(dev, &pipelineLayoutCI, nullptr,
                             &pipelineLayout) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create SSAO pipeline layout");
  }

  VkShaderModule shader = loadComputeSpv(dev, "Shaders/ssao.comp.spv");
  VkPipelineShaderStageCreateInfo stageCI{};
  stageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stageCI.module = shader;
  stageCI.pName = "main";
  VkComputePipelineCreateInfo pipelineCI{};
  pipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineCI.stage = stageCI;
  pipelineCI.layout = pipelineLayout;
  if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pipelineCI, nullptr,
                               &pipeline) != VK_SUCCESS) {
    vkDestroyShaderModule(dev, shader, nullptr);
    throw std::runtime_error("Failed to create SSAO compute pipeline");
  }
  vkDestroyShaderModule(dev, shader, nullptr);

  VkCommandBuffer cmd = beginCommandBuffer(dev, device->getComputeCommandPool());
  std::vector<VkImageMemoryBarrier> barriers;
  barriers.reserve(images.size());
  for (const AllocatedImage &image : images) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.get();
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barriers.push_back(barrier);
  }
  if (!barriers.empty()) {
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, static_cast<uint32_t>(barriers.size()),
                         barriers.data());
  }
  endAndSubmitCommandBuffer(dev, device->getComputeCommandPool(),
                            device->getComputeQueue(), cmd);
}

void SsaoPass::cleanup() {
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
  if (resultSampler != VK_NULL_HANDLE)
    vkDestroySampler(dev, resultSampler, nullptr);

  pipeline = VK_NULL_HANDLE;
  pipelineLayout = VK_NULL_HANDLE;
  descriptorPool = VK_NULL_HANDLE;
  outputSetLayout = VK_NULL_HANDLE;
  resultSampler = VK_NULL_HANDLE;
  outputSets.clear();
  imageViews.clear();
  images.clear();
  renderExtent = {};
  device = nullptr;
}

void SsaoPass::recordCommands(VkCommandBuffer cmd, uint32_t currentImage,
                              VkDescriptorSet vpSet,
                              VkDescriptorSet gBufferSet) {
  vkResetCommandBuffer(cmd, 0);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
    throw std::runtime_error("Failed to begin SSAO compute command buffer");

  if (currentImage >= images.size() || currentImage >= outputSets.size() ||
      pipeline == VK_NULL_HANDLE) {
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
      throw std::runtime_error("Failed to end SSAO compute command buffer");
    return;
  }

  VkImageMemoryBarrier toGeneral{};
  toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  toGeneral.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  toGeneral.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toGeneral.image = images[currentImage].get();
  toGeneral.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &toGeneral);

  vkdbgBeginLabel(cmd, "Async SSAO Compute", 0.2f, 0.7f, 1.0f);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  std::array<VkDescriptorSet, 3> sets = {vpSet, gBufferSet,
                                         outputSets[currentImage]};
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
                          0, static_cast<uint32_t>(sets.size()), sets.data(),
                          0, nullptr);
  vkCmdDispatch(cmd, (renderExtent.width + 7) / 8,
                (renderExtent.height + 7) / 8, 1);
  vkdbgEndLabel(cmd);

  VkImageMemoryBarrier toSampled = toGeneral;
  toSampled.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  toSampled.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  toSampled.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  toSampled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &toSampled);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
    throw std::runtime_error("Failed to end SSAO compute command buffer");
}

std::vector<VkImageView> SsaoPass::views() const {
  std::vector<VkImageView> out;
  out.reserve(imageViews.size());
  for (const ImageViewHandle &view : imageViews)
    out.push_back(view.get());
  return out;
}
