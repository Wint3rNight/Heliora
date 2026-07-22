#include "GpuDrivenGBufferPass.h"

#include "RenderResources.h"
#include "VulkanDebug.h"
#include "VulkanSync.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {
static_assert(sizeof(GpuDrivenMeshRecord) == 64,
              "GPU-driven mesh records must stay std430-friendly");

struct GpuDrivenFrustumData {
  glm::vec4 planes[6];
  glm::mat4 viewProj = glm::mat4(1.0f);
  glm::vec4 hzbParams = glm::vec4(0.0f);
  uint32_t meshCount = 0;
  uint32_t _pad[3] = {};
};

VkShaderModule createLocalShaderModule(VkDevice device,
                                       const std::vector<char> &code) {
  VkShaderModuleCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  ci.codeSize = code.size();
  ci.pCode = reinterpret_cast<const uint32_t *>(code.data());
  VkShaderModule module = VK_NULL_HANDLE;
  if (vkCreateShaderModule(device, &ci, nullptr, &module) != VK_SUCCESS)
    throw std::runtime_error("Failed to create shader module");
  return module;
}

} // namespace

bool GpuDrivenGBufferPass::ensureBuffer(AllocatedBuffer &buffer,
                                        VkDeviceSize &capacity,
                                        VkDeviceSize requiredSize,
                                        VkBufferUsageFlags usage,
                                        VmaMemoryUsage memoryUsage,
                                        VmaAllocationCreateFlags memoryFlags) {
  if (requiredSize == 0)
    requiredSize = 4;
  if (buffer && capacity >= requiredSize)
    return false;

  // Grow with 1.5x headroom: exact-fit sizing destroyed and reallocated the
  // buffer (and re-wrote the descriptor set) every time the draw count grew
  // past its previous maximum by even one element.
  VkDeviceSize newCapacity =
      std::max(requiredSize, capacity + capacity / 2);
  buffer.reset();
  createBuffer(device->getAllocator(), newCapacity, usage, memoryUsage,
               memoryFlags, &buffer);
  capacity = newCapacity;
  return true;
}

const GpuDrivenGBufferPass::GeometryRange *
GpuDrivenGBufferPass::findGeometry(const Mesh *mesh, int lod) const {
  // O(1) lookup — the previous linear scan was O(drawItems × ranges) per
  // frame. Key packs (mesh pointer, lod); lod is always < 8.
  auto it = geometryRangeLookup.find(geometryKey(mesh, lod));
  if (it == geometryRangeLookup.end())
    return nullptr;
  return &geometryRanges[it->second];
}

void GpuDrivenGBufferPass::uploadStaticGeometry() {
  if (!device || staticVertices.empty() || staticIndices.empty())
    return;

  const VkDeviceSize vertexBytes =
      static_cast<VkDeviceSize>(staticVertices.size() * sizeof(Vertex));
  const VkDeviceSize indexBytes =
      static_cast<VkDeviceSize>(staticIndices.size() * sizeof(uint32_t));

  ensureBuffer(staticVertexBuffer, staticVertexBufferSize, vertexBytes,
               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
               RENDER_DEVICE_ALLOCATION_FLAGS);
  ensureBuffer(staticIndexBuffer, staticIndexBufferSize, indexBytes,
               VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
               RENDER_DEVICE_ALLOCATION_FLAGS);

  AllocatedBuffer vertexStaging;
  createBuffer(device->getAllocator(), vertexBytes,
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO,
               RENDER_UPLOAD_ALLOCATION_FLAGS, &vertexStaging);
  uploadToAllocation(device->getAllocator(), vertexStaging.getAllocation(),
                     staticVertices.data(), vertexBytes);
  copyBuffer(device->getLogicalDevice(), device->getGraphicsQueue(),
             device->getGraphicsCommandPool(), vertexStaging.get(),
             staticVertexBuffer.get(), vertexBytes);

  AllocatedBuffer indexStaging;
  createBuffer(device->getAllocator(), indexBytes,
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO,
               RENDER_UPLOAD_ALLOCATION_FLAGS, &indexStaging);
  uploadToAllocation(device->getAllocator(), indexStaging.getAllocation(),
                     staticIndices.data(), indexBytes);
  copyBuffer(device->getLogicalDevice(), device->getGraphicsQueue(),
             device->getGraphicsCommandPool(), indexStaging.get(),
             staticIndexBuffer.get(), indexBytes);
}

void GpuDrivenGBufferPass::registerModelGeometry(int modelId,
                                                 ModelManager &modelManager) {
  if (std::find(registeredModelIds.begin(), registeredModelIds.end(),
                modelId) == registeredModelIds.end()) {
    registeredModelIds.push_back(modelId);
  }
  registerModelGeometryInternal(modelId, modelManager);
}

void GpuDrivenGBufferPass::registerModelGeometryInternal(
    int modelId, ModelManager &modelManager) {
  if (!device)
    return;

  MeshModel *model = modelManager.getModel(modelId);
  if (!model)
    return;

  bool changed = false;
  for (size_t meshIndex = 0; meshIndex < model->getMeshCount(); ++meshIndex) {
    const Mesh *mesh = model->getMesh(meshIndex);
    if (!mesh)
      continue;

    int32_t vertexOffset = -1;
    for (const GeometryRange &range : geometryRanges) {
      if (range.mesh == mesh) {
        vertexOffset = range.vertexOffset;
        break;
      }
    }
    if (vertexOffset < 0) {
      const std::vector<Vertex> &vertices = mesh->getCpuVertices();
      if (vertices.empty())
        continue;
      if (staticVertices.size() + vertices.size() >
          static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("GPU-driven vertex arena exceeded int32 range");
      }
      vertexOffset = static_cast<int32_t>(staticVertices.size());
      staticVertices.insert(staticVertices.end(), vertices.begin(),
                            vertices.end());
      changed = true;
    }

    for (int lod = 0; lod < mesh->getLodCount(); ++lod) {
      if (findGeometry(mesh, lod))
        continue;
      const std::vector<uint32_t> &indices = mesh->getCpuIndices(lod);
      if (indices.empty())
        continue;
      if (staticIndices.size() + indices.size() >
          static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("GPU-driven index arena exceeded uint32 range");
      }

      GeometryRange range{};
      range.mesh = mesh;
      range.lod = lod;
      range.firstIndex = static_cast<uint32_t>(staticIndices.size());
      range.indexCount = static_cast<uint32_t>(indices.size());
      range.vertexOffset = vertexOffset;
      staticIndices.insert(staticIndices.end(), indices.begin(), indices.end());
      geometryRangeLookup[geometryKey(mesh, lod)] = geometryRanges.size();
      geometryRanges.push_back(range);
      changed = true;
    }
  }

  if (!changed)
    return;

  if (staticVertexBuffer || staticIndexBuffer)
    vkDeviceWaitIdle(device->getLogicalDevice());
  uploadStaticGeometry();
}

void GpuDrivenGBufferPass::uploadMeshRecords(FrameResources &frame,
                                             const void *records,
                                             VkDeviceSize bytes,
                                             uint32_t recordCount) {
  meshCountValue = recordCount;
  if (bytes == 0 || records == nullptr)
    return;

  if (ensureBuffer(frame.meshBuffer, frame.meshBufferSize, bytes,
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
    frame.descriptorDirty = true;
  }

  uploadToAllocation(device->getAllocator(), frame.meshBuffer.getAllocation(),
                     records, bytes);
}

void GpuDrivenGBufferPass::create(
    VulkanDevice &newDevice, ModelManager &modelManager,
    const DescriptorManager &descriptorManager,
    VkRenderPass gBufferRenderPass, VkExtent2D extent,
    const std::vector<ImageViewHandle> &gBufferDepthViews,
    VkPipelineCache pipelineCache) {
  device = &newDevice;
  renderExtent = extent;
  VkDevice dev = device->getLogicalDevice();
  const size_t swapCount = gBufferDepthViews.size();

  hzbFormat = VK_FORMAT_R32_SFLOAT;
  {
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(device->getPhysicalDevice(), hzbFormat,
                                        &props);
    const VkFormatFeatureFlags required =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
        VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
    if ((props.optimalTilingFeatures & required) != required)
      throw std::runtime_error("HZB requires R32_SFLOAT sampled storage images");
  }

  std::array<VkDescriptorSetLayoutBinding, 8> bindings{};
  for (uint32_t i = 0; i < 5; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorCount = 1;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
  }
  bindings[5].binding = 5;
  bindings[5].descriptorCount = 1;
  bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  for (uint32_t i = 6; i < 8; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorCount = 1;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo layoutCI{};
  layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutCI.bindingCount = static_cast<uint32_t>(bindings.size());
  layoutCI.pBindings = bindings.data();
  if (vkCreateDescriptorSetLayout(dev, &layoutCI, nullptr, &setLayout) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create GPU-driven descriptor layout");

  std::array<VkDescriptorSetLayoutBinding, 2> hzbBindings{};
  hzbBindings[0].binding = 0;
  hzbBindings[0].descriptorCount = 1;
  hzbBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  hzbBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  hzbBindings[1].binding = 1;
  hzbBindings[1].descriptorCount = 1;
  hzbBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  hzbBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  VkDescriptorSetLayoutCreateInfo hzbLayoutCI{};
  hzbLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  hzbLayoutCI.bindingCount = static_cast<uint32_t>(hzbBindings.size());
  hzbLayoutCI.pBindings = hzbBindings.data();
  if (vkCreateDescriptorSetLayout(dev, &hzbLayoutCI, nullptr,
                                  &hzbBuildSetLayout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create HZB build descriptor layout");

  hzbExtent = {std::max(1u, extent.width / 2),
               std::max(1u, extent.height / 2)};
  hzbMipCount = 1;
  uint32_t maxDim = std::max(hzbExtent.width, hzbExtent.height);
  while (maxDim > 1) {
    maxDim >>= 1;
    ++hzbMipCount;
  }

  hzbImages.resize(swapCount);
  hzbViews.resize(swapCount);
  hzbMipViews.resize(swapCount);
  hzbValid.assign(swapCount, false);
  frames.resize(swapCount);

  VmaAllocationCreateInfo hzbAlloc{};
  hzbAlloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  hzbAlloc.flags = RENDER_DEVICE_ALLOCATION_FLAGS;
  for (size_t i = 0; i < swapCount; ++i) {
    VkImageCreateInfo imageCI{};
    imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCI.imageType = VK_IMAGE_TYPE_2D;
    imageCI.extent = {hzbExtent.width, hzbExtent.height, 1};
    imageCI.mipLevels = hzbMipCount;
    imageCI.arrayLayers = 1;
    imageCI.format = hzbFormat;
    imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkImage rawImage = VK_NULL_HANDLE;
    VmaAllocation rawAlloc = VK_NULL_HANDLE;
    if (vmaCreateImage(device->getAllocator(), &imageCI, &hzbAlloc, &rawImage,
                       &rawAlloc, nullptr) != VK_SUCCESS)
      throw std::runtime_error("Failed to create GPU-driven HZB image");
    hzbImages[i] = AllocatedImage(device->getAllocator(), rawImage, rawAlloc);

    VkImageViewCreateInfo viewCI{};
    viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image = hzbImages[i].get();
    viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format = hzbFormat;
    viewCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, hzbMipCount, 0,
                               1};
    VkImageView fullView = VK_NULL_HANDLE;
    if (vkCreateImageView(dev, &viewCI, nullptr, &fullView) != VK_SUCCESS)
      throw std::runtime_error("Failed to create HZB image view");
    hzbViews[i] = ImageViewHandle(dev, fullView);

    hzbMipViews[i].resize(hzbMipCount);
    for (uint32_t mip = 0; mip < hzbMipCount; ++mip) {
      VkImageViewCreateInfo mipViewCI = viewCI;
      mipViewCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1};
      VkImageView mipView = VK_NULL_HANDLE;
      if (vkCreateImageView(dev, &mipViewCI, nullptr, &mipView) != VK_SUCCESS)
        throw std::runtime_error("Failed to create HZB mip view");
      hzbMipViews[i][mip] = ImageViewHandle(dev, mipView);
    }
  }

  {
    VkCommandBuffer cmd =
        beginCommandBuffer(dev, device->getGraphicsCommandPool());
    std::vector<VkImageMemoryBarrier> initBarriers;
    initBarriers.reserve(hzbImages.size());
    for (const AllocatedImage &image : hzbImages) {
      VkImageMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      barrier.srcAccessMask = 0;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = image.get();
      barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, hzbMipCount,
                                  0, 1};
      initBarriers.push_back(barrier);
    }
    recordImageBarriers2(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         static_cast<uint32_t>(initBarriers.size()),
                         initBarriers.data());
    endAndSubmitCommandBuffer(dev, device->getGraphicsCommandPool(),
                              device->getGraphicsQueue(), cmd);
  }

  VkSamplerCreateInfo hzbSamplerCI{};
  hzbSamplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  hzbSamplerCI.magFilter = VK_FILTER_NEAREST;
  hzbSamplerCI.minFilter = VK_FILTER_NEAREST;
  hzbSamplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  hzbSamplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  hzbSamplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  hzbSamplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  hzbSamplerCI.maxLod = static_cast<float>(hzbMipCount);
  if (vkCreateSampler(dev, &hzbSamplerCI, nullptr, &hzbSampler) != VK_SUCCESS)
    throw std::runtime_error("Failed to create HZB sampler");

  std::array<VkDescriptorPoolSize, 2> poolSizes{};
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSizes[0].descriptorCount = static_cast<uint32_t>(swapCount * 7);
  poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  poolSizes[1].descriptorCount = static_cast<uint32_t>(swapCount);
  VkDescriptorPoolCreateInfo poolCI{};
  poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolCI.maxSets = static_cast<uint32_t>(swapCount);
  poolCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolCI.pPoolSizes = poolSizes.data();
  if (vkCreateDescriptorPool(dev, &poolCI, nullptr, &descriptorPool) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create GPU-driven descriptor pool");

  const uint32_t hzbSetCount = static_cast<uint32_t>(swapCount * hzbMipCount);
  std::array<VkDescriptorPoolSize, 2> hzbPoolSizes{};
  hzbPoolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  hzbPoolSizes[0].descriptorCount = hzbSetCount;
  hzbPoolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  hzbPoolSizes[1].descriptorCount = hzbSetCount;
  VkDescriptorPoolCreateInfo hzbPoolCI{};
  hzbPoolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  hzbPoolCI.maxSets = hzbSetCount;
  hzbPoolCI.poolSizeCount = static_cast<uint32_t>(hzbPoolSizes.size());
  hzbPoolCI.pPoolSizes = hzbPoolSizes.data();
  if (vkCreateDescriptorPool(dev, &hzbPoolCI, nullptr,
                             &hzbBuildDescriptorPool) != VK_SUCCESS)
    throw std::runtime_error("Failed to create HZB build descriptor pool");

  descriptorSets.resize(swapCount);
  std::vector<VkDescriptorSetLayout> setLayouts(swapCount, setLayout);
  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descriptorPool;
  allocInfo.descriptorSetCount = static_cast<uint32_t>(swapCount);
  allocInfo.pSetLayouts = setLayouts.data();
  if (vkAllocateDescriptorSets(dev, &allocInfo, descriptorSets.data()) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to allocate GPU-driven descriptor sets");

  hzbBuildSets.resize(hzbSetCount);
  std::vector<VkDescriptorSetLayout> hzbSetLayouts(hzbSetCount,
                                                   hzbBuildSetLayout);
  VkDescriptorSetAllocateInfo hzbAllocInfo{};
  hzbAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  hzbAllocInfo.descriptorPool = hzbBuildDescriptorPool;
  hzbAllocInfo.descriptorSetCount = hzbSetCount;
  hzbAllocInfo.pSetLayouts = hzbSetLayouts.data();
  if (vkAllocateDescriptorSets(dev, &hzbAllocInfo, hzbBuildSets.data()) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to allocate HZB build descriptor sets");

  for (FrameResources &frame : frames) {
    ensureBuffer(frame.meshBuffer, frame.meshBufferSize,
                 sizeof(GpuDrivenMeshRecord), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureBuffer(frame.transformBuffer, frame.transformBufferSize,
                 sizeof(GpuDrivenTransformRecord),
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureBuffer(frame.indirectBuffer, frame.indirectBufferSize,
                 sizeof(VkDrawIndexedIndirectCommand),
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                 VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                 RENDER_DEVICE_ALLOCATION_FLAGS);
    ensureBuffer(frame.noCullIndirectBuffer, frame.noCullIndirectBufferSize,
                 sizeof(VkDrawIndexedIndirectCommand),
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                 VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                 RENDER_DEVICE_ALLOCATION_FLAGS);
    createBuffer(device->getAllocator(), sizeof(uint32_t),
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                 RENDER_DEVICE_ALLOCATION_FLAGS, &frame.countBuffer);
    createBuffer(device->getAllocator(), sizeof(uint32_t),
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                 RENDER_DEVICE_ALLOCATION_FLAGS, &frame.noCullCountBuffer);
    createBuffer(device->getAllocator(), sizeof(GpuDrivenFrustumData),
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                 &frame.frustumBuffer);
    frame.descriptorDirty = true;
  }

  for (size_t image = 0; image < swapCount; ++image) {
    for (uint32_t mip = 0; mip < hzbMipCount; ++mip) {
      const size_t setIndex = image * hzbMipCount + mip;
      VkDescriptorImageInfo srcInfo{};
      srcInfo.imageLayout = (mip == 0)
                                ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      srcInfo.sampler = hzbSampler;
      srcInfo.imageView =
          (mip == 0) ? gBufferDepthViews[image].get()
                     : hzbMipViews[image][mip - 1].get();
      VkDescriptorImageInfo dstInfo{};
      dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      dstInfo.imageView = hzbMipViews[image][mip].get();
      std::array<VkWriteDescriptorSet, 2> writes{};
      writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[0].dstSet = hzbBuildSets[setIndex];
      writes[0].dstBinding = 0;
      writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      writes[0].descriptorCount = 1;
      writes[0].pImageInfo = &srcInfo;
      writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[1].dstSet = hzbBuildSets[setIndex];
      writes[1].dstBinding = 1;
      writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      writes[1].descriptorCount = 1;
      writes[1].pImageInfo = &dstInfo;
      vkUpdateDescriptorSets(dev, static_cast<uint32_t>(writes.size()),
                             writes.data(), 0, nullptr);
    }
  }
  for (uint32_t image = 0; image < static_cast<uint32_t>(swapCount); ++image)
    updateDescriptorSet(image);

  {
    auto compCode = readFile("../Shaders/gpu_cull.comp.spv");
    VkShaderModule compMod = createLocalShaderModule(dev, compCode);
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = compMod;
    stage.pName = "main";

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &setLayout;
    if (vkCreatePipelineLayout(dev, &layoutInfo, nullptr,
                               &cullPipelineLayout) != VK_SUCCESS)
      throw std::runtime_error("Failed to create GPU cull pipeline layout");

    VkComputePipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeInfo.stage = stage;
    pipeInfo.layout = cullPipelineLayout;
    if (vkCreateComputePipelines(dev, pipelineCache, 1, &pipeInfo, nullptr,
                                 &cullPipeline) != VK_SUCCESS)
      throw std::runtime_error("Failed to create GPU cull pipeline");
    vkDestroyShaderModule(dev, compMod, nullptr);
  }

  {
    auto compCode = readFile("../Shaders/hzb_downsample.comp.spv");
    VkShaderModule compMod = createLocalShaderModule(dev, compCode);
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = compMod;
    stage.pName = "main";

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &hzbBuildSetLayout;
    if (vkCreatePipelineLayout(dev, &layoutInfo, nullptr,
                               &hzbBuildPipelineLayout) != VK_SUCCESS)
      throw std::runtime_error("Failed to create HZB build pipeline layout");

    VkComputePipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeInfo.stage = stage;
    pipeInfo.layout = hzbBuildPipelineLayout;
    if (vkCreateComputePipelines(dev, pipelineCache, 1, &pipeInfo, nullptr,
                                 &hzbBuildPipeline) != VK_SUCCESS)
      throw std::runtime_error("Failed to create HZB build pipeline");
    vkDestroyShaderModule(dev, compMod, nullptr);
  }

  {
    auto vertCode = readFile("../Shaders/shader_gpu.vert.spv");
    auto fragCode = readFile("../Shaders/shader_gpu.frag.spv");
    VkShaderModule vertMod = createLocalShaderModule(dev, vertCode);
    VkShaderModule fragMod = createLocalShaderModule(dev, fragCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(Vertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    std::array<VkVertexInputAttributeDescription, 6> attrs{};
    attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)};
    attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, col)};
    attrs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, tex)};
    attrs[3] = {3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};
    attrs[4] = {4, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, tangent)};
    attrs[5] = {5, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, bitangent)};
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attrs.size());
    vertexInput.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp = {0, 0, static_cast<float>(extent.width),
                     static_cast<float>(extent.height), 0, 1};
    VkRect2D sc = {{0, 0}, extent};
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &vp;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &sc;

    std::array<VkDynamicState, 2> dynStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                               VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynState{};
    dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynState.dynamicStateCount = static_cast<uint32_t>(dynStates.size());
    dynState.pDynamicStates = dynStates.data();

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_BACK_BIT;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo msaa{};
    msaa.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{};
    depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blendOff{};
    blendOff.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                              VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT |
                              VK_COLOR_COMPONENT_A_BIT;
    std::array<VkPipelineColorBlendAttachmentState, 3> blends = {
        blendOff, blendOff, blendOff};
    VkPipelineColorBlendStateCreateInfo blendState{};
    blendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blendState.attachmentCount = static_cast<uint32_t>(blends.size());
    blendState.pAttachments = blends.data();

    std::array<VkDescriptorSetLayout, 3> graphicsSetLayouts = {
        descriptorManager.getVPLayout(), descriptorManager.getBindlessLayout(),
        setLayout};
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount =
        static_cast<uint32_t>(graphicsSetLayouts.size());
    layoutInfo.pSetLayouts = graphicsSetLayouts.data();
    if (vkCreatePipelineLayout(dev, &layoutInfo, nullptr,
                               &gBufferPipelineLayout) != VK_SUCCESS)
      throw std::runtime_error("Failed to create GPU-driven G-buffer layout");

    VkGraphicsPipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.stageCount = 2;
    pipeInfo.pStages = stages;
    pipeInfo.pVertexInputState = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAssembly;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &raster;
    pipeInfo.pMultisampleState = &msaa;
    pipeInfo.pDepthStencilState = &depth;
    pipeInfo.pColorBlendState = &blendState;
    pipeInfo.pDynamicState = &dynState;
    pipeInfo.layout = gBufferPipelineLayout;
    pipeInfo.renderPass = gBufferRenderPass;
    pipeInfo.subpass = 0;
    if (vkCreateGraphicsPipelines(dev, pipelineCache, 1, &pipeInfo, nullptr,
                                  &gBufferPipeline) != VK_SUCCESS)
      throw std::runtime_error("Failed to create GPU-driven G-buffer pipeline");

    raster.cullMode = VK_CULL_MODE_NONE;
    if (vkCreateGraphicsPipelines(dev, pipelineCache, 1, &pipeInfo, nullptr,
                                  &gBufferNoCullPipeline) != VK_SUCCESS)
      throw std::runtime_error(
          "Failed to create no-cull GPU-driven G-buffer pipeline");

    vkDestroyShaderModule(dev, fragMod, nullptr);
    vkDestroyShaderModule(dev, vertMod, nullptr);
  }

  for (int modelId : registeredModelIds)
    registerModelGeometryInternal(modelId, modelManager);
}

void GpuDrivenGBufferPass::cleanup() {
  if (!device)
    return;

  VkDevice dev = device->getLogicalDevice();
  if (gBufferPipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(dev, gBufferPipeline, nullptr);
  if (gBufferNoCullPipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(dev, gBufferNoCullPipeline, nullptr);
  if (gBufferPipelineLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(dev, gBufferPipelineLayout, nullptr);
  if (hzbBuildPipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(dev, hzbBuildPipeline, nullptr);
  if (hzbBuildPipelineLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(dev, hzbBuildPipelineLayout, nullptr);
  if (cullPipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(dev, cullPipeline, nullptr);
  if (cullPipelineLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(dev, cullPipelineLayout, nullptr);
  if (hzbSampler != VK_NULL_HANDLE)
    vkDestroySampler(dev, hzbSampler, nullptr);
  if (hzbBuildDescriptorPool != VK_NULL_HANDLE)
    vkDestroyDescriptorPool(dev, hzbBuildDescriptorPool, nullptr);
  if (descriptorPool != VK_NULL_HANDLE)
    vkDestroyDescriptorPool(dev, descriptorPool, nullptr);
  if (hzbBuildSetLayout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(dev, hzbBuildSetLayout, nullptr);
  if (setLayout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(dev, setLayout, nullptr);

  gBufferPipeline = VK_NULL_HANDLE;
  gBufferNoCullPipeline = VK_NULL_HANDLE;
  gBufferPipelineLayout = VK_NULL_HANDLE;
  hzbBuildPipeline = VK_NULL_HANDLE;
  hzbBuildPipelineLayout = VK_NULL_HANDLE;
  cullPipeline = VK_NULL_HANDLE;
  cullPipelineLayout = VK_NULL_HANDLE;
  hzbSampler = VK_NULL_HANDLE;
  hzbBuildDescriptorPool = VK_NULL_HANDLE;
  descriptorPool = VK_NULL_HANDLE;
  hzbBuildSetLayout = VK_NULL_HANDLE;
  setLayout = VK_NULL_HANDLE;

  descriptorSets.clear();
  hzbBuildSets.clear();
  hzbMipViews.clear();
  hzbViews.clear();
  hzbImages.clear();
  hzbValid.clear();

  frames.clear();
  staticVertexBuffer.reset();
  staticIndexBuffer.reset();
  staticVertexBufferSize = 0;
  staticIndexBufferSize = 0;
  staticVertices.clear();
  staticIndices.clear();
  geometryRanges.clear();
  geometryRangeLookup.clear();
  renderExtent = {};
  hzbExtent = {};
  hzbMipCount = 0;
  hzbFormat = VK_FORMAT_UNDEFINED;
  candidateCountValue = 0;
  meshCountValue = 0;
  lastFrameUsedValue = false;
  device = nullptr;
}

void GpuDrivenGBufferPass::updateDescriptorSet(uint32_t imageIndex) {
  if (!device || imageIndex >= descriptorSets.size() ||
      imageIndex >= frames.size() || hzbViews.size() != descriptorSets.size())
    return;

  FrameResources &frame = frames[imageIndex];
  if (!frame.meshBuffer || !frame.transformBuffer || !frame.indirectBuffer ||
      !frame.noCullIndirectBuffer || !frame.countBuffer ||
      !frame.noCullCountBuffer || !frame.frustumBuffer)
    return;
  if (!frame.descriptorDirty)
    return;

  std::array<VkDescriptorBufferInfo, 7> infos{};
  infos[0] = {frame.meshBuffer.get(), 0, VK_WHOLE_SIZE};
  infos[1] = {frame.frustumBuffer.get(), 0, VK_WHOLE_SIZE};
  infos[2] = {frame.transformBuffer.get(), 0, VK_WHOLE_SIZE};
  infos[3] = {frame.indirectBuffer.get(), 0, VK_WHOLE_SIZE};
  infos[4] = {frame.countBuffer.get(), 0, VK_WHOLE_SIZE};
  infos[5] = {frame.noCullIndirectBuffer.get(), 0, VK_WHOLE_SIZE};
  infos[6] = {frame.noCullCountBuffer.get(), 0, VK_WHOLE_SIZE};

  VkDescriptorImageInfo hzbInfo{};
  hzbInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  hzbInfo.imageView = hzbViews[imageIndex].get();
  hzbInfo.sampler = hzbSampler;

  std::array<VkWriteDescriptorSet, 8> writes{};
  const std::array<uint32_t, 7> bufferBindings = {0, 1, 2, 3, 4, 6, 7};
  for (uint32_t i = 0; i < static_cast<uint32_t>(bufferBindings.size()); ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = descriptorSets[imageIndex];
    writes[i].dstBinding = bufferBindings[i];
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].descriptorCount = 1;
    writes[i].pBufferInfo = &infos[i];
  }
  writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[7].dstSet = descriptorSets[imageIndex];
  writes[7].dstBinding = 5;
  writes[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  writes[7].descriptorCount = 1;
  writes[7].pImageInfo = &hzbInfo;
  vkUpdateDescriptorSets(device->getLogicalDevice(),
                         static_cast<uint32_t>(writes.size()), writes.data(),
                         0, nullptr);
  frame.descriptorDirty = false;
}

void GpuDrivenGBufferPass::beginFrame(uint32_t candidateCount) {
  candidateCountValue = candidateCount;
  meshCountValue = 0;
  lastFrameUsedValue = false;
}

bool GpuDrivenGBufferPass::prepareFrame(
    uint32_t imageIndex, const std::vector<GBufferDrawItem> &drawItems,
    const glm::vec4 frustumPlanes[6], const glm::mat4 &viewProj, bool enabled,
    bool hzbEnabled, uint32_t minCandidates) {
  if (!device || !enabled || imageIndex >= frames.size() ||
      candidateCountValue < minCandidates || cullPipeline == VK_NULL_HANDLE ||
      gBufferPipeline == VK_NULL_HANDLE ||
      gBufferNoCullPipeline == VK_NULL_HANDLE ||
      imageIndex >= descriptorSets.size() || !hasStaticGeometry()) {
    meshCountValue = 0;
    return false;
  }

  FrameResources &frame = frames[imageIndex];
  // Persistent scratch: cleared, not reallocated, every frame.
  std::vector<GpuDrivenMeshRecord> &meshRecords = frameMeshRecords;
  std::vector<GpuDrivenTransformRecord> &transforms = frameTransforms;
  meshRecords.clear();
  transforms.clear();
  meshRecords.reserve(drawItems.size());
  transforms.reserve(drawItems.size());

  for (const GBufferDrawItem &item : drawItems) {
    const GeometryRange *range = findGeometry(item.mesh, item.lod);
    if (!range)
      continue;

    uint32_t baseInstance = static_cast<uint32_t>(transforms.size());
    if (item.kind == GBufferDrawKind::Instanced && item.instances) {
      for (const InstanceData &instance : *item.instances) {
        GpuDrivenTransformRecord tr{};
        tr.model = instance.model;
        tr.normal = instance.normal;
        tr.texIdx0 = item.push.texIdx0;
        tr.texIdx1 = item.push.texIdx1;
        transforms.push_back(tr);
      }
    } else {
      GpuDrivenTransformRecord tr{};
      tr.model = item.push.model;
      tr.normal = item.push.normal;
      tr.texIdx0 = item.push.texIdx0;
      tr.texIdx1 = item.push.texIdx1;
      transforms.push_back(tr);
    }

    GpuDrivenMeshRecord record{};
    record.aabbMin = glm::vec4(item.aabbMin, 0.0f);
    record.aabbMax = glm::vec4(item.aabbMax, 0.0f);
    record.draw = glm::uvec4(range->firstIndex, range->indexCount,
                             static_cast<uint32_t>(range->vertexOffset),
                             baseInstance);
    record.flags = glm::uvec4(item.instanceCount,
                              static_cast<uint32_t>(item.lod),
                              static_cast<uint32_t>(item.kind),
                              static_cast<uint32_t>(item.cullMode));
    meshRecords.push_back(record);
  }

  const VkDeviceSize transformBytes = static_cast<VkDeviceSize>(
      transforms.size() * sizeof(GpuDrivenTransformRecord));
  const VkDeviceSize indirectBytes = static_cast<VkDeviceSize>(
      std::max<size_t>(1, meshRecords.size()) *
      sizeof(VkDrawIndexedIndirectCommand));

  if (ensureBuffer(frame.transformBuffer, frame.transformBufferSize,
                   transformBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
    frame.descriptorDirty = true;
  }
  if (ensureBuffer(frame.indirectBuffer, frame.indirectBufferSize,
                   indirectBytes,
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                       VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                   VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                   RENDER_DEVICE_ALLOCATION_FLAGS)) {
    frame.descriptorDirty = true;
  }
  if (ensureBuffer(frame.noCullIndirectBuffer,
                   frame.noCullIndirectBufferSize, indirectBytes,
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                       VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                   VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                   RENDER_DEVICE_ALLOCATION_FLAGS)) {
    frame.descriptorDirty = true;
  }

  auto upload = [&](AllocatedBuffer &buffer, const void *src,
                    VkDeviceSize bytes) {
    if (bytes == 0 || src == nullptr)
      return;
    uploadToAllocation(device->getAllocator(), buffer.getAllocation(), src,
                       bytes);
  };
  upload(frame.transformBuffer, transforms.data(), transformBytes);

  uploadMeshRecords(
      frame, meshRecords.data(),
      static_cast<VkDeviceSize>(meshRecords.size() *
                                sizeof(GpuDrivenMeshRecord)),
      static_cast<uint32_t>(meshRecords.size()));

  GpuDrivenFrustumData frustumData{};
  for (int i = 0; i < 6; ++i)
    frustumData.planes[i] = frustumPlanes[i];
  frustumData.viewProj = viewProj;
  const bool hzbReady = hzbEnabled && imageIndex < hzbValid.size() &&
                        hzbValid[imageIndex];
  frustumData.hzbParams = glm::vec4(
      static_cast<float>(hzbExtent.width), static_cast<float>(hzbExtent.height),
      static_cast<float>(hzbMipCount), hzbReady ? 1.0f : 0.0f);
  frustumData.meshCount = meshCountValue;
  upload(frame.frustumBuffer, &frustumData, sizeof(frustumData));
  updateDescriptorSet(imageIndex);

  return meshCountValue > 0;
}

bool GpuDrivenGBufferPass::canRecordIndirect(uint32_t imageIndex, bool enabled,
                                             uint32_t minCandidates) const {
  return device && enabled && imageIndex < frames.size() &&
         imageIndex < descriptorSets.size() && meshCountValue > 0 &&
         candidateCountValue >= minCandidates && hasStaticGeometry() &&
         cullPipeline != VK_NULL_HANDLE && cullPipelineLayout != VK_NULL_HANDLE &&
         gBufferPipeline != VK_NULL_HANDLE &&
         gBufferNoCullPipeline != VK_NULL_HANDLE &&
         gBufferPipelineLayout != VK_NULL_HANDLE;
}

void GpuDrivenGBufferPass::recordIndirectGBuffer(
    VkCommandBuffer cmd, uint32_t imageIndex,
    const VkRenderPassBeginInfo &renderPassInfo, const VkViewport &viewport,
    const VkRect2D &scissor, VkDescriptorSet vpSet,
    VkDescriptorSet bindlessSet) {
  if (!canRecordIndirect(imageIndex, true, 0))
    return;

  FrameResources &frame = frames[imageIndex];
  lastFrameUsedValue = true;

  vkdbgBeginLabel(cmd, "GPU Cull + Indirect Build", 0.8f, 0.25f, 1.0f);
  vkCmdFillBuffer(cmd, frame.countBuffer.get(), 0, sizeof(uint32_t), 0);
  vkCmdFillBuffer(cmd, frame.noCullCountBuffer.get(), 0, sizeof(uint32_t), 0);
  std::array<VkBufferMemoryBarrier, 2> countClearBarriers{};
  for (VkBufferMemoryBarrier &barrier : countClearBarriers) {
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.offset = 0;
    barrier.size = sizeof(uint32_t);
  }
  countClearBarriers[0].buffer = frame.countBuffer.get();
  countClearBarriers[1].buffer = frame.noCullCountBuffer.get();
  recordBufferBarriers2(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        static_cast<uint32_t>(countClearBarriers.size()),
                        countClearBarriers.data());

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cullPipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          cullPipelineLayout, 0, 1,
                          &descriptorSets[imageIndex], 0, nullptr);
  vkCmdDispatch(cmd, (meshCountValue + 63) / 64, 1, 1);

  std::array<VkBufferMemoryBarrier, 4> cullBarriers{};
  cullBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  cullBarriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  cullBarriers[0].dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  cullBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  cullBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  cullBarriers[0].buffer = frame.indirectBuffer.get();
  cullBarriers[0].offset = 0;
  cullBarriers[0].size = VK_WHOLE_SIZE;
  cullBarriers[1] = cullBarriers[0];
  cullBarriers[1].buffer = frame.countBuffer.get();
  cullBarriers[1].size = sizeof(uint32_t);
  cullBarriers[2] = cullBarriers[0];
  cullBarriers[2].buffer = frame.noCullIndirectBuffer.get();
  cullBarriers[3] = cullBarriers[0];
  cullBarriers[3].buffer = frame.noCullCountBuffer.get();
  cullBarriers[3].size = sizeof(uint32_t);
  recordBufferBarriers2(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                        static_cast<uint32_t>(cullBarriers.size()),
                        cullBarriers.data());
  vkdbgEndLabel(cmd);

  vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  vkCmdSetScissor(cmd, 0, 1, &scissor);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gBufferPipeline);
  std::array<VkDescriptorSet, 3> gpuSets = {
      vpSet, bindlessSet, descriptorSets[imageIndex]};
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          gBufferPipelineLayout, 0,
                          static_cast<uint32_t>(gpuSets.size()),
                          gpuSets.data(), 0, nullptr);
  VkBuffer vb = staticVertexBuffer.get();
  VkDeviceSize vbOffset = 0;
  vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vbOffset);
  vkCmdBindIndexBuffer(cmd, staticIndexBuffer.get(), 0, VK_INDEX_TYPE_UINT32);
  vkCmdDrawIndexedIndirectCount(
      cmd, frame.indirectBuffer.get(), 0, frame.countBuffer.get(), 0,
      meshCountValue, sizeof(VkDrawIndexedIndirectCommand));
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    gBufferNoCullPipeline);
  vkCmdDrawIndexedIndirectCount(
      cmd, frame.noCullIndirectBuffer.get(), 0,
      frame.noCullCountBuffer.get(), 0, meshCountValue,
      sizeof(VkDrawIndexedIndirectCommand));
  vkCmdEndRenderPass(cmd);
}

void GpuDrivenGBufferPass::recordHzbBuild(
    VkCommandBuffer cmd, uint32_t imageIndex,
    const std::vector<AllocatedImage> &gBufferDepthImages,
    VkFormat gBufferDepthFormat, bool enabled, bool hzbEnabled,
    uint32_t minCandidates) {
  if (!enabled || !hzbEnabled || candidateCountValue == 0 ||
      candidateCountValue < minCandidates) {
    if (imageIndex < hzbValid.size())
      hzbValid[imageIndex] = false;
    return;
  }

  if (!device || hzbBuildPipeline == VK_NULL_HANDLE ||
      imageIndex >= hzbImages.size() || imageIndex >= gBufferDepthImages.size() ||
      hzbMipCount == 0 || hzbBuildSets.empty())
    return;

  vkdbgBeginLabel(cmd, "Build HZB Depth Pyramid", 0.15f, 0.7f, 1.0f);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hzbBuildPipeline);

  VkImageMemoryBarrier depthReadBarrier{};
  depthReadBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  depthReadBarrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
  depthReadBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
  depthReadBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  depthReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  depthReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  depthReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  depthReadBarrier.image = gBufferDepthImages[imageIndex].get();
  depthReadBarrier.subresourceRange = {
      renderImageAspectMask(gBufferDepthFormat), 0, 1, 0, 1};
  recordImageBarrier2(cmd, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, depthReadBarrier);

  const bool valid = imageIndex < hzbValid.size() && hzbValid[imageIndex];
  for (uint32_t mip = 0; mip < hzbMipCount; ++mip) {
    VkImageMemoryBarrier toGeneral{};
    toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toGeneral.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneral.srcAccessMask = valid ? VK_ACCESS_SHADER_READ_BIT : 0;
    toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.image = hzbImages[imageIndex].get();
    toGeneral.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1};
    recordImageBarrier2(cmd,
                        valid ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                              : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, toGeneral);

    const size_t setIndex = imageIndex * hzbMipCount + mip;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            hzbBuildPipelineLayout, 0, 1,
                            &hzbBuildSets[setIndex], 0, nullptr);
    const uint32_t w = std::max(1u, hzbExtent.width >> mip);
    const uint32_t h = std::max(1u, hzbExtent.height >> mip);
    vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);

    VkImageMemoryBarrier toRead{};
    toRead.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toRead.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toRead.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    toRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toRead.image = hzbImages[imageIndex].get();
    toRead.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1};
    recordImageBarrier2(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, toRead);
  }

  if (imageIndex < hzbValid.size())
    hzbValid[imageIndex] = true;
  vkdbgEndLabel(cmd);
}
