#include "ThumbnailRenderer.h"

#include "Model.h"
#include "VulkanDevice.h"

#include "stb_image.h"
#include "stb_image_write.h"

#include <spdlog/spdlog.h>

#include <array>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace {

constexpr VkFormat kColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr VkDeviceSize kPictureBytes = VkDeviceSize(kThumbnailSize) * kThumbnailSize * 4;
constexpr VkImageSubresourceRange kColorRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

struct ThumbnailPush {
  glm::mat4 viewProj;
  glm::vec4 light;
  glm::uvec4 texture; // albedo index, alpha masked
};

void makeImage(const VulkanDevice &device, uint32_t size, VkFormat format,
               VkImageUsageFlags usage, VkImageAspectFlags aspect, AllocatedImage &image,
               ImageViewHandle &view) {
  VkImageCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.format = format;
  ci.extent = {size, size, 1};
  ci.mipLevels = 1;
  ci.arrayLayers = 1;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = usage;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VmaAllocationCreateInfo alloc = {};
  alloc.usage = VMA_MEMORY_USAGE_AUTO;
  VkImage raw = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  if (vmaCreateImage(device.getAllocator(), &ci, &alloc, &raw, &allocation, nullptr) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create a thumbnail image");
  image = AllocatedImage(device.getAllocator(), raw, allocation);
  VkImageViewCreateInfo vi = {};
  vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  vi.image = raw;
  vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vi.format = format;
  vi.subresourceRange = {aspect, 0, 1, 0, 1};
  VkImageView rawView = VK_NULL_HANDLE;
  if (vkCreateImageView(device.getLogicalDevice(), &vi, nullptr, &rawView) != VK_SUCCESS)
    throw std::runtime_error("Failed to create a thumbnail image view");
  view = ImageViewHandle(device.getLogicalDevice(), rawView);
}

void barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to,
             VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
             VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
  VkImageMemoryBarrier2 b = {};
  b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
  b.srcStageMask = srcStage;
  b.srcAccessMask = srcAccess;
  b.dstStageMask = dstStage;
  b.dstAccessMask = dstAccess;
  b.oldLayout = from;
  b.newLayout = to;
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = image;
  b.subresourceRange = kColorRange;
  VkDependencyInfo dep = {};
  dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
  dep.imageMemoryBarrierCount = 1;
  dep.pImageMemoryBarriers = &b;
  vkCmdPipelineBarrier2(cmd, &dep);
}

VkShaderModule loadShader(VkDevice device, const char *path) {
  const std::vector<char> code = readFile(path);
  VkShaderModuleCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  ci.codeSize = code.size();
  ci.pCode = reinterpret_cast<const uint32_t *>(code.data());
  VkShaderModule module = VK_NULL_HANDLE;
  if (vkCreateShaderModule(device, &ci, nullptr, &module) != VK_SUCCESS)
    throw std::runtime_error(std::string("Failed to create shader ") + path);
  return module;
}

std::string picturePath(const std::string &folder, const std::string &key) {
  return (std::filesystem::path(folder) / (key + ".png")).string();
}

} // namespace

void ThumbnailRenderer::create(const VulkanDevice &device, VkPipelineCache cache,
                               VkDescriptorSetLayout bindlessLayout) {
  VkDevice dev = device.getLogicalDevice();
  makeImage(device, kThumbnailAtlasSize, kColorFormat,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, atlasImage, atlasImageView);
  makeImage(device, kThumbnailSize, kColorFormat,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, tileImage, tileView);
  makeImage(device, kThumbnailSize, kDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT, depthImage, depthView);
  createBuffer(device.getAllocator(), kPictureBytes,
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VMA_MEMORY_USAGE_AUTO,
               VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
               &staging);
  VmaAllocationInfo info = {};
  vmaGetAllocationInfo(device.getAllocator(), staging.getAllocation(), &info);
  stagingData = info.pMappedData;

  // The tile: cleared to see-through, drawn, then copied out.
  std::array<VkAttachmentDescription, 2> attachments = {};
  attachments[0].format = kColorFormat;
  attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
  attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  attachments[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  attachments[1].format = kDepthFormat;
  attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
  attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  VkAttachmentReference colorRef = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkAttachmentReference depthRef = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorRef;
  subpass.pDepthStencilAttachment = &depthRef;
  // The last picture's copy out of the tile, and the depth clear, before
  // this one draws; the drawing before this one's copy.
  std::array<VkSubpassDependency, 2> deps = {};
  deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
  deps[0].dstSubpass = 0;
  deps[0].srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  deps[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  deps[1].srcSubpass = 0;
  deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
  deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  deps[1].dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
  deps[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  VkRenderPassCreateInfo rp = {};
  rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  rp.attachmentCount = static_cast<uint32_t>(attachments.size());
  rp.pAttachments = attachments.data();
  rp.subpassCount = 1;
  rp.pSubpasses = &subpass;
  rp.dependencyCount = static_cast<uint32_t>(deps.size());
  rp.pDependencies = deps.data();
  if (vkCreateRenderPass(dev, &rp, nullptr, &renderPass) != VK_SUCCESS)
    throw std::runtime_error("Failed to create the thumbnail render pass");

  const std::array<VkImageView, 2> views = {tileView.get(), depthView.get()};
  VkFramebufferCreateInfo fb = {};
  fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  fb.renderPass = renderPass;
  fb.attachmentCount = static_cast<uint32_t>(views.size());
  fb.pAttachments = views.data();
  fb.width = kThumbnailSize;
  fb.height = kThumbnailSize;
  fb.layers = 1;
  if (vkCreateFramebuffer(dev, &fb, nullptr, &framebuffer) != VK_SUCCESS)
    throw std::runtime_error("Failed to create the thumbnail framebuffer");

  VkPushConstantRange push = {};
  push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  push.size = sizeof(ThumbnailPush);
  VkPipelineLayoutCreateInfo lci = {};
  lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  lci.setLayoutCount = 1;
  lci.pSetLayouts = &bindlessLayout;
  lci.pushConstantRangeCount = 1;
  lci.pPushConstantRanges = &push;
  if (vkCreatePipelineLayout(dev, &lci, nullptr, &layout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create the thumbnail pipeline layout");

  VkShaderModule vert = loadShader(dev, "../Shaders/thumbnail.vert.spv");
  VkShaderModule frag = loadShader(dev, "../Shaders/thumbnail.frag.spv");
  std::array<VkPipelineShaderStageCreateInfo, 2> stages = {};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vert;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = frag;
  stages[1].pName = "main";

  VkVertexInputBindingDescription binding = {0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
  const std::array<VkVertexInputAttributeDescription, 4> attrs = {{
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)},
      {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, col)},
      {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, tex)},
      {3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
  }};
  VkPipelineVertexInputStateCreateInfo vertexInput = {};
  vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInput.vertexBindingDescriptionCount = 1;
  vertexInput.pVertexBindingDescriptions = &binding;
  vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
  vertexInput.pVertexAttributeDescriptions = attrs.data();
  VkPipelineInputAssemblyStateCreateInfo assembly = {};
  assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkViewport viewport = {0, 0, float(kThumbnailSize), float(kThumbnailSize), 0, 1};
  VkRect2D scissor = {{0, 0}, {kThumbnailSize, kThumbnailSize}};
  VkPipelineViewportStateCreateInfo viewportState = {};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.pViewports = &viewport;
  viewportState.scissorCount = 1;
  viewportState.pScissors = &scissor;
  VkPipelineRasterizationStateCreateInfo raster = {};
  raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.cullMode = VK_CULL_MODE_NONE; // any winding, double-sided or not
  raster.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo multisample = {};
  multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineDepthStencilStateCreateInfo depth = {};
  depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depth.depthTestEnable = VK_TRUE;
  depth.depthWriteEnable = VK_TRUE;
  depth.depthCompareOp = VK_COMPARE_OP_LESS;
  VkPipelineColorBlendAttachmentState blendAttachment = {};
  blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                   VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo blend = {};
  blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  blend.attachmentCount = 1;
  blend.pAttachments = &blendAttachment;
  VkGraphicsPipelineCreateInfo pci = {};
  pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pci.stageCount = static_cast<uint32_t>(stages.size());
  pci.pStages = stages.data();
  pci.pVertexInputState = &vertexInput;
  pci.pInputAssemblyState = &assembly;
  pci.pViewportState = &viewportState;
  pci.pRasterizationState = &raster;
  pci.pMultisampleState = &multisample;
  pci.pDepthStencilState = &depth;
  pci.pColorBlendState = &blend;
  pci.layout = layout;
  pci.renderPass = renderPass;
  pci.subpass = 0;
  const VkResult made = vkCreateGraphicsPipelines(dev, cache, 1, &pci, nullptr, &pipeline);
  vkDestroyShaderModule(dev, vert, nullptr);
  vkDestroyShaderModule(dev, frag, nullptr);
  if (made != VK_SUCCESS)
    throw std::runtime_error("Failed to create the thumbnail pipeline");

  // The atlas starts see-through, ready to be sampled.
  VkCommandBuffer cmd = beginCommandBuffer(dev, device.getGraphicsCommandPool());
  barrier(cmd, atlasImage.get(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_CLEAR_BIT,
          VK_ACCESS_2_TRANSFER_WRITE_BIT);
  const VkClearColorValue clear = {{0.0f, 0.0f, 0.0f, 0.0f}};
  vkCmdClearColorImage(cmd, atlasImage.get(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1,
                       &kColorRange);
  barrier(cmd, atlasImage.get(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_CLEAR_BIT,
          VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
          VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  endAndSubmitCommandBuffer(dev, device.getGraphicsCommandPool(), device.getGraphicsQueue(), cmd);
}

void ThumbnailRenderer::cleanup(VkDevice device) {
  if (pipeline)
    vkDestroyPipeline(device, pipeline, nullptr);
  if (layout)
    vkDestroyPipelineLayout(device, layout, nullptr);
  if (framebuffer)
    vkDestroyFramebuffer(device, framebuffer, nullptr);
  if (renderPass)
    vkDestroyRenderPass(device, renderPass, nullptr);
  pipeline = VK_NULL_HANDLE;
  layout = VK_NULL_HANDLE;
  framebuffer = VK_NULL_HANDLE;
  renderPass = VK_NULL_HANDLE;
  staging.reset();
  stagingData = nullptr;
  depthView.reset();
  depthImage.reset();
  tileView.reset();
  tileImage.reset();
  atlasImageView.reset();
  atlasImage.reset();
  cells.clear();
  nextCell = 0;
}

std::optional<int> ThumbnailRenderer::cellOf(const std::string &key) const {
  const auto it = cells.find(key);
  if (it == cells.end())
    return std::nullopt;
  return it->second;
}

bool ThumbnailRenderer::full() const {
  return nextCell >= kThumbnailsPerRow * kThumbnailsPerRow;
}

int ThumbnailRenderer::takeCell(const std::string &key) {
  const int cell = nextCell++;
  cells[key] = cell;
  return cell;
}

// The picture into its cell: from the tile just drawn, or from the staging
// buffer (a cached PNG). The atlas leaves sampling for the copy and goes
// back after.
void ThumbnailRenderer::copyIntoAtlas(VkCommandBuffer cmd, int cell, bool fromTile) {
  barrier(cmd, atlasImage.get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
          VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
          VK_ACCESS_2_TRANSFER_WRITE_BIT);
  const VkOffset3D at = {(cell % kThumbnailsPerRow) * kThumbnailSize,
                         (cell / kThumbnailsPerRow) * kThumbnailSize, 0};
  const VkExtent3D size = {kThumbnailSize, kThumbnailSize, 1};
  if (fromTile) {
    VkImageCopy region = {};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstOffset = at;
    region.extent = size;
    vkCmdCopyImage(cmd, tileImage.get(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, atlasImage.get(),
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  } else {
    VkBufferImageCopy region = {};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset = at;
    region.imageExtent = size;
    vkCmdCopyBufferToImage(cmd, staging.get(), atlasImage.get(),
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  }
  barrier(cmd, atlasImage.get(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
          VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
          VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

bool ThumbnailRenderer::loadCached(const VulkanDevice &device, const std::string &key,
                                   const std::string &folder) {
  if (full() || cells.contains(key))
    return cells.contains(key);
  const std::string path = picturePath(folder, key);
  if (!std::filesystem::exists(path))
    return false;
  int width = 0, height = 0, channels = 0;
  stbi_uc *pixels = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
  if (!pixels)
    return false;
  const bool fits = width == kThumbnailSize && height == kThumbnailSize;
  if (fits)
    std::memcpy(stagingData, pixels, kPictureBytes);
  stbi_image_free(pixels);
  if (!fits)
    return false;
  VkDevice dev = device.getLogicalDevice();
  VkCommandBuffer cmd = beginCommandBuffer(dev, device.getGraphicsCommandPool());
  copyIntoAtlas(cmd, takeCell(key), /*fromTile=*/false);
  endAndSubmitCommandBuffer(dev, device.getGraphicsCommandPool(), device.getGraphicsQueue(), cmd);
  return true;
}

bool ThumbnailRenderer::render(const VulkanDevice &device, const std::string &key,
                               const MeshModel &model, glm::vec3 boundsMin, glm::vec3 boundsMax,
                               VkDescriptorSet bindlessSet, const std::string &folder) {
  if (full())
    return false;
  VkDevice dev = device.getLogicalDevice();
  VkCommandBuffer cmd = beginCommandBuffer(dev, device.getGraphicsCommandPool());

  std::array<VkClearValue, 2> clears = {};
  clears[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
  clears[1].depthStencil = {1.0f, 0};
  VkRenderPassBeginInfo begin = {};
  begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  begin.renderPass = renderPass;
  begin.framebuffer = framebuffer;
  begin.renderArea = {{0, 0}, {kThumbnailSize, kThumbnailSize}};
  begin.clearValueCount = static_cast<uint32_t>(clears.size());
  begin.pClearValues = clears.data();
  vkCmdBeginRenderPass(cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &bindlessSet, 0,
                          nullptr);
  ThumbnailPush push;
  push.viewProj = thumbnailViewProj(boundsMin, boundsMax);
  push.light = glm::vec4(thumbnailLightDirection(), 0.0f);
  for (size_t i = 0; i < model.getMeshCount(); ++i) {
    const Mesh *mesh = model.getMesh(i);
    if (!mesh || mesh->getIndexCount(0) == 0)
      continue;
    const Material &material = mesh->getMaterial();
    push.texture = glm::uvec4(static_cast<uint32_t>(material.albedoTextureId),
                              material.alphaMasked ? 1u : 0u, 0u, 0u);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof push, &push);
    const VkBuffer vertices = mesh->getVertexBuffer();
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertices, &offset);
    vkCmdBindIndexBuffer(cmd, mesh->getIndexBuffer(0), 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, static_cast<uint32_t>(mesh->getIndexCount(0)), 1, 0, 0, 0);
  }
  vkCmdEndRenderPass(cmd); // the tile is now ready to copy from

  const int cell = takeCell(key);
  copyIntoAtlas(cmd, cell, /*fromTile=*/true);
  VkBufferImageCopy out = {};
  out.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  out.imageExtent = {kThumbnailSize, kThumbnailSize, 1};
  vkCmdCopyImageToBuffer(cmd, tileImage.get(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         staging.get(), 1, &out);
  endAndSubmitCommandBuffer(dev, device.getGraphicsCommandPool(), device.getGraphicsQueue(), cmd);

  // Saved, so next time it's a file read instead of loading the model.
  std::error_code error;
  std::filesystem::create_directories(folder, error);
  const std::string path = picturePath(folder, key);
  if (!stbi_write_png(path.c_str(), kThumbnailSize, kThumbnailSize, 4, stagingData,
                      kThumbnailSize * 4))
    spdlog::warn("Couldn't save the thumbnail {}", path);
  return true;
}
