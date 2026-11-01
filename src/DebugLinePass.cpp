#include "DebugLinePass.h"

#include "VulkanDevice.h"

#include <algorithm>
#include <array>
#include <bit>
#include <stdexcept>

namespace {

struct LinePush {
  glm::mat4 viewProj;
  glm::vec2 invExtent;
  glm::vec2 nearFar;
};

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

} // namespace

void DebugLinePass::create(const VulkanDevice &device,
                           VkRenderPass imguiRenderPass,
                           VkPipelineCache cache) {
  VkDevice dev = device.getLogicalDevice();

  VkDescriptorSetLayoutBinding depthBinding = {};
  depthBinding.binding = 0;
  depthBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  depthBinding.descriptorCount = 1;
  depthBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  VkDescriptorSetLayoutCreateInfo setCI = {};
  setCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  setCI.bindingCount = 1;
  setCI.pBindings = &depthBinding;
  if (vkCreateDescriptorSetLayout(dev, &setCI, nullptr, &setLayout) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create debug line set layout");

  VkPushConstantRange push = {};
  push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  push.size = sizeof(LinePush);
  VkPipelineLayoutCreateInfo layoutCI = {};
  layoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutCI.setLayoutCount = 1;
  layoutCI.pSetLayouts = &setLayout;
  layoutCI.pushConstantRangeCount = 1;
  layoutCI.pPushConstantRanges = &push;
  if (vkCreatePipelineLayout(dev, &layoutCI, nullptr, &layout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create debug line pipeline layout");

  VkShaderModule vert = loadShader(dev, "../Shaders/debug_line.vert.spv");
  VkShaderModule frag = loadShader(dev, "../Shaders/debug_line.frag.spv");
  std::array<VkPipelineShaderStageCreateInfo, 2> stages = {};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vert;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = frag;
  stages[1].pName = "main";

  // PhysicsDebugVertex: vec3 position, then RGBA8 colour.
  VkVertexInputBindingDescription binding = {0, sizeof(PhysicsDebugVertex),
                                             VK_VERTEX_INPUT_RATE_VERTEX};
  std::array<VkVertexInputAttributeDescription, 2> attributes = {{
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT,
       offsetof(PhysicsDebugVertex, position)},
      {1, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(PhysicsDebugVertex, color)},
  }};
  VkPipelineVertexInputStateCreateInfo vertexInput = {};
  vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInput.vertexBindingDescriptionCount = 1;
  vertexInput.pVertexBindingDescriptions = &binding;
  vertexInput.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(attributes.size());
  vertexInput.pVertexAttributeDescriptions = attributes.data();

  VkPipelineInputAssemblyStateCreateInfo assembly = {};
  assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  assembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

  VkPipelineViewportStateCreateInfo viewport = {};
  viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport.viewportCount = 1;
  viewport.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo raster = {};
  raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.cullMode = VK_CULL_MODE_NONE;
  raster.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo multisample = {};
  multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState blendAttachment = {};
  blendAttachment.blendEnable = VK_TRUE;
  blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
  blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
  blendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo blend = {};
  blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  blend.attachmentCount = 1;
  blend.pAttachments = &blendAttachment;

  std::array<VkDynamicState, 2> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                                 VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic = {};
  dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
  dynamic.pDynamicStates = dynamicStates.data();

  VkGraphicsPipelineCreateInfo pipelineCI = {};
  pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineCI.stageCount = static_cast<uint32_t>(stages.size());
  pipelineCI.pStages = stages.data();
  pipelineCI.pVertexInputState = &vertexInput;
  pipelineCI.pInputAssemblyState = &assembly;
  pipelineCI.pViewportState = &viewport;
  pipelineCI.pRasterizationState = &raster;
  pipelineCI.pMultisampleState = &multisample;
  pipelineCI.pColorBlendState = &blend;
  pipelineCI.pDynamicState = &dynamic;
  pipelineCI.layout = layout;
  pipelineCI.renderPass = imguiRenderPass;
  pipelineCI.subpass = 0;
  const VkResult result =
      vkCreateGraphicsPipelines(dev, cache, 1, &pipelineCI, nullptr, &pipeline);
  vkDestroyShaderModule(dev, vert, nullptr);
  vkDestroyShaderModule(dev, frag, nullptr);
  if (result != VK_SUCCESS)
    throw std::runtime_error("Failed to create debug line pipeline");
}

void DebugLinePass::setDepthViews(VkDevice device,
                                  const std::vector<VkImageView> &views,
                                  VkSampler sampler) {
  if (pool != VK_NULL_HANDLE)
    vkDestroyDescriptorPool(device, pool, nullptr); // frees its sets too
  const uint32_t count = static_cast<uint32_t>(views.size());

  VkDescriptorPoolSize size = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                               count};
  VkDescriptorPoolCreateInfo poolCI = {};
  poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolCI.maxSets = count;
  poolCI.poolSizeCount = 1;
  poolCI.pPoolSizes = &size;
  if (vkCreateDescriptorPool(device, &poolCI, nullptr, &pool) != VK_SUCCESS)
    throw std::runtime_error("Failed to create debug line descriptor pool");

  std::vector<VkDescriptorSetLayout> layouts(count, setLayout);
  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = pool;
  allocInfo.descriptorSetCount = count;
  allocInfo.pSetLayouts = layouts.data();
  depthSets.resize(count);
  if (vkAllocateDescriptorSets(device, &allocInfo, depthSets.data()) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to allocate debug line descriptor sets");

  for (uint32_t i = 0; i < count; ++i) {
    // Same layout the composite pass samples it in, which it's still in
    // when the ImGui pass runs.
    VkDescriptorImageInfo image = {
        sampler, views[i], VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = depthSets[i];
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &image;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
  }

  // One vertex buffer per swap image, so a frame's lines never overwrite
  // ones the GPU is still drawing.
  vertexBuffers.resize(count);
  capacities.resize(count, 0);
  vertexCounts.resize(count, 0);
}

void DebugLinePass::upload(const VulkanDevice &device, uint32_t imageIndex,
                           const std::vector<PhysicsDebugVertex> &lines) {
  if (imageIndex >= vertexBuffers.size())
    return;
  const uint32_t count = static_cast<uint32_t>(lines.size());
  vertexCounts[imageIndex] = count;
  if (count == 0)
    return;
  if (count > capacities[imageIndex]) {
    // Grow in powers of two so a scene that keeps adding bodies doesn't
    // reallocate every frame.
    capacities[imageIndex] = std::bit_ceil(std::max(count, 4096u));
    createBuffer(device.getAllocator(),
                 capacities[imageIndex] * sizeof(PhysicsDebugVertex),
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                 &vertexBuffers[imageIndex]);
  }
  uploadToAllocation(device.getAllocator(),
                     vertexBuffers[imageIndex].getAllocation(), lines.data(),
                     count * sizeof(PhysicsDebugVertex));
}

void DebugLinePass::record(VkCommandBuffer cmd, uint32_t imageIndex,
                           VkExtent2D extent, const glm::mat4 &viewProj,
                           float nearPlane, float farPlane) const {
  if (imageIndex >= vertexCounts.size() || vertexCounts[imageIndex] == 0)
    return;
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
  const VkViewport viewport = {0.0f, 0.0f, float(extent.width),
                               float(extent.height), 0.0f, 1.0f};
  const VkRect2D scissor = {{0, 0}, extent};
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  vkCmdSetScissor(cmd, 0, 1, &scissor);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1,
                          &depthSets[imageIndex], 0, nullptr);
  const LinePush push = {viewProj,
                         glm::vec2(1.0f / extent.width, 1.0f / extent.height),
                         glm::vec2(nearPlane, farPlane)};
  vkCmdPushConstants(cmd, layout,
                     VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                     0, sizeof(push), &push);
  const VkBuffer buffer = vertexBuffers[imageIndex].get();
  const VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(cmd, 0, 1, &buffer, &offset);
  vkCmdDraw(cmd, vertexCounts[imageIndex], 1, 0, 0);
}

void DebugLinePass::cleanup(VkDevice device) {
  vertexBuffers.clear(); // AllocatedBuffer frees itself
  capacities.clear();
  vertexCounts.clear();
  if (pipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(device, pipeline, nullptr);
  if (layout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(device, layout, nullptr);
  if (pool != VK_NULL_HANDLE)
    vkDestroyDescriptorPool(device, pool, nullptr);
  if (setLayout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
  pipeline = VK_NULL_HANDLE;
  layout = VK_NULL_HANDLE;
  pool = VK_NULL_HANDLE;
  setLayout = VK_NULL_HANDLE;
  depthSets.clear();
}
