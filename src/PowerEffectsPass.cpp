#include "PowerEffectsPass.h"

#include "Model.h"
#include "Vertex.h"
#include "VulkanDevice.h"

#include <algorithm>
#include <array>
#include <bit>
#include <stdexcept>
#include <string>

namespace {

// Matches glow.vert and glow.frag.
struct GlowPush {
  glm::mat4 model;
  glm::mat4 normal;
  glm::vec4 colour; // rgb, strength
  glm::vec4 params; // time, how far the shell sits out along the normals, the floor
};

// Matches power_fx.frag.
struct StripPush {
  glm::vec4 params; // time
};

// How far the glow's shell sits out from the surface, m: in front of it, so
// it never fights the surface's own depth.
constexpr float kShellOffset = 0.012f;

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

VkPipelineLayout makeLayout(VkDevice device, VkDescriptorSetLayout sceneLayout, uint32_t pushSize) {
  VkPushConstantRange push = {};
  push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  push.size = pushSize;
  VkPipelineLayoutCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  ci.setLayoutCount = 1;
  ci.pSetLayouts = &sceneLayout;
  ci.pushConstantRangeCount = 1;
  ci.pPushConstantRanges = &push;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  if (vkCreatePipelineLayout(device, &ci, nullptr, &layout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create a power effects pipeline layout");
  return layout;
}

// Additive, depth tested (reversed Z) but not written, both faces, into the
// lit pass's subpass 1.
VkPipeline makePipeline(VkDevice device, VkPipelineCache cache, VkRenderPass litPass,
                        VkPipelineLayout layout, const char *vertPath, const char *fragPath,
                        const VkPipelineVertexInputStateCreateInfo &vertexInput) {
  VkShaderModule vert = loadShader(device, vertPath);
  VkShaderModule frag = loadShader(device, fragPath);
  std::array<VkPipelineShaderStageCreateInfo, 2> stages = {};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vert;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = frag;
  stages[1].pName = "main";

  VkPipelineInputAssemblyStateCreateInfo assembly = {};
  assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  VkPipelineViewportStateCreateInfo viewport = {};
  viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport.viewportCount = 1;
  viewport.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo raster = {};
  raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.cullMode = VK_CULL_MODE_NONE;
  raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  raster.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo multisample = {};
  multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo depth = {};
  depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depth.depthTestEnable = VK_TRUE;
  depth.depthWriteEnable = VK_FALSE;
  depth.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL; // reversed Z

  VkPipelineColorBlendAttachmentState add = {};
  add.blendEnable = VK_TRUE;
  add.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
  add.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
  add.colorBlendOp = VK_BLEND_OP_ADD;
  add.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  add.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  add.alphaBlendOp = VK_BLEND_OP_ADD;
  add.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                       VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo blend = {};
  blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  blend.attachmentCount = 1;
  blend.pAttachments = &add;

  std::array<VkDynamicState, 2> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                                 VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic = {};
  dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
  dynamic.pDynamicStates = dynamicStates.data();

  VkGraphicsPipelineCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  ci.stageCount = static_cast<uint32_t>(stages.size());
  ci.pStages = stages.data();
  ci.pVertexInputState = &vertexInput;
  ci.pInputAssemblyState = &assembly;
  ci.pViewportState = &viewport;
  ci.pRasterizationState = &raster;
  ci.pMultisampleState = &multisample;
  ci.pDepthStencilState = &depth;
  ci.pColorBlendState = &blend;
  ci.pDynamicState = &dynamic;
  ci.layout = layout;
  ci.renderPass = litPass;
  ci.subpass = 1;
  VkPipeline pipeline = VK_NULL_HANDLE;
  const VkResult result = vkCreateGraphicsPipelines(device, cache, 1, &ci, nullptr, &pipeline);
  vkDestroyShaderModule(device, vert, nullptr);
  vkDestroyShaderModule(device, frag, nullptr);
  if (result != VK_SUCCESS)
    throw std::runtime_error(std::string("Failed to create the pipeline for ") + fragPath);
  return pipeline;
}

} // namespace

void PowerEffectsPass::create(const VulkanDevice &device, VkRenderPass litPass,
                              VkDescriptorSetLayout sceneLayout, VkPipelineCache cache) {
  VkDevice dev = device.getLogicalDevice();
  glowLayout = makeLayout(dev, sceneLayout, sizeof(GlowPush));
  stripLayout = makeLayout(dev, sceneLayout, sizeof(StripPush));

  // The glow reads only a mesh's positions and normals.
  const VkVertexInputBindingDescription meshBinding = {0, sizeof(Vertex),
                                                       VK_VERTEX_INPUT_RATE_VERTEX};
  const std::array<VkVertexInputAttributeDescription, 2> meshAttributes = {{
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)},
      {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
  }};
  VkPipelineVertexInputStateCreateInfo meshInput = {};
  meshInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  meshInput.vertexBindingDescriptionCount = 1;
  meshInput.pVertexBindingDescriptions = &meshBinding;
  meshInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(meshAttributes.size());
  meshInput.pVertexAttributeDescriptions = meshAttributes.data();
  glowPipeline = makePipeline(dev, cache, litPass, glowLayout, "../Shaders/glow.vert.spv",
                              "../Shaders/glow.frag.spv", meshInput);

  const VkVertexInputBindingDescription stripBinding = {0, sizeof(EffectVertex),
                                                        VK_VERTEX_INPUT_RATE_VERTEX};
  const std::array<VkVertexInputAttributeDescription, 4> stripAttributes = {{
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(EffectVertex, position)},
      {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(EffectVertex, core)},
      {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(EffectVertex, edge)},
      {3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(EffectVertex, uv)},
  }};
  VkPipelineVertexInputStateCreateInfo stripInput = {};
  stripInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  stripInput.vertexBindingDescriptionCount = 1;
  stripInput.pVertexBindingDescriptions = &stripBinding;
  stripInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(stripAttributes.size());
  stripInput.pVertexAttributeDescriptions = stripAttributes.data();
  stripPipeline = makePipeline(dev, cache, litPass, stripLayout, "../Shaders/power_fx.vert.spv",
                               "../Shaders/power_fx.frag.spv", stripInput);
}

void PowerEffectsPass::upload(const VulkanDevice &device, uint32_t imageIndex,
                              const std::vector<EffectVertex> &strips) {
  if (imageIndex >= vertexBuffers.size()) {
    vertexBuffers.resize(imageIndex + 1);
    capacities.resize(imageIndex + 1, 0);
    vertexCounts.resize(imageIndex + 1, 0);
  }
  const uint32_t count = static_cast<uint32_t>(strips.size());
  vertexCounts[imageIndex] = count;
  if (count == 0)
    return;
  if (count > capacities[imageIndex]) {
    capacities[imageIndex] = std::bit_ceil(std::max(count, 2048u));
    createBuffer(device.getAllocator(), capacities[imageIndex] * sizeof(EffectVertex),
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                 &vertexBuffers[imageIndex]);
  }
  uploadToAllocation(device.getAllocator(), vertexBuffers[imageIndex].getAllocation(),
                     strips.data(), count * sizeof(EffectVertex));
}

void PowerEffectsPass::record(VkCommandBuffer cmd, uint32_t imageIndex, VkDescriptorSet sceneSet,
                              const VkViewport &viewport, const VkRect2D &scissor,
                              const std::vector<GlowDraw> &glows, float time) const {
  const bool strips = imageIndex < vertexCounts.size() && vertexCounts[imageIndex] > 0;
  if (glows.empty() && !strips)
    return;
  if (!glows.empty()) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, glowPipeline);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, glowLayout, 0, 1, &sceneSet, 0,
                            nullptr);
    for (const GlowDraw &g : glows) {
      if (!g.mesh || g.mesh->getIndexCount() == 0)
        continue;
      const GlowPush push = {g.model, g.normal, glm::vec4(g.colour, g.strength),
                             glm::vec4(time, kShellOffset, g.floor, 0.0f)};
      vkCmdPushConstants(cmd, glowLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                         0, sizeof(push), &push);
      const VkBuffer vertices = g.mesh->getVertexBuffer();
      const VkDeviceSize offset = 0;
      vkCmdBindVertexBuffers(cmd, 0, 1, &vertices, &offset);
      vkCmdBindIndexBuffer(cmd, g.mesh->getIndexBuffer(0), 0, VK_INDEX_TYPE_UINT32);
      vkCmdDrawIndexed(cmd, static_cast<uint32_t>(g.mesh->getIndexCount(0)), 1, 0, 0, 0);
    }
  }
  if (strips) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, stripPipeline);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, stripLayout, 0, 1, &sceneSet, 0,
                            nullptr);
    const StripPush push = {glm::vec4(time, 0.0f, 0.0f, 0.0f)};
    vkCmdPushConstants(cmd, stripLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(push), &push);
    const VkBuffer buffer = vertexBuffers[imageIndex].get();
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &buffer, &offset);
    vkCmdDraw(cmd, vertexCounts[imageIndex], 1, 0, 0);
  }
}

void PowerEffectsPass::cleanup(VkDevice device) {
  vertexBuffers.clear(); // AllocatedBuffer frees itself
  capacities.clear();
  vertexCounts.clear();
  for (VkPipeline *p : {&glowPipeline, &stripPipeline})
    if (*p != VK_NULL_HANDLE) {
      vkDestroyPipeline(device, *p, nullptr);
      *p = VK_NULL_HANDLE;
    }
  for (VkPipelineLayout *l : {&glowLayout, &stripLayout})
    if (*l != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(device, *l, nullptr);
      *l = VK_NULL_HANDLE;
    }
}
