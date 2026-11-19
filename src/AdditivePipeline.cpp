#include "AdditivePipeline.h"

#include "Utilities.h"

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

VkShaderModule loadShaderModule(VkDevice device, const char *path) {
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

VkPipeline makeAdditivePipeline(VkDevice device, VkPipelineCache cache, VkRenderPass litPass,
                                VkPipelineLayout layout, const char *vertPath, const char *fragPath,
                                const VkPipelineVertexInputStateCreateInfo &vertexInput) {
  VkShaderModule vert = loadShaderModule(device, vertPath);
  VkShaderModule frag = loadShaderModule(device, fragPath);
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

