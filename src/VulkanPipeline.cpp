#include "VulkanPipeline.h"
#include "Utilities.h"

#include <array>
#include <stdexcept>

void VulkanPipeline::createPipelines(VkDevice device, VkRenderPass renderPass,
                                     VkExtent2D extent,
                                     const DescriptorManager &descriptors) {
  // read in spirv shader bytecode
  auto vertShaderCode = readFile("../Shaders/shader.vert.spv");
  auto fragShaderCode = readFile("../Shaders/shader.frag.spv");

  // create shader module
  VkShaderModule vertShaderModule = createShaderModule(device, vertShaderCode);
  VkShaderModule fragShaderModule = createShaderModule(device, fragShaderCode);

  // vertex shader stage creation info
  VkPipelineShaderStageCreateInfo vertShaderCreateInfo = {};
  vertShaderCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertShaderCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT; // shader stage name
  vertShaderCreateInfo.module =
      vertShaderModule; // shader module containing code for shader stage
  vertShaderCreateInfo.pName = "main"; // entry point function

  // fragment shader stage creation info
  VkPipelineShaderStageCreateInfo fragShaderCreateInfo = {};
  fragShaderCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragShaderCreateInfo.stage =
      VK_SHADER_STAGE_FRAGMENT_BIT; // shader stage name
  fragShaderCreateInfo.module =
      fragShaderModule; // shader module containing code for shader stage
  fragShaderCreateInfo.pName = "main"; // entry point function

  // putting shader stage create info into array because graphics pipeline
  // create info takes in an array of shader
  VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderCreateInfo,
                                                    fragShaderCreateInfo};

  // how the data for a single vertex looks like as a whole
  VkVertexInputBindingDescription bindingDescription = {};
  bindingDescription.binding = 0; // can bind multiple streams of data
  bindingDescription.stride = sizeof(Vertex); // size of a single vertex
  bindingDescription.inputRate =
      VK_VERTEX_INPUT_RATE_VERTEX; // how to move to the next data entry after
                                   // each vertex

  // how the data for the attributes is described
  std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};
  // POSITION ATTRIBUTE
  attributeDescriptions[0].binding = 0;
  attributeDescriptions[0].location = 0;
  attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[0].offset = offsetof(Vertex, pos);

  // COLOR ATTRIBUTE
  attributeDescriptions[1].binding = 0;
  attributeDescriptions[1].location = 1;
  attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[1].offset = offsetof(Vertex, col);

  // TEXTURE COORDINATE ATTRIBUTE
  attributeDescriptions[2].binding = 0;
  attributeDescriptions[2].location = 2;
  attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
  attributeDescriptions[2].offset = offsetof(Vertex, tex);

  // vertex input stage creation info
  VkPipelineVertexInputStateCreateInfo vertexInputCreateInfo = {};
  vertexInputCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputCreateInfo.vertexBindingDescriptionCount = 1;
  vertexInputCreateInfo.pVertexBindingDescriptions = &bindingDescription;
  vertexInputCreateInfo.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(attributeDescriptions.size());
  vertexInputCreateInfo.pVertexAttributeDescriptions =
      attributeDescriptions.data();

  // input assembly stage creation info
  VkPipelineInputAssemblyStateCreateInfo inputAssemblyCreateInfo = {};
  inputAssemblyCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssemblyCreateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  inputAssemblyCreateInfo.primitiveRestartEnable = VK_FALSE;

  // viewport and scissor stage creation info
  VkViewport viewport = {};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = (float)extent.width;
  viewport.height = (float)extent.height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkRect2D scissor = {};
  scissor.offset = {0, 0};
  scissor.extent = extent;

  VkPipelineViewportStateCreateInfo viewportStateCreateInfo = {};
  viewportStateCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportStateCreateInfo.viewportCount = 1;
  viewportStateCreateInfo.pViewports = &viewport;
  viewportStateCreateInfo.scissorCount = 1;
  viewportStateCreateInfo.pScissors = &scissor;

  // rasterization stage creation info
  VkPipelineRasterizationStateCreateInfo rasterizerCreateInfo = {};
  rasterizerCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizerCreateInfo.depthClampEnable = VK_FALSE;
  rasterizerCreateInfo.rasterizerDiscardEnable = VK_FALSE;
  rasterizerCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizerCreateInfo.lineWidth = 1.0f;
  rasterizerCreateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizerCreateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizerCreateInfo.depthBiasEnable = VK_FALSE;

  // multisampling stage creation info
  VkPipelineMultisampleStateCreateInfo multisamplingCreateInfo = {};
  multisamplingCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisamplingCreateInfo.sampleShadingEnable = VK_FALSE;
  multisamplingCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  // color blending stage creation info
  VkPipelineColorBlendAttachmentState colorState = {};
  colorState.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorState.blendEnable = VK_TRUE;
  colorState.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  colorState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  colorState.colorBlendOp = VK_BLEND_OP_ADD;
  colorState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  colorState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  colorState.alphaBlendOp = VK_BLEND_OP_ADD;

  VkPipelineColorBlendStateCreateInfo colorBlendingCreateInfo = {};
  colorBlendingCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlendingCreateInfo.logicOpEnable = VK_FALSE;
  colorBlendingCreateInfo.attachmentCount = 1;
  colorBlendingCreateInfo.pAttachments = &colorState;

  // pipeline layout creation info
  VkDescriptorSetLayout vpLayout = descriptors.getVPLayout();
  VkDescriptorSetLayout samplerLayout = descriptors.getSamplerLayout();
  std::array<VkDescriptorSetLayout, 2> descriptorSetLayouts = {vpLayout,
                                                               samplerLayout};

  VkPushConstantRange pushConstantRange = descriptors.getPushConstantRange();

  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
  pipelineLayoutCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCreateInfo.setLayoutCount =
      static_cast<uint32_t>(descriptorSetLayouts.size());
  pipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayouts.data();
  pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
  pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

  // create pipeline layout
  VkResult result = vkCreatePipelineLayout(
      device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create pipeline layout");
  }

  // depth and stencil testing
  VkPipelineDepthStencilStateCreateInfo depthStencilCreateInfo = {};
  depthStencilCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencilCreateInfo.depthTestEnable = VK_TRUE;
  depthStencilCreateInfo.depthWriteEnable = VK_TRUE;
  depthStencilCreateInfo.depthCompareOp = VK_COMPARE_OP_LESS;
  depthStencilCreateInfo.depthBoundsTestEnable = VK_FALSE;
  depthStencilCreateInfo.stencilTestEnable = VK_FALSE;

  // graphics pipeline create info
  VkGraphicsPipelineCreateInfo pipelineCreateInfo = {};
  pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineCreateInfo.stageCount = 2;
  pipelineCreateInfo.pStages = shaderStages;
  pipelineCreateInfo.pVertexInputState = &vertexInputCreateInfo;
  pipelineCreateInfo.pInputAssemblyState = &inputAssemblyCreateInfo;
  pipelineCreateInfo.pViewportState = &viewportStateCreateInfo;
  pipelineCreateInfo.pDynamicState = nullptr;
  pipelineCreateInfo.pRasterizationState = &rasterizerCreateInfo;
  pipelineCreateInfo.pMultisampleState = &multisamplingCreateInfo;
  pipelineCreateInfo.pDepthStencilState = &depthStencilCreateInfo;
  pipelineCreateInfo.pColorBlendState = &colorBlendingCreateInfo;
  pipelineCreateInfo.layout = pipelineLayout;
  pipelineCreateInfo.renderPass = renderPass;
  pipelineCreateInfo.subpass = 0;

  pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
  pipelineCreateInfo.basePipelineIndex = -1;

  // create graphics pipeline
  result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
                                     &pipelineCreateInfo, nullptr,
                                     &graphicsPipeline);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create graphics pipeline");
  }

  // destroy shader modules that are no longer needed
  vkDestroyShaderModule(device, fragShaderModule, nullptr);
  vkDestroyShaderModule(device, vertShaderModule, nullptr);

  // CREATE SECOND GRAPHICS PIPELINE FOR SECOND SUBPASS
  auto secondVertShaderCode = readFile("../Shaders/second.vert.spv");
  auto secondFragShaderCode = readFile("../Shaders/second.frag.spv");

  // build shaders
  VkShaderModule secondVertShaderModule =
      createShaderModule(device, secondVertShaderCode);
  VkShaderModule secondFragShaderModule =
      createShaderModule(device, secondFragShaderCode);

  // vertex shader stage creation info
  vertShaderCreateInfo.module = secondVertShaderModule;

  // fragment shader stage creation info
  fragShaderCreateInfo.module = secondFragShaderModule;

  // create second graphics pipeline with mostly the same settings as the first
  VkPipelineShaderStageCreateInfo secondShaderStages[] = {vertShaderCreateInfo,
                                                          fragShaderCreateInfo};

  // no vertex data for second pass
  vertexInputCreateInfo.vertexBindingDescriptionCount = 0;
  vertexInputCreateInfo.pVertexBindingDescriptions = nullptr;
  vertexInputCreateInfo.vertexAttributeDescriptionCount = 0;
  vertexInputCreateInfo.pVertexAttributeDescriptions = nullptr;

  // disable depth testing and writing for second subpass
  depthStencilCreateInfo.depthWriteEnable = VK_FALSE;

  // create new pipeline layout
  VkDescriptorSetLayout inputLayout = descriptors.getInputLayout();
  VkPipelineLayoutCreateInfo secondPipelineLayoutCreateInfo = {};
  secondPipelineLayoutCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  secondPipelineLayoutCreateInfo.setLayoutCount = 1;
  secondPipelineLayoutCreateInfo.pSetLayouts = &inputLayout;
  secondPipelineLayoutCreateInfo.pushConstantRangeCount = 0;
  secondPipelineLayoutCreateInfo.pPushConstantRanges = nullptr;

  // create second pipeline layout
  result = vkCreatePipelineLayout(device, &secondPipelineLayoutCreateInfo,
                                  nullptr, &secondPipelineLayout);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create second pipeline layout");
  }

  // create second graphics pipeline
  pipelineCreateInfo.pStages = secondShaderStages;
  pipelineCreateInfo.layout = secondPipelineLayout;
  pipelineCreateInfo.subpass = 1;

  // create second pipeline
  result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
                                     &pipelineCreateInfo, nullptr,
                                     &secondPipeline);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create second graphics pipeline");
  }
  // destroy second shader modules
  vkDestroyShaderModule(device, secondFragShaderModule, nullptr);
  vkDestroyShaderModule(device, secondVertShaderModule, nullptr);
}

void VulkanPipeline::cleanup(VkDevice device) {
  vkDestroyPipeline(device, secondPipeline, nullptr);
  vkDestroyPipelineLayout(device, secondPipelineLayout, nullptr);
  vkDestroyPipeline(device, graphicsPipeline, nullptr);
  vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

VkShaderModule
VulkanPipeline::createShaderModule(VkDevice device,
                                   const std::vector<char> &code) {
  // shader module creation
  VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
  shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shaderModuleCreateInfo.codeSize = code.size();
  shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t *>(
      code.data()); // pointer to code data, need to be uint32_t pointer,
                    // so we cast from char pointer

  VkShaderModule shaderModule;
  VkResult result = vkCreateShaderModule(device, &shaderModuleCreateInfo,
                                         nullptr, &shaderModule);

  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create shader module");
  }

  return shaderModule;
}
