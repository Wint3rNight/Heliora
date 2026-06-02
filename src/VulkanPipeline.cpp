#include "VulkanPipeline.h"
#include "Utilities.h"
#include <cstddef>

#include <array>
#include <stdexcept>

// ---------------------------------------------------------------------------
void VulkanPipeline::createPipelines(VkDevice device, VkRenderPass gBufferPass,
                                     VkRenderPass litPass,
                                     VkRenderPass compositionPass,
                                     VkRenderPass shadowPassRP,
                                     VkRenderPass ssgiPass,
                                     VkExtent2D extent,
                                     VkPipelineCache pipelineCache,
                                     const DescriptorManager &descriptors) {
  // ---- Common vertex input: full Vertex struct (6 attrs) ----
  VkVertexInputBindingDescription binding = {};
  binding.binding = 0;
  binding.stride = sizeof(Vertex);
  binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  std::array<VkVertexInputAttributeDescription, 6> attrs = {};
  attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)};
  attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, col)};
  attrs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, tex)};
  attrs[3] = {3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};
  attrs[4] = {4, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, tangent)};
  attrs[5] = {5, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, bitangent)};

  VkPipelineVertexInputStateCreateInfo vertexInput = {};
  vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInput.vertexBindingDescriptionCount = 1;
  vertexInput.pVertexBindingDescriptions = &binding;
  vertexInput.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(attrs.size());
  vertexInput.pVertexAttributeDescriptions = attrs.data();

  // ---- Common states ----
  VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
  inputAssembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  VkViewport vp = {0, 0, (float)extent.width, (float)extent.height, 0, 1};
  VkRect2D sc = {{0, 0}, extent};
  VkPipelineViewportStateCreateInfo vpState = {};
  vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  vpState.viewportCount = 1;
  vpState.pViewports = &vp;
  vpState.scissorCount = 1;
  vpState.pScissors = &sc;

  std::array<VkDynamicState, 2> dynStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                             VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynState = {};
  dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynState.dynamicStateCount = static_cast<uint32_t>(dynStates.size());
  dynState.pDynamicStates = dynStates.data();

  // Geometry pipelines additionally allow dynamic cull-mode flips, so the
  // G-Buffer pass can disable culling per-draw for materials marked
  // doubleSided (foliage in Sponza). Fullscreen-quad pipelines stay on the
  // viewport+scissor-only dynState above.
  std::array<VkDynamicState, 3> geoDynStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                                VK_DYNAMIC_STATE_SCISSOR,
                                                VK_DYNAMIC_STATE_CULL_MODE};
  VkPipelineDynamicStateCreateInfo geoDynState = {};
  geoDynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  geoDynState.dynamicStateCount = static_cast<uint32_t>(geoDynStates.size());
  geoDynState.pDynamicStates = geoDynStates.data();

  VkPipelineRasterizationStateCreateInfo rasterizer = {};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

  VkPipelineMultisampleStateCreateInfo msaa = {};
  msaa.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo depthStencil = {};
  depthStencil.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable = VK_TRUE;
  depthStencil.depthWriteEnable = VK_TRUE;
  depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

  // ---- Blend state for a single opaque attachment ----
  VkPipelineColorBlendAttachmentState blendOff = {};
  blendOff.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                            VK_COLOR_COMPONENT_G_BIT |
                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  blendOff.blendEnable = VK_FALSE;

  // 1. GEOMETRY / G-BUFFER PIPELINE  (shader.vert + shader.frag → 3 MRTs)
  auto vertCode = readFile("../Shaders/shader.vert.spv");
  auto fragCode = readFile("../Shaders/shader.frag.spv");
  VkShaderModule vertMod = createShaderModule(device, vertCode);
  VkShaderModule fragMod = createShaderModule(device, fragCode);

  VkPipelineShaderStageCreateInfo geoStages[2] = {};
  geoStages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  nullptr,
                  0,
                  VK_SHADER_STAGE_VERTEX_BIT,
                  vertMod,
                  "main",
                  nullptr};
  geoStages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  nullptr,
                  0,
                  VK_SHADER_STAGE_FRAGMENT_BIT,
                  fragMod,
                  "main",
                  nullptr};

  // 3 MRT blend states (one per G-buffer attachment)
  std::array<VkPipelineColorBlendAttachmentState, 3> gbBlend = {
      blendOff, blendOff, blendOff};
  VkPipelineColorBlendStateCreateInfo gbBlendState = {};
  gbBlendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  gbBlendState.attachmentCount = 3;
  gbBlendState.pAttachments = gbBlend.data();

  // Phase 7.2: G-buffer pipeline uses the bindless texture array at set 1.
  // Material indices are passed via push constants, and the fragment shader
  // indexes into the array with nonuniformEXT.
  VkDescriptorSetLayout geoLayouts[] = {descriptors.getVPLayout(),
                                        descriptors.getBindlessLayout()};
  VkPushConstantRange pcRange = descriptors.getPushConstantRange();

  VkPipelineLayoutCreateInfo geoLayoutCI = {};
  geoLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  geoLayoutCI.setLayoutCount = 2;
  geoLayoutCI.pSetLayouts = geoLayouts;
  geoLayoutCI.pushConstantRangeCount = 1;
  geoLayoutCI.pPushConstantRanges = &pcRange;
  if (vkCreatePipelineLayout(device, &geoLayoutCI, nullptr, &pipelineLayout) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create geometry pipeline layout");

  VkGraphicsPipelineCreateInfo geoPCI = {};
  geoPCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  geoPCI.stageCount = 2;
  geoPCI.pStages = geoStages;
  geoPCI.pVertexInputState = &vertexInput;
  geoPCI.pInputAssemblyState = &inputAssembly;
  geoPCI.pViewportState = &vpState;
  geoPCI.pDynamicState = &geoDynState;
  geoPCI.pRasterizationState = &rasterizer;
  geoPCI.pMultisampleState = &msaa;
  geoPCI.pDepthStencilState = &depthStencil;
  geoPCI.pColorBlendState = &gbBlendState;
  geoPCI.layout = pipelineLayout;
  geoPCI.renderPass = gBufferPass;
  geoPCI.subpass = 0;
  if (vkCreateGraphicsPipelines(device, pipelineCache, 1, &geoPCI, nullptr,
                                &graphicsPipeline) != VK_SUCCESS)
    throw std::runtime_error("Failed to create geometry/G-buffer pipeline");

  vkDestroyShaderModule(device, fragMod, nullptr);
  vkDestroyShaderModule(device, vertMod, nullptr);

  // 1b. INSTANCED G-BUFFER PIPELINE  (shader_instanced.vert + shader.frag)
  //     Model matrix comes from per-instance vertex buffer (binding 1),
  //     so no push constants needed.
  {
    auto iVertCode = readFile("../Shaders/shader_instanced.vert.spv");
    auto iFrag = readFile("../Shaders/shader.frag.spv");
    VkShaderModule iVertMod = createShaderModule(device, iVertCode);
    VkShaderModule iFragMod = createShaderModule(device, iFrag);

    VkPipelineShaderStageCreateInfo iStages[2] = {};
    iStages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  nullptr,
                  0,
                  VK_SHADER_STAGE_VERTEX_BIT,
                  iVertMod,
                  "main",
                  nullptr};
    iStages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  nullptr,
                  0,
                  VK_SHADER_STAGE_FRAGMENT_BIT,
                  iFragMod,
                  "main",
                  nullptr};

    // Per-vertex binding (same as before)
    VkVertexInputBindingDescription bindings[2] = {};
    bindings[0] = binding; // binding 0, per-vertex
    bindings[1].binding = 1;
    bindings[1].stride = sizeof(InstanceData);
    bindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    // Original 6 per-vertex attrs + 8 per-instance attrs (2 mat4s = 8 vec4s)
    std::array<VkVertexInputAttributeDescription, 14> iAttrs = {};
    for (int i = 0; i < 6; i++)
      iAttrs[i] = attrs[i]; // copy existing
    // instanceModel at locations 6-9
    for (int i = 0; i < 4; i++)
      iAttrs[6 + i] = {static_cast<uint32_t>(6 + i), 1,
                       VK_FORMAT_R32G32B32A32_SFLOAT,
                       static_cast<uint32_t>(i * 16)};
    // instanceNormal at locations 10-13
    for (int i = 0; i < 4; i++)
      iAttrs[10 + i] = {static_cast<uint32_t>(10 + i), 1,
                        VK_FORMAT_R32G32B32A32_SFLOAT,
                        static_cast<uint32_t>(64 + i * 16)};

    VkPipelineVertexInputStateCreateInfo iVI = {};
    iVI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    iVI.vertexBindingDescriptionCount = 2;
    iVI.pVertexBindingDescriptions = bindings;
    iVI.vertexAttributeDescriptionCount = static_cast<uint32_t>(iAttrs.size());
    iVI.pVertexAttributeDescriptions = iAttrs.data();

    // Layout: same descriptor sets + push constants as geometry pipeline.
    // Phase 7.2: shader.frag now uses push constants for bindless indices,
    // so the instanced pipeline must declare the same range.
    VkPipelineLayoutCreateInfo iLayoutCI = {};
    iLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    iLayoutCI.setLayoutCount = 2;
    iLayoutCI.pSetLayouts = geoLayouts;
    iLayoutCI.pushConstantRangeCount = 1;
    iLayoutCI.pPushConstantRanges = &pcRange;
    if (vkCreatePipelineLayout(device, &iLayoutCI, nullptr,
                               &instancedPipelineLayout) != VK_SUCCESS)
      throw std::runtime_error("Failed to create instanced pipeline layout");

    VkGraphicsPipelineCreateInfo iPCI = geoPCI;
    iPCI.pStages = iStages;
    iPCI.pVertexInputState = &iVI;
    iPCI.layout = instancedPipelineLayout;
    if (vkCreateGraphicsPipelines(device, pipelineCache, 1, &iPCI, nullptr,
                                  &instancedPipeline) != VK_SUCCESS)
      throw std::runtime_error("Failed to create instanced G-buffer pipeline");

    vkDestroyShaderModule(device, iFragMod, nullptr);
    vkDestroyShaderModule(device, iVertMod, nullptr);
  }

  // 2. SHADOW PIPELINE  (shadow.vert + shadow.frag, depth-only with alpha
  // test). The fragment stage discards sub-threshold alpha so Sponza foliage
  // casts leaf-shaped shadows instead of solid quad blocks. UV is read from
  // the vertex stream; the material's albedo sampler set (set=1) is bound
  // before each draw, reusing the existing material descriptor.
  auto shadowVertCode = readFile("../Shaders/shadow.vert.spv");
  auto shadowFragCode = readFile("../Shaders/shadow.frag.spv");
  VkShaderModule shadowVertMod = createShaderModule(device, shadowVertCode);
  VkShaderModule shadowFragMod = createShaderModule(device, shadowFragCode);

  VkPipelineShaderStageCreateInfo shadowStages[2] = {};
  shadowStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shadowStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  shadowStages[0].module = shadowVertMod;
  shadowStages[0].pName = "main";
  shadowStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shadowStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  shadowStages[1].module = shadowFragMod;
  shadowStages[1].pName = "main";

  VkVertexInputAttributeDescription shadowAttrs[2] = {
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)},
      {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, tex)}};
  VkPipelineVertexInputStateCreateInfo shadowVI = {};
  shadowVI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  shadowVI.vertexBindingDescriptionCount = 1;
  shadowVI.pVertexBindingDescriptions = &binding;
  shadowVI.vertexAttributeDescriptionCount = 2;
  shadowVI.pVertexAttributeDescriptions = shadowAttrs;

  VkPushConstantRange shadowPC = {
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
      sizeof(ShadowPushConstants)};
  // Phase 7.2: shadow pipeline uses the bindless layout. The fragment shader
  // indexes into it with the albedoIdx from push constants for alpha test.
  VkDescriptorSetLayout shadowSetLayouts[] = {descriptors.getBindlessLayout()};
  VkPipelineLayoutCreateInfo shadowLayoutCI = {};
  shadowLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  shadowLayoutCI.setLayoutCount = 1;
  shadowLayoutCI.pSetLayouts = shadowSetLayouts;
  shadowLayoutCI.pushConstantRangeCount = 1;
  shadowLayoutCI.pPushConstantRanges = &shadowPC;
  if (vkCreatePipelineLayout(device, &shadowLayoutCI, nullptr,
                             &shadowPipelineLayout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create shadow pipeline layout");

  VkPipelineRasterizationStateCreateInfo shadowRast = rasterizer;
  shadowRast.depthBiasEnable = VK_TRUE;
  shadowRast.depthBiasConstantFactor = 1.25f;
  shadowRast.depthBiasSlopeFactor = 1.75f;
  // cullMode will be overridden each draw via vkCmdSetCullMode.

  VkPipelineDepthStencilStateCreateInfo shadowDS = depthStencil;
  shadowDS.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

  VkPipelineColorBlendStateCreateInfo shadowBlend = {};
  shadowBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  shadowBlend.attachmentCount = 0;

  // Shadow pipeline allows runtime cull-mode flips so the user can A/B
  // back-face vs front-face culling without rebuilding.
  std::array<VkDynamicState, 3> shadowDynStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                                   VK_DYNAMIC_STATE_SCISSOR,
                                                   VK_DYNAMIC_STATE_CULL_MODE};
  VkPipelineDynamicStateCreateInfo shadowDynState = {};
  shadowDynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  shadowDynState.dynamicStateCount =
      static_cast<uint32_t>(shadowDynStates.size());
  shadowDynState.pDynamicStates = shadowDynStates.data();

  VkGraphicsPipelineCreateInfo shadowPCI = geoPCI;
  shadowPCI.stageCount = 2;
  shadowPCI.pStages = shadowStages;
  shadowPCI.pVertexInputState = &shadowVI;
  shadowPCI.pRasterizationState = &shadowRast;
  shadowPCI.pDepthStencilState = &shadowDS;
  shadowPCI.pColorBlendState = &shadowBlend;
  shadowPCI.pDynamicState = &shadowDynState;
  shadowPCI.layout = shadowPipelineLayout;
  shadowPCI.renderPass = shadowPassRP;
  shadowPCI.subpass = 0;
  if (vkCreateGraphicsPipelines(device, pipelineCache, 1, &shadowPCI, nullptr,
                                &shadowPipeline) != VK_SUCCESS)
    throw std::runtime_error("Failed to create shadow pipeline");

  vkDestroyShaderModule(device, shadowVertMod, nullptr);
  vkDestroyShaderModule(device, shadowFragMod, nullptr);

  // 3a. LIT PBR PIPELINE  (second.vert + lit.frag → litBuffer, lit pass)
  // Same shape as the SSR composite pipeline below; differs only in shader
  // and render pass.
  auto fullscreenVertCode = readFile("../Shaders/second.vert.spv");
  auto litFragCode        = readFile("../Shaders/lit.frag.spv");
  VkShaderModule fullscreenVertMod = createShaderModule(device, fullscreenVertCode);
  VkShaderModule litFragMod        = createShaderModule(device, litFragCode);

  VkPipelineShaderStageCreateInfo litStages[2] = {};
  litStages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                  VK_SHADER_STAGE_VERTEX_BIT, fullscreenVertMod, "main", nullptr};
  litStages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                  VK_SHADER_STAGE_FRAGMENT_BIT, litFragMod, "main", nullptr};

  // No vertex buffer for fullscreen triangle
  VkPipelineVertexInputStateCreateInfo emptyVI = {};
  emptyVI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkDescriptorSetLayout litLayoutsArr[] = {descriptors.getVPLayout(),
                                           descriptors.getGBufferLayout()};
  VkPipelineLayoutCreateInfo litLayoutCI = {};
  litLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  litLayoutCI.setLayoutCount = 2;
  litLayoutCI.pSetLayouts = litLayoutsArr;
  if (vkCreatePipelineLayout(device, &litLayoutCI, nullptr,
                             &litPipelineLayout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create lit pipeline layout");

  VkPipelineDepthStencilStateCreateInfo noDepth = {};
  noDepth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

  VkPipelineColorBlendStateCreateInfo singleColorBlend = {};
  singleColorBlend.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  singleColorBlend.attachmentCount = 1;
  singleColorBlend.pAttachments = &blendOff;

  VkGraphicsPipelineCreateInfo litPCI = geoPCI;
  litPCI.stageCount = 2;
  litPCI.pStages = litStages;
  litPCI.pVertexInputState = &emptyVI;
  litPCI.pDepthStencilState = &noDepth;
  litPCI.pColorBlendState = &singleColorBlend;
  litPCI.pDynamicState = &dynState; // no cull-mode dynamic state on fullscreen
  litPCI.layout = litPipelineLayout;
  litPCI.renderPass = litPass;
  litPCI.subpass = 0;
  if (vkCreateGraphicsPipelines(device, pipelineCache, 1, &litPCI, nullptr,
                                &litPipeline) != VK_SUCCESS)
    throw std::runtime_error("Failed to create lit pipeline");

  vkDestroyShaderModule(device, litFragMod, nullptr);

  // 3a.1. TRANSPARENT FORWARD PIPELINES
  // Draw alphaMode=BLEND meshes into the HDR lit buffer after deferred
  // lighting. Subpass 1 loads the G-buffer depth as read-only; these pipelines
  // depth-test against opaque geometry but do not write depth.
  {
    auto tVertCode = readFile("../Shaders/shader.vert.spv");
    auto tFragCode = readFile("../Shaders/transparent.frag.spv");
    VkShaderModule tVertMod = createShaderModule(device, tVertCode);
    VkShaderModule tFragMod = createShaderModule(device, tFragCode);

    VkPipelineShaderStageCreateInfo tStages[2] = {};
    tStages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr,
                  0, VK_SHADER_STAGE_VERTEX_BIT, tVertMod, "main", nullptr};
    tStages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr,
                  0, VK_SHADER_STAGE_FRAGMENT_BIT, tFragMod, "main", nullptr};

    VkDescriptorSetLayout transparentLayouts[] = {
        descriptors.getVPLayout(), descriptors.getBindlessLayout()};
    VkPipelineLayoutCreateInfo transparentLayoutCI = {};
    transparentLayoutCI.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    transparentLayoutCI.setLayoutCount = 2;
    transparentLayoutCI.pSetLayouts = transparentLayouts;
    transparentLayoutCI.pushConstantRangeCount = 1;
    transparentLayoutCI.pPushConstantRanges = &pcRange;
    if (vkCreatePipelineLayout(device, &transparentLayoutCI, nullptr,
                               &transparentPipelineLayout) != VK_SUCCESS)
      throw std::runtime_error("Failed to create transparent pipeline layout");

    VkPipelineDepthStencilStateCreateInfo transparentDepth = depthStencil;
    transparentDepth.depthWriteEnable = VK_FALSE;
    transparentDepth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState alphaBlend = blendOff;
    alphaBlend.blendEnable = VK_TRUE;
    alphaBlend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    alphaBlend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    alphaBlend.colorBlendOp = VK_BLEND_OP_ADD;
    alphaBlend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    alphaBlend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    alphaBlend.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo transparentBlend = {};
    transparentBlend.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    transparentBlend.attachmentCount = 1;
    transparentBlend.pAttachments = &alphaBlend;

    VkGraphicsPipelineCreateInfo transparentPCI = geoPCI;
    transparentPCI.pStages = tStages;
    transparentPCI.pDepthStencilState = &transparentDepth;
    transparentPCI.pColorBlendState = &transparentBlend;
    transparentPCI.layout = transparentPipelineLayout;
    transparentPCI.renderPass = litPass;
    transparentPCI.subpass = 1;
    if (vkCreateGraphicsPipelines(device, pipelineCache, 1, &transparentPCI,
                                  nullptr, &transparentPipeline) != VK_SUCCESS)
      throw std::runtime_error("Failed to create transparent pipeline");

    vkDestroyShaderModule(device, tVertMod, nullptr);

    auto tiVertCode = readFile("../Shaders/shader_instanced.vert.spv");
    VkShaderModule tiVertMod = createShaderModule(device, tiVertCode);
    VkPipelineShaderStageCreateInfo tiStages[2] = {};
    tiStages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr,
                   0, VK_SHADER_STAGE_VERTEX_BIT, tiVertMod, "main", nullptr};
    tiStages[1] = tStages[1];

    VkVertexInputBindingDescription tiBindings[2] = {};
    tiBindings[0] = binding;
    tiBindings[1].binding = 1;
    tiBindings[1].stride = sizeof(InstanceData);
    tiBindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    std::array<VkVertexInputAttributeDescription, 14> tiAttrs = {};
    for (int i = 0; i < 6; ++i)
      tiAttrs[i] = attrs[i];
    for (int i = 0; i < 4; ++i)
      tiAttrs[6 + i] = {static_cast<uint32_t>(6 + i), 1,
                        VK_FORMAT_R32G32B32A32_SFLOAT,
                        static_cast<uint32_t>(i * 16)};
    for (int i = 0; i < 4; ++i)
      tiAttrs[10 + i] = {static_cast<uint32_t>(10 + i), 1,
                         VK_FORMAT_R32G32B32A32_SFLOAT,
                         static_cast<uint32_t>(64 + i * 16)};

    VkPipelineVertexInputStateCreateInfo tiVI = {};
    tiVI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    tiVI.vertexBindingDescriptionCount = 2;
    tiVI.pVertexBindingDescriptions = tiBindings;
    tiVI.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(tiAttrs.size());
    tiVI.pVertexAttributeDescriptions = tiAttrs.data();

    if (vkCreatePipelineLayout(device, &transparentLayoutCI, nullptr,
                               &transparentInstancedPipelineLayout) !=
        VK_SUCCESS)
      throw std::runtime_error(
          "Failed to create transparent instanced pipeline layout");

    VkGraphicsPipelineCreateInfo tiPCI = transparentPCI;
    tiPCI.pStages = tiStages;
    tiPCI.pVertexInputState = &tiVI;
    tiPCI.layout = transparentInstancedPipelineLayout;
    if (vkCreateGraphicsPipelines(device, pipelineCache, 1, &tiPCI, nullptr,
                                  &transparentInstancedPipeline) != VK_SUCCESS)
      throw std::runtime_error(
          "Failed to create transparent instanced pipeline");

    vkDestroyShaderModule(device, tiVertMod, nullptr);
    vkDestroyShaderModule(device, tFragMod, nullptr);
  }

  // 3a'. SSGI PIPELINE  (second.vert + ssgi.frag → ssgiBuffer, ssgi pass).
  // Same fullscreen-quad shape as lit; sees the same descriptor sets
  // (VP + G-buffer) so it can sample G-buffer + scene UBO directly.
  // Output is a single R16 HDR color attachment; lit.frag samples it
  // (via gbuffer set binding 5) and applies the cross-bilateral filter.
  auto ssgiFragCode = readFile("../Shaders/ssgi.frag.spv");
  VkShaderModule ssgiFragMod = createShaderModule(device, ssgiFragCode);

  VkPipelineShaderStageCreateInfo ssgiStages[2] = {};
  ssgiStages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                   VK_SHADER_STAGE_VERTEX_BIT, fullscreenVertMod, "main", nullptr};
  ssgiStages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                   VK_SHADER_STAGE_FRAGMENT_BIT, ssgiFragMod, "main", nullptr};

  VkDescriptorSetLayout ssgiLayoutsArr[] = {descriptors.getVPLayout(),
                                            descriptors.getGBufferLayout(),
                                            descriptors.getSsgiPrevLayout()};
  VkPipelineLayoutCreateInfo ssgiLayoutCI = {};
  ssgiLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  ssgiLayoutCI.setLayoutCount = 3;
  ssgiLayoutCI.pSetLayouts = ssgiLayoutsArr;
  if (vkCreatePipelineLayout(device, &ssgiLayoutCI, nullptr,
                             &ssgiPipelineLayout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create SSGI pipeline layout");

  VkGraphicsPipelineCreateInfo ssgiPCI = geoPCI;
  ssgiPCI.stageCount = 2;
  ssgiPCI.pStages = ssgiStages;
  ssgiPCI.pVertexInputState = &emptyVI;
  ssgiPCI.pDepthStencilState = &noDepth;
  ssgiPCI.pColorBlendState = &singleColorBlend;
  ssgiPCI.pDynamicState = &dynState;
  ssgiPCI.layout = ssgiPipelineLayout;
  ssgiPCI.renderPass = ssgiPass;
  ssgiPCI.subpass = 0;
  if (vkCreateGraphicsPipelines(device, pipelineCache, 1, &ssgiPCI, nullptr,
                                &ssgiPipeline) != VK_SUCCESS)
    throw std::runtime_error("Failed to create SSGI pipeline");

  vkDestroyShaderModule(device, ssgiFragMod, nullptr);

  // 3b. SSR COMPOSITE PIPELINE  (second.vert + ssr_composite.frag → colorBuffer,
  // composition subpass 0). Shares fullscreen vert shader and most state
  // with the lit pipeline.
  auto compositeFragCode = readFile("../Shaders/ssr_composite.frag.spv");
  VkShaderModule compositeFragMod = createShaderModule(device, compositeFragCode);

  VkPipelineShaderStageCreateInfo deferredStages[2] = {};
  deferredStages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                       nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,
                       fullscreenVertMod, "main", nullptr};
  deferredStages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                       nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT,
                       compositeFragMod, "main", nullptr};

  VkDescriptorSetLayout deferredLayouts[] = {descriptors.getVPLayout(),
                                             descriptors.getGBufferLayout()};
  VkPipelineLayoutCreateInfo deferredLayoutCI = {};
  deferredLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  deferredLayoutCI.setLayoutCount = 2;
  deferredLayoutCI.pSetLayouts = deferredLayouts;
  if (vkCreatePipelineLayout(device, &deferredLayoutCI, nullptr,
                             &deferredPipelineLayout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create SSR composite pipeline layout");

  VkGraphicsPipelineCreateInfo deferredPCI = geoPCI;
  deferredPCI.stageCount = 2;
  deferredPCI.pStages = deferredStages;
  deferredPCI.pVertexInputState = &emptyVI;
  deferredPCI.pDepthStencilState = &noDepth;
  deferredPCI.pColorBlendState = &singleColorBlend;
  deferredPCI.pDynamicState = &dynState; // fullscreen — no dynamic cull mode
  deferredPCI.layout = deferredPipelineLayout;
  deferredPCI.renderPass = compositionPass;
  deferredPCI.subpass = 0;
  if (vkCreateGraphicsPipelines(device, pipelineCache, 1, &deferredPCI, nullptr,
                                &deferredPipeline) != VK_SUCCESS)
    throw std::runtime_error("Failed to create SSR composite pipeline");

  vkDestroyShaderModule(device, compositeFragMod, nullptr);
  vkDestroyShaderModule(device, fullscreenVertMod, nullptr);

  // 4. TONE-MAPPING / POST PIPELINE  (second.vert + second.frag → swapchain,
  // subpass 1)
  auto secondVertCode = readFile("../Shaders/second.vert.spv");
  auto secondFragCode = readFile("../Shaders/second.frag.spv");
  VkShaderModule secondVertMod = createShaderModule(device, secondVertCode);
  VkShaderModule secondFragMod = createShaderModule(device, secondFragCode);

  VkPipelineShaderStageCreateInfo secondStages[2] = {};
  secondStages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                     nullptr,
                     0,
                     VK_SHADER_STAGE_VERTEX_BIT,
                     secondVertMod,
                     "main",
                     nullptr};
  secondStages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                     nullptr,
                     0,
                     VK_SHADER_STAGE_FRAGMENT_BIT,
                     secondFragMod,
                     "main",
                     nullptr};

  // Tonemap subpass now has 3 descriptor sets:
  //   set 0 = VP/scene UBO (for prevViewProj, taaParams, viewportSize, invProj,
  //           invView).
  //   set 1 = input attachment (colorBuffer from subpass 0).
  //   set 2 = TAA samplers (history-prev + depth).
  std::array<VkDescriptorSetLayout, 3> secondSetLayouts = {
      descriptors.getVPLayout(), descriptors.getInputLayout(),
      descriptors.getTaaLayout()};
  VkPipelineLayoutCreateInfo secondLayoutCI = {};
  secondLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  secondLayoutCI.setLayoutCount = static_cast<uint32_t>(secondSetLayouts.size());
  secondLayoutCI.pSetLayouts = secondSetLayouts.data();
  if (vkCreatePipelineLayout(device, &secondLayoutCI, nullptr,
                             &secondPipelineLayout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create tone-mapping pipeline layout");

  // Tonemap subpass writes 2 color attachments (swap LDR + history HDR).
  std::array<VkPipelineColorBlendAttachmentState, 2> secondBlends = {blendOff,
                                                                     blendOff};
  VkPipelineColorBlendStateCreateInfo secondBlendState = {};
  secondBlendState.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  secondBlendState.attachmentCount =
      static_cast<uint32_t>(secondBlends.size());
  secondBlendState.pAttachments = secondBlends.data();

  VkGraphicsPipelineCreateInfo secondPCI = deferredPCI;
  secondPCI.pStages = secondStages;
  secondPCI.pColorBlendState = &secondBlendState;
  secondPCI.layout = secondPipelineLayout;
  secondPCI.subpass = 1;
  if (vkCreateGraphicsPipelines(device, pipelineCache, 1, &secondPCI, nullptr,
                                &secondPipeline) != VK_SUCCESS)
    throw std::runtime_error("Failed to create tone-mapping pipeline");

  vkDestroyShaderModule(device, secondFragMod, nullptr);
  vkDestroyShaderModule(device, secondVertMod, nullptr);
}

// ---------------------------------------------------------------------------
void VulkanPipeline::cleanup(VkDevice device) {
  vkDestroyPipeline(device, secondPipeline, nullptr);
  vkDestroyPipelineLayout(device, secondPipelineLayout, nullptr);
  vkDestroyPipeline(device, deferredPipeline, nullptr);
  vkDestroyPipelineLayout(device, deferredPipelineLayout, nullptr);
  vkDestroyPipeline(device, ssgiPipeline, nullptr);
  vkDestroyPipelineLayout(device, ssgiPipelineLayout, nullptr);
  vkDestroyPipeline(device, transparentInstancedPipeline, nullptr);
  vkDestroyPipelineLayout(device, transparentInstancedPipelineLayout, nullptr);
  vkDestroyPipeline(device, transparentPipeline, nullptr);
  vkDestroyPipelineLayout(device, transparentPipelineLayout, nullptr);
  vkDestroyPipeline(device, litPipeline, nullptr);
  vkDestroyPipelineLayout(device, litPipelineLayout, nullptr);
  vkDestroyPipeline(device, shadowPipeline, nullptr);
  vkDestroyPipelineLayout(device, shadowPipelineLayout, nullptr);
  vkDestroyPipeline(device, instancedPipeline, nullptr);
  vkDestroyPipelineLayout(device, instancedPipelineLayout, nullptr);
  vkDestroyPipeline(device, graphicsPipeline, nullptr);
  vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

// ---------------------------------------------------------------------------
VkShaderModule
VulkanPipeline::createShaderModule(VkDevice device,
                                   const std::vector<char> &code) {
  VkShaderModuleCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  ci.codeSize = code.size();
  ci.pCode = reinterpret_cast<const uint32_t *>(code.data());
  VkShaderModule mod;
  if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS)
    throw std::runtime_error("Failed to create shader module");
  return mod;
}
