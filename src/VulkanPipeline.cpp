#include "VulkanPipeline.h"
#include "Utilities.h"
#include <cstddef>

#include <array>
#include <fstream>
#include <spdlog/spdlog.h>
#include <stdexcept>

// Helper: load pipeline cache from disk
static VkPipelineCache createPipelineCache(VkDevice device) {
  VkPipelineCacheCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;

  std::vector<char> data;
  std::ifstream f("pipeline_cache.bin", std::ios::ate | std::ios::binary);
  if (f.is_open()) {
    data.resize((size_t)f.tellg());
    f.seekg(0);
    f.read(data.data(), data.size());
    f.close();
    ci.initialDataSize = data.size();
    ci.pInitialData = data.data();
    spdlog::info("Loaded pipeline cache ({} bytes)", data.size());
  }

  VkPipelineCache cache = VK_NULL_HANDLE;
  vkCreatePipelineCache(device, &ci, nullptr, &cache);
  return cache;
}

// ---------------------------------------------------------------------------
void VulkanPipeline::createPipelines(VkDevice device, VkRenderPass gBufferPass,
                                     VkRenderPass litPass,
                                     VkRenderPass compositionPass,
                                     VkRenderPass shadowPassRP,
                                     VkExtent2D extent,
                                     const DescriptorManager &descriptors) {
  pipelineCache = createPipelineCache(device);

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

  VkDescriptorSetLayout geoLayouts[] = {descriptors.getVPLayout(),
                                        descriptors.getSamplerLayout()};
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
  geoPCI.pDynamicState = &dynState;
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

    // Layout: same descriptor sets as geometry pipeline, no push constants
    VkPipelineLayoutCreateInfo iLayoutCI = {};
    iLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    iLayoutCI.setLayoutCount = 2;
    iLayoutCI.pSetLayouts = geoLayouts;
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

  // 2. SHADOW PIPELINE  (shadow.vert only, depth-only pass)
  auto shadowVertCode = readFile("../Shaders/shadow.vert.spv");
  VkShaderModule shadowVertMod = createShaderModule(device, shadowVertCode);

  VkPipelineShaderStageCreateInfo shadowStage = {};
  shadowStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shadowStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
  shadowStage.module = shadowVertMod;
  shadowStage.pName = "main";

  VkVertexInputAttributeDescription posAttr = {0, 0, VK_FORMAT_R32G32B32_SFLOAT,
                                               offsetof(Vertex, pos)};
  VkPipelineVertexInputStateCreateInfo shadowVI = {};
  shadowVI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  shadowVI.vertexBindingDescriptionCount = 1;
  shadowVI.pVertexBindingDescriptions = &binding;
  shadowVI.vertexAttributeDescriptionCount = 1;
  shadowVI.pVertexAttributeDescriptions = &posAttr;

  VkPushConstantRange shadowPC = {VK_SHADER_STAGE_VERTEX_BIT, 0,
                                  sizeof(ShadowPushConstants)};
  VkPipelineLayoutCreateInfo shadowLayoutCI = {};
  shadowLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  shadowLayoutCI.pushConstantRangeCount = 1;
  shadowLayoutCI.pPushConstantRanges = &shadowPC;
  if (vkCreatePipelineLayout(device, &shadowLayoutCI, nullptr,
                             &shadowPipelineLayout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create shadow pipeline layout");

  VkPipelineRasterizationStateCreateInfo shadowRast = rasterizer;
  shadowRast.depthBiasEnable = VK_TRUE;
  shadowRast.depthBiasConstantFactor = 1.25f;
  shadowRast.depthBiasSlopeFactor = 1.75f;

  VkPipelineDepthStencilStateCreateInfo shadowDS = depthStencil;
  shadowDS.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

  VkPipelineColorBlendStateCreateInfo shadowBlend = {};
  shadowBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  shadowBlend.attachmentCount = 0;

  VkGraphicsPipelineCreateInfo shadowPCI = geoPCI;
  shadowPCI.stageCount = 1;
  shadowPCI.pStages = &shadowStage;
  shadowPCI.pVertexInputState = &shadowVI;
  shadowPCI.pRasterizationState = &shadowRast;
  shadowPCI.pDepthStencilState = &shadowDS;
  shadowPCI.pColorBlendState = &shadowBlend;
  shadowPCI.layout = shadowPipelineLayout;
  shadowPCI.renderPass = shadowPassRP;
  shadowPCI.subpass = 0;
  if (vkCreateGraphicsPipelines(device, pipelineCache, 1, &shadowPCI, nullptr,
                                &shadowPipeline) != VK_SUCCESS)
    throw std::runtime_error("Failed to create shadow pipeline");

  vkDestroyShaderModule(device, shadowVertMod, nullptr);

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
  litPCI.layout = litPipelineLayout;
  litPCI.renderPass = litPass;
  litPCI.subpass = 0;
  if (vkCreateGraphicsPipelines(device, pipelineCache, 1, &litPCI, nullptr,
                                &litPipeline) != VK_SUCCESS)
    throw std::runtime_error("Failed to create lit pipeline");

  vkDestroyShaderModule(device, litFragMod, nullptr);

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

  VkDescriptorSetLayout inputLayout = descriptors.getInputLayout();
  VkPipelineLayoutCreateInfo secondLayoutCI = {};
  secondLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  secondLayoutCI.setLayoutCount = 1;
  secondLayoutCI.pSetLayouts = &inputLayout;
  if (vkCreatePipelineLayout(device, &secondLayoutCI, nullptr,
                             &secondPipelineLayout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create tone-mapping pipeline layout");

  VkGraphicsPipelineCreateInfo secondPCI = deferredPCI;
  secondPCI.pStages = secondStages;
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
  if (pipelineCache != VK_NULL_HANDLE) {
    size_t sz = 0;
    if (vkGetPipelineCacheData(device, pipelineCache, &sz, nullptr) ==
            VK_SUCCESS &&
        sz > 0) {
      std::vector<char> data(sz);
      if (vkGetPipelineCacheData(device, pipelineCache, &sz, data.data()) ==
          VK_SUCCESS) {
        std::ofstream f("pipeline_cache.bin", std::ios::binary);
        if (f.is_open()) {
          f.write(data.data(), sz);
          spdlog::info("Saved pipeline cache ({} bytes)", sz);
        }
      }
    }
    vkDestroyPipelineCache(device, pipelineCache, nullptr);
  }

  vkDestroyPipeline(device, secondPipeline, nullptr);
  vkDestroyPipelineLayout(device, secondPipelineLayout, nullptr);
  vkDestroyPipeline(device, deferredPipeline, nullptr);
  vkDestroyPipelineLayout(device, deferredPipelineLayout, nullptr);
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
