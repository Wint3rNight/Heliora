#include "RenderPassManager.h"

#include <array>
#include <stdexcept>

// G-buffer render pass
// 4 attachments: GB0 (color+metallic), GB1 (normal+roughness), GB2 (AO), depth
// Single subpass; all color outputs transition to SHADER_READ_ONLY_OPTIMAL.
void RenderPassManager::createGBufferRenderPass(VkDevice device,
                                                VkFormat gb0Format,
                                                VkFormat gb1Format,
                                                VkFormat gb2Format,
                                                VkFormat depthFormat) {
  std::array<VkAttachmentDescription, 4> attachments = {};

  // GB0
  attachments[0].format = gb0Format;
  attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
  attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  // GB1
  attachments[1] = attachments[0];
  attachments[1].format = gb1Format;

  // GB2
  attachments[2] = attachments[0];
  attachments[2].format = gb2Format;

  // Depth
  attachments[3].format = depthFormat;
  attachments[3].samples = VK_SAMPLE_COUNT_1_BIT;
  attachments[3].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachments[3].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[3].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachments[3].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[3].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  attachments[3].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

  std::array<VkAttachmentReference, 3> colorRefs = {};
  for (uint32_t i = 0; i < 3; ++i) {
    colorRefs[i].attachment = i;
    colorRefs[i].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  }
  VkAttachmentReference depthRef = {};
  depthRef.attachment = 3;
  depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 3;
  subpass.pColorAttachments = colorRefs.data();
  subpass.pDepthStencilAttachment = &depthRef;

  std::array<VkSubpassDependency, 2> deps = {};
  deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
  deps[0].dstSubpass = 0;
  deps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

  deps[1].srcSubpass = 0;
  deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
  deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                         VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  VkRenderPassCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  ci.attachmentCount = static_cast<uint32_t>(attachments.size());
  ci.pAttachments = attachments.data();
  ci.subpassCount = 1;
  ci.pSubpasses = &subpass;
  ci.dependencyCount = static_cast<uint32_t>(deps.size());
  ci.pDependencies = deps.data();
  if (vkCreateRenderPass(device, &ci, nullptr, &gBufferRenderPass) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create G-buffer render pass");
}

// Lit render pass (1 attachment: litBuffer)
// Single subpass: PBR + IBL + SSAO + bloom + FXAA + fog → litBuffer.
// finalLayout = SHADER_READ_ONLY_OPTIMAL so the composition pass can sample.

void RenderPassManager::createLitRenderPass(VkDevice device,
                                            VkFormat litFormat) {
  VkAttachmentDescription att = {};
  att.format         = litFormat;
  att.samples        = VK_SAMPLE_COUNT_1_BIT;
  att.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
  att.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
  att.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  att.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
  att.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkAttachmentReference colorRef = {};
  colorRef.attachment = 0;
  colorRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments    = &colorRef;

  std::array<VkSubpassDependency, 2> deps = {};
  deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
  deps[0].dstSubpass    = 0;
  deps[0].srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  deps[1].srcSubpass    = 0;
  deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
  deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  VkRenderPassCreateInfo ci = {};
  ci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  ci.attachmentCount = 1;
  ci.pAttachments    = &att;
  ci.subpassCount    = 1;
  ci.pSubpasses      = &subpass;
  ci.dependencyCount = static_cast<uint32_t>(deps.size());
  ci.pDependencies   = deps.data();
  if (vkCreateRenderPass(device, &ci, nullptr, &litRenderPass) != VK_SUCCESS)
    throw std::runtime_error("Failed to create lit render pass");
}

// ---------------------------------------------------------------------------
// Composition render pass (3 attachments: swapchain + colorBuffer + history)
// Subpass 0: SSR composite (samples litBuffer + G-buffer) → colorBuffer
// Subpass 1: TAA + ACES+gamma (colorBuffer as input attachment, prev-history
//            sampled, depth sampled) → swapchain (LDR) + history (HDR)
// ---------------------------------------------------------------------------
void RenderPassManager::createRenderPass(VkDevice device,
                                         VkFormat swapchainFormat,
                                         VkFormat colorFormat,
                                         VkFormat historyFormat) {
  // Attachment 0: swapchain output (LDR, post-tonemap)
  VkAttachmentDescription swapAtt = {};
  swapAtt.format = swapchainFormat;
  swapAtt.samples = VK_SAMPLE_COUNT_1_BIT;
  swapAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  swapAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  swapAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  swapAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  swapAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  swapAtt.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // ImGui renders on top before present

  // Attachment 1: HDR colorBuffer (written by subpass 0, read by subpass 1)
  VkAttachmentDescription colorAtt = {};
  colorAtt.format = colorFormat;
  colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
  colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAtt.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  // Attachment 2: HDR history (write-target this frame; sampled next frame).
  // initialLayout=UNDEFINED — the prev frame finished with SHADER_READ_ONLY,
  // but we don't need the prev content (we sample the OTHER ping-pong image
  // via descriptor). finalLayout=SHADER_READ_ONLY so next frame's sampler
  // bind is well-defined.
  VkAttachmentDescription histAtt = {};
  histAtt.format = historyFormat;
  histAtt.samples = VK_SAMPLE_COUNT_1_BIT;
  histAtt.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  histAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  histAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  histAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  histAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  histAtt.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  std::array<VkAttachmentDescription, 3> attachments = {swapAtt, colorAtt,
                                                        histAtt};

  // Subpass 0 writes to colorBuffer (attachment 1)
  VkAttachmentReference colorRef = {};
  colorRef.attachment = 1;
  colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass0 = {};
  subpass0.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass0.colorAttachmentCount = 1;
  subpass0.pColorAttachments = &colorRef;

  // Subpass 1 reads colorBuffer as input attachment, writes to swapchain
  // (attachment 0) AND history (attachment 2)
  std::array<VkAttachmentReference, 2> subpass1ColorRefs = {};
  subpass1ColorRefs[0].attachment = 0;
  subpass1ColorRefs[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  subpass1ColorRefs[1].attachment = 2;
  subpass1ColorRefs[1].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference inputRef = {};
  inputRef.attachment = 1;
  inputRef.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkSubpassDescription subpass1 = {};
  subpass1.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass1.colorAttachmentCount =
      static_cast<uint32_t>(subpass1ColorRefs.size());
  subpass1.pColorAttachments = subpass1ColorRefs.data();
  subpass1.inputAttachmentCount = 1;
  subpass1.pInputAttachments = &inputRef;

  std::array<VkSubpassDescription, 2> subpasses = {subpass0, subpass1};

  std::array<VkSubpassDependency, 4> deps = {};
  // External → subpass 0 (covers swapchain + colorBuffer transitions in)
  deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
  deps[0].dstSubpass = 0;
  deps[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  deps[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
  deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                          VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  // Subpass 0 → subpass 1
  deps[1].srcSubpass = 0;
  deps[1].dstSubpass = 1;
  deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  deps[1].dstAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;

  // Subpass 1 → external (covers history → SHADER_READ_ONLY for next frame
  // and the swapchain → ImGui pass).
  deps[2].srcSubpass = 1;
  deps[2].dstSubpass = VK_SUBPASS_EXTERNAL;
  deps[2].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  deps[2].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  deps[2].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                          VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  deps[2].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;

  // External → subpass 1: ensure the PREVIOUS frame's TAA fragment-shader
  // read of the OTHER history image (now being written this frame as the
  // ping-pong target) completes before we start writing the same image.
  // Without this dep the layout transition + write could race with the
  // prev frame's sampler read.
  deps[3].srcSubpass = VK_SUBPASS_EXTERNAL;
  deps[3].dstSubpass = 1;
  deps[3].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  deps[3].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  deps[3].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  deps[3].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkRenderPassCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  ci.attachmentCount = static_cast<uint32_t>(attachments.size());
  ci.pAttachments = attachments.data();
  ci.subpassCount = static_cast<uint32_t>(subpasses.size());
  ci.pSubpasses = subpasses.data();
  ci.dependencyCount = static_cast<uint32_t>(deps.size());
  ci.pDependencies = deps.data();
  if (vkCreateRenderPass(device, &ci, nullptr, &renderPass) != VK_SUCCESS)
    throw std::runtime_error("Failed to create composition render pass");
}

// Shadow render pass (depth-only, unchanged)
void RenderPassManager::createShadowRenderPass(VkDevice device,
                                               VkFormat depthFormat) {
  VkAttachmentDescription depthAtt = {};
  depthAtt.format = depthFormat;
  depthAtt.samples = VK_SAMPLE_COUNT_1_BIT;
  depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  depthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depthAtt.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

  VkAttachmentReference depthRef = {};
  depthRef.attachment = 0;
  depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.pDepthStencilAttachment = &depthRef;

  std::array<VkSubpassDependency, 2> deps = {};
  deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
  deps[0].dstSubpass = 0;
  deps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  deps[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  deps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

  deps[1].srcSubpass = 0;
  deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
  deps[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  VkRenderPassCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  ci.attachmentCount = 1;
  ci.pAttachments = &depthAtt;
  ci.subpassCount = 1;
  ci.pSubpasses = &subpass;
  ci.dependencyCount = static_cast<uint32_t>(deps.size());
  ci.pDependencies = deps.data();
  if (vkCreateRenderPass(device, &ci, nullptr, &shadowRenderPass) != VK_SUCCESS)
    throw std::runtime_error("Failed to create shadow render pass");
}

void RenderPassManager::createImGuiRenderPass(VkDevice device,
                                              VkFormat swapchainFormat) {
  // Single swapchain attachment. LOAD to preserve the ACES output underneath.
  VkAttachmentDescription att = {};
  att.format         = swapchainFormat;
  att.samples        = VK_SAMPLE_COUNT_1_BIT;
  att.loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
  att.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
  att.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  att.initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  att.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference colorRef = {};
  colorRef.attachment = 0;
  colorRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments    = &colorRef;

  VkSubpassDependency dep = {};
  dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
  dep.dstSubpass    = 0;
  dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkRenderPassCreateInfo ci = {};
  ci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  ci.attachmentCount = 1;
  ci.pAttachments    = &att;
  ci.subpassCount    = 1;
  ci.pSubpasses      = &subpass;
  ci.dependencyCount = 1;
  ci.pDependencies   = &dep;
  if (vkCreateRenderPass(device, &ci, nullptr, &imguiRenderPass) != VK_SUCCESS)
    throw std::runtime_error("Failed to create ImGui render pass");
}

void RenderPassManager::cleanup(VkDevice device) {
  if (imguiRenderPass != VK_NULL_HANDLE)
    vkDestroyRenderPass(device, imguiRenderPass, nullptr);
  if (litRenderPass != VK_NULL_HANDLE)
    vkDestroyRenderPass(device, litRenderPass, nullptr);
  vkDestroyRenderPass(device, gBufferRenderPass, nullptr);
  vkDestroyRenderPass(device, renderPass, nullptr);
  vkDestroyRenderPass(device, shadowRenderPass, nullptr);
}
