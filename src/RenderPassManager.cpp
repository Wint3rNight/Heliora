#include "RenderPassManager.h"

#include <array>
#include <stdexcept>

void RenderPassManager::createRenderPass(VkDevice device,
                                         VkFormat swapchainFormat,
                                         VkFormat colorFormat,
                                         VkFormat depthFormat) {
  // array of subpasses
  std::array<VkSubpassDescription, 2> subpasses{};

  // ATTACHMENTS
  // subpass 1 attachments + reference (input)

  // color attachment(input)
  VkAttachmentDescription colorAttachment = {};
  colorAttachment.format = colorFormat;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  colorAttachment.loadOp =
      VK_ATTACHMENT_LOAD_OP_CLEAR; // clear the color attachment at the start of
                                   // the render pass
  colorAttachment.storeOp =
      VK_ATTACHMENT_STORE_OP_STORE; // store the color attachment's contents to
                                    // memory at the end of the render pass
  colorAttachment.stencilLoadOp =
      VK_ATTACHMENT_LOAD_OP_DONT_CARE; // we don't use stencil, so don't care
  colorAttachment.stencilStoreOp =
      VK_ATTACHMENT_STORE_OP_DONT_CARE; // we don't use stencil, so don't care
  colorAttachment.initialLayout =
      VK_IMAGE_LAYOUT_UNDEFINED; // don't care about initial layout of color
  colorAttachment.finalLayout =
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // optimal layout for color
                                                // attachment

  // depth attachment(input)
  VkAttachmentDescription depthAttachment = {};
  depthAttachment.format = depthFormat;
  depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  depthAttachment.loadOp =
      VK_ATTACHMENT_LOAD_OP_CLEAR; // clear depth at the start of the render
                                   // pass
  depthAttachment.storeOp =
      VK_ATTACHMENT_STORE_OP_DONT_CARE; // we don't need depth after render pass
  depthAttachment.stencilLoadOp =
      VK_ATTACHMENT_LOAD_OP_DONT_CARE; // we don't use stencil, so don't care
  depthAttachment.stencilStoreOp =
      VK_ATTACHMENT_STORE_OP_DONT_CARE; // we don't use stencil, so don't care
  depthAttachment.initialLayout =
      VK_IMAGE_LAYOUT_UNDEFINED; // don't care about initial layout of depth
  depthAttachment.finalLayout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; // optimal layout for
                                                        // depth attachment

  // color attachment ( input )reference
  VkAttachmentReference colorAttachmentReference = {};
  colorAttachmentReference.attachment = 1; // we only have one color attachment,
                                           // so index is 1
  colorAttachmentReference.layout =
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // optimal layout for color
                                                // attachment during subpass

  // depth attachment (input) reference
  VkAttachmentReference depthAttachmentReference = {};
  depthAttachmentReference.attachment = 2; // we only have one depth attachment,
                                           // so index is 2
  depthAttachmentReference.layout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; // optimal layout for
                                                        // depth attachment
                                                        // during subpass

  subpasses[0].pipelineBindPoint =
      VK_PIPELINE_BIND_POINT_GRAPHICS;   // this is a graphics subpass
  subpasses[0].colorAttachmentCount = 1; // we have one color attachment
  subpasses[0].pColorAttachments =
      &colorAttachmentReference; // reference to color attachment in subpass
  subpasses[0].pDepthStencilAttachment =
      &depthAttachmentReference; // reference to depth attachment in subpass

  // Subpass 2 attachments + reference (output)

  // swapchain color attachment of render pass
  VkAttachmentDescription swapchainColorAttachment = {};
  swapchainColorAttachment.format =
      swapchainFormat; // format of color attachment
  swapchainColorAttachment.samples =
      VK_SAMPLE_COUNT_1_BIT; // number of samples to use
  swapchainColorAttachment.loadOp =
      VK_ATTACHMENT_LOAD_OP_CLEAR; // what to do with attachment before
                                   // rendering
  swapchainColorAttachment.storeOp =
      VK_ATTACHMENT_STORE_OP_STORE; // what to do with attachment after
                                    // rendering
  swapchainColorAttachment.stencilLoadOp =
      VK_ATTACHMENT_LOAD_OP_DONT_CARE; // what to do with stencil before
                                       // rendering (not using stencil)
  swapchainColorAttachment.stencilStoreOp =
      VK_ATTACHMENT_STORE_OP_DONT_CARE; // what to do with stencil after
                                        // rendering (not using stencil)

  // framebuffer images will be stored as an image but images can be given
  // different layout for optimal use, so we need to specify the layout of the
  // image during different stages of the render pass
  swapchainColorAttachment.initialLayout =
      VK_IMAGE_LAYOUT_UNDEFINED; // layout of attachment before rendering(first
                                 // transition)
  swapchainColorAttachment.finalLayout =
      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // layout of attachment after
                                       // rendering(layout of attachment when it
                                       // will be presented in the swap
                                       // chain(final transition))

  // attachment reference uses an attachment index to specify which attachment
  // to reference and the layout it will be in during a subpass
  VkAttachmentReference swapchainColorAttachmentReference = {};
  swapchainColorAttachmentReference.attachment = 0;
  swapchainColorAttachmentReference.layout =
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // layout of attachment during
                                                // subpass(second transition)

  // References to attachment that subpass will take input from
  std::array<VkAttachmentReference, 2> inputReferences;
  inputReferences[0].attachment = 1;
  inputReferences[0].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  inputReferences[1].attachment = 2;
  inputReferences[1].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  // setup subpass 2
  subpasses[1].pipelineBindPoint =
      VK_PIPELINE_BIND_POINT_GRAPHICS;   // this is a graphics subpass
  subpasses[1].colorAttachmentCount = 1; // we have one color attachment
  subpasses[1].pColorAttachments =
      &swapchainColorAttachmentReference; // reference to color attachment in
                                          // subpass
  subpasses[1].inputAttachmentCount = static_cast<uint32_t>(
      inputReferences.size()); // reference to input attachments in subpass
  subpasses[1].pInputAttachments = inputReferences.data();
  subpasses[1].pDepthStencilAttachment = nullptr;

  // SUBPASS DEPENDENCIES

  // determine subpass dependencies for layout transitions
  std::array<VkSubpassDependency, 3> subpassDependencies;
  // Conversion from VK_IMAGE_LAYOUT_UNDEFINED to
  // VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
  //  transition must happen after the render pass finishes and before the next
  //  render pass begins, so it must wait on the color attachment output stage
  //  of the first subpass to finish and must happen before the color attachment
  //  output stage of the second subpass begins

  //(after the previous render pass finishes)
  subpassDependencies[0].srcSubpass =
      VK_SUBPASS_EXTERNAL; // subpass index of source of dependency is external
                           // to the render pass
  subpassDependencies[0].srcStageMask =
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT; // wait for all operations to be
  // finished
  subpassDependencies[0].srcAccessMask =
      VK_ACCESS_MEMORY_READ_BIT; // wait until memory is no longer being read
  // (before starting the next render pass)
  subpassDependencies[0].dstSubpass = 0; // our subpass is the destination
  subpassDependencies[0].dstStageMask =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // before starting color
                                                     // attachment output stage
  subpassDependencies[0].dstAccessMask =
      VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; // wait until color attachment is no
  // longer being read from or written
  subpassDependencies[0].dependencyFlags = 0;

  // subpass 1 layout (color/deapth) to subpass 2 layout (shader read)
  subpassDependencies[1].srcSubpass = 0; // our subpass is the source
  subpassDependencies[1].srcStageMask =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // wait for color
                                                     // attachment output stage
  subpassDependencies[1].srcAccessMask =
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; // wait until color attachment is no
                                            // longer being written to
  subpassDependencies[1].dstSubpass = 1;    // subpass 2 is the destination
  subpassDependencies[1].dstStageMask =
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT; // before starting fragment shader
  subpassDependencies[1].dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT; // wait until shader is no longer reading from
  // attachment before starting to write to it in
  // the next subpass
  subpassDependencies[1].dependencyFlags = 0;

  // Conversion from VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL to
  // VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
  subpassDependencies[2].srcSubpass = 0; // our subpass is the source
  subpassDependencies[2].srcStageMask =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // wait for color
                                                     // attachment output stage
  subpassDependencies[2].srcAccessMask =
      VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; // wait until color attachment is no
                                            // longer being read from or written
  // before starting the next render pass
  subpassDependencies[2].dstSubpass =
      VK_SUBPASS_EXTERNAL; // subpass index of destination of dependency is
                           // external to the render pass
  subpassDependencies[2].dstStageMask =
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT; // before starting the next render
                                            // pass
  subpassDependencies[2].dstAccessMask =
      VK_ACCESS_MEMORY_READ_BIT; // wait until memory is no longer being read
  subpassDependencies[2].dependencyFlags = 0;

  std::array<VkAttachmentDescription, 3> renderPassAttachments = {
      swapchainColorAttachment,
      colorAttachment,
      depthAttachment,
  };

  // create info for render pass creation
  VkRenderPassCreateInfo renderPassCreateInfo = {};
  renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassCreateInfo.attachmentCount = static_cast<uint32_t>(
      renderPassAttachments.size()); // number of attachments in render pass
  renderPassCreateInfo.pAttachments =
      renderPassAttachments.data(); // list of attachments in render pass
  renderPassCreateInfo.subpassCount = static_cast<uint32_t>(
      subpasses.size()); // number of subpasses in render pass
  renderPassCreateInfo.pSubpasses =
      subpasses.data(); // list of subpasses in render pass
  renderPassCreateInfo.dependencyCount =
      static_cast<uint32_t>(subpassDependencies.size()); // number of subpass
                                                         // dependencies
  renderPassCreateInfo.pDependencies =
      subpassDependencies.data(); // list of subpass dependencies

  VkResult result = vkCreateRenderPass(device, &renderPassCreateInfo, nullptr,
                                       &renderPass);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create render pass");
  }
}

void RenderPassManager::cleanup(VkDevice device) {
  vkDestroyRenderPass(device, renderPass, nullptr);
}
