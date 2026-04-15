#include "VulkanRenderer.h"
#include "Model.h"
#include "Utilities.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <vector>

VulkanRenderer::VulkanRenderer() {}

int VulkanRenderer::init(GLFWwindow *newWindow) {
  window = newWindow;

  try {
    // 1. Device (instance, physical/logical device, surface, command pool)
    device.init(window);

    // 2. Determine formats needed for render pass (no Vulkan objects created)
    VkFormat swapchainFormat = swapchain.queryImageFormat(device);

    VkFormat colorFormat = swapchain.chooseSupportedFormat(
        device.getPhysicalDevice(), {VK_FORMAT_R8G8B8A8_UNORM},
        VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkFormat depthFormat = swapchain.chooseSupportedFormat(
        device.getPhysicalDevice(),
        {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D32_SFLOAT,
         VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);

    // 3. Render pass (needs formats but no swapchain)
    renderPassManager.createRenderPass(device.getLogicalDevice(),
                                       swapchainFormat, colorFormat,
                                       depthFormat);

    // 4. Swapchain (created once with valid render pass)
    swapchain.init(device, renderPassManager.getRenderPass(), window);

    // 6. Descriptors
    descriptorManager.init(device.getLogicalDevice(),
                           device.getPhysicalDevice(),
                           swapchain.getImageCount());
    descriptorManager.recreateInputSets(device.getLogicalDevice(), swapchain);

    // 7. Pipeline
    pipeline.createPipelines(device.getLogicalDevice(),
                             renderPassManager.getRenderPass(),
                             swapchain.getExtent(), descriptorManager);

    // 8. Init Resource Managers
    textureManager.init(device);

    // 9. Synchronization
    createSynchronization();

    // 10. Projection matrix
    uboViewProjection.projection = glm::perspective(
        glm::radians(45.0f),
        (float)swapchain.getExtent().width /
            (float)swapchain.getExtent().height,
        0.1f, 100.0f);

    uboViewProjection.view =
        glm::lookAt(glm::vec3(10.0f, .0f, 20.0f),
                    glm::vec3(0.0f, 0.0f, -2.0f),
                    glm::vec3(0.0f, 1.0f, 0.0f));

    uboViewProjection.projection[1][1] *=
        -1; // invert the y coordinate of the clip coordinates, because
            // vulkan has a different coordinate system than opengl

    // 11. Default no-texture texture
    textureManager.loadTexture("plain.png", device, descriptorManager);

  } catch (const std::runtime_error &e) {
    printf("%s\n", e.what());
    return EXIT_FAILURE;
  }
  return 0;
}

int VulkanRenderer::createMeshModel(const std::string &modelFile) {
  return modelManager.loadModel(modelFile, device, textureManager,
                                descriptorManager);
}

SceneNode &VulkanRenderer::getRootNode() { return rootNode; }

void VulkanRenderer::updateCameraView(const glm::mat4 &viewMatrix) {
  uboViewProjection.view = viewMatrix;
}

void VulkanRenderer::notifyResize() { framebufferResized = true; }

void VulkanRenderer::draw() {
  VkDevice logicalDevice = device.getLogicalDevice();

  // wait for the fence to signal that command buffer has finished executing
  vkWaitForFences(logicalDevice, 1, &drawFences[currentFrame], VK_TRUE,
                  std::numeric_limits<uint64_t>::max());

  // acquire an image from the swap chain and signal when it is available
  uint32_t imageIndex;
  VkResult acquireResult = vkAcquireNextImageKHR(
      logicalDevice, swapchain.getSwapchain(),
      std::numeric_limits<uint64_t>::max(), imageAvailable[currentFrame],
      VK_NULL_HANDLE, &imageIndex);

  if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR || framebufferResized) {
    framebufferResized = false;
    recreateSwapChain();
    return;
  } else if (acquireResult != VK_SUCCESS &&
             acquireResult != VK_SUBOPTIMAL_KHR) {
    throw std::runtime_error("Failed to acquire swap chain image");
  }

  if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
    vkWaitForFences(logicalDevice, 1, &imagesInFlight[imageIndex], VK_TRUE,
                    UINT64_MAX);
  }

  imagesInFlight[imageIndex] = drawFences[currentFrame];

  vkResetFences(logicalDevice, 1, &drawFences[currentFrame]);

  recordCommands(imageIndex);

  descriptorManager.updateUniformBuffer(logicalDevice, imageIndex,
                                        &uboViewProjection,
                                        sizeof(UboViewProjection));

  // submit command buffer to graphics queue for execution
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = &imageAvailable[currentFrame];
  VkPipelineStageFlags waitStages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  submitInfo.pWaitDstStageMask = waitStages;
  submitInfo.commandBufferCount = 1;
  VkCommandBuffer cmdBuffer = swapchain.getCommandBuffer(imageIndex);
  submitInfo.pCommandBuffers = &cmdBuffer;
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = &renderFinished[currentFrame];

  VkResult result = vkQueueSubmit(device.getGraphicsQueue(), 1, &submitInfo,
                                  drawFences[currentFrame]);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to submit draw command buffer");
  }

  // present the current image to the swap chain
  VkPresentInfoKHR presentInfo = {};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &renderFinished[currentFrame];
  presentInfo.swapchainCount = 1;
  VkSwapchainKHR sc = swapchain.getSwapchain();
  presentInfo.pSwapchains = &sc;
  presentInfo.pImageIndices = &imageIndex;

  result = vkQueuePresentKHR(device.getPresentationQueue(), &presentInfo);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
      framebufferResized) {
    framebufferResized = false;
    recreateSwapChain();
    return;
  } else if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to present swap chain image");
  }
  currentFrame = (currentFrame + 1) % MAX_FRAMES_DRAWS;
}

void VulkanRenderer::recreateSwapChain() {
  swapchain.recreate(device, renderPassManager.getRenderPass(), window);

  imagesInFlight.assign(swapchain.getImageCount(), VK_NULL_HANDLE);

  // Re-create input descriptor sets so they reference the new
  // color/depth image views instead of the destroyed ones
  descriptorManager.recreateInputSets(device.getLogicalDevice(), swapchain);
}

void VulkanRenderer::cleanup() {
  vkDeviceWaitIdle(device.getLogicalDevice());

  modelManager.cleanup();
  textureManager.cleanup(device.getLogicalDevice());

  // Destroy subsystems (reverse order of creation)
  descriptorManager.cleanup(device.getLogicalDevice(),
                            swapchain.getImageCount());
  swapchain.cleanup(device.getLogicalDevice());

  // Synchronization
  for (size_t i = 0; i < MAX_FRAMES_DRAWS; i++) {
    vkDestroySemaphore(device.getLogicalDevice(), renderFinished[i], nullptr);
    vkDestroySemaphore(device.getLogicalDevice(), imageAvailable[i], nullptr);
    vkDestroyFence(device.getLogicalDevice(), drawFences[i], nullptr);
  }

  pipeline.cleanup(device.getLogicalDevice());
  renderPassManager.cleanup(device.getLogicalDevice());
  device.cleanup();
}

VulkanRenderer::~VulkanRenderer() {}

// --- Command recording ---

void VulkanRenderer::recordCommands(uint32_t currentImage) {
  VkCommandBuffer cmdBuffer = swapchain.getCommandBuffer(currentImage);
  vkResetCommandBuffer(cmdBuffer, 0);

  VkCommandBufferBeginInfo bufferBeginInfo = {};
  bufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  VkRenderPassBeginInfo renderPassBeginInfo = {};
  renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassBeginInfo.renderPass = renderPassManager.getRenderPass();
  renderPassBeginInfo.renderArea.offset = {0, 0};
  renderPassBeginInfo.renderArea.extent = swapchain.getExtent();

  std::array<VkClearValue, 3> clearValues = {};
  clearValues[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
  clearValues[1].color = {0.6f, 0.65f, 0.4f, 1.0f};
  clearValues[2].depthStencil.depth = 1.0f;
  renderPassBeginInfo.pClearValues = clearValues.data();
  renderPassBeginInfo.clearValueCount =
      static_cast<uint32_t>(clearValues.size());

  renderPassBeginInfo.framebuffer = swapchain.getFramebuffer(currentImage);

  VkResult result = vkBeginCommandBuffer(cmdBuffer, &bufferBeginInfo);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to begin recording command buffer");
  }

  vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo,
                       VK_SUBPASS_CONTENTS_INLINE);

  // bind the graphics pipeline so that it will be used for rendering
  vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipeline.getGraphicsPipeline());
  auto renderNode = [&](auto &self, SceneNode *node) -> void {
    if (node->getModelId() >= 0) {
      MeshModel *thisModel = modelManager.getModel(node->getModelId());
      if (thisModel) {
        glm::mat4 modelMatrix = node->getGlobalTransform();

        vkCmdPushConstants(cmdBuffer, pipeline.getGraphicsLayout(),
                           VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Model),
                           &modelMatrix);

        for (size_t k = 0; k < thisModel->getMeshCount(); k++) {
          const Mesh *mesh = thisModel->getMesh(k);
          VkBuffer vertexBuffers[] = {mesh->getVertexBuffer()};
          VkDeviceSize offsets[] = {0};
          vkCmdBindVertexBuffers(cmdBuffer, 0, 1, vertexBuffers, offsets);
          vkCmdBindIndexBuffer(cmdBuffer, mesh->getIndexBuffer(), 0,
                               VK_INDEX_TYPE_UINT32);

          std::array<VkDescriptorSet, 2> descriptorSetGroup = {
              descriptorManager.getVPSet(currentImage),
              descriptorManager.getSamplerSet(
                  mesh->getMaterial().albedoTextureId)};

          vkCmdBindDescriptorSets(
              cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
              pipeline.getGraphicsLayout(), 0,
              static_cast<uint32_t>(descriptorSetGroup.size()),
              descriptorSetGroup.data(), 0, nullptr);

          vkCmdDrawIndexed(cmdBuffer, mesh->getIndexCount(), 1, 0, 0, 0);
        }
      }
    }
    for (auto &child : node->getChildren()) {
      self(self, child.get());
    }
  };

  renderNode(renderNode, &rootNode);

  // start second subpass to use input attachments
  vkCmdNextSubpass(cmdBuffer, VK_SUBPASS_CONTENTS_INLINE);

  vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipeline.getSecondPipeline());

  VkDescriptorSet inputSet = descriptorManager.getInputSet(currentImage);
  vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipeline.getSecondLayout(), 0, 1, &inputSet, 0,
                          nullptr);

  vkCmdDraw(cmdBuffer, 3, 1, 0, 0);

  vkCmdEndRenderPass(cmdBuffer);

  result = vkEndCommandBuffer(cmdBuffer);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to stop recording command buffer");
  }
}

// --- Synchronization ---

void VulkanRenderer::createSynchronization() {
  imageAvailable.resize(MAX_FRAMES_DRAWS);
  renderFinished.resize(MAX_FRAMES_DRAWS);
  drawFences.resize(MAX_FRAMES_DRAWS);
  imagesInFlight.resize(swapchain.getImageCount(), VK_NULL_HANDLE);

  VkSemaphoreCreateInfo semaphoreCreateInfo = {};
  semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkFenceCreateInfo fenceCreateInfo = {};
  fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  VkDevice logicalDevice = device.getLogicalDevice();
  for (size_t i = 0; i < MAX_FRAMES_DRAWS; i++) {
    if (vkCreateSemaphore(logicalDevice, &semaphoreCreateInfo, nullptr,
                          &imageAvailable[i]) != VK_SUCCESS ||
        vkCreateSemaphore(logicalDevice, &semaphoreCreateInfo, nullptr,
                          &renderFinished[i]) != VK_SUCCESS ||
        vkCreateFence(logicalDevice, &fenceCreateInfo, nullptr,
                      &drawFences[i]) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create synchronization primitives");
    }
  }
}

