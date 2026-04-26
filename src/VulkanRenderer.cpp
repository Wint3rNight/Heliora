#include "VulkanRenderer.h"
#include "Model.h"
#include "Utilities.h"
#include <spdlog/spdlog.h>

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
    // 1. Device
    device.init(window);

    // 2. Determine formats
    VkFormat swapchainFormat = swapchain.queryImageFormat(device);

    VkFormat colorFormat = swapchain.chooseSupportedFormat(
        device.getPhysicalDevice(),
        {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM},
        VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT);

    shadowDepthFormat = swapchain.chooseSupportedFormat(
        device.getPhysicalDevice(),
        {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D32_SFLOAT,
         VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);

    gBufferDepthFormat = shadowDepthFormat;

    VkFormat gb0Fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormat gb1Fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormat gb2Fmt = VK_FORMAT_R8G8B8A8_UNORM;

    // 3. Render passes
    renderPassManager.createGBufferRenderPass(
        device.getLogicalDevice(), gb0Fmt, gb1Fmt, gb2Fmt, gBufferDepthFormat);
    renderPassManager.createRenderPass(device.getLogicalDevice(),
                                       swapchainFormat, colorFormat);
    renderPassManager.createShadowRenderPass(device.getLogicalDevice(),
                                             shadowDepthFormat);

    // 4. Swapchain
    swapchain.init(device, renderPassManager.getRenderPass(), window);

    // 5. Shadow resources (directional + point)
    createShadowResources();

    // 6. Descriptors
    descriptorManager.init(device.getLogicalDevice(), device.getAllocator(),
                           swapchain.getImageCount());
    descriptorManager.recreateInputSets(device.getLogicalDevice(), swapchain);
    descriptorManager.updateShadowMapDescriptor(
        device.getLogicalDevice(), shadowDepthImageView.get(),
        pointShadowCubeView.get(), shadowSampler);

    // 7. Texture manager (samplers only at this point)
    textureManager.init(device);

    // 8. G-buffer images, framebuffers, descriptor sets
    createGBuffer();

    // 9. Pipeline (needs all 3 render passes + descriptor layouts)
    pipeline.createPipelines(device.getLogicalDevice(),
                             renderPassManager.getGBufferRenderPass(),
                             renderPassManager.getRenderPass(),
                             renderPassManager.getShadowRenderPass(),
                             swapchain.getExtent(), descriptorManager);

    // 10. IBL + SSAO resources (requires textureManager and descriptorManager)
    initIBL();

    // 11. Synchronization
    createSynchronization();

    // 12. Performance metrics
    QueueFamilyIndices qi = device.getQueueFamilies();
    metrics.init(device.getLogicalDevice(), device.getPhysicalDevice(),
                 static_cast<uint32_t>(qi.graphicsFamily));

    // 13. Scene UBO defaults
    sceneUbo.projection = glm::perspective(
        glm::radians(45.0f),
        static_cast<float>(swapchain.getExtent().width) /
            static_cast<float>(swapchain.getExtent().height),
        0.1f, 100.0f);
    sceneUbo.projection[1][1] *= -1;

    sceneUbo.view = glm::lookAt(glm::vec3(10.0f, 0.0f, 20.0f),
                                glm::vec3(0.0f, 0.0f, -2.0f),
                                glm::vec3(0.0f, 1.0f, 0.0f));
    sceneUbo.invView = glm::inverse(sceneUbo.view);
    sceneUbo.invProj = glm::inverse(sceneUbo.projection);

    sceneUbo.cameraPosition = glm::vec4(10.0f, 0.0f, 20.0f, 1.0f);

    sceneUbo.directionalLight.direction =
        glm::vec4(glm::normalize(glm::vec3(-0.35f, -1.0f, -0.25f)), 0.0f);
    sceneUbo.directionalLight.colorIntensity = glm::vec4(1.0f, 0.96f, 0.88f, 0.55f);

    sceneUbo.pointLights[0].position       = glm::vec4(2.0f,  3.0f, 5.0f, 1.0f);
    sceneUbo.pointLights[0].colorIntensity = glm::vec4(0.55f, 0.75f, 1.0f, 2.0f);
    sceneUbo.pointLights[1].position       = glm::vec4(-4.0f, 2.0f, 1.0f, 1.0f);
    sceneUbo.pointLights[1].colorIntensity = glm::vec4(1.0f,  0.55f, 0.42f, 1.2f);

    sceneUbo.spotLights[0].position       = glm::vec4(0.0f, 5.0f, 6.0f, 1.0f);
    sceneUbo.spotLights[0].direction      = glm::vec4(glm::normalize(glm::vec3(0.0f, -0.75f, -1.0f)), 0.0f);
    sceneUbo.spotLights[0].colorIntensity = glm::vec4(1.0f, 0.92f, 0.75f, 2.0f);
    sceneUbo.spotLights[0].cutoffAngles   = glm::vec4(
        glm::cos(glm::radians(12.5f)), glm::cos(glm::radians(20.0f)), 0.0f, 0.0f);

    sceneUbo.shadowParams = glm::vec4(25.0f, 0.0f, 0.0f, 0.0f);
    sceneUbo.lightCounts  = glm::ivec4(2, 1, 0, 0);

    updateLightSpaceMatrix();
    updatePointShadowMatrices();

    // 14. Default albedo texture
    textureManager.loadTexture("plain.png", device, descriptorManager);

  } catch (const std::runtime_error &e) {
    spdlog::critical("Renderer initialization failed: {}", e.what());
    return EXIT_FAILURE;
  }
  return 0;
}

int VulkanRenderer::createMeshModel(const std::string &modelFile) {
  return modelManager.loadModel(modelFile, device, textureManager, descriptorManager);
}

SceneNode &VulkanRenderer::getRootNode() { return rootNode; }

void VulkanRenderer::updateCameraView(const glm::mat4 &viewMatrix,
                                      const glm::vec3 &cameraPosition) {
  sceneUbo.view           = viewMatrix;
  sceneUbo.cameraPosition = glm::vec4(cameraPosition, 1.0f);
  sceneUbo.invView        = glm::inverse(viewMatrix);
  sceneUbo.invProj        = glm::inverse(sceneUbo.projection);
}

void VulkanRenderer::notifyResize() { framebufferResized = true; }

// ---------------------------------------------------------------------------
// Draw
// ---------------------------------------------------------------------------

void VulkanRenderer::draw() {
  VkDevice logicalDevice = device.getLogicalDevice();

  vkWaitForFences(logicalDevice, 1, &drawFences[currentFrame], VK_TRUE,
                  std::numeric_limits<uint64_t>::max());

  metrics.beginFrame();

  uint32_t imageIndex;
  VkResult acquireResult = vkAcquireNextImageKHR(
      logicalDevice, swapchain.getSwapchain(),
      std::numeric_limits<uint64_t>::max(), imageAvailable[currentFrame],
      VK_NULL_HANDLE, &imageIndex);

  if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR || framebufferResized) {
    framebufferResized = false;
    recreateSwapChain();
    return;
  } else if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
    throw std::runtime_error("Failed to acquire swap chain image");
  }

  if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)
    vkWaitForFences(logicalDevice, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);

  imagesInFlight[imageIndex] = drawFences[currentFrame];
  vkResetFences(logicalDevice, 1, &drawFences[currentFrame]);

  recordCommands(imageIndex);

  descriptorManager.updateUniformBuffer(device.getAllocator(), imageIndex,
                                        &sceneUbo, sizeof(SceneUniformBuffer));

  VkSubmitInfo submitInfo = {};
  submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.waitSemaphoreCount   = 1;
  submitInfo.pWaitSemaphores      = &imageAvailable[currentFrame];
  VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
  submitInfo.pWaitDstStageMask    = waitStages;
  submitInfo.commandBufferCount   = 1;
  VkCommandBuffer cmdBuffer       = swapchain.getCommandBuffer(imageIndex);
  submitInfo.pCommandBuffers      = &cmdBuffer;
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores    = &renderFinished[currentFrame];

  if (vkQueueSubmit(device.getGraphicsQueue(), 1, &submitInfo, drawFences[currentFrame]) != VK_SUCCESS)
    throw std::runtime_error("Failed to submit draw command buffer");

  VkPresentInfoKHR presentInfo = {};
  presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores    = &renderFinished[currentFrame];
  presentInfo.swapchainCount     = 1;
  VkSwapchainKHR sc              = swapchain.getSwapchain();
  presentInfo.pSwapchains        = &sc;
  presentInfo.pImageIndices      = &imageIndex;

  VkResult result = vkQueuePresentKHR(device.getPresentationQueue(), &presentInfo);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
    framebufferResized = false;
    recreateSwapChain();
    return;
  } else if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to present swap chain image");
  }
  currentFrame = (currentFrame + 1) % MAX_FRAMES_DRAWS;
  metrics.endFrame(logicalDevice);
}

// ---------------------------------------------------------------------------
// Swapchain recreation
// ---------------------------------------------------------------------------

void VulkanRenderer::recreateSwapChain() {
  swapchain.recreate(device, renderPassManager.getRenderPass(), window);

  cleanupGBuffer();
  createGBuffer();

  imagesInFlight.assign(swapchain.getImageCount(), VK_NULL_HANDLE);

  descriptorManager.recreateInputSets(device.getLogicalDevice(), swapchain);

  sceneUbo.projection = glm::perspective(
      glm::radians(45.0f),
      static_cast<float>(swapchain.getExtent().width) /
          static_cast<float>(swapchain.getExtent().height),
      0.1f, 100.0f);
  sceneUbo.projection[1][1] *= -1;
  sceneUbo.invProj = glm::inverse(sceneUbo.projection);
}

// ---------------------------------------------------------------------------
// G-buffer resource management
// ---------------------------------------------------------------------------

void VulkanRenderer::createGBuffer() {
  size_t   count  = swapchain.getImageCount();
  VkDevice dev    = device.getLogicalDevice();
  VkFormat gb0Fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
  VkFormat gb1Fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
  VkFormat gb2Fmt = VK_FORMAT_R8G8B8A8_UNORM;

  gBuffer0Images.resize(count);   gBuffer0Views.resize(count);
  gBuffer1Images.resize(count);   gBuffer1Views.resize(count);
  gBuffer2Images.resize(count);   gBuffer2Views.resize(count);
  gBufferDepthImages.resize(count); gBufferDepthViews.resize(count);
  gBufferFramebuffers.resize(count, VK_NULL_HANDLE);

  VmaAllocationCreateInfo aci = {};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  auto makeColorImg = [&](AllocatedImage &img, ImageViewHandle &view, VkFormat fmt) {
    VkImageCreateInfo ci = {};
    ci.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType   = VK_IMAGE_TYPE_2D;
    ci.extent      = { swapchain.getExtent().width, swapchain.getExtent().height, 1 };
    ci.mipLevels   = 1; ci.arrayLayers = 1;
    ci.format      = fmt;
    ci.tiling      = VK_IMAGE_TILING_OPTIMAL;
    ci.usage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.samples     = VK_SAMPLE_COUNT_1_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkImage rawImg = VK_NULL_HANDLE; VmaAllocation alloc = VK_NULL_HANDLE;
    if (vmaCreateImage(device.getAllocator(), &ci, &aci, &rawImg, &alloc, nullptr) != VK_SUCCESS)
      throw std::runtime_error("Failed to create G-buffer color image");
    img = AllocatedImage(device.getAllocator(), rawImg, alloc);

    VkImageViewCreateInfo vci = {};
    vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image    = img.get();
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = fmt;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VkImageView v;
    if (vkCreateImageView(dev, &vci, nullptr, &v) != VK_SUCCESS)
      throw std::runtime_error("Failed to create G-buffer color view");
    view = ImageViewHandle(dev, v);
  };

  auto makeDepthImg = [&](AllocatedImage &img, ImageViewHandle &view) {
    VkImageCreateInfo ci = {};
    ci.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType   = VK_IMAGE_TYPE_2D;
    ci.extent      = { swapchain.getExtent().width, swapchain.getExtent().height, 1 };
    ci.mipLevels   = 1; ci.arrayLayers = 1;
    ci.format      = gBufferDepthFormat;
    ci.tiling      = VK_IMAGE_TILING_OPTIMAL;
    ci.usage       = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.samples     = VK_SAMPLE_COUNT_1_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkImage rawImg = VK_NULL_HANDLE; VmaAllocation alloc = VK_NULL_HANDLE;
    if (vmaCreateImage(device.getAllocator(), &ci, &aci, &rawImg, &alloc, nullptr) != VK_SUCCESS)
      throw std::runtime_error("Failed to create G-buffer depth image");
    img = AllocatedImage(device.getAllocator(), rawImg, alloc);

    VkImageViewCreateInfo vci = {};
    vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image    = img.get();
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = gBufferDepthFormat;
    vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
    VkImageView v;
    if (vkCreateImageView(dev, &vci, nullptr, &v) != VK_SUCCESS)
      throw std::runtime_error("Failed to create G-buffer depth view");
    view = ImageViewHandle(dev, v);
  };

  for (size_t i = 0; i < count; i++) {
    makeColorImg(gBuffer0Images[i], gBuffer0Views[i], gb0Fmt);
    makeColorImg(gBuffer1Images[i], gBuffer1Views[i], gb1Fmt);
    makeColorImg(gBuffer2Images[i], gBuffer2Views[i], gb2Fmt);
    makeDepthImg(gBufferDepthImages[i], gBufferDepthViews[i]);

    std::array<VkImageView, 4> attachments = {
        gBuffer0Views[i].get(), gBuffer1Views[i].get(),
        gBuffer2Views[i].get(), gBufferDepthViews[i].get()
    };
    VkFramebufferCreateInfo fbci = {};
    fbci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbci.renderPass      = renderPassManager.getGBufferRenderPass();
    fbci.attachmentCount = static_cast<uint32_t>(attachments.size());
    fbci.pAttachments    = attachments.data();
    fbci.width           = swapchain.getExtent().width;
    fbci.height          = swapchain.getExtent().height;
    fbci.layers          = 1;
    if (vkCreateFramebuffer(dev, &fbci, nullptr, &gBufferFramebuffers[i]) != VK_SUCCESS)
      throw std::runtime_error("Failed to create G-buffer framebuffer");
  }

  std::vector<VkImageView> gb0v, gb1v, gb2v, depv;
  gb0v.reserve(count); gb1v.reserve(count); gb2v.reserve(count); depv.reserve(count);
  for (size_t i = 0; i < count; i++) {
    gb0v.push_back(gBuffer0Views[i].get());
    gb1v.push_back(gBuffer1Views[i].get());
    gb2v.push_back(gBuffer2Views[i].get());
    depv.push_back(gBufferDepthViews[i].get());
  }
  descriptorManager.recreateGBufferSets(device.getLogicalDevice(),
                                        gb0v, gb1v, gb2v, depv,
                                        textureManager.getTextureSampler());
}

void VulkanRenderer::cleanupGBuffer() {
  VkDevice dev = device.getLogicalDevice();
  for (VkFramebuffer fb : gBufferFramebuffers)
    if (fb != VK_NULL_HANDLE) vkDestroyFramebuffer(dev, fb, nullptr);
  gBufferFramebuffers.clear();
  gBufferDepthViews.clear();  gBufferDepthImages.clear();
  gBuffer2Views.clear();      gBuffer2Images.clear();
  gBuffer1Views.clear();      gBuffer1Images.clear();
  gBuffer0Views.clear();      gBuffer0Images.clear();
}

// ---------------------------------------------------------------------------
// IBL resource management
// ---------------------------------------------------------------------------

void VulkanRenderer::initIBL() {
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(device.getPhysicalDevice(), &props);

  // Sampler for irradiance / prefiltered / brdf-lut / skybox (mip-capable)
  VkSamplerCreateInfo iblCi = {};
  iblCi.sType          = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  iblCi.magFilter      = VK_FILTER_LINEAR;
  iblCi.minFilter      = VK_FILTER_LINEAR;
  iblCi.mipmapMode     = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  iblCi.addressModeU   = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  iblCi.addressModeV   = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  iblCi.addressModeW   = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  iblCi.minLod         = 0.0f;
  iblCi.maxLod         = static_cast<float>(IBL_PREFILTER_MIPS);
  iblCi.maxAnisotropy  = 1.0f;
  if (vkCreateSampler(device.getLogicalDevice(), &iblCi, nullptr, &iblSampler) != VK_SUCCESS)
    throw std::runtime_error("Failed to create IBL sampler");

  // Sampler for 4x4 SSAO noise (tiled nearest)
  VkSamplerCreateInfo noiseCi = {};
  noiseCi.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  noiseCi.magFilter    = VK_FILTER_NEAREST;
  noiseCi.minFilter    = VK_FILTER_NEAREST;
  noiseCi.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  noiseCi.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  noiseCi.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  noiseCi.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  noiseCi.maxAnisotropy = 1.0f;
  if (vkCreateSampler(device.getLogicalDevice(), &noiseCi, nullptr, &ssaoNoiseSampler) != VK_SUCCESS)
    throw std::runtime_error("Failed to create SSAO noise sampler");

  spdlog::info("IBL: loading skybox and generating irradiance / prefiltered maps...");
  iblSkyboxImageIndex      = textureManager.loadSkybox(
      "Resources/HDRIs/kloppenheim_02_1k.hdr", device, descriptorManager);
  irradianceImageIndex     = textureManager.createIrradianceMap(iblSkyboxImageIndex, device);
  prefilteredEnvImageIndex = textureManager.createPrefilteredEnvMap(iblSkyboxImageIndex, device);
  brdfLutImageIndex        = textureManager.loadBrdfLut(device);
  ssaoNoiseImageIndex      = textureManager.createSsaoNoiseTexture(device);
  spdlog::info("IBL: done.");

  descriptorManager.updateIblDescriptors(
      device.getLogicalDevice(),
      textureManager.getImageView(irradianceImageIndex),     iblSampler,
      textureManager.getImageView(prefilteredEnvImageIndex),
      textureManager.getImageView(brdfLutImageIndex),
      textureManager.getImageView(ssaoNoiseImageIndex),      ssaoNoiseSampler,
      textureManager.getImageView(iblSkyboxImageIndex));
}

void VulkanRenderer::cleanupIBL() {
  VkDevice dev = device.getLogicalDevice();
  if (ssaoNoiseSampler != VK_NULL_HANDLE) {
    vkDestroySampler(dev, ssaoNoiseSampler, nullptr);
    ssaoNoiseSampler = VK_NULL_HANDLE;
  }
  if (iblSampler != VK_NULL_HANDLE) {
    vkDestroySampler(dev, iblSampler, nullptr);
    iblSampler = VK_NULL_HANDLE;
  }
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

void VulkanRenderer::cleanup() {
  vkDeviceWaitIdle(device.getLogicalDevice());

  metrics.printReport(device.getAllocator());

  modelManager.cleanup();
  textureManager.cleanup(device.getLogicalDevice(), device.getAllocator());
  cleanupGBuffer();
  cleanupIBL();
  cleanupShadowResources();

  descriptorManager.cleanup(device.getLogicalDevice(), device.getAllocator(),
                            swapchain.getImageCount());
  swapchain.cleanup(device.getLogicalDevice(), device.getAllocator());

  for (size_t i = 0; i < MAX_FRAMES_DRAWS; i++) {
    vkDestroySemaphore(device.getLogicalDevice(), renderFinished[i], nullptr);
    vkDestroySemaphore(device.getLogicalDevice(), imageAvailable[i], nullptr);
    vkDestroyFence(device.getLogicalDevice(), drawFences[i], nullptr);
  }

  pipeline.cleanup(device.getLogicalDevice());
  renderPassManager.cleanup(device.getLogicalDevice());
  metrics.cleanup(device.getLogicalDevice());
  device.cleanup();
}

VulkanRenderer::~VulkanRenderer() {}

// ---------------------------------------------------------------------------
// Command recording
// ---------------------------------------------------------------------------

void VulkanRenderer::recordCommands(uint32_t currentImage) {
  VkCommandBuffer cmd = swapchain.getCommandBuffer(currentImage);
  vkResetCommandBuffer(cmd, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
    throw std::runtime_error("Failed to begin recording command buffer");

  metrics.resetGpuQueries(cmd);

  // Shadow passes (unchanged)
  recordShadowPass(cmd);
  recordPointShadowPass(cmd);

  VkViewport viewport = {};
  viewport.width    = static_cast<float>(swapchain.getExtent().width);
  viewport.height   = static_cast<float>(swapchain.getExtent().height);
  viewport.maxDepth = 1.0f;
  VkRect2D scissor = { {0, 0}, swapchain.getExtent() };

  // --- G-buffer pass ---
  {
    std::array<VkClearValue, 4> clears{};
    clears[3].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo rpbi = {};
    rpbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass      = renderPassManager.getGBufferRenderPass();
    rpbi.framebuffer     = gBufferFramebuffers[currentImage];
    rpbi.renderArea      = { {0, 0}, swapchain.getExtent() };
    rpbi.clearValueCount = static_cast<uint32_t>(clears.size());
    rpbi.pClearValues    = clears.data();

    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.getGraphicsPipeline());

    auto renderNode = [&](auto &self, SceneNode *node) -> void {
      if (node->getModelId() >= 0) {
        MeshModel *mdl = modelManager.getModel(node->getModelId());
        if (mdl) {
          ModelPushConstants push{};
          push.model  = node->getGlobalTransform();
          push.normal = glm::transpose(glm::inverse(push.model));
          vkCmdPushConstants(cmd, pipeline.getGraphicsLayout(),
                             VK_SHADER_STAGE_VERTEX_BIT, 0,
                             sizeof(ModelPushConstants), &push);

          for (size_t k = 0; k < mdl->getMeshCount(); k++) {
            const Mesh *mesh = mdl->getMesh(k);
            VkBuffer     vb[]  = { mesh->getVertexBuffer() };
            VkDeviceSize off[] = { 0 };
            vkCmdBindVertexBuffers(cmd, 0, 1, vb, off);
            vkCmdBindIndexBuffer(cmd, mesh->getIndexBuffer(), 0, VK_INDEX_TYPE_UINT32);

            std::array<VkDescriptorSet, 2> sets = {
                descriptorManager.getVPSet(currentImage),
                descriptorManager.getSamplerSet(mesh->getMaterial().descriptorSetId)
            };
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeline.getGraphicsLayout(), 0, 2, sets.data(), 0, nullptr);
            vkCmdDrawIndexed(cmd, mesh->getIndexCount(), 1, 0, 0, 0);
            metrics.recordDrawCall(mesh->getIndexCount());
          }
        }
      }
      for (auto &child : node->getChildren()) self(self, child.get());
    };
    renderNode(renderNode, &rootNode);

    vkCmdEndRenderPass(cmd);
  }

  // --- Composition pass (deferred PBR + ACES tone-mapping) ---
  {
    std::array<VkClearValue, 2> clears{};
    clears[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
    clears[1].color = { 0.0f, 0.0f, 0.0f, 1.0f };

    VkRenderPassBeginInfo rpbi = {};
    rpbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass      = renderPassManager.getRenderPass();
    rpbi.framebuffer     = swapchain.getFramebuffer(currentImage);
    rpbi.renderArea      = { {0, 0}, swapchain.getExtent() };
    rpbi.clearValueCount = static_cast<uint32_t>(clears.size());
    rpbi.pClearValues    = clears.data();

    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    metrics.beginGpuTimestamp(cmd);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Subpass 0: deferred PBR + IBL + SSAO + bloom + FXAA → colorBuffer
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.getDeferredPipeline());
    std::array<VkDescriptorSet, 2> deferredSets = {
        descriptorManager.getVPSet(currentImage),
        descriptorManager.getGBufferSet(currentImage)
    };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getDeferredLayout(), 0, 2, deferredSets.data(), 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    // Subpass 1: ACES + gamma → swapchain
    vkCmdNextSubpass(cmd, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.getSecondPipeline());
    VkDescriptorSet inputSet = descriptorManager.getInputSet(currentImage);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getSecondLayout(), 0, 1, &inputSet, 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    metrics.endGpuTimestamp(cmd);
    vkCmdEndRenderPass(cmd);
  }

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
    throw std::runtime_error("Failed to stop recording command buffer");
}

// ---------------------------------------------------------------------------
// Shadow passes (unchanged)
// ---------------------------------------------------------------------------

void VulkanRenderer::recordShadowPass(VkCommandBuffer cmdBuffer) {
  VkClearValue clearValue = {};
  clearValue.depthStencil.depth = 1.0f;

  VkRenderPassBeginInfo rpbi = {};
  rpbi.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rpbi.renderPass        = renderPassManager.getShadowRenderPass();
  rpbi.framebuffer       = shadowFramebuffer;
  rpbi.renderArea.extent = { SHADOW_MAP_SIZE, SHADOW_MAP_SIZE };
  rpbi.clearValueCount   = 1;
  rpbi.pClearValues      = &clearValue;

  vkCmdBeginRenderPass(cmdBuffer, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

  VkViewport vp = { 0, 0, static_cast<float>(SHADOW_MAP_SIZE), static_cast<float>(SHADOW_MAP_SIZE), 0, 1 };
  VkRect2D   sc = { {0,0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE} };
  vkCmdSetViewport(cmdBuffer, 0, 1, &vp);
  vkCmdSetScissor(cmdBuffer, 0, 1, &sc);
  vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.getShadowPipeline());

  auto renderNodeShadow = [&](auto &self, SceneNode *node) -> void {
    if (node->getModelId() >= 0) {
      MeshModel *mdl = modelManager.getModel(node->getModelId());
      if (mdl) {
        ShadowPushConstants push{};
        push.model            = node->getGlobalTransform();
        push.lightSpaceMatrix = sceneUbo.lightSpaceMatrix;
        vkCmdPushConstants(cmdBuffer, pipeline.getShadowLayout(),
                           VK_SHADER_STAGE_VERTEX_BIT, 0,
                           sizeof(ShadowPushConstants), &push);
        for (size_t k = 0; k < mdl->getMeshCount(); k++) {
          const Mesh *mesh = mdl->getMesh(k);
          VkBuffer vb[] = { mesh->getVertexBuffer() }; VkDeviceSize off[] = { 0 };
          vkCmdBindVertexBuffers(cmdBuffer, 0, 1, vb, off);
          vkCmdBindIndexBuffer(cmdBuffer, mesh->getIndexBuffer(), 0, VK_INDEX_TYPE_UINT32);
          vkCmdDrawIndexed(cmdBuffer, mesh->getIndexCount(), 1, 0, 0, 0);
        }
      }
    }
    for (auto &child : node->getChildren()) self(self, child.get());
  };
  renderNodeShadow(renderNodeShadow, &rootNode);
  vkCmdEndRenderPass(cmdBuffer);
}

void VulkanRenderer::recordPointShadowPass(VkCommandBuffer cmdBuffer) {
  VkClearValue clearValue = {};
  clearValue.depthStencil.depth = 1.0f;

  for (uint32_t face = 0; face < 6; ++face) {
    VkRenderPassBeginInfo rpbi = {};
    rpbi.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass        = renderPassManager.getShadowRenderPass();
    rpbi.framebuffer       = pointShadowFramebuffers[face];
    rpbi.renderArea.extent = { POINT_SHADOW_MAP_SIZE, POINT_SHADOW_MAP_SIZE };
    rpbi.clearValueCount   = 1;
    rpbi.pClearValues      = &clearValue;

    vkCmdBeginRenderPass(cmdBuffer, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp = { 0, 0, static_cast<float>(POINT_SHADOW_MAP_SIZE),
                      static_cast<float>(POINT_SHADOW_MAP_SIZE), 0, 1 };
    VkRect2D   sc = { {0,0}, {POINT_SHADOW_MAP_SIZE, POINT_SHADOW_MAP_SIZE} };
    vkCmdSetViewport(cmdBuffer, 0, 1, &vp);
    vkCmdSetScissor(cmdBuffer, 0, 1, &sc);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.getShadowPipeline());

    auto renderNodeShadow = [&](auto &self, SceneNode *node) -> void {
      if (node->getModelId() >= 0) {
        MeshModel *mdl = modelManager.getModel(node->getModelId());
        if (mdl) {
          ShadowPushConstants push{};
          push.model            = node->getGlobalTransform();
          push.lightSpaceMatrix = pointShadowMatrices[face];
          vkCmdPushConstants(cmdBuffer, pipeline.getShadowLayout(),
                             VK_SHADER_STAGE_VERTEX_BIT, 0,
                             sizeof(ShadowPushConstants), &push);
          for (size_t k = 0; k < mdl->getMeshCount(); k++) {
            const Mesh *mesh = mdl->getMesh(k);
            VkBuffer vb[] = { mesh->getVertexBuffer() }; VkDeviceSize off[] = { 0 };
            vkCmdBindVertexBuffers(cmdBuffer, 0, 1, vb, off);
            vkCmdBindIndexBuffer(cmdBuffer, mesh->getIndexBuffer(), 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmdBuffer, mesh->getIndexCount(), 1, 0, 0, 0);
          }
        }
      }
      for (auto &child : node->getChildren()) self(self, child.get());
    };
    renderNodeShadow(renderNodeShadow, &rootNode);
    vkCmdEndRenderPass(cmdBuffer);
  }
}

// ---------------------------------------------------------------------------
// Light helpers (unchanged)
// ---------------------------------------------------------------------------

void VulkanRenderer::updateLightSpaceMatrix() {
  glm::vec3 lightDir      = glm::normalize(glm::vec3(sceneUbo.directionalLight.direction));
  glm::vec3 lightPosition = -lightDir * 18.0f + glm::vec3(0.0f, 2.0f, 0.0f);
  glm::mat4 lightView     = glm::lookAt(lightPosition, glm::vec3(0), glm::vec3(0,1,0));
  glm::mat4 lightProj     = glm::ortho(-14.0f, 14.0f, -14.0f, 14.0f, 0.1f, 45.0f);
  lightProj[1][1] *= -1.0f;
  sceneUbo.lightSpaceMatrix = lightProj * lightView;
}

void VulkanRenderer::updatePointShadowMatrices() {
  pointShadowMatrices.resize(6);
  glm::vec3 lp       = glm::vec3(sceneUbo.pointLights[0].position);
  float     farPlane = sceneUbo.shadowParams.x;
  glm::mat4 proj     = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, farPlane);
  proj[1][1] *= -1.0f;

  pointShadowMatrices[0] = proj * glm::lookAt(lp, lp + glm::vec3( 1, 0, 0), glm::vec3(0,-1, 0));
  pointShadowMatrices[1] = proj * glm::lookAt(lp, lp + glm::vec3(-1, 0, 0), glm::vec3(0,-1, 0));
  pointShadowMatrices[2] = proj * glm::lookAt(lp, lp + glm::vec3( 0, 1, 0), glm::vec3(0, 0, 1));
  pointShadowMatrices[3] = proj * glm::lookAt(lp, lp + glm::vec3( 0,-1, 0), glm::vec3(0, 0,-1));
  pointShadowMatrices[4] = proj * glm::lookAt(lp, lp + glm::vec3( 0, 0, 1), glm::vec3(0,-1, 0));
  pointShadowMatrices[5] = proj * glm::lookAt(lp, lp + glm::vec3( 0, 0,-1), glm::vec3(0,-1, 0));

  for (size_t i = 0; i < 6; ++i)
    sceneUbo.pointShadowMatrices[i] = pointShadowMatrices[i];
}

// ---------------------------------------------------------------------------
// Shadow resource creation (unchanged)
// ---------------------------------------------------------------------------

void VulkanRenderer::createShadowResources() {
  VkImageCreateInfo imageCreateInfo = {};
  imageCreateInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageCreateInfo.imageType     = VK_IMAGE_TYPE_2D;
  imageCreateInfo.extent        = { SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1 };
  imageCreateInfo.mipLevels     = 1;
  imageCreateInfo.arrayLayers   = 1;
  imageCreateInfo.format        = shadowDepthFormat;
  imageCreateInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
  imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageCreateInfo.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  imageCreateInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
  imageCreateInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo allocCreateInfo = {};
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  VkImage img = VK_NULL_HANDLE; VmaAllocation alloc = VK_NULL_HANDLE;
  if (vmaCreateImage(device.getAllocator(), &imageCreateInfo, &allocCreateInfo, &img, &alloc, nullptr) != VK_SUCCESS)
    throw std::runtime_error("Failed to create shadow depth image");
  shadowDepthImage = AllocatedImage(device.getAllocator(), img, alloc);

  VkImageViewCreateInfo viewInfo = {};
  viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image    = shadowDepthImage.get();
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format   = shadowDepthFormat;
  viewInfo.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
  VkImageView view = VK_NULL_HANDLE;
  if (vkCreateImageView(device.getLogicalDevice(), &viewInfo, nullptr, &view) != VK_SUCCESS)
    throw std::runtime_error("Failed to create shadow image view");
  shadowDepthImageView = ImageViewHandle(device.getLogicalDevice(), view);

  VkSamplerCreateInfo samplerInfo = {};
  samplerInfo.sType         = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter     = VK_FILTER_LINEAR;
  samplerInfo.minFilter     = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerInfo.addressModeU  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  samplerInfo.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  samplerInfo.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  samplerInfo.borderColor   = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
  samplerInfo.maxLod        = 1.0f;
  samplerInfo.maxAnisotropy = 1.0f;
  if (vkCreateSampler(device.getLogicalDevice(), &samplerInfo, nullptr, &shadowSampler) != VK_SUCCESS)
    throw std::runtime_error("Failed to create shadow sampler");

  VkFramebufferCreateInfo fbInfo = {};
  fbInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  fbInfo.renderPass      = renderPassManager.getShadowRenderPass();
  fbInfo.attachmentCount = 1;
  VkImageView att        = shadowDepthImageView.get();
  fbInfo.pAttachments    = &att;
  fbInfo.width           = SHADOW_MAP_SIZE;
  fbInfo.height          = SHADOW_MAP_SIZE;
  fbInfo.layers          = 1;
  if (vkCreateFramebuffer(device.getLogicalDevice(), &fbInfo, nullptr, &shadowFramebuffer) != VK_SUCCESS)
    throw std::runtime_error("Failed to create shadow framebuffer");

  // Point shadow cubemap
  VkImageCreateInfo cubeInfo = imageCreateInfo;
  cubeInfo.flags       = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  cubeInfo.extent      = { POINT_SHADOW_MAP_SIZE, POINT_SHADOW_MAP_SIZE, 1 };
  cubeInfo.arrayLayers = 6;

  img = VK_NULL_HANDLE; alloc = VK_NULL_HANDLE;
  if (vmaCreateImage(device.getAllocator(), &cubeInfo, &allocCreateInfo, &img, &alloc, nullptr) != VK_SUCCESS)
    throw std::runtime_error("Failed to create point shadow depth cubemap");
  pointShadowDepthImage = AllocatedImage(device.getAllocator(), img, alloc);

  VkImageViewCreateInfo cubeViewInfo = viewInfo;
  cubeViewInfo.image    = pointShadowDepthImage.get();
  cubeViewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
  cubeViewInfo.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 6 };
  view = VK_NULL_HANDLE;
  if (vkCreateImageView(device.getLogicalDevice(), &cubeViewInfo, nullptr, &view) != VK_SUCCESS)
    throw std::runtime_error("Failed to create point shadow cubemap view");
  pointShadowCubeView = ImageViewHandle(device.getLogicalDevice(), view);

  pointShadowFaceViews.clear();
  pointShadowFramebuffers.resize(6, VK_NULL_HANDLE);
  for (uint32_t face = 0; face < 6; ++face) {
    VkImageViewCreateInfo faceViewInfo = viewInfo;
    faceViewInfo.image    = pointShadowDepthImage.get();
    faceViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    faceViewInfo.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, face, 1 };
    VkImageView fv = VK_NULL_HANDLE;
    if (vkCreateImageView(device.getLogicalDevice(), &faceViewInfo, nullptr, &fv) != VK_SUCCESS)
      throw std::runtime_error("Failed to create point shadow face view");
    pointShadowFaceViews.emplace_back(device.getLogicalDevice(), fv);

    VkFramebufferCreateInfo pfbInfo = fbInfo;
    pfbInfo.width  = POINT_SHADOW_MAP_SIZE;
    pfbInfo.height = POINT_SHADOW_MAP_SIZE;
    VkImageView patt = pointShadowFaceViews.back().get();
    pfbInfo.pAttachments = &patt;
    if (vkCreateFramebuffer(device.getLogicalDevice(), &pfbInfo, nullptr,
                            &pointShadowFramebuffers[face]) != VK_SUCCESS)
      throw std::runtime_error("Failed to create point shadow framebuffer");
  }
}

void VulkanRenderer::cleanupShadowResources() {
  for (VkFramebuffer fb : pointShadowFramebuffers)
    if (fb != VK_NULL_HANDLE) vkDestroyFramebuffer(device.getLogicalDevice(), fb, nullptr);
  pointShadowFramebuffers.clear();
  pointShadowFaceViews.clear();
  pointShadowCubeView.reset();
  pointShadowDepthImage.reset();

  if (shadowFramebuffer != VK_NULL_HANDLE) {
    vkDestroyFramebuffer(device.getLogicalDevice(), shadowFramebuffer, nullptr);
    shadowFramebuffer = VK_NULL_HANDLE;
  }
  if (shadowSampler != VK_NULL_HANDLE) {
    vkDestroySampler(device.getLogicalDevice(), shadowSampler, nullptr);
    shadowSampler = VK_NULL_HANDLE;
  }
  shadowDepthImageView.reset();
  shadowDepthImage.reset();
}

// ---------------------------------------------------------------------------
// Synchronization (unchanged)
// ---------------------------------------------------------------------------

void VulkanRenderer::createSynchronization() {
  imageAvailable.resize(MAX_FRAMES_DRAWS);
  renderFinished.resize(MAX_FRAMES_DRAWS);
  drawFences.resize(MAX_FRAMES_DRAWS);
  imagesInFlight.resize(swapchain.getImageCount(), VK_NULL_HANDLE);

  VkSemaphoreCreateInfo semInfo = {};
  semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  VkDevice dev = device.getLogicalDevice();
  for (size_t i = 0; i < MAX_FRAMES_DRAWS; i++) {
    if (vkCreateSemaphore(dev, &semInfo, nullptr, &imageAvailable[i]) != VK_SUCCESS ||
        vkCreateSemaphore(dev, &semInfo, nullptr, &renderFinished[i]) != VK_SUCCESS ||
        vkCreateFence(dev, &fenceInfo, nullptr, &drawFences[i]) != VK_SUCCESS)
      throw std::runtime_error("Failed to create synchronization primitives");
  }
}
