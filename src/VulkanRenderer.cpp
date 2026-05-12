#include "VulkanRenderer.h"
#include "Model.h"
#include "Utilities.h"
#include "VulkanDebug.h"
#include <spdlog/spdlog.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <array>
#include <cfloat>
#include <cmath>
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
    litFormat = colorFormat;

    // 3. Render passes
    renderPassManager.createGBufferRenderPass(
        device.getLogicalDevice(), gb0Fmt, gb1Fmt, gb2Fmt, gBufferDepthFormat);
    renderPassManager.createLitRenderPass(device.getLogicalDevice(), litFormat);
    renderPassManager.createRenderPass(device.getLogicalDevice(),
                                       swapchainFormat, colorFormat);
    renderPassManager.createShadowRenderPass(device.getLogicalDevice(),
                                             shadowDepthFormat);
    renderPassManager.createImGuiRenderPass(device.getLogicalDevice(),
                                            swapchainFormat);

    // 4. Swapchain
    swapchain.init(device, renderPassManager.getRenderPass(), window);

    // 5. Shadow resources (directional + point)
    createShadowResources();

    // 6. Descriptors
    descriptorManager.init(device.getLogicalDevice(), device.getAllocator(),
                           swapchain.getImageCount());
    descriptorManager.recreateInputSets(device.getLogicalDevice(), swapchain);
    descriptorManager.updateShadowMapDescriptor(
        device.getLogicalDevice(), csmArrayView.get(),
        pointShadowCubeView.get(), shadowSampler);

    // 7. Texture manager (samplers only at this point)
    textureManager.init(device);

    // 8. Lit-buffer resources (created BEFORE G-buffer set update so that
    //    recreateGBufferSets has the lit views to bind).
    createLitResources();

    // 9. G-buffer images, framebuffers, descriptor sets (binds lit views too)
    createGBuffer();

    // 10. Pipeline (needs all 4 render passes + descriptor layouts)
    pipeline.createPipelines(
        device.getLogicalDevice(), renderPassManager.getGBufferRenderPass(),
        renderPassManager.getLitRenderPass(), renderPassManager.getRenderPass(),
        renderPassManager.getShadowRenderPass(), swapchain.getExtent(),
        descriptorManager);

    // 10. IBL + SSAO resources (requires textureManager and descriptorManager)
    initIBL();

    // 11. Synchronization
    createSynchronization();

    // 12. Performance metrics
    QueueFamilyIndices qi = device.getQueueFamilies();
    metrics.init(device.getLogicalDevice(), device.getPhysicalDevice(),
                 static_cast<uint32_t>(qi.graphicsFamily));

    // 13. Scene UBO defaults
    rebuildProjection();

    sceneUbo.view =
        glm::lookAt(glm::vec3(10.0f, 0.0f, 20.0f), glm::vec3(0.0f, 0.0f, -2.0f),
                    glm::vec3(0.0f, 1.0f, 0.0f));
    sceneUbo.invView = glm::inverse(sceneUbo.view);
    sceneUbo.invProj = glm::inverse(sceneUbo.projection);

    sceneUbo.cameraPosition = glm::vec4(10.0f, 0.0f, 20.0f, 1.0f);

    // Warm midday sun coming through the atrium roof openings
    sceneUbo.directionalLight.direction =
        glm::vec4(glm::normalize(glm::vec3(-0.3f, -1.0f, 0.2f)), 0.0f);
    sceneUbo.directionalLight.colorIntensity =
        glm::vec4(1.0f, 0.93f, 0.78f, 5.5f);

    // Warm amber torchlight on the arcade pillars (arch height ~4.5m, x=±8.5m)
    sceneUbo.pointLights[0].position = glm::vec4(8.5f, 4.5f, 2.0f, 1.0f);
    sceneUbo.pointLights[0].colorIntensity =
        glm::vec4(1.0f, 0.70f, 0.35f, 6.0f);
    sceneUbo.pointLights[1].position = glm::vec4(-8.5f, 4.5f, -2.0f, 1.0f);
    sceneUbo.pointLights[1].colorIntensity =
        glm::vec4(1.0f, 0.68f, 0.32f, 6.0f);

    sceneUbo.spotLights[0].position = glm::vec4(0.0f, 5.0f, 6.0f, 1.0f);
    sceneUbo.spotLights[0].direction =
        glm::vec4(glm::normalize(glm::vec3(0.0f, -0.75f, -1.0f)), 0.0f);
    sceneUbo.spotLights[0].colorIntensity = glm::vec4(1.0f, 0.88f, 0.65f, 1.5f);
    sceneUbo.spotLights[0].cutoffAngles =
        glm::vec4(glm::cos(glm::radians(12.5f)), glm::cos(glm::radians(20.0f)),
                  0.0f, 0.0f);

    sceneUbo.shadowParams = glm::vec4(imguiPointShadowFar, 0.0f, 0.0f, 0.0f);
    sceneUbo.fogParams = glm::vec4(imguiFogDensity, 0.25f, imguiFogClamp, 0.0f);
    sceneUbo.lightCounts =
        glm::ivec4(2, 0, 0, 0); // spot lights off; sun+IBL+torches only

    updateLightSpaceMatrices();
    updatePointShadowMatrices();

    // 14. Default albedo texture
    textureManager.loadTexture("plain.png", device, descriptorManager);

    // 15. ImGui overlay
    initImGui();
    createImGuiFramebuffers();

  } catch (const std::runtime_error &e) {
    spdlog::critical("Renderer initialization failed: {}", e.what());
    return EXIT_FAILURE;
  }
  return 0;
}

int VulkanRenderer::createMeshModel(const std::string &modelFile) {
  return modelManager.loadModel(modelFile, device, textureManager,
                                descriptorManager);
}

SceneNode &VulkanRenderer::getRootNode() { return rootNode; }

void VulkanRenderer::updateCameraView(const glm::mat4 &viewMatrix,
                                      const glm::vec3 &cameraPosition) {
  sceneUbo.view = viewMatrix;
  sceneUbo.cameraPosition = glm::vec4(cameraPosition, 1.0f);
  sceneUbo.invView = glm::inverse(viewMatrix);
  sceneUbo.invProj = glm::inverse(sceneUbo.projection);
}

void VulkanRenderer::notifyResize() { framebufferResized = true; }

// Draw

void VulkanRenderer::draw() {
  VkDevice logicalDevice = device.getLogicalDevice();

  metrics.beginFrame();

  vkWaitForFences(logicalDevice, 1, &drawFences[currentFrame], VK_TRUE,
                  std::numeric_limits<uint64_t>::max());

  // Check resize BEFORE acquiring — a successful acquire signals
  // imageAvailable, and returning early without consuming it would leave it
  // signaled on the next frame.
  if (framebufferResized) {
    framebufferResized = false;
    recreateSwapChain();
    return;
  }

  uint32_t imageIndex;
  VkResult acquireResult = vkAcquireNextImageKHR(
      logicalDevice, swapchain.getSwapchain(),
      std::numeric_limits<uint64_t>::max(), imageAvailable[currentFrame],
      VK_NULL_HANDLE, &imageIndex);

  if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
    // Failed acquire → semaphore NOT signaled per spec → safe to recreate
    recreateSwapChain();
    return;
  } else if (acquireResult != VK_SUCCESS &&
             acquireResult != VK_SUBOPTIMAL_KHR) {
    throw std::runtime_error("Failed to acquire swap chain image");
  }

  if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)
    vkWaitForFences(logicalDevice, 1, &imagesInFlight[imageIndex], VK_TRUE,
                    UINT64_MAX);

  imagesInFlight[imageIndex] = drawFences[currentFrame];
  vkResetFences(logicalDevice, 1, &drawFences[currentFrame]);

  updateLightSpaceMatrices();

  recordCommands(imageIndex);

  descriptorManager.updateUniformBuffer(device.getAllocator(), imageIndex,
                                        &sceneUbo, sizeof(SceneUniformBuffer));

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

  if (vkQueueSubmit(device.getGraphicsQueue(), 1, &submitInfo,
                    drawFences[currentFrame]) != VK_SUCCESS)
    throw std::runtime_error("Failed to submit draw command buffer");

  VkPresentInfoKHR presentInfo = {};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &renderFinished[currentFrame];
  presentInfo.swapchainCount = 1;
  VkSwapchainKHR sc = swapchain.getSwapchain();
  presentInfo.pSwapchains = &sc;
  presentInfo.pImageIndices = &imageIndex;

  VkResult result =
      vkQueuePresentKHR(device.getPresentationQueue(), &presentInfo);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
      framebufferResized) {
    framebufferResized = false;
    recreateSwapChain();
    return;
  } else if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to present swap chain image");
  }
  currentFrame = (currentFrame + 1) % MAX_FRAMES_DRAWS;
  metrics.endFrame(logicalDevice);
}

// Swapchain recreation

void VulkanRenderer::recreateSwapChain() {
  swapchain.recreate(device, renderPassManager.getRenderPass(), window);

  cleanupImGuiFramebuffers();
  createImGuiFramebuffers();

  cleanupGBuffer();
  cleanupLitResources();
  createLitResources();
  createGBuffer();

  imagesInFlight.assign(swapchain.getImageCount(), VK_NULL_HANDLE);

  descriptorManager.recreateInputSets(device.getLogicalDevice(), swapchain);

  rebuildProjection();
}

// G-buffer resource management

void VulkanRenderer::createGBuffer() {
  size_t count = swapchain.getImageCount();
  VkDevice dev = device.getLogicalDevice();
  VkFormat gb0Fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
  VkFormat gb1Fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
  VkFormat gb2Fmt = VK_FORMAT_R8G8B8A8_UNORM;

  gBuffer0Images.resize(count);
  gBuffer0Views.resize(count);
  gBuffer1Images.resize(count);
  gBuffer1Views.resize(count);
  gBuffer2Images.resize(count);
  gBuffer2Views.resize(count);
  gBufferDepthImages.resize(count);
  gBufferDepthViews.resize(count);
  gBufferFramebuffers.resize(count, VK_NULL_HANDLE);

  VmaAllocationCreateInfo aci = {};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  auto makeColorImg = [&](AllocatedImage &img, ImageViewHandle &view,
                          VkFormat fmt) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = {swapchain.getExtent().width, swapchain.getExtent().height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.format = fmt;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkImage rawImg = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    if (vmaCreateImage(device.getAllocator(), &ci, &aci, &rawImg, &alloc,
                       nullptr) != VK_SUCCESS)
      throw std::runtime_error("Failed to create G-buffer color image");
    img = AllocatedImage(device.getAllocator(), rawImg, alloc);

    VkImageViewCreateInfo vci = {};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = img.get();
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView v;
    if (vkCreateImageView(dev, &vci, nullptr, &v) != VK_SUCCESS)
      throw std::runtime_error("Failed to create G-buffer color view");
    view = ImageViewHandle(dev, v);
  };

  auto makeDepthImg = [&](AllocatedImage &img, ImageViewHandle &view) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = {swapchain.getExtent().width, swapchain.getExtent().height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.format = gBufferDepthFormat;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkImage rawImg = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    if (vmaCreateImage(device.getAllocator(), &ci, &aci, &rawImg, &alloc,
                       nullptr) != VK_SUCCESS)
      throw std::runtime_error("Failed to create G-buffer depth image");
    img = AllocatedImage(device.getAllocator(), rawImg, alloc);

    VkImageViewCreateInfo vci = {};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = img.get();
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = gBufferDepthFormat;
    vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
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
        gBuffer0Views[i].get(), gBuffer1Views[i].get(), gBuffer2Views[i].get(),
        gBufferDepthViews[i].get()};
    VkFramebufferCreateInfo fbci = {};
    fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbci.renderPass = renderPassManager.getGBufferRenderPass();
    fbci.attachmentCount = static_cast<uint32_t>(attachments.size());
    fbci.pAttachments = attachments.data();
    fbci.width = swapchain.getExtent().width;
    fbci.height = swapchain.getExtent().height;
    fbci.layers = 1;
    if (vkCreateFramebuffer(dev, &fbci, nullptr, &gBufferFramebuffers[i]) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to create G-buffer framebuffer");
  }

  std::vector<VkImageView> gb0v, gb1v, gb2v, depv, litv;
  gb0v.reserve(count);
  gb1v.reserve(count);
  gb2v.reserve(count);
  depv.reserve(count);
  litv.reserve(count);
  for (size_t i = 0; i < count; i++) {
    gb0v.push_back(gBuffer0Views[i].get());
    gb1v.push_back(gBuffer1Views[i].get());
    gb2v.push_back(gBuffer2Views[i].get());
    depv.push_back(gBufferDepthViews[i].get());
    litv.push_back(litViews[i].get());
  }
  descriptorManager.recreateGBufferSets(device.getLogicalDevice(), gb0v, gb1v,
                                        gb2v, depv, litv,
                                        textureManager.getTextureSampler());
}

void VulkanRenderer::cleanupGBuffer() {
  VkDevice dev = device.getLogicalDevice();
  for (VkFramebuffer fb : gBufferFramebuffers)
    if (fb != VK_NULL_HANDLE)
      vkDestroyFramebuffer(dev, fb, nullptr);
  gBufferFramebuffers.clear();
  gBufferDepthViews.clear();
  gBufferDepthImages.clear();
  gBuffer2Views.clear();
  gBuffer2Images.clear();
  gBuffer1Views.clear();
  gBuffer1Images.clear();
  gBuffer0Views.clear();
  gBuffer0Images.clear();
}

// Lit-buffer resource management (post-PBR HDR, sampled by SSR composite)

void VulkanRenderer::createLitResources() {
  VkDevice dev = device.getLogicalDevice();
  size_t count = swapchain.getImageCount();

  litImages.clear();
  litViews.clear();
  litFramebuffers.assign(count, VK_NULL_HANDLE);
  litImages.reserve(count);
  litViews.reserve(count);

  VmaAllocationCreateInfo aci = {};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  for (size_t i = 0; i < count; ++i) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = {swapchain.getExtent().width, swapchain.getExtent().height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.format = litFormat;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkImage rawImg = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    if (vmaCreateImage(device.getAllocator(), &ci, &aci, &rawImg, &alloc,
                       nullptr) != VK_SUCCESS)
      throw std::runtime_error("Failed to create lit image");
    litImages.emplace_back(device.getAllocator(), rawImg, alloc);

    VkImageViewCreateInfo vci = {};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = litImages.back().get();
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = litFormat;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView v = VK_NULL_HANDLE;
    if (vkCreateImageView(dev, &vci, nullptr, &v) != VK_SUCCESS)
      throw std::runtime_error("Failed to create lit view");
    litViews.emplace_back(dev, v);

    VkImageView attachment = v;
    VkFramebufferCreateInfo fbci = {};
    fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbci.renderPass = renderPassManager.getLitRenderPass();
    fbci.attachmentCount = 1;
    fbci.pAttachments = &attachment;
    fbci.width = swapchain.getExtent().width;
    fbci.height = swapchain.getExtent().height;
    fbci.layers = 1;
    if (vkCreateFramebuffer(dev, &fbci, nullptr, &litFramebuffers[i]) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to create lit framebuffer");
  }

  // One sampler shared across all swapchain images (linear, clamp).
  if (litSampler == VK_NULL_HANDLE) {
    VkSamplerCreateInfo sci = {};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxAnisotropy = 1.0f;
    if (vkCreateSampler(dev, &sci, nullptr, &litSampler) != VK_SUCCESS)
      throw std::runtime_error("Failed to create lit sampler");
  }
}

void VulkanRenderer::cleanupLitResources() {
  VkDevice dev = device.getLogicalDevice();
  for (VkFramebuffer fb : litFramebuffers)
    if (fb != VK_NULL_HANDLE)
      vkDestroyFramebuffer(dev, fb, nullptr);
  litFramebuffers.clear();
  litViews.clear();
  litImages.clear();
  if (litSampler != VK_NULL_HANDLE) {
    vkDestroySampler(dev, litSampler, nullptr);
    litSampler = VK_NULL_HANDLE;
  }
}

// IBL resource management

void VulkanRenderer::initIBL() {
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(device.getPhysicalDevice(), &props);

  // Sampler for irradiance / prefiltered / brdf-lut / skybox (mip-capable)
  VkSamplerCreateInfo iblCi = {};
  iblCi.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  iblCi.magFilter = VK_FILTER_LINEAR;
  iblCi.minFilter = VK_FILTER_LINEAR;
  iblCi.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  iblCi.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  iblCi.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  iblCi.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  iblCi.minLod = 0.0f;
  iblCi.maxLod = static_cast<float>(IBL_PREFILTER_MIPS);
  iblCi.maxAnisotropy = 1.0f;
  if (vkCreateSampler(device.getLogicalDevice(), &iblCi, nullptr,
                      &iblSampler) != VK_SUCCESS)
    throw std::runtime_error("Failed to create IBL sampler");

  // Sampler for 4x4 SSAO noise (tiled nearest)
  VkSamplerCreateInfo noiseCi = {};
  noiseCi.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  noiseCi.magFilter = VK_FILTER_NEAREST;
  noiseCi.minFilter = VK_FILTER_NEAREST;
  noiseCi.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  noiseCi.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  noiseCi.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  noiseCi.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  noiseCi.maxAnisotropy = 1.0f;
  if (vkCreateSampler(device.getLogicalDevice(), &noiseCi, nullptr,
                      &ssaoNoiseSampler) != VK_SUCCESS)
    throw std::runtime_error("Failed to create SSAO noise sampler");

  spdlog::info(
      "IBL: loading skybox and generating irradiance / prefiltered maps...");
  iblSkyboxImageIndex = textureManager.loadSkybox(
      "Resources/HDRIs/kloppenheim_02_1k.hdr", device, descriptorManager);
  irradianceImageIndex =
      textureManager.createIrradianceMap(iblSkyboxImageIndex, device);
  prefilteredEnvImageIndex =
      textureManager.createPrefilteredEnvMap(iblSkyboxImageIndex, device);
  brdfLutImageIndex = textureManager.loadBrdfLut(device);
  ssaoNoiseImageIndex = textureManager.createSsaoNoiseTexture(device);
  spdlog::info("IBL: done.");

  descriptorManager.updateIblDescriptors(
      device.getLogicalDevice(),
      textureManager.getImageView(irradianceImageIndex), iblSampler,
      textureManager.getImageView(prefilteredEnvImageIndex),
      textureManager.getImageView(brdfLutImageIndex),
      textureManager.getImageView(ssaoNoiseImageIndex), ssaoNoiseSampler,
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

// Cleanup

void VulkanRenderer::addInstancedModel(
    int modelId, const std::vector<glm::mat4> &transforms) {
  InstancedDrawable drawable;
  drawable.modelId = modelId;
  drawable.instances.reserve(transforms.size());
  for (const glm::mat4 &t : transforms) {
    InstanceData id{};
    id.model = t;
    id.normal = glm::transpose(glm::inverse(t));
    drawable.instances.push_back(id);
  }

  // Compute a world-space bounding sphere covering every instance.
  // Used in recordCommands to cull/LOD the entire batch.
  MeshModel *mdl = modelManager.getModel(modelId);
  float modelRadius = mdl ? mdl->boundingRadius : 1.0f;
  glm::vec3 modelCenter = mdl ? mdl->boundingCenter : glm::vec3(0.0f);
  glm::vec3 minP(FLT_MAX), maxP(-FLT_MAX);
  for (const InstanceData &id : drawable.instances) {
    glm::vec3 wc = glm::vec3(id.model * glm::vec4(modelCenter, 1.0f));
    minP = glm::min(minP, wc);
    maxP = glm::max(maxP, wc);
  }
  drawable.groupCenter = (minP + maxP) * 0.5f;
  // Largest distance from group center to any instance, plus the model's own
  // radius scaled by the largest instance scale.
  float maxInstScale = 0.0f;
  float maxInstDist = 0.0f;
  for (const InstanceData &id : drawable.instances) {
    glm::vec3 wc = glm::vec3(id.model * glm::vec4(modelCenter, 1.0f));
    maxInstDist = glm::max(maxInstDist, glm::length(wc - drawable.groupCenter));
    float s = glm::max(glm::length(glm::vec3(id.model[0])),
                       glm::max(glm::length(glm::vec3(id.model[1])),
                                glm::length(glm::vec3(id.model[2]))));
    maxInstScale = glm::max(maxInstScale, s);
  }
  drawable.groupRadius = maxInstDist + modelRadius * maxInstScale;

  VkDeviceSize bufferSize = sizeof(InstanceData) * drawable.instances.size();
  createBuffer(device.getAllocator(), bufferSize,
               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
               VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
               &drawable.instanceBuffer);

  void *data;
  vmaMapMemory(device.getAllocator(), drawable.instanceBuffer.getAllocation(),
               &data);
  memcpy(data, drawable.instances.data(), static_cast<size_t>(bufferSize));
  vmaUnmapMemory(device.getAllocator(),
                 drawable.instanceBuffer.getAllocation());

  instancedDrawables.push_back(std::move(drawable));
}

void VulkanRenderer::cleanup() {
  vkDeviceWaitIdle(device.getLogicalDevice());

  metrics.printReport(device.getAllocator());

  instancedDrawables.clear(); // AllocatedBuffer RAII destroys GPU buffers
  cleanupImGuiFramebuffers();
  cleanupImGui();
  modelManager.cleanup();
  textureManager.cleanup(device.getLogicalDevice(), device.getAllocator());
  cleanupGBuffer();
  cleanupLitResources();
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

// Command recording

static void extractFrustumPlanes(const glm::mat4 &vp, glm::vec4 planes[6]) {
  // Gribb-Hartmann: GLM is column-major, so transpose to access rows as columns
  glm::mat4 t = glm::transpose(vp);
  planes[0] = t[3] + t[0]; // left
  planes[1] = t[3] - t[0]; // right
  planes[2] = t[3] + t[1]; // bottom
  planes[3] = t[3] - t[1]; // top
  planes[4] = t[3] + t[2]; // near
  planes[5] = t[3] - t[2]; // far
  for (int i = 0; i < 6; i++)
    planes[i] /= glm::length(glm::vec3(planes[i]));
}

static bool sphereInFrustum(const glm::vec4 planes[6], glm::vec3 center,
                            float radius) {
  for (int i = 0; i < 6; i++)
    if (glm::dot(glm::vec3(planes[i]), center) + planes[i].w < -radius)
      return false;
  return true;
}

void VulkanRenderer::recordCommands(uint32_t currentImage) {
  VkCommandBuffer cmd = swapchain.getCommandBuffer(currentImage);
  vkResetCommandBuffer(cmd, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
    throw std::runtime_error("Failed to begin recording command buffer");

  metrics.resetGpuQueries(cmd);

  vkdbgBeginLabel(cmd, "CSM Shadow Pass", 1.0f, 0.4f, 0.1f);
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::Shadow);
  recordShadowPass(cmd);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::Shadow);
  vkdbgEndLabel(cmd);

  vkdbgBeginLabel(cmd, "Point Shadow Pass", 1.0f, 0.6f, 0.1f);
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::PointShadow);
  recordPointShadowPass(cmd);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::PointShadow);
  vkdbgEndLabel(cmd);

  VkViewport viewport = {};
  viewport.width = static_cast<float>(swapchain.getExtent().width);
  viewport.height = static_cast<float>(swapchain.getExtent().height);
  viewport.maxDepth = 1.0f;
  VkRect2D scissor = {{0, 0}, swapchain.getExtent()};

  // --- G-buffer pass ---
  {
    std::array<VkClearValue, 4> clears{};
    clears[3].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpbi = {};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = renderPassManager.getGBufferRenderPass();
    rpbi.framebuffer = gBufferFramebuffers[currentImage];
    rpbi.renderArea = {{0, 0}, swapchain.getExtent()};
    rpbi.clearValueCount = static_cast<uint32_t>(clears.size());
    rpbi.pClearValues = clears.data();

    vkdbgBeginLabel(cmd, "G-Buffer Pass", 0.1f, 0.4f, 1.0f);
    metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::GBuffer);
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getGraphicsPipeline());

    glm::vec4 frustumPlanes[6];
    extractFrustumPlanes(sceneUbo.projection * sceneUbo.view, frustumPlanes);

    // Bind the VP descriptor set once per frame; only rebind the per-material
    // set when the material id changes between adjacent meshes (Sponza has
    // many runs of meshes sharing the same descriptor set).
    int lastMaterialId = -1;
    VkDescriptorSet vpSet = descriptorManager.getVPSet(currentImage);

    auto renderNode = [&](auto &self, SceneNode *node) -> void {
      if (node->getModelId() >= 0) {
        MeshModel *mdl = modelManager.getModel(node->getModelId());
        if (mdl) {
          glm::mat4 m = node->getGlobalTransform();
          glm::vec3 wCenter =
              glm::vec3(m * glm::vec4(mdl->boundingCenter, 1.0f));
          float maxScale = glm::max(glm::length(glm::vec3(m[0])),
                                    glm::max(glm::length(glm::vec3(m[1])),
                                             glm::length(glm::vec3(m[2]))));
          bool inFrustum = sphereInFrustum(frustumPlanes, wCenter,
                                           mdl->boundingRadius * maxScale);

          if (inFrustum) {
            // Pick LOD from camera distance to bounding sphere center
            float camDist =
                glm::length(wCenter - glm::vec3(sceneUbo.cameraPosition));
            int lod = (camDist < imguiLodNear)  ? 0
                      : (camDist < imguiLodFar) ? 1
                                                : 2;

            ModelPushConstants push{};
            push.model = m;
            push.normal = node->getNormalMatrix();
            vkCmdPushConstants(cmd, pipeline.getGraphicsLayout(),
                               VK_SHADER_STAGE_VERTEX_BIT, 0,
                               sizeof(ModelPushConstants), &push);

            for (size_t k = 0; k < mdl->getMeshCount(); k++) {
              const Mesh *mesh = mdl->getMesh(k);

              // Per-mesh frustum cull (Sponza has ~100 submeshes; many
              // off-screen)
              glm::vec3 meshWCenter =
                  glm::vec3(m * glm::vec4(mesh->boundingCenter, 1.0f));
              if (!sphereInFrustum(frustumPlanes, meshWCenter,
                                   mesh->boundingRadius * maxScale))
                continue;

              int useLod = std::min(lod, mesh->getLodCount() - 1);
              VkBuffer vb[] = {mesh->getVertexBuffer()};
              VkDeviceSize off[] = {0};
              vkCmdBindVertexBuffers(cmd, 0, 1, vb, off);
              vkCmdBindIndexBuffer(cmd, mesh->getIndexBuffer(useLod), 0,
                                   VK_INDEX_TYPE_UINT32);

              int matId = mesh->getMaterial().descriptorSetId;
              if (matId != lastMaterialId) {
                std::array<VkDescriptorSet, 2> sets = {
                    vpSet, descriptorManager.getSamplerSet(matId)};
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        pipeline.getGraphicsLayout(), 0, 2,
                                        sets.data(), 0, nullptr);
                lastMaterialId = matId;
              }
              int idxCount = mesh->getIndexCount(useLod);
              vkCmdDrawIndexed(cmd, idxCount, 1, 0, 0, 0);
              metrics.recordDrawCall(idxCount);
            }
          }
        }
      }
      for (auto &child : node->getChildren())
        self(self, child.get());
    };
    renderNode(renderNode, &rootNode);

    // Instanced drawables: one draw call per mesh, all transforms in instance
    // buffer. Frustum-cull the entire batch by its precomputed group sphere,
    // and pick LOD by distance to the group center.
    if (!instancedDrawables.empty()) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipeline.getInstancedPipeline());
      for (const InstancedDrawable &drawable : instancedDrawables) {
        MeshModel *mdl = modelManager.getModel(drawable.modelId);
        if (!mdl)
          continue;

        if (!sphereInFrustum(frustumPlanes, drawable.groupCenter,
                             drawable.groupRadius))
          continue;

        float groupDist = glm::length(drawable.groupCenter -
                                      glm::vec3(sceneUbo.cameraPosition));
        int instLod = (groupDist < imguiLodNear)  ? 0
                      : (groupDist < imguiLodFar) ? 1
                                                  : 2;

        VkBuffer instanceBuf = drawable.instanceBuffer.get();
        VkDeviceSize instanceOff = 0;

        for (size_t k = 0; k < mdl->getMeshCount(); k++) {
          const Mesh *mesh = mdl->getMesh(k);
          int useLod = std::min(instLod, mesh->getLodCount() - 1);
          VkBuffer vb[] = {mesh->getVertexBuffer()};
          VkDeviceSize off[] = {0};
          vkCmdBindVertexBuffers(cmd, 0, 1, vb, off);
          vkCmdBindVertexBuffers(cmd, 1, 1, &instanceBuf, &instanceOff);
          vkCmdBindIndexBuffer(cmd, mesh->getIndexBuffer(useLod), 0,
                               VK_INDEX_TYPE_UINT32);

          std::array<VkDescriptorSet, 2> sets = {
              descriptorManager.getVPSet(currentImage),
              descriptorManager.getSamplerSet(
                  mesh->getMaterial().descriptorSetId)};
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  pipeline.getInstancedLayout(), 0, 2,
                                  sets.data(), 0, nullptr);
          uint32_t instanceCount =
              static_cast<uint32_t>(drawable.instances.size());
          int idxCount = mesh->getIndexCount(useLod);
          vkCmdDrawIndexed(cmd, idxCount, instanceCount, 0, 0, 0);
          metrics.recordDrawCall(idxCount * instanceCount);
        }
      }
    }

    vkCmdEndRenderPass(cmd);
    metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::GBuffer);
    vkdbgEndLabel(cmd);
  }

  // --- Lit pass (PBR + IBL + SSAO + bloom + FXAA + fog → litBuffer) ---
  {
    VkClearValue clear = {};
    clear.color = {0.0f, 0.0f, 0.0f, 1.0f};

    VkRenderPassBeginInfo rpbi = {};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = renderPassManager.getLitRenderPass();
    rpbi.framebuffer = litFramebuffers[currentImage];
    rpbi.renderArea = {{0, 0}, swapchain.getExtent()};
    rpbi.clearValueCount = 1;
    rpbi.pClearValues = &clear;

    vkdbgBeginLabel(cmd, "Deferred Lighting (PBR+IBL+SSAO+Bloom)", 0.1f, 0.8f,
                    0.3f);
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getLitPipeline());
    std::array<VkDescriptorSet, 2> litSets = {
        descriptorManager.getVPSet(currentImage),
        descriptorManager.getGBufferSet(currentImage)};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getLitLayout(), 0, 2, litSets.data(), 0,
                            nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
    vkdbgEndLabel(cmd);
  }

  // --- Composition pass (SSR composite + ACES tone-mapping) ---
  {
    std::array<VkClearValue, 2> clears{};
    clears[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
    clears[1].color = {0.0f, 0.0f, 0.0f, 1.0f};

    VkRenderPassBeginInfo rpbi = {};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = renderPassManager.getRenderPass();
    rpbi.framebuffer = swapchain.getFramebuffer(currentImage);
    rpbi.renderArea = {{0, 0}, swapchain.getExtent()};
    rpbi.clearValueCount = static_cast<uint32_t>(clears.size());
    rpbi.pClearValues = clears.data();

    vkdbgBeginLabel(cmd, "Composition (SSR + ACES Tonemap)", 0.6f, 0.2f, 0.8f);
    metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::Deferred);
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Subpass 0: SSR composite (samples litBuffer + G-buffer) → colorBuffer
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getDeferredPipeline());
    std::array<VkDescriptorSet, 2> deferredSets = {
        descriptorManager.getVPSet(currentImage),
        descriptorManager.getGBufferSet(currentImage)};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getDeferredLayout(), 0, 2,
                            deferredSets.data(), 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    // Subpass 1: ACES + gamma → swapchain
    vkCmdNextSubpass(cmd, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getSecondPipeline());
    VkDescriptorSet inputSet = descriptorManager.getInputSet(currentImage);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getSecondLayout(), 0, 1, &inputSet, 0,
                            nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRenderPass(cmd);
    metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::Deferred);
    vkdbgEndLabel(cmd);
  }

  vkdbgBeginLabel(cmd, "ImGui Pass", 0.9f, 0.8f, 0.1f);
  recordImGuiCommands(cmd, currentImage);
  vkdbgEndLabel(cmd);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
    throw std::runtime_error("Failed to stop recording command buffer");
}

// Shadow passes (unchanged)

void VulkanRenderer::recordShadowPass(VkCommandBuffer cmdBuffer) {
  VkViewport vp = {0,
                   0,
                   static_cast<float>(SHADOW_MAP_SIZE),
                   static_cast<float>(SHADOW_MAP_SIZE),
                   0,
                   1};
  VkRect2D sc = {{0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}};

  // Per-cascade LOD: near cascades full detail, far cascades simplified.
  // Shadow silhouettes are forgiving so this is a free win.
  auto cascadeLod = [](int cascade) {
    if (cascade == 0)
      return 0;
    if (cascade == 1)
      return 1;
    return 2; // cascades 2 and 3
  };

  auto renderNodeShadow = [&](auto &self, SceneNode *node, const glm::mat4 &lsm,
                              int lodIndex) -> void {
    if (node->getModelId() >= 0) {
      MeshModel *mdl = modelManager.getModel(node->getModelId());
      if (mdl) {
        ShadowPushConstants push{};
        push.model = node->getGlobalTransform();
        push.lightSpaceMatrix = lsm;
        vkCmdPushConstants(cmdBuffer, pipeline.getShadowLayout(),
                           VK_SHADER_STAGE_VERTEX_BIT, 0,
                           sizeof(ShadowPushConstants), &push);
        for (size_t k = 0; k < mdl->getMeshCount(); k++) {
          const Mesh *mesh = mdl->getMesh(k);
          int useLod = std::min(lodIndex, mesh->getLodCount() - 1);
          VkBuffer vb[] = {mesh->getVertexBuffer()};
          VkDeviceSize off[] = {0};
          vkCmdBindVertexBuffers(cmdBuffer, 0, 1, vb, off);
          vkCmdBindIndexBuffer(cmdBuffer, mesh->getIndexBuffer(useLod), 0,
                               VK_INDEX_TYPE_UINT32);
          vkCmdDrawIndexed(cmdBuffer, mesh->getIndexCount(useLod), 1, 0, 0, 0);
        }
      }
    }
    for (auto &child : node->getChildren())
      self(self, child.get(), lsm, lodIndex);
  };

  for (int cascade = 0; cascade < NUM_CSM_CASCADES; cascade++) {
    VkClearValue clearValue = {};
    clearValue.depthStencil.depth = 1.0f;

    VkRenderPassBeginInfo rpbi = {};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = renderPassManager.getShadowRenderPass();
    rpbi.framebuffer = csmFramebuffers[cascade];
    rpbi.renderArea.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE};
    rpbi.clearValueCount = 1;
    rpbi.pClearValues = &clearValue;

    char cascadeLabel[32];
    snprintf(cascadeLabel, sizeof(cascadeLabel), "Cascade %d", cascade);
    vkdbgBeginLabel(cmdBuffer, cascadeLabel, 1.0f, 0.55f + cascade * 0.1f, 0.1f);
    vkCmdBeginRenderPass(cmdBuffer, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdSetViewport(cmdBuffer, 0, 1, &vp);
    vkCmdSetScissor(cmdBuffer, 0, 1, &sc);
    // Front-face culling avoids self-shadow acne and (more importantly here)
    // captures occluders whose only face points away from the sun — the
    // single-sided thin geometry that Sponza uses for ceilings and arch caps.
    vkCmdSetCullMode(cmdBuffer,
                     imguiShadowFrontFaceCull ? VK_CULL_MODE_FRONT_BIT
                                              : VK_CULL_MODE_BACK_BIT);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getShadowPipeline());
    renderNodeShadow(renderNodeShadow, &rootNode,
                     sceneUbo.lightSpaceMatrices[cascade], cascadeLod(cascade));
    vkCmdEndRenderPass(cmdBuffer);
    vkdbgEndLabel(cmdBuffer);
  }
}

void VulkanRenderer::recordPointShadowPass(VkCommandBuffer cmdBuffer) {
  VkClearValue clearValue = {};
  clearValue.depthStencil.depth = 1.0f;

  for (uint32_t face = 0; face < 6; ++face) {
    char faceLabel[32];
    snprintf(faceLabel, sizeof(faceLabel), "Point Shadow Face %u", face);
    vkdbgBeginLabel(cmdBuffer, faceLabel, 1.0f, 0.65f, 0.2f);
    VkRenderPassBeginInfo rpbi = {};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = renderPassManager.getShadowRenderPass();
    rpbi.framebuffer = pointShadowFramebuffers[face];
    rpbi.renderArea.extent = {POINT_SHADOW_MAP_SIZE, POINT_SHADOW_MAP_SIZE};
    rpbi.clearValueCount = 1;
    rpbi.pClearValues = &clearValue;

    vkCmdBeginRenderPass(cmdBuffer, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp = {0,
                     0,
                     static_cast<float>(POINT_SHADOW_MAP_SIZE),
                     static_cast<float>(POINT_SHADOW_MAP_SIZE),
                     0,
                     1};
    VkRect2D sc = {{0, 0}, {POINT_SHADOW_MAP_SIZE, POINT_SHADOW_MAP_SIZE}};
    vkCmdSetViewport(cmdBuffer, 0, 1, &vp);
    vkCmdSetScissor(cmdBuffer, 0, 1, &sc);
    vkCmdSetCullMode(cmdBuffer,
                     imguiShadowFrontFaceCull ? VK_CULL_MODE_FRONT_BIT
                                              : VK_CULL_MODE_BACK_BIT);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getShadowPipeline());

    auto renderNodeShadow = [&](auto &self, SceneNode *node) -> void {
      if (node->getModelId() >= 0) {
        MeshModel *mdl = modelManager.getModel(node->getModelId());
        if (mdl) {
          ShadowPushConstants push{};
          push.model = node->getGlobalTransform();
          push.lightSpaceMatrix = pointShadowMatrices[face];
          vkCmdPushConstants(cmdBuffer, pipeline.getShadowLayout(),
                             VK_SHADER_STAGE_VERTEX_BIT, 0,
                             sizeof(ShadowPushConstants), &push);
          for (size_t k = 0; k < mdl->getMeshCount(); k++) {
            const Mesh *mesh = mdl->getMesh(k);
            int useLod = std::min(2, mesh->getLodCount() - 1);
            VkBuffer vb[] = {mesh->getVertexBuffer()};
            VkDeviceSize off[] = {0};
            vkCmdBindVertexBuffers(cmdBuffer, 0, 1, vb, off);
            vkCmdBindIndexBuffer(cmdBuffer, mesh->getIndexBuffer(useLod), 0,
                                 VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmdBuffer, mesh->getIndexCount(useLod), 1, 0, 0,
                             0);
          }
        }
      }
      for (auto &child : node->getChildren())
        self(self, child.get());
    };
    renderNodeShadow(renderNodeShadow, &rootNode);
    vkCmdEndRenderPass(cmdBuffer);
    vkdbgEndLabel(cmdBuffer);
  }
}

// Light helpers (unchanged)

void VulkanRenderer::updateLightSpaceMatrices() {
  const float nearPlane = 0.1f;
  // CSM covers up to imguiCsmFar (clamped to draw distance). Fragments past
  // this fall through every cascade and the shader's fallback treats them as
  // lit, so on Crytek-Sponza-scale models this needs to track the actual hall
  // length, not a tiny default.
  const float csmFar = std::min(imguiCsmFar, imguiDrawDistance);
  const float farPlane = csmFar;
  const float lambda = 0.75f; // blend between log and uniform splits

  // Practical split scheme (Engel 2006)
  float splits[NUM_CSM_CASCADES];
  for (int i = 0; i < NUM_CSM_CASCADES; i++) {
    float p = (i + 1) / float(NUM_CSM_CASCADES);
    float logSplit = nearPlane * std::pow(farPlane / nearPlane, p);
    float uniSplit = nearPlane + (farPlane - nearPlane) * p;
    splits[i] = lambda * logSplit + (1.0f - lambda) * uniSplit;
  }
  sceneUbo.cascadeSplits =
      glm::vec4(splits[0], splits[1], splits[2], splits[3]);

  glm::vec3 lightDir =
      glm::normalize(glm::vec3(sceneUbo.directionalLight.direction));
  glm::mat4 lightView = glm::lookAt(-lightDir * 100.0f, glm::vec3(0.0f),
                                    glm::vec3(0.0f, 1.0f, 0.0f));
  glm::mat4 invCam = glm::inverse(sceneUbo.projection * sceneUbo.view);

  // Helper: convert view-space distance to NDC z
  auto viewDepthToNdcZ = [&](float d) -> float {
    glm::vec4 clip = sceneUbo.projection * glm::vec4(0.0f, 0.0f, -d, 1.0f);
    return clip.z / clip.w;
  };

  float prevSplit = nearPlane;
  for (int i = 0; i < NUM_CSM_CASCADES; i++) {
    float nearNdc = viewDepthToNdcZ(prevSplit);
    float farNdc = viewDepthToNdcZ(splits[i]);

    // 8 NDC corners of this cascade's frustum slice
    glm::vec3 corners[8];
    int k = 0;
    for (float nx : {-1.0f, 1.0f})
      for (float ny : {-1.0f, 1.0f})
        for (float nz : {nearNdc, farNdc}) {
          glm::vec4 w = invCam * glm::vec4(nx, ny, nz, 1.0f);
          corners[k++] = glm::vec3(w) / w.w;
        }

    // Bounding sphere centred on the camera world position. The camera position
    // is rotation-invariant (it doesn't move when you look left/right), so the
    // shadow map XY coverage stays constant as the camera rotates. The old
    // circumcenter (average of corners) shifts with yaw at wide FOV, causing
    // objects to fall off the edge of the shadow map on rotation.
    glm::vec3 frustumCenter = glm::vec3(glm::inverse(sceneUbo.view)[3]);

    float sphereRadius = 0.0f;
    for (auto &c : corners)
      sphereRadius = glm::max(sphereRadius, glm::length(c - frustumCenter));

    // Convert center to light space and snap to texel grid so the shadow map
    // does not shimmer as the camera translates.
    glm::vec3 lsCenter = glm::vec3(lightView * glm::vec4(frustumCenter, 1.0f));
    float texelSz = 2.0f * sphereRadius / float(SHADOW_MAP_SIZE);
    if (texelSz > 0.0f) {
      lsCenter.x = std::floor(lsCenter.x / texelSz) * texelSz;
      lsCenter.y = std::floor(lsCenter.y / texelSz) * texelSz;
    }

    glm::vec3 mn(lsCenter.x - sphereRadius, lsCenter.y - sphereRadius,
                 lsCenter.z - sphereRadius - csmFar);
    glm::vec3 mx(lsCenter.x + sphereRadius, lsCenter.y + sphereRadius,
                 lsCenter.z + sphereRadius + csmFar);

    glm::mat4 lightProj = glm::ortho(mn.x, mx.x, mn.y, mx.y, mn.z, mx.z);
    lightProj[1][1] *= -1.0f;
    sceneUbo.lightSpaceMatrices[i] = lightProj * lightView;

    prevSplit = splits[i];
  }
}

void VulkanRenderer::updatePointShadowMatrices() {
  pointShadowMatrices.resize(6);
  glm::vec3 lp = glm::vec3(sceneUbo.pointLights[0].position);
  float farPlane = sceneUbo.shadowParams.x;
  glm::mat4 proj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, farPlane);
  proj[1][1] *= -1.0f;

  pointShadowMatrices[0] =
      proj * glm::lookAt(lp, lp + glm::vec3(1, 0, 0), glm::vec3(0, -1, 0));
  pointShadowMatrices[1] =
      proj * glm::lookAt(lp, lp + glm::vec3(-1, 0, 0), glm::vec3(0, -1, 0));
  pointShadowMatrices[2] =
      proj * glm::lookAt(lp, lp + glm::vec3(0, 1, 0), glm::vec3(0, 0, 1));
  pointShadowMatrices[3] =
      proj * glm::lookAt(lp, lp + glm::vec3(0, -1, 0), glm::vec3(0, 0, -1));
  pointShadowMatrices[4] =
      proj * glm::lookAt(lp, lp + glm::vec3(0, 0, 1), glm::vec3(0, -1, 0));
  pointShadowMatrices[5] =
      proj * glm::lookAt(lp, lp + glm::vec3(0, 0, -1), glm::vec3(0, -1, 0));

  for (size_t i = 0; i < 6; ++i)
    sceneUbo.pointShadowMatrices[i] = pointShadowMatrices[i];
}

// Shadow resource creation (unchanged)

void VulkanRenderer::createShadowResources() {
  VkDevice dev = device.getLogicalDevice();

  // --- CSM: 4-layer depth array ---
  VkImageCreateInfo imageCreateInfo = {};
  imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
  imageCreateInfo.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1};
  imageCreateInfo.mipLevels = 1;
  imageCreateInfo.arrayLayers = NUM_CSM_CASCADES;
  imageCreateInfo.format = shadowDepthFormat;
  imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageCreateInfo.usage =
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo allocCreateInfo = {};
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  VkImage img = VK_NULL_HANDLE;
  VmaAllocation alloc = VK_NULL_HANDLE;
  if (vmaCreateImage(device.getAllocator(), &imageCreateInfo, &allocCreateInfo,
                     &img, &alloc, nullptr) != VK_SUCCESS)
    throw std::runtime_error("Failed to create CSM depth array image");
  csmDepthImage = AllocatedImage(device.getAllocator(), img, alloc);

  // Array view (all 4 layers) — used by deferred shader as sampler2DArray
  VkImageViewCreateInfo arrayViewInfo = {};
  arrayViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  arrayViewInfo.image = csmDepthImage.get();
  arrayViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  arrayViewInfo.format = shadowDepthFormat;
  arrayViewInfo.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0,
                                    NUM_CSM_CASCADES};
  VkImageView view = VK_NULL_HANDLE;
  if (vkCreateImageView(dev, &arrayViewInfo, nullptr, &view) != VK_SUCCESS)
    throw std::runtime_error("Failed to create CSM array view");
  csmArrayView = ImageViewHandle(dev, view);

  // Per-cascade layer views + framebuffers (one per cascade)
  csmLayerViews.clear();
  csmFramebuffers.resize(NUM_CSM_CASCADES, VK_NULL_HANDLE);
  VkFramebufferCreateInfo fbInfo = {};
  fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  fbInfo.renderPass = renderPassManager.getShadowRenderPass();
  fbInfo.attachmentCount = 1;
  fbInfo.width = SHADOW_MAP_SIZE;
  fbInfo.height = SHADOW_MAP_SIZE;
  fbInfo.layers = 1;
  for (int i = 0; i < NUM_CSM_CASCADES; i++) {
    VkImageViewCreateInfo layerViewInfo = arrayViewInfo;
    layerViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    layerViewInfo.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1,
                                      static_cast<uint32_t>(i), 1};
    VkImageView lv = VK_NULL_HANDLE;
    if (vkCreateImageView(dev, &layerViewInfo, nullptr, &lv) != VK_SUCCESS)
      throw std::runtime_error("Failed to create CSM layer view");
    csmLayerViews.emplace_back(dev, lv);
    fbInfo.pAttachments = &lv;
    if (vkCreateFramebuffer(dev, &fbInfo, nullptr, &csmFramebuffers[i]) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to create CSM framebuffer");
  }

  VkSamplerCreateInfo samplerInfo = {};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
  samplerInfo.maxLod = 1.0f;
  samplerInfo.maxAnisotropy = 1.0f;
  if (vkCreateSampler(dev, &samplerInfo, nullptr, &shadowSampler) != VK_SUCCESS)
    throw std::runtime_error("Failed to create shadow sampler");

  // Point shadow cubemap
  VkImageCreateInfo cubeInfo = imageCreateInfo;
  cubeInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  cubeInfo.extent = {POINT_SHADOW_MAP_SIZE, POINT_SHADOW_MAP_SIZE, 1};
  cubeInfo.arrayLayers = 6;

  img = VK_NULL_HANDLE;
  alloc = VK_NULL_HANDLE;
  if (vmaCreateImage(device.getAllocator(), &cubeInfo, &allocCreateInfo, &img,
                     &alloc, nullptr) != VK_SUCCESS)
    throw std::runtime_error("Failed to create point shadow depth cubemap");
  pointShadowDepthImage = AllocatedImage(device.getAllocator(), img, alloc);

  VkImageViewCreateInfo cubeViewInfo = {};
  cubeViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  cubeViewInfo.image = pointShadowDepthImage.get();
  cubeViewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
  cubeViewInfo.format = shadowDepthFormat;
  cubeViewInfo.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 6};
  view = VK_NULL_HANDLE;
  if (vkCreateImageView(device.getLogicalDevice(), &cubeViewInfo, nullptr,
                        &view) != VK_SUCCESS)
    throw std::runtime_error("Failed to create point shadow cubemap view");
  pointShadowCubeView = ImageViewHandle(device.getLogicalDevice(), view);

  pointShadowFaceViews.clear();
  pointShadowFramebuffers.resize(6, VK_NULL_HANDLE);
  for (uint32_t face = 0; face < 6; ++face) {
    VkImageViewCreateInfo faceViewInfo = {};
    faceViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    faceViewInfo.image = pointShadowDepthImage.get();
    faceViewInfo.format = shadowDepthFormat;
    faceViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    faceViewInfo.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, face, 1};
    VkImageView fv = VK_NULL_HANDLE;
    if (vkCreateImageView(device.getLogicalDevice(), &faceViewInfo, nullptr,
                          &fv) != VK_SUCCESS)
      throw std::runtime_error("Failed to create point shadow face view");
    pointShadowFaceViews.emplace_back(device.getLogicalDevice(), fv);

    VkFramebufferCreateInfo pfbInfo = fbInfo;
    pfbInfo.width = POINT_SHADOW_MAP_SIZE;
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
    if (fb != VK_NULL_HANDLE)
      vkDestroyFramebuffer(device.getLogicalDevice(), fb, nullptr);
  pointShadowFramebuffers.clear();
  pointShadowFaceViews.clear();
  pointShadowCubeView.reset();
  pointShadowDepthImage.reset();

  for (VkFramebuffer fb : csmFramebuffers)
    if (fb != VK_NULL_HANDLE)
      vkDestroyFramebuffer(device.getLogicalDevice(), fb, nullptr);
  csmFramebuffers.clear();
  csmLayerViews.clear();
  csmArrayView.reset();
  csmDepthImage.reset();

  if (shadowSampler != VK_NULL_HANDLE) {
    vkDestroySampler(device.getLogicalDevice(), shadowSampler, nullptr);
    shadowSampler = VK_NULL_HANDLE;
  }
}

// ImGui integration

void VulkanRenderer::initImGui() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  ImGui::StyleColorsDark();

  ImGui_ImplGlfw_InitForVulkan(window, true);

  QueueFamilyIndices qi = device.getQueueFamilies();
  ImGui_ImplVulkan_InitInfo info = {};
  info.ApiVersion = VK_API_VERSION_1_3;
  info.Instance = device.getInstance();
  info.PhysicalDevice = device.getPhysicalDevice();
  info.Device = device.getLogicalDevice();
  info.QueueFamily = static_cast<uint32_t>(qi.graphicsFamily);
  info.Queue = device.getGraphicsQueue();
  info.DescriptorPoolSize = 16; // let ImGui manage its own pool
  info.MinImageCount = 2;
  info.ImageCount = static_cast<uint32_t>(swapchain.getImageCount());
  info.PipelineInfoMain.RenderPass = renderPassManager.getImGuiRenderPass();
  info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
  ImGui_ImplVulkan_Init(&info);
}

void VulkanRenderer::cleanupImGui() {
  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}

void VulkanRenderer::createImGuiFramebuffers() {
  const auto &images = swapchain.getImages();
  imguiFramebuffers.resize(images.size(), VK_NULL_HANDLE);
  for (size_t i = 0; i < images.size(); i++) {
    VkImageView view = images[i].imageView.get();
    VkFramebufferCreateInfo fbci = {};
    fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbci.renderPass = renderPassManager.getImGuiRenderPass();
    fbci.attachmentCount = 1;
    fbci.pAttachments = &view;
    fbci.width = swapchain.getExtent().width;
    fbci.height = swapchain.getExtent().height;
    fbci.layers = 1;
    if (vkCreateFramebuffer(device.getLogicalDevice(), &fbci, nullptr,
                            &imguiFramebuffers[i]) != VK_SUCCESS)
      throw std::runtime_error("Failed to create ImGui framebuffer");
  }
}

void VulkanRenderer::cleanupImGuiFramebuffers() {
  for (VkFramebuffer fb : imguiFramebuffers)
    if (fb != VK_NULL_HANDLE)
      vkDestroyFramebuffer(device.getLogicalDevice(), fb, nullptr);
  imguiFramebuffers.clear();
}

void VulkanRenderer::setImGuiCameraInfo(glm::vec3 pos, float speed) {
  imguiCameraPos = pos;
  imguiCameraSpeed = speed;
}

void VulkanRenderer::rebuildProjection() {
  float aspect = static_cast<float>(swapchain.getExtent().width) /
                 static_cast<float>(swapchain.getExtent().height);
  // Near=1.0 (not 0.1) because the scene is Crytek-scale (~3km long); a 0.1m
  // near plane burns most of the D32F depth range on the first metre and
  // leaves the rest of the hall z-fighting. 1m is below any meaningful
  // first-person clip distance at this scale.
  sceneUbo.projection = glm::perspective(glm::radians(imguiCameraFov), aspect,
                                         1.0f, imguiDrawDistance);
  sceneUbo.projection[1][1] *= -1;
  sceneUbo.invProj = glm::inverse(sceneUbo.projection);
}

void VulkanRenderer::setFov(float fovDegrees) {
  imguiCameraFov = fovDegrees;
  rebuildProjection();
}

void VulkanRenderer::setDrawDistance(float dist) {
  imguiDrawDistance = dist;
  rebuildProjection();
}

void VulkanRenderer::buildImGuiUI() {
  // Performance panel — pinned top-left
  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(340, 230), ImGuiCond_Always);
  ImGui::Begin("Performance", nullptr,
               ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoCollapse);
  ImGui::Text("FPS: %.1f  |  Avg: %.2f ms", metrics.getAverageFps(),
              metrics.getAverageFrameTimeMs());
  frameTimeGraphData[frameTimeGraphOffset] =
      static_cast<float>(metrics.getAverageFrameTimeMs());
  frameTimeGraphOffset = (frameTimeGraphOffset + 1) % 128;
  ImGui::PlotLines("##ft", frameTimeGraphData, 128, frameTimeGraphOffset,
                   "Frame time (ms)", 0.0f, 50.0f, ImVec2(322, 60));
  ImGui::Text("Shadow:   %.2f ms",
              metrics.getPassGpuTimeMs(PerformanceMetrics::GpuPass::Shadow));
  ImGui::Text(
      "PtShadow: %.2f ms",
      metrics.getPassGpuTimeMs(PerformanceMetrics::GpuPass::PointShadow));
  ImGui::Text("GBuffer:  %.2f ms",
              metrics.getPassGpuTimeMs(PerformanceMetrics::GpuPass::GBuffer));
  ImGui::Text("Deferred: %.2f ms",
              metrics.getPassGpuTimeMs(PerformanceMetrics::GpuPass::Deferred));
  ImGui::Text("Draws: %u  |  Tris: %uk", metrics.getLastDrawCallCount(),
              metrics.getLastTriangleCount() / 1000);
  auto vram = PerformanceMetrics::queryVram(device.getAllocator());
  ImGui::Text("VRAM: %llu MiB / %llu MiB",
              static_cast<unsigned long long>(vram.usedBytes >> 20),
              static_cast<unsigned long long>(vram.budgetBytes >> 20));
  ImGui::End();

  // Camera panel
  ImGui::SetNextWindowPos(ImVec2(10, 250), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(340, 120), ImGuiCond_Always);
  ImGui::Begin("Camera", nullptr,
               ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoCollapse);
  ImGui::Text("Pos: (%.1f, %.1f, %.1f)", imguiCameraPos.x, imguiCameraPos.y,
              imguiCameraPos.z);
  ImGui::SliderFloat("Speed", &imguiCameraSpeed, 0.5f, 50.0f);
  if (ImGui::SliderFloat("FOV", &imguiCameraFov, 30.0f, 120.0f))
    setFov(imguiCameraFov);
  if (ImGui::SliderFloat("Draw Dist", &imguiDrawDistance, 100.0f, 20000.0f))
    setDrawDistance(imguiDrawDistance);
  ImGui::End();

  // Lighting panel
  ImGui::SetNextWindowPos(ImVec2(10, 380), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(340, 230), ImGuiCond_Always);
  ImGui::Begin("Lighting", nullptr,
               ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoCollapse);
  if (ImGui::CollapsingHeader("Directional Light",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    glm::vec3 dir = glm::vec3(sceneUbo.directionalLight.direction);
    if (ImGui::DragFloat3("Dir##sun", &dir.x, 0.01f, -1.0f, 1.0f)) {
      sceneUbo.directionalLight.direction =
          glm::vec4(glm::normalize(dir), 0.0f);
      updateLightSpaceMatrices();
    }
    glm::vec3 col = glm::vec3(sceneUbo.directionalLight.colorIntensity);
    if (ImGui::ColorEdit3("Color##sun", &col.x))
      sceneUbo.directionalLight.colorIntensity =
          glm::vec4(col, sceneUbo.directionalLight.colorIntensity.a);
    float intens = sceneUbo.directionalLight.colorIntensity.a;
    if (ImGui::SliderFloat("Intensity##sun", &intens, 0.0f, 5.0f))
      sceneUbo.directionalLight.colorIntensity.a = intens;
  }
  int pointCount = sceneUbo.lightCounts.x;
  for (int i = 0; i < pointCount; i++) {
    char label[32];
    snprintf(label, sizeof(label), "Point Light %d", i);
    if (ImGui::CollapsingHeader(label)) {
      glm::vec3 pos = glm::vec3(sceneUbo.pointLights[i].position);
      char id[24];
      snprintf(id, sizeof(id), "Pos##pt%d", i);
      if (ImGui::DragFloat3(id, &pos.x, 0.1f))
        sceneUbo.pointLights[i].position = glm::vec4(pos, 1.0f);
      glm::vec3 col = glm::vec3(sceneUbo.pointLights[i].colorIntensity);
      snprintf(id, sizeof(id), "Color##pt%d", i);
      if (ImGui::ColorEdit3(id, &col.x))
        sceneUbo.pointLights[i].colorIntensity =
            glm::vec4(col, sceneUbo.pointLights[i].colorIntensity.a);
      float intens = sceneUbo.pointLights[i].colorIntensity.a;
      snprintf(id, sizeof(id), "Intensity##pt%d", i);
      if (ImGui::SliderFloat(id, &intens, 0.0f, 10.0f))
        sceneUbo.pointLights[i].colorIntensity.a = intens;
    }
  }
  if (ImGui::CollapsingHeader("Post & Perf")) {
    if (ImGui::SliderFloat("Fog density", &imguiFogDensity, 0.0f, 0.02f,
                           "%.4f"))
      sceneUbo.fogParams.x = imguiFogDensity;
    if (ImGui::SliderFloat("Fog clamp", &imguiFogClamp, 0.0f, 1.0f))
      sceneUbo.fogParams.z = imguiFogClamp;
    ImGui::SliderFloat("LOD near", &imguiLodNear, 1.0f, 60.0f);
    ImGui::SliderFloat("LOD far", &imguiLodFar, imguiLodNear + 1.0f, 200.0f);
    if (ImGui::SliderFloat("Pt shadow far", &imguiPointShadowFar, 5.0f,
                           100.0f)) {
      sceneUbo.shadowParams.x = imguiPointShadowFar;
      updatePointShadowMatrices();
    }
  }
  ImGui::End();

  // Debug views panel
  ImGui::SetNextWindowPos(ImVec2(10, 620), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(340, 70), ImGuiCond_Always);
  ImGui::Begin("Debug Views", nullptr,
               ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoCollapse);
  const char *debugModes[] = {"None",     "Albedo",    "Normals",
                              "Metallic", "Roughness", "Depth"};
  if (ImGui::Combo("G-Buffer", &imguiDebugMode, debugModes, 6))
    sceneUbo.debugMode = imguiDebugMode;
  ImGui::End();

  // Visual-quality fix toggles (Phase 2 A/B controls)
  ImGui::SetNextWindowPos(ImVec2(10, 700), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(340, 250), ImGuiCond_Always);
  ImGui::Begin("Visual Fixes", nullptr,
               ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoCollapse);
  if (ImGui::Checkbox("P1: sRGB albedo decode", &imguiSrgbAlbedoDecode))
    sceneUbo.qualityToggles.x = imguiSrgbAlbedoDecode ? 1.0f : 0.0f;
  if (ImGui::Checkbox("P2: Specular AA", &imguiSpecAAEnable))
    sceneUbo.qualityToggles.y = imguiSpecAAEnable ? 1.0f : 0.0f;
  if (ImGui::SliderFloat("AA variance", &imguiSpecAAVariance, 0.0f, 1.0f,
                         "%.3f"))
    sceneUbo.qualityToggles.z = imguiSpecAAVariance;
  if (ImGui::SliderFloat("AA threshold", &imguiSpecAAThreshold, 0.0f, 0.5f,
                         "%.3f"))
    sceneUbo.qualityToggles.w = imguiSpecAAThreshold;
  if (ImGui::Checkbox("P7: Mipmaps + aniso", &imguiMipmapsEnable))
    sceneUbo.qualityToggles2.x = imguiMipmapsEnable ? 1.0f : 0.0f;
  ImGui::Checkbox("P5: Shadow front-face cull", &imguiShadowFrontFaceCull);
  ImGui::SliderFloat("CSM far", &imguiCsmFar, 100.0f, 5000.0f, "%.0f");
  ImGui::End();
}

void VulkanRenderer::recordImGuiCommands(VkCommandBuffer cmd,
                                         uint32_t imageIndex) {
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  buildImGuiUI();

  ImGui::Render();

  VkClearValue clear = {};
  VkRenderPassBeginInfo rpbi = {};
  rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rpbi.renderPass = renderPassManager.getImGuiRenderPass();
  rpbi.framebuffer = imguiFramebuffers[imageIndex];
  rpbi.renderArea = {{0, 0}, swapchain.getExtent()};
  rpbi.clearValueCount = 1;
  rpbi.pClearValues = &clear;
  vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
  vkCmdEndRenderPass(cmd);
}

// Synchronization (unchanged)

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
    if (vkCreateSemaphore(dev, &semInfo, nullptr, &imageAvailable[i]) !=
            VK_SUCCESS ||
        vkCreateSemaphore(dev, &semInfo, nullptr, &renderFinished[i]) !=
            VK_SUCCESS ||
        vkCreateFence(dev, &fenceInfo, nullptr, &drawFences[i]) != VK_SUCCESS)
      throw std::runtime_error("Failed to create synchronization primitives");
  }
}
