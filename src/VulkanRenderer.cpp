#include "VulkanRenderer.h"
#include "Model.h"
#include "Utilities.h"
#include "VulkanDebug.h"
#include <spdlog/spdlog.h>

#include <chrono>
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
                                       swapchainFormat, colorFormat,
                                       litFormat);
    renderPassManager.createShadowRenderPass(device.getLogicalDevice(),
                                             shadowDepthFormat);
    renderPassManager.createImGuiRenderPass(device.getLogicalDevice(),
                                            swapchainFormat);

    // 4. Swapchain (images + colorBuffer + command buffers — composite
    //     framebuffers are owned by this class and created below in step 8c)
    swapchain.init(device, window);

    // 5. Shadow resources (directional + point)
    createShadowResources();

    // 6. Descriptors
    descriptorManager.init(device.getLogicalDevice(), device.getAllocator(),
                           swapchain.getImageCount());
    descriptorManager.recreateInputSets(device.getLogicalDevice(), swapchain);
    descriptorManager.updateShadowMapDescriptor(
        device.getLogicalDevice(), csmArrayView.get(),
        pointShadowCubeView.get(), csmShadowSampler, shadowSampler);

    // 7. Texture manager (samplers only at this point)
    textureManager.init(device);

    // 8. Lit-buffer resources (created BEFORE G-buffer set update so that
    //    recreateGBufferSets has the lit views to bind).
    createLitResources();

    // 8b. TAA history images. Format matches litFormat so they live in the
    //     same HDR-precision domain. Currently allocated but not yet wired
    //     into the render pass (Phase 6 work-in-progress — plan file
    //     "Session 2" section has the remaining wire-up steps).
    createTaaResources();

    // 8c. Composite framebuffers (3 attachments: swap + colorBuffer + history)
    //     Must be created after TAA history images exist and after the
    //     composition render pass is created.
    createCompositeFramebuffers();

    // 9. G-buffer images, framebuffers, descriptor sets (binds lit views too)
    createGBuffer();

    // 9b. TAA descriptor sets (history-prev + depth + colorBuffer-current).
    //     Needs taaHistory views (from createTaaResources), gBufferDepth
    //     views (from createGBuffer), and colorBuffer views (from swapchain
    //     init). Order: must follow all three.
    {
      std::vector<VkImageView> histViews = {taaHistoryViews[0].get(),
                                            taaHistoryViews[1].get()};
      std::vector<VkImageView> depthViews;
      std::vector<VkImageView> colorViews;
      depthViews.reserve(gBufferDepthViews.size());
      colorViews.reserve(swapchain.getImageCount());
      for (const auto &v : gBufferDepthViews)
        depthViews.push_back(v.get());
      for (size_t i = 0; i < swapchain.getImageCount(); ++i)
        colorViews.push_back(swapchain.getColorBufferView(i));
      descriptorManager.recreateTaaSets(device.getLogicalDevice(), histViews,
                                        depthViews, colorViews, taaSampler);
    }

    // 10. Pipeline (needs all 4 render passes + descriptor layouts)
    pipeline.createPipelines(
        device.getLogicalDevice(), renderPassManager.getGBufferRenderPass(),
        renderPassManager.getLitRenderPass(), renderPassManager.getRenderPass(),
        renderPassManager.getShadowRenderPass(), swapchain.getExtent(),
        descriptorManager);

    // 10. IBL + SSAO resources (requires textureManager and descriptorManager)
    initIBL();

    // 10b. Auto-exposure compute resources. Must come after createLitResources()
    //      (binds litViews) and after swapchain init (knows image count).
    createAutoExposureResources();

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
    // Intensity 2.5 (was 5.5) — at 5.5 the floor pixels under direct sun
    // overshoot AgX's mid-tone band and the tile pattern crushes into a
    // uniform bright wash. 2.5 lands the lit floor in AgX's preserved-
    // detail range so authored albedo+normal variation actually survives
    // the tonemap. This is a stopgap until the histogram auto-exposure
    // (N1 in mds/sponza_visual_diagnosis.md) can drive exposure
    // automatically; once that lands, sun intensity becomes a free knob.
    sceneUbo.directionalLight.colorIntensity =
        glm::vec4(1.0f, 0.93f, 0.78f, 2.5f);

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

    // shadowParams.y/.z/.w piggyback SSGI tunables since they were unused.
    // .x stays as the point-shadow far plane.
    sceneUbo.shadowParams = glm::vec4(imguiPointShadowFar, imguiSsgiIntensity,
                                      0.0f, 0.0f);
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

  // Day/night cycle: advance sim-hour at imguiDayNightSpeed sim-hours per
  // real-second, then place the sun on a great-circle arc through the
  // zenith. Sunrise at hour 6 (east horizon), zenith at hour 12, sunset at
  // hour 18 (west horizon); below the horizon outside [6, 18].
  if (imguiDayNightEnable) {
    static auto lastTime = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now - lastTime).count();
    lastTime = now;
    imguiDayNightHour += dt * imguiDayNightSpeed;
    imguiDayNightHour = std::fmod(imguiDayNightHour, 24.0f);
    if (imguiDayNightHour < 0.0f) imguiDayNightHour += 24.0f;

    float t     = (imguiDayNightHour - 6.0f) / 12.0f; // 0 at dawn, 1 at dusk
    float angle = t * 3.1415927f;                     // 0..π over daylight
    // Slight tilt off the X axis so shadows aren't perfectly E-W parallel.
    glm::vec3 east = glm::normalize(glm::vec3(1.0f, 0.0f, 0.2f));
    glm::vec3 up   = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 sunPos = east * std::cos(angle) + up * std::sin(angle);
    sceneUbo.directionalLight.direction = glm::vec4(-sunPos, 0.0f);

    // Intensity follows elevation: max at zenith, ambient floor at night.
    // Warm tint kicks in only near the horizon, not while underground.
    float elevation = sunPos.y;
    float dayFactor = glm::smoothstep(-0.05f, 0.35f, elevation);
    float horizon   = glm::clamp(1.0f - std::abs(elevation) * 3.0f, 0.0f, 1.0f) *
                      (elevation > 0.0f ? 1.0f : 0.0f);
    glm::vec3 dayColor(1.0f, 0.93f, 0.78f);
    glm::vec3 duskColor(1.0f, 0.55f, 0.30f);
    glm::vec3 color = glm::mix(dayColor, duskColor, horizon);
    // Peak 2.5 (was 5.5) — see comment in createSceneUBO. Stopgap until
    // histogram auto-exposure replaces the manual EV slider.
    sceneUbo.directionalLight.colorIntensity =
        glm::vec4(color, 2.5f * dayFactor + 0.05f);

    // IBL/sky baked from a daytime cubemap — attenuate it with the same
    // day factor so glossy surfaces don't keep reflecting baked daylight
    // through the night. User's manual slider still scales on top.
    sceneUbo.qualityToggles2.w =
        imguiIblIntensity * (0.08f + 0.92f * dayFactor);
  } else {
    // Day/night cycle off: manual slider drives the ambient term directly.
    sceneUbo.qualityToggles2.w = imguiIblIntensity;
  }

  sceneUbo.qualityToggles2.y = imguiUseGeomNormalOnly ? 1.0f : 0.0f;
  // Exposure path. Two contributors:
  //   1. Histogram auto-exposure (recordAutoExposurePass) wrote a target
  //      exposure scalar into autoExpResultMapped[currentFrame] one or two
  //      frames ago. We lerp the running adapted value toward it with the
  //      eye-adaptation time constant. drawFence ensures the GPU has
  //      finished writing this slot before we read.
  //   2. The manual EV slider stays as a bias offset (exp2 stops).
  // Auto-exposure off → adapted value pinned to 1.0; slider acts as before.
  {
    static auto exposureLastTime = std::chrono::steady_clock::now();
    auto exposureNow = std::chrono::steady_clock::now();
    float exposureDt = std::chrono::duration<float>(
                           exposureNow - exposureLastTime).count();
    exposureLastTime = exposureNow;
    // Clamp dt to keep first-frame / debugger-paused jumps sane.
    exposureDt = glm::clamp(exposureDt, 0.0f, 0.25f);

    if (autoExpEnabled && !autoExpResultMapped.empty()) {
      float target =
          *reinterpret_cast<const float *>(autoExpResultMapped[currentFrame]);
      if (!std::isfinite(target) || target <= 0.0f) target = 1.0f;
      // Clamp the target tightly. The Bruop formula `H = 1 / (9.6 × avg)`
      // routinely returns 3–4× on Sponza-interior shots (lots of dark
      // pixels drag avgLum low), which then amplifies every noise term
      // by the same factor. Hard cap at 2.5× displayed amplification so
      // residual SSGI / spec-AA grain doesn't get pushed past the
      // perceptual visibility threshold. Lower bound 0.5× keeps dark
      // scenes from going pitch black under a bright sun.
      target = glm::clamp(target, 0.5f, 2.5f);
      float alpha = 1.0f - std::exp(-exposureDt / kAutoExpTauSeconds);
      autoExpAdaptedValue =
          autoExpAdaptedValue + alpha * (target - autoExpAdaptedValue);
    } else {
      autoExpAdaptedValue = 1.0f;
    }
  }
  sceneUbo.qualityToggles2.x =
      autoExpAdaptedValue * std::exp2(imguiExposureEV);
  sceneUbo.qualityToggles2.z = imguiMinSurfaceRoughness;
  sceneUbo.qualityToggles.y  = imguiSkyOcclusionFloor;
  // shadowParams.y = SSGI intensity, .z = sharpening strength
  // (.w left at 0 → shader uses default for SSGI depth tolerance).
  sceneUbo.shadowParams.y    = imguiSsgiIntensity;
  sceneUbo.shadowParams.z    = imguiSharpness;

  // --- TAA per-frame state ---
  // Halton(2,3) sub-pixel jitter for free supersampling AA + temporal noise
  // washing. We sequence over 8 samples and wrap. Convert pixel offsets in
  // [-0.5, 0.5] to NDC offsets by *2/viewport, since clip.xy is in [-w, w]
  // and after divide we want a *2/viewport NDC shift to land on the
  // intended sub-pixel position.
  auto halton = [](uint32_t i, uint32_t base) {
    float f = 1.0f;
    float r = 0.0f;
    while (i > 0) { f /= float(base); r += f * float(i % base); i /= base; }
    return r;
  };
  uint32_t haltonIdx = (taaFrameCounter % 8) + 1; // skip i=0 → (0,0)
  glm::vec2 jitterPx = glm::vec2(halton(haltonIdx, 2) - 0.5f,
                                 halton(haltonIdx, 3) - 0.5f);
  VkExtent2D ext = swapchain.getExtent();
  glm::vec2 jitterNDC = glm::vec2(jitterPx.x * 2.0f / float(ext.width),
                                  jitterPx.y * 2.0f / float(ext.height));
  // prevViewProj must hold the *jittered* VP of the prev frame so that
  // multiplying a world position by it yields the prev frame's *jittered*
  // NDC — the location at which that surface point was actually rasterized
  // in the history image. Sampling history at the un-jittered prev NDC was
  // off by half a pixel each frame, which bilinear-blurred the image over
  // many frames (a textbook TAA blur bug).
  sceneUbo.prevViewProj = taaPrevViewProj;
  // Reset projection to the un-jittered baseline BEFORE applying this
  // frame's Halton offset. The previous implementation mutated
  // sceneUbo.projection in-place every frame, which accumulated the
  // per-frame offsets into a random-walk drift over time — the relation
  // between Halton index and actual sub-pixel position broke, so TAA
  // could never converge to clean supersampled edges (the symptom the
  // user reported: "textures slightly move all the time", and severely
  // jagged edges even on a static camera).
  sceneUbo.projection = taaBaseProjection;
  const bool taaEnable = true;
  if (taaEnable) {
    // Modify column 2 (the z coupling) so the offset is a constant screen-
    // space delta independent of vertex depth: clip.x_new = clip.x +
    // jitterNDC.x * w. After perspective divide this becomes a constant
    // +jitterNDC NDC offset on every fragment regardless of depth.
    sceneUbo.projection[2][0] -= jitterNDC.x;
    sceneUbo.projection[2][1] -= jitterNDC.y;
  }
  // invProj must invert whichever projection was just installed.
  sceneUbo.invProj = glm::inverse(sceneUbo.projection);
  // Save THIS frame's jittered VP for next frame's reprojection. Must come
  // AFTER the jitter mutation above.
  taaPrevViewProj = sceneUbo.projection * sceneUbo.view;
  sceneUbo.taaParams = glm::vec4(taaEnable ? jitterNDC.x : 0.0f,
                                 taaEnable ? jitterNDC.y : 0.0f,
                                 taaEnable ? 1.0f : 0.0f,
                                 taaHistoryValid ? 1.0f : 0.0f);
  sceneUbo.viewportSize = glm::vec4(float(ext.width), float(ext.height),
                                    1.0f / float(ext.width),
                                    1.0f / float(ext.height));
  // taaHistoryValid is the input to THIS frame's TAA blend (so the first
  // frame after init/resize must read it as false). Promote it to true for
  // the NEXT frame now that we're about to render content into the history.
  taaHistoryValid = true;

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
  // renderFinished is indexed by acquired swap-image, not by frame-in-flight,
  // so the presentation engine's wait binds to the right semaphore.
  submitInfo.pSignalSemaphores = &renderFinished[imageIndex];

  if (vkQueueSubmit(device.getGraphicsQueue(), 1, &submitInfo,
                    drawFences[currentFrame]) != VK_SUCCESS)
    throw std::runtime_error("Failed to submit draw command buffer");

  VkPresentInfoKHR presentInfo = {};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &renderFinished[imageIndex];
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
  // Advance TAA frame counter AFTER this frame finishes recording so the
  // Halton jitter index used above (line ~288) matches the parity that
  // recordCommands picked for the framebuffer / descriptor set.
  taaFrameCounter++;
  metrics.endFrame(logicalDevice);
}

// Swapchain recreation

void VulkanRenderer::recreateSwapChain() {
  swapchain.recreate(device, window);

  cleanupImGuiFramebuffers();
  createImGuiFramebuffers();

  cleanupAutoExposureResources();
  cleanupCompositeFramebuffers();
  cleanupGBuffer();
  cleanupLitResources();
  cleanupTaaResources();
  createLitResources();
  createTaaResources();
  createCompositeFramebuffers();
  createGBuffer();
  // Recreate auto-exposure AFTER lit (descriptors reference litViews).
  createAutoExposureResources();

  imagesInFlight.assign(swapchain.getImageCount(), VK_NULL_HANDLE);

  // renderFinished is sized by swap-image count; if the new swapchain has
  // a different count, resize and (re)create the missing semaphores.
  // Destroying and re-creating all of them is cheap and avoids stale
  // signaled state from before the resize.
  {
    VkDevice dev = device.getLogicalDevice();
    for (VkSemaphore s : renderFinished)
      vkDestroySemaphore(dev, s, nullptr);
    renderFinished.assign(swapchain.getImageCount(), VK_NULL_HANDLE);
    VkSemaphoreCreateInfo semInfo = {};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (size_t i = 0; i < renderFinished.size(); ++i) {
      if (vkCreateSemaphore(dev, &semInfo, nullptr, &renderFinished[i]) !=
          VK_SUCCESS)
        throw std::runtime_error("Failed to recreate renderFinished semaphore");
    }
  }

  descriptorManager.recreateInputSets(device.getLogicalDevice(), swapchain);
  {
    std::vector<VkImageView> histViews = {taaHistoryViews[0].get(),
                                          taaHistoryViews[1].get()};
    std::vector<VkImageView> depthViews;
    std::vector<VkImageView> colorViews;
    depthViews.reserve(gBufferDepthViews.size());
    colorViews.reserve(swapchain.getImageCount());
    for (const auto &v : gBufferDepthViews)
      depthViews.push_back(v.get());
    for (size_t i = 0; i < swapchain.getImageCount(); ++i)
      colorViews.push_back(swapchain.getColorBufferView(i));
    descriptorManager.recreateTaaSets(device.getLogicalDevice(), histViews,
                                      depthViews, colorViews, taaSampler);
  }

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

// TAA history images. Two persistent HDR images that ping-pong each frame:
// frame N writes taaHistory[N&1], samples taaHistory[(N+1)&1]. Format
// matches litFormat so reads from history feed straight into the same
// HDR-precision pipeline. Usage: SAMPLED (read as history-prev) +
// COLOR_ATTACHMENT (written as history-curr by the composite render pass
// once it gains the 2nd color attachment in the next session).
void VulkanRenderer::createTaaResources() {
  VkDevice dev = device.getLogicalDevice();
  taaHistoryImages.clear();
  taaHistoryViews.clear();
  taaHistoryImages.reserve(2);
  taaHistoryViews.reserve(2);

  VmaAllocationCreateInfo aci = {};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  for (int i = 0; i < 2; ++i) {
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
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage rawImg = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    if (vmaCreateImage(device.getAllocator(), &ci, &aci, &rawImg, &alloc,
                       nullptr) != VK_SUCCESS)
      throw std::runtime_error("Failed to create TAA history image");
    taaHistoryImages.emplace_back(device.getAllocator(), rawImg, alloc);

    VkImageViewCreateInfo vci = {};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = taaHistoryImages.back().get();
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = litFormat;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView v = VK_NULL_HANDLE;
    if (vkCreateImageView(dev, &vci, nullptr, &v) != VK_SUCCESS)
      throw std::runtime_error("Failed to create TAA history view");
    taaHistoryViews.emplace_back(dev, v);
  }

  // Transition both to SHADER_READ_ONLY_OPTIMAL so the first frame's
  // sampler bind doesn't read undefined data. The composite render pass
  // (when wired up next session) will declare initialLayout=UNDEFINED on
  // the history attachment with loadOp=DONT_CARE, so the transition back
  // to COLOR_ATTACHMENT_OPTIMAL is handled by the render pass itself.
  VkCommandBuffer cmd = beginCommandBuffer(dev, device.getGraphicsCommandPool());
  for (int i = 0; i < 2; ++i) {
    VkImageMemoryBarrier b = {};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = taaHistoryImages[i].get();
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.srcAccessMask = 0;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &b);
  }
  endAndSubmitCommandBuffer(dev, device.getGraphicsCommandPool(),
                            device.getGraphicsQueue(), cmd);

  // Sampler used by the TAA shader to read history-prev and depth.
  // CLAMP_TO_EDGE so reprojected UVs that slide just past the screen edge
  // sample the edge texel rather than wrapping; the shader's explicit
  // [0,1] bounds check handles true disocclusion.
  if (taaSampler == VK_NULL_HANDLE) {
    VkSamplerCreateInfo sci = {};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxAnisotropy = 1.0f;
    if (vkCreateSampler(dev, &sci, nullptr, &taaSampler) != VK_SUCCESS)
      throw std::runtime_error("Failed to create TAA sampler");
  }

  // Fresh allocation → history is undefined → don't blend the first frame.
  taaHistoryValid = false;
}

void VulkanRenderer::cleanupTaaResources() {
  // ImageViewHandle / AllocatedImage destructors handle vkDestroyImageView /
  // vmaDestroyImage. We just clear the vectors.
  taaHistoryViews.clear();
  taaHistoryImages.clear();
  if (taaSampler != VK_NULL_HANDLE) {
    vkDestroySampler(device.getLogicalDevice(), taaSampler, nullptr);
    taaSampler = VK_NULL_HANDLE;
  }
}

// Composite framebuffers: 2 * swapCount entries, indexed by
// (parity * swapCount + swapIdx). Each binds:
//   attachment 0 = swap image view (per swapIdx)
//   attachment 1 = colorBuffer view (per swapIdx)
//   attachment 2 = TAA history view (per parity — alternates each frame)
void VulkanRenderer::createCompositeFramebuffers() {
  VkDevice dev = device.getLogicalDevice();
  size_t swapCount = swapchain.getImageCount();
  compositeFramebuffers.assign(2 * swapCount, VK_NULL_HANDLE);

  for (size_t parity = 0; parity < 2; ++parity) {
    for (size_t i = 0; i < swapCount; ++i) {
      std::array<VkImageView, 3> atts = {swapchain.getSwapImageView(i),
                                         swapchain.getColorBufferView(i),
                                         taaHistoryViews[parity].get()};
      VkFramebufferCreateInfo ci = {};
      ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      ci.renderPass = renderPassManager.getRenderPass();
      ci.attachmentCount = static_cast<uint32_t>(atts.size());
      ci.pAttachments = atts.data();
      ci.width = swapchain.getExtent().width;
      ci.height = swapchain.getExtent().height;
      ci.layers = 1;
      VkFramebuffer fb = VK_NULL_HANDLE;
      if (vkCreateFramebuffer(dev, &ci, nullptr, &fb) != VK_SUCCESS)
        throw std::runtime_error("Failed to create composite framebuffer");
      compositeFramebuffers[parity * swapCount + i] = fb;
    }
  }
}

void VulkanRenderer::cleanupCompositeFramebuffers() {
  VkDevice dev = device.getLogicalDevice();
  for (VkFramebuffer fb : compositeFramebuffers)
    if (fb != VK_NULL_HANDLE)
      vkDestroyFramebuffer(dev, fb, nullptr);
  compositeFramebuffers.clear();
}

// --- Auto-exposure (histogram-based) ---------------------------------------
// Two-pass compute path. Pass 1 builds a 256-bin log-luminance histogram of
// the litBuffer (post-deferred HDR). Pass 2 reduces the histogram to a
// single target exposure scalar, written into a host-visible result buffer
// per frame-in-flight. The CPU reads the *previous* frame's value next
// frame (one-frame latency, irrelevant for ~1.5 s eye adaptation) and
// lerps the running adapted-exposure toward it, then pushes that into
// sceneUbo.qualityToggles2.x.

namespace {
VkShaderModule loadComputeSpv(VkDevice device, const std::string &relPath) {
  // Same dual-candidate resolve pattern as TextureManager's loader so the
  // binary works whether launched from build/ or repo root.
  std::vector<std::string> candidates = {relPath, "../" + relPath};
  std::string found;
  for (const auto &c : candidates)
    if (std::filesystem::exists(c)) { found = c; break; }
  if (found.empty())
    throw std::runtime_error("Compute shader not found: " + relPath);

  auto code = readFile(found);
  VkShaderModuleCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  ci.codeSize = code.size();
  ci.pCode = reinterpret_cast<const uint32_t *>(code.data());
  VkShaderModule mod = VK_NULL_HANDLE;
  if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS)
    throw std::runtime_error("Failed to create shader module: " + found);
  return mod;
}
} // namespace

void VulkanRenderer::createAutoExposureResources() {
  VkDevice dev = device.getLogicalDevice();
  VmaAllocator alloc = device.getAllocator();
  const size_t swapCount = swapchain.getImageCount();

  // Device-local histogram buffer. 256 uints × 4 B = 1 KiB.
  createBuffer(alloc, sizeof(uint32_t) * 256,
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0,
               &autoExpHistogramBuffer);

  // Zero the histogram once at creation so frame 0 of exposure.comp doesn't
  // read garbage. exposure.comp resets bins per-frame after reducing.
  {
    VkCommandBuffer cb = beginCommandBuffer(dev, device.getGraphicsCommandPool());
    vkCmdFillBuffer(cb, autoExpHistogramBuffer.get(), 0, VK_WHOLE_SIZE, 0);
    endAndSubmitCommandBuffer(dev, device.getGraphicsCommandPool(),
                              device.getGraphicsQueue(), cb);
  }

  // Host-visible result buffers (one per frame-in-flight). Persistently
  // mapped; CPU reads each frame, GPU writes via storage-buffer binding.
  autoExpResultBuffers.clear();
  autoExpResultMapped.clear();
  autoExpResultBuffers.reserve(MAX_FRAMES_DRAWS);
  autoExpResultMapped.reserve(MAX_FRAMES_DRAWS);
  for (int i = 0; i < MAX_FRAMES_DRAWS; ++i) {
    AllocatedBuffer buf;
    // 16 bytes = vec4 alignment for std430. Only the first float is used.
    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = 16;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo aci = {};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    // RANDOM + MAPPED: keep the mapping valid across frames and allow
    // both writes (CPU init / debug) and reads. HOST_ACCESS implies
    // HOST_VISIBLE; AUTO picks coherent on platforms that support it.
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer rawBuf = VK_NULL_HANDLE;
    VmaAllocation rawAlloc = VK_NULL_HANDLE;
    VmaAllocationInfo rawInfo = {};
    if (vmaCreateBuffer(alloc, &bci, &aci, &rawBuf, &rawAlloc, &rawInfo) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to create auto-exposure result buffer");

    // Seed initial value 1.0 so first few frames render at neutral scale
    // before the compute path runs.
    *reinterpret_cast<float *>(rawInfo.pMappedData) = 1.0f;

    autoExpResultBuffers.emplace_back(alloc, rawBuf, rawAlloc);
    autoExpResultMapped.push_back(rawInfo.pMappedData);
  }

  // Set layouts.
  {
    std::array<VkDescriptorSetLayoutBinding, 2> b{};
    b[0].binding = 0;
    b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[0].descriptorCount = 1;
    b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[1].binding = 1;
    b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[1].descriptorCount = 1;
    b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = 2;
    ci.pBindings = b.data();
    if (vkCreateDescriptorSetLayout(dev, &ci, nullptr,
                                    &autoExpHistogramSetLayout) != VK_SUCCESS)
      throw std::runtime_error("Failed to create autoExp histogram set layout");
  }
  {
    std::array<VkDescriptorSetLayoutBinding, 2> b{};
    b[0].binding = 0;
    b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[0].descriptorCount = 1;
    b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[1].binding = 1;
    b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[1].descriptorCount = 1;
    b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = 2;
    ci.pBindings = b.data();
    if (vkCreateDescriptorSetLayout(dev, &ci, nullptr,
                                    &autoExpExposureSetLayout) != VK_SUCCESS)
      throw std::runtime_error("Failed to create autoExp exposure set layout");
  }

  // Pipeline layouts (push constants carry per-dispatch params).
  {
    struct HistPC {
      float minLogLum;
      float invLogLumRange;
      uint32_t width;
      uint32_t height;
    };
    VkPushConstantRange pc = {};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = sizeof(HistPC);
    VkPipelineLayoutCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ci.setLayoutCount = 1;
    ci.pSetLayouts = &autoExpHistogramSetLayout;
    ci.pushConstantRangeCount = 1;
    ci.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(dev, &ci, nullptr,
                               &autoExpHistogramPipelineLayout) != VK_SUCCESS)
      throw std::runtime_error("Failed to create autoExp histogram pipeline layout");
  }
  {
    struct ExpPC {
      float minLogLum;
      float logLumRange;
      uint32_t numPixels;
    };
    VkPushConstantRange pc = {};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = sizeof(ExpPC);
    VkPipelineLayoutCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ci.setLayoutCount = 1;
    ci.pSetLayouts = &autoExpExposureSetLayout;
    ci.pushConstantRangeCount = 1;
    ci.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(dev, &ci, nullptr,
                               &autoExpExposurePipelineLayout) != VK_SUCCESS)
      throw std::runtime_error("Failed to create autoExp exposure pipeline layout");
  }

  // Descriptor pool. swapCount histogram sets + MAX_FRAMES_DRAWS exposure
  // sets; each consumes a few descriptors.
  {
    std::array<VkDescriptorPoolSize, 2> ps{};
    ps[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ps[0].descriptorCount = static_cast<uint32_t>(swapCount);
    ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps[1].descriptorCount = static_cast<uint32_t>(swapCount + 2 * MAX_FRAMES_DRAWS);
    VkDescriptorPoolCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.maxSets = static_cast<uint32_t>(swapCount + MAX_FRAMES_DRAWS);
    ci.poolSizeCount = static_cast<uint32_t>(ps.size());
    ci.pPoolSizes = ps.data();
    if (vkCreateDescriptorPool(dev, &ci, nullptr, &autoExpDescriptorPool) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to create autoExp descriptor pool");
  }

  // Allocate sets.
  {
    std::vector<VkDescriptorSetLayout> layouts(swapCount,
                                               autoExpHistogramSetLayout);
    VkDescriptorSetAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = autoExpDescriptorPool;
    ai.descriptorSetCount = static_cast<uint32_t>(swapCount);
    ai.pSetLayouts = layouts.data();
    autoExpHistogramSets.resize(swapCount);
    if (vkAllocateDescriptorSets(dev, &ai, autoExpHistogramSets.data()) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to alloc autoExp histogram sets");
  }
  {
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_DRAWS,
                                               autoExpExposureSetLayout);
    VkDescriptorSetAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = autoExpDescriptorPool;
    ai.descriptorSetCount = MAX_FRAMES_DRAWS;
    ai.pSetLayouts = layouts.data();
    autoExpExposureSets.resize(MAX_FRAMES_DRAWS);
    if (vkAllocateDescriptorSets(dev, &ai, autoExpExposureSets.data()) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to alloc autoExp exposure sets");
  }

  // Write descriptors. Histogram set i ↔ litViews[i] + histogramBuffer.
  for (size_t i = 0; i < swapCount; ++i) {
    VkDescriptorImageInfo ii = {};
    ii.imageView = litViews[i].get();
    ii.sampler = litSampler;
    ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorBufferInfo bi = {};
    bi.buffer = autoExpHistogramBuffer.get();
    bi.offset = 0;
    bi.range = VK_WHOLE_SIZE;
    std::array<VkWriteDescriptorSet, 2> w{};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = autoExpHistogramSets[i];
    w[0].dstBinding = 0;
    w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[0].pImageInfo = &ii;
    w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[1].dstSet = autoExpHistogramSets[i];
    w[1].dstBinding = 1;
    w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[1].pBufferInfo = &bi;
    vkUpdateDescriptorSets(dev, static_cast<uint32_t>(w.size()), w.data(), 0,
                           nullptr);
  }
  // Exposure set i ↔ histogramBuffer + result buffer i.
  for (int i = 0; i < MAX_FRAMES_DRAWS; ++i) {
    VkDescriptorBufferInfo hi = {};
    hi.buffer = autoExpHistogramBuffer.get();
    hi.offset = 0;
    hi.range = VK_WHOLE_SIZE;
    VkDescriptorBufferInfo ri = {};
    ri.buffer = autoExpResultBuffers[i].get();
    ri.offset = 0;
    ri.range = VK_WHOLE_SIZE;
    std::array<VkWriteDescriptorSet, 2> w{};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = autoExpExposureSets[i];
    w[0].dstBinding = 0;
    w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[0].pBufferInfo = &hi;
    w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[1].dstSet = autoExpExposureSets[i];
    w[1].dstBinding = 1;
    w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[1].pBufferInfo = &ri;
    vkUpdateDescriptorSets(dev, static_cast<uint32_t>(w.size()), w.data(), 0,
                           nullptr);
  }

  // Pipelines.
  {
    VkShaderModule histMod = loadComputeSpv(dev, "Shaders/histogram.comp.spv");
    VkShaderModule expMod  = loadComputeSpv(dev, "Shaders/exposure.comp.spv");
    auto build = [&](VkShaderModule m, VkPipelineLayout pl) {
      VkPipelineShaderStageCreateInfo s = {};
      s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      s.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      s.module = m;
      s.pName = "main";
      VkComputePipelineCreateInfo ci = {};
      ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
      ci.stage = s;
      ci.layout = pl;
      VkPipeline p = VK_NULL_HANDLE;
      if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p) !=
          VK_SUCCESS)
        throw std::runtime_error("Failed to create autoExp compute pipeline");
      return p;
    };
    autoExpHistogramPipeline = build(histMod, autoExpHistogramPipelineLayout);
    autoExpExposurePipeline  = build(expMod,  autoExpExposurePipelineLayout);
    vkDestroyShaderModule(dev, histMod, nullptr);
    vkDestroyShaderModule(dev, expMod,  nullptr);
  }
}

void VulkanRenderer::cleanupAutoExposureResources() {
  VkDevice dev = device.getLogicalDevice();
  if (autoExpHistogramPipeline) {
    vkDestroyPipeline(dev, autoExpHistogramPipeline, nullptr);
    autoExpHistogramPipeline = VK_NULL_HANDLE;
  }
  if (autoExpExposurePipeline) {
    vkDestroyPipeline(dev, autoExpExposurePipeline, nullptr);
    autoExpExposurePipeline = VK_NULL_HANDLE;
  }
  if (autoExpHistogramPipelineLayout) {
    vkDestroyPipelineLayout(dev, autoExpHistogramPipelineLayout, nullptr);
    autoExpHistogramPipelineLayout = VK_NULL_HANDLE;
  }
  if (autoExpExposurePipelineLayout) {
    vkDestroyPipelineLayout(dev, autoExpExposurePipelineLayout, nullptr);
    autoExpExposurePipelineLayout = VK_NULL_HANDLE;
  }
  if (autoExpDescriptorPool) {
    vkDestroyDescriptorPool(dev, autoExpDescriptorPool, nullptr);
    autoExpDescriptorPool = VK_NULL_HANDLE;
  }
  autoExpHistogramSets.clear();
  autoExpExposureSets.clear();
  if (autoExpHistogramSetLayout) {
    vkDestroyDescriptorSetLayout(dev, autoExpHistogramSetLayout, nullptr);
    autoExpHistogramSetLayout = VK_NULL_HANDLE;
  }
  if (autoExpExposureSetLayout) {
    vkDestroyDescriptorSetLayout(dev, autoExpExposureSetLayout, nullptr);
    autoExpExposureSetLayout = VK_NULL_HANDLE;
  }
  autoExpResultBuffers.clear();        // AllocatedBuffer RAII unmaps
  autoExpResultMapped.clear();
  autoExpHistogramBuffer.reset();
}

void VulkanRenderer::recordAutoExposurePass(VkCommandBuffer cmd,
                                            uint32_t currentImage) {
  if (!autoExpEnabled || autoExpHistogramPipeline == VK_NULL_HANDLE) return;

  VkExtent2D ext = swapchain.getExtent();
  const float logLumRange  = kAutoExpMaxLogLum - kAutoExpMinLogLum;
  const float invLogLumRng = 1.0f / logLumRange;

  vkdbgBeginLabel(cmd, "Auto-Exposure (histogram + reduce)", 0.95f, 0.85f, 0.2f);

  // The lit render pass declares finalLayout=SHADER_READ_ONLY_OPTIMAL, but
  // the implicit subpass dependency stops at BOTTOM_OF_PIPE — we need to
  // explicitly extend the visibility into COMPUTE_SHADER for the sampler
  // read below to see committed pixel writes.
  {
    VkImageMemoryBarrier b = {};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = litImages[currentImage].get();
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &b);
  }

  // Pass 1: build histogram.
  {
    struct HistPC {
      float minLogLum;
      float invLogLumRange;
      uint32_t width;
      uint32_t height;
    } pc{kAutoExpMinLogLum, invLogLumRng, ext.width, ext.height};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      autoExpHistogramPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            autoExpHistogramPipelineLayout, 0, 1,
                            &autoExpHistogramSets[currentImage], 0, nullptr);
    vkCmdPushConstants(cmd, autoExpHistogramPipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    uint32_t gx = (ext.width  + 15) / 16;
    uint32_t gy = (ext.height + 15) / 16;
    vkCmdDispatch(cmd, gx, gy, 1);
  }

  // Histogram writes → exposure pass reads. Same buffer; need a memory
  // dependency between the two compute dispatches.
  {
    VkMemoryBarrier mb = {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
  }

  // Pass 2: reduce → exposure result.
  {
    struct ExpPC {
      float minLogLum;
      float logLumRange;
      uint32_t numPixels;
    } pc{kAutoExpMinLogLum, logLumRange, ext.width * ext.height};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      autoExpExposurePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            autoExpExposurePipelineLayout, 0, 1,
                            &autoExpExposureSets[currentFrame], 0, nullptr);
    vkCmdPushConstants(cmd, autoExpExposurePipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, 1, 1, 1);
  }

  // Result buffer is host-coherent, but Vulkan still needs an explicit
  // memory dependency from device write → host read so the read on the
  // CPU side after the fence is well-defined.
  {
    VkMemoryBarrier mb = {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0,
                         nullptr);
  }

  vkdbgEndLabel(cmd);
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
  cleanupAutoExposureResources();
  cleanupCompositeFramebuffers();
  cleanupGBuffer();
  cleanupLitResources();
  cleanupTaaResources();
  cleanupIBL();
  cleanupShadowResources();

  descriptorManager.cleanup(device.getLogicalDevice(), device.getAllocator(),
                            swapchain.getImageCount());
  swapchain.cleanup(device.getLogicalDevice(), device.getAllocator());

  for (size_t i = 0; i < MAX_FRAMES_DRAWS; i++) {
    vkDestroySemaphore(device.getLogicalDevice(), imageAvailable[i], nullptr);
    vkDestroyFence(device.getLogicalDevice(), drawFences[i], nullptr);
  }
  for (VkSemaphore s : renderFinished)
    vkDestroySemaphore(device.getLogicalDevice(), s, nullptr);
  renderFinished.clear();

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
    // Cull mode is now dynamic so the per-draw override below can flip to
    // NONE for materials marked doubleSided (Sponza foliage). Default
    // matches the original static state.
    vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
    VkCullModeFlags lastCullMode = VK_CULL_MODE_BACK_BIT;

    glm::vec4 frustumPlanes[6];
    extractFrustumPlanes(sceneUbo.projection * sceneUbo.view, frustumPlanes);

    // Phase 7.2: bind VP + bindless texture array once per frame.
    // Individual materials no longer have their own descriptor sets;
    // texture indices are passed via push constants per-draw.
    VkDescriptorSet vpSet = descriptorManager.getVPSet(currentImage);
    std::array<VkDescriptorSet, 2> gbSets = {
        vpSet, descriptorManager.getBindlessSet()};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getGraphicsLayout(), 0, 2,
                            gbSets.data(), 0, nullptr);

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

              const Material &mat = mesh->getMaterial();
              VkCullModeFlags wantCull = mat.doubleSided
                                             ? VK_CULL_MODE_NONE
                                             : VK_CULL_MODE_BACK_BIT;
              if (wantCull != lastCullMode) {
                vkCmdSetCullMode(cmd, wantCull);
                lastCullMode = wantCull;
              }
              // Phase 7.2: fill bindless texture indices from material.
              // texIdx1.y is a packed bitfield of per-material flags
              // (bit 0 = isCloth). Read by shader.frag → gBuffer2.g.
              uint32_t materialFlags = mat.isCloth ? 1u : 0u;
              push.texIdx0 = glm::uvec4(
                  static_cast<uint32_t>(mat.albedoTextureId),
                  static_cast<uint32_t>(mat.normalTextureId),
                  static_cast<uint32_t>(mat.metallicTextureId),
                  static_cast<uint32_t>(mat.roughnessTextureId));
              push.texIdx1 = glm::uvec4(
                  static_cast<uint32_t>(mat.aoTextureId), materialFlags, 0u, 0u);
              vkCmdPushConstants(cmd, pipeline.getGraphicsLayout(),
                                 VK_SHADER_STAGE_VERTEX_BIT |
                                     VK_SHADER_STAGE_FRAGMENT_BIT,
                                 0, sizeof(ModelPushConstants), &push);
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
      // Instanced pipeline shares geoDynState (CULL_MODE dynamic) — set it
      // explicitly. Instanced models in the scene are opaque, no per-mesh
      // override needed.
      vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
      // Phase 7.2: bind VP + bindless for instanced pipeline too.
      std::array<VkDescriptorSet, 2> instSets = {
          descriptorManager.getVPSet(currentImage),
          descriptorManager.getBindlessSet()};
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipeline.getInstancedLayout(), 0, 2,
                              instSets.data(), 0, nullptr);
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
          // Phase 7.2: push bindless texture indices for this mesh.
          // model/normal are unused by the instanced vertex shader (it reads
          // them from the instance buffer), but the struct must match.
          const Material &iMat = mesh->getMaterial();
          uint32_t iMaterialFlags = iMat.isCloth ? 1u : 0u;
          ModelPushConstants iPush{};
          iPush.texIdx0 = glm::uvec4(
              static_cast<uint32_t>(iMat.albedoTextureId),
              static_cast<uint32_t>(iMat.normalTextureId),
              static_cast<uint32_t>(iMat.metallicTextureId),
              static_cast<uint32_t>(iMat.roughnessTextureId));
          iPush.texIdx1 = glm::uvec4(
              static_cast<uint32_t>(iMat.aoTextureId), iMaterialFlags, 0u, 0u);
          vkCmdPushConstants(cmd, pipeline.getInstancedLayout(),
                             VK_SHADER_STAGE_VERTEX_BIT |
                                 VK_SHADER_STAGE_FRAGMENT_BIT,
                             0, sizeof(ModelPushConstants), &iPush);
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

  // --- Auto-exposure (compute) ---
  // Runs between lit and composite: lit is now in SHADER_READ_ONLY_OPTIMAL
  // (the render pass's finalLayout). Pass 1 histograms it; pass 2 reduces
  // and writes one float into a host-visible buffer that next frame's CPU
  // reads to drive eye adaptation.
  recordAutoExposurePass(cmd, currentImage);

  // --- Composition pass (SSR composite + TAA + ACES tone-mapping) ---
  {
    // 3 attachments: swap (clear), colorBuffer (clear), history (DONT_CARE).
    // Clear values for DONT_CARE attachments are ignored but the array still
    // needs the right element count.
    std::array<VkClearValue, 3> clears{};
    clears[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
    clears[1].color = {0.0f, 0.0f, 0.0f, 1.0f};
    clears[2].color = {0.0f, 0.0f, 0.0f, 1.0f};

    // Parity selects which TAA history image is THIS frame's write target.
    // taaFrameCounter is incremented at end-of-draw, so its current value is
    // the index of the frame we're recording.
    uint32_t parity = taaFrameCounter & 1u;
    size_t fbIdx = parity * swapchain.getImageCount() + currentImage;

    VkRenderPassBeginInfo rpbi = {};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = renderPassManager.getRenderPass();
    rpbi.framebuffer = compositeFramebuffers[fbIdx];
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

    // Subpass 1: TAA reprojection + ACES + gamma → swapchain (LDR) + history (HDR)
    vkCmdNextSubpass(cmd, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getSecondPipeline());
    // 3 sets: scene UBO (0), input attachment (1), TAA samplers (2).
    // TAA set is picked by parity to point at the OTHER ping-pong image.
    std::array<VkDescriptorSet, 3> secondSets = {
        descriptorManager.getVPSet(currentImage),
        descriptorManager.getInputSet(currentImage),
        descriptorManager.getTaaSet(fbIdx)};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getSecondLayout(), 0,
                            static_cast<uint32_t>(secondSets.size()),
                            secondSets.data(), 0, nullptr);
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
  // Cascade 2 covers the mid-range where walls/columns are still big enough
  // for missing geometry to leak sun onto interior floors — keep it at LOD 1.
  // Only cascade 3 (farthest) drops to LOD 2.
  auto cascadeLod = [](int cascade) {
    if (cascade == 0)
      return 0;
    if (cascade <= 2)
      return 1;
    return 2; // cascade 3 only
  };

  // Default shadow cull mode is set in the per-cascade loop below; track the
  // current state here so the per-mesh override only flips on transitions.
  // OFF state is CULL_NONE rather than BACK_BIT: glTF wall winding is not
  // guaranteed to face the sun, and a back-cull setup would miss the same
  // single-sided walls that front-cull does — just on the opposite side.
  VkCullModeFlags defaultShadowCull = imguiShadowFrontFaceCull
                                          ? VK_CULL_MODE_FRONT_BIT
                                          : VK_CULL_MODE_NONE;
  VkCullModeFlags shadowLastCull = defaultShadowCull;

  auto renderNodeShadow = [&](auto &self, SceneNode *node, const glm::mat4 &lsm,
                              int lodIndex) -> void {
    if (node->getModelId() >= 0) {
      MeshModel *mdl = modelManager.getModel(node->getModelId());
      if (mdl) {
        for (size_t k = 0; k < mdl->getMeshCount(); k++) {
          const Mesh *mesh = mdl->getMesh(k);
          // doubleSided foliage: front-face culling discards the lit side and
          // the only remaining face faces away from the sun, so the leaf
          // casts no shadow. Use NONE so both sides write depth.
          VkCullModeFlags wantCull =
              mesh->getMaterial().doubleSided ? VK_CULL_MODE_NONE
                                              : defaultShadowCull;
          if (wantCull != shadowLastCull) {
            vkCmdSetCullMode(cmdBuffer, wantCull);
            shadowLastCull = wantCull;
          }
          // Phase 7.2: push model + LSM + albedo bindless index per mesh.
          ShadowPushConstants push{};
          push.model = node->getGlobalTransform();
          push.lightSpaceMatrix = lsm;
          push.albedoIdx =
              static_cast<uint32_t>(mesh->getMaterial().albedoTextureId);
          vkCmdPushConstants(cmdBuffer, pipeline.getShadowLayout(),
                             VK_SHADER_STAGE_VERTEX_BIT |
                                 VK_SHADER_STAGE_FRAGMENT_BIT,
                             0, sizeof(ShadowPushConstants), &push);
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
    // doubleSided meshes get overridden to NONE per-draw inside the lambda.
    vkCmdSetCullMode(cmdBuffer, defaultShadowCull);
    shadowLastCull = defaultShadowCull;
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getShadowPipeline());
    // Phase 7.2: bind bindless set once per cascade.
    VkDescriptorSet bindlessSet = descriptorManager.getBindlessSet();
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getShadowLayout(), 0, 1, &bindlessSet,
                            0, nullptr);
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
    VkCullModeFlags defaultPointCull = imguiShadowFrontFaceCull
                                           ? VK_CULL_MODE_FRONT_BIT
                                           : VK_CULL_MODE_BACK_BIT;
    vkCmdSetCullMode(cmdBuffer, defaultPointCull);
    VkCullModeFlags pointLastCull = defaultPointCull;
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getShadowPipeline());
    // Phase 7.2: bind bindless set once per face.
    VkDescriptorSet bindlessSet = descriptorManager.getBindlessSet();
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getShadowLayout(), 0, 1, &bindlessSet,
                            0, nullptr);

    auto renderNodeShadow = [&](auto &self, SceneNode *node) -> void {
      if (node->getModelId() >= 0) {
        MeshModel *mdl = modelManager.getModel(node->getModelId());
        if (mdl) {
          for (size_t k = 0; k < mdl->getMeshCount(); k++) {
            const Mesh *mesh = mdl->getMesh(k);
            VkCullModeFlags wantCull =
                mesh->getMaterial().doubleSided ? VK_CULL_MODE_NONE
                                                : defaultPointCull;
            if (wantCull != pointLastCull) {
              vkCmdSetCullMode(cmdBuffer, wantCull);
              pointLastCull = wantCull;
            }
            // Phase 7.2: push per-mesh with albedo index.
            ShadowPushConstants push{};
            push.model = node->getGlobalTransform();
            push.lightSpaceMatrix = pointShadowMatrices[face];
            push.albedoIdx =
                static_cast<uint32_t>(mesh->getMaterial().albedoTextureId);
            vkCmdPushConstants(cmdBuffer, pipeline.getShadowLayout(),
                               VK_SHADER_STAGE_VERTEX_BIT |
                                   VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(ShadowPushConstants), &push);
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
  // When lightDir is near-parallel to world-up, glm::lookAt's basis
  // degenerates (right = cross(up, -lightDir) → 0 → divide-by-zero NaN that
  // propagates through every cascade matrix and reads back as fractured
  // geometry at high noon / midnight). Swap to a horizontal "up" when the
  // sun is within ~8° of zenith so the orthonormal basis stays defined.
  glm::vec3 lightUp = (std::abs(lightDir.y) > 0.99f)
                          ? glm::vec3(0.0f, 0.0f, 1.0f)
                          : glm::vec3(0.0f, 1.0f, 0.0f);
  glm::mat4 lightView =
      glm::lookAt(-lightDir * 100.0f, glm::vec3(0.0f), lightUp);
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

  // Compare-enabled sibling for the CSM array: turns each tap into a
  // hardware-bilinear-filtered depth comparison instead of a binary one,
  // killing the stair-step / dithered PCF pattern.
  VkSamplerCreateInfo csmSamplerInfo = samplerInfo;
  csmSamplerInfo.compareEnable = VK_TRUE;
  csmSamplerInfo.compareOp     = VK_COMPARE_OP_LESS_OR_EQUAL;
  if (vkCreateSampler(dev, &csmSamplerInfo, nullptr, &csmShadowSampler) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create CSM compare sampler");

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
  if (csmShadowSampler != VK_NULL_HANDLE) {
    vkDestroySampler(device.getLogicalDevice(), csmShadowSampler, nullptr);
    csmShadowSampler = VK_NULL_HANDLE;
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
  // Capture the un-jittered baseline — draw() will copy this into
  // sceneUbo.projection at the start of every frame before applying the
  // per-frame Halton offset, so the jitter does not accumulate across
  // frames as a random-walk drift.
  taaBaseProjection = sceneUbo.projection;
  // Any FOV/aspect/draw-distance change invalidates TAA history because the
  // reprojection math depends on the previous frame's projection.
  taaHistoryValid = false;
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
  ImGui::SetNextWindowSize(ImVec2(340, 95), ImGuiCond_Always);
  ImGui::Begin("Debug Views", nullptr,
               ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoCollapse);
  const char *debugModes[] = {"None",          "Albedo",      "Normals",
                              "Metallic",      "Roughness",   "Depth",
                              "Shadow vis",    "SSAO factor", "Direct only",
                              "Indirect only", "Direct (no shadow)",
                              "SSGI bounce"};
  if (ImGui::Combo("G-Buffer", &imguiDebugMode, debugModes, 12))
    sceneUbo.debugMode = imguiDebugMode;
  ImGui::Checkbox("Use geometric normal only", &imguiUseGeomNormalOnly);
  ImGui::End();

  // Tunables + scene controls. The per-fix A/B checkboxes that lived here
  // before were retired once each fix was validated — keeping only the
  // values that actually need runtime tuning.
  ImGui::SetNextWindowPos(ImVec2(10, 700), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(340, 340), ImGuiCond_Always);
  ImGui::Begin("Scene Controls", nullptr,
               ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoCollapse);
  if (ImGui::SliderFloat("AA variance", &imguiSpecAAVariance, 0.0f, 2.0f,
                         "%.3f"))
    sceneUbo.qualityToggles.z = imguiSpecAAVariance;
  if (ImGui::SliderFloat("AA threshold", &imguiSpecAAThreshold, 0.0f, 1.0f,
                         "%.3f"))
    sceneUbo.qualityToggles.w = imguiSpecAAThreshold;
  if (ImGui::SliderFloat("IBL roughness floor", &imguiIblRoughnessFloor, 0.0f,
                         1.0f, "%.3f"))
    sceneUbo.qualityToggles.x = imguiIblRoughnessFloor;
  // Min-roughness floor for non-metals at the g-buffer write.
  ImGui::SliderFloat("Min surface roughness", &imguiMinSurfaceRoughness, 0.0f,
                     1.0f, "%.2f");
  // Sky-occlusion proxy floor — bottom of IBL multiplier in shadow.
  ImGui::SliderFloat("Sky occlusion floor", &imguiSkyOcclusionFloor, 0.0f,
                     1.0f, "%.2f");
  ImGui::SliderFloat("CSM far", &imguiCsmFar, 100.0f, 5000.0f, "%.0f");
  ImGui::Checkbox("Shadow front-face cull", &imguiShadowFrontFaceCull);
  ImGui::SliderFloat("IBL / sky intensity", &imguiIblIntensity, 0.0f, 2.0f,
                     "%.2f");
  // Auto-exposure (histogram-based). When on, the manual EV slider acts
  // as a bias on top of the adapted target.
  ImGui::Checkbox("Auto-exposure", &autoExpEnabled);
  ImGui::SameLine();
  ImGui::Text("(%.2fx)", autoExpAdaptedValue);
  // Pre-tonemap exposure (EV stops). Lifts midtones into AgX's linear
  // range. With auto-exposure on this is a bias offset, not absolute.
  ImGui::SliderFloat("Exposure (EV)", &imguiExposureEV, -3.0f, 3.0f, "%+.2f");
  // SSGI intensity — one-bounce diffuse gather. 0 = off, 1 = default,
  // 2 = strong (visible artifacts at silhouettes).
  ImGui::SliderFloat("SSGI intensity", &imguiSsgiIntensity, 0.0f, 2.0f,
                     "%.2f");
  // Sharpening strength after AgX. 0 = off, 0.4 = default light, 1.0 =
  // strong / starts ringing.
  ImGui::SliderFloat("Sharpness", &imguiSharpness, 0.0f, 1.0f, "%.2f");
  ImGui::Separator();
  ImGui::Checkbox("Day/night cycle", &imguiDayNightEnable);
  ImGui::SliderFloat("Sim hour", &imguiDayNightHour, 0.0f, 24.0f, "%.2f h");
  ImGui::SliderFloat("Speed (sim-h / real-s)", &imguiDayNightSpeed, 0.1f,
                     600.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
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
  size_t swapCount = swapchain.getImageCount();
  imageAvailable.resize(MAX_FRAMES_DRAWS);
  drawFences.resize(MAX_FRAMES_DRAWS);
  // renderFinished is per-swap-image, not per-frame-in-flight (see header
  // comment for the validation rationale).
  renderFinished.resize(swapCount);
  imagesInFlight.resize(swapCount, VK_NULL_HANDLE);

  VkSemaphoreCreateInfo semInfo = {};
  semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  VkDevice dev = device.getLogicalDevice();
  for (size_t i = 0; i < MAX_FRAMES_DRAWS; i++) {
    if (vkCreateSemaphore(dev, &semInfo, nullptr, &imageAvailable[i]) !=
            VK_SUCCESS ||
        vkCreateFence(dev, &fenceInfo, nullptr, &drawFences[i]) != VK_SUCCESS)
      throw std::runtime_error("Failed to create synchronization primitives");
  }
  for (size_t i = 0; i < swapCount; i++) {
    if (vkCreateSemaphore(dev, &semInfo, nullptr, &renderFinished[i]) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to create renderFinished semaphore");
  }
}
