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

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

class CommandThreadPool {
public:
  explicit CommandThreadPool(uint32_t workerCount) {
    workers.reserve(workerCount);
    for (uint32_t i = 0; i < workerCount; ++i)
      workers.emplace_back([this] { workerLoop(); });
  }

  ~CommandThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
    }
    workCv.notify_all();
    for (std::thread &worker : workers) {
      if (worker.joinable())
        worker.join();
    }
  }

  uint32_t size() const { return static_cast<uint32_t>(workers.size()); }

  void dispatch(uint32_t taskCount,
                const std::function<void(uint32_t)> &callback) {
    if (taskCount == 0)
      return;

    {
      std::lock_guard<std::mutex> lock(mutex);
      firstException = nullptr;
      outstandingTasks += taskCount;
      for (uint32_t i = 0; i < taskCount; ++i) {
        tasks.emplace_back([this, callback, i] {
          try {
            callback(i);
          } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            if (!firstException)
              firstException = std::current_exception();
          }
        });
      }
    }

    workCv.notify_all();
    waitIdle();

    std::exception_ptr failure;
    {
      std::lock_guard<std::mutex> lock(mutex);
      failure = firstException;
      firstException = nullptr;
    }
    if (failure)
      std::rethrow_exception(failure);
  }

private:
  void waitIdle() {
    std::unique_lock<std::mutex> lock(mutex);
    doneCv.wait(lock, [this] {
      return tasks.empty() && outstandingTasks == 0;
    });
  }

  void workerLoop() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex);
        workCv.wait(lock, [this] { return stopping || !tasks.empty(); });
        if (stopping && tasks.empty())
          return;
        task = std::move(tasks.front());
        tasks.pop_front();
      }

      task();

      {
        std::lock_guard<std::mutex> lock(mutex);
        if (outstandingTasks > 0)
          --outstandingTasks;
        if (tasks.empty() && outstandingTasks == 0)
          doneCv.notify_all();
      }
    }
  }

  std::vector<std::thread> workers;
  std::deque<std::function<void()>> tasks;
  mutable std::mutex mutex;
  std::condition_variable workCv;
  std::condition_variable doneCv;
  uint32_t outstandingTasks = 0;
  bool stopping = false;
  std::exception_ptr firstException;
};

namespace {
struct RendererQueueSharingInfo {
  VkSharingMode mode = VK_SHARING_MODE_EXCLUSIVE;
  uint32_t familyCount = 0;
  std::array<uint32_t, 2> families{};
};

RendererQueueSharingInfo rendererGraphicsComputeSharing(
    const VulkanDevice &device) {
  QueueFamilyIndices indices = device.getQueueFamilies();
  RendererQueueSharingInfo sharing{};
  if (indices.hasDedicatedCompute()) {
    sharing.mode = VK_SHARING_MODE_CONCURRENT;
    sharing.families = {static_cast<uint32_t>(indices.graphicsFamily),
                        static_cast<uint32_t>(indices.computeFamily)};
    sharing.familyCount = 2;
  }
  return sharing;
}

void applyRendererQueueSharing(VkImageCreateInfo &ci,
                               const RendererQueueSharingInfo &sharing) {
  ci.sharingMode = sharing.mode;
  if (sharing.mode == VK_SHARING_MODE_CONCURRENT) {
    ci.queueFamilyIndexCount = sharing.familyCount;
    ci.pQueueFamilyIndices = sharing.families.data();
  }
}
} // namespace

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
    bloomFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    ssaoFormat = VK_FORMAT_R16_SFLOAT;
    {
      VkFormatProperties props{};
      vkGetPhysicalDeviceFormatProperties(device.getPhysicalDevice(),
                                          bloomFormat, &props);
      const VkFormatFeatureFlags required =
          VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
          VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
      if ((props.optimalTilingFeatures & required) != required)
        throw std::runtime_error(
            "Bloom requires R16G16B16A16_SFLOAT sampled storage images");
    }
    {
      VkFormatProperties props{};
      vkGetPhysicalDeviceFormatProperties(device.getPhysicalDevice(),
                                          ssaoFormat, &props);
      const VkFormatFeatureFlags required =
          VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
          VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
      if ((props.optimalTilingFeatures & required) != required)
        throw std::runtime_error(
            "Async SSAO requires R16_SFLOAT sampled storage images");
    }

    // 3. Render passes
    renderPassManager.createGBufferRenderPass(
        device.getLogicalDevice(), gb0Fmt, gb1Fmt, gb2Fmt, gBufferDepthFormat);
    renderPassManager.createLitRenderPass(device.getLogicalDevice(), litFormat);
    renderPassManager.createSsgiRenderPass(device.getLogicalDevice(), litFormat);
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
    createBloomResources();

    // 8a. SSGI bounce-buffer images + framebuffers. Sampled by lit.frag,
    //     so the G-buffer descriptor set (set 1) gets a 6th binding
    //     pointing at these views.
    createSsgiResources();
    createSsaoResources();

    // 8b. TAA history images. Format matches litFormat so they live in the
    //     same HDR-precision domain.
    createTaaResources();

    // 8c. Composite framebuffers (3 attachments: swap + colorBuffer + history)
    //     Must be created after TAA history images exist and after the
    //     composition render pass is created.
    createCompositeFramebuffers();

    // 9. G-buffer images, framebuffers, descriptor sets (binds lit views too)
    createGBuffer();

    // 9b. TAA descriptor sets (history-prev + depth + colorBuffer-current
    //     + bloom pyramid result).
    //     Needs taaHistory views (from createTaaResources), gBufferDepth
    //     views (from createGBuffer), colorBuffer views (from swapchain init),
    //     and bloom mip 0 views (from createBloomResources). Order: must
    //     follow all four.
    {
      std::vector<VkImageView> histViews;
      std::vector<VkImageView> depthViews;
      std::vector<VkImageView> colorViews;
      std::vector<VkImageView> bloomViews;
      histViews.reserve(taaHistoryViews.size());
      depthViews.reserve(gBufferDepthViews.size());
      colorViews.reserve(swapchain.getImageCount());
      bloomViews.reserve(swapchain.getImageCount());
      for (const auto &v : taaHistoryViews)
        histViews.push_back(v.get());
      for (const auto &v : gBufferDepthViews)
        depthViews.push_back(v.get());
      for (size_t i = 0; i < swapchain.getImageCount(); ++i)
        colorViews.push_back(swapchain.getColorBufferView(i));
      for (const auto &v : bloomMips[0].views)
        bloomViews.push_back(v.get());
      descriptorManager.recreateTaaSets(device.getLogicalDevice(), histViews,
                                        depthViews, colorViews, bloomViews,
                                        taaSampler);
    }

    // 10. Pipeline (needs all 5 render passes + descriptor layouts)
    pipeline.createPipelines(
        device.getLogicalDevice(), renderPassManager.getGBufferRenderPass(),
        renderPassManager.getLitRenderPass(), renderPassManager.getRenderPass(),
        renderPassManager.getShadowRenderPass(),
        renderPassManager.getSsgiRenderPass(), swapchain.getExtent(),
        descriptorManager);
    createGpuDrivenResources();

    // 10. IBL resources (requires textureManager and descriptorManager)
    initIBL();

    // 10b. Auto-exposure compute resources. Must come after createLitResources()
    //      (binds litViews) and after swapchain init (knows image count).
    createAutoExposureResources();

    // 11. Synchronization
    createSynchronization();
    createAsyncFrameCommandBuffers();
    createThreadedCommandResources();

    // 12. Performance metrics
    QueueFamilyIndices qi = device.getQueueFamilies();
    metrics.init(device.getLogicalDevice(), device.getPhysicalDevice(),
                 static_cast<uint32_t>(qi.graphicsFamily), MAX_FRAMES_DRAWS);

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

    // shadowParams.y/.z/.w piggyback post/SSGI tunables since they were unused.
    // .x stays as the point-shadow far plane; .w toggles SSR.
    sceneUbo.shadowParams = glm::vec4(imguiPointShadowFar, imguiSsgiIntensity,
                                      imguiSharpness,
                                      imguiSsrEnabled ? 1.0f : 0.0f);
    sceneUbo.fogParams =
        glm::vec4(imguiFogDensity, 0.25f, imguiFogClamp,
                  static_cast<float>(std::clamp(imguiSsgiSamples, 4, 12)));
    sceneUbo.lightCounts =
        glm::ivec4(1, 0, 0x3f, 0); // z = lighting-isolation bit mask

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
  int modelId = modelManager.loadModel(modelFile, device, textureManager,
                                       descriptorManager);
  gpuDrivenModelIds.push_back(modelId);
  registerGpuDrivenModelGeometry(modelId);
  return modelId;
}

SceneNode &VulkanRenderer::getRootNode() { return rootNode; }

void VulkanRenderer::updateCameraView(const glm::mat4 &viewMatrix,
                                      const glm::vec3 &cameraPosition) {
  if (taaHasLastCamera) {
    float posDelta = glm::length(cameraPosition - taaLastCameraPos);
    float viewDelta = 0.0f;
    for (int c = 0; c < 4; ++c) {
      for (int r = 0; r < 4; ++r) {
        viewDelta = std::max(
            viewDelta, std::abs(viewMatrix[c][r] - taaLastView[c][r]));
      }
    }
    cameraMovedThisFrame = posDelta > 0.001f || viewDelta > 0.0001f;
  } else {
    cameraMovedThisFrame = true;
    taaHasLastCamera = true;
  }
  taaLastCameraPos = cameraPosition;
  taaLastView = viewMatrix;
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

  using CpuClock = std::chrono::high_resolution_clock;
  auto elapsedMs = [](CpuClock::time_point start) {
    return std::chrono::duration<double, std::milli>(CpuClock::now() - start)
        .count();
  };

  auto phaseStart = CpuClock::now();
  vkWaitForFences(logicalDevice, 1, &drawFences[currentFrame], VK_TRUE,
                  std::numeric_limits<uint64_t>::max());
  metrics.recordCpuPhase(PerformanceMetrics::CpuPhase::WaitFence,
                         elapsedMs(phaseStart));
  // This is the expected frame-in-flight throttle: if the CPU finishes
  // recording faster than the GPU can consume work, it waits here before
  // reusing this frame slot's fence, acquire semaphore, query range, and
  // exposure result buffer. It is idle time, not active CPU renderer cost.
  metrics.collectGpuResults(logicalDevice, currentFrame);

  // Check resize BEFORE acquiring — a successful acquire signals
  // imageAvailable, and returning early without consuming it would leave it
  // signaled on the next frame.
  if (framebufferResized) {
    framebufferResized = false;
    recreateSwapChain();
    return;
  }

  uint32_t imageIndex;
  phaseStart = CpuClock::now();
  VkResult acquireResult = vkAcquireNextImageKHR(
      logicalDevice, swapchain.getSwapchain(),
      std::numeric_limits<uint64_t>::max(), imageAvailable[currentFrame],
      VK_NULL_HANDLE, &imageIndex);
  metrics.recordCpuPhase(PerformanceMetrics::CpuPhase::Acquire,
                         elapsedMs(phaseStart));

  if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
    // Failed acquire → semaphore NOT signaled per spec → safe to recreate
    recreateSwapChain();
    return;
  } else if (acquireResult != VK_SUCCESS &&
             acquireResult != VK_SUBOPTIMAL_KHR) {
    throw std::runtime_error("Failed to acquire swap chain image");
  }

  phaseStart = CpuClock::now();
  VkFence imageFence = imagesInFlight[imageIndex];
  if (imageFence != VK_NULL_HANDLE) {
    bool fenceStillOwnsImage = false;
    for (size_t frame = 0; frame < drawFences.size(); ++frame) {
      if (drawFences[frame] == imageFence) {
        fenceStillOwnsImage =
            frame < frameImageInFlight.size() &&
            frameImageInFlight[frame] == imageIndex;
        break;
      }
    }

    if (fenceStillOwnsImage) {
      vkWaitForFences(logicalDevice, 1, &imageFence, VK_TRUE, UINT64_MAX);
    } else {
      imagesInFlight[imageIndex] = VK_NULL_HANDLE;
    }
  }
  metrics.recordCpuPhase(PerformanceMetrics::CpuPhase::ImageFence,
                         elapsedMs(phaseStart));

  const uint32_t invalidImageIndex = std::numeric_limits<uint32_t>::max();
  if (currentFrame < static_cast<int>(frameImageInFlight.size())) {
    uint32_t previousImage = frameImageInFlight[currentFrame];
    if (previousImage != invalidImageIndex &&
        previousImage < imagesInFlight.size() &&
        imagesInFlight[previousImage] == drawFences[currentFrame]) {
      imagesInFlight[previousImage] = VK_NULL_HANDLE;
    }
    frameImageInFlight[currentFrame] = imageIndex;
  }
  imagesInFlight[imageIndex] = drawFences[currentFrame];
  vkResetFences(logicalDevice, 1, &drawFences[currentFrame]);

  phaseStart = CpuClock::now();
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
  sceneUbo.qualityToggles.x  = imguiIblRoughnessFloor;
  sceneUbo.qualityToggles2.z = imguiMinSurfaceRoughness;
  sceneUbo.qualityToggles.y  = imguiSkyOcclusionFloor;
  sceneUbo.qualityToggles.z  = imguiSpecAAVariance;
  sceneUbo.qualityToggles.w  = imguiSpecAAThreshold;
  int lightingMask = 0;
  if (imguiEnableSunDirect)   lightingMask |= 1 << 0;
  if (imguiEnablePointLights) lightingMask |= 1 << 1;
  if (imguiEnableSpotLights)  lightingMask |= 1 << 2;
  if (imguiEnableIblAmbient)  lightingMask |= 1 << 3;
  if (imguiEnableSsgiBounce)  lightingMask |= 1 << 4;
  if (imguiEnableBloom)       lightingMask |= 1 << 5;
  sceneUbo.lightCounts.z = lightingMask;
  sceneUbo.fogParams.w =
      static_cast<float>(std::clamp(imguiSsgiSamples, 4, 12));
  // shadowParams.y = active SSGI intensity, .z = sharpening strength.
  // If SSGI bounce is disabled in lighting isolation, keep the pass cheap by
  // making ssgi.frag return immediately instead of running the sample loop.
  sceneUbo.shadowParams.y    =
      imguiEnableSsgiBounce ? imguiSsgiIntensity : 0.0f;
  sceneUbo.shadowParams.z    = imguiSharpness;
  sceneUbo.shadowParams.w    = imguiSsrEnabled ? 1.0f : 0.0f;

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
  const bool taaStable = !imguiResponsiveTaa || !cameraMovedThisFrame;
  const bool taaEnable = imguiTaaEnabled && taaStable;
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
                                 (taaHistoryValid && taaStable) ? 1.0f : 0.0f);
  sceneUbo.viewportSize = glm::vec4(float(ext.width), float(ext.height),
                                    1.0f / float(ext.width),
                                    1.0f / float(ext.height));
  // taaHistoryValid is the input to THIS frame's TAA blend (so the first
  // frame after init/resize must read it as false). Promote it to true for
  // the NEXT frame now that we're about to render content into the history.
  taaHistoryValid = taaEnable;

  updateLightSpaceMatrices();
  metrics.recordCpuPhase(PerformanceMetrics::CpuPhase::Update,
                         elapsedMs(phaseStart));

  metrics.setActiveGpuQueryFrame(currentFrame);
  phaseStart = CpuClock::now();
  recordGBufferCommands(imageIndex);
  recordSsaoComputeCommands(ssaoCommandBuffers[imageIndex], imageIndex);
  recordShadowCommands(shadowCommandBuffers[imageIndex]);
  recordPostCommands(postCommandBuffers[imageIndex], imageIndex);
  metrics.recordCpuPhase(PerformanceMetrics::CpuPhase::Record,
                         elapsedMs(phaseStart));

  phaseStart = CpuClock::now();
  descriptorManager.updateUniformBuffer(device.getAllocator(), imageIndex,
                                        &sceneUbo, sizeof(SceneUniformBuffer));
  metrics.recordCpuPhase(PerformanceMetrics::CpuPhase::Upload,
                         elapsedMs(phaseStart));

  phaseStart = CpuClock::now();
  VkSemaphore timeline = asyncComputeTimeline[currentFrame];
  const uint64_t gbufferReadyValue =
      asyncComputeTimelineValue[currentFrame] + 1;
  const uint64_t ssaoReadyValue = asyncComputeTimelineValue[currentFrame] + 2;
  asyncComputeTimelineValue[currentFrame] = ssaoReadyValue;

  VkCommandBuffer gbufferCmd = swapchain.getCommandBuffer(imageIndex);
  VkCommandBuffer ssaoCmd = ssaoCommandBuffers[imageIndex];
  VkCommandBuffer shadowCmd = shadowCommandBuffers[imageIndex];
  VkCommandBuffer postCmd = postCommandBuffers[imageIndex];

  uint64_t zeroValue = 0;
  VkPipelineStageFlags imageWaitStage =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkTimelineSemaphoreSubmitInfo gbufferTimelineInfo = {};
  gbufferTimelineInfo.sType =
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  gbufferTimelineInfo.waitSemaphoreValueCount = 1;
  gbufferTimelineInfo.pWaitSemaphoreValues = &zeroValue; // binary wait
  gbufferTimelineInfo.signalSemaphoreValueCount = 1;
  gbufferTimelineInfo.pSignalSemaphoreValues = &gbufferReadyValue;
  VkSubmitInfo gbufferSubmit = {};
  gbufferSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  gbufferSubmit.pNext = &gbufferTimelineInfo;
  gbufferSubmit.waitSemaphoreCount = 1;
  gbufferSubmit.pWaitSemaphores = &imageAvailable[currentFrame];
  gbufferSubmit.pWaitDstStageMask = &imageWaitStage;
  gbufferSubmit.commandBufferCount = 1;
  gbufferSubmit.pCommandBuffers = &gbufferCmd;
  gbufferSubmit.signalSemaphoreCount = 1;
  gbufferSubmit.pSignalSemaphores = &timeline;
  if (vkQueueSubmit(device.getGraphicsQueue(), 1, &gbufferSubmit,
                    VK_NULL_HANDLE) != VK_SUCCESS) {
    throw std::runtime_error("Failed to submit G-buffer command buffer");
  }

  VkPipelineStageFlags computeWaitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  VkTimelineSemaphoreSubmitInfo ssaoTimelineInfo = {};
  ssaoTimelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  ssaoTimelineInfo.waitSemaphoreValueCount = 1;
  ssaoTimelineInfo.pWaitSemaphoreValues = &gbufferReadyValue;
  ssaoTimelineInfo.signalSemaphoreValueCount = 1;
  ssaoTimelineInfo.pSignalSemaphoreValues = &ssaoReadyValue;
  VkSubmitInfo ssaoSubmit = {};
  ssaoSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  ssaoSubmit.pNext = &ssaoTimelineInfo;
  ssaoSubmit.waitSemaphoreCount = 1;
  ssaoSubmit.pWaitSemaphores = &timeline;
  ssaoSubmit.pWaitDstStageMask = &computeWaitStage;
  ssaoSubmit.commandBufferCount = 1;
  ssaoSubmit.pCommandBuffers = &ssaoCmd;
  ssaoSubmit.signalSemaphoreCount = 1;
  ssaoSubmit.pSignalSemaphores = &timeline;
  if (vkQueueSubmit(device.getComputeQueue(), 1, &ssaoSubmit,
                    VK_NULL_HANDLE) != VK_SUCCESS) {
    throw std::runtime_error("Failed to submit SSAO compute command buffer");
  }

  VkSubmitInfo shadowSubmit = {};
  shadowSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  shadowSubmit.commandBufferCount = 1;
  shadowSubmit.pCommandBuffers = &shadowCmd;
  if (vkQueueSubmit(device.getGraphicsQueue(), 1, &shadowSubmit,
                    VK_NULL_HANDLE) != VK_SUCCESS) {
    throw std::runtime_error("Failed to submit shadow command buffer");
  }

  uint64_t renderFinishedSignalValue = 0; // binary signal, ignored
  VkPipelineStageFlags postWaitStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  VkTimelineSemaphoreSubmitInfo postTimelineInfo = {};
  postTimelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  postTimelineInfo.waitSemaphoreValueCount = 1;
  postTimelineInfo.pWaitSemaphoreValues = &ssaoReadyValue;
  postTimelineInfo.signalSemaphoreValueCount = 1;
  postTimelineInfo.pSignalSemaphoreValues = &renderFinishedSignalValue;
  VkSubmitInfo postSubmit = {};
  postSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  postSubmit.pNext = &postTimelineInfo;
  postSubmit.waitSemaphoreCount = 1;
  postSubmit.pWaitSemaphores = &timeline;
  postSubmit.pWaitDstStageMask = &postWaitStage;
  postSubmit.commandBufferCount = 1;
  postSubmit.pCommandBuffers = &postCmd;
  postSubmit.signalSemaphoreCount = 1;
  // renderFinished is indexed by acquired swap-image, not by frame-in-flight,
  // so the presentation engine's wait binds to the right semaphore.
  postSubmit.pSignalSemaphores = &renderFinished[imageIndex];
  if (vkQueueSubmit(device.getGraphicsQueue(), 1, &postSubmit,
                    drawFences[currentFrame]) != VK_SUCCESS) {
    throw std::runtime_error("Failed to submit post command buffer");
  }
  metrics.recordCpuPhase(PerformanceMetrics::CpuPhase::Submit,
                         elapsedMs(phaseStart));
  metrics.markGpuQueriesSubmitted(currentFrame);

  VkPresentInfoKHR presentInfo = {};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &renderFinished[imageIndex];
  presentInfo.swapchainCount = 1;
  VkSwapchainKHR sc = swapchain.getSwapchain();
  presentInfo.pSwapchains = &sc;
  presentInfo.pImageIndices = &imageIndex;

  phaseStart = CpuClock::now();
  VkResult result =
      vkQueuePresentKHR(device.getPresentationQueue(), &presentInfo);
  metrics.recordCpuPhase(PerformanceMetrics::CpuPhase::Present,
                         elapsedMs(phaseStart));
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
  // Halton jitter index used above matches the history ring slots that
  // recordCommands picked for framebuffers / descriptor sets.
  taaFrameCounter++;
  metrics.endFrame();
}

// Swapchain recreation

void VulkanRenderer::recreateSwapChain() {
  vkDeviceWaitIdle(device.getLogicalDevice());
  cleanupAsyncFrameCommandBuffers();
  swapchain.recreate(device, window);

  cleanupImGuiFramebuffers();
  createImGuiFramebuffers();

  cleanupAutoExposureResources();
  cleanupCompositeFramebuffers();
  cleanupGpuDrivenResources();
  cleanupGBuffer();
  cleanupSsaoResources();
  cleanupBloomResources();
  cleanupSsgiResources();
  cleanupLitResources();
  cleanupTaaResources();
  createLitResources();
  createBloomResources();
  createSsgiResources();
  createSsaoResources();
  createTaaResources();
  createCompositeFramebuffers();
  createGBuffer();
  createGpuDrivenResources();
  // Recreate auto-exposure AFTER lit (descriptors reference litViews).
  createAutoExposureResources();

  imagesInFlight.assign(swapchain.getImageCount(), VK_NULL_HANDLE);
  frameImageInFlight.assign(MAX_FRAMES_DRAWS,
                            std::numeric_limits<uint32_t>::max());

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
    std::vector<VkImageView> histViews;
    std::vector<VkImageView> depthViews;
    std::vector<VkImageView> colorViews;
    std::vector<VkImageView> bloomViews;
    histViews.reserve(taaHistoryViews.size());
    depthViews.reserve(gBufferDepthViews.size());
    colorViews.reserve(swapchain.getImageCount());
    bloomViews.reserve(swapchain.getImageCount());
    for (const auto &v : taaHistoryViews)
      histViews.push_back(v.get());
    for (const auto &v : gBufferDepthViews)
      depthViews.push_back(v.get());
    for (size_t i = 0; i < swapchain.getImageCount(); ++i)
      colorViews.push_back(swapchain.getColorBufferView(i));
    for (const auto &v : bloomMips[0].views)
      bloomViews.push_back(v.get());
    descriptorManager.recreateTaaSets(device.getLogicalDevice(), histViews,
                                      depthViews, colorViews, bloomViews,
                                      taaSampler);
  }

  createAsyncFrameCommandBuffers();
  rebuildProjection();
}

// G-buffer resource management

void VulkanRenderer::createGBuffer() {
  size_t count = swapchain.getImageCount();
  VkDevice dev = device.getLogicalDevice();
  VkFormat gb0Fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
  VkFormat gb1Fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
  VkFormat gb2Fmt = VK_FORMAT_R8G8B8A8_UNORM;
  const RendererQueueSharingInfo queueSharing =
      rendererGraphicsComputeSharing(device);

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
    applyRendererQueueSharing(ci, queueSharing);
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
    applyRendererQueueSharing(ci, queueSharing);
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

  std::vector<VkImageView> gb0v, gb1v, gb2v, depv, litv, ssaov;
  gb0v.reserve(count);
  gb1v.reserve(count);
  gb2v.reserve(count);
  depv.reserve(count);
  litv.reserve(count);
  ssaov.reserve(count);
  for (size_t i = 0; i < count; i++) {
    gb0v.push_back(gBuffer0Views[i].get());
    gb1v.push_back(gBuffer1Views[i].get());
    gb2v.push_back(gBuffer2Views[i].get());
    depv.push_back(gBufferDepthViews[i].get());
    litv.push_back(litViews[i].get());
    ssaov.push_back(ssaoViews[i].get());
  }
  // SSGI views are a temporal history ring. DescriptorManager produces
  // historyCount * swapCount G-buffer sets indexed
  // (historyIndex * swapCount + i); per history index H, binding 5 =
  // ssgiHistoryViews[H].
  std::vector<VkImageView> ssgv;
  ssgv.reserve(ssgiHistoryViews.size());
  for (const auto &view : ssgiHistoryViews)
    ssgv.push_back(view.get());
  descriptorManager.recreateGBufferSets(device.getLogicalDevice(), gb0v, gb1v,
                                        gb2v, depv, litv, ssgv,
                                        ssaov,
                                        textureManager.getTextureSampler(),
                                        ssaoResultSampler);
  {
    std::vector<VkImageView> ssgiHv;
    ssgiHv.reserve(ssgiHistoryViews.size());
    for (const auto &view : ssgiHistoryViews)
      ssgiHv.push_back(view.get());
    descriptorManager.recreateSsgiPrevSets(device.getLogicalDevice(), ssgiHv,
                                           ssgiSampler);
  }
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

// SSGI bounce-buffer resources. This is intentionally half-resolution: the
// pass is a diffuse, cross-bilaterally filtered bounce term, and the full-res
// path-check gather is too expensive for Sponza.
void VulkanRenderer::createSsgiResources() {
  VkDevice dev = device.getLogicalDevice();
  VkExtent2D fullExtent = swapchain.getExtent();
  VkExtent2D ssgiExtent = {std::max(1u, fullExtent.width / 2),
                           std::max(1u, fullExtent.height / 2)};

  ssgiHistoryImages.clear();
  ssgiHistoryViews.clear();
  const size_t historyCount = MAX_FRAMES_DRAWS + 1;
  ssgiFramebuffers.assign(historyCount, VK_NULL_HANDLE);
  ssgiHistoryImages.reserve(historyCount);
  ssgiHistoryViews.reserve(historyCount);

  VmaAllocationCreateInfo aci = {};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  for (size_t i = 0; i < historyCount; ++i) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = {ssgiExtent.width, ssgiExtent.height, 1};
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
      throw std::runtime_error("Failed to create SSGI history image");
    ssgiHistoryImages.emplace_back(device.getAllocator(), rawImg, alloc);

    VkImageViewCreateInfo vci = {};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = ssgiHistoryImages.back().get();
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = litFormat;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView v = VK_NULL_HANDLE;
    if (vkCreateImageView(dev, &vci, nullptr, &v) != VK_SUCCESS)
      throw std::runtime_error("Failed to create SSGI history view");
    ssgiHistoryViews.emplace_back(dev, v);

    VkImageView attachment = v;
    VkFramebufferCreateInfo fbci = {};
    fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbci.renderPass = renderPassManager.getSsgiRenderPass();
    fbci.attachmentCount = 1;
    fbci.pAttachments = &attachment;
    fbci.width = ssgiExtent.width;
    fbci.height = ssgiExtent.height;
    fbci.layers = 1;
    if (vkCreateFramebuffer(dev, &fbci, nullptr, &ssgiFramebuffers[i]) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to create SSGI framebuffer");
  }

  // Transition both history images to SHADER_READ_ONLY_OPTIMAL so the
  // FIRST frame's set-2 sampler read doesn't hit UNDEFINED. The render
  // pass declares initialLayout=UNDEFINED + loadOp=CLEAR for the SSGI
  // attachment, so the write side handles its own transition each frame.
  VkCommandBuffer cmd = beginCommandBuffer(dev, device.getGraphicsCommandPool());
  for (size_t i = 0; i < historyCount; ++i) {
    VkImageMemoryBarrier b = {};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = ssgiHistoryImages[i].get();
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.srcAccessMask = 0;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &b);
  }
  endAndSubmitCommandBuffer(dev, device.getGraphicsCommandPool(),
                            device.getGraphicsQueue(), cmd);

  if (ssgiSampler == VK_NULL_HANDLE) {
    VkSamplerCreateInfo sci = {};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = 0.0f;
    if (vkCreateSampler(dev, &sci, nullptr, &ssgiSampler) != VK_SUCCESS)
      throw std::runtime_error("Failed to create SSGI sampler");
  }
}

void VulkanRenderer::cleanupSsgiResources() {
  VkDevice dev = device.getLogicalDevice();
  for (VkFramebuffer fb : ssgiFramebuffers)
    if (fb != VK_NULL_HANDLE)
      vkDestroyFramebuffer(dev, fb, nullptr);
  ssgiFramebuffers.clear();
  ssgiHistoryViews.clear();
  ssgiHistoryImages.clear();
  if (ssgiSampler != VK_NULL_HANDLE) {
    vkDestroySampler(dev, ssgiSampler, nullptr);
    ssgiSampler = VK_NULL_HANDLE;
  }
}

// TAA history images. The ring is one larger than the max frames-in-flight,
// so a frame can sample the previous logical frame without another in-flight
// frame overwriting that image too early. Format matches litFormat so reads
// from history feed straight into the same HDR-precision pipeline. Usage:
// SAMPLED (read as history-prev) + COLOR_ATTACHMENT (written as history-curr).
void VulkanRenderer::createTaaResources() {
  VkDevice dev = device.getLogicalDevice();
  taaHistoryImages.clear();
  taaHistoryViews.clear();
  const size_t historyCount = MAX_FRAMES_DRAWS + 1;
  taaHistoryImages.reserve(historyCount);
  taaHistoryViews.reserve(historyCount);

  VmaAllocationCreateInfo aci = {};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  for (size_t i = 0; i < historyCount; ++i) {
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
  for (size_t i = 0; i < historyCount; ++i) {
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
  // Shared with SSGI: ssgi.frag reads scene.taaParams.w (==taaHistoryValid),
  // so dropping this also drops the SSGI temporal blend on the post-resize
  // frame. createSsgiResources runs alongside createTaaResources on every
  // swapchain recreate; both temporal history rings invalidate together.
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

// Composite framebuffers: historyCount * swapCount entries, indexed by
// (historyIndex * swapCount + swapIdx). Each binds:
//   attachment 0 = swap image view (per swapIdx)
//   attachment 1 = colorBuffer view (per swapIdx)
//   attachment 2 = TAA history view (per history ring slot)
void VulkanRenderer::createCompositeFramebuffers() {
  VkDevice dev = device.getLogicalDevice();
  size_t swapCount = swapchain.getImageCount();
  size_t historyCount = taaHistoryViews.size();
  compositeFramebuffers.assign(historyCount * swapCount, VK_NULL_HANDLE);

  for (size_t historyIndex = 0; historyIndex < historyCount; ++historyIndex) {
    for (size_t i = 0; i < swapCount; ++i) {
      std::array<VkImageView, 3> atts = {swapchain.getSwapImageView(i),
                                         swapchain.getColorBufferView(i),
                                         taaHistoryViews[historyIndex].get()};
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
      compositeFramebuffers[historyIndex * swapCount + i] = fb;
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

void VulkanRenderer::createSsaoResources() {
  cleanupSsaoResources();

  VkDevice dev = device.getLogicalDevice();
  const size_t swapCount = swapchain.getImageCount();
  const VkExtent2D extent = swapchain.getExtent();
  const RendererQueueSharingInfo queueSharing =
      rendererGraphicsComputeSharing(device);

  ssaoImages.clear();
  ssaoViews.clear();
  ssaoImages.reserve(swapCount);
  ssaoViews.reserve(swapCount);

  VmaAllocationCreateInfo aci = {};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  for (size_t i = 0; i < swapCount; ++i) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = {extent.width, extent.height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.format = ssaoFormat;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    applyRendererQueueSharing(ci, queueSharing);

    VkImage rawImg = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    if (vmaCreateImage(device.getAllocator(), &ci, &aci, &rawImg, &alloc,
                       nullptr) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create SSAO image");
    }
    ssaoImages.emplace_back(device.getAllocator(), rawImg, alloc);

    VkImageViewCreateInfo vci = {};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = ssaoImages.back().get();
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = ssaoFormat;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view = VK_NULL_HANDLE;
    if (vkCreateImageView(dev, &vci, nullptr, &view) != VK_SUCCESS)
      throw std::runtime_error("Failed to create SSAO image view");
    ssaoViews.emplace_back(dev, view);
  }

  if (ssaoResultSampler == VK_NULL_HANDLE) {
    VkSamplerCreateInfo sci = {};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxAnisotropy = 1.0f;
    if (vkCreateSampler(dev, &sci, nullptr, &ssaoResultSampler) != VK_SUCCESS)
      throw std::runtime_error("Failed to create SSAO sampler");
  }

  VkDescriptorSetLayoutBinding outBinding = {};
  outBinding.binding = 0;
  outBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  outBinding.descriptorCount = 1;
  outBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo dslCI = {};
  dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslCI.bindingCount = 1;
  dslCI.pBindings = &outBinding;
  if (vkCreateDescriptorSetLayout(dev, &dslCI, nullptr,
                                  &ssaoOutputSetLayout) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create SSAO output descriptor layout");
  }

  VkDescriptorPoolSize poolSize = {};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  poolSize.descriptorCount = static_cast<uint32_t>(swapCount);
  VkDescriptorPoolCreateInfo poolCI = {};
  poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolCI.maxSets = static_cast<uint32_t>(swapCount);
  poolCI.poolSizeCount = 1;
  poolCI.pPoolSizes = &poolSize;
  if (vkCreateDescriptorPool(dev, &poolCI, nullptr, &ssaoDescriptorPool) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create SSAO descriptor pool");
  }

  std::vector<VkDescriptorSetLayout> layouts(swapCount, ssaoOutputSetLayout);
  ssaoOutputSets.assign(swapCount, VK_NULL_HANDLE);
  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = ssaoDescriptorPool;
  allocInfo.descriptorSetCount = static_cast<uint32_t>(swapCount);
  allocInfo.pSetLayouts = layouts.data();
  if (vkAllocateDescriptorSets(dev, &allocInfo, ssaoOutputSets.data()) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate SSAO descriptor sets");
  }

  for (size_t i = 0; i < swapCount; ++i) {
    VkDescriptorImageInfo imageInfo = {};
    imageInfo.imageView = ssaoViews[i].get();
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = ssaoOutputSets[i];
    write.dstBinding = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.descriptorCount = 1;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);
  }

  std::array<VkDescriptorSetLayout, 3> setLayouts = {
      descriptorManager.getVPLayout(), descriptorManager.getGBufferLayout(),
      ssaoOutputSetLayout};
  VkPipelineLayoutCreateInfo plCI = {};
  plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plCI.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
  plCI.pSetLayouts = setLayouts.data();
  if (vkCreatePipelineLayout(dev, &plCI, nullptr, &ssaoPipelineLayout) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create SSAO pipeline layout");
  }

  VkShaderModule shader = loadComputeSpv(dev, "Shaders/ssao.comp.spv");
  VkPipelineShaderStageCreateInfo stageCI = {};
  stageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stageCI.module = shader;
  stageCI.pName = "main";
  VkComputePipelineCreateInfo pipelineCI = {};
  pipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineCI.stage = stageCI;
  pipelineCI.layout = ssaoPipelineLayout;
  if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pipelineCI, nullptr,
                               &ssaoPipeline) != VK_SUCCESS) {
    vkDestroyShaderModule(dev, shader, nullptr);
    throw std::runtime_error("Failed to create SSAO compute pipeline");
  }
  vkDestroyShaderModule(dev, shader, nullptr);

  VkCommandBuffer cmd = beginCommandBuffer(dev, device.getComputeCommandPool());
  std::vector<VkImageMemoryBarrier> barriers;
  barriers.reserve(ssaoImages.size());
  for (const AllocatedImage &img : ssaoImages) {
    VkImageMemoryBarrier b = {};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcAccessMask = 0;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img.get();
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barriers.push_back(b);
  }
  if (!barriers.empty()) {
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, static_cast<uint32_t>(barriers.size()),
                         barriers.data());
  }
  endAndSubmitCommandBuffer(dev, device.getComputeCommandPool(),
                            device.getComputeQueue(), cmd);
}

void VulkanRenderer::cleanupSsaoResources() {
  VkDevice dev = device.getLogicalDevice();
  if (dev == VK_NULL_HANDLE)
    return;

  if (ssaoPipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(dev, ssaoPipeline, nullptr);
    ssaoPipeline = VK_NULL_HANDLE;
  }
  if (ssaoPipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(dev, ssaoPipelineLayout, nullptr);
    ssaoPipelineLayout = VK_NULL_HANDLE;
  }
  if (ssaoDescriptorPool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(dev, ssaoDescriptorPool, nullptr);
    ssaoDescriptorPool = VK_NULL_HANDLE;
  }
  ssaoOutputSets.clear();
  if (ssaoOutputSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(dev, ssaoOutputSetLayout, nullptr);
    ssaoOutputSetLayout = VK_NULL_HANDLE;
  }
  ssaoViews.clear();
  ssaoImages.clear();
  if (ssaoResultSampler != VK_NULL_HANDLE) {
    vkDestroySampler(dev, ssaoResultSampler, nullptr);
    ssaoResultSampler = VK_NULL_HANDLE;
  }
}

void VulkanRenderer::createBloomResources() {
  VkDevice dev = device.getLogicalDevice();
  const size_t swapCount = swapchain.getImageCount();
  VkExtent2D extent = swapchain.getExtent();

  for (uint32_t level = 0; level < BLOOM_MIP_COUNT; ++level) {
    bloomMips[level].extent = extent;
    bloomMips[level].images.clear();
    bloomMips[level].views.clear();
    bloomMips[level].images.reserve(swapCount);
    bloomMips[level].views.reserve(swapCount);

    VmaAllocationCreateInfo aci = {};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    for (size_t i = 0; i < swapCount; ++i) {
      VkImageCreateInfo ci = {};
      ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      ci.imageType = VK_IMAGE_TYPE_2D;
      ci.extent = {extent.width, extent.height, 1};
      ci.mipLevels = 1;
      ci.arrayLayers = 1;
      ci.format = bloomFormat;
      ci.tiling = VK_IMAGE_TILING_OPTIMAL;
      ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
      ci.samples = VK_SAMPLE_COUNT_1_BIT;
      ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

      VkImage rawImg = VK_NULL_HANDLE;
      VmaAllocation alloc = VK_NULL_HANDLE;
      if (vmaCreateImage(device.getAllocator(), &ci, &aci, &rawImg, &alloc,
                         nullptr) != VK_SUCCESS)
        throw std::runtime_error("Failed to create bloom image");
      bloomMips[level].images.emplace_back(device.getAllocator(), rawImg,
                                           alloc);

      VkImageViewCreateInfo vci = {};
      vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      vci.image = bloomMips[level].images.back().get();
      vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
      vci.format = bloomFormat;
      vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      VkImageView view = VK_NULL_HANDLE;
      if (vkCreateImageView(dev, &vci, nullptr, &view) != VK_SUCCESS)
        throw std::runtime_error("Failed to create bloom image view");
      bloomMips[level].views.emplace_back(dev, view);
    }

    extent.width = std::max(1u, extent.width / 2);
    extent.height = std::max(1u, extent.height / 2);
  }

  // Prime every bloom image into shader-read layout. recordBloomPass flips the
  // current swap image to GENERAL while compute writes, then back to shader-read
  // for the tonemap/TAA subpass.
  {
    VkCommandBuffer cmd =
        beginCommandBuffer(dev, device.getGraphicsCommandPool());
    for (uint32_t level = 0; level < BLOOM_MIP_COUNT; ++level) {
      for (size_t i = 0; i < swapCount; ++i) {
        VkImageMemoryBarrier b = {};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = bloomMips[level].images[i].get();
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &b);
      }
    }
    endAndSubmitCommandBuffer(dev, device.getGraphicsCommandPool(),
                              device.getGraphicsQueue(), cmd);
  }

  if (bloomSampler == VK_NULL_HANDLE) {
    VkSamplerCreateInfo sci = {};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxAnisotropy = 1.0f;
    sci.minLod = 0.0f;
    sci.maxLod = 0.0f;
    if (vkCreateSampler(dev, &sci, nullptr, &bloomSampler) != VK_SUCCESS)
      throw std::runtime_error("Failed to create bloom sampler");
  }

  {
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ci.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(dev, &ci, nullptr, &bloomSetLayout) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to create bloom descriptor layout");
  }

  struct BloomDownsamplePC {
    float srcTexelSize[2];
    float threshold;
    float knee;
    int prefilter;
  };
  struct BloomUpsamplePC {
    float srcTexelSize[2];
    float radius;
    float intensity;
    int finalPass;
  };

  auto createBloomPipelineLayout = [&](uint32_t pcSize) {
    VkPushConstantRange pc = {};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = pcSize;
    VkPipelineLayoutCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ci.setLayoutCount = 1;
    ci.pSetLayouts = &bloomSetLayout;
    ci.pushConstantRangeCount = 1;
    ci.pPushConstantRanges = &pc;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(dev, &ci, nullptr, &layout) != VK_SUCCESS)
      throw std::runtime_error("Failed to create bloom pipeline layout");
    return layout;
  };
  bloomDownsamplePipelineLayout =
      createBloomPipelineLayout(sizeof(BloomDownsamplePC));
  bloomUpsamplePipelineLayout =
      createBloomPipelineLayout(sizeof(BloomUpsamplePC));

  const uint32_t downCount =
      static_cast<uint32_t>(swapCount * BLOOM_MIP_COUNT);
  const uint32_t upCount =
      static_cast<uint32_t>(swapCount * (BLOOM_MIP_COUNT - 1));
  {
    std::array<VkDescriptorPoolSize, 2> sizes{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = downCount + upCount;
    sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[1].descriptorCount = downCount + upCount;
    VkDescriptorPoolCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.maxSets = downCount + upCount;
    ci.poolSizeCount = static_cast<uint32_t>(sizes.size());
    ci.pPoolSizes = sizes.data();
    if (vkCreateDescriptorPool(dev, &ci, nullptr, &bloomDescriptorPool) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to create bloom descriptor pool");
  }

  auto allocateBloomSets = [&](std::vector<VkDescriptorSet> &sets,
                               uint32_t count) {
    std::vector<VkDescriptorSetLayout> layouts(count, bloomSetLayout);
    sets.assign(count, VK_NULL_HANDLE);
    VkDescriptorSetAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = bloomDescriptorPool;
    ai.descriptorSetCount = count;
    ai.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(dev, &ai, sets.data()) != VK_SUCCESS)
      throw std::runtime_error("Failed to allocate bloom descriptor sets");
  };
  allocateBloomSets(bloomDownsampleSets, downCount);
  allocateBloomSets(bloomUpsampleSets, upCount);

  for (size_t swapIdx = 0; swapIdx < swapCount; ++swapIdx) {
    for (uint32_t level = 0; level < BLOOM_MIP_COUNT; ++level) {
      VkDescriptorImageInfo src = {};
      src.sampler = bloomSampler;
      src.imageView =
          (level == 0) ? litViews[swapIdx].get()
                       : bloomMips[level - 1].views[swapIdx].get();
      src.imageLayout = (level == 0) ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                     : VK_IMAGE_LAYOUT_GENERAL;

      VkDescriptorImageInfo dst = {};
      dst.imageView = bloomMips[level].views[swapIdx].get();
      dst.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

      VkDescriptorSet set =
          bloomDownsampleSets[swapIdx * BLOOM_MIP_COUNT + level];
      std::array<VkWriteDescriptorSet, 2> writes{};
      writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[0].dstSet = set;
      writes[0].dstBinding = 0;
      writes[0].descriptorCount = 1;
      writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      writes[0].pImageInfo = &src;
      writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[1].dstSet = set;
      writes[1].dstBinding = 1;
      writes[1].descriptorCount = 1;
      writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      writes[1].pImageInfo = &dst;
      vkUpdateDescriptorSets(dev, static_cast<uint32_t>(writes.size()),
                             writes.data(), 0, nullptr);
    }

    for (uint32_t level = 1; level < BLOOM_MIP_COUNT; ++level) {
      VkDescriptorImageInfo src = {};
      src.sampler = bloomSampler;
      src.imageView = bloomMips[level].views[swapIdx].get();
      src.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

      VkDescriptorImageInfo dst = {};
      dst.imageView = bloomMips[level - 1].views[swapIdx].get();
      dst.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

      VkDescriptorSet set =
          bloomUpsampleSets[swapIdx * (BLOOM_MIP_COUNT - 1) + (level - 1)];
      std::array<VkWriteDescriptorSet, 2> writes{};
      writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[0].dstSet = set;
      writes[0].dstBinding = 0;
      writes[0].descriptorCount = 1;
      writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      writes[0].pImageInfo = &src;
      writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[1].dstSet = set;
      writes[1].dstBinding = 1;
      writes[1].descriptorCount = 1;
      writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      writes[1].pImageInfo = &dst;
      vkUpdateDescriptorSets(dev, static_cast<uint32_t>(writes.size()),
                             writes.data(), 0, nullptr);
    }
  }

  auto createComputePipeline = [&](const char *path, VkPipelineLayout layout) {
    VkShaderModule mod = loadComputeSpv(dev, path);
    VkPipelineShaderStageCreateInfo stage = {};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = mod;
    stage.pName = "main";
    VkComputePipelineCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage = stage;
    ci.layout = layout;
    VkPipeline pipe = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr,
                                 &pipe) != VK_SUCCESS)
      throw std::runtime_error("Failed to create bloom compute pipeline");
    vkDestroyShaderModule(dev, mod, nullptr);
    return pipe;
  };
  bloomDownsamplePipeline =
      createComputePipeline("Shaders/bloom_downsample.comp.spv",
                            bloomDownsamplePipelineLayout);
  bloomUpsamplePipeline =
      createComputePipeline("Shaders/bloom_upsample.comp.spv",
                            bloomUpsamplePipelineLayout);
}

void VulkanRenderer::cleanupBloomResources() {
  VkDevice dev = device.getLogicalDevice();
  if (bloomDownsamplePipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(dev, bloomDownsamplePipeline, nullptr);
    bloomDownsamplePipeline = VK_NULL_HANDLE;
  }
  if (bloomUpsamplePipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(dev, bloomUpsamplePipeline, nullptr);
    bloomUpsamplePipeline = VK_NULL_HANDLE;
  }
  if (bloomDownsamplePipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(dev, bloomDownsamplePipelineLayout, nullptr);
    bloomDownsamplePipelineLayout = VK_NULL_HANDLE;
  }
  if (bloomUpsamplePipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(dev, bloomUpsamplePipelineLayout, nullptr);
    bloomUpsamplePipelineLayout = VK_NULL_HANDLE;
  }
  if (bloomDescriptorPool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(dev, bloomDescriptorPool, nullptr);
    bloomDescriptorPool = VK_NULL_HANDLE;
  }
  bloomDownsampleSets.clear();
  bloomUpsampleSets.clear();
  if (bloomSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(dev, bloomSetLayout, nullptr);
    bloomSetLayout = VK_NULL_HANDLE;
  }
  for (auto &mip : bloomMips) {
    mip.views.clear();
    mip.images.clear();
    mip.extent = {};
  }
  if (bloomSampler != VK_NULL_HANDLE) {
    vkDestroySampler(dev, bloomSampler, nullptr);
    bloomSampler = VK_NULL_HANDLE;
  }
}

void VulkanRenderer::recordBloomPass(VkCommandBuffer cmd,
                                     uint32_t currentImage) {
  if ((!imguiEnableBloom && imguiDebugMode != 13) ||
      bloomDownsamplePipeline == VK_NULL_HANDLE ||
      bloomUpsamplePipeline == VK_NULL_HANDLE)
    return;

  vkdbgBeginLabel(cmd, "Bloom Pyramid (downsample + upsample)", 1.0f, 0.55f,
                  0.2f);

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

  std::array<VkImageMemoryBarrier, BLOOM_MIP_COUNT> toGeneral{};
  for (uint32_t level = 0; level < BLOOM_MIP_COUNT; ++level) {
    auto &b = toGeneral[level];
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = bloomMips[level].images[currentImage].get();
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  }
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, static_cast<uint32_t>(toGeneral.size()),
                       toGeneral.data());

  auto barrierBloomMip = [&](uint32_t level) {
    VkImageMemoryBarrier b = {};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = bloomMips[level].images[currentImage].get();
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &b);
  };

  struct BloomDownsamplePC {
    float srcTexelSize[2];
    float threshold;
    float knee;
    int prefilter;
  };
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    bloomDownsamplePipeline);
  for (uint32_t level = 0; level < BLOOM_MIP_COUNT; ++level) {
    VkExtent2D srcExtent =
        (level == 0) ? swapchain.getExtent() : bloomMips[level - 1].extent;
    VkExtent2D dstExtent = bloomMips[level].extent;
    BloomDownsamplePC pc = {
        {1.0f / static_cast<float>(srcExtent.width),
         1.0f / static_cast<float>(srcExtent.height)},
        imguiBloomThreshold, imguiBloomThreshold * 0.5f,
        level == 0 ? 1 : 0};
    VkDescriptorSet set =
        bloomDownsampleSets[currentImage * BLOOM_MIP_COUNT + level];
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            bloomDownsamplePipelineLayout, 0, 1, &set, 0,
                            nullptr);
    vkCmdPushConstants(cmd, bloomDownsamplePipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (dstExtent.width + 7) / 8,
                  (dstExtent.height + 7) / 8, 1);
    barrierBloomMip(level);
  }

  struct BloomUpsamplePC {
    float srcTexelSize[2];
    float radius;
    float intensity;
    int finalPass;
  };
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    bloomUpsamplePipeline);
  for (int level = static_cast<int>(BLOOM_MIP_COUNT) - 1; level >= 1;
       --level) {
    VkExtent2D srcExtent = bloomMips[level].extent;
    VkExtent2D dstExtent = bloomMips[level - 1].extent;
    BloomUpsamplePC pc = {
        {1.0f / static_cast<float>(srcExtent.width),
         1.0f / static_cast<float>(srcExtent.height)},
        imguiBloomRadius, imguiBloomIntensity, level == 1 ? 1 : 0};
    VkDescriptorSet set =
        bloomUpsampleSets[currentImage * (BLOOM_MIP_COUNT - 1) +
                          (static_cast<uint32_t>(level) - 1)];
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            bloomUpsamplePipelineLayout, 0, 1, &set, 0,
                            nullptr);
    vkCmdPushConstants(cmd, bloomUpsamplePipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (dstExtent.width + 7) / 8,
                  (dstExtent.height + 7) / 8, 1);
    barrierBloomMip(static_cast<uint32_t>(level - 1));
  }

  std::array<VkImageMemoryBarrier, BLOOM_MIP_COUNT> toRead{};
  for (uint32_t level = 0; level < BLOOM_MIP_COUNT; ++level) {
    auto &b = toRead[level];
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = bloomMips[level].images[currentImage].get();
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  }
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, static_cast<uint32_t>(toRead.size()),
                       toRead.data());

  vkdbgEndLabel(cmd);
}

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
  cleanupGpuDrivenResources();
  cleanupGBuffer();
  cleanupSsaoResources();
  cleanupBloomResources();
  cleanupSsgiResources();
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
  for (VkSemaphore s : asyncComputeTimeline)
    vkDestroySemaphore(device.getLogicalDevice(), s, nullptr);
  asyncComputeTimeline.clear();
  asyncComputeTimelineValue.clear();
  cleanupAsyncFrameCommandBuffers();
  cleanupThreadedCommandResources();

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

static const char *presentModeName(VkPresentModeKHR mode) {
  switch (mode) {
  case VK_PRESENT_MODE_IMMEDIATE_KHR:
    return "IMMEDIATE";
  case VK_PRESENT_MODE_MAILBOX_KHR:
    return "MAILBOX";
  case VK_PRESENT_MODE_FIFO_KHR:
    return "FIFO";
  case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
    return "FIFO_RELAXED";
  default:
    return "UNKNOWN";
  }
}

static VkImageAspectFlags depthStencilAspectMask(VkFormat format) {
  switch (format) {
  case VK_FORMAT_D16_UNORM_S8_UINT:
  case VK_FORMAT_D24_UNORM_S8_UINT:
  case VK_FORMAT_D32_SFLOAT_S8_UINT:
    return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
  default:
    return VK_IMAGE_ASPECT_DEPTH_BIT;
  }
}

enum class GBufferDrawKind {
  Mesh,
  Instanced,
};

struct GBufferDrawItem {
  GBufferDrawKind kind = GBufferDrawKind::Mesh;
  const Mesh *mesh = nullptr;
  int lod = 0;
  ModelPushConstants push{};
  glm::vec3 aabbMin = glm::vec3(0.0f);
  glm::vec3 aabbMax = glm::vec3(0.0f);
  VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
  VkBuffer instanceBuffer = VK_NULL_HANDLE;
  const std::vector<InstanceData> *instances = nullptr;
  uint32_t instanceCount = 1;
  uint32_t indexCount = 0;
  uint32_t materialId = 0;
  uint32_t transformId = 0;
};

struct GpuDrivenMeshRecord {
  glm::vec4 aabbMin;            // xyz = world AABB min
  glm::vec4 aabbMax;            // xyz = world AABB max
  glm::uvec4 draw;              // x=firstIndex, y=indexCount, z=vertexOffset, w=baseInstance
  glm::uvec4 flags;             // x=instanceCount, y=LOD, z=kind, w=cull
};

static_assert(sizeof(GpuDrivenMeshRecord) == 64,
              "GPU-driven mesh records must stay std430-friendly");

struct GpuDrivenTransformRecord {
  glm::mat4 model;
  glm::mat4 normal;
  glm::uvec4 texIdx0;
  glm::uvec4 texIdx1;
};

struct GpuDrivenFrustumData {
  glm::vec4 planes[6];
  glm::mat4 viewProj = glm::mat4(1.0f);
  // x/y = HZB base extent, z = mip count, w = enable flag.
  glm::vec4 hzbParams = glm::vec4(0.0f);
  uint32_t meshCount = 0;
  uint32_t _pad[3] = {};
};

static glm::uvec4 materialTextureIndices0(const Material &mat) {
  return glm::uvec4(static_cast<uint32_t>(mat.albedoTextureId),
                    static_cast<uint32_t>(mat.normalTextureId),
                    static_cast<uint32_t>(mat.metallicTextureId),
                    static_cast<uint32_t>(mat.roughnessTextureId));
}

static glm::uvec4 materialTextureIndices1(const Material &mat) {
  const uint32_t materialFlags =
      (mat.isCloth ? 1u : 0u) | (mat.alphaMasked ? 2u : 0u);
  const uint32_t alphaCutoff255 = static_cast<uint32_t>(
      glm::clamp(mat.alphaCutoff, 0.0f, 1.0f) * 255.0f + 0.5f);
  return glm::uvec4(static_cast<uint32_t>(mat.aoTextureId), materialFlags,
                    alphaCutoff255, 0u);
}

static VkShaderModule createLocalShaderModule(VkDevice device,
                                              const std::vector<char> &code) {
  VkShaderModuleCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  ci.codeSize = code.size();
  ci.pCode = reinterpret_cast<const uint32_t *>(code.data());
  VkShaderModule module = VK_NULL_HANDLE;
  if (vkCreateShaderModule(device, &ci, nullptr, &module) != VK_SUCCESS)
    throw std::runtime_error("Failed to create shader module");
  return module;
}

bool VulkanRenderer::ensureGpuDrivenBuffer(AllocatedBuffer &buffer,
                                           VkDeviceSize &capacity,
                                           VkDeviceSize requiredSize,
                                           VkBufferUsageFlags usage) {
  if (requiredSize == 0)
    requiredSize = 4;
  if (buffer && capacity >= requiredSize)
    return false;

  buffer.reset();
  createBuffer(device.getAllocator(), requiredSize, usage, VMA_MEMORY_USAGE_AUTO,
               VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, &buffer);
  capacity = requiredSize;
  return true;
}

const VulkanRenderer::GpuDrivenGeometryRange *
VulkanRenderer::findGpuDrivenGeometry(const Mesh *mesh, int lod) const {
  for (const GpuDrivenGeometryRange &range : gpuDrivenGeometryRanges) {
    if (range.mesh == mesh && range.lod == lod)
      return &range;
  }
  return nullptr;
}

void VulkanRenderer::uploadGpuDrivenStaticGeometry() {
  if (gpuDrivenStaticVertices.empty() || gpuDrivenStaticIndices.empty())
    return;

  const VkDeviceSize vertexBytes = static_cast<VkDeviceSize>(
      gpuDrivenStaticVertices.size() * sizeof(Vertex));
  const VkDeviceSize indexBytes = static_cast<VkDeviceSize>(
      gpuDrivenStaticIndices.size() * sizeof(uint32_t));

  ensureGpuDrivenBuffer(gpuDrivenStaticVertexBuffer,
                        gpuDrivenStaticVertexBufferSize, vertexBytes,
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  ensureGpuDrivenBuffer(gpuDrivenStaticIndexBuffer,
                        gpuDrivenStaticIndexBufferSize, indexBytes,
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

  void *mapped = nullptr;
  vmaMapMemory(device.getAllocator(),
               gpuDrivenStaticVertexBuffer.getAllocation(), &mapped);
  memcpy(mapped, gpuDrivenStaticVertices.data(), static_cast<size_t>(vertexBytes));
  vmaUnmapMemory(device.getAllocator(),
                 gpuDrivenStaticVertexBuffer.getAllocation());

  vmaMapMemory(device.getAllocator(),
               gpuDrivenStaticIndexBuffer.getAllocation(), &mapped);
  memcpy(mapped, gpuDrivenStaticIndices.data(), static_cast<size_t>(indexBytes));
  vmaUnmapMemory(device.getAllocator(),
                 gpuDrivenStaticIndexBuffer.getAllocation());
}

void VulkanRenderer::registerGpuDrivenModelGeometry(int modelId) {
  MeshModel *model = modelManager.getModel(modelId);
  if (!model)
    return;

  bool changed = false;
  for (size_t meshIndex = 0; meshIndex < model->getMeshCount(); ++meshIndex) {
    const Mesh *mesh = model->getMesh(meshIndex);
    if (!mesh)
      continue;

    int32_t vertexOffset = -1;
    for (const GpuDrivenGeometryRange &range : gpuDrivenGeometryRanges) {
      if (range.mesh == mesh) {
        vertexOffset = range.vertexOffset;
        break;
      }
    }
    if (vertexOffset < 0) {
      const std::vector<Vertex> &vertices = mesh->getCpuVertices();
      if (vertices.empty())
        continue;
      if (gpuDrivenStaticVertices.size() + vertices.size() >
          static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("GPU-driven vertex arena exceeded int32 range");
      }
      vertexOffset = static_cast<int32_t>(gpuDrivenStaticVertices.size());
      gpuDrivenStaticVertices.insert(gpuDrivenStaticVertices.end(),
                                     vertices.begin(), vertices.end());
      changed = true;
    }

    for (int lod = 0; lod < mesh->getLodCount(); ++lod) {
      if (findGpuDrivenGeometry(mesh, lod))
        continue;
      const std::vector<uint32_t> &indices = mesh->getCpuIndices(lod);
      if (indices.empty())
        continue;
      if (gpuDrivenStaticIndices.size() + indices.size() >
          static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("GPU-driven index arena exceeded uint32 range");
      }

      GpuDrivenGeometryRange range{};
      range.mesh = mesh;
      range.lod = lod;
      range.firstIndex = static_cast<uint32_t>(gpuDrivenStaticIndices.size());
      range.indexCount = static_cast<uint32_t>(indices.size());
      range.vertexOffset = vertexOffset;
      gpuDrivenStaticIndices.insert(gpuDrivenStaticIndices.end(),
                                    indices.begin(), indices.end());
      gpuDrivenGeometryRanges.push_back(range);
      changed = true;
    }
  }

  if (!changed)
    return;

  if (gpuDrivenStaticVertexBuffer || gpuDrivenStaticIndexBuffer)
    vkDeviceWaitIdle(device.getLogicalDevice());
  uploadGpuDrivenStaticGeometry();
}

void VulkanRenderer::uploadGpuDrivenMeshRecords(GpuDrivenFrameResources &frame,
                                                const void *records,
                                                VkDeviceSize bytes,
                                                uint32_t recordCount) {
  gpuDrivenMeshCount = recordCount;
  if (bytes == 0 || records == nullptr)
    return;

  if (ensureGpuDrivenBuffer(frame.meshBuffer, frame.meshBufferSize, bytes,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
    frame.descriptorDirty = true;
  }

  void *mapped = nullptr;
  vmaMapMemory(device.getAllocator(), frame.meshBuffer.getAllocation(), &mapped);
  memcpy(mapped, records, static_cast<size_t>(bytes));
  vmaUnmapMemory(device.getAllocator(), frame.meshBuffer.getAllocation());
}

void VulkanRenderer::createGpuDrivenResources() {
  VkDevice dev = device.getLogicalDevice();
  const size_t swapCount = swapchain.getImageCount();

  gpuDrivenHzbFormat = VK_FORMAT_R32_SFLOAT;
  {
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(device.getPhysicalDevice(),
                                        gpuDrivenHzbFormat, &props);
    const VkFormatFeatureFlags required =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
        VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
    if ((props.optimalTilingFeatures & required) != required)
      throw std::runtime_error("HZB requires R32_SFLOAT sampled storage images");
  }

  std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
  for (uint32_t i = 0; i < 5; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorCount = 1;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
                             VK_SHADER_STAGE_COMPUTE_BIT;
  }
  bindings[5].binding = 5;
  bindings[5].descriptorCount = 1;
  bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo layoutCI{};
  layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutCI.bindingCount = static_cast<uint32_t>(bindings.size());
  layoutCI.pBindings = bindings.data();
  if (vkCreateDescriptorSetLayout(dev, &layoutCI, nullptr,
                                  &gpuDrivenSetLayout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create GPU-driven descriptor layout");

  std::array<VkDescriptorSetLayoutBinding, 2> hzbBindings{};
  hzbBindings[0].binding = 0;
  hzbBindings[0].descriptorCount = 1;
  hzbBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  hzbBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  hzbBindings[1].binding = 1;
  hzbBindings[1].descriptorCount = 1;
  hzbBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  hzbBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  VkDescriptorSetLayoutCreateInfo hzbLayoutCI{};
  hzbLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  hzbLayoutCI.bindingCount = static_cast<uint32_t>(hzbBindings.size());
  hzbLayoutCI.pBindings = hzbBindings.data();
  if (vkCreateDescriptorSetLayout(dev, &hzbLayoutCI, nullptr,
                                  &gpuDrivenHzbBuildSetLayout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create HZB build descriptor layout");

  gpuDrivenHzbExtent = {std::max(1u, swapchain.getExtent().width / 2),
                        std::max(1u, swapchain.getExtent().height / 2)};
  gpuDrivenHzbMipCount = 1;
  uint32_t maxDim = std::max(gpuDrivenHzbExtent.width,
                             gpuDrivenHzbExtent.height);
  while (maxDim > 1) {
    maxDim >>= 1;
    ++gpuDrivenHzbMipCount;
  }

  gpuDrivenHzbImages.resize(swapCount);
  gpuDrivenHzbViews.resize(swapCount);
  gpuDrivenHzbMipViews.resize(swapCount);
  gpuDrivenHzbValid.assign(swapCount, false);
  gpuDrivenFrames.resize(swapCount);

  VmaAllocationCreateInfo hzbAlloc{};
  hzbAlloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  for (size_t i = 0; i < swapCount; ++i) {
    VkImageCreateInfo imageCI{};
    imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCI.imageType = VK_IMAGE_TYPE_2D;
    imageCI.extent = {gpuDrivenHzbExtent.width, gpuDrivenHzbExtent.height, 1};
    imageCI.mipLevels = gpuDrivenHzbMipCount;
    imageCI.arrayLayers = 1;
    imageCI.format = gpuDrivenHzbFormat;
    imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkImage rawImage = VK_NULL_HANDLE;
    VmaAllocation rawAlloc = VK_NULL_HANDLE;
    if (vmaCreateImage(device.getAllocator(), &imageCI, &hzbAlloc, &rawImage,
                       &rawAlloc, nullptr) != VK_SUCCESS)
      throw std::runtime_error("Failed to create GPU-driven HZB image");
    gpuDrivenHzbImages[i] =
        AllocatedImage(device.getAllocator(), rawImage, rawAlloc);

    VkImageViewCreateInfo viewCI{};
    viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image = gpuDrivenHzbImages[i].get();
    viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format = gpuDrivenHzbFormat;
    viewCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                               gpuDrivenHzbMipCount, 0, 1};
    VkImageView fullView = VK_NULL_HANDLE;
    if (vkCreateImageView(dev, &viewCI, nullptr, &fullView) != VK_SUCCESS)
      throw std::runtime_error("Failed to create HZB image view");
    gpuDrivenHzbViews[i] = ImageViewHandle(dev, fullView);

    gpuDrivenHzbMipViews[i].resize(gpuDrivenHzbMipCount);
    for (uint32_t mip = 0; mip < gpuDrivenHzbMipCount; ++mip) {
      VkImageViewCreateInfo mipViewCI = viewCI;
      mipViewCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1};
      VkImageView mipView = VK_NULL_HANDLE;
      if (vkCreateImageView(dev, &mipViewCI, nullptr, &mipView) != VK_SUCCESS)
        throw std::runtime_error("Failed to create HZB mip view");
      gpuDrivenHzbMipViews[i][mip] = ImageViewHandle(dev, mipView);
    }
  }

  {
    VkCommandBuffer cmd = beginCommandBuffer(dev, device.getGraphicsCommandPool());
    std::vector<VkImageMemoryBarrier> initBarriers;
    initBarriers.reserve(gpuDrivenHzbImages.size());
    for (const AllocatedImage &image : gpuDrivenHzbImages) {
      VkImageMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      barrier.srcAccessMask = 0;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = image.get();
      barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                  gpuDrivenHzbMipCount, 0, 1};
      initBarriers.push_back(barrier);
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr,
                         static_cast<uint32_t>(initBarriers.size()),
                         initBarriers.data());
    endAndSubmitCommandBuffer(dev, device.getGraphicsCommandPool(),
                              device.getGraphicsQueue(), cmd);
  }

  VkSamplerCreateInfo hzbSamplerCI{};
  hzbSamplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  hzbSamplerCI.magFilter = VK_FILTER_NEAREST;
  hzbSamplerCI.minFilter = VK_FILTER_NEAREST;
  hzbSamplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  hzbSamplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  hzbSamplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  hzbSamplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  hzbSamplerCI.maxLod = static_cast<float>(gpuDrivenHzbMipCount);
  if (vkCreateSampler(dev, &hzbSamplerCI, nullptr, &gpuDrivenHzbSampler) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create HZB sampler");

  std::array<VkDescriptorPoolSize, 2> poolSizes{};
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSizes[0].descriptorCount = static_cast<uint32_t>(swapCount * 5);
  poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  poolSizes[1].descriptorCount = static_cast<uint32_t>(swapCount);
  VkDescriptorPoolCreateInfo poolCI{};
  poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolCI.maxSets = static_cast<uint32_t>(swapCount);
  poolCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolCI.pPoolSizes = poolSizes.data();
  if (vkCreateDescriptorPool(dev, &poolCI, nullptr,
                             &gpuDrivenDescriptorPool) != VK_SUCCESS)
    throw std::runtime_error("Failed to create GPU-driven descriptor pool");

  const uint32_t hzbSetCount =
      static_cast<uint32_t>(swapCount * gpuDrivenHzbMipCount);
  std::array<VkDescriptorPoolSize, 2> hzbPoolSizes{};
  hzbPoolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  hzbPoolSizes[0].descriptorCount = hzbSetCount;
  hzbPoolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  hzbPoolSizes[1].descriptorCount = hzbSetCount;
  VkDescriptorPoolCreateInfo hzbPoolCI{};
  hzbPoolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  hzbPoolCI.maxSets = hzbSetCount;
  hzbPoolCI.poolSizeCount = static_cast<uint32_t>(hzbPoolSizes.size());
  hzbPoolCI.pPoolSizes = hzbPoolSizes.data();
  if (vkCreateDescriptorPool(dev, &hzbPoolCI, nullptr,
                             &gpuDrivenHzbBuildDescriptorPool) != VK_SUCCESS)
    throw std::runtime_error("Failed to create HZB build descriptor pool");

  gpuDrivenDescriptorSets.resize(swapCount);
  std::vector<VkDescriptorSetLayout> setLayouts(swapCount, gpuDrivenSetLayout);
  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = gpuDrivenDescriptorPool;
  allocInfo.descriptorSetCount = static_cast<uint32_t>(swapCount);
  allocInfo.pSetLayouts = setLayouts.data();
  if (vkAllocateDescriptorSets(dev, &allocInfo,
                               gpuDrivenDescriptorSets.data()) != VK_SUCCESS)
    throw std::runtime_error("Failed to allocate GPU-driven descriptor sets");

  gpuDrivenHzbBuildSets.resize(hzbSetCount);
  std::vector<VkDescriptorSetLayout> hzbSetLayouts(
      hzbSetCount, gpuDrivenHzbBuildSetLayout);
  VkDescriptorSetAllocateInfo hzbAllocInfo{};
  hzbAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  hzbAllocInfo.descriptorPool = gpuDrivenHzbBuildDescriptorPool;
  hzbAllocInfo.descriptorSetCount = hzbSetCount;
  hzbAllocInfo.pSetLayouts = hzbSetLayouts.data();
  if (vkAllocateDescriptorSets(dev, &hzbAllocInfo,
                               gpuDrivenHzbBuildSets.data()) != VK_SUCCESS)
    throw std::runtime_error("Failed to allocate HZB build descriptor sets");

  for (GpuDrivenFrameResources &frame : gpuDrivenFrames) {
    ensureGpuDrivenBuffer(frame.meshBuffer, frame.meshBufferSize,
                          sizeof(GpuDrivenMeshRecord),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureGpuDrivenBuffer(frame.transformBuffer, frame.transformBufferSize,
                          sizeof(GpuDrivenTransformRecord),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureGpuDrivenBuffer(frame.indirectBuffer, frame.indirectBufferSize,
                          sizeof(VkDrawIndexedIndirectCommand),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    createBuffer(device.getAllocator(), sizeof(uint32_t),
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VMA_MEMORY_USAGE_AUTO, 0, &frame.countBuffer);
    createBuffer(device.getAllocator(), sizeof(GpuDrivenFrustumData),
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VMA_MEMORY_USAGE_AUTO,
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                 &frame.frustumBuffer);
    frame.descriptorDirty = true;
  }

  for (size_t image = 0; image < swapCount; ++image) {
    for (uint32_t mip = 0; mip < gpuDrivenHzbMipCount; ++mip) {
      const size_t setIndex = image * gpuDrivenHzbMipCount + mip;
      VkDescriptorImageInfo srcInfo{};
      srcInfo.imageLayout = (mip == 0)
                                ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      srcInfo.sampler = gpuDrivenHzbSampler;
      srcInfo.imageView = (mip == 0) ? gBufferDepthViews[image].get()
                                     : gpuDrivenHzbMipViews[image][mip - 1].get();
      VkDescriptorImageInfo dstInfo{};
      dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      dstInfo.imageView = gpuDrivenHzbMipViews[image][mip].get();
      std::array<VkWriteDescriptorSet, 2> writes{};
      writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[0].dstSet = gpuDrivenHzbBuildSets[setIndex];
      writes[0].dstBinding = 0;
      writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      writes[0].descriptorCount = 1;
      writes[0].pImageInfo = &srcInfo;
      writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[1].dstSet = gpuDrivenHzbBuildSets[setIndex];
      writes[1].dstBinding = 1;
      writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      writes[1].descriptorCount = 1;
      writes[1].pImageInfo = &dstInfo;
      vkUpdateDescriptorSets(dev, static_cast<uint32_t>(writes.size()),
                             writes.data(), 0, nullptr);
    }
  }
  for (uint32_t image = 0; image < static_cast<uint32_t>(swapCount); ++image)
    updateGpuDrivenDescriptorSet(image);

  for (int modelId : gpuDrivenModelIds)
    registerGpuDrivenModelGeometry(modelId);

  {
    auto compCode = readFile("../Shaders/gpu_cull.comp.spv");
    VkShaderModule compMod = createLocalShaderModule(dev, compCode);
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = compMod;
    stage.pName = "main";

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &gpuDrivenSetLayout;
    if (vkCreatePipelineLayout(dev, &layoutInfo, nullptr,
                               &gpuCullPipelineLayout) != VK_SUCCESS)
      throw std::runtime_error("Failed to create GPU cull pipeline layout");

    VkComputePipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeInfo.stage = stage;
    pipeInfo.layout = gpuCullPipelineLayout;
    if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pipeInfo, nullptr,
                                 &gpuCullPipeline) != VK_SUCCESS)
      throw std::runtime_error("Failed to create GPU cull pipeline");
    vkDestroyShaderModule(dev, compMod, nullptr);
  }

  {
    auto compCode = readFile("../Shaders/hzb_downsample.comp.spv");
    VkShaderModule compMod = createLocalShaderModule(dev, compCode);
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = compMod;
    stage.pName = "main";

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &gpuDrivenHzbBuildSetLayout;
    if (vkCreatePipelineLayout(dev, &layoutInfo, nullptr,
                               &gpuDrivenHzbBuildPipelineLayout) != VK_SUCCESS)
      throw std::runtime_error("Failed to create HZB build pipeline layout");

    VkComputePipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeInfo.stage = stage;
    pipeInfo.layout = gpuDrivenHzbBuildPipelineLayout;
    if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pipeInfo, nullptr,
                                 &gpuDrivenHzbBuildPipeline) != VK_SUCCESS)
      throw std::runtime_error("Failed to create HZB build pipeline");
    vkDestroyShaderModule(dev, compMod, nullptr);
  }

  {
    auto vertCode = readFile("../Shaders/shader_gpu.vert.spv");
    auto fragCode = readFile("../Shaders/shader_gpu.frag.spv");
    VkShaderModule vertMod = createLocalShaderModule(dev, vertCode);
    VkShaderModule fragMod = createLocalShaderModule(dev, fragCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(Vertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    std::array<VkVertexInputAttributeDescription, 6> attrs{};
    attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)};
    attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, col)};
    attrs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, tex)};
    attrs[3] = {3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};
    attrs[4] = {4, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, tangent)};
    attrs[5] = {5, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, bitangent)};
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attrs.size());
    vertexInput.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp = {0, 0, static_cast<float>(swapchain.getExtent().width),
                     static_cast<float>(swapchain.getExtent().height), 0, 1};
    VkRect2D sc = {{0, 0}, swapchain.getExtent()};
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &vp;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &sc;

    std::array<VkDynamicState, 2> dynStates = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynState{};
    dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynState.dynamicStateCount = static_cast<uint32_t>(dynStates.size());
    dynState.pDynamicStates = dynStates.data();

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo msaa{};
    msaa.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{};
    depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blendOff{};
    blendOff.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                              VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT |
                              VK_COLOR_COMPONENT_A_BIT;
    std::array<VkPipelineColorBlendAttachmentState, 3> blends = {
        blendOff, blendOff, blendOff};
    VkPipelineColorBlendStateCreateInfo blendState{};
    blendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blendState.attachmentCount = static_cast<uint32_t>(blends.size());
    blendState.pAttachments = blends.data();

    std::array<VkDescriptorSetLayout, 3> setLayouts = {
        descriptorManager.getVPLayout(), descriptorManager.getBindlessLayout(),
        gpuDrivenSetLayout};
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
    layoutInfo.pSetLayouts = setLayouts.data();
    if (vkCreatePipelineLayout(dev, &layoutInfo, nullptr,
                               &gpuDrivenGBufferPipelineLayout) != VK_SUCCESS)
      throw std::runtime_error("Failed to create GPU-driven G-buffer layout");

    VkGraphicsPipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.stageCount = 2;
    pipeInfo.pStages = stages;
    pipeInfo.pVertexInputState = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAssembly;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &raster;
    pipeInfo.pMultisampleState = &msaa;
    pipeInfo.pDepthStencilState = &depth;
    pipeInfo.pColorBlendState = &blendState;
    pipeInfo.pDynamicState = &dynState;
    pipeInfo.layout = gpuDrivenGBufferPipelineLayout;
    pipeInfo.renderPass = renderPassManager.getGBufferRenderPass();
    pipeInfo.subpass = 0;
    if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pipeInfo, nullptr,
                                  &gpuDrivenGBufferPipeline) != VK_SUCCESS)
      throw std::runtime_error("Failed to create GPU-driven G-buffer pipeline");

    vkDestroyShaderModule(dev, fragMod, nullptr);
    vkDestroyShaderModule(dev, vertMod, nullptr);
  }
}

void VulkanRenderer::cleanupGpuDrivenResources() {
  VkDevice dev = device.getLogicalDevice();
  if (gpuDrivenGBufferPipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(dev, gpuDrivenGBufferPipeline, nullptr);
  if (gpuDrivenGBufferPipelineLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(dev, gpuDrivenGBufferPipelineLayout, nullptr);
  if (gpuDrivenHzbBuildPipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(dev, gpuDrivenHzbBuildPipeline, nullptr);
  if (gpuDrivenHzbBuildPipelineLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(dev, gpuDrivenHzbBuildPipelineLayout, nullptr);
  if (gpuCullPipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(dev, gpuCullPipeline, nullptr);
  if (gpuCullPipelineLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(dev, gpuCullPipelineLayout, nullptr);
  if (gpuDrivenHzbSampler != VK_NULL_HANDLE)
    vkDestroySampler(dev, gpuDrivenHzbSampler, nullptr);
  if (gpuDrivenHzbBuildDescriptorPool != VK_NULL_HANDLE)
    vkDestroyDescriptorPool(dev, gpuDrivenHzbBuildDescriptorPool, nullptr);
  if (gpuDrivenDescriptorPool != VK_NULL_HANDLE)
    vkDestroyDescriptorPool(dev, gpuDrivenDescriptorPool, nullptr);
  if (gpuDrivenHzbBuildSetLayout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(dev, gpuDrivenHzbBuildSetLayout, nullptr);
  if (gpuDrivenSetLayout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(dev, gpuDrivenSetLayout, nullptr);

  gpuDrivenGBufferPipeline = VK_NULL_HANDLE;
  gpuDrivenGBufferPipelineLayout = VK_NULL_HANDLE;
  gpuDrivenHzbBuildPipeline = VK_NULL_HANDLE;
  gpuDrivenHzbBuildPipelineLayout = VK_NULL_HANDLE;
  gpuCullPipeline = VK_NULL_HANDLE;
  gpuCullPipelineLayout = VK_NULL_HANDLE;
  gpuDrivenHzbSampler = VK_NULL_HANDLE;
  gpuDrivenHzbBuildDescriptorPool = VK_NULL_HANDLE;
  gpuDrivenDescriptorPool = VK_NULL_HANDLE;
  gpuDrivenHzbBuildSetLayout = VK_NULL_HANDLE;
  gpuDrivenSetLayout = VK_NULL_HANDLE;
  gpuDrivenDescriptorSets.clear();
  gpuDrivenHzbBuildSets.clear();
  gpuDrivenHzbMipViews.clear();
  gpuDrivenHzbViews.clear();
  gpuDrivenHzbImages.clear();
  gpuDrivenHzbValid.clear();

  gpuDrivenFrames.clear();
  gpuDrivenStaticVertexBuffer.reset();
  gpuDrivenStaticIndexBuffer.reset();
  gpuDrivenStaticVertexBufferSize = 0;
  gpuDrivenStaticIndexBufferSize = 0;
  gpuDrivenStaticVertices.clear();
  gpuDrivenStaticIndices.clear();
  gpuDrivenGeometryRanges.clear();
  gpuDrivenHzbExtent = {};
  gpuDrivenHzbMipCount = 0;
  gpuDrivenHzbFormat = VK_FORMAT_UNDEFINED;
  gpuDrivenCandidateCount = 0;
  gpuDrivenMeshCount = 0;
  gpuDrivenLastFrameUsed = false;
}

void VulkanRenderer::updateGpuDrivenDescriptorSet(uint32_t imageIndex) {
  if (imageIndex >= gpuDrivenDescriptorSets.size() ||
      imageIndex >= gpuDrivenFrames.size() ||
      gpuDrivenHzbViews.size() != gpuDrivenDescriptorSets.size())
    return;

  GpuDrivenFrameResources &frame = gpuDrivenFrames[imageIndex];
  if (!frame.meshBuffer || !frame.transformBuffer || !frame.indirectBuffer ||
      !frame.countBuffer || !frame.frustumBuffer)
    return;
  if (!frame.descriptorDirty)
    return;

  std::array<VkDescriptorBufferInfo, 5> infos{};
  infos[0] = {frame.meshBuffer.get(), 0, VK_WHOLE_SIZE};
  infos[1] = {frame.frustumBuffer.get(), 0, VK_WHOLE_SIZE};
  infos[2] = {frame.transformBuffer.get(), 0, VK_WHOLE_SIZE};
  infos[3] = {frame.indirectBuffer.get(), 0, VK_WHOLE_SIZE};
  infos[4] = {frame.countBuffer.get(), 0, VK_WHOLE_SIZE};

  VkDescriptorImageInfo hzbInfo{};
  hzbInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  hzbInfo.imageView = gpuDrivenHzbViews[imageIndex].get();
  hzbInfo.sampler = gpuDrivenHzbSampler;

  std::array<VkWriteDescriptorSet, 6> writes{};
  for (uint32_t i = 0; i < 5; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = gpuDrivenDescriptorSets[imageIndex];
    writes[i].dstBinding = i;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].descriptorCount = 1;
    writes[i].pBufferInfo = &infos[i];
  }
  writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[5].dstSet = gpuDrivenDescriptorSets[imageIndex];
  writes[5].dstBinding = 5;
  writes[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  writes[5].descriptorCount = 1;
  writes[5].pImageInfo = &hzbInfo;
  vkUpdateDescriptorSets(device.getLogicalDevice(),
                         static_cast<uint32_t>(writes.size()), writes.data(),
                         0, nullptr);
  frame.descriptorDirty = false;
}

void VulkanRenderer::recordGpuDrivenHzbBuild(VkCommandBuffer cmd,
                                             uint32_t currentImage) {
  const uint32_t minCandidates =
      static_cast<uint32_t>(std::max(0, imguiGpuDrivenMinCandidates));
  if (!imguiGpuDrivenEnabled || !imguiHzbCullingEnabled ||
      gpuDrivenCandidateCount == 0 || gpuDrivenCandidateCount < minCandidates) {
    if (currentImage < gpuDrivenHzbValid.size())
      gpuDrivenHzbValid[currentImage] = false;
    return;
  }

  if (gpuDrivenHzbBuildPipeline == VK_NULL_HANDLE ||
      currentImage >= gpuDrivenHzbImages.size() ||
      currentImage >= gBufferDepthImages.size() ||
      gpuDrivenHzbMipCount == 0 || gpuDrivenHzbBuildSets.empty())
    return;

  vkdbgBeginLabel(cmd, "Build HZB Depth Pyramid", 0.15f, 0.7f, 1.0f);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    gpuDrivenHzbBuildPipeline);

  VkImageMemoryBarrier depthReadBarrier{};
  depthReadBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  depthReadBarrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
  depthReadBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
  depthReadBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  depthReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  depthReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  depthReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  depthReadBarrier.image = gBufferDepthImages[currentImage].get();
  depthReadBarrier.subresourceRange = {
      depthStencilAspectMask(gBufferDepthFormat), 0, 1, 0, 1};
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &depthReadBarrier);

  const bool valid = currentImage < gpuDrivenHzbValid.size() &&
                     gpuDrivenHzbValid[currentImage];
  for (uint32_t mip = 0; mip < gpuDrivenHzbMipCount; ++mip) {
    VkImageMemoryBarrier toGeneral{};
    toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toGeneral.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneral.srcAccessMask = valid ? VK_ACCESS_SHADER_READ_BIT : 0;
    toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.image = gpuDrivenHzbImages[currentImage].get();
    toGeneral.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1};
    vkCmdPipelineBarrier(cmd,
                         valid ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                               : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &toGeneral);

    const size_t setIndex = currentImage * gpuDrivenHzbMipCount + mip;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            gpuDrivenHzbBuildPipelineLayout, 0, 1,
                            &gpuDrivenHzbBuildSets[setIndex], 0, nullptr);
    const uint32_t w = std::max(1u, gpuDrivenHzbExtent.width >> mip);
    const uint32_t h = std::max(1u, gpuDrivenHzbExtent.height >> mip);
    vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);

    VkImageMemoryBarrier toRead{};
    toRead.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toRead.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toRead.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    toRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toRead.image = gpuDrivenHzbImages[currentImage].get();
    toRead.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &toRead);
  }

  if (currentImage < gpuDrivenHzbValid.size())
    gpuDrivenHzbValid[currentImage] = true;
  vkdbgEndLabel(cmd);
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

  // Temporal history indices. Rings are larger than the number of frames in
  // flight, so an in-flight frame's previous-history read is not overwritten
  // by a later CPU submission.
  uint32_t ssgiHistoryIndex =
      static_cast<uint32_t>(taaFrameCounter % ssgiHistoryViews.size());
  uint32_t taaHistoryIndex =
      static_cast<uint32_t>(taaFrameCounter % taaHistoryViews.size());

  recordGBufferPass(cmd, currentImage, viewport, scissor);

  // --- SSGI pass (one-bounce diffuse + temporal reproject) ---
  // Reads G-buffer (already in SHADER_READ_ONLY layout per the gbuffer
  // pass's finalLayout) + scene UBO + prev-frame SSGI history (set 2);
  // writes the blended result to ssgiHistoryImages[ssgiHistoryIndex]. lit.frag
  // samples this same history output with cross-bilateral (set 1, binding 5).
  {
    VkExtent2D ssgiExtent = {std::max(1u, swapchain.getExtent().width / 2),
                             std::max(1u, swapchain.getExtent().height / 2)};
    VkViewport ssgiViewport = viewport;
    ssgiViewport.width = static_cast<float>(ssgiExtent.width);
    ssgiViewport.height = static_cast<float>(ssgiExtent.height);
    VkRect2D ssgiScissor = {{0, 0}, ssgiExtent};

    VkClearValue clear = {};
    clear.color = {0.0f, 0.0f, 0.0f, 0.0f};

    VkRenderPassBeginInfo rpbi = {};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = renderPassManager.getSsgiRenderPass();
    rpbi.framebuffer = ssgiFramebuffers[ssgiHistoryIndex];
    rpbi.renderArea = {{0, 0}, ssgiExtent};
    rpbi.clearValueCount = 1;
    rpbi.pClearValues = &clear;

    vkdbgBeginLabel(cmd, "SSGI (reproject + blend)", 0.95f, 0.55f, 0.20f);
    metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::SSGI);
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdSetViewport(cmd, 0, 1, &ssgiViewport);
    vkCmdSetScissor(cmd, 0, 1, &ssgiScissor);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getSsgiPipeline());
    std::array<VkDescriptorSet, 3> ssgiSets = {
        descriptorManager.getVPSet(currentImage),
        descriptorManager.getGBufferSet(ssgiHistoryIndex, currentImage),
        descriptorManager.getSsgiPrevSet(ssgiHistoryIndex)};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getSsgiLayout(), 0, 3, ssgiSets.data(),
                            0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
    metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::SSGI);
    vkdbgEndLabel(cmd);
  }

  // --- Lit pass (PBR + IBL + SSAO + FXAA + fog → litBuffer) ---
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

    vkdbgBeginLabel(cmd, "Deferred Lighting (PBR+IBL+SSAO)", 0.1f, 0.8f,
                    0.3f);
    metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::Lit);
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getLitPipeline());
    std::array<VkDescriptorSet, 2> litSets = {
        descriptorManager.getVPSet(currentImage),
        descriptorManager.getGBufferSet(ssgiHistoryIndex, currentImage)};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getLitLayout(), 0, 2, litSets.data(), 0,
                            nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
    metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::Lit);
    vkdbgEndLabel(cmd);
  }

  // --- Bloom pyramid (compute) ---
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::Bloom);
  recordBloomPass(cmd, currentImage);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::Bloom);

  // --- Auto-exposure (compute) ---
  // Runs between lit and composite: lit is now in SHADER_READ_ONLY_OPTIMAL
  // (the render pass's finalLayout). Pass 1 histograms it; pass 2 reduces
  // and writes one float into a host-visible buffer that next frame's CPU
  // reads to drive eye adaptation.
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::AutoExposure);
  recordAutoExposurePass(cmd, currentImage);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::AutoExposure);

  // --- Composition pass (SSR composite + TAA + ACES tone-mapping) ---
  {
    // 3 attachments: swap (clear), colorBuffer (clear), history (DONT_CARE).
    // Clear values for DONT_CARE attachments are ignored but the array still
    // needs the right element count.
    std::array<VkClearValue, 3> clears{};
    clears[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
    clears[1].color = {0.0f, 0.0f, 0.0f, 1.0f};
    clears[2].color = {0.0f, 0.0f, 0.0f, 1.0f};

    // History index selects which TAA history image is THIS frame's write target.
    // taaFrameCounter is incremented at end-of-draw, so its current value is
    // the index of the frame we're recording.
    size_t fbIdx = taaHistoryIndex * swapchain.getImageCount() + currentImage;

    VkRenderPassBeginInfo rpbi = {};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = renderPassManager.getRenderPass();
    rpbi.framebuffer = compositeFramebuffers[fbIdx];
    rpbi.renderArea = {{0, 0}, swapchain.getExtent()};
    rpbi.clearValueCount = static_cast<uint32_t>(clears.size());
    rpbi.pClearValues = clears.data();

    vkdbgBeginLabel(cmd, "Composition (SSR + ACES Tonemap)", 0.6f, 0.2f, 0.8f);
    metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::Composite);
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Subpass 0: SSR composite (samples litBuffer + G-buffer) → colorBuffer
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getDeferredPipeline());
    std::array<VkDescriptorSet, 2> deferredSets = {
        descriptorManager.getVPSet(currentImage),
        descriptorManager.getGBufferSet(ssgiHistoryIndex, currentImage)};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getDeferredLayout(), 0, 2,
                            deferredSets.data(), 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    // Subpass 1: TAA reprojection + ACES + gamma → swapchain (LDR) + history (HDR)
    vkCmdNextSubpass(cmd, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getSecondPipeline());
    // 3 sets: scene UBO (0), input attachment (1), TAA samplers (2).
    // TAA set points at the previous history ring slot.
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
    metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::Composite);
    vkdbgEndLabel(cmd);
  }

  vkdbgBeginLabel(cmd, "ImGui Pass", 0.9f, 0.8f, 0.1f);
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::ImGui);
  recordImGuiCommands(cmd, currentImage);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::ImGui);
  vkdbgEndLabel(cmd);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
    throw std::runtime_error("Failed to stop recording command buffer");
}

void VulkanRenderer::recordGBufferCommands(uint32_t currentImage) {
  VkCommandBuffer cmd = swapchain.getCommandBuffer(currentImage);
  vkResetCommandBuffer(cmd, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
    throw std::runtime_error("Failed to begin G-buffer command buffer");

  metrics.resetGpuQueries(cmd);

  VkViewport viewport = {};
  viewport.width = static_cast<float>(swapchain.getExtent().width);
  viewport.height = static_cast<float>(swapchain.getExtent().height);
  viewport.maxDepth = 1.0f;
  VkRect2D scissor = {{0, 0}, swapchain.getExtent()};
  recordGBufferPass(cmd, currentImage, viewport, scissor);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
    throw std::runtime_error("Failed to end G-buffer command buffer");
}

void VulkanRenderer::recordShadowCommands(VkCommandBuffer cmd) {
  vkResetCommandBuffer(cmd, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
    throw std::runtime_error("Failed to begin shadow command buffer");

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

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
    throw std::runtime_error("Failed to end shadow command buffer");
}

void VulkanRenderer::recordSsaoComputeCommands(VkCommandBuffer cmd,
                                               uint32_t currentImage) {
  vkResetCommandBuffer(cmd, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
    throw std::runtime_error("Failed to begin SSAO compute command buffer");

  VkImageMemoryBarrier toGeneral = {};
  toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  toGeneral.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  toGeneral.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toGeneral.image = ssaoImages[currentImage].get();
  toGeneral.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &toGeneral);

  vkdbgBeginLabel(cmd, "Async SSAO Compute", 0.2f, 0.7f, 1.0f);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssaoPipeline);
  const uint32_t ssgiHistoryIndex =
      static_cast<uint32_t>(taaFrameCounter % ssgiHistoryViews.size());
  std::array<VkDescriptorSet, 3> sets = {
      descriptorManager.getVPSet(currentImage),
      descriptorManager.getGBufferSet(ssgiHistoryIndex, currentImage),
      ssaoOutputSets[currentImage]};
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ssaoPipelineLayout, 0,
                          static_cast<uint32_t>(sets.size()), sets.data(), 0,
                          nullptr);
  VkExtent2D extent = swapchain.getExtent();
  vkCmdDispatch(cmd, (extent.width + 7) / 8, (extent.height + 7) / 8, 1);
  vkdbgEndLabel(cmd);

  VkImageMemoryBarrier toSampled = toGeneral;
  toSampled.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  toSampled.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  toSampled.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  toSampled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &toSampled);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
    throw std::runtime_error("Failed to end SSAO compute command buffer");
}

void VulkanRenderer::recordPostCommands(VkCommandBuffer cmd,
                                        uint32_t currentImage) {
  vkResetCommandBuffer(cmd, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
    throw std::runtime_error("Failed to begin post command buffer");

  VkViewport viewport = {};
  viewport.width = static_cast<float>(swapchain.getExtent().width);
  viewport.height = static_cast<float>(swapchain.getExtent().height);
  viewport.maxDepth = 1.0f;
  VkRect2D scissor = {{0, 0}, swapchain.getExtent()};

  uint32_t ssgiHistoryIndex =
      static_cast<uint32_t>(taaFrameCounter % ssgiHistoryViews.size());
  uint32_t taaHistoryIndex =
      static_cast<uint32_t>(taaFrameCounter % taaHistoryViews.size());

  // --- SSGI pass (one-bounce diffuse + temporal reproject) ---
  {
    VkExtent2D ssgiExtent = {std::max(1u, swapchain.getExtent().width / 2),
                             std::max(1u, swapchain.getExtent().height / 2)};
    VkViewport ssgiViewport = viewport;
    ssgiViewport.width = static_cast<float>(ssgiExtent.width);
    ssgiViewport.height = static_cast<float>(ssgiExtent.height);
    VkRect2D ssgiScissor = {{0, 0}, ssgiExtent};

    VkClearValue clear = {};
    clear.color = {0.0f, 0.0f, 0.0f, 0.0f};

    VkRenderPassBeginInfo rpbi = {};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = renderPassManager.getSsgiRenderPass();
    rpbi.framebuffer = ssgiFramebuffers[ssgiHistoryIndex];
    rpbi.renderArea = {{0, 0}, ssgiExtent};
    rpbi.clearValueCount = 1;
    rpbi.pClearValues = &clear;

    vkdbgBeginLabel(cmd, "SSGI (reproject + blend)", 0.95f, 0.55f, 0.20f);
    metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::SSGI);
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdSetViewport(cmd, 0, 1, &ssgiViewport);
    vkCmdSetScissor(cmd, 0, 1, &ssgiScissor);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getSsgiPipeline());
    std::array<VkDescriptorSet, 3> ssgiSets = {
        descriptorManager.getVPSet(currentImage),
        descriptorManager.getGBufferSet(ssgiHistoryIndex, currentImage),
        descriptorManager.getSsgiPrevSet(ssgiHistoryIndex)};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getSsgiLayout(), 0, 3, ssgiSets.data(),
                            0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
    metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::SSGI);
    vkdbgEndLabel(cmd);
  }

  // --- Lit pass (PBR + IBL + async SSAO + FXAA + fog -> litBuffer) ---
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

    vkdbgBeginLabel(cmd, "Deferred Lighting (PBR+IBL+SSAO)", 0.1f, 0.8f,
                    0.3f);
    metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::Lit);
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getLitPipeline());
    std::array<VkDescriptorSet, 2> litSets = {
        descriptorManager.getVPSet(currentImage),
        descriptorManager.getGBufferSet(ssgiHistoryIndex, currentImage)};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getLitLayout(), 0, 2, litSets.data(), 0,
                            nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
    metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::Lit);
    vkdbgEndLabel(cmd);
  }

  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::Bloom);
  recordBloomPass(cmd, currentImage);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::Bloom);

  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::AutoExposure);
  recordAutoExposurePass(cmd, currentImage);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::AutoExposure);

  // --- Composition pass (SSR composite + TAA + tone-mapping) ---
  {
    std::array<VkClearValue, 3> clears{};
    clears[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
    clears[1].color = {0.0f, 0.0f, 0.0f, 1.0f};
    clears[2].color = {0.0f, 0.0f, 0.0f, 1.0f};

    size_t fbIdx = taaHistoryIndex * swapchain.getImageCount() + currentImage;

    VkRenderPassBeginInfo rpbi = {};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = renderPassManager.getRenderPass();
    rpbi.framebuffer = compositeFramebuffers[fbIdx];
    rpbi.renderArea = {{0, 0}, swapchain.getExtent()};
    rpbi.clearValueCount = static_cast<uint32_t>(clears.size());
    rpbi.pClearValues = clears.data();

    vkdbgBeginLabel(cmd, "Composition (SSR + AgX Tonemap)", 0.6f, 0.2f, 0.8f);
    metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::Composite);
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getDeferredPipeline());
    std::array<VkDescriptorSet, 2> deferredSets = {
        descriptorManager.getVPSet(currentImage),
        descriptorManager.getGBufferSet(ssgiHistoryIndex, currentImage)};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getDeferredLayout(), 0, 2,
                            deferredSets.data(), 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdNextSubpass(cmd, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getSecondPipeline());
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
    metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::Composite);
    vkdbgEndLabel(cmd);
  }

  vkdbgBeginLabel(cmd, "ImGui Pass", 0.9f, 0.8f, 0.1f);
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::ImGui);
  recordImGuiCommands(cmd, currentImage);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::ImGui);
  vkdbgEndLabel(cmd);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
    throw std::runtime_error("Failed to end post command buffer");
}

void VulkanRenderer::recordGBufferPass(VkCommandBuffer cmd, uint32_t currentImage,
                                       const VkViewport &viewport,
                                       const VkRect2D &scissor) {
  std::vector<GBufferDrawItem> drawItems;
  drawItems.reserve(128);
  GpuDrivenFrameResources *gpuFrame =
      currentImage < gpuDrivenFrames.size() ? &gpuDrivenFrames[currentImage]
                                            : nullptr;

  glm::vec4 frustumPlanes[6];
  extractFrustumPlanes(sceneUbo.projection * sceneUbo.view, frustumPlanes);

  auto appendNode = [&](auto &self, SceneNode *node) -> void {
    if (node->getModelId() >= 0) {
      MeshModel *mdl = modelManager.getModel(node->getModelId());
      if (mdl) {
        glm::mat4 model = node->getGlobalTransform();
        glm::vec3 wCenter =
            glm::vec3(model * glm::vec4(mdl->boundingCenter, 1.0f));
        float maxScale =
            glm::max(glm::length(glm::vec3(model[0])),
                     glm::max(glm::length(glm::vec3(model[1])),
                              glm::length(glm::vec3(model[2]))));
        if (sphereInFrustum(frustumPlanes, wCenter,
                            mdl->boundingRadius * maxScale)) {
          float camDist =
              glm::length(wCenter - glm::vec3(sceneUbo.cameraPosition));
          int lod = (camDist < imguiLodNear)  ? 0
                    : (camDist < imguiLodFar) ? 1
                                              : 2;

          ModelPushConstants basePush{};
          basePush.model = model;
          basePush.normal = node->getNormalMatrix();

          for (size_t k = 0; k < mdl->getMeshCount(); k++) {
            const Mesh *mesh = mdl->getMesh(k);
            glm::vec3 meshWCenter =
                glm::vec3(model * glm::vec4(mesh->boundingCenter, 1.0f));
            if (!sphereInFrustum(frustumPlanes, meshWCenter,
                                 mesh->boundingRadius * maxScale))
              continue;

            const Material &mat = mesh->getMaterial();
            GBufferDrawItem item{};
            item.kind = GBufferDrawKind::Mesh;
            item.mesh = mesh;
            item.lod = std::min(lod, mesh->getLodCount() - 1);
            item.push = basePush;
            item.push.texIdx0 = materialTextureIndices0(mat);
            item.push.texIdx1 = materialTextureIndices1(mat);
            glm::vec3 radiusVec(mesh->boundingRadius * maxScale);
            item.aabbMin = meshWCenter - radiusVec;
            item.aabbMax = meshWCenter + radiusVec;
            item.cullMode =
                mat.doubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
            item.indexCount = static_cast<uint32_t>(mesh->getIndexCount(item.lod));
            item.materialId = static_cast<uint32_t>(drawItems.size());
            item.transformId = static_cast<uint32_t>(drawItems.size());
            drawItems.push_back(item);
          }
        }
      }
    }
    for (auto &child : node->getChildren())
      self(self, child.get());
  };
  appendNode(appendNode, &rootNode);

  // Instanced drawables still become ordinary draw items here. The secondary
  // command buffer recorder switches to the instanced pipeline for these items.
  for (const InstancedDrawable &drawable : instancedDrawables) {
    MeshModel *mdl = modelManager.getModel(drawable.modelId);
    if (!mdl || drawable.instances.empty())
      continue;
    if (!sphereInFrustum(frustumPlanes, drawable.groupCenter,
                         drawable.groupRadius))
      continue;

    float groupDist =
        glm::length(drawable.groupCenter - glm::vec3(sceneUbo.cameraPosition));
    int instLod = (groupDist < imguiLodNear)  ? 0
                  : (groupDist < imguiLodFar) ? 1
                                              : 2;
    for (size_t k = 0; k < mdl->getMeshCount(); k++) {
      const Mesh *mesh = mdl->getMesh(k);
      const Material &mat = mesh->getMaterial();
      GBufferDrawItem item{};
      item.kind = GBufferDrawKind::Instanced;
      item.mesh = mesh;
      item.lod = std::min(instLod, mesh->getLodCount() - 1);
      item.push.texIdx0 = materialTextureIndices0(mat);
      item.push.texIdx1 = materialTextureIndices1(mat);
      glm::vec3 radiusVec(drawable.groupRadius);
      item.aabbMin = drawable.groupCenter - radiusVec;
      item.aabbMax = drawable.groupCenter + radiusVec;
      item.cullMode = VK_CULL_MODE_BACK_BIT;
      item.instanceBuffer = drawable.instanceBuffer.get();
      item.instances = &drawable.instances;
      item.instanceCount = static_cast<uint32_t>(drawable.instances.size());
      item.indexCount = static_cast<uint32_t>(mesh->getIndexCount(item.lod));
      item.materialId = static_cast<uint32_t>(drawItems.size());
      item.transformId = static_cast<uint32_t>(drawItems.size());
      drawItems.push_back(item);
    }
  }

  gpuDrivenCandidateCount = static_cast<uint32_t>(drawItems.size());
  gpuDrivenMeshCount = 0;
  gpuDrivenLastFrameUsed = false;

  const uint32_t minGpuDrivenCandidates =
      static_cast<uint32_t>(std::max(0, imguiGpuDrivenMinCandidates));
  const bool canPrepareGpuDriven =
      gpuFrame && imguiGpuDrivenEnabled &&
      gpuDrivenCandidateCount >= minGpuDrivenCandidates &&
      gpuDrivenGBufferPipeline != VK_NULL_HANDLE &&
      gpuCullPipeline != VK_NULL_HANDLE &&
      currentImage < gpuDrivenDescriptorSets.size() &&
      gpuDrivenStaticVertexBuffer && gpuDrivenStaticIndexBuffer;

  std::vector<GpuDrivenMeshRecord> gpuMeshRecords;
  std::vector<GpuDrivenTransformRecord> gpuTransforms;
  if (canPrepareGpuDriven && !drawItems.empty()) {
    gpuMeshRecords.reserve(drawItems.size());
    for (const GBufferDrawItem &item : drawItems) {
      const GpuDrivenGeometryRange *range =
          findGpuDrivenGeometry(item.mesh, item.lod);
      if (!range)
        continue;

      uint32_t baseInstance = static_cast<uint32_t>(gpuTransforms.size());
      if (item.kind == GBufferDrawKind::Instanced && item.instances) {
        for (const InstanceData &instance : *item.instances) {
          GpuDrivenTransformRecord tr{};
          tr.model = instance.model;
          tr.normal = instance.normal;
          tr.texIdx0 = item.push.texIdx0;
          tr.texIdx1 = item.push.texIdx1;
          gpuTransforms.push_back(tr);
        }
      } else {
        GpuDrivenTransformRecord tr{};
        tr.model = item.push.model;
        tr.normal = item.push.normal;
        tr.texIdx0 = item.push.texIdx0;
        tr.texIdx1 = item.push.texIdx1;
        gpuTransforms.push_back(tr);
      }

      GpuDrivenMeshRecord record{};
      record.aabbMin = glm::vec4(item.aabbMin, 0.0f);
      record.aabbMax = glm::vec4(item.aabbMax, 0.0f);
      record.draw = glm::uvec4(range->firstIndex, range->indexCount,
                               static_cast<uint32_t>(range->vertexOffset),
                               baseInstance);
      record.flags = glm::uvec4(item.instanceCount,
                                static_cast<uint32_t>(item.lod),
                                static_cast<uint32_t>(item.kind),
                                static_cast<uint32_t>(item.cullMode));
      gpuMeshRecords.push_back(record);
    }

    const VkDeviceSize transformBytes = static_cast<VkDeviceSize>(
        gpuTransforms.size() * sizeof(GpuDrivenTransformRecord));
    const VkDeviceSize indirectBytes = static_cast<VkDeviceSize>(
        std::max<size_t>(1, gpuMeshRecords.size()) *
        sizeof(VkDrawIndexedIndirectCommand));

    if (ensureGpuDrivenBuffer(gpuFrame->transformBuffer,
                              gpuFrame->transformBufferSize, transformBytes,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
      gpuFrame->descriptorDirty = true;
    }
    if (ensureGpuDrivenBuffer(gpuFrame->indirectBuffer,
                              gpuFrame->indirectBufferSize, indirectBytes,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                  VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT)) {
      gpuFrame->descriptorDirty = true;
    }

    auto upload = [&](AllocatedBuffer &buffer, const void *src,
                      VkDeviceSize bytes) {
      if (bytes == 0 || src == nullptr)
        return;
      void *mapped = nullptr;
      vmaMapMemory(device.getAllocator(), buffer.getAllocation(), &mapped);
      memcpy(mapped, src, static_cast<size_t>(bytes));
      vmaUnmapMemory(device.getAllocator(), buffer.getAllocation());
    };
    upload(gpuFrame->transformBuffer, gpuTransforms.data(), transformBytes);

    uploadGpuDrivenMeshRecords(
        *gpuFrame, gpuMeshRecords.data(),
        static_cast<VkDeviceSize>(gpuMeshRecords.size() *
                                  sizeof(GpuDrivenMeshRecord)),
        static_cast<uint32_t>(gpuMeshRecords.size()));

    GpuDrivenFrustumData frustumData{};
    for (int i = 0; i < 6; ++i)
      frustumData.planes[i] = frustumPlanes[i];
    frustumData.viewProj = sceneUbo.projection * sceneUbo.view;
    const bool hzbReady = imguiHzbCullingEnabled &&
                          currentImage < gpuDrivenHzbValid.size() &&
                          gpuDrivenHzbValid[currentImage];
    frustumData.hzbParams = glm::vec4(
        static_cast<float>(gpuDrivenHzbExtent.width),
        static_cast<float>(gpuDrivenHzbExtent.height),
        static_cast<float>(gpuDrivenHzbMipCount),
        hzbReady ? 1.0f : 0.0f);
    frustumData.meshCount = gpuDrivenMeshCount;
    if (gpuFrame->frustumBuffer) {
      void *mapped = nullptr;
      vmaMapMemory(device.getAllocator(),
                   gpuFrame->frustumBuffer.getAllocation(), &mapped);
      memcpy(mapped, &frustumData, sizeof(frustumData));
      vmaUnmapMemory(device.getAllocator(),
                     gpuFrame->frustumBuffer.getAllocation());
    }
    updateGpuDrivenDescriptorSet(currentImage);
  }

  for (const GBufferDrawItem &item : drawItems)
    metrics.recordDrawCall(item.indexCount * item.instanceCount);

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

  const VkDescriptorSet vpSet = descriptorManager.getVPSet(currentImage);
  const VkDescriptorSet bindlessSet = descriptorManager.getBindlessSet();
  const std::array<VkDescriptorSet, 2> graphicsSets = {vpSet, bindlessSet};
  const std::array<VkDescriptorSet, 2> instancedSets = {vpSet, bindlessSet};

  auto recordDrawCommands = [&](VkCommandBuffer drawCmd, size_t begin,
                                size_t end) {
    vkCmdSetViewport(drawCmd, 0, 1, &viewport);
    vkCmdSetScissor(drawCmd, 0, 1, &scissor);
    GBufferDrawKind boundKind = GBufferDrawKind::Mesh;
    bool pipelineBound = false;
    VkCullModeFlags lastCull = VK_CULL_MODE_BACK_BIT;
    bool cullValid = false;

    for (size_t i = begin; i < end; ++i) {
      const GBufferDrawItem &item = drawItems[i];
      if (!pipelineBound || boundKind != item.kind) {
        if (item.kind == GBufferDrawKind::Mesh) {
          vkCmdBindPipeline(drawCmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getGraphicsPipeline());
          vkCmdBindDescriptorSets(drawCmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  pipeline.getGraphicsLayout(), 0, 2,
                                  graphicsSets.data(), 0, nullptr);
        } else {
          vkCmdBindPipeline(drawCmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getInstancedPipeline());
          vkCmdBindDescriptorSets(drawCmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  pipeline.getInstancedLayout(), 0, 2,
                                  instancedSets.data(), 0, nullptr);
        }
        boundKind = item.kind;
        pipelineBound = true;
        cullValid = false;
      }

      if (!cullValid || item.cullMode != lastCull) {
        vkCmdSetCullMode(drawCmd, item.cullMode);
        lastCull = item.cullMode;
        cullValid = true;
      }

      VkBuffer vb[] = {item.mesh->getVertexBuffer()};
      VkDeviceSize off[] = {0};
      vkCmdBindVertexBuffers(drawCmd, 0, 1, vb, off);
      if (item.kind == GBufferDrawKind::Instanced) {
        VkDeviceSize instanceOff = 0;
        vkCmdBindVertexBuffers(drawCmd, 1, 1, &item.instanceBuffer,
                               &instanceOff);
      }
      vkCmdBindIndexBuffer(drawCmd, item.mesh->getIndexBuffer(item.lod), 0,
                           VK_INDEX_TYPE_UINT32);
      VkPipelineLayout layout =
          item.kind == GBufferDrawKind::Mesh ? pipeline.getGraphicsLayout()
                                             : pipeline.getInstancedLayout();
      vkCmdPushConstants(drawCmd, layout,
                         VK_SHADER_STAGE_VERTEX_BIT |
                             VK_SHADER_STAGE_FRAGMENT_BIT,
                         0, sizeof(ModelPushConstants), &item.push);
      vkCmdDrawIndexed(drawCmd, item.indexCount, item.instanceCount, 0, 0, 0);
    }
  };

  const bool canRecordThreaded =
      commandThreadPool && threadedCommandWorkerCount > 0 &&
      currentFrame < static_cast<int>(threadedCommandFrames.size()) &&
      !threadedCommandFrames[currentFrame].empty() && !drawItems.empty();

  const bool useGpuDriven =
      imguiGpuDrivenEnabled && gpuDrivenGBufferPipeline != VK_NULL_HANDLE &&
      gpuCullPipeline != VK_NULL_HANDLE &&
      currentImage < gpuDrivenDescriptorSets.size() &&
      gpuFrame != nullptr && gpuDrivenMeshCount > 0 &&
      gpuDrivenStaticVertexBuffer && gpuDrivenStaticIndexBuffer &&
      gpuDrivenCandidateCount >= minGpuDrivenCandidates;
  gpuDrivenLastFrameUsed = useGpuDriven;

  if (useGpuDriven) {
    vkdbgBeginLabel(cmd, "GPU Cull + Indirect Build", 0.8f, 0.25f, 1.0f);
    vkCmdFillBuffer(cmd, gpuFrame->countBuffer.get(), 0, sizeof(uint32_t), 0);
    VkBufferMemoryBarrier countClearBarrier{};
    countClearBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    countClearBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    countClearBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                      VK_ACCESS_SHADER_WRITE_BIT;
    countClearBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    countClearBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    countClearBarrier.buffer = gpuFrame->countBuffer.get();
    countClearBarrier.offset = 0;
    countClearBarrier.size = sizeof(uint32_t);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         1, &countClearBarrier, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gpuCullPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            gpuCullPipelineLayout, 0, 1,
                            &gpuDrivenDescriptorSets[currentImage], 0, nullptr);
    vkCmdDispatch(cmd, (gpuDrivenMeshCount + 63) / 64, 1, 1);

    std::array<VkBufferMemoryBarrier, 2> cullBarriers{};
    cullBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    cullBarriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    cullBarriers[0].dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    cullBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    cullBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    cullBarriers[0].buffer = gpuFrame->indirectBuffer.get();
    cullBarriers[0].offset = 0;
    cullBarriers[0].size = VK_WHOLE_SIZE;
    cullBarriers[1] = cullBarriers[0];
    cullBarriers[1].buffer = gpuFrame->countBuffer.get();
    cullBarriers[1].size = sizeof(uint32_t);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 0, nullptr,
                         static_cast<uint32_t>(cullBarriers.size()),
                         cullBarriers.data(), 0, nullptr);
    vkdbgEndLabel(cmd);

    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      gpuDrivenGBufferPipeline);
    std::array<VkDescriptorSet, 3> gpuSets = {
        vpSet, bindlessSet, gpuDrivenDescriptorSets[currentImage]};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            gpuDrivenGBufferPipelineLayout, 0,
                            static_cast<uint32_t>(gpuSets.size()),
                            gpuSets.data(), 0, nullptr);
    VkBuffer vb = gpuDrivenStaticVertexBuffer.get();
    VkDeviceSize vbOffset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vbOffset);
    vkCmdBindIndexBuffer(cmd, gpuDrivenStaticIndexBuffer.get(), 0,
                         VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexedIndirectCount(
        cmd, gpuFrame->indirectBuffer.get(), 0, gpuFrame->countBuffer.get(), 0,
        gpuDrivenMeshCount, sizeof(VkDrawIndexedIndirectCommand));
    vkCmdEndRenderPass(cmd);
    recordGpuDrivenHzbBuild(cmd, currentImage);
    metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::GBuffer);
    vkdbgEndLabel(cmd);
    return;
  }

  if (!canRecordThreaded) {
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    recordDrawCommands(cmd, 0, drawItems.size());
    vkCmdEndRenderPass(cmd);
    recordGpuDrivenHzbBuild(cmd, currentImage);
    metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::GBuffer);
    vkdbgEndLabel(cmd);
    return;
  }

  auto &frameResources = threadedCommandFrames[currentFrame];
  const uint32_t workerCount = std::min<uint32_t>(
      threadedCommandWorkerCount, static_cast<uint32_t>(drawItems.size()));
  std::vector<VkCommandBuffer> secondaries(workerCount, VK_NULL_HANDLE);

  for (uint32_t i = 0; i < workerCount; ++i) {
    ThreadCommandFrameResources &res = frameResources[i];
    vkResetCommandPool(device.getLogicalDevice(), res.pool, 0);
    secondaries[i] = res.gBufferSecondary;
  }

  const VkRenderPass gbufferPass = renderPassManager.getGBufferRenderPass();
  const VkFramebuffer gbufferFramebuffer = gBufferFramebuffers[currentImage];

  auto recordRange = [&](uint32_t workerIndex) {
    const size_t begin =
        (drawItems.size() * size_t(workerIndex)) / size_t(workerCount);
    const size_t end =
        (drawItems.size() * size_t(workerIndex + 1)) / size_t(workerCount);
    VkCommandBuffer secondary = secondaries[workerIndex];

    VkCommandBufferInheritanceInfo inheritance{};
    inheritance.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
    inheritance.renderPass = gbufferPass;
    inheritance.subpass = 0;
    inheritance.framebuffer = gbufferFramebuffer;

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT |
                      VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
    beginInfo.pInheritanceInfo = &inheritance;
    if (vkBeginCommandBuffer(secondary, &beginInfo) != VK_SUCCESS)
      throw std::runtime_error("Failed to begin G-buffer secondary command buffer");

    recordDrawCommands(secondary, begin, end);
    if (vkEndCommandBuffer(secondary) != VK_SUCCESS)
      throw std::runtime_error("Failed to end G-buffer secondary command buffer");
  };

  commandThreadPool->dispatch(workerCount, recordRange);

  vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
  vkCmdExecuteCommands(cmd, workerCount, secondaries.data());
  vkCmdEndRenderPass(cmd);
  recordGpuDrivenHzbBuild(cmd, currentImage);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::GBuffer);
  vkdbgEndLabel(cmd);
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

  // CSM shadow casters must use the authored mesh. Sponza's walls, roof
  // caps, and arch inserts are thin occluders; meshoptimizer's reduced LODs
  // can remove or move exactly the triangles that should block the sun. That
  // makes the visible G-buffer mesh look intact while the shadow map has
  // holes, which shows up as hard rectangular "sun through wall" patches on
  // interior floors/walls. Keep all cascades at LOD0 and spend the shadow
  // pass cost on correctness.
  auto cascadeLod = [](int cascade) {
    (void)cascade;
    return 0;
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
                              const glm::vec4 casterPlanes[6],
                              int lodIndex) -> void {
    if (node->getModelId() >= 0) {
      MeshModel *mdl = modelManager.getModel(node->getModelId());
      if (mdl) {
        const glm::mat4 model = node->getGlobalTransform();
        float maxScale =
            glm::max(glm::length(glm::vec3(model[0])),
                     glm::max(glm::length(glm::vec3(model[1])),
                              glm::length(glm::vec3(model[2]))));
        for (size_t k = 0; k < mdl->getMeshCount(); k++) {
          const Mesh *mesh = mdl->getMesh(k);
          glm::vec3 meshWCenter =
              glm::vec3(model * glm::vec4(mesh->boundingCenter, 1.0f));
          if (imguiCullShadowCasters &&
              !sphereInFrustum(casterPlanes, meshWCenter,
                               mesh->boundingRadius * maxScale))
            continue;

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
          push.model = model;
          push.lightSpaceMatrix = lsm;
          push.albedoIdx =
              static_cast<uint32_t>(mesh->getMaterial().albedoTextureId);
          uint32_t materialFlags =
              (mesh->getMaterial().isCloth ? 1u : 0u) |
              (mesh->getMaterial().alphaMasked ? 2u : 0u);
          push.materialFlags = materialFlags;
          push.alphaCutoff255 = static_cast<uint32_t>(
              glm::clamp(mesh->getMaterial().alphaCutoff, 0.0f, 1.0f) *
                  255.0f +
              0.5f);
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
      self(self, child.get(), lsm, casterPlanes, lodIndex);
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
    glm::vec4 casterPlanes[6];
    extractFrustumPlanes(sceneUbo.lightSpaceMatrices[cascade], casterPlanes);
    renderNodeShadow(renderNodeShadow, &rootNode,
                     sceneUbo.lightSpaceMatrices[cascade], casterPlanes,
                     cascadeLod(cascade));
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
                                           : VK_CULL_MODE_NONE;
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
            uint32_t materialFlags =
                (mesh->getMaterial().isCloth ? 1u : 0u) |
                (mesh->getMaterial().alphaMasked ? 2u : 0u);
            push.materialFlags = materialFlags;
            push.alphaCutoff255 = static_cast<uint32_t>(
                glm::clamp(mesh->getMaterial().alphaCutoff, 0.0f, 1.0f) *
                    255.0f +
                0.5f);
            vkCmdPushConstants(cmdBuffer, pipeline.getShadowLayout(),
                               VK_SHADER_STAGE_VERTEX_BIT |
                                   VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(ShadowPushConstants), &push);
            int useLod = 0;
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
  ImGui::SetNextWindowSize(ImVec2(340, 300), ImGuiCond_Always);
  ImGui::Begin("Performance", nullptr,
               ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoCollapse);
  ImGui::Text("CPU submit FPS: %.1f | CPU: %.2f ms | Avg: %.2f ms",
              metrics.getAverageFps(), metrics.getLastFrameTimeMs(),
              metrics.getAverageFrameTimeMs());
  frameTimeGraphData[frameTimeGraphOffset] =
      static_cast<float>(metrics.getLastFrameTimeMs());
  frameTimeGraphOffset = (frameTimeGraphOffset + 1) % 128;
  ImGui::PlotLines("##ft", frameTimeGraphData, 128, frameTimeGraphOffset,
                   "Frame time (ms)", 0.0f, 50.0f, ImVec2(322, 60));
  ImGui::Text("Shadow: %.2f | Pt: %.2f ms",
              metrics.getPassGpuTimeMs(PerformanceMetrics::GpuPass::Shadow),
              metrics.getPassGpuTimeMs(PerformanceMetrics::GpuPass::PointShadow));
  ImGui::Text("GBuffer: %.2f | SSGI: %.2f ms",
              metrics.getPassGpuTimeMs(PerformanceMetrics::GpuPass::GBuffer),
              metrics.getPassGpuTimeMs(PerformanceMetrics::GpuPass::SSGI));
  ImGui::Text("Lit: %.2f | Bloom: %.2f ms",
              metrics.getPassGpuTimeMs(PerformanceMetrics::GpuPass::Lit),
              metrics.getPassGpuTimeMs(PerformanceMetrics::GpuPass::Bloom));
  ImGui::Text("AutoExp: %.2f | Comp: %.2f ms",
              metrics.getPassGpuTimeMs(PerformanceMetrics::GpuPass::AutoExposure),
              metrics.getPassGpuTimeMs(PerformanceMetrics::GpuPass::Composite));
  ImGui::Text("ImGui: %.2f ms",
              metrics.getPassGpuTimeMs(PerformanceMetrics::GpuPass::ImGui));
  ImGui::Text("GPU total: %.2f | avg %.2f ms", metrics.getLastGpuTimeMs(),
              metrics.getAverageGpuTimeMs());
  ImGui::Text("CPU wait/acq/pres: %.2f | %.2f | %.2f ms",
              metrics.getCpuPhaseTimeMs(
                  PerformanceMetrics::CpuPhase::WaitFence),
              metrics.getCpuPhaseTimeMs(
                  PerformanceMetrics::CpuPhase::Acquire),
              metrics.getCpuPhaseTimeMs(
                  PerformanceMetrics::CpuPhase::Present));
  ImGui::Text("CPU upd/rec/sub: %.2f | %.2f | %.2f ms",
              metrics.getCpuPhaseTimeMs(PerformanceMetrics::CpuPhase::Update),
              metrics.getCpuPhaseTimeMs(PerformanceMetrics::CpuPhase::Record),
              metrics.getCpuPhaseTimeMs(PerformanceMetrics::CpuPhase::Submit));
  ImGui::Text("CPU active: %.2f | sync wait: %.2f ms",
              metrics.getCpuActiveTimeMs(), metrics.getCpuSyncWaitTimeMs());
  ImGui::Text("CPU measured: %.2f | avg %.2f ms",
              metrics.getCpuPhaseTotalMs(),
              metrics.getAverageCpuPhaseTotalMs());
  ImGui::Text("Draws: %u  |  Tris: %uk", metrics.getLastDrawCallCount(),
              metrics.getLastTriangleCount() / 1000);
  auto vram = PerformanceMetrics::queryVram(device.getAllocator());
  ImGui::Text("VRAM: %llu MiB / %llu MiB",
              static_cast<unsigned long long>(vram.usedBytes >> 20),
              static_cast<unsigned long long>(vram.budgetBytes >> 20));
  ImGui::End();

  // Camera panel
  ImGui::SetNextWindowPos(ImVec2(10, 320), ImGuiCond_Always);
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
  ImGui::SetNextWindowPos(ImVec2(10, 450), ImGuiCond_Always);
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
  ImGui::SetNextWindowPos(ImVec2(10, 690), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(340, 95), ImGuiCond_Always);
  ImGui::Begin("Debug Views", nullptr,
               ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoCollapse);
  const char *debugModes[] = {"None",          "Albedo",      "Normals",
                              "Metallic",      "Roughness",   "Depth",
                              "Shadow vis",    "SSAO factor", "Direct only",
                              "Indirect only", "Direct (no shadow)",
                              "SSGI bounce",   "SSGI raw",    "Bloom only"};
  if (ImGui::Combo("G-Buffer", &imguiDebugMode, debugModes, 14))
    sceneUbo.debugMode = imguiDebugMode;
  ImGui::Checkbox("Use geometric normal only", &imguiUseGeomNormalOnly);
  ImGui::End();

  // Tunables + scene controls. The per-fix A/B checkboxes that lived here
  // before were retired once each fix was validated — keeping only the
  // values that actually need runtime tuning.
  ImGui::SetNextWindowPos(ImVec2(10, 770), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(340, 420), ImGuiCond_Always);
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
  ImGui::Checkbox("Cull shadow casters", &imguiCullShadowCasters);
  ImGui::Checkbox("GPU-driven G-buffer", &imguiGpuDrivenEnabled);
  ImGui::Checkbox("HZB occlusion cull", &imguiHzbCullingEnabled);
  ImGui::SliderInt("GPU min candidates", &imguiGpuDrivenMinCandidates, 0, 2048);
  ImGui::Text("GPU candidates: %u | submitted: %u",
              gpuDrivenCandidateCount, gpuDrivenMeshCount);
  ImGui::Text("GPU path: %s",
              gpuDrivenLastFrameUsed ? "indirect" : "direct fallback");
  if (ImGui::Checkbox("Uncapped present", &imguiPreferMailboxPresent)) {
    swapchain.setPreferMailbox(imguiPreferMailboxPresent);
    framebufferResized = true;
  }
  ImGui::SameLine();
  ImGui::Text("actual: %s",
              presentModeName(swapchain.getActivePresentMode()));
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
  ImGui::SliderInt("SSGI samples", &imguiSsgiSamples, 4, 12);
  ImGui::SeparatorText("Lighting isolation");
  ImGui::Checkbox("Sun direct", &imguiEnableSunDirect);
  ImGui::Checkbox("Point lights", &imguiEnablePointLights);
  ImGui::Checkbox("Spot lights", &imguiEnableSpotLights);
  ImGui::Checkbox("IBL ambient", &imguiEnableIblAmbient);
  ImGui::Checkbox("SSGI bounce", &imguiEnableSsgiBounce);
  if (ImGui::Checkbox("Bloom", &imguiEnableBloom))
    taaHistoryValid = false;
  if (imguiEnableBloom) {
    float bloomSensitivity =
        (4.0f - imguiBloomThreshold) / (4.0f - 0.2f);
    if (ImGui::SliderFloat("Bloom sensitivity", &bloomSensitivity, 0.0f, 1.0f,
                           "%.2f")) {
      imguiBloomThreshold = glm::mix(4.0f, 0.2f, bloomSensitivity);
    }
    ImGui::SliderFloat("Bloom intensity", &imguiBloomIntensity, 0.0f, 1.0f,
                       "%.3f");
    ImGui::SliderFloat("Bloom radius", &imguiBloomRadius, 0.5f, 4.0f,
                       "%.2f");
  }
  ImGui::Checkbox("SSR reflections", &imguiSsrEnabled);
  ImGui::Checkbox("TAA", &imguiTaaEnabled);
  ImGui::Checkbox("Responsive TAA", &imguiResponsiveTaa);
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

void VulkanRenderer::createSynchronization() {
  size_t swapCount = swapchain.getImageCount();
  imageAvailable.resize(MAX_FRAMES_DRAWS);
  drawFences.resize(MAX_FRAMES_DRAWS);
  asyncComputeTimeline.resize(MAX_FRAMES_DRAWS, VK_NULL_HANDLE);
  asyncComputeTimelineValue.assign(MAX_FRAMES_DRAWS, 0);
  // renderFinished is per-swap-image, not per-frame-in-flight (see header
  // comment for the validation rationale).
  renderFinished.resize(swapCount);
  imagesInFlight.resize(swapCount, VK_NULL_HANDLE);
  frameImageInFlight.resize(MAX_FRAMES_DRAWS,
                            std::numeric_limits<uint32_t>::max());

  VkSemaphoreCreateInfo semInfo = {};
  semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkSemaphoreTypeCreateInfo timelineInfo = {};
  timelineInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  timelineInfo.initialValue = 0;
  VkSemaphoreCreateInfo timelineSemInfo = {};
  timelineSemInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  timelineSemInfo.pNext = &timelineInfo;
  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  VkDevice dev = device.getLogicalDevice();
  for (size_t i = 0; i < MAX_FRAMES_DRAWS; i++) {
    if (vkCreateSemaphore(dev, &semInfo, nullptr, &imageAvailable[i]) !=
            VK_SUCCESS ||
        vkCreateSemaphore(dev, &timelineSemInfo, nullptr,
                          &asyncComputeTimeline[i]) != VK_SUCCESS ||
        vkCreateFence(dev, &fenceInfo, nullptr, &drawFences[i]) != VK_SUCCESS)
      throw std::runtime_error("Failed to create synchronization primitives");
  }
  for (size_t i = 0; i < swapCount; i++) {
    if (vkCreateSemaphore(dev, &semInfo, nullptr, &renderFinished[i]) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to create renderFinished semaphore");
  }
}

void VulkanRenderer::createAsyncFrameCommandBuffers() {
  cleanupAsyncFrameCommandBuffers();

  const size_t count = swapchain.getImageCount();
  VkDevice dev = device.getLogicalDevice();
  shadowCommandBuffers.assign(count, VK_NULL_HANDLE);
  postCommandBuffers.assign(count, VK_NULL_HANDLE);
  ssaoCommandBuffers.assign(count, VK_NULL_HANDLE);

  auto allocate = [&](VkCommandPool pool, std::vector<VkCommandBuffer> &out,
                      const char *label) {
    VkCommandBufferAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = static_cast<uint32_t>(out.size());
    if (!out.empty() && vkAllocateCommandBuffers(dev, &ai, out.data()) !=
                            VK_SUCCESS) {
      throw std::runtime_error(std::string("Failed to allocate ") + label +
                               " command buffers");
    }
  };

  allocate(device.getGraphicsCommandPool(), shadowCommandBuffers, "shadow");
  allocate(device.getGraphicsCommandPool(), postCommandBuffers, "post");
  allocate(device.getComputeCommandPool(), ssaoCommandBuffers, "SSAO compute");
}

void VulkanRenderer::cleanupAsyncFrameCommandBuffers() {
  VkDevice dev = device.getLogicalDevice();
  if (dev == VK_NULL_HANDLE)
    return;

  auto freeBuffers = [&](VkCommandPool pool, std::vector<VkCommandBuffer> &bufs) {
    if (!bufs.empty()) {
      vkFreeCommandBuffers(dev, pool, static_cast<uint32_t>(bufs.size()),
                           bufs.data());
      bufs.clear();
    }
  };

  freeBuffers(device.getGraphicsCommandPool(), shadowCommandBuffers);
  freeBuffers(device.getGraphicsCommandPool(), postCommandBuffers);
  freeBuffers(device.getComputeCommandPool(), ssaoCommandBuffers);
}

void VulkanRenderer::createThreadedCommandResources() {
  cleanupThreadedCommandResources();

  const uint32_t hardwareThreads =
      std::max(1u, std::thread::hardware_concurrency());
  // Keep this conservative: command recording benefits from a few workers, but
  // oversubscribing the CPU can hurt frame pacing more than it helps.
  threadedCommandWorkerCount =
      std::min<uint32_t>(4u, std::max(1u, hardwareThreads > 1
                                              ? hardwareThreads - 1u
                                              : 1u));
  commandThreadPool =
      std::make_unique<CommandThreadPool>(threadedCommandWorkerCount);

  QueueFamilyIndices qi = device.getQueueFamilies();
  VkDevice dev = device.getLogicalDevice();
  threadedCommandFrames.assign(MAX_FRAMES_DRAWS, {});

  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                   VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = static_cast<uint32_t>(qi.graphicsFamily);

  for (size_t frame = 0; frame < threadedCommandFrames.size(); ++frame) {
    threadedCommandFrames[frame].resize(threadedCommandWorkerCount);
    for (uint32_t worker = 0; worker < threadedCommandWorkerCount; ++worker) {
      ThreadCommandFrameResources &res = threadedCommandFrames[frame][worker];
      if (vkCreateCommandPool(dev, &poolInfo, nullptr, &res.pool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create worker command pool");

      VkCommandBufferAllocateInfo allocInfo{};
      allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      allocInfo.commandPool = res.pool;
      allocInfo.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
      allocInfo.commandBufferCount = 1;
      if (vkAllocateCommandBuffers(dev, &allocInfo, &res.gBufferSecondary) !=
          VK_SUCCESS)
        throw std::runtime_error("Failed to allocate G-buffer secondary command buffer");
    }
  }

  spdlog::info("Phase 7.4 threaded command recording: {} worker(s)",
               threadedCommandWorkerCount);
}

void VulkanRenderer::cleanupThreadedCommandResources() {
  commandThreadPool.reset();

  VkDevice dev = device.getLogicalDevice();
  if (dev != VK_NULL_HANDLE) {
    for (auto &frameResources : threadedCommandFrames) {
      for (ThreadCommandFrameResources &res : frameResources) {
        if (res.pool != VK_NULL_HANDLE) {
          vkDestroyCommandPool(dev, res.pool, nullptr);
          res.pool = VK_NULL_HANDLE;
          res.gBufferSecondary = VK_NULL_HANDLE;
        }
      }
    }
  }

  threadedCommandFrames.clear();
  threadedCommandWorkerCount = 0;
}
