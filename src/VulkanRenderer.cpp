#include "VulkanRenderer.h"
#include "CameraProjection.h"
#include "ImageCapture.h"
#include "Frustum.h"
#include "Model.h"
#include "Picking.h"
#include "PlayerStart.h"
#include "Spawning.h"
#include "ScenePhysics.h"
#include "Utilities.h"
#include "VulkanDebug.h"
#include <spdlog/spdlog.h>

#include <chrono>
#include <cstdlib>
#include <format>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <glm/gtc/constants.hpp>

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
#include <utility>
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

VulkanRenderer::VulkanRenderer() {}

int VulkanRenderer::init(GLFWwindow *newWindow) {
  window = newWindow;
  // First, so everything set up below starts from the saved values.
  loadSettings();
  loadCharacter();

  try {
    // 1. Device
    device.init(window);
    pipelineCache.init(device.getLogicalDevice());

    // 2. Determine formats
    VkFormat swapchainFormat = swapchain.queryImageFormat(device);

    VkFormat colorFormat = swapchain.chooseSupportedFormat(
        device.getPhysicalDevice(),
        {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM},
        VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT);

    shadowDepthFormat = swapchain.chooseSupportedFormat(
        device.getPhysicalDevice(),
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
         VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);

    gBufferDepthFormat = shadowDepthFormat;
    sceneColorFormat = colorFormat;

    VkFormat gb0Fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormat gb1Fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormat gb2Fmt = VK_FORMAT_R8G8B8A8_UNORM;
    litFormat = colorFormat;

    // 3. Render passes
    renderPassManager.createGBufferRenderPass(
        device.getLogicalDevice(), gb0Fmt, gb1Fmt, gb2Fmt, gBufferDepthFormat);
    renderPassManager.createLitRenderPass(device.getLogicalDevice(), litFormat,
                                          gBufferDepthFormat);
    renderPassManager.createSsgiRenderPass(device.getLogicalDevice(), litFormat);
    renderPassManager.createRenderPass(device.getLogicalDevice(),
                                       swapchainFormat, colorFormat,
                                       litFormat);
    renderPassManager.createShadowRenderPass(device.getLogicalDevice(),
                                             shadowDepthFormat);
    renderPassManager.createImGuiRenderPass(device.getLogicalDevice(),
                                            swapchainFormat);

    // 4. Swapchain (images + command buffers). The scene starts at the
    //    window's size, filling it, until something says otherwise.
    swapchain.init(device, window);
    registerSwapchainResources();
    sceneExtent = swapchain.getExtent();
    sceneArea = {glm::vec2(0.0f),
                 glm::vec2(sceneExtent.width, sceneExtent.height)};
    sceneResize.emplace(glm::ivec2(sceneExtent.width, sceneExtent.height));
    {
      // The finished picture reaches the window by a blit, stretched while
      // the scene's area is resized.
      VkFormatProperties props = {};
      vkGetPhysicalDeviceFormatProperties(device.getPhysicalDevice(),
                                          swapchainFormat, &props);
      const VkFormatFeatureFlags needed =
          VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
          VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
      if ((props.optimalTilingFeatures & needed) != needed)
        throw std::runtime_error(
            "The swapchain format can't be blitted with filtering here");
    }

    // 5. Shadow resources (directional + point)
    shadowPass.create(device, renderPassManager.getShadowRenderPass(),
                      shadowDepthFormat);

    // 6. Descriptors. The scene UBO is sampled by the async SSAO dispatch on
    // the dedicated compute queue (when one exists), so it needs CONCURRENT
    // graphics+compute sharing — EXCLUSIVE reads across families without an
    // ownership transfer are formally undefined.
    QueueFamilyIndices queueFamilies = device.getQueueFamilies();
    uint32_t sharedUboFamilies[2] = {
        static_cast<uint32_t>(queueFamilies.graphicsFamily),
        static_cast<uint32_t>(queueFamilies.computeFamily)};
    const bool shareUboWithCompute = device.hasDedicatedComputeQueue();
    descriptorManager.init(device.getLogicalDevice(), device.getAllocator(),
                           swapchain.getImageCount(),
                           shareUboWithCompute ? sharedUboFamilies : nullptr,
                           shareUboWithCompute ? 2u : 0u);
    descriptorManager.updateShadowMapDescriptor(
        device.getLogicalDevice(), shadowPass.csmView(),
        shadowPass.pointCubeView(), shadowPass.csmSampler(),
        shadowPass.pointSampler());

    // 7. Texture manager (samplers only at this point)
    textureManager.init(device, pipelineCache.get());

    // 8. Every image the scene's size decides: lit, bloom, SSGI, SSAO, TAA
    //    history, the composite's, the G-buffer, the GPU-driven pass's HZB,
    //    auto-exposure, and the descriptor sets that point at them.
    skinningPass.create(device, pipelineCache.get());
    debugLinePass.create(device, renderPassManager.getImGuiRenderPass(),
                         pipelineCache.get());
    createSceneTargets();

    // 10. Pipeline (needs all 5 render passes + descriptor layouts)
    pipeline.createPipelines(
        device.getLogicalDevice(), renderPassManager.getGBufferRenderPass(),
        renderPassManager.getLitRenderPass(), renderPassManager.getRenderPass(),
        renderPassManager.getShadowRenderPass(),
        renderPassManager.getSsgiRenderPass(), swapchain.getExtent(),
        pipelineCache.get(), descriptorManager);
    materialProbePass.create(device, MAX_FRAMES_DRAWS, descriptorManager,
                             pipelineCache.get());

    // 10. IBL resources (requires textureManager and descriptorManager)
    initIBL();

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

    applySunAndFog();

    // Left over from Sponza's torches. Off by default, see
    // imguiEnablePointLights.
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
    sceneUbo.visualToggles =
        glm::vec4(imguiEdgeAA, imguiAlphaDither, imguiShadowSoftness,
                  imguiContactGrounding);
    sceneUbo.lightCounts =
        glm::ivec4(1, 0, 0x3f, 0); // z = lighting-isolation bit mask

    shadowPass.updateLightSpaceMatrices(sceneUbo, imguiCsmFar,
                                        imguiDrawDistance);
    shadowPass.updatePointShadowMatrices(sceneUbo);

    // 14. Default albedo texture
    textureManager.loadTexture("plain.png", device, descriptorManager);

    // 15. ImGui overlay
    imguiLayer.init(window, device, renderPassManager.getImGuiRenderPass(),
                    static_cast<uint32_t>(swapchain.getImageCount()));
    // The panel layout lives next to the scenes and settings, so Debug and
    // Release share one.
    imguiLayer.setLayoutFile(std::filesystem::is_directory("Resources")
                                 ? "Resources/editor_layout.ini"
                                 : "../Resources/editor_layout.ini");
    imguiLayer.createFramebuffers(device.getLogicalDevice(),
                                  renderPassManager.getImGuiRenderPass(),
                                  swapchain.getExtent(), swapchain.getImages());

  } catch (const std::runtime_error &e) {
    spdlog::critical("Renderer initialization failed: {}", e.what());
    return EXIT_FAILURE;
  }
  return 0;
}

int VulkanRenderer::createMeshModel(const std::string &modelFile) {
  int modelId = modelManager.loadModel(modelFile, device, textureManager,
                                       descriptorManager);
  gpuDrivenGBufferPass.registerModelGeometry(modelId, modelManager);
  return modelId;
}

namespace {

// The model cache's name for a generated shape: shape, size, colour and the
// world grid. Entries with the same key share one model.
std::string primitiveModelKey(const SceneModelEntry &entry) {
  const glm::vec3 c = entry.color;
  std::string key;
  switch (*entry.shape) {
  case PrimitiveShape::Box:
    key = std::format("#box {} {} {}", entry.size.x, entry.size.y,
                      entry.size.z);
    break;
  case PrimitiveShape::Sphere:
    key = std::format("#sphere {}", entry.radius);
    break;
  case PrimitiveShape::Capsule:
    key = std::format("#capsule {} {}", entry.radius, entry.height);
    break;
  case PrimitiveShape::Plane:
    key = std::format("#plane {} {}", entry.size.x, entry.size.z);
    break;
  }
  key += std::format(" color {} {} {}", c.r, c.g, c.b);
  // A static box and a dynamic one can't share a model: only one has the
  // grid.
  if (showsWorldGrid(entry))
    key += " grid";
  return key;
}

} // namespace

int VulkanRenderer::createPrimitiveModel(const SceneModelEntry &entry) {
  const glm::vec3 c = entry.color;
  const std::string key = primitiveModelKey(entry);
  const bool worldGrid = showsWorldGrid(entry);
  if (int cached = modelManager.findModel(key); cached >= 0)
    return cached;

  MeshData mesh;
  bool withLods = false; // boxes and planes have nothing to simplify
  switch (*entry.shape) {
  case PrimitiveShape::Box:
    mesh = makeBox(entry.size, c);
    break;
  case PrimitiveShape::Sphere:
    mesh = makeSphere(entry.radius, c);
    withLods = true;
    break;
  case PrimitiveShape::Capsule:
    mesh = makeCapsule(entry.radius, entry.height, c);
    withLods = true;
    break;
  case PrimitiveShape::Plane:
    mesh = makePlane(entry.size.x, entry.size.z, c);
    break;
  }
  const int modelId =
      modelManager.createModel(key, std::move(mesh), withLods, worldGrid,
                               device, textureManager, descriptorManager);
  gpuDrivenGBufferPass.registerModelGeometry(modelId, modelManager);
  return modelId;
}

// Everything the Save settings buttons keep, with the range a value from
// the file must be in. Diagnostic switches are left out on purpose (GPU-
// driven, HZB, threaded recording, min candidates, uncapped present, the
// debug view, geometric normals and the lighting isolation boxes): they start
// at the code defaults every launch, so one left off after a test can't skew
// later measurements.
void VulkanRenderer::loadSettings() {
  // Next to the scenes. The game usually runs from a build folder.
  settingsPath = std::filesystem::is_directory("Resources")
                     ? "Resources/settings.cfg"
                     : "../Resources/settings.cfg";
  PhysicsPanel &p = physicsPanel;
  PhysicsWorld::Config &c = physicsPanel.config;
  CharacterTuning &ch = characterPanel.character;
  ThirdPersonCameraTuning &cam = characterPanel.camera;
  settingBindings = {
      {"render", "spec_aa_variance", &imguiSpecAAVariance, 0, 4},
      {"render", "spec_aa_threshold", &imguiSpecAAThreshold, 0, 1},
      {"render", "ibl_roughness_floor", &imguiIblRoughnessFloor, 0, 1},
      {"render", "min_surface_roughness", &imguiMinSurfaceRoughness, 0, 1},
      {"render", "sky_occlusion_floor", &imguiSkyOcclusionFloor, 0, 1},
      {"render", "csm_far", &imguiCsmFar, 10, 10000},
      {"render", "shadow_front_face_cull", &imguiShadowFrontFaceCull},
      {"render", "cull_shadow_casters", &imguiCullShadowCasters},
      {"render", "ibl_intensity", &imguiIblIntensity, 0, 4},
      {"render", "auto_exposure", &autoExpEnabled},
      {"render", "exposure_ev", &imguiExposureEV, -6, 6},
      {"render", "ssgi_intensity", &imguiSsgiIntensity, 0, 4},
      {"render", "ssgi_samples", &imguiSsgiSamples, 4, 12},
      {"render", "bloom", &imguiEnableBloom},
      {"render", "bloom_threshold", &imguiBloomThreshold, 0.2, 4},
      {"render", "bloom_intensity", &imguiBloomIntensity, 0, 1},
      {"render", "bloom_radius", &imguiBloomRadius, 0.5, 4},
      {"render", "ssr", &imguiSsrEnabled},
      {"render", "taa", &imguiTaaEnabled},
      {"render", "responsive_taa", &imguiResponsiveTaa},
      {"render", "sharpness", &imguiSharpness, 0, 1},
      {"render", "edge_aa", &imguiEdgeAA, 0, 1},
      {"render", "alpha_dither", &imguiAlphaDither, 0, 1},
      {"render", "shadow_softness", &imguiShadowSoftness, 0.5, 2.5},
      {"render", "contact_grounding", &imguiContactGrounding, 0, 1},
      {"render", "day_night_cycle", &imguiDayNightEnable},
      {"render", "day_night_speed", &imguiDayNightSpeed, 0.1, 600},
      {"render", "day_night_hour", &imguiDayNightHour, 0, 24},
      {"render", "fov", &imguiCameraFov, 30, 120},
      {"render", "draw_distance", &imguiDrawDistance, 10, 20000},
      {"render", "normal_strength", &imguiNormalStrength, 0, 1.5},
      {"render", "lod_near", &imguiLodNear, 1, 200},
      {"render", "lod_far", &imguiLodFar, 2, 1000},
      {"render", "point_shadow_far", &imguiPointShadowFar, 1, 500},
      {"physics", "gravity", &c.gravity, -100, 100},
      {"physics", "steps_per_second", &p.stepsPerSecond, 10, 480},
      {"physics", "max_catch_up_steps", &p.maxCatchUpSteps, 1, 16},
      {"physics", "velocity_steps", &c.velocitySteps, 2, 64},
      {"physics", "position_steps", &c.positionSteps, 1, 64},
      {"physics", "sleep_delay", &c.sleepDelay, 0, 60},
      {"physics", "sleep_speed", &c.sleepSpeed, 0, 10},
      {"physics", "penetration_slop", &c.penetrationSlop, 0, 1},
      {"physics", "default_friction", &p.defaultFriction, 0, 10},
      {"physics", "worker_threads", &c.workerThreads, 0, 64},
      {"physics", "max_bodies", &c.maxBodies, 64, 1 << 20},
      {"physics", "max_body_pairs", &c.maxBodyPairs, 64, 1 << 22},
      {"physics", "max_contact_constraints", &c.maxContactConstraints, 64,
       1 << 17},
      // Angles in radians, as the code keeps them; the tab shows degrees.
      {"character", "run_speed", &ch.runSpeed, 0.5, 20},
      {"character", "sprint_speed", &ch.sprintSpeed, 0.5, 30},
      {"character", "crouch_speed", &ch.crouchSpeed, 0.2, 10},
      {"character", "ground_acceleration", &ch.groundAcceleration, 1, 200},
      {"character", "ground_braking", &ch.groundBraking, 1, 200},
      {"character", "air_acceleration", &ch.airAcceleration, 0, 100},
      {"character", "turn_grip", &ch.turnGrip, 1, 100},
      {"character", "turn_rate", &ch.turnRate, 0.5, 63},
      {"character", "aim_acceleration", &ch.aimAcceleration, 1, 200},
      {"character", "jump_height", &ch.jumpHeight, 0.1, 5},
      {"character", "coyote_time", &ch.coyoteTime, 0, 0.5},
      {"character", "jump_buffer", &ch.jumpBuffer, 0, 0.5},
      {"character", "dodge_distance", &ch.dodgeDistance, 0, 20},
      {"character", "dodge_time", &ch.dodgeTime, 0.05, 1},
      {"character", "dodge_cooldown", &ch.dodgeCooldown, 0, 5},
      {"character", "max_slope", &ch.maxSlope, 0.08, 1.4},
      {"character", "step_up", &ch.stepUp, 0, 1},
      {"character", "stick_down", &ch.stickDown, 0, 2},
      {"character", "push_strength", &ch.pushStrength, 0, 5000},
      {"character", "radius", &ch.radius, 0.1, 1},
      {"character", "height", &ch.height, 0.5, 3},
      {"character", "crouch_height", &ch.crouchHeight, 0.3, 3},
      {"character", "mass", &ch.mass, 1, 1000},
      {"character", "camera_shoulder_height", &cam.pivotHeight, 0, 3},
      {"character", "camera_distance", &cam.distance, 0.5, 30},
      {"character", "camera_closest", &cam.minDistance, 0.5, 20},
      {"character", "camera_furthest", &cam.maxDistance, 0.5, 30},
      {"character", "camera_zoom_step", &cam.zoomStep, 0.05, 5},
      {"character", "camera_lowest_look", &cam.minPitch, -1.553, 0},
      {"character", "camera_highest_look", &cam.maxPitch, 0, 1.553},
      {"character", "camera_mouse_sensitivity", &cam.mouseSensitivity, 0.0002,
       0.02},
      {"character", "camera_aim_right", &cam.aimRight, -2, 2},
      {"character", "camera_aim_up", &cam.aimUp, -1, 2},
      {"character", "camera_aim_distance", &cam.aimDistance, 0.5, 10},
      {"character", "camera_aim_time", &cam.aimTime, 0.01, 1},
      {"character", "camera_follow_time", &cam.followTime, 0, 1},
      {"character", "camera_most_lag", &cam.maxLag, 0, 5},
      {"character", "camera_wall_margin", &cam.collisionRadius, 0.01, 1},
      {"character", "camera_ease_out_time", &cam.easeOutTime, 0, 3},
  };
  writeSettings(settingsDefaults, settingBindings);
  settingsFile = SettingsFile::load(settingsPath);
  readSettings(settingsFile, settingBindings);
  // A hand-edited file can hold values that don't fit together.
  ch = consistent(ch);
  cam = consistent(cam);
}

bool VulkanRenderer::saveSettings(std::string_view section) {
  writeSettings(settingsFile, settingBindings, section, settingsDefaults);
  try {
    settingsFile.save(settingsPath);
    settingsStatus = std::format("Saved {} settings", section);
    spdlog::info("Saved {} settings to {}", section, settingsPath);
    return true;
  } catch (const std::exception &e) {
    settingsStatus = std::format("Couldn't save: {}", e.what());
    spdlog::error("Couldn't save settings: {}", e.what());
    return false;
  }
}

std::vector<std::string> VulkanRenderer::unsavedSettingsSections() const {
  // While the day/night cycle runs its hour moves on its own, which isn't
  // an edit, so it's left out.
  std::vector<SettingBinding> checked;
  for (const SettingBinding &b : settingBindings)
    if (!(imguiDayNightEnable && b.key == "day_night_hour"))
      checked.push_back(b);
  std::vector<std::string> sections;
  for (const char *section : {"render", "physics", "character"})
    if (hasUnsavedSettings(settingsFile, checked, section, settingsDefaults))
      sections.push_back(section);
  return sections;
}

void VulkanRenderer::resetSettings(std::string_view section) {
  readSettings(settingsDefaults, settingBindings, section);
  // Values that only take effect through something built from them.
  rebuildProjection();
  taaHistoryValid = false;
  physicsPanel.settingsChanged = true;
  characterPanel.changed = true;
  const char *name = section == "render"    ? "Render"
                     : section == "physics" ? "Physics"
                                            : "Character";
  settingsStatus =
      std::format("{} settings are back to the defaults, not saved yet", name);
}

void VulkanRenderer::discoverScenes(const std::string &directory) {
  // Where SceneIO::discover looks, so Save as writes next to the others.
  scenesDirectory = std::filesystem::is_directory(directory)
                        ? directory
                        : (std::filesystem::path("..") / directory).string();
  availableScenes = SceneIO::discover(directory);
  availableSceneNames.clear();
  availableSceneNames.reserve(availableScenes.size());
  for (const SceneDescription &scene : availableScenes)
    availableSceneNames.push_back(scene.name);
  spdlog::info("Discovered {} scene file(s) in {}", availableScenes.size(),
               directory);
}

void VulkanRenderer::discoverModels(const std::string &folder) {
  modelFiles = findModelFiles(folder);
  spdlog::info("Found {} model file(s) in {}", modelFiles.size(), folder);
}

void VulkanRenderer::loadScene(const SceneDescription &desc, bool moveCamera) {
  // Rare, user-triggered operation: a full idle wait is the simple and safe
  // way to retire every in-flight frame before instance buffers and the
  // scene graph are torn down.
  vkDeviceWaitIdle(device.getLogicalDevice());
  const auto loadStart = std::chrono::steady_clock::now();

  // The character first: it lives in the world replaced below, and its
  // capsule in the tree cleared below. A new one comes at the end if still
  // playing.
  animator.reset();
  player.reset();
  playerNode = nullptr;
  // Physics next: physicsLinks points into the tree cleared below.
  // HELIORA_PHYSICS_THREADS overrides Jolt's worker count, for measuring.
  physicsLinks.clear();
  PhysicsWorld::Config physicsConfig = physicsPanel.config;
  if (const char *threads = std::getenv("HELIORA_PHYSICS_THREADS"))
    physicsConfig.workerThreads =
        static_cast<uint32_t>(std::max(0, std::atoi(threads)));
  physics = std::make_unique<PhysicsWorld>(physicsConfig);
  physicsClock = FixedStepClock(1.0 / physicsPanel.stepsPerSecond,
                                physicsPanel.maxCatchUpSteps);
  physicsPanel.settingsChanged = false; // the new world has them all
  physicsPanel.thrownBalls = 0;
  playSeconds = 0.0;
  // Stop passes currentScene itself; copying it onto itself is harmless.
  currentScene = desc;
  rootNode.clearChildren();
  instancedDrawables.clear(); // AllocatedBuffer RAII frees instance buffers

  // A new scene's entries are different ones, and its history starts here.
  // Stop and undo keep the selection and the history.
  if (moveCamera) {
    // Where the camera preset puts the fly camera, for a spawn below it
    // before main() has passed it in.
    flyCameraPosition = currentScene.camera.position;
    flyCameraYaw = currentScene.camera.yaw;
    selection.reset();
    history.clear();
    history.push(currentScene);
    savedScene = currentScene;
    saveStatus.clear();
  }
  updateUnsaved();

  for (int i = 0; i < static_cast<int>(currentScene.models.size()); ++i)
    addEntry({SceneEntryRef::Kind::Model, i});
  for (int i = 0; i < static_cast<int>(currentScene.grids.size()); ++i)
    addEntry({SceneEntryRef::Kind::Grid, i});

  for (const SceneRingEntry &ring : desc.rings) {
    if (ring.count <= 0)
      continue;
    try {
      int modelId = createMeshModel(ring.path);
      std::vector<glm::mat4> transforms;
      transforms.reserve(static_cast<size_t>(ring.count));
      for (int i = 0; i < ring.count; i++) {
        float angle = glm::two_pi<float>() * static_cast<float>(i) /
                      static_cast<float>(ring.count);
        glm::vec3 pos(ring.radius * std::cos(angle), ring.height,
                      ring.radius * std::sin(angle));
        glm::mat4 t = glm::translate(glm::mat4(1.0f), pos);
        t = glm::rotate(t, angle, glm::vec3(0.0f, 1.0f, 0.0f));
        t = glm::scale(t, glm::vec3(ring.scale));
        transforms.push_back(t);
      }
      addInstancedModel(modelId, transforms);
    } catch (const std::exception &e) {
      spdlog::error("Scene '{}': failed to load ring model '{}': {}", desc.name,
                    ring.path, e.what());
    }
  }

  // Bodies went in one by one, which leaves Jolt's broad phase tree
  // unbalanced.
  physics->optimizeBroadPhase();
  // The thrown ball's mesh is made here, while the GPU is idle: building it
  // mid-session would mean a GPU upload in the middle of the frame loop.
  try {
    ballModelId = createPrimitiveModel(thrownBallEntry());
  } catch (const std::exception &e) {
    ballModelId = -1;
    spdlog::error("Couldn't make the thrown ball's mesh: {}", e.what());
  }
  // The character's capsules, standing and crouched, for the same reason.
  ensurePlayerModels();
  if (playing)
    spawnPlayer();
  rootNode.update(glm::mat4(1.0f), 0.0f);
  updateEditorLines();

  if (!desc.environmentHdr.empty() &&
      desc.environmentHdr != activeEnvironmentHdr) {
    spdlog::warn("Scene '{}' requests environment '{}', but runtime HDRI "
                 "switching is not implemented yet; keeping the current sky.",
                 desc.name, desc.environmentHdr);
  }

  if (moveCamera && cameraPresetCallback)
    cameraPresetCallback(desc.camera.position, desc.camera.yaw,
                         desc.camera.pitch, desc.camera.speed);

  // The new scene shares no image history with the old one. The HZB pyramid
  // must be invalidated too — occlusion-culling the new scene against the
  // old scene's depth pops meshes for the first frames after a switch.
  // (The SSGI history ring keeps a few frames of the old scene's bounce
  // light; that ghost fades within ~10 frames of temporal blend and is
  // accepted rather than adding a clear path.)
  taaHistoryValid = false;
  gpuDrivenGBufferPass.invalidateHzb();

  spdlog::info("{} scene '{}' ({} model(s), {} grid(s), {} instanced "
               "ring(s), {} physics bodies, {} Jolt workers) in {:.0f} ms",
               moveCamera ? "Loaded" : "Rebuilt", desc.name,
               desc.models.size(), desc.grids.size(),
               desc.rings.size(), physics->bodyCount(),
               physicsConfig.workerThreads,
               std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - loadStart)
                   .count());
}

// Runs every frame, so the sliders need no path of their own and the scene's
// sun comes back when the day/night cycle is turned off.
void VulkanRenderer::applySunAndFog() {
  const SceneSun &sun = currentScene.sun;
  sceneUbo.directionalLight.direction =
      glm::vec4(glm::normalize(sun.direction), 0.0f);
  sceneUbo.directionalLight.colorIntensity =
      glm::vec4(sun.color, sun.intensity);
  const SceneFog &fog = currentScene.fog;
  sceneUbo.fogParams.x = fog.density;
  sceneUbo.fogParams.y = fog.falloff;
  sceneUbo.fogParams.z = fog.max;
}

int VulkanRenderer::modelFor(const SceneModelEntry &entry) {
  return entry.shape ? createPrimitiveModel(entry)
                     : createMeshModel(entry.path);
}

// A node for the entry, plus a physics body if it has one.
void VulkanRenderer::addEntryNode(const SceneModelEntry &entry, int modelId,
                                  SceneEntryRef ref) {
  auto node = std::make_unique<SceneNode>();
  node->setModelId(modelId);
  node->setEntry(ref);
  node->setTransform(entry.transform);
  node->setAnimation(entry.animation);
  if (entry.body) {
    // A model file's collider is built from its own triangles.
    PhysicsBodyDesc body = bodyDescFor(
        entry, entry.shape ? nullptr : modelManager.collisionParts(modelId),
        physicsPanel.defaultFriction);
    body.userData = physicsLinks.size();
    const PhysicsBodyId id = physics->addBody(body);
    if (id.isValid()) {
      node->setPhysicsLink(static_cast<int>(physicsLinks.size()));
      physicsLinks.push_back({node.get(), id});
    }
  }
  rootNode.addChild(std::move(node));
}

void VulkanRenderer::addEntry(SceneEntryRef ref) {
  const bool isGrid = ref.kind == SceneEntryRef::Kind::Grid;
  const SceneModelEntry &model = isGrid ? currentScene.grids[ref.index].model
                                        : currentScene.models[ref.index];
  try {
    // Every copy in a grid shares one model, so it's made once.
    const int modelId = modelFor(model);
    if (isGrid)
      for (const SceneModelEntry &copy :
           expandGrid(currentScene.grids[ref.index]))
        addEntryNode(copy, modelId, ref);
    else
      addEntryNode(model, modelId, ref);
  } catch (const std::exception &e) {
    spdlog::error("Scene '{}': failed to load {} '{}': {}", currentScene.name,
                  isGrid ? "grid model" : "model",
                  model.shape ? "generated shape" : model.path, e.what());
  }
}

void VulkanRenderer::removeEntryNodes(SceneEntryRef ref) {
  for (PhysicsLink &link : physicsLinks)
    if (link.node && link.node->entry() == ref) {
      physics->removeBody(link.body);
      link = {};
    }
  rootNode.removeChildren(
      [&](const SceneNode &node) { return node.entry() == ref; });
}

void VulkanRenderer::placeEntry(SceneEntryRef ref) {
  if (ref.kind == SceneEntryRef::Kind::Player) {
    updateEditorLines(); // it's only lines
    return;
  }
  auto place = [&](SceneNode &node, const SceneModelEntry &entry) {
    node.setTransform(entry.transform);
    if (node.physicsLink() >= 0) {
      // From bodyDescFor, since a plane's body sits under its surface.
      const PhysicsBodyDesc body =
          bodyDescFor(entry, nullptr, physicsPanel.defaultFriction);
      physics->setPose(physicsLinks[node.physicsLink()].body,
                       {body.position, body.rotation});
    }
  };
  const bool isGrid = ref.kind == SceneEntryRef::Kind::Grid;
  // A grid's nodes went in in expandGrid's order and stay in it.
  const std::vector<SceneModelEntry> copies =
      isGrid ? expandGrid(currentScene.grids[ref.index])
             : std::vector<SceneModelEntry>{currentScene.models[ref.index]};
  size_t next = 0;
  for (const auto &node : rootNode.getChildren())
    if (node->entry() == ref && next < copies.size())
      place(*node, copies[next++]);
  rootNode.update(glm::mat4(1.0f), playSeconds);
  updateEditorLines();
}

void VulkanRenderer::shiftEntryRefs(SceneEntryRef::Kind kind, int from,
                                    int by) {
  for (const auto &node : rootNode.getChildren()) {
    SceneEntryRef ref = node->entry();
    if (ref.kind == kind && ref.index >= from) {
      ref.index += by;
      node->setEntry(ref);
    }
  }
}

void VulkanRenderer::applyEntryOp(EntryOp op, SceneEntryRef ref) {
  // The player start has nothing to rebuild or split, only to delete.
  if (ref.kind == SceneEntryRef::Kind::Player) {
    if (op == EntryOp::Delete && currentScene.player) {
      removeEntry(currentScene, ref);
      selection.reset();
      recordEdit();
    }
    return;
  }
  const size_t count = ref.kind == SceneEntryRef::Kind::Grid
                           ? currentScene.grids.size()
                           : currentScene.models.size();
  if (ref.kind == SceneEntryRef::Kind::None || ref.index < 0 ||
      ref.index >= static_cast<int>(count))
    return;
  switch (op) {
  case EntryOp::Rebuild:
    removeEntryNodes(ref);
    addEntry(ref);
    break;
  case EntryOp::Delete:
    removeEntryNodes(ref);
    removeEntry(currentScene, ref);
    shiftEntryRefs(ref.kind, ref.index + 1, -1);
    selection.reset();
    break;
  case EntryOp::Split: {
    if (ref.kind != SceneEntryRef::Kind::Grid)
      return;
    removeEntryNodes(ref);
    const std::vector<SceneEntryRef> copies =
        splitGrid(currentScene, ref.index);
    shiftEntryRefs(SceneEntryRef::Kind::Grid, ref.index + 1, -1);
    if (!copies.empty())
      shiftEntryRefs(SceneEntryRef::Kind::Model, copies.front().index,
                     static_cast<int>(copies.size()));
    for (const SceneEntryRef &copy : copies)
      addEntry(copy);
    selection.reset();
    break;
  }
  }
  // Bodies added one by one leave Jolt's broad phase tree unbalanced.
  physics->optimizeBroadPhase();
  // A rebuild only catches the nodes up with an edit that was already
  // recorded; a delete or split is a step of its own.
  if (op != EntryOp::Rebuild)
    recordEdit();
}

// One new object from the spawn list, added like a thrown ball: its model,
// node and body, without a scene reload. It lands where the camera looks and
// gets selected, and adding it is one undo step.
void VulkanRenderer::spawnEntry(SceneModelEntry entry) {
  if (playing)
    return;
  int modelId = -1;
  try {
    modelId = modelFor(entry);
  } catch (const std::exception &e) {
    sceneError = std::format("Couldn't add '{}': {}", entry.path, e.what());
    spdlog::error("{}", sceneError);
    return;
  }
  // Its box, so it can be set against the surface instead of half in it.
  const std::optional<ModelManager::Bounds> bounds =
      modelManager.localBounds(modelId);
  const float scale = entry.transform.scale;
  const glm::vec3 boundsMin = bounds ? bounds->min * scale : glm::vec3(0.0f);
  const glm::vec3 boundsMax = bounds ? bounds->max * scale : glm::vec3(0.0f);
  // The ray through the middle of the screen, which is where the camera
  // looks; any viewport size gives the same ray for its centre.
  const PickRay ray =
      rayFromScreen(glm::vec2(0.5f), glm::vec2(1.0f),
                    glm::inverse(taaBaseProjection * sceneUbo.view));
  const std::optional<PhysicsRayHit> hit =
      physics ? physics->castRay(ray.origin, ray.direction, kSpawnReach)
              : std::nullopt;
  entry.transform.position = spawnPosition(ray, hit, boundsMin, boundsMax);

  const SceneEntryRef ref = addModel(currentScene, std::move(entry));
  addEntry(ref);
  rootNode.update(glm::mat4(1.0f), playSeconds);
  selection = ref;
  updateEditorLines();
  recordEdit();
  sceneError.clear();
  spdlog::info("Added [model] {}{}", ref.index,
               hit ? "" : ", 5 m ahead with no surface in reach");
}

// The spawn list's Player start: where the camera looks, facing away from
// it, like a new object. A scene has one, so an existing start moves (and
// keeps its comment). One undo step either way.
void VulkanRenderer::placePlayerStart() {
  if (playing || !physics)
    return;
  const PickRay ray =
      rayFromScreen(glm::vec2(0.5f), glm::vec2(1.0f),
                    glm::inverse(taaBaseProjection * sceneUbo.view));
  const CharacterTuning &tuning = characterPanel.character;
  ScenePlayerStart start = playerStartAt(
      ray, physics->castRay(ray.origin, ray.direction, kSpawnReach),
      tuning.radius, tuning.height);
  if (currentScene.player)
    start.info = currentScene.player->info;
  currentScene.player = start;
  selection = SceneEntryRef{SceneEntryRef::Kind::Player, 0};
  updateEditorLines();
  recordEdit();
  spdlog::info("Player start at ({:.2f}, {:.2f}, {:.2f}), facing {:.0f} degrees",
               start.position.x, start.position.y, start.position.z,
               start.yaw);
}

void VulkanRenderer::requestHistoryStep(HistoryStep step) {
  if (!playing && !imguiLayer.editInProgress())
    historyRequest = step;
}

void VulkanRenderer::applyHistoryStep() {
  const HistoryStep step = std::exchange(historyRequest, HistoryStep::None);
  if (playing)
    return;
  const std::optional<SceneDescription> scene =
      step == HistoryStep::Undo ? history.undo() : history.redo();
  const char *what = step == HistoryStep::Undo ? "undo" : "redo";
  if (!scene) {
    spdlog::info("Nothing to {}", what);
    return;
  }
  // The same index is the same object only if no entry came or went.
  const bool startGone = selection &&
                         selection->kind == SceneEntryRef::Kind::Player &&
                         !scene->player;
  if (scene->models.size() != currentScene.models.size() ||
      scene->grids.size() != currentScene.grids.size() || startGone)
    selection.reset();
  // The file and name are where the scene is saved, not part of the edits,
  // so undoing past a Save as still saves to the new file.
  SceneDescription restored = *scene;
  restored.name = currentScene.name;
  restored.filePath = currentScene.filePath;
  const auto start = std::chrono::steady_clock::now();
  loadScene(restored, /*moveCamera=*/false);
  spdlog::info("Applied {} in {:.1f} ms", what,
               std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
                   .count());
}

bool VulkanRenderer::loadSceneAt(int index) {
  if (index < 0 || index >= static_cast<int>(availableScenes.size())) {
    spdlog::error("loadSceneAt: index {} out of range ({} scenes)", index,
                  availableScenes.size());
    return false;
  }
  // From disk again, so edits made by hand since startup show up. A file
  // that no longer parses leaves the current scene running.
  SceneDescription fresh;
  try {
    fresh = SceneIO::loadFromFile(availableScenes[index].filePath);
  } catch (const std::exception &e) {
    sceneError = std::format("Couldn't load '{}': {}",
                             availableScenes[index].name, e.what());
    spdlog::error("{}", sceneError);
    return false;
  }
  sceneError.clear();
  availableScenes[index] = fresh;
  availableSceneNames[index] = fresh.name;
  loadScene(fresh);
  activeSceneIndex = index;
  return true;
}

void VulkanRenderer::requestSceneLoad(int index) {
  if (sceneUnsaved) {
    prompt = {UiPrompt::Kind::UnsavedSwitch, currentScene.name};
    promptSceneIndex = index;
    return;
  }
  pendingSceneIndex = index;
}

void VulkanRenderer::requestSave() {
  if (promptOpen())
    return;
  if (currentScene.filePath.empty())
    requestSaveAs();
  else
    saveScene();
}

void VulkanRenderer::requestSaveAs() {
  if (!promptOpen())
    prompt = {UiPrompt::Kind::SaveAs};
}

void VulkanRenderer::requestQuit() {
  if (promptOpen())
    return;
  const bool settingsUnsaved = !unsavedSettingsSections().empty();
  if (!sceneUnsaved && !settingsUnsaved) {
    quitting = true;
    return;
  }
  prompt = {UiPrompt::Kind::UnsavedQuit, currentScene.name, sceneUnsaved,
            settingsUnsaved};
}

bool VulkanRenderer::saveScene() {
  const std::string file =
      std::filesystem::path(currentScene.filePath).filename().string();
  try {
    // Through a temporary file and a rename, so a failed save leaves the
    // old file as it was.
    SceneIO::saveToFile(currentScene, currentScene.filePath);
  } catch (const std::exception &e) {
    saveStatus = std::format("Couldn't save {}: {}", file, e.what());
    spdlog::error("{}", saveStatus);
    return false;
  }
  savedScene = currentScene;
  updateUnsaved();
  saveStatus = std::format("Saved {}", file);
  spdlog::info("Saved scene '{}' to {}", currentScene.name,
               currentScene.filePath);
  return true;
}

bool VulkanRenderer::saveSceneAs(const std::string &path,
                                 const std::string &name) {
  SceneDescription renamed = currentScene;
  renamed.name = name;
  renamed.filePath = path;
  const std::string file = std::filesystem::path(path).filename().string();
  try {
    SceneIO::saveToFile(renamed, path);
  } catch (const std::exception &e) {
    saveStatus = std::format("Couldn't save {}: {}", file, e.what());
    spdlog::error("{}", saveStatus);
    return false;
  }
  currentScene.name = name;
  currentScene.filePath = path;
  savedScene = currentScene;
  updateUnsaved();
  // The picker: the same file again takes over its entry, a new one joins.
  std::error_code error;
  int index = -1;
  for (int i = 0; i < static_cast<int>(availableScenes.size()); ++i)
    if (std::filesystem::equivalent(availableScenes[i].filePath, path, error))
      index = i;
  if (index < 0) {
    index = static_cast<int>(availableScenes.size());
    availableScenes.push_back(currentScene);
    availableSceneNames.push_back(name);
  } else {
    availableScenes[index] = currentScene;
    availableSceneNames[index] = name;
  }
  activeSceneIndex = index;
  saveStatus = std::format("Saved as {}", file);
  spdlog::info("Saved scene '{}' as {}", name, path);
  return true;
}

void VulkanRenderer::answerPrompt(PromptAnswer answer,
                                  const std::string &text) {
  const UiPrompt::Kind kind = prompt.kind;
  auto close = [&] { prompt = {}; };
  if (answer == PromptAnswer::Cancel) {
    // Cancelling Replace goes back to the name, so it can be changed.
    if (kind == UiPrompt::Kind::Replace)
      prompt = {UiPrompt::Kind::SaveAs};
    else
      close();
    return;
  }
  switch (kind) {
  case UiPrompt::Kind::UnsavedQuit: {
    std::string problem; // a save that failed keeps the game open
    if (answer == PromptAnswer::Save) {
      if (sceneUnsaved && !saveScene())
        problem = saveStatus;
      for (const std::string &section : unsavedSettingsSections())
        if (!saveSettings(section))
          problem = settingsStatus;
    }
    if (problem.empty())
      quitting = true;
    else
      prompt.error = problem;
    break;
  }
  case UiPrompt::Kind::UnsavedSwitch:
    if (answer == PromptAnswer::Save && !saveScene()) {
      prompt.error = saveStatus;
      break;
    }
    pendingSceneIndex = promptSceneIndex;
    close();
    break;
  case UiPrompt::Kind::SaveAs: {
    const std::optional<std::string> file = sceneFileNameFor(text);
    if (!file) {
      prompt.error = "Use letters, digits, spaces, - and _ only.";
      break;
    }
    const std::string path =
        (std::filesystem::path(scenesDirectory) / *file).string();
    std::error_code error;
    const bool sameFile =
        std::filesystem::equivalent(path, currentScene.filePath, error);
    std::string name = text;
    name.erase(0, name.find_first_not_of(' '));
    name.erase(name.find_last_not_of(' ') + 1);
    if (std::filesystem::exists(path) && !sameFile) {
      replaceName = name;
      replacePath = path;
      prompt = {UiPrompt::Kind::Replace, *file};
      break;
    }
    if (saveSceneAs(path, name))
      close();
    else
      prompt.error = saveStatus;
    break;
  }
  case UiPrompt::Kind::Replace:
    if (saveSceneAs(replacePath, replaceName))
      close();
    else
      prompt = {UiPrompt::Kind::SaveAs, {}, false, false, saveStatus};
    break;
  case UiPrompt::Kind::None:
    break;
  }
}

void VulkanRenderer::stepPhysics(float frameSeconds) {
  if (!physics)
    return;
  if (physicsPanel.settingsChanged) {
    physicsPanel.settingsChanged = false;
    physics->applySettings(physicsPanel.config);
    // Only when the rate changed: a new clock forgets the time it was owed,
    // and rebuilding it every frame of a slider drag would stall physics.
    const double step = 1.0 / physicsPanel.stepsPerSecond;
    if (physicsClock.stepSeconds() != step ||
        physicsClock.maxStepsPerFrame() != physicsPanel.maxCatchUpSteps)
      physicsClock = FixedStepClock(step, physicsPanel.maxCatchUpSteps);
  }
  if (characterPanel.changed) {
    characterPanel.changed = false;
    if (player)
      player->setTuning(characterPanel.character, characterPanel.camera);
    updateEditorLines(); // the player start's outline is the body's size
  }
  const auto start = std::chrono::steady_clock::now();
  int steps;
  float alpha;
  if (!playing) {
    // Edit mode: bodies stay where the layout put them.
    steps = 0;
    physicsPanel.stepRequested = false;
    alpha = 1.0f;
  } else if (physicsPanel.paused) {
    // Paused: only the Step button advances, and the latest step is drawn
    // as is.
    steps = physicsPanel.stepRequested ? 1 : 0;
    physicsPanel.stepRequested = false;
    alpha = 1.0f;
  } else {
    steps = physicsClock.advance(frameSeconds);
    alpha = physicsClock.alpha();
  }
  const float stepSeconds = static_cast<float>(physicsClock.stepSeconds());
  if (player)
    player->control(playerControls);
  for (int i = 0; i < steps; ++i) {
    // The character moves first, then the bodies, like Jolt's samples.
    if (player)
      player->step(stepSeconds);
    physics->step(stepSeconds);
  }
  const auto stepped = std::chrono::steady_clock::now();
  // Runs on frames with no step too: alpha still moved, so the blend did.
  for (const PhysicsMovedPose &moved : physics->movedPoses(alpha))
    if (SceneNode *node = physicsLinks[moved.userData].node)
      node->setPose(moved.pose.position, moved.pose.rotation);
  if (player)
    posePlayer(alpha, frameSeconds);
  const auto posed = std::chrono::steady_clock::now();

  using Ms = std::chrono::duration<double, std::milli>;
  physicsLog.frames++;
  physicsLog.steps += steps;
  physicsLog.stepMs += Ms(stepped - start).count();
  physicsLog.poseMs += Ms(posed - stepped).count();
  if (physicsLog.frames == 120) {
    spdlog::info("[physics] bodies={} awake={} steps={} step={:.3f}ms "
                 "poses={:.3f}ms/frame dropped={}",
                 physics->bodyCount(), physics->activeBodyCount(),
                 physicsLog.steps,
                 physicsLog.steps ? physicsLog.stepMs / physicsLog.steps : 0.0,
                 physicsLog.poseMs / physicsLog.frames,
                 physics->droppedContactSteps());
    physicsLog = {};
  }
  physicsPanel.bodies = physics->bodyCount();
  physicsPanel.awakeBodies = physics->activeBodyCount();
  physicsPanel.droppedContactSteps = physics->droppedContactSteps();
  metrics.recordCpuPhase(
      PerformanceMetrics::CpuPhase::Physics,
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start)
          .count());
}

SceneModelEntry VulkanRenderer::thrownBallEntry() {
  SceneModelEntry ball;
  ball.shape = PrimitiveShape::Sphere;
  ball.radius = 0.2f;
  ball.color = glm::vec3(0.9f, 0.9f, 0.95f);
  ball.body = PhysicsMotion::Dynamic;
  ball.mass = 5.0f;
  ball.restitution = 0.3f;
  return ball;
}

SceneModelEntry VulkanRenderer::playerCapsuleEntry(float radius,
                                                   float height) {
  SceneModelEntry capsule;
  capsule.shape = PrimitiveShape::Capsule;
  capsule.radius = radius;
  capsule.height = height;
  capsule.color = glm::vec3(0.62f, 0.16f, 0.2f); // the UI's crimson
  return capsule;
}

void VulkanRenderer::ensurePlayerModels() {
  const CharacterTuning &tuning = characterPanel.character;
  const SceneModelEntry standing =
      playerCapsuleEntry(tuning.radius, tuning.height);
  const SceneModelEntry crouched =
      playerCapsuleEntry(tuning.radius, tuning.crouchHeight);
  // The Play after a change of body size needs new meshes, uploaded while
  // nothing in flight can be drawing.
  const bool newCharacter = characterAsset && characterModelId < 0;
  if (modelManager.findModel(primitiveModelKey(standing)) < 0 ||
      modelManager.findModel(primitiveModelKey(crouched)) < 0 || newCharacter)
    vkDeviceWaitIdle(device.getLogicalDevice());
  try {
    playerModelId = createPrimitiveModel(standing);
    crouchedPlayerModelId = createPrimitiveModel(crouched);
  } catch (const std::exception &e) {
    playerModelId = crouchedPlayerModelId = -1;
    spdlog::error("Couldn't make the character's capsule: {}", e.what());
  }
  if (newCharacter)
    createCharacterModel();
}

// The character's mesh as a model with its embedded texture, in the
// GPU-driven pass's shared buffer too, and handed to the skinning pass.
// Once: it doesn't change with the scene.
void VulkanRenderer::createCharacterModel() {
  const CharacterAsset &asset = *characterAsset;
  try {
    const Material material = textureManager.loadMaterialFromMemory(
        "character|" + asset.albedoName, asset.albedo, device, descriptorManager);
    characterModelId =
        modelManager.createModel("#character", asset.mesh, false, false, device,
                                 textureManager, descriptorManager, &material);
  } catch (const std::exception &e) {
    characterModelId = -1;
    spdlog::error("Couldn't make the character's model, keeping the capsule: {}",
                  e.what());
    return;
  }
  // The bind-pose sphere doesn't hold a crouch or a roll: one around the
  // whole height, wide enough for the arms out, so culling never drops him.
  const float height = asset.boundsMax.y - asset.boundsMin.y;
  modelManager.getModel(characterModelId)
      ->setBoundingSphere(glm::vec3(0.0f, asset.boundsMin.y + 0.5f * height, 0.0f),
                          0.75f * height);
  gpuDrivenGBufferPass.registerModelGeometry(characterModelId, modelManager);
  skinningPass.setMesh(asset.mesh.vertices, asset.skin,
                       asset.skeleton.bones.size());
}

float VulkanRenderer::characterScale() const {
  const float modelHeight = characterAsset->boundsMax.y - characterAsset->boundsMin.y;
  return modelHeight > 0.0f && player
             ? player->character().tuning().height / modelHeight
             : 1.0f;
}

// The animator's pose, or idle before there's one.
std::vector<glm::mat4> VulkanRenderer::characterPose() const {
  const Skeleton &skeleton = characterAsset->skeleton;
  if (animator)
    return modelSpace(skeleton, animator->pose());
  Pose pose = bindPose(skeleton);
  if (const CharacterClipData *idle = characterAsset->clip(ClipSlot::Idle))
    sampleClip(idle->clip, skeleton,
               static_cast<float>(characterClipTime) + idle->start, idle->loop,
               pose);
  return modelSpace(skeleton, pose);
}

// At the scene's [player], or on the ground below the fly camera facing its
// way, with the Character tab's tuning. Its capsule is a plain node: not in
// the scene file, not selectable.
void VulkanRenderer::spawnPlayer() {
  if (!physics || player)
    return;
  ensurePlayerModels();
  glm::vec3 feet = flyCameraPosition;
  float yaw;
  if (currentScene.player) {
    feet = currentScene.player->position;
    yaw = glm::radians(currentScene.player->yaw);
  } else {
    if (const std::optional<PhysicsRayHit> ground = physics->castRay(
            flyCameraPosition, glm::vec3(0.0f, -1.0f, 0.0f), 1000.0f))
      feet.y -= ground->distance;
    // Camera's yaw is LearnOpenGL's, in degrees: -90 looks down -z, where
    // the character's 0 does.
    yaw = -glm::radians(flyCameraYaw) - glm::half_pi<float>();
  }
  player = std::make_unique<Player>(*physics, feet, yaw,
                                    characterPanel.character,
                                    characterPanel.camera);
  const bool animated = characterModelId >= 0;
  if (animated)
    animator = std::make_unique<CharacterAnimator>(
        characterAsset->skeleton, characterAsset->library, characterScale());
  if (const int model = animated ? characterModelId : playerModelId; model >= 0) {
    auto node = std::make_unique<SceneNode>();
    node->setModelId(model);
    if (animated) {
      // Posing keeps the node's scale: the model file's units to the body's.
      NodeTransform scaled;
      scaled.scale = characterScale();
      node->setTransform(scaled);
    }
    playerNode = node.get();
    rootNode.addChild(std::move(node));
  }
  posePlayer(1.0f, 0.0f);
}

// The camera follows, and the capsule is drawn where the character is this
// frame: blended between the last two steps, the short one while crouched.
void VulkanRenderer::posePlayer(float alpha, float frameSeconds) {
  player->follow(alpha, frameSeconds);
  if (!playerNode)
    return;
  const CharacterController &character = player->character();
  const glm::vec3 feet = character.drawnPosition(alpha);
  skeletonLines.clear();
  if (characterModelId >= 0) {
    characterClipTime += frameSeconds;
    if (animator) {
      // What the controller is doing, in the character's own frame for
      // the directional clips: x to its right, y ahead.
      const glm::vec3 velocity = character.velocity();
      const float yaw = character.yaw();
      const glm::vec3 ahead = yawForward(yaw);
      const glm::vec3 right(std::cos(yaw), 0.0f, -std::sin(yaw));
      AnimatorInput input;
      input.speed = glm::length(glm::vec2(velocity.x, velocity.z));
      input.direction = glm::vec2(glm::dot(velocity, right), glm::dot(velocity, ahead));
      input.grounded = character.grounded();
      input.crouched = character.height() < character.tuning().height - 1e-3f;
      input.aiming = playerControls.aim;
      input.dodging = character.dodging();
      input.verticalSpeed = velocity.y;
      // The run, sprint and crouch walk play in full at the controller's
      // own speeds for them.
      input.runSpeed = character.tuning().runSpeed;
      input.sprintSpeed = character.tuning().sprintSpeed;
      input.crouchSpeed = character.tuning().crouchSpeed;
      animator->update(input, frameSeconds);
    }
    const std::vector<glm::mat4> model = characterPose();
    characterSkinning = skinningMatrices(characterAsset->skeleton, model);
    // Show skeleton draws the bones instead of the body.
    const bool skeleton = characterPanel.showSkeleton;
    const int shown = skeleton ? -1 : characterModelId;
    if (playerNode->getModelId() != shown)
      playerNode->setModelId(shown);
    // Feet on the capsule's feet, turned to its facing: Mixamo models face
    // +z, the character's yaw 0 faces -z.
    const glm::quat facing = glm::angleAxis(
        character.drawnYaw(alpha) + glm::pi<float>(), glm::vec3(0.0f, 1.0f, 0.0f));
    playerNode->setPose(feet, facing);
    if (skeleton)
      updateSkeletonLines(glm::translate(glm::mat4(1.0f), feet) *
                              glm::mat4_cast(facing) *
                              glm::scale(glm::mat4(1.0f), glm::vec3(characterScale())),
                          model);
    return;
  }
  const float height = character.height();
  const int model = height < character.tuning().height && crouchedPlayerModelId >= 0
                        ? crouchedPlayerModelId
                        : playerModelId;
  if (playerNode->getModelId() != model)
    playerNode->setModelId(model);
  // The capsule mesh is centred on its middle, the character's position is
  // its feet.
  playerNode->setPose(character.drawnPosition(alpha) +
                          glm::vec3(0.0f, 0.5f * height, 0.0f),
                      glm::angleAxis(character.drawnYaw(alpha),
                                     glm::vec3(0.0f, 1.0f, 0.0f)));
}

void VulkanRenderer::loadCharacter() {
  const std::string path = resolveCharacterPath("Resources/Characters/ganfaul.character");
  if (path.empty()) {
    spdlog::info("No Resources/Characters/ganfaul.character: the character is a capsule");
    return;
  }
  std::string error;
  try {
    characterAsset = importCharacter(CharacterIO::loadFromFile(path), error);
  } catch (const std::exception &e) {
    error = e.what();
  }
  if (!characterAsset)
    spdlog::warn("Character not loaded, keeping the capsule: {}", error);
  if (std::getenv("HELIORA_SHOW_SKELETON"))
    characterPanel.showSkeleton = true;
}

// Each bone joined to its parent, the rig's own bones only (not the file's
// root or mesh nodes).
void VulkanRenderer::updateSkeletonLines(const glm::mat4 &placement,
                                         const std::vector<glm::mat4> &model) {
  const Skeleton &skeleton = characterAsset->skeleton;
  // The UI's warm yellow (kGizmoHot, 0xf5d36b) as RGBA8 bytes.
  constexpr uint32_t kBoneColor = 0xf5u | 0xd3u << 8 | 0x6bu << 16 | 0xffu << 24;
  auto isRig = [&](int i) {
    return skeleton.bones[i].name.rfind("mixamorig", 0) == 0;
  };
  for (size_t i = 0; i < skeleton.bones.size(); ++i) {
    const int parent = skeleton.bones[i].parent;
    if (parent < 0 || !isRig(static_cast<int>(i)) || !isRig(parent))
      continue;
    skeletonLines.push_back({glm::vec3(placement * model[parent][3]), kBoneColor});
    skeletonLines.push_back({glm::vec3(placement * model[i][3]), kBoneColor});
  }
}

void VulkanRenderer::throwBall(glm::vec3 from, glm::vec3 direction) {
  if (!physics || ballModelId < 0 || !playing)
    return;
  SceneModelEntry ball = thrownBallEntry();
  // A metre ahead of the camera, so it doesn't start inside the near plane.
  ball.transform.position = from + direction * 1.0f;
  auto node = std::make_unique<SceneNode>();
  node->setModelId(ballModelId);
  node->setTransform(ball.transform);
  PhysicsBodyDesc body = bodyDescFor(ball, nullptr, physicsPanel.defaultFriction);
  body.linearVelocity = direction * 25.0f; // m/s, a hard throw
  body.userData = physicsLinks.size();
  const PhysicsBodyId id = physics->addBody(body);
  if (!id.isValid())
    return; // world full; PhysicsWorld already logged it
  node->setPhysicsLink(static_cast<int>(physicsLinks.size()));
  physicsLinks.push_back({node.get(), id});
  rootNode.addChild(std::move(node));
  ++physicsPanel.thrownBalls;
}

namespace {

// The UI's bright purple (kPurpleBright, 0x9a63b8) as RGBA8 bytes.
constexpr uint32_t kSelectionColor = 0x9au | 0x63u << 8 | 0xb8u << 16 |
                                     0xffu << 24;
// The player start while it isn't selected: the UI's bright red
// (kRedBright, 0xc93a50), close to the character's crimson.
constexpr uint32_t kPlayerStartColor = 0xc9u | 0x3au << 8 | 0x50u << 16 |
                                       0xffu << 24;

// Corner i has bit 0 set for max x, bit 1 for max y, bit 2 for max z.
std::array<glm::vec3, 8> boxCorners(glm::vec3 lo, glm::vec3 hi,
                                    const glm::mat4 &transform) {
  std::array<glm::vec3, 8> corners;
  for (int i = 0; i < 8; ++i)
    corners[i] = glm::vec3(
        transform * glm::vec4(i & 1 ? hi.x : lo.x, i & 2 ? hi.y : lo.y,
                              i & 4 ? hi.z : lo.z, 1.0f));
  return corners;
}

// The 12 edges: each corner joined to the corners one axis over.
void appendBoxLines(std::vector<PhysicsDebugVertex> &lines,
                    const std::array<glm::vec3, 8> &corners, uint32_t color) {
  for (int i = 0; i < 8; ++i)
    for (int axis = 1; axis < 8; axis <<= 1)
      if (!(i & axis)) {
        lines.push_back({corners[i], color});
        lines.push_back({corners[i | axis], color});
      }
}

float largestScale(const glm::mat4 &m) {
  return std::max({glm::length(glm::vec3(m[0])), glm::length(glm::vec3(m[1])),
                   glm::length(glm::vec3(m[2]))});
}

} // namespace

bool VulkanRenderer::entryHasBody(SceneEntryRef ref) const {
  if (ref.kind == SceneEntryRef::Kind::Model)
    return currentScene.models[ref.index].body.has_value();
  if (ref.kind == SceneEntryRef::Kind::Grid)
    return currentScene.grids[ref.index].model.body.has_value();
  return false;
}

void VulkanRenderer::selectAt(glm::vec2 windowPixel) {
  if (playing || !physics || !sceneArea.contains(windowPixel))
    return;
  // Unjittered, so the ray matches what's on screen on average. The area's
  // size, not the images', so it also matches while the picture is
  // stretched during a resize.
  const PickRay ray =
      rayFromScreen(sceneArea.toScene(windowPixel), sceneArea.size,
                    glm::inverse(taaBaseProjection * sceneUbo.view));
  std::optional<SceneEntryRef> picked;
  float nearest = imguiDrawDistance;
  if (const std::optional<PhysicsRayHit> hit =
          physics->castRay(ray.origin, ray.direction, nearest)) {
    nearest = hit->distance;
    if (const SceneNode *node = physicsLinks[hit->userData].node)
      picked = node->entry();
  }
  // Objects without a body have no collider to hit, so their bounding sphere
  // stands in. It's rough, but these are rare in a physics game.
  for (const auto &node : rootNode.getChildren()) {
    const SceneEntryRef ref = node->entry();
    if (ref.kind == SceneEntryRef::Kind::None || entryHasBody(ref))
      continue;
    const MeshModel *model = modelManager.getModel(node->getModelId());
    if (!model)
      continue;
    const glm::mat4 &m = node->getGlobalTransform();
    const std::optional<float> distance = rayHitsSphere(
        ray, glm::vec3(m * glm::vec4(model->boundingCenter, 1.0f)),
        model->boundingRadius * largestScale(m));
    if (distance && *distance < nearest) {
      nearest = *distance;
      picked = ref;
    }
  }
  // The player start has no body or mesh, only the character's capsule.
  if (currentScene.player) {
    const CharacterTuning &tuning = characterPanel.character;
    const std::optional<float> distance = rayHitsPlayerStart(
        ray, *currentScene.player, tuning.radius, tuning.height);
    if (distance && *distance < nearest) {
      nearest = *distance;
      picked = SceneEntryRef{SceneEntryRef::Kind::Player, 0};
    }
  }
  if (picked && picked->kind == SceneEntryRef::Kind::None)
    picked.reset(); // a thrown ball, which isn't part of the layout
  selection = picked;
  updateEditorLines();
  if (!selection)
    spdlog::info("Selection cleared");
  else if (selection->kind == SceneEntryRef::Kind::Player)
    spdlog::info("Selected [player]");
  else
    spdlog::info("Selected {} {}",
                 selection->kind == SceneEntryRef::Kind::Model ? "[model]"
                                                               : "[grid]",
                 selection->index);
}

// The player start, always there in edit mode, and the selection's
// outline.
void VulkanRenderer::updateEditorLines() {
  editorLines.clear();
  selectionMass = 0.0f;
  selectionBodies = 0;
  const bool startSelected =
      selection && selection->kind == SceneEntryRef::Kind::Player;
  if (currentScene.player) {
    const CharacterTuning &tuning = characterPanel.character;
    appendPlayerStartLines(editorLines, *currentScene.player, tuning.radius,
                           tuning.height,
                           startSelected ? kSelectionColor : kPlayerStartColor);
  }
  if (!selection || startSelected)
    return;
  for (const PhysicsLink &link : physicsLinks)
    if (link.node && link.node->entry() == *selection) {
      if (selectionBodies++ == 0)
        selectionMass = physics->mass(link.body);
    }
  // A centimetre out, so the outline doesn't sit in the object's own faces.
  constexpr float kMargin = 0.01f;
  // One object gets a box turned with it; a grid gets one box around every
  // copy, lined up with the world.
  glm::vec3 lo(std::numeric_limits<float>::max());
  glm::vec3 hi(std::numeric_limits<float>::lowest());
  bool any = false;
  for (const auto &node : rootNode.getChildren()) {
    if (node->entry() != *selection)
      continue;
    const std::optional<ModelManager::Bounds> bounds =
        modelManager.localBounds(node->getModelId());
    if (!bounds)
      continue;
    const std::array<glm::vec3, 8> corners =
        boxCorners(bounds->min - kMargin, bounds->max + kMargin,
                   node->getGlobalTransform());
    if (selection->kind == SceneEntryRef::Kind::Model) {
      appendBoxLines(editorLines, corners, kSelectionColor);
      return;
    }
    for (const glm::vec3 &corner : corners) {
      lo = glm::min(lo, corner);
      hi = glm::max(hi, corner);
    }
    any = true;
  }
  if (any)
    appendBoxLines(editorLines, boxCorners(lo, hi, glm::mat4(1.0f)),
                   kSelectionColor);
}

// The copy goes one width of the original's box (around every copy, for a
// grid) along the world axis nearest the camera's right, so it shows up
// beside it. Its nodes and bodies are added like a spawn's; one undo step.
// Refused when the physics world has no room for its bodies.
void VulkanRenderer::applyDuplicate() {
  if (playing || !selection || !physics)
    return;
  const SceneEntryRef ref = *selection;
  const bool isGrid = ref.kind == SceneEntryRef::Kind::Grid;
  if (ref.kind != SceneEntryRef::Kind::Model && !isGrid)
    return;
  glm::vec3 lo(std::numeric_limits<float>::max());
  glm::vec3 hi(std::numeric_limits<float>::lowest());
  bool any = false;
  for (const auto &node : rootNode.getChildren()) {
    if (node->entry() != ref)
      continue;
    const std::optional<ModelManager::Bounds> bounds =
        modelManager.localBounds(node->getModelId());
    if (!bounds)
      continue;
    for (const glm::vec3 &corner :
         boxCorners(bounds->min, bounds->max, node->getGlobalTransform())) {
      lo = glm::min(lo, corner);
      hi = glm::max(hi, corner);
    }
    any = true;
  }
  if (!any)
    return;
  uint32_t bodies = 0;
  if (entryHasBody(ref)) {
    const glm::ivec3 n = isGrid ? currentScene.grids[ref.index].count
                                : glm::ivec3(1);
    bodies = uint32_t(n.x) * uint32_t(n.y) * uint32_t(n.z);
  }
  if (physics->bodyCount() + bodies > physics->bodyLimit()) {
    sceneError = std::format("No room for a copy: the physics world holds {} "
                             "bodies",
                             physics->bodyLimit());
    spdlog::warn("{}", sceneError);
    return;
  }
  // The camera's right, out of the inverse view.
  const std::optional<SceneEntryRef> copy =
      duplicateEntry(currentScene, ref,
                     duplicateOffset(lo, hi, glm::vec3(sceneUbo.invView[0])));
  if (!copy)
    return;
  addEntry(*copy);
  physics->optimizeBroadPhase();
  rootNode.update(glm::mat4(1.0f), playSeconds);
  selection = copy;
  updateEditorLines();
  recordEdit();
  sceneError.clear();
  spdlog::info("Duplicated {} {} as {}", isGrid ? "[grid]" : "[model]",
               ref.index, copy->index);
}

void VulkanRenderer::updateScene(float frameSeconds) {
  if (playing)
    playSeconds += frameSeconds;
  rootNode.update(glm::mat4(1.0f), playSeconds);
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

void VulkanRenderer::updateMaterialProbePixel() {
  double cursorX = 0.0;
  double cursorY = 0.0;
  glfwGetCursorPos(window, &cursorX, &cursorY);
  const glm::vec2 cursor(cursorX, cursorY);
  // Off the scene reads nothing (the probe skips a pixel past the edge).
  const glm::uvec2 none(sceneExtent.width, sceneExtent.height);
  if (sceneExtent.width == 0 || sceneExtent.height == 0 ||
      !sceneArea.contains(cursor)) {
    materialProbePixel = none;
    return;
  }
  // The area can be a different size from the images for a few frames
  // while it's resized; the picture is stretched to fit it.
  const glm::vec2 pixel = sceneArea.toScene(cursor) *
                          glm::vec2(none) / sceneArea.size;
  materialProbePixel = glm::min(glm::uvec2(pixel), none - 1u);
}

// Draw

// One frame. The UI goes first, so everything recorded afterwards sees this
// frame's edits; then the GPU work. Each step's CPU time goes to the
// Performance panel.
void VulkanRenderer::draw() {
  using Clock = std::chrono::high_resolution_clock;
  using Phase = PerformanceMetrics::CpuPhase;
  Clock::time_point start;
  auto begin = [&] { start = Clock::now(); };
  auto end = [&](Phase phase) {
    metrics.recordCpuPhase(
        phase,
        std::chrono::duration<double, std::milli>(Clock::now() - start).count());
  };

  // The frame time shown is draw() from here to the end; physics and input
  // run before it, in main.
  metrics.beginFrame();
  updateSceneArea();
  begin();
  buildUi();
  end(Phase::Ui);
  applyPendingChanges();

  begin();
  waitForFrameSlot();
  end(Phase::WaitFence);
  // This slot's last frame is done: its timestamps and probe can be read.
  metrics.collectGpuResults(device.getLogicalDevice(), currentFrame);
  materialProbeResult = materialProbePass.read(currentFrame);

  // Check resize BEFORE acquiring: a successful acquire signals
  // imageAvailable, and returning early without consuming it would leave it
  // signalled on the next frame.
  if (framebufferResized) {
    framebufferResized = false;
    recreateSwapChain();
    return;
  }
  uint32_t imageIndex = 0;
  begin();
  const bool acquired = acquireSwapImage(imageIndex);
  end(Phase::Acquire);
  if (!acquired)
    return;
  begin();
  waitForSwapImage(imageIndex);
  end(Phase::ImageFence);

  begin();
  updateFrameUniforms();
  end(Phase::Update);

  metrics.setActiveGpuQueryFrame(currentFrame);
  begin();
  recordFrame(imageIndex);
  end(Phase::Record);

  begin();
  descriptorManager.updateUniformBuffer(device.getAllocator(), imageIndex,
                                        &sceneUbo, sizeof(SceneUniformBuffer));
  end(Phase::Upload);

  begin();
  submitFrame(imageIndex);
  end(Phase::Submit);
  metrics.markGpuQueriesSubmitted(currentFrame);

  begin();
  const bool presented = presentFrame(imageIndex);
  end(Phase::Present);
  if (!presented)
    return;
  currentFrame = (currentFrame + 1) % MAX_FRAMES_DRAWS;
  // Advance TAA frame counter AFTER this frame finishes recording so the
  // Halton jitter index used above matches the history ring slots that
  // recordPostCommands picked for framebuffers / descriptor sets.
  taaFrameCounter++;
  metrics.endFrame();
}

// What the UI and the editor asked for since the last frame, applied before
// any of this frame's GPU work. The ones that replace GPU resources wait for
// the device to go idle first, so nothing in flight still uses the old ones.
void VulkanRenderer::applyPendingChanges() {
  VkDevice logicalDevice = device.getLogicalDevice();
  // Apply a UI-requested scene switch before any of this frame's work
  // starts. loadSceneAt() waits for device idle, so nothing recorded or
  // submitted this frame can reference torn-down scene resources.
  if (pendingSceneIndex >= 0) {
    int requested = pendingSceneIndex;
    pendingSceneIndex = -1;
    loadSceneAt(requested);
  }
  if (!pendingEntryOps.empty()) {
    vkDeviceWaitIdle(logicalDevice);
    for (const auto &[op, ref] : std::exchange(pendingEntryOps, {}))
      applyEntryOp(op, ref);
    rootNode.update(glm::mat4(1.0f), playSeconds);
    updateEditorLines();
  }
  if (pendingSpawn) {
    vkDeviceWaitIdle(logicalDevice);
    SceneModelEntry entry = std::move(*pendingSpawn);
    pendingSpawn.reset();
    spawnEntry(std::move(entry));
  }
  if (std::exchange(pendingPlayerStart, false))
    placePlayerStart();
  if (std::exchange(duplicateRequested, false)) {
    vkDeviceWaitIdle(logicalDevice);
    applyDuplicate();
  }
  // The scene's images rebuilt once a new size has settled (a divider let
  // go, a window resize finished).
  {
    const bool mouseDown =
        glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS ||
        glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS ||
        glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    if (const std::optional<glm::ivec2> size =
            sceneResize->update(glm::ivec2(sceneArea.size), mouseDown))
      recreateSceneTargets({uint32_t(size->x), uint32_t(size->y)});
  }
  if (historyRequest != HistoryStep::None)
    applyHistoryStep();
  if (playToggleRequested) {
    playToggleRequested = false;
    playing = !playing;
    // Stop: back to the layout. Thrown balls, the character and everything
    // physics moved go, since the layout never had them.
    if (!playing)
      loadScene(currentScene, /*moveCamera=*/false);
    else
      spawnPlayer();
  }
}

// This frame slot's previous frame has to finish before its fence, acquire
// semaphore, query range and exposure buffer are reused. When the CPU runs
// ahead of the GPU this is where it waits: idle time, not renderer cost.
void VulkanRenderer::waitForFrameSlot() {
  vkWaitForFences(device.getLogicalDevice(), 1, &drawFences[currentFrame],
                  VK_TRUE, std::numeric_limits<uint64_t>::max());
}

// False when the swapchain was out of date and got rebuilt instead; the
// frame is skipped.
bool VulkanRenderer::acquireSwapImage(uint32_t &imageIndex) {
  const VkResult result = vkAcquireNextImageKHR(
      device.getLogicalDevice(), swapchain.getSwapchain(),
      std::numeric_limits<uint64_t>::max(), imageAvailable[currentFrame],
      VK_NULL_HANDLE, &imageIndex);
  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    // Failed acquire: the semaphore isn't signalled, so it's safe to rebuild.
    recreateSwapChain();
    return false;
  }
  if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    throw std::runtime_error("Failed to acquire swap chain image");
  return true;
}

// Waits if another frame slot still renders into this swap image, then
// hands the image to this slot's fence.
void VulkanRenderer::waitForSwapImage(uint32_t imageIndex) {
  VkDevice logicalDevice = device.getLogicalDevice();
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

}

// Everything the shaders read this frame: sun and fog, the render settings,
// TAA's jitter, the shadow matrices.
void VulkanRenderer::updateFrameUniforms() {
  applySunAndFog();
  updateDayNightSun();
  packRenderSettings();
  applyTaaJitter();
  shadowPass.updateLightSpaceMatrices(sceneUbo, imguiCsmFar,
                                      imguiDrawDistance);
  shadowPass.updatePointShadowMatrices(sceneUbo);
  updateMaterialProbePixel();
}

// The day/night cycle, when it's on: the hour advances at imguiDayNightSpeed
// sim-hours per real second and the sun follows an arc through the zenith.
// Sunrise at hour 6 (east horizon), zenith at 12, sunset at 18 (west); below
// the horizon outside 6 to 18. Off, the ambient slider drives the sky light
// directly.
void VulkanRenderer::updateDayNightSun() {
  if (!imguiDayNightEnable) {
    sceneUbo.qualityToggles2.w = imguiIblIntensity;
    return;
  }
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
  // Peak 2.5 (was 5.5), see SceneSun. Stopgap until histogram auto-exposure
  // replaces the manual EV slider.
  sceneUbo.directionalLight.colorIntensity =
      glm::vec4(color, 2.5f * dayFactor + 0.05f);

  // IBL/sky baked from a daytime cubemap — attenuate it with the same
  // day factor so glossy surfaces don't keep reflecting baked daylight
  // through the night. User's manual slider still scales on top.
  sceneUbo.qualityToggles2.w =
      imguiIblIntensity * (0.08f + 0.92f * dayFactor);
}

// The Render and Lighting panels' settings, packed into the UBO's toggles.
void VulkanRenderer::packRenderSettings() {
  sceneUbo.qualityToggles2.y =
      imguiUseGeomNormalOnly ? 0.0f : imguiNormalStrength;
  sceneUbo.qualityToggles2.x =
      autoExposurePass.updateExposureScale(currentFrame, autoExpEnabled,
                                           imguiExposureEV);
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
  sceneUbo.shadowParams.x    = imguiPointShadowFar;
  sceneUbo.shadowParams.z    = imguiSharpness;
  sceneUbo.shadowParams.w    = imguiSsrEnabled ? 1.0f : 0.0f;
  sceneUbo.visualToggles =
      glm::vec4(imguiEdgeAA, imguiAlphaDither, imguiShadowSoftness,
                imguiContactGrounding);

}

// TAA's per-frame state: this frame's sub-pixel jitter, last frame's matrix
// for reprojection, and whether the history can be blended.
void VulkanRenderer::applyTaaJitter() {
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
  VkExtent2D ext = sceneExtent;
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

}

// The frame's four command buffers, from one collection of the scene's
// meshes.
void VulkanRenderer::recordFrame(uint32_t imageIndex) {
  collectMeshInstances();
  recordGBufferCommands(imageIndex);
  recordSsaoComputeCommands(ssaoCommandBuffers[imageIndex], imageIndex);
  recordShadowCommands(shadowCommandBuffers[imageIndex]);
  recordPostCommands(postCommandBuffers[imageIndex], imageIndex);
}

// Four submits: the G-buffer, SSAO on the compute queue once the G-buffer
// is done, the shadows, and the post chain once SSAO is done, which
// signals the frame's fence and the semaphore present waits on.
void VulkanRenderer::submitFrame(uint32_t imageIndex) {
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
  metrics.recordQueueSubmit();
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
  metrics.recordQueueSubmit();
  if (vkQueueSubmit(device.getComputeQueue(), 1, &ssaoSubmit,
                    VK_NULL_HANDLE) != VK_SUCCESS) {
    throw std::runtime_error("Failed to submit SSAO compute command buffer");
  }

  VkSubmitInfo shadowSubmit = {};
  shadowSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  shadowSubmit.commandBufferCount = 1;
  shadowSubmit.pCommandBuffers = &shadowCmd;
  metrics.recordQueueSubmit();
  if (vkQueueSubmit(device.getGraphicsQueue(), 1, &shadowSubmit,
                    VK_NULL_HANDLE) != VK_SUCCESS) {
    throw std::runtime_error("Failed to submit shadow command buffer");
  }

  uint64_t renderFinishedSignalValue = 0; // binary signal, ignored
  // TRANSFER for the copy into the swap image, the first thing that writes it.
  VkPipelineStageFlags postWaitStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                       VK_PIPELINE_STAGE_TRANSFER_BIT;
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
  metrics.recordQueueSubmit();
  if (vkQueueSubmit(device.getGraphicsQueue(), 1, &postSubmit,
                    drawFences[currentFrame]) != VK_SUCCESS) {
    throw std::runtime_error("Failed to submit post command buffer");
  }
}

// False when the swapchain had to be rebuilt.
bool VulkanRenderer::presentFrame(uint32_t imageIndex) {
  VkPresentInfoKHR presentInfo = {};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &renderFinished[imageIndex];
  presentInfo.swapchainCount = 1;
  VkSwapchainKHR sc = swapchain.getSwapchain();
  presentInfo.pSwapchains = &sc;
  presentInfo.pImageIndices = &imageIndex;

  const VkResult result =
      vkQueuePresentKHR(device.getPresentationQueue(), &presentInfo);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
      framebufferResized) {
    framebufferResized = false;
    recreateSwapChain();
    return false;
  }
  if (result != VK_SUCCESS)
    throw std::runtime_error("Failed to present swap chain image");
  return true;
}

// Swapchain recreation

void VulkanRenderer::recreateSwapChain() {
  // Only what the window owns: its images, the UI's framebuffers and the
  // per-image sync and uniform buffers. The scene's images follow the
  // scene's area, which updateSceneArea and sceneResize look after, unless
  // the number of swap images changed: everything per image is remade then.
  vkDeviceWaitIdle(device.getLogicalDevice());
  const size_t oldImageCount = swapchain.getImageCount();
  for (size_t i = 0; i < oldImageCount; ++i)
    renderResources.unregisterImage(swapchain.getSwapImage(i));
  cleanupAsyncFrameCommandBuffers();
  swapchain.recreate(device, window);
  const bool imageCountChanged = swapchain.getImageCount() != oldImageCount;
  if (imageCountChanged) {
    cleanupSceneTargets();
    renderResources.clear();
  }
  registerSwapchainResources();

  // Present-mode toggles commonly change minImageCount; every per-image
  // descriptor resource must track the new count or acquired indices run
  // past the init-time arrays.
  {
    QueueFamilyIndices queueFamilies = device.getQueueFamilies();
    uint32_t sharedUboFamilies[2] = {
        static_cast<uint32_t>(queueFamilies.graphicsFamily),
        static_cast<uint32_t>(queueFamilies.computeFamily)};
    const bool shareUbo = device.hasDedicatedComputeQueue();
    if (descriptorManager.recreatePerImageResources(
            device.getLogicalDevice(), device.getAllocator(),
            swapchain.getImageCount(), shareUbo ? sharedUboFamilies : nullptr,
            shareUbo ? 2u : 0u)) {
      descriptorManager.updateShadowMapDescriptor(
          device.getLogicalDevice(), shadowPass.csmView(),
          shadowPass.pointCubeView(), shadowPass.csmSampler(),
          shadowPass.pointSampler());
      descriptorManager.updateIblDescriptors(
          device.getLogicalDevice(),
          textureManager.getImageView(irradianceImageIndex), iblSampler,
          textureManager.getImageView(prefilteredEnvImageIndex),
          textureManager.getImageView(brdfLutImageIndex),
          textureManager.getImageView(ssaoNoiseImageIndex), ssaoNoiseSampler,
          textureManager.getImageView(iblSkyboxImageIndex));
    }
  }

  imguiLayer.createFramebuffers(device.getLogicalDevice(),
                                renderPassManager.getImGuiRenderPass(),
                                swapchain.getExtent(), swapchain.getImages());
  if (imageCountChanged)
    createSceneTargets();
  updateSceneArea();

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

  createAsyncFrameCommandBuffers();
}

void VulkanRenderer::registerSwapchainResources() {
  const size_t count = swapchain.getImageCount();
  // Resizing wipes every frame's targets, so only when the count changed;
  // the scene's images are filled in by createSceneTargets.
  if (renderResources.frameCount() != count)
    renderResources.resizeFrames(count);

  const QueueFamilyIndices queues = device.getQueueFamilies();
  const uint32_t graphicsFamily = static_cast<uint32_t>(queues.graphicsFamily);
  const VkSharingMode swapSharing =
      queues.graphicsFamily != queues.presentationFamily
          ? VK_SHARING_MODE_CONCURRENT
          : VK_SHARING_MODE_EXCLUSIVE;

  for (size_t i = 0; i < count; ++i) {
    // Freshly created swapchain images are in UNDEFINED, not PRESENT_SRC —
    // they have never been presented. Registering PRESENT_SRC would make a
    // future RenderResources::transition() emit an invalid oldLayout.
    renderResources.registerImage(
        {"swapchain[" + std::to_string(i) + "]", swapchain.getSwapImage(i),
         swapchain.getImageFormat(), VK_IMAGE_ASPECT_COLOR_BIT, 1, 1,
         VK_IMAGE_LAYOUT_UNDEFINED, swapSharing, graphicsFamily});
    renderResources.frame(i).swapchainImage = swapchain.getSwapImage(i);
  }
}

namespace {

AllocatedImage makeSceneImage(VmaAllocator allocator, VkExtent2D extent,
                              VkFormat format, VkImageUsageFlags usage) {
  VkImageCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.extent = {extent.width, extent.height, 1};
  ci.mipLevels = 1;
  ci.arrayLayers = 1;
  ci.format = format;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  ci.usage = usage;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VmaAllocationCreateInfo aci = {};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  aci.flags = RENDER_DEVICE_ALLOCATION_FLAGS;
  VkImage image = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  if (vmaCreateImage(allocator, &ci, &aci, &image, &allocation, nullptr) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create a scene image");
  return AllocatedImage(allocator, image, allocation);
}

ImageViewHandle makeColorView(VkDevice device, VkImage image,
                              VkFormat format) {
  VkImageViewCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  ci.image = image;
  ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  ci.format = format;
  ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkImageView view = VK_NULL_HANDLE;
  if (vkCreateImageView(device, &ci, nullptr, &view) != VK_SUCCESS)
    throw std::runtime_error("Failed to create a scene image view");
  return ImageViewHandle(device, view);
}

} // namespace

// The composite's two images per swap image: the HDR colour buffer between
// the SSR and TAA subpasses, and the finished picture that's copied into the
// window.
void VulkanRenderer::createSceneColorImages() {
  const size_t count = swapchain.getImageCount();
  const uint32_t graphicsFamily =
      static_cast<uint32_t>(device.getQueueFamilies().graphicsFamily);
  sceneColorImages.resize(count);
  sceneColorViews.resize(count);
  sceneOutputImages.resize(count);
  sceneOutputViews.resize(count);
  for (size_t i = 0; i < count; ++i) {
    // SAMPLED is needed for TAA's 3×3 neighborhood clamp: the tonemap subpass
    // reads the center pixel via subpassInput (zero-overhead, tile-local) and
    // simultaneously samples 8 neighbors via a normal sampler binding. Within
    // a single subpass it is legal to bind the same view as both input
    // attachment AND combined-image-sampler — both read at SHADER_READ_ONLY,
    // no writes are happening in this subpass, so there is no feedback loop.
    sceneColorImages[i] = makeSceneImage(
        device.getAllocator(), sceneExtent, sceneColorFormat,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    sceneColorViews[i] = makeColorView(device.getLogicalDevice(),
                                       sceneColorImages[i].get(),
                                       sceneColorFormat);
    sceneOutputImages[i] = makeSceneImage(
        device.getAllocator(), sceneExtent, swapchain.getImageFormat(),
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    sceneOutputViews[i] = makeColorView(device.getLogicalDevice(),
                                        sceneOutputImages[i].get(),
                                        swapchain.getImageFormat());

    renderResources.registerImage(
        {"compositeColor[" + std::to_string(i) + "]",
         sceneColorImages[i].get(), sceneColorFormat,
         VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, VK_IMAGE_LAYOUT_UNDEFINED,
         VK_SHARING_MODE_EXCLUSIVE, graphicsFamily});
    renderResources.registerImage(
        {"sceneOutput[" + std::to_string(i) + "]", sceneOutputImages[i].get(),
         swapchain.getImageFormat(), VK_IMAGE_ASPECT_COLOR_BIT, 1, 1,
         VK_IMAGE_LAYOUT_UNDEFINED, VK_SHARING_MODE_EXCLUSIVE,
         graphicsFamily});
    FrameRenderTargets &targets = renderResources.frame(i);
    targets.colorBuffer = sceneColorImages[i].get();
    targets.sceneOutput = sceneOutputImages[i].get();
  }
}

void VulkanRenderer::createSceneTargets() {
  createSceneColorImages();
  createLitResources();
  bloomPass.create(device, sceneExtent, swapchain.getImageCount(), litViews,
                   pipelineCache.get());
  // SSGI's bounce images are sampled by lit.frag, so the G-buffer
  // descriptor set (made in createGBuffer) points at them.
  ssgiPass.create(device, sceneExtent, litFormat,
                  renderPassManager.getSsgiRenderPass(), MAX_FRAMES_DRAWS + 1);
  ssaoPass.create(device, sceneExtent, swapchain.getImageCount(),
                  descriptorManager, pipelineCache.get());
  createTaaResources();
  {
    std::vector<VkImageView> outputViews;
    std::vector<VkImageView> colorViews;
    std::vector<VkImageView> historyViews;
    for (size_t i = 0; i < swapchain.getImageCount(); ++i) {
      outputViews.push_back(sceneOutputViews[i].get());
      colorViews.push_back(sceneColorViews[i].get());
    }
    for (const ImageViewHandle &view : taaHistoryViews)
      historyViews.push_back(view.get());
    compositePass.create(device.getLogicalDevice(),
                         renderPassManager.getRenderPass(), sceneExtent,
                         outputViews, colorViews, historyViews);
  }
  createGBuffer();
  createLitFramebuffers();
  gpuDrivenGBufferPass.create(device, modelManager, descriptorManager,
                              renderPassManager.getGBufferRenderPass(),
                              sceneExtent, gBufferDepthViews,
                              pipelineCache.get());
  // After lit: its descriptors point at litViews.
  autoExposurePass.create(device, sceneExtent, litViews, litSampler,
                          pipelineCache.get());

  std::vector<VkImageView> colorViews;
  for (const ImageViewHandle &view : sceneColorViews)
    colorViews.push_back(view.get());
  descriptorManager.recreateInputSets(device.getLogicalDevice(), colorViews);
  // TAA: history-prev + depth + colorBuffer-current + bloom pyramid result.
  {
    std::vector<VkImageView> histViews;
    std::vector<VkImageView> depthViews;
    for (const auto &v : taaHistoryViews)
      histViews.push_back(v.get());
    for (const auto &v : gBufferDepthViews)
      depthViews.push_back(v.get());
    descriptorManager.recreateTaaSets(device.getLogicalDevice(), histViews,
                                      depthViews, colorViews,
                                      bloomPass.mip0Views(), taaSampler);
    debugLinePass.setDepthViews(device.getLogicalDevice(), depthViews,
                                taaSampler);
  }
  rebuildProjection();
  taaHistoryValid = false; // nothing to blend with at a new size
}

void VulkanRenderer::cleanupSceneTargets() {
  autoExposurePass.cleanup();
  compositePass.cleanup();
  gpuDrivenGBufferPass.cleanup();
  cleanupLitFramebuffers();
  cleanupGBuffer();
  ssaoPass.cleanup();
  bloomPass.cleanup();
  ssgiPass.cleanup();
  cleanupLitResources();
  cleanupTaaResources();
  sceneColorViews.clear();
  sceneColorImages.clear();
  sceneOutputViews.clear();
  sceneOutputImages.clear();
}

// A rebuild at a new size, once the scene's area has settled. Rare and
// user-driven, so a full idle wait is the simple way to retire every frame
// still using the old images.
void VulkanRenderer::recreateSceneTargets(VkExtent2D extent) {
  vkDeviceWaitIdle(device.getLogicalDevice());
  const auto start = std::chrono::steady_clock::now();
  cleanupSceneTargets();
  // The old images' entries go with them; the swapchain's come straight
  // back, and the new scene images register as they're made.
  renderResources.clear();
  registerSwapchainResources();
  sceneExtent = extent;
  createSceneTargets();
  spdlog::info("Scene images rebuilt at {}x{} in {:.1f} ms", extent.width,
               extent.height,
               std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
                   .count());
}

// Where the scene goes in the window: the space the docked panels left last
// frame, or the whole window with the UI hidden. Window pixels are the
// swapchain's pixels here (the game runs on X11, which doesn't scale).
void VulkanRenderer::updateSceneArea() {
  const VkExtent2D window = swapchain.getExtent();
  const glm::vec2 windowSize(window.width, window.height);
  sceneArea = {glm::vec2(0.0f), windowSize};
  if (const std::optional<SceneViewport> docked =
          imguiLayer.layoutSceneArea()) {
    const glm::vec2 lo = glm::clamp(docked->origin, glm::vec2(0.0f),
                                    windowSize);
    const glm::vec2 hi =
        glm::clamp(docked->origin + docked->size, lo, windowSize);
    sceneArea = {glm::floor(lo), glm::floor(hi) - glm::floor(lo)};
  }
}

// The scene's finished picture into its area of the window, stretched if
// the area has changed size since its images were made. The rest of the
// window is cleared for the UI to draw over.
void VulkanRenderer::copySceneToWindow(VkCommandBuffer cmd,
                                       uint32_t imageIndex) {
  if (imageIndex >= renderResources.frameCount() ||
      imageIndex >= sceneOutputImages.size())
    return;
  const VkImage window = renderResources.frame(imageIndex).swapchainImage;
  const VkExtent2D windowSize = swapchain.getExtent();
  // TRANSFER is in the post submit's wait stages, so this chains to the
  // swap image being acquired.
  renderResources.transition(cmd, window, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             VK_ACCESS_TRANSFER_WRITE_BIT);

  const glm::ivec2 limit(windowSize.width, windowSize.height);
  const glm::ivec2 lo = glm::clamp(glm::ivec2(sceneArea.origin),
                                   glm::ivec2(0), limit);
  const glm::ivec2 hi =
      glm::clamp(glm::ivec2(sceneArea.origin + sceneArea.size), lo, limit);
  const bool coversWindow = lo == glm::ivec2(0) && hi == limit;
  if (!coversWindow) {
    const VkClearColorValue background = {{0.05f, 0.045f, 0.06f, 1.0f}};
    const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                                           1};
    vkCmdClearColorImage(cmd, window, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         &background, 1, &range);
    // The blit writes over part of what the clear wrote.
    VkMemoryBarrier2 afterClear = {};
    afterClear.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    afterClear.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    afterClear.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    afterClear.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
    afterClear.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    VkDependencyInfo dependency = {};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &afterClear;
    vkCmdPipelineBarrier2(cmd, &dependency);
  }
  if (hi.x > lo.x && hi.y > lo.y) {
    VkImageBlit blit = {};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {int32_t(sceneExtent.width),
                          int32_t(sceneExtent.height), 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[0] = {lo.x, lo.y, 0};
    blit.dstOffsets[1] = {hi.x, hi.y, 1};
    // Same size is a straight copy; a stretched one gets filtered.
    const bool sameSize = uint32_t(hi.x - lo.x) == sceneExtent.width &&
                          uint32_t(hi.y - lo.y) == sceneExtent.height;
    vkCmdBlitImage(cmd, sceneOutputImages[imageIndex].get(),
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, window,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                   sameSize ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);
  }
  // The UI pass loads what's there and draws around and over it.
  renderResources.transition(
      cmd, window, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
          VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
}

void VulkanRenderer::requestCapture(const std::string &path) {
  if (!swapchain.canCopyFrom()) {
    spdlog::error("This GPU can't copy out of the window's images, so there's "
                  "no capture");
    return;
  }
  capturePath = path;
  captureState = CaptureState::Requested;
}

// The whole window as it will be shown: the scene, the editor's lines and
// any UI. Out of PRESENT_SRC for the copy and back, and a barrier that makes
// the copy visible to the CPU.
void VulkanRenderer::recordCapture(VkCommandBuffer cmd, uint32_t imageIndex) {
  const VkImage image = renderResources.frame(imageIndex).swapchainImage;
  renderResources.transition(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                             VK_ACCESS_TRANSFER_READ_BIT);
  captureExtent = swapchain.getExtent();
  createBuffer(device.getAllocator(),
               VkDeviceSize(captureExtent.width) * captureExtent.height * 4,
               VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO,
               VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT, &captureBuffer);
  VkBufferImageCopy region = {};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = {captureExtent.width, captureExtent.height, 1};
  vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         captureBuffer.get(), 1, &region);
  renderResources.transition(cmd, image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             VK_ACCESS_TRANSFER_READ_BIT, 0);
  VkMemoryBarrier2 toHost = {};
  toHost.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
  toHost.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  toHost.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  toHost.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
  toHost.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
  VkDependencyInfo dependency = {};
  dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
  dependency.memoryBarrierCount = 1;
  dependency.pMemoryBarriers = &toHost;
  vkCmdPipelineBarrier2(cmd, &dependency);
  captureState = CaptureState::Recorded;
}

bool VulkanRenderer::finishCapture() {
  if (captureState != CaptureState::Recorded)
    return false;
  vkDeviceWaitIdle(device.getLogicalDevice());
  VmaAllocator allocator = device.getAllocator();
  vmaInvalidateAllocation(allocator, captureBuffer.getAllocation(), 0,
                          VK_WHOLE_SIZE);
  void *mapped = nullptr;
  bool written = false;
  if (vmaMapMemory(allocator, captureBuffer.getAllocation(), &mapped) ==
      VK_SUCCESS) {
    const VkFormat format = swapchain.getImageFormat();
    const bool bgra = format == VK_FORMAT_B8G8R8A8_UNORM ||
                      format == VK_FORMAT_B8G8R8A8_SRGB;
    written = writeCapturePng(capturePath, static_cast<uint8_t *>(mapped),
                              captureExtent.width, captureExtent.height,
                              captureExtent.width * 4, bgra);
    vmaUnmapMemory(allocator, captureBuffer.getAllocation());
  }
  if (written)
    spdlog::info("Captured {}x{} to {}", captureExtent.width,
                 captureExtent.height, capturePath);
  else
    spdlog::error("Couldn't write the capture to {}", capturePath);
  captureBuffer.reset();
  captureState = CaptureState::None;
  return true;
}

void VulkanRenderer::noteGBufferFinalLayouts(uint32_t imageIndex) {
  if (imageIndex >= renderResources.frameCount())
    return;

  const FrameRenderTargets &targets = renderResources.frame(imageIndex);
  constexpr VkPipelineStageFlags colorStage =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  constexpr VkAccessFlags colorAccess = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  renderResources.noteLayout(targets.gBuffer0,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             colorStage, colorAccess);
  renderResources.noteLayout(targets.gBuffer1,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             colorStage, colorAccess);
  renderResources.noteLayout(targets.gBuffer2,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             colorStage, colorAccess);
  renderResources.noteLayout(targets.gBufferDepth,
                             VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
}

void VulkanRenderer::noteLitFinalLayout(uint32_t imageIndex) {
  if (imageIndex >= renderResources.frameCount())
    return;

  const FrameRenderTargets &targets = renderResources.frame(imageIndex);
  renderResources.noteLayout(targets.lit,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
}

void VulkanRenderer::noteCompositeFinalLayouts(uint32_t imageIndex,
                                               uint32_t historyIndex) {
  if (imageIndex >= renderResources.frameCount())
    return;

  const FrameRenderTargets &targets = renderResources.frame(imageIndex);
  renderResources.noteLayout(targets.sceneOutput,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
  renderResources.noteLayout(targets.colorBuffer,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
  if (historyIndex < taaHistoryImages.size()) {
    renderResources.noteLayout(taaHistoryImages[historyIndex].get(),
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
  }
}

// G-buffer resource management

void VulkanRenderer::createGBuffer() {
  size_t count = swapchain.getImageCount();
  VkDevice dev = device.getLogicalDevice();
  VkFormat gb0Fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
  VkFormat gb1Fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
  VkFormat gb2Fmt = VK_FORMAT_R8G8B8A8_UNORM;
  const QueueFamilyIndices queues = device.getQueueFamilies();
  const uint32_t graphicsFamily = static_cast<uint32_t>(queues.graphicsFamily);
  const RenderQueueSharingInfo queueSharing =
      renderGraphicsComputeSharing(queues);

  if (renderResources.frameCount() != count)
    renderResources.resizeFrames(count);

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
  aci.flags = RENDER_DEVICE_ALLOCATION_FLAGS;

  auto makeColorImg = [&](AllocatedImage &img, ImageViewHandle &view,
                          VkFormat fmt) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = {sceneExtent.width, sceneExtent.height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.format = fmt;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    applyRenderQueueSharing(ci, queueSharing);
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
    ci.extent = {sceneExtent.width, sceneExtent.height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.format = gBufferDepthFormat;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    applyRenderQueueSharing(ci, queueSharing);
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

    renderResources.registerImage(
        {"gBuffer0[" + std::to_string(i) + "]", gBuffer0Images[i].get(),
         gb0Fmt, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1,
         VK_IMAGE_LAYOUT_UNDEFINED, queueSharing.mode, graphicsFamily});
    renderResources.registerImage(
        {"gBuffer1[" + std::to_string(i) + "]", gBuffer1Images[i].get(),
         gb1Fmt, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1,
         VK_IMAGE_LAYOUT_UNDEFINED, queueSharing.mode, graphicsFamily});
    renderResources.registerImage(
        {"gBuffer2[" + std::to_string(i) + "]", gBuffer2Images[i].get(),
         gb2Fmt, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1,
         VK_IMAGE_LAYOUT_UNDEFINED, queueSharing.mode, graphicsFamily});
    renderResources.registerImage(
        {"gBufferDepth[" + std::to_string(i) + "]",
         gBufferDepthImages[i].get(), gBufferDepthFormat,
         renderImageAspectMask(gBufferDepthFormat), 1, 1,
         VK_IMAGE_LAYOUT_UNDEFINED, queueSharing.mode, graphicsFamily});

    FrameRenderTargets &targets = renderResources.frame(i);
    targets.gBuffer0 = gBuffer0Images[i].get();
    targets.gBuffer1 = gBuffer1Images[i].get();
    targets.gBuffer2 = gBuffer2Images[i].get();
    targets.gBufferDepth = gBufferDepthImages[i].get();

    std::array<VkImageView, 4> attachments = {
        gBuffer0Views[i].get(), gBuffer1Views[i].get(), gBuffer2Views[i].get(),
        gBufferDepthViews[i].get()};
    VkFramebufferCreateInfo fbci = {};
    fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbci.renderPass = renderPassManager.getGBufferRenderPass();
    fbci.attachmentCount = static_cast<uint32_t>(attachments.size());
    fbci.pAttachments = attachments.data();
    fbci.width = sceneExtent.width;
    fbci.height = sceneExtent.height;
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
  // SSGI views are a temporal history ring. DescriptorManager produces
  // historyCount * swapCount G-buffer sets indexed
  // (historyIndex * swapCount + i); per history index H, binding 5 =
  // SsgiPass history view H.
  std::vector<VkImageView> ssgv = ssgiPass.views();
  const std::vector<VkImageView> ssaov = ssaoPass.views();
  descriptorManager.recreateGBufferSets(device.getLogicalDevice(), gb0v, gb1v,
                                        gb2v, depv, litv, ssgv,
                                        ssaov,
                                        textureManager.getTextureSampler(),
                                        ssaoPass.sampler());
  {
    std::vector<VkImageView> ssgiHv = ssgiPass.views();
    descriptorManager.recreateSsgiPrevSets(device.getLogicalDevice(), ssgiHv,
                                           ssgiPass.sampler());
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
  const QueueFamilyIndices queues = device.getQueueFamilies();
  const uint32_t graphicsFamily = static_cast<uint32_t>(queues.graphicsFamily);

  if (renderResources.frameCount() != count)
    renderResources.resizeFrames(count);

  litImages.clear();
  litViews.clear();
  litFramebuffers.clear();
  litImages.reserve(count);
  litViews.reserve(count);

  VmaAllocationCreateInfo aci = {};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  aci.flags = RENDER_DEVICE_ALLOCATION_FLAGS;

  for (size_t i = 0; i < count; ++i) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = {sceneExtent.width, sceneExtent.height, 1};
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

    renderResources.registerImage(
        {"lit[" + std::to_string(i) + "]", litImages.back().get(), litFormat,
         VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, VK_IMAGE_LAYOUT_UNDEFINED,
         VK_SHARING_MODE_EXCLUSIVE, graphicsFamily});
    renderResources.frame(i).lit = litImages.back().get();

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

void VulkanRenderer::createLitFramebuffers() {
  VkDevice dev = device.getLogicalDevice();
  const size_t count = swapchain.getImageCount();
  if (litViews.size() != count || gBufferDepthViews.size() != count)
    throw std::runtime_error("Lit framebuffer attachments are not ready");

  cleanupLitFramebuffers();
  litFramebuffers.assign(count, VK_NULL_HANDLE);
  for (size_t i = 0; i < count; ++i) {
    std::array<VkImageView, 2> attachments = {
        litViews[i].get(), gBufferDepthViews[i].get()};
    VkFramebufferCreateInfo fbci = {};
    fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbci.renderPass = renderPassManager.getLitRenderPass();
    fbci.attachmentCount = static_cast<uint32_t>(attachments.size());
    fbci.pAttachments = attachments.data();
    fbci.width = sceneExtent.width;
    fbci.height = sceneExtent.height;
    fbci.layers = 1;
    if (vkCreateFramebuffer(dev, &fbci, nullptr, &litFramebuffers[i]) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to create lit framebuffer");
  }
}

void VulkanRenderer::cleanupLitFramebuffers() {
  VkDevice dev = device.getLogicalDevice();
  for (VkFramebuffer fb : litFramebuffers)
    if (fb != VK_NULL_HANDLE)
      vkDestroyFramebuffer(dev, fb, nullptr);
  litFramebuffers.clear();
}

void VulkanRenderer::cleanupLitResources() {
  VkDevice dev = device.getLogicalDevice();
  cleanupLitFramebuffers();
  litViews.clear();
  litImages.clear();
  if (litSampler != VK_NULL_HANDLE) {
    vkDestroySampler(dev, litSampler, nullptr);
    litSampler = VK_NULL_HANDLE;
  }
}

// TAA history images. The ring is one larger than the max frames-in-flight,
// so a frame can sample the previous logical frame without another in-flight
// frame overwriting that image too early. Format matches litFormat so reads
// from history feed straight into the same HDR-precision pipeline. Usage:
// SAMPLED (read as history-prev) + COLOR_ATTACHMENT (written as history-curr).
void VulkanRenderer::createTaaResources() {
  VkDevice dev = device.getLogicalDevice();
  const QueueFamilyIndices queues = device.getQueueFamilies();
  const uint32_t graphicsFamily = static_cast<uint32_t>(queues.graphicsFamily);
  taaHistoryImages.clear();
  taaHistoryViews.clear();
  const size_t historyCount = MAX_FRAMES_DRAWS + 1;
  taaHistoryImages.reserve(historyCount);
  taaHistoryViews.reserve(historyCount);

  VmaAllocationCreateInfo aci = {};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  aci.flags = RENDER_DEVICE_ALLOCATION_FLAGS;

  for (size_t i = 0; i < historyCount; ++i) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = {sceneExtent.width, sceneExtent.height, 1};
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

    renderResources.registerImage(
        {"taaHistory[" + std::to_string(i) + "]",
         taaHistoryImages.back().get(), litFormat, VK_IMAGE_ASPECT_COLOR_BIT,
         1, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_SHARING_MODE_EXCLUSIVE,
         graphicsFamily});
  }

  // Transition both to SHADER_READ_ONLY_OPTIMAL so the first frame's
  // sampler bind doesn't read undefined data. The composite render pass
  // (when wired up next session) will declare initialLayout=UNDEFINED on
  // the history attachment with loadOp=DONT_CARE, so the transition back
  // to COLOR_ATTACHMENT_OPTIMAL is handled by the render pass itself.
  VkCommandBuffer cmd = beginCommandBuffer(dev, device.getGraphicsCommandPool());
  for (size_t i = 0; i < historyCount; ++i) {
    renderResources.transition(
        cmd, taaHistoryImages[i].get(),
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, VK_ACCESS_SHADER_READ_BIT);
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
  // frame. SsgiPass is recreated alongside createTaaResources on every
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
  activeEnvironmentHdr = "Resources/HDRIs/kloppenheim_02_1k.hdr";
  iblSkyboxImageIndex =
      textureManager.loadSkybox(activeEnvironmentHdr, device, descriptorManager);
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
  // Used to cull and pick the LOD for the entire batch.
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

  uploadToAllocation(device.getAllocator(),
                     drawable.instanceBuffer.getAllocation(),
                     drawable.instances.data(), bufferSize);

  instancedDrawables.push_back(std::move(drawable));
}

void VulkanRenderer::cleanup() {
  vkDeviceWaitIdle(device.getLogicalDevice());
  // The world goes now, while main() runs. Left to the destructor, it would
  // go after main() returns, when Jolt's own statics (its default material)
  // may already be gone, and Jolt would release that material twice.
  animator.reset();
  player.reset(); // takes its character out of the world below
  physicsLinks.clear();
  physics.reset();

  metrics.printReport(device.getAllocator());

  renderResources.clear();
  instancedDrawables.clear(); // AllocatedBuffer RAII destroys GPU buffers
  imguiLayer.cleanup(device.getLogicalDevice());
  debugLinePass.cleanup(device.getLogicalDevice());
  skinningPass.cleanup();
  modelManager.cleanup();
  textureManager.cleanup(device.getLogicalDevice(), device.getAllocator());
  materialProbePass.cleanup();
  cleanupSceneTargets();
  gpuDrivenGBufferPass.releaseGeometry();
  cleanupIBL();
  shadowPass.cleanup();

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
  pipelineCache.cleanup(device.getLogicalDevice());
  renderPassManager.cleanup(device.getLogicalDevice());
  metrics.cleanup(device.getLogicalDevice());
  device.cleanup();
}

VulkanRenderer::~VulkanRenderer() {}

// Command recording

static glm::uvec4 materialTextureIndices0(const Material &mat) {
  return glm::uvec4(static_cast<uint32_t>(mat.albedoTextureId),
                    static_cast<uint32_t>(mat.normalTextureId),
                    static_cast<uint32_t>(mat.metallicTextureId),
                    static_cast<uint32_t>(mat.roughnessTextureId));
}

static glm::uvec4 materialTextureIndices1(const Material &mat) {
  const uint32_t materialFlags =
      (mat.isCloth ? 1u : 0u) | (mat.alphaMasked ? 2u : 0u) |
      (mat.alphaBlended ? 4u : 0u) | (mat.worldGrid ? 8u : 0u);
  const uint32_t alphaCutoff255 = static_cast<uint32_t>(
      glm::clamp(mat.alphaCutoff, 0.0f, 1.0f) * 255.0f + 0.5f);
  return glm::uvec4(static_cast<uint32_t>(mat.aoTextureId), materialFlags,
                    alphaCutoff255, 0u);
}

static void sortGBufferDrawItemsForState(std::vector<GBufferDrawItem> &items) {
  std::stable_sort(items.begin(), items.end(),
                   [](const GBufferDrawItem &a, const GBufferDrawItem &b) {
                     if (a.kind != b.kind)
                       return static_cast<int>(a.kind) <
                              static_cast<int>(b.kind);
                     if (a.cullMode != b.cullMode)
                       return a.cullMode < b.cullMode;
                     if (a.mesh != b.mesh)
                       return std::less<const Mesh *>{}(a.mesh, b.mesh);
                     return a.lod < b.lod;
                   });

  for (size_t i = 0; i < items.size(); ++i) {
    items[i].materialId = static_cast<uint32_t>(i);
    items[i].transformId = static_cast<uint32_t>(i);
  }
}

struct GBufferStateStats {
  uint32_t pipelineBinds = 0;
  uint32_t dynamicStateChanges = 0;
};

struct TransparentDrawItem {
  GBufferDrawKind kind = GBufferDrawKind::Mesh;
  const Mesh *mesh = nullptr;
  int lod = 0;
  ModelPushConstants push{};
  VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
  VkBuffer instanceBuffer = VK_NULL_HANDLE;
  const std::vector<InstanceData> *instances = nullptr;
  uint32_t instanceCount = 1;
  uint32_t indexCount = 0;
  float distanceSq = 0.0f;
};

static GBufferStateStats estimateGBufferStateStats(
    const std::vector<GBufferDrawItem> &items, uint32_t partitionCount) {
  GBufferStateStats stats{};
  partitionCount = std::max(1u, partitionCount);

  for (uint32_t partition = 0; partition < partitionCount; ++partition) {
    const size_t begin =
        (items.size() * size_t(partition)) / size_t(partitionCount);
    const size_t end =
        (items.size() * size_t(partition + 1)) / size_t(partitionCount);

    GBufferDrawKind boundKind = GBufferDrawKind::Mesh;
    bool pipelineBound = false;
    VkCullModeFlags lastCull = VK_CULL_MODE_BACK_BIT;
    bool cullValid = false;

    for (size_t i = begin; i < end; ++i) {
      const GBufferDrawItem &item = items[i];
      if (!pipelineBound || boundKind != item.kind) {
        stats.pipelineBinds++;
        boundKind = item.kind;
        pipelineBound = true;
        cullValid = false;
      }

      if (!cullValid || item.cullMode != lastCull) {
        stats.dynamicStateChanges++;
        lastCull = item.cullMode;
        cullValid = true;
      }
    }
  }

  return stats;
}

void VulkanRenderer::recordGBufferCommands(uint32_t currentImage) {
  VkCommandBuffer cmd = swapchain.getCommandBuffer(currentImage);
  vkResetCommandBuffer(cmd, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
    throw std::runtime_error("Failed to begin G-buffer command buffer");

  metrics.resetGpuQueries(cmd);

  // The character bent for this frame, before anything draws it.
  if (player && characterModelId >= 0 && skinningPass.hasMesh())
    if (const MeshModel *character = modelManager.getModel(characterModelId);
        character && character->getMeshCount() > 0) {
      const Mesh *mesh = character->getMesh(0);
      skinningPass.updateMatrices(currentFrame, characterSkinning);
      skinningPass.record(cmd, currentFrame, mesh->getVertexBuffer(),
                          gpuDrivenGBufferPass.arenaVertexBuffer(),
                          gpuDrivenGBufferPass.arenaFirstVertex(mesh));
    }

  VkViewport viewport = {};
  viewport.width = static_cast<float>(sceneExtent.width);
  viewport.height = static_cast<float>(sceneExtent.height);
  viewport.maxDepth = 1.0f;
  VkRect2D scissor = {{0, 0}, sceneExtent};
  recordGBufferPass(cmd, currentImage, viewport, scissor);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
    throw std::runtime_error("Failed to end G-buffer command buffer");
}

// Every mesh in the scene with its world bounding sphere, once a frame: the
// opaque ones for the G-buffer and shadow passes, the blended ones for the
// transparent pass. The vectors keep their memory between frames.
void VulkanRenderer::collectMeshInstances() {
  frameMeshes.clear();
  frameBlendedMeshes.clear();
  const glm::vec3 camera = glm::vec3(sceneUbo.cameraPosition);
  auto visit = [&](auto &self, const SceneNode &node) -> void {
    if (MeshModel *model = modelManager.getModel(node.getModelId())) {
      const glm::mat4 world = node.getGlobalTransform();
      const float maxScale =
          glm::max(glm::length(glm::vec3(world[0])),
                   glm::max(glm::length(glm::vec3(world[1])),
                            glm::length(glm::vec3(world[2]))));
      const float modelDistance = glm::length(
          glm::vec3(world * glm::vec4(model->boundingCenter, 1.0f)) - camera);
      for (size_t k = 0; k < model->getMeshCount(); ++k) {
        const Mesh *mesh = model->getMesh(k);
        std::vector<MeshInstance> &list = mesh->getMaterial().alphaBlended
                                              ? frameBlendedMeshes
                                              : frameMeshes;
        list.push_back(
            {&node, mesh,
             glm::vec3(world * glm::vec4(mesh->boundingCenter, 1.0f)),
             mesh->boundingRadius * maxScale, modelDistance});
      }
    }
    for (const auto &child : node.getChildren())
      self(self, *child);
  };
  visit(visit, rootNode);
}

void VulkanRenderer::recordShadowCommands(VkCommandBuffer cmd) {
  vkResetCommandBuffer(cmd, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
    throw std::runtime_error("Failed to begin shadow command buffer");

  vkdbgBeginLabel(cmd, "CSM Shadow Pass", 1.0f, 0.4f, 0.1f);
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::Shadow);
  shadowPass.recordCsm(cmd, frameMeshes, pipeline, descriptorManager, sceneUbo,
                       imguiShadowFrontFaceCull, imguiCullShadowCasters);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::Shadow);
  vkdbgEndLabel(cmd);

  // Only point light 0 casts a shadow, and lit.frag samples the cube only
  // when point lights are enabled and counted, so skip the six-face render
  // otherwise (about 1.2 ms per frame). The timestamps stay so the overlay
  // reads ~0 ms instead of a missing query.
  const bool pointShadowNeeded =
      imguiEnablePointLights && sceneUbo.lightCounts.x > 0;
  vkdbgBeginLabel(cmd, "Point Shadow Pass", 1.0f, 0.6f, 0.1f);
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::PointShadow);
  if (pointShadowNeeded || !pointShadowMapInitialized) {
    shadowPass.recordPoint(cmd, frameMeshes, pipeline, descriptorManager,
                           imguiShadowFrontFaceCull, imguiCullShadowCasters);
    pointShadowMapInitialized = true;
  }
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::PointShadow);
  vkdbgEndLabel(cmd);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
    throw std::runtime_error("Failed to end shadow command buffer");
}

void VulkanRenderer::recordSsaoComputeCommands(VkCommandBuffer cmd,
                                               uint32_t currentImage) {
  const uint32_t ssgiHistoryIndex = ssgiPass.historyIndex(taaFrameCounter);
  ssaoPass.recordCommands(
      cmd, currentImage, descriptorManager.getVPSet(currentImage),
      descriptorManager.getGBufferSet(ssgiHistoryIndex, currentImage));
}

void VulkanRenderer::recordPostCommands(VkCommandBuffer cmd,
                                        uint32_t currentImage) {
  vkResetCommandBuffer(cmd, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
    throw std::runtime_error("Failed to begin post command buffer");

  VkViewport viewport = {};
  viewport.width = static_cast<float>(sceneExtent.width);
  viewport.height = static_cast<float>(sceneExtent.height);
  viewport.maxDepth = 1.0f;
  VkRect2D scissor = {{0, 0}, sceneExtent};

  uint32_t ssgiHistoryIndex = ssgiPass.historyIndex(taaFrameCounter);
  uint32_t taaHistoryIndex =
      static_cast<uint32_t>(taaFrameCounter % taaHistoryViews.size());

  materialProbePass.record(
      cmd, static_cast<uint32_t>(currentFrame),
      descriptorManager.getVPSet(currentImage),
      descriptorManager.getGBufferSet(ssgiHistoryIndex, currentImage),
      materialProbePixel);

  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::SSGI);
  ssgiPass.record(cmd, ssgiHistoryIndex,
                  descriptorManager.getVPSet(currentImage),
                  descriptorManager.getGBufferSet(ssgiHistoryIndex,
                                                  currentImage),
                  descriptorManager.getSsgiPrevSet(ssgiHistoryIndex),
                  pipeline.getSsgiPipeline(), pipeline.getSsgiLayout());
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::SSGI);

  // --- Lit pass (PBR + IBL + async SSAO + FXAA + fog -> litBuffer) ---
  {
    VkClearValue clear = {};
    clear.color = {0.0f, 0.0f, 0.0f, 1.0f};

    VkRenderPassBeginInfo rpbi = {};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = renderPassManager.getLitRenderPass();
    rpbi.framebuffer = litFramebuffers[currentImage];
    rpbi.renderArea = {{0, 0}, sceneExtent};
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
    vkCmdNextSubpass(cmd, VK_SUBPASS_CONTENTS_INLINE);
    recordTransparentPass(cmd, currentImage, viewport, scissor);
    vkCmdEndRenderPass(cmd);
    noteLitFinalLayout(currentImage);
    metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::Lit);
    vkdbgEndLabel(cmd);
  }

  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::Bloom);
  const bool bloomRecorded =
      bloomPass.record(cmd, currentImage, litImages, imguiEnableBloom,
                       imguiDebugMode, imguiBloomThreshold, imguiBloomRadius,
                       imguiBloomIntensity);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::Bloom);

  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::AutoExposure);
  autoExposurePass.record(cmd, currentImage, currentFrame, litImages,
                          autoExpEnabled, bloomRecorded);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::AutoExposure);

  const size_t compositeFbIdx =
      compositePass.framebufferIndex(taaHistoryIndex, currentImage);
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::Composite);
  compositePass.record(
      cmd, currentImage, taaHistoryIndex,
      descriptorManager.getVPSet(currentImage),
      descriptorManager.getGBufferSet(ssgiHistoryIndex, currentImage),
      descriptorManager.getInputSet(currentImage),
      descriptorManager.getTaaSet(compositeFbIdx),
      pipeline.getDeferredPipeline(), pipeline.getDeferredLayout(),
      pipeline.getSecondPipeline(), pipeline.getSecondLayout());
  noteCompositeFinalLayouts(currentImage, taaHistoryIndex);
  // Counted with the composite: it's the last step of the scene's picture.
  copySceneToWindow(cmd, currentImage);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::Composite);

  vkdbgBeginLabel(cmd, "ImGui Pass", 0.9f, 0.8f, 0.1f);
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::ImGui);
  recordImGuiCommands(cmd, currentImage);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::ImGui);
  vkdbgEndLabel(cmd);
  if (captureState == CaptureState::Requested)
    recordCapture(cmd, currentImage);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
    throw std::runtime_error("Failed to end post command buffer");
}

static int lodForDistance(float distance, float lodNear, float lodFar,
                          const Mesh &mesh) {
  const int lod = distance < lodNear ? 0 : distance < lodFar ? 1 : 2;
  return std::min(lod, mesh.getLodCount() - 1);
}

// Hands every opaque mesh and instanced group to the GPU-driven pass when
// the scene has at least minCandidates of them and the pass is ready. False
// means the CPU path draws this frame.
bool VulkanRenderer::prepareGpuDrivenFrame(uint32_t currentImage,
                                           const glm::vec4 frustumPlanes[6],
                                           uint32_t minCandidates) {
  uint32_t candidates = static_cast<uint32_t>(frameMeshes.size());
  for (const InstancedDrawable &drawable : instancedDrawables) {
    const MeshModel *mdl = modelManager.getModel(drawable.modelId);
    if (!mdl || drawable.instances.empty())
      continue;
    for (size_t k = 0; k < mdl->getMeshCount(); k++)
      if (!mdl->getMesh(k)->getMaterial().alphaBlended)
        ++candidates;
  }
  gpuDrivenGBufferPass.beginFrame(candidates);
  if (!gpuDrivenGBufferPass.beginRecords(currentImage, imguiGpuDrivenEnabled,
                                         minCandidates))
    return false;

  for (const MeshInstance &instance : frameMeshes) {
    const Mesh &mesh = *instance.mesh;
    const Material &mat = mesh.getMaterial();
    const int lod =
        lodForDistance(instance.modelDistance, imguiLodNear, imguiLodFar, mesh);
    const glm::vec3 radius(instance.radius);
    if (gpuDrivenGBufferPass.addMesh(
            &mesh, lod, instance.node->getGlobalTransform(),
            instance.node->getNormalMatrix(), materialTextureIndices0(mat),
            materialTextureIndices1(mat), instance.center - radius,
            instance.center + radius,
            mat.doubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT))
      metrics.recordDrawCall(static_cast<uint32_t>(mesh.getIndexCount(lod)));
  }

  const glm::vec3 camera(sceneUbo.cameraPosition);
  for (const InstancedDrawable &drawable : instancedDrawables) {
    const MeshModel *mdl = modelManager.getModel(drawable.modelId);
    if (!mdl || drawable.instances.empty())
      continue;
    const float distance = glm::length(drawable.groupCenter - camera);
    const glm::vec3 radius(drawable.groupRadius);
    for (size_t k = 0; k < mdl->getMeshCount(); k++) {
      const Mesh &mesh = *mdl->getMesh(k);
      const Material &mat = mesh.getMaterial();
      if (mat.alphaBlended)
        continue;
      const int lod = lodForDistance(distance, imguiLodNear, imguiLodFar, mesh);
      if (gpuDrivenGBufferPass.addInstanced(
              &mesh, lod, drawable.instances, materialTextureIndices0(mat),
              materialTextureIndices1(mat), drawable.groupCenter - radius,
              drawable.groupCenter + radius))
        metrics.recordDrawCall(static_cast<uint32_t>(
            mesh.getIndexCount(lod) * drawable.instances.size()));
    }
  }

  // 0.001 of standard depth at a 1 m near plane, the margin this had before
  // reversed-Z: projection[3][2] is about the near plane, and reversed depth
  // differences scale with it.
  const float hzbDepthMargin = 0.001f * sceneUbo.projection[3][2];
  gpuDrivenGBufferPass.uploadRecords(currentImage, frustumPlanes,
                                     sceneUbo.projection * sceneUbo.view,
                                     hzbDepthMargin, imguiHzbCullingEnabled);
  return gpuDrivenGBufferPass.canRecordIndirect(
      currentImage, imguiGpuDrivenEnabled, minCandidates);
}

// The CPU path: the meshes in view, sorted to keep pipeline and buffer
// changes down. Instanced groups are culled and LOD-picked as a whole.
void VulkanRenderer::collectGBufferDrawItems(
    std::vector<GBufferDrawItem> &drawItems, const glm::vec4 frustumPlanes[6]) {
  drawItems.reserve(128);
  for (const MeshInstance &instance : frameMeshes) {
    if (!sphereInFrustum(frustumPlanes, instance.center, instance.radius))
      continue;
    const Mesh *mesh = instance.mesh;
    const Material &mat = mesh->getMaterial();
    GBufferDrawItem item{};
    item.kind = GBufferDrawKind::Mesh;
    item.mesh = mesh;
    item.lod =
        lodForDistance(instance.modelDistance, imguiLodNear, imguiLodFar, *mesh);
    item.push.model = instance.node->getGlobalTransform();
    item.push.normal = instance.node->getNormalMatrix();
    item.push.texIdx0 = materialTextureIndices0(mat);
    item.push.texIdx1 = materialTextureIndices1(mat);
    const glm::vec3 radius(instance.radius);
    item.aabbMin = instance.center - radius;
    item.aabbMax = instance.center + radius;
    item.cullMode = mat.doubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
    item.indexCount = static_cast<uint32_t>(mesh->getIndexCount(item.lod));
    drawItems.push_back(item);
  }

  // Instanced drawables still become ordinary draw items here. The secondary
  // command buffer recorder switches to the instanced pipeline for these items.
  const glm::vec3 camera(sceneUbo.cameraPosition);
  for (const InstancedDrawable &drawable : instancedDrawables) {
    MeshModel *mdl = modelManager.getModel(drawable.modelId);
    if (!mdl || drawable.instances.empty())
      continue;
    if (!sphereInFrustum(frustumPlanes, drawable.groupCenter,
                         drawable.groupRadius))
      continue;
    const float distance = glm::length(drawable.groupCenter - camera);
    for (size_t k = 0; k < mdl->getMeshCount(); k++) {
      const Mesh *mesh = mdl->getMesh(k);
      const Material &mat = mesh->getMaterial();
      if (mat.alphaBlended)
        continue;
      GBufferDrawItem item{};
      item.kind = GBufferDrawKind::Instanced;
      item.mesh = mesh;
      item.lod = lodForDistance(distance, imguiLodNear, imguiLodFar, *mesh);
      item.push.texIdx0 = materialTextureIndices0(mat);
      item.push.texIdx1 = materialTextureIndices1(mat);
      const glm::vec3 radius(drawable.groupRadius);
      item.aabbMin = drawable.groupCenter - radius;
      item.aabbMax = drawable.groupCenter + radius;
      item.cullMode = VK_CULL_MODE_BACK_BIT;
      item.instanceBuffer = drawable.instanceBuffer.get();
      item.instances = &drawable.instances;
      item.instanceCount = static_cast<uint32_t>(drawable.instances.size());
      item.indexCount = static_cast<uint32_t>(mesh->getIndexCount(item.lod));
      drawItems.push_back(item);
    }
  }
  sortGBufferDrawItemsForState(drawItems);
}

// Draws items [begin, end) into drawCmd, binding the mesh or instanced
// pipeline and the cull mode only when they change (the items are sorted for
// that).
void VulkanRenderer::recordGBufferDraws(VkCommandBuffer drawCmd,
                                        const std::vector<GBufferDrawItem> &items,
                                        size_t begin, size_t end,
                                        const VkViewport &viewport,
                                        const VkRect2D &scissor,
                                        VkDescriptorSet vpSet,
                                        VkDescriptorSet bindlessSet) {
  const std::array<VkDescriptorSet, 2> graphicsSets = {vpSet, bindlessSet};
  const std::array<VkDescriptorSet, 2> instancedSets = {vpSet, bindlessSet};
  vkCmdSetViewport(drawCmd, 0, 1, &viewport);
  vkCmdSetScissor(drawCmd, 0, 1, &scissor);
  GBufferDrawKind boundKind = GBufferDrawKind::Mesh;
  bool pipelineBound = false;
  VkCullModeFlags lastCull = VK_CULL_MODE_BACK_BIT;
  bool cullValid = false;

  for (size_t i = begin; i < end; ++i) {
    const GBufferDrawItem &item = items[i];
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
}

// The CPU path split across the worker threads: each records a share of the
// items into its own secondary command buffer, and the primary runs them in
// one render pass. Off by default; at a few dozen draws it didn't beat one
// thread.
void VulkanRenderer::recordGBufferThreaded(
    VkCommandBuffer cmd, uint32_t currentImage, const VkRenderPassBeginInfo &rpbi,
    const std::vector<GBufferDrawItem> &items, const VkViewport &viewport,
    const VkRect2D &scissor, VkDescriptorSet vpSet,
    VkDescriptorSet bindlessSet) {
  auto &frameResources = threadedCommandFrames[currentFrame];
  const uint32_t workerCount = std::min<uint32_t>(
      threadedCommandWorkerCount, static_cast<uint32_t>(items.size()));
  const GBufferStateStats stateStats =
      estimateGBufferStateStats(items, workerCount);
  metrics.recordPipelineBind(stateStats.pipelineBinds);
  metrics.recordDynamicStateChange(stateStats.dynamicStateChanges);
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
        (items.size() * size_t(workerIndex)) / size_t(workerCount);
    const size_t end =
        (items.size() * size_t(workerIndex + 1)) / size_t(workerCount);
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

    recordGBufferDraws(secondary, items, begin, end, viewport, scissor, vpSet,
                       bindlessSet);
    if (vkEndCommandBuffer(secondary) != VK_SUCCESS)
      throw std::runtime_error("Failed to end G-buffer secondary command buffer");
  };

  commandThreadPool->dispatch(workerCount, recordRange);

  vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
  vkCmdExecuteCommands(cmd, workerCount, secondaries.data());
  vkCmdEndRenderPass(cmd);
}

void VulkanRenderer::recordGBufferPass(VkCommandBuffer cmd, uint32_t currentImage,
                                       const VkViewport &viewport,
                                       const VkRect2D &scissor) {
  glm::vec4 frustumPlanes[6];
  extractFrustumPlanes(sceneUbo.projection * sceneUbo.view, frustumPlanes);
  const uint32_t minGpuDrivenCandidates =
      static_cast<uint32_t>(std::max(0, imguiGpuDrivenMinCandidates));

  // Big scenes go to the GPU-driven pass whole: its compute shader culls, so
  // the CPU neither culls nor sorts them. Small ones are culled, sorted and
  // drawn here.
  const bool useGpuDriven =
      prepareGpuDrivenFrame(currentImage, frustumPlanes, minGpuDrivenCandidates);
  std::vector<GBufferDrawItem> drawItems;
  if (!useGpuDriven) {
    collectGBufferDrawItems(drawItems, frustumPlanes);
    for (const GBufferDrawItem &item : drawItems)
      metrics.recordDrawCall(item.indexCount * item.instanceCount);
  }

  std::array<VkClearValue, 4> clears{};
  clears[3].depthStencil = {0.0f, 0}; // reversed-Z: 0 is the far plane

  VkRenderPassBeginInfo rpbi = {};
  rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rpbi.renderPass = renderPassManager.getGBufferRenderPass();
  rpbi.framebuffer = gBufferFramebuffers[currentImage];
  rpbi.renderArea = {{0, 0}, sceneExtent};
  rpbi.clearValueCount = static_cast<uint32_t>(clears.size());
  rpbi.pClearValues = clears.data();

  vkdbgBeginLabel(cmd, "G-Buffer Pass", 0.1f, 0.4f, 1.0f);
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::GBuffer);

  const VkDescriptorSet vpSet = descriptorManager.getVPSet(currentImage);
  const VkDescriptorSet bindlessSet = descriptorManager.getBindlessSet();
  const bool canRecordThreaded =
      imguiThreadedGBufferEnabled && commandThreadPool &&
      threadedCommandWorkerCount > 0 &&
      currentFrame < static_cast<int>(threadedCommandFrames.size()) &&
      !threadedCommandFrames[currentFrame].empty() && !drawItems.empty();

  if (useGpuDriven) {
    metrics.recordPipelineBind(3);      // cull compute + two G-buffer pipelines
    metrics.recordDynamicStateChange(2); // viewport + scissor
    gpuDrivenGBufferPass.recordIndirectGBuffer(
        cmd, currentImage, rpbi, viewport, scissor, vpSet, bindlessSet);
  } else if (canRecordThreaded) {
    recordGBufferThreaded(cmd, currentImage, rpbi, drawItems, viewport,
                          scissor, vpSet, bindlessSet);
  } else {
    const GBufferStateStats stateStats =
        estimateGBufferStateStats(drawItems, 1);
    metrics.recordPipelineBind(stateStats.pipelineBinds);
    metrics.recordDynamicStateChange(stateStats.dynamicStateChanges);
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    recordGBufferDraws(cmd, drawItems, 0, drawItems.size(), viewport, scissor,
                       vpSet, bindlessSet);
    vkCmdEndRenderPass(cmd);
  }
  noteGBufferFinalLayouts(currentImage);
  gpuDrivenGBufferPass.recordHzbBuild(
      cmd, currentImage, gBufferDepthImages, gBufferDepthFormat,
      imguiGpuDrivenEnabled, imguiHzbCullingEnabled, minGpuDrivenCandidates);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::GBuffer);
  vkdbgEndLabel(cmd);
}

void VulkanRenderer::recordTransparentPass(VkCommandBuffer cmd,
                                           uint32_t currentImage,
                                           const VkViewport &viewport,
                                           const VkRect2D &scissor) {
  std::vector<TransparentDrawItem> drawItems;
  drawItems.reserve(32);

  glm::vec4 frustumPlanes[6];
  extractFrustumPlanes(sceneUbo.projection * sceneUbo.view, frustumPlanes);
  const glm::vec3 cameraPos = glm::vec3(sceneUbo.cameraPosition);

  for (const MeshInstance &instance : frameBlendedMeshes) {
    if (!sphereInFrustum(frustumPlanes, instance.center, instance.radius))
      continue;
    const Mesh *mesh = instance.mesh;
    const Material &mat = mesh->getMaterial();
    TransparentDrawItem item{};
    item.kind = GBufferDrawKind::Mesh;
    item.mesh = mesh;
    item.lod =
        lodForDistance(instance.modelDistance, imguiLodNear, imguiLodFar, *mesh);
    item.push.model = instance.node->getGlobalTransform();
    item.push.normal = instance.node->getNormalMatrix();
    item.push.texIdx0 = materialTextureIndices0(mat);
    item.push.texIdx1 = materialTextureIndices1(mat);
    item.cullMode = mat.doubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
    item.indexCount = static_cast<uint32_t>(mesh->getIndexCount(item.lod));
    const glm::vec3 toCamera = instance.center - cameraPos;
    item.distanceSq = glm::dot(toCamera, toCamera);
    drawItems.push_back(item);
  }

  for (const InstancedDrawable &drawable : instancedDrawables) {
    MeshModel *mdl = modelManager.getModel(drawable.modelId);
    if (!mdl || drawable.instances.empty())
      continue;
    if (!sphereInFrustum(frustumPlanes, drawable.groupCenter,
                         drawable.groupRadius))
      continue;

    float groupDist = glm::length(drawable.groupCenter - cameraPos);
    int instLod = (groupDist < imguiLodNear)  ? 0
                  : (groupDist < imguiLodFar) ? 1
                                              : 2;
    for (size_t k = 0; k < mdl->getMeshCount(); ++k) {
      const Mesh *mesh = mdl->getMesh(k);
      const Material &mat = mesh->getMaterial();
      if (!mat.alphaBlended)
        continue;

      TransparentDrawItem item{};
      item.kind = GBufferDrawKind::Instanced;
      item.mesh = mesh;
      item.lod = std::min(instLod, mesh->getLodCount() - 1);
      item.push.texIdx0 = materialTextureIndices0(mat);
      item.push.texIdx1 = materialTextureIndices1(mat);
      item.cullMode =
          mat.doubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
      item.instanceBuffer = drawable.instanceBuffer.get();
      item.instances = &drawable.instances;
      item.instanceCount = static_cast<uint32_t>(drawable.instances.size());
      item.indexCount = static_cast<uint32_t>(mesh->getIndexCount(item.lod));
      const glm::vec3 toCamera = drawable.groupCenter - cameraPos;
      item.distanceSq = glm::dot(toCamera, toCamera);
      drawItems.push_back(item);
    }
  }

  if (drawItems.empty())
    return;

  std::stable_sort(drawItems.begin(), drawItems.end(),
                   [](const TransparentDrawItem &a,
                      const TransparentDrawItem &b) {
                     return a.distanceSq > b.distanceSq;
                   });

  vkdbgBeginLabel(cmd, "Transparent Forward Pass", 0.1f, 0.65f, 0.9f);
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  vkCmdSetScissor(cmd, 0, 1, &scissor);

  const std::array<VkDescriptorSet, 2> sets = {
      descriptorManager.getVPSet(currentImage),
      descriptorManager.getBindlessSet()};

  GBufferDrawKind boundKind = GBufferDrawKind::Mesh;
  bool pipelineBound = false;
  VkCullModeFlags lastCull = VK_CULL_MODE_BACK_BIT;
  bool cullValid = false;

  for (const TransparentDrawItem &item : drawItems) {
    if (!pipelineBound || boundKind != item.kind) {
      VkPipeline pipe = item.kind == GBufferDrawKind::Mesh
                            ? pipeline.getTransparentPipeline()
                            : pipeline.getTransparentInstancedPipeline();
      VkPipelineLayout layout = item.kind == GBufferDrawKind::Mesh
                                    ? pipeline.getTransparentLayout()
                                    : pipeline.getTransparentInstancedLayout();
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0,
                              static_cast<uint32_t>(sets.size()), sets.data(),
                              0, nullptr);
      boundKind = item.kind;
      pipelineBound = true;
      cullValid = false;
    }

    if (!cullValid || item.cullMode != lastCull) {
      vkCmdSetCullMode(cmd, item.cullMode);
      lastCull = item.cullMode;
      cullValid = true;
    }

    VkBuffer vb[] = {item.mesh->getVertexBuffer()};
    VkDeviceSize off[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vb, off);
    if (item.kind == GBufferDrawKind::Instanced) {
      VkDeviceSize instanceOff = 0;
      vkCmdBindVertexBuffers(cmd, 1, 1, &item.instanceBuffer, &instanceOff);
    }
    vkCmdBindIndexBuffer(cmd, item.mesh->getIndexBuffer(item.lod), 0,
                         VK_INDEX_TYPE_UINT32);

    VkPipelineLayout layout = item.kind == GBufferDrawKind::Mesh
                                  ? pipeline.getTransparentLayout()
                                  : pipeline.getTransparentInstancedLayout();
    vkCmdPushConstants(cmd, layout,
                       VK_SHADER_STAGE_VERTEX_BIT |
                           VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(ModelPushConstants), &item.push);
    vkCmdDrawIndexed(cmd, item.indexCount, item.instanceCount, 0, 0, 0);
    metrics.recordDrawCall(item.indexCount * item.instanceCount);
  }

  vkdbgEndLabel(cmd);
}

// ImGui integration

void VulkanRenderer::setImGuiCameraInfo(glm::vec3 pos, float speed) {
  imguiCameraPos = pos;
  imguiCameraSpeed = speed;
}

void VulkanRenderer::setCameraPresetCallback(CameraPresetCallback callback) {
  cameraPresetCallback = std::move(callback);
}

void VulkanRenderer::rebuildProjection() {
  float aspect = static_cast<float>(sceneExtent.width) /
                 static_cast<float>(sceneExtent.height);
  // Reversed-Z (cameraProjection): depth 1 at the near plane, cleared to 0,
  // tested with GREATER.
  sceneUbo.projection = cameraProjection(glm::radians(imguiCameraFov), aspect,
                                         kCameraNear, imguiDrawDistance);
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

// This frame's ImGui frame: every panel, the gizmo and the prompts, with
// the renderer's state and callbacks handed over in one context. It runs at
// the start of the frame, before anything is recorded, so the frame draws
// what the UI just changed; recordImGuiCommands only records the result.
void VulkanRenderer::buildUi() {
  DebugUiContext ui{
      metrics,
      device.getAllocator(),
      sceneUbo,
      imguiCameraPos,
      imguiCameraSpeed,
      imguiCameraFov,
      imguiDrawDistance,
      imguiLodNear,
      imguiLodFar,
      imguiPointShadowFar,
      currentScene.sun,
      currentScene.fog,
      imguiDebugMode,
      imguiSpecAAVariance,
      imguiSpecAAThreshold,
      imguiIblRoughnessFloor,
      imguiShadowFrontFaceCull,
      imguiCullShadowCasters,
      imguiCsmFar,
      imguiIblIntensity,
      imguiExposureEV,
      imguiMinSurfaceRoughness,
      imguiSkyOcclusionFloor,
      imguiSsgiIntensity,
      imguiSsgiSamples,
      imguiEnableSunDirect,
      imguiEnablePointLights,
      imguiEnableSpotLights,
      imguiEnableIblAmbient,
      imguiEnableSsgiBounce,
      imguiEnableBloom,
      imguiBloomThreshold,
      imguiBloomIntensity,
      imguiBloomRadius,
      imguiSsrEnabled,
      imguiTaaEnabled,
      imguiResponsiveTaa,
      imguiPreferMailboxPresent,
      imguiSharpness,
      imguiEdgeAA,
      imguiAlphaDither,
      imguiShadowSoftness,
      imguiContactGrounding,
      imguiDayNightEnable,
      imguiDayNightSpeed,
      imguiDayNightHour,
      imguiNormalStrength,
      imguiUseGeomNormalOnly,
      imguiGpuDrivenEnabled,
      imguiHzbCullingEnabled,
      imguiGpuDrivenMinCandidates,
      imguiThreadedGBufferEnabled,
      autoExpEnabled,
      &availableSceneNames,
      activeSceneIndex,
      autoExposurePass.adaptedValue(),
      materialProbeResult,
      gpuDrivenGBufferPass.candidateCount(),
      gpuDrivenGBufferPass.meshCount(),
      gpuDrivenGBufferPass.lastFrameUsed(),
      threadedCommandWorkerCount,
      swapchain.getActivePresentMode(),
      [this] { rebuildProjection(); },
      [this] { shadowPass.updatePointShadowMatrices(sceneUbo); },
      [this] { taaHistoryValid = false; },
      [this](bool preferMailbox) {
        swapchain.setPreferMailbox(preferMailbox);
        framebufferResized = true;
      },
      [this](int index) { requestSceneLoad(index); }};
  ui.physics = &physicsPanel;
  characterPanel.characterLoaded = characterAsset.has_value();
  ui.character = &characterPanel;
  ui.bodiesInWorld = physics ? physics->bodyCount() : 0;
  ui.bodyLimit = physics ? physics->bodyLimit() : 0;
  ui.onSaveSettings = [this](const char *section) { saveSettings(section); };
  ui.onResetSettings = [this](const char *section) { resetSettings(section); };
  ui.settingsStatus = &settingsStatus;
  ui.playing = playing;
  ui.onTogglePlay = [this] { togglePlayMode(); };
  ui.sceneError = &sceneError;
  ui.scene = &currentScene;
  ui.selection = playing ? std::nullopt : selection;
  ui.selectionMass = selectionMass;
  ui.selectionBodies = selectionBodies;
  ui.onEntryMoved = [this](SceneEntryRef ref) { placeEntry(ref); };
  ui.onEntryChanged = [this](SceneEntryRef ref) {
    pendingEntryOps.push_back({EntryOp::Rebuild, ref});
  };
  ui.onEntryDeleted = [this](SceneEntryRef ref) {
    pendingEntryOps.push_back({EntryOp::Delete, ref});
  };
  ui.onGridSplit = [this](SceneEntryRef ref) {
    pendingEntryOps.push_back({EntryOp::Split, ref});
  };
  ui.viewProj = taaBaseProjection * sceneUbo.view;
  ui.sceneArea = sceneArea;
  ui.onEditFinished = [this] { recordEdit(); };
  ui.modelFiles = &modelFiles;
  ui.onSpawn = [this](SceneModelEntry entry) {
    if (!playing)
      pendingSpawn = std::move(entry);
  };
  ui.onDuplicate = [this] { duplicateSelection(); };
  ui.onSpawnPlayerStart = [this] {
    if (!playing)
      pendingPlayerStart = true;
  };
  ui.onRescanModels = [this] { discoverModels(); };
  ui.sceneName = &currentScene.name;
  ui.sceneUnsaved = sceneUnsaved;
  ui.onSave = [this] { requestSave(); };
  ui.onSaveAs = [this] { requestSaveAs(); };
  ui.onQuit = [this] { requestQuit(); };
  ui.saveStatus = &saveStatus;
  ui.prompt = &prompt;
  ui.onPromptAnswer = [this](PromptAnswer answer, const std::string &text) {
    answerPrompt(answer, text);
  };
  imguiLayer.buildFrame(ui);
}

void VulkanRenderer::recordImGuiCommands(VkCommandBuffer cmd, uint32_t imageIndex) {
  // Debug lines go in the ImGui pass, under the panels: the physics
  // wireframes (only built while F2 is on) and, in edit mode, the player
  // start and the selection's outline. Copied together only when both are
  // showing.
  std::vector<const std::vector<PhysicsDebugVertex> *> sources;
  if (physicsPanel.showWireframes && physics)
    sources.push_back(&physics->debugLines(glm::vec3(sceneUbo.cameraPosition)));
  if (!playing && !editorLines.empty())
    sources.push_back(&editorLines);
  if (playing && !skeletonLines.empty())
    sources.push_back(&skeletonLines);
  const std::vector<PhysicsDebugVertex> *lines = nullptr;
  if (sources.size() == 1) {
    lines = sources.front();
  } else if (sources.size() > 1) {
    debugLineScratch.clear();
    for (const auto *source : sources)
      debugLineScratch.insert(debugLineScratch.end(), source->begin(),
                              source->end());
    lines = &debugLineScratch;
  }
  const bool drawPhysicsLines = lines != nullptr;
  if (drawPhysicsLines)
    debugLinePass.upload(device, imageIndex, *lines);
  imguiLayer.record(
      cmd, renderPassManager.getImGuiRenderPass(), swapchain.getExtent(),
      imageIndex, [&](VkCommandBuffer c) {
        if (drawPhysicsLines)
          debugLinePass.record(c, imageIndex, sceneArea, swapchain.getExtent(),
                               taaBaseProjection * sceneUbo.view, kCameraNear,
                               imguiDrawDistance);
      });
  if (imageIndex < renderResources.frameCount()) {
    renderResources.noteLayout(renderResources.frame(imageIndex).swapchainImage,
                               VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
  }
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
