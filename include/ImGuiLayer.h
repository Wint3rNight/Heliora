#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include <functional>
#include <string>
#include <vector>

#include "PerformanceMetrics.h"
#include "Utilities.h"
#include "VulkanDevice.h"

struct DebugUiContext {
  PerformanceMetrics &metrics;
  VmaAllocator allocator = VK_NULL_HANDLE;
  SceneUniformBuffer &sceneUbo;

  glm::vec3 &cameraPos;
  float &cameraSpeed;
  float &cameraFov;
  float &drawDistance;
  float &lodNear;
  float &lodFar;
  float &pointShadowFar;
  float &fogDensity;
  float &fogClamp;
  int &debugMode;
  float &specAAVariance;
  float &specAAThreshold;
  float &iblRoughnessFloor;
  bool &shadowFrontFaceCull;
  bool &cullShadowCasters;
  float &csmFar;
  float &iblIntensity;
  float &exposureEV;
  float &minSurfaceRoughness;
  float &skyOcclusionFloor;
  float &ssgiIntensity;
  int &ssgiSamples;
  bool &enableSunDirect;
  bool &enablePointLights;
  bool &enableSpotLights;
  bool &enableIblAmbient;
  bool &enableSsgiBounce;
  bool &enableBloom;
  float &bloomThreshold;
  float &bloomIntensity;
  float &bloomRadius;
  bool &ssrEnabled;
  bool &taaEnabled;
  bool &responsiveTaa;
  bool &preferMailboxPresent;
  float &sharpness;
  float &edgeAA;
  float &alphaDither;
  float &shadowSoftness;
  float &contactGrounding;
  bool &dayNightEnable;
  float &dayNightSpeed;
  float &dayNightHour;
  float &normalStrength;
  bool &useGeomNormalOnly;
  bool &gpuDrivenEnabled;
  bool &hzbCullingEnabled;
  int &gpuDrivenMinCandidates;
  bool &threadedGBufferEnabled;
  bool &autoExposureEnabled;

  // Scene picker: display names of discovered .scene files and the active
  // selection. Selection changes go through onSceneSelected, which defers
  // the actual switch to the top of the next frame.
  const std::vector<std::string> *sceneNames = nullptr;
  int activeSceneIndex = -1;

  float autoExposureAdaptedValue = 1.0f;
  MaterialProbeResult materialProbe = {};
  uint32_t gpuDrivenCandidateCount = 0;
  uint32_t gpuDrivenMeshCount = 0;
  bool gpuDrivenLastFrameUsed = false;
  uint32_t threadedGBufferWorkers = 0;
  VkPresentModeKHR activePresentMode = VK_PRESENT_MODE_FIFO_KHR;

  std::function<void()> onProjectionChanged;
  std::function<void()> onDirectionalLightChanged;
  std::function<void()> onPointShadowChanged;
  std::function<void()> onTaaHistoryInvalidated;
  std::function<void()> onSponzaReferencePreset;
  std::function<void(bool)> onPresentModeChanged;
  std::function<void(int)> onSceneSelected;
};

class ImGuiLayer {
public:
  void init(GLFWwindow *window, VulkanDevice &device, VkRenderPass renderPass,
            uint32_t imageCount);
  void cleanup(VkDevice device);

  void createFramebuffers(VkDevice device, VkRenderPass renderPass,
                          VkExtent2D extent,
                          const std::vector<SwapChainImage> &images);
  void cleanupFramebuffers(VkDevice device);

  void record(VkCommandBuffer cmd, VkRenderPass renderPass, VkExtent2D extent,
              uint32_t imageIndex, DebugUiContext &ui);

  bool wantsMouse() const;

private:
  void buildUi(DebugUiContext &ui);

  std::vector<VkFramebuffer> framebuffers;
  float frameTimeGraphData[128] = {};
  int frameTimeGraphOffset = 0;
  bool initialized = false;
};
