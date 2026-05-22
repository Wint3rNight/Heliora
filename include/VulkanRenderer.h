#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

#include "DescriptorManager.h"
#include "Model.h"
#include "ModelManager.h"
#include "PerformanceMetrics.h"
#include "RenderPassManager.h"
#include "SceneNode.h"
#include "TextureManager.h"
#include "Utilities.h"
#include "VulkanDevice.h"
#include "VulkanPipeline.h"
#include "VulkanSwapchain.h"

struct InstancedDrawable {
  int modelId;
  std::vector<InstanceData> instances;
  AllocatedBuffer instanceBuffer;
  // World-space bounding sphere covering all instances. Used for whole-batch
  // frustum culling and LOD selection.
  glm::vec3 groupCenter = glm::vec3(0.0f);
  float groupRadius = 0.0f;
};

class VulkanRenderer {
public:
  VulkanRenderer();

  int init(GLFWwindow *newWindow);
  void draw();
  int createMeshModel(const std::string &modelFile);
  SceneNode &getRootNode();
  void updateCameraView(const glm::mat4 &viewMatrix,
                        const glm::vec3 &cameraPosition);
  void addInstancedModel(int modelId, const std::vector<glm::mat4> &transforms);
  void cleanup();
  void notifyResize();

  void setImGuiCameraInfo(glm::vec3 pos, float speed);
  void setFov(float fovDegrees);
  void setDrawDistance(float dist);
  float getCameraSpeed() const { return imguiCameraSpeed; }
  bool imguiWantsMouse() const { return ImGui::GetIO().WantCaptureMouse; }

  ~VulkanRenderer();

private:
  GLFWwindow *window = nullptr;
  int currentFrame = 0;

  SceneNode rootNode;
  SceneUniformBuffer sceneUbo = {};
  VkFormat shadowDepthFormat = VK_FORMAT_UNDEFINED;

  // --- Directional shadow (Cascaded Shadow Maps) ---
  AllocatedImage csmDepthImage;
  ImageViewHandle csmArrayView;
  std::vector<ImageViewHandle> csmLayerViews;
  std::vector<VkFramebuffer> csmFramebuffers;
  VkSampler shadowSampler = VK_NULL_HANDLE;       // plain — point shadow cube
  VkSampler csmShadowSampler = VK_NULL_HANDLE;    // compare-enabled — CSM HW PCF

  // --- Omnidirectional point shadow ---
  AllocatedImage pointShadowDepthImage;
  ImageViewHandle pointShadowCubeView;
  std::vector<ImageViewHandle> pointShadowFaceViews;
  std::vector<VkFramebuffer> pointShadowFramebuffers;
  std::vector<glm::mat4> pointShadowMatrices;

  // --- G-buffer (per swapchain image) ---
  VkFormat gBufferDepthFormat = VK_FORMAT_UNDEFINED;
  std::vector<AllocatedImage> gBuffer0Images, gBuffer1Images, gBuffer2Images,
      gBufferDepthImages;
  std::vector<ImageViewHandle> gBuffer0Views, gBuffer1Views, gBuffer2Views,
      gBufferDepthViews;
  std::vector<VkFramebuffer> gBufferFramebuffers;

  // --- Lit buffer (post-PBR HDR, sampled by SSR composite pass) ---
  VkFormat litFormat = VK_FORMAT_UNDEFINED;
  std::vector<AllocatedImage> litImages;
  std::vector<ImageViewHandle> litViews;
  std::vector<VkFramebuffer> litFramebuffers;
  VkSampler litSampler = VK_NULL_HANDLE;

  // --- TAA history (HDR, ping-pong, persistent across frames) ---
  // Two physical images that alternate roles each frame:
  //   frame N writes taaHistory[N&1], samples taaHistory[(N+1)&1]
  // taaFramebuffers has 2 * swap-image-count entries; index by:
  //   parity = taaFrameCounter & 1
  //   slot   = parity * swapCount + swapIndex
  std::vector<AllocatedImage> taaHistoryImages;   // size 2
  std::vector<ImageViewHandle> taaHistoryViews;   // size 2
  // Sampler used by the TAA shader to read history-prev and depth.
  // CLAMP_TO_EDGE so reprojected UVs that drift just past the edge fall
  // back gracefully (treated as off-screen by the shader's bounds check).
  VkSampler taaSampler = VK_NULL_HANDLE;
  // Composite framebuffers — 3 attachments (swap, colorBuffer, history).
  // Size: 2 * swapCount. Index: parity * swapCount + swapIdx, where
  // parity = taaFrameCounter & 1. The history attachment at slot s is
  // taaHistoryViews[parity], so each parity owns one of the two ping-pong
  // images as its write target this frame.
  std::vector<VkFramebuffer> compositeFramebuffers;

  // --- IBL resources ---
  int iblSkyboxImageIndex = -1; // index into TextureManager::textureImages
  int irradianceImageIndex = -1;
  int prefilteredEnvImageIndex = -1;
  int brdfLutImageIndex = -1;
  int ssaoNoiseImageIndex = -1;
  VkSampler iblSampler = VK_NULL_HANDLE;
  VkSampler ssaoNoiseSampler = VK_NULL_HANDLE;

  // --- Instanced drawables ---
  std::vector<InstancedDrawable> instancedDrawables;

  // --- Subsystems ---
  VulkanDevice device;
  VulkanSwapchain swapchain;
  RenderPassManager renderPassManager;
  DescriptorManager descriptorManager;
  VulkanPipeline pipeline;
  TextureManager textureManager;
  ModelManager modelManager;
  PerformanceMetrics metrics;

  // --- Synchronization ---
  // imageAvailable & drawFences are sized by MAX_FRAMES_DRAWS (frames in
  // flight). renderFinished is sized by swapchain image count and indexed
  // by the acquired imageIndex — the presentation engine binds its wait
  // to the swap image, not to our frame-in-flight slot. Indexing this by
  // currentFrame caused a validation hit when the swap engine had not
  // finished presenting the image whose semaphore we tried to reuse.
  // See https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html
  std::vector<VkSemaphore> imageAvailable;
  std::vector<VkSemaphore> renderFinished;
  std::vector<VkFence> drawFences;
  std::vector<VkFence> imagesInFlight;
  bool framebufferResized = false;

  // --- ImGui ---
  std::vector<VkFramebuffer> imguiFramebuffers;
  glm::vec3 imguiCameraPos = {};
  float imguiCameraSpeed = 15.0f;
  float imguiCameraFov = 45.0f;
  float imguiDrawDistance = 8000.0f;
  float imguiLodNear = 15.0f;
  float imguiLodFar = 45.0f;
  float imguiPointShadowFar = 40.0f;
  // Fog tunables (mirrored into sceneUbo.fogParams so the shader can read).
  float imguiFogDensity = 0.0f;
  float imguiFogClamp = 0.0f;
  int imguiDebugMode = 0;
  float imguiSpecAAVariance = 0.25f;       // tunable
  // Karis-style spec-AA threshold cap. 0.18 (the plan's KAPPA) leaves
  // visible per-pixel sparkle on Sponza's dense cloth weave / foliage at
  // glancing angles. Bumping to 0.5 lets the kernel saturate more, folding
  // sub-pixel normal variance into much higher effective roughness on
  // high-frequency surfaces while leaving low-variance surfaces unchanged.
  float imguiSpecAAThreshold = 0.5f;       // tunable
  // IBL specular mip floor. 0.3 forces the prefiltered env fetch to use at
  // least ~mip 1 of 4 — kills the per-pixel sky-reflection sparkle from
  // normal-mapped cloth at grazing.
  float imguiIblRoughnessFloor = 0.3f;     // tunable
  bool imguiShadowFrontFaceCull = false;   // off → CULL_NONE in shadow pass.
                                           // Front-face cull silently drops
                                           // any single-sided wall whose only
                                           // triangle faces the sun, so the
                                           // floor's shadow lookup misses the
                                           // wall as an occluder and reads as
                                           // lit. NONE captures both winding
                                           // orientations; slope-scaled bias
                                           // + receiver-plane gradient
                                           // already handle the acne that
                                           // front-cull was protecting.
  float imguiCsmFar = 2000.0f;             // cascade far plane
  float imguiIblIntensity = 1.0f;          // manual IBL/sky multiplier
  // Exposure stops applied BEFORE the ACES tonemap in second.frag. ACES has
  // a steep toe — without a pre-scale, midtones get crushed to black on
  // shadowed Sponza interiors. +1 to +2 stops typically matches the warm
  // reference look once the sky-occlusion proxy is also lifted.
  float imguiExposureEV = 0.0f;            // exposure stops, -3..+3
  // Min-roughness floor applied at the G-buffer write for non-metallic
  // surfaces only. Sponza glTF's roughness maps bottom out too low — that
  // produces the wet-floor banner reflection and stone-pillar glitter.
  // 0 = disabled (use authored roughness); ~0.35–0.5 is the Sponza sweet
  // spot. Metals (metallic > 0.5) are untouched so trim still reads glossy.
  float imguiMinSurfaceRoughness = 0.35f;
  // Sky-occlusion proxy floor — bottom of the IBL multiplier in shadowed
  // pixels. 0.30 = old conservative ("dead-black interior"), 0.55 = new
  // default ("warm interior"), 1.0 = unbounded (full sky everywhere).
  float imguiSkyOcclusionFloor = 0.55f;
  bool imguiDayNightEnable = false;        // day/night animation
  float imguiDayNightSpeed = 60.0f;        // sim-hours per real-second
  float imguiDayNightHour = 12.0f;         // current sim-hour [0..24)
  bool imguiUseGeomNormalOnly = false;     // P2 diag: bypass normal-map sampling

  // --- TAA state ---
  // Frame counter drives the Halton index and the history ping-pong parity.
  // Wraps every 8 frames for jitter (matches Halton sample count) but the
  // ping-pong only needs parity, so we can just use frameCounter & 1.
  uint32_t taaFrameCounter = 0;
  // Un-jittered baseline projection. Updated only by rebuildProjection().
  // Each draw() starts by copying this into sceneUbo.projection BEFORE
  // applying the per-frame Halton jitter, so the jitter doesn't accumulate
  // across frames into a random-walk drift of the projection matrix.
  glm::mat4 taaBaseProjection = glm::mat4(1.0f);
  // Jittered VP of the previous frame. Used by the TAA shader to
  // reproject each pixel's world position to where it landed in last
  // frame's image (which was rendered with last frame's jitter).
  glm::mat4 taaPrevViewProj = glm::mat4(1.0f);
  bool taaHistoryValid = false;            // dropped on swapchain recreate / first frame
  float frameTimeGraphData[128] = {};
  int frameTimeGraphOffset = 0;

  // --- Init helpers ---
  void createSynchronization();
  void createShadowResources();
  void cleanupShadowResources();
  void createGBuffer();
  void cleanupGBuffer();
  void createLitResources();
  void cleanupLitResources();
  void createTaaResources();
  void cleanupTaaResources();
  void createCompositeFramebuffers();
  void cleanupCompositeFramebuffers();
  void initIBL();
  void cleanupIBL();
  void rebuildProjection();
  void initImGui();
  void cleanupImGui();
  void createImGuiFramebuffers();
  void cleanupImGuiFramebuffers();

  void updateLightSpaceMatrices();
  void updatePointShadowMatrices();

  // --- Per-frame ---
  void recordCommands(uint32_t currentImage);
  void recordShadowPass(VkCommandBuffer cmdBuffer);
  void recordPointShadowPass(VkCommandBuffer cmdBuffer);
  void recordImGuiCommands(VkCommandBuffer cmd, uint32_t imageIndex);
  void buildImGuiUI();
  void recreateSwapChain();
};
