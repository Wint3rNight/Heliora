#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "DescriptorManager.h"
#include "ImGuiLayer.h"
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

class CommandThreadPool;

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
  bool imguiWantsMouse() const { return imguiLayer.wantsMouse(); }

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

  // --- Bloom pyramid (HDR post process) ---
  // Six separately allocated full/half/quarter/... resolution images per swapchain
  // image. Compute shaders downsample litBuffer into the pyramid, then
  // upsample additively back to mip 0. The tonemap/TAA pass samples mip 0.
  static constexpr uint32_t BLOOM_MIP_COUNT = 6;
  struct BloomMipResources {
    VkExtent2D extent = {};
    std::vector<AllocatedImage> images;
    std::vector<ImageViewHandle> views;
  };
  VkFormat bloomFormat = VK_FORMAT_UNDEFINED;
  std::array<BloomMipResources, BLOOM_MIP_COUNT> bloomMips;
  VkSampler bloomSampler = VK_NULL_HANDLE;
  VkDescriptorSetLayout bloomSetLayout = VK_NULL_HANDLE;
  VkDescriptorPool bloomDescriptorPool = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> bloomDownsampleSets;
  std::vector<VkDescriptorSet> bloomUpsampleSets;
  VkPipelineLayout bloomDownsamplePipelineLayout = VK_NULL_HANDLE;
  VkPipelineLayout bloomUpsamplePipelineLayout = VK_NULL_HANDLE;
  VkPipeline bloomDownsamplePipeline = VK_NULL_HANDLE;
  VkPipeline bloomUpsamplePipeline = VK_NULL_HANDLE;

  // --- SSGI bounce history (HDR, persistent across frames) ---
  // Ring is one larger than the max frames-in-flight so frame N+K never
  // overwrites the image that an in-flight frame is still sampling.
  // The current frame's *output* is what lit.frag samples (set 1 binding 5),
  // so binding 5 must rotate with the history index.
  // See mds/sponza_visual_diagnosis.md N6/N9.
  std::vector<AllocatedImage>  ssgiHistoryImages;   // size MAX_FRAMES_DRAWS + 1
  std::vector<ImageViewHandle> ssgiHistoryViews;    // size MAX_FRAMES_DRAWS + 1
  std::vector<VkFramebuffer>   ssgiFramebuffers;    // one per history image
  // Sampler used to read ssgiHistoryPrev in ssgi.frag (set 2 binding 0).
  // CLAMP_TO_EDGE + LINEAR so reprojected UVs near the screen edge sample
  // the edge texel; the shader bounds-checks for true off-screen reject.
  VkSampler ssgiSampler = VK_NULL_HANDLE;

  // --- Async SSAO output (Phase 7.5) ---
  // One full-resolution single-channel image per swapchain image. The compute
  // queue writes it after the G-buffer submit; the graphics queue samples it
  // in lit.frag after waiting on the async timeline semaphore.
  VkFormat ssaoFormat = VK_FORMAT_UNDEFINED;
  std::vector<AllocatedImage> ssaoImages;
  std::vector<ImageViewHandle> ssaoViews;
  VkSampler ssaoResultSampler = VK_NULL_HANDLE;
  VkDescriptorSetLayout ssaoOutputSetLayout = VK_NULL_HANDLE;
  VkDescriptorPool ssaoDescriptorPool = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> ssaoOutputSets;
  VkPipelineLayout ssaoPipelineLayout = VK_NULL_HANDLE;
  VkPipeline ssaoPipeline = VK_NULL_HANDLE;

  // --- TAA history (HDR, persistent across frames) ---
  // Same ring sizing as SSGI: MAX_FRAMES_DRAWS + 1 avoids overwriting a
  // history image that a still-in-flight frame is sampling.
  std::vector<AllocatedImage> taaHistoryImages;   // size MAX_FRAMES_DRAWS + 1
  std::vector<ImageViewHandle> taaHistoryViews;   // size MAX_FRAMES_DRAWS + 1
  // Sampler used by the TAA shader to read history-prev and depth.
  // CLAMP_TO_EDGE so reprojected UVs that drift just past the edge fall
  // back gracefully (treated as off-screen by the shader's bounds check).
  VkSampler taaSampler = VK_NULL_HANDLE;
  // Composite framebuffers — 3 attachments (swap, colorBuffer, history).
  // Size: taaHistoryViews.size() * swapCount. Index:
  // historyIndex * swapCount + swapIdx.
  std::vector<VkFramebuffer> compositeFramebuffers;

  // --- Auto-exposure (histogram-based) ---
  // Two-pass compute: pass 1 builds a 256-bin log-luminance histogram of
  // litBuffer; pass 2 reduces it to a target exposure scalar (Lagarde
  // 2014: H = 1 / (9.6 × avgLum)). CPU reads the previous frame's result
  // each draw, lerps with an eye-adaptation time constant, and pushes
  // into sceneUbo.qualityToggles2.x (the same slot the manual EV slider
  // used to drive — the slider becomes a bias added on top).
  // Refs: bruop.github.io/exposure, PKRenderer.
  VkDescriptorSetLayout autoExpHistogramSetLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout autoExpExposureSetLayout  = VK_NULL_HANDLE;
  VkPipelineLayout      autoExpHistogramPipelineLayout = VK_NULL_HANDLE;
  VkPipelineLayout      autoExpExposurePipelineLayout  = VK_NULL_HANDLE;
  VkPipeline            autoExpHistogramPipeline = VK_NULL_HANDLE;
  VkPipeline            autoExpExposurePipeline  = VK_NULL_HANDLE;
  VkDescriptorPool      autoExpDescriptorPool    = VK_NULL_HANDLE;
  // One histogram set per swap image (binds litViews[image]); one
  // exposure set per frame-in-flight (binds the host-visible result for
  // that frame). Reset bin counts live across dispatches — exposure.comp
  // resets them after reducing.
  std::vector<VkDescriptorSet> autoExpHistogramSets;
  std::vector<VkDescriptorSet> autoExpExposureSets;
  // Device-local 256 × uint32 histogram, persistent across frames (the
  // exposure pass zeros each bin after reading it).
  AllocatedBuffer autoExpHistogramBuffer;
  // One small (16-byte) host-visible result buffer per frame-in-flight.
  // Persistently mapped; CPU reads the previous-frame value before the
  // current frame's compute dispatch overwrites it.
  std::vector<AllocatedBuffer> autoExpResultBuffers;
  std::vector<void *>          autoExpResultMapped;
  // Running CPU-side adapted exposure. Lerped each frame toward the
  // GPU-computed target with `tau` seconds time constant. Initialized
  // to 1.0 so the first few pre-compute frames render at neutral scale.
  float autoExpAdaptedValue = 1.0f;
  bool  autoExpEnabled = true;   // ImGui toggle; off → manual EV only.
  // Log-luminance range covered by the 256 bins. Values outside clip to
  // bin 1 or bin 254; the Sponza HDR composite under default lighting
  // sits roughly in [-6, +4] so we widen a touch on each side.
  static constexpr float kAutoExpMinLogLum   = -10.0f;
  static constexpr float kAutoExpMaxLogLum   = +4.0f;
  // Eye-adaptation time constant in seconds. ~1.5 s feels natural for
  // daylight adaptation; raise for slower / film-like behaviour.
  static constexpr float kAutoExpTauSeconds  = 1.5f;

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

  // --- GPU-driven rendering staging (Phase 7.6) ---
  // CPU-filled for now; the compute culling pass will consume this SSBO when
  // indirect draw generation is wired in.
  struct GpuDrivenFrameResources {
    AllocatedBuffer meshBuffer;
    AllocatedBuffer transformBuffer;
    AllocatedBuffer indirectBuffer;
    AllocatedBuffer noCullIndirectBuffer;
    AllocatedBuffer countBuffer;
    AllocatedBuffer noCullCountBuffer;
    AllocatedBuffer frustumBuffer;
    VkDeviceSize meshBufferSize = 0;
    VkDeviceSize transformBufferSize = 0;
    VkDeviceSize indirectBufferSize = 0;
    VkDeviceSize noCullIndirectBufferSize = 0;
    bool descriptorDirty = true;
  };

  struct GpuDrivenGeometryRange {
    const Mesh *mesh = nullptr;
    int lod = 0;
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    int32_t vertexOffset = 0;
  };

  std::vector<GpuDrivenFrameResources> gpuDrivenFrames;
  AllocatedBuffer gpuDrivenStaticVertexBuffer;
  AllocatedBuffer gpuDrivenStaticIndexBuffer;
  VkDeviceSize gpuDrivenStaticVertexBufferSize = 0;
  VkDeviceSize gpuDrivenStaticIndexBufferSize = 0;
  std::vector<Vertex> gpuDrivenStaticVertices;
  std::vector<uint32_t> gpuDrivenStaticIndices;
  std::vector<GpuDrivenGeometryRange> gpuDrivenGeometryRanges;
  std::vector<int> gpuDrivenModelIds;
  uint32_t gpuDrivenCandidateCount = 0;
  uint32_t gpuDrivenMeshCount = 0;
  std::vector<AllocatedImage> gpuDrivenHzbImages;
  std::vector<ImageViewHandle> gpuDrivenHzbViews;
  std::vector<std::vector<ImageViewHandle>> gpuDrivenHzbMipViews;
  std::vector<bool> gpuDrivenHzbValid;
  VkExtent2D gpuDrivenHzbExtent = {};
  uint32_t gpuDrivenHzbMipCount = 0;
  VkFormat gpuDrivenHzbFormat = VK_FORMAT_UNDEFINED;
  VkSampler gpuDrivenHzbSampler = VK_NULL_HANDLE;
  VkDescriptorSetLayout gpuDrivenSetLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout gpuDrivenHzbBuildSetLayout = VK_NULL_HANDLE;
  VkDescriptorPool gpuDrivenDescriptorPool = VK_NULL_HANDLE;
  VkDescriptorPool gpuDrivenHzbBuildDescriptorPool = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> gpuDrivenDescriptorSets;
  std::vector<VkDescriptorSet> gpuDrivenHzbBuildSets;
  VkPipelineLayout gpuCullPipelineLayout = VK_NULL_HANDLE;
  VkPipeline gpuCullPipeline = VK_NULL_HANDLE;
  VkPipelineLayout gpuDrivenHzbBuildPipelineLayout = VK_NULL_HANDLE;
  VkPipeline gpuDrivenHzbBuildPipeline = VK_NULL_HANDLE;
  VkPipelineLayout gpuDrivenGBufferPipelineLayout = VK_NULL_HANDLE;
  VkPipeline gpuDrivenGBufferPipeline = VK_NULL_HANDLE;
  VkPipeline gpuDrivenGBufferNoCullPipeline = VK_NULL_HANDLE;
  bool imguiGpuDrivenEnabled = true;
  bool imguiHzbCullingEnabled = true;
  int imguiGpuDrivenMinCandidates = 256;
  bool gpuDrivenLastFrameUsed = false;

  // --- Multi-threaded command recording (Phase 7.4) ---
  struct ThreadCommandFrameResources {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer gBufferSecondary = VK_NULL_HANDLE;
  };
  std::vector<std::vector<ThreadCommandFrameResources>> threadedCommandFrames;
  std::unique_ptr<CommandThreadPool> commandThreadPool;
  uint32_t threadedCommandWorkerCount = 0;

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
  // imagesInFlight maps swap image -> frame fence. frameImageInFlight is the
  // reverse owner map, used to clear stale image entries when a frame fence is
  // reused for a different acquired image.
  // See https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html
  std::vector<VkSemaphore> imageAvailable;
  std::vector<VkSemaphore> renderFinished;
  std::vector<VkSemaphore> asyncComputeTimeline;
  std::vector<uint64_t> asyncComputeTimelineValue;
  std::vector<VkFence> drawFences;
  std::vector<VkFence> imagesInFlight;
  std::vector<uint32_t> frameImageInFlight;

  // Split graphics/compute command buffers for Phase 7.5 async SSAO.
  // swapchain.getCommandBuffer(image) records the G-buffer submit; these
  // additional primaries record shadow work and post-lighting work.
  std::vector<VkCommandBuffer> shadowCommandBuffers;
  std::vector<VkCommandBuffer> postCommandBuffers;
  std::vector<VkCommandBuffer> ssaoCommandBuffers;
  bool framebufferResized = false;

  // --- ImGui ---
  ImGuiLayer imguiLayer;
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
  float imguiSpecAAVariance = 1.25f;       // tunable
  // Karis-style spec-AA threshold cap. 0.18 (the plan's KAPPA) leaves
  // visible per-pixel sparkle on Sponza's dense cloth weave / foliage at
  // glancing angles. Bumping to 0.5 lets the kernel saturate more, folding
  // sub-pixel normal variance into much higher effective roughness on
  // high-frequency surfaces while leaving low-variance surfaces unchanged.
  float imguiSpecAAThreshold = 1.0f;       // tunable
  // IBL specular mip floor. This only affects global cubemap specular, not
  // direct-light highlights. Keeping it moderately high prevents the
  // environment map's bright texels from reading as rectangular local lights
  // on glossy indoor floors.
  float imguiIblRoughnessFloor = 0.45f;    // tunable
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
  bool imguiCullShadowCasters = false;      // opt-in only: tight CSM caster
                                           // culling can remove off-slice
                                           // occluders that still cast into
                                           // the cascade.
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
  // SSGI (screen-space one-bounce diffuse) intensity. 0 = disabled.
  // Dropped 1.0 → 0.6 alongside the per-sample firefly clamp tightening
  // and the close-range fade: 1.0 over-emphasized the residual SSGI grain
  // on sun-lit floor pixels once auto-exposure started amplifying. 0.6
  // keeps the warm-bounce feel without the visible noise floor.
  float imguiSsgiIntensity = 0.6f;
  // SSGI sample quality. 8 is the default performance/quality balance; 12 is
  // the old high-quality path, useful for A/B checks when chasing artifacts.
  int imguiSsgiSamples = 8;
  bool imguiEnableSunDirect = true;
  bool imguiEnablePointLights = true;
  bool imguiEnableSpotLights = true;
  bool imguiEnableIblAmbient = true;
  bool imguiEnableSsgiBounce = true;
  bool imguiEnableBloom = true;
  float imguiBloomThreshold = 0.8f;
  float imguiBloomIntensity = 0.12f;
  float imguiBloomRadius = 1.25f;
  bool imguiSsrEnabled = false;            // off by default: avoids rough Sponza SSR leaks
  bool imguiTaaEnabled = false;
  bool imguiResponsiveTaa = true;          // drop history while camera moves
  bool imguiPreferMailboxPresent = true;   // on = MAILBOX / uncapped present
  // CAS-style sharpening strength. Dropped 0.4 → 0.10 — the previous 0.4
  // was amplifying the residual SSGI/spec-AA noise into a visible chunky
  // pattern on the floor. 0.10 still restores TAA-softening on real edges
  // without exaggerating noise. Push back toward 0.3–0.4 only if the
  // image looks too soft after the noise-reduction pass.
  float imguiSharpness = 0.10f;
  bool imguiDayNightEnable = false;        // day/night animation
  float imguiDayNightSpeed = 60.0f;        // sim-hours per real-second
  float imguiDayNightHour = 12.0f;         // current sim-hour [0..24)
  bool imguiUseGeomNormalOnly = false;     // P2 diag: bypass normal-map sampling

  // --- TAA state ---
  // Frame counter drives the Halton index and temporal history ring slots.
  // Jitter wraps every 8 frames; history slots wrap by their ring size.
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
  glm::vec3 taaLastCameraPos = glm::vec3(0.0f);
  glm::mat4 taaLastView = glm::mat4(1.0f);
  bool taaHasLastCamera = false;
  bool cameraMovedThisFrame = false;

  // --- Init helpers ---
  void createSynchronization();
  void createAsyncFrameCommandBuffers();
  void cleanupAsyncFrameCommandBuffers();
  void createThreadedCommandResources();
  void cleanupThreadedCommandResources();
  void createShadowResources();
  void cleanupShadowResources();
  void createGBuffer();
  void cleanupGBuffer();
  void createLitResources();
  void cleanupLitResources();
  void createBloomResources();
  void cleanupBloomResources();
  void recordBloomPass(VkCommandBuffer cmd, uint32_t currentImage);
  void createSsgiResources();
  void cleanupSsgiResources();
  void createSsaoResources();
  void cleanupSsaoResources();
  void createTaaResources();
  void cleanupTaaResources();
  void createCompositeFramebuffers();
  void cleanupCompositeFramebuffers();
  void createAutoExposureResources();
  void cleanupAutoExposureResources();
  void recordAutoExposurePass(VkCommandBuffer cmd, uint32_t currentImage);
  void createGpuDrivenResources();
  void cleanupGpuDrivenResources();
  void updateGpuDrivenDescriptorSet(uint32_t imageIndex);
  void recordGpuDrivenHzbBuild(VkCommandBuffer cmd, uint32_t currentImage);
  void initIBL();
  void cleanupIBL();
  void rebuildProjection();

  void updateLightSpaceMatrices();
  void updatePointShadowMatrices();
  void registerGpuDrivenModelGeometry(int modelId);
  const GpuDrivenGeometryRange *findGpuDrivenGeometry(const Mesh *mesh,
                                                      int lod) const;
  void uploadGpuDrivenStaticGeometry();
  void uploadGpuDrivenMeshRecords(GpuDrivenFrameResources &frame,
                                  const void *records, VkDeviceSize bytes,
                                  uint32_t recordCount);
  bool ensureGpuDrivenBuffer(AllocatedBuffer &buffer, VkDeviceSize &capacity,
                             VkDeviceSize requiredSize,
                             VkBufferUsageFlags usage);

  // --- Per-frame ---
  void recordCommands(uint32_t currentImage);
  void recordGBufferCommands(uint32_t currentImage);
  void recordShadowCommands(VkCommandBuffer cmd);
  void recordSsaoComputeCommands(VkCommandBuffer cmd, uint32_t currentImage);
  void recordPostCommands(VkCommandBuffer cmd, uint32_t currentImage);
  void recordGBufferPass(VkCommandBuffer cmd, uint32_t currentImage,
                         const VkViewport &viewport,
                         const VkRect2D &scissor);
  void recordShadowPass(VkCommandBuffer cmdBuffer);
  void recordPointShadowPass(VkCommandBuffer cmdBuffer);
  void recordImGuiCommands(VkCommandBuffer cmd, uint32_t imageIndex);
  void recreateSwapChain();
};
