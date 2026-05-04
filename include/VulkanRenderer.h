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
  VkSampler shadowSampler = VK_NULL_HANDLE;

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
  float imguiDrawDistance = 2000.0f;
  float imguiLodNear = 15.0f;
  float imguiLodFar = 45.0f;
  float imguiPointShadowFar = 40.0f;
  // Fog tunables (mirrored into sceneUbo.fogParams so the shader can read).
  float imguiFogDensity = 0.0f;
  float imguiFogClamp = 0.0f;
  int imguiDebugMode = 0;
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
