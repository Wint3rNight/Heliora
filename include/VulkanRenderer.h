#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "DescriptorManager.h"
#include "PerformanceMetrics.h"
#include "Model.h"
#include "ModelManager.h"
#include "SceneNode.h"
#include "TextureManager.h"
#include "RenderPassManager.h"
#include "Utilities.h"
#include "VulkanDevice.h"
#include "VulkanPipeline.h"
#include "VulkanSwapchain.h"

struct InstancedDrawable {
  int                    modelId;
  std::vector<InstanceData> instances;
  AllocatedBuffer        instanceBuffer;
};

class VulkanRenderer {
public:
  VulkanRenderer();

  int init(GLFWwindow *newWindow);
  void draw();
  int createMeshModel(const std::string &modelFile);
  SceneNode &getRootNode();
  void updateCameraView(const glm::mat4 &viewMatrix, const glm::vec3 &cameraPosition);
  void addInstancedModel(int modelId, const std::vector<glm::mat4> &transforms);
  void cleanup();
  void notifyResize();

  ~VulkanRenderer();

private:
  GLFWwindow *window = nullptr;
  int currentFrame   = 0;

  SceneNode           rootNode;
  SceneUniformBuffer  sceneUbo = {};
  VkFormat            shadowDepthFormat = VK_FORMAT_UNDEFINED;

  // --- Directional shadow (Cascaded Shadow Maps) ---
  AllocatedImage               csmDepthImage;
  ImageViewHandle              csmArrayView;
  std::vector<ImageViewHandle> csmLayerViews;
  std::vector<VkFramebuffer>   csmFramebuffers;
  VkSampler                    shadowSampler = VK_NULL_HANDLE;

  // --- Omnidirectional point shadow ---
  AllocatedImage             pointShadowDepthImage;
  ImageViewHandle            pointShadowCubeView;
  std::vector<ImageViewHandle> pointShadowFaceViews;
  std::vector<VkFramebuffer>   pointShadowFramebuffers;
  std::vector<glm::mat4>       pointShadowMatrices;

  // --- G-buffer (per swapchain image) ---
  VkFormat gBufferDepthFormat = VK_FORMAT_UNDEFINED;
  std::vector<AllocatedImage>  gBuffer0Images,  gBuffer1Images,  gBuffer2Images,  gBufferDepthImages;
  std::vector<ImageViewHandle> gBuffer0Views,   gBuffer1Views,   gBuffer2Views,   gBufferDepthViews;
  std::vector<VkFramebuffer>   gBufferFramebuffers;
  std::vector<VkFramebuffer>   deferredFramebuffers; // colorBuffer-only, for deferred subpass 0

  // --- IBL resources ---
  int iblSkyboxImageIndex     = -1; // index into TextureManager::textureImages
  int irradianceImageIndex    = -1;
  int prefilteredEnvImageIndex = -1;
  int brdfLutImageIndex       = -1;
  int ssaoNoiseImageIndex     = -1;
  VkSampler iblSampler       = VK_NULL_HANDLE;
  VkSampler ssaoNoiseSampler = VK_NULL_HANDLE;

  // --- Instanced drawables ---
  std::vector<InstancedDrawable> instancedDrawables;

  // --- Subsystems ---
  VulkanDevice      device;
  VulkanSwapchain   swapchain;
  RenderPassManager renderPassManager;
  DescriptorManager descriptorManager;
  VulkanPipeline    pipeline;
  TextureManager    textureManager;
  ModelManager      modelManager;
  PerformanceMetrics metrics;

  // --- Synchronization ---
  std::vector<VkSemaphore> imageAvailable;
  std::vector<VkSemaphore> renderFinished;
  std::vector<VkFence>     drawFences;
  std::vector<VkFence>     imagesInFlight;
  bool framebufferResized = false;

  // --- Init helpers ---
  void createSynchronization();
  void createShadowResources();
  void cleanupShadowResources();
  void createGBuffer();
  void cleanupGBuffer();
  void initIBL();
  void cleanupIBL();

  void updateLightSpaceMatrices();
  void updatePointShadowMatrices();

  // --- Per-frame ---
  void recordCommands(uint32_t currentImage);
  void recordShadowPass(VkCommandBuffer cmdBuffer);
  void recordPointShadowPass(VkCommandBuffer cmdBuffer);
  void recreateSwapChain();
};
