#pragma once

#include "DescriptorManager.h"
#include "ModelManager.h"
#include "SceneNode.h"
#include "Utilities.h"
#include "VulkanDevice.h"
#include "VulkanPipeline.h"

#include <glm/glm.hpp>
#include <vector>
#include <vulkan/vulkan.h>

class ShadowPass {
public:
  void create(VulkanDevice &device, VkRenderPass renderPass,
              VkFormat depthFormat);
  void cleanup();

  void updateLightSpaceMatrices(SceneUniformBuffer &sceneUbo, float csmFar,
                                float drawDistance);
  void updatePointShadowMatrices(SceneUniformBuffer &sceneUbo);

  void recordCsm(VkCommandBuffer cmd, SceneNode &rootNode,
                 ModelManager &modelManager, const VulkanPipeline &pipeline,
                 const DescriptorManager &descriptorManager,
                 const SceneUniformBuffer &sceneUbo, bool frontFaceCull,
                 bool cullShadowCasters);
  void recordPoint(VkCommandBuffer cmd, SceneNode &rootNode,
                   ModelManager &modelManager, const VulkanPipeline &pipeline,
                   const DescriptorManager &descriptorManager,
                   bool frontFaceCull);

  VkImageView csmView() const { return csmArrayView.get(); }
  VkImageView pointCubeView() const { return pointShadowCubeView.get(); }
  VkSampler pointSampler() const { return shadowSampler; }
  VkSampler csmSampler() const { return csmShadowSampler; }

private:
  VulkanDevice *device = nullptr;
  VkRenderPass renderPass = VK_NULL_HANDLE;
  VkFormat depthFormat = VK_FORMAT_UNDEFINED;

  AllocatedImage csmDepthImage;
  ImageViewHandle csmArrayView;
  std::vector<ImageViewHandle> csmLayerViews;
  std::vector<VkFramebuffer> csmFramebuffers;
  VkSampler shadowSampler = VK_NULL_HANDLE;
  VkSampler csmShadowSampler = VK_NULL_HANDLE;

  AllocatedImage pointShadowDepthImage;
  ImageViewHandle pointShadowCubeView;
  std::vector<ImageViewHandle> pointShadowFaceViews;
  std::vector<VkFramebuffer> pointShadowFramebuffers;
  std::vector<glm::mat4> pointShadowMatrices;
};
