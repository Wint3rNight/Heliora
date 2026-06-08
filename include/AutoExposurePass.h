#pragma once

#include "Utilities.h"
#include "VulkanDevice.h"

#include <chrono>
#include <vector>
#include <vulkan/vulkan.h>

class AutoExposurePass {
public:
  void create(VulkanDevice &device, VkExtent2D extent,
              const std::vector<ImageViewHandle> &litViews,
              VkSampler litSampler, VkPipelineCache pipelineCache);
  void cleanup();

  void record(VkCommandBuffer cmd, uint32_t currentImage, int currentFrame,
              const std::vector<AllocatedImage> &litImages, bool enabled,
              bool litInputReadyForCompute);

  float updateExposureScale(int currentFrame, bool enabled, float exposureEV);
  float adaptedValue() const { return adaptedExposure; }
  void resetAdaptation(float exposureScale = 1.0f);

private:
  static constexpr float kMinLogLum = -10.0f;
  static constexpr float kMaxLogLum = 4.0f;
  static constexpr float kTauSeconds = 1.5f;

  VulkanDevice *device = nullptr;
  VkExtent2D renderExtent = {};

  VkDescriptorSetLayout histogramSetLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout exposureSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout histogramPipelineLayout = VK_NULL_HANDLE;
  VkPipelineLayout exposurePipelineLayout = VK_NULL_HANDLE;
  VkPipeline histogramPipeline = VK_NULL_HANDLE;
  VkPipeline exposurePipeline = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> histogramSets;
  std::vector<VkDescriptorSet> exposureSets;

  AllocatedBuffer histogramBuffer;
  std::vector<AllocatedBuffer> resultBuffers;
  std::vector<void *> resultMapped;

  float adaptedExposure = 1.0f;
  bool hasLastUpdate = false;
  std::chrono::steady_clock::time_point lastUpdate;
};
