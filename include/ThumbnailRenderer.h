#pragma once

#include "Thumbnails.h"
#include "Utilities.h"

#include <vulkan/vulkan.h>

#include <optional>
#include <string>
#include <unordered_map>

class MeshModel;
class VulkanDevice;

// The Assets panel's pictures: an asset drawn on its own into a 128x128
// image with one fixed light, copied into a cell of a shared 2048x2048
// atlas (one texture for every tile), and saved as a PNG so it's drawn only
// once. Work happens in one-off command buffers that wait for the GPU: the
// caller makes sure nothing in flight reads the atlas (vkDeviceWaitIdle).
class ThumbnailRenderer {
public:
  void create(const VulkanDevice &device, VkPipelineCache cache,
              VkDescriptorSetLayout bindlessLayout);
  void cleanup(VkDevice device);

  VkImageView atlasView() const { return atlasImageView.get(); }
  // The cell holding key's picture, if it has one.
  std::optional<int> cellOf(const std::string &key) const;
  bool full() const;

  // Key's picture from folder/key.png into a free cell; false if there's
  // no such file, it isn't a thumbnail, or the atlas is full.
  bool loadCached(const VulkanDevice &device, const std::string &key,
                  const std::string &folder);
  // The model drawn into a free cell, framed around its bounds, and saved
  // as folder/key.png; false if the atlas is full.
  bool render(const VulkanDevice &device, const std::string &key, const MeshModel &model,
              glm::vec3 boundsMin, glm::vec3 boundsMax, VkDescriptorSet bindlessSet,
              const std::string &folder);

private:
  int takeCell(const std::string &key);
  void copyIntoAtlas(VkCommandBuffer cmd, int cell, bool fromTile);

  AllocatedImage atlasImage;
  ImageViewHandle atlasImageView;
  AllocatedImage tileImage;
  ImageViewHandle tileView;
  AllocatedImage depthImage;
  ImageViewHandle depthView;
  AllocatedBuffer staging; // one picture, mapped, both ways
  void *stagingData = nullptr;
  VkRenderPass renderPass = VK_NULL_HANDLE;
  VkFramebuffer framebuffer = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  std::unordered_map<std::string, int> cells;
  int nextCell = 0;
};
