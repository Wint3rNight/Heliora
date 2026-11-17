#pragma once

#include "Thumbnails.h"
#include "Utilities.h"

#include <vulkan/vulkan.h>

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

class MeshModel;
class VulkanDevice;

// One indexed draw of a thumbnail: a model's mesh, or the character's mesh
// skinned into a buffer of its own.
struct ThumbnailDraw {
  VkBuffer vertices = VK_NULL_HANDLE;
  VkBuffer indices = VK_NULL_HANDLE;
  uint32_t indexCount = 0;
  uint32_t albedo = 0; // bindless index
  bool alphaMasked = false;
};
std::vector<ThumbnailDraw> thumbnailDraws(const MeshModel &model);

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
  // The draws rendered into a free cell, framed around the bounds, and
  // saved as folder/key.png; false if the atlas is full. before records
  // into the same command buffer first (skinning the character).
  bool render(const VulkanDevice &device, const std::string &key,
              const std::vector<ThumbnailDraw> &draws, glm::vec3 boundsMin, glm::vec3 boundsMax,
              VkDescriptorSet bindlessSet, const std::string &folder,
              const std::function<void(VkCommandBuffer)> &before = {});

  // The live preview: drawn into an image of its own inside the frame's
  // command buffer, before the UI samples it (the hovered clip, playing).
  VkImageView liveView() const { return liveImageView.get(); }
  void recordLive(VkCommandBuffer cmd, const std::vector<ThumbnailDraw> &draws,
                  glm::vec3 boundsMin, glm::vec3 boundsMax, VkDescriptorSet bindlessSet);

private:
  int takeCell(const std::string &key);
  void copyIntoAtlas(VkCommandBuffer cmd, int cell, bool fromTile);
  void recordDraws(VkCommandBuffer cmd, VkRenderPass pass, VkFramebuffer target,
                   const std::vector<ThumbnailDraw> &draws, glm::vec3 boundsMin,
                   glm::vec3 boundsMax, VkDescriptorSet bindlessSet);

  AllocatedImage atlasImage;
  ImageViewHandle atlasImageView;
  AllocatedImage tileImage;
  ImageViewHandle tileView;
  AllocatedImage depthImage;
  ImageViewHandle depthView;
  AllocatedImage liveImage;
  ImageViewHandle liveImageView;
  VkRenderPass liveRenderPass = VK_NULL_HANDLE; // ends ready to sample
  VkFramebuffer liveFramebuffer = VK_NULL_HANDLE;
  AllocatedBuffer staging; // one picture, mapped, both ways
  void *stagingData = nullptr;
  VkRenderPass renderPass = VK_NULL_HANDLE;
  VkFramebuffer framebuffer = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  std::unordered_map<std::string, int> cells;
  int nextCell = 0;
};
