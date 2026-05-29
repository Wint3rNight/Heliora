#pragma once

#include <fstream>
#define GLFW_INCLUDE_VULKAN
#include "vk_mem_alloc.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <stdexcept>
#include <utility>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

class AllocatedBuffer {
public:
  AllocatedBuffer() = default;
  AllocatedBuffer(VmaAllocator allocator, VkBuffer buffer,
                  VmaAllocation allocation)
      : allocator(allocator), buffer(buffer), allocation(allocation) {}

  ~AllocatedBuffer() { reset(); }

  AllocatedBuffer(const AllocatedBuffer &) = delete;
  AllocatedBuffer &operator=(const AllocatedBuffer &) = delete;

  AllocatedBuffer(AllocatedBuffer &&other) noexcept { swap(other); }

  AllocatedBuffer &operator=(AllocatedBuffer &&other) noexcept {
    if (this != &other) {
      reset();
      swap(other);
    }
    return *this;
  }

  void reset() {
    if (buffer != VK_NULL_HANDLE) {
      vmaDestroyBuffer(allocator, buffer, allocation);
    }
    allocator = VK_NULL_HANDLE;
    buffer = VK_NULL_HANDLE;
    allocation = VK_NULL_HANDLE;
  }

  VkBuffer get() const { return buffer; }
  VmaAllocation getAllocation() const { return allocation; }
  explicit operator bool() const { return buffer != VK_NULL_HANDLE; }

private:
  void swap(AllocatedBuffer &other) noexcept {
    std::swap(allocator, other.allocator);
    std::swap(buffer, other.buffer);
    std::swap(allocation, other.allocation);
  }

  VmaAllocator allocator = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
};

class AllocatedImage {
public:
  AllocatedImage() = default;
  AllocatedImage(VmaAllocator allocator, VkImage image,
                 VmaAllocation allocation)
      : allocator(allocator), image(image), allocation(allocation) {}

  ~AllocatedImage() { reset(); }

  AllocatedImage(const AllocatedImage &) = delete;
  AllocatedImage &operator=(const AllocatedImage &) = delete;

  AllocatedImage(AllocatedImage &&other) noexcept { swap(other); }

  AllocatedImage &operator=(AllocatedImage &&other) noexcept {
    if (this != &other) {
      reset();
      swap(other);
    }
    return *this;
  }

  void reset() {
    if (image != VK_NULL_HANDLE) {
      vmaDestroyImage(allocator, image, allocation);
    }
    allocator = VK_NULL_HANDLE;
    image = VK_NULL_HANDLE;
    allocation = VK_NULL_HANDLE;
  }

  VkImage get() const { return image; }
  VmaAllocation getAllocation() const { return allocation; }
  explicit operator bool() const { return image != VK_NULL_HANDLE; }

private:
  void swap(AllocatedImage &other) noexcept {
    std::swap(allocator, other.allocator);
    std::swap(image, other.image);
    std::swap(allocation, other.allocation);
  }

  VmaAllocator allocator = VK_NULL_HANDLE;
  VkImage image = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
};

class ImageViewHandle {
public:
  ImageViewHandle() = default;
  ImageViewHandle(VkDevice device, VkImageView imageView)
      : device(device), imageView(imageView) {}

  ~ImageViewHandle() { reset(); }

  ImageViewHandle(const ImageViewHandle &) = delete;
  ImageViewHandle &operator=(const ImageViewHandle &) = delete;

  ImageViewHandle(ImageViewHandle &&other) noexcept { swap(other); }

  ImageViewHandle &operator=(ImageViewHandle &&other) noexcept {
    if (this != &other) {
      reset();
      swap(other);
    }
    return *this;
  }

  void reset() {
    if (imageView != VK_NULL_HANDLE) {
      vkDestroyImageView(device, imageView, nullptr);
    }
    device = VK_NULL_HANDLE;
    imageView = VK_NULL_HANDLE;
  }

  VkImageView get() const { return imageView; }
  explicit operator bool() const { return imageView != VK_NULL_HANDLE; }

private:
  void swap(ImageViewHandle &other) noexcept {
    std::swap(device, other.device);
    std::swap(imageView, other.imageView);
  }

  VkDevice device = VK_NULL_HANDLE;
  VkImageView imageView = VK_NULL_HANDLE;
};

const int MAX_FRAMES_DRAWS = 2;
const int MAX_OBJECTS = 40;
const int MAX_POINT_LIGHTS = 4;
const int MAX_SPOT_LIGHTS = 2;
const uint32_t SHADOW_MAP_SIZE = 2048;
const uint32_t POINT_SHADOW_MAP_SIZE = 1024;
constexpr int NUM_CSM_CASCADES = 4;

const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    // Phase 7.2 — bindless: single global texture array, indexed in shader,
    // populated as textures load. Replaces the per-material descriptor sets
    // that we used to allocate one-per-Material.
    VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME};

// Bindless texture array size. 4096 slots is plenty for Sponza (~75 unique
// textures) + DamagedHelmet + headroom. The shader declares a `[]` array
// and `VARIABLE_DESCRIPTOR_COUNT` lets us bind only as many as we allocate.
constexpr uint32_t MAX_BINDLESS_TEXTURES = 4096;

// vertex data representation
struct Vertex {
  glm::vec3 pos; // position attribute for the vertex
  glm::vec3 col; // color attribute for the vertex
  glm::vec2 tex; // texture coordinate attribute
  glm::vec3 normal;
  glm::vec3 tangent;
  glm::vec3 bitangent;
};

struct DirectionalLight {
  alignas(16) glm::vec4 direction;
  alignas(16) glm::vec4 colorIntensity;
};

struct PointLight {
  alignas(16) glm::vec4 position;
  alignas(16) glm::vec4 colorIntensity;
};

struct SpotLight {
  alignas(16) glm::vec4 position;
  alignas(16) glm::vec4 direction;
  alignas(16) glm::vec4 colorIntensity;
  alignas(16) glm::vec4 cutoffAngles;
};

struct SceneUniformBuffer {
  alignas(16) glm::mat4 projection;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 lightSpaceMatrices[NUM_CSM_CASCADES];
  alignas(16) glm::mat4 pointShadowMatrices[6];
  alignas(16) glm::vec4 cameraPosition;
  alignas(16) glm::vec4 cascadeSplits;
  alignas(16) DirectionalLight directionalLight;
  alignas(16) PointLight pointLights[MAX_POINT_LIGHTS];
  alignas(16) SpotLight spotLights[MAX_SPOT_LIGHTS];
  alignas(16) glm::ivec4 lightCounts;
  alignas(16) glm::vec4 shadowParams;
  alignas(16) glm::mat4 invProj;
  alignas(16) glm::mat4 invView;
  // (density, falloff, clampMax, ssgiSamples) — exposed via ImGui at runtime.
  alignas(16) glm::vec4 fogParams = glm::vec4(0.002f, 0.25f, 0.4f, 8.0f);
  int debugMode = 0;
  // Tunable shader parameters (visual-fix toggles dropped — fixes are
  // permanent now).
  // x = IBL roughness floor (suppresses off-light sparkle on textured
  //     surfaces without affecting direct-light specular)
  // y = sky-occlusion floor (mix(y, 1.0, 1-sunShadow) — bottom of the
  //     IBL ambient scale in shadowed regions; trades "wall light leak"
  //     vs. "dead-black interior". 0.30 = original conservative, 0.55 =
  //     reference-matching warmth.)
  // z = SPEC_AA_VARIANCE
  // w = SPEC_AA_THRESHOLD
  // SPEC_AA_VARIANCE bumped 0.25 → 0.75 — the multi-pixel normal-map sparkle
  // on Sponza stone wasn't sub-pixel (where AA was already firing) but
  // multi-pixel; raising the variance multiplier makes the AA kernel reach
  // its threshold cap at smaller normal-map gradients, widening the
  // effective spec lobe on textured surfaces. SPEC_AA_THRESHOLD synced to
  // 0.5 to match the imgui slider's actual default value.
  // Spec-AA variance 0.75 → 1.25 and threshold cap 0.5 → 1.0. Auto-exposure
  // amplified the previous gap: close-range floor pixels under bright sun
  // had strong per-pixel normal-map gradients that the old kernel didn't
  // saturate on, and CAS+auto-exposure surfaced the residual shimmer as
  // the chunky pattern in the floor close-up. The new defaults make the
  // AA kernel saturate on milder gradients and let it widen nearly to
  // matte where the noise is dense.
  alignas(16) glm::vec4 qualityToggles = glm::vec4(0.45f, 0.55f, 1.25f, 1.0f);
  // x = exposure linear multiplier (= exp2(EV stops); applied in second.frag
  //     before ACES tonemap; 1.0 = neutral).
  // y = "use geometric normal only" diagnostic (1 = bypass normal-map).
  // z = minimum surface roughness floor — applied in shader.frag (g-buffer
  //     write) for NON-METALLIC surfaces. Kills wet-floor banner reflections
  //     and stone-column sparkle by raising the authored roughness floor.
  //     0 = disabled. Typical: 0.35–0.5 for Sponza floors.
  // w = ambient intensity (drives IBL + skybox dimming so night doesn't
  //     reflect baked daylight off every glossy surface).
  alignas(16) glm::vec4 qualityToggles2 = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);
  // TAA state. Populated by VulkanRenderer each frame.
  //   prevViewProj: previous frame's jittered projection*view. Reprojection
  //     samples history where the surface actually landed in the previous
  //     jittered frame.
  //   taaParams.xy: current frame's sub-pixel jitter in clip-space NDC units
  //     (i.e. already divided by viewport*2). The CPU bakes this into the
  //     projection; second.frag reconstructs with the matching inverse
  //     projection, so it does not subtract jitter manually.
  //   taaParams.z:  taaEnable (0 = passthrough, 1 = TAA on).
  //   taaParams.w:  historyValid (0 on first frame / after resize / on debug
  //     mode change).
  //   viewportSize: pixel dimensions of the screen — needed because the TAA
  //     shader can't query a sampler bound as input attachment.
  alignas(16) glm::mat4 prevViewProj = glm::mat4(1.0f);
  alignas(16) glm::vec4 taaParams = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
  alignas(16) glm::vec4 viewportSize = glm::vec4(0.0f);
};

const int IBL_PREFILTER_MIPS = 5;

// Push constants for the G-buffer pass. Used by both shader.vert and
// shader.frag — vertex needs the matrices, fragment needs the bindless
// texture indices (5 maps × uint32 = 20 bytes packed into one ivec4 + uint).
// Total = 128 (matrices) + 32 (indices) = 160 bytes, well under the 256-byte
// guaranteed maxPushConstantsSize.
struct ModelPushConstants {
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 normal;
  // (albedo, normal, metallic, roughness, ao) bindless indices into the
  // global texture array. 0xFFFFFFFF reserved as "no texture" sentinel —
  // shader path-checks against that and falls back to a literal value.
  alignas(16) glm::uvec4 texIdx0; // (albedo, normal, metallic, roughness)
  // texIdx1 = (ao, materialFlags, alphaCutoff255, _)
  // materialFlags bit 0 = cloth, bit 1 = glTF alphaMode MASK.
  alignas(16) glm::uvec4 texIdx1;
};

struct InstanceData {
  glm::mat4 model;
  glm::mat4 normal;
};

struct ShadowPushConstants {
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 lightSpaceMatrix;
  // Phase 7.2: bindless albedo index for alpha-test discard on foliage.
  uint32_t albedoIdx;
  uint32_t materialFlags;
  uint32_t alphaCutoff255;
};

// Keep only the general utility structures here
struct QueueFamilyIndices {
  int graphicsFamily = -1;     // location of the graphics queue family
  int presentationFamily = -1; // location of the presentation queue family
  int computeFamily = -1;      // compute-capable queue, dedicated if available

  // check if the queue family indices are valid
  bool isValid() const {
    return graphicsFamily >= 0 && presentationFamily >= 0 &&
           computeFamily >= 0;
  }

  bool hasDedicatedCompute() const {
    return computeFamily >= 0 && computeFamily != graphicsFamily;
  }
};

struct SwapChainDetails {
  VkSurfaceCapabilitiesKHR
      surfaceCapabilities; // surafece properties like image size/extent
  std::vector<VkSurfaceFormatKHR>
      formats; // surface image formats, color depth, etc
  std::vector<VkPresentModeKHR>
      presentModes; // how images should be presented to the screen
};

struct SwapChainImage {
  VkImage image = VK_NULL_HANDLE;
  ImageViewHandle imageView;
};

static std::vector<char> readFile(const std::string &filename) {
  // open file at the end to get the size, and in binary mode
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  // get file size by reading the position of the cursor
  size_t fileSize = (size_t)file.tellg();
  std::vector<char> fileBuffer(fileSize);

  // go back to the beginning of the file and read all the bytes at once
  file.seekg(0);
  file.read(fileBuffer.data(), fileSize);
  file.close();

  return fileBuffer;
}

static uint32_t findMemoryTypeIndex(
    VkPhysicalDevice physicalDevice, uint32_t allowedTypes,
    VkMemoryPropertyFlags
        properties) { // get properties of physical device memory
  VkPhysicalDeviceMemoryProperties memoryProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

  for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
    // check if memory type is among allowed types and has required properties
    if ((allowedTypes & (1 << i)) &&
        (memoryProperties.memoryTypes[i].propertyFlags & properties) ==
            properties) { // desired properity bit flags are set
      return i;           // return index of memory type
    }
  }
  throw std::runtime_error("Failed to find suitable memory type!");
}

static void createBuffer(VmaAllocator allocator, VkDeviceSize bufferSize,
                         VkBufferUsageFlags bufferUsage,
                         VmaMemoryUsage memoryUsage,
                         VmaAllocationCreateFlags memoryFlags,
                         AllocatedBuffer *outBuffer) {
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = bufferSize;   // size of buffer in bytes
  bufferInfo.usage = bufferUsage; // type of buffer we want to create
  bufferInfo.sharingMode =
      VK_SHARING_MODE_EXCLUSIVE; // buffer is shared between multiple

  VmaAllocationCreateInfo allocInfo = {};
  allocInfo.usage = memoryUsage;
  allocInfo.flags = memoryFlags;

  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  VkResult result = vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer,
                                    &allocation, nullptr);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("failed to create VMA buffer!");
  }
  *outBuffer = AllocatedBuffer(allocator, buffer, allocation);
}

static VkCommandBuffer beginCommandBuffer(VkDevice device,
                                          VkCommandPool commandPool) {
  // command buffer to hold transder commands
  VkCommandBuffer commandBuffer;

  // command buffer allocation info
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = commandPool;
  allocInfo.commandBufferCount = 1;

  // allocate command buffer from the command pool
  vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

  // info to begin the command buffer recording
  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags =
      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // only using it once so
                                                   // optimize for that

  // begin recording commands to the command buffer
  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  return commandBuffer;
}

static void endAndSubmitCommandBuffer(VkDevice device,
                                      VkCommandPool commandPool, VkQueue queue,
                                      VkCommandBuffer commandBuffer) {

  vkEndCommandBuffer(commandBuffer);

  // submit info about the command buffer to submit it to the queue
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  // submit transfer command buffer to the transfer queue and wait for it to
  // finish
  vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue); // wait for the transfer to finish before
                          // cleaning up the command buffer

  // free the command buffer back to the command pool
  vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

static void copyBuffer(VkDevice device, VkQueue transferQueue,
                       VkCommandPool transferCommandPool, VkBuffer srcBuffer,
                       VkBuffer dstBuffer, VkDeviceSize bufferSize) {

  // create buffer
  VkCommandBuffer transferCommandBuffer =
      beginCommandBuffer(device, transferCommandPool);

  //  region of data to copy and the buffers to copy between
  VkBufferCopy bufferCopyRegion = {};
  bufferCopyRegion.srcOffset = 0; // optional
  bufferCopyRegion.dstOffset = 0; // optional
  bufferCopyRegion.size = bufferSize;

  // command to copy src buffer to dst buffer and the region of data to copy
  vkCmdCopyBuffer(transferCommandBuffer, srcBuffer, dstBuffer, 1,
                  &bufferCopyRegion);

  endAndSubmitCommandBuffer(device, transferCommandPool, transferQueue,
                            transferCommandBuffer);
}

static void copyImageBuffer(VkDevice device, VkQueue transferQueue,
                            VkCommandPool transferCommandPool,
                            VkBuffer srcBuffer, VkImage dstImage,
                            uint32_t width, uint32_t height,
                            uint32_t layerCount = 1) {
  // create command buffer
  VkCommandBuffer transferCommandBuffer =
      beginCommandBuffer(device, transferCommandPool);

  // region of data to copy and the buffers to copy between
  VkBufferImageCopy imageRegion = {};
  imageRegion.bufferOffset = 0;    // optional
  imageRegion.bufferRowLength = 0; // optional
  imageRegion.bufferImageHeight =
      0; // optional (if row length and image height are 0, texels are tightly
         // packed in buffer)
  imageRegion.imageSubresource.aspectMask =
      VK_IMAGE_ASPECT_COLOR_BIT;             // which aspect of image to copy
  imageRegion.imageSubresource.mipLevel = 0; // mip level to copy to
  imageRegion.imageSubresource.baseArrayLayer =
      0; // starting array layer to copy to
  imageRegion.imageSubresource.layerCount =
      layerCount;                      // number of array layers to copy
  imageRegion.imageOffset = {0, 0, 0}; // offset into image to start copying to
  imageRegion.imageExtent = {width, height, 1}; // size of region to copy

  // command to copy buffer to image and the region of data to copy
  vkCmdCopyBufferToImage(transferCommandBuffer, srcBuffer, dstImage,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageRegion);

  endAndSubmitCommandBuffer(device, transferCommandPool, transferQueue,
                            transferCommandBuffer);
}

static void transitionImageLayout(VkDevice device, VkQueue queue,
                                  VkCommandPool commandPool, VkImage image,
                                  VkImageLayout oldLayout,
                                  VkImageLayout newLayout,
                                  uint32_t layerCount = 1) {

  VkCommandBuffer commandBuffer = beginCommandBuffer(device, commandPool);

  VkImageMemoryBarrier imageMemoryBarrier = {};
  imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  imageMemoryBarrier.oldLayout = oldLayout; // layout to transition from
  imageMemoryBarrier.newLayout = newLayout; // layout to transition to
  imageMemoryBarrier.srcQueueFamilyIndex =
      VK_QUEUE_FAMILY_IGNORED; // queue family to transition from
  imageMemoryBarrier.dstQueueFamilyIndex =
      VK_QUEUE_FAMILY_IGNORED;      // queue family to transition to
  imageMemoryBarrier.image = image; // image to transition
  imageMemoryBarrier.subresourceRange.aspectMask =
      VK_IMAGE_ASPECT_COLOR_BIT; // aspect of image to transition
  imageMemoryBarrier.subresourceRange.baseMipLevel =
      0; // first mip level to transition
  imageMemoryBarrier.subresourceRange.levelCount =
      1; // number of mip levels to transition
  imageMemoryBarrier.subresourceRange.baseArrayLayer =
      0; // first array layer to transition
  imageMemoryBarrier.subresourceRange.layerCount =
      layerCount; // number of array layers to transition

  VkPipelineStageFlags
      srcStage; // pipeline stage to wait for before starting the transition
  VkPipelineStageFlags dstStage; // pipeline stage to start after the transition

  // if transitioning from new image to image ready to receive data
  if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
      newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    imageMemoryBarrier.srcAccessMask = 0; // memory access before transition
    imageMemoryBarrier.dstAccessMask =
        VK_ACCESS_TRANSFER_WRITE_BIT; // memory access after transition

    srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  }
  // if transitioning from transfer ready image to image ready to be read in
  // shader
  else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
           newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  }
  vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0, nullptr, 0,
                       nullptr, 1, &imageMemoryBarrier);

  endAndSubmitCommandBuffer(device, commandPool, queue, commandBuffer);
}
