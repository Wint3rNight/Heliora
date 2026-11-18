#pragma once

#include "DescriptorManager.h"
#include "Model.h"
#include "ModelManager.h"
#include "Utilities.h"
#include "VulkanDevice.h"

#include <cstdint>
#include <glm/glm.hpp>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

enum class GBufferDrawKind {
  Mesh,
  Instanced,
};

// GPU-visible records consumed by gpu_cull.comp (std430 SSBO layouts).
struct GpuDrivenMeshRecord {
  glm::vec4 aabbMin;
  glm::vec4 aabbMax;
  glm::uvec4 draw;
  glm::uvec4 flags;
};

struct GpuDrivenTransformRecord {
  glm::mat4 model;
  glm::mat4 normal;
  glm::uvec4 texIdx0;
  glm::uvec4 texIdx1;
};

struct GBufferDrawItem {
  GBufferDrawKind kind = GBufferDrawKind::Mesh;
  const Mesh *mesh = nullptr;
  int lod = 0;
  ModelPushConstants push{};
  glm::vec3 aabbMin = glm::vec3(0.0f);
  glm::vec3 aabbMax = glm::vec3(0.0f);
  VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
  VkBuffer instanceBuffer = VK_NULL_HANDLE;
  const std::vector<InstanceData> *instances = nullptr;
  uint32_t instanceCount = 1;
  uint32_t indexCount = 0;
  uint32_t materialId = 0;
  uint32_t transformId = 0;
};

class GpuDrivenGBufferPass {
public:
  void create(VulkanDevice &device, ModelManager &modelManager,
              const DescriptorManager &descriptorManager,
              VkRenderPass gBufferRenderPass, VkExtent2D extent,
              const std::vector<ImageViewHandle> &gBufferDepthViews,
              VkPipelineCache pipelineCache);
  // Frees everything sized to the scene; the geometry arena stays, so a
  // rebuild doesn't upload every model again.
  void cleanup();
  // The arena, at shutdown.
  void releaseGeometry();

  // Adds a model's meshes to the arena and uploads only them.
  void registerModelGeometry(int modelId, ModelManager &modelManager);
  // Many at once, with one upload: pieces cut in the same frame.
  void registerModelGeometries(const std::vector<int> &modelIds, ModelManager &modelManager);
  // A registered model whose vertices were rewritten in place (the same
  // count): its range of the arena too, in the open upload batch. Call it
  // after any registering in the same batch, since growing the arena
  // replaces its buffer.
  void updateModelVertices(int modelId, ModelManager &modelManager);
  // Room for this many more vertices and indices than the arena holds now,
  // so what's added later doesn't grow it (which waits for the GPU). Waits
  // itself if it has to grow now; call it with no upload batch open.
  void reserveGeometry(size_t moreVertices, size_t moreIndices);
  // Scene switches must call this: the depth pyramid still holds the OLD
  // scene's depth, and occlusion-testing the new scene's AABBs against it
  // wrongly culls visible meshes for the first frame.
  void invalidateHzb() { hzbValid = false; }
  // A frame on the GPU-driven path: beginFrame with the number of opaque
  // meshes in the scene, then if beginRecords says the pass can take them,
  // addMesh and addInstanced for every one (no culling or order needed, the
  // compute shader culls), then uploadRecords. The add calls return false
  // for a mesh whose geometry was never registered; it isn't drawn.
  void beginFrame(uint32_t candidateCount);
  bool beginRecords(uint32_t imageIndex, bool enabled, uint32_t minCandidates);
  bool addMesh(const Mesh *mesh, int lod, const glm::mat4 &model,
               const glm::mat4 &normal, const glm::uvec4 &texIdx0,
               const glm::uvec4 &texIdx1, glm::vec3 aabbMin, glm::vec3 aabbMax,
               VkCullModeFlags cullMode);
  bool addInstanced(const Mesh *mesh, int lod,
                    const std::vector<InstanceData> &instances,
                    const glm::uvec4 &texIdx0, const glm::uvec4 &texIdx1,
                    glm::vec3 aabbMin, glm::vec3 aabbMax);
  // hzbDepthMargin: how much smaller (reversed-Z) an object's nearest depth
  // must be than the farthest occluder's before it counts as hidden.
  void uploadRecords(uint32_t imageIndex, const glm::vec4 frustumPlanes[6],
                     const glm::mat4 &viewProj, float hzbDepthMargin,
                     bool hzbEnabled);
  bool canRecordIndirect(uint32_t imageIndex, bool enabled,
                         uint32_t minCandidates) const;
  void recordIndirectGBuffer(VkCommandBuffer cmd, uint32_t imageIndex,
                             const VkRenderPassBeginInfo &renderPassInfo,
                             const VkViewport &viewport,
                             const VkRect2D &scissor, VkDescriptorSet vpSet,
                             VkDescriptorSet bindlessSet);
  void recordHzbBuild(VkCommandBuffer cmd, uint32_t imageIndex,
                      const std::vector<AllocatedImage> &gBufferDepthImages,
                      VkFormat gBufferDepthFormat, bool enabled,
                      bool hzbEnabled, uint32_t minCandidates);

  uint32_t candidateCount() const { return candidateCountValue; }
  uint32_t meshCount() const { return meshCountValue; }
  bool lastFrameUsed() const { return lastFrameUsedValue; }

private:
  struct FrameResources {
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

  struct GeometryRange {
    const Mesh *mesh = nullptr;
    int lod = 0;
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    int32_t vertexOffset = 0;
  };

  bool ensureBuffer(AllocatedBuffer &buffer, VkDeviceSize &capacity,
                    VkDeviceSize requiredSize, VkBufferUsageFlags usage,
                    VmaMemoryUsage memoryUsage = VMA_MEMORY_USAGE_AUTO,
                    VmaAllocationCreateFlags memoryFlags =
                        RENDER_UPLOAD_ALLOCATION_FLAGS);
  const GeometryRange *findGeometry(const Mesh *mesh, int lod) const;
  bool appendModelGeometry(int modelId, ModelManager &modelManager);
  bool growArenaBuffer(AllocatedBuffer &buffer, VkDeviceSize &capacity,
                       VkDeviceSize needed, VkBufferUsageFlags usage);
  void syncGeometry();
  void uploadMeshRecords(FrameResources &frame, const void *records,
                         VkDeviceSize bytes, uint32_t recordCount);
  void updateDescriptorSet(uint32_t imageIndex);
  bool hasStaticGeometry() const {
    return staticVertexBuffer && staticIndexBuffer;
  }

  VulkanDevice *device = nullptr;
  VkExtent2D renderExtent = {};

  std::vector<FrameResources> frames;
  AllocatedBuffer staticVertexBuffer;
  AllocatedBuffer staticIndexBuffer;
  VkDeviceSize staticVertexBufferSize = 0;
  VkDeviceSize staticIndexBufferSize = 0;
  // The arena's CPU copy, and how much of it is on the GPU already.
  std::vector<Vertex> staticVertices;
  std::vector<uint32_t> staticIndices;
  size_t uploadedVertexCount = 0;
  size_t uploadedIndexCount = 0;
  std::unordered_map<const Mesh *, int32_t> meshVertexOffsets;

public:
  // The shared vertex buffer, and where a mesh's vertices start in it
  // (counted in vertices), or -1 if it isn't in it: the skinning pass writes
  // a skinned mesh's vertices over its copy here every frame. The buffer
  // can be replaced when the arena grows, so ask each frame.
  VkBuffer arenaVertexBuffer() const { return staticVertexBuffer.get(); }
  int32_t arenaFirstVertex(const Mesh *mesh) const {
    const auto it = meshVertexOffsets.find(mesh);
    return it == meshVertexOffsets.end() ? -1 : it->second;
  }

private:
  std::vector<GeometryRange> geometryRanges;
  // (mesh pointer, lod) → index into geometryRanges. lod < 8 always, so it
  // packs into the pointer's low bits (allocations are ≥ 8-byte aligned).
  static uint64_t geometryKey(const Mesh *mesh, int lod) {
    return (reinterpret_cast<uint64_t>(mesh) << 3) |
           static_cast<uint64_t>(lod & 7);
  }
  std::unordered_map<uint64_t, size_t> geometryRangeLookup;
  // Persistent per-frame scratch (avoids re-allocating every frame).
  std::vector<GpuDrivenMeshRecord> frameMeshRecords;
  std::vector<GpuDrivenTransformRecord> frameTransforms;

  // One pyramid shared by every swap image (see create()).
  AllocatedImage hzbImage;
  ImageViewHandle hzbView;
  std::vector<ImageViewHandle> hzbMipViews;
  bool hzbValid = false;
  VkExtent2D hzbExtent = {};
  uint32_t hzbMipCount = 0;
  VkFormat hzbFormat = VK_FORMAT_UNDEFINED;
  VkSampler hzbSampler = VK_NULL_HANDLE;

  VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout hzbBuildSetLayout = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  VkDescriptorPool hzbBuildDescriptorPool = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> descriptorSets;
  std::vector<VkDescriptorSet> hzbBuildSets;
  VkPipelineLayout cullPipelineLayout = VK_NULL_HANDLE;
  VkPipeline cullPipeline = VK_NULL_HANDLE;
  VkPipelineLayout hzbBuildPipelineLayout = VK_NULL_HANDLE;
  VkPipeline hzbBuildPipeline = VK_NULL_HANDLE;
  VkPipelineLayout gBufferPipelineLayout = VK_NULL_HANDLE;
  VkPipeline gBufferPipeline = VK_NULL_HANDLE;
  VkPipeline gBufferNoCullPipeline = VK_NULL_HANDLE;

  uint32_t candidateCountValue = 0;
  uint32_t meshCountValue = 0;
  bool lastFrameUsedValue = false;
};
