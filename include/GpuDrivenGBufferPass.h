#pragma once

#include "DescriptorManager.h"
#include "Model.h"
#include "ModelManager.h"
#include "Utilities.h"
#include "VulkanDevice.h"

#include <glm/glm.hpp>
#include <vector>
#include <vulkan/vulkan.h>

enum class GBufferDrawKind {
  Mesh,
  Instanced,
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
  void cleanup();

  void registerModelGeometry(int modelId, ModelManager &modelManager);
  void beginFrame(uint32_t candidateCount);
  bool prepareFrame(uint32_t imageIndex,
                    const std::vector<GBufferDrawItem> &drawItems,
                    const glm::vec4 frustumPlanes[6],
                    const glm::mat4 &viewProj, bool enabled,
                    bool hzbEnabled, uint32_t minCandidates);
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
  void registerModelGeometryInternal(int modelId, ModelManager &modelManager);
  void uploadStaticGeometry();
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
  std::vector<Vertex> staticVertices;
  std::vector<uint32_t> staticIndices;
  std::vector<GeometryRange> geometryRanges;
  std::vector<int> registeredModelIds;

  std::vector<AllocatedImage> hzbImages;
  std::vector<ImageViewHandle> hzbViews;
  std::vector<std::vector<ImageViewHandle>> hzbMipViews;
  std::vector<bool> hzbValid;
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
