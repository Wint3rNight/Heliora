#pragma once

#include "Skinning.h"
#include "Utilities.h"
#include "VulkanDevice.h"

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include <vector>

// Bends the character's mesh on the GPU once a frame (Shaders/skinning.comp),
// before the G-buffer and shadows, into the vertex buffers they already
// draw from: the mesh's own and its copy in the GPU-driven pass's shared
// buffer. So the passes need no skinned versions of their shaders, and the
// mesh is bent once instead of once per pass.
class SkinningPass {
public:
  void create(VulkanDevice &device, VkPipelineCache pipelineCache);
  void cleanup();

  // The mesh to bend, uploaded once: its bind-pose vertices, their bones and
  // weights, and how many skinning matrices a frame brings.
  void setMesh(const std::vector<Vertex> &bindVertices,
               const std::vector<VertexSkin> &skin, size_t boneCount);
  bool hasMesh() const { return vertexCount > 0; }

  // This frame slot's skinning matrices. Call after the slot's fence, since
  // the frame two before may still be reading them until then.
  void updateMatrices(int frame, const std::vector<glm::mat4> &skinning);

  // Records the dispatch writing into meshVertices, and into arena from
  // arenaFirstVertex on unless that's negative (not in the shared buffer),
  // with a barrier before (the last frame's draws have read the vertices)
  // and after (this frame's draws see the new ones).
  void record(VkCommandBuffer cmd, int frame, VkBuffer meshVertices,
              VkBuffer arena, int32_t arenaFirstVertex);

private:
  VulkanDevice *device = nullptr;
  VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> sets; // one per frame slot

  AllocatedBuffer bindBuffer;
  AllocatedBuffer skinBuffer;
  std::vector<AllocatedBuffer> matrixBuffers; // one per frame slot, mapped
  uint32_t vertexCount = 0;
  size_t matrixCount = 0;
};
