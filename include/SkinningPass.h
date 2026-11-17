#pragma once

#include "Skinning.h"
#include "Utilities.h"
#include "VulkanDevice.h"

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include <vector>

// Bends characters' meshes on the GPU once a frame (Shaders/skinning.comp),
// before the G-buffer and shadows, into the vertex buffers they already
// draw from: each copy's own and its place in the GPU-driven pass's shared
// buffer. So the passes need no skinned versions of their shaders, and a
// mesh is bent once instead of once per pass.
//
// A mesh is uploaded once and shared; each character drawn is a copy of it
// with skinning matrices of its own. One pipeline for all of them, and every
// copy's dispatch between one barrier before and one after.
class SkinningPass {
public:
  void create(VulkanDevice &device, VkPipelineCache pipelineCache);
  void cleanup();

  // A mesh to bend: its bind-pose vertices, their bones and weights, and how
  // many skinning matrices a frame brings. -1 if it has no vertices or its
  // weights don't match them.
  int addMesh(const std::vector<Vertex> &bindVertices,
              const std::vector<VertexSkin> &skin, size_t boneCount);
  // A copy of a mesh, bent by matrices of its own. Kept until cleanup(), so
  // the caller reuses copies rather than adding one per spawn.
  int addCopy(int mesh);
  bool hasCopy(int copy) const {
    return copy >= 0 && copy < static_cast<int>(copies.size());
  }
  size_t vertexCount(int copy) const;

  // This frame slot's skinning matrices for a copy. Call after the slot's
  // fence, since the frame two before may still be reading them until then.
  void updateMatrices(int copy, int frame, const std::vector<glm::mat4> &skinning);

  // Where a copy's bent vertices go: meshVertices, and arena from
  // arenaFirstVertex on unless that's negative (not in the shared buffer).
  struct Target {
    int copy = -1;
    VkBuffer meshVertices = VK_NULL_HANDLE;
    VkBuffer arena = VK_NULL_HANDLE;
    int32_t arenaFirstVertex = -1;
  };
  // Records every target's dispatch, with a barrier before (the last
  // frame's draws have read the vertices) and after (this frame's draws see
  // the new ones). Nothing at all for no targets.
  void record(VkCommandBuffer cmd, int frame, const std::vector<Target> &targets);

private:
  struct Mesh {
    AllocatedBuffer bind;
    AllocatedBuffer skin;
    uint32_t vertexCount = 0;
    size_t matrixCount = 0;
  };
  struct Copy {
    int mesh = -1;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sets;          // one per frame slot
    std::vector<AllocatedBuffer> matrixBuffers; // one per frame slot, mapped
  };

  VulkanDevice *device = nullptr;
  VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  std::vector<Mesh> meshes;
  std::vector<Copy> copies;
};
