#include "ParticlePass.h"

#include "AdditivePipeline.h"
#include "VulkanDebug.h"
#include "VulkanDevice.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace {

// Matches Shaders/particles.comp.
struct SimPush {
  float dt;
  float time;
  uint32_t spawnCount;
  uint32_t head;
  glm::vec4 gravity;
  uint32_t capacity;
  uint32_t pad[3];
};

constexpr uint32_t kGroupSize = 256; // particles.comp's local_size_x
constexpr uint32_t kMostImages = 8;  // descriptor sets the pool holds

void barrier(VkCommandBuffer cmd, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
             VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
  VkMemoryBarrier2 b{};
  b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
  b.srcStageMask = srcStage;
  b.srcAccessMask = srcAccess;
  b.dstStageMask = dstStage;
  b.dstAccessMask = dstAccess;
  VkDependencyInfo dependency{};
  dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
  dependency.memoryBarrierCount = 1;
  dependency.pMemoryBarriers = &b;
  vkCmdPipelineBarrier2(cmd, &dependency);
}

} // namespace

void ParticlePass::create(const VulkanDevice &newDevice, VkRenderPass litPass,
                          VkDescriptorSetLayout sceneLayout, VkPipelineCache cache) {
  device = &newDevice;
  const VkDevice dev = device->getLogicalDevice();

  // 0: the ring (moved by the compute shader, read by the sprites' vertex
  // shader); 1: the frame's new ones.
  std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
  bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                 VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT, nullptr};
  bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo setCI{};
  setCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  setCI.bindingCount = static_cast<uint32_t>(bindings.size());
  setCI.pBindings = bindings.data();
  if (vkCreateDescriptorSetLayout(dev, &setCI, nullptr, &setLayout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create the particle set layout");

  VkDescriptorPoolSize size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * kMostImages};
  VkDescriptorPoolCreateInfo poolCI{};
  poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolCI.maxSets = kMostImages;
  poolCI.poolSizeCount = 1;
  poolCI.pPoolSizes = &size;
  if (vkCreateDescriptorPool(dev, &poolCI, nullptr, &pool) != VK_SUCCESS)
    throw std::runtime_error("Failed to create the particle descriptor pool");

  createBuffer(device->getAllocator(), kCapacity * sizeof(ParticleSpawn),
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VMA_MEMORY_USAGE_AUTO, RENDER_DEVICE_ALLOCATION_FLAGS, &particles);

  VkPushConstantRange simPush{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimPush)};
  VkPipelineLayoutCreateInfo simLayoutCI{};
  simLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  simLayoutCI.setLayoutCount = 1;
  simLayoutCI.pSetLayouts = &setLayout;
  simLayoutCI.pushConstantRangeCount = 1;
  simLayoutCI.pPushConstantRanges = &simPush;
  if (vkCreatePipelineLayout(dev, &simLayoutCI, nullptr, &simLayout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create the particle simulation layout");
  const VkShaderModule comp = loadShaderModule(dev, "../Shaders/particles.comp.spv");
  VkComputePipelineCreateInfo simCI{};
  simCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  simCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  simCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  simCI.stage.module = comp;
  simCI.stage.pName = "main";
  simCI.layout = simLayout;
  const VkResult result = vkCreateComputePipelines(dev, cache, 1, &simCI, nullptr, &simPipeline);
  vkDestroyShaderModule(dev, comp, nullptr);
  if (result != VK_SUCCESS)
    throw std::runtime_error("Failed to create the particle simulation pipeline");

  // The sprites: set 0 the camera, set 1 the ring; no vertex buffer, the
  // vertex shader builds each sprite's six corners from its index.
  const std::array<VkDescriptorSetLayout, 2> drawSets = {sceneLayout, setLayout};
  VkPipelineLayoutCreateInfo drawLayoutCI{};
  drawLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  drawLayoutCI.setLayoutCount = static_cast<uint32_t>(drawSets.size());
  drawLayoutCI.pSetLayouts = drawSets.data();
  if (vkCreatePipelineLayout(dev, &drawLayoutCI, nullptr, &drawLayout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create the particle draw layout");
  VkPipelineVertexInputStateCreateInfo noVertices{};
  noVertices.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  drawPipeline = makeAdditivePipeline(dev, cache, litPass, drawLayout, "../Shaders/particles.vert.spv",
                                      "../Shaders/particles.frag.spv", noVertices);
}

void ParticlePass::ensureImage(uint32_t imageIndex) {
  while (sets.size() <= imageIndex) {
    if (sets.size() >= kMostImages)
      throw std::runtime_error("More swap images than the particle pass has sets for");
    AllocatedBuffer spawns;
    createBuffer(device->getAllocator(), kMostSpawns * sizeof(ParticleSpawn),
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                 RENDER_UPLOAD_ALLOCATION_FLAGS, &spawns); // mapped for good
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &setLayout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(device->getLogicalDevice(), &allocInfo, &set) != VK_SUCCESS)
      throw std::runtime_error("Failed to allocate a particle descriptor set");
    const std::array<VkDescriptorBufferInfo, 2> infos = {{
        {particles.get(), 0, VK_WHOLE_SIZE},
        {spawns.get(), 0, VK_WHOLE_SIZE},
    }};
    std::array<VkWriteDescriptorSet, 2> writes{};
    for (uint32_t i = 0; i < writes.size(); ++i) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = set;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device->getLogicalDevice(), static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
    sets.push_back(set);
    spawnBuffers.push_back(std::move(spawns));
  }
}

void ParticlePass::simulate(VkCommandBuffer cmd, uint32_t imageIndex,
                            const std::vector<ParticleSpawn> &spawns, float dt, float time,
                            glm::vec3 gravity) {
  const uint32_t count = std::min(static_cast<uint32_t>(spawns.size()), kMostSpawns);
  aliveFor = std::max(0.0f, aliveFor - dt);
  for (uint32_t i = 0; i < count; ++i)
    aliveFor = std::max(aliveFor, spawns[i].life);
  if (aliveFor <= 0.0f && count == 0)
    return; // nothing alive, nothing new
  ensureImage(imageIndex);
  if (count > 0)
    uploadToAllocation(device->getAllocator(), spawnBuffers[imageIndex].getAllocation(),
                       spawns.data(), count * sizeof(ParticleSpawn));

  vkdbgBeginLabel(cmd, "Particles", 0.6f, 0.8f, 1.0f);
  // The frames before, on this queue, moved the ring (their compute) and drew
  // from it (their vertex shader): their writes reach this frame's compute,
  // and their reads are done before it, or a clear, writes over them.
  barrier(cmd, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT,
          VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
              VK_ACCESS_2_TRANSFER_WRITE_BIT);
  if (!cleared) {
    // Every slot dead (life 0) before the dispatch reads them.
    vkCmdFillBuffer(cmd, particles.get(), 0, VK_WHOLE_SIZE, 0);
    barrier(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    cleared = true;
  }
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, simPipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, simLayout, 0, 1, &sets[imageIndex], 0,
                          nullptr);
  const SimPush push = {dt, time, count, head, glm::vec4(gravity, 0.0f), kCapacity, {0, 0, 0}};
  vkCmdPushConstants(cmd, simLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
  vkCmdDispatch(cmd, (kCapacity + kGroupSize - 1) / kGroupSize, 1, 1);
  // This frame's sprites read what it wrote.
  barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
          VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
  vkdbgEndLabel(cmd);
  head = (head + count) % kCapacity;
}

void ParticlePass::reset() {
  aliveFor = 0.0f;
  head = 0;
  cleared = false;
}

void ParticlePass::draw(VkCommandBuffer cmd, uint32_t imageIndex, VkDescriptorSet sceneSet,
                        const VkViewport &viewport, const VkRect2D &scissor) const {
  if (aliveFor <= 0.0f || imageIndex >= sets.size())
    return;
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, drawPipeline);
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  vkCmdSetScissor(cmd, 0, 1, &scissor);
  const std::array<VkDescriptorSet, 2> bound = {sceneSet, sets[imageIndex]};
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, drawLayout, 0,
                          static_cast<uint32_t>(bound.size()), bound.data(), 0, nullptr);
  // Six corners a sprite, one sprite a slot; dead ones fall outside the view.
  vkCmdDraw(cmd, 6, kCapacity, 0, 0);
}

void ParticlePass::cleanup(VkDevice dev) {
  spawnBuffers.clear();
  particles = AllocatedBuffer();
  sets.clear();
  if (simPipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(dev, simPipeline, nullptr);
  if (drawPipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(dev, drawPipeline, nullptr);
  if (simLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(dev, simLayout, nullptr);
  if (drawLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(dev, drawLayout, nullptr);
  if (pool != VK_NULL_HANDLE)
    vkDestroyDescriptorPool(dev, pool, nullptr);
  if (setLayout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(dev, setLayout, nullptr);
  simPipeline = drawPipeline = VK_NULL_HANDLE;
  simLayout = drawLayout = VK_NULL_HANDLE;
  pool = VK_NULL_HANDLE;
  setLayout = VK_NULL_HANDLE;
}
