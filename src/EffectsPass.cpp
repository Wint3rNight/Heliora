#include "EffectsPass.h"

#include "AdditivePipeline.h"
#include "VulkanDevice.h"

#include <algorithm>
#include <array>
#include <bit>
#include <stdexcept>

namespace {

// Matches effect_strip.frag.
struct StripPush {
  glm::vec4 params; // time
};

} // namespace

void EffectsPass::create(const VulkanDevice &device, VkRenderPass litPass,
                         VkDescriptorSetLayout sceneLayout, VkPipelineCache cache) {
  VkDevice dev = device.getLogicalDevice();
  VkPushConstantRange push = {};
  push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  push.size = sizeof(StripPush);
  VkPipelineLayoutCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  ci.setLayoutCount = 1;
  ci.pSetLayouts = &sceneLayout;
  ci.pushConstantRangeCount = 1;
  ci.pPushConstantRanges = &push;
  if (vkCreatePipelineLayout(dev, &ci, nullptr, &stripLayout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create the effects pipeline layout");

  const VkVertexInputBindingDescription binding = {0, sizeof(EffectVertex),
                                                   VK_VERTEX_INPUT_RATE_VERTEX};
  const std::array<VkVertexInputAttributeDescription, 4> attributes = {{
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(EffectVertex, position)},
      {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(EffectVertex, core)},
      {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(EffectVertex, edge)},
      {3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(EffectVertex, uv)},
  }};
  VkPipelineVertexInputStateCreateInfo input = {};
  input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  input.vertexBindingDescriptionCount = 1;
  input.pVertexBindingDescriptions = &binding;
  input.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
  input.pVertexAttributeDescriptions = attributes.data();
  stripPipeline = makeAdditivePipeline(dev, cache, litPass, stripLayout,
                                       "../Shaders/effect_strip.vert.spv",
                                       "../Shaders/effect_strip.frag.spv", input);
}

void EffectsPass::upload(const VulkanDevice &device, uint32_t imageIndex,
                         const std::vector<EffectVertex> &strips) {
  if (imageIndex >= vertexBuffers.size()) {
    vertexBuffers.resize(imageIndex + 1);
    capacities.resize(imageIndex + 1, 0);
    vertexCounts.resize(imageIndex + 1, 0);
  }
  const uint32_t count = static_cast<uint32_t>(strips.size());
  vertexCounts[imageIndex] = count;
  if (count == 0)
    return;
  if (count > capacities[imageIndex]) {
    capacities[imageIndex] = std::bit_ceil(std::max(count, 2048u));
    createBuffer(device.getAllocator(), capacities[imageIndex] * sizeof(EffectVertex),
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                 &vertexBuffers[imageIndex]);
  }
  uploadToAllocation(device.getAllocator(), vertexBuffers[imageIndex].getAllocation(),
                     strips.data(), count * sizeof(EffectVertex));
}

void EffectsPass::record(VkCommandBuffer cmd, uint32_t imageIndex, VkDescriptorSet sceneSet,
                         const VkViewport &viewport, const VkRect2D &scissor, float time) const {
  if (imageIndex >= vertexCounts.size() || vertexCounts[imageIndex] == 0)
    return;
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, stripPipeline);
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  vkCmdSetScissor(cmd, 0, 1, &scissor);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, stripLayout, 0, 1, &sceneSet, 0,
                          nullptr);
  const StripPush push = {glm::vec4(time, 0.0f, 0.0f, 0.0f)};
  vkCmdPushConstants(cmd, stripLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                     sizeof(push), &push);
  const VkBuffer buffer = vertexBuffers[imageIndex].get();
  const VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(cmd, 0, 1, &buffer, &offset);
  vkCmdDraw(cmd, vertexCounts[imageIndex], 1, 0, 0);
}

void EffectsPass::cleanup(VkDevice device) {
  vertexBuffers.clear(); // AllocatedBuffer frees itself
  capacities.clear();
  vertexCounts.clear();
  if (stripPipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(device, stripPipeline, nullptr);
    stripPipeline = VK_NULL_HANDLE;
  }
  if (stripLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(device, stripLayout, nullptr);
    stripLayout = VK_NULL_HANDLE;
  }
}
