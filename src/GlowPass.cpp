#include "GlowPass.h"

#include "AdditivePipeline.h"
#include "Model.h"
#include "Vertex.h"
#include "VulkanDevice.h"

#include <array>
#include <stdexcept>

namespace {

// Matches glow.vert and glow.frag.
struct GlowPush {
  glm::mat4 model;
  glm::mat4 normal;
  glm::vec4 colour; // rgb, strength
  glm::vec4 params; // time, how far the shell sits out along the normals, the floor
};

// How far the glow's shell sits out from the surface, m: in front of it, so
// it never fights the surface's own depth.
constexpr float kShellOffset = 0.012f;

} // namespace

void GlowPass::create(const VulkanDevice &device, VkRenderPass litPass, VkDescriptorSetLayout sceneLayout,
                      VkPipelineCache cache) {
  VkDevice dev = device.getLogicalDevice();
  VkPushConstantRange push = {};
  push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  push.size = sizeof(GlowPush);
  VkPipelineLayoutCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  ci.setLayoutCount = 1;
  ci.pSetLayouts = &sceneLayout;
  ci.pushConstantRangeCount = 1;
  ci.pPushConstantRanges = &push;
  if (vkCreatePipelineLayout(dev, &ci, nullptr, &layout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create the glow's pipeline layout");
  // The glow reads only a mesh's positions and normals.
  const VkVertexInputBindingDescription binding = {0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
  const std::array<VkVertexInputAttributeDescription, 2> attributes = {{
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)},
      {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
  }};
  VkPipelineVertexInputStateCreateInfo input = {};
  input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  input.vertexBindingDescriptionCount = 1;
  input.pVertexBindingDescriptions = &binding;
  input.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
  input.pVertexAttributeDescriptions = attributes.data();
  pipeline = makeAdditivePipeline(dev, cache, litPass, layout, "../Shaders/glow.vert.spv",
                                  "../Shaders/glow.frag.spv", input);
}

void GlowPass::record(VkCommandBuffer cmd, VkDescriptorSet sceneSet, const VkViewport &viewport,
                      const VkRect2D &scissor, const std::vector<GlowDraw> &glows, float time) const {
  if (glows.empty() || pipeline == VK_NULL_HANDLE)
    return;
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  vkCmdSetScissor(cmd, 0, 1, &scissor);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &sceneSet, 0, nullptr);
  for (const GlowDraw &g : glows) {
    if (!g.mesh || g.mesh->getIndexCount() == 0)
      continue;
    const GlowPush push = {g.model, g.normal, glm::vec4(g.colour, g.strength),
                           glm::vec4(time, kShellOffset, g.floor, 0.0f)};
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(push), &push);
    const VkBuffer vertices = g.mesh->getVertexBuffer();
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertices, &offset);
    vkCmdBindIndexBuffer(cmd, g.mesh->getIndexBuffer(0), 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, static_cast<uint32_t>(g.mesh->getIndexCount(0)), 1, 0, 0, 0);
  }
}

void GlowPass::cleanup(VkDevice device) {
  if (pipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(device, pipeline, nullptr);
  if (layout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(device, layout, nullptr);
  pipeline = VK_NULL_HANDLE;
  layout = VK_NULL_HANDLE;
}
