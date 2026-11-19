#pragma once

#include <vulkan/vulkan.h>

// What the light effects and particles share: their shaders loaded, and the
// pipeline they draw with in the lit pass's transparent subpass.

// A SPIR-V file's shader module; throws if it can't be read.
VkShaderModule loadShaderModule(VkDevice device, const char *path);
// Triangles added on top (ONE, ONE), depth tested (reversed Z) but not
// written, both faces, viewport and scissor dynamic, into the lit pass's
// subpass 1, before bloom.
VkPipeline makeAdditivePipeline(VkDevice device, VkPipelineCache cache, VkRenderPass litPass,
                                VkPipelineLayout layout, const char *vertPath, const char *fragPath,
                                const VkPipelineVertexInputStateCreateInfo &vertexInput);
