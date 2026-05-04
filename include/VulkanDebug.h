#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Phase 7.1 — VK_EXT_debug_utils helpers
//
// Function pointers are loaded unconditionally at device-init time
// (vkGetDeviceProcAddr returns nullptr if the extension is absent, and every
// helper here null-checks before calling, so release builds without the layer
// are safe).
//
// Usage:
//   vkdbgBeginLabel(cmd, "Shadow Pass", 1.f, 0.3f, 0.3f);
//   ...record commands...
//   vkdbgEndLabel(cmd);
//
//   vkdbgName(device, shadowImage, VK_OBJECT_TYPE_IMAGE, "CSM Array");
// ---------------------------------------------------------------------------

// Loaded in VulkanDevice::loadDebugFunctions()
extern PFN_vkCmdBeginDebugUtilsLabelEXT  g_vkCmdBeginDebugUtilsLabel;
extern PFN_vkCmdEndDebugUtilsLabelEXT    g_vkCmdEndDebugUtilsLabel;
extern PFN_vkCmdInsertDebugUtilsLabelEXT g_vkCmdInsertDebugUtilsLabel;
extern PFN_vkSetDebugUtilsObjectNameEXT  g_vkSetDebugUtilsObjectName;

inline void vkdbgBeginLabel(VkCommandBuffer cmd, const char *name,
                             float r = 0.5f, float g = 0.5f, float b = 0.5f) {
    if (!g_vkCmdBeginDebugUtilsLabel) return;
    VkDebugUtilsLabelEXT info{};
    info.sType      = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    info.pLabelName = name;
    info.color[0]   = r;
    info.color[1]   = g;
    info.color[2]   = b;
    info.color[3]   = 1.0f;
    g_vkCmdBeginDebugUtilsLabel(cmd, &info);
}

inline void vkdbgEndLabel(VkCommandBuffer cmd) {
    if (g_vkCmdEndDebugUtilsLabel) g_vkCmdEndDebugUtilsLabel(cmd);
}

inline void vkdbgInsertLabel(VkCommandBuffer cmd, const char *name,
                              float r = 1.f, float g = 1.f, float b = 0.f) {
    if (!g_vkCmdInsertDebugUtilsLabel) return;
    VkDebugUtilsLabelEXT info{};
    info.sType      = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    info.pLabelName = name;
    info.color[0]   = r;
    info.color[1]   = g;
    info.color[2]   = b;
    info.color[3]   = 1.0f;
    g_vkCmdInsertDebugUtilsLabel(cmd, &info);
}

// Name any Vulkan object (image, buffer, pipeline, render pass, …)
inline void vkdbgName(VkDevice device, uint64_t handle,
                       VkObjectType type, const char *name) {
    if (!g_vkSetDebugUtilsObjectName) return;
    VkDebugUtilsObjectNameInfoEXT info{};
    info.sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.objectType   = type;
    info.objectHandle = handle;
    info.pObjectName  = name;
    g_vkSetDebugUtilsObjectName(device, &info);
}

// Convenience overloads for the most common handle types
inline void vkdbgNameImage   (VkDevice d, VkImage       h, const char *n) { vkdbgName(d, (uint64_t)h, VK_OBJECT_TYPE_IMAGE,        n); }
inline void vkdbgNameBuffer  (VkDevice d, VkBuffer      h, const char *n) { vkdbgName(d, (uint64_t)h, VK_OBJECT_TYPE_BUFFER,       n); }
inline void vkdbgNamePipeline(VkDevice d, VkPipeline    h, const char *n) { vkdbgName(d, (uint64_t)h, VK_OBJECT_TYPE_PIPELINE,     n); }
inline void vkdbgNameRP      (VkDevice d, VkRenderPass  h, const char *n) { vkdbgName(d, (uint64_t)h, VK_OBJECT_TYPE_RENDER_PASS,  n); }
inline void vkdbgNameFB      (VkDevice d, VkFramebuffer h, const char *n) { vkdbgName(d, (uint64_t)h, VK_OBJECT_TYPE_FRAMEBUFFER,  n); }
inline void vkdbgNameSampler (VkDevice d, VkSampler     h, const char *n) { vkdbgName(d, (uint64_t)h, VK_OBJECT_TYPE_SAMPLER,      n); }
