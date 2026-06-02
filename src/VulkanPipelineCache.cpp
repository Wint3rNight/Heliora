#include "VulkanPipelineCache.h"

#include <fstream>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vector>

namespace {
constexpr const char *kPipelineCachePath = "pipeline_cache.bin";

std::vector<char> readPipelineCacheData() {
  std::ifstream file(kPipelineCachePath, std::ios::ate | std::ios::binary);
  if (!file.is_open())
    return {};

  std::vector<char> data(static_cast<size_t>(file.tellg()));
  file.seekg(0);
  file.read(data.data(), static_cast<std::streamsize>(data.size()));
  spdlog::info("Loaded pipeline cache ({} bytes)", data.size());
  return data;
}
} // namespace

void VulkanPipelineCache::init(VkDevice device) {
  if (cache != VK_NULL_HANDLE)
    return;

  std::vector<char> data = readPipelineCacheData();
  VkPipelineCacheCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  ci.initialDataSize = data.size();
  ci.pInitialData = data.empty() ? nullptr : data.data();

  VkResult result = vkCreatePipelineCache(device, &ci, nullptr, &cache);
  if (result != VK_SUCCESS && !data.empty()) {
    spdlog::warn("Ignoring incompatible pipeline cache data");
    ci.initialDataSize = 0;
    ci.pInitialData = nullptr;
    result = vkCreatePipelineCache(device, &ci, nullptr, &cache);
  }
  if (result != VK_SUCCESS)
    throw std::runtime_error("Failed to create pipeline cache");
}

void VulkanPipelineCache::cleanup(VkDevice device) {
  if (cache == VK_NULL_HANDLE)
    return;

  size_t size = 0;
  if (vkGetPipelineCacheData(device, cache, &size, nullptr) == VK_SUCCESS &&
      size > 0) {
    std::vector<char> data(size);
    if (vkGetPipelineCacheData(device, cache, &size, data.data()) ==
        VK_SUCCESS) {
      std::ofstream file(kPipelineCachePath, std::ios::binary);
      file.write(data.data(), static_cast<std::streamsize>(size));
      spdlog::info("Saved pipeline cache ({} bytes)", size);
    }
  }

  vkDestroyPipelineCache(device, cache, nullptr);
  cache = VK_NULL_HANDLE;
}
