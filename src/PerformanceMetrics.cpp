#include "PerformanceMetrics.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

// Lifecycle

void PerformanceMetrics::init(VkDevice device, VkPhysicalDevice physicalDevice,
                              uint32_t graphicsQueueFamilyIndex) {
  frameTimeHistory.resize(HISTORY_SIZE, 0.0);

  // Query the timestamp period (nanoseconds per tick) from physical device
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(physicalDevice, &props);
  timestampPeriod = props.limits.timestampPeriod;

  // Check if timestamps are supported on the graphics queue
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                           nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                           queueFamilies.data());

  if (graphicsQueueFamilyIndex < queueFamilyCount &&
      queueFamilies[graphicsQueueFamilyIndex].timestampValidBits > 0) {
    gpuTimingAvailable = true;
  }

  if (!gpuTimingAvailable) {
    spdlog::warn("GPU timestamp queries not supported on this queue family");
    return;
  }

  // Create a query pool with 2 slots: begin and end timestamps
  VkQueryPoolCreateInfo queryPoolInfo = {};
  queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
  queryPoolInfo.queryCount = 2;

  if (vkCreateQueryPool(device, &queryPoolInfo, nullptr,
                        &timestampQueryPool) != VK_SUCCESS) {
    spdlog::warn("Failed to create timestamp query pool");
    gpuTimingAvailable = false;
  }
}

void PerformanceMetrics::cleanup(VkDevice device) {
  if (timestampQueryPool != VK_NULL_HANDLE) {
    vkDestroyQueryPool(device, timestampQueryPool, nullptr);
    timestampQueryPool = VK_NULL_HANDLE;
  }
}

// Per-frame API

void PerformanceMetrics::beginFrame() {
  frameStart = Clock::now();
  currentDrawCalls = 0;
  currentTriangles = 0;
}

void PerformanceMetrics::resetGpuQueries(VkCommandBuffer cmd) {
  if (!gpuTimingAvailable)
    return;
  vkCmdResetQueryPool(cmd, timestampQueryPool, 0, 2);
}

void PerformanceMetrics::beginGpuTimestamp(VkCommandBuffer cmd) {
  if (!gpuTimingAvailable)
    return;
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                      timestampQueryPool, 0);
}

void PerformanceMetrics::endGpuTimestamp(VkCommandBuffer cmd) {
  if (!gpuTimingAvailable)
    return;
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                      timestampQueryPool, 1);
}

void PerformanceMetrics::recordDrawCall(uint32_t indexCount) {
  currentDrawCalls++;
  currentTriangles += indexCount / 3;
}

void PerformanceMetrics::endFrame(VkDevice device) {
  // CPU frame time
  auto frameEnd = Clock::now();
  double frameTimeMs =
      std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();

  // Update rolling history
  rollingSum -= frameTimeHistory[historyIndex];
  frameTimeHistory[historyIndex] = frameTimeMs;
  rollingSum += frameTimeMs;
  historyIndex = (historyIndex + 1) % HISTORY_SIZE;
  if (!historyFull && historyIndex == 0)
    historyFull = true;

  // Update min/max
  if (frameTimeMs < minFrameTime)
    minFrameTime = frameTimeMs;
  if (frameTimeMs > maxFrameTime)
    maxFrameTime = frameTimeMs;

  totalFrames++;

  // Snapshot per-frame counters
  lastDrawCalls = currentDrawCalls;
  lastTriangles = currentTriangles;

  // Retrieve GPU timestamps
  if (gpuTimingAvailable && timestampQueryPool != VK_NULL_HANDLE) {
    uint64_t timestamps[2] = {0, 0};
    VkResult result = vkGetQueryPoolResults(
        device, timestampQueryPool, 0, 2, sizeof(timestamps), timestamps,
        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
    if (result == VK_SUCCESS && timestamps[1] > timestamps[0]) {
      double gpuNs =
          static_cast<double>(timestamps[1] - timestamps[0]) * timestampPeriod;
      lastGpuTimeMs = gpuNs / 1'000'000.0;
    }
  }
}

// Queries

double PerformanceMetrics::getAverageFrameTimeMs() const {
  size_t count = historyFull ? HISTORY_SIZE : historyIndex;
  if (count == 0)
    return 0.0;
  return rollingSum / static_cast<double>(count);
}

double PerformanceMetrics::getMinFrameTimeMs() const { return minFrameTime; }

double PerformanceMetrics::getMaxFrameTimeMs() const { return maxFrameTime; }

double PerformanceMetrics::getAverageFps() const {
  double avg = getAverageFrameTimeMs();
  if (avg <= 0.0)
    return 0.0;
  return 1000.0 / avg;
}

uint32_t PerformanceMetrics::getLastDrawCallCount() const {
  return lastDrawCalls;
}

uint32_t PerformanceMetrics::getLastTriangleCount() const {
  return lastTriangles;
}

double PerformanceMetrics::getLastGpuTimeMs() const { return lastGpuTimeMs; }

// VRAM

PerformanceMetrics::VramStats
PerformanceMetrics::queryVram(VmaAllocator allocator) {
  VramStats stats;
  VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
  vmaGetHeapBudgets(allocator, budgets);

  // Sum over all heaps
  const VkPhysicalDeviceMemoryProperties *memProps;
  vmaGetMemoryProperties(allocator, &memProps);
  for (uint32_t i = 0; i < memProps->memoryHeapCount; i++) {
    if (memProps->memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      stats.usedBytes += budgets[i].usage;
      stats.budgetBytes += budgets[i].budget;
    }
  }
  return stats;
}

// Report

void PerformanceMetrics::printReport(VmaAllocator allocator) const {
  spdlog::info("╔══════════════════════════════════════════════╗");
  spdlog::info("║        PERFORMANCE BENCHMARK REPORT         ║");
  spdlog::info("╠══════════════════════════════════════════════╣");
  spdlog::info("║  Total Frames:        {:>20}  ║", totalFrames);
  spdlog::info("║  Avg Frame Time:      {:>17.2f} ms  ║", getAverageFrameTimeMs());
  spdlog::info("║  Min Frame Time:      {:>17.2f} ms  ║", minFrameTime);
  spdlog::info("║  Max Frame Time:      {:>17.2f} ms  ║", maxFrameTime);
  spdlog::info("║  Avg FPS:             {:>17.1f}     ║", getAverageFps());
  spdlog::info("║  Draw Calls/Frame:    {:>20}  ║", lastDrawCalls);
  spdlog::info("║  Triangles/Frame:     {:>20}  ║", lastTriangles);

  if (gpuTimingAvailable) {
    spdlog::info("║  GPU Time (last):     {:>17.2f} ms  ║", lastGpuTimeMs);
  } else {
    spdlog::info("║  GPU Time:            {:>20}  ║", "N/A");
  }

  VramStats vram = queryVram(allocator);
  double usedMB = static_cast<double>(vram.usedBytes) / (1024.0 * 1024.0);
  double budgetMB = static_cast<double>(vram.budgetBytes) / (1024.0 * 1024.0);
  spdlog::info("║  VRAM Used:           {:>14.1f} MiB  ║", usedMB);
  spdlog::info("║  VRAM Budget:         {:>14.1f} MiB  ║", budgetMB);
  spdlog::info("╚══════════════════════════════════════════════╝");
}
