#include "PerformanceMetrics.h"
#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

void PerformanceMetrics::init(VkDevice device, VkPhysicalDevice physicalDevice,
                              uint32_t graphicsQueueFamilyIndex) {
  frameTimeHistory.resize(HISTORY_SIZE, 0.0);

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(physicalDevice, &props);
  timestampPeriod = props.limits.timestampPeriod;

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

  VkQueryPoolCreateInfo queryPoolInfo = {};
  queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
  queryPoolInfo.queryCount = NUM_GPU_PASSES * 2; // begin + end per pass

  if (vkCreateQueryPool(device, &queryPoolInfo, nullptr, &timestampQueryPool) !=
      VK_SUCCESS) {
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

void PerformanceMetrics::beginFrame() {
  frameStart = Clock::now();
  currentDrawCalls = 0;
  currentTriangles = 0;
}

void PerformanceMetrics::resetGpuQueries(VkCommandBuffer cmd) {
  if (!gpuTimingAvailable)
    return;
  vkCmdResetQueryPool(cmd, timestampQueryPool, 0, NUM_GPU_PASSES * 2);
}

void PerformanceMetrics::beginPassTimestamp(VkCommandBuffer cmd, GpuPass pass) {
  if (!gpuTimingAvailable)
    return;
  uint32_t slot = static_cast<uint32_t>(pass) * 2;
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                      timestampQueryPool, slot);
}

void PerformanceMetrics::endPassTimestamp(VkCommandBuffer cmd, GpuPass pass) {
  if (!gpuTimingAvailable)
    return;
  uint32_t slot = static_cast<uint32_t>(pass) * 2 + 1;
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                      timestampQueryPool, slot);
}

void PerformanceMetrics::recordDrawCall(uint32_t indexCount) {
  currentDrawCalls++;
  currentTriangles += indexCount / 3;
}

void PerformanceMetrics::endFrame(VkDevice device) {
  auto frameEnd = Clock::now();
  double frameTimeMs =
      std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();

  rollingSum -= frameTimeHistory[historyIndex];
  frameTimeHistory[historyIndex] = frameTimeMs;
  rollingSum += frameTimeMs;
  historyIndex = (historyIndex + 1) % HISTORY_SIZE;
  if (!historyFull && historyIndex == 0)
    historyFull = true;

  if (frameTimeMs < minFrameTime)
    minFrameTime = frameTimeMs;
  if (frameTimeMs > maxFrameTime)
    maxFrameTime = frameTimeMs;
  totalFrames++;

  lastDrawCalls = currentDrawCalls;
  lastTriangles = currentTriangles;

  if (gpuTimingAvailable && timestampQueryPool != VK_NULL_HANDLE) {
    uint64_t ts[NUM_GPU_PASSES * 2] = {};
    VkResult r = vkGetQueryPoolResults(
        device, timestampQueryPool, 0, NUM_GPU_PASSES * 2, sizeof(ts), ts,
        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (r == VK_SUCCESS) {
      for (int p = 0; p < NUM_GPU_PASSES; p++) {
        uint64_t begin = ts[p * 2], end = ts[p * 2 + 1];
        if (end > begin)
          lastPassGpuMs[p] =
              static_cast<double>(end - begin) * timestampPeriod / 1e6;
        else
          lastPassGpuMs[p] = 0.0;
      }
    }
  }

  // Periodic one-line snapshot — useful when shutdown is killed by timeout
  // (so the full report in printReport() may never run).
  if (totalFrames > 0 && totalFrames % 120 == 0) {
    spdlog::info("[perf] frame={} avg={:.2f}ms fps={:.1f} draws={} tris={}k "
                 "shadow={:.2f} ptShadow={:.2f} gbuf={:.2f} deferred={:.2f}",
                 totalFrames, getAverageFrameTimeMs(), getAverageFps(),
                 lastDrawCalls, lastTriangles / 1000,
                 lastPassGpuMs[(int)GpuPass::Shadow],
                 lastPassGpuMs[(int)GpuPass::PointShadow],
                 lastPassGpuMs[(int)GpuPass::GBuffer],
                 lastPassGpuMs[(int)GpuPass::Deferred]);
  }
}

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
  return (avg > 0.0) ? 1000.0 / avg : 0.0;
}

uint32_t PerformanceMetrics::getLastDrawCallCount() const {
  return lastDrawCalls;
}
uint32_t PerformanceMetrics::getLastTriangleCount() const {
  return lastTriangles;
}

double PerformanceMetrics::getLastGpuTimeMs() const {
  double total = 0.0;
  for (int p = 0; p < NUM_GPU_PASSES; p++)
    total += lastPassGpuMs[p];
  return total;
}

double PerformanceMetrics::getPassGpuTimeMs(GpuPass pass) const {
  return lastPassGpuMs[static_cast<int>(pass)];
}

PerformanceMetrics::VramStats
PerformanceMetrics::queryVram(VmaAllocator allocator) {
  VramStats stats;
  VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
  vmaGetHeapBudgets(allocator, budgets);
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

void PerformanceMetrics::printReport(VmaAllocator allocator) const {
  spdlog::info("╔══════════════════════════════════════════════╗");
  spdlog::info("║        PERFORMANCE BENCHMARK REPORT         ║");
  spdlog::info("╠══════════════════════════════════════════════╣");
  spdlog::info("║  Total Frames:        {:>20}  ║", totalFrames);
  spdlog::info("║  Avg Frame Time:      {:>17.2f} ms  ║",
               getAverageFrameTimeMs());
  spdlog::info("║  Min Frame Time:      {:>17.2f} ms  ║", minFrameTime);
  spdlog::info("║  Max Frame Time:      {:>17.2f} ms  ║", maxFrameTime);
  spdlog::info("║  Avg FPS:             {:>17.1f}     ║", getAverageFps());
  spdlog::info("║  Draw Calls/Frame:    {:>20}  ║", lastDrawCalls);
  spdlog::info("║  Triangles/Frame:     {:>20}  ║", lastTriangles);
  spdlog::info("╠══════════════════════════════════════════════╣");
  if (gpuTimingAvailable) {
    spdlog::info("║  GPU Pass Timing (last frame):               ║");
    spdlog::info("║    Shadow:          {:>17.3f} ms  ║", lastPassGpuMs[0]);
    spdlog::info("║    Point Shadow:    {:>17.3f} ms  ║", lastPassGpuMs[1]);
    spdlog::info("║    G-Buffer:        {:>17.3f} ms  ║", lastPassGpuMs[2]);
    spdlog::info("║    Deferred PBR:    {:>17.3f} ms  ║", lastPassGpuMs[3]);
    spdlog::info("║    Total GPU:       {:>17.3f} ms  ║", getLastGpuTimeMs());
  } else {
    spdlog::info("║  GPU Timing:          {:>20}  ║", "N/A");
  }
  VramStats vram = queryVram(allocator);
  double usedMB = static_cast<double>(vram.usedBytes) / (1024.0 * 1024.0);
  double budgetMB = static_cast<double>(vram.budgetBytes) / (1024.0 * 1024.0);
  spdlog::info("╠══════════════════════════════════════════════╣");
  spdlog::info("║  VRAM Used:           {:>14.1f} MiB  ║", usedMB);
  spdlog::info("║  VRAM Budget:         {:>14.1f} MiB  ║", budgetMB);
  spdlog::info("╚══════════════════════════════════════════════╝");
}
