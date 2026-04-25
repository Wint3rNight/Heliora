#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"

// Lightweight performance tracker that collects frame timing, draw call counts,
// triangle counts, VRAM usage, and GPU timing per frame.  Designed to impose
// near-zero overhead so it can ship in every build.

class PerformanceMetrics {
public:
  PerformanceMetrics() = default;
  ~PerformanceMetrics() = default;

  // Non-copyable
  PerformanceMetrics(const PerformanceMetrics &) = delete;
  PerformanceMetrics &operator=(const PerformanceMetrics &) = delete;

  //Lifecycle
  // Call once after VkDevice + VkPhysicalDevice are ready.
  void init(VkDevice device, VkPhysicalDevice physicalDevice,
            uint32_t graphicsQueueFamilyIndex);
  // Call once during shutdown before destroying the device.
  void cleanup(VkDevice device);

  // Per-frame API
  // Call at the very start of a frame (before acquire).
  void beginFrame();

  // Call inside command buffer recording, BEFORE vkCmdBeginRenderPass.
  void resetGpuQueries(VkCommandBuffer cmd);
  // Call inside command buffer recording, AFTER vkCmdBeginRenderPass.
  void beginGpuTimestamp(VkCommandBuffer cmd);
  // Call inside command buffer recording, AFTER the last draw call.
  void endGpuTimestamp(VkCommandBuffer cmd);

  // Call every time you issue a vkCmdDrawIndexed.
  void recordDrawCall(uint32_t indexCount);

  // Call at the very end of a frame (after present).
  // gpuQueueIdle should be true when the GPU work for this frame has completed
  // (i.e. after vkQueueWaitIdle or fence wait).
  void endFrame(VkDevice device);

  // Queries
  double getAverageFrameTimeMs() const;
  double getMinFrameTimeMs() const;
  double getMaxFrameTimeMs() const;
  double getAverageFps() const;
  uint32_t getLastDrawCallCount() const;
  uint32_t getLastTriangleCount() const;
  double getLastGpuTimeMs() const;

  // VRAM query — call whenever you need current stats.
  struct VramStats {
    uint64_t usedBytes = 0;
    uint64_t budgetBytes = 0;
  };
  static VramStats queryVram(VmaAllocator allocator);

  // Print a full benchmark summary to spdlog.
  void printReport(VmaAllocator allocator) const;

  uint64_t getTotalFrames() const { return totalFrames; }

private:
  // CPU timing
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = Clock::time_point;

  TimePoint frameStart{};
  double rollingSum = 0.0; // sum of last N frame times (ms)
  static constexpr size_t HISTORY_SIZE = 300; // ~5 seconds at 60fps
  std::vector<double> frameTimeHistory;
  size_t historyIndex = 0;
  bool historyFull = false;

  double minFrameTime = std::numeric_limits<double>::max();
  double maxFrameTime = 0.0;
  uint64_t totalFrames = 0;

  // Per-frame counters
  uint32_t currentDrawCalls = 0;
  uint32_t currentTriangles = 0;
  uint32_t lastDrawCalls = 0;
  uint32_t lastTriangles = 0;

  // GPU timing
  VkQueryPool timestampQueryPool = VK_NULL_HANDLE;
  float timestampPeriod = 0.0f; // nanoseconds per tick
  double lastGpuTimeMs = 0.0;
  bool gpuTimingAvailable = false;
};
