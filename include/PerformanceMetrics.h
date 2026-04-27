#pragma once

#include "vk_mem_alloc.h"
#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

class PerformanceMetrics {
public:
  PerformanceMetrics() = default;
  ~PerformanceMetrics() = default;

  PerformanceMetrics(const PerformanceMetrics &) = delete;
  PerformanceMetrics &operator=(const PerformanceMetrics &) = delete;

  // GPU pass identifiers for per-pass timing
  enum class GpuPass { Shadow = 0, PointShadow = 1, GBuffer = 2, Deferred = 3 };
  static constexpr int NUM_GPU_PASSES = 4;

  // Lifecycle
  void init(VkDevice device, VkPhysicalDevice physicalDevice,
            uint32_t graphicsQueueFamilyIndex);
  void cleanup(VkDevice device);

  // Per-frame API
  void beginFrame();
  void resetGpuQueries(VkCommandBuffer cmd);

  // Per-pass GPU timestamps — call begin/end around each render pass
  void beginPassTimestamp(VkCommandBuffer cmd, GpuPass pass);
  void endPassTimestamp(VkCommandBuffer cmd, GpuPass pass);

  void recordDrawCall(uint32_t indexCount);
  void endFrame(VkDevice device);

  // Queries
  double getAverageFrameTimeMs() const;
  double getMinFrameTimeMs() const;
  double getMaxFrameTimeMs() const;
  double getAverageFps() const;
  uint32_t getLastDrawCallCount() const;
  uint32_t getLastTriangleCount() const;
  double getLastGpuTimeMs() const; // total GPU time (sum of all passes)
  double getPassGpuTimeMs(GpuPass pass) const;

  struct VramStats {
    uint64_t usedBytes = 0, budgetBytes = 0;
  };
  static VramStats queryVram(VmaAllocator allocator);

  void printReport(VmaAllocator allocator) const;

  uint64_t getTotalFrames() const { return totalFrames; }

private:
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = Clock::time_point;

  TimePoint frameStart{};
  double rollingSum = 0.0;
  static constexpr size_t HISTORY_SIZE = 300;
  std::vector<double> frameTimeHistory;
  size_t historyIndex = 0;
  bool historyFull = false;

  double minFrameTime = std::numeric_limits<double>::max();
  double maxFrameTime = 0.0;
  uint64_t totalFrames = 0;

  uint32_t currentDrawCalls = 0, currentTriangles = 0;
  uint32_t lastDrawCalls = 0, lastTriangles = 0;

  // GPU timing — NUM_GPU_PASSES * 2 query slots (begin + end per pass)
  VkQueryPool timestampQueryPool = VK_NULL_HANDLE;
  float timestampPeriod = 0.0f;
  bool gpuTimingAvailable = false;
  double lastPassGpuMs[NUM_GPU_PASSES] = {};
};
