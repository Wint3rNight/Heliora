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
  enum class GpuPass {
    Shadow = 0,
    PointShadow = 1,
    GBuffer = 2,
    SSGI = 3,
    Lit = 4,
    Bloom = 5,
    AutoExposure = 6,
    Composite = 7,
    ImGui = 8
  };
  static constexpr int NUM_GPU_PASSES = 9;

  enum class CpuPhase {
    WaitFence = 0,
    Acquire = 1,
    ImageFence = 2,
    Update = 3,
    Record = 4,
    Upload = 5,
    Submit = 6,
    Present = 7
  };
  static constexpr int NUM_CPU_PHASES = 8;

  // Lifecycle
  void init(VkDevice device, VkPhysicalDevice physicalDevice,
            uint32_t graphicsQueueFamilyIndex, uint32_t queryFrameCount = 1);
  void cleanup(VkDevice device);

  // Per-frame API
  void beginFrame();
  void collectGpuResults(VkDevice device, uint32_t frameIndex);
  void setActiveGpuQueryFrame(uint32_t frameIndex);
  void resetGpuQueries(VkCommandBuffer cmd);
  void markGpuQueriesSubmitted(uint32_t frameIndex);

  // Per-pass GPU timestamps — call begin/end around each render pass
  void beginPassTimestamp(VkCommandBuffer cmd, GpuPass pass);
  void endPassTimestamp(VkCommandBuffer cmd, GpuPass pass);

  void recordDrawCall(uint32_t indexCount);
  void recordCpuPhase(CpuPhase phase, double milliseconds);
  void endFrame();

  // Queries
  double getAverageFrameTimeMs() const;
  double getLastFrameTimeMs() const { return lastFrameTimeMs; }
  double getMinFrameTimeMs() const;
  double getMaxFrameTimeMs() const;
  double getAverageFps() const;
  uint32_t getLastDrawCallCount() const;
  uint32_t getLastTriangleCount() const;
  double getLastGpuTimeMs() const; // total GPU time (sum of all passes)
  double getAverageGpuTimeMs() const;
  double getPassGpuTimeMs(GpuPass pass) const;
  double getAveragePassGpuTimeMs(GpuPass pass) const;
  double getCpuPhaseTimeMs(CpuPhase phase) const;
  double getAverageCpuPhaseTimeMs(CpuPhase phase) const;
  double getCpuActiveTimeMs() const;
  double getAverageCpuActiveTimeMs() const;
  double getCpuSyncWaitTimeMs() const;
  double getAverageCpuSyncWaitTimeMs() const;
  double getCpuPhaseTotalMs() const;
  double getAverageCpuPhaseTotalMs() const;

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
  double lastFrameTimeMs = 0.0;
  uint64_t totalFrames = 0;

  uint32_t currentDrawCalls = 0, currentTriangles = 0;
  uint32_t lastDrawCalls = 0, lastTriangles = 0;

  // GPU timing — each frame-in-flight owns NUM_GPU_PASSES * 2 query slots.
  // Results are read only after that frame slot's fence has completed, so the
  // performance overlay does not force a same-frame CPU/GPU sync.
  VkQueryPool timestampQueryPool = VK_NULL_HANDLE;
  float timestampPeriod = 0.0f;
  bool gpuTimingAvailable = false;
  uint32_t gpuQueryFrameCount = 1;
  uint32_t activeGpuQueryFrame = 0;
  std::vector<uint8_t> gpuQueryFrameValid;
  double lastPassGpuMs[NUM_GPU_PASSES] = {};
  double totalPassGpuMs[NUM_GPU_PASSES] = {};
  uint64_t gpuTimingFrames = 0;

  double lastCpuPhaseMs[NUM_CPU_PHASES] = {};
  double totalCpuPhaseMs[NUM_CPU_PHASES] = {};
  uint64_t cpuPhaseSamples[NUM_CPU_PHASES] = {};

  uint32_t gpuQueryBase(uint32_t frameIndex) const {
    return (frameIndex % gpuQueryFrameCount) * NUM_GPU_PASSES * 2;
  }
};
