#include "PerformanceMetrics.h"
#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

void PerformanceMetrics::init(VkDevice device, VkPhysicalDevice physicalDevice,
                              uint32_t graphicsQueueFamilyIndex,
                              uint32_t queryFrameCount) {
  frameTimeHistory.resize(HISTORY_SIZE, 0.0);
  gpuQueryFrameCount = std::max(1u, queryFrameCount);
  activeGpuQueryFrame = 0;
  gpuQueryFrameValid.assign(gpuQueryFrameCount, 0);

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
  queryPoolInfo.queryCount =
      gpuQueryFrameCount * NUM_GPU_PASSES * 2; // begin + end per pass/frame

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

void PerformanceMetrics::collectGpuResults(VkDevice device,
                                           uint32_t frameIndex) {
  if (!gpuTimingAvailable || timestampQueryPool == VK_NULL_HANDLE ||
      gpuQueryFrameValid.empty())
    return;

  uint32_t queryFrame = frameIndex % gpuQueryFrameCount;
  if (!gpuQueryFrameValid[queryFrame])
    return;

  uint64_t ts[NUM_GPU_PASSES * 2] = {};
  VkResult r = vkGetQueryPoolResults(
      device, timestampQueryPool, gpuQueryBase(queryFrame),
      NUM_GPU_PASSES * 2, sizeof(ts), ts, sizeof(uint64_t),
      VK_QUERY_RESULT_64_BIT);
  if (r != VK_SUCCESS)
    return;

  for (int p = 0; p < NUM_GPU_PASSES; p++) {
    uint64_t begin = ts[p * 2], end = ts[p * 2 + 1];
    if (end > begin)
      lastPassGpuMs[p] =
          static_cast<double>(end - begin) * timestampPeriod / 1e6;
    else
      lastPassGpuMs[p] = 0.0;
    totalPassGpuMs[p] += lastPassGpuMs[p];
  }
  gpuTimingFrames++;
}

void PerformanceMetrics::setActiveGpuQueryFrame(uint32_t frameIndex) {
  activeGpuQueryFrame = frameIndex % gpuQueryFrameCount;
}

void PerformanceMetrics::resetGpuQueries(VkCommandBuffer cmd) {
  if (!gpuTimingAvailable)
    return;
  vkCmdResetQueryPool(cmd, timestampQueryPool,
                      gpuQueryBase(activeGpuQueryFrame), NUM_GPU_PASSES * 2);
  if (!gpuQueryFrameValid.empty())
    gpuQueryFrameValid[activeGpuQueryFrame] = 0;
}

void PerformanceMetrics::beginPassTimestamp(VkCommandBuffer cmd, GpuPass pass) {
  if (!gpuTimingAvailable)
    return;
  uint32_t slot =
      gpuQueryBase(activeGpuQueryFrame) + static_cast<uint32_t>(pass) * 2;
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                      timestampQueryPool, slot);
}

void PerformanceMetrics::endPassTimestamp(VkCommandBuffer cmd, GpuPass pass) {
  if (!gpuTimingAvailable)
    return;
  uint32_t slot =
      gpuQueryBase(activeGpuQueryFrame) + static_cast<uint32_t>(pass) * 2 + 1;
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                      timestampQueryPool, slot);
}

void PerformanceMetrics::markGpuQueriesSubmitted(uint32_t frameIndex) {
  if (!gpuTimingAvailable || gpuQueryFrameValid.empty())
    return;
  gpuQueryFrameValid[frameIndex % gpuQueryFrameCount] = 1;
}

void PerformanceMetrics::recordDrawCall(uint32_t indexCount) {
  currentDrawCalls++;
  currentTriangles += indexCount / 3;
}

void PerformanceMetrics::recordCpuPhase(CpuPhase phase, double milliseconds) {
  int idx = static_cast<int>(phase);
  if (idx < 0 || idx >= NUM_CPU_PHASES)
    return;
  lastCpuPhaseMs[idx] = milliseconds;
  totalCpuPhaseMs[idx] += milliseconds;
  cpuPhaseSamples[idx]++;
}

void PerformanceMetrics::endFrame() {
  auto frameEnd = Clock::now();
  double frameTimeMs =
      std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
  lastFrameTimeMs = frameTimeMs;

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

  // Periodic one-line snapshot — useful when shutdown is killed by timeout
  // (so the full report in printReport() may never run).
  if (totalFrames > 0 && totalFrames % 120 == 0) {
    spdlog::info("[perf] frame={} avg={:.2f}ms fps={:.1f} draws={} tris={}k "
                 "shadow={:.2f} ptShadow={:.2f} gbuf={:.2f} ssgi={:.2f} "
                 "lit={:.2f} bloom={:.2f} autoExp={:.2f} comp={:.2f} "
                 "imgui={:.2f}",
                 totalFrames, getAverageFrameTimeMs(), getAverageFps(),
                 lastDrawCalls, lastTriangles / 1000,
                 lastPassGpuMs[(int)GpuPass::Shadow],
                 lastPassGpuMs[(int)GpuPass::PointShadow],
                 lastPassGpuMs[(int)GpuPass::GBuffer],
                 lastPassGpuMs[(int)GpuPass::SSGI],
                 lastPassGpuMs[(int)GpuPass::Lit],
                 lastPassGpuMs[(int)GpuPass::Bloom],
                 lastPassGpuMs[(int)GpuPass::AutoExposure],
                 lastPassGpuMs[(int)GpuPass::Composite],
                 lastPassGpuMs[(int)GpuPass::ImGui]);
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

double PerformanceMetrics::getAverageGpuTimeMs() const {
  if (gpuTimingFrames == 0)
    return 0.0;
  double total = 0.0;
  for (int p = 0; p < NUM_GPU_PASSES; p++)
    total += totalPassGpuMs[p];
  return total / static_cast<double>(gpuTimingFrames);
}

double PerformanceMetrics::getPassGpuTimeMs(GpuPass pass) const {
  return lastPassGpuMs[static_cast<int>(pass)];
}

double PerformanceMetrics::getAveragePassGpuTimeMs(GpuPass pass) const {
  if (gpuTimingFrames == 0)
    return 0.0;
  return totalPassGpuMs[static_cast<int>(pass)] /
         static_cast<double>(gpuTimingFrames);
}

double PerformanceMetrics::getCpuPhaseTimeMs(CpuPhase phase) const {
  return lastCpuPhaseMs[static_cast<int>(phase)];
}

double PerformanceMetrics::getAverageCpuPhaseTimeMs(CpuPhase phase) const {
  int idx = static_cast<int>(phase);
  if (cpuPhaseSamples[idx] == 0)
    return 0.0;
  return totalCpuPhaseMs[idx] / static_cast<double>(cpuPhaseSamples[idx]);
}

double PerformanceMetrics::getCpuActiveTimeMs() const {
  return getCpuPhaseTimeMs(CpuPhase::Update) +
         getCpuPhaseTimeMs(CpuPhase::Record) +
         getCpuPhaseTimeMs(CpuPhase::Upload) +
         getCpuPhaseTimeMs(CpuPhase::Submit);
}

double PerformanceMetrics::getAverageCpuActiveTimeMs() const {
  return getAverageCpuPhaseTimeMs(CpuPhase::Update) +
         getAverageCpuPhaseTimeMs(CpuPhase::Record) +
         getAverageCpuPhaseTimeMs(CpuPhase::Upload) +
         getAverageCpuPhaseTimeMs(CpuPhase::Submit);
}

double PerformanceMetrics::getCpuSyncWaitTimeMs() const {
  return getCpuPhaseTimeMs(CpuPhase::WaitFence) +
         getCpuPhaseTimeMs(CpuPhase::Acquire) +
         getCpuPhaseTimeMs(CpuPhase::ImageFence) +
         getCpuPhaseTimeMs(CpuPhase::Present);
}

double PerformanceMetrics::getAverageCpuSyncWaitTimeMs() const {
  return getAverageCpuPhaseTimeMs(CpuPhase::WaitFence) +
         getAverageCpuPhaseTimeMs(CpuPhase::Acquire) +
         getAverageCpuPhaseTimeMs(CpuPhase::ImageFence) +
         getAverageCpuPhaseTimeMs(CpuPhase::Present);
}

double PerformanceMetrics::getCpuPhaseTotalMs() const {
  double total = 0.0;
  for (int p = 0; p < NUM_CPU_PHASES; p++)
    total += lastCpuPhaseMs[p];
  return total;
}

double PerformanceMetrics::getAverageCpuPhaseTotalMs() const {
  double total = 0.0;
  for (int p = 0; p < NUM_CPU_PHASES; p++) {
    if (cpuPhaseSamples[p] > 0)
      total += totalCpuPhaseMs[p] / static_cast<double>(cpuPhaseSamples[p]);
  }
  return total;
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
    spdlog::info("║  GPU Pass Timing (last / avg):               ║");
    spdlog::info("║    Shadow:          {:>7.3f} / {:>7.3f} ms  ║",
                 lastPassGpuMs[(int)GpuPass::Shadow],
                 getAveragePassGpuTimeMs(GpuPass::Shadow));
    spdlog::info("║    Point Shadow:    {:>7.3f} / {:>7.3f} ms  ║",
                 lastPassGpuMs[(int)GpuPass::PointShadow],
                 getAveragePassGpuTimeMs(GpuPass::PointShadow));
    spdlog::info("║    G-Buffer:        {:>7.3f} / {:>7.3f} ms  ║",
                 lastPassGpuMs[(int)GpuPass::GBuffer],
                 getAveragePassGpuTimeMs(GpuPass::GBuffer));
    spdlog::info("║    SSGI:            {:>7.3f} / {:>7.3f} ms  ║",
                 lastPassGpuMs[(int)GpuPass::SSGI],
                 getAveragePassGpuTimeMs(GpuPass::SSGI));
    spdlog::info("║    Lit:             {:>7.3f} / {:>7.3f} ms  ║",
                 lastPassGpuMs[(int)GpuPass::Lit],
                 getAveragePassGpuTimeMs(GpuPass::Lit));
    spdlog::info("║    Bloom:           {:>7.3f} / {:>7.3f} ms  ║",
                 lastPassGpuMs[(int)GpuPass::Bloom],
                 getAveragePassGpuTimeMs(GpuPass::Bloom));
    spdlog::info("║    Auto Exposure:   {:>7.3f} / {:>7.3f} ms  ║",
                 lastPassGpuMs[(int)GpuPass::AutoExposure],
                 getAveragePassGpuTimeMs(GpuPass::AutoExposure));
    spdlog::info("║    Composite:       {:>7.3f} / {:>7.3f} ms  ║",
                 lastPassGpuMs[(int)GpuPass::Composite],
                 getAveragePassGpuTimeMs(GpuPass::Composite));
    spdlog::info("║    ImGui:           {:>7.3f} / {:>7.3f} ms  ║",
                 lastPassGpuMs[(int)GpuPass::ImGui],
                 getAveragePassGpuTimeMs(GpuPass::ImGui));
    spdlog::info("║    Total GPU:       {:>7.3f} / {:>7.3f} ms  ║",
                 getLastGpuTimeMs(), getAverageGpuTimeMs());
  } else {
    spdlog::info("║  GPU Timing:          {:>20}  ║", "N/A");
  }
  spdlog::info("╠══════════════════════════════════════════════╣");
  spdlog::info("║  CPU Phase Timing (last / avg):              ║");
  spdlog::info("║    Wait fence:      {:>7.3f} / {:>7.3f} ms  ║",
               getCpuPhaseTimeMs(CpuPhase::WaitFence),
               getAverageCpuPhaseTimeMs(CpuPhase::WaitFence));
  spdlog::info("║    Acquire image:   {:>7.3f} / {:>7.3f} ms  ║",
               getCpuPhaseTimeMs(CpuPhase::Acquire),
               getAverageCpuPhaseTimeMs(CpuPhase::Acquire));
  spdlog::info("║    Image fence:     {:>7.3f} / {:>7.3f} ms  ║",
               getCpuPhaseTimeMs(CpuPhase::ImageFence),
               getAverageCpuPhaseTimeMs(CpuPhase::ImageFence));
  spdlog::info("║    Update scene:    {:>7.3f} / {:>7.3f} ms  ║",
               getCpuPhaseTimeMs(CpuPhase::Update),
               getAverageCpuPhaseTimeMs(CpuPhase::Update));
  spdlog::info("║    Record cmds:     {:>7.3f} / {:>7.3f} ms  ║",
               getCpuPhaseTimeMs(CpuPhase::Record),
               getAverageCpuPhaseTimeMs(CpuPhase::Record));
  spdlog::info("║    Upload UBO:      {:>7.3f} / {:>7.3f} ms  ║",
               getCpuPhaseTimeMs(CpuPhase::Upload),
               getAverageCpuPhaseTimeMs(CpuPhase::Upload));
  spdlog::info("║    Submit:          {:>7.3f} / {:>7.3f} ms  ║",
               getCpuPhaseTimeMs(CpuPhase::Submit),
               getAverageCpuPhaseTimeMs(CpuPhase::Submit));
  spdlog::info("║    Present:         {:>7.3f} / {:>7.3f} ms  ║",
               getCpuPhaseTimeMs(CpuPhase::Present),
               getAverageCpuPhaseTimeMs(CpuPhase::Present));
  spdlog::info("║    CPU active:      {:>7.3f} / {:>7.3f} ms  ║",
               getCpuActiveTimeMs(), getAverageCpuActiveTimeMs());
  spdlog::info("║    Sync/WSI wait:   {:>7.3f} / {:>7.3f} ms  ║",
               getCpuSyncWaitTimeMs(), getAverageCpuSyncWaitTimeMs());
  spdlog::info("║    CPU measured:    {:>7.3f} / {:>7.3f} ms  ║",
               getCpuPhaseTotalMs(), getAverageCpuPhaseTotalMs());
  VramStats vram = queryVram(allocator);
  double usedMB = static_cast<double>(vram.usedBytes) / (1024.0 * 1024.0);
  double budgetMB = static_cast<double>(vram.budgetBytes) / (1024.0 * 1024.0);
  spdlog::info("╠══════════════════════════════════════════════╣");
  spdlog::info("║  VRAM Used:           {:>14.1f} MiB  ║", usedMB);
  spdlog::info("║  VRAM Budget:         {:>14.1f} MiB  ║", budgetMB);
  spdlog::info("╚══════════════════════════════════════════════╝");
}
