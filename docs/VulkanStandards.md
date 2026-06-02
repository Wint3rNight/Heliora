# Vulkan Standards Notes

These are the local renderer rules for Phase 2.6. They are meant to keep future
work aligned with the project's current direction rather than to prescribe a
full engine rewrite.

## Resource Ownership

- GPU memory goes through VMA via `AllocatedBuffer` and `AllocatedImage`.
- Enable `VK_EXT_memory_budget` when the GPU supports it and create VMA with
  `VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT`; otherwise let VMA use estimated
  budgets so device selection does not reject otherwise valid GPUs.
- Large frame/render targets should use
  `VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT` with
  `VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE`.
- Static GPU geometry and GPU-written indirect/count buffers should be
  device-preferred. Upload static data through staging instead of keeping
  persistent static arenas host-visible.
- Host-visible allocations are for staging, per-frame upload buffers, and
  readback buffers.
- Pass-local images, views, samplers, descriptor pools, pipelines, and
  framebuffers are destroyed by the pass or subsystem that creates them.
- Frame-sized render targets are registered in `RenderResources` so layout,
  aspect, and queue-family state can move toward one frame-graph-lite path.
- Depth-only resources should prefer `VK_FORMAT_D32_SFLOAT`. Use a stencil
  format only when a pass actually reads or writes stencil.

## Synchronization

- New central image transitions should use Vulkan 1.3 synchronization2 through
  `RenderResources::transition`.
- Pass-local barriers should use the helpers in `VulkanSync.h`, which translate
  legacy stage/access names into `vkCmdPipelineBarrier2` records.
- Do not add app-level `vkCmdPipelineBarrier` callsites; keep any remaining
  legacy references contained to vendored code.
- Batch adjacent initialization barriers where the source/destination stage and
  access masks are identical.
- Dedicated compute queues cannot use graphics-only stage masks. Compute
  command buffers should use compute/transfer/all-command stages only.
- Keep read-after-write barriers explicit even when old and new layouts are the
  same; same-layout does not mean no dependency is needed.
- Bloom and HZB intentionally keep per-mip compute dependency barriers where the
  next dispatch samples the previous dispatch output. Remove only barriers that
  have no later consumer.

## Queue Submits

- The current high-level submit shape is G-buffer, async SSAO, shadow, and post.
- A RenderDoc capture on the RTX 3050 Laptop GPU verified the intended 4-submit
  shape: graphics G-buffer signals the async timeline, compute SSAO waits on
  that signal, shadow work is submitted on graphics while SSAO runs, and the
  post submit waits for SSAO before sampling it.
- Extra submits are only worth keeping if GPU captures show real overlap or
  latency benefit. Do not merge the shadow and post submits if that would make
  shadow work wait for async SSAO and destroy the overlap.
- CPU timing alone is not enough for queue-submit decisions; use GPU captures
  and per-pass timestamps.
- The performance overlay reports per-frame submit count so submit changes are
  visible during A/B runs.

## Descriptors

- Do not update descriptor sets that can still be referenced by pending command
  buffers unless the layout was created with the correct update-after-bind or
  update-unused-while-pending flags.
- Prefer per-frame descriptor sets for frame-varying resources.
- Bindless texture descriptors may use update-after-bind because the bindless
  layout is explicitly created for that policy.
- Non-bindless descriptor writes should stay in startup/recreate paths or in
  per-swap-image dirty updates after that image's in-flight fence has completed.

## Pipelines And Draw Order

- Keep using the shared `VulkanPipelineCache` for graphics and compute pipeline
  creation.
- Sort opaque G-buffer draw items by pipeline/material/cull mode where
  correctness allows it.
- Keep A/B switches for costly command-recording strategies until captures prove
  the faster path.
- RenderDoc confirms the threaded CPU G-buffer path records secondary command
  buffers in parallel. Keep the toggle until CPU timing proves it is faster than
  single-threaded primary recording for the target scene.
- Threaded CPU G-buffer recording is opt-in by default. The captured 37-draw
  Sponza view proved correctness, not a performance win.
- GPU-driven G-buffer should report its cull compute pipeline plus the two
  indirect graphics pipeline binds so overlay stats remain comparable with the
  CPU path.
- Pipeline bind reduction must be measured; do not trade major code complexity
  for unproven wins.
- Hot shader reload should recreate affected development pipelines only, while
  startup/prewarm owns known production pipelines.

## Performance Capture Protocol

Use this protocol before claiming a Phase 2.6 performance win:

1. Build Release.
2. Disable validation layers.
3. Use a fixed resolution, fixed camera, and fixed scene preset.
4. Capture before/after with RenderDoc or Nsight Graphics.
5. Compare GPU pass timings, queue overlap, barrier counts, pipeline binds, CPU
   command-record time, and CPU submit/present wait time.
6. Keep the change only if the capture explains the win or the change fixes a
   validation/correctness issue.
