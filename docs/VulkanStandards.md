# Vulkan Standards Notes

These are the local renderer rules for Phase 2.6. They are meant to keep future
work aligned with the project's current direction rather than to prescribe a
full engine rewrite.

## Resource Ownership

- GPU memory goes through VMA via `AllocatedBuffer` and `AllocatedImage`.
- Pass-local images, views, samplers, descriptor pools, pipelines, and
  framebuffers are destroyed by the pass or subsystem that creates them.
- Frame-sized render targets are registered in `RenderResources` so layout,
  aspect, and queue-family state can move toward one frame-graph-lite path.
- Depth-only resources should prefer `VK_FORMAT_D32_SFLOAT`. Use a stencil
  format only when a pass actually reads or writes stencil.

## Synchronization

- New central image transitions should use Vulkan 1.3 synchronization2 through
  `RenderResources::transition`.
- Existing manual `vkCmdPipelineBarrier` callsites are tolerated while passes
  are being migrated, but new broad barriers should not be added casually.
- Batch adjacent initialization barriers where the source/destination stage and
  access masks are identical.
- Dedicated compute queues cannot use graphics-only stage masks. Compute
  command buffers should use compute/transfer/all-command stages only.
- Keep read-after-write barriers explicit even when old and new layouts are the
  same; same-layout does not mean no dependency is needed.

## Queue Submits

- The current high-level submit shape is G-buffer, async SSAO, shadow, and post.
- Extra submits are only worth keeping if GPU captures show real overlap or
  latency benefit.
- CPU timing alone is not enough for queue-submit decisions; use GPU captures
  and per-pass timestamps.

## Descriptors

- Do not update descriptor sets that can still be referenced by pending command
  buffers unless the layout was created with the correct update-after-bind or
  update-unused-while-pending flags.
- Prefer per-frame descriptor sets for frame-varying resources.
- Bindless texture descriptors may use update-after-bind because the bindless
  layout is explicitly created for that policy.

## Pipelines And Draw Order

- Keep using the pipeline cache.
- Sort draw items by pipeline/material/cull mode where correctness allows it.
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
