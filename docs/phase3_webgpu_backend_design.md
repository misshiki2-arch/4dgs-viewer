# Phase 3 WebGPU Backend Design

Date: 2026-06-09

This document is the Phase 3 design contract for moving the realtime 4DGS
viewer toward a normal WebGPU rendering backend. It is not a step history. It is
the contract that future Phase 3 implementation steps should preserve when they
change samples, display adapters, selectors, presentation, validation, or
summary fields.

## 1. Phase 3 Goal

Phase 3 moves realtime 4DGS rendering toward WebGPU as the normal rendering
backend.

- WebGPU normal backend owns new compute-shaped rendering work: visible records,
  tile-list generation, tile composite output, render-target handoff, and viewer
  canvas presentation.
- WebGL2 remains fallback, validation path, and regression oracle. It should not
  be mixed with WebGPU presentation in the same frame.
- Three.js and OrbitControls are camera input adapters. They provide user input,
  camera state, canvas sizing, and viewer shell behavior, but they are not the
  core 4DGS renderer.
- CUDA Reference remains the comparison and validation baseline for fixed
  reference captures. It is not an interactive viewer backend.

The important boundary is that WebGPU display experiments must advance toward a
real viewer backend without quietly relying on WebGL2 drawing, fallback samples,
or summary-only success.

## 2. Backend Responsibilities

### WebGPU Normal Backend

WebGPU owns the new normal rendering path. In the current Phase 3 state this
means storage-buffer fixed records, bounded tile composite outputs, true native
bounded color samples, constrained display adapter output, and guarded viewer
canvas presentation.

WebGPU success must be supported by explicit runtime signals:

- `status=ok` for the relevant stage.
- `anyMismatch=false` or equivalent comparison success where applicable.
- explicit sample counts and source lineage.
- no WebGPU validation errors, invalid bind groups, invalid command buffers, or
  queue submission failures.

### WebGL2 Fallback / Validation Oracle

WebGL2 stays useful as the stable fallback and validation oracle. It should keep
proving reference behavior, raw attribute interpretation, Transform Feedback
contracts, and regression summaries. It must not become a hidden participant in
WebGPU viewer canvas presentation.

### Common Contracts / Shared Data Structures

Common modules define stable vocabulary and shape contracts. Record and tile-list
contracts live in `common_4dgs_record_contracts.js` and
`common_4dgs_tile_list_contracts.js`. Bounded display samples use the Phase 3
sample contract in `common_4dgs_sample_contracts.js`.

New shared contracts should be small, explicit, and consumed by runtime code as
well as summaries. A contract name in JSON is not enough if selector or present
code still interprets a different shape.

### Selector / Present Adapter

The selector chooses the best bounded color source. The present adapter takes
only presentable samples and submits guarded WebGPU work to the viewer canvas.
These are separate responsibilities:

- selector decides source priority and fallback eligibility.
- present validates that selected samples are actually drawable.
- present must not add fallback samples when selector samples exist.

### Runtime Summary / Validation Tools

Runtime summaries and tools make the contract observable. `summarize_step_json.py`
must show source selection, sample counts, fallback suppression, presentability,
and failure boundaries. `check_step_files.py` may report missing PNG or
non-runtime files during runtime-only WebGPU dry-run captures; that condition is
not the success criterion for these captures.

## 3. Step37 -> Step40 -> Selector -> Present Flow

The current true native bounded color path is:

1. Step37 tile composite accumulation:
   Produces bounded accumulation samples from tile composite inputs. Step50 may
   seed native accumulation from render handoff sample records when earlier
   accumulation inputs are empty; that seed must be reported as seed lineage, not
   hidden as a final display source.
2. Step38 framebuffer-free tile output:
   Converts bounded accumulation samples into framebuffer-free tile output
   samples.
3. Step39 render target handoff:
   Converts Step38 output into bounded render-target-shaped samples.
4. Step40 constrained display adapter:
   Writes bounded render-target samples into an `rgba8unorm` storage texture,
   reads back sample pixels, and compares the result.
5. Color source selector:
   Chooses the highest-priority available bounded color source. True Step40
   constrained display samples are preferred before native bridge samples and
   render-handoff-derived fallback.
6. Viewer canvas bounded color present:
   Uses selector-selected samples under `webgpu-exclusive` ownership and
   `webgpuAllowViewerCanvasPresentation=true` to submit bounded color quads to
   the viewer canvas WebGPU current texture.

This flow is successful only when the selector and present adapter consume the
same sample contract and the present output contains only the selected source.

## 4. Sample Contract

Bounded color samples use this normal form once they cross selector or present
boundaries:

```text
source: stable source string
colorSource: human-readable color origin
recordIndex / srcIndex / sampleKind / valid: lineage fields when available
samplePx: { x, y } in viewer or texture pixel space
colorAlpha: { r, g, b, a } normalized to [0, 1]
upstreamSample: optional original sample for debugging/lineage
```

### Sample ID / Source / Lineage

Every sample must carry a stable `source` and enough lineage to identify where
it came from. `source` names the contract boundary, not merely the object that
last copied it. Examples:

- `webgpuConstrainedDisplayAdapterDryRunComparison.sampleTexturePixels`
- `webgpuRenderTargetHandoffDryRunComparison.sampleRenderTargetPixels`
- `webgpuFramebufferFreeTileOutputDryRunComparison.sampleTileOutputs`
- `webgpuRenderHandoffStub.sampleRecords`

Seed lineage may mention render handoff, but a selected Step40 sample remains a
Step40 sample if it was produced and validated through Step37-40.

### Color

The canonical color is `colorAlpha: { r, g, b, a }` in normalized float space.
Input code may accept arrays such as `rgbaFloat`, `resolvedColor`, or
`colorAlpha`, but selector and present must normalize those inputs into the
object form above.

`rgbaFloat` and `colorAlpha` are not competing meanings at the selector/present
boundary. They are accepted input shapes that become the same normal form.
Premultiplied display output is a present policy, not the sample contract.
Current bounded presents force display alpha opaque while preserving the sample
RGB source and reporting that SH/color parity remains deferred.

### Pixel Coordinates

`samplePx` is the present coordinate. Step40 may also expose `pixel`,
`texturePixel`, `originalPixel`, and `pixelRemappedForUniqueness`.

- `pixel` or `texturePixel`: pixel used for texture write/readback.
- `originalPixel`: pixel before duplicate remap.
- `samplePx`: normalized selector/present coordinate.
- `pixelRemappedForUniqueness`: true when Step40 moved a duplicate pixel before
  texture write.

Present code must clamp `samplePx` to the viewer canvas extent. It must not
reinterpret duplicate remap metadata as an instruction to merge samples.

### Duplicate Pixel Policy

Parallel texture writes to the same pixel are ambiguous. Step40 must prevent
that before dispatch by deterministic remap:

```text
duplicatePixelPolicy = deterministic-remap-before-parallel-texture-write
duplicatePixelCountBeforeRemap = count of collisions found
duplicatePixelRemapCount = count of samples remapped
duplicatePixelCount = unresolved duplicate count after remap
```

Step40 can be `ok` only when unresolved duplicate count is zero and texture
readback comparison succeeds.

### True Native / Bridge / Fallback

- True native samples are generated by the Step37 -> Step40 runtime path in the
  same capture.
- Native bridge samples are shaped from bounded native-compatible data when true
  stage samples are not present.
- Fallback samples are reference-assisted render handoff samples used only when
  selector-selected samples are unavailable.

Fallback success must never be reported as true native success.

## 5. Selector Contract

The selector reports:

- `sourcePriority`
- `sourceAvailability`
- `selectedSourceKind`
- `selectedColorSource`
- `selectedSampleCount`
- `selectedColorSamples`
- `fallbackPolicy`
- `nativeBridgePolicy`

Source priority is:

1. Step40 constrained display adapter true native samples.
2. Step40 native bridge samples.
3. Step39 true native samples.
4. Step39 native bridge samples.
5. Step38 true native samples.
6. Step38 native bridge samples.
7. render-handoff-derived fallback.

If `selectedColorSamples.length > 0`, present must use those samples and must
suppress fallback source scanning. Fallback is allowed only when no selector
selected samples exist. Bridge samples are allowed as intermediate fallback
inside the native bounded color sample contract, but they must be reported as
bridge and not as true native.

## 6. Viewer Canvas Present Contract

Present receives bounded color samples in the selector/present normal form. A
sample is presentable when it has:

- a normalized `colorAlpha` object or accepted color input shape,
- a finite pixel coordinate source that can become `samplePx`,
- a source and color source for summary lineage.

`colorPresentSampleCount` is the number of presentable samples actually encoded
into the viewer canvas render pass. It is not the raw selected sample count.
Runtime summaries should report both:

- `selectorSelectedRawSampleCount`
- `selectorPresentableSampleCount`

`commandBufferSubmitted=true` requires all guards and at least one presentable
sample:

- WebGPU device available.
- viewer canvas provided.
- `webgpu-exclusive` backend mode.
- `webgpuAllowViewerCanvasPresentation=true`.
- WebGL2 frame lifecycle suppressed.
- viewer canvas currentTexture path ready.
- bounded first-present already succeeded.
- presentable color sample count greater than zero.

Status meanings:

- `ok`: command buffer submitted, current texture acquired, no present error.
- `blocked`: guard, sample, context, or submit requirement was not satisfied.
- `mismatch`: comparison stage found data mismatch. Present currently blocks
  rather than reporting mismatch for missing samples.

When selector samples exist, `selectionMode=selector-selected-samples`,
`fallbackAllowed=false`, and
`fallbackSuppressedBySelectorSamples=true`. `presentedSamples` must contain only
the selected source in that mode.

## 7. Validation / Summary Contract

For runtime-only WebGPU dry-run captures, `check_step_files.py` may report
`ERROR_MISSING_REQUIRED_FILES` because PNG and full comparison files are not
saved. That is expected when the capture prefix intentionally includes only
runtime, camera, limited draw, and WebGPU dry-run JSON.

Success is judged primarily by the WebGPU dry-run JSON and
`summarize_step_json.py`.

Summary output must show at least:

- Step37/38/39/40 status.
- Step40 duplicate pixel remap counters and policy.
- selector `selectedSourceKind` and `selectedSampleCount`.
- present `selectionMode`.
- `selectorSelectedRawSampleCount`.
- `selectorPresentableSampleCount`.
- `fallbackAllowed`.
- `fallbackSuppressedBySelectorSamples`.
- `colorPresentSampleCount`.
- `sampleSources`.
- `presentedSamples`.
- `commandBufferSubmitted`.
- `viewerCanvasContextConfiguredForColorPresent`.

Any WebGPU validation error, `ReferenceError`, `TypeError`, invalid bind group,
invalid command buffer, or invalid queue submit means the step is not clean even
if JSON files were saved. Python cache files are not artifacts; tracked
`__pycache__` or `.pyc` files are forbidden.

## 8. Future Implementation Rules

- New sample shapes must update this design doc and
  `common_4dgs_sample_contracts.js` in the same step.
- Do not make summary fields compensate for runtime contract mismatches.
- Do not classify fallback output as true native output.
- Do not mix WebGPU presentation and WebGL2 rendering in the same frame.
- Do not change camera, projection, or control contracts while implementing
  bounded display plumbing.
- Keep SH/color parity explicit. A bounded present may use reference-assisted
  color, but it must report that SH evaluation remains deferred.
- Prefer responsibility-sized files when a new contract crosses selector,
  present, runtime, and summary boundaries.
- Related changes may land together when the summary separates failure causes.
- Production viewer connection, broad UI changes, and interactive camera
  implementation are separate steps.

## 9. Current Step50 Fix3 Baseline

The clean baseline before Step51 is:

- Step37, Step38, Step39, and Step40 are `status=ok`.
- Step40 uses deterministic duplicate-pixel remap and reports zero unresolved
  duplicate pixels.
- selector chooses `step40-constrained-display-adapter`.
- present uses `selectionMode=selector-selected-samples`.
- raw selected samples and presentable samples are both 2.
- `sampleSources` contains only
  `webgpuConstrainedDisplayAdapterDryRunComparison.sampleTexturePixels`.
- `webgpuRenderHandoffStub.sampleRecords` remains available as fallback policy
  but is not mixed into presented samples.

Future steps should preserve this baseline unless they explicitly replace it
with a stronger true native path.

## 10. Step52 Backend Frame Prototype

Step52 introduces `webgpuBackendFramePrototype` as the first WebGPU backend
frame unit. It does not connect the production viewer loop. Instead, it
coordinates the already validated pieces as one bounded frame:

1. viewer canvas currentTexture acquisition,
2. bounded first-present guard,
3. native bounded color sample retention,
4. bounded color source selection,
5. viewer canvas bounded color present submission,
6. backend frame summary.

The frame prototype is `ok` only when all of these are true:

- currentTexture path is ready under `webgpu-exclusive`,
- bounded first-present has succeeded,
- native bounded samples are ready,
- selector has selected samples,
- present submitted a command buffer,
- selector samples were used,
- fallback samples were suppressed by selector samples,
- WebGL2 hybrid rendering was prevented.

This gives later WebGPU backend work a single frame-level contract to extend.
Future steps should add more true native inputs or broader frame work inside
this unit before connecting production display scheduling.

## 11. Step53 Backend Frame Input Expansion

Step53 keeps Step40 as the highest-priority stable source, but the backend
frame no longer treats Step40 as the only true native input shape. Step39 render
target handoff samples and Step38 framebuffer-free tile output samples are
normalized through the same bounded sample contract:

- Step40 uses `rgbaFloat` from constrained display texture samples.
- Step39 uses `resolvedRgb` from render target handoff samples.
- Step38 uses `resolvedRgb` from framebuffer-free tile output samples.

`webgpuBackendFramePrototype.inputSourceContract` reports raw sample counts,
presentable sample counts, selected source kind, true native source count, and
whether input readiness has expanded beyond Step40. Selector priority is still
Step40, then Step39, then Step38, then render-handoff-derived fallback. Fallback
output remains a fallback and must not be classified as true native success.

## 12. Step54 Frame Budget and Continuation Readiness

Step54 keeps the production viewer loop disconnected, but the backend frame is
prepared to behave like a repeatable frame unit. `webgpuBackendFramePrototype`
reports two additional contracts:

- `frameBudgetContract`: the bounded N-sample budget for selected and presented
  samples. The current cap is `DEFAULT_MAX_BOUNDED_COLOR_SAMPLES`, and the
  frame reports selected count, present count, budget utilization, and whether
  the frame could scale within the current budget.
- `continuationFrameContract`: a dry-run-safe repeated frame contract. It
  records frame index, previous-frame status when provided, and confirms the
  coordinator can be invoked repeatedly under the same exclusive canvas,
  sample, and fallback contracts.

`backendFrameReady` includes currentTexture readiness, first-present success,
selector/present success, fallback suppression, WebGL2 hybrid prevention,
sample-budget readiness, and continuation readiness. This is still not a
production display connection or interactive camera implementation.

## 13. Step55 Repeated-Run Lifecycle Prototype

Step55 keeps Step54 as the single submitted backend frame and adds
`webgpuBackendFrameLifecyclePrototype` as the repeated-run coordinator boundary.
The lifecycle prototype does not connect a production frame scheduler and does
not add extra viewer canvas submissions after the initial Step54 frame. Instead,
it validates that the same backend frame unit can be represented as a small
sequence of frame summaries:

- frame state: each summary carries `frameIndex`, previous-frame status, and
  readiness so later work can replace the dry-run replay with a real scheduler.
- lifecycle guards: currentTexture readiness, bounded first-present success,
  selector-selected samples, fallback suppression, and WebGL2 hybrid prevention
  must remain stable across the repeated frame summaries.
- submission policy: frame 0 owns the actual Step54 command submission;
  continuation frames are dry-run lifecycle checks and must not be counted as
  extra production submissions.
- budget continuity: `frameBudgetContract` and `continuationFrameContract` from
  Step54 are carried into `lifecycleContract` so repeated frames inherit the
  bounded sample budget and reusable-frame constraints.

This moves the backend prototype from a single bounded present toward a
frame-lifecycle shape while keeping WebGPU normal backend, WebGL2 fallback /
validation oracle, Three.js camera input adapter, and CUDA Reference roles
separate.

## 14. Step56 Controlled Repeated Execution

Step56 keeps the production viewer loop disconnected, but it stops treating
continuation frames as replay-only summaries. `webgpuBackendFrameControlledRepeatedExecution`
uses the Step54 backend frame as frame 0, then invokes the same backend frame
builder for continuation frames under the same guarded `webgpu-exclusive`
ownership contract.

- execution policy: the controlled run is limited to two or three backend
  frames inside one capture. It is not a `requestAnimationFrame` production
  scheduler.
- per-frame guard: every executed frame must reacquire the viewer canvas
  currentTexture path, keep bounded first-present readiness, use selector
  samples, suppress fallback mixing, and keep WebGL2 hybrid rendering disabled.
- submit policy: every controlled frame must submit bounded color present work
  and wait for queue completion. The summary reports
  `executedBackendFrameSubmissions`, `repeatedSubmitCount`, and per-frame submit
  status.
- state continuity: each continuation frame receives the previous backend frame
  prototype and reports previous-frame readiness so later scheduler work can
  replace the capture-local loop without changing the sample or fallback
  contract.

This is the first repeated-submit backend frame step. It still does not connect
production display scheduling, mouse interaction, or WGSL SH/color parity.

## 15. Step57 Viewer Loop Adapter Boundary

Step57 keeps production scheduling disconnected, but wraps the Step56 controlled
execution path in `webgpuBackendViewerLoopAdapter`. The adapter is the manual,
bounded entrypoint that a future viewer loop can call without changing the
sample source contract or the exclusive canvas guard.

- frame execution API: `frameExecutionApiContract` defines a single frame call
  shape with `frameIndex`, previous backend frame state, camera snapshot,
  exclusive canvas guard, and sample/fallback contracts as inputs.
- resource lifecycle: `frameResourceLifecycleContract` separates reusable
  resources such as `GPUDevice`, `viewerCanvasState`, Step38-40 bounded samples,
  and selector policy from per-frame resources such as currentTexture, command
  encoder, render pass, command buffer, and queue completion.
- adapter policy: the adapter is callable from a future viewer loop, but
  `productionLoopConnected` remains false. It only runs the bounded controlled
  frame count used by the capture.
- camera boundary: Three.js camera data is treated as a camera input snapshot;
  the adapter does not implement interactive camera behavior or mutate the
  projection contract.

## 16. Step58 Viewer Lifecycle Integration Boundary

Step58 keeps the production viewer loop disconnected, but moves the Step57
adapter closer to the existing viewer render lifecycle. The integration boundary
is guarded by `webgpuBackendMode=webgpu-exclusive`,
`webgpuAllowViewerCanvasPresentation=true`, and the explicit
`webgpuBackendViewerLoopHook=true` flag.

- viewer lifecycle hook: `viewer_app_gpu.renderCurrentFrame` now exposes a
  guarded `webgpuBackendViewerLifecycleIntegrationBoundary` summary when WebGPU
  exclusive mode suppresses the WebGL2 render frame.
- adapter connection: the WebGPU dry-run connects the same boundary to
  `webgpuBackendViewerLoopAdapter`, so the Step57 adapter result and the viewer
  lifecycle hook can be validated together without enabling production
  scheduling.
- ownership policy: WebGL2 remains suppressed under the exclusive guard, and
  WebGPU presentation is not mixed with an active WebGL2 viewer canvas.
- camera boundary: the hook carries a Three.js camera snapshot into the boundary
  contract, but it does not implement interactive camera behavior or mutate the
  projection contract.

This moves the prototype from repeated submits toward a viewer-loop-callable
backend API while preserving WebGPU normal backend, WebGL2 fallback / validation
oracle, and CUDA Reference role separation.

## 17. Step59 Controlled Viewer Lifecycle Execution Path

Step59 keeps production scheduling disconnected, but advances the Step58
boundary from hook readiness to a guarded controlled execution path. When the
viewer frame enters `renderCurrentFrame` with `webgpuBackendMode=webgpu-exclusive`,
`webgpuAllowViewerCanvasPresentation=true`, and the explicit
`webgpuBackendViewerLoopHook=true` flag, the hook can invoke one bounded WebGPU
backend execution path and summarize the result as
`webgpuBackendViewerLifecycleControlledExecution`.

- guarded invocation: the controlled path is only requested after the Step58
  integration boundary is ready. WebGL2 rendering remains suppressed for that
  frame, so WebGPU presentation is not mixed with an active WebGL2 viewer
  canvas.
- adapter execution: the hook calls the same Step57 viewer-loop adapter path
  used by the dry-run backend frame prototype, then records invocation count,
  submitted frame count, repeated submit count, selected source, and sample
  count without storing the full backend dry-run result in the viewer render
  result.
- viewer-owned inputs: the hook passes a viewer lifecycle camera snapshot,
  canvas state, backend mode, presentation guard, and frame invocation source
  into the controlled execution summary. Three.js / OrbitControls remain camera
  input adapters rather than the WebGPU renderer core.
- fallback policy: Step40 true native selected samples remain the success path.
  Render-handoff or reference-assisted fallback remains available only when no
  selector samples exist and must not be reported as true native success.

This is the first path where the existing viewer render lifecycle can explicitly
invoke a controlled WebGPU backend frame execution. It still does not enable the
production requestAnimationFrame loop by default, implement interactive camera
ownership, or complete WGSL SH/color parity.
