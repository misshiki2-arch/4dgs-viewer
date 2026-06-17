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

## 18. Step60 Viewer Backend Frame Executor Boundary

Step60 separates viewer lifecycle execution from capture/debug recording. The
existing `renderCurrentFrame` guarded hook now calls a viewer backend frame
executor boundary, and that executor calls the low-level WebGPU backend frame
runner. `captureWebGpuVisibleRecordDryRunDebug` remains a JSON recorder and
validation oracle rather than the execution boundary used by the viewer
lifecycle.

- executor boundary: `webgpuBackendViewerFrameExecutor` records the guarded
  viewer backend call inputs: backend mode, viewer canvas presentation guard,
  hook flag, camera snapshot, canvas ownership state, frame index, and backend
  submit result.
- capture separation: capture/debug code can still record the same backend
  result, but Step60 marks `captureDebugFunctionDependency=false` for the viewer
  executor contract and records the capture layer as a validation oracle.
- WebGL2 separation: the executor only runs under the same exclusive guard as
  Step59 and keeps WebGL2 frame lifecycle suppression visible in the summary.
- future data scale: the backend boundary must not require all Gaussians or all
  temporal frames to be uploaded and resident in GPU/VRAM at once. Streaming,
  chunking, LOD, and partial upload are not implemented in Step60, but record
  contracts, buffer layouts, and backend frame boundaries should remain
  extensible so those policies can be added without redefining the viewer
  lifecycle ownership model.

This moves the prototype closer to a normal WebGPU viewer backend entrypoint
while keeping production scheduling, interactive camera ownership, streaming,
and WGSL SH/color parity out of scope for this step.

## 19. Step61 Viewer Backend Runtime Runner Contract

Step61 adds a runtime runner layer below `webgpuBackendViewerFrameExecutor`.
The executor still owns the guarded viewer lifecycle call, but the runner owns
the replaceable backend frame execution contract. The current implementation
still uses the validated WebGPU dry-run backend behind that contract; future
normal WebGPU backend work can replace that implementation without changing the
viewer lifecycle guard or recorder shape.

- runner contract: `webgpuBackendRuntimeRunner` records backend mode, viewer
  canvas guard, hook state, frame index, camera snapshot availability, canvas
  ownership state, resource lifecycle, submit counts, and backend implementation
  kind.
- recorder separation: capture/debug remains a validation oracle and JSON
  recorder. It observes runner output through `recorderObservation`; it is not
  the runner's execution boundary.
- canonical present summary: selected source kind, selection mode, color present
  sample count, sample sources, presented sample count, fallback mixing status,
  command submission, and queue completion are exposed in one runner summary.
- replacement policy: the runner marks the current dry-run implementation as
  replaceable. A production WebGPU backend frame runner should preserve this
  contract while swapping the backend implementation.
- data scale policy: Step61 still does not implement streaming, chunking, LOD,
  or partial upload, and it continues to avoid a design requirement that all
  Gaussians or all temporal frames must be resident in GPU/VRAM at once.

This makes the path `renderCurrentFrame -> executor -> runtime runner -> backend
implementation` explicit while preserving WebGPU exclusive ownership, WebGL2
fallback / validation oracle separation, Three.js camera input adapter status,
and deferred WGSL SH/color parity.

## 20. Step62 First Normal WebGPU Backend Implementation Path

Step62 adds the first normal WebGPU backend implementation path below
`webgpuBackendRuntimeRunner`. The runner can now select
`webgpu-normal-backend-frame-implementation` with an explicit
`webgpuBackendImplementation` flag instead of treating the validated dry-run
runtime as the only callable backend body.

- implementation selection: `webgpuBackendRuntimeRunner.runnerContract` records
  the selected backend implementation kind and whether it came from the explicit
  normal WebGPU backend path or the validation-oracle dry-run path.
- normal implementation path: `webgpuNormalBackendFrameImplementation` owns the
  first normal backend implementation contract, consumes the existing guarded
  viewer canvas ownership state, and requires Step40 true native selected
  samples, submitted bounded color present output, and preserved fallback
  suppression.
- recorder separation: the WebGPU visible-record dry-run remains a validation
  oracle and JSON recorder. It can provide validated bounded input and observe
  runner output, but the normal implementation path is now selected and reported
  as the backend implementation body behind the runner contract.
- fallback policy: render-handoff fallback remains available only when selector
  samples are absent. It must not be mixed into a normal implementation success
  that selected Step40 constrained-display-adapter samples.
- scope limits: production scheduling, interactive camera ownership, SH color
  parity, streaming, chunking, LOD, and partial upload remain future work. The
  Step60/61 data-scale constraint still applies: the backend boundary must not
  require all Gaussians or all temporal frames to be resident in GPU/VRAM at
  once.

This moves the final segment of
`renderCurrentFrame -> executor -> runtime runner -> backend implementation`
from a dry-run-only body to a selectable normal WebGPU backend implementation
path, while preserving WebGPU exclusive ownership and WebGL2 fallback /
validation-oracle separation.

## 21. Step63 Normal Backend Frame Input and Present Output Contracts

Step63 moves more responsibility into
`webgpu-normal-backend-frame-implementation`. The normal backend implementation
now owns a viewer frame input contract and a present output contract instead of
only reporting dry-run observations from the validation oracle.

- frame input contract: `frameInputContract` records frame index, invocation
  source, camera snapshot availability, viewer canvas state, exclusive guard
  state, and source selection policy. This is the normal backend's execution
  input boundary.
- present output contract: `presentOutputContract` records selected source
  kind, selection mode, presented samples, sample sources, submit state,
  queue completion, and fallback policy as the normal backend's canonical
  output boundary.
- oracle separation: the WebGPU visible-record dry-run can still provide
  comparison/debug observations and the currently validated bounded samples, but
  it does not own the normal backend frame input or present output contracts.
- Step40 stable path: Step63 continues to require
  `step40-constrained-display-adapter` selected samples, two presented samples,
  no render-handoff fallback mixing, and three controlled frame submissions for
  success.
- scope limits: production scheduling, interactive camera implementation,
  streaming, chunking, LOD, partial upload, and WGSL SH/color parity remain
  future work. Full dataset GPU residency is still not required by the backend
  boundary.

This keeps the Step62 selectable implementation path, but gives the normal
backend implementation its own input and output contracts so later production
rendering can replace the validation-backed body without changing the viewer
lifecycle guard.

## Step64 Frame Constants And Uniform Resource Boundary

Step64 moves the normal backend input boundary from "camera snapshot exists" to
explicit frame constants that a future WebGPU renderer can consume.
`renderCurrentFrame()` still treats Three.js as a camera input adapter, but now
passes view matrix, projection matrix, view-projection matrix, viewport, time,
and frame index into `webgpu-normal-backend-frame-implementation`.

- frame constants contract: `frameConstantsContract` is owned by the normal
  backend implementation and records projection availability, camera pose
  availability, viewport, time, frame index, and matrix readiness. This is the
  backend-facing camera/projection input boundary.
- uniform resource preparation contract:
  `uniformResourcePreparationContract` defines the future uniform buffer layout
  for frame constants, including required fields, available/missing fields,
  byte size, padded binding size, and alignment policy. Step64 does not create
  or upload the GPU buffer yet.
- oracle separation: the visible-record dry-run recorder can still observe and
  compare the frame result, but it does not own frame constants or the uniform
  resource boundary.
- Step63 baseline: Step40 constrained-display selected samples, two presented
  samples, fallback suppression, and three controlled frame submissions remain
  the success baseline.
- scope limits: production scheduler connection, interactive camera behavior,
  streaming, chunking, LOD, partial upload, full-dataset GPU residency, and WGSL
  SH/color parity remain future work.

This gives the first normal backend implementation a concrete shader-input
preparation boundary without changing the validated bounded present body.

## Step65 Normal Backend Uniform Resource Lifecycle

Step65 turns the Step64 uniform preparation boundary into an actual WebGPU
resource lifecycle owned by `webgpu-normal-backend-frame-implementation`.
The normal backend now packs frame constants into a `Float32Array`, creates a
uniform `GPUBuffer`, writes the packed data with `queue.writeBuffer`, and records
the lifecycle as `uniformResourceLifecycleContract`.

- resource ownership: the normal backend owns the uniform buffer resource
  lifecycle. The visible-record recorder can observe the result, but it does not
  own buffer creation, update, or disposal.
- packed layout: Step65 uses the Step64 layout
  `frameIndex_time_viewportWidth_viewportHeight`, `viewMatrix4x4`,
  `projectionMatrix4x4`, and `viewProjectionMatrix4x4`. The payload is 52
  `float32` values, 208 bytes, padded to the 256-byte binding boundary recorded
  by the preparation contract.
- lifecycle policy: Step65 uses a per-call transient buffer for controlled
  execution and destroys it after write validation. A future runner-owned device
  and resource cache can replace this without changing the frame constants
  contract.
- bind group boundary: Step65 records that the resource is ready for a future
  bind group boundary, but it does not create bind groups or consume the uniform
  in WGSL yet.
- baseline: Step40 constrained-display selected samples, two presented samples,
  fallback suppression, and three controlled frame submissions remain the
  success baseline.
- scope limits: production scheduling, interactive camera behavior, streaming,
  chunking, LOD, partial upload, full-dataset GPU residency, and WGSL SH/color
  parity remain future work.

This is the first normal backend step where frame constants become a real GPU
resource instead of only a summary contract.

## Step66 Bind Group And Minimal Uniform Consumption

Step66 connects the Step65 uniform buffer to a real bind group and a minimal
WGSL compute pass owned by `webgpu-normal-backend-frame-implementation`.
The pass reads the first `vec4<f32>` of the frame constants uniform and writes it
to a small storage buffer that is copied back for validation.

- bind group ownership: the normal backend creates the bind group layout and
  bind group for the frame constants uniform. This confirms the resource is not
  only writable, but also bindable by a backend shader boundary.
- minimal WGSL consumption: Step66 uses a one-workgroup compute pass to read
  `frameIndex`, `timeSeconds`, `viewport.width`, and `viewport.height` from the
  uniform buffer. The readback must match the packed frame constants prefix.
- validation output: `uniformShaderConsumptionContract` records bind group
  layout creation, bind group creation, compute pipeline creation, dispatch,
  readback completion, expected values, actual values, and max absolute
  difference.
- baseline: Step40 constrained-display selected samples, two presented samples,
  fallback suppression, and three controlled frame submissions remain the
  success baseline.
- scope limits: this is not production Gaussian shading. WGSL SH/color parity,
  scene-data shader consumption, production scheduling, interactive camera
  behavior, streaming, chunking, LOD, partial upload, and full-dataset GPU
  residency remain future work.

This gives the normal backend a small but real shader consumption boundary for
frame constants before the backend starts consuming scene data.

## Step67 Selected Sample Storage Buffer Consumption

Step67 extends the Step66 uniform bind group boundary with a normal-backend-owned
storage buffer for the Step40 true native selected samples. The selected samples
remain the same constrained-display source used by the stable present path; the
normal backend now packs their minimal pixel, color, source classification, and
record fields into GPU memory and validates that WGSL can read those fields.

- sample resource ownership: `webgpu-normal-backend-frame-implementation`
  creates and writes a transient selected-sample storage buffer after the
  Step40 selector/present output is known.
- packed sample fields: each sample stores `samplePx.x`, `samplePx.y`,
  `colorAlpha.rgba`, a numeric source-kind code, and `recordIndex`.
- minimal WGSL consumption: the Step67 compute pass reads both the frame
  constants uniform and the first packed selected sample through one bind group,
  then copies the combined output back for CPU comparison.
- fallback policy: render-handoff fallback samples are rejected as sample-buffer
  input when selector-selected Step40 samples are present.
- baseline: Step66 uniform readback, Step40 selected source, two presented
  samples, fallback suppression, and three controlled frame submissions remain
  the success baseline.
- scope limits: this is still not production Gaussian shading, WGSL SH/color
  parity, production scheduling, interactive camera behavior, streaming,
  chunking, LOD, partial upload, or full-dataset GPU residency.

This gives the normal backend its first tiny scene-data GPU consumption boundary
without changing the viewer loop into a production scheduler.

## Step68 Minimal GPU Color Output Surface

Step68 turns the Step67 selected-sample consumption boundary into a minimal GPU
color output surface generation boundary. The normal backend still uses the
Step40 constrained-display selected samples, but the compute pass now writes
`colorAlpha.rgba` into an RGBA float storage-buffer surface at coordinates
derived from each sample's `samplePx.x/y`.

- output resource ownership: `webgpu-normal-backend-frame-implementation`
  creates a transient `storage-buffer-rgba-float-surface` owned by the normal
  backend.
- GPU write path: WGSL reads the selected sample storage buffer, computes a
  bounded pixel index from `samplePx.x/y`, and writes the sample color into the
  output surface.
- validation output: the output surface is copied back and compared with a CPU
  expected surface built from the same Step40 selected sample fields.
- presentation boundary: the surface is marked as ready for a future
  viewer-canvas currentTexture, render-target texture, or storage-texture copy
  path, but Step68 does not connect it to production presentation.
- baseline: Step67 uniform/sample readback, Step40 selected source, two
  presented samples, fallback suppression, and three controlled frame
  submissions remain the success baseline.
- scope limits: this is not production Gaussian shading, WGSL SH/color parity,
  production scheduling, interactive camera behavior, streaming, chunking, LOD,
  partial upload, or full-dataset GPU residency.

This gives the normal backend a GPU-generated color surface that can become the
next handoff target before it is wired to viewer-canvas presentation.
