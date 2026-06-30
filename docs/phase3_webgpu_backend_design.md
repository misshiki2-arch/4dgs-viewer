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

## Step69 Normal Backend Output / Presentation Handoff Boundary

Step69 keeps the Step68 storage-buffer RGBA float surface, then treats it as
the normal backend's current frame output. The compute-generated surface is
copied on the GPU into a backend-owned handoff buffer, and that handoff buffer is
read back only to validate that the future presentation boundary receives the
same pixels.

- output contract: `common_4dgs_backend_output_contracts.js` builds and
  validates `normalBackendOutputContract`; it records format, extent,
  coordinate origin, source surface, ownership, lifecycle, and future
  presentation targets from explicit inputs.
- GPU-side handoff: the Step68 color output surface is connected to the handoff
  resource with `copyBufferToBuffer`; CPU readback validates the handoff without
  turning it into production presentation.
- ownership: `webgpu-normal-backend-frame-implementation` owns the current
  frame output until a future viewer-canvas currentTexture, render-target, or
  storage-texture copy adapter explicitly accepts it.
- guardrails: viewer canvas presentation, render target presentation, and
  production scheduling remain disconnected; WebGL2 is still a fallback and
  validation oracle, not a same-frame presentation partner.
- baseline: Step67 uniform/sample consumption, Step68 output surface generation,
  Step40 selected samples, fallback suppression, and three controlled frame
  submissions remain required.
- failure behavior: unavailable or mismatch cases still return structured
  output/handoff contracts, so summary generation does not depend on free
  variables from the implementation body.

This creates the resource/contract boundary needed before the normal backend
output can be wired to an actual presentation pass.

## Step70 WebGPU-Only Guarded Presentation Adapter

Step70 consumes the Step69 normal-backend output with a WebGPU-only guarded
presentation adapter. The adapter is invoked from the normal backend runtime
chain while the handoff buffer is still alive, then a compute pass reads that
handoff buffer and writes an offscreen `rgba8unorm` storage texture that is
compatible with a future presentation path.

- adapter contract: `common_4dgs_backend_output_contracts.js` builds
  `guardedPresentationAdapterContract`, including success and unavailable
  readiness fields.
- GPU consumption: `webgpu_guarded_presentation_adapter.js` reads the handoff
  storage buffer and writes an offscreen storage texture with a WebGPU compute
  command.
- validation output: the offscreen texture is copied back with padded
  `bytesPerRow` handling and compared against the expected quantized RGBA8
  surface derived from the Step69 handoff data.
- guardrails: currentTexture and render-target presentation remain
  disconnected; the adapter proves consumption of the handoff output without
  entering production scheduling or same-frame WebGL2 presentation.
- baseline: Step67 uniform/sample consumption, Step68 color output surface,
  and Step69 handoff output contracts remain required before the adapter can
  report ready.

This gives the normal backend its first WebGPU presentation-layer consumer
without yet taking over production viewer-canvas presentation.

## Step71 Viewer Presentation Bridge Boundary

Step71 promotes the Step70 adapter output from a validated offscreen consumer
into a viewer presentation lifecycle bridge. The adapter still consumes the
Step69 handoff resource, but it now also copies the adapter's `rgba8unorm`
texture into a render-target bridge texture that a future viewer presentation
pass can own.

- currentTexture attempt: the contract records a guarded currentTexture
  connection attempt. The Step71 normal backend does not yet receive the viewer
  canvas WebGPU context at this boundary, so direct currentTexture connection is
  blocked with an explicit reason instead of being silently skipped.
- render-target bridge: `webgpu_guarded_presentation_adapter.js` performs a
  GPU-side `copyTextureToTexture` from the Step70 adapter target into a
  presentation bridge texture, then validates the bridge readback against the
  adapter output.
- contract output: `common_4dgs_backend_output_contracts.js` builds
  `presentationBridgeContract` with currentTexture capability state, bridge
  resource format/extent/ownership/lifecycle, GPU command path, and future
  presentation targets.
- guardrails: the bridge remains WebGPU-only, keeps WebGL2 hybrid rendering
  disabled, does not enter production scheduling, and preserves the Step67
  uniform/sample, Step68 color output surface, Step69 handoff, and Step70
  adapter readiness requirements.

This creates a concrete render-target handoff for the viewer presentation
lifecycle while leaving currentTexture takeover for the next guarded boundary
where the viewer canvas WebGPU context is explicitly passed into the adapter.

## Step72 Guarded CurrentTexture Handoff Boundary

Step72 passes the viewer canvas lifecycle state into the Step70/71 guarded
presentation adapter. The normal backend now provides `viewerCanvasState` to
the uniform/sample/color output resource lifecycle, and the adapter can use that
canvas only under `webgpu-exclusive`, `webgpuAllowViewerCanvasPresentation=true`,
and suppressed WebGL2 frame lifecycle.

- currentTexture path: the adapter configures the viewer canvas WebGPU context,
  acquires `getCurrentTexture()`, renders the adapter output texture into that
  currentTexture with a small render pass, and validates the copied readback
  against the adapter output.
- format policy: the bridge samples the `rgba8unorm` adapter target in a render
  pass instead of using a blind texture copy, so `rgba8unorm` and `bgra8unorm`
  canvas formats remain an explicit render-target conversion boundary.
- render-target bridge: the Step71 render-target bridge remains available as a
  lifecycle validation handoff, but Step72 readiness additionally requires the
  currentTexture render pass, readback, and match result.
- guardrails: WebGL2 hybrid rendering stays disabled, production scheduling is
  not connected, fallback samples are not promoted to true native output, and
  full-dataset GPU residency is still not required.

This is the first normal-backend guarded path where the presentation adapter
receives viewer canvas lifecycle state and writes the adapter output into the
actual viewer canvas currentTexture boundary.

## Step73 Viewer-Owned Guarded Presentation Pass Boundary

Step73 keeps the Step72 currentTexture render pass intact, but moves ownership
of that result up to the viewer frame lifecycle. The `renderCurrentFrame()`
guarded path now receives a structured viewer-owned presentation pass contract
from `webgpuBackendViewerFrameExecutor`, so the pass is no longer represented
only as a normal-backend validation artifact.

- viewer-owned pass: `webgpuBackendViewerFrameExecutor` consumes the runtime
  runner's normal backend `presentationBridgeContract` and builds a
  `viewerFramePresentationPassContract` through the shared backend output
  contract helper.
- ownership contract: the contract records that `renderCurrentFrame` invoked
  the executor chain, the runtime runner completed, currentTexture was
  acquired/configured, a render pass was submitted, and readback matched the
  adapter output.
- recorder separation: dry-run capture observes the executor-owned contract and
  remains a validation oracle; it does not own the presentation pass.
- guardrails: the pass still requires `webgpu-exclusive`,
  `webgpuAllowViewerCanvasPresentation=true`, `webgpuBackendViewerLoopHook=true`,
  suppressed WebGL2 frame lifecycle, no fallback sample mixing, and no
  production scheduler connection.

This establishes the viewer frame lifecycle as the owner of the guarded WebGPU
presentation pass boundary. A future step can promote this boundary into a
scheduler contract without changing the WebGPU/WebGL2 separation or requiring
full dataset GPU residency.

## Step74 Scheduler-Owned Guarded Frame Presentation Boundary

Step74 promotes the Step73 viewer-owned presentation pass into a guarded
scheduler/frame-loop boundary. The existing `createRenderScheduler()` remains a
bounded requestAnimationFrame scheduler, but it now records the frame request,
callback entry, render invocation, render completion, and consumed viewer-owned
presentation pass as a structured contract.

- scheduler boundary: `viewer_render_scheduler` passes a scheduler frame state
  into `renderCurrentFrame()` and attaches `webgpuSchedulerFramePresentationBoundary`
  to the returned render result.
- ownership split: the scheduler owns frame request and completion accounting;
  `renderCurrentFrame -> executor -> runtime runner -> normal backend` still owns
  the guarded WebGPU work and currentTexture render pass.
- contract source: the scheduler boundary consumes Step73
  `viewerFramePresentationPassContract` through the shared backend output
  contract builder and reports currentTexture acquire/render/readback readiness.
- scope limits: this is a scheduler-owned guarded boundary, not a full
  production scheduler connection. WebGL2 hybrid rendering stays disabled,
  fallback samples are not promoted to true native output, and full-dataset GPU
  residency is not required.

The next step can decide how much of this guarded scheduler boundary should
become the regular WebGPU frame loop contract without mixing WebGPU presentation
with WebGL2 rendering.

## Step75 Camera/Control/Scheduler-Aware Visible Output Boundary

Step75 changes the success criterion from "the guarded currentTexture plumbing
exists" to "the WebGPU normal backend produces camera-aware visible output
through that plumbing." The implementation keeps Three.js and OrbitControls on
the input-adapter side: the backend consumes the viewer camera snapshot, frame
constants, projection matrices, viewport, and scheduler frame state, but does
not own or mutate controls.

- selected approach: Step75 combines the Step40 true-native bounded source with
  visible-record data. When WebGPU visible records are available, their
  camera/projection-derived `px/py/depth` and reference-assisted `colorAlpha`
  become normal-backend visible samples. If that input is unavailable, the
  contract clearly falls back to enlarged Step40 true-native selected samples
  rather than render-handoff fallback samples.
- visible output: the normal backend packs these visible samples into its GPU
  sample storage buffer. The minimal WGSL pass writes enlarged color patches
  into the backend-owned color output surface, then the existing output handoff,
  guarded presentation adapter, presentation bridge, viewer-owned pass, and
  scheduler-owned currentTexture boundary consume the result.
- scheduler contract: `cameraAwareVisibleOutputContract` records that the
  visible output used camera/projection data, scheduler-owned frame request
  state, and the guarded currentTexture path. This is intentionally stronger
  than a debug clear or fixed-color rectangle.
- guardrails: WebGPU and WebGL2 display paths remain non-hybrid,
  fallback samples are not mixed with selector-selected samples, production
  scheduling is not fully connected, and full-dataset GPU residency is not
  required.

This gives Phase 3 its first explicitly visible WebGPU normal-backend output
boundary while preserving the validation-oracle role of the dry-run recorder
and the WebGL2 fallback path.

## Step76 Many Camera-Aware Visible Samples Boundary

Step76 moves from Step75's two enlarged Step40 patches to many camera-aware
samples without changing the scheduler/currentTexture ownership model. The
preferred long-term path remains true WebGPU visible records with
`validRecordCount > 0`, but the current capture can still produce zero valid
visible-record samples. Step76 therefore uses an explicit validation-assisted
bridge: screenCoarse candidate indices and CPU-materialized 4D state positions
are projected with the viewer WebGPU projection contract, colored from the
reference-assisted render payload fields, then passed into the normal backend
sample buffer.

- selected approach: B, validation-assisted bridge. It is not reported as
  true-native visible-record success, and it is not render-handoff fallback.
- input contract: `cameraAwareVisibleOutputContract` records
  `inputSourceKind`, `inputSourceLineage`, `sourceClassification`,
  visible-record sample count, bridge sample count, rendered patch count, and
  whether positions came from the viewer camera/projection contract.
- output path: the same normal backend GPU sample buffer, color output surface,
  guarded adapter, presentation bridge, viewer-owned presentation pass, and
  scheduler-owned currentTexture path consume the larger sample set.
- guardrails: debug fill remains false, WebGPU/WebGL2 hybrid presentation is
  prevented, fallback samples are not mixed, production scheduler connection,
  full Gaussian shading, SH parity, streaming, chunking, LOD, and partial upload
  remain deferred.

This produces a denser visible WebGPU output while keeping lineage honest: if
the true WebGPU visible-record path is still empty, the summary explains why the
frame succeeded through the validation-assisted bridge instead.

Step76 fix1 tightens that honesty rule. The summary may only mark Step76 as
successful when nonzero validation-assisted bridge samples are generated and
the normal backend sample buffer actually consumes that bridge lineage. If the
bridge batch is empty and the frame uses Step40 selector-selected samples
instead, the capture is reported as blocked/incomplete for Step76 even if the
Step75 currentTexture path remains healthy. The bridge projection helper may
use a camera/view-derived validation fallback placement for screenCoarse
candidates that fail the strict visible-record depth gate, but that path remains
classified as `bridge`, never as true-native visible-record output.

## Step77 WebGPU-Owned Visible Sample Generation Boundary

Step77 reduces the Step76 validation-assisted bridge dependency by adding a
WebGPU-owned native-compatible sample generation boundary. The true
visible-record path is still preferred, but when `validRecordCount` remains 0,
the backend can now run a small WebGPU compute pass over screenCoarse candidate
indices and viewer projection params to create a normal-backend-consumable
sample batch. This is classified as `native-compatible`, not true-native
visible-record success.

- selected approach: B, WebGPU-owned native-compatible sample generation. The
  compute pass owns sample position/color generation for this boundary; the
  Step76 validation-assisted bridge remains available as a baseline/fallback
  lineage and is reported separately.
- input contract: `cameraAwareVisibleOutputContract` records
  `webgpuOwnedSampleCount`, `webgpuOwnedGenerationMode`,
  `webgpuOwnedProjectionGate`, `validationAssistedBridgeSampleCount`,
  consumed source kind/lineage/classification, and currentTexture readiness.
- output path: the generated sample batch flows through the existing normal
  backend sample storage buffer, color output surface, guarded adapter,
  presentation bridge, viewer-owned presentation pass, and scheduler-owned
  currentTexture path.
- guardrails: WebGPU-owned generation is not WebGL2 fallback, does not claim
  true-native success, does not mix fallback samples, and keeps production
  scheduler connection, full Gaussian shading, SH parity, streaming, chunking,
  LOD, and partial upload deferred.

The next true-native step is to replace the candidate-derived native-compatible
placement with a WebGPU visible-record projection/visibility gate that produces
`validRecordCount > 0` without relying on the validation bridge.

## Step78 True WebGPU Visible Record Boundary

Step78 moves the primary path from Step77's native-compatible sample generator
to a nonzero WebGPU visible-record path. The existing WGSL visible-record
compute now reports explicit projection/visibility gate diagnostics and can
use raw xyz as a minimal WebGPU-owned position source when the CPU-materialized
4D conditional state is unavailable for the current screenCoarse candidate
batch. The visible-record output itself owns `srcIndex`, `valid`, `px`, `py`,
and `depth`, and the normal backend consumes those records before the guarded
adapter/currentTexture presentation path.

- selected approach: A/B. The existing visible-record projection path is kept,
  with minimal gate repair and diagnostics rather than replacing the backend
  with another bridge.
- success contract: Step78 is only successful when `validRecordCount > 0`,
  `consumedSourceKind=visible-record`, and the currentTexture readback path
  remains ready. If the path relies on raw xyz repair because 4D conditional
  `statePositions` are unavailable, the visible-record classification is
  `true-native-minimal-visible-record` rather than full 4D state driven
  `true-native`.
- gate diagnostics: `webgpuVisibleRecordGateSummary` records state-position
  availability, raw-position repair count, projection gate pass count, and the
  next full 4D state gate if the minimal raw xyz repair was needed.
- classification: Step78 is not Step77 native-compatible fallback and not the
  Step76 validation-assisted bridge, but raw xyz repair also means it must not
  be reported as full 4D state driven true native. Summary should expose
  `rawPositionRepairUsed=true`, `full4DStateDrivenTrueNative=false`, and
  `trueVisibleRecordPathClassification=true-native-minimal-visible-record` when
  this minimal visible-record path is active.
- baselines: Step76 validation-assisted bridge samples and Step77
  native-compatible samples may remain available as diagnostics/baselines, but
  they are not counted as Step78 success.

This is still not full 4DGS parity: SH/color evaluation, full 4D conditional
state in WGSL, depth sort, compaction, tile-list generation, streaming, chunking,
LOD, and partial upload remain deferred.

## Step79 WebGPU 4D State-to-Visible Pipeline Boundary

Step79 moves the visible-record input from Step78's raw xyz repair branch to an
explicit 4D state source boundary. The state source is still CPU-materialized
for this prototype, but it is now passed through the WebGPU `statePositions`
buffer and consumed by the WGSL visible-record projection path before the
normal backend renders the resulting visible records through the guarded
currentTexture path.

- selected approach: A. The existing `statePositions` and visible-record compute
  path are kept, with a small state-source contract that records whether each
  visible row came from a computed 4D state position, a minimal baseline state
  source, or the old raw xyz repair path.
- success contract: Step79 is successful when `validRecordCount > 0`,
  `consumedSourceKind=visible-record`, the normal backend consumes records whose
  projection came from the `statePositions` source, raw xyz repair is not needed
  for visible-record validity, and the currentTexture readback path remains
  ready under the WebGPU-exclusive guard.
- classification: `full-4d-state-driven-visible-record` is reserved for rows
  whose visible records all came from computed 4D state positions. If the source
  uses baseline state positions because conditional 4D evaluation culled the
  current batch, Summary reports `minimal-4d-state-source-visible-record` and
  keeps the next full 4D state gate explicit.
- baselines: Step76 validation-assisted bridge, Step77 native-compatible
  samples, and Step78 raw xyz repair remain diagnostics only. They must not be
  counted as Step79 success.

Full WGSL 4D state evaluation, SH/color parity, depth sort, compaction,
tile-list generation, streaming, chunking, LOD, and partial upload remain
deferred.

## Step80 WebGPU 4D State Evaluation Pipeline

Step80 replaces the Step79 CPU-materialized baseline `statePositions` source
with a small WebGPU 4D state evaluator boundary. The evaluator consumes
screenCoarse candidate raw positions plus per-record time/scale-time values,
uses the viewer build timestamp and state parameters, writes computed
`statePositions` on the GPU, and feeds those positions into the existing
visible-record projection compute before the normal backend renders through
the guarded currentTexture path.

- selected approach: B. A focused `webgpu_4d_state_evaluator.js` module owns
  the partial WebGPU state evaluation pass so the visible-record compute stays
  responsible for projection/visibility rather than for all 4D state math.
- success contract: Step80 is successful when
  `computed4DStatePositionCount > 0`, `webgpuComputedStatePositions=true`,
  visible records generated from those computed positions are consumed by the
  normal backend, raw xyz repair is not needed, and currentTexture readback
  remains ready under the WebGPU-exclusive guard.
- classification: this step is `partial-webgpu-4d-state-evaluated`, not full
  4D state driven parity. Full WGSL covariance/rotation/SH parity remains
  deferred and must not be claimed by this boundary.
- diagnostics: `webgpuVisibleRecordGateSummary` records
  `partialWebGpu4DStateRecordCount`, `webgpu4DStateEvaluationMode`,
  baseline-state record count, raw repair count, projection gate pass count,
  and the next full 4D state gate.
- baselines: Step76 validation-assisted bridge, Step77 native-compatible
  samples, Step78 raw xyz repair, and Step79 CPU-materialized state positions
  remain diagnostic baselines only. They must not be counted as Step80
  WebGPU-computed state evaluation success.

Full realtime 4DGS rendering still requires WGSL parity for the 4D conditional
state/covariance path, SH/color evaluation, depth sort, compaction, tile-list
generation, streaming, chunking, LOD, and partial upload.

## Step81 WebGPU 4D Gaussian Attribute Evaluation Pipeline

Step81 extends the Step80 partial WebGPU state evaluator so the same WebGPU
pipeline also emits a partial Gaussian render-attribute payload. The evaluator
computes state positions, `radiusPx`, `colorAlpha.rgb`, `colorAlpha.a`, and a
temporal weight from candidate raw position/opacity, `f_dc`, scale, and
time/scale-time inputs. The normal backend then consumes visible-record
`px/py/depth` together with the WebGPU-computed `colorAlpha` and point radius
through the existing sample buffer, color output surface, guarded adapter, and
currentTexture path.

- selected approach: B/C. The evaluator module owns partial 4D state and
  partial Gaussian attribute evaluation; the visible-record compute remains
  responsible for projection and visibility.
- success contract: Step81 is successful when nonzero WebGPU-computed render
  attributes are produced, every normal-backend visible sample consumes that
  computed attribute payload, computed state visible records remain consumed,
  and currentTexture readback stays ready under the WebGPU-exclusive guard.
- attribute classification: `radiusPx`, `colorAlpha.rgb`, `colorAlpha.a`, and
  `temporalWeight` are `partial-webgpu-computed`. Full conic/covariance,
  full SH color, tile range, and depth sort remain deferred and must not be
  reported as full WebGPU attribute parity.
- lineage: Summary reports
  `webgpuGaussianAttributeEvaluationContract`, `computedRenderAttributeCount`,
  `computedRenderPayloadConsumed`, `renderAttributeSources`, and
  `normalBackendConsumedComputedRenderAttributes` so Step81 cannot pass by
  silently falling back to reference-assisted color or selected sample payloads.
- Step81 fix1 tightens the normal-backend consumption proof: the sample
  resource lifecycle must show computed `radiusPx` and `colorAlpha` were used
  for the packed visible samples, and `temporalWeight` must be either reflected
  into alpha or explicitly reported as unavailable/unused. This keeps partial
  attribute success separate from full conic/covariance/SH parity.

Full realtime 4DGS rendering still requires WGSL parity for conditional
covariance/conic, complete SH/color evaluation, depth sort, compaction,
tile-list generation, streaming, chunking, LOD, and partial upload.

## Step82 WebGPU Gaussian Footprint and Tile Payload Pipeline

Step82 extends the Step81 partial evaluator with a screen-space Gaussian
footprint payload. The WebGPU evaluator emits an isotropic `conic`,
`covariance2D`, `radiusPx`, depth-derived `sortKey`, and footprint area from
the computed radius/state output. The visible-record bridge combines that
payload with WebGPU `px/py/depth` to derive bounded `aabb` and `tileRange` for
the normal backend sample buffer.

- selected approach: B/C. The existing WebGPU evaluator owns partial state,
  attribute, and footprint generation; visible-record projection still owns
  screen position, while the normal backend consumes the resulting sample
  payload through its GPU sample buffer and currentTexture path.
- computed footprint fields: `conic`, `covariance2D`, `radiusPx`, `depth`, and
  `sortKey` are `partial-webgpu-gaussian-footprint`; `aabb` and `tileRange` are
  derived from WebGPU projected positions plus computed radius before normal
  backend packing.
- deferred fields: full 4D covariance projection, anisotropic conic parity,
  GPU-native aabb/tileRange generation, depth-sort dispatch, compaction, and
  tile-list scatter remain future work and must not be reported as full
  footprint parity.
- success contract: Step82 requires Step81 computed `radiusPx/colorAlpha` to
  remain consumed, nonzero WebGPU footprint payloads to be generated, every
  normal-backend visible sample to consume computed conic/aabb/tileRange
  evidence, and currentTexture readback to remain ready under the
  WebGPU-exclusive guard.

Step82 fix1 clarifies the boundary between consumed partial payload and full
parity work. Normal backend `aabb` and `tileRange` consumption means
`partial-derived-from-webgpu-projected-px-py-and-computed-radius`: the values
are packed into the normal backend sample buffer and read back through WGSL,
but full GPU-native aabb/tileRange parity remains deferred until the dedicated
footprint/tile pipeline owns those calculations. The normal backend sample
readback validation now covers the full packed sample stride instead of the
older selected-sample prefix, so resolved Step40-era storage-buffer failure
wording should no longer appear as a Step82 failure.

Full realtime 4DGS rendering still requires full covariance/conic parity,
tile-list generation/scatter, depth sorting, complete SH/color evaluation,
streaming, chunking, LOD, and partial upload.

## Step83 WebGPU Tile-Aware Render Input Pipeline

Step83 moves the Step82 partial footprint payload into an explicit
tile-aware render input boundary. A WebGPU compute pass consumes visible-record
samples plus computed footprint fields and emits GPU tile records containing a
screen-space aabb, tile range, tile-reference count, depth key, sort key, and
source record metadata. A second tile-aware consumer pass reads those tile
records and produces a validation summary so Step83 cannot pass by only
attaching metadata to the normal backend sample path.

- selected approach: A/C. Step82's WebGPU-computed footprint remains the input,
  while Step83 adds GPU-generated tile records and an explicit tile-aware
  consumer. Full per-tile scatter, prefix/list ownership, depth-sort dispatch,
  and the final tile compositor remain deferred.
- computed tile payload: `gpu-native-aabb`, `gpu-native-tileRange`,
  `tile-record`, `depth-key`, `sort-key`, and `tile-reference-count` are
  generated by the Step83 WebGPU tile input pass.
- success contract: Step83 requires nonzero tile records, tile-aware consumer
  readback, Step82 computed footprint consumption, Step81 computed attribute
  consumption, and the existing WebGPU-exclusive currentTexture path to remain
  ready without WebGL2 hybrid rendering or fallback sample mixing.
- classification: Step83 is `partial-webgpu-tile-aware-render-input`, not full
  GPU tile-list/scatter/sort/compositor parity. The normal backend remains the
  bounded visible output path while tile-aware rendering ownership is staged
  behind a separate consumer boundary.

Full realtime 4DGS rendering still requires GPU tile-list scatter/prefix
ownership, depth sorting, final tile compositing, complete SH/color evaluation,
streaming, chunking, LOD, and partial upload.

## Step84 WebGPU GPU-Owned Tile List Layout Pipeline

Step84 turns the Step83 tile-aware records into a GPU-owned tile list layout
that later depth-sort and tile-compositor stages can consume without rebuilding
the list on the CPU. A WebGPU scatter pass writes splat references into a
fixed-capacity per-tile reference list, a table pass emits one `offset/count`
entry per tile, and a tile-list consumer pass follows those offsets to read the
reference list and produce the validation summary.

- selected approach: A/C. The tile list is generated and consumed on the GPU,
  while full parallel prefix-sum compaction, resize/second-pass overflow
  handling, depth-sort dispatch, and final tile compositing remain deferred.
- GPU-owned layout fields: `gpu-owned-offset-count-table`,
  `gpu-owned-splat-reference-list`, `reference-depth-key`,
  `reference-sort-key`, and a `fixed-capacity-tile-offset-layout`.
- success contract: Step84 requires a nonzero reference count, nonempty tiles,
  a GPU offset/count table, a GPU splat reference list, consumer readback, and
  proof that the consumer followed the table into the reference list.
- classification: Step84 is `partial-webgpu-gpu-owned-tile-list-layout`, not a
  full GPU tile-list pipeline. The reference list keeps depth/sort keys so the
  next stage can connect depth sort without changing the visible-record or
  normal-backend display path.

Full realtime 4DGS rendering still requires compacted prefix-owned tile lists,
overflow-resize policy, per-tile depth sorting, final tile compositing,
complete SH/color evaluation, streaming, chunking, LOD, and partial upload.

## Step85 WebGPU Tile-List Compositor Path

Step85 connects the Step84 GPU-owned tile list layout to a partial WebGPU tile
compositor. The compositor reads each tile's `offset/count` table entry,
traverses the GPU splat reference list, fetches the visible sample payload, and
writes a tile-space `rgba8unorm` output texture with a minimal alpha
accumulation. The existing currentTexture path remains active through the
guarded normal-backend presentation path while the compositor output texture is
validated as the next renderer-owned output boundary.

- selected approach: A/B/C. A dedicated compositor module consumes the
  GPU-owned tile list, performs partial accumulation, and writes an output
  texture. Full depth sorting, CUDA compositor parity, and the final production
  tile compositor remain deferred.
- compositor inputs: Step84 `offset/count` table, splat reference list,
  reference depth/sort keys, and the Step82/Step81 visible sample payload.
- success contract: Step85 requires nonzero composited references, proof that
  the compositor read the offset/count table and traversed the reference list,
  output texture write/readback evidence, and preservation of the Step84 tile
  list, Step83 tile input, Step82 footprint, Step81 attributes, and
  currentTexture path.
- tile count contract: `processedTileCount` is the full tile grid dispatched by
  the compositor. `compositedTileCount` and `nonEmptyCompositedTileCount` both
  describe tiles with at least one traversed reference, so they must agree and
  remain less than or equal to the processed tile count.
- classification: Step85 is `partial-webgpu-tile-list-compositor`, not a full
  WebGPU tile renderer. Ordering is `unsorted-fixed-reference-order` until the
  depth-sort stage owns per-tile ordering.

Full realtime 4DGS rendering still requires per-tile depth sorting, final tile
compositing against the viewer target, CUDA/compositor parity, compacted prefix
ownership for large tile lists, complete SH/color evaluation, streaming,
chunking, LOD, and partial upload.

## Step87 WebGPU Tile Depth Ordering for Compositor

Step87 advances the Step85 compositor from fixed reference traversal to
depth-aware traversal while preserving the Step84 GPU-owned tile list and the
Step86 backend boundary contract. The selected approach is B: the compositor
WGSL pass uses the depth/sort keys already stored in the Step84 reference list
and selects references in descending sort-key order inside each fixed-capacity
tile. This keeps the compositor connected to the GPU-owned `offset/count` table
and reference list without introducing a CPU-side sorted list.

- selected approach: B, with C-style validation fields. A separate full
  per-tile sort pass remains deferred until prefix/list compaction and the final
  compositor are ready.
- compositor ownership: WebGPU owns the order-aware compositor pass. The viewer
  shell still owns capture/query/canvas wiring, and Three.js/OrbitControls stay
  as camera/input adapters.
- success contract: Step87 requires `tileDepthOrderingReady`,
  `orderAwareCompositorUsed`, `depthKeyConsumed`, `sortKeyConsumed`,
  `compositorConsumedDepthOrderedReferences`, and matching
  `orderedReferenceCount` / `sourceReferenceCount`.
- preservation contract: Step87 keeps the Step84 GPU-owned tile list, Step85
  tile compositor/currentTexture path, and Step86 dirty/backend boundary
  contract intact. `dirtyTileList` and `dirtyCompositorInput` remain the
  invalidation entrypoints for this stage.
- classification: Step87 is a partial WebGPU depth-ordering boundary, not full
  CUDA depth parity and not a final production compositor. Full parallel
  per-tile sorting, CUDA compositor parity, final tile compositing, compacted
  prefix/list ownership, streaming, chunking, LOD, and partial upload remain
  deferred.

## Step88 WebGPU Tile-Compositor Frame Implementation

Step88 introduces `webgpu-tile-compositor-frame-implementation` as the first
frame implementation boundary where the WebGPU tile compositor owns the primary
one-frame pass chain. The selected approach is A: add a dedicated frame
implementation contract instead of stretching the older
`webgpu-normal-backend-frame-implementation` label over the tile-compositor
path. This keeps the normal backend available for fallback/regression while
making successful Step88 runs identify the tile compositor as the presentation
owner.

- frame implementation ownership: WebGPU owns the state, attribute, footprint,
  tile input, tile list, depth ordering, tile compositor, and presentation pass
  chain for the Step88 frame. The viewer shell still owns URL/query, capture,
  and canvas adapter wiring; Three.js/OrbitControls remain camera/input
  adapters, not rendering core.
- presentation boundary: the Step85 compositor output texture is rendered into
  the guarded viewer-canvas `currentTexture` path. Step88 records
  `compositorOutputPresentedToCurrentTexture`,
  `currentTextureConnectionReady`, and
  `currentTextureReadbackMatchesCompositorOutput` as the frame implementation
  presentation evidence. Step88 fix1 also records
  `presentationFrameCount`, `compositorPresentationFrameCount`,
  `currentTextureUsesWebGpuTileCompositorOutput`, `currentTextureSource`, and
  `presentationStableUntilCapture` as capture-end evidence. Step88 fix2 adds
  viewer-loop persistence evidence with
  `viewerLoopFrameImplementationActive`,
  `frameImplementationRegisteredWithViewerLoop`,
  `compositorOutputPresentedByViewerLoop`,
  `presentationPersistsAfterDelay`,
  `presentationPersistsAcrossAnimationFrames`, and overwrite guards so a
  capture-only transient presentation is not mistaken for a maintained viewer
  presentation path. Step88 fix3 also records
  `viewerLoopRuntimeFatalErrorDetected` and
  `viewerLoopRuntimeFatalError`; any RAF/runtime exception in the viewer-loop
  frame implementation blocks Step88 readiness instead of being hidden behind a
  later capture-only presentation. Step88 fix4 registers the tile compositor
  frame implementation as the scheduler's active continuous viewer-loop path
  while the guarded `webgpu-exclusive` tile-compositor mode is selected, and
  derives `viewerLoopFrameImplementationActive` from the scheduler RAF
  invocation plus the compositor currentTexture evidence instead of from the
  already-final `frameImplementationReady` flag. Step88 fix5 tightens that
  persistence contract from "presented across a few RAFs" to sampled visual
  stability. The tile compositor currentTexture presentation records
  `presentationSampleFrameCount`, `presentationNonBlankFrameCount`,
  `presentationBlankFrameCount`, `presentationAllSampledFramesNonBlank`,
  `presentationAlternatingBlankDetected`, `presentationStableVisualOutput`,
  nonzero pixel ratio bounds, frame hash changes,
  `compositorOutputPresentedEverySampledFrame`,
  `canvasClearBetweenCompositorFramesDetected`, and
  `viewerLoopPresentationCadenceStable`. Step88 success now requires sampled
  frames to remain nonblank with no alternating blank/clear frame between
  compositor presentations. Step88 fix6 separates compositor update from
  presentation heartbeat: dirty state may skip the expensive state/attribute/
  footprint/tile-list/depth/compositor update, but an active
  `webgpu-tile-compositor-frame-implementation` must still present the last
  valid compositor output texture on every viewer RAF. The cached-output
  heartbeat records `presentationHeartbeatReady`,
  `presentationHeartbeatRunsEveryViewerRaf`,
  `presentationDecoupledFromCompositorUpdate`,
  `lastValidCompositorOutputCached`,
  `lastValidCompositorOutputPresentedOnCleanFrames`,
  `dirtySkippedCompositorUpdateButPresentedCachedOutput`,
  `noBlankFrameBetweenHeartbeatPresentations`,
  `canvasVisibleOutputStableAcrossRaf`, and `visualFlickerDetected` so a
  present-only sample taken immediately after compositor update is not confused
  with stable viewer-loop presentation. Step88 fix7 hardens the WebGPU
  device/currentTexture lifecycle for that heartbeat: only the compositor output
  texture is cached, `context.getCurrentTexture()` and its `TextureView` are
  acquired fresh for each presentation, and the canvas context is refreshed for
  the compositor device before presentation so another backend path cannot leave
  the context associated with a stale device. The Summary records
  `webgpuDeviceConsistencyReady`,
  `presentationDeviceMatchesCompositorDevice`,
  `currentTextureViewFreshPerPresentation`,
  `crossDeviceTextureViewUseDetected`, and WebGPU validation / invalid command
  buffer / queue submit failure flags; any such failure blocks Step88 rather
  than being reported as stable presentation. Step88 fix8 adds final present
  source tracing for every sampled viewer RAF. Stable presentation is no longer
  inferred from nonblank pixels alone: the Summary records
  `finalPresentSourceTracingReady`, `finalPresentSourceSequence`,
  per-source final-present counts, `tileCompositorOwnsFinalPresentation`,
  `finalPresentSourceStable`, `finalPresentSourceAlternates`, and
  `summaryCanDetectObservedFlicker`. In tile-compositor mode, a clean frame may
  skip compositor update, but the final canvas presentation owner must still be
  the WebGPU tile compositor heartbeat using the last valid compositor output.
  Step88 fix9 separates startup transient evidence from steady-state evidence:
  a brief blank/clear/no-op before the first valid compositor output is tracked
  as `startupTransientObserved`, while readiness depends on
  `steadyStateSamplingReady`, `steadyStateSampledRafCount`,
  `steadyStateFinalPresentSourceSequence`,
  `steadyStateTileCompositorOwnsFinalPresentation`,
  `steadyStateVisualFlickerDetected`,
  `presentationPersistsAfterStartup`, and
  `presentationPersistsAcrossSteadyStateRaf`. A run with only one sampled RAF
  remains blocked because it cannot prove steady viewer-loop ownership. Step88
  fix10 moves the final-present tracing source to a viewer-loop RAF ring buffer
  instead of relying only on capture-command timing. The ring buffer keeps the
  recent final presentation owner history (`rafTraceRingBufferReady`,
  `rafTraceRecordedFromViewerLoopStart`, `rafTraceRingBufferFrameCount`) and
  records whether capture saw pre-command history
  (`rafTraceCapturedBeforeCommandStart`). Readiness requires
  `requiredSteadyStateRafCount >= 8`,
  `steadyStateSampledRafCount >= requiredSteadyStateRafCount`, and zero
  steady-state blank / clear / no-op / unknown / normal backend / WebGL2
  fallback final-present frames. Startup transients remain visible through
  `startupTransientFinalPresentSourceSequence`; they do not mask a later
  steady-state flicker.
- normal backend dependency: successful Step88 runs require
  `normalBackendPresentationUsed=false`,
  `normalBackendPresentationBypassed=true`, and
  `normalBackendDependencyReduced=true`. The older normal backend remains a
  fallback/regression path, but it is not the successful Step88 presentation
  source.
- preservation contract: Step88 requires Step85 tile compositor preservation,
  Step86 dirty/backend boundary preservation, and Step87 depth-ordering
  preservation before the tile-compositor frame implementation can report ready.
- classification: Step88 is a tile-compositor-owned WebGPU frame implementation
  boundary, not full CUDA compositor parity and not a completed production
  renderer. Full CUDA parity, final production compositor behavior, full
  parallel sorting, compacted prefix/list ownership, streaming, chunking, LOD,
  and partial upload remain deferred.

## Step86 Phase 3 WebGPU Backend Boundary and Dirty Contract Hardening

Step86 freezes the ownership boundaries added during Steps80-85 before the
backend moves into depth sorting, final compositing, chunking, LOD, or
streaming. It is not a new rendering feature step. Its success condition is that
the runtime capture can report the Phase 3 backend boundary and dirty update
entrypoints through common contract fields while preserving the Step85
tile-list compositor path.

- viewer shell: owns URL/query parsing, capture entrypoints, and camera/canvas
  adapter wiring. `viewer_app_gpu.js` should not accumulate new WebGPU pass
  construction logic; backend modules own state, attribute, footprint, tile
  input, tile list, and compositor passes.
- three adapter: Three.js and OrbitControls remain camera input, canvas
  integration, and debug-view adapters. They are not the rendering core and do
  not own WebGPU backend pass state.
- common contracts: shared record, projection, tile, compositor, dirty update,
  and capture summary contracts are the comparison boundary for WebGPU, WebGL2,
  and CUDA Reference. Backend-specific record formats should not be introduced
  outside these contracts.
- tools: `make_step_url.py`, `make_capture_commands.py`,
  `check_step_files.py`, and `summarize_step_json.py` own capture command
  generation, saved-result checking, step summary extraction, and contract
  validation reporting. They do not own runtime backend execution.
- WebGPU backend: owns state evaluation, Gaussian attribute evaluation,
  footprint evaluation, tile-aware input, GPU-owned tile lists, and the partial
  tile compositor. The Step86 boundary keeps `webgpu-exclusive` presentation
  separate from WebGL2 fallback/oracle rendering.
- WebGL2: remains fallback, validation, and regression oracle. It must not be
  mixed with WebGPU presentation in the same displayed frame.
- CUDA Reference: remains a fixed comparison reference and is not an
  interactive viewer backend.

The Step86 dirty update contract is an entrypoint contract rather than a full
incremental scheduler. It exposes:

- `dirtyCameraConstants`
- `dirtyTimeState`
- `dirtyVisibleRecords`
- `dirtyTileList`
- `dirtyCompositorInput`

Current runtime captures still rebuild the bounded pipeline for validation, but
these flags establish the dependency graph needed for future incremental frame
updates. `dirtyCameraConstants` and `dirtyTimeState` invalidate visible records;
visible-record changes invalidate tile input and tile lists; tile-list or
footprint changes invalidate compositor input. The contract also records that
`fullDatasetGpuResidencyRequired=false`, so future chunk/LOD/streaming work can
attach without turning the viewer into an all-Gaussian, all-frame residency
requirement.

Step86 keeps the Step85 runtime path intact and reports that preservation
directly in the Step86 summary. The preserved signals include the Step85 tile
compositor path, the `currentTexture` path-maintained flag, and the compositor
output readback match used as the Step85 currentTexture preservation evidence.
Partial WebGPU tile-list compositor output remains the current renderer-owned
output boundary, while full depth sort, CUDA compositor parity, final production
compositor, compacted prefix/list ownership, streaming, chunking, LOD, and
partial upload remain deferred.
