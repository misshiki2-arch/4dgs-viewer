# Phase 3 Backend Responsibility Plan

Date: 2026-05-22

This note turns the Phase 3 backend split into file-level responsibilities. It
does not move code or add WebGPU math. The goal is to keep WebGL2 useful as a
fallback and validation oracle while the normal realtime backend moves toward a
standalone WebGPU pipeline.

## Goals

- Make WebGPU the long-term normal rendering backend.
- Keep WebGL2 as fallback, regression oracle, and validation path.
- Treat Three.js as the UI, camera-input, and OrbitControls adapter, not as the
  owner of 4DGS projection or record semantics.
- Move shared 4DGS contracts into small common modules before moving behavior.
- Keep CUDA-aligned fixed reference capture separate from interactive viewer
  controls.

## Non-goals

- No WebGL2/WebGPU hybrid rendering within one frame.
- No display connection in this step.
- No new sort, compaction, tile-list, or tile composite implementation.
- No camera-up or axis trial-and-error.
- No large directory split before contracts are stable.

## Current File Responsibilities

### `demo/js/viewer_app_gpu.js`

Current role:

- Main application orchestrator for loading data, applying query state, running
  WebGL2 rendering, debug capture, validation summaries, and WebGPU dry-run
  hooks.
- Owns too many cross-cutting decisions today: runtime mode selection, capture
  assembly, candidate source lookup, and validation JSON packaging.

Target role:

- Viewer shell and adapter wiring.
- Choose backend mode from URL/runtime state.
- Pass normalized scene, camera, and debug options into backend modules.
- Collect backend summaries without owning backend math.

Do not move yet:

- Capture plumbing and saved JSON assembly. It is still the lowest-risk place to
  preserve existing validation output while contracts are being separated.

### `demo/js/viewer_scene_setup.js`

Current role:

- Creates Three.js camera and OrbitControls.
- Sets canvas size and creates the existing WebGL2 GPU renderer.
- Applies viewer camera presets.

Target role:

- Three.js adapter only: canvas, camera input, OrbitControls, resize, and UI
  camera preset application.
- Should not own 4DGS projection, visible-record, candidate, or tile-list
  contracts.

### `demo/js/viewer_query_state.js`

Current role:

- Parses URL state and emits deterministic URL parameters.
- Carries WebGL2 validation flags, WebGPU dry-run flags, camera contract flags,
  capture/debug options, and preset-derived state.

Target role:

- URL/state adapter.
- Keep parsing explicit, but route stable contract names through common modules
  once those modules exist.
- Avoid becoming the source of truth for backend semantics.

### `demo/js/common_4dgs_record_contracts.js`

Current role:

- First common contract module.
- Defines the WebGPU visible fixed-record schema, field list, field compute
  modes, projection contract names, and deferred field names.

Target role:

- Source of truth for shared record names and schema labels.
- Expand only with small, stable contracts that are consumed by both runtime and
  tools.

Next likely additions:

- Projection contract helpers and metadata names.
- Record comparison field layout and mismatch classification names.
- Later, packed layout references if WebGPU and WebGL2 summaries need a common
  packed contract vocabulary.

### `demo/js/webgpu_visible_record_dry_run_runtime.js`

Current role:

- WebGPU fixed-record dry-run runtime.
- Initializes adapter/device, builds CPU reference records, uploads storage
  buffers, dispatches WGSL, reads back output, compares fields, and writes
  WebGPU debug JSON.
- Already uses `common_4dgs_record_contracts.js` for field and contract names.

Target role:

- Temporary WebGPU validation backend for fixed-record compute.
- Eventually split into:
  - WebGPU device/buffer/dispatch utilities.
  - WebGPU visible-record compute pass.
  - Common projection/record comparison helpers.

Do not move yet:

- WGSL shader string and readback path. They are still changing quickly and are
  easier to validate in one local module.

### `demo/js/gpu_visible_item_builder.js`

Current role:

- CPU reference visible-item builder.
- Uses 4D state math, screen splat projection, SH color, AABB, radius, conic,
  alpha, and tile range helpers.

Target role:

- CPU reference oracle for WebGL2 and WebGPU validation.
- Candidate for common/reference area later, but not for a broad move yet.

Why not move now:

- It pulls together projection, color, tile, and state math. Moving it before
  the projection contract is explicit would make the split look cleaner than it
  really is.

### `demo/js/gpu_buffer_layout_utils.js`

Current role:

- Packed visible layout v2 field metadata and offset helpers.

Target role:

- Common packed-record contract candidate.
- Can remain where it is until WebGPU starts producing packed or packed-like
  records beyond dry-run validation.

### `demo/js/gpu_visible_record_raw_dry_run_runtime.js`

Current role:

- WebGL2 raw attribute texture plus Transform Feedback validation runtime.
- Owns raw fixed-record and packed-like dry-run comparison, known boundary
  mismatch classification, and display connection readiness.

Target role:

- WebGL2 validation oracle.
- Should keep proving the WebGL2 fixed-record path without being turned into a
  production backend for sort, compaction, or tile-list generation.

Future cleanup:

- Read common field names once the common contracts are stable.
- Keep WebGL2-specific GL setup, shaders, texture upload, and TF readback local.

### `demo/js/gpu_raw_attribute_texture_runtime.js`

Current role:

- WebGL2 raw 4DGS attribute texture creation and summary.

Target role:

- WebGL2 validation infrastructure.
- Do not force it into WebGPU. WebGPU should use storage-buffer upload contracts
  instead.

### `tools/make_step_url.py`

Current role:

- Deterministic URL generation for stable, validation, fixed-reference, and
  WebGPU dry-run URLs.

Target role:

- Tool-side preset adapter.
- Should generate explicit URL state, not encode hidden renderer behavior.

### `tools/make_capture_commands.py`

Current role:

- Capture command generation for validation/debug/runtime-only flows.

Target role:

- Tool-side capture adapter.
- Keep stable PNG capture opt-in because stable mode does not preserve the
  drawing buffer by default.

### `tools/summarize_step_json.py`

Current role:

- Reads saved JSON summaries and reports WebGL2/WebGPU validation status.

Target role:

- Summary reader for stable contract names.
- Should tolerate old JSON while recognizing common contract labels.

### `docs/phase2_webgl2_closure_summary.md`

Current role:

- Phase 2 closure record: WebGL2 boundary, stable/validation/fixed roles, and
  WebGPU migration rationale.

Target role:

- Historical boundary document.
- New Phase 3 backend split details should live in this document, not overwrite
  the Phase 2 closure.

## Responsibility Split

### Three.js adapter

Owns:

- Canvas setup and resize.
- User camera input and OrbitControls.
- UI-facing camera presets.
- Passing an interactive camera state to a backend.

Does not own:

- 4DGS record layout.
- CUDA-aligned projection semantics.
- WebGPU storage-buffer layout.
- Tile-list generation.
- Final WebGPU rendering math.

### Common contracts

Owns:

- Fixed-record field names, offsets, and schema labels.
- Projection contract names and metadata shape.
- Comparison output names and mismatch classification vocabulary.
- Eventually shared CPU reference helpers that are pure and backend-neutral.

Does not own:

- GPU API resource lifetime.
- Browser capture plumbing.
- Three.js controls.
- Backend-specific shader source.

### WebGPU backend

Owns:

- Adapter/device feature checks.
- Storage-buffer upload.
- WGSL visible-record compute.
- Future state/covariance/radius/conic/alpha compute.
- Future compaction, prefix sum, sort, tile binning, and tile-list generation.
- Future WebGPU-only display path.

Does not own:

- WebGL2 fallback.
- OrbitControls behavior.
- CUDA reference PNG capture policy.

### WebGL2 backend and validation oracle

Owns:

- Current stable/fallback WebGL2 viewer.
- WebGL2 screenCoarse candidate path.
- Raw attribute texture and Transform Feedback validation.
- Packed-like dry-run validation.
- Regression evidence against CPU reference and saved PNG/JSON captures.

Does not own:

- New production sort, compaction, tile-list generation, or WebGPU backend work.

## Camera And Projection Contract

Keep two contracts separate:

- Fixed reference contract: CUDA-aligned projection/sign/intrinsics for evidence
  capture and regression comparison.
- Interactive contract: Three.js/OrbitControls camera input for stable public
  viewer behavior.

Phase 3 should make the projection metadata explicit in common modules before
trying to make the interactive camera visually identical to fixed reference
captures. The unresolved screen-axis and projection differences from Phase 2
should not be patched by changing `cameraUp` guesses.

## Migration Order

1. Keep WebGL2 validation stable and unchanged while WebGPU dry-runs evolve.
2. Expand common contracts in small pieces:
   - fixed-record fields already exist;
   - projection contract helper should come next;
   - comparison schema/tolerance helpers after that.
3. Keep WebGPU fixed-record dry-run as the first compute backend.
4. Move one compute responsibility at a time into WGSL:
   - state position upload contract;
   - valid/projection gates;
   - radius/conic/alpha;
   - fixed packed-like record.
5. Add WebGPU-only compaction/prefix sum after fixed records are trustworthy.
6. Add WebGPU sort or sort-compatible ordering.
7. Add WebGPU tile binning and tile-list generation.
8. Only then design the WebGPU display connection and tile composite contract.

## Step 8 Recommendation

The next minimal implementation should extract the WebGPU projection contract
builder into a small common module, for example:

- `demo/js/common_4dgs_projection_contracts.js`

Scope:

- Move or wrap `matrixToRows4`, `flattenRows4`, and the metadata constants needed
  to describe `threejs-view-projection-ndc` and
  `cuda-plus-z-forward-fx-fy-cx-cy`.
- Keep behavior identical.
- Keep WGSL unchanged.
- Keep WebGL2 raw visible dry-run unchanged except for optional future imports of
  stable names.
- Update `summarize_step_json.py` only if it needs to recognize a new schema
  label.

Why this is the right next unit:

- Step4 made projection the first real WebGPU math contract.
- Projection is shared by WebGPU compute, CPU reference comparison, fixed
  reference capture, and future WebGPU display.
- It is smaller and safer than moving the full CPU visible-item builder.

## Later Steps

Step 9 candidates:

- Add a common record comparison schema helper.
- Add WebGPU raw/storage-buffer input contract metadata.

Step 10+ candidates:

- Move more minimal record fields into WGSL.
- Add richer fixed-record fields.
- Design compaction and prefix-sum contracts.

Keep deferred:

- Display connection.
- Tile composite rewrites.
- GPU sort and tile binning implementation.
- Large directory split into `common/`, `webgl2/`, and `webgpu/`.
- Camera/control visual parity work beyond explicit contract metadata.
