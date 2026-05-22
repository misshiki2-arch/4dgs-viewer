# Phase 2 WebGL2 Closure Summary

Date: 2026-05-21

This document closes Phase 2 of the 4DGS GPU viewer work. Phase 2 tested how far
the current WebGL2 path can move toward a realtime 4DGS viewer without forcing
sort, compaction, variable packing, tile-list generation, or tile composite
rewrites into an awkward WebGL2 shape.

## Scope

Phase 2 focused on the WebGL2 viewer path under `demo/4dgs_gpu_viewer.html` and
the saved validation results in `/home/demo/work/json`.

It did not implement WebGPU, GPU depth sort, GPU compaction, GPU tile-list
generation, tile composite rewrites, compression, LOD, streaming, or
Group-of-Frames.

## Evidence Used

Key commits:

- `9344c02` Add Step110 validated screenCoarse promotion
- `0fd3feb` Add Step111 validated candidate timing summary
- `0e6ad93` Add Step114 GPU visible record dry-run comparison
- `5e9f4eb` Add Step116 raw attribute texture visible record dry-run
- `e7512e9` Add Step118A richer raw visible record dry-run
- `115bdfb` Add Step120A packed-like raw visible record dry-run
- `9c463aa` Add Step122A display connection readiness summary
- `b83ff6e` Add Step124 CPU post-candidate timing breakdown
- `2715388` Add Step125 multi-view CPU post-candidate breakdown analysis

Saved result prefixes:

- `step120a_000151_v13`
- `step122a_000151_v13`
- `step124_000151_v13`
- `step125a_000151_v13`
- `step125a_000056_v08`
- `step125a_000083_v16`
- `step125a_000195_v26`
- `step125a_000198_v03`

Useful tools:

- `tools/summarize_step_json.py`
- `tools/check_step_files.py`
- `tools/analyze_cpu_post_candidate_breakdown.py`
- `tools/make_step_url.py`
- `tools/make_capture_commands.py`

## Achieved In Phase 2

The WebGL2 path successfully reached:

- GPU `screenCoarse` candidate generation.
- `validated-only` promotion to `displayCandidateSource=gpu-candidate`.
- CPU fallback preservation.
- Raw 4DGS attribute upload into WebGL2 textures.
- Vertex shader `texelFetch` from raw attribute textures.
- Transform Feedback fixed-record output.
- Minimal visible fixed-record comparison for `srcIndex`, `valid`, `px`, `py`,
  `depth`, and `aabb`.
- Richer fixed-record comparison for `radius`, `conic`, `alpha`, and
  `tileRange`.
- Packed-like fixed-record dry-run close to packed layout v2.
- Known AABB/tileRange integer-boundary rounding differences classified instead
  of hidden.
- Display connection readiness reporting.
- CPU post-candidate timing breakdown.
- Multi-view CPU post-candidate breakdown comparison.

The important result is that WebGL2 naturally supports fixed-size record
generation and validation. It does not naturally close the full display-input
contract.

## WebGL2 Natural Boundary

The following are natural fits for WebGL2 in this codebase:

- Fixed-size candidate buffers.
- Fixed-size attribute textures.
- Vertex shader raw fetch.
- Transform Feedback output.
- Debug readback for validation.
- Fixed-record field-by-field comparison.
- Validated-only promotion while CPU fallback remains available.

These are useful enough to keep as a WebGL2 validation and fallback path.

## WebGL2 Stop Line

Phase 2 should not push the following further in WebGL2:

- Depth sort.
- Variable packing and compaction.
- Prefix sum offsets.
- Tile-list GPU generation.
- Tile reference scatter.
- Sorted-visible-order display connection.
- Tile composite input contract rewiring.
- `colorAlpha.rgb` and SH parity in the WebGL2 fixed-record display path.

These can be approximated with WebGL2 multi-pass texture or Transform Feedback
pipelines, but doing so would increase readback pressure, state complexity, and
debug cost. That would move away from the final realtime 4DGS viewer goal.

## CPU Post-Candidate Breakdown Conclusion

Step125A measured five camera/view conditions. The dominant stage was
`visible-loop` in all five.

| Prefix | Total ms | Visible loop ms | Sort ms | Packed ms | Tile-list ms | Candidate | Visible | Tile refs | Dominant |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `step125a_000151_v13` | 453.700 | 424.525 | 9.740 | 12.610 | 6.585 | 40319 | 27721 | 82849 | visible-loop |
| `step125a_000056_v08` | 212.170 | 188.425 | 6.875 | 12.140 | 4.520 | 27937 | 21234 | 58628 | visible-loop |
| `step125a_000083_v16` | 426.490 | 383.795 | 10.340 | 22.255 | 9.880 | 42503 | 29676 | 94647 | visible-loop |
| `step125a_000195_v26` | 539.975 | 392.080 | 20.070 | 118.310 | 9.420 | 35625 | 24999 | 75936 | visible-loop |
| `step125a_000198_v03` | 451.505 | 389.400 | 15.500 | 40.220 | 6.140 | 30372 | 21167 | 67348 | visible-loop |

The visible-loop share ranged from about `0.726` to `0.936`. Additional
`screenCoarseMaxCount` scaling could refine slopes, but it is not required to
close Phase 2.

## WebGPU Migration Rationale

WebGPU is the right next target because the remaining work is compute-shaped:

- Storage-buffer visible records.
- GPU visible loop over candidates.
- Compute prefix sum and compaction.
- GPU depth sort or sort-compatible approximation.
- GPU tile binning.
- Compute-driven tile-list generation.
- Cleaner separation between packed records and display input contracts.

The WebGL2 work already proved the raw attribute interpretation, projection,
validity, conic/alpha/tileRange subset, and packed-like layout. Phase 3 should
reuse those contracts and reference comparisons rather than rediscovering them.

## WebGL2 Stable Mode Preset

The WebGL2 stable/fallback preset should avoid validation readbacks and heavy
debug captures.

Generate it with:

```bash
python3 tools/make_step_url.py --preset stable --camera-name 000151_v13
```

Recommended URL conditions:

```text
drawPath=gpu-screen
tileCompositePath=accumulation
tileCompositePrimitive=quad
datasetViewMatrixMode=threejs
gpuCandidateRuntime=limited-draw
gpuCandidateSourceMode=screenCoarse
gpuCandidateScreenCoarseMaxCount=65536
gpuCandidateScreenCoarseMinRadiusPx=0.25
gpuCandidateScreenCoarseRequireInViewport=true
gpuCandidateScreenCoarseDepthMode=positive
gpuCandidatePromotePolicy=validated-only
gpuCandidateAllowReadbackInDraw=false
gpuCandidateCoverageCompare=false
gpuCandidateCompare=false
gpuRawVisibleRecordDryRun=false
debugPreserveDrawingBuffer=0
inspectJsonMode=slim
useNativeRot4d=true
useNativeMarginal=true
```

Stable mode should keep CPU fallback available. If the runtime cannot safely use
GPU candidates without readback in a given browser/session, it should fall back
instead of silently entering a debug validation path.

Stable mode is not a CUDA reference-capture mode. Step131 showed that the
reference image match is established by the `cuda-aligned` projection/sign
contract, while the stable viewer uses the normal Three.js interactive camera
path. As a result, stable mode prioritizes natural controls, fallback behavior,
and public/outreach usability over exact CUDA reference image orientation.
Projection and screen-axis contract differences should be carried into Phase 3's
common camera/control design rather than solved by trial-and-error `cameraUp`
changes in WebGL2.

## Stable Mode Disabled Items

Disable these by default:

- Sync debug readback in draw.
- Candidate compare.
- Coverage compare.
- Visible/packed compare.
- Raw visible record dry-run.
- Packed-like dry-run.
- ScreenCoarse sweep capture.
- Canvas capture.
- Association/live same-state capture.
- `debugPreserveDrawingBuffer`.
- Large debug JSON capture.

If a stable session needs a one-off screenshot, open it with
`debugPreserveDrawingBuffer=true` or use a capture path that renders immediately
before reading the canvas. With the default stable URL
`debugPreserveDrawingBuffer=false`, a saved PNG may be black even when the page
is visibly rendered.

## Fixed Reference Mode

Fixed reference mode is the Phase 2 CUDA comparison path. Generate it by using
the validation preset with an explicit CUDA-aligned view:

```bash
python3 tools/make_step_url.py \
  --preset validation \
  --camera-name 000151_v13 \
  --dataset-view-matrix-mode cuda-aligned
```

This mode keeps `datasetTransformMatrix`, intrinsics, and the CUDA-aligned
projection/sign contract together. It is appropriate for PNG/JSON evidence and
reference comparison, but it is not the public interactive camera contract.

## Validation / Debug Mode Preset

Validation mode should keep the full Phase 2 evidence path available.

Generate the matching URL with:

```bash
python3 tools/make_step_url.py --preset validation --camera-name 000151_v13
```

Generate the matching capture commands with:

```bash
python3 tools/make_capture_commands.py \
  --preset validation \
  --step step_validation_000151_v13 \
  --source-mode screenCoarse
```

Recommended URL conditions:

```text
debugPreserveDrawingBuffer=1
drawPath=gpu-screen
tileCompositePath=accumulation
tileCompositePrimitive=quad
datasetViewMatrixMode=threejs
inspectSource=actual-draw
inspectJsonMode=slim
gpuCandidateRuntime=limited-draw
gpuCandidateSourceMode=screenCoarse
gpuCandidateScreenCoarseMaxCount=65536
gpuCandidateScreenCoarseMinRadiusPx=0.25
gpuCandidateScreenCoarseRequireInViewport=true
gpuCandidateScreenCoarseDepthMode=positive
gpuCandidateReadbackMode=sync-debug
gpuCandidateAllowReadbackInDraw=true
gpuCandidatePromotePolicy=validated-only
gpuCandidateCoverageCompare=true
gpuCandidateCompare=true
gpuRawVisibleRecordDryRun=true
gpuRawVisibleRecordMode=packed-like
gpuRawVisibleRecordFields=srcIndex,valid,px,py,depth,aabb,radius,conic,alpha,tileRange
gpuRawAttributeTexture=true
gpuRawVisibleRecordReadback=sync-debug
useNativeRot4d=true
useNativeMarginal=true
```

Validation mode should preserve:

- `check_step_files.py`
- `summarize_step_json.py`
- `analyze_cpu_post_candidate_breakdown.py`
- `make_step_url.py`
- `make_capture_commands.py`
- Candidate source compare.
- Coverage compare.
- Dry-run visible/packed compare.
- Raw visible record dry-run.
- Packed-like dry-run.
- Display connection readiness.
- CPU post-candidate breakdown.
- Multi-view breakdown aggregation.

## Tool Preset Override Rules

`tools/make_step_url.py` and `tools/make_capture_commands.py` both support
presets as defaults. Explicit CLI arguments override the preset values. This
allows a stable URL to be generated and then selectively adjusted without adding
viewer-side preset state.

The stable and validation presets use dataset camera metadata as the initial
viewer camera only. With `--camera-name` or `--camera-meta-json`, they keep
`datasetTransformMatrix` in the URL but set
`cameraControlContract=interactive-from-reference` and
`cameraOrientationPolicy=roll-free-reference-screen-up`. The viewer then derives
an OrbitControls-friendly interactive camera from the reference camera contract,
instead of having the tools guess `cameraPosition`, `cameraTarget`, and
`cameraUp`. They also set `datasetViewMatrixMode=threejs` so mouse interaction
continues to drive the rendered camera. If a fixed CUDA reference view is needed
for a specific validation capture, pass `--dataset-view-matrix-mode cuda-aligned`
explicitly.

Useful commands:

```bash
# Stable/fallback URL with heavy debug features off.
python3 tools/make_step_url.py --preset stable --camera-name 000151_v13

# Validation URL with Phase 2 compare/readback/dry-run enabled.
python3 tools/make_step_url.py --preset validation --camera-name 000151_v13

# No capture commands for stable mode.
python3 tools/make_capture_commands.py --preset stable --step stable_000151_v13

# Runtime-only lightweight capture for quick sanity checks.
python3 tools/make_capture_commands.py --preset runtime-only --step runtime_000151_v13

# Full validation capture.
python3 tools/make_capture_commands.py --preset validation --step validation_000151_v13
```

## Phase 2 Completion Criteria

Phase 2 can be considered complete when:

- This closure summary is committed.
- A Phase 2 tag is created.
- The saved Step120A/122A/124/125 results are kept available.
- Stable and validation URL presets are documented.
- Phase 3 branch starts from this stable point.

No further WebGL2 GPU implementation is required for Phase 2 completion.

## Tag And Branch Candidates

Suggested tag:

```text
phase2-webgl2-fixed-record-validation
```

Suggested Phase 3 branch:

```text
phase3-webgpu-compute-prototype
```

Optional stable branch:

```text
webgl2-stable-fallback
```

## Phase 3 Separation Plan

Phase 3 should keep shared math/reference contracts separate from backend code.

Recommended module boundaries:

- `common`: camera/query state, 4D Gaussian math contracts, CPU reference
  comparison helpers, packed record schema, debug JSON schema.
- `webgl2`: current fallback renderer, screenCoarse candidate path,
  Transform Feedback validation tools, stable/validation presets.
- `webgpu`: compute visible loop, storage-buffer visible records, prefix
  sum/compaction, sort/binning experiments, tile-list generation prototype.

First WebGPU task:

- Reproduce the Phase 2 fixed-record contract in WebGPU storage buffers.
- Compare against the same CPU reference subset.
- Do not connect to display until compute visible records, compaction, ordering,
  and tile-list contracts are validated.

The WebGL2 path should remain a fallback and regression oracle during this work.
