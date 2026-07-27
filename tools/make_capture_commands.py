#!/usr/bin/env python3
"""
make_capture_commands.py

Generate DevTools Console commands for 4DGS Viewer Step captures.

Typical use:
  python3 tools/make_capture_commands.py \
    --step step107_000151_v13 \
    --source-mode screenCoarse

Range example:
  python3 tools/make_capture_commands.py \
    --step step105_000151_v13 \
    --source-mode range

Output:
  JavaScript commands to paste into Chrome DevTools Console.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List


STEP114_CAPTURE_CONTRACTS = (
    (
        "step114_fix10_fix6_fix1",
        "phase3-step114-fix10-fix6-fix1",
        "phase3-step114-fix10-fix6-fix1-production-webgpu-runtime-selection-correction",
    ),
    (
        "step114_fix10_fix6",
        "phase3-step114-fix10-fix6",
        "phase3-step114-fix10-fix6-production-canvas-writer-ownership-correction",
    ),
    (
        "step114_fix10_fix5",
        "phase3-step114-fix10-fix5",
        "phase3-step114-fix10-fix5-coalesced-scheduler-request-drain",
    ),
    (
        "step114_fix10_fix4",
        "phase3-step114-fix10-fix4",
        "phase3-step114-fix10-fix4-synchronous-command-boundary-and-initial-render-trigger",
    ),
    (
        "step114_fix10_fix3",
        "phase3-step114-fix10-fix3",
        "phase3-step114-fix10-fix3-final-browser-presentation-evidence",
    ),
    (
        "step114_fix10_fix2",
        "phase3-step114-fix10-fix2",
        "phase3-step114-fix10-fix2-browser-visible-presentation-and-saved-png-evidence-correction",
    ),
    (
        "step114_fix10_fix1",
        "phase3-step114-fix10-fix1",
        "phase3-step114-fix10-fix1-initial-production-presentation-observation-correction",
    ),
    (
        "step114_fix10",
        "phase3-step114-fix10",
        "phase3-step114-fix10-canonical-projection-rotation-contract-initial-presentation",
    ),
    (
        "step114_fix9",
        "phase3-step114-fix9",
        "phase3-step114-fix9-production-temporal-motion-delta-direct-parity",
    ),
    (
        "step114_fix8",
        "phase3-step114-fix8",
        "phase3-step114-fix8-fixed-time-capture-isolation-temporal-revalidation",
    ),
    (
        "step114_fix7",
        "phase3-step114-fix7",
        "phase3-step114-fix7-temporal-deformation-lineage-first-mismatch-closure",
    ),
    (
        "step114_fix6",
        "phase3-step114-fix6",
        "phase3-step114-fix6-gaussian-index-lineage-full-pre-cull-stage-evidence",
    ),
    (
        "step114_fix5",
        "phase3-step114-fix5",
        "phase3-step114-fix5-pre-cull-evidence-referenceerror-closure",
    ),
    (
        "step114_fix4",
        "phase3-step114-fix4",
        "phase3-step114-fix4-canonical-index-lineage-pre-cull-production-evidence",
    ),
    (
        "step114_fix3",
        "phase3-step114-fix3",
        "phase3-step114-fix3-common-gaussian-direct-comparison",
    ),
    (
        "step114_fix2",
        "phase3-step114-fix2",
        "phase3-step114-fix2-direct-cuda-rasterizer-render-state-audit",
    ),
)


def resolve_step114_capture_contract(step: str) -> tuple[str, str] | None:
    if "step114" not in step:
        return None
    for label, phase_step, comparison_mode in STEP114_CAPTURE_CONTRACTS:
        if label in step:
            return phase_step, comparison_mode
    return (
        "phase3-step114",
        "phase3-step114-cuda-reference-provenance-render-state-audit",
    )


def uses_fix10_fix4_causal_evidence(step: str) -> bool:
    contract = resolve_step114_capture_contract(step)
    return contract is not None and contract[0] in {
        "phase3-step114-fix10-fix4",
        "phase3-step114-fix10-fix5",
        "phase3-step114-fix10-fix6",
        "phase3-step114-fix10-fix6-fix1",
    }


def js_bool(value: str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return "true"
    if text in {"0", "false", "no", "n", "off"}:
        return "false"

    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def js_number_array(value: str) -> str:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    numbers = [str(float(item)).rstrip("0").rstrip(".") for item in items]
    return "[" + ", ".join(numbers) + "]"


def js_string_array(value: str) -> str:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return "[" + ", ".join(quote(item) for item in items) + "]"


def js_bool_array(value: str) -> str:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return "[" + ", ".join(js_bool(item) for item in items) + "]"


def parse_int_list(value: str) -> List[int]:
    if not str(value).strip():
        return []
    out: List[int] = []
    seen = set()
    for item in str(value).split(","):
        text = item.strip()
        if not text:
            continue
        index = int(text)
        if index in seen:
            continue
        seen.add(index)
        out.append(index)
    return out


def load_cuda_manifest_selected_indices(path: str | None) -> List[int]:
    if not path:
        return []
    manifest_path = Path(path)
    if not manifest_path.exists():
        return []
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence = (
        data.get("imageSpaceConvention", {})
        .get("directRasterizerScreenCoordinateEvidence", {})
    )
    selected = evidence.get("selectionPolicy", {}).get("selectedIndices")
    if not isinstance(selected, list):
        selected = evidence.get("selectedIndices")
    if not isinstance(selected, list):
        records = evidence.get("records")
        selected = [
            item.get("srcIndex")
            for item in records
            if isinstance(item, dict) and item.get("srcIndex") is not None
        ] if isinstance(records, list) else []
    return parse_int_list(",".join(str(item) for item in selected))


def js_int_array(values: List[int]) -> str:
    return "[" + ", ".join(str(int(value)) for value in values) + "]"


def build_source_compare_command(args: argparse.Namespace) -> str:
    step = args.step

    if args.source_mode == "screenCoarse":
        return f"""await window.gpuViewerDebug.downloadJsonDebug(
  await window.gpuViewerDebug.captureGpuCandidateScreenCoarseCompareDebug({{
    ensureCurrentFrame: false,
    maxCount: {args.screen_coarse_max_count},
    minRadiusPx: {args.screen_coarse_min_radius_px},
    requireInViewport: {js_bool(args.screen_coarse_require_in_viewport)},
    depthMode: {quote(args.screen_coarse_depth_mode)},
    maxMismatches: {args.max_mismatches}
  }}),
  {quote(step + '_gpu_candidate_screen_coarse_compare.json')}
);"""

    if args.source_mode == "range":
        return f"""await window.gpuViewerDebug.downloadJsonDebug(
  await window.gpuViewerDebug.captureGpuCandidateSourceCompareDebug({{
    ensureCurrentFrame: false,
    sourceMode: 'range',
    rangeStart: {args.range_start},
    rangeCount: {args.range_count},
    readbackMode: {quote(args.readback_mode)},
    maxMismatches: {args.max_mismatches}
  }}),
  {quote(step + '_gpu_candidate_source_compare.json')}
);"""

    if args.source_mode == "visibleSrcIndices":
        return f"""await window.gpuViewerDebug.downloadJsonDebug(
  await window.gpuViewerDebug.captureGpuCandidateDryRunVisibleComparisonDebug({{
    subsetMode: 'visibleSrcIndices',
    subsetCount: {args.subset_count},
    filterMode: {quote(args.filter_mode)},
    epsilon: {args.epsilon},
    maxMismatches: {args.max_mismatches}
  }}),
  {quote(step + '_gpu_candidate_dryrun_visible_compare.json')}
);"""

    raise ValueError(f"Unsupported source mode: {args.source_mode}")


def build_coverage_command(args: argparse.Namespace) -> str:
    if args.source_mode == "screenCoarse":
        return f"""await window.gpuViewerDebug.downloadJsonDebug(
  await window.gpuViewerDebug.captureGpuCandidateCoverageDebug({{
    ensureCurrentFrame: false,
    sourceMode: 'screenCoarse',
    maxCount: {args.screen_coarse_max_count},
    minRadiusPx: {args.screen_coarse_min_radius_px},
    requireInViewport: {js_bool(args.screen_coarse_require_in_viewport)},
    depthMode: {quote(args.screen_coarse_depth_mode)},
    maxMisses: {args.max_misses}
  }}),
  {quote(args.step + '_gpu_candidate_coverage.json')}
);"""

    if args.source_mode == "range":
        return f"""await window.gpuViewerDebug.downloadJsonDebug(
  await window.gpuViewerDebug.captureGpuCandidateCoverageDebug({{
    ensureCurrentFrame: false,
    sourceMode: 'range',
    rangeStart: {args.range_start},
    rangeCount: {args.range_count},
    maxMisses: {args.max_misses}
  }}),
  {quote(args.step + '_gpu_candidate_coverage.json')}
);"""

    if args.source_mode == "visibleSrcIndices":
        return f"""await window.gpuViewerDebug.downloadJsonDebug(
  await window.gpuViewerDebug.captureGpuCandidateCoverageDebug({{
    ensureCurrentFrame: false,
    sourceMode: 'visibleSrcIndices',
    subsetCount: {args.subset_count},
    filterMode: {quote(args.filter_mode)},
    maxMisses: {args.max_misses}
  }}),
  {quote(args.step + '_gpu_candidate_coverage.json')}
);"""

    raise ValueError(f"Unsupported source mode: {args.source_mode}")


def build_dryrun_visible_compare_command(args: argparse.Namespace) -> str:
    if args.source_mode == "screenCoarse":
        return f"""await window.gpuViewerDebug.downloadJsonDebug(
  await window.gpuViewerDebug.captureGpuCandidateScreenCoarseDryRunVisibleComparisonDebug({{
    ensureCurrentFrame: false,
    readbackMode: {quote(args.readback_mode)},
    maxCount: {args.screen_coarse_max_count},
    minRadiusPx: {args.screen_coarse_min_radius_px},
    requireInViewport: {js_bool(args.screen_coarse_require_in_viewport)},
    depthMode: {quote(args.screen_coarse_depth_mode)},
    filterMode: {quote(args.filter_mode)},
    epsilon: {args.epsilon},
    maxMismatches: {args.max_mismatches},
    maxMisses: {args.max_misses}
  }}),
  {quote(args.step + '_gpu_candidate_screen_coarse_dryrun_visible_compare.json')}
);"""

    return ""


def build_sweep_command(args: argparse.Namespace) -> str:
    if args.source_mode != "screenCoarse":
        return ""

    return f"""await window.gpuViewerDebug.downloadJsonDebug(
  await window.gpuViewerDebug.captureGpuCandidateScreenCoarseSweepComparisonDebug({{
    ensureCurrentFrame: false,
    readbackMode: {quote(args.readback_mode)},
    maxCounts: {js_number_array(args.sweep_max_counts)},
    minRadiusPxValues: {js_number_array(args.sweep_min_radius_px)},
    requireInViewportValues: {js_bool_array(args.sweep_require_in_viewport)},
    depthModes: {js_string_array(args.sweep_depth_modes)},
    filterMode: {quote(args.filter_mode)},
    epsilon: {args.epsilon},
    maxMismatches: {args.max_mismatches},
    maxMisses: {args.max_misses},
    includeCaseResults: {js_bool(args.sweep_include_case_results)}
  }}),
  {quote(args.step + '_gpu_candidate_screen_coarse_sweep_summary.json')}
);"""


def build_visible_record_dryrun_command(args: argparse.Namespace) -> str:
    if args.source_mode != "screenCoarse":
        return ""

    return f"""await window.gpuViewerDebug.downloadJsonDebug(
  await window.gpuViewerDebug.captureGpuVisibleRecordDryRunDebug({{
    ensureCurrentFrame: false,
    sourceMode: 'screenCoarse',
    readbackMode: {quote(args.visible_record_readback)},
    maxRecords: {args.visible_record_max_count},
    epsilon: {args.epsilon},
    maxMismatches: {args.max_mismatches}
  }}),
  {quote(args.step + '_gpu_visible_record_dryrun_compare.json')}
);"""


def build_raw_visible_record_dryrun_command(args: argparse.Namespace) -> str:
    if args.source_mode != "screenCoarse":
        return ""

    return f"""await window.gpuViewerDebug.downloadJsonDebug(
  await window.gpuViewerDebug.captureGpuRawVisibleRecordDryRunDebug({{
    ensureCurrentFrame: false,
    recordMode: {quote(args.raw_visible_record_mode)},
    readbackMode: {quote(args.raw_visible_record_readback)},
    maxRecords: {args.raw_visible_record_max_count},
    epsilon: {args.raw_visible_record_epsilon},
    maxMismatches: {args.max_mismatches}
  }}),
  {quote(args.step + '_gpu_raw_visible_record_dryrun_compare.json')}
);"""


def build_webgpu_visible_record_dryrun_command(args: argparse.Namespace) -> str:
    if args.source_mode != "screenCoarse":
        return ""
    backend_implementation_line = (
        f"\n    webgpuBackendImplementation: {quote(args.webgpu_backend_implementation)},"
        if args.webgpu_backend_implementation
        else ""
    )
    phase_step = None
    comparison_mode = None
    fixed_reference_camera_mode_line = ""
    step114_contract = resolve_step114_capture_contract(args.step)
    if step114_contract:
        phase_step, comparison_mode = step114_contract
        fixed_reference_camera_mode_line = (
            "\n    fixedReferenceCameraMode: true,"
            "\n    fixedReferenceCameraActivationMode: "
            "'cuda-aligned-fixed-reference-camera',"
            "\n    referenceCameraMode: 'cuda-aligned-fixed-reference-camera',"
        )
    elif "step113" in args.step:
        phase_step = "phase3-step113"
        comparison_mode = (
            "phase3-step113-cuda-webgpu-covariance-jacobian-conic-parity-closure"
        )
        fixed_reference_camera_mode_line = (
            "\n    fixedReferenceCameraMode: true,"
            "\n    fixedReferenceCameraActivationMode: "
            "'cuda-aligned-fixed-reference-camera',"
            "\n    referenceCameraMode: 'cuda-aligned-fixed-reference-camera',"
        )
    elif "step112" in args.step:
        phase_step = "phase3-step112"
        comparison_mode = (
            "phase3-step112-fixed-reference-camera-projection-orientation-parity-closure"
        )
        fixed_reference_camera_mode_line = (
            "\n    fixedReferenceCameraMode: true,"
            "\n    fixedReferenceCameraActivationMode: "
            "'cuda-aligned-fixed-reference-camera',"
            "\n    referenceCameraMode: 'cuda-aligned-fixed-reference-camera',"
        )
    elif "step111" in args.step:
        phase_step = "phase3-step111"
        comparison_mode = (
            "phase3-step111-rendering-pipeline-parity-gap-closure"
        )
        fixed_reference_camera_mode_line = (
            "\n    fixedReferenceCameraMode: true,"
            "\n    fixedReferenceCameraActivationMode: "
            "'cuda-aligned-fixed-reference-camera',"
            "\n    referenceCameraMode: 'cuda-aligned-fixed-reference-camera',"
        )
    elif "step110" in args.step:
        phase_step = "phase3-step110"
        comparison_mode = (
            "phase3-step110-fixed-condition-visual-comparison-readiness"
        )
        fixed_reference_camera_mode_line = (
            "\n    fixedReferenceCameraMode: true,"
            "\n    fixedReferenceCameraActivationMode: "
            "'cuda-aligned-fixed-reference-camera',"
            "\n    referenceCameraMode: 'cuda-aligned-fixed-reference-camera',"
        )
    elif "step109" in args.step:
        phase_step = "phase3-step109"
        comparison_mode = (
            "phase3-step109-fixed-reference-camera-activation-routing"
        )
        fixed_reference_camera_mode_line = (
            "\n    fixedReferenceCameraMode: true,"
            "\n    fixedReferenceCameraActivationMode: "
            "'cuda-aligned-fixed-reference-camera',"
            "\n    referenceCameraMode: 'cuda-aligned-fixed-reference-camera',"
        )
    elif "step108" in args.step:
        phase_step = "phase3-step108"
        comparison_mode = (
            "phase3-step108-cuda-reference-camera-evidence-activation-gate"
        )
    elif "step107" in args.step:
        phase_step = "phase3-step107"
        comparison_mode = (
            "phase3-step107-fixed-reference-camera-contract-design-gate"
        )
    elif "step106" in args.step:
        phase_step = "phase3-step106"
        comparison_mode = (
            "phase3-step106-capability-based-regression-gate"
        )
    elif "step105" in args.step:
        phase_step = "phase3-step105"
        comparison_mode = (
            "phase3-step105-cuda-reference-visual-parity-baseline"
        )
    elif "step104" in args.step:
        phase_step = "phase3-step104"
        comparison_mode = (
            "phase3-step104-compositor-work-reduction-visual-safety-gate"
        )
    elif "step103" in args.step:
        phase_step = "phase3-step103"
        comparison_mode = (
            "phase3-step103-production-runtime-boundary-work-reduction-gate"
        )
    elif "step102" in args.step:
        phase_step = "phase3-step102"
        comparison_mode = (
            "phase3-step102-production-resource-lifecycle-bottleneck-gate"
        )
    elif "step101" in args.step:
        phase_step = "phase3-step101"
        comparison_mode = (
            "phase3-step101-selective-dirty-dependency-execution-runtime"
        )
    elif "step100" in args.step:
        phase_step = "phase3-step100"
        comparison_mode = (
            "phase3-step100-unified-production-interaction-scheduler-runtime"
        )
    elif "step99" in args.step:
        phase_step = "phase3-step99"
        comparison_mode = "phase3-step99-interactive-camera-viewport-dirty-runtime"
    phase_step_line = (
        f"\n    phaseStep: {quote(phase_step)},"
        if phase_step
        else ""
    )
    comparison_mode_line = (
        f"\n    comparisonMode: {quote(comparison_mode)},"
        if comparison_mode
        else ""
    )
    selected_indices = args.canonical_src_indices
    if not selected_indices and args.cuda_reference_manifest:
        selected_indices = load_cuda_manifest_selected_indices(args.cuda_reference_manifest)
    canonical_indices_line = (
        f"\n    canonicalComparisonSrcIndices: {js_int_array(selected_indices)},"
        if selected_indices and "step114" in args.step
        else ""
    )

    return f"""if (typeof recordStep114CommandCausalStage === 'function') {{
  recordStep114CommandCausalStage('diagnostic-dry-run-started', false, {{
    ensureCurrentFrame: false,
    productionFrameRequested: false
  }});
}}
try {{
  var webgpuVisibleRecordDryRunResult =
    await window.gpuViewerDebug.captureWebGpuVisibleRecordDryRunDebug({{
    ensureCurrentFrame: false,
    maxRecords: {args.webgpu_visible_record_max_count},
    epsilon: {args.webgpu_visible_record_epsilon},
    maxMismatches: {args.max_mismatches},{backend_implementation_line}{phase_step_line}{comparison_mode_line}{canonical_indices_line}{fixed_reference_camera_mode_line}
  }});
  window.__phase3LatestWebGpuVisibleRecordDryRunResult =
    webgpuVisibleRecordDryRunResult;
  if (typeof recordStep114CommandCausalStage === 'function') {{
    recordStep114CommandCausalStage('diagnostic-dry-run-completed', false, {{
      status: webgpuVisibleRecordDryRunResult?.status ?? null,
      productionFrameRequested: false
    }});
  }}

  await window.gpuViewerDebug.downloadJsonDebug(
    webgpuVisibleRecordDryRunResult,
    {quote(args.step + '_webgpu_visible_record_dryrun_compare.json')}
  );

  await window.gpuViewerDebug.downloadJsonDebug(
    {{
      schemaVersion: 'phase3-capture-status-v1',
      captureTarget: 'webgpu-visible-record-dry-run',
      status: 'ok',
      reason: 'ok',
      captureFatalError: false,
      captureExceptionRecorded: false,
      captureErrorName: null,
      captureErrorMessage: null,
      captureErrorStack: null
    }},
    {quote(args.step + '_webgpu_visible_record_dryrun_capture_status.json')}
  );
}} catch (error) {{
  if (typeof recordStep114CommandCausalStage === 'function') {{
    recordStep114CommandCausalStage('diagnostic-dry-run-failed', false, {{
      errorName: error?.name ?? 'Error',
      errorMessage: error?.message ?? String(error)
    }});
  }}
  await window.gpuViewerDebug.downloadJsonDebug(
    {{
      schemaVersion: 'phase3-capture-status-v1',
      captureTarget: 'webgpu-visible-record-dry-run',
      status: 'error',
      reason: 'capture-exception',
      captureFatalError: true,
      captureExceptionRecorded: true,
      captureErrorName: error?.name ?? 'Error',
      captureErrorMessage: error?.message ?? String(error),
      captureErrorStack: error?.stack ?? null,
      captureErrorString: String(error)
    }},
    {quote(args.step + '_webgpu_visible_record_dryrun_capture_status.json')}
  );
  console.error('WebGPU visible record dry-run capture failed', error);
}}"""


def build_runtime_summary_command(args: argparse.Namespace) -> str:
    return f"""var runtime = await window.gpuViewerDebug.captureGpuCandidateRuntimeSummaryDebug();

await window.gpuViewerDebug.downloadJsonDebug(
  runtime,
  {quote(args.step + '_gpu_candidate_runtime_summary.json')}
);

await window.gpuViewerDebug.downloadJsonDebug(
  runtime?.limitedDrawSummary,
  {quote(args.step + '_limited_draw_summary.json')}
);"""


def build_visible_compare_command(args: argparse.Namespace) -> str:
    return f"""await window.gpuViewerDebug.downloadJsonDebug(
  await window.gpuViewerDebug.captureVisibleComparisonDebug({{
    ensureCurrentFrame: false,
    epsilon: {args.epsilon},
    maxMismatches: {args.max_mismatches}
  }}),
  {quote(args.step + '_visible_compare.json')}
);"""


def build_live_same_state_command(args: argparse.Namespace) -> str:
    indices = ", ".join(str(v) for v in args.indices)

    return f"""await window.gpuViewerDebug.downloadLiveSameStateTileAndAssociationDebugJson(
  {{
    pixel: [{args.pixel_x}, {args.pixel_y}],
    indices: [{indices}],
    maxItems: {args.max_items},
    maxEntries: {args.max_entries},
    includeAllEntries: {js_bool(args.include_all_entries)}
  }},
  {{
    tileAccumulation: {quote(args.step + '_live_same_state.json')},
    association: {quote(args.step + '_association.json')},
    summary: {quote(args.step + '_summary.json')}
  }}
);"""


def build_png_command(args: argparse.Namespace) -> str:
    if (
        "step111" in args.step
        or "step112" in args.step
        or "step113" in args.step
        or "step114" in args.step
    ):
        return f"""if (typeof recordStep114CommandCausalStage === 'function') {{
  recordStep114CommandCausalStage('png-capture-started', false, {{
    captureSource: 'last-valid-webgpu-tile-compositor-output'
  }});
}}
var pngCaptureResult = await window.gpuViewerDebug.saveCurrentCanvasPng({{
  name: {quote(args.step + '_canvas.png')},
  captureSource: 'last-valid-webgpu-tile-compositor-output',
  fallbackToCanvasOnCaptureFailure: false,
  renderBeforeCapture: false
}});
var {{ blob: _pngBlob, ...pngCaptureStatus }} = pngCaptureResult ?? {{}};
if (typeof recordStep114CommandCausalStage === 'function') {{
  recordStep114CommandCausalStage('png-capture-completed', false, {{
    status: pngCaptureStatus?.status ?? null,
    captureBlobIdentity: pngCaptureStatus?.captureBlobIdentity ?? null,
    capturedOutputGeneration: pngCaptureStatus?.capturedOutputGeneration ?? null
  }});
}}
if (typeof step114FixedTimeCaptureIsolation !== 'undefined') {{
  step114FixedTimeCaptureIsolation.captureFrame = {{
    ...(step114FixedTimeCaptureIsolation.captureFrame ?? {{}}),
    captureArtifactFrameIdentity:
      pngCaptureStatus.capturedFrameIdentity ??
      pngCaptureStatus.capturedStateIdentity ??
      null,
    pngProductionGeneration:
      pngCaptureStatus.productionOutputGeneration ?? null,
    pngPresentedGeneration:
      pngCaptureStatus.presentedOutputGeneration ?? null,
    pngCapturedGeneration:
      pngCaptureStatus.capturedOutputGeneration ?? null
  }};
}}
await window.gpuViewerDebug.downloadJsonDebug(
  pngCaptureStatus,
  {quote(args.step + '_png_capture_status.json')}
);"""
    return f"""await window.gpuViewerDebug.saveCurrentCanvasPng({{
  name: {quote(args.step + '_canvas.png')},
  renderBeforeCapture: false
}});"""


def build_webgpu_render_state_manifest_command(args: argparse.Namespace) -> str:
    if "step114" not in args.step:
        return ""
    phase_step, comparison_mode = resolve_step114_capture_contract(args.step)
    selected_indices = args.canonical_src_indices
    if not selected_indices and args.cuda_reference_manifest:
        selected_indices = load_cuda_manifest_selected_indices(args.cuda_reference_manifest)
    return f"""await window.gpuViewerDebug.downloadJsonDebug(
  window.gpuViewerDebug.captureWebGpuRenderStateManifestDebug({{
    phaseStep: {quote(phase_step)},
    comparisonMode: {quote(comparison_mode)},
    canonicalComparisonSrcIndices: {js_int_array(selected_indices)},
    captureStateContract: (
      typeof step114FixedTimeCaptureIsolation !== 'undefined'
        ? step114FixedTimeCaptureIsolation
        : (window.__phase3Step114FixedTimeCaptureIsolation ?? null)
    ),
    webgpuVisibleRecordDryRunResult: (
      typeof webgpuVisibleRecordDryRunResult !== 'undefined'
        ? webgpuVisibleRecordDryRunResult
        : (window.__phase3LatestWebGpuVisibleRecordDryRunResult ?? null)
    ),
    pngCaptureStatus: (typeof pngCaptureStatus !== 'undefined' ? pngCaptureStatus : null)
  }}),
  {quote(args.step + '_webgpu_render_state_manifest.json')}
);"""


def build_initial_presentation_evidence_command(args: argparse.Namespace) -> str:
    uses_causal_evidence = uses_fix10_fix4_causal_evidence(args.step)
    step114_contract = resolve_step114_capture_contract(args.step)
    phase_step = step114_contract[0] if step114_contract else None
    production_runtime_behavior_changed = js_bool(
        phase_step in {
            "phase3-step114-fix10-fix5",
            "phase3-step114-fix10-fix6",
            "phase3-step114-fix10-fix6-fix1",
        }
    )
    if (
        "step114_fix10_fix1" not in args.step
        and "step114_fix10_fix2" not in args.step
        and "step114_fix10_fix3" not in args.step
        and not uses_causal_evidence
    ):
        return ""
    final_presentation_download = ""
    if "step114_fix10_fix3" in args.step or uses_causal_evidence:
        final_presentation_download = f"""
await window.gpuViewerDebug.downloadJsonDebug(
  {{
    schemaVersion: 'phase3-final-canvas-presentation-evidence-v1',
    urlOnlyBoundary:
      step114FixedTimeCaptureIsolation?.urlOnlyFinalPresentationBoundary ?? null,
    captureBoundary:
      step114FixedTimeCaptureIsolation?.captureFinalPresentationBoundary ?? null,
    productionRuntimeBehaviorChanged: {production_runtime_behavior_changed}
  }},
  {quote(args.step + '_final_presentation_evidence.json')}
);"""
    if uses_causal_evidence:
        final_presentation_download += f"""
if (typeof recordStep114CommandCausalStage === 'function') {{
  recordStep114CommandCausalStage('causal-evidence-save', false, {{
    pngBlobIdentity: typeof pngCaptureStatus !== 'undefined'
      ? pngCaptureStatus?.captureBlobIdentity ?? null
      : null
  }});
}}
await window.gpuViewerDebug.downloadJsonDebug(
  {{
    ...(step114FixedTimeCaptureIsolation?.commandEraCausalTrace ?? {{}}),
    pngCaptureStatus:
      typeof pngCaptureStatus !== 'undefined' ? pngCaptureStatus : null,
    initialPresentation:
      step114FixedTimeCaptureIsolation?.initialPresentation ?? null,
    captureFrame: step114FixedTimeCaptureIsolation?.captureFrame ?? null,
    captureFinalPresentationBoundary:
      step114FixedTimeCaptureIsolation?.captureFinalPresentationBoundary ?? null
  }},
  {quote(args.step + '_causal_presentation_evidence.json')}
);"""
    return f"""await window.gpuViewerDebug.downloadJsonDebug(
  typeof step114FixedTimeCaptureIsolation !== 'undefined'
    ? step114FixedTimeCaptureIsolation
    : (window.__phase3Step114FixedTimeCaptureIsolation ?? {{
        schemaVersion: 'phase3-step114-fixed-time-capture-isolation-v2',
        blockedReason: 'fixed-time-capture-contract-unavailable'
      }}),
  {quote(args.step + '_initial_presentation_evidence.json')}
);{final_presentation_download}"""


def build_camera_control_debug_command(args: argparse.Namespace) -> str:
    return f"""await window.gpuViewerDebug.downloadJsonDebug(
  window.gpuViewerDebug.getCameraDebugState(),
  {quote(args.step + '_camera_control_initial.json')}
);

// After manual pan/rotate/zoom testing, run this second command to capture the post-interaction state:
// await window.gpuViewerDebug.downloadJsonDebug(
//   window.gpuViewerDebug.getCameraDebugState(),
//   {quote(args.step + '_camera_control_after_manual.json')}
// );"""


def build_fix10_fix4_capture_preamble(
    args: argparse.Namespace,
    phase_step: str,
) -> str:
    production_runtime_behavior_changed = js_bool(
        phase_step in {
            "phase3-step114-fix10-fix5",
            "phase3-step114-fix10-fix6",
            "phase3-step114-fix10-fix6-fix1",
        }
    )
    expected_runtime_contract = {
        "requestedRuntime": args.expected_runtime,
        "effectiveDisplayRuntime": args.expected_effective_display_runtime,
        "backendMode": args.expected_webgpu_backend_mode,
        "backendImplementation": args.expected_webgpu_backend_implementation,
        "canvasPresentationEnabled": (
            js_bool(args.expected_webgpu_canvas_presentation) == "true"
            if args.expected_webgpu_canvas_presentation is not None
            else None
        ),
        "viewerLoopHookEnabled": (
            js_bool(args.expected_webgpu_viewer_loop_hook) == "true"
            if args.expected_webgpu_viewer_loop_hook is not None
            else None
        ),
    }
    runtime_preflight = ""
    if any(value is not None for value in expected_runtime_contract.values()):
        expected_json = json.dumps(expected_runtime_contract, separators=(",", ":"))
        runtime_preflight = f"""
var expectedProductionRuntimeContract = {expected_json};
var productionRuntimeValidation =
  typeof window.gpuViewerDebug.validateExpectedProductionRuntimeContract === 'function'
    ? window.gpuViewerDebug.validateExpectedProductionRuntimeContract(
        step114CommandStartFence?.productionRuntimeContract ?? null,
        expectedProductionRuntimeContract
      )
    : {{
        schemaVersion: 'phase3-expected-production-runtime-validation-v1',
        ready: false,
        expected: expectedProductionRuntimeContract,
        actual: step114CommandStartFence?.productionRuntimeContract ?? null,
        mismatchFields: ['runtime-contract-validator-api'],
        blockedReason: 'production-runtime-contract-validator-unavailable'
      }};
if (productionRuntimeValidation?.ready !== true) {{
  await window.gpuViewerDebug.downloadJsonDebug(
    {{
      schemaVersion: 'phase3-production-runtime-mismatch-artifact-v1',
      captureBlockedBeforeMutation: true,
      synchronousCommandStartFence: step114CommandStartFence,
      productionRuntimeValidation
    }},
    {quote(args.step + '_runtime_mismatch.json')}
  );
  throw new Error('capture-blocked-production-runtime-mismatch');
}}
"""
    return f"""var step114CommandStartFence =
  typeof window.gpuViewerDebug.getSynchronousCommandStartFence === 'function'
    ? window.gpuViewerDebug.getSynchronousCommandStartFence()
    : {{
        schemaVersion: 'phase3-synchronous-command-start-fence-v1',
        source: 'synchronous-command-start-fence-api-unavailable',
        commandStartTimestampMs: performance.now(),
        canonicalBoundary: {{
          finalCanvasEventSequence: null,
          schedulerLatestRequestSequence: null,
          schedulerLatestRequestIdentity: null
        }},
        synchronousReadOnlyFence: null,
        asyncOperationBeforeFence: false,
        mutationBeforeFence: false,
        blockedReason: 'synchronous-command-start-fence-api-unavailable'
      }};
{runtime_preflight}var preCaptureInitialPresentationSnapshot =
  step114CommandStartFence?.initialPresentationSnapshot ?? null;
var preCaptureFinalPresentationBoundary =
  step114CommandStartFence?.urlOnlyFinalPresentation ?? null;
var step114CommandCausalEvents = [];
var recordStep114CommandCausalStage = function(stage, mutatesRuntime, detail = {{}}) {{
  var schedulerState =
    window.gpuViewerDebug.getRenderSchedulerSynchronousSnapshot?.() ?? null;
  var initialState =
    window.gpuViewerDebug.getInitialProductionPresentationSnapshot?.() ?? null;
  var event = {{
    schemaVersion: 'phase3-command-era-causal-event-v1',
    causalSequence: step114CommandCausalEvents.length + 1,
    timestampMs: performance.now(),
    stage,
    mutatesRuntime: mutatesRuntime === true,
    schedulerLatestRequestIdentity: schedulerState?.latestRequestIdentity ?? null,
    schedulerPendingRequestCount: schedulerState?.pendingRequestCount ?? null,
    productionFrameInFlight: schedulerState?.productionFrameInFlight ?? null,
    latestProductionRequestIdentity:
      initialState?.latestProductionRequestIdentity ?? null,
    latestProductionGeneration: initialState?.latestProductionGeneration ?? null,
    latestCompositorGeneration: initialState?.latestCompositorGeneration ?? null,
    latestPresentedGeneration: initialState?.latestPresentedGeneration ?? null,
    detail
  }};
  step114CommandCausalEvents.push(event);
  return event;
}};
recordStep114CommandCausalStage('command-start-fence-captured', false, {{
  fence: step114CommandStartFence?.canonicalBoundary ?? null
}});
var step114FixedTimeCaptureIsolation = {{
  schemaVersion: 'phase3-step114-fixed-time-capture-isolation-v3',
  selectedPhaseStep: {quote(phase_step)},
  policy: 'common-fixed-time-production-capture-after-synchronous-read-only-fence',
  selectedIsolationMode: 'synchronous-fence-before-readiness-or-state-mutation',
  requestedTimeFromUrl: Number(new URLSearchParams(window.location.search).get('time')),
  requestedDatasetTimeFromUrl: Number(new URLSearchParams(window.location.search).get('datasetTime')),
  expectedProductionRuntimeContract:
    typeof expectedProductionRuntimeContract !== 'undefined'
      ? expectedProductionRuntimeContract
      : null,
  productionRuntimeValidation:
    typeof productionRuntimeValidation !== 'undefined'
      ? productionRuntimeValidation
      : null,
  synchronousCommandStartFence: step114CommandStartFence,
  preCaptureSnapshot: preCaptureInitialPresentationSnapshot,
  urlOnlyFinalPresentationBoundary: preCaptureFinalPresentationBoundary,
  captureCommandBoundary: {{
    startedAtMs: step114CommandStartFence?.commandStartTimestampMs ?? null,
    finalCanvasEventSequence:
      step114CommandStartFence?.canonicalBoundary?.finalCanvasEventSequence ?? null,
    schedulerLatestRequestSequence:
      step114CommandStartFence?.canonicalBoundary?.schedulerLatestRequestSequence ?? null,
    schedulerLatestRequestIdentity:
      step114CommandStartFence?.canonicalBoundary?.schedulerLatestRequestIdentity ?? null,
    snapshotWasFirstRuntimeOperation: true,
    synchronousFenceEstablished: step114CommandStartFence?.synchronousReadOnlyFence === true,
    onlyReadOnlySnapshotsBeforeBoundary: true,
    noAsyncOperationBeforeFence: step114CommandStartFence?.asyncOperationBeforeFence === false,
    noMutationBeforeFence: step114CommandStartFence?.mutationBeforeFence === false,
    snapshotTakenBeforeReadinessWait: true,
    snapshotTakenBeforeSceneRetry: true,
    snapshotTakenBeforeScheduleRender: true,
    snapshotTakenBeforeGpuReadback: true,
    baselineProductionGeneration:
      preCaptureInitialPresentationSnapshot?.latestProductionGeneration ?? null,
    baselineCompositorGeneration:
      preCaptureInitialPresentationSnapshot?.latestCompositorGeneration ?? null,
    baselinePresentedGeneration:
      preCaptureInitialPresentationSnapshot?.latestPresentedGeneration ?? null
  }},
  schedulerProbe: {{ executedInCaptureCommand: false }},
  cameraDirtyProbe: {{ executedInCaptureCommand: false }},
  stateRestoration: {{ required: false, performed: false }},
  fixedFrameWaitMs: {args.render_wait_ms},
  freshGenerationTimeoutMs: {args.viewer_data_ready_timeout_ms},
  productionRuntimeBehaviorChanged: {production_runtime_behavior_changed},
  commandEraCausalTrace: {{
    schemaVersion: 'phase3-command-era-causal-trace-v1',
    fenceEventSequence:
      step114CommandStartFence?.canonicalBoundary?.finalCanvasEventSequence ?? null,
    events: step114CommandCausalEvents,
    firstMutation: null,
    firstNonblankPresentation: null
  }}
}};
window.__phase3Step114FixedTimeCaptureIsolation = step114FixedTimeCaptureIsolation;
recordStep114CommandCausalStage('readiness-check-started', false);
var viewerDebugDataReadiness =
  typeof window.gpuViewerDebug.waitForViewerDebugDataReady === 'function'
    ? await window.gpuViewerDebug.waitForViewerDebugDataReady({{
        timeoutMs: {args.viewer_data_ready_timeout_ms},
        retryDefaultScene: true,
        requireRaw: true
      }})
    : {{ ready: null, blockedReason: 'viewer-readiness-api-unavailable' }};
step114FixedTimeCaptureIsolation.viewerDebugDataReadiness = viewerDebugDataReadiness;
recordStep114CommandCausalStage('readiness-check-completed', false, {{
  ready: viewerDebugDataReadiness?.ready ?? null,
  attempts: viewerDebugDataReadiness?.attempts ?? []
}});
if (viewerDebugDataReadiness?.attempts?.some(
  item => item?.stage === 'start-default-scene-load-retry'
)) {{
  recordStep114CommandCausalStage('scene-retry-started', true, {{
    cause: 'readiness-api-started-default-scene-load-retry'
  }});
}}
recordStep114CommandCausalStage('fixed-time-state-verified', false, {{
  requestedTime: step114FixedTimeCaptureIsolation.requestedTimeFromUrl,
  requestedDatasetTime: step114FixedTimeCaptureIsolation.requestedDatasetTimeFromUrl,
  stateMutationPerformed: false
}});
var captureScheduleRequest = null;
if (typeof window.gpuViewerDebug.scheduleRender === 'function') {{
  recordStep114CommandCausalStage('force-production-schedule-requested', true, {{
    source: 'fixed-time-artifact-capture',
    forceProductionUpdate: true
  }});
  captureScheduleRequest = await window.gpuViewerDebug.scheduleRender({{
    source: 'fixed-time-artifact-capture',
    forceProductionUpdate: true,
    metadata: {{ policy: 'common-fixed-time-production-capture' }}
  }});
}}
recordStep114CommandCausalStage('schedule-request-returned', false, {{
  requestIdentity: captureScheduleRequest?.requestIdentity ?? null,
  requestDisposition: captureScheduleRequest?.disposition ?? null
}});
step114FixedTimeCaptureIsolation.captureFrame = {{
  schemaVersion: 'phase3-fixed-time-capture-frame-request-v1',
  requestIdentity: captureScheduleRequest?.requestIdentity ?? null,
  requestSource: captureScheduleRequest?.source ?? null,
  requestDisposition: captureScheduleRequest?.disposition ?? null,
  forceProductionUpdate: captureScheduleRequest?.forceProductionUpdate ?? null,
  baselineProductionGeneration:
    step114FixedTimeCaptureIsolation.captureCommandBoundary.baselineProductionGeneration,
  baselineCompositorGeneration:
    step114FixedTimeCaptureIsolation.captureCommandBoundary.baselineCompositorGeneration,
  baselinePresentedGeneration:
    step114FixedTimeCaptureIsolation.captureCommandBoundary.baselinePresentedGeneration
}};
var captureGenerationWaitStartedAtMs = performance.now();
var captureProductionPresentationTrace = null;
var captureProductionFrameEvidence = null;
var captureGenerationIsFresh = false;
while (performance.now() - captureGenerationWaitStartedAtMs <
  step114FixedTimeCaptureIsolation.freshGenerationTimeoutMs) {{
  captureProductionPresentationTrace =
    window.gpuViewerDebug.getInitialProductionPresentationSnapshot?.() ?? null;
  captureProductionFrameEvidence =
    captureProductionPresentationTrace?.frameHistory?.find(
      frame => frame.requestIdentity === captureScheduleRequest?.requestIdentity
    ) ?? null;
  var baselineProductionGeneration =
    step114FixedTimeCaptureIsolation.captureFrame.baselineProductionGeneration;
  var captureProductionGeneration =
    captureProductionFrameEvidence?.productionGeneration ?? null;
  captureGenerationIsFresh =
    Number.isFinite(Number(captureProductionGeneration)) &&
    (baselineProductionGeneration === null ||
      !Number.isFinite(Number(baselineProductionGeneration)) ||
      Number(captureProductionGeneration) > Number(baselineProductionGeneration));
  if (captureProductionFrameEvidence?.productionFrameCompleted === true &&
      captureGenerationIsFresh) break;
  await new Promise(resolve => setTimeout(resolve, 16));
}}
step114FixedTimeCaptureIsolation.postCaptureRequestTrace =
  captureProductionPresentationTrace;
step114FixedTimeCaptureIsolation.captureFrame = {{
  ...step114FixedTimeCaptureIsolation.captureFrame,
  productionFrameCompleted:
    captureProductionFrameEvidence?.productionFrameCompleted ?? false,
  productionGeneration: captureProductionFrameEvidence?.productionGeneration ?? null,
  compositorGeneration: captureProductionFrameEvidence?.compositorGeneration ?? null,
  presentedGeneration: captureProductionFrameEvidence?.presentedGeneration ?? null,
  productionFrameIdentity:
    captureProductionFrameEvidence?.productionFrameIdentity ?? null,
  presentedFrameIdentity:
    captureProductionFrameEvidence?.presentedFrameIdentity ?? null,
  productionSourceRequestIdentity:
    captureProductionFrameEvidence?.productionSourceRequestIdentity ?? null,
  compositorOutputGenerated:
    captureProductionFrameEvidence?.compositorOutputGenerated ?? null,
  viewerCanvasPresented: captureProductionFrameEvidence?.viewerCanvasPresented ?? null,
  knownNonblank: captureProductionFrameEvidence?.knownNonblank ?? null,
  logicalPresentationSucceeded:
    captureProductionFrameEvidence?.logicalPresentationSucceeded ?? null,
  freshGenerationObserved: captureGenerationIsFresh === true,
  waitTimedOut: !captureProductionFrameEvidence || captureGenerationIsFresh !== true,
  waitedMs: performance.now() - captureGenerationWaitStartedAtMs,
  blockedReason: captureProductionFrameEvidence?.blockedReason ?? null,
  runtimeError: captureProductionFrameEvidence?.runtimeError ?? null
}};
recordStep114CommandCausalStage('production-frame-observed', false, {{
  requestIdentity: step114FixedTimeCaptureIsolation.captureFrame.requestIdentity,
  productionSourceRequestIdentity:
    step114FixedTimeCaptureIsolation.captureFrame.productionSourceRequestIdentity,
  generation: step114FixedTimeCaptureIsolation.captureFrame.productionGeneration,
  completed: step114FixedTimeCaptureIsolation.captureFrame.productionFrameCompleted
}});
recordStep114CommandCausalStage('compositor-output-observed', false, {{
  generated: step114FixedTimeCaptureIsolation.captureFrame.compositorOutputGenerated,
  generation: step114FixedTimeCaptureIsolation.captureFrame.compositorGeneration
}});
var captureFinalPresentationBoundary =
  typeof window.gpuViewerDebug.waitForFinalCanvasPresentationQuiescence === 'function'
    ? await window.gpuViewerDebug.waitForFinalCanvasPresentationQuiescence({{
        boundaryKind: 'post-capture-quiescent',
        expectedRequestIdentity:
          step114FixedTimeCaptureIsolation.captureFrame.requestIdentity,
        expectedGeneration:
          step114FixedTimeCaptureIsolation.captureFrame.productionGeneration,
        expectedFrameIdentity:
          step114FixedTimeCaptureIsolation.captureFrame.productionFrameIdentity,
        requiredConsecutive: 3,
        pollIntervalMs: 25,
        timeoutMs: 2000
      }})
    : {{
        schemaVersion: 'phase3-final-canvas-presentation-boundary-v1',
        browserVisibleResult: null,
        classification: 'unknown-quiescence-api-unavailable',
        unknownOrBlockedReason: 'passive-quiescence-api-unavailable'
      }};
step114FixedTimeCaptureIsolation.captureFinalPresentationBoundary =
  captureFinalPresentationBoundary;
recordStep114CommandCausalStage('canvas-presentation-and-quiescence-observed', false, {{
  browserVisibleResult: captureFinalPresentationBoundary?.browserVisibleResult ?? null,
  finalCanvasEventIdentity:
    captureFinalPresentationBoundary?.finalCanvasEventIdentity ?? null,
  quiescent:
    captureFinalPresentationBoundary?.quiescenceObservation?.quiescent ?? null,
  timedOut: captureFinalPresentationBoundary?.quiescenceTimedOut ?? null
}});
var fenceEventSequence =
  step114CommandStartFence?.canonicalBoundary?.finalCanvasEventSequence ?? null;
var firstCommandEraNonblankPresentation =
  captureFinalPresentationBoundary?.eventHistory?.find(event =>
    Number.isFinite(Number(fenceEventSequence)) &&
    Number(event?.eventSequence) > Number(fenceEventSequence) &&
    event?.sourcePixelResult === 'nonblank' &&
    event?.canvasWriteCompleted === true &&
    event?.presentationFailed !== true
  ) ?? null;
step114FixedTimeCaptureIsolation.commandEraCausalTrace.firstMutation =
  step114CommandCausalEvents.find(event => event.mutatesRuntime === true) ?? null;
step114FixedTimeCaptureIsolation.commandEraCausalTrace.firstNonblankPresentation =
  firstCommandEraNonblankPresentation;
step114FixedTimeCaptureIsolation.commandEraCausalTrace.firstNonblankCausalClassification =
  firstCommandEraNonblankPresentation == null
    ? 'command-era-nonblank-presentation-not-observed'
    : firstCommandEraNonblankPresentation.sourceRequestIdentity ===
        captureScheduleRequest?.requestIdentity
      ? 'fixed-time-force-production-request-produced-first-command-era-nonblank-presentation'
      : viewerDebugDataReadiness?.attempts?.some(
          item => item?.stage === 'start-default-scene-load-retry'
        )
        ? 'readiness-scene-retry-preceded-first-command-era-nonblank-presentation'
        : 'pre-existing-asynchronous-runtime-work-produced-first-command-era-nonblank-presentation';
step114FixedTimeCaptureIsolation.captureFrame = {{
  ...step114FixedTimeCaptureIsolation.captureFrame,
  browserVisibleFinalPresentationKnown:
    captureFinalPresentationBoundary?.browserVisibleResult === true ||
    captureFinalPresentationBoundary?.browserVisibleResult === false,
  browserVisiblePixelNonblank:
    captureFinalPresentationBoundary?.browserVisibleResult ?? null,
  finalCanvasEventIdentity:
    captureFinalPresentationBoundary?.finalCanvasEventIdentity ?? null,
  finalPresentationSource:
    captureFinalPresentationBoundary?.finalPresentationSource ?? null,
  finalPresentationBlockedReason:
    captureFinalPresentationBoundary?.unknownOrBlockedReason ?? null
}};
var urlOnlyFinalPresentationResult =
  preCaptureFinalPresentationBoundary?.browserVisibleResult ?? null;
var captureFrameSucceeded =
  step114FixedTimeCaptureIsolation.captureFrame.productionFrameCompleted === true &&
  step114FixedTimeCaptureIsolation.captureFrame.freshGenerationObserved === true &&
  step114FixedTimeCaptureIsolation.captureFrame.logicalPresentationSucceeded === true &&
  step114FixedTimeCaptureIsolation.captureFrame.browserVisiblePixelNonblank === true;
step114FixedTimeCaptureIsolation.initialPresentation = {{
  ...preCaptureInitialPresentationSnapshot,
  schemaVersion: 'phase3-initial-production-presentation-observation-v4',
  finalCanvasPresentationBoundary: preCaptureFinalPresentationBoundary,
  browserVisibleFinalPresentationKnown:
    urlOnlyFinalPresentationResult === true || urlOnlyFinalPresentationResult === false,
  browserVisiblePixelNonblank: urlOnlyFinalPresentationResult,
  urlLoadAloneGaussianVisible: urlOnlyFinalPresentationResult,
  observationPoint: 'synchronous-command-start-fence',
  captureTriggered: captureScheduleRequest !== null,
  captureCommandDependencyKnown:
    urlOnlyFinalPresentationResult === true ||
    (urlOnlyFinalPresentationResult === false && captureFrameSucceeded),
  captureCommandDependencyRemaining:
    urlOnlyFinalPresentationResult === true
      ? false
      : urlOnlyFinalPresentationResult === false && captureFrameSucceeded &&
          firstCommandEraNonblankPresentation != null
        ? true
        : null
}};
step114FixedTimeCaptureIsolation.initialAndCaptureGenerationSeparated =
  Number.isFinite(Number(
    step114FixedTimeCaptureIsolation.initialPresentation.initialProductionGeneration
  )) && Number.isFinite(Number(
    step114FixedTimeCaptureIsolation.captureFrame.productionGeneration
  ))
    ? Number(step114FixedTimeCaptureIsolation.initialPresentation.initialProductionGeneration) !==
      Number(step114FixedTimeCaptureIsolation.captureFrame.productionGeneration)
    : null;
"""


def build_fixed_time_capture_preamble(
    args: argparse.Namespace,
    phase_step: str,
) -> str:
    if phase_step in {
        "phase3-step114-fix10-fix4",
        "phase3-step114-fix10-fix5",
        "phase3-step114-fix10-fix6",
        "phase3-step114-fix10-fix6-fix1",
    }:
        return build_fix10_fix4_capture_preamble(args, phase_step)
    return f"""var preCaptureInitialPresentationSnapshot =
  typeof window.gpuViewerDebug.getInitialProductionPresentationSnapshot === 'function'
    ? window.gpuViewerDebug.getInitialProductionPresentationSnapshot()
    : {{
        schemaVersion: 'phase3-initial-production-presentation-trace-v2',
        source: 'read-only-snapshot-api-unavailable',
        snapshotTakenAtMs: performance.now(),
        readOnlySnapshot: {{
          getterMutatesRuntimeState: null,
          cloneReturnedToCaller: null
        }},
        classification: 'initial-presentation-evidence-insufficient',
        blockedReason: 'read-only-initial-presentation-snapshot-api-unavailable'
      }};
var preCaptureFinalPresentationBoundary =
  typeof window.gpuViewerDebug.getFinalCanvasPresentationBoundarySnapshot === 'function'
    ? window.gpuViewerDebug.getFinalCanvasPresentationBoundarySnapshot({{
        boundaryKind: 'url-only',
        expectedRequestIdentity:
          preCaptureInitialPresentationSnapshot?.initialRequestIdentity ?? null,
        expectedGeneration:
          preCaptureInitialPresentationSnapshot?.initialProductionGeneration ?? null,
        expectedFrameIdentity:
          preCaptureInitialPresentationSnapshot?.initialFrameIdentity ?? null,
        requiredSteadyStateEventCount: 8
      }})
    : {{
        schemaVersion: 'phase3-final-canvas-presentation-boundary-v1',
        boundaryKind: 'url-only',
        boundaryTimestampMs: performance.now(),
        browserVisibleResult: null,
        classification: 'unknown-final-canvas-trace-api-unavailable',
        unknownOrBlockedReason:
          'read-only-final-canvas-presentation-snapshot-api-unavailable'
      }};
var step114FixedTimeCaptureIsolation = {{
  schemaVersion: 'phase3-step114-fixed-time-capture-isolation-v2',
  selectedPhaseStep: {quote(phase_step)},
  policy: 'common-fixed-time-production-capture-after-read-only-initial-snapshot',
  selectedIsolationMode: 'snapshot-before-readiness-or-state-mutation',
  requestedTimeFromUrl: Number(new URLSearchParams(window.location.search).get('time')),
  requestedDatasetTimeFromUrl: Number(new URLSearchParams(window.location.search).get('datasetTime')),
  preCaptureSnapshot: preCaptureInitialPresentationSnapshot,
  urlOnlyFinalPresentationBoundary: preCaptureFinalPresentationBoundary,
  captureCommandBoundary: {{
    startedAtMs:
      preCaptureFinalPresentationBoundary?.boundaryTimestampMs ??
      preCaptureInitialPresentationSnapshot?.snapshotTakenAtMs ??
      null,
    snapshotWasFirstRuntimeOperation: true,
    onlyReadOnlySnapshotsBeforeBoundary: true,
    snapshotTakenBeforeReadinessWait: true,
    snapshotTakenBeforeSceneRetry: true,
    snapshotTakenBeforeScheduleRender: true,
    snapshotTakenBeforeGpuReadback: true,
    baselineProductionGeneration:
      preCaptureInitialPresentationSnapshot?.latestProductionGeneration ?? null,
    baselineCompositorGeneration:
      preCaptureInitialPresentationSnapshot?.latestCompositorGeneration ?? null,
    baselinePresentedGeneration:
      preCaptureInitialPresentationSnapshot?.latestPresentedGeneration ?? null
  }},
  schedulerProbe: {{
    executedInCaptureCommand: false,
    separatedReason: 'scheduler-probe-mutates-time-by-configured-delta'
  }},
  cameraDirtyProbe: {{
    executedInCaptureCommand: false,
    separatedReason: 'camera-dirty-probe-mutates-camera-state'
  }},
  stateRestoration: {{
    required: false,
    performed: false,
    reason: 'state-changing-probes-not-run-before-step114-artifact-capture'
  }},
  captureOrder: [
    'read-only-initial-production-presentation-snapshot',
    'read-only-url-only-final-canvas-presentation-boundary',
    'wait-viewer-debug-data-ready',
    'request-fresh-fixed-state-production-frame',
    'wait-for-matching-request-and-fresh-generation',
    'passive-capture-final-presentation-steady-state-boundary',
    'capture-png-from-last-valid-production-output',
    'capture-webgpu-direct-evidence',
    'capture-webgpu-render-state-manifest',
    'capture-initial-presentation-evidence'
  ],
  fixedFrameWaitMs: {args.render_wait_ms},
  freshGenerationTimeoutMs: {args.viewer_data_ready_timeout_ms},
  productionRuntimeBehaviorChanged: false,
  captureSchedulerMetadataExtended: true
}};
window.__phase3Step114FixedTimeCaptureIsolation = step114FixedTimeCaptureIsolation;
if (typeof window.gpuViewerDebug.waitForViewerDebugDataReady === 'function') {{
  var viewerDebugDataReadiness =
    await window.gpuViewerDebug.waitForViewerDebugDataReady({{
      timeoutMs: {args.viewer_data_ready_timeout_ms},
      retryDefaultScene: true,
      requireRaw: true
    }});
  step114FixedTimeCaptureIsolation.viewerDebugDataReadiness = viewerDebugDataReadiness;
  console.log('viewerDebugDataReadiness', viewerDebugDataReadiness);
}}
var captureScheduleRequest = null;
if (typeof window.gpuViewerDebug.scheduleRender === 'function') {{
  captureScheduleRequest = await window.gpuViewerDebug.scheduleRender({{
    source: 'fixed-time-artifact-capture',
    forceProductionUpdate: true,
    metadata: {{
      policy: 'common-fixed-time-production-capture',
      requestedTime: step114FixedTimeCaptureIsolation.requestedTimeFromUrl
    }}
  }});
}}
step114FixedTimeCaptureIsolation.captureFrame = {{
  schemaVersion: 'phase3-fixed-time-capture-frame-request-v1',
  requestIdentity: captureScheduleRequest?.requestIdentity ?? null,
  requestSource: captureScheduleRequest?.source ?? null,
  requestDisposition: captureScheduleRequest?.disposition ?? null,
  forceProductionUpdate: captureScheduleRequest?.forceProductionUpdate ?? null,
  baselineProductionGeneration:
    step114FixedTimeCaptureIsolation.captureCommandBoundary.baselineProductionGeneration,
  baselineCompositorGeneration:
    step114FixedTimeCaptureIsolation.captureCommandBoundary.baselineCompositorGeneration,
  baselinePresentedGeneration:
    step114FixedTimeCaptureIsolation.captureCommandBoundary.baselinePresentedGeneration,
  productionFrameCompleted: false,
  freshGenerationObserved: false,
  waitTimedOut: false
}};
var captureGenerationWaitStartedAtMs = performance.now();
var captureProductionPresentationTrace = null;
var captureProductionFrameEvidence = null;
while (
  performance.now() - captureGenerationWaitStartedAtMs <
    step114FixedTimeCaptureIsolation.freshGenerationTimeoutMs
) {{
  captureProductionPresentationTrace =
    typeof window.gpuViewerDebug.getInitialProductionPresentationSnapshot === 'function'
      ? window.gpuViewerDebug.getInitialProductionPresentationSnapshot()
      : null;
  captureProductionFrameEvidence =
    captureProductionPresentationTrace?.frameHistory?.find(
      frame =>
        frame.requestIdentity ===
        step114FixedTimeCaptureIsolation.captureFrame.requestIdentity
    ) ?? null;
  var baselineProductionGeneration =
    step114FixedTimeCaptureIsolation.captureFrame.baselineProductionGeneration;
  var captureProductionGeneration =
    captureProductionFrameEvidence?.productionGeneration ?? null;
  var captureGenerationIsFresh =
    captureProductionGeneration !== null &&
    Number.isFinite(Number(captureProductionGeneration)) &&
    (
      baselineProductionGeneration === null ||
      !Number.isFinite(Number(baselineProductionGeneration)) ||
      Number(captureProductionGeneration) > Number(baselineProductionGeneration)
    );
  if (
    captureProductionFrameEvidence?.productionFrameCompleted === true &&
    captureGenerationIsFresh
  ) {{
    break;
  }}
  await new Promise(resolve => setTimeout(resolve, 16));
}}
step114FixedTimeCaptureIsolation.postCaptureRequestTrace =
  captureProductionPresentationTrace;
step114FixedTimeCaptureIsolation.captureFrame = {{
  ...step114FixedTimeCaptureIsolation.captureFrame,
  productionFrameCompleted:
    captureProductionFrameEvidence?.productionFrameCompleted ?? false,
  productionGeneration:
    captureProductionFrameEvidence?.productionGeneration ?? null,
  compositorGeneration:
    captureProductionFrameEvidence?.compositorGeneration ?? null,
  presentedGeneration:
    captureProductionFrameEvidence?.presentedGeneration ?? null,
  productionFrameIdentity:
    captureProductionFrameEvidence?.productionFrameIdentity ?? null,
  presentedFrameIdentity:
    captureProductionFrameEvidence?.presentedFrameIdentity ?? null,
  compositorOutputGenerated:
    captureProductionFrameEvidence?.compositorOutputGenerated ?? null,
  viewerCanvasPresented:
    captureProductionFrameEvidence?.viewerCanvasPresented ?? null,
  knownNonblank: captureProductionFrameEvidence?.knownNonblank ?? null,
  logicalPresentationSucceeded:
    captureProductionFrameEvidence?.logicalPresentationSucceeded ?? null,
  currentTextureRgbPixelEvidenceKnown:
    captureProductionFrameEvidence?.currentTextureRgbPixelEvidenceKnown ?? null,
  browserVisibleFinalPresentationKnown:
    captureProductionFrameEvidence?.browserVisibleFinalPresentationKnown ?? null,
  browserVisiblePixelNonblank:
    captureProductionFrameEvidence?.browserVisiblePixelNonblank ?? null,
  finalPresentSourceTracingReady:
    captureProductionFrameEvidence?.finalPresentSourceTracingReady ?? null,
  finalPresentSourceStable:
    captureProductionFrameEvidence?.finalPresentSourceStable ?? null,
  finalPresentSourceAlternates:
    captureProductionFrameEvidence?.finalPresentSourceAlternates ?? null,
  finalPresentSourceSequence:
    captureProductionFrameEvidence?.finalPresentSourceSequence ?? [],
  tileCompositorOwnsFinalPresentation:
    captureProductionFrameEvidence?.tileCompositorOwnsFinalPresentation ?? null,
  steadyStateSamplingReady:
    captureProductionFrameEvidence?.steadyStateSamplingReady ?? null,
  steadyStateSampledRafCount:
    captureProductionFrameEvidence?.steadyStateSampledRafCount ?? null,
  presentationSource:
    captureProductionFrameEvidence?.presentationSource ?? null,
  blockedReason: captureProductionFrameEvidence?.blockedReason ?? null,
  runtimeError: captureProductionFrameEvidence?.runtimeError ?? null,
  freshGenerationObserved: captureGenerationIsFresh === true,
  waitTimedOut:
    !captureProductionFrameEvidence || captureGenerationIsFresh !== true,
  waitedMs: performance.now() - captureGenerationWaitStartedAtMs
}};
var captureFinalPresentationBoundary =
  typeof window.gpuViewerDebug.waitForFinalCanvasPresentationSteadyState === 'function'
    ? await window.gpuViewerDebug.waitForFinalCanvasPresentationSteadyState({{
        boundaryKind: 'post-capture',
        expectedRequestIdentity:
          step114FixedTimeCaptureIsolation.captureFrame.requestIdentity,
        expectedGeneration:
          step114FixedTimeCaptureIsolation.captureFrame.productionGeneration,
        expectedFrameIdentity:
          step114FixedTimeCaptureIsolation.captureFrame.productionFrameIdentity,
        requiredSteadyStateEventCount: 8,
        requiredRafCount: 8
      }})
    : {{
        schemaVersion: 'phase3-final-canvas-presentation-boundary-v1',
        boundaryKind: 'post-capture',
        boundaryTimestampMs: performance.now(),
        browserVisibleResult: null,
        classification: 'unknown-final-canvas-trace-api-unavailable',
        unknownOrBlockedReason:
          'passive-final-canvas-steady-state-api-unavailable'
      }};
step114FixedTimeCaptureIsolation.captureFinalPresentationBoundary =
  captureFinalPresentationBoundary;
step114FixedTimeCaptureIsolation.captureFrame = {{
  ...step114FixedTimeCaptureIsolation.captureFrame,
  browserVisibleFinalPresentationKnown:
    captureFinalPresentationBoundary?.browserVisibleResult === true ||
    captureFinalPresentationBoundary?.browserVisibleResult === false,
  browserVisiblePixelNonblank:
    captureFinalPresentationBoundary?.browserVisibleResult ?? null,
  finalCanvasEventIdentity:
    captureFinalPresentationBoundary?.finalCanvasEventIdentity ?? null,
  finalPresentationSource:
    captureFinalPresentationBoundary?.finalPresentationSource ?? null,
  finalPresentationBoundaryIdentity:
    captureFinalPresentationBoundary?.boundaryIdentity ?? null,
  finalPresentationBlockedReason:
    captureFinalPresentationBoundary?.unknownOrBlockedReason ?? null
}};
var urlOnlyFinalPresentationResult =
  preCaptureFinalPresentationBoundary?.browserVisibleResult ?? null;
var initialPresentationSucceededBeforeCapture =
  urlOnlyFinalPresentationResult === true;
var captureFrameSucceeded =
  step114FixedTimeCaptureIsolation.captureFrame.productionFrameCompleted === true &&
  step114FixedTimeCaptureIsolation.captureFrame.freshGenerationObserved === true &&
  step114FixedTimeCaptureIsolation.captureFrame.viewerCanvasPresented === true &&
  step114FixedTimeCaptureIsolation.captureFrame.logicalPresentationSucceeded === true &&
  step114FixedTimeCaptureIsolation.captureFrame.browserVisibleFinalPresentationKnown === true &&
  step114FixedTimeCaptureIsolation.captureFrame.browserVisiblePixelNonblank === true;
step114FixedTimeCaptureIsolation.initialPresentation = {{
  ...preCaptureInitialPresentationSnapshot,
  schemaVersion: 'phase3-initial-production-presentation-observation-v3',
  finalCanvasPresentationBoundary: preCaptureFinalPresentationBoundary,
  browserVisibleFinalPresentationKnown:
    preCaptureFinalPresentationBoundary?.browserVisibleResult === true ||
    preCaptureFinalPresentationBoundary?.browserVisibleResult === false,
  browserVisiblePixelNonblank:
    preCaptureFinalPresentationBoundary?.browserVisibleResult ?? null,
  urlLoadAloneGaussianVisible:
    preCaptureFinalPresentationBoundary?.browserVisibleResult ?? null,
  observationPoint: 'capture-command-first-read-only-operation',
  captureTriggered: captureScheduleRequest !== null,
  captureCommandDependencyKnown:
    initialPresentationSucceededBeforeCapture ||
    (urlOnlyFinalPresentationResult === false && captureFrameSucceeded),
  captureCommandDependencyRemaining:
    initialPresentationSucceededBeforeCapture
      ? false
      : urlOnlyFinalPresentationResult === false && captureFrameSucceeded
        ? true
        : null,
  classification:
    initialPresentationSucceededBeforeCapture
      ? 'url-only-initial-production-presentation-succeeded'
      : urlOnlyFinalPresentationResult === false && captureFrameSucceeded
        ? 'capture-command-or-retry-generated-first-observed-presentation'
        : preCaptureInitialPresentationSnapshot?.classification ??
          'initial-presentation-evidence-insufficient'
}};
step114FixedTimeCaptureIsolation.initialAndCaptureGenerationSeparated =
  step114FixedTimeCaptureIsolation.initialPresentation.initialProductionGeneration !== null &&
  step114FixedTimeCaptureIsolation.captureFrame.productionGeneration !== null &&
  Number.isFinite(Number(
    step114FixedTimeCaptureIsolation.initialPresentation.initialProductionGeneration
  )) &&
  Number.isFinite(Number(
    step114FixedTimeCaptureIsolation.captureFrame.productionGeneration
  ))
    ? Number(step114FixedTimeCaptureIsolation.captureFrame.productionGeneration) !==
      Number(
        step114FixedTimeCaptureIsolation.initialPresentation.initialProductionGeneration
      )
    : null;
"""


def build_preamble(args: argparse.Namespace) -> str:
    if not args.include_preamble:
        return ""

    fixed_time_contract = resolve_step114_capture_contract(args.step)
    if (
        fixed_time_contract
        and any(label in args.step for label in (
            "step114_fix8",
            "step114_fix9",
            "step114_fix10",
        ))
    ):
        phase_step, _comparison_mode = fixed_time_contract
        return build_fixed_time_capture_preamble(args, phase_step)

    if (
        "step98" in args.step
        or "step99" in args.step
        or "step100" in args.step
        or "step101" in args.step
        or "step102" in args.step
        or "step103" in args.step
        or "step104" in args.step
        or "step105" in args.step
        or "step106" in args.step
        or "step107" in args.step
        or "step108" in args.step
        or "step109" in args.step
        or "step110" in args.step
        or "step111" in args.step
        or "step112" in args.step
        or "step113" in args.step
        or "step114" in args.step
    ):
        camera_probe = ""
        if (
            "step99" in args.step
            or "step100" in args.step
            or "step101" in args.step
            or "step102" in args.step
            or "step103" in args.step
            or "step104" in args.step
            or "step105" in args.step
            or "step106" in args.step
            or "step107" in args.step
            or "step108" in args.step
            or "step109" in args.step
            or "step110" in args.step
            or "step111" in args.step
            or "step112" in args.step
            or "step113" in args.step
            or "step114" in args.step
        ):
            camera_probe = f"""if (typeof window.gpuViewerDebug.runViewerCameraDirtySchedulerProbe === 'function') {{
  var viewerCameraDirtySchedulerProbe =
    await window.gpuViewerDebug.runViewerCameraDirtySchedulerProbe({{
      waitMs: {args.render_wait_ms},
      cameraDelta: 0.025
    }});
  console.log('viewerCameraDirtySchedulerProbe', viewerCameraDirtySchedulerProbe);
}}
"""
        return f"""if (typeof window.gpuViewerDebug.waitForViewerDebugDataReady === 'function') {{
  var viewerDebugDataReadiness =
    await window.gpuViewerDebug.waitForViewerDebugDataReady({{
      timeoutMs: {args.viewer_data_ready_timeout_ms},
      retryDefaultScene: true,
      requireRaw: true
    }});
  console.log('viewerDebugDataReadiness', viewerDebugDataReadiness);
}}
if (typeof window.gpuViewerDebug.runViewerConnectedSchedulerProbe === 'function') {{
  var viewerConnectedSchedulerProbe =
    await window.gpuViewerDebug.runViewerConnectedSchedulerProbe({{
      waitMs: {args.render_wait_ms},
      timeDelta: 0.05
    }});
  console.log('viewerConnectedSchedulerProbe', viewerConnectedSchedulerProbe);
}} else {{
  window.gpuViewerDebug.scheduleRender();
  await new Promise(r => setTimeout(r, {args.render_wait_ms}));
}}
{camera_probe}
"""

    return f"""window.gpuViewerDebug.scheduleRender();
await new Promise(r => setTimeout(r, {args.render_wait_ms}));
if (typeof window.gpuViewerDebug.waitForViewerDebugDataReady === 'function') {{
  var viewerDebugDataReadiness =
    await window.gpuViewerDebug.waitForViewerDebugDataReady({{
      timeoutMs: {args.viewer_data_ready_timeout_ms},
      retryDefaultScene: true,
      requireRaw: true
    }});
  console.log('viewerDebugDataReadiness', viewerDebugDataReadiness);
}}
"""


def build_commands(args: argparse.Namespace) -> str:
    parts: List[str] = []
    parts.append(
        f"// Capture preset={args.preset or 'legacy'} expects "
        f"gpuCandidatePromotePolicy={args.promote_policy}; open the matching URL first."
    )
    if args.preset == "stable" and js_bool(args.include_png) == "true":
        parts.append(
            "// Stable URLs normally use debugPreserveDrawingBuffer=false; "
            "PNG capture can be black unless the page was opened with "
            "debugPreserveDrawingBuffer=true or the capture renders immediately before reading."
        )

    preamble = build_preamble(args)
    if preamble:
        parts.append(preamble)

    fixed_time_capture_selected = (
        resolve_step114_capture_contract(args.step) is not None
        and any(label in args.step for label in (
            "step114_fix8",
            "step114_fix9",
            "step114_fix10",
        ))
    )
    if args.include_png and fixed_time_capture_selected:
        parts.append(build_png_command(args))

    if args.include_source_compare:
        parts.append(build_source_compare_command(args))

    if args.include_dryrun_visible:
        command = build_dryrun_visible_compare_command(args)
        if command:
            parts.append(command)

    if args.include_sweep:
        command = build_sweep_command(args)
        if command:
            parts.append(command)

    if args.include_visible_record_dryrun:
        command = build_visible_record_dryrun_command(args)
        if command:
            parts.append(command)

    if args.include_raw_visible_record_dryrun:
        command = build_raw_visible_record_dryrun_command(args)
        if command:
            parts.append(command)

    if args.include_webgpu_visible_record_dryrun:
        command = build_webgpu_visible_record_dryrun_command(args)
        if command:
            parts.append(command)

    if args.include_coverage:
        parts.append(build_coverage_command(args))

    if args.include_runtime:
        parts.append(build_runtime_summary_command(args))

    if args.include_visible_compare:
        parts.append(build_visible_compare_command(args))

    if args.include_live_same_state:
        parts.append(build_live_same_state_command(args))

    if args.include_png and not fixed_time_capture_selected:
        parts.append(build_png_command(args))

    if args.include_webgpu_render_state_manifest:
        command = build_webgpu_render_state_manifest_command(args)
        if command:
            parts.append(command)

    initial_presentation_command = build_initial_presentation_evidence_command(args)
    if initial_presentation_command:
        parts.append(initial_presentation_command)

    if args.include_camera_control_debug:
        parts.append(build_camera_control_debug_command(args))

    return "\n\n".join(parts)


def parse_indices(value: str) -> List[int]:
    if not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def option_was_provided(argv: list[str], option_names: tuple[str, ...]) -> bool:
    for item in argv[1:]:
        for name in option_names:
            if item == name or item.startswith(name + "="):
                return True
    return False


def set_if_not_provided(
    args: argparse.Namespace,
    argv: list[str],
    dest: str,
    option_names: tuple[str, ...],
    value: object,
) -> None:
    if not option_was_provided(argv, option_names):
        setattr(args, dest, value)


def apply_preset(args: argparse.Namespace, argv: list[str]) -> None:
    if args.preset is None:
        return

    if args.preset == "stable":
        values = {
            "include_preamble": "false",
            "include_source_compare": "false",
            "include_coverage": "false",
            "include_dryrun_visible": "false",
            "include_sweep": "false",
            "include_visible_record_dryrun": "false",
            "include_raw_visible_record_dryrun": "false",
            "include_webgpu_visible_record_dryrun": "false",
            "include_runtime": "false",
            "include_visible_compare": "false",
            "include_live_same_state": "false",
            "include_png": "false",
            "include_webgpu_render_state_manifest": "false",
            "include_camera_control_debug": "false",
            "promote_policy": "validated-only",
        }
    elif args.preset == "runtime-only":
        values = {
            "include_preamble": "true",
            "include_source_compare": "false",
            "include_coverage": "false",
            "include_dryrun_visible": "false",
            "include_sweep": "false",
            "include_visible_record_dryrun": "false",
            "include_raw_visible_record_dryrun": "false",
            "include_webgpu_visible_record_dryrun": "false",
            "include_runtime": "true",
            "include_visible_compare": "false",
            "include_live_same_state": "false",
            "include_png": "false",
            "include_webgpu_render_state_manifest": "false",
            "include_camera_control_debug": "true",
            "promote_policy": "validated-only",
        }
    elif args.preset == "validation":
        values = {
            "include_preamble": "true",
            "include_source_compare": "true",
            "include_coverage": "true",
            "include_dryrun_visible": "true",
            "include_sweep": "false",
            "include_visible_record_dryrun": "false",
            "include_raw_visible_record_dryrun": "true",
            "include_webgpu_visible_record_dryrun": "false",
            "include_runtime": "true",
            "include_visible_compare": "true",
            "include_live_same_state": "true",
            "include_png": "true",
            "include_webgpu_render_state_manifest": "false",
            "include_camera_control_debug": "true",
            "readback_mode": "sync-debug",
            "raw_visible_record_readback": "sync-debug",
            "raw_visible_record_mode": "packed-like",
            "promote_policy": "validated-only",
        }
    else:
        raise ValueError(f"Unsupported preset: {args.preset}")

    option_names = {
        "include_preamble": ("--include-preamble",),
        "include_source_compare": ("--include-source-compare",),
        "include_coverage": ("--include-coverage",),
        "include_dryrun_visible": ("--include-dryrun-visible",),
        "include_sweep": ("--include-sweep",),
        "include_visible_record_dryrun": ("--include-visible-record-dryrun",),
        "include_raw_visible_record_dryrun": ("--include-raw-visible-record-dryrun",),
        "include_webgpu_visible_record_dryrun": ("--include-webgpu-visible-record-dryrun",),
        "include_runtime": ("--include-runtime",),
        "include_visible_compare": ("--include-visible-compare",),
        "include_live_same_state": ("--include-live-same-state",),
        "include_png": ("--include-png",),
        "include_webgpu_render_state_manifest": (
            "--include-webgpu-render-state-manifest",
        ),
        "include_camera_control_debug": ("--include-camera-control-debug",),
        "readback_mode": ("--readback-mode",),
        "raw_visible_record_readback": ("--raw-visible-record-readback",),
        "raw_visible_record_mode": ("--raw-visible-record-mode",),
        "promote_policy": ("--promote-policy",),
    }

    for dest, value in values.items():
        set_if_not_provided(args, argv, dest, option_names[dest], value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DevTools Console capture commands for 4DGS Viewer."
    )

    parser.add_argument(
        "--step",
        required=True,
        help="Step prefix, e.g. step107_000151_v13.",
    )
    parser.add_argument(
        "--preset",
        choices=["stable", "runtime-only", "validation"],
        default=None,
        help="Optional capture preset. Individual CLI arguments override preset values.",
    )
    parser.add_argument(
        "--source-mode",
        default="screenCoarse",
        choices=["screenCoarse", "range", "visibleSrcIndices"],
        help="Candidate source mode.",
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Optional output JS file path.",
    )

    parser.add_argument(
        "--include-preamble",
        default="true",
        help="Include scheduleRender and wait. Default: true.",
    )
    parser.add_argument(
        "--render-wait-ms",
        type=int,
        default=500,
        help="Wait time after scheduleRender. Default: 500.",
    )
    parser.add_argument(
        "--viewer-data-ready-timeout-ms",
        type=int,
        default=10000,
        help="Wait time for viewer debug raw data readiness. Default: 10000.",
    )

    parser.add_argument(
        "--include-source-compare",
        default="true",
        help="Include candidate source compare capture. Default: true.",
    )
    parser.add_argument(
        "--include-coverage",
        default="true",
        help="Include coverage capture. Default: true.",
    )
    parser.add_argument(
        "--include-dryrun-visible",
        default="true",
        help="Include screenCoarse dry-run visible / packed compare capture. Default: true.",
    )
    parser.add_argument(
        "--include-sweep",
        default="false",
        help="Include Step109 screenCoarse sweep capture. Default: false.",
    )
    parser.add_argument(
        "--include-visible-record-dryrun",
        default="false",
        help="Include Step114 GPU visible fixed-record dry-run capture. Default: false.",
    )
    parser.add_argument(
        "--include-raw-visible-record-dryrun",
        default="false",
        help="Include Step120A raw attribute texture packed-like fixed-record dry-run capture. Default: false.",
    )
    parser.add_argument(
        "--include-webgpu-visible-record-dryrun",
        default="false",
        help="Include Phase 3 WebGPU fixed-record compute dry-run capture. Default: false.",
    )
    parser.add_argument(
        "--include-runtime",
        default="true",
        help="Include runtime, limited draw, and Step111 timing summary captures. Default: true.",
    )
    parser.add_argument(
        "--include-visible-compare",
        default="true",
        help="Include visible compare capture. Default: true.",
    )
    parser.add_argument(
        "--include-live-same-state",
        default="true",
        help="Include live same-state / association / summary capture. Default: true.",
    )
    parser.add_argument(
        "--include-png",
        default="true",
        help="Include canvas PNG capture. Default: true.",
    )
    parser.add_argument(
        "--include-webgpu-render-state-manifest",
        default="false",
        help="Include Step114 WebGPU render-state manifest capture. Default: false.",
    )
    parser.add_argument(
        "--include-camera-control-debug",
        default="false",
        help="Include Step131 camera/control contract debug capture. Default: false.",
    )

    # Common debug values.
    parser.add_argument("--epsilon", default="1e-6")
    parser.add_argument("--max-mismatches", type=int, default=32)
    parser.add_argument("--max-misses", type=int, default=32)
    parser.add_argument("--readback-mode", default="sync-debug")
    parser.add_argument(
        "--promote-policy",
        default="never",
        choices=["never", "compare-ok", "async-ready", "validated-only"],
        help="Expected gpuCandidatePromotePolicy in the already-open viewer URL.",
    )

    # Range.
    parser.add_argument("--range-start", type=int, default=0)
    parser.add_argument("--range-count", type=int, default=65536)

    # screenCoarse.
    parser.add_argument("--screen-coarse-max-count", type=int, default=65536)
    parser.add_argument("--screen-coarse-min-radius-px", type=float, default=0.25)
    parser.add_argument("--screen-coarse-require-in-viewport", default="true")
    parser.add_argument("--screen-coarse-depth-mode", default="positive")

    # Step109 sweep.
    parser.add_argument("--sweep-max-counts", default="4096,8192,16384,32768,65536")
    parser.add_argument("--sweep-min-radius-px", default="0.25")
    parser.add_argument("--sweep-require-in-viewport", default="true")
    parser.add_argument("--sweep-depth-modes", default="positive")
    parser.add_argument("--sweep-include-case-results", default="false")

    # Step114 visible fixed-record dry-run.
    parser.add_argument(
        "--cuda-reference-manifest",
        default=None,
        help="CUDA render-state manifest used to seed Step114 canonical srcIndex capture.",
    )
    parser.add_argument(
        "--canonical-src-indices",
        type=parse_int_list,
        default=[],
        help="Comma-separated canonical srcIndex set for Step114 direct comparison.",
    )
    parser.add_argument("--visible-record-readback", default="sync-debug")
    parser.add_argument("--visible-record-max-count", type=int, default=65536)
    parser.add_argument("--raw-visible-record-readback", default="sync-debug")
    parser.add_argument("--raw-visible-record-mode", default="packed-like")
    parser.add_argument("--raw-visible-record-max-count", type=int, default=65536)
    parser.add_argument("--raw-visible-record-epsilon", default="1e-3")
    parser.add_argument("--webgpu-visible-record-max-count", type=int, default=65536)
    parser.add_argument("--webgpu-visible-record-epsilon", default="1e-3")
    parser.add_argument("--webgpu-backend-implementation", default=None)
    parser.add_argument("--expected-runtime", default=None)
    parser.add_argument("--expected-effective-display-runtime", default=None)
    parser.add_argument("--expected-webgpu-backend-mode", default=None)
    parser.add_argument("--expected-webgpu-backend-implementation", default=None)
    parser.add_argument("--expected-webgpu-canvas-presentation", default=None)
    parser.add_argument("--expected-webgpu-viewer-loop-hook", default=None)

    # visibleSrcIndices.
    parser.add_argument("--subset-count", type=int, default=1024)
    parser.add_argument("--filter-mode", default="all-valid")

    # Live same-state fixed values used in current workflow.
    parser.add_argument("--pixel-x", type=int, default=655)
    parser.add_argument("--pixel-y", type=int, default=363)
    parser.add_argument(
        "--indices",
        type=parse_indices,
        default=parse_indices("2718004,2735566,1181906,2471537"),
        help="Comma-separated original indices.",
    )
    parser.add_argument("--max-items", type=int, default=2048)
    parser.add_argument("--max-entries", type=int, default=2048)
    parser.add_argument("--include-all-entries", default="true")

    args = parser.parse_args()
    apply_preset(args, sys.argv)

    # Convert include flags to bool after parsing.
    args.include_preamble = js_bool(args.include_preamble) == "true"
    args.include_source_compare = js_bool(args.include_source_compare) == "true"
    args.include_dryrun_visible = js_bool(args.include_dryrun_visible) == "true"
    args.include_sweep = js_bool(args.include_sweep) == "true"
    args.include_visible_record_dryrun = js_bool(args.include_visible_record_dryrun) == "true"
    args.include_raw_visible_record_dryrun = js_bool(args.include_raw_visible_record_dryrun) == "true"
    args.include_webgpu_visible_record_dryrun = js_bool(args.include_webgpu_visible_record_dryrun) == "true"
    args.include_coverage = js_bool(args.include_coverage) == "true"
    args.include_runtime = js_bool(args.include_runtime) == "true"
    args.include_visible_compare = js_bool(args.include_visible_compare) == "true"
    args.include_live_same_state = js_bool(args.include_live_same_state) == "true"
    args.include_png = js_bool(args.include_png) == "true"
    args.include_webgpu_render_state_manifest = (
        js_bool(args.include_webgpu_render_state_manifest) == "true"
    )
    args.include_camera_control_debug = js_bool(args.include_camera_control_debug) == "true"

    return args


def main() -> int:
    args = parse_args()
    commands = build_commands(args)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(commands + "\n", encoding="utf-8")

    print(commands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
