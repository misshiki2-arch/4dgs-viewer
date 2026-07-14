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
import sys
from pathlib import Path
from typing import List


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
    if "step113" in args.step:
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

    return f"""try {{
  var webgpuVisibleRecordDryRunResult =
    await window.gpuViewerDebug.captureWebGpuVisibleRecordDryRunDebug({{
    ensureCurrentFrame: false,
    maxRecords: {args.webgpu_visible_record_max_count},
    epsilon: {args.webgpu_visible_record_epsilon},
    maxMismatches: {args.max_mismatches},{backend_implementation_line}{phase_step_line}{comparison_mode_line}{fixed_reference_camera_mode_line}
  }});

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
    if "step111" in args.step or "step112" in args.step or "step113" in args.step:
        return f"""var pngCaptureResult = await window.gpuViewerDebug.saveCurrentCanvasPng({{
  name: {quote(args.step + '_canvas.png')},
  captureSource: 'last-valid-webgpu-tile-compositor-output',
  fallbackToCanvasOnCaptureFailure: false,
  renderBeforeCapture: false
}});
var {{ blob: _pngBlob, ...pngCaptureStatus }} = pngCaptureResult ?? {{}};
await window.gpuViewerDebug.downloadJsonDebug(
  pngCaptureStatus,
  {quote(args.step + '_png_capture_status.json')}
);"""
    return f"""await window.gpuViewerDebug.saveCurrentCanvasPng({{
  name: {quote(args.step + '_canvas.png')},
  renderBeforeCapture: false
}});"""


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


def build_preamble(args: argparse.Namespace) -> str:
    if not args.include_preamble:
        return ""

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

    if args.include_png:
        parts.append(build_png_command(args))

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
    parser.add_argument("--visible-record-readback", default="sync-debug")
    parser.add_argument("--visible-record-max-count", type=int, default=65536)
    parser.add_argument("--raw-visible-record-readback", default="sync-debug")
    parser.add_argument("--raw-visible-record-mode", default="packed-like")
    parser.add_argument("--raw-visible-record-max-count", type=int, default=65536)
    parser.add_argument("--raw-visible-record-epsilon", default="1e-3")
    parser.add_argument("--webgpu-visible-record-max-count", type=int, default=65536)
    parser.add_argument("--webgpu-visible-record-epsilon", default="1e-3")
    parser.add_argument("--webgpu-backend-implementation", default=None)

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
