#!/usr/bin/env python3
"""
summarize_step_json.py

Summarize saved Step JSON files for 4DGS Viewer debugging.

Typical use:
  python3 tools/summarize_step_json.py \
    --dir /home/demo/work/json \
    --prefix step107_000151_v13

With JSON output:
  python3 tools/summarize_step_json.py \
    --dir /home/demo/work/json \
    --prefix step107_000151_v13 \
    --json /home/demo/work/json/step107_000151_v13_summary_extract.json

Purpose:
- Extract key fields from Step JSON files.
- Summarize candidate source, coverage, runtime, fallback, visible compare,
  association, live same-state, and summary JSON.
- Avoid depending on jq.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


KNOWN_SUFFIXES = [
    "gpu_candidate_screen_coarse_compare",
    "gpu_candidate_screen_coarse_dryrun_visible_compare",
    "gpu_candidate_screen_coarse_sweep_summary",
    "gpu_visible_record_dryrun_compare",
    "gpu_raw_visible_record_dryrun_compare",
    "webgpu_visible_record_dryrun_compare",
    "webgpu_visible_record_dryrun_capture_status",
    "fixed_condition_visual_comparison",
    "png_capture_status",
    "gpu_candidate_source_compare",
    "gpu_candidate_coverage",
    "gpu_candidate_runtime_summary",
    "limited_draw_summary",
    "visible_compare",
    "live_same_state",
    "association",
    "summary",
]

WEBGPU_VISIBLE_RECORD_DRY_RUN_SCHEMA_VERSION = (
    "phase3-step2-webgpu-visible-record-dry-run-v1"
)


def load_json_if_exists(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        return None, "missing"
    if path.stat().st_size == 0:
        return None, "empty file"
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
        if isinstance(value, dict):
            return value, None
        return {"_value": value}, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def get_path(obj: Any, paths: Iterable[str], default: Any = None) -> Any:
    """
    Read the first existing dotted path from a nested dict/list object.

    Example:
      get_path(data, ["runtimeSummary.requestedRuntime", "requestedRuntime"])
    """
    for path in paths:
        cur = obj
        ok = True
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            elif isinstance(cur, list):
                try:
                    cur = cur[int(part)]
                except Exception:
                    ok = False
                    break
            else:
                ok = False
                break
        if ok:
            return cur
    return default


def compact_list(value: Any, max_items: int = 8) -> Any:
    if not isinstance(value, list):
        return value
    if len(value) <= max_items:
        return value
    return value[:max_items] + [f"...({len(value) - max_items} more)"]


def numeric_value(value: Any, default: float = 0) -> float:
    try:
        if value is None or isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def compare_frame_identity(
    left: Any,
    right: Any,
    *,
    required_keys: Iterable[str],
) -> Dict[str, Any]:
    left = left if isinstance(left, dict) else {}
    right = right if isinstance(right, dict) else {}
    keys = list(required_keys)
    mismatched_keys = []
    missing_keys = []
    for key in keys:
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value is None or right_value is None:
            missing_keys.append(key)
        elif left_value != right_value:
            mismatched_keys.append(key)
    return {
        "matches": not mismatched_keys and not missing_keys,
        "mismatchedKeys": mismatched_keys,
        "missingKeys": missing_keys,
        "requiredKeys": keys,
    }


def _png_channels_for_color_type(color_type: int) -> int:
    return {
        0: 1,  # grayscale
        2: 3,  # truecolor
        4: 2,  # grayscale + alpha
        6: 4,  # truecolor + alpha
    }.get(color_type, 0)


def _unfilter_png_scanlines(
    raw: bytes,
    *,
    width: int,
    height: int,
    channels: int,
    bit_depth: int,
) -> List[bytes]:
    if bit_depth != 8 or width <= 0 or height <= 0 or channels <= 0:
        raise ValueError("unsupported PNG format for pixel diagnostics")
    stride = width * channels
    bpp = channels
    rows: List[bytes] = []
    offset = 0
    previous = bytearray(stride)
    for _ in range(height):
        if offset >= len(raw):
            raise ValueError("truncated PNG scanline data")
        filter_type = raw[offset]
        offset += 1
        scanline = bytearray(raw[offset : offset + stride])
        offset += stride
        if len(scanline) != stride:
            raise ValueError("truncated PNG scanline")
        for i, value in enumerate(scanline):
            left = scanline[i - bpp] if i >= bpp else 0
            up = previous[i]
            up_left = previous[i - bpp] if i >= bpp else 0
            if filter_type == 0:
                restored = value
            elif filter_type == 1:
                restored = value + left
            elif filter_type == 2:
                restored = value + up
            elif filter_type == 3:
                restored = value + ((left + up) // 2)
            elif filter_type == 4:
                p = left + up - up_left
                pa = abs(p - left)
                pb = abs(p - up)
                pc = abs(p - up_left)
                predictor = left if pa <= pb and pa <= pc else up if pb <= pc else up_left
                restored = value + predictor
            else:
                raise ValueError(f"unsupported PNG filter type {filter_type}")
            scanline[i] = restored & 0xFF
        rows.append(bytes(scanline))
        previous = scanline
    return rows


def extract_png_capture_diagnostic(path: Path) -> Dict[str, Any]:
    diagnostic: Dict[str, Any] = {
        "pngCaptured": path.exists(),
        "pngPath": str(path),
        "pngWidth": None,
        "pngHeight": None,
        "pngNonzeroRgb": None,
        "pngNonblackRatio": None,
        "pngNonblackBBox": None,
        "pngMaxRgb": None,
        "pngSha256": None,
        "pngDiagnosticError": None,
    }
    if not path.exists():
        diagnostic["pngDiagnosticError"] = "missing-png"
        return diagnostic
    data = path.read_bytes()
    diagnostic["pngSha256"] = hashlib.sha256(data).hexdigest()
    try:
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("not a PNG file")
        offset = 8
        width = height = bit_depth = color_type = None
        idat_chunks: List[bytes] = []
        while offset + 8 <= len(data):
            length = struct.unpack(">I", data[offset : offset + 4])[0]
            chunk_type = data[offset + 4 : offset + 8]
            chunk_data = data[offset + 8 : offset + 8 + length]
            offset += 12 + length
            if chunk_type == b"IHDR":
                width, height, bit_depth, color_type = struct.unpack(
                    ">IIBB", chunk_data[:10]
                )
            elif chunk_type == b"IDAT":
                idat_chunks.append(chunk_data)
            elif chunk_type == b"IEND":
                break
        if width is None or height is None or bit_depth is None or color_type is None:
            raise ValueError("missing PNG IHDR")
        channels = _png_channels_for_color_type(color_type)
        if channels <= 0:
            raise ValueError(f"unsupported PNG color type {color_type}")
        rows = _unfilter_png_scanlines(
            zlib.decompress(b"".join(idat_chunks)),
            width=width,
            height=height,
            channels=channels,
            bit_depth=bit_depth,
        )
        nonzero_rgb = 0
        max_rgb = 0
        min_x = min_y = None
        max_x = max_y = None
        for y, row in enumerate(rows):
            for x in range(width):
                base = x * channels
                if color_type == 0:
                    r = g = b = row[base]
                    a = 255
                elif color_type == 2:
                    r, g, b = row[base : base + 3]
                    a = 255
                elif color_type == 4:
                    r = g = b = row[base]
                    a = row[base + 1]
                else:
                    r, g, b, a = row[base : base + 4]
                pixel_max = max(r, g, b)
                max_rgb = max(max_rgb, pixel_max)
                if a != 0 and pixel_max > 0:
                    nonzero_rgb += 1
                    min_x = x if min_x is None else min(min_x, x)
                    min_y = y if min_y is None else min(min_y, y)
                    max_x = x if max_x is None else max(max_x, x)
                    max_y = y if max_y is None else max(max_y, y)
        diagnostic.update(
            {
                "pngWidth": width,
                "pngHeight": height,
                "pngNonzeroRgb": nonzero_rgb,
                "pngNonblackRatio": (
                    nonzero_rgb / (width * height) if width and height else None
                ),
                "pngNonblackBBox": (
                    [min_x, min_y, max_x, max_y] if nonzero_rgb > 0 else None
                ),
                "pngMaxRgb": max_rgb,
            }
        )
    except Exception as exc:  # noqa: BLE001
        diagnostic["pngDiagnosticError"] = str(exc)
    return diagnostic


def extract_fixed_condition_visual_comparison(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schemaVersion": get_path(data, ["schemaVersion"]),
        "comparisonExecuted": True,
        "referenceImagePath": get_path(data, ["referenceImagePath", "a"]),
        "webgpuPngPath": get_path(data, ["webgpuPngPath", "b"]),
        "referenceImageSource": get_path(data, ["referenceImageSource"]),
        "referenceImageSourceKind": get_path(data, ["referenceImageSourceKind"]),
        "webgpuPngSource": get_path(data, ["webgpuPngSource"]),
        "webgpuPngSourceKind": get_path(data, ["webgpuPngSourceKind"]),
        "referenceImageSha256": get_path(data, ["aSha256"]),
        "webgpuPngSha256": get_path(data, ["bSha256"]),
        "sameSha256": get_path(data, ["sameSha256"]),
        "imageSizeMatch": get_path(data, ["sameSize"]),
        "comparisonChannelMode": get_path(data, ["comparisonChannelMode"]),
        "comparisonConditionReady": get_path(data, ["comparisonConditionReady"]),
        "pixelMae": get_path(data, ["mae"]),
        "pixelRmse": get_path(data, ["rmse"]),
        "maxAbsDifference": get_path(data, ["maxAbsError"]),
        "differingPixelCount": get_path(data, ["differentPixelCountAnyChannel"]),
        "differingPixelRatio": get_path(data, ["differentPixelRatioAnyChannel"]),
        "referenceImageStats": get_path(data, ["referenceImageStats"]),
        "webgpuImageStats": get_path(data, ["webgpuImageStats"]),
        "comparisonConditions": get_path(data, ["comparisonConditions"], {}),
        "orientationDiagnostic": get_path(data, ["orientationDiagnostic"], {}),
        "visualMismatchClassification": get_path(data, ["visualMismatchClassification"]),
        "absDiffImage": get_path(data, ["absDiffImage"]),
        "error": get_path(data, ["error"]),
    }


def classify_reference_source_kind(path_or_source: Optional[str]) -> str:
    if not path_or_source:
        return "unknown"
    normalized = str(path_or_source).replace("\\", "/")
    if "/outputs/" in normalized and "cuda_reference" in normalized:
        if normalized.endswith("_render.png"):
            return "cuda-reference-render-output"
        return "cuda-reference-derived-output"
    if "/data/4dgs_sph_scene/images/" in normalized:
        return "dataset-fixed-reference-image"
    if normalized.endswith("_gt.png"):
        return "dataset-ground-truth-image"
    return "explicit-reference-image"


def related_step_prefix(prefix: str, step_number: int) -> Optional[str]:
    match = re.match(r"^(?P<head>.*?step)(?P<step>\d+)(?:_fix\d+)?(?P<tail>_.*)$", prefix)
    if not match:
        return None
    return f"{match.group('head')}{step_number}{match.group('tail')}"


def build_step108_output_regression_diagnostic(
    base_dir: Path,
    prefix: str,
    current_output_diagnostic: Dict[str, Any],
) -> Dict[str, Any]:
    related: Dict[int, Dict[str, Any]] = {}
    for step_number in (107, 108, 109):
        related_prefix = related_step_prefix(prefix, step_number)
        if related_prefix is None:
            related[step_number] = {
                "prefix": None,
                "diagnostic": {"pngCaptured": False, "pngDiagnosticError": "prefix-unavailable"},
            }
            continue
        related[step_number] = {
            "prefix": related_prefix,
            "diagnostic": extract_png_capture_diagnostic(
                base_dir / f"{related_prefix}_canvas.png"
            ),
        }

    def png_nonblank(step_number: int) -> Optional[bool]:
        value = related[step_number]["diagnostic"].get("pngNonzeroRgb")
        return value > 0 if isinstance(value, int) else None

    step107_nonblank = png_nonblank(107)
    step108_nonblank = png_nonblank(108)
    step109_nonblank = png_nonblank(109)
    regression_detected = step107_nonblank is True and step108_nonblank is False
    presentation_nonblank = current_output_diagnostic.get("presentationOutputNonblank")
    saved_matches_runtime = current_output_diagnostic.get("savedPngMatchesRuntimeOutput")
    if regression_detected and presentation_nonblank is True and saved_matches_runtime is False:
        root_cause_candidate = (
            "saved-png-capture-target-or-default-framebuffer-mismatch-after-step108"
        )
    elif regression_detected:
        root_cause_candidate = "step108-saved-png-output-regression"
    else:
        root_cause_candidate = "no-step107-to-step108-png-regression-detected"

    evidence = {
        "step107Prefix": related[107]["prefix"],
        "step108Prefix": related[108]["prefix"],
        "step109Prefix": related[109]["prefix"],
        "step107PngSha256": related[107]["diagnostic"].get("pngSha256"),
        "step108PngSha256": related[108]["diagnostic"].get("pngSha256"),
        "step109PngSha256": related[109]["diagnostic"].get("pngSha256"),
        "step107PngNonzeroRgb": related[107]["diagnostic"].get("pngNonzeroRgb"),
        "step108PngNonzeroRgb": related[108]["diagnostic"].get("pngNonzeroRgb"),
        "step109PngNonzeroRgb": related[109]["diagnostic"].get("pngNonzeroRgb"),
        "step108AndStep109PngShaMatch": (
            related[108]["diagnostic"].get("pngSha256") is not None
            and related[108]["diagnostic"].get("pngSha256")
            == related[109]["diagnostic"].get("pngSha256")
        ),
        "gitDiffInterpretation": (
            "Step107-to-Step108 diff added CUDA camera evidence and Step108 capture mode; "
            "saveCurrentCanvasPng itself is unchanged, so black PNG is most consistent with "
            "capture/default-framebuffer target or timing mismatch rather than camera-axis tuning."
        ),
    }
    return {
        "step108OutputRegressionDetected": regression_detected,
        "step108OutputRegressionRootCauseCandidate": root_cause_candidate,
        "step108OutputRegressionEvidence": evidence,
        "step107PngNonblank": step107_nonblank,
        "step108PngNonblank": step108_nonblank,
        "step109PngNonblank": step109_nonblank,
        "relatedPngDiagnostics": {
            "step107": related[107],
            "step108": related[108],
            "step109": related[109],
        },
    }


def detect_webgpu_error_subtypes(*sources: Any) -> Dict[str, bool]:
    text_parts: List[str] = []
    for source in sources:
        if source is None:
            continue
        if isinstance(source, str):
            text_parts.append(source)
            continue
        try:
            text_parts.append(json.dumps(source, ensure_ascii=False))
        except (TypeError, ValueError):
            text_parts.append(str(source))
    error_text = "\n".join(text_parts)
    lower_text = error_text.lower()
    return {
        "wgslParseErrorDetected": (
            "wgsl" in lower_text
            and ("parse error" in lower_text or "error while parsing wgsl" in lower_text)
        ),
        "shaderModuleInvalidDetected": (
            "invalid shadermodule" in lower_text
            or "shadermodule invalid" in lower_text
        ),
        "computePipelineInvalidDetected": (
            "invalid computepipeline" in lower_text
            or "computepipeline invalid" in lower_text
        ),
        "bindGroupInvalidDetected": (
            "invalid bindgroup" in lower_text
            or "bindgroup invalid" in lower_text
        ),
        "invalidCommandBufferDetected": (
            "invalid commandbuffer" in lower_text
            or "invalid command buffer" in lower_text
        ),
        "queueSubmitFailureDetected": (
            "queue.submit" in lower_text
            or "queue submit" in lower_text
            or "submit failed" in lower_text
        ),
        "runtimeFatalErrorDetected": (
            "runtime fatal" in lower_text
            or "fatal error" in lower_text
            or "uncaught" in lower_text
        ),
    }


def comparison_metric_delta(
    comparison_result: Optional[Dict[str, Any]],
    baseline_result: Optional[Dict[str, Any]],
    key: str,
) -> Optional[Dict[str, Any]]:
    current = get_path(comparison_result, [key])
    baseline = get_path(baseline_result, [key])
    if current is None or baseline is None:
        return None
    try:
        current_value = float(current)
        baseline_value = float(baseline)
    except (TypeError, ValueError):
        return None
    delta = current_value - baseline_value
    if abs(delta) <= 1e-12:
        direction = "unchanged"
    elif delta < 0:
        direction = "decreased"
    else:
        direction = "increased"
    return {
        "current": current_value,
        "baseline": baseline_value,
        "delta": delta,
        "direction": direction,
    }


def classify_camera_aware_input_source_kind(
    source_mode: Any,
    runtime_visible_sample_count: Any,
    normal_visible_sample_count: Any,
) -> str:
    source_mode_text = str(source_mode or "")
    if "webgpu-visible-record" in source_mode_text:
        return "visible-record"
    if "validation-assisted" in source_mode_text:
        return "validation-assisted-bridge"
    if "step40-constrained-display-adapter" in source_mode_text:
        return "step40-selected-samples"
    if runtime_visible_sample_count and normal_visible_sample_count:
        return "mixed"
    if normal_visible_sample_count:
        return "selected-samples"
    return "unavailable"


def describe_camera_aware_input_lineage(source_kind: str) -> str:
    if source_kind == "visible-record":
        return (
            "WebGPU visible-record px/py/depth plus reference-assisted colorAlpha "
            "were used as normal-backend camera-aware visible samples"
        )
    if source_kind == "step40-selected-samples":
        return (
            "visible-record produced no valid visible samples; Step40 constrained-display "
            "selector-selected samples were enlarged into camera/projection-aware patches"
        )
    if source_kind == "validation-assisted-bridge":
        return (
            "visible-record produced no valid visible samples; screenCoarse candidate "
            "state positions were projected with the viewer WebGPU projection contract "
            "and used as validation-assisted bridge samples"
        )
    if source_kind == "mixed":
        return (
            "visible-record samples and selected samples both contributed to the "
            "camera-aware visible output input"
        )
    if source_kind == "selected-samples":
        return (
            "normal backend used selector-selected samples as the camera-aware visible "
            "output input"
        )
    return "camera-aware visible output input was unavailable"


def build_step76_many_camera_aware_visible_summary(
    summary: Dict[str, Any],
    webgpu_camera_aware_visible_output: Dict[str, Any],
    webgpu_visible_record_camera_aware_visible_output: Dict[str, Any],
    validation_assisted_camera_aware_visible_output: Dict[str, Any],
    webgpu_normal_backend_frame_implementation: Dict[str, Any],
    webgpu_normal_backend_frame_implementation_validation: Dict[str, Any],
) -> Dict[str, Any]:
    runtime_contract = get_path(
        webgpu_visible_record_camera_aware_visible_output,
        ["contract"],
        {},
    )
    bridge_contract = get_path(
        validation_assisted_camera_aware_visible_output,
        ["contract"],
        {},
    )
    selected_contract = get_path(
        webgpu_camera_aware_visible_output,
        ["contract"],
        {},
    )
    normal_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["cameraAwareVisibleOutputContract"],
        {},
    )
    sample_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["sampleResourceLifecycleContract"],
        {},
    )
    color_surface_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["colorOutputSurfaceLifecycleContract"],
        {},
    )
    presentation_bridge_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["presentationBridgeContract"],
        {},
    )
    guarded_presentation_adapter_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["guardedPresentationAdapterContract"],
        {},
    )
    source_mode = get_path(normal_contract, ["sourceMode"])
    input_source_kind = get_path(
        normal_contract,
        ["inputSourceKind"],
        get_path(selected_contract, ["inputSourceKind"]),
    )
    if not input_source_kind:
        input_source_kind = classify_camera_aware_input_source_kind(
            source_mode,
            get_path(runtime_contract, ["sampleCount"], 0),
            get_path(normal_contract, ["sampleCount"], 0),
        )
    camera_sample_count = get_path(
        normal_contract,
        ["sampleCount"],
        get_path(webgpu_normal_backend_frame_implementation, ["visibleOutputSampleCount"], 0),
    )
    current_texture_ready = get_path(
        webgpu_normal_backend_frame_implementation_validation,
        ["currentTextureConnectionReady"],
    )
    current_texture_readback_matches = get_path(
        presentation_bridge_contract,
        ["currentTextureReadbackMatchesAdapterOutput"],
    )
    webgl2_hybrid_prevented = get_path(
        webgpu_normal_backend_frame_implementation_validation,
        ["webgl2HybridRenderingPrevented"],
    )
    fallback_samples_mixed = get_path(
        normal_contract,
        ["fallbackSamplesMixed"],
        get_path(presentation_bridge_contract, ["fallbackSamplesMixed"]),
    )
    no_fallback_mixing = get_path(
        webgpu_normal_backend_frame_implementation_validation,
        ["noFallbackMixing"],
    )
    source_classification = get_path(
        normal_contract,
        ["sourceClassification"],
        get_path(selected_contract, ["sourceClassification"]),
    )
    consumed_source_kind = get_path(
        normal_contract,
        ["consumedSourceKind"],
        input_source_kind,
    )
    consumed_source_lineage = get_path(
        normal_contract,
        ["consumedSourceLineage"],
        get_path(
            normal_contract,
            ["inputSourceLineage"],
            describe_camera_aware_input_lineage(str(input_source_kind)),
        ),
    )
    consumed_source_classification = get_path(
        normal_contract,
        ["consumedSourceClassification"],
        source_classification,
    )
    consumed_sample_count = get_path(
        normal_contract,
        ["consumedSampleCount"],
        camera_sample_count,
    )
    bridge_sample_count = get_path(bridge_contract, ["sampleCount"], 0)
    visible_input_sample_count = get_path(
        normal_contract, ["visibleInputSampleCount"], camera_sample_count
    )
    rendered_sample_patch_count = get_path(
        normal_contract, ["renderedSamplePatchCount"], camera_sample_count
    )
    bridge_generation_reason = get_path(
        normal_contract,
        ["bridgeGenerationReason"],
        get_path(bridge_contract, ["bridgeGenerationReason"]),
    )
    sample_sources = compact_list(
        get_path(
            sample_contract,
            ["sampleSources"],
            get_path(webgpu_normal_backend_frame_implementation, ["sampleSources"], []),
        )
    )
    bridge_sources_consumed = any(
        "validationAssistedScreenCoarseBridge" in str(source)
        for source in sample_sources
    )
    step40_sources_consumed = any(
        "webgpuConstrainedDisplayAdapterDryRunComparison" in str(source)
        for source in sample_sources
    )
    if (
        input_source_kind == "validation-assisted-bridge"
        and not bridge_sources_consumed
        and step40_sources_consumed
    ):
        consumed_source_kind = "step40-selected-samples"
        consumed_source_lineage = (
            "normal backend sample buffer consumed Step40 constrained-display "
            "selector-selected samples; validation-assisted bridge samples were "
            "not present in the saved capture"
        )
        consumed_source_classification = "true-native-bounded-sample"
    bridge_path_consumed = (
        input_source_kind == "validation-assisted-bridge"
        and consumed_source_kind == "validation-assisted-bridge"
        and consumed_source_classification == "bridge"
        and bool(bridge_sample_count)
        and bool(visible_input_sample_count)
        and bool(rendered_sample_patch_count)
        and bool(consumed_sample_count)
        and bridge_sources_consumed
    )
    phase_step = get_path(summary, ["phaseStep"])
    applies_to_step76 = phase_step == "phase3-step76"
    success = (
        applies_to_step76
        and
        get_path(normal_contract, ["cameraAwareVisibleOutputReady"]) is True
        and bridge_path_consumed
        and bool(camera_sample_count)
        and current_texture_ready is True
        and current_texture_readback_matches is True
        and webgl2_hybrid_prevented is True
        and fallback_samples_mixed is False
        and no_fallback_mixing is True
    )
    visible_record_valid_count = get_path(
        runtime_contract,
        ["validRecordCount"],
        get_path(summary, ["validRecordCount"], 0),
    )
    zero_reason = None
    if success and visible_record_valid_count == 0:
        zero_reason = (
            "visible-record valid samples remained 0; Step76 used the "
            f"{input_source_kind} input as an explicit {source_classification} path, "
            "projected through the viewer camera/projection contract and rendered "
            "through the normal WebGPU sample buffer/currentTexture chain"
        )
    blocked_reason = None
    if not success:
        if not applies_to_step76:
            blocked_reason = "summary-phase-step-is-not-phase3-step76"
        elif not bridge_sample_count:
            blocked_reason = (
                "validation-assisted bridge sample count is 0; "
                f"bridgeGenerationReason={bridge_generation_reason or 'unavailable'}"
            )
        elif not bridge_path_consumed:
            blocked_reason = (
                "validation-assisted bridge was not the nonzero normal-backend "
                "consumed sample source; Step76 fix1 requires bridge samples, not "
                "Step40 selected-sample fallback, for success"
            )
        elif get_path(normal_contract, ["cameraAwareVisibleOutputReady"]) is not True:
            blocked_reason = "camera-aware-visible-output-contract-not-ready"
        elif current_texture_ready is not True:
            blocked_reason = "currentTexture-connection-not-ready"
        elif current_texture_readback_matches is not True:
            blocked_reason = "currentTexture-readback-did-not-match-adapter-output"
        elif webgl2_hybrid_prevented is not True:
            blocked_reason = "webgl2-hybrid-rendering-not-prevented"
        elif fallback_samples_mixed is not False or no_fallback_mixing is not True:
            blocked_reason = "fallback-samples-mixed-or-not-proven-suppressed"
    return {
        "step76Decision": "success" if success else "blocked",
        "step76BlockedReason": blocked_reason,
        "selectedApproach": get_path(normal_contract, ["selectedApproach"]),
        "phaseStep": phase_step,
        "step76SummaryApplies": applies_to_step76,
        "cameraAwareVisibleOutputReady": get_path(
            normal_contract, ["cameraAwareVisibleOutputReady"]
        ),
        "inputSourceKind": input_source_kind,
        "inputSourceLineage": get_path(
            normal_contract,
            ["inputSourceLineage"],
            describe_camera_aware_input_lineage(str(input_source_kind)),
        ),
        "sourceClassification": source_classification,
        "consumedSourceKind": consumed_source_kind,
        "consumedSourceLineage": consumed_source_lineage,
        "consumedSourceClassification": consumed_source_classification,
        "consumedSampleCount": consumed_sample_count,
        "bridgePathConsumedByNormalBackend": bridge_path_consumed,
        "sourceMode": source_mode,
        "visibleRecordValidCount": visible_record_valid_count,
        "runtimeVisibleRecordSampleCount": get_path(
            runtime_contract, ["sampleCount"], 0
        ),
        "validationAssistedBridgeSampleCount": get_path(
            bridge_contract, ["sampleCount"], 0
        ),
        "bridgeGeneratedSampleCount": get_path(
            normal_contract,
            ["bridgeGeneratedSampleCount"],
            get_path(bridge_contract, ["bridgeGeneratedSampleCount"]),
        ),
        "strictProjectedSampleCount": get_path(
            normal_contract,
            ["strictProjectedSampleCount"],
            get_path(bridge_contract, ["strictProjectedSampleCount"]),
        ),
        "bridgeProjectionFallbackCount": get_path(
            normal_contract,
            ["bridgeProjectionFallbackCount"],
            get_path(bridge_contract, ["bridgeProjectionFallbackCount"]),
        ),
        "bridgeInvalidStateFallbackCount": get_path(
            normal_contract,
            ["bridgeInvalidStateFallbackCount"],
            get_path(bridge_contract, ["bridgeInvalidStateFallbackCount"]),
        ),
        "bridgeProjectionRejectedCount": get_path(
            normal_contract,
            ["bridgeProjectionRejectedCount"],
            get_path(bridge_contract, ["bridgeProjectionRejectedCount"]),
        ),
        "bridgeGenerationReason": bridge_generation_reason,
        "visibleInputSampleCount": visible_input_sample_count,
        "renderedSamplePatchCount": rendered_sample_patch_count,
        "cameraAwareVisibleSampleCount": camera_sample_count,
        "selectedSamplePatchCount": get_path(sample_contract, ["sampleCount"]),
        "enlargedPatchPixelCount": get_path(
            color_surface_contract, ["writtenPixelCount"]
        ),
        "outputPointRadiusPx": get_path(normal_contract, ["outputPointRadiusPx"]),
        "debugFillUsed": get_path(normal_contract, ["debugFillUsed"], False),
        "usesViewerCameraProjection": get_path(
            normal_contract, ["visibleOutputUsesCameraProjection"]
        ),
        "cameraProjectionDerivedPositions": get_path(
            normal_contract, ["cameraProjectionDerivedPositions"]
        ),
        "schedulerOwnedPath": (
            get_path(normal_contract, ["visibleOutputUsesSchedulerOwnedFramePath"])
            is True
            and get_path(normal_contract, ["schedulerFramePresentationBoundaryReady"])
            is True
            and get_path(normal_contract, ["schedulerOwnsFrameRequest"]) is True
        ),
        "guardedPresentationAdapterReady": get_path(
            guarded_presentation_adapter_contract,
            ["guardedPresentationAdapterReady"],
        ),
        "presentationTargetReadable": get_path(
            guarded_presentation_adapter_contract,
            ["presentationTargetReadable"],
        ),
        "presentationTargetMatchesExpected": get_path(
            guarded_presentation_adapter_contract,
            ["presentationTargetMatchesExpected"],
        ),
        "presentationTargetMaxAbsDiff": get_path(
            guarded_presentation_adapter_contract,
            ["presentationTargetMaxAbsDiff"],
        ),
        "viewerPresentationBridgeReady": get_path(
            presentation_bridge_contract,
            ["viewerPresentationBridgeReady"],
        ),
        "renderTargetBridgeReady": get_path(
            presentation_bridge_contract,
            ["renderTargetBridgeReady"],
        ),
        "currentTextureConnectionReady": current_texture_ready,
        "currentTextureReadbackMatchesAdapterOutput": current_texture_readback_matches,
        "currentTextureMaxAbsDiff": get_path(
            presentation_bridge_contract,
            ["currentTextureMaxAbsDiff"],
        ),
        "webgpuExclusiveGuard": get_path(
            webgpu_normal_backend_frame_implementation_validation,
            ["guardAllowed"],
        ),
        "webgl2HybridRenderingPrevented": webgl2_hybrid_prevented,
        "fallbackSamplesMixed": fallback_samples_mixed,
        "noFallbackMixing": no_fallback_mixing,
        "selectedSourceKind": get_path(
            webgpu_normal_backend_frame_implementation, ["selectedSourceKind"]
        ),
        "selectionMode": get_path(
            webgpu_normal_backend_frame_implementation, ["selectionMode"]
        ),
        "sampleSources": sample_sources,
        "bridgeSourcesConsumed": bridge_sources_consumed,
        "trueNativeSuccessClaimed": source_classification == "true-native",
        "visibleRecordZeroSuccessReason": zero_reason,
        "firstValidationFailures": compact_list(
            get_path(
                webgpu_normal_backend_frame_implementation_validation,
                ["firstValidationFailures"],
                [],
            )
        ),
    }


def build_step77_webgpu_owned_visible_summary(
    summary: Dict[str, Any],
    webgpu_owned_camera_aware_visible_output: Dict[str, Any],
    validation_assisted_camera_aware_visible_output: Dict[str, Any],
    webgpu_normal_backend_frame_implementation: Dict[str, Any],
    webgpu_normal_backend_frame_implementation_validation: Dict[str, Any],
) -> Dict[str, Any]:
    selected_contract = get_path(
        webgpu_owned_camera_aware_visible_output,
        ["contract"],
        {},
    )
    generation_summary = get_path(
        webgpu_owned_camera_aware_visible_output,
        ["generationSummary"],
        {},
    )
    bridge_contract = get_path(
        validation_assisted_camera_aware_visible_output,
        ["contract"],
        {},
    )
    normal_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["cameraAwareVisibleOutputContract"],
        {},
    )
    sample_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["sampleResourceLifecycleContract"],
        {},
    )
    color_surface_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["colorOutputSurfaceLifecycleContract"],
        {},
    )
    presentation_bridge_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["presentationBridgeContract"],
        {},
    )
    guarded_presentation_adapter_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["guardedPresentationAdapterContract"],
        {},
    )
    sample_sources = compact_list(
        get_path(
            sample_contract,
            ["sampleSources"],
            get_path(webgpu_normal_backend_frame_implementation, ["sampleSources"], []),
        )
    )
    webgpu_owned_sources_consumed = any(
        "webgpuOwnedScreenCoarseSamples" in str(source)
        for source in sample_sources
    )
    phase_step = get_path(summary, ["phaseStep"])
    webgpu_owned_sample_count = get_path(
        normal_contract,
        ["webgpuOwnedSampleCount"],
        get_path(generation_summary, ["webgpuOwnedSampleCount"], 0),
    )
    consumed_source_kind = get_path(normal_contract, ["consumedSourceKind"])
    consumed_source_classification = get_path(
        normal_contract, ["consumedSourceClassification"]
    )
    camera_sample_count = get_path(
        normal_contract,
        ["sampleCount"],
        get_path(webgpu_normal_backend_frame_implementation, ["visibleOutputSampleCount"], 0),
    )
    current_texture_ready = get_path(
        webgpu_normal_backend_frame_implementation_validation,
        ["currentTextureConnectionReady"],
    )
    current_texture_readback_matches = get_path(
        presentation_bridge_contract,
        ["currentTextureReadbackMatchesAdapterOutput"],
    )
    webgl2_hybrid_prevented = get_path(
        webgpu_normal_backend_frame_implementation_validation,
        ["webgl2HybridRenderingPrevented"],
    )
    fallback_samples_mixed = get_path(
        normal_contract,
        ["fallbackSamplesMixed"],
        get_path(presentation_bridge_contract, ["fallbackSamplesMixed"]),
    )
    no_fallback_mixing = get_path(
        webgpu_normal_backend_frame_implementation_validation,
        ["noFallbackMixing"],
    )
    webgpu_owned_path_consumed = (
        consumed_source_kind == "webgpu-owned-native-compatible-samples"
        and consumed_source_classification == "native-compatible"
        and bool(webgpu_owned_sample_count)
        and bool(camera_sample_count)
        and webgpu_owned_sources_consumed
    )
    success = (
        phase_step == "phase3-step77"
        and get_path(normal_contract, ["cameraAwareVisibleOutputReady"]) is True
        and webgpu_owned_path_consumed
        and current_texture_ready is True
        and current_texture_readback_matches is True
        and webgl2_hybrid_prevented is True
        and fallback_samples_mixed is False
        and no_fallback_mixing is True
    )
    blocked_reason = None
    if not success:
        if phase_step != "phase3-step77":
            blocked_reason = "summary-phase-step-is-not-phase3-step77"
        elif not webgpu_owned_sample_count:
            blocked_reason = "webgpu-owned-sample-count-is-0"
        elif not webgpu_owned_path_consumed:
            blocked_reason = (
                "normal backend did not consume the WebGPU-owned "
                "native-compatible sample batch"
            )
        elif get_path(normal_contract, ["cameraAwareVisibleOutputReady"]) is not True:
            blocked_reason = "camera-aware-visible-output-contract-not-ready"
        elif current_texture_ready is not True:
            blocked_reason = "currentTexture-connection-not-ready"
        elif current_texture_readback_matches is not True:
            blocked_reason = "currentTexture-readback-did-not-match-adapter-output"
        elif webgl2_hybrid_prevented is not True:
            blocked_reason = "webgl2-hybrid-rendering-not-prevented"
        elif fallback_samples_mixed is not False or no_fallback_mixing is not True:
            blocked_reason = "fallback-samples-mixed-or-not-proven-suppressed"
    return {
        "step77Decision": "success" if success else "blocked",
        "step77BlockedReason": blocked_reason,
        "selectedApproach": get_path(normal_contract, ["selectedApproach"]),
        "phaseStep": phase_step,
        "step77SummaryApplies": phase_step == "phase3-step77",
        "cameraAwareVisibleOutputReady": get_path(
            normal_contract, ["cameraAwareVisibleOutputReady"]
        ),
        "inputSourceKind": get_path(normal_contract, ["inputSourceKind"]),
        "inputSourceLineage": get_path(normal_contract, ["inputSourceLineage"]),
        "sourceClassification": get_path(normal_contract, ["sourceClassification"]),
        "consumedSourceKind": consumed_source_kind,
        "consumedSourceLineage": get_path(normal_contract, ["consumedSourceLineage"]),
        "consumedSourceClassification": consumed_source_classification,
        "consumedSampleCount": get_path(normal_contract, ["consumedSampleCount"]),
        "webgpuOwnedSampleGenerationReady": get_path(
            generation_summary, ["webgpuOwnedSampleGenerationReady"]
        ),
        "webgpuOwnedSampleCount": webgpu_owned_sample_count,
        "webgpuOwnedGenerationMode": get_path(
            normal_contract,
            ["webgpuOwnedGenerationMode"],
            get_path(generation_summary, ["sourceMode"]),
        ),
        "webgpuOwnedGenerationReason": get_path(
            normal_contract, ["webgpuOwnedGenerationReason"]
        ),
        "webgpuOwnedProjectionGate": get_path(
            normal_contract,
            ["webgpuOwnedProjectionGate"],
            get_path(generation_summary, ["projectionGate"]),
        ),
        "webgpuOwnedPathConsumedByNormalBackend": webgpu_owned_path_consumed,
        "validationAssistedBridgeSampleCount": get_path(
            normal_contract,
            ["validationAssistedBridgeSampleCount"],
            get_path(bridge_contract, ["sampleCount"], 0),
        ),
        "visibleRecordValidCount": get_path(summary, ["validRecordCount"], 0),
        "visibleInputSampleCount": get_path(normal_contract, ["visibleInputSampleCount"]),
        "renderedSamplePatchCount": get_path(
            normal_contract, ["renderedSamplePatchCount"]
        ),
        "cameraAwareVisibleSampleCount": camera_sample_count,
        "selectedSamplePatchCount": get_path(sample_contract, ["sampleCount"]),
        "enlargedPatchPixelCount": get_path(
            color_surface_contract, ["writtenPixelCount"]
        ),
        "outputPointRadiusPx": get_path(normal_contract, ["outputPointRadiusPx"]),
        "debugFillUsed": get_path(normal_contract, ["debugFillUsed"], False),
        "usesViewerCameraProjection": get_path(
            normal_contract, ["visibleOutputUsesCameraProjection"]
        ),
        "cameraProjectionDerivedPositions": get_path(
            normal_contract, ["cameraProjectionDerivedPositions"]
        ),
        "schedulerOwnedPath": (
            get_path(normal_contract, ["visibleOutputUsesSchedulerOwnedFramePath"])
            is True
            and get_path(normal_contract, ["schedulerFramePresentationBoundaryReady"])
            is True
            and get_path(normal_contract, ["schedulerOwnsFrameRequest"]) is True
        ),
        "guardedPresentationAdapterReady": get_path(
            guarded_presentation_adapter_contract,
            ["guardedPresentationAdapterReady"],
        ),
        "presentationTargetReadable": get_path(
            guarded_presentation_adapter_contract, ["presentationTargetReadable"]
        ),
        "presentationTargetMatchesExpected": get_path(
            guarded_presentation_adapter_contract,
            ["presentationTargetMatchesExpected"],
        ),
        "presentationTargetMaxAbsDiff": get_path(
            guarded_presentation_adapter_contract,
            ["presentationTargetMaxAbsDiff"],
        ),
        "viewerPresentationBridgeReady": get_path(
            presentation_bridge_contract, ["viewerPresentationBridgeReady"]
        ),
        "currentTextureConnectionReady": current_texture_ready,
        "currentTextureReadbackMatchesAdapterOutput": current_texture_readback_matches,
        "currentTextureMaxAbsDiff": get_path(
            presentation_bridge_contract, ["currentTextureMaxAbsDiff"]
        ),
        "webgpuExclusiveGuard": get_path(
            webgpu_normal_backend_frame_implementation_validation,
            ["guardAllowed"],
        ),
        "webgl2HybridRenderingPrevented": webgl2_hybrid_prevented,
        "fallbackSamplesMixed": fallback_samples_mixed,
        "noFallbackMixing": no_fallback_mixing,
        "sampleSources": sample_sources,
        "bridgeStillAvailableAsBaseline": get_path(bridge_contract, ["sampleCount"], 0) > 0,
        "trueNativeSuccessClaimed": get_path(normal_contract, ["sourceClassification"]) == "true-native",
        "nextTrueNativeVisibleRecordBlocker": (
            "visible-record valid samples remain 0; Step77 owns a WebGPU "
            "native-compatible sample generation boundary, but full true-native "
            "visible-record projection/visibility gate remains the next target"
        ),
        "firstValidationFailures": compact_list(
            get_path(
                webgpu_normal_backend_frame_implementation_validation,
                ["firstValidationFailures"],
                [],
            )
        ),
    }


def build_step78_true_visible_record_summary(
    summary: Dict[str, Any],
    webgpu_visible_record_camera_aware_visible_output: Dict[str, Any],
    webgpu_owned_camera_aware_visible_output: Dict[str, Any],
    validation_assisted_camera_aware_visible_output: Dict[str, Any],
    webgpu_normal_backend_frame_implementation: Dict[str, Any],
    webgpu_normal_backend_frame_implementation_validation: Dict[str, Any],
) -> Dict[str, Any]:
    runtime_contract = get_path(
        webgpu_visible_record_camera_aware_visible_output,
        ["contract"],
        {},
    )
    normal_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["cameraAwareVisibleOutputContract"],
        {},
    )
    sample_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["sampleResourceLifecycleContract"],
        {},
    )
    color_surface_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["colorOutputSurfaceLifecycleContract"],
        {},
    )
    presentation_bridge_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["presentationBridgeContract"],
        {},
    )
    guarded_presentation_adapter_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["guardedPresentationAdapterContract"],
        {},
    )
    gate_summary = get_path(summary, ["webgpuVisibleRecordGateSummary"], {})
    bridge_contract = get_path(
        validation_assisted_camera_aware_visible_output,
        ["contract"],
        {},
    )
    owned_generation_summary = get_path(
        webgpu_owned_camera_aware_visible_output,
        ["generationSummary"],
        {},
    )
    sample_sources = compact_list(
        get_path(
            sample_contract,
            ["sampleSources"],
            get_path(webgpu_normal_backend_frame_implementation, ["sampleSources"], []),
        )
    )
    visible_record_sources_consumed = any(
        "webgpuVisibleRecordDryRun.cameraAwareVisibleRecords" in str(source)
        for source in sample_sources
    )
    phase_step = get_path(summary, ["phaseStep"])
    visible_record_valid_count = get_path(
        summary,
        ["validRecordCount"],
        get_path(gate_summary, ["validRecordCount"], 0),
    )
    raw_position_repair_record_count = get_path(
        gate_summary, ["rawPositionRepairRecordCount"], 0
    )
    raw_position_repair_used = bool(raw_position_repair_record_count)
    visible_record_path_classification = get_path(
        normal_contract,
        ["visibleRecordPathClassification"],
        get_path(gate_summary, ["trueVisibleRecordPathClassification"]),
    )
    normal_source_classification = get_path(normal_contract, ["sourceClassification"])
    source_classification = (
        visible_record_path_classification
        if raw_position_repair_used and visible_record_path_classification
        else normal_source_classification
    )
    consumed_source_kind = get_path(normal_contract, ["consumedSourceKind"])
    normal_consumed_source_classification = get_path(
        normal_contract,
        ["consumedSourceClassification"],
    )
    consumed_source_classification = (
        visible_record_path_classification
        if raw_position_repair_used and visible_record_path_classification
        else normal_consumed_source_classification
    )
    camera_sample_count = get_path(
        normal_contract,
        ["sampleCount"],
        get_path(webgpu_normal_backend_frame_implementation, ["visibleOutputSampleCount"], 0),
    )
    current_texture_ready = get_path(
        webgpu_normal_backend_frame_implementation_validation,
        ["currentTextureConnectionReady"],
    )
    current_texture_readback_matches = get_path(
        presentation_bridge_contract,
        ["currentTextureReadbackMatchesAdapterOutput"],
    )
    webgl2_hybrid_prevented = get_path(
        webgpu_normal_backend_frame_implementation_validation,
        ["webgl2HybridRenderingPrevented"],
    )
    fallback_samples_mixed = get_path(
        normal_contract,
        ["fallbackSamplesMixed"],
        get_path(presentation_bridge_contract, ["fallbackSamplesMixed"]),
    )
    no_fallback_mixing = get_path(
        webgpu_normal_backend_frame_implementation_validation,
        ["noFallbackMixing"],
    )
    true_visible_record_consumed = (
        consumed_source_kind == "visible-record"
        and consumed_source_classification
        in ("true-native", "true-native-minimal-visible-record")
        and visible_record_sources_consumed
        and bool(visible_record_valid_count)
        and bool(camera_sample_count)
    )
    state_position_record_count = get_path(gate_summary, ["statePositionRecordCount"])
    full_4d_state_driven_true_native = (
        consumed_source_kind == "visible-record"
        and consumed_source_classification == "true-native"
        and bool(state_position_record_count)
        and not raw_position_repair_used
    )
    success = (
        phase_step == "phase3-step78"
        and get_path(normal_contract, ["cameraAwareVisibleOutputReady"]) is True
        and true_visible_record_consumed
        and current_texture_ready is True
        and current_texture_readback_matches is True
        and webgl2_hybrid_prevented is True
        and fallback_samples_mixed is False
        and no_fallback_mixing is True
    )
    blocked_reason = None
    if not success:
        if phase_step != "phase3-step78":
            blocked_reason = "summary-phase-step-is-not-phase3-step78"
        elif not visible_record_valid_count:
            blocked_reason = "validRecordCount-is-0"
        elif not true_visible_record_consumed:
            blocked_reason = "normal-backend-did-not-consume-true-visible-record-source"
        elif get_path(normal_contract, ["cameraAwareVisibleOutputReady"]) is not True:
            blocked_reason = "camera-aware-visible-output-contract-not-ready"
        elif current_texture_ready is not True:
            blocked_reason = "currentTexture-connection-not-ready"
        elif current_texture_readback_matches is not True:
            blocked_reason = "currentTexture-readback-did-not-match-adapter-output"
        elif webgl2_hybrid_prevented is not True:
            blocked_reason = "webgl2-hybrid-rendering-not-prevented"
        elif fallback_samples_mixed is not False or no_fallback_mixing is not True:
            blocked_reason = "fallback-samples-mixed-or-not-proven-suppressed"
    return {
        "step78Decision": "success" if success else "blocked",
        "step78BlockedReason": blocked_reason,
        "selectedApproach": "A/B-visible-record-gate-repair",
        "phaseStep": phase_step,
        "step78SummaryApplies": phase_step == "phase3-step78",
        "cameraAwareVisibleOutputReady": get_path(
            normal_contract, ["cameraAwareVisibleOutputReady"]
        ),
        "inputSourceKind": get_path(normal_contract, ["inputSourceKind"]),
        "inputSourceLineage": get_path(normal_contract, ["inputSourceLineage"]),
        "sourceClassification": source_classification,
        "consumedSourceKind": consumed_source_kind,
        "consumedSourceLineage": get_path(normal_contract, ["consumedSourceLineage"]),
        "consumedSourceClassification": consumed_source_classification,
        "consumedSampleCount": get_path(normal_contract, ["consumedSampleCount"]),
        "trueVisibleRecordConsumedByNormalBackend": true_visible_record_consumed,
        "validRecordCount": visible_record_valid_count,
        "runtimeVisibleRecordSampleCount": get_path(runtime_contract, ["sampleCount"], 0),
        "statePositionAvailableCount": get_path(
            gate_summary, ["statePositionAvailableCount"]
        ),
        "statePositionUnavailableCount": get_path(
            gate_summary, ["statePositionUnavailableCount"]
        ),
        "statePositionRecordCount": get_path(
            gate_summary, ["statePositionRecordCount"]
        ),
        "rawPositionRepairRecordCount": raw_position_repair_record_count,
        "rawPositionRepairUsed": raw_position_repair_used,
        "projectionGatePassedCount": get_path(
            gate_summary, ["projectionGatePassedCount"]
        ),
        "visibilityGateMode": get_path(gate_summary, ["visibilityGateMode"]),
        "trueVisibleRecordPathReady": get_path(
            gate_summary, ["trueVisibleRecordPathReady"]
        ),
        "trueVisibleRecordPathClassification": get_path(
            gate_summary, ["trueVisibleRecordPathClassification"]
        ),
        "full4DStateDrivenTrueNative": full_4d_state_driven_true_native,
        "full4DStateDrivenTrueNativeClaimed": full_4d_state_driven_true_native,
        "nextFull4DStateGate": get_path(gate_summary, ["nextFull4DStateGate"]),
        "webgpuOwnedNativeCompatibleSampleCount": get_path(
            owned_generation_summary, ["webgpuOwnedSampleCount"], 0
        ),
        "validationAssistedBridgeSampleCount": get_path(
            bridge_contract, ["sampleCount"], 0
        ),
        "visibleInputSampleCount": get_path(normal_contract, ["visibleInputSampleCount"]),
        "renderedSamplePatchCount": get_path(
            normal_contract, ["renderedSamplePatchCount"]
        ),
        "cameraAwareVisibleSampleCount": camera_sample_count,
        "selectedSamplePatchCount": get_path(sample_contract, ["sampleCount"]),
        "enlargedPatchPixelCount": get_path(
            color_surface_contract, ["writtenPixelCount"]
        ),
        "outputPointRadiusPx": get_path(normal_contract, ["outputPointRadiusPx"]),
        "debugFillUsed": get_path(normal_contract, ["debugFillUsed"], False),
        "usesViewerCameraProjection": get_path(
            normal_contract, ["visibleOutputUsesCameraProjection"]
        ),
        "cameraProjectionDerivedPositions": get_path(
            normal_contract, ["cameraProjectionDerivedPositions"]
        ),
        "schedulerOwnedPath": (
            get_path(normal_contract, ["visibleOutputUsesSchedulerOwnedFramePath"])
            is True
            and get_path(normal_contract, ["schedulerFramePresentationBoundaryReady"])
            is True
            and get_path(normal_contract, ["schedulerOwnsFrameRequest"]) is True
        ),
        "guardedPresentationAdapterReady": get_path(
            guarded_presentation_adapter_contract,
            ["guardedPresentationAdapterReady"],
        ),
        "presentationTargetReadable": get_path(
            guarded_presentation_adapter_contract, ["presentationTargetReadable"]
        ),
        "presentationTargetMatchesExpected": get_path(
            guarded_presentation_adapter_contract,
            ["presentationTargetMatchesExpected"],
        ),
        "presentationTargetMaxAbsDiff": get_path(
            guarded_presentation_adapter_contract,
            ["presentationTargetMaxAbsDiff"],
        ),
        "viewerPresentationBridgeReady": get_path(
            presentation_bridge_contract, ["viewerPresentationBridgeReady"]
        ),
        "currentTextureConnectionReady": current_texture_ready,
        "currentTextureReadbackMatchesAdapterOutput": current_texture_readback_matches,
        "currentTextureMaxAbsDiff": get_path(
            presentation_bridge_contract, ["currentTextureMaxAbsDiff"]
        ),
        "webgpuExclusiveGuard": get_path(
            webgpu_normal_backend_frame_implementation_validation,
            ["guardAllowed"],
        ),
        "webgl2HybridRenderingPrevented": webgl2_hybrid_prevented,
        "fallbackSamplesMixed": fallback_samples_mixed,
        "noFallbackMixing": no_fallback_mixing,
        "sampleSources": sample_sources,
        "nativeCompatibleFallbackUsed": consumed_source_kind
        == "webgpu-owned-native-compatible-samples",
        "bridgeFallbackUsed": consumed_source_kind == "validation-assisted-bridge",
        "trueNativeSuccessClaimed": full_4d_state_driven_true_native,
        "firstValidationFailures": compact_list(
            get_path(
                webgpu_normal_backend_frame_implementation_validation,
                ["firstValidationFailures"],
                [],
            )
        ),
    }


def build_step79_4d_state_visible_pipeline_summary(
    summary: Dict[str, Any],
    webgpu_visible_record_camera_aware_visible_output: Dict[str, Any],
    webgpu_owned_camera_aware_visible_output: Dict[str, Any],
    validation_assisted_camera_aware_visible_output: Dict[str, Any],
    webgpu_normal_backend_frame_implementation: Dict[str, Any],
    webgpu_normal_backend_frame_implementation_validation: Dict[str, Any],
) -> Dict[str, Any]:
    runtime_contract = get_path(
        webgpu_visible_record_camera_aware_visible_output,
        ["contract"],
        {},
    )
    normal_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["cameraAwareVisibleOutputContract"],
        {},
    )
    sample_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["sampleResourceLifecycleContract"],
        {},
    )
    color_surface_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["colorOutputSurfaceLifecycleContract"],
        {},
    )
    presentation_bridge_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["presentationBridgeContract"],
        {},
    )
    guarded_presentation_adapter_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["guardedPresentationAdapterContract"],
        {},
    )
    gate_summary = get_path(summary, ["webgpuVisibleRecordGateSummary"], {})
    state_source_contract = get_path(summary, ["webgpu4DStateSourceContract"], {})
    bridge_contract = get_path(
        validation_assisted_camera_aware_visible_output,
        ["contract"],
        {},
    )
    owned_generation_summary = get_path(
        webgpu_owned_camera_aware_visible_output,
        ["generationSummary"],
        {},
    )
    sample_sources = compact_list(
        get_path(
            sample_contract,
            ["sampleSources"],
            get_path(webgpu_normal_backend_frame_implementation, ["sampleSources"], []),
        )
    )
    visible_record_sources_consumed = any(
        "webgpuVisibleRecordDryRun.cameraAwareVisibleRecords" in str(source)
        for source in sample_sources
    )
    phase_step = get_path(summary, ["phaseStep"])
    visible_record_valid_count = get_path(
        summary,
        ["validRecordCount"],
        get_path(gate_summary, ["validRecordCount"], 0),
    )
    consumed_source_kind = get_path(normal_contract, ["consumedSourceKind"])
    consumed_source_classification = get_path(
        normal_contract, ["consumedSourceClassification"]
    )
    camera_sample_count = get_path(
        normal_contract,
        ["sampleCount"],
        get_path(webgpu_normal_backend_frame_implementation, ["visibleOutputSampleCount"], 0),
    )
    current_texture_ready = get_path(
        webgpu_normal_backend_frame_implementation_validation,
        ["currentTextureConnectionReady"],
    )
    current_texture_readback_matches = get_path(
        presentation_bridge_contract,
        ["currentTextureReadbackMatchesAdapterOutput"],
    )
    webgl2_hybrid_prevented = get_path(
        webgpu_normal_backend_frame_implementation_validation,
        ["webgl2HybridRenderingPrevented"],
    )
    fallback_samples_mixed = get_path(
        normal_contract,
        ["fallbackSamplesMixed"],
        get_path(presentation_bridge_contract, ["fallbackSamplesMixed"]),
    )
    no_fallback_mixing = get_path(
        webgpu_normal_backend_frame_implementation_validation,
        ["noFallbackMixing"],
    )
    state_position_record_count = get_path(gate_summary, ["statePositionRecordCount"], 0)
    full_4d_state_position_record_count = get_path(
        gate_summary, ["full4DStatePositionRecordCount"], 0
    )
    state_source_baseline_record_count = get_path(
        gate_summary, ["stateSourceBaselineRecordCount"], 0
    )
    raw_position_repair_record_count = get_path(
        gate_summary, ["rawPositionRepairRecordCount"], 0
    )
    path_classification = get_path(
        gate_summary, ["trueVisibleRecordPathClassification"]
    )
    state_source_ready = get_path(
        state_source_contract,
        ["fourDStateSourceReady"],
        get_path(gate_summary, ["fourDStateSourceReady"]),
    )
    state_source_consumed = (
        consumed_source_kind == "visible-record"
        and visible_record_sources_consumed
        and bool(state_position_record_count)
        and bool(visible_record_valid_count)
        and bool(camera_sample_count)
    )
    full_4d_state_driven_visible_record_path = (
        path_classification == "full-4d-state-driven-visible-record"
        and full_4d_state_position_record_count == visible_record_valid_count
        and raw_position_repair_record_count == 0
    )
    minimal_4d_state_source_visible_record_path = (
        path_classification == "minimal-4d-state-source-visible-record"
        and state_position_record_count > 0
        and raw_position_repair_record_count == 0
    )
    raw_xyz_repair_dependency_reduced = (
        state_position_record_count > 0 and raw_position_repair_record_count == 0
    )
    success = (
        phase_step == "phase3-step79"
        and get_path(normal_contract, ["cameraAwareVisibleOutputReady"]) is True
        and state_source_ready is True
        and state_source_consumed
        and raw_xyz_repair_dependency_reduced
        and current_texture_ready is True
        and current_texture_readback_matches is True
        and webgl2_hybrid_prevented is True
        and fallback_samples_mixed is False
        and no_fallback_mixing is True
    )
    blocked_reason = None
    if not success:
        if phase_step != "phase3-step79":
            blocked_reason = "summary-phase-step-is-not-phase3-step79"
        elif state_source_ready is not True:
            blocked_reason = "4d-state-source-contract-not-ready"
        elif not visible_record_valid_count:
            blocked_reason = "validRecordCount-is-0"
        elif not state_source_consumed:
            blocked_reason = "normal-backend-did-not-consume-4d-state-visible-record-source"
        elif not raw_xyz_repair_dependency_reduced:
            blocked_reason = "visible-record-path-still-depends-on-raw-xyz-repair"
        elif get_path(normal_contract, ["cameraAwareVisibleOutputReady"]) is not True:
            blocked_reason = "camera-aware-visible-output-contract-not-ready"
        elif current_texture_ready is not True:
            blocked_reason = "currentTexture-connection-not-ready"
        elif current_texture_readback_matches is not True:
            blocked_reason = "currentTexture-readback-did-not-match-adapter-output"
        elif webgl2_hybrid_prevented is not True:
            blocked_reason = "webgl2-hybrid-rendering-not-prevented"
        elif fallback_samples_mixed is not False or no_fallback_mixing is not True:
            blocked_reason = "fallback-samples-mixed-or-not-proven-suppressed"
    return {
        "step79Decision": "success" if success else "blocked",
        "step79BlockedReason": blocked_reason,
        "selectedApproach": "A-4d-state-source-visible-record-pipeline",
        "phaseStep": phase_step,
        "step79SummaryApplies": phase_step == "phase3-step79",
        "webgpu4DStateSourceReady": state_source_ready,
        "stateSourceMode": get_path(state_source_contract, ["stateSourceMode"]),
        "stateSourceClassification": get_path(
            state_source_contract, ["sourceClassification"]
        ),
        "computed4DStatePositionCount": get_path(
            state_source_contract, ["computed4DStatePositionCount"], 0
        ),
        "baselineStatePositionCount": get_path(
            state_source_contract, ["baselineStatePositionCount"], 0
        ),
        "unavailableStatePositionCount": get_path(
            state_source_contract, ["unavailableStatePositionCount"], 0
        ),
        "full4DStateEvaluationInWgsl": get_path(
            state_source_contract, ["full4DStateEvaluationInWgsl"]
        ),
        "cameraAwareVisibleOutputReady": get_path(
            normal_contract, ["cameraAwareVisibleOutputReady"]
        ),
        "inputSourceKind": get_path(normal_contract, ["inputSourceKind"]),
        "inputSourceLineage": get_path(normal_contract, ["inputSourceLineage"]),
        "sourceClassification": get_path(normal_contract, ["sourceClassification"]),
        "consumedSourceKind": consumed_source_kind,
        "consumedSourceLineage": get_path(normal_contract, ["consumedSourceLineage"]),
        "consumedSourceClassification": consumed_source_classification,
        "consumedSampleCount": get_path(normal_contract, ["consumedSampleCount"]),
        "stateSourceVisibleRecordConsumedByNormalBackend": state_source_consumed,
        "validRecordCount": visible_record_valid_count,
        "runtimeVisibleRecordSampleCount": get_path(runtime_contract, ["sampleCount"], 0),
        "statePositionRecordCount": state_position_record_count,
        "full4DStatePositionRecordCount": full_4d_state_position_record_count,
        "stateSourceBaselineRecordCount": state_source_baseline_record_count,
        "rawPositionRepairRecordCount": raw_position_repair_record_count,
        "rawXyzRepairDependencyReduced": raw_xyz_repair_dependency_reduced,
        "projectionGatePassedCount": get_path(
            gate_summary, ["projectionGatePassedCount"]
        ),
        "visibilityGateMode": get_path(gate_summary, ["visibilityGateMode"]),
        "trueVisibleRecordPathReady": get_path(
            gate_summary, ["trueVisibleRecordPathReady"]
        ),
        "trueVisibleRecordPathClassification": path_classification,
        "full4DStateDrivenVisibleRecordPath": full_4d_state_driven_visible_record_path,
        "minimal4DStateSourceVisibleRecordPath":
            minimal_4d_state_source_visible_record_path,
        "nextFull4DStateGate": get_path(gate_summary, ["nextFull4DStateGate"]),
        "webgpuOwnedNativeCompatibleSampleCount": get_path(
            owned_generation_summary, ["webgpuOwnedSampleCount"], 0
        ),
        "validationAssistedBridgeSampleCount": get_path(
            bridge_contract, ["sampleCount"], 0
        ),
        "visibleInputSampleCount": get_path(normal_contract, ["visibleInputSampleCount"]),
        "renderedSamplePatchCount": get_path(
            normal_contract, ["renderedSamplePatchCount"]
        ),
        "cameraAwareVisibleSampleCount": camera_sample_count,
        "selectedSamplePatchCount": get_path(sample_contract, ["sampleCount"]),
        "enlargedPatchPixelCount": get_path(
            color_surface_contract, ["writtenPixelCount"]
        ),
        "outputPointRadiusPx": get_path(normal_contract, ["outputPointRadiusPx"]),
        "debugFillUsed": get_path(normal_contract, ["debugFillUsed"], False),
        "usesViewerCameraProjection": get_path(
            normal_contract, ["visibleOutputUsesCameraProjection"]
        ),
        "cameraProjectionDerivedPositions": get_path(
            normal_contract, ["cameraProjectionDerivedPositions"]
        ),
        "schedulerOwnedPath": (
            get_path(normal_contract, ["visibleOutputUsesSchedulerOwnedFramePath"])
            is True
            and get_path(normal_contract, ["schedulerFramePresentationBoundaryReady"])
            is True
            and get_path(normal_contract, ["schedulerOwnsFrameRequest"]) is True
        ),
        "guardedPresentationAdapterReady": get_path(
            guarded_presentation_adapter_contract,
            ["guardedPresentationAdapterReady"],
        ),
        "presentationTargetReadable": get_path(
            guarded_presentation_adapter_contract, ["presentationTargetReadable"]
        ),
        "presentationTargetMatchesExpected": get_path(
            guarded_presentation_adapter_contract,
            ["presentationTargetMatchesExpected"],
        ),
        "presentationTargetMaxAbsDiff": get_path(
            guarded_presentation_adapter_contract,
            ["presentationTargetMaxAbsDiff"],
        ),
        "viewerPresentationBridgeReady": get_path(
            presentation_bridge_contract, ["viewerPresentationBridgeReady"]
        ),
        "currentTextureConnectionReady": current_texture_ready,
        "currentTextureReadbackMatchesAdapterOutput": current_texture_readback_matches,
        "currentTextureMaxAbsDiff": get_path(
            presentation_bridge_contract, ["currentTextureMaxAbsDiff"]
        ),
        "webgpuExclusiveGuard": get_path(
            webgpu_normal_backend_frame_implementation_validation,
            ["guardAllowed"],
        ),
        "webgl2HybridRenderingPrevented": webgl2_hybrid_prevented,
        "fallbackSamplesMixed": fallback_samples_mixed,
        "noFallbackMixing": no_fallback_mixing,
        "sampleSources": sample_sources,
        "nativeCompatibleFallbackUsed": consumed_source_kind
        == "webgpu-owned-native-compatible-samples",
        "bridgeFallbackUsed": consumed_source_kind == "validation-assisted-bridge",
        "rawRepairFallbackUsed": raw_position_repair_record_count > 0,
        "firstValidationFailures": compact_list(
            get_path(
                webgpu_normal_backend_frame_implementation_validation,
                ["firstValidationFailures"],
                [],
            )
        ),
    }


def build_step80_webgpu_4d_state_evaluation_summary(
    summary: Dict[str, Any],
    webgpu_visible_record_camera_aware_visible_output: Dict[str, Any],
    webgpu_owned_camera_aware_visible_output: Dict[str, Any],
    validation_assisted_camera_aware_visible_output: Dict[str, Any],
    webgpu_normal_backend_frame_implementation: Dict[str, Any],
    webgpu_normal_backend_frame_implementation_validation: Dict[str, Any],
) -> Dict[str, Any]:
    runtime_contract = get_path(
        webgpu_visible_record_camera_aware_visible_output,
        ["contract"],
        {},
    )
    normal_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["cameraAwareVisibleOutputContract"],
        {},
    )
    sample_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["sampleResourceLifecycleContract"],
        {},
    )
    color_surface_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["colorOutputSurfaceLifecycleContract"],
        {},
    )
    presentation_bridge_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["presentationBridgeContract"],
        {},
    )
    guarded_presentation_adapter_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["guardedPresentationAdapterContract"],
        {},
    )
    gate_summary = get_path(summary, ["webgpuVisibleRecordGateSummary"], {})
    state_source_contract = get_path(summary, ["webgpu4DStateSourceContract"], {})
    bridge_contract = get_path(
        validation_assisted_camera_aware_visible_output,
        ["contract"],
        {},
    )
    owned_generation_summary = get_path(
        webgpu_owned_camera_aware_visible_output,
        ["generationSummary"],
        {},
    )
    sample_sources = compact_list(
        get_path(
            sample_contract,
            ["sampleSources"],
            get_path(webgpu_normal_backend_frame_implementation, ["sampleSources"], []),
        )
    )
    visible_record_sources_consumed = any(
        "webgpuVisibleRecordDryRun.cameraAwareVisibleRecords" in str(source)
        for source in sample_sources
    )
    phase_step = get_path(summary, ["phaseStep"])
    visible_record_valid_count = get_path(
        summary,
        ["validRecordCount"],
        get_path(gate_summary, ["validRecordCount"], 0),
    )
    consumed_source_kind = get_path(normal_contract, ["consumedSourceKind"])
    consumed_source_classification = get_path(
        normal_contract, ["consumedSourceClassification"]
    )
    camera_sample_count = get_path(
        normal_contract,
        ["sampleCount"],
        get_path(webgpu_normal_backend_frame_implementation, ["visibleOutputSampleCount"], 0),
    )
    current_texture_ready = get_path(
        webgpu_normal_backend_frame_implementation_validation,
        ["currentTextureConnectionReady"],
    )
    current_texture_readback_matches = get_path(
        presentation_bridge_contract,
        ["currentTextureReadbackMatchesAdapterOutput"],
    )
    webgl2_hybrid_prevented = get_path(
        webgpu_normal_backend_frame_implementation_validation,
        ["webgl2HybridRenderingPrevented"],
    )
    fallback_samples_mixed = get_path(
        normal_contract,
        ["fallbackSamplesMixed"],
        get_path(presentation_bridge_contract, ["fallbackSamplesMixed"]),
    )
    no_fallback_mixing = get_path(
        webgpu_normal_backend_frame_implementation_validation,
        ["noFallbackMixing"],
    )
    state_position_record_count = get_path(gate_summary, ["statePositionRecordCount"], 0)
    full_4d_state_position_record_count = get_path(
        gate_summary, ["full4DStatePositionRecordCount"], 0
    )
    partial_webgpu_4d_state_record_count = get_path(
        gate_summary, ["partialWebGpu4DStateRecordCount"], 0
    )
    state_source_baseline_record_count = get_path(
        gate_summary, ["stateSourceBaselineRecordCount"], 0
    )
    raw_position_repair_record_count = get_path(
        gate_summary, ["rawPositionRepairRecordCount"], 0
    )
    computed_4d_state_position_count = get_path(
        state_source_contract, ["computed4DStatePositionCount"], 0
    )
    path_classification = get_path(
        gate_summary, ["trueVisibleRecordPathClassification"]
    )
    state_source_ready = get_path(
        state_source_contract,
        ["fourDStateSourceReady"],
        get_path(gate_summary, ["fourDStateSourceReady"]),
    )
    webgpu_computed_state_positions = (
        get_path(state_source_contract, ["webgpuComputedStatePositions"]) is True
        or get_path(gate_summary, ["webgpuComputedStatePositions"]) is True
    )
    computed_state_records_projected = (
        partial_webgpu_4d_state_record_count > 0
        or full_4d_state_position_record_count > 0
    )
    computed_state_visible_record_consumed = (
        consumed_source_kind == "visible-record"
        and visible_record_sources_consumed
        and computed_state_records_projected
        and bool(visible_record_valid_count)
        and bool(camera_sample_count)
    )
    partial_webgpu_4d_state_evaluated_visible_record_path = (
        path_classification == "partial-webgpu-4d-state-evaluated-visible-record"
        and partial_webgpu_4d_state_record_count > 0
        and raw_position_repair_record_count == 0
    )
    full_webgpu_4d_state_evaluated_visible_record_path = (
        path_classification == "full-webgpu-4d-state-evaluated-visible-record"
        and full_4d_state_position_record_count == visible_record_valid_count
        and raw_position_repair_record_count == 0
    )
    success = (
        phase_step == "phase3-step80"
        and get_path(normal_contract, ["cameraAwareVisibleOutputReady"]) is True
        and state_source_ready is True
        and webgpu_computed_state_positions
        and computed_4d_state_position_count > 0
        and computed_state_visible_record_consumed
        and raw_position_repair_record_count == 0
        and current_texture_ready is True
        and current_texture_readback_matches is True
        and webgl2_hybrid_prevented is True
        and fallback_samples_mixed is False
        and no_fallback_mixing is True
    )
    blocked_reason = None
    if not success:
        if phase_step != "phase3-step80":
            blocked_reason = "summary-phase-step-is-not-phase3-step80"
        elif state_source_ready is not True:
            blocked_reason = "webgpu-4d-state-source-contract-not-ready"
        elif not webgpu_computed_state_positions:
            blocked_reason = "state-source-not-produced-by-webgpu-evaluator"
        elif computed_4d_state_position_count <= 0:
            blocked_reason = "computed4DStatePositionCount-is-0"
        elif not visible_record_valid_count:
            blocked_reason = "validRecordCount-is-0"
        elif not computed_state_records_projected:
            blocked_reason = "computed-state-positions-did-not-project-to-visible-records"
        elif not computed_state_visible_record_consumed:
            blocked_reason = "normal-backend-did-not-consume-computed-state-visible-records"
        elif raw_position_repair_record_count != 0:
            blocked_reason = "visible-record-path-still-depends-on-raw-xyz-repair"
        elif get_path(normal_contract, ["cameraAwareVisibleOutputReady"]) is not True:
            blocked_reason = "camera-aware-visible-output-contract-not-ready"
        elif current_texture_ready is not True:
            blocked_reason = "currentTexture-connection-not-ready"
        elif current_texture_readback_matches is not True:
            blocked_reason = "currentTexture-readback-did-not-match-adapter-output"
        elif webgl2_hybrid_prevented is not True:
            blocked_reason = "webgl2-hybrid-rendering-not-prevented"
        elif fallback_samples_mixed is not False or no_fallback_mixing is not True:
            blocked_reason = "fallback-samples-mixed-or-not-proven-suppressed"
    return {
        "step80Decision": "success" if success else "blocked",
        "step80BlockedReason": blocked_reason,
        "selectedApproach": "B-webgpu-partial-4d-state-evaluator-visible-record",
        "phaseStep": phase_step,
        "step80SummaryApplies": phase_step == "phase3-step80",
        "webgpu4DStateSourceReady": state_source_ready,
        "stateSourceMode": get_path(state_source_contract, ["stateSourceMode"]),
        "stateSourceClassification": get_path(
            state_source_contract, ["sourceClassification"]
        ),
        "webgpuComputedStatePositions": webgpu_computed_state_positions,
        "webgpu4DStateEvaluationMode": get_path(
            state_source_contract, ["webgpu4DStateEvaluationMode"]
        ),
        "computed4DStatePositionCount": computed_4d_state_position_count,
        "baselineStatePositionCount": get_path(
            state_source_contract, ["baselineStatePositionCount"], 0
        ),
        "unavailableStatePositionCount": get_path(
            state_source_contract, ["unavailableStatePositionCount"], 0
        ),
        "full4DStateEvaluationInWgsl": get_path(
            state_source_contract, ["full4DStateEvaluationInWgsl"]
        ),
        "cameraAwareVisibleOutputReady": get_path(
            normal_contract, ["cameraAwareVisibleOutputReady"]
        ),
        "inputSourceKind": get_path(normal_contract, ["inputSourceKind"]),
        "inputSourceLineage": get_path(normal_contract, ["inputSourceLineage"]),
        "sourceClassification": get_path(normal_contract, ["sourceClassification"]),
        "consumedSourceKind": consumed_source_kind,
        "consumedSourceLineage": get_path(normal_contract, ["consumedSourceLineage"]),
        "consumedSourceClassification": consumed_source_classification,
        "consumedSampleCount": get_path(normal_contract, ["consumedSampleCount"]),
        "computedStateVisibleRecordConsumedByNormalBackend":
            computed_state_visible_record_consumed,
        "validRecordCount": visible_record_valid_count,
        "runtimeVisibleRecordSampleCount": get_path(runtime_contract, ["sampleCount"], 0),
        "statePositionRecordCount": state_position_record_count,
        "full4DStatePositionRecordCount": full_4d_state_position_record_count,
        "partialWebGpu4DStateRecordCount": partial_webgpu_4d_state_record_count,
        "stateSourceBaselineRecordCount": state_source_baseline_record_count,
        "rawPositionRepairRecordCount": raw_position_repair_record_count,
        "rawXyzRepairDependencyReduced": (
            state_position_record_count > 0 and raw_position_repair_record_count == 0
        ),
        "projectionGatePassedCount": get_path(
            gate_summary, ["projectionGatePassedCount"]
        ),
        "visibilityGateMode": get_path(gate_summary, ["visibilityGateMode"]),
        "trueVisibleRecordPathReady": get_path(
            gate_summary, ["trueVisibleRecordPathReady"]
        ),
        "trueVisibleRecordPathClassification": path_classification,
        "partialWebGpu4DStateEvaluatedVisibleRecordPath":
            partial_webgpu_4d_state_evaluated_visible_record_path,
        "fullWebGpu4DStateEvaluatedVisibleRecordPath":
            full_webgpu_4d_state_evaluated_visible_record_path,
        "nextFull4DStateGate": get_path(gate_summary, ["nextFull4DStateGate"]),
        "webgpuOwnedNativeCompatibleSampleCount": get_path(
            owned_generation_summary, ["webgpuOwnedSampleCount"], 0
        ),
        "validationAssistedBridgeSampleCount": get_path(
            bridge_contract, ["sampleCount"], 0
        ),
        "visibleInputSampleCount": get_path(normal_contract, ["visibleInputSampleCount"]),
        "renderedSamplePatchCount": get_path(
            normal_contract, ["renderedSamplePatchCount"]
        ),
        "cameraAwareVisibleSampleCount": camera_sample_count,
        "selectedSamplePatchCount": get_path(sample_contract, ["sampleCount"]),
        "enlargedPatchPixelCount": get_path(
            color_surface_contract, ["writtenPixelCount"]
        ),
        "outputPointRadiusPx": get_path(normal_contract, ["outputPointRadiusPx"]),
        "debugFillUsed": get_path(normal_contract, ["debugFillUsed"], False),
        "usesViewerCameraProjection": get_path(
            normal_contract, ["visibleOutputUsesCameraProjection"]
        ),
        "cameraProjectionDerivedPositions": get_path(
            normal_contract, ["cameraProjectionDerivedPositions"]
        ),
        "schedulerOwnedPath": (
            get_path(normal_contract, ["visibleOutputUsesSchedulerOwnedFramePath"])
            is True
            and get_path(normal_contract, ["schedulerFramePresentationBoundaryReady"])
            is True
            and get_path(normal_contract, ["schedulerOwnsFrameRequest"]) is True
        ),
        "guardedPresentationAdapterReady": get_path(
            guarded_presentation_adapter_contract,
            ["guardedPresentationAdapterReady"],
        ),
        "viewerPresentationBridgeReady": get_path(
            presentation_bridge_contract, ["viewerPresentationBridgeReady"]
        ),
        "currentTextureConnectionReady": current_texture_ready,
        "currentTextureReadbackMatchesAdapterOutput": current_texture_readback_matches,
        "webgpuExclusiveGuard": get_path(
            webgpu_normal_backend_frame_implementation_validation,
            ["guardAllowed"],
        ),
        "webgl2HybridRenderingPrevented": webgl2_hybrid_prevented,
        "fallbackSamplesMixed": fallback_samples_mixed,
        "noFallbackMixing": no_fallback_mixing,
        "sampleSources": sample_sources,
        "fullWebGpu4DStateEvaluationReady":
            full_webgpu_4d_state_evaluated_visible_record_path,
        "partialWebGpu4DStateEvaluationReady":
            partial_webgpu_4d_state_evaluated_visible_record_path,
        "minimal4DStateSourceFallbackUsed": get_path(
            state_source_contract, ["sourceClassification"]
        )
        == "minimal-4d-state-source",
        "nativeCompatibleFallbackUsed": consumed_source_kind
        == "webgpu-owned-native-compatible-samples",
        "bridgeFallbackUsed": consumed_source_kind == "validation-assisted-bridge",
        "rawRepairFallbackUsed": raw_position_repair_record_count > 0,
        "firstValidationFailures": compact_list(
            get_path(
                webgpu_normal_backend_frame_implementation_validation,
                ["firstValidationFailures"],
                [],
            )
        ),
    }


def build_step81_webgpu_gaussian_attribute_evaluation_summary(
    summary: Dict[str, Any],
    webgpu_visible_record_camera_aware_visible_output: Dict[str, Any],
    webgpu_owned_camera_aware_visible_output: Dict[str, Any],
    validation_assisted_camera_aware_visible_output: Dict[str, Any],
    webgpu_normal_backend_frame_implementation: Dict[str, Any],
    webgpu_normal_backend_frame_implementation_validation: Dict[str, Any],
) -> Dict[str, Any]:
    step80 = build_step80_webgpu_4d_state_evaluation_summary(
        summary,
        webgpu_visible_record_camera_aware_visible_output,
        webgpu_owned_camera_aware_visible_output,
        validation_assisted_camera_aware_visible_output,
        webgpu_normal_backend_frame_implementation,
        webgpu_normal_backend_frame_implementation_validation,
    )
    normal_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["cameraAwareVisibleOutputContract"],
        {},
    )
    sample_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["sampleResourceLifecycleContract"],
        {},
    )
    gate_summary = get_path(summary, ["webgpuVisibleRecordGateSummary"], {})
    attr_contract = get_path(
        summary,
        ["webgpuGaussianAttributeEvaluationContract"],
        get_path(normal_contract, ["webgpuGaussianAttributeEvaluationContract"], {}),
    )
    phase_step = get_path(summary, ["phaseStep"])
    computed_attr_count = get_path(
        attr_contract,
        ["computedRenderAttributeCount"],
        get_path(gate_summary, ["computedRenderAttributeCount"], 0),
    )
    computed_attr_sample_count = get_path(
        normal_contract,
        ["computedRenderAttributeSampleCount"],
        get_path(
            sample_contract,
            ["computedRenderAttributeSampleCount"],
            get_path(sample_contract, ["webgpuComputedRenderAttributeSampleCount"], 0),
        ),
    )
    camera_sample_count = get_path(
        normal_contract,
        ["sampleCount"],
        get_path(webgpu_normal_backend_frame_implementation, ["visibleOutputSampleCount"], 0),
    )
    computed_payload_consumed = (
        get_path(normal_contract, ["computedRenderPayloadConsumed"]) is True
        or get_path(sample_contract, ["computedRenderPayloadConsumed"]) is True
    )
    attr_ready = (
        get_path(attr_contract, ["gaussianAttributeEvaluationReady"]) is True
        and get_path(attr_contract, ["webgpuComputedRenderAttributes"]) is True
        and computed_attr_count > 0
    )
    normal_backend_consumed_attributes = (
        get_path(
            sample_contract,
            ["normalBackendConsumedComputedRenderAttributes"],
        )
        is True
        or (
            computed_payload_consumed
            and computed_attr_sample_count > 0
            and computed_attr_sample_count == camera_sample_count
            and get_path(sample_contract, ["computedRadiusConsumed"]) is True
            and get_path(sample_contract, ["computedColorAlphaConsumed"]) is True
        )
    )
    success = (
        phase_step == "phase3-step81"
        and step80.get("webgpuComputedStatePositions") is True
        and step80.get("computed4DStatePositionCount", 0) > 0
        and step80.get("computedStateVisibleRecordConsumedByNormalBackend") is True
        and attr_ready
        and normal_backend_consumed_attributes
        and step80.get("currentTextureConnectionReady") is True
        and step80.get("currentTextureReadbackMatchesAdapterOutput") is True
        and step80.get("webgl2HybridRenderingPrevented") is True
        and step80.get("fallbackSamplesMixed") is False
        and step80.get("noFallbackMixing") is True
    )
    blocked_reason = None
    if not success:
        if phase_step != "phase3-step81":
            blocked_reason = "summary-phase-step-is-not-phase3-step81"
        elif step80.get("webgpuComputedStatePositions") is not True:
            blocked_reason = "webgpu-computed-state-positions-not-ready"
        elif step80.get("computedStateVisibleRecordConsumedByNormalBackend") is not True:
            blocked_reason = "computed-state-visible-records-not-consumed"
        elif not attr_ready:
            blocked_reason = "webgpu-gaussian-attribute-evaluation-not-ready"
        elif not normal_backend_consumed_attributes:
            blocked_reason = "normal-backend-did-not-consume-computed-render-attributes"
        elif step80.get("currentTextureConnectionReady") is not True:
            blocked_reason = "currentTexture-connection-not-ready"
        elif step80.get("currentTextureReadbackMatchesAdapterOutput") is not True:
            blocked_reason = "currentTexture-readback-did-not-match-adapter-output"
        elif step80.get("webgl2HybridRenderingPrevented") is not True:
            blocked_reason = "webgl2-hybrid-rendering-not-prevented"
        elif (
            step80.get("fallbackSamplesMixed") is not False
            or step80.get("noFallbackMixing") is not True
        ):
            blocked_reason = "fallback-samples-mixed-or-not-proven-suppressed"
    return {
        "step81Decision": "success" if success else "blocked",
        "step81BlockedReason": blocked_reason,
        "selectedApproach": "B/C-webgpu-partial-gaussian-attribute-evaluator",
        "phaseStep": phase_step,
        "step81SummaryApplies": phase_step == "phase3-step81",
        "webgpu4DStateSourceReady": step80.get("webgpu4DStateSourceReady"),
        "webgpuComputedStatePositions": step80.get("webgpuComputedStatePositions"),
        "computed4DStatePositionCount": step80.get(
            "computed4DStatePositionCount"
        ),
        "computedStateVisibleRecordConsumedByNormalBackend": step80.get(
            "computedStateVisibleRecordConsumedByNormalBackend"
        ),
        "gaussianAttributeEvaluationReady": get_path(
            attr_contract, ["gaussianAttributeEvaluationReady"]
        ),
        "webgpuComputedRenderAttributes": get_path(
            attr_contract, ["webgpuComputedRenderAttributes"]
        ),
        "computedRenderAttributeCount": computed_attr_count,
        "computedAttributeFields": get_path(
            attr_contract, ["computedAttributeFields"], []
        ),
        "partialAttributeFields": get_path(
            attr_contract, ["partialAttributeFields"], []
        ),
        "baselineAttributeFields": get_path(
            attr_contract, ["baselineAttributeFields"], []
        ),
        "fallbackAttributeFields": get_path(
            attr_contract, ["fallbackAttributeFields"], []
        ),
        "referenceAssistedAttributeFields": get_path(
            attr_contract, ["referenceAssistedAttributeFields"], []
        ),
        "deferredAttributeFields": get_path(
            attr_contract, ["deferredAttributeFields"], []
        ),
        "normalBackendPointRadiusPx": get_path(
            attr_contract, ["normalBackendPointRadiusPx"]
        ),
        "averageComputedRadiusPx": get_path(
            attr_contract, ["averageComputedRadiusPx"]
        ),
        "averageComputedAlpha": get_path(attr_contract, ["averageComputedAlpha"]),
        "renderAttributeClassification": get_path(
            attr_contract, ["renderAttributeClassification"]
        ),
        "renderPayloadClassification": get_path(
            attr_contract, ["renderPayloadClassification"]
        ),
        "fullGaussianAttributeEvaluationInWgsl": get_path(
            attr_contract, ["fullGaussianAttributeEvaluationInWgsl"]
        ),
        "computedRenderPayloadConsumed": computed_payload_consumed,
        "computedRenderAttributeSampleCount": computed_attr_sample_count,
        "normalBackendConsumedComputedRenderAttributes":
            normal_backend_consumed_attributes,
        "computedRadiusConsumed": get_path(
            sample_contract, ["computedRadiusConsumed"]
        ),
        "computedColorAlphaConsumed": get_path(
            sample_contract, ["computedColorAlphaConsumed"]
        ),
        "computedTemporalWeightAvailable": get_path(
            sample_contract, ["computedTemporalWeightAvailable"]
        ),
        "computedTemporalWeightUsage": get_path(
            sample_contract, ["computedTemporalWeightUsage"]
        ),
        "sourceClassification": get_path(normal_contract, ["sourceClassification"]),
        "consumedSourceKind": get_path(normal_contract, ["consumedSourceKind"]),
        "consumedSourceClassification": get_path(
            normal_contract, ["consumedSourceClassification"]
        ),
        "validRecordCount": step80.get("validRecordCount"),
        "partialWebGpu4DStateRecordCount": step80.get(
            "partialWebGpu4DStateRecordCount"
        ),
        "renderedSamplePatchCount": step80.get("renderedSamplePatchCount"),
        "cameraAwareVisibleSampleCount": step80.get(
            "cameraAwareVisibleSampleCount"
        ),
        "enlargedPatchPixelCount": step80.get("enlargedPatchPixelCount"),
        "outputPointRadiusPx": get_path(normal_contract, ["outputPointRadiusPx"]),
        "debugFillUsed": step80.get("debugFillUsed"),
        "usesViewerCameraProjection": step80.get("usesViewerCameraProjection"),
        "schedulerOwnedPath": step80.get("schedulerOwnedPath"),
        "currentTextureConnectionReady": step80.get(
            "currentTextureConnectionReady"
        ),
        "currentTextureReadbackMatchesAdapterOutput": step80.get(
            "currentTextureReadbackMatchesAdapterOutput"
        ),
        "webgpuExclusiveGuard": step80.get("webgpuExclusiveGuard"),
        "webgl2HybridRenderingPrevented": step80.get(
            "webgl2HybridRenderingPrevented"
        ),
        "fallbackSamplesMixed": step80.get("fallbackSamplesMixed"),
        "noFallbackMixing": step80.get("noFallbackMixing"),
        "sampleSources": step80.get("sampleSources"),
        "renderAttributeSources": get_path(
            sample_contract, ["renderAttributeSources"], []
        ),
        "fullWebGpu4DStateEvaluationReady": step80.get(
            "fullWebGpu4DStateEvaluationReady"
        ),
        "partialWebGpu4DStateEvaluationReady": step80.get(
            "partialWebGpu4DStateEvaluationReady"
        ),
        "minimal4DStateSourceFallbackUsed": step80.get(
            "minimal4DStateSourceFallbackUsed"
        ),
        "nativeCompatibleFallbackUsed": step80.get("nativeCompatibleFallbackUsed"),
        "bridgeFallbackUsed": step80.get("bridgeFallbackUsed"),
        "rawRepairFallbackUsed": step80.get("rawRepairFallbackUsed"),
        "firstValidationFailures": step80.get("firstValidationFailures", []),
    }


def build_step82_webgpu_gaussian_footprint_pipeline_summary(
    summary: Dict[str, Any],
    webgpu_visible_record_camera_aware_visible_output: Dict[str, Any],
    webgpu_owned_camera_aware_visible_output: Dict[str, Any],
    validation_assisted_camera_aware_visible_output: Dict[str, Any],
    webgpu_normal_backend_frame_implementation: Dict[str, Any],
    webgpu_normal_backend_frame_implementation_validation: Dict[str, Any],
) -> Dict[str, Any]:
    step81 = build_step81_webgpu_gaussian_attribute_evaluation_summary(
        summary,
        webgpu_visible_record_camera_aware_visible_output,
        webgpu_owned_camera_aware_visible_output,
        validation_assisted_camera_aware_visible_output,
        webgpu_normal_backend_frame_implementation,
        webgpu_normal_backend_frame_implementation_validation,
    )
    normal_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["cameraAwareVisibleOutputContract"],
        {},
    )
    sample_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["sampleResourceLifecycleContract"],
        {},
    )
    footprint_contract = get_path(
        summary,
        ["webgpuGaussianFootprintEvaluationContract"],
        get_path(normal_contract, ["webgpuGaussianFootprintEvaluationContract"], {}),
    )
    phase_step = get_path(summary, ["phaseStep"])
    computed_footprint_count = get_path(
        footprint_contract, ["computedFootprintPayloadCount"], 0
    )
    computed_footprint_sample_count = get_path(
        normal_contract,
        ["computedFootprintPayloadSampleCount"],
        get_path(
            sample_contract,
            ["computedFootprintPayloadSampleCount"],
            get_path(sample_contract, ["webgpuComputedFootprintPayloadSampleCount"], 0),
        ),
    )
    camera_sample_count = get_path(
        normal_contract,
        ["sampleCount"],
        get_path(webgpu_normal_backend_frame_implementation, ["visibleOutputSampleCount"], 0),
    )
    footprint_payload_consumed = (
        get_path(normal_contract, ["computedFootprintPayloadConsumed"]) is True
        or get_path(sample_contract, ["computedFootprintPayloadConsumed"]) is True
    )
    normal_backend_consumed_footprint = (
        get_path(
            sample_contract,
            ["normalBackendConsumedComputedFootprintPayload"],
        )
        is True
        or (
            footprint_payload_consumed
            and computed_footprint_sample_count > 0
            and computed_footprint_sample_count == camera_sample_count
            and get_path(sample_contract, ["computedConicConsumed"]) is True
            and get_path(sample_contract, ["computedAabbConsumed"]) is True
            and get_path(sample_contract, ["computedTileRangeConsumed"]) is True
        )
    )
    partial_aabb_derived_consumed = (
        get_path(sample_contract, ["partialAabbDerivedConsumed"]) is True
        or get_path(sample_contract, ["computedAabbConsumed"]) is True
    )
    partial_tile_range_derived_consumed = (
        get_path(sample_contract, ["partialTileRangeDerivedConsumed"]) is True
        or get_path(sample_contract, ["computedTileRangeConsumed"]) is True
    )
    footprint_ready = (
        get_path(footprint_contract, ["gaussianFootprintEvaluationReady"]) is True
        and get_path(footprint_contract, ["webgpuComputedFootprintPayload"]) is True
        and computed_footprint_count > 0
    )
    step81_attributes_consumed = (
        step81.get("normalBackendConsumedComputedRenderAttributes") is True
        and step81.get("computedRadiusConsumed") is True
        and step81.get("computedColorAlphaConsumed") is True
    )
    success = (
        phase_step == "phase3-step82"
        and step81.get("webgpuComputedStatePositions") is True
        and step81.get("computedStateVisibleRecordConsumedByNormalBackend") is True
        and step81_attributes_consumed
        and footprint_ready
        and normal_backend_consumed_footprint
        and step81.get("currentTextureConnectionReady") is True
        and step81.get("currentTextureReadbackMatchesAdapterOutput") is True
        and step81.get("webgl2HybridRenderingPrevented") is True
        and step81.get("fallbackSamplesMixed") is False
        and step81.get("noFallbackMixing") is True
    )
    blocked_reason = None
    if not success:
        if phase_step != "phase3-step82":
            blocked_reason = "summary-phase-step-is-not-phase3-step82"
        elif not step81_attributes_consumed:
            blocked_reason = "step81-computed-render-attributes-not-preserved"
        elif not footprint_ready:
            blocked_reason = "webgpu-gaussian-footprint-evaluation-not-ready"
        elif not normal_backend_consumed_footprint:
            blocked_reason = "normal-backend-did-not-consume-computed-footprint-payload"
        elif step81.get("currentTextureConnectionReady") is not True:
            blocked_reason = "currentTexture-connection-not-ready"
        elif step81.get("currentTextureReadbackMatchesAdapterOutput") is not True:
            blocked_reason = "currentTexture-readback-did-not-match-adapter-output"
        elif step81.get("webgl2HybridRenderingPrevented") is not True:
            blocked_reason = "webgl2-hybrid-rendering-not-prevented"
        elif (
            step81.get("fallbackSamplesMixed") is not False
            or step81.get("noFallbackMixing") is not True
        ):
            blocked_reason = "fallback-samples-mixed-or-not-proven-suppressed"
    return {
        "step82Decision": "success" if success else "blocked",
        "step82BlockedReason": blocked_reason,
        "selectedApproach": "B/C-webgpu-partial-gaussian-footprint-evaluator",
        "phaseStep": phase_step,
        "step82SummaryApplies": phase_step == "phase3-step82",
        "webgpuComputedStatePositions": step81.get("webgpuComputedStatePositions"),
        "computed4DStatePositionCount": step81.get("computed4DStatePositionCount"),
        "computedStateVisibleRecordConsumedByNormalBackend": step81.get(
            "computedStateVisibleRecordConsumedByNormalBackend"
        ),
        "step81ComputedRenderAttributesPreserved": step81_attributes_consumed,
        "computedRadiusConsumed": step81.get("computedRadiusConsumed"),
        "computedColorAlphaConsumed": step81.get("computedColorAlphaConsumed"),
        "computedTemporalWeightAvailable": step81.get(
            "computedTemporalWeightAvailable"
        ),
        "gaussianFootprintEvaluationReady": get_path(
            footprint_contract, ["gaussianFootprintEvaluationReady"]
        ),
        "webgpuComputedFootprintPayload": get_path(
            footprint_contract, ["webgpuComputedFootprintPayload"]
        ),
        "computedFootprintPayloadCount": computed_footprint_count,
        "computedFootprintFields": get_path(
            footprint_contract, ["computedFootprintFields"], []
        ),
        "partialFootprintFields": get_path(
            footprint_contract, ["partialFootprintFields"], []
        ),
        "baselineFootprintFields": get_path(
            footprint_contract, ["baselineFootprintFields"], []
        ),
        "fallbackFootprintFields": get_path(
            footprint_contract, ["fallbackFootprintFields"], []
        ),
        "deferredFootprintFields": get_path(
            footprint_contract, ["deferredFootprintFields"], []
        ),
        "footprintPayloadClassification": get_path(
            footprint_contract, ["footprintPayloadClassification"]
        ),
        "fullGaussianFootprintEvaluationInWgsl": get_path(
            footprint_contract, ["fullGaussianFootprintEvaluationInWgsl"]
        ),
        "computedFootprintPayloadConsumed": footprint_payload_consumed,
        "computedFootprintPayloadSampleCount": computed_footprint_sample_count,
        "normalBackendConsumedComputedFootprintPayload":
            normal_backend_consumed_footprint,
        "computedConicConsumed": get_path(
            sample_contract, ["computedConicConsumed"]
        ),
        "computedAabbConsumed": get_path(
            sample_contract, ["computedAabbConsumed"]
        ),
        "computedTileRangeConsumed": get_path(
            sample_contract, ["computedTileRangeConsumed"]
        ),
        "partialAabbDerivedConsumed": partial_aabb_derived_consumed,
        "partialTileRangeDerivedConsumed": partial_tile_range_derived_consumed,
        "aabbConsumptionMode": get_path(
            sample_contract,
            ["aabbConsumptionMode"],
            "partial-derived-from-webgpu-projected-px-py-and-computed-radius"
            if partial_aabb_derived_consumed
            else None,
        ),
        "tileRangeConsumptionMode": get_path(
            sample_contract,
            ["tileRangeConsumptionMode"],
            "partial-derived-from-webgpu-projected-px-py-and-computed-radius"
            if partial_tile_range_derived_consumed
            else None,
        ),
        "fullGpuNativeAabbParityDeferred": get_path(
            sample_contract,
            ["fullGpuNativeAabbParityDeferred"],
            "gpu-aabb-from-projected-center"
            in get_path(footprint_contract, ["deferredFootprintFields"], []),
        ),
        "fullGpuNativeTileRangeParityDeferred": get_path(
            sample_contract,
            ["fullGpuNativeTileRangeParityDeferred"],
            "gpu-tileRange-from-aabb"
            in get_path(footprint_contract, ["deferredFootprintFields"], []),
        ),
        "footprintPayloadSources": get_path(
            sample_contract, ["footprintPayloadSources"], []
        ),
        "renderedSamplePatchCount": step81.get("renderedSamplePatchCount"),
        "cameraAwareVisibleSampleCount": step81.get(
            "cameraAwareVisibleSampleCount"
        ),
        "currentTextureConnectionReady": step81.get(
            "currentTextureConnectionReady"
        ),
        "currentTextureReadbackMatchesAdapterOutput": step81.get(
            "currentTextureReadbackMatchesAdapterOutput"
        ),
        "webgpuExclusiveGuard": step81.get("webgpuExclusiveGuard"),
        "webgl2HybridRenderingPrevented": step81.get(
            "webgl2HybridRenderingPrevented"
        ),
        "fallbackSamplesMixed": step81.get("fallbackSamplesMixed"),
        "noFallbackMixing": step81.get("noFallbackMixing"),
        "fullAttributeSuccessClaimed": step81.get(
            "fullGaussianAttributeEvaluationInWgsl"
        ) is True,
        "fullFootprintSuccessClaimed": get_path(
            footprint_contract, ["fullGaussianFootprintEvaluationInWgsl"]
        ) is True,
        "firstValidationFailures": step81.get("firstValidationFailures", []),
    }


def build_step83_webgpu_tile_aware_render_input_summary(
    summary: Dict[str, Any],
    webgpu_visible_record_camera_aware_visible_output: Dict[str, Any],
    webgpu_owned_camera_aware_visible_output: Dict[str, Any],
    validation_assisted_camera_aware_visible_output: Dict[str, Any],
    webgpu_normal_backend_frame_implementation: Dict[str, Any],
    webgpu_normal_backend_frame_implementation_validation: Dict[str, Any],
) -> Dict[str, Any]:
    step82 = build_step82_webgpu_gaussian_footprint_pipeline_summary(
        summary,
        webgpu_visible_record_camera_aware_visible_output,
        webgpu_owned_camera_aware_visible_output,
        validation_assisted_camera_aware_visible_output,
        webgpu_normal_backend_frame_implementation,
        webgpu_normal_backend_frame_implementation_validation,
    )
    phase_step = get_path(summary, ["phaseStep"])
    tile_contract = get_path(
        summary,
        ["webgpuTileAwareRenderInputContract"],
        get_path(
            webgpu_normal_backend_frame_implementation,
            ["webgpuTileAwareRenderInputContract"],
            {},
        ),
    )
    generated_fields = get_path(tile_contract, ["generatedPayloadFields"], [])
    deferred_fields = get_path(tile_contract, ["deferredTilePayloadFields"], [])
    generated_tile_record_count = get_path(
        tile_contract, ["generatedTileRecordCount"], 0
    )
    tile_aware_ready = (
        get_path(tile_contract, ["tileAwareRenderInputReady"]) is True
        and get_path(tile_contract, ["tileAwareConsumerReady"]) is True
        and get_path(tile_contract, ["tileAwareConsumerConsumed"]) is True
        and generated_tile_record_count > 0
    )
    footprint_baseline_preserved = (
        step82.get("webgpuComputedFootprintPayload") is True
        and step82.get("normalBackendConsumedComputedFootprintPayload") is True
        and step82.get("computedConicConsumed") is True
        and step82.get("partialAabbDerivedConsumed") is True
        and step82.get("partialTileRangeDerivedConsumed") is True
    )
    current_texture_ready = (
        step82.get("currentTextureConnectionReady") is True
        and step82.get("currentTextureReadbackMatchesAdapterOutput") is True
    )
    success = (
        phase_step == "phase3-step83"
        and tile_aware_ready
        and footprint_baseline_preserved
        and current_texture_ready
        and step82.get("webgl2HybridRenderingPrevented") is True
        and step82.get("fallbackSamplesMixed") is False
        and step82.get("noFallbackMixing") is True
    )
    blocked_reason = None
    if not success:
        if phase_step != "phase3-step83":
            blocked_reason = "summary-phase-step-is-not-phase3-step83"
        elif not tile_aware_ready:
            blocked_reason = "webgpu-tile-aware-render-input-or-consumer-not-ready"
        elif not footprint_baseline_preserved:
            blocked_reason = "step82-computed-footprint-baseline-not-preserved"
        elif not current_texture_ready:
            blocked_reason = "currentTexture-path-not-ready"
        elif step82.get("webgl2HybridRenderingPrevented") is not True:
            blocked_reason = "webgl2-hybrid-rendering-not-prevented"
        elif (
            step82.get("fallbackSamplesMixed") is not False
            or step82.get("noFallbackMixing") is not True
        ):
            blocked_reason = "fallback-samples-mixed-or-not-proven-suppressed"
    return {
        "step83Decision": "success" if success else "blocked",
        "step83BlockedReason": blocked_reason,
        "selectedApproach": "A/C-webgpu-tile-aware-render-input-from-footprint",
        "phaseStep": phase_step,
        "step83SummaryApplies": phase_step == "phase3-step83",
        "webgpuComputedTileAwareRenderInput": get_path(
            tile_contract, ["tileAwareRenderInputReady"]
        ) is True,
        "tileAwareRenderInputReady": get_path(
            tile_contract, ["tileAwareRenderInputReady"]
        ),
        "tileAwareConsumerReady": get_path(
            tile_contract, ["tileAwareConsumerReady"]
        ),
        "tileAwareConsumerConsumed": get_path(
            tile_contract, ["tileAwareConsumerConsumed"]
        ),
        "tileAwareConsumerReadbackCompleted": get_path(
            tile_contract, ["tileAwareConsumerReadbackCompleted"]
        ),
        "candidateSampleCount": get_path(tile_contract, ["candidateSampleCount"], 0),
        "generatedTileRecordCount": generated_tile_record_count,
        "tileRecordFloatStride": get_path(
            tile_contract, ["tileRecordFloatStride"]
        ),
        "tileSize": get_path(tile_contract, ["tileSize"]),
        "tileGrid": get_path(tile_contract, ["tileGrid"]),
        "totalTileReferenceCount": get_path(
            tile_contract, ["totalTileReferenceCount"], 0
        ),
        "maxTileReferenceCount": get_path(
            tile_contract, ["maxTileReferenceCount"], 0
        ),
        "averageTileReferenceCount": get_path(
            tile_contract, ["averageTileReferenceCount"]
        ),
        "generatedPayloadFields": generated_fields,
        "partialTilePayloadFields": get_path(
            tile_contract, ["partialTilePayloadFields"], []
        ),
        "deferredTilePayloadFields": deferred_fields,
        "tilePayloadClassification": get_path(
            tile_contract, ["tilePayloadClassification"]
        ),
        "gpuNativeAabbGenerated": "gpu-native-aabb" in generated_fields,
        "gpuNativeTileRangeGenerated": "gpu-native-tileRange" in generated_fields,
        "tileRecordsGenerated": "tile-record" in generated_fields,
        "depthKeyGenerated": "depth-key" in generated_fields,
        "sortKeyGenerated": "sort-key" in generated_fields,
        "tileAwareConsumerIsExplicit": get_path(
            tile_contract, ["tileAwareConsumerConsumed"]
        ) is True,
        "normalBackendOnlyMetadataPath": get_path(
            tile_contract, ["tileAwareConsumerConsumed"]
        ) is not True,
        "normalBackendFallbackMaintained": get_path(
            tile_contract, ["normalBackendFallbackMaintained"]
        ),
        "fullTileListScatterInWgsl": get_path(
            tile_contract, ["fullTileListScatterInWgsl"]
        ),
        "fullDepthSortInWgsl": get_path(
            tile_contract, ["fullDepthSortInWgsl"]
        ),
        "finalTileCompositorImplemented": get_path(
            tile_contract, ["finalTileCompositorImplemented"]
        ),
        "fullTileListScatterDeferred": "full-tile-list-scatter" in deferred_fields,
        "fullDepthSortDispatchDeferred": "full-depth-sort-dispatch" in deferred_fields,
        "finalTileCompositorDeferred": "final-tile-compositor" in deferred_fields,
        "step82ComputedFootprintPayloadPreserved":
            footprint_baseline_preserved,
        "step81ComputedAttributesPreserved":
            step82.get("step81ComputedRenderAttributesPreserved"),
        "computedConicConsumed": step82.get("computedConicConsumed"),
        "partialAabbDerivedConsumed": step82.get("partialAabbDerivedConsumed"),
        "partialTileRangeDerivedConsumed": step82.get(
            "partialTileRangeDerivedConsumed"
        ),
        "currentTextureConnectionReady": step82.get(
            "currentTextureConnectionReady"
        ),
        "currentTextureReadbackMatchesAdapterOutput": step82.get(
            "currentTextureReadbackMatchesAdapterOutput"
        ),
        "webgpuExclusiveGuard": step82.get("webgpuExclusiveGuard"),
        "webgl2HybridRenderingPrevented": step82.get(
            "webgl2HybridRenderingPrevented"
        ),
        "fallbackSamplesMixed": step82.get("fallbackSamplesMixed"),
        "noFallbackMixing": step82.get("noFallbackMixing"),
        "fullTilePipelineSuccessClaimed": (
            get_path(tile_contract, ["fullTileListScatterInWgsl"]) is True
            and get_path(tile_contract, ["fullDepthSortInWgsl"]) is True
            and get_path(tile_contract, ["finalTileCompositorImplemented"]) is True
        ),
        "firstValidationFailures": step82.get("firstValidationFailures", []),
    }


def build_step84_webgpu_gpu_owned_tile_list_layout_summary(
    summary: Dict[str, Any],
    webgpu_visible_record_camera_aware_visible_output: Dict[str, Any],
    webgpu_owned_camera_aware_visible_output: Dict[str, Any],
    validation_assisted_camera_aware_visible_output: Dict[str, Any],
    webgpu_normal_backend_frame_implementation: Dict[str, Any],
    webgpu_normal_backend_frame_implementation_validation: Dict[str, Any],
) -> Dict[str, Any]:
    step82 = build_step82_webgpu_gaussian_footprint_pipeline_summary(
        summary,
        webgpu_visible_record_camera_aware_visible_output,
        webgpu_owned_camera_aware_visible_output,
        validation_assisted_camera_aware_visible_output,
        webgpu_normal_backend_frame_implementation,
        webgpu_normal_backend_frame_implementation_validation,
    )
    tile_input_contract = get_path(
        summary,
        ["webgpuTileAwareRenderInputContract"],
        {},
    )
    layout_contract = get_path(
        summary,
        ["webgpuGpuOwnedTileListLayoutContract"],
        get_path(
            webgpu_normal_backend_frame_implementation,
            ["webgpuGpuOwnedTileListLayoutContract"],
            {},
        ),
    )
    phase_step = get_path(summary, ["phaseStep"])
    generated_fields = get_path(layout_contract, ["generatedLayoutFields"], [])
    deferred_fields = get_path(layout_contract, ["deferredLayoutFields"], [])
    gpu_owned_layout_ready = (
        get_path(layout_contract, ["gpuOwnedTileListLayoutReady"]) is True
        and get_path(layout_contract, ["offsetCountTableCreated"]) is True
        and get_path(layout_contract, ["splatReferenceListCreated"]) is True
        and get_path(layout_contract, ["tileListConsumerReady"]) is True
        and get_path(layout_contract, ["tileListConsumerConsumed"]) is True
        and get_path(layout_contract, ["consumerFollowedOffsetCountTable"]) is True
        and get_path(layout_contract, ["totalTileReferenceCount"], 0) > 0
    )
    step83_tile_input_preserved = (
        get_path(tile_input_contract, ["tileAwareRenderInputReady"]) is True
        and get_path(tile_input_contract, ["tileAwareConsumerConsumed"]) is True
        and get_path(tile_input_contract, ["generatedTileRecordCount"], 0) > 0
    )
    footprint_baseline_preserved = (
        step82.get("webgpuComputedFootprintPayload") is True
        and step82.get("normalBackendConsumedComputedFootprintPayload") is True
        and step82.get("computedConicConsumed") is True
    )
    current_texture_ready = (
        step82.get("currentTextureConnectionReady") is True
        and step82.get("currentTextureReadbackMatchesAdapterOutput") is True
    )
    success = (
        phase_step == "phase3-step84"
        and gpu_owned_layout_ready
        and step83_tile_input_preserved
        and footprint_baseline_preserved
        and current_texture_ready
        and step82.get("webgl2HybridRenderingPrevented") is True
        and step82.get("fallbackSamplesMixed") is False
        and step82.get("noFallbackMixing") is True
    )
    blocked_reason = None
    if not success:
        if phase_step != "phase3-step84":
            blocked_reason = "summary-phase-step-is-not-phase3-step84"
        elif not gpu_owned_layout_ready:
            blocked_reason = "gpu-owned-tile-list-layout-or-consumer-not-ready"
        elif not step83_tile_input_preserved:
            blocked_reason = "step83-tile-aware-render-input-not-preserved"
        elif not footprint_baseline_preserved:
            blocked_reason = "step82-computed-footprint-baseline-not-preserved"
        elif not current_texture_ready:
            blocked_reason = "currentTexture-path-not-ready"
        elif step82.get("webgl2HybridRenderingPrevented") is not True:
            blocked_reason = "webgl2-hybrid-rendering-not-prevented"
        elif (
            step82.get("fallbackSamplesMixed") is not False
            or step82.get("noFallbackMixing") is not True
        ):
            blocked_reason = "fallback-samples-mixed-or-not-proven-suppressed"
    return {
        "step84Decision": "success" if success else "blocked",
        "step84BlockedReason": blocked_reason,
        "selectedApproach": "A/C-fixed-capacity-gpu-owned-tile-list-layout",
        "phaseStep": phase_step,
        "step84SummaryApplies": phase_step == "phase3-step84",
        "gpuOwnedTileListLayoutReady": get_path(
            layout_contract, ["gpuOwnedTileListLayoutReady"]
        ),
        "tileListConsumerReady": get_path(
            layout_contract, ["tileListConsumerReady"]
        ),
        "tileListConsumerConsumed": get_path(
            layout_contract, ["tileListConsumerConsumed"]
        ),
        "tileListConsumerReadbackCompleted": get_path(
            layout_contract, ["tileListConsumerReadbackCompleted"]
        ),
        "consumerFollowedOffsetCountTable": get_path(
            layout_contract, ["consumerFollowedOffsetCountTable"]
        ),
        "offsetCountTableCreated": get_path(
            layout_contract, ["offsetCountTableCreated"]
        ),
        "splatReferenceListCreated": get_path(
            layout_contract, ["splatReferenceListCreated"]
        ),
        "referenceListStoresDepthKey": get_path(
            layout_contract, ["referenceListStoresDepthKey"]
        ),
        "referenceListStoresSortKey": get_path(
            layout_contract, ["referenceListStoresSortKey"]
        ),
        "candidateTileRecordCount": get_path(
            layout_contract, ["candidateTileRecordCount"], 0
        ),
        "tileCount": get_path(layout_contract, ["tileCount"], 0),
        "tileSize": get_path(layout_contract, ["tileSize"]),
        "maxRefsPerTile": get_path(layout_contract, ["maxRefsPerTile"]),
        "totalTileReferenceCount": get_path(
            layout_contract, ["totalTileReferenceCount"], 0
        ),
        "consumedTileReferenceCount": get_path(
            layout_contract, ["consumedTileReferenceCount"], 0
        ),
        "nonEmptyTileCount": get_path(
            layout_contract, ["nonEmptyTileCount"], 0
        ),
        "maxRefsPerTileObserved": get_path(
            layout_contract, ["maxRefsPerTileObserved"], 0
        ),
        "overflowCount": get_path(layout_contract, ["overflowCount"], 0),
        "generatedLayoutFields": generated_fields,
        "deferredLayoutFields": deferred_fields,
        "tileListLayoutClassification": get_path(
            layout_contract, ["tileListLayoutClassification"]
        ),
        "gpuOwnedOffsetCountTableGenerated":
            "gpu-owned-offset-count-table" in generated_fields,
        "gpuOwnedSplatReferenceListGenerated":
            "gpu-owned-splat-reference-list" in generated_fields,
        "depthKeyStoredForNextSort": get_path(
            layout_contract, ["referenceListStoresDepthKey"]
        ),
        "sortKeyStoredForNextSort": get_path(
            layout_contract, ["referenceListStoresSortKey"]
        ),
        "nextDepthSortInputReady": get_path(
            layout_contract, ["nextDepthSortInputReady"]
        ),
        "nextTileCompositorInputReady": get_path(
            layout_contract, ["nextTileCompositorInputReady"]
        ),
        "normalBackendOnlyMetadataPath": not gpu_owned_layout_ready,
        "step83TileAwareInputPreserved": step83_tile_input_preserved,
        "step82ComputedFootprintPayloadPreserved":
            footprint_baseline_preserved,
        "step81ComputedAttributesPreserved":
            step82.get("step81ComputedRenderAttributesPreserved"),
        "fullParallelPrefixSumInWgsl": get_path(
            layout_contract, ["fullParallelPrefixSumInWgsl"]
        ),
        "fullTileListCompactionInWgsl": get_path(
            layout_contract, ["fullTileListCompactionInWgsl"]
        ),
        "fullDepthSortInWgsl": get_path(
            layout_contract, ["fullDepthSortInWgsl"]
        ),
        "finalTileCompositorImplemented": get_path(
            layout_contract, ["finalTileCompositorImplemented"]
        ),
        "fullPrefixSumDeferred":
            "parallel-prefix-sum-offset-compaction" in deferred_fields,
        "fullCompactionDeferred":
            "overflow-resize-second-pass" in deferred_fields,
        "fullDepthSortDeferred":
            "full-depth-sort-dispatch" in deferred_fields,
        "finalTileCompositorDeferred":
            "final-tile-compositor" in deferred_fields,
        "fullGpuTileListPipelineSuccessClaimed": (
            get_path(layout_contract, ["fullParallelPrefixSumInWgsl"]) is True
            and get_path(layout_contract, ["fullTileListCompactionInWgsl"]) is True
            and get_path(layout_contract, ["fullDepthSortInWgsl"]) is True
            and get_path(layout_contract, ["finalTileCompositorImplemented"]) is True
        ),
        "currentTextureConnectionReady": step82.get(
            "currentTextureConnectionReady"
        ),
        "currentTextureReadbackMatchesAdapterOutput": step82.get(
            "currentTextureReadbackMatchesAdapterOutput"
        ),
        "webgpuExclusiveGuard": step82.get("webgpuExclusiveGuard"),
        "webgl2HybridRenderingPrevented": step82.get(
            "webgl2HybridRenderingPrevented"
        ),
        "fallbackSamplesMixed": step82.get("fallbackSamplesMixed"),
        "noFallbackMixing": step82.get("noFallbackMixing"),
        "firstValidationFailures": step82.get("firstValidationFailures", []),
    }


def build_step85_webgpu_tile_list_compositor_summary(
    summary: Dict[str, Any],
    webgpu_visible_record_camera_aware_visible_output: Dict[str, Any],
    webgpu_owned_camera_aware_visible_output: Dict[str, Any],
    validation_assisted_camera_aware_visible_output: Dict[str, Any],
    webgpu_normal_backend_frame_implementation: Dict[str, Any],
    webgpu_normal_backend_frame_implementation_validation: Dict[str, Any],
) -> Dict[str, Any]:
    step84 = build_step84_webgpu_gpu_owned_tile_list_layout_summary(
        summary,
        webgpu_visible_record_camera_aware_visible_output,
        webgpu_owned_camera_aware_visible_output,
        validation_assisted_camera_aware_visible_output,
        webgpu_normal_backend_frame_implementation,
        webgpu_normal_backend_frame_implementation_validation,
    )
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        get_path(
            webgpu_normal_backend_frame_implementation,
            ["webgpuTileListCompositorContract"],
            {},
        ),
    )
    phase_step = get_path(summary, ["phaseStep"])
    generated_fields = get_path(
        compositor_contract, ["generatedCompositorFields"], []
    )
    deferred_fields = get_path(
        compositor_contract, ["deferredCompositorFields"], []
    )
    compositor_ready = (
        get_path(compositor_contract, ["tileCompositorReady"]) is True
        and get_path(compositor_contract, ["compositorPassSubmitted"]) is True
        and get_path(compositor_contract, ["compositorReadOffsetCountTable"]) is True
        and get_path(compositor_contract, ["compositorTraversedReferenceList"]) is True
        and get_path(compositor_contract, ["outputTextureCreated"]) is True
        and get_path(compositor_contract, ["outputTextureWritten"]) is True
        and get_path(compositor_contract, ["compositedReferenceCount"], 0) > 0
    )
    reference_count_matches = (
        get_path(compositor_contract, ["compositedReferenceCount"], 0)
        == get_path(compositor_contract, ["sourceTotalTileReferenceCount"], -1)
        == step84.get("totalTileReferenceCount")
    )
    processed_tile_count = get_path(
        compositor_contract, ["processedTileCount"], None
    )
    if processed_tile_count is None:
        output_width = get_path(compositor_contract, ["outputWidth"], 0) or 0
        output_height = get_path(compositor_contract, ["outputHeight"], 0) or 0
        processed_tile_count = output_width * output_height
    composited_tile_count = get_path(
        compositor_contract, ["compositedTileCount"], 0
    )
    non_empty_composited_tile_count = get_path(
        compositor_contract, ["nonEmptyCompositedTileCount"], 0
    )
    tile_count_aggregation_consistent = (
        processed_tile_count >= non_empty_composited_tile_count > 0
        and composited_tile_count == non_empty_composited_tile_count
    )
    success = (
        phase_step == "phase3-step85"
        and compositor_ready
        and reference_count_matches
        and tile_count_aggregation_consistent
        and step84.get("gpuOwnedTileListLayoutReady") is True
        and step84.get("step83TileAwareInputPreserved") is True
        and step84.get("step82ComputedFootprintPayloadPreserved") is True
        and step84.get("step81ComputedAttributesPreserved") is True
        and step84.get("currentTextureConnectionReady") is True
        and step84.get("currentTextureReadbackMatchesAdapterOutput") is True
        and step84.get("webgl2HybridRenderingPrevented") is True
        and step84.get("fallbackSamplesMixed") is False
        and step84.get("noFallbackMixing") is True
    )
    blocked_reason = None
    if not success:
        if phase_step != "phase3-step85":
            blocked_reason = "summary-phase-step-is-not-phase3-step85"
        elif not compositor_ready:
            blocked_reason = "webgpu-tile-list-compositor-not-ready"
        elif not reference_count_matches:
            blocked_reason = "compositor-reference-count-does-not-match-source-tile-list"
        elif not tile_count_aggregation_consistent:
            blocked_reason = "compositor-tile-count-aggregation-inconsistent"
        elif step84.get("gpuOwnedTileListLayoutReady") is not True:
            blocked_reason = "step84-gpu-owned-tile-list-layout-not-preserved"
        elif step84.get("currentTextureConnectionReady") is not True:
            blocked_reason = "currentTexture-path-not-ready"
        elif step84.get("webgl2HybridRenderingPrevented") is not True:
            blocked_reason = "webgl2-hybrid-rendering-not-prevented"
        elif (
            step84.get("fallbackSamplesMixed") is not False
            or step84.get("noFallbackMixing") is not True
        ):
            blocked_reason = "fallback-samples-mixed-or-not-proven-suppressed"
    return {
        "step85Decision": "success" if success else "blocked",
        "step85BlockedReason": blocked_reason,
        "selectedApproach": "A/B/C-partial-webgpu-tile-list-compositor",
        "phaseStep": phase_step,
        "step85SummaryApplies": phase_step == "phase3-step85",
        "tileCompositorReady": get_path(
            compositor_contract, ["tileCompositorReady"]
        ),
        "compositorPassSubmitted": get_path(
            compositor_contract, ["compositorPassSubmitted"]
        ),
        "compositorReadOffsetCountTable": get_path(
            compositor_contract, ["compositorReadOffsetCountTable"]
        ),
        "compositorTraversedReferenceList": get_path(
            compositor_contract, ["compositorTraversedReferenceList"]
        ),
        "outputTextureCreated": get_path(
            compositor_contract, ["outputTextureCreated"]
        ),
        "outputTextureWritten": get_path(
            compositor_contract, ["outputTextureWritten"]
        ),
        "outputTextureReadbackMatchesSummary": get_path(
            compositor_contract, ["outputTextureReadbackMatchesSummary"]
        ),
        "outputResourceKind": get_path(
            compositor_contract, ["outputResourceKind"]
        ),
        "outputFormat": get_path(compositor_contract, ["outputFormat"]),
        "outputWidth": get_path(compositor_contract, ["outputWidth"]),
        "outputHeight": get_path(compositor_contract, ["outputHeight"]),
        "processedTileCount": processed_tile_count,
        "compositedTileCount": composited_tile_count,
        "nonEmptyCompositedTileCount": non_empty_composited_tile_count,
        "tileCountAggregationConsistent": tile_count_aggregation_consistent,
        "compositedTileCountMeaning":
            "tiles-with-at-least-one-composited-reference",
        "processedTileCountMeaning": "all-tile-grid-entries-dispatched",
        "compositedReferenceCount": get_path(
            compositor_contract, ["compositedReferenceCount"], 0
        ),
        "sourceTotalTileReferenceCount": get_path(
            compositor_contract, ["sourceTotalTileReferenceCount"], 0
        ),
        "compositorReferenceCountMatchesStep84": reference_count_matches,
        "overflowCount": get_path(compositor_contract, ["overflowCount"], 0),
        "orderHandling": get_path(compositor_contract, ["orderHandling"]),
        "generatedCompositorFields": generated_fields,
        "deferredCompositorFields": deferred_fields,
        "compositorClassification": get_path(
            compositor_contract, ["compositorClassification"]
        ),
        "normalBackendOnlyMetadataPath": not compositor_ready,
        "fullDepthSortInWgsl": get_path(
            compositor_contract, ["fullDepthSortInWgsl"]
        ),
        "fullCudaParity": get_path(compositor_contract, ["fullCudaParity"]),
        "finalProductionTileCompositor": get_path(
            compositor_contract, ["finalProductionTileCompositor"]
        ),
        "fullDepthSortDeferred": (
            "full-depth-sort-dispatch" in deferred_fields
            or "full-parallel-per-tile-sort-dispatch" in deferred_fields
        ),
        "fullCudaParityDeferred": "cuda-compositor-parity" in deferred_fields,
        "finalProductionCompositorDeferred":
            "final-production-tile-compositor" in deferred_fields,
        "fullWebGpuTileRendererSuccessClaimed": (
            get_path(compositor_contract, ["fullDepthSortInWgsl"]) is True
            and get_path(compositor_contract, ["fullCudaParity"]) is True
            and get_path(compositor_contract, ["finalProductionTileCompositor"]) is True
        ),
        "step84GpuOwnedTileListLayoutPreserved":
            step84.get("gpuOwnedTileListLayoutReady"),
        "step84TotalTileReferenceCount":
            step84.get("totalTileReferenceCount"),
        "step84NonEmptyTileCount": step84.get("nonEmptyTileCount"),
        "step84MaxRefsPerTileObserved":
            step84.get("maxRefsPerTileObserved"),
        "step84OverflowCount": step84.get("overflowCount"),
        "step83TileAwareInputPreserved":
            step84.get("step83TileAwareInputPreserved"),
        "step82ComputedFootprintPayloadPreserved":
            step84.get("step82ComputedFootprintPayloadPreserved"),
        "step81ComputedAttributesPreserved":
            step84.get("step81ComputedAttributesPreserved"),
        "currentTexturePathMaintained": get_path(
            compositor_contract, ["currentTexturePathMaintained"]
        ),
        "currentTextureConnectionReady": step84.get(
            "currentTextureConnectionReady"
        ),
        "currentTextureReadbackMatchesAdapterOutput": step84.get(
            "currentTextureReadbackMatchesAdapterOutput"
        ),
        "webgpuExclusiveGuard": step84.get("webgpuExclusiveGuard"),
        "webgl2HybridRenderingPrevented": step84.get(
            "webgl2HybridRenderingPrevented"
        ),
        "fallbackSamplesMixed": step84.get("fallbackSamplesMixed"),
        "noFallbackMixing": step84.get("noFallbackMixing"),
        "firstValidationFailures": step84.get("firstValidationFailures", []),
    }


def build_step86_backend_boundary_and_dirty_contract_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    boundary_contract = get_path(
        summary,
        ["webgpuPhase3BackendBoundaryContract"],
        {},
    )
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    tile_list_contract = get_path(
        summary,
        ["webgpuGpuOwnedTileListLayoutContract"],
        {},
    )
    tile_input_contract = get_path(
        summary,
        ["webgpuTileAwareRenderInputContract"],
        {},
    )
    phase_step = get_path(summary, ["phaseStep"])
    dirty_flags = {
        "dirtyCameraConstants": get_path(
            boundary_contract, ["dirtyCameraConstants"]
        ),
        "dirtyTimeState": get_path(boundary_contract, ["dirtyTimeState"]),
        "dirtyVisibleRecords": get_path(
            boundary_contract, ["dirtyVisibleRecords"]
        ),
        "dirtyTileList": get_path(boundary_contract, ["dirtyTileList"]),
        "dirtyCompositorInput": get_path(
            boundary_contract, ["dirtyCompositorInput"]
        ),
    }
    dirty_contract_ready = (
        get_path(boundary_contract, ["dirtyUpdateContractReady"]) is True
        and all(value is True for value in dirty_flags.values())
    )
    responsibility_boundary_ready = (
        get_path(boundary_contract, ["viewerShellOwnsUrlQueryAndCapture"]) is True
        and get_path(
            boundary_contract,
            ["viewerShellOwnsCameraCanvasAdapterConnection"],
        )
        is True
        and get_path(boundary_contract, ["threeAdapterOwnsCameraInput"]) is True
        and get_path(boundary_contract, ["threeAdapterOwnsOrbitControls"]) is True
        and get_path(boundary_contract, ["threeAdapterIsRenderingCore"]) is False
        and get_path(boundary_contract, ["webgpuOwnsBackendPasses"]) is True
        and get_path(boundary_contract, ["webgl2Role"])
        == "fallback-validation-regression-oracle"
        and get_path(boundary_contract, ["cudaReferenceRole"])
        == "fixed-reference-not-interactive-backend"
    )
    common_boundary_ready = (
        get_path(boundary_contract, ["commonContractBoundaryReady"]) is True
        and get_path(boundary_contract, ["backendRecordFormatShared"]) is True
        and get_path(
            boundary_contract, ["independentBackendRecordFormatsAdded"]
        )
        is False
    )
    tools_boundary_ready = (
        get_path(
            boundary_contract,
            ["toolsOwnCaptureCommandGeneration"],
            phase_step == "phase3-step86",
        )
        is True
        and get_path(
            boundary_contract,
            ["toolsOwnStepSummary"],
            phase_step == "phase3-step86",
        )
        is True
        and get_path(
            boundary_contract,
            ["toolsOwnContractValidation"],
            phase_step == "phase3-step86",
        )
        is True
        and get_path(
            boundary_contract,
            ["toolsDoNotOwnRuntimeBackend"],
            phase_step == "phase3-step86",
        )
        is True
    )
    step85_tile_compositor_path_preserved = (
        get_path(
            boundary_contract,
            ["step85TileCompositorPathPreserved"],
            get_path(compositor_contract, ["tileCompositorReady"]) is True,
        )
        is True
    )
    step85_current_texture_path_maintained = (
        get_path(
            boundary_contract,
            ["step85CurrentTexturePathMaintained"],
            get_path(compositor_contract, ["currentTexturePathMaintained"]) is True,
        )
        is True
    )
    step85_current_texture_connection_ready = (
        get_path(
            boundary_contract,
            ["step85CurrentTextureConnectionReady"],
            get_path(compositor_contract, ["currentTexturePathMaintained"]) is True,
        )
        is True
    )
    step85_current_texture_readback_matches = (
        get_path(
            boundary_contract,
            ["step85CurrentTextureReadbackMatchesAdapterOutput"],
            get_path(compositor_contract, ["outputTextureReadbackMatchesSummary"])
            is True,
        )
        is True
    )
    step85_preserved = (
        get_path(boundary_contract, ["step85RuntimePathPreserved"]) is True
        and get_path(compositor_contract, ["tileCompositorReady"]) is True
        and get_path(tile_list_contract, ["gpuOwnedTileListLayoutReady"]) is True
        and get_path(tile_input_contract, ["tileAwareRenderInputReady"]) is True
        and step85_tile_compositor_path_preserved
        and step85_current_texture_path_maintained
        and step85_current_texture_connection_ready
        and step85_current_texture_readback_matches
    )
    no_hybrid_or_full_residency = (
        get_path(boundary_contract, ["webgpuWebgl2SameFramePresentationMixed"])
        is False
        and get_path(boundary_contract, ["fullDatasetGpuResidencyRequired"])
        is False
    )
    success = (
        phase_step == "phase3-step86"
        and get_path(boundary_contract, ["phase3BackendBoundaryReady"]) is True
        and dirty_contract_ready
        and responsibility_boundary_ready
        and tools_boundary_ready
        and common_boundary_ready
        and step85_preserved
        and no_hybrid_or_full_residency
    )
    blocked_reason = None
    if not success:
        if phase_step != "phase3-step86":
            blocked_reason = "summary-phase-step-is-not-phase3-step86"
        elif not dirty_contract_ready:
            blocked_reason = "dirty-update-contract-not-ready"
        elif not responsibility_boundary_ready:
            blocked_reason = "phase3-responsibility-boundary-not-ready"
        elif not tools_boundary_ready:
            blocked_reason = "tools-responsibility-boundary-not-ready"
        elif not common_boundary_ready:
            blocked_reason = "common-contract-boundary-not-ready"
        elif not step85_preserved:
            blocked_reason = "step85-runtime-path-not-preserved"
        elif not no_hybrid_or_full_residency:
            blocked_reason = "hybrid-presentation-or-full-residency-constraint-violated"
    return {
        "step86Decision": "success" if success else "blocked",
        "step86BlockedReason": blocked_reason,
        "selectedApproach": "A+B-with-small-boundary-comment",
        "phaseStep": phase_step,
        "step86SummaryApplies": phase_step == "phase3-step86",
        "phase3BackendBoundaryReady": get_path(
            boundary_contract, ["phase3BackendBoundaryReady"]
        ),
        "dirtyUpdateContractReady": dirty_contract_ready,
        **dirty_flags,
        "viewerShellOwnsUrlQueryAndCapture": get_path(
            boundary_contract, ["viewerShellOwnsUrlQueryAndCapture"]
        ),
        "viewerShellOwnsCameraCanvasAdapterConnection": get_path(
            boundary_contract, ["viewerShellOwnsCameraCanvasAdapterConnection"]
        ),
        "threeAdapterOwnsCameraInput": get_path(
            boundary_contract, ["threeAdapterOwnsCameraInput"]
        ),
        "threeAdapterOwnsOrbitControls": get_path(
            boundary_contract, ["threeAdapterOwnsOrbitControls"]
        ),
        "threeAdapterIsRenderingCore": get_path(
            boundary_contract, ["threeAdapterIsRenderingCore"]
        ),
        "webgpuOwnsBackendPasses": get_path(
            boundary_contract, ["webgpuOwnsBackendPasses"]
        ),
        "webgl2Role": get_path(boundary_contract, ["webgl2Role"]),
        "cudaReferenceRole": get_path(boundary_contract, ["cudaReferenceRole"]),
        "toolsOwnCaptureCommandGeneration": get_path(
            boundary_contract,
            ["toolsOwnCaptureCommandGeneration"],
            phase_step == "phase3-step86",
        ),
        "toolsOwnStepSummary": get_path(
            boundary_contract,
            ["toolsOwnStepSummary"],
            phase_step == "phase3-step86",
        ),
        "toolsOwnContractValidation": get_path(
            boundary_contract,
            ["toolsOwnContractValidation"],
            phase_step == "phase3-step86",
        ),
        "toolsDoNotOwnRuntimeBackend": get_path(
            boundary_contract,
            ["toolsDoNotOwnRuntimeBackend"],
            phase_step == "phase3-step86",
        ),
        "toolsBoundaryReady": tools_boundary_ready,
        "commonContractBoundaryReady": common_boundary_ready,
        "backendRecordFormatShared": get_path(
            boundary_contract, ["backendRecordFormatShared"]
        ),
        "independentBackendRecordFormatsAdded": get_path(
            boundary_contract, ["independentBackendRecordFormatsAdded"]
        ),
        "fullDatasetGpuResidencyRequired": get_path(
            boundary_contract, ["fullDatasetGpuResidencyRequired"]
        ),
        "webgpuWebgl2SameFramePresentationMixed": get_path(
            boundary_contract, ["webgpuWebgl2SameFramePresentationMixed"]
        ),
        "viewerAppGpuNewWebGpuPassResponsibilitiesAdded": get_path(
            boundary_contract,
            ["viewerAppGpuNewWebGpuPassResponsibilitiesAdded"],
        ),
        "step85RuntimePathPreserved": step85_preserved,
        "step85TileCompositorPathPreserved":
            step85_tile_compositor_path_preserved,
        "step85CurrentTexturePathMaintained":
            step85_current_texture_path_maintained,
        "step85CurrentTextureConnectionReady":
            step85_current_texture_connection_ready,
        "step85CurrentTextureReadbackMatchesAdapterOutput":
            step85_current_texture_readback_matches,
        "step85CurrentTexturePreservationSource": get_path(
            boundary_contract,
            ["step85CurrentTexturePreservationSource"],
            "step85-tile-compositor-contract-currentTexturePathMaintained-and-outputTextureReadbackMatchesSummary",
        ),
        "nextDepthSortBoundaryReady": get_path(
            boundary_contract, ["nextDepthSortBoundaryReady"]
        ),
        "nextFinalCompositorBoundaryReady": get_path(
            boundary_contract, ["nextFinalCompositorBoundaryReady"]
        ),
        "nextChunkLodStreamingBoundaryReady": get_path(
            boundary_contract, ["nextChunkLodStreamingBoundaryReady"]
        ),
        "reason": get_path(boundary_contract, ["reason"]),
    }


def build_step87_tile_depth_ordering_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    boundary_contract = get_path(
        summary,
        ["webgpuPhase3BackendBoundaryContract"],
        {},
    )
    tile_list_contract = get_path(
        summary,
        ["webgpuGpuOwnedTileListLayoutContract"],
        {},
    )
    depth_contract = get_path(
        compositor_contract,
        ["tileDepthOrderingContract"],
        {},
    )
    phase_step = get_path(summary, ["phaseStep"])
    deferred_fields = get_path(
        compositor_contract, ["deferredCompositorFields"], []
    )
    source_reference_count = get_path(
        depth_contract,
        ["sourceReferenceCount"],
        get_path(compositor_contract, ["sourceTotalTileReferenceCount"], 0),
    )
    ordered_reference_count = get_path(
        depth_contract,
        ["orderedReferenceCount"],
        get_path(compositor_contract, ["orderedReferenceCount"], 0),
    )
    ordered_reference_count_matches_source = (
        get_path(
            depth_contract,
            ["orderedReferenceCountMatchesSource"],
            get_path(
                compositor_contract,
                ["orderedReferenceCountMatchesSource"],
            ),
        )
        is True
        and ordered_reference_count == source_reference_count
        and source_reference_count > 0
    )
    step85_preserved = (
        get_path(compositor_contract, ["tileCompositorReady"]) is True
        and get_path(compositor_contract, ["compositorReadOffsetCountTable"]) is True
        and get_path(compositor_contract, ["compositorTraversedReferenceList"]) is True
        and get_path(compositor_contract, ["outputTextureWritten"]) is True
        and get_path(compositor_contract, ["currentTexturePathMaintained"]) is True
    )
    step86_preserved = (
        get_path(boundary_contract, ["phase3BackendBoundaryReady"]) is True
        and get_path(boundary_contract, ["dirtyUpdateContractReady"]) is True
        and get_path(boundary_contract, ["dirtyTileList"]) is True
        and get_path(boundary_contract, ["dirtyCompositorInput"]) is True
        and get_path(boundary_contract, ["toolsDoNotOwnRuntimeBackend"]) is True
        and get_path(boundary_contract, ["backendRecordFormatShared"]) is True
        and get_path(
            boundary_contract, ["webgpuWebgl2SameFramePresentationMixed"]
        )
        is False
    )
    current_texture_preserved = (
        get_path(boundary_contract, ["step85CurrentTextureConnectionReady"]) is True
        and get_path(
            boundary_contract,
            ["step85CurrentTextureReadbackMatchesAdapterOutput"],
        )
        is True
    )
    tile_depth_ordering_ready = (
        get_path(depth_contract, ["tileDepthOrderingReady"]) is True
        or get_path(compositor_contract, ["tileDepthOrderingReady"]) is True
    )
    success = (
        phase_step == "phase3-step87"
        and tile_depth_ordering_ready
        and get_path(depth_contract, ["depthOrderPassSubmitted"]) is True
        and get_path(depth_contract, ["orderAwareCompositorUsed"]) is True
        and get_path(depth_contract, ["depthKeyConsumed"]) is True
        and get_path(depth_contract, ["sortKeyConsumed"]) is True
        and get_path(
            depth_contract,
            ["compositorConsumedDepthOrderedReferences"],
        )
        is True
        and ordered_reference_count_matches_source
        and get_path(tile_list_contract, ["gpuOwnedTileListLayoutReady"]) is True
        and step85_preserved
        and step86_preserved
        and current_texture_preserved
    )
    blocked_reason = None
    if not success:
        if phase_step != "phase3-step87":
            blocked_reason = "summary-phase-step-is-not-phase3-step87"
        elif not tile_depth_ordering_ready:
            blocked_reason = "tile-depth-ordering-not-ready"
        elif get_path(depth_contract, ["orderAwareCompositorUsed"]) is not True:
            blocked_reason = "order-aware-compositor-not-used"
        elif get_path(depth_contract, ["depthKeyConsumed"]) is not True:
            blocked_reason = "depth-key-not-consumed"
        elif get_path(depth_contract, ["sortKeyConsumed"]) is not True:
            blocked_reason = "sort-key-not-consumed"
        elif not ordered_reference_count_matches_source:
            blocked_reason = "ordered-reference-count-does-not-match-source"
        elif not step85_preserved:
            blocked_reason = "step85-tile-compositor-path-not-preserved"
        elif not step86_preserved:
            blocked_reason = "step86-boundary-contract-not-preserved"
        elif not current_texture_preserved:
            blocked_reason = "currentTexture-path-not-preserved"
    return {
        "step87Decision": "success" if success else "blocked",
        "step87BlockedReason": blocked_reason,
        "selectedApproach": "B-order-aware-compositor-with-depth-key-consumption",
        "phaseStep": phase_step,
        "step87SummaryApplies": phase_step == "phase3-step87",
        "tileDepthOrderingReady": tile_depth_ordering_ready,
        "depthOrderPassSubmitted": get_path(
            depth_contract, ["depthOrderPassSubmitted"]
        ),
        "orderAwareCompositorUsed": get_path(
            depth_contract, ["orderAwareCompositorUsed"]
        ),
        "depthKeyConsumed": get_path(depth_contract, ["depthKeyConsumed"]),
        "sortKeyConsumed": get_path(depth_contract, ["sortKeyConsumed"]),
        "compositorConsumedDepthOrderedReferences": get_path(
            depth_contract, ["compositorConsumedDepthOrderedReferences"]
        ),
        "orderedReferenceCount": ordered_reference_count,
        "sourceReferenceCount": source_reference_count,
        "orderedReferenceCountMatchesSource":
            ordered_reference_count_matches_source,
        "orderHandling": get_path(depth_contract, ["orderHandling"]),
        "fixedCapacityPerTile": get_path(
            depth_contract, ["fixedCapacityPerTile"]
        ),
        "fullParallelPerTileSortInWgsl": get_path(
            depth_contract, ["fullParallelPerTileSortInWgsl"]
        ),
        "fullCudaDepthParity": get_path(
            depth_contract, ["fullCudaDepthParity"]
        ),
        "fullCudaDepthParityDeferred": get_path(
            depth_contract,
            ["fullCudaDepthParityDeferred"],
            "cuda-compositor-parity" in deferred_fields,
        ),
        "finalProductionCompositor": get_path(
            depth_contract, ["finalProductionCompositor"]
        ),
        "finalProductionCompositorDeferred": get_path(
            depth_contract,
            ["finalProductionCompositorDeferred"],
            "final-production-tile-compositor" in deferred_fields,
        ),
        "step85TileCompositorPathPreserved": step85_preserved,
        "step86BoundaryContractPreserved": step86_preserved,
        "step84GpuOwnedTileListLayoutPreserved": get_path(
            tile_list_contract, ["gpuOwnedTileListLayoutReady"]
        ),
        "step85CurrentTexturePathMaintained": get_path(
            compositor_contract, ["currentTexturePathMaintained"]
        ),
        "step85CurrentTextureConnectionReady": get_path(
            boundary_contract, ["step85CurrentTextureConnectionReady"]
        ),
        "step85CurrentTextureReadbackMatchesAdapterOutput": get_path(
            boundary_contract,
            ["step85CurrentTextureReadbackMatchesAdapterOutput"],
        ),
        "webgpuWebgl2SameFramePresentationMixed": get_path(
            boundary_contract, ["webgpuWebgl2SameFramePresentationMixed"]
        ),
        "fallbackMixingPrevented": (
            get_path(boundary_contract, ["webgpuWebgl2SameFramePresentationMixed"])
            is False
            and get_path(boundary_contract, ["backendRecordFormatShared"]) is True
        ),
        "fullRendererSuccessClaimed": (
            get_path(depth_contract, ["fullParallelPerTileSortInWgsl"]) is True
            and get_path(depth_contract, ["fullCudaDepthParity"]) is True
            and get_path(depth_contract, ["finalProductionCompositor"]) is True
        ),
        "reason": get_path(depth_contract, ["reason"]),
    }


def build_step88_tile_compositor_frame_implementation_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    frame_contract = get_path(
        summary,
        ["webgpuTileCompositorFrameImplementation"],
        {},
    )
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    boundary_contract = get_path(
        summary,
        ["webgpuPhase3BackendBoundaryContract"],
        {},
    )
    depth_contract = get_path(
        compositor_contract,
        ["tileDepthOrderingContract"],
        {},
    )
    phase_step = get_path(summary, ["phaseStep"])
    step85_preserved = (
        get_path(frame_contract, ["step85TileCompositorPathPreserved"]) is True
        and get_path(compositor_contract, ["tileCompositorReady"]) is True
        and get_path(compositor_contract, ["compositorReadOffsetCountTable"]) is True
        and get_path(compositor_contract, ["compositorTraversedReferenceList"]) is True
        and get_path(compositor_contract, ["outputTextureWritten"]) is True
    )
    step86_preserved = (
        get_path(frame_contract, ["step86BoundaryContractPreserved"]) is True
        and get_path(boundary_contract, ["dirtyUpdateContractReady"]) is True
        and get_path(boundary_contract, ["toolsDoNotOwnRuntimeBackend"]) is True
        and get_path(boundary_contract, ["backendRecordFormatShared"]) is True
    )
    step87_preserved = (
        get_path(frame_contract, ["step87DepthOrderingPreserved"]) is True
        and get_path(depth_contract, ["tileDepthOrderingReady"]) is True
        and get_path(depth_contract, ["orderAwareCompositorUsed"]) is True
        and get_path(
            depth_contract,
            ["compositorConsumedDepthOrderedReferences"],
        )
        is True
    )
    current_texture_connected = (
        get_path(
            frame_contract,
            ["compositorOutputPresentedToCurrentTexture"],
        )
        is True
        and get_path(frame_contract, ["currentTextureConnectionReady"]) is True
        and get_path(
            frame_contract,
            ["currentTextureReadbackMatchesCompositorOutput"],
        )
        is True
    )
    pass_chain_ready = all(
        get_path(frame_contract, [field]) is True
        for field in [
            "statePassReady",
            "attributePassReady",
            "footprintPassReady",
            "tileInputPassReady",
            "tileListPassReady",
            "depthOrderingPassReady",
            "tileCompositorPassReady",
            "presentationPassReady",
        ]
    )
    normal_backend_bypassed = (
        get_path(frame_contract, ["normalBackendFrameImplementationUsed"]) is False
        and get_path(frame_contract, ["normalBackendPresentationUsed"]) is False
        and get_path(frame_contract, ["normalBackendPresentationBypassed"]) is True
        and get_path(frame_contract, ["normalBackendDependencyReduced"]) is True
    )
    fallback_mix_prevented = (
        get_path(frame_contract, ["webgpuWebgl2SameFramePresentationMixed"]) is False
        and get_path(frame_contract, ["fallbackMixingPrevented"]) is True
    )
    device_consistency_ready = (
        get_path(frame_contract, ["webgpuDeviceConsistencyReady"]) is True
        and get_path(frame_contract, ["presentationDeviceMatchesCompositorDevice"])
        is True
        and get_path(frame_contract, ["currentTextureViewFreshPerPresentation"])
        is True
        and get_path(frame_contract, ["currentTextureViewReusedAcrossFrames"])
        is False
        and get_path(frame_contract, ["staleTextureViewReuseDetected"]) is False
        and get_path(frame_contract, ["crossDeviceTextureViewUseDetected"]) is False
        and get_path(frame_contract, ["webgpuValidationErrorDetected"]) is False
        and get_path(frame_contract, ["invalidCommandBufferDetected"]) is False
        and get_path(frame_contract, ["queueSubmitFailureDetected"]) is False
    )
    heartbeat_presentation_ready = (
        get_path(frame_contract, ["presentationHeartbeatReady"]) is True
        and get_path(
            frame_contract,
            ["presentationHeartbeatRunsEveryViewerRaf"],
        )
        is True
        and get_path(
            frame_contract,
            ["presentationDecoupledFromCompositorUpdate"],
        )
        is True
        and get_path(frame_contract, ["lastValidCompositorOutputCached"])
        is True
        and get_path(
            frame_contract,
            ["lastValidCompositorOutputPresentedOnCleanFrames"],
        )
        is True
        and numeric_value(
            get_path(frame_contract, ["presentationHeartbeatFrameCount"]),
            0,
        )
        >= 1
        and get_path(
            frame_contract,
            ["presentationHeartbeatFrameCountMatchesSampledRaf"],
        )
        is True
        and get_path(
            frame_contract,
            ["dirtySkippedCompositorUpdateButPresentedCachedOutput"],
        )
        is True
        and get_path(
            frame_contract,
            ["noBlankFrameBetweenHeartbeatPresentations"],
        )
        is True
        and get_path(frame_contract, ["canvasVisibleOutputStableAcrossRaf"])
        is True
        and get_path(frame_contract, ["visualFlickerDetected"]) is False
    )
    final_present_source_ready = (
        get_path(frame_contract, ["finalPresentSourceTracingReady"]) is True
        and numeric_value(get_path(frame_contract, ["sampledRafCount"]), 0) >= 1
        and get_path(frame_contract, ["rafTraceRingBufferReady"]) is True
        and get_path(frame_contract, ["rafTraceRecordedFromViewerLoopStart"]) is True
        and numeric_value(
            get_path(frame_contract, ["rafTraceRingBufferFrameCount"]),
            0,
        )
        >= numeric_value(get_path(frame_contract, ["sampledRafCount"]), 0)
        and get_path(frame_contract, ["summaryCanDetectObservedFlicker"]) is True
    )
    required_steady_state_raf_count = numeric_value(
        get_path(frame_contract, ["requiredSteadyStateRafCount"]),
        0,
    )
    steady_state_presentation_ready = (
        get_path(frame_contract, ["steadyStateSamplingReady"]) is True
        and required_steady_state_raf_count >= 8
        and numeric_value(
            get_path(frame_contract, ["steadyStateSampledRafCount"]),
            0,
        )
        >= required_steady_state_raf_count
        and get_path(
            frame_contract,
            ["steadyStateTileCompositorOwnsFinalPresentation"],
        )
        is True
        and get_path(frame_contract, ["steadyStateFinalPresentSourceStable"])
        is True
        and get_path(frame_contract, ["steadyStateFinalPresentSourceAlternates"])
        is False
        and numeric_value(
            get_path(frame_contract, ["steadyStateBlankFrameCount"]),
            0,
        )
        == 0
        and numeric_value(
            get_path(frame_contract, ["steadyStateNoOpFrameCount"]),
            0,
        )
        == 0
        and numeric_value(
            get_path(frame_contract, ["steadyStateClearFrameCount"]),
            0,
        )
        == 0
        and numeric_value(
            get_path(frame_contract, ["steadyStateUnknownFrameCount"]),
            0,
        )
        == 0
        and numeric_value(
            get_path(frame_contract, ["steadyStateNormalBackendFrameCount"]),
            0,
        )
        == 0
        and numeric_value(
            get_path(frame_contract, ["steadyStateWebgl2FallbackFrameCount"]),
            0,
        )
        == 0
        and get_path(frame_contract, ["steadyStateVisualFlickerDetected"])
        is False
        and get_path(frame_contract, ["summaryCanDetectStartupTransient"])
        is True
        and get_path(frame_contract, ["summaryCanDetectSteadyStateFlicker"])
        is True
        and get_path(frame_contract, ["presentationPersistsAfterStartup"])
        is True
        and get_path(
            frame_contract,
            ["presentationPersistsAcrossSteadyStateRaf"],
        )
        is True
        and get_path(frame_contract, ["captureSteadyStateWaitTimedOut"]) is False
    )
    sampled_presentation_ready = (
        numeric_value(
            get_path(frame_contract, ["presentationSampleFrameCount"]),
            0,
        )
        >= 1
        and get_path(frame_contract, ["presentationAllSampledFramesNonBlank"])
        is True
        and get_path(frame_contract, ["presentationAlternatingBlankDetected"])
        is False
        and get_path(frame_contract, ["presentationStableVisualOutput"])
        is True
        and get_path(
            frame_contract,
            ["compositorOutputPresentedEverySampledFrame"],
        )
        is True
        and get_path(
            frame_contract,
            ["canvasClearBetweenCompositorFramesDetected"],
        )
        is False
        and get_path(frame_contract, ["viewerLoopPresentationCadenceStable"])
        is True
    )
    viewer_loop_presentation_persistent = (
        get_path(frame_contract, ["viewerLoopFrameImplementationActive"]) is True
        and get_path(frame_contract, ["frameImplementationRegisteredWithViewerLoop"])
        is True
        and get_path(frame_contract, ["compositorOutputPresentedByViewerLoop"])
        is True
        and get_path(frame_contract, ["presentationPersistsAfterDelay"]) is True
        and get_path(frame_contract, ["presentationPersistsAcrossAnimationFrames"])
        is True
        and numeric_value(
            get_path(frame_contract, ["animationFramePresentationCount"]),
            0,
        )
        >= 1
        and sampled_presentation_ready
        and heartbeat_presentation_ready
        and final_present_source_ready
        and steady_state_presentation_ready
        and get_path(
            frame_contract,
            ["canvasOverwriteAfterCompositorPresentationDetected"],
        )
        is False
        and get_path(
            frame_contract,
            ["normalBackendOverwriteAfterCompositorPresentationDetected"],
        )
        is False
        and get_path(
            frame_contract,
            ["fallbackOverwriteAfterCompositorPresentationDetected"],
        )
        is False
        and get_path(frame_contract, ["viewerLoopRuntimeFatalErrorDetected"])
        is False
    )
    success = (
        phase_step == "phase3-step88"
        and get_path(frame_contract, ["frameImplementationReady"]) is True
        and get_path(frame_contract, ["frameImplementationSelected"]) is True
        and get_path(frame_contract, ["frameImplementationExecuted"]) is True
        and get_path(frame_contract, ["webgpuOwnsFramePassChain"]) is True
        and pass_chain_ready
        and normal_backend_bypassed
        and current_texture_connected
        and get_path(
            frame_contract,
            ["currentTextureUsesWebGpuTileCompositorOutput"],
        )
        is True
        and get_path(frame_contract, ["presentationStableUntilCapture"]) is True
        and viewer_loop_presentation_persistent
        and step85_preserved
        and step86_preserved
        and step87_preserved
        and fallback_mix_prevented
        and device_consistency_ready
        and get_path(frame_contract, ["fullRendererSuccessClaimed"]) is False
    )
    blocked_reason = None
    if not success:
        if phase_step != "phase3-step88":
            blocked_reason = "summary-phase-step-is-not-phase3-step88"
        elif get_path(frame_contract, ["frameImplementationSelected"]) is not True:
            blocked_reason = "tile-compositor-frame-implementation-not-selected"
        elif get_path(frame_contract, ["frameImplementationExecuted"]) is not True:
            blocked_reason = "tile-compositor-frame-implementation-not-executed"
        elif not pass_chain_ready:
            blocked_reason = "webgpu-frame-pass-chain-not-ready"
        elif not normal_backend_bypassed:
            blocked_reason = "normal-backend-presentation-not-bypassed"
        elif get_path(frame_contract, ["webgpuValidationErrorDetected"]) is True:
            blocked_reason = "webgpu-validation-error-detected"
        elif get_path(frame_contract, ["invalidCommandBufferDetected"]) is True:
            blocked_reason = "invalid-command-buffer-detected"
        elif get_path(frame_contract, ["queueSubmitFailureDetected"]) is True:
            blocked_reason = "queue-submit-failure-detected"
        elif get_path(frame_contract, ["crossDeviceTextureViewUseDetected"]) is True:
            blocked_reason = "cross-device-texture-view-use-detected"
        elif get_path(frame_contract, ["staleTextureViewReuseDetected"]) is True:
            blocked_reason = "stale-texture-view-reuse-detected"
        elif not current_texture_connected:
            blocked_reason = "tile-compositor-output-not-connected-to-currentTexture"
        elif (
            get_path(
                frame_contract,
                ["currentTextureUsesWebGpuTileCompositorOutput"],
            )
            is not True
        ):
            blocked_reason = "currentTexture-source-is-not-webgpu-tile-compositor-output"
        elif get_path(frame_contract, ["presentationStableUntilCapture"]) is not True:
            blocked_reason = "tile-compositor-presentation-not-stable-until-capture"
        elif (
            get_path(frame_contract, ["viewerLoopRuntimeFatalErrorDetected"])
            is True
        ):
            blocked_reason = "viewer-loop-runtime-fatal-error-detected"
        elif get_path(frame_contract, ["webgpuDeviceConsistencyReady"]) is not True:
            blocked_reason = "webgpu-device-consistency-not-ready"
        elif (
            get_path(
                frame_contract,
                ["presentationDeviceMatchesCompositorDevice"],
            )
            is not True
        ):
            blocked_reason = "presentation-device-does-not-match-compositor-device"
        elif (
            get_path(frame_contract, ["currentTextureViewFreshPerPresentation"])
            is not True
        ):
            blocked_reason = "currentTexture-view-not-fresh-per-presentation"
        elif (
            get_path(frame_contract, ["currentTextureViewReusedAcrossFrames"])
            is True
        ):
            blocked_reason = "currentTexture-view-reused-across-frames"
        elif get_path(frame_contract, ["presentationStableVisualOutput"]) is not True:
            blocked_reason = "viewer-loop-tile-compositor-visual-output-not-stable"
        elif (
            get_path(frame_contract, ["presentationAllSampledFramesNonBlank"])
            is not True
        ):
            blocked_reason = "viewer-loop-tile-compositor-blank-frame-detected"
        elif (
            get_path(frame_contract, ["presentationAlternatingBlankDetected"])
            is True
        ):
            blocked_reason = "viewer-loop-tile-compositor-alternating-blank-detected"
        elif (
            get_path(
                frame_contract,
                ["compositorOutputPresentedEverySampledFrame"],
            )
            is not True
        ):
            blocked_reason = "viewer-loop-tile-compositor-not-presented-every-sampled-frame"
        elif (
            get_path(
                frame_contract,
                ["canvasClearBetweenCompositorFramesDetected"],
            )
            is True
        ):
            blocked_reason = "canvas-clear-between-compositor-frames-detected"
        elif (
            get_path(frame_contract, ["viewerLoopPresentationCadenceStable"])
            is not True
        ):
            blocked_reason = "viewer-loop-presentation-cadence-not-stable"
        elif get_path(frame_contract, ["presentationHeartbeatReady"]) is not True:
            blocked_reason = "presentation-heartbeat-not-ready"
        elif (
            get_path(
                frame_contract,
                ["presentationHeartbeatRunsEveryViewerRaf"],
            )
            is not True
        ):
            blocked_reason = "presentation-heartbeat-not-running-every-viewer-raf"
        elif (
            get_path(
                frame_contract,
                ["presentationDecoupledFromCompositorUpdate"],
            )
            is not True
        ):
            blocked_reason = "presentation-not-decoupled-from-compositor-update"
        elif get_path(frame_contract, ["lastValidCompositorOutputCached"]) is not True:
            blocked_reason = "last-valid-compositor-output-not-cached"
        elif (
            get_path(
                frame_contract,
                ["lastValidCompositorOutputPresentedOnCleanFrames"],
            )
            is not True
        ):
            blocked_reason = "cached-compositor-output-not-presented-on-clean-frames"
        elif (
            get_path(
                frame_contract,
                ["presentationHeartbeatFrameCountMatchesSampledRaf"],
            )
            is not True
        ):
            blocked_reason = "presentation-heartbeat-frame-count-does-not-match-sampled-raf"
        elif (
            get_path(
                frame_contract,
                ["dirtySkippedCompositorUpdateButPresentedCachedOutput"],
            )
            is not True
        ):
            blocked_reason = "clean-frame-did-not-present-cached-compositor-output"
        elif (
            get_path(
                frame_contract,
                ["noBlankFrameBetweenHeartbeatPresentations"],
            )
            is not True
        ):
            blocked_reason = "blank-frame-between-heartbeat-presentations"
        elif get_path(frame_contract, ["canvasVisibleOutputStableAcrossRaf"]) is not True:
            blocked_reason = "canvas-visible-output-not-stable-across-raf"
        elif get_path(frame_contract, ["visualFlickerDetected"]) is True:
            blocked_reason = "visual-flicker-detected"
        elif (
            get_path(frame_contract, ["finalPresentSourceTracingReady"])
            is not True
        ):
            blocked_reason = "final-present-source-tracing-not-ready"
        elif get_path(frame_contract, ["rafTraceRingBufferReady"]) is not True:
            blocked_reason = "raf-trace-ring-buffer-not-ready"
        elif (
            get_path(frame_contract, ["rafTraceRecordedFromViewerLoopStart"])
            is not True
        ):
            blocked_reason = "raf-trace-not-recorded-from-viewer-loop-start"
        elif required_steady_state_raf_count < 8:
            blocked_reason = "required-steady-state-raf-count-too-low"
        elif get_path(frame_contract, ["steadyStateSamplingReady"]) is not True:
            blocked_reason = "steady-state-sampling-not-ready"
        elif (
            numeric_value(
                get_path(frame_contract, ["steadyStateSampledRafCount"]),
                0,
            )
            < required_steady_state_raf_count
        ):
            blocked_reason = "steady-state-sampled-raf-count-too-low"
        elif (
            get_path(
                frame_contract,
                ["steadyStateTileCompositorOwnsFinalPresentation"],
            )
            is not True
        ):
            blocked_reason = "steady-state-final-presentation-not-owned-by-tile-compositor"
        elif (
            get_path(frame_contract, ["steadyStateFinalPresentSourceStable"])
            is not True
        ):
            blocked_reason = "steady-state-final-present-source-not-stable"
        elif (
            get_path(frame_contract, ["steadyStateFinalPresentSourceAlternates"])
            is True
        ):
            blocked_reason = "steady-state-final-present-source-alternates"
        elif numeric_value(
            get_path(frame_contract, ["steadyStateBlankFrameCount"]),
            0,
        ) > 0:
            blocked_reason = "steady-state-blank-frame-detected"
        elif numeric_value(
            get_path(frame_contract, ["steadyStateNoOpFrameCount"]),
            0,
        ) > 0:
            blocked_reason = "steady-state-no-op-final-present-detected"
        elif numeric_value(
            get_path(frame_contract, ["steadyStateClearFrameCount"]),
            0,
        ) > 0:
            blocked_reason = "steady-state-clear-final-present-detected"
        elif numeric_value(
            get_path(frame_contract, ["steadyStateUnknownFrameCount"]),
            0,
        ) > 0:
            blocked_reason = "steady-state-unknown-final-present-detected"
        elif numeric_value(
            get_path(frame_contract, ["steadyStateNormalBackendFrameCount"]),
            0,
        ) > 0:
            blocked_reason = "steady-state-normal-backend-final-present-detected"
        elif numeric_value(
            get_path(frame_contract, ["steadyStateWebgl2FallbackFrameCount"]),
            0,
        ) > 0:
            blocked_reason = "steady-state-webgl2-fallback-final-present-detected"
        elif (
            get_path(frame_contract, ["steadyStateVisualFlickerDetected"])
            is True
        ):
            blocked_reason = "steady-state-visual-flicker-detected"
        elif (
            get_path(frame_contract, ["summaryCanDetectStartupTransient"])
            is not True
        ):
            blocked_reason = "summary-cannot-detect-startup-transient"
        elif (
            get_path(frame_contract, ["summaryCanDetectSteadyStateFlicker"])
            is not True
        ):
            blocked_reason = "summary-cannot-detect-steady-state-flicker"
        elif (
            get_path(frame_contract, ["presentationPersistsAfterStartup"])
            is not True
        ):
            blocked_reason = "presentation-does-not-persist-after-startup"
        elif (
            get_path(
                frame_contract,
                ["presentationPersistsAcrossSteadyStateRaf"],
            )
            is not True
        ):
            blocked_reason = "presentation-does-not-persist-across-steady-state-raf"
        elif get_path(frame_contract, ["captureSteadyStateWaitTimedOut"]) is True:
            blocked_reason = "steady-state-sampling-wait-timed-out"
        elif (
            not steady_state_presentation_ready
            and get_path(frame_contract, ["tileCompositorOwnsFinalPresentation"])
            is not True
        ):
            blocked_reason = "tile-compositor-does-not-own-final-presentation"
        elif (
            not steady_state_presentation_ready
            and get_path(frame_contract, ["finalPresentSourceStable"])
            is not True
        ):
            blocked_reason = "final-present-source-not-stable"
        elif (
            not steady_state_presentation_ready
            and get_path(frame_contract, ["finalPresentSourceAlternates"])
            is True
        ):
            blocked_reason = "final-present-source-alternates"
        elif (
            not steady_state_presentation_ready
            and numeric_value(
                get_path(frame_contract, ["normalBackendFinalPresentFrameCount"]),
                0,
            )
            > 0
        ):
            blocked_reason = "normal-backend-final-present-detected"
        elif (
            not steady_state_presentation_ready
            and numeric_value(
                get_path(frame_contract, ["webgl2FallbackFinalPresentFrameCount"]),
                0,
            )
            > 0
        ):
            blocked_reason = "webgl2-fallback-final-present-detected"
        elif (
            not steady_state_presentation_ready
            and numeric_value(
                get_path(frame_contract, ["debugClearFinalPresentFrameCount"]),
                0,
            )
            > 0
        ):
            blocked_reason = "debug-clear-final-present-detected"
        elif (
            not steady_state_presentation_ready
            and numeric_value(
                get_path(frame_contract, ["canvasClearFinalPresentFrameCount"]),
                0,
            )
            > 0
        ):
            blocked_reason = "canvas-clear-final-present-detected"
        elif (
            not steady_state_presentation_ready
            and numeric_value(
                get_path(frame_contract, ["noOpFinalPresentFrameCount"]),
                0,
            )
            > 0
        ):
            blocked_reason = "no-op-final-present-detected"
        elif (
            not steady_state_presentation_ready
            and numeric_value(
                get_path(frame_contract, ["unknownFinalPresentFrameCount"]),
                0,
            )
            > 0
        ):
            blocked_reason = "unknown-final-present-detected"
        elif (
            get_path(frame_contract, ["summaryCanDetectObservedFlicker"])
            is not True
        ):
            blocked_reason = "summary-cannot-detect-observed-flicker"
        elif (
            not steady_state_presentation_ready
            and
            get_path(frame_contract, ["tileCompositorOwnsFinalPresentation"])
            is not True
        ):
            blocked_reason = "tile-compositor-does-not-own-final-presentation"
        elif not viewer_loop_presentation_persistent:
            blocked_reason = "viewer-loop-tile-compositor-presentation-not-persistent"
        elif not step85_preserved:
            blocked_reason = "step85-tile-compositor-path-not-preserved"
        elif not step86_preserved:
            blocked_reason = "step86-boundary-contract-not-preserved"
        elif not step87_preserved:
            blocked_reason = "step87-depth-ordering-not-preserved"
        elif not fallback_mix_prevented:
            blocked_reason = "fallback-or-webgl2-hybrid-mixing-not-prevented"
        elif get_path(frame_contract, ["frameImplementationReady"]) is not True:
            blocked_reason = "tile-compositor-frame-implementation-not-ready"
    return {
        "step88Decision": "success" if success else "blocked",
        "step88BlockedReason": blocked_reason,
        "selectedApproach": "A-new-webgpu-tile-compositor-frame-implementation",
        "phaseStep": phase_step,
        "step88SummaryApplies": phase_step == "phase3-step88",
        "frameImplementationKind": get_path(
            frame_contract,
            ["frameImplementationKind"],
        ),
        "selectedFrameImplementation": get_path(
            frame_contract,
            ["selectedFrameImplementation"],
            get_path(frame_contract, ["frameImplementationKind"]),
        ),
        "webgpuTileCompositorFrameImplementationReady": get_path(
            frame_contract,
            ["frameImplementationReady"],
        ),
        "frameImplementationMode": get_path(
            frame_contract,
            ["frameImplementationMode"],
        ),
        "frameImplementationReady": get_path(
            frame_contract,
            ["frameImplementationReady"],
        ),
        "frameImplementationSelected": get_path(
            frame_contract,
            ["frameImplementationSelected"],
        ),
        "frameImplementationExecuted": get_path(
            frame_contract,
            ["frameImplementationExecuted"],
        ),
        "webgpuOwnsFramePassChain": get_path(
            frame_contract,
            ["webgpuOwnsFramePassChain"],
        ),
        "statePassReady": get_path(frame_contract, ["statePassReady"]),
        "attributePassReady": get_path(frame_contract, ["attributePassReady"]),
        "footprintPassReady": get_path(frame_contract, ["footprintPassReady"]),
        "tileInputPassReady": get_path(frame_contract, ["tileInputPassReady"]),
        "tileListPassReady": get_path(frame_contract, ["tileListPassReady"]),
        "depthOrderingPassReady": get_path(
            frame_contract,
            ["depthOrderingPassReady"],
        ),
        "tileCompositorPassReady": get_path(
            frame_contract,
            ["tileCompositorPassReady"],
        ),
        "presentationPassReady": get_path(
            frame_contract,
            ["presentationPassReady"],
        ),
        "normalBackendPresentationUsed": get_path(
            frame_contract,
            ["normalBackendPresentationUsed"],
        ),
        "normalBackendFrameImplementationUsed": get_path(
            frame_contract,
            ["normalBackendFrameImplementationUsed"],
        ),
        "normalBackendPresentationBypassed": get_path(
            frame_contract,
            ["normalBackendPresentationBypassed"],
        ),
        "normalBackendDependencyReduced": get_path(
            frame_contract,
            ["normalBackendDependencyReduced"],
        ),
        "compositorOutputPresentedToCurrentTexture": get_path(
            frame_contract,
            ["compositorOutputPresentedToCurrentTexture"],
        ),
        "depthAwareCompositorPresented": get_path(
            frame_contract,
            ["depthAwareCompositorPresented"],
        ),
        "currentTextureConnectionReady": get_path(
            frame_contract,
            ["currentTextureConnectionReady"],
        ),
        "currentTextureSource": get_path(
            frame_contract,
            ["currentTextureSource"],
        ),
        "currentTextureUsesWebGpuTileCompositorOutput": get_path(
            frame_contract,
            ["currentTextureUsesWebGpuTileCompositorOutput"],
        ),
        "currentTextureReadbackMatchesCompositorOutput": get_path(
            frame_contract,
            ["currentTextureReadbackMatchesCompositorOutput"],
        ),
        "presentationFrameCount": get_path(
            frame_contract,
            ["presentationFrameCount"],
        ),
        "compositorPresentationFrameCount": get_path(
            frame_contract,
            ["compositorPresentationFrameCount"],
        ),
        "presentationStableUntilCapture": get_path(
            frame_contract,
            ["presentationStableUntilCapture"],
        ),
        "viewerLoopFrameImplementationActive": get_path(
            frame_contract,
            ["viewerLoopFrameImplementationActive"],
        ),
        "frameImplementationRegisteredWithViewerLoop": get_path(
            frame_contract,
            ["frameImplementationRegisteredWithViewerLoop"],
        ),
        "compositorOutputPresentedByViewerLoop": get_path(
            frame_contract,
            ["compositorOutputPresentedByViewerLoop"],
        ),
        "presentationPersistsAfterDelay": get_path(
            frame_contract,
            ["presentationPersistsAfterDelay"],
        ),
        "presentationPersistenceDelayMs": get_path(
            frame_contract,
            ["presentationPersistenceDelayMs"],
        ),
        "presentationPersistsAcrossAnimationFrames": get_path(
            frame_contract,
            ["presentationPersistsAcrossAnimationFrames"],
        ),
        "animationFramePresentationCount": get_path(
            frame_contract,
            ["animationFramePresentationCount"],
        ),
        "presentationSampleFrameCount": get_path(
            frame_contract,
            ["presentationSampleFrameCount"],
        ),
        "presentationNonBlankFrameCount": get_path(
            frame_contract,
            ["presentationNonBlankFrameCount"],
        ),
        "presentationBlankFrameCount": get_path(
            frame_contract,
            ["presentationBlankFrameCount"],
        ),
        "presentationAllSampledFramesNonBlank": get_path(
            frame_contract,
            ["presentationAllSampledFramesNonBlank"],
        ),
        "presentationAlternatingBlankDetected": get_path(
            frame_contract,
            ["presentationAlternatingBlankDetected"],
        ),
        "presentationStableVisualOutput": get_path(
            frame_contract,
            ["presentationStableVisualOutput"],
        ),
        "presentationNonzeroPixelRatioMin": get_path(
            frame_contract,
            ["presentationNonzeroPixelRatioMin"],
        ),
        "presentationNonzeroPixelRatioMax": get_path(
            frame_contract,
            ["presentationNonzeroPixelRatioMax"],
        ),
        "presentationFrameHashChanges": get_path(
            frame_contract,
            ["presentationFrameHashChanges"],
        ),
        "compositorOutputPresentedEverySampledFrame": get_path(
            frame_contract,
            ["compositorOutputPresentedEverySampledFrame"],
        ),
        "canvasClearBetweenCompositorFramesDetected": get_path(
            frame_contract,
            ["canvasClearBetweenCompositorFramesDetected"],
        ),
        "viewerLoopPresentationCadenceStable": get_path(
            frame_contract,
            ["viewerLoopPresentationCadenceStable"],
        ),
        "presentationHeartbeatReady": get_path(
            frame_contract,
            ["presentationHeartbeatReady"],
        ),
        "presentationHeartbeatRunsEveryViewerRaf": get_path(
            frame_contract,
            ["presentationHeartbeatRunsEveryViewerRaf"],
        ),
        "presentationDecoupledFromCompositorUpdate": get_path(
            frame_contract,
            ["presentationDecoupledFromCompositorUpdate"],
        ),
        "lastValidCompositorOutputCached": get_path(
            frame_contract,
            ["lastValidCompositorOutputCached"],
        ),
        "lastValidCompositorOutputPresentedOnCleanFrames": get_path(
            frame_contract,
            ["lastValidCompositorOutputPresentedOnCleanFrames"],
        ),
        "compositorUpdateFrameCount": get_path(
            frame_contract,
            ["compositorUpdateFrameCount"],
        ),
        "presentationHeartbeatFrameCount": get_path(
            frame_contract,
            ["presentationHeartbeatFrameCount"],
        ),
        "presentationHeartbeatFrameCountMatchesSampledRaf": get_path(
            frame_contract,
            ["presentationHeartbeatFrameCountMatchesSampledRaf"],
        ),
        "dirtySkippedCompositorUpdateButPresentedCachedOutput": get_path(
            frame_contract,
            ["dirtySkippedCompositorUpdateButPresentedCachedOutput"],
        ),
        "noBlankFrameBetweenHeartbeatPresentations": get_path(
            frame_contract,
            ["noBlankFrameBetweenHeartbeatPresentations"],
        ),
        "canvasVisibleOutputStableAcrossRaf": get_path(
            frame_contract,
            ["canvasVisibleOutputStableAcrossRaf"],
        ),
        "visualFlickerDetected": get_path(
            frame_contract,
            ["visualFlickerDetected"],
        ),
        "finalPresentSourceTracingReady": get_path(
            frame_contract,
            ["finalPresentSourceTracingReady"],
        ),
        "sampledRafCount": get_path(frame_contract, ["sampledRafCount"]),
        "tileCompositorFinalPresentFrameCount": get_path(
            frame_contract,
            ["tileCompositorFinalPresentFrameCount"],
        ),
        "heartbeatFinalPresentFrameCount": get_path(
            frame_contract,
            ["heartbeatFinalPresentFrameCount"],
        ),
        "normalBackendFinalPresentFrameCount": get_path(
            frame_contract,
            ["normalBackendFinalPresentFrameCount"],
        ),
        "webgl2FallbackFinalPresentFrameCount": get_path(
            frame_contract,
            ["webgl2FallbackFinalPresentFrameCount"],
        ),
        "debugClearFinalPresentFrameCount": get_path(
            frame_contract,
            ["debugClearFinalPresentFrameCount"],
        ),
        "canvasClearFinalPresentFrameCount": get_path(
            frame_contract,
            ["canvasClearFinalPresentFrameCount"],
        ),
        "noOpFinalPresentFrameCount": get_path(
            frame_contract,
            ["noOpFinalPresentFrameCount"],
        ),
        "unknownFinalPresentFrameCount": get_path(
            frame_contract,
            ["unknownFinalPresentFrameCount"],
        ),
        "finalPresentSourceStable": get_path(
            frame_contract,
            ["finalPresentSourceStable"],
        ),
        "finalPresentSourceAlternates": get_path(
            frame_contract,
            ["finalPresentSourceAlternates"],
        ),
        "finalPresentSourceSequence": get_path(
            frame_contract,
            ["finalPresentSourceSequence"],
        ),
        "tileCompositorOwnsFinalPresentation": get_path(
            frame_contract,
            ["tileCompositorOwnsFinalPresentation"],
        ),
        "summaryCanDetectObservedFlicker": get_path(
            frame_contract,
            ["summaryCanDetectObservedFlicker"],
        ),
        "rafTraceRingBufferReady": get_path(
            frame_contract,
            ["rafTraceRingBufferReady"],
        ),
        "rafTraceRecordedFromViewerLoopStart": get_path(
            frame_contract,
            ["rafTraceRecordedFromViewerLoopStart"],
        ),
        "rafTraceCapturedBeforeCommandStart": get_path(
            frame_contract,
            ["rafTraceCapturedBeforeCommandStart"],
        ),
        "rafTraceRingBufferFrameCount": get_path(
            frame_contract,
            ["rafTraceRingBufferFrameCount"],
        ),
        "requiredSteadyStateRafCount": get_path(
            frame_contract,
            ["requiredSteadyStateRafCount"],
        ),
        "startupTransientObserved": get_path(
            frame_contract,
            ["startupTransientObserved"],
        ),
        "startupTransientFrameCount": get_path(
            frame_contract,
            ["startupTransientFrameCount"],
        ),
        "startupTransientFinalPresentSourceSequence": get_path(
            frame_contract,
            ["startupTransientFinalPresentSourceSequence"],
        ),
        "firstValidCompositorOutputFrame": get_path(
            frame_contract,
            ["firstValidCompositorOutputFrame"],
        ),
        "steadyStateSamplingReady": get_path(
            frame_contract,
            ["steadyStateSamplingReady"],
        ),
        "steadyStateSampledRafCount": get_path(
            frame_contract,
            ["steadyStateSampledRafCount"],
        ),
        "steadyStateSamplingWindowStartFrame": get_path(
            frame_contract,
            ["steadyStateSamplingWindowStartFrame"],
        ),
        "steadyStateSamplingWindowEndFrame": get_path(
            frame_contract,
            ["steadyStateSamplingWindowEndFrame"],
        ),
        "steadyStateFinalPresentSourceSequence": get_path(
            frame_contract,
            ["steadyStateFinalPresentSourceSequence"],
        ),
        "steadyStateTileCompositorOwnsFinalPresentation": get_path(
            frame_contract,
            ["steadyStateTileCompositorOwnsFinalPresentation"],
        ),
        "steadyStateFinalPresentSourceStable": get_path(
            frame_contract,
            ["steadyStateFinalPresentSourceStable"],
        ),
        "steadyStateFinalPresentSourceAlternates": get_path(
            frame_contract,
            ["steadyStateFinalPresentSourceAlternates"],
        ),
        "steadyStateBlankFrameCount": get_path(
            frame_contract,
            ["steadyStateBlankFrameCount"],
        ),
        "steadyStateNoOpFrameCount": get_path(
            frame_contract,
            ["steadyStateNoOpFrameCount"],
        ),
        "steadyStateClearFrameCount": get_path(
            frame_contract,
            ["steadyStateClearFrameCount"],
        ),
        "steadyStateUnknownFrameCount": get_path(
            frame_contract,
            ["steadyStateUnknownFrameCount"],
        ),
        "steadyStateNormalBackendFrameCount": get_path(
            frame_contract,
            ["steadyStateNormalBackendFrameCount"],
        ),
        "steadyStateWebgl2FallbackFrameCount": get_path(
            frame_contract,
            ["steadyStateWebgl2FallbackFrameCount"],
        ),
        "steadyStateVisualFlickerDetected": get_path(
            frame_contract,
            ["steadyStateVisualFlickerDetected"],
        ),
        "summaryCanDetectStartupTransient": get_path(
            frame_contract,
            ["summaryCanDetectStartupTransient"],
        ),
        "summaryCanDetectSteadyStateFlicker": get_path(
            frame_contract,
            ["summaryCanDetectSteadyStateFlicker"],
        ),
        "presentationPersistsAfterStartup": get_path(
            frame_contract,
            ["presentationPersistsAfterStartup"],
        ),
        "presentationPersistsAcrossSteadyStateRaf": get_path(
            frame_contract,
            ["presentationPersistsAcrossSteadyStateRaf"],
        ),
        "captureWaitedForSteadyStateRaf": get_path(
            frame_contract,
            ["captureWaitedForSteadyStateRaf"],
        ),
        "captureSteadyStateWaitTimedOut": get_path(
            frame_contract,
            ["captureSteadyStateWaitTimedOut"],
        ),
        "webgpuDeviceConsistencyReady": get_path(
            frame_contract,
            ["webgpuDeviceConsistencyReady"],
        ),
        "presentationDeviceMatchesCompositorDevice": get_path(
            frame_contract,
            ["presentationDeviceMatchesCompositorDevice"],
        ),
        "currentTextureViewFreshPerPresentation": get_path(
            frame_contract,
            ["currentTextureViewFreshPerPresentation"],
        ),
        "currentTextureViewReusedAcrossFrames": get_path(
            frame_contract,
            ["currentTextureViewReusedAcrossFrames"],
        ),
        "staleTextureViewReuseDetected": get_path(
            frame_contract,
            ["staleTextureViewReuseDetected"],
        ),
        "crossDeviceTextureViewUseDetected": get_path(
            frame_contract,
            ["crossDeviceTextureViewUseDetected"],
        ),
        "contextReconfiguredOnDeviceChange": get_path(
            frame_contract,
            ["contextReconfiguredOnDeviceChange"],
        ),
        "compositorOutputCacheInvalidatedOnDeviceChange": get_path(
            frame_contract,
            ["compositorOutputCacheInvalidatedOnDeviceChange"],
        ),
        "webgpuValidationErrorDetected": get_path(
            frame_contract,
            ["webgpuValidationErrorDetected"],
        ),
        "invalidCommandBufferDetected": get_path(
            frame_contract,
            ["invalidCommandBufferDetected"],
        ),
        "queueSubmitFailureDetected": get_path(
            frame_contract,
            ["queueSubmitFailureDetected"],
        ),
        "presentationErrorName": get_path(
            frame_contract,
            ["presentationErrorName"],
        ),
        "presentationErrorMessage": get_path(
            frame_contract,
            ["presentationErrorMessage"],
        ),
        "canvasOverwriteAfterCompositorPresentationDetected": get_path(
            frame_contract,
            ["canvasOverwriteAfterCompositorPresentationDetected"],
        ),
        "normalBackendOverwriteAfterCompositorPresentationDetected": get_path(
            frame_contract,
            ["normalBackendOverwriteAfterCompositorPresentationDetected"],
        ),
        "fallbackOverwriteAfterCompositorPresentationDetected": get_path(
            frame_contract,
            ["fallbackOverwriteAfterCompositorPresentationDetected"],
        ),
        "viewerLoopRuntimeFatalErrorDetected": get_path(
            frame_contract,
            ["viewerLoopRuntimeFatalErrorDetected"],
        ),
        "viewerLoopRuntimeFatalError": get_path(
            frame_contract,
            ["viewerLoopRuntimeFatalError"],
        ),
        "step85TileCompositorPathPreserved": step85_preserved,
        "step86BoundaryContractPreserved": step86_preserved,
        "step87DepthOrderingPreserved": step87_preserved,
        "webgpuWebgl2SameFramePresentationMixed": get_path(
            frame_contract,
            ["webgpuWebgl2SameFramePresentationMixed"],
        ),
        "fallbackMixingPrevented": get_path(
            frame_contract,
            ["fallbackMixingPrevented"],
        ),
        "fullCudaParityDeferred": get_path(
            frame_contract,
            ["fullCudaParityDeferred"],
        ),
        "finalProductionCompositorDeferred": get_path(
            frame_contract,
            ["finalProductionCompositorDeferred"],
        ),
        "fullRendererSuccessClaimed": get_path(
            frame_contract,
            ["fullRendererSuccessClaimed"],
        ),
        "sourceTileCompositorContractVersion": get_path(
            frame_contract,
            ["sourceTileCompositorContractVersion"],
        ),
        "sourceDepthOrderingContractVersion": get_path(
            frame_contract,
            ["sourceDepthOrderingContractVersion"],
        ),
        "sourceBoundaryContractVersion": get_path(
            frame_contract,
            ["sourceBoundaryContractVersion"],
        ),
        "reason": get_path(frame_contract, ["reason"]),
    }


def build_step89_real_tile_compositor_output_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    frame_contract = get_path(
        summary,
        ["webgpuTileCompositorFrameImplementation"],
        {},
    )
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    phase_step = get_path(summary, ["phaseStep"])
    steady_state_counts_clear = (
        numeric_value(get_path(frame_contract, ["steadyStateBlankFrameCount"]), 0) == 0
        and numeric_value(get_path(frame_contract, ["steadyStateNoOpFrameCount"]), 0) == 0
        and numeric_value(get_path(frame_contract, ["steadyStateClearFrameCount"]), 0) == 0
        and numeric_value(get_path(frame_contract, ["steadyStateUnknownFrameCount"]), 0) == 0
        and numeric_value(get_path(frame_contract, ["steadyStateNormalBackendFrameCount"]), 0) == 0
        and numeric_value(get_path(frame_contract, ["steadyStateWebgl2FallbackFrameCount"]), 0) == 0
    )
    step88_presentation_contract_preserved = (
        get_path(frame_contract, ["selectedFrameImplementation"])
        == "webgpu-tile-compositor-frame-implementation"
        and get_path(frame_contract, ["frameImplementationSelected"]) is True
        and get_path(frame_contract, ["frameImplementationExecuted"]) is True
        and get_path(frame_contract, ["frameImplementationReady"]) is True
        and get_path(frame_contract, ["steadyStateSamplingReady"]) is True
        and get_path(frame_contract, ["steadyStateTileCompositorOwnsFinalPresentation"]) is True
        and get_path(frame_contract, ["steadyStateFinalPresentSourceStable"]) is True
        and get_path(frame_contract, ["steadyStateFinalPresentSourceAlternates"]) is False
        and steady_state_counts_clear
        and get_path(frame_contract, ["presentationPersistsAfterStartup"]) is True
        and get_path(frame_contract, ["presentationPersistsAcrossSteadyStateRaf"]) is True
        and get_path(frame_contract, ["currentTextureUsesWebGpuTileCompositorOutput"]) is True
        and get_path(frame_contract, ["currentTextureViewFreshPerPresentation"]) is True
        and get_path(frame_contract, ["webgpuDeviceConsistencyReady"]) is True
        and get_path(frame_contract, ["webgpuValidationErrorDetected"]) is False
        and get_path(frame_contract, ["invalidCommandBufferDetected"]) is False
        and get_path(frame_contract, ["queueSubmitFailureDetected"]) is False
        and get_path(frame_contract, ["step85TileCompositorPathPreserved"]) is True
        and get_path(frame_contract, ["step86BoundaryContractPreserved"]) is True
        and get_path(frame_contract, ["step87DepthOrderingPreserved"]) is True
        and get_path(frame_contract, ["webgpuWebgl2SameFramePresentationMixed"]) is False
        and get_path(frame_contract, ["fallbackMixingPrevented"]) is True
    )
    real_output_ready = (
        get_path(compositor_contract, ["realTileCompositorOutputReady"]) is True
        and get_path(compositor_contract, ["debugOutputBypassedForCompositor"]) is True
        and get_path(compositor_contract, ["gaussianAttributePayloadConsumed"]) is True
        and get_path(compositor_contract, ["footprintPayloadConsumed"]) is True
        and get_path(compositor_contract, ["orderedTileReferencesConsumed"]) is True
        and get_path(compositor_contract, ["depthOrderedAccumulationUsed"]) is True
        and get_path(compositor_contract, ["alphaAccumulationUsed"]) is True
        and get_path(compositor_contract, ["colorAccumulationUsed"]) is True
        and numeric_value(get_path(compositor_contract, ["tileCompositorContributionCount"]), 0) > 0
        and numeric_value(get_path(compositor_contract, ["tileCompositorNonzeroOutputRatio"]), 0) > 0
        and get_path(compositor_contract, ["tileCompositorOutputChangedFromDebugPattern"]) is True
    )
    success = (
        phase_step == "phase3-step89"
        and step88_presentation_contract_preserved
        and real_output_ready
        and get_path(frame_contract, ["fullRendererSuccessClaimed"]) is False
    )
    blocked_reason = None
    if not success:
        if phase_step != "phase3-step89":
            blocked_reason = "summary-phase-step-is-not-phase3-step89"
        elif not step88_presentation_contract_preserved:
            blocked_reason = "step88-steady-state-presentation-contract-not-preserved"
        elif get_path(compositor_contract, ["realTileCompositorOutputReady"]) is not True:
            blocked_reason = "real-tile-compositor-output-not-ready"
        elif get_path(compositor_contract, ["debugOutputBypassedForCompositor"]) is not True:
            blocked_reason = "debug-output-not-bypassed-for-compositor"
        elif get_path(compositor_contract, ["gaussianAttributePayloadConsumed"]) is not True:
            blocked_reason = "gaussian-attribute-payload-not-consumed"
        elif get_path(compositor_contract, ["footprintPayloadConsumed"]) is not True:
            blocked_reason = "footprint-payload-not-consumed"
        elif get_path(compositor_contract, ["orderedTileReferencesConsumed"]) is not True:
            blocked_reason = "ordered-tile-references-not-consumed"
        elif get_path(compositor_contract, ["depthOrderedAccumulationUsed"]) is not True:
            blocked_reason = "depth-ordered-accumulation-not-used"
        elif get_path(compositor_contract, ["alphaAccumulationUsed"]) is not True:
            blocked_reason = "alpha-accumulation-not-used"
        elif get_path(compositor_contract, ["colorAccumulationUsed"]) is not True:
            blocked_reason = "color-accumulation-not-used"
        else:
            blocked_reason = "step89-real-compositor-output-validation-failed"
    return {
        "step89Decision": "success" if success else "blocked",
        "step89BlockedReason": blocked_reason,
        "step89SelectedGoal": "A+B+C-real-tile-compositor-output-and-payload-consumption",
        "phaseStep": phase_step,
        "step89SummaryApplies": phase_step == "phase3-step89",
        "selectedFrameImplementation": get_path(frame_contract, ["selectedFrameImplementation"]),
        "frameImplementationSelected": get_path(frame_contract, ["frameImplementationSelected"]),
        "frameImplementationExecuted": get_path(frame_contract, ["frameImplementationExecuted"]),
        "frameImplementationReady": get_path(frame_contract, ["frameImplementationReady"]),
        "realTileCompositorOutputReady": get_path(compositor_contract, ["realTileCompositorOutputReady"]),
        "debugOutputBypassedForCompositor": get_path(compositor_contract, ["debugOutputBypassedForCompositor"]),
        "gaussianAttributePayloadConsumed": get_path(compositor_contract, ["gaussianAttributePayloadConsumed"]),
        "footprintPayloadConsumed": get_path(compositor_contract, ["footprintPayloadConsumed"]),
        "orderedTileReferencesConsumed": get_path(compositor_contract, ["orderedTileReferencesConsumed"]),
        "depthOrderedAccumulationUsed": get_path(compositor_contract, ["depthOrderedAccumulationUsed"]),
        "alphaAccumulationUsed": get_path(compositor_contract, ["alphaAccumulationUsed"]),
        "colorAccumulationUsed": get_path(compositor_contract, ["colorAccumulationUsed"]),
        "tileCompositorContributionCount": get_path(compositor_contract, ["tileCompositorContributionCount"]),
        "tileCompositorNonzeroOutputRatio": get_path(compositor_contract, ["tileCompositorNonzeroOutputRatio"]),
        "tileCompositorOutputChangedFromDebugPattern": get_path(
            compositor_contract,
            ["tileCompositorOutputChangedFromDebugPattern"],
        ),
        "generatedCompositorFields": get_path(compositor_contract, ["generatedCompositorFields"], []),
        "deferredCompositorFields": get_path(compositor_contract, ["deferredCompositorFields"], []),
        "step88PresentationContractPreserved": step88_presentation_contract_preserved,
        "steadyStateSampledRafCount": get_path(frame_contract, ["steadyStateSampledRafCount"]),
        "steadyStateTileCompositorOwnsFinalPresentation": get_path(
            frame_contract,
            ["steadyStateTileCompositorOwnsFinalPresentation"],
        ),
        "steadyStateFinalPresentSourceStable": get_path(
            frame_contract,
            ["steadyStateFinalPresentSourceStable"],
        ),
        "steadyStateFinalPresentSourceAlternates": get_path(
            frame_contract,
            ["steadyStateFinalPresentSourceAlternates"],
        ),
        "steadyStateBlankFrameCount": get_path(frame_contract, ["steadyStateBlankFrameCount"]),
        "steadyStateNoOpFrameCount": get_path(frame_contract, ["steadyStateNoOpFrameCount"]),
        "steadyStateClearFrameCount": get_path(frame_contract, ["steadyStateClearFrameCount"]),
        "steadyStateUnknownFrameCount": get_path(frame_contract, ["steadyStateUnknownFrameCount"]),
        "steadyStateNormalBackendFrameCount": get_path(
            frame_contract,
            ["steadyStateNormalBackendFrameCount"],
        ),
        "steadyStateWebgl2FallbackFrameCount": get_path(
            frame_contract,
            ["steadyStateWebgl2FallbackFrameCount"],
        ),
        "currentTextureUsesWebGpuTileCompositorOutput": get_path(
            frame_contract,
            ["currentTextureUsesWebGpuTileCompositorOutput"],
        ),
        "currentTextureViewFreshPerPresentation": get_path(
            frame_contract,
            ["currentTextureViewFreshPerPresentation"],
        ),
        "webgpuDeviceConsistencyReady": get_path(frame_contract, ["webgpuDeviceConsistencyReady"]),
        "webgpuValidationErrorDetected": get_path(frame_contract, ["webgpuValidationErrorDetected"]),
        "invalidCommandBufferDetected": get_path(frame_contract, ["invalidCommandBufferDetected"]),
        "queueSubmitFailureDetected": get_path(frame_contract, ["queueSubmitFailureDetected"]),
        "step85TileCompositorPathPreserved": get_path(frame_contract, ["step85TileCompositorPathPreserved"]),
        "step86BoundaryContractPreserved": get_path(frame_contract, ["step86BoundaryContractPreserved"]),
        "step87DepthOrderingPreserved": get_path(frame_contract, ["step87DepthOrderingPreserved"]),
        "webgpuWebgl2SameFramePresentationMixed": get_path(
            frame_contract,
            ["webgpuWebgl2SameFramePresentationMixed"],
        ),
        "fallbackMixingPrevented": get_path(frame_contract, ["fallbackMixingPrevented"]),
        "fullCudaParityDeferred": get_path(frame_contract, ["fullCudaParityDeferred"]),
        "finalProductionCompositorDeferred": get_path(
            frame_contract,
            ["finalProductionCompositorDeferred"],
        ),
        "fullRendererSuccessClaimed": get_path(frame_contract, ["fullRendererSuccessClaimed"]),
    }


def build_step90_realtime_runtime_path_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    frame_contract = get_path(
        summary,
        ["webgpuTileCompositorFrameImplementation"],
        {},
    )
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    phase_step = get_path(summary, ["phaseStep"])
    steady_state_counts_clear = (
        numeric_value(get_path(frame_contract, ["steadyStateBlankFrameCount"]), 0) == 0
        and numeric_value(get_path(frame_contract, ["steadyStateNoOpFrameCount"]), 0) == 0
        and numeric_value(get_path(frame_contract, ["steadyStateClearFrameCount"]), 0) == 0
        and numeric_value(get_path(frame_contract, ["steadyStateUnknownFrameCount"]), 0) == 0
        and numeric_value(get_path(frame_contract, ["steadyStateNormalBackendFrameCount"]), 0) == 0
        and numeric_value(get_path(frame_contract, ["steadyStateWebgl2FallbackFrameCount"]), 0) == 0
    )
    step88_presentation_contract_preserved = (
        get_path(frame_contract, ["selectedFrameImplementation"])
        == "webgpu-tile-compositor-frame-implementation"
        and get_path(frame_contract, ["frameImplementationSelected"]) is True
        and get_path(frame_contract, ["frameImplementationExecuted"]) is True
        and get_path(frame_contract, ["frameImplementationReady"]) is True
        and get_path(frame_contract, ["steadyStateSamplingReady"]) is True
        and get_path(frame_contract, ["steadyStateTileCompositorOwnsFinalPresentation"]) is True
        and get_path(frame_contract, ["steadyStateFinalPresentSourceStable"]) is True
        and get_path(frame_contract, ["steadyStateFinalPresentSourceAlternates"]) is False
        and steady_state_counts_clear
        and get_path(frame_contract, ["currentTextureUsesWebGpuTileCompositorOutput"]) is True
        and get_path(frame_contract, ["currentTextureViewFreshPerPresentation"]) is True
        and get_path(frame_contract, ["webgpuDeviceConsistencyReady"]) is True
        and get_path(frame_contract, ["webgpuValidationErrorDetected"]) is False
        and get_path(frame_contract, ["invalidCommandBufferDetected"]) is False
        and get_path(frame_contract, ["queueSubmitFailureDetected"]) is False
        and get_path(frame_contract, ["step85TileCompositorPathPreserved"]) is True
        and get_path(frame_contract, ["step86BoundaryContractPreserved"]) is True
        and get_path(frame_contract, ["step87DepthOrderingPreserved"]) is True
        and get_path(frame_contract, ["webgpuWebgl2SameFramePresentationMixed"]) is False
        and get_path(frame_contract, ["fallbackMixingPrevented"]) is True
    )
    step89_output_preserved = (
        get_path(compositor_contract, ["step89RealCompositorOutputPreserved"]) is True
        and get_path(compositor_contract, ["realTileCompositorOutputReady"]) is True
        and get_path(compositor_contract, ["gaussianAttributePayloadConsumed"]) is True
        and get_path(compositor_contract, ["footprintPayloadConsumed"]) is True
        and get_path(compositor_contract, ["orderedTileReferencesConsumed"]) is True
        and get_path(compositor_contract, ["depthOrderedAccumulationUsed"]) is True
        and get_path(compositor_contract, ["alphaAccumulationUsed"]) is True
        and get_path(compositor_contract, ["colorAccumulationUsed"]) is True
        and numeric_value(get_path(compositor_contract, ["tileCompositorContributionCount"]), 0) > 0
        and numeric_value(get_path(compositor_contract, ["nonzeroOutputRatio"]), 0) > 0
    )
    realtime_runtime_ready = (
        get_path(compositor_contract, ["realTimeRuntimePathReady"]) is True
        and get_path(compositor_contract, ["readbackFreeSteadyStateCompositorUsed"]) is True
        and get_path(compositor_contract, ["runtimeCompositorDoesNotDependOnCaptureReadback"]) is True
        and get_path(compositor_contract, ["gpuOwnedRuntimeResourcesUsed"]) is True
        and get_path(compositor_contract, ["diagnosticReadbackSeparatedFromRuntimePath"]) is True
        and get_path(compositor_contract, ["debugPathSeparatedFromRuntimePath"]) is True
        and get_path(compositor_contract, ["runtimeOutputReadyWithoutTextureReadback"]) is True
        and get_path(compositor_contract, ["runtimeTelemetryReady"]) is True
        and get_path(compositor_contract, ["cpuGpuSyncDependencyReduced"]) is True
        and get_path(compositor_contract, ["realtimeReadinessImproved"]) is True
        and numeric_value(get_path(compositor_contract, ["compositorDispatchCount"]), 0) > 0
        and numeric_value(get_path(compositor_contract, ["compositorWorkItemCount"]), 0) > 0
        and numeric_value(get_path(compositor_contract, ["tileReferenceCount"]), 0) > 0
        and numeric_value(get_path(compositor_contract, ["orderedReferenceCount"]), 0) > 0
        and numeric_value(get_path(compositor_contract, ["accumulationContributionCount"]), 0) > 0
    )
    success = (
        phase_step == "phase3-step90"
        and step88_presentation_contract_preserved
        and step89_output_preserved
        and realtime_runtime_ready
        and get_path(frame_contract, ["fullRendererSuccessClaimed"]) is False
    )
    blocked_reason = None
    if not success:
        if phase_step != "phase3-step90":
            blocked_reason = "summary-phase-step-is-not-phase3-step90"
        elif not step88_presentation_contract_preserved:
            blocked_reason = "step88-steady-state-presentation-contract-not-preserved"
        elif not step89_output_preserved:
            blocked_reason = "step89-real-compositor-output-not-preserved"
        elif not realtime_runtime_ready:
            blocked_reason = "step90-realtime-runtime-path-not-ready"
        else:
            blocked_reason = "step90-runtime-validation-failed"
    return {
        "step90Decision": "success" if success else "blocked",
        "step90BlockedReason": blocked_reason,
        "step90SelectedGoal": "A+B-readback-free-gpu-owned-runtime-path-plus-telemetry",
        "phaseStep": phase_step,
        "step90SummaryApplies": phase_step == "phase3-step90",
        "selectedFrameImplementation": get_path(frame_contract, ["selectedFrameImplementation"]),
        "frameImplementationSelected": get_path(frame_contract, ["frameImplementationSelected"]),
        "frameImplementationExecuted": get_path(frame_contract, ["frameImplementationExecuted"]),
        "frameImplementationReady": get_path(frame_contract, ["frameImplementationReady"]),
        "realTimeRuntimePathReady": get_path(compositor_contract, ["realTimeRuntimePathReady"]),
        "readbackFreeSteadyStateCompositorUsed": get_path(
            compositor_contract,
            ["readbackFreeSteadyStateCompositorUsed"],
        ),
        "runtimeCompositorDoesNotDependOnCaptureReadback": get_path(
            compositor_contract,
            ["runtimeCompositorDoesNotDependOnCaptureReadback"],
        ),
        "gpuOwnedRuntimeResourcesUsed": get_path(compositor_contract, ["gpuOwnedRuntimeResourcesUsed"]),
        "diagnosticReadbackSeparatedFromRuntimePath": get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        ),
        "debugPathSeparatedFromRuntimePath": get_path(
            compositor_contract,
            ["debugPathSeparatedFromRuntimePath"],
        ),
        "runtimeOutputReadyWithoutTextureReadback": get_path(
            compositor_contract,
            ["runtimeOutputReadyWithoutTextureReadback"],
        ),
        "diagnosticTextureReadbackUsed": get_path(
            compositor_contract,
            ["diagnosticTextureReadbackUsed"],
        ),
        "diagnosticSummaryReadbackUsed": get_path(
            compositor_contract,
            ["diagnosticSummaryReadbackUsed"],
        ),
        "step89RealCompositorOutputPreserved": step89_output_preserved,
        "gaussianAttributePayloadConsumed": get_path(compositor_contract, ["gaussianAttributePayloadConsumed"]),
        "footprintPayloadConsumed": get_path(compositor_contract, ["footprintPayloadConsumed"]),
        "orderedTileReferencesConsumed": get_path(compositor_contract, ["orderedTileReferencesConsumed"]),
        "depthOrderedAccumulationUsed": get_path(compositor_contract, ["depthOrderedAccumulationUsed"]),
        "alphaAccumulationUsed": get_path(compositor_contract, ["alphaAccumulationUsed"]),
        "colorAccumulationUsed": get_path(compositor_contract, ["colorAccumulationUsed"]),
        "compositorDispatchCount": get_path(compositor_contract, ["compositorDispatchCount"]),
        "compositorWorkItemCount": get_path(compositor_contract, ["compositorWorkItemCount"]),
        "tileReferenceCount": get_path(compositor_contract, ["tileReferenceCount"]),
        "orderedReferenceCount": get_path(compositor_contract, ["orderedReferenceCount"]),
        "accumulationContributionCount": get_path(
            compositor_contract,
            ["accumulationContributionCount"],
        ),
        "nonzeroOutputRatio": get_path(compositor_contract, ["nonzeroOutputRatio"]),
        "runtimeTelemetryReady": get_path(compositor_contract, ["runtimeTelemetryReady"]),
        "cpuGpuSyncDependencyReduced": get_path(compositor_contract, ["cpuGpuSyncDependencyReduced"]),
        "realtimeReadinessImproved": get_path(compositor_contract, ["realtimeReadinessImproved"]),
        "step88PresentationContractPreserved": step88_presentation_contract_preserved,
        "steadyStateSampledRafCount": get_path(frame_contract, ["steadyStateSampledRafCount"]),
        "steadyStateTileCompositorOwnsFinalPresentation": get_path(
            frame_contract,
            ["steadyStateTileCompositorOwnsFinalPresentation"],
        ),
        "steadyStateBlankFrameCount": get_path(frame_contract, ["steadyStateBlankFrameCount"]),
        "steadyStateNoOpFrameCount": get_path(frame_contract, ["steadyStateNoOpFrameCount"]),
        "steadyStateClearFrameCount": get_path(frame_contract, ["steadyStateClearFrameCount"]),
        "steadyStateUnknownFrameCount": get_path(frame_contract, ["steadyStateUnknownFrameCount"]),
        "steadyStateNormalBackendFrameCount": get_path(
            frame_contract,
            ["steadyStateNormalBackendFrameCount"],
        ),
        "steadyStateWebgl2FallbackFrameCount": get_path(
            frame_contract,
            ["steadyStateWebgl2FallbackFrameCount"],
        ),
        "currentTextureUsesWebGpuTileCompositorOutput": get_path(
            frame_contract,
            ["currentTextureUsesWebGpuTileCompositorOutput"],
        ),
        "currentTextureViewFreshPerPresentation": get_path(
            frame_contract,
            ["currentTextureViewFreshPerPresentation"],
        ),
        "webgpuDeviceConsistencyReady": get_path(frame_contract, ["webgpuDeviceConsistencyReady"]),
        "webgpuValidationErrorDetected": get_path(frame_contract, ["webgpuValidationErrorDetected"]),
        "invalidCommandBufferDetected": get_path(frame_contract, ["invalidCommandBufferDetected"]),
        "queueSubmitFailureDetected": get_path(frame_contract, ["queueSubmitFailureDetected"]),
        "step85TileCompositorPathPreserved": get_path(frame_contract, ["step85TileCompositorPathPreserved"]),
        "step86BoundaryContractPreserved": get_path(frame_contract, ["step86BoundaryContractPreserved"]),
        "step87DepthOrderingPreserved": get_path(frame_contract, ["step87DepthOrderingPreserved"]),
        "webgpuWebgl2SameFramePresentationMixed": get_path(
            frame_contract,
            ["webgpuWebgl2SameFramePresentationMixed"],
        ),
        "fallbackMixingPrevented": get_path(frame_contract, ["fallbackMixingPrevented"]),
        "deferredProductionItems": get_path(compositor_contract, ["deferredProductionItems"], []),
        "fullCudaParityDeferred": get_path(frame_contract, ["fullCudaParityDeferred"]),
        "finalProductionCompositorDeferred": get_path(
            frame_contract,
            ["finalProductionCompositorDeferred"],
        ),
        "fullRendererSuccessClaimed": get_path(frame_contract, ["fullRendererSuccessClaimed"]),
    }


def build_step91_gpu_side_tile_ordering_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    frame_contract = get_path(
        summary,
        ["webgpuTileCompositorFrameImplementation"],
        {},
    )
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    phase_step = get_path(summary, ["phaseStep"])
    steady_state_counts_clear = (
        numeric_value(get_path(frame_contract, ["steadyStateBlankFrameCount"]), 0) == 0
        and numeric_value(get_path(frame_contract, ["steadyStateNoOpFrameCount"]), 0) == 0
        and numeric_value(get_path(frame_contract, ["steadyStateClearFrameCount"]), 0) == 0
        and numeric_value(get_path(frame_contract, ["steadyStateUnknownFrameCount"]), 0) == 0
        and numeric_value(get_path(frame_contract, ["steadyStateNormalBackendFrameCount"]), 0) == 0
        and numeric_value(get_path(frame_contract, ["steadyStateWebgl2FallbackFrameCount"]), 0) == 0
    )
    step88_presentation_contract_preserved = (
        get_path(frame_contract, ["selectedFrameImplementation"])
        == "webgpu-tile-compositor-frame-implementation"
        and get_path(frame_contract, ["frameImplementationSelected"]) is True
        and get_path(frame_contract, ["frameImplementationExecuted"]) is True
        and get_path(frame_contract, ["frameImplementationReady"]) is True
        and get_path(frame_contract, ["steadyStateSamplingReady"]) is True
        and get_path(frame_contract, ["steadyStateTileCompositorOwnsFinalPresentation"]) is True
        and get_path(frame_contract, ["steadyStateFinalPresentSourceStable"]) is True
        and get_path(frame_contract, ["steadyStateFinalPresentSourceAlternates"]) is False
        and steady_state_counts_clear
        and get_path(frame_contract, ["currentTextureUsesWebGpuTileCompositorOutput"]) is True
        and get_path(frame_contract, ["currentTextureViewFreshPerPresentation"]) is True
        and get_path(frame_contract, ["webgpuDeviceConsistencyReady"]) is True
        and get_path(frame_contract, ["webgpuValidationErrorDetected"]) is False
        and get_path(frame_contract, ["invalidCommandBufferDetected"]) is False
        and get_path(frame_contract, ["queueSubmitFailureDetected"]) is False
        and get_path(frame_contract, ["step85TileCompositorPathPreserved"]) is True
        and get_path(frame_contract, ["step86BoundaryContractPreserved"]) is True
        and get_path(frame_contract, ["step87DepthOrderingPreserved"]) is True
        and get_path(frame_contract, ["webgpuWebgl2SameFramePresentationMixed"]) is False
        and get_path(frame_contract, ["fallbackMixingPrevented"]) is True
    )
    step89_output_preserved = (
        get_path(compositor_contract, ["step89RealCompositorOutputPreserved"]) is True
        and get_path(compositor_contract, ["realTileCompositorOutputReady"]) is True
        and get_path(compositor_contract, ["gaussianAttributePayloadConsumed"]) is True
        and get_path(compositor_contract, ["footprintPayloadConsumed"]) is True
        and get_path(compositor_contract, ["orderedTileReferencesConsumed"]) is True
        and get_path(compositor_contract, ["depthOrderedAccumulationUsed"]) is True
        and get_path(compositor_contract, ["alphaAccumulationUsed"]) is True
        and get_path(compositor_contract, ["colorAccumulationUsed"]) is True
        and numeric_value(get_path(compositor_contract, ["tileCompositorContributionCount"]), 0) > 0
        and numeric_value(get_path(compositor_contract, ["nonzeroOutputRatio"]), 0) > 0
    )
    step90_runtime_preserved = (
        get_path(compositor_contract, ["step90RuntimePathPreserved"]) is True
        and get_path(compositor_contract, ["realTimeRuntimePathReady"]) is True
        and get_path(compositor_contract, ["readbackFreeSteadyStateCompositorUsed"]) is True
        and get_path(compositor_contract, ["runtimeCompositorDoesNotDependOnCaptureReadback"]) is True
        and get_path(compositor_contract, ["gpuOwnedRuntimeResourcesUsed"]) is True
        and get_path(compositor_contract, ["diagnosticReadbackSeparatedFromRuntimePath"]) is True
        and get_path(compositor_contract, ["runtimeTelemetryReady"]) is True
        and get_path(compositor_contract, ["cpuGpuSyncDependencyReduced"]) is True
        and get_path(compositor_contract, ["realtimeReadinessImproved"]) is True
    )
    gpu_side_ordering_ready = (
        get_path(compositor_contract, ["gpuSideTileOrderingReady"]) is True
        and get_path(compositor_contract, ["perTileOrderingRuntimePathUsed"]) is True
        and get_path(compositor_contract, ["orderedReferencesGeneratedOrUpdatedOnGpu"]) is True
        and get_path(compositor_contract, ["orderedReferencesConsumedByProductionAccumulation"]) is True
        and get_path(compositor_contract, ["productionAccumulationPathImproved"]) is True
        and get_path(compositor_contract, ["tileReferenceBufferLifecycleReady"]) is True
        and get_path(compositor_contract, ["orderedReferenceCountMatchesSource"]) is True
        and numeric_value(get_path(compositor_contract, ["tileReferenceCount"]), 0) > 0
        and numeric_value(get_path(compositor_contract, ["orderedReferenceCount"]), 0) > 0
        and numeric_value(get_path(compositor_contract, ["sortOrOrderingDispatchCount"]), 0) > 0
        and numeric_value(get_path(compositor_contract, ["orderingWorkItemCount"]), 0) > 0
        and numeric_value(get_path(compositor_contract, ["orderedReferenceBufferBytes"]), 0) > 0
        and numeric_value(get_path(compositor_contract, ["gpuOwnedOrderedReferenceRatio"]), 0) >= 1
    )
    success = (
        phase_step == "phase3-step91"
        and step88_presentation_contract_preserved
        and step89_output_preserved
        and step90_runtime_preserved
        and gpu_side_ordering_ready
        and get_path(frame_contract, ["fullRendererSuccessClaimed"]) is False
    )
    blocked_reason = None
    if not success:
        if phase_step != "phase3-step91":
            blocked_reason = "summary-phase-step-is-not-phase3-step91"
        elif not step88_presentation_contract_preserved:
            blocked_reason = "step88-steady-state-presentation-contract-not-preserved"
        elif not step89_output_preserved:
            blocked_reason = "step89-real-compositor-output-not-preserved"
        elif not step90_runtime_preserved:
            blocked_reason = "step90-realtime-runtime-path-not-preserved"
        elif not gpu_side_ordering_ready:
            blocked_reason = "step91-gpu-side-tile-ordering-not-ready"
        else:
            blocked_reason = "step91-runtime-validation-failed"
    return {
        "step91Decision": "success" if success else "blocked",
        "step91BlockedReason": blocked_reason,
        "step91SelectedGoal":
            "A+B+C-gpu-side-ordered-reference-buffer-production-accumulation",
        "phaseStep": phase_step,
        "step91SummaryApplies": phase_step == "phase3-step91",
        "selectedFrameImplementation": get_path(frame_contract, ["selectedFrameImplementation"]),
        "frameImplementationSelected": get_path(frame_contract, ["frameImplementationSelected"]),
        "frameImplementationExecuted": get_path(frame_contract, ["frameImplementationExecuted"]),
        "frameImplementationReady": get_path(frame_contract, ["frameImplementationReady"]),
        "gpuSideTileOrderingReady": get_path(compositor_contract, ["gpuSideTileOrderingReady"]),
        "perTileOrderingRuntimePathUsed": get_path(compositor_contract, ["perTileOrderingRuntimePathUsed"]),
        "orderedReferencesGeneratedOrUpdatedOnGpu": get_path(
            compositor_contract,
            ["orderedReferencesGeneratedOrUpdatedOnGpu"],
        ),
        "orderedReferencesConsumedByProductionAccumulation": get_path(
            compositor_contract,
            ["orderedReferencesConsumedByProductionAccumulation"],
        ),
        "productionAccumulationPathImproved": get_path(
            compositor_contract,
            ["productionAccumulationPathImproved"],
        ),
        "tileReferenceBufferLifecycleReady": get_path(
            compositor_contract,
            ["tileReferenceBufferLifecycleReady"],
        ),
        "tileReferenceCount": get_path(compositor_contract, ["tileReferenceCount"]),
        "orderedReferenceCount": get_path(compositor_contract, ["orderedReferenceCount"]),
        "orderedReferenceCountMatchesSource": get_path(
            compositor_contract,
            ["orderedReferenceCountMatchesSource"],
        ),
        "sortOrOrderingDispatchCount": get_path(compositor_contract, ["sortOrOrderingDispatchCount"]),
        "orderingWorkItemCount": get_path(compositor_contract, ["orderingWorkItemCount"]),
        "orderingScratchBufferBytes": get_path(compositor_contract, ["orderingScratchBufferBytes"]),
        "orderedReferenceBufferBytes": get_path(compositor_contract, ["orderedReferenceBufferBytes"]),
        "gpuOwnedOrderedReferenceRatio": get_path(
            compositor_contract,
            ["gpuOwnedOrderedReferenceRatio"],
        ),
        "step91ProductionAccumulationMode": get_path(
            compositor_contract,
            ["step91ProductionAccumulationMode"],
        ),
        "step90RuntimePathPreserved": step90_runtime_preserved,
        "realTimeRuntimePathReady": get_path(compositor_contract, ["realTimeRuntimePathReady"]),
        "readbackFreeSteadyStateCompositorUsed": get_path(
            compositor_contract,
            ["readbackFreeSteadyStateCompositorUsed"],
        ),
        "runtimeCompositorDoesNotDependOnCaptureReadback": get_path(
            compositor_contract,
            ["runtimeCompositorDoesNotDependOnCaptureReadback"],
        ),
        "gpuOwnedRuntimeResourcesUsed": get_path(compositor_contract, ["gpuOwnedRuntimeResourcesUsed"]),
        "diagnosticReadbackSeparatedFromRuntimePath": get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        ),
        "step89RealCompositorOutputPreserved": step89_output_preserved,
        "gaussianAttributePayloadConsumed": get_path(compositor_contract, ["gaussianAttributePayloadConsumed"]),
        "footprintPayloadConsumed": get_path(compositor_contract, ["footprintPayloadConsumed"]),
        "orderedTileReferencesConsumed": get_path(compositor_contract, ["orderedTileReferencesConsumed"]),
        "depthOrderedAccumulationUsed": get_path(compositor_contract, ["depthOrderedAccumulationUsed"]),
        "alphaAccumulationUsed": get_path(compositor_contract, ["alphaAccumulationUsed"]),
        "colorAccumulationUsed": get_path(compositor_contract, ["colorAccumulationUsed"]),
        "tileCompositorContributionCount": get_path(
            compositor_contract,
            ["tileCompositorContributionCount"],
        ),
        "nonzeroOutputRatio": get_path(compositor_contract, ["nonzeroOutputRatio"]),
        "step88PresentationContractPreserved": step88_presentation_contract_preserved,
        "steadyStateSampledRafCount": get_path(frame_contract, ["steadyStateSampledRafCount"]),
        "steadyStateTileCompositorOwnsFinalPresentation": get_path(
            frame_contract,
            ["steadyStateTileCompositorOwnsFinalPresentation"],
        ),
        "steadyStateBlankFrameCount": get_path(frame_contract, ["steadyStateBlankFrameCount"]),
        "steadyStateNoOpFrameCount": get_path(frame_contract, ["steadyStateNoOpFrameCount"]),
        "steadyStateClearFrameCount": get_path(frame_contract, ["steadyStateClearFrameCount"]),
        "steadyStateUnknownFrameCount": get_path(frame_contract, ["steadyStateUnknownFrameCount"]),
        "steadyStateNormalBackendFrameCount": get_path(
            frame_contract,
            ["steadyStateNormalBackendFrameCount"],
        ),
        "steadyStateWebgl2FallbackFrameCount": get_path(
            frame_contract,
            ["steadyStateWebgl2FallbackFrameCount"],
        ),
        "currentTextureUsesWebGpuTileCompositorOutput": get_path(
            frame_contract,
            ["currentTextureUsesWebGpuTileCompositorOutput"],
        ),
        "currentTextureViewFreshPerPresentation": get_path(
            frame_contract,
            ["currentTextureViewFreshPerPresentation"],
        ),
        "webgpuDeviceConsistencyReady": get_path(frame_contract, ["webgpuDeviceConsistencyReady"]),
        "webgpuValidationErrorDetected": get_path(frame_contract, ["webgpuValidationErrorDetected"]),
        "invalidCommandBufferDetected": get_path(frame_contract, ["invalidCommandBufferDetected"]),
        "queueSubmitFailureDetected": get_path(frame_contract, ["queueSubmitFailureDetected"]),
        "step85TileCompositorPathPreserved": get_path(frame_contract, ["step85TileCompositorPathPreserved"]),
        "step86BoundaryContractPreserved": get_path(frame_contract, ["step86BoundaryContractPreserved"]),
        "step87DepthOrderingPreserved": get_path(frame_contract, ["step87DepthOrderingPreserved"]),
        "webgpuWebgl2SameFramePresentationMixed": get_path(
            frame_contract,
            ["webgpuWebgl2SameFramePresentationMixed"],
        ),
        "fallbackMixingPrevented": get_path(frame_contract, ["fallbackMixingPrevented"]),
        "generatedCompositorFields": get_path(compositor_contract, ["generatedCompositorFields"], []),
        "deferredProductionItems": get_path(compositor_contract, ["deferredProductionItems"], []),
        "deferredCompositorFields": get_path(compositor_contract, ["deferredCompositorFields"], []),
        "fullCudaParityDeferred": get_path(frame_contract, ["fullCudaParityDeferred"]),
        "finalProductionCompositorDeferred": get_path(
            frame_contract,
            ["finalProductionCompositorDeferred"],
        ),
        "fullRendererSuccessClaimed": get_path(frame_contract, ["fullRendererSuccessClaimed"]),
    }


def build_step92_per_tile_depth_sort_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    frame_contract = get_path(
        summary,
        ["webgpuTileCompositorFrameImplementation"],
        {},
    )
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    phase_step = get_path(summary, ["phaseStep"])
    steady_state_counts_clear = (
        numeric_value(get_path(frame_contract, ["steadyStateBlankFrameCount"]), 0) == 0
        and numeric_value(get_path(frame_contract, ["steadyStateNoOpFrameCount"]), 0) == 0
        and numeric_value(get_path(frame_contract, ["steadyStateClearFrameCount"]), 0) == 0
        and numeric_value(get_path(frame_contract, ["steadyStateUnknownFrameCount"]), 0) == 0
        and numeric_value(get_path(frame_contract, ["steadyStateNormalBackendFrameCount"]), 0) == 0
        and numeric_value(get_path(frame_contract, ["steadyStateWebgl2FallbackFrameCount"]), 0) == 0
    )
    required_steady_state_raf_count = numeric_value(
        get_path(frame_contract, ["requiredSteadyStateRafCount"]),
        0,
    )
    steady_state_sampled_raf_count = numeric_value(
        get_path(frame_contract, ["steadyStateSampledRafCount"]),
        0,
    )
    steady_state_contract_ready = (
        get_path(frame_contract, ["steadyStateSamplingReady"]) is True
        and required_steady_state_raf_count >= 8
        and steady_state_sampled_raf_count >= required_steady_state_raf_count
        and get_path(frame_contract, ["steadyStateTileCompositorOwnsFinalPresentation"])
        is True
        and get_path(frame_contract, ["steadyStateFinalPresentSourceStable"]) is True
        and get_path(frame_contract, ["steadyStateFinalPresentSourceAlternates"])
        is False
        and steady_state_counts_clear
        and get_path(frame_contract, ["steadyStateVisualFlickerDetected"]) is False
        and get_path(frame_contract, ["summaryCanDetectStartupTransient"]) is True
        and get_path(frame_contract, ["summaryCanDetectSteadyStateFlicker"])
        is True
        and get_path(frame_contract, ["presentationPersistsAfterStartup"]) is True
        and get_path(
            frame_contract,
            ["presentationPersistsAcrossSteadyStateRaf"],
        )
        is True
        and get_path(frame_contract, ["captureSteadyStateWaitTimedOut"]) is False
    )
    heartbeat_contract_ready = (
        get_path(frame_contract, ["presentationHeartbeatReady"]) is True
        and get_path(frame_contract, ["presentationHeartbeatRunsEveryViewerRaf"])
        is True
        and get_path(frame_contract, ["presentationDecoupledFromCompositorUpdate"])
        is True
        and get_path(frame_contract, ["lastValidCompositorOutputCached"]) is True
        and get_path(
            frame_contract,
            ["lastValidCompositorOutputPresentedOnCleanFrames"],
        )
        is True
        and numeric_value(
            get_path(frame_contract, ["presentationHeartbeatFrameCount"]),
            0,
        )
        >= required_steady_state_raf_count
        and get_path(
            frame_contract,
            ["presentationHeartbeatFrameCountMatchesSampledRaf"],
        )
        is True
        and get_path(
            frame_contract,
            ["dirtySkippedCompositorUpdateButPresentedCachedOutput"],
        )
        is True
        and get_path(
            frame_contract,
            ["noBlankFrameBetweenHeartbeatPresentations"],
        )
        is True
    )
    final_present_trace_ready = (
        get_path(frame_contract, ["finalPresentSourceTracingReady"]) is True
        and get_path(frame_contract, ["rafTraceRingBufferReady"]) is True
        and get_path(frame_contract, ["rafTraceRecordedFromViewerLoopStart"]) is True
        and numeric_value(get_path(frame_contract, ["rafTraceRingBufferFrameCount"]), 0)
        >= steady_state_sampled_raf_count
        and get_path(frame_contract, ["summaryCanDetectObservedFlicker"]) is True
    )
    device_and_texture_ready = (
        get_path(frame_contract, ["currentTextureUsesWebGpuTileCompositorOutput"])
        is True
        and get_path(frame_contract, ["currentTextureViewFreshPerPresentation"])
        is True
        and get_path(
            frame_contract,
            ["currentTextureReadbackMatchesCompositorOutput"],
        )
        is True
        and get_path(frame_contract, ["webgpuDeviceConsistencyReady"]) is True
        and get_path(frame_contract, ["presentationDeviceMatchesCompositorDevice"])
        is True
        and get_path(frame_contract, ["currentTextureViewReusedAcrossFrames"])
        is False
        and get_path(frame_contract, ["staleTextureViewReuseDetected"]) is False
        and get_path(frame_contract, ["crossDeviceTextureViewUseDetected"]) is False
    )
    runtime_error_free = (
        get_path(frame_contract, ["webgpuValidationErrorDetected"]) is False
        and get_path(frame_contract, ["invalidCommandBufferDetected"]) is False
        and get_path(frame_contract, ["queueSubmitFailureDetected"]) is False
        and get_path(frame_contract, ["viewerLoopRuntimeFatalErrorDetected"]) is False
        and get_path(
            frame_contract,
            ["canvasOverwriteAfterCompositorPresentationDetected"],
        )
        is False
        and get_path(
            frame_contract,
            ["normalBackendOverwriteAfterCompositorPresentationDetected"],
        )
        is False
        and get_path(
            frame_contract,
            ["fallbackOverwriteAfterCompositorPresentationDetected"],
        )
        is False
    )
    step88_presentation_contract_preserved = (
        get_path(frame_contract, ["selectedFrameImplementation"])
        == "webgpu-tile-compositor-frame-implementation"
        and get_path(frame_contract, ["frameImplementationSelected"]) is True
        and get_path(frame_contract, ["frameImplementationExecuted"]) is True
        and steady_state_contract_ready
        and heartbeat_contract_ready
        and final_present_trace_ready
        and device_and_texture_ready
        and runtime_error_free
        and get_path(frame_contract, ["step85TileCompositorPathPreserved"]) is True
        and get_path(frame_contract, ["step86BoundaryContractPreserved"]) is True
        and get_path(frame_contract, ["step87DepthOrderingPreserved"]) is True
        and get_path(frame_contract, ["webgpuWebgl2SameFramePresentationMixed"]) is False
        and get_path(frame_contract, ["fallbackMixingPrevented"]) is True
    )
    step88_presentation_contract_failure_reason = None
    if not step88_presentation_contract_preserved:
        if (
            get_path(frame_contract, ["selectedFrameImplementation"])
            != "webgpu-tile-compositor-frame-implementation"
        ):
            step88_presentation_contract_failure_reason = (
                "tile-compositor-frame-implementation-not-selected"
            )
        elif get_path(frame_contract, ["frameImplementationSelected"]) is not True:
            step88_presentation_contract_failure_reason = (
                "frame-implementation-not-selected"
            )
        elif get_path(frame_contract, ["frameImplementationExecuted"]) is not True:
            step88_presentation_contract_failure_reason = (
                "frame-implementation-not-executed"
            )
        elif not steady_state_contract_ready:
            if get_path(frame_contract, ["steadyStateSamplingReady"]) is not True:
                step88_presentation_contract_failure_reason = (
                    "steady-state-sampling-not-ready"
                )
            elif required_steady_state_raf_count < 8:
                step88_presentation_contract_failure_reason = (
                    "required-steady-state-raf-count-too-low"
                )
            elif steady_state_sampled_raf_count < required_steady_state_raf_count:
                step88_presentation_contract_failure_reason = (
                    "steady-state-sampled-raf-count-too-low"
                )
            elif (
                get_path(
                    frame_contract,
                    ["steadyStateTileCompositorOwnsFinalPresentation"],
                )
                is not True
            ):
                step88_presentation_contract_failure_reason = (
                    "steady-state-final-presentation-not-owned-by-tile-compositor"
                )
            elif get_path(frame_contract, ["steadyStateFinalPresentSourceStable"]) is not True:
                step88_presentation_contract_failure_reason = (
                    "steady-state-final-present-source-not-stable"
                )
            elif get_path(frame_contract, ["steadyStateFinalPresentSourceAlternates"]) is True:
                step88_presentation_contract_failure_reason = (
                    "steady-state-final-present-source-alternates"
                )
            elif not steady_state_counts_clear:
                step88_presentation_contract_failure_reason = (
                    "steady-state-present-source-contamination-detected"
                )
            elif get_path(frame_contract, ["steadyStateVisualFlickerDetected"]) is True:
                step88_presentation_contract_failure_reason = (
                    "steady-state-visual-flicker-detected"
                )
            elif get_path(frame_contract, ["summaryCanDetectStartupTransient"]) is not True:
                step88_presentation_contract_failure_reason = (
                    "summary-cannot-detect-startup-transient"
                )
            elif get_path(frame_contract, ["summaryCanDetectSteadyStateFlicker"]) is not True:
                step88_presentation_contract_failure_reason = (
                    "summary-cannot-detect-steady-state-flicker"
                )
            elif get_path(frame_contract, ["presentationPersistsAfterStartup"]) is not True:
                step88_presentation_contract_failure_reason = (
                    "presentation-does-not-persist-after-startup"
                )
            elif (
                get_path(
                    frame_contract,
                    ["presentationPersistsAcrossSteadyStateRaf"],
                )
                is not True
            ):
                step88_presentation_contract_failure_reason = (
                    "presentation-does-not-persist-across-steady-state-raf"
                )
            else:
                step88_presentation_contract_failure_reason = (
                    "steady-state-presentation-contract-not-ready"
                )
        elif not heartbeat_contract_ready:
            step88_presentation_contract_failure_reason = (
                "presentation-heartbeat-contract-not-ready"
            )
        elif not final_present_trace_ready:
            step88_presentation_contract_failure_reason = (
                "final-present-source-trace-not-ready"
            )
        elif not device_and_texture_ready:
            step88_presentation_contract_failure_reason = (
                "currentTexture-or-device-lifecycle-not-ready"
            )
        elif not runtime_error_free:
            step88_presentation_contract_failure_reason = (
                "presentation-runtime-error-or-overwrite-detected"
            )
        elif get_path(frame_contract, ["step85TileCompositorPathPreserved"]) is not True:
            step88_presentation_contract_failure_reason = (
                "step85-tile-compositor-path-not-preserved"
            )
        elif get_path(frame_contract, ["step86BoundaryContractPreserved"]) is not True:
            step88_presentation_contract_failure_reason = (
                "step86-boundary-contract-not-preserved"
            )
        elif get_path(frame_contract, ["step87DepthOrderingPreserved"]) is not True:
            step88_presentation_contract_failure_reason = (
                "step87-depth-ordering-not-preserved"
            )
        elif get_path(frame_contract, ["webgpuWebgl2SameFramePresentationMixed"]) is not False:
            step88_presentation_contract_failure_reason = (
                "webgpu-webgl2-same-frame-presentation-mixed"
            )
        elif get_path(frame_contract, ["fallbackMixingPrevented"]) is not True:
            step88_presentation_contract_failure_reason = (
                "fallback-mixing-not-prevented"
            )
        else:
            step88_presentation_contract_failure_reason = (
                "step88-presentation-contract-not-preserved"
            )
    step89_output_preserved = (
        get_path(compositor_contract, ["step89RealCompositorOutputPreserved"]) is True
        and get_path(compositor_contract, ["realTileCompositorOutputReady"]) is True
        and get_path(compositor_contract, ["gaussianAttributePayloadConsumed"]) is True
        and get_path(compositor_contract, ["footprintPayloadConsumed"]) is True
        and get_path(compositor_contract, ["orderedTileReferencesConsumed"]) is True
        and get_path(compositor_contract, ["depthOrderedAccumulationUsed"]) is True
        and get_path(compositor_contract, ["alphaAccumulationUsed"]) is True
        and get_path(compositor_contract, ["colorAccumulationUsed"]) is True
        and numeric_value(get_path(compositor_contract, ["tileCompositorContributionCount"]), 0) > 0
        and numeric_value(get_path(compositor_contract, ["nonzeroOutputRatio"]), 0) > 0
    )
    step90_runtime_preserved = (
        get_path(compositor_contract, ["step90RuntimePathPreserved"]) is True
        and get_path(compositor_contract, ["realTimeRuntimePathReady"]) is True
        and get_path(compositor_contract, ["readbackFreeSteadyStateCompositorUsed"]) is True
        and get_path(compositor_contract, ["runtimeCompositorDoesNotDependOnCaptureReadback"]) is True
        and get_path(compositor_contract, ["gpuOwnedRuntimeResourcesUsed"]) is True
        and get_path(compositor_contract, ["diagnosticReadbackSeparatedFromRuntimePath"]) is True
        and get_path(compositor_contract, ["runtimeTelemetryReady"]) is True
        and get_path(compositor_contract, ["cpuGpuSyncDependencyReduced"]) is True
        and get_path(compositor_contract, ["realtimeReadinessImproved"]) is True
    )
    step91_ordered_reference_path_preserved = (
        get_path(compositor_contract, ["step91OrderedReferenceRuntimePathPreserved"]) is True
        and get_path(compositor_contract, ["gpuSideTileOrderingReady"]) is True
        and get_path(compositor_contract, ["perTileOrderingRuntimePathUsed"]) is True
        and get_path(compositor_contract, ["orderedReferencesGeneratedOrUpdatedOnGpu"]) is True
        and get_path(compositor_contract, ["orderedReferencesConsumedByProductionAccumulation"]) is True
        and get_path(compositor_contract, ["productionAccumulationPathImproved"]) is True
        and get_path(compositor_contract, ["tileReferenceBufferLifecycleReady"]) is True
        and get_path(compositor_contract, ["orderedReferenceCountMatchesSource"]) is True
        and numeric_value(get_path(compositor_contract, ["tileReferenceCount"]), 0) > 0
        and numeric_value(get_path(compositor_contract, ["orderedReferenceCount"]), 0) > 0
    )
    per_tile_sort_ready = (
        get_path(compositor_contract, ["gpuSidePerTileSortReady"]) is True
        and get_path(compositor_contract, ["boundedPerTileSortUsed"]) is True
        and get_path(compositor_contract, ["depthKeyBufferConsumed"]) is True
        and get_path(compositor_contract, ["depthSortedOrderedReferencesGenerated"]) is True
        and get_path(compositor_contract, ["depthSortedReferencesConsumedByAccumulation"]) is True
        and get_path(compositor_contract, ["sortedAccumulationPathUsed"]) is True
        and numeric_value(get_path(compositor_contract, ["sortDispatchCount"]), 0) > 0
        and numeric_value(get_path(compositor_contract, ["sortWorkItemCount"]), 0) > 0
        and numeric_value(get_path(compositor_contract, ["sortedTileCount"]), 0) > 0
        and numeric_value(get_path(compositor_contract, ["sortedReferenceCount"]), 0) > 0
        and numeric_value(get_path(compositor_contract, ["unsortedFallbackTileCount"]), 1) == 0
        and numeric_value(get_path(compositor_contract, ["maxReferencesPerTile"]), 0) > 0
        and numeric_value(get_path(compositor_contract, ["avgReferencesPerTile"]), 0) > 0
        and numeric_value(get_path(compositor_contract, ["sortOrOrderingBufferBytes"]), 0) > 0
    )
    success = (
        phase_step == "phase3-step92"
        and step88_presentation_contract_preserved
        and step89_output_preserved
        and step90_runtime_preserved
        and step91_ordered_reference_path_preserved
        and per_tile_sort_ready
        and get_path(frame_contract, ["fullRendererSuccessClaimed"]) is False
    )
    blocked_reason = None
    if not success:
        if phase_step != "phase3-step92":
            blocked_reason = "summary-phase-step-is-not-phase3-step92"
        elif not step88_presentation_contract_preserved:
            blocked_reason = "step88-steady-state-presentation-contract-not-preserved"
        elif not step89_output_preserved:
            blocked_reason = "step89-real-compositor-output-not-preserved"
        elif not step90_runtime_preserved:
            blocked_reason = "step90-realtime-runtime-path-not-preserved"
        elif not step91_ordered_reference_path_preserved:
            blocked_reason = "step91-ordered-reference-runtime-path-not-preserved"
        elif not per_tile_sort_ready:
            blocked_reason = "step92-gpu-side-per-tile-depth-sort-not-ready"
        else:
            blocked_reason = "step92-runtime-validation-failed"
    return {
        "step92Decision": "success" if success else "blocked",
        "step92BlockedReason": blocked_reason,
        "step92SelectedGoal": "A+B+C+D-bounded-gpu-per-tile-depth-sort",
        "phaseStep": phase_step,
        "step92SummaryApplies": phase_step == "phase3-step92",
        "selectedFrameImplementation": get_path(frame_contract, ["selectedFrameImplementation"]),
        "frameImplementationSelected": get_path(frame_contract, ["frameImplementationSelected"]),
        "frameImplementationExecuted": get_path(frame_contract, ["frameImplementationExecuted"]),
        "frameImplementationReady": get_path(frame_contract, ["frameImplementationReady"]),
        "gpuSidePerTileSortReady": get_path(compositor_contract, ["gpuSidePerTileSortReady"]),
        "boundedPerTileSortUsed": get_path(compositor_contract, ["boundedPerTileSortUsed"]),
        "depthKeyBufferConsumed": get_path(compositor_contract, ["depthKeyBufferConsumed"]),
        "depthSortedOrderedReferencesGenerated": get_path(
            compositor_contract,
            ["depthSortedOrderedReferencesGenerated"],
        ),
        "depthSortedReferencesConsumedByAccumulation": get_path(
            compositor_contract,
            ["depthSortedReferencesConsumedByAccumulation"],
        ),
        "sortedAccumulationPathUsed": get_path(compositor_contract, ["sortedAccumulationPathUsed"]),
        "sortDispatchCount": get_path(compositor_contract, ["sortDispatchCount"]),
        "sortWorkItemCount": get_path(compositor_contract, ["sortWorkItemCount"]),
        "sortedTileCount": get_path(compositor_contract, ["sortedTileCount"]),
        "sortedReferenceCount": get_path(compositor_contract, ["sortedReferenceCount"]),
        "unsortedFallbackTileCount": get_path(compositor_contract, ["unsortedFallbackTileCount"]),
        "maxReferencesPerTile": get_path(compositor_contract, ["maxReferencesPerTile"]),
        "avgReferencesPerTile": get_path(compositor_contract, ["avgReferencesPerTile"]),
        "orderedReferenceCountMatchesSource": get_path(
            compositor_contract,
            ["orderedReferenceCountMatchesSource"],
        ),
        "sortOrOrderingBufferBytes": get_path(compositor_contract, ["sortOrOrderingBufferBytes"]),
        "tileReferenceCount": get_path(compositor_contract, ["tileReferenceCount"]),
        "orderedReferenceCount": get_path(compositor_contract, ["orderedReferenceCount"]),
        "accumulationContributionCount": get_path(
            compositor_contract,
            ["accumulationContributionCount"],
        ),
        "nonzeroOutputRatio": get_path(compositor_contract, ["nonzeroOutputRatio"]),
        "step91OrderedReferenceRuntimePathPreserved": step91_ordered_reference_path_preserved,
        "gpuSideTileOrderingReady": get_path(compositor_contract, ["gpuSideTileOrderingReady"]),
        "orderedReferencesGeneratedOrUpdatedOnGpu": get_path(
            compositor_contract,
            ["orderedReferencesGeneratedOrUpdatedOnGpu"],
        ),
        "orderedReferencesConsumedByProductionAccumulation": get_path(
            compositor_contract,
            ["orderedReferencesConsumedByProductionAccumulation"],
        ),
        "step90RuntimePathPreserved": step90_runtime_preserved,
        "step89RealCompositorOutputPreserved": step89_output_preserved,
        "step88PresentationContractPreserved": step88_presentation_contract_preserved,
        "step88PresentationContractFailureReason": (
            step88_presentation_contract_failure_reason or "none"
        ),
        "step88PresentationContractPreservationMode": (
            "steady-state-final-present-source-contract"
        ),
        "steadyStateSamplingReady": get_path(frame_contract, ["steadyStateSamplingReady"]),
        "requiredSteadyStateRafCount": get_path(
            frame_contract,
            ["requiredSteadyStateRafCount"],
        ),
        "steadyStateSampledRafCount": get_path(frame_contract, ["steadyStateSampledRafCount"]),
        "steadyStateTileCompositorOwnsFinalPresentation": get_path(
            frame_contract,
            ["steadyStateTileCompositorOwnsFinalPresentation"],
        ),
        "steadyStateFinalPresentSourceStable": get_path(
            frame_contract,
            ["steadyStateFinalPresentSourceStable"],
        ),
        "steadyStateFinalPresentSourceAlternates": get_path(
            frame_contract,
            ["steadyStateFinalPresentSourceAlternates"],
        ),
        "steadyStateBlankFrameCount": get_path(frame_contract, ["steadyStateBlankFrameCount"]),
        "steadyStateNoOpFrameCount": get_path(frame_contract, ["steadyStateNoOpFrameCount"]),
        "steadyStateClearFrameCount": get_path(frame_contract, ["steadyStateClearFrameCount"]),
        "steadyStateUnknownFrameCount": get_path(frame_contract, ["steadyStateUnknownFrameCount"]),
        "steadyStateNormalBackendFrameCount": get_path(
            frame_contract,
            ["steadyStateNormalBackendFrameCount"],
        ),
        "steadyStateWebgl2FallbackFrameCount": get_path(
            frame_contract,
            ["steadyStateWebgl2FallbackFrameCount"],
        ),
        "steadyStateVisualFlickerDetected": get_path(
            frame_contract,
            ["steadyStateVisualFlickerDetected"],
        ),
        "summaryCanDetectStartupTransient": get_path(
            frame_contract,
            ["summaryCanDetectStartupTransient"],
        ),
        "summaryCanDetectSteadyStateFlicker": get_path(
            frame_contract,
            ["summaryCanDetectSteadyStateFlicker"],
        ),
        "presentationPersistsAfterStartup": get_path(
            frame_contract,
            ["presentationPersistsAfterStartup"],
        ),
        "presentationPersistsAcrossSteadyStateRaf": get_path(
            frame_contract,
            ["presentationPersistsAcrossSteadyStateRaf"],
        ),
        "presentationHeartbeatReady": get_path(
            frame_contract,
            ["presentationHeartbeatReady"],
        ),
        "presentationHeartbeatRunsEveryViewerRaf": get_path(
            frame_contract,
            ["presentationHeartbeatRunsEveryViewerRaf"],
        ),
        "presentationDecoupledFromCompositorUpdate": get_path(
            frame_contract,
            ["presentationDecoupledFromCompositorUpdate"],
        ),
        "lastValidCompositorOutputPresentedOnCleanFrames": get_path(
            frame_contract,
            ["lastValidCompositorOutputPresentedOnCleanFrames"],
        ),
        "finalPresentSourceTracingReady": get_path(
            frame_contract,
            ["finalPresentSourceTracingReady"],
        ),
        "rafTraceRingBufferReady": get_path(
            frame_contract,
            ["rafTraceRingBufferReady"],
        ),
        "gaussianAttributePayloadConsumed": get_path(compositor_contract, ["gaussianAttributePayloadConsumed"]),
        "footprintPayloadConsumed": get_path(compositor_contract, ["footprintPayloadConsumed"]),
        "orderedTileReferencesConsumed": get_path(compositor_contract, ["orderedTileReferencesConsumed"]),
        "depthOrderedAccumulationUsed": get_path(compositor_contract, ["depthOrderedAccumulationUsed"]),
        "alphaAccumulationUsed": get_path(compositor_contract, ["alphaAccumulationUsed"]),
        "colorAccumulationUsed": get_path(compositor_contract, ["colorAccumulationUsed"]),
        "currentTextureUsesWebGpuTileCompositorOutput": get_path(
            frame_contract,
            ["currentTextureUsesWebGpuTileCompositorOutput"],
        ),
        "currentTextureViewFreshPerPresentation": get_path(
            frame_contract,
            ["currentTextureViewFreshPerPresentation"],
        ),
        "currentTextureReadbackMatchesCompositorOutput": get_path(
            frame_contract,
            ["currentTextureReadbackMatchesCompositorOutput"],
        ),
        "webgpuDeviceConsistencyReady": get_path(frame_contract, ["webgpuDeviceConsistencyReady"]),
        "presentationDeviceMatchesCompositorDevice": get_path(
            frame_contract,
            ["presentationDeviceMatchesCompositorDevice"],
        ),
        "currentTextureViewReusedAcrossFrames": get_path(
            frame_contract,
            ["currentTextureViewReusedAcrossFrames"],
        ),
        "staleTextureViewReuseDetected": get_path(
            frame_contract,
            ["staleTextureViewReuseDetected"],
        ),
        "crossDeviceTextureViewUseDetected": get_path(
            frame_contract,
            ["crossDeviceTextureViewUseDetected"],
        ),
        "webgpuValidationErrorDetected": get_path(frame_contract, ["webgpuValidationErrorDetected"]),
        "invalidCommandBufferDetected": get_path(frame_contract, ["invalidCommandBufferDetected"]),
        "queueSubmitFailureDetected": get_path(frame_contract, ["queueSubmitFailureDetected"]),
        "viewerLoopRuntimeFatalErrorDetected": get_path(
            frame_contract,
            ["viewerLoopRuntimeFatalErrorDetected"],
        ),
        "canvasOverwriteAfterCompositorPresentationDetected": get_path(
            frame_contract,
            ["canvasOverwriteAfterCompositorPresentationDetected"],
        ),
        "normalBackendOverwriteAfterCompositorPresentationDetected": get_path(
            frame_contract,
            ["normalBackendOverwriteAfterCompositorPresentationDetected"],
        ),
        "fallbackOverwriteAfterCompositorPresentationDetected": get_path(
            frame_contract,
            ["fallbackOverwriteAfterCompositorPresentationDetected"],
        ),
        "step85TileCompositorPathPreserved": get_path(frame_contract, ["step85TileCompositorPathPreserved"]),
        "step86BoundaryContractPreserved": get_path(frame_contract, ["step86BoundaryContractPreserved"]),
        "step87DepthOrderingPreserved": get_path(frame_contract, ["step87DepthOrderingPreserved"]),
        "webgpuWebgl2SameFramePresentationMixed": get_path(
            frame_contract,
            ["webgpuWebgl2SameFramePresentationMixed"],
        ),
        "fallbackMixingPrevented": get_path(frame_contract, ["fallbackMixingPrevented"]),
        "generatedCompositorFields": get_path(compositor_contract, ["generatedCompositorFields"], []),
        "deferredProductionItems": get_path(compositor_contract, ["deferredProductionItems"], []),
        "deferredCompositorFields": get_path(compositor_contract, ["deferredCompositorFields"], []),
        "fullCudaParityDeferred": get_path(frame_contract, ["fullCudaParityDeferred"]),
        "finalProductionCompositorDeferred": get_path(
            frame_contract,
            ["finalProductionCompositorDeferred"],
        ),
        "fullRendererSuccessClaimed": get_path(frame_contract, ["fullRendererSuccessClaimed"]),
    }


def build_step93_overflow_aware_tile_ordering_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    frame_contract = get_path(
        summary,
        ["webgpuTileCompositorFrameImplementation"],
        {},
    )
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    phase_step = get_path(summary, ["phaseStep"])
    step92_summary = build_step92_per_tile_depth_sort_summary(summary)
    step88_presentation_contract_preserved = (
        get_path(step92_summary, ["step88PresentationContractPreserved"]) is True
    )
    step89_output_preserved = (
        get_path(step92_summary, ["step89RealCompositorOutputPreserved"]) is True
    )
    step90_runtime_preserved = (
        get_path(step92_summary, ["step90RuntimePathPreserved"]) is True
    )
    step91_ordered_reference_path_preserved = (
        get_path(step92_summary, ["step91OrderedReferenceRuntimePathPreserved"])
        is True
    )
    step92_bounded_sort_path_preserved = (
        get_path(compositor_contract, ["gpuSidePerTileSortReady"]) is True
        and get_path(compositor_contract, ["boundedPerTileSortUsed"]) is True
        and get_path(compositor_contract, ["depthKeyBufferConsumed"]) is True
        and get_path(
            compositor_contract,
            ["depthSortedOrderedReferencesGenerated"],
        )
        is True
        and get_path(
            compositor_contract,
            ["depthSortedReferencesConsumedByAccumulation"],
        )
        is True
        and get_path(compositor_contract, ["sortedAccumulationPathUsed"]) is True
        and numeric_value(get_path(compositor_contract, ["sortDispatchCount"]), 0)
        > 0
        and numeric_value(get_path(compositor_contract, ["sortWorkItemCount"]), 0)
        > 0
        and numeric_value(get_path(compositor_contract, ["sortedTileCount"]), 0)
        > 0
        and numeric_value(
            get_path(compositor_contract, ["sortedReferenceCount"]),
            0,
        )
        > 0
        and (
            get_path(compositor_contract, ["orderedReferenceCountMatchesSource"])
            is True
            or get_path(
                compositor_contract,
                ["sortedReferenceCountMatchesSourceOrCapacityPolicy"],
            )
            is True
        )
        and get_path(
            compositor_contract,
            ["sortedAccumulationCapacityPolicyUsed"],
        )
        is True
    )
    overflow_aware_ready = (
        get_path(compositor_contract, ["overflowAwareOrderingReady"]) is True
        and numeric_value(get_path(compositor_contract, ["sortCapacityLimit"]), 0)
        > 0
        and numeric_value(get_path(compositor_contract, ["overflowTileCount"]), -1)
        >= 0
        and numeric_value(
            get_path(compositor_contract, ["overflowReferenceCount"]),
            -1,
        )
        >= 0
        and numeric_value(
            get_path(compositor_contract, ["droppedReferenceCount"]),
            -1,
        )
        >= 0
        and get_path(
            compositor_contract,
            ["sortedReferenceCountMatchesSourceOrCapacityPolicy"],
        )
        is True
        and get_path(compositor_contract, ["sortedAccumulationCapacityPolicyUsed"])
        is True
    )
    scalable_sort_ready = (
        get_path(compositor_contract, ["scalableSortPreparationReady"]) is True
        and get_path(compositor_contract, ["sortScratchBufferReady"]) is True
        and get_path(
            compositor_contract,
            ["tileHistogramOrCapacityTableReady"],
        )
        is True
        and numeric_value(
            get_path(compositor_contract, ["sortOrOrderingBufferBytes"]),
            0,
        )
        > 0
    )
    lifecycle_ready = (
        get_path(
            compositor_contract,
            ["productionOrderedReferenceLifecycleReady"],
        )
        is True
        and get_path(compositor_contract, ["tileReferenceBufferLifecycleReady"])
        is True
        and get_path(compositor_contract, ["gpuOwnedRuntimeResourcesUsed"]) is True
    )
    error_free = (
        get_path(frame_contract, ["webgpuValidationErrorDetected"]) is False
        and get_path(frame_contract, ["invalidCommandBufferDetected"]) is False
        and get_path(frame_contract, ["queueSubmitFailureDetected"]) is False
        and get_path(frame_contract, ["webgpuWebgl2SameFramePresentationMixed"])
        is False
        and get_path(frame_contract, ["fallbackMixingPrevented"]) is True
        and get_path(frame_contract, ["fullRendererSuccessClaimed"]) is False
    )
    success = (
        phase_step == "phase3-step93"
        and overflow_aware_ready
        and scalable_sort_ready
        and lifecycle_ready
        and step92_bounded_sort_path_preserved
        and step91_ordered_reference_path_preserved
        and step90_runtime_preserved
        and step89_output_preserved
        and step88_presentation_contract_preserved
        and error_free
    )
    blocked_reason = None
    if not success:
        if phase_step != "phase3-step93":
            blocked_reason = "summary-phase-step-is-not-phase3-step93"
        elif not overflow_aware_ready:
            blocked_reason = "step93-overflow-aware-ordering-not-ready"
        elif not scalable_sort_ready:
            blocked_reason = "step93-scalable-sort-preparation-not-ready"
        elif not lifecycle_ready:
            blocked_reason = "step93-production-ordered-reference-lifecycle-not-ready"
        elif not step92_bounded_sort_path_preserved:
            blocked_reason = "step92-bounded-sort-path-not-preserved"
        elif not step91_ordered_reference_path_preserved:
            blocked_reason = "step91-ordered-reference-runtime-path-not-preserved"
        elif not step90_runtime_preserved:
            blocked_reason = "step90-realtime-runtime-path-not-preserved"
        elif not step89_output_preserved:
            blocked_reason = "step89-real-compositor-output-not-preserved"
        elif not step88_presentation_contract_preserved:
            blocked_reason = "step88-presentation-contract-not-preserved"
        else:
            blocked_reason = "step93-runtime-validation-failed"
    return {
        "step93Decision": "success" if success else "blocked",
        "step93BlockedReason": blocked_reason,
        "step93SelectedGoal":
            "A+B+C+D+E-overflow-aware-scalable-tile-ordering",
        "phaseStep": phase_step,
        "step93SummaryApplies": phase_step == "phase3-step93",
        "overflowAwareOrderingReady": get_path(
            compositor_contract,
            ["overflowAwareOrderingReady"],
        ),
        "sortCapacityLimit": get_path(compositor_contract, ["sortCapacityLimit"]),
        "overflowTileCount": get_path(compositor_contract, ["overflowTileCount"]),
        "overflowReferenceCount": get_path(
            compositor_contract,
            ["overflowReferenceCount"],
        ),
        "droppedReferenceCount": get_path(
            compositor_contract,
            ["droppedReferenceCount"],
        ),
        "overflowHandlingPolicy": get_path(
            compositor_contract,
            ["overflowHandlingPolicy"],
        ),
        "sortedReferenceCount": get_path(
            compositor_contract,
            ["sortedReferenceCount"],
        ),
        "sortedReferenceCountMatchesSourceOrCapacityPolicy": get_path(
            compositor_contract,
            ["sortedReferenceCountMatchesSourceOrCapacityPolicy"],
        ),
        "capacityUtilizationMax": get_path(
            compositor_contract,
            ["capacityUtilizationMax"],
        ),
        "capacityUtilizationAvg": get_path(
            compositor_contract,
            ["capacityUtilizationAvg"],
        ),
        "scalableSortPreparationReady": get_path(
            compositor_contract,
            ["scalableSortPreparationReady"],
        ),
        "sortScratchBufferReady": get_path(
            compositor_contract,
            ["sortScratchBufferReady"],
        ),
        "tileHistogramOrCapacityTableReady": get_path(
            compositor_contract,
            ["tileHistogramOrCapacityTableReady"],
        ),
        "productionOrderedReferenceLifecycleReady": get_path(
            compositor_contract,
            ["productionOrderedReferenceLifecycleReady"],
        ),
        "sortedAccumulationCapacityPolicyUsed": get_path(
            compositor_contract,
            ["sortedAccumulationCapacityPolicyUsed"],
        ),
        "sortDispatchCount": get_path(compositor_contract, ["sortDispatchCount"]),
        "sortWorkItemCount": get_path(compositor_contract, ["sortWorkItemCount"]),
        "sortOrOrderingBufferBytes": get_path(
            compositor_contract,
            ["sortOrOrderingBufferBytes"],
        ),
        "tileReferenceCount": get_path(compositor_contract, ["tileReferenceCount"]),
        "orderedReferenceCount": get_path(
            compositor_contract,
            ["orderedReferenceCount"],
        ),
        "orderedReferenceCountMatchesSource": get_path(
            compositor_contract,
            ["orderedReferenceCountMatchesSource"],
        ),
        "depthSortedReferencesConsumedByAccumulation": get_path(
            compositor_contract,
            ["depthSortedReferencesConsumedByAccumulation"],
        ),
        "sortedAccumulationPathUsed": get_path(
            compositor_contract,
            ["sortedAccumulationPathUsed"],
        ),
        "step92BoundedSortPathPreserved": step92_bounded_sort_path_preserved,
        "step91OrderedReferenceRuntimePathPreserved":
            step91_ordered_reference_path_preserved,
        "step90RuntimePathPreserved": step90_runtime_preserved,
        "step89RealCompositorOutputPreserved": step89_output_preserved,
        "step88PresentationContractPreserved":
            step88_presentation_contract_preserved,
        "steadyStateSampledRafCount": get_path(
            frame_contract,
            ["steadyStateSampledRafCount"],
        ),
        "steadyStateTileCompositorOwnsFinalPresentation": get_path(
            frame_contract,
            ["steadyStateTileCompositorOwnsFinalPresentation"],
        ),
        "steadyStateBlankFrameCount": get_path(
            frame_contract,
            ["steadyStateBlankFrameCount"],
        ),
        "steadyStateNoOpFrameCount": get_path(
            frame_contract,
            ["steadyStateNoOpFrameCount"],
        ),
        "steadyStateClearFrameCount": get_path(
            frame_contract,
            ["steadyStateClearFrameCount"],
        ),
        "steadyStateUnknownFrameCount": get_path(
            frame_contract,
            ["steadyStateUnknownFrameCount"],
        ),
        "steadyStateNormalBackendFrameCount": get_path(
            frame_contract,
            ["steadyStateNormalBackendFrameCount"],
        ),
        "steadyStateWebgl2FallbackFrameCount": get_path(
            frame_contract,
            ["steadyStateWebgl2FallbackFrameCount"],
        ),
        "currentTextureUsesWebGpuTileCompositorOutput": get_path(
            frame_contract,
            ["currentTextureUsesWebGpuTileCompositorOutput"],
        ),
        "currentTextureViewFreshPerPresentation": get_path(
            frame_contract,
            ["currentTextureViewFreshPerPresentation"],
        ),
        "webgpuDeviceConsistencyReady": get_path(
            frame_contract,
            ["webgpuDeviceConsistencyReady"],
        ),
        "webgpuValidationErrorDetected": get_path(
            frame_contract,
            ["webgpuValidationErrorDetected"],
        ),
        "invalidCommandBufferDetected": get_path(
            frame_contract,
            ["invalidCommandBufferDetected"],
        ),
        "queueSubmitFailureDetected": get_path(
            frame_contract,
            ["queueSubmitFailureDetected"],
        ),
        "webgpuWebgl2SameFramePresentationMixed": get_path(
            frame_contract,
            ["webgpuWebgl2SameFramePresentationMixed"],
        ),
        "fallbackMixingPrevented": get_path(
            frame_contract,
            ["fallbackMixingPrevented"],
        ),
        "generatedCompositorFields": get_path(
            compositor_contract,
            ["generatedCompositorFields"],
            [],
        ),
        "deferredProductionItems": get_path(
            compositor_contract,
            ["deferredProductionItems"],
            [],
        ),
        "deferredCompositorFields": get_path(
            compositor_contract,
            ["deferredCompositorFields"],
            [],
        ),
        "fullCudaParityDeferred": get_path(
            frame_contract,
            ["fullCudaParityDeferred"],
        ),
        "finalProductionCompositorDeferred": get_path(
            frame_contract,
            ["finalProductionCompositorDeferred"],
        ),
        "fullRendererSuccessClaimed": get_path(
            frame_contract,
            ["fullRendererSuccessClaimed"],
        ),
    }


def build_step94_parallel_per_tile_sort_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    frame_contract = get_path(
        summary,
        ["webgpuTileCompositorFrameImplementation"],
        {},
    )
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    phase_step = get_path(summary, ["phaseStep"])
    step92_summary = build_step92_per_tile_depth_sort_summary(summary)
    step88_presentation_contract_preserved = (
        get_path(step92_summary, ["step88PresentationContractPreserved"]) is True
    )
    step89_output_preserved = (
        get_path(step92_summary, ["step89RealCompositorOutputPreserved"]) is True
    )
    step90_runtime_preserved = (
        get_path(step92_summary, ["step90RuntimePathPreserved"]) is True
    )
    step91_ordered_reference_path_preserved = (
        get_path(step92_summary, ["step91OrderedReferenceRuntimePathPreserved"])
        is True
    )
    step92_bounded_sort_path_preserved = (
        get_path(compositor_contract, ["gpuSidePerTileSortReady"]) is True
        and get_path(compositor_contract, ["boundedPerTileSortUsed"]) is True
        and get_path(compositor_contract, ["depthKeyBufferConsumed"]) is True
        and get_path(
            compositor_contract,
            ["depthSortedReferencesConsumedByAccumulation"],
        )
        is True
        and get_path(compositor_contract, ["sortedAccumulationPathUsed"]) is True
        and numeric_value(get_path(compositor_contract, ["sortedReferenceCount"]), 0)
        > 0
    )
    step93_overflow_policy_preserved = (
        get_path(compositor_contract, ["step93OverflowPolicyPreserved"]) is True
        and get_path(compositor_contract, ["overflowAwareOrderingReady"]) is True
        and get_path(compositor_contract, ["scalableSortPreparationReady"]) is True
        and get_path(
            compositor_contract,
            ["sortedAccumulationCapacityPolicyUsed"],
        )
        is True
        and get_path(
            compositor_contract,
            ["sortedReferenceCountMatchesSourceOrCapacityPolicy"],
        )
        is True
    )
    reference_seed_usage_ready = (
        get_path(compositor_contract, ["copyBufferUsageValid"]) is True
        and (
            get_path(compositor_contract, ["referenceSeedComputePassUsed"])
            is True
            or (
                get_path(compositor_contract, ["referenceSeedCopyUsed"]) is True
                and get_path(compositor_contract, ["referenceSeedSourceHasCopySrc"])
                is True
                and get_path(
                    compositor_contract,
                    ["referenceSeedDestinationHasCopyDst"],
                )
                is True
            )
        )
    )
    parallel_sort_ready = (
        get_path(compositor_contract, ["gpuParallelPerTileSortReady"]) is True
        and get_path(compositor_contract, ["workgroupParallelSortUsed"]) is True
        and get_path(compositor_contract, ["parallelSortAlgorithm"])
        == "workgroup-bitonic-sort-v1-descending-sort-key"
        and numeric_value(
            get_path(compositor_contract, ["parallelSortStageCount"]),
            0,
        )
        > 0
        and numeric_value(get_path(compositor_contract, ["sortDispatchCount"]), 0)
        > 0
        and numeric_value(get_path(compositor_contract, ["sortWorkgroupCount"]), 0)
        > 0
        and numeric_value(get_path(compositor_contract, ["sortWorkItemCount"]), 0)
        > 0
        and get_path(
            compositor_contract,
            ["parallelSortedBufferReady"],
        )
        is True
        and get_path(
            compositor_contract,
            ["parallelSortedBufferNonEmpty"],
        )
        is True
    )
    order_evidence_ready = (
        get_path(compositor_contract, ["sortOrderSampleCheckReady"]) is True
        and numeric_value(
            get_path(compositor_contract, ["sortOrderViolationCount"]),
            1,
        )
        == 0
        and get_path(compositor_contract, ["depthKeyBufferConsumed"]) is True
    )
    sorted_accumulation_consumed = (
        get_path(
            compositor_contract,
            ["depthSortedReferencesConsumedByAccumulation"],
        )
        is True
        and get_path(compositor_contract, ["sortedAccumulationPathUsed"]) is True
        and get_path(compositor_contract, ["depthOrderedAccumulationUsed"]) is True
        and get_path(compositor_contract, ["alphaAccumulationUsed"]) is True
        and get_path(compositor_contract, ["colorAccumulationUsed"]) is True
        and numeric_value(
            get_path(compositor_contract, ["tileCompositorContributionCount"]),
            0,
        )
        > 0
        and get_path(
            compositor_contract,
            ["parallelSortedBufferPromotedToAccumulation"],
        )
        is True
    )
    error_free = (
        get_path(frame_contract, ["webgpuValidationErrorDetected"]) is False
        and get_path(frame_contract, ["invalidCommandBufferDetected"]) is False
        and get_path(frame_contract, ["queueSubmitFailureDetected"]) is False
        and get_path(frame_contract, ["webgpuWebgl2SameFramePresentationMixed"])
        is False
        and get_path(frame_contract, ["fallbackMixingPrevented"]) is True
        and get_path(frame_contract, ["fullRendererSuccessClaimed"]) is False
    )
    success = (
        phase_step == "phase3-step94"
        and reference_seed_usage_ready
        and parallel_sort_ready
        and order_evidence_ready
        and sorted_accumulation_consumed
        and step93_overflow_policy_preserved
        and step92_bounded_sort_path_preserved
        and step91_ordered_reference_path_preserved
        and step90_runtime_preserved
        and step89_output_preserved
        and step88_presentation_contract_preserved
        and get_path(
            compositor_contract,
            ["visualOutputDegeneratedDetected"],
        )
        is not True
        and error_free
    )
    blocked_reason = None
    if not success:
        if phase_step != "phase3-step94":
            blocked_reason = "summary-phase-step-is-not-phase3-step94"
        elif not reference_seed_usage_ready:
            blocked_reason = "step94-reference-seed-buffer-usage-invalid"
        elif not parallel_sort_ready:
            blocked_reason = "step94-parallel-per-tile-sort-not-ready"
        elif (
            get_path(compositor_contract, ["visualOutputDegeneratedDetected"])
            is True
        ):
            blocked_reason = "step94-visual-output-degenerated"
        elif not order_evidence_ready:
            blocked_reason = "step94-sort-order-evidence-not-ready"
        elif not sorted_accumulation_consumed:
            blocked_reason = "step94-sorted-refs-not-consumed-by-accumulation"
        elif not step93_overflow_policy_preserved:
            blocked_reason = "step93-overflow-policy-not-preserved"
        elif not step92_bounded_sort_path_preserved:
            blocked_reason = "step92-bounded-sort-path-not-preserved"
        elif not step91_ordered_reference_path_preserved:
            blocked_reason = "step91-ordered-reference-runtime-path-not-preserved"
        elif not step90_runtime_preserved:
            blocked_reason = "step90-realtime-runtime-path-not-preserved"
        elif not step89_output_preserved:
            blocked_reason = "step89-real-compositor-output-not-preserved"
        elif not step88_presentation_contract_preserved:
            blocked_reason = "step88-presentation-contract-not-preserved"
        else:
            blocked_reason = "step94-runtime-validation-failed"
    return {
        "step94Decision": "success" if success else "blocked",
        "step94BlockedReason": blocked_reason,
        "step94SelectedGoal":
            "A+C+B+D+E-workgroup-parallel-per-tile-sort-v1",
        "phaseStep": phase_step,
        "step94SummaryApplies": phase_step == "phase3-step94",
        "gpuParallelPerTileSortReady": get_path(
            compositor_contract,
            ["gpuParallelPerTileSortReady"],
        ),
        "workgroupParallelSortUsed": get_path(
            compositor_contract,
            ["workgroupParallelSortUsed"],
        ),
        "parallelSortAlgorithm": get_path(
            compositor_contract,
            ["parallelSortAlgorithm"],
        ),
        "parallelSortStageCount": get_path(
            compositor_contract,
            ["parallelSortStageCount"],
        ),
        "sortDispatchCount": get_path(compositor_contract, ["sortDispatchCount"]),
        "sortWorkgroupCount": get_path(
            compositor_contract,
            ["sortWorkgroupCount"],
        ),
        "sortWorkItemCount": get_path(
            compositor_contract,
            ["sortWorkItemCount"],
        ),
        "sortedTileCount": get_path(compositor_contract, ["sortedTileCount"]),
        "sortedReferenceCount": get_path(
            compositor_contract,
            ["sortedReferenceCount"],
        ),
        "sortOrderViolationCount": get_path(
            compositor_contract,
            ["sortOrderViolationCount"],
        ),
        "sortOrderSampleCheckReady": get_path(
            compositor_contract,
            ["sortOrderSampleCheckReady"],
        ),
        "parallelSortFailureReason": get_path(
            compositor_contract,
            ["parallelSortFailureReason"],
        ),
        "parallelSortedBufferPromotedToAccumulation": get_path(
            compositor_contract,
            ["parallelSortedBufferPromotedToAccumulation"],
        ),
        "parallelSortedBufferReady": get_path(
            compositor_contract,
            ["parallelSortedBufferReady"],
        ),
        "parallelSortedBufferNonEmpty": get_path(
            compositor_contract,
            ["parallelSortedBufferNonEmpty"],
        ),
        "referenceSeedCopyUsed": get_path(
            compositor_contract,
            ["referenceSeedCopyUsed"],
        ),
        "referenceSeedComputePassUsed": get_path(
            compositor_contract,
            ["referenceSeedComputePassUsed"],
        ),
        "referenceSeedSourceHasCopySrc": get_path(
            compositor_contract,
            ["referenceSeedSourceHasCopySrc"],
        ),
        "referenceSeedDestinationHasCopyDst": get_path(
            compositor_contract,
            ["referenceSeedDestinationHasCopyDst"],
        ),
        "copyBufferUsageValid": get_path(
            compositor_contract,
            ["copyBufferUsageValid"],
        ),
        "parallelSortOutputGuardUsed": get_path(
            compositor_contract,
            ["parallelSortOutputGuardUsed"],
        ),
        "preservedBoundedSortFallbackUsed": get_path(
            compositor_contract,
            ["preservedBoundedSortFallbackUsed"],
        ),
        "visualOutputDegeneratedDetected": get_path(
            compositor_contract,
            ["visualOutputDegeneratedDetected"],
        ),
        "depthSortedReferencesConsumedByAccumulation": get_path(
            compositor_contract,
            ["depthSortedReferencesConsumedByAccumulation"],
        ),
        "sortedAccumulationPathUsed": get_path(
            compositor_contract,
            ["sortedAccumulationPathUsed"],
        ),
        "tileReferenceCount": get_path(compositor_contract, ["tileReferenceCount"]),
        "orderedReferenceCount": get_path(
            compositor_contract,
            ["orderedReferenceCount"],
        ),
        "orderedReferenceCountMatchesSource": get_path(
            compositor_contract,
            ["orderedReferenceCountMatchesSource"],
        ),
        "sortCapacityLimit": get_path(compositor_contract, ["sortCapacityLimit"]),
        "overflowTileCount": get_path(compositor_contract, ["overflowTileCount"]),
        "overflowReferenceCount": get_path(
            compositor_contract,
            ["overflowReferenceCount"],
        ),
        "droppedReferenceCount": get_path(
            compositor_contract,
            ["droppedReferenceCount"],
        ),
        "sortedReferenceCountMatchesSourceOrCapacityPolicy": get_path(
            compositor_contract,
            ["sortedReferenceCountMatchesSourceOrCapacityPolicy"],
        ),
        "capacityUtilizationMax": get_path(
            compositor_contract,
            ["capacityUtilizationMax"],
        ),
        "capacityUtilizationAvg": get_path(
            compositor_contract,
            ["capacityUtilizationAvg"],
        ),
        "step93OverflowPolicyPreserved": step93_overflow_policy_preserved,
        "step92BoundedSortPathPreserved": step92_bounded_sort_path_preserved,
        "step91OrderedReferenceRuntimePathPreserved":
            step91_ordered_reference_path_preserved,
        "step90RuntimePathPreserved": step90_runtime_preserved,
        "step89RealCompositorOutputPreserved": step89_output_preserved,
        "step88PresentationContractPreserved":
            step88_presentation_contract_preserved,
        "steadyStateTileCompositorOwnsFinalPresentation": get_path(
            frame_contract,
            ["steadyStateTileCompositorOwnsFinalPresentation"],
        ),
        "steadyStateBlankFrameCount": get_path(
            frame_contract,
            ["steadyStateBlankFrameCount"],
        ),
        "steadyStateNoOpFrameCount": get_path(
            frame_contract,
            ["steadyStateNoOpFrameCount"],
        ),
        "steadyStateClearFrameCount": get_path(
            frame_contract,
            ["steadyStateClearFrameCount"],
        ),
        "steadyStateUnknownFrameCount": get_path(
            frame_contract,
            ["steadyStateUnknownFrameCount"],
        ),
        "currentTextureUsesWebGpuTileCompositorOutput": get_path(
            frame_contract,
            ["currentTextureUsesWebGpuTileCompositorOutput"],
        ),
        "currentTextureViewFreshPerPresentation": get_path(
            frame_contract,
            ["currentTextureViewFreshPerPresentation"],
        ),
        "webgpuDeviceConsistencyReady": get_path(
            frame_contract,
            ["webgpuDeviceConsistencyReady"],
        ),
        "webgpuValidationErrorDetected": get_path(
            frame_contract,
            ["webgpuValidationErrorDetected"],
        ),
        "invalidCommandBufferDetected": get_path(
            frame_contract,
            ["invalidCommandBufferDetected"],
        ),
        "queueSubmitFailureDetected": get_path(
            frame_contract,
            ["queueSubmitFailureDetected"],
        ),
        "webgpuWebgl2SameFramePresentationMixed": get_path(
            frame_contract,
            ["webgpuWebgl2SameFramePresentationMixed"],
        ),
        "fallbackMixingPrevented": get_path(
            frame_contract,
            ["fallbackMixingPrevented"],
        ),
        "generatedCompositorFields": get_path(
            compositor_contract,
            ["generatedCompositorFields"],
            [],
        ),
        "deferredProductionItems": get_path(
            compositor_contract,
            ["deferredProductionItems"],
            [],
        ),
        "deferredCompositorFields": get_path(
            compositor_contract,
            ["deferredCompositorFields"],
            [],
        ),
        "fullRendererSuccessClaimed": get_path(
            frame_contract,
            ["fullRendererSuccessClaimed"],
        ),
    }


def build_step96_production_tile_compositor_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    frame_contract = get_path(
        summary,
        ["webgpuTileCompositorFrameImplementation"],
        {},
    )
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    error_subtypes = detect_webgpu_error_subtypes(
        frame_contract,
        compositor_contract,
        get_path(summary, ["captureErrorString"]),
        get_path(summary, ["captureErrorStack"]),
        get_path(summary, ["captureErrorMessage"]),
        get_path(summary, ["firstValidationFailures"]),
    )
    wgsl_parse_error_detected = (
        get_path(frame_contract, ["wgslParseErrorDetected"]) is True
        or get_path(compositor_contract, ["wgslParseErrorDetected"]) is True
        or error_subtypes["wgslParseErrorDetected"]
    )
    shader_module_invalid_detected = (
        get_path(frame_contract, ["shaderModuleInvalidDetected"]) is True
        or get_path(compositor_contract, ["shaderModuleInvalidDetected"]) is True
        or error_subtypes["shaderModuleInvalidDetected"]
    )
    compute_pipeline_invalid_detected = (
        get_path(frame_contract, ["computePipelineInvalidDetected"]) is True
        or get_path(compositor_contract, ["computePipelineInvalidDetected"]) is True
        or error_subtypes["computePipelineInvalidDetected"]
    )
    bind_group_invalid_detected = (
        get_path(frame_contract, ["bindGroupInvalidDetected"]) is True
        or get_path(compositor_contract, ["bindGroupInvalidDetected"]) is True
        or error_subtypes["bindGroupInvalidDetected"]
    )
    phase_step = get_path(summary, ["phaseStep"])
    step94_parallel_sort_preserved = (
        get_path(compositor_contract, ["step94ParallelSortPreserved"]) is True
        and get_path(compositor_contract, ["gpuParallelPerTileSortReady"]) is True
        and get_path(compositor_contract, ["workgroupParallelSortUsed"]) is True
        and get_path(compositor_contract, ["parallelSortedBufferReady"]) is True
        and get_path(compositor_contract, ["parallelSortedBufferNonEmpty"]) is True
        and get_path(
            compositor_contract,
            ["parallelSortedBufferPromotedToAccumulation"],
        )
        is True
    )
    step93_overflow_policy_preserved = (
        get_path(compositor_contract, ["step93OverflowPolicyPreserved"]) is True
        and get_path(compositor_contract, ["overflowAwareOrderingReady"]) is True
        and get_path(compositor_contract, ["scalableSortPreparationReady"]) is True
        and get_path(
            compositor_contract,
            ["sortedAccumulationCapacityPolicyUsed"],
        )
        is True
    )
    step92_bounded_sort_path_preserved = (
        get_path(compositor_contract, ["gpuSidePerTileSortReady"]) is True
        and get_path(compositor_contract, ["boundedPerTileSortUsed"]) is True
        and get_path(
            compositor_contract,
            ["sortedReferenceCountMatchesSourceOrCapacityPolicy"],
        )
        is True
    )
    step91_ordered_reference_path_preserved = (
        get_path(compositor_contract, ["step91OrderedReferenceRuntimePathPreserved"])
        is True
        and get_path(compositor_contract, ["orderedReferencesConsumedByProductionAccumulation"])
        is True
    )
    step90_runtime_preserved = (
        get_path(compositor_contract, ["step90RuntimePathPreserved"]) is True
        and get_path(compositor_contract, ["realTimeRuntimePathReady"]) is True
        and get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        )
        is True
    )
    step89_output_preserved = (
        get_path(compositor_contract, ["step89RealCompositorOutputPreserved"]) is True
        and get_path(compositor_contract, ["realTileCompositorOutputReady"]) is True
    )
    step88_presentation_contract_preserved = (
        get_path(compositor_contract, ["step88PresentationContractPreserved"]) is True
        or get_path(
            frame_contract,
            ["steadyStateTileCompositorOwnsFinalPresentation"],
        )
        is True
    )
    production_ready = (
        phase_step == "phase3-step96"
        and get_path(compositor_contract, ["productionTileCompositorReady"]) is True
        and get_path(compositor_contract, ["productionTileCompositorPathUsed"]) is True
        and get_path(
            compositor_contract,
            ["productionAccumulationConsumedParallelSortedRefs"],
        )
        is True
        and get_path(compositor_contract, ["activeTileDispatchReady"]) is True
        and get_path(compositor_contract, ["activeTileDispatchUsed"]) is True
        and numeric_value(get_path(compositor_contract, ["activeTileCount"]), 0) > 0
        and numeric_value(
            get_path(compositor_contract, ["activeTilePixelWorkItemCount"]),
            0,
        )
        > 0
        and get_path(compositor_contract, ["inactiveBackgroundHandlingReady"]) is True
        and get_path(
            compositor_contract,
            ["outputTextureProducedByProductionCompositor"],
        )
        is True
        and get_path(compositor_contract, ["readyBufferGuardUsed"]) is True
        and get_path(compositor_contract, ["debugOutputBypassedForProduction"])
        is True
        and get_path(compositor_contract, ["fallbackOnlyCompositorUsed"]) is False
        and get_path(compositor_contract, ["normalBackendPresentationUsed"]) is False
        and numeric_value(
            get_path(compositor_contract, ["webgl2FallbackFinalPresentFrameCount"]),
            1,
        )
        == 0
        and get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromProductionPath"],
        )
        is True
        and step94_parallel_sort_preserved
        and step93_overflow_policy_preserved
        and step92_bounded_sort_path_preserved
        and step91_ordered_reference_path_preserved
        and step90_runtime_preserved
        and step89_output_preserved
        and step88_presentation_contract_preserved
        and get_path(compositor_contract, ["visualOutputDegeneratedDetected"])
        is not True
        and get_path(frame_contract, ["webgpuValidationErrorDetected"]) is False
        and get_path(frame_contract, ["invalidCommandBufferDetected"]) is False
        and get_path(frame_contract, ["queueSubmitFailureDetected"]) is False
        and get_path(frame_contract, ["webgpuWebgl2SameFramePresentationMixed"])
        is False
        and get_path(frame_contract, ["fallbackMixingPrevented"]) is True
        and get_path(frame_contract, ["fullRendererSuccessClaimed"]) is False
    )
    blocked_reason = None
    if not production_ready:
        if phase_step != "phase3-step96":
            blocked_reason = "summary-phase-step-is-not-phase3-step96"
        elif get_path(compositor_contract, ["productionTileCompositorReady"]) is not True:
            blocked_reason = "production-tile-compositor-not-ready"
        elif get_path(
            compositor_contract,
            ["productionAccumulationConsumedParallelSortedRefs"],
        ) is not True:
            blocked_reason = "production-accumulation-did-not-consume-parallel-sorted-refs"
        elif get_path(compositor_contract, ["activeTileDispatchUsed"]) is not True:
            blocked_reason = "active-tile-dispatch-not-used"
        elif get_path(compositor_contract, ["inactiveBackgroundHandlingReady"]) is not True:
            blocked_reason = "inactive-background-handling-not-ready"
        elif not step94_parallel_sort_preserved:
            blocked_reason = "step94-parallel-sort-not-preserved"
        elif not step93_overflow_policy_preserved:
            blocked_reason = "step93-overflow-policy-not-preserved"
        elif not step92_bounded_sort_path_preserved:
            blocked_reason = "step92-bounded-sort-path-not-preserved"
        elif not step91_ordered_reference_path_preserved:
            blocked_reason = "step91-ordered-reference-runtime-path-not-preserved"
        elif not step90_runtime_preserved:
            blocked_reason = "step90-runtime-path-not-preserved"
        elif not step89_output_preserved:
            blocked_reason = "step89-real-compositor-output-not-preserved"
        elif not step88_presentation_contract_preserved:
            blocked_reason = "step88-presentation-contract-not-preserved"
        elif get_path(compositor_contract, ["fallbackOnlyCompositorUsed"]) is True:
            blocked_reason = "fallback-only-compositor-used"
        elif get_path(compositor_contract, ["visualOutputDegeneratedDetected"]) is True:
            blocked_reason = "visual-output-degenerated"
        elif wgsl_parse_error_detected:
            blocked_reason = "wgsl-parse-error-detected"
        elif shader_module_invalid_detected:
            blocked_reason = "shader-module-invalid-detected"
        elif compute_pipeline_invalid_detected:
            blocked_reason = "compute-pipeline-invalid-detected"
        elif bind_group_invalid_detected:
            blocked_reason = "bind-group-invalid-detected"
        else:
            blocked_reason = "step96-runtime-validation-failed"
    return {
        "step96Decision": "success" if production_ready else "blocked",
        "step96BlockedReason": blocked_reason,
        "step96SelectedGoal":
            "A+B+C+D-production-tile-compositor-v1-integration",
        "phaseStep": phase_step,
        "step96SummaryApplies": phase_step == "phase3-step96",
        "productionTileCompositorReady": get_path(
            compositor_contract,
            ["productionTileCompositorReady"],
        ),
        "productionTileCompositorPathUsed": get_path(
            compositor_contract,
            ["productionTileCompositorPathUsed"],
        ),
        "productionAccumulationConsumedParallelSortedRefs": get_path(
            compositor_contract,
            ["productionAccumulationConsumedParallelSortedRefs"],
        ),
        "parallelSortedBufferReady": get_path(
            compositor_contract,
            ["parallelSortedBufferReady"],
        ),
        "parallelSortedBufferNonEmpty": get_path(
            compositor_contract,
            ["parallelSortedBufferNonEmpty"],
        ),
        "parallelSortedBufferPromotedToAccumulation": get_path(
            compositor_contract,
            ["parallelSortedBufferPromotedToAccumulation"],
        ),
        "activeTileDispatchReady": get_path(
            compositor_contract,
            ["activeTileDispatchReady"],
        ),
        "activeTileDispatchUsed": get_path(
            compositor_contract,
            ["activeTileDispatchUsed"],
        ),
        "activeTileCount": get_path(compositor_contract, ["activeTileCount"]),
        "nonEmptyTileCount": get_path(
            compositor_contract,
            ["nonEmptyCompositedTileCount"],
        ),
        "activeTilePixelWorkItemCount": get_path(
            compositor_contract,
            ["activeTilePixelWorkItemCount"],
        ),
        "fullScreenPixelWorkAvoided": get_path(
            compositor_contract,
            ["fullScreenPixelWorkAvoided"],
        ),
        "accumulationWorkReductionRatio": get_path(
            compositor_contract,
            ["accumulationWorkReductionRatio"],
        ),
        "inactiveBackgroundHandlingReady": get_path(
            compositor_contract,
            ["inactiveBackgroundHandlingReady"],
        ),
        "inactiveTileCount": get_path(
            compositor_contract,
            ["inactiveTileCount"],
        ),
        "inactivePixelOrTileWritePolicy": get_path(
            compositor_contract,
            ["inactivePixelOrTileWritePolicy"],
        ),
        "outputTextureProducedByProductionCompositor": get_path(
            compositor_contract,
            ["outputTextureProducedByProductionCompositor"],
        ),
        "lastValidOutputPreservedForCleanFrames": get_path(
            compositor_contract,
            ["lastValidOutputPreservedForCleanFrames"],
        ),
        "readyBufferGuardUsed": get_path(
            compositor_contract,
            ["readyBufferGuardUsed"],
        ),
        "invalidOrEmptyBufferRejected": get_path(
            compositor_contract,
            ["invalidOrEmptyBufferRejected"],
        ),
        "debugOutputBypassedForProduction": get_path(
            compositor_contract,
            ["debugOutputBypassedForProduction"],
        ),
        "fallbackOnlyCompositorUsed": get_path(
            compositor_contract,
            ["fallbackOnlyCompositorUsed"],
        ),
        "diagnosticReadbackSeparatedFromProductionPath": get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromProductionPath"],
        ),
        "normalBackendPresentationUsed": get_path(
            compositor_contract,
            ["normalBackendPresentationUsed"],
        ),
        "webgl2FallbackFinalPresentFrameCount": get_path(
            compositor_contract,
            ["webgl2FallbackFinalPresentFrameCount"],
        ),
        "visualOutputDegeneratedDetected": get_path(
            compositor_contract,
            ["visualOutputDegeneratedDetected"],
        ),
        "step94ParallelSortPreserved": step94_parallel_sort_preserved,
        "step93OverflowPolicyPreserved": step93_overflow_policy_preserved,
        "step92BoundedSortPathPreserved": step92_bounded_sort_path_preserved,
        "step91OrderedReferenceRuntimePathPreserved":
            step91_ordered_reference_path_preserved,
        "step90RuntimePathPreserved": step90_runtime_preserved,
        "step89RealCompositorOutputPreserved": step89_output_preserved,
        "step88PresentationContractPreserved":
            step88_presentation_contract_preserved,
        "steadyStateTileCompositorOwnsFinalPresentation": get_path(
            frame_contract,
            ["steadyStateTileCompositorOwnsFinalPresentation"],
        ),
        "steadyStateBlankFrameCount": get_path(
            frame_contract,
            ["steadyStateBlankFrameCount"],
        ),
        "steadyStateNoOpFrameCount": get_path(
            frame_contract,
            ["steadyStateNoOpFrameCount"],
        ),
        "steadyStateClearFrameCount": get_path(
            frame_contract,
            ["steadyStateClearFrameCount"],
        ),
        "steadyStateUnknownFrameCount": get_path(
            frame_contract,
            ["steadyStateUnknownFrameCount"],
        ),
        "currentTextureUsesWebGpuTileCompositorOutput": get_path(
            frame_contract,
            ["currentTextureUsesWebGpuTileCompositorOutput"],
        ),
        "currentTextureViewFreshPerPresentation": get_path(
            frame_contract,
            ["currentTextureViewFreshPerPresentation"],
        ),
        "webgpuDeviceConsistencyReady": get_path(
            frame_contract,
            ["webgpuDeviceConsistencyReady"],
        ),
        "webgpuValidationErrorDetected": get_path(
            frame_contract,
            ["webgpuValidationErrorDetected"],
        ),
        "wgslParseErrorDetected": wgsl_parse_error_detected,
        "shaderModuleInvalidDetected": shader_module_invalid_detected,
        "computePipelineInvalidDetected": compute_pipeline_invalid_detected,
        "bindGroupInvalidDetected": bind_group_invalid_detected,
        "invalidCommandBufferDetected": get_path(
            frame_contract,
            ["invalidCommandBufferDetected"],
        ),
        "queueSubmitFailureDetected": get_path(
            frame_contract,
            ["queueSubmitFailureDetected"],
        ),
        "webgpuWebgl2SameFramePresentationMixed": get_path(
            frame_contract,
            ["webgpuWebgl2SameFramePresentationMixed"],
        ),
        "fallbackMixingPrevented": get_path(
            frame_contract,
            ["fallbackMixingPrevented"],
        ),
        "generatedCompositorFields": get_path(
            compositor_contract,
            ["generatedCompositorFields"],
            [],
        ),
        "deferredProductionItems": get_path(
            compositor_contract,
            ["deferredProductionItems"],
            [],
        ),
        "deferredCompositorFields": get_path(
            compositor_contract,
            ["deferredCompositorFields"],
            [],
        ),
        "fullRendererSuccessClaimed": get_path(
            frame_contract,
            ["fullRendererSuccessClaimed"],
        ),
    }


def build_step97_time_driven_production_runtime_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    frame_contract = get_path(
        summary,
        ["webgpuTileCompositorFrameImplementation"],
        {},
    )
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    error_subtypes = detect_webgpu_error_subtypes(
        frame_contract,
        compositor_contract,
        get_path(summary, ["captureErrorString"]),
        get_path(summary, ["captureErrorStack"]),
        get_path(summary, ["captureErrorMessage"]),
        get_path(summary, ["firstValidationFailures"]),
    )
    phase_step = get_path(summary, ["phaseStep"])
    wgsl_parse_error_detected = (
        get_path(frame_contract, ["wgslParseErrorDetected"]) is True
        or get_path(compositor_contract, ["wgslParseErrorDetected"]) is True
        or error_subtypes["wgslParseErrorDetected"]
    )
    shader_module_invalid_detected = (
        get_path(frame_contract, ["shaderModuleInvalidDetected"]) is True
        or get_path(compositor_contract, ["shaderModuleInvalidDetected"]) is True
        or error_subtypes["shaderModuleInvalidDetected"]
    )
    compute_pipeline_invalid_detected = (
        get_path(frame_contract, ["computePipelineInvalidDetected"]) is True
        or get_path(compositor_contract, ["computePipelineInvalidDetected"]) is True
        or error_subtypes["computePipelineInvalidDetected"]
    )
    bind_group_invalid_detected = (
        get_path(frame_contract, ["bindGroupInvalidDetected"]) is True
        or get_path(compositor_contract, ["bindGroupInvalidDetected"]) is True
        or error_subtypes["bindGroupInvalidDetected"]
    )
    step96_production_preserved = (
        get_path(compositor_contract, ["step96ProductionTileCompositorPreserved"])
        is True
        and get_path(compositor_contract, ["productionTileCompositorReady"]) is True
        and get_path(compositor_contract, ["productionTileCompositorPathUsed"]) is True
        and get_path(
            compositor_contract,
            ["productionAccumulationConsumedParallelSortedRefs"],
        )
        is True
    )
    step94_parallel_sort_preserved = (
        get_path(compositor_contract, ["step94ParallelSortPreserved"]) is True
        and get_path(compositor_contract, ["gpuParallelPerTileSortReady"]) is True
        and get_path(compositor_contract, ["workgroupParallelSortUsed"]) is True
    )
    step93_overflow_policy_preserved = (
        get_path(compositor_contract, ["step93OverflowPolicyPreserved"]) is True
        and get_path(compositor_contract, ["overflowAwareOrderingReady"]) is True
    )
    step90_runtime_preserved = (
        get_path(compositor_contract, ["step90RuntimePathPreserved"]) is True
        and get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        )
        is True
    )
    step88_presentation_contract_preserved = (
        get_path(compositor_contract, ["step88PresentationContractPreserved"]) is True
        or get_path(
            frame_contract,
            ["steadyStateTileCompositorOwnsFinalPresentation"],
        )
        is True
    )
    runtime_frame_count = numeric_value(
        get_path(compositor_contract, ["runtimeFrameCount"]),
        0,
    )
    updated_stage_names = get_path(
        compositor_contract,
        ["updatedStageNames"],
        [],
    )
    skipped_stage_names = get_path(
        compositor_contract,
        ["skippedStageNames"],
        [],
    )
    raw_deferred_production_items = get_path(
        compositor_contract,
        ["deferredProductionItems"],
        [],
    )
    deferred_production_items = (
        list(raw_deferred_production_items)
        if isinstance(raw_deferred_production_items, list)
        else []
    )
    if "full-interactive-scheduler" not in deferred_production_items:
        deferred_production_items.append("full-interactive-scheduler")
    time_driven_ready = (
        phase_step == "phase3-step97"
        and get_path(compositor_contract, ["timeDrivenProductionRuntimeReady"])
        is True
        and get_path(compositor_contract, ["multiFrameProductionRuntimeUsed"])
        is True
        and runtime_frame_count >= 2
        and get_path(compositor_contract, ["timeStateAdvancedAcrossFrames"])
        is True
        and get_path(compositor_contract, ["frameStateAdvancedAcrossFrames"])
        is True
        and get_path(compositor_contract, ["productionOutputUpdatedAcrossFrames"])
        is True
        and get_path(compositor_contract, ["dirtyDependencyExecutorUsed"]) is True
        and isinstance(updated_stage_names, list)
        and len(updated_stage_names) > 0
        and isinstance(skipped_stage_names, list)
        and get_path(
            compositor_contract,
            ["productionCompositorUpdatedOnDirtyFrames"],
        )
        is True
        and get_path(compositor_contract, ["cleanFrameFastPathUsed"]) is True
        and get_path(compositor_contract, ["lastValidProductionOutputReused"])
        is True
        and get_path(
            compositor_contract,
            ["lastValidOutputPresentedOnCleanFrames"],
        )
        is True
        and get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        )
        is True
        and get_path(compositor_contract, ["fallbackOnlyCompositorUsed"]) is False
        and get_path(compositor_contract, ["debugOutputBypassedForProduction"])
        is True
        and get_path(compositor_contract, ["visualOutputDegeneratedDetected"])
        is not True
        and step96_production_preserved
        and step94_parallel_sort_preserved
        and step93_overflow_policy_preserved
        and step90_runtime_preserved
        and step88_presentation_contract_preserved
        and wgsl_parse_error_detected is False
        and shader_module_invalid_detected is False
        and compute_pipeline_invalid_detected is False
        and bind_group_invalid_detected is False
        and get_path(frame_contract, ["webgpuValidationErrorDetected"]) is False
        and get_path(frame_contract, ["invalidCommandBufferDetected"]) is False
        and get_path(frame_contract, ["queueSubmitFailureDetected"]) is False
        and get_path(frame_contract, ["webgpuWebgl2SameFramePresentationMixed"])
        is False
        and get_path(frame_contract, ["fallbackMixingPrevented"]) is True
        and get_path(frame_contract, ["fullRendererSuccessClaimed"]) is False
    )
    blocked_reason = None
    if not time_driven_ready:
        if phase_step != "phase3-step97":
            blocked_reason = "summary-phase-step-is-not-phase3-step97"
        elif get_path(compositor_contract, ["timeDrivenProductionRuntimeReady"]) is not True:
            blocked_reason = "time-driven-production-runtime-not-ready"
        elif get_path(compositor_contract, ["multiFrameProductionRuntimeUsed"]) is not True:
            blocked_reason = "multi-frame-production-runtime-not-used"
        elif runtime_frame_count < 2:
            blocked_reason = "runtime-frame-count-too-low"
        elif get_path(compositor_contract, ["productionOutputUpdatedAcrossFrames"]) is not True:
            blocked_reason = "production-output-not-updated-across-frames"
        elif get_path(compositor_contract, ["cleanFrameFastPathUsed"]) is not True:
            blocked_reason = "clean-frame-fast-path-not-used"
        elif not step96_production_preserved:
            blocked_reason = "step96-production-tile-compositor-not-preserved"
        elif not step94_parallel_sort_preserved:
            blocked_reason = "step94-parallel-sort-not-preserved"
        elif not step93_overflow_policy_preserved:
            blocked_reason = "step93-overflow-policy-not-preserved"
        elif not step90_runtime_preserved:
            blocked_reason = "step90-runtime-path-not-preserved"
        elif not step88_presentation_contract_preserved:
            blocked_reason = "step88-presentation-contract-not-preserved"
        elif get_path(compositor_contract, ["fallbackOnlyCompositorUsed"]) is True:
            blocked_reason = "fallback-only-compositor-used"
        elif wgsl_parse_error_detected:
            blocked_reason = "wgsl-parse-error-detected"
        elif shader_module_invalid_detected:
            blocked_reason = "shader-module-invalid-detected"
        elif compute_pipeline_invalid_detected:
            blocked_reason = "compute-pipeline-invalid-detected"
        elif bind_group_invalid_detected:
            blocked_reason = "bind-group-invalid-detected"
        else:
            blocked_reason = "step97-runtime-validation-failed"
    return {
        "step97Decision": "success" if time_driven_ready else "blocked",
        "step97BlockedReason": blocked_reason,
        "step97SelectedGoal":
            "A+B+C+D-time-driven-multi-frame-production-runtime-v1",
        "phaseStep": phase_step,
        "step97SummaryApplies": phase_step == "phase3-step97",
        "timeDrivenProductionRuntimeReady": get_path(
            compositor_contract,
            ["timeDrivenProductionRuntimeReady"],
        ),
        "multiFrameProductionRuntimeUsed": get_path(
            compositor_contract,
            ["multiFrameProductionRuntimeUsed"],
        ),
        "runtimeFrameCount": get_path(compositor_contract, ["runtimeFrameCount"]),
        "timeStateAdvancedAcrossFrames": get_path(
            compositor_contract,
            ["timeStateAdvancedAcrossFrames"],
        ),
        "frameStateAdvancedAcrossFrames": get_path(
            compositor_contract,
            ["frameStateAdvancedAcrossFrames"],
        ),
        "productionOutputUpdatedAcrossFrames": get_path(
            compositor_contract,
            ["productionOutputUpdatedAcrossFrames"],
        ),
        "dirtyDependencyExecutorUsed": get_path(
            compositor_contract,
            ["dirtyDependencyExecutorUsed"],
        ),
        "updatedStageNames": updated_stage_names,
        "skippedStageNames": skipped_stage_names,
        "productionCompositorUpdatedOnDirtyFrames": get_path(
            compositor_contract,
            ["productionCompositorUpdatedOnDirtyFrames"],
        ),
        "cleanFrameFastPathUsed": get_path(
            compositor_contract,
            ["cleanFrameFastPathUsed"],
        ),
        "lastValidProductionOutputReused": get_path(
            compositor_contract,
            ["lastValidProductionOutputReused"],
        ),
        "lastValidOutputPresentedOnCleanFrames": get_path(
            compositor_contract,
            ["lastValidOutputPresentedOnCleanFrames"],
        ),
        "diagnosticReadbackSeparatedFromRuntimePath": get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        ),
        "fallbackOnlyCompositorUsed": get_path(
            compositor_contract,
            ["fallbackOnlyCompositorUsed"],
        ),
        "debugOutputBypassedForProduction": get_path(
            compositor_contract,
            ["debugOutputBypassedForProduction"],
        ),
        "visualOutputDegeneratedDetected": get_path(
            compositor_contract,
            ["visualOutputDegeneratedDetected"],
        ),
        "step96ProductionTileCompositorPreserved":
            step96_production_preserved,
        "step94ParallelSortPreserved": step94_parallel_sort_preserved,
        "step93OverflowPolicyPreserved": step93_overflow_policy_preserved,
        "step90RuntimePathPreserved": step90_runtime_preserved,
        "step88PresentationContractPreserved":
            step88_presentation_contract_preserved,
        "step89RealCompositorOutputPreserved": get_path(
            compositor_contract,
            ["step89RealCompositorOutputPreserved"],
        ),
        "step91OrderedReferenceRuntimePathPreserved": get_path(
            compositor_contract,
            ["step91OrderedReferenceRuntimePathPreserved"],
        ),
        "step92BoundedSortPathPreserved": get_path(
            compositor_contract,
            ["gpuSidePerTileSortReady"],
        ),
        "step85TileCompositorPathPreserved": get_path(
            frame_contract,
            ["step85TileCompositorPathPreserved"],
        ),
        "step86BoundaryContractPreserved": get_path(
            frame_contract,
            ["step86BoundaryContractPreserved"],
        ),
        "step87DepthOrderingPreserved": get_path(
            frame_contract,
            ["step87DepthOrderingPreserved"],
        ),
        "currentTextureUsesWebGpuTileCompositorOutput": get_path(
            frame_contract,
            ["currentTextureUsesWebGpuTileCompositorOutput"],
        ),
        "currentTextureViewFreshPerPresentation": get_path(
            frame_contract,
            ["currentTextureViewFreshPerPresentation"],
        ),
        "webgpuDeviceConsistencyReady": get_path(
            frame_contract,
            ["webgpuDeviceConsistencyReady"],
        ),
        "wgslParseErrorDetected": wgsl_parse_error_detected,
        "shaderModuleInvalidDetected": shader_module_invalid_detected,
        "computePipelineInvalidDetected": compute_pipeline_invalid_detected,
        "bindGroupInvalidDetected": bind_group_invalid_detected,
        "invalidCommandBufferDetected": get_path(
            frame_contract,
            ["invalidCommandBufferDetected"],
        ),
        "queueSubmitFailureDetected": get_path(
            frame_contract,
            ["queueSubmitFailureDetected"],
        ),
        "deferredProductionItems": deferred_production_items,
        "fullRendererSuccessClaimed": get_path(
            frame_contract,
            ["fullRendererSuccessClaimed"],
        ),
    }


def build_step98_viewer_connected_interactive_scheduler_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    frame_contract = get_path(
        summary,
        ["webgpuTileCompositorFrameImplementation"],
        {},
    )
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    error_subtypes = detect_webgpu_error_subtypes(
        frame_contract,
        compositor_contract,
        get_path(summary, ["captureErrorString"]),
        get_path(summary, ["captureErrorStack"]),
        get_path(summary, ["captureErrorMessage"]),
        get_path(summary, ["firstValidationFailures"]),
    )
    phase_step = get_path(summary, ["phaseStep"])
    wgsl_parse_error_detected = (
        get_path(frame_contract, ["wgslParseErrorDetected"]) is True
        or get_path(compositor_contract, ["wgslParseErrorDetected"]) is True
        or error_subtypes["wgslParseErrorDetected"]
    )
    shader_module_invalid_detected = (
        get_path(frame_contract, ["shaderModuleInvalidDetected"]) is True
        or get_path(compositor_contract, ["shaderModuleInvalidDetected"]) is True
        or error_subtypes["shaderModuleInvalidDetected"]
    )
    compute_pipeline_invalid_detected = (
        get_path(frame_contract, ["computePipelineInvalidDetected"]) is True
        or get_path(compositor_contract, ["computePipelineInvalidDetected"]) is True
        or error_subtypes["computePipelineInvalidDetected"]
    )
    bind_group_invalid_detected = (
        get_path(frame_contract, ["bindGroupInvalidDetected"]) is True
        or get_path(compositor_contract, ["bindGroupInvalidDetected"]) is True
        or error_subtypes["bindGroupInvalidDetected"]
    )
    deferred_production_items = get_path(
        compositor_contract,
        ["deferredProductionItems"],
        [],
    )
    if not isinstance(deferred_production_items, list):
        deferred_production_items = []
    else:
        deferred_production_items = list(deferred_production_items)
    for item in [
        "full-cuda-parity",
        "final-production-compositor",
        "full-parallel-sort-parity",
        "chunk-lod-streaming",
        "complete-interactive-control-parity",
    ]:
        if item not in deferred_production_items:
            deferred_production_items.append(item)

    scheduler_frame_count = numeric_value(
        get_path(compositor_contract, ["schedulerFrameCount"]),
        0,
    )
    viewer_time_before = get_path(compositor_contract, ["viewerTimeBefore"])
    viewer_time_after = get_path(compositor_contract, ["viewerTimeAfter"])
    viewer_time_delta = get_path(compositor_contract, ["viewerTimeDelta"])
    time_control_evidence_ready = (
        get_path(compositor_contract, ["timeControlEvidenceReady"]) is True
        and numeric_value(viewer_time_before, None) is not None
        and numeric_value(viewer_time_after, None) is not None
        and numeric_value(viewer_time_delta, None) is not None
        and numeric_value(viewer_time_before, 0)
        != numeric_value(viewer_time_after, 0)
        and get_path(
            compositor_contract,
            ["timeControlEvidenceFromSchedulerProbe"],
        )
        is True
        and get_path(compositor_contract, ["timeControlEvidenceUsesFixedValue"])
        is not True
    )
    step97_multi_frame_preserved = (
        get_path(compositor_contract, ["step97MultiFrameRuntimePreserved"])
        is True
        and get_path(compositor_contract, ["timeDrivenProductionRuntimeReady"])
        is True
        and get_path(compositor_contract, ["multiFrameProductionRuntimeUsed"])
        is True
    )
    step96_production_preserved = (
        get_path(compositor_contract, ["step96ProductionTileCompositorPreserved"])
        is True
        and get_path(compositor_contract, ["productionTileCompositorReady"]) is True
    )
    step94_parallel_sort_preserved = (
        get_path(compositor_contract, ["step94ParallelSortPreserved"]) is True
        and get_path(compositor_contract, ["gpuParallelPerTileSortReady"]) is True
    )
    step88_presentation_contract_preserved = (
        get_path(compositor_contract, ["step88PresentationContractPreserved"]) is True
        or get_path(
            frame_contract,
            ["steadyStateTileCompositorOwnsFinalPresentation"],
        )
        is True
    )
    scheduler_ready = (
        phase_step == "phase3-step98"
        and get_path(
            compositor_contract,
            ["viewerConnectedInteractiveSchedulerReady"],
        )
        is True
        and get_path(compositor_contract, ["viewerTimeStateConnectedToRuntime"])
        is True
        and get_path(
            compositor_contract,
            ["playbackOrTimeSliderDrivesDirtyTimeState"],
        )
        is True
        and get_path(compositor_contract, ["rafSchedulerInvokesProductionRuntime"])
        is True
        and scheduler_frame_count > 0
        and time_control_evidence_ready
        and get_path(compositor_contract, ["timeStateChangedByViewerControl"])
        is True
        and get_path(
            compositor_contract,
            ["dirtyTimeStateTriggeredProductionUpdate"],
        )
        is True
        and get_path(
            compositor_contract,
            ["productionRuntimeUpdatedFromViewerScheduler"],
        )
        is True
        and get_path(compositor_contract, ["cleanFrameReuseUnderScheduler"])
        is True
        and get_path(
            compositor_contract,
            ["lastValidProductionOutputPresentedByScheduler"],
        )
        is True
        and step97_multi_frame_preserved
        and step96_production_preserved
        and step94_parallel_sort_preserved
        and step88_presentation_contract_preserved
        and get_path(compositor_contract, ["fallbackOnlyCompositorUsed"]) is False
        and get_path(compositor_contract, ["debugOutputBypassedForProduction"])
        is True
        and get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        )
        is True
        and get_path(compositor_contract, ["visualOutputDegeneratedDetected"])
        is not True
        and wgsl_parse_error_detected is False
        and shader_module_invalid_detected is False
        and compute_pipeline_invalid_detected is False
        and bind_group_invalid_detected is False
        and get_path(frame_contract, ["webgpuValidationErrorDetected"]) is False
        and get_path(frame_contract, ["invalidCommandBufferDetected"]) is False
        and get_path(frame_contract, ["queueSubmitFailureDetected"]) is False
        and get_path(frame_contract, ["webgpuWebgl2SameFramePresentationMixed"])
        is False
        and get_path(frame_contract, ["fallbackMixingPrevented"]) is True
        and get_path(frame_contract, ["fullRendererSuccessClaimed"]) is False
    )
    blocked_reason = None
    if not scheduler_ready:
        if phase_step != "phase3-step98":
            blocked_reason = "summary-phase-step-is-not-phase3-step98"
        elif get_path(
            compositor_contract,
            ["viewerConnectedInteractiveSchedulerReady"],
        ) is not True:
            blocked_reason = "viewer-connected-interactive-scheduler-not-ready"
        elif get_path(compositor_contract, ["viewerTimeStateConnectedToRuntime"]) is not True:
            blocked_reason = "viewer-time-state-not-connected-to-runtime"
        elif get_path(compositor_contract, ["rafSchedulerInvokesProductionRuntime"]) is not True:
            blocked_reason = "raf-scheduler-does-not-invoke-production-runtime"
        elif scheduler_frame_count <= 0:
            blocked_reason = "scheduler-frame-count-too-low"
        elif not time_control_evidence_ready:
            blocked_reason = "time-control-evidence-not-ready"
        elif get_path(
            compositor_contract,
            ["dirtyTimeStateTriggeredProductionUpdate"],
        ) is not True:
            blocked_reason = "dirty-time-state-did-not-trigger-production-update"
        elif get_path(compositor_contract, ["cleanFrameReuseUnderScheduler"]) is not True:
            blocked_reason = "clean-frame-reuse-under-scheduler-not-used"
        elif not step97_multi_frame_preserved:
            blocked_reason = "step97-multi-frame-runtime-not-preserved"
        elif not step96_production_preserved:
            blocked_reason = "step96-production-tile-compositor-not-preserved"
        elif not step94_parallel_sort_preserved:
            blocked_reason = "step94-parallel-sort-not-preserved"
        elif not step88_presentation_contract_preserved:
            blocked_reason = "step88-presentation-contract-not-preserved"
        elif wgsl_parse_error_detected:
            blocked_reason = "wgsl-parse-error-detected"
        elif shader_module_invalid_detected:
            blocked_reason = "shader-module-invalid-detected"
        elif compute_pipeline_invalid_detected:
            blocked_reason = "compute-pipeline-invalid-detected"
        elif bind_group_invalid_detected:
            blocked_reason = "bind-group-invalid-detected"
        elif get_path(frame_contract, ["webgpuValidationErrorDetected"]) is True:
            blocked_reason = "webgpu-validation-error-detected"
        else:
            blocked_reason = "step98-scheduler-validation-failed"
    return {
        "step98Decision": "success" if scheduler_ready else "blocked",
        "step98BlockedReason": blocked_reason,
        "step98SelectedGoal":
            "A+B+C+D-viewer-connected-interactive-time-scheduler-v1",
        "phaseStep": phase_step,
        "step98SummaryApplies": phase_step == "phase3-step98",
        "viewerConnectedInteractiveSchedulerReady": get_path(
            compositor_contract,
            ["viewerConnectedInteractiveSchedulerReady"],
        ),
        "viewerTimeStateConnectedToRuntime": get_path(
            compositor_contract,
            ["viewerTimeStateConnectedToRuntime"],
        ),
        "playbackOrTimeSliderDrivesDirtyTimeState": get_path(
            compositor_contract,
            ["playbackOrTimeSliderDrivesDirtyTimeState"],
        ),
        "rafSchedulerInvokesProductionRuntime": get_path(
            compositor_contract,
            ["rafSchedulerInvokesProductionRuntime"],
        ),
        "schedulerFrameCount": get_path(
            compositor_contract,
            ["schedulerFrameCount"],
        ),
        "timeControlEvidenceReady": get_path(
            compositor_contract,
            ["timeControlEvidenceReady"],
        ),
        "timeControlEvidenceSource": get_path(
            compositor_contract,
            ["timeControlEvidenceSource"],
        ),
        "viewerTimeBefore": viewer_time_before,
        "viewerTimeAfter": viewer_time_after,
        "viewerTimeDelta": viewer_time_delta,
        "timeControlEvidenceFromSchedulerProbe": get_path(
            compositor_contract,
            ["timeControlEvidenceFromSchedulerProbe"],
        ),
        "timeControlEvidenceUsesFixedValue": get_path(
            compositor_contract,
            ["timeControlEvidenceUsesFixedValue"],
        ),
        "timeStateChangedByViewerControl": get_path(
            compositor_contract,
            ["timeStateChangedByViewerControl"],
        ),
        "dirtyTimeStateTriggeredProductionUpdate": get_path(
            compositor_contract,
            ["dirtyTimeStateTriggeredProductionUpdate"],
        ),
        "productionRuntimeUpdatedFromViewerScheduler": get_path(
            compositor_contract,
            ["productionRuntimeUpdatedFromViewerScheduler"],
        ),
        "cleanFrameReuseUnderScheduler": get_path(
            compositor_contract,
            ["cleanFrameReuseUnderScheduler"],
        ),
        "lastValidProductionOutputPresentedByScheduler": get_path(
            compositor_contract,
            ["lastValidProductionOutputPresentedByScheduler"],
        ),
        "step97MultiFrameRuntimePreserved": step97_multi_frame_preserved,
        "step96ProductionTileCompositorPreserved": step96_production_preserved,
        "step94ParallelSortPreserved": step94_parallel_sort_preserved,
        "step93OverflowPolicyPreserved": get_path(
            compositor_contract,
            ["step93OverflowPolicyPreserved"],
        ),
        "step90RuntimePathPreserved": get_path(
            compositor_contract,
            ["step90RuntimePathPreserved"],
        ),
        "step88PresentationContractPreserved":
            step88_presentation_contract_preserved,
        "step85TileCompositorPathPreserved": get_path(
            frame_contract,
            ["step85TileCompositorPathPreserved"],
        ),
        "step86BoundaryContractPreserved": get_path(
            frame_contract,
            ["step86BoundaryContractPreserved"],
        ),
        "step87DepthOrderingPreserved": get_path(
            frame_contract,
            ["step87DepthOrderingPreserved"],
        ),
        "fallbackOnlyCompositorUsed": get_path(
            compositor_contract,
            ["fallbackOnlyCompositorUsed"],
        ),
        "debugOutputBypassedForProduction": get_path(
            compositor_contract,
            ["debugOutputBypassedForProduction"],
        ),
        "diagnosticReadbackSeparatedFromRuntimePath": get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        ),
        "visualOutputDegeneratedDetected": get_path(
            compositor_contract,
            ["visualOutputDegeneratedDetected"],
        ),
        "wgslParseErrorDetected": wgsl_parse_error_detected,
        "shaderModuleInvalidDetected": shader_module_invalid_detected,
        "computePipelineInvalidDetected": compute_pipeline_invalid_detected,
        "bindGroupInvalidDetected": bind_group_invalid_detected,
        "webgpuValidationErrorDetected": get_path(
            frame_contract,
            ["webgpuValidationErrorDetected"],
        ),
        "invalidCommandBufferDetected": get_path(
            frame_contract,
            ["invalidCommandBufferDetected"],
        ),
        "queueSubmitFailureDetected": get_path(
            frame_contract,
            ["queueSubmitFailureDetected"],
        ),
        "deferredProductionItems": deferred_production_items,
        "fullRendererSuccessClaimed": get_path(
            frame_contract,
            ["fullRendererSuccessClaimed"],
        ),
    }


def build_step99_interactive_camera_dirty_runtime_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    frame_contract = get_path(
        summary,
        ["webgpuTileCompositorFrameImplementation"],
        {},
    )
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    error_subtypes = detect_webgpu_error_subtypes(
        frame_contract,
        compositor_contract,
        get_path(summary, ["captureErrorString"]),
        get_path(summary, ["captureErrorStack"]),
        get_path(summary, ["captureErrorMessage"]),
        get_path(summary, ["firstValidationFailures"]),
    )
    phase_step = get_path(summary, ["phaseStep"])
    wgsl_parse_error_detected = (
        get_path(frame_contract, ["wgslParseErrorDetected"]) is True
        or get_path(compositor_contract, ["wgslParseErrorDetected"]) is True
        or error_subtypes["wgslParseErrorDetected"]
    )
    shader_module_invalid_detected = (
        get_path(frame_contract, ["shaderModuleInvalidDetected"]) is True
        or get_path(compositor_contract, ["shaderModuleInvalidDetected"]) is True
        or error_subtypes["shaderModuleInvalidDetected"]
    )
    compute_pipeline_invalid_detected = (
        get_path(frame_contract, ["computePipelineInvalidDetected"]) is True
        or get_path(compositor_contract, ["computePipelineInvalidDetected"]) is True
        or error_subtypes["computePipelineInvalidDetected"]
    )
    bind_group_invalid_detected = (
        get_path(frame_contract, ["bindGroupInvalidDetected"]) is True
        or get_path(compositor_contract, ["bindGroupInvalidDetected"]) is True
        or error_subtypes["bindGroupInvalidDetected"]
    )
    deferred_production_items = get_path(
        compositor_contract,
        ["deferredProductionItems"],
        [],
    )
    if not isinstance(deferred_production_items, list):
        deferred_production_items = []
    else:
        deferred_production_items = list(deferred_production_items)
    for item in [
        "full-cuda-parity",
        "final-production-compositor",
        "full-parallel-sort-parity",
        "complete-interactive-control-parity",
        "camera-visual-parity",
        "chunk-lod-streaming",
    ]:
        if item not in deferred_production_items:
            deferred_production_items.append(item)

    camera_delta = numeric_value(
        get_path(compositor_contract, ["cameraConstantsMaxAbsDelta"]),
        0,
    )
    camera_evidence_ready = (
        get_path(compositor_contract, ["cameraControlEvidenceReady"]) is True
        and camera_delta > 0
        and get_path(
            compositor_contract,
            ["cameraControlEvidenceFromSchedulerProbe"],
        )
        is True
        and get_path(
            compositor_contract,
            ["cameraControlEvidenceUsesFixedValue"],
        )
        is not True
    )
    step98_preserved = (
        get_path(compositor_contract, ["step98ViewerTimeSchedulerPreserved"])
        is True
        and get_path(
            compositor_contract,
            ["viewerConnectedInteractiveSchedulerReady"],
        )
        is True
    )
    step97_preserved = (
        get_path(compositor_contract, ["step97MultiFrameRuntimePreserved"])
        is True
        and get_path(compositor_contract, ["timeDrivenProductionRuntimeReady"])
        is True
    )
    step96_preserved = (
        get_path(compositor_contract, ["step96ProductionTileCompositorPreserved"])
        is True
        and get_path(compositor_contract, ["productionTileCompositorReady"]) is True
    )
    step94_preserved = (
        get_path(compositor_contract, ["step94ParallelSortPreserved"]) is True
        and get_path(compositor_contract, ["gpuParallelPerTileSortReady"]) is True
    )
    step88_preserved = (
        get_path(compositor_contract, ["step88PresentationContractPreserved"]) is True
        or get_path(
            frame_contract,
            ["steadyStateTileCompositorOwnsFinalPresentation"],
        )
        is True
    )
    camera_runtime_ready = (
        phase_step == "phase3-step99"
        and get_path(
            compositor_contract,
            ["interactiveCameraDirtyRuntimeReady"],
        )
        is True
        and get_path(
            compositor_contract,
            ["phase2CameraContractAssumptionsAdopted"],
        )
        is True
        and get_path(
            compositor_contract,
            ["fixedReferenceAndInteractiveCameraSeparated"],
        )
        is True
        and get_path(compositor_contract, ["threeJsCameraAdapterOnly"]) is True
        and get_path(
            compositor_contract,
            ["cudaReferenceNotInteractiveBackend"],
        )
        is True
        and get_path(
            compositor_contract,
            ["viewerCameraStateConnectedToRuntime"],
        )
        is True
        and get_path(compositor_contract, ["viewerCameraStateChangedByProbe"])
        is True
        and get_path(compositor_contract, ["cameraConstantsChanged"]) is True
        and camera_evidence_ready
        and get_path(
            compositor_contract,
            ["dirtyCameraConstantsTriggeredProductionUpdate"],
        )
        is True
        and get_path(
            compositor_contract,
            ["productionRuntimeUpdatedFromViewerCameraScheduler"],
        )
        is True
        and get_path(compositor_contract, ["cleanFrameReuseAfterCameraStabilized"])
        is True
        and get_path(
            compositor_contract,
            ["lastValidProductionOutputPresentedAfterCameraCleanFrame"],
        )
        is True
        and step98_preserved
        and step97_preserved
        and step96_preserved
        and step94_preserved
        and step88_preserved
        and get_path(compositor_contract, ["fallbackOnlyCompositorUsed"]) is False
        and get_path(compositor_contract, ["debugOutputBypassedForProduction"])
        is True
        and get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        )
        is True
        and get_path(compositor_contract, ["visualOutputDegeneratedDetected"])
        is not True
        and wgsl_parse_error_detected is False
        and shader_module_invalid_detected is False
        and compute_pipeline_invalid_detected is False
        and bind_group_invalid_detected is False
        and get_path(frame_contract, ["webgpuValidationErrorDetected"]) is False
        and get_path(frame_contract, ["invalidCommandBufferDetected"]) is False
        and get_path(frame_contract, ["queueSubmitFailureDetected"]) is False
        and get_path(frame_contract, ["webgpuWebgl2SameFramePresentationMixed"])
        is False
        and get_path(frame_contract, ["fallbackMixingPrevented"]) is True
        and get_path(frame_contract, ["fullRendererSuccessClaimed"]) is False
    )
    blocked_reason = None
    if not camera_runtime_ready:
        if phase_step != "phase3-step99":
            blocked_reason = "summary-phase-step-is-not-phase3-step99"
        elif get_path(
            compositor_contract,
            ["interactiveCameraDirtyRuntimeReady"],
        ) is not True:
            blocked_reason = "interactive-camera-dirty-runtime-not-ready"
        elif not camera_evidence_ready:
            blocked_reason = "camera-control-evidence-not-ready"
        elif get_path(
            compositor_contract,
            ["dirtyCameraConstantsTriggeredProductionUpdate"],
        ) is not True:
            blocked_reason = "dirty-camera-constants-did-not-trigger-production-update"
        elif not step98_preserved:
            blocked_reason = "step98-viewer-time-scheduler-not-preserved"
        elif not step97_preserved:
            blocked_reason = "step97-multi-frame-runtime-not-preserved"
        elif not step96_preserved:
            blocked_reason = "step96-production-tile-compositor-not-preserved"
        elif not step94_preserved:
            blocked_reason = "step94-parallel-sort-not-preserved"
        elif not step88_preserved:
            blocked_reason = "step88-presentation-contract-not-preserved"
        elif wgsl_parse_error_detected:
            blocked_reason = "wgsl-parse-error-detected"
        elif shader_module_invalid_detected:
            blocked_reason = "shader-module-invalid-detected"
        elif compute_pipeline_invalid_detected:
            blocked_reason = "compute-pipeline-invalid-detected"
        elif bind_group_invalid_detected:
            blocked_reason = "bind-group-invalid-detected"
        else:
            blocked_reason = "step99-camera-dirty-runtime-validation-failed"
    return {
        "step99Decision": "success" if camera_runtime_ready else "blocked",
        "step99BlockedReason": blocked_reason,
        "step99SelectedGoal":
            "A+B+D+E+F-interactive-camera-viewport-dirty-runtime-v1",
        "phaseStep": phase_step,
        "step99SummaryApplies": phase_step == "phase3-step99",
        "phase2CameraContractAssumptionsAdopted": get_path(
            compositor_contract,
            ["phase2CameraContractAssumptionsAdopted"],
        ),
        "phase3ResponsibilityPlanReferenced": get_path(
            compositor_contract,
            ["phase3ResponsibilityPlanReferenced"],
        ),
        "phase3BackendDesignReferenced": get_path(
            compositor_contract,
            ["phase3BackendDesignReferenced"],
        ),
        "fixedReferenceAndInteractiveCameraSeparated": get_path(
            compositor_contract,
            ["fixedReferenceAndInteractiveCameraSeparated"],
        ),
        "threeJsCameraAdapterOnly": get_path(
            compositor_contract,
            ["threeJsCameraAdapterOnly"],
        ),
        "cudaReferenceNotInteractiveBackend": get_path(
            compositor_contract,
            ["cudaReferenceNotInteractiveBackend"],
        ),
        "viewerCameraStateConnectedToRuntime": get_path(
            compositor_contract,
            ["viewerCameraStateConnectedToRuntime"],
        ),
        "viewerCameraStateChangedByProbe": get_path(
            compositor_contract,
            ["viewerCameraStateChangedByProbe"],
        ),
        "cameraConstantsChanged": get_path(
            compositor_contract,
            ["cameraConstantsChanged"],
        ),
        "dirtyCameraConstantsTriggeredProductionUpdate": get_path(
            compositor_contract,
            ["dirtyCameraConstantsTriggeredProductionUpdate"],
        ),
        "viewportStateConnectedToRuntime": get_path(
            compositor_contract,
            ["viewportStateConnectedToRuntime"],
        ),
        "dirtyViewportTriggeredProductionUpdate": get_path(
            compositor_contract,
            ["dirtyViewportTriggeredProductionUpdate"],
        ),
        "dirtyViewportDeferredReason": get_path(
            compositor_contract,
            ["dirtyViewportDeferredReason"],
        ),
        "productionRuntimeUpdatedFromViewerCameraScheduler": get_path(
            compositor_contract,
            ["productionRuntimeUpdatedFromViewerCameraScheduler"],
        ),
        "cleanFrameReuseAfterCameraStabilized": get_path(
            compositor_contract,
            ["cleanFrameReuseAfterCameraStabilized"],
        ),
        "lastValidProductionOutputPresentedAfterCameraCleanFrame": get_path(
            compositor_contract,
            ["lastValidProductionOutputPresentedAfterCameraCleanFrame"],
        ),
        "cameraControlEvidenceReady": get_path(
            compositor_contract,
            ["cameraControlEvidenceReady"],
        ),
        "cameraControlEvidenceSource": get_path(
            compositor_contract,
            ["cameraControlEvidenceSource"],
        ),
        "cameraControlEvidenceFromSchedulerProbe": get_path(
            compositor_contract,
            ["cameraControlEvidenceFromSchedulerProbe"],
        ),
        "cameraControlEvidenceUsesFixedValue": get_path(
            compositor_contract,
            ["cameraControlEvidenceUsesFixedValue"],
        ),
        "cameraPositionBefore": get_path(
            compositor_contract,
            ["cameraPositionBefore"],
        ),
        "cameraPositionAfter": get_path(
            compositor_contract,
            ["cameraPositionAfter"],
        ),
        "cameraQuaternionBefore": get_path(
            compositor_contract,
            ["cameraQuaternionBefore"],
        ),
        "cameraQuaternionAfter": get_path(
            compositor_contract,
            ["cameraQuaternionAfter"],
        ),
        "cameraConstantsMaxAbsDelta": get_path(
            compositor_contract,
            ["cameraConstantsMaxAbsDelta"],
        ),
        "viewportBefore": get_path(compositor_contract, ["viewportBefore"]),
        "viewportAfter": get_path(compositor_contract, ["viewportAfter"]),
        "viewportChangedByProbe": get_path(
            compositor_contract,
            ["viewportChangedByProbe"],
        ),
        "step98ViewerTimeSchedulerPreserved": step98_preserved,
        "step97MultiFrameRuntimePreserved": step97_preserved,
        "step96ProductionTileCompositorPreserved": step96_preserved,
        "step94ParallelSortPreserved": step94_preserved,
        "step88PresentationContractPreserved": step88_preserved,
        "step85TileCompositorPathPreserved": get_path(
            frame_contract,
            ["step85TileCompositorPathPreserved"],
        ),
        "step86BoundaryContractPreserved": get_path(
            frame_contract,
            ["step86BoundaryContractPreserved"],
        ),
        "step87DepthOrderingPreserved": get_path(
            frame_contract,
            ["step87DepthOrderingPreserved"],
        ),
        "fallbackOnlyCompositorUsed": get_path(
            compositor_contract,
            ["fallbackOnlyCompositorUsed"],
        ),
        "debugOutputBypassedForProduction": get_path(
            compositor_contract,
            ["debugOutputBypassedForProduction"],
        ),
        "diagnosticReadbackSeparatedFromRuntimePath": get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        ),
        "visualOutputDegeneratedDetected": get_path(
            compositor_contract,
            ["visualOutputDegeneratedDetected"],
        ),
        "wgslParseErrorDetected": wgsl_parse_error_detected,
        "shaderModuleInvalidDetected": shader_module_invalid_detected,
        "computePipelineInvalidDetected": compute_pipeline_invalid_detected,
        "bindGroupInvalidDetected": bind_group_invalid_detected,
        "webgpuValidationErrorDetected": get_path(
            frame_contract,
            ["webgpuValidationErrorDetected"],
        ),
        "invalidCommandBufferDetected": get_path(
            frame_contract,
            ["invalidCommandBufferDetected"],
        ),
        "queueSubmitFailureDetected": get_path(
            frame_contract,
            ["queueSubmitFailureDetected"],
        ),
        "deferredProductionItems": deferred_production_items,
        "fullRendererSuccessClaimed": get_path(
            frame_contract,
            ["fullRendererSuccessClaimed"],
        ),
    }


def build_step100_unified_interaction_scheduler_runtime_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    frame_contract = get_path(
        summary,
        ["webgpuTileCompositorFrameImplementation"],
        {},
    )
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    error_subtypes = detect_webgpu_error_subtypes(
        frame_contract,
        compositor_contract,
        get_path(summary, ["captureErrorString"]),
        get_path(summary, ["captureErrorStack"]),
        get_path(summary, ["captureErrorMessage"]),
        get_path(summary, ["firstValidationFailures"]),
    )
    phase_step = get_path(summary, ["phaseStep"])
    wgsl_parse_error_detected = (
        get_path(frame_contract, ["wgslParseErrorDetected"]) is True
        or get_path(compositor_contract, ["wgslParseErrorDetected"]) is True
        or error_subtypes["wgslParseErrorDetected"]
    )
    shader_module_invalid_detected = (
        get_path(frame_contract, ["shaderModuleInvalidDetected"]) is True
        or get_path(compositor_contract, ["shaderModuleInvalidDetected"]) is True
        or error_subtypes["shaderModuleInvalidDetected"]
    )
    compute_pipeline_invalid_detected = (
        get_path(frame_contract, ["computePipelineInvalidDetected"]) is True
        or get_path(compositor_contract, ["computePipelineInvalidDetected"]) is True
        or error_subtypes["computePipelineInvalidDetected"]
    )
    bind_group_invalid_detected = (
        get_path(frame_contract, ["bindGroupInvalidDetected"]) is True
        or get_path(compositor_contract, ["bindGroupInvalidDetected"]) is True
        or error_subtypes["bindGroupInvalidDetected"]
    )
    deferred_production_items = get_path(
        compositor_contract,
        ["deferredProductionItems"],
        [],
    )
    if not isinstance(deferred_production_items, list):
        deferred_production_items = []
    else:
        deferred_production_items = list(deferred_production_items)
    for item in [
        "full-cuda-parity",
        "final-production-compositor",
        "final-production-compositor-parity",
        "full-parallel-sort-parity",
        "complete-interactive-control-parity",
        "camera-visual-parity",
        "chunk-lod-streaming",
        "early-termination-v1",
    ]:
        if item not in deferred_production_items:
            deferred_production_items.append(item)

    dirty_frame_count = numeric_value(
        get_path(compositor_contract, ["dirtyFrameCount"]),
        0,
    )
    clean_frame_reuse_count = numeric_value(
        get_path(compositor_contract, ["cleanFrameReuseCount"]),
        0,
    )
    production_update_count = numeric_value(
        get_path(compositor_contract, ["productionUpdateCount"]),
        0,
    )
    step99_preserved = (
        get_path(compositor_contract, ["step99InteractiveCameraDirtyPreserved"])
        is True
        and get_path(
            compositor_contract,
            ["interactiveCameraDirtyRuntimeReady"],
        )
        is True
    )
    step98_preserved = (
        get_path(compositor_contract, ["step98ViewerTimeSchedulerPreserved"])
        is True
        and get_path(
            compositor_contract,
            ["viewerConnectedInteractiveSchedulerReady"],
        )
        is True
    )
    step97_preserved = (
        get_path(compositor_contract, ["step97MultiFrameRuntimePreserved"])
        is True
        and get_path(compositor_contract, ["timeDrivenProductionRuntimeReady"])
        is True
    )
    step96_preserved = (
        get_path(compositor_contract, ["step96ProductionTileCompositorPreserved"])
        is True
        and get_path(compositor_contract, ["productionTileCompositorReady"]) is True
    )
    step94_preserved = (
        get_path(compositor_contract, ["step94ParallelSortPreserved"]) is True
        and get_path(compositor_contract, ["gpuParallelPerTileSortReady"]) is True
    )
    step88_preserved = (
        get_path(compositor_contract, ["step88PresentationContractPreserved"]) is True
        or get_path(
            frame_contract,
            ["steadyStateTileCompositorOwnsFinalPresentation"],
        )
        is True
    )
    unified_ready = (
        phase_step == "phase3-step100"
        and get_path(compositor_contract, ["phase2CameraContractAssumptionsAdopted"])
        is True
        and get_path(
            compositor_contract,
            ["fixedReferenceAndInteractiveCameraSeparated"],
        )
        is True
        and get_path(compositor_contract, ["threeJsCameraAdapterOnly"]) is True
        and get_path(compositor_contract, ["cudaReferenceNotInteractiveBackend"])
        is True
        and get_path(compositor_contract, ["unifiedInteractionSchedulerReady"])
        is True
        and get_path(compositor_contract, ["timeAndCameraDirtyPathsUnified"])
        is True
        and get_path(
            compositor_contract,
            ["captureProbeRuntimeBoundarySeparated"],
        )
        is True
        and get_path(
            compositor_contract,
            ["captureProbeStimulatesViewerStateOnly"],
        )
        is True
        and get_path(
            compositor_contract,
            ["dirtyDependencyGraphConsumedByProductionRuntime"],
        )
        is True
        and get_path(
            compositor_contract,
            ["dirtyTimeStateTriggersUnifiedProductionUpdate"],
        )
        is True
        and get_path(
            compositor_contract,
            ["dirtyCameraConstantsTriggersUnifiedProductionUpdate"],
        )
        is True
        and isinstance(
            get_path(
                compositor_contract,
                ["dirtyViewportIntegratedOrDeferredReason"],
            ),
            str,
        )
        and get_path(
            compositor_contract,
            ["productionRuntimeUpdatedByUnifiedInteractionScheduler"],
        )
        is True
        and get_path(
            compositor_contract,
            ["cleanFrameReuseAfterUnifiedInteractionStabilized"],
        )
        is True
        and get_path(
            compositor_contract,
            ["lastValidProductionOutputPresentedAfterUnifiedCleanFrame"],
        )
        is True
        and get_path(
            compositor_contract,
            ["realtimeFrameBudgetTelemetryReady"],
        )
        is True
        and dirty_frame_count > 0
        and clean_frame_reuse_count > 0
        and production_update_count > 0
        and step99_preserved
        and step98_preserved
        and step97_preserved
        and step96_preserved
        and step94_preserved
        and step88_preserved
        and get_path(compositor_contract, ["fallbackOnlyCompositorUsed"]) is False
        and get_path(compositor_contract, ["debugOutputBypassedForProduction"])
        is True
        and get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        )
        is True
        and get_path(compositor_contract, ["visualOutputDegeneratedDetected"])
        is not True
        and wgsl_parse_error_detected is False
        and shader_module_invalid_detected is False
        and compute_pipeline_invalid_detected is False
        and bind_group_invalid_detected is False
        and get_path(frame_contract, ["webgpuValidationErrorDetected"]) is False
        and get_path(frame_contract, ["invalidCommandBufferDetected"]) is False
        and get_path(frame_contract, ["queueSubmitFailureDetected"]) is False
        and get_path(frame_contract, ["webgpuWebgl2SameFramePresentationMixed"])
        is False
        and get_path(frame_contract, ["fallbackMixingPrevented"]) is True
        and get_path(frame_contract, ["fullRendererSuccessClaimed"]) is False
    )
    blocked_reason = None
    if not unified_ready:
        if phase_step != "phase3-step100":
            blocked_reason = "summary-phase-step-is-not-phase3-step100"
        elif get_path(
            compositor_contract,
            ["unifiedInteractionSchedulerReady"],
        ) is not True:
            blocked_reason = "unified-interaction-scheduler-not-ready"
        elif get_path(
            compositor_contract,
            ["timeAndCameraDirtyPathsUnified"],
        ) is not True:
            blocked_reason = "time-and-camera-dirty-paths-not-unified"
        elif get_path(
            compositor_contract,
            ["dirtyDependencyGraphConsumedByProductionRuntime"],
        ) is not True:
            blocked_reason = "dirty-dependency-graph-not-consumed"
        elif get_path(
            compositor_contract,
            ["realtimeFrameBudgetTelemetryReady"],
        ) is not True:
            blocked_reason = "realtime-frame-budget-telemetry-not-ready"
        elif not step99_preserved:
            blocked_reason = "step99-interactive-camera-dirty-not-preserved"
        elif not step98_preserved:
            blocked_reason = "step98-viewer-time-scheduler-not-preserved"
        elif not step97_preserved:
            blocked_reason = "step97-multi-frame-runtime-not-preserved"
        elif not step96_preserved:
            blocked_reason = "step96-production-tile-compositor-not-preserved"
        elif not step94_preserved:
            blocked_reason = "step94-parallel-sort-not-preserved"
        elif not step88_preserved:
            blocked_reason = "step88-presentation-contract-not-preserved"
        elif wgsl_parse_error_detected:
            blocked_reason = "wgsl-parse-error-detected"
        elif shader_module_invalid_detected:
            blocked_reason = "shader-module-invalid-detected"
        elif compute_pipeline_invalid_detected:
            blocked_reason = "compute-pipeline-invalid-detected"
        elif bind_group_invalid_detected:
            blocked_reason = "bind-group-invalid-detected"
        else:
            blocked_reason = "step100-unified-scheduler-validation-failed"

    return {
        "step100Decision": "success" if unified_ready else "blocked",
        "step100BlockedReason": blocked_reason,
        "step100SelectedGoal":
            "A+B+D-unified-production-interaction-scheduler-runtime-v1",
        "phaseStep": phase_step,
        "step100SummaryApplies": phase_step == "phase3-step100",
        "phase2CameraContractAssumptionsPreserved": get_path(
            compositor_contract,
            ["phase2CameraContractAssumptionsAdopted"],
        ),
        "fixedReferenceAndInteractiveCameraSeparated": get_path(
            compositor_contract,
            ["fixedReferenceAndInteractiveCameraSeparated"],
        ),
        "threeJsCameraAdapterOnly": get_path(
            compositor_contract,
            ["threeJsCameraAdapterOnly"],
        ),
        "cudaReferenceNotInteractiveBackend": get_path(
            compositor_contract,
            ["cudaReferenceNotInteractiveBackend"],
        ),
        "unifiedInteractionSchedulerReady": get_path(
            compositor_contract,
            ["unifiedInteractionSchedulerReady"],
        ),
        "timeAndCameraDirtyPathsUnified": get_path(
            compositor_contract,
            ["timeAndCameraDirtyPathsUnified"],
        ),
        "captureProbeRuntimeBoundarySeparated": get_path(
            compositor_contract,
            ["captureProbeRuntimeBoundarySeparated"],
        ),
        "captureProbeStimulatesViewerStateOnly": get_path(
            compositor_contract,
            ["captureProbeStimulatesViewerStateOnly"],
        ),
        "dirtyDependencyGraphConsumedByProductionRuntime": get_path(
            compositor_contract,
            ["dirtyDependencyGraphConsumedByProductionRuntime"],
        ),
        "dirtyTimeStateTriggersUnifiedProductionUpdate": get_path(
            compositor_contract,
            ["dirtyTimeStateTriggersUnifiedProductionUpdate"],
        ),
        "dirtyCameraConstantsTriggersUnifiedProductionUpdate": get_path(
            compositor_contract,
            ["dirtyCameraConstantsTriggersUnifiedProductionUpdate"],
        ),
        "dirtyViewportIntegratedOrDeferredReason": get_path(
            compositor_contract,
            ["dirtyViewportIntegratedOrDeferredReason"],
        ),
        "productionRuntimeUpdatedByUnifiedInteractionScheduler": get_path(
            compositor_contract,
            ["productionRuntimeUpdatedByUnifiedInteractionScheduler"],
        ),
        "cleanFrameReuseAfterUnifiedInteractionStabilized": get_path(
            compositor_contract,
            ["cleanFrameReuseAfterUnifiedInteractionStabilized"],
        ),
        "lastValidProductionOutputPresentedAfterUnifiedCleanFrame": get_path(
            compositor_contract,
            ["lastValidProductionOutputPresentedAfterUnifiedCleanFrame"],
        ),
        "realtimeFrameBudgetTelemetryReady": get_path(
            compositor_contract,
            ["realtimeFrameBudgetTelemetryReady"],
        ),
        "dirtyFrameCount": get_path(compositor_contract, ["dirtyFrameCount"]),
        "cleanFrameReuseCount": get_path(
            compositor_contract,
            ["cleanFrameReuseCount"],
        ),
        "productionUpdateCount": get_path(
            compositor_contract,
            ["productionUpdateCount"],
        ),
        "step99InteractiveCameraDirtyPreserved": step99_preserved,
        "step98ViewerTimeSchedulerPreserved": step98_preserved,
        "step97MultiFrameRuntimePreserved": step97_preserved,
        "step96ProductionTileCompositorPreserved": step96_preserved,
        "step94ParallelSortPreserved": step94_preserved,
        "step88PresentationContractPreserved": step88_preserved,
        "step85TileCompositorPathPreserved": get_path(
            frame_contract,
            ["step85TileCompositorPathPreserved"],
        ),
        "step86BoundaryContractPreserved": get_path(
            frame_contract,
            ["step86BoundaryContractPreserved"],
        ),
        "step87DepthOrderingPreserved": get_path(
            frame_contract,
            ["step87DepthOrderingPreserved"],
        ),
        "fallbackOnlyCompositorUsed": get_path(
            compositor_contract,
            ["fallbackOnlyCompositorUsed"],
        ),
        "debugOutputBypassedForProduction": get_path(
            compositor_contract,
            ["debugOutputBypassedForProduction"],
        ),
        "diagnosticReadbackSeparatedFromRuntimePath": get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        ),
        "visualOutputDegeneratedDetected": get_path(
            compositor_contract,
            ["visualOutputDegeneratedDetected"],
        ),
        "wgslParseErrorDetected": wgsl_parse_error_detected,
        "shaderModuleInvalidDetected": shader_module_invalid_detected,
        "computePipelineInvalidDetected": compute_pipeline_invalid_detected,
        "bindGroupInvalidDetected": bind_group_invalid_detected,
        "webgpuValidationErrorDetected": get_path(
            frame_contract,
            ["webgpuValidationErrorDetected"],
        ),
        "invalidCommandBufferDetected": get_path(
            frame_contract,
            ["invalidCommandBufferDetected"],
        ),
        "queueSubmitFailureDetected": get_path(
            frame_contract,
            ["queueSubmitFailureDetected"],
        ),
        "deferredProductionItems": deferred_production_items,
        "fullRendererSuccessClaimed": get_path(
            frame_contract,
            ["fullRendererSuccessClaimed"],
        ),
    }


def build_step101_selective_dirty_dependency_execution_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    frame_contract = get_path(
        summary,
        ["webgpuTileCompositorFrameImplementation"],
        {},
    )
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    error_subtypes = detect_webgpu_error_subtypes(
        frame_contract,
        compositor_contract,
        get_path(summary, ["captureErrorString"]),
        get_path(summary, ["captureErrorStack"]),
        get_path(summary, ["captureErrorMessage"]),
        get_path(summary, ["firstValidationFailures"]),
    )
    phase_step = get_path(summary, ["phaseStep"])
    wgsl_parse_error_detected = (
        get_path(frame_contract, ["wgslParseErrorDetected"]) is True
        or get_path(compositor_contract, ["wgslParseErrorDetected"]) is True
        or error_subtypes["wgslParseErrorDetected"]
    )
    shader_module_invalid_detected = (
        get_path(frame_contract, ["shaderModuleInvalidDetected"]) is True
        or get_path(compositor_contract, ["shaderModuleInvalidDetected"]) is True
        or error_subtypes["shaderModuleInvalidDetected"]
    )
    compute_pipeline_invalid_detected = (
        get_path(frame_contract, ["computePipelineInvalidDetected"]) is True
        or get_path(compositor_contract, ["computePipelineInvalidDetected"]) is True
        or error_subtypes["computePipelineInvalidDetected"]
    )
    bind_group_invalid_detected = (
        get_path(frame_contract, ["bindGroupInvalidDetected"]) is True
        or get_path(compositor_contract, ["bindGroupInvalidDetected"]) is True
        or error_subtypes["bindGroupInvalidDetected"]
    )
    deferred_production_items = get_path(
        compositor_contract,
        ["deferredProductionItems"],
        [],
    )
    if not isinstance(deferred_production_items, list):
        deferred_production_items = []
    else:
        deferred_production_items = list(deferred_production_items)
    for item in [
        "full-cuda-parity",
        "final-production-compositor",
        "final-production-compositor-parity",
        "full-parallel-sort-parity",
        "complete-interactive-control-parity",
        "camera-visual-parity",
        "viewport-resize-dirty-probe",
        "chunk-lod-streaming",
        "early-termination-v1",
    ]:
        if item not in deferred_production_items:
            deferred_production_items.append(item)

    dirty_reason_sequence = get_path(
        compositor_contract,
        ["dirtyReasonSequence"],
        [],
    )
    if not isinstance(dirty_reason_sequence, list):
        dirty_reason_sequence = []
    dirty_frame_count = numeric_value(
        get_path(compositor_contract, ["dirtyFrameCount"]),
        0,
    )
    clean_frame_reuse_count = numeric_value(
        get_path(compositor_contract, ["cleanFrameReuseCount"]),
        0,
    )
    production_update_count = numeric_value(
        get_path(compositor_contract, ["productionUpdateCount"]),
        0,
    )
    skipped_stage_count = numeric_value(
        get_path(compositor_contract, ["skippedStageCount"]),
        0,
    )
    reused_resource_count = numeric_value(
        get_path(compositor_contract, ["reusedResourceCount"]),
        0,
    )
    step100_preserved = (
        get_path(
            compositor_contract,
            ["step100UnifiedInteractionSchedulerPreserved"],
        )
        is True
        and get_path(compositor_contract, ["unifiedInteractionSchedulerReady"])
        is True
    )
    step99_preserved = (
        get_path(compositor_contract, ["step99InteractiveCameraDirtyPreserved"])
        is True
        and get_path(
            compositor_contract,
            ["interactiveCameraDirtyRuntimeReady"],
        )
        is True
    )
    step98_preserved = (
        get_path(compositor_contract, ["step98ViewerTimeSchedulerPreserved"])
        is True
        and get_path(
            compositor_contract,
            ["viewerConnectedInteractiveSchedulerReady"],
        )
        is True
    )
    step97_preserved = (
        get_path(compositor_contract, ["step97MultiFrameRuntimePreserved"])
        is True
        and get_path(compositor_contract, ["timeDrivenProductionRuntimeReady"])
        is True
    )
    step96_preserved = (
        get_path(compositor_contract, ["step96ProductionTileCompositorPreserved"])
        is True
        and get_path(compositor_contract, ["productionTileCompositorReady"]) is True
    )
    step94_preserved = (
        get_path(compositor_contract, ["step94ParallelSortPreserved"]) is True
        and get_path(compositor_contract, ["gpuParallelPerTileSortReady"]) is True
    )
    step88_preserved = (
        get_path(compositor_contract, ["step88PresentationContractPreserved"]) is True
        or get_path(
            frame_contract,
            ["steadyStateTileCompositorOwnsFinalPresentation"],
        )
        is True
    )
    selective_ready = (
        phase_step == "phase3-step101"
        and get_path(
            compositor_contract,
            ["selectiveDirtyDependencyExecutionReady"],
        )
        is True
        and get_path(compositor_contract, ["dirtyReasonClassificationReady"])
        is True
        and "time-dirty" in dirty_reason_sequence
        and "camera-dirty" in dirty_reason_sequence
        and "clean-frame" in dirty_reason_sequence
        and get_path(compositor_contract, ["timeDirtyStagePlanReady"]) is True
        and get_path(compositor_contract, ["cameraDirtyStagePlanReady"]) is True
        and isinstance(
            get_path(
                compositor_contract,
                ["viewportDirtyIntegratedOrDeferredReason"],
            ),
            str,
        )
        and get_path(compositor_contract, ["cleanFrameStagePlanReady"]) is True
        and get_path(compositor_contract, ["selectiveStageInvalidationUsed"])
        is True
        and get_path(
            compositor_contract,
            ["productionRuntimeUpdatedBySelectiveExecutor"],
        )
        is True
        and get_path(
            compositor_contract,
            ["cleanFrameReuseAfterSelectiveExecution"],
        )
        is True
        and get_path(
            compositor_contract,
            ["lastValidProductionOutputPresentedAfterSelectiveCleanFrame"],
        )
        is True
        and get_path(
            compositor_contract,
            ["realtimeWorkloadBudgetTelemetryReady"],
        )
        is True
        and dirty_frame_count > 0
        and clean_frame_reuse_count > 0
        and production_update_count > 0
        and skipped_stage_count > 0
        and reused_resource_count > 0
        and step100_preserved
        and step99_preserved
        and step98_preserved
        and step97_preserved
        and step96_preserved
        and step94_preserved
        and step88_preserved
        and get_path(compositor_contract, ["fallbackOnlyCompositorUsed"]) is False
        and get_path(compositor_contract, ["debugOutputBypassedForProduction"])
        is True
        and get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        )
        is True
        and get_path(compositor_contract, ["visualOutputDegeneratedDetected"])
        is not True
        and wgsl_parse_error_detected is False
        and shader_module_invalid_detected is False
        and compute_pipeline_invalid_detected is False
        and bind_group_invalid_detected is False
        and get_path(frame_contract, ["webgpuValidationErrorDetected"]) is False
        and get_path(frame_contract, ["invalidCommandBufferDetected"]) is False
        and get_path(frame_contract, ["queueSubmitFailureDetected"]) is False
        and get_path(frame_contract, ["webgpuWebgl2SameFramePresentationMixed"])
        is False
        and get_path(frame_contract, ["fallbackMixingPrevented"]) is True
        and get_path(frame_contract, ["fullRendererSuccessClaimed"]) is False
    )
    blocked_reason = None
    if not selective_ready:
        if phase_step != "phase3-step101":
            blocked_reason = "summary-phase-step-is-not-phase3-step101"
        elif get_path(
            compositor_contract,
            ["selectiveDirtyDependencyExecutionReady"],
        ) is not True:
            blocked_reason = "selective-dirty-dependency-execution-not-ready"
        elif get_path(compositor_contract, ["dirtyReasonClassificationReady"]) is not True:
            blocked_reason = "dirty-reason-classification-not-ready"
        elif get_path(compositor_contract, ["selectiveStageInvalidationUsed"]) is not True:
            blocked_reason = "selective-stage-invalidation-not-used"
        elif get_path(
            compositor_contract,
            ["realtimeWorkloadBudgetTelemetryReady"],
        ) is not True:
            blocked_reason = "realtime-workload-budget-telemetry-not-ready"
        elif not step100_preserved:
            blocked_reason = "step100-unified-scheduler-not-preserved"
        elif not step99_preserved:
            blocked_reason = "step99-interactive-camera-dirty-not-preserved"
        elif not step98_preserved:
            blocked_reason = "step98-viewer-time-scheduler-not-preserved"
        elif not step97_preserved:
            blocked_reason = "step97-multi-frame-runtime-not-preserved"
        elif not step96_preserved:
            blocked_reason = "step96-production-tile-compositor-not-preserved"
        elif not step94_preserved:
            blocked_reason = "step94-parallel-sort-not-preserved"
        elif not step88_preserved:
            blocked_reason = "step88-presentation-contract-not-preserved"
        elif wgsl_parse_error_detected:
            blocked_reason = "wgsl-parse-error-detected"
        elif shader_module_invalid_detected:
            blocked_reason = "shader-module-invalid-detected"
        elif compute_pipeline_invalid_detected:
            blocked_reason = "compute-pipeline-invalid-detected"
        elif bind_group_invalid_detected:
            blocked_reason = "bind-group-invalid-detected"
        else:
            blocked_reason = "step101-selective-dirty-validation-failed"

    return {
        "step101Decision": "success" if selective_ready else "blocked",
        "step101BlockedReason": blocked_reason,
        "step101SelectedGoal":
            "A+B+C+D-selective-dirty-dependency-execution-runtime-v1",
        "phaseStep": phase_step,
        "step101SummaryApplies": phase_step == "phase3-step101",
        "selectiveDirtyDependencyExecutionReady": get_path(
            compositor_contract,
            ["selectiveDirtyDependencyExecutionReady"],
        ),
        "dirtyReasonClassificationReady": get_path(
            compositor_contract,
            ["dirtyReasonClassificationReady"],
        ),
        "dirtyReasonSequence": dirty_reason_sequence,
        "timeDirtyStagePlanReady": get_path(
            compositor_contract,
            ["timeDirtyStagePlanReady"],
        ),
        "cameraDirtyStagePlanReady": get_path(
            compositor_contract,
            ["cameraDirtyStagePlanReady"],
        ),
        "viewportDirtyIntegratedOrDeferredReason": get_path(
            compositor_contract,
            ["viewportDirtyIntegratedOrDeferredReason"],
        ),
        "cleanFrameStagePlanReady": get_path(
            compositor_contract,
            ["cleanFrameStagePlanReady"],
        ),
        "selectiveStageInvalidationUsed": get_path(
            compositor_contract,
            ["selectiveStageInvalidationUsed"],
        ),
        "updatedStageNamesByDirtyReason": get_path(
            compositor_contract,
            ["updatedStageNamesByDirtyReason"],
        ),
        "skippedStageNamesByDirtyReason": get_path(
            compositor_contract,
            ["skippedStageNamesByDirtyReason"],
        ),
        "reusedResourceNamesByDirtyReason": get_path(
            compositor_contract,
            ["reusedResourceNamesByDirtyReason"],
        ),
        "productionRuntimeUpdatedBySelectiveExecutor": get_path(
            compositor_contract,
            ["productionRuntimeUpdatedBySelectiveExecutor"],
        ),
        "cleanFrameReuseAfterSelectiveExecution": get_path(
            compositor_contract,
            ["cleanFrameReuseAfterSelectiveExecution"],
        ),
        "lastValidProductionOutputPresentedAfterSelectiveCleanFrame": get_path(
            compositor_contract,
            ["lastValidProductionOutputPresentedAfterSelectiveCleanFrame"],
        ),
        "realtimeWorkloadBudgetTelemetryReady": get_path(
            compositor_contract,
            ["realtimeWorkloadBudgetTelemetryReady"],
        ),
        "dirtyFrameCount": get_path(compositor_contract, ["dirtyFrameCount"]),
        "cleanFrameReuseCount": get_path(
            compositor_contract,
            ["cleanFrameReuseCount"],
        ),
        "productionUpdateCount": get_path(
            compositor_contract,
            ["productionUpdateCount"],
        ),
        "skippedStageCount": get_path(
            compositor_contract,
            ["skippedStageCount"],
        ),
        "reusedResourceCount": get_path(
            compositor_contract,
            ["reusedResourceCount"],
        ),
        "step100UnifiedInteractionSchedulerPreserved": step100_preserved,
        "step99InteractiveCameraDirtyPreserved": step99_preserved,
        "step98ViewerTimeSchedulerPreserved": step98_preserved,
        "step97MultiFrameRuntimePreserved": step97_preserved,
        "step96ProductionTileCompositorPreserved": step96_preserved,
        "step94ParallelSortPreserved": step94_preserved,
        "step88PresentationContractPreserved": step88_preserved,
        "step85TileCompositorPathPreserved": get_path(
            frame_contract,
            ["step85TileCompositorPathPreserved"],
        ),
        "step86BoundaryContractPreserved": get_path(
            frame_contract,
            ["step86BoundaryContractPreserved"],
        ),
        "step87DepthOrderingPreserved": get_path(
            frame_contract,
            ["step87DepthOrderingPreserved"],
        ),
        "fallbackOnlyCompositorUsed": get_path(
            compositor_contract,
            ["fallbackOnlyCompositorUsed"],
        ),
        "debugOutputBypassedForProduction": get_path(
            compositor_contract,
            ["debugOutputBypassedForProduction"],
        ),
        "diagnosticReadbackSeparatedFromRuntimePath": get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        ),
        "visualOutputDegeneratedDetected": get_path(
            compositor_contract,
            ["visualOutputDegeneratedDetected"],
        ),
        "wgslParseErrorDetected": wgsl_parse_error_detected,
        "shaderModuleInvalidDetected": shader_module_invalid_detected,
        "computePipelineInvalidDetected": compute_pipeline_invalid_detected,
        "bindGroupInvalidDetected": bind_group_invalid_detected,
        "webgpuValidationErrorDetected": get_path(
            frame_contract,
            ["webgpuValidationErrorDetected"],
        ),
        "invalidCommandBufferDetected": get_path(
            frame_contract,
            ["invalidCommandBufferDetected"],
        ),
        "queueSubmitFailureDetected": get_path(
            frame_contract,
            ["queueSubmitFailureDetected"],
        ),
        "deferredProductionItems": deferred_production_items,
        "fullRendererSuccessClaimed": get_path(
            frame_contract,
            ["fullRendererSuccessClaimed"],
        ),
    }


def build_step102_production_resource_lifecycle_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    frame_contract = get_path(
        summary,
        ["webgpuTileCompositorFrameImplementation"],
        {},
    )
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    error_subtypes = detect_webgpu_error_subtypes(
        frame_contract,
        compositor_contract,
        get_path(summary, ["captureErrorString"]),
        get_path(summary, ["captureErrorStack"]),
        get_path(summary, ["captureErrorMessage"]),
        get_path(summary, ["firstValidationFailures"]),
    )
    phase_step = get_path(summary, ["phaseStep"])
    wgsl_parse_error_detected = (
        get_path(frame_contract, ["wgslParseErrorDetected"]) is True
        or get_path(compositor_contract, ["wgslParseErrorDetected"]) is True
        or error_subtypes["wgslParseErrorDetected"]
    )
    shader_module_invalid_detected = (
        get_path(frame_contract, ["shaderModuleInvalidDetected"]) is True
        or get_path(compositor_contract, ["shaderModuleInvalidDetected"]) is True
        or error_subtypes["shaderModuleInvalidDetected"]
    )
    compute_pipeline_invalid_detected = (
        get_path(frame_contract, ["computePipelineInvalidDetected"]) is True
        or get_path(compositor_contract, ["computePipelineInvalidDetected"]) is True
        or error_subtypes["computePipelineInvalidDetected"]
    )
    bind_group_invalid_detected = (
        get_path(frame_contract, ["bindGroupInvalidDetected"]) is True
        or get_path(compositor_contract, ["bindGroupInvalidDetected"]) is True
        or error_subtypes["bindGroupInvalidDetected"]
    )
    deferred_production_items = get_path(
        compositor_contract,
        ["deferredProductionItems"],
        [],
    )
    if not isinstance(deferred_production_items, list):
        deferred_production_items = []
    else:
        deferred_production_items = list(deferred_production_items)
    for item in [
        "full-cuda-parity",
        "final-production-compositor",
        "final-production-compositor-parity",
        "full-parallel-sort-parity",
        "complete-interactive-control-parity",
        "camera-visual-parity",
        "viewport-resize-resource-reallocation-probe",
        "visual-parity-diagnostics",
        "chunk-lod-streaming",
        "early-termination-v1",
    ]:
        if item not in deferred_production_items:
            deferred_production_items.append(item)

    persistent_resource_names = get_path(
        compositor_contract,
        ["persistentResourceNames"],
        [],
    )
    if not isinstance(persistent_resource_names, list):
        persistent_resource_names = []
    remaining_transient_resource_names = get_path(
        compositor_contract,
        ["remainingTransientResourceNames"],
        [],
    )
    if not isinstance(remaining_transient_resource_names, list):
        remaining_transient_resource_names = []
    resource_reuse_count = numeric_value(
        get_path(compositor_contract, ["resourceReuseCount"]),
        0,
    )
    persistent_resource_count = numeric_value(
        get_path(compositor_contract, ["persistentResourceCount"]),
        0,
    )
    step101_preserved = (
        get_path(
            compositor_contract,
            ["step101SelectiveDirtyDependencyPreserved"],
        )
        is True
        and get_path(
            compositor_contract,
            ["selectiveDirtyDependencyExecutionReady"],
        )
        is True
    )
    step100_preserved = (
        get_path(
            compositor_contract,
            ["step100UnifiedInteractionSchedulerPreserved"],
        )
        is True
        and get_path(compositor_contract, ["unifiedInteractionSchedulerReady"])
        is True
    )
    step99_preserved = (
        get_path(compositor_contract, ["step99InteractiveCameraDirtyPreserved"])
        is True
        and get_path(
            compositor_contract,
            ["interactiveCameraDirtyRuntimeReady"],
        )
        is True
    )
    step98_preserved = (
        get_path(compositor_contract, ["step98ViewerTimeSchedulerPreserved"])
        is True
        and get_path(
            compositor_contract,
            ["viewerConnectedInteractiveSchedulerReady"],
        )
        is True
    )
    step97_preserved = (
        get_path(compositor_contract, ["step97MultiFrameRuntimePreserved"])
        is True
        and get_path(compositor_contract, ["timeDrivenProductionRuntimeReady"])
        is True
    )
    step96_preserved = (
        get_path(compositor_contract, ["step96ProductionTileCompositorPreserved"])
        is True
        and get_path(compositor_contract, ["productionTileCompositorReady"]) is True
    )
    step94_preserved = (
        get_path(compositor_contract, ["step94ParallelSortPreserved"]) is True
        and get_path(compositor_contract, ["gpuParallelPerTileSortReady"]) is True
    )
    step88_preserved = (
        get_path(compositor_contract, ["step88PresentationContractPreserved"]) is True
        or get_path(
            frame_contract,
            ["steadyStateTileCompositorOwnsFinalPresentation"],
        )
        is True
    )
    lifecycle_ready = (
        phase_step == "phase3-step102"
        and get_path(compositor_contract, ["productionResourceLifecycleReady"])
        is True
        and get_path(compositor_contract, ["persistentGpuResourceCacheReady"])
        is True
        and len(persistent_resource_names) > 0
        and get_path(compositor_contract, ["resourceReallocationBoundaryReady"])
        is True
        and isinstance(
            get_path(compositor_contract, ["resourceReallocationPolicy"]),
            str,
        )
        and get_path(
            compositor_contract,
            ["outputTextureReallocationBoundaryReady"],
        )
        is True
        and get_path(
            compositor_contract,
            ["selectiveExecutorConnectedToResourceReuse"],
        )
        is True
        and get_path(
            compositor_contract,
            ["dirtyReasonResourceReusePolicyReady"],
        )
        is True
        and get_path(compositor_contract, ["cleanFrameUsesPersistentOutput"])
        is True
        and get_path(compositor_contract, ["dirtyFrameReusesPersistentResources"])
        is True
        and get_path(compositor_contract, ["realtimeBottleneckEvidenceReady"])
        is True
        and isinstance(
            get_path(compositor_contract, ["bottleneckClassification"]),
            str,
        )
        and isinstance(
            get_path(compositor_contract, ["nextStepRecommendedGoal"]),
            str,
        )
        and resource_reuse_count > 0
        and persistent_resource_count > 0
        and step101_preserved
        and step100_preserved
        and step99_preserved
        and step98_preserved
        and step97_preserved
        and step96_preserved
        and step94_preserved
        and step88_preserved
        and get_path(compositor_contract, ["fallbackOnlyCompositorUsed"]) is False
        and get_path(compositor_contract, ["debugOutputBypassedForProduction"])
        is True
        and get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        )
        is True
        and get_path(compositor_contract, ["visualOutputDegeneratedDetected"])
        is not True
        and wgsl_parse_error_detected is False
        and shader_module_invalid_detected is False
        and compute_pipeline_invalid_detected is False
        and bind_group_invalid_detected is False
        and get_path(frame_contract, ["webgpuValidationErrorDetected"]) is False
        and get_path(frame_contract, ["invalidCommandBufferDetected"]) is False
        and get_path(frame_contract, ["queueSubmitFailureDetected"]) is False
        and get_path(frame_contract, ["webgpuWebgl2SameFramePresentationMixed"])
        is False
        and get_path(frame_contract, ["fallbackMixingPrevented"]) is True
        and get_path(frame_contract, ["fullRendererSuccessClaimed"]) is False
    )
    blocked_reason = None
    if not lifecycle_ready:
        if phase_step != "phase3-step102":
            blocked_reason = "summary-phase-step-is-not-phase3-step102"
        elif get_path(compositor_contract, ["productionResourceLifecycleReady"]) is not True:
            blocked_reason = "production-resource-lifecycle-not-ready"
        elif get_path(compositor_contract, ["persistentGpuResourceCacheReady"]) is not True:
            blocked_reason = "persistent-gpu-resource-cache-not-ready"
        elif get_path(
            compositor_contract,
            ["selectiveExecutorConnectedToResourceReuse"],
        ) is not True:
            blocked_reason = "selective-executor-resource-reuse-not-connected"
        elif get_path(compositor_contract, ["realtimeBottleneckEvidenceReady"]) is not True:
            blocked_reason = "realtime-bottleneck-evidence-not-ready"
        elif not step101_preserved:
            blocked_reason = "step101-selective-dirty-not-preserved"
        elif not step100_preserved:
            blocked_reason = "step100-unified-scheduler-not-preserved"
        elif wgsl_parse_error_detected:
            blocked_reason = "wgsl-parse-error-detected"
        elif shader_module_invalid_detected:
            blocked_reason = "shader-module-invalid-detected"
        elif compute_pipeline_invalid_detected:
            blocked_reason = "compute-pipeline-invalid-detected"
        elif bind_group_invalid_detected:
            blocked_reason = "bind-group-invalid-detected"
        else:
            blocked_reason = "step102-resource-lifecycle-validation-failed"

    return {
        "step102Decision": "success" if lifecycle_ready else "blocked",
        "step102BlockedReason": blocked_reason,
        "step102SelectedGoal":
            "A+B+C+D-production-resource-lifecycle-bottleneck-gate-v1",
        "phaseStep": phase_step,
        "step102SummaryApplies": phase_step == "phase3-step102",
        "productionResourceLifecycleReady": get_path(
            compositor_contract,
            ["productionResourceLifecycleReady"],
        ),
        "persistentGpuResourceCacheReady": get_path(
            compositor_contract,
            ["persistentGpuResourceCacheReady"],
        ),
        "persistentResourceNames": persistent_resource_names,
        "transientResourceNames": get_path(
            compositor_contract,
            ["transientResourceNames"],
        ),
        "remainingTransientResourceNames": remaining_transient_resource_names,
        "resourceReallocationBoundaryReady": get_path(
            compositor_contract,
            ["resourceReallocationBoundaryReady"],
        ),
        "resourceReallocationPolicy": get_path(
            compositor_contract,
            ["resourceReallocationPolicy"],
        ),
        "resourceReallocationReasons": get_path(
            compositor_contract,
            ["resourceReallocationReasons"],
        ),
        "outputTextureReallocationBoundaryReady": get_path(
            compositor_contract,
            ["outputTextureReallocationBoundaryReady"],
        ),
        "viewportResizeResourceReallocationDeferredReason": get_path(
            compositor_contract,
            ["viewportResizeResourceReallocationDeferredReason"],
        ),
        "selectiveExecutorConnectedToResourceReuse": get_path(
            compositor_contract,
            ["selectiveExecutorConnectedToResourceReuse"],
        ),
        "dirtyReasonResourceReusePolicyReady": get_path(
            compositor_contract,
            ["dirtyReasonResourceReusePolicyReady"],
        ),
        "resourceReusePolicyByDirtyReason": get_path(
            compositor_contract,
            ["resourceReusePolicyByDirtyReason"],
        ),
        "persistentResourceReuseByDirtyReason": get_path(
            compositor_contract,
            ["persistentResourceReuseByDirtyReason"],
        ),
        "resourceAllocationCount": get_path(
            compositor_contract,
            ["resourceAllocationCount"],
        ),
        "resourceReuseCount": get_path(
            compositor_contract,
            ["resourceReuseCount"],
        ),
        "resourceReallocationCount": get_path(
            compositor_contract,
            ["resourceReallocationCount"],
        ),
        "transientResourceCount": get_path(
            compositor_contract,
            ["transientResourceCount"],
        ),
        "persistentResourceCount": get_path(
            compositor_contract,
            ["persistentResourceCount"],
        ),
        "cleanFrameUsesPersistentOutput": get_path(
            compositor_contract,
            ["cleanFrameUsesPersistentOutput"],
        ),
        "dirtyFrameReusesPersistentResources": get_path(
            compositor_contract,
            ["dirtyFrameReusesPersistentResources"],
        ),
        "realtimeBottleneckEvidenceReady": get_path(
            compositor_contract,
            ["realtimeBottleneckEvidenceReady"],
        ),
        "bottleneckClassification": get_path(
            compositor_contract,
            ["bottleneckClassification"],
        ),
        "bottleneckStageName": get_path(
            compositor_contract,
            ["bottleneckStageName"],
        ),
        "bottleneckEvidence": get_path(
            compositor_contract,
            ["bottleneckEvidence"],
        ),
        "nextStepRecommendedGoal": get_path(
            compositor_contract,
            ["nextStepRecommendedGoal"],
        ),
        "earlyTerminationDeferredReason": get_path(
            compositor_contract,
            ["earlyTerminationDeferredReason"],
        ),
        "chunkLodStreamingReadiness": get_path(
            compositor_contract,
            ["chunkLodStreamingReadiness"],
        ),
        "visualParityDiagnosticsDeferredReason": get_path(
            compositor_contract,
            ["visualParityDiagnosticsDeferredReason"],
        ),
        "step101SelectiveDirtyDependencyPreserved": step101_preserved,
        "step100UnifiedInteractionSchedulerPreserved": step100_preserved,
        "step99InteractiveCameraDirtyPreserved": step99_preserved,
        "step98ViewerTimeSchedulerPreserved": step98_preserved,
        "step97MultiFrameRuntimePreserved": step97_preserved,
        "step96ProductionTileCompositorPreserved": step96_preserved,
        "step94ParallelSortPreserved": step94_preserved,
        "step88PresentationContractPreserved": step88_preserved,
        "step85TileCompositorPathPreserved": get_path(
            frame_contract,
            ["step85TileCompositorPathPreserved"],
        ),
        "step86BoundaryContractPreserved": get_path(
            frame_contract,
            ["step86BoundaryContractPreserved"],
        ),
        "step87DepthOrderingPreserved": get_path(
            frame_contract,
            ["step87DepthOrderingPreserved"],
        ),
        "fallbackOnlyCompositorUsed": get_path(
            compositor_contract,
            ["fallbackOnlyCompositorUsed"],
        ),
        "debugOutputBypassedForProduction": get_path(
            compositor_contract,
            ["debugOutputBypassedForProduction"],
        ),
        "diagnosticReadbackSeparatedFromRuntimePath": get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        ),
        "visualOutputDegeneratedDetected": get_path(
            compositor_contract,
            ["visualOutputDegeneratedDetected"],
        ),
        "wgslParseErrorDetected": wgsl_parse_error_detected,
        "shaderModuleInvalidDetected": shader_module_invalid_detected,
        "computePipelineInvalidDetected": compute_pipeline_invalid_detected,
        "bindGroupInvalidDetected": bind_group_invalid_detected,
        "webgpuValidationErrorDetected": get_path(
            frame_contract,
            ["webgpuValidationErrorDetected"],
        ),
        "invalidCommandBufferDetected": get_path(
            frame_contract,
            ["invalidCommandBufferDetected"],
        ),
        "queueSubmitFailureDetected": get_path(
            frame_contract,
            ["queueSubmitFailureDetected"],
        ),
        "deferredProductionItems": deferred_production_items,
        "fullRendererSuccessClaimed": get_path(
            frame_contract,
            ["fullRendererSuccessClaimed"],
        ),
    }


def build_step103_runtime_boundary_work_reduction_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    frame_contract = get_path(
        summary,
        ["webgpuTileCompositorFrameImplementation"],
        {},
    )
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    error_subtypes = detect_webgpu_error_subtypes(
        frame_contract,
        compositor_contract,
        get_path(summary, ["captureErrorString"]),
        get_path(summary, ["captureErrorStack"]),
        get_path(summary, ["captureErrorMessage"]),
        get_path(summary, ["firstValidationFailures"]),
    )
    phase_step = get_path(summary, ["phaseStep"])
    wgsl_parse_error_detected = (
        get_path(frame_contract, ["wgslParseErrorDetected"]) is True
        or get_path(compositor_contract, ["wgslParseErrorDetected"]) is True
        or error_subtypes["wgslParseErrorDetected"]
    )
    shader_module_invalid_detected = (
        get_path(frame_contract, ["shaderModuleInvalidDetected"]) is True
        or get_path(compositor_contract, ["shaderModuleInvalidDetected"]) is True
        or error_subtypes["shaderModuleInvalidDetected"]
    )
    compute_pipeline_invalid_detected = (
        get_path(frame_contract, ["computePipelineInvalidDetected"]) is True
        or get_path(compositor_contract, ["computePipelineInvalidDetected"]) is True
        or error_subtypes["computePipelineInvalidDetected"]
    )
    bind_group_invalid_detected = (
        get_path(frame_contract, ["bindGroupInvalidDetected"]) is True
        or get_path(compositor_contract, ["bindGroupInvalidDetected"]) is True
        or error_subtypes["bindGroupInvalidDetected"]
    )
    deferred_production_items = get_path(
        compositor_contract,
        ["deferredProductionItems"],
        [],
    )
    if not isinstance(deferred_production_items, list):
        deferred_production_items = []
    else:
        deferred_production_items = list(deferred_production_items)
    for item in [
        "full-cuda-parity",
        "final-production-compositor",
        "final-production-compositor-parity",
        "full-parallel-sort-parity",
        "complete-interactive-control-parity",
        "camera-visual-parity",
        "viewport-resize-resource-reallocation-probe",
        "visual-parity-diagnostics",
        "chunk-lod-streaming",
        "early-termination-v1-alpha-threshold-and-visual-parity-gate",
    ]:
        if item not in deferred_production_items:
            deferred_production_items.append(item)

    step102_preserved = (
        get_path(
            compositor_contract,
            ["step102ProductionResourceLifecyclePreserved"],
        )
        is True
        and get_path(compositor_contract, ["productionResourceLifecycleReady"])
        is True
    )
    step101_preserved = (
        get_path(
            compositor_contract,
            ["step101SelectiveDirtyDependencyPreserved"],
        )
        is True
        and get_path(
            compositor_contract,
            ["selectiveDirtyDependencyExecutionReady"],
        )
        is True
    )
    step100_preserved = (
        get_path(
            compositor_contract,
            ["step100UnifiedInteractionSchedulerPreserved"],
        )
        is True
        and get_path(compositor_contract, ["unifiedInteractionSchedulerReady"])
        is True
    )
    step99_preserved = (
        get_path(compositor_contract, ["step99InteractiveCameraDirtyPreserved"])
        is True
        and get_path(
            compositor_contract,
            ["interactiveCameraDirtyRuntimeReady"],
        )
        is True
    )
    step98_preserved = (
        get_path(compositor_contract, ["step98ViewerTimeSchedulerPreserved"])
        is True
        and get_path(
            compositor_contract,
            ["viewerConnectedInteractiveSchedulerReady"],
        )
        is True
    )
    step97_preserved = (
        get_path(compositor_contract, ["step97MultiFrameRuntimePreserved"])
        is True
        and get_path(compositor_contract, ["timeDrivenProductionRuntimeReady"])
        is True
    )
    step96_preserved = (
        get_path(compositor_contract, ["step96ProductionTileCompositorPreserved"])
        is True
        and get_path(compositor_contract, ["productionTileCompositorReady"]) is True
    )
    step94_preserved = (
        get_path(compositor_contract, ["step94ParallelSortPreserved"]) is True
        and get_path(compositor_contract, ["gpuParallelPerTileSortReady"]) is True
    )
    step88_preserved = (
        get_path(compositor_contract, ["step88PresentationContractPreserved"]) is True
        or get_path(
            frame_contract,
            ["steadyStateTileCompositorOwnsFinalPresentation"],
        )
        is True
    )
    boundary_ready = (
        phase_step == "phase3-step103"
        and get_path(
            compositor_contract,
            ["productionRuntimeBoundaryReviewReady"],
        )
        is True
        and get_path(
            compositor_contract,
            ["productionSteadyStateNoReadbackBoundaryReady"],
        )
        is True
        and get_path(
            compositor_contract,
            ["readbackLeakIntoProductionRuntimeDetected"],
        )
        is False
        and get_path(
            compositor_contract,
            ["diagnosticCaptureReadbackGateReady"],
        )
        is True
        and get_path(
            compositor_contract,
            ["diagnosticReadbackResourcesIsolated"],
        )
        is True
        and get_path(compositor_contract, ["runtimeBoundaryHardened"]) is True
        and get_path(compositor_contract, ["workReductionPathImplemented"])
        is True
        and get_path(compositor_contract, ["activeTileWorkReductionReady"])
        is True
        and numeric_value(
            get_path(compositor_contract, ["activeTilePixelWorkItemCount"]),
            0,
        )
        > 0
        and numeric_value(
            get_path(compositor_contract, ["fullScreenPixelWorkAvoided"]),
            0,
        )
        > 0
        and numeric_value(
            get_path(compositor_contract, ["accumulationWorkReductionRatio"]),
            0,
        )
        > 0
        and get_path(compositor_contract, ["earlyTerminationCandidateReady"])
        is True
        and isinstance(
            get_path(compositor_contract, ["updatedBottleneckClassification"]),
            str,
        )
        and isinstance(
            get_path(compositor_contract, ["updatedNextStepRecommendedGoal"]),
            str,
        )
        and step102_preserved
        and step101_preserved
        and step100_preserved
        and step99_preserved
        and step98_preserved
        and step97_preserved
        and step96_preserved
        and step94_preserved
        and step88_preserved
        and get_path(compositor_contract, ["fallbackOnlyCompositorUsed"]) is False
        and get_path(compositor_contract, ["debugOutputBypassedForProduction"])
        is True
        and get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        )
        is True
        and get_path(compositor_contract, ["visualOutputDegeneratedDetected"])
        is not True
        and wgsl_parse_error_detected is False
        and shader_module_invalid_detected is False
        and compute_pipeline_invalid_detected is False
        and bind_group_invalid_detected is False
        and get_path(frame_contract, ["webgpuValidationErrorDetected"]) is False
        and get_path(frame_contract, ["invalidCommandBufferDetected"]) is False
        and get_path(frame_contract, ["queueSubmitFailureDetected"]) is False
        and get_path(frame_contract, ["webgpuWebgl2SameFramePresentationMixed"])
        is False
        and get_path(frame_contract, ["fallbackMixingPrevented"]) is True
        and get_path(frame_contract, ["fullRendererSuccessClaimed"]) is False
    )
    blocked_reason = None
    if not boundary_ready:
        if phase_step != "phase3-step103":
            blocked_reason = "summary-phase-step-is-not-phase3-step103"
        elif get_path(
            compositor_contract,
            ["readbackLeakIntoProductionRuntimeDetected"],
        ) is True:
            blocked_reason = "production-readback-leak-detected"
        elif get_path(
            compositor_contract,
            ["diagnosticCaptureReadbackGateReady"],
        ) is not True:
            blocked_reason = "diagnostic-capture-readback-gate-not-ready"
        elif get_path(compositor_contract, ["workReductionPathImplemented"]) is not True:
            blocked_reason = "work-reduction-path-not-implemented"
        elif not step102_preserved:
            blocked_reason = "step102-resource-lifecycle-not-preserved"
        elif wgsl_parse_error_detected:
            blocked_reason = "wgsl-parse-error-detected"
        elif shader_module_invalid_detected:
            blocked_reason = "shader-module-invalid-detected"
        elif compute_pipeline_invalid_detected:
            blocked_reason = "compute-pipeline-invalid-detected"
        elif bind_group_invalid_detected:
            blocked_reason = "bind-group-invalid-detected"
        else:
            blocked_reason = "step103-runtime-boundary-validation-failed"

    return {
        "step103Decision": "success" if boundary_ready else "blocked",
        "step103BlockedReason": blocked_reason,
        "step103SelectedGoal":
            "A+B-review-plus-C-active-tile-work-reduction-and-G-next-step-selection",
        "phaseStep": phase_step,
        "step103SummaryApplies": phase_step == "phase3-step103",
        "productionRuntimeBoundaryReviewReady": get_path(
            compositor_contract,
            ["productionRuntimeBoundaryReviewReady"],
        ),
        "productionSteadyStateNoReadbackBoundaryReady": get_path(
            compositor_contract,
            ["productionSteadyStateNoReadbackBoundaryReady"],
        ),
        "readbackLeakIntoProductionRuntimeDetected": get_path(
            compositor_contract,
            ["readbackLeakIntoProductionRuntimeDetected"],
        ),
        "diagnosticCaptureReadbackGateReady": get_path(
            compositor_contract,
            ["diagnosticCaptureReadbackGateReady"],
        ),
        "diagnosticReadbackResourcesIsolated": get_path(
            compositor_contract,
            ["diagnosticReadbackResourcesIsolated"],
        ),
        "runtimeBoundaryHardened": get_path(
            compositor_contract,
            ["runtimeBoundaryHardened"],
        ),
        "selectedWorkReductionPath": get_path(
            compositor_contract,
            ["selectedWorkReductionPath"],
        ),
        "workReductionPathImplemented": get_path(
            compositor_contract,
            ["workReductionPathImplemented"],
        ),
        "activeTileWorkReductionReady": get_path(
            compositor_contract,
            ["activeTileWorkReductionReady"],
        ),
        "activeTilePixelWorkItemCount": get_path(
            compositor_contract,
            ["activeTilePixelWorkItemCount"],
        ),
        "fullScreenPixelWorkAvoided": get_path(
            compositor_contract,
            ["fullScreenPixelWorkAvoided"],
        ),
        "accumulationWorkReductionRatio": get_path(
            compositor_contract,
            ["accumulationWorkReductionRatio"],
        ),
        "earlyTerminationCandidateReady": get_path(
            compositor_contract,
            ["earlyTerminationCandidateReady"],
        ),
        "earlyTerminationV1Used": get_path(
            compositor_contract,
            ["earlyTerminationV1Used"],
        ),
        "earlyTerminationEvaluatedReferenceCount": get_path(
            compositor_contract,
            ["earlyTerminationEvaluatedReferenceCount"],
        ),
        "earlyTerminationSkippedReferenceCount": get_path(
            compositor_contract,
            ["earlyTerminationSkippedReferenceCount"],
        ),
        "earlyTerminationSkipRatio": get_path(
            compositor_contract,
            ["earlyTerminationSkipRatio"],
        ),
        "earlyTerminationDeferredReason": get_path(
            compositor_contract,
            ["earlyTerminationDeferredReason"],
        ),
        "viewportResizeResourceReallocationIntegrated": get_path(
            compositor_contract,
            ["viewportResizeResourceReallocationIntegrated"],
        ),
        "viewportResizeResourceReallocationDeferredReason": get_path(
            compositor_contract,
            ["viewportResizeResourceReallocationDeferredReason"],
        ),
        "visualParityDiagnosticsReadiness": get_path(
            compositor_contract,
            ["visualParityDiagnosticsReadiness"],
        ),
        "chunkLodStreamingReadiness": get_path(
            compositor_contract,
            ["chunkLodStreamingReadiness"],
        ),
        "bottleneckClassification": get_path(
            compositor_contract,
            ["bottleneckClassification"],
        ),
        "updatedBottleneckClassification": get_path(
            compositor_contract,
            ["updatedBottleneckClassification"],
        ),
        "nextStepRecommendedGoal": get_path(
            compositor_contract,
            ["nextStepRecommendedGoal"],
        ),
        "updatedNextStepRecommendedGoal": get_path(
            compositor_contract,
            ["updatedNextStepRecommendedGoal"],
        ),
        "bottleneckEvidence": get_path(
            compositor_contract,
            ["bottleneckEvidence"],
        ),
        "step102ProductionResourceLifecyclePreserved": step102_preserved,
        "step101SelectiveDirtyDependencyPreserved": step101_preserved,
        "step100UnifiedInteractionSchedulerPreserved": step100_preserved,
        "step99InteractiveCameraDirtyPreserved": step99_preserved,
        "step98ViewerTimeSchedulerPreserved": step98_preserved,
        "step97MultiFrameRuntimePreserved": step97_preserved,
        "step96ProductionTileCompositorPreserved": step96_preserved,
        "step94ParallelSortPreserved": step94_preserved,
        "step88PresentationContractPreserved": step88_preserved,
        "step85TileCompositorPathPreserved": get_path(
            frame_contract,
            ["step85TileCompositorPathPreserved"],
        ),
        "step86BoundaryContractPreserved": get_path(
            frame_contract,
            ["step86BoundaryContractPreserved"],
        ),
        "step87DepthOrderingPreserved": get_path(
            frame_contract,
            ["step87DepthOrderingPreserved"],
        ),
        "fallbackOnlyCompositorUsed": get_path(
            compositor_contract,
            ["fallbackOnlyCompositorUsed"],
        ),
        "debugOutputBypassedForProduction": get_path(
            compositor_contract,
            ["debugOutputBypassedForProduction"],
        ),
        "diagnosticReadbackSeparatedFromRuntimePath": get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        ),
        "visualOutputDegeneratedDetected": get_path(
            compositor_contract,
            ["visualOutputDegeneratedDetected"],
        ),
        "wgslParseErrorDetected": wgsl_parse_error_detected,
        "shaderModuleInvalidDetected": shader_module_invalid_detected,
        "computePipelineInvalidDetected": compute_pipeline_invalid_detected,
        "bindGroupInvalidDetected": bind_group_invalid_detected,
        "webgpuValidationErrorDetected": get_path(
            frame_contract,
            ["webgpuValidationErrorDetected"],
        ),
        "invalidCommandBufferDetected": get_path(
            frame_contract,
            ["invalidCommandBufferDetected"],
        ),
        "queueSubmitFailureDetected": get_path(
            frame_contract,
            ["queueSubmitFailureDetected"],
        ),
        "deferredProductionItems": deferred_production_items,
        "fullRendererSuccessClaimed": get_path(
            frame_contract,
            ["fullRendererSuccessClaimed"],
        ),
    }


def build_step104_compositor_work_reduction_visual_safety_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    frame_contract = get_path(
        summary,
        ["webgpuTileCompositorFrameImplementation"],
        {},
    )
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    error_subtypes = detect_webgpu_error_subtypes(
        frame_contract,
        compositor_contract,
        get_path(summary, ["captureErrorString"]),
        get_path(summary, ["captureErrorStack"]),
        get_path(summary, ["captureErrorMessage"]),
        get_path(summary, ["firstValidationFailures"]),
    )
    phase_step = get_path(summary, ["phaseStep"])
    wgsl_parse_error_detected = (
        get_path(frame_contract, ["wgslParseErrorDetected"]) is True
        or get_path(compositor_contract, ["wgslParseErrorDetected"]) is True
        or error_subtypes["wgslParseErrorDetected"]
    )
    shader_module_invalid_detected = (
        get_path(frame_contract, ["shaderModuleInvalidDetected"]) is True
        or get_path(compositor_contract, ["shaderModuleInvalidDetected"]) is True
        or error_subtypes["shaderModuleInvalidDetected"]
    )
    compute_pipeline_invalid_detected = (
        get_path(frame_contract, ["computePipelineInvalidDetected"]) is True
        or get_path(compositor_contract, ["computePipelineInvalidDetected"]) is True
        or error_subtypes["computePipelineInvalidDetected"]
    )
    bind_group_invalid_detected = (
        get_path(frame_contract, ["bindGroupInvalidDetected"]) is True
        or get_path(compositor_contract, ["bindGroupInvalidDetected"]) is True
        or error_subtypes["bindGroupInvalidDetected"]
    )
    deferred_production_items = get_path(
        compositor_contract,
        ["deferredProductionItems"],
        [],
    )
    if not isinstance(deferred_production_items, list):
        deferred_production_items = []
    else:
        deferred_production_items = list(deferred_production_items)
    for item in [
        "full-cuda-parity",
        "final-production-compositor",
        "final-production-compositor-parity",
        "full-parallel-sort-parity",
        "complete-interactive-control-parity",
        "camera-visual-parity",
        "viewport-resize-resource-reallocation-probe",
        "visual-parity-diagnostics",
        "chunk-lod-streaming",
        "early-termination-v1-alpha-threshold-and-visual-parity-gate",
        "early-termination-on-path-output-delta-probe",
    ]:
        if item not in deferred_production_items:
            deferred_production_items.append(item)

    step103_preserved = (
        get_path(
            compositor_contract,
            ["step103ProductionRuntimeBoundaryReviewPreserved"],
        )
        is True
        and get_path(
            compositor_contract,
            ["productionRuntimeBoundaryReviewReady"],
        )
        is True
    )
    step102_preserved = (
        get_path(
            compositor_contract,
            ["step102ProductionResourceLifecyclePreserved"],
        )
        is True
        and get_path(compositor_contract, ["productionResourceLifecycleReady"])
        is True
    )
    step101_preserved = (
        get_path(
            compositor_contract,
            ["step101SelectiveDirtyDependencyPreserved"],
        )
        is True
        and get_path(
            compositor_contract,
            ["selectiveDirtyDependencyExecutionReady"],
        )
        is True
    )
    step100_preserved = (
        get_path(
            compositor_contract,
            ["step100UnifiedInteractionSchedulerPreserved"],
        )
        is True
        and get_path(compositor_contract, ["unifiedInteractionSchedulerReady"])
        is True
    )
    step99_preserved = (
        get_path(compositor_contract, ["step99InteractiveCameraDirtyPreserved"])
        is True
        and get_path(
            compositor_contract,
            ["interactiveCameraDirtyRuntimeReady"],
        )
        is True
    )
    step98_preserved = (
        get_path(compositor_contract, ["step98ViewerTimeSchedulerPreserved"])
        is True
        and get_path(
            compositor_contract,
            ["viewerConnectedInteractiveSchedulerReady"],
        )
        is True
    )
    step97_preserved = (
        get_path(compositor_contract, ["step97MultiFrameRuntimePreserved"])
        is True
        and get_path(compositor_contract, ["timeDrivenProductionRuntimeReady"])
        is True
    )
    step96_preserved = (
        get_path(compositor_contract, ["step96ProductionTileCompositorPreserved"])
        is True
        and get_path(compositor_contract, ["productionTileCompositorReady"]) is True
    )
    step94_preserved = (
        get_path(compositor_contract, ["step94ParallelSortPreserved"]) is True
        and get_path(compositor_contract, ["gpuParallelPerTileSortReady"]) is True
    )
    step88_preserved = (
        get_path(compositor_contract, ["step88PresentationContractPreserved"]) is True
        or get_path(
            frame_contract,
            ["steadyStateTileCompositorOwnsFinalPresentation"],
        )
        is True
    )
    safety_ready = (
        phase_step == "phase3-step104"
        and get_path(compositor_contract, ["workReductionReadinessReviewReady"])
        is True
        and get_path(compositor_contract, ["earlyTerminationPolicyReady"])
        is True
        and isinstance(
            get_path(compositor_contract, ["earlyTerminationEnablePolicy"]),
            str,
        )
        and get_path(compositor_contract, ["earlyTerminationEnabled"]) is False
        and get_path(
            compositor_contract,
            ["earlyTerminationDisabledBySafetyGate"],
        )
        is True
        and isinstance(
            get_path(compositor_contract, ["earlyTerminationDisableReason"]),
            str,
        )
        and get_path(compositor_contract, ["visualSafetyGateReady"]) is True
        and get_path(compositor_contract, ["visualQualityRiskGateReady"]) is True
        and isinstance(
            get_path(compositor_contract, ["visualQualityRiskLevel"]),
            str,
        )
        and get_path(
            compositor_contract,
            ["earlyTerminationOnOffComparisonReady"],
        )
        is True
        and isinstance(
            get_path(
                compositor_contract,
                ["earlyTerminationOnOffComparisonMode"],
            ),
            str,
        )
        and get_path(
            compositor_contract,
            ["earlyTerminationOffPathReferenceReady"],
        )
        is True
        and get_path(compositor_contract, ["earlyTerminationOnPathExecuted"])
        is False
        and get_path(
            compositor_contract,
            ["earlyTerminationOnPathDisabledBySafetyGate"],
        )
        is True
        and get_path(
            compositor_contract,
            ["earlyTerminationOnOffOutputDeltaReady"],
        )
        is False
        and get_path(compositor_contract, ["safeThresholdPolicyReady"]) is True
        and isinstance(
            get_path(compositor_contract, ["safeThresholdPolicy"]),
            str,
        )
        and get_path(
            compositor_contract,
            ["safeMinimalEarlyTerminationV1Used"],
        )
        is False
        and isinstance(
            get_path(
                compositor_contract,
                ["safeMinimalEarlyTerminationDeferredReason"],
            ),
            str,
        )
        and isinstance(
            get_path(
                compositor_contract,
                ["finalCompositorVisualParityDiagnosticsReadiness"],
            ),
            str,
        )
        and isinstance(
            get_path(compositor_contract, ["step104BottleneckClassification"]),
            str,
        )
        and isinstance(
            get_path(compositor_contract, ["step104NextStepRecommendedGoal"]),
            str,
        )
        and step103_preserved
        and step102_preserved
        and step101_preserved
        and step100_preserved
        and step99_preserved
        and step98_preserved
        and step97_preserved
        and step96_preserved
        and step94_preserved
        and step88_preserved
        and get_path(compositor_contract, ["fallbackOnlyCompositorUsed"]) is False
        and get_path(compositor_contract, ["debugOutputBypassedForProduction"])
        is True
        and get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        )
        is True
        and get_path(compositor_contract, ["visualOutputDegeneratedDetected"])
        is not True
        and wgsl_parse_error_detected is False
        and shader_module_invalid_detected is False
        and compute_pipeline_invalid_detected is False
        and bind_group_invalid_detected is False
        and get_path(frame_contract, ["webgpuValidationErrorDetected"]) is False
        and get_path(frame_contract, ["invalidCommandBufferDetected"]) is False
        and get_path(frame_contract, ["queueSubmitFailureDetected"]) is False
        and get_path(frame_contract, ["webgpuWebgl2SameFramePresentationMixed"])
        is False
        and get_path(frame_contract, ["fallbackMixingPrevented"]) is True
        and get_path(frame_contract, ["fullRendererSuccessClaimed"]) is False
    )
    blocked_reason = None
    if not safety_ready:
        if phase_step != "phase3-step104":
            blocked_reason = "summary-phase-step-is-not-phase3-step104"
        elif get_path(compositor_contract, ["visualSafetyGateReady"]) is not True:
            blocked_reason = "visual-safety-gate-not-ready"
        elif get_path(
            compositor_contract,
            ["earlyTerminationDisabledBySafetyGate"],
        ) is not True:
            blocked_reason = "early-termination-policy-not-gated"
        elif get_path(
            compositor_contract,
            ["earlyTerminationOnOffComparisonReady"],
        ) is not True:
            blocked_reason = "early-termination-on-off-comparison-not-ready"
        elif not step103_preserved:
            blocked_reason = "step103-boundary-review-not-preserved"
        elif wgsl_parse_error_detected:
            blocked_reason = "wgsl-parse-error-detected"
        elif shader_module_invalid_detected:
            blocked_reason = "shader-module-invalid-detected"
        elif compute_pipeline_invalid_detected:
            blocked_reason = "compute-pipeline-invalid-detected"
        elif bind_group_invalid_detected:
            blocked_reason = "bind-group-invalid-detected"
        else:
            blocked_reason = "step104-visual-safety-validation-failed"

    return {
        "step104Decision": "success" if safety_ready else "blocked",
        "step104BlockedReason": blocked_reason,
        "step104SelectedGoal":
            "A+B+C+D-work-reduction-readiness-visual-safety-gate-plus-H-next-step-selection",
        "phaseStep": phase_step,
        "step104SummaryApplies": phase_step == "phase3-step104",
        "workReductionReadinessReviewReady": get_path(
            compositor_contract,
            ["workReductionReadinessReviewReady"],
        ),
        "selectedWorkReductionPath": get_path(
            compositor_contract,
            ["selectedWorkReductionPath"],
        ),
        "workReductionPathImplemented": get_path(
            compositor_contract,
            ["workReductionPathImplemented"],
        ),
        "activeTileWorkReductionReady": get_path(
            compositor_contract,
            ["activeTileWorkReductionReady"],
        ),
        "activeTilePixelWorkItemCount": get_path(
            compositor_contract,
            ["activeTilePixelWorkItemCount"],
        ),
        "fullScreenPixelWorkAvoided": get_path(
            compositor_contract,
            ["fullScreenPixelWorkAvoided"],
        ),
        "accumulationWorkReductionRatio": get_path(
            compositor_contract,
            ["accumulationWorkReductionRatio"],
        ),
        "earlyTerminationPolicyReady": get_path(
            compositor_contract,
            ["earlyTerminationPolicyReady"],
        ),
        "earlyTerminationEnablePolicy": get_path(
            compositor_contract,
            ["earlyTerminationEnablePolicy"],
        ),
        "earlyTerminationEnabled": get_path(
            compositor_contract,
            ["earlyTerminationEnabled"],
        ),
        "earlyTerminationDisabledBySafetyGate": get_path(
            compositor_contract,
            ["earlyTerminationDisabledBySafetyGate"],
        ),
        "earlyTerminationDisableReason": get_path(
            compositor_contract,
            ["earlyTerminationDisableReason"],
        ),
        "visualSafetyGateReady": get_path(
            compositor_contract,
            ["visualSafetyGateReady"],
        ),
        "visualQualityRiskGateReady": get_path(
            compositor_contract,
            ["visualQualityRiskGateReady"],
        ),
        "visualQualityRiskLevel": get_path(
            compositor_contract,
            ["visualQualityRiskLevel"],
        ),
        "visualSafetyEvidence": get_path(
            compositor_contract,
            ["visualSafetyEvidence"],
        ),
        "earlyTerminationOnOffComparisonReady": get_path(
            compositor_contract,
            ["earlyTerminationOnOffComparisonReady"],
        ),
        "earlyTerminationOnOffComparisonMode": get_path(
            compositor_contract,
            ["earlyTerminationOnOffComparisonMode"],
        ),
        "earlyTerminationOffPathReferenceReady": get_path(
            compositor_contract,
            ["earlyTerminationOffPathReferenceReady"],
        ),
        "earlyTerminationOnPathExecuted": get_path(
            compositor_contract,
            ["earlyTerminationOnPathExecuted"],
        ),
        "earlyTerminationOnPathDisabledBySafetyGate": get_path(
            compositor_contract,
            ["earlyTerminationOnPathDisabledBySafetyGate"],
        ),
        "earlyTerminationOnOffOutputDeltaReady": get_path(
            compositor_contract,
            ["earlyTerminationOnOffOutputDeltaReady"],
        ),
        "safeThresholdPolicyReady": get_path(
            compositor_contract,
            ["safeThresholdPolicyReady"],
        ),
        "safeThresholdPolicy": get_path(
            compositor_contract,
            ["safeThresholdPolicy"],
        ),
        "safeThresholdValue": get_path(
            compositor_contract,
            ["safeThresholdValue"],
        ),
        "safeMinimalEarlyTerminationV1Used": get_path(
            compositor_contract,
            ["safeMinimalEarlyTerminationV1Used"],
        ),
        "safeMinimalEarlyTerminationDeferredReason": get_path(
            compositor_contract,
            ["safeMinimalEarlyTerminationDeferredReason"],
        ),
        "finalCompositorVisualParityDiagnosticsReadiness": get_path(
            compositor_contract,
            ["finalCompositorVisualParityDiagnosticsReadiness"],
        ),
        "step104BottleneckClassification": get_path(
            compositor_contract,
            ["step104BottleneckClassification"],
        ),
        "step104NextStepRecommendedGoal": get_path(
            compositor_contract,
            ["step104NextStepRecommendedGoal"],
        ),
        "updatedBottleneckClassification": get_path(
            compositor_contract,
            ["step104BottleneckClassification"],
        ),
        "updatedNextStepRecommendedGoal": get_path(
            compositor_contract,
            ["step104NextStepRecommendedGoal"],
        ),
        "nextStepRecommendedGoal": get_path(
            compositor_contract,
            ["step104NextStepRecommendedGoal"],
        ),
        "step103ProductionRuntimeBoundaryReviewPreserved": step103_preserved,
        "step102ProductionResourceLifecyclePreserved": step102_preserved,
        "step101SelectiveDirtyDependencyPreserved": step101_preserved,
        "step101SelectiveDirtyExecutionPreserved": step101_preserved,
        "step100UnifiedInteractionSchedulerPreserved": step100_preserved,
        "step99InteractiveCameraDirtyPreserved": step99_preserved,
        "step98ViewerTimeSchedulerPreserved": step98_preserved,
        "step97MultiFrameRuntimePreserved": step97_preserved,
        "step96ProductionTileCompositorPreserved": step96_preserved,
        "step94ParallelSortPreserved": step94_preserved,
        "step88PresentationContractPreserved": step88_preserved,
        "step85TileCompositorPathPreserved": get_path(
            frame_contract,
            ["step85TileCompositorPathPreserved"],
        ),
        "step86BoundaryContractPreserved": get_path(
            frame_contract,
            ["step86BoundaryContractPreserved"],
        ),
        "step87DepthOrderingPreserved": get_path(
            frame_contract,
            ["step87DepthOrderingPreserved"],
        ),
        "fallbackOnlyCompositorUsed": get_path(
            compositor_contract,
            ["fallbackOnlyCompositorUsed"],
        ),
        "debugOutputBypassedForProduction": get_path(
            compositor_contract,
            ["debugOutputBypassedForProduction"],
        ),
        "diagnosticReadbackSeparatedFromRuntimePath": get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        ),
        "visualOutputDegeneratedDetected": get_path(
            compositor_contract,
            ["visualOutputDegeneratedDetected"],
        ),
        "wgslParseErrorDetected": wgsl_parse_error_detected,
        "shaderModuleInvalidDetected": shader_module_invalid_detected,
        "computePipelineInvalidDetected": compute_pipeline_invalid_detected,
        "bindGroupInvalidDetected": bind_group_invalid_detected,
        "webgpuValidationErrorDetected": get_path(
            frame_contract,
            ["webgpuValidationErrorDetected"],
        ),
        "invalidCommandBufferDetected": get_path(
            frame_contract,
            ["invalidCommandBufferDetected"],
        ),
        "queueSubmitFailureDetected": get_path(
            frame_contract,
            ["queueSubmitFailureDetected"],
        ),
        "deferredProductionItems": deferred_production_items,
        "fullRendererSuccessClaimed": get_path(
            frame_contract,
            ["fullRendererSuccessClaimed"],
        ),
    }


def build_step105_cuda_reference_visual_parity_baseline_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    frame_contract = get_path(
        summary,
        ["webgpuTileCompositorFrameImplementation"],
        {},
    )
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    error_subtypes = detect_webgpu_error_subtypes(
        frame_contract,
        compositor_contract,
        get_path(summary, ["captureErrorString"]),
        get_path(summary, ["captureErrorStack"]),
        get_path(summary, ["captureErrorMessage"]),
        get_path(summary, ["firstValidationFailures"]),
    )
    phase_step = get_path(summary, ["phaseStep"])
    wgsl_parse_error_detected = (
        get_path(frame_contract, ["wgslParseErrorDetected"]) is True
        or get_path(compositor_contract, ["wgslParseErrorDetected"]) is True
        or error_subtypes["wgslParseErrorDetected"]
    )
    shader_module_invalid_detected = (
        get_path(frame_contract, ["shaderModuleInvalidDetected"]) is True
        or get_path(compositor_contract, ["shaderModuleInvalidDetected"]) is True
        or error_subtypes["shaderModuleInvalidDetected"]
    )
    compute_pipeline_invalid_detected = (
        get_path(frame_contract, ["computePipelineInvalidDetected"]) is True
        or get_path(compositor_contract, ["computePipelineInvalidDetected"]) is True
        or error_subtypes["computePipelineInvalidDetected"]
    )
    bind_group_invalid_detected = (
        get_path(frame_contract, ["bindGroupInvalidDetected"]) is True
        or get_path(compositor_contract, ["bindGroupInvalidDetected"]) is True
        or error_subtypes["bindGroupInvalidDetected"]
    )
    deferred_production_items = get_path(
        compositor_contract,
        ["deferredProductionItems"],
        [],
    )
    if not isinstance(deferred_production_items, list):
        deferred_production_items = []
    else:
        deferred_production_items = list(deferred_production_items)
    for item in [
        "full-cuda-parity",
        "final-production-compositor",
        "final-production-compositor-parity",
        "full-parallel-sort-parity",
        "complete-interactive-control-parity",
        "camera-visual-parity",
        "early-termination-v1-alpha-threshold-and-visual-parity-gate",
        "early-termination-on-path-output-delta-probe",
        "runtime-pixel-diff-against-reference-image",
        "cuda-reference-execution-refresh",
        "chunk-lod-streaming",
    ]:
        if item not in deferred_production_items:
            deferred_production_items.append(item)

    step104_preserved = (
        get_path(compositor_contract, ["step104VisualSafetyGatePreserved"])
        is True
        and get_path(compositor_contract, ["visualSafetyGateReady"]) is True
        and get_path(
            compositor_contract,
            ["earlyTerminationDisabledBySafetyGate"],
        )
        is True
    )
    step88_presentation_contract_preserved = (
        get_path(compositor_contract, ["step88PresentationContractPreserved"])
        is True
        or get_path(
            frame_contract,
            ["steadyStateTileCompositorOwnsFinalPresentation"],
        )
        is True
    )
    step88_presentation_contract_preservation_source = (
        "compositor-contract"
        if get_path(compositor_contract, ["step88PresentationContractPreserved"])
        is True
        else (
            "frame-steady-state-final-presentation-evidence"
            if get_path(
                frame_contract,
                ["steadyStateTileCompositorOwnsFinalPresentation"],
            )
            is True
            else "missing"
        )
    )
    preservation_ready = (
        step104_preserved
        and get_path(
            compositor_contract,
            ["step103ProductionRuntimeBoundaryReviewPreserved"],
        )
        is True
        and get_path(
            compositor_contract,
            ["step102ProductionResourceLifecyclePreserved"],
        )
        is True
        and get_path(
            compositor_contract,
            ["step101SelectiveDirtyDependencyPreserved"],
        )
        is True
        and get_path(
            compositor_contract,
            ["step100UnifiedInteractionSchedulerPreserved"],
        )
        is True
        and get_path(compositor_contract, ["step99InteractiveCameraDirtyPreserved"])
        is True
        and get_path(compositor_contract, ["step98ViewerTimeSchedulerPreserved"])
        is True
        and get_path(compositor_contract, ["step97MultiFrameRuntimePreserved"])
        is True
        and get_path(compositor_contract, ["step96ProductionTileCompositorPreserved"])
        is True
        and get_path(compositor_contract, ["step94ParallelSortPreserved"]) is True
        and step88_presentation_contract_preserved
    )
    camera_mismatch_classified = (
        get_path(compositor_contract, ["fixedReferenceCameraGateReady"]) is True
        and get_path(compositor_contract, ["cameraContractMismatchDetected"])
        is True
        and get_path(compositor_contract, ["visualParityComparisonAllowed"])
        is False
        and get_path(compositor_contract, ["visualMismatchClassification"])
        == "camera-contract-mismatch"
        and isinstance(
            get_path(compositor_contract, ["cameraContractMismatchReason"]),
            str,
        )
    )
    fixed_reference_camera_ready_for_comparison = (
        get_path(compositor_contract, ["fixedReferenceCameraGateReady"]) is True
        and get_path(
            compositor_contract,
            ["usesCudaAlignedFixedReferenceCamera"],
        )
        is True
        and get_path(compositor_contract, ["visualParityComparisonAllowed"])
        is True
        and get_path(compositor_contract, ["cameraContractMismatchDetected"])
        is False
    )
    parity_baseline_ready = (
        phase_step == "phase3-step105"
        and get_path(compositor_contract, ["productionOutputCaptureReady"]) is True
        and get_path(compositor_contract, ["webgpuProductionOutputCaptureReady"])
        is True
        and get_path(compositor_contract, ["referenceImageInputReady"]) is True
        and get_path(compositor_contract, ["cudaReferenceInputReady"]) is True
        and get_path(compositor_contract, ["visualComparisonConditionsReady"])
        is True
        and get_path(
            compositor_contract,
            ["cameraViewportBackgroundColorSpaceRecorded"],
        )
        is True
        and get_path(compositor_contract, ["fixedReferenceCameraGateReady"])
        is True
        and isinstance(get_path(compositor_contract, ["referenceCameraMode"]), str)
        and get_path(
            compositor_contract,
            ["interactiveCameraExcludedFromReferenceComparison"],
        )
        is True
        and isinstance(get_path(compositor_contract, ["cameraMetadataName"]), str)
        and get_path(compositor_contract, ["cameraProjectionContractReady"])
        is True
        and get_path(compositor_contract, ["viewportComparisonContractReady"])
        is True
        and get_path(compositor_contract, ["backgroundComparisonContractReady"])
        is True
        and get_path(
            compositor_contract,
            ["pixelCoordinateComparisonContractReady"],
        )
        is True
        and get_path(compositor_contract, ["screenSpaceConventionReady"]) is True
        and (
            fixed_reference_camera_ready_for_comparison
            or camera_mismatch_classified
        )
        and get_path(compositor_contract, ["visualParityMetricReady"]) is True
        and isinstance(
            get_path(compositor_contract, ["visualParityMetricMode"]),
            str,
        )
        and isinstance(
            get_path(compositor_contract, ["visualMismatchClassification"]),
            str,
        )
        and isinstance(
            get_path(compositor_contract, ["step105NextStepRecommendedGoal"]),
            str,
        )
        and get_path(compositor_contract, ["earlyTerminationRemainsDisabled"])
        is True
        and get_path(compositor_contract, ["lodStreamingRemainsDisabled"]) is True
        and preservation_ready
        and get_path(compositor_contract, ["fallbackOnlyCompositorUsed"]) is False
        and get_path(compositor_contract, ["debugOutputBypassedForProduction"])
        is True
        and get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        )
        is True
        and get_path(compositor_contract, ["visualOutputDegeneratedDetected"])
        is not True
        and wgsl_parse_error_detected is False
        and shader_module_invalid_detected is False
        and compute_pipeline_invalid_detected is False
        and bind_group_invalid_detected is False
        and get_path(frame_contract, ["invalidCommandBufferDetected"]) is False
        and get_path(frame_contract, ["queueSubmitFailureDetected"]) is False
        and get_path(frame_contract, ["webgpuWebgl2SameFramePresentationMixed"])
        is False
        and get_path(frame_contract, ["fallbackMixingPrevented"]) is True
        and get_path(frame_contract, ["fullRendererSuccessClaimed"]) is False
    )
    blocked_reason = None
    if not parity_baseline_ready:
        if phase_step != "phase3-step105":
            blocked_reason = "summary-phase-step-is-not-phase3-step105"
        elif get_path(compositor_contract, ["productionOutputCaptureReady"]) is not True:
            blocked_reason = "production-output-capture-not-ready"
        elif get_path(compositor_contract, ["referenceImageInputReady"]) is not True:
            blocked_reason = "reference-image-input-not-ready"
        elif get_path(compositor_contract, ["fixedReferenceCameraGateReady"]) is not True:
            blocked_reason = "fixed-reference-camera-gate-not-ready"
        elif not (
            fixed_reference_camera_ready_for_comparison
            or camera_mismatch_classified
        ):
            blocked_reason = "fixed-reference-camera-gate-unclassified"
        elif get_path(compositor_contract, ["visualParityMetricReady"]) is not True:
            blocked_reason = "visual-parity-metric-not-ready"
        elif not step104_preserved:
            blocked_reason = "step104-visual-safety-gate-not-preserved"
        elif wgsl_parse_error_detected:
            blocked_reason = "wgsl-parse-error-detected"
        elif shader_module_invalid_detected:
            blocked_reason = "shader-module-invalid-detected"
        elif compute_pipeline_invalid_detected:
            blocked_reason = "compute-pipeline-invalid-detected"
        elif bind_group_invalid_detected:
            blocked_reason = "bind-group-invalid-detected"
        else:
            blocked_reason = "step105-visual-parity-baseline-validation-failed"

    return {
        "step105Decision": "success" if parity_baseline_ready else "blocked",
        "step105BlockedReason": blocked_reason,
        "step105SelectedGoal":
            "A+B+C+D+F-reference-output-capture-visual-parity-baseline",
        "phaseStep": phase_step,
        "step105SummaryApplies": phase_step == "phase3-step105",
        "productionOutputCaptureReady": get_path(
            compositor_contract,
            ["productionOutputCaptureReady"],
        ),
        "webgpuProductionOutputCaptureReady": get_path(
            compositor_contract,
            ["webgpuProductionOutputCaptureReady"],
        ),
        "referenceImageInputReady": get_path(
            compositor_contract,
            ["referenceImageInputReady"],
        ),
        "cudaReferenceInputReady": get_path(
            compositor_contract,
            ["cudaReferenceInputReady"],
        ),
        "referenceImageSource": get_path(
            compositor_contract,
            ["referenceImageSource"],
        ),
        "referenceImagePath": get_path(
            compositor_contract,
            ["referenceImagePath"],
        ),
        "referenceImageLabel": get_path(
            compositor_contract,
            ["referenceImageLabel"],
        ),
        "fixedReferenceCameraGateReady": get_path(
            compositor_contract,
            ["fixedReferenceCameraGateReady"],
        ),
        "referenceCameraMode": get_path(
            compositor_contract,
            ["referenceCameraMode"],
        ),
        "usesCudaAlignedFixedReferenceCamera": get_path(
            compositor_contract,
            ["usesCudaAlignedFixedReferenceCamera"],
        ),
        "interactiveCameraExcludedFromReferenceComparison": get_path(
            compositor_contract,
            ["interactiveCameraExcludedFromReferenceComparison"],
        ),
        "cameraMetadataName": get_path(
            compositor_contract,
            ["cameraMetadataName"],
        ),
        "cameraProjectionContractReady": get_path(
            compositor_contract,
            ["cameraProjectionContractReady"],
        ),
        "viewportComparisonContractReady": get_path(
            compositor_contract,
            ["viewportComparisonContractReady"],
        ),
        "backgroundComparisonContractReady": get_path(
            compositor_contract,
            ["backgroundComparisonContractReady"],
        ),
        "pixelCoordinateComparisonContractReady": get_path(
            compositor_contract,
            ["pixelCoordinateComparisonContractReady"],
        ),
        "screenSpaceConventionReady": get_path(
            compositor_contract,
            ["screenSpaceConventionReady"],
        ),
        "cameraContractMismatchDetected": get_path(
            compositor_contract,
            ["cameraContractMismatchDetected"],
        ),
        "cameraContractMismatchReason": get_path(
            compositor_contract,
            ["cameraContractMismatchReason"],
        ),
        "visualParityComparisonAllowed": get_path(
            compositor_contract,
            ["visualParityComparisonAllowed"],
        ),
        "visualComparisonConditionsReady": get_path(
            compositor_contract,
            ["visualComparisonConditionsReady"],
        ),
        "visualComparisonConditions": get_path(
            compositor_contract,
            ["visualComparisonConditions"],
        ),
        "cameraViewportBackgroundColorSpaceRecorded": get_path(
            compositor_contract,
            ["cameraViewportBackgroundColorSpaceRecorded"],
        ),
        "visualParityMetricReady": get_path(
            compositor_contract,
            ["visualParityMetricReady"],
        ),
        "visualParityMetricMode": get_path(
            compositor_contract,
            ["visualParityMetricMode"],
        ),
        "visualParityDifferenceSummary": get_path(
            compositor_contract,
            ["visualParityDifferenceSummary"],
        ),
        "visualMismatchClassification": get_path(
            compositor_contract,
            ["visualMismatchClassification"],
        ),
        "finalCompositorParityDiagnosticsReady": get_path(
            compositor_contract,
            ["finalCompositorParityDiagnosticsReady"],
        ),
        "finalCompositorParityDiagnosticsMode": get_path(
            compositor_contract,
            ["finalCompositorParityDiagnosticsMode"],
        ),
        "nextStepRecommendedGoal": get_path(
            compositor_contract,
            ["step105NextStepRecommendedGoal"],
        ),
        "step104VisualSafetyGatePreserved": step104_preserved,
        "step103ProductionRuntimeBoundaryReviewPreserved": get_path(
            compositor_contract,
            ["step103ProductionRuntimeBoundaryReviewPreserved"],
        ),
        "step102ProductionResourceLifecyclePreserved": get_path(
            compositor_contract,
            ["step102ProductionResourceLifecyclePreserved"],
        ),
        "step101SelectiveDirtyExecutionPreserved": get_path(
            compositor_contract,
            ["step101SelectiveDirtyDependencyPreserved"],
        ),
        "step100UnifiedInteractionSchedulerPreserved": get_path(
            compositor_contract,
            ["step100UnifiedInteractionSchedulerPreserved"],
        ),
        "step99InteractiveCameraDirtyPreserved": get_path(
            compositor_contract,
            ["step99InteractiveCameraDirtyPreserved"],
        ),
        "step98ViewerTimeSchedulerPreserved": get_path(
            compositor_contract,
            ["step98ViewerTimeSchedulerPreserved"],
        ),
        "step97MultiFrameRuntimePreserved": get_path(
            compositor_contract,
            ["step97MultiFrameRuntimePreserved"],
        ),
        "step96ProductionTileCompositorPreserved": get_path(
            compositor_contract,
            ["step96ProductionTileCompositorPreserved"],
        ),
        "step94ParallelSortPreserved": get_path(
            compositor_contract,
            ["step94ParallelSortPreserved"],
        ),
        "step88PresentationContractPreserved":
            step88_presentation_contract_preserved,
        "step88PresentationContractPreservationSource":
            step88_presentation_contract_preservation_source,
        "step85TileCompositorPathPreserved": get_path(
            frame_contract,
            ["step85TileCompositorPathPreserved"],
        ),
        "step86BoundaryContractPreserved": get_path(
            frame_contract,
            ["step86BoundaryContractPreserved"],
        ),
        "step87DepthOrderingPreserved": get_path(
            frame_contract,
            ["step87DepthOrderingPreserved"],
        ),
        "earlyTerminationRemainsDisabled": get_path(
            compositor_contract,
            ["earlyTerminationRemainsDisabled"],
        ),
        "lodStreamingRemainsDisabled": get_path(
            compositor_contract,
            ["lodStreamingRemainsDisabled"],
        ),
        "fallbackOnlyCompositorUsed": get_path(
            compositor_contract,
            ["fallbackOnlyCompositorUsed"],
        ),
        "debugOutputBypassedForProduction": get_path(
            compositor_contract,
            ["debugOutputBypassedForProduction"],
        ),
        "diagnosticReadbackSeparatedFromRuntimePath": get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        ),
        "visualOutputDegeneratedDetected": get_path(
            compositor_contract,
            ["visualOutputDegeneratedDetected"],
        ),
        "wgslParseErrorDetected": wgsl_parse_error_detected,
        "shaderModuleInvalidDetected": shader_module_invalid_detected,
        "computePipelineInvalidDetected": compute_pipeline_invalid_detected,
        "bindGroupInvalidDetected": bind_group_invalid_detected,
        "webgpuValidationErrorDetected": get_path(
            frame_contract,
            ["webgpuValidationErrorDetected"],
        ),
        "invalidCommandBufferDetected": get_path(
            frame_contract,
            ["invalidCommandBufferDetected"],
        ),
        "queueSubmitFailureDetected": get_path(
            frame_contract,
            ["queueSubmitFailureDetected"],
        ),
        "deferredProductionItems": deferred_production_items,
        "fullRendererSuccessClaimed": get_path(
            frame_contract,
            ["fullRendererSuccessClaimed"],
        ),
    }


def build_step106_capability_based_regression_gate_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    frame_contract = get_path(
        summary,
        ["webgpuTileCompositorFrameImplementation"],
        {},
    )
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    error_subtypes = detect_webgpu_error_subtypes(
        frame_contract,
        compositor_contract,
        get_path(summary, ["captureErrorString"]),
        get_path(summary, ["captureErrorStack"]),
        get_path(summary, ["captureErrorMessage"]),
        get_path(summary, ["firstValidationFailures"]),
    )
    phase_step = get_path(summary, ["phaseStep"])
    wgsl_parse_error_detected = (
        get_path(frame_contract, ["wgslParseErrorDetected"]) is True
        or get_path(compositor_contract, ["wgslParseErrorDetected"]) is True
        or error_subtypes["wgslParseErrorDetected"]
    )
    shader_module_invalid_detected = (
        get_path(frame_contract, ["shaderModuleInvalidDetected"]) is True
        or get_path(compositor_contract, ["shaderModuleInvalidDetected"]) is True
        or error_subtypes["shaderModuleInvalidDetected"]
    )
    compute_pipeline_invalid_detected = (
        get_path(frame_contract, ["computePipelineInvalidDetected"]) is True
        or get_path(compositor_contract, ["computePipelineInvalidDetected"]) is True
        or error_subtypes["computePipelineInvalidDetected"]
    )
    bind_group_invalid_detected = (
        get_path(frame_contract, ["bindGroupInvalidDetected"]) is True
        or get_path(compositor_contract, ["bindGroupInvalidDetected"]) is True
        or error_subtypes["bindGroupInvalidDetected"]
    )
    camera_mismatch_classified = (
        get_path(compositor_contract, ["fixedReferenceCameraGateReady"]) is True
        and get_path(compositor_contract, ["cameraContractMismatchDetected"])
        is True
        and get_path(compositor_contract, ["visualParityComparisonAllowed"])
        is False
        and get_path(compositor_contract, ["visualMismatchClassification"])
        == "camera-contract-mismatch"
    )
    capability_gate_failure_reasons = get_path(
        compositor_contract,
        ["capabilityGateFailureReasons"],
        [],
    )
    if not isinstance(capability_gate_failure_reasons, list):
        capability_gate_failure_reasons = ["capability-failure-reasons-invalid"]
    missing_evidence_detected = (
        get_path(compositor_contract, ["missingEvidenceDetected"]) is True
    )
    missing_evidence_reasons = get_path(
        compositor_contract,
        ["missingEvidenceReasons"],
        [],
    )
    if not isinstance(missing_evidence_reasons, list):
        missing_evidence_reasons = ["missing-evidence-reasons-invalid"]
    contract_source_mismatch_detected = (
        get_path(compositor_contract, ["contractSourceMismatchDetected"]) is True
    )
    contract_source_mismatch_reasons = get_path(
        compositor_contract,
        ["contractSourceMismatchReasons"],
        [],
    )
    if not isinstance(contract_source_mismatch_reasons, list):
        contract_source_mismatch_reasons = [
            "contract-source-mismatch-reasons-invalid"
        ]
    capability_ready = (
        get_path(compositor_contract, ["capabilityBasedRegressionGateReady"])
        is True
        and get_path(compositor_contract, ["presentationCapabilityReady"]) is True
        and get_path(
            compositor_contract,
            ["schedulerDirtyUpdateCapabilityReady"],
        )
        is True
        and get_path(compositor_contract, ["resourceLifecycleCapabilityReady"])
        is True
        and get_path(
            compositor_contract,
            ["cameraReferenceComparisonCapabilityReady"],
        )
        is True
        and get_path(compositor_contract, ["visualSafetyCapabilityReady"]) is True
        and get_path(
            compositor_contract,
            ["diagnosticsReadbackSeparationCapabilityReady"],
        )
        is True
        and capability_gate_failure_reasons == []
        and missing_evidence_detected is False
        and missing_evidence_reasons == []
        and contract_source_mismatch_detected is False
        and contract_source_mismatch_reasons == []
    )
    legacy_mapping_ready = (
        get_path(compositor_contract, ["legacyStepFlagMappingReady"]) is True
        and get_path(compositor_contract, ["legacyFlagToCapabilityCoverageReady"])
        is True
        and get_path(compositor_contract, ["legacyStepFlagsHardBlocker"]) is False
        and get_path(compositor_contract, ["legacyStepFlagsDiagnosticOnly"]) is True
        and isinstance(
            get_path(compositor_contract, ["legacyStepFlagMappingPolicy"]),
            str,
        )
    )
    step105_camera_gate_preserved = (
        get_path(compositor_contract, ["fixedReferenceCameraGateReady"]) is True
        and get_path(
            compositor_contract,
            ["interactiveCameraExcludedFromReferenceComparison"],
        )
        is True
        and (
            get_path(compositor_contract, ["visualParityComparisonAllowed"])
            is True
            or camera_mismatch_classified
        )
    )
    regression_gate_ready = (
        phase_step == "phase3-step106"
        and capability_ready
        and legacy_mapping_ready
        and step105_camera_gate_preserved
        and get_path(compositor_contract, ["earlyTerminationRemainsDisabled"])
        is True
        and get_path(compositor_contract, ["lodStreamingRemainsDisabled"]) is True
        and get_path(compositor_contract, ["fallbackOnlyCompositorUsed"]) is False
        and get_path(compositor_contract, ["debugOutputBypassedForProduction"])
        is True
        and get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        )
        is True
        and get_path(compositor_contract, ["visualOutputDegeneratedDetected"])
        is not True
        and wgsl_parse_error_detected is False
        and shader_module_invalid_detected is False
        and compute_pipeline_invalid_detected is False
        and bind_group_invalid_detected is False
        and get_path(frame_contract, ["invalidCommandBufferDetected"]) is False
        and get_path(frame_contract, ["queueSubmitFailureDetected"]) is False
        and get_path(frame_contract, ["webgpuWebgl2SameFramePresentationMixed"])
        is False
        and get_path(frame_contract, ["fallbackMixingPrevented"]) is True
        and get_path(frame_contract, ["fullRendererSuccessClaimed"]) is False
    )
    blocked_reason = None
    if not regression_gate_ready:
        if phase_step != "phase3-step106":
            blocked_reason = "summary-phase-step-is-not-phase3-step106"
        elif not capability_ready:
            blocked_reason = "capability-regression-gate-not-ready"
        elif not legacy_mapping_ready:
            blocked_reason = "legacy-step-flag-mapping-not-diagnostic-only"
        elif not step105_camera_gate_preserved:
            blocked_reason = "step105-camera-gate-not-preserved"
        elif wgsl_parse_error_detected:
            blocked_reason = "wgsl-parse-error-detected"
        elif shader_module_invalid_detected:
            blocked_reason = "shader-module-invalid-detected"
        elif compute_pipeline_invalid_detected:
            blocked_reason = "compute-pipeline-invalid-detected"
        elif bind_group_invalid_detected:
            blocked_reason = "bind-group-invalid-detected"
        else:
            blocked_reason = "capability-regression-gate-validation-failed"

    deferred_production_items = get_path(
        compositor_contract,
        ["deferredProductionItems"],
        [],
    )
    if not isinstance(deferred_production_items, list):
        deferred_production_items = []
    return {
        "step106Decision": "success" if regression_gate_ready else "blocked",
        "step106BlockedReason": blocked_reason,
        "step106SelectedGoal":
            "A+B+C+D-capability-based-regression-gate-with-E/F/G/H-support",
        "phaseStep": phase_step,
        "step106SummaryApplies": phase_step == "phase3-step106",
        "capabilityBasedGateReviewReady": get_path(
            compositor_contract,
            ["capabilityBasedGateReviewReady"],
        ),
        "capabilityBasedRegressionGateReady": get_path(
            compositor_contract,
            ["capabilityBasedRegressionGateReady"],
        ),
        "presentationCapabilityReady": get_path(
            compositor_contract,
            ["presentationCapabilityReady"],
        ),
        "schedulerDirtyUpdateCapabilityReady": get_path(
            compositor_contract,
            ["schedulerDirtyUpdateCapabilityReady"],
        ),
        "resourceLifecycleCapabilityReady": get_path(
            compositor_contract,
            ["resourceLifecycleCapabilityReady"],
        ),
        "cameraReferenceComparisonCapabilityReady": get_path(
            compositor_contract,
            ["cameraReferenceComparisonCapabilityReady"],
        ),
        "visualSafetyCapabilityReady": get_path(
            compositor_contract,
            ["visualSafetyCapabilityReady"],
        ),
        "diagnosticsReadbackSeparationCapabilityReady": get_path(
            compositor_contract,
            ["diagnosticsReadbackSeparationCapabilityReady"],
        ),
        "canonicalEvidenceSourcesByCapability": get_path(
            compositor_contract,
            ["canonicalEvidenceSourcesByCapability"],
            {},
        ),
        "missingEvidenceDetected": missing_evidence_detected,
        "missingEvidenceReasons": missing_evidence_reasons,
        "contractSourceMismatchDetected": contract_source_mismatch_detected,
        "contractSourceMismatchReasons": contract_source_mismatch_reasons,
        "capabilityGateFailureReasons": capability_gate_failure_reasons,
        "legacyStepFlagMappingReady": get_path(
            compositor_contract,
            ["legacyStepFlagMappingReady"],
        ),
        "legacyFlagToCapabilityCoverageReady": get_path(
            compositor_contract,
            ["legacyFlagToCapabilityCoverageReady"],
        ),
        "legacyStepFlagMappingPolicy": get_path(
            compositor_contract,
            ["legacyStepFlagMappingPolicy"],
        ),
        "legacyStepFlagsHardBlocker": get_path(
            compositor_contract,
            ["legacyStepFlagsHardBlocker"],
        ),
        "legacyStepFlagsDiagnosticOnly": get_path(
            compositor_contract,
            ["legacyStepFlagsDiagnosticOnly"],
        ),
        "legacyStepFlagMapping": get_path(
            compositor_contract,
            ["legacyStepFlagMapping"],
            {},
        ),
        "step105CameraGatePreserved": step105_camera_gate_preserved,
        "fixedReferenceCameraGateReady": get_path(
            compositor_contract,
            ["fixedReferenceCameraGateReady"],
        ),
        "visualParityComparisonAllowed": get_path(
            compositor_contract,
            ["visualParityComparisonAllowed"],
        ),
        "visualMismatchClassification": get_path(
            compositor_contract,
            ["visualMismatchClassification"],
        ),
        "cameraContractMismatchDetected": get_path(
            compositor_contract,
            ["cameraContractMismatchDetected"],
        ),
        "cameraContractMismatchReason": get_path(
            compositor_contract,
            ["cameraContractMismatchReason"],
        ),
        "nextStepRecommendedGoal": get_path(
            compositor_contract,
            ["step106NextStepRecommendedGoal"],
            get_path(compositor_contract, ["step105NextStepRecommendedGoal"]),
        ),
        "earlyTerminationRemainsDisabled": get_path(
            compositor_contract,
            ["earlyTerminationRemainsDisabled"],
        ),
        "lodStreamingRemainsDisabled": get_path(
            compositor_contract,
            ["lodStreamingRemainsDisabled"],
        ),
        "fallbackOnlyCompositorUsed": get_path(
            compositor_contract,
            ["fallbackOnlyCompositorUsed"],
        ),
        "debugOutputBypassedForProduction": get_path(
            compositor_contract,
            ["debugOutputBypassedForProduction"],
        ),
        "diagnosticReadbackSeparatedFromRuntimePath": get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        ),
        "visualOutputDegeneratedDetected": get_path(
            compositor_contract,
            ["visualOutputDegeneratedDetected"],
        ),
        "wgslParseErrorDetected": wgsl_parse_error_detected,
        "shaderModuleInvalidDetected": shader_module_invalid_detected,
        "computePipelineInvalidDetected": compute_pipeline_invalid_detected,
        "bindGroupInvalidDetected": bind_group_invalid_detected,
        "webgpuValidationErrorDetected": get_path(
            frame_contract,
            ["webgpuValidationErrorDetected"],
        ),
        "invalidCommandBufferDetected": get_path(
            frame_contract,
            ["invalidCommandBufferDetected"],
        ),
        "queueSubmitFailureDetected": get_path(
            frame_contract,
            ["queueSubmitFailureDetected"],
        ),
        "deferredProductionItems": deferred_production_items,
        "fullRendererSuccessClaimed": get_path(
            frame_contract,
            ["fullRendererSuccessClaimed"],
        ),
    }


def build_step107_fixed_reference_camera_contract_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    frame_contract = get_path(
        summary,
        ["webgpuTileCompositorFrameImplementation"],
        {},
    )
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    error_subtypes = detect_webgpu_error_subtypes(
        frame_contract,
        compositor_contract,
        get_path(summary, ["captureErrorString"]),
        get_path(summary, ["captureErrorStack"]),
        get_path(summary, ["captureErrorMessage"]),
        get_path(summary, ["firstValidationFailures"]),
    )
    phase_step = get_path(summary, ["phaseStep"])
    wgsl_parse_error_detected = (
        get_path(frame_contract, ["wgslParseErrorDetected"]) is True
        or get_path(compositor_contract, ["wgslParseErrorDetected"]) is True
        or error_subtypes["wgslParseErrorDetected"]
    )
    shader_module_invalid_detected = (
        get_path(frame_contract, ["shaderModuleInvalidDetected"]) is True
        or get_path(compositor_contract, ["shaderModuleInvalidDetected"]) is True
        or error_subtypes["shaderModuleInvalidDetected"]
    )
    compute_pipeline_invalid_detected = (
        get_path(frame_contract, ["computePipelineInvalidDetected"]) is True
        or get_path(compositor_contract, ["computePipelineInvalidDetected"]) is True
        or error_subtypes["computePipelineInvalidDetected"]
    )
    bind_group_invalid_detected = (
        get_path(frame_contract, ["bindGroupInvalidDetected"]) is True
        or get_path(compositor_contract, ["bindGroupInvalidDetected"]) is True
        or error_subtypes["bindGroupInvalidDetected"]
    )
    camera_mismatch_classified = (
        get_path(compositor_contract, ["cameraContractMismatchDetected"]) is True
        and get_path(compositor_contract, ["visualParityComparisonAllowed"])
        is False
        and get_path(compositor_contract, ["visualMismatchClassification"])
        == "camera-contract-mismatch"
    )
    responsibility_ready = (
        get_path(compositor_contract, ["cameraResponsibilityBoundaryReady"])
        is True
        and isinstance(
            get_path(
                compositor_contract,
                ["threeJsInteractiveCameraResponsibility"],
            ),
            str,
        )
        and isinstance(
            get_path(
                compositor_contract,
                ["fixedReferenceCameraResponsibility"],
            ),
            str,
        )
        and isinstance(
            get_path(
                compositor_contract,
                ["webgpuBackendCameraResponsibility"],
            ),
            str,
        )
        and isinstance(
            get_path(compositor_contract, ["toolsCameraResponsibility"]),
            str,
        )
    )
    canonical_sources = get_path(
        compositor_contract,
        ["cameraCanonicalSources"],
        {},
    )
    canonical_sources_ready = (
        get_path(compositor_contract, ["cameraCanonicalSourcesReady"]) is True
        and isinstance(canonical_sources, dict)
        and isinstance(canonical_sources.get("cameraMetadata"), str)
        and isinstance(canonical_sources.get("viewMatrix"), str)
        and isinstance(canonical_sources.get("projectionMatrix"), str)
        and isinstance(canonical_sources.get("viewport"), str)
        and isinstance(canonical_sources.get("pixelCoordinates"), str)
    )
    coordinate_convention_summary = get_path(
        compositor_contract,
        ["coordinateConventionSummary"],
        {},
    )
    coordinate_convention_ready = (
        get_path(compositor_contract, ["coordinateConventionReady"]) is True
        and isinstance(coordinate_convention_summary, dict)
        and isinstance(coordinate_convention_summary.get("axisConvention"), str)
        and isinstance(coordinate_convention_summary.get("handedness"), str)
        and isinstance(coordinate_convention_summary.get("yFlipPolicy"), str)
        and isinstance(
            coordinate_convention_summary.get("pixelCoordinateOrigin"),
            str,
        )
    )
    webgpu_camera_constants_contract = get_path(
        compositor_contract,
        ["webgpuCameraConstantsContract"],
        {},
    )
    webgpu_camera_constants_ready = (
        get_path(compositor_contract, ["webgpuCameraConstantsContractReady"])
        is True
        and isinstance(webgpu_camera_constants_contract, dict)
        and isinstance(webgpu_camera_constants_contract.get("consumer"), str)
        and isinstance(webgpu_camera_constants_contract.get("source"), str)
    )
    fixed_reference_contract_ready = (
        get_path(compositor_contract, ["fixedReferenceCameraContractReady"])
        is True
        and get_path(
            compositor_contract,
            ["fixedReferenceCameraContractReviewReady"],
        )
        is True
        and get_path(compositor_contract, ["fixedReferenceCameraGateReady"])
        is True
        and get_path(
            compositor_contract,
            ["interactiveAndFixedCameraSeparated"],
        )
        is True
        and responsibility_ready
        and canonical_sources_ready
        and coordinate_convention_ready
        and webgpu_camera_constants_ready
        and (
            get_path(compositor_contract, ["visualParityComparisonAllowed"])
            is True
            or camera_mismatch_classified
        )
    )
    fixed_reference_camera_activation_ready = (
        get_path(
            compositor_contract,
            ["usesCudaAlignedFixedReferenceCamera"],
        )
        is True
    )
    visual_parity_comparison_gate_ready = (
        get_path(compositor_contract, ["visualParityComparisonAllowed"]) is True
        or camera_mismatch_classified
    )
    visual_parity_comparison_blocked_reason = None
    if get_path(compositor_contract, ["visualParityComparisonAllowed"]) is not True:
        if camera_mismatch_classified:
            visual_parity_comparison_blocked_reason = get_path(
                compositor_contract,
                ["cameraContractMismatchReason"],
                "camera-contract-mismatch",
            )
        else:
            visual_parity_comparison_blocked_reason = (
                "visual-parity-comparison-not-allowed-and-unclassified"
            )
    preservation_ready = (
        get_path(compositor_contract, ["capabilityBasedRegressionGateReady"])
        is True
        and get_path(compositor_contract, ["step105CameraGatePreserved"]) is True
        and get_path(compositor_contract, ["earlyTerminationRemainsDisabled"])
        is True
        and get_path(compositor_contract, ["lodStreamingRemainsDisabled"]) is True
    )
    full_renderer_success_claimed = (
        get_path(frame_contract, ["fullRendererSuccessClaimed"]) is True
        or get_path(compositor_contract, ["fullRendererSuccessClaimed"]) is True
    )
    step107_decision_policy = (
        "design-gate-success-requires-fixed-reference-camera-contract-"
        "responsibility-canonical-sources-coordinate-convention-preservation-"
        "and-error-free-runtime; fixed-camera-activation-and-visual-parity-"
        "comparison-allowed-are-reported-gates-not-required-for-design-success"
    )
    step107_validation_predicates = [
        {
            "name": "phase-step-is-step107",
            "gate": "design",
            "ready": phase_step == "phase3-step107",
            "requiredForDecision": True,
            "reason": "summary must be generated for phase3-step107",
        },
        {
            "name": "step107-camera-contract-evidence-ready",
            "gate": "design",
            "ready": (
                get_path(
                    compositor_contract,
                    ["step107CameraContractReady"],
                )
                is True
                or fixed_reference_contract_ready
            ),
            "requiredForDecision": True,
            "reason": "Step107 camera contract evidence must be available from the common aggregate flag or decomposed fixed-reference design gate",
        },
        {
            "name": "fixed-reference-camera-design-gate-ready",
            "gate": "design",
            "ready": fixed_reference_contract_ready,
            "requiredForDecision": True,
            "reason": "canonical camera sources, coordinate convention, and backend constants must be readable",
        },
        {
            "name": "fixed-reference-camera-activation-ready",
            "gate": "activation",
            "ready": fixed_reference_camera_activation_ready,
            "requiredForDecision": False,
            "reason": "usesCudaAlignedFixedReferenceCamera is a follow-up activation gate for reference comparison",
        },
        {
            "name": "visual-parity-comparison-gate-classified",
            "gate": "visual-parity-comparison",
            "ready": visual_parity_comparison_gate_ready,
            "requiredForDecision": True,
            "reason": "comparison must either be allowed or blocked with a camera mismatch classification",
        },
        {
            "name": "step106-and-step105-preserved",
            "gate": "preservation",
            "ready": preservation_ready,
            "requiredForDecision": True,
            "reason": "Step106 capability gate and Step105 camera gate must remain intact",
        },
        {
            "name": "fallback-only-compositor-not-used",
            "gate": "runtime-boundary",
            "ready": get_path(compositor_contract, ["fallbackOnlyCompositorUsed"])
            is False,
            "requiredForDecision": True,
            "reason": "design evidence must not come from fallback-only compositor success",
        },
        {
            "name": "debug-output-bypassed-for-production",
            "gate": "runtime-boundary",
            "ready": get_path(
                compositor_contract,
                ["debugOutputBypassedForProduction"],
            )
            is True,
            "requiredForDecision": True,
            "reason": "production path must remain separated from debug output",
        },
        {
            "name": "diagnostic-readback-separated",
            "gate": "runtime-boundary",
            "ready": get_path(
                compositor_contract,
                ["diagnosticReadbackSeparatedFromRuntimePath"],
            )
            is True,
            "requiredForDecision": True,
            "reason": "diagnostic readback must stay outside the runtime path",
        },
        {
            "name": "visual-output-not-degenerated",
            "gate": "runtime-boundary",
            "ready": get_path(
                compositor_contract,
                ["visualOutputDegeneratedDetected"],
            )
            is not True,
            "requiredForDecision": True,
            "reason": "Step107 must not accept degenerated visual output",
        },
        {
            "name": "wgsl-parse-error-not-detected",
            "gate": "error",
            "ready": wgsl_parse_error_detected is False,
            "requiredForDecision": True,
            "reason": "WGSL parse errors block the design gate",
        },
        {
            "name": "shader-module-invalid-not-detected",
            "gate": "error",
            "ready": shader_module_invalid_detected is False,
            "requiredForDecision": True,
            "reason": "invalid shader modules block the design gate",
        },
        {
            "name": "compute-pipeline-invalid-not-detected",
            "gate": "error",
            "ready": compute_pipeline_invalid_detected is False,
            "requiredForDecision": True,
            "reason": "invalid compute pipelines block the design gate",
        },
        {
            "name": "bind-group-invalid-not-detected",
            "gate": "error",
            "ready": bind_group_invalid_detected is False,
            "requiredForDecision": True,
            "reason": "invalid bind groups block the design gate",
        },
        {
            "name": "invalid-command-buffer-not-detected",
            "gate": "error",
            "ready": get_path(
                frame_contract,
                ["invalidCommandBufferDetected"],
            )
            is False,
            "requiredForDecision": True,
            "reason": "invalid command buffers block the design gate",
        },
        {
            "name": "queue-submit-failure-not-detected",
            "gate": "error",
            "ready": get_path(frame_contract, ["queueSubmitFailureDetected"])
            is False,
            "requiredForDecision": True,
            "reason": "queue submit failures block the design gate",
        },
        {
            "name": "webgpu-webgl2-same-frame-not-mixed",
            "gate": "runtime-boundary",
            "ready": get_path(
                frame_contract,
                ["webgpuWebgl2SameFramePresentationMixed"],
            )
            is not True,
            "requiredForDecision": True,
            "reason": "WebGPU and WebGL2 must not be mixed in the same frame",
        },
        {
            "name": "fallback-mixing-prevented-or-not-detected",
            "gate": "runtime-boundary",
            "ready": get_path(frame_contract, ["fallbackMixingPrevented"]) is True
            or get_path(
                frame_contract,
                ["webgpuWebgl2SameFramePresentationMixed"],
            )
            is not True,
            "requiredForDecision": True,
            "reason": "fallback mixing must be prevented or absent from evidence",
        },
        {
            "name": "full-renderer-success-not-claimed",
            "gate": "scope",
            "ready": full_renderer_success_claimed is False,
            "requiredForDecision": True,
            "reason": "Step107 must not claim full renderer success",
        },
    ]
    step107_failed_predicates = [
        predicate
        for predicate in step107_validation_predicates
        if predicate["requiredForDecision"] is True
        and predicate["ready"] is not True
    ]
    step107_ready = len(step107_failed_predicates) == 0
    blocked_reason = None
    if not step107_ready:
        blocked_reason = step107_failed_predicates[0]["name"]
    blocked_reason_detailed = None
    if step107_failed_predicates:
        first_failed = step107_failed_predicates[0]
        blocked_reason_detailed = (
            f"{first_failed['name']}: {first_failed['reason']}"
        )

    deferred_production_items = get_path(
        compositor_contract,
        ["deferredProductionItems"],
        [],
    )
    if not isinstance(deferred_production_items, list):
        deferred_production_items = []
    return {
        "step107Decision": "success" if step107_ready else "blocked",
        "step107BlockedReason": blocked_reason,
        "step107BlockedReasonDetailed": blocked_reason_detailed,
        "step107DecisionPolicy": step107_decision_policy,
        "step107ValidationPredicates": step107_validation_predicates,
        "step107FailedPredicates": step107_failed_predicates,
        "step107SelectedGoal":
            "A+B+C+D-fixed-reference-camera-coordinate-responsibility-design-gate",
        "phaseStep": phase_step,
        "step107SummaryApplies": phase_step == "phase3-step107",
        "fixedReferenceCameraDesignGateReady": fixed_reference_contract_ready,
        "fixedReferenceCameraActivationReady":
            fixed_reference_camera_activation_ready,
        "visualParityComparisonGateReady":
            visual_parity_comparison_gate_ready,
        "visualParityComparisonBlockedReason":
            visual_parity_comparison_blocked_reason,
        "fixedReferenceCameraContractReviewReady": get_path(
            compositor_contract,
            ["fixedReferenceCameraContractReviewReady"],
        ),
        "fixedReferenceCameraContractReady": get_path(
            compositor_contract,
            ["fixedReferenceCameraContractReady"],
        ),
        "cameraResponsibilityBoundaryReady": get_path(
            compositor_contract,
            ["cameraResponsibilityBoundaryReady"],
        ),
        "threeJsInteractiveCameraResponsibility": get_path(
            compositor_contract,
            ["threeJsInteractiveCameraResponsibility"],
        ),
        "fixedReferenceCameraResponsibility": get_path(
            compositor_contract,
            ["fixedReferenceCameraResponsibility"],
        ),
        "webgpuBackendCameraResponsibility": get_path(
            compositor_contract,
            ["webgpuBackendCameraResponsibility"],
        ),
        "toolsCameraResponsibility": get_path(
            compositor_contract,
            ["toolsCameraResponsibility"],
        ),
        "interactiveAndFixedCameraSeparated": get_path(
            compositor_contract,
            ["interactiveAndFixedCameraSeparated"],
        ),
        "cameraCanonicalSourcesReady": get_path(
            compositor_contract,
            ["cameraCanonicalSourcesReady"],
        ),
        "cameraCanonicalSources": canonical_sources,
        "coordinateConventionReady": get_path(
            compositor_contract,
            ["coordinateConventionReady"],
        ),
        "coordinateConventionSummary": coordinate_convention_summary,
        "webgpuCameraConstantsContractReady": get_path(
            compositor_contract,
            ["webgpuCameraConstantsContractReady"],
        ),
        "webgpuCameraConstantsContract": webgpu_camera_constants_contract,
        "fixedReferenceCaptureEvidenceReady": get_path(
            compositor_contract,
            ["fixedReferenceCaptureEvidenceReady"],
        ),
        "fixedReferenceCameraGateReady": get_path(
            compositor_contract,
            ["fixedReferenceCameraGateReady"],
        ),
        "referenceCameraMode": get_path(
            compositor_contract,
            ["referenceCameraMode"],
        ),
        "usesCudaAlignedFixedReferenceCamera": get_path(
            compositor_contract,
            ["usesCudaAlignedFixedReferenceCamera"],
        ),
        "interactiveCameraExcludedFromReferenceComparison": get_path(
            compositor_contract,
            ["interactiveCameraExcludedFromReferenceComparison"],
        ),
        "cameraMetadataName": get_path(
            compositor_contract,
            ["cameraMetadataName"],
        ),
        "visualParityComparisonAllowed": get_path(
            compositor_contract,
            ["visualParityComparisonAllowed"],
        ),
        "cameraMismatchClassificationRefined": get_path(
            compositor_contract,
            ["cameraMismatchClassificationRefined"],
        ),
        "cameraMismatchClassification": get_path(
            compositor_contract,
            ["cameraMismatchClassification"],
        ),
        "visualMismatchClassification": get_path(
            compositor_contract,
            ["visualMismatchClassification"],
        ),
        "cameraContractMismatchDetected": get_path(
            compositor_contract,
            ["cameraContractMismatchDetected"],
        ),
        "cameraContractMismatchReason": get_path(
            compositor_contract,
            ["cameraContractMismatchReason"],
        ),
        "step106CapabilityGatePreserved": get_path(
            compositor_contract,
            ["capabilityBasedRegressionGateReady"],
        ),
        "step105CameraGatePreserved": get_path(
            compositor_contract,
            ["step105CameraGatePreserved"],
        ),
        "nextStepRecommendedGoal": get_path(
            compositor_contract,
            ["step107NextStepRecommendedGoal"],
        ),
        "earlyTerminationRemainsDisabled": get_path(
            compositor_contract,
            ["earlyTerminationRemainsDisabled"],
        ),
        "lodStreamingRemainsDisabled": get_path(
            compositor_contract,
            ["lodStreamingRemainsDisabled"],
        ),
        "fallbackOnlyCompositorUsed": get_path(
            compositor_contract,
            ["fallbackOnlyCompositorUsed"],
        ),
        "debugOutputBypassedForProduction": get_path(
            compositor_contract,
            ["debugOutputBypassedForProduction"],
        ),
        "diagnosticReadbackSeparatedFromRuntimePath": get_path(
            compositor_contract,
            ["diagnosticReadbackSeparatedFromRuntimePath"],
        ),
        "visualOutputDegeneratedDetected": get_path(
            compositor_contract,
            ["visualOutputDegeneratedDetected"],
        ),
        "wgslParseErrorDetected": wgsl_parse_error_detected,
        "shaderModuleInvalidDetected": shader_module_invalid_detected,
        "computePipelineInvalidDetected": compute_pipeline_invalid_detected,
        "bindGroupInvalidDetected": bind_group_invalid_detected,
        "webgpuValidationErrorDetected": get_path(
            frame_contract,
            ["webgpuValidationErrorDetected"],
        ),
        "invalidCommandBufferDetected": get_path(
            frame_contract,
            ["invalidCommandBufferDetected"],
        ),
        "queueSubmitFailureDetected": get_path(
            frame_contract,
            ["queueSubmitFailureDetected"],
        ),
        "deferredProductionItems": deferred_production_items,
        "fullRendererSuccessClaimed": get_path(
            frame_contract,
            ["fullRendererSuccessClaimed"],
        ),
    }


def build_step108_cuda_reference_camera_evidence_summary(
    summary: dict,
) -> dict:
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    frame_contract = get_path(
        summary,
        ["webgpuTileCompositorFrameImplementation"],
        {},
    )
    error_subtypes = detect_webgpu_error_subtypes(
        frame_contract,
        compositor_contract,
        get_path(summary, ["captureErrorString"]),
        get_path(summary, ["captureErrorStack"]),
        get_path(summary, ["captureErrorMessage"]),
        get_path(summary, ["firstValidationFailures"]),
    )
    phase_step = get_path(summary, ["phaseStep"])
    missing_camera_evidence_reasons = get_path(
        compositor_contract,
        ["missingCameraEvidenceReasons"],
        [],
    )
    if not isinstance(missing_camera_evidence_reasons, list):
        missing_camera_evidence_reasons = []
    activation_blocked_reason = get_path(
        compositor_contract,
        ["fixedReferenceCameraActivationBlockedReason"],
    )
    visual_parity_allowed = get_path(
        compositor_contract,
        ["visualParityComparisonAllowed"],
    )
    visual_parity_blocked_reason = None
    if visual_parity_allowed is not True:
        visual_parity_blocked_reason = (
            get_path(compositor_contract, ["cameraContractMismatchReason"])
            or activation_blocked_reason
            or "visual-parity-comparison-not-allowed"
        )
    step107_design_gate_preserved = (
        get_path(compositor_contract, ["step107DesignGatePreserved"]) is True
        or get_path(compositor_contract, ["step107CameraContractReady"]) is True
        or get_path(compositor_contract, ["fixedReferenceCameraContractReady"])
        is True
    )
    full_renderer_success_claimed = (
        get_path(frame_contract, ["fullRendererSuccessClaimed"]) is True
        or get_path(compositor_contract, ["fullRendererSuccessClaimed"]) is True
    )
    step108_validation_predicates = [
        {
            "name": "phase-step-is-step108",
            "gate": "capture",
            "ready": phase_step == "phase3-step108",
            "requiredForDecision": True,
            "reason": "summary must be generated for phase3-step108",
        },
        {
            "name": "cuda-reference-camera-evidence-missing-or-incomplete",
            "gate": "camera-evidence",
            "ready": get_path(
                compositor_contract,
                ["cudaReferenceCameraEvidenceReady"],
            )
            is True,
            "requiredForDecision": True,
            "reason": "CUDA reference camera metadata, view/projection, viewport, screen-space, and background evidence must be readable",
        },
        {
            "name": "no-missing-camera-evidence",
            "gate": "camera-evidence",
            "ready": (
                get_path(
                    compositor_contract,
                    ["missingCameraEvidenceDetected"],
                )
                is False
                and len(missing_camera_evidence_reasons) == 0
            ),
            "requiredForDecision": True,
            "reason": "camera evidence gaps must block activation gate success",
        },
        {
            "name": "fixed-reference-camera-activation-gate-classified",
            "gate": "activation",
            "ready": (
                get_path(
                    compositor_contract,
                    ["fixedReferenceCameraActivationReady"],
                )
                is True
                or isinstance(activation_blocked_reason, str)
            ),
            "requiredForDecision": True,
            "reason": "activation must either be ready or report a blocked reason",
        },
        {
            "name": "webgpu-camera-constants-source-ready",
            "gate": "backend-camera-constants",
            "ready": isinstance(
                get_path(compositor_contract, ["webgpuCameraConstantsSource"]),
                str,
            ),
            "requiredForDecision": True,
            "reason": "WebGPU backend camera constants source must be named",
        },
        {
            "name": "interactive-camera-separated-from-fixed-reference",
            "gate": "responsibility-boundary",
            "ready": get_path(
                compositor_contract,
                ["interactiveCameraSeparatedFromFixedReference"],
            )
            is True,
            "requiredForDecision": True,
            "reason": "interactive viewer camera must remain separated from fixed reference comparison camera",
        },
        {
            "name": "visual-parity-comparison-gate-classified",
            "gate": "visual-parity-comparison",
            "ready": (
                visual_parity_allowed is True
                or isinstance(visual_parity_blocked_reason, str)
            ),
            "requiredForDecision": True,
            "reason": "visual parity comparison must either be allowed or blocked with a reason",
        },
        {
            "name": "step107-design-gate-preserved",
            "gate": "preservation",
            "ready": step107_design_gate_preserved,
            "requiredForDecision": True,
            "reason": "Step107 fixed reference camera design gate must remain intact",
        },
        {
            "name": "step106-capability-gate-preserved",
            "gate": "preservation",
            "ready": get_path(
                compositor_contract,
                ["capabilityBasedRegressionGateReady"],
            )
            is True,
            "requiredForDecision": True,
            "reason": "Step106 capability gate must remain intact",
        },
        {
            "name": "step105-camera-gate-preserved",
            "gate": "preservation",
            "ready": get_path(compositor_contract, ["step105CameraGatePreserved"])
            is True,
            "requiredForDecision": True,
            "reason": "Step105 camera gate must remain intact",
        },
        {
            "name": "early-termination-remains-disabled",
            "gate": "scope",
            "ready": get_path(
                compositor_contract,
                ["earlyTerminationRemainsDisabled"],
            )
            is True,
            "requiredForDecision": True,
            "reason": "Step108 must not enable early termination",
        },
        {
            "name": "lod-streaming-remains-disabled",
            "gate": "scope",
            "ready": get_path(compositor_contract, ["lodStreamingRemainsDisabled"])
            is True,
            "requiredForDecision": True,
            "reason": "Step108 must not enable LOD or streaming",
        },
        {
            "name": "visual-output-not-degenerated",
            "gate": "runtime-boundary",
            "ready": get_path(
                compositor_contract,
                ["visualOutputDegeneratedDetected"],
            )
            is False,
            "requiredForDecision": True,
            "reason": "Step108 must not accept degenerated visual output",
        },
        {
            "name": "wgsl-parse-error-not-detected",
            "gate": "error",
            "ready": (
                get_path(compositor_contract, ["wgslParseErrorDetected"]) is not True
                and error_subtypes["wgslParseErrorDetected"] is False
            ),
            "requiredForDecision": True,
            "reason": "WGSL parse errors block Step108",
        },
        {
            "name": "shader-module-invalid-not-detected",
            "gate": "error",
            "ready": (
                get_path(compositor_contract, ["shaderModuleInvalidDetected"])
                is not True
                and error_subtypes["shaderModuleInvalidDetected"] is False
            ),
            "requiredForDecision": True,
            "reason": "invalid shader modules block Step108",
        },
        {
            "name": "compute-pipeline-invalid-not-detected",
            "gate": "error",
            "ready": (
                get_path(compositor_contract, ["computePipelineInvalidDetected"])
                is not True
                and error_subtypes["computePipelineInvalidDetected"] is False
            ),
            "requiredForDecision": True,
            "reason": "invalid compute pipelines block Step108",
        },
        {
            "name": "bind-group-invalid-not-detected",
            "gate": "error",
            "ready": (
                get_path(compositor_contract, ["bindGroupInvalidDetected"])
                is not True
                and error_subtypes["bindGroupInvalidDetected"] is False
            ),
            "requiredForDecision": True,
            "reason": "invalid bind groups block Step108",
        },
        {
            "name": "invalid-command-buffer-not-detected",
            "gate": "error",
            "ready": get_path(compositor_contract, ["invalidCommandBufferDetected"])
            is False,
            "requiredForDecision": True,
            "reason": "invalid command buffers block Step108",
        },
        {
            "name": "queue-submit-failure-not-detected",
            "gate": "error",
            "ready": get_path(compositor_contract, ["queueSubmitFailureDetected"])
            is False,
            "requiredForDecision": True,
            "reason": "queue submit failures block Step108",
        },
        {
            "name": "full-renderer-success-not-claimed",
            "gate": "scope",
            "ready": full_renderer_success_claimed is False,
            "requiredForDecision": True,
            "reason": "Step108 must not claim full renderer success",
        },
    ]
    failed_predicates = [
        predicate
        for predicate in step108_validation_predicates
        if predicate.get("requiredForDecision") is True
        and predicate.get("ready") is not True
    ]
    blocked_reason = failed_predicates[0]["name"] if failed_predicates else None
    blocked_reason_detailed = None
    if failed_predicates:
        first_failed = failed_predicates[0]
        blocked_reason_detailed = (
            f"{first_failed['name']}: {first_failed['reason']}"
        )
    step108_ready = len(failed_predicates) == 0
    return {
        "step108Decision": "success" if step108_ready else "blocked",
        "step108BlockedReason": blocked_reason,
        "step108BlockedReasonDetailed": blocked_reason_detailed,
        "step108SelectedGoal":
            "A+B+C-cuda-reference-camera-evidence-and-fixed-reference-activation-gate",
        "phaseStep": phase_step,
        "step108ValidationPredicates": step108_validation_predicates,
        "step108FailedPredicates": failed_predicates,
        "cudaReferenceCameraEvidenceReady": get_path(
            compositor_contract,
            ["cudaReferenceCameraEvidenceReady"],
        ),
        "cudaReferenceCameraEvidenceSources": get_path(
            compositor_contract,
            ["cudaReferenceCameraEvidenceSources"],
            {},
        ),
        "fixedReferenceCameraActivationReady": get_path(
            compositor_contract,
            ["fixedReferenceCameraActivationReady"],
        ),
        "fixedReferenceCameraActivationBlockedReason":
            activation_blocked_reason,
        "cameraMetadataName": get_path(
            compositor_contract,
            ["cameraMetadataName"],
        ),
        "viewMatrixEvidence": get_path(
            compositor_contract,
            ["viewMatrixEvidence"],
            {},
        ),
        "projectionMatrixEvidence": get_path(
            compositor_contract,
            ["projectionMatrixEvidence"],
            {},
        ),
        "viewportEvidence": get_path(
            compositor_contract,
            ["viewportEvidence"],
            {},
        ),
        "screenSpaceConventionEvidence": get_path(
            compositor_contract,
            ["screenSpaceConventionEvidence"],
            {},
        ),
        "backgroundPolicyEvidence": get_path(
            compositor_contract,
            ["backgroundPolicyEvidence"],
            {},
        ),
        "webgpuCameraConstantsSource": get_path(
            compositor_contract,
            ["webgpuCameraConstantsSource"],
        ),
        "interactiveCameraSeparatedFromFixedReference": get_path(
            compositor_contract,
            ["interactiveCameraSeparatedFromFixedReference"],
        ),
        "visualParityComparisonAllowed": visual_parity_allowed,
        "visualParityComparisonBlockedReason":
            visual_parity_blocked_reason,
        "missingCameraEvidenceDetected": get_path(
            compositor_contract,
            ["missingCameraEvidenceDetected"],
        ),
        "missingCameraEvidenceReasons": missing_camera_evidence_reasons,
        "step107DesignGatePreserved": step107_design_gate_preserved,
        "step106CapabilityGatePreserved": get_path(
            compositor_contract,
            ["capabilityBasedRegressionGateReady"],
        ),
        "step105CameraGatePreserved": get_path(
            compositor_contract,
            ["step105CameraGatePreserved"],
        ),
        "nextStepRecommendedGoal": get_path(
            compositor_contract,
            ["step108NextStepRecommendedGoal"],
        ),
        "visualMismatchClassification": get_path(
            compositor_contract,
            ["visualMismatchClassification"],
        ),
        "earlyTerminationRemainsDisabled": get_path(
            compositor_contract,
            ["earlyTerminationRemainsDisabled"],
        ),
        "lodStreamingRemainsDisabled": get_path(
            compositor_contract,
            ["lodStreamingRemainsDisabled"],
        ),
        "visualOutputDegeneratedDetected": get_path(
            compositor_contract,
            ["visualOutputDegeneratedDetected"],
        ),
        "wgslParseErrorDetected": get_path(
            compositor_contract,
            ["wgslParseErrorDetected"],
            error_subtypes["wgslParseErrorDetected"],
        )
        is True
        or error_subtypes["wgslParseErrorDetected"],
        "shaderModuleInvalidDetected": get_path(
            compositor_contract,
            ["shaderModuleInvalidDetected"],
            error_subtypes["shaderModuleInvalidDetected"],
        )
        is True
        or error_subtypes["shaderModuleInvalidDetected"],
        "computePipelineInvalidDetected": get_path(
            compositor_contract,
            ["computePipelineInvalidDetected"],
            error_subtypes["computePipelineInvalidDetected"],
        )
        is True
        or error_subtypes["computePipelineInvalidDetected"],
        "bindGroupInvalidDetected": get_path(
            compositor_contract,
            ["bindGroupInvalidDetected"],
            error_subtypes["bindGroupInvalidDetected"],
        )
        is True
        or error_subtypes["bindGroupInvalidDetected"],
        "webgpuValidationErrorDetected": get_path(
            compositor_contract,
            ["webgpuValidationErrorDetected"],
        ),
        "invalidCommandBufferDetected": get_path(
            compositor_contract,
            ["invalidCommandBufferDetected"],
        ),
        "queueSubmitFailureDetected": get_path(
            compositor_contract,
            ["queueSubmitFailureDetected"],
        ),
        "deferredProductionItems": get_path(
            compositor_contract,
            ["deferredProductionItems"],
            [],
        ),
        "fullRendererSuccessClaimed": full_renderer_success_claimed,
    }


def build_step109_fixed_reference_camera_activation_summary(
    summary: dict,
) -> dict:
    compositor_contract = get_path(
        summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    frame_contract = get_path(
        summary,
        ["webgpuTileCompositorFrameImplementation"],
        {},
    )
    error_subtypes = detect_webgpu_error_subtypes(
        frame_contract,
        compositor_contract,
        get_path(summary, ["captureErrorString"]),
        get_path(summary, ["captureErrorStack"]),
        get_path(summary, ["captureErrorMessage"]),
        get_path(summary, ["firstValidationFailures"]),
    )
    phase_step = get_path(summary, ["phaseStep"])
    missing_camera_evidence_reasons = get_path(
        compositor_contract,
        ["missingCameraEvidenceReasons"],
        [],
    )
    if not isinstance(missing_camera_evidence_reasons, list):
        missing_camera_evidence_reasons = []
    activation_blocked_reason = get_path(
        compositor_contract,
        ["fixedReferenceCameraActivationBlockedReason"],
    )
    visual_parity_allowed = get_path(
        compositor_contract,
        ["visualParityComparisonAllowed"],
    )
    visual_parity_blocked_reason = None
    if visual_parity_allowed is not True:
        visual_parity_blocked_reason = (
            get_path(compositor_contract, ["cameraContractMismatchReason"])
            or activation_blocked_reason
            or "visual-parity-comparison-not-allowed"
        )
    full_renderer_success_claimed = (
        get_path(frame_contract, ["fullRendererSuccessClaimed"]) is True
        or get_path(compositor_contract, ["fullRendererSuccessClaimed"]) is True
    )
    step106_capability_preserved = (
        get_path(compositor_contract, ["capabilityBasedRegressionGateReady"])
        is True
    )
    step106_capability_failure_reasons = get_path(
        compositor_contract,
        ["capabilityGateFailureReasons"],
        [],
    )
    if not isinstance(step106_capability_failure_reasons, list):
        step106_capability_failure_reasons = [
            "capability-gate-failure-reasons-invalid"
        ]
    step106_missing_evidence_reasons = get_path(
        compositor_contract,
        ["missingEvidenceReasons"],
        [],
    )
    if not isinstance(step106_missing_evidence_reasons, list):
        step106_missing_evidence_reasons = [
            "missing-evidence-reasons-invalid"
        ]
    step106_capability_source = (
        "webgpuTileListCompositorContract.capabilityBasedRegressionGateReady"
        if step106_capability_preserved
        else "step109-capture-diagnostic-capability-preservation"
    )
    step106_capability_failure_reason = None
    if not step106_capability_preserved:
        if step106_capability_failure_reasons:
            step106_capability_failure_reason = "; ".join(
                str(reason) for reason in step106_capability_failure_reasons
            )
        elif step106_missing_evidence_reasons:
            step106_capability_failure_reason = "; ".join(
                str(reason) for reason in step106_missing_evidence_reasons
            )
        else:
            step106_capability_failure_reason = (
                "step106-capability-evidence-not-emitted-for-step109-capture"
            )
    step109_validation_predicates = [
        {
            "name": "phase-step-is-step109",
            "gate": "capture",
            "ready": phase_step == "phase3-step109",
            "requiredForDecision": True,
            "reason": "summary must be generated for phase3-step109",
        },
        {
            "name": "step108-camera-evidence-preserved",
            "gate": "camera-evidence",
            "ready": get_path(
                compositor_contract,
                ["step108CameraEvidencePreserved"],
            )
            is True,
            "requiredForDecision": True,
            "reason": "Step108 CUDA camera evidence must remain complete",
        },
        {
            "name": "fixed-reference-camera-activation-ready",
            "gate": "activation",
            "ready": get_path(
                compositor_contract,
                ["fixedReferenceCameraActivationReady"],
            )
            is True,
            "requiredForDecision": True,
            "reason": "fixed reference camera mode must be explicitly activated",
        },
        {
            "name": "uses-cuda-aligned-fixed-reference-camera",
            "gate": "activation",
            "ready": get_path(
                compositor_contract,
                ["usesCudaAlignedFixedReferenceCamera"],
            )
            is True,
            "requiredForDecision": True,
            "reason": "WebGPU camera constants must come from CUDA-aligned fixed reference camera evidence",
        },
        {
            "name": "interactive-camera-excluded-from-reference-comparison",
            "gate": "responsibility-boundary",
            "ready": get_path(
                compositor_contract,
                ["interactiveCameraExcludedFromReferenceComparison"],
            )
            is True
            and get_path(
                compositor_contract,
                ["interactiveCameraSeparatedFromFixedReference"],
            )
            is True,
            "requiredForDecision": True,
            "reason": "interactive camera must not be the fixed reference comparison source",
        },
        {
            "name": "camera-constants-routing-ready",
            "gate": "backend-camera-constants",
            "ready": get_path(
                compositor_contract,
                ["cameraConstantsRoutingReady"],
            )
            is True,
            "requiredForDecision": True,
            "reason": "CUDA reference camera evidence must be routed to WebGPU camera constants",
        },
        {
            "name": "no-missing-camera-evidence",
            "gate": "camera-evidence",
            "ready": (
                get_path(
                    compositor_contract,
                    ["missingCameraEvidenceDetected"],
                )
                is False
                and len(missing_camera_evidence_reasons) == 0
            ),
            "requiredForDecision": True,
            "reason": "activation must not ignore missing camera evidence",
        },
        {
            "name": "visual-parity-comparison-gate-classified",
            "gate": "visual-parity-comparison",
            "ready": (
                visual_parity_allowed is True
                or (
                    isinstance(visual_parity_blocked_reason, str)
                    and visual_parity_blocked_reason
                    != "interactive-camera-active-fixed-reference-camera-required"
                )
            ),
            "requiredForDecision": True,
            "reason": "visual parity comparison must be allowed or blocked by a non-interactive-camera reason after activation",
        },
        {
            "name": "step107-design-gate-preserved",
            "gate": "preservation",
            "ready": get_path(compositor_contract, ["step107DesignGatePreserved"])
            is True,
            "requiredForDecision": True,
            "reason": "Step107 fixed reference design gate must remain intact",
        },
        {
            "name": "step106-capability-gate-preservation-classified",
            "gate": "preservation",
            "ready": (
                step106_capability_preserved
                or isinstance(step106_capability_failure_reason, str)
            ),
            "requiredForDecision": False,
            "reason": (
                "Step106 capability preservation is diagnostic for Step109; "
                "missing capability evidence is reported separately instead of "
                "becoming the Step109 activation blocked reason"
            ),
        },
        {
            "name": "step105-camera-gate-preserved",
            "gate": "preservation",
            "ready": get_path(compositor_contract, ["step105CameraGatePreserved"])
            is True,
            "requiredForDecision": True,
            "reason": "Step105 camera gate must remain intact",
        },
        {
            "name": "early-termination-remains-disabled",
            "gate": "scope",
            "ready": get_path(
                compositor_contract,
                ["earlyTerminationRemainsDisabled"],
            )
            is True,
            "requiredForDecision": True,
            "reason": "Step109 must not enable early termination",
        },
        {
            "name": "lod-streaming-remains-disabled",
            "gate": "scope",
            "ready": get_path(compositor_contract, ["lodStreamingRemainsDisabled"])
            is True,
            "requiredForDecision": True,
            "reason": "Step109 must not enable LOD or streaming",
        },
        {
            "name": "visual-output-not-degenerated",
            "gate": "runtime-boundary",
            "ready": get_path(
                compositor_contract,
                ["visualOutputDegeneratedDetected"],
            )
            is False,
            "requiredForDecision": True,
            "reason": "Step109 must not accept degenerated visual output",
        },
        {
            "name": "wgsl-parse-error-not-detected",
            "gate": "error",
            "ready": (
                get_path(compositor_contract, ["wgslParseErrorDetected"]) is not True
                and error_subtypes["wgslParseErrorDetected"] is False
            ),
            "requiredForDecision": True,
            "reason": "WGSL parse errors block Step109",
        },
        {
            "name": "shader-module-invalid-not-detected",
            "gate": "error",
            "ready": (
                get_path(compositor_contract, ["shaderModuleInvalidDetected"])
                is not True
                and error_subtypes["shaderModuleInvalidDetected"] is False
            ),
            "requiredForDecision": True,
            "reason": "invalid shader modules block Step109",
        },
        {
            "name": "compute-pipeline-invalid-not-detected",
            "gate": "error",
            "ready": (
                get_path(compositor_contract, ["computePipelineInvalidDetected"])
                is not True
                and error_subtypes["computePipelineInvalidDetected"] is False
            ),
            "requiredForDecision": True,
            "reason": "invalid compute pipelines block Step109",
        },
        {
            "name": "bind-group-invalid-not-detected",
            "gate": "error",
            "ready": (
                get_path(compositor_contract, ["bindGroupInvalidDetected"])
                is not True
                and error_subtypes["bindGroupInvalidDetected"] is False
            ),
            "requiredForDecision": True,
            "reason": "invalid bind groups block Step109",
        },
        {
            "name": "invalid-command-buffer-not-detected",
            "gate": "error",
            "ready": get_path(compositor_contract, ["invalidCommandBufferDetected"])
            is False,
            "requiredForDecision": True,
            "reason": "invalid command buffers block Step109",
        },
        {
            "name": "queue-submit-failure-not-detected",
            "gate": "error",
            "ready": get_path(compositor_contract, ["queueSubmitFailureDetected"])
            is False,
            "requiredForDecision": True,
            "reason": "queue submit failures block Step109",
        },
        {
            "name": "full-renderer-success-not-claimed",
            "gate": "scope",
            "ready": full_renderer_success_claimed is False,
            "requiredForDecision": True,
            "reason": "Step109 must not claim full renderer success",
        },
    ]
    failed_predicates = [
        predicate
        for predicate in step109_validation_predicates
        if predicate.get("requiredForDecision") is True
        and predicate.get("ready") is not True
    ]
    blocked_reason = failed_predicates[0]["name"] if failed_predicates else None
    blocked_reason_detailed = None
    if failed_predicates:
        first_failed = failed_predicates[0]
        blocked_reason_detailed = (
            f"{first_failed['name']}: {first_failed['reason']}"
        )
    step109_ready = len(failed_predicates) == 0
    return {
        "step109Decision": "success" if step109_ready else "blocked",
        "step109BlockedReason": blocked_reason,
        "step109BlockedReasonDetailed": blocked_reason_detailed,
        "step109DecisionPolicy": (
            "Step109 success requires phase-step, fixed-reference activation, "
            "CUDA-aligned camera constants routing, camera evidence, output "
            "non-degeneration, and WebGPU error-free evidence. Step106 "
            "capability preservation is reported as diagnostic/source evidence "
            "unless a concrete Step109 capability failure is present."
        ),
        "step109SelectedGoal":
            "A+B+C-fixed-reference-camera-activation-and-webgpu-camera-constants-routing",
        "phaseStep": phase_step,
        "step109ValidationPredicates": step109_validation_predicates,
        "step109FailedPredicates": failed_predicates,
        "fixedReferenceCameraActivationReady": get_path(
            compositor_contract,
            ["fixedReferenceCameraActivationReady"],
        ),
        "fixedReferenceCameraActivationMode": get_path(
            compositor_contract,
            ["fixedReferenceCameraActivationMode"],
        ),
        "fixedReferenceCameraActivationBlockedReason":
            activation_blocked_reason,
        "usesCudaAlignedFixedReferenceCamera": get_path(
            compositor_contract,
            ["usesCudaAlignedFixedReferenceCamera"],
        ),
        "interactiveCameraExcludedFromReferenceComparison": get_path(
            compositor_contract,
            ["interactiveCameraExcludedFromReferenceComparison"],
        ),
        "webgpuCameraConstantsSource": get_path(
            compositor_contract,
            ["webgpuCameraConstantsSource"],
        ),
        "cameraConstantsRoutingReady": get_path(
            compositor_contract,
            ["cameraConstantsRoutingReady"],
        ),
        "cameraMetadataName": get_path(
            compositor_contract,
            ["cameraMetadataName"],
        ),
        "viewMatrixSource": get_path(compositor_contract, ["viewMatrixSource"]),
        "projectionMatrixSource": get_path(
            compositor_contract,
            ["projectionMatrixSource"],
        ),
        "viewportSource": get_path(compositor_contract, ["viewportSource"]),
        "screenSpaceConventionSource": get_path(
            compositor_contract,
            ["screenSpaceConventionSource"],
        ),
        "backgroundPolicySource": get_path(
            compositor_contract,
            ["backgroundPolicySource"],
        ),
        "visualParityComparisonAllowed": visual_parity_allowed,
        "visualParityComparisonBlockedReason":
            visual_parity_blocked_reason,
        "cameraMismatchClassification": get_path(
            compositor_contract,
            ["visualMismatchClassification"],
        ),
        "missingCameraEvidenceDetected": get_path(
            compositor_contract,
            ["missingCameraEvidenceDetected"],
        ),
        "missingCameraEvidenceReasons": missing_camera_evidence_reasons,
        "step108CameraEvidencePreserved": get_path(
            compositor_contract,
            ["step108CameraEvidencePreserved"],
        ),
        "step107DesignGatePreserved": get_path(
            compositor_contract,
            ["step107DesignGatePreserved"],
        ),
        "step106CapabilityGatePreserved": get_path(
            compositor_contract,
            ["capabilityBasedRegressionGateReady"],
        ),
        "step106CapabilityGatePreservationSource": step106_capability_source,
        "step106CapabilityGateFailureReason":
            step106_capability_failure_reason,
        "step106CapabilityGateFailureReasons":
            step106_capability_failure_reasons,
        "step106CapabilityMissingEvidenceReasons":
            step106_missing_evidence_reasons,
        "step105CameraGatePreserved": get_path(
            compositor_contract,
            ["step105CameraGatePreserved"],
        ),
        "nextStepRecommendedGoal": get_path(
            compositor_contract,
            ["step109NextStepRecommendedGoal"],
        ),
        "earlyTerminationRemainsDisabled": get_path(
            compositor_contract,
            ["earlyTerminationRemainsDisabled"],
        ),
        "lodStreamingRemainsDisabled": get_path(
            compositor_contract,
            ["lodStreamingRemainsDisabled"],
        ),
        "visualOutputDegeneratedDetected": get_path(
            compositor_contract,
            ["visualOutputDegeneratedDetected"],
        ),
        "wgslParseErrorDetected": get_path(
            compositor_contract,
            ["wgslParseErrorDetected"],
            error_subtypes["wgslParseErrorDetected"],
        )
        is True
        or error_subtypes["wgslParseErrorDetected"],
        "shaderModuleInvalidDetected": get_path(
            compositor_contract,
            ["shaderModuleInvalidDetected"],
            error_subtypes["shaderModuleInvalidDetected"],
        )
        is True
        or error_subtypes["shaderModuleInvalidDetected"],
        "computePipelineInvalidDetected": get_path(
            compositor_contract,
            ["computePipelineInvalidDetected"],
            error_subtypes["computePipelineInvalidDetected"],
        )
        is True
        or error_subtypes["computePipelineInvalidDetected"],
        "bindGroupInvalidDetected": get_path(
            compositor_contract,
            ["bindGroupInvalidDetected"],
            error_subtypes["bindGroupInvalidDetected"],
        )
        is True
        or error_subtypes["bindGroupInvalidDetected"],
        "webgpuValidationErrorDetected": get_path(
            compositor_contract,
            ["webgpuValidationErrorDetected"],
        ),
        "invalidCommandBufferDetected": get_path(
            compositor_contract,
            ["invalidCommandBufferDetected"],
        ),
        "queueSubmitFailureDetected": get_path(
            compositor_contract,
            ["queueSubmitFailureDetected"],
        ),
        "deferredProductionItems": get_path(
            compositor_contract,
            ["deferredProductionItems"],
            [],
        ),
        "fullRendererSuccessClaimed": full_renderer_success_claimed,
    }


def build_step110_fixed_condition_visual_comparison_summary(
    summary: Dict[str, Any],
    comparison_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    compositor_contract = get_path(summary, ["webgpuTileListCompositorContract"], {})
    phase_step = get_path(summary, ["phaseStep"])
    output_diagnostic = get_path(summary, ["outputCaptureDiagnostic"], {})
    comparison_contract = get_path(
        compositor_contract,
        ["fixedConditionVisualComparisonContract"],
        {},
    )
    comparison_sources = get_path(
        compositor_contract,
        ["fixedConditionVisualComparisonSources"],
        get_path(comparison_contract, ["sources"], {}),
    )
    comparison_conditions = get_path(
        compositor_contract,
        ["fixedConditionVisualComparisonConditions"],
        get_path(comparison_contract, ["conditions"], {}),
    )
    comparison_result_conditions = get_path(
        comparison_result,
        ["comparisonConditions"],
        {},
    )
    reference_image_source = (
        get_path(comparison_result, ["referenceImageSource"])
        or get_path(comparison_result, ["referenceImagePath"])
        or get_path(comparison_sources, ["referenceImage"])
    )
    webgpu_png_source = (
        get_path(comparison_result, ["webgpuPngSource"])
        or get_path(comparison_result, ["webgpuPngPath"])
        or get_path(comparison_sources, ["webgpuPng"])
    )
    reference_image_source_kind = (
        get_path(comparison_result, ["referenceImageSourceKind"])
        or classify_reference_source_kind(reference_image_source)
    )
    webgpu_png_source_kind = (
        get_path(comparison_result, ["webgpuPngSourceKind"])
        or ("webgpu-saved-png" if webgpu_png_source else "unknown")
    )
    def comparison_condition_value(key: str, default: Any = None) -> Any:
        return get_path(
            comparison_conditions,
            [key],
            get_path(comparison_result_conditions, [key], default),
        )

    comparison_executed = comparison_result is not None or (
        get_path(compositor_contract, ["fixedConditionVisualComparisonExecuted"]) is True
    )
    capture_nonblank = get_path(output_diagnostic, ["captureOutputNonblank"])
    presentation_nonblank = get_path(output_diagnostic, ["presentationOutputNonblank"])
    saved_matches_runtime = get_path(output_diagnostic, ["savedPngMatchesRuntimeOutput"])
    visual_degenerated = get_path(
        output_diagnostic,
        ["visualOutputDegeneratedDetected"],
        get_path(compositor_contract, ["visualOutputDegeneratedDetected"]),
    )
    comparison_inputs_ready = (
        get_path(compositor_contract, ["fixedConditionVisualComparisonInputsReady"])
        is True
        or (
            comparison_result is not None
            and get_path(comparison_result, ["comparisonConditionReady"]) is True
        )
    )
    comparison_contract_ready = (
        get_path(compositor_contract, ["fixedConditionVisualComparisonContractReady"])
        is True
        or get_path(comparison_contract, ["implementationReady"]) is True
        or (
            comparison_result is not None
            and reference_image_source is not None
            and webgpu_png_source is not None
        )
    )
    comparison_condition_ready = (
        get_path(comparison_result, ["comparisonConditionReady"]) is True
        if comparison_result is not None
        else comparison_inputs_ready
    )
    fixed_reference_ready = (
        get_path(compositor_contract, ["fixedReferenceCameraActivationReady"]) is True
        and get_path(compositor_contract, ["cameraConstantsRoutingReady"]) is True
        and get_path(compositor_contract, ["usesCudaAlignedFixedReferenceCamera"]) is True
    ) or (
        comparison_condition_value("fixedReferenceCameraMode")
        == "cuda-aligned-fixed-reference-camera"
        and "cuda-reference" in str(
            comparison_condition_value("webgpuCameraConstantsSource", "")
        )
    )
    error_subtypes = detect_webgpu_error_subtypes(
        get_path(summary, ["webgpuTileCompositorFrameImplementation"], {}),
        compositor_contract,
        get_path(summary, ["captureErrorString"]),
        get_path(summary, ["captureErrorStack"]),
        get_path(summary, ["captureErrorMessage"]),
        get_path(summary, ["firstValidationFailures"]),
    )
    no_webgpu_errors = not any(error_subtypes.values())
    if comparison_executed:
        if get_path(comparison_result, ["imageSizeMatch"]) is not True:
            comparison_state = "comparison-blocked"
        elif comparison_condition_ready is not True:
            comparison_state = "comparison-blocked"
        elif get_path(comparison_result, ["sameSha256"]) is True:
            comparison_state = "comparison-completed-with-parity-candidate"
        elif get_path(comparison_result, ["differingPixelCount"], 0) > 0:
            comparison_state = "comparison-completed-with-mismatch"
        else:
            comparison_state = "comparison-executed"
    elif phase_step == "phase3-step110" and capture_nonblank is True:
        comparison_state = "awaiting-comparison-result"
    elif phase_step == "phase3-step110":
        comparison_state = "awaiting-browser-capture"
    else:
        comparison_state = "implementation-ready-awaiting-browser-capture"

    predicate_specs = [
        (
            "phase-step-is-step110",
            phase_step in {None, "phase3-step110"},
            "summary must be generated for phase3-step110 or an implementation-ready dry run",
        ),
        (
            "reference-image-source-ready",
            isinstance(reference_image_source, str) and len(reference_image_source) > 0,
            "Reference image source path must be available",
        ),
        (
            "reference-image-source-kind-classified",
            reference_image_source_kind != "unknown",
            "Reference image source kind must be classified",
        ),
        (
            "webgpu-png-source-ready",
            isinstance(webgpu_png_source, str)
            and len(webgpu_png_source) > 0
            and webgpu_png_source != "pending-browser-capture-png",
            "WebGPU saved PNG source must be available after browser capture",
        ),
        (
            "fixed-reference-camera-routing-ready",
            fixed_reference_ready is True,
            "fixed reference camera activation and CUDA-aligned camera constants routing must be ready",
        ),
        (
            "fixed-condition-comparison-contract-ready",
            comparison_contract_ready is True,
            "fixed-condition comparison contract must be readable",
        ),
        (
            "comparison-inputs-ready",
            comparison_inputs_ready is True,
            "comparison input sources and conditions must be ready",
        ),
        (
            "capture-output-nonblank",
            capture_nonblank is True,
            "saved PNG must be nonblank",
        ),
        (
            "presentation-output-nonblank",
            presentation_nonblank is True,
            "presentation output must be nonblank",
        ),
        (
            "saved-png-matches-runtime-output",
            saved_matches_runtime is True,
            "saved PNG must match runtime output nonblank state",
        ),
        (
            "visual-output-not-degenerated",
            visual_degenerated is not True,
            "visual output degeneration must not be detected",
        ),
        (
            "comparison-executed",
            comparison_executed is True,
            "comparison JSON must be available after browser capture",
        ),
        (
            "comparison-condition-ready",
            comparison_condition_ready is True,
            "comparison conditions must be complete and compatible",
        ),
        (
            "comparison-image-size-match",
            (not comparison_executed)
            or get_path(comparison_result, ["imageSizeMatch"]) is True,
            "Reference and WebGPU PNG image sizes must match",
        ),
        (
            "webgpu-error-free",
            no_webgpu_errors is True,
            "WGSL/WebGPU/command/queue errors must be absent",
        ),
    ]
    failed_predicates = [
        {"name": name, "reason": reason}
        for name, ready, reason in predicate_specs
        if ready is not True
    ]
    blocked_reasons = [item["name"] for item in failed_predicates]

    implementation_ready = (
        fixed_reference_ready is True
        and comparison_contract_ready is True
        and no_webgpu_errors is True
    )
    if comparison_executed:
        decision = "success" if not blocked_reasons else "blocked"
    elif implementation_ready:
        decision = "pending"
    else:
        decision = "blocked"
    classification = (
        get_path(comparison_result, ["visualMismatchClassification"])
        if comparison_result is not None
        else get_path(
            compositor_contract,
            ["fixedConditionVisualComparisonClassification"],
            "awaiting-comparison-result"
            if implementation_ready
            else "comparison-condition-mismatch",
        )
    )
    return {
        "step110Decision": decision,
        "step110DecisionState": comparison_state,
        "step110FailedPredicates": failed_predicates,
        "step110BlockedReason": blocked_reasons[0] if blocked_reasons else None,
        "step110BlockedReasons": blocked_reasons,
        "step110SelectedGoal":
            "A+B+C+D-fixed-condition-visual-comparison-readiness-infrastructure",
        "phaseStep": phase_step,
        "fixedConditionVisualComparisonImplementationReady": implementation_ready,
        "fixedConditionVisualComparisonContractReady": comparison_contract_ready,
        "fixedConditionVisualComparisonInputsReady": comparison_inputs_ready,
        "fixedConditionVisualComparisonAwaitingBrowserCapture":
            not comparison_executed and phase_step != "phase3-step110",
        "fixedConditionVisualComparisonAwaitingComparisonResult":
            not comparison_executed,
        "fixedConditionVisualComparisonExecuted": comparison_executed,
        "fixedConditionVisualComparisonContract": comparison_contract,
        "referenceImageSource": reference_image_source,
        "referenceImageSourcePath": reference_image_source,
        "referenceImageSourceKind": reference_image_source_kind,
        "webgpuPngSource": webgpu_png_source,
        "webgpuPngSourceKind": webgpu_png_source_kind,
        "cameraLabel": comparison_condition_value("cameraLabel"),
        "frameLabel": comparison_condition_value("frameLabel"),
        "datasetTime": comparison_condition_value("datasetTime"),
        "imageResolution": comparison_condition_value("imageResolution"),
        "viewport": comparison_condition_value("viewport"),
        "backgroundPolicy": comparison_condition_value("backgroundPolicy"),
        "colorSpacePolicy": comparison_condition_value("colorSpacePolicy"),
        "pixelOrigin": comparison_condition_value("pixelOrigin"),
        "yCoordinateConvention": get_path(
            comparison_conditions,
            ["yCoordinateConvention"],
            get_path(comparison_result_conditions, ["yCoordinateConvention"]),
        ),
        "screenSpaceConvention": get_path(
            comparison_conditions,
            ["screenSpaceConvention"],
            get_path(comparison_result_conditions, ["screenSpaceConvention"]),
        ),
        "captureSource": get_path(
            comparison_sources,
            ["captureSource"],
            get_path(comparison_result_conditions, ["captureSource"]),
        ),
        "fixedReferenceCameraMode": get_path(
            comparison_conditions,
            ["fixedReferenceCameraMode"],
            get_path(comparison_result_conditions, ["fixedReferenceCameraMode"]),
        ),
        "webgpuCameraConstantsSource": get_path(
            comparison_conditions,
            ["webgpuCameraConstantsSource"],
            get_path(
                comparison_result_conditions,
                ["webgpuCameraConstantsSource"],
                get_path(compositor_contract, ["webgpuCameraConstantsSource"]),
            ),
        ),
        "usesCudaAlignedFixedReferenceCamera": (
            get_path(compositor_contract, ["usesCudaAlignedFixedReferenceCamera"])
            is True
            or comparison_condition_value("fixedReferenceCameraMode")
            == "cuda-aligned-fixed-reference-camera"
        ),
        "cameraConstantsRoutingReady": (
            get_path(compositor_contract, ["cameraConstantsRoutingReady"]) is True
            or "cuda-reference" in str(
                comparison_condition_value("webgpuCameraConstantsSource", "")
            )
        ),
        "runtimePresentationPngConsistency": get_path(
            comparison_conditions,
            ["runtimePresentationPngConsistency"],
            get_path(
                comparison_result_conditions,
                ["runtimePresentationPngConsistency"],
                {},
            ),
        ),
        "pngCaptured": get_path(output_diagnostic, ["pngCaptured"]),
        "pngNonzeroRgb": get_path(output_diagnostic, ["pngNonzeroRgb"]),
        "pngNonblackRatio": get_path(output_diagnostic, ["pngNonblackRatio"]),
        "pngSha256": get_path(output_diagnostic, ["pngSha256"]),
        "captureOutputNonblank": capture_nonblank,
        "presentationOutputNonblank": presentation_nonblank,
        "savedPngMatchesRuntimeOutput": saved_matches_runtime,
        "visualOutputDegeneratedDetected": visual_degenerated,
        "comparisonResult": comparison_result or {},
        "comparisonImageSizeMatch": get_path(comparison_result, ["imageSizeMatch"]),
        "comparisonChannelMode": get_path(
            comparison_result,
            ["comparisonChannelMode"],
            get_path(compositor_contract, ["comparisonChannelMode"]),
        ),
        "pixelMae": get_path(comparison_result, ["pixelMae"]),
        "pixelRmse": get_path(comparison_result, ["pixelRmse"]),
        "maxAbsDifference": get_path(comparison_result, ["maxAbsDifference"]),
        "differingPixelCount": get_path(comparison_result, ["differingPixelCount"]),
        "differingPixelRatio": get_path(comparison_result, ["differingPixelRatio"]),
        "referenceImageStats": get_path(comparison_result, ["referenceImageStats"]),
        "webgpuImageStats": get_path(comparison_result, ["webgpuImageStats"]),
        "visualMismatchClassification": classification,
        "comparisonInfrastructureReady": decision == "success",
        "visualParityClaimed": False,
        "visualParityAchieved": (
            get_path(comparison_result, ["sameSha256"]) is True
            and classification == "parity-candidate"
        ),
        "visualParityStatus": (
            "parity-candidate-not-claimed"
            if get_path(comparison_result, ["sameSha256"]) is True
            else (
                "not-achieved"
                if comparison_executed
                else "not-evaluated"
            )
        ),
        "fallbackOnlySuccessUsed": False,
        "debugOnlySuccessUsed": False,
        "webgpuWebgl2MixedFrameDetected": False,
        "step109FixedReferenceCameraActivationPreserved": fixed_reference_ready,
        "step108CameraEvidencePreserved": get_path(
            compositor_contract,
            ["step108CameraEvidencePreserved"],
        ),
        "step107DesignGatePreserved": get_path(
            compositor_contract,
            ["step107DesignGatePreserved"],
        ),
        "step106CapabilityGatePreservationSource": get_path(
            compositor_contract,
            ["step106CapabilityGatePreservationSource"],
        ),
        "step105CameraGatePreserved": get_path(
            compositor_contract,
            ["step105CameraGatePreserved"],
        ),
        "earlyTerminationRemainsDisabled": get_path(
            compositor_contract,
            ["earlyTerminationRemainsDisabled"],
        ),
        "lodStreamingRemainsDisabled": get_path(
            compositor_contract,
            ["lodStreamingRemainsDisabled"],
        ),
        **error_subtypes,
        "deferredProductionItems": get_path(
            compositor_contract,
            ["deferredProductionItems"],
            [],
        ),
        "nextStepRecommendedGoal": get_path(
            compositor_contract,
            ["step110NextStepRecommendedGoal"],
            "run-fixed-condition-visual-comparison-after-browser-capture-v1",
        ),
    }


def build_step111_pipeline_parity_gap_closure_summary(
    summary: Dict[str, Any],
    comparison_result: Optional[Dict[str, Any]] = None,
    step110_comparison_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    compositor_contract = get_path(summary, ["webgpuTileListCompositorContract"], {})
    phase_step = get_path(summary, ["phaseStep"])
    production_used = (
        get_path(compositor_contract, ["step111ProductionRuntimeGapClosureUsed"])
        is True
    )
    consumption_evidence = get_path(
        compositor_contract,
        ["step111ProductionConsumptionEvidence"],
        {},
    )
    step110_preserved = (
        get_path(compositor_contract, ["fixedConditionVisualComparisonContractReady"])
        is True
        and get_path(compositor_contract, ["fixedConditionVisualComparisonInputsReady"])
        is True
        and get_path(compositor_contract, ["fixedReferenceCameraActivationReady"]) is True
        and get_path(compositor_contract, ["cameraConstantsRoutingReady"]) is True
        and get_path(compositor_contract, ["usesCudaAlignedFixedReferenceCamera"]) is True
    )
    error_subtypes = detect_webgpu_error_subtypes(
        get_path(summary, ["webgpuTileCompositorFrameImplementation"], {}),
        compositor_contract,
        comparison_result,
        get_path(summary, ["captureErrorString"]),
        get_path(summary, ["captureErrorStack"]),
        get_path(summary, ["captureErrorMessage"]),
        get_path(summary, ["firstValidationFailures"]),
    )
    no_webgpu_errors = not any(error_subtypes.values())
    output_diagnostic = get_path(summary, ["outputCaptureDiagnostic"], {})
    comparison_executed = comparison_result is not None
    capture_output_nonblank = get_path(output_diagnostic, ["captureOutputNonblank"])
    presentation_output_nonblank = get_path(
        output_diagnostic,
        ["presentationOutputNonblank"],
    )
    saved_png_matches_runtime = get_path(
        output_diagnostic,
        ["savedPngMatchesRuntimeOutput"],
    )
    saved_png_matches_presented = get_path(
        output_diagnostic,
        ["savedPngMatchesPresentedOutput"],
    )
    orientation_evidence = get_path(
        output_diagnostic,
        ["orientationEvidence"],
        get_path(compositor_contract, ["presentationCaptureOrientationEvidence"], {}),
    )
    freshness_evidence = get_path(
        output_diagnostic,
        ["captureFreshnessEvidence"],
        get_path(compositor_contract, ["captureFreshnessEvidence"], {}),
    )
    orientation_consistency_known = get_path(
        orientation_evidence,
        ["orientationConsistencyKnown"],
    )
    orientation_mismatch_detected = get_path(
        orientation_evidence,
        ["orientationMismatchDetected"],
    )
    capture_freshness_known = get_path(
        freshness_evidence,
        ["captureFreshnessKnown"],
    )
    capture_matches_presented_frame = get_path(
        freshness_evidence,
        ["captureMatchesPresentedFrame"],
    )
    stale_capture_detected = get_path(
        freshness_evidence,
        ["staleCaptureDetected"],
    )
    visual_output_degenerated = get_path(
        output_diagnostic,
        ["visualOutputDegeneratedDetected"],
        get_path(compositor_contract, ["visualOutputDegeneratedDetected"]),
    )
    comparison_predicate_specs = [
        (
            "comparison-executed",
            comparison_executed,
            "Step111 fixed-condition comparison JSON must be available after browser capture",
        ),
        (
            "comparison-image-size-match",
            (not comparison_executed)
            or get_path(comparison_result, ["imageSizeMatch"]) is True,
            "Reference and WebGPU PNG image sizes must match",
        ),
        (
            "capture-output-nonblank",
            capture_output_nonblank is True,
            "saved PNG must be nonblank before comparison metrics can describe rendering output",
        ),
        (
            "presentation-output-nonblank",
            presentation_output_nonblank is True,
            "presentation output must be nonblank",
        ),
        (
            "saved-png-matches-runtime-output",
            saved_png_matches_runtime is True,
            "saved PNG must match runtime/presentation output evidence",
        ),
        (
            "visual-output-not-degenerated",
            visual_output_degenerated is not True,
            "capture/presentation output degeneration must not be detected",
        ),
    ]
    comparison_failed_predicates = [
        {"name": name, "reason": reason}
        for name, ready, reason in comparison_predicate_specs
        if ready is not True
    ]
    comparison_blocked_reasons = [
        item["name"] for item in comparison_failed_predicates
    ]
    metric_change = {
        "pixelMae": comparison_metric_delta(
            comparison_result,
            step110_comparison_result,
            "pixelMae",
        ),
        "pixelRmse": comparison_metric_delta(
            comparison_result,
            step110_comparison_result,
            "pixelRmse",
        ),
        "maxAbsDifference": comparison_metric_delta(
            comparison_result,
            step110_comparison_result,
            "maxAbsDifference",
        ),
        "differingPixelCount": comparison_metric_delta(
            comparison_result,
            step110_comparison_result,
            "differingPixelCount",
        ),
        "differingPixelRatio": comparison_metric_delta(
            comparison_result,
            step110_comparison_result,
            "differingPixelRatio",
        ),
    }
    predicate_specs = [
        (
            "phase-step-is-step111",
            phase_step in {None, "phase3-step111"},
            "summary must be generated for phase3-step111 or an implementation dry run",
        ),
        (
            "pipeline-parity-stage-map-readable",
            isinstance(
                get_path(compositor_contract, ["cudaWebgpuPipelineParityStageMap"]),
                list,
            )
            and len(get_path(compositor_contract, ["cudaWebgpuPipelineParityStageMap"], [])) > 0,
            "CUDA/WebGPU pipeline stage mapping must be readable",
        ),
        (
            "selected-gap-readable",
            isinstance(get_path(compositor_contract, ["step111SelectedGap"]), str),
            "selected structural gap must be readable",
        ),
        (
            "gap-before-after-classification-readable",
            isinstance(
                get_path(compositor_contract, ["step111GapBeforeClassification"]),
                str,
            )
            and isinstance(
                get_path(compositor_contract, ["step111GapAfterClassification"]),
                str,
            ),
            "before/after gap classification must be readable",
        ),
        (
            "production-runtime-gap-closure-consumed",
            production_used,
            "new structural gap closure must be consumed by production runtime",
        ),
        (
            "scale-aware-conic-consumed",
            get_path(compositor_contract, ["scaleAwareConicPayloadConsumed"]) is True
            and numeric_value(
                get_path(compositor_contract, ["anisotropicFootprintReferenceCount"]),
                0,
            )
            > 0,
            "scale-aware anisotropic footprint payload must be consumed",
        ),
        (
            "step110-fixed-condition-comparison-preserved",
            step110_preserved,
            "Step110 fixed reference comparison infrastructure must remain readable",
        ),
        (
            "visual-parity-not-claimed-without-new-capture",
            get_path(compositor_contract, ["fullCudaParity"]) is not True
            and get_path(compositor_contract, ["fullRendererSuccessClaimed"]) is not True,
            "visual/CUDA parity must not be claimed by Step111 implementation alone",
        ),
        (
            "webgpu-error-free",
            no_webgpu_errors,
            "WGSL/WebGPU/command/queue errors must be absent",
        ),
    ]
    failed_predicates = [
        {"name": name, "reason": reason}
        for name, ready, reason in predicate_specs
        if ready is not True
    ]
    blocked_reasons = [item["name"] for item in failed_predicates]
    fix2_predicate_specs = [
        (
            "orientation-consistency-known",
            orientation_consistency_known is True,
            "presentation/capture orientation contract must prove saved PNG matches canonical Viewer presentation orientation",
        ),
        (
            "saved-png-matches-presented-output",
            saved_png_matches_presented is True,
            "saved PNG must match presented output, not only raw production texture",
        ),
        (
            "no-orientation-mismatch",
            orientation_mismatch_detected is False,
            "capture and presentation Y transforms must not diverge",
        ),
        (
            "capture-freshness-known",
            capture_freshness_known is True,
            "capture must identify the current presented camera/time/frame output generation",
        ),
        (
            "capture-matches-presented-frame",
            capture_matches_presented_frame is True,
            "captured output generation must match the presented output generation",
        ),
        (
            "stale-capture-not-detected",
            stale_capture_detected is False,
            "last-valid output capture must not be stale",
        ),
        (
            "step111-gap-closure-preserved",
            len(blocked_reasons) == 0,
            "Step111 structural gap closure must remain successful",
        ),
    ]
    fix2_failed_predicates = [
        {"name": name, "reason": reason}
        for name, ready, reason in fix2_predicate_specs
        if ready is not True
    ]
    fix2_blocked_reasons = [item["name"] for item in fix2_failed_predicates]
    return {
        "step111Decision": "success" if not blocked_reasons else "blocked",
        "step111FailedPredicates": failed_predicates,
        "step111BlockedReason": blocked_reasons[0] if blocked_reasons else None,
        "step111BlockedReasons": blocked_reasons,
        "step111Fix2Decision": "success" if not fix2_blocked_reasons else "blocked",
        "step111Fix2FailedPredicates": fix2_failed_predicates,
        "step111Fix2BlockedReasons": fix2_blocked_reasons,
        "step111SelectedGoal": get_path(
            compositor_contract,
            ["step111SelectedGoal"],
        ),
        "cudaWebgpuPipelineParityStageMap": get_path(
            compositor_contract,
            ["cudaWebgpuPipelineParityStageMap"],
            [],
        ),
        "step111SelectedGap": get_path(compositor_contract, ["step111SelectedGap"]),
        "step111GapBeforeClassification": get_path(
            compositor_contract,
            ["step111GapBeforeClassification"],
        ),
        "step111GapAfterClassification": get_path(
            compositor_contract,
            ["step111GapAfterClassification"],
        ),
        "step111ProductionRuntimeGapClosureUsed": production_used,
        "step111ProductionConsumptionEvidence": consumption_evidence,
        "step111ApproximationReplaced": get_path(
            compositor_contract,
            ["step111ApproximationReplaced"],
        ),
        "step111RemainingApproximations": get_path(
            compositor_contract,
            ["step111RemainingApproximations"],
            [],
        ),
        "step111DeferredStructuralGaps": get_path(
            compositor_contract,
            ["step111DeferredStructuralGaps"],
            [],
        ),
        "scaleAwareConicPayloadConsumed": get_path(
            compositor_contract,
            ["scaleAwareConicPayloadConsumed"],
        ),
        "anisotropicFootprintReferenceCount": get_path(
            compositor_contract,
            ["anisotropicFootprintReferenceCount"],
        ),
        "anisotropicFootprintRatio": get_path(
            compositor_contract,
            ["anisotropicFootprintRatio"],
        ),
        "conicFallbackReferenceCount": get_path(
            compositor_contract,
            ["conicFallbackReferenceCount"],
        ),
        "step110FixedConditionComparisonPreserved": step110_preserved,
        "fixedReferenceCameraActivationReady": get_path(
            compositor_contract,
            ["fixedReferenceCameraActivationReady"],
        ),
        "usesCudaAlignedFixedReferenceCamera": get_path(
            compositor_contract,
            ["usesCudaAlignedFixedReferenceCamera"],
        ),
        "cameraConstantsRoutingReady": get_path(
            compositor_contract,
            ["cameraConstantsRoutingReady"],
        ),
        "step111ComparisonExecuted": comparison_executed,
        "step111ComparisonFailedPredicates": comparison_failed_predicates,
        "step111ComparisonBlockedReason": (
            comparison_blocked_reasons[0] if comparison_blocked_reasons else None
        ),
        "step111ComparisonBlockedReasons": comparison_blocked_reasons,
        "comparisonImageSizeMatch": get_path(
            comparison_result,
            ["imageSizeMatch"],
        ),
        "comparisonChannelMode": get_path(
            comparison_result,
            ["comparisonChannelMode"],
            get_path(compositor_contract, ["comparisonChannelMode"]),
        ),
        "pixelMae": get_path(comparison_result, ["pixelMae"]),
        "pixelRmse": get_path(comparison_result, ["pixelRmse"]),
        "maxAbsDifference": get_path(comparison_result, ["maxAbsDifference"]),
        "differingPixelCount": get_path(
            comparison_result,
            ["differingPixelCount"],
        ),
        "differingPixelRatio": get_path(
            comparison_result,
            ["differingPixelRatio"],
        ),
        "step110ComparisonMetricBaseline": {
            "pixelMae": get_path(step110_comparison_result, ["pixelMae"]),
            "pixelRmse": get_path(step110_comparison_result, ["pixelRmse"]),
            "maxAbsDifference": get_path(
                step110_comparison_result,
                ["maxAbsDifference"],
            ),
            "differingPixelCount": get_path(
                step110_comparison_result,
                ["differingPixelCount"],
            ),
            "differingPixelRatio": get_path(
                step110_comparison_result,
                ["differingPixelRatio"],
            ),
        },
        "step110ToStep111MetricChange": metric_change,
        "mismatchClassification": get_path(
            comparison_result,
            ["visualMismatchClassification"],
            get_path(
                output_diagnostic,
                ["visualOutputDegenerationReason"],
                "awaiting-comparison-result",
            ),
        ),
        "captureSourceKind": get_path(output_diagnostic, ["captureSourceKind"]),
        "pngCaptureStatus": get_path(output_diagnostic, ["pngCaptureStatus"], {}),
        "captureOutputNonblank": capture_output_nonblank,
        "presentationOutputNonblank": presentation_output_nonblank,
        "savedPngMatchesRuntimeOutput": saved_png_matches_runtime,
        "savedPngMatchesRawProductionOutput": get_path(
            output_diagnostic,
            ["savedPngMatchesRawProductionOutput"],
        ),
        "savedPngMatchesPresentedOutput": saved_png_matches_presented,
        "orientationEvidence": orientation_evidence,
        "rawProductionOrientation": {
            "productionTextureOrigin": get_path(
                orientation_evidence,
                ["productionTextureOrigin"],
            ),
            "productionTextureYAxisDirection": get_path(
                orientation_evidence,
                ["productionTextureYAxisDirection"],
            ),
        },
        "presentationOrientation": {
            "presentationUvTransform": get_path(
                orientation_evidence,
                ["presentationUvTransform"],
            ),
            "presentationVerticalFlipApplied": get_path(
                orientation_evidence,
                ["presentationVerticalFlipApplied"],
            ),
            "canonicalPresentationOrientation": get_path(
                orientation_evidence,
                ["canonicalPresentationOrientation"],
            ),
        },
        "savedPngOrientation": get_path(
            orientation_evidence,
            ["savedPngOrientation"],
        ),
        "captureReadbackRowOrder": get_path(
            orientation_evidence,
            ["captureReadbackRowOrder"],
        ),
        "pngEncoderRowOrder": get_path(
            orientation_evidence,
            ["pngEncoderRowOrder"],
        ),
        "captureVerticalFlipApplied": get_path(
            orientation_evidence,
            ["captureVerticalFlipApplied"],
        ),
        "captureMatchesCanonicalPresentationOrientation": get_path(
            orientation_evidence,
            ["captureMatchesCanonicalPresentationOrientation"],
        ),
        "orientationConsistencyKnown": orientation_consistency_known,
        "orientationMismatchDetected": orientation_mismatch_detected,
        "orientationMismatchClassification": get_path(
            orientation_evidence,
            ["orientationMismatchClassification"],
        ),
        "captureFreshnessEvidence": freshness_evidence,
        "captureFreshnessKnown": capture_freshness_known,
        "requestedStateIdentity": get_path(
            freshness_evidence,
            ["requestedStateIdentity"],
        ),
        "presentedFrameIdentity": get_path(
            freshness_evidence,
            ["presentedFrameIdentity"],
        ),
        "capturedFrameIdentity": get_path(
            freshness_evidence,
            ["capturedFrameIdentity"],
        ),
        "captureMatchesPresentedFrame": capture_matches_presented_frame,
        "captureMatchesRequestedState": get_path(
            freshness_evidence,
            ["captureMatchesRequestedState"],
        ),
        "captureVsPresentedFrameIdentity": get_path(
            freshness_evidence,
            ["captureVsPresentedFrameIdentity"],
        ),
        "captureVsRequestedStateIdentity": get_path(
            freshness_evidence,
            ["captureVsRequestedStateIdentity"],
        ),
        "capturePresentedFrameMismatchedFields": get_path(
            freshness_evidence,
            ["capturePresentedFrameMismatchedFields"],
            [],
        ),
        "capturePresentedFrameMissingFields": get_path(
            freshness_evidence,
            ["capturePresentedFrameMissingFields"],
            [],
        ),
        "captureRequestedStateMismatchedFields": get_path(
            freshness_evidence,
            ["captureRequestedStateMismatchedFields"],
            [],
        ),
        "captureRequestedStateMissingFields": get_path(
            freshness_evidence,
            ["captureRequestedStateMissingFields"],
            [],
        ),
        "staleCaptureDetected": stale_capture_detected,
        "productionOutputGeneration": get_path(
            freshness_evidence,
            ["productionOutputGeneration"],
        ),
        "presentedOutputGeneration": get_path(
            freshness_evidence,
            ["presentedOutputGeneration"],
        ),
        "capturedOutputGeneration": get_path(
            freshness_evidence,
            ["capturedOutputGeneration"],
        ),
        "captureFreshnessClassification": get_path(
            freshness_evidence,
            ["captureFreshnessClassification"],
        ),
        "orientationDiagnostic": get_path(
            comparison_result,
            ["orientationDiagnostic"],
            {},
        ),
        "normalOrientationMae": get_path(
            comparison_result,
            ["orientationDiagnostic.normal.mae"],
        ),
        "normalOrientationRmse": get_path(
            comparison_result,
            ["orientationDiagnostic.normal.rmse"],
        ),
        "normalOrientationDifferingRatio": get_path(
            comparison_result,
            ["orientationDiagnostic.normal.differentPixelRatioAnyChannel"],
        ),
        "verticalFlipOrientationMae": get_path(
            comparison_result,
            ["orientationDiagnostic.verticalFlip.mae"],
        ),
        "verticalFlipOrientationRmse": get_path(
            comparison_result,
            ["orientationDiagnostic.verticalFlip.rmse"],
        ),
        "verticalFlipOrientationDifferingRatio": get_path(
            comparison_result,
            ["orientationDiagnostic.verticalFlip.differentPixelRatioAnyChannel"],
        ),
        "lowerErrorOrientation": get_path(
            comparison_result,
            ["orientationDiagnostic.lowerErrorOrientation"],
        ),
        "orientationDiagnosticClassification": get_path(
            comparison_result,
            ["orientationDiagnostic.classification"],
        ),
        "visualOutputDegeneratedDetected": visual_output_degenerated,
        "visualOutputDegenerationReason": get_path(
            output_diagnostic,
            ["visualOutputDegenerationReason"],
        ),
        "visualOutputDegenerationClassification": get_path(
            output_diagnostic,
            ["visualOutputDegenerationClassification"],
        ),
        "visualParityClaimed": get_path(compositor_contract, ["fullCudaParity"]) is True,
        "visualParityAchieved": (
            comparison_executed
            and get_path(comparison_result, ["sameSha256"]) is True
            and get_path(comparison_result, ["visualMismatchClassification"])
            == "parity-candidate"
        ),
        "fullRendererSuccessClaimed": get_path(
            compositor_contract,
            ["fullRendererSuccessClaimed"],
        ),
        "earlyTerminationRemainsDisabled": get_path(
            compositor_contract,
            ["earlyTerminationRemainsDisabled"],
        ),
        "lodStreamingRemainsDisabled": get_path(
            compositor_contract,
            ["lodStreamingRemainsDisabled"],
        ),
        "nextStepRecommendedGoal": get_path(
            compositor_contract,
            ["step111NextStepRecommendedGoal"],
        ),
        **error_subtypes,
    }


def build_step112_camera_projection_orientation_parity_summary(
    summary: Dict[str, Any],
    comparison_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    compositor_contract = get_path(summary, ["webgpuTileListCompositorContract"], {})
    phase_step = get_path(summary, ["phaseStep"])
    representative = get_path(
        compositor_contract,
        ["step112RepresentativePointComparison"],
        {},
    )
    representative_points = get_path(representative, ["points"], [])
    representative_ready = (
        get_path(representative, ["ready"]) is True
        and isinstance(representative_points, list)
        and len(representative_points) >= 3
    )
    first_mismatch_stage = get_path(
        compositor_contract,
        ["step112FirstMismatchStage"],
    )
    center_projection_ready = (
        get_path(compositor_contract, ["step112CenterProjectionParityReady"])
        is True
    )
    view_projection_pixel_ready = (
        get_path(compositor_contract, ["step112ViewProjectionPixelParityReady"])
        is True
    )
    conic_convention_consistent = (
        get_path(
            compositor_contract,
            ["step112CenterProjectionConicConventionConsistent"],
        )
        is True
    )
    production_consumption_ready = (
        get_path(compositor_contract, ["step112ProductionRuntimeConsumptionReady"])
        is True
    )
    step110_preserved = (
        get_path(compositor_contract, ["fixedConditionVisualComparisonContractReady"])
        is True
        and get_path(compositor_contract, ["fixedConditionVisualComparisonInputsReady"])
        is True
        and get_path(compositor_contract, ["fixedReferenceCameraActivationReady"]) is True
        and get_path(compositor_contract, ["cameraConstantsRoutingReady"]) is True
        and get_path(compositor_contract, ["usesCudaAlignedFixedReferenceCamera"]) is True
    )
    step111_preserved = (
        get_path(compositor_contract, ["step111ProductionRuntimeGapClosureUsed"])
        is True
        and get_path(compositor_contract, ["scaleAwareConicPayloadConsumed"]) is True
        and numeric_value(
            get_path(compositor_contract, ["anisotropicFootprintReferenceCount"]),
            0,
        )
        > 0
        and get_path(compositor_contract, ["conicFallbackReferenceCount"]) == 0
    )
    output_diagnostic = get_path(summary, ["outputCaptureDiagnostic"], {})
    capture_preserved = (
        get_path(output_diagnostic, ["captureOutputNonblank"]) is True
        and get_path(output_diagnostic, ["presentationOutputNonblank"]) is True
        and get_path(output_diagnostic, ["savedPngMatchesRuntimeOutput"]) is True
        and get_path(output_diagnostic, ["visualOutputDegeneratedDetected"]) is not True
    )
    error_subtypes = detect_webgpu_error_subtypes(
        get_path(summary, ["webgpuTileCompositorFrameImplementation"], {}),
        compositor_contract,
        comparison_result,
        get_path(summary, ["captureErrorString"]),
        get_path(summary, ["captureErrorStack"]),
        get_path(summary, ["captureErrorMessage"]),
        get_path(summary, ["firstValidationFailures"]),
    )
    no_webgpu_errors = not any(error_subtypes.values())
    visual_parity_claimed = (
        get_path(compositor_contract, ["fullCudaParity"]) is True
        or get_path(compositor_contract, ["fullRendererSuccessClaimed"]) is True
        or (
            comparison_result is not None
            and get_path(comparison_result, ["visualMismatchClassification"])
            == "parity-candidate"
        )
    )
    predicate_specs = [
        (
            "phase-step-is-step112",
            phase_step in {None, "phase3-step112"},
            "summary must be generated for phase3-step112 or an implementation dry run",
        ),
        (
            "selected-candidates-readable",
            len(get_path(compositor_contract, ["step112SelectedCandidates"], [])) > 0,
            "Step112 selected candidate list must be readable",
        ),
        (
            "coordinate-stage-map-readable",
            len(get_path(compositor_contract, ["step112CoordinateTransformStageMap"], []))
            > 0,
            "world/camera/clip/pixel/visible-record/compositor stage map must be readable",
        ),
        (
            "canonical-camera-projection-sources-readable",
            isinstance(
                get_path(
                    compositor_contract,
                    ["step112CanonicalCameraProjectionSources"],
                ),
                dict,
            )
            and len(
                get_path(
                    compositor_contract,
                    ["step112CanonicalCameraProjectionSources"],
                    {},
                )
            )
            > 0,
            "canonical camera/projection/viewport sources must be readable",
        ),
        (
            "coordinate-convention-readable",
            isinstance(
                get_path(compositor_contract, ["step112CoordinateConvention"]),
                dict,
            )
            and len(get_path(compositor_contract, ["step112CoordinateConvention"], {}))
            > 0,
            "coordinate convention summary must be readable",
        ),
        (
            "representative-point-comparison-ready",
            representative_ready,
            "at least three non-collinear representative point projection comparisons must be readable",
        ),
        (
            "view-projection-pixel-parity-ready",
            view_projection_pixel_ready,
            "CUDA canonical center projection must match WebGPU visible-record center projection",
        ),
        (
            "center-projection-parity-ready",
            center_projection_ready,
            "center projection parity must be established before classifying rendering differences",
        ),
        (
            "no-representative-first-mismatch-stage",
            first_mismatch_stage in {None, "none"},
            "first mismatch stage must be absent for Step112 closure",
        ),
        (
            "center-projection-conic-convention-consistent",
            conic_convention_consistent,
            "center projection evidence and production conic payload must share the same screen-space convention",
        ),
        (
            "production-runtime-consumption-ready",
            production_consumption_ready,
            "Step112 evidence must be consumed by the production tile compositor path",
        ),
        (
            "step110-fixed-condition-comparison-preserved",
            step110_preserved,
            "Step110 fixed reference comparison infrastructure must remain intact",
        ),
        (
            "step111-structural-gap-closure-preserved",
            step111_preserved,
            "Step111 scale-aware anisotropic footprint path must remain active",
        ),
        (
            "capture-presentation-png-preserved",
            capture_preserved,
            "presentation/capture/saved PNG consistency must remain nondegenerate",
        ),
        (
            "visual-parity-not-claimed",
            visual_parity_claimed is False,
            "Step112 must not claim visual/CUDA parity before browser recapture comparison",
        ),
        (
            "webgpu-error-free",
            no_webgpu_errors,
            "WGSL/WebGPU/command/queue errors must be absent",
        ),
    ]
    failed_predicates = [
        {"name": name, "reason": reason}
        for name, ready, reason in predicate_specs
        if ready is not True
    ]
    blocked_reasons = [item["name"] for item in failed_predicates]
    return {
        "step112Decision": "success" if not blocked_reasons else "blocked",
        "step112FailedPredicates": failed_predicates,
        "step112BlockedReason": blocked_reasons[0] if blocked_reasons else None,
        "step112BlockedReasons": blocked_reasons,
        "step112SelectedGoal": get_path(
            compositor_contract,
            ["step112SelectedGoal"],
        ),
        "step112SelectedCandidates": get_path(
            compositor_contract,
            ["step112SelectedCandidates"],
            [],
        ),
        "step112RootCause": get_path(
            compositor_contract,
            ["step112RootCause"],
        ),
        "step112CoordinateTransformStageMap": get_path(
            compositor_contract,
            ["step112CoordinateTransformStageMap"],
            [],
        ),
        "step112CanonicalCameraProjectionSources": get_path(
            compositor_contract,
            ["step112CanonicalCameraProjectionSources"],
            {},
        ),
        "step112CoordinateConvention": get_path(
            compositor_contract,
            ["step112CoordinateConvention"],
            {},
        ),
        "step112RepresentativePointComparison": representative,
        "step112RepresentativePointCount": len(representative_points)
        if isinstance(representative_points, list)
        else 0,
        "step112FirstMismatchStage": first_mismatch_stage,
        "step112ViewProjectionPixelParityReady": view_projection_pixel_ready,
        "step112CenterProjectionParityReady": center_projection_ready,
        "step112CenterProjectionConicConventionConsistent":
            conic_convention_consistent,
        "step112ProductionRuntimeConsumptionReady": production_consumption_ready,
        "step110FixedConditionComparisonPreserved": step110_preserved,
        "step111StructuralGapClosurePreserved": step111_preserved,
        "step111ProductionRuntimeGapClosureUsed": get_path(
            compositor_contract,
            ["step111ProductionRuntimeGapClosureUsed"],
        ),
        "scaleAwareConicPayloadConsumed": get_path(
            compositor_contract,
            ["scaleAwareConicPayloadConsumed"],
        ),
        "anisotropicFootprintReferenceCount": get_path(
            compositor_contract,
            ["anisotropicFootprintReferenceCount"],
        ),
        "conicFallbackReferenceCount": get_path(
            compositor_contract,
            ["conicFallbackReferenceCount"],
        ),
        "capturePresentationPngPreserved": capture_preserved,
        "presentationOutputNonblank": get_path(
            output_diagnostic,
            ["presentationOutputNonblank"],
        ),
        "captureOutputNonblank": get_path(
            output_diagnostic,
            ["captureOutputNonblank"],
        ),
        "savedPngMatchesRuntimeOutput": get_path(
            output_diagnostic,
            ["savedPngMatchesRuntimeOutput"],
        ),
        "visualOutputDegeneratedDetected": get_path(
            output_diagnostic,
            ["visualOutputDegeneratedDetected"],
        ),
        "step112RemainingCameraProjectionGaps": get_path(
            compositor_contract,
            ["step112RemainingCameraProjectionGaps"],
            [],
        ),
        "visualParityClaimed": visual_parity_claimed,
        "visualParityAchieved": False,
        "earlyTerminationRemainsDisabled": get_path(
            compositor_contract,
            ["earlyTerminationRemainsDisabled"],
        ),
        "lodStreamingRemainsDisabled": get_path(
            compositor_contract,
            ["lodStreamingRemainsDisabled"],
        ),
        "nextStepRecommendedGoal": get_path(
            compositor_contract,
            ["step112NextStepRecommendedGoal"],
        ),
        **error_subtypes,
    }


def build_output_capture_consistency_diagnostic(
    webgpu_summary: Dict[str, Any],
    png_diagnostic: Dict[str, Any],
    png_capture_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    compositor_contract = get_path(
        webgpu_summary,
        ["webgpuTileListCompositorContract"],
        {},
    )
    png_captured = png_diagnostic.get("pngCaptured") is True
    png_nonzero_rgb = png_diagnostic.get("pngNonzeroRgb")
    capture_output_nonblank = (
        png_nonzero_rgb > 0 if isinstance(png_nonzero_rgb, int) else None
    )
    presentation_nonblank_frame_count = numeric_value(
        get_path(compositor_contract, ["presentationNonBlankFrameCount"]),
        -1,
    )
    presentation_nonzero_ratio_max = numeric_value(
        get_path(compositor_contract, ["presentationNonzeroPixelRatioMax"]),
        -1,
    )
    runtime_nonzero_ratio = numeric_value(
        get_path(
            compositor_contract,
            ["nonzeroOutputRatio", "tileCompositorNonzeroOutputRatio"],
        ),
        -1,
    )
    presentation_output_nonblank = None
    presentation_source_kind = None
    if presentation_nonblank_frame_count >= 0 or presentation_nonzero_ratio_max >= 0:
        presentation_output_nonblank = (
            presentation_nonblank_frame_count > 0
            or presentation_nonzero_ratio_max > 0
        )
        presentation_source_kind = "webgpu-tile-compositor-presentation-frame-samples"
    runtime_output_nonblank = None
    runtime_output_evidence_source = None
    if runtime_nonzero_ratio >= 0:
        runtime_output_nonblank = runtime_nonzero_ratio > 0
        runtime_output_evidence_source = "webgpu-tile-compositor-output-nonzero-ratio"
    runtime_output_nonblank_evidence_ready = (
        runtime_output_nonblank is not None or presentation_output_nonblank is not None
    )
    effective_runtime_nonblank = (
        runtime_output_nonblank
        if runtime_output_nonblank is not None
        else presentation_output_nonblank
    )
    saved_png_matches_runtime_output = None
    if capture_output_nonblank is not None and effective_runtime_nonblank is not None:
        saved_png_matches_runtime_output = capture_output_nonblank == effective_runtime_nonblank
    orientation_evidence = get_path(png_capture_status, ["orientationEvidence"], {})
    requested_state_identity = get_path(
        png_capture_status, ["requestedStateIdentity"]
    )
    presented_frame_identity = get_path(
        png_capture_status, ["presentedFrameIdentity"]
    )
    captured_frame_identity = get_path(
        png_capture_status, ["capturedFrameIdentity"]
    )
    presented_identity_required_keys = [
        "generation",
        "datasetCameraLabel",
        "datasetFrameNumber",
        "datasetTime",
        "referenceCameraLabel",
        "outputWidth",
        "outputHeight",
    ]
    requested_identity_required_keys = [
        "datasetCameraLabel",
        "datasetFrameNumber",
        "datasetTime",
        "referenceCameraLabel",
        "outputWidth",
        "outputHeight",
    ]
    capture_vs_presented_identity = get_path(
        png_capture_status, ["captureVsPresentedFrameIdentity"]
    )
    if not isinstance(capture_vs_presented_identity, dict):
        capture_vs_presented_identity = compare_frame_identity(
            captured_frame_identity,
            presented_frame_identity,
            required_keys=presented_identity_required_keys,
        )
    capture_vs_requested_identity = get_path(
        png_capture_status, ["captureVsRequestedStateIdentity"]
    )
    if not isinstance(capture_vs_requested_identity, dict):
        capture_vs_requested_identity = compare_frame_identity(
            captured_frame_identity,
            requested_state_identity,
            required_keys=requested_identity_required_keys,
        )
    capture_matches_presented_frame = (
        presented_frame_identity is not None
        and get_path(capture_vs_presented_identity, ["matches"]) is True
    )
    capture_matches_requested_state = (
        requested_state_identity is not None
        and get_path(capture_vs_requested_identity, ["matches"]) is True
    )
    stale_capture_detected = (
        presented_frame_identity is not None
        and len(get_path(capture_vs_presented_identity, ["mismatchedKeys"], [])) > 0
    )
    capture_freshness_known = (
        capture_matches_presented_frame
        and capture_matches_requested_state
        and stale_capture_detected is False
    )
    capture_freshness_classification = (
        "captured-current-presented-fixed-reference-frame"
        if capture_freshness_known
        else "stale-last-valid-output-detected"
        if stale_capture_detected
        else "capture-requested-state-mismatch"
        if requested_state_identity is not None
        and capture_matches_requested_state is not True
        else "capture-freshness-evidence-missing-or-incomplete"
    )
    freshness_evidence = {
        "productionOutputGeneration": get_path(
            png_capture_status, ["productionOutputGeneration"]
        ),
        "presentedOutputGeneration": get_path(
            png_capture_status, ["presentedOutputGeneration"]
        ),
        "capturedOutputGeneration": get_path(
            png_capture_status, ["capturedOutputGeneration"]
        ),
        "requestedStateIdentity": requested_state_identity,
        "presentedFrameIdentity": presented_frame_identity,
        "capturedFrameIdentity": captured_frame_identity,
        "captureVsPresentedFrameIdentity": capture_vs_presented_identity,
        "captureVsRequestedStateIdentity": capture_vs_requested_identity,
        "captureMatchesPresentedFrame": capture_matches_presented_frame,
        "captureMatchesRequestedState": capture_matches_requested_state,
        "capturePresentedFrameMismatchedFields": get_path(
            capture_vs_presented_identity, ["mismatchedKeys"], []
        ),
        "capturePresentedFrameMissingFields": get_path(
            capture_vs_presented_identity, ["missingKeys"], []
        ),
        "captureRequestedStateMismatchedFields": get_path(
            capture_vs_requested_identity, ["mismatchedKeys"], []
        ),
        "captureRequestedStateMissingFields": get_path(
            capture_vs_requested_identity, ["missingKeys"], []
        ),
        "staleCaptureDetected": stale_capture_detected,
        "captureFreshnessKnown": capture_freshness_known,
        "captureFreshnessClassification": capture_freshness_classification,
    }
    saved_png_matches_presented_output = get_path(
        orientation_evidence,
        ["savedPngMatchesPresentedOutput"],
    )
    live_display_and_saved_png_consistency_known = (
        saved_png_matches_presented_output is True
        and get_path(orientation_evidence, ["orientationConsistencyKnown"]) is True
    )
    visual_degenerated = get_path(
        compositor_contract,
        ["visualOutputDegeneratedDetected"],
    )
    reason = None
    classification = None
    if png_captured and capture_output_nonblank is False:
        visual_degenerated = True
        if effective_runtime_nonblank is True:
            reason = "runtime-output-nonblank-but-saved-png-blank"
            classification = "capture-output-blank-runtime-presentation-mismatch"
        elif effective_runtime_nonblank is False:
            reason = "capture-and-runtime-output-blank"
            classification = "visual-output-blank"
        else:
            reason = "capture-output-blank-runtime-output-evidence-missing"
            classification = "capture-output-blank-runtime-output-evidence-missing"
    elif png_captured and capture_output_nonblank is True:
        reason = "saved-png-nonblank"
        classification = "capture-output-nonblank"
        if visual_degenerated is None:
            visual_degenerated = False
    elif not png_captured:
        reason = "saved-png-missing"
        classification = "capture-output-evidence-missing"
    else:
        reason = "saved-png-diagnostic-unavailable"
        classification = "capture-output-evidence-missing"
    return {
        **png_diagnostic,
        "captureTarget": get_path(webgpu_summary, ["captureTarget"]),
        "captureSourceKind": (
            get_path(png_capture_status, ["captureSourceKind"])
            or get_path(png_capture_status, ["source"])
            or "saved-png-current-viewer-canvas-via-saveCurrentCanvasPng"
            if png_captured
            else None
        ),
        "pngCaptureStatus": png_capture_status or {},
        "presentationSourceKind": presentation_source_kind,
        "runtimeOutputNonblankEvidenceSource": runtime_output_evidence_source
            or presentation_source_kind,
        "captureOutputNonblank": capture_output_nonblank,
        "presentationOutputNonblank": presentation_output_nonblank,
        "runtimeOutputNonblankEvidenceReady": runtime_output_nonblank_evidence_ready,
        "savedPngMatchesRuntimeOutput": saved_png_matches_runtime_output,
        "savedPngMatchesRawProductionOutput": get_path(
            orientation_evidence,
            ["savedPngMatchesRawProductionOutput"],
        ),
        "savedPngMatchesPresentedOutput": saved_png_matches_presented_output,
        "liveDisplayAndSavedPngConsistencyKnown":
            live_display_and_saved_png_consistency_known,
        "orientationEvidence": orientation_evidence,
        "captureFreshnessEvidence": freshness_evidence,
        "visualOutputDegeneratedDetected": visual_degenerated,
        "visualOutputDegenerationReason": reason,
        "visualOutputDegenerationClassification": classification,
    }


def apply_output_capture_diagnostic_to_step_summary(
    step_summary: Optional[Dict[str, Any]],
    diagnostic: Dict[str, Any],
    *,
    step_name: str,
) -> None:
    if not isinstance(step_summary, dict) or not diagnostic:
        return
    for key in [
        "captureTarget",
        "captureSourceKind",
        "presentationSourceKind",
        "runtimeOutputNonblankEvidenceSource",
        "pngCaptured",
        "pngWidth",
        "pngHeight",
        "pngNonzeroRgb",
        "pngNonblackRatio",
        "pngNonblackBBox",
        "pngMaxRgb",
        "pngSha256",
        "captureOutputNonblank",
        "presentationOutputNonblank",
        "runtimeOutputNonblankEvidenceReady",
        "savedPngMatchesRuntimeOutput",
        "savedPngMatchesRawProductionOutput",
        "savedPngMatchesPresentedOutput",
        "liveDisplayAndSavedPngConsistencyKnown",
        "orientationEvidence",
        "captureFreshnessEvidence",
        "visualOutputDegenerationReason",
        "visualOutputDegenerationClassification",
        "step108OutputRegressionDetected",
        "step108OutputRegressionRootCauseCandidate",
        "step108OutputRegressionEvidence",
        "step107PngNonblank",
        "step108PngNonblank",
        "step109PngNonblank",
    ]:
        step_summary[key] = diagnostic.get(key)
    if diagnostic.get("visualOutputDegeneratedDetected") is True:
        step_summary["visualOutputDegeneratedDetected"] = True
    if step_name != "step109":
        return
    if diagnostic.get("visualOutputDegeneratedDetected") is not True:
        return
    predicates = step_summary.get("step109ValidationPredicates")
    if not isinstance(predicates, list):
        predicates = []
    output_predicate = {
        "name": diagnostic.get("visualOutputDegenerationReason")
        or "visual-output-degenerated",
        "gate": "output-capture-consistency",
        "ready": False,
        "requiredForDecision": True,
        "reason": "saved PNG and runtime/presentation output evidence must be nonblank and consistent before Step109 can pass",
    }
    predicates = [
        predicate
        for predicate in predicates
        if predicate.get("gate") != "output-capture-consistency"
    ]
    predicates.insert(1 if predicates else 0, output_predicate)
    failed_predicates = [
        predicate
        for predicate in predicates
        if predicate.get("requiredForDecision") is True
        and predicate.get("ready") is not True
    ]
    step_summary["step109ValidationPredicates"] = predicates
    step_summary["step109FailedPredicates"] = failed_predicates
    step_summary["step109Decision"] = "blocked"
    step_summary["step109BlockedReason"] = output_predicate["name"]
    step_summary["step109BlockedReasonDetailed"] = (
        f"{output_predicate['name']}: {output_predicate['reason']}"
    )


def build_step75_camera_aware_visible_summary(
    summary: Dict[str, Any],
    webgpu_camera_aware_visible_output: Dict[str, Any],
    webgpu_normal_backend_frame_implementation: Dict[str, Any],
    webgpu_normal_backend_frame_implementation_validation: Dict[str, Any],
) -> Dict[str, Any]:
    runtime_contract = get_path(
        webgpu_camera_aware_visible_output,
        ["contract"],
        {},
    )
    normal_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["cameraAwareVisibleOutputContract"],
        {},
    )
    sample_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["sampleResourceLifecycleContract"],
        {},
    )
    color_surface_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["colorOutputSurfaceLifecycleContract"],
        {},
    )
    presentation_bridge_contract = get_path(
        webgpu_normal_backend_frame_implementation,
        ["presentationBridgeContract"],
        {},
    )
    visible_record_valid_count = get_path(
        runtime_contract,
        ["validRecordCount"],
        get_path(summary, ["validRecordCount"], 0),
    )
    runtime_visible_sample_count = get_path(runtime_contract, ["sampleCount"], 0)
    camera_aware_visible_sample_count = get_path(
        normal_contract,
        ["sampleCount"],
        get_path(
            webgpu_normal_backend_frame_implementation,
            ["visibleOutputSampleCount"],
            0,
        ),
    )
    source_mode = get_path(normal_contract, ["sourceMode"])
    source_kind = classify_camera_aware_input_source_kind(
        source_mode,
        runtime_visible_sample_count,
        camera_aware_visible_sample_count,
    )
    camera_ready = (
        get_path(normal_contract, ["cameraAwareVisibleOutputReady"]) is True
        or get_path(
            webgpu_normal_backend_frame_implementation,
            ["cameraAwareVisibleOutputReady"],
        )
        is True
        or get_path(
            webgpu_normal_backend_frame_implementation_validation,
            ["cameraAwareVisibleOutputReady"],
        )
        is True
    )
    current_texture_ready = get_path(
        webgpu_normal_backend_frame_implementation_validation,
        ["currentTextureConnectionReady"],
    )
    current_texture_readback_matches = get_path(
        presentation_bridge_contract,
        ["currentTextureReadbackMatchesAdapterOutput"],
    )
    webgl2_hybrid_prevented = get_path(
        webgpu_normal_backend_frame_implementation_validation,
        ["webgl2HybridRenderingPrevented"],
    )
    fallback_samples_mixed = get_path(
        normal_contract,
        ["fallbackSamplesMixed"],
        get_path(presentation_bridge_contract, ["fallbackSamplesMixed"]),
    )
    no_fallback_mixing = get_path(
        webgpu_normal_backend_frame_implementation_validation,
        ["noFallbackMixing"],
    )
    success = (
        camera_ready is True
        and current_texture_ready is True
        and current_texture_readback_matches is True
        and webgl2_hybrid_prevented is True
        and fallback_samples_mixed is False
        and no_fallback_mixing is True
        and bool(camera_aware_visible_sample_count)
    )
    valid_record_zero_success_reason = None
    if success and visible_record_valid_count == 0:
        valid_record_zero_success_reason = (
            "visible-record valid samples were 0, so Step75 succeeded via "
            "Step40 constrained-display selector-selected samples enlarged into "
            "camera/projection-aware patches; this is not render-handoff fallback "
            "and fallback mixing stayed suppressed"
        )
    return {
        "step75Decision": "success" if success else "blocked",
        "cameraAwareVisibleOutputContractStatus": get_path(
            normal_contract,
            ["status"],
        ),
        "cameraAwareVisibleOutputReady": camera_ready,
        "cameraAwareInputSourceKind": source_kind,
        "cameraAwareInputSourceLineage": describe_camera_aware_input_lineage(
            source_kind
        ),
        "cameraAwareInputSourceMode": source_mode,
        "runtimeVisibleRecordContractStatus": get_path(
            runtime_contract,
            ["status"],
        ),
        "runtimeVisibleRecordReason": get_path(runtime_contract, ["reason"]),
        "visibleRecordValidCount": visible_record_valid_count,
        "runtimeVisibleRecordSampleCount": runtime_visible_sample_count,
        "cameraAwareVisibleSampleCount": camera_aware_visible_sample_count,
        "selectedSamplePatchCount": get_path(sample_contract, ["sampleCount"]),
        "enlargedPatchPixelCount": get_path(
            color_surface_contract,
            ["writtenPixelCount"],
        ),
        "outputPointRadiusPx": get_path(normal_contract, ["outputPointRadiusPx"]),
        "debugFillUsed": False,
        "usesViewerCameraProjection": get_path(
            normal_contract,
            ["visibleOutputUsesCameraProjection"],
        ),
        "schedulerOwnedPath": (
            get_path(normal_contract, ["visibleOutputUsesSchedulerOwnedFramePath"])
            is True
            and get_path(
                normal_contract,
                ["schedulerFramePresentationBoundaryReady"],
            )
            is True
            and get_path(normal_contract, ["schedulerOwnsFrameRequest"]) is True
        ),
        "currentTextureConnectionReady": current_texture_ready,
        "currentTextureReadbackMatchesAdapterOutput": (
            current_texture_readback_matches
        ),
        "webgl2HybridRenderingPrevented": webgl2_hybrid_prevented,
        "fallbackSamplesMixed": fallback_samples_mixed,
        "noFallbackMixing": no_fallback_mixing,
        "selectedSourceKind": get_path(
            webgpu_normal_backend_frame_implementation,
            ["selectedSourceKind"],
        ),
        "selectionMode": get_path(
            webgpu_normal_backend_frame_implementation,
            ["selectionMode"],
        ),
        "sampleSources": compact_list(
            get_path(
                webgpu_normal_backend_frame_implementation,
                ["sampleSources"],
                [],
            )
        ),
        "validRecordZeroSuccessReason": valid_record_zero_success_reason,
        "firstValidationFailures": compact_list(
            get_path(
                webgpu_normal_backend_frame_implementation_validation,
                ["firstValidationFailures"],
                [],
            )
        ),
    }


def extract_capture_status(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "captureTarget": get_path(data, ["captureTarget"]),
        "captureStatus": get_path(data, ["status"]),
        "captureReason": get_path(data, ["reason"]),
        "captureFatalError": get_path(data, ["captureFatalError"]),
        "captureExceptionRecorded": get_path(data, ["captureExceptionRecorded"]),
        "captureErrorName": get_path(data, ["captureErrorName"]),
        "captureErrorMessage": get_path(data, ["captureErrorMessage"]),
        "captureErrorString": get_path(data, ["captureErrorString"]),
        "captureErrorStack": get_path(data, ["captureErrorStack"]),
    }


def extract_candidate_compare(data: Dict[str, Any]) -> Dict[str, Any]:
    candidate_comparison = get_path(
        data,
        [
            "candidateComparison",
            "comparison.candidateComparison",
            "summary.candidateComparison",
            "sourceCompare.candidateComparison",
        ],
        {},
    )

    candidate_indices = get_path(
        data,
        [
            "candidateIndices",
            "comparison.candidateIndices",
            "summary.candidateIndices",
        ],
        {},
    )

    gpu_summary = get_path(
        data,
        [
            "gpuCandidateSummary",
            "candidateSummary.gpuCandidateSummary",
            "summary.gpuCandidateSummary",
        ],
        {},
    )

    return {
        "status": get_path(data, ["status", "summary.status", "gpuCandidateSummary.status"]),
        "sourceMode": get_path(
            data,
            [
                "sourceMode",
                "gpuCandidateSourceMode",
                "candidateSourceSummary.sourceMode",
                "sourceSummary.sourceMode",
                "summary.sourceMode",
            ],
        ),
        "rangeStart": get_path(
            data,
            [
                "rangeStart",
                "candidateRangeStart",
                "sourceSummary.rangeStart",
                "candidateSourceSummary.rangeStart",
            ],
        ),
        "rangeCount": get_path(
            data,
            [
                "rangeCount",
                "candidateRangeCount",
                "sourceSummary.rangeCount",
                "candidateSourceSummary.rangeCount",
            ],
        ),
        "candidateCount": get_path(
            data,
            [
                "candidateCount",
                "gpuCandidateCount",
                "gpuCandidateSummary.candidateCount",
                "summary.candidateCount",
            ],
        ),
        "gpuCandidateStatus": get_path(gpu_summary, ["status"]),
        "candidateAnyMismatch": get_path(candidate_comparison, ["anyMismatch"]),
        "candidateReferenceCount": get_path(
            candidate_comparison,
            ["referenceCount", "cpuCount", "referenceCandidateCount"],
        ),
        "candidateCountCompared": get_path(
            candidate_comparison,
            ["candidateCount", "gpuCount", "gpuCandidateCount"],
        ),
        "orderMismatchCount": get_path(
            candidate_comparison,
            ["orderMismatchCount", "mismatchCount"],
        ),
        "firstMismatches": compact_list(
            get_path(candidate_comparison, ["firstMismatches", "mismatches"], [])
        ),
        "candidateIndicesAnyMismatch": get_path(candidate_indices, ["anyMismatch"]),
        "readbackMode": get_path(
            data,
            [
                "readbackMode",
                "readbackSummary.mode",
                "candidateSourceSummary.readbackMode",
                "sourceSummary.readbackMode",
            ],
        ),
    }


def extract_coverage(data: Dict[str, Any]) -> Dict[str, Any]:
    coverage = get_path(
        data,
        [
            "candidateCoverageSummary",
            "coverageSummary",
            "limitedDrawSummary.candidateCoverageSummary",
            "runtimeSummary.candidateCoverageSummary",
            "summary.candidateCoverageSummary",
        ],
        data,
    )

    return {
        "sourceMode": get_path(coverage, ["sourceMode", "gpuCandidateSourceMode"]),
        "candidateRangeStart": get_path(coverage, ["candidateRangeStart", "rangeStart"]),
        "candidateRangeCount": get_path(coverage, ["candidateRangeCount", "rangeCount"]),
        "gpuCandidateCount": get_path(coverage, ["gpuCandidateCount", "candidateCount"]),
        "cpuVisibleCount": get_path(coverage, ["cpuVisibleCount", "visibleCount"]),
        "visibleHitCount": get_path(coverage, ["visibleHitCount"]),
        "visibleMissCount": get_path(coverage, ["visibleMissCount"]),
        "visibleCoverageRatio": get_path(coverage, ["visibleCoverageRatio"]),
        "missingVisibleSrcIndices": compact_list(
            get_path(coverage, ["missingVisibleSrcIndices"], [])
        ),
        "packedVisibleCount": get_path(coverage, ["packedVisibleCount"]),
        "packedHitCount": get_path(coverage, ["packedHitCount"]),
        "packedMissCount": get_path(coverage, ["packedMissCount"]),
        "packedCoverageRatio": get_path(coverage, ["packedCoverageRatio"]),
        "missingPackedSrcIndices": compact_list(
            get_path(coverage, ["missingPackedSrcIndices"], [])
        ),
    }


def extract_runtime(data: Dict[str, Any]) -> Dict[str, Any]:
    runtime = get_path(data, ["runtimeSummary"], data)
    fallback = get_path(data, ["fallback", "runtimeSummary.fallback"], {})

    limited = get_path(
        data,
        ["limitedDrawSummary", "runtimeSummary.limitedDrawSummary"],
        {},
    )

    return {
        "requestedRuntime": get_path(runtime, ["requestedRuntime", "runtime"]),
        "effectiveDisplayRuntime": get_path(runtime, ["effectiveDisplayRuntime"]),
        "sourceMode": get_path(
            runtime,
            [
                "sourceMode",
                "gpuCandidateSourceMode",
                "candidateSourceSummary.sourceMode",
            ],
        ),
        "promotePolicy": get_path(runtime, ["promotePolicy", "gpuCandidatePromotePolicy"]),
        "displayCandidateSource": get_path(
            data,
            [
                "displayCandidateSource",
                "runtimeSummary.displayCandidateSource",
                "limitedDrawSummary.displayCandidateSource",
            ],
        ),
        "gpuCandidateUsedForDisplay": get_path(
            data,
            [
                "gpuCandidateUsedForDisplay",
                "runtimeSummary.gpuCandidateUsedForDisplay",
                "limitedDrawSummary.gpuCandidateUsedForDisplay",
            ],
        ),
        "limitedDrawUsedForCandidateSource": get_path(
            data,
            [
                "limitedDrawUsedForCandidateSource",
                "runtimeSummary.limitedDrawUsedForCandidateSource",
                "limitedDrawSummary.limitedDrawUsedForCandidateSource",
            ],
        ),
        "fallbackAction": get_path(fallback, ["action"]),
        "fallbackReason": get_path(fallback, ["reason", "reasons"]),
        "limitedDrawStatus": get_path(limited, ["status"]),
        "limitedDrawReason": get_path(limited, ["reason"]),
        "candidateInfoOverrideProvided": get_path(
            limited,
            ["candidateInfoOverrideProvided"],
        ),
        "promotionDecision": get_path(
            data,
            [
                "promotionDecision",
                "limitedDrawSummary.promotionDecision",
                "runtimeSummary.limitedDrawSummary.promotionDecision",
            ],
        ),
    }


def extract_visible_compare(data: Dict[str, Any]) -> Dict[str, Any]:
    visible_comparison = get_path(data, ["visibleComparison"], data)

    visible_items = get_path(
        visible_comparison,
        ["visibleItems", "visibleComparison.visibleItems"],
        {},
    )
    packed_payload = get_path(
        visible_comparison,
        ["packedPayload", "visibleComparison.packedPayload"],
        {},
    )
    candidate_comparison = get_path(
        visible_comparison,
        ["candidateComparison", "visibleComparison.candidateComparison"],
        {},
    )

    return {
        "anyMismatch": get_path(visible_comparison, ["anyMismatch"]),
        "candidateAnyMismatch": get_path(candidate_comparison, ["anyMismatch"]),
        "visibleItemsAnyMismatch": get_path(visible_items, ["anyMismatch"]),
        "visibleItemsReferenceCount": get_path(
            visible_items,
            ["referenceCount", "referenceVisibleCount"],
        ),
        "visibleItemsCandidateCount": get_path(
            visible_items,
            ["candidateCount", "candidateVisibleCount"],
        ),
        "packedPayloadAnyMismatch": get_path(packed_payload, ["anyMismatch"]),
        "packedPayloadReferenceCount": get_path(packed_payload, ["referenceCount"]),
        "packedPayloadCandidateCount": get_path(packed_payload, ["candidateCount"]),
        "packedPayloadMaxAbs": get_path(
            packed_payload,
            ["maxAbs", "maxAbsError", "maxAbsDiff"],
        ),
    }


def extract_dryrun_visible_compare(data: Dict[str, Any]) -> Dict[str, Any]:
    visible = extract_visible_compare(data)
    coverage = get_path(data, ["coverageSummary"], {})
    candidate = extract_candidate_compare(data)
    return {
        "status": get_path(data, ["status"]),
        "reason": get_path(data, ["reason"]),
        "sourceMode": get_path(data, ["sourceMode", "candidateSourceSummary.sourceMode"]),
        "gpuCandidateCount": get_path(
            data,
            [
                "coverageSummary.gpuCandidateCount",
                "gpuCandidateSummary.candidateCount",
                "candidateSourceSummary.candidateCount",
            ],
        ),
        "visibleCoverageRatio": get_path(coverage, ["visibleCoverageRatio"]),
        "visibleMissCount": get_path(coverage, ["visibleMissCount"]),
        "candidateAnyMismatch": candidate.get("candidateAnyMismatch"),
        "visibleItemsAnyMismatch": visible.get("visibleItemsAnyMismatch"),
        "visibleItemsReferenceCount": visible.get("visibleItemsReferenceCount"),
        "visibleItemsCandidateCount": visible.get("visibleItemsCandidateCount"),
        "packedPayloadAnyMismatch": visible.get("packedPayloadAnyMismatch"),
        "packedPayloadReferenceCount": visible.get("packedPayloadReferenceCount"),
        "packedPayloadCandidateCount": visible.get("packedPayloadCandidateCount"),
        "packedPayloadMaxAbs": visible.get("packedPayloadMaxAbs"),
        "mismatchClassification": get_path(data, ["mismatchClassification"]),
        "anyMismatch": get_path(data, ["anyMismatch"]),
        "displayCandidateSource": get_path(data, ["displayCandidateSource"]),
        "gpuCandidateUsedForDisplay": get_path(data, ["gpuCandidateUsedForDisplay"]),
        "limitedDrawUsedForCandidateSource": get_path(data, ["limitedDrawUsedForCandidateSource"]),
    }


def extract_screen_coarse_sweep(data: Dict[str, Any]) -> Dict[str, Any]:
    summary = get_path(data, ["summary"], {})
    cases = get_path(data, ["cases"], [])
    first_success = None
    first_shortage = None
    if isinstance(cases, list):
        for case in cases:
            if not isinstance(case, dict):
                continue
            if first_success is None and case.get("mismatchClassification") == "none":
                first_success = case
            if first_shortage is None and case.get("mismatchClassification") == "candidate-shortage":
                first_shortage = case

    def compact_case(case: Any) -> Any:
        if not isinstance(case, dict):
            return None
        return {
            "caseId": case.get("caseId"),
            "candidateCount": case.get("candidateCount"),
            "visibleCoverageRatio": case.get("visibleCoverageRatio"),
            "visibleMissCount": case.get("visibleMissCount"),
            "mismatchClassification": case.get("mismatchClassification"),
            "visibleItemsAnyMismatch": case.get("visibleItemsAnyMismatch"),
            "packedPayloadAnyMismatch": case.get("packedPayloadAnyMismatch"),
            "timing": case.get("timing"),
        }

    return {
        "status": get_path(data, ["status"]),
        "reason": get_path(data, ["reason"]),
        "sourceMode": get_path(data, ["sourceMode"]),
        "caseCount": get_path(summary, ["caseCount"]),
        "successCaseCount": get_path(summary, ["successCaseCount"]),
        "shortageCaseCount": get_path(summary, ["shortageCaseCount"]),
        "mismatchCaseCount": get_path(summary, ["mismatchCaseCount"]),
        "totalMs": get_path(summary, ["totalMs"]),
        "displayCandidateSource": get_path(data, ["displayCandidateSource"]),
        "gpuCandidateUsedForDisplay": get_path(data, ["gpuCandidateUsedForDisplay"]),
        "limitedDrawUsedForCandidateSource": get_path(data, ["limitedDrawUsedForCandidateSource"]),
        "firstSuccessCase": compact_case(first_success),
        "firstShortageCase": compact_case(first_shortage),
    }


def extract_promotion_validation(data: Dict[str, Any]) -> Dict[str, Any]:
    validation = get_path(
        data,
        [
            "promotionValidation",
            "limitedDrawSummary.promotionValidation",
            "runtimeSummary.limitedDrawSummary.promotionValidation",
        ],
        {},
    )
    failure_reasons = get_path(validation, ["failureReasons"], [])
    if isinstance(failure_reasons, list):
        failure_reasons = [
            item.get("code") if isinstance(item, dict) else item
            for item in failure_reasons
        ]
    return {
        "promotionDecision": get_path(
            data,
            [
                "promotionDecision",
                "limitedDrawSummary.promotionDecision",
                "runtimeSummary.limitedDrawSummary.promotionDecision",
                "promotionValidation.promotionDecision",
            ],
        ),
        "sourceMode": get_path(validation, ["sourceMode"]),
        "promotePolicy": get_path(validation, ["promotePolicy"]),
        "gpuCandidateCount": get_path(validation, ["gpuCandidateCount"]),
        "visibleCoverageRatio": get_path(validation, ["visibleCoverageRatio"]),
        "visibleMissCount": get_path(validation, ["visibleMissCount"]),
        "candidateAnyMismatch": get_path(validation, ["candidateAnyMismatch"]),
        "visibleItemsAnyMismatch": get_path(validation, ["visibleItemsAnyMismatch"]),
        "packedPayloadAnyMismatch": get_path(validation, ["packedPayloadAnyMismatch"]),
        "mismatchClassification": get_path(validation, ["mismatchClassification"]),
        "failureReasons": failure_reasons,
        "displayCandidateSource": get_path(validation, ["displayCandidateSource"]),
        "gpuCandidateUsedForDisplay": get_path(validation, ["gpuCandidateUsedForDisplay"]),
        "limitedDrawUsedForCandidateSource": get_path(
            validation,
            ["limitedDrawUsedForCandidateSource"],
        ),
        "candidateInfoOverrideProvided": get_path(
            validation,
            ["candidateInfoOverrideProvided"],
        ),
    }


def extract_step111_timing(data: Dict[str, Any]) -> Dict[str, Any]:
    timing = get_path(
        data,
        [
            "step111TimingSummary",
            "limitedDrawSummary.step111TimingSummary",
            "runtimeSummary.limitedDrawSummary.step111TimingSummary",
        ],
        {},
    )
    return {
        "displayCandidateSource": get_path(timing, ["displayCandidateSource"]),
        "promotionDecision": get_path(timing, ["promotionDecision"]),
        "fallbackReason": get_path(timing, ["fallbackReason"]),
        "gpuCandidateCount": get_path(timing, ["gpuCandidateCount"]),
        "visibleCoverageRatio": get_path(timing, ["visibleCoverageRatio"]),
        "candidateSourceMs": get_path(timing, ["candidateSourceMs"]),
        "cpuCandidateSourceMs": get_path(timing, ["cpuCandidateSourceMs"]),
        "transformFeedbackDrawMs": get_path(timing, ["transformFeedbackDrawMs"]),
        "readbackMs": get_path(timing, ["readbackMs"]),
        "promotionValidationMs": get_path(timing, ["promotionValidationMs"]),
        "referenceVisibleLoopMs": get_path(timing, ["referenceVisibleLoopMs"]),
        "referenceVisibleSortMs": get_path(timing, ["referenceVisibleSortMs"]),
        "visibleBuildMs": get_path(timing, ["visibleBuildMs"]),
        "visibleLoopMs": get_path(timing, ["visibleLoopMs"]),
        "visibleSortMs": get_path(timing, ["visibleSortMs"]),
        "packedBuildMs": get_path(timing, ["packedBuildMs"]),
        "totalVisiblePackedBuildMs": get_path(timing, ["totalVisiblePackedBuildMs"]),
        "actualDisplayVisibleBuildMs": get_path(timing, ["actualDisplayVisibleBuildMs"]),
        "actualDisplayPackedBuildMs": get_path(timing, ["actualDisplayPackedBuildMs"]),
        "actualDisplayTotalBuildMs": get_path(timing, ["actualDisplayTotalBuildMs"]),
        "actualDisplayVisibleLoopMs": get_path(timing, ["actualDisplayVisibleLoopMs"]),
        "actualDisplayVisibleSortMs": get_path(timing, ["actualDisplayVisibleSortMs"]),
        "actualDisplayTileListBuildMs": get_path(timing, ["actualDisplayTileListBuildMs"]),
        "actualDisplayCpuPostCandidateTotalMs": get_path(
            timing,
            ["actualDisplayCpuPostCandidateTotalMs"],
        ),
        "cpuPostCandidateBreakdown": get_path(timing, ["cpuPostCandidateBreakdown"]),
        "totalLimitedDrawMs": get_path(timing, ["totalLimitedDrawMs"]),
        "remainingCpuDependencies": get_path(timing, ["remainingCpuDependencies"], []),
    }


def extract_gpu_visible_record_dryrun(data: Dict[str, Any]) -> Dict[str, Any]:
    summary = get_path(
        data,
        [
            "gpuVisibleRecordDryRunSummary",
            "limitedDrawSummary.gpuVisibleRecordDryRunSummary",
            "runtimeSummary.limitedDrawSummary.gpuVisibleRecordDryRunSummary",
        ],
        None,
    )
    if not isinstance(summary, dict):
        summary = data if get_path(data, ["schemaVersion"]) == "step114-gpu-visible-record-dry-run-v1" else {}
    timing = get_path(summary, ["timing"], {})
    record_comparison = get_path(summary, ["recordComparison"], {})
    reference_visible = get_path(summary, ["referenceVisibleComparison"], {})
    return {
        "status": get_path(summary, ["status"]),
        "reason": get_path(summary, ["reason"]),
        "sourceMode": get_path(summary, ["sourceMode"]),
        "computeMode": get_path(summary, ["computeMode"]),
        "candidateCount": get_path(summary, ["candidateCount"]),
        "recordCount": get_path(summary, ["recordCount"]),
        "validRecordCount": get_path(summary, ["validRecordCount"]),
        "mismatchClassification": get_path(summary, ["mismatchClassification"]),
        "anyMismatch": get_path(summary, ["anyMismatch"]),
        "recordAnyMismatch": get_path(record_comparison, ["anyMismatch"]),
        "recordFieldMismatchCount": get_path(record_comparison, ["fieldMismatchCount"]),
        "referenceVisibleAnyMismatch": get_path(reference_visible, ["anyMismatch"]),
        "referenceVisibleMissingRecordCount": get_path(reference_visible, ["missingRecordCount"]),
        "referenceVisibleInvalidRecordCount": get_path(reference_visible, ["invalidRecordCount"]),
        "referenceVisibleFieldMismatchCount": get_path(reference_visible, ["fieldMismatchCount"]),
        "displayCandidateSource": get_path(summary, ["displayCandidateSource"]),
        "gpuCandidateUsedForDisplay": get_path(summary, ["gpuCandidateUsedForDisplay"]),
        "limitedDrawUsedForCandidateSource": get_path(summary, ["limitedDrawUsedForCandidateSource"]),
        "transformFeedbackDrawMs": get_path(timing, ["transformFeedbackDrawMs"]),
        "readbackMs": get_path(timing, ["readbackMs"]),
        "compareMs": get_path(timing, ["compareMs"]),
        "totalMs": get_path(timing, ["totalMs"]),
    }


def extract_gpu_raw_visible_record_dryrun(data: Dict[str, Any]) -> Dict[str, Any]:
    summary = get_path(
        data,
        [
            "gpuRawVisibleRecordDryRunSummary",
            "limitedDrawSummary.gpuRawVisibleRecordDryRunSummary",
            "runtimeSummary.limitedDrawSummary.gpuRawVisibleRecordDryRunSummary",
        ],
        None,
    )
    if not isinstance(summary, dict):
        summary = data if get_path(data, ["schemaVersion"]) == "step116-raw-visible-record-dry-run-v1" else {}
    timing = get_path(summary, ["timing"], {})
    record_comparison = get_path(summary, ["recordComparison"], {})
    packed_like_comparison = get_path(summary, ["packedLikeComparison"], {})
    comparison_reference = get_path(summary, ["comparisonReference"], {})
    packed_like_reference = get_path(summary, ["packedLikeComparisonReference"], {})
    display_readiness = get_path(summary, ["displayConnectionReadiness"], {})
    first_mismatches = get_path(record_comparison, ["firstMismatches"], [])
    mismatch_fields = sorted({
        item.get("field")
        for item in first_mismatches
        if isinstance(item, dict) and item.get("field") is not None
    })
    return {
        "status": get_path(summary, ["status"]),
        "reason": get_path(summary, ["reason"]),
        "computeMode": get_path(summary, ["computeMode"]),
        "recordMode": get_path(summary, ["recordMode"]),
        "rawVisibleRecordMode": get_path(summary, ["rawVisibleRecordMode"]),
        "cpuReferenceMode": get_path(comparison_reference, ["cpuReferenceMode"]),
        "aabbReferenceMode": get_path(comparison_reference, ["aabbReferenceMode"]),
        "canonicalCpuAabbMode": get_path(comparison_reference, ["canonicalCpuAabbMode"]),
        "comparisonReferenceNote": get_path(comparison_reference, ["note"]),
        "implementedFields": get_path(summary, ["implementedFields"], []),
        "deferredFields": get_path(summary, ["deferredFields"], []),
        "packedLikeImplementedFields": get_path(summary, ["packedLikeImplementedFields"], []),
        "packedLikeDeferredFields": get_path(summary, ["packedLikeDeferredFields"], []),
        "packedLikeReferenceMode": get_path(packed_like_reference, ["cpuReferenceMode"]),
        "packedLikeOrderMode": get_path(packed_like_reference, ["orderMode"]),
        "packedLikeColorRgbMode": get_path(packed_like_reference, ["colorRgbMode"]),
        "candidateCount": get_path(summary, ["candidateCount"]),
        "recordCount": get_path(summary, ["recordCount"]),
        "validRecordCount": get_path(summary, ["validRecordCount"]),
        "mismatchClassification": get_path(summary, ["mismatchClassification"]),
        "anyMismatch": get_path(summary, ["anyMismatch"]),
        "recordAnyMismatch": get_path(record_comparison, ["anyMismatch"]),
        "fieldMismatchCount": get_path(record_comparison, ["fieldMismatchCount"]),
        "fieldMismatchFields": mismatch_fields,
        "firstMismatches": first_mismatches,
        "packedLikeMismatchClassification": get_path(summary, ["packedLikeMismatchClassification"]),
        "packedLikeAnyMismatch": get_path(packed_like_comparison, ["anyMismatch"]),
        "packedLikeFieldMismatchCount": get_path(packed_like_comparison, ["fieldMismatchCount"]),
        "packedLikeFirstMismatches": get_path(packed_like_comparison, ["firstMismatches"], []),
        "displayConnectionStatus": get_path(display_readiness, ["status"]),
        "displayConnectionAllowed": get_path(display_readiness, ["displayConnectionAllowed"]),
        "displayConnectionClassification": get_path(
            display_readiness,
            ["displayConnectionClassification"],
        ),
        "displayConnectionReason": get_path(display_readiness, ["reason"]),
        "displayConnectionSatisfied": get_path(display_readiness, ["satisfied"], []),
        "displayConnectionUnresolved": get_path(display_readiness, ["unresolved"], []),
        "displayConnectionBlocked": get_path(display_readiness, ["blocked"], []),
        "displayConnectionWebgl2Limits": get_path(
            display_readiness,
            ["webgl2LimitCandidates"],
            [],
        ),
        "displayConnectionWebgpuSignals": get_path(
            display_readiness,
            ["webgpuMigrationSignals"],
            [],
        ),
        "displayCandidateSource": get_path(summary, ["displayCandidateSource"]),
        "gpuCandidateUsedForDisplay": get_path(summary, ["gpuCandidateUsedForDisplay"]),
        "limitedDrawUsedForCandidateSource": get_path(summary, ["limitedDrawUsedForCandidateSource"]),
        "fallbackReason": get_path(summary, ["fallbackReason"]),
        "rawTextureUploadMs": get_path(timing, ["rawTextureUploadMs"]),
        "transformFeedbackDrawMs": get_path(timing, ["transformFeedbackDrawMs"]),
        "readbackMs": get_path(timing, ["readbackMs"]),
        "compareMs": get_path(timing, ["compareMs"]),
        "packedLikeCompareMs": get_path(timing, ["packedLikeCompareMs"]),
        "totalMs": get_path(timing, ["totalMs"]),
    }


def extract_webgpu_visible_record_dryrun(data: Dict[str, Any]) -> Dict[str, Any]:
    summary = get_path(
        data,
        [
            "webgpuVisibleRecordDryRun",
            "webgpuVisibleRecordDryRunSummary",
            "limitedDrawSummary.webgpuVisibleRecordDryRunSummary",
            "runtimeSummary.limitedDrawSummary.webgpuVisibleRecordDryRunSummary",
        ],
        None,
    )
    if not isinstance(summary, dict):
        summary = (
            data
            if get_path(data, ["schemaVersion"]) == WEBGPU_VISIBLE_RECORD_DRY_RUN_SCHEMA_VERSION
            else {}
        )
    timing = get_path(summary, ["timing"], {})
    record_comparison = get_path(summary, ["recordComparison"], {})
    webgpu = get_path(summary, ["webgpu"], {})
    counts_offsets = get_path(summary, ["tileCountsToOffsetsDryRun"], {})
    counts_offsets_validation = get_path(
        counts_offsets, ["validationSummary"], {}
    )
    counts_offsets_metadata = get_path(counts_offsets, ["metadata"], {})
    counts_offsets_capacity = get_path(counts_offsets, ["capacity"], {})
    counts_offsets_self_comparison = get_path(
        summary, ["tileCountsOffsetsSelfComparison"], {}
    )
    webgpu_tile_counts = get_path(summary, ["webgpuTileCountsDryRun"], {})
    tile_counts_webgpu_comparison = get_path(
        summary, ["tileCountsWebGpuComparison"], {}
    )
    offsets_from_webgpu_counts = get_path(
        summary, ["tileOffsetsFromWebGpuCountsDryRun"], {}
    )
    offsets_from_webgpu_counts_validation = get_path(
        offsets_from_webgpu_counts, ["validationSummary"], {}
    )
    offsets_from_webgpu_counts_metadata = get_path(
        offsets_from_webgpu_counts, ["metadata"], {}
    )
    offsets_from_webgpu_counts_capacity = get_path(
        offsets_from_webgpu_counts, ["capacity"], {}
    )
    tile_offsets_prefix_comparison = get_path(
        summary, ["tileOffsetsPrefixComparison"], {}
    )
    webgpu_tile_offsets_prefix = get_path(
        summary, ["webgpuTileOffsetsPrefixDryRun"], {}
    )
    webgpu_tile_offsets_prefix_validation = get_path(
        webgpu_tile_offsets_prefix, ["validationSummary"], {}
    )
    webgpu_tile_offsets_prefix_metadata = get_path(
        webgpu_tile_offsets_prefix, ["metadata"], {}
    )
    webgpu_tile_offsets_prefix_capacity = get_path(
        webgpu_tile_offsets_prefix, ["capacity"], {}
    )
    tile_offsets_webgpu_prefix_comparison = get_path(
        summary, ["tileOffsetsWebGpuPrefixComparison"], {}
    )
    scatter_validation_boundary = get_path(
        summary, ["scatterValidationBoundary"], {}
    )
    scatter_validation_summary = get_path(
        scatter_validation_boundary, ["validationSummary"], {}
    )
    scatter_validation_capacity = get_path(
        scatter_validation_boundary, ["capacity"], {}
    )
    tile_indices_self_comparison = get_path(
        summary, ["tileIndicesSelfComparison"], {}
    )
    tile_indices_self_capacity = get_path(
        tile_indices_self_comparison, ["capacity"], {}
    )
    tile_indices_webgpu_scatter_comparison = get_path(
        summary, ["tileIndicesWebGpuScatterComparison"], {}
    )
    tile_indices_webgpu_scatter_capacity = get_path(
        tile_indices_webgpu_scatter_comparison, ["capacity"], {}
    )
    tile_indices_webgpu_scatter_validation = get_path(
        tile_indices_webgpu_scatter_comparison, ["validationSummary"], {}
    )
    tile_list_summary_comparison = get_path(
        summary, ["tileListSummaryComparison"], {}
    )
    tile_list_summary_metadata = get_path(
        tile_list_summary_comparison, ["metadataComparison"], {}
    )
    tile_list_summary_validation = get_path(
        tile_list_summary_comparison, ["validationSummary"], {}
    )
    tile_list_summary_capacity = get_path(
        tile_list_summary_comparison, ["capacity"], {}
    )
    webgpu_tile_list_backend_output = get_path(
        summary, ["webgpuTileListBackendOutput"], {}
    )
    webgpu_tile_list_backend_handoff = get_path(
        webgpu_tile_list_backend_output, ["handoffReadiness"], {}
    )
    webgpu_tile_list_backend_validation = get_path(
        webgpu_tile_list_backend_output, ["validationSummary"], {}
    )
    render_payload_sort_readiness = get_path(
        summary, ["renderPayloadSortReadiness"], {}
    )
    render_payload_readiness = get_path(
        render_payload_sort_readiness, ["payloadReadiness"], {}
    )
    sort_readiness = get_path(
        render_payload_sort_readiness, ["sortReadiness"], {}
    )
    readiness_summary = get_path(
        render_payload_sort_readiness, ["readinessSummary"], {}
    )
    depth_sort_comparison = get_path(summary, ["depthSortComparison"], {})
    depth_sort_validation = get_path(
        depth_sort_comparison, ["validationSummary"], {}
    )
    webgpu_render_handoff_stub = get_path(
        summary, ["webgpuRenderHandoffStub"], {}
    )
    render_handoff_validation = get_path(
        webgpu_render_handoff_stub, ["validationSummary"], {}
    )
    webgpu_tile_composite_handoff_stub = get_path(
        summary, ["webgpuTileCompositeHandoffStub"], {}
    )
    tile_composite_handoff_validation = get_path(
        webgpu_tile_composite_handoff_stub, ["validationSummary"], {}
    )
    webgpu_tile_composite_shader_handoff = get_path(
        summary, ["webgpuTileCompositeShaderHandoff"], {}
    )
    tile_composite_shader_handoff_validation = get_path(
        webgpu_tile_composite_shader_handoff, ["validationSummary"], {}
    )
    webgpu_tile_composite_shader_dry_run_comparison = get_path(
        summary, ["webgpuTileCompositeShaderDryRunComparison"], {}
    )
    tile_composite_shader_dry_run_validation = get_path(
        webgpu_tile_composite_shader_dry_run_comparison,
        ["validationSummary"],
        {},
    )
    webgpu_tile_composite_accumulation_dry_run_comparison = get_path(
        summary, ["webgpuTileCompositeAccumulationDryRunComparison"], {}
    )
    tile_composite_accumulation_dry_run_validation = get_path(
        webgpu_tile_composite_accumulation_dry_run_comparison,
        ["validationSummary"],
        {},
    )
    webgpu_framebuffer_free_tile_output_dry_run_comparison = get_path(
        summary, ["webgpuFramebufferFreeTileOutputDryRunComparison"], {}
    )
    framebuffer_free_tile_output_dry_run_validation = get_path(
        webgpu_framebuffer_free_tile_output_dry_run_comparison,
        ["validationSummary"],
        {},
    )
    webgpu_render_target_handoff_dry_run_comparison = get_path(
        summary, ["webgpuRenderTargetHandoffDryRunComparison"], {}
    )
    render_target_handoff_dry_run_validation = get_path(
        webgpu_render_target_handoff_dry_run_comparison,
        ["validationSummary"],
        {},
    )
    webgpu_constrained_display_adapter_dry_run_comparison = get_path(
        summary, ["webgpuConstrainedDisplayAdapterDryRunComparison"], {}
    )
    constrained_display_adapter_dry_run_validation = get_path(
        webgpu_constrained_display_adapter_dry_run_comparison,
        ["validationSummary"],
        {},
    )
    webgpu_guarded_first_display_experiment = get_path(
        summary, ["webgpuGuardedFirstDisplayExperiment"], {}
    )
    guarded_first_display_validation = get_path(
        webgpu_guarded_first_display_experiment,
        ["validationSummary"],
        {},
    )
    webgpu_canvas_presentation_adapter_dry_run_comparison = get_path(
        summary, ["webgpuCanvasPresentationAdapterDryRunComparison"], {}
    )
    canvas_presentation_adapter_validation = get_path(
        webgpu_canvas_presentation_adapter_dry_run_comparison,
        ["validationSummary"],
        {},
    )
    webgpu_exclusive_canvas_handoff_readiness = get_path(
        summary, ["webgpuExclusiveCanvasHandoffReadiness"], {}
    )
    exclusive_canvas_handoff_validation = get_path(
        webgpu_exclusive_canvas_handoff_readiness,
        ["validationSummary"],
        {},
    )
    webgpu_exclusive_frame_lifecycle_switch = get_path(
        summary, ["webgpuExclusiveFrameLifecycleSwitch"], {}
    )
    exclusive_frame_lifecycle_validation = get_path(
        webgpu_exclusive_frame_lifecycle_switch,
        ["validationSummary"],
        {},
    )
    webgpu_viewer_canvas_current_texture_path = get_path(
        summary, ["webgpuViewerCanvasCurrentTexturePath"], {}
    )
    viewer_canvas_current_texture_validation = get_path(
        webgpu_viewer_canvas_current_texture_path,
        ["validationSummary"],
        {},
    )
    webgpu_viewer_canvas_bounded_first_present = get_path(
        summary, ["webgpuViewerCanvasBoundedFirstPresent"], {}
    )
    viewer_canvas_bounded_first_present_validation = get_path(
        webgpu_viewer_canvas_bounded_first_present,
        ["validationSummary"],
        {},
    )
    webgpu_viewer_canvas_native_bounded_color_samples = get_path(
        summary, ["webgpuViewerCanvasNativeBoundedColorSamples"], {}
    )
    viewer_canvas_native_bounded_color_samples_validation = get_path(
        webgpu_viewer_canvas_native_bounded_color_samples,
        ["validationSummary"],
        {},
    )
    webgpu_viewer_canvas_bounded_color_source_selector = get_path(
        summary, ["webgpuViewerCanvasBoundedColorSourceSelector"], {}
    )
    viewer_canvas_bounded_color_source_selector_validation = get_path(
        webgpu_viewer_canvas_bounded_color_source_selector,
        ["validationSummary"],
        {},
    )
    webgpu_viewer_canvas_bounded_color_present = get_path(
        summary, ["webgpuViewerCanvasBoundedColorPresent"], {}
    )
    viewer_canvas_bounded_color_present_validation = get_path(
        webgpu_viewer_canvas_bounded_color_present,
        ["validationSummary"],
        {},
    )
    webgpu_backend_frame_prototype = get_path(
        summary, ["webgpuBackendFramePrototype"], {}
    )
    webgpu_backend_frame_validation = get_path(
        webgpu_backend_frame_prototype,
        ["validationSummary"],
        {},
    )
    webgpu_backend_frame_lifecycle_prototype = get_path(
        summary, ["webgpuBackendFrameLifecyclePrototype"], {}
    )
    webgpu_backend_frame_lifecycle_validation = get_path(
        webgpu_backend_frame_lifecycle_prototype,
        ["validationSummary"],
        {},
    )
    webgpu_backend_frame_controlled_repeated_execution = get_path(
        summary, ["webgpuBackendFrameControlledRepeatedExecution"], {}
    )
    webgpu_backend_frame_controlled_repeated_execution_validation = get_path(
        webgpu_backend_frame_controlled_repeated_execution,
        ["validationSummary"],
        {},
    )
    webgpu_backend_viewer_loop_adapter = get_path(
        summary, ["webgpuBackendViewerLoopAdapter"], {}
    )
    webgpu_backend_viewer_loop_adapter_validation = get_path(
        webgpu_backend_viewer_loop_adapter,
        ["validationSummary"],
        {},
    )
    webgpu_backend_viewer_lifecycle_integration = get_path(
        summary, ["webgpuBackendViewerLifecycleIntegrationBoundary"], {}
    )
    webgpu_backend_viewer_lifecycle_integration_validation = get_path(
        webgpu_backend_viewer_lifecycle_integration,
        ["validationSummary"],
        {},
    )
    webgpu_backend_viewer_lifecycle_controlled_execution = get_path(
        summary, ["webgpuBackendViewerLifecycleControlledExecution"], {}
    )
    webgpu_backend_viewer_lifecycle_controlled_execution_validation = get_path(
        webgpu_backend_viewer_lifecycle_controlled_execution,
        ["validationSummary"],
        {},
    )
    webgpu_backend_viewer_frame_executor = get_path(
        summary, ["webgpuBackendViewerFrameExecutor"], {}
    )
    webgpu_backend_viewer_frame_presentation_pass = get_path(
        summary, ["webgpuBackendViewerFramePresentationPass"], {}
    )
    webgpu_scheduler_frame_presentation_boundary = get_path(
        summary, ["webgpuSchedulerFramePresentationBoundary"], {}
    )
    webgpu_backend_viewer_frame_executor_validation = get_path(
        webgpu_backend_viewer_frame_executor,
        ["validationSummary"],
        {},
    )
    webgpu_backend_runtime_runner = get_path(
        summary, ["webgpuBackendRuntimeRunner"], {}
    )
    webgpu_backend_runtime_runner_validation = get_path(
        webgpu_backend_runtime_runner,
        ["validationSummary"],
        {},
    )
    webgpu_normal_backend_frame_implementation = get_path(
        summary, ["webgpuNormalBackendFrameImplementation"], {}
    )
    webgpu_normal_backend_frame_implementation_validation = get_path(
        webgpu_normal_backend_frame_implementation,
        ["validationSummary"],
        {},
    )
    webgpu_camera_aware_visible_output = get_path(
        summary, ["webgpuCameraAwareVisibleOutput"], {}
    )
    webgpu_visible_record_camera_aware_visible_output = get_path(
        summary, ["webgpuVisibleRecordCameraAwareVisibleOutput"], {}
    )
    validation_assisted_camera_aware_visible_output = get_path(
        summary, ["validationAssistedCameraAwareVisibleOutput"], {}
    )
    webgpu_owned_camera_aware_visible_output = get_path(
        summary, ["webgpuOwnedCameraAwareVisibleOutput"], {}
    )
    step75_camera_aware_visible_output = (
        build_step75_camera_aware_visible_summary(
            summary,
            webgpu_camera_aware_visible_output,
            webgpu_normal_backend_frame_implementation,
            webgpu_normal_backend_frame_implementation_validation,
        )
    )
    step76_many_camera_aware_visible_output = (
        build_step76_many_camera_aware_visible_summary(
            summary,
            webgpu_camera_aware_visible_output,
            webgpu_visible_record_camera_aware_visible_output,
            validation_assisted_camera_aware_visible_output,
            webgpu_normal_backend_frame_implementation,
            webgpu_normal_backend_frame_implementation_validation,
        )
    )
    step77_webgpu_owned_visible_output = (
        build_step77_webgpu_owned_visible_summary(
            summary,
            webgpu_owned_camera_aware_visible_output,
            validation_assisted_camera_aware_visible_output,
            webgpu_normal_backend_frame_implementation,
            webgpu_normal_backend_frame_implementation_validation,
        )
    )
    step78_true_visible_record_output = (
        build_step78_true_visible_record_summary(
            summary,
            webgpu_visible_record_camera_aware_visible_output,
            webgpu_owned_camera_aware_visible_output,
            validation_assisted_camera_aware_visible_output,
            webgpu_normal_backend_frame_implementation,
            webgpu_normal_backend_frame_implementation_validation,
        )
    )
    step79_4d_state_visible_pipeline = (
        build_step79_4d_state_visible_pipeline_summary(
            summary,
            webgpu_visible_record_camera_aware_visible_output,
            webgpu_owned_camera_aware_visible_output,
            validation_assisted_camera_aware_visible_output,
            webgpu_normal_backend_frame_implementation,
            webgpu_normal_backend_frame_implementation_validation,
        )
    )
    step80_webgpu_4d_state_evaluation_pipeline = (
        build_step80_webgpu_4d_state_evaluation_summary(
            summary,
            webgpu_visible_record_camera_aware_visible_output,
            webgpu_owned_camera_aware_visible_output,
            validation_assisted_camera_aware_visible_output,
            webgpu_normal_backend_frame_implementation,
            webgpu_normal_backend_frame_implementation_validation,
        )
    )
    step81_webgpu_gaussian_attribute_evaluation_pipeline = (
        build_step81_webgpu_gaussian_attribute_evaluation_summary(
            summary,
            webgpu_visible_record_camera_aware_visible_output,
            webgpu_owned_camera_aware_visible_output,
            validation_assisted_camera_aware_visible_output,
            webgpu_normal_backend_frame_implementation,
            webgpu_normal_backend_frame_implementation_validation,
        )
    )
    step82_webgpu_gaussian_footprint_pipeline = (
        build_step82_webgpu_gaussian_footprint_pipeline_summary(
            summary,
            webgpu_visible_record_camera_aware_visible_output,
            webgpu_owned_camera_aware_visible_output,
            validation_assisted_camera_aware_visible_output,
            webgpu_normal_backend_frame_implementation,
            webgpu_normal_backend_frame_implementation_validation,
        )
    )
    step83_webgpu_tile_aware_render_input_pipeline = (
        build_step83_webgpu_tile_aware_render_input_summary(
            summary,
            webgpu_visible_record_camera_aware_visible_output,
            webgpu_owned_camera_aware_visible_output,
            validation_assisted_camera_aware_visible_output,
            webgpu_normal_backend_frame_implementation,
            webgpu_normal_backend_frame_implementation_validation,
        )
    )
    step84_webgpu_gpu_owned_tile_list_layout_pipeline = (
        build_step84_webgpu_gpu_owned_tile_list_layout_summary(
            summary,
            webgpu_visible_record_camera_aware_visible_output,
            webgpu_owned_camera_aware_visible_output,
            validation_assisted_camera_aware_visible_output,
            webgpu_normal_backend_frame_implementation,
            webgpu_normal_backend_frame_implementation_validation,
        )
    )
    step85_webgpu_tile_list_compositor_pipeline = (
        build_step85_webgpu_tile_list_compositor_summary(
            summary,
            webgpu_visible_record_camera_aware_visible_output,
            webgpu_owned_camera_aware_visible_output,
            validation_assisted_camera_aware_visible_output,
            webgpu_normal_backend_frame_implementation,
            webgpu_normal_backend_frame_implementation_validation,
        )
    )
    step86_backend_boundary_and_dirty_contract = (
        build_step86_backend_boundary_and_dirty_contract_summary(summary)
    )
    step87_tile_depth_ordering_for_compositor = (
        build_step87_tile_depth_ordering_summary(summary)
    )
    step88_tile_compositor_frame_implementation = (
        build_step88_tile_compositor_frame_implementation_summary(summary)
    )
    step89_real_tile_compositor_output = (
        build_step89_real_tile_compositor_output_summary(summary)
    )
    step90_realtime_runtime_path = (
        build_step90_realtime_runtime_path_summary(summary)
    )
    step91_gpu_side_tile_ordering = (
        build_step91_gpu_side_tile_ordering_summary(summary)
    )
    step92_per_tile_depth_sort = (
        build_step92_per_tile_depth_sort_summary(summary)
    )
    step93_overflow_aware_tile_ordering = (
        build_step93_overflow_aware_tile_ordering_summary(summary)
    )
    step94_parallel_per_tile_sort = (
        build_step94_parallel_per_tile_sort_summary(summary)
    )
    step96_production_tile_compositor = (
        build_step96_production_tile_compositor_summary(summary)
    )
    step97_time_driven_production_runtime = (
        build_step97_time_driven_production_runtime_summary(summary)
    )
    step98_viewer_connected_interactive_scheduler = (
        build_step98_viewer_connected_interactive_scheduler_summary(summary)
    )
    step99_interactive_camera_dirty_runtime = (
        build_step99_interactive_camera_dirty_runtime_summary(summary)
    )
    step100_unified_interaction_scheduler_runtime = (
        build_step100_unified_interaction_scheduler_runtime_summary(summary)
    )
    step101_selective_dirty_dependency_execution = (
        build_step101_selective_dirty_dependency_execution_summary(summary)
    )
    step102_production_resource_lifecycle = (
        build_step102_production_resource_lifecycle_summary(summary)
    )
    step103_production_runtime_boundary_review = (
        build_step103_runtime_boundary_work_reduction_summary(summary)
    )
    step104_compositor_work_reduction_visual_safety = (
        build_step104_compositor_work_reduction_visual_safety_summary(summary)
    )
    step105_cuda_reference_visual_parity_baseline = (
        build_step105_cuda_reference_visual_parity_baseline_summary(summary)
    )
    step106_capability_based_regression_gate = (
        build_step106_capability_based_regression_gate_summary(summary)
    )
    step107_fixed_reference_camera_contract = (
        build_step107_fixed_reference_camera_contract_summary(summary)
    )
    step108_cuda_reference_camera_evidence = (
        build_step108_cuda_reference_camera_evidence_summary(summary)
    )
    step109_fixed_reference_camera_activation = (
        build_step109_fixed_reference_camera_activation_summary(summary)
    )
    return {
        "status": get_path(summary, ["status"]),
        "reason": get_path(summary, ["reason"]),
        "computeMode": get_path(summary, ["computeMode"]),
        "scaffoldMode": get_path(summary, ["scaffoldMode"]),
        "phaseStep": get_path(summary, ["phaseStep"]),
        "captureStatus": "ok",
        "captureReason": "compare-json-present",
        "captureFatalError": False,
        "captureExceptionRecorded": False,
        "captureErrorName": None,
        "captureErrorMessage": None,
        "captureSource": get_path(summary, ["metadata", "captureSource"]),
        "viewerDataReadiness": get_path(summary, ["metadata", "viewerDataReadiness"], {}),
        "candidateInputSource": get_path(summary, ["metadata", "candidateInputSource"]),
        "candidateInputReason": get_path(summary, ["metadata", "candidateInputReason"]),
        "implementedFields": get_path(summary, ["implementedFields"], []),
        "wgslComputedFields": get_path(summary, ["wgslComputedFields"], []),
        "wgslReferenceAssistedFields": get_path(
            summary, ["wgslReferenceAssistedFields"], []
        ),
        "cpuMaterializedFields": get_path(summary, ["cpuMaterializedFields"], []),
        "fieldComputeModes": get_path(summary, ["fieldComputeModes"], {}),
        "deferredFields": get_path(summary, ["deferredFields"], []),
        "step75CameraAwareVisibleOutput": step75_camera_aware_visible_output,
        "step76ManyCameraAwareVisibleOutput": step76_many_camera_aware_visible_output,
        "step77WebGpuOwnedVisibleOutput": step77_webgpu_owned_visible_output,
        "step78TrueVisibleRecordOutput": step78_true_visible_record_output,
        "step79WebGpu4DStateVisiblePipeline": step79_4d_state_visible_pipeline,
        "step80WebGpu4DStateEvaluationPipeline":
            step80_webgpu_4d_state_evaluation_pipeline,
        "step81WebGpuGaussianAttributeEvaluationPipeline":
            step81_webgpu_gaussian_attribute_evaluation_pipeline,
        "step82WebGpuGaussianFootprintPipeline":
            step82_webgpu_gaussian_footprint_pipeline,
        "step83WebGpuTileAwareRenderInputPipeline":
            step83_webgpu_tile_aware_render_input_pipeline,
        "step84WebGpuGpuOwnedTileListLayoutPipeline":
            step84_webgpu_gpu_owned_tile_list_layout_pipeline,
        "step85WebGpuTileListCompositorPipeline":
            step85_webgpu_tile_list_compositor_pipeline,
        "step86BackendBoundaryAndDirtyContract":
            step86_backend_boundary_and_dirty_contract,
        "step87TileDepthOrderingForCompositor":
            step87_tile_depth_ordering_for_compositor,
        "step88TileCompositorFrameImplementation":
            step88_tile_compositor_frame_implementation,
        "step89RealTileCompositorOutput":
            step89_real_tile_compositor_output,
        "step90RealtimeRuntimePath":
            step90_realtime_runtime_path,
        "step91GpuSideTileOrdering":
            step91_gpu_side_tile_ordering,
        "step92PerTileDepthSort":
            step92_per_tile_depth_sort,
        "step93OverflowAwareTileOrdering":
            step93_overflow_aware_tile_ordering,
        "step94ParallelPerTileSort":
            step94_parallel_per_tile_sort,
        "step96ProductionTileCompositor":
            step96_production_tile_compositor,
        "step97TimeDrivenProductionRuntime":
            step97_time_driven_production_runtime,
        "step98ViewerConnectedInteractiveScheduler":
            step98_viewer_connected_interactive_scheduler,
        "step99InteractiveCameraDirtyRuntime":
            step99_interactive_camera_dirty_runtime,
        "step100UnifiedInteractionSchedulerRuntime":
            step100_unified_interaction_scheduler_runtime,
        "step101SelectiveDirtyDependencyExecution":
            step101_selective_dirty_dependency_execution,
        "step102ProductionResourceLifecycle":
            step102_production_resource_lifecycle,
        "step103ProductionRuntimeBoundaryReview":
            step103_production_runtime_boundary_review,
        "step104CompositorWorkReductionVisualSafety":
            step104_compositor_work_reduction_visual_safety,
        "step105CudaReferenceVisualParityBaseline":
            step105_cuda_reference_visual_parity_baseline,
        "step106CapabilityBasedRegressionGate":
            step106_capability_based_regression_gate,
        "step107FixedReferenceCameraContract":
            step107_fixed_reference_camera_contract,
        "step108CudaReferenceCameraEvidence":
            step108_cuda_reference_camera_evidence,
        "step109FixedReferenceCameraActivation":
            step109_fixed_reference_camera_activation,
        "comparisonContract": get_path(summary, ["comparisonContract"], {}),
        "comparisonTolerance": get_path(summary, ["comparisonTolerance"], {}),
        "radiusContract": get_path(summary, ["radiusContract"], {}),
        "radiusComputeMode": get_path(summary, ["radiusComputeMode"]),
        "covarianceContract": get_path(summary, ["covarianceContract"], {}),
        "conicContract": get_path(summary, ["conicContract"], {}),
        "conicComputeMode": get_path(summary, ["conicComputeMode"]),
        "aabbContract": get_path(summary, ["aabbContract"], {}),
        "tileRangeContract": get_path(summary, ["tileRangeContract"], {}),
        "boundsComputeMode": get_path(summary, ["boundsComputeMode"], {}),
        "tileListContract": get_path(summary, ["tileListContract"], {}),
        "tileListComputeMode": get_path(summary, ["tileListComputeMode"]),
        "tileListCapacityContract": get_path(
            summary, ["tileListCapacityContract"], {}
        ),
        "tileListCapacityComputeMode": get_path(
            summary, ["tileListCapacityComputeMode"]
        ),
        "tileListValidationContract": get_path(
            summary, ["tileListValidationContract"], {}
        ),
        "tileListValidationComputeMode": get_path(
            summary, ["tileListValidationComputeMode"]
        ),
        "tileListValidationUnitContract": get_path(
            summary, ["tileListValidationUnitContract"], {}
        ),
        "tileListValidationUnitComputeMode": get_path(
            summary, ["tileListValidationUnitComputeMode"]
        ),
        "tileCountsOffsetsComparisonSurfaceContract": get_path(
            summary, ["tileCountsOffsetsComparisonSurfaceContract"], {}
        ),
        "tileCountsOffsetsComparisonSurfaceComputeMode": get_path(
            summary, ["tileCountsOffsetsComparisonSurfaceComputeMode"]
        ),
        "tileCountsToOffsetsDryRun": {
            "status": get_path(counts_offsets, ["status"]),
            "mode": get_path(counts_offsets, ["mode"]),
            "computeMode": get_path(counts_offsets, ["computeMode"]),
            "implementedInWgsl": get_path(counts_offsets, ["implementedInWgsl"]),
            "scatterImplemented": get_path(counts_offsets, ["scatterImplemented"]),
            "tileGrid": get_path(counts_offsets, ["tileGrid"], {}),
            "recordCounts": get_path(counts_offsets, ["recordCounts"], {}),
            "tileCountsValid": get_path(
                counts_offsets_validation, ["tileCountsValid"]
            ),
            "prefixOffsetsValid": get_path(
                counts_offsets_validation, ["prefixOffsetsValid"]
            ),
            "totalTileRefsConsistent": get_path(
                counts_offsets_validation, ["totalTileRefsConsistent"]
            ),
            "capacityStatus": get_path(
                counts_offsets_validation, ["capacityStatus"]
            ),
            "firstValidationFailures": get_path(
                counts_offsets_validation, ["firstValidationFailures"], []
            ),
            "tileOffsetsPolicy": get_path(
                counts_offsets_metadata, ["tileOffsetsPolicy"]
            ),
            "tileOffsetsTerminalValue": get_path(
                counts_offsets_metadata, ["tileOffsetsTerminalValue"]
            ),
            "totalTileRefs": get_path(counts_offsets_metadata, ["totalTileRefs"]),
            "maxRefsPerTile": get_path(
                counts_offsets_capacity, ["maxRefsPerTile"]
            ),
            "nonEmptyTiles": get_path(counts_offsets_capacity, ["nonEmptyTiles"]),
        },
        "tileCountsOffsetsSelfComparison": {
            "status": get_path(counts_offsets_self_comparison, ["status"]),
            "mode": get_path(counts_offsets_self_comparison, ["mode"]),
            "expectedSource": get_path(
                counts_offsets_self_comparison, ["expectedSource"]
            ),
            "actualSource": get_path(counts_offsets_self_comparison, ["actualSource"]),
            "implementedInWgsl": get_path(
                counts_offsets_self_comparison, ["implementedInWgsl"]
            ),
            "webgpuComputed": get_path(
                counts_offsets_self_comparison, ["webgpuComputed"]
            ),
            "scatterCompared": get_path(
                counts_offsets_self_comparison, ["scatterCompared"]
            ),
            "anyMismatch": get_path(counts_offsets_self_comparison, ["anyMismatch"]),
            "mismatchClassification": get_path(
                counts_offsets_self_comparison, ["mismatchClassification"]
            ),
            "tileCountsMismatchCount": get_path(
                counts_offsets_self_comparison, ["tileCountsMismatchCount"]
            ),
            "tileOffsetsMismatchCount": get_path(
                counts_offsets_self_comparison, ["tileOffsetsMismatchCount"]
            ),
            "totalTileRefsMismatch": get_path(
                counts_offsets_self_comparison, ["totalTileRefsMismatch"]
            ),
            "capacityStatusMismatch": get_path(
                counts_offsets_self_comparison, ["capacityStatusMismatch"]
            ),
            "maxAbsCountDelta": get_path(
                counts_offsets_self_comparison, ["maxAbsCountDelta"]
            ),
            "maxAbsOffsetDelta": get_path(
                counts_offsets_self_comparison, ["maxAbsOffsetDelta"]
            ),
            "firstMismatches": compact_list(
                get_path(counts_offsets_self_comparison, ["firstMismatches"], [])
            ),
            "sampleTiles": compact_list(
                get_path(counts_offsets_self_comparison, ["sampleTiles"], [])
            ),
        },
        "webgpuTileCountsDryRun": {
            "status": get_path(webgpu_tile_counts, ["status"]),
            "mode": get_path(webgpu_tile_counts, ["mode"]),
            "implementedInWgsl": get_path(webgpu_tile_counts, ["implementedInWgsl"]),
            "tileOffsetsComputed": get_path(
                webgpu_tile_counts, ["tileOffsetsComputed"]
            ),
            "prefixSumImplemented": get_path(
                webgpu_tile_counts, ["prefixSumImplemented"]
            ),
            "scatterImplemented": get_path(webgpu_tile_counts, ["scatterImplemented"]),
            "tileGrid": get_path(webgpu_tile_counts, ["tileGrid"], {}),
            "recordCounts": get_path(webgpu_tile_counts, ["recordCounts"], {}),
            "totalTileRefs": get_path(
                webgpu_tile_counts, ["metadata.totalTileRefs"]
            ),
            "maxRefsPerTile": get_path(
                webgpu_tile_counts, ["metadata.maxRefsPerTile"]
            ),
            "nonEmptyTiles": get_path(
                webgpu_tile_counts, ["metadata.nonEmptyTiles"]
            ),
            "capacityStatus": get_path(
                webgpu_tile_counts, ["capacity.capacityStatus"]
            ),
        },
        "tileCountsWebGpuComparison": {
            "status": get_path(tile_counts_webgpu_comparison, ["status"]),
            "mode": get_path(tile_counts_webgpu_comparison, ["mode"]),
            "expectedSource": get_path(
                tile_counts_webgpu_comparison, ["expectedSource"]
            ),
            "actualSource": get_path(
                tile_counts_webgpu_comparison, ["actualSource"]
            ),
            "implementedInWgsl": get_path(
                tile_counts_webgpu_comparison, ["implementedInWgsl"]
            ),
            "webgpuComputed": get_path(
                tile_counts_webgpu_comparison, ["webgpuComputed"]
            ),
            "tileOffsetsCompared": get_path(
                tile_counts_webgpu_comparison, ["tileOffsetsCompared"]
            ),
            "prefixSumImplemented": get_path(
                tile_counts_webgpu_comparison, ["prefixSumImplemented"]
            ),
            "scatterCompared": get_path(
                tile_counts_webgpu_comparison, ["scatterCompared"]
            ),
            "anyMismatch": get_path(tile_counts_webgpu_comparison, ["anyMismatch"]),
            "mismatchClassification": get_path(
                tile_counts_webgpu_comparison, ["mismatchClassification"]
            ),
            "tileCountsMismatchCount": get_path(
                tile_counts_webgpu_comparison, ["tileCountsMismatchCount"]
            ),
            "totalTileRefsMismatch": get_path(
                tile_counts_webgpu_comparison, ["totalTileRefsMismatch"]
            ),
            "capacityStatusMismatch": get_path(
                tile_counts_webgpu_comparison, ["capacityStatusMismatch"]
            ),
            "maxAbsCountDelta": get_path(
                tile_counts_webgpu_comparison, ["maxAbsCountDelta"]
            ),
            "firstMismatches": compact_list(
                get_path(tile_counts_webgpu_comparison, ["firstMismatches"], [])
            ),
            "sampleTiles": compact_list(
                get_path(tile_counts_webgpu_comparison, ["sampleTiles"], [])
            ),
        },
        "tileOffsetsFromWebGpuCountsDryRun": {
            "status": get_path(offsets_from_webgpu_counts, ["status"]),
            "mode": get_path(offsets_from_webgpu_counts, ["mode"]),
            "source": get_path(offsets_from_webgpu_counts, ["source"]),
            "implementedInWgsl": get_path(
                offsets_from_webgpu_counts, ["implementedInWgsl"]
            ),
            "webgpuPrefixComputed": get_path(
                offsets_from_webgpu_counts, ["webgpuPrefixComputed"]
            ),
            "scatterImplemented": get_path(
                offsets_from_webgpu_counts, ["scatterImplemented"]
            ),
            "tileGrid": get_path(offsets_from_webgpu_counts, ["tileGrid"], {}),
            "recordCounts": get_path(
                offsets_from_webgpu_counts, ["recordCounts"], {}
            ),
            "prefixOffsetsValid": get_path(
                offsets_from_webgpu_counts_validation, ["prefixOffsetsValid"]
            ),
            "totalTileRefsConsistent": get_path(
                offsets_from_webgpu_counts_validation,
                ["totalTileRefsConsistent"],
            ),
            "capacityStatus": get_path(
                offsets_from_webgpu_counts_validation, ["capacityStatus"]
            ),
            "firstValidationFailures": get_path(
                offsets_from_webgpu_counts_validation,
                ["firstValidationFailures"],
                [],
            ),
            "tileOffsetsPolicy": get_path(
                offsets_from_webgpu_counts_metadata, ["tileOffsetsPolicy"]
            ),
            "tileOffsetsTerminalValue": get_path(
                offsets_from_webgpu_counts_metadata, ["tileOffsetsTerminalValue"]
            ),
            "totalTileRefs": get_path(
                offsets_from_webgpu_counts_metadata, ["totalTileRefs"]
            ),
            "maxRefsPerTile": get_path(
                offsets_from_webgpu_counts_capacity, ["maxRefsPerTile"]
            ),
            "nonEmptyTiles": get_path(
                offsets_from_webgpu_counts_capacity, ["nonEmptyTiles"]
            ),
        },
        "tileOffsetsPrefixComparison": {
            "status": get_path(tile_offsets_prefix_comparison, ["status"]),
            "mode": get_path(tile_offsets_prefix_comparison, ["mode"]),
            "expectedSource": get_path(
                tile_offsets_prefix_comparison, ["expectedSource"]
            ),
            "actualSource": get_path(
                tile_offsets_prefix_comparison, ["actualSource"]
            ),
            "implementedInWgsl": get_path(
                tile_offsets_prefix_comparison, ["implementedInWgsl"]
            ),
            "webgpuComputed": get_path(
                tile_offsets_prefix_comparison, ["webgpuComputed"]
            ),
            "webgpuPrefixComputed": get_path(
                tile_offsets_prefix_comparison, ["webgpuPrefixComputed"]
            ),
            "scatterCompared": get_path(
                tile_offsets_prefix_comparison, ["scatterCompared"]
            ),
            "anyMismatch": get_path(
                tile_offsets_prefix_comparison, ["anyMismatch"]
            ),
            "mismatchClassification": get_path(
                tile_offsets_prefix_comparison, ["mismatchClassification"]
            ),
            "tileOffsetsMismatchCount": get_path(
                tile_offsets_prefix_comparison, ["tileOffsetsMismatchCount"]
            ),
            "totalTileRefsMismatch": get_path(
                tile_offsets_prefix_comparison, ["totalTileRefsMismatch"]
            ),
            "capacityStatusMismatch": get_path(
                tile_offsets_prefix_comparison, ["capacityStatusMismatch"]
            ),
            "maxAbsOffsetDelta": get_path(
                tile_offsets_prefix_comparison, ["maxAbsOffsetDelta"]
            ),
            "prefixOffsetsValid": get_path(
                tile_offsets_prefix_comparison, ["prefixOffsetsValid"]
            ),
            "totalTileRefsConsistent": get_path(
                tile_offsets_prefix_comparison, ["totalTileRefsConsistent"]
            ),
            "firstMismatches": compact_list(
                get_path(tile_offsets_prefix_comparison, ["firstMismatches"], [])
            ),
            "sampleTiles": compact_list(
                get_path(tile_offsets_prefix_comparison, ["sampleTiles"], [])
            ),
        },
        "webgpuTileOffsetsPrefixDryRun": {
            "status": get_path(webgpu_tile_offsets_prefix, ["status"]),
            "mode": get_path(webgpu_tile_offsets_prefix, ["mode"]),
            "source": get_path(webgpu_tile_offsets_prefix, ["source"]),
            "implementedInWgsl": get_path(
                webgpu_tile_offsets_prefix, ["implementedInWgsl"]
            ),
            "webgpuPrefixComputed": get_path(
                webgpu_tile_offsets_prefix, ["webgpuPrefixComputed"]
            ),
            "scatterImplemented": get_path(
                webgpu_tile_offsets_prefix, ["scatterImplemented"]
            ),
            "tileGrid": get_path(webgpu_tile_offsets_prefix, ["tileGrid"], {}),
            "recordCounts": get_path(
                webgpu_tile_offsets_prefix, ["recordCounts"], {}
            ),
            "prefixOffsetsValid": get_path(
                webgpu_tile_offsets_prefix_validation, ["prefixOffsetsValid"]
            ),
            "totalTileRefsConsistent": get_path(
                webgpu_tile_offsets_prefix_validation,
                ["totalTileRefsConsistent"],
            ),
            "capacityStatus": get_path(
                webgpu_tile_offsets_prefix_validation, ["capacityStatus"]
            ),
            "firstValidationFailures": get_path(
                webgpu_tile_offsets_prefix_validation,
                ["firstValidationFailures"],
                [],
            ),
            "tileOffsetsPolicy": get_path(
                webgpu_tile_offsets_prefix_metadata, ["tileOffsetsPolicy"]
            ),
            "tileOffsetsTerminalValue": get_path(
                webgpu_tile_offsets_prefix_metadata, ["tileOffsetsTerminalValue"]
            ),
            "totalTileRefs": get_path(
                webgpu_tile_offsets_prefix_metadata, ["totalTileRefs"]
            ),
            "maxRefsPerTile": get_path(
                webgpu_tile_offsets_prefix_capacity, ["maxRefsPerTile"]
            ),
            "nonEmptyTiles": get_path(
                webgpu_tile_offsets_prefix_capacity, ["nonEmptyTiles"]
            ),
        },
        "tileOffsetsWebGpuPrefixComparison": {
            "status": get_path(tile_offsets_webgpu_prefix_comparison, ["status"]),
            "mode": get_path(tile_offsets_webgpu_prefix_comparison, ["mode"]),
            "expectedSource": get_path(
                tile_offsets_webgpu_prefix_comparison, ["expectedSource"]
            ),
            "actualSource": get_path(
                tile_offsets_webgpu_prefix_comparison, ["actualSource"]
            ),
            "implementedInWgsl": get_path(
                tile_offsets_webgpu_prefix_comparison, ["implementedInWgsl"]
            ),
            "webgpuComputed": get_path(
                tile_offsets_webgpu_prefix_comparison, ["webgpuComputed"]
            ),
            "webgpuPrefixComputed": get_path(
                tile_offsets_webgpu_prefix_comparison, ["webgpuPrefixComputed"]
            ),
            "scatterCompared": get_path(
                tile_offsets_webgpu_prefix_comparison, ["scatterCompared"]
            ),
            "anyMismatch": get_path(
                tile_offsets_webgpu_prefix_comparison, ["anyMismatch"]
            ),
            "mismatchClassification": get_path(
                tile_offsets_webgpu_prefix_comparison,
                ["mismatchClassification"],
            ),
            "tileOffsetsMismatchCount": get_path(
                tile_offsets_webgpu_prefix_comparison,
                ["tileOffsetsMismatchCount"],
            ),
            "totalTileRefsMismatch": get_path(
                tile_offsets_webgpu_prefix_comparison,
                ["totalTileRefsMismatch"],
            ),
            "capacityStatusMismatch": get_path(
                tile_offsets_webgpu_prefix_comparison,
                ["capacityStatusMismatch"],
            ),
            "maxAbsOffsetDelta": get_path(
                tile_offsets_webgpu_prefix_comparison, ["maxAbsOffsetDelta"]
            ),
            "prefixOffsetsValid": get_path(
                tile_offsets_webgpu_prefix_comparison, ["prefixOffsetsValid"]
            ),
            "totalTileRefsConsistent": get_path(
                tile_offsets_webgpu_prefix_comparison,
                ["totalTileRefsConsistent"],
            ),
            "firstMismatches": compact_list(
                get_path(
                    tile_offsets_webgpu_prefix_comparison,
                    ["firstMismatches"],
                    [],
                )
            ),
            "sampleTiles": compact_list(
                get_path(
                    tile_offsets_webgpu_prefix_comparison,
                    ["sampleTiles"],
                    [],
                )
            ),
        },
        "scatterValidationBoundary": {
            "status": get_path(scatter_validation_boundary, ["status"]),
            "mode": get_path(scatter_validation_boundary, ["mode"]),
            "source": get_path(scatter_validation_boundary, ["source"]),
            "implementedInWgsl": get_path(
                scatter_validation_boundary, ["implementedInWgsl"]
            ),
            "webgpuScatterComputed": get_path(
                scatter_validation_boundary, ["webgpuScatterComputed"]
            ),
            "tileIndicesMaterialized": get_path(
                scatter_validation_boundary, ["tileIndicesMaterialized"]
            ),
            "scatterCompared": get_path(
                scatter_validation_boundary, ["scatterCompared"]
            ),
            "tileGrid": get_path(scatter_validation_boundary, ["tileGrid"], {}),
            "recordCounts": get_path(
                scatter_validation_boundary, ["recordCounts"], {}
            ),
            "writeCursorPolicy": get_path(
                scatter_validation_boundary, ["writeCursorPolicy"], {}
            ),
            "writeCursorInitialValid": get_path(
                scatter_validation_summary, ["writeCursorInitialValid"]
            ),
            "writeCursorFinalValid": get_path(
                scatter_validation_summary, ["writeCursorFinalValid"]
            ),
            "scatterOutputValid": get_path(
                scatter_validation_summary, ["scatterOutputValid"]
            ),
            "totalTileRefsConsistent": get_path(
                scatter_validation_summary, ["totalTileRefsConsistent"]
            ),
            "capacityStatus": get_path(
                scatter_validation_summary, ["capacityStatus"]
            ),
            "webgpuPrefixMatchesReference": get_path(
                scatter_validation_summary, ["webgpuPrefixMatchesReference"]
            ),
            "firstValidationFailures": get_path(
                scatter_validation_summary, ["firstValidationFailures"], []
            ),
            "totalTileRefs": get_path(
                scatter_validation_capacity, ["totalTileRefs"]
            ),
            "maxRefsPerTile": get_path(
                scatter_validation_capacity, ["maxRefsPerTile"]
            ),
            "nonEmptyTiles": get_path(
                scatter_validation_capacity, ["nonEmptyTiles"]
            ),
            "capacityOverflowCount": get_path(
                scatter_validation_capacity, ["capacityOverflowCount"]
            ),
            "firstScatterWrites": compact_list(
                get_path(scatter_validation_boundary, ["firstScatterWrites"], [])
            ),
            "sampleTiles": compact_list(
                get_path(scatter_validation_boundary, ["sampleTiles"], [])
            ),
        },
        "tileIndicesSelfComparison": {
            "status": get_path(tile_indices_self_comparison, ["status"]),
            "mode": get_path(tile_indices_self_comparison, ["mode"]),
            "expectedSource": get_path(
                tile_indices_self_comparison, ["expectedSource"]
            ),
            "actualSource": get_path(
                tile_indices_self_comparison, ["actualSource"]
            ),
            "implementedInWgsl": get_path(
                tile_indices_self_comparison, ["implementedInWgsl"]
            ),
            "webgpuScatterComputed": get_path(
                tile_indices_self_comparison, ["webgpuScatterComputed"]
            ),
            "tileIndicesMaterialized": get_path(
                tile_indices_self_comparison, ["tileIndicesMaterialized"]
            ),
            "tileIndicesStoredInJson": get_path(
                tile_indices_self_comparison, ["tileIndicesStoredInJson"]
            ),
            "scatterCompared": get_path(
                tile_indices_self_comparison, ["scatterCompared"]
            ),
            "anyMismatch": get_path(tile_indices_self_comparison, ["anyMismatch"]),
            "mismatchClassification": get_path(
                tile_indices_self_comparison, ["mismatchClassification"]
            ),
            "tileIndicesMismatchCount": get_path(
                tile_indices_self_comparison, ["tileIndicesMismatchCount"]
            ),
            "orderingMismatchCount": get_path(
                tile_indices_self_comparison, ["orderingMismatchCount"]
            ),
            "capacityStatusMismatch": get_path(
                tile_indices_self_comparison, ["capacityStatusMismatch"]
            ),
            "maxAbsIndexDelta": get_path(
                tile_indices_self_comparison, ["maxAbsIndexDelta"]
            ),
            "recordCounts": get_path(
                tile_indices_self_comparison, ["recordCounts"], {}
            ),
            "capacityStatus": get_path(
                tile_indices_self_capacity, ["capacityStatus"]
            ),
            "capacityOverflowCount": get_path(
                tile_indices_self_capacity, ["capacityOverflowCount"]
            ),
            "totalTileRefs": get_path(
                tile_indices_self_capacity, ["totalTileRefs"]
            ),
            "orderingPolicy": get_path(
                tile_indices_self_comparison, ["orderingPolicy"], {}
            ),
            "firstMismatches": compact_list(
                get_path(tile_indices_self_comparison, ["firstMismatches"], [])
            ),
            "sampleTiles": compact_list(
                get_path(tile_indices_self_comparison, ["sampleTiles"], [])
            ),
        },
        "tileIndicesWebGpuScatterComparison": {
            "status": get_path(tile_indices_webgpu_scatter_comparison, ["status"]),
            "mode": get_path(tile_indices_webgpu_scatter_comparison, ["mode"]),
            "expectedSource": get_path(
                tile_indices_webgpu_scatter_comparison, ["expectedSource"]
            ),
            "actualSource": get_path(
                tile_indices_webgpu_scatter_comparison, ["actualSource"]
            ),
            "source": get_path(tile_indices_webgpu_scatter_comparison, ["source"]),
            "implementedInWgsl": get_path(
                tile_indices_webgpu_scatter_comparison, ["implementedInWgsl"]
            ),
            "webgpuScatterComputed": get_path(
                tile_indices_webgpu_scatter_comparison, ["webgpuScatterComputed"]
            ),
            "tileIndicesMaterialized": get_path(
                tile_indices_webgpu_scatter_comparison, ["tileIndicesMaterialized"]
            ),
            "tileIndicesStoredInJson": get_path(
                tile_indices_webgpu_scatter_comparison, ["tileIndicesStoredInJson"]
            ),
            "scatterCompared": get_path(
                tile_indices_webgpu_scatter_comparison, ["scatterCompared"]
            ),
            "fullTileListGeneration": get_path(
                tile_indices_webgpu_scatter_comparison, ["fullTileListGeneration"]
            ),
            "sortImplemented": get_path(
                tile_indices_webgpu_scatter_comparison, ["sortImplemented"]
            ),
            "displayConnectionImplemented": get_path(
                tile_indices_webgpu_scatter_comparison,
                ["displayConnectionImplemented"],
            ),
            "anyMismatch": get_path(
                tile_indices_webgpu_scatter_comparison, ["anyMismatch"]
            ),
            "mismatchClassification": get_path(
                tile_indices_webgpu_scatter_comparison,
                ["mismatchClassification"],
            ),
            "tileIndicesMismatchCount": get_path(
                tile_indices_webgpu_scatter_comparison,
                ["tileIndicesMismatchCount"],
            ),
            "orderingMismatchCount": get_path(
                tile_indices_webgpu_scatter_comparison, ["orderingMismatchCount"]
            ),
            "writeCursorMismatchCount": get_path(
                tile_indices_webgpu_scatter_comparison,
                ["writeCursorMismatchCount"],
            ),
            "capacityStatusMismatch": get_path(
                tile_indices_webgpu_scatter_comparison,
                ["capacityStatusMismatch"],
            ),
            "maxAbsIndexDelta": get_path(
                tile_indices_webgpu_scatter_comparison, ["maxAbsIndexDelta"]
            ),
            "recordCounts": get_path(
                tile_indices_webgpu_scatter_comparison, ["recordCounts"], {}
            ),
            "capacityStatus": get_path(
                tile_indices_webgpu_scatter_capacity, ["capacityStatus"]
            ),
            "capacityOverflowCount": get_path(
                tile_indices_webgpu_scatter_capacity, ["capacityOverflowCount"]
            ),
            "totalTileRefs": get_path(
                tile_indices_webgpu_scatter_capacity, ["totalTileRefs"]
            ),
            "writeCursorFinalValid": get_path(
                tile_indices_webgpu_scatter_validation,
                ["writeCursorFinalValid"],
            ),
            "scatterOutputValid": get_path(
                tile_indices_webgpu_scatter_validation, ["scatterOutputValid"]
            ),
            "orderingPolicy": get_path(
                tile_indices_webgpu_scatter_comparison, ["orderingPolicy"], {}
            ),
            "firstMismatches": compact_list(
                get_path(
                    tile_indices_webgpu_scatter_comparison,
                    ["firstMismatches"],
                    [],
                )
            ),
            "sampleTiles": compact_list(
                get_path(
                    tile_indices_webgpu_scatter_comparison, ["sampleTiles"], []
                )
            ),
        },
        "tileListSummaryComparison": {
            "status": get_path(tile_list_summary_comparison, ["status"]),
            "mode": get_path(tile_list_summary_comparison, ["mode"]),
            "source": get_path(tile_list_summary_comparison, ["source"]),
            "expectedSource": get_path(
                tile_list_summary_comparison, ["expectedSource"]
            ),
            "actualSource": get_path(
                tile_list_summary_comparison, ["actualSource"]
            ),
            "implementedInWgsl": get_path(
                tile_list_summary_comparison, ["implementedInWgsl"]
            ),
            "nonDisplayOnly": get_path(
                tile_list_summary_comparison, ["nonDisplayOnly"]
            ),
            "fullTileListGeneration": get_path(
                tile_list_summary_comparison, ["fullTileListGeneration"]
            ),
            "sortImplemented": get_path(
                tile_list_summary_comparison, ["sortImplemented"]
            ),
            "displayConnectionImplemented": get_path(
                tile_list_summary_comparison, ["displayConnectionImplemented"]
            ),
            "tileIndicesStoredInJson": get_path(
                tile_list_summary_comparison, ["tileIndicesStoredInJson"]
            ),
            "anyMismatch": get_path(tile_list_summary_comparison, ["anyMismatch"]),
            "mismatchClassification": get_path(
                tile_list_summary_comparison, ["mismatchClassification"]
            ),
            "stageStatuses": get_path(
                tile_list_summary_comparison, ["stageStatuses"], {}
            ),
            "mismatchCounts": get_path(
                tile_list_summary_comparison, ["mismatchCounts"], {}
            ),
            "totalTileRefsMismatch": get_path(
                tile_list_summary_metadata, ["totalTileRefsMismatch"]
            ),
            "capacityStatusMismatch": get_path(
                tile_list_summary_metadata, ["capacityStatusMismatch"]
            ),
            "expectedTotalTileRefs": get_path(
                tile_list_summary_metadata, ["expectedTotalTileRefs"]
            ),
            "actualTotalTileRefs": get_path(
                tile_list_summary_metadata, ["actualTotalTileRefs"]
            ),
            "capacityStatus": get_path(
                tile_list_summary_capacity, ["capacityStatus"]
            ),
            "capacityOverflowCount": get_path(
                tile_list_summary_capacity, ["capacityOverflowCount"]
            ),
            "countsValid": get_path(tile_list_summary_validation, ["countsValid"]),
            "offsetsValid": get_path(
                tile_list_summary_validation, ["offsetsValid"]
            ),
            "indicesValid": get_path(
                tile_list_summary_validation, ["indicesValid"]
            ),
            "orderingValid": get_path(
                tile_list_summary_validation, ["orderingValid"]
            ),
            "writeCursorFinalValid": get_path(
                tile_list_summary_validation, ["writeCursorFinalValid"]
            ),
            "scatterOutputValid": get_path(
                tile_list_summary_validation, ["scatterOutputValid"]
            ),
            "recordCounts": get_path(
                tile_list_summary_comparison, ["recordCounts"], {}
            ),
            "orderingPolicy": get_path(
                tile_list_summary_comparison, ["orderingPolicy"], {}
            ),
            "firstMismatches": compact_list(
                get_path(tile_list_summary_comparison, ["firstMismatches"], [])
            ),
            "sampleTiles": compact_list(
                get_path(tile_list_summary_comparison, ["sampleTiles"], [])
            ),
        },
        "webgpuTileListBackendOutput": {
            "status": get_path(webgpu_tile_list_backend_output, ["status"]),
            "mode": get_path(webgpu_tile_list_backend_output, ["mode"]),
            "source": get_path(webgpu_tile_list_backend_output, ["source"]),
            "backendStage": get_path(
                webgpu_tile_list_backend_output, ["backendStage"]
            ),
            "backendOutputReady": get_path(
                webgpu_tile_list_backend_output, ["backendOutputReady"]
            ),
            "nonDisplayOnly": get_path(
                webgpu_tile_list_backend_output, ["nonDisplayOnly"]
            ),
            "fullTileListGeneration": get_path(
                webgpu_tile_list_backend_output, ["fullTileListGeneration"]
            ),
            "sortImplemented": get_path(
                webgpu_tile_list_backend_output, ["sortImplemented"]
            ),
            "displayConnectionImplemented": get_path(
                webgpu_tile_list_backend_output, ["displayConnectionImplemented"]
            ),
            "tileIndicesStoredInJson": get_path(
                webgpu_tile_list_backend_output, ["tileIndicesStoredInJson"]
            ),
            "outputBuffers": get_path(
                webgpu_tile_list_backend_output, ["outputBuffers"], {}
            ),
            "tileGrid": get_path(webgpu_tile_list_backend_output, ["tileGrid"], {}),
            "recordCounts": get_path(
                webgpu_tile_list_backend_output, ["recordCounts"], {}
            ),
            "capacity": get_path(webgpu_tile_list_backend_output, ["capacity"], {}),
            "countsValid": get_path(
                webgpu_tile_list_backend_validation, ["countsValid"]
            ),
            "offsetsValid": get_path(
                webgpu_tile_list_backend_validation, ["offsetsValid"]
            ),
            "indicesValid": get_path(
                webgpu_tile_list_backend_validation, ["indicesValid"]
            ),
            "orderingValid": get_path(
                webgpu_tile_list_backend_validation, ["orderingValid"]
            ),
            "scatterOutputValid": get_path(
                webgpu_tile_list_backend_validation, ["scatterOutputValid"]
            ),
            "handoffStatus": get_path(
                webgpu_tile_list_backend_handoff, ["status"]
            ),
            "displayConnectionAllowed": get_path(
                webgpu_tile_list_backend_handoff, ["displayConnectionAllowed"]
            ),
            "handoffSatisfied": get_path(
                webgpu_tile_list_backend_handoff, ["satisfied"], []
            ),
            "handoffUnresolved": get_path(
                webgpu_tile_list_backend_handoff, ["unresolved"], []
            ),
            "handoffBlocked": compact_list(
                get_path(webgpu_tile_list_backend_handoff, ["blocked"], [])
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_tile_list_backend_output, ["nextBackendPrototypeStep"]
            ),
        },
        "renderPayloadSortReadiness": {
            "status": get_path(render_payload_sort_readiness, ["status"]),
            "mode": get_path(render_payload_sort_readiness, ["mode"]),
            "source": get_path(render_payload_sort_readiness, ["source"]),
            "backendOutputReady": get_path(
                render_payload_sort_readiness, ["backendOutputReady"]
            ),
            "displayConnectionAllowed": get_path(
                render_payload_sort_readiness, ["displayConnectionAllowed"]
            ),
            "sortImplemented": get_path(
                render_payload_sort_readiness, ["sortImplemented"]
            ),
            "compactionImplemented": get_path(
                render_payload_sort_readiness, ["compactionImplemented"]
            ),
            "renderPayloadGpuImplemented": get_path(
                render_payload_sort_readiness, ["renderPayloadGpuImplemented"]
            ),
            "tileCompositeImplemented": get_path(
                render_payload_sort_readiness, ["tileCompositeImplemented"]
            ),
            "payloadStatus": get_path(render_payload_readiness, ["status"]),
            "payloadRequiredFields": get_path(
                render_payload_readiness, ["requiredFields"], []
            ),
            "payloadMissingFields": get_path(
                render_payload_readiness, ["missingFields"], []
            ),
            "sortStatus": get_path(sort_readiness, ["status"]),
            "sortKey": get_path(sort_readiness, ["sortKey"]),
            "currentOrdering": get_path(sort_readiness, ["currentOrdering"]),
            "requiredOrdering": get_path(sort_readiness, ["requiredOrdering"]),
            "sortInputs": get_path(sort_readiness, ["inputs"], {}),
            "tileListBackendReady": get_path(
                readiness_summary, ["tileListBackendReady"]
            ),
            "renderPayloadReady": get_path(
                readiness_summary, ["renderPayloadReady"]
            ),
            "sortPrototypeReady": get_path(
                readiness_summary, ["sortPrototypeReady"]
            ),
            "nextRecommendedUnit": get_path(
                readiness_summary, ["nextRecommendedUnit"]
            ),
            "blockers": compact_list(
                get_path(render_payload_sort_readiness, ["blockers"], [])
            ),
            "nextBackendPrototypeStep": get_path(
                render_payload_sort_readiness, ["nextBackendPrototypeStep"]
            ),
        },
        "depthSortComparison": {
            "status": get_path(depth_sort_comparison, ["status"]),
            "mode": get_path(depth_sort_comparison, ["mode"]),
            "source": get_path(depth_sort_comparison, ["source"]),
            "expectedSource": get_path(
                depth_sort_comparison, ["expectedSource"]
            ),
            "actualSource": get_path(depth_sort_comparison, ["actualSource"]),
            "implementedInWgsl": get_path(
                depth_sort_comparison, ["implementedInWgsl"]
            ),
            "webgpuSortComputed": get_path(
                depth_sort_comparison, ["webgpuSortComputed"]
            ),
            "cpuStagedSortComputed": get_path(
                depth_sort_comparison, ["cpuStagedSortComputed"]
            ),
            "nonDisplayOnly": get_path(depth_sort_comparison, ["nonDisplayOnly"]),
            "displayConnectionAllowed": get_path(
                depth_sort_comparison, ["displayConnectionAllowed"]
            ),
            "tileCompositeImplemented": get_path(
                depth_sort_comparison, ["tileCompositeImplemented"]
            ),
            "sortedIndicesStoredInJson": get_path(
                depth_sort_comparison, ["sortedIndicesStoredInJson"]
            ),
            "anyMismatch": get_path(depth_sort_comparison, ["anyMismatch"]),
            "mismatchClassification": get_path(
                depth_sort_comparison, ["mismatchClassification"]
            ),
            "sortMismatchCount": get_path(
                depth_sort_comparison, ["sortMismatchCount"]
            ),
            "exactSortDifferenceCount": get_path(
                depth_sort_comparison, ["exactSortDifferenceCount"]
            ),
            "nearTieSortDifferenceCount": get_path(
                depth_sort_comparison, ["nearTieSortDifferenceCount"]
            ),
            "orderingMismatchCount": get_path(
                depth_sort_comparison, ["orderingMismatchCount"]
            ),
            "depthKeyMismatchCount": get_path(
                depth_sort_comparison, ["depthKeyMismatchCount"]
            ),
            "sortedTileMismatchCount": get_path(
                depth_sort_comparison, ["sortedTileMismatchCount"]
            ),
            "exactSortedTileDifferenceCount": get_path(
                depth_sort_comparison, ["exactSortedTileDifferenceCount"]
            ),
            "maxAbsDepthDelta": get_path(
                depth_sort_comparison, ["maxAbsDepthDelta"]
            ),
            "depthKeyPolicy": get_path(
                depth_sort_comparison, ["depthKeyPolicy"], {}
            ),
            "recordCounts": get_path(depth_sort_comparison, ["recordCounts"], {}),
            "sortOutputValid": get_path(
                depth_sort_validation, ["sortOutputValid"]
            ),
            "orderingValid": get_path(depth_sort_validation, ["orderingValid"]),
            "depthKeysValid": get_path(depth_sort_validation, ["depthKeysValid"]),
            "firstMismatches": compact_list(
                get_path(depth_sort_comparison, ["firstMismatches"], [])
            ),
            "firstSortDifferences": compact_list(
                get_path(depth_sort_comparison, ["firstSortDifferences"], [])
            ),
            "sampleTiles": compact_list(
                get_path(depth_sort_comparison, ["sampleTiles"], [])
            ),
        },
        "webgpuRenderHandoffStub": {
            "status": get_path(webgpu_render_handoff_stub, ["status"]),
            "mode": get_path(webgpu_render_handoff_stub, ["mode"]),
            "source": get_path(webgpu_render_handoff_stub, ["source"]),
            "backendOutputReady": get_path(
                webgpu_render_handoff_stub, ["backendOutputReady"]
            ),
            "depthSortReady": get_path(
                webgpu_render_handoff_stub, ["depthSortReady"]
            ),
            "renderHandoffStubReady": get_path(
                webgpu_render_handoff_stub, ["renderHandoffStubReady"]
            ),
            "displayConnectionAllowed": get_path(
                webgpu_render_handoff_stub, ["displayConnectionAllowed"]
            ),
            "tileCompositeImplemented": get_path(
                webgpu_render_handoff_stub, ["tileCompositeImplemented"]
            ),
            "renderPayloadGpuImplemented": get_path(
                webgpu_render_handoff_stub, ["renderPayloadGpuImplemented"]
            ),
            "partialPayloadMaterialized": get_path(
                webgpu_render_handoff_stub, ["partialPayloadMaterialized"]
            ),
            "referenceAssistedPayloadFields": get_path(
                webgpu_render_handoff_stub, ["referenceAssistedPayloadFields"], []
            ),
            "payloadStoredInJson": get_path(
                webgpu_render_handoff_stub, ["payloadStoredInJson"]
            ),
            "payloadLayout": get_path(
                webgpu_render_handoff_stub, ["payloadLayout"], {}
            ),
            "outputBuffer": get_path(
                webgpu_render_handoff_stub, ["outputBuffer"], {}
            ),
            "recordCounts": get_path(
                webgpu_render_handoff_stub, ["recordCounts"], {}
            ),
            "payloadShapeValid": get_path(
                render_handoff_validation, ["payloadShapeValid"]
            ),
            "populatedFieldsValid": get_path(
                render_handoff_validation, ["populatedFieldsValid"]
            ),
            "populatedFieldMismatchCount": get_path(
                render_handoff_validation, ["populatedFieldMismatchCount"]
            ),
            "referenceAssistedFieldsValid": get_path(
                render_handoff_validation, ["referenceAssistedFieldsValid"]
            ),
            "referenceAssistedFieldMismatchCount": get_path(
                render_handoff_validation, ["referenceAssistedFieldMismatchCount"]
            ),
            "referenceAssistedFieldMaxAbsDelta": get_path(
                render_handoff_validation, ["referenceAssistedFieldMaxAbsDelta"]
            ),
            "payloadFieldComparisonSummary": get_path(
                webgpu_render_handoff_stub, ["payloadFieldComparisonSummary"], {}
            ),
            "payloadFieldComparisons": get_path(
                webgpu_render_handoff_stub, ["payloadFieldComparisons"], {}
            ),
            "firstValidationFailures": compact_list(
                get_path(render_handoff_validation, ["firstValidationFailures"], [])
            ),
            "blockers": compact_list(
                get_path(webgpu_render_handoff_stub, ["blockers"], [])
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_render_handoff_stub, ["nextBackendPrototypeStep"]
            ),
            "sampleRecords": compact_list(
                get_path(webgpu_render_handoff_stub, ["sampleRecords"], [])
            ),
        },
        "webgpuTileCompositeHandoffStub": {
            "status": get_path(webgpu_tile_composite_handoff_stub, ["status"]),
            "mode": get_path(webgpu_tile_composite_handoff_stub, ["mode"]),
            "source": get_path(webgpu_tile_composite_handoff_stub, ["source"]),
            "nonDisplayOnly": get_path(
                webgpu_tile_composite_handoff_stub, ["nonDisplayOnly"]
            ),
            "tileCompositeHandoffStubImplemented": get_path(
                webgpu_tile_composite_handoff_stub,
                ["tileCompositeHandoffStubImplemented"],
            ),
            "tileCompositeHandoffStubReady": get_path(
                webgpu_tile_composite_handoff_stub,
                ["tileCompositeHandoffStubReady"],
            ),
            "tileCompositeImplemented": get_path(
                webgpu_tile_composite_handoff_stub, ["tileCompositeImplemented"]
            ),
            "framebufferImplemented": get_path(
                webgpu_tile_composite_handoff_stub, ["framebufferImplemented"]
            ),
            "displayConnectionAllowed": get_path(
                webgpu_tile_composite_handoff_stub, ["displayConnectionAllowed"]
            ),
            "shPolicy": get_path(
                webgpu_tile_composite_handoff_stub, ["shPolicy"], {}
            ),
            "payloadFieldsConsumed": get_path(
                webgpu_tile_composite_handoff_stub, ["payloadFieldsConsumed"], []
            ),
            "recordCounts": get_path(
                webgpu_tile_composite_handoff_stub, ["recordCounts"], {}
            ),
            "payloadShapeValid": get_path(
                tile_composite_handoff_validation, ["payloadShapeValid"]
            ),
            "tileOffsetsShapeValid": get_path(
                tile_composite_handoff_validation, ["tileOffsetsShapeValid"]
            ),
            "tileIndicesShapeValid": get_path(
                tile_composite_handoff_validation, ["tileIndicesShapeValid"]
            ),
            "depthSortReady": get_path(
                tile_composite_handoff_validation, ["depthSortReady"]
            ),
            "renderHandoffReady": get_path(
                tile_composite_handoff_validation, ["renderHandoffReady"]
            ),
            "tileListBackendReady": get_path(
                tile_composite_handoff_validation, ["tileListBackendReady"]
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    tile_composite_handoff_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(webgpu_tile_composite_handoff_stub, ["blockers"], [])
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_tile_composite_handoff_stub, ["nextBackendPrototypeStep"]
            ),
            "sampleTiles": compact_list(
                get_path(webgpu_tile_composite_handoff_stub, ["sampleTiles"], [])
            ),
        },
        "webgpuTileCompositeShaderHandoff": {
            "status": get_path(webgpu_tile_composite_shader_handoff, ["status"]),
            "mode": get_path(webgpu_tile_composite_shader_handoff, ["mode"]),
            "source": get_path(webgpu_tile_composite_shader_handoff, ["source"]),
            "nonDisplayOnly": get_path(
                webgpu_tile_composite_shader_handoff, ["nonDisplayOnly"]
            ),
            "tileCompositeShaderHandoffImplemented": get_path(
                webgpu_tile_composite_shader_handoff,
                ["tileCompositeShaderHandoffImplemented"],
            ),
            "tileCompositeShaderHandoffReady": get_path(
                webgpu_tile_composite_shader_handoff,
                ["tileCompositeShaderHandoffReady"],
            ),
            "tileCompositeShaderImplemented": get_path(
                webgpu_tile_composite_shader_handoff,
                ["tileCompositeShaderImplemented"],
            ),
            "tileCompositeImplemented": get_path(
                webgpu_tile_composite_shader_handoff, ["tileCompositeImplemented"]
            ),
            "framebufferImplemented": get_path(
                webgpu_tile_composite_shader_handoff, ["framebufferImplemented"]
            ),
            "displayConnectionAllowed": get_path(
                webgpu_tile_composite_shader_handoff, ["displayConnectionAllowed"]
            ),
            "orderedTileIndicesStoredInJson": get_path(
                webgpu_tile_composite_shader_handoff,
                ["orderedTileIndicesStoredInJson"],
            ),
            "renderPayloadStoredInJson": get_path(
                webgpu_tile_composite_shader_handoff, ["renderPayloadStoredInJson"]
            ),
            "shPolicy": get_path(
                webgpu_tile_composite_shader_handoff, ["shPolicy"], {}
            ),
            "shaderInputBuffers": get_path(
                webgpu_tile_composite_shader_handoff,
                ["shaderInputBuffers"],
                {},
            ),
            "shaderPacketLayout": get_path(
                webgpu_tile_composite_shader_handoff,
                ["shaderPacketLayout"],
                {},
            ),
            "recordCounts": get_path(
                webgpu_tile_composite_shader_handoff, ["recordCounts"], {}
            ),
            "payloadShapeValid": get_path(
                tile_composite_shader_handoff_validation, ["payloadShapeValid"]
            ),
            "tileOffsetsShapeValid": get_path(
                tile_composite_shader_handoff_validation, ["tileOffsetsShapeValid"]
            ),
            "tileIndicesShapeValid": get_path(
                tile_composite_shader_handoff_validation, ["tileIndicesShapeValid"]
            ),
            "orderedTileIndicesShapeValid": get_path(
                tile_composite_shader_handoff_validation,
                ["orderedTileIndicesShapeValid"],
            ),
            "depthOrderingValid": get_path(
                tile_composite_shader_handoff_validation, ["depthOrderingValid"]
            ),
            "orderingViolationCount": get_path(
                tile_composite_shader_handoff_validation,
                ["orderingViolationCount"],
            ),
            "renderHandoffReady": get_path(
                tile_composite_shader_handoff_validation,
                ["renderHandoffReady"],
            ),
            "tileCompositeHandoffReady": get_path(
                tile_composite_shader_handoff_validation,
                ["tileCompositeHandoffReady"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    tile_composite_shader_handoff_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(webgpu_tile_composite_shader_handoff, ["blockers"], [])
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_tile_composite_shader_handoff,
                ["nextBackendPrototypeStep"],
            ),
            "sampleTiles": compact_list(
                get_path(webgpu_tile_composite_shader_handoff, ["sampleTiles"], [])
            ),
        },
        "webgpuTileCompositeShaderDryRunComparison": {
            "status": get_path(
                webgpu_tile_composite_shader_dry_run_comparison, ["status"]
            ),
            "mode": get_path(
                webgpu_tile_composite_shader_dry_run_comparison, ["mode"]
            ),
            "source": get_path(
                webgpu_tile_composite_shader_dry_run_comparison, ["source"]
            ),
            "expectedSource": get_path(
                webgpu_tile_composite_shader_dry_run_comparison,
                ["expectedSource"],
            ),
            "actualSource": get_path(
                webgpu_tile_composite_shader_dry_run_comparison,
                ["actualSource"],
            ),
            "nonDisplayOnly": get_path(
                webgpu_tile_composite_shader_dry_run_comparison,
                ["nonDisplayOnly"],
            ),
            "implementedInWgsl": get_path(
                webgpu_tile_composite_shader_dry_run_comparison,
                ["implementedInWgsl"],
            ),
            "tileCompositeShaderComputed": get_path(
                webgpu_tile_composite_shader_dry_run_comparison,
                ["tileCompositeShaderComputed"],
            ),
            "tileCompositeImplemented": get_path(
                webgpu_tile_composite_shader_dry_run_comparison,
                ["tileCompositeImplemented"],
            ),
            "framebufferImplemented": get_path(
                webgpu_tile_composite_shader_dry_run_comparison,
                ["framebufferImplemented"],
            ),
            "displayConnectionAllowed": get_path(
                webgpu_tile_composite_shader_dry_run_comparison,
                ["displayConnectionAllowed"],
            ),
            "shPolicy": get_path(
                webgpu_tile_composite_shader_dry_run_comparison,
                ["shPolicy"],
                {},
            ),
            "shaderEvaluationPolicy": get_path(
                webgpu_tile_composite_shader_dry_run_comparison,
                ["shaderEvaluationPolicy"],
                {},
            ),
            "anyMismatch": get_path(
                webgpu_tile_composite_shader_dry_run_comparison,
                ["anyMismatch"],
            ),
            "mismatchClassification": get_path(
                webgpu_tile_composite_shader_dry_run_comparison,
                ["mismatchClassification"],
            ),
            "sampleCount": get_path(
                webgpu_tile_composite_shader_dry_run_comparison,
                ["sampleCount"],
            ),
            "sampleMismatchCount": get_path(
                webgpu_tile_composite_shader_dry_run_comparison,
                ["sampleMismatchCount"],
            ),
            "maxAbsPowerDelta": get_path(
                webgpu_tile_composite_shader_dry_run_comparison,
                ["maxAbsPowerDelta"],
            ),
            "maxAbsAlphaDelta": get_path(
                webgpu_tile_composite_shader_dry_run_comparison,
                ["maxAbsAlphaDelta"],
            ),
            "maxAbsColorDelta": get_path(
                webgpu_tile_composite_shader_dry_run_comparison,
                ["maxAbsColorDelta"],
            ),
            "renderPayloadShapeValid": get_path(
                tile_composite_shader_dry_run_validation,
                ["renderPayloadShapeValid"],
            ),
            "sampleBufferShapeValid": get_path(
                tile_composite_shader_dry_run_validation,
                ["sampleBufferShapeValid"],
            ),
            "outputShapeValid": get_path(
                tile_composite_shader_dry_run_validation,
                ["outputShapeValid"],
            ),
            "shaderSamplesValid": get_path(
                tile_composite_shader_dry_run_validation,
                ["shaderSamplesValid"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    tile_composite_shader_dry_run_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "firstMismatches": compact_list(
                get_path(
                    webgpu_tile_composite_shader_dry_run_comparison,
                    ["firstMismatches"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(
                    webgpu_tile_composite_shader_dry_run_comparison,
                    ["blockers"],
                    [],
                )
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_tile_composite_shader_dry_run_comparison,
                ["nextBackendPrototypeStep"],
            ),
            "sampleEvaluations": compact_list(
                get_path(
                    webgpu_tile_composite_shader_dry_run_comparison,
                    ["sampleEvaluations"],
                    [],
                )
            ),
        },
        "webgpuTileCompositeAccumulationDryRunComparison": {
            "status": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["status"],
            ),
            "mode": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["mode"],
            ),
            "source": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["source"],
            ),
            "expectedSource": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["expectedSource"],
            ),
            "actualSource": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["actualSource"],
            ),
            "nonDisplayOnly": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["nonDisplayOnly"],
            ),
            "implementedInWgsl": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["implementedInWgsl"],
            ),
            "tileCompositeAccumulationComputed": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["tileCompositeAccumulationComputed"],
            ),
            "tileCompositeImplemented": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["tileCompositeImplemented"],
            ),
            "framebufferImplemented": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["framebufferImplemented"],
            ),
            "displayConnectionAllowed": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["displayConnectionAllowed"],
            ),
            "orderedTileIndicesStoredInJson": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["orderedTileIndicesStoredInJson"],
            ),
            "renderPayloadStoredInJson": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["renderPayloadStoredInJson"],
            ),
            "shPolicy": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["shPolicy"],
                {},
            ),
            "accumulationPolicy": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["accumulationPolicy"],
                {},
            ),
            "anyMismatch": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["anyMismatch"],
            ),
            "mismatchClassification": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["mismatchClassification"],
            ),
            "sampleTileCount": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["sampleTileCount"],
            ),
            "accumulationMismatchCount": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["accumulationMismatchCount"],
            ),
            "maxAbsAccumColorDelta": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["maxAbsAccumColorDelta"],
            ),
            "maxAbsAccumAlphaDelta": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["maxAbsAccumAlphaDelta"],
            ),
            "maxAbsTransmittanceDelta": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["maxAbsTransmittanceDelta"],
            ),
            "renderPayloadShapeValid": get_path(
                tile_composite_accumulation_dry_run_validation,
                ["renderPayloadShapeValid"],
            ),
            "orderedTileIndicesShapeValid": get_path(
                tile_composite_accumulation_dry_run_validation,
                ["orderedTileIndicesShapeValid"],
            ),
            "sourceOrderedTileIndicesShapeValid": get_path(
                tile_composite_accumulation_dry_run_validation,
                ["sourceOrderedTileIndicesShapeValid"],
            ),
            "effectiveOrderedTileIndicesShapeValid": get_path(
                tile_composite_accumulation_dry_run_validation,
                ["effectiveOrderedTileIndicesShapeValid"],
            ),
            "sampleTileInputValid": get_path(
                tile_composite_accumulation_dry_run_validation,
                ["sampleTileInputValid"],
            ),
            "sampleBufferShapeValid": get_path(
                tile_composite_accumulation_dry_run_validation,
                ["sampleBufferShapeValid"],
            ),
            "outputShapeValid": get_path(
                tile_composite_accumulation_dry_run_validation,
                ["outputShapeValid"],
            ),
            "accumulationSamplesValid": get_path(
                tile_composite_accumulation_dry_run_validation,
                ["accumulationSamplesValid"],
            ),
            "seededNativeAccumulationInput": get_path(
                tile_composite_accumulation_dry_run_validation,
                ["seededNativeAccumulationInput"],
            ),
            "sampleSource": get_path(
                tile_composite_accumulation_dry_run_validation,
                ["sampleSource"],
            ),
            "seedSourceKind": get_path(
                tile_composite_accumulation_dry_run_validation,
                ["seedSourceKind"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    tile_composite_accumulation_dry_run_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "firstMismatches": compact_list(
                get_path(
                    webgpu_tile_composite_accumulation_dry_run_comparison,
                    ["firstMismatches"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(
                    webgpu_tile_composite_accumulation_dry_run_comparison,
                    ["blockers"],
                    [],
                )
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_tile_composite_accumulation_dry_run_comparison,
                ["nextBackendPrototypeStep"],
            ),
            "sampleTileAccumulations": compact_list(
                get_path(
                    webgpu_tile_composite_accumulation_dry_run_comparison,
                    ["sampleTileAccumulations"],
                    [],
                )
            ),
        },
        "webgpuFramebufferFreeTileOutputDryRunComparison": {
            "status": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["status"],
            ),
            "mode": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["mode"],
            ),
            "source": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["source"],
            ),
            "expectedSource": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["expectedSource"],
            ),
            "actualSource": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["actualSource"],
            ),
            "nonDisplayOnly": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["nonDisplayOnly"],
            ),
            "implementedInWgsl": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["implementedInWgsl"],
            ),
            "tileOutputPackingComputed": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["tileOutputPackingComputed"],
            ),
            "framebufferFreeOutputComputed": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["framebufferFreeOutputComputed"],
            ),
            "productionTileCompositeImplemented": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["productionTileCompositeImplemented"],
            ),
            "framebufferImplemented": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["framebufferImplemented"],
            ),
            "displayConnectionAllowed": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["displayConnectionAllowed"],
            ),
            "tileOutputStoredInJson": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["tileOutputStoredInJson"],
            ),
            "shPolicy": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["shPolicy"],
                {},
            ),
            "outputPackingPolicy": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["outputPackingPolicy"],
                {},
            ),
            "anyMismatch": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["anyMismatch"],
            ),
            "mismatchClassification": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["mismatchClassification"],
            ),
            "sampleTileCount": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["sampleTileCount"],
            ),
            "tileOutputMismatchCount": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["tileOutputMismatchCount"],
            ),
            "maxAbsResolvedColorDelta": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["maxAbsResolvedColorDelta"],
            ),
            "maxAbsCoverageAlphaDelta": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["maxAbsCoverageAlphaDelta"],
            ),
            "maxAbsFinalTransmittanceDelta": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["maxAbsFinalTransmittanceDelta"],
            ),
            "accumulationSummaryValid": get_path(
                framebuffer_free_tile_output_dry_run_validation,
                ["accumulationSummaryValid"],
            ),
            "sampleTileInputValid": get_path(
                framebuffer_free_tile_output_dry_run_validation,
                ["sampleTileInputValid"],
            ),
            "sampleBufferShapeValid": get_path(
                framebuffer_free_tile_output_dry_run_validation,
                ["sampleBufferShapeValid"],
            ),
            "outputBufferShapeValid": get_path(
                framebuffer_free_tile_output_dry_run_validation,
                ["outputBufferShapeValid"],
            ),
            "tileOutputSamplesValid": get_path(
                framebuffer_free_tile_output_dry_run_validation,
                ["tileOutputSamplesValid"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    framebuffer_free_tile_output_dry_run_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "firstMismatches": compact_list(
                get_path(
                    webgpu_framebuffer_free_tile_output_dry_run_comparison,
                    ["firstMismatches"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(
                    webgpu_framebuffer_free_tile_output_dry_run_comparison,
                    ["blockers"],
                    [],
                )
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_framebuffer_free_tile_output_dry_run_comparison,
                ["nextBackendPrototypeStep"],
            ),
            "sampleTileOutputs": compact_list(
                get_path(
                    webgpu_framebuffer_free_tile_output_dry_run_comparison,
                    ["sampleTileOutputs"],
                    [],
                )
            ),
        },
        "webgpuRenderTargetHandoffDryRunComparison": {
            "status": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["status"],
            ),
            "mode": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["mode"],
            ),
            "source": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["source"],
            ),
            "expectedSource": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["expectedSource"],
            ),
            "actualSource": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["actualSource"],
            ),
            "nonDisplayOnly": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["nonDisplayOnly"],
            ),
            "implementedInWgsl": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["implementedInWgsl"],
            ),
            "renderTargetSamplePackingComputed": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["renderTargetSamplePackingComputed"],
            ),
            "renderTargetHandoffReady": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["renderTargetHandoffReady"],
            ),
            "productionTileCompositeImplemented": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["productionTileCompositeImplemented"],
            ),
            "framebufferImplemented": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["framebufferImplemented"],
            ),
            "displayConnectionImplemented": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["displayConnectionImplemented"],
            ),
            "displayConnectionAllowed": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["displayConnectionAllowed"],
            ),
            "renderTargetSamplesStoredInJson": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["renderTargetSamplesStoredInJson"],
            ),
            "renderTargetContract": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["renderTargetContract"],
                {},
            ),
            "cameraProjectionContract": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["cameraProjectionContract"],
                {},
            ),
            "shPolicy": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["shPolicy"],
                {},
            ),
            "anyMismatch": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["anyMismatch"],
            ),
            "mismatchClassification": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["mismatchClassification"],
            ),
            "samplePixelCount": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["samplePixelCount"],
            ),
            "samplePixelMismatchCount": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["samplePixelMismatchCount"],
            ),
            "pixelCoordinateMismatchCount": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["pixelCoordinateMismatchCount"],
            ),
            "maxAbsResolvedColorDelta": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["maxAbsResolvedColorDelta"],
            ),
            "maxAbsCoverageAlphaDelta": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["maxAbsCoverageAlphaDelta"],
            ),
            "maxAbsFinalTransmittanceDelta": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["maxAbsFinalTransmittanceDelta"],
            ),
            "framebufferFreeOutputValid": get_path(
                render_target_handoff_dry_run_validation,
                ["framebufferFreeOutputValid"],
            ),
            "samplePixelInputValid": get_path(
                render_target_handoff_dry_run_validation,
                ["samplePixelInputValid"],
            ),
            "sampleBufferShapeValid": get_path(
                render_target_handoff_dry_run_validation,
                ["sampleBufferShapeValid"],
            ),
            "outputBufferShapeValid": get_path(
                render_target_handoff_dry_run_validation,
                ["outputBufferShapeValid"],
            ),
            "renderTargetSamplesValid": get_path(
                render_target_handoff_dry_run_validation,
                ["renderTargetSamplesValid"],
            ),
            "cameraProjectionContractCompatible": get_path(
                render_target_handoff_dry_run_validation,
                ["cameraProjectionContractCompatible"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    render_target_handoff_dry_run_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "firstMismatches": compact_list(
                get_path(
                    webgpu_render_target_handoff_dry_run_comparison,
                    ["firstMismatches"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(
                    webgpu_render_target_handoff_dry_run_comparison,
                    ["blockers"],
                    [],
                )
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_render_target_handoff_dry_run_comparison,
                ["nextBackendPrototypeStep"],
            ),
            "sampleRenderTargetPixels": compact_list(
                get_path(
                    webgpu_render_target_handoff_dry_run_comparison,
                    ["sampleRenderTargetPixels"],
                    [],
                )
            ),
        },
        "webgpuConstrainedDisplayAdapterDryRunComparison": {
            "status": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["status"],
            ),
            "mode": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["mode"],
            ),
            "source": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["source"],
            ),
            "expectedSource": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["expectedSource"],
            ),
            "actualSource": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["actualSource"],
            ),
            "nonDisplayOnly": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["nonDisplayOnly"],
            ),
            "implementedInWgsl": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["implementedInWgsl"],
            ),
            "constrainedDisplayExperiment": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["constrainedDisplayExperiment"],
            ),
            "displayAdapterDryRunComputed": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["displayAdapterDryRunComputed"],
            ),
            "renderTargetTextureWritten": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["renderTargetTextureWritten"],
            ),
            "textureReadbackCompared": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["textureReadbackCompared"],
            ),
            "framebufferAdapterImplemented": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["framebufferAdapterImplemented"],
            ),
            "framebufferImplemented": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["framebufferImplemented"],
            ),
            "canvasPresentationImplemented": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["canvasPresentationImplemented"],
            ),
            "displayConnectionImplemented": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["displayConnectionImplemented"],
            ),
            "displayConnectionAllowed": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["displayConnectionAllowed"],
            ),
            "textureStoredInJson": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["textureStoredInJson"],
            ),
            "displayAdapterContract": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["displayAdapterContract"],
                {},
            ),
            "shPolicy": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["shPolicy"],
                {},
            ),
            "anyMismatch": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["anyMismatch"],
            ),
            "mismatchClassification": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["mismatchClassification"],
            ),
            "samplePixelCount": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["samplePixelCount"],
            ),
            "duplicatePixelCount": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["duplicatePixelCount"],
            ),
            "duplicatePixelCountBeforeRemap": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["duplicatePixelCountBeforeRemap"],
            ),
            "duplicatePixelRemapCount": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["duplicatePixelRemapCount"],
            ),
            "duplicatePixelPolicy": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["duplicatePixelPolicy"],
            ),
            "texturePixelMismatchCount": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["texturePixelMismatchCount"],
            ),
            "maxAbsTextureColorDelta": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["maxAbsTextureColorDelta"],
            ),
            "maxAbsTextureAlphaDelta": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["maxAbsTextureAlphaDelta"],
            ),
            "renderTargetHandoffValid": get_path(
                constrained_display_adapter_dry_run_validation,
                ["renderTargetHandoffValid"],
            ),
            "samplePixelInputValid": get_path(
                constrained_display_adapter_dry_run_validation,
                ["samplePixelInputValid"],
            ),
            "textureExtentValid": get_path(
                constrained_display_adapter_dry_run_validation,
                ["textureExtentValid"],
            ),
            "textureReadbackShapeValid": get_path(
                constrained_display_adapter_dry_run_validation,
                ["textureReadbackShapeValid"],
            ),
            "duplicatePixelFree": get_path(
                constrained_display_adapter_dry_run_validation,
                ["duplicatePixelFree"],
            ),
            "validationDuplicatePixelCountBeforeRemap": get_path(
                constrained_display_adapter_dry_run_validation,
                ["duplicatePixelCountBeforeRemap"],
            ),
            "validationDuplicatePixelRemapCount": get_path(
                constrained_display_adapter_dry_run_validation,
                ["duplicatePixelRemapCount"],
            ),
            "validationDuplicatePixelPolicy": get_path(
                constrained_display_adapter_dry_run_validation,
                ["duplicatePixelPolicy"],
            ),
            "displayAdapterSamplesValid": get_path(
                constrained_display_adapter_dry_run_validation,
                ["displayAdapterSamplesValid"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    constrained_display_adapter_dry_run_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "firstMismatches": compact_list(
                get_path(
                    webgpu_constrained_display_adapter_dry_run_comparison,
                    ["firstMismatches"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(
                    webgpu_constrained_display_adapter_dry_run_comparison,
                    ["blockers"],
                    [],
                )
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_constrained_display_adapter_dry_run_comparison,
                ["nextBackendPrototypeStep"],
            ),
            "sampleTexturePixels": compact_list(
                get_path(
                    webgpu_constrained_display_adapter_dry_run_comparison,
                    ["sampleTexturePixels"],
                    [],
                )
            ),
        },
        "webgpuGuardedFirstDisplayExperiment": {
            "status": get_path(
                webgpu_guarded_first_display_experiment,
                ["status"],
            ),
            "mode": get_path(
                webgpu_guarded_first_display_experiment,
                ["mode"],
            ),
            "source": get_path(
                webgpu_guarded_first_display_experiment,
                ["source"],
            ),
            "expectedSource": get_path(
                webgpu_guarded_first_display_experiment,
                ["expectedSource"],
            ),
            "actualSource": get_path(
                webgpu_guarded_first_display_experiment,
                ["actualSource"],
            ),
            "guardedFirstDisplayExperiment": get_path(
                webgpu_guarded_first_display_experiment,
                ["guardedFirstDisplayExperiment"],
            ),
            "presentationGuardEnabled": get_path(
                webgpu_guarded_first_display_experiment,
                ["presentationGuardEnabled"],
            ),
            "displayExperimentOnly": get_path(
                webgpu_guarded_first_display_experiment,
                ["displayExperimentOnly"],
            ),
            "sourceTextureWritten": get_path(
                webgpu_guarded_first_display_experiment,
                ["sourceTextureWritten"],
            ),
            "presentationCandidateTextureWritten": get_path(
                webgpu_guarded_first_display_experiment,
                ["presentationCandidateTextureWritten"],
            ),
            "presentationCopyExecuted": get_path(
                webgpu_guarded_first_display_experiment,
                ["presentationCopyExecuted"],
            ),
            "presentationTextureReadbackCompared": get_path(
                webgpu_guarded_first_display_experiment,
                ["presentationTextureReadbackCompared"],
            ),
            "presentationCandidateReady": get_path(
                webgpu_guarded_first_display_experiment,
                ["presentationCandidateReady"],
            ),
            "framebufferAdapterImplemented": get_path(
                webgpu_guarded_first_display_experiment,
                ["framebufferAdapterImplemented"],
            ),
            "framebufferImplemented": get_path(
                webgpu_guarded_first_display_experiment,
                ["framebufferImplemented"],
            ),
            "canvasPresentationImplemented": get_path(
                webgpu_guarded_first_display_experiment,
                ["canvasPresentationImplemented"],
            ),
            "displayConnectionImplemented": get_path(
                webgpu_guarded_first_display_experiment,
                ["displayConnectionImplemented"],
            ),
            "displayConnectionAllowed": get_path(
                webgpu_guarded_first_display_experiment,
                ["displayConnectionAllowed"],
            ),
            "webgl2HybridRenderingAllowed": get_path(
                webgpu_guarded_first_display_experiment,
                ["webgl2HybridRenderingAllowed"],
            ),
            "presentationTextureStoredInJson": get_path(
                webgpu_guarded_first_display_experiment,
                ["presentationTextureStoredInJson"],
            ),
            "displayGuardContract": get_path(
                webgpu_guarded_first_display_experiment,
                ["displayGuardContract"],
                {},
            ),
            "cameraProjectionContract": get_path(
                webgpu_guarded_first_display_experiment,
                ["cameraProjectionContract"],
                {},
            ),
            "shPolicy": get_path(
                webgpu_guarded_first_display_experiment,
                ["shPolicy"],
                {},
            ),
            "anyMismatch": get_path(
                webgpu_guarded_first_display_experiment,
                ["anyMismatch"],
            ),
            "mismatchClassification": get_path(
                webgpu_guarded_first_display_experiment,
                ["mismatchClassification"],
            ),
            "samplePixelCount": get_path(
                webgpu_guarded_first_display_experiment,
                ["samplePixelCount"],
            ),
            "duplicatePixelCount": get_path(
                webgpu_guarded_first_display_experiment,
                ["duplicatePixelCount"],
            ),
            "presentationPixelMismatchCount": get_path(
                webgpu_guarded_first_display_experiment,
                ["presentationPixelMismatchCount"],
            ),
            "maxAbsPresentationColorDelta": get_path(
                webgpu_guarded_first_display_experiment,
                ["maxAbsPresentationColorDelta"],
            ),
            "maxAbsPresentationAlphaDelta": get_path(
                webgpu_guarded_first_display_experiment,
                ["maxAbsPresentationAlphaDelta"],
            ),
            "constrainedDisplayAdapterValid": get_path(
                guarded_first_display_validation,
                ["constrainedDisplayAdapterValid"],
            ),
            "samplePixelInputValid": get_path(
                guarded_first_display_validation,
                ["samplePixelInputValid"],
            ),
            "sourceTextureShapeValid": get_path(
                guarded_first_display_validation,
                ["sourceTextureShapeValid"],
            ),
            "presentationTextureShapeValid": get_path(
                guarded_first_display_validation,
                ["presentationTextureShapeValid"],
            ),
            "presentationCopyValid": get_path(
                guarded_first_display_validation,
                ["presentationCopyValid"],
            ),
            "presentationReadbackShapeValid": get_path(
                guarded_first_display_validation,
                ["presentationReadbackShapeValid"],
            ),
            "duplicatePixelFree": get_path(
                guarded_first_display_validation,
                ["duplicatePixelFree"],
            ),
            "guardFlagRequired": get_path(
                guarded_first_display_validation,
                ["guardFlagRequired"],
            ),
            "exclusiveBackendModeRequired": get_path(
                guarded_first_display_validation,
                ["exclusiveBackendModeRequired"],
            ),
            "cameraProjectionContractUnchanged": get_path(
                guarded_first_display_validation,
                ["cameraProjectionContractUnchanged"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    guarded_first_display_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "firstMismatches": compact_list(
                get_path(
                    webgpu_guarded_first_display_experiment,
                    ["firstMismatches"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(
                    webgpu_guarded_first_display_experiment,
                    ["blockers"],
                    [],
                )
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_guarded_first_display_experiment,
                ["nextBackendPrototypeStep"],
            ),
            "samplePresentationPixels": compact_list(
                get_path(
                    webgpu_guarded_first_display_experiment,
                    ["samplePresentationPixels"],
                    [],
                )
            ),
        },
        "webgpuCanvasPresentationAdapterDryRunComparison": {
            "status": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["status"],
            ),
            "mode": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["mode"],
            ),
            "source": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["source"],
            ),
            "expectedSource": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["expectedSource"],
            ),
            "actualSource": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["actualSource"],
            ),
            "boundedCanvasPresentationExperiment": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["boundedCanvasPresentationExperiment"],
            ),
            "detachedCanvasPresentationImplemented": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["detachedCanvasPresentationImplemented"],
            ),
            "detachedCanvasKind": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["detachedCanvasKind"],
            ),
            "viewerCanvasPresentationImplemented": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["viewerCanvasPresentationImplemented"],
            ),
            "productionDisplayConnectionImplemented": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["productionDisplayConnectionImplemented"],
            ),
            "displayConnectionAllowed": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["displayConnectionAllowed"],
            ),
            "contextGetCurrentTextureUsed": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["contextGetCurrentTextureUsed"],
            ),
            "currentTextureWritten": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["currentTextureWritten"],
            ),
            "currentTextureReadbackCompared": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["currentTextureReadbackCompared"],
            ),
            "canvasPresentationProbeSucceeded": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["canvasPresentationProbeSucceeded"],
            ),
            "viewerCanvasPresentationAllowed": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["viewerCanvasPresentationAllowed"],
            ),
            "framebufferImplemented": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["framebufferImplemented"],
            ),
            "canvasPresentationImplemented": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["canvasPresentationImplemented"],
            ),
            "webgl2HybridRenderingAllowed": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["webgl2HybridRenderingAllowed"],
            ),
            "canvasTextureStoredInJson": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["canvasTextureStoredInJson"],
            ),
            "canvasPresentationContract": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["canvasPresentationContract"],
                {},
            ),
            "viewerCanvasGuard": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["viewerCanvasGuard"],
                {},
            ),
            "cameraProjectionContract": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["cameraProjectionContract"],
                {},
            ),
            "shPolicy": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["shPolicy"],
                {},
            ),
            "anyMismatch": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["anyMismatch"],
            ),
            "mismatchClassification": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["mismatchClassification"],
            ),
            "samplePixelCount": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["samplePixelCount"],
            ),
            "duplicatePixelCount": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["duplicatePixelCount"],
            ),
            "canvasPixelMismatchCount": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["canvasPixelMismatchCount"],
            ),
            "maxAbsCanvasColorDelta": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["maxAbsCanvasColorDelta"],
            ),
            "maxAbsCanvasAlphaDelta": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["maxAbsCanvasAlphaDelta"],
            ),
            "guardedFirstDisplayValid": get_path(
                canvas_presentation_adapter_validation,
                ["guardedFirstDisplayValid"],
            ),
            "detachedCanvasAvailable": get_path(
                canvas_presentation_adapter_validation,
                ["detachedCanvasAvailable"],
            ),
            "webgpuCanvasContextAvailable": get_path(
                canvas_presentation_adapter_validation,
                ["webgpuCanvasContextAvailable"],
            ),
            "currentTextureAvailable": get_path(
                canvas_presentation_adapter_validation,
                ["currentTextureAvailable"],
            ),
            "currentTextureWriteValid": get_path(
                canvas_presentation_adapter_validation,
                ["currentTextureWriteValid"],
            ),
            "currentTextureReadbackShapeValid": get_path(
                canvas_presentation_adapter_validation,
                ["currentTextureReadbackShapeValid"],
            ),
            "viewerCanvasGuardActive": get_path(
                canvas_presentation_adapter_validation,
                ["viewerCanvasGuardActive"],
            ),
            "webgl2HybridRenderingPrevented": get_path(
                canvas_presentation_adapter_validation,
                ["webgl2HybridRenderingPrevented"],
            ),
            "cameraProjectionContractUnchanged": get_path(
                canvas_presentation_adapter_validation,
                ["cameraProjectionContractUnchanged"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    canvas_presentation_adapter_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "firstMismatches": compact_list(
                get_path(
                    webgpu_canvas_presentation_adapter_dry_run_comparison,
                    ["firstMismatches"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(
                    webgpu_canvas_presentation_adapter_dry_run_comparison,
                    ["blockers"],
                    [],
                )
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_canvas_presentation_adapter_dry_run_comparison,
                ["nextBackendPrototypeStep"],
            ),
            "sampleCanvasPixels": compact_list(
                get_path(
                    webgpu_canvas_presentation_adapter_dry_run_comparison,
                    ["sampleCanvasPixels"],
                    [],
                )
            ),
        },
        "webgpuExclusiveCanvasHandoffReadiness": {
            "status": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["status"],
            ),
            "mode": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["mode"],
            ),
            "source": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["source"],
            ),
            "expectedSource": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["expectedSource"],
            ),
            "actualSource": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["actualSource"],
            ),
            "normalBackendBoundaryImplemented": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["normalBackendBoundaryImplemented"],
            ),
            "exclusiveBackendModeRequested": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["exclusiveBackendModeRequested"],
            ),
            "exclusiveBackendModeReady": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["exclusiveBackendModeReady"],
            ),
            "viewerCanvasHandoffAllowed": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["viewerCanvasHandoffAllowed"],
            ),
            "viewerCanvasPresentationImplemented": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["viewerCanvasPresentationImplemented"],
            ),
            "productionDisplayConnectionImplemented": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["productionDisplayConnectionImplemented"],
            ),
            "displayConnectionAllowed": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["displayConnectionAllowed"],
            ),
            "webgl2HybridRenderingAllowed": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["webgl2HybridRenderingAllowed"],
            ),
            "webgpuNormalBackendCandidate": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["webgpuNormalBackendCandidate"],
            ),
            "requestedBackendMode": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["requestedBackendMode"],
            ),
            "supportedBackendModes": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["supportedBackendModes"],
                [],
            ),
            "backendRoleContract": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["backendRoleContract"],
                {},
            ),
            "viewerCanvasOwnershipContract": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["viewerCanvasOwnershipContract"],
                {},
            ),
            "canvasPresentationAdapterReuse": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["canvasPresentationAdapterReuse"],
                {},
            ),
            "cameraProjectionContract": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["cameraProjectionContract"],
                {},
            ),
            "shPolicy": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["shPolicy"],
                {},
            ),
            "anyMismatch": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["anyMismatch"],
            ),
            "mismatchClassification": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["mismatchClassification"],
            ),
            "canvasPresentationAdapterValid": get_path(
                exclusive_canvas_handoff_validation,
                ["canvasPresentationAdapterValid"],
            ),
            "detachedCanvasCurrentTextureValidated": get_path(
                exclusive_canvas_handoff_validation,
                ["detachedCanvasCurrentTextureValidated"],
            ),
            "viewerCanvasProvided": get_path(
                exclusive_canvas_handoff_validation,
                ["viewerCanvasProvided"],
            ),
            "viewerCanvasWebgl2Active": get_path(
                exclusive_canvas_handoff_validation,
                ["viewerCanvasWebgl2Active"],
            ),
            "viewerCanvasPresentationGuardEnabled": get_path(
                exclusive_canvas_handoff_validation,
                ["viewerCanvasPresentationGuardEnabled"],
            ),
            "webgl2HybridRenderingPrevented": get_path(
                exclusive_canvas_handoff_validation,
                ["webgl2HybridRenderingPrevented"],
            ),
            "cameraProjectionContractUnchanged": get_path(
                exclusive_canvas_handoff_validation,
                ["cameraProjectionContractUnchanged"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    exclusive_canvas_handoff_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(
                    webgpu_exclusive_canvas_handoff_readiness,
                    ["blockers"],
                    [],
                )
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_exclusive_canvas_handoff_readiness,
                ["nextBackendPrototypeStep"],
            ),
        },
        "webgpuExclusiveFrameLifecycleSwitch": {
            "status": get_path(
                webgpu_exclusive_frame_lifecycle_switch,
                ["status"],
            ),
            "mode": get_path(
                webgpu_exclusive_frame_lifecycle_switch,
                ["mode"],
            ),
            "source": get_path(
                webgpu_exclusive_frame_lifecycle_switch,
                ["source"],
            ),
            "exclusiveFrameLifecycleSwitchImplemented": get_path(
                webgpu_exclusive_frame_lifecycle_switch,
                ["exclusiveFrameLifecycleSwitchImplemented"],
            ),
            "viewerCanvasLifecycleSwitchRequested": get_path(
                webgpu_exclusive_frame_lifecycle_switch,
                ["viewerCanvasLifecycleSwitchRequested"],
            ),
            "viewerCanvasLifecycleSwitched": get_path(
                webgpu_exclusive_frame_lifecycle_switch,
                ["viewerCanvasLifecycleSwitched"],
            ),
            "viewerCanvasCurrentTexturePathReady": get_path(
                webgpu_exclusive_frame_lifecycle_switch,
                ["viewerCanvasCurrentTexturePathReady"],
            ),
            "viewerCanvasPresentationImplemented": get_path(
                webgpu_exclusive_frame_lifecycle_switch,
                ["viewerCanvasPresentationImplemented"],
            ),
            "displayConnectionAllowed": get_path(
                webgpu_exclusive_frame_lifecycle_switch,
                ["displayConnectionAllowed"],
            ),
            "webgl2HybridRenderingAllowed": get_path(
                webgpu_exclusive_frame_lifecycle_switch,
                ["webgl2HybridRenderingAllowed"],
            ),
            "requestedBackendMode": get_path(
                webgpu_exclusive_frame_lifecycle_switch,
                ["requestedBackendMode"],
            ),
            "frameLifecycleContract": get_path(
                webgpu_exclusive_frame_lifecycle_switch,
                ["frameLifecycleContract"],
                {},
            ),
            "viewerCanvasOwnershipContract": get_path(
                webgpu_exclusive_frame_lifecycle_switch,
                ["viewerCanvasOwnershipContract"],
                {},
            ),
            "cameraProjectionContract": get_path(
                webgpu_exclusive_frame_lifecycle_switch,
                ["cameraProjectionContract"],
                {},
            ),
            "shPolicy": get_path(
                webgpu_exclusive_frame_lifecycle_switch,
                ["shPolicy"],
                {},
            ),
            "canvasPresentationAdapterValid": get_path(
                exclusive_frame_lifecycle_validation,
                ["canvasPresentationAdapterValid"],
            ),
            "exclusiveBackendModeRequested": get_path(
                exclusive_frame_lifecycle_validation,
                ["exclusiveBackendModeRequested"],
            ),
            "viewerCanvasPresentationGuardEnabled": get_path(
                exclusive_frame_lifecycle_validation,
                ["viewerCanvasPresentationGuardEnabled"],
            ),
            "viewerCanvasWebgl2Active": get_path(
                exclusive_frame_lifecycle_validation,
                ["viewerCanvasWebgl2Active"],
            ),
            "webgl2FrameLifecycleSuppressed": get_path(
                exclusive_frame_lifecycle_validation,
                ["webgl2FrameLifecycleSuppressed"],
            ),
            "webgl2HybridRenderingPrevented": get_path(
                exclusive_frame_lifecycle_validation,
                ["webgl2HybridRenderingPrevented"],
            ),
            "cameraProjectionContractUnchanged": get_path(
                exclusive_frame_lifecycle_validation,
                ["cameraProjectionContractUnchanged"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    exclusive_frame_lifecycle_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(
                    webgpu_exclusive_frame_lifecycle_switch,
                    ["blockers"],
                    [],
                )
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_exclusive_frame_lifecycle_switch,
                ["nextBackendPrototypeStep"],
            ),
        },
        "webgpuViewerCanvasCurrentTexturePath": {
            "status": get_path(
                webgpu_viewer_canvas_current_texture_path,
                ["status"],
            ),
            "mode": get_path(
                webgpu_viewer_canvas_current_texture_path,
                ["mode"],
            ),
            "source": get_path(
                webgpu_viewer_canvas_current_texture_path,
                ["source"],
            ),
            "viewerCanvasCurrentTexturePathImplemented": get_path(
                webgpu_viewer_canvas_current_texture_path,
                ["viewerCanvasCurrentTexturePathImplemented"],
            ),
            "viewerCanvasCurrentTexturePathReady": get_path(
                webgpu_viewer_canvas_current_texture_path,
                ["viewerCanvasCurrentTexturePathReady"],
            ),
            "viewerCanvasContextConfigured": get_path(
                webgpu_viewer_canvas_current_texture_path,
                ["viewerCanvasContextConfigured"],
            ),
            "viewerCanvasCurrentTextureAcquired": get_path(
                webgpu_viewer_canvas_current_texture_path,
                ["viewerCanvasCurrentTextureAcquired"],
            ),
            "viewerCanvasPresentationImplemented": get_path(
                webgpu_viewer_canvas_current_texture_path,
                ["viewerCanvasPresentationImplemented"],
            ),
            "displayConnectionAllowed": get_path(
                webgpu_viewer_canvas_current_texture_path,
                ["displayConnectionAllowed"],
            ),
            "webgl2HybridRenderingAllowed": get_path(
                webgpu_viewer_canvas_current_texture_path,
                ["webgl2HybridRenderingAllowed"],
            ),
            "requestedBackendMode": get_path(
                webgpu_viewer_canvas_current_texture_path,
                ["requestedBackendMode"],
            ),
            "textureFormat": get_path(
                webgpu_viewer_canvas_current_texture_path,
                ["textureFormat"],
            ),
            "outputExtent": get_path(
                webgpu_viewer_canvas_current_texture_path,
                ["outputExtent"],
                {},
            ),
            "viewerCanvasOwnershipContract": get_path(
                webgpu_viewer_canvas_current_texture_path,
                ["viewerCanvasOwnershipContract"],
                {},
            ),
            "currentTextureContract": get_path(
                webgpu_viewer_canvas_current_texture_path,
                ["currentTextureContract"],
                {},
            ),
            "cameraProjectionContract": get_path(
                webgpu_viewer_canvas_current_texture_path,
                ["cameraProjectionContract"],
                {},
            ),
            "shPolicy": get_path(
                webgpu_viewer_canvas_current_texture_path,
                ["shPolicy"],
                {},
            ),
            "exclusiveBackendModeRequested": get_path(
                viewer_canvas_current_texture_validation,
                ["exclusiveBackendModeRequested"],
            ),
            "viewerCanvasPresentationGuardEnabled": get_path(
                viewer_canvas_current_texture_validation,
                ["viewerCanvasPresentationGuardEnabled"],
            ),
            "viewerCanvasWebgl2Active": get_path(
                viewer_canvas_current_texture_validation,
                ["viewerCanvasWebgl2Active"],
            ),
            "webgl2FrameLifecycleSuppressed": get_path(
                viewer_canvas_current_texture_validation,
                ["webgl2FrameLifecycleSuppressed"],
            ),
            "webgpuCanvasContextAvailable": get_path(
                viewer_canvas_current_texture_validation,
                ["webgpuCanvasContextAvailable"],
            ),
            "currentTextureAvailable": get_path(
                viewer_canvas_current_texture_validation,
                ["currentTextureAvailable"],
            ),
            "webgl2HybridRenderingPrevented": get_path(
                viewer_canvas_current_texture_validation,
                ["webgl2HybridRenderingPrevented"],
            ),
            "cameraProjectionContractUnchanged": get_path(
                viewer_canvas_current_texture_validation,
                ["cameraProjectionContractUnchanged"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    viewer_canvas_current_texture_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(
                    webgpu_viewer_canvas_current_texture_path,
                    ["blockers"],
                    [],
                )
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_viewer_canvas_current_texture_path,
                ["nextBackendPrototypeStep"],
            ),
        },
        "webgpuViewerCanvasBoundedFirstPresent": {
            "status": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["status"],
            ),
            "mode": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["mode"],
            ),
            "source": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["source"],
            ),
            "boundedViewerCanvasFirstPresentImplemented": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["boundedViewerCanvasFirstPresentImplemented"],
            ),
            "boundedViewerCanvasFirstPresentSucceeded": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["boundedViewerCanvasFirstPresentSucceeded"],
            ),
            "viewerCanvasPresentationImplemented": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["viewerCanvasPresentationImplemented"],
            ),
            "productionDisplayConnectionImplemented": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["productionDisplayConnectionImplemented"],
            ),
            "displayConnectionAllowed": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["displayConnectionAllowed"],
            ),
            "webgl2HybridRenderingAllowed": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["webgl2HybridRenderingAllowed"],
            ),
            "requestedBackendMode": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["requestedBackendMode"],
            ),
            "guardAllowed": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["guardAllowed"],
            ),
            "viewerCanvasContextConfiguredForPresent": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["viewerCanvasContextConfiguredForPresent"],
            ),
            "viewerCanvasCurrentTextureAcquiredForPresent": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["viewerCanvasCurrentTextureAcquiredForPresent"],
            ),
            "commandBufferSubmitted": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["commandBufferSubmitted"],
            ),
            "submittedWorkDone": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["submittedWorkDone"],
            ),
            "textureFormat": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["textureFormat"],
            ),
            "outputExtent": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["outputExtent"],
                {},
            ),
            "presentColorContract": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["presentColorContract"],
                {},
            ),
            "viewerCanvasOwnershipContract": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["viewerCanvasOwnershipContract"],
                {},
            ),
            "firstPresentContract": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["firstPresentContract"],
                {},
            ),
            "cameraProjectionContract": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["cameraProjectionContract"],
                {},
            ),
            "shPolicy": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["shPolicy"],
                {},
            ),
            "exclusiveBackendModeRequested": get_path(
                viewer_canvas_bounded_first_present_validation,
                ["exclusiveBackendModeRequested"],
            ),
            "viewerCanvasPresentationGuardEnabled": get_path(
                viewer_canvas_bounded_first_present_validation,
                ["viewerCanvasPresentationGuardEnabled"],
            ),
            "viewerCanvasWebgl2Active": get_path(
                viewer_canvas_bounded_first_present_validation,
                ["viewerCanvasWebgl2Active"],
            ),
            "webgl2FrameLifecycleSuppressed": get_path(
                viewer_canvas_bounded_first_present_validation,
                ["webgl2FrameLifecycleSuppressed"],
            ),
            "currentTexturePathReady": get_path(
                viewer_canvas_bounded_first_present_validation,
                ["currentTexturePathReady"],
            ),
            "webgl2HybridRenderingPrevented": get_path(
                viewer_canvas_bounded_first_present_validation,
                ["webgl2HybridRenderingPrevented"],
            ),
            "cameraProjectionContractUnchanged": get_path(
                viewer_canvas_bounded_first_present_validation,
                ["cameraProjectionContractUnchanged"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    viewer_canvas_bounded_first_present_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(
                    webgpu_viewer_canvas_bounded_first_present,
                    ["blockers"],
                    [],
                )
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_viewer_canvas_bounded_first_present,
                ["nextBackendPrototypeStep"],
            ),
        },
        "webgpuViewerCanvasNativeBoundedColorSamples": {
            "status": get_path(
                webgpu_viewer_canvas_native_bounded_color_samples,
                ["status"],
            ),
            "mode": get_path(
                webgpu_viewer_canvas_native_bounded_color_samples,
                ["mode"],
            ),
            "source": get_path(
                webgpu_viewer_canvas_native_bounded_color_samples,
                ["source"],
            ),
            "nativeBoundedSamplesBridgeImplemented": get_path(
                webgpu_viewer_canvas_native_bounded_color_samples,
                ["nativeBoundedSamplesBridgeImplemented"],
            ),
            "nativeBoundedSamplesReady": get_path(
                webgpu_viewer_canvas_native_bounded_color_samples,
                ["nativeBoundedSamplesReady"],
            ),
            "selectedNativeSourceKind": get_path(
                webgpu_viewer_canvas_native_bounded_color_samples,
                ["selectedNativeSourceKind"],
            ),
            "bridgeSeedSourceKind": get_path(
                webgpu_viewer_canvas_native_bounded_color_samples,
                ["bridgeSeedSourceKind"],
            ),
            "generatedFromRenderHandoff": get_path(
                webgpu_viewer_canvas_native_bounded_color_samples,
                ["generatedFromRenderHandoff"],
            ),
            "sampleTileOutputCount": get_path(
                webgpu_viewer_canvas_native_bounded_color_samples,
                ["sampleTileOutputCount"],
            ),
            "sampleRenderTargetPixelCount": get_path(
                webgpu_viewer_canvas_native_bounded_color_samples,
                ["sampleRenderTargetPixelCount"],
            ),
            "sampleTexturePixelCount": get_path(
                webgpu_viewer_canvas_native_bounded_color_samples,
                ["sampleTexturePixelCount"],
            ),
            "sourceAvailabilityBeforeBridge": get_path(
                webgpu_viewer_canvas_native_bounded_color_samples,
                ["sourceAvailabilityBeforeBridge"],
                {},
            ),
            "sourceAvailabilityAfterBridge": get_path(
                webgpu_viewer_canvas_native_bounded_color_samples,
                ["sourceAvailabilityAfterBridge"],
                {},
            ),
            "nativeSamplePolicy": get_path(
                webgpu_viewer_canvas_native_bounded_color_samples,
                ["nativeSamplePolicy"],
                {},
            ),
            "step40NativeSamplesReady": get_path(
                viewer_canvas_native_bounded_color_samples_validation,
                ["step40NativeSamplesReady"],
            ),
            "step39NativeSamplesReady": get_path(
                viewer_canvas_native_bounded_color_samples_validation,
                ["step39NativeSamplesReady"],
            ),
            "step38NativeSamplesReady": get_path(
                viewer_canvas_native_bounded_color_samples_validation,
                ["step38NativeSamplesReady"],
            ),
            "renderHandoffSeedAvailable": get_path(
                viewer_canvas_native_bounded_color_samples_validation,
                ["renderHandoffSeedAvailable"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    viewer_canvas_native_bounded_color_samples_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(
                    webgpu_viewer_canvas_native_bounded_color_samples,
                    ["blockers"],
                    [],
                )
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_viewer_canvas_native_bounded_color_samples,
                ["nextBackendPrototypeStep"],
            ),
        },
        "webgpuViewerCanvasBoundedColorSourceSelector": {
            "status": get_path(
                webgpu_viewer_canvas_bounded_color_source_selector,
                ["status"],
            ),
            "mode": get_path(
                webgpu_viewer_canvas_bounded_color_source_selector,
                ["mode"],
            ),
            "source": get_path(
                webgpu_viewer_canvas_bounded_color_source_selector,
                ["source"],
            ),
            "boundedColorSourceSelectorImplemented": get_path(
                webgpu_viewer_canvas_bounded_color_source_selector,
                ["boundedColorSourceSelectorImplemented"],
            ),
            "boundedColorSourceReady": get_path(
                webgpu_viewer_canvas_bounded_color_source_selector,
                ["boundedColorSourceReady"],
            ),
            "selectedSourceKind": get_path(
                webgpu_viewer_canvas_bounded_color_source_selector,
                ["selectedSourceKind"],
            ),
            "selectedColorSource": get_path(
                webgpu_viewer_canvas_bounded_color_source_selector,
                ["selectedColorSource"],
            ),
            "selectedSampleCount": get_path(
                webgpu_viewer_canvas_bounded_color_source_selector,
                ["selectedSampleCount"],
            ),
            "sourcePriority": compact_list(
                get_path(
                    webgpu_viewer_canvas_bounded_color_source_selector,
                    ["sourcePriority"],
                    [],
                )
            ),
            "sourceAvailability": get_path(
                webgpu_viewer_canvas_bounded_color_source_selector,
                ["sourceAvailability"],
                {},
            ),
            "nativeBridgePolicy": get_path(
                webgpu_viewer_canvas_bounded_color_source_selector,
                ["nativeBridgePolicy"],
                {},
            ),
            "fallbackPolicy": get_path(
                webgpu_viewer_canvas_bounded_color_source_selector,
                ["fallbackPolicy"],
                {},
            ),
            "step40SamplesAvailable": get_path(
                viewer_canvas_bounded_color_source_selector_validation,
                ["step40SamplesAvailable"],
            ),
            "step40NativeBridgeSamplesAvailable": get_path(
                viewer_canvas_bounded_color_source_selector_validation,
                ["step40NativeBridgeSamplesAvailable"],
            ),
            "step39SamplesAvailable": get_path(
                viewer_canvas_bounded_color_source_selector_validation,
                ["step39SamplesAvailable"],
            ),
            "step39NativeBridgeSamplesAvailable": get_path(
                viewer_canvas_bounded_color_source_selector_validation,
                ["step39NativeBridgeSamplesAvailable"],
            ),
            "step38SamplesAvailable": get_path(
                viewer_canvas_bounded_color_source_selector_validation,
                ["step38SamplesAvailable"],
            ),
            "step38NativeBridgeSamplesAvailable": get_path(
                viewer_canvas_bounded_color_source_selector_validation,
                ["step38NativeBridgeSamplesAvailable"],
            ),
            "renderHandoffFallbackAvailable": get_path(
                viewer_canvas_bounded_color_source_selector_validation,
                ["renderHandoffFallbackAvailable"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    viewer_canvas_bounded_color_source_selector_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(
                    webgpu_viewer_canvas_bounded_color_source_selector,
                    ["blockers"],
                    [],
                )
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_viewer_canvas_bounded_color_source_selector,
                ["nextBackendPrototypeStep"],
            ),
        },
        "webgpuViewerCanvasBoundedColorPresent": {
            "status": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["status"],
            ),
            "mode": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["mode"],
            ),
            "source": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["source"],
            ),
            "boundedViewerCanvasColorPresentImplemented": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["boundedViewerCanvasColorPresentImplemented"],
            ),
            "boundedViewerCanvasColorPresentSucceeded": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["boundedViewerCanvasColorPresentSucceeded"],
            ),
            "viewerCanvasPresentationImplemented": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["viewerCanvasPresentationImplemented"],
            ),
            "productionDisplayConnectionImplemented": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["productionDisplayConnectionImplemented"],
            ),
            "displayConnectionAllowed": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["displayConnectionAllowed"],
            ),
            "webgl2HybridRenderingAllowed": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["webgl2HybridRenderingAllowed"],
            ),
            "requestedBackendMode": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["requestedBackendMode"],
            ),
            "guardAllowed": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["guardAllowed"],
            ),
            "viewerCanvasContextConfiguredForColorPresent": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["viewerCanvasContextConfiguredForColorPresent"],
            ),
            "viewerCanvasCurrentTextureAcquiredForColorPresent": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["viewerCanvasCurrentTextureAcquiredForColorPresent"],
            ),
            "commandBufferSubmitted": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["commandBufferSubmitted"],
            ),
            "submittedWorkDone": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["submittedWorkDone"],
            ),
            "textureFormat": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["textureFormat"],
            ),
            "outputExtent": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["outputExtent"],
                {},
            ),
            "colorPresentSampleCount": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["colorPresentSampleCount"],
            ),
            "selectionMode": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["colorOutputContract.selectionMode"],
            ),
            "selectorSelectedSamplesUsed": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["colorOutputContract.selectorSelectedSamplesUsed"],
            ),
            "selectorSelectedRawSampleCount": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["colorOutputContract.selectorSelectedRawSampleCount"],
            ),
            "selectorPresentableSampleCount": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["colorOutputContract.selectorPresentableSampleCount"],
            ),
            "fallbackAllowed": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["colorOutputContract.fallbackAllowed"],
            ),
            "fallbackSuppressedBySelectorSamples": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["colorOutputContract.fallbackSuppressedBySelectorSamples"],
            ),
            "vertexCount": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["vertexCount"],
            ),
            "colorOutputContract": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["colorOutputContract"],
                {},
            ),
            "viewerCanvasOwnershipContract": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["viewerCanvasOwnershipContract"],
                {},
            ),
            "cameraProjectionContract": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["cameraProjectionContract"],
                {},
            ),
            "shPolicy": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["shPolicy"],
                {},
            ),
            "exclusiveBackendModeRequested": get_path(
                viewer_canvas_bounded_color_present_validation,
                ["exclusiveBackendModeRequested"],
            ),
            "viewerCanvasPresentationGuardEnabled": get_path(
                viewer_canvas_bounded_color_present_validation,
                ["viewerCanvasPresentationGuardEnabled"],
            ),
            "viewerCanvasWebgl2Active": get_path(
                viewer_canvas_bounded_color_present_validation,
                ["viewerCanvasWebgl2Active"],
            ),
            "webgl2FrameLifecycleSuppressed": get_path(
                viewer_canvas_bounded_color_present_validation,
                ["webgl2FrameLifecycleSuppressed"],
            ),
            "currentTexturePathReady": get_path(
                viewer_canvas_bounded_color_present_validation,
                ["currentTexturePathReady"],
            ),
            "boundedFirstPresentSucceeded": get_path(
                viewer_canvas_bounded_color_present_validation,
                ["boundedFirstPresentSucceeded"],
            ),
            "colorSamplesAvailable": get_path(
                viewer_canvas_bounded_color_present_validation,
                ["colorSamplesAvailable"],
            ),
            "webgl2HybridRenderingPrevented": get_path(
                viewer_canvas_bounded_color_present_validation,
                ["webgl2HybridRenderingPrevented"],
            ),
            "cameraProjectionContractUnchanged": get_path(
                viewer_canvas_bounded_color_present_validation,
                ["cameraProjectionContractUnchanged"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    viewer_canvas_bounded_color_present_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(
                    webgpu_viewer_canvas_bounded_color_present,
                    ["blockers"],
                    [],
                )
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_viewer_canvas_bounded_color_present,
                ["nextBackendPrototypeStep"],
            ),
        },
        "webgpuBackendFramePrototype": {
            "status": get_path(webgpu_backend_frame_prototype, ["status"]),
            "mode": get_path(webgpu_backend_frame_prototype, ["mode"]),
            "source": get_path(webgpu_backend_frame_prototype, ["source"]),
            "backendFramePrototypeImplemented": get_path(
                webgpu_backend_frame_prototype,
                ["backendFramePrototypeImplemented"],
            ),
            "backendFrameReady": get_path(
                webgpu_backend_frame_prototype,
                ["backendFrameReady"],
            ),
            "requestedBackendMode": get_path(
                webgpu_backend_frame_prototype,
                ["requestedBackendMode"],
            ),
            "productionDisplayConnectionImplemented": get_path(
                webgpu_backend_frame_prototype,
                ["productionDisplayConnectionImplemented"],
            ),
            "displayConnectionAllowed": get_path(
                webgpu_backend_frame_prototype,
                ["displayConnectionAllowed"],
            ),
            "webgl2HybridRenderingAllowed": get_path(
                webgpu_backend_frame_prototype,
                ["webgl2HybridRenderingAllowed"],
            ),
            "frameUnitContract": get_path(
                webgpu_backend_frame_prototype,
                ["frameUnitContract"],
                {},
            ),
            "inputSourceContract": get_path(
                webgpu_backend_frame_prototype,
                ["inputSourceContract"],
                {},
            ),
            "frameBudgetContract": get_path(
                webgpu_backend_frame_prototype,
                ["frameBudgetContract"],
                {},
            ),
            "continuationFrameContract": get_path(
                webgpu_backend_frame_prototype,
                ["continuationFrameContract"],
                {},
            ),
            "fallbackPolicy": get_path(
                webgpu_backend_frame_prototype,
                ["fallbackPolicy"],
                {},
            ),
            "currentTexturePathReady": get_path(
                webgpu_backend_frame_validation,
                ["currentTexturePathReady"],
            ),
            "boundedFirstPresentSucceeded": get_path(
                webgpu_backend_frame_validation,
                ["boundedFirstPresentSucceeded"],
            ),
            "nativeBoundedSamplesReady": get_path(
                webgpu_backend_frame_validation,
                ["nativeBoundedSamplesReady"],
            ),
            "selectorReady": get_path(
                webgpu_backend_frame_validation,
                ["selectorReady"],
            ),
            "presentSucceeded": get_path(
                webgpu_backend_frame_validation,
                ["presentSucceeded"],
            ),
            "selectorSelectedSamplesUsed": get_path(
                webgpu_backend_frame_validation,
                ["selectorSelectedSamplesUsed"],
            ),
            "fallbackSuppressedBySelectorSamples": get_path(
                webgpu_backend_frame_validation,
                ["fallbackSuppressedBySelectorSamples"],
            ),
            "backendFrameInputExpandedBeyondStep40": get_path(
                webgpu_backend_frame_validation,
                ["backendFrameInputExpandedBeyondStep40"],
            ),
            "frameBudgetReady": get_path(
                webgpu_backend_frame_validation,
                ["frameBudgetReady"],
            ),
            "continuationFrameReady": get_path(
                webgpu_backend_frame_validation,
                ["continuationFrameReady"],
            ),
            "presentableTrueNativeSourceKinds": compact_list(
                get_path(
                    webgpu_backend_frame_validation,
                    ["presentableTrueNativeSourceKinds"],
                    [],
                )
            ),
            "webgl2HybridRenderingPrevented": get_path(
                webgpu_backend_frame_validation,
                ["webgl2HybridRenderingPrevented"],
            ),
            "cameraProjectionContractUnchanged": get_path(
                webgpu_backend_frame_validation,
                ["cameraProjectionContractUnchanged"],
            ),
            "shEvaluationDeferred": get_path(
                webgpu_backend_frame_validation,
                ["shEvaluationDeferred"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    webgpu_backend_frame_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(webgpu_backend_frame_prototype, ["blockers"], [])
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_backend_frame_prototype,
                ["nextBackendPrototypeStep"],
            ),
        },
        "webgpuBackendFrameLifecyclePrototype": {
            "status": get_path(
                webgpu_backend_frame_lifecycle_prototype, ["status"]
            ),
            "mode": get_path(
                webgpu_backend_frame_lifecycle_prototype, ["mode"]
            ),
            "source": get_path(
                webgpu_backend_frame_lifecycle_prototype, ["source"]
            ),
            "contractVersion": get_path(
                webgpu_backend_frame_lifecycle_prototype,
                ["contractVersion"],
            ),
            "lifecyclePrototypeImplemented": get_path(
                webgpu_backend_frame_lifecycle_prototype,
                ["lifecyclePrototypeImplemented"],
            ),
            "lifecycleReady": get_path(
                webgpu_backend_frame_lifecycle_prototype,
                ["lifecycleReady"],
            ),
            "productionDisplayConnectionImplemented": get_path(
                webgpu_backend_frame_lifecycle_prototype,
                ["productionDisplayConnectionImplemented"],
            ),
            "displayConnectionAllowed": get_path(
                webgpu_backend_frame_lifecycle_prototype,
                ["displayConnectionAllowed"],
            ),
            "webgl2HybridRenderingAllowed": get_path(
                webgpu_backend_frame_lifecycle_prototype,
                ["webgl2HybridRenderingAllowed"],
            ),
            "requestedFrameCount": get_path(
                webgpu_backend_frame_lifecycle_prototype,
                ["requestedFrameCount"],
            ),
            "executedBackendFrameSubmissions": get_path(
                webgpu_backend_frame_lifecycle_prototype,
                ["executedBackendFrameSubmissions"],
            ),
            "simulatedContinuationFrameCount": get_path(
                webgpu_backend_frame_lifecycle_prototype,
                ["simulatedContinuationFrameCount"],
            ),
            "lifecycleContract": get_path(
                webgpu_backend_frame_lifecycle_prototype,
                ["lifecycleContract"],
                {},
            ),
            "frameSummaries": compact_list(
                get_path(
                    webgpu_backend_frame_lifecycle_prototype,
                    ["frameSummaries"],
                    [],
                )
            ),
            "selectedSourceKind": get_path(
                webgpu_backend_frame_lifecycle_prototype,
                ["selectedSourceKind"],
            ),
            "colorPresentSampleCount": get_path(
                webgpu_backend_frame_lifecycle_prototype,
                ["colorPresentSampleCount"],
            ),
            "sampleSources": compact_list(
                get_path(
                    webgpu_backend_frame_lifecycle_prototype,
                    ["sampleSources"],
                    [],
                )
            ),
            "fallbackPolicy": get_path(
                webgpu_backend_frame_lifecycle_prototype,
                ["fallbackPolicy"],
                {},
            ),
            "repeatedRunReady": get_path(
                webgpu_backend_frame_lifecycle_validation,
                ["repeatedRunReady"],
            ),
            "allFramesReady": get_path(
                webgpu_backend_frame_lifecycle_validation,
                ["allFramesReady"],
            ),
            "frameIndicesMonotonic": get_path(
                webgpu_backend_frame_lifecycle_validation,
                ["frameIndicesMonotonic"],
            ),
            "guardStableAcrossFrames": get_path(
                webgpu_backend_frame_lifecycle_validation,
                ["guardStableAcrossFrames"],
            ),
            "sampleCountsStable": get_path(
                webgpu_backend_frame_lifecycle_validation,
                ["sampleCountsStable"],
            ),
            "noExtraSubmitsAfterInitialFrame": get_path(
                webgpu_backend_frame_lifecycle_validation,
                ["noExtraSubmitsAfterInitialFrame"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    webgpu_backend_frame_lifecycle_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(
                    webgpu_backend_frame_lifecycle_prototype,
                    ["blockers"],
                    [],
                )
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_backend_frame_lifecycle_prototype,
                ["nextBackendPrototypeStep"],
            ),
        },
        "webgpuBackendFrameControlledRepeatedExecution": {
            "status": get_path(
                webgpu_backend_frame_controlled_repeated_execution,
                ["status"],
            ),
            "mode": get_path(
                webgpu_backend_frame_controlled_repeated_execution,
                ["mode"],
            ),
            "source": get_path(
                webgpu_backend_frame_controlled_repeated_execution,
                ["source"],
            ),
            "contractVersion": get_path(
                webgpu_backend_frame_controlled_repeated_execution,
                ["contractVersion"],
            ),
            "controlledRepeatedExecutionImplemented": get_path(
                webgpu_backend_frame_controlled_repeated_execution,
                ["controlledRepeatedExecutionImplemented"],
            ),
            "controlledRepeatedExecutionReady": get_path(
                webgpu_backend_frame_controlled_repeated_execution,
                ["controlledRepeatedExecutionReady"],
            ),
            "productionDisplayConnectionImplemented": get_path(
                webgpu_backend_frame_controlled_repeated_execution,
                ["productionDisplayConnectionImplemented"],
            ),
            "displayConnectionAllowed": get_path(
                webgpu_backend_frame_controlled_repeated_execution,
                ["displayConnectionAllowed"],
            ),
            "webgl2HybridRenderingAllowed": get_path(
                webgpu_backend_frame_controlled_repeated_execution,
                ["webgl2HybridRenderingAllowed"],
            ),
            "requestedFrameCount": get_path(
                webgpu_backend_frame_controlled_repeated_execution,
                ["requestedFrameCount"],
            ),
            "executedBackendFrameSubmissions": get_path(
                webgpu_backend_frame_controlled_repeated_execution,
                ["executedBackendFrameSubmissions"],
            ),
            "simulatedContinuationFrameCount": get_path(
                webgpu_backend_frame_controlled_repeated_execution,
                ["simulatedContinuationFrameCount"],
            ),
            "repeatedSubmitCount": get_path(
                webgpu_backend_frame_controlled_repeated_execution,
                ["repeatedSubmitCount"],
            ),
            "controlledExecutionContract": get_path(
                webgpu_backend_frame_controlled_repeated_execution,
                ["controlledExecutionContract"],
                {},
            ),
            "frameSummaries": compact_list(
                get_path(
                    webgpu_backend_frame_controlled_repeated_execution,
                    ["frameSummaries"],
                    [],
                )
            ),
            "selectedSourceKind": get_path(
                webgpu_backend_frame_controlled_repeated_execution,
                ["selectedSourceKind"],
            ),
            "colorPresentSampleCount": get_path(
                webgpu_backend_frame_controlled_repeated_execution,
                ["colorPresentSampleCount"],
            ),
            "sampleSources": compact_list(
                get_path(
                    webgpu_backend_frame_controlled_repeated_execution,
                    ["sampleSources"],
                    [],
                )
            ),
            "fallbackPolicy": get_path(
                webgpu_backend_frame_controlled_repeated_execution,
                ["fallbackPolicy"],
                {},
            ),
            "controlledRepeatedExecutionReadyValidation": get_path(
                webgpu_backend_frame_controlled_repeated_execution_validation,
                ["controlledRepeatedExecutionReady"],
            ),
            "allFramesReady": get_path(
                webgpu_backend_frame_controlled_repeated_execution_validation,
                ["allFramesReady"],
            ),
            "frameIndicesMonotonic": get_path(
                webgpu_backend_frame_controlled_repeated_execution_validation,
                ["frameIndicesMonotonic"],
            ),
            "allFramesSubmitted": get_path(
                webgpu_backend_frame_controlled_repeated_execution_validation,
                ["allFramesSubmitted"],
            ),
            "guardStableAcrossFrames": get_path(
                webgpu_backend_frame_controlled_repeated_execution_validation,
                ["guardStableAcrossFrames"],
            ),
            "sampleCountsStable": get_path(
                webgpu_backend_frame_controlled_repeated_execution_validation,
                ["sampleCountsStable"],
            ),
            "previousFrameChainValid": get_path(
                webgpu_backend_frame_controlled_repeated_execution_validation,
                ["previousFrameChainValid"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    webgpu_backend_frame_controlled_repeated_execution_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(
                    webgpu_backend_frame_controlled_repeated_execution,
                    ["blockers"],
                    [],
                )
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_backend_frame_controlled_repeated_execution,
                ["nextBackendPrototypeStep"],
            ),
        },
        "webgpuBackendViewerLoopAdapter": {
            "status": get_path(
                webgpu_backend_viewer_loop_adapter, ["status"]
            ),
            "mode": get_path(
                webgpu_backend_viewer_loop_adapter, ["mode"]
            ),
            "source": get_path(
                webgpu_backend_viewer_loop_adapter, ["source"]
            ),
            "contractVersion": get_path(
                webgpu_backend_viewer_loop_adapter,
                ["contractVersion"],
            ),
            "viewerLoopAdapterImplemented": get_path(
                webgpu_backend_viewer_loop_adapter,
                ["viewerLoopAdapterImplemented"],
            ),
            "viewerLoopAdapterReady": get_path(
                webgpu_backend_viewer_loop_adapter,
                ["viewerLoopAdapterReady"],
            ),
            "productionDisplayConnectionImplemented": get_path(
                webgpu_backend_viewer_loop_adapter,
                ["productionDisplayConnectionImplemented"],
            ),
            "displayConnectionAllowed": get_path(
                webgpu_backend_viewer_loop_adapter,
                ["displayConnectionAllowed"],
            ),
            "webgl2HybridRenderingAllowed": get_path(
                webgpu_backend_viewer_loop_adapter,
                ["webgl2HybridRenderingAllowed"],
            ),
            "requestedFrameCount": get_path(
                webgpu_backend_viewer_loop_adapter,
                ["requestedFrameCount"],
            ),
            "frameExecutionApiContract": get_path(
                webgpu_backend_viewer_loop_adapter,
                ["frameExecutionApiContract"],
                {},
            ),
            "frameResourceLifecycleContract": get_path(
                webgpu_backend_viewer_loop_adapter,
                ["frameResourceLifecycleContract"],
                {},
            ),
            "selectedSourceKind": get_path(
                webgpu_backend_viewer_loop_adapter,
                ["selectedSourceKind"],
            ),
            "selectionMode": get_path(
                webgpu_backend_viewer_loop_adapter,
                ["selectionMode"],
            ),
            "colorPresentSampleCount": get_path(
                webgpu_backend_viewer_loop_adapter,
                ["colorPresentSampleCount"],
            ),
            "executedBackendFrameSubmissions": get_path(
                webgpu_backend_viewer_loop_adapter,
                ["executedBackendFrameSubmissions"],
            ),
            "repeatedSubmitCount": get_path(
                webgpu_backend_viewer_loop_adapter,
                ["repeatedSubmitCount"],
            ),
            "fallbackPolicy": get_path(
                webgpu_backend_viewer_loop_adapter,
                ["fallbackPolicy"],
                {},
            ),
            "viewerLoopAdapterReadyValidation": get_path(
                webgpu_backend_viewer_loop_adapter_validation,
                ["viewerLoopAdapterReady"],
            ),
            "controlledReady": get_path(
                webgpu_backend_viewer_loop_adapter_validation,
                ["controlledReady"],
            ),
            "adapterCallable": get_path(
                webgpu_backend_viewer_loop_adapter_validation,
                ["adapterCallable"],
            ),
            "resourceLifecycleReady": get_path(
                webgpu_backend_viewer_loop_adapter_validation,
                ["resourceLifecycleReady"],
            ),
            "webgl2HybridRenderingPrevented": get_path(
                webgpu_backend_viewer_loop_adapter_validation,
                ["webgl2HybridRenderingPrevented"],
            ),
            "fallbackPolicyPreserved": get_path(
                webgpu_backend_viewer_loop_adapter_validation,
                ["fallbackPolicyPreserved"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    webgpu_backend_viewer_loop_adapter_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(webgpu_backend_viewer_loop_adapter, ["blockers"], [])
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_backend_viewer_loop_adapter,
                ["nextBackendPrototypeStep"],
            ),
        },
        "webgpuBackendViewerLifecycleIntegrationBoundary": {
            "status": get_path(
                webgpu_backend_viewer_lifecycle_integration, ["status"]
            ),
            "mode": get_path(
                webgpu_backend_viewer_lifecycle_integration, ["mode"]
            ),
            "source": get_path(
                webgpu_backend_viewer_lifecycle_integration, ["source"]
            ),
            "contractVersion": get_path(
                webgpu_backend_viewer_lifecycle_integration,
                ["contractVersion"],
            ),
            "integrationBoundaryImplemented": get_path(
                webgpu_backend_viewer_lifecycle_integration,
                ["integrationBoundaryImplemented"],
            ),
            "integrationBoundaryReady": get_path(
                webgpu_backend_viewer_lifecycle_integration,
                ["integrationBoundaryReady"],
            ),
            "productionDisplayConnectionImplemented": get_path(
                webgpu_backend_viewer_lifecycle_integration,
                ["productionDisplayConnectionImplemented"],
            ),
            "displayConnectionAllowed": get_path(
                webgpu_backend_viewer_lifecycle_integration,
                ["displayConnectionAllowed"],
            ),
            "webgl2HybridRenderingAllowed": get_path(
                webgpu_backend_viewer_lifecycle_integration,
                ["webgl2HybridRenderingAllowed"],
            ),
            "hookContract": get_path(
                webgpu_backend_viewer_lifecycle_integration,
                ["hookContract"],
                {},
            ),
            "cameraSnapshotContract": get_path(
                webgpu_backend_viewer_lifecycle_integration,
                ["cameraSnapshotContract"],
                {},
            ),
            "adapterInvocationContract": get_path(
                webgpu_backend_viewer_lifecycle_integration,
                ["adapterInvocationContract"],
                {},
            ),
            "viewerCanvasOwnershipContract": get_path(
                webgpu_backend_viewer_lifecycle_integration,
                ["viewerCanvasOwnershipContract"],
                {},
            ),
            "selectedSourceKind": get_path(
                webgpu_backend_viewer_lifecycle_integration,
                ["selectedSourceKind"],
            ),
            "colorPresentSampleCount": get_path(
                webgpu_backend_viewer_lifecycle_integration,
                ["colorPresentSampleCount"],
            ),
            "executedBackendFrameSubmissions": get_path(
                webgpu_backend_viewer_lifecycle_integration,
                ["executedBackendFrameSubmissions"],
            ),
            "repeatedSubmitCount": get_path(
                webgpu_backend_viewer_lifecycle_integration,
                ["repeatedSubmitCount"],
            ),
            "fallbackPolicy": get_path(
                webgpu_backend_viewer_lifecycle_integration,
                ["fallbackPolicy"],
                {},
            ),
            "integrationBoundaryReadyValidation": get_path(
                webgpu_backend_viewer_lifecycle_integration_validation,
                ["integrationBoundaryReady"],
            ),
            "hookAllowed": get_path(
                webgpu_backend_viewer_lifecycle_integration_validation,
                ["hookAllowed"],
            ),
            "webgl2FrameLifecycleSuppressed": get_path(
                webgpu_backend_viewer_lifecycle_integration_validation,
                ["webgl2FrameLifecycleSuppressed"],
            ),
            "webgl2HybridRenderingPrevented": get_path(
                webgpu_backend_viewer_lifecycle_integration_validation,
                ["webgl2HybridRenderingPrevented"],
            ),
            "cameraSnapshotProvided": get_path(
                webgpu_backend_viewer_lifecycle_integration_validation,
                ["cameraSnapshotProvided"],
            ),
            "adapterReady": get_path(
                webgpu_backend_viewer_lifecycle_integration_validation,
                ["adapterReady"],
            ),
            "fallbackPolicyPreserved": get_path(
                webgpu_backend_viewer_lifecycle_integration_validation,
                ["fallbackPolicyPreserved"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    webgpu_backend_viewer_lifecycle_integration_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(
                    webgpu_backend_viewer_lifecycle_integration,
                    ["blockers"],
                    [],
                )
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_backend_viewer_lifecycle_integration,
                ["nextBackendPrototypeStep"],
            ),
        },
        "webgpuBackendViewerLifecycleControlledExecution": {
            "status": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution, ["status"]
            ),
            "mode": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution, ["mode"]
            ),
            "source": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution, ["source"]
            ),
            "contractVersion": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution,
                ["contractVersion"],
            ),
            "controlledExecutionImplemented": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution,
                ["controlledExecutionImplemented"],
            ),
            "controlledExecutionReady": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution,
                ["controlledExecutionReady"],
            ),
            "productionDisplayConnectionImplemented": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution,
                ["productionDisplayConnectionImplemented"],
            ),
            "displayConnectionAllowed": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution,
                ["displayConnectionAllowed"],
            ),
            "webgl2HybridRenderingAllowed": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution,
                ["webgl2HybridRenderingAllowed"],
            ),
            "invocationContract": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution,
                ["invocationContract"],
                {},
            ),
            "selectedSourceKind": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution,
                ["selectedSourceKind"],
            ),
            "selectionMode": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution,
                ["selectionMode"],
            ),
            "colorPresentSampleCount": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution,
                ["colorPresentSampleCount"],
            ),
            "invocationCount": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution,
                ["invocationCount"],
            ),
            "submittedFrameCount": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution,
                ["submittedFrameCount"],
            ),
            "executedBackendFrameSubmissions": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution,
                ["executedBackendFrameSubmissions"],
            ),
            "repeatedSubmitCount": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution,
                ["repeatedSubmitCount"],
            ),
            "fallbackPolicy": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution,
                ["fallbackPolicy"],
                {},
            ),
            "controlledExecutionReadyValidation": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution_validation,
                ["controlledExecutionReady"],
            ),
            "invocationRequested": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution_validation,
                ["invocationRequested"],
            ),
            "integrationBoundaryReady": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution_validation,
                ["integrationBoundaryReady"],
            ),
            "adapterReady": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution_validation,
                ["adapterReady"],
            ),
            "webgl2FrameLifecycleSuppressed": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution_validation,
                ["webgl2FrameLifecycleSuppressed"],
            ),
            "webgl2HybridRenderingPrevented": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution_validation,
                ["webgl2HybridRenderingPrevented"],
            ),
            "cameraSnapshotProvided": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution_validation,
                ["cameraSnapshotProvided"],
            ),
            "fallbackPolicyPreserved": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution_validation,
                ["fallbackPolicyPreserved"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    webgpu_backend_viewer_lifecycle_controlled_execution_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(
                    webgpu_backend_viewer_lifecycle_controlled_execution,
                    ["blockers"],
                    [],
                )
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_backend_viewer_lifecycle_controlled_execution,
                ["nextBackendPrototypeStep"],
            ),
        },
        "webgpuBackendViewerFrameExecutor": {
            "status": get_path(webgpu_backend_viewer_frame_executor, ["status"]),
            "mode": get_path(webgpu_backend_viewer_frame_executor, ["mode"]),
            "source": get_path(webgpu_backend_viewer_frame_executor, ["source"]),
            "contractVersion": get_path(
                webgpu_backend_viewer_frame_executor,
                ["contractVersion"],
            ),
            "executorImplemented": get_path(
                webgpu_backend_viewer_frame_executor,
                ["executorImplemented"],
            ),
            "executorReady": get_path(
                webgpu_backend_viewer_frame_executor,
                ["executorReady"],
            ),
            "productionDisplayConnectionImplemented": get_path(
                webgpu_backend_viewer_frame_executor,
                ["productionDisplayConnectionImplemented"],
            ),
            "displayConnectionAllowed": get_path(
                webgpu_backend_viewer_frame_executor,
                ["displayConnectionAllowed"],
            ),
            "webgl2HybridRenderingAllowed": get_path(
                webgpu_backend_viewer_frame_executor,
                ["webgl2HybridRenderingAllowed"],
            ),
            "executorContract": get_path(
                webgpu_backend_viewer_frame_executor,
                ["executorContract"],
                {},
            ),
            "recorderObservation": get_path(
                webgpu_backend_viewer_frame_executor,
                ["recorderObservation"],
                {},
            ),
            "viewerFramePresentationPassContract": get_path(
                webgpu_backend_viewer_frame_executor,
                ["viewerFramePresentationPassContract"],
                {},
            ),
            "webgpuBackendViewerFramePresentationPass": get_path(
                webgpu_backend_viewer_frame_executor,
                ["webgpuBackendViewerFramePresentationPass"],
                {},
            ),
            "viewerFramePresentationPassReady": get_path(
                webgpu_backend_viewer_frame_executor,
                [
                    "viewerFramePresentationPassContract",
                    "viewerFramePresentationPassReady",
                ],
            ),
            "viewerOwnsCurrentTextureLifecycle": get_path(
                webgpu_backend_viewer_frame_executor,
                [
                    "viewerFramePresentationPassContract",
                    "viewerOwnsCurrentTextureLifecycle",
                ],
            ),
            "presentationPassCurrentTextureConnected": get_path(
                webgpu_backend_viewer_frame_executor,
                [
                    "viewerFramePresentationPassContract",
                    "currentTextureConnected",
                ],
            ),
            "presentationPassCurrentTextureReadbackMatchesAdapterOutput": get_path(
                webgpu_backend_viewer_frame_executor,
                [
                    "viewerFramePresentationPassContract",
                    "currentTextureReadbackMatchesAdapterOutput",
                ],
            ),
            "presentationPassDebugCaptureOwnsPresentationPass": get_path(
                webgpu_backend_viewer_frame_executor,
                [
                    "viewerFramePresentationPassContract",
                    "debugCaptureOwnsPresentationPass",
                ],
            ),
            "invocationCount": get_path(
                webgpu_backend_viewer_frame_executor,
                ["invocationCount"],
            ),
            "submittedFrameCount": get_path(
                webgpu_backend_viewer_frame_executor,
                ["submittedFrameCount"],
            ),
            "executedBackendFrameSubmissions": get_path(
                webgpu_backend_viewer_frame_executor,
                ["executedBackendFrameSubmissions"],
            ),
            "repeatedSubmitCount": get_path(
                webgpu_backend_viewer_frame_executor,
                ["repeatedSubmitCount"],
            ),
            "selectedSourceKind": get_path(
                webgpu_backend_viewer_frame_executor,
                ["selectedSourceKind"],
            ),
            "selectionMode": get_path(
                webgpu_backend_viewer_frame_executor,
                ["selectionMode"],
            ),
            "colorPresentSampleCount": get_path(
                webgpu_backend_viewer_frame_executor,
                ["colorPresentSampleCount"],
            ),
            "fallbackPolicy": get_path(
                webgpu_backend_viewer_frame_executor,
                ["fallbackPolicy"],
                {},
            ),
            "executorReadyValidation": get_path(
                webgpu_backend_viewer_frame_executor_validation,
                ["executorReady"],
            ),
            "guardAllowed": get_path(
                webgpu_backend_viewer_frame_executor_validation,
                ["guardAllowed"],
            ),
            "integrationBoundaryReady": get_path(
                webgpu_backend_viewer_frame_executor_validation,
                ["integrationBoundaryReady"],
            ),
            "backendFrameResultProvided": get_path(
                webgpu_backend_viewer_frame_executor_validation,
                ["backendFrameResultProvided"],
            ),
            "adapterReady": get_path(
                webgpu_backend_viewer_frame_executor_validation,
                ["adapterReady"],
            ),
            "controlledExecutionReady": get_path(
                webgpu_backend_viewer_frame_executor_validation,
                ["controlledExecutionReady"],
            ),
            "viewerFramePresentationPassReadyValidation": get_path(
                webgpu_backend_viewer_frame_executor_validation,
                ["viewerFramePresentationPassReady"],
            ),
            "webgl2HybridRenderingPrevented": get_path(
                webgpu_backend_viewer_frame_executor_validation,
                ["webgl2HybridRenderingPrevented"],
            ),
            "fallbackPolicyPreserved": get_path(
                webgpu_backend_viewer_frame_executor_validation,
                ["fallbackPolicyPreserved"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    webgpu_backend_viewer_frame_executor_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(webgpu_backend_viewer_frame_executor, ["blockers"], [])
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_backend_viewer_frame_executor,
                ["nextBackendPrototypeStep"],
            ),
        },
        "webgpuBackendViewerFramePresentationPass": {
            "status": get_path(
                webgpu_backend_viewer_frame_presentation_pass,
                ["status"],
            ),
            "contractVersion": get_path(
                webgpu_backend_viewer_frame_presentation_pass,
                ["contractVersion"],
            ),
            "presentationPassMode": get_path(
                webgpu_backend_viewer_frame_presentation_pass,
                ["presentationPassMode"],
            ),
            "viewerFramePresentationPassReady": get_path(
                webgpu_backend_viewer_frame_presentation_pass,
                ["viewerFramePresentationPassReady"],
            ),
            "calledFromViewerFrameLifecycle": get_path(
                webgpu_backend_viewer_frame_presentation_pass,
                ["calledFromViewerFrameLifecycle"],
            ),
            "calledFromExecutorChain": get_path(
                webgpu_backend_viewer_frame_presentation_pass,
                ["calledFromExecutorChain"],
            ),
            "viewerOwnsCurrentTextureLifecycle": get_path(
                webgpu_backend_viewer_frame_presentation_pass,
                ["viewerOwnsCurrentTextureLifecycle"],
            ),
            "currentTextureConnectionAttempted": get_path(
                webgpu_backend_viewer_frame_presentation_pass,
                ["currentTextureConnectionAttempted"],
            ),
            "currentTextureConnected": get_path(
                webgpu_backend_viewer_frame_presentation_pass,
                ["currentTextureConnected"],
            ),
            "currentTextureAcquired": get_path(
                webgpu_backend_viewer_frame_presentation_pass,
                ["currentTextureAcquired"],
            ),
            "currentTextureRenderPassSubmitted": get_path(
                webgpu_backend_viewer_frame_presentation_pass,
                ["currentTextureRenderPassSubmitted"],
            ),
            "currentTextureReadbackCompleted": get_path(
                webgpu_backend_viewer_frame_presentation_pass,
                ["currentTextureReadbackCompleted"],
            ),
            "currentTextureReadbackMatchesAdapterOutput": get_path(
                webgpu_backend_viewer_frame_presentation_pass,
                ["currentTextureReadbackMatchesAdapterOutput"],
            ),
            "debugCaptureOwnsPresentationPass": get_path(
                webgpu_backend_viewer_frame_presentation_pass,
                ["debugCaptureOwnsPresentationPass"],
            ),
            "productionSchedulerConnected": get_path(
                webgpu_backend_viewer_frame_presentation_pass,
                ["productionSchedulerConnected"],
            ),
            "webgl2HybridRenderingAllowed": get_path(
                webgpu_backend_viewer_frame_presentation_pass,
                ["webgl2HybridRenderingAllowed"],
            ),
            "fallbackSamplesMixed": get_path(
                webgpu_backend_viewer_frame_presentation_pass,
                ["fallbackSamplesMixed"],
            ),
            "reason": get_path(
                webgpu_backend_viewer_frame_presentation_pass,
                ["reason"],
            ),
        },
        "webgpuSchedulerFramePresentationBoundary": {
            "status": get_path(
                webgpu_scheduler_frame_presentation_boundary,
                ["status"],
            ),
            "contractVersion": get_path(
                webgpu_scheduler_frame_presentation_boundary,
                ["contractVersion"],
            ),
            "boundaryMode": get_path(
                webgpu_scheduler_frame_presentation_boundary,
                ["boundaryMode"],
            ),
            "schedulerFramePresentationBoundaryReady": get_path(
                webgpu_scheduler_frame_presentation_boundary,
                ["schedulerFramePresentationBoundaryReady"],
            ),
            "schedulerOwnsFrameRequest": get_path(
                webgpu_scheduler_frame_presentation_boundary,
                ["schedulerOwnsFrameRequest"],
            ),
            "schedulerOwnsPresentationBoundary": get_path(
                webgpu_scheduler_frame_presentation_boundary,
                ["schedulerOwnsPresentationBoundary"],
            ),
            "calledFromSchedulerFrameLoop": get_path(
                webgpu_scheduler_frame_presentation_boundary,
                ["calledFromSchedulerFrameLoop"],
            ),
            "frameRequestIssued": get_path(
                webgpu_scheduler_frame_presentation_boundary,
                ["frameRequestIssued"],
            ),
            "requestAnimationFrameCallbackEntered": get_path(
                webgpu_scheduler_frame_presentation_boundary,
                ["requestAnimationFrameCallbackEntered"],
            ),
            "renderFrameInvoked": get_path(
                webgpu_scheduler_frame_presentation_boundary,
                ["renderFrameInvoked"],
            ),
            "renderFrameCompleted": get_path(
                webgpu_scheduler_frame_presentation_boundary,
                ["renderFrameCompleted"],
            ),
            "viewerFramePresentationPassConsumed": get_path(
                webgpu_scheduler_frame_presentation_boundary,
                ["viewerFramePresentationPassConsumed"],
            ),
            "currentTextureConnected": get_path(
                webgpu_scheduler_frame_presentation_boundary,
                ["currentTextureConnected"],
            ),
            "currentTextureRenderPassSubmitted": get_path(
                webgpu_scheduler_frame_presentation_boundary,
                ["currentTextureRenderPassSubmitted"],
            ),
            "currentTextureReadbackCompleted": get_path(
                webgpu_scheduler_frame_presentation_boundary,
                ["currentTextureReadbackCompleted"],
            ),
            "currentTextureReadbackMatchesAdapterOutput": get_path(
                webgpu_scheduler_frame_presentation_boundary,
                ["currentTextureReadbackMatchesAdapterOutput"],
            ),
            "debugCaptureOwnsPresentationPass": get_path(
                webgpu_scheduler_frame_presentation_boundary,
                ["debugCaptureOwnsPresentationPass"],
            ),
            "productionSchedulerConnected": get_path(
                webgpu_scheduler_frame_presentation_boundary,
                ["productionSchedulerConnected"],
            ),
            "webgl2HybridRenderingAllowed": get_path(
                webgpu_scheduler_frame_presentation_boundary,
                ["webgl2HybridRenderingAllowed"],
            ),
            "fallbackSamplesMixed": get_path(
                webgpu_scheduler_frame_presentation_boundary,
                ["fallbackSamplesMixed"],
            ),
            "reason": get_path(
                webgpu_scheduler_frame_presentation_boundary,
                ["reason"],
            ),
        },
        "webgpuBackendRuntimeRunner": {
            "status": get_path(webgpu_backend_runtime_runner, ["status"]),
            "mode": get_path(webgpu_backend_runtime_runner, ["mode"]),
            "source": get_path(webgpu_backend_runtime_runner, ["source"]),
            "contractVersion": get_path(
                webgpu_backend_runtime_runner,
                ["contractVersion"],
            ),
            "runtimeRunnerImplemented": get_path(
                webgpu_backend_runtime_runner,
                ["runtimeRunnerImplemented"],
            ),
            "runtimeRunnerReady": get_path(
                webgpu_backend_runtime_runner,
                ["runtimeRunnerReady"],
            ),
            "productionDisplayConnectionImplemented": get_path(
                webgpu_backend_runtime_runner,
                ["productionDisplayConnectionImplemented"],
            ),
            "displayConnectionAllowed": get_path(
                webgpu_backend_runtime_runner,
                ["displayConnectionAllowed"],
            ),
            "webgl2HybridRenderingAllowed": get_path(
                webgpu_backend_runtime_runner,
                ["webgl2HybridRenderingAllowed"],
            ),
            "runnerContract": get_path(
                webgpu_backend_runtime_runner,
                ["runnerContract"],
                {},
            ),
            "recorderObservation": get_path(
                webgpu_backend_runtime_runner,
                ["recorderObservation"],
                {},
            ),
            "selectedSourceKind": get_path(
                webgpu_backend_runtime_runner,
                ["selectedSourceKind"],
            ),
            "selectionMode": get_path(
                webgpu_backend_runtime_runner,
                ["selectionMode"],
            ),
            "colorPresentSampleCount": get_path(
                webgpu_backend_runtime_runner,
                ["colorPresentSampleCount"],
            ),
            "presentedSampleCount": get_path(
                webgpu_backend_runtime_runner,
                ["presentedSampleCount"],
            ),
            "sampleSources": compact_list(
                get_path(webgpu_backend_runtime_runner, ["sampleSources"], [])
            ),
            "executedBackendFrameSubmissions": get_path(
                webgpu_backend_runtime_runner,
                ["executedBackendFrameSubmissions"],
            ),
            "repeatedSubmitCount": get_path(
                webgpu_backend_runtime_runner,
                ["repeatedSubmitCount"],
            ),
            "canonicalPresentSummary": get_path(
                webgpu_backend_runtime_runner,
                ["canonicalPresentSummary"],
                {},
            ),
            "resourceLifecycleSummary": get_path(
                webgpu_backend_runtime_runner,
                ["resourceLifecycleSummary"],
                {},
            ),
            "runtimeRunnerReadyValidation": get_path(
                webgpu_backend_runtime_runner_validation,
                ["runtimeRunnerReady"],
            ),
            "guardAllowed": get_path(
                webgpu_backend_runtime_runner_validation,
                ["guardAllowed"],
            ),
            "adapterReady": get_path(
                webgpu_backend_runtime_runner_validation,
                ["adapterReady"],
            ),
            "selectedTrueNativeSource": get_path(
                webgpu_backend_runtime_runner_validation,
                ["selectedTrueNativeSource"],
            ),
            "presentReady": get_path(
                webgpu_backend_runtime_runner_validation,
                ["presentReady"],
            ),
            "noFallbackMixing": get_path(
                webgpu_backend_runtime_runner_validation,
                ["noFallbackMixing"],
            ),
            "resourceLifecycleReady": get_path(
                webgpu_backend_runtime_runner_validation,
                ["resourceLifecycleReady"],
            ),
            "webgl2HybridRenderingPrevented": get_path(
                webgpu_backend_runtime_runner_validation,
                ["webgl2HybridRenderingPrevented"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    webgpu_backend_runtime_runner_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(webgpu_backend_runtime_runner, ["blockers"], [])
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_backend_runtime_runner,
                ["nextBackendPrototypeStep"],
            ),
        },
        "webgpuNormalBackendFrameImplementation": {
            "status": get_path(
                webgpu_normal_backend_frame_implementation,
                ["status"],
            ),
            "mode": get_path(
                webgpu_normal_backend_frame_implementation,
                ["mode"],
            ),
            "source": get_path(
                webgpu_normal_backend_frame_implementation,
                ["source"],
            ),
            "contractVersion": get_path(
                webgpu_normal_backend_frame_implementation,
                ["contractVersion"],
            ),
            "implementationKind": get_path(
                webgpu_normal_backend_frame_implementation,
                ["implementationKind"],
            ),
            "normalBackendImplementationImplemented": get_path(
                webgpu_normal_backend_frame_implementation,
                ["normalBackendImplementationImplemented"],
            ),
            "normalBackendImplementationReady": get_path(
                webgpu_normal_backend_frame_implementation,
                ["normalBackendImplementationReady"],
            ),
            "implementationContract": get_path(
                webgpu_normal_backend_frame_implementation,
                ["implementationContract"],
                {},
            ),
            "frameInputContract": get_path(
                webgpu_normal_backend_frame_implementation,
                ["frameInputContract"],
                {},
            ),
            "frameConstantsContract": get_path(
                webgpu_normal_backend_frame_implementation,
                ["frameConstantsContract"],
                {},
            ),
            "uniformResourcePreparationContract": get_path(
                webgpu_normal_backend_frame_implementation,
                ["uniformResourcePreparationContract"],
                {},
            ),
            "uniformResourceLifecycleContract": get_path(
                webgpu_normal_backend_frame_implementation,
                ["uniformResourceLifecycleContract"],
                {},
            ),
            "uniformResourceLifecycleSummary": get_path(
                webgpu_normal_backend_frame_implementation,
                ["uniformResourceLifecycleSummary"],
                {},
            ),
            "uniformShaderConsumptionContract": get_path(
                webgpu_normal_backend_frame_implementation,
                ["uniformShaderConsumptionContract"],
                {},
            ),
            "sampleResourceLifecycleContract": get_path(
                webgpu_normal_backend_frame_implementation,
                ["sampleResourceLifecycleContract"],
                {},
            ),
            "colorOutputSurfaceLifecycleContract": get_path(
                webgpu_normal_backend_frame_implementation,
                ["colorOutputSurfaceLifecycleContract"],
                {},
            ),
            "normalBackendOutputContract": get_path(
                webgpu_normal_backend_frame_implementation,
                ["normalBackendOutputContract"],
                {},
            ),
            "presentationHandoffContract": get_path(
                webgpu_normal_backend_frame_implementation,
                ["presentationHandoffContract"],
                {},
            ),
            "guardedPresentationAdapterContract": get_path(
                webgpu_normal_backend_frame_implementation,
                ["guardedPresentationAdapterContract"],
                {},
            ),
            "presentationBridgeContract": get_path(
                webgpu_normal_backend_frame_implementation,
                ["presentationBridgeContract"],
                {},
            ),
            "cameraAwareVisibleOutputContract": get_path(
                webgpu_normal_backend_frame_implementation,
                ["cameraAwareVisibleOutputContract"],
                {},
            ),
            "cameraAwareVisibleOutputReady": get_path(
                webgpu_normal_backend_frame_implementation,
                ["cameraAwareVisibleOutputReady"],
            ),
            "visibleOutputSampleCount": get_path(
                webgpu_normal_backend_frame_implementation,
                ["visibleOutputSampleCount"],
            ),
            "visibleOutputUsesCameraProjection": get_path(
                webgpu_normal_backend_frame_implementation,
                ["visibleOutputUsesCameraProjection"],
            ),
            "visibleOutputUsesSchedulerOwnedFramePath": get_path(
                webgpu_normal_backend_frame_implementation,
                ["visibleOutputUsesSchedulerOwnedFramePath"],
            ),
            "visibleOutputUsesCurrentTexturePath": get_path(
                webgpu_normal_backend_frame_implementation,
                ["visibleOutputUsesCurrentTexturePath"],
            ),
            "cameraAwareVisibleOutputSourceMode": get_path(
                webgpu_normal_backend_frame_implementation,
                ["cameraAwareVisibleOutputContract", "sourceMode"],
            ),
            "cameraAwareVisibleOutputPointRadiusPx": get_path(
                webgpu_normal_backend_frame_implementation,
                ["cameraAwareVisibleOutputContract", "outputPointRadiusPx"],
            ),
            "runtimeCameraAwareVisibleOutput": {
                "status": get_path(
                    webgpu_camera_aware_visible_output,
                    ["contract", "status"],
                ),
                "sampleCount": get_path(
                    webgpu_camera_aware_visible_output,
                    ["contract", "sampleCount"],
                ),
                "sourceMode": get_path(
                    webgpu_camera_aware_visible_output,
                    ["contract", "sourceMode"],
                ),
                "validRecordCount": get_path(
                    webgpu_camera_aware_visible_output,
                    ["contract", "validRecordCount"],
                ),
                "outputPointRadiusPx": get_path(
                    webgpu_camera_aware_visible_output,
                    ["contract", "outputPointRadiusPx"],
                ),
            },
            "normalBackendOutputValidation": get_path(
                webgpu_normal_backend_frame_implementation,
                [
                    "uniformShaderConsumptionContract",
                    "normalBackendOutputValidation",
                ],
                {},
            ),
            "presentOutputContract": get_path(
                webgpu_normal_backend_frame_implementation,
                ["presentOutputContract"],
                {},
            ),
            "selectedSourceKind": get_path(
                webgpu_normal_backend_frame_implementation,
                ["selectedSourceKind"],
            ),
            "selectionMode": get_path(
                webgpu_normal_backend_frame_implementation,
                ["selectionMode"],
            ),
            "colorPresentSampleCount": get_path(
                webgpu_normal_backend_frame_implementation,
                ["colorPresentSampleCount"],
            ),
            "presentedSampleCount": get_path(
                webgpu_normal_backend_frame_implementation,
                ["presentedSampleCount"],
            ),
            "sampleSources": compact_list(
                get_path(
                    webgpu_normal_backend_frame_implementation,
                    ["sampleSources"],
                    [],
                )
            ),
            "presentSummary": get_path(
                webgpu_normal_backend_frame_implementation,
                ["presentSummary"],
                {},
            ),
            "resourceSummary": get_path(
                webgpu_normal_backend_frame_implementation,
                ["resourceSummary"],
                {},
            ),
            "normalBackendImplementationReadyValidation": get_path(
                webgpu_normal_backend_frame_implementation_validation,
                ["normalBackendImplementationReady"],
            ),
            "guardAllowed": get_path(
                webgpu_normal_backend_frame_implementation_validation,
                ["guardAllowed"],
            ),
            "selectedTrueNativeSource": get_path(
                webgpu_normal_backend_frame_implementation_validation,
                ["selectedTrueNativeSource"],
            ),
            "presentReady": get_path(
                webgpu_normal_backend_frame_implementation_validation,
                ["presentReady"],
            ),
            "noFallbackMixing": get_path(
                webgpu_normal_backend_frame_implementation_validation,
                ["noFallbackMixing"],
            ),
            "resourceLifecycleReady": get_path(
                webgpu_normal_backend_frame_implementation_validation,
                ["resourceLifecycleReady"],
            ),
            "sampleResourceConsumptionReady": get_path(
                webgpu_normal_backend_frame_implementation_validation,
                ["sampleResourceConsumptionReady"],
            ),
            "colorOutputSurfaceReady": get_path(
                webgpu_normal_backend_frame_implementation_validation,
                ["colorOutputSurfaceReady"],
            ),
            "normalBackendOutputReady": get_path(
                webgpu_normal_backend_frame_implementation_validation,
                ["normalBackendOutputReady"],
            ),
            "presentationHandoffReady": get_path(
                webgpu_normal_backend_frame_implementation_validation,
                ["presentationHandoffReady"],
            ),
            "guardedPresentationAdapterReady": get_path(
                webgpu_normal_backend_frame_implementation_validation,
                ["guardedPresentationAdapterReady"],
            ),
            "viewerPresentationBridgeReady": get_path(
                webgpu_normal_backend_frame_implementation_validation,
                ["viewerPresentationBridgeReady"],
            ),
            "currentTextureConnectionReady": get_path(
                webgpu_normal_backend_frame_implementation_validation,
                ["currentTextureConnectionReady"],
            ),
            "cameraAwareVisibleOutputReadyValidation": get_path(
                webgpu_normal_backend_frame_implementation_validation,
                ["cameraAwareVisibleOutputReady"],
            ),
            "renderTargetBridgeReady": get_path(
                webgpu_normal_backend_frame_implementation,
                ["presentationBridgeContract", "renderTargetBridgeReady"],
            ),
            "currentTextureConnectionAttempted": get_path(
                webgpu_normal_backend_frame_implementation,
                [
                    "presentationBridgeContract",
                    "currentTextureConnectionAttempted",
                ],
            ),
            "currentTextureConnected": get_path(
                webgpu_normal_backend_frame_implementation,
                ["presentationBridgeContract", "currentTextureConnected"],
            ),
            "currentTextureConfigured": get_path(
                webgpu_normal_backend_frame_implementation,
                ["presentationBridgeContract", "currentTextureConfigured"],
            ),
            "currentTextureReadbackMatchesAdapterOutput": get_path(
                webgpu_normal_backend_frame_implementation,
                [
                    "presentationBridgeContract",
                    "currentTextureReadbackMatchesAdapterOutput",
                ],
            ),
            "webgl2HybridRenderingPrevented": get_path(
                webgpu_normal_backend_frame_implementation_validation,
                ["webgl2HybridRenderingPrevented"],
            ),
            "firstValidationFailures": compact_list(
                get_path(
                    webgpu_normal_backend_frame_implementation_validation,
                    ["firstValidationFailures"],
                    [],
                )
            ),
            "blockers": compact_list(
                get_path(
                    webgpu_normal_backend_frame_implementation,
                    ["blockers"],
                    [],
                )
            ),
            "nextBackendPrototypeStep": get_path(
                webgpu_normal_backend_frame_implementation,
                ["nextBackendPrototypeStep"],
            ),
        },
        "inputContract": get_path(summary, ["inputContract"], {}),
        "bufferContract": get_path(summary, ["bufferContract"], {}),
        "inputBufferModes": get_path(summary, ["inputBufferModes"], {}),
        "candidateCount": get_path(summary, ["candidateCount"]),
        "recordCount": get_path(summary, ["recordCount"]),
        "validRecordCount": get_path(summary, ["validRecordCount"]),
        "mismatchClassification": get_path(summary, ["mismatchClassification"]),
        "anyMismatch": get_path(summary, ["anyMismatch"]),
        "recordAnyMismatch": get_path(record_comparison, ["anyMismatch"]),
        "fieldMismatchCount": get_path(record_comparison, ["fieldMismatchCount"]),
        "firstMismatches": get_path(record_comparison, ["firstMismatches"], []),
        "adapterDeviceMs": get_path(timing, ["adapterDeviceMs"]),
        "bufferUploadMs": get_path(timing, ["bufferUploadMs"]),
        "computeDispatchMs": get_path(timing, ["computeDispatchMs"]),
        "readbackMs": get_path(timing, ["readbackMs"]),
        "compareMs": get_path(timing, ["compareMs"]),
        "totalMs": get_path(timing, ["totalMs"]),
        "projectionParamMode": get_path(webgpu, ["projectionParamMode"]),
        "statePositionUploadMode": get_path(webgpu, ["statePositionUploadMode"]),
    }


def extract_association(data: Dict[str, Any]) -> Dict[str, Any]:
    summary = get_path(data, ["summary"], data)

    return {
        "mismatchCount": get_path(summary, ["mismatchCount"]),
        "associationMismatchLikely": get_path(
            summary,
            ["associationMismatchLikely"],
        ),
        "associationPackedCount": get_path(
            summary,
            ["associationPackedCount", "packedCount"],
        ),
        "tilePayloadCount": get_path(
            data,
            ["tilePayloadCount", "summary.tilePayloadCount"],
        ),
    }


def extract_render_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "visibleCount": get_path(data, ["visibleCount", "summary.visibleCount"]),
        "tilePayloadCount": get_path(
            data,
            [
                "tilePayloadCount",
                "summary.tilePayloadCount",
                "tilePayloadCount.value",
            ],
        ),
        "tilePayloadCountEqual": get_path(
            data,
            ["tilePayloadCount.equal", "summary.tilePayloadCount.equal"],
        ),
        "usedCuda4DStateHelper": get_path(
            data,
            ["usedCuda4DStateHelper", "summary.usedCuda4DStateHelper"],
        ),
        "stateConvention": get_path(
            data,
            ["stateConvention", "summary.stateConvention"],
        ),
        "stateHelperVersion": get_path(
            data,
            ["stateHelperVersion", "summary.stateHelperVersion"],
        ),
        "framebufferDeltaAbsMax": get_path(
            data,
            [
                "framebufferDeltaAbsMax",
                "summary.framebufferDeltaAbsMax",
                "framebufferReplay.framebufferDeltaAbsMax",
            ],
        ),
    }


def collect_existing_json(base_dir: Path, prefix: str) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for suffix in KNOWN_SUFFIXES:
        path = base_dir / f"{prefix}_{suffix}.json"
        if path.exists():
            found[suffix] = path
    return found


def summarize_step(base_dir: Path, prefix: str) -> Dict[str, Any]:
    files = collect_existing_json(base_dir, prefix)

    result: Dict[str, Any] = {
        "baseDir": str(base_dir),
        "prefix": prefix,
        "files": {suffix: str(path) for suffix, path in files.items()},
        "missingKnownSuffixes": [
            suffix for suffix in KNOWN_SUFFIXES if suffix not in files
        ],
        "candidate": None,
        "coverage": None,
        "runtime": None,
        "limitedDraw": None,
        "visibleCompare": None,
        "dryRunVisibleCompare": None,
        "screenCoarseSweep": None,
        "promotionValidation": None,
        "step111Timing": None,
        "step111PipelineParityGapClosure": None,
        "step112CameraProjectionOrientationParity": None,
        "gpuVisibleRecordDryRun": None,
        "gpuRawVisibleRecordDryRun": None,
        "webgpuVisibleRecordDryRun": None,
        "outputCaptureDiagnostic": None,
        "fixedConditionVisualComparison": None,
        "association": None,
        "renderSummary": None,
        "loadErrors": {},
    }

    loaded: Dict[str, Dict[str, Any]] = {}
    for suffix, path in files.items():
        data, error = load_json_if_exists(path)
        if error:
            result["loadErrors"][suffix] = error
        elif data is not None:
            loaded[suffix] = data

    if "gpu_candidate_screen_coarse_compare" in loaded:
        result["candidate"] = extract_candidate_compare(
            loaded["gpu_candidate_screen_coarse_compare"]
        )
    elif "gpu_candidate_source_compare" in loaded:
        result["candidate"] = extract_candidate_compare(
            loaded["gpu_candidate_source_compare"]
        )

    if "gpu_candidate_coverage" in loaded:
        result["coverage"] = extract_coverage(loaded["gpu_candidate_coverage"])

    if "gpu_candidate_runtime_summary" in loaded:
        result["runtime"] = extract_runtime(loaded["gpu_candidate_runtime_summary"])

    if "limited_draw_summary" in loaded:
        result["limitedDraw"] = extract_runtime(loaded["limited_draw_summary"])
        result["promotionValidation"] = extract_promotion_validation(
            loaded["limited_draw_summary"]
        )
        result["step111Timing"] = extract_step111_timing(
            loaded["limited_draw_summary"]
        )
        result["gpuVisibleRecordDryRun"] = extract_gpu_visible_record_dryrun(
            loaded["limited_draw_summary"]
        )
        result["gpuRawVisibleRecordDryRun"] = extract_gpu_raw_visible_record_dryrun(
            loaded["limited_draw_summary"]
        )
        if result.get("coverage") is None:
            result["coverage"] = extract_coverage(loaded["limited_draw_summary"])

    if "visible_compare" in loaded:
        result["visibleCompare"] = extract_visible_compare(loaded["visible_compare"])

    if "gpu_candidate_screen_coarse_dryrun_visible_compare" in loaded:
        result["dryRunVisibleCompare"] = extract_dryrun_visible_compare(
            loaded["gpu_candidate_screen_coarse_dryrun_visible_compare"]
        )

    if "gpu_candidate_screen_coarse_sweep_summary" in loaded:
        result["screenCoarseSweep"] = extract_screen_coarse_sweep(
            loaded["gpu_candidate_screen_coarse_sweep_summary"]
        )

    if "gpu_visible_record_dryrun_compare" in loaded:
        result["gpuVisibleRecordDryRun"] = extract_gpu_visible_record_dryrun(
            loaded["gpu_visible_record_dryrun_compare"]
        )

    if "gpu_raw_visible_record_dryrun_compare" in loaded:
        result["gpuRawVisibleRecordDryRun"] = extract_gpu_raw_visible_record_dryrun(
            loaded["gpu_raw_visible_record_dryrun_compare"]
        )

    webgpu_visible_capture_status = None
    if "webgpu_visible_record_dryrun_capture_status" in loaded:
        webgpu_visible_capture_status = extract_capture_status(
            loaded["webgpu_visible_record_dryrun_capture_status"]
        )

    if "webgpu_visible_record_dryrun_compare" in loaded:
        result["webgpuVisibleRecordDryRun"] = extract_webgpu_visible_record_dryrun(
            loaded["webgpu_visible_record_dryrun_compare"]
        )
        if webgpu_visible_capture_status is not None:
            result["webgpuVisibleRecordDryRun"].update(webgpu_visible_capture_status)
    elif webgpu_visible_capture_status is not None:
        result["webgpuVisibleRecordDryRun"] = {
            "status": webgpu_visible_capture_status.get("captureStatus"),
            "reason": webgpu_visible_capture_status.get("captureReason"),
            "computeMode": None,
            "scaffoldMode": None,
            "phaseStep": None,
            **webgpu_visible_capture_status,
        }

    fixed_condition_visual_comparison = None
    if "fixed_condition_visual_comparison" in loaded:
        fixed_condition_visual_comparison = extract_fixed_condition_visual_comparison(
            loaded["fixed_condition_visual_comparison"]
        )
        result["fixedConditionVisualComparison"] = fixed_condition_visual_comparison
    step110_fixed_condition_visual_comparison = None
    step110_prefix = related_step_prefix(prefix, 110)
    if step110_prefix:
        step110_data, _ = load_json_if_exists(
            base_dir / f"{step110_prefix}_fixed_condition_visual_comparison.json"
        )
        if isinstance(step110_data, dict):
            step110_fixed_condition_visual_comparison = (
                extract_fixed_condition_visual_comparison(step110_data)
            )

    png_diagnostic = extract_png_capture_diagnostic(base_dir / f"{prefix}_canvas.png")
    result["outputCaptureDiagnostic"] = png_diagnostic
    if isinstance(result.get("webgpuVisibleRecordDryRun"), dict):
        output_diagnostic_source = loaded.get(
            "webgpu_visible_record_dryrun_compare",
            result["webgpuVisibleRecordDryRun"],
        )
        output_diagnostic = build_output_capture_consistency_diagnostic(
            output_diagnostic_source,
            png_diagnostic,
            loaded.get("png_capture_status"),
        )
        if output_diagnostic.get("captureTarget") is None:
            output_diagnostic["captureTarget"] = result["webgpuVisibleRecordDryRun"].get(
                "captureTarget"
            )
        output_diagnostic.update(
            build_step108_output_regression_diagnostic(
                base_dir,
                prefix,
                output_diagnostic,
            )
        )
        result["outputCaptureDiagnostic"] = output_diagnostic
        result["webgpuVisibleRecordDryRun"]["outputCaptureDiagnostic"] = (
            output_diagnostic
        )
        for step_key, step_name in [
            ("step107FixedReferenceCameraContract", "step107"),
            ("step108CudaReferenceCameraEvidence", "step108"),
            ("step109FixedReferenceCameraActivation", "step109"),
        ]:
            apply_output_capture_diagnostic_to_step_summary(
                result["webgpuVisibleRecordDryRun"].get(step_key),
                output_diagnostic,
                step_name=step_name,
            )
        result["webgpuVisibleRecordDryRun"][
            "step110FixedConditionVisualComparison"
        ] = build_step110_fixed_condition_visual_comparison_summary(
            {
                **output_diagnostic_source,
                "outputCaptureDiagnostic": output_diagnostic,
            },
            fixed_condition_visual_comparison,
        )
        result["webgpuVisibleRecordDryRun"][
            "step111PipelineParityGapClosure"
        ] = build_step111_pipeline_parity_gap_closure_summary(
            {
                **output_diagnostic_source,
                "outputCaptureDiagnostic": output_diagnostic,
            },
            fixed_condition_visual_comparison,
            step110_fixed_condition_visual_comparison,
        )
        result["step111PipelineParityGapClosure"] = result[
            "webgpuVisibleRecordDryRun"
        ]["step111PipelineParityGapClosure"]
        result["webgpuVisibleRecordDryRun"][
            "step112CameraProjectionOrientationParity"
        ] = build_step112_camera_projection_orientation_parity_summary(
            {
                **output_diagnostic_source,
                "outputCaptureDiagnostic": output_diagnostic,
            },
            fixed_condition_visual_comparison,
        )
        result["step112CameraProjectionOrientationParity"] = result[
            "webgpuVisibleRecordDryRun"
        ]["step112CameraProjectionOrientationParity"]
    elif fixed_condition_visual_comparison is not None:
        result["fixedConditionVisualComparison"] = fixed_condition_visual_comparison

    if "association" in loaded:
        result["association"] = extract_association(loaded["association"])

    if "summary" in loaded:
        result["renderSummary"] = extract_render_summary(loaded["summary"])
    elif "live_same_state" in loaded:
        result["renderSummary"] = extract_render_summary(loaded["live_same_state"])

    return result


def fmt_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def print_section(title: str, data: Optional[Dict[str, Any]]) -> None:
    print(f"\n{title}")
    if data is None:
        print("- not available")
        return
    for key, value in data.items():
        print(f"- {key}: {fmt_value(value)}")


def print_human_summary(summary: Dict[str, Any]) -> None:
    print("Step JSON summary")
    print(f"- baseDir: {summary['baseDir']}")
    print(f"- prefix: {summary['prefix']}")
    print(f"- files: {len(summary['files'])}")
    if summary["loadErrors"]:
        print(f"- loadErrors: {summary['loadErrors']}")
    if summary["missingKnownSuffixes"]:
        print(
            "- missingKnownSuffixes: "
            + ", ".join(summary["missingKnownSuffixes"])
        )

    print_section("Candidate compare", summary.get("candidate"))
    print_section("Coverage", summary.get("coverage"))
    print_section("Runtime", summary.get("runtime"))
    print_section("Limited draw", summary.get("limitedDraw"))
    print_section("Visible compare", summary.get("visibleCompare"))
    print_section("Dry-run visible compare", summary.get("dryRunVisibleCompare"))
    print_section("ScreenCoarse sweep", summary.get("screenCoarseSweep"))
    print_section("Promotion validation", summary.get("promotionValidation"))
    print_section("Step111 timing", summary.get("step111Timing"))
    print_section(
        "Step111 pipeline parity gap closure",
        summary.get("step111PipelineParityGapClosure"),
    )
    print_section(
        "Step112 camera projection orientation parity",
        summary.get("step112CameraProjectionOrientationParity"),
    )
    print_section("GPU visible record dry-run", summary.get("gpuVisibleRecordDryRun"))
    print_section("GPU raw visible record dry-run", summary.get("gpuRawVisibleRecordDryRun"))
    print_section("Output capture diagnostic", summary.get("outputCaptureDiagnostic"))
    print_section(
        "Step75 camera-aware visible output",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step75CameraAwareVisibleOutput"
        ),
    )
    print_section(
        "Step76 many camera-aware visible output",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step76ManyCameraAwareVisibleOutput"
        ),
    )
    print_section(
        "Step77 WebGPU-owned visible samples",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step77WebGpuOwnedVisibleOutput"
        ),
    )
    print_section(
        "Step78 true WebGPU visible records",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step78TrueVisibleRecordOutput"
        ),
    )
    print_section(
        "Step79 WebGPU 4D state visible pipeline",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step79WebGpu4DStateVisiblePipeline"
        ),
    )
    print_section(
        "Step80 WebGPU 4D state evaluation pipeline",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step80WebGpu4DStateEvaluationPipeline"
        ),
    )
    print_section(
        "Step81 WebGPU Gaussian attribute evaluation pipeline",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step81WebGpuGaussianAttributeEvaluationPipeline"
        ),
    )
    print_section(
        "Step82 WebGPU Gaussian footprint pipeline",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step82WebGpuGaussianFootprintPipeline"
        ),
    )
    print_section(
        "Step83 WebGPU tile-aware render input pipeline",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step83WebGpuTileAwareRenderInputPipeline"
        ),
    )
    print_section(
        "Step84 WebGPU GPU-owned tile list layout pipeline",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step84WebGpuGpuOwnedTileListLayoutPipeline"
        ),
    )
    print_section(
        "Step85 WebGPU tile-list compositor pipeline",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step85WebGpuTileListCompositorPipeline"
        ),
    )
    print_section(
        "Step86 WebGPU backend boundary and dirty contract",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step86BackendBoundaryAndDirtyContract"
        ),
    )
    print_section(
        "Step87 WebGPU tile depth ordering for compositor",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step87TileDepthOrderingForCompositor"
        ),
    )
    print_section(
        "Step88 WebGPU tile-compositor frame implementation",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step88TileCompositorFrameImplementation"
        ),
    )
    print_section(
        "Step89 real WebGPU tile-compositor output",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step89RealTileCompositorOutput"
        ),
    )
    print_section(
        "Step90 realtime WebGPU compositor runtime path",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step90RealtimeRuntimePath"
        ),
    )
    print_section(
        "Step91 WebGPU GPU-side tile ordering",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step91GpuSideTileOrdering"
        ),
    )
    print_section(
        "Step92 WebGPU per-tile depth sort",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step92PerTileDepthSort"
        ),
    )
    print_section(
        "Step93 WebGPU overflow-aware tile ordering",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step93OverflowAwareTileOrdering"
        ),
    )
    print_section(
        "Step94 WebGPU parallel per-tile sort",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step94ParallelPerTileSort"
        ),
    )
    print_section(
        "Step96 WebGPU production tile compositor",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step96ProductionTileCompositor"
        ),
    )
    print_section(
        "Step97 WebGPU time-driven production runtime",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step97TimeDrivenProductionRuntime"
        ),
    )
    print_section(
        "Step98 WebGPU viewer-connected interactive scheduler",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step98ViewerConnectedInteractiveScheduler"
        ),
    )
    print_section(
        "Step99 WebGPU interactive camera dirty runtime",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step99InteractiveCameraDirtyRuntime"
        ),
    )
    print_section(
        "Step100 WebGPU unified production interaction scheduler",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step100UnifiedInteractionSchedulerRuntime"
        ),
    )
    print_section(
        "Step101 WebGPU selective dirty dependency execution",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step101SelectiveDirtyDependencyExecution"
        ),
    )
    print_section(
        "Step102 WebGPU production resource lifecycle",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step102ProductionResourceLifecycle"
        ),
    )
    print_section(
        "Step103 WebGPU production runtime boundary review",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step103ProductionRuntimeBoundaryReview"
        ),
    )
    print_section(
        "Step104 WebGPU compositor work reduction visual safety",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step104CompositorWorkReductionVisualSafety"
        ),
    )
    print_section(
        "Step105 WebGPU CUDA reference visual parity baseline",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step105CudaReferenceVisualParityBaseline"
        ),
    )
    print_section(
        "Step106 WebGPU capability-based regression gate",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step106CapabilityBasedRegressionGate"
        ),
    )
    print_section(
        "Step107 WebGPU fixed reference camera contract",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step107FixedReferenceCameraContract"
        ),
    )
    print_section(
        "Step108 WebGPU CUDA reference camera evidence",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step108CudaReferenceCameraEvidence"
        ),
    )
    print_section(
        "Step109 WebGPU fixed reference camera activation",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step109FixedReferenceCameraActivation"
        ),
    )
    print_section(
        "Step110 fixed-condition visual comparison readiness",
        summary.get("webgpuVisibleRecordDryRun", {}).get(
            "step110FixedConditionVisualComparison"
        ),
    )
    print_section(
        "Fixed-condition visual comparison result",
        summary.get("fixedConditionVisualComparison"),
    )
    print_section("WebGPU visible record dry-run", summary.get("webgpuVisibleRecordDryRun"))
    print_section("Association", summary.get("association"))
    print_section("Render summary", summary.get("renderSummary"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize key values from saved Step JSON files."
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing saved JSON files.",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Step file prefix, e.g. step107_000151_v13.",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Optional output path for extracted JSON summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(args.dir)
    prefix = args.prefix

    summary = summarize_step(base_dir, prefix)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print_human_summary(summary)

    return 0 if not summary["loadErrors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
