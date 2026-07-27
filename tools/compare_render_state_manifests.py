#!/usr/bin/env python3
"""Compare CUDA and WebGPU render-state manifests for Step114.

This tool does not claim visual parity.  It records whether the manifest
evidence is complete enough to allow a future fixed-condition image comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


CRITICAL_IMAGE_SPACE_KEYS = [
    "pixelOrigin",
    "yDirection",
    "ndcToPixel",
    "halfPixelConvention",
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_identity(path: Path) -> Dict[str, Any]:
    return {
        "absolutePath": str(path),
        "exists": path.exists(),
        "sizeBytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": sha256_file(path),
    }


def get_path(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def unwrap_status(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value and "status" in value:
        return value.get("value")
    return value


def status_of(value: Any) -> str:
    if isinstance(value, dict) and isinstance(value.get("status"), str):
        return value["status"]
    if value is None:
        return "missing"
    return "available"


def is_unknown(value: Any) -> bool:
    return value is None or status_of(value) in {"unknown", "missing"}


def collect_unknown_fields(manifest: Dict[str, Any], paths: Iterable[str]) -> list[Dict[str, Any]]:
    unknowns = []
    for path in paths:
        value = get_path(manifest, path)
        if is_unknown(value):
            unknowns.append(
                {
                    "path": path,
                    "status": status_of(value),
                    "reason": value.get("reason") if isinstance(value, dict) else None,
                }
            )
    return unknowns


def compare_scalar(cuda: Dict[str, Any], webgpu: Dict[str, Any], cuda_path: str, webgpu_path: str) -> Dict[str, Any]:
    cuda_value = unwrap_status(get_path(cuda, cuda_path))
    webgpu_value = unwrap_status(get_path(webgpu, webgpu_path))
    return {
        "cudaPath": cuda_path,
        "webgpuPath": webgpu_path,
        "cudaValue": cuda_value,
        "webgpuValue": webgpu_value,
        "match": cuda_value == webgpu_value and cuda_value is not None,
    }


def flatten_numeric_list(value: Any) -> list[float]:
    if isinstance(value, dict) and "value" in value:
        value = value.get("value")
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(flatten_numeric_list(item))
        return out
    if isinstance(value, dict):
        out = []
        for key in sorted(value):
            out.extend(flatten_numeric_list(value[key]))
        return out
    return []


def max_abs_delta(a: Any, b: Any) -> Optional[float]:
    av = flatten_numeric_list(a)
    bv = flatten_numeric_list(b)
    if not av or not bv or len(av) != len(bv):
        return None
    return max(abs(x - y) for x, y in zip(av, bv))


def collect_src_index_records(obj: Any, out: Optional[Dict[int, Dict[str, Any]]] = None) -> Dict[int, Dict[str, Any]]:
    if out is None:
        out = {}
    if isinstance(obj, dict):
        src_index = obj.get("srcIndex", obj.get("sourceIndex", obj.get("idx")))
        if isinstance(src_index, (int, float)) and int(src_index) == src_index:
            out.setdefault(int(src_index), obj)
        for value in obj.values():
            collect_src_index_records(value, out)
    elif isinstance(obj, list):
        for value in obj:
            collect_src_index_records(value, out)
    return out


def get_direct_rasterizer_evidence(cuda_manifest: Dict[str, Any]) -> Dict[str, Any]:
    value = get_path(
        cuda_manifest,
        "imageSpaceConvention.directRasterizerScreenCoordinateEvidence",
        {},
    )
    return value if isinstance(value, dict) else {}


def get_cuda_selected_src_indices(cuda_evidence: Dict[str, Any]) -> list[int]:
    selected = get_path(cuda_evidence, "selectionPolicy.selectedIndices", None)
    if not isinstance(selected, list):
        selected = cuda_evidence.get("selectedIndices")
    if not isinstance(selected, list):
        selected = [
            item.get("srcIndex")
            for item in cuda_evidence.get("records", [])
            if isinstance(item, dict)
        ]
    out: list[int] = []
    seen = set()
    for item in selected or []:
        if isinstance(item, (int, float)) and int(item) == item and int(item) not in seen:
            seen.add(int(item))
            out.append(int(item))
    return out


def collect_direct_webgpu_records(webgpu_manifest: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    evidence = get_path(webgpu_manifest, "directGaussianEvidence", {})
    records = evidence.get("records") if isinstance(evidence, dict) else None
    if isinstance(records, list):
        out: Dict[int, Dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            src_index = record.get("srcIndex", record.get("sourceIndex", record.get("idx")))
            if isinstance(src_index, (int, float)) and int(src_index) == src_index:
                out.setdefault(int(src_index), record)
        return out
    return {}


def collect_cuda_direct_records(cuda_evidence: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    records = cuda_evidence.get("records")
    if isinstance(records, list):
        return collect_src_index_records(records)
    return collect_src_index_records(cuda_evidence)


def compare_vec_stage(
    *,
    stage: str,
    cuda_record: Dict[str, Any],
    webgpu_record: Dict[str, Any],
    cuda_keys: list[str],
    webgpu_keys: list[str],
    tolerance: float,
) -> Dict[str, Any]:
    cuda_value = None
    webgpu_value = None
    for key in cuda_keys:
        value = get_path(cuda_record, key) if "." in key else cuda_record.get(key)
        if value is not None:
            cuda_value = value
            break
    for key in webgpu_keys:
        value = get_path(webgpu_record, key) if "." in key else webgpu_record.get(key)
        if value is not None:
            webgpu_value = value
            break
    delta = max_abs_delta(cuda_value, webgpu_value)
    component_summary = component_delta_summary(cuda_value, webgpu_value)
    available = delta is not None
    return {
        "stage": stage,
        "cudaValue": cuda_value,
        "webgpuValue": webgpu_value,
        "maxAbsDelta": delta,
        "componentMaxAbsDelta": component_summary["componentMaxAbsDelta"],
        "componentSignDiffers": component_summary["componentSignDiffers"],
        "componentPermutationLikely": component_summary["componentPermutationLikely"],
        "tolerance": tolerance,
        "match": available and delta <= tolerance,
        "available": available,
        "status": (
            "matched"
            if available and delta <= tolerance
            else ("mismatched" if available else "missing")
        ),
        "missingReason": None if available else "cuda-or-webgpu-stage-evidence-missing",
    }


def compare_bool_stage(stage: str, cuda_value: Any, webgpu_value: Any) -> Dict[str, Any]:
    match = cuda_value is not None and webgpu_value is not None and bool(cuda_value) == bool(webgpu_value)
    available = cuda_value is not None and webgpu_value is not None
    return {
        "stage": stage,
        "cudaValue": cuda_value,
        "webgpuValue": webgpu_value,
        "match": match,
        "available": available,
        "status": (
            "matched"
            if match
            else ("mismatched" if available else "missing")
        ),
        "missingReason": None if available else "cuda-or-webgpu-stage-evidence-missing",
    }


def compare_temporal_scalar_stage(
    stage: str,
    cuda_value: Any,
    webgpu_value: Any,
    tolerance: float,
) -> Dict[str, Any]:
    cuda_unwrapped = unwrap_status(cuda_value)
    webgpu_unwrapped = unwrap_status(webgpu_value)
    try:
        delta = abs(float(cuda_unwrapped) - float(webgpu_unwrapped))
    except (TypeError, ValueError):
        delta = None
    available = delta is not None
    return {
        "stage": stage,
        "cudaValue": cuda_unwrapped,
        "webgpuValue": webgpu_unwrapped,
        "maxAbsDelta": delta,
        "tolerance": tolerance,
        "match": available and delta <= tolerance,
        "available": available,
        "status": (
            "matched"
            if available and delta <= tolerance
            else ("mismatched" if available else "missing")
        ),
        "missingReason": None if available else "cuda-or-webgpu-temporal-stage-evidence-missing",
    }


def compare_temporal_bool_stage(stage: str, cuda_value: Any, webgpu_value: Any) -> Dict[str, Any]:
    cuda_unwrapped = unwrap_status(cuda_value)
    webgpu_unwrapped = unwrap_status(webgpu_value)
    available = cuda_unwrapped is not None and webgpu_unwrapped is not None
    match = available and bool(cuda_unwrapped) == bool(webgpu_unwrapped)
    return {
        "stage": stage,
        "cudaValue": cuda_unwrapped,
        "webgpuValue": webgpu_unwrapped,
        "match": match,
        "available": available,
        "status": (
            "matched"
            if match
            else ("mismatched" if available else "missing")
        ),
        "missingReason": None if available else "cuda-or-webgpu-temporal-stage-evidence-missing",
    }


def vector_delta(a: Any, b: Any) -> Optional[list[float]]:
    av = flatten_numeric_list(a)
    bv = flatten_numeric_list(b)
    if not av or not bv or len(av) != len(bv):
        return None
    return [x - y for x, y in zip(av, bv)]


def normalize_quat4(value: Any) -> Optional[list[float]]:
    values = flatten_numeric_list(value)
    if len(values) < 4:
        return None
    q = values[:4]
    length = math.sqrt(sum(item * item for item in q))
    if not math.isfinite(length) or length <= 1e-8:
        return None
    return [item / length for item in q]


def mat4_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [
            sum(a[row][k] * b[k][col] for k in range(4))
            for col in range(4)
        ]
        for row in range(4)
    ]


def cuda_glm_rotation4d(rotation: Any, rotation_r: Any) -> Optional[list[list[float]]]:
    ql = normalize_quat4(rotation)
    qr = normalize_quat4(rotation_r)
    if ql is None or qr is None:
        return None
    a, b, c, d = ql
    p, q, r, s = qr
    ml = [
        [a, -b, c, -d],
        [b, a, -d, -c],
        [-c, d, a, -b],
        [d, c, b, a],
    ]
    mr = [
        [p, -q, r, s],
        [q, p, -s, r],
        [-r, s, p, q],
        [-s, -r, -q, p],
    ]
    return mat4_mul(mr, ml)


def derive_cuda_temporal_coupling(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    scale = flatten_numeric_list(record.get("scale"))
    if len(scale) < 3:
        return None
    scale_t = record.get("scaleT")
    try:
        scale_t_value = float(scale_t)
    except (TypeError, ValueError):
        return None
    rotation = cuda_glm_rotation4d(record.get("rotation"), record.get("rotationR"))
    if rotation is None:
        return None
    scale_sq = [
        scale[0] * scale[0],
        scale[1] * scale[1],
        scale[2] * scale[2],
        scale_t_value * scale_t_value,
    ]
    cols = [
        [rotation[row][col] for row in range(4)]
        for col in range(4)
    ]

    def sigma_component(a: int, b: int) -> float:
        return sum(scale_sq[row] * cols[a][row] * cols[b][row] for row in range(4))

    cov_t = sigma_component(3, 3)
    if abs(cov_t) <= 1e-8:
        return None
    cov12 = [sigma_component(0, 3), sigma_component(1, 3), sigma_component(2, 3)]
    return {
        "covT": cov_t,
        "cov12": cov12,
        "invCovT": 1.0 / cov_t,
    }


def component_delta_summary(a: Any, b: Any) -> Dict[str, Any]:
    av = flatten_numeric_list(a)
    bv = flatten_numeric_list(b)
    if not av or not bv or len(av) != len(bv):
        return {
            "available": False,
            "componentMaxAbsDelta": [],
            "componentSignDiffers": [],
            "componentPermutationLikely": None,
        }
    deltas = [abs(x - y) for x, y in zip(av, bv)]
    sign_differs = [
        (x < 0 < y) or (y < 0 < x)
        for x, y in zip(av, bv)
    ]
    sorted_a = sorted(round(abs(x), 6) for x in av)
    sorted_b = sorted(round(abs(x), 6) for x in bv)
    same_magnitudes = sorted_a == sorted_b and any(deltas)
    return {
        "available": True,
        "componentMaxAbsDelta": deltas,
        "componentSignDiffers": sign_differs,
        "componentPermutationLikely": same_magnitudes and max(deltas, default=0.0) > 1e-6,
    }


def compare_motion_delta_internal_stage(
    stage: str,
    cuda_value: Any,
    webgpu_value: Any,
    tolerance: float,
    cuda_source: str,
    webgpu_source: str,
) -> Dict[str, Any]:
    delta = max_abs_delta(cuda_value, webgpu_value)
    available = delta is not None
    component_summary = component_delta_summary(cuda_value, webgpu_value)
    return {
        "stage": stage,
        "cudaValue": cuda_value,
        "webgpuValue": webgpu_value,
        "cudaValueSource": cuda_source,
        "webgpuValueSource": webgpu_source,
        "available": available,
        "match": available and delta <= tolerance,
        "status": (
            "matched"
            if available and delta <= tolerance
            else ("mismatched" if available else "missing")
        ),
        "maxAbsDelta": delta,
        "componentMaxAbsDelta": component_summary["componentMaxAbsDelta"],
        "componentSignDiffers": component_summary["componentSignDiffers"],
        "componentPermutationLikely": component_summary["componentPermutationLikely"],
        "tolerance": tolerance,
        "missingReason": None if available else "cuda-or-webgpu-motion-delta-internal-stage-evidence-missing",
    }


def find_webgpu_motion_stage(webgpu_record: Dict[str, Any], stage: str) -> Dict[str, Any]:
    stages = get_path(webgpu_record, "temporalEvaluation.motionDeltaInternalStages", [])
    if isinstance(stages, list):
        for item in stages:
            if isinstance(item, dict) and item.get("stage") == stage:
                return item
    return {}


def stage_value(mapping: Dict[str, Any], key: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    return mapping.get(key)


def build_temporal_source_parameter_stages(
    cuda_record: Dict[str, Any],
    webgpu_record: Dict[str, Any],
) -> list[Dict[str, Any]]:
    source_params = find_webgpu_motion_stage(webgpu_record, "temporal-source-parameters")
    source_value = source_params.get("webgpuValue")
    timestamp_params = find_webgpu_motion_stage(webgpu_record, "timestamp-derived-scalar-inputs")
    timestamp_value = timestamp_params.get("webgpuValue")
    source = source_params.get(
        "valueSource",
        "webgpu-temporal-source-parameters-missing",
    )
    timestamp_source = timestamp_params.get(
        "valueSource",
        "webgpu-temporal-timestamp-inputs-missing",
    )
    return [
        compare_motion_delta_internal_stage(
            "temporal-source-scale-xyz",
            cuda_record.get("scale"),
            stage_value(source_value, "scaleXYZ"),
            1e-5,
            "cuda-production-rasterizer-preprocess-kernel-scale-field",
            source,
        ),
        compare_motion_delta_internal_stage(
            "temporal-source-scale-t",
            cuda_record.get("scaleT"),
            stage_value(source_value, "scaleT"),
            1e-5,
            "cuda-production-rasterizer-preprocess-kernel-scaleT-field",
            source,
        ),
        compare_motion_delta_internal_stage(
            "temporal-source-left-4d-quaternion",
            cuda_record.get("rotation"),
            stage_value(source_value, "rotation"),
            1e-5,
            "cuda-production-rasterizer-preprocess-kernel-left-4d-quaternion-field",
            source,
        ),
        compare_motion_delta_internal_stage(
            "temporal-source-right-4d-quaternion",
            cuda_record.get("rotationR"),
            stage_value(source_value, "rotationR"),
            1e-5,
            "cuda-production-rasterizer-preprocess-kernel-right-4d-quaternion-field",
            source,
        ),
        compare_motion_delta_internal_stage(
            "temporal-source-gaussian-time",
            cuda_record.get("gaussianTime"),
            stage_value(source_value, "gaussianTime"),
            1e-5,
            "cuda-production-rasterizer-preprocess-kernel-gaussian-time-field",
            source,
        ),
        compare_motion_delta_internal_stage(
            "temporal-source-evaluated-time",
            cuda_record.get("timestamp"),
            stage_value(timestamp_value, "actualEvaluatedTimestamp"),
            1e-5,
            "cuda-production-rasterizer-preprocess-kernel-render-timestamp",
            timestamp_source,
        ),
        compare_motion_delta_internal_stage(
            "temporal-source-time-delta",
            cuda_record.get("timeDelta"),
            stage_value(timestamp_value, "timeDelta"),
            1e-5,
            "cuda-production-rasterizer-preprocess-kernel-timeDelta-field",
            timestamp_source,
        ),
    ]


def build_motion_delta_internal_stage_comparison(
    cuda_record: Dict[str, Any],
    webgpu_record: Dict[str, Any],
) -> Dict[str, Any]:
    temporal = webgpu_record.get("temporalEvaluation")
    if not isinstance(temporal, dict):
        temporal = {}
    cuda_delta = vector_delta(
        cuda_record.get("worldPositionAfterTemporal"),
        cuda_record.get("worldPositionInput"),
    )
    webgpu_delta = temporal.get("temporalMotionDelta")
    cuda_post = cuda_record.get("worldPositionAfterTemporal")
    webgpu_post = temporal.get("postTemporalWorldPosition")
    cuda_pre = cuda_record.get("worldPositionInput")
    webgpu_pre = temporal.get("preTemporalWorldPosition")
    timestamp_params = find_webgpu_motion_stage(webgpu_record, "timestamp-derived-scalar-inputs")
    coupling = find_webgpu_motion_stage(webgpu_record, "conditional-covariance-temporal-coupling")
    raw_output = find_webgpu_motion_stage(webgpu_record, "deformation-raw-output")
    weighted_delta = find_webgpu_motion_stage(webgpu_record, "weight-lifetime-activation-applied-delta")
    converted_delta = find_webgpu_motion_stage(webgpu_record, "axis-unit-scale-converted-delta")
    final_delta = find_webgpu_motion_stage(webgpu_record, "final-temporal-motion-delta")
    pre_plus_delta = find_webgpu_motion_stage(webgpu_record, "pre-world-plus-delta")
    cuda_coupling = derive_cuda_temporal_coupling(cuda_record)
    stages = [
        *build_temporal_source_parameter_stages(cuda_record, webgpu_record),
        compare_motion_delta_internal_stage(
            "timestamp-derived-scalar-inputs",
            {
                "requestedTimestamp": cuda_record.get("timestamp"),
                "actualEvaluatedTimestamp": cuda_record.get("timestamp"),
                "timeDelta": cuda_record.get("timeDelta"),
                "effectiveScaleT": cuda_record.get("scaleT"),
            },
            timestamp_params.get("webgpuValue"),
            1e-5,
            "cuda-production-rasterizer-preprocess-kernel-debug-row",
            timestamp_params.get("valueSource", "webgpu-temporal-timestamp-inputs-missing"),
        ),
        compare_motion_delta_internal_stage(
            "conditional-covariance-temporal-coupling",
            cuda_coupling,
            coupling.get("webgpuValue"),
            2e-5,
            "verified-derived-from-cuda-production-debug-row-scale-rotation-scaleT",
            coupling.get("valueSource", "webgpu-temporal-coupling-missing"),
        ),
        compare_motion_delta_internal_stage(
            "deformation-raw-output",
            cuda_delta,
            raw_output.get("webgpuValue"),
            1e-5,
            "cuda-production-rasterizer-worldPositionAfterTemporal-minus-worldPositionInput",
            raw_output.get("valueSource", "webgpu-deformation-raw-output-missing"),
        ),
        compare_motion_delta_internal_stage(
            "weight-lifetime-activation-applied-delta",
            cuda_delta,
            weighted_delta.get("webgpuValue"),
            1e-5,
            "cuda-production-rasterizer-motion-delta-after-temporal-mask",
            weighted_delta.get("valueSource", "webgpu-weighted-motion-delta-missing"),
        ),
        compare_motion_delta_internal_stage(
            "axis-unit-scale-converted-delta",
            cuda_delta,
            converted_delta.get("webgpuValue"),
            1e-5,
            "cuda-production-rasterizer-world-space-motion-delta",
            converted_delta.get("valueSource", "webgpu-axis-unit-delta-missing"),
        ),
        compare_motion_delta_internal_stage(
            "final-temporal-motion-delta",
            cuda_delta,
            webgpu_delta,
            1e-5,
            "cuda-production-rasterizer-worldPositionAfterTemporal-minus-worldPositionInput",
            final_delta.get("valueSource", "webgpu-production-statepositions-buffer-consumed-by-visible-record-dispatch"),
        ),
        compare_motion_delta_internal_stage(
            "pre-world-plus-delta",
            cuda_post,
            webgpu_post,
            1e-5,
            "cuda-production-rasterizer-worldPositionAfterTemporal",
            pre_plus_delta.get("valueSource", "webgpu-production-statepositions-buffer-consumed-by-visible-record-dispatch"),
        ),
    ]
    first_mismatch = first_failed_stage(stages)
    missing = has_missing_stage_evidence(stages)
    if first_mismatch in {
        "temporal-source-scale-xyz",
        "temporal-source-scale-t",
        "temporal-source-left-4d-quaternion",
        "temporal-source-right-4d-quaternion",
        "temporal-source-gaussian-time",
    }:
        root = "B-field-order-stride-alignment-or-component-layout-mismatch"
    elif first_mismatch in {
        "temporal-source-evaluated-time",
        "temporal-source-time-delta",
        "timestamp-derived-scalar-inputs",
    }:
        root = "C-timestamp-scalar-or-basis-input-mismatch"
    elif first_mismatch == "conditional-covariance-temporal-coupling":
        root = "D-deformation-formula-or-network-intermediate-mismatch"
    elif first_mismatch in {
        "deformation-raw-output",
        "final-temporal-motion-delta",
        "pre-world-plus-delta",
    }:
        root = "D-deformation-formula-or-network-operation-mismatch"
    elif first_mismatch == "weight-lifetime-activation-applied-delta":
        root = "E-activation-weight-lifetime-mask-order-mismatch"
    elif first_mismatch == "axis-unit-scale-converted-delta":
        root = "F-axis-permutation-sign-unit-or-scale-mismatch"
    elif missing:
        root = "motion-delta-internal-stage-evidence-missing-before-comparison"
    else:
        root = "none"
    return {
        "schemaVersion": "phase3-step114-fix9-motion-delta-internal-stage-comparison-v1",
        "canonicalSourceParameterStageSchemaVersion":
            "phase3-step114-fix10-temporal-source-parameter-stage-comparison-v1",
        "stages": stages,
        "firstMotionDeltaMismatchSubstage": None if missing and first_mismatch is None else first_mismatch,
        "motionDeltaEvidenceMissing": missing,
        "rootCauseClassification": root,
        "cudaProductionTemporalPath": {
            "entryPoint": "CUDA reference production rasterizer preprocess kernel",
            "motionDeltaDefinition": "worldPositionAfterTemporal - worldPositionInput",
            "actualEvidenceSource": cuda_record.get("actualEvidenceSource")
            or "cuda-production-rasterizer-preprocess-kernel-debug-row",
        },
        "webgpuProductionTemporalPath": {
            "entryPoint": "WebGPU 4D state evaluator production WGSL dispatch",
            "motionDeltaDefinition": "statePositions.xyz - raw source xyz consumed by visible-record dispatch",
            "actualEvidenceSource": temporal.get("actualEvidenceSource"),
            "temporalPathClassification": temporal.get("temporalPathClassification"),
            "temporalPathProductionOwner": temporal.get("temporalPathProductionOwner"),
        },
        "preTemporalWorldPositionMatch": max_abs_delta(cuda_pre, webgpu_pre) is not None
        and max_abs_delta(cuda_pre, webgpu_pre) <= 1e-5,
    }


def build_temporal_substage_comparison(
    cuda_record: Dict[str, Any],
    webgpu_record: Dict[str, Any],
) -> Dict[str, Any]:
    temporal = webgpu_record.get("temporalEvaluation")
    if not isinstance(temporal, dict):
        temporal = {}
    cuda_delta = vector_delta(
        cuda_record.get("worldPositionAfterTemporal"),
        cuda_record.get("worldPositionInput"),
    )
    webgpu_delta = temporal.get("temporalMotionDelta")
    stages = [
        compare_temporal_scalar_stage(
            "requested-timestamp",
            cuda_record.get("timestamp"),
            temporal.get("requestedTimestamp"),
            1e-5,
        ),
        compare_temporal_scalar_stage(
            "actual-evaluated-timestamp",
            cuda_record.get("timestamp"),
            temporal.get("actualEvaluatedTimestamp"),
            1e-5,
        ),
        compare_temporal_scalar_stage(
            "gaussian-time",
            cuda_record.get("gaussianTime"),
            temporal.get("gaussianTime"),
            1e-5,
        ),
        compare_temporal_scalar_stage(
            "time-delta",
            cuda_record.get("timeDelta"),
            temporal.get("timeDelta"),
            1e-5,
        ),
        compare_temporal_scalar_stage(
            "scale-t",
            cuda_record.get("scaleT"),
            temporal.get("effectiveScaleT"),
            1e-5,
        ),
        compare_vec_stage(
            stage="pre-temporal-world-position",
            cuda_record=cuda_record,
            webgpu_record={"preTemporalWorldPosition": temporal.get("preTemporalWorldPosition")},
            cuda_keys=["worldPositionInput", "sourceWorldPosition", "worldPosition"],
            webgpu_keys=["preTemporalWorldPosition"],
            tolerance=1e-5,
        ),
        compare_vec_stage(
            stage="temporal-motion-delta",
            cuda_record={"temporalMotionDelta": cuda_delta},
            webgpu_record={"temporalMotionDelta": webgpu_delta},
            cuda_keys=["temporalMotionDelta"],
            webgpu_keys=["temporalMotionDelta"],
            tolerance=1e-5,
        ),
        compare_vec_stage(
            stage="post-temporal-world-position",
            cuda_record=cuda_record,
            webgpu_record={"postTemporalWorldPosition": temporal.get("postTemporalWorldPosition")},
            cuda_keys=["worldPositionAfterTemporal", "temporalWorldPosition", "worldPosition"],
            webgpu_keys=["postTemporalWorldPosition"],
            tolerance=1e-5,
        ),
        compare_temporal_bool_stage(
            "temporal-validity",
            cuda_record.get("valid"),
            get_path(temporal, "temporalValidity.validForProjectionInput"),
        ),
    ]
    first_mismatch = first_failed_stage(stages)
    missing = has_missing_stage_evidence(stages)
    if first_mismatch in {"requested-timestamp", "actual-evaluated-timestamp", "gaussian-time", "time-delta"}:
        root = "A-timestamp-time-normalization-or-time-duration-mismatch"
    elif first_mismatch == "scale-t":
        root = "B-temporal-parameter-conversion-or-layout-mismatch"
    elif first_mismatch in {"temporal-motion-delta", "post-temporal-world-position"}:
        root = "D-webgpu-temporal-deformation-path-differs-from-cuda-production"
    elif first_mismatch == "pre-temporal-world-position":
        root = "E-checkpoint-asset-lineage-or-source-position-mismatch"
    elif first_mismatch == "temporal-validity":
        root = "F-temporal-validity-vocabulary-or-lifetime-classification-mismatch"
    elif missing:
        root = "temporal-evidence-missing-before-comparison"
    else:
        root = "none"
    motion_delta_internal = build_motion_delta_internal_stage_comparison(cuda_record, webgpu_record)
    return {
        "stages": stages,
        "firstTemporalMismatchSubstage": None if missing and first_mismatch is None else first_mismatch,
        "temporalEvidenceMissing": missing,
        "rootCauseClassification": root,
        "webgpuTemporalActualSource": temporal.get("actualEvidenceSource"),
        "webgpuTemporalEvaluatorImplementation": temporal.get("temporalEvaluatorImplementation"),
        "motionDeltaInternalStageComparison": motion_delta_internal,
    }


def first_failed_stage(stage_records: list[Dict[str, Any]]) -> Optional[str]:
    for record in stage_records:
        if record.get("available") is True and record.get("match") is not True:
            return record.get("stage")
    return None


def has_missing_stage_evidence(stage_records: list[Dict[str, Any]]) -> bool:
    return any(record.get("available") is not True for record in stage_records)


def summarize_stage_errors(comparisons: list[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for comparison in comparisons:
        for stage in comparison.get("stages", []):
            name = stage.get("stage")
            if not name:
                continue
            current = out.setdefault(
                name,
                {
                    "availableCount": 0,
                    "matchCount": 0,
                    "mismatchCount": 0,
                    "missingCount": 0,
                    "maxAbsDelta": None,
                    "componentMaxAbsDelta": None,
                    "tolerance": stage.get("tolerance"),
                },
            )
            if stage.get("available") is True:
                current["availableCount"] += 1
                if stage.get("match") is not True:
                    current["mismatchCount"] += 1
            else:
                current["missingCount"] += 1
            if stage.get("match") is True:
                current["matchCount"] += 1
            delta = stage.get("maxAbsDelta")
            if delta is not None:
                current["maxAbsDelta"] = (
                    delta
                    if current["maxAbsDelta"] is None
                    else max(current["maxAbsDelta"], delta)
                )
            component_delta = stage.get("componentMaxAbsDelta")
            if isinstance(component_delta, list):
                current_delta = current.get("componentMaxAbsDelta")
                if not isinstance(current_delta, list):
                    current["componentMaxAbsDelta"] = component_delta
                elif len(current_delta) == len(component_delta):
                    current["componentMaxAbsDelta"] = [
                        max(a, b)
                        for a, b in zip(current_delta, component_delta)
                    ]
    return out


def classify_record_zero_cause(
    *,
    cuda_selected: list[int],
    webgpu_manifest: Dict[str, Any],
    webgpu_records: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    evidence = get_path(webgpu_manifest, "directGaussianEvidence", {})
    canonical = get_path(webgpu_manifest, "canonicalComparisonIndexSet", {})
    src_semantics = get_path(webgpu_manifest, "srcIndexSemantics", {})
    if webgpu_records:
        cause = "records-present"
    elif isinstance(evidence, dict) and evidence.get("recordCount") == 0 and cuda_selected:
        cause = "C-webgpu-production-runtime-did-not-capture-requested-src-indices"
    elif isinstance(canonical, dict) and canonical.get("missing") is True:
        cause = "B-canonical-index-set-not-passed-to-webgpu-capture"
    elif not isinstance(evidence, dict) or not evidence:
        cause = "B-webgpu-render-state-manifest-missing-direct-record-section"
    elif src_semantics.get("mappingDecision") in {"blocked-index-mapping-unknown", "runtime-original-index-preserved-but-checkpoint-to-asset-mapping-artifact-missing"}:
        cause = "D-index-mapping-not-directly-proven"
    else:
        cause = "A-comparison-tool-schema-path-mismatch-or-empty-webgpu-direct-records"
    return {
        "cause": cause,
        "cudaSelectedCount": len(cuda_selected),
        "webgpuDirectEvidencePresent": isinstance(evidence, dict) and bool(evidence),
        "webgpuDirectEvidenceRecordCount": evidence.get("recordCount") if isinstance(evidence, dict) else None,
        "canonicalIndexSetPresent": isinstance(canonical, dict) and bool(canonical),
        "canonicalIndexSetSelectedCount": canonical.get("selectedCount") if isinstance(canonical, dict) else None,
        "srcIndexMappingDecision": src_semantics.get("mappingDecision") if isinstance(src_semantics, dict) else None,
    }


def build_semantic_matrix_comparison(cuda_manifest: Dict[str, Any], webgpu_manifest: Dict[str, Any]) -> Dict[str, Any]:
    raw_world_view_delta = max_abs_delta(
        get_path(cuda_manifest, "camera.worldViewTransform"),
        get_path(webgpu_manifest, "camera.worldViewTransform"),
    )
    raw_full_proj_delta = max_abs_delta(
        get_path(cuda_manifest, "camera.fullProjTransform"),
        get_path(webgpu_manifest, "camera.fullProjTransform"),
    )
    cuda_convention = get_path(cuda_manifest, "camera.matrixConvention", {})
    webgpu_convention = get_path(webgpu_manifest, "camera.matrixConvention", {})
    can_normalize = (
        isinstance(cuda_convention, dict)
        and isinstance(webgpu_convention, dict)
        and cuda_convention.get("role")
        and webgpu_convention.get("role")
        and cuda_convention.get("layout")
        and webgpu_convention.get("layout")
    )
    return {
        "worldViewTransform": {
            "rawMatrixMatch": raw_world_view_delta == 0,
            "rawMaxAbsDelta": raw_world_view_delta,
            "declaredConventionNormalizedMatch": None,
            "representativePointTransformMatch": None,
        },
        "fullProjTransform": {
            "rawMatrixMatch": raw_full_proj_delta == 0,
            "rawMaxAbsDelta": raw_full_proj_delta,
            "declaredConventionNormalizedMatch": None,
            "representativePointTransformMatch": None,
        },
        "normalizationStatus": "ready" if can_normalize else "unknown",
        "normalizationBlockedReason": None if can_normalize else "matrix-role-layout-vector-convention-not-directly-declared-on-both-manifests",
        "cudaMatrixConvention": cuda_convention,
        "webgpuMatrixConvention": webgpu_convention,
    }


def classify_projection_root_cause(
    projection_summary: Dict[str, Any],
    first_projection_mismatch: Optional[str],
) -> str:
    if not first_projection_mismatch:
        if any(item.get("missingCount", 0) > 0 for item in projection_summary.values()):
            return "projection-stage-evidence-missing-before-comparison"
        return "none"
    camera_stage = projection_summary.get("camera-space-position", {})
    screen_stage = projection_summary.get("screen-space-center", {})
    if (
        first_projection_mismatch in {"clip-position", "ndc-position"}
        and camera_stage.get("mismatchCount", 0) == 0
        and camera_stage.get("missingCount", 0) == 0
        and screen_stage.get("mismatchCount", 0) == 0
        and screen_stage.get("missingCount", 0) == 0
    ):
        return "G-cuda-debug-clip-ndc-field-and-webgpu-canonical-field-vocabulary-mismatch"
    if first_projection_mismatch == "clip-position":
        return "F-production-projection-clip-calculation-mismatch"
    if first_projection_mismatch == "ndc-position":
        return "B-perspective-divide-or-ndc-stage-correspondence-mismatch"
    if first_projection_mismatch == "screen-space-center":
        return "D-y-direction-viewport-origin-or-pixel-center-convention-mismatch"
    if first_projection_mismatch == "depth":
        return "C-ndc-depth-range-or-depth-sign-convention-mismatch"
    return "projection-stage-mismatch-unclassified"


def build_projection_contract_comparison(cuda_manifest: Dict[str, Any], webgpu_manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schemaVersion": "phase3-step114-fix10-clip-ndc-canonical-contract-v1",
        "cudaProjectionConvention": {
            "matrixConvention": get_path(cuda_manifest, "camera.matrixConvention", {}),
            "imageSpaceConvention": get_path(cuda_manifest, "imageSpaceConvention", {}),
            "intrinsics": get_path(cuda_manifest, "camera.intrinsics", {}),
        },
        "webgpuProjectionConvention": {
            "matrixConvention": get_path(webgpu_manifest, "camera.matrixConvention", {}),
            "imageSpaceConvention": get_path(webgpu_manifest, "imageSpaceConvention", {}),
            "intrinsics": get_path(webgpu_manifest, "camera.intrinsics", {}),
            "projectionContract": get_path(webgpu_manifest, "camera.fullProjTransform", {}),
        },
        "stageVocabulary": [
            "camera-space-position",
            "clip-position",
            "ndc-position",
            "screen-space-center",
            "depth",
        ],
        "comparisonPrinciple":
            "camera-space and screen-space center agreement prevents clip/NDC debug-field mismatch from being treated as a production projection fix by itself",
    }


def build_direct_src_index_comparison(cuda_manifest: Dict[str, Any], webgpu_manifest: Dict[str, Any]) -> Dict[str, Any]:
    cuda_evidence = get_direct_rasterizer_evidence(cuda_manifest)
    cuda_selected = get_cuda_selected_src_indices(cuda_evidence)
    cuda_records = collect_cuda_direct_records(cuda_evidence)
    webgpu_records = collect_direct_webgpu_records(webgpu_manifest)
    common_indices = sorted(set(cuda_records).intersection(webgpu_records))
    comparisons = []
    missing_stage_record_count = 0
    mismatched_stage_record_count = 0
    for src_index in common_indices:
        cuda_record = cuda_records[src_index]
        webgpu_record = webgpu_records[src_index]
        record_invalid = webgpu_record.get("invalid") is True
        record_culled = webgpu_record.get("culled") is True
        stages = [
            compare_vec_stage(
                stage="source-world-position",
                cuda_record=cuda_record,
                webgpu_record=webgpu_record,
                cuda_keys=["worldPositionInput", "sourceWorldPosition", "worldPosition", "position", "world"],
                webgpu_keys=["rawSourcePosition", "sourceWorldPosition", "worldPosition"],
                tolerance=1e-5,
            ),
            compare_vec_stage(
                stage="temporal-world-position",
                cuda_record=cuda_record,
                webgpu_record=webgpu_record,
                cuda_keys=["worldPositionAfterTemporal", "temporalWorldPosition", "worldPosition", "position"],
                webgpu_keys=["temporalWorldPosition", "worldPosition"],
                tolerance=1e-5,
            ),
            compare_vec_stage(
                stage="camera-space-position",
                cuda_record=cuda_record,
                webgpu_record=webgpu_record,
                cuda_keys=["cameraSpacePositionUnclamped", "cameraSpacePosition", "viewPosition"],
                webgpu_keys=["cameraSpacePosition", "viewPosition"],
                tolerance=1e-4,
            ),
            compare_vec_stage(
                stage="clip-position",
                cuda_record=cuda_record,
                webgpu_record=webgpu_record,
                cuda_keys=["clip", "clipPosition"],
                webgpu_keys=["clipPosition", "clip"],
                tolerance=1e-4,
            ),
            compare_vec_stage(
                stage="ndc-position",
                cuda_record=cuda_record,
                webgpu_record=webgpu_record,
                cuda_keys=["ndc", "ndcPosition"],
                webgpu_keys=["ndc"],
                tolerance=1e-4,
            ),
            compare_vec_stage(
                stage="screen-space-center",
                cuda_record=cuda_record,
                webgpu_record=webgpu_record,
                cuda_keys=["screenCenter", "pointImage"],
                webgpu_keys=["screenCenter", "centerPx"],
                tolerance=1e-3,
            ),
            compare_vec_stage(
                stage="depth",
                cuda_record=cuda_record,
                webgpu_record=webgpu_record,
                cuda_keys=["depth"],
                webgpu_keys=["depth", "z"],
                tolerance=1e-4,
            ),
            compare_bool_stage(
                "valid-culled-classification",
                cuda_record.get("valid", None),
                webgpu_record.get("valid", None),
            ),
            compare_vec_stage(
                stage="radius-footprint",
                cuda_record=cuda_record,
                webgpu_record=webgpu_record,
                cuda_keys=["radius"],
                webgpu_keys=["radius", "footprintRadius"],
                tolerance=1e-3,
            ),
            compare_vec_stage(
                stage="conic-coefficients",
                cuda_record=cuda_record,
                webgpu_record=webgpu_record,
                cuda_keys=["conic"],
                webgpu_keys=["conic", "productionFootprintConic"],
                tolerance=1e-4,
            ),
            compare_vec_stage(
                stage="screen-space-covariance",
                cuda_record=cuda_record,
                webgpu_record=webgpu_record,
                cuda_keys=["covariance2D", "projectedCovariance2D", "screenSpaceCovariance"],
                webgpu_keys=["covariance2D", "productionFootprintCovariance2D"],
                tolerance=1e-4,
            ),
            compare_vec_stage(
                stage="footprint-bounds",
                cuda_record=cuda_record,
                webgpu_record=webgpu_record,
                cuda_keys=["footprintBounds", "bounds"],
                webgpu_keys=["footprintBounds", "bounds"],
                tolerance=1e-3,
            ),
        ]
        center_delta = next(
            (stage.get("maxAbsDelta") for stage in stages if stage.get("stage") == "screen-space-center"),
            None,
        )
        depth_delta = next(
            (stage.get("maxAbsDelta") for stage in stages if stage.get("stage") == "depth"),
            None,
        )
        radius_delta = next(
            (stage.get("maxAbsDelta") for stage in stages if stage.get("stage") == "radius-footprint"),
            None,
        )
        record_missing_before_comparison = (
            record_invalid or has_missing_stage_evidence(stages)
        )
        record_first_mismatch = (
            None if record_missing_before_comparison else first_failed_stage(stages)
        )
        if record_missing_before_comparison:
            missing_stage_record_count += 1
        if record_first_mismatch:
            mismatched_stage_record_count += 1
        temporal_comparison = build_temporal_substage_comparison(cuda_record, webgpu_record)
        comparisons.append(
            {
                "srcIndex": src_index,
                "cudaActualSource": cuda_record.get("actualEvidenceSource")
                or "cuda-production-rasterizer-direct-evidence",
                "webgpuActualSource": webgpu_record.get("actualEvidenceSource")
                or webgpu_record.get("source")
                or "webgpu-render-state-manifest-direct-gaussian-evidence",
                "recordStatus": (
                    "invalid"
                    if record_invalid
                    else ("culled" if record_culled else "available")
                ),
                "recordCulled": record_culled,
                "recordInvalid": record_invalid,
                "culledStage": webgpu_record.get("culledStage"),
                "culledReason": webgpu_record.get("cullReason") or webgpu_record.get("missingReason"),
                "stages": stages,
                "firstMismatchStage": record_first_mismatch,
                "evidenceMissingBeforeComparison": record_missing_before_comparison,
                "centerMaxAbsDelta": center_delta,
                "depthMaxAbsDelta": depth_delta,
                "radiusMaxAbsDelta": radius_delta,
                "validClassificationMatch": next(
                    (stage.get("match") for stage in stages if stage.get("stage") == "valid-culled-classification"),
                    None,
                ),
                "temporalEvaluationComparison": temporal_comparison,
                "comparisonReady": (
                    not record_invalid
                    and not record_missing_before_comparison
                    and any(stage.get("available") is True for stage in stages)
                ),
            }
        )
    missing_webgpu = sorted(set(cuda_records).difference(webgpu_records))
    first_mismatch = None
    for comparison in comparisons:
        if comparison.get("firstMismatchStage"):
            first_mismatch = comparison.get("firstMismatchStage")
            break
    evidence_missing = (
        len(common_indices) <= 0
        or any(comparison.get("evidenceMissingBeforeComparison") for comparison in comparisons)
    )
    temporal_first_mismatch = next(
        (
            comparison.get("temporalEvaluationComparison", {}).get("firstTemporalMismatchSubstage")
            for comparison in comparisons
            if comparison.get("temporalEvaluationComparison", {}).get("firstTemporalMismatchSubstage")
        ),
        None,
    )
    motion_delta_first_mismatch = next(
        (
            get_path(
                comparison,
                "temporalEvaluationComparison.motionDeltaInternalStageComparison.firstMotionDeltaMismatchSubstage",
            )
            for comparison in comparisons
            if get_path(
                comparison,
                "temporalEvaluationComparison.motionDeltaInternalStageComparison.firstMotionDeltaMismatchSubstage",
            )
        ),
        None,
    )
    temporal_missing = any(
        comparison.get("temporalEvaluationComparison", {}).get("temporalEvidenceMissing")
        for comparison in comparisons
    )
    motion_delta_missing = any(
        get_path(
            comparison,
            "temporalEvaluationComparison.motionDeltaInternalStageComparison.motionDeltaEvidenceMissing",
        )
        for comparison in comparisons
    )
    temporal_root_causes = sorted(
        {
            comparison.get("temporalEvaluationComparison", {}).get("rootCauseClassification")
            for comparison in comparisons
            if comparison.get("temporalEvaluationComparison", {}).get("rootCauseClassification")
            and comparison.get("temporalEvaluationComparison", {}).get("rootCauseClassification") != "none"
        }
    )
    motion_delta_root_causes = sorted(
        {
            get_path(
                comparison,
                "temporalEvaluationComparison.motionDeltaInternalStageComparison.rootCauseClassification",
            )
            for comparison in comparisons
            if get_path(
                comparison,
                "temporalEvaluationComparison.motionDeltaInternalStageComparison.rootCauseClassification",
            )
            and get_path(
                comparison,
                "temporalEvaluationComparison.motionDeltaInternalStageComparison.rootCauseClassification",
            ) != "none"
        }
    )
    projection_stage_names = {
        "camera-space-position",
        "clip-position",
        "ndc-position",
        "screen-space-center",
        "depth",
    }
    radius_stage_names = {
        "radius-footprint",
        "conic-coefficients",
        "screen-space-covariance",
        "footprint-bounds",
    }
    projection_stage_summary = summarize_stage_errors(
        [
            {
                "stages": [
                    stage for stage in comparison.get("stages", [])
                    if stage.get("stage") in projection_stage_names
                ]
            }
            for comparison in comparisons
        ]
    )
    radius_stage_summary = summarize_stage_errors(
        [
            {
                "stages": [
                    stage for stage in comparison.get("stages", [])
                    if stage.get("stage") in radius_stage_names
                ]
            }
            for comparison in comparisons
        ]
    )
    first_projection_mismatch = next(
        (
            stage.get("stage")
            for comparison in comparisons
            for stage in comparison.get("stages", [])
            if stage.get("stage") in projection_stage_names
            and stage.get("available") is True
            and stage.get("match") is not True
        ),
        None,
    )
    invalid_webgpu_count = sum(
        1 for record in webgpu_records.values() if record.get("invalid") is True
    )
    culled_webgpu_count = sum(
        1 for record in webgpu_records.values() if record.get("culled") is True
    )
    decision_ready = (
        comparisons
        and first_mismatch is None
        and not evidence_missing
        and invalid_webgpu_count == 0
    )
    if first_mismatch:
        blocked_reason = f"first-mismatch-stage:{first_mismatch}"
    elif evidence_missing:
        blocked_reason = "evidence-missing-before-comparison"
    elif invalid_webgpu_count > 0:
        blocked_reason = "webgpu-direct-record-invalid"
    elif not comparisons:
        blocked_reason = "missing-common-cuda-webgpu-src-index-records"
    else:
        blocked_reason = None
    zero_cause = classify_record_zero_cause(
        cuda_selected=cuda_selected,
        webgpu_manifest=webgpu_manifest,
        webgpu_records=webgpu_records,
    )
    return {
        "available": bool(cuda_records),
        "canonicalComparisonIndexSet": {
            "source": "cuda-direct-rasterizer-selection-policy",
            "selectedSrcIndices": cuda_selected,
            "selectedCount": len(cuda_selected),
        },
        "cudaActualEvidenceSource": "cuda-production-rasterizer-preprocess-kernel-direct-evidence",
        "webgpuActualEvidenceSource": get_path(
            webgpu_manifest,
            "directGaussianEvidence.actualEvidenceSource",
            "missing-webgpu-direct-gaussian-evidence",
        ),
        "cudaRecordCount": len(cuda_records),
        "webgpuSrcIndexRecordCount": len(webgpu_records),
        "commonSrcIndexCount": len(common_indices),
        "missingWebGpuSrcIndices": missing_webgpu,
        "duplicateWebGpuRecordCount": sum(
            max(0, int(record.get("duplicateOccurrenceCount", 0) or 0))
            for record in webgpu_records.values()
        ),
        "invalidWebGpuRecordCount": sum(
            1 for record in webgpu_records.values() if record.get("invalid") is True
        ),
        "culledWebGpuRecordCount": culled_webgpu_count,
        "missingWebGpuRecordCount": len(missing_webgpu),
        "missingStageEvidenceRecordCount": missing_stage_record_count,
        "mismatchedStageRecordCount": mismatched_stage_record_count,
        "evidenceMissingBeforeComparison": evidence_missing,
        "webgpuRecordZeroCause": zero_cause,
        "comparisons": comparisons,
        "stageErrorSummary": summarize_stage_errors(comparisons),
        "temporalStageComparisonSummary": summarize_stage_errors(
            [
                {
                    "stages": comparison.get("temporalEvaluationComparison", {}).get("stages", [])
                }
                for comparison in comparisons
            ]
        ),
        "motionDeltaInternalStageComparisonSummary": summarize_stage_errors(
            [
                {
                    "stages": get_path(
                        comparison,
                        "temporalEvaluationComparison.motionDeltaInternalStageComparison.stages",
                        [],
                    )
                }
                for comparison in comparisons
            ]
        ),
        "projectionCanonicalStageComparisonSummary": projection_stage_summary,
        "projectionCanonicalContract": build_projection_contract_comparison(cuda_manifest, webgpu_manifest),
        "firstProjectionMismatchSubstage": first_projection_mismatch,
        "projectionRootCauseClassification": classify_projection_root_cause(
            projection_stage_summary,
            first_projection_mismatch,
        ),
        "radiusFootprintStageComparisonSummary": radius_stage_summary,
        "firstTemporalMismatchSubstage": temporal_first_mismatch,
        "temporalEvidenceMissing": temporal_missing,
        "temporalRootCauseClassifications": temporal_root_causes,
        "firstMotionDeltaMismatchSubstage": motion_delta_first_mismatch,
        "motionDeltaEvidenceMissing": motion_delta_missing,
        "motionDeltaRootCauseClassifications": motion_delta_root_causes,
        "firstMismatchStage": first_mismatch,
        "diagonalOrientationMismatchClassification": (
            "camera-space-or-earlier-mismatch"
            if first_mismatch in {"source-world-position", "temporal-world-position", "camera-space-position"}
            else (
                "projection-debug-vocabulary-or-canonical-stage-mismatch"
                if first_mismatch in {"clip-position", "ndc-position"}
                else ("image-space-or-pixel-coordinate-mismatch" if first_mismatch == "screen-space-center" else None)
            )
        ),
        "decision": (
            "ready" if decision_ready else "blocked"
        ),
        "blockedReason": blocked_reason,
    }


def build_comparison(
    *,
    cuda_manifest: Dict[str, Any],
    webgpu_manifest: Optional[Dict[str, Any]],
    cuda_manifest_path: Path,
    webgpu_manifest_path: Optional[Path],
    old_cuda_manifest: Optional[Dict[str, Any]] = None,
    old_cuda_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    webgpu = webgpu_manifest or {}
    has_webgpu = webgpu_manifest is not None
    cuda_required_paths = [
        "lineage.checkpoint.sha256",
        "lineage.datasetRoot",
        "renderState.cameraLabel",
        "renderState.frameNumber",
        "renderState.viewId",
        "renderState.timestamp",
        "renderState.width",
        "renderState.height",
        "camera.intrinsics.fx",
        "camera.intrinsics.fy",
        "camera.intrinsics.cx",
        "camera.intrinsics.cy",
        "camera.worldViewTransform",
        "camera.fullProjTransform",
        *(f"imageSpaceConvention.{key}" for key in CRITICAL_IMAGE_SPACE_KEYS),
    ]
    webgpu_required_paths = [
        "lineage.gaussianAsset.identity.sha256",
        "lineage.gaussianAsset.gaussianCount",
        "renderState.cameraLabel",
        "renderState.frameNumber",
        "renderState.viewId",
        "renderState.datasetTime",
        "renderState.viewport.width",
        "renderState.viewport.height",
        "renderState.webgpuCameraConstantsSource",
        "camera.intrinsics.fx",
        "camera.intrinsics.fy",
        "camera.intrinsics.cx",
        "camera.intrinsics.cy",
        "camera.worldViewTransform",
        "camera.fullProjTransform",
        *(f"imageSpaceConvention.{key}" for key in CRITICAL_IMAGE_SPACE_KEYS),
    ]
    cuda_unknowns = collect_unknown_fields(cuda_manifest, cuda_required_paths)
    webgpu_unknowns = (
        collect_unknown_fields(webgpu, webgpu_required_paths)
        if has_webgpu
        else [{"path": "webgpuRenderStateManifest", "status": "missing", "reason": "missing WebGPU render-state manifest"}]
    )
    field_comparisons = {
        "cameraLabel": compare_scalar(cuda_manifest, webgpu, "renderState.cameraLabel", "renderState.cameraLabel"),
        "frameNumber": compare_scalar(cuda_manifest, webgpu, "renderState.frameNumber", "renderState.frameNumber"),
        "viewId": compare_scalar(cuda_manifest, webgpu, "renderState.viewId", "renderState.viewId"),
        "timestamp": compare_scalar(cuda_manifest, webgpu, "renderState.timestamp", "renderState.datasetTime"),
        "width": compare_scalar(cuda_manifest, webgpu, "renderState.width", "renderState.viewport.width"),
        "height": compare_scalar(cuda_manifest, webgpu, "renderState.height", "renderState.viewport.height"),
        "backgroundPolicy": compare_scalar(cuda_manifest, webgpu, "renderState.backgroundPolicy.rgb", "renderState.backgroundPolicy.rgb"),
        "worldViewTransform": compare_scalar(cuda_manifest, webgpu, "camera.worldViewTransform", "camera.worldViewTransform"),
        "fullProjTransform": compare_scalar(cuda_manifest, webgpu, "camera.fullProjTransform", "camera.fullProjTransform"),
    }
    direct_src_index_comparison = build_direct_src_index_comparison(cuda_manifest, webgpu)
    semantic_matrix_comparison = build_semantic_matrix_comparison(cuda_manifest, webgpu)
    contradictions = [
        name for name, comparison in field_comparisons.items()
        if comparison.get("cudaValue") is not None
        and comparison.get("webgpuValue") is not None
        and comparison.get("match") is not True
    ]
    cuda_render = get_path(cuda_manifest, "artifacts.render", {})
    old_new_match = None
    old_new_artifacts = None
    if old_cuda_manifest is not None:
        old_new_match = {
            "cameraLabel": compare_scalar(old_cuda_manifest, cuda_manifest, "renderState.cameraLabel", "renderState.cameraLabel"),
            "frameNumber": compare_scalar(old_cuda_manifest, cuda_manifest, "renderState.frameNumber", "renderState.frameNumber"),
            "viewId": compare_scalar(old_cuda_manifest, cuda_manifest, "renderState.viewId", "renderState.viewId"),
            "timestamp": compare_scalar(old_cuda_manifest, cuda_manifest, "renderState.timestamp", "renderState.timestamp"),
            "checkpointSha256": compare_scalar(old_cuda_manifest, cuda_manifest, "lineage.checkpoint.sha256", "lineage.checkpoint.sha256"),
        }
        old_new_artifacts = {
            "oldManifest": file_identity(old_cuda_manifest_path) if old_cuda_manifest_path else None,
            "newManifest": file_identity(cuda_manifest_path),
            "oldRender": get_path(old_cuda_manifest, "artifacts.render", {}),
            "newRender": cuda_render,
        }

    blocked_reasons = []
    failed_predicates = []
    if cuda_unknowns:
        blocked_reasons.append("cuda-reference-manifest-has-unknown-critical-fields")
        failed_predicates.append("cuda-reference-critical-fields-known")
    if webgpu_unknowns:
        blocked_reasons.append("webgpu-render-state-manifest-has-missing-or-unknown-critical-fields")
        failed_predicates.append("webgpu-render-state-critical-fields-known")
    if contradictions:
        blocked_reasons.append("cuda-webgpu-render-state-contradictory-fields")
        failed_predicates.append("cuda-webgpu-render-state-fields-match")
    if not has_webgpu:
        blocked_reasons.append("missing-webgpu-render-state-manifest")
        failed_predicates.append("webgpu-render-state-manifest-present")
    direct_evidence = get_path(
        cuda_manifest,
        "imageSpaceConvention.directRasterizerScreenCoordinateEvidence",
        {},
    )
    if not direct_evidence or direct_evidence.get("available") is not True:
        blocked_reasons.append("cuda-direct-rasterizer-screen-coordinate-evidence-missing")
        failed_predicates.append("cuda-direct-rasterizer-screen-coordinate-evidence-present")
    if has_webgpu and direct_src_index_comparison.get("decision") != "ready":
        blocked_reasons.append(direct_src_index_comparison.get("blockedReason") or "direct-src-index-comparison-not-ready")
        failed_predicates.append("direct-cuda-webgpu-src-index-comparison-ready")
    src_index_semantics = get_path(webgpu, "srcIndexSemantics", {})
    if has_webgpu and isinstance(src_index_semantics, dict):
        mapping_decision = src_index_semantics.get("mappingDecision")
        if mapping_decision and mapping_decision != "ready":
            blocked_reasons.append(mapping_decision)
            failed_predicates.append("src-index-lineage-or-mapping-known")
    direct_temporal_summary = direct_src_index_comparison.get("temporalStageComparisonSummary", {})
    temporal_stage_ready = (
        direct_src_index_comparison.get("commonSrcIndexCount", 0) > 0
        and bool(direct_temporal_summary)
        and not direct_src_index_comparison.get("temporalEvidenceMissing")
    )
    source_stage = get_path(
        direct_src_index_comparison,
        "stageErrorSummary.source-world-position",
        {},
    )
    source_world_mapping_ready = (
        source_stage.get("availableCount") == len(get_path(direct_src_index_comparison, "canonicalComparisonIndexSet.selectedSrcIndices", []))
        and source_stage.get("mismatchCount") == 0
        and source_stage.get("missingCount") == 0
        and source_stage.get("availableCount", 0) > 0
    )
    fix7_ready = (
        has_webgpu
        and source_world_mapping_ready
        and temporal_stage_ready
    )
    fixed_time_capture_state = get_path(webgpu, "renderState.fixedTimeCaptureState", {})
    fix8_ready = (
        fix7_ready
        and fixed_time_capture_state.get("artifactsShareFixedTime") is True
        and fixed_time_capture_state.get("artifactsShareFixedFrame") is True
        and fixed_time_capture_state.get("probeStateMutationDetected") is False
    )
    motion_delta_summary = direct_src_index_comparison.get("motionDeltaInternalStageComparisonSummary", {})
    final_motion_delta_stage = motion_delta_summary.get("final-temporal-motion-delta", {})
    post_world_stage = motion_delta_summary.get("pre-world-plus-delta", {})
    motion_delta_stage_ready = (
        fix8_ready
        and bool(motion_delta_summary)
        and not direct_src_index_comparison.get("motionDeltaEvidenceMissing")
    )
    motion_delta_match_ready = (
        motion_delta_stage_ready
        and final_motion_delta_stage.get("availableCount", 0) > 0
        and final_motion_delta_stage.get("mismatchCount") == 0
        and final_motion_delta_stage.get("missingCount") == 0
        and post_world_stage.get("availableCount", 0) > 0
        and post_world_stage.get("mismatchCount") == 0
        and post_world_stage.get("missingCount") == 0
    )
    fix9_ready = motion_delta_stage_ready
    projection_summary = direct_src_index_comparison.get("projectionCanonicalStageComparisonSummary", {})
    radius_summary = direct_src_index_comparison.get("radiusFootprintStageComparisonSummary", {})
    rotation_contract = get_path(webgpu, "rotationInputPackingContract", {})
    initial_presentation = get_path(webgpu, "initialProductionPresentation", {})
    resolved_cuda_manifest = {
        **file_identity(cuda_manifest_path),
        "schemaVersion": cuda_manifest.get("schemaVersion"),
        "manifestKind": cuda_manifest.get("manifestKind"),
        "loadStatus": "loaded",
        "parseStatus": "parsed-json-object",
        "productionEvidenceSource": get_path(
            cuda_manifest,
            "imageSpaceConvention.directRasterizerScreenCoordinateEvidence.actualEvidenceSource",
            "cuda-reference-render-state-manifest",
        ),
    }
    temporal_source_summary = {
        name: value
        for name, value in motion_delta_summary.items()
        if name.startswith("temporal-source-")
    }
    temporal_source_ready = (
        bool(temporal_source_summary)
        and all(item.get("availableCount", 0) > 0 for item in temporal_source_summary.values())
        and all(item.get("missingCount", 0) == 0 for item in temporal_source_summary.values())
    )
    projection_evidence_ready = (
        bool(projection_summary)
        and any(item.get("availableCount", 0) > 0 for item in projection_summary.values())
    )
    radius_evidence_ready = (
        bool(radius_summary)
        and any(item.get("availableCount", 0) > 0 for item in radius_summary.values())
    )
    fix10_core_ready = (
        fix9_ready
        and resolved_cuda_manifest["exists"] is True
        and rotation_contract.get("contractReady") is True
        and temporal_source_ready
        and projection_evidence_ready
        and radius_evidence_ready
        and isinstance(initial_presentation, dict)
        and bool(initial_presentation)
    )
    initial_presentation_classification = initial_presentation.get("classification")
    decisive_initial_presentation_classifications = {
        "url-only-initial-production-presentation-succeeded",
        "initial-schedule-request-not-observed",
        "initial-scheduled-production-frame-not-completed",
        "initial-production-frame-completed-without-compositor-output",
        "initial-compositor-output-not-presented-to-current-texture",
        "initial-presentation-completed-but-black",
        "capture-command-or-retry-generated-first-observed-presentation",
        "initial-browser-visible-final-presentation-evidence-unavailable",
        "initial-browser-visible-pixel-evidence-unavailable",
    }
    pre_capture_snapshot_ready = (
        initial_presentation.get("preCaptureSnapshotAvailable") is True
        and initial_presentation.get("preCaptureSnapshotReadOnly") is True
        and initial_presentation.get("preCaptureSnapshotCapturedBeforeMutation") is True
    )
    capture_request_identity = initial_presentation.get("captureRequestIdentity")
    initial_request_identity = initial_presentation.get("initialRequestIdentity")
    initial_capture_request_separated = (
        isinstance(capture_request_identity, str)
        and bool(capture_request_identity)
        and capture_request_identity != initial_request_identity
    )
    initial_generation = initial_presentation.get("initialProductionGeneration")
    capture_generation = initial_presentation.get("captureProductionGeneration")
    initial_capture_generation_separation_observed = (
        initial_presentation.get("initialAndCaptureGenerationSeparated") is True
        if initial_generation is not None
        else (
            capture_generation is not None
            and initial_presentation_classification
            in decisive_initial_presentation_classifications
            and initial_presentation_classification
            != "url-only-initial-production-presentation-succeeded"
        )
    )
    capture_frame_evidence_ready = (
        initial_presentation.get("captureFreshGenerationObserved") is True
        and capture_generation is not None
        and initial_presentation.get("captureCompositorGeneration") is not None
        and initial_presentation.get("capturePresentedGeneration") is not None
        and isinstance(initial_presentation.get("captureProductionFrameIdentity"), dict)
        and bool(initial_presentation.get("captureProductionFrameIdentity"))
        and isinstance(initial_presentation.get("captureArtifactFrameIdentity"), dict)
        and bool(initial_presentation.get("captureArtifactFrameIdentity"))
        and initial_presentation.get("captureArtifactMatchesCaptureProductionFrame") is True
    )
    initial_observation_decisive = (
        initial_presentation_classification
        in decisive_initial_presentation_classifications
    )
    capture_dependency_decisive = (
        initial_presentation.get("captureCommandDependencyRemaining")
        in (True, False)
    )
    runtime_behavior_preserved = (
        initial_presentation.get("runtimeBehaviorChanged") is False
    )
    initial_logical_presentation_ready = (
        initial_presentation.get("urlOnlyLogicalPresentationResult") is True
    )
    initial_runtime_pixel_backed_ready = (
        initial_presentation.get("browserVisibleFinalPresentationKnown") is True
        and initial_presentation.get("urlOnlyPixelBackedPresentationResult") is True
    )
    initial_url_presentation_ready = (
        initial_presentation_classification
        == "url-only-initial-production-presentation-succeeded"
        and initial_presentation.get("urlLoadAloneGaussianVisible") is True
        and initial_presentation.get("captureCommandDependencyRemaining") is False
        and initial_logical_presentation_ready
        and initial_runtime_pixel_backed_ready
    )
    capture_logical_presentation_ready = (
        initial_presentation.get("captureLogicalPresentationResult") is True
    )
    capture_runtime_pixel_backed_ready = (
        initial_presentation.get("captureBrowserVisibleFinalPresentationKnown") is True
        and initial_presentation.get("capturePixelBackedPresentationResult") is True
    )
    capture_blob_identity_ready = (
        isinstance(initial_presentation.get("captureBlobIdentity"), dict)
        and bool(initial_presentation.get("captureBlobIdentity", {}).get("sha256"))
    )
    encoded_png_blob_pixel_result = get_path(
        initial_presentation,
        "encodedPngPixelEvidence.pixelClassification",
    )
    encoded_png_blob_pixel_ready = encoded_png_blob_pixel_result in {"black", "nonblank"}
    fix10_fix1_ready = (
        fix10_core_ready
        and pre_capture_snapshot_ready
        and initial_capture_request_separated
        and initial_capture_generation_separation_observed
        and capture_frame_evidence_ready
        and initial_observation_decisive
        and capture_dependency_decisive
        and runtime_behavior_preserved
        and initial_url_presentation_ready
    )
    fix10_ready = fix10_fix1_ready and initial_url_presentation_ready
    fix10_fix2_runtime_evidence_ready = (
        initial_logical_presentation_ready
        and initial_runtime_pixel_backed_ready
        and capture_logical_presentation_ready
        and capture_runtime_pixel_backed_ready
        and capture_blob_identity_ready
        and encoded_png_blob_pixel_ready
    )
    final_canvas_evidence = initial_presentation.get(
        "finalCanvasPresentationEvidence", {}
    )
    url_final_boundary = final_canvas_evidence.get("urlOnlyBoundary", {})
    capture_final_boundary = final_canvas_evidence.get("captureBoundary", {})
    independent_presentation_predicates = final_canvas_evidence.get(
        "independentPredicates", {}
    )
    alpha_normalization_evidence = initial_presentation.get(
        "alphaNormalizationEvidence", {}
    )

    def boundary_contract_ready(boundary: Dict[str, Any], *, capture: bool) -> bool:
        browser_result = boundary.get("browserVisibleResult")
        classification = boundary.get("classification")
        classified = (
            browser_result in (True, False)
            or (
                browser_result is None
                and isinstance(classification, str)
                and classification.startswith("unknown-")
                and bool(boundary.get("unknownOrBlockedReason"))
            )
        )
        final_event_identified = bool(boundary.get("finalCanvasEventIdentity"))
        final_write_absence_identified = (
            boundary.get("successfulCanvasWriteCount") == 0
            and classification == "unknown-no-successful-final-canvas-write"
            and bool(boundary.get("unknownOrBlockedReason"))
        )
        final_state_uniquely_classified = (
            final_event_identified or final_write_absence_identified
        )
        identity_classified = (
            boundary.get("finalSourceIdentityKnown") in (True, False)
            and boundary.get("finalSourceIdentityMatchesExpected") in (True, False)
        )
        success_identity_ready = (
            browser_result is not True
            or (
                boundary.get("finalSourceIdentityKnown") is True
                and boundary.get("finalSourceIdentityMatchesExpected") is True
            )
        )
        final_pixel_result_classified = boundary.get("finalSourcePixelResult") in {
            "black",
            "nonblank",
            "unknown",
        }
        read_only = get_path(
            boundary,
            "readOnlySnapshot.getterMutatesRuntimeState",
        ) is False
        passive_capture_observation = (
            not capture
            or (
                (
                    get_path(
                        boundary,
                        "passiveSteadyStateObservation.requestedProductionFrame",
                    ) is False
                    and get_path(
                        boundary,
                        "passiveSteadyStateObservation.scheduledRender",
                    ) is False
                    and get_path(
                        boundary,
                        "passiveSteadyStateObservation.performedGpuReadback",
                    ) is False
                    and get_path(
                        boundary,
                        "passiveSteadyStateObservation.wroteCanvas",
                    ) is False
                )
                or (
                    get_path(
                        boundary,
                        "passiveQuiescenceObservation.requestedProductionFrame",
                    ) is False
                    and get_path(
                        boundary,
                        "passiveQuiescenceObservation.scheduledRender",
                    ) is False
                    and get_path(
                        boundary,
                        "passiveQuiescenceObservation.performedGpuReadback",
                    ) is False
                    and get_path(
                        boundary,
                        "passiveQuiescenceObservation.wroteCanvas",
                    ) is False
                )
            )
        )
        return (
            boundary.get("schemaVersion")
            == "phase3-final-canvas-presentation-boundary-v1"
            and bool(boundary.get("boundaryIdentity"))
            and final_state_uniquely_classified
            and identity_classified
            and success_identity_ready
            and final_pixel_result_classified
            and boundary.get("laterOverwriteDetected") in (True, False)
            and boundary.get("steadyStateConfirmed") in (True, False)
            and read_only
            and passive_capture_observation
            and classified
        )

    url_final_boundary_ready = boundary_contract_ready(
        url_final_boundary,
        capture=False,
    )
    capture_final_boundary_ready = boundary_contract_ready(
        capture_final_boundary,
        capture=True,
    )
    alpha_normalization_ready = (
        alpha_normalization_evidence.get("appliedToOpaqueWebGpuProductionPngCaptureOnly")
        is True
        and alpha_normalization_evidence.get("genericTransparentPngCaptureUnaffected")
        is True
        and alpha_normalization_evidence.get("rgbInvariant") is True
        and alpha_normalization_evidence.get("alphaOnlyChanged") is True
        and isinstance(
            alpha_normalization_evidence.get("preNormalizationSourceIdentity"),
            dict,
        )
        and isinstance(
            alpha_normalization_evidence.get("postNormalizationBlobIdentity"),
            dict,
        )
    )
    fix10_fix3_ready = (
        final_canvas_evidence.get("boundariesSeparated") is True
        and url_final_boundary_ready
        and capture_final_boundary_ready
        and all(
            independent_presentation_predicates.get(name) is True
            for name in (
                "productionOutput",
                "finalBrowserPresentation",
                "encodedPngBlob",
                "savedPngFile",
            )
        )
        and alpha_normalization_ready
        and initial_presentation.get("runtimeError") is None
        and final_canvas_evidence.get("productionRuntimeBehaviorChanged") is False
    )
    fix10_fix3_blocked_reasons = [
        reason
        for reason, failed in [
            (
                "url-and-capture-final-presentation-boundaries-not-separated",
                final_canvas_evidence.get("boundariesSeparated") is not True,
            ),
            ("url-only-final-canvas-boundary-not-ready", not url_final_boundary_ready),
            ("capture-final-canvas-boundary-not-ready", not capture_final_boundary_ready),
            (
                "presentation-evidence-independent-predicates-not-preserved",
                not all(
                    independent_presentation_predicates.get(name) is True
                    for name in (
                        "productionOutput",
                        "finalBrowserPresentation",
                        "encodedPngBlob",
                        "savedPngFile",
                    )
                ),
            ),
            ("alpha-normalization-scope-or-rgb-invariance-not-ready", not alpha_normalization_ready),
            (
                "final-presentation-trace-modified-production-runtime",
                final_canvas_evidence.get("productionRuntimeBehaviorChanged") is not False,
            ),
            (
                "final-presentation-runtime-error-detected",
                initial_presentation.get("runtimeError") is not None,
            ),
        ]
        if failed
    ]
    fix10_fix3_capture = (
        final_canvas_evidence.get("schemaVersion")
        == "phase3-final-canvas-presentation-evidence-v1"
        and isinstance(url_final_boundary, dict)
        and isinstance(capture_final_boundary, dict)
    )
    synchronous_fence = initial_presentation.get(
        "synchronousCommandStartFence", {}
    )
    causal_trace = initial_presentation.get("commandEraCausalTrace", {})
    initial_identity_chain = initial_presentation.get(
        "initialRequestPresentationIdentityChain", {}
    )
    url_coverage = initial_presentation.get(
        "urlOnlyCanvasWritePathCoverage", {}
    )
    capture_coverage = initial_presentation.get(
        "captureCanvasWritePathCoverage", {}
    )
    url_quiescence = initial_presentation.get("urlOnlyQuiescenceEvidence", {})
    capture_quiescence = initial_presentation.get(
        "captureQuiescenceEvidence", {}
    )
    fixed_camera_contract = get_path(
        webgpu, "renderState.fixedReferenceCameraContract", {}
    )
    fix10_fix4_capture = (
        isinstance(synchronous_fence, dict)
        and synchronous_fence.get("schemaVersion")
        == "phase3-synchronous-command-start-fence-v1"
    )
    fix10_fix4_ready = (
        fix10_fix4_capture
        and synchronous_fence.get("synchronousReadOnlyFence") is True
        and synchronous_fence.get("capturedInSingleJavaScriptTurn") is True
        and synchronous_fence.get("asyncOperationBeforeFence") is False
        and synchronous_fence.get("mutationBeforeFence") is False
        and get_path(
            synchronous_fence,
            "canonicalBoundary.finalCanvasEventSequence",
        ) is not None
        and causal_trace.get("schemaVersion")
        == "phase3-command-era-causal-trace-v1"
        and isinstance(causal_trace.get("events"), list)
        and url_coverage.get("coverageComplete") is True
        and url_coverage.get("unregisteredWritePathCount") == 0
        and capture_coverage.get("coverageComplete") is True
        and capture_coverage.get("unregisteredWritePathCount") == 0
        and url_quiescence.get("quiescent") in (True, False)
        and capture_quiescence.get("quiescent") in (True, False)
        and initial_identity_chain.get("requestToProductionIdentityMatches")
        in (True, False)
        and initial_identity_chain.get("requestToFinalCanvasIdentityMatches")
        in (True, False)
        and fixed_camera_contract.get("fixedReferenceCameraMode")
        == initial_presentation.get("fixedCameraApplied")
        and fixed_camera_contract.get("cameraMathChangedByContractNormalization")
        is False
        and get_path(
            initial_presentation,
            "startupRuntimeCorrection.stepNameDependent",
        ) is False
    )
    fix10_fix4_blocked_reasons = [
        reason
        for reason, failed in [
            ("synchronous-command-start-fence-not-ready", not fix10_fix4_capture),
            (
                "async-or-mutation-before-command-fence-not-excluded",
                synchronous_fence.get("asyncOperationBeforeFence") is not False
                or synchronous_fence.get("mutationBeforeFence") is not False,
            ),
            (
                "command-era-causal-trace-not-ready",
                causal_trace.get("schemaVersion")
                != "phase3-command-era-causal-trace-v1",
            ),
            (
                "current-texture-write-path-coverage-incomplete",
                url_coverage.get("coverageComplete") is not True
                or capture_coverage.get("coverageComplete") is not True,
            ),
            (
                "quiescence-result-not-classified",
                url_quiescence.get("quiescent") not in (True, False)
                or capture_quiescence.get("quiescent") not in (True, False),
            ),
            (
                "initial-request-presentation-identity-chain-not-classified",
                initial_identity_chain.get("requestToProductionIdentityMatches")
                not in (True, False)
                or initial_identity_chain.get("requestToFinalCanvasIdentityMatches")
                not in (True, False),
            ),
            (
                "fixed-reference-camera-contract-inconsistent",
                fixed_camera_contract.get("fixedReferenceCameraMode")
                != initial_presentation.get("fixedCameraApplied"),
            ),
        ]
        if failed
    ]
    fix10_fix1_blocked_reasons = [
        reason
        for reason, failed in [
            ("fix10-core-evidence-not-ready", not fix10_core_ready),
            ("pre-capture-read-only-snapshot-not-ready", not pre_capture_snapshot_ready),
            ("initial-and-capture-request-not-separated", not initial_capture_request_separated),
            (
                "initial-and-capture-generation-separation-not-observed",
                not initial_capture_generation_separation_observed,
            ),
            ("fresh-capture-frame-evidence-not-ready", not capture_frame_evidence_ready),
            ("initial-presentation-cause-classification-not-decisive", not initial_observation_decisive),
            ("capture-command-dependency-not-decisive", not capture_dependency_decisive),
            ("production-runtime-behavior-changed-without-runtime-defect-evidence", not runtime_behavior_preserved),
            ("url-only-browser-visible-pixel-evidence-not-ready", not initial_runtime_pixel_backed_ready),
        ]
        if failed
    ]
    comparison_failed_predicates = [
        *failed_predicates,
        *fix10_fix1_blocked_reasons,
        *(
            []
            if initial_url_presentation_ready
            else ["url-only-initial-production-presentation-not-ready"]
        ),
        *(
            fix10_fix3_blocked_reasons
            if fix10_fix3_capture
            else []
        ),
        *(fix10_fix4_blocked_reasons if fix10_fix4_capture else []),
    ]
    comparison_blocked_reasons = [
        *blocked_reasons,
        *fix10_fix1_blocked_reasons,
        *(
            []
            if initial_url_presentation_ready
            else ["url-only-initial-production-presentation-not-ready"]
        ),
        *(
            fix10_fix3_blocked_reasons
            if fix10_fix3_capture
            else []
        ),
        *(fix10_fix4_blocked_reasons if fix10_fix4_capture else []),
    ]

    return {
        "schemaVersion": "phase3-step114-render-state-manifest-comparison-v1",
        "comparisonKind": "cuda-webgpu-render-state-manifest-comparison",
        "cudaReferenceProvenanceDecision": "blocked" if cuda_unknowns else "ready",
        "cudaReferenceRegenerationDecision":
            "ready" if old_cuda_manifest is not None else "pending-old-new-regeneration-comparison",
        "renderStateManifestComparisonDecision": "blocked" if comparison_blocked_reasons else "ready",
        "visualComparisonReady": not comparison_blocked_reasons,
        "step114Fix7ImplementationDecision": "ready" if fix7_ready else "blocked",
        "step114Fix7ImplementationBlockedReasons": [
            reason
            for reason, failed in [
                ("missing-webgpu-render-state-manifest", not has_webgpu),
                ("source-world-position-lineage-evidence-not-ready", not source_world_mapping_ready),
                ("temporal-stage-comparison-evidence-missing", not temporal_stage_ready),
            ]
            if failed
        ],
        "step114Fix8ImplementationDecision": "ready" if fix8_ready else "blocked",
        "step114Fix8ImplementationBlockedReasons": [
            reason
            for reason, failed in [
                ("missing-webgpu-render-state-manifest", not has_webgpu),
                ("fix7-lineage-temporal-comparison-not-ready", not fix7_ready),
                (
                    "fixed-time-artifacts-do-not-share-state",
                    fixed_time_capture_state.get("artifactsShareFixedTime") is not True,
                ),
                (
                    "capture-frame-identity-not-shared",
                    fixed_time_capture_state.get("artifactsShareFixedFrame") is not True,
                ),
                (
                    "state-changing-probe-ran-during-step114-capture",
                    fixed_time_capture_state.get("probeStateMutationDetected") is True,
                ),
            ]
            if failed
        ],
        "step114Fix9ImplementationDecision": "ready" if fix9_ready else "blocked",
        "step114Fix9ImplementationBlockedReasons": [
            reason
            for reason, failed in [
                ("fix8-fixed-time-temporal-comparison-not-ready", not fix8_ready),
                ("motion-delta-internal-stage-comparison-evidence-missing", not motion_delta_stage_ready),
            ]
            if failed
        ],
        "step114Fix9ProductionTemporalMotionDeltaMatch": motion_delta_match_ready,
        "step114Fix9ProductionCorrectionApplied": (
            "pending-browser-recapture" if not has_webgpu else "webgpu-conditional-temporal-mean-path-observed-in-manifest"
        ),
        "step114Fix10ImplementationDecision": "ready" if fix10_ready else "blocked",
        "step114Fix10ImplementationBlockedReasons": [
            reason
            for reason, failed in [
                ("fix9-temporal-motion-delta-evidence-not-ready", not fix9_ready),
                ("cuda-reference-manifest-not-resolved", resolved_cuda_manifest["exists"] is not True),
                ("rotation-input-packing-contract-not-ready", rotation_contract.get("contractReady") is not True),
                ("temporal-source-parameter-stage-comparison-not-ready", not temporal_source_ready),
                ("projection-canonical-stage-comparison-missing", not projection_evidence_ready),
                ("radius-footprint-stage-comparison-missing", not radius_evidence_ready),
                ("initial-production-presentation-contract-missing", not isinstance(initial_presentation, dict) or not bool(initial_presentation)),
                ("fix10-fix1-initial-presentation-observation-not-ready", not fix10_fix1_ready),
                ("url-only-initial-production-presentation-not-ready", not initial_url_presentation_ready),
            ]
            if failed
        ],
        "step114Fix10Fix1ImplementationDecision": "ready" if fix10_fix1_ready else "blocked",
        "step114Fix10Fix1ImplementationBlockedReasons": fix10_fix1_blocked_reasons,
        "step114Fix10Fix1InitialPresentationCauseClassification":
            initial_presentation_classification,
        "step114Fix10Fix1PreCaptureSnapshotReady": pre_capture_snapshot_ready,
        "step114Fix10Fix1InitialCaptureRequestSeparated":
            initial_capture_request_separated,
        "step114Fix10Fix1InitialCaptureGenerationSeparated":
            initial_capture_generation_separation_observed,
        "step114Fix10Fix1CaptureFrameEvidenceReady": capture_frame_evidence_ready,
        "step114Fix10Fix1UrlOnlyInitialPresentationReady":
            initial_url_presentation_ready,
        "step114Fix10Fix1RuntimeBehaviorChanged":
            initial_presentation.get("runtimeBehaviorChanged"),
        "step114Fix10Fix2RuntimeEvidenceDecision":
            "ready" if fix10_fix2_runtime_evidence_ready else "blocked",
        "step114Fix10Fix2RuntimeEvidenceBlockedReasons": [
            reason
            for reason, failed in [
                ("url-only-logical-presentation-not-ready", not initial_logical_presentation_ready),
                ("url-only-pixel-backed-presentation-not-ready", not initial_runtime_pixel_backed_ready),
                ("capture-logical-presentation-not-ready", not capture_logical_presentation_ready),
                ("capture-pixel-backed-presentation-not-ready", not capture_runtime_pixel_backed_ready),
                ("capture-blob-identity-not-ready", not capture_blob_identity_ready),
                ("encoded-png-blob-pixel-evidence-not-ready", not encoded_png_blob_pixel_ready),
            ]
            if failed
        ],
        "step114Fix10Fix3ImplementationDecision":
            "ready" if fix10_fix3_ready else "blocked",
        "step114Fix10Fix3ImplementationBlockedReasons":
            fix10_fix3_blocked_reasons,
        "step114Fix10Fix4ImplementationDecision":
            "ready" if fix10_fix4_ready else "blocked",
        "step114Fix10Fix4ImplementationBlockedReasons":
            fix10_fix4_blocked_reasons,
        "synchronousCommandStartFence": synchronous_fence,
        "commandEraCausalTrace": causal_trace,
        "initialRequestPresentationIdentityChain": initial_identity_chain,
        "urlOnlyCanvasWritePathCoverage": url_coverage,
        "captureCanvasWritePathCoverage": capture_coverage,
        "urlOnlyQuiescenceEvidence": url_quiescence,
        "captureQuiescenceEvidence": capture_quiescence,
        "fixedReferenceCameraContract": fixed_camera_contract,
        "urlOnlyFinalBrowserPresentation": url_final_boundary,
        "captureFinalBrowserPresentation": capture_final_boundary,
        "finalCanvasPresentationEvidence": final_canvas_evidence,
        "alphaNormalizationEvidence": alpha_normalization_evidence,
        "urlOnlyLogicalPresentationResult":
            initial_presentation.get("urlOnlyLogicalPresentationResult"),
        "urlOnlyPixelBackedPresentationResult":
            initial_presentation.get("urlOnlyPixelBackedPresentationResult"),
        "captureLogicalPresentationResult":
            initial_presentation.get("captureLogicalPresentationResult"),
        "capturePixelBackedPresentationResult":
            initial_presentation.get("capturePixelBackedPresentationResult"),
        "captureBlobIdentity": initial_presentation.get("captureBlobIdentity"),
        "encodedPngPixelEvidence":
            initial_presentation.get("encodedPngPixelEvidence"),
        "motionDeltaInternalStageComparisonSummary": motion_delta_summary,
        "temporalSourceParameterStageComparisonSummary": temporal_source_summary,
        "firstMotionDeltaMismatchSubstage": direct_src_index_comparison.get("firstMotionDeltaMismatchSubstage"),
        "motionDeltaEvidenceMissing": direct_src_index_comparison.get("motionDeltaEvidenceMissing"),
        "motionDeltaRootCauseClassifications": direct_src_index_comparison.get("motionDeltaRootCauseClassifications"),
        "projectionCanonicalStageComparisonSummary": projection_summary,
        "projectionCanonicalContract": direct_src_index_comparison.get("projectionCanonicalContract"),
        "firstProjectionMismatchSubstage": direct_src_index_comparison.get("firstProjectionMismatchSubstage"),
        "projectionRootCauseClassification": direct_src_index_comparison.get("projectionRootCauseClassification"),
        "radiusFootprintStageComparisonSummary": radius_summary,
        "rotationInputPackingContract": rotation_contract,
        "initialProductionPresentation": initial_presentation,
        "step114Fix10ProductionFixApplied": False,
        "step114Fix10ProductionFixDescription":
            "no production camera/projection/Y-flip/radius math changed; fix10 records canonical evidence and vocabulary classification only",
        "step114Fix10BeforeAfterComparison": {
            "before": "fix9 first direct stage mismatch was clip-ndc with temporal motion delta matched",
            "after": {
                "firstMotionDeltaMismatchSubstage": direct_src_index_comparison.get("firstMotionDeltaMismatchSubstage"),
                "firstProjectionMismatchSubstage": direct_src_index_comparison.get("firstProjectionMismatchSubstage"),
                "firstDirectStageMismatch": direct_src_index_comparison.get("firstMismatchStage"),
            },
        },
        "fixedTimeCaptureState": fixed_time_capture_state,
        "failedPredicates": comparison_failed_predicates,
        "blockedReasons": comparison_blocked_reasons,
        "cudaReferenceManifest": resolved_cuda_manifest,
        "webgpuRenderStateManifest": file_identity(webgpu_manifest_path) if webgpu_manifest_path else None,
        "oldNewCudaReferenceMatch": old_new_match,
        "oldNewCudaReferenceArtifacts": old_new_artifacts,
        "cudaReference": {
            "runId": cuda_manifest.get("runId"),
            "renderSource": cuda_render,
            "checkpoint": get_path(cuda_manifest, "lineage.checkpoint", {}),
            "datasetRoot": get_path(cuda_manifest, "lineage.datasetRoot"),
            "reuseLegacyTrainingReport": get_path(cuda_manifest, "lineage.reuseLegacyTrainingReport"),
            "cameraLabel": get_path(cuda_manifest, "renderState.cameraLabel"),
            "frameNumber": get_path(cuda_manifest, "renderState.frameNumber"),
            "viewId": get_path(cuda_manifest, "renderState.viewId"),
            "timestamp": get_path(cuda_manifest, "renderState.timestamp"),
            "gaussianCount": get_path(cuda_manifest, "renderState.gaussianCount"),
            "imageSpaceConvention": get_path(cuda_manifest, "imageSpaceConvention", {}),
        },
        "webgpu": {
            "cameraLabel": get_path(webgpu, "renderState.cameraLabel"),
            "frameNumber": get_path(webgpu, "renderState.frameNumber"),
            "viewId": get_path(webgpu, "renderState.viewId"),
            "datasetTime": get_path(webgpu, "renderState.datasetTime"),
            "fixedReferenceCameraMode": get_path(webgpu, "renderState.fixedReferenceCameraMode"),
            "webgpuCameraConstantsSource": get_path(webgpu, "renderState.webgpuCameraConstantsSource"),
            "gaussianAsset": get_path(webgpu, "lineage.gaussianAsset", {}),
            "imageSpaceConvention": get_path(webgpu, "imageSpaceConvention", {}),
            "presentationAndCapture": get_path(webgpu, "presentationAndCapture", {}),
        },
        "fieldComparisons": field_comparisons,
        "semanticMatrixComparison": semantic_matrix_comparison,
        "srcIndexSemantics": src_index_semantics,
        "unknownCriticalFields": {
            "cudaReference": cuda_unknowns,
            "webgpu": webgpu_unknowns,
        },
        "contradictoryFields": contradictions,
        "directRasterizerScreenCoordinateEvidence": direct_evidence,
        "directCudaWebGpuSrcIndexComparison": direct_src_index_comparison,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Step114 render-state manifests.")
    parser.add_argument("--cuda-manifest", required=True)
    parser.add_argument("--webgpu-manifest", default=None)
    parser.add_argument("--old-cuda-manifest", default=None)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    cuda_manifest_path = Path(args.cuda_manifest)
    webgpu_manifest_path = Path(args.webgpu_manifest) if args.webgpu_manifest else None
    old_cuda_manifest_path = Path(args.old_cuda_manifest) if args.old_cuda_manifest else None
    summary = build_comparison(
        cuda_manifest=load_json(cuda_manifest_path),
        webgpu_manifest=load_json(webgpu_manifest_path) if webgpu_manifest_path and webgpu_manifest_path.exists() else None,
        cuda_manifest_path=cuda_manifest_path,
        webgpu_manifest_path=webgpu_manifest_path,
        old_cuda_manifest=load_json(old_cuda_manifest_path) if old_cuda_manifest_path and old_cuda_manifest_path.exists() else None,
        old_cuda_manifest_path=old_cuda_manifest_path,
    )
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
