#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCHEMA_VERSION = "phase3-capture-command-contract-v1"
EXPECTED_RUNTIME_FIELDS = (
    "requestedRuntime",
    "effectiveDisplayRuntime",
    "backendMode",
    "backendImplementation",
    "canvasPresentationEnabled",
    "viewerLoopHookEnabled",
)
ALLOWED_VIEWER_DEBUG_APIS = {
    "getSynchronousCommandStartFence",
    "validateExpectedProductionRuntimeContract",
    "waitForViewerDebugDataReady",
    "scheduleRender",
    "getInitialProductionPresentationSnapshot",
    "getLastRenderResult",
    "waitForFinalCanvasPresentationQuiescence",
    "captureWebGpuVisibleRecordDryRunDebug",
    "downloadJsonDebug",
    "captureGpuCandidateRuntimeSummaryDebug",
    "saveCurrentCanvasPng",
}


def mask_javascript_comments(source: str) -> str:
    result = list(source)
    index = 0
    state = "code"
    quote = ""
    while index < len(source):
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if char in ("'", '"', "`"):
                state = "string"
                quote = char
            elif char == "/" and next_char == "/":
                result[index] = result[index + 1] = " "
                index += 1
                state = "line-comment"
            elif char == "/" and next_char == "*":
                result[index] = result[index + 1] = " "
                index += 1
                state = "block-comment"
        elif state == "string":
            if char == "\\":
                index += 1
            elif char == quote:
                state = "code"
        elif state == "line-comment":
            if char == "\n":
                state = "code"
            else:
                result[index] = " "
        elif state == "block-comment":
            if char == "*" and next_char == "/":
                result[index] = result[index + 1] = " "
                index += 1
                state = "code"
            elif char != "\n":
                result[index] = " "
        index += 1
    return "".join(result)


def find_positions(code: str, pattern: str, flags: int = 0) -> List[int]:
    return [match.start() for match in re.finditer(pattern, code, flags)]


def extract_unique_property(code: str, name: str) -> Tuple[Optional[str], List[str]]:
    values = [
        match.group(2)
        for match in re.finditer(
            rf"\b{re.escape(name)}\s*:\s*(['\"])([^'\"]+)\1",
            code,
        )
    ]
    unique_values = list(dict.fromkeys(values))
    return (unique_values[0] if len(unique_values) == 1 else None), unique_values


def find_artifact_positions(code: str, suffix: str) -> List[Tuple[int, str]]:
    pattern = re.compile(rf"(['\"])([^'\"]*{re.escape(suffix)})\1")
    return [(match.start(), match.group(2)) for match in pattern.finditer(code)]


def first_position(positions: Iterable[int]) -> Optional[int]:
    values = list(positions)
    return min(values) if values else None


def build_capture_command_contract(command_path: Path) -> Dict[str, Any]:
    source = command_path.read_text(encoding="utf-8")
    code = mask_javascript_comments(source)
    verification_errors: List[str] = []

    policy, policy_values = extract_unique_property(code, "policy")
    phase_step, phase_step_values = extract_unique_property(code, "phaseStep")
    comparison_mode, comparison_mode_values = extract_unique_property(
        code, "comparisonMode"
    )
    for field_name, value, values in (
        ("policy", policy, policy_values),
        ("phaseStep", phase_step, phase_step_values),
        ("comparisonMode", comparison_mode, comparison_mode_values),
    ):
        if value is None:
            verification_errors.append(
                f"{field_name}-missing-or-inconsistent:{values!r}"
            )

    runtime_preflight: Dict[str, Any] = {
        "expected": None,
        "validatorCallCount": 0,
        "expectedContractComplete": False,
    }
    runtime_contract_match = re.search(
        r"\bvar\s+genericExpectedProductionRuntimeContract\s*=\s*"
        r"(\{[^\n;]*\})\s*;",
        code,
    )
    if runtime_contract_match:
        try:
            expected_contract = json.loads(runtime_contract_match.group(1))
            runtime_preflight["expected"] = expected_contract
            runtime_preflight["expectedContractComplete"] = all(
                field in expected_contract and expected_contract[field] is not None
                for field in EXPECTED_RUNTIME_FIELDS
            )
        except json.JSONDecodeError as error:
            verification_errors.append(f"runtime-preflight-json-invalid:{error}")
    else:
        verification_errors.append("runtime-preflight-expected-contract-missing")
    runtime_preflight_positions = find_positions(
        code,
        r"window\.gpuViewerDebug\.validateExpectedProductionRuntimeContract\s*\(",
    )
    runtime_preflight["validatorCallCount"] = len(runtime_preflight_positions)

    fresh_request_positions = find_positions(
        code,
        r"genericFreshProductionRequest\s*=\s*await\s+"
        r"window\.gpuViewerDebug\.scheduleRender\s*\(",
    )
    production_schedule_positions = find_positions(
        code, r"window\.gpuViewerDebug\.scheduleRender\s*\("
    )
    force_update_positions = find_positions(
        code, r"\bforceProductionUpdate\s*:\s*true\b"
    )
    diagnostic_positions = find_positions(
        code,
        r"window\.gpuViewerDebug\.captureWebGpuVisibleRecordDryRunDebug\s*\(",
    )
    runtime_summary_positions = find_positions(
        code,
        r"window\.gpuViewerDebug\.captureGpuCandidateRuntimeSummaryDebug\s*\(",
    )
    png_capture_positions = find_positions(
        code, r"window\.gpuViewerDebug\.saveCurrentCanvasPng\s*\("
    )
    png_download_positions = find_positions(
        code, r"genericProductionPngDownloadLink\.click\s*\("
    )
    completion_fence_positions = find_positions(
        code,
        r"genericFreshProductionCaptureLifecycle\.productionCompletionFence\s*=",
    )

    artifact_markers = {
        "diagnostic-result-json": find_artifact_positions(
            code, "_webgpu_visible_record_dryrun_compare.json"
        ),
        "diagnostic-detail-json": find_artifact_positions(
            code, "_webgpu_visible_record_lineage.json"
        ),
        "diagnostic-status-json": find_artifact_positions(
            code, "_webgpu_visible_record_dryrun_capture_status.json"
        ),
        "runtime-summary-json": find_artifact_positions(
            code, "_gpu_candidate_runtime_summary.json"
        ),
        "limited-draw-json": find_artifact_positions(
            code, "_limited_draw_summary.json"
        ),
        "png-status-json": find_artifact_positions(
            code, "_png_capture_status.json"
        ),
    }
    stage_positions: Dict[str, Optional[int]] = {
        "runtime-preflight": first_position(runtime_preflight_positions),
        "fresh-production-request": first_position(fresh_request_positions),
        "production-completion-fence": first_position(completion_fence_positions),
        "diagnostic-compute-readback": first_position(diagnostic_positions),
        "diagnostic-result-json": first_position(
            position for position, _ in artifact_markers["diagnostic-result-json"]
        ),
        "diagnostic-detail-json": first_position(
            position for position, _ in artifact_markers["diagnostic-detail-json"]
        ),
        "diagnostic-status-json": first_position(
            position for position, _ in artifact_markers["diagnostic-status-json"]
        ),
        "runtime-summary-json": first_position(
            position for position, _ in artifact_markers["runtime-summary-json"]
        ),
        "limited-draw-json": first_position(
            position for position, _ in artifact_markers["limited-draw-json"]
        ),
        "png-capture": first_position(png_capture_positions),
        "png-status-json": first_position(
            position for position, _ in artifact_markers["png-status-json"]
        ),
        "png": first_position(png_download_positions),
    }
    ordered_stage_names = [
        name for name, position in stage_positions.items() if position is not None
    ]
    stages = [
        {"name": name, "position": stage_positions[name]}
        for name in ordered_stage_names
    ]
    required_stage_names = [
        name for name in stage_positions if name != "diagnostic-detail-json"
    ]
    missing_stages = [
        name for name in required_stage_names if stage_positions[name] is None
    ]
    if missing_stages:
        verification_errors.append("ordered-stages-missing:" + ",".join(missing_stages))
    stage_values = [stage_positions[name] for name in ordered_stage_names]
    stages_ordered = (
        not missing_stages
        and all(
            int(stage_values[index]) < int(stage_values[index + 1])
            for index in range(len(stage_values) - 1)
        )
    )
    if not stages_ordered:
        verification_errors.append("ordered-stages-out-of-order")

    png_call_match = re.search(
        r"window\.gpuViewerDebug\.saveCurrentCanvasPng\s*\(\s*\{"
        r"(?P<body>.*?)\}\s*\)",
        code,
        re.DOTALL,
    )
    png_body = png_call_match.group("body") if png_call_match else ""
    png_render_before_capture_false = bool(
        re.search(r"\brenderBeforeCapture\s*:\s*false\b", png_body)
    )
    png_download_deferred = bool(re.search(r"\bdownload\s*:\s*false\b", png_body))
    png_fallback_disabled = bool(
        re.search(r"\bfallbackToCanvasOnCaptureFailure\s*:\s*false\b", png_body)
    )
    production_png_capture_source = bool(
        re.search(
            r"\bcaptureSource\s*:\s*"
            r"['\"]last-valid-webgpu-tile-compositor-output['\"]",
            png_body,
        )
    )
    png_capture_position = first_position(png_capture_positions)
    png_status_position = stage_positions["png-status-json"]
    png_position = first_position(png_download_positions)
    code_after_png_capture = (
        code[png_capture_position:] if png_capture_position is not None else ""
    )
    code_after_png = code[png_position:] if png_position is not None else ""
    png_after_mutation_patterns = {
        "production-schedule": r"window\.gpuViewerDebug\.scheduleRender\s*\(",
        "forced-production-update": r"\bforceProductionUpdate\s*:\s*true\b",
        "render-current-frame": r"\brenderCurrentFrame\s*\(",
        "cleanup-clear-reset-call": (
            r"\b[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*"
            r"(?:cleanup|clear|reset)[A-Za-z_$\d]*\s*\("
        ),
    }
    png_after_mutations = [
        name
        for name, pattern in png_after_mutation_patterns.items()
        if re.search(pattern, code_after_png_capture, re.IGNORECASE)
    ]
    artifact_save_positions = [
        position
        for markers in artifact_markers.values()
        for position, _ in markers
    ] + png_download_positions
    png_last_artifact = (
        png_position is not None
        and artifact_save_positions
        and png_position == max(artifact_save_positions)
        and not re.search(
            r"window\.gpuViewerDebug\.downloadJsonDebug\s*\(", code_after_png
        )
    )
    png_capture_result_retained = all(
        marker in code
        for marker in (
            "var genericProductionPngCaptureResult =",
            "genericProductionPngCaptureResult?.blob ?? null",
            "captureBlobIdentity?.sha256",
            "genericProductionPngCaptureResult.productionOutputGeneration",
            "genericProductionPngCaptureResult.presentedOutputGeneration",
            "genericProductionPngCaptureResult.capturedOutputGeneration",
            "genericProductionPngCaptureResult.staleCaptureDetected",
            "genericProductionPngCaptureResult.captureFreshnessKnown",
            "genericProductionPngCaptureResult.captureFreshnessClassification",
        )
    )
    png_download_uses_captured_blob = all(
        marker in code
        for marker in (
            "URL.createObjectURL(genericProductionPngBlob)",
            "genericProductionPngDownloadLink.href = genericProductionPngObjectUrl",
            "genericProductionPngDownloadLink.download = genericProductionPngFileName",
        )
    )
    png_status_before_download = (
        png_capture_position is not None
        and png_status_position is not None
        and png_position is not None
        and int(png_capture_position) < int(png_status_position) < int(png_position)
    )

    viewer_debug_api_calls = sorted(
        set(
            re.findall(
                r"window\.gpuViewerDebug\.([A-Za-z_$][\w$]*)\s*(?:\?\.)?\(",
                code,
            )
        )
    )
    unexpected_runtime_api_calls = sorted(
        set(viewer_debug_api_calls) - ALLOWED_VIEWER_DEBUG_APIS
    )
    heartbeat_calls = find_positions(
        code,
        r"\b[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*heartbeat"
        r"[A-Za-z_$\d]*\s*\(",
        re.IGNORECASE,
    )
    raf_calls = find_positions(code, r"\brequestAnimationFrame\s*\(")
    continuation_calls = find_positions(
        code,
        r"\b[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*continuation"
        r"[A-Za-z_$\d]*\s*\(",
        re.IGNORECASE,
    )
    cleanup_clear_calls = find_positions(
        code,
        r"\b[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*"
        r"(?:cleanup|clear|reset)[A-Za-z_$\d]*\s*\(",
        re.IGNORECASE,
    )

    predicates = {
        "policyReadFromGeneratedCommand": policy is not None,
        "phaseStepReadFromGeneratedCommand": phase_step is not None,
        "comparisonModeReadFromGeneratedCommand": comparison_mode is not None,
        "runtimePreflightPresent": len(runtime_preflight_positions) == 1,
        "runtimePreflightExpectedContractComplete": runtime_preflight[
            "expectedContractComplete"
        ],
        "freshProductionRequestExactlyOnce": len(fresh_request_positions) == 1,
        "forceProductionUpdateExactlyOnce": len(force_update_positions) == 1,
        "duplicateProductionScheduleAbsent": len(production_schedule_positions) == 1,
        "freshProductionGenerationRequired": all(
            marker in code
            for marker in (
                "genericProductionGenerationIsFresh",
                "genericBaselineProductionGeneration",
                "Number(genericProductionGeneration) >",
            )
        ),
        "requestIdentityPreserved": (
            "genericFreshProductionRequest.requestIdentity" in code
        ),
        "productionSourceRequestIdentityRequired": bool(
            re.search(
                r"productionSourceRequestIdentity\s*===\s*"
                r"genericFreshProductionRequest\.requestIdentity",
                code,
            )
        ),
        "compositorOutputRequired": bool(
            re.search(r"compositorOutputGenerated\s*===\s*true", code)
        ),
        "viewerCanvasPresentationRequired": bool(
            re.search(r"viewerCanvasPresented\s*===\s*true", code)
        ),
        "logicalPresentationRequired": bool(
            re.search(r"logicalPresentationSucceeded\s*===\s*true", code)
        ),
        "finalPresentationBoundaryWaitPresent": bool(
            re.search(
                r"window\.gpuViewerDebug\.waitForFinalCanvasPresentationQuiescence"
                r"\s*\(",
                code,
            )
        ),
        "completionFenceBeforeDiagnostic": (
            first_position(completion_fence_positions) is not None
            and first_position(diagnostic_positions) is not None
            and int(first_position(completion_fence_positions))
            < int(first_position(diagnostic_positions))
        ),
        "stagesOrdered": stages_ordered,
        "canonicalDiagnosticArtifactSelectedFromBundle": all(
            marker in code
            for marker in (
                "webgpuVisibleRecordDiagnosticArtifacts",
                ".canonicalDiagnosticResult",
                ".detailedLineageArtifact",
            )
        ),
        "diagnosticArtifactProvenancePresent": all(
            marker in code
            for marker in (
                "phase3-diagnostic-artifact-provenance-v1",
                "artifactSetIdentity",
                "artifactProvenance",
                "capturePrefix",
                "requestIdentity",
                "productionGeneration",
                "frameIdentity",
            )
        ),
        "detailArtifactBeforeDiagnosticStatus": (
            stage_positions["diagnostic-detail-json"] is None
            or (
                stage_positions["diagnostic-result-json"] is not None
                and stage_positions["diagnostic-status-json"] is not None
                and int(stage_positions["diagnostic-result-json"])
                < int(stage_positions["diagnostic-detail-json"])
                < int(stage_positions["diagnostic-status-json"])
            )
        ),
        "diagnosticBeforeProductionRequestAbsent": (
            first_position(diagnostic_positions) is not None
            and first_position(fresh_request_positions) is not None
            and int(first_position(fresh_request_positions))
            < int(first_position(diagnostic_positions))
        ),
        "pngBeforeDiagnosticAbsent": (
            png_capture_position is not None
            and first_position(diagnostic_positions) is not None
            and int(first_position(diagnostic_positions))
            < int(png_capture_position)
        ),
        "pngIsLastArtifactSave": png_last_artifact,
        "pngRenderBeforeCaptureFalse": png_render_before_capture_false,
        "productionPngCapturePathUsed": production_png_capture_source,
        "pngCaptureDownloadDeferred": png_download_deferred,
        "pngFallbackDisabled": png_fallback_disabled,
        "pngCaptureResultRetained": png_capture_result_retained,
        "pngStatusArtifactPresent": png_status_position is not None,
        "pngStatusBeforePngDownload": png_status_before_download,
        "pngDownloadUsesCapturedBlob": png_download_uses_captured_blob,
        "pngAfterProductionMutationAbsent": not png_after_mutations,
        "stateMutatingReadinessRetryAbsent": (
            not re.search(r"\bretryDefaultScene\s*:\s*true\b", code)
            and bool(re.search(r"\bretryDefaultScene\s*:\s*false\b", code))
        ),
        "heartbeatCallAbsent": not heartbeat_calls,
        "presentOnlyRafCallAbsent": not raf_calls,
        "schedulerContinuationCallAbsent": not continuation_calls,
        "productionCleanupOrClearCallAbsent": not cleanup_clear_calls,
        "unexpectedRuntimeApiCallAbsent": not unexpected_runtime_api_calls,
        "stepSpecificFixedTimePolicyAbsent": (
            "step114FixedTimeCaptureIsolation" not in code
            and "phase3-step114-fixed-time-capture-isolation" not in code
        ),
    }
    for name, passed in predicates.items():
        if passed is not True:
            verification_errors.append(f"predicate-failed:{name}")
    if png_after_mutations:
        verification_errors.append(
            "png-after-production-mutation:" + ",".join(png_after_mutations)
        )
    if unexpected_runtime_api_calls:
        verification_errors.append(
            "unexpected-runtime-api-calls:" + ",".join(unexpected_runtime_api_calls)
        )

    counts = {
        "freshProductionRequest": len(fresh_request_positions),
        "forceProductionUpdateTrue": len(force_update_positions),
        "diagnosticCaptureCall": len(diagnostic_positions),
        "runtimeSummaryCaptureCall": len(runtime_summary_positions),
        "diagnosticDetailArtifactSave": len(
            artifact_markers["diagnostic-detail-json"]
        ),
        "pngCaptureCall": len(png_capture_positions),
        "pngStatusArtifactSave": len(artifact_markers["png-status-json"]),
        "pngSaveCall": len(png_download_positions),
        "productionScheduleCall": len(production_schedule_positions),
    }
    for name in (
        "diagnosticCaptureCall",
        "runtimeSummaryCaptureCall",
        "pngCaptureCall",
        "pngStatusArtifactSave",
        "pngSaveCall",
    ):
        if counts[name] != 1:
            verification_errors.append(f"call-count-invalid:{name}:{counts[name]}")

    return {
        "schemaVersion": SCHEMA_VERSION,
        "sourceCommandPath": str(command_path.resolve()),
        "policy": policy,
        "phaseStep": phase_step,
        "comparisonMode": comparison_mode,
        "runtimePreflight": runtime_preflight,
        "stages": stages,
        "counts": counts,
        "predicates": predicates,
        "runtimeApiCalls": viewer_debug_api_calls,
        "verificationErrors": list(dict.fromkeys(verification_errors)),
        "decision": "ready" if not verification_errors else "blocked",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the static contract of a generated capture command."
    )
    parser.add_argument("--command", required=True, help="Generated JavaScript path.")
    parser.add_argument("--json", required=True, help="Output contract JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command_path = Path(args.command)
    contract = build_capture_command_contract(command_path)
    output_path = Path(args.json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "capture command contract verification "
        f"{contract['decision']}: {command_path}"
    )
    return 0 if contract["decision"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
