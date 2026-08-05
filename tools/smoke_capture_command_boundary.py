#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERIC_CAPTURE_LIFECYCLE = "fresh-production-diagnostic-json-png"
REGRESSION_SCHEMA_VERSION = "phase3-capture-command-boundary-regression-v1"
STEP114_LEGACY_FIXTURE_SHA256 = (
    "68794d066cb3fe180d718bec34a4cdeff5da68435a5ce5f4ce8b0b9a1d4dea1b"
)


def build_generic_capture_command(
    temp_dir: str,
    prefix: str,
    phase_step: str,
    comparison_mode: str,
) -> str:
    output = Path(temp_dir) / f"{prefix}_capture_commands.js"
    subprocess.run(
        [
            "python3",
            str(ROOT / "tools" / "make_capture_commands.py"),
            "--step",
            prefix,
            "--preset",
            "stable",
            "--capture-lifecycle",
            GENERIC_CAPTURE_LIFECYCLE,
            "--phase-step",
            phase_step,
            "--comparison-mode",
            comparison_mode,
            "--include-preamble",
            "true",
            "--include-webgpu-visible-record-dryrun",
            "true",
            "--include-runtime",
            "true",
            "--include-png",
            "true",
            "--expected-runtime",
            "webgpu",
            "--expected-effective-display-runtime",
            "webgpu-production",
            "--expected-webgpu-backend-mode",
            "webgpu-exclusive",
            "--expected-webgpu-backend-implementation",
            "webgpu-tile-compositor-frame-implementation",
            "--expected-webgpu-canvas-presentation",
            "true",
            "--expected-webgpu-viewer-loop-hook",
            "true",
            "--out",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(["node", "--check", str(output)], cwd=ROOT, check=True)
    return output.read_text(encoding="utf-8")


def assert_generic_capture_lifecycle(
    source: str,
    prefix: str,
    phase_step: str,
    comparison_mode: str,
) -> None:
    preflight_index = source.index("var genericProductionRuntimeValidation =")
    readiness_index = source.index("var genericViewerDebugDataReadiness =")
    request_index = source.index(
        "genericFreshProductionRequest = await window.gpuViewerDebug.scheduleRender("
    )
    completion_index = source.index(
        "genericFreshProductionCaptureLifecycle.productionCompletionFence ="
    )
    diagnostic_index = source.index(
        "await window.gpuViewerDebug.captureWebGpuVisibleRecordDryRunDebug("
    )
    diagnostic_json_index = source.index(
        f"'{prefix}_webgpu_visible_record_dryrun_compare.json'"
    )
    diagnostic_status_index = source.index(
        f"'{prefix}_webgpu_visible_record_dryrun_capture_status.json'"
    )
    runtime_json_index = source.index(
        f"'{prefix}_gpu_candidate_runtime_summary.json'"
    )
    png_index = source.index("await window.gpuViewerDebug.saveCurrentCanvasPng(")
    assert (
        preflight_index
        < readiness_index
        < request_index
        < completion_index
        < diagnostic_index
        < diagnostic_json_index
        < diagnostic_status_index
        < runtime_json_index
        < png_index
    )
    assert source.count("window.gpuViewerDebug.scheduleRender(") == 1
    assert source.count("forceProductionUpdate: true") == 1
    assert source.count("freshProductionRequestCount += 1") == 1
    assert source.count("await window.gpuViewerDebug.saveCurrentCanvasPng(") == 1
    assert "retryDefaultScene: false" in source
    assert "retryDefaultScene: true" not in source
    assert "renderBeforeCapture: false" in source[png_index:]
    assert "scheduleRender(" not in source[png_index:]
    assert "forceProductionUpdate: true" not in source[png_index:]
    assert f"phaseStep: '{phase_step}'" in source
    assert f"comparisonMode: '{comparison_mode}'" in source
    for suffix in (
        "_webgpu_visible_record_dryrun_compare.json",
        "_webgpu_visible_record_dryrun_capture_status.json",
        "_gpu_candidate_runtime_summary.json",
        "_limited_draw_summary.json",
        "_canvas.png",
    ):
        assert f"'{prefix}{suffix}'" in source


def run_checks() -> dict:
    with tempfile.TemporaryDirectory() as temp_dir:
        sources = {}
        for fix in ("fix4", "fix5", "fix6", "fix6_fix1"):
            output = Path(temp_dir) / f"capture_{fix}.js"
            command = [
                    "python3",
                    str(ROOT / "tools" / "make_capture_commands.py"),
                    "--step",
                    f"phase3_step114_fix10_{fix}_000151_v13",
                    "--preset",
                    "stable",
                    "--include-preamble",
                    "true",
                    "--include-webgpu-render-state-manifest",
                    "true",
                    "--out",
                    str(output),
                ]
            if fix == "fix6_fix1":
                command.extend(
                    [
                        "--expected-runtime",
                        "webgpu",
                        "--expected-effective-display-runtime",
                        "webgpu-production",
                        "--expected-webgpu-backend-mode",
                        "webgpu-exclusive",
                        "--expected-webgpu-backend-implementation",
                        "webgpu-tile-compositor-frame-implementation",
                        "--expected-webgpu-canvas-presentation",
                        "true",
                        "--expected-webgpu-viewer-loop-hook",
                        "true",
                    ]
                )
            subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            sources[fix] = output.read_text(encoding="utf-8")
        generic_sources = {
            "phase3_step117_000151_v13": build_generic_capture_command(
                temp_dir,
                "phase3_step117_000151_v13",
                "phase3-step117",
                "phase3-step117-production-capture-diagnostic-isolation-recovery",
            ),
            "generic_capture_policy_test": build_generic_capture_command(
                temp_dir,
                "generic_capture_policy_test",
                "generic-phase",
                "generic-comparison-mode",
            ),
        }
        legacy_step117_output = Path(temp_dir) / "legacy_step117.js"
        subprocess.run(
            [
                "python3",
                str(ROOT / "tools" / "make_capture_commands.py"),
                "--step",
                "phase3_step117_000151_v13",
                "--preset",
                "stable",
                "--include-preamble",
                "true",
                "--out",
                str(legacy_step117_output),
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        legacy_step117_source = legacy_step117_output.read_text(encoding="utf-8")
        step114_order_output = Path(temp_dir) / "step114_fixed_time_order.js"
        subprocess.run(
            [
                "python3",
                str(ROOT / "tools" / "make_capture_commands.py"),
                "--step",
                "phase3_step114_fix10_fix6_fix1_000151_v13",
                "--preset",
                "stable",
                "--include-preamble",
                "true",
                "--include-webgpu-visible-record-dryrun",
                "true",
                "--include-png",
                "true",
                "--expected-runtime",
                "webgpu",
                "--expected-effective-display-runtime",
                "webgpu-production",
                "--expected-webgpu-backend-mode",
                "webgpu-exclusive",
                "--expected-webgpu-backend-implementation",
                "webgpu-tile-compositor-frame-implementation",
                "--expected-webgpu-canvas-presentation",
                "true",
                "--expected-webgpu-viewer-loop-hook",
                "true",
                "--out",
                str(step114_order_output),
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        step114_order_source = step114_order_output.read_text(encoding="utf-8")
    source = sources["fix5"]
    assert "phase3-step114-fix10-fix5" in source
    assert (
        "phase3-step114-fix10-fix5-coalesced-scheduler-request-drain"
        in source
    )
    assert "productionRuntimeBehaviorChanged: true" in source
    assert "productionRuntimeBehaviorChanged: false" in sources["fix4"]
    assert "phase3-step114-fix10-fix6" in sources["fix6"]
    assert (
        "phase3-step114-fix10-fix6-production-canvas-writer-ownership-correction"
        in sources["fix6"]
    )
    assert "productionRuntimeBehaviorChanged: true" in sources["fix6"]
    fix1_source = sources["fix6_fix1"]
    assert "phase3-step114-fix10-fix6-fix1" in fix1_source
    assert (
        "phase3-step114-fix10-fix6-fix1-production-webgpu-runtime-selection-correction"
        in fix1_source
    )
    assert "capture-blocked-production-runtime-mismatch" in fix1_source
    assert "phase3-production-runtime-mismatch-artifact-v1" in fix1_source
    validation_index = fix1_source.index("var productionRuntimeValidation =")
    readiness_index = fix1_source.index("var viewerDebugDataReadiness =")
    schedule_index = fix1_source.index("var captureScheduleRequest = null")
    assert validation_index < readiness_index < schedule_index
    mismatch_start = fix1_source.index(
        "if (productionRuntimeValidation?.ready !== true)"
    )
    mismatch_end = fix1_source.index("var preCaptureInitialPresentationSnapshot")
    mismatch_source = fix1_source[mismatch_start:mismatch_end]
    assert "_runtime_mismatch.json" in mismatch_source
    assert "scheduleRender(" not in mismatch_source
    assert "waitForViewerDebugDataReady(" not in mismatch_source
    assert "saveCurrentCanvasPng(" not in mismatch_source
    assert "captureWebGpuRenderStateManifestDebug(" not in mismatch_source
    executable = next(
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("//")
    )
    assert executable == "var step114CommandStartFence ="
    fence_end = source.index("var preCaptureInitialPresentationSnapshot")
    before_fence = source[:fence_end]
    for prohibited in (
        "await ",
        "new Promise",
        "setTimeout(",
        "requestAnimationFrame(",
        "scheduleRender(",
        "waitForViewerDebugDataReady(",
        "saveCurrentCanvasPng(",
    ):
        assert prohibited not in before_fence, prohibited
    assert_generic_capture_lifecycle(
        generic_sources["phase3_step117_000151_v13"],
        "phase3_step117_000151_v13",
        "phase3-step117",
        "phase3-step117-production-capture-diagnostic-isolation-recovery",
    )
    assert_generic_capture_lifecycle(
        generic_sources["generic_capture_policy_test"],
        "generic_capture_policy_test",
        "generic-phase",
        "generic-comparison-mode",
    )
    generic_policy_source = generic_sources["generic_capture_policy_test"]
    assert "phase3-step114" not in generic_policy_source
    assert "step114FixedTimeCaptureIsolation" not in generic_policy_source
    assert GENERIC_CAPTURE_LIFECYCLE not in legacy_step117_source
    assert "genericFreshProductionCaptureLifecycle" not in legacy_step117_source
    legacy_byte_equivalent = (
        hashlib.sha256(fix1_source.encode("utf-8")).hexdigest()
        == STEP114_LEGACY_FIXTURE_SHA256
    )
    assert legacy_byte_equivalent
    step114_request_index = step114_order_source.index(
        "captureScheduleRequest = await window.gpuViewerDebug.scheduleRender("
    )
    step114_png_index = step114_order_source.index(
        "await window.gpuViewerDebug.saveCurrentCanvasPng("
    )
    step114_diagnostic_index = step114_order_source.index(
        "await window.gpuViewerDebug.captureWebGpuVisibleRecordDryRunDebug("
    )
    step114_fixed_time_ordering_preserved = (
        step114_request_index < step114_png_index < step114_diagnostic_index
    )
    assert step114_fixed_time_ordering_preserved
    step114_artifact_naming_preserved = all(
        name in step114_order_source
        for name in (
            "phase3_step114_fix10_fix6_fix1_000151_v13_canvas.png",
            "phase3_step114_fix10_fix6_fix1_000151_v13_webgpu_visible_record_dryrun_compare.json",
            "phase3_step114_fix10_fix6_fix1_000151_v13_webgpu_visible_record_dryrun_capture_status.json",
        )
    )
    assert step114_artifact_naming_preserved
    summary_source = (ROOT / "tools" / "summarize_step_json.py").read_text()
    assert '== "phase3-step114-fix10-fix3"' not in summary_source
    assert "phase3-final-canvas-presentation-evidence-v1" in summary_source
    assert "phase3-synchronous-command-start-fence-v1" in summary_source
    spec = importlib.util.spec_from_file_location(
        "summarize_step_json",
        ROOT / "tools" / "summarize_step_json.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    step_summary = {
        "initialProductionPresentation": {
            "synchronousCommandStartFence": {
                "schemaVersion": "phase3-synchronous-command-start-fence-v1",
                "synchronousReadOnlyFence": True,
                "capturedInSingleJavaScriptTurn": True,
                "asyncOperationBeforeFence": False,
                "mutationBeforeFence": False,
            },
            "finalCanvasPresentationEvidence": {
                "urlOnlyBoundary": {"browserVisibleResult": None},
                "captureBoundary": {"browserVisibleResult": None},
            },
        },
        "step114Fix10Fix4ImplementationDecision": "blocked",
    }
    module.apply_step114_fix10_fix4_evidence(step_summary)
    assert step_summary["summaryOverlaySelectedByCanonicalEvidenceSchema"] is True
    assert step_summary["summaryOverlaySelectedByStepName"] is False
    return {
        "schemaVersion": REGRESSION_SCHEMA_VERSION,
        "checks": [
            {"name": "step114-runtime-preflight-boundary", "passed": True},
            {"name": "step114-synchronous-command-fence", "passed": True},
            {"name": "step114-legacy-command-byte-equivalence", "passed": True},
            {"name": "step114-fixed-time-ordering", "passed": True},
            {"name": "step114-artifact-naming", "passed": True},
            {"name": "generic-lifecycle-canonical-prefix", "passed": True},
            {"name": "generic-lifecycle-generic-prefix", "passed": True},
            {"name": "generic-lifecycle-explicit-selection", "passed": True},
            {"name": "summary-overlay-schema-selection", "passed": True},
        ],
        "legacyByteEquivalent": legacy_byte_equivalent,
        "step114FixedTimeOrderingPreserved": (
            step114_fixed_time_ordering_preserved
        ),
        "step114ArtifactNamingPreserved": step114_artifact_naming_preserved,
        "genericLifecycleChecks": {
            "phase3_step117_000151_v13": True,
            "generic_capture_policy_test": True,
            "stepNameDoesNotSelectPolicy": True,
        },
        "decision": "ready",
        "failureMessages": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run capture command boundary regression checks."
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Optional path for machine-readable regression results.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_checks()
    except Exception as error:
        result = {
            "schemaVersion": REGRESSION_SCHEMA_VERSION,
            "checks": [
                {
                    "name": "capture-command-boundary-suite",
                    "passed": False,
                }
            ],
            "legacyByteEquivalent": None,
            "step114FixedTimeOrderingPreserved": None,
            "step114ArtifactNamingPreserved": None,
            "genericLifecycleChecks": None,
            "decision": "blocked",
            "failureMessages": [f"{type(error).__name__}: {error}"],
        }
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if result["decision"] == "ready":
        print("capture command boundary smoke tests passed")
        return 0
    print("capture command boundary smoke tests failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
