#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import tempfile
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
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
    print("capture command boundary smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
