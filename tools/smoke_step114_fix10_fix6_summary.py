#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_summary_module():
    spec = importlib.util.spec_from_file_location(
        "summarize_step_json", ROOT / "tools" / "summarize_step_json.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_step_summary() -> dict:
    coverage = {
        "coverageComplete": True,
        "unregisteredWritePathCount": 0,
    }
    event = {
        "eventKind": "production-presentation",
        "presentationPathIdentity": "webgpu-tile-compositor-current-texture",
        "sourcePixelResult": "nonblank",
    }
    boundary = {
        "browserVisibleResult": True,
        "canvasWritePathCoverage": coverage,
        "finalCanvasEvent": event,
    }
    return {
        "initialProductionPresentation": {
            "synchronousCommandStartFence": {
                "schemaVersion": "phase3-synchronous-command-start-fence-v1",
                "commandStartTimestampMs": 100,
                "schedulerSnapshot": {
                    "pendingRequestCount": 0,
                    "productionFrameInFlight": False,
                },
                "productionRuntimeContract": {
                    "schemaVersion": "phase3-production-runtime-selection-contract-v1",
                    "readOnlySnapshot": True,
                    "runtimeEvidenceCurrent": True,
                    "snapshotTakenAtMs": 100,
                    "requestedRuntime": "webgpu",
                    "effectiveDisplayRuntime": "webgpu-production",
                    "productionSelectionReady": True,
                    "backendMode": "webgpu-exclusive",
                    "backendImplementation": (
                        "webgpu-tile-compositor-frame-implementation"
                    ),
                    "canvasPresentationEnabled": True,
                    "viewerLoopHookEnabled": True,
                    "actualProductionPresentationPath": (
                        "webgpu-tile-compositor-current-texture"
                    ),
                },
            },
            "finalCanvasPresentationEvidence": {
                "urlOnlyBoundary": boundary,
                "captureBoundary": boundary,
            },
        },
        "step114Fix10ImplementationDecision": "ready",
        "step114Decision": "ready",
    }


def main() -> int:
    module = load_summary_module()
    webgpu = {
        "webgpuViewerCanvasBoundedFirstPresent": {
            "boundedViewerCanvasFirstPresentImplemented": True,
            "commandBufferSubmitted": False,
        },
        "webgpuViewerCanvasBoundedColorPresent": {
            "boundedViewerCanvasColorPresentImplemented": True,
            "commandBufferSubmitted": False,
            "colorPresentSampleCount": 1,
        },
    }
    current = build_step_summary()
    module.apply_step114_fix10_fix6_evidence(current, webgpu)
    assert current["step114Fix10Fix6ImplementationDecision"] == "ready"
    assert current["step114Fix10Fix6EvidenceSelectedBySchema"] is True
    assert current["step114Fix10Fix6EvidenceSelectedByStepName"] is False

    stale = copy.deepcopy(build_step_summary())
    stale["initialProductionPresentation"]["synchronousCommandStartFence"][
        "productionRuntimeContract"
    ]["snapshotTakenAtMs"] = 99
    module.apply_step114_fix10_fix6_evidence(stale, webgpu)
    assert stale["step114Fix10Fix6ImplementationDecision"] == "blocked"
    assert "runtimeEvidenceCurrent" in stale[
        "step114Fix10Fix6ImplementationBlockedReasons"
    ]
    print("Step114 Fix10 Fix6 Summary smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
