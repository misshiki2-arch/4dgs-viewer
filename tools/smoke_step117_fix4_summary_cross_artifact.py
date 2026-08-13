#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from png_pixel_evidence import _write_rgba_png, inspect_png_file
from smoke_step117_fix2_summary_consumer import canonical_v2, capture_status
from summarize_step_json import (
    CAPTURE_COMMAND_CONTRACT_SCHEMA_VERSION,
    FRESH_PRODUCTION_CAPTURE_LIFECYCLE_POLICY,
    PRODUCTION_PNG_CAPTURE_STATUS_SCHEMA_VERSION,
    summarize_step,
)


def write_json(directory: Path, prefix: str, suffix: str, value: dict) -> None:
    (directory / f"{prefix}_{suffix}.json").write_text(
        json.dumps(value), encoding="utf-8"
    )


def frame_identity(generation: int = 9) -> dict:
    return {
        "generation": generation,
        "datasetCameraLabel": "000151_v13",
        "datasetFrameNumber": 151,
        "datasetTime": 23.2,
        "referenceCameraLabel": "000151_v13",
        "outputWidth": 2,
        "outputHeight": 2,
    }


def production_runtime(request_identity: str = "viewer-render-request:7") -> dict:
    return {
        "productionRuntimeContract": {
            "schemaVersion": "phase3-production-runtime-selection-contract-v1",
            "source": "common-production-runtime-selection-contract",
            "readOnlySnapshot": True,
            "runtimeEvidenceCurrent": True,
            "requestedRuntime": "webgpu",
            "effectiveDisplayRuntime": "webgpu-production",
            "backendMode": "webgpu-exclusive",
            "backendImplementation": (
                "webgpu-tile-compositor-frame-implementation"
            ),
            "canvasPresentationEnabled": True,
            "viewerLoopHookEnabled": True,
            "productionSelectionReady": True,
            "actualProductionPresentationPath": (
                "webgpu-tile-compositor-current-texture"
            ),
            "actualPresentationSource": (
                "cached-webgpu-tile-compositor-output-texture"
            ),
            "actualPresentationEventIdentity": "final-canvas-event:9",
            "actualPresentationSourceRequestIdentity": request_identity,
            "actualPresentedGeneration": 9,
            "gpuCandidateRuntime": "cpu-reference",
            "gpuCandidateRuntimeIsProductionDisplayRuntime": False,
        }
    }


def capture_contract() -> dict:
    predicates = {
        name: True
        for name in (
            "freshProductionRequestExactlyOnce",
            "forceProductionUpdateExactlyOnce",
            "duplicateProductionScheduleAbsent",
            "completionFenceBeforeDiagnostic",
            "stagesOrdered",
            "pngBeforeDiagnosticAbsent",
            "productionPngCapturePathUsed",
            "pngCaptureDownloadDeferred",
            "pngFallbackDisabled",
            "pngCaptureResultRetained",
            "pngStatusArtifactPresent",
            "pngStatusBeforePngDownload",
            "pngDownloadUsesCapturedBlob",
            "pngIsLastArtifactSave",
            "pngRenderBeforeCaptureFalse",
            "pngAfterProductionMutationAbsent",
        )
    }
    return {
        "schemaVersion": CAPTURE_COMMAND_CONTRACT_SCHEMA_VERSION,
        "policy": FRESH_PRODUCTION_CAPTURE_LIFECYCLE_POLICY,
        "phaseStep": "phase3-step117",
        "comparisonMode": "ownership-preservation",
        "runtimePreflight": {"expectedContractComplete": True},
        "stages": [
            {"name": name, "position": index}
            for index, name in enumerate(
                (
                    "runtime-preflight",
                    "fresh-production-request",
                    "production-completion-fence",
                    "diagnostic-compute-readback",
                    "diagnostic-result-json",
                    "diagnostic-status-json",
                    "runtime-summary-json",
                    "limited-draw-json",
                    "png-capture",
                    "png-status-json",
                    "png",
                )
            )
        ],
        "counts": {
            "freshProductionRequest": 1,
            "forceProductionUpdateTrue": 1,
            "diagnosticCaptureCall": 1,
            "runtimeSummaryCaptureCall": 1,
            "pngCaptureCall": 1,
            "pngStatusArtifactSave": 1,
            "pngSaveCall": 1,
            "productionScheduleCall": 1,
        },
        "predicates": predicates,
        "verificationErrors": [],
        "decision": "ready",
    }


def production_png_status(prefix: str, sha256: str) -> dict:
    presented_identity = frame_identity()
    requested_identity = {
        key: value for key, value in presented_identity.items() if key != "generation"
    }
    identity_comparison = {
        "matches": True,
        "mismatchedKeys": [],
        "missingKeys": [],
    }
    return {
        "schemaVersion": PRODUCTION_PNG_CAPTURE_STATUS_SCHEMA_VERSION,
        "artifactRole": "compact-production-png-capture-status",
        "captureLifecyclePolicy": FRESH_PRODUCTION_CAPTURE_LIFECYCLE_POLICY,
        "status": "success",
        "reason": None,
        "source": "cached-last-valid-webgpu-tile-compositor-output-texture-readback",
        "captureSourceKind": (
            "cached-last-valid-webgpu-tile-compositor-output-texture-readback"
        ),
        "requestedCaptureSource": "last-valid-webgpu-tile-compositor-output",
        "fileName": f"{prefix}_canvas.png",
        "outputWidth": 2,
        "outputHeight": 2,
        "captureBlobIdentity": {
            "schemaVersion": "phase3-capture-png-blob-identity-v1",
            "fileName": f"{prefix}_canvas.png",
            "mimeType": "image/png",
            "sizeBytes": 1,
            "sha256": sha256,
        },
        "encodedPngPixelEvidence": {
            "schemaVersion": "phase3-encoded-png-blob-pixel-evidence-v1",
            "decodeSupported": True,
            "decodeCompleted": True,
            "width": 2,
            "height": 2,
            "rgbNonzeroPixelCount": 1,
            "rgbNonblackRatio": 0.25,
            "rgbMax": 255,
            "pixelClassification": "nonblank",
            "decodeError": None,
        },
        "outputStats": {
            "nonzeroPixelCount": 1,
            "rgbNonzeroPixelCount": 1,
            "rgbNonblackRatio": 0.25,
            "rgbMax": 255,
        },
        "productionOutputGeneration": 9,
        "presentedOutputGeneration": 9,
        "capturedOutputGeneration": 9,
        "requestedStateIdentity": requested_identity,
        "presentedFrameIdentity": presented_identity,
        "capturedFrameIdentity": frame_identity(),
        "captureVsPresentedFrameIdentity": identity_comparison,
        "captureVsRequestedStateIdentity": identity_comparison,
        "captureMatchesPresentedFrame": True,
        "captureMatchesRequestedState": True,
        "staleCaptureDetected": False,
        "captureFreshnessKnown": True,
        "captureFreshnessClassification": (
            "captured-current-presented-fixed-reference-frame"
        ),
        "encodedBlobRetainedForFinalDownload": True,
    }


def write_fixture(directory: Path, prefix: str) -> None:
    request_identity = "viewer-render-request:7"
    canonical = canonical_v2()
    canonical["artifactSetIdentity"] = prefix
    canonical["artifactProvenance"] = {
        "schemaVersion": "phase3-diagnostic-artifact-provenance-v1",
        "capturePrefix": prefix,
        "requestIdentity": request_identity,
        "productionGeneration": 9,
        "frameIdentity": frame_identity(),
    }
    write_json(directory, prefix, "webgpu_visible_record_dryrun_compare", canonical)
    write_json(
        directory,
        prefix,
        "webgpu_visible_record_dryrun_capture_status",
        capture_status(requested=False, present=False),
    )
    write_json(directory, prefix, "gpu_candidate_runtime_summary", production_runtime())
    write_json(directory, prefix, "capture_command_contract", capture_contract())
    png_path = directory / f"{prefix}_canvas.png"
    _write_rgba_png(
        png_path,
        [bytes([255, 0, 0, 255, 0, 0, 0, 255]), bytes([0, 0, 0, 255] * 2)],
        2,
        2,
    )
    sha256 = inspect_png_file(png_path)["sha256"]
    write_json(
        directory,
        prefix,
        "png_capture_status",
        production_png_status(prefix, sha256),
    )


def confirmation(directory: Path, prefix: str) -> tuple[dict, dict]:
    summary = summarize_step(directory, prefix)
    value = summary["step117CrossArtifactConfirmation"]
    assert isinstance(value, dict)
    return summary, value


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)

        ready_prefix = "step117_cross_artifact_ready"
        write_fixture(directory, ready_prefix)
        ready_summary, ready = confirmation(directory, ready_prefix)
        assert ready_summary["loadErrors"] == {}
        assert ready["machineReadableStep117Decision"] == "ready"
        assert ready["productionOwnershipPresentationPreservationDecision"] == (
            "ready"
        )
        assert ready["step117OverallDecision"] == (
            "browser-user-observation-pending"
        )
        assert ready["browserObservationInferred"] is False
        assert ready["crossArtifactIdentity"]["decision"] == "ready"
        assert ready["png"]["blobSavedFileIdentityMatch"] is True

        mismatch_prefix = "step117_comparison_mismatch_separate"
        write_fixture(directory, mismatch_prefix)
        mismatch_path = directory / (
            f"{mismatch_prefix}_webgpu_visible_record_dryrun_compare.json"
        )
        mismatch = json.loads(mismatch_path.read_text(encoding="utf-8"))
        mismatch["comparison"]["anyMismatch"] = True
        mismatch["comparison"]["fieldMismatchCount"] = 4
        mismatch_path.write_text(json.dumps(mismatch), encoding="utf-8")
        _, mismatch_confirmation = confirmation(directory, mismatch_prefix)
        assert mismatch_confirmation["diagnostic"]["comparisonDecision"] == (
            "mismatch"
        )
        assert mismatch_confirmation["machineReadableStep117Decision"] == "ready"
        assert mismatch_confirmation["comparisonMismatchBlocksPreservation"] is False

        identity_prefix = "step117_generation_identity_mismatch"
        write_fixture(directory, identity_prefix)
        status_path = directory / f"{identity_prefix}_png_capture_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["capturedOutputGeneration"] = 10
        status_path.write_text(json.dumps(status), encoding="utf-8")
        _, identity_confirmation = confirmation(directory, identity_prefix)
        assert identity_confirmation["machineReadableStep117Decision"] == "blocked"
        assert "production-cross-artifact-identity-mismatch" in (
            identity_confirmation["blockedReasons"]
        )

        missing_png_prefix = "step117_missing_png_status"
        write_fixture(directory, missing_png_prefix)
        (directory / f"{missing_png_prefix}_png_capture_status.json").unlink()
        _, missing_png = confirmation(directory, missing_png_prefix)
        assert missing_png["machineReadableStep117Decision"] == "blocked"
        assert "png-capture-status-artifact-missing" in missing_png["blockedReasons"]

        malformed_schema_prefix = "step117_malformed_png_schema"
        write_fixture(directory, malformed_schema_prefix)
        malformed_path = directory / (
            f"{malformed_schema_prefix}_png_capture_status.json"
        )
        malformed = json.loads(malformed_path.read_text(encoding="utf-8"))
        malformed["schemaVersion"] = "incompatible-png-status"
        malformed_path.write_text(json.dumps(malformed), encoding="utf-8")
        malformed_summary, malformed_confirmation = confirmation(
            directory, malformed_schema_prefix
        )
        assert "png_capture_status" in malformed_summary["loadErrors"]
        assert malformed_confirmation["machineReadableStep117Decision"] == "blocked"

        lifecycle_prefix = "step117_lifecycle_contract_failure"
        write_fixture(directory, lifecycle_prefix)
        contract_path = directory / f"{lifecycle_prefix}_capture_command_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract["predicates"]["pngDownloadUsesCapturedBlob"] = False
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        _, lifecycle_confirmation = confirmation(directory, lifecycle_prefix)
        assert lifecycle_confirmation["captureLifecycle"]["decision"] == "blocked"
        assert lifecycle_confirmation["machineReadableStep117Decision"] == "blocked"

        stale_prefix = "step117_blob_saved_identity_mismatch"
        write_fixture(directory, stale_prefix)
        stale_path = directory / f"{stale_prefix}_png_capture_status.json"
        stale = json.loads(stale_path.read_text(encoding="utf-8"))
        stale["captureBlobIdentity"]["sha256"] = "0" * 64
        stale_path.write_text(json.dumps(stale), encoding="utf-8")
        _, stale_confirmation = confirmation(directory, stale_prefix)
        assert stale_confirmation["png"]["decision"] == "blocked"
        assert stale_confirmation["machineReadableStep117Decision"] == "blocked"

        compact_serialized = json.dumps(ready)
        assert "raw-detail-must-not-enter-summary" not in compact_serialized

    print("Step117 fix4 Summary cross-artifact smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
