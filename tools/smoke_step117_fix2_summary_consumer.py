#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from summarize_step_json import (
    WEBGPU_VISIBLE_RECORD_DIAGNOSTIC_SCHEMA_VERSION,
    WEBGPU_VISIBLE_RECORD_DRY_RUN_SCHEMA_VERSION,
    WEBGPU_VISIBLE_RECORD_LINEAGE_SCHEMA_VERSION,
    summarize_step,
)


def write_json(directory: Path, prefix: str, suffix: str, value: dict) -> None:
    (directory / f"{prefix}_{suffix}.json").write_text(
        json.dumps(value), encoding="utf-8"
    )


def canonical_v2(*, requested: bool = False, present: bool = False) -> dict:
    return {
        "schemaVersion": WEBGPU_VISIBLE_RECORD_DIAGNOSTIC_SCHEMA_VERSION,
        "artifactRole": "canonical-compact-diagnostic-result",
        "artifactSetIdentity": "artifact-set-1",
        "artifactProvenance": {
            "capturePrefix": "fixture",
            "requestIdentity": 7,
            "productionGeneration": 9,
        },
        "sourceRuntimeResultSchemaVersion": (
            WEBGPU_VISIBLE_RECORD_DRY_RUN_SCHEMA_VERSION
        ),
        "productionRuntimeContract": {
            "effectiveDisplayRuntime": "must-not-be-used-from-diagnostic"
        },
        "phaseStep": "generic-phase",
        "status": "ok",
        "reason": "ok",
        "input": {
            "computeMode": "webgpu-storage-buffer-compute-fixed-record",
            "scaffoldMode": "diagnostic",
            "comparisonMode": "generic-comparison",
            "candidateInputSource": "synthetic",
            "inputContract": {"schemaVersion": "input-v1"},
            "inputBufferModes": {"raw": "storage"},
            "recordFloats": 12,
            "recordLayout": [["srcIndex", 0, 1]],
        },
        "cardinality": {
            "candidateCount": 65536,
            "computedRecordCount": 65536,
            "validRecordCount": 65000,
            "cpuReferenceValidRecordCount": 65000,
            "comparedRecordCount": 65536,
            "serializedFirstMismatchCount": 1,
            "serializedDetailedLineageRecordCount": 0,
            "computeCountIndependentFromSerializedDetail": True,
        },
        "execution": {
            "adapterInfoAvailable": True,
            "projectionParamMode": "canonical",
            "statePositionUploadMode": "storage",
            "candidateBufferCount": 65536,
            "outputBufferBytes": 3145728,
        },
        "comparison": {
            "contract": {"schemaVersion": "comparison-v1"},
            "tolerance": {"epsilon": 0.001},
            "anyMismatch": False,
            "fieldMismatchCount": 0,
            "maxAbsError": 0,
            "mismatchClassification": "none",
            "firstMismatches": [],
            "firstMismatchCount": 0,
            "firstMismatchLimit": 16,
            "firstMismatchesTruncated": False,
        },
        "stageSummaries": {"stateSource": {"status": "ok"}},
        "diagnosticStageAggregates": {
            "tileCountsWebGpuComparison": {
                "status": "ok",
                "anyMismatch": False,
            }
        },
        "validation": {
            "status": "completed",
            "diagnosticValidationSemanticsPreserved": True,
            "comparisonMismatchIsExecutionFailure": False,
        },
        "timing": {"totalMs": 12.5},
        "serializationPolicy": {
            "canonicalCardinalityMode": (
                "aggregate-plus-fixed-bounded-evidence"
            ),
            "computeRecordCountIndependent": True,
            "firstMismatchLimit": 16,
            "detailedLineageHardLimit": 32,
            "fullBackendSubresultsSerialized": False,
            "runtimeHistorySerialized": False,
            "captureOrchestrationSerialized": False,
            "legacyPayloadAliasesSerialized": False,
        },
        "detailedLineageArtifact": {
            "requested": requested,
            "required": requested,
            "present": present,
            "schemaVersion": (
                WEBGPU_VISIBLE_RECORD_LINEAGE_SCHEMA_VERSION if present else None
            ),
            "selectedRecordCount": 2 if present else 0,
            "selectedSrcIndices": [10, 20] if present else [],
            "suggestedSuffix": "_webgpu_visible_record_lineage.json",
        },
    }


def lineage_v1() -> dict:
    large_value = "raw-detail-must-not-enter-summary-" + ("x" * 4096)
    records = [
        {"srcIndex": index, "temporalEvaluation": {"large": large_value}}
        for index in (10, 20)
    ]
    return {
        "schemaVersion": WEBGPU_VISIBLE_RECORD_LINEAGE_SCHEMA_VERSION,
        "artifactRole": "optional-bounded-detailed-lineage",
        "artifactSetIdentity": "artifact-set-1",
        "artifactProvenance": {
            "capturePrefix": "fixture",
            "requestIdentity": 7,
            "productionGeneration": 9,
        },
        "sourceDiagnosticSchemaVersion": (
            WEBGPU_VISIBLE_RECORD_DRY_RUN_SCHEMA_VERSION
        ),
        "selection": {
            "mode": "explicit-src-indices",
            "requestedExplicitSrcIndexCount": 2,
            "effectiveLimit": 8,
            "hardLimit": 32,
            "selectedRecordCount": 2,
            "selectedSrcIndices": [10, 20],
            "missingExplicitSrcIndices": [],
            "selectionTruncated": False,
        },
        "actualEvidenceSource": "webgpu-production-readback",
        "actualEvidenceDispatch": "same-diagnostic-dispatch",
        "fieldAvailabilitySummary": {"clipPositionCount": 2},
        "recordCount": 2,
        "records": records,
        "productionDiagnosticSeparation": {
            "productionOutputDependsOnDetailedLineage": False,
            "diagnosticReadbackIsProductionDependency": False,
        },
    }


def capture_status(*, requested: bool, present: bool) -> dict:
    return {
        "schemaVersion": "phase3-capture-status-v1",
        "captureTarget": "webgpu-visible-record-dry-run",
        "status": "ok",
        "reason": "ok",
        "captureFatalError": False,
        "captureExceptionRecorded": False,
        "canonicalDiagnosticSchemaVersion": (
            WEBGPU_VISIBLE_RECORD_DIAGNOSTIC_SCHEMA_VERSION
        ),
        "canonicalDiagnosticSerializationSucceeded": True,
        "detailedLineageRequested": requested,
        "detailedLineagePresent": present,
        "detailedLineageSchemaVersion": (
            WEBGPU_VISIBLE_RECORD_LINEAGE_SCHEMA_VERSION if present else None
        ),
        "detailedLineageSerializationSucceeded": present,
    }


def runtime_summary() -> dict:
    return {
        "productionRuntimeContract": {
            "schemaVersion": "phase3-production-runtime-selection-contract-v1",
            "requestedRuntime": "webgpu",
            "effectiveDisplayRuntime": "webgpu-production",
            "backendMode": "webgpu-exclusive",
            "backendImplementation": (
                "webgpu-tile-compositor-frame-implementation"
            ),
            "canvasPresentationEnabled": True,
            "viewerLoopHookEnabled": True,
        }
    }


def write_design_c_fixture(
    directory: Path,
    prefix: str,
    *,
    requested: bool,
    include_lineage: bool,
) -> None:
    write_json(
        directory,
        prefix,
        "webgpu_visible_record_dryrun_compare",
        canonical_v2(requested=requested, present=include_lineage),
    )
    write_json(
        directory,
        prefix,
        "webgpu_visible_record_dryrun_capture_status",
        capture_status(requested=requested, present=include_lineage),
    )
    write_json(directory, prefix, "gpu_candidate_runtime_summary", runtime_summary())
    if include_lineage:
        write_json(
            directory,
            prefix,
            "webgpu_visible_record_lineage",
            lineage_v1(),
        )


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)

        optional_prefix = "design_c_optional_detail"
        write_design_c_fixture(
            directory,
            optional_prefix,
            requested=False,
            include_lineage=False,
        )
        optional_summary = summarize_step(directory, optional_prefix)
        assert optional_summary["loadErrors"] == {}
        optional_contract = optional_summary["designCDiagnosticArtifacts"]
        assert optional_contract["decision"] == "ready"
        assert optional_contract["detailDecision"] == "not-requested"
        assert optional_contract["browserObservationInferred"] is False
        assert optional_summary["runtime"]["effectiveDisplayRuntime"] == (
            "webgpu-production"
        )
        assert optional_summary["webgpuVisibleRecordDryRun"][
            "diagnosticEvidenceSource"
        ] == "compact-canonical-diagnostic-result-v2"
        assert "productionRuntimeContract" not in (
            optional_summary["webgpuVisibleRecordDryRun"]
        )
        assert "captureFatalError" not in (
            optional_summary["webgpuVisibleRecordDryRun"]
        )
        assert optional_summary["webgpuVisibleRecordCaptureStatus"][
            "captureStatus"
        ] == "ok"
        assert optional_contract["runtimeArtifactSource"] == (
            "gpu_candidate_runtime_summary"
        )
        assert optional_contract["runtimeEvidenceSource"] == (
            "production-runtime-selection-contract"
        )
        assert optional_contract["captureStatusSource"] == (
            "webgpu_visible_record_dryrun_capture_status"
        )

        mismatch_prefix = "design_c_comparison_mismatch"
        mismatch_canonical = canonical_v2()
        mismatch_canonical["comparison"]["anyMismatch"] = True
        mismatch_canonical["comparison"]["fieldMismatchCount"] = 3
        mismatch_canonical["comparison"]["mismatchClassification"] = (
            "validation-mismatch"
        )
        write_json(
            directory,
            mismatch_prefix,
            "webgpu_visible_record_dryrun_compare",
            mismatch_canonical,
        )
        write_json(
            directory,
            mismatch_prefix,
            "webgpu_visible_record_dryrun_capture_status",
            capture_status(requested=False, present=False),
        )
        write_json(
            directory,
            mismatch_prefix,
            "gpu_candidate_runtime_summary",
            runtime_summary(),
        )
        mismatch_summary = summarize_step(directory, mismatch_prefix)
        mismatch_contract = mismatch_summary["designCDiagnosticArtifacts"]
        assert mismatch_contract["diagnosticExecutionDecision"] == "ready"
        assert mismatch_contract["comparisonDecision"] == "mismatch"
        assert mismatch_contract["decision"] == "ready"

        required_missing_prefix = "design_c_required_detail_missing"
        write_design_c_fixture(
            directory,
            required_missing_prefix,
            requested=True,
            include_lineage=False,
        )
        required_missing_summary = summarize_step(
            directory, required_missing_prefix
        )
        required_missing_contract = required_missing_summary[
            "designCDiagnosticArtifacts"
        ]
        assert required_missing_contract["decision"] == "blocked"
        assert "required-detailed-lineage-artifact-missing" in (
            required_missing_contract["blockedReasons"]
        )

        required_prefix = "design_c_required_detail_present"
        write_design_c_fixture(
            directory,
            required_prefix,
            requested=True,
            include_lineage=True,
        )
        required_summary = summarize_step(directory, required_prefix)
        assert required_summary["loadErrors"] == {}
        required_contract = required_summary["designCDiagnosticArtifacts"]
        assert required_contract["decision"] == "ready"
        assert required_contract["detailDecision"] == "ready"
        lineage_summary = required_summary["webgpuVisibleRecordLineage"]
        assert lineage_summary["selectedSrcIndices"] == [10, 20]
        assert lineage_summary["rawRecordsCopiedToSummary"] is False
        serialized_summary = json.dumps(required_summary)
        assert "raw-detail-must-not-enter-summary" not in serialized_summary

        legacy_prefix = "legacy_v1"
        write_json(
            directory,
            legacy_prefix,
            "webgpu_visible_record_dryrun_compare",
            {
                "schemaVersion": WEBGPU_VISIBLE_RECORD_DRY_RUN_SCHEMA_VERSION,
                "status": "ok",
                "reason": "ok",
                "computeMode": "legacy",
                "candidateCount": 8,
                "recordCount": 8,
                "validRecordCount": 8,
                "recordComparison": {
                    "anyMismatch": False,
                    "fieldMismatchCount": 0,
                    "firstMismatches": [],
                },
            },
        )
        legacy_summary = summarize_step(directory, legacy_prefix)
        assert legacy_summary["loadErrors"] == {}
        assert legacy_summary["webgpuVisibleRecordDryRun"][
            "diagnosticEvidenceSource"
        ] == "legacy-visible-record-dry-run-v1"
        assert legacy_summary["designCDiagnosticArtifacts"] is None

        malformed_prefix = "malformed_v2"
        malformed = canonical_v2()
        malformed["cardinality"]["serializedDetailedLineageRecordCount"] = 1
        write_json(
            directory,
            malformed_prefix,
            "webgpu_visible_record_dryrun_compare",
            malformed,
        )
        malformed_summary = summarize_step(directory, malformed_prefix)
        assert "webgpu_visible_record_dryrun_compare" in (
            malformed_summary["loadErrors"]
        )
        assert malformed_summary["designCDiagnosticArtifacts"] is None

        incompatible_prefix = "incompatible_schema"
        write_json(
            directory,
            incompatible_prefix,
            "webgpu_visible_record_dryrun_compare",
            {"schemaVersion": "unsupported-schema"},
        )
        incompatible_summary = summarize_step(directory, incompatible_prefix)
        assert "webgpu_visible_record_dryrun_compare" in (
            incompatible_summary["loadErrors"]
        )

        orphan_lineage_prefix = "orphan_lineage"
        write_json(
            directory,
            orphan_lineage_prefix,
            "webgpu_visible_record_lineage",
            lineage_v1(),
        )
        orphan_lineage_summary = summarize_step(
            directory, orphan_lineage_prefix
        )
        assert "webgpu_visible_record_lineage" in (
            orphan_lineage_summary["loadErrors"]
        )
        assert orphan_lineage_summary["webgpuVisibleRecordLineage"] is None

        capture_error_prefix = "capture_status_error"
        write_design_c_fixture(
            directory,
            capture_error_prefix,
            requested=False,
            include_lineage=False,
        )
        capture_error = capture_status(requested=False, present=False)
        capture_error["status"] = "error"
        capture_error["captureFatalError"] = True
        capture_error["captureExceptionRecorded"] = True
        write_json(
            directory,
            capture_error_prefix,
            "webgpu_visible_record_dryrun_capture_status",
            capture_error,
        )
        capture_error_summary = summarize_step(directory, capture_error_prefix)
        capture_error_contract = capture_error_summary[
            "designCDiagnosticArtifacts"
        ]
        assert capture_error_contract["captureArtifactDecision"] == "blocked"
        assert "capture-artifact-status-not-ready" in (
            capture_error_contract["blockedReasons"]
        )

        malformed_status_prefix = "malformed_capture_status"
        write_design_c_fixture(
            directory,
            malformed_status_prefix,
            requested=False,
            include_lineage=False,
        )
        malformed_status = capture_status(requested=False, present=False)
        malformed_status.pop("canonicalDiagnosticSerializationSucceeded")
        write_json(
            directory,
            malformed_status_prefix,
            "webgpu_visible_record_dryrun_capture_status",
            malformed_status,
        )
        malformed_status_summary = summarize_step(
            directory, malformed_status_prefix
        )
        assert "webgpu_visible_record_dryrun_capture_status" in (
            malformed_status_summary["loadErrors"]
        )
        malformed_status_contract = malformed_status_summary[
            "designCDiagnosticArtifacts"
        ]
        assert malformed_status_contract["decision"] == "blocked"
        assert "capture-status-artifact-missing" in (
            malformed_status_contract["blockedReasons"]
        )

    print("Step117 fix2 Summary consumer smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
