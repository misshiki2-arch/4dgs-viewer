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
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


KNOWN_SUFFIXES = [
    "gpu_candidate_screen_coarse_compare",
    "gpu_candidate_screen_coarse_dryrun_visible_compare",
    "gpu_candidate_screen_coarse_sweep_summary",
    "gpu_visible_record_dryrun_compare",
    "gpu_raw_visible_record_dryrun_compare",
    "webgpu_visible_record_dryrun_compare",
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
    return {
        "status": get_path(summary, ["status"]),
        "reason": get_path(summary, ["reason"]),
        "computeMode": get_path(summary, ["computeMode"]),
        "scaffoldMode": get_path(summary, ["scaffoldMode"]),
        "phaseStep": get_path(summary, ["phaseStep"]),
        "implementedFields": get_path(summary, ["implementedFields"], []),
        "wgslComputedFields": get_path(summary, ["wgslComputedFields"], []),
        "wgslReferenceAssistedFields": get_path(
            summary, ["wgslReferenceAssistedFields"], []
        ),
        "cpuMaterializedFields": get_path(summary, ["cpuMaterializedFields"], []),
        "fieldComputeModes": get_path(summary, ["fieldComputeModes"], {}),
        "deferredFields": get_path(summary, ["deferredFields"], []),
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
        "gpuVisibleRecordDryRun": None,
        "gpuRawVisibleRecordDryRun": None,
        "webgpuVisibleRecordDryRun": None,
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

    if "webgpu_visible_record_dryrun_compare" in loaded:
        result["webgpuVisibleRecordDryRun"] = extract_webgpu_visible_record_dryrun(
            loaded["webgpu_visible_record_dryrun_compare"]
        )

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
    print_section("GPU visible record dry-run", summary.get("gpuVisibleRecordDryRun"))
    print_section("GPU raw visible record dry-run", summary.get("gpuRawVisibleRecordDryRun"))
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
