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
