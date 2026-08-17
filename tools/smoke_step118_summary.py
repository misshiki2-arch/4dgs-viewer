#!/usr/bin/env python3
from copy import deepcopy
import json
import tempfile
from pathlib import Path

from smoke_step117_fix4_summary_cross_artifact import write_fixture
from summarize_step_json import (
    build_step118_native_production_frame_data_path_summary,
    extract_runtime,
    summarize_step,
)


workset = {
    "contractVersion": "phase3-production-resident-workset-v1",
    "residentWorksetReady": True,
    "sceneRecordCount": 100,
    "residentRecordCount": 64,
    "nonResidentRecordCount": 36,
    "nonResidentRecordsExplicit": True,
    "diagnosticMaxRecordsUsed": False,
    "diagnosticCandidateSourceUsed": False,
}
execution_plan = {
    "contractVersion": "phase3-production-tile-execution-plan-v1",
    "gpuExecutionPlanReady": True,
    "resourceIdentity": "production-plan:7",
    "planIdentity": 7,
    "scatterConsumesPlan": True,
    "sortConsumesPlan": True,
    "compositorConsumesPlan": True,
    "productionCriticalReadbackUsed": False,
    "intermediateCpuControlRoundTripUsed": False,
    "sceneDependentCpuPlanMaterialized": False,
    "schedulerContinuationUsed": False,
}
terminal_observer = {
    "schemaVersion": "phase3-production-tile-execution-plan-terminal-observer-v1",
    "evidenceRole": "terminal-post-production-submission-observer",
    "productionControlInput": False,
    "rawPlanWordsPublished": False,
    "observerReady": True,
    "planIdentity": 7,
    "requiredReferenceCount": 80,
    "scatteredReferenceCount": 80,
    "sortedReferenceCount": 80,
    "compositedReferenceCount": 80,
    "overflowReferenceCount": 0,
    "capacityOverflowDetected": False,
    "capacityOverflowFailClosed": True,
}
bounded_execution = {
    "contractVersion": "phase3-production-gpu-bounded-execution-v1",
    "boundedExecutionReady": True,
    "allStagesCompleted": True,
    "gpuResourceLineageMaintained": True,
    "recordReferenceCapacitySeparated": True,
    "silentDropAllowed": False,
    "schedulerContinuationUsed": False,
    "inputReferenceCount": 80,
    "completedReferenceCount": 80,
}
data_path = {
    "contractVersion": "phase3-native-webgpu-production-frame-data-path-v1",
    "nativeProductionFrameDataPathReady": True,
    "diagnosticIndependent": True,
    "gpuResourceLineagePreserved": True,
    "countsMatch": True,
    "allResourceIdentitiesPresent": True,
    "capacityReady": True,
    "compositorSubmitted": True,
    "cpuReferenceUsedAsProductionInput": False,
    "diagnosticReadbackUsedAsProductionInput": False,
    "javascriptVisibleSamplesUsedAsProductionInput": False,
    "diagnosticMaxRecordsUsedAsProductionLimit": False,
    "capacityOverflowDetected": False,
    "capacityOverflowFailClosed": True,
    "silentDropAllowed": False,
    "gpuExecutionPlanContract": execution_plan,
    "terminalExecutionPlanObserver": terminal_observer,
    "boundedExecutionContract": bounded_execution,
}
step117_ready = {
    "machineReadableStep117Decision": "ready",
    "productionOwnershipPresentationPreservationDecision": "ready",
    "productionRuntimePresentation": {"decision": "ready"},
}
runtime = extract_runtime(
    {
        "runtimeSummary": {},
        "lastRenderResultSummary": {
            "productionResidentWorksetContract": workset,
            "webgpuProductionFrameDataPathContract": data_path,
        },
    }
)


def summarize_with_data_path(candidate):
    candidate_runtime = {
        **runtime,
        "webgpuProductionFrameDataPathContract": candidate,
    }
    return build_step118_native_production_frame_data_path_summary(
        candidate_runtime, step117_ready
    )


ready = summarize_with_data_path(data_path)
assert ready["schemaVersion"] == "phase3-step118-summary-v2"
assert ready["step118Decision"] == "ready", ready
assert ready["fix4Acceptance"]["decision"] == "ready"
assert ready["fix4Acceptance"]["executionPlan"]["identityMatches"] is True
assert ready["fix4Acceptance"]["referenceCounts"]["allMatch"] is True

missing_observer = deepcopy(data_path)
missing_observer.pop("terminalExecutionPlanObserver")
blocked = summarize_with_data_path(missing_observer)
assert blocked["step118Decision"] == "blocked", blocked
assert "terminal-execution-plan-evidence-not-ready" in blocked["blockedReasons"]

identity_mismatch = deepcopy(data_path)
identity_mismatch["terminalExecutionPlanObserver"]["planIdentity"] = 8
blocked = summarize_with_data_path(identity_mismatch)
assert "terminal-execution-plan-identity-mismatch" in blocked["blockedReasons"]

consumer_mismatch = deepcopy(data_path)
consumer_mismatch["gpuExecutionPlanContract"]["sortConsumesPlan"] = False
blocked = summarize_with_data_path(consumer_mismatch)
assert "production-plan-consumption-not-ready" in blocked["blockedReasons"]

count_mismatch = deepcopy(data_path)
count_mismatch["terminalExecutionPlanObserver"]["sortedReferenceCount"] = 79
blocked = summarize_with_data_path(count_mismatch)
assert "terminal-reference-counts-mismatch" in blocked["blockedReasons"]

cpu_dependency = deepcopy(data_path)
cpu_dependency["gpuExecutionPlanContract"][
    "intermediateCpuControlRoundTripUsed"
] = True
blocked = summarize_with_data_path(cpu_dependency)
assert "production-critical-cpu-dependency-detected" in blocked["blockedReasons"]

fail_open = deepcopy(data_path)
fail_open["capacityOverflowFailClosed"] = False
fail_open["terminalExecutionPlanObserver"]["capacityOverflowFailClosed"] = False
blocked = summarize_with_data_path(fail_open)
assert "production-overflow-fail-closed-not-ready" in blocked["blockedReasons"]

overflow_fail_closed = deepcopy(data_path)
overflow_fail_closed["nativeProductionFrameDataPathReady"] = False
overflow_fail_closed["capacityOverflowDetected"] = True
overflow_fail_closed["compositorSubmitted"] = False
overflow_fail_closed["terminalExecutionPlanObserver"].update(
    {
        "observerReady": False,
        "capacityOverflowDetected": True,
        "scatteredReferenceCount": 0,
        "sortedReferenceCount": 0,
        "compositedReferenceCount": 0,
        "overflowReferenceCount": 1,
    }
)
blocked = summarize_with_data_path(overflow_fail_closed)
assert blocked["step118Decision"] == "blocked"
assert blocked["fix4Acceptance"]["overflowFailClosedDecision"] == "ready"

step117_missing = build_step118_native_production_frame_data_path_summary(runtime)
assert step117_missing["step118Decision"] == "blocked"
assert (
    "step117-production-preservation-not-ready"
    in step117_missing["blockedReasons"]
)

malformed = build_step118_native_production_frame_data_path_summary({})
assert malformed["step118Decision"] == "blocked", malformed
assert malformed["blockedReasons"]

with tempfile.TemporaryDirectory() as temporary_directory:
    directory = Path(temporary_directory)
    prefix = "step118_fix6_summary_workflow"
    write_fixture(directory, prefix)
    runtime_path = directory / f"{prefix}_gpu_candidate_runtime_summary.json"
    runtime_artifact = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime_artifact["lastRenderResultSummary"] = {
        "productionResidentWorksetContract": workset,
        "webgpuProductionFrameDataPathContract": data_path,
    }
    runtime_path.write_text(json.dumps(runtime_artifact), encoding="utf-8")
    workflow_summary = summarize_step(directory, prefix)
    workflow_step118 = workflow_summary[
        "step118NativeWebGpuProductionFrameDataPath"
    ]
    assert workflow_summary["captureCommandContract"]["decision"] == "ready"
    assert workflow_summary["step117CrossArtifactConfirmation"][
        "machineReadableStep117Decision"
    ] == "ready"
    assert workflow_step118["step118Decision"] == "ready", workflow_step118
    assert workflow_step118["fix4Acceptance"]["decision"] == "ready"

print("Step118 Summary smoke tests passed")
