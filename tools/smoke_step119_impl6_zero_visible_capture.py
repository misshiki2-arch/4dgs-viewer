#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from verify_capture_command_contract import build_capture_command_contract


ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE = "fresh-production-diagnostic-json-png"
DEFAULT_EXPECTATION = "production-nonblank"
ZERO_EXPECTATION = "zero-visible-production-black"


def generate(directory: Path, expectation: str) -> Path:
    output = directory / f"{expectation}_capture_commands.js"
    command = [
        "python3",
        "-B",
        str(ROOT / "tools" / "make_capture_commands.py"),
            "--step",
            f"impl6_{expectation}",
            "--preset",
            "runtime-only",
            "--capture-lifecycle",
            LIFECYCLE,
            "--phase-step",
            "phase3-step119-impl6",
            "--comparison-mode",
            "phase3-step119-population-aligned-zero-visible-control",
            "--include-webgpu-visible-record-dryrun",
            "true",
            "--include-runtime",
            "true",
            "--include-png",
            "true",
            "--include-camera-control-debug",
            "false",
            "--webgpu-backend-implementation",
            "webgpu-tile-compositor-frame-implementation",
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
    ]
    if expectation != DEFAULT_EXPECTATION:
        command.extend(["--production-capture-expectation", expectation])
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(["node", "--check", str(output)], cwd=ROOT, check=True)
    contract = build_capture_command_contract(output)
    assert contract["decision"] == "ready", contract["verificationErrors"]
    return output


def fixture_source(command: str, scenario: dict) -> str:
    scenario_json = json.dumps(scenario, separators=(",", ":"))
    return f"""
const scenario = {scenario_json};
const requestIdentity = 'viewer-render-request:11';
const generation = 11;
const targetFrame = {{
  generation,
  datasetCameraLabel: '000151_v13',
  datasetFrameNumber: 151,
  datasetTime: 23.2,
  referenceCameraLabel: '000151_v13',
  outputWidth: 1280,
  outputHeight: 720,
  frameHash: 'fixture-frame-hash'
}};
const pixel = scenario.pixel ?? 'black';
const eventKind = scenario.eventKind ??
  (scenario.cached === false ? 'production-presentation' : 'cached-production-presentation');
const writeCompleted = scenario.writeIncomplete !== true &&
  eventKind !== 'presentation-failure';
const productionEvent = {{
  eventSequence: 1,
  eventKind: 'production-presentation',
  presentationSource: 'webgpu-tile-compositor-output-texture',
  sourceRequestIdentity: requestIdentity,
  productionGeneration: generation,
  compositorGeneration: generation,
  presentedGeneration: generation,
  frameIdentity: targetFrame,
  sourcePixelResult: pixel,
  canvasWriteAttempted: true,
  canvasWriteSubmitted: true,
  canvasWriteCompleted: true,
  staleSource: false,
  presentationFailed: false
}};
const finalGeneration = scenario.differentGeneration ? 12 : generation;
const finalEvent = {{
  eventSequence: scenario.cached === false ? 1 : 2,
  eventKind,
  presentationSource:
    eventKind === 'cached-production-presentation'
      ? 'cached-webgpu-tile-compositor-output-texture'
      : eventKind === 'black-fallback'
        ? 'explicit-black-fallback'
        : eventKind === 'clear'
          ? 'explicit-canvas-clear'
          : 'webgpu-tile-compositor-output-texture',
  sourceRequestIdentity: requestIdentity,
  productionGeneration: finalGeneration,
  compositorGeneration: finalGeneration,
  presentedGeneration: writeCompleted ? finalGeneration : null,
  frameIdentity: {{ ...targetFrame, generation: finalGeneration }},
  sourcePixelResult: writeCompleted ? pixel : 'unknown',
  canvasWriteAttempted: true,
  canvasWriteSubmitted: scenario.writeIncomplete !== true,
  canvasWriteCompleted: writeCompleted,
  staleSource: scenario.stale === true,
  presentationFailed: eventKind === 'presentation-failure',
  error: eventKind === 'presentation-failure'
    ? {{ name: 'OperationError', message: 'fixture presentation failure' }}
    : null
}};
const policyFacts = {{
  productionPresentationPath:
    eventKind === 'production-presentation' ||
    eventKind === 'cached-production-presentation',
  cachedProductionPresentationPath:
    eventKind === 'cached-production-presentation',
  fallbackPresentationPath: eventKind === 'black-fallback',
  clearPresentationPath: eventKind === 'clear',
  sourcePixelResult: pixel,
  sameSourcePersistence: scenario.laterReplacement !== true &&
    scenario.differentGeneration !== true && scenario.stale !== true,
  samePixelResultPersistence: scenario.laterPixelChange !== true,
  laterSourceReplacement: scenario.laterReplacement === true,
  laterPixelResultChange: scenario.laterPixelChange === true,
  laterClear: eventKind === 'clear',
  laterFallback: eventKind === 'black-fallback',
  laterStaleSource: scenario.stale === true,
  laterDifferentGeneration: scenario.differentGeneration === true,
  quiescenceKnown: true,
  quiescent: scenario.nonQuiescent !== true,
  presentationFailed: eventKind === 'presentation-failure',
  webgpuValidationErrorDetected: scenario.validationError === true,
  invalidCommandBufferDetected: scenario.invalidCommandBuffer === true,
  queueSubmitFailureDetected: scenario.queueSubmitError === true
}};
const boundary = {{
  browserVisibleResult: pixel === 'nonblank' &&
    eventKind !== 'black-fallback' && eventKind !== 'clear' &&
    eventKind !== 'presentation-failure',
  quiescenceObservation: {{ quiescent: scenario.nonQuiescent !== true }},
  finalSourceRequestIdentity: requestIdentity,
  finalProductionGeneration: finalGeneration,
  finalCompositorGeneration: finalGeneration,
  finalPresentedGeneration: writeCompleted ? finalGeneration : null,
  finalFrameIdentity: finalEvent.frameIdentity,
  finalSourceIdentityKnown: scenario.identityUnknown !== true,
  finalSourceIdentityMatchesExpected:
    scenario.identityUnknown !== true && scenario.identityMismatch !== true,
  finalPresentationEventKind: eventKind,
  finalPresentationSource: finalEvent.presentationSource,
  finalSourcePixelResult: writeCompleted ? pixel : 'unknown',
  finalCanvasWriteAttempted: true,
  finalCanvasWriteSubmitted: scenario.writeIncomplete !== true,
  finalCanvasWriteCompleted: writeCompleted,
  laterOverwriteDetected: scenario.laterOverwrite === true,
  policyNeutralPresentationFacts: policyFacts,
  finalCanvasEvent: finalEvent,
  eventHistory: [
    ...(scenario.missingFreshProductionEvent ? [] : [productionEvent]),
    ...(scenario.cached === false ? [] : [finalEvent])
  ]
}};
const completionReady = scenario.executionIncomplete !== true;
const sourceRecordCount = scenario.sourcePopulationZero ? 0 : 524288;
const observer = {{
  executionCompletionContract: {{
    contractVersion: 'phase3-production-tile-execution-completion-v1',
    status: completionReady ? 'completed' : 'blocked',
    executionCompletionReady: completionReady,
    workClassification: 'zero-reference',
    terminalObserverCompleted: completionReady,
    staticPlanShapeMatches: completionReady,
    stageCountsMatch: completionReady,
    capacityRangeReady: completionReady,
    workloadShapeReady: completionReady
  }},
  recordCount: sourceRecordCount,
  tileCount: 3600,
  referenceCapacity: 8388608,
  requiredReferenceCount: 0,
  requiredPaddedReferenceCapacity: 0,
  scatteredReferenceCount: 0,
  sortedReferenceCount: 0,
  compositedReferenceCount: 0,
  compactOffsetTableReady: completionReady,
  capacityOverflowDetected: scenario.overflow === true,
  overflowReferenceCount: scenario.overflow ? 1 : 0
}};
const capacity = {{
  tileReferenceCapacityReady: completionReady && !scenario.overflow && sourceRecordCount > 0,
  recordCount: sourceRecordCount,
  tileCount: 3600,
  allocatedReferenceCapacity: 8388608,
  requiredReferenceCount: 0,
  requiredPaddedReferenceCapacity: 0,
  writtenReferenceCount: 0,
  capacityOverflowDetected: scenario.overflow === true,
  silentDropAllowed: scenario.silentDrop === true
}};
const runtimeError = scenario.runtimeError
  ? {{ name: 'GPUDeviceLostError', message: 'GPU device lost during fixture' }}
  : null;
const frameEvidence = {{
  requestIdentity,
  productionFrameCompleted: true,
  productionGeneration: generation,
  compositorGeneration: generation,
  presentedGeneration: generation,
  productionSourceRequestIdentity:
    scenario.requestMismatch ? 'viewer-render-request:other' : requestIdentity,
  productionFrameIdentity: targetFrame,
  compositorOutputGenerated: true,
  compositorOutputReady: scenario.outputIncomplete !== true,
  viewerCanvasPresented: true,
  logicalPresentationSucceeded: true,
  runtimeError,
  blockedReason: scenario.legacyBlockedReason
    ? 'viewer-loop-persistent-tile-compositor-presentation-not-observed'
    : null
}};
const runtimeResult = {{
  webgpuProductionFrameDataPathContract: {{
    terminalExecutionPlanObserver: observer,
    tileReferenceCapacityContract: capacity
  }},
  webgpuBackendViewerFrameExecutor: {{
    executionError: runtimeError
  }}
}};
const pngPixel = scenario.pngPixel ?? pixel;
const pngFresh = scenario.pngStale !== true;
const pngWidth = scenario.pngWrongDimensions ? 640 : 1280;
const pngResult = {{
  blob: new Blob([new Uint8Array([137, 80, 78, 71])], {{ type: 'image/png' }}),
  fileName: 'impl6_' + scenario.expectation + '_canvas.png',
  schemaVersion: 'phase3-production-output-png-capture-v1',
  status: 'success',
  reason: null,
  source: 'cached-last-valid-webgpu-tile-compositor-output-texture-readback',
  captureSourceKind: 'cached-last-valid-webgpu-tile-compositor-output-texture-readback',
  requestedCaptureSource: 'last-valid-webgpu-tile-compositor-output',
  outputWidth: pngWidth,
  outputHeight: 720,
  captureBlobIdentity: {{
    fileName: 'impl6_' + scenario.expectation + '_canvas.png',
    sha256: 'fixture-sha256',
    mimeType: 'image/png',
    sizeBytes: 4
  }},
  encodedPngPixelEvidence: {{
    decodeCompleted: true,
    pixelClassification: pngPixel,
    width: pngWidth,
    height: 720,
    rgbNonzeroPixelCount: pngPixel === 'black' ? 0 : 1,
    rgbMax: pngPixel === 'black' ? 0 : 255
  }},
  outputStats: {{
    rgbNonzeroPixelCount: pngPixel === 'black' ? 0 : 1,
    rgbMax: pngPixel === 'black' ? 0 : 255
  }},
  productionOutputGeneration: generation,
  presentedOutputGeneration: generation,
  capturedOutputGeneration: generation,
  captureFreshnessKnown: pngFresh,
  staleCaptureDetected: !pngFresh,
  captureMatchesPresentedFrame: pngFresh,
  captureMatchesRequestedState: pngFresh,
  capturedFrameIdentity: targetFrame,
  presentedFrameIdentity: targetFrame,
  requestedStateIdentity: targetFrame
}};
const downloads = [];
globalThis.copy = () => {{}};
globalThis.URL.createObjectURL = () => 'blob:fixture';
globalThis.URL.revokeObjectURL = () => {{}};
globalThis.document = {{
  body: {{ appendChild() {{}} }},
  createElement() {{
    return {{ click() {{}}, remove() {{}}, href: '', download: '' }};
  }}
}};
console.log = () => {{}};
console.error = () => {{}};
globalThis.window = {{
  location: {{
    search: '?datasetCameraLabel=000151_v13&datasetFrameNumber=151' +
      '&datasetTime=23.2&time=23.2&fixedCanvasWidth=1280&fixedCanvasHeight=720'
  }},
  gpuViewerDebug: {{
    getSynchronousCommandStartFence() {{
      return {{
        synchronousReadOnlyFence: true,
        productionRuntimeContract: {{}},
        initialPresentationSnapshot: {{ latestProductionGeneration: 10 }}
      }};
    }},
    validateExpectedProductionRuntimeContract() {{ return {{ ready: true }}; }},
    async waitForViewerDebugDataReady() {{ return {{ ready: true }}; }},
    async scheduleRender() {{
      return {{
        requestIdentity,
        source: 'generic-fresh-production-artifact-capture',
        disposition: 'scheduled',
        forceProductionUpdate: true
      }};
    }},
    getInitialProductionPresentationSnapshot() {{
      return {{ frameHistory: [frameEvidence] }};
    }},
    async waitForFinalCanvasPresentationQuiescence() {{ return boundary; }},
    getLastRenderResult() {{ return runtimeResult; }},
    async captureWebGpuVisibleRecordDryRunDebug() {{
      return {{
        canonicalDiagnosticResult: {{ schemaVersion: 'diagnostic-v2', status: 'ok' }},
        detailedLineageArtifact: null
      }};
    }},
    async downloadJsonDebug(value, name) {{ downloads.push({{ name, value }}); }},
    async captureGpuCandidateRuntimeSummaryDebug() {{
      return {{ limitedDrawSummary: null }};
    }},
    async saveCurrentCanvasPng(options) {{
      const expectedName = 'impl6_' + scenario.expectation + '_canvas.png';
      if (options.name !== expectedName) throw new Error('fixture PNG name mismatch');
      return pngResult;
    }}
  }}
}};

(async () => {{
  try {{
{command}
    return {{
      ok: true,
      expectation: genericCaptureExpectationContract,
      pngStatus: genericProductionPngCaptureStatus,
      result: genericProductionCaptureResult,
      downloads: downloads.map(item => item.name)
    }};
  }} catch (error) {{
    return {{
      ok: false,
      error: error?.message ?? String(error),
      expectation:
        typeof genericCaptureExpectationContract === 'undefined'
          ? null
          : genericCaptureExpectationContract,
      downloads: downloads.map(item => item.name)
    }};
  }}
}})().then(result => process.stdout.write(JSON.stringify(result)));
"""


def run_scenario(command_path: Path, scenario: dict) -> dict:
    source = command_path.read_text(encoding="utf-8")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mjs", encoding="utf-8", delete=False
    ) as fixture:
        fixture.write(fixture_source(source, scenario))
        fixture_path = Path(fixture.name)
    try:
        completed = subprocess.run(
            ["node", str(fixture_path)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)
    finally:
        fixture_path.unlink(missing_ok=True)


def assert_outcome(command: Path, scenario: dict, expected: bool) -> dict:
    result = run_scenario(command, scenario)
    assert result["ok"] is expected, (scenario, result)
    return result


def main() -> int:
    with tempfile.TemporaryDirectory() as temp_dir:
        directory = Path(temp_dir)
        default_command = generate(directory, DEFAULT_EXPECTATION)
        zero_command = generate(directory, ZERO_EXPECTATION)

        default_success = assert_outcome(
            default_command,
            {"expectation": DEFAULT_EXPECTATION, "pixel": "nonblank"},
            True,
        )
        assert default_success["expectation"]["expectationPolicy"] == DEFAULT_EXPECTATION
        assert default_success["expectation"]["ready"] is True
        assert_outcome(
            default_command,
            {"expectation": DEFAULT_EXPECTATION, "pixel": "black"},
            False,
        )
        assert_outcome(
            default_command,
            {
                "expectation": DEFAULT_EXPECTATION,
                "pixel": "nonblank",
                "pngPixel": "black",
            },
            False,
        )

        zero_success = assert_outcome(
            zero_command,
            {"expectation": ZERO_EXPECTATION, "pixel": "black"},
            True,
        )
        contract = zero_success["expectation"]
        assert contract["explicitZeroVisibleOptIn"] is True
        assert contract["canonicalExecution"]["ready"] is True
        assert contract["canonicalExecution"]["sourcePopulationPositive"] is True
        assert contract["outputAndPresentation"]["freshProductionEventObserved"] is True
        assert contract["outputAndPresentation"][
            "cachedPresentationContinuesFreshProduction"
        ] is True
        assert contract["pngEvidence"]["pixelClassification"] == "black"
        assert contract["pngEvidence"]["captureFreshnessKnown"] is True
        assert contract["pngEvidence"]["dimensionsMatchTarget"] is True
        assert contract["pngEvidence"]["blobSha256"] == "fixture-sha256"
        assert zero_success["pngStatus"]["captureExpectationContract"]["ready"] is True

        assert_outcome(
            zero_command,
            {"expectation": ZERO_EXPECTATION, "pixel": "black", "cached": False},
            True,
        )

        negative_scenarios = [
            {"pixel": "nonblank"},
            {"pixel": "unknown"},
            {"eventKind": "black-fallback"},
            {"eventKind": "clear"},
            {"eventKind": "presentation-failure"},
            {"stale": True},
            {"differentGeneration": True},
            {"laterReplacement": True},
            {"laterPixelChange": True},
            {"laterOverwrite": True},
            {"writeIncomplete": True},
            {"nonQuiescent": True},
            {"runtimeError": True},
            {"validationError": True},
            {"invalidCommandBuffer": True},
            {"queueSubmitError": True},
            {"executionIncomplete": True},
            {"outputIncomplete": True},
            {"sourcePopulationZero": True},
            {"missingFreshProductionEvent": True},
            {"requestMismatch": True},
            {"identityUnknown": True},
            {"identityMismatch": True},
            {"overflow": True},
            {"silentDrop": True},
            {"pngStale": True},
            {"pngWrongDimensions": True},
            {"pngPixel": "nonblank"},
        ]
        for scenario in negative_scenarios:
            assert_outcome(
                zero_command,
                {"expectation": ZERO_EXPECTATION, "pixel": "black", **scenario},
                False,
            )

        legacy_reason = assert_outcome(
            zero_command,
            {
                "expectation": ZERO_EXPECTATION,
                "pixel": "black",
                "legacyBlockedReason": True,
            },
            True,
        )
        runtime_health = legacy_reason["expectation"]["runtimeHealth"]
        assert runtime_health["legacyNonblankBlockedReason"] == (
            "viewer-loop-persistent-tile-compositor-presentation-not-observed"
        )
        assert runtime_health["legacyNonblankBlockedReasonUsedAsRuntimeError"] is False
        assert runtime_health["actualRuntimeErrorName"] is None

        invalid_cli = subprocess.run(
            [
                "python3",
                "-B",
                str(ROOT / "tools" / "make_capture_commands.py"),
                "--step",
                "invalid-zero-opt-in",
                "--production-capture-expectation",
                ZERO_EXPECTATION,
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert invalid_cli.returncode != 0
        assert "requires --capture-lifecycle" in invalid_cli.stderr

    print("Step119 Impl6 zero-visible capture smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
