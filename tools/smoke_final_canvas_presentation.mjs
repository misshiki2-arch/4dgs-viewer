import assert from 'node:assert/strict';

import {
  createFinalCanvasPresentationTraceRecorder,
  FINAL_CANVAS_PRESENTATION_PATHS
} from '../demo/js/common_4dgs_final_canvas_presentation.js';
import {
  createInitialProductionPresentationRecorder
} from '../demo/js/common_4dgs_initial_production_presentation.js';
import {
  buildFixedReferenceCameraContract
} from '../demo/js/common_4dgs_render_state_manifest.js';
import {
  buildOpaqueWebGpuPngAlphaNormalizationEvidence
} from '../demo/js/debug_download_utils.js';

function presentationEvent({
  generation,
  requestIdentity = 'viewer-render-request:1',
  pixelResult = 'nonblank',
  eventKind = 'production-presentation'
  , timestampMs = undefined
}) {
  return {
    eventKind,
    presentationSource:
      eventKind === 'clear'
        ? 'canvas-clear'
        : 'webgpu-tile-compositor-output-texture',
    sourceRequestIdentity: requestIdentity,
    presentingRequestIdentity: requestIdentity,
    productionGeneration: generation,
    compositorGeneration: generation,
    presentedGeneration: generation,
    frameIdentity: {
      generation,
      datasetCameraLabel: 'camera',
      datasetFrameNumber: 1,
      datasetTime: 2,
      outputWidth: 4,
      outputHeight: 4
    },
    sourcePixelEvidenceIdentity: {
      outputTextureIdentity: `output:${generation}`,
      generation,
      frameHash: `hash:${generation}:${pixelResult}`
    },
    sourcePixelResult: pixelResult,
    canvasWriteAttempted: true,
    canvasWriteSubmitted: true,
    canvasWriteCompleted: true,
    staleSource: false,
    presentationFailed: false
    , timestampMs
  };
}

{
  const recorder = createFinalCanvasPresentationTraceRecorder({
    activePathIdentities: [FINAL_CANVAS_PRESENTATION_PATHS.TILE_COMPOSITOR]
  });
  recorder.recordEvent(presentationEvent({ generation: 1, timestampMs: 10 }));
  recorder.recordEvent(presentationEvent({ generation: 2, timestampMs: 20 }));
  const evidence = recorder.getSnapshot({
    boundaryKind: 'pre-fence',
    boundaryTimestampMs: 15,
    expectedRequestIdentity: 'viewer-render-request:1',
    expectedGeneration: 1,
    requiredSteadyStateEventCount: 1
  });
  assert.equal(evidence.eventCountAtBoundary, 1);
  assert.equal(evidence.finalPresentedGeneration, 1);
}

{
  const recorder = createFinalCanvasPresentationTraceRecorder();
  const pending = recorder.observeQuiescence({
    schedulerSnapshot: {
      pendingRequestCount: 1,
      renderPending: true,
      productionFrameInFlight: false,
      needsRenderAgain: false
    }
  });
  assert.equal(pending.quiescent, false);
}

{
  const recorder = createFinalCanvasPresentationTraceRecorder();
  const idle = {
    pendingRequestCount: 0,
    renderPending: false,
    productionFrameInFlight: false,
    needsRenderAgain: false
  };
  recorder.recordEvent(presentationEvent({ generation: 1 }));
  recorder.observeQuiescence({ schedulerSnapshot: idle, requiredConsecutive: 2 });
  recorder.recordEvent(presentationEvent({ generation: 1 }));
  const changed = recorder.observeQuiescence({
    schedulerSnapshot: idle,
    requiredConsecutive: 2
  });
  assert.equal(changed.quiescent, false);
  assert.equal(changed.consecutiveStableObservationCount, 1);
}

{
  const recorder = createFinalCanvasPresentationTraceRecorder();
  recorder.recordEvent(presentationEvent({ generation: 1 }));
  const evidence = recorder.getSnapshot({
    boundaryKind: 'timeout',
    expectedRequestIdentity: 'viewer-render-request:1',
    expectedGeneration: 1,
    requiredSteadyStateEventCount: 1,
    requireQuiescence: true,
    quiescenceEvidence: { quiescent: false }
  });
  assert.equal(evidence.browserVisibleResult, null);
  assert.equal(evidence.classification, 'unknown-final-canvas-quiescence');
}

{
  const recorder = createFinalCanvasPresentationTraceRecorder({
    activePathIdentities: ['unregistered-current-texture-writer']
  });
  recorder.recordEvent(presentationEvent({ generation: 1 }));
  const runtime = recorder.getRuntimeStateSnapshot();
  const evidence = recorder.getSnapshot({
    boundaryKind: 'coverage',
    expectedRequestIdentity: 'viewer-render-request:1',
    expectedGeneration: 1,
    requiredSteadyStateEventCount: 1,
    requireCanvasWritePathCoverage: true
  });
  assert.equal(runtime.canvasWritePathCoverage.unregisteredWritePathCount, 1);
  assert.equal(evidence.browserVisibleResult, null);
  assert.equal(evidence.classification, 'unknown-final-canvas-write-path-coverage');
}

{
  const recorder = createInitialProductionPresentationRecorder();
  recorder.recordScheduleRequest({
    requestIdentity: 'viewer-render-request:2',
    source: 'default-scene-loaded'
  });
  recorder.recordFrameCompleted({
    schedulerFrameState: {
      requestIdentity: 'viewer-render-request:2',
      requestSource: 'default-scene-loaded'
    },
    frameEvidence: {
      productionFrameCompleted: true,
      productionGeneration: 1,
      productionSourceRequestIdentity: 'viewer-render-request:2'
    }
  });
  const snapshot = recorder.getSnapshot();
  assert.equal(snapshot.initialRequestIdentity, 'viewer-render-request:2');
  assert.equal(
    snapshot.initialProductionSourceRequestIdentity,
    'viewer-render-request:2'
  );
}

{
  const contract = buildFixedReferenceCameraContract({
    datasetViewMatrixMode: 'cuda-aligned',
    fixedReferenceCameraActivationMode: 'cuda-aligned-fixed-reference-camera'
  });
  assert.equal(contract.fixedReferenceCameraMode, true);
  assert.equal(contract.cameraMathChangedByContractNormalization, false);
}

function snapshot(recorder, generation, requiredSteadyStateEventCount = 1) {
  return recorder.getSnapshot({
    boundaryKind: 'smoke',
    expectedRequestIdentity: 'viewer-render-request:1',
    expectedGeneration: generation,
    expectedFrameIdentity: {
      generation,
      datasetCameraLabel: 'camera',
      datasetFrameNumber: 1,
      datasetTime: 2,
      outputWidth: 4,
      outputHeight: 4
    },
    requiredSteadyStateEventCount
  });
}

{
  const recorder = createFinalCanvasPresentationTraceRecorder();
  recorder.recordEvent(presentationEvent({ generation: 1 }));
  recorder.recordEvent(presentationEvent({
    generation: 1,
    pixelResult: 'black',
    eventKind: 'clear'
  }));
  const evidence = snapshot(recorder, 1);
  assert.equal(evidence.browserVisibleResult, false);
  assert.equal(evidence.laterOverwriteDetected, true);
  assert.equal(evidence.laterClearDetected, true);
}

{
  const recorder = createFinalCanvasPresentationTraceRecorder();
  recorder.recordEvent(presentationEvent({
    generation: 1,
    pixelResult: 'black',
    eventKind: 'black-fallback'
  }));
  recorder.recordEvent(presentationEvent({ generation: 1 }));
  const evidence = snapshot(recorder, 1);
  assert.equal(evidence.browserVisibleResult, true);
  assert.equal(evidence.finalSourcePixelResult, 'nonblank');
}

{
  const recorder = createFinalCanvasPresentationTraceRecorder();
  recorder.recordEvent(presentationEvent({ generation: 2 }));
  const evidence = snapshot(recorder, 1);
  assert.equal(evidence.browserVisibleResult, null);
  assert.equal(evidence.classification, 'unknown-final-canvas-source-identity');
}

{
  const recorder = createFinalCanvasPresentationTraceRecorder();
  recorder.recordEvent(presentationEvent({ generation: 1 }));
  const evidence = snapshot(recorder, 1, 2);
  assert.equal(evidence.browserVisibleResult, null);
  assert.equal(evidence.classification, 'unknown-final-canvas-steady-state');
}

{
  const evidence = buildOpaqueWebGpuPngAlphaNormalizationEvidence({
    preNormalizationPixelEvidence: {
      rgbHash: 'rgb-same',
      rgbNonzeroPixelCount: 3,
      rgbMax: 24,
      alphaMin: 0,
      alphaMax: 32,
      alphaOpaquePixelCount: 0
    },
    postNormalizationPixelEvidence: {
      rgbHash: 'rgb-same',
      rgbNonzeroPixelCount: 3,
      rgbMax: 24,
      alphaMin: 255,
      alphaMax: 255,
      alphaOpaquePixelCount: 16
    },
    preNormalizationSourceIdentity: { outputTextureIdentity: 'output:1' },
    postNormalizationBlobIdentity: { sha256: 'blob' },
    width: 4,
    height: 4
  });
  assert.equal(evidence.rgbInvariant, true);
  assert.equal(evidence.alphaOnlyChanged, true);
  assert.equal(evidence.genericTransparentPngCaptureUnaffected, true);
}

console.log('final canvas presentation smoke tests passed');
