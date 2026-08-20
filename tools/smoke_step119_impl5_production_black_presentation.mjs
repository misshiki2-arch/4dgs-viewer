import assert from 'node:assert/strict';

import {
  buildLastValidProductionOutputCacheDecision,
  buildPolicyNeutralProductionPresentationContract,
  createFinalCanvasPresentationTraceRecorder,
  FINAL_CANVAS_POLICY_NEUTRAL_PRESENTATION_SCHEMA_VERSION
} from '../demo/js/common_4dgs_final_canvas_presentation.js';
import {
  buildWebGpuTileListCompositorContract
} from '../demo/js/common_4dgs_record_contracts.js';

const requestIdentity = 'viewer-render-request:5';

function presentationContract({
  eventKind = 'production-presentation',
  pixelResult = 'nonblank',
  canonicalOutputCompletionReady = true,
  presentationFailed = false,
  staleSource = false,
  validationError = false
} = {}) {
  return buildPolicyNeutralProductionPresentationContract({
    canonicalOutputCompletionReady,
    productionWorkClassification:
      pixelResult === 'black' ? 'zero-reference' : 'nonzero-reference',
    presentationEventKind: eventKind,
    presentationSource: eventKind === 'cached-production-presentation'
      ? 'cached-webgpu-tile-compositor-output-texture'
      : 'webgpu-tile-compositor-output-texture',
    sourcePixelResult: pixelResult,
    sourceIdentityKnown: true,
    sourceIdentityMatchesExpected: true,
    canvasWriteAttempted: true,
    canvasWriteSubmitted: true,
    canvasWriteCompleted: presentationFailed === false,
    sameSourcePersistence: true,
    samePixelResultPersistence: true,
    persistenceObservedEventCount: 1,
    staleSource,
    presentationFailed,
    webgpuValidationErrorDetected: validationError
  });
}

function event({
  generation = 5,
  eventKind = 'production-presentation',
  pixelResult = 'black',
  staleSource = false,
  failed = false
} = {}) {
  return {
    eventKind,
    presentationSource:
      eventKind === 'cached-production-presentation'
        ? 'cached-webgpu-tile-compositor-output-texture'
        : eventKind === 'clear'
          ? 'explicit-canvas-clear'
          : eventKind === 'black-fallback'
            ? 'explicit-black-fallback'
            : 'webgpu-tile-compositor-output-texture',
    sourceRequestIdentity: requestIdentity,
    presentingRequestIdentity: requestIdentity,
    productionGeneration: generation,
    compositorGeneration: generation,
    presentedGeneration: failed ? null : generation,
    frameIdentity: {
      generation,
      frameHash: `hash:${generation}:${pixelResult}`,
      datasetCameraLabel: '000151_v13',
      datasetFrameNumber: 151,
      datasetTime: 23.2,
      outputWidth: 1280,
      outputHeight: 720
    },
    sourcePixelResult: failed ? 'unknown' : pixelResult,
    canvasWriteAttempted: true,
    canvasWriteSubmitted: true,
    canvasWriteCompleted: failed === false,
    staleSource,
    presentationFailed: failed
  };
}

function snapshot(recorder, { generation = 5, quiescent = true } = {}) {
  return recorder.getSnapshot({
    boundaryKind: 'step119-impl5-focused-smoke',
    expectedRequestIdentity: requestIdentity,
    expectedGeneration: generation,
    expectedFrameIdentity: {
      generation,
      datasetCameraLabel: '000151_v13',
      datasetFrameNumber: 151,
      datasetTime: 23.2,
      outputWidth: 1280,
      outputHeight: 720
    },
    requiredSteadyStateEventCount: 2,
    quiescenceEvidence: { quiescent }
  });
}

const productionNonblank = presentationContract();
assert.equal(
  productionNonblank.schemaVersion,
  FINAL_CANVAS_POLICY_NEUTRAL_PRESENTATION_SCHEMA_VERSION
);
assert.equal(productionNonblank.productionPresentationPath, true);
assert.equal(productionNonblank.sourcePixelResult, 'nonblank');
assert.equal(productionNonblank.genericPresentationCompletionReady, true);
assert.equal(productionNonblank.productionNonblankPresentationReady, true);
assert.equal(productionNonblank.productionBlackPresentationReady, false);
assert.equal(productionNonblank.lastValidCachePromotionReady, true);

const nonblankCompositor = buildWebGpuTileListCompositorContract({
  tileCompositorReady: true,
  boundedExecutionContract: { boundedExecutionReady: true },
  compositorPassSubmitted: true,
  compositorReadbackCompleted: true,
  compositorReadOffsetCountTable: true,
  compositorTraversedReferenceList: true,
  outputTextureCreated: true,
  outputTextureWritten: true,
  outputTextureReadbackMatchesSummary: true,
  processedTileCount: 3600,
  compositedTileCount: 1,
  nonEmptyCompositedTileCount: 1,
  compositedReferenceCount: 1,
  sourceTotalTileReferenceCount: 1,
  overflowCount: 0,
  compositorCurrentTextureRenderPassSubmitted: true,
  compositorCurrentTextureReadbackCompleted: true,
  compositorCurrentTextureReadbackNonZero: true,
  presentationFrameCount: 1,
  compositorPresentationFrameCount: 1,
  tileCompositorOutputPresentedToCurrentTexture: true,
  currentTexturePathMaintained: true,
  currentTextureUsesWebGpuTileCompositorOutput: true,
  presentationSampleFrameCount: 1,
  presentationNonBlankFrameCount: 1,
  presentationBlankFrameCount: 0,
  presentationAllSampledFramesNonBlank: true,
  presentationStableVisualOutput: true,
  policyNeutralPresentationContract: productionNonblank,
  lastValidOutputCacheDecision: buildLastValidProductionOutputCacheDecision({
    presentationContract: productionNonblank
  })
});
assert.equal(nonblankCompositor.presentationNonBlankFrameCount, 1);
assert.equal(nonblankCompositor.presentationAllSampledFramesNonBlank, true);
assert.equal(nonblankCompositor.presentationStableVisualOutput, true);
assert.equal(nonblankCompositor.compositorCurrentTextureReadbackNonZero, true);

const productionBlack = presentationContract({ pixelResult: 'black' });
assert.equal(productionBlack.presentationEventKind, 'production-presentation');
assert.equal(productionBlack.sourcePixelResult, 'black');
assert.equal(productionBlack.currentTextureWriteCompleted, true);
assert.equal(productionBlack.sourceIdentityMatchesExpected, true);
assert.equal(productionBlack.genericPresentationCompletionReady, true);
assert.equal(productionBlack.productionBlackPresentationReady, true);
assert.equal(productionBlack.productionNonblankPresentationReady, false);
assert.equal(productionBlack.fallbackPresentationPath, false);
assert.equal(productionBlack.lastValidCachePromotionReady, true);

const cachedBlack = presentationContract({
  eventKind: 'cached-production-presentation',
  pixelResult: 'black'
});
assert.equal(cachedBlack.cachedProductionPresentationPath, true);
assert.equal(cachedBlack.sourcePixelResult, 'black');
assert.equal(cachedBlack.genericPresentationCompletionReady, true);
assert.equal(cachedBlack.cachedProductionPresentationReady, true);
assert.equal(cachedBlack.lastValidCachePromotionReady, false);

const invalidZero = presentationContract({
  pixelResult: 'black',
  canonicalOutputCompletionReady: false
});
assert.equal(invalidZero.genericPresentationCompletionReady, false);
assert.equal(invalidZero.lastValidCachePromotionReady, false);
const rollbackDecision = buildLastValidProductionOutputCacheDecision({
  presentationContract: invalidZero,
  previousCacheAvailable: true,
  deviceChanged: false
});
assert.equal(rollbackDecision.promoteCandidate, false);
assert.equal(rollbackDecision.discardCandidate, true);
assert.equal(rollbackDecision.retainPrevious, true);

for (const [eventKind, failed] of [
  ['black-fallback', false],
  ['clear', false],
  ['presentation-failure', true]
]) {
  const contract = presentationContract({
    eventKind,
    pixelResult: eventKind === 'presentation-failure' ? 'unknown' : 'black',
    presentationFailed: failed
  });
  assert.equal(contract.genericPresentationCompletionReady, false, eventKind);
  assert.equal(contract.lastValidCachePromotionReady, false, eventKind);
}

const zeroCompositor = buildWebGpuTileListCompositorContract({
  tileCompositorReady: true,
  boundedExecutionContract: { boundedExecutionReady: true },
  compositorPassSubmitted: true,
  compositorReadbackCompleted: true,
  compositorReadOffsetCountTable: true,
  compositorTraversedReferenceList: true,
  outputTextureCreated: true,
  outputTextureWritten: true,
  outputTextureReadbackMatchesSummary: true,
  processedTileCount: 3600,
  compositedTileCount: 0,
  nonEmptyCompositedTileCount: 0,
  compositedReferenceCount: 0,
  sourceTotalTileReferenceCount: 0,
  overflowCount: 0,
  compositorCurrentTextureRenderPassSubmitted: true,
  compositorCurrentTextureReadbackCompleted: true,
  compositorCurrentTextureReadbackNonZero: false,
  presentationFrameCount: 1,
  compositorPresentationFrameCount: 1,
  tileCompositorOutputPresentedToCurrentTexture: true,
  currentTexturePathMaintained: true,
  currentTextureUsesWebGpuTileCompositorOutput: true,
  presentationSampleFrameCount: 1,
  presentationNonBlankFrameCount: 0,
  presentationBlankFrameCount: 1,
  presentationAllSampledFramesNonBlank: false,
  presentationStableVisualOutput: false,
  policyNeutralPresentationContract: productionBlack,
  lastValidOutputCacheDecision: buildLastValidProductionOutputCacheDecision({
    presentationContract: productionBlack
  })
});
assert.equal(zeroCompositor.canonicalOutputCompletionReady, true);
assert.equal(zeroCompositor.tileCompositorOutputPresentedToCurrentTexture, true);
assert.equal(zeroCompositor.currentTexturePathMaintained, true);
assert.equal(
  zeroCompositor.policyNeutralPresentationContract
    .genericPresentationCompletionReady,
  true
);
assert.equal(zeroCompositor.presentationNonBlankFrameCount, 0);
assert.equal(zeroCompositor.presentationAllSampledFramesNonBlank, false);
assert.equal(zeroCompositor.presentationStableVisualOutput, false);
assert.equal(zeroCompositor.compositorCurrentTextureReadbackNonZero, false);

{
  const recorder = createFinalCanvasPresentationTraceRecorder();
  recorder.recordEvent(event());
  recorder.recordEvent(event({ eventKind: 'cached-production-presentation' }));
  const evidence = snapshot(recorder);
  const facts = evidence.policyNeutralPresentationFacts;
  assert.equal(evidence.finalPresentationEventKind, 'cached-production-presentation');
  assert.equal(evidence.finalSourcePixelResult, 'black');
  assert.equal(evidence.browserVisibleResult, false);
  assert.equal(facts.productionPresentationPath, true);
  assert.equal(facts.cachedProductionPresentationPath, true);
  assert.equal(facts.sameSourcePersistence, true);
  assert.equal(facts.samePixelResultPersistence, true);
  assert.equal(facts.laterSourceReplacement, false);
  assert.equal(facts.laterPixelResultChange, false);
  assert.equal(facts.quiescenceKnown, true);
  assert.equal(facts.quiescent, true);
}

{
  const recorder = createFinalCanvasPresentationTraceRecorder();
  recorder.recordEvent(event({ pixelResult: 'nonblank' }));
  recorder.recordEvent(event({ pixelResult: 'black' }));
  const facts = snapshot(recorder).policyNeutralPresentationFacts;
  assert.equal(facts.sameSourcePersistence, true);
  assert.equal(facts.samePixelResultPersistence, false);
  assert.equal(facts.laterPixelResultChange, true);
}

{
  const recorder = createFinalCanvasPresentationTraceRecorder();
  recorder.recordEvent(event());
  recorder.recordEvent(event({ generation: 6 }));
  const facts = snapshot(recorder).policyNeutralPresentationFacts;
  assert.equal(facts.laterSourceReplacement, true);
  assert.equal(facts.laterDifferentGeneration, true);
}

for (const [eventKind, expectedField] of [
  ['clear', 'laterClear'],
  ['black-fallback', 'laterFallback']
]) {
  const recorder = createFinalCanvasPresentationTraceRecorder();
  recorder.recordEvent(event());
  recorder.recordEvent(event({ eventKind }));
  const facts = snapshot(recorder).policyNeutralPresentationFacts;
  assert.equal(facts[expectedField], true, eventKind);
  assert.equal(facts.genericPresentationCompletionReady, false, eventKind);
}

{
  const recorder = createFinalCanvasPresentationTraceRecorder();
  recorder.recordEvent(event());
  recorder.recordEvent(event({ staleSource: true }));
  const facts = snapshot(recorder).policyNeutralPresentationFacts;
  assert.equal(facts.laterStaleSource, true);
  assert.equal(facts.laterSourceReplacement, true);
}

{
  const recorder = createFinalCanvasPresentationTraceRecorder();
  recorder.recordEvent(event());
  recorder.recordEvent(event({ eventKind: 'presentation-failure', failed: true }));
  const evidence = snapshot(recorder, { quiescent: false });
  assert.equal(evidence.policyNeutralPresentationFacts.presentationFailed, true);
  assert.equal(evidence.policyNeutralPresentationFacts.quiescenceKnown, true);
  assert.equal(evidence.policyNeutralPresentationFacts.quiescent, false);
}

console.log('Step119 Impl5 production-black presentation smoke tests passed');
