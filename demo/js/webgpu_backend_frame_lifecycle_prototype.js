export const WEBGPU_BACKEND_FRAME_LIFECYCLE_PROTOTYPE_MODE =
  'webgpu-backend-frame-lifecycle-prototype';

export const WEBGPU_BACKEND_FRAME_LIFECYCLE_CONTRACT_VERSION =
  'phase3-step55-backend-frame-lifecycle-contract-v1';

const DEFAULT_REPEATED_FRAME_COUNT = 3;

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function clampFrameCount(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return DEFAULT_REPEATED_FRAME_COUNT;
  return Math.max(2, Math.min(3, Math.floor(n)));
}

function buildRepeatedFrameSummary({ frameIndex, previousFrameSummary, backendFramePrototype }) {
  const frameUnitContract = backendFramePrototype?.frameUnitContract ?? {};
  const validationSummary = backendFramePrototype?.validationSummary ?? {};
  const frameBudgetContract =
    frameUnitContract.frameBudgetContract ?? backendFramePrototype?.frameBudgetContract ?? {};
  const continuationFrameContract =
    frameUnitContract.continuationFrameContract ??
    backendFramePrototype?.continuationFrameContract ??
    {};
  const frameReady =
    backendFramePrototype?.backendFrameReady === true &&
    validationSummary.backendFrameReady === true &&
    frameBudgetContract.selectedSamplesWithinBudget === true &&
    frameBudgetContract.colorPresentWithinBudget === true &&
    continuationFrameContract.repeatedFrameCallable === true;
  const commandSubmissionMode =
    frameIndex === 0
      ? 'initial-backend-frame-submission'
      : 'lifecycle-contract-replay-no-extra-submit';
  return {
    frameIndex,
    status: frameReady ? 'ok' : 'blocked',
    frameReady,
    previousFrameReady:
      previousFrameSummary?.frameReady ?? (frameIndex === 0 ? null : false),
    previousFrameStatus:
      previousFrameSummary?.status ?? (frameIndex === 0 ? null : 'missing'),
    currentTextureAcquisition:
      frameUnitContract.currentTextureAcquisition === true,
    boundedFirstPresentSucceeded:
      validationSummary.boundedFirstPresentSucceeded === true,
    selectedSourceKind: frameUnitContract.selectedSourceKind ?? null,
    selectionMode: frameUnitContract.selectionMode ?? null,
    selectedSampleCount: frameUnitContract.selectedSampleCount ?? 0,
    colorPresentSampleCount: frameUnitContract.colorPresentSampleCount ?? 0,
    commandSubmissionMode,
    commandBufferSubmitted:
      frameIndex === 0 && frameUnitContract.commandBufferSubmitted === true,
    selectorSelectedSamplesUsed:
      validationSummary.selectorSelectedSamplesUsed === true,
    fallbackSuppressedBySelectorSamples:
      validationSummary.fallbackSuppressedBySelectorSamples === true,
    webgl2HybridRenderingPrevented:
      validationSummary.webgl2HybridRenderingPrevented === true,
    frameBudgetReady: validationSummary.frameBudgetReady === true,
    continuationFrameReady: validationSummary.continuationFrameReady === true,
    budgetUtilization: frameBudgetContract.budgetUtilization ?? null,
    sampleSources: frameUnitContract.sampleSources ?? []
  };
}

function buildValidationSummary({ frameSummaries, backendFramePrototype }) {
  const validationSummary = backendFramePrototype?.validationSummary ?? {};
  const allFramesReady =
    frameSummaries.length > 0 && frameSummaries.every((frame) => frame.frameReady);
  const frameIndicesMonotonic = frameSummaries.every(
    (frame, index) => frame.frameIndex === index
  );
  const guardStableAcrossFrames = frameSummaries.every(
    (frame) =>
      frame.currentTextureAcquisition &&
      frame.boundedFirstPresentSucceeded &&
      frame.selectorSelectedSamplesUsed &&
      frame.fallbackSuppressedBySelectorSamples &&
      frame.webgl2HybridRenderingPrevented
  );
  const sampleCountsStable = frameSummaries.every(
    (frame) =>
      frame.selectedSampleCount === frameSummaries[0]?.selectedSampleCount &&
      frame.colorPresentSampleCount === frameSummaries[0]?.colorPresentSampleCount
  );
  const noExtraSubmitsAfterInitialFrame = frameSummaries
    .slice(1)
    .every((frame) => frame.commandBufferSubmitted === false);
  const repeatedRunReady =
    backendFramePrototype?.backendFrameReady === true &&
    validationSummary.backendFrameReady === true &&
    allFramesReady &&
    frameIndicesMonotonic &&
    guardStableAcrossFrames &&
    sampleCountsStable &&
    noExtraSubmitsAfterInitialFrame;
  const firstValidationFailures = [];
  if (!allFramesReady) {
    firstValidationFailures.push({
      stage: 'repeated-frame-readiness',
      reason: 'one or more repeated backend frame summaries are not ready'
    });
  }
  if (!frameIndicesMonotonic) {
    firstValidationFailures.push({
      stage: 'frame-index',
      reason: 'repeated backend frame indices are not monotonic'
    });
  }
  if (!guardStableAcrossFrames) {
    firstValidationFailures.push({
      stage: 'lifecycle-guard',
      reason: 'exclusive canvas, selector, fallback, or WebGL2 guard changed across frames'
    });
  }
  if (!sampleCountsStable) {
    firstValidationFailures.push({
      stage: 'sample-count-stability',
      reason: 'selected or presented sample counts changed across repeated frames'
    });
  }
  if (!noExtraSubmitsAfterInitialFrame) {
    firstValidationFailures.push({
      stage: 'dry-run-submit-policy',
      reason: 'dry-run lifecycle prototype should not submit extra command buffers after frame 0'
    });
  }
  return {
    repeatedRunReady,
    allFramesReady,
    frameIndicesMonotonic,
    guardStableAcrossFrames,
    sampleCountsStable,
    noExtraSubmitsAfterInitialFrame,
    firstValidationFailures
  };
}

export function buildWebGpuBackendFrameLifecyclePrototype({
  webgpuBackendFramePrototype = null,
  repeatedFrameCount = DEFAULT_REPEATED_FRAME_COUNT
} = {}) {
  const startMs = nowMs();
  const frameCount = clampFrameCount(repeatedFrameCount);
  const frameSummaries = [];
  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    frameSummaries.push(
      buildRepeatedFrameSummary({
        frameIndex,
        previousFrameSummary: frameSummaries[frameIndex - 1] ?? null,
        backendFramePrototype: webgpuBackendFramePrototype
      })
    );
  }
  const validationSummary = buildValidationSummary({
    frameSummaries,
    backendFramePrototype: webgpuBackendFramePrototype
  });
  const lifecycleReady = validationSummary.repeatedRunReady === true;
  const frameUnitContract = webgpuBackendFramePrototype?.frameUnitContract ?? {};
  return {
    mode: WEBGPU_BACKEND_FRAME_LIFECYCLE_PROTOTYPE_MODE,
    status: lifecycleReady ? 'ok' : 'blocked',
    source:
      'Phase 3 Step55 WebGPU backend repeated-run lifecycle prototype over the Step54 frame unit',
    contractVersion: WEBGPU_BACKEND_FRAME_LIFECYCLE_CONTRACT_VERSION,
    lifecyclePrototypeImplemented: true,
    lifecycleReady,
    productionDisplayConnectionImplemented: false,
    displayConnectionAllowed: false,
    webgl2HybridRenderingAllowed: false,
    requestedFrameCount: frameCount,
    executedBackendFrameSubmissions: frameUnitContract.commandBufferSubmitted === true ? 1 : 0,
    simulatedContinuationFrameCount: Math.max(0, frameCount - 1),
    lifecycleContract: {
      contractVersion: WEBGPU_BACKEND_FRAME_LIFECYCLE_CONTRACT_VERSION,
      repeatedFrameCount: frameCount,
      repeatedRunMode: 'dry-run-lifecycle-contract-replay',
      frameStatePolicy:
        'frameIndex and previousFrameSummary are carried across repeated backend frame summaries',
      currentTexturePolicy:
        'frame 0 uses the Step54 currentTexture acquisition; continuation frames validate the same exclusive ownership guard without production scheduling',
      presentSubmissionPolicy:
        'only the Step54 backend frame submission is counted; continuation frames do not submit additional command buffers in this dry-run prototype',
      frameBudgetContract:
        frameUnitContract.frameBudgetContract ??
        webgpuBackendFramePrototype?.frameBudgetContract ??
        {},
      continuationFrameContract:
        frameUnitContract.continuationFrameContract ??
        webgpuBackendFramePrototype?.continuationFrameContract ??
        {}
    },
    frameSummaries,
    selectedSourceKind: frameUnitContract.selectedSourceKind ?? null,
    colorPresentSampleCount: frameUnitContract.colorPresentSampleCount ?? null,
    sampleSources: frameUnitContract.sampleSources ?? [],
    fallbackPolicy: webgpuBackendFramePrototype?.fallbackPolicy ?? {},
    validationSummary,
    firstValidationFailures: validationSummary.firstValidationFailures,
    blockers: lifecycleReady
      ? [
          {
            stage: 'production-frame-scheduler',
            reason:
              'repeated-run lifecycle prototype is ready; production requestAnimationFrame scheduling remains intentionally disconnected'
          },
          {
            stage: 'interactive-camera',
            reason:
              'Three.js and OrbitControls remain camera input adapters; interactive camera implementation is outside Step55'
          },
          {
            stage: 'sh-color-evaluation',
            reason: 'WGSL SH/color evaluation parity remains deferred'
          }
        ]
      : [
          {
            stage: 'webgpu-backend-frame-lifecycle-prototype',
            reason:
              'repeated frame lifecycle requires Step54 backend frame readiness, stable guards, and stable selector/present samples'
          }
        ],
    nextBackendPrototypeStep: lifecycleReady
      ? 'connect lifecycle prototype to a guarded production frame scheduler while preserving exclusive canvas ownership'
      : 'restore Step54 backend frame readiness before expanding repeated lifecycle execution',
    timing: {
      webgpuBackendFrameLifecyclePrototypeMs: nowMs() - startMs
    }
  };
}
