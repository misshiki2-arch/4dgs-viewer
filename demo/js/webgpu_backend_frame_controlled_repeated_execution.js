export const WEBGPU_BACKEND_FRAME_CONTROLLED_REPEATED_EXECUTION_MODE =
  'webgpu-backend-frame-controlled-repeated-execution';

export const WEBGPU_BACKEND_FRAME_CONTROLLED_REPEATED_EXECUTION_CONTRACT_VERSION =
  'phase3-step56-backend-frame-controlled-repeated-execution-contract-v1';

const DEFAULT_CONTROLLED_FRAME_COUNT = 3;

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function clampFrameCount(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return DEFAULT_CONTROLLED_FRAME_COUNT;
  return Math.max(2, Math.min(3, Math.floor(n)));
}

function errorToString(error) {
  if (!error) return null;
  return error.message ?? String(error);
}

function summarizeExecutedFrame({
  frameIndex,
  previousFrameSummary,
  backendFramePrototype,
  executionError = null
}) {
  const frameUnitContract = backendFramePrototype?.frameUnitContract ?? {};
  const validationSummary = backendFramePrototype?.validationSummary ?? {};
  const boundedFirstPresent =
    backendFramePrototype?.webgpuViewerCanvasBoundedFirstPresent ?? {};
  const boundedColorPresent =
    backendFramePrototype?.webgpuViewerCanvasBoundedColorPresent ?? {};
  const frameReady =
    !executionError &&
    backendFramePrototype?.backendFrameReady === true &&
    validationSummary.backendFrameReady === true &&
    frameUnitContract.currentTextureAcquisition === true &&
    frameUnitContract.commandBufferSubmitted === true &&
    boundedColorPresent?.submittedWorkDone === true;
  return {
    frameIndex,
    status: frameReady ? 'ok' : 'blocked',
    frameReady,
    previousFrameReady:
      previousFrameSummary?.frameReady ?? (frameIndex === 0 ? null : false),
    previousFrameStatus:
      previousFrameSummary?.status ?? (frameIndex === 0 ? null : 'missing'),
    executionMode: 'controlled-webgpu-backend-frame-execution',
    executionError: errorToString(executionError),
    currentTextureAcquisition:
      frameUnitContract.currentTextureAcquisition === true,
    boundedFirstPresentSucceeded:
      validationSummary.boundedFirstPresentSucceeded === true,
    boundedFirstPresentSubmitted:
      boundedFirstPresent?.commandBufferSubmitted === true,
    colorPresentSubmitted:
      boundedColorPresent?.commandBufferSubmitted === true,
    colorPresentSubmittedWorkDone:
      boundedColorPresent?.submittedWorkDone === true,
    commandBufferSubmitted:
      frameUnitContract.commandBufferSubmitted === true,
    selectedSourceKind: frameUnitContract.selectedSourceKind ?? null,
    selectionMode: frameUnitContract.selectionMode ?? null,
    selectedSampleCount: frameUnitContract.selectedSampleCount ?? 0,
    colorPresentSampleCount: frameUnitContract.colorPresentSampleCount ?? 0,
    selectorSelectedSamplesUsed:
      validationSummary.selectorSelectedSamplesUsed === true,
    fallbackSuppressedBySelectorSamples:
      validationSummary.fallbackSuppressedBySelectorSamples === true,
    webgl2HybridRenderingPrevented:
      validationSummary.webgl2HybridRenderingPrevented === true,
    frameBudgetReady: validationSummary.frameBudgetReady === true,
    continuationFrameReady: validationSummary.continuationFrameReady === true,
    sampleSources: frameUnitContract.sampleSources ?? [],
    firstValidationFailures: validationSummary.firstValidationFailures ?? []
  };
}

function buildValidationSummary({ frameSummaries }) {
  const allFramesReady =
    frameSummaries.length > 0 && frameSummaries.every((frame) => frame.frameReady);
  const frameIndicesMonotonic = frameSummaries.every(
    (frame, index) => frame.frameIndex === index
  );
  const allFramesSubmitted = frameSummaries.every(
    (frame) =>
      frame.commandBufferSubmitted === true &&
      frame.colorPresentSubmitted === true &&
      frame.colorPresentSubmittedWorkDone === true
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
  const previousFrameChainValid = frameSummaries.every((frame, index) => {
    if (index === 0) return frame.previousFrameReady === null;
    return frame.previousFrameReady === true && frame.previousFrameStatus === 'ok';
  });
  const controlledRepeatedExecutionReady =
    allFramesReady &&
    frameIndicesMonotonic &&
    allFramesSubmitted &&
    guardStableAcrossFrames &&
    sampleCountsStable &&
    previousFrameChainValid;
  const firstValidationFailures = [];
  if (!allFramesReady) {
    firstValidationFailures.push({
      stage: 'controlled-frame-readiness',
      reason: 'one or more controlled backend frame executions are not ready'
    });
  }
  if (!frameIndicesMonotonic) {
    firstValidationFailures.push({
      stage: 'frame-index',
      reason: 'controlled backend frame indices are not monotonic'
    });
  }
  if (!allFramesSubmitted) {
    firstValidationFailures.push({
      stage: 'controlled-submit',
      reason: 'each controlled backend frame must submit color present work and wait for completion'
    });
  }
  if (!guardStableAcrossFrames) {
    firstValidationFailures.push({
      stage: 'execution-guard',
      reason: 'exclusive canvas, selector, fallback, or WebGL2 guard changed across executed frames'
    });
  }
  if (!sampleCountsStable) {
    firstValidationFailures.push({
      stage: 'sample-count-stability',
      reason: 'selected or presented sample counts changed across controlled frames'
    });
  }
  if (!previousFrameChainValid) {
    firstValidationFailures.push({
      stage: 'previous-frame-chain',
      reason: 'previous frame readiness did not carry forward across controlled frames'
    });
  }
  return {
    controlledRepeatedExecutionReady,
    allFramesReady,
    frameIndicesMonotonic,
    allFramesSubmitted,
    guardStableAcrossFrames,
    sampleCountsStable,
    previousFrameChainValid,
    firstValidationFailures
  };
}

export async function buildWebGpuBackendFrameControlledRepeatedExecution({
  initialBackendFramePrototype = null,
  repeatedFrameCount = DEFAULT_CONTROLLED_FRAME_COUNT,
  executeBackendFrame = null
} = {}) {
  const startMs = nowMs();
  const frameCount = clampFrameCount(repeatedFrameCount);
  const frameSummaries = [];
  const backendFramePrototypes = [];
  let previousBackendFramePrototype = null;
  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    let backendFramePrototype = null;
    let executionError = null;
    try {
      if (frameIndex === 0 && initialBackendFramePrototype) {
        backendFramePrototype = initialBackendFramePrototype;
      } else if (typeof executeBackendFrame === 'function') {
        backendFramePrototype = await executeBackendFrame({
          frameIndex,
          previousBackendFramePrototype
        });
      } else {
        throw new Error('executeBackendFrame callback is unavailable');
      }
    } catch (error) {
      executionError = error;
    }
    const frameSummary = summarizeExecutedFrame({
      frameIndex,
      previousFrameSummary: frameSummaries[frameIndex - 1] ?? null,
      backendFramePrototype,
      executionError
    });
    frameSummaries.push(frameSummary);
    backendFramePrototypes.push(backendFramePrototype);
    previousBackendFramePrototype = backendFramePrototype;
  }
  const validationSummary = buildValidationSummary({ frameSummaries });
  const controlledRepeatedExecutionReady =
    validationSummary.controlledRepeatedExecutionReady === true;
  const firstFrameUnitContract =
    backendFramePrototypes[0]?.frameUnitContract ?? {};
  return {
    mode: WEBGPU_BACKEND_FRAME_CONTROLLED_REPEATED_EXECUTION_MODE,
    status: controlledRepeatedExecutionReady ? 'ok' : 'blocked',
    source:
      'Phase 3 Step56 controlled repeated WebGPU backend frame execution with guarded repeated submits',
    contractVersion:
      WEBGPU_BACKEND_FRAME_CONTROLLED_REPEATED_EXECUTION_CONTRACT_VERSION,
    controlledRepeatedExecutionImplemented: true,
    controlledRepeatedExecutionReady,
    productionDisplayConnectionImplemented: false,
    displayConnectionAllowed: false,
    webgl2HybridRenderingAllowed: false,
    requestedFrameCount: frameCount,
    executedBackendFrameSubmissions: frameSummaries.filter(
      (frame) => frame.commandBufferSubmitted
    ).length,
    simulatedContinuationFrameCount: 0,
    repeatedSubmitCount: frameSummaries.filter(
      (frame) => frame.colorPresentSubmitted
    ).length,
    controlledExecutionContract: {
      contractVersion:
        WEBGPU_BACKEND_FRAME_CONTROLLED_REPEATED_EXECUTION_CONTRACT_VERSION,
      repeatedFrameCount: frameCount,
      repeatedRunMode: 'controlled-webgpu-backend-frame-execution',
      frameStatePolicy:
        'frameIndex and previousBackendFramePrototype are carried into each executed backend frame',
      currentTexturePolicy:
        'each controlled frame re-enters the guarded currentTexture acquisition path under webgpu-exclusive ownership',
      presentSubmissionPolicy:
        'each controlled frame must submit bounded color present work and await queue completion',
      productionLoopConnected: false
    },
    frameSummaries,
    selectedSourceKind: firstFrameUnitContract.selectedSourceKind ?? null,
    colorPresentSampleCount: firstFrameUnitContract.colorPresentSampleCount ?? null,
    sampleSources: firstFrameUnitContract.sampleSources ?? [],
    fallbackPolicy: backendFramePrototypes[0]?.fallbackPolicy ?? {},
    validationSummary,
    firstValidationFailures: validationSummary.firstValidationFailures,
    blockers: controlledRepeatedExecutionReady
      ? [
          {
            stage: 'production-frame-scheduler',
            reason:
              'controlled repeated execution succeeded; production requestAnimationFrame scheduling remains intentionally disconnected'
          },
          {
            stage: 'interactive-camera',
            reason:
              'Three.js and OrbitControls remain camera input adapters; interactive camera implementation is outside Step56'
          },
          {
            stage: 'sh-color-evaluation',
            reason: 'WGSL SH/color evaluation parity remains deferred'
          }
        ]
      : [
          {
            stage: 'webgpu-backend-frame-controlled-repeated-execution',
            reason:
              'controlled repeated execution requires every frame to acquire currentTexture, submit color present work, and preserve guards'
          }
        ],
    nextBackendPrototypeStep: controlledRepeatedExecutionReady
      ? 'connect controlled repeated execution to a guarded frame scheduler while preserving exclusive canvas ownership'
      : 'restore controlled repeated execution readiness before expanding scheduling',
    timing: {
      webgpuBackendFrameControlledRepeatedExecutionMs: nowMs() - startMs
    }
  };
}
