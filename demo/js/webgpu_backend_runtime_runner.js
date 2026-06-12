export const WEBGPU_BACKEND_RUNTIME_RUNNER_MODE =
  'webgpu-backend-runtime-runner';

export const WEBGPU_BACKEND_RUNTIME_RUNNER_CONTRACT_VERSION =
  'phase3-step61-backend-runtime-runner-contract-v1';

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function buildRunnerContract({
  requestedBackendMode,
  allowViewerCanvasPresentation,
  enableViewerLoopHook,
  invocationSource,
  frameIndex,
  cameraSnapshot,
  viewerCanvasState,
  executorContract
}) {
  return {
    contractVersion: WEBGPU_BACKEND_RUNTIME_RUNNER_CONTRACT_VERSION,
    runnerMode: 'viewer-backend-runtime-frame-runner',
    invocationSource,
    frameIndex,
    requestedBackendMode,
    allowViewerCanvasPresentation: allowViewerCanvasPresentation === true,
    enableViewerLoopHook: enableViewerLoopHook === true,
    callableFromViewerBackendExecutor: true,
    backendImplementationKind: 'webgpu-visible-record-dry-run-runtime',
    backendImplementationReplaceable: true,
    recorderObserverSeparated: true,
    validationOracleRole: 'capture/dry-run can observe runner output without owning execution',
    productionLoopConnected: false,
    streamingImplemented: false,
    fullDatasetGpuResidencyRequired: false,
    cameraSnapshotProvided: !!cameraSnapshot,
    viewerCanvasProvided: viewerCanvasState?.provided === true,
    viewerCanvasContextMode: viewerCanvasState?.contextMode ?? null,
    webgl2FrameLifecycleSuppressed:
      viewerCanvasState?.webgl2FrameLifecycleSuppressed === true,
    executorContractVersion: executorContract?.contractVersion ?? null
  };
}

function buildCanonicalPresentSummary(backendFrameResult) {
  const present =
    backendFrameResult?.webgpuViewerCanvasBoundedColorPresent ?? {};
  const colorOutputContract = present?.colorOutputContract ?? {};
  const presentedSamples = Array.isArray(colorOutputContract.presentedSamples)
    ? colorOutputContract.presentedSamples
    : [];
  const sampleSources = Array.isArray(colorOutputContract.sampleSources)
    ? colorOutputContract.sampleSources
    : [];
  return {
    selectedSourceKind:
      backendFrameResult?.webgpuViewerCanvasBoundedColorSourceSelector
        ?.selectedSourceKind ??
      colorOutputContract.selectorSourceKind ??
      null,
    selectionMode: colorOutputContract.selectionMode ?? null,
    colorPresentSampleCount: present?.colorPresentSampleCount ?? null,
    selectorSelectedSamplesUsed:
      colorOutputContract.selectorSelectedSamplesUsed === true,
    fallbackSuppressedBySelectorSamples:
      colorOutputContract.fallbackSuppressedBySelectorSamples === true,
    commandBufferSubmitted: present?.commandBufferSubmitted === true,
    submittedWorkDone: present?.submittedWorkDone === true,
    presentedSampleCount: presentedSamples.length,
    sampleSources,
    presentedSamples,
    containsRenderHandoffFallback:
      sampleSources.includes('webgpuRenderHandoffStub.sampleRecords') ||
      presentedSamples.some(
        (sample) => sample?.source === 'webgpuRenderHandoffStub.sampleRecords'
      )
  };
}

function buildResourceLifecycleSummary(backendFrameResult) {
  const repeated =
    backendFrameResult?.webgpuBackendFrameControlledRepeatedExecution ?? {};
  const frameSummaries = Array.isArray(repeated.frameSummaries)
    ? repeated.frameSummaries
    : [];
  return {
    requestedFrameCount: repeated.requestedFrameCount ?? null,
    executedBackendFrameSubmissions:
      repeated.executedBackendFrameSubmissions ?? null,
    repeatedSubmitCount: repeated.repeatedSubmitCount ?? null,
    allFramesSubmitted:
      repeated.validationSummary?.allFramesSubmitted === true,
    frameIndices: frameSummaries.map((frame) => frame.frameIndex),
    perFrameCurrentTextureAcquired: frameSummaries.every(
      (frame) => frame.currentTextureAcquisition === true
    ),
    perFrameSubmitCompleted: frameSummaries.every(
      (frame) => frame.colorPresentSubmittedWorkDone === true
    )
  };
}

function buildValidationSummary({
  runnerContract,
  backendFrameResult,
  canonicalPresentSummary,
  resourceLifecycleSummary,
  executionError
}) {
  const guardAllowed =
    runnerContract.requestedBackendMode === 'webgpu-exclusive' &&
    runnerContract.allowViewerCanvasPresentation === true &&
    runnerContract.enableViewerLoopHook === true;
  const backendFrameResultProvided = !!backendFrameResult;
  const adapterReady =
    backendFrameResult?.webgpuBackendViewerLoopAdapter?.viewerLoopAdapterReady === true;
  const selectedTrueNativeSource =
    canonicalPresentSummary.selectedSourceKind ===
    'step40-constrained-display-adapter';
  const presentReady =
    canonicalPresentSummary.commandBufferSubmitted &&
    canonicalPresentSummary.submittedWorkDone &&
    canonicalPresentSummary.colorPresentSampleCount === 2;
  const noFallbackMixing =
    canonicalPresentSummary.containsRenderHandoffFallback === false &&
    canonicalPresentSummary.fallbackSuppressedBySelectorSamples === true;
  const resourceLifecycleReady =
    resourceLifecycleSummary.executedBackendFrameSubmissions === 3 &&
    resourceLifecycleSummary.repeatedSubmitCount === 3 &&
    resourceLifecycleSummary.allFramesSubmitted === true;
  const webgl2HybridRenderingPrevented =
    runnerContract.webgl2FrameLifecycleSuppressed === true;
  const runtimeRunnerReady =
    guardAllowed &&
    backendFrameResultProvided &&
    adapterReady &&
    selectedTrueNativeSource &&
    presentReady &&
    noFallbackMixing &&
    resourceLifecycleReady &&
    webgl2HybridRenderingPrevented &&
    !executionError;
  const firstValidationFailures = [];
  if (!guardAllowed) {
    firstValidationFailures.push({
      stage: 'runtime-runner-guard',
      reason:
        'runtime runner requires webgpu-exclusive, viewer canvas presentation guard, and webgpuBackendViewerLoopHook=true'
    });
  }
  if (!backendFrameResultProvided) {
    firstValidationFailures.push({
      stage: 'backend-frame-result',
      reason: 'runtime runner did not receive a backend frame result'
    });
  }
  if (!adapterReady) {
    firstValidationFailures.push({
      stage: 'viewer-loop-adapter',
      reason: 'runtime runner requires a ready viewer loop adapter'
    });
  }
  if (!selectedTrueNativeSource) {
    firstValidationFailures.push({
      stage: 'source-selection',
      reason: 'runtime runner expects the Step40 constrained display adapter source'
    });
  }
  if (!presentReady) {
    firstValidationFailures.push({
      stage: 'bounded-color-present',
      reason: 'runtime runner requires submitted bounded color present output'
    });
  }
  if (!noFallbackMixing) {
    firstValidationFailures.push({
      stage: 'fallback-policy',
      reason: 'runtime runner must not mix render-handoff fallback samples with selected native samples'
    });
  }
  if (!resourceLifecycleReady) {
    firstValidationFailures.push({
      stage: 'resource-lifecycle',
      reason: 'runtime runner expects three controlled backend frame submits'
    });
  }
  if (!webgl2HybridRenderingPrevented) {
    firstValidationFailures.push({
      stage: 'webgl2-lifecycle-suppression',
      reason: 'runtime runner cannot mix WebGPU presentation with WebGL2 rendering'
    });
  }
  if (executionError) {
    firstValidationFailures.push({
      stage: 'runtime-runner-execution-error',
      reason: executionError.message ?? 'runtime runner threw'
    });
  }
  return {
    runtimeRunnerReady,
    guardAllowed,
    backendFrameResultProvided,
    adapterReady,
    selectedTrueNativeSource,
    presentReady,
    noFallbackMixing,
    resourceLifecycleReady,
    webgl2HybridRenderingPrevented,
    firstValidationFailures
  };
}

export async function runWebGpuBackendRuntimeFrame({
  requestedBackendMode = 'webgl2-fallback',
  allowViewerCanvasPresentation = false,
  enableViewerLoopHook = false,
  invocationSource = 'viewer-backend-runtime-runner',
  frameIndex = 0,
  cameraSnapshot = null,
  viewerCanvasState = null,
  executorContract = null,
  runBackendFrame = null
} = {}) {
  const startMs = nowMs();
  const runnerContract = buildRunnerContract({
    requestedBackendMode,
    allowViewerCanvasPresentation,
    enableViewerLoopHook,
    invocationSource,
    frameIndex,
    cameraSnapshot,
    viewerCanvasState,
    executorContract
  });
  let backendFrameResult = null;
  let executionError = null;
  if (typeof runBackendFrame === 'function') {
    try {
      backendFrameResult = await runBackendFrame({
        frameIndex,
        invocationSource,
        runnerContract
      });
    } catch (error) {
      executionError = {
        name: error?.name ?? 'Error',
        message: error?.message ?? String(error)
      };
    }
  }
  const canonicalPresentSummary =
    buildCanonicalPresentSummary(backendFrameResult);
  const resourceLifecycleSummary =
    buildResourceLifecycleSummary(backendFrameResult);
  const validationSummary = buildValidationSummary({
    runnerContract,
    backendFrameResult,
    canonicalPresentSummary,
    resourceLifecycleSummary,
    executionError
  });
  const runtimeRunnerReady =
    validationSummary.runtimeRunnerReady === true;
  const summary = {
    mode: WEBGPU_BACKEND_RUNTIME_RUNNER_MODE,
    status: runtimeRunnerReady ? 'ok' : 'blocked',
    source:
      'Phase 3 Step61 viewer backend runtime runner contract for replaceable WebGPU backend frame execution',
    contractVersion: WEBGPU_BACKEND_RUNTIME_RUNNER_CONTRACT_VERSION,
    runtimeRunnerImplemented: true,
    runtimeRunnerReady,
    productionDisplayConnectionImplemented: false,
    displayConnectionAllowed: false,
    webgl2HybridRenderingAllowed: false,
    runnerContract,
    canonicalPresentSummary,
    resourceLifecycleSummary,
    selectedSourceKind: canonicalPresentSummary.selectedSourceKind,
    selectionMode: canonicalPresentSummary.selectionMode,
    colorPresentSampleCount:
      canonicalPresentSummary.colorPresentSampleCount,
    presentedSampleCount: canonicalPresentSummary.presentedSampleCount,
    sampleSources: canonicalPresentSummary.sampleSources,
    presentedSamples: canonicalPresentSummary.presentedSamples,
    executedBackendFrameSubmissions:
      resourceLifecycleSummary.executedBackendFrameSubmissions,
    repeatedSubmitCount: resourceLifecycleSummary.repeatedSubmitCount,
    validationSummary,
    executionError,
    firstValidationFailures: validationSummary.firstValidationFailures,
    blockers: runtimeRunnerReady
      ? [
          {
            stage: 'production-frame-scheduler',
            reason:
              'runtime runner is ready; production scheduling remains intentionally disconnected'
          },
          {
            stage: 'streaming-lod',
            reason:
              'streaming, chunking, LOD, and partial upload remain future runner policies'
          },
          {
            stage: 'sh-color-evaluation',
            reason: 'WGSL SH/color evaluation parity remains deferred'
          }
        ]
      : [
          {
            stage: 'webgpu-backend-runtime-runner',
            reason:
              'runner requires exclusive guards, ready adapter output, canonical present output, and preserved fallback policy'
          }
        ],
    nextBackendPrototypeStep: runtimeRunnerReady
      ? 'replace the dry-run backend implementation behind the same runner contract with a production WebGPU backend frame implementation'
      : 'restore runtime runner readiness before production backend substitution',
    timing: {
      webgpuBackendRuntimeRunnerMs: nowMs() - startMs
    }
  };
  return {
    summary,
    backendFrameResult,
    executionError
  };
}
