import {
  WEBGPU_NORMAL_BACKEND_FRAME_IMPLEMENTATION_MODE,
  runWebGpuNormalBackendFrameImplementation
} from './webgpu_normal_backend_frame_implementation.js';

export const WEBGPU_BACKEND_RUNTIME_RUNNER_MODE =
  'webgpu-backend-runtime-runner';

export const WEBGPU_BACKEND_RUNTIME_RUNNER_CONTRACT_VERSION =
  'phase3-step70-backend-runtime-runner-contract-v1';

export const WEBGPU_BACKEND_DRY_RUN_IMPLEMENTATION_KIND =
  'webgpu-visible-record-dry-run-runtime';

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
  executorContract,
  backendImplementationKind
}) {
  const selectedBackendImplementationKind =
    backendImplementationKind === WEBGPU_NORMAL_BACKEND_FRAME_IMPLEMENTATION_MODE
      ? WEBGPU_NORMAL_BACKEND_FRAME_IMPLEMENTATION_MODE
      : WEBGPU_BACKEND_DRY_RUN_IMPLEMENTATION_KIND;
  return {
    contractVersion: WEBGPU_BACKEND_RUNTIME_RUNNER_CONTRACT_VERSION,
    runnerMode: 'viewer-backend-runtime-frame-runner',
    invocationSource,
    frameIndex,
    requestedBackendMode,
    allowViewerCanvasPresentation: allowViewerCanvasPresentation === true,
    enableViewerLoopHook: enableViewerLoopHook === true,
    callableFromViewerBackendExecutor: true,
    backendImplementationKind: selectedBackendImplementationKind,
    backendImplementationSelectionMode:
      selectedBackendImplementationKind === WEBGPU_NORMAL_BACKEND_FRAME_IMPLEMENTATION_MODE
        ? 'explicit-normal-webgpu-backend-implementation'
        : 'validation-oracle-dry-run-implementation',
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
  executionError,
  normalBackendImplementation
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
  const normalBackendImplementationRequested =
    runnerContract.backendImplementationKind ===
    WEBGPU_NORMAL_BACKEND_FRAME_IMPLEMENTATION_MODE;
  const normalBackendImplementationReady =
    !normalBackendImplementationRequested ||
    normalBackendImplementation?.normalBackendImplementationReady === true;
  const runtimeRunnerReady =
    guardAllowed &&
    backendFrameResultProvided &&
    adapterReady &&
    selectedTrueNativeSource &&
    presentReady &&
    noFallbackMixing &&
    resourceLifecycleReady &&
    webgl2HybridRenderingPrevented &&
    normalBackendImplementationReady &&
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
  if (!normalBackendImplementationReady) {
    firstValidationFailures.push({
      stage: 'normal-backend-implementation',
      reason:
        'runtime runner selected the normal WebGPU backend implementation, but it was not ready'
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
    normalBackendImplementationRequested,
    normalBackendImplementationReady,
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
  backendImplementationKind = WEBGPU_BACKEND_DRY_RUN_IMPLEMENTATION_KIND,
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
    executorContract,
    backendImplementationKind
  });
  let backendFrameResult = null;
  let executionError = null;
  let normalBackendImplementationResult = null;
  if (typeof runBackendFrame === 'function') {
    try {
      if (
        runnerContract.backendImplementationKind ===
        WEBGPU_NORMAL_BACKEND_FRAME_IMPLEMENTATION_MODE
      ) {
        normalBackendImplementationResult =
          await runWebGpuNormalBackendFrameImplementation({
            frameIndex,
            invocationSource,
            runnerContract,
            cameraSnapshot,
            viewerCanvasState,
            runBackendFrame
          });
        backendFrameResult = normalBackendImplementationResult.backendFrameResult;
        executionError = normalBackendImplementationResult.executionError;
      } else {
        backendFrameResult = await runBackendFrame({
          frameIndex,
          invocationSource,
          runnerContract,
          backendImplementationKind: runnerContract.backendImplementationKind
        });
      }
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
    executionError,
    normalBackendImplementation: normalBackendImplementationResult?.summary ?? null
  });
  const runtimeRunnerReady =
    validationSummary.runtimeRunnerReady === true;
  const summary = {
    mode: WEBGPU_BACKEND_RUNTIME_RUNNER_MODE,
    status: runtimeRunnerReady ? 'ok' : 'blocked',
    source:
      'Phase 3 Step70 viewer backend runtime runner executes a normal backend with guarded presentation adapter consumption',
    contractVersion: WEBGPU_BACKEND_RUNTIME_RUNNER_CONTRACT_VERSION,
    runtimeRunnerImplemented: true,
    runtimeRunnerReady,
    productionDisplayConnectionImplemented: false,
    displayConnectionAllowed: false,
    webgl2HybridRenderingAllowed: false,
    runnerContract,
    webgpuNormalBackendFrameImplementation:
      normalBackendImplementationResult?.summary ?? null,
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
      ? 'replace the validation-oracle-backed normal implementation body with production WebGPU backend rendering behind the same runner contract'
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
