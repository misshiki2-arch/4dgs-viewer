export const WEBGPU_NORMAL_BACKEND_FRAME_IMPLEMENTATION_MODE =
  'webgpu-normal-backend-frame-implementation';

export const WEBGPU_NORMAL_BACKEND_FRAME_IMPLEMENTATION_CONTRACT_VERSION =
  'phase3-step63-normal-backend-frame-implementation-v1';

export const WEBGPU_NORMAL_BACKEND_FRAME_INPUT_CONTRACT_VERSION =
  'phase3-step63-normal-backend-frame-input-contract-v1';

export const WEBGPU_NORMAL_BACKEND_PRESENT_OUTPUT_CONTRACT_VERSION =
  'phase3-step63-normal-backend-present-output-contract-v1';

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function buildImplementationContract({
  frameIndex,
  invocationSource,
  runnerContract
}) {
  return {
    contractVersion:
      WEBGPU_NORMAL_BACKEND_FRAME_IMPLEMENTATION_CONTRACT_VERSION,
    implementationMode: 'first-normal-webgpu-backend-frame-path',
    implementationKind: WEBGPU_NORMAL_BACKEND_FRAME_IMPLEMENTATION_MODE,
    invocationSource,
    frameIndex,
    requestedBackendMode: runnerContract?.requestedBackendMode ?? null,
    allowViewerCanvasPresentation:
      runnerContract?.allowViewerCanvasPresentation === true,
    enableViewerLoopHook: runnerContract?.enableViewerLoopHook === true,
    viewerCanvasProvided: runnerContract?.viewerCanvasProvided === true,
    viewerCanvasContextMode: runnerContract?.viewerCanvasContextMode ?? null,
    webgl2FrameLifecycleSuppressed:
      runnerContract?.webgl2FrameLifecycleSuppressed === true,
    cameraSnapshotProvided: runnerContract?.cameraSnapshotProvided === true,
    validationOracleInputKind: 'webgpu-visible-record-dry-run-runtime',
    dryRunRecorderOwnsExecution: false,
    normalBackendOwnsImplementationPath: true,
    productionLoopConnected: false,
    interactiveCameraImplemented: false,
    streamingImplemented: false,
    fullDatasetGpuResidencyRequired: false
  };
}

function buildFrameInputContract({
  frameIndex,
  invocationSource,
  runnerContract,
  cameraSnapshot,
  viewerCanvasState
}) {
  return {
    contractVersion: WEBGPU_NORMAL_BACKEND_FRAME_INPUT_CONTRACT_VERSION,
    inputMode: 'normal-webgpu-backend-viewer-frame-input',
    frameIndex,
    invocationSource,
    requestedBackendMode: runnerContract?.requestedBackendMode ?? null,
    allowViewerCanvasPresentation:
      runnerContract?.allowViewerCanvasPresentation === true,
    enableViewerLoopHook: runnerContract?.enableViewerLoopHook === true,
    cameraSnapshotProvided: !!cameraSnapshot,
    cameraSnapshotSource: cameraSnapshot
      ? 'viewer-lifecycle-camera-snapshot'
      : 'not-provided',
    projectionContractProvided:
      !!cameraSnapshot?.projectionContract ||
      !!cameraSnapshot?.deterministicState?.projectionContract,
    canvasState: {
      provided:
        viewerCanvasState?.provided === true ||
        runnerContract?.viewerCanvasProvided === true,
      contextMode:
        viewerCanvasState?.contextMode ??
        runnerContract?.viewerCanvasContextMode ??
        null,
      requestedBackendMode:
        viewerCanvasState?.requestedBackendMode ??
        runnerContract?.requestedBackendMode ??
        null,
      allowViewerCanvasPresentation:
        viewerCanvasState?.allowViewerCanvasPresentation === true ||
        runnerContract?.allowViewerCanvasPresentation === true,
      webgl2FrameLifecycleSuppressed:
        viewerCanvasState?.webgl2FrameLifecycleSuppressed === true ||
        runnerContract?.webgl2FrameLifecycleSuppressed === true
    },
    guardState: {
      exclusiveBackendMode:
        runnerContract?.requestedBackendMode === 'webgpu-exclusive',
      viewerCanvasPresentationAllowed:
        runnerContract?.allowViewerCanvasPresentation === true,
      viewerLoopHookEnabled: runnerContract?.enableViewerLoopHook === true,
      webgl2FrameLifecycleSuppressed:
        runnerContract?.webgl2FrameLifecycleSuppressed === true
    },
    sourceSelectionPolicy: {
      mode: 'normal-backend-owned-source-selection-contract',
      preferredSourceKind: 'step40-constrained-display-adapter',
      fallbackKeptForEmptySelectorSamples: true,
      fallbackMustNotMixWithSelectedSamples: true,
      trueNativeRequiredForSuccess: true
    },
    validationOracleRole:
      'dry-run recorder may provide comparison/debug observations, but normal backend owns this frame input contract',
    normalBackendOwnsFrameInput: true,
    productionLoopConnected: false,
    interactiveCameraImplemented: false,
    streamingImplemented: false,
    fullDatasetGpuResidencyRequired: false
  };
}

function buildPresentOutputContract(backendFrameResult) {
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
    contractVersion: WEBGPU_NORMAL_BACKEND_PRESENT_OUTPUT_CONTRACT_VERSION,
    outputMode: 'normal-webgpu-backend-present-output',
    normalBackendOwnsPresentOutput: true,
    selectedSourceKind:
      backendFrameResult?.webgpuViewerCanvasBoundedColorSourceSelector
        ?.selectedSourceKind ??
      colorOutputContract.selectorSourceKind ??
      null,
    selectionMode: colorOutputContract.selectionMode ?? null,
    colorPresentSampleCount: present?.colorPresentSampleCount ?? null,
    commandBufferSubmitted: present?.commandBufferSubmitted === true,
    submittedWorkDone: present?.submittedWorkDone === true,
    viewerCanvasContextConfiguredForColorPresent:
      present?.viewerCanvasContextConfiguredForColorPresent === true,
    selectorSelectedSamplesUsed:
      colorOutputContract.selectorSelectedSamplesUsed === true,
    fallbackSuppressedBySelectorSamples:
      colorOutputContract.fallbackSuppressedBySelectorSamples === true,
    presentedSampleCount: presentedSamples.length,
    sampleSources,
    presentedSamples,
    containsRenderHandoffFallback:
      sampleSources.includes('webgpuRenderHandoffStub.sampleRecords') ||
      presentedSamples.some(
        (sample) => sample?.source === 'webgpuRenderHandoffStub.sampleRecords'
      ),
    validationOracleSource:
      'webgpu-visible-record-dry-run-runtime-observed-present-output',
    fallbackPolicy: {
      fallbackKeptForEmptySelectorSamples:
        colorOutputContract.fallbackKeptForEmptySelectorSamples === true,
      fallbackAllowedForThisFrame:
        colorOutputContract.fallbackAllowedForThisFrame === true,
      fallbackSuppressedBySelectorSamples:
        colorOutputContract.fallbackSuppressedBySelectorSamples === true,
      selectedSamplesClassifiedAsTrueNative:
        colorOutputContract.selectedSamplesClassifiedAsTrueNative === true ||
        backendFrameResult?.webgpuViewerCanvasBoundedColorSourceSelector
          ?.selectedSourceKind === 'step40-constrained-display-adapter'
    }
  };
}

function buildResourceSummary(backendFrameResult) {
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
    frameIndices: frameSummaries.map((frame) => frame.frameIndex),
    allFramesSubmitted:
      repeated.validationSummary?.allFramesSubmitted === true,
    perFrameCurrentTextureAcquired: frameSummaries.every(
      (frame) => frame.currentTextureAcquisition === true
    ),
    perFrameSubmitCompleted: frameSummaries.every(
      (frame) => frame.colorPresentSubmittedWorkDone === true
    )
  };
}

function buildValidationSummary({
  implementationContract,
  frameInputContract,
  presentOutputContract,
  resourceSummary,
  backendFrameResult,
  executionError
}) {
  const guardAllowed =
    implementationContract.requestedBackendMode === 'webgpu-exclusive' &&
    implementationContract.allowViewerCanvasPresentation === true &&
    implementationContract.enableViewerLoopHook === true;
  const frameInputReady =
    frameInputContract.normalBackendOwnsFrameInput === true &&
    frameInputContract.guardState?.exclusiveBackendMode === true &&
    frameInputContract.guardState?.viewerCanvasPresentationAllowed === true &&
    frameInputContract.guardState?.viewerLoopHookEnabled === true &&
    frameInputContract.canvasState?.webgl2FrameLifecycleSuppressed === true;
  const backendFrameResultProvided = !!backendFrameResult;
  const selectedTrueNativeSource =
    presentOutputContract.selectedSourceKind ===
    'step40-constrained-display-adapter';
  const presentReady =
    presentOutputContract.normalBackendOwnsPresentOutput === true &&
    presentOutputContract.commandBufferSubmitted &&
    presentOutputContract.submittedWorkDone &&
    presentOutputContract.colorPresentSampleCount === 2 &&
    presentOutputContract.presentedSampleCount === 2;
  const noFallbackMixing =
    presentOutputContract.containsRenderHandoffFallback === false &&
    presentOutputContract.fallbackSuppressedBySelectorSamples === true;
  const resourceLifecycleReady =
    resourceSummary.executedBackendFrameSubmissions === 3 &&
    resourceSummary.repeatedSubmitCount === 3 &&
    resourceSummary.allFramesSubmitted === true;
  const webgl2HybridRenderingPrevented =
    implementationContract.webgl2FrameLifecycleSuppressed === true;
  const normalBackendImplementationReady =
    guardAllowed &&
    frameInputReady &&
    backendFrameResultProvided &&
    selectedTrueNativeSource &&
    presentReady &&
    noFallbackMixing &&
    resourceLifecycleReady &&
    webgl2HybridRenderingPrevented &&
    !executionError;
  const firstValidationFailures = [];
  if (!guardAllowed) {
    firstValidationFailures.push({
      stage: 'normal-backend-guard',
      reason:
        'normal WebGPU backend implementation requires webgpu-exclusive, viewer canvas presentation guard, and webgpuBackendViewerLoopHook=true'
    });
  }
  if (!backendFrameResultProvided) {
    firstValidationFailures.push({
      stage: 'normal-backend-frame-result',
      reason: 'normal WebGPU backend implementation did not receive backend frame output'
    });
  }
  if (!frameInputReady) {
    firstValidationFailures.push({
      stage: 'normal-backend-frame-input',
      reason:
        'normal WebGPU backend implementation requires an owned frame input contract with exclusive canvas guard state'
    });
  }
  if (!selectedTrueNativeSource) {
    firstValidationFailures.push({
      stage: 'normal-backend-source-selection',
      reason:
        'normal WebGPU backend implementation expects the Step40 constrained display adapter true native source'
    });
  }
  if (!presentReady) {
    firstValidationFailures.push({
      stage: 'normal-backend-present',
      reason:
        'normal WebGPU backend implementation requires submitted bounded color present output'
    });
  }
  if (!noFallbackMixing) {
    firstValidationFailures.push({
      stage: 'normal-backend-fallback-policy',
      reason:
        'normal WebGPU backend implementation cannot mix render-handoff fallback with selected native samples'
    });
  }
  if (!resourceLifecycleReady) {
    firstValidationFailures.push({
      stage: 'normal-backend-resource-lifecycle',
      reason:
        'normal WebGPU backend implementation expects three controlled frame submissions'
    });
  }
  if (!webgl2HybridRenderingPrevented) {
    firstValidationFailures.push({
      stage: 'normal-backend-webgl2-suppression',
      reason:
        'normal WebGPU backend implementation cannot run while WebGL2 owns the viewer canvas frame lifecycle'
    });
  }
  if (executionError) {
    firstValidationFailures.push({
      stage: 'normal-backend-execution-error',
      reason: executionError.message ?? 'normal WebGPU backend implementation threw'
    });
  }
  return {
    normalBackendImplementationReady,
    guardAllowed,
    frameInputReady,
    backendFrameResultProvided,
    selectedTrueNativeSource,
    presentReady,
    noFallbackMixing,
    resourceLifecycleReady,
    webgl2HybridRenderingPrevented,
    firstValidationFailures
  };
}

export async function runWebGpuNormalBackendFrameImplementation({
  frameIndex = 0,
  invocationSource = 'webgpu-normal-backend-frame-implementation',
  runnerContract = null,
  cameraSnapshot = null,
  viewerCanvasState = null,
  runBackendFrame = null
} = {}) {
  const startMs = nowMs();
  const implementationContract = buildImplementationContract({
    frameIndex,
    invocationSource,
    runnerContract
  });
  const frameInputContract = buildFrameInputContract({
    frameIndex,
    invocationSource,
    runnerContract,
    cameraSnapshot,
    viewerCanvasState
  });
  let backendFrameResult = null;
  let executionError = null;
  if (typeof runBackendFrame === 'function') {
    try {
      backendFrameResult = await runBackendFrame({
        frameIndex,
        invocationSource,
        runnerContract,
        backendImplementationKind:
          WEBGPU_NORMAL_BACKEND_FRAME_IMPLEMENTATION_MODE,
        implementationContract,
        frameInputContract
      });
    } catch (error) {
      executionError = {
        name: error?.name ?? 'Error',
        message: error?.message ?? String(error)
      };
    }
  }
  const presentOutputContract = buildPresentOutputContract(backendFrameResult);
  const resourceSummary = buildResourceSummary(backendFrameResult);
  const validationSummary = buildValidationSummary({
    implementationContract,
    frameInputContract,
    presentOutputContract,
    resourceSummary,
    backendFrameResult,
    executionError
  });
  const normalBackendImplementationReady =
    validationSummary.normalBackendImplementationReady === true;
  const summary = {
    mode: WEBGPU_NORMAL_BACKEND_FRAME_IMPLEMENTATION_MODE,
    status: normalBackendImplementationReady ? 'ok' : 'blocked',
    source:
      'Phase 3 Step63 normal WebGPU backend implementation owns frame input and present output contracts',
    contractVersion:
      WEBGPU_NORMAL_BACKEND_FRAME_IMPLEMENTATION_CONTRACT_VERSION,
    implementationKind: WEBGPU_NORMAL_BACKEND_FRAME_IMPLEMENTATION_MODE,
    normalBackendImplementationImplemented: true,
    normalBackendImplementationReady,
    productionDisplayConnectionImplemented: false,
    displayConnectionAllowed: false,
    webgl2HybridRenderingAllowed: false,
    implementationContract,
    frameInputContract,
    presentOutputContract,
    presentSummary: presentOutputContract,
    resourceSummary,
    selectedSourceKind: presentOutputContract.selectedSourceKind,
    selectionMode: presentOutputContract.selectionMode,
    colorPresentSampleCount: presentOutputContract.colorPresentSampleCount,
    presentedSampleCount: presentOutputContract.presentedSampleCount,
    sampleSources: presentOutputContract.sampleSources,
    presentedSamples: presentOutputContract.presentedSamples,
    executedBackendFrameSubmissions:
      resourceSummary.executedBackendFrameSubmissions,
    repeatedSubmitCount: resourceSummary.repeatedSubmitCount,
    validationSummary,
    executionError,
    firstValidationFailures: validationSummary.firstValidationFailures,
    blockers: normalBackendImplementationReady
      ? [
          {
            stage: 'production-frame-scheduler',
            reason:
              'normal backend implementation path is callable; production scheduling remains intentionally disconnected'
          },
          {
            stage: 'streaming-lod',
            reason:
              'streaming, chunking, LOD, and partial upload remain future backend policies'
          },
          {
            stage: 'sh-color-evaluation',
            reason: 'WGSL SH/color evaluation parity remains deferred'
          }
        ]
      : [
          {
            stage: 'normal-webgpu-backend-implementation',
            reason:
              'normal implementation requires exclusive guards, true native selected samples, submitted present output, and preserved fallback policy'
          }
        ],
    nextBackendPrototypeStep: normalBackendImplementationReady
      ? 'replace the validation-oracle-backed bounded present body with production WebGPU backend rendering behind the same implementation contract'
      : 'restore normal backend implementation readiness before replacing the implementation body',
    timing: {
      webgpuNormalBackendFrameImplementationMs: nowMs() - startMs
    }
  };
  return {
    summary,
    backendFrameResult,
    executionError
  };
}
