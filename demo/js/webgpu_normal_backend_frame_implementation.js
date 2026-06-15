import {
  prepareNormalBackendUniformResources
} from './webgpu_normal_backend_uniform_resources.js';

export const WEBGPU_NORMAL_BACKEND_FRAME_IMPLEMENTATION_MODE =
  'webgpu-normal-backend-frame-implementation';

export const WEBGPU_NORMAL_BACKEND_FRAME_IMPLEMENTATION_CONTRACT_VERSION =
  'phase3-step65-normal-backend-frame-implementation-v1';

export const WEBGPU_NORMAL_BACKEND_FRAME_INPUT_CONTRACT_VERSION =
  'phase3-step65-normal-backend-frame-input-contract-v1';

export const WEBGPU_NORMAL_BACKEND_PRESENT_OUTPUT_CONTRACT_VERSION =
  'phase3-step65-normal-backend-present-output-contract-v1';

export const WEBGPU_NORMAL_BACKEND_FRAME_CONSTANTS_CONTRACT_VERSION =
  'phase3-step65-normal-backend-frame-constants-contract-v1';

export const WEBGPU_NORMAL_BACKEND_UNIFORM_RESOURCE_PREPARATION_CONTRACT_VERSION =
  'phase3-step65-normal-backend-uniform-resource-preparation-contract-v1';

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function finiteNumberArray(value, expectedLength = null) {
  if (!Array.isArray(value)) {
    return null;
  }
  if (expectedLength !== null && value.length !== expectedLength) {
    return null;
  }
  if (!value.every(isFiniteNumber)) {
    return null;
  }
  return value.slice();
}

function compactMatrixSummary(value) {
  const matrix = finiteNumberArray(value, 16);
  return {
    provided: !!matrix,
    elementCount: matrix ? matrix.length : 0,
    values: matrix
  };
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
  viewerCanvasState,
  frameConstantsContract = null,
  uniformResourcePreparationContract = null
}) {
  const projectionContractProvided =
    frameConstantsContract?.projectionContractProvided === true ||
    !!cameraSnapshot?.projectionContract ||
    !!cameraSnapshot?.deterministicState?.projectionContract;
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
    projectionContractProvided,
    frameConstantsProvided:
      frameConstantsContract?.frameConstantsReady === true,
    frameConstantsContractVersion:
      frameConstantsContract?.contractVersion ?? null,
    uniformResourcePreparationProvided:
      uniformResourcePreparationContract?.uniformResourcePreparationReady === true,
    uniformResourcePreparationContractVersion:
      uniformResourcePreparationContract?.contractVersion ?? null,
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

function buildFrameConstantsContract({
  frameIndex,
  invocationSource,
  runnerContract,
  cameraSnapshot,
  viewerCanvasState
}) {
  const frameConstants = cameraSnapshot?.frameConstants ?? {};
  const viewportSource =
    frameConstants.viewport ??
    cameraSnapshot?.viewport ??
    viewerCanvasState?.viewport ??
    null;
  const viewport = {
    x: isFiniteNumber(viewportSource?.x) ? viewportSource.x : 0,
    y: isFiniteNumber(viewportSource?.y) ? viewportSource.y : 0,
    width: isFiniteNumber(viewportSource?.width)
      ? viewportSource.width
      : isFiniteNumber(cameraSnapshot?.canvasWidth)
        ? cameraSnapshot.canvasWidth
        : null,
    height: isFiniteNumber(viewportSource?.height)
      ? viewportSource.height
      : isFiniteNumber(cameraSnapshot?.canvasHeight)
        ? cameraSnapshot.canvasHeight
        : null,
    devicePixelRatio: isFiniteNumber(viewportSource?.devicePixelRatio)
      ? viewportSource.devicePixelRatio
      : null
  };
  const viewMatrix = compactMatrixSummary(
    frameConstants.viewMatrix ?? cameraSnapshot?.viewMatrix
  );
  const projectionMatrix = compactMatrixSummary(
    frameConstants.projectionMatrix ?? cameraSnapshot?.projectionMatrix
  );
  const viewProjectionMatrix = compactMatrixSummary(
    frameConstants.viewProjectionMatrix ?? cameraSnapshot?.viewProjectionMatrix
  );
  const projectionContract =
    cameraSnapshot?.projectionContract ??
    cameraSnapshot?.deterministicState?.projectionContract ??
    null;
  const timeSeconds =
    isFiniteNumber(frameConstants.timeSeconds)
      ? frameConstants.timeSeconds
      : isFiniteNumber(cameraSnapshot?.timeSeconds)
        ? cameraSnapshot.timeSeconds
        : null;
  const cameraPosition = finiteNumberArray(cameraSnapshot?.cameraPosition, 3);
  const cameraQuaternion = finiteNumberArray(cameraSnapshot?.cameraQuaternion, 4);
  const controlsTarget = finiteNumberArray(cameraSnapshot?.controlsTarget, 3);
  const viewportReady =
    isFiniteNumber(viewport.width) &&
    viewport.width > 0 &&
    isFiniteNumber(viewport.height) &&
    viewport.height > 0;
  const frameConstantsReady =
    !!cameraSnapshot &&
    viewportReady &&
    viewMatrix.provided &&
    projectionMatrix.provided &&
    viewProjectionMatrix.provided;
  return {
    contractVersion: WEBGPU_NORMAL_BACKEND_FRAME_CONSTANTS_CONTRACT_VERSION,
    constantsMode: 'normal-webgpu-backend-frame-constants',
    frameIndex,
    invocationSource,
    requestedBackendMode: runnerContract?.requestedBackendMode ?? null,
    normalBackendOwnsFrameConstants: true,
    cameraSnapshotSource: cameraSnapshot
      ? 'viewer-lifecycle-camera-snapshot'
      : 'not-provided',
    cameraSnapshotProvided: !!cameraSnapshot,
    projectionContractProvided:
      !!projectionContract ||
      (projectionMatrix.provided && viewProjectionMatrix.provided),
    projectionContractSource: projectionContract
      ? 'viewer-lifecycle-camera-snapshot'
      : projectionMatrix.provided && viewProjectionMatrix.provided
        ? 'viewer-camera-matrix-derived'
        : 'not-provided',
    viewport,
    timeSeconds,
    frameIndexUniformValue: frameIndex,
    camera: {
      positionProvided: !!cameraPosition,
      quaternionProvided: !!cameraQuaternion,
      controlsTargetProvided: !!controlsTarget,
      position: cameraPosition,
      quaternion: cameraQuaternion,
      controlsTarget
    },
    matrices: {
      viewMatrix,
      projectionMatrix,
      viewProjectionMatrix
    },
    frameConstantsReady,
    productionLoopConnected: false,
    interactiveCameraImplemented: false,
    streamingImplemented: false,
    fullDatasetGpuResidencyRequired: false
  };
}

function buildUniformResourcePreparationContract(frameConstantsContract) {
  const matrices = frameConstantsContract?.matrices ?? {};
  const availableUniformFields = [];
  const missingUniformFields = [];
  const addField = (name, ready) => {
    if (ready) {
      availableUniformFields.push(name);
    } else {
      missingUniformFields.push(name);
    }
  };
  addField('frameIndex', isFiniteNumber(frameConstantsContract?.frameIndex));
  addField('timeSeconds', frameConstantsContract?.timeSeconds !== null);
  addField(
    'viewport',
    isFiniteNumber(frameConstantsContract?.viewport?.width) &&
      isFiniteNumber(frameConstantsContract?.viewport?.height)
  );
  addField('viewMatrix', matrices.viewMatrix?.provided === true);
  addField('projectionMatrix', matrices.projectionMatrix?.provided === true);
  addField('viewProjectionMatrix', matrices.viewProjectionMatrix?.provided === true);
  const uniformFloat32Count = 4 + 16 + 16 + 16;
  const uniformByteLength = uniformFloat32Count * 4;
  const paddedUniformByteLength = 256;
  const uniformResourcePreparationReady =
    frameConstantsContract?.frameConstantsReady === true &&
    missingUniformFields.length === 0;
  return {
    contractVersion:
      WEBGPU_NORMAL_BACKEND_UNIFORM_RESOURCE_PREPARATION_CONTRACT_VERSION,
    resourceMode: 'normal-webgpu-backend-frame-uniform-resource-preparation',
    normalBackendOwnsUniformResourceBoundary: true,
    uniformBufferPlanned: true,
    gpuUniformBufferCreatedThisStep: false,
    gpuUniformBufferUploadExecutedThisStep: false,
    futureGpuBufferUpdateReady: uniformResourcePreparationReady,
    uniformResourcePreparationReady,
    requiredUniformFields: [
      'frameIndex',
      'timeSeconds',
      'viewport',
      'viewMatrix',
      'projectionMatrix',
      'viewProjectionMatrix'
    ],
    availableUniformFields,
    missingUniformFields,
    layout: {
      layoutMode: 'mat4x4-plus-vec4-frame-constants',
      float32Order: [
        'frameIndex_time_viewportWidth_viewportHeight',
        'viewMatrix4x4',
        'projectionMatrix4x4',
        'viewProjectionMatrix4x4'
      ],
      uniformFloat32Count,
      uniformByteLength,
      paddedUniformByteLength,
      minBindingSizeBytes: paddedUniformByteLength,
      alignmentPolicy:
        '256-byte padded uniform buffer binding boundary for future WebGPU updates'
    },
    validationOracleOwnsResource: false,
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

function buildUniformResourceLifecycleSummary(uniformResourceLifecycleContract) {
  return {
    status: uniformResourceLifecycleContract?.status ?? 'unavailable',
    normalBackendOwnsGpuUniformResource:
      uniformResourceLifecycleContract?.normalBackendOwnsGpuUniformResource === true,
    uniformBufferCreated:
      uniformResourceLifecycleContract?.uniformBufferCreated === true,
    uniformBufferWriteSubmitted:
      uniformResourceLifecycleContract?.uniformBufferWriteSubmitted === true,
    uniformBufferWriteCompleted:
      uniformResourceLifecycleContract?.uniformBufferWriteCompleted === true,
    uniformBufferDestroyed:
      uniformResourceLifecycleContract?.uniformBufferDestroyed === true,
    queueWriteBufferUsed:
      uniformResourceLifecycleContract?.queueWriteBufferUsed === true,
    bindGroupReadyBoundary:
      uniformResourceLifecycleContract?.bindGroupReadyBoundary === true,
    deviceOwnershipMode:
      uniformResourceLifecycleContract?.deviceOwnershipMode ?? null,
    packedByteLength:
      uniformResourceLifecycleContract?.packedByteLength ?? null,
    paddedUniformByteLength:
      uniformResourceLifecycleContract?.paddedUniformByteLength ?? null,
    minBindingSizeBytes:
      uniformResourceLifecycleContract?.minBindingSizeBytes ?? null
  };
}

function buildValidationSummary({
  implementationContract,
  frameInputContract,
  presentOutputContract,
  frameConstantsContract,
  uniformResourcePreparationContract,
  uniformResourceLifecycleContract,
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
  const frameConstantsReady =
    frameConstantsContract?.normalBackendOwnsFrameConstants === true &&
    frameConstantsContract?.frameConstantsReady === true &&
    frameConstantsContract?.projectionContractProvided === true;
  const uniformResourcePreparationReady =
    uniformResourcePreparationContract?.normalBackendOwnsUniformResourceBoundary === true &&
    uniformResourcePreparationContract?.uniformResourcePreparationReady === true;
  const uniformResourceLifecycleReady =
    uniformResourceLifecycleContract?.normalBackendOwnsGpuUniformResource === true &&
    uniformResourceLifecycleContract?.uniformBufferCreated === true &&
    uniformResourceLifecycleContract?.uniformBufferWriteSubmitted === true &&
    uniformResourceLifecycleContract?.queueWriteBufferUsed === true &&
    uniformResourceLifecycleContract?.bindGroupReadyBoundary === true;
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
    frameConstantsReady &&
    uniformResourcePreparationReady &&
    uniformResourceLifecycleReady &&
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
  if (!frameConstantsReady) {
    firstValidationFailures.push({
      stage: 'normal-backend-frame-constants',
      reason:
        'normal WebGPU backend implementation requires view/projection/viewport frame constants from the viewer lifecycle'
    });
  }
  if (!uniformResourcePreparationReady) {
    firstValidationFailures.push({
      stage: 'normal-backend-uniform-resource-preparation',
      reason:
        'normal WebGPU backend implementation requires a ready uniform resource preparation boundary for frame constants'
    });
  }
  if (!uniformResourceLifecycleReady) {
    firstValidationFailures.push({
      stage: 'normal-backend-uniform-resource-lifecycle',
      reason:
        'normal WebGPU backend implementation requires GPU uniform buffer creation and queue.writeBuffer ownership'
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
    frameConstantsReady,
    uniformResourcePreparationReady,
    uniformResourceLifecycleReady,
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
  const frameConstantsContract = buildFrameConstantsContract({
    frameIndex,
    invocationSource,
    runnerContract,
    cameraSnapshot,
    viewerCanvasState
  });
  const uniformResourcePreparationContract =
    buildUniformResourcePreparationContract(frameConstantsContract);
  const uniformResourceLifecycleContract =
    await prepareNormalBackendUniformResources({
      frameConstantsContract,
      uniformResourcePreparationContract
    });
  const frameInputContract = buildFrameInputContract({
    frameIndex,
    invocationSource,
    runnerContract,
    cameraSnapshot,
    viewerCanvasState,
    frameConstantsContract,
    uniformResourcePreparationContract
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
        frameInputContract,
        frameConstantsContract,
        uniformResourcePreparationContract,
        uniformResourceLifecycleContract
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
  const uniformResourceLifecycleSummary =
    buildUniformResourceLifecycleSummary(uniformResourceLifecycleContract);
  const validationSummary = buildValidationSummary({
    implementationContract,
    frameInputContract,
    frameConstantsContract,
    uniformResourcePreparationContract,
    uniformResourceLifecycleContract,
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
      'Phase 3 Step65 normal WebGPU backend implementation owns GPU uniform buffer resource lifecycle for frame constants',
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
    frameConstantsContract,
    uniformResourcePreparationContract,
    uniformResourceLifecycleContract,
    uniformResourceLifecycleSummary,
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
              'normal backend implementation path owns frame constants and a GPU uniform resource lifecycle; production scheduling remains intentionally disconnected'
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
              'normal implementation requires exclusive guards, frame constants, GPU uniform resource lifecycle, true native selected samples, submitted present output, and preserved fallback policy'
          }
        ],
    nextBackendPrototypeStep: normalBackendImplementationReady
      ? 'connect the normal backend uniform buffer resource to a bind group and shader input boundary'
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
