export const WEBGPU_NORMAL_BACKEND_OUTPUT_HANDOFF_CONTRACT_VERSION =
  'phase3-step69-normal-backend-output-presentation-handoff-v1';

export const WEBGPU_NORMAL_BACKEND_PRESENTATION_HANDOFF_CONTRACT_VERSION =
  'phase3-step69-normal-backend-presentation-handoff-boundary-v1';

export const WEBGPU_NORMAL_BACKEND_GUARDED_PRESENTATION_ADAPTER_CONTRACT_VERSION =
  'phase3-step70-webgpu-only-guarded-presentation-adapter-v1';

export const WEBGPU_NORMAL_BACKEND_PRESENTATION_BRIDGE_CONTRACT_VERSION =
  'phase3-step72-viewer-canvas-current-texture-presentation-bridge-v1';

export const WEBGPU_VIEWER_FRAME_PRESENTATION_PASS_CONTRACT_VERSION =
  'phase3-step73-viewer-frame-lifecycle-owned-presentation-pass-v1';

export const WEBGPU_SCHEDULER_FRAME_PRESENTATION_BOUNDARY_CONTRACT_VERSION =
  'phase3-step74-scheduler-owned-guarded-webgpu-frame-presentation-boundary-v1';

export const WEBGPU_CAMERA_AWARE_VISIBLE_OUTPUT_CONTRACT_VERSION =
  'phase3-step76-camera-control-scheduler-aware-visible-output-v2';

const DEFAULT_FUTURE_PRESENTATION_TARGETS = [
  'viewer-canvas-current-texture',
  'render-target-texture',
  'storage-texture-copy'
];

function arrayFrom(value) {
  if (Array.isArray(value)) return value;
  if (ArrayBuffer.isView(value) && typeof value.length === 'number') {
    return Array.from(value);
  }
  return [];
}

function finiteNumberOr(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function maxAbsDiff(actual, expected) {
  const actualArray = arrayFrom(actual);
  const expectedArray = arrayFrom(expected);
  const length = Math.max(actualArray.length, expectedArray.length);
  let diff = 0;
  for (let i = 0; i < length; i++) {
    diff = Math.max(
      diff,
      Math.abs(finiteNumberOr(actualArray[i], 0) - finiteNumberOr(expectedArray[i], 0))
    );
  }
  return diff;
}

function convertRgbaBytesForTextureFormat(bytes, textureFormat) {
  const source = arrayFrom(bytes);
  if (textureFormat !== 'bgra8unorm') return source;
  const converted = source.slice();
  for (let i = 0; i + 3 < converted.length; i += 4) {
    const r = converted[i];
    converted[i] = converted[i + 2];
    converted[i + 2] = r;
  }
  return converted;
}

export function buildCameraAwareVisibleOutputContract({
  status = 'ok',
  sourceMode = 'webgpu-visible-record-camera-aware-samples',
  step = 'phase3-step76',
  selectedApproach = null,
  inputSourceKind = null,
  inputSourceLineage = null,
  sourceClassification = 'true-native',
  sampleCount = 0,
  maxSampleCount = 0,
  candidateRecordCount = null,
  validRecordCount = null,
  visibleRecordSampleCount = null,
  visibleInputSampleCount = null,
  renderedSamplePatchCount = null,
  bridgeGeneratedSampleCount = null,
  strictProjectedSampleCount = null,
  bridgeProjectionFallbackCount = null,
  bridgeInvalidStateFallbackCount = null,
  bridgeProjectionRejectedCount = null,
  bridgeGenerationReason = null,
  consumedSourceKind = null,
  consumedSourceLineage = null,
  consumedSourceClassification = null,
  consumedSampleCount = null,
  enlargedPatchPixelCount = null,
  outputPointRadiusPx = 0,
  visibleSamples = [],
  debugFillUsed = false,
  cameraProjectionDerivedPositions = true,
  cameraSnapshotProvided = false,
  projectionContractProvided = false,
  frameConstantsReady = false,
  schedulerFramePresentationBoundaryReady = false,
  schedulerOwnsFrameRequest = false,
  currentTextureConnected = false,
  currentTextureRenderPassSubmitted = false,
  currentTextureReadbackMatchesAdapterOutput = false,
  webgl2HybridRenderingAllowed = false,
  fallbackSamplesMixed = false,
  reason = null
} = {}) {
  const cameraAwareVisibleOutputReady =
    status === 'ok' &&
    sampleCount > 0 &&
    cameraSnapshotProvided === true &&
    projectionContractProvided === true &&
    frameConstantsReady === true &&
    schedulerFramePresentationBoundaryReady === true &&
    schedulerOwnsFrameRequest === true &&
    currentTextureConnected === true &&
    currentTextureRenderPassSubmitted === true &&
    currentTextureReadbackMatchesAdapterOutput === true &&
    webgl2HybridRenderingAllowed === false &&
    fallbackSamplesMixed === false;
  return {
    contractVersion: WEBGPU_CAMERA_AWARE_VISIBLE_OUTPUT_CONTRACT_VERSION,
    status: cameraAwareVisibleOutputReady ? 'ok' : status,
    outputMode: 'camera-control-scheduler-aware-visible-webgpu-output',
    step,
    selectedApproach,
    cameraAwareVisibleOutputReady,
    visibleOutputUsesCameraProjection: true,
    cameraProjectionDerivedPositions,
    visibleOutputUsesSchedulerOwnedFramePath: true,
    visibleOutputUsesCurrentTexturePath: true,
    sourceMode,
    inputSourceKind,
    inputSourceLineage,
    sourceClassification,
    consumedSourceKind: consumedSourceKind ?? inputSourceKind,
    consumedSourceLineage: consumedSourceLineage ?? inputSourceLineage,
    consumedSourceClassification:
      consumedSourceClassification ?? sourceClassification,
    consumedSampleCount: consumedSampleCount ?? sampleCount,
    sampleCount,
    maxSampleCount,
    candidateRecordCount,
    validRecordCount,
    visibleRecordSampleCount,
    visibleInputSampleCount: visibleInputSampleCount ?? sampleCount,
    renderedSamplePatchCount: renderedSamplePatchCount ?? sampleCount,
    bridgeGeneratedSampleCount,
    strictProjectedSampleCount,
    bridgeProjectionFallbackCount,
    bridgeInvalidStateFallbackCount,
    bridgeProjectionRejectedCount,
    bridgeGenerationReason,
    enlargedPatchPixelCount,
    outputPointRadiusPx,
    debugFillUsed,
    cameraSnapshotProvided,
    projectionContractProvided,
    frameConstantsReady,
    schedulerFramePresentationBoundaryReady,
    schedulerOwnsFrameRequest,
    currentTextureConnected,
    currentTextureRenderPassSubmitted,
    currentTextureReadbackMatchesAdapterOutput,
    webgl2HybridRenderingAllowed,
    fallbackSamplesMixed,
    samplePreview: visibleSamples.slice(0, 8),
    successCriteria: [
      'viewer camera/projection produces samplePx positions',
      'normal backend consumes visible samples as GPU storage input',
      'GPU compute writes an enlarged visible color output surface',
      'guarded adapter presents that surface through currentTexture',
      'WebGL2 rendering is not mixed into the same display frame'
    ],
    reason
  };
}

export function buildUnavailableCameraAwareVisibleOutputContract(
  reason,
  extra = {}
) {
  return buildCameraAwareVisibleOutputContract({
    status: 'unavailable',
    reason,
    ...extra
  });
}

export function buildPresentationHandoffContract({
  status = 'ok',
  futurePresentationTargets = DEFAULT_FUTURE_PRESENTATION_TARGETS,
  productionPresentationConnected = false,
  viewerCanvasCurrentTextureConnected = false,
  renderTargetTextureConnected = false,
  storageTextureCopyConnected = false,
  webgl2HybridRenderingAllowed = false,
  reason = null
} = {}) {
  const presentationHandoffReady =
    status === 'ok' &&
    productionPresentationConnected === false &&
    viewerCanvasCurrentTextureConnected === false;
  return {
    contractVersion: WEBGPU_NORMAL_BACKEND_PRESENTATION_HANDOFF_CONTRACT_VERSION,
    status: presentationHandoffReady ? 'ok' : status,
    presentationHandoffReady,
    futurePresentationTargets,
    viewerCanvasCurrentTextureConnected,
    renderTargetTextureConnected,
    storageTextureCopyConnected,
    productionPresentationConnected,
    webgl2HybridRenderingAllowed,
    reason
  };
}

export function buildUnavailableNormalBackendOutputContracts(reason, extra = {}) {
  const presentationHandoffContract = buildPresentationHandoffContract({
    status: 'unavailable',
    reason
  });
  return {
    normalBackendOutputContract: {
      contractVersion: WEBGPU_NORMAL_BACKEND_OUTPUT_HANDOFF_CONTRACT_VERSION,
      status: 'unavailable',
      outputMode: 'normal-backend-current-frame-output',
      normalBackendOwnsCurrentFrameOutput: true,
      gpuSideHandoffCopySubmitted: false,
      handoffReadbackCompleted: false,
      handoffReadbackMatchesExpected: false,
      handoffReadbackMatchesColorOutputSurface: false,
      normalBackendOutputMatchesExpected: false,
      normalBackendOutputReady: false,
      presentationHandoffContract,
      reason,
      ...extra
    },
    presentationHandoffContract
  };
}

export function buildNormalBackendOutputContract({
  expectedSurfaceSummary,
  expectedSurfaceData,
  colorOutputSurfaceReadback,
  handoffReadback,
  sampleCount,
  sourceKind = 'step40-constrained-display-adapter',
  fallbackSamplesMixed = false,
  epsilon = 1e-6
} = {}) {
  const expectedArray = arrayFrom(expectedSurfaceData);
  const colorSurfaceArray = arrayFrom(colorOutputSurfaceReadback);
  const handoffArray = arrayFrom(handoffReadback);
  const normalBackendOutputMaxAbsDiff = maxAbsDiff(handoffArray, expectedArray);
  const handoffColorSurfaceMaxAbsDiff = maxAbsDiff(handoffArray, colorSurfaceArray);
  const handoffReadbackMatchesExpected = normalBackendOutputMaxAbsDiff <= epsilon;
  const handoffReadbackMatchesColorOutputSurface =
    handoffColorSurfaceMaxAbsDiff <= epsilon;
  const presentationHandoffContract = buildPresentationHandoffContract({
    status:
      handoffReadbackMatchesExpected &&
      handoffReadbackMatchesColorOutputSurface &&
      fallbackSamplesMixed === false
        ? 'ok'
        : 'mismatch'
  });
  const normalBackendOutputReady =
    handoffReadbackMatchesExpected &&
    handoffReadbackMatchesColorOutputSurface &&
    fallbackSamplesMixed === false &&
    presentationHandoffContract.presentationHandoffReady === true;
  const outputContract = {
    contractVersion: WEBGPU_NORMAL_BACKEND_OUTPUT_HANDOFF_CONTRACT_VERSION,
    status: normalBackendOutputReady ? 'ok' : 'mismatch',
    outputMode: 'normal-backend-current-frame-output',
    normalBackendOwnsCurrentFrameOutput: true,
    sourceSurfaceMode: expectedSurfaceSummary?.outputSurfaceMode ?? null,
    sourceSurfaceKind: expectedSurfaceSummary?.outputResourceKind ?? null,
    handoffResourceKind: 'storage-buffer-rgba-float-handoff',
    outputFormat: expectedSurfaceSummary?.outputFormat ?? 'rgba32float',
    outputWidth: expectedSurfaceSummary?.surfaceWidth ?? null,
    outputHeight: expectedSurfaceSummary?.surfaceHeight ?? null,
    outputExtent: expectedSurfaceSummary?.outputExtent ?? null,
    coordinateOrigin: expectedSurfaceSummary?.coordinateOrigin ?? null,
    coordinateMapping: expectedSurfaceSummary?.coordinateMapping ?? null,
    sourceSurfaceOriginPx: expectedSurfaceSummary?.surfaceOriginPx ?? null,
    sourceSurfacePixelCount: expectedSurfaceSummary?.surfacePixelCount ?? null,
    packedFloat32Count: expectedArray.length,
    packedByteLength: expectedArray.length * 4,
    gpuSideHandoffConnection:
      'copyBufferToBuffer-color-output-surface-to-handoff-buffer',
    gpuSideHandoffCopySubmitted: true,
    handoffReadbackCompleted: handoffArray.length > 0,
    handoffReadbackMatchesExpected,
    handoffReadbackMatchesColorOutputSurface,
    normalBackendOutputMatchesExpected: handoffReadbackMatchesExpected,
    normalBackendOutputReady,
    normalBackendOutputMaxAbsDiff,
    handoffColorSurfaceMaxAbsDiff,
    expectedFirstPixels: expectedArray.slice(0, 16),
    handoffReadbackFirstPixels: handoffArray.slice(0, 16),
    outputResourceLifecycle: {
      created: true,
      gpuSideConnectedFromColorOutputSurface: true,
      readbackCompleted: handoffArray.length > 0,
      destroyed: true,
      owner: 'webgpu-normal-backend-frame-implementation',
      ownershipPolicy:
        'normal backend owns this frame output until a future presentation adapter accepts an explicit handoff'
    },
    presentationHandoffContract,
    inputSamples: {
      sampleCount,
      sourceKind,
      fallbackSamplesMixed,
      packedFieldInputs: expectedSurfaceSummary?.packedFieldInputs ?? []
    },
    fullDatasetGpuResidencyRequired: false
  };
  return {
    normalBackendOutputContract: outputContract,
    presentationHandoffContract
  };
}

export function validateNormalBackendOutputContracts({
  normalBackendOutputContract,
  presentationHandoffContract
} = {}) {
  const normalBackendOutputReady =
    normalBackendOutputContract?.normalBackendOwnsCurrentFrameOutput === true &&
    normalBackendOutputContract?.gpuSideHandoffCopySubmitted === true &&
    normalBackendOutputContract?.handoffReadbackCompleted === true &&
    normalBackendOutputContract?.handoffReadbackMatchesExpected === true &&
    normalBackendOutputContract?.handoffReadbackMatchesColorOutputSurface === true &&
    normalBackendOutputContract?.normalBackendOutputMatchesExpected === true;
  const presentationHandoffReady =
    presentationHandoffContract?.presentationHandoffReady === true &&
    presentationHandoffContract?.productionPresentationConnected === false &&
    presentationHandoffContract?.viewerCanvasCurrentTextureConnected === false;
  return {
    normalBackendOutputReady,
    presentationHandoffReady,
    status:
      normalBackendOutputReady && presentationHandoffReady ? 'ok' : 'blocked'
  };
}

export function buildUnavailableGuardedPresentationAdapterContract(
  reason,
  extra = {}
) {
  const presentationBridgeContract =
    extra.presentationBridgeContract ??
    buildUnavailablePresentationBridgeContract(reason);
  return {
    contractVersion:
      WEBGPU_NORMAL_BACKEND_GUARDED_PRESENTATION_ADAPTER_CONTRACT_VERSION,
    status: 'unavailable',
    adapterMode: 'webgpu-only-guarded-presentation-adapter',
    guardedPresentationAdapterReady: false,
    webgpuOnlyGuardedPresentationAdapterCalled: false,
    normalBackendOutputConsumed: false,
    handoffResourceConsumedGpuSide: false,
    presentationCompatibleTargetCreated: false,
    presentationTargetWriteSubmitted: false,
    presentationTargetReadbackCompleted: false,
    presentationTargetMatchesExpected: false,
    presentationBridgeContract,
    viewerPresentationBridgeReady: false,
    currentTextureConnectionAttempted: false,
    currentTextureConnected: false,
    currentTextureBlockedReason: reason,
    renderTargetBridgeReady: false,
    renderTargetTextureConnected: false,
    productionCanvasPresentationConnected: false,
    viewerCanvasCurrentTextureConnected: false,
    webgl2HybridRenderingAllowed: false,
    reason,
    ...extra
  };
}

export function buildGuardedPresentationAdapterContract({
  normalBackendOutputContract,
  presentationHandoffContract,
  targetFormat = 'rgba8unorm',
  targetKind = 'offscreen-storage-texture-presentation-compatible-target',
  targetWidth = 0,
  targetHeight = 0,
  expectedBytes,
  readbackBytes,
  gpuWriteSubmitted = false,
  readbackCompleted = false,
  submittedWorkDone = false,
  epsilon = 0
} = {}) {
  const expectedArray = arrayFrom(expectedBytes);
  const readbackArray = arrayFrom(readbackBytes);
  const targetMaxAbsDiff = maxAbsDiff(readbackArray, expectedArray);
  const presentationTargetMatchesExpected =
    readbackCompleted === true && targetMaxAbsDiff <= epsilon;
  const presentationTargetReadable =
    readbackCompleted === true && readbackArray.length > 0;
  const normalBackendOutputReady =
    normalBackendOutputContract?.normalBackendOutputReady === true;
  const handoffReady =
    presentationHandoffContract?.presentationHandoffReady === true;
  const guardedPresentationAdapterReady =
    normalBackendOutputReady &&
    handoffReady &&
    gpuWriteSubmitted === true &&
    presentationTargetReadable;
  const reason = guardedPresentationAdapterReady
    ? null
    : 'guarded-presentation-adapter-validation-not-ready';
  return {
    contractVersion:
      WEBGPU_NORMAL_BACKEND_GUARDED_PRESENTATION_ADAPTER_CONTRACT_VERSION,
    status: guardedPresentationAdapterReady ? 'ok' : 'blocked',
    adapterMode: 'webgpu-only-guarded-presentation-adapter',
    guardedPresentationAdapterReady,
    webgpuOnlyGuardedPresentationAdapterCalled: true,
    normalBackendOutputConsumed: normalBackendOutputReady,
    handoffResourceConsumedGpuSide: gpuWriteSubmitted === true,
    sourceOutputContractVersion:
      normalBackendOutputContract?.contractVersion ?? null,
    sourcePresentationHandoffContractVersion:
      presentationHandoffContract?.contractVersion ?? null,
    sourceHandoffResourceKind:
      normalBackendOutputContract?.handoffResourceKind ?? null,
    sourceFormat: normalBackendOutputContract?.outputFormat ?? null,
    sourceWidth: normalBackendOutputContract?.outputWidth ?? null,
    sourceHeight: normalBackendOutputContract?.outputHeight ?? null,
    sourceCoordinateOrigin:
      normalBackendOutputContract?.coordinateOrigin ?? null,
    sourceCoordinateMapping:
      normalBackendOutputContract?.coordinateMapping ?? null,
    targetResourceKind: targetKind,
    targetFormat,
    targetWidth,
    targetHeight,
    targetExtent: { width: targetWidth, height: targetHeight },
    targetCoordinateOrigin:
      normalBackendOutputContract?.coordinateOrigin ?? null,
    targetCoordinateMapping:
      'adapter writes normalized handoff rgba into the same bounded texture coordinates',
    presentationCompatibleTargetCreated: targetWidth > 0 && targetHeight > 0,
    presentationTargetWriteSubmitted: gpuWriteSubmitted === true,
    presentationTargetReadbackCompleted: readbackCompleted === true,
    presentationTargetReadable,
    presentationTargetMatchesExpected,
    presentationTargetMaxAbsDiff: targetMaxAbsDiff,
    expectedFirstBytes: expectedArray.slice(0, 16),
    readbackFirstBytes: readbackArray.slice(0, 16),
    submittedWorkDone,
    gpuCommandPath:
      'compute-pass-read-normal-backend-handoff-buffer-write-offscreen-storage-texture',
    futurePresentationTargets:
      presentationHandoffContract?.futurePresentationTargets ??
      DEFAULT_FUTURE_PRESENTATION_TARGETS,
    productionCanvasPresentationConnected: false,
    viewerCanvasCurrentTextureConnected: false,
    renderTargetTextureConnected: false,
    storageTextureCopyConnected: false,
    webgl2HybridRenderingAllowed: false,
    fallbackSamplesMixed:
      normalBackendOutputContract?.inputSamples?.fallbackSamplesMixed === true,
    productionShaderImplemented: false,
    shColorParityImplemented: false,
    streamingImplemented: false,
    fullDatasetGpuResidencyRequired: false,
    reason
  };
}

export function buildUnavailablePresentationBridgeContract(reason, extra = {}) {
  return {
    contractVersion: WEBGPU_NORMAL_BACKEND_PRESENTATION_BRIDGE_CONTRACT_VERSION,
    status: 'unavailable',
    bridgeMode: 'viewer-canvas-current-texture-or-render-target-bridge',
    viewerPresentationBridgeReady: false,
    guardedPresentationAdapterOutputConsumed: false,
    currentTextureConnectionAttempted: false,
    currentTextureConnected: false,
    currentTextureContextProvided: false,
    currentTextureConfigured: false,
    currentTextureAcquired: false,
    currentTextureRenderPassSubmitted: false,
    currentTextureReadbackCompleted: false,
    currentTextureReadbackMatchesAdapterOutput: false,
    currentTextureConnectionMode: 'not-attempted',
    currentTextureBlockedReason: reason,
    renderTargetBridgeReady: false,
    renderTargetTextureCreated: false,
    renderTargetGpuCopySubmitted: false,
    renderTargetReadbackCompleted: false,
    renderTargetMatchesAdapterOutput: false,
    productionCanvasPresentationConnected: false,
    viewerCanvasCurrentTextureConnected: false,
    webgl2HybridRenderingAllowed: false,
    reason,
    ...extra
  };
}

export function buildUnavailableViewerFramePresentationPassContract(
  reason,
  extra = {}
) {
  return {
    contractVersion: WEBGPU_VIEWER_FRAME_PRESENTATION_PASS_CONTRACT_VERSION,
    status: 'unavailable',
    presentationPassMode:
      'viewer-frame-lifecycle-owned-guarded-webgpu-presentation-pass',
    viewerFramePresentationPassReady: false,
    calledFromViewerFrameLifecycle: false,
    calledFromExecutorChain: false,
    debugCaptureOwnsPresentationPass: false,
    viewerOwnsCurrentTextureLifecycle: false,
    currentTextureConnectionAttempted: false,
    currentTextureConnected: false,
    currentTextureAcquired: false,
    currentTextureRenderPassSubmitted: false,
    currentTextureReadbackCompleted: false,
    currentTextureReadbackMatchesAdapterOutput: false,
    webgl2HybridRenderingAllowed: false,
    fallbackSamplesMixed: false,
    productionSchedulerConnected: false,
    reason,
    ...extra
  };
}

export function buildViewerFramePresentationPassContract({
  executorContract,
  runtimeRunner,
  presentationBridgeContract,
  invocationSource = 'viewer-frame-lifecycle-presentation-pass',
  frameIndex = 0
} = {}) {
  const currentTextureConnected =
    presentationBridgeContract?.currentTextureConnected === true;
  const currentTextureReadbackMatchesAdapterOutput =
    presentationBridgeContract?.currentTextureReadbackMatchesAdapterOutput ===
    true;
  const calledFromViewerFrameLifecycle =
    executorContract?.callableFromViewerLifecycle === true &&
    String(invocationSource).includes('renderCurrentFrame');
  const calledFromExecutorChain =
    executorContract?.directBackendRunner === 'webgpuBackendRuntimeRunner' &&
    runtimeRunner?.runtimeRunnerReady === true;
  const guardAllowed =
    executorContract?.requestedBackendMode === 'webgpu-exclusive' &&
    executorContract?.allowViewerCanvasPresentation === true &&
    executorContract?.enableViewerLoopHook === true;
  const webgl2HybridRenderingPrevented =
    presentationBridgeContract?.webgl2HybridRenderingAllowed === false;
  const fallbackSamplesMixed =
    presentationBridgeContract?.fallbackSamplesMixed === true;
  const viewerFramePresentationPassReady =
    guardAllowed &&
    calledFromViewerFrameLifecycle &&
    calledFromExecutorChain &&
    currentTextureConnected &&
    presentationBridgeContract?.currentTextureAcquired === true &&
    presentationBridgeContract?.currentTextureRenderPassSubmitted === true &&
    presentationBridgeContract?.currentTextureReadbackCompleted === true &&
    currentTextureReadbackMatchesAdapterOutput &&
    webgl2HybridRenderingPrevented &&
    fallbackSamplesMixed === false;
  const reason = viewerFramePresentationPassReady
    ? null
    : 'viewer-frame-lifecycle-owned-presentation-pass-validation-not-ready';
  return {
    contractVersion: WEBGPU_VIEWER_FRAME_PRESENTATION_PASS_CONTRACT_VERSION,
    status: viewerFramePresentationPassReady ? 'ok' : 'blocked',
    presentationPassMode:
      'viewer-frame-lifecycle-owned-guarded-webgpu-presentation-pass',
    viewerFramePresentationPassReady,
    calledFromViewerFrameLifecycle,
    calledFromExecutorChain,
    invocationSource,
    frameIndex,
    owner: 'viewer-frame-lifecycle',
    executorContractVersion: executorContract?.contractVersion ?? null,
    runtimeRunnerContractVersion: runtimeRunner?.contractVersion ?? null,
    sourcePresentationBridgeContractVersion:
      presentationBridgeContract?.contractVersion ?? null,
    sourceConnectionMode:
      presentationBridgeContract?.currentTextureConnectionMode ?? null,
    targetResourceKind:
      presentationBridgeContract?.currentTextureTargetResourceKind ?? null,
    targetFormat: presentationBridgeContract?.currentTextureFormat ?? null,
    currentTextureLifecycle:
      presentationBridgeContract?.currentTextureLifecycle ?? null,
    currentTextureCapabilityCheck:
      presentationBridgeContract?.currentTextureCapabilityCheck ?? null,
    viewerOwnsCurrentTextureLifecycle: currentTextureConnected,
    currentTextureConnectionAttempted:
      presentationBridgeContract?.currentTextureConnectionAttempted === true,
    currentTextureConnected,
    currentTextureContextProvided:
      presentationBridgeContract?.currentTextureContextProvided === true,
    currentTextureConfigured:
      presentationBridgeContract?.currentTextureConfigured === true,
    currentTextureAcquired:
      presentationBridgeContract?.currentTextureAcquired === true,
    currentTextureRenderPassSubmitted:
      presentationBridgeContract?.currentTextureRenderPassSubmitted === true,
    currentTextureReadbackCompleted:
      presentationBridgeContract?.currentTextureReadbackCompleted === true,
    currentTextureReadbackMatchesAdapterOutput,
    currentTextureMaxAbsDiff:
      presentationBridgeContract?.currentTextureMaxAbsDiff ?? null,
    submittedWorkDone: presentationBridgeContract?.submittedWorkDone === true,
    gpuCommandPath: presentationBridgeContract?.gpuCommandPath ?? null,
    renderTargetBridgeRetainedForValidation:
      presentationBridgeContract?.renderTargetBridgeReady === true,
    debugCaptureOwnsPresentationPass: false,
    validationOracleRole:
      'capture/dry-run observes the viewer-owned presentation pass but does not own it',
    productionSchedulerConnected: false,
    productionCanvasPresentationConnected:
      presentationBridgeContract?.productionCanvasPresentationConnected === true,
    webgl2HybridRenderingAllowed:
      presentationBridgeContract?.webgl2HybridRenderingAllowed === true,
    fallbackSamplesMixed,
    fullDatasetGpuResidencyRequired: false,
    reason
  };
}

export function buildUnavailableSchedulerFramePresentationBoundaryContract(
  reason,
  extra = {}
) {
  return {
    contractVersion:
      WEBGPU_SCHEDULER_FRAME_PRESENTATION_BOUNDARY_CONTRACT_VERSION,
    status: 'unavailable',
    boundaryMode:
      'scheduler-owned-guarded-webgpu-frame-presentation-boundary',
    schedulerFramePresentationBoundaryReady: false,
    schedulerOwnsFrameRequest: false,
    schedulerOwnsPresentationBoundary: false,
    calledFromSchedulerFrameLoop: false,
    frameRequestIssued: false,
    requestAnimationFrameCallbackEntered: false,
    renderFrameInvoked: false,
    renderFrameCompleted: false,
    viewerFramePresentationPassConsumed: false,
    currentTextureConnected: false,
    currentTextureRenderPassSubmitted: false,
    currentTextureReadbackCompleted: false,
    currentTextureReadbackMatchesAdapterOutput: false,
    debugCaptureOwnsPresentationPass: false,
    productionSchedulerConnected: false,
    webgl2HybridRenderingAllowed: false,
    fallbackSamplesMixed: false,
    fullDatasetGpuResidencyRequired: false,
    reason,
    ...extra
  };
}

export function buildSchedulerFramePresentationBoundaryContract({
  schedulerState,
  viewerFramePresentationPassContract,
  requestedBackendMode = 'webgl2-fallback',
  allowViewerCanvasPresentation = false,
  enableViewerLoopHook = false,
  backendImplementationKind = null,
  frameIndex = 0,
  phase = 'completed'
} = {}) {
  const calledFromSchedulerFrameLoop =
    schedulerState?.calledFromSchedulerFrameLoop === true;
  const frameRequestIssued = schedulerState?.frameRequestIssued === true;
  const requestAnimationFrameCallbackEntered =
    schedulerState?.requestAnimationFrameCallbackEntered === true;
  const renderFrameInvoked = schedulerState?.renderFrameInvoked === true;
  const renderFrameCompleted = schedulerState?.renderFrameCompleted === true;
  const guardAllowed =
    requestedBackendMode === 'webgpu-exclusive' &&
    allowViewerCanvasPresentation === true &&
    enableViewerLoopHook === true &&
    backendImplementationKind ===
      'webgpu-normal-backend-frame-implementation';
  const viewerFramePresentationPassReady =
    viewerFramePresentationPassContract?.viewerFramePresentationPassReady ===
    true;
  const currentTextureConnected =
    viewerFramePresentationPassContract?.currentTextureConnected === true;
  const currentTextureRenderPassSubmitted =
    viewerFramePresentationPassContract?.currentTextureRenderPassSubmitted ===
    true;
  const currentTextureReadbackCompleted =
    viewerFramePresentationPassContract?.currentTextureReadbackCompleted ===
    true;
  const currentTextureReadbackMatchesAdapterOutput =
    viewerFramePresentationPassContract
      ?.currentTextureReadbackMatchesAdapterOutput === true;
  const webgl2HybridRenderingAllowed =
    viewerFramePresentationPassContract?.webgl2HybridRenderingAllowed === true;
  const fallbackSamplesMixed =
    viewerFramePresentationPassContract?.fallbackSamplesMixed === true;
  const schedulerFramePresentationBoundaryReady =
    phase === 'completed' &&
    guardAllowed &&
    calledFromSchedulerFrameLoop &&
    frameRequestIssued &&
    requestAnimationFrameCallbackEntered &&
    renderFrameInvoked &&
    renderFrameCompleted &&
    viewerFramePresentationPassReady &&
    currentTextureConnected &&
    currentTextureRenderPassSubmitted &&
    currentTextureReadbackCompleted &&
    currentTextureReadbackMatchesAdapterOutput &&
    webgl2HybridRenderingAllowed === false &&
    fallbackSamplesMixed === false;
  const reason = schedulerFramePresentationBoundaryReady
    ? null
    : 'scheduler-owned-guarded-webgpu-frame-presentation-boundary-validation-not-ready';
  return {
    contractVersion:
      WEBGPU_SCHEDULER_FRAME_PRESENTATION_BOUNDARY_CONTRACT_VERSION,
    status: schedulerFramePresentationBoundaryReady ? 'ok' : 'blocked',
    boundaryMode:
      'scheduler-owned-guarded-webgpu-frame-presentation-boundary',
    schedulerFramePresentationBoundaryReady,
    schedulerOwnsFrameRequest: frameRequestIssued,
    schedulerOwnsPresentationBoundary:
      schedulerFramePresentationBoundaryReady,
    calledFromSchedulerFrameLoop,
    frameRequestIssued,
    requestAnimationFrameCallbackEntered,
    renderFrameInvoked,
    renderFrameCompleted,
    frameIndex,
    requestedBackendMode,
    allowViewerCanvasPresentation: allowViewerCanvasPresentation === true,
    enableViewerLoopHook: enableViewerLoopHook === true,
    backendImplementationKind,
    guardAllowed,
    sourceViewerFramePresentationPassContractVersion:
      viewerFramePresentationPassContract?.contractVersion ?? null,
    viewerFramePresentationPassConsumed: viewerFramePresentationPassReady,
    viewerFramePresentationPassReady,
    currentTextureConnected,
    currentTextureAcquired:
      viewerFramePresentationPassContract?.currentTextureAcquired === true,
    currentTextureRenderPassSubmitted,
    currentTextureReadbackCompleted,
    currentTextureReadbackMatchesAdapterOutput,
    currentTextureMaxAbsDiff:
      viewerFramePresentationPassContract?.currentTextureMaxAbsDiff ?? null,
    gpuCommandPath: viewerFramePresentationPassContract?.gpuCommandPath ?? null,
    debugCaptureOwnsPresentationPass: false,
    validationOracleRole:
      'capture/dry-run observes the scheduler-owned frame presentation boundary but does not own it',
    productionSchedulerConnected: false,
    productionSchedulerConnectionMode:
      'guarded-scheduler-boundary-only',
    webgl2HybridRenderingAllowed,
    fallbackSamplesMixed,
    fullDatasetGpuResidencyRequired: false,
    reason
  };
}

export function buildPresentationBridgeContract({
  guardedPresentationAdapterContract,
  normalBackendOutputContract,
  presentationHandoffContract,
  targetFormat = 'rgba8unorm',
  targetKind = 'render-target-texture-presentation-bridge',
  targetWidth = 0,
  targetHeight = 0,
  expectedBytes,
  readbackBytes,
  renderTargetCreated = false,
  gpuCopySubmitted = false,
  readbackCompleted = false,
  currentTextureConnectionAttempted = true,
  currentTextureConnected = false,
  currentTextureContextProvided = false,
  currentTextureConfigured = false,
  currentTextureAcquired = false,
  currentTextureRenderPassSubmitted = false,
  currentTextureReadbackCompleted = false,
  currentTextureReadbackMatchesAdapterOutput = false,
  currentTextureFormat = null,
  currentTextureReadbackBytes,
  currentTextureBlockedReason =
    'viewer-canvas-current-texture-context-not-owned-by-guarded-adapter',
  submittedWorkDone = false,
  epsilon = 0
} = {}) {
  const expectedArray = arrayFrom(expectedBytes);
  const readbackArray = arrayFrom(readbackBytes);
  const renderTargetMaxAbsDiff = maxAbsDiff(readbackArray, expectedArray);
  const renderTargetMatchesAdapterOutput =
    readbackCompleted === true && renderTargetMaxAbsDiff <= epsilon;
  const currentTextureReadbackArray = arrayFrom(currentTextureReadbackBytes);
  const expectedCurrentTextureArray = convertRgbaBytesForTextureFormat(
    expectedArray,
    currentTextureFormat
  );
  const currentTextureMaxAbsDiff = maxAbsDiff(
    currentTextureReadbackArray,
    expectedCurrentTextureArray
  );
  const currentTextureMatchesAdapterOutput =
    currentTextureReadbackCompleted === true &&
    currentTextureReadbackMatchesAdapterOutput === true &&
    currentTextureMaxAbsDiff <= epsilon;
  const guardedAdapterReady =
    guardedPresentationAdapterContract?.guardedPresentationAdapterReady === true &&
    guardedPresentationAdapterContract?.handoffResourceConsumedGpuSide === true &&
    guardedPresentationAdapterContract?.presentationTargetReadbackCompleted === true;
  const renderTargetBridgeReady =
    guardedAdapterReady &&
    renderTargetCreated === true &&
    gpuCopySubmitted === true &&
    renderTargetMatchesAdapterOutput;
  const viewerPresentationBridgeReady =
    currentTextureConnected === true || renderTargetBridgeReady;
  const currentTextureConnectionMode = currentTextureConnected
    ? 'viewer-canvas-current-texture-render-pass'
    : currentTextureConnectionAttempted
      ? 'attempted-render-target-bridge-fallback'
      : 'not-attempted-render-target-bridge';
  const reason = viewerPresentationBridgeReady
    ? null
    : 'viewer-presentation-bridge-validation-not-ready';
  return {
    contractVersion: WEBGPU_NORMAL_BACKEND_PRESENTATION_BRIDGE_CONTRACT_VERSION,
    status: viewerPresentationBridgeReady ? 'ok' : 'blocked',
    bridgeMode: 'viewer-canvas-current-texture-or-render-target-bridge',
    viewerPresentationBridgeReady,
    guardedPresentationAdapterOutputConsumed: guardedAdapterReady,
    sourceGuardedPresentationAdapterContractVersion:
      guardedPresentationAdapterContract?.contractVersion ?? null,
    sourceTargetResourceKind:
      guardedPresentationAdapterContract?.targetResourceKind ?? null,
    sourceTargetFormat: guardedPresentationAdapterContract?.targetFormat ?? null,
    sourceTargetWidth: guardedPresentationAdapterContract?.targetWidth ?? null,
    sourceTargetHeight: guardedPresentationAdapterContract?.targetHeight ?? null,
    sourceCoordinateOrigin:
      guardedPresentationAdapterContract?.targetCoordinateOrigin ??
      normalBackendOutputContract?.coordinateOrigin ??
      null,
    sourceCoordinateMapping:
      guardedPresentationAdapterContract?.targetCoordinateMapping ??
      normalBackendOutputContract?.coordinateMapping ??
      null,
    currentTextureConnectionAttempted,
    currentTextureConnected,
    currentTextureContextProvided,
    currentTextureConfigured,
    currentTextureAcquired,
    currentTextureRenderPassSubmitted,
    currentTextureReadbackCompleted,
    currentTextureReadbackMatchesAdapterOutput:
      currentTextureMatchesAdapterOutput,
    currentTextureMaxAbsDiff,
    currentTextureFormat,
    currentTextureTargetResourceKind: 'viewer-canvas-currentTexture',
    currentTextureLifecycle: {
      owner: 'viewer canvas WebGPU lifecycle under webgpu-exclusive guard',
      configuredBy: 'webgpu-only-guarded-presentation-adapter',
      commandPath:
        'render pass samples the guarded adapter target into currentTexture',
      productionSchedulerConnected: false
    },
    currentTextureConnectionMode,
    currentTextureBlockedReason: currentTextureConnected
      ? null
      : currentTextureBlockedReason,
    currentTextureCapabilityCheck: {
      attempted: currentTextureConnectionAttempted,
      webgpuExclusiveRequired: true,
      allowViewerCanvasPresentationRequired: true,
      viewerCanvasWebGpuContextProvided: currentTextureContextProvided,
      viewerCanvasWebGpuContextConfigured: currentTextureConfigured,
      viewerCanvasCurrentTextureAcquired: currentTextureAcquired,
      renderPassSubmitted: currentTextureRenderPassSubmitted,
      readbackCompleted: currentTextureReadbackCompleted,
      readbackMatchesAdapterOutput: currentTextureMatchesAdapterOutput,
      reason: currentTextureConnected ? null : currentTextureBlockedReason
    },
    renderTargetBridgeReady,
    renderTargetTextureCreated: renderTargetCreated === true,
    renderTargetGpuCopySubmitted: gpuCopySubmitted === true,
    renderTargetReadbackCompleted: readbackCompleted === true,
    renderTargetMatchesAdapterOutput,
    renderTargetMaxAbsDiff,
    targetResourceKind: targetKind,
    targetFormat,
    targetWidth,
    targetHeight,
    targetExtent: { width: targetWidth, height: targetHeight },
    targetCoordinateOrigin:
      normalBackendOutputContract?.coordinateOrigin ??
      guardedPresentationAdapterContract?.targetCoordinateOrigin ??
      null,
    targetCoordinateMapping:
      'bridge preserves adapter output texel coordinates for a future viewer presentation pass',
    outputOwnership: {
      owner: 'webgpu-only-guarded-presentation-adapter',
      lifecycle:
        currentTextureConnected
          ? 'adapter output is rendered to viewer canvas currentTexture; render-target bridge remains a validation handoff'
          : 'transient render-target bridge texture is GPU-copied from adapter output and destroyed after validation',
      futureOwner: currentTextureConnected
        ? 'viewer canvas currentTexture lifecycle'
        : 'viewer presentation lifecycle'
    },
    connectionMode: currentTextureConnected
      ? 'current-texture-direct'
      : 'render-target-bridge',
    gpuCommandPath:
      currentTextureConnected
        ? 'render-pass-sample-guarded-adapter-target-to-viewer-canvas-currentTexture'
        : 'copyTextureToTexture-guarded-adapter-target-to-render-target-bridge',
    productionCanvasPresentationConnected: false,
    viewerCanvasCurrentTextureConnected: currentTextureConnected === true,
    renderTargetTextureConnected: renderTargetBridgeReady,
    storageTextureCopyConnected: false,
    webgl2HybridRenderingAllowed: false,
    fallbackSamplesMixed:
      normalBackendOutputContract?.inputSamples?.fallbackSamplesMixed === true,
    futurePresentationTargets:
      presentationHandoffContract?.futurePresentationTargets ??
      DEFAULT_FUTURE_PRESENTATION_TARGETS,
    expectedFirstBytes: expectedArray.slice(0, 16),
    readbackFirstBytes: readbackArray.slice(0, 16),
    submittedWorkDone,
    productionSchedulerImplemented: false,
    productionShaderImplemented: false,
    shColorParityImplemented: false,
    streamingImplemented: false,
    fullDatasetGpuResidencyRequired: false,
    reason
  };
}
