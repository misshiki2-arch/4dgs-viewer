export const WEBGPU_NORMAL_BACKEND_OUTPUT_HANDOFF_CONTRACT_VERSION =
  'phase3-step69-normal-backend-output-presentation-handoff-v1';

export const WEBGPU_NORMAL_BACKEND_PRESENTATION_HANDOFF_CONTRACT_VERSION =
  'phase3-step69-normal-backend-presentation-handoff-boundary-v1';

export const WEBGPU_NORMAL_BACKEND_GUARDED_PRESENTATION_ADAPTER_CONTRACT_VERSION =
  'phase3-step70-webgpu-only-guarded-presentation-adapter-v1';

export const WEBGPU_NORMAL_BACKEND_PRESENTATION_BRIDGE_CONTRACT_VERSION =
  'phase3-step71-viewer-canvas-render-target-presentation-bridge-v1';

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
  const normalBackendOutputReady =
    normalBackendOutputContract?.normalBackendOutputReady === true;
  const handoffReady =
    presentationHandoffContract?.presentationHandoffReady === true;
  const guardedPresentationAdapterReady =
    normalBackendOutputReady &&
    handoffReady &&
    gpuWriteSubmitted === true &&
    presentationTargetMatchesExpected;
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
  const guardedAdapterReady =
    guardedPresentationAdapterContract?.guardedPresentationAdapterReady === true &&
    guardedPresentationAdapterContract?.presentationTargetMatchesExpected === true;
  const renderTargetBridgeReady =
    guardedAdapterReady &&
    renderTargetCreated === true &&
    gpuCopySubmitted === true &&
    renderTargetMatchesAdapterOutput;
  const viewerPresentationBridgeReady =
    currentTextureConnected === true || renderTargetBridgeReady;
  const currentTextureConnectionMode = currentTextureConnected
    ? 'viewer-canvas-current-texture'
    : 'blocked-render-target-bridge';
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
    currentTextureConnectionMode,
    currentTextureBlockedReason: currentTextureConnected
      ? null
      : currentTextureBlockedReason,
    currentTextureCapabilityCheck: {
      attempted: currentTextureConnectionAttempted,
      webgpuExclusiveRequired: true,
      allowViewerCanvasPresentationRequired: true,
      viewerCanvasWebGpuContextProvided: false,
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
        'transient render-target bridge texture is GPU-copied from adapter output and destroyed after validation',
      futureOwner: 'viewer presentation lifecycle'
    },
    connectionMode: currentTextureConnected
      ? 'current-texture-direct'
      : 'render-target-bridge',
    gpuCommandPath:
      'copyTextureToTexture-guarded-adapter-target-to-render-target-bridge',
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
