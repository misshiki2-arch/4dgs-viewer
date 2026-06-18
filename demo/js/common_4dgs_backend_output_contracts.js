export const WEBGPU_NORMAL_BACKEND_OUTPUT_HANDOFF_CONTRACT_VERSION =
  'phase3-step69-normal-backend-output-presentation-handoff-v1';

export const WEBGPU_NORMAL_BACKEND_PRESENTATION_HANDOFF_CONTRACT_VERSION =
  'phase3-step69-normal-backend-presentation-handoff-boundary-v1';

const DEFAULT_FUTURE_PRESENTATION_TARGETS = [
  'viewer-canvas-current-texture',
  'render-target-texture',
  'storage-texture-copy'
];

function arrayFrom(value) {
  if (Array.isArray(value)) return value;
  if (value instanceof Float32Array) return Array.from(value);
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
