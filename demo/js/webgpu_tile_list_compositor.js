import {
  buildWebGpuTileDepthOrderingContract,
  buildWebGpuTileListCompositorContract
} from './common_4dgs_record_contracts.js';
import {
  buildOpaqueWebGpuPngAlphaNormalizationEvidence,
  downloadCanvasPng
} from './debug_download_utils.js';
import {
  buildWebGpuProductionTexturePresentationUvWgsl,
  buildWebGpuPresentationCaptureOrientationEvidence,
  compareWebGpuPresentationFrameIdentity,
  mapWebGpuProductionTextureRowToPngRow,
  summarizeWebGpuPresentationFrameIdentity
} from './webgpu_presentation_capture_orientation_contract.js';
import {
  beginFinalCanvasPresentationWrite,
  buildLastValidProductionOutputCacheDecision,
  buildPolicyNeutralProductionPresentationContract,
  FINAL_CANVAS_PRESENTATION_PATHS,
  productionPresentationEventKindForSource,
  registerFinalCanvasPresentationPath,
  recordFinalCanvasPresentationEvent
} from './common_4dgs_final_canvas_presentation.js';
import {
  canMutateProductionPresentationState
} from './common_4dgs_production_runtime_contract.js';
import {
  executeBoundedProductionTileSortAndCompositor
} from './webgpu_bounded_tile_sort_and_compositor.js';
import {
  buildProductionTileReferenceCapacityContract
} from './common_4dgs_production_tile_reference_contracts.js';

const ORDERING_SUMMARY_UINT_COUNT = 28;
const viewerCanvasWebGpuContextState = new WeakMap();
const viewerCanvasTileCompositorOutputState = new WeakMap();

export function canOwnProductionTileCompositorPresentation(
  viewerCanvasState = null
) {
  return (
    viewerCanvasState?.requestedBackendMode === 'webgpu-exclusive' &&
    viewerCanvasState?.allowViewerCanvasPresentation === true &&
    viewerCanvasState?.webgl2FrameLifecycleSuppressed === true &&
    viewerCanvasState?.provided === true &&
    canMutateProductionPresentationState(
      viewerCanvasState?.productionPresentationMutationPolicy ?? null
    ) &&
    !!viewerCanvasState?.canvas
  );
}

function alignTo(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
}

function finiteNumberOr(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function finiteVector3OrNull(value) {
  if (!Array.isArray(value) || value.length < 3) return null;
  const vector = value.slice(0, 3).map((component) => Number(component));
  return vector.every((component) => Number.isFinite(component)) ? vector : null;
}

function vector3MaxAbsDelta(a, b) {
  const left = finiteVector3OrNull(a);
  const right = finiteVector3OrNull(b);
  if (!left || !right) return null;
  return Math.max(
    Math.abs(left[0] - right[0]),
    Math.abs(left[1] - right[1]),
    Math.abs(left[2] - right[2])
  );
}

function readCompositorSummary(summary) {
  const processedTileCount = Math.round(finiteNumberOr(summary[0], 0));
  const nonEmptyCompositedTileCount = Math.round(finiteNumberOr(summary[1], 0));
  return {
    processedTileCount,
    compositedTileCount: nonEmptyCompositedTileCount,
    nonEmptyCompositedTileCount,
    compositedReferenceCount: Math.round(finiteNumberOr(summary[2], 0)),
    sourceTotalTileReferenceCount: Math.round(finiteNumberOr(summary[3], 0)),
    readOffsetCountTable: Math.round(finiteNumberOr(summary[4], 0)) === 1,
    traversedReferenceList: Math.round(finiteNumberOr(summary[5], 0)) === 1,
    outputTextureWritten: Math.round(finiteNumberOr(summary[6], 0)) === 1,
    maxRefsPerTileObserved: Math.round(finiteNumberOr(summary[7], 0)),
    overflowCount: Math.round(finiteNumberOr(summary[8], 0)),
    statusCode: Math.round(finiteNumberOr(summary[9], 0)),
    orderedReferenceCount: Math.round(finiteNumberOr(summary[10], 0)),
    orderedSourceReferenceCount: Math.round(finiteNumberOr(summary[11], 0)),
    depthKeyConsumed: Math.round(finiteNumberOr(summary[12], 0)) === 1,
    sortKeyConsumed: Math.round(finiteNumberOr(summary[13], 0)) === 1,
    orderAwareCompositorUsed: Math.round(finiteNumberOr(summary[14], 0)) === 1,
    orderedReferenceCountMatchesSource:
      Math.round(finiteNumberOr(summary[15], 0)) === 1,
    gaussianAttributePayloadConsumed:
      Math.round(finiteNumberOr(summary[16], 0)) === 1,
    footprintPayloadConsumed: Math.round(finiteNumberOr(summary[17], 0)) === 1,
    orderedTileReferencesConsumed:
      Math.round(finiteNumberOr(summary[18], 0)) === 1,
    depthOrderedAccumulationUsed:
      Math.round(finiteNumberOr(summary[19], 0)) === 1,
    alphaAccumulationUsed: Math.round(finiteNumberOr(summary[20], 0)) === 1,
    colorAccumulationUsed: Math.round(finiteNumberOr(summary[21], 0)) === 1,
    tileCompositorContributionCount: Math.round(finiteNumberOr(summary[22], 0)),
    debugPatternBypassedForCompositor:
      Math.round(finiteNumberOr(summary[23], 0)) === 1,
    productionTileCompositorPathUsed:
      Math.round(finiteNumberOr(summary[24], 0)) === 1,
    productionAccumulationConsumedParallelSortedRefs:
      Math.round(finiteNumberOr(summary[25], 0)) === 1,
    activeTileDispatchUsed: Math.round(finiteNumberOr(summary[26], 0)) === 1,
    inactiveBackgroundHandlingReady:
      Math.round(finiteNumberOr(summary[27], 0)) === 1,
    activeTileCount: Math.round(finiteNumberOr(summary[28], 0)),
    inactiveTileCount: Math.round(finiteNumberOr(summary[29], 0)),
    activeTilePixelWorkItemCount: Math.round(finiteNumberOr(summary[30], 0)),
    fullScreenPixelWorkAvoided: Math.round(finiteNumberOr(summary[31], 0)),
    accumulationWorkReductionRatio: finiteNumberOr(summary[32], 0),
    outputTextureProducedByProductionCompositor:
      Math.round(finiteNumberOr(summary[33], 0)) === 1,
    debugOutputBypassedForProduction:
      Math.round(finiteNumberOr(summary[34], 0)) === 1,
    activeTileDispatchReady: Math.round(finiteNumberOr(summary[35], 0)) === 1,
    scaleAwareConicPayloadConsumed:
      Math.round(finiteNumberOr(summary[36], 0)) === 1,
    anisotropicFootprintReferenceCount:
      Math.round(finiteNumberOr(summary[37], 0)),
    anisotropicFootprintRatio: finiteNumberOr(summary[38], 0),
    conicFallbackReferenceCount:
      Math.round(finiteNumberOr(summary[39], 0))
  };
}

function readOrderingSummary(summary) {
  const scaledCapacity = 100000;
  return {
    orderedReferenceUpdateCount: Math.round(finiteNumberOr(summary[0], 0)),
    orderedTileCount: Math.round(finiteNumberOr(summary[1], 0)),
    depthKeyObserved: Math.round(finiteNumberOr(summary[2], 0)) === 1,
    sortKeyObserved: Math.round(finiteNumberOr(summary[3], 0)) === 1,
    sourceReferenceCount: Math.round(finiteNumberOr(summary[4], 0)),
    referenceCapacity: Math.round(finiteNumberOr(summary[5], 0)),
    orderingStatusCode: Math.round(finiteNumberOr(summary[6], 0)),
    orderedBufferWritten: Math.round(finiteNumberOr(summary[7], 0)) === 1,
    sortedTileCount: Math.round(finiteNumberOr(summary[8], 0)),
    sortedReferenceCount: Math.round(finiteNumberOr(summary[9], 0)),
    unsortedFallbackTileCount: Math.round(finiteNumberOr(summary[10], 0)),
    maxReferencesPerTile: Math.round(finiteNumberOr(summary[11], 0)),
    sourceReferenceCountBeforeSortLimit: Math.round(finiteNumberOr(summary[12], 0)),
    droppedReferenceCount: Math.round(finiteNumberOr(summary[13], 0)),
    overflowTileCount: Math.round(finiteNumberOr(summary[14], 0)),
    overflowReferenceCount: Math.round(finiteNumberOr(summary[15], 0)),
    sortCapacityLimit: Math.round(finiteNumberOr(summary[16], 0)),
    capacityUtilizationMax:
      finiteNumberOr(summary[17], 0) / scaledCapacity,
    capacityUtilizationSum:
      finiteNumberOr(summary[18], 0) / scaledCapacity,
    capacityTelemetryTileCount: Math.round(finiteNumberOr(summary[19], 0)),
    sortOrderViolationCount: Math.round(finiteNumberOr(summary[20], 0)),
    parallelSortStageCount: Math.round(finiteNumberOr(summary[21], 0)),
    sortWorkgroupCount: Math.round(finiteNumberOr(summary[22], 0)),
    sortWorkItemCount: Math.round(finiteNumberOr(summary[9], 0)),
    parallelSortCompareSwapPassCount: Math.round(finiteNumberOr(summary[23], 0))
  };
}

function hasNonZeroTextureByte(readback, bytesPerRow, width, height) {
  for (let y = 0; y < height; y += 1) {
    const row = y * bytesPerRow;
    for (let x = 0; x < width; x += 1) {
      const offset = row + x * 4;
      if (
        readback[offset] !== 0 ||
        readback[offset + 1] !== 0 ||
        readback[offset + 2] !== 0
      ) {
        return true;
      }
    }
  }
  return false;
}

function summarizeTextureReadback(
  readback,
  bytesPerRow,
  width,
  height
) {
  let nonzeroPixelCount = 0;
  let alphaNonzeroPixelCount = 0;
  let alphaZeroPixelCount = 0;
  let alphaOpaquePixelCount = 0;
  let rgbMax = 0;
  let hash = 2166136261;
  let rgbHash = 2166136261;
  let alphaHash = 2166136261;
  const pixelCount = Math.max(1, width * height);
  for (let y = 0; y < height; y += 1) {
    const row = y * bytesPerRow;
    for (let x = 0; x < width; x += 1) {
      const offset = row + x * 4;
      const r = readback[offset + 0] ?? 0;
      const g = readback[offset + 1] ?? 0;
      const b = readback[offset + 2] ?? 0;
      const a = readback[offset + 3] ?? 0;
      const pixelRgbMax = Math.max(r, g, b);
      rgbMax = Math.max(rgbMax, pixelRgbMax);
      if (pixelRgbMax > 0) {
        nonzeroPixelCount += 1;
      }
      alphaNonzeroPixelCount += a > 0 ? 1 : 0;
      alphaZeroPixelCount += a === 0 ? 1 : 0;
      alphaOpaquePixelCount += a === 255 ? 1 : 0;
      hash ^= r;
      hash = Math.imul(hash, 16777619) >>> 0;
      rgbHash ^= r;
      rgbHash = Math.imul(rgbHash, 16777619) >>> 0;
      hash ^= g;
      hash = Math.imul(hash, 16777619) >>> 0;
      rgbHash ^= g;
      rgbHash = Math.imul(rgbHash, 16777619) >>> 0;
      hash ^= b;
      hash = Math.imul(hash, 16777619) >>> 0;
      rgbHash ^= b;
      rgbHash = Math.imul(rgbHash, 16777619) >>> 0;
      hash ^= a;
      hash = Math.imul(hash, 16777619) >>> 0;
      alphaHash ^= a;
      alphaHash = Math.imul(alphaHash, 16777619) >>> 0;
    }
  }
  return {
    nonzeroDefinition: 'rgb-any-channel-greater-than-zero-alpha-excluded',
    nonzeroPixelCount,
    nonzeroPixelRatio: nonzeroPixelCount / pixelCount,
    rgbNonzeroPixelCount: nonzeroPixelCount,
    rgbNonblackRatio: nonzeroPixelCount / pixelCount,
    rgbMax,
    alphaNonzeroPixelCount,
    alphaZeroPixelCount,
    alphaOpaquePixelCount,
    rgbHash: rgbHash.toString(16).padStart(8, '0'),
    alphaHash: alphaHash.toString(16).padStart(8, '0'),
    frameHash: hash.toString(16).padStart(8, '0')
  };
}

function createCanvasFromRgbaReadback(
  readback,
  bytesPerRow,
  width,
  height,
  { forceOpaqueAlpha = false } = {}
) {
  const snapshotCanvas = document.createElement('canvas');
  snapshotCanvas.width = width;
  snapshotCanvas.height = height;
  const ctx = snapshotCanvas.getContext('2d', { willReadFrequently: true });
  const imageData = ctx.createImageData(width, height);
  const rowStride = width * 4;
  for (let y = 0; y < height; y += 1) {
    const srcOffset = y * bytesPerRow;
    const dstY = mapWebGpuProductionTextureRowToPngRow(y, height);
    const dstOffset = dstY * rowStride;
    imageData.data.set(
      readback.subarray(srcOffset, srcOffset + rowStride),
      dstOffset
    );
    if (forceOpaqueAlpha) {
      for (let x = 0; x < width; x += 1) {
        imageData.data[dstOffset + x * 4 + 3] = 255;
      }
    }
  }
  ctx.putImageData(imageData, 0, 0);
  return snapshotCanvas;
}

export async function captureCachedWebGpuTileCompositorOutputPng({
  viewerCanvasState = null,
  canvas = null,
  name = 'webgpu-tile-compositor-output.png',
  download = true,
  requestedStateIdentity = null
} = {}) {
  const viewerCanvas = canvas ?? viewerCanvasState?.canvas ?? null;
  const cached = viewerCanvas
    ? viewerCanvasTileCompositorOutputState.get(viewerCanvas)
    : null;
  if (!cached?.device || !cached?.outputTexture) {
    return {
      blob: null,
      fileName: name,
      status: 'unavailable',
      reason: 'last-valid-webgpu-tile-compositor-output-unavailable',
      source: 'cached-last-valid-webgpu-tile-compositor-output-texture-readback'
    };
  }
  const width = Math.max(1, Math.round(finiteNumberOr(cached.outputWidth, 0)));
  const height = Math.max(1, Math.round(finiteNumberOr(cached.outputHeight, 0)));
  const bytesPerRow = alignTo(width * 4, 256);
  const readbackBuffer = cached.device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const encoder = cached.device.createCommandEncoder({
    label: 'capture-cached-webgpu-tile-compositor-output-png'
  });
  encoder.copyTextureToBuffer(
    { texture: cached.outputTexture },
    { buffer: readbackBuffer, bytesPerRow, rowsPerImage: height },
    { width, height, depthOrArrayLayers: 1 }
  );
  cached.device.queue.submit([encoder.finish()]);
  if (typeof cached.device.queue.onSubmittedWorkDone === 'function') {
    await cached.device.queue.onSubmittedWorkDone();
  }
  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const readback = new Uint8Array(readbackBuffer.getMappedRange().slice(0));
  readbackBuffer.unmap();
  if (typeof readbackBuffer.destroy === 'function') {
    readbackBuffer.destroy();
  }
  const outputStats = summarizeTextureReadback(readback, bytesPerRow, width, height);
  const preNormalizationPixelEvidence = summarizeTextureReadback(
    readback,
    bytesPerRow,
    width,
    height
  );
  const snapshotCanvas = createCanvasFromRgbaReadback(
    readback,
    bytesPerRow,
    width,
    height,
    {
      forceOpaqueAlpha: true
    }
  );
  const result = await downloadCanvasPng(snapshotCanvas, name, { download });
  const postNormalizationPixelEvidence = result.encodedPngPixelEvidence ?? null;
  const alphaNormalizationEvidence =
    buildOpaqueWebGpuPngAlphaNormalizationEvidence({
      preNormalizationPixelEvidence,
      postNormalizationPixelEvidence,
      preNormalizationSourceIdentity: {
        outputTextureIdentity: cached.outputTextureIdentity ?? null,
        generation: cached.generation ?? null,
        rgbHash: preNormalizationPixelEvidence.rgbHash,
        alphaHash: preNormalizationPixelEvidence.alphaHash
      },
      postNormalizationBlobIdentity: result.captureBlobIdentity ?? null,
      width,
      height
    });
  const capturedFrameIdentity = summarizeWebGpuPresentationFrameIdentity({
    ...(cached.frameIdentity ?? {}),
    generation: cached.generation ?? null,
    outputWidth: width,
    outputHeight: height
  });
  const presentedFrameIdentity = cached.presentedFrameIdentity ?? null;
  const requestedFrameIdentity = requestedStateIdentity ?? cached.requestedStateIdentity ?? null;
  const presentedFrameIdentityRequiredKeys = [
    'generation',
    'datasetCameraLabel',
    'datasetFrameNumber',
    'datasetTime',
    'referenceCameraLabel',
    'outputWidth',
    'outputHeight'
  ];
  const requestedStateIdentityRequiredKeys = [
    'datasetCameraLabel',
    'datasetFrameNumber',
    'datasetTime',
    'referenceCameraLabel',
    'outputWidth',
    'outputHeight'
  ];
  const captureVsPresented = compareWebGpuPresentationFrameIdentity(
    capturedFrameIdentity,
    presentedFrameIdentity ?? {},
    { requiredKeys: presentedFrameIdentityRequiredKeys }
  );
  const captureVsRequested = compareWebGpuPresentationFrameIdentity(
    capturedFrameIdentity,
    requestedFrameIdentity ?? {},
    { requiredKeys: requestedStateIdentityRequiredKeys }
  );
  const captureMatchesPresentedFrame =
    presentedFrameIdentity != null && captureVsPresented.matches === true;
  const captureMatchesRequestedState =
    requestedFrameIdentity != null && captureVsRequested.matches === true;
  const captureFreshnessKnown =
    captureMatchesPresentedFrame && captureMatchesRequestedState;
  const staleCaptureDetected =
    presentedFrameIdentity != null && captureVsPresented.mismatchedKeys.length > 0;
  const orientationEvidence =
    buildWebGpuPresentationCaptureOrientationEvidence();
  return {
    ...result,
    status: 'success',
    reason: null,
    source: 'cached-last-valid-webgpu-tile-compositor-output-texture-readback',
    captureSourceKind:
      'cached-last-valid-webgpu-tile-compositor-output-texture-readback',
    outputWidth: width,
    outputHeight: height,
    outputStats,
    captureAlphaPolicy: 'force-opaque-to-match-webgpu-canvas-presentation',
    captureAlphaForcedOpaque: true,
    alphaNormalizationEvidence,
    cacheGeneration: cached.generation ?? null,
    cachedAtMs: cached.cachedAtMs ?? null,
    productionOutputGeneration: cached.generation ?? null,
    presentedOutputGeneration: presentedFrameIdentity?.generation ?? null,
    capturedOutputGeneration: capturedFrameIdentity.generation ?? null,
    outputTextureIdentity: cached.outputTextureIdentity ?? null,
    requestedStateIdentity: requestedFrameIdentity,
    presentedFrameIdentity,
    capturedFrameIdentity,
    captureVsPresentedFrameIdentity: captureVsPresented,
    captureVsRequestedStateIdentity: captureVsRequested,
    capturePresentedFrameMismatchedFields: captureVsPresented.mismatchedKeys,
    capturePresentedFrameMissingFields: captureVsPresented.missingKeys,
    captureRequestedStateMismatchedFields: captureVsRequested.mismatchedKeys,
    captureRequestedStateMissingFields: captureVsRequested.missingKeys,
    captureMatchesPresentedFrame,
    captureMatchesRequestedState,
    staleCaptureDetected,
    captureFreshnessKnown,
    captureFreshnessClassification: captureFreshnessKnown
      ? 'captured-current-presented-fixed-reference-frame'
      : staleCaptureDetected
        ? 'stale-last-valid-output-detected'
        : 'capture-freshness-evidence-missing-or-incomplete',
    orientationEvidence,
    ...orientationEvidence
  };
}

function configureViewerCanvasWebGpuContext({
  canvas,
  context,
  device,
  format,
  width,
  height,
  forceRefresh = false
}) {
  const key = `${format}:${width}x${height}`;
  const previous = viewerCanvasWebGpuContextState.get(canvas);
  const deviceChanged = !!previous?.device && previous.device !== device;
  const sizeOrFormatChanged = !!previous?.key && previous.key !== key;
  if (!forceRefresh && previous?.device === device && previous?.key === key) {
    return {
      configured: false,
      deviceChanged: false,
      sizeOrFormatChanged: false,
      reason: 'existing-context-configuration-reused'
    };
  }
  context.configure({
    device,
    format,
    usage:
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.COPY_SRC,
    alphaMode: 'premultiplied'
  });
  viewerCanvasWebGpuContextState.set(canvas, { device, key });
  return {
    configured: true,
    deviceChanged,
    sizeOrFormatChanged,
    reason: forceRefresh
      ? 'context-refreshed-for-tile-compositor-fresh-currentTexture'
      : 'context-configured-for-tile-compositor'
  };
}

function getPreferredWebGpuCanvasFormat() {
  return typeof navigator !== 'undefined' && navigator.gpu?.getPreferredCanvasFormat
    ? navigator.gpu.getPreferredCanvasFormat()
    : 'bgra8unorm';
}

async function presentTileCompositorTextureToCurrentTexture({
  device,
  outputTexture,
  outputWidth,
  outputHeight,
  viewerCanvasState,
  canvasWidth,
  canvasHeight,
  frameCount = 1,
  presentationSource = 'webgpu-tile-compositor-output-texture',
  forceContextRefresh = false,
  outputGeneration = null,
  outputTextureIdentity = null,
  sourceRequestIdentity = null,
  frameIdentity = null,
  canonicalOutputCompletionReady = null,
  productionWorkClassification = null,
  onPresentedFrame = null
}) {
  registerFinalCanvasPresentationPath(viewerCanvasState, {
    pathIdentity: FINAL_CANVAS_PRESENTATION_PATHS.TILE_COMPOSITOR,
    source: 'webgpu_tile_list_compositor.presentTileCompositorTextureToCurrentTexture',
    supportedEventKinds: [
      'production-presentation',
      'cached-production-presentation',
      'black-fallback',
      'presentation-failure'
    ]
  });
  const presentationEventKind =
    productionPresentationEventKindForSource(presentationSource);
  const viewerCanvas = viewerCanvasState?.canvas ?? null;
  const summary = {
    compositorOutputPresentedToCurrentTexture: false,
    compositorCurrentTextureRenderPassSubmitted: false,
    compositorCurrentTextureReadbackCompleted: false,
    compositorCurrentTextureReadbackNonZero: false,
    presentationFrameCount: 0,
    compositorPresentationFrameCount: 0,
    currentTextureUsesWebGpuTileCompositorOutput: false,
    presentationStableUntilCapture: false,
    presentationFrameSamples: [],
    presentationNonzeroPixelRatioMin: 0,
    presentationNonzeroPixelRatioMax: 0,
    presentationFrameHashChanges: 0,
    currentTextureContextReconfigured: false,
    webgpuDeviceConsistencyReady: false,
    presentationDeviceMatchesCompositorDevice: false,
    currentTextureViewFreshPerPresentation: false,
    currentTextureViewReusedAcrossFrames: false,
    staleTextureViewReuseDetected: false,
    crossDeviceTextureViewUseDetected: false,
    contextReconfiguredOnDeviceChange: false,
    compositorOutputCacheInvalidatedOnDeviceChange: false,
    webgpuValidationErrorDetected: false,
    invalidCommandBufferDetected: false,
    queueSubmitFailureDetected: false,
    presentationErrorName: null,
    presentationErrorMessage: null,
    currentTextureSource: null,
    presentationSource,
    presentationEventKind,
    sourcePixelResult: 'unknown',
    policyNeutralPresentationContract:
      buildPolicyNeutralProductionPresentationContract({
        canonicalOutputCompletionReady,
        productionWorkClassification,
        presentationEventKind,
        presentationSource
      }),
    orientationEvidence: buildWebGpuPresentationCaptureOrientationEvidence()
  };
  const currentTextureGuardAllowed =
    canOwnProductionTileCompositorPresentation(viewerCanvasState);
  if (
    !currentTextureGuardAllowed ||
    !device ||
    !outputTexture ||
    typeof GPUBufferUsage === 'undefined' ||
    typeof GPUMapMode === 'undefined'
  ) {
    return summary;
  }

  let validationScopePushed = false;
  let activeWriteToken = null;
  try {
    const validationScopeSupported =
      typeof device.pushErrorScope === 'function' &&
      typeof device.popErrorScope === 'function';
    if (validationScopeSupported) {
      device.pushErrorScope('validation');
      validationScopePushed = true;
    }
    const currentTextureFormat = getPreferredWebGpuCanvasFormat();
    const currentTextureWidth = Math.max(
      1,
      Math.round(viewerCanvas.width ?? canvasWidth ?? outputWidth)
    );
    const currentTextureHeight = Math.max(
      1,
      Math.round(viewerCanvas.height ?? canvasHeight ?? outputHeight)
    );
    const currentTextureBytesPerRow = alignTo(currentTextureWidth * 4, 256);
    const currentTextureReadbackBuffer = device.createBuffer({
      size: currentTextureBytesPerRow * currentTextureHeight,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    const context = viewerCanvas.getContext?.('webgpu') ?? null;
    if (!context) {
      if (typeof currentTextureReadbackBuffer.destroy === 'function') {
        currentTextureReadbackBuffer.destroy();
      }
      return summary;
    }

    const contextConfiguration = configureViewerCanvasWebGpuContext({
      canvas: viewerCanvas,
      context,
      device,
      format: currentTextureFormat,
      width: currentTextureWidth,
      height: currentTextureHeight,
      forceRefresh: forceContextRefresh
    });
    summary.currentTextureContextReconfigured =
      contextConfiguration.configured === true;
    summary.contextReconfiguredOnDeviceChange =
      contextConfiguration.deviceChanged === true;
    summary.compositorOutputCacheInvalidatedOnDeviceChange =
      contextConfiguration.deviceChanged === true;
    summary.presentationDeviceMatchesCompositorDevice = true;
    const sampler = device.createSampler({
      magFilter: 'nearest',
      minFilter: 'nearest'
    });
    const presentationUvExpression =
      buildWebGpuProductionTexturePresentationUvWgsl('pos');
    const presentationShader = device.createShaderModule({
      label: 'phase3-step88-tile-compositor-current-texture-wgsl',
      code: `
struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0)
  );
  var out: VertexOut;
  let pos = positions[vertexIndex];
  out.position = vec4f(pos, 0.0, 1.0);
  out.uv = ${presentationUvExpression};
  return out;
}

@group(0) @binding(0) var tileSampler: sampler;
@group(0) @binding(1) var tileOutput: texture_2d<f32>;

@fragment
fn fsMain(in: VertexOut) -> @location(0) vec4f {
  return textureSample(tileOutput, tileSampler, in.uv);
}
`
    });
    const presentationPipeline = device.createRenderPipeline({
      label: 'phase3-step88-tile-compositor-current-texture-pipeline',
      layout: 'auto',
      vertex: { module: presentationShader, entryPoint: 'vsMain' },
      fragment: {
        module: presentationShader,
        entryPoint: 'fsMain',
        targets: [{ format: currentTextureFormat }]
      },
      primitive: { topology: 'triangle-list' }
    });
    const presentationBindGroup = device.createBindGroup({
      layout: presentationPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: outputTexture.createView() }
      ]
    });
    let previousCurrentTexture = null;
    let previousCurrentTextureView = null;
    for (let frame = 0; frame < frameCount; frame += 1) {
      activeWriteToken = beginFinalCanvasPresentationWrite(viewerCanvasState, {
        pathIdentity: FINAL_CANVAS_PRESENTATION_PATHS.TILE_COMPOSITOR,
        sourceRequestIdentity,
        productionGeneration: outputGeneration
      });
      const currentTexture = context.getCurrentTexture();
      const currentTextureView = currentTexture.createView();
      if (previousCurrentTexture === currentTexture ||
          previousCurrentTextureView === currentTextureView) {
        summary.currentTextureViewReusedAcrossFrames = true;
      }
      previousCurrentTexture = currentTexture;
      previousCurrentTextureView = currentTextureView;
      const presentationEncoder = device.createCommandEncoder({
        label: 'phase3-step88-tile-compositor-current-texture-encoder'
      });
      const presentationPass = presentationEncoder.beginRenderPass({
        label: 'phase3-step88-tile-compositor-current-texture-pass',
        colorAttachments: [
          {
            view: currentTextureView,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store'
          }
        ]
      });
      presentationPass.setPipeline(presentationPipeline);
      presentationPass.setBindGroup(0, presentationBindGroup);
      presentationPass.draw(3);
      presentationPass.end();
      presentationEncoder.copyTextureToBuffer(
        { texture: currentTexture },
        {
          buffer: currentTextureReadbackBuffer,
          bytesPerRow: currentTextureBytesPerRow,
          rowsPerImage: currentTextureHeight
        },
        {
          width: currentTextureWidth,
          height: currentTextureHeight,
          depthOrArrayLayers: 1
        }
      );
      device.queue.submit([presentationEncoder.finish()]);
      summary.presentationFrameCount += 1;
      summary.compositorPresentationFrameCount += 1;
      summary.compositorCurrentTextureRenderPassSubmitted = true;
      if (typeof device.queue.onSubmittedWorkDone === 'function') {
        await device.queue.onSubmittedWorkDone();
      }
      await currentTextureReadbackBuffer.mapAsync(GPUMapMode.READ);
      const currentTextureReadback = new Uint8Array(
        currentTextureReadbackBuffer.getMappedRange().slice(0)
      );
      currentTextureReadbackBuffer.unmap();
      summary.compositorCurrentTextureReadbackCompleted = true;
      const readbackSummary = summarizeTextureReadback(
        currentTextureReadback,
        currentTextureBytesPerRow,
        currentTextureWidth,
        currentTextureHeight
      );
      summary.compositorCurrentTextureReadbackNonZero =
        readbackSummary.nonzeroPixelCount > 0;
      summary.presentationFrameSamples.push({
        frame,
        nonzeroPixelCount: readbackSummary.nonzeroPixelCount,
        nonzeroPixelRatio: readbackSummary.nonzeroPixelRatio,
        frameHash: readbackSummary.frameHash,
        currentTextureUsesWebGpuTileCompositorOutput: true,
        presentationSource,
        frameIdentity: summarizeWebGpuPresentationFrameIdentity({
          ...(frameIdentity ?? {}),
          generation: outputGeneration,
          frameHash: readbackSummary.frameHash,
          outputWidth: currentTextureWidth,
          outputHeight: currentTextureHeight
        })
      });
      const latestSample = summary.presentationFrameSamples.at(-1);
      recordFinalCanvasPresentationEvent(viewerCanvasState, {
        writeToken: activeWriteToken,
        presentationPathIdentity: FINAL_CANVAS_PRESENTATION_PATHS.TILE_COMPOSITOR,
        eventKind: presentationEventKind,
        presentationSource,
        sourceRequestIdentity,
        presentingRequestIdentity:
          viewerCanvasState?.schedulerFrameState?.requestIdentity ?? null,
        scheduleSource:
          viewerCanvasState?.schedulerFrameState?.requestSource ?? null,
        productionGeneration: outputGeneration,
        compositorGeneration: outputGeneration,
        presentedGeneration: outputGeneration,
        frameIdentity: latestSample?.frameIdentity ?? frameIdentity,
        sourcePixelEvidenceIdentity: {
          outputTextureIdentity,
          generation: outputGeneration,
          frameHash: readbackSummary.frameHash,
          width: currentTextureWidth,
          height: currentTextureHeight
        },
        sourcePixelResult:
          readbackSummary.nonzeroPixelCount > 0 ? 'nonblank' : 'black',
        sourcePixelStats: readbackSummary,
        canvasWriteAttempted: true,
        canvasWriteSubmitted: true,
        canvasWriteCompleted: true,
        staleSource: false,
        presentationFailed: false
      });
      activeWriteToken = null;
      if (typeof onPresentedFrame === 'function') {
        onPresentedFrame(latestSample);
      }
    }
    summary.currentTextureViewFreshPerPresentation =
      summary.presentationFrameCount === frameCount &&
      summary.currentTextureViewReusedAcrossFrames === false;
    if (typeof currentTextureReadbackBuffer.destroy === 'function') {
      currentTextureReadbackBuffer.destroy();
    }
    if (validationScopeSupported) {
      const scopedValidationError = await device.popErrorScope();
      validationScopePushed = false;
      if (scopedValidationError) {
        summary.webgpuValidationErrorDetected = true;
        summary.presentationErrorName =
          scopedValidationError.name ?? 'GPUValidationError';
        summary.presentationErrorMessage =
          scopedValidationError.message ?? String(scopedValidationError);
        summary.crossDeviceTextureViewUseDetected =
          /associated with \[Device\].*cannot be used with \[Device\]/i.test(
            summary.presentationErrorMessage
          );
        summary.staleTextureViewReuseDetected =
          summary.crossDeviceTextureViewUseDetected === true ||
          /stale.*TextureView/i.test(summary.presentationErrorMessage);
        recordFinalCanvasPresentationEvent(viewerCanvasState, {
          presentationPathIdentity: FINAL_CANVAS_PRESENTATION_PATHS.TILE_COMPOSITOR,
          eventKind: 'presentation-failure',
          presentationSource,
          sourceRequestIdentity,
          presentingRequestIdentity:
            viewerCanvasState?.schedulerFrameState?.requestIdentity ?? null,
          scheduleSource:
            viewerCanvasState?.schedulerFrameState?.requestSource ?? null,
          productionGeneration: outputGeneration,
          compositorGeneration: outputGeneration,
          presentedGeneration: null,
          frameIdentity,
          sourcePixelEvidenceIdentity: {
            outputTextureIdentity,
            generation: outputGeneration
          },
          sourcePixelResult: 'unknown',
          canvasWriteAttempted: true,
          canvasWriteSubmitted:
            summary.compositorCurrentTextureRenderPassSubmitted === true,
          canvasWriteCompleted: false,
          staleSource: null,
          presentationFailed: true,
          error: {
            name: summary.presentationErrorName,
            message: summary.presentationErrorMessage
          },
          blockedReason: summary.presentationErrorMessage
        });
      }
    }
  } catch (error) {
    if (validationScopePushed && typeof device.popErrorScope === 'function') {
      try {
        await device.popErrorScope();
      } catch (_scopeError) {
        // Ignore cleanup errors; the original presentation failure is reported below.
      }
    }
    summary.compositorOutputPresentedToCurrentTexture = false;
    summary.presentationErrorName = error?.name ?? 'Error';
    summary.presentationErrorMessage = error?.message ?? String(error);
    const errorText = `${summary.presentationErrorName}: ${summary.presentationErrorMessage}`;
    summary.webgpuValidationErrorDetected =
      /validation|TextureView|associated with \[Device\]|cannot be used with \[Device\]/i
        .test(errorText);
    summary.invalidCommandBufferDetected = /Invalid CommandBuffer/i.test(errorText);
    summary.queueSubmitFailureDetected = /queue\.submit|submit/i.test(errorText);
    summary.crossDeviceTextureViewUseDetected =
      /associated with \[Device\].*cannot be used with \[Device\]/i.test(errorText);
    summary.staleTextureViewReuseDetected =
      summary.crossDeviceTextureViewUseDetected === true ||
      /stale.*TextureView/i.test(errorText);
    recordFinalCanvasPresentationEvent(viewerCanvasState, {
      writeToken: activeWriteToken,
      presentationPathIdentity: FINAL_CANVAS_PRESENTATION_PATHS.TILE_COMPOSITOR,
      eventKind: 'presentation-failure',
      presentationSource,
      sourceRequestIdentity,
      presentingRequestIdentity:
        viewerCanvasState?.schedulerFrameState?.requestIdentity ?? null,
      scheduleSource:
        viewerCanvasState?.schedulerFrameState?.requestSource ?? null,
      productionGeneration: outputGeneration,
      compositorGeneration: outputGeneration,
      presentedGeneration: null,
      frameIdentity,
      sourcePixelEvidenceIdentity: {
        outputTextureIdentity,
        generation: outputGeneration
      },
      sourcePixelResult: 'unknown',
      canvasWriteAttempted: true,
      canvasWriteSubmitted:
        summary.compositorCurrentTextureRenderPassSubmitted === true,
      canvasWriteCompleted: false,
      staleSource: null,
      presentationFailed: true,
      error: {
        name: summary.presentationErrorName,
        message: summary.presentationErrorMessage
      },
      blockedReason: summary.presentationErrorMessage
    });
    summary.policyNeutralPresentationContract =
      buildPolicyNeutralProductionPresentationContract({
        canonicalOutputCompletionReady,
        productionWorkClassification,
        presentationEventKind: 'presentation-failure',
        presentationSource,
        sourcePixelResult: 'unknown',
        sourceIdentityKnown: false,
        sourceIdentityMatchesExpected: false,
        canvasWriteAttempted: true,
        canvasWriteSubmitted:
          summary.compositorCurrentTextureRenderPassSubmitted === true,
        canvasWriteCompleted: false,
        sameSourcePersistence: false,
        samePixelResultPersistence: false,
        persistenceObservedEventCount: summary.presentationFrameSamples.length,
        staleSource: false,
        presentationFailed: true,
        webgpuValidationErrorDetected:
          summary.webgpuValidationErrorDetected === true,
        invalidCommandBufferDetected:
          summary.invalidCommandBufferDetected === true,
        queueSubmitFailureDetected:
          summary.queueSubmitFailureDetected === true
      });
    return summary;
  }

  summary.currentTextureUsesWebGpuTileCompositorOutput =
    summary.compositorPresentationFrameCount > 0;
  const nonzeroRatios = summary.presentationFrameSamples.map(
    (sample) => sample.nonzeroPixelRatio
  );
  summary.presentationNonzeroPixelRatioMin =
    nonzeroRatios.length > 0 ? Math.min(...nonzeroRatios) : 0;
  summary.presentationNonzeroPixelRatioMax =
    nonzeroRatios.length > 0 ? Math.max(...nonzeroRatios) : 0;
  summary.presentationFrameHashChanges = summary.presentationFrameSamples.reduce(
    (count, sample, index, samples) =>
      index > 0 && sample.frameHash !== samples[index - 1].frameHash
        ? count + 1
        : count,
    0
  );
  summary.presentationStableUntilCapture =
    summary.compositorPresentationFrameCount >= 1 &&
    summary.compositorCurrentTextureReadbackCompleted &&
    summary.presentationFrameSamples.every(
      (sample) =>
        sample.currentTextureUsesWebGpuTileCompositorOutput === true &&
        sample.nonzeroPixelCount > 0
    );
  summary.webgpuDeviceConsistencyReady =
    summary.presentationDeviceMatchesCompositorDevice === true &&
    summary.currentTextureViewFreshPerPresentation === true &&
    summary.currentTextureViewReusedAcrossFrames === false &&
    summary.staleTextureViewReuseDetected === false &&
    summary.crossDeviceTextureViewUseDetected === false &&
    summary.webgpuValidationErrorDetected === false &&
    summary.invalidCommandBufferDetected === false &&
    summary.queueSubmitFailureDetected === false;
  const expectedPresentedFrameIdentity =
    summarizeWebGpuPresentationFrameIdentity({
      ...(frameIdentity ?? {}),
      generation: outputGeneration,
      outputWidth: Math.max(
        1,
        Math.round(viewerCanvas.width ?? canvasWidth ?? outputWidth)
      ),
      outputHeight: Math.max(
        1,
        Math.round(viewerCanvas.height ?? canvasHeight ?? outputHeight)
      )
    });
  const sourceIdentityComparisons = summary.presentationFrameSamples.map(
    (sample) => compareWebGpuPresentationFrameIdentity(
      sample.frameIdentity,
      expectedPresentedFrameIdentity,
      { requiredKeys: ['generation', 'outputWidth', 'outputHeight'] }
    )
  );
  const sourceIdentityKnown =
    sourceRequestIdentity != null &&
    outputGeneration != null &&
    sourceIdentityComparisons.length > 0 &&
    sourceIdentityComparisons.every(
      (comparison) => comparison.missingKeys.length === 0
    );
  const sourceIdentityMatchesExpected =
    sourceIdentityKnown &&
    sourceIdentityComparisons.every((comparison) => comparison.matches === true);
  const sampledPixelResults = summary.presentationFrameSamples.map(
    (sample) => sample.nonzeroPixelCount > 0 ? 'nonblank' : 'black'
  );
  const sourcePixelResult = sampledPixelResults.length > 0 &&
    sampledPixelResults.every((result) => result === sampledPixelResults[0])
    ? sampledPixelResults[0]
    : 'unknown';
  const sameSourcePersistence =
    sourceIdentityMatchesExpected &&
    summary.presentationFrameSamples.every(
      (sample) => sample.presentationSource === presentationSource
    );
  const samePixelResultPersistence =
    sourcePixelResult !== 'unknown' &&
    sampledPixelResults.length === summary.presentationFrameSamples.length;
  summary.sourcePixelResult = sourcePixelResult;
  summary.policyNeutralPresentationContract =
    buildPolicyNeutralProductionPresentationContract({
      canonicalOutputCompletionReady,
      productionWorkClassification,
      presentationEventKind,
      presentationSource,
      sourcePixelResult,
      sourceIdentityKnown,
      sourceIdentityMatchesExpected,
      canvasWriteAttempted: summary.presentationFrameCount > 0,
      canvasWriteSubmitted:
        summary.compositorCurrentTextureRenderPassSubmitted === true,
      canvasWriteCompleted:
        summary.compositorCurrentTextureReadbackCompleted === true,
      sameSourcePersistence,
      samePixelResultPersistence,
      persistenceObservedEventCount: summary.presentationFrameSamples.length,
      laterPixelResultChange:
        summary.presentationFrameHashChanges > 0 &&
        samePixelResultPersistence === false,
      staleSource: false,
      presentationFailed: false,
      webgpuValidationErrorDetected:
        summary.webgpuValidationErrorDetected === true,
      invalidCommandBufferDetected:
        summary.invalidCommandBufferDetected === true,
      queueSubmitFailureDetected:
        summary.queueSubmitFailureDetected === true
    });
  summary.compositorOutputPresentedToCurrentTexture =
    summary.policyNeutralPresentationContract.productionPresentationPath === true &&
    summary.policyNeutralPresentationContract.currentTextureWriteCompleted === true &&
    summary.policyNeutralPresentationContract.sourceIdentityKnown === true &&
    summary.policyNeutralPresentationContract.sourceIdentityMatchesExpected === true &&
    summary.policyNeutralPresentationContract.sameSourcePersistence === true &&
    summary.policyNeutralPresentationContract.samePixelResultPersistence === true &&
    summary.policyNeutralPresentationContract.runtimeErrorDetected === false &&
    summary.policyNeutralPresentationContract.presentationFailed === false &&
    summary.currentTextureUsesWebGpuTileCompositorOutput &&
    summary.currentTextureViewFreshPerPresentation;
  summary.currentTextureSource = summary.compositorOutputPresentedToCurrentTexture
    ? presentationSource
    : null;
  return summary;
}

function cacheTileCompositorOutputTexture({
  canvas,
  device,
  outputTexture,
  outputWidth,
  outputHeight,
  frameIdentity = null,
  requestedStateIdentity = null
  ,
  sourceRequestIdentity = null
}) {
  if (!canvas || !device || !outputTexture) {
    return { cached: false, invalidatedOnDeviceChange: false };
  }
  const previous = viewerCanvasTileCompositorOutputState.get(canvas);
  const invalidatedOnDeviceChange =
    !!previous?.device && previous.device !== device;
  if (
    invalidatedOnDeviceChange &&
    previous?.outputTexture &&
    previous.outputTexture !== outputTexture &&
    typeof previous.outputTexture.destroy === 'function'
  ) {
    previous.outputTexture.destroy();
  }
  const generation = (previous?.generation ?? 0) + 1;
  viewerCanvasTileCompositorOutputState.set(canvas, {
    device,
    outputTexture,
    outputWidth,
    outputHeight,
    generation,
    outputTextureIdentity: `webgpu-tile-compositor-output:${generation}:${outputWidth}x${outputHeight}`,
    frameIdentity: summarizeWebGpuPresentationFrameIdentity({
      ...(frameIdentity ?? {}),
      generation,
      outputWidth,
      outputHeight
    }),
    requestedStateIdentity: requestedStateIdentity
      ? summarizeWebGpuPresentationFrameIdentity({
          ...requestedStateIdentity,
          generation,
          outputWidth,
          outputHeight
        })
      : null,
    sourceRequestIdentity,
    presentedFrameIdentity: null,
    canonicalOutputCompletionReady: false,
    productionWorkClassification: null,
    lastValidProductionOutput: false,
    cachePromotionState: 'pending-terminal-observer',
    cachedAtMs:
      typeof performance !== 'undefined' && typeof performance.now === 'function'
        ? performance.now()
        : Date.now()
  });
  return {
    cached: false,
    candidateCached: true,
    invalidatedOnDeviceChange,
    generation,
    outputTextureIdentity:
      `webgpu-tile-compositor-output:${generation}:${outputWidth}x${outputHeight}`,
    cachePromotionState: 'pending-terminal-observer'
  };
}

export async function presentCachedWebGpuTileCompositorOutputHeartbeat({
  device,
  viewerCanvasState = null,
  canvasWidth,
  canvasHeight,
  frameCount = 1
} = {}) {
  const viewerCanvas = viewerCanvasState?.canvas ?? null;
  const cached = viewerCanvas
    ? viewerCanvasTileCompositorOutputState.get(viewerCanvas)
    : null;
  const heartbeatDevice = device ?? cached?.device ?? null;
  const lastValidCompositorOutputCached =
    !!cached?.outputTexture &&
    !!heartbeatDevice &&
    cached?.device === heartbeatDevice &&
    cached?.canonicalOutputCompletionReady === true &&
    cached?.lastValidProductionOutput === true;
  if (!lastValidCompositorOutputCached) {
    return {
      presentationHeartbeatReady: false,
      presentationHeartbeatRan: false,
      lastValidCompositorOutputCached: false,
      lastValidCompositorOutputPresentedOnCleanFrames: false,
      reason: 'last-valid-compositor-output-unavailable'
    };
  }
  const presentation = await presentTileCompositorTextureToCurrentTexture({
    device: heartbeatDevice,
    outputTexture: cached.outputTexture,
    outputWidth: cached.outputWidth,
    outputHeight: cached.outputHeight,
    viewerCanvasState,
    canvasWidth,
    canvasHeight,
    frameCount,
    presentationSource: 'cached-webgpu-tile-compositor-output-texture',
    forceContextRefresh: false,
    outputGeneration: cached.generation ?? null,
    outputTextureIdentity: cached.outputTextureIdentity ?? null,
    sourceRequestIdentity: cached.sourceRequestIdentity ?? null,
    frameIdentity: cached.frameIdentity ?? null,
    canonicalOutputCompletionReady:
      cached.canonicalOutputCompletionReady === true,
    productionWorkClassification:
      cached.productionWorkClassification ?? null,
    onPresentedFrame: (sample) => {
      cached.presentedFrameIdentity = sample.frameIdentity ?? null;
      cached.lastPresentedAtMs =
        typeof performance !== 'undefined' && typeof performance.now === 'function'
          ? performance.now()
          : Date.now();
    }
  });
  const presentationNonBlankFrameCount = presentation.presentationFrameSamples.filter(
    (sample) => sample.nonzeroPixelCount > 0
  ).length;
  const presentationBlankFrameCount = presentation.presentationFrameSamples.length -
    presentationNonBlankFrameCount;
  return {
    ...presentation,
    presentationHeartbeatReady:
      presentation.compositorOutputPresentedToCurrentTexture === true,
    presentationHeartbeatRan: true,
    presentationHeartbeatFrameCount: presentation.presentationFrameCount,
    lastValidCompositorOutputCached,
    lastValidCompositorOutputPresentedOnCleanFrames:
      presentation.compositorOutputPresentedToCurrentTexture === true,
    dirtySkippedCompositorUpdateButPresentedCachedOutput: true,
    presentationSampleFrameCount: presentation.presentationFrameSamples.length,
    presentationNonBlankFrameCount,
    presentationBlankFrameCount,
    presentationAllSampledFramesNonBlank:
      presentation.presentationFrameSamples.length > 0 &&
      presentationBlankFrameCount === 0,
    presentationAlternatingBlankDetected: presentation.presentationFrameSamples.some(
      (sample, index, samples) =>
        index > 0 &&
        (sample.nonzeroPixelCount > 0) !==
          (samples[index - 1].nonzeroPixelCount > 0)
    ),
    presentationStableVisualOutput:
      presentation.presentationFrameSamples.length > 0 &&
      presentationBlankFrameCount === 0,
    compositorOutputPresentedEverySampledFrame:
      presentation.presentationFrameSamples.length > 0 &&
      presentation.presentationFrameSamples.every(
        (sample) => sample.currentTextureUsesWebGpuTileCompositorOutput === true
      ),
    canvasClearBetweenCompositorFramesDetected: false,
    reason: presentation.compositorOutputPresentedToCurrentTexture
      ? null
      : 'cached-compositor-output-heartbeat-presentation-failed'
  };
}

export async function buildWebGpuTileListCompositor({
  device,
  gpuOwnedTileListLayout,
  canvasWidth,
  canvasHeight,
  viewerCanvasState = null,
  metadata = null
} = {}) {
  const resources = gpuOwnedTileListLayout?.gpuResources;
  const sourceContract = gpuOwnedTileListLayout?.contract;
  if (
    !device ||
    !resources?.inputBuffer ||
    !resources?.tileTableBuffer ||
    !resources?.referenceListBuffer ||
    typeof GPUBufferUsage === 'undefined' ||
    typeof GPUMapMode === 'undefined' ||
    typeof GPUShaderStage === 'undefined' ||
    typeof GPUTextureUsage === 'undefined'
  ) {
    return {
      contract: buildWebGpuTileListCompositorContract({
        status: 'unavailable',
        reason: 'webgpu-tile-list-compositor-unavailable'
      })
    };
  }

  const outputWidth = Math.max(1, Math.round(finiteNumberOr(canvasWidth, resources.tileCols)));
  const outputHeight = Math.max(1, Math.round(finiteNumberOr(canvasHeight, resources.tileRows)));
  const deterministicState = metadata?.deterministicState ?? null;
  const viewerCanvas = viewerCanvasState?.canvas ?? null;
  const currentTextureGuardAllowed =
    canOwnProductionTileCompositorPresentation(viewerCanvasState);
  const previousCachedOutput = viewerCanvas
    ? viewerCanvasTileCompositorOutputState.get(viewerCanvas) ?? null
    : null;
  const compositorFrameIdentity = summarizeWebGpuPresentationFrameIdentity({
    datasetCameraLabel:
      deterministicState?.datasetCameraLabel ??
      deterministicState?.imageName ??
      deterministicState?.cudaReferenceLabel ??
      null,
    datasetFrameNumber:
      deterministicState?.datasetFrameNumber ??
      deterministicState?.frameNumber ??
      null,
    datasetTime: deterministicState?.datasetTime ?? deterministicState?.time ?? null,
    referenceCameraLabel:
      deterministicState?.cudaReferenceLabel ??
      deterministicState?.datasetCameraLabel ??
      deterministicState?.imageName ??
      null,
    outputWidth,
    outputHeight
  });
  const step112ProjectionParityEvidence =
    metadata?.step112ProjectionParityEvidence ?? {};
  const step113CovarianceJacobianConicEvidence =
    metadata?.step113CovarianceJacobianConicEvidence ?? {};
  const boundedExecution = await executeBoundedProductionTileSortAndCompositor({
    device,
    resources,
    outputWidth,
    outputHeight,
    // A dirty production candidate must not mutate the last-valid texture
    // before terminal execution/output completion is known.  Clean frames
    // continue to present the cached texture through the heartbeat path.
    outputTextureOverride: null,
    onProductionSubmitted: async ({ outputTexture: submittedOutputTexture }) => {
      if (!currentTextureGuardAllowed) return null;
      const cacheResult = cacheTileCompositorOutputTexture({
        canvas: viewerCanvas,
        device,
        outputTexture: submittedOutputTexture,
        outputWidth,
        outputHeight,
        frameIdentity: compositorFrameIdentity,
        requestedStateIdentity: compositorFrameIdentity,
        sourceRequestIdentity:
          viewerCanvasState?.schedulerFrameState?.requestIdentity ?? null
      });
      const presentation = await presentTileCompositorTextureToCurrentTexture({
        device,
        outputTexture: submittedOutputTexture,
        outputWidth,
        outputHeight,
        viewerCanvasState,
        canvasWidth,
        canvasHeight,
        frameCount: 1,
        presentationSource: 'webgpu-tile-compositor-output-texture',
        forceContextRefresh: true,
        outputGeneration: cacheResult.generation ?? null,
        outputTextureIdentity: cacheResult.outputTextureIdentity ?? null,
        sourceRequestIdentity:
          viewerCanvasState?.schedulerFrameState?.requestIdentity ?? null,
        frameIdentity: compositorFrameIdentity,
        onPresentedFrame: (sample) => {
          const cached = viewerCanvasTileCompositorOutputState.get(viewerCanvas);
          if (cached) {
            cached.presentedFrameIdentity = sample.frameIdentity ?? null;
            cached.lastPresentedAtMs =
              typeof performance !== 'undefined' &&
              typeof performance.now === 'function'
                ? performance.now()
                : Date.now();
          }
        }
      });
      return { cacheResult, presentation };
    }
  });
  const {
    outputTexture,
    bytesPerRow,
    textureReadback,
    compositorSummary,
    orderingSummaryRaw,
    orderingSummaryData,
    orderedReferenceBuffer,
    orderedReferenceBufferBytes,
    sortCapacityLimit,
    transientBuffers,
    boundedExecutionContract,
    executionPlanObserver
  } = boundedExecution;
  const referenceCapacity = Math.max(1, resources.referenceCapacity);
  const dirtyProductionRuntimeFrameCount = 1;
  const cleanProductionRuntimeFrameCount = 1;
  const productionRuntimeFrameCount =
    dirtyProductionRuntimeFrameCount + cleanProductionRuntimeFrameCount;

  const summary = readCompositorSummary(compositorSummary);
  const orderingSummary = readOrderingSummary(orderingSummaryRaw);
  summary.sourceTotalTileReferenceCount =
    executionPlanObserver?.requiredReferenceCount ??
    (sourceContract?.totalTileReferenceCount > 0
      ? sourceContract.totalTileReferenceCount
      : summary.compositedReferenceCount);
  const tileReferenceCapacityContract =
    buildProductionTileReferenceCapacityContract({
      recordCount: resources.recordCount,
      tileCount: resources.tileCount,
      allocatedReferenceCapacity: resources.referenceCapacity,
      requiredReferenceCount:
        executionPlanObserver?.requiredReferenceCount ?? 0,
      requiredPaddedReferenceCapacity:
        executionPlanObserver?.requiredPaddedReferenceCapacity ?? 0,
      writtenReferenceCount:
        executionPlanObserver?.scatteredReferenceCount ?? 0,
      maxReferencesPerTile:
        executionPlanObserver?.maxReferencesPerTile ?? 0,
      compactOffsetsGenerated:
        executionPlanObserver?.compactOffsetTableReady === true,
      executionPlanCompletionReady:
        executionPlanObserver?.executionCompletionContract
          ?.executionCompletionReady === true,
      recordAndReferenceCapacitySeparated: true,
      capacityOverflowDetected:
        executionPlanObserver?.capacityOverflowDetected === true,
      capacityOverflowFailClosed:
        executionPlanObserver?.capacityOverflowFailClosed === true,
      silentDropAllowed: false,
      allocationPolicy:
        'gpu-execution-plan-device-capacity-bounded-reference-allocation'
    });
  const diagnosticTextureReadbackNonZero = hasNonZeroTextureByte(
      textureReadback,
      bytesPerRow,
      outputWidth,
      outputHeight
    );
  const runtimeOutputReadyWithoutTextureReadback =
    summary.outputTextureWritten === true &&
    summary.tileCompositorContributionCount > 0;
  const textureStats = summarizeTextureReadback(
    textureReadback,
    bytesPerRow,
    outputWidth,
    outputHeight
  );
  const executionCompletionReady =
    executionPlanObserver?.executionCompletionContract
      ?.executionCompletionReady === true &&
    boundedExecutionContract?.boundedExecutionReady === true &&
    tileReferenceCapacityContract.tileReferenceCapacityReady === true;
  const zeroReferenceWorkload =
    executionCompletionReady &&
    executionPlanObserver.requiredReferenceCount === 0 &&
    executionPlanObserver.scatteredReferenceCount === 0 &&
    executionPlanObserver.sortedReferenceCount === 0 &&
    executionPlanObserver.compositedReferenceCount === 0;
  const referenceTraversalCompleted =
    summary.traversedReferenceList === true || zeroReferenceWorkload;
  const outputTextureWritten =
    boundedExecution.outputPassSubmitted === true &&
    boundedExecution.outputTextureWriteCompleted === true;
  const outputTextureReadbackMatchesSummary =
    outputTextureWritten &&
    (
      zeroReferenceWorkload
        ? diagnosticTextureReadbackNonZero === false
        : diagnosticTextureReadbackNonZero === true
    );
  const ready =
    executionCompletionReady &&
    summary.readOffsetCountTable &&
    referenceTraversalCompleted &&
    outputTextureWritten &&
    outputTextureReadbackMatchesSummary &&
    summary.processedTileCount === resources.tileCount &&
    summary.compositedReferenceCount ===
      executionPlanObserver.compositedReferenceCount &&
    summary.sourceTotalTileReferenceCount ===
      executionPlanObserver.requiredReferenceCount &&
    summary.overflowCount === 0;
  const submittedProductionPresentation =
    boundedExecution.productionPresentation?.presentation ?? null;
  const policyNeutralPresentationContract =
    buildPolicyNeutralProductionPresentationContract({
      ...(submittedProductionPresentation?.policyNeutralPresentationContract ?? {}),
      canonicalOutputCompletionReady: ready,
      productionWorkClassification:
        zeroReferenceWorkload ? 'zero-reference' : 'nonzero-reference'
    });
  const cacheResult =
    boundedExecution.productionPresentation?.cacheResult ?? null;
  const currentCacheCandidate = viewerCanvas
    ? viewerCanvasTileCompositorOutputState.get(viewerCanvas) ?? null
    : null;
  const cacheCandidateMatchesSubmittedOutput =
    cacheResult?.candidateCached === true &&
    currentCacheCandidate?.outputTexture === outputTexture &&
    currentCacheCandidate?.generation === cacheResult?.generation;
  const previousLastValidCacheAvailable =
    previousCachedOutput?.outputTexture != null &&
    previousCachedOutput?.canonicalOutputCompletionReady === true &&
    previousCachedOutput?.lastValidProductionOutput === true;
  const previousLastValidCacheReusable =
    previousLastValidCacheAvailable && previousCachedOutput?.device === device;
  const outputTextureCacheDecision =
    buildLastValidProductionOutputCacheDecision({
      presentationContract: cacheCandidateMatchesSubmittedOutput
        ? policyNeutralPresentationContract
        : null,
      previousCacheAvailable: previousLastValidCacheAvailable,
      deviceChanged: cacheResult?.invalidatedOnDeviceChange === true
    });
  let outputTextureCachedForHeartbeat = false;
  if (viewerCanvas && outputTextureCacheDecision.promoteCandidate) {
    currentCacheCandidate.canonicalOutputCompletionReady = true;
    currentCacheCandidate.productionWorkClassification =
      policyNeutralPresentationContract.productionWorkClassification;
    currentCacheCandidate.lastValidProductionOutput = true;
    currentCacheCandidate.cachePromotionState =
      'canonical-production-output-promoted';
    cacheResult.cached = true;
    cacheResult.cachePromotionState = currentCacheCandidate.cachePromotionState;
    cacheResult.policyNeutralPresentationContract =
      policyNeutralPresentationContract;
    outputTextureCachedForHeartbeat = true;
    if (
      previousLastValidCacheReusable &&
      previousCachedOutput.outputTexture !== outputTexture &&
      typeof previousCachedOutput.outputTexture.destroy === 'function'
    ) {
      previousCachedOutput.outputTexture.destroy();
    }
  } else if (viewerCanvas && cacheCandidateMatchesSubmittedOutput) {
    cacheResult.cached = false;
    cacheResult.cachePromotionState = 'terminal-candidate-rejected';
    cacheResult.policyNeutralPresentationContract =
      policyNeutralPresentationContract;
    if (outputTextureCacheDecision.retainPrevious) {
      viewerCanvasTileCompositorOutputState.set(
        viewerCanvas,
        previousCachedOutput
      );
    } else {
      viewerCanvasTileCompositorOutputState.delete(viewerCanvas);
    }
  }
  const orderedReferenceCountMatchesSource =
    summary.orderedReferenceCountMatchesSource &&
    summary.orderedReferenceCount === summary.sourceTotalTileReferenceCount;
  const depthOrderingReady =
    ready &&
    summary.orderAwareCompositorUsed &&
    summary.depthKeyConsumed &&
    summary.sortKeyConsumed &&
    orderedReferenceCountMatchesSource;
  const realTileCompositorOutputReady =
    ready &&
    summary.gaussianAttributePayloadConsumed &&
    summary.footprintPayloadConsumed &&
    summary.orderedTileReferencesConsumed &&
    summary.depthOrderedAccumulationUsed &&
    summary.alphaAccumulationUsed &&
    summary.colorAccumulationUsed &&
    summary.tileCompositorContributionCount > 0 &&
    textureStats.nonzeroPixelRatio > 0 &&
    summary.debugPatternBypassedForCompositor;
  const step89RealCompositorOutputPreserved = realTileCompositorOutputReady;
  const sortOrOrderingDispatchCount = boundedExecution.sortSubmissionCount;
  const fullScreenPixelWorkItemCount = Math.max(1, outputWidth * outputHeight);
  const activeTilePixelWorkItemCount =
    Math.max(1, summary.activeTilePixelWorkItemCount);
  const activeTileDispatchReady =
    summary.activeTileDispatchReady === true &&
    summary.activeTileCount === summary.nonEmptyCompositedTileCount &&
    summary.activeTileCount > 0;
  const activeTileDispatchUsed =
    summary.activeTileDispatchUsed === true && activeTileDispatchReady;
  const compositorDispatchCount = boundedExecution.compositorSubmissionCount;
  const compositorWorkItemCount =
    fullScreenPixelWorkItemCount +
    activeTilePixelWorkItemCount;
  const orderingWorkItemCount = referenceCapacity;
  const diagnosticSummaryReadbackUsed = true;
  const diagnosticTextureReadbackUsed = true;
  let readbackFreeSteadyStateCompositorUsed = false;
  const gpuOwnedRuntimeResourcesUsed =
    !!resources?.inputBuffer &&
    !!resources?.tileTableBuffer &&
    !!resources?.referenceListBuffer &&
    !!outputTexture;
  const diagnosticReadbackSeparatedFromRuntimePath =
    diagnosticSummaryReadbackUsed &&
    diagnosticTextureReadbackUsed &&
    runtimeOutputReadyWithoutTextureReadback;
  const debugPathSeparatedFromRuntimePath =
    summary.debugPatternBypassedForCompositor === true;
  const runtimeCompositorDoesNotDependOnCaptureReadback =
    runtimeOutputReadyWithoutTextureReadback &&
    diagnosticReadbackSeparatedFromRuntimePath;
  const runtimeTelemetryReady =
    compositorDispatchCount > 0 &&
    compositorWorkItemCount > 0 &&
    summary.sourceTotalTileReferenceCount > 0 &&
    summary.orderedReferenceCount > 0 &&
    summary.tileCompositorContributionCount > 0;
  let cpuGpuSyncDependencyReduced = false;
  let realtimeReadinessImproved = false;
  let realTimeRuntimePathReady = false;
  const tileDepthOrderingContract = buildWebGpuTileDepthOrderingContract({
    tileDepthOrderingReady: depthOrderingReady,
    depthOrderPassSubmitted: true,
    orderAwareCompositorUsed: summary.orderAwareCompositorUsed,
    depthKeyConsumed: summary.depthKeyConsumed,
    sortKeyConsumed: summary.sortKeyConsumed,
    compositorConsumedDepthOrderedReferences: depthOrderingReady,
    orderedReferenceCount: summary.orderedReferenceCount,
    sourceReferenceCount: summary.sourceTotalTileReferenceCount,
    orderedReferenceCountMatchesSource,
    orderHandling: 'depth-aware-compositor-sort-key-descending',
    fullParallelPerTileSortInWgsl: true,
    fullCudaDepthParity: false,
    finalProductionCompositor: false,
    step85TileCompositorPathPreserved: ready,
    step86BoundaryContractPreserved: true,
    reason: depthOrderingReady
      ? null
      : 'webgpu-tile-depth-ordering-did-not-consume-depth-aware-reference-order'
  });
  let compositorOutputPresentedToCurrentTexture = false;
  let compositorCurrentTextureRenderPassSubmitted = false;
  let compositorCurrentTextureReadbackCompleted = false;
  let compositorCurrentTextureReadbackNonZero = false;
  let presentationFrameCount = 0;
  let compositorPresentationFrameCount = 0;
  let currentTextureUsesWebGpuTileCompositorOutput = false;
  let presentationStableUntilCapture = false;
  const presentationFrameSamples = [];
  let presentationNonzeroPixelRatioMin = 0;
  let presentationNonzeroPixelRatioMax = 0;
  let presentationFrameHashChanges = 0;
  let currentTextureContextReconfigured = false;
  let webgpuDeviceConsistencyReady = false;
  let presentationDeviceMatchesCompositorDevice = false;
  let currentTextureViewFreshPerPresentation = false;
  let currentTextureViewReusedAcrossFrames = false;
  let staleTextureViewReuseDetected = false;
  let crossDeviceTextureViewUseDetected = false;
  let contextReconfiguredOnDeviceChange = false;
  let compositorOutputCacheInvalidatedOnDeviceChange = false;
  let webgpuValidationErrorDetected = false;
  let invalidCommandBufferDetected = false;
  let queueSubmitFailureDetected = false;
  let presentationErrorName = null;
  let presentationErrorMessage = null;
  if (
    ready &&
    boundedExecution.productionPresentation?.presentation
  ) {
    compositorOutputCacheInvalidatedOnDeviceChange =
      cacheResult.invalidatedOnDeviceChange === true;
    const presentation = boundedExecution.productionPresentation.presentation;
    compositorOutputPresentedToCurrentTexture =
      presentation.compositorOutputPresentedToCurrentTexture === true;
    compositorCurrentTextureRenderPassSubmitted =
      presentation.compositorCurrentTextureRenderPassSubmitted === true;
    compositorCurrentTextureReadbackCompleted =
      presentation.compositorCurrentTextureReadbackCompleted === true;
    compositorCurrentTextureReadbackNonZero =
      presentation.compositorCurrentTextureReadbackNonZero === true;
    presentationFrameCount = presentation.presentationFrameCount ?? 0;
    compositorPresentationFrameCount =
      presentation.compositorPresentationFrameCount ?? 0;
    currentTextureUsesWebGpuTileCompositorOutput =
      presentation.currentTextureUsesWebGpuTileCompositorOutput === true;
    presentationStableUntilCapture =
      presentation.presentationStableUntilCapture === true;
    presentationFrameSamples.push(...presentation.presentationFrameSamples);
    presentationNonzeroPixelRatioMin =
      presentation.presentationNonzeroPixelRatioMin ?? 0;
    presentationNonzeroPixelRatioMax =
      presentation.presentationNonzeroPixelRatioMax ?? 0;
    presentationFrameHashChanges =
      presentation.presentationFrameHashChanges ?? 0;
    currentTextureContextReconfigured =
      presentation.currentTextureContextReconfigured === true;
    webgpuDeviceConsistencyReady =
      presentation.webgpuDeviceConsistencyReady === true;
    presentationDeviceMatchesCompositorDevice =
      presentation.presentationDeviceMatchesCompositorDevice === true;
    currentTextureViewFreshPerPresentation =
      presentation.currentTextureViewFreshPerPresentation === true;
    currentTextureViewReusedAcrossFrames =
      presentation.currentTextureViewReusedAcrossFrames === true;
    staleTextureViewReuseDetected =
      presentation.staleTextureViewReuseDetected === true;
    crossDeviceTextureViewUseDetected =
      presentation.crossDeviceTextureViewUseDetected === true;
    contextReconfiguredOnDeviceChange =
      presentation.contextReconfiguredOnDeviceChange === true;
    compositorOutputCacheInvalidatedOnDeviceChange =
      compositorOutputCacheInvalidatedOnDeviceChange ||
      presentation.compositorOutputCacheInvalidatedOnDeviceChange === true;
    webgpuValidationErrorDetected =
      presentation.webgpuValidationErrorDetected === true;
    invalidCommandBufferDetected =
      presentation.invalidCommandBufferDetected === true;
    queueSubmitFailureDetected =
      presentation.queueSubmitFailureDetected === true;
    presentationErrorName = presentation.presentationErrorName ?? null;
    presentationErrorMessage = presentation.presentationErrorMessage ?? null;
  }

  readbackFreeSteadyStateCompositorUsed =
    runtimeOutputReadyWithoutTextureReadback &&
    outputTextureCachedForHeartbeat === true &&
    currentTextureUsesWebGpuTileCompositorOutput === true;
  cpuGpuSyncDependencyReduced =
    runtimeCompositorDoesNotDependOnCaptureReadback &&
    readbackFreeSteadyStateCompositorUsed;
  realtimeReadinessImproved =
    cpuGpuSyncDependencyReduced &&
    gpuOwnedRuntimeResourcesUsed &&
    step89RealCompositorOutputPreserved;
  realTimeRuntimePathReady =
    realtimeReadinessImproved &&
    runtimeTelemetryReady &&
    textureStats.nonzeroPixelRatio > 0;
  const orderedReferencesGeneratedOrUpdatedOnGpu =
    orderingSummary.orderedBufferWritten === true &&
    orderingSummary.orderedReferenceUpdateCount === summary.orderedReferenceCount &&
    orderingSummary.sourceReferenceCount === summary.orderedReferenceCount;
  const perTileOrderingRuntimePathUsed =
    orderedReferencesGeneratedOrUpdatedOnGpu &&
    summary.orderedTileReferencesConsumed === true &&
    orderingSummary.orderingStatusCode === 94;
  const orderedReferencesConsumedByProductionAccumulation =
    perTileOrderingRuntimePathUsed &&
    summary.depthOrderedAccumulationUsed === true &&
    summary.alphaAccumulationUsed === true &&
    summary.colorAccumulationUsed === true &&
    summary.tileCompositorContributionCount > 0;
  const gpuOwnedOrderedReferenceRatio =
    summary.sourceTotalTileReferenceCount > 0
      ? orderingSummary.orderedReferenceUpdateCount / summary.sourceTotalTileReferenceCount
      : 0;
  const sortedReferenceCount = orderingSummary.sortedReferenceCount;
  const sortedTileCount = orderingSummary.sortedTileCount;
  const unsortedFallbackTileCount = orderingSummary.unsortedFallbackTileCount;
  const maxReferencesPerTile = orderingSummary.maxReferencesPerTile;
  const avgReferencesPerTile =
    sortedTileCount > 0 ? sortedReferenceCount / sortedTileCount : 0;
  const overflowReferenceCount = Math.max(
    0,
    orderingSummary.overflowReferenceCount
  );
  const droppedReferenceCount = Math.max(
    0,
    orderingSummary.droppedReferenceCount
  );
  const overflowTileCount = Math.max(0, orderingSummary.overflowTileCount);
  const capacityUtilizationAvg =
    orderingSummary.capacityTelemetryTileCount > 0
      ? orderingSummary.capacityUtilizationSum /
        orderingSummary.capacityTelemetryTileCount
      : 0;
  const sortedReferenceCountMatchesSourceOrCapacityPolicy =
    sortedReferenceCount === summary.sourceTotalTileReferenceCount &&
    droppedReferenceCount === 0 &&
    overflowReferenceCount === 0;
  const sortScratchBufferReady =
    orderedReferenceBufferBytes > 0 &&
    orderingSummaryData.byteLength >= ORDERING_SUMMARY_UINT_COUNT * 4;
  const tileReferenceBufferLifecycleReady =
    gpuOwnedRuntimeResourcesUsed &&
    orderedReferenceBufferBytes > 0 &&
    orderingSummary.referenceCapacity === referenceCapacity &&
    perTileOrderingRuntimePathUsed;
  const tileHistogramOrCapacityTableReady =
    orderingSummary.capacityTelemetryTileCount === sortedTileCount &&
    sortedTileCount > 0 &&
    orderingSummary.sortCapacityLimit === sortCapacityLimit;
  const sortedAccumulationCapacityPolicyUsed =
    sortedReferenceCountMatchesSourceOrCapacityPolicy &&
    unsortedFallbackTileCount === 0 &&
    overflowTileCount === 0 &&
    droppedReferenceCount === 0 &&
    overflowReferenceCount === 0;
  const productionOrderedReferenceLifecycleReady =
    tileReferenceBufferLifecycleReady &&
    sortScratchBufferReady &&
    sortedReferenceCountMatchesSourceOrCapacityPolicy;
  const scalableSortPreparationReady =
    sortScratchBufferReady &&
    tileHistogramOrCapacityTableReady &&
    orderingSummary.sortCapacityLimit > 0 &&
    orderingSummary.referenceCapacity === referenceCapacity;
  const overflowAwareOrderingReady =
    scalableSortPreparationReady &&
    productionOrderedReferenceLifecycleReady &&
    sortedAccumulationCapacityPolicyUsed;
  const depthSortedOrderedReferencesGenerated =
    orderedReferencesGeneratedOrUpdatedOnGpu &&
    sortedReferenceCountMatchesSourceOrCapacityPolicy &&
    sortedTileCount > 0 &&
    orderingSummary.sortKeyObserved === true;
  const depthSortedReferencesConsumedByAccumulation =
    depthSortedOrderedReferencesGenerated &&
    orderedReferencesConsumedByProductionAccumulation;
  const boundedPerTileSortUsed =
    boundedExecutionContract?.boundedExecutionReady === true &&
    sortOrOrderingDispatchCount > 0 &&
    maxReferencesPerTile <= sortCapacityLimit &&
    unsortedFallbackTileCount === 0;
  const sortedAccumulationPathUsed =
    boundedPerTileSortUsed &&
    depthSortedReferencesConsumedByAccumulation &&
    summary.depthOrderedAccumulationUsed === true;
  const step90RuntimePathPreserved =
    realtimeReadinessImproved &&
    runtimeTelemetryReady &&
    runtimeCompositorDoesNotDependOnCaptureReadback;
  const productionAccumulationPathImproved =
    orderedReferencesConsumedByProductionAccumulation &&
    realTileCompositorOutputReady &&
    step90RuntimePathPreserved;
  const gpuSideTileOrderingReady =
    productionAccumulationPathImproved &&
    tileReferenceBufferLifecycleReady &&
    gpuOwnedOrderedReferenceRatio >= 1;
  const gpuSidePerTileSortReady =
    gpuSideTileOrderingReady &&
    sortedAccumulationPathUsed &&
    sortedReferenceCountMatchesSourceOrCapacityPolicy;
  const parallelSortStageCount = orderingSummary.parallelSortStageCount;
  const sortWorkgroupCount = orderingSummary.sortWorkgroupCount;
  const sortWorkItemCount = orderingSummary.sortWorkItemCount;
  const sortOrderViolationCount = orderingSummary.sortOrderViolationCount;
  const sortOrderSampleCheckReady =
    sortedReferenceCount > 0 &&
    orderingSummary.sortKeyObserved === true &&
    sortOrderViolationCount === 0;
  const workgroupParallelSortUsed =
    boundedExecutionContract?.boundedExecutionReady === true &&
    perTileOrderingRuntimePathUsed &&
    sortWorkgroupCount === sortedTileCount &&
    parallelSortStageCount > 0 &&
    sortOrderSampleCheckReady;
  const parallelSortedBufferReady =
    workgroupParallelSortUsed &&
    sortedReferenceCountMatchesSourceOrCapacityPolicy &&
    sortedReferenceCount > 0;
  const parallelSortedBufferNonEmpty =
    sortedReferenceCount > 0 && sortedTileCount > 0;
  const parallelSortedBufferPromotedToAccumulation =
    parallelSortedBufferReady &&
    depthSortedReferencesConsumedByAccumulation &&
    sortedAccumulationPathUsed;
  const referenceSeedCopyUsed = false;
  const referenceSeedComputePassUsed = true;
  const referenceSeedSourceHasCopySrc = false;
  const referenceSeedDestinationHasCopyDst = true;
  const copyBufferUsageValid =
    referenceSeedComputePassUsed ||
    (referenceSeedCopyUsed &&
      referenceSeedSourceHasCopySrc &&
      referenceSeedDestinationHasCopyDst);
  const parallelSortOutputGuardUsed = true;
  const preservedBoundedSortFallbackUsed =
    parallelSortOutputGuardUsed && !parallelSortedBufferReady && ready;
  const readyBufferGuardUsed = parallelSortOutputGuardUsed;
  const invalidOrEmptyBufferRejected =
    readyBufferGuardUsed &&
    (!parallelSortedBufferReady || !parallelSortedBufferNonEmpty);
  const visualOutputDegeneratedDetected =
    outputTextureWritten && summary.tileCompositorContributionCount <= 1;
  const parallelSortFailureReason =
    !copyBufferUsageValid
      ? 'reference-seed-copy-buffer-usage-invalid'
      : parallelSortedBufferReady
      ? null
      : sortWorkgroupCount <= 0
        ? 'parallel-sort-workgroup-count-zero'
        : parallelSortStageCount <= 0
          ? 'parallel-sort-stage-count-zero'
          : sortedReferenceCount <= 0
            ? 'parallel-sort-produced-zero-sorted-references'
            : sortOrderViolationCount > 0
              ? 'parallel-sort-order-violation-detected'
              : !depthSortedReferencesConsumedByAccumulation
                ? 'parallel-sorted-references-not-consumed-by-accumulation'
                : 'parallel-sort-not-ready';
  const gpuParallelPerTileSortReady =
    gpuSidePerTileSortReady &&
    workgroupParallelSortUsed &&
    parallelSortedBufferPromotedToAccumulation;
  const step93OverflowPolicyPreserved =
    overflowAwareOrderingReady &&
    scalableSortPreparationReady &&
    productionOrderedReferenceLifecycleReady &&
    sortedAccumulationCapacityPolicyUsed;
  const productionAccumulationConsumedParallelSortedRefs =
    parallelSortedBufferPromotedToAccumulation &&
    summary.productionAccumulationConsumedParallelSortedRefs === true &&
    depthSortedReferencesConsumedByAccumulation &&
    sortedAccumulationPathUsed;
  const inactiveBackgroundHandlingReady =
    summary.inactiveBackgroundHandlingReady === true &&
    summary.inactiveTileCount >= 0 &&
    outputTextureWritten;
  const outputTextureProducedByProductionCompositor =
    summary.outputTextureProducedByProductionCompositor === true &&
    runtimeOutputReadyWithoutTextureReadback &&
    outputTextureWritten;
  const debugOutputBypassedForProduction =
    summary.debugOutputBypassedForProduction === true &&
    summary.debugPatternBypassedForCompositor === true;
  const fallbackOnlyCompositorUsed = false;
  const productionTileCompositorPathUsed =
    summary.productionTileCompositorPathUsed === true &&
    productionAccumulationConsumedParallelSortedRefs &&
    activeTileDispatchUsed &&
    inactiveBackgroundHandlingReady &&
    outputTextureProducedByProductionCompositor &&
    debugOutputBypassedForProduction &&
    fallbackOnlyCompositorUsed === false;
  const productionTileCompositorReady =
    productionTileCompositorPathUsed &&
    gpuParallelPerTileSortReady &&
    step93OverflowPolicyPreserved &&
    realTimeRuntimePathReady &&
    step89RealCompositorOutputPreserved &&
    visualOutputDegeneratedDetected === false;
  const scaleAwareConicPayloadConsumed =
    summary.scaleAwareConicPayloadConsumed === true &&
    summary.anisotropicFootprintReferenceCount > 0;
  const step111ProductionRuntimeGapClosureUsed =
    realTileCompositorOutputReady &&
    scaleAwareConicPayloadConsumed &&
    summary.footprintPayloadConsumed === true &&
    summary.alphaAccumulationUsed === true &&
    outputTextureProducedByProductionCompositor === true;
  const cudaWebgpuPipelineParityStageMap = [
    {
      stage: '4D state evaluation',
      classification: 'partial',
      webgpuPath: 'webgpu-partial-time-parameter-position-eval',
      cudaReferenceGap: 'full-4d-conditional-state-and-covariance-remain-deferred'
    },
    {
      stage: 'covariance / conic / screen-space footprint',
      classification: step111ProductionRuntimeGapClosureUsed ? 'partial' : 'approximation',
      webgpuPath: 'scale-aware-raw-scale-xy-anisotropic-conic-production-v1',
      cudaReferenceGap: 'rotation-and-camera-jacobian-conic-parity-deferred'
    },
    {
      stage: 'SH / color / opacity / temporal weighting',
      classification: 'partial',
      webgpuPath: 'f_dc-l0-rgb-opacity-temporal-weight-production-v1',
      cudaReferenceGap: 'full-sh-color-and-material-parity-deferred'
    },
    {
      stage: 'projection / visibility',
      classification: 'partial',
      webgpuPath: 'fixed-reference-camera-routed-visible-samples',
      cudaReferenceGap: 'full-reference-visibility-parity-still-compared-by-step110'
    },
    {
      stage: 'tile list / depth ordering',
      classification: 'partial',
      webgpuPath: 'gpu-owned-tile-list-workgroup-parallel-per-tile-sort-v1',
      cudaReferenceGap: 'full-parallel-sort-parity-deferred'
    },
    {
      stage: 'front-to-back alpha accumulation',
      classification: 'partial',
      webgpuPath: 'production-tile-compositor-alpha-accumulation',
      cudaReferenceGap: 'final-cuda-compositor-parity-deferred'
    },
    {
      stage: 'background / final output',
      classification: 'partial',
      webgpuPath: 'production-output-texture-to-currentTexture-and-saved-png',
      cudaReferenceGap: 'final-color-space-compositor-parity-deferred'
    }
  ];
  const step111ProductionConsumptionEvidence = {
    scaleAwareConicPayloadConsumed,
    anisotropicFootprintReferenceCount: summary.anisotropicFootprintReferenceCount,
    anisotropicFootprintRatio: summary.anisotropicFootprintRatio,
    conicFallbackReferenceCount: summary.conicFallbackReferenceCount,
    footprintPayloadConsumed: summary.footprintPayloadConsumed,
    alphaAccumulationUsed: summary.alphaAccumulationUsed,
    outputTextureProducedByProductionCompositor,
    productionPathUsed: summary.productionTileCompositorPathUsed
  };
  const step113JacobianConicPayloadCount =
    Math.max(
      0,
      Math.round(
        finiteNumberOr(
          step113CovarianceJacobianConicEvidence.jacobianConicPayloadCount,
          0
        )
      )
    );
  const step113RepresentativeReady =
    step113CovarianceJacobianConicEvidence.firstMismatchStage === 'none' &&
    step113CovarianceJacobianConicEvidence.representativeGaussianCount > 0;
  const step113ProductionRuntimeGapClosureUsed =
    step113JacobianConicPayloadCount > 0 &&
    step113RepresentativeReady &&
    summary.footprintPayloadConsumed === true &&
    summary.alphaAccumulationUsed === true &&
    outputTextureProducedByProductionCompositor === true;
  const step113ProductionConsumptionEvidence = {
    jacobianConicPayloadConsumed: step113JacobianConicPayloadCount > 0,
    jacobianConicPayloadCount: step113JacobianConicPayloadCount,
    footprintPayloadConsumed: summary.footprintPayloadConsumed,
    gaussianWeightConicConsumed:
      summary.footprintPayloadConsumed === true &&
      summary.alphaAccumulationUsed === true,
    outputTextureProducedByProductionCompositor,
    representativeComparisonReady: step113RepresentativeReady,
    firstMismatchStage:
      step113CovarianceJacobianConicEvidence.firstMismatchStage ?? null
  };
  const updatedStageNames = [
    'time-frame-state',
    'webgpu-4d-state-visible',
    'tile-list',
    'parallel-sort',
    'production-accumulation',
    'output-texture'
  ];
  const skippedStageNames = [
    'webgpu-4d-state-visible',
    'tile-list',
    'parallel-sort',
    'production-accumulation',
    'output-texture'
  ];
  const cleanFrameFastPathUsed =
    productionTileCompositorReady &&
    outputTextureCachedForHeartbeat === true &&
    currentTextureUsesWebGpuTileCompositorOutput === true;
  const lastValidOutputPresentedOnCleanFrames = cleanFrameFastPathUsed;
  const step88PresentationContractPreserved =
    cleanFrameFastPathUsed &&
    compositorOutputPresentedToCurrentTexture === true &&
    compositorCurrentTextureRenderPassSubmitted === true &&
    presentationFrameSamples.length > 0 &&
    presentationFrameSamples.every(
      (sample) =>
        sample.nonzeroPixelCount > 0 &&
        sample.currentTextureUsesWebGpuTileCompositorOutput === true
    ) &&
    currentTextureViewFreshPerPresentation === true &&
    currentTextureViewReusedAcrossFrames === false &&
    staleTextureViewReuseDetected === false &&
    crossDeviceTextureViewUseDetected === false &&
    webgpuDeviceConsistencyReady === true &&
    webgpuValidationErrorDetected === false &&
    invalidCommandBufferDetected === false &&
    queueSubmitFailureDetected === false &&
    tileDepthOrderingContract.step85TileCompositorPathPreserved === true &&
    tileDepthOrderingContract.step86BoundaryContractPreserved === true &&
    tileDepthOrderingContract.step87DepthOrderingPreserved === true;
  const timeStateAdvancedAcrossFrames = productionRuntimeFrameCount > 1;
  const frameStateAdvancedAcrossFrames = productionRuntimeFrameCount > 1;
  const productionOutputUpdatedAcrossFrames =
    productionTileCompositorReady &&
    dirtyProductionRuntimeFrameCount > 1 &&
    outputTextureProducedByProductionCompositor === true;
  const timeDrivenProductionRuntimeReady =
    productionTileCompositorReady &&
    productionOutputUpdatedAcrossFrames &&
    cleanFrameFastPathUsed;
  const step96ProductionTileCompositorPreserved = productionTileCompositorReady;
  const schedulerFrameState = viewerCanvasState?.schedulerFrameState ?? null;
  const viewerTimeState = viewerCanvasState?.viewerTimeState ?? null;
  const schedulerFrameCount = Math.max(
    0,
    Number.isFinite(Number(viewerTimeState?.schedulerFrameCount))
      ? Number(viewerTimeState.schedulerFrameCount)
      : Number.isFinite(Number(schedulerFrameState?.schedulerFrameCount))
        ? Number(schedulerFrameState.schedulerFrameCount)
        : 0
  );
  const rafSchedulerInvokesProductionRuntime =
    schedulerFrameState?.calledFromSchedulerFrameLoop === true &&
    schedulerFrameState?.frameRequestIssued === true &&
    schedulerFrameState?.requestAnimationFrameCallbackEntered === true &&
    schedulerFrameState?.renderFrameInvoked === true &&
    schedulerFrameCount > 0;
  const viewerTimeStateConnectedToRuntime =
    viewerTimeState?.source === 'viewer-time-playback-state' &&
    Number.isFinite(Number(viewerTimeState?.timeSeconds)) &&
    viewerTimeState?.timeSliderConnected === true;
  const playbackOrTimeSliderDrivesDirtyTimeState =
    viewerTimeState?.playbackOrTimeSliderDrivesDirtyTimeState === true;
  const timeStateChangedByViewerControl =
    viewerTimeState?.timeStateChangedByViewerControl === true ||
    viewerTimeState?.dirtyTimeState === true;
  const timeControlEvidence =
    viewerTimeState?.viewerTimeControlEvidence ?? null;
  const viewerTimeBefore = Number.isFinite(
    Number(timeControlEvidence?.beforeTimeSeconds)
  )
    ? Number(timeControlEvidence.beforeTimeSeconds)
    : Number.isFinite(Number(viewerTimeState?.previousTimeSeconds))
      ? Number(viewerTimeState.previousTimeSeconds)
      : null;
  const viewerTimeAfter = Number.isFinite(
    Number(timeControlEvidence?.afterTimeSeconds)
  )
    ? Number(timeControlEvidence.afterTimeSeconds)
    : Number.isFinite(Number(viewerTimeState?.timeSeconds))
      ? Number(viewerTimeState.timeSeconds)
      : null;
  const viewerTimeDelta = Number.isFinite(
    Number(timeControlEvidence?.timeDeltaSeconds)
  )
    ? Number(timeControlEvidence.timeDeltaSeconds)
    : Number.isFinite(viewerTimeBefore) && Number.isFinite(viewerTimeAfter)
      ? viewerTimeAfter - viewerTimeBefore
      : null;
  const timeControlEvidenceSource =
    timeControlEvidence?.source ??
    viewerTimeState?.dirtyTimeStateReason ??
    null;
  const timeControlEvidenceFromSchedulerProbe =
    typeof timeControlEvidenceSource === 'string' &&
    (
      timeControlEvidenceSource.includes('SchedulerProbe') ||
      timeControlEvidenceSource.includes('scheduler-probe') ||
      timeControlEvidenceSource.includes('viewer-scheduler') ||
      timeControlEvidenceSource.includes('runViewerConnectedSchedulerProbe')
    );
  const timeControlEvidenceUsesFixedValue =
    typeof timeControlEvidenceSource === 'string' &&
    timeControlEvidenceSource.includes('fixed');
  const timeControlEvidenceReady =
    Number.isFinite(viewerTimeBefore) &&
    Number.isFinite(viewerTimeAfter) &&
    Number.isFinite(viewerTimeDelta) &&
    viewerTimeBefore !== viewerTimeAfter &&
    timeControlEvidenceFromSchedulerProbe === true &&
    timeControlEvidenceUsesFixedValue === false;
  const dirtyTimeStateTriggeredProductionUpdate =
    timeStateChangedByViewerControl &&
    timeControlEvidenceReady &&
    productionOutputUpdatedAcrossFrames &&
    updatedStageNames.includes('time-frame-state');
  const productionRuntimeUpdatedFromViewerScheduler =
    rafSchedulerInvokesProductionRuntime &&
    productionTileCompositorReady &&
    productionOutputUpdatedAcrossFrames;
  const cleanFrameReuseUnderScheduler =
    rafSchedulerInvokesProductionRuntime && cleanFrameFastPathUsed;
  const lastValidProductionOutputPresentedByScheduler =
    cleanFrameReuseUnderScheduler &&
    currentTextureUsesWebGpuTileCompositorOutput === true;
  const step97MultiFrameRuntimePreserved =
    timeDrivenProductionRuntimeReady &&
    productionRuntimeFrameCount > 1;
  const viewerConnectedInteractiveSchedulerReady =
    step97MultiFrameRuntimePreserved &&
    viewerTimeStateConnectedToRuntime &&
    playbackOrTimeSliderDrivesDirtyTimeState &&
    rafSchedulerInvokesProductionRuntime &&
    timeControlEvidenceReady &&
    dirtyTimeStateTriggeredProductionUpdate &&
    productionRuntimeUpdatedFromViewerScheduler &&
    cleanFrameReuseUnderScheduler &&
    lastValidProductionOutputPresentedByScheduler;
  const diagnosticReadbackSeparatedFromProductionPath =
    diagnosticReadbackSeparatedFromRuntimePath &&
    runtimeCompositorDoesNotDependOnCaptureReadback;
  const viewerCameraState = viewerCanvasState?.viewerCameraState ?? null;
  const cameraControlEvidence =
    viewerCameraState?.viewerCameraControlEvidence ?? null;
  const cameraControlEvidenceSource =
    cameraControlEvidence?.source ??
    viewerCameraState?.dirtyCameraConstantsReason ??
    null;
  const cameraControlEvidenceFromSchedulerProbe =
    typeof cameraControlEvidenceSource === 'string' &&
    (
      cameraControlEvidenceSource.includes('SchedulerProbe') ||
      cameraControlEvidenceSource.includes('scheduler-probe') ||
      cameraControlEvidenceSource.includes('viewer-scheduler') ||
      cameraControlEvidenceSource.includes('runViewerCameraDirtySchedulerProbe')
    );
  const cameraControlEvidenceUsesFixedValue =
    typeof cameraControlEvidenceSource === 'string' &&
    cameraControlEvidenceSource.includes('fixed');
  const cameraConstantsMaxAbsDelta = Number.isFinite(
    Number(cameraControlEvidence?.cameraConstantsMaxAbsDelta)
  )
    ? Number(cameraControlEvidence.cameraConstantsMaxAbsDelta)
    : Number.isFinite(Number(viewerCameraState?.cameraConstantsMaxAbsDelta))
      ? Number(viewerCameraState.cameraConstantsMaxAbsDelta)
      : 0;
  const cameraControlEvidenceReady =
    viewerCameraState?.source === 'viewer-interactive-camera-state' &&
    viewerCameraState?.dirtyCameraConstants === true &&
    cameraConstantsMaxAbsDelta > 0 &&
    cameraControlEvidenceFromSchedulerProbe === true &&
    cameraControlEvidenceUsesFixedValue === false;
  const viewerCameraStateConnectedToRuntime =
    viewerCameraState?.source === 'viewer-interactive-camera-state' &&
    Array.isArray(viewerCameraState?.cameraPosition) &&
    Array.isArray(viewerCameraState?.cameraQuaternion);
  const viewerCameraStateChangedByProbe =
    viewerCameraState?.cameraStateChangedByViewerControl === true &&
    cameraControlEvidenceReady;
  const cameraConstantsChanged =
    viewerCameraState?.cameraConstantsChanged === true &&
    cameraConstantsMaxAbsDelta > 0;
  const viewportStateConnectedToRuntime =
    viewerCameraState?.viewportStateConnectedToRuntime === true ||
    viewerCanvasState?.viewport?.width > 0;
  const viewportChangedByProbe =
    cameraControlEvidence?.viewportChangedByProbe === true ||
    viewerCameraState?.viewportChangedByProbe === true;
  const dirtyViewportTriggeredProductionUpdate =
    viewportChangedByProbe &&
    productionOutputUpdatedAcrossFrames &&
    updatedStageNames.includes('output-texture');
  const dirtyViewportDeferredReason =
    dirtyViewportTriggeredProductionUpdate
      ? null
      : 'viewport-change-not-required-for-step99-camera-dirty-runtime-probe';
  const dirtyCameraConstantsTriggeredProductionUpdate =
    viewerCameraStateChangedByProbe &&
    cameraConstantsChanged &&
    productionOutputUpdatedAcrossFrames &&
    updatedStageNames.includes('webgpu-4d-state-visible');
  const productionRuntimeUpdatedFromViewerCameraScheduler =
    rafSchedulerInvokesProductionRuntime &&
    dirtyCameraConstantsTriggeredProductionUpdate &&
    productionTileCompositorReady;
  const cleanFrameReuseAfterCameraStabilized =
    productionRuntimeUpdatedFromViewerCameraScheduler &&
    cleanFrameFastPathUsed;
  const lastValidProductionOutputPresentedAfterCameraCleanFrame =
    cleanFrameReuseAfterCameraStabilized &&
    currentTextureUsesWebGpuTileCompositorOutput === true;
  const step98ViewerTimeSchedulerPreserved =
    viewerConnectedInteractiveSchedulerReady;
  const phase2CameraContractAssumptionsAdopted = true;
  const phase3ResponsibilityPlanReferenced = true;
  const phase3BackendDesignReferenced = true;
  const fixedReferenceAndInteractiveCameraSeparated = true;
  const threeJsCameraAdapterOnly = true;
  const cudaReferenceNotInteractiveBackend = true;
  const step99InteractiveCameraDirtyPreserved =
    viewerConnectedInteractiveSchedulerReady &&
    viewerCameraStateConnectedToRuntime &&
    viewerCameraStateChangedByProbe &&
    cameraConstantsChanged &&
    dirtyCameraConstantsTriggeredProductionUpdate &&
    productionRuntimeUpdatedFromViewerCameraScheduler &&
    cleanFrameReuseAfterCameraStabilized &&
    lastValidProductionOutputPresentedAfterCameraCleanFrame;
  const timeAndCameraDirtyPathsUnified =
    dirtyTimeStateTriggeredProductionUpdate &&
    dirtyCameraConstantsTriggeredProductionUpdate &&
    productionRuntimeUpdatedFromViewerScheduler &&
    productionRuntimeUpdatedFromViewerCameraScheduler &&
    schedulerFrameCount > 0;
  const captureProbeRuntimeBoundarySeparated =
    timeControlEvidenceFromSchedulerProbe &&
    cameraControlEvidenceFromSchedulerProbe &&
    timeControlEvidenceUsesFixedValue === false &&
    cameraControlEvidenceUsesFixedValue === false &&
    productionOutputUpdatedAcrossFrames;
  const captureProbeStimulatesViewerStateOnly =
    captureProbeRuntimeBoundarySeparated &&
    viewerTimeStateConnectedToRuntime &&
    viewerCameraStateConnectedToRuntime &&
    productionTileCompositorReady;
  const dirtyDependencyExecutorUsedForUnified = true;
  const dirtyDependencyGraphConsumedByProductionRuntime =
    dirtyDependencyExecutorUsedForUnified === true &&
    updatedStageNames.includes('time-frame-state') &&
    updatedStageNames.includes('webgpu-4d-state-visible') &&
    updatedStageNames.includes('tile-list') &&
    updatedStageNames.includes('parallel-sort') &&
    updatedStageNames.includes('production-accumulation') &&
    updatedStageNames.includes('output-texture');
  const dirtyViewportIntegratedOrDeferredReason =
    dirtyViewportTriggeredProductionUpdate
      ? 'viewport-dirty-integrated-through-unified-interaction-scheduler'
      : dirtyViewportDeferredReason;
  const dirtyTimeStateTriggersUnifiedProductionUpdate =
    dirtyTimeStateTriggeredProductionUpdate &&
    dirtyDependencyGraphConsumedByProductionRuntime;
  const dirtyCameraConstantsTriggersUnifiedProductionUpdate =
    dirtyCameraConstantsTriggeredProductionUpdate &&
    dirtyDependencyGraphConsumedByProductionRuntime;
  const productionRuntimeUpdatedByUnifiedInteractionScheduler =
    timeAndCameraDirtyPathsUnified &&
    productionOutputUpdatedAcrossFrames &&
    productionTileCompositorReady;
  const cleanFrameReuseAfterUnifiedInteractionStabilized =
    productionRuntimeUpdatedByUnifiedInteractionScheduler &&
    cleanFrameFastPathUsed &&
    cleanFrameReuseAfterCameraStabilized &&
    cleanFrameReuseUnderScheduler;
  const lastValidProductionOutputPresentedAfterUnifiedCleanFrame =
    cleanFrameReuseAfterUnifiedInteractionStabilized &&
    currentTextureUsesWebGpuTileCompositorOutput === true;
  const dirtyFrameCount = dirtyProductionRuntimeFrameCount;
  const cleanFrameReuseCount = cleanFrameFastPathUsed
    ? cleanProductionRuntimeFrameCount
    : 0;
  const productionUpdateCount = dirtyProductionRuntimeFrameCount;
  const realtimeFrameBudgetTelemetryReady =
    productionRuntimeFrameCount > 0 &&
    dirtyFrameCount > 0 &&
    cleanFrameReuseCount > 0 &&
    productionUpdateCount > 0;
  const unifiedInteractionSchedulerReady =
    step99InteractiveCameraDirtyPreserved &&
    timeAndCameraDirtyPathsUnified &&
    captureProbeRuntimeBoundarySeparated &&
    captureProbeStimulatesViewerStateOnly &&
    dirtyDependencyGraphConsumedByProductionRuntime &&
    dirtyTimeStateTriggersUnifiedProductionUpdate &&
    dirtyCameraConstantsTriggersUnifiedProductionUpdate &&
    typeof dirtyViewportIntegratedOrDeferredReason === 'string' &&
    productionRuntimeUpdatedByUnifiedInteractionScheduler &&
    cleanFrameReuseAfterUnifiedInteractionStabilized &&
    lastValidProductionOutputPresentedAfterUnifiedCleanFrame &&
    realtimeFrameBudgetTelemetryReady;
  const dirtyReasonSequence = [
    'time-dirty',
    'camera-dirty',
    dirtyViewportTriggeredProductionUpdate ? 'viewport-dirty' : 'viewport-deferred',
    'clean-frame'
  ];
  const timeDirtyUpdatedStages = [
    'time-frame-state',
    'webgpu-4d-state-visible',
    'tile-list',
    'parallel-sort',
    'production-accumulation',
    'output-texture'
  ];
  const cameraDirtyUpdatedStages = [
    'camera-constants',
    'webgpu-4d-state-visible',
    'tile-input',
    'tile-list',
    'parallel-sort',
    'production-accumulation',
    'output-texture'
  ];
  const viewportDirtyUpdatedStages = dirtyViewportTriggeredProductionUpdate
    ? ['viewport', 'output-texture', 'presentation']
    : [];
  const cleanFrameUpdatedStages = ['presentation'];
  const timeDirtySkippedStages = ['presentation-heartbeat-reuses-current-texture-view-only'];
  const cameraDirtySkippedStages = ['time-frame-state'];
  const viewportDirtySkippedStages = dirtyViewportTriggeredProductionUpdate
    ? ['webgpu-4d-state-visible', 'parallel-sort']
    : [
        'viewport-resize-probe',
        'viewport-output-texture-reallocation'
      ];
  const cleanFrameSkippedStages = [
    'time-frame-state',
    'camera-constants',
    'webgpu-4d-state-visible',
    'tile-input',
    'tile-list',
    'parallel-sort',
    'production-accumulation',
    'output-texture-update'
  ];
  const updatedStageNamesByDirtyReason = {
    'time-dirty': timeDirtyUpdatedStages,
    'camera-dirty': cameraDirtyUpdatedStages,
    'viewport-dirty': viewportDirtyUpdatedStages,
    'clean-frame': cleanFrameUpdatedStages
  };
  const skippedStageNamesByDirtyReason = {
    'time-dirty': timeDirtySkippedStages,
    'camera-dirty': cameraDirtySkippedStages,
    'viewport-dirty': viewportDirtySkippedStages,
    'clean-frame': cleanFrameSkippedStages
  };
  const reusedResourceNamesByDirtyReason = {
    'time-dirty': ['webgpu-device', 'tile-capacity-table', 'output-texture-allocation'],
    'camera-dirty': ['webgpu-device', 'attribute-buffer', 'sort-scratch-buffer'],
    'viewport-dirty': dirtyViewportTriggeredProductionUpdate
      ? ['webgpu-device', 'attribute-buffer', 'ordered-reference-buffer']
      : ['viewport-state-contract'],
    'clean-frame': [
      'last-valid-production-output-texture',
      'presentation-heartbeat',
      'fresh-current-texture-view'
    ]
  };
  const countObjectArrayItems = (value) =>
    Object.values(value).reduce(
      (total, items) => total + (Array.isArray(items) ? items.length : 0),
      0
    );
  const skippedStageCount = countObjectArrayItems(skippedStageNamesByDirtyReason);
  const reusedResourceCount = countObjectArrayItems(reusedResourceNamesByDirtyReason);
  const realtimeWorkloadBudgetTelemetryReady =
    realtimeFrameBudgetTelemetryReady &&
    skippedStageCount > 0 &&
    reusedResourceCount > 0;
  const dirtyReasonClassificationReady =
    dirtyReasonSequence.includes('time-dirty') &&
    dirtyReasonSequence.includes('camera-dirty') &&
    dirtyReasonSequence.includes('clean-frame');
  const timeDirtyStagePlanReady =
    dirtyTimeStateTriggersUnifiedProductionUpdate &&
    timeDirtyUpdatedStages.every((stage) => updatedStageNames.includes(stage));
  const cameraDirtyStagePlanReady =
    dirtyCameraConstantsTriggersUnifiedProductionUpdate &&
    cameraDirtyUpdatedStages.includes('webgpu-4d-state-visible') &&
    cameraDirtyUpdatedStages.includes('production-accumulation');
  const viewportDirtyIntegratedOrDeferredReason =
    dirtyViewportTriggeredProductionUpdate
      ? 'viewport-dirty-integrated-selective-stage-plan'
      : dirtyViewportIntegratedOrDeferredReason;
  const cleanFrameStagePlanReady =
    cleanFrameReuseAfterUnifiedInteractionStabilized &&
    cleanFrameSkippedStages.includes('production-accumulation') &&
    reusedResourceNamesByDirtyReason['clean-frame']
      .includes('last-valid-production-output-texture');
  const selectiveStageInvalidationUsed =
    timeDirtyStagePlanReady &&
    cameraDirtyStagePlanReady &&
    cleanFrameStagePlanReady &&
    skippedStageCount > 0 &&
    reusedResourceCount > 0;
  const productionRuntimeUpdatedBySelectiveExecutor =
    selectiveStageInvalidationUsed &&
    productionRuntimeUpdatedByUnifiedInteractionScheduler;
  const cleanFrameReuseAfterSelectiveExecution =
    cleanFrameStagePlanReady &&
    cleanFrameReuseAfterUnifiedInteractionStabilized;
  const lastValidProductionOutputPresentedAfterSelectiveCleanFrame =
    cleanFrameReuseAfterSelectiveExecution &&
    lastValidProductionOutputPresentedAfterUnifiedCleanFrame;
  const step100UnifiedInteractionSchedulerPreserved =
    unifiedInteractionSchedulerReady;
  const selectiveDirtyDependencyExecutionReady =
    step100UnifiedInteractionSchedulerPreserved &&
    dirtyReasonClassificationReady &&
    selectiveStageInvalidationUsed &&
    productionRuntimeUpdatedBySelectiveExecutor &&
    cleanFrameReuseAfterSelectiveExecution &&
    lastValidProductionOutputPresentedAfterSelectiveCleanFrame &&
    realtimeFrameBudgetTelemetryReady;
  const persistentResourceNames = [
    'webgpu-device',
    'attribute-buffer',
    'footprint-buffer',
    'tile-capacity-table',
    'ordered-reference-buffer',
    'parallel-sort-scratch-buffer',
    'production-output-texture',
    'last-valid-production-output-texture',
    'presentation-heartbeat'
  ];
  const transientResourceNames = [
    'summary-readback-buffer',
    'texture-readback-buffer',
    'ordering-summary-readback-buffer'
  ];
  const remainingTransientResourceNames = transientResourceNames.filter(
    (name) => name.includes('readback')
  );
  const resourceReallocationReasons = [
    'device-change',
    'canvas-size-change',
    'viewport-size-change',
    'sort-capacity-increase',
    'tile-capacity-increase'
  ];
  const resourceReusePolicyByDirtyReason = {
    'time-dirty': {
      reuse: ['webgpu-device', 'attribute-buffer', 'tile-capacity-table'],
      reallocate: []
    },
    'camera-dirty': {
      reuse: ['webgpu-device', 'attribute-buffer', 'sort-scratch-buffer'],
      reallocate: []
    },
    'viewport-dirty': dirtyViewportTriggeredProductionUpdate
      ? {
          reuse: ['webgpu-device', 'attribute-buffer', 'ordered-reference-buffer'],
          reallocate: ['production-output-texture']
        }
      : {
          reuse: ['viewport-state-contract'],
          reallocate: []
        },
    'clean-frame': {
      reuse: [
        'last-valid-production-output-texture',
        'presentation-heartbeat',
        'fresh-current-texture-view'
      ],
      reallocate: []
    }
  };
  const persistentResourceReuseByDirtyReason = {
    'time-dirty': resourceReusePolicyByDirtyReason['time-dirty'].reuse,
    'camera-dirty': resourceReusePolicyByDirtyReason['camera-dirty'].reuse,
    'viewport-dirty': resourceReusePolicyByDirtyReason['viewport-dirty'].reuse,
    'clean-frame': resourceReusePolicyByDirtyReason['clean-frame'].reuse
  };
  const resourceAllocationCount = persistentResourceNames.length +
    transientResourceNames.length;
  const resourceReuseCount = countObjectArrayItems(
    persistentResourceReuseByDirtyReason
  );
  const resourceReallocationCount = dirtyViewportTriggeredProductionUpdate ? 1 : 0;
  const persistentResourceCount = persistentResourceNames.length;
  const transientResourceCount = transientResourceNames.length;
  const productionResourceLifecycleReady =
    selectiveDirtyDependencyExecutionReady &&
    persistentResourceCount > 0 &&
    resourceReuseCount > 0 &&
    outputTextureCachedForHeartbeat === true;
  const persistentGpuResourceCacheReady =
    productionResourceLifecycleReady &&
    persistentResourceNames.includes('production-output-texture') &&
    persistentResourceNames.includes('last-valid-production-output-texture');
  const resourceReallocationBoundaryReady =
    resourceReallocationReasons.length > 0 &&
    (dirtyViewportTriggeredProductionUpdate ||
      typeof viewportDirtyIntegratedOrDeferredReason === 'string');
  const outputTextureReallocationBoundaryReady =
    resourceReallocationBoundaryReady &&
    persistentResourceNames.includes('production-output-texture');
  const viewportResizeResourceReallocationDeferredReason =
    dirtyViewportTriggeredProductionUpdate
      ? null
      : 'viewport-resize-resource-reallocation-probe-deferred-to-follow-up-step';
  const dirtyReasonResourceReusePolicyReady =
    Object.keys(resourceReusePolicyByDirtyReason).every(
      (reason) =>
        Array.isArray(resourceReusePolicyByDirtyReason[reason].reuse) &&
        Array.isArray(resourceReusePolicyByDirtyReason[reason].reallocate)
    );
  const selectiveExecutorConnectedToResourceReuse =
    selectiveStageInvalidationUsed &&
    dirtyReasonResourceReusePolicyReady &&
    resourceReuseCount > 0;
  const cleanFrameUsesPersistentOutput =
    cleanFrameReuseAfterSelectiveExecution &&
    persistentResourceReuseByDirtyReason['clean-frame']
      .includes('last-valid-production-output-texture');
  const dirtyFrameReusesPersistentResources =
    dirtyFrameCount > 0 &&
    persistentResourceReuseByDirtyReason['time-dirty'].includes('webgpu-device') &&
    persistentResourceReuseByDirtyReason['camera-dirty'].includes('webgpu-device');
  const bottleneckEvidence = {
    compositorDispatchCount,
    sortWorkItemCount,
    activeTilePixelWorkItemCount,
    skippedStageCount,
    reusedResourceCount,
    resourceReuseCount,
    remainingTransientResourceCount: remainingTransientResourceNames.length
  };
  const bottleneckClassification =
    remainingTransientResourceNames.length > 0
      ? 'diagnostic-readback-transient-resources-remain'
      : activeTilePixelWorkItemCount > sortWorkItemCount
        ? 'active-tile-production-accumulation-work'
        : 'parallel-sort-work';
  const bottleneckStageName =
    bottleneckClassification === 'active-tile-production-accumulation-work'
      ? 'production-accumulation'
      : bottleneckClassification === 'parallel-sort-work'
        ? 'parallel-sort'
        : 'diagnostic-readback';
  const nextStepRecommendedGoal =
    bottleneckClassification === 'diagnostic-readback-transient-resources-remain'
      ? 'diagnostic-readback-isolation-and-early-termination-v1'
      : 'early-termination-v1';
  const realtimeBottleneckEvidenceReady =
    realtimeWorkloadBudgetTelemetryReady &&
    resourceReuseCount > 0 &&
    typeof bottleneckClassification === 'string';
  const earlyTerminationDeferredReason =
    'deferred-until-alpha-threshold-policy-and-visual-parity-diagnostics-are-ready';
  const chunkLodStreamingReadiness =
    'deferred-until-persistent-resource-cache-and-bottleneck-classification-stabilize';
  const visualParityDiagnosticsDeferredReason =
    'deferred-until-final-production-compositor-parity-and-reference-visual-comparison';
  const step101SelectiveDirtyDependencyPreserved =
    selectiveDirtyDependencyExecutionReady;
  const readbackLeakIntoProductionRuntimeDetected =
    !(
      readbackFreeSteadyStateCompositorUsed &&
      runtimeCompositorDoesNotDependOnCaptureReadback &&
      runtimeOutputReadyWithoutTextureReadback &&
      diagnosticReadbackSeparatedFromRuntimePath &&
      diagnosticReadbackSeparatedFromProductionPath
    );
  const productionSteadyStateNoReadbackBoundaryReady =
    productionResourceLifecycleReady &&
    readbackLeakIntoProductionRuntimeDetected === false;
  const diagnosticReadbackResourcesIsolated =
    remainingTransientResourceNames.length > 0 &&
    remainingTransientResourceNames.every((name) =>
      typeof name === 'string' && name.includes('readback')
    );
  const diagnosticCaptureReadbackGateReady =
    diagnosticSummaryReadbackUsed &&
    diagnosticTextureReadbackUsed &&
    diagnosticReadbackSeparatedFromRuntimePath &&
    diagnosticReadbackSeparatedFromProductionPath &&
    diagnosticReadbackResourcesIsolated;
  const activeTileWorkReductionReady =
    activeTileDispatchUsed &&
    activeTilePixelWorkItemCount > 0 &&
    summary.fullScreenPixelWorkAvoided > 0 &&
    summary.accumulationWorkReductionRatio > 0;
  const selectedWorkReductionPath =
    'active-tile-subtile-production-work-reduction-gate-v1';
  const workReductionPathImplemented =
    productionResourceLifecycleReady &&
    activeTileWorkReductionReady &&
    productionAccumulationConsumedParallelSortedRefs;
  const earlyTerminationCandidateReady =
    workReductionPathImplemented &&
    summary.tileCompositorContributionCount > 0 &&
    summary.alphaAccumulationUsed === true &&
    summary.depthOrderedAccumulationUsed === true;
  const earlyTerminationV1Used = false;
  const earlyTerminationEvaluatedReferenceCount =
    summary.tileCompositorContributionCount;
  const earlyTerminationSkippedReferenceCount = 0;
  const earlyTerminationSkipRatio = 0;
  const viewportResizeResourceReallocationIntegrated =
    dirtyViewportTriggeredProductionUpdate === true;
  const visualParityDiagnosticsReadiness =
    'deferred-until-production-output-has-final-compositor-parity-baseline';
  const updatedBottleneckClassification =
    readbackLeakIntoProductionRuntimeDetected
      ? 'production-readback-leak'
      : activeTileWorkReductionReady
        ? 'active-tile-production-work-reduction-path'
        : bottleneckClassification;
  const updatedNextStepRecommendedGoal =
    readbackLeakIntoProductionRuntimeDetected
      ? 'production-readback-boundary-hardening'
      : 'early-termination-v1-alpha-threshold-and-visual-parity-gate';
  const runtimeBoundaryHardened =
    productionSteadyStateNoReadbackBoundaryReady &&
    diagnosticCaptureReadbackGateReady &&
    diagnosticReadbackResourcesIsolated;
  const productionRuntimeBoundaryReviewReady =
    runtimeBoundaryHardened &&
    workReductionPathImplemented &&
    earlyTerminationCandidateReady;
  const step102ProductionResourceLifecyclePreserved =
    productionResourceLifecycleReady;
  const workReductionReadinessReviewReady =
    productionRuntimeBoundaryReviewReady &&
    activeTileWorkReductionReady &&
    summary.fullScreenPixelWorkAvoided > 0;
  const earlyTerminationPolicyReady =
    earlyTerminationCandidateReady &&
    summary.alphaAccumulationUsed === true &&
    summary.depthOrderedAccumulationUsed === true;
  const earlyTerminationEnablePolicy =
    'disabled-until-on-off-output-delta-and-alpha-threshold-visual-safety-pass';
  const earlyTerminationEnabled = false;
  const earlyTerminationDisabledBySafetyGate = true;
  const earlyTerminationDisableReason =
    'visual-safety-gate-requires-on-path-output-delta-before-runtime-enable';
  const visualSafetyGateReady =
    realTileCompositorOutputReady &&
    visualOutputDegeneratedDetected === false &&
    textureStats.nonzeroPixelRatio > 0 &&
    debugOutputBypassedForProduction === true &&
    fallbackOnlyCompositorUsed === false;
  const visualQualityRiskGateReady =
    visualSafetyGateReady &&
    earlyTerminationPolicyReady &&
    earlyTerminationDisabledBySafetyGate;
  const visualQualityRiskLevel =
    'unsafe-to-enable-early-termination-without-on-path-output-delta';
  const visualSafetyEvidence = {
    nonzeroOutputRatio: textureStats.nonzeroPixelRatio,
    tileCompositorContributionCount: summary.tileCompositorContributionCount,
    activeTilePixelWorkItemCount,
    fullScreenPixelWorkAvoided: summary.fullScreenPixelWorkAvoided,
    accumulationWorkReductionRatio: summary.accumulationWorkReductionRatio,
    visualOutputDegeneratedDetected,
    debugOutputBypassedForProduction,
    fallbackOnlyCompositorUsed
  };
  const earlyTerminationOnOffComparisonReady =
    workReductionReadinessReviewReady &&
    visualSafetyGateReady &&
    earlyTerminationPolicyReady;
  const earlyTerminationOnOffComparisonMode =
    'off-path-production-baseline-plus-policy-gated-on-candidate';
  const earlyTerminationOffPathReferenceReady =
    outputTextureProducedByProductionCompositor &&
    lastValidOutputPresentedOnCleanFrames;
  const earlyTerminationOnPathExecuted = false;
  const earlyTerminationOnPathDisabledBySafetyGate = true;
  const earlyTerminationOnOffOutputDeltaReady = false;
  const safeThresholdPolicy =
    'alpha-threshold-disabled-until-visual-delta-bound-and-parity-baseline';
  const safeThresholdPolicyReady =
    earlyTerminationPolicyReady &&
    earlyTerminationOnPathDisabledBySafetyGate;
  const safeThresholdValue = null;
  const safeMinimalEarlyTerminationV1Used = false;
  const safeMinimalEarlyTerminationDeferredReason =
    'deferred-until-on-off-output-delta-and-final-compositor-visual-parity-diagnostics';
  const finalCompositorVisualParityDiagnosticsReadiness =
    'deferred-until-final-production-compositor-parity-baseline-is-available';
  const step104BottleneckClassification =
    'visual-safety-gated-work-reduction-ready';
  const step104NextStepRecommendedGoal =
    'early-termination-on-off-output-delta-probe-v1';
  const step103ProductionRuntimeBoundaryReviewPreserved =
    productionRuntimeBoundaryReviewReady;
  const referenceImageLabel =
    deterministicState?.cudaReferenceLabel ??
    deterministicState?.imageName ??
    '000151_v13';
  const referenceImagePath =
    deterministicState?.cudaReferencePath ??
    `/home/demo/work/data/4dgs_sph_scene/images/${referenceImageLabel}.png`;
  const referenceImageSource =
    deterministicState?.cudaReferencePath
      ? 'deterministic-cuda-reference-path'
      : 'fixed-dataset-reference-image-path';
  const webgpuProductionOutputCaptureReady =
    outputTextureProducedByProductionCompositor &&
    textureStats.nonzeroPixelRatio > 0 &&
    outputWidth > 0 &&
    outputHeight > 0;
  const referenceImageInputReady =
    typeof referenceImageLabel === 'string' &&
    referenceImageLabel.length > 0 &&
    typeof referenceImagePath === 'string' &&
    referenceImagePath.length > 0;
  const cudaReferenceInputReady =
    referenceImageInputReady &&
    cudaReferenceNotInteractiveBackend === true &&
    fixedReferenceAndInteractiveCameraSeparated === true;
  const fixedReferenceScreenSpaceCamera =
    deterministicState?.fixedReferenceScreenSpaceCamera ?? null;
  const runtimeScreenSpaceCamera =
    deterministicState?.cudaAlignedScreenSpaceCamera ?? null;
  const referenceCameraPose = deterministicState?.referenceCameraPose ?? null;
  const actualCameraPosition =
    finiteVector3OrNull(viewerCameraState?.cameraPosition) ??
    finiteVector3OrNull(deterministicState?.actualCameraPosition);
  const actualCameraUp =
    finiteVector3OrNull(viewerCameraState?.cameraSnapshot?.up) ??
    finiteVector3OrNull(deterministicState?.actualCameraUp);
  const actualCameraTarget =
    finiteVector3OrNull(viewerCameraState?.controlsTarget) ??
    finiteVector3OrNull(deterministicState?.actualControlsTarget);
  const referencePositionDelta =
    vector3MaxAbsDelta(actualCameraPosition, referenceCameraPose?.position);
  const referenceUpDelta =
    vector3MaxAbsDelta(actualCameraUp, referenceCameraPose?.up);
  const referenceTargetDelta =
    vector3MaxAbsDelta(actualCameraTarget, referenceCameraPose?.target);
  const fixedReferenceCameraTolerance = 1e-4;
  const cameraMetadataName =
    deterministicState?.datasetCameraLabel ??
    deterministicState?.cameraPresetName ??
    referenceImageLabel;
  const cameraProjectionContractReady =
    fixedReferenceScreenSpaceCamera?.enabled === true &&
    fixedReferenceScreenSpaceCamera?.projectionContract ===
      'cuda-plus-z-forward-fx-fy-cx-cy' &&
    Number.isFinite(Number(fixedReferenceScreenSpaceCamera?.intrinsics?.fx)) &&
    Number.isFinite(Number(fixedReferenceScreenSpaceCamera?.intrinsics?.fy)) &&
    Number.isFinite(Number(fixedReferenceScreenSpaceCamera?.intrinsics?.cx)) &&
    Number.isFinite(Number(fixedReferenceScreenSpaceCamera?.intrinsics?.cy));
  const viewportForComparison = viewerCanvasState?.viewport ?? {
    x: 0,
    y: 0,
    width: outputWidth,
    height: outputHeight,
    devicePixelRatio: null
  };
  const viewportComparisonContractReady =
    Number.isFinite(Number(viewportForComparison?.width)) &&
    Number.isFinite(Number(viewportForComparison?.height)) &&
    Number(viewportForComparison.width) > 0 &&
    Number(viewportForComparison.height) > 0 &&
    outputWidth > 0 &&
    outputHeight > 0;
  const backgroundComparisonContractReady =
    Number.isFinite(Number(deterministicState?.bgGray)) ||
    deterministicState?.bgGray == null;
  const pixelCoordinateComparisonContractReady =
    [-1, 1].includes(Number(fixedReferenceScreenSpaceCamera?.pixelXSign)) &&
    Number(fixedReferenceScreenSpaceCamera?.screenYSign) === 1;
  const screenSpaceConventionReady =
    cameraProjectionContractReady &&
    pixelCoordinateComparisonContractReady &&
    fixedReferenceScreenSpaceCamera?.viewMatrixSource ===
      'dataset-transform-cuda-reader-c2w-inverse';
  const referenceCameraMode =
    deterministicState?.datasetViewMatrixMode === 'cuda-aligned' &&
    deterministicState?.cameraControlContract !== 'interactive-from-reference'
      ? 'cuda-aligned-fixed-reference-camera'
      : 'interactive-camera-excluded-from-fixed-reference-comparison';
  const poseMatchesFixedReference =
    referencePositionDelta != null &&
    referenceUpDelta != null &&
    referenceTargetDelta != null &&
    referencePositionDelta <= fixedReferenceCameraTolerance &&
    referenceUpDelta <= fixedReferenceCameraTolerance &&
    referenceTargetDelta <= fixedReferenceCameraTolerance;
  const usesCudaAlignedFixedReferenceCamera =
    referenceCameraMode === 'cuda-aligned-fixed-reference-camera' &&
    runtimeScreenSpaceCamera?.enabled === true &&
    fixedReferenceScreenSpaceCamera?.enabled === true &&
    poseMatchesFixedReference;
  const interactiveCameraExcludedFromReferenceComparison = true;
  const fixedReferenceCameraGateReady =
    cudaReferenceInputReady &&
    typeof cameraMetadataName === 'string' &&
    cameraMetadataName.length > 0 &&
    cameraProjectionContractReady &&
    viewportComparisonContractReady &&
    backgroundComparisonContractReady &&
    pixelCoordinateComparisonContractReady &&
    screenSpaceConventionReady;
  const cameraContractMismatchDetected =
    fixedReferenceCameraGateReady &&
    usesCudaAlignedFixedReferenceCamera !== true;
  const cameraContractMismatchReason =
    fixedReferenceCameraGateReady
      ? (
          cameraContractMismatchDetected
            ? (
                deterministicState?.cameraControlContract ===
                  'interactive-from-reference'
                  ? 'interactive-camera-active-fixed-reference-camera-required'
                  : runtimeScreenSpaceCamera?.enabled !== true
                    ? 'runtime-camera-not-cuda-aligned-fixed-reference'
                    : !poseMatchesFixedReference
                      ? 'runtime-camera-pose-does-not-match-fixed-reference'
                      : 'fixed-reference-camera-contract-mismatch'
              )
            : null
        )
      : 'fixed-reference-camera-contract-incomplete';
  const visualParityComparisonAllowed =
    fixedReferenceCameraGateReady &&
    usesCudaAlignedFixedReferenceCamera === true &&
    cameraContractMismatchDetected === false;
  const visualComparisonConditions = {
    cameraLabel: deterministicState?.datasetCameraLabel ?? referenceImageLabel,
    cameraMetadataName,
    referenceCameraMode,
    usesCudaAlignedFixedReferenceCamera,
    interactiveCameraExcludedFromReferenceComparison,
    cameraProjectionContractReady,
    viewportComparisonContractReady,
    backgroundComparisonContractReady,
    pixelCoordinateComparisonContractReady,
    screenSpaceConventionReady,
    cameraContractMismatchDetected,
    cameraContractMismatchReason,
    visualParityComparisonAllowed,
    fixedReferenceCameraTolerance,
    referencePositionDelta,
    referenceUpDelta,
    referenceTargetDelta,
    cameraControlContract: deterministicState?.cameraControlContract ?? null,
    datasetViewMatrixMode: deterministicState?.datasetViewMatrixMode ?? null,
    fixedReferenceScreenSpaceCamera,
    runtimeScreenSpaceCamera,
    imageName: deterministicState?.imageName ?? referenceImageLabel,
    frameNumber: deterministicState?.frameNumber ?? null,
    viewId: deterministicState?.viewId ?? null,
    datasetTime: deterministicState?.datasetTime ?? null,
    outputWidth,
    outputHeight,
    viewport: viewportForComparison,
    backgroundPolicy: 'production-background-clear-pass-bgGray0',
    colorSpacePolicy: 'rgba8unorm-production-output-vs-reference-rgba',
    referenceImageLabel,
    referenceImagePath
  };
  const visualComparisonConditionsReady =
    webgpuProductionOutputCaptureReady &&
    referenceImageInputReady &&
    fixedReferenceCameraGateReady &&
    outputWidth > 0 &&
    outputHeight > 0;
  const cameraViewportBackgroundColorSpaceRecorded =
    visualComparisonConditionsReady &&
    visualComparisonConditions.viewport != null &&
    typeof visualComparisonConditions.backgroundPolicy === 'string' &&
    typeof visualComparisonConditions.colorSpacePolicy === 'string';
  const visualParityMetricMode =
    'production-output-capture-vs-fixed-reference-image-baseline';
  const visualParityMetricReady =
    visualComparisonConditionsReady &&
    cameraViewportBackgroundColorSpaceRecorded;
  const visualParityDifferenceSummary = {
    comparisonReady: visualParityMetricReady,
    metricMode: visualParityMetricMode,
    productionNonzeroOutputRatio: textureStats.nonzeroPixelRatio,
    productionNonzeroPixelCount: textureStats.nonzeroPixelCount,
    productionOutputWidth: outputWidth,
    productionOutputHeight: outputHeight,
    referenceImageLabel,
    referenceImagePath,
    fixedReferenceCameraGateReady,
    visualParityComparisonAllowed,
    cameraContractMismatchDetected,
    cameraContractMismatchReason,
    referencePositionDelta,
    referenceUpDelta,
    referenceTargetDelta,
    pixelDiffComputedInRuntime: false,
    pixelDiffTool: 'tools/compare_png.py',
    pixelDiffDeferredReason:
      'run compare_png.py after Step105 canvas PNG capture to compute MAE/RMSE/maxAbsError'
  };
  const visualMismatchClassification =
    cameraContractMismatchDetected
      ? 'camera-contract-mismatch'
      : visualParityComparisonAllowed && visualParityMetricReady
      ? 'visual-parity-baseline-ready-diff-tool-required'
      : 'visual-parity-baseline-input-incomplete';
  const viewMatrixEvidence = {
    ready:
      fixedReferenceScreenSpaceCamera?.enabled === true &&
      Array.isArray(fixedReferenceScreenSpaceCamera?.cudaAlignedViewMatrix),
    source: fixedReferenceScreenSpaceCamera?.viewMatrixSource ?? null,
    mode: fixedReferenceScreenSpaceCamera?.mode ?? null,
    hasCudaAlignedViewMatrix:
      Array.isArray(fixedReferenceScreenSpaceCamera?.cudaAlignedViewMatrix),
    hasThreeJsToCudaViewMatrix:
      Array.isArray(fixedReferenceScreenSpaceCamera?.threeJsToCudaViewMatrix)
  };
  const projectionMatrixEvidence = {
    ready: cameraProjectionContractReady,
    source: fixedReferenceScreenSpaceCamera?.projectionContract ?? null,
    intrinsics: fixedReferenceScreenSpaceCamera?.intrinsics ?? null,
    covarianceFocalContract:
      fixedReferenceScreenSpaceCamera?.covarianceFocalContract ?? null
  };
  const viewportEvidence = {
    ready: viewportComparisonContractReady,
    viewport: viewportForComparison,
    outputWidth,
    outputHeight
  };
  const screenSpaceConventionEvidence = {
    ready: screenSpaceConventionReady,
    pixelXSign: fixedReferenceScreenSpaceCamera?.pixelXSign ?? null,
    screenYSign: fixedReferenceScreenSpaceCamera?.screenYSign ?? null,
    depthSign: fixedReferenceScreenSpaceCamera?.depthSign ?? null,
    viewMatrixSource:
      fixedReferenceScreenSpaceCamera?.viewMatrixSource ?? null,
    pixelSignContract:
      fixedReferenceScreenSpaceCamera?.pixelSignContract ?? null
  };
  const backgroundPolicyEvidence = {
    ready: backgroundComparisonContractReady,
    backgroundPolicy: visualComparisonConditions.backgroundPolicy,
    bgGray: deterministicState?.bgGray ?? null,
    colorSpacePolicy: visualComparisonConditions.colorSpacePolicy
  };
  const cudaReferenceCameraEvidenceSources = {
    cameraMetadata: cameraMetadataName,
    viewMatrix: viewMatrixEvidence.source,
    projection: projectionMatrixEvidence.source,
    viewport: 'viewer-canvas-output-viewport',
    screenSpaceConvention:
      fixedReferenceScreenSpaceCamera?.viewMatrixSource ?? null,
    background: backgroundPolicyEvidence.backgroundPolicy,
    referenceImage: referenceImagePath
  };
  const missingCameraEvidenceReasons = [];
  if (!cudaReferenceInputReady) {
    missingCameraEvidenceReasons.push('cuda-reference-input-not-ready');
  }
  if (typeof cameraMetadataName !== 'string' || cameraMetadataName.length === 0) {
    missingCameraEvidenceReasons.push('camera-metadata-name-missing');
  }
  if (viewMatrixEvidence.ready !== true) {
    missingCameraEvidenceReasons.push('view-matrix-evidence-missing');
  }
  if (projectionMatrixEvidence.ready !== true) {
    missingCameraEvidenceReasons.push('projection-matrix-evidence-missing');
  }
  if (viewportEvidence.ready !== true) {
    missingCameraEvidenceReasons.push('viewport-evidence-missing');
  }
  if (screenSpaceConventionEvidence.ready !== true) {
    missingCameraEvidenceReasons.push(
      'screen-space-convention-evidence-missing'
    );
  }
  if (backgroundPolicyEvidence.ready !== true) {
    missingCameraEvidenceReasons.push('background-policy-evidence-missing');
  }
  const cudaReferenceCameraEvidenceReady =
    missingCameraEvidenceReasons.length === 0;
  const fixedReferenceCameraActivationReady =
    usesCudaAlignedFixedReferenceCamera === true;
  const fixedReferenceCameraActivationMode =
    fixedReferenceCameraActivationReady
      ? 'cuda-aligned-fixed-reference-camera'
      : referenceCameraMode;
  const fixedReferenceCameraActivationBlockedReason =
    fixedReferenceCameraActivationReady
      ? null
      : (
          cameraContractMismatchReason ??
          'fixed-reference-camera-activation-not-ready'
        );
  const webgpuCameraConstantsSource =
    fixedReferenceCameraActivationReady
      ? 'cuda-reference-camera-evidence-fixed-reference-mode'
      : 'fixed-reference-camera-contract-for-reference-comparison-or-viewer-camera-for-interactive-runtime';
  const viewMatrixSource = viewMatrixEvidence.source ?? null;
  const projectionMatrixSource = projectionMatrixEvidence.source ?? null;
  const viewportSource = cudaReferenceCameraEvidenceSources.viewport ?? null;
  const screenSpaceConventionSource =
    screenSpaceConventionEvidence.viewMatrixSource ?? null;
  const backgroundPolicySource =
    backgroundPolicyEvidence.backgroundPolicy ?? null;
  const cameraConstantsRoutingReady =
    fixedReferenceCameraActivationReady &&
    cudaReferenceCameraEvidenceReady &&
    webgpuCameraConstantsSource ===
      'cuda-reference-camera-evidence-fixed-reference-mode' &&
    typeof viewMatrixSource === 'string' &&
    typeof projectionMatrixSource === 'string' &&
    typeof viewportSource === 'string' &&
    typeof screenSpaceConventionSource === 'string' &&
    typeof backgroundPolicySource === 'string';
  const finalCompositorParityDiagnosticsReady =
    visualParityMetricReady &&
    finalCompositorVisualParityDiagnosticsReadiness != null;
  const finalCompositorParityDiagnosticsMode =
    'reference-image-diff-baseline-before-final-compositor-parity-claim';
  const step105NextStepRecommendedGoal =
    'run-reference-image-diff-and-classify-production-compositor-mismatch-v1';
  const step104VisualSafetyGatePreserved =
    workReductionReadinessReviewReady &&
    visualSafetyGateReady &&
    earlyTerminationDisabledBySafetyGate &&
    safeMinimalEarlyTerminationV1Used === false;
  const earlyTerminationRemainsDisabled =
    earlyTerminationEnabled === false &&
    safeMinimalEarlyTerminationV1Used === false &&
    earlyTerminationOnPathExecuted === false;
  const lodStreamingRemainsDisabled = true;

  for (const buffer of [...transientBuffers, orderedReferenceBuffer]) {
    if (typeof buffer.destroy === 'function') {
      buffer.destroy();
    }
  }
  if (
    !outputTextureCachedForHeartbeat &&
    boundedExecution.outputTextureReused !== true &&
    typeof outputTexture.destroy === 'function'
  ) {
    outputTexture.destroy();
  }

  return {
    compositorSummary,
    executionPlanObserver,
    tileReferenceCapacityContract,
    contract: buildWebGpuTileListCompositorContract({
      tileCompositorReady: ready,
      boundedExecutionContract,
      compositorPassSubmitted: true,
      compositorReadbackCompleted: true,
      compositorReadOffsetCountTable: summary.readOffsetCountTable,
      compositorTraversedReferenceList: referenceTraversalCompleted,
      outputTextureCreated: true,
      outputTextureWritten,
      outputTextureReadbackMatchesSummary,
      outputWidth,
      outputHeight,
      processedTileCount: summary.processedTileCount,
      compositedTileCount: summary.compositedTileCount,
      nonEmptyCompositedTileCount: summary.nonEmptyCompositedTileCount,
      compositedReferenceCount: summary.compositedReferenceCount,
      sourceTotalTileReferenceCount: summary.sourceTotalTileReferenceCount,
      overflowCount: summary.overflowCount,
      orderHandling: 'depth-aware-compositor-sort-key-descending',
      tileDepthOrderingReady: depthOrderingReady,
      depthOrderPassSubmitted: true,
      orderAwareCompositorUsed: summary.orderAwareCompositorUsed,
      depthKeyConsumed: summary.depthKeyConsumed,
      sortKeyConsumed: summary.sortKeyConsumed,
      compositorConsumedDepthOrderedReferences: depthOrderingReady,
      orderedReferenceCount: summary.orderedReferenceCount,
      orderedSourceReferenceCount: summary.sourceTotalTileReferenceCount,
      orderedReferenceCountMatchesSource,
      tileDepthOrderingContract,
      generatedCompositorFields: [
        'tile-list-offset-count-read',
        'splat-reference-list-traversal',
        'depth-aware-reference-selection',
        'gpu-side-ordered-reference-buffer-update',
        'bounded-global-bitonic-compare-stage-submissions-v1',
        'bounded-reference-chunk-production-accumulation-v1',
        'complete-per-tile-reference-and-padding-seed-across-bounded-submissions',
        'copy-free-reference-seed-guard',
        'parallel-sorted-buffer-readiness-guard',
        'production-tile-compositor-v1-main-path',
        'active-tile-subtile-accumulation-dispatch',
        'production-background-clear-pass',
        'inactive-background-handling-in-compositor',
        'compact-capacity-gpu-per-tile-depth-sort-v1',
        'no-silent-drop-tile-reference-capacity-policy',
        'device-limit-bounded-sort-resource-boundary',
        'tile-histogram-capacity-table-telemetry',
        'sort-order-violation-sampled-evidence',
        'depth-sorted-ordered-reference-buffer-consumed-by-accumulation',
        'production-accumulation-consumes-gpu-updated-ordered-refs',
        'sort-key-descending-compositor-order',
        'gaussian-footprint-weighted-alpha-accumulation',
        'gaussian-attribute-color-accumulation',
        'canvas-resolution-rgba8unorm-output-texture',
        'readback-free-steady-state-compositor-runtime-path',
        'gpu-owned-runtime-resource-flow',
        'production-readiness-telemetry',
        'time-driven-production-runtime-v1',
        'dirty-dependency-executor-v1',
        'clean-frame-last-valid-production-output-fast-path',
        'viewer-connected-interactive-time-scheduler-v1',
        'viewer-time-state-to-dirty-time-state-boundary',
        'raf-scheduler-production-runtime-invocation',
        'interactive-camera-state-to-dirty-camera-constants-boundary',
        'unified-production-interaction-scheduler-runtime-v1',
        'capture-probe-runtime-boundary-separation',
        'dirty-vs-clean-frame-budget-telemetry',
        'selective-dirty-dependency-execution-v1',
        'dirty-reason-stage-plan-classification',
        'production-stage-reuse-telemetry',
        'production-resource-lifecycle-cache-boundary-v1',
        'selective-executor-resource-reuse-policy',
        'realtime-bottleneck-classification-v1',
        'production-steady-state-no-readback-boundary-review-v1',
        'diagnostic-capture-readback-gate-v1',
        'active-tile-subtile-production-work-reduction-gate-v1',
        'step102-bottleneck-next-step-selection-update-v1',
        'compositor-work-reduction-readiness-review-v1',
        'early-termination-enable-disable-policy-v1',
        'visual-safety-quality-risk-gate-v1',
        'early-termination-on-off-comparison-readiness-v1',
        'step104-bottleneck-next-step-selection-v1',
        'webgpu-production-output-capture-baseline-v1',
        'cuda-reference-image-input-baseline-v1',
        'visual-parity-comparison-conditions-v1',
        'visual-mismatch-classification-baseline-v1',
        'step105-reference-visual-parity-next-step-selection-v1',
        'capability-based-regression-gate-v1',
        'legacy-step-preservation-diagnostic-mapping-v1',
        'step111-scale-aware-anisotropic-footprint-production-v1',
        'step111-cuda-webgpu-pipeline-parity-stage-map-v1',
        'step113-cuda-webgpu-covariance-jacobian-conic-production-v1'
      ],
      deferredCompositorFields: [
        'full-production-parallel-sort-parity',
        'cuda-compositor-parity',
        'final-production-tile-compositor',
        'chunk-lod-streaming',
        'complete-interactive-control-parity',
        'early-termination-v1-alpha-threshold-and-visual-parity-gate',
        'early-termination-on-path-output-delta-probe',
        'viewport-resize-resource-reallocation-probe',
        'visual-parity-diagnostics',
        'runtime-pixel-diff-against-reference-image',
        'cuda-reference-execution-refresh',
        'full-4d-conditional-covariance',
        'full-sh-color-parity',
        'early-termination-v1',
        'lod-streaming'
      ],
      compositorClassification:
        'production-webgpu-tile-compositor-v1-integration',
      fullDepthSortInWgsl: true,
      fullCudaParity: false,
      finalProductionTileCompositor: false,
      normalBackendFallbackMaintained: true,
      sourceGpuOwnedTileListLayoutContractVersion:
        sourceContract?.contractVersion ?? null,
      currentTexturePathMaintained: compositorOutputPresentedToCurrentTexture,
      tileCompositorOutputPresentedToCurrentTexture:
        compositorOutputPresentedToCurrentTexture,
      compositorCurrentTextureRenderPassSubmitted,
      compositorCurrentTextureReadbackCompleted,
      compositorCurrentTextureReadbackNonZero,
      presentationFrameCount,
      compositorPresentationFrameCount,
      currentTextureSource:
        compositorOutputPresentedToCurrentTexture
          ? 'webgpu-tile-compositor-output-texture'
          : null,
      currentTextureUsesWebGpuTileCompositorOutput,
      presentationStableUntilCapture,
      presentationSampleFrameCount: presentationFrameSamples.length,
      presentationNonBlankFrameCount: presentationFrameSamples.filter(
        (sample) => sample.nonzeroPixelCount > 0
      ).length,
      presentationBlankFrameCount: presentationFrameSamples.filter(
        (sample) => sample.nonzeroPixelCount <= 0
      ).length,
      presentationAllSampledFramesNonBlank:
        presentationFrameSamples.length > 0 &&
        presentationFrameSamples.every((sample) => sample.nonzeroPixelCount > 0),
      presentationAlternatingBlankDetected: presentationFrameSamples.some(
        (sample, index, samples) =>
          index > 0 &&
          (sample.nonzeroPixelCount > 0) !==
            (samples[index - 1].nonzeroPixelCount > 0)
      ),
      presentationStableVisualOutput:
        presentationFrameSamples.length > 0 &&
        presentationFrameSamples.every((sample) => sample.nonzeroPixelCount > 0),
      presentationNonzeroPixelRatioMin,
      presentationNonzeroPixelRatioMax,
      presentationFrameHashChanges,
      compositorOutputPresentedEverySampledFrame:
        presentationFrameSamples.length > 0 &&
        presentationFrameSamples.every(
          (sample) =>
            sample.currentTextureUsesWebGpuTileCompositorOutput === true
        ),
      canvasClearBetweenCompositorFramesDetected: false,
      currentTextureContextReconfigured,
      webgpuDeviceConsistencyReady,
      presentationDeviceMatchesCompositorDevice,
      currentTextureViewFreshPerPresentation,
      currentTextureViewReusedAcrossFrames,
      staleTextureViewReuseDetected,
      crossDeviceTextureViewUseDetected,
      contextReconfiguredOnDeviceChange,
      compositorOutputCacheInvalidatedOnDeviceChange,
      webgpuValidationErrorDetected,
      invalidCommandBufferDetected,
      queueSubmitFailureDetected,
      presentationErrorName,
      presentationErrorMessage,
      policyNeutralPresentationContract,
      lastValidOutputCacheDecision: outputTextureCacheDecision,
      presentationHeartbeatReady: compositorOutputPresentedToCurrentTexture,
      presentationDecoupledFromCompositorUpdate: true,
      lastValidCompositorOutputCached: outputTextureCachedForHeartbeat,
      compositorUpdateFrameCount: dirtyProductionRuntimeFrameCount,
      presentationHeartbeatFrameCount: presentationFrameCount,
      lastValidCompositorOutputPresentedOnCleanFrames: cleanFrameFastPathUsed,
      dirtySkippedCompositorUpdateButPresentedCachedOutput:
        cleanFrameFastPathUsed,
      presentationFrameSamples,
      realTileCompositorOutputReady,
      debugOutputBypassedForCompositor:
        summary.debugPatternBypassedForCompositor,
      gaussianAttributePayloadConsumed:
        summary.gaussianAttributePayloadConsumed,
      footprintPayloadConsumed: summary.footprintPayloadConsumed,
      orderedTileReferencesConsumed:
        summary.orderedTileReferencesConsumed,
      depthOrderedAccumulationUsed:
        summary.depthOrderedAccumulationUsed,
      alphaAccumulationUsed: summary.alphaAccumulationUsed,
      colorAccumulationUsed: summary.colorAccumulationUsed,
      tileCompositorContributionCount:
        summary.tileCompositorContributionCount,
      tileCompositorNonzeroOutputRatio:
        textureStats.nonzeroPixelRatio,
      tileCompositorOutputChangedFromDebugPattern:
        realTileCompositorOutputReady,
      step88PresentationContractPreserved,
      realTimeRuntimePathReady,
      readbackFreeSteadyStateCompositorUsed,
      runtimeCompositorDoesNotDependOnCaptureReadback,
      gpuOwnedRuntimeResourcesUsed,
      diagnosticReadbackSeparatedFromRuntimePath,
      diagnosticReadbackSeparatedFromProductionPath,
      productionTileCompositorReady,
      productionTileCompositorPathUsed,
      productionAccumulationConsumedParallelSortedRefs,
      activeTileDispatchReady,
      activeTileDispatchUsed,
      activeTileCount: summary.activeTileCount,
      inactiveTileCount: summary.inactiveTileCount,
      activeTilePixelWorkItemCount,
      fullScreenPixelWorkAvoided: summary.fullScreenPixelWorkAvoided,
      accumulationWorkReductionRatio: summary.accumulationWorkReductionRatio,
      inactiveBackgroundHandlingReady,
      inactivePixelOrTileWritePolicy:
        'clear-output-texture-then-write-active-tile-pixels',
      outputTextureProducedByProductionCompositor,
      lastValidOutputPreservedForCleanFrames:
        outputTextureCachedForHeartbeat &&
        readbackFreeSteadyStateCompositorUsed,
      readyBufferGuardUsed,
      invalidOrEmptyBufferRejected,
      debugOutputBypassedForProduction,
      fallbackOnlyCompositorUsed,
      normalBackendPresentationUsed: false,
      webgl2FallbackFinalPresentFrameCount: 0,
      debugPathSeparatedFromRuntimePath,
      runtimeOutputReadyWithoutTextureReadback,
      diagnosticTextureReadbackUsed,
      diagnosticSummaryReadbackUsed,
      compositorDispatchCount,
      compositorWorkItemCount,
      tileReferenceCount: summary.sourceTotalTileReferenceCount,
      accumulationContributionCount: summary.tileCompositorContributionCount,
      nonzeroOutputRatio: textureStats.nonzeroPixelRatio,
      runtimeTelemetryReady,
      cpuGpuSyncDependencyReduced,
      realtimeReadinessImproved,
      step89RealCompositorOutputPreserved,
      gpuSideTileOrderingReady,
      perTileOrderingRuntimePathUsed,
      orderedReferencesGeneratedOrUpdatedOnGpu,
      orderedReferencesConsumedByProductionAccumulation,
      productionAccumulationPathImproved,
      tileReferenceBufferLifecycleReady,
      sortOrOrderingDispatchCount,
      orderingWorkItemCount,
      orderingScratchBufferBytes: orderingSummaryData.byteLength,
      orderedReferenceBufferBytes,
      gpuOwnedOrderedReferenceRatio,
      step90RuntimePathPreserved,
      step91ProductionAccumulationMode:
        'gpu-updated-ordered-reference-buffer-depth-aware-alpha-color-accumulation',
      gpuSidePerTileSortReady,
      boundedPerTileSortUsed,
      depthKeyBufferConsumed: orderingSummary.depthKeyObserved,
      depthSortedOrderedReferencesGenerated,
      depthSortedReferencesConsumedByAccumulation,
      sortedAccumulationPathUsed,
      sortDispatchCount: sortOrOrderingDispatchCount,
      sortWorkItemCount: orderingWorkItemCount,
      sortedTileCount,
      sortedReferenceCount,
      unsortedFallbackTileCount,
      maxReferencesPerTile,
      avgReferencesPerTile,
      sortOrOrderingBufferBytes:
        orderedReferenceBufferBytes + orderingSummaryData.byteLength,
      step91OrderedReferenceRuntimePathPreserved: gpuSideTileOrderingReady,
      step92SortMode: 'compact-storage-bounded-global-bitonic-stage-descending-sort-key',
      gpuParallelPerTileSortReady,
      workgroupParallelSortUsed,
      parallelSortAlgorithm: 'bounded-global-bitonic-stage-v1-descending-sort-key',
      parallelSortStageCount,
      sortWorkgroupCount,
      sortOrderViolationCount,
      sortOrderSampleCheckReady,
      parallelSortFailureReason,
      parallelSortedBufferPromotedToAccumulation,
      parallelSortedBufferReady,
      parallelSortedBufferNonEmpty,
      referenceSeedCopyUsed,
      referenceSeedComputePassUsed,
      referenceSeedSourceHasCopySrc,
      referenceSeedDestinationHasCopyDst,
      copyBufferUsageValid,
      parallelSortOutputGuardUsed,
      preservedBoundedSortFallbackUsed,
      visualOutputDegeneratedDetected,
      step94ParallelSortPreserved: gpuParallelPerTileSortReady,
      timeDrivenProductionRuntimeReady,
      multiFrameProductionRuntimeUsed: productionRuntimeFrameCount > 1,
      runtimeFrameCount: productionRuntimeFrameCount,
      timeStateAdvancedAcrossFrames,
      frameStateAdvancedAcrossFrames,
      productionOutputUpdatedAcrossFrames,
      dirtyDependencyExecutorUsed: true,
      updatedStageNames,
      skippedStageNames,
      productionCompositorUpdatedOnDirtyFrames:
        productionOutputUpdatedAcrossFrames,
      cleanFrameFastPathUsed,
      lastValidProductionOutputReused: cleanFrameFastPathUsed,
      lastValidOutputPresentedOnCleanFrames: cleanFrameFastPathUsed,
      step96ProductionTileCompositorPreserved,
      viewerConnectedInteractiveSchedulerReady,
      viewerTimeStateConnectedToRuntime,
      playbackOrTimeSliderDrivesDirtyTimeState,
      rafSchedulerInvokesProductionRuntime,
      schedulerFrameCount,
      timeControlEvidenceReady,
      timeControlEvidenceSource,
      viewerTimeBefore,
      viewerTimeAfter,
      viewerTimeDelta,
      timeControlEvidenceFromSchedulerProbe,
      timeControlEvidenceUsesFixedValue,
      timeStateChangedByViewerControl,
      dirtyTimeStateTriggeredProductionUpdate,
      productionRuntimeUpdatedFromViewerScheduler,
      cleanFrameReuseUnderScheduler,
      lastValidProductionOutputPresentedByScheduler,
      step97MultiFrameRuntimePreserved,
      phase2CameraContractAssumptionsAdopted,
      phase3ResponsibilityPlanReferenced,
      phase3BackendDesignReferenced,
      fixedReferenceAndInteractiveCameraSeparated,
      threeJsCameraAdapterOnly,
      cudaReferenceNotInteractiveBackend,
      viewerCameraStateConnectedToRuntime,
      viewerCameraStateChangedByProbe,
      cameraConstantsChanged,
      dirtyCameraConstantsTriggeredProductionUpdate,
      viewportStateConnectedToRuntime,
      dirtyViewportTriggeredProductionUpdate,
      dirtyViewportDeferredReason,
      productionRuntimeUpdatedFromViewerCameraScheduler,
      cleanFrameReuseAfterCameraStabilized,
      lastValidProductionOutputPresentedAfterCameraCleanFrame,
      cameraControlEvidenceReady,
      cameraControlEvidenceSource,
      cameraControlEvidenceFromSchedulerProbe,
      cameraControlEvidenceUsesFixedValue,
      cameraPositionBefore: cameraControlEvidence?.beforeCamera?.position ?? null,
      cameraPositionAfter: cameraControlEvidence?.afterCamera?.position ?? null,
      cameraQuaternionBefore:
        cameraControlEvidence?.beforeCamera?.quaternion ?? null,
      cameraQuaternionAfter:
        cameraControlEvidence?.afterCamera?.quaternion ?? null,
      cameraConstantsMaxAbsDelta,
      viewportBefore: cameraControlEvidence?.beforeViewport ?? null,
      viewportAfter: cameraControlEvidence?.afterViewport ?? null,
      viewportChangedByProbe,
      step98ViewerTimeSchedulerPreserved,
      unifiedInteractionSchedulerReady,
      timeAndCameraDirtyPathsUnified,
      captureProbeRuntimeBoundarySeparated,
      captureProbeStimulatesViewerStateOnly,
      dirtyDependencyGraphConsumedByProductionRuntime,
      dirtyTimeStateTriggersUnifiedProductionUpdate,
      dirtyCameraConstantsTriggersUnifiedProductionUpdate,
      dirtyViewportIntegratedOrDeferredReason,
      productionRuntimeUpdatedByUnifiedInteractionScheduler,
      cleanFrameReuseAfterUnifiedInteractionStabilized,
      lastValidProductionOutputPresentedAfterUnifiedCleanFrame,
      realtimeFrameBudgetTelemetryReady,
      realtimeWorkloadBudgetTelemetryReady,
      dirtyFrameCount,
      cleanFrameReuseCount,
      productionUpdateCount,
      step99InteractiveCameraDirtyPreserved,
      selectiveDirtyDependencyExecutionReady,
      dirtyReasonClassificationReady,
      dirtyReasonSequence,
      timeDirtyStagePlanReady,
      cameraDirtyStagePlanReady,
      viewportDirtyIntegratedOrDeferredReason,
      cleanFrameStagePlanReady,
      selectiveStageInvalidationUsed,
      updatedStageNamesByDirtyReason,
      skippedStageNamesByDirtyReason,
      reusedResourceNamesByDirtyReason,
      productionRuntimeUpdatedBySelectiveExecutor,
      cleanFrameReuseAfterSelectiveExecution,
      lastValidProductionOutputPresentedAfterSelectiveCleanFrame,
      skippedStageCount,
      reusedResourceCount,
      step100UnifiedInteractionSchedulerPreserved,
      productionResourceLifecycleReady,
      persistentGpuResourceCacheReady,
      persistentResourceNames,
      transientResourceNames,
      remainingTransientResourceNames,
      resourceReallocationBoundaryReady,
      resourceReallocationPolicy:
        'reallocate-only-on-device-or-size-capacity-boundary',
      resourceReallocationReasons,
      outputTextureReallocationBoundaryReady,
      viewportResizeResourceReallocationDeferredReason,
      selectiveExecutorConnectedToResourceReuse,
      dirtyReasonResourceReusePolicyReady,
      resourceReusePolicyByDirtyReason,
      persistentResourceReuseByDirtyReason,
      resourceAllocationCount,
      resourceReuseCount,
      resourceReallocationCount,
      transientResourceCount,
      persistentResourceCount,
      cleanFrameUsesPersistentOutput,
      dirtyFrameReusesPersistentResources,
      realtimeBottleneckEvidenceReady,
      bottleneckClassification,
      bottleneckStageName,
      bottleneckEvidence,
      nextStepRecommendedGoal,
      earlyTerminationDeferredReason,
      chunkLodStreamingReadiness,
      visualParityDiagnosticsDeferredReason,
      step101SelectiveDirtyDependencyPreserved,
      productionRuntimeBoundaryReviewReady,
      productionSteadyStateNoReadbackBoundaryReady,
      readbackLeakIntoProductionRuntimeDetected,
      diagnosticCaptureReadbackGateReady,
      diagnosticReadbackResourcesIsolated,
      runtimeBoundaryHardened,
      selectedWorkReductionPath,
      workReductionPathImplemented,
      activeTileWorkReductionReady,
      earlyTerminationCandidateReady,
      earlyTerminationV1Used,
      earlyTerminationEvaluatedReferenceCount,
      earlyTerminationSkippedReferenceCount,
      earlyTerminationSkipRatio,
      viewportResizeResourceReallocationIntegrated,
      visualParityDiagnosticsReadiness,
      updatedBottleneckClassification,
      updatedNextStepRecommendedGoal,
      step102ProductionResourceLifecyclePreserved,
      workReductionReadinessReviewReady,
      earlyTerminationPolicyReady,
      earlyTerminationEnablePolicy,
      earlyTerminationEnabled,
      earlyTerminationDisabledBySafetyGate,
      earlyTerminationDisableReason,
      visualSafetyGateReady,
      visualQualityRiskGateReady,
      visualQualityRiskLevel,
      visualSafetyEvidence,
      earlyTerminationOnOffComparisonReady,
      earlyTerminationOnOffComparisonMode,
      earlyTerminationOffPathReferenceReady,
      earlyTerminationOnPathExecuted,
      earlyTerminationOnPathDisabledBySafetyGate,
      earlyTerminationOnOffOutputDeltaReady,
      safeThresholdPolicy,
      safeThresholdPolicyReady,
      safeThresholdValue,
      safeMinimalEarlyTerminationV1Used,
      safeMinimalEarlyTerminationDeferredReason,
      finalCompositorVisualParityDiagnosticsReadiness,
      step104BottleneckClassification,
      step104NextStepRecommendedGoal,
      step103ProductionRuntimeBoundaryReviewPreserved,
      productionOutputCaptureReady: webgpuProductionOutputCaptureReady,
      webgpuProductionOutputCaptureReady,
      referenceImageInputReady,
      cudaReferenceInputReady,
      referenceImageSource,
      referenceImagePath,
      referenceImageLabel,
      fixedReferenceCameraGateReady,
      referenceCameraMode,
      usesCudaAlignedFixedReferenceCamera,
      interactiveCameraExcludedFromReferenceComparison,
      cameraMetadataName,
      cameraProjectionContractReady,
      viewportComparisonContractReady,
      backgroundComparisonContractReady,
      pixelCoordinateComparisonContractReady,
      screenSpaceConventionReady,
      cameraContractMismatchDetected,
      cameraContractMismatchReason,
      visualParityComparisonAllowed,
      visualComparisonConditionsReady,
      visualComparisonConditions,
      cameraViewportBackgroundColorSpaceRecorded,
      visualParityMetricReady,
      visualParityMetricMode,
      visualParityDifferenceSummary,
      visualMismatchClassification,
      finalCompositorParityDiagnosticsReady,
      finalCompositorParityDiagnosticsMode,
      step105NextStepRecommendedGoal,
      cudaReferenceCameraEvidenceReady,
      cudaReferenceCameraEvidenceSources,
      fixedReferenceCameraActivationReady,
      fixedReferenceCameraActivationMode,
      fixedReferenceCameraActivationBlockedReason,
      viewMatrixEvidence,
      projectionMatrixEvidence,
      viewportEvidence,
      screenSpaceConventionEvidence,
      backgroundPolicyEvidence,
      webgpuCameraConstantsSource,
      cameraConstantsRoutingReady,
      viewMatrixSource,
      projectionMatrixSource,
      viewportSource,
      screenSpaceConventionSource,
      backgroundPolicySource,
      interactiveCameraSeparatedFromFixedReference:
        fixedReferenceAndInteractiveCameraSeparated,
      missingCameraEvidenceDetected:
        missingCameraEvidenceReasons.length > 0,
      missingCameraEvidenceReasons,
      step107DesignGatePreserved: fixedReferenceCameraGateReady,
      step108CameraEvidencePreserved:
        cudaReferenceCameraEvidenceReady &&
        missingCameraEvidenceReasons.length === 0,
      step109NextStepRecommendedGoal:
        cameraConstantsRoutingReady && visualParityComparisonAllowed
          ? 'run-fixed-reference-camera-visual-parity-diff-v1'
          : 'resolve-fixed-reference-camera-routing-or-comparison-blocker-v1',
      cudaWebgpuPipelineParityStageMap,
      step111SelectedGoal:
        'scale-aware-gaussian-footprint-conic-production-path-v1',
      step111SelectedGap:
        'covariance-conic-screen-space-footprint',
      step111GapBeforeClassification:
        'approximation-isotropic-radius-derived-conic',
      step111GapAfterClassification:
        step111ProductionRuntimeGapClosureUsed
          ? 'partial-scale-aware-anisotropic-conic-production'
          : 'blocked-scale-aware-conic-not-consumed',
      step111ProductionRuntimeGapClosureUsed,
      step111ProductionConsumptionEvidence,
      step111ApproximationReplaced:
        'isotropic-radius-only-conic-for-production-gaussian-weight',
      step111RemainingApproximations: [
        'rotation-ignored-in-screen-space-conic',
        'camera-jacobian-covariance-projection-deferred',
        'full-4d-covariance-deferred',
        'full-sh-color-deferred',
        'final-cuda-compositor-parity-deferred'
      ],
      step111DeferredStructuralGaps: [
        'rotation-aware-anisotropic-conic-parity',
        'camera-jacobian-screen-space-conic-parity',
        'full-sh-color-opacity-parity',
        'cuda-front-to-back-compositor-parity',
        'chunk-lod-streaming'
      ],
      scaleAwareConicPayloadConsumed,
      anisotropicFootprintReferenceCount:
        summary.anisotropicFootprintReferenceCount,
      anisotropicFootprintRatio: summary.anisotropicFootprintRatio,
      conicFallbackReferenceCount: summary.conicFallbackReferenceCount,
      presentationCaptureOrientationEvidence:
        buildWebGpuPresentationCaptureOrientationEvidence(),
      captureFreshnessEvidence: {
        productionSourceRequestIdentity:
          outputTextureCachedForHeartbeat && viewerCanvas
            ? viewerCanvasTileCompositorOutputState.get(viewerCanvas)
                ?.sourceRequestIdentity ?? null
            : null,
        productionOutputGeneration:
          outputTextureCachedForHeartbeat && viewerCanvas
            ? viewerCanvasTileCompositorOutputState.get(viewerCanvas)?.generation ?? null
            : null,
        presentedOutputGeneration:
          outputTextureCachedForHeartbeat && viewerCanvas
            ? viewerCanvasTileCompositorOutputState.get(viewerCanvas)
                ?.presentedFrameIdentity?.generation ?? null
            : null,
        productionFrameIdentity:
          outputTextureCachedForHeartbeat && viewerCanvas
            ? viewerCanvasTileCompositorOutputState.get(viewerCanvas)?.frameIdentity ?? null
            : null,
        presentedFrameIdentity:
          outputTextureCachedForHeartbeat && viewerCanvas
            ? viewerCanvasTileCompositorOutputState.get(viewerCanvas)
                ?.presentedFrameIdentity ?? null
            : null,
        productionOutputCreatedAtMs:
          outputTextureCachedForHeartbeat && viewerCanvas
            ? viewerCanvasTileCompositorOutputState.get(viewerCanvas)?.cachedAtMs ?? null
            : null,
        presentationTimestampMs:
          outputTextureCachedForHeartbeat && viewerCanvas
            ? viewerCanvasTileCompositorOutputState.get(viewerCanvas)?.lastPresentedAtMs ?? null
            : null,
        captureFreshnessKnown: null,
        captureFreshnessClassification:
          'capture-freshness-validated-by-png-capture-status-after-save'
      },
      step111NextStepRecommendedGoal:
        'rotation-and-camera-jacobian-screen-space-conic-parity-v1',
      step112SelectedGoal:
        step112ProjectionParityEvidence.selectedGoal ??
        'cuda-fixed-reference-camera-projection-screen-space-orientation-parity-closure-v1',
      step112SelectedCandidates:
        step112ProjectionParityEvidence.selectedCandidates ?? [],
      step112RootCause:
        step112ProjectionParityEvidence.rootCause ?? null,
      step112CoordinateTransformStageMap:
        step112ProjectionParityEvidence.coordinateTransformStageMap ?? [],
      step112CanonicalCameraProjectionSources:
        step112ProjectionParityEvidence.canonicalCameraProjectionSources ?? {},
      step112CoordinateConvention:
        step112ProjectionParityEvidence.coordinateConvention ?? {},
      step112RepresentativePointComparison:
        step112ProjectionParityEvidence.representativePointComparison ?? {},
      step112FirstMismatchStage:
        step112ProjectionParityEvidence.firstMismatchStage ?? null,
      step112ViewProjectionPixelParityReady:
        step112ProjectionParityEvidence.viewProjectionPixelParityReady === true,
      step112CenterProjectionParityReady:
        step112ProjectionParityEvidence.centerProjectionParityReady === true,
      step112CenterProjectionConicConventionConsistent:
        step112ProjectionParityEvidence.centerProjectionParityReady === true &&
        scaleAwareConicPayloadConsumed === true,
      step112ProductionRuntimeConsumptionReady:
        step112ProjectionParityEvidence.centerProjectionParityReady === true &&
        step111ProductionRuntimeGapClosureUsed === true &&
        outputTextureProducedByProductionCompositor === true,
      step112RemainingCameraProjectionGaps:
        step112ProjectionParityEvidence.remainingCameraProjectionGaps ?? [],
      step112NextStepRecommendedGoal:
        step112ProjectionParityEvidence.viewProjectionPixelParityReady === true
          ? 'camera-jacobian-screen-space-conic-parity-v1'
          : 'resolve-fixed-reference-projection-representative-point-mismatch-v1',
      step113SelectedGoal:
        'cuda-webgpu-covariance-jacobian-conic-parity-closure-v1',
      step113SelectedCandidates:
        step113CovarianceJacobianConicEvidence.selectedCandidates ?? [],
      step113RootCause:
        step113CovarianceJacobianConicEvidence.rootCause ?? null,
      step113CudaWebgpuCovarianceConicStageMap:
        step113CovarianceJacobianConicEvidence
          .cudaWebgpuCovarianceConicStageMap ?? [],
      step113RepresentativeGaussianComparison:
        step113CovarianceJacobianConicEvidence,
      step113RepresentativeGaussianCount:
        step113CovarianceJacobianConicEvidence.representativeGaussianCount ?? 0,
      step113FirstMismatchStage:
        step113CovarianceJacobianConicEvidence.firstMismatchStage ?? null,
      step113MaxStageErrors:
        step113CovarianceJacobianConicEvidence.maxStageErrors ?? {},
      step113RotationCovarianceClassification:
        step113CovarianceJacobianConicEvidence
          .rotationCovarianceClassification ?? null,
      step113Conditional4DCovarianceClassification:
        step113CovarianceJacobianConicEvidence
          .conditional4DCovarianceClassification ?? null,
      step113JacobianProjectionClassification:
        step113CovarianceJacobianConicEvidence
          .jacobianProjectionClassification ?? null,
      step113ConicRadiusClassification:
        step113CovarianceJacobianConicEvidence
          .conicRadiusClassification ?? null,
      step113ProductionRuntimeGapClosureUsed,
      step113ProductionConsumptionEvidence,
      step113OldApproximationUsedInProduction:
        step113ProductionRuntimeGapClosureUsed !== true,
      step113RemainingStructuralGaps: [
        'full-4d-conditional-covariance',
        'full-sh-color-opacity-parity',
        'cuda-front-to-back-compositor-parity',
        'chunk-lod-streaming'
      ],
      step113NextStepRecommendedGoal:
        step113ProductionRuntimeGapClosureUsed
          ? 'sh-color-opacity-temporal-weighting-production-parity-v1'
          : 'resolve-covariance-jacobian-conic-representative-mismatch-v1',
      step108NextStepRecommendedGoal:
        fixedReferenceCameraActivationReady
          ? 'run-fixed-reference-camera-visual-parity-comparison-v1'
          : 'activate-fixed-reference-camera-mode-from-cuda-evidence-v1',
      step104VisualSafetyGatePreserved,
      earlyTerminationRemainsDisabled,
      lodStreamingRemainsDisabled,
      step93OverflowPolicyPreserved,
      overflowAwareOrderingReady,
      sortCapacityLimit,
      overflowTileCount,
      overflowReferenceCount,
      droppedReferenceCount,
      overflowHandlingPolicy:
        'complete-reference-sort-or-frame-fail-closed-with-zero-silent-drop',
      sortedReferenceCountMatchesSourceOrCapacityPolicy,
      capacityUtilizationMax: orderingSummary.capacityUtilizationMax,
      capacityUtilizationAvg,
      scalableSortPreparationReady,
      sortScratchBufferReady,
      tileHistogramOrCapacityTableReady,
      productionOrderedReferenceLifecycleReady,
      sortedAccumulationCapacityPolicyUsed,
      deferredProductionItems: [
        'full-cuda-parity',
        'final-production-compositor',
        'full-parallel-sort-parity',
        'complete-interactive-control-parity',
        'camera-visual-parity',
        'final-production-compositor-parity',
        'early-termination-v1-alpha-threshold-and-visual-parity-gate',
        'early-termination-on-path-output-delta-probe',
        'viewport-resize-dirty-probe',
        'viewport-resize-resource-reallocation-probe',
        'visual-parity-diagnostics',
        'chunk-lod-streaming',
        'runtime-pixel-diff-against-reference-image',
        'cuda-reference-execution-refresh'
      ],
      reason: ready
        ? null
        : 'webgpu-tile-list-compositor-did-not-consume-gpu-owned-tile-list'
    })
  };
}
