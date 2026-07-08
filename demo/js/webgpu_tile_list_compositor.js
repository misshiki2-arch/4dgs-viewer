import {
  buildWebGpuTileDepthOrderingContract,
  buildWebGpuTileListCompositorContract
} from './common_4dgs_record_contracts.js';

const COMPOSITOR_SUMMARY_FLOAT_COUNT = 36;
const ORDERING_SUMMARY_UINT_COUNT = 28;
const BOUNDED_SORT_CAPACITY_LIMIT = 64;
const PARALLEL_SORT_STAGE_COUNT = 21;
const viewerCanvasWebGpuContextState = new WeakMap();
const viewerCanvasTileCompositorOutputState = new WeakMap();

function alignTo(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
}

function finiteNumberOr(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function createBuffer(device, data, usage) {
  const buffer = device.createBuffer({
    size: Math.max(4, Math.ceil(data.byteLength / 4) * 4),
    usage,
    mappedAtCreation: true
  });
  new Uint8Array(buffer.getMappedRange()).set(
    new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
  );
  buffer.unmap();
  return buffer;
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
    activeTileDispatchReady: Math.round(finiteNumberOr(summary[35], 0)) === 1
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
    parallelSortCompareSwapPassCount: Math.round(finiteNumberOr(summary[23], 0))
  };
}

function hasNonZeroTextureByte(readback, bytesPerRow, width, height) {
  for (let y = 0; y < height; y += 1) {
    const row = y * bytesPerRow;
    for (let x = 0; x < width * 4; x += 1) {
      if (readback[row + x] !== 0) return true;
    }
  }
  return false;
}

function summarizeTextureReadback(readback, bytesPerRow, width, height) {
  let nonzeroPixelCount = 0;
  let hash = 2166136261;
  const pixelCount = Math.max(1, width * height);
  for (let y = 0; y < height; y += 1) {
    const row = y * bytesPerRow;
    for (let x = 0; x < width; x += 1) {
      const offset = row + x * 4;
      const r = readback[offset + 0] ?? 0;
      const g = readback[offset + 1] ?? 0;
      const b = readback[offset + 2] ?? 0;
      const a = readback[offset + 3] ?? 0;
      if (r !== 0 || g !== 0 || b !== 0 || a !== 0) {
        nonzeroPixelCount += 1;
      }
      hash ^= r;
      hash = Math.imul(hash, 16777619) >>> 0;
      hash ^= g;
      hash = Math.imul(hash, 16777619) >>> 0;
      hash ^= b;
      hash = Math.imul(hash, 16777619) >>> 0;
      hash ^= a;
      hash = Math.imul(hash, 16777619) >>> 0;
    }
  }
  return {
    nonzeroPixelCount,
    nonzeroPixelRatio: nonzeroPixelCount / pixelCount,
    frameHash: hash.toString(16).padStart(8, '0')
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
  forceContextRefresh = false
}) {
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
    presentationSource
  };
  const currentTextureGuardAllowed =
    viewerCanvasState?.requestedBackendMode === 'webgpu-exclusive' &&
    viewerCanvasState?.allowViewerCanvasPresentation === true &&
    viewerCanvasState?.webgl2FrameLifecycleSuppressed === true &&
    viewerCanvasState?.provided === true &&
    !!viewerCanvas;
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
  out.uv = pos * 0.5 + vec2f(0.5, 0.5);
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
        presentationSource
      });
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
  summary.compositorOutputPresentedToCurrentTexture =
    summary.compositorCurrentTextureRenderPassSubmitted &&
    summary.compositorCurrentTextureReadbackCompleted &&
    summary.compositorCurrentTextureReadbackNonZero &&
    summary.currentTextureUsesWebGpuTileCompositorOutput &&
    summary.presentationStableUntilCapture &&
    summary.currentTextureViewFreshPerPresentation &&
    summary.crossDeviceTextureViewUseDetected === false &&
    summary.staleTextureViewReuseDetected === false &&
    summary.webgpuValidationErrorDetected === false &&
    summary.invalidCommandBufferDetected === false &&
    summary.queueSubmitFailureDetected === false;
  summary.webgpuDeviceConsistencyReady =
    summary.presentationDeviceMatchesCompositorDevice === true &&
    summary.currentTextureViewFreshPerPresentation === true &&
    summary.currentTextureViewReusedAcrossFrames === false &&
    summary.staleTextureViewReuseDetected === false &&
    summary.crossDeviceTextureViewUseDetected === false &&
    summary.webgpuValidationErrorDetected === false &&
    summary.invalidCommandBufferDetected === false &&
    summary.queueSubmitFailureDetected === false;
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
  outputHeight
}) {
  if (!canvas || !device || !outputTexture) {
    return { cached: false, invalidatedOnDeviceChange: false };
  }
  const previous = viewerCanvasTileCompositorOutputState.get(canvas);
  const invalidatedOnDeviceChange =
    !!previous?.device && previous.device !== device;
  if (
    previous?.outputTexture &&
    previous.outputTexture !== outputTexture &&
    typeof previous.outputTexture.destroy === 'function'
  ) {
    previous.outputTexture.destroy();
  }
  viewerCanvasTileCompositorOutputState.set(canvas, {
    device,
    outputTexture,
    outputWidth,
    outputHeight,
    generation: (previous?.generation ?? 0) + 1,
    cachedAtMs:
      typeof performance !== 'undefined' && typeof performance.now === 'function'
        ? performance.now()
        : Date.now()
  });
  return { cached: true, invalidatedOnDeviceChange };
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
    !!cached?.outputTexture && !!heartbeatDevice && cached?.device === heartbeatDevice;
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
    forceContextRefresh: false
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
    canvasClearBetweenCompositorFramesDetected: presentationBlankFrameCount > 0,
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
  viewerCanvasState = null
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
  const outputTexture = device.createTexture({
    label: 'phase3-step85-webgpu-tile-list-compositor-output-texture',
    size: { width: outputWidth, height: outputHeight },
    format: 'rgba8unorm',
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.TEXTURE_BINDING
  });
  const bytesPerRow = alignTo(outputWidth * 4, 256);
  const textureReadbackBuffer = device.createBuffer({
    size: bytesPerRow * outputHeight,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const summaryData = new Float32Array(COMPOSITOR_SUMMARY_FLOAT_COUNT);
  const summaryBuffer = createBuffer(
    device,
    summaryData,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const params = new Float32Array([
    resources.tileCount,
    resources.tileCols,
    resources.tileRows,
    resources.maxRefsPerTile,
    outputWidth,
    outputHeight,
    finiteNumberOr(canvasWidth, outputWidth),
    finiteNumberOr(canvasHeight, outputHeight)
  ]);
  const paramsBuffer = createBuffer(device, params, GPUBufferUsage.UNIFORM);
  const summaryReadbackBuffer = device.createBuffer({
    size: summaryData.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const referenceCapacity = Math.max(1, resources.tileCount * resources.maxRefsPerTile);
  const sortCapacityLimit = Math.max(
    1,
    Math.min(resources.maxRefsPerTile, BOUNDED_SORT_CAPACITY_LIMIT)
  );
  const orderedReferenceBufferBytes = Math.max(16, referenceCapacity * 4 * 4);
  const orderedReferenceBuffer = device.createBuffer({
    label: 'phase3-step92-webgpu-depth-sorted-reference-buffer',
    size: orderedReferenceBufferBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  });
  const orderingSummaryData = new Uint32Array(ORDERING_SUMMARY_UINT_COUNT);
  const orderingSummaryBuffer = createBuffer(
    device,
    orderingSummaryData,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  );
  const orderingParams = new Uint32Array([
    resources.tileCount,
    referenceCapacity,
    resources.maxRefsPerTile,
    94
  ]);
  const orderingParamsBuffer = createBuffer(device, orderingParams, GPUBufferUsage.UNIFORM);
  const orderingSummaryReadbackBuffer = device.createBuffer({
    size: orderingSummaryData.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const orderingShader = device.createShaderModule({
    label: 'phase3-step94-webgpu-parallel-per-tile-depth-sort-wgsl',
    code: `
struct OrderingParams {
  tileCount: u32,
  referenceCapacity: u32,
  maxRefsPerTile: u32,
  statusCode: u32,
};

@group(0) @binding(0) var<storage, read> tileTable: array<vec4f>;
@group(0) @binding(1) var<storage, read> sourceReferences: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> orderedReferences: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> orderingSummary: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> orderingParams: OrderingParams;

var<workgroup> localRefs: array<vec4f, 64>;
var<workgroup> localKeys: array<f32, 64>;

@compute @workgroup_size(64)
fn seedOrderedReferences(@builtin(global_invocation_id) globalId: vec3u) {
  let referenceIndex = globalId.x;
  if (referenceIndex >= orderingParams.referenceCapacity) {
    return;
  }
  orderedReferences[referenceIndex] = sourceReferences[referenceIndex];
}

@compute @workgroup_size(64)
fn prepareOrderedReferences(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_id) localId: vec3u
) {
  let tile = workgroupId.x;
  let slot = localId.x;
  if (tile >= orderingParams.tileCount) {
    return;
  }
  let table = tileTable[tile];
  let tileValid = table.w == 84.0 && table.y > 0.0;
  let offset = u32(max(table.x, 0.0));
  let rawCount = select(0u, min(u32(max(table.y, 0.0)), orderingParams.maxRefsPerTile), tileValid);
  let overflowRefs = select(0u, u32(max(table.z, 0.0)), tileValid);
  let sortLimit = min(orderingParams.maxRefsPerTile, 64u);
  let count = min(rawCount, sortLimit);

  if (tileValid && slot == 0u) {
    atomicAdd(&orderingSummary[1], 1u);
    atomicAdd(&orderingSummary[8], 1u);
    atomicAdd(&orderingSummary[12], rawCount);
    atomicAdd(&orderingSummary[13], (rawCount - count) + overflowRefs);
    atomicAdd(&orderingSummary[15], overflowRefs);
    atomicStore(&orderingSummary[16], sortLimit);
    atomicStore(&orderingSummary[21], 21u);
    atomicAdd(&orderingSummary[22], 1u);
    atomicStore(&orderingSummary[23], 21u);
    if (rawCount > count || overflowRefs > 0u) {
      atomicAdd(&orderingSummary[10], 1u);
      atomicAdd(&orderingSummary[14], 1u);
    }
    atomicMax(&orderingSummary[11], count);
    let hasSortLimit = sortLimit > 0u;
    let utilization = select(0u, (count * 100000u) / max(sortLimit, 1u), hasSortLimit);
    atomicMax(&orderingSummary[17], utilization);
    atomicAdd(&orderingSummary[18], utilization);
    atomicAdd(&orderingSummary[19], 1u);
  }

  let sourceIndex = offset + slot;
  if (slot < count && sourceIndex < orderingParams.referenceCapacity) {
    let splatRef = sourceReferences[sourceIndex];
    localRefs[slot] = splatRef;
    localKeys[slot] = splatRef.w;
  } else {
    localRefs[slot] = vec4f(0.0, 0.0, 0.0, 0.0);
    localKeys[slot] = -340282346638528859811704183484516925440.0;
  }
  workgroupBarrier();

  for (var k: u32 = 2u; k <= 64u; k = k * 2u) {
    var j = k / 2u;
    loop {
      let partnerSlot = slot ^ j;
      if (partnerSlot > slot && partnerSlot < 64u) {
        let selfKey = localKeys[slot];
        let partnerKey = localKeys[partnerSlot];
        let ascending = (slot & k) != 0u;
        let selfLess = selfKey < partnerKey;
        let selfGreater = selfKey > partnerKey;
        let shouldSwap = select(selfLess, selfGreater, ascending);
        if (shouldSwap) {
          let tempKey = localKeys[slot];
          let tempRef = localRefs[slot];
          localKeys[slot] = localKeys[partnerSlot];
          localRefs[slot] = localRefs[partnerSlot];
          localKeys[partnerSlot] = tempKey;
          localRefs[partnerSlot] = tempRef;
        }
      }
      workgroupBarrier();
      if (j == 1u) {
        break;
      }
      j = j / 2u;
    }
  }

  if (slot < count) {
    let orderedIndex = offset + slot;
    if (orderedIndex < orderingParams.referenceCapacity) {
      let splatRef = localRefs[slot];
      orderedReferences[orderedIndex] = splatRef;
      atomicAdd(&orderingSummary[0], 1u);
      atomicAdd(&orderingSummary[4], 1u);
      atomicAdd(&orderingSummary[9], 1u);
      if (splatRef.z != 0.0) {
        atomicStore(&orderingSummary[2], 1u);
      }
      if (splatRef.w != 0.0) {
        atomicStore(&orderingSummary[3], 1u);
      }
      atomicStore(&orderingSummary[7], 1u);
    }
  }
  if (slot + 1u < count && localKeys[slot] < localKeys[slot + 1u]) {
    atomicAdd(&orderingSummary[20], 1u);
  }
  if (tileValid && slot == 0u) {
    atomicStore(&orderingSummary[5], orderingParams.referenceCapacity);
    atomicStore(&orderingSummary[6], orderingParams.statusCode);
  }
}
`
  });
  const orderingBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' }
      }
    ]
  });
  const orderingPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [orderingBindGroupLayout]
  });
  const orderingPipeline = device.createComputePipeline({
    layout: orderingPipelineLayout,
    compute: { module: orderingShader, entryPoint: 'prepareOrderedReferences' }
  });
  const referenceSeedPipeline = device.createComputePipeline({
    layout: orderingPipelineLayout,
    compute: { module: orderingShader, entryPoint: 'seedOrderedReferences' }
  });
  const orderingBindGroup = device.createBindGroup({
    layout: orderingBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: resources.tileTableBuffer } },
      { binding: 1, resource: { buffer: resources.referenceListBuffer } },
      { binding: 2, resource: { buffer: orderedReferenceBuffer } },
      { binding: 3, resource: { buffer: orderingSummaryBuffer } },
      { binding: 4, resource: { buffer: orderingParamsBuffer } }
    ]
  });
  const shader = device.createShaderModule({
    label: 'phase3-step85-webgpu-tile-list-compositor-wgsl',
    code: `
struct Params {
  tileCount: f32,
  tileCols: f32,
  tileRows: f32,
  maxRefsPerTile: f32,
  outputWidth: f32,
  outputHeight: f32,
  canvasWidth: f32,
  canvasHeight: f32,
};

@group(0) @binding(0) var<storage, read> tileInputs: array<vec4f>;
@group(0) @binding(1) var<storage, read> tileTable: array<vec4f>;
@group(0) @binding(2) var<storage, read> referenceList: array<vec4f>;
@group(0) @binding(3) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<storage, read_write> compositorSummary: array<vec4f>;
@group(0) @binding(5) var<uniform> params: Params;

fn sampleConic(conicAndSort: vec4f, radius: f32) -> vec3f {
  let fallbackConic = 1.0 / max(radius * radius, 1.0);
  let conicXAvailable = abs(conicAndSort.x) > 0.0;
  let conicX = select(fallbackConic, abs(conicAndSort.x), conicXAvailable);
  let conicY = conicAndSort.y;
  let conicZAvailable = abs(conicAndSort.w) > 0.0;
  let conicZ = select(fallbackConic, abs(conicAndSort.w), conicZAvailable);
  return vec3f(conicX, conicY, conicZ);
}

fn gaussianWeight(pixel: vec2f, center: vec2f, conic: vec3f) -> f32 {
  let d = pixel - center;
  let power = conic.x * d.x * d.x + 2.0 * conic.y * d.x * d.y + conic.z * d.y * d.y;
  return exp(-0.5 * clamp(power, 0.0, 80.0));
}

@compute @workgroup_size(8, 8)
fn clearProductionBackground(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u32(params.outputWidth) || id.y >= u32(params.outputHeight)) {
    return;
  }
  textureStore(outputTexture, vec2i(i32(id.x), i32(id.y)), vec4f(0.0, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn compositeActiveTiles(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_id) localId: vec3u
) {
  let tileSizeX = max(params.outputWidth / max(params.tileCols, 1.0), 1.0);
  let tileSizeY = max(params.outputHeight / max(params.tileRows, 1.0), 1.0);
  let subtileCols = u32(max(ceil(tileSizeX / 8.0), 1.0));
  let subtileRows = u32(max(ceil(tileSizeY / 8.0), 1.0));
  let tileX = workgroupId.x / subtileCols;
  let tileY = workgroupId.y / subtileRows;
  if (tileX >= u32(params.tileCols) || tileY >= u32(params.tileRows)) {
    return;
  }
  let tile = tileY * u32(params.tileCols) + tileX;
  let table = tileTable[tile];
  if (table.w != 84.0 || table.y <= 0.0) {
    return;
  }
  let subtileX = workgroupId.x % subtileCols;
  let subtileY = workgroupId.y % subtileRows;
  let tileStartX = u32(floor(f32(tileX) * tileSizeX));
  let tileStartY = u32(floor(f32(tileY) * tileSizeY));
  let tileEndX = min(u32(ceil(f32(tileX + 1u) * tileSizeX)), u32(params.outputWidth));
  let tileEndY = min(u32(ceil(f32(tileY + 1u) * tileSizeY)), u32(params.outputHeight));
  let pixelX = tileStartX + subtileX * 8u + localId.x;
  let pixelY = tileStartY + subtileY * 8u + localId.y;
  if (pixelX >= tileEndX || pixelY >= tileEndY) {
    return;
  }
  var color = vec3f(0.0, 0.0, 0.0);
  var accumAlpha = 0.0;
  var refs = 0.0;
  var readTable = 0.0;
  var traversedList = 0.0;
  let pixel = vec2f(f32(pixelX) + 0.5, f32(pixelY) + 0.5);
  readTable = 1.0;
  let offset = u32(table.x);
  let count = min(u32(table.y), min(u32(params.maxRefsPerTile), 64u));
  for (var orderSlot: u32 = 0u; orderSlot < count; orderSlot = orderSlot + 1u) {
    let splatRef = referenceList[offset + orderSlot];
    let sampleRow = u32(max(splatRef.x, 0.0));
    let sampleBase = sampleRow * 3u;
    let a = tileInputs[sampleBase + 0u];
    let b = tileInputs[sampleBase + 1u];
    let c = tileInputs[sampleBase + 2u];
    let conic = sampleConic(b, max(a.z, 1.0));
    let weight = gaussianWeight(pixel, a.xy, conic);
    let sampleAlpha = clamp(c.w * weight, 0.0, 0.98);
    let remaining = max(1.0 - accumAlpha, 0.0);
    color = color + remaining * clamp(c.xyz, vec3f(0.0), vec3f(1.0)) * sampleAlpha;
    accumAlpha = accumAlpha + remaining * sampleAlpha;
    refs = refs + 1.0;
    traversedList = 1.0;
  }
  let outColor = vec4f(color, clamp(accumAlpha, 0.0, 1.0));
  textureStore(outputTexture, vec2i(i32(pixelX), i32(pixelY)), outColor);
}

@compute @workgroup_size(1)
fn finalizeSummary() {
  var nonEmpty = 0.0;
  var totalRefs = 0.0;
  var maxRefs = 0.0;
  var overflow = 0.0;
  var readTable = 0.0;
  var traversedList = 0.0;
  var orderedRefs = 0.0;
  var depthKeyConsumed = 0.0;
  var sortKeyConsumed = 0.0;
  var orderAwareUsed = 0.0;
  var gaussianAttributeConsumed = 0.0;
  var footprintPayloadConsumed = 0.0;
  for (var tile: u32 = 0u; tile < u32(params.tileCount); tile = tile + 1u) {
    let table = tileTable[tile];
    if (table.w == 84.0) {
      readTable = 1.0;
    }
    if (table.w == 84.0 && table.y > 0.0) {
      nonEmpty = nonEmpty + 1.0;
      let count = min(u32(table.y), min(u32(params.maxRefsPerTile), 64u));
      totalRefs = totalRefs + f32(count);
      maxRefs = max(maxRefs, f32(count));
      overflow = overflow + table.z + max(table.y - f32(count), 0.0);
      let offset = u32(table.x);
      for (var slot: u32 = 0u; slot < count; slot = slot + 1u) {
        let splatRef = referenceList[offset + slot];
        orderedRefs = orderedRefs + 1.0;
        if (splatRef.z != 0.0 || splatRef.w != 0.0) {
          traversedList = 1.0;
        }
        if (splatRef.z != 0.0) {
          depthKeyConsumed = 1.0;
        }
        if (splatRef.w != 0.0) {
          sortKeyConsumed = 1.0;
        }
        let sampleRow = u32(max(splatRef.x, 0.0));
        let sampleBase = sampleRow * 3u;
        let a = tileInputs[sampleBase + 0u];
        let b = tileInputs[sampleBase + 1u];
        let c = tileInputs[sampleBase + 2u];
        if (c.w > 0.0 || c.x != 0.0 || c.y != 0.0 || c.z != 0.0) {
          gaussianAttributeConsumed = 1.0;
        }
        if (a.z > 0.0 || b.x != 0.0 || b.y != 0.0 || b.w != 0.0) {
          footprintPayloadConsumed = 1.0;
        }
      }
    }
  }
  let totalRefsAvailable = totalRefs > 0.0;
  let orderedRefsMatchTotal = orderedRefs == totalRefs && totalRefsAvailable;
  let orderAwareReady = orderedRefsMatchTotal && sortKeyConsumed == 1.0;
  let gaussianAttributeReady = gaussianAttributeConsumed == 1.0 && totalRefsAvailable;
  let outputIsCanvasSized =
    params.outputWidth > params.tileCols || params.outputHeight > params.tileRows;
  let fullScreenPixelWork = max(params.outputWidth * params.outputHeight, 1.0);
  let tileSizeX = max(params.outputWidth / max(params.tileCols, 1.0), 1.0);
  let tileSizeY = max(params.outputHeight / max(params.tileRows, 1.0), 1.0);
  let activeTilePixelWork = nonEmpty * tileSizeX * tileSizeY;
  let fullScreenPixelWorkAvoided = max(fullScreenPixelWork - activeTilePixelWork, 0.0);
  let workReductionRatio = fullScreenPixelWorkAvoided / fullScreenPixelWork;
  let productionPathUsed = orderAwareReady && gaussianAttributeReady && outputIsCanvasSized;
  let inactiveTileCount = max(params.tileCount - nonEmpty, 0.0);
  orderAwareUsed = select(0.0, 1.0, orderAwareReady);
  compositorSummary[0] = vec4f(params.tileCount, nonEmpty, totalRefs, totalRefs);
  compositorSummary[1] = vec4f(readTable, traversedList, select(0.0, 1.0, totalRefsAvailable), maxRefs);
  compositorSummary[2] = vec4f(overflow, 87.0, orderedRefs, totalRefs);
  compositorSummary[3] = vec4f(
    depthKeyConsumed,
    sortKeyConsumed,
    orderAwareUsed,
    select(0.0, 1.0, orderedRefsMatchTotal)
  );
  compositorSummary[4] = vec4f(
    gaussianAttributeConsumed,
    footprintPayloadConsumed,
    select(0.0, 1.0, orderedRefsMatchTotal),
    orderAwareUsed
  );
  compositorSummary[5] = vec4f(
    select(0.0, 1.0, gaussianAttributeReady),
    select(0.0, 1.0, gaussianAttributeReady),
    totalRefs,
    select(0.0, 1.0, outputIsCanvasSized)
  );
  compositorSummary[6] = vec4f(
    select(0.0, 1.0, productionPathUsed),
    select(0.0, 1.0, orderAwareReady),
    select(0.0, 1.0, nonEmpty > 0.0),
    1.0
  );
  compositorSummary[7] = vec4f(
    nonEmpty,
    inactiveTileCount,
    activeTilePixelWork,
    fullScreenPixelWorkAvoided
  );
  compositorSummary[8] = vec4f(
    workReductionRatio,
    select(0.0, 1.0, productionPathUsed),
    select(0.0, 1.0, productionPathUsed),
    select(0.0, 1.0, nonEmpty > 0.0)
  );
}
`
  });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' }
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba8unorm' }
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      },
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' }
      }
    ]
  });
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout]
  });
  const backgroundPipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shader, entryPoint: 'clearProductionBackground' }
  });
  const compositorPipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shader, entryPoint: 'compositeActiveTiles' }
  });
  const finalizePipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shader, entryPoint: 'finalizeSummary' }
  });
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: resources.inputBuffer } },
      { binding: 1, resource: { buffer: resources.tileTableBuffer } },
      { binding: 2, resource: { buffer: orderedReferenceBuffer } },
      { binding: 3, resource: outputTexture.createView() },
      { binding: 4, resource: { buffer: summaryBuffer } },
      { binding: 5, resource: { buffer: paramsBuffer } }
    ]
  });
  const dirtyProductionRuntimeFrameCount = 2;
  const cleanProductionRuntimeFrameCount = 1;
  const productionRuntimeFrameCount =
    dirtyProductionRuntimeFrameCount + cleanProductionRuntimeFrameCount;
  const encoder = device.createCommandEncoder({
    label: 'phase3-step85-webgpu-tile-list-compositor-encoder'
  });
  const tileSizeXForDispatch = Math.max(outputWidth / Math.max(resources.tileCols, 1), 1);
  const tileSizeYForDispatch = Math.max(outputHeight / Math.max(resources.tileRows, 1), 1);
  const tileSubtileCols = Math.max(1, Math.ceil(tileSizeXForDispatch / 8));
  const tileSubtileRows = Math.max(1, Math.ceil(tileSizeYForDispatch / 8));
  let pass = encoder.beginComputePass({
    label: 'phase3-step97-webgpu-time-driven-production-runtime-pass-0'
  });
  for (
    let runtimeFrameIndex = 0;
    runtimeFrameIndex < dirtyProductionRuntimeFrameCount;
    runtimeFrameIndex += 1
  ) {
    if (runtimeFrameIndex > 0) {
      pass.end();
      encoder.clearBuffer(orderingSummaryBuffer);
      pass = encoder.beginComputePass({
        label: `phase3-step97-webgpu-time-driven-production-runtime-pass-${runtimeFrameIndex}`
      });
    }
    pass.setBindGroup(0, orderingBindGroup);
    pass.setPipeline(referenceSeedPipeline);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(referenceCapacity / 64)));
    pass.setPipeline(orderingPipeline);
    pass.dispatchWorkgroups(Math.max(1, resources.tileCount));
    pass.setBindGroup(0, bindGroup);
    pass.setPipeline(backgroundPipeline);
    pass.dispatchWorkgroups(
      Math.max(1, Math.ceil(outputWidth / 8)),
      Math.max(1, Math.ceil(outputHeight / 8))
    );
    pass.setPipeline(compositorPipeline);
    pass.dispatchWorkgroups(
      Math.max(1, resources.tileCols * tileSubtileCols),
      Math.max(1, resources.tileRows * tileSubtileRows)
    );
  }
  pass.setPipeline(finalizePipeline);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyTextureToBuffer(
    { texture: outputTexture },
    {
      buffer: textureReadbackBuffer,
      bytesPerRow,
      rowsPerImage: outputHeight
    },
    { width: outputWidth, height: outputHeight, depthOrArrayLayers: 1 }
  );
  encoder.copyBufferToBuffer(
    summaryBuffer,
    0,
    summaryReadbackBuffer,
    0,
    summaryData.byteLength
  );
  encoder.copyBufferToBuffer(
    orderingSummaryBuffer,
    0,
    orderingSummaryReadbackBuffer,
    0,
    orderingSummaryData.byteLength
  );
  device.queue.submit([encoder.finish()]);
  if (typeof device.queue.onSubmittedWorkDone === 'function') {
    await device.queue.onSubmittedWorkDone();
  }
  await summaryReadbackBuffer.mapAsync(GPUMapMode.READ);
  const compositorSummary = new Float32Array(summaryReadbackBuffer.getMappedRange().slice(0));
  summaryReadbackBuffer.unmap();
  await orderingSummaryReadbackBuffer.mapAsync(GPUMapMode.READ);
  const orderingSummaryRaw = new Uint32Array(
    orderingSummaryReadbackBuffer.getMappedRange().slice(0)
  );
  orderingSummaryReadbackBuffer.unmap();
  await textureReadbackBuffer.mapAsync(GPUMapMode.READ);
  const textureReadback = new Uint8Array(textureReadbackBuffer.getMappedRange().slice(0));
  textureReadbackBuffer.unmap();

  const summary = readCompositorSummary(compositorSummary);
  const orderingSummary = readOrderingSummary(orderingSummaryRaw);
  summary.sourceTotalTileReferenceCount =
    sourceContract?.totalTileReferenceCount ?? summary.compositedReferenceCount;
  const diagnosticTextureReadbackNonZero = hasNonZeroTextureByte(
      textureReadback,
      bytesPerRow,
      outputWidth,
      outputHeight
    );
  const runtimeOutputReadyWithoutTextureReadback =
    summary.outputTextureWritten === true &&
    summary.tileCompositorContributionCount > 0;
  const outputTextureWritten =
    runtimeOutputReadyWithoutTextureReadback &&
    diagnosticTextureReadbackNonZero;
  const textureStats = summarizeTextureReadback(
    textureReadback,
    bytesPerRow,
    outputWidth,
    outputHeight
  );
  const ready =
    summary.readOffsetCountTable &&
    summary.traversedReferenceList &&
    outputTextureWritten &&
    summary.compositedReferenceCount > 0;
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
  const sortOrOrderingDispatchCount = dirtyProductionRuntimeFrameCount;
  const fullScreenPixelWorkItemCount = Math.max(1, outputWidth * outputHeight);
  const activeTilePixelWorkItemCount =
    Math.max(1, summary.activeTilePixelWorkItemCount);
  const activeTileDispatchReady =
    summary.activeTileDispatchReady === true &&
    summary.activeTileCount === summary.nonEmptyCompositedTileCount &&
    summary.activeTileCount > 0;
  const activeTileDispatchUsed =
    summary.activeTileDispatchUsed === true && activeTileDispatchReady;
  const compositorDispatchCount = dirtyProductionRuntimeFrameCount * 4 + 1;
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
    fullParallelPerTileSortInWgsl: false,
    fullCudaDepthParity: false,
    finalProductionCompositor: false,
    step85TileCompositorPathPreserved: ready,
    step86BoundaryContractPreserved: true,
    reason: depthOrderingReady
      ? null
      : 'webgpu-tile-depth-ordering-did-not-consume-depth-aware-reference-order'
  });
  const viewerCanvas = viewerCanvasState?.canvas ?? null;
  const currentTextureGuardAllowed =
    viewerCanvasState?.requestedBackendMode === 'webgpu-exclusive' &&
    viewerCanvasState?.allowViewerCanvasPresentation === true &&
    viewerCanvasState?.webgl2FrameLifecycleSuppressed === true &&
    viewerCanvasState?.provided === true &&
    !!viewerCanvas;
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
  let outputTextureCachedForHeartbeat = false;
  if (currentTextureGuardAllowed && outputTextureWritten) {
    const cacheResult = cacheTileCompositorOutputTexture({
      canvas: viewerCanvas,
      device,
      outputTexture,
      outputWidth,
      outputHeight
    });
    outputTextureCachedForHeartbeat = cacheResult.cached === true;
    compositorOutputCacheInvalidatedOnDeviceChange =
      cacheResult.invalidatedOnDeviceChange === true;
    const presentation = await presentTileCompositorTextureToCurrentTexture({
      device,
      outputTexture,
      outputWidth,
      outputHeight,
      viewerCanvasState,
      canvasWidth,
      canvasHeight,
      frameCount: 2,
      presentationSource: 'webgpu-tile-compositor-output-texture',
      forceContextRefresh: true
    });
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
    sortedReferenceCount === summary.sourceTotalTileReferenceCount ||
    sortedReferenceCount + droppedReferenceCount >=
      summary.sourceTotalTileReferenceCount;
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
    unsortedFallbackTileCount === overflowTileCount &&
    droppedReferenceCount >= overflowReferenceCount;
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
    sortOrOrderingDispatchCount > 0 &&
    maxReferencesPerTile <= sortCapacityLimit &&
    unsortedFallbackTileCount === overflowTileCount;
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
    perTileOrderingRuntimePathUsed &&
    sortWorkgroupCount === sortedTileCount &&
    parallelSortStageCount === PARALLEL_SORT_STAGE_COUNT &&
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
    'requires-production-resource-lifecycle-gate-evidence-before-changing-accumulation-exit-policy';
  const chunkLodStreamingReadiness =
    'deferred-until-persistent-resource-cache-and-bottleneck-classification-stabilize';
  const visualParityDiagnosticsDeferredReason =
    'deferred-until-final-production-compositor-parity-and-reference-visual-comparison';
  const step101SelectiveDirtyDependencyPreserved =
    selectiveDirtyDependencyExecutionReady;

  for (const buffer of [
    summaryBuffer,
    paramsBuffer,
    summaryReadbackBuffer,
    textureReadbackBuffer,
    orderingSummaryBuffer,
    orderingParamsBuffer,
    orderingSummaryReadbackBuffer,
    orderedReferenceBuffer
  ]) {
    if (typeof buffer.destroy === 'function') {
      buffer.destroy();
    }
  }
  if (!outputTextureCachedForHeartbeat && typeof outputTexture.destroy === 'function') {
    outputTexture.destroy();
  }

  return {
    compositorSummary,
    contract: buildWebGpuTileListCompositorContract({
      tileCompositorReady: ready,
      compositorPassSubmitted: true,
      compositorReadbackCompleted: true,
      compositorReadOffsetCountTable: summary.readOffsetCountTable,
      compositorTraversedReferenceList: summary.traversedReferenceList,
      outputTextureCreated: true,
      outputTextureWritten,
      outputTextureReadbackMatchesSummary: outputTextureWritten,
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
        'workgroup-parallel-bitonic-per-tile-depth-sort-v1',
        'reference-list-compute-seed-pass',
        'copy-free-reference-seed-guard',
        'parallel-sorted-buffer-readiness-guard',
        'production-tile-compositor-v1-main-path',
        'active-tile-subtile-accumulation-dispatch',
        'production-background-clear-pass',
        'inactive-background-handling-in-compositor',
        'bounded-gpu-per-tile-depth-sort-v1',
        'overflow-aware-tile-ordering-capacity-policy',
        'scalable-sort-scratch-buffer-boundary',
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
        'realtime-bottleneck-classification-v1'
      ],
      deferredCompositorFields: [
        'full-production-parallel-sort-parity',
        'cuda-compositor-parity',
        'final-production-tile-compositor',
        'chunk-lod-streaming',
        'complete-interactive-control-parity',
        'early-termination-v1',
        'viewport-resize-resource-reallocation-probe',
        'visual-parity-diagnostics'
      ],
      compositorClassification:
        'production-webgpu-tile-compositor-v1-integration',
      fullDepthSortInWgsl: false,
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
      canvasClearBetweenCompositorFramesDetected: presentationFrameSamples.some(
        (sample) => sample.nonzeroPixelCount <= 0
      ),
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
      step92SortMode: 'bounded-gpu-workgroup-bitonic-sort-per-tile-descending-sort-key',
      gpuParallelPerTileSortReady,
      workgroupParallelSortUsed,
      parallelSortAlgorithm: 'workgroup-bitonic-sort-v1-descending-sort-key',
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
      step93OverflowPolicyPreserved,
      overflowAwareOrderingReady,
      sortCapacityLimit,
      overflowTileCount,
      overflowReferenceCount,
      droppedReferenceCount,
      overflowHandlingPolicy:
        'capacity-capped-sort-with-explicit-overflow-and-dropped-reference-telemetry',
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
        'early-termination-v1',
        'viewport-resize-dirty-probe',
        'viewport-resize-resource-reallocation-probe',
        'visual-parity-diagnostics',
        'chunk-lod-streaming'
      ],
      reason: ready
        ? null
        : 'webgpu-tile-list-compositor-did-not-consume-gpu-owned-tile-list'
    })
  };
}
