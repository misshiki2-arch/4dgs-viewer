import {
  buildWebGpuTileDepthOrderingContract,
  buildWebGpuTileListCompositorContract
} from './common_4dgs_record_contracts.js';

const COMPOSITOR_SUMMARY_FLOAT_COUNT = 24;
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
      Math.round(finiteNumberOr(summary[23], 0)) === 1
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
  let conicX = select(fallbackConic, abs(conicAndSort.x), abs(conicAndSort.x) > 0.0);
  let conicY = conicAndSort.y;
  let conicZ = select(fallbackConic, abs(conicAndSort.w), abs(conicAndSort.w) > 0.0);
  return vec3f(conicX, conicY, conicZ);
}

fn gaussianWeight(pixel: vec2f, center: vec2f, conic: vec3f) -> f32 {
  let d = pixel - center;
  let power = conic.x * d.x * d.x + 2.0 * conic.y * d.x * d.y + conic.z * d.y * d.y;
  return exp(-0.5 * clamp(power, 0.0, 80.0));
}

@compute @workgroup_size(8, 8)
fn compositeTiles(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u32(params.outputWidth) || id.y >= u32(params.outputHeight)) {
    return;
  }
  let tileSizeX = max(params.canvasWidth / max(params.tileCols, 1.0), 1.0);
  let tileSizeY = max(params.canvasHeight / max(params.tileRows, 1.0), 1.0);
  let tileX = min(u32(floor(f32(id.x) / tileSizeX)), u32(params.tileCols) - 1u);
  let tileY = min(u32(floor(f32(id.y) / tileSizeY)), u32(params.tileRows) - 1u);
  let tile = tileY * u32(params.tileCols) + tileX;
  let table = tileTable[tile];
  var color = vec3f(0.0, 0.0, 0.0);
  var accumAlpha = 0.0;
  var refs = 0.0;
  var readTable = 0.0;
  var traversedList = 0.0;
  let pixel = vec2f(f32(id.x) + 0.5, f32(id.y) + 0.5);
  if (table.w == 84.0 && table.y > 0.0) {
    readTable = 1.0;
    let offset = u32(table.x);
    let count = u32(table.y);
    var consumed: array<u32, 64>;
    for (var initSlot: u32 = 0u; initSlot < 64u; initSlot = initSlot + 1u) {
      consumed[initSlot] = 0u;
    }
    for (var orderSlot: u32 = 0u; orderSlot < count; orderSlot = orderSlot + 1u) {
      var bestSlot = 0u;
      var bestKey = -340282346638528859811704183484516925440.0;
      for (var scanSlot: u32 = 0u; scanSlot < count; scanSlot = scanSlot + 1u) {
        let candidateRef = referenceList[offset + scanSlot];
        let candidateKey = candidateRef.w;
        if (consumed[scanSlot] == 0u && candidateKey >= bestKey) {
          bestKey = candidateKey;
          bestSlot = scanSlot;
        }
      }
      consumed[bestSlot] = 1u;
      let splatRef = referenceList[offset + bestSlot];
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
  }
  let outColor = vec4f(color, clamp(accumAlpha, 0.0, 1.0));
  textureStore(outputTexture, vec2i(i32(id.x), i32(id.y)), outColor);
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
      totalRefs = totalRefs + table.y;
      maxRefs = max(maxRefs, table.y);
      overflow = overflow + table.z;
      let offset = u32(table.x);
      let count = u32(table.y);
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
  orderAwareUsed = select(0.0, 1.0, orderedRefs == totalRefs && totalRefs > 0.0 && sortKeyConsumed == 1.0);
  compositorSummary[0] = vec4f(params.tileCount, nonEmpty, totalRefs, totalRefs);
  compositorSummary[1] = vec4f(readTable, traversedList, select(0.0, 1.0, totalRefs > 0.0), maxRefs);
  compositorSummary[2] = vec4f(overflow, 87.0, orderedRefs, totalRefs);
  compositorSummary[3] = vec4f(
    depthKeyConsumed,
	    sortKeyConsumed,
	    orderAwareUsed,
	    select(0.0, 1.0, orderedRefs == totalRefs && totalRefs > 0.0)
	  );
  compositorSummary[4] = vec4f(
    gaussianAttributeConsumed,
    footprintPayloadConsumed,
    select(0.0, 1.0, orderedRefs == totalRefs && totalRefs > 0.0),
    orderAwareUsed
  );
  compositorSummary[5] = vec4f(
    select(0.0, 1.0, gaussianAttributeConsumed == 1.0 && totalRefs > 0.0),
    select(0.0, 1.0, gaussianAttributeConsumed == 1.0 && totalRefs > 0.0),
    totalRefs,
    select(0.0, 1.0, params.outputWidth > params.tileCols || params.outputHeight > params.tileRows)
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
  const compositorPipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shader, entryPoint: 'compositeTiles' }
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
      { binding: 2, resource: { buffer: resources.referenceListBuffer } },
      { binding: 3, resource: outputTexture.createView() },
      { binding: 4, resource: { buffer: summaryBuffer } },
      { binding: 5, resource: { buffer: paramsBuffer } }
    ]
  });
  const encoder = device.createCommandEncoder({
    label: 'phase3-step85-webgpu-tile-list-compositor-encoder'
  });
  const pass = encoder.beginComputePass({
    label: 'phase3-step85-webgpu-tile-list-compositor-pass'
  });
  pass.setBindGroup(0, bindGroup);
  pass.setPipeline(compositorPipeline);
  pass.dispatchWorkgroups(
    Math.max(1, Math.ceil(outputWidth / 8)),
    Math.max(1, Math.ceil(outputHeight / 8))
  );
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
  device.queue.submit([encoder.finish()]);
  if (typeof device.queue.onSubmittedWorkDone === 'function') {
    await device.queue.onSubmittedWorkDone();
  }
  await summaryReadbackBuffer.mapAsync(GPUMapMode.READ);
  const compositorSummary = new Float32Array(summaryReadbackBuffer.getMappedRange().slice(0));
  summaryReadbackBuffer.unmap();
  await textureReadbackBuffer.mapAsync(GPUMapMode.READ);
  const textureReadback = new Uint8Array(textureReadbackBuffer.getMappedRange().slice(0));
  textureReadbackBuffer.unmap();

  const summary = readCompositorSummary(compositorSummary);
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
  const compositorDispatchCount = 2;
  const compositorWorkItemCount = Math.max(1, outputWidth * outputHeight);
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

  for (const buffer of [summaryBuffer, paramsBuffer, summaryReadbackBuffer, textureReadbackBuffer]) {
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
        'sort-key-descending-compositor-order',
        'gaussian-footprint-weighted-alpha-accumulation',
        'gaussian-attribute-color-accumulation',
        'canvas-resolution-rgba8unorm-output-texture',
        'readback-free-steady-state-compositor-runtime-path',
        'gpu-owned-runtime-resource-flow'
      ],
      deferredCompositorFields: [
        'full-parallel-per-tile-sort-dispatch',
        'cuda-compositor-parity',
        'final-production-tile-compositor',
        'chunk-lod-streaming'
      ],
      compositorClassification: 'partial-webgpu-realtime-tile-compositor-runtime',
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
      compositorUpdateFrameCount: 1,
      presentationHeartbeatFrameCount: presentationFrameCount,
      lastValidCompositorOutputPresentedOnCleanFrames: false,
      dirtySkippedCompositorUpdateButPresentedCachedOutput: false,
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
      deferredProductionItems: [
        'full-cuda-parity',
        'final-production-compositor',
        'full-parallel-sort',
        'chunk-lod-streaming'
      ],
      reason: ready
        ? null
        : 'webgpu-tile-list-compositor-did-not-consume-gpu-owned-tile-list'
    })
  };
}
