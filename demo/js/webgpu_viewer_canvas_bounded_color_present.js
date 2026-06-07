import {
  canProbeViewerCanvasCurrentTexturePath
} from './webgpu_viewer_canvas_current_texture_path.js';
import {
  WEBGPU_BACKEND_MODE_VALUES,
  normalizeWebGpuBackendMode
} from './webgpu_exclusive_canvas_handoff.js';

export const WEBGPU_VIEWER_CANVAS_BOUNDED_COLOR_PRESENT_MODE =
  'webgpu-viewer-canvas-bounded-color-present';

const VERTEX_FLOATS = 6;
const MAX_PRESENT_SAMPLES = 8;

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function clamp01(value, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(1, Math.max(0, n));
}

function normalizeCanvasFormat(format) {
  return format === 'bgra8unorm' || format === 'rgba8unorm'
    ? format
    : 'rgba8unorm';
}

function normalizeColorArray(value) {
  if (!Array.isArray(value) && !(value instanceof Float32Array)) return null;
  if (value.length < 3) return null;
  const a = value.length >= 4 ? value[3] : 1;
  return {
    r: clamp01(value[0], 0),
    g: clamp01(value[1], 0),
    b: clamp01(value[2], 0),
    a: clamp01(a, 1)
  };
}

function colorFromSample(sample) {
  return (
    normalizeColorArray(sample?.actual?.rgbaFloat) ??
    normalizeColorArray(sample?.expected?.rgbaFloat) ??
    normalizeColorArray(sample?.actual?.resolvedColor) ??
    normalizeColorArray(sample?.expected?.resolvedColor) ??
    normalizeColorArray(sample?.actual?.colorAlpha) ??
    normalizeColorArray(sample?.expected?.colorAlpha) ??
    normalizeColorArray(sample?.colorAlpha) ??
    normalizeColorArray(sample?.payload?.colorAlpha) ??
    normalizeColorArray(sample?.referenceAssisted?.colorAlpha) ??
    normalizeColorArray(sample?.renderPayload?.colorAlpha)
  );
}

function samplePxFromSample(sample, fallbackIndex) {
  const px =
    sample?.pixel ??
    sample?.samplePx ??
    sample?.actual?.pixel ??
    sample?.expected?.pixel ??
    sample?.centerPx;
  if (Array.isArray(px) && px.length >= 2) {
    return { x: Number(px[0]), y: Number(px[1]) };
  }
  if (px && typeof px === 'object') {
    return { x: Number(px.x ?? px[0]), y: Number(px.y ?? px[1]) };
  }
  const offset = fallbackIndex * 24;
  return { x: 32 + offset, y: 32 + offset };
}

function pushSamplesFromList({
  samples,
  list,
  source,
  colorSource,
  canvasWidth,
  canvasHeight
}) {
  if (!Array.isArray(list)) return;
  for (const sample of list) {
    if (samples.length >= MAX_PRESENT_SAMPLES) return;
    const color = colorFromSample(sample);
    if (!color) continue;
    const samplePx = samplePxFromSample(sample, samples.length);
    const x = Number.isFinite(samplePx.x) ? samplePx.x : 32 + samples.length * 24;
    const y = Number.isFinite(samplePx.y) ? samplePx.y : 32 + samples.length * 24;
    samples.push({
      source,
      colorSource,
      recordIndex: sample?.recordIndex ?? sample?.anchorRecordIndex ?? null,
      sampleKind: sample?.sampleKind ?? null,
      srcIndex: sample?.srcIndex ?? null,
      valid: sample?.valid ?? null,
      samplePx: {
        x: Math.min(Math.max(0, x), Math.max(0, canvasWidth - 1)),
        y: Math.min(Math.max(0, y), Math.max(0, canvasHeight - 1))
      },
      colorAlpha: color
    });
  }
}

function buildColorSamples({
  webgpuConstrainedDisplayAdapterDryRunComparison,
  webgpuRenderTargetHandoffDryRunComparison,
  webgpuFramebufferFreeTileOutputDryRunComparison,
  webgpuRenderHandoffStub,
  canvasWidth,
  canvasHeight
}) {
  const samples = [];
  pushSamplesFromList({
    samples,
    list: webgpuConstrainedDisplayAdapterDryRunComparison?.sampleTexturePixels,
    source: 'webgpuConstrainedDisplayAdapterDryRunComparison.sampleTexturePixels',
    colorSource: 'Step40 constrained display adapter rgbaFloat sample',
    canvasWidth,
    canvasHeight
  });
  pushSamplesFromList({
    samples,
    list: webgpuRenderTargetHandoffDryRunComparison?.sampleRenderTargetPixels,
    source: 'webgpuRenderTargetHandoffDryRunComparison.sampleRenderTargetPixels',
    colorSource: 'Step39 render target handoff resolvedColor sample',
    canvasWidth,
    canvasHeight
  });
  pushSamplesFromList({
    samples,
    list: webgpuFramebufferFreeTileOutputDryRunComparison?.sampleTileOutputs,
    source: 'webgpuFramebufferFreeTileOutputDryRunComparison.sampleTileOutputs',
    colorSource: 'Step38 framebuffer-free tile output resolvedColor sample',
    canvasWidth,
    canvasHeight
  });
  pushSamplesFromList({
    samples,
    list: webgpuRenderHandoffStub?.sampleRecords,
    source: 'webgpuRenderHandoffStub.sampleRecords',
    colorSource: 'reference-assisted render payload colorAlpha.rgb',
    canvasWidth,
    canvasHeight
  });
  return samples;
}

function createVertexData(samples, canvasWidth, canvasHeight) {
  const vertices = [];
  const markerHalfSizePx = Math.max(4, Math.min(18, Math.round(Math.min(canvasWidth, canvasHeight) * 0.0125)));
  for (const sample of samples) {
    const x0 = Math.max(0, sample.samplePx.x - markerHalfSizePx);
    const x1 = Math.min(canvasWidth, sample.samplePx.x + markerHalfSizePx);
    const y0 = Math.max(0, sample.samplePx.y - markerHalfSizePx);
    const y1 = Math.min(canvasHeight, sample.samplePx.y + markerHalfSizePx);
    const ndc = (x, y) => ({
      x: (x / Math.max(1, canvasWidth)) * 2 - 1,
      y: 1 - (y / Math.max(1, canvasHeight)) * 2
    });
    const p00 = ndc(x0, y0);
    const p10 = ndc(x1, y0);
    const p01 = ndc(x0, y1);
    const p11 = ndc(x1, y1);
    const color = sample.colorAlpha;
    const displayAlpha = Math.max(color.a, 1);
    const push = (p) => {
      vertices.push(p.x, p.y, color.r, color.g, color.b, displayAlpha);
    };
    push(p00); push(p10); push(p01);
    push(p10); push(p11); push(p01);
  }
  return new Float32Array(vertices);
}

function createBuffer(device, data, usage) {
  const buffer = device.createBuffer({
    size: Math.max(4, data.byteLength),
    usage,
    mappedAtCreation: true
  });
  new Float32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

function buildFailures({
  deviceAvailable,
  viewerCanvasProvided,
  exclusiveBackendModeRequested,
  allowViewerCanvasPresentation,
  webgl2FrameLifecycleSuppressed,
  viewerCanvasWebgl2Active,
  currentTexturePathReady,
  boundedFirstPresentSucceeded,
  colorSampleCount,
  colorPresentSubmitted
}) {
  const failures = [];
  if (!deviceAvailable) failures.push({ stage: 'device', reason: 'webgpu device is unavailable' });
  if (!viewerCanvasProvided) failures.push({ stage: 'viewer-canvas', reason: 'viewer canvas was not provided' });
  if (!exclusiveBackendModeRequested) failures.push({ stage: 'backend-mode', reason: 'webgpu-exclusive backend mode was not requested' });
  if (!allowViewerCanvasPresentation) failures.push({ stage: 'presentation-guard', reason: 'viewer canvas presentation guard is not enabled' });
  if (!webgl2FrameLifecycleSuppressed) failures.push({ stage: 'frame-lifecycle', reason: 'WebGL2 frame lifecycle was not suppressed before bounded color present' });
  if (viewerCanvasWebgl2Active) failures.push({ stage: 'canvas-ownership', reason: 'viewer canvas is already owned by WebGL2' });
  if (!currentTexturePathReady) failures.push({ stage: 'current-texture-path', reason: 'viewer canvas currentTexture path is not ready' });
  if (!boundedFirstPresentSucceeded) failures.push({ stage: 'bounded-first-present', reason: 'Step46 bounded first-present did not succeed before bounded color present' });
  if (colorSampleCount <= 0) failures.push({ stage: 'bounded-color-samples', reason: 'no 4DGS-derived bounded color samples are available' });
  if (!colorPresentSubmitted) failures.push({ stage: 'bounded-color-present', reason: 'bounded color present command buffer was not submitted' });
  return failures;
}

export async function buildWebGpuViewerCanvasBoundedColorPresent({
  device,
  viewerCanvasState = null,
  webgpuViewerCanvasCurrentTexturePath = null,
  webgpuViewerCanvasBoundedFirstPresent = null,
  webgpuRenderHandoffStub = null,
  webgpuFramebufferFreeTileOutputDryRunComparison = null,
  webgpuRenderTargetHandoffDryRunComparison = null,
  webgpuConstrainedDisplayAdapterDryRunComparison = null
} = {}) {
  const startMs = nowMs();
  const requestedBackendMode = normalizeWebGpuBackendMode(
    viewerCanvasState?.requestedBackendMode
  );
  const contextMode = viewerCanvasState?.contextMode ?? 'not-provided';
  const canvas = viewerCanvasState?.canvas ?? null;
  const viewerCanvasProvided = viewerCanvasState?.provided === true && !!canvas;
  const allowViewerCanvasPresentation =
    viewerCanvasState?.allowViewerCanvasPresentation === true;
  const webgl2FrameLifecycleSuppressed =
    viewerCanvasState?.webgl2FrameLifecycleSuppressed === true;
  const exclusiveBackendModeRequested =
    requestedBackendMode === WEBGPU_BACKEND_MODE_VALUES.WEBGPU_EXCLUSIVE;
  const viewerCanvasWebgl2Active = contextMode === 'webgl2-active';
  const currentTexturePathReady =
    webgpuViewerCanvasCurrentTexturePath?.viewerCanvasCurrentTexturePathReady === true;
  const boundedFirstPresentSucceeded =
    webgpuViewerCanvasBoundedFirstPresent?.boundedViewerCanvasFirstPresentSucceeded === true;
  const guardAllowed = canProbeViewerCanvasCurrentTexturePath({
    requestedBackendMode,
    allowViewerCanvasPresentation,
    contextMode,
    webgl2FrameLifecycleSuppressed
  }) && currentTexturePathReady && boundedFirstPresentSucceeded;
  const textureFormat = normalizeCanvasFormat(
    webgpuViewerCanvasCurrentTexturePath?.textureFormat ??
      (typeof navigator !== 'undefined' && navigator.gpu?.getPreferredCanvasFormat
        ? navigator.gpu.getPreferredCanvasFormat()
        : 'rgba8unorm')
  );
  const width = Math.max(1, Math.round(canvas?.width ?? canvas?.clientWidth ?? 1));
  const height = Math.max(1, Math.round(canvas?.height ?? canvas?.clientHeight ?? 1));
  const colorSamples = buildColorSamples({
    webgpuConstrainedDisplayAdapterDryRunComparison,
    webgpuRenderTargetHandoffDryRunComparison,
    webgpuFramebufferFreeTileOutputDryRunComparison,
    webgpuRenderHandoffStub,
    canvasWidth: width,
    canvasHeight: height
  });
  const vertexData = createVertexData(colorSamples, width, height);

  let context = null;
  let currentTexture = null;
  let commandBufferSubmitted = false;
  let submittedWorkDone = false;
  let presentError = null;

  if (device && viewerCanvasProvided && guardAllowed && vertexData.length > 0) {
    try {
      context = canvas.getContext?.('webgpu') ?? null;
      if (context) {
        context.configure({
          device,
          format: textureFormat,
          usage:
            GPUTextureUsage.RENDER_ATTACHMENT |
            GPUTextureUsage.COPY_SRC |
            GPUTextureUsage.COPY_DST,
          alphaMode: 'premultiplied'
        });
        currentTexture = context.getCurrentTexture();
        const shader = device.createShaderModule({
          label: 'phase3-step47-viewer-canvas-bounded-color-present-shader',
          code: `
struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
};

@vertex
fn vsMain(
  @location(0) position: vec2f,
  @location(1) color: vec4f
) -> VertexOut {
  var out: VertexOut;
  out.position = vec4f(position, 0.0, 1.0);
  out.color = color;
  return out;
}

@fragment
fn fsMain(in: VertexOut) -> @location(0) vec4f {
  return in.color;
}
`
        });
        const pipeline = device.createRenderPipeline({
          label: 'phase3-step47-viewer-canvas-bounded-color-present-pipeline',
          layout: 'auto',
          vertex: {
            module: shader,
            entryPoint: 'vsMain',
            buffers: [
              {
                arrayStride: VERTEX_FLOATS * 4,
                attributes: [
                  { shaderLocation: 0, offset: 0, format: 'float32x2' },
                  { shaderLocation: 1, offset: 2 * 4, format: 'float32x4' }
                ]
              }
            ]
          },
          fragment: {
            module: shader,
            entryPoint: 'fsMain',
            targets: [{ format: textureFormat }]
          },
          primitive: { topology: 'triangle-list' }
        });
        const vertexBuffer = createBuffer(device, vertexData, GPUBufferUsage.VERTEX);
        const encoder = device.createCommandEncoder({
          label: 'phase3-step47-viewer-canvas-bounded-color-present'
        });
        const pass = encoder.beginRenderPass({
          colorAttachments: [
            {
              view: currentTexture.createView(),
              clearValue: { r: 0, g: 0, b: 0, a: 1 },
              loadOp: 'clear',
              storeOp: 'store'
            }
          ]
        });
        pass.setPipeline(pipeline);
        pass.setVertexBuffer(0, vertexBuffer);
        pass.draw(vertexData.length / VERTEX_FLOATS);
        pass.end();
        device.queue.submit([encoder.finish()]);
        commandBufferSubmitted = true;
        if (typeof device.queue.onSubmittedWorkDone === 'function') {
          await device.queue.onSubmittedWorkDone();
          submittedWorkDone = true;
        }
      }
    } catch (error) {
      presentError = error;
    }
  }

  const contextConfigured = !!context;
  const currentTextureAcquired = !!currentTexture;
  const boundedColorPresentSucceeded =
    commandBufferSubmitted && !presentError && currentTextureAcquired;
  const firstValidationFailures = buildFailures({
    deviceAvailable: !!device,
    viewerCanvasProvided,
    exclusiveBackendModeRequested,
    allowViewerCanvasPresentation,
    webgl2FrameLifecycleSuppressed,
    viewerCanvasWebgl2Active,
    currentTexturePathReady,
    boundedFirstPresentSucceeded,
    colorSampleCount: colorSamples.length,
    colorPresentSubmitted: commandBufferSubmitted
  });
  if (!guardAllowed) {
    firstValidationFailures.push({
      stage: 'exclusive-color-present-guard',
      reason:
        'bounded color present requires Step46 first-present success under guarded webgpu-exclusive ownership'
    });
  }
  if (presentError) {
    firstValidationFailures.push({
      stage: 'queue-submit-or-render-pass',
      reason: presentError.message ?? String(presentError)
    });
  }

  return {
    mode: WEBGPU_VIEWER_CANVAS_BOUNDED_COLOR_PRESENT_MODE,
    status: boundedColorPresentSucceeded ? 'ok' : 'blocked',
    source:
      'Step47 guarded viewer-canvas render pass using bounded 4DGS-derived color samples',
    expectedSource:
      'Step38-40 bounded display samples when available, otherwise Step32 render handoff reference-assisted colorAlpha.rgb samples',
    actualSource: boundedColorPresentSucceeded
      ? 'bounded 4DGS-derived color sample quads submitted to the real viewer canvas WebGPU currentTexture'
      : 'bounded color present was not submitted because a guard, sample, or render pass requirement failed',
    boundedViewerCanvasColorPresentImplemented: true,
    boundedViewerCanvasColorPresentSucceeded: boundedColorPresentSucceeded,
    viewerCanvasPresentationImplemented: boundedColorPresentSucceeded,
    productionDisplayConnectionImplemented: false,
    displayConnectionAllowed: false,
    webgl2HybridRenderingAllowed: false,
    requestedBackendMode,
    guardAllowed,
    canvasContextKind: contextConfigured ? 'viewer-canvas-webgpu' : null,
    viewerCanvasContextConfiguredForColorPresent: contextConfigured,
    viewerCanvasCurrentTextureAcquiredForColorPresent: currentTextureAcquired,
    commandBufferSubmitted,
    submittedWorkDone,
    textureFormat,
    outputExtent: { canvasWidth: width, canvasHeight: height },
    colorPresentSampleCount: colorSamples.length,
    vertexCount: vertexData.length / VERTEX_FLOATS,
    colorOutputContract: {
      boundedOnly: true,
      clearOnly: false,
      colorSourcesAttempted: [
        'Step40 constrained display adapter rgbaFloat samples',
        'Step39 render target handoff resolvedColor samples',
        'Step38 framebuffer-free tile output resolvedColor samples',
        'reference-assisted render payload colorAlpha.rgb samples'
      ],
      selectedColorSource: colorSamples[0]?.colorSource ?? null,
      sampleSources: [...new Set(colorSamples.map((sample) => sample.source))],
      presentedSamples: colorSamples,
      sampleMapping:
        'sample pixel coordinates are clamped to the viewer canvas and rendered as bounded quads',
      alphaDisplayPolicy:
        'rgb comes from the 4DGS-derived sample; display alpha is forced opaque for this bounded presentation proof',
      shEvaluationRequiredForThisPresent: false
    },
    viewerCanvasOwnershipContract: {
      viewerCanvasProvided,
      contextMode,
      allowViewerCanvasPresentation,
      requestedOwner: exclusiveBackendModeRequested ? 'webgpu' : 'webgl2-fallback',
      currentOwner: boundedColorPresentSucceeded
        ? 'webgpu-exclusive-bounded-color-present'
        : viewerCanvasWebgl2Active
          ? 'webgl2'
          : 'none-or-unknown',
      webgl2FrameLifecycleSuppressed,
      requiresExclusiveBackendMode: true,
      requiresPresentationGuard: true,
      requiresCurrentTexturePathReady: true,
      requiresBoundedFirstPresentSucceeded: true,
      requiresNoActiveWebGL2Context: true
    },
    cameraProjectionContract: {
      fixedReferenceCapture:
        'Step47 presents bounded 4DGS-derived color samples and leaves fixed-reference camera/projection data comparable',
      interactiveViewer:
        'Three.js camera and OrbitControls remain camera input adapters; no mouse/control implementation changes',
      controlsImpact:
        'no interactive camera behavior is implemented or changed in Step47'
    },
    shPolicy: {
      requiredForThisBoundedColorPresent: false,
      status: 'deferred',
      fallbackColorSource:
        'reference-assisted colorAlpha.rgb payload when Step38-40 display samples are unavailable',
      displayImpact:
        'bounded color present validates viewer canvas color output plumbing before WGSL SH/color parity or production display connection'
    },
    anyMismatch: !boundedColorPresentSucceeded,
    mismatchClassification: boundedColorPresentSucceeded
      ? 'none'
      : 'viewerCanvasBoundedColorPresentBlocked',
    validationSummary: {
      deviceAvailable: !!device,
      viewerCanvasProvided,
      exclusiveBackendModeRequested,
      viewerCanvasPresentationGuardEnabled: allowViewerCanvasPresentation,
      viewerCanvasWebgl2Active,
      webgl2FrameLifecycleSuppressed,
      currentTexturePathReady,
      boundedFirstPresentSucceeded,
      guardAllowed,
      colorSamplesAvailable: colorSamples.length > 0,
      colorPresentSampleCount: colorSamples.length,
      viewerCanvasContextConfiguredForColorPresent: contextConfigured,
      currentTextureAvailableForColorPresent: currentTextureAcquired,
      commandBufferSubmitted,
      submittedWorkDone,
      boundedViewerCanvasColorPresentSucceeded: boundedColorPresentSucceeded,
      webgl2HybridRenderingPrevented: !viewerCanvasWebgl2Active,
      displayConnectionPrevented: true,
      cameraProjectionContractUnchanged: true,
      firstValidationFailures
    },
    blockers: boundedColorPresentSucceeded
      ? [
          {
            stage: 'production-display-connection',
            reason:
              'bounded color present succeeded; full viewer rendering remains intentionally disconnected'
          },
          {
            stage: 'sh-color-evaluation',
            reason: 'WGSL SH/color evaluation parity remains deferred'
          }
        ]
      : [
          {
            stage: 'viewer-canvas-bounded-color-present',
            reason:
              'bounded color present requires exclusive backend mode, first-present success, color samples, and a successful render pass submit'
          }
        ],
    nextBackendPrototypeStep: boundedColorPresentSucceeded
      ? 'connect guarded tile-composite or render-target output to viewer canvas color present under the same exclusive lifecycle boundary'
      : 'resolve bounded color sample or WebGPU validation failures before connecting broader display output',
    timing: {
      viewerCanvasBoundedColorPresentMs: nowMs() - startMs
    }
  };
}
