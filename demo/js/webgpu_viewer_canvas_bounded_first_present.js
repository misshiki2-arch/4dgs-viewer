import {
  canProbeViewerCanvasCurrentTexturePath
} from './webgpu_viewer_canvas_current_texture_path.js';
import {
  WEBGPU_BACKEND_MODE_VALUES,
  normalizeWebGpuBackendMode
} from './webgpu_exclusive_canvas_handoff.js';

export const WEBGPU_VIEWER_CANVAS_BOUNDED_FIRST_PRESENT_MODE =
  'webgpu-viewer-canvas-bounded-first-present';

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function normalizeCanvasFormat(format) {
  return format === 'bgra8unorm' || format === 'rgba8unorm'
    ? format
    : 'rgba8unorm';
}

function clamp01(value, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(1, Math.max(0, n));
}

function normalizeColorAlpha(colorAlpha) {
  if (!Array.isArray(colorAlpha) && !(colorAlpha instanceof Float32Array)) {
    return null;
  }
  if (colorAlpha.length < 4) return null;
  return {
    r: clamp01(colorAlpha[0], 0),
    g: clamp01(colorAlpha[1], 0),
    b: clamp01(colorAlpha[2], 0),
    a: clamp01(colorAlpha[3], 1)
  };
}

function findReferenceAssistedColor(webgpuRenderHandoffStub) {
  const candidates = [
    webgpuRenderHandoffStub?.sampleRecords,
    webgpuRenderHandoffStub?.samplePayloads,
    webgpuRenderHandoffStub?.records,
    webgpuRenderHandoffStub?.referenceAssistedSamples
  ];
  for (const list of candidates) {
    if (!Array.isArray(list)) continue;
    for (const sample of list) {
      const color =
        normalizeColorAlpha(sample?.colorAlpha) ??
        normalizeColorAlpha(sample?.payload?.colorAlpha) ??
        normalizeColorAlpha(sample?.referenceAssisted?.colorAlpha) ??
        normalizeColorAlpha(sample?.renderPayload?.colorAlpha);
      if (color) return color;
    }
  }
  return null;
}

function buildPresentColor({ webgpuRenderHandoffStub } = {}) {
  const referenceAssistedColor = findReferenceAssistedColor(webgpuRenderHandoffStub);
  if (referenceAssistedColor) {
    return {
      clearValue: referenceAssistedColor,
      colorSource: 'reference-assisted colorAlpha.rgb payload',
      referenceAssistedColorUsed: true,
      diagnosticFallbackUsed: false
    };
  }
  return {
    clearValue: { r: 0.0625, g: 0.125, b: 0.1875, a: 1 },
    colorSource: 'guarded diagnostic clear color',
    referenceAssistedColorUsed: false,
    diagnosticFallbackUsed: true
  };
}

function buildFailures({
  deviceAvailable,
  viewerCanvasProvided,
  exclusiveBackendModeRequested,
  allowViewerCanvasPresentation,
  webgl2FrameLifecycleSuppressed,
  viewerCanvasWebgl2Active,
  currentTexturePathReady,
  firstPresentSubmitted
}) {
  const failures = [];
  if (!deviceAvailable) {
    failures.push({ stage: 'device', reason: 'webgpu device is unavailable' });
  }
  if (!viewerCanvasProvided) {
    failures.push({ stage: 'viewer-canvas', reason: 'viewer canvas was not provided' });
  }
  if (!exclusiveBackendModeRequested) {
    failures.push({
      stage: 'backend-mode',
      reason: 'webgpu-exclusive backend mode was not requested'
    });
  }
  if (!allowViewerCanvasPresentation) {
    failures.push({
      stage: 'presentation-guard',
      reason: 'viewer canvas presentation guard is not enabled'
    });
  }
  if (!webgl2FrameLifecycleSuppressed) {
    failures.push({
      stage: 'frame-lifecycle',
      reason: 'WebGL2 frame lifecycle was not suppressed before bounded first-present'
    });
  }
  if (viewerCanvasWebgl2Active) {
    failures.push({
      stage: 'canvas-ownership',
      reason: 'viewer canvas is already owned by WebGL2'
    });
  }
  if (!currentTexturePathReady) {
    failures.push({
      stage: 'current-texture-path',
      reason: 'viewer canvas currentTexture path is not ready'
    });
  }
  if (!firstPresentSubmitted) {
    failures.push({
      stage: 'bounded-first-present',
      reason: 'bounded viewer canvas first-present command buffer was not submitted'
    });
  }
  return failures;
}

export async function buildWebGpuViewerCanvasBoundedFirstPresent({
  device,
  viewerCanvasState = null,
  webgpuViewerCanvasCurrentTexturePath = null,
  webgpuRenderHandoffStub = null
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
  const guardAllowed = canProbeViewerCanvasCurrentTexturePath({
    requestedBackendMode,
    allowViewerCanvasPresentation,
    contextMode,
    webgl2FrameLifecycleSuppressed
  }) && currentTexturePathReady;
  const preferredFormat =
    webgpuViewerCanvasCurrentTexturePath?.textureFormat ??
    (typeof navigator !== 'undefined' && navigator.gpu?.getPreferredCanvasFormat
      ? navigator.gpu.getPreferredCanvasFormat()
      : 'rgba8unorm');
  const textureFormat = normalizeCanvasFormat(preferredFormat);
  const width = Math.max(1, Math.round(canvas?.width ?? canvas?.clientWidth ?? 1));
  const height = Math.max(1, Math.round(canvas?.height ?? canvas?.clientHeight ?? 1));
  const presentColor = buildPresentColor({ webgpuRenderHandoffStub });

  let context = null;
  let currentTexture = null;
  let commandBufferSubmitted = false;
  let submittedWorkDone = false;
  let presentError = null;

  if (device && viewerCanvasProvided && guardAllowed) {
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
        const encoder = device.createCommandEncoder({
          label: 'phase3-step46-viewer-canvas-bounded-first-present'
        });
        const pass = encoder.beginRenderPass({
          colorAttachments: [
            {
              view: currentTexture.createView(),
              clearValue: presentColor.clearValue,
              loadOp: 'clear',
              storeOp: 'store'
            }
          ]
        });
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
  const boundedFirstPresentSucceeded =
    commandBufferSubmitted && !presentError && currentTextureAcquired;
  const firstValidationFailures = buildFailures({
    deviceAvailable: !!device,
    viewerCanvasProvided,
    exclusiveBackendModeRequested,
    allowViewerCanvasPresentation,
    webgl2FrameLifecycleSuppressed,
    viewerCanvasWebgl2Active,
    currentTexturePathReady,
    firstPresentSubmitted: commandBufferSubmitted
  });
  if (!guardAllowed) {
    firstValidationFailures.push({
      stage: 'exclusive-first-present-guard',
      reason:
        'bounded first-present requires Step45 currentTexture readiness under guarded webgpu-exclusive ownership'
    });
  }
  if (!contextConfigured && guardAllowed) {
    firstValidationFailures.push({
      stage: 'viewer-canvas-webgpu-context',
      reason: 'viewer canvas WebGPU context could not be configured for bounded first-present'
    });
  }
  if (presentError) {
    firstValidationFailures.push({
      stage: 'queue-submit-or-render-pass',
      reason: presentError.message ?? String(presentError)
    });
  }

  return {
    mode: WEBGPU_VIEWER_CANVAS_BOUNDED_FIRST_PRESENT_MODE,
    status: boundedFirstPresentSucceeded ? 'ok' : 'blocked',
    source:
      'Step46 guarded viewer-canvas currentTexture render pass and queue submit',
    expectedSource:
      'Step45 viewer canvas currentTexture readiness plus exclusive frame lifecycle switch',
    actualSource: boundedFirstPresentSucceeded
      ? 'bounded clear render pass submitted to the real viewer canvas WebGPU currentTexture'
      : 'bounded first-present was not submitted because one or more exclusive guards failed',
    boundedViewerCanvasFirstPresentImplemented: true,
    boundedViewerCanvasFirstPresentSucceeded: boundedFirstPresentSucceeded,
    viewerCanvasPresentationImplemented: boundedFirstPresentSucceeded,
    productionDisplayConnectionImplemented: false,
    displayConnectionAllowed: false,
    webgl2HybridRenderingAllowed: false,
    requestedBackendMode,
    guardAllowed,
    canvasContextKind: contextConfigured ? 'viewer-canvas-webgpu' : null,
    viewerCanvasContextConfiguredForPresent: contextConfigured,
    viewerCanvasCurrentTextureAcquiredForPresent: currentTextureAcquired,
    commandBufferSubmitted,
    submittedWorkDone,
    textureFormat,
    outputExtent: { canvasWidth: width, canvasHeight: height },
    presentColorContract: {
      colorSource: presentColor.colorSource,
      referenceAssistedColorUsed: presentColor.referenceAssistedColorUsed,
      diagnosticFallbackUsed: presentColor.diagnosticFallbackUsed,
      clearValue: presentColor.clearValue,
      shEvaluationRequiredForThisPresent: false,
      visualParityPolicy:
        'bounded first-present proves guarded viewer canvas presentation mechanics; full color parity still requires WGSL SH/color evaluation or reference-assisted production display mode'
    },
    viewerCanvasOwnershipContract: {
      viewerCanvasProvided,
      contextMode,
      allowViewerCanvasPresentation,
      requestedOwner: exclusiveBackendModeRequested ? 'webgpu' : 'webgl2-fallback',
      currentOwner: boundedFirstPresentSucceeded
        ? 'webgpu-exclusive-bounded-first-present'
        : viewerCanvasWebgl2Active
          ? 'webgl2'
          : 'none-or-unknown',
      webgl2FrameLifecycleSuppressed,
      requiresExclusiveBackendMode: true,
      requiresPresentationGuard: true,
      requiresCurrentTexturePathReady: true,
      requiresNoActiveWebGL2Context: true,
      handoffPolicy:
        'bounded first-present is allowed only after WebGL2 frame lifecycle is suppressed and Step45 currentTexture readiness is confirmed'
    },
    firstPresentContract: {
      boundedOnly: true,
      productionViewerConnected: false,
      renderWork:
        'single clear-only render pass to viewer canvas currentTexture, used as the first guarded presentation proof',
      resizePolicy:
        'reuse the viewer canvas capture extent; future normal backend should reconfigure on resize before each frame',
      nextConnectionBoundary:
        'replace diagnostic/reference-assisted clear with normal WebGPU tile compositing only after display, color, and interaction contracts are explicit'
    },
    cameraProjectionContract: {
      fixedReferenceCapture:
        'Step46 presents a bounded diagnostic/reference-assisted clear and leaves fixed-reference camera/projection data comparable',
      interactiveViewer:
        'Three.js camera and OrbitControls remain camera input adapters; no mouse/control implementation changes',
      controlsImpact:
        'no interactive camera behavior is implemented or changed in Step46'
    },
    shPolicy: {
      requiredForThisBoundedFirstPresent: false,
      status: 'deferred',
      fallbackColorSource: 'reference-assisted colorAlpha.rgb payload when present, otherwise guarded diagnostic clear color',
      displayImpact:
        'bounded first-present validates canvas presentation mechanics before WGSL SH/color parity; photometric viewer parity remains deferred'
    },
    anyMismatch: !boundedFirstPresentSucceeded,
    mismatchClassification: boundedFirstPresentSucceeded
      ? 'none'
      : 'viewerCanvasBoundedFirstPresentBlocked',
    validationSummary: {
      deviceAvailable: !!device,
      viewerCanvasProvided,
      exclusiveBackendModeRequested,
      viewerCanvasPresentationGuardEnabled: allowViewerCanvasPresentation,
      viewerCanvasWebgl2Active,
      webgl2FrameLifecycleSuppressed,
      currentTexturePathReady,
      guardAllowed,
      viewerCanvasContextConfiguredForPresent: contextConfigured,
      currentTextureAvailableForPresent: currentTextureAcquired,
      commandBufferSubmitted,
      submittedWorkDone,
      boundedViewerCanvasFirstPresentSucceeded: boundedFirstPresentSucceeded,
      webgl2HybridRenderingPrevented: !viewerCanvasWebgl2Active,
      displayConnectionPrevented: true,
      cameraProjectionContractUnchanged: true,
      firstValidationFailures
    },
    blockers: boundedFirstPresentSucceeded
      ? [
          {
            stage: 'production-display-connection',
            reason:
              'bounded first-present succeeded; production viewer rendering remains intentionally disconnected'
          },
          {
            stage: 'sh-color-evaluation',
            reason: 'WGSL SH/color evaluation parity remains deferred'
          }
        ]
      : [
          {
            stage: 'viewer-canvas-bounded-first-present',
            reason:
              'bounded first-present requires exclusive backend mode, presentation guard, currentTexture readiness, and a successful render pass submit'
          }
        ],
    nextBackendPrototypeStep: boundedFirstPresentSucceeded
      ? 'connect a guarded WebGPU display adapter from the tile composite output to viewer canvas under the same exclusive lifecycle boundary'
      : 'resolve bounded first-present guard or WebGPU validation failures before connecting display adapter output',
    timing: {
      viewerCanvasBoundedFirstPresentMs: nowMs() - startMs
    }
  };
}
