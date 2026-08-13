import {
  WEBGPU_BACKEND_MODE_VALUES,
  normalizeWebGpuBackendMode
} from './webgpu_exclusive_canvas_handoff.js';

export const WEBGPU_VIEWER_CANVAS_CURRENT_TEXTURE_PATH_MODE =
  'webgpu-viewer-canvas-current-texture-path-readiness';

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

function buildUnavailable(reason, extra = {}) {
  return {
    mode: WEBGPU_VIEWER_CANVAS_CURRENT_TEXTURE_PATH_MODE,
    status: 'unavailable',
    reason,
    source: 'viewerCanvasState',
    viewerCanvasCurrentTexturePathImplemented: true,
    viewerCanvasCurrentTexturePathReady: false,
    viewerCanvasContextConfigured: false,
    viewerCanvasCurrentTextureAcquired: false,
    viewerCanvasPresentationImplemented: false,
    productionDisplayConnectionImplemented: false,
    displayConnectionAllowed: false,
    webgl2HybridRenderingAllowed: false,
    anyMismatch: true,
    mismatchClassification: 'viewerCanvasCurrentTexturePathUnavailable',
    validationSummary: {
      firstValidationFailures: [{ stage: 'input', reason }]
    },
    ...extra
  };
}

function buildFailures({
  deviceAvailable,
  viewerCanvasProvided,
  exclusiveBackendModeRequested,
  allowViewerCanvasPresentation,
  webgl2FrameLifecycleSuppressed,
  viewerCanvasWebgl2Active,
  webgpuCanvasContextAvailable,
  currentTextureAcquired
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
      reason: 'WebGL2 frame lifecycle was not suppressed before viewer canvas WebGPU ownership'
    });
  }
  if (viewerCanvasWebgl2Active) {
    failures.push({
      stage: 'canvas-ownership',
      reason: 'viewer canvas is already owned by WebGL2'
    });
  }
  if (!webgpuCanvasContextAvailable) {
    failures.push({
      stage: 'webgpu-canvas-context',
      reason: 'viewer canvas WebGPU context is unavailable'
    });
  }
  if (!currentTextureAcquired) {
    failures.push({
      stage: 'current-texture',
      reason: 'viewer canvas currentTexture was not acquired'
    });
  }
  return failures;
}

export function canProbeViewerCanvasCurrentTexturePath({
  requestedBackendMode = null,
  allowViewerCanvasPresentation = false,
  contextMode = 'not-provided',
  webgl2FrameLifecycleSuppressed = false
} = {}) {
  return (
    normalizeWebGpuBackendMode(requestedBackendMode) ===
      WEBGPU_BACKEND_MODE_VALUES.WEBGPU_EXCLUSIVE &&
    allowViewerCanvasPresentation === true &&
    contextMode !== 'webgl2-active' &&
    webgl2FrameLifecycleSuppressed === true
  );
}

export async function buildWebGpuViewerCanvasCurrentTexturePathReadiness({
  device,
  viewerCanvasState = null,
  canvasContextMutationAllowed = true
} = {}) {
  const startMs = nowMs();
  if (!device) return buildUnavailable('webgpu-device-unavailable');

  const canvas = viewerCanvasState?.canvas ?? null;
  const requestedBackendMode = normalizeWebGpuBackendMode(
    viewerCanvasState?.requestedBackendMode
  );
  const contextMode = viewerCanvasState?.contextMode ?? 'not-provided';
  const viewerCanvasProvided =
    viewerCanvasState?.provided === true && !!canvas;
  const allowViewerCanvasPresentation =
    viewerCanvasState?.allowViewerCanvasPresentation === true;
  const webgl2FrameLifecycleSuppressed =
    viewerCanvasState?.webgl2FrameLifecycleSuppressed === true;
  const exclusiveBackendModeRequested =
    requestedBackendMode === WEBGPU_BACKEND_MODE_VALUES.WEBGPU_EXCLUSIVE;
  const viewerCanvasWebgl2Active = contextMode === 'webgl2-active';
  const probeAllowed = canProbeViewerCanvasCurrentTexturePath({
    requestedBackendMode,
    allowViewerCanvasPresentation,
    contextMode,
    webgl2FrameLifecycleSuppressed
  });

  if (!viewerCanvasProvided) {
    return buildUnavailable('viewer-canvas-unavailable', {
      requestedBackendMode,
      validationSummary: {
        viewerCanvasProvided: false,
        firstValidationFailures: [
          { stage: 'viewer-canvas', reason: 'viewer canvas was not provided' }
        ]
      }
    });
  }

  let context = null;
  let contextError = null;
  let currentTexture = null;
  let currentTextureError = null;
  const preferredFormat =
    typeof navigator !== 'undefined' && navigator.gpu?.getPreferredCanvasFormat
      ? navigator.gpu.getPreferredCanvasFormat()
      : 'rgba8unorm';
  const textureFormat = normalizeCanvasFormat(preferredFormat);
  const usage =
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.COPY_SRC |
    GPUTextureUsage.COPY_DST;
  const width = Math.max(1, Math.round(canvas.width ?? canvas.clientWidth ?? 1));
  const height = Math.max(1, Math.round(canvas.height ?? canvas.clientHeight ?? 1));

  if (canvasContextMutationAllowed !== true) {
    return {
      mode: WEBGPU_VIEWER_CANVAS_CURRENT_TEXTURE_PATH_MODE,
      status: 'ok',
      reason: 'live-viewer-canvas-owned-by-production-presentation',
      source: 'viewerCanvasState + production diagnostic ownership isolation',
      viewerCanvasCurrentTexturePathImplemented: true,
      viewerCanvasCurrentTexturePathReady: false,
      viewerCanvasContextConfigured: false,
      viewerCanvasCurrentTextureAcquired: false,
      viewerCanvasContextMutationSuppressed: true,
      viewerCanvasPresentationImplemented: false,
      productionDisplayConnectionImplemented: false,
      displayConnectionAllowed: false,
      webgl2HybridRenderingAllowed: false,
      requestedBackendMode,
      canvasContextKind: null,
      textureFormat,
      requestedPreferredFormat: preferredFormat,
      usage: ['render_attachment', 'copy_src', 'copy_dst'],
      outputExtent: { canvasWidth: width, canvasHeight: height },
      viewerCanvasOwnershipContract: {
        viewerCanvasProvided,
        contextMode,
        allowViewerCanvasPresentation,
        requestedOwner: exclusiveBackendModeRequested ? 'webgpu' : 'webgl2-fallback',
        currentOwner: 'production-presentation-owner',
        webgl2FrameLifecycleSuppressed,
        requiresExclusiveBackendMode: true,
        requiresPresentationGuard: true,
        requiresNoActiveWebGL2Context: true,
        handoffPolicy:
          'production diagnostic does not configure or acquire the live viewer canvas currentTexture'
      },
      currentTextureContract: {
        adapterLineage:
          'production diagnostic currentTexture access is isolated from the live viewer canvas',
        readinessDefinition:
          'production presentation owner retains context and currentTexture ownership',
        firstPresentPolicy:
          'diagnostic path does not configure, acquire, encode, or submit live canvas work',
        resizePolicy:
          'production presentation validates device, format, and extent'
      },
      cameraProjectionContract: {
        fixedReferenceCapture:
          'diagnostic ownership isolation does not change fixed-reference camera/projection data',
        interactiveViewer:
          'Three.js camera and OrbitControls remain input adapters',
        controlsImpact: 'no interactive camera behavior is changed'
      },
      shPolicy: {
        requiredForThisCurrentTexturePath: false,
        status: 'deferred',
        fallbackColorSource: null,
        displayImpact: 'diagnostic isolation does not change production color output'
      },
      anyMismatch: false,
      mismatchClassification: 'none',
      validationSummary: {
        deviceAvailable: true,
        viewerCanvasProvided,
        exclusiveBackendModeRequested,
        viewerCanvasPresentationGuardEnabled: allowViewerCanvasPresentation,
        viewerCanvasWebgl2Active,
        webgl2FrameLifecycleSuppressed,
        probeAllowed: false,
        viewerCanvasContextMutationSuppressed: true,
        webgpuCanvasContextAvailable: null,
        viewerCanvasContextConfigured: false,
        currentTextureAvailable: false,
        currentTextureExtentValid: null,
        viewerCanvasCurrentTexturePathReady: false,
        webgl2HybridRenderingPrevented: !viewerCanvasWebgl2Active,
        displayConnectionPrevented: true,
        cameraProjectionContractUnchanged: true,
        firstValidationFailures: []
      },
      blockers: [],
      nextBackendPrototypeStep:
        'reuse the production presentation owner without diagnostic canvas mutation',
      timing: {
        viewerCanvasCurrentTexturePathMs: nowMs() - startMs
      }
    };
  }

  if (probeAllowed) {
    try {
      context = canvas.getContext?.('webgpu') ?? null;
      if (context) {
        context.configure({
          device,
          format: textureFormat,
          usage,
          alphaMode: 'premultiplied'
        });
        currentTexture = context.getCurrentTexture();
      }
    } catch (error) {
      if (!context) {
        contextError = error;
      } else {
        currentTextureError = error;
      }
    }
  }

  const webgpuCanvasContextAvailable = !!context;
  const currentTextureAcquired = !!currentTexture;
  const currentTextureExtentValid =
    currentTextureAcquired && width > 0 && height > 0;
  const viewerCanvasCurrentTexturePathReady =
    probeAllowed &&
    webgpuCanvasContextAvailable &&
    currentTextureAcquired &&
    currentTextureExtentValid;
  const firstValidationFailures = buildFailures({
    deviceAvailable: true,
    viewerCanvasProvided,
    exclusiveBackendModeRequested,
    allowViewerCanvasPresentation,
    webgl2FrameLifecycleSuppressed,
    viewerCanvasWebgl2Active,
    webgpuCanvasContextAvailable,
    currentTextureAcquired
  });
  if (!probeAllowed) {
    firstValidationFailures.push({
      stage: 'exclusive-guard',
      reason:
        'viewer canvas currentTexture path is probed only in guarded webgpu-exclusive mode after WebGL2 frame suppression'
    });
  }
  if (contextError) {
    firstValidationFailures.push({
      stage: 'webgpu-canvas-context',
      reason: contextError.message ?? String(contextError)
    });
  }
  if (currentTextureError) {
    firstValidationFailures.push({
      stage: 'current-texture',
      reason: currentTextureError.message ?? String(currentTextureError)
    });
  }

  return {
    mode: WEBGPU_VIEWER_CANVAS_CURRENT_TEXTURE_PATH_MODE,
    status: 'ok',
    source: 'viewerCanvasState + Step42 currentTexture adapter contract',
    expectedSource:
      'Step42 proves getCurrentTexture adapter mechanics on a detached WebGPU canvas',
    actualSource:
      'Step45 configures the real viewer canvas WebGPU context only under guarded webgpu-exclusive ownership and acquires currentTexture readiness',
    viewerCanvasCurrentTexturePathImplemented: true,
    viewerCanvasCurrentTexturePathReady,
    viewerCanvasContextConfigured: webgpuCanvasContextAvailable,
    viewerCanvasCurrentTextureAcquired: currentTextureAcquired,
    viewerCanvasPresentationImplemented: false,
    productionDisplayConnectionImplemented: false,
    boundedFirstPresentImplemented: false,
    displayConnectionAllowed: false,
    webgl2HybridRenderingAllowed: false,
    requestedBackendMode,
    canvasContextKind: webgpuCanvasContextAvailable ? 'viewer-canvas-webgpu' : null,
    textureFormat,
    requestedPreferredFormat: preferredFormat,
    usage: ['render_attachment', 'copy_src', 'copy_dst'],
    outputExtent: { canvasWidth: width, canvasHeight: height },
    viewerCanvasOwnershipContract: {
      viewerCanvasProvided,
      contextMode,
      allowViewerCanvasPresentation,
      requestedOwner: exclusiveBackendModeRequested ? 'webgpu' : 'webgl2-fallback',
      currentOwner: viewerCanvasCurrentTexturePathReady
        ? 'webgpu-exclusive-current-texture-ready'
        : viewerCanvasWebgl2Active
          ? 'webgl2'
          : 'none-or-unknown',
      webgl2FrameLifecycleSuppressed,
      requiresExclusiveBackendMode: true,
      requiresPresentationGuard: true,
      requiresNoActiveWebGL2Context: true,
      handoffPolicy:
        'viewer canvas currentTexture is acquired only after exclusive WebGPU mode owns the frame lifecycle'
    },
    currentTextureContract: {
      adapterLineage:
        'reuses the Step42 currentTexture adapter contract, changing only the target canvas from detached to viewer canvas under exclusive ownership',
      readinessDefinition:
        'WebGPU context configured and getCurrentTexture succeeds without WebGL2 ownership or production display connection',
      firstPresentPolicy:
        'do not encode or submit viewer canvas drawing work in Step45; first-present remains separately guarded',
      resizePolicy:
        'read viewer canvas width/height at capture time; future normal backend should reconfigure on resize before each present'
    },
    cameraProjectionContract: {
      fixedReferenceCapture:
        'Step45 changes only canvas ownership/currentTexture readiness and leaves fixed-reference camera/projection data comparable',
      interactiveViewer:
        'Three.js camera and OrbitControls remain camera input adapters; no mouse/control implementation changes',
      controlsImpact:
        'no interactive camera behavior is implemented or changed in Step45'
    },
    shPolicy: {
      requiredForThisCurrentTexturePath: false,
      status: 'deferred',
      fallbackColorSource: 'reference-assisted colorAlpha.rgb payload',
      displayImpact:
        'currentTexture readiness can be validated before WGSL SH/color parity; visual parity still requires SH/color evaluation or explicit reference-assisted display mode'
    },
    anyMismatch: !viewerCanvasCurrentTexturePathReady,
    mismatchClassification: viewerCanvasCurrentTexturePathReady
      ? 'none'
      : 'viewerCanvasCurrentTexturePathNotReady',
    validationSummary: {
      deviceAvailable: true,
      viewerCanvasProvided,
      exclusiveBackendModeRequested,
      viewerCanvasPresentationGuardEnabled: allowViewerCanvasPresentation,
      viewerCanvasWebgl2Active,
      webgl2FrameLifecycleSuppressed,
      probeAllowed,
      webgpuCanvasContextAvailable,
      viewerCanvasContextConfigured: webgpuCanvasContextAvailable,
      currentTextureAvailable: currentTextureAcquired,
      currentTextureExtentValid,
      viewerCanvasCurrentTexturePathReady,
      webgl2HybridRenderingPrevented: !viewerCanvasWebgl2Active,
      displayConnectionPrevented: true,
      cameraProjectionContractUnchanged: true,
      firstValidationFailures
    },
    blockers: viewerCanvasCurrentTexturePathReady
      ? [
          {
            stage: 'bounded-viewer-canvas-first-present',
            reason:
              'viewer canvas currentTexture path is ready; bounded first-present remains intentionally guarded for the next display experiment'
          },
          {
            stage: 'sh-color-evaluation',
            reason: 'WGSL SH/color evaluation parity remains deferred'
          }
        ]
      : [
          {
            stage: 'viewer-canvas-current-texture-path',
            reason:
              'viewer canvas currentTexture path requires exclusive backend mode, presentation guard, WebGL2 frame suppression, WebGPU context, and currentTexture acquisition'
          }
        ],
    nextBackendPrototypeStep: viewerCanvasCurrentTexturePathReady
      ? 'perform a bounded viewer-canvas first-present with an explicit reference-assisted display guard'
      : 'resolve viewer canvas WebGPU context/currentTexture readiness blockers before first-present',
    timing: {
      viewerCanvasCurrentTexturePathMs: nowMs() - startMs
    }
  };
}
