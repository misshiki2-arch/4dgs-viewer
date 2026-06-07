import {
  WEBGPU_BACKEND_MODE_VALUES,
  normalizeWebGpuBackendMode
} from './webgpu_exclusive_canvas_handoff.js';

export const WEBGPU_EXCLUSIVE_FRAME_LIFECYCLE_SWITCH_MODE =
  'webgpu-exclusive-frame-lifecycle-switch';

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function buildFailures({
  adapterReady,
  viewerCanvasProvided,
  exclusiveBackendModeRequested,
  allowViewerCanvasPresentation,
  viewerCanvasWebgl2Active
}) {
  const failures = [];
  if (!adapterReady) {
    failures.push({
      stage: 'canvas-presentation-adapter',
      reason: 'Step42 canvas currentTexture adapter is not ready'
    });
  }
  if (!viewerCanvasProvided) {
    failures.push({
      stage: 'viewer-canvas',
      reason: 'viewer canvas state was not provided'
    });
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
  if (viewerCanvasWebgl2Active) {
    failures.push({
      stage: 'canvas-ownership',
      reason: 'viewer canvas is already owned by WebGL2'
    });
  }
  return failures;
}

export function shouldUseWebGpuExclusiveFrameLifecycle({
  requestedBackendMode = null,
  allowViewerCanvasPresentation = false
} = {}) {
  return (
    normalizeWebGpuBackendMode(requestedBackendMode) ===
      WEBGPU_BACKEND_MODE_VALUES.WEBGPU_EXCLUSIVE &&
    allowViewerCanvasPresentation === true
  );
}

export function buildWebGpuExclusiveFrameLifecycleSwitch({
  webgpuCanvasPresentationAdapterDryRunComparison,
  webgpuExclusiveCanvasHandoffReadiness,
  viewerCanvasState = null
} = {}) {
  const startMs = nowMs();
  const requestedBackendMode = normalizeWebGpuBackendMode(
    viewerCanvasState?.requestedBackendMode
  );
  const contextMode = viewerCanvasState?.contextMode ?? 'not-provided';
  const viewerCanvasProvided = viewerCanvasState?.provided === true;
  const allowViewerCanvasPresentation =
    viewerCanvasState?.allowViewerCanvasPresentation === true;
  const webgl2FrameLifecycleSuppressed =
    viewerCanvasState?.webgl2FrameLifecycleSuppressed === true;
  const exclusiveBackendModeRequested =
    requestedBackendMode === WEBGPU_BACKEND_MODE_VALUES.WEBGPU_EXCLUSIVE;
  const viewerCanvasWebgl2Active = contextMode === 'webgl2-active';
  const adapterReady =
    webgpuCanvasPresentationAdapterDryRunComparison?.canvasPresentationProbeSucceeded ===
      true ||
    webgpuExclusiveCanvasHandoffReadiness?.canvasPresentationAdapterValid === true;
  const viewerCanvasLifecycleSwitchRequested =
    exclusiveBackendModeRequested && allowViewerCanvasPresentation;
  const viewerCanvasLifecycleSwitched =
    adapterReady &&
    viewerCanvasProvided &&
    viewerCanvasLifecycleSwitchRequested &&
    webgl2FrameLifecycleSuppressed &&
    !viewerCanvasWebgl2Active;
  const viewerCanvasCurrentTexturePathReady =
    viewerCanvasLifecycleSwitched &&
    webgpuExclusiveCanvasHandoffReadiness?.viewerCanvasHandoffAllowed === true;
  const firstValidationFailures = buildFailures({
    adapterReady,
    viewerCanvasProvided,
    exclusiveBackendModeRequested,
    allowViewerCanvasPresentation,
    viewerCanvasWebgl2Active
  });
  if (viewerCanvasLifecycleSwitchRequested && !webgl2FrameLifecycleSuppressed) {
    firstValidationFailures.push({
      stage: 'webgl2-frame-lifecycle',
      reason:
        'exclusive WebGPU lifecycle was requested but WebGL2 debug frame suppression did not run'
    });
  }

  return {
    mode: WEBGPU_EXCLUSIVE_FRAME_LIFECYCLE_SWITCH_MODE,
    status: 'ok',
    source: 'webgpuExclusiveCanvasHandoffReadiness',
    expectedSource:
      'Step43 readiness defines when the Step42 currentTexture adapter may be routed to the viewer canvas',
    actualSource:
      'Step44 evaluates the viewer frame lifecycle switch before WebGL2 owns the canvas',
    exclusiveFrameLifecycleSwitchImplemented: true,
    viewerCanvasLifecycleSwitchRequested,
    viewerCanvasLifecycleSwitched,
    viewerCanvasCurrentTexturePathReady,
    viewerCanvasPresentationImplemented: false,
    productionDisplayConnectionImplemented: false,
    framebufferImplemented: false,
    canvasPresentationImplemented: false,
    displayConnectionAllowed: false,
    webgl2HybridRenderingAllowed: false,
    webgpuNormalBackendCandidate: true,
    requestedBackendMode,
    supportedBackendModes:
      webgpuExclusiveCanvasHandoffReadiness?.supportedBackendModes ?? [
        ...Object.values(WEBGPU_BACKEND_MODE_VALUES)
      ],
    frameLifecycleContract: {
      webgpuExclusive:
        'skip WebGL2 frame initialization, let WebGPU own the viewer canvas lifecycle, then route currentTexture through the Step42 adapter',
      webgl2Fallback:
        'keep the existing WebGL2 renderer as fallback, validation, and regression oracle',
      webgpuDryRun:
        'run WebGPU backend validation without taking viewer canvas ownership',
      validationOracle:
        'do not mix WebGPU presentation and WebGL2 validation in the same viewer frame'
    },
    viewerCanvasOwnershipContract: {
      viewerCanvasProvided,
      contextMode,
      allowViewerCanvasPresentation,
      requestedOwner: exclusiveBackendModeRequested ? 'webgpu' : 'webgl2-fallback',
      currentOwner: viewerCanvasLifecycleSwitched
        ? 'webgpu-exclusive-pending'
        : viewerCanvasWebgl2Active
          ? 'webgl2'
          : 'none-or-unknown',
      webgl2FrameLifecycleSuppressed,
      requiresExclusiveBackendMode: true,
      requiresPresentationGuard: true,
      requiresNoActiveWebGL2Context: true,
      handoffPolicy:
        'only webgpu-exclusive mode with the presentation guard enabled may move the viewer canvas lifecycle to WebGPU'
    },
    cameraProjectionContract: {
      fixedReferenceCapture:
        'Step44 keeps fixed-reference dry-run camera/projection data comparable while switching only canvas lifecycle ownership',
      interactiveViewer:
        'future WebGPU normal backend should consume per-frame projection uniforms from the existing camera input adapter',
      controlsImpact:
        'no mouse, OrbitControls, or camera interaction implementation changes in Step44'
    },
    shPolicy: {
      requiredForThisLifecycleSwitch: false,
      status: 'deferred',
      fallbackColorSource: 'reference-assisted colorAlpha.rgb payload',
      displayImpact:
        'exclusive lifecycle plumbing can proceed with reference-assisted color; full visual parity still needs WGSL SH/color evaluation or an explicit reference-assisted display mode'
    },
    anyMismatch: false,
    mismatchClassification: 'none',
    validationSummary: {
      canvasPresentationAdapterValid: adapterReady,
      exclusiveBackendModeRequested,
      viewerCanvasPresentationGuardEnabled: allowViewerCanvasPresentation,
      viewerCanvasProvided,
      viewerCanvasWebgl2Active,
      webgl2FrameLifecycleSuppressed,
      viewerCanvasLifecycleSwitchRequested,
      viewerCanvasLifecycleSwitched,
      viewerCanvasCurrentTexturePathReady,
      webgl2HybridRenderingPrevented:
        !viewerCanvasLifecycleSwitched || !viewerCanvasWebgl2Active,
      cameraProjectionContractUnchanged: true,
      firstValidationFailures
    },
    blockers: viewerCanvasCurrentTexturePathReady
      ? [
          {
            stage: 'bounded-viewer-canvas-first-present',
            reason:
              'viewer canvas lifecycle can switch to WebGPU, but bounded first-present is intentionally left for the next guarded display experiment'
          }
        ]
      : [
          {
            stage: 'exclusive-frame-lifecycle',
            reason:
              'viewer canvas lifecycle switch requires webgpu-exclusive mode, presentation guard, ready currentTexture adapter, and no active WebGL2 context'
          }
        ],
    nextBackendPrototypeStep: viewerCanvasCurrentTexturePathReady
      ? 'perform a bounded viewer-canvas currentTexture first-present in webgpu-exclusive mode'
      : 'run with webgpuBackendMode=webgpu-exclusive and webgpuAllowViewerCanvasPresentation=true so the viewer canvas lifecycle can switch before WebGL2 initialization',
    timing: {
      exclusiveFrameLifecycleSwitchMs: nowMs() - startMs
    }
  };
}
