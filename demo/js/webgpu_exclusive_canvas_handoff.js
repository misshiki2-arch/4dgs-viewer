export const WEBGPU_EXCLUSIVE_CANVAS_HANDOFF_MODE =
  'webgpu-exclusive-canvas-handoff-readiness';

export const WEBGPU_BACKEND_MODE_VALUES = Object.freeze({
  WEBGL2_FALLBACK: 'webgl2-fallback',
  WEBGPU_DRY_RUN: 'webgpu-dry-run',
  WEBGPU_EXCLUSIVE: 'webgpu-exclusive'
});

export const WEBGPU_BACKEND_MODE_SET = new Set(
  Object.values(WEBGPU_BACKEND_MODE_VALUES)
);

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

export function normalizeWebGpuBackendMode(value) {
  const mode = typeof value === 'string' ? value.trim() : '';
  return WEBGPU_BACKEND_MODE_SET.has(mode)
    ? mode
    : WEBGPU_BACKEND_MODE_VALUES.WEBGL2_FALLBACK;
}

function buildUnavailable(reason) {
  return {
    mode: WEBGPU_EXCLUSIVE_CANVAS_HANDOFF_MODE,
    status: 'unavailable',
    reason,
    source: 'webgpuCanvasPresentationAdapterDryRunComparison',
    normalBackendBoundaryImplemented: true,
    exclusiveBackendModeRequested: false,
    exclusiveBackendModeReady: false,
    viewerCanvasHandoffAllowed: false,
    viewerCanvasPresentationImplemented: false,
    displayConnectionAllowed: false,
    webgl2HybridRenderingAllowed: false,
    anyMismatch: true,
    mismatchClassification: 'exclusiveCanvasHandoffUnavailable',
    validationSummary: {
      canvasPresentationAdapterValid: false,
      firstValidationFailures: [{ stage: 'input', reason }]
    }
  };
}

export function buildWebGpuExclusiveCanvasHandoffReadiness({
  webgpuCanvasPresentationAdapterDryRunComparison,
  viewerCanvasState = null
}) {
  const startMs = nowMs();
  if (
    !webgpuCanvasPresentationAdapterDryRunComparison ||
    webgpuCanvasPresentationAdapterDryRunComparison.status !== 'ok'
  ) {
    return buildUnavailable(
      webgpuCanvasPresentationAdapterDryRunComparison?.reason ??
        webgpuCanvasPresentationAdapterDryRunComparison?.status ??
        'canvas-presentation-adapter-dry-run-unavailable'
    );
  }

  const requestedBackendMode = normalizeWebGpuBackendMode(
    viewerCanvasState?.requestedBackendMode
  );
  const contextMode = viewerCanvasState?.contextMode ?? 'not-provided';
  const viewerCanvasProvided = viewerCanvasState?.provided === true;
  const allowViewerCanvasPresentation =
    viewerCanvasState?.allowViewerCanvasPresentation === true;
  const exclusiveBackendModeRequested =
    requestedBackendMode === WEBGPU_BACKEND_MODE_VALUES.WEBGPU_EXCLUSIVE;
  const viewerCanvasWebgl2Active = contextMode === 'webgl2-active';
  const adapterReady =
    webgpuCanvasPresentationAdapterDryRunComparison.canvasPresentationProbeSucceeded === true;
  const exclusiveBackendModeReady =
    adapterReady &&
    viewerCanvasProvided &&
    exclusiveBackendModeRequested &&
    allowViewerCanvasPresentation &&
    !viewerCanvasWebgl2Active;
  const viewerCanvasHandoffAllowed = exclusiveBackendModeReady;

  const firstValidationFailures = [];
  if (!adapterReady) {
    firstValidationFailures.push({
      stage: 'canvas-presentation-adapter',
      reason: 'Step42 canvas presentation adapter is not ready'
    });
  }
  if (!viewerCanvasProvided) {
    firstValidationFailures.push({
      stage: 'viewer-canvas',
      reason: 'viewer canvas state was not provided'
    });
  }
  if (!exclusiveBackendModeRequested) {
    firstValidationFailures.push({
      stage: 'backend-mode',
      reason: 'webgpu-exclusive backend mode was not requested'
    });
  }
  if (!allowViewerCanvasPresentation) {
    firstValidationFailures.push({
      stage: 'presentation-guard',
      reason: 'viewer canvas presentation guard is not enabled'
    });
  }
  if (viewerCanvasWebgl2Active) {
    firstValidationFailures.push({
      stage: 'canvas-ownership',
      reason: 'viewer canvas is currently owned by WebGL2'
    });
  }

  return {
    mode: WEBGPU_EXCLUSIVE_CANVAS_HANDOFF_MODE,
    status: 'ok',
    source: 'webgpuCanvasPresentationAdapterDryRunComparison',
    expectedSource:
      'Step42 detached canvas currentTexture adapter establishes the presentation lifecycle',
    actualSource:
      'Step43 evaluates whether the same adapter can be routed to the viewer canvas under an exclusive WebGPU backend guard',
    normalBackendBoundaryImplemented: true,
    exclusiveBackendModeRequested,
    exclusiveBackendModeReady,
    viewerCanvasHandoffAllowed,
    viewerCanvasPresentationImplemented: false,
    productionDisplayConnectionImplemented: false,
    displayConnectionAllowed: false,
    webgl2HybridRenderingAllowed: false,
    webgpuNormalBackendCandidate: true,
    requestedBackendMode,
    supportedBackendModes: [...WEBGPU_BACKEND_MODE_SET],
    backendRoleContract: {
      webgpuNormalBackend:
        'exclusive owner of WebGPU canvas presentation when requested and validated',
      webgl2Fallback:
        'fallback, validation, and regression oracle; not mixed into WebGPU presentation frames',
      cameraInputAdapter:
        'Three.js camera and OrbitControls provide camera/projection input only'
    },
    viewerCanvasOwnershipContract: {
      viewerCanvasProvided,
      contextMode,
      allowViewerCanvasPresentation,
      requiresExclusiveBackendMode: true,
      requiresViewerCanvasWithoutActiveWebGL2Context: true,
      currentOwner: viewerCanvasWebgl2Active ? 'webgl2' : 'none-or-unknown',
      requestedOwner: exclusiveBackendModeRequested ? 'webgpu' : 'webgl2-fallback',
      handoffPolicy:
        'route Step42 currentTexture adapter to the viewer canvas only when WebGPU exclusive mode owns the canvas lifecycle',
      blockedReason: viewerCanvasHandoffAllowed
        ? 'none'
        : firstValidationFailures.map((failure) => failure.reason).join('; ')
    },
    canvasPresentationAdapterReuse: {
      canReuseStep42Adapter: adapterReady,
      reusableTextureFormat:
        webgpuCanvasPresentationAdapterDryRunComparison.canvasPresentationContract
          ?.textureFormat ?? null,
      reusableUsage:
        webgpuCanvasPresentationAdapterDryRunComparison.canvasPresentationContract
          ?.usage ?? [],
      sampleWritePolicy:
        webgpuCanvasPresentationAdapterDryRunComparison.canvasPresentationContract
          ?.sampleWritePolicy ?? null,
      targetSwitch:
        'replace detached canvas currentTexture with viewer canvas currentTexture after exclusive backend ownership is active'
    },
    cameraProjectionContract: {
      fixedReferenceCapture:
        'Step43 does not mutate camera/projection data and keeps fixed-reference dry-run samples comparable',
      interactiveViewer:
        'future exclusive WebGPU backend consumes per-frame projection uniforms produced by the existing camera input adapter',
      controlsImpact:
        'no mouse, OrbitControls, or camera interaction implementation changes in Step43'
    },
    shPolicy: {
      requiredForThisHandoff: false,
      status: 'deferred',
      fallbackColorSource: 'reference-assisted colorAlpha.rgb payload',
      displayImpact:
        'exclusive first-display can remain reference-assisted for plumbing validation; full visual parity still requires WGSL SH/color evaluation or an explicit reference-assisted display mode'
    },
    anyMismatch: false,
    mismatchClassification: 'none',
    validationSummary: {
      canvasPresentationAdapterValid: adapterReady,
      detachedCanvasCurrentTextureValidated:
        webgpuCanvasPresentationAdapterDryRunComparison.contextGetCurrentTextureUsed === true &&
        webgpuCanvasPresentationAdapterDryRunComparison.currentTextureReadbackCompared === true,
      viewerCanvasProvided,
      exclusiveBackendModeRequested,
      viewerCanvasPresentationGuardEnabled: allowViewerCanvasPresentation,
      viewerCanvasWebgl2Active,
      exclusiveBackendModeReady,
      viewerCanvasHandoffAllowed,
      webgl2HybridRenderingPrevented: !viewerCanvasHandoffAllowed || !viewerCanvasWebgl2Active,
      cameraProjectionContractUnchanged: true,
      firstValidationFailures
    },
    blockers: viewerCanvasHandoffAllowed
      ? []
      : [
          {
            stage: 'viewer-canvas-exclusive-ownership',
            reason:
              'viewer canvas handoff remains blocked until webgpu-exclusive mode is requested and WebGL2 no longer owns the canvas'
          }
        ],
    nextBackendPrototypeStep: viewerCanvasHandoffAllowed
      ? 'perform a bounded viewer-canvas currentTexture first-present using the Step42 adapter'
      : 'add an exclusive WebGPU backend frame lifecycle switch that initializes the viewer canvas without a WebGL2 context, then reuse the Step42 currentTexture adapter',
    timing: {
      exclusiveCanvasHandoffReadinessMs: nowMs() - startMs
    }
  };
}
