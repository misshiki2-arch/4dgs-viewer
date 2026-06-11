import { shouldUseWebGpuExclusiveFrameLifecycle } from './webgpu_exclusive_frame_lifecycle_switch.js';

export const WEBGPU_BACKEND_VIEWER_LIFECYCLE_INTEGRATION_MODE =
  'webgpu-backend-viewer-lifecycle-integration-boundary';

export const WEBGPU_BACKEND_VIEWER_LIFECYCLE_INTEGRATION_CONTRACT_VERSION =
  'phase3-step58-backend-viewer-lifecycle-integration-boundary-v1';

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

export function shouldUseWebGpuBackendViewerLifecycleIntegration({
  requestedBackendMode = 'webgl2-fallback',
  allowViewerCanvasPresentation = false,
  enableViewerLoopHook = false
} = {}) {
  return (
    enableViewerLoopHook === true &&
    shouldUseWebGpuExclusiveFrameLifecycle({
      requestedBackendMode,
      allowViewerCanvasPresentation
    })
  );
}

function buildHookContract({
  hookAllowed,
  requestedBackendMode,
  allowViewerCanvasPresentation,
  enableViewerLoopHook,
  renderLifecycleStage
}) {
  return {
    contractVersion: WEBGPU_BACKEND_VIEWER_LIFECYCLE_INTEGRATION_CONTRACT_VERSION,
    hookMode: 'guarded-viewer-render-lifecycle-hook',
    renderLifecycleStage,
    hookEntryPoint: 'viewer_app_gpu.renderCurrentFrame',
    callableFromExistingViewerLifecycle: true,
    explicitHookFlag: 'webgpuBackendViewerLoopHook',
    explicitHookEnabled: enableViewerLoopHook === true,
    requestedBackendMode,
    allowViewerCanvasPresentation: allowViewerCanvasPresentation === true,
    hookAllowed,
    requiresExclusiveBackendMode: true,
    requiresViewerCanvasPresentationGuard: true,
    productionLoopConnected: false
  };
}

function buildCameraSnapshotContract(cameraSnapshot) {
  return {
    provided: !!cameraSnapshot,
    source: cameraSnapshot ? 'viewer-lifecycle-camera-snapshot' : 'not-provided',
    cameraInputAdapterMode: 'threejs-camera-input-adapter',
    interactiveCameraImplemented: false,
    projectionContractMutableByHook: false
  };
}

function buildAdapterInvocationContract({ adapterResult, adapterInvocationSource }) {
  const adapterReady = adapterResult?.viewerLoopAdapterReady === true;
  return {
    adapterInvocationSource,
    adapterResultProvided: !!adapterResult,
    adapterReady,
    adapterMode: adapterResult?.mode ?? null,
    adapterStatus: adapterResult?.status ?? null,
    executedBackendFrameSubmissions:
      adapterResult?.executedBackendFrameSubmissions ?? null,
    repeatedSubmitCount: adapterResult?.repeatedSubmitCount ?? null,
    selectedSourceKind: adapterResult?.selectedSourceKind ?? null,
    colorPresentSampleCount: adapterResult?.colorPresentSampleCount ?? null
  };
}

function buildValidationSummary({
  hookContract,
  viewerCanvasState,
  cameraSnapshotContract,
  adapterInvocationContract
}) {
  const hookAllowed = hookContract.hookAllowed === true;
  const webgl2FrameLifecycleSuppressed =
    viewerCanvasState?.webgl2FrameLifecycleSuppressed === true;
  const viewerCanvasProvided = viewerCanvasState?.provided === true;
  const cameraSnapshotProvided = cameraSnapshotContract.provided === true;
  const adapterReady = adapterInvocationContract.adapterReady === true;
  const webgl2HybridRenderingPrevented =
    viewerCanvasState?.contextMode !== 'webgl2-active' &&
    webgl2FrameLifecycleSuppressed;
  const fallbackPolicyPreserved =
    adapterInvocationContract.adapterResultProvided === false ||
    adapterReady;
  const integrationBoundaryReady =
    hookAllowed &&
    viewerCanvasProvided &&
    webgl2HybridRenderingPrevented &&
    cameraSnapshotProvided &&
    fallbackPolicyPreserved;
  const firstValidationFailures = [];
  if (!hookAllowed) {
    firstValidationFailures.push({
      stage: 'viewer-lifecycle-hook-guard',
      reason:
        'viewer lifecycle hook requires webgpu-exclusive mode, viewer canvas presentation guard, and webgpuBackendViewerLoopHook=true'
    });
  }
  if (!viewerCanvasProvided) {
    firstValidationFailures.push({
      stage: 'viewer-canvas-state',
      reason: 'viewer lifecycle hook needs viewer canvas state from the caller'
    });
  }
  if (!webgl2HybridRenderingPrevented) {
    firstValidationFailures.push({
      stage: 'webgl2-lifecycle-suppression',
      reason: 'viewer lifecycle hook requires WebGL2 frame lifecycle suppression'
    });
  }
  if (!cameraSnapshotProvided) {
    firstValidationFailures.push({
      stage: 'camera-snapshot',
      reason: 'viewer lifecycle hook needs a camera snapshot boundary'
    });
  }
  if (!fallbackPolicyPreserved) {
    firstValidationFailures.push({
      stage: 'adapter-fallback-policy',
      reason: 'viewer loop adapter result was provided but not ready'
    });
  }
  return {
    integrationBoundaryReady,
    hookAllowed,
    viewerCanvasProvided,
    webgl2FrameLifecycleSuppressed,
    webgl2HybridRenderingPrevented,
    cameraSnapshotProvided,
    adapterReady,
    fallbackPolicyPreserved,
    firstValidationFailures
  };
}

export function buildWebGpuBackendViewerLifecycleIntegrationBoundary({
  requestedBackendMode = 'webgl2-fallback',
  allowViewerCanvasPresentation = false,
  enableViewerLoopHook = false,
  renderLifecycleStage = 'unknown',
  viewerCanvasState = null,
  cameraSnapshot = null,
  adapterResult = null,
  adapterInvocationSource = 'not-invoked'
} = {}) {
  const startMs = nowMs();
  const hookAllowed = shouldUseWebGpuBackendViewerLifecycleIntegration({
    requestedBackendMode,
    allowViewerCanvasPresentation,
    enableViewerLoopHook
  });
  const hookContract = buildHookContract({
    hookAllowed,
    requestedBackendMode,
    allowViewerCanvasPresentation,
    enableViewerLoopHook,
    renderLifecycleStage
  });
  const cameraSnapshotContract = buildCameraSnapshotContract(cameraSnapshot);
  const adapterInvocationContract = buildAdapterInvocationContract({
    adapterResult,
    adapterInvocationSource
  });
  const validationSummary = buildValidationSummary({
    hookContract,
    viewerCanvasState,
    cameraSnapshotContract,
    adapterInvocationContract
  });
  const integrationBoundaryReady =
    validationSummary.integrationBoundaryReady === true;
  return {
    mode: WEBGPU_BACKEND_VIEWER_LIFECYCLE_INTEGRATION_MODE,
    status: integrationBoundaryReady ? 'ok' : 'blocked',
    source:
      'Phase 3 Step58 guarded viewer render lifecycle integration boundary for the WebGPU backend viewer loop adapter',
    contractVersion: WEBGPU_BACKEND_VIEWER_LIFECYCLE_INTEGRATION_CONTRACT_VERSION,
    integrationBoundaryImplemented: true,
    integrationBoundaryReady,
    productionDisplayConnectionImplemented: false,
    displayConnectionAllowed: false,
    webgl2HybridRenderingAllowed: false,
    hookContract,
    cameraSnapshotContract,
    adapterInvocationContract,
    viewerCanvasOwnershipContract: {
      viewerCanvasProvided: viewerCanvasState?.provided === true,
      contextMode: viewerCanvasState?.contextMode ?? null,
      requestedBackendMode,
      allowViewerCanvasPresentation: allowViewerCanvasPresentation === true,
      webgl2FrameLifecycleSuppressed:
        viewerCanvasState?.webgl2FrameLifecycleSuppressed === true,
      requiresExclusiveBackendMode: true,
      requiresPresentationGuard: true,
      productionLoopConnected: false
    },
    selectedSourceKind: adapterResult?.selectedSourceKind ?? null,
    colorPresentSampleCount: adapterResult?.colorPresentSampleCount ?? null,
    executedBackendFrameSubmissions:
      adapterResult?.executedBackendFrameSubmissions ?? null,
    repeatedSubmitCount: adapterResult?.repeatedSubmitCount ?? null,
    fallbackPolicy: adapterResult?.fallbackPolicy ?? {},
    validationSummary,
    firstValidationFailures: validationSummary.firstValidationFailures,
    blockers: integrationBoundaryReady
      ? [
          {
            stage: 'production-frame-scheduler',
            reason:
              'viewer lifecycle hook boundary is ready; production requestAnimationFrame scheduling remains intentionally disconnected'
          },
          {
            stage: 'interactive-camera',
            reason:
              'Three.js and OrbitControls remain camera input adapters; interactive camera implementation is outside Step58'
          },
          {
            stage: 'sh-color-evaluation',
            reason: 'WGSL SH/color evaluation parity remains deferred'
          }
        ]
      : [
          {
            stage: 'viewer-lifecycle-integration-boundary',
            reason:
              'integration boundary requires the exclusive guard, viewer canvas state, camera snapshot, and preserved adapter policy'
          }
        ],
    nextBackendPrototypeStep: integrationBoundaryReady
      ? 'wire the guarded viewer lifecycle hook to a scheduler-owned backend frame invocation without enabling the production loop by default'
      : 'restore guarded viewer lifecycle integration readiness before scheduler wiring',
    timing: {
      webgpuBackendViewerLifecycleIntegrationBoundaryMs: nowMs() - startMs
    }
  };
}
