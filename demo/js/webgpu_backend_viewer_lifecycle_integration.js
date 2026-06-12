import { shouldUseWebGpuExclusiveFrameLifecycle } from './webgpu_exclusive_frame_lifecycle_switch.js';

export const WEBGPU_BACKEND_VIEWER_LIFECYCLE_INTEGRATION_MODE =
  'webgpu-backend-viewer-lifecycle-integration-boundary';

export const WEBGPU_BACKEND_VIEWER_LIFECYCLE_INTEGRATION_CONTRACT_VERSION =
  'phase3-step58-backend-viewer-lifecycle-integration-boundary-v1';

export const WEBGPU_BACKEND_VIEWER_LIFECYCLE_CONTROLLED_EXECUTION_MODE =
  'webgpu-backend-viewer-lifecycle-controlled-execution';

export const WEBGPU_BACKEND_VIEWER_LIFECYCLE_CONTROLLED_EXECUTION_CONTRACT_VERSION =
  'phase3-step59-backend-viewer-lifecycle-controlled-execution-v1';

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

function buildControlledExecutionInvocationContract({
  invocationRequested,
  invocationSource,
  integrationBoundary,
  adapterResult
}) {
  const adapterReady = adapterResult?.viewerLoopAdapterReady === true;
  const controlledRepeatedExecution =
    adapterResult?.webgpuBackendFrameControlledRepeatedExecution ?? {};
  return {
    contractVersion:
      WEBGPU_BACKEND_VIEWER_LIFECYCLE_CONTROLLED_EXECUTION_CONTRACT_VERSION,
    invocationMode: 'guarded-render-current-frame-controlled-webgpu-backend-execution',
    invocationSource,
    invocationRequested: invocationRequested === true,
    invocationCount: adapterResult ? 1 : 0,
    integrationBoundaryReady: integrationBoundary?.integrationBoundaryReady === true,
    adapterResultProvided: !!adapterResult,
    adapterReady,
    requestedFrameCount:
      adapterResult?.requestedFrameCount ??
      controlledRepeatedExecution?.requestedFrameCount ??
      null,
    executedBackendFrameSubmissions:
      adapterResult?.executedBackendFrameSubmissions ??
      controlledRepeatedExecution?.executedBackendFrameSubmissions ??
      null,
    repeatedSubmitCount:
      adapterResult?.repeatedSubmitCount ??
      controlledRepeatedExecution?.repeatedSubmitCount ??
      null,
    selectedSourceKind: adapterResult?.selectedSourceKind ?? null,
    selectionMode:
      adapterResult?.selectionMode ??
      controlledRepeatedExecution?.selectionMode ??
      adapterResult?.initialBackendFramePrototype?.webgpuViewerCanvasBoundedColorPresent
        ?.selectionMode ??
      null,
    colorPresentSampleCount: adapterResult?.colorPresentSampleCount ?? null
  };
}

function buildControlledExecutionValidationSummary({
  invocationContract,
  integrationBoundary,
  webgl2FrameLifecycleSuppressed,
  cameraSnapshot
}) {
  const invocationRequested = invocationContract.invocationRequested === true;
  const integrationBoundaryReady =
    integrationBoundary?.integrationBoundaryReady === true;
  const adapterReady = invocationContract.adapterReady === true;
  const submittedFrameCount = invocationContract.executedBackendFrameSubmissions ?? 0;
  const repeatedSubmitCount = invocationContract.repeatedSubmitCount ?? 0;
  const webgl2HybridRenderingPrevented = webgl2FrameLifecycleSuppressed === true;
  const cameraSnapshotProvided = !!cameraSnapshot;
  const fallbackPolicyPreserved =
    integrationBoundary?.fallbackPolicy?.selectorSelectedSamplesUsed !== false;
  const controlledExecutionReady =
    invocationRequested &&
    integrationBoundaryReady &&
    adapterReady &&
    submittedFrameCount > 0 &&
    repeatedSubmitCount > 0 &&
    webgl2HybridRenderingPrevented &&
    cameraSnapshotProvided &&
    fallbackPolicyPreserved;
  const firstValidationFailures = [];
  if (!invocationRequested) {
    firstValidationFailures.push({
      stage: 'controlled-execution-invocation',
      reason: 'controlled WebGPU backend execution was not requested by the guarded viewer hook'
    });
  }
  if (!integrationBoundaryReady) {
    firstValidationFailures.push({
      stage: 'viewer-lifecycle-integration-boundary',
      reason: 'controlled execution requires a ready Step58 integration boundary'
    });
  }
  if (!adapterReady) {
    firstValidationFailures.push({
      stage: 'viewer-loop-adapter',
      reason: 'controlled execution requires a ready WebGPU backend viewer loop adapter result'
    });
  }
  if (submittedFrameCount <= 0 || repeatedSubmitCount <= 0) {
    firstValidationFailures.push({
      stage: 'backend-frame-submit',
      reason: 'controlled execution must submit at least one guarded WebGPU backend frame'
    });
  }
  if (!webgl2HybridRenderingPrevented) {
    firstValidationFailures.push({
      stage: 'webgl2-lifecycle-suppression',
      reason: 'controlled execution cannot mix WebGPU presentation with WebGL2 rendering in the same frame'
    });
  }
  if (!cameraSnapshotProvided) {
    firstValidationFailures.push({
      stage: 'camera-snapshot',
      reason: 'controlled execution needs the viewer lifecycle camera snapshot'
    });
  }
  if (!fallbackPolicyPreserved) {
    firstValidationFailures.push({
      stage: 'fallback-policy',
      reason: 'controlled execution must not report fallback-presented samples as true native success'
    });
  }
  return {
    controlledExecutionReady,
    invocationRequested,
    integrationBoundaryReady,
    adapterReady,
    submittedFrameCount,
    repeatedSubmitCount,
    webgl2FrameLifecycleSuppressed: webgl2FrameLifecycleSuppressed === true,
    webgl2HybridRenderingPrevented,
    cameraSnapshotProvided,
    fallbackPolicyPreserved,
    firstValidationFailures
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

export function buildWebGpuBackendViewerLifecycleControlledExecution({
  integrationBoundary = null,
  adapterResult = null,
  invocationRequested = false,
  invocationSource = 'not-invoked',
  webgl2FrameLifecycleSuppressed = false,
  cameraSnapshot = null
} = {}) {
  const startMs = nowMs();
  const invocationContract = buildControlledExecutionInvocationContract({
    invocationRequested,
    invocationSource,
    integrationBoundary,
    adapterResult
  });
  const validationSummary = buildControlledExecutionValidationSummary({
    invocationContract,
    integrationBoundary,
    webgl2FrameLifecycleSuppressed,
    cameraSnapshot
  });
  const controlledExecutionReady =
    validationSummary.controlledExecutionReady === true;
  return {
    mode: WEBGPU_BACKEND_VIEWER_LIFECYCLE_CONTROLLED_EXECUTION_MODE,
    status: controlledExecutionReady ? 'ok' : 'blocked',
    source:
      'Phase 3 Step59 guarded renderCurrentFrame hook executes a controlled WebGPU backend frame path behind explicit exclusive flags',
    contractVersion:
      WEBGPU_BACKEND_VIEWER_LIFECYCLE_CONTROLLED_EXECUTION_CONTRACT_VERSION,
    controlledExecutionImplemented: true,
    controlledExecutionReady,
    productionDisplayConnectionImplemented: false,
    displayConnectionAllowed: false,
    webgl2HybridRenderingAllowed: false,
    invocationContract,
    selectedSourceKind: invocationContract.selectedSourceKind,
    selectionMode: invocationContract.selectionMode,
    colorPresentSampleCount: invocationContract.colorPresentSampleCount,
    invocationCount: invocationContract.invocationCount,
    submittedFrameCount: validationSummary.submittedFrameCount,
    executedBackendFrameSubmissions:
      invocationContract.executedBackendFrameSubmissions,
    repeatedSubmitCount: invocationContract.repeatedSubmitCount,
    fallbackPolicy: adapterResult?.fallbackPolicy ?? {},
    validationSummary,
    firstValidationFailures: validationSummary.firstValidationFailures,
    blockers: controlledExecutionReady
      ? [
          {
            stage: 'production-frame-scheduler',
            reason:
              'controlled execution is callable from renderCurrentFrame, but the production requestAnimationFrame loop remains intentionally disconnected'
          },
          {
            stage: 'interactive-camera',
            reason:
              'camera data is a snapshot boundary; interactive camera ownership remains outside Step59'
          },
          {
            stage: 'sh-color-evaluation',
            reason: 'WGSL SH/color evaluation parity remains deferred'
          }
        ]
      : [
          {
            stage: 'controlled-webgpu-backend-frame-execution',
            reason:
              'controlled execution requires the explicit hook guard, ready integration boundary, ready adapter, and at least one backend submit'
          }
        ],
    nextBackendPrototypeStep: controlledExecutionReady
      ? 'promote the guarded controlled execution path toward a scheduler-owned backend frame loop without enabling it by default'
      : 'restore controlled viewer lifecycle execution readiness before scheduler integration',
    timing: {
      webgpuBackendViewerLifecycleControlledExecutionMs: nowMs() - startMs
    }
  };
}
