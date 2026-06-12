import {
  buildWebGpuBackendViewerLifecycleControlledExecution
} from './webgpu_backend_viewer_lifecycle_integration.js';
import {
  runWebGpuBackendRuntimeFrame
} from './webgpu_backend_runtime_runner.js';

export const WEBGPU_BACKEND_VIEWER_FRAME_EXECUTOR_MODE =
  'webgpu-backend-viewer-frame-executor-boundary';

export const WEBGPU_BACKEND_VIEWER_FRAME_EXECUTOR_CONTRACT_VERSION =
  'phase3-step63-backend-viewer-frame-executor-boundary-v1';

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function buildExecutorContract({
  requestedBackendMode,
  allowViewerCanvasPresentation,
  enableViewerLoopHook,
  invocationSource,
  frameIndex,
  cameraSnapshot,
  viewerCanvasState,
  backendImplementationKind
}) {
  return {
    contractVersion: WEBGPU_BACKEND_VIEWER_FRAME_EXECUTOR_CONTRACT_VERSION,
    executorMode: 'guarded-viewer-backend-frame-execution-boundary',
    invocationSource,
    frameIndex,
    requestedBackendMode,
    allowViewerCanvasPresentation: allowViewerCanvasPresentation === true,
    enableViewerLoopHook: enableViewerLoopHook === true,
    callableFromViewerLifecycle: true,
    captureDebugFunctionDependency: false,
    directBackendRunner: 'webgpuBackendRuntimeRunner',
    backendImplementationKind,
    recorderRole: 'capture/dry-run debug remains validation oracle and JSON recorder',
    productionLoopConnected: false,
    streamingImplemented: false,
    cameraSnapshotProvided: !!cameraSnapshot,
    viewerCanvasProvided: viewerCanvasState?.provided === true,
    viewerCanvasContextMode: viewerCanvasState?.contextMode ?? null
  };
}

function buildValidationSummary({
  executorContract,
  integrationBoundary,
  backendFrameResult,
  controlledExecution,
  executionError
}) {
  const guardAllowed =
    executorContract.requestedBackendMode === 'webgpu-exclusive' &&
    executorContract.allowViewerCanvasPresentation === true &&
    executorContract.enableViewerLoopHook === true;
  const integrationBoundaryReady =
    integrationBoundary?.integrationBoundaryReady === true;
  const backendFrameResultProvided = !!backendFrameResult;
  const adapterReady =
    backendFrameResult?.webgpuBackendViewerLoopAdapter?.viewerLoopAdapterReady === true;
  const controlledExecutionReady =
    controlledExecution?.controlledExecutionReady === true;
  const webgl2HybridRenderingPrevented =
    integrationBoundary?.validationSummary?.webgl2HybridRenderingPrevented === true;
  const fallbackPolicyPreserved =
    controlledExecution?.validationSummary?.fallbackPolicyPreserved === true;
  const executorReady =
    guardAllowed &&
    integrationBoundaryReady &&
    backendFrameResultProvided &&
    adapterReady &&
    controlledExecutionReady &&
    webgl2HybridRenderingPrevented &&
    fallbackPolicyPreserved &&
    !executionError;
  const firstValidationFailures = [];
  if (!guardAllowed) {
    firstValidationFailures.push({
      stage: 'viewer-backend-executor-guard',
      reason:
        'viewer backend executor requires webgpu-exclusive, viewer canvas presentation guard, and webgpuBackendViewerLoopHook=true'
    });
  }
  if (!integrationBoundaryReady) {
    firstValidationFailures.push({
      stage: 'viewer-lifecycle-integration-boundary',
      reason: 'viewer backend executor requires the Step58 integration boundary to be ready'
    });
  }
  if (!backendFrameResultProvided) {
    firstValidationFailures.push({
      stage: 'backend-frame-result',
      reason: 'viewer backend executor did not receive a backend frame result'
    });
  }
  if (!adapterReady) {
    firstValidationFailures.push({
      stage: 'viewer-loop-adapter',
      reason: 'viewer backend executor requires a ready backend viewer loop adapter'
    });
  }
  if (!controlledExecutionReady) {
    firstValidationFailures.push({
      stage: 'controlled-execution',
      reason: 'viewer backend executor controlled execution summary is not ready'
    });
  }
  if (!webgl2HybridRenderingPrevented) {
    firstValidationFailures.push({
      stage: 'webgl2-lifecycle-suppression',
      reason: 'viewer backend executor cannot run while WebGL2 presentation is active on the same frame'
    });
  }
  if (!fallbackPolicyPreserved) {
    firstValidationFailures.push({
      stage: 'fallback-policy',
      reason: 'viewer backend executor must preserve true native sample selection and fallback suppression'
    });
  }
  if (executionError) {
    firstValidationFailures.push({
      stage: 'backend-frame-execution-error',
      reason: executionError.message ?? 'backend frame executor threw'
    });
  }
  return {
    executorReady,
    guardAllowed,
    integrationBoundaryReady,
    backendFrameResultProvided,
    adapterReady,
    controlledExecutionReady,
    webgl2HybridRenderingPrevented,
    fallbackPolicyPreserved,
    firstValidationFailures
  };
}

export async function executeWebGpuBackendViewerFrame({
  requestedBackendMode = 'webgl2-fallback',
  allowViewerCanvasPresentation = false,
  enableViewerLoopHook = false,
  invocationSource = 'viewer-render-lifecycle-backend-executor',
  frameIndex = 0,
  integrationBoundary = null,
  cameraSnapshot = null,
  viewerCanvasState = null,
  backendImplementationKind = 'webgpu-visible-record-dry-run-runtime',
  runBackendFrame = null
} = {}) {
  const startMs = nowMs();
  const executorContract = buildExecutorContract({
    requestedBackendMode,
    allowViewerCanvasPresentation,
    enableViewerLoopHook,
    invocationSource,
    frameIndex,
    cameraSnapshot,
    viewerCanvasState,
    backendImplementationKind
  });
  let backendFrameResult = null;
  let executionError = null;
  let runtimeRunner = null;
  if (
    typeof runBackendFrame === 'function' &&
    integrationBoundary?.integrationBoundaryReady === true
  ) {
    runtimeRunner = await runWebGpuBackendRuntimeFrame({
      requestedBackendMode,
      allowViewerCanvasPresentation,
      enableViewerLoopHook,
      invocationSource,
      frameIndex,
      cameraSnapshot,
      viewerCanvasState,
      executorContract,
      backendImplementationKind,
      runBackendFrame
    });
    backendFrameResult = runtimeRunner.backendFrameResult;
    executionError = runtimeRunner.executionError;
  }
  const adapterResult = backendFrameResult?.webgpuBackendViewerLoopAdapter ?? null;
  const controlledExecution =
    buildWebGpuBackendViewerLifecycleControlledExecution({
      integrationBoundary,
      adapterResult,
      invocationRequested: integrationBoundary?.integrationBoundaryReady === true,
      invocationSource,
      webgl2FrameLifecycleSuppressed:
        viewerCanvasState?.webgl2FrameLifecycleSuppressed === true,
      cameraSnapshot
    });
  const validationSummary = buildValidationSummary({
    executorContract,
    integrationBoundary,
    backendFrameResult,
    controlledExecution,
    executionError
  });
  const executorReady = validationSummary.executorReady === true;
  const summary = {
    mode: WEBGPU_BACKEND_VIEWER_FRAME_EXECUTOR_MODE,
    status: executorReady ? 'ok' : 'blocked',
    source:
      'Phase 3 Step63 viewer backend frame executor delegates normal backend frame input and present output ownership to the runtime runner',
    contractVersion: WEBGPU_BACKEND_VIEWER_FRAME_EXECUTOR_CONTRACT_VERSION,
    executorImplemented: true,
    executorReady,
    productionDisplayConnectionImplemented: false,
    displayConnectionAllowed: false,
    webgl2HybridRenderingAllowed: false,
    executorContract,
    runtimeRunner: runtimeRunner?.summary ?? null,
    controlledExecution,
    invocationCount: controlledExecution.invocationCount ?? 0,
    submittedFrameCount: controlledExecution.submittedFrameCount ?? 0,
    executedBackendFrameSubmissions:
      controlledExecution.executedBackendFrameSubmissions ?? null,
    repeatedSubmitCount: controlledExecution.repeatedSubmitCount ?? null,
    selectedSourceKind: controlledExecution.selectedSourceKind ?? null,
    selectionMode: controlledExecution.selectionMode ?? null,
    colorPresentSampleCount: controlledExecution.colorPresentSampleCount ?? null,
    fallbackPolicy: controlledExecution.fallbackPolicy ?? {},
    canonicalPresentSummary:
      runtimeRunner?.summary?.canonicalPresentSummary ?? null,
    resourceLifecycleSummary:
      runtimeRunner?.summary?.resourceLifecycleSummary ?? null,
    validationSummary,
    executionError,
    firstValidationFailures: validationSummary.firstValidationFailures,
    blockers: executorReady
      ? [
          {
            stage: 'production-frame-scheduler',
            reason:
              'viewer backend executor is ready; production scheduling remains intentionally disconnected'
          },
          {
            stage: 'streaming-lod',
            reason:
              'streaming, chunking, LOD, and partial upload policies remain future backend extensions'
          },
          {
            stage: 'sh-color-evaluation',
            reason: 'WGSL SH/color evaluation parity remains deferred'
          }
        ]
      : [
          {
            stage: 'viewer-backend-frame-executor',
            reason:
              'executor requires exclusive guards, a ready integration boundary, ready adapter result, and preserved fallback policy'
          }
        ],
    nextBackendPrototypeStep: executorReady
      ? 'connect the viewer backend executor to a scheduler-owned frame loop without using capture/debug as the execution boundary'
      : 'restore viewer backend executor readiness before scheduler integration',
    timing: {
      webgpuBackendViewerFrameExecutorMs: nowMs() - startMs
    }
  };
  return {
    summary,
    backendFrameResult,
    controlledExecution,
    executionError
  };
}
