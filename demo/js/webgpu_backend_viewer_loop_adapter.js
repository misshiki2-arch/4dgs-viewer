import {
  buildWebGpuBackendFrameControlledRepeatedExecution
} from './webgpu_backend_frame_controlled_repeated_execution.js';

export const WEBGPU_BACKEND_VIEWER_LOOP_ADAPTER_MODE =
  'webgpu-backend-viewer-loop-adapter';

export const WEBGPU_BACKEND_VIEWER_LOOP_ADAPTER_CONTRACT_VERSION =
  'phase3-step57-backend-viewer-loop-adapter-contract-v1';

const DEFAULT_ADAPTER_FRAME_COUNT = 3;

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function clampFrameCount(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return DEFAULT_ADAPTER_FRAME_COUNT;
  return Math.max(2, Math.min(3, Math.floor(n)));
}

function buildCameraSnapshotContract(cameraSnapshot) {
  return {
    provided: !!cameraSnapshot,
    source: cameraSnapshot ? 'caller-provided-camera-snapshot' : 'fixed-reference-dry-run',
    cameraInputAdapterMode: 'threejs-camera-input-adapter',
    interactiveCameraImplemented: false,
    projectionContractMutableByAdapter: false
  };
}

function buildFrameExecutionApiContract({ frameCount, cameraSnapshot }) {
  return {
    contractVersion: WEBGPU_BACKEND_VIEWER_LOOP_ADAPTER_CONTRACT_VERSION,
    apiMode: 'manual-bounded-viewer-loop-frame-execution',
    callableFromViewerLoop: true,
    productionLoopConnected: false,
    boundedFrameCount: frameCount,
    frameCallInputs: [
      'frameIndex',
      'previousBackendFramePrototype',
      'cameraSnapshot',
      'exclusiveCanvasGuard',
      'sampleAndFallbackContracts'
    ],
    frameCallOutputs: [
      'backendFramePrototype',
      'currentTextureAcquisition',
      'submitResult',
      'submittedWorkDone',
      'frameSummary'
    ],
    cameraSnapshotContract: buildCameraSnapshotContract(cameraSnapshot)
  };
}

function buildFrameResourceLifecycleContract({ controlledExecution }) {
  const frames = controlledExecution?.frameSummaries ?? [];
  const allFramesSubmitted =
    controlledExecution?.validationSummary?.allFramesSubmitted === true;
  return {
    contractVersion: WEBGPU_BACKEND_VIEWER_LOOP_ADAPTER_CONTRACT_VERSION,
    lifecycleMode: 'manual-bounded-frame-resource-lifecycle',
    reusableResources: [
      'GPUDevice',
      'viewerCanvasState',
      'Step38-40 true native bounded samples',
      'bounded color source selector policy',
      'sample/fallback contracts'
    ],
    perFrameResources: [
      'viewer canvas currentTexture',
      'command encoder',
      'render pass',
      'command buffer',
      'queue submission completion'
    ],
    resourceDisposalPolicy:
      'per-frame WebGPU command resources are scoped to each executed frame; reusable dry-run inputs are retained by the adapter caller',
    perFrameCurrentTextureAcquired: frames.every(
      (frame) => frame.currentTextureAcquisition === true
    ),
    perFrameSubmitCompleted: frames.every(
      (frame) => frame.colorPresentSubmittedWorkDone === true
    ),
    allFramesSubmitted,
    resourceLifecycleReady:
      frames.length > 0 &&
      allFramesSubmitted &&
      frames.every((frame) => frame.frameReady === true)
  };
}

function buildValidationSummary({
  controlledExecution,
  frameExecutionApiContract,
  frameResourceLifecycleContract
}) {
  const controlledReady =
    controlledExecution?.controlledRepeatedExecutionReady === true;
  const adapterCallable =
    frameExecutionApiContract.callableFromViewerLoop === true &&
    frameExecutionApiContract.productionLoopConnected === false;
  const resourceLifecycleReady =
    frameResourceLifecycleContract.resourceLifecycleReady === true;
  const webgl2HybridRenderingPrevented =
    controlledExecution?.webgl2HybridRenderingAllowed === false;
  const fallbackPolicyPreserved =
    controlledExecution?.fallbackPolicy?.fallbackSuppressedBySelectorSamples === true;
  const viewerLoopAdapterReady =
    controlledReady &&
    adapterCallable &&
    resourceLifecycleReady &&
    webgl2HybridRenderingPrevented &&
    fallbackPolicyPreserved;
  const firstValidationFailures = [];
  if (!controlledReady) {
    firstValidationFailures.push({
      stage: 'controlled-repeated-execution',
      reason: 'controlled repeated backend execution is not ready'
    });
  }
  if (!adapterCallable) {
    firstValidationFailures.push({
      stage: 'viewer-loop-api',
      reason: 'manual viewer loop adapter API is not callable under the current guard'
    });
  }
  if (!resourceLifecycleReady) {
    firstValidationFailures.push({
      stage: 'frame-resource-lifecycle',
      reason: 'per-frame resources did not complete for every controlled frame'
    });
  }
  if (!webgl2HybridRenderingPrevented) {
    firstValidationFailures.push({
      stage: 'canvas-ownership',
      reason: 'viewer loop adapter requires WebGPU presentation without WebGL2 hybrid rendering'
    });
  }
  if (!fallbackPolicyPreserved) {
    firstValidationFailures.push({
      stage: 'fallback-policy',
      reason: 'selector samples must suppress fallback mixing for the adapter path'
    });
  }
  return {
    viewerLoopAdapterReady,
    controlledReady,
    adapterCallable,
    resourceLifecycleReady,
    webgl2HybridRenderingPrevented,
    fallbackPolicyPreserved,
    firstValidationFailures
  };
}

export async function buildWebGpuBackendViewerLoopAdapter({
  initialBackendFramePrototype = null,
  repeatedFrameCount = DEFAULT_ADAPTER_FRAME_COUNT,
  executeBackendFrame = null,
  cameraSnapshot = null
} = {}) {
  const startMs = nowMs();
  const frameCount = clampFrameCount(repeatedFrameCount);
  const frameExecutionApiContract = buildFrameExecutionApiContract({
    frameCount,
    cameraSnapshot
  });
  const webgpuBackendFrameControlledRepeatedExecution =
    await buildWebGpuBackendFrameControlledRepeatedExecution({
      initialBackendFramePrototype,
      repeatedFrameCount: frameCount,
      executeBackendFrame
    });
  const frameResourceLifecycleContract = buildFrameResourceLifecycleContract({
    controlledExecution: webgpuBackendFrameControlledRepeatedExecution
  });
  const validationSummary = buildValidationSummary({
    controlledExecution: webgpuBackendFrameControlledRepeatedExecution,
    frameExecutionApiContract,
    frameResourceLifecycleContract
  });
  const viewerLoopAdapterReady =
    validationSummary.viewerLoopAdapterReady === true;
  return {
    mode: WEBGPU_BACKEND_VIEWER_LOOP_ADAPTER_MODE,
    status: viewerLoopAdapterReady ? 'ok' : 'blocked',
    source:
      'Phase 3 Step57 controlled WebGPU backend viewer loop adapter with frame execution API and resource lifecycle',
    contractVersion: WEBGPU_BACKEND_VIEWER_LOOP_ADAPTER_CONTRACT_VERSION,
    viewerLoopAdapterImplemented: true,
    viewerLoopAdapterReady,
    productionDisplayConnectionImplemented: false,
    displayConnectionAllowed: false,
    webgl2HybridRenderingAllowed: false,
    requestedFrameCount: frameCount,
    frameExecutionApiContract,
    frameResourceLifecycleContract,
    webgpuBackendFrameControlledRepeatedExecution,
    selectedSourceKind:
      webgpuBackendFrameControlledRepeatedExecution?.selectedSourceKind ?? null,
    colorPresentSampleCount:
      webgpuBackendFrameControlledRepeatedExecution?.colorPresentSampleCount ?? null,
    executedBackendFrameSubmissions:
      webgpuBackendFrameControlledRepeatedExecution?.executedBackendFrameSubmissions ?? 0,
    repeatedSubmitCount:
      webgpuBackendFrameControlledRepeatedExecution?.repeatedSubmitCount ?? 0,
    fallbackPolicy:
      webgpuBackendFrameControlledRepeatedExecution?.fallbackPolicy ?? {},
    validationSummary,
    firstValidationFailures: validationSummary.firstValidationFailures,
    blockers: viewerLoopAdapterReady
      ? [
          {
            stage: 'production-frame-scheduler',
            reason:
              'viewer loop adapter is ready; production requestAnimationFrame scheduling remains intentionally disconnected'
          },
          {
            stage: 'interactive-camera',
            reason:
              'Three.js and OrbitControls remain camera input adapters; interactive camera implementation is outside Step57'
          },
          {
            stage: 'sh-color-evaluation',
            reason: 'WGSL SH/color evaluation parity remains deferred'
          }
        ]
      : [
          {
            stage: 'webgpu-backend-viewer-loop-adapter',
            reason:
              'viewer loop adapter requires controlled execution readiness, stable guards, and complete per-frame resources'
          }
        ],
    nextBackendPrototypeStep: viewerLoopAdapterReady
      ? 'connect the manual bounded viewer loop adapter to a guarded scheduler entrypoint'
      : 'restore viewer loop adapter readiness before scheduler integration',
    timing: {
      webgpuBackendViewerLoopAdapterMs: nowMs() - startMs,
      ...webgpuBackendFrameControlledRepeatedExecution.timing
    }
  };
}
