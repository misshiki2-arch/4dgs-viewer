import { buildWebGpuViewerCanvasCurrentTexturePathReadiness } from './webgpu_viewer_canvas_current_texture_path.js';
import { buildWebGpuViewerCanvasBoundedFirstPresent } from './webgpu_viewer_canvas_bounded_first_present.js';
import { buildWebGpuViewerCanvasNativeBoundedColorSamples } from './webgpu_viewer_canvas_native_bounded_color_samples.js';
import { buildWebGpuViewerCanvasBoundedColorSourceSelector } from './webgpu_viewer_canvas_bounded_color_source_selector.js';
import { buildWebGpuViewerCanvasBoundedColorPresent } from './webgpu_viewer_canvas_bounded_color_present.js';

export const WEBGPU_BACKEND_FRAME_PROTOTYPE_MODE =
  'webgpu-backend-frame-prototype';

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function buildValidationSummary({
  webgpuViewerCanvasCurrentTexturePath,
  webgpuViewerCanvasBoundedFirstPresent,
  webgpuViewerCanvasNativeBoundedColorSamples,
  webgpuViewerCanvasBoundedColorSourceSelector,
  webgpuViewerCanvasBoundedColorPresent
}) {
  const colorOutputContract =
    webgpuViewerCanvasBoundedColorPresent?.colorOutputContract ?? {};
  const validationSummary =
    webgpuViewerCanvasBoundedColorPresent?.validationSummary ?? {};
  const currentTexturePathReady =
    webgpuViewerCanvasCurrentTexturePath?.viewerCanvasCurrentTexturePathReady === true;
  const boundedFirstPresentSucceeded =
    webgpuViewerCanvasBoundedFirstPresent?.boundedViewerCanvasFirstPresentSucceeded === true;
  const nativeBoundedSamplesReady =
    webgpuViewerCanvasNativeBoundedColorSamples?.nativeBoundedSamplesReady === true;
  const selectorReady =
    webgpuViewerCanvasBoundedColorSourceSelector?.boundedColorSourceReady === true;
  const presentSucceeded =
    webgpuViewerCanvasBoundedColorPresent?.boundedViewerCanvasColorPresentSucceeded === true;
  const selectorSelectedSamplesUsed =
    colorOutputContract.selectorSelectedSamplesUsed === true;
  const fallbackSuppressedBySelectorSamples =
    colorOutputContract.fallbackSuppressedBySelectorSamples === true;
  const webgl2HybridRenderingPrevented =
    validationSummary.webgl2HybridRenderingPrevented === true &&
    webgpuViewerCanvasBoundedColorPresent?.webgl2HybridRenderingAllowed === false;
  const backendFrameReady =
    currentTexturePathReady &&
    boundedFirstPresentSucceeded &&
    nativeBoundedSamplesReady &&
    selectorReady &&
    presentSucceeded &&
    selectorSelectedSamplesUsed &&
    fallbackSuppressedBySelectorSamples &&
    webgl2HybridRenderingPrevented;
  const firstValidationFailures = [];
  if (!currentTexturePathReady) {
    firstValidationFailures.push({
      stage: 'current-texture-path',
      reason: 'viewer canvas currentTexture path is not ready for the backend frame'
    });
  }
  if (!boundedFirstPresentSucceeded) {
    firstValidationFailures.push({
      stage: 'bounded-first-present',
      reason: 'bounded first-present did not succeed before backend color present'
    });
  }
  if (!nativeBoundedSamplesReady) {
    firstValidationFailures.push({
      stage: 'native-bounded-color-samples',
      reason: 'native bounded color samples are not ready for selector input'
    });
  }
  if (!selectorReady) {
    firstValidationFailures.push({
      stage: 'bounded-color-source-selector',
      reason: 'selector did not produce a bounded color source'
    });
  }
  if (!presentSucceeded) {
    firstValidationFailures.push({
      stage: 'bounded-color-present',
      reason: 'viewer canvas bounded color present did not submit successfully'
    });
  }
  if (!selectorSelectedSamplesUsed) {
    firstValidationFailures.push({
      stage: 'sample-selection',
      reason: 'backend frame requires selector-selected samples for this prototype'
    });
  }
  if (!fallbackSuppressedBySelectorSamples) {
    firstValidationFailures.push({
      stage: 'fallback-policy',
      reason: 'fallback samples must be suppressed when selector samples are present'
    });
  }
  if (!webgl2HybridRenderingPrevented) {
    firstValidationFailures.push({
      stage: 'canvas-ownership',
      reason: 'backend frame requires WebGPU presentation without WebGL2 hybrid rendering'
    });
  }
  return {
    backendFrameReady,
    currentTexturePathReady,
    boundedFirstPresentSucceeded,
    nativeBoundedSamplesReady,
    selectorReady,
    presentSucceeded,
    selectorSelectedSamplesUsed,
    fallbackSuppressedBySelectorSamples,
    webgl2HybridRenderingPrevented,
    cameraProjectionContractUnchanged:
      validationSummary.cameraProjectionContractUnchanged === true,
    shEvaluationDeferred:
      webgpuViewerCanvasBoundedColorPresent?.shPolicy?.status === 'deferred',
    firstValidationFailures
  };
}

export async function buildWebGpuBackendFramePrototype({
  device,
  viewerCanvasState = null,
  webgpuRenderHandoffStub = null,
  webgpuFramebufferFreeTileOutputDryRunComparison = null,
  webgpuRenderTargetHandoffDryRunComparison = null,
  webgpuConstrainedDisplayAdapterDryRunComparison = null,
  canvasWidth = 1,
  canvasHeight = 1
} = {}) {
  const startMs = nowMs();
  const webgpuViewerCanvasCurrentTexturePath =
    await buildWebGpuViewerCanvasCurrentTexturePathReadiness({
      device,
      viewerCanvasState
    });
  const webgpuViewerCanvasBoundedFirstPresent =
    await buildWebGpuViewerCanvasBoundedFirstPresent({
      device,
      viewerCanvasState,
      webgpuViewerCanvasCurrentTexturePath,
      webgpuRenderHandoffStub
    });
  const webgpuViewerCanvasNativeBoundedColorSamples =
    buildWebGpuViewerCanvasNativeBoundedColorSamples({
      webgpuFramebufferFreeTileOutputDryRunComparison,
      webgpuRenderTargetHandoffDryRunComparison,
      webgpuConstrainedDisplayAdapterDryRunComparison,
      webgpuRenderHandoffStub,
      canvasWidth,
      canvasHeight
    });
  const webgpuViewerCanvasBoundedColorSourceSelector =
    buildWebGpuViewerCanvasBoundedColorSourceSelector({
      webgpuViewerCanvasNativeBoundedColorSamples,
      webgpuConstrainedDisplayAdapterDryRunComparison,
      webgpuRenderTargetHandoffDryRunComparison,
      webgpuFramebufferFreeTileOutputDryRunComparison,
      webgpuRenderHandoffStub
    });
  const webgpuViewerCanvasBoundedColorPresent =
    await buildWebGpuViewerCanvasBoundedColorPresent({
      device,
      viewerCanvasState,
      webgpuViewerCanvasCurrentTexturePath,
      webgpuViewerCanvasBoundedFirstPresent,
      webgpuViewerCanvasBoundedColorSourceSelector,
      webgpuRenderHandoffStub,
      webgpuFramebufferFreeTileOutputDryRunComparison,
      webgpuRenderTargetHandoffDryRunComparison,
      webgpuConstrainedDisplayAdapterDryRunComparison
    });
  const colorOutputContract =
    webgpuViewerCanvasBoundedColorPresent?.colorOutputContract ?? {};
  const validationSummary = buildValidationSummary({
    webgpuViewerCanvasCurrentTexturePath,
    webgpuViewerCanvasBoundedFirstPresent,
    webgpuViewerCanvasNativeBoundedColorSamples,
    webgpuViewerCanvasBoundedColorSourceSelector,
    webgpuViewerCanvasBoundedColorPresent
  });
  const backendFrameReady = validationSummary.backendFrameReady;
  return {
    mode: WEBGPU_BACKEND_FRAME_PROTOTYPE_MODE,
    status: backendFrameReady ? 'ok' : 'blocked',
    source:
      'Phase 3 Step52 WebGPU backend frame prototype coordinating currentTexture, selector, present, and summary',
    backendFramePrototypeImplemented: true,
    backendFrameReady,
    productionDisplayConnectionImplemented: false,
    displayConnectionAllowed: false,
    webgl2HybridRenderingAllowed: false,
    requestedBackendMode:
      webgpuViewerCanvasBoundedColorPresent?.requestedBackendMode ?? null,
    frameUnitContract: {
      frameStages: [
        'viewer-canvas-current-texture-acquisition',
        'bounded-first-present-guard',
        'native-bounded-color-sample-retention',
        'bounded-color-source-selection',
        'viewer-canvas-bounded-color-present-submission',
        'backend-frame-summary'
      ],
      currentTextureAcquisition:
        webgpuViewerCanvasCurrentTexturePath?.viewerCanvasCurrentTexturePathReady === true,
      selectedSourceKind:
        webgpuViewerCanvasBoundedColorSourceSelector?.selectedSourceKind ?? null,
      selectionMode: colorOutputContract.selectionMode ?? null,
      selectedSampleCount:
        webgpuViewerCanvasBoundedColorSourceSelector?.selectedSampleCount ?? null,
      selectorSelectedRawSampleCount:
        colorOutputContract.selectorSelectedRawSampleCount ?? null,
      selectorPresentableSampleCount:
        colorOutputContract.selectorPresentableSampleCount ?? null,
      colorPresentSampleCount:
        webgpuViewerCanvasBoundedColorPresent?.colorPresentSampleCount ?? null,
      commandBufferSubmitted:
        webgpuViewerCanvasBoundedColorPresent?.commandBufferSubmitted === true,
      submittedWorkDone:
        webgpuViewerCanvasBoundedColorPresent?.submittedWorkDone === true,
      sampleContract: colorOutputContract.sampleContract ?? null,
      sampleSources: colorOutputContract.sampleSources ?? [],
      presentedSamples: colorOutputContract.presentedSamples ?? []
    },
    fallbackPolicy: {
      fallbackKeptForEmptySelectorSamples: true,
      fallbackAllowedForThisFrame: colorOutputContract.fallbackAllowed === true,
      fallbackSuppressedBySelectorSamples:
        colorOutputContract.fallbackSuppressedBySelectorSamples === true,
      selectedSamplesClassifiedAsTrueNative:
        webgpuViewerCanvasBoundedColorSourceSelector?.selectedSourceKind ===
        'step40-constrained-display-adapter'
    },
    viewerCanvasOwnershipContract:
      webgpuViewerCanvasBoundedColorPresent?.viewerCanvasOwnershipContract ?? {},
    cameraProjectionContract:
      webgpuViewerCanvasBoundedColorPresent?.cameraProjectionContract ?? {},
    shPolicy: webgpuViewerCanvasBoundedColorPresent?.shPolicy ?? {},
    validationSummary,
    firstValidationFailures: validationSummary.firstValidationFailures,
    blockers: backendFrameReady
      ? [
          {
            stage: 'production-display-connection',
            reason:
              'backend frame prototype succeeded; production viewer rendering remains intentionally disconnected'
          },
          {
            stage: 'interactive-camera',
            reason:
              'Three.js and OrbitControls remain camera input adapters; interactive camera implementation is outside Step52'
          },
          {
            stage: 'sh-color-evaluation',
            reason: 'WGSL SH/color evaluation parity remains deferred'
          }
        ]
      : [
          {
            stage: 'webgpu-backend-frame-prototype',
            reason:
              'backend frame requires currentTexture readiness, selector samples, and bounded color present submission'
          }
        ],
    nextBackendPrototypeStep: backendFrameReady
      ? 'expand backend frame inputs beyond bounded samples while keeping selector/present/fallback contracts stable'
      : 'restore backend frame readiness before expanding the normal backend prototype',
    webgpuViewerCanvasCurrentTexturePath,
    webgpuViewerCanvasBoundedFirstPresent,
    webgpuViewerCanvasNativeBoundedColorSamples,
    webgpuViewerCanvasBoundedColorSourceSelector,
    webgpuViewerCanvasBoundedColorPresent,
    timing: {
      webgpuBackendFramePrototypeMs: nowMs() - startMs,
      ...webgpuViewerCanvasCurrentTexturePath.timing,
      ...webgpuViewerCanvasBoundedFirstPresent.timing,
      ...webgpuViewerCanvasNativeBoundedColorSamples.timing,
      ...webgpuViewerCanvasBoundedColorSourceSelector.timing,
      ...webgpuViewerCanvasBoundedColorPresent.timing
    }
  };
}
