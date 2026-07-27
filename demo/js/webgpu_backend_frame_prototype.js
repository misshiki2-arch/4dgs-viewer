import { buildWebGpuViewerCanvasCurrentTexturePathReadiness } from './webgpu_viewer_canvas_current_texture_path.js';
import { buildWebGpuViewerCanvasBoundedFirstPresent } from './webgpu_viewer_canvas_bounded_first_present.js';
import { buildWebGpuViewerCanvasNativeBoundedColorSamples } from './webgpu_viewer_canvas_native_bounded_color_samples.js';
import { buildWebGpuViewerCanvasBoundedColorSourceSelector } from './webgpu_viewer_canvas_bounded_color_source_selector.js';
import { buildWebGpuViewerCanvasBoundedColorPresent } from './webgpu_viewer_canvas_bounded_color_present.js';
import {
  WEBGPU_TILE_COMPOSITOR_FRAME_IMPLEMENTATION_MODE
} from './webgpu_tile_compositor_frame_implementation.js';
import {
  DEFAULT_MAX_BOUNDED_COLOR_SAMPLES,
  normalizeBoundedColorSamples
} from './common_4dgs_sample_contracts.js';

export const WEBGPU_BACKEND_FRAME_PROTOTYPE_MODE =
  'webgpu-backend-frame-prototype';

export const WEBGPU_BACKEND_FRAME_CONTINUATION_CONTRACT_VERSION =
  'phase3-step54-backend-frame-continuation-contract-v1';

export const WEBGPU_BACKEND_FRAME_BUDGET_CONTRACT_VERSION =
  'phase3-step54-backend-frame-budget-contract-v1';

export function shouldAllowDiagnosticCanvasPresentation(
  backendImplementationKind = null
) {
  // The production tile compositor owns the viewer currentTexture; bounded
  // presentation remains available to prototype and diagnostic backends.
  return backendImplementationKind !==
    WEBGPU_TILE_COMPOSITOR_FRAME_IMPLEMENTATION_MODE;
}

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function countSamples(value) {
  return Array.isArray(value) ? value.length : 0;
}

function normalizeFrameInputSamples({ list, source, colorSource }) {
  return normalizeBoundedColorSamples({
    list,
    source,
    colorSource,
    maxSamples: DEFAULT_MAX_BOUNDED_COLOR_SAMPLES,
    preserveSampleSource: false,
    includeUpstreamSample: false
  });
}

function buildFrameInputSourceContract({
  webgpuFramebufferFreeTileOutputDryRunComparison,
  webgpuRenderTargetHandoffDryRunComparison,
  webgpuConstrainedDisplayAdapterDryRunComparison,
  webgpuViewerCanvasBoundedColorSourceSelector,
  webgpuViewerCanvasBoundedColorPresent
}) {
  const step40Samples = normalizeFrameInputSamples({
    list: webgpuConstrainedDisplayAdapterDryRunComparison?.sampleTexturePixels,
    source: 'webgpuConstrainedDisplayAdapterDryRunComparison.sampleTexturePixels',
    colorSource: 'Step40 constrained display adapter rgbaFloat sample'
  });
  const step39Samples = normalizeFrameInputSamples({
    list: webgpuRenderTargetHandoffDryRunComparison?.sampleRenderTargetPixels,
    source: 'webgpuRenderTargetHandoffDryRunComparison.sampleRenderTargetPixels',
    colorSource: 'Step39 render target handoff resolvedRgb sample'
  });
  const step38Samples = normalizeFrameInputSamples({
    list: webgpuFramebufferFreeTileOutputDryRunComparison?.sampleTileOutputs,
    source: 'webgpuFramebufferFreeTileOutputDryRunComparison.sampleTileOutputs',
    colorSource: 'Step38 framebuffer-free tile output resolvedRgb sample'
  });
  const colorOutputContract =
    webgpuViewerCanvasBoundedColorPresent?.colorOutputContract ?? {};
  const selectedSourceKind =
    webgpuViewerCanvasBoundedColorSourceSelector?.selectedSourceKind ?? null;
  const sourceAvailability = [
    {
      sourceKind: 'step40-constrained-display-adapter',
      sourcePath: 'webgpuConstrainedDisplayAdapterDryRunComparison.sampleTexturePixels',
      status: webgpuConstrainedDisplayAdapterDryRunComparison?.status ?? null,
      rawSampleCount: countSamples(
        webgpuConstrainedDisplayAdapterDryRunComparison?.sampleTexturePixels
      ),
      presentableSampleCount: step40Samples.length,
      trueNative: true,
      selectedForFrame: selectedSourceKind === 'step40-constrained-display-adapter'
    },
    {
      sourceKind: 'step39-render-target-handoff',
      sourcePath: 'webgpuRenderTargetHandoffDryRunComparison.sampleRenderTargetPixels',
      status: webgpuRenderTargetHandoffDryRunComparison?.status ?? null,
      rawSampleCount: countSamples(
        webgpuRenderTargetHandoffDryRunComparison?.sampleRenderTargetPixels
      ),
      presentableSampleCount: step39Samples.length,
      trueNative: true,
      selectedForFrame: selectedSourceKind === 'step39-render-target-handoff'
    },
    {
      sourceKind: 'step38-framebuffer-free-tile-output',
      sourcePath: 'webgpuFramebufferFreeTileOutputDryRunComparison.sampleTileOutputs',
      status: webgpuFramebufferFreeTileOutputDryRunComparison?.status ?? null,
      rawSampleCount: countSamples(
        webgpuFramebufferFreeTileOutputDryRunComparison?.sampleTileOutputs
      ),
      presentableSampleCount: step38Samples.length,
      trueNative: true,
      selectedForFrame: selectedSourceKind === 'step38-framebuffer-free-tile-output'
    }
  ];
  const presentableTrueNativeSourceKinds = sourceAvailability
    .filter((source) => source.trueNative && source.presentableSampleCount > 0)
    .map((source) => source.sourceKind);
  return {
    contractVersion: 'phase3-step53-backend-frame-input-source-contract-v1',
    maxBoundedColorSamples: DEFAULT_MAX_BOUNDED_COLOR_SAMPLES,
    sourcePriority: [
      'step40-constrained-display-adapter',
      'step39-render-target-handoff',
      'step38-framebuffer-free-tile-output',
      'render-handoff-derived-render-target'
    ],
    selectedSourceKind,
    selectedSourceIsTrueNative:
      selectedSourceKind === 'step40-constrained-display-adapter' ||
      selectedSourceKind === 'step39-render-target-handoff' ||
      selectedSourceKind === 'step38-framebuffer-free-tile-output',
    selectedSampleCount:
      webgpuViewerCanvasBoundedColorSourceSelector?.selectedSampleCount ?? 0,
    selectorPresentableSampleCount:
      colorOutputContract.selectorPresentableSampleCount ?? 0,
    colorPresentSampleCount:
      webgpuViewerCanvasBoundedColorPresent?.colorPresentSampleCount ?? 0,
    sourceAvailability,
    presentableTrueNativeSourceKinds,
    trueNativeInputSourceCount: presentableTrueNativeSourceKinds.length,
    backendFrameInputExpandedBeyondStep40:
      step39Samples.length > 0 || step38Samples.length > 0,
    fallbackSourceKind: 'render-handoff-derived-render-target',
    fallbackUsedForThisFrame: colorOutputContract.fallbackAllowed === true,
    fallbackSuppressedBySelectorSamples:
      colorOutputContract.fallbackSuppressedBySelectorSamples === true
  };
}

function buildFrameBudgetContract({
  frameInputSourceContract,
  webgpuViewerCanvasBoundedColorSourceSelector,
  webgpuViewerCanvasBoundedColorPresent,
  requestedSampleBudget = DEFAULT_MAX_BOUNDED_COLOR_SAMPLES
}) {
  const selectedSampleCount =
    webgpuViewerCanvasBoundedColorSourceSelector?.selectedSampleCount ?? 0;
  const colorPresentSampleCount =
    webgpuViewerCanvasBoundedColorPresent?.colorPresentSampleCount ?? 0;
  const boundedSampleBudget = Math.max(
    1,
    Math.min(
      DEFAULT_MAX_BOUNDED_COLOR_SAMPLES,
      Number.isFinite(Number(requestedSampleBudget))
        ? Math.floor(Number(requestedSampleBudget))
        : DEFAULT_MAX_BOUNDED_COLOR_SAMPLES
    )
  );
  const sourceAvailability = frameInputSourceContract?.sourceAvailability ?? [];
  const maxPresentableSourceSampleCount = sourceAvailability.reduce(
    (maxCount, source) => Math.max(maxCount, source.presentableSampleCount ?? 0),
    0
  );
  return {
    contractVersion: WEBGPU_BACKEND_FRAME_BUDGET_CONTRACT_VERSION,
    budgetMode: 'bounded-n-samples-frame-budget',
    boundedSampleBudget,
    maxBoundedColorSamples: DEFAULT_MAX_BOUNDED_COLOR_SAMPLES,
    selectedSampleCount,
    colorPresentSampleCount,
    maxPresentableSourceSampleCount,
    selectedSamplesWithinBudget: selectedSampleCount <= boundedSampleBudget,
    colorPresentWithinBudget: colorPresentSampleCount <= boundedSampleBudget,
    canScaleWithinCurrentBudget:
      maxPresentableSourceSampleCount > colorPresentSampleCount &&
      colorPresentSampleCount < boundedSampleBudget,
    budgetUtilization:
      boundedSampleBudget > 0 ? colorPresentSampleCount / boundedSampleBudget : 0,
    sourceBudgetPolicy:
      'Step54 keeps Step40 as selected source while exposing bounded N-sample capacity for repeated backend frames'
  };
}

function buildContinuationFrameContract({
  frameIndex = 0,
  previousBackendFramePrototype = null,
  backendFrameReady
}) {
  return {
    contractVersion: WEBGPU_BACKEND_FRAME_CONTINUATION_CONTRACT_VERSION,
    frameIndex: Number.isFinite(Number(frameIndex)) ? Math.max(0, Math.floor(Number(frameIndex))) : 0,
    repeatedFrameCallable: true,
    statelessDryRunFrame: true,
    productionLoopConnected: false,
    previousFrameProvided: !!previousBackendFramePrototype,
    previousFrameStatus: previousBackendFramePrototype?.status ?? null,
    previousFrameReady: previousBackendFramePrototype?.backendFrameReady ?? null,
    currentFrameReady: backendFrameReady,
    continuityPolicy:
      'backend frame coordinator can be invoked repeatedly under the same exclusive canvas and sample/fallback contracts',
    nextFrameInputReusePolicy:
      'reuse current source priority and sample budget until production scheduling owns frame cadence'
  };
}

function buildValidationSummary({
  webgpuViewerCanvasCurrentTexturePath,
  webgpuViewerCanvasBoundedFirstPresent,
  webgpuViewerCanvasNativeBoundedColorSamples,
  webgpuViewerCanvasBoundedColorSourceSelector,
  webgpuViewerCanvasBoundedColorPresent,
  frameInputSourceContract,
  frameBudgetContract,
  continuationFrameContract,
  diagnosticCanvasPresentationAllowed = true
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
  const boundedFirstPresentReady =
    boundedFirstPresentSucceeded || diagnosticCanvasPresentationAllowed === false;
  const diagnosticPresentReady =
    presentSucceeded || diagnosticCanvasPresentationAllowed === false;
  const selectorSelectedSamplesUsed =
    colorOutputContract.selectorSelectedSamplesUsed === true;
  const fallbackSuppressedBySelectorSamples =
    colorOutputContract.fallbackSuppressedBySelectorSamples === true;
  const webgl2HybridRenderingPrevented =
    validationSummary.webgl2HybridRenderingPrevented === true &&
    webgpuViewerCanvasBoundedColorPresent?.webgl2HybridRenderingAllowed === false;
  const backendFrameInputExpandedBeyondStep40 =
    frameInputSourceContract?.backendFrameInputExpandedBeyondStep40 === true;
  const frameBudgetReady =
    frameBudgetContract?.selectedSamplesWithinBudget === true &&
    frameBudgetContract?.colorPresentWithinBudget === true;
  const continuationFrameReady =
    continuationFrameContract?.repeatedFrameCallable === true &&
    continuationFrameContract?.productionLoopConnected === false;
  const backendFrameReady =
    currentTexturePathReady &&
    boundedFirstPresentReady &&
    nativeBoundedSamplesReady &&
    selectorReady &&
    diagnosticPresentReady &&
    selectorSelectedSamplesUsed &&
    fallbackSuppressedBySelectorSamples &&
    webgl2HybridRenderingPrevented &&
    frameBudgetReady &&
    continuationFrameReady;
  const firstValidationFailures = [];
  if (!currentTexturePathReady) {
    firstValidationFailures.push({
      stage: 'current-texture-path',
      reason: 'viewer canvas currentTexture path is not ready for the backend frame'
    });
  }
  if (!boundedFirstPresentReady) {
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
  if (!diagnosticPresentReady) {
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
  if (!frameBudgetReady) {
    firstValidationFailures.push({
      stage: 'frame-budget',
      reason: 'selected or presented bounded samples exceed the backend frame sample budget'
    });
  }
  if (!continuationFrameReady) {
    firstValidationFailures.push({
      stage: 'frame-continuation',
      reason: 'backend frame is not marked reusable under the current dry-run contract'
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
    backendFrameInputExpandedBeyondStep40,
    frameBudgetReady,
    continuationFrameReady,
    presentableTrueNativeSourceKinds:
      frameInputSourceContract?.presentableTrueNativeSourceKinds ?? [],
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
  canvasHeight = 1,
  frameIndex = 0,
  previousBackendFramePrototype = null,
  requestedSampleBudget = DEFAULT_MAX_BOUNDED_COLOR_SAMPLES,
  backendImplementationKind = null
} = {}) {
  const startMs = nowMs();
  const diagnosticCanvasPresentationAllowed =
    shouldAllowDiagnosticCanvasPresentation(backendImplementationKind);
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
      webgpuRenderHandoffStub,
      diagnosticCanvasPresentationAllowed
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
      webgpuConstrainedDisplayAdapterDryRunComparison,
      diagnosticCanvasPresentationAllowed
    });
  const colorOutputContract =
    webgpuViewerCanvasBoundedColorPresent?.colorOutputContract ?? {};
  const frameInputSourceContract = buildFrameInputSourceContract({
    webgpuFramebufferFreeTileOutputDryRunComparison,
    webgpuRenderTargetHandoffDryRunComparison,
    webgpuConstrainedDisplayAdapterDryRunComparison,
    webgpuViewerCanvasBoundedColorSourceSelector,
    webgpuViewerCanvasBoundedColorPresent
  });
  const frameBudgetContract = buildFrameBudgetContract({
    frameInputSourceContract,
    webgpuViewerCanvasBoundedColorSourceSelector,
    webgpuViewerCanvasBoundedColorPresent,
    requestedSampleBudget
  });
  const validationSummary = buildValidationSummary({
    webgpuViewerCanvasCurrentTexturePath,
    webgpuViewerCanvasBoundedFirstPresent,
    webgpuViewerCanvasNativeBoundedColorSamples,
    webgpuViewerCanvasBoundedColorSourceSelector,
    webgpuViewerCanvasBoundedColorPresent,
    frameInputSourceContract,
    frameBudgetContract,
    continuationFrameContract: buildContinuationFrameContract({
      frameIndex,
      previousBackendFramePrototype,
      backendFrameReady: false
    }),
    diagnosticCanvasPresentationAllowed
  });
  const backendFrameReady = validationSummary.backendFrameReady;
  const continuationFrameContract = buildContinuationFrameContract({
    frameIndex,
    previousBackendFramePrototype,
    backendFrameReady
  });
  return {
    mode: WEBGPU_BACKEND_FRAME_PROTOTYPE_MODE,
    status: backendFrameReady ? 'ok' : 'blocked',
    source:
      'Phase 3 Step54 WebGPU backend frame prototype with bounded sample budget and continuation readiness',
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
      inputSourceContract: frameInputSourceContract,
      frameBudgetContract,
      continuationFrameContract,
      sampleSources: colorOutputContract.sampleSources ?? [],
      presentedSamples: colorOutputContract.presentedSamples ?? []
    },
    inputSourceContract: frameInputSourceContract,
    frameBudgetContract,
    continuationFrameContract,
    fallbackPolicy: {
      fallbackKeptForEmptySelectorSamples: true,
      fallbackAllowedForThisFrame: colorOutputContract.fallbackAllowed === true,
      fallbackSuppressedBySelectorSamples:
        colorOutputContract.fallbackSuppressedBySelectorSamples === true,
      selectedSamplesClassifiedAsTrueNative:
        webgpuViewerCanvasBoundedColorSourceSelector?.selectedSourceKind ===
        'step40-constrained-display-adapter' ||
        webgpuViewerCanvasBoundedColorSourceSelector?.selectedSourceKind ===
        'step39-render-target-handoff' ||
        webgpuViewerCanvasBoundedColorSourceSelector?.selectedSourceKind ===
        'step38-framebuffer-free-tile-output'
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
              'Three.js and OrbitControls remain camera input adapters; interactive camera implementation is outside Step54'
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
      ? 'expand backend frame work beyond bounded samples while keeping selector/present/fallback contracts stable'
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
