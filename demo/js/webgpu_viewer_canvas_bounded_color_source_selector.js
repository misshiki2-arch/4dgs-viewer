import {
  DEFAULT_MAX_BOUNDED_COLOR_SAMPLES,
  colorToRgbaFloatArray,
  normalizeBoundedColorSamples
} from './common_4dgs_sample_contracts.js';

export const WEBGPU_VIEWER_CANVAS_BOUNDED_COLOR_SOURCE_SELECTOR_MODE =
  'webgpu-viewer-canvas-bounded-color-source-selector';

const MAX_SELECTED_SAMPLES = DEFAULT_MAX_BOUNDED_COLOR_SAMPLES;

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function normalizeSelectedSamples({ list, source, colorSource }) {
  return normalizeBoundedColorSamples({
    list,
    source,
    colorSource,
    maxSamples: MAX_SELECTED_SAMPLES,
    preserveSampleSource: false,
    includeUpstreamSample: true
  });
}

function buildRenderTargetSamplesFromRenderHandoff(webgpuRenderHandoffStub) {
  const selectedSamples = normalizeSelectedSamples({
    list: webgpuRenderHandoffStub?.sampleRecords,
    source: 'webgpuRenderHandoffStub.sampleRecords',
    colorSource: 'reference-assisted render payload colorAlpha.rgb'
  });
  return selectedSamples.map((sample) => ({
    source: 'webgpuViewerCanvasBoundedColorSourceSelector.renderHandoffDerivedRenderTarget',
    colorSource: 'render-target-shaped reference-assisted colorAlpha.rgb',
    recordIndex: sample.recordIndex,
    sampleKind: sample.sampleKind ?? 'render-handoff-derived',
    srcIndex: sample.srcIndex,
    valid: sample.valid,
    samplePx: sample.samplePx,
    colorAlpha: sample.colorAlpha,
    expected: {
      rgbaFloat: colorToRgbaFloatArray(sample.colorAlpha),
      resolvedColor: colorToRgbaFloatArray(sample.colorAlpha)
    },
    actual: {
      rgbaFloat: colorToRgbaFloatArray(sample.colorAlpha),
      resolvedColor: colorToRgbaFloatArray(sample.colorAlpha)
    },
    derivation:
      'shapes Step32 reference-assisted render payload colorAlpha.rgb as a bounded render-target sample when Step38-40 samples are unavailable'
  }));
}

function countSamples(value) {
  return Array.isArray(value) ? value.length : 0;
}

export function buildWebGpuViewerCanvasBoundedColorSourceSelector({
  webgpuViewerCanvasNativeBoundedColorSamples = null,
  webgpuConstrainedDisplayAdapterDryRunComparison = null,
  webgpuRenderTargetHandoffDryRunComparison = null,
  webgpuFramebufferFreeTileOutputDryRunComparison = null,
  webgpuRenderHandoffStub = null
} = {}) {
  const startMs = nowMs();
  const nativeBridgeStep40Samples = normalizeSelectedSamples({
    list: webgpuViewerCanvasNativeBoundedColorSamples?.sampleTexturePixels,
    source: 'webgpuViewerCanvasNativeBoundedColorSamples.sampleTexturePixels',
    colorSource: 'Step50 native-compatible constrained display rgbaFloat sample'
  });
  const nativeBridgeStep39Samples = normalizeSelectedSamples({
    list: webgpuViewerCanvasNativeBoundedColorSamples?.sampleRenderTargetPixels,
    source: 'webgpuViewerCanvasNativeBoundedColorSamples.sampleRenderTargetPixels',
    colorSource: 'Step53 native-compatible render target resolvedRgb sample'
  });
  const nativeBridgeStep38Samples = normalizeSelectedSamples({
    list: webgpuViewerCanvasNativeBoundedColorSamples?.sampleTileOutputs,
    source: 'webgpuViewerCanvasNativeBoundedColorSamples.sampleTileOutputs',
    colorSource: 'Step53 native-compatible framebuffer-free tile output resolvedRgb sample'
  });
  const step40Samples = normalizeSelectedSamples({
    list: webgpuConstrainedDisplayAdapterDryRunComparison?.sampleTexturePixels,
    source: 'webgpuConstrainedDisplayAdapterDryRunComparison.sampleTexturePixels',
    colorSource: 'Step40 constrained display adapter rgbaFloat sample'
  });
  const step39Samples = normalizeSelectedSamples({
    list: webgpuRenderTargetHandoffDryRunComparison?.sampleRenderTargetPixels,
    source: 'webgpuRenderTargetHandoffDryRunComparison.sampleRenderTargetPixels',
    colorSource: 'Step39 render target handoff resolvedRgb sample'
  });
  const step38Samples = normalizeSelectedSamples({
    list: webgpuFramebufferFreeTileOutputDryRunComparison?.sampleTileOutputs,
    source: 'webgpuFramebufferFreeTileOutputDryRunComparison.sampleTileOutputs',
    colorSource: 'Step38 framebuffer-free tile output resolvedRgb sample'
  });
  const derivedRenderTargetSamples =
    buildRenderTargetSamplesFromRenderHandoff(webgpuRenderHandoffStub);

  const sourceCandidates = [
    {
      sourceKind: 'step40-constrained-display-adapter',
      status: webgpuConstrainedDisplayAdapterDryRunComparison?.status ?? null,
      upstreamSampleCount: countSamples(webgpuConstrainedDisplayAdapterDryRunComparison?.sampleTexturePixels),
      normalizedSampleCount: step40Samples.length,
      selectedSamples: step40Samples,
      unavailableReason:
        webgpuConstrainedDisplayAdapterDryRunComparison?.firstMismatches?.[0]?.reason ??
        null
    },
    {
      sourceKind: 'step40-constrained-display-adapter-native-bridge',
      status: webgpuViewerCanvasNativeBoundedColorSamples?.status ?? null,
      upstreamSampleCount: countSamples(webgpuViewerCanvasNativeBoundedColorSamples?.sampleTexturePixels),
      normalizedSampleCount: nativeBridgeStep40Samples.length,
      selectedSamples: nativeBridgeStep40Samples,
      unavailableReason:
        webgpuViewerCanvasNativeBoundedColorSamples?.validationSummary
          ?.firstValidationFailures?.[0]?.reason ??
        null
    },
    {
      sourceKind: 'step39-render-target-handoff',
      status: webgpuRenderTargetHandoffDryRunComparison?.status ?? null,
      upstreamSampleCount: countSamples(webgpuRenderTargetHandoffDryRunComparison?.sampleRenderTargetPixels),
      normalizedSampleCount: step39Samples.length,
      selectedSamples: step39Samples,
      unavailableReason:
        webgpuRenderTargetHandoffDryRunComparison?.firstMismatches?.[0]?.reason ??
        null
    },
    {
      sourceKind: 'step39-render-target-handoff-native-bridge',
      status: webgpuViewerCanvasNativeBoundedColorSamples?.status ?? null,
      upstreamSampleCount: countSamples(webgpuViewerCanvasNativeBoundedColorSamples?.sampleRenderTargetPixels),
      normalizedSampleCount: nativeBridgeStep39Samples.length,
      selectedSamples: nativeBridgeStep39Samples,
      unavailableReason:
        webgpuViewerCanvasNativeBoundedColorSamples?.validationSummary
          ?.firstValidationFailures?.[0]?.reason ??
        null
    },
    {
      sourceKind: 'step38-framebuffer-free-tile-output',
      status: webgpuFramebufferFreeTileOutputDryRunComparison?.status ?? null,
      upstreamSampleCount: countSamples(webgpuFramebufferFreeTileOutputDryRunComparison?.sampleTileOutputs),
      normalizedSampleCount: step38Samples.length,
      selectedSamples: step38Samples,
      unavailableReason:
        webgpuFramebufferFreeTileOutputDryRunComparison?.firstMismatches?.[0]?.reason ??
        null
    },
    {
      sourceKind: 'step38-framebuffer-free-tile-output-native-bridge',
      status: webgpuViewerCanvasNativeBoundedColorSamples?.status ?? null,
      upstreamSampleCount: countSamples(webgpuViewerCanvasNativeBoundedColorSamples?.sampleTileOutputs),
      normalizedSampleCount: nativeBridgeStep38Samples.length,
      selectedSamples: nativeBridgeStep38Samples,
      unavailableReason:
        webgpuViewerCanvasNativeBoundedColorSamples?.validationSummary
          ?.firstValidationFailures?.[0]?.reason ??
        null
    },
    {
      sourceKind: 'render-handoff-derived-render-target',
      status: webgpuRenderHandoffStub?.status ?? null,
      upstreamSampleCount: countSamples(webgpuRenderHandoffStub?.sampleRecords),
      normalizedSampleCount: derivedRenderTargetSamples.length,
      selectedSamples: derivedRenderTargetSamples,
      unavailableReason: null
    }
  ];
  const selectedCandidate =
    sourceCandidates.find((candidate) => candidate.normalizedSampleCount > 0) ??
    sourceCandidates[sourceCandidates.length - 1];
  const selectedColorSamples = selectedCandidate?.selectedSamples ?? [];
  const boundedColorSourceReady = selectedColorSamples.length > 0;

  return {
    mode: WEBGPU_VIEWER_CANVAS_BOUNDED_COLOR_SOURCE_SELECTOR_MODE,
    status: boundedColorSourceReady ? 'ok' : 'unavailable',
    source:
      'Step50 bounded viewer-canvas color source selector for true native tile/render-target/display samples',
    boundedColorSourceSelectorImplemented: true,
    boundedColorSourceReady,
    selectedSourceKind: selectedCandidate?.sourceKind ?? null,
    selectedColorSource:
      selectedColorSamples[0]?.colorSource ??
      null,
    selectedSampleCount: selectedColorSamples.length,
    selectedColorSamples,
    selectedRenderTargetSamples:
      selectedCandidate?.sourceKind === 'render-handoff-derived-render-target'
        ? selectedColorSamples
        : [],
    sourcePriority: sourceCandidates.map((candidate) => ({
      sourceKind: candidate.sourceKind,
      status: candidate.status,
      upstreamSampleCount: candidate.upstreamSampleCount,
      normalizedSampleCount: candidate.normalizedSampleCount,
      unavailableReason: candidate.unavailableReason
    })),
    sourceAvailability: {
      step40ConstrainedDisplayAdapterSamples: step40Samples.length,
      step40NativeBridgeSamples: nativeBridgeStep40Samples.length,
      step39RenderTargetHandoffSamples: step39Samples.length,
      step39NativeBridgeSamples: nativeBridgeStep39Samples.length,
      step38FramebufferFreeTileOutputSamples: step38Samples.length,
      step38NativeBridgeSamples: nativeBridgeStep38Samples.length,
      renderHandoffDerivedRenderTargetSamples: derivedRenderTargetSamples.length
    },
    fallbackPolicy: {
      referenceAssistedFallbackKept: true,
      fallbackSource: 'webgpuRenderHandoffStub.sampleRecords colorAlpha.rgb',
      fallbackShape:
        'render-handoff samples are normalized into render-target-shaped bounded samples so viewer canvas color present can consume the same selected sample contract'
    },
    nativeBridgePolicy: {
      nativeBridgeEnabled:
        webgpuViewerCanvasNativeBoundedColorSamples?.nativeBoundedSamplesBridgeImplemented === true,
      originalStep38To40SamplesPreferred: true,
      selectedNativeSourceKind:
        webgpuViewerCanvasNativeBoundedColorSamples?.selectedNativeSourceKind ?? null,
      generatedFromRenderHandoff:
        webgpuViewerCanvasNativeBoundedColorSamples?.generatedFromRenderHandoff === true
    },
    anyMismatch: !boundedColorSourceReady,
    mismatchClassification: boundedColorSourceReady
      ? 'none'
      : 'viewerCanvasBoundedColorSourceUnavailable',
    validationSummary: {
      step40SamplesAvailable: step40Samples.length > 0,
      step40NativeBridgeSamplesAvailable: nativeBridgeStep40Samples.length > 0,
      step39SamplesAvailable: step39Samples.length > 0,
      step39NativeBridgeSamplesAvailable: nativeBridgeStep39Samples.length > 0,
      step38SamplesAvailable: step38Samples.length > 0,
      step38NativeBridgeSamplesAvailable: nativeBridgeStep38Samples.length > 0,
      renderHandoffFallbackAvailable: derivedRenderTargetSamples.length > 0,
      boundedColorSourceReady,
      selectedSourceKind: selectedCandidate?.sourceKind ?? null,
      firstValidationFailures: boundedColorSourceReady
        ? []
        : [
            {
              stage: 'bounded-color-source-selector',
              reason:
                'no Step38-40 samples or render-handoff reference-assisted color samples are available'
            }
          ]
    },
    blockers: boundedColorSourceReady
      ? [
          {
            stage: 'tile-render-target-native-samples',
            reason:
              'selector prefers true Step38-40 samples, then native-compatible bridge samples, before using render-handoff-derived fallback'
          }
        ]
      : [
          {
            stage: 'bounded-color-source-selector',
            reason:
              'bounded color present needs Step38-40 samples or render-handoff reference-assisted samples'
          }
        ],
    nextBackendPrototypeStep: boundedColorSourceReady
      ? 'expand true Step38-40 tile/render-target samples while keeping viewer canvas bounded color present contract stable'
      : 'restore bounded color source availability before expanding viewer canvas color output',
    timing: {
      viewerCanvasBoundedColorSourceSelectorMs: nowMs() - startMs
    }
  };
}
