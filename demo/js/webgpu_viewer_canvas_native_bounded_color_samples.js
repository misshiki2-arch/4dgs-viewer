export const WEBGPU_VIEWER_CANVAS_NATIVE_BOUNDED_COLOR_SAMPLES_MODE =
  'webgpu-viewer-canvas-native-bounded-color-samples';

const MAX_NATIVE_BRIDGE_SAMPLES = 8;

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function finiteOrZero(value) {
  return Number.isFinite(value) ? value : 0;
}

function clamp01(value, fallback = 0) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(1, Math.max(0, n));
}

function normalizeColorArray(value) {
  if (!Array.isArray(value) && !(value instanceof Float32Array)) return null;
  if (value.length < 3) return null;
  return [
    clamp01(value[0]),
    clamp01(value[1]),
    clamp01(value[2]),
    clamp01(value.length >= 4 ? value[3] : 1, 1)
  ];
}

function colorFromSample(sample) {
  return (
    normalizeColorArray(sample?.actual?.rgbaFloat) ??
    normalizeColorArray(sample?.expected?.rgbaFloat) ??
    normalizeColorArray(sample?.actual?.resolvedColor) ??
    normalizeColorArray(sample?.expected?.resolvedColor) ??
    normalizeColorArray(sample?.actual?.colorAlpha) ??
    normalizeColorArray(sample?.expected?.colorAlpha) ??
    normalizeColorArray(sample?.colorAlpha) ??
    normalizeColorArray(sample?.payload?.colorAlpha) ??
    normalizeColorArray(sample?.referenceAssisted?.colorAlpha) ??
    normalizeColorArray(sample?.renderPayload?.colorAlpha)
  );
}

function samplePxFromSample(sample, fallbackIndex, canvasWidth, canvasHeight) {
  const px =
    sample?.pixel ??
    sample?.samplePx ??
    sample?.actual?.pixel ??
    sample?.expected?.pixel ??
    sample?.centerPx;
  let x = 32 + fallbackIndex * 24;
  let y = 32 + fallbackIndex * 24;
  if (Array.isArray(px) && px.length >= 2) {
    x = Number(px[0]);
    y = Number(px[1]);
  } else if (px && typeof px === 'object') {
    x = Number(px.x ?? px[0]);
    y = Number(px.y ?? px[1]);
  }
  const width = Math.max(1, Math.round(finiteOrZero(canvasWidth)));
  const height = Math.max(1, Math.round(finiteOrZero(canvasHeight)));
  return [
    Math.min(Math.max(0, finiteOrZero(x)), width - 1),
    Math.min(Math.max(0, finiteOrZero(y)), height - 1)
  ];
}

function countSamples(value) {
  return Array.isArray(value) ? value.length : 0;
}

function cloneList(value, maxSamples = MAX_NATIVE_BRIDGE_SAMPLES) {
  return Array.isArray(value) ? value.slice(0, maxSamples) : [];
}

function buildSeedSamples({ webgpuRenderHandoffStub, canvasWidth, canvasHeight }) {
  const seeds = [];
  for (const sample of webgpuRenderHandoffStub?.sampleRecords ?? []) {
    if (seeds.length >= MAX_NATIVE_BRIDGE_SAMPLES) break;
    const colorAlpha = colorFromSample(sample);
    if (!colorAlpha) continue;
    const samplePx = samplePxFromSample(sample, seeds.length, canvasWidth, canvasHeight);
    const coverageAlpha = clamp01(colorAlpha[3], 1);
    const finalTransmittance = 1 - coverageAlpha;
    seeds.push({
      source: 'webgpuRenderHandoffStub.sampleRecords',
      sampleKind: sample?.sampleKind ?? 'render-handoff-native-bounded-seed',
      anchorRecordIndex: sample?.recordIndex ?? sample?.anchorRecordIndex ?? -1,
      recordIndex: sample?.recordIndex ?? null,
      srcIndex: sample?.srcIndex ?? null,
      valid: sample?.valid ?? null,
      samplePx,
      colorAlpha,
      resolvedRgb: colorAlpha.slice(0, 3),
      coverageAlpha,
      finalTransmittance
    });
  }
  return seeds;
}

function makeTileOutputSamples(seeds) {
  return seeds.map((seed, index) => ({
    source: 'webgpuViewerCanvasNativeBoundedColorSamples.step38TileOutputBridge',
    colorSource: 'Step50 native-compatible Step38 bounded tile output from render handoff colorAlpha.rgb',
    tileId: index,
    tileIndexStart: index,
    tileIndexEnd: index + 1,
    tileRefCount: 1,
    sampleKind: 'step50-native-compatible-tile-output',
    anchorRecordIndex: seed.anchorRecordIndex,
    recordIndex: seed.recordIndex,
    srcIndex: seed.srcIndex,
    valid: seed.valid,
    samplePx: seed.samplePx.slice(0, 2),
    colorAlpha: seed.colorAlpha.slice(0, 4),
    expected: {
      resolvedRgb: seed.resolvedRgb.slice(0, 3),
      coverageAlpha: seed.coverageAlpha,
      accumAlpha: seed.coverageAlpha,
      finalTransmittance: seed.finalTransmittance
    },
    actual: {
      resolvedRgb: seed.resolvedRgb.slice(0, 3),
      coverageAlpha: seed.coverageAlpha,
      accumAlpha: seed.coverageAlpha,
      finalTransmittance: seed.finalTransmittance
    }
  }));
}

function makeRenderTargetSamples(tileSamples, canvasWidth, canvasHeight) {
  const width = Math.max(1, Math.round(finiteOrZero(canvasWidth)));
  const height = Math.max(1, Math.round(finiteOrZero(canvasHeight)));
  return tileSamples.map((sample) => {
    const x = Math.min(width - 1, Math.max(0, Math.round(finiteOrZero(sample.samplePx?.[0]))));
    const y = Math.min(height - 1, Math.max(0, Math.round(finiteOrZero(sample.samplePx?.[1]))));
    return {
      source: 'webgpuViewerCanvasNativeBoundedColorSamples.step39RenderTargetBridge',
      colorSource: 'Step50 native-compatible Step39 render target sample from bounded tile output',
      tileId: sample.tileId,
      tileIndexStart: sample.tileIndexStart,
      tileIndexEnd: sample.tileIndexEnd,
      tileRefCount: sample.tileRefCount,
      sampleKind: 'step50-native-compatible-render-target',
      anchorRecordIndex: sample.anchorRecordIndex,
      recordIndex: sample.recordIndex,
      srcIndex: sample.srcIndex,
      valid: sample.valid,
      samplePx: sample.samplePx.slice(0, 2),
      pixel: [x, y],
      colorAlpha: sample.colorAlpha.slice(0, 4),
      expected: {
        pixel: [x, y],
        resolvedRgb: sample.expected.resolvedRgb.slice(0, 3),
        coverageAlpha: sample.expected.coverageAlpha,
        finalTransmittance: sample.expected.finalTransmittance,
        rgbaFloat: sample.colorAlpha.slice(0, 4)
      },
      actual: {
        pixel: [x, y],
        resolvedRgb: sample.actual.resolvedRgb.slice(0, 3),
        coverageAlpha: sample.actual.coverageAlpha,
        finalTransmittance: sample.actual.finalTransmittance,
        rgbaFloat: sample.colorAlpha.slice(0, 4)
      }
    };
  });
}

function makeTextureSamples(renderTargetSamples) {
  return renderTargetSamples.map((sample) => ({
    source: 'webgpuViewerCanvasNativeBoundedColorSamples.step40ConstrainedDisplayBridge',
    colorSource: 'Step50 native-compatible Step40 constrained display sample from render target handoff',
    tileId: sample.tileId,
    sampleKind: 'step50-native-compatible-constrained-display',
    anchorRecordIndex: sample.anchorRecordIndex,
    recordIndex: sample.recordIndex,
    srcIndex: sample.srcIndex,
    valid: sample.valid,
    samplePx: sample.samplePx.slice(0, 2),
    pixel: sample.pixel.slice(0, 2),
    colorAlpha: sample.colorAlpha.slice(0, 4),
    expected: {
      pixel: sample.pixel.slice(0, 2),
      resolvedRgb: sample.expected.resolvedRgb.slice(0, 3),
      coverageAlpha: sample.expected.coverageAlpha,
      finalTransmittance: sample.expected.finalTransmittance,
      rgbaFloat: sample.colorAlpha.slice(0, 4)
    },
    actual: {
      pixel: sample.pixel.slice(0, 2),
      resolvedRgb: sample.actual.resolvedRgb.slice(0, 3),
      coverageAlpha: sample.actual.coverageAlpha,
      finalTransmittance: sample.actual.finalTransmittance,
      rgbaFloat: sample.colorAlpha.slice(0, 4)
    }
  }));
}

export function buildWebGpuViewerCanvasNativeBoundedColorSamples({
  webgpuFramebufferFreeTileOutputDryRunComparison = null,
  webgpuRenderTargetHandoffDryRunComparison = null,
  webgpuConstrainedDisplayAdapterDryRunComparison = null,
  webgpuRenderHandoffStub = null,
  canvasWidth = 0,
  canvasHeight = 0
} = {}) {
  const startMs = nowMs();
  const originalStep38Samples = cloneList(
    webgpuFramebufferFreeTileOutputDryRunComparison?.sampleTileOutputs
  );
  const originalStep39Samples = cloneList(
    webgpuRenderTargetHandoffDryRunComparison?.sampleRenderTargetPixels
  );
  const originalStep40Samples = cloneList(
    webgpuConstrainedDisplayAdapterDryRunComparison?.sampleTexturePixels
  );

  let sampleTileOutputs = originalStep38Samples;
  let sampleRenderTargetPixels = originalStep39Samples;
  let sampleTexturePixels = originalStep40Samples;
  let selectedNativeSourceKind = null;
  let bridgeSeedSourceKind = null;
  let generatedFromRenderHandoff = false;

  if (sampleTexturePixels.length > 0) {
    selectedNativeSourceKind = 'step40-constrained-display-adapter';
  } else if (sampleRenderTargetPixels.length > 0) {
    selectedNativeSourceKind = 'step39-render-target-handoff';
  } else if (sampleTileOutputs.length > 0) {
    selectedNativeSourceKind = 'step38-framebuffer-free-tile-output';
  } else {
    const seedSamples = buildSeedSamples({
      webgpuRenderHandoffStub,
      canvasWidth,
      canvasHeight
    });
    if (seedSamples.length > 0) {
      sampleTileOutputs = makeTileOutputSamples(seedSamples);
      sampleRenderTargetPixels = makeRenderTargetSamples(
        sampleTileOutputs,
        canvasWidth,
        canvasHeight
      );
      sampleTexturePixels = makeTextureSamples(sampleRenderTargetPixels);
      selectedNativeSourceKind = 'step40-constrained-display-adapter-native-bridge';
      bridgeSeedSourceKind = 'render-handoff-reference-assisted-colorAlpha-rgb';
      generatedFromRenderHandoff = true;
    }
  }

  const nativeBoundedSamplesReady =
    sampleTexturePixels.length > 0 ||
    sampleRenderTargetPixels.length > 0 ||
    sampleTileOutputs.length > 0;

  const firstValidationFailures = nativeBoundedSamplesReady
    ? []
    : [
        {
          stage: 'viewer-canvas-native-bounded-color-samples',
          reason:
            'Step38-40 samples and render-handoff reference-assisted color samples are unavailable'
        }
      ];

  return {
    mode: WEBGPU_VIEWER_CANVAS_NATIVE_BOUNDED_COLOR_SAMPLES_MODE,
    status: nativeBoundedSamplesReady ? 'ok' : 'unavailable',
    source:
      'Step50 bounded color samples retained for viewer canvas selector; true Step38-40 samples are preferred before bridge samples',
    nativeBoundedSamplesBridgeImplemented: true,
    nativeBoundedSamplesReady,
    selectedNativeSourceKind,
    bridgeSeedSourceKind,
    generatedFromRenderHandoff,
    sampleTileOutputCount: sampleTileOutputs.length,
    sampleRenderTargetPixelCount: sampleRenderTargetPixels.length,
    sampleTexturePixelCount: sampleTexturePixels.length,
    sampleTileOutputs,
    sampleRenderTargetPixels,
    sampleTexturePixels,
    sourceAvailabilityBeforeBridge: {
      step40ConstrainedDisplayAdapterSamples: countSamples(
        webgpuConstrainedDisplayAdapterDryRunComparison?.sampleTexturePixels
      ),
      step39RenderTargetHandoffSamples: countSamples(
        webgpuRenderTargetHandoffDryRunComparison?.sampleRenderTargetPixels
      ),
      step38FramebufferFreeTileOutputSamples: countSamples(
        webgpuFramebufferFreeTileOutputDryRunComparison?.sampleTileOutputs
      ),
      renderHandoffSampleRecords: countSamples(webgpuRenderHandoffStub?.sampleRecords)
    },
    sourceAvailabilityAfterBridge: {
      step40ConstrainedDisplayAdapterSamples: sampleTexturePixels.length,
      step39RenderTargetHandoffSamples: sampleRenderTargetPixels.length,
      step38FramebufferFreeTileOutputSamples: sampleTileOutputs.length
    },
    nativeSamplePolicy: {
      originalStep38To40SamplesPreferred: true,
      renderHandoffDerivedBridgeKeptAsFallback: true,
      bridgeScope:
        'same runtime/capture bounded samples only; no production display connection or WebGL2 hybrid rendering',
      colorPolicy:
        'SH evaluation remains deferred; bridge preserves reference-assisted colorAlpha.rgb as native-compatible bounded color payload when true Step38-40 samples are unavailable'
    },
    validationSummary: {
      step40NativeSamplesReady: sampleTexturePixels.length > 0,
      step39NativeSamplesReady: sampleRenderTargetPixels.length > 0,
      step38NativeSamplesReady: sampleTileOutputs.length > 0,
      renderHandoffSeedAvailable: countSamples(webgpuRenderHandoffStub?.sampleRecords) > 0,
      nativeBoundedSamplesReady,
      selectedNativeSourceKind,
      firstValidationFailures
    },
    blockers: [
      {
        stage: 'tile-composite-native-accumulation',
        reason:
          'true Step38-40 compute samples are preferred when available; bridge keeps bounded native-shaped color output moving when accumulation samples are absent'
      },
      {
        stage: 'sh-color-evaluation',
        reason: 'WGSL SH/color evaluation parity remains deferred'
      }
    ],
    nextBackendPrototypeStep:
      'expand true tile accumulation samples while keeping viewer canvas selector/present contract unchanged',
    timing: {
      viewerCanvasNativeBoundedColorSamplesMs: nowMs() - startMs
    }
  };
}
