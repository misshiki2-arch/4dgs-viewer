export const WEBGPU_CANVAS_PRESENTATION_ADAPTER_DRY_RUN_COMPARISON_MODE =
  'webgpu-canvas-presentation-adapter-dry-run-comparison';

const DEFAULT_EPSILON = 1e-3;
const TEXTURE_BYTES_PER_PIXEL = 4;
const TEXTURE_BYTES_PER_ROW_ALIGNMENT = 256;

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function finiteOrZero(value) {
  return Number.isFinite(value) ? value : 0;
}

function alignTo(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
}

function unorm8ToFloat(value) {
  return value / 255;
}

function normalizeCanvasFormat(format) {
  return format === 'bgra8unorm' || format === 'rgba8unorm'
    ? format
    : 'rgba8unorm';
}

function rgbaToTextureBytes(rgba8, textureFormat) {
  if (textureFormat === 'bgra8unorm') {
    return [rgba8[2], rgba8[1], rgba8[0], rgba8[3]];
  }
  return rgba8.slice(0, 4);
}

function textureBytesToRgba(bytes, textureFormat) {
  if (textureFormat === 'bgra8unorm') {
    return [bytes[2], bytes[1], bytes[0], bytes[3]];
  }
  return bytes.slice(0, 4);
}

function createDetachedCanvas(width, height) {
  if (typeof document !== 'undefined' && typeof document.createElement === 'function') {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return { canvas, kind: 'detached-html-canvas' };
  }
  if (typeof OffscreenCanvas !== 'undefined') {
    return {
      canvas: new OffscreenCanvas(width, height),
      kind: 'detached-offscreen-canvas'
    };
  }
  return null;
}

function buildUnavailable(reason, extra = {}) {
  return {
    mode: WEBGPU_CANVAS_PRESENTATION_ADAPTER_DRY_RUN_COMPARISON_MODE,
    status: 'unavailable',
    reason,
    source: 'webgpuGuardedFirstDisplayExperiment',
    boundedCanvasPresentationExperiment: true,
    detachedCanvasPresentationImplemented: false,
    viewerCanvasPresentationImplemented: false,
    productionDisplayConnectionImplemented: false,
    displayConnectionAllowed: false,
    webgl2HybridRenderingAllowed: false,
    contextGetCurrentTextureUsed: false,
    currentTextureWritten: false,
    currentTextureReadbackCompared: false,
    anyMismatch: true,
    mismatchClassification: 'canvasPresentationAdapterDryRunUnavailable',
    firstMismatches: [{ kind: 'input', reason }],
    sampleCanvasPixels: [],
    ...extra
  };
}

function makeCanvasSamples(webgpuGuardedFirstDisplayExperiment, maxPixelSamples) {
  const samples = [];
  const seenPixels = new Set();
  let duplicatePixelCount = 0;
  for (const sample of webgpuGuardedFirstDisplayExperiment?.samplePresentationPixels ?? []) {
    const expected = sample?.expected;
    const actual = sample?.actual;
    if (!expected || !actual) continue;
    if (!Array.isArray(expected.rgba8) || !Array.isArray(actual.rgba8)) continue;
    const pixel = Array.isArray(sample.pixel) ? sample.pixel : [0, 0];
    const pixelX = Math.max(0, Math.round(finiteOrZero(pixel[0])));
    const pixelY = Math.max(0, Math.round(finiteOrZero(pixel[1])));
    const key = `${pixelX},${pixelY}`;
    if (seenPixels.has(key)) duplicatePixelCount += 1;
    seenPixels.add(key);
    samples.push({
      tileId: sample.tileId ?? -1,
      sampleKind: sample.sampleKind ?? 'unknown',
      anchorRecordIndex: sample.anchorRecordIndex ?? -1,
      samplePx: Array.isArray(sample.samplePx) ? sample.samplePx.slice(0, 2) : [0, 0],
      pixel: [pixelX, pixelY],
      expected,
      actual
    });
    if (samples.length >= maxPixelSamples) break;
  }
  return { samples, duplicatePixelCount };
}

function compareCanvasPixel({ sample, expectedRgba8, actualRgba8, epsilon }) {
  const firstMismatches = [];
  let mismatch = false;
  let maxAbsCanvasColorDelta = 0;
  let maxAbsCanvasAlphaDelta = 0;
  for (let i = 0; i < 4; i += 1) {
    const expectedFloat = unorm8ToFloat(expectedRgba8[i]);
    const actualFloat = unorm8ToFloat(actualRgba8[i]);
    const delta = Math.abs(actualFloat - expectedFloat);
    if (i < 3) {
      maxAbsCanvasColorDelta = Math.max(maxAbsCanvasColorDelta, delta);
    } else {
      maxAbsCanvasAlphaDelta = Math.max(maxAbsCanvasAlphaDelta, delta);
    }
    if (delta > epsilon) {
      mismatch = true;
      firstMismatches.push({
        field: i < 3 ? 'resolvedRgb' : 'coverageAlpha',
        component: i < 3 ? i : 0,
        expected: expectedFloat,
        actual: actualFloat,
        expectedU8: expectedRgba8[i],
        actualU8: actualRgba8[i],
        absDelta: delta,
        tileId: sample.tileId,
        sampleKind: sample.sampleKind,
        anchorRecordIndex: sample.anchorRecordIndex,
        pixel: sample.pixel,
        samplePx: sample.samplePx
      });
    }
  }
  return {
    mismatch,
    maxAbsCanvasColorDelta,
    maxAbsCanvasAlphaDelta,
    firstMismatches
  };
}

export async function buildWebGpuCanvasPresentationAdapterDryRunComparison({
  device,
  webgpuGuardedFirstDisplayExperiment,
  epsilon = DEFAULT_EPSILON,
  maxPixelSamples = 8,
  viewerCanvasState = null
}) {
  const startMs = nowMs();
  if (!device) return buildUnavailable('webgpu-device-unavailable');
  if (
    !webgpuGuardedFirstDisplayExperiment ||
    webgpuGuardedFirstDisplayExperiment.status !== 'ok'
  ) {
    return buildUnavailable(
      webgpuGuardedFirstDisplayExperiment?.reason ??
        webgpuGuardedFirstDisplayExperiment?.status ??
        'guarded-first-display-experiment-unavailable'
    );
  }

  const guardContract =
    webgpuGuardedFirstDisplayExperiment.displayGuardContract ?? {};
  const extent = guardContract.outputExtent ?? {};
  const canvasWidth = Math.max(1, Math.round(finiteOrZero(extent.canvasWidth)));
  const canvasHeight = Math.max(1, Math.round(finiteOrZero(extent.canvasHeight)));
  const { samples, duplicatePixelCount } = makeCanvasSamples(
    webgpuGuardedFirstDisplayExperiment,
    maxPixelSamples
  );
  if (samples.length <= 0) {
    return buildUnavailable('canvas-presentation-samples-unavailable');
  }

  const detached = createDetachedCanvas(canvasWidth, canvasHeight);
  if (!detached) {
    return buildUnavailable('detached-canvas-unavailable', {
      detachedCanvasPresentationImplemented: false
    });
  }
  const context = detached.canvas.getContext?.('webgpu') ?? null;
  if (!context) {
    return buildUnavailable('webgpu-canvas-context-unavailable', {
      detachedCanvasKind: detached.kind
    });
  }

  const preferredFormat =
    typeof navigator !== 'undefined' && navigator.gpu?.getPreferredCanvasFormat
      ? navigator.gpu.getPreferredCanvasFormat()
      : 'rgba8unorm';
  const textureFormat = normalizeCanvasFormat(preferredFormat);
  const usage =
    GPUTextureUsage.COPY_DST |
    GPUTextureUsage.COPY_SRC |
    GPUTextureUsage.RENDER_ATTACHMENT;
  context.configure({
    device,
    format: textureFormat,
    usage,
    alphaMode: 'premultiplied'
  });

  const unpaddedBytesPerRow = canvasWidth * TEXTURE_BYTES_PER_PIXEL;
  const bytesPerRow = alignTo(
    unpaddedBytesPerRow,
    TEXTURE_BYTES_PER_ROW_ALIGNMENT
  );
  const textureBytes = new Uint8Array(bytesPerRow * canvasHeight);
  for (const sample of samples) {
    const rgba8 = sample.actual.rgba8.slice(0, 4);
    const encoded = rgbaToTextureBytes(rgba8, textureFormat);
    const offset =
      sample.pixel[1] * bytesPerRow + sample.pixel[0] * TEXTURE_BYTES_PER_PIXEL;
    textureBytes[offset + 0] = encoded[0];
    textureBytes[offset + 1] = encoded[1];
    textureBytes[offset + 2] = encoded[2];
    textureBytes[offset + 3] = encoded[3];
  }

  const readbackBuffer = device.createBuffer({
    size: bytesPerRow * canvasHeight,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  const computeStartMs = nowMs();
  const currentTexture = context.getCurrentTexture();
  device.queue.writeTexture(
    { texture: currentTexture },
    textureBytes,
    {
      bytesPerRow,
      rowsPerImage: canvasHeight
    },
    {
      width: canvasWidth,
      height: canvasHeight
    }
  );
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture: currentTexture },
    {
      buffer: readbackBuffer,
      bytesPerRow,
      rowsPerImage: canvasHeight
    },
    { width: canvasWidth, height: canvasHeight }
  );
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const computeMs = nowMs() - computeStartMs;

  const readbackStartMs = nowMs();
  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const readbackBytes = new Uint8Array(readbackBuffer.getMappedRange().slice(0));
  readbackBuffer.unmap();
  const readbackMs = nowMs() - readbackStartMs;

  let canvasPixelMismatchCount = 0;
  let maxAbsCanvasColorDelta = 0;
  let maxAbsCanvasAlphaDelta = 0;
  const firstMismatches = [];
  const sampleCanvasPixels = [];
  for (const sample of samples) {
    const expectedRgba8 = sample.expected.rgba8.slice(0, 4);
    const offset =
      sample.pixel[1] * bytesPerRow + sample.pixel[0] * TEXTURE_BYTES_PER_PIXEL;
    const actualTextureBytes = [
      readbackBytes[offset + 0],
      readbackBytes[offset + 1],
      readbackBytes[offset + 2],
      readbackBytes[offset + 3]
    ];
    const actualRgba8 = textureBytesToRgba(actualTextureBytes, textureFormat);
    const comparison = compareCanvasPixel({
      sample,
      expectedRgba8,
      actualRgba8,
      epsilon
    });
    maxAbsCanvasColorDelta = Math.max(
      maxAbsCanvasColorDelta,
      comparison.maxAbsCanvasColorDelta
    );
    maxAbsCanvasAlphaDelta = Math.max(
      maxAbsCanvasAlphaDelta,
      comparison.maxAbsCanvasAlphaDelta
    );
    if (comparison.mismatch) {
      canvasPixelMismatchCount += 1;
      for (const mismatch of comparison.firstMismatches) {
        if (firstMismatches.length < 8) firstMismatches.push(mismatch);
      }
    }
    if (sampleCanvasPixels.length < 8) {
      sampleCanvasPixels.push({
        tileId: sample.tileId,
        sampleKind: sample.sampleKind,
        anchorRecordIndex: sample.anchorRecordIndex,
        samplePx: sample.samplePx,
        pixel: sample.pixel,
        expected: {
          rgba8: expectedRgba8,
          rgbaFloat: expectedRgba8.map(unorm8ToFloat)
        },
        actual: {
          rgba8: actualRgba8,
          rgbaFloat: actualRgba8.map(unorm8ToFloat),
          textureBytes: actualTextureBytes
        }
      });
    }
  }

  const viewerCanvasContextMode =
    viewerCanvasState?.contextMode ?? 'not-provided';
  const viewerCanvasPresentationAllowed =
    viewerCanvasState?.allowViewerCanvasPresentation === true &&
    viewerCanvasContextMode !== 'webgl2-active';
  const firstValidationFailures = [];
  if (webgpuGuardedFirstDisplayExperiment.status !== 'ok') {
    firstValidationFailures.push({
      stage: 'guarded-first-display',
      reason: 'source experiment is not ok'
    });
  }
  if (duplicatePixelCount > 0) {
    firstValidationFailures.push({
      stage: 'canvas-write-order',
      reason: 'duplicate sample pixels would make canvas presentation order ambiguous'
    });
  }
  if (viewerCanvasContextMode === 'webgl2-active') {
    firstValidationFailures.push({
      stage: 'viewer-canvas-guard',
      reason:
        'viewer canvas already has a WebGL2 context; Step42 uses detached canvas presentation instead of hybrid display'
    });
  }

  const anyMismatch = canvasPixelMismatchCount > 0 || duplicatePixelCount > 0;
  return {
    mode: WEBGPU_CANVAS_PRESENTATION_ADAPTER_DRY_RUN_COMPARISON_MODE,
    status: anyMismatch ? 'mismatch' : 'ok',
    source: 'webgpuGuardedFirstDisplayExperiment bounded presentation samples',
    expectedSource:
      'CPU quantized rgba8 sample pixels carried through Step41 expected samples',
    actualSource:
      'WebGPU writes bounded samples into a detached canvas current texture, copies it back, and compares sample pixels',
    boundedCanvasPresentationExperiment: true,
    detachedCanvasPresentationImplemented: true,
    detachedCanvasKind: detached.kind,
    viewerCanvasPresentationImplemented: false,
    productionDisplayConnectionImplemented: false,
    displayConnectionAllowed: false,
    contextGetCurrentTextureUsed: true,
    currentTextureWritten: true,
    currentTextureReadbackCompared: true,
    canvasPresentationProbeSucceeded: !anyMismatch,
    viewerCanvasPresentationAllowed,
    framebufferImplemented: false,
    canvasPresentationImplemented: true,
    webgl2HybridRenderingAllowed: false,
    canvasTextureStoredInJson: false,
    canvasPresentationContract: {
      role: 'bounded WebGPU canvas presentation adapter dry-run',
      canvasKind: detached.kind,
      textureFormat,
      requestedPreferredFormat: preferredFormat,
      usage: ['copy_dst', 'copy_src', 'render_attachment'],
      outputExtent: { canvasWidth, canvasHeight },
      sampleWritePolicy:
        'write bounded Step41 pixels into a detached canvas current texture for lifecycle validation',
      viewerCanvasPolicy:
        'do not present to the existing viewer canvas while WebGL2 owns that canvas',
      fixedReferenceMode:
        'inherits Step41 fixed-reference sample pixels for bounded presentation validation',
      interactiveViewerMode:
        'future exclusive WebGPU backend can use the same adapter with per-frame projection uniforms and the viewer canvas current texture',
      webgl2OraclePolicy:
        'WebGL2 remains fallback/regression oracle and is not mixed into this WebGPU canvas presentation path'
    },
    viewerCanvasGuard: {
      viewerCanvasProvided: viewerCanvasState?.provided === true,
      contextMode: viewerCanvasContextMode,
      allowViewerCanvasPresentation:
        viewerCanvasState?.allowViewerCanvasPresentation === true,
      viewerCanvasPresentationAllowed,
      reason:
        viewerCanvasContextMode === 'webgl2-active'
          ? 'exclusive WebGPU backend mode is required before using the viewer canvas current texture'
          : 'viewer canvas presentation remains guarded until explicitly enabled'
    },
    cameraProjectionContract: {
      fixedReferenceCapture:
        'inherits Step41 sample pixels and does not mutate camera/projection data',
      interactiveViewer:
        'no OrbitControls or mouse input changes in Step42; future WebGPU renderer should consume explicit per-frame projection uniforms'
    },
    shPolicy: {
      requiredForThisExperiment: false,
      status: 'deferred',
      fallbackColorSource: 'reference-assisted colorAlpha.rgb payload',
      displayImpact:
        'bounded canvas presentation can validate output plumbing with reference-assisted color; full visual parity still requires WGSL SH/color evaluation or an explicit reference-assisted display mode'
    },
    anyMismatch,
    mismatchClassification: anyMismatch ? 'canvasPresentationAdapterDryRunMismatch' : 'none',
    samplePixelCount: samples.length,
    duplicatePixelCount,
    canvasPixelMismatchCount,
    maxAbsCanvasColorDelta,
    maxAbsCanvasAlphaDelta,
    firstMismatches,
    validationSummary: {
      guardedFirstDisplayValid:
        webgpuGuardedFirstDisplayExperiment.status === 'ok',
      detachedCanvasAvailable: true,
      webgpuCanvasContextAvailable: true,
      currentTextureAvailable: true,
      currentTextureWriteValid: true,
      currentTextureReadbackShapeValid:
        readbackBytes.length === bytesPerRow * canvasHeight,
      duplicatePixelFree: duplicatePixelCount === 0,
      viewerCanvasGuardActive: viewerCanvasPresentationAllowed === false,
      webgl2HybridRenderingPrevented: true,
      cameraProjectionContractUnchanged: true,
      firstValidationFailures
    },
    sampleCanvasPixels,
    blockers: [
      {
        stage: 'viewer-canvas-presentation',
        reason:
          'Step42 validates a detached WebGPU canvas current texture but does not present to the existing viewer canvas'
      },
      {
        stage: 'exclusive-webgpu-backend-frame-lifecycle',
        reason:
          'The viewer canvas can be used after an exclusive WebGPU backend mode owns canvas/context lifecycle'
      },
      {
        stage: 'sh-color-evaluation',
        reason: 'WGSL SH/color evaluation parity remains deferred'
      }
    ],
    nextBackendPrototypeStep:
      'enable a guarded exclusive-WebGPU viewer canvas path that uses the same currentTexture adapter, or keep this adapter detached while adding a normal-backend frame lifecycle switch',
    timing: {
      canvasPresentationAdapterWriteAndCopyMs: computeMs,
      canvasPresentationAdapterReadbackMs: readbackMs,
      canvasPresentationAdapterComparisonMs: nowMs() - startMs
    }
  };
}
