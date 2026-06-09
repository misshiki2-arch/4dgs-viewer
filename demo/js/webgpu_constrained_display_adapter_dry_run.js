export const WEBGPU_CONSTRAINED_DISPLAY_ADAPTER_DRY_RUN_COMPARISON_MODE =
  'webgpu-constrained-display-adapter-dry-run-comparison';

const SAMPLE_FLOATS = 12;
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

function clamp01(value) {
  return Math.min(1, Math.max(0, finiteOrZero(value)));
}

function alignTo(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
}

function createBuffer(device, data, usage) {
  const buffer = device.createBuffer({
    size: Math.max(4, Math.ceil(data.byteLength / 4) * 4),
    usage,
    mappedAtCreation: true
  });
  const mapped = new Uint8Array(buffer.getMappedRange());
  mapped.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  buffer.unmap();
  return buffer;
}

function buildUnavailable(reason) {
  return {
    mode: WEBGPU_CONSTRAINED_DISPLAY_ADAPTER_DRY_RUN_COMPARISON_MODE,
    status: 'unavailable',
    reason,
    source: 'webgpuRenderTargetHandoffDryRunComparison',
    nonDisplayOnly: true,
    implementedInWgsl: true,
    constrainedDisplayExperiment: true,
    displayAdapterDryRunComputed: false,
    renderTargetTextureWritten: false,
    textureReadbackCompared: false,
    framebufferAdapterImplemented: false,
    framebufferImplemented: false,
    canvasPresentationImplemented: false,
    displayConnectionImplemented: false,
    displayConnectionAllowed: false,
    anyMismatch: true,
    mismatchClassification: 'constrainedDisplayAdapterDryRunUnavailable',
    firstMismatches: [{ kind: 'input', reason }],
    sampleTexturePixels: []
  };
}

function quantizeUnorm8(value) {
  return Math.min(255, Math.max(0, Math.round(clamp01(value) * 255)));
}

function unorm8ToFloat(value) {
  return value / 255;
}

function pixelKey(pixelX, pixelY) {
  return `${pixelX},${pixelY}`;
}

function findDeterministicUniquePixel({
  pixelX,
  pixelY,
  width,
  height,
  seenPixels
}) {
  const clampedX = Math.min(Math.max(0, pixelX), Math.max(0, width - 1));
  const clampedY = Math.min(Math.max(0, pixelY), Math.max(0, height - 1));
  const originalKey = pixelKey(clampedX, clampedY);
  if (!seenPixels.has(originalKey)) {
    return { pixel: [clampedX, clampedY], remapped: false };
  }

  const pixelCount = Math.max(1, width * height);
  const startIndex = clampedY * width + clampedX;
  for (let offset = 1; offset < pixelCount; offset += 1) {
    const index = (startIndex + offset) % pixelCount;
    const candidateX = index % width;
    const candidateY = Math.floor(index / width);
    if (!seenPixels.has(pixelKey(candidateX, candidateY))) {
      return { pixel: [candidateX, candidateY], remapped: true };
    }
  }

  return { pixel: [clampedX, clampedY], remapped: true, unresolved: true };
}

function makeTextureSamples({
  webgpuRenderTargetHandoffDryRunComparison,
  maxPixelSamples,
  canvasWidth,
  canvasHeight
}) {
  const samples = [];
  const seenPixels = new Set();
  let duplicatePixelCountBeforeRemap = 0;
  let duplicatePixelRemapCount = 0;
  let unresolvedDuplicatePixelCount = 0;
  for (const sample of webgpuRenderTargetHandoffDryRunComparison?.sampleRenderTargetPixels ?? []) {
    const expected = sample?.expected;
    const actual = sample?.actual;
    if (!expected || !actual) continue;
    if (!Array.isArray(expected.resolvedRgb) || !Array.isArray(actual.resolvedRgb)) continue;
    const pixel = Array.isArray(actual.pixel) ? actual.pixel : expected.pixel;
    const originalPixelX = Math.max(0, Math.round(finiteOrZero(pixel?.[0])));
    const originalPixelY = Math.max(0, Math.round(finiteOrZero(pixel?.[1])));
    const uniquePixel = findDeterministicUniquePixel({
      pixelX: originalPixelX,
      pixelY: originalPixelY,
      width: canvasWidth,
      height: canvasHeight,
      seenPixels
    });
    if (uniquePixel.remapped) {
      duplicatePixelCountBeforeRemap += 1;
      duplicatePixelRemapCount += 1;
    }
    if (uniquePixel.unresolved) unresolvedDuplicatePixelCount += 1;
    const [pixelX, pixelY] = uniquePixel.pixel;
    seenPixels.add(pixelKey(pixelX, pixelY));
    samples.push({
      tileId: sample.tileId ?? -1,
      sampleKind: sample.sampleKind ?? 'unknown',
      anchorRecordIndex: sample.anchorRecordIndex ?? -1,
      samplePx: Array.isArray(sample.samplePx) ? sample.samplePx.slice(0, 2) : [0, 0],
      pixel: [pixelX, pixelY],
      originalPixel: [originalPixelX, originalPixelY],
      pixelRemappedForUniqueness: uniquePixel.remapped === true,
      expected,
      actual
    });
    if (samples.length >= maxPixelSamples) break;
  }
  return {
    samples,
    duplicatePixelCountBeforeRemap,
    duplicatePixelRemapCount,
    unresolvedDuplicatePixelCount
  };
}

function packExpectedRgba8(sample) {
  return [
    quantizeUnorm8(sample.expected.resolvedRgb?.[0]),
    quantizeUnorm8(sample.expected.resolvedRgb?.[1]),
    quantizeUnorm8(sample.expected.resolvedRgb?.[2]),
    quantizeUnorm8(sample.expected.coverageAlpha)
  ];
}

function compareTexturePixel({ sample, expectedRgba8, actualRgba8, epsilon }) {
  const firstMismatches = [];
  let mismatch = false;
  let maxAbsTextureColorDelta = 0;
  let maxAbsTextureAlphaDelta = 0;
  for (let i = 0; i < 4; i += 1) {
    const expectedFloat = unorm8ToFloat(expectedRgba8[i]);
    const actualFloat = unorm8ToFloat(actualRgba8[i]);
    const delta = Math.abs(actualFloat - expectedFloat);
    if (i < 3) {
      maxAbsTextureColorDelta = Math.max(maxAbsTextureColorDelta, delta);
    } else {
      maxAbsTextureAlphaDelta = Math.max(maxAbsTextureAlphaDelta, delta);
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
        absDelta: delta
      });
    }
  }
  return {
    mismatch,
    maxAbsTextureColorDelta,
    maxAbsTextureAlphaDelta,
    firstMismatches: firstMismatches.map((entry) => ({
      ...entry,
      tileId: sample.tileId,
      sampleKind: sample.sampleKind,
      anchorRecordIndex: sample.anchorRecordIndex,
      pixel: sample.pixel,
      samplePx: sample.samplePx
    }))
  };
}

export async function buildWebGpuConstrainedDisplayAdapterDryRunComparison({
  device,
  webgpuRenderTargetHandoffDryRunComparison,
  epsilon = DEFAULT_EPSILON,
  maxPixelSamples = 8
}) {
  const startMs = nowMs();
  if (!device) return buildUnavailable('webgpu-device-unavailable');
  if (
    !webgpuRenderTargetHandoffDryRunComparison ||
    webgpuRenderTargetHandoffDryRunComparison.status !== 'ok'
  ) {
    return buildUnavailable(
      webgpuRenderTargetHandoffDryRunComparison?.reason ??
        webgpuRenderTargetHandoffDryRunComparison?.status ??
        'render-target-handoff-dry-run-unavailable'
    );
  }

  const extent =
    webgpuRenderTargetHandoffDryRunComparison.renderTargetContract?.extent ?? {};
  const canvasWidth = Math.max(1, Math.round(finiteOrZero(extent.canvasWidth)));
  const canvasHeight = Math.max(1, Math.round(finiteOrZero(extent.canvasHeight)));
  const {
    samples,
    duplicatePixelCountBeforeRemap,
    duplicatePixelRemapCount,
    unresolvedDuplicatePixelCount
  } = makeTextureSamples({
    webgpuRenderTargetHandoffDryRunComparison,
    maxPixelSamples,
    canvasWidth,
    canvasHeight
  });
  if (samples.length <= 0) {
    return buildUnavailable('constrained-display-adapter-samples-unavailable');
  }

  const sampleData = new Float32Array(samples.length * SAMPLE_FLOATS);
  samples.forEach((sample, index) => {
    const base = index * SAMPLE_FLOATS;
    sampleData[base + 0] = sample.pixel[0];
    sampleData[base + 1] = sample.pixel[1];
    sampleData[base + 2] = clamp01(sample.actual.resolvedRgb?.[0]);
    sampleData[base + 3] = clamp01(sample.actual.resolvedRgb?.[1]);
    sampleData[base + 4] = clamp01(sample.actual.resolvedRgb?.[2]);
    sampleData[base + 5] = clamp01(sample.actual.coverageAlpha);
    sampleData[base + 6] = finiteOrZero(sample.actual.finalTransmittance);
    sampleData[base + 7] = sample.tileId;
    sampleData[base + 8] = sample.anchorRecordIndex;
    sampleData[base + 9] = finiteOrZero(sample.samplePx?.[0]);
    sampleData[base + 10] = finiteOrZero(sample.samplePx?.[1]);
    sampleData[base + 11] = 0;
  });

  const shader = device.createShaderModule({
    label: 'phase3-step40-constrained-display-adapter-dry-run-wgsl',
    code: `
struct Params {
  sampleCount: u32,
  width: u32,
  height: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> samples: array<vec4f>;
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let sampleIndex = id.x;
  if (sampleIndex >= params.sampleCount) {
    return;
  }
  let pixelAndColor0 = samples[sampleIndex * 3u + 0u];
  let color1AndIds = samples[sampleIndex * 3u + 1u];
  let x = min(u32(pixelAndColor0.x), params.width - 1u);
  let y = min(u32(pixelAndColor0.y), params.height - 1u);
  let rgba = vec4f(pixelAndColor0.z, pixelAndColor0.w, color1AndIds.x, color1AndIds.y);
  textureStore(outputTexture, vec2i(i32(x), i32(y)), rgba);
}`
  });

  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shader,
      entryPoint: 'main'
    }
  });
  const sampleBuffer = createBuffer(device, sampleData, GPUBufferUsage.STORAGE);
  const paramsBuffer = createBuffer(
    device,
    new Uint32Array([samples.length, canvasWidth, canvasHeight, 0]),
    GPUBufferUsage.UNIFORM
  );
  const texture = device.createTexture({
    size: { width: canvasWidth, height: canvasHeight },
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC
  });
  const unpaddedBytesPerRow = canvasWidth * TEXTURE_BYTES_PER_PIXEL;
  const bytesPerRow = alignTo(
    unpaddedBytesPerRow,
    TEXTURE_BYTES_PER_ROW_ALIGNMENT
  );
  const readbackBuffer = device.createBuffer({
    size: bytesPerRow * canvasHeight,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: sampleBuffer } },
      { binding: 1, resource: texture.createView() },
      { binding: 2, resource: { buffer: paramsBuffer } }
    ]
  });

  const computeStartMs = nowMs();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(samples.length / 64));
  pass.end();
  encoder.copyTextureToBuffer(
    { texture },
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
  const textureBytes = new Uint8Array(readbackBuffer.getMappedRange().slice(0));
  readbackBuffer.unmap();
  const readbackMs = nowMs() - readbackStartMs;

  let texturePixelMismatchCount = 0;
  let maxAbsTextureColorDelta = 0;
  let maxAbsTextureAlphaDelta = 0;
  const firstMismatches = [];
  const sampleTexturePixels = [];
  for (const sample of samples) {
    const expectedRgba8 = packExpectedRgba8(sample);
    const offset =
      sample.pixel[1] * bytesPerRow + sample.pixel[0] * TEXTURE_BYTES_PER_PIXEL;
    const actualRgba8 = [
      textureBytes[offset + 0],
      textureBytes[offset + 1],
      textureBytes[offset + 2],
      textureBytes[offset + 3]
    ];
    const comparison = compareTexturePixel({
      sample,
      expectedRgba8,
      actualRgba8,
      epsilon
    });
    maxAbsTextureColorDelta = Math.max(
      maxAbsTextureColorDelta,
      comparison.maxAbsTextureColorDelta
    );
    maxAbsTextureAlphaDelta = Math.max(
      maxAbsTextureAlphaDelta,
      comparison.maxAbsTextureAlphaDelta
    );
    if (comparison.mismatch) {
      texturePixelMismatchCount += 1;
      for (const mismatch of comparison.firstMismatches) {
        if (firstMismatches.length < 8) firstMismatches.push(mismatch);
      }
    }
    if (sampleTexturePixels.length < 8) {
      sampleTexturePixels.push({
        tileId: sample.tileId,
        sampleKind: sample.sampleKind,
        anchorRecordIndex: sample.anchorRecordIndex,
        samplePx: sample.samplePx,
        pixel: sample.pixel,
        originalPixel: sample.originalPixel,
        pixelRemappedForUniqueness: sample.pixelRemappedForUniqueness,
        expected: {
          rgba8: expectedRgba8,
          rgbaFloat: expectedRgba8.map(unorm8ToFloat)
        },
        actual: {
          rgba8: actualRgba8,
          rgbaFloat: actualRgba8.map(unorm8ToFloat)
        }
      });
    }
  }

  const firstValidationFailures = [];
  if (webgpuRenderTargetHandoffDryRunComparison.status !== 'ok') {
    firstValidationFailures.push({
      stage: 'render-target-handoff',
      reason: 'source comparison is not ok'
    });
  }
  if (samples.length <= 0) {
    firstValidationFailures.push({
      stage: 'texture-samples',
      reason: 'no render target samples available'
    });
  }
  if (unresolvedDuplicatePixelCount > 0) {
    firstValidationFailures.push({
      stage: 'texture-write-order',
      reason: 'duplicate sample pixels could not be remapped before parallel texture writes'
    });
  }

  const anyMismatch = texturePixelMismatchCount > 0 || unresolvedDuplicatePixelCount > 0;
  return {
    mode: WEBGPU_CONSTRAINED_DISPLAY_ADAPTER_DRY_RUN_COMPARISON_MODE,
    status: anyMismatch ? 'mismatch' : 'ok',
    source: 'webgpuRenderTargetHandoffDryRunComparison bounded render target samples',
    expectedSource: 'CPU quantized rgba8 render target sample values from Step39 expected samples',
    actualSource: 'WebGPU compute writes Step39 actual samples into an rgba8unorm storage texture and reads back sample pixels',
    nonDisplayOnly: true,
    implementedInWgsl: true,
    constrainedDisplayExperiment: true,
    displayAdapterDryRunComputed: true,
    renderTargetTextureWritten: true,
    textureReadbackCompared: true,
    framebufferAdapterImplemented: true,
    framebufferImplemented: false,
    canvasPresentationImplemented: false,
    displayConnectionImplemented: false,
    displayConnectionAllowed: false,
    textureStoredInJson: false,
    displayAdapterContract: {
      role: 'bounded normal-backend display adapter dry-run',
      textureFormat: 'rgba8unorm',
      textureUsage: ['storage_binding', 'copy_src'],
      outputExtent: { canvasWidth, canvasHeight },
      bytesPerPixel: TEXTURE_BYTES_PER_PIXEL,
      bytesPerRow,
      presentationPolicy: 'no canvas presentation in Step40',
      fixedReferenceMode:
        'uses the Step39 fixed-reference render target samples for validation',
      interactiveViewerMode:
        'same adapter can accept per-frame render target samples after projection uniforms are updated by the camera input adapter'
    },
    shPolicy: {
      requiredForThisDryRun: false,
      status: 'deferred',
      fallbackColorSource: 'reference-assisted colorAlpha.rgb payload',
      displayImpact:
        'bounded display adapter can validate texture writes now; full visual parity still needs WGSL SH/color evaluation or an explicit reference-assisted display mode'
    },
    anyMismatch,
    mismatchClassification: anyMismatch ? 'constrainedDisplayAdapterDryRunMismatch' : 'none',
    samplePixelCount: samples.length,
    duplicatePixelCount: unresolvedDuplicatePixelCount,
    duplicatePixelCountBeforeRemap,
    duplicatePixelRemapCount,
    duplicatePixelPolicy:
      'deterministic-remap-before-parallel-texture-write',
    texturePixelMismatchCount,
    maxAbsTextureColorDelta,
    maxAbsTextureAlphaDelta,
    firstMismatches,
    validationSummary: {
      renderTargetHandoffValid:
        webgpuRenderTargetHandoffDryRunComparison.status === 'ok',
      samplePixelInputValid: samples.length > 0,
      textureExtentValid: canvasWidth > 0 && canvasHeight > 0,
      textureReadbackShapeValid:
        textureBytes.length === bytesPerRow * canvasHeight,
      duplicatePixelFree: unresolvedDuplicatePixelCount === 0,
      duplicatePixelCountBeforeRemap,
      duplicatePixelRemapCount,
      duplicatePixelPolicy:
        'deterministic-remap-before-parallel-texture-write',
      displayAdapterSamplesValid: samples.length > 0,
      firstValidationFailures
    },
    sampleTexturePixels,
    blockers: [
      {
        stage: 'canvas-presentation',
        reason: 'Step40 writes and reads a GPU texture but does not present it to the viewer canvas'
      },
      {
        stage: 'production-framebuffer-lifecycle',
        reason: 'Texture allocation and frame lifecycle are still dry-run only'
      },
      {
        stage: 'sh-color-evaluation',
        reason: 'WGSL SH/color evaluation parity remains deferred'
      }
    ],
    nextBackendPrototypeStep:
      'connect this constrained display adapter to a guarded WebGPU canvas presentation path or resolve the SH/reference-assisted color policy for first display',
    timing: {
      constrainedDisplayAdapterDryRunComputeMs: computeMs,
      constrainedDisplayAdapterDryRunReadbackMs: readbackMs,
      constrainedDisplayAdapterDryRunComparisonMs: nowMs() - startMs
    }
  };
}
