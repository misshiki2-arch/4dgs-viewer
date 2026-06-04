export const WEBGPU_GUARDED_FIRST_DISPLAY_EXPERIMENT_MODE =
  'webgpu-guarded-first-display-experiment';

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

function unorm8ToFloat(value) {
  return value / 255;
}

function buildUnavailable(reason) {
  return {
    mode: WEBGPU_GUARDED_FIRST_DISPLAY_EXPERIMENT_MODE,
    status: 'unavailable',
    reason,
    source: 'webgpuConstrainedDisplayAdapterDryRunComparison',
    guardedFirstDisplayExperiment: true,
    presentationGuardEnabled: true,
    displayExperimentOnly: true,
    sourceTextureWritten: false,
    presentationCandidateTextureWritten: false,
    presentationCopyExecuted: false,
    presentationTextureReadbackCompared: false,
    presentationCandidateReady: false,
    framebufferAdapterImplemented: true,
    framebufferImplemented: false,
    canvasPresentationImplemented: false,
    displayConnectionImplemented: false,
    displayConnectionAllowed: false,
    webgl2HybridRenderingAllowed: false,
    anyMismatch: true,
    mismatchClassification: 'guardedFirstDisplayExperimentUnavailable',
    firstMismatches: [{ kind: 'input', reason }],
    samplePresentationPixels: []
  };
}

function makePresentationSamples(webgpuConstrainedDisplayAdapterDryRunComparison, maxPixelSamples) {
  const samples = [];
  const seenPixels = new Set();
  let duplicatePixelCount = 0;
  for (const sample of webgpuConstrainedDisplayAdapterDryRunComparison?.sampleTexturePixels ?? []) {
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

function comparePresentationPixel({ sample, expectedRgba8, actualRgba8, epsilon }) {
  const firstMismatches = [];
  let mismatch = false;
  let maxAbsPresentationColorDelta = 0;
  let maxAbsPresentationAlphaDelta = 0;
  for (let i = 0; i < 4; i += 1) {
    const expectedFloat = unorm8ToFloat(expectedRgba8[i]);
    const actualFloat = unorm8ToFloat(actualRgba8[i]);
    const delta = Math.abs(actualFloat - expectedFloat);
    if (i < 3) {
      maxAbsPresentationColorDelta = Math.max(maxAbsPresentationColorDelta, delta);
    } else {
      maxAbsPresentationAlphaDelta = Math.max(maxAbsPresentationAlphaDelta, delta);
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
    maxAbsPresentationColorDelta,
    maxAbsPresentationAlphaDelta,
    firstMismatches
  };
}

export async function buildWebGpuGuardedFirstDisplayExperiment({
  device,
  webgpuConstrainedDisplayAdapterDryRunComparison,
  epsilon = DEFAULT_EPSILON,
  maxPixelSamples = 8
}) {
  const startMs = nowMs();
  if (!device) return buildUnavailable('webgpu-device-unavailable');
  if (
    !webgpuConstrainedDisplayAdapterDryRunComparison ||
    webgpuConstrainedDisplayAdapterDryRunComparison.status !== 'ok'
  ) {
    return buildUnavailable(
      webgpuConstrainedDisplayAdapterDryRunComparison?.reason ??
        webgpuConstrainedDisplayAdapterDryRunComparison?.status ??
        'constrained-display-adapter-dry-run-unavailable'
    );
  }

  const extent =
    webgpuConstrainedDisplayAdapterDryRunComparison.displayAdapterContract
      ?.outputExtent ?? {};
  const canvasWidth = Math.max(1, Math.round(finiteOrZero(extent.canvasWidth)));
  const canvasHeight = Math.max(1, Math.round(finiteOrZero(extent.canvasHeight)));
  const { samples, duplicatePixelCount } = makePresentationSamples(
    webgpuConstrainedDisplayAdapterDryRunComparison,
    maxPixelSamples
  );
  if (samples.length <= 0) {
    return buildUnavailable('guarded-first-display-samples-unavailable');
  }

  const sampleData = new Float32Array(samples.length * SAMPLE_FLOATS);
  samples.forEach((sample, index) => {
    const base = index * SAMPLE_FLOATS;
    const rgbaFloat = Array.isArray(sample.actual.rgbaFloat)
      ? sample.actual.rgbaFloat
      : sample.actual.rgba8.map(unorm8ToFloat);
    sampleData[base + 0] = sample.pixel[0];
    sampleData[base + 1] = sample.pixel[1];
    sampleData[base + 2] = clamp01(rgbaFloat[0]);
    sampleData[base + 3] = clamp01(rgbaFloat[1]);
    sampleData[base + 4] = clamp01(rgbaFloat[2]);
    sampleData[base + 5] = clamp01(rgbaFloat[3]);
    sampleData[base + 6] = sample.tileId;
    sampleData[base + 7] = sample.anchorRecordIndex;
    sampleData[base + 8] = finiteOrZero(sample.samplePx?.[0]);
    sampleData[base + 9] = finiteOrZero(sample.samplePx?.[1]);
    sampleData[base + 10] = 0;
    sampleData[base + 11] = 0;
  });

  const shader = device.createShaderModule({
    label: 'phase3-step41-guarded-first-display-source-texture-wgsl',
    code: `
struct Params {
  sampleCount: u32,
  width: u32,
  height: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> samples: array<vec4f>;
@group(0) @binding(1) var sourceTexture: texture_storage_2d<rgba8unorm, write>;
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
  textureStore(sourceTexture, vec2i(i32(x), i32(y)), rgba);
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
  const sourceTexture = device.createTexture({
    size: { width: canvasWidth, height: canvasHeight },
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC
  });
  const presentationTexture = device.createTexture({
    size: { width: canvasWidth, height: canvasHeight },
    format: 'rgba8unorm',
    usage:
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.RENDER_ATTACHMENT
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
      { binding: 1, resource: sourceTexture.createView() },
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
  encoder.copyTextureToTexture(
    { texture: sourceTexture },
    { texture: presentationTexture },
    { width: canvasWidth, height: canvasHeight }
  );
  encoder.copyTextureToBuffer(
    { texture: presentationTexture },
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

  let presentationPixelMismatchCount = 0;
  let maxAbsPresentationColorDelta = 0;
  let maxAbsPresentationAlphaDelta = 0;
  const firstMismatches = [];
  const samplePresentationPixels = [];
  for (const sample of samples) {
    const expectedRgba8 = sample.expected.rgba8.slice(0, 4);
    const offset =
      sample.pixel[1] * bytesPerRow + sample.pixel[0] * TEXTURE_BYTES_PER_PIXEL;
    const actualRgba8 = [
      textureBytes[offset + 0],
      textureBytes[offset + 1],
      textureBytes[offset + 2],
      textureBytes[offset + 3]
    ];
    const comparison = comparePresentationPixel({
      sample,
      expectedRgba8,
      actualRgba8,
      epsilon
    });
    maxAbsPresentationColorDelta = Math.max(
      maxAbsPresentationColorDelta,
      comparison.maxAbsPresentationColorDelta
    );
    maxAbsPresentationAlphaDelta = Math.max(
      maxAbsPresentationAlphaDelta,
      comparison.maxAbsPresentationAlphaDelta
    );
    if (comparison.mismatch) {
      presentationPixelMismatchCount += 1;
      for (const mismatch of comparison.firstMismatches) {
        if (firstMismatches.length < 8) firstMismatches.push(mismatch);
      }
    }
    if (samplePresentationPixels.length < 8) {
      samplePresentationPixels.push({
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
          rgbaFloat: actualRgba8.map(unorm8ToFloat)
        }
      });
    }
  }

  const firstValidationFailures = [];
  if (webgpuConstrainedDisplayAdapterDryRunComparison.status !== 'ok') {
    firstValidationFailures.push({
      stage: 'constrained-display-adapter',
      reason: 'source comparison is not ok'
    });
  }
  if (samples.length <= 0) {
    firstValidationFailures.push({
      stage: 'presentation-samples',
      reason: 'no display adapter samples available'
    });
  }
  if (duplicatePixelCount > 0) {
    firstValidationFailures.push({
      stage: 'presentation-write-order',
      reason:
        'duplicate sample pixels would make first-display presentation order ambiguous'
    });
  }

  const anyMismatch =
    presentationPixelMismatchCount > 0 || duplicatePixelCount > 0;
  return {
    mode: WEBGPU_GUARDED_FIRST_DISPLAY_EXPERIMENT_MODE,
    status: anyMismatch ? 'mismatch' : 'ok',
    source:
      'webgpuConstrainedDisplayAdapterDryRunComparison bounded texture samples',
    expectedSource:
      'CPU quantized rgba8 sample pixels carried through the Step40 display adapter summary',
    actualSource:
      'WebGPU writes a source texture, copies it to a presentation-candidate texture, and reads back bounded pixels',
    guardedFirstDisplayExperiment: true,
    presentationGuardEnabled: true,
    displayExperimentOnly: true,
    sourceTextureWritten: true,
    presentationCandidateTextureWritten: true,
    presentationCopyExecuted: true,
    presentationTextureReadbackCompared: true,
    presentationCandidateReady: !anyMismatch,
    framebufferAdapterImplemented: true,
    framebufferImplemented: false,
    canvasPresentationImplemented: false,
    displayConnectionImplemented: false,
    displayConnectionAllowed: false,
    webgl2HybridRenderingAllowed: false,
    presentationTextureStoredInJson: false,
    displayGuardContract: {
      guardFlag: 'webgpuGuardedFirstDisplayExperiment',
      guardedPath: 'source texture -> presentation candidate texture',
      textureFormat: 'rgba8unorm',
      textureUsage: {
        sourceTexture: ['storage_binding', 'copy_src'],
        presentationCandidateTexture: [
          'copy_dst',
          'copy_src',
          'render_attachment'
        ]
      },
      outputExtent: { canvasWidth, canvasHeight },
      requiresCanvasForRealPresentation: true,
      requiresWebGpuCanvasContextForRealPresentation: true,
      requiresExclusiveWebGpuBackendForRealPresentation: true,
      webgl2HybridRenderingAllowed: false,
      fixedReferenceMode:
        'uses Step40 fixed-reference display adapter samples for guarded validation',
      interactiveViewerMode:
        'future normal backend can replace presentationCandidateTexture with context.getCurrentTexture() after camera uniforms are supplied by the camera input adapter',
      webgl2OraclePolicy:
        'WebGL2 remains a fallback/regression oracle and is not mixed into the WebGPU presentation path'
    },
    cameraProjectionContract: {
      fixedReferenceCapture:
        'inherits Step39/Step40 sample pixels and does not mutate camera/projection data',
      interactiveViewer:
        'no OrbitControls or camera input changes in Step41; future WebGPU renderer should consume explicit projection uniforms'
    },
    shPolicy: {
      requiredForThisExperiment: false,
      status: 'deferred',
      fallbackColorSource: 'reference-assisted colorAlpha.rgb payload',
      displayImpact:
        'guarded first display can use reference-assisted color for initial validation, but full parity still requires WGSL SH/color evaluation or an explicit reference-assisted display mode'
    },
    anyMismatch,
    mismatchClassification: anyMismatch ? 'guardedFirstDisplayExperimentMismatch' : 'none',
    samplePixelCount: samples.length,
    duplicatePixelCount,
    presentationPixelMismatchCount,
    maxAbsPresentationColorDelta,
    maxAbsPresentationAlphaDelta,
    firstMismatches,
    validationSummary: {
      constrainedDisplayAdapterValid:
        webgpuConstrainedDisplayAdapterDryRunComparison.status === 'ok',
      samplePixelInputValid: samples.length > 0,
      sourceTextureShapeValid: canvasWidth > 0 && canvasHeight > 0,
      presentationTextureShapeValid: canvasWidth > 0 && canvasHeight > 0,
      presentationCopyValid: true,
      presentationReadbackShapeValid:
        textureBytes.length === bytesPerRow * canvasHeight,
      duplicatePixelFree: duplicatePixelCount === 0,
      guardFlagRequired: true,
      exclusiveBackendModeRequired: true,
      cameraProjectionContractUnchanged: true,
      firstValidationFailures
    },
    samplePresentationPixels,
    blockers: [
      {
        stage: 'canvas-presentation',
        reason:
          'Step41 validates a presentation-candidate texture but does not call getCurrentTexture() or present to the viewer canvas'
      },
      {
        stage: 'normal-backend-frame-lifecycle',
        reason:
          'A guarded normal-backend frame lifecycle is still needed before production viewer connection'
      },
      {
        stage: 'sh-color-evaluation',
        reason: 'WGSL SH/color evaluation parity remains deferred'
      }
    ],
    nextBackendPrototypeStep:
      'replace the presentation-candidate texture with a guarded WebGPU canvas current texture in an exclusive WebGPU backend experiment, or resolve SH/reference-assisted display policy before doing so',
    timing: {
      guardedFirstDisplayExperimentComputeMs: computeMs,
      guardedFirstDisplayExperimentReadbackMs: readbackMs,
      guardedFirstDisplayExperimentComparisonMs: nowMs() - startMs
    }
  };
}
