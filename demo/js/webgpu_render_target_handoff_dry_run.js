export const WEBGPU_RENDER_TARGET_HANDOFF_DRY_RUN_COMPARISON_MODE =
  'webgpu-render-target-handoff-dry-run-comparison';

const SAMPLE_FLOATS = 12;
const OUTPUT_FLOATS = 12;
const DEFAULT_EPSILON = 1e-3;

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function finiteOrZero(value) {
  return Number.isFinite(value) ? value : 0;
}

function finiteOrNull(value) {
  return Number.isFinite(value) ? value : null;
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
    mode: WEBGPU_RENDER_TARGET_HANDOFF_DRY_RUN_COMPARISON_MODE,
    status: 'unavailable',
    reason,
    source: 'webgpuFramebufferFreeTileOutputDryRunComparison',
    nonDisplayOnly: true,
    implementedInWgsl: true,
    renderTargetSamplePackingComputed: false,
    renderTargetHandoffReady: false,
    productionTileCompositeImplemented: false,
    framebufferImplemented: false,
    displayConnectionImplemented: false,
    displayConnectionAllowed: false,
    anyMismatch: true,
    mismatchClassification: 'renderTargetHandoffDryRunUnavailable',
    firstMismatches: [{ kind: 'input', reason }],
    sampleRenderTargetPixels: []
  };
}

function makeRenderTargetSamples({
  webgpuFramebufferFreeTileOutputDryRunComparison,
  canvasWidth,
  canvasHeight,
  maxPixelSamples
}) {
  const samples = [];
  const width = Math.max(1, Math.round(canvasWidth ?? 0));
  const height = Math.max(1, Math.round(canvasHeight ?? 0));
  for (const sample of webgpuFramebufferFreeTileOutputDryRunComparison?.sampleTileOutputs ?? []) {
    const expected = sample?.expected;
    const actual = sample?.actual;
    const px = Array.isArray(sample?.samplePx) ? sample.samplePx : [0, 0];
    if (!expected || !actual) continue;
    if (!Array.isArray(expected.resolvedRgb) || !Array.isArray(actual.resolvedRgb)) continue;
    const pixelX = Math.min(width - 1, Math.max(0, Math.round(finiteOrZero(px[0]))));
    const pixelY = Math.min(height - 1, Math.max(0, Math.round(finiteOrZero(px[1]))));
    samples.push({
      tileId: sample.tileId ?? -1,
      tileIndexStart: sample.tileIndexStart ?? -1,
      tileIndexEnd: sample.tileIndexEnd ?? -1,
      tileRefCount: sample.tileRefCount ?? 0,
      sampleKind: sample.sampleKind ?? 'unknown',
      anchorRecordIndex: sample.anchorRecordIndex ?? -1,
      samplePx: [finiteOrZero(px[0]), finiteOrZero(px[1])],
      pixel: [pixelX, pixelY],
      expected,
      actual
    });
    if (samples.length >= maxPixelSamples) break;
  }
  return { samples, width, height };
}

function packExpectedSample(sample) {
  return {
    pixel: sample.pixel,
    resolvedRgb: [
      finiteOrZero(sample.expected.resolvedRgb?.[0]),
      finiteOrZero(sample.expected.resolvedRgb?.[1]),
      finiteOrZero(sample.expected.resolvedRgb?.[2])
    ],
    coverageAlpha: finiteOrZero(sample.expected.coverageAlpha),
    finalTransmittance: finiteOrZero(sample.expected.finalTransmittance),
    tileId: sample.tileId,
    anchorRecordIndex: sample.anchorRecordIndex
  };
}

function comparePackedSample({ sample, expected, actual, epsilon }) {
  const firstMismatches = [];
  let mismatch = false;
  let maxAbsResolvedColorDelta = 0;
  let maxAbsCoverageAlphaDelta =
    Math.abs(actual.coverageAlpha - expected.coverageAlpha);
  let maxAbsFinalTransmittanceDelta =
    Math.abs(actual.finalTransmittance - expected.finalTransmittance);
  let pixelCoordinateMismatch = false;

  for (let i = 0; i < 2; i += 1) {
    if (actual.pixel[i] !== expected.pixel[i]) {
      pixelCoordinateMismatch = true;
      mismatch = true;
      firstMismatches.push({
        field: 'pixel',
        component: i,
        expected: expected.pixel[i],
        actual: actual.pixel[i],
        absDelta: Math.abs(actual.pixel[i] - expected.pixel[i])
      });
    }
  }

  for (let i = 0; i < 3; i += 1) {
    const delta = Math.abs(actual.resolvedRgb[i] - expected.resolvedRgb[i]);
    maxAbsResolvedColorDelta = Math.max(maxAbsResolvedColorDelta, delta);
    if (delta > epsilon) {
      mismatch = true;
      firstMismatches.push({
        field: 'resolvedRgb',
        component: i,
        expected: expected.resolvedRgb[i],
        actual: actual.resolvedRgb[i],
        absDelta: delta
      });
    }
  }

  if (maxAbsCoverageAlphaDelta > epsilon) {
    mismatch = true;
    firstMismatches.push({
      field: 'coverageAlpha',
      expected: expected.coverageAlpha,
      actual: actual.coverageAlpha,
      absDelta: maxAbsCoverageAlphaDelta
    });
  }
  if (maxAbsFinalTransmittanceDelta > epsilon) {
    mismatch = true;
    firstMismatches.push({
      field: 'finalTransmittance',
      expected: expected.finalTransmittance,
      actual: actual.finalTransmittance,
      absDelta: maxAbsFinalTransmittanceDelta
    });
  }

  return {
    mismatch,
    pixelCoordinateMismatch,
    maxAbsResolvedColorDelta,
    maxAbsCoverageAlphaDelta,
    maxAbsFinalTransmittanceDelta,
    firstMismatches: firstMismatches.map((entry) => ({
      ...entry,
      tileId: sample.tileId,
      sampleKind: sample.sampleKind,
      anchorRecordIndex: sample.anchorRecordIndex,
      samplePx: sample.samplePx
    }))
  };
}

export async function buildWebGpuRenderTargetHandoffDryRunComparison({
  device,
  webgpuFramebufferFreeTileOutputDryRunComparison,
  projectionContract = null,
  canvasWidth = 0,
  canvasHeight = 0,
  epsilon = DEFAULT_EPSILON,
  maxPixelSamples = 8
}) {
  const startMs = nowMs();
  if (!device) return buildUnavailable('webgpu-device-unavailable');
  if (
    !webgpuFramebufferFreeTileOutputDryRunComparison ||
    webgpuFramebufferFreeTileOutputDryRunComparison.status !== 'ok'
  ) {
    return buildUnavailable(
      webgpuFramebufferFreeTileOutputDryRunComparison?.reason ??
        webgpuFramebufferFreeTileOutputDryRunComparison?.status ??
        'framebuffer-free-tile-output-dry-run-unavailable'
    );
  }

  const { samples, width, height } = makeRenderTargetSamples({
    webgpuFramebufferFreeTileOutputDryRunComparison,
    canvasWidth,
    canvasHeight,
    maxPixelSamples
  });
  if (samples.length <= 0) {
    return buildUnavailable('render-target-handoff-samples-unavailable');
  }

  const sampleData = new Float32Array(samples.length * SAMPLE_FLOATS);
  samples.forEach((sample, index) => {
    const base = index * SAMPLE_FLOATS;
    sampleData[base + 0] = sample.pixel[0];
    sampleData[base + 1] = sample.pixel[1];
    sampleData[base + 2] = finiteOrZero(sample.actual.resolvedRgb?.[0]);
    sampleData[base + 3] = finiteOrZero(sample.actual.resolvedRgb?.[1]);
    sampleData[base + 4] = finiteOrZero(sample.actual.resolvedRgb?.[2]);
    sampleData[base + 5] = finiteOrZero(sample.actual.coverageAlpha);
    sampleData[base + 6] = finiteOrZero(sample.actual.finalTransmittance);
    sampleData[base + 7] = sample.tileId;
    sampleData[base + 8] = sample.anchorRecordIndex;
    sampleData[base + 9] = sample.samplePx[0];
    sampleData[base + 10] = sample.samplePx[1];
    sampleData[base + 11] = 0;
  });

  const shader = device.createShaderModule({
    label: 'phase3-step39-render-target-handoff-dry-run-wgsl',
    code: `
struct Params {
  sampleCount: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> samples: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let sampleIndex = id.x;
  if (sampleIndex >= params.sampleCount) {
    return;
  }
  let pixelAndColor0 = samples[sampleIndex * 3u + 0u];
  let color1AndIds = samples[sampleIndex * 3u + 1u];
  let sourcePixel = samples[sampleIndex * 3u + 2u];
  output[sampleIndex * 3u + 0u] = pixelAndColor0;
  output[sampleIndex * 3u + 1u] = color1AndIds;
  output[sampleIndex * 3u + 2u] = sourcePixel;
}`
  });

  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shader,
      entryPoint: 'main'
    }
  });
  const outputData = new Float32Array(samples.length * OUTPUT_FLOATS);
  const sampleBuffer = createBuffer(device, sampleData, GPUBufferUsage.STORAGE);
  const outputBuffer = createBuffer(
    device,
    outputData,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const paramsBuffer = createBuffer(
    device,
    new Uint32Array([samples.length, 0, 0, 0]),
    GPUBufferUsage.UNIFORM
  );
  const readbackBuffer = device.createBuffer({
    size: outputData.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: sampleBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
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
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputData.byteLength);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const computeMs = nowMs() - computeStartMs;

  const readbackStartMs = nowMs();
  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const actualOutput = new Float32Array(readbackBuffer.getMappedRange().slice(0));
  readbackBuffer.unmap();
  const readbackMs = nowMs() - readbackStartMs;

  let samplePixelMismatchCount = 0;
  let pixelCoordinateMismatchCount = 0;
  let maxAbsResolvedColorDelta = 0;
  let maxAbsCoverageAlphaDelta = 0;
  let maxAbsFinalTransmittanceDelta = 0;
  const firstMismatches = [];
  const sampleRenderTargetPixels = [];

  for (let i = 0; i < samples.length; i += 1) {
    const sample = samples[i];
    const expected = packExpectedSample(sample);
    const base = i * OUTPUT_FLOATS;
    const actual = {
      pixel: [
        Math.round(actualOutput[base + 0]),
        Math.round(actualOutput[base + 1])
      ],
      resolvedRgb: [
        actualOutput[base + 2],
        actualOutput[base + 3],
        actualOutput[base + 4]
      ],
      coverageAlpha: actualOutput[base + 5],
      finalTransmittance: actualOutput[base + 6],
      tileId: Math.round(actualOutput[base + 7]),
      anchorRecordIndex: Math.round(actualOutput[base + 8]),
      samplePx: [actualOutput[base + 9], actualOutput[base + 10]]
    };
    const comparison = comparePackedSample({ sample, expected, actual, epsilon });
    maxAbsResolvedColorDelta = Math.max(
      maxAbsResolvedColorDelta,
      comparison.maxAbsResolvedColorDelta
    );
    maxAbsCoverageAlphaDelta = Math.max(
      maxAbsCoverageAlphaDelta,
      comparison.maxAbsCoverageAlphaDelta
    );
    maxAbsFinalTransmittanceDelta = Math.max(
      maxAbsFinalTransmittanceDelta,
      comparison.maxAbsFinalTransmittanceDelta
    );
    if (comparison.mismatch) {
      samplePixelMismatchCount += 1;
      for (const mismatch of comparison.firstMismatches) {
        if (firstMismatches.length < 8) firstMismatches.push(mismatch);
      }
    }
    if (comparison.pixelCoordinateMismatch) pixelCoordinateMismatchCount += 1;
    if (sampleRenderTargetPixels.length < 8) {
      sampleRenderTargetPixels.push({
        tileId: sample.tileId,
        sampleKind: sample.sampleKind,
        anchorRecordIndex: sample.anchorRecordIndex,
        samplePx: sample.samplePx,
        expected,
        actual
      });
    }
  }

  const firstValidationFailures = [];
  if (webgpuFramebufferFreeTileOutputDryRunComparison.status !== 'ok') {
    firstValidationFailures.push({
      stage: 'framebuffer-free-output',
      reason: 'source comparison is not ok'
    });
  }
  if (samples.length <= 0) {
    firstValidationFailures.push({
      stage: 'render-target-samples',
      reason: 'no bounded samples available'
    });
  }
  if (sampleData.length !== samples.length * SAMPLE_FLOATS) {
    firstValidationFailures.push({
      stage: 'sample-buffer',
      reason: 'unexpected sample buffer length'
    });
  }
  if (actualOutput.length !== samples.length * OUTPUT_FLOATS) {
    firstValidationFailures.push({
      stage: 'output-buffer',
      reason: 'unexpected output buffer length'
    });
  }

  const anyMismatch = samplePixelMismatchCount > 0;
  const renderTargetHandoffReady =
    !anyMismatch &&
    firstValidationFailures.length === 0 &&
    webgpuFramebufferFreeTileOutputDryRunComparison.status === 'ok';

  return {
    mode: WEBGPU_RENDER_TARGET_HANDOFF_DRY_RUN_COMPARISON_MODE,
    status: anyMismatch ? 'mismatch' : 'ok',
    source: 'webgpuFramebufferFreeTileOutputDryRunComparison bounded resolved tile output samples',
    expectedSource: 'CPU reference render-target sample packing from expected framebuffer-free tile outputs',
    actualSource: 'WebGPU compute render-target sample packing from WebGPU framebuffer-free tile outputs',
    nonDisplayOnly: true,
    implementedInWgsl: true,
    renderTargetSamplePackingComputed: true,
    renderTargetHandoffReady,
    productionTileCompositeImplemented: false,
    framebufferImplemented: false,
    displayConnectionImplemented: false,
    displayConnectionAllowed: false,
    renderTargetSamplesStoredInJson: false,
    renderTargetContract: {
      role: 'normal-backend render target handoff validation sample buffer',
      coordinateSpace: 'canvas pixel coordinates',
      pixelQuantization: 'round samplePx to nearest integer pixel for bounded validation samples',
      extent: {
        canvasWidth: width,
        canvasHeight: height
      },
      validationSampleLayout:
        'pixel.xy + resolvedRgb + coverageAlpha + finalTransmittance + tile/sample ids',
      futureRenderTargetFormatPolicy:
        'production backend may choose rgba16float/rgba8unorm; this dry-run keeps float32 samples for validation'
    },
    cameraProjectionContract: {
      projectionParamMode: projectionContract?.mode ?? null,
      fixedReferenceMode:
        'current dry-run capture uses fixed reference camera/projection params',
      interactiveViewerMode:
        'compatible when the WebGPU normal backend receives updated per-frame projection uniforms from the camera input adapter',
      threeJsRole: 'camera and OrbitControls remain input adapters, not the WebGPU renderer core',
      sourceSummary: projectionContract ?? null
    },
    shPolicy: {
      requiredForThisDryRun: false,
      status: 'deferred',
      fallbackColorSource: 'reference-assisted colorAlpha.rgb payload',
      displayImpact:
        'initial display experiments can use reference-assisted color; full color parity still needs WGSL SH evaluation'
    },
    anyMismatch,
    mismatchClassification: anyMismatch ? 'renderTargetHandoffDryRunMismatch' : 'none',
    samplePixelCount: samples.length,
    samplePixelMismatchCount,
    pixelCoordinateMismatchCount,
    maxAbsResolvedColorDelta,
    maxAbsCoverageAlphaDelta,
    maxAbsFinalTransmittanceDelta,
    firstMismatches,
    validationSummary: {
      framebufferFreeOutputValid:
        webgpuFramebufferFreeTileOutputDryRunComparison.status === 'ok',
      samplePixelInputValid: samples.length > 0,
      sampleBufferShapeValid: sampleData.length === samples.length * SAMPLE_FLOATS,
      outputBufferShapeValid: actualOutput.length === samples.length * OUTPUT_FLOATS,
      renderTargetSamplesValid: samples.length > 0,
      cameraProjectionContractCompatible:
        typeof projectionContract === 'object' && projectionContract !== null,
      firstValidationFailures
    },
    sampleRenderTargetPixels,
    blockers: [
      {
        stage: 'production-framebuffer',
        reason: 'This validates bounded render target handoff samples only'
      },
      {
        stage: 'display-connection',
        reason: 'No canvas/framebuffer presentation path is connected in Step39'
      },
      {
        stage: 'sh-color-evaluation',
        reason: 'WGSL SH/color evaluation parity remains deferred'
      }
    ],
    nextBackendPrototypeStep:
      'use this render target handoff contract to introduce a constrained WebGPU display experiment or broaden output coverage before presentation',
    timing: {
      renderTargetHandoffDryRunComputeMs: computeMs,
      renderTargetHandoffDryRunReadbackMs: readbackMs,
      renderTargetHandoffDryRunComparisonMs: nowMs() - startMs
    }
  };
}
