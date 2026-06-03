export const WEBGPU_FRAMEBUFFER_FREE_TILE_OUTPUT_DRY_RUN_COMPARISON_MODE =
  'webgpu-framebuffer-free-tile-output-dry-run-comparison';

const SAMPLE_FLOATS = 8;
const OUTPUT_FLOATS = 8;
const DEFAULT_EPSILON = 1e-3;

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function finiteOrZero(value) {
  return Number.isFinite(value) ? value : 0;
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
    mode: WEBGPU_FRAMEBUFFER_FREE_TILE_OUTPUT_DRY_RUN_COMPARISON_MODE,
    status: 'unavailable',
    reason,
    source: 'webgpuTileCompositeAccumulationDryRunComparison',
    nonDisplayOnly: true,
    implementedInWgsl: true,
    tileOutputPackingComputed: false,
    framebufferFreeOutputComputed: false,
    productionTileCompositeImplemented: false,
    framebufferImplemented: false,
    displayConnectionAllowed: false,
    anyMismatch: true,
    mismatchClassification: 'framebufferFreeTileOutputDryRunUnavailable',
    firstMismatches: [{ kind: 'input', reason }],
    sampleTileOutputs: []
  };
}

function makeOutputSamples(webgpuTileCompositeAccumulationDryRunComparison, maxTileSamples = 8) {
  const samples = [];
  for (const sample of webgpuTileCompositeAccumulationDryRunComparison?.sampleTileAccumulations ?? []) {
    const actual = sample?.actual;
    const expected = sample?.expected;
    if (!actual || !expected) continue;
    if (!Array.isArray(actual.accumColor) || !Array.isArray(expected.accumColor)) continue;
    samples.push({
      tileId: sample.tileId ?? -1,
      tileIndexStart: sample.tileIndexStart ?? -1,
      tileIndexEnd: sample.tileIndexEnd ?? -1,
      tileRefCount: sample.tileRefCount ?? 0,
      sampleKind: sample.sampleKind ?? 'unknown',
      anchorRecordIndex: sample.anchorRecordIndex ?? -1,
      samplePx: Array.isArray(sample.samplePx) ? sample.samplePx.slice(0, 2) : [0, 0],
      expected,
      actual
    });
    if (samples.length >= maxTileSamples) break;
  }
  return samples;
}

function packResolvedOutput(accumulation, bgGray01) {
  const finalTransmittance = finiteOrZero(accumulation.finalTransmittance);
  const resolvedRgb = [
    finiteOrZero(accumulation.accumColor?.[0]) + finalTransmittance * bgGray01,
    finiteOrZero(accumulation.accumColor?.[1]) + finalTransmittance * bgGray01,
    finiteOrZero(accumulation.accumColor?.[2]) + finalTransmittance * bgGray01
  ];
  return {
    resolvedRgb,
    coverageAlpha: 1.0 - finalTransmittance,
    accumAlpha: finiteOrZero(accumulation.accumAlpha),
    finalTransmittance
  };
}

function compareTileOutput({ sample, expected, actual, epsilon }) {
  const firstMismatches = [];
  let mismatch = false;
  let maxAbsResolvedColorDelta = 0;
  let maxAbsCoverageAlphaDelta =
    Math.abs(actual.coverageAlpha - expected.coverageAlpha);
  let maxAbsFinalTransmittanceDelta =
    Math.abs(actual.finalTransmittance - expected.finalTransmittance);

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
    maxAbsResolvedColorDelta,
    maxAbsCoverageAlphaDelta,
    maxAbsFinalTransmittanceDelta,
    firstMismatches: firstMismatches.map((entry) => ({
      ...entry,
      tileId: sample.tileId,
      tileIndexStart: sample.tileIndexStart,
      tileIndexEnd: sample.tileIndexEnd,
      sampleKind: sample.sampleKind,
      anchorRecordIndex: sample.anchorRecordIndex
    }))
  };
}

export async function buildWebGpuFramebufferFreeTileOutputDryRunComparison({
  device,
  webgpuTileCompositeAccumulationDryRunComparison,
  epsilon = DEFAULT_EPSILON,
  bgGray01 = 0,
  maxTileSamples = 8
}) {
  const startMs = nowMs();
  if (!device) return buildUnavailable('webgpu-device-unavailable');
  if (
    !webgpuTileCompositeAccumulationDryRunComparison ||
    webgpuTileCompositeAccumulationDryRunComparison.status !== 'ok'
  ) {
    return buildUnavailable(
      webgpuTileCompositeAccumulationDryRunComparison?.reason ??
        webgpuTileCompositeAccumulationDryRunComparison?.status ??
        'webgpu-tile-composite-accumulation-dry-run-unavailable'
    );
  }

  const samples = makeOutputSamples(
    webgpuTileCompositeAccumulationDryRunComparison,
    maxTileSamples
  );
  if (samples.length <= 0) {
    return buildUnavailable('framebuffer-free-tile-output-samples-unavailable');
  }

  const sampleData = new Float32Array(samples.length * SAMPLE_FLOATS);
  samples.forEach((sample, index) => {
    const base = index * SAMPLE_FLOATS;
    sampleData[base + 0] = finiteOrZero(sample.actual.accumColor?.[0]);
    sampleData[base + 1] = finiteOrZero(sample.actual.accumColor?.[1]);
    sampleData[base + 2] = finiteOrZero(sample.actual.accumColor?.[2]);
    sampleData[base + 3] = finiteOrZero(sample.actual.accumAlpha);
    sampleData[base + 4] = finiteOrZero(sample.actual.finalTransmittance);
    sampleData[base + 5] = bgGray01;
    sampleData[base + 6] = sample.tileId;
    sampleData[base + 7] = 0;
  });

  const shader = device.createShaderModule({
    label: 'phase3-step38-framebuffer-free-tile-output-dry-run-wgsl',
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
  let accum = samples[sampleIndex * 2u + 0u];
  let tileSampleInfo = samples[sampleIndex * 2u + 1u];
  let bgGray01 = tileSampleInfo.y;
  let finalTransmittance = tileSampleInfo.x;
  let resolvedRgb = accum.rgb + vec3f(bgGray01) * finalTransmittance;
  let coverageAlpha = 1.0 - finalTransmittance;
  output[sampleIndex * 2u + 0u] = vec4f(resolvedRgb, coverageAlpha);
  output[sampleIndex * 2u + 1u] = vec4f(accum.a, finalTransmittance, tileSampleInfo.z, 0.0);
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

  let tileOutputMismatchCount = 0;
  let maxAbsResolvedColorDelta = 0;
  let maxAbsCoverageAlphaDelta = 0;
  let maxAbsFinalTransmittanceDelta = 0;
  const firstMismatches = [];
  const sampleTileOutputs = [];
  for (let i = 0; i < samples.length; i += 1) {
    const sample = samples[i];
    const expected = packResolvedOutput(sample.expected, bgGray01);
    const actual = {
      resolvedRgb: [
        actualOutput[i * OUTPUT_FLOATS + 0],
        actualOutput[i * OUTPUT_FLOATS + 1],
        actualOutput[i * OUTPUT_FLOATS + 2]
      ],
      coverageAlpha: actualOutput[i * OUTPUT_FLOATS + 3],
      accumAlpha: actualOutput[i * OUTPUT_FLOATS + 4],
      finalTransmittance: actualOutput[i * OUTPUT_FLOATS + 5],
      tileIdEcho: Math.round(actualOutput[i * OUTPUT_FLOATS + 6])
    };
    const comparison = compareTileOutput({ sample, expected, actual, epsilon });
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
      tileOutputMismatchCount += 1;
      for (const mismatch of comparison.firstMismatches) {
        if (firstMismatches.length < 8) firstMismatches.push(mismatch);
      }
    }
    if (sampleTileOutputs.length < 8) {
      sampleTileOutputs.push({
        tileId: sample.tileId,
        tileIndexStart: sample.tileIndexStart,
        tileIndexEnd: sample.tileIndexEnd,
        tileRefCount: sample.tileRefCount,
        sampleKind: sample.sampleKind,
        anchorRecordIndex: sample.anchorRecordIndex,
        samplePx: sample.samplePx,
        expected,
        actual
      });
    }
  }

  const anyMismatch = tileOutputMismatchCount > 0;
  return {
    mode: WEBGPU_FRAMEBUFFER_FREE_TILE_OUTPUT_DRY_RUN_COMPARISON_MODE,
    status: anyMismatch ? 'mismatch' : 'ok',
    source: 'webgpuTileCompositeAccumulationDryRunComparison bounded sample tile accumulations',
    expectedSource: 'CPU reference framebuffer-free output packing from expected accumulation samples',
    actualSource: 'WebGPU compute framebuffer-free output packing from WebGPU accumulation samples',
    nonDisplayOnly: true,
    implementedInWgsl: true,
    tileOutputPackingComputed: true,
    framebufferFreeOutputComputed: true,
    productionTileCompositeImplemented: false,
    framebufferImplemented: false,
    displayConnectionAllowed: false,
    tileOutputStoredInJson: false,
    shPolicy: {
      requiredForThisDryRun: false,
      status: 'deferred',
      fallbackColorSource: 'reference-assisted colorAlpha.rgb payload',
      unresolved: 'WGSL SH/color evaluation parity'
    },
    outputPackingPolicy: {
      samplePolicy: 'bounded Step37 sampleTileAccumulations',
      resolvedRgb: 'accumColor + finalTransmittance * bgGray01',
      coverageAlpha: '1 - finalTransmittance',
      bgGray01,
      framebufferConnection: 'deferred'
    },
    anyMismatch,
    mismatchClassification: anyMismatch ? 'framebufferFreeTileOutputDryRunMismatch' : 'none',
    sampleTileCount: samples.length,
    tileOutputMismatchCount,
    maxAbsResolvedColorDelta,
    maxAbsCoverageAlphaDelta,
    maxAbsFinalTransmittanceDelta,
    firstMismatches,
    validationSummary: {
      accumulationSummaryValid:
        webgpuTileCompositeAccumulationDryRunComparison.status === 'ok',
      sampleTileInputValid: samples.length > 0,
      sampleBufferShapeValid: sampleData.length === samples.length * SAMPLE_FLOATS,
      outputBufferShapeValid: actualOutput.length === samples.length * OUTPUT_FLOATS,
      tileOutputSamplesValid: samples.length > 0,
      firstValidationFailures: []
    },
    sampleTileOutputs,
    blockers: [
      { stage: 'framebuffer', reason: 'WebGPU framebuffer/display connection intentionally deferred' },
      { stage: 'production-tile-composite-output', reason: 'This validates bounded framebuffer-free tile output packing only' },
      { stage: 'sh-color-evaluation', reason: 'WGSL SH/color evaluation parity remains deferred' }
    ],
    nextBackendPrototypeStep:
      'promote framebuffer-free tile output validation toward broader tile samples or resolve SH color parity before display connection',
    timing: {
      framebufferFreeTileOutputDryRunComputeMs: computeMs,
      framebufferFreeTileOutputDryRunReadbackMs: readbackMs,
      framebufferFreeTileOutputDryRunComparisonMs: nowMs() - startMs
    }
  };
}
