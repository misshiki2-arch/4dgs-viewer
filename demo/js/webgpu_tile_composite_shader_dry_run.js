import { computeVisiblePackBaseFloatOffset } from './gpu_buffer_layout_utils.js';

export const WEBGPU_TILE_COMPOSITE_SHADER_DRY_RUN_COMPARISON_MODE =
  'webgpu-tile-composite-shader-dry-run-comparison';

const SAMPLE_FLOATS = 4;
const OUTPUT_FLOATS = 8;
const DEFAULT_EPSILON = 1e-4;

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
    mode: WEBGPU_TILE_COMPOSITE_SHADER_DRY_RUN_COMPARISON_MODE,
    status: 'unavailable',
    reason,
    source: 'webgpuTileCompositeShaderHandoff',
    nonDisplayOnly: true,
    implementedInWgsl: true,
    tileCompositeShaderComputed: false,
    tileCompositeImplemented: false,
    framebufferImplemented: false,
    displayConnectionAllowed: false,
    anyMismatch: true,
    mismatchClassification: 'tileCompositeShaderDryRunUnavailable',
    firstMismatches: [{ kind: 'input', reason }],
    sampleEvaluations: []
  };
}

function readPayload(payload, recordIndex) {
  const base = computeVisiblePackBaseFloatOffset(recordIndex);
  return {
    centerPx: [payload[base + 0], payload[base + 1]],
    radiusPx: payload[base + 2],
    depth: payload[base + 3],
    colorAlpha: [
      payload[base + 4],
      payload[base + 5],
      payload[base + 6],
      payload[base + 7]
    ],
    conic: [payload[base + 8], payload[base + 9], payload[base + 10]],
    miscAabb: [
      payload[base + 12],
      payload[base + 13],
      payload[base + 14],
      payload[base + 15]
    ]
  };
}

function evaluateCpuSample(payload, recordIndex, sampleX, sampleY) {
  const p = readPayload(payload, recordIndex);
  const dx = p.centerPx[0] - sampleX;
  const dy = p.centerPx[1] - sampleY;
  const power =
    -0.5 * (p.conic[0] * dx * dx + p.conic[2] * dy * dy) -
    p.conic[1] * dx * dy;
  const alpha = Math.min(0.99, p.colorAlpha[3] * Math.exp(power));
  const survives = power <= 0.0 && alpha >= (1.0 / 255.0);
  return {
    power,
    alpha,
    survives,
    color: p.colorAlpha.slice(0, 3),
    premultipliedColor: [
      p.colorAlpha[0] * alpha,
      p.colorAlpha[1] * alpha,
      p.colorAlpha[2] * alpha
    ]
  };
}

function makeShaderSamples(webgpuTileCompositeShaderHandoff, renderPayload, maxSamples = 32) {
  const samples = [];
  for (const tile of webgpuTileCompositeShaderHandoff?.sampleTiles ?? []) {
    for (const packet of tile?.firstShaderPackets ?? []) {
      const recordIndex = packet?.recordIndex;
      if (!Number.isFinite(recordIndex)) continue;
      const payload = readPayload(renderPayload, recordIndex | 0);
      const radius = Math.max(0, finiteOrZero(payload.radiusPx));
      const offset = Math.min(8, radius) * 0.5;
      samples.push({
        tileId: tile.tileId ?? -1,
        localIndex: packet.localIndex ?? 0,
        recordIndex: recordIndex | 0,
        sampleKind: 'center',
        samplePx: [payload.centerPx[0], payload.centerPx[1]]
      });
      if (offset > 0 && samples.length < maxSamples) {
        samples.push({
          tileId: tile.tileId ?? -1,
          localIndex: packet.localIndex ?? 0,
          recordIndex: recordIndex | 0,
          sampleKind: 'radius-half-x',
          samplePx: [payload.centerPx[0] + offset, payload.centerPx[1]]
        });
      }
      if (samples.length >= maxSamples) return samples.slice(0, maxSamples);
    }
  }
  return samples.slice(0, maxSamples);
}

function compareSample({
  sample,
  expected,
  actual,
  epsilon
}) {
  const firstMismatches = [];
  let maxAbsPowerDelta = Math.abs(actual.power - expected.power);
  let maxAbsAlphaDelta = Math.abs(actual.alpha - expected.alpha);
  let maxAbsColorDelta = 0;
  let mismatch = false;
  if (maxAbsPowerDelta > epsilon) {
    mismatch = true;
    firstMismatches.push({ field: 'power', expected: expected.power, actual: actual.power, absDelta: maxAbsPowerDelta });
  }
  if (maxAbsAlphaDelta > epsilon) {
    mismatch = true;
    firstMismatches.push({ field: 'alpha', expected: expected.alpha, actual: actual.alpha, absDelta: maxAbsAlphaDelta });
  }
  if (actual.survives !== expected.survives) {
    mismatch = true;
    firstMismatches.push({ field: 'survivesFragment', expected: expected.survives, actual: actual.survives });
  }
  for (let i = 0; i < 3; i += 1) {
    const delta = Math.abs(actual.premultipliedColor[i] - expected.premultipliedColor[i]);
    maxAbsColorDelta = Math.max(maxAbsColorDelta, delta);
    if (delta > epsilon) {
      mismatch = true;
      firstMismatches.push({
        field: 'premultipliedColor',
        component: i,
        expected: expected.premultipliedColor[i],
        actual: actual.premultipliedColor[i],
        absDelta: delta
      });
    }
  }
  return {
    mismatch,
    maxAbsPowerDelta,
    maxAbsAlphaDelta,
    maxAbsColorDelta,
    firstMismatches: firstMismatches.map((entry) => ({
      ...entry,
      tileId: sample.tileId,
      localIndex: sample.localIndex,
      recordIndex: sample.recordIndex,
      sampleKind: sample.sampleKind
    }))
  };
}

export async function buildWebGpuTileCompositeShaderDryRunComparison({
  device,
  webgpuTileCompositeShaderHandoff,
  webgpuRenderHandoffStub,
  epsilon = DEFAULT_EPSILON,
  maxSamples = 32
}) {
  const startMs = nowMs();
  if (!device) return buildUnavailable('webgpu-device-unavailable');
  if (!webgpuTileCompositeShaderHandoff || webgpuTileCompositeShaderHandoff.status !== 'ok') {
    return buildUnavailable(
      webgpuTileCompositeShaderHandoff?.reason ??
        webgpuTileCompositeShaderHandoff?.status ??
        'webgpu-tile-composite-shader-handoff-unavailable'
    );
  }
  if (!webgpuRenderHandoffStub || webgpuRenderHandoffStub.status !== 'ok') {
    return buildUnavailable(
      webgpuRenderHandoffStub?.reason ??
        webgpuRenderHandoffStub?.status ??
        'webgpu-render-handoff-stub-unavailable'
    );
  }
  const renderPayload = webgpuRenderHandoffStub.transientRenderPayload;
  if (!(renderPayload instanceof Float32Array)) {
    return buildUnavailable('transient-render-payload-unavailable');
  }

  const samples = makeShaderSamples(webgpuTileCompositeShaderHandoff, renderPayload, maxSamples);
  if (samples.length <= 0) {
    return buildUnavailable('tile-composite-shader-samples-unavailable');
  }

  const sampleData = new Float32Array(samples.length * SAMPLE_FLOATS);
  samples.forEach((sample, index) => {
    const base = index * SAMPLE_FLOATS;
    sampleData[base + 0] = sample.recordIndex;
    sampleData[base + 1] = sample.samplePx[0];
    sampleData[base + 2] = sample.samplePx[1];
    sampleData[base + 3] = 0;
  });

  const shader = device.createShaderModule({
    label: 'phase3-step36-tile-composite-shader-dry-run-wgsl',
    code: `
struct Params {
  sampleCount: u32,
  payloadFloatsPerRecord: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> renderPayload: array<f32>;
@group(0) @binding(1) var<storage, read> samples: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let sampleIndex = id.x;
  if (sampleIndex >= params.sampleCount) {
    return;
  }
  let sample = samples[sampleIndex];
  let recordIndex = u32(sample.x);
  let samplePx = vec2f(sample.y, sample.z);
  let base = recordIndex * params.payloadFloatsPerRecord;
  let centerPx = vec2f(renderPayload[base + 0u], renderPayload[base + 1u]);
  let colorAlpha = vec4f(
    renderPayload[base + 4u],
    renderPayload[base + 5u],
    renderPayload[base + 6u],
    renderPayload[base + 7u]
  );
  let conic = vec3f(
    renderPayload[base + 8u],
    renderPayload[base + 9u],
    renderPayload[base + 10u]
  );
  let d = centerPx - samplePx;
  let power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
  let alpha = min(0.99, colorAlpha.a * exp(power));
  let survives = select(0.0, 1.0, power <= 0.0 && alpha >= (1.0 / 255.0));
  let premul = colorAlpha.rgb * alpha;
  output[sampleIndex * 2u + 0u] = vec4f(power, alpha, survives, 0.0);
  output[sampleIndex * 2u + 1u] = vec4f(premul, colorAlpha.a);
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
  const payloadBuffer = createBuffer(device, renderPayload, GPUBufferUsage.STORAGE);
  const sampleBuffer = createBuffer(device, sampleData, GPUBufferUsage.STORAGE);
  const outputBuffer = createBuffer(
    device,
    outputData,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const paramsBuffer = createBuffer(
    device,
    new Uint32Array([samples.length, 16, 0, 0]),
    GPUBufferUsage.UNIFORM
  );
  const readbackBuffer = device.createBuffer({
    size: outputData.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: payloadBuffer } },
      { binding: 1, resource: { buffer: sampleBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: paramsBuffer } }
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

  let sampleMismatchCount = 0;
  let maxAbsPowerDelta = 0;
  let maxAbsAlphaDelta = 0;
  let maxAbsColorDelta = 0;
  const firstMismatches = [];
  const sampleEvaluations = [];
  for (let i = 0; i < samples.length; i += 1) {
    const sample = samples[i];
    const expected = evaluateCpuSample(renderPayload, sample.recordIndex, sample.samplePx[0], sample.samplePx[1]);
    const actual = {
      power: actualOutput[i * OUTPUT_FLOATS + 0],
      alpha: actualOutput[i * OUTPUT_FLOATS + 1],
      survives: actualOutput[i * OUTPUT_FLOATS + 2] > 0.5,
      premultipliedColor: [
        actualOutput[i * OUTPUT_FLOATS + 4],
        actualOutput[i * OUTPUT_FLOATS + 5],
        actualOutput[i * OUTPUT_FLOATS + 6]
      ]
    };
    const comparison = compareSample({ sample, expected, actual, epsilon });
    maxAbsPowerDelta = Math.max(maxAbsPowerDelta, comparison.maxAbsPowerDelta);
    maxAbsAlphaDelta = Math.max(maxAbsAlphaDelta, comparison.maxAbsAlphaDelta);
    maxAbsColorDelta = Math.max(maxAbsColorDelta, comparison.maxAbsColorDelta);
    if (comparison.mismatch) {
      sampleMismatchCount += 1;
      for (const mismatch of comparison.firstMismatches) {
        if (firstMismatches.length < 8) firstMismatches.push(mismatch);
      }
    }
    if (sampleEvaluations.length < 8) {
      sampleEvaluations.push({
        tileId: sample.tileId,
        localIndex: sample.localIndex,
        recordIndex: sample.recordIndex,
        sampleKind: sample.sampleKind,
        samplePx: sample.samplePx,
        expected: {
          power: expected.power,
          alpha: expected.alpha,
          survives: expected.survives,
          premultipliedColor: expected.premultipliedColor
        },
        actual
      });
    }
  }

  const anyMismatch = sampleMismatchCount > 0;
  return {
    mode: WEBGPU_TILE_COMPOSITE_SHADER_DRY_RUN_COMPARISON_MODE,
    status: anyMismatch ? 'mismatch' : 'ok',
    source: 'webgpuTileCompositeShaderHandoff + webgpuRenderHandoffStub.transientRenderPayload',
    expectedSource: 'CPU reference shader arithmetic over transient render payload samples',
    actualSource: 'WebGPU compute shader dry-run over transient render payload samples',
    nonDisplayOnly: true,
    implementedInWgsl: true,
    tileCompositeShaderComputed: true,
    tileCompositeImplemented: false,
    framebufferImplemented: false,
    displayConnectionAllowed: false,
    shPolicy: {
      requiredForThisDryRun: false,
      status: 'deferred',
      fallbackColorSource: 'reference-assisted colorAlpha.rgb payload',
      unresolved: 'WGSL SH/color evaluation parity'
    },
    shaderEvaluationPolicy: {
      formula: 'power = -0.5 * (conic.x * dx^2 + conic.z * dy^2) - conic.y * dx * dy; alpha = min(0.99, colorAlpha.a * exp(power))',
      output: 'premultiplied colorAlpha.rgb * alpha plus alpha/survival flag',
      samplePolicy: 'bounded center and radius-half-x samples from Step35 firstShaderPackets'
    },
    anyMismatch,
    mismatchClassification: anyMismatch ? 'tileCompositeShaderDryRunMismatch' : 'none',
    sampleCount: samples.length,
    sampleMismatchCount,
    maxAbsPowerDelta,
    maxAbsAlphaDelta,
    maxAbsColorDelta,
    firstMismatches,
    validationSummary: {
      renderPayloadShapeValid: renderPayload.length === (webgpuRenderHandoffStub.outputBuffer?.floatCount ?? renderPayload.length),
      sampleBufferShapeValid: sampleData.length === samples.length * SAMPLE_FLOATS,
      outputShapeValid: actualOutput.length === samples.length * OUTPUT_FLOATS,
      shaderSamplesValid: samples.length > 0,
      firstValidationFailures: []
    },
    sampleEvaluations,
    blockers: [
      { stage: 'tile-composite-fragment-shader', reason: 'This is a compute dry-run, not framebuffer tile composite execution' },
      { stage: 'sh-color-evaluation', reason: 'WGSL SH/color evaluation parity remains deferred' },
      { stage: 'framebuffer', reason: 'WebGPU framebuffer/display connection intentionally deferred' }
    ],
    nextBackendPrototypeStep: 'webgpu-tile-composite-fragment-or-framebuffer-non-display-validation',
    timing: {
      tileCompositeShaderDryRunComputeMs: computeMs,
      tileCompositeShaderDryRunReadbackMs: readbackMs,
      tileCompositeShaderDryRunComparisonMs: nowMs() - startMs
    }
  };
}
