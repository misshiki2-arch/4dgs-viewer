import { computeVisiblePackBaseFloatOffset } from './gpu_buffer_layout_utils.js';

export const WEBGPU_TILE_COMPOSITE_ACCUMULATION_DRY_RUN_COMPARISON_MODE =
  'webgpu-tile-composite-accumulation-dry-run-comparison';

const SAMPLE_FLOATS = 4;
const OUTPUT_FLOATS = 8;
const DEFAULT_EPSILON = 1e-3;
const EARLY_OUT_TRANSMITTANCE = 0.0001;
const MIN_ALPHA = 1.0 / 255.0;

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
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
    mode: WEBGPU_TILE_COMPOSITE_ACCUMULATION_DRY_RUN_COMPARISON_MODE,
    status: 'unavailable',
    reason,
    source: 'webgpuTileCompositeShaderHandoff + webgpuRenderHandoffStub',
    nonDisplayOnly: true,
    implementedInWgsl: true,
    tileCompositeAccumulationComputed: false,
    tileCompositeImplemented: false,
    framebufferImplemented: false,
    displayConnectionAllowed: false,
    anyMismatch: true,
    mismatchClassification: 'tileCompositeAccumulationDryRunUnavailable',
    firstMismatches: [{ kind: 'input', reason }],
    sampleTileAccumulations: []
  };
}

function readPayload(payload, recordIndex) {
  const base = computeVisiblePackBaseFloatOffset(recordIndex);
  return {
    centerPx: [payload[base + 0], payload[base + 1]],
    colorAlpha: [
      payload[base + 4],
      payload[base + 5],
      payload[base + 6],
      payload[base + 7]
    ],
    conic: [payload[base + 8], payload[base + 9], payload[base + 10]]
  };
}

function makeTileSamples(webgpuTileCompositeShaderHandoff, renderPayload, maxTileSamples = 4) {
  const samples = [];
  for (const tile of webgpuTileCompositeShaderHandoff?.sampleTiles ?? []) {
    if (!Number.isFinite(tile?.tileIndexStart) || !Number.isFinite(tile?.tileIndexEnd)) continue;
    if ((tile.tileRefCount ?? 0) <= 0) continue;
    const firstPacket = tile.firstShaderPackets?.find((packet) =>
      Number.isFinite(packet?.recordIndex)
    );
    if (!firstPacket) continue;
    const payload = readPayload(renderPayload, firstPacket.recordIndex | 0);
    samples.push({
      tileId: tile.tileId ?? -1,
      tileIndexStart: tile.tileIndexStart | 0,
      tileIndexEnd: tile.tileIndexEnd | 0,
      tileRefCount: Math.max(0, (tile.tileIndexEnd | 0) - (tile.tileIndexStart | 0)),
      sampleKind: 'first-packet-center',
      anchorRecordIndex: firstPacket.recordIndex | 0,
      samplePx: [payload.centerPx[0], payload.centerPx[1]]
    });
    if (samples.length >= maxTileSamples) break;
  }
  return samples;
}

function evaluateRecordAtSample(payload, recordIndex, sampleX, sampleY) {
  const p = readPayload(payload, recordIndex);
  const dx = p.centerPx[0] - sampleX;
  const dy = p.centerPx[1] - sampleY;
  const power =
    -0.5 * (p.conic[0] * dx * dx + p.conic[2] * dy * dy) -
    p.conic[1] * dx * dy;
  const alpha = Math.min(0.99, p.colorAlpha[3] * Math.exp(power));
  const survives = power <= 0.0 && alpha >= MIN_ALPHA;
  return {
    power,
    alpha,
    survives,
    color: p.colorAlpha.slice(0, 3)
  };
}

function evaluateCpuTileAccumulation({
  renderPayload,
  orderedTileIndices,
  tileIndexStart,
  tileIndexEnd,
  samplePx
}) {
  const accumColor = [0, 0, 0];
  let accumAlpha = 0;
  let transmittance = 1;
  let visitedRefCount = 0;
  let contributingRefCount = 0;
  let earlyOut = false;

  for (let i = tileIndexStart; i < tileIndexEnd; i += 1) {
    if (transmittance < EARLY_OUT_TRANSMITTANCE) {
      earlyOut = true;
      break;
    }
    visitedRefCount += 1;
    const recordIndex = orderedTileIndices[i];
    const evaluation = evaluateRecordAtSample(
      renderPayload,
      recordIndex,
      samplePx[0],
      samplePx[1]
    );
    if (!evaluation.survives) continue;
    contributingRefCount += 1;
    const weightedAlpha = transmittance * evaluation.alpha;
    accumColor[0] += weightedAlpha * evaluation.color[0];
    accumColor[1] += weightedAlpha * evaluation.color[1];
    accumColor[2] += weightedAlpha * evaluation.color[2];
    accumAlpha += weightedAlpha;
    transmittance *= 1.0 - evaluation.alpha;
  }

  return {
    accumColor,
    accumAlpha,
    finalTransmittance: transmittance,
    visitedRefCount,
    contributingRefCount,
    earlyOut
  };
}

function compareAccumulation({ sample, expected, actual, epsilon }) {
  const firstMismatches = [];
  let maxAbsAccumColorDelta = 0;
  let maxAbsAccumAlphaDelta = Math.abs(actual.accumAlpha - expected.accumAlpha);
  let maxAbsTransmittanceDelta =
    Math.abs(actual.finalTransmittance - expected.finalTransmittance);
  let mismatch = false;

  for (let i = 0; i < 3; i += 1) {
    const delta = Math.abs(actual.accumColor[i] - expected.accumColor[i]);
    maxAbsAccumColorDelta = Math.max(maxAbsAccumColorDelta, delta);
    if (delta > epsilon) {
      mismatch = true;
      firstMismatches.push({
        field: 'accumColor',
        component: i,
        expected: expected.accumColor[i],
        actual: actual.accumColor[i],
        absDelta: delta
      });
    }
  }
  if (maxAbsAccumAlphaDelta > epsilon) {
    mismatch = true;
    firstMismatches.push({
      field: 'accumAlpha',
      expected: expected.accumAlpha,
      actual: actual.accumAlpha,
      absDelta: maxAbsAccumAlphaDelta
    });
  }
  if (maxAbsTransmittanceDelta > epsilon) {
    mismatch = true;
    firstMismatches.push({
      field: 'finalTransmittance',
      expected: expected.finalTransmittance,
      actual: actual.finalTransmittance,
      absDelta: maxAbsTransmittanceDelta
    });
  }
  if (actual.visitedRefCount !== expected.visitedRefCount) {
    mismatch = true;
    firstMismatches.push({
      field: 'visitedRefCount',
      expected: expected.visitedRefCount,
      actual: actual.visitedRefCount
    });
  }
  if (actual.contributingRefCount !== expected.contributingRefCount) {
    mismatch = true;
    firstMismatches.push({
      field: 'contributingRefCount',
      expected: expected.contributingRefCount,
      actual: actual.contributingRefCount
    });
  }
  if (actual.earlyOut !== expected.earlyOut) {
    mismatch = true;
    firstMismatches.push({
      field: 'earlyOut',
      expected: expected.earlyOut,
      actual: actual.earlyOut
    });
  }

  return {
    mismatch,
    maxAbsAccumColorDelta,
    maxAbsAccumAlphaDelta,
    maxAbsTransmittanceDelta,
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

export async function buildWebGpuTileCompositeAccumulationDryRunComparison({
  device,
  webgpuTileCompositeShaderHandoff,
  webgpuRenderHandoffStub,
  epsilon = DEFAULT_EPSILON,
  maxTileSamples = 4
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
  const orderedTileIndices = webgpuTileCompositeShaderHandoff.transientOrderedTileIndices;
  if (!(renderPayload instanceof Float32Array)) {
    return buildUnavailable('transient-render-payload-unavailable');
  }
  if (!(orderedTileIndices instanceof Uint32Array)) {
    return buildUnavailable('transient-ordered-tile-indices-unavailable');
  }

  const samples = makeTileSamples(
    webgpuTileCompositeShaderHandoff,
    renderPayload,
    maxTileSamples
  );
  if (samples.length <= 0) {
    return buildUnavailable('tile-accumulation-samples-unavailable');
  }

  const sampleData = new Float32Array(samples.length * SAMPLE_FLOATS);
  samples.forEach((sample, index) => {
    const base = index * SAMPLE_FLOATS;
    sampleData[base + 0] = sample.tileIndexStart;
    sampleData[base + 1] = sample.tileIndexEnd;
    sampleData[base + 2] = sample.samplePx[0];
    sampleData[base + 3] = sample.samplePx[1];
  });

  const shader = device.createShaderModule({
    label: 'phase3-step37-tile-composite-accumulation-dry-run-wgsl',
    code: `
struct Params {
  sampleCount: u32,
  payloadFloatsPerRecord: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> renderPayload: array<f32>;
@group(0) @binding(1) var<storage, read> orderedTileIndices: array<u32>;
@group(0) @binding(2) var<storage, read> samples: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> output: array<vec4f>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let sampleIndex = id.x;
  if (sampleIndex >= params.sampleCount) {
    return;
  }
  let sample = samples[sampleIndex];
  let tileStart = u32(sample.x);
  let tileEnd = u32(sample.y);
  let samplePx = vec2f(sample.z, sample.w);
  var accumColor = vec3f(0.0);
  var accumAlpha = 0.0;
  var transmittance = 1.0;
  var visitedRefCount = 0u;
  var contributingRefCount = 0u;
  var earlyOut = 0u;
  var i = tileStart;

  loop {
    if (i >= tileEnd) {
      break;
    }
    if (transmittance < ${EARLY_OUT_TRANSMITTANCE}) {
      earlyOut = 1u;
      break;
    }
    visitedRefCount += 1u;
    let recordIndex = orderedTileIndices[i];
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
    if (power <= 0.0 && alpha >= ${MIN_ALPHA}) {
      let weightedAlpha = transmittance * alpha;
      accumColor += colorAlpha.rgb * weightedAlpha;
      accumAlpha += weightedAlpha;
      transmittance *= 1.0 - alpha;
      contributingRefCount += 1u;
    }
    i += 1u;
  }
  output[sampleIndex * 2u + 0u] = vec4f(accumColor, accumAlpha);
  output[sampleIndex * 2u + 1u] = vec4f(
    transmittance,
    f32(visitedRefCount),
    f32(contributingRefCount),
    f32(earlyOut)
  );
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
  const orderedIndicesBuffer = createBuffer(device, orderedTileIndices, GPUBufferUsage.STORAGE);
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
      { binding: 1, resource: { buffer: orderedIndicesBuffer } },
      { binding: 2, resource: { buffer: sampleBuffer } },
      { binding: 3, resource: { buffer: outputBuffer } },
      { binding: 4, resource: { buffer: paramsBuffer } }
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

  let accumulationMismatchCount = 0;
  let maxAbsAccumColorDelta = 0;
  let maxAbsAccumAlphaDelta = 0;
  let maxAbsTransmittanceDelta = 0;
  const firstMismatches = [];
  const sampleTileAccumulations = [];
  for (let i = 0; i < samples.length; i += 1) {
    const sample = samples[i];
    const expected = evaluateCpuTileAccumulation({
      renderPayload,
      orderedTileIndices,
      tileIndexStart: sample.tileIndexStart,
      tileIndexEnd: sample.tileIndexEnd,
      samplePx: sample.samplePx
    });
    const actual = {
      accumColor: [
        actualOutput[i * OUTPUT_FLOATS + 0],
        actualOutput[i * OUTPUT_FLOATS + 1],
        actualOutput[i * OUTPUT_FLOATS + 2]
      ],
      accumAlpha: actualOutput[i * OUTPUT_FLOATS + 3],
      finalTransmittance: actualOutput[i * OUTPUT_FLOATS + 4],
      visitedRefCount: Math.round(actualOutput[i * OUTPUT_FLOATS + 5]),
      contributingRefCount: Math.round(actualOutput[i * OUTPUT_FLOATS + 6]),
      earlyOut: actualOutput[i * OUTPUT_FLOATS + 7] > 0.5
    };
    const comparison = compareAccumulation({ sample, expected, actual, epsilon });
    maxAbsAccumColorDelta = Math.max(
      maxAbsAccumColorDelta,
      comparison.maxAbsAccumColorDelta
    );
    maxAbsAccumAlphaDelta = Math.max(
      maxAbsAccumAlphaDelta,
      comparison.maxAbsAccumAlphaDelta
    );
    maxAbsTransmittanceDelta = Math.max(
      maxAbsTransmittanceDelta,
      comparison.maxAbsTransmittanceDelta
    );
    if (comparison.mismatch) {
      accumulationMismatchCount += 1;
      for (const mismatch of comparison.firstMismatches) {
        if (firstMismatches.length < 8) firstMismatches.push(mismatch);
      }
    }
    if (sampleTileAccumulations.length < 8) {
      sampleTileAccumulations.push({
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

  const anyMismatch = accumulationMismatchCount > 0;
  return {
    mode: WEBGPU_TILE_COMPOSITE_ACCUMULATION_DRY_RUN_COMPARISON_MODE,
    status: anyMismatch ? 'mismatch' : 'ok',
    source: 'webgpuTileCompositeShaderHandoff transient orderedTileIndices + webgpuRenderHandoffStub transient renderPayload',
    expectedSource: 'CPU reference front-to-back accumulation over bounded sample tiles',
    actualSource: 'WebGPU compute front-to-back accumulation dry-run over bounded sample tiles',
    nonDisplayOnly: true,
    implementedInWgsl: true,
    tileCompositeAccumulationComputed: true,
    tileCompositeImplemented: false,
    framebufferImplemented: false,
    displayConnectionAllowed: false,
    orderedTileIndicesStoredInJson: false,
    renderPayloadStoredInJson: false,
    shPolicy: {
      requiredForThisDryRun: false,
      status: 'deferred',
      fallbackColorSource: 'reference-assisted colorAlpha.rgb payload',
      unresolved: 'WGSL SH/color evaluation parity'
    },
    accumulationPolicy: {
      samplePolicy: 'bounded non-empty Step35 sampleTiles at first-packet center pixel',
      ordering: 'Step35 orderedTileIndices, ascending depth with recordIndex tie-break',
      equation:
        'front-to-back accumColor += transmittance * colorAlpha.rgb * alpha; transmittance *= (1 - alpha)',
      earlyOutTransmittance: EARLY_OUT_TRANSMITTANCE,
      minAlpha: MIN_ALPHA,
      framebufferConnection: 'deferred'
    },
    anyMismatch,
    mismatchClassification: anyMismatch ? 'tileCompositeAccumulationDryRunMismatch' : 'none',
    sampleTileCount: samples.length,
    accumulationMismatchCount,
    maxAbsAccumColorDelta,
    maxAbsAccumAlphaDelta,
    maxAbsTransmittanceDelta,
    firstMismatches,
    validationSummary: {
      renderPayloadShapeValid:
        renderPayload.length === (webgpuRenderHandoffStub.outputBuffer?.floatCount ?? renderPayload.length),
      orderedTileIndicesShapeValid:
        orderedTileIndices.length ===
          (webgpuTileCompositeShaderHandoff.shaderInputBuffers?.orderedTileIndices?.length ??
            orderedTileIndices.length),
      sampleTileInputValid: samples.length > 0,
      sampleBufferShapeValid: sampleData.length === samples.length * SAMPLE_FLOATS,
      outputShapeValid: actualOutput.length === samples.length * OUTPUT_FLOATS,
      accumulationSamplesValid: samples.length > 0,
      firstValidationFailures: []
    },
    sampleTileAccumulations,
    blockers: [
      { stage: 'framebuffer', reason: 'WebGPU framebuffer/display connection intentionally deferred' },
      { stage: 'tile-composite-production-path', reason: 'This is a bounded non-display compute dry-run, not the production compositor' },
      { stage: 'sh-color-evaluation', reason: 'WGSL SH/color evaluation parity remains deferred' }
    ],
    nextBackendPrototypeStep:
      'promote bounded tile accumulation dry-run toward framebuffer-free tile output validation or SH color parity',
    timing: {
      tileCompositeAccumulationDryRunComputeMs: computeMs,
      tileCompositeAccumulationDryRunReadbackMs: readbackMs,
      tileCompositeAccumulationDryRunComparisonMs: nowMs() - startMs
    }
  };
}
