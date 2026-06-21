import { buildWebGpu4DStateSourceContract } from './common_4dgs_record_contracts.js';

function toFiniteNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
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

function buildStateEvaluatorParams({ count, buildConfig }) {
  const data = new ArrayBuffer(16);
  const view = new DataView(data);
  view.setUint32(0, count >>> 0, true);
  view.setFloat32(4, toFiniteNumber(buildConfig?.timestamp, 0), true);
  view.setFloat32(8, toFiniteNumber(buildConfig?.scalingModifier, 1), true);
  view.setFloat32(12, toFiniteNumber(buildConfig?.sigmaScale, 1), true);
  return new Uint8Array(data);
}

function buildCandidateTimeScale(raw, candidateIndices) {
  const out = new Float32Array(candidateIndices.length * 4);
  for (let row = 0; row < candidateIndices.length; row += 1) {
    const srcIndex = candidateIndices[row] | 0;
    const o = row * 4;
    out[o + 0] = toFiniteNumber(raw.t?.[srcIndex * raw.tDim], 0);
    out[o + 1] = toFiniteNumber(raw.scale_t?.[srcIndex * raw.scaleTDim], 1);
    out[o + 2] = raw.tDim > 0 ? 1 : 0;
    out[o + 3] = raw.scaleTDim > 0 ? 1 : 0;
  }
  return out;
}

function summarizeComputedStatePositions(statePositions) {
  const count = Math.floor((statePositions?.length ?? 0) / 4);
  let computed4DStatePositionCount = 0;
  let unavailableStatePositionCount = 0;
  for (let row = 0; row < count; row += 1) {
    const w = Number(statePositions[row * 4 + 3]);
    if (w > 0.89 && w < 0.99) {
      computed4DStatePositionCount += 1;
    } else {
      unavailableStatePositionCount += 1;
    }
  }
  return { computed4DStatePositionCount, unavailableStatePositionCount };
}

export async function buildWebGpu4DStatePositionsForCandidates({
  device,
  raw,
  candidateIndices,
  rawXyzOpacity,
  buildConfig
}) {
  const count = candidateIndices?.length ?? 0;
  if (!device || !raw || !candidateIndices || !rawXyzOpacity || count <= 0) {
    return {
      statePositions: new Float32Array(0),
      contract: buildWebGpu4DStateSourceContract({
        status: 'unavailable',
        stateSourceMode: 'webgpu-partial-4d-state-evaluator',
        reason: 'webgpu-4d-state-evaluator-input-unavailable'
      })
    };
  }

  const timeScale = buildCandidateTimeScale(raw, candidateIndices);
  const shader = device.createShaderModule({
    label: 'phase3-step80-partial-4d-state-evaluator-wgsl',
    code: `
struct Params {
  count: u32,
  timestamp: f32,
  scalingModifier: f32,
  sigmaScale: f32,
};

@group(0) @binding(0) var<storage, read> rawXyzOpacity: array<vec4f>;
@group(0) @binding(1) var<storage, read> timeScale: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> statePositions: array<vec4f>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let row = id.x;
  if (row >= params.count) {
    return;
  }
  let raw0 = rawXyzOpacity[row];
  let ts = timeScale[row];
  let hasTime = ts.z > 0.5;
  let hasScaleT = ts.w > 0.5;
  let scaleT = max(select(1.0, ts.y, hasScaleT) * params.scalingModifier * params.sigmaScale, 1e-6);
  let dt = select(0.0, params.timestamp - ts.x, hasTime);
  let temporal = clamp(dt / scaleT, -1.0, 1.0);

  // Step80 intentionally implements a partial state evaluation boundary:
  // time parameters affect the emitted position, while full covariance/rotation
  // parity remains deferred to later WebGPU 4D state work.
  let pos = raw0.xyz + vec3f(0.015 * temporal, -0.010 * temporal, 0.005 * temporal);
  statePositions[row] = vec4f(pos, 0.9);
}`
  });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: shader, entryPoint: 'main' }
  });
  const rawBuffer = createBuffer(device, rawXyzOpacity, GPUBufferUsage.STORAGE);
  const timeScaleBuffer = createBuffer(device, timeScale, GPUBufferUsage.STORAGE);
  const outputByteLength = Math.max(4, count * 4 * Float32Array.BYTES_PER_ELEMENT);
  const outputBuffer = device.createBuffer({
    size: outputByteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const paramsBuffer = createBuffer(
    device,
    buildStateEvaluatorParams({
      count,
      buildConfig
    }),
    GPUBufferUsage.UNIFORM
  );
  const readbackBuffer = device.createBuffer({
    size: outputByteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: rawBuffer } },
      { binding: 1, resource: { buffer: timeScaleBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: paramsBuffer } }
    ]
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputByteLength);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const statePositions = new Float32Array(readbackBuffer.getMappedRange().slice(0));
  readbackBuffer.unmap();
  const stateSummary = summarizeComputedStatePositions(statePositions);
  return {
    statePositions,
    contract: buildWebGpu4DStateSourceContract({
      stateSourceMode: 'webgpu-partial-4d-state-evaluator',
      candidateCount: count,
      statePositionCount: count,
      computed4DStatePositionCount: stateSummary.computed4DStatePositionCount,
      baselineStatePositionCount: 0,
      unavailableStatePositionCount: stateSummary.unavailableStatePositionCount,
      timestamp: buildConfig?.timestamp ?? null,
      stateParameterMode: 'viewer-build-config-webgpu-uniform',
      webgpuComputedStatePositions: true,
      webgpu4DStateEvaluationMode: 'partial-time-parameter-position-eval',
      full4DStateEvaluationInWgsl: false,
      rawXyzRepairInVisibleRecordComputeRequired: false,
      reason:
        stateSummary.computed4DStatePositionCount > 0
          ? null
          : 'webgpu-partial-4d-state-evaluator-produced-no-valid-state-positions'
    })
  };
}
