import {
  buildWebGpuTileAwareRenderInputContract
} from './common_4dgs_record_contracts.js';

const TILE_INPUT_FLOAT_STRIDE = 12;
const TILE_RECORD_FLOAT_STRIDE = 16;
const TILE_CONSUMER_FLOAT_COUNT = 8;

function finiteNumberOr(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function createBuffer(device, data, usage) {
  const buffer = device.createBuffer({
    size: Math.max(4, Math.ceil(data.byteLength / 4) * 4),
    usage,
    mappedAtCreation: true
  });
  new Uint8Array(buffer.getMappedRange()).set(
    new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
  );
  buffer.unmap();
  return buffer;
}

function packTileInputSamples(samples) {
  const data = new Float32Array(samples.length * TILE_INPUT_FLOAT_STRIDE);
  samples.forEach((sample, index) => {
    const offset = index * TILE_INPUT_FLOAT_STRIDE;
    const footprint = sample?.footprintPayload ?? {};
    const conic = Array.isArray(sample?.conic) ? sample.conic : [];
    data[offset + 0] = finiteNumberOr(sample?.samplePx?.x, 0);
    data[offset + 1] = finiteNumberOr(sample?.samplePx?.y, 0);
    data[offset + 2] = finiteNumberOr(
      footprint.radiusPx ?? sample?.renderAttribute?.radiusPx,
      0
    );
    data[offset + 3] = finiteNumberOr(sample?.depth, 0);
    data[offset + 4] = finiteNumberOr(sample?.recordIndex, index);
    data[offset + 5] = finiteNumberOr(sample?.srcIndex, sample?.recordIndex ?? index);
    data[offset + 6] = finiteNumberOr(conic[0], 0);
    data[offset + 7] = finiteNumberOr(conic[1], 0);
    data[offset + 8] = finiteNumberOr(conic[2], 0);
    data[offset + 9] = finiteNumberOr(sample?.sortKey, sample?.depth ?? 0);
    data[offset + 10] =
      sample?.footprintPayloadSource === 'webgpu-gaussian-footprint-evaluator'
        ? 1
        : 0;
    data[offset + 11] = 0;
  });
  return data;
}

function summarizeTileRecords(records, recordCount) {
  let totalTileReferenceCount = 0;
  let maxTileReferenceCount = 0;
  let validTileRecordCount = 0;
  for (let index = 0; index < recordCount; index += 1) {
    const offset = index * TILE_RECORD_FLOAT_STRIDE;
    const sourceCode = records[offset + 15];
    const tileRefs = records[offset + 12];
    if (sourceCode === 83 && Number.isFinite(tileRefs) && tileRefs > 0) {
      validTileRecordCount += 1;
      totalTileReferenceCount += tileRefs;
      maxTileReferenceCount = Math.max(maxTileReferenceCount, tileRefs);
    }
  }
  return {
    validTileRecordCount,
    totalTileReferenceCount,
    maxTileReferenceCount,
    averageTileReferenceCount:
      validTileRecordCount > 0
        ? totalTileReferenceCount / validTileRecordCount
        : null
  };
}

export async function buildWebGpuTileAwareRenderInput({
  device,
  visibleSamples,
  canvasWidth,
  canvasHeight,
  tileSize = 16
} = {}) {
  const samples = Array.isArray(visibleSamples) ? visibleSamples : [];
  if (
    !device ||
    samples.length <= 0 ||
    typeof GPUBufferUsage === 'undefined' ||
    typeof GPUMapMode === 'undefined' ||
    typeof GPUShaderStage === 'undefined'
  ) {
    return {
      tileRecords: new Float32Array(0),
      consumerSummary: new Float32Array(0),
      contract: buildWebGpuTileAwareRenderInputContract({
        status: 'unavailable',
        reason: 'webgpu-tile-aware-render-input-unavailable'
      })
    };
  }

  const inputData = packTileInputSamples(samples);
  const recordCount = samples.length;
  const tileCols = Math.max(1, Math.ceil(finiteNumberOr(canvasWidth, 1) / tileSize));
  const tileRows = Math.max(1, Math.ceil(finiteNumberOr(canvasHeight, 1) / tileSize));
  const params = new Float32Array([
    recordCount,
    finiteNumberOr(canvasWidth, 1),
    finiteNumberOr(canvasHeight, 1),
    tileSize,
    tileCols,
    tileRows,
    0,
    0
  ]);
  const outputData = new Float32Array(recordCount * TILE_RECORD_FLOAT_STRIDE);
  const consumerData = new Float32Array(TILE_CONSUMER_FLOAT_COUNT);

  const inputBuffer = createBuffer(device, inputData, GPUBufferUsage.STORAGE);
  const paramsBuffer = createBuffer(device, params, GPUBufferUsage.UNIFORM);
  const tileRecordBuffer = createBuffer(
    device,
    outputData,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const consumerSummaryBuffer = createBuffer(
    device,
    consumerData,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const tileRecordReadbackBuffer = device.createBuffer({
    size: outputData.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const consumerReadbackBuffer = device.createBuffer({
    size: consumerData.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const shader = device.createShaderModule({
    label: 'phase3-step83-webgpu-tile-aware-render-input-wgsl',
    code: `
struct Params {
  recordCount: f32,
  canvasWidth: f32,
  canvasHeight: f32,
  tileSize: f32,
  tileCols: f32,
  tileRows: f32,
  _pad0: f32,
  _pad1: f32,
};

@group(0) @binding(0) var<storage, read> tileInputs: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> tileRecords: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> consumerSummary: array<vec4f>;
@group(0) @binding(3) var<uniform> params: Params;

fn clampFloor(value: f32, lo: f32, hi: f32) -> f32 {
  return clamp(floor(value), lo, hi);
}

@compute @workgroup_size(64)
fn generateTileRecords(@builtin(global_invocation_id) id: vec3u) {
  let row = id.x;
  if (row >= u32(params.recordCount)) {
    return;
  }
  let base = row * 3u;
  let a = tileInputs[base + 0u];
  let b = tileInputs[base + 1u];
  let c = tileInputs[base + 2u];
  let px = a.x;
  let py = a.y;
  let radius = max(a.z, 0.0);
  let minX = clampFloor(px - radius, 0.0, max(params.canvasWidth - 1.0, 0.0));
  let minY = clampFloor(py - radius, 0.0, max(params.canvasHeight - 1.0, 0.0));
  let maxX = clamp(ceil(px + radius), 0.0, max(params.canvasWidth - 1.0, 0.0));
  let maxY = clamp(ceil(py + radius), 0.0, max(params.canvasHeight - 1.0, 0.0));
  let minTileX = clampFloor(minX / params.tileSize, 0.0, max(params.tileCols - 1.0, 0.0));
  let minTileY = clampFloor(minY / params.tileSize, 0.0, max(params.tileRows - 1.0, 0.0));
  let maxTileX = clampFloor(maxX / params.tileSize, 0.0, max(params.tileCols - 1.0, 0.0));
  let maxTileY = clampFloor(maxY / params.tileSize, 0.0, max(params.tileRows - 1.0, 0.0));
  let spanX = maxTileX - minTileX + 1.0;
  let spanY = maxTileY - minTileY + 1.0;
  let tileRefs = max(spanX * spanY, 0.0);
  let outBase = row * 4u;
  tileRecords[outBase + 0u] = vec4f(a.w, b.x, minX, minY);
  tileRecords[outBase + 1u] = vec4f(maxX, maxY, minTileX, minTileY);
  tileRecords[outBase + 2u] = vec4f(maxTileX, maxTileY, a.w + 0.0001 * a.w, b.y);
  tileRecords[outBase + 3u] = vec4f(tileRefs, spanX, spanY, 83.0);
}

@compute @workgroup_size(1)
fn consumeTileRecords() {
  var totalRefs = 0.0;
  var maxRefs = 0.0;
  var consumed = 0.0;
  var minSort = 3.402823e38;
  var maxSort = -3.402823e38;
  for (var row: u32 = 0u; row < u32(params.recordCount); row = row + 1u) {
    let base = row * 4u;
    let c = tileRecords[base + 2u];
    let d = tileRecords[base + 3u];
    if (d.w == 83.0 && d.x > 0.0) {
      consumed = consumed + 1.0;
      totalRefs = totalRefs + d.x;
      maxRefs = max(maxRefs, d.x);
      minSort = min(minSort, c.y);
      maxSort = max(maxSort, c.y);
    }
  }
  consumerSummary[0] = vec4f(params.recordCount, consumed, totalRefs, maxRefs);
  consumerSummary[1] = vec4f(select(0.0, minSort, consumed > 0.0), select(0.0, maxSort, consumed > 0.0), 83.0, 0.0);
}
`
  });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' }
      }
    ]
  });
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout]
  });
  const pipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shader, entryPoint: 'generateTileRecords' }
  });
  const consumerPipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shader, entryPoint: 'consumeTileRecords' }
  });
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: tileRecordBuffer } },
      { binding: 2, resource: { buffer: consumerSummaryBuffer } },
      { binding: 3, resource: { buffer: paramsBuffer } }
    ]
  });
  const encoder = device.createCommandEncoder({
    label: 'phase3-step83-webgpu-tile-aware-render-input-encoder'
  });
  const pass = encoder.beginComputePass({
    label: 'phase3-step83-webgpu-tile-aware-render-input-pass'
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(recordCount / 64)));
  pass.setPipeline(consumerPipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyBufferToBuffer(
    tileRecordBuffer,
    0,
    tileRecordReadbackBuffer,
    0,
    outputData.byteLength
  );
  encoder.copyBufferToBuffer(
    consumerSummaryBuffer,
    0,
    consumerReadbackBuffer,
    0,
    consumerData.byteLength
  );
  device.queue.submit([encoder.finish()]);
  if (typeof device.queue.onSubmittedWorkDone === 'function') {
    await device.queue.onSubmittedWorkDone();
  }
  await tileRecordReadbackBuffer.mapAsync(GPUMapMode.READ);
  const tileRecords = new Float32Array(tileRecordReadbackBuffer.getMappedRange().slice(0));
  tileRecordReadbackBuffer.unmap();
  await consumerReadbackBuffer.mapAsync(GPUMapMode.READ);
  const consumerSummary = new Float32Array(consumerReadbackBuffer.getMappedRange().slice(0));
  consumerReadbackBuffer.unmap();

  for (const buffer of [
    inputBuffer,
    paramsBuffer,
    tileRecordBuffer,
    consumerSummaryBuffer,
    tileRecordReadbackBuffer,
    consumerReadbackBuffer
  ]) {
    if (typeof buffer.destroy === 'function') {
      buffer.destroy();
    }
  }

  const recordSummary = summarizeTileRecords(tileRecords, recordCount);
  const consumerConsumedCount = Math.round(finiteNumberOr(consumerSummary[1], 0));
  const consumerTotalRefs = Math.round(finiteNumberOr(consumerSummary[2], 0));
  const tileAwareConsumerConsumed =
    consumerConsumedCount === recordSummary.validTileRecordCount &&
    consumerTotalRefs === recordSummary.totalTileReferenceCount &&
    recordSummary.validTileRecordCount > 0;
  return {
    tileRecords,
    consumerSummary,
    contract: buildWebGpuTileAwareRenderInputContract({
      candidateSampleCount: recordCount,
      generatedTileRecordCount: recordSummary.validTileRecordCount,
      tileAwareConsumerReadbackCompleted: true,
      tileAwareConsumerConsumed,
      tileAwareRenderInputReady: recordSummary.validTileRecordCount > 0,
      tileAwareConsumerReady: tileAwareConsumerConsumed,
      generatedPayloadFields: [
        'gpu-native-aabb',
        'gpu-native-tileRange',
        'tile-record',
        'depth-key',
        'sort-key',
        'tile-reference-count'
      ],
      partialTilePayloadFields: [
        'tile-records-from-visible-record-footprint',
        'depth-key-from-visible-record-depth',
        'sort-key-from-footprint-sort-key'
      ],
      deferredTilePayloadFields: [
        'full-tile-list-scatter',
        'full-depth-sort-dispatch',
        'tile-list-prefix-sum',
        'final-tile-compositor'
      ],
      tileSize,
      tileGrid: { tileCols, tileRows, tileCount: tileCols * tileRows },
      totalTileReferenceCount: recordSummary.totalTileReferenceCount,
      maxTileReferenceCount: recordSummary.maxTileReferenceCount,
      averageTileReferenceCount: recordSummary.averageTileReferenceCount,
      tilePayloadClassification: 'partial-webgpu-tile-aware-render-input',
      fullTileListScatterInWgsl: false,
      fullDepthSortInWgsl: false,
      finalTileCompositorImplemented: false,
      normalBackendFallbackMaintained: true,
      reason: tileAwareConsumerConsumed
        ? null
        : 'webgpu-tile-aware-consumer-did-not-match-generated-record-summary'
    })
  };
}
