import {
  buildWebGpuGpuOwnedTileListLayoutContract
} from './common_4dgs_record_contracts.js';

const TILE_INPUT_FLOAT_STRIDE = 12;
const TILE_TABLE_FLOAT_STRIDE = 4;
const REFERENCE_FLOAT_STRIDE = 4;
const CONSUMER_SUMMARY_FLOAT_COUNT = 12;

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
    const colorAlpha = Array.isArray(sample?.colorAlpha)
      ? sample.colorAlpha
      : [
          sample?.colorAlpha?.r,
          sample?.colorAlpha?.g,
          sample?.colorAlpha?.b,
          sample?.colorAlpha?.a
        ];
    data[offset + 0] = finiteNumberOr(sample?.samplePx?.x, 0);
    data[offset + 1] = finiteNumberOr(sample?.samplePx?.y, 0);
    data[offset + 2] = finiteNumberOr(
      footprint.radiusPx ?? sample?.renderAttribute?.radiusPx,
      0
    );
    data[offset + 3] = finiteNumberOr(sample?.depth, 0);
    data[offset + 4] = finiteNumberOr(sample?.recordIndex, index);
    data[offset + 5] = finiteNumberOr(sample?.srcIndex, sample?.recordIndex ?? index);
    data[offset + 6] = finiteNumberOr(sample?.sortKey, sample?.depth ?? 0);
    data[offset + 7] =
      sample?.footprintPayloadSource === 'webgpu-gaussian-footprint-evaluator'
        ? 1
        : 0;
    data[offset + 8] = finiteNumberOr(colorAlpha[0], 1);
    data[offset + 9] = finiteNumberOr(colorAlpha[1], 0);
    data[offset + 10] = finiteNumberOr(colorAlpha[2], 0);
    data[offset + 11] = finiteNumberOr(colorAlpha[3], 1);
  });
  return data;
}

function readConsumerSummary(summary) {
  return {
    tileCount: Math.round(finiteNumberOr(summary[0], 0)),
    nonEmptyTileCount: Math.round(finiteNumberOr(summary[1], 0)),
    totalTileReferenceCount: Math.round(finiteNumberOr(summary[2], 0)),
    maxRefsPerTileObserved: Math.round(finiteNumberOr(summary[3], 0)),
    overflowCount: Math.round(finiteNumberOr(summary[4], 0)),
    consumedTileReferenceCount: Math.round(finiteNumberOr(summary[5], 0)),
    consumerFollowedOffsetCountTable:
      Math.round(finiteNumberOr(summary[8], 0)) === 84,
    referenceListStoresDepthKey:
      Math.round(finiteNumberOr(summary[9], 0)) === 1,
    referenceListStoresSortKey:
      Math.round(finiteNumberOr(summary[10], 0)) === 1
  };
}

export async function buildWebGpuGpuOwnedTileListLayout({
  device,
  visibleSamples,
  canvasWidth,
  canvasHeight,
  tileSize = 16,
  maxRefsPerTile = 64,
  sourceTileAwareRenderInputContract = null,
  keepGpuResources = false
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
      consumerSummary: new Float32Array(0),
      contract: buildWebGpuGpuOwnedTileListLayoutContract({
        status: 'unavailable',
        reason: 'webgpu-gpu-owned-tile-list-layout-unavailable'
      })
    };
  }

  const recordCount = samples.length;
  const tileCols = Math.max(1, Math.ceil(finiteNumberOr(canvasWidth, 1) / tileSize));
  const tileRows = Math.max(1, Math.ceil(finiteNumberOr(canvasHeight, 1) / tileSize));
  const tileCount = tileCols * tileRows;
  const referenceCapacity = tileCount * maxRefsPerTile;
  const inputData = packTileInputSamples(samples);
  const tileCounts = new Uint32Array(tileCount);
  const tileTable = new Float32Array(tileCount * TILE_TABLE_FLOAT_STRIDE);
  const referenceList = new Float32Array(referenceCapacity * REFERENCE_FLOAT_STRIDE);
  const consumerSummaryData = new Float32Array(CONSUMER_SUMMARY_FLOAT_COUNT);
  const params = new Float32Array([
    recordCount,
    finiteNumberOr(canvasWidth, 1),
    finiteNumberOr(canvasHeight, 1),
    tileSize,
    tileCols,
    tileRows,
    tileCount,
    maxRefsPerTile
  ]);

  const inputBuffer = createBuffer(device, inputData, GPUBufferUsage.STORAGE);
  const tileCountsBuffer = createBuffer(
    device,
    tileCounts,
    GPUBufferUsage.STORAGE
  );
  const tileTableBuffer = createBuffer(
    device,
    tileTable,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const referenceListBuffer = createBuffer(
    device,
    referenceList,
    GPUBufferUsage.STORAGE
  );
  const consumerSummaryBuffer = createBuffer(
    device,
    consumerSummaryData,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const paramsBuffer = createBuffer(device, params, GPUBufferUsage.UNIFORM);
  const consumerReadbackBuffer = device.createBuffer({
    size: consumerSummaryData.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  const shader = device.createShaderModule({
    label: 'phase3-step84-webgpu-gpu-owned-tile-list-layout-wgsl',
    code: `
struct Params {
  recordCount: f32,
  canvasWidth: f32,
  canvasHeight: f32,
  tileSize: f32,
  tileCols: f32,
  tileRows: f32,
  tileCount: f32,
  maxRefsPerTile: f32,
};

@group(0) @binding(0) var<storage, read> tileInputs: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> tileCounts: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> tileTable: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> referenceList: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> consumerSummary: array<vec4f>;
@group(0) @binding(5) var<uniform> params: Params;

fn clampFloor(value: f32, lo: f32, hi: f32) -> f32 {
  return clamp(floor(value), lo, hi);
}

fn tileIndex(x: u32, y: u32) -> u32 {
  return y * u32(params.tileCols) + x;
}

@compute @workgroup_size(64)
fn scatterReferences(@builtin(global_invocation_id) id: vec3u) {
  let row = id.x;
  if (row >= u32(params.recordCount)) {
    return;
  }
  let base = row * 3u;
  let a = tileInputs[base + 0u];
  let b = tileInputs[base + 1u];
  let px = a.x;
  let py = a.y;
  let radius = max(a.z, 0.0);
  let minX = clampFloor(px - radius, 0.0, max(params.canvasWidth - 1.0, 0.0));
  let minY = clampFloor(py - radius, 0.0, max(params.canvasHeight - 1.0, 0.0));
  let maxX = clamp(ceil(px + radius), 0.0, max(params.canvasWidth - 1.0, 0.0));
  let maxY = clamp(ceil(py + radius), 0.0, max(params.canvasHeight - 1.0, 0.0));
  let minTileX = u32(clampFloor(minX / params.tileSize, 0.0, max(params.tileCols - 1.0, 0.0)));
  let minTileY = u32(clampFloor(minY / params.tileSize, 0.0, max(params.tileRows - 1.0, 0.0)));
  let maxTileX = u32(clampFloor(maxX / params.tileSize, 0.0, max(params.tileCols - 1.0, 0.0)));
  let maxTileY = u32(clampFloor(maxY / params.tileSize, 0.0, max(params.tileRows - 1.0, 0.0)));
  let maxRefs = u32(params.maxRefsPerTile);
  for (var ty = minTileY; ty <= maxTileY; ty = ty + 1u) {
    for (var tx = minTileX; tx <= maxTileX; tx = tx + 1u) {
      let tile = tileIndex(tx, ty);
      let slot = atomicAdd(&tileCounts[tile], 1u);
      if (slot < maxRefs) {
        let refOffset = tile * maxRefs + slot;
        referenceList[refOffset] = vec4f(f32(row), b.x, a.w + 0.0001 * a.w, b.z);
      }
    }
  }
}

@compute @workgroup_size(64)
fn buildOffsetCountTable(@builtin(global_invocation_id) id: vec3u) {
  let tile = id.x;
  if (tile >= u32(params.tileCount)) {
    return;
  }
  let count = atomicLoad(&tileCounts[tile]);
  let maxRefs = u32(params.maxRefsPerTile);
  let clipped = min(count, maxRefs);
  let overflow = count - clipped;
  tileTable[tile] = vec4f(f32(tile * maxRefs), f32(clipped), f32(overflow), 84.0);
}

@compute @workgroup_size(1)
fn consumeGpuOwnedTileList() {
  var nonEmpty = 0.0;
  var totalRefs = 0.0;
  var consumedRefs = 0.0;
  var maxRefs = 0.0;
  var overflow = 0.0;
  var depthKeySeen = 0.0;
  var sortKeySeen = 0.0;
  for (var tile: u32 = 0u; tile < u32(params.tileCount); tile = tile + 1u) {
    let table = tileTable[tile];
    if (table.w == 84.0 && table.y > 0.0) {
      nonEmpty = nonEmpty + 1.0;
      totalRefs = totalRefs + table.y;
      maxRefs = max(maxRefs, table.y);
      overflow = overflow + table.z;
      let offset = u32(table.x);
      let count = u32(table.y);
      for (var slot: u32 = 0u; slot < count; slot = slot + 1u) {
        let splatRef = referenceList[offset + slot];
        consumedRefs = consumedRefs + 1.0;
        if (splatRef.z != 0.0) {
          depthKeySeen = 1.0;
        }
        if (splatRef.w != 0.0) {
          sortKeySeen = 1.0;
        }
      }
    }
  }
  consumerSummary[0] = vec4f(params.tileCount, nonEmpty, totalRefs, maxRefs);
  consumerSummary[1] = vec4f(overflow, consumedRefs, params.maxRefsPerTile, 0.0);
  consumerSummary[2] = vec4f(84.0, depthKeySeen, sortKeySeen, 0.0);
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
        buffer: { type: 'storage' }
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      },
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' }
      }
    ]
  });
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout]
  });
  const scatterPipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shader, entryPoint: 'scatterReferences' }
  });
  const tablePipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shader, entryPoint: 'buildOffsetCountTable' }
  });
  const consumerPipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shader, entryPoint: 'consumeGpuOwnedTileList' }
  });
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: tileCountsBuffer } },
      { binding: 2, resource: { buffer: tileTableBuffer } },
      { binding: 3, resource: { buffer: referenceListBuffer } },
      { binding: 4, resource: { buffer: consumerSummaryBuffer } },
      { binding: 5, resource: { buffer: paramsBuffer } }
    ]
  });
  const encoder = device.createCommandEncoder({
    label: 'phase3-step84-webgpu-gpu-owned-tile-list-layout-encoder'
  });
  const pass = encoder.beginComputePass({
    label: 'phase3-step84-webgpu-gpu-owned-tile-list-layout-pass'
  });
  pass.setBindGroup(0, bindGroup);
  pass.setPipeline(scatterPipeline);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(recordCount / 64)));
  pass.setPipeline(tablePipeline);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(tileCount / 64)));
  pass.setPipeline(consumerPipeline);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyBufferToBuffer(
    consumerSummaryBuffer,
    0,
    consumerReadbackBuffer,
    0,
    consumerSummaryData.byteLength
  );
  device.queue.submit([encoder.finish()]);
  if (typeof device.queue.onSubmittedWorkDone === 'function') {
    await device.queue.onSubmittedWorkDone();
  }
  await consumerReadbackBuffer.mapAsync(GPUMapMode.READ);
  const consumerSummary = new Float32Array(consumerReadbackBuffer.getMappedRange().slice(0));
  consumerReadbackBuffer.unmap();

  const summary = readConsumerSummary(consumerSummary);
  const consumed =
    summary.consumerFollowedOffsetCountTable &&
    summary.totalTileReferenceCount > 0 &&
    summary.consumedTileReferenceCount === summary.totalTileReferenceCount;
  const result = {
    consumerSummary,
    gpuResources: keepGpuResources
      ? {
          inputBuffer,
          tileTableBuffer,
          referenceListBuffer,
          tileCount,
          tileCols,
          tileRows,
          tileSize,
          maxRefsPerTile,
          recordCount,
          referenceCapacity
        }
      : null,
    contract: buildWebGpuGpuOwnedTileListLayoutContract({
      candidateTileRecordCount: recordCount,
      tileCount,
      tileSize,
      maxRefsPerTile,
      offsetCountTableCreated: true,
      splatReferenceListCreated: true,
      referenceListStoresDepthKey: summary.referenceListStoresDepthKey,
      referenceListStoresSortKey: summary.referenceListStoresSortKey,
      gpuOwnedTileListLayoutReady: consumed,
      tileListConsumerReady: consumed,
      tileListConsumerConsumed: consumed,
      tileListConsumerReadbackCompleted: true,
      consumerFollowedOffsetCountTable: summary.consumerFollowedOffsetCountTable,
      totalTileReferenceCount: summary.totalTileReferenceCount,
      consumedTileReferenceCount: summary.consumedTileReferenceCount,
      nonEmptyTileCount: summary.nonEmptyTileCount,
      maxRefsPerTileObserved: summary.maxRefsPerTileObserved,
      overflowCount: summary.overflowCount,
      generatedLayoutFields: [
        'gpu-owned-offset-count-table',
        'gpu-owned-splat-reference-list',
        'reference-depth-key',
        'reference-sort-key',
        'fixed-capacity-tile-offset-layout'
      ],
      deferredLayoutFields: [
        'parallel-prefix-sum-offset-compaction',
        'overflow-resize-second-pass',
        'full-depth-sort-dispatch',
        'final-tile-compositor'
      ],
      tileListLayoutClassification:
        'partial-webgpu-gpu-owned-tile-list-layout',
      fullParallelPrefixSumInWgsl: false,
      fullTileListCompactionInWgsl: false,
      fullDepthSortInWgsl: false,
      finalTileCompositorImplemented: false,
      nextDepthSortInputReady: summary.referenceListStoresDepthKey,
      nextTileCompositorInputReady: consumed,
      sourceTileAwareRenderInputContractVersion:
        sourceTileAwareRenderInputContract?.contractVersion ?? null,
      reason: consumed
        ? null
        : 'webgpu-gpu-owned-tile-list-consumer-did-not-follow-offset-count-reference-list'
    })
  };
  if (!keepGpuResources) {
    for (const buffer of [
      inputBuffer,
      tileCountsBuffer,
      tileTableBuffer,
      referenceListBuffer,
      consumerSummaryBuffer,
      paramsBuffer
    ]) {
      if (typeof buffer.destroy === 'function') {
        buffer.destroy();
      }
    }
  } else {
    for (const buffer of [tileCountsBuffer, consumerSummaryBuffer, paramsBuffer]) {
      if (typeof buffer.destroy === 'function') {
        buffer.destroy();
      }
    }
  }
  if (typeof consumerReadbackBuffer.destroy === 'function') {
    consumerReadbackBuffer.destroy();
  }
  return result;
}
