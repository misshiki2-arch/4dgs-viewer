import {
  buildWebGpuGpuOwnedTileListLayoutContract
} from './common_4dgs_record_contracts.js';
import {
  buildProductionGpuExecutionContract,
  resolveProductionGpuExecutionLimits,
  splitProductionGpuWork
} from './common_4dgs_production_gpu_execution_contracts.js';
import {
  resolveProductionTileReferenceAllocation
} from './common_4dgs_production_tile_reference_contracts.js';
import {
  PRODUCTION_TILE_EXECUTION_PLAN_STATUS,
  PRODUCTION_TILE_EXECUTION_PLAN_WORD
} from './common_4dgs_production_tile_execution_plan_contracts.js';
import {
  createProductionTileExecutionPlanResources
} from './webgpu_production_tile_execution_plan.js';

const TILE_INPUT_FLOAT_STRIDE = 12;
const REFERENCE_FLOAT_STRIDE = 4;
let nextGpuOwnedTileListResourceIdentity = 1;

function finiteNumberOr(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function createBuffer(device, data, usage, label = undefined) {
  const buffer = device.createBuffer({
    label,
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

function createEmptyBuffer(device, size, usage, label = undefined) {
  return device.createBuffer({
    label,
    size: Math.max(4, Math.ceil(size / 4) * 4),
    usage
  });
}

function packTileInputSamples(samples) {
  const data = new Float32Array(samples.length * TILE_INPUT_FLOAT_STRIDE);
  samples.forEach((sample, index) => {
    const offset = index * TILE_INPUT_FLOAT_STRIDE;
    const footprint = sample?.footprintPayload ?? {};
    const conic = Array.isArray(sample?.conic)
      ? sample.conic
      : (Array.isArray(footprint?.conic) ? footprint.conic : []);
    const colorAlpha = Array.isArray(sample?.colorAlpha)
      ? sample.colorAlpha
      : [sample?.colorAlpha?.r, sample?.colorAlpha?.g,
          sample?.colorAlpha?.b, sample?.colorAlpha?.a];
    data[offset + 0] = finiteNumberOr(sample?.samplePx?.x, 0);
    data[offset + 1] = finiteNumberOr(sample?.samplePx?.y, 0);
    data[offset + 2] = finiteNumberOr(
      footprint.radiusPx ?? sample?.renderAttribute?.radiusPx, 0
    );
    data[offset + 3] = finiteNumberOr(sample?.depth, 0);
    data[offset + 4] = finiteNumberOr(conic[0], 0);
    data[offset + 5] = finiteNumberOr(conic[1], 0);
    data[offset + 6] = finiteNumberOr(sample?.sortKey, sample?.depth ?? 0);
    data[offset + 7] = finiteNumberOr(conic[2], 0);
    data[offset + 8] = finiteNumberOr(colorAlpha[0], 1);
    data[offset + 9] = finiteNumberOr(colorAlpha[1], 0);
    data[offset + 10] = finiteNumberOr(colorAlpha[2], 0);
    data[offset + 11] = finiteNumberOr(colorAlpha[3], 1);
  });
  return data;
}

export async function buildWebGpuGpuOwnedTileListLayout({
  device,
  visibleSamples,
  tileInputResource = null,
  canvasWidth,
  canvasHeight,
  tileSize = 16,
  sourceTileAwareRenderInputContract = null,
  keepGpuResources = false
} = {}) {
  const samples = Array.isArray(visibleSamples) ? visibleSamples : [];
  const externalInputBuffer = tileInputResource?.buffer ?? null;
  const externalRecordCount = Math.max(
    0, Math.floor(Number(tileInputResource?.recordCount) || 0)
  );
  if (
    !device ||
    (samples.length <= 0 && (!externalInputBuffer || externalRecordCount <= 0)) ||
    typeof GPUBufferUsage === 'undefined' ||
    typeof GPUShaderStage === 'undefined'
  ) {
    return {
      consumerSummary: new Uint32Array(0),
      contract: buildWebGpuGpuOwnedTileListLayoutContract({
        status: 'unavailable',
        reason: 'webgpu-gpu-owned-tile-list-layout-unavailable'
      })
    };
  }

  const recordCount = externalInputBuffer ? externalRecordCount : samples.length;
  const layoutResourceIdentity =
    `gpu-owned-tile-list-resource-${nextGpuOwnedTileListResourceIdentity++}`;
  const tileCols = Math.max(1, Math.ceil(finiteNumberOr(canvasWidth, 1) / tileSize));
  const tileRows = Math.max(1, Math.ceil(finiteNumberOr(canvasHeight, 1) / tileSize));
  const tileCount = tileCols * tileRows;
  const allocation = resolveProductionTileReferenceAllocation({
    device, recordCount, tileCount
  });
  const executionLimits = resolveProductionGpuExecutionLimits({ device, tileCount });
  const recordBatches = splitProductionGpuWork(
    recordCount, executionLimits.recordBatchSize
  );
  const referenceCapacity = allocation.allocatedReferenceCapacity;
  const inputData = externalInputBuffer ? null : packTileInputSamples(samples);
  const tileCounts = new Uint32Array(tileCount);

  const inputBuffer = externalInputBuffer ??
    createBuffer(device, inputData, GPUBufferUsage.STORAGE);
  const tileCountsBuffer = createBuffer(
    device, tileCounts,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  );
  const referenceListBuffer = createEmptyBuffer(
    device,
    referenceCapacity * REFERENCE_FLOAT_STRIDE * Float32Array.BYTES_PER_ELEMENT,
    GPUBufferUsage.STORAGE,
    'phase3-production-compact-splat-reference-list-buffer'
  );

  const shader = device.createShaderModule({
    label: 'phase3-production-bounded-compact-gpu-owned-tile-list-wgsl',
    code: `
struct Params {
  recordCount: u32,
  canvasWidth: u32,
  canvasHeight: u32,
  tileSize: u32,
  tileCols: u32,
  tileRows: u32,
  tileCount: u32,
  referenceCapacity: u32,
  batchStart: u32,
  batchEnd: u32,
};

@group(0) @binding(0) var<storage, read> tileInputs: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> tileCounts: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> tileTable: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> referenceList: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> executionPlan: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: Params;

fn tileIndex(x: u32, y: u32) -> u32 {
  return y * params.tileCols + x;
}

fn tileBounds(row: u32) -> vec4u {
  let base = row * 3u;
  let a = tileInputs[base + 0u];
  let canvasMax = vec2f(f32(params.canvasWidth - 1u), f32(params.canvasHeight - 1u));
  let minimum = clamp(floor(a.xy - vec2f(a.z)), vec2f(0.0), canvasMax);
  let maximum = clamp(ceil(a.xy + vec2f(a.z)), vec2f(0.0), canvasMax);
  let tileMax = vec2f(f32(params.tileCols - 1u), f32(params.tileRows - 1u));
  let minTile = vec2u(clamp(floor(minimum / f32(params.tileSize)), vec2f(0.0), tileMax));
  let maxTile = vec2u(clamp(floor(maximum / f32(params.tileSize)), vec2f(0.0), tileMax));
  return vec4u(minTile, maxTile);
}

@compute @workgroup_size(64)
fn countReferences(@builtin(global_invocation_id) id: vec3u) {
  let row = params.batchStart + id.x;
  if (row >= params.batchEnd || row >= params.recordCount) { return; }
  let a = tileInputs[row * 3u + 0u];
  let colorAlpha = tileInputs[row * 3u + 2u];
  if (a.z <= 0.0 || colorAlpha.w <= 0.0) { return; }
  let bounds = tileBounds(row);
  for (var ty = bounds.y; ty <= bounds.w; ty = ty + 1u) {
    for (var tx = bounds.x; tx <= bounds.z; tx = tx + 1u) {
      atomicAdd(&tileCounts[tileIndex(tx, ty)], 1u);
    }
  }
}

@compute @workgroup_size(64)
fn scatterReferences(@builtin(global_invocation_id) id: vec3u) {
  if (atomicLoad(&executionPlan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.status}u]) !=
      ${PRODUCTION_TILE_EXECUTION_PLAN_STATUS.ready}u) { return; }
  let row = params.batchStart + id.x;
  if (row >= params.batchEnd || row >= params.recordCount) { return; }
  let base = row * 3u;
  let a = tileInputs[base + 0u];
  let b = tileInputs[base + 1u];
  let colorAlpha = tileInputs[base + 2u];
  if (a.z <= 0.0 || colorAlpha.w <= 0.0) { return; }
  let bounds = tileBounds(row);
  for (var ty = bounds.y; ty <= bounds.w; ty = ty + 1u) {
    for (var tx = bounds.x; tx <= bounds.z; tx = tx + 1u) {
      let tile = tileIndex(tx, ty);
      let slot = atomicAdd(&tileCounts[tile], 1u);
      let referenceIndex = u32(tileTable[tile].x) + slot;
      if (referenceIndex < params.referenceCapacity) {
        let splatRef = vec4f(f32(row), b.x, a.w + 0.0001 * a.w, b.z);
        referenceList[referenceIndex] = splatRef;
        atomicAdd(&executionPlan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.scatteredReferenceCount}u], 1u);
      } else {
        atomicStore(&executionPlan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.status}u],
          ${PRODUCTION_TILE_EXECUTION_PLAN_STATUS.executionFailure}u);
      }
    }
  }
}
`
  });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
    ]
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  const pipelines = new Map();
  const getPipeline = (entryPoint) => {
    if (!pipelines.has(entryPoint)) {
      pipelines.set(entryPoint, device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module: shader, entryPoint }
      }));
    }
    return pipelines.get(entryPoint);
  };
  const temporaryParamsBuffers = [];
  let executionPlanResources = null;
  function submitRecordBatch(entryPoint, batch) {
    const params = new Uint32Array([
      recordCount,
      Math.max(1, Math.round(finiteNumberOr(canvasWidth, 1))),
      Math.max(1, Math.round(finiteNumberOr(canvasHeight, 1))),
      tileSize, tileCols, tileRows, tileCount, referenceCapacity,
      batch.start, batch.end
    ]);
    const paramsBuffer = createBuffer(device, params, GPUBufferUsage.UNIFORM);
    temporaryParamsBuffers.push(paramsBuffer);
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: tileCountsBuffer } },
        { binding: 2, resource: { buffer: tileTableBuffer } },
        { binding: 3, resource: { buffer: referenceListBuffer } },
        { binding: 4, resource: { buffer: executionPlanResources.planBuffer } },
        { binding: 5, resource: { buffer: paramsBuffer } }
      ]
    });
    const encoder = device.createCommandEncoder({
      label: `phase3-production-bounded-tile-list-${entryPoint}-encoder`
    });
    const pass = encoder.beginComputePass({
      label: `phase3-production-bounded-tile-list-${entryPoint}-pass`
    });
    pass.setPipeline(getPipeline(entryPoint));
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil((batch.end - batch.start) / 64)));
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  // Count submissions precede the GPU scan in queue order.  No scene-dependent
  // value is mapped to JavaScript between these stages.
  const pendingCountParams = [];
  for (const batch of recordBatches) {
    const params = new Uint32Array([
      recordCount,
      Math.max(1, Math.round(finiteNumberOr(canvasWidth, 1))),
      Math.max(1, Math.round(finiteNumberOr(canvasHeight, 1))),
      tileSize, tileCols, tileRows, tileCount, referenceCapacity,
      batch.start, batch.end
    ]);
    const paramsBuffer = createBuffer(device, params, GPUBufferUsage.UNIFORM);
    pendingCountParams.push(paramsBuffer);
    const countPlanPlaceholder = createEmptyBuffer(
      device, 16 * Uint32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.STORAGE,
      'phase3-production-count-plan-placeholder-buffer'
    );
    const countTileTablePlaceholder = createEmptyBuffer(
      device, tileCount * 4 * Float32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.STORAGE,
      'phase3-production-count-tile-table-placeholder-buffer'
    );
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: tileCountsBuffer } },
        { binding: 2, resource: { buffer: countTileTablePlaceholder } },
        { binding: 3, resource: { buffer: referenceListBuffer } },
        { binding: 4, resource: { buffer: countPlanPlaceholder } },
        { binding: 5, resource: { buffer: paramsBuffer } }
      ]
    });
    const encoder = device.createCommandEncoder({
      label: 'phase3-production-bounded-tile-list-countReferences-encoder'
    });
    const pass = encoder.beginComputePass({
      label: 'phase3-production-bounded-tile-list-countReferences-pass'
    });
    pass.setPipeline(getPipeline('countReferences'));
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil((batch.end - batch.start) / 64)));
    pass.end();
    device.queue.submit([encoder.finish()]);
    temporaryParamsBuffers.push(
      countPlanPlaceholder,
      countTileTablePlaceholder
    );
  }
  temporaryParamsBuffers.push(...pendingCountParams);

  executionPlanResources = createProductionTileExecutionPlanResources({
    device,
    tileCountsBuffer,
    tileCount,
    recordCount,
    referenceCapacity,
    sourceTileInputResourceIdentity:
      tileInputResource?.resourceIdentity ?? layoutResourceIdentity,
    resourceIdentity: `${layoutResourceIdentity}:execution-plan`,
    planIdentity: nextGpuOwnedTileListResourceIdentity,
    compositorDispatchX: tileCols * Math.max(1, Math.ceil(tileSize / 8)),
    compositorDispatchY: tileRows * Math.max(1, Math.ceil(tileSize / 8))
  });
  const { tileTableBuffer } = executionPlanResources;
  {
    const encoder = device.createCommandEncoder({
      label: 'phase3-production-bounded-tile-scatter-reset-encoder'
    });
    encoder.clearBuffer(tileCountsBuffer);
    device.queue.submit([encoder.finish()]);
  }

  for (const batch of recordBatches) submitRecordBatch('scatterReferences', batch);

  const consumed = executionPlanResources.contract.gpuExecutionPlanReady === true;
  const tileReferenceExecutionContract = buildProductionGpuExecutionContract({
    stage: 'production-tile-reference-count-and-scatter',
    limits: executionLimits,
    inputRecordCount: recordCount,
    inputReferenceCount: 0,
    completedRecordCount: recordCount,
    completedReferenceCount: 0,
    countSubmissionCount: recordBatches.length,
    scatterSubmissionCount: recordBatches.length,
    maximumRecordsInSubmission: Math.min(recordCount, executionLimits.recordBatchSize),
    maximumReferencesInSubmission: 0,
    maximumReferencesPerPixelInvocation: 0,
    gpuResourceLineageMaintained: consumed,
    recordReferenceCapacitySeparated: true,
    silentDropAllowed: false,
    schedulerContinuationUsed: false,
    allStagesCompleted: consumed,
    reason: consumed ? null : 'production-tile-reference-gpu-plan-not-ready'
  });
  const result = {
    consumerSummary: new Uint32Array(0),
    gpuResources: keepGpuResources
      ? {
          inputBuffer, tileTableBuffer, referenceListBuffer,
          tileChunkTableBuffer: executionPlanResources.tileChunkTableBuffer,
          executionPlanBuffer: executionPlanResources.planBuffer,
          compositorIndirectBuffer:
            executionPlanResources.compositorIndirectBuffer,
          executionPlanContract: executionPlanResources.contract,
          executionPlanTopology: executionPlanResources.topology,
          tileCount, tileCols, tileRows, tileSize, recordCount,
          referenceCapacity,
          inputResourceIdentity: tileInputResource?.resourceIdentity ?? null,
          sourceWorksetResourceIdentity:
            tileInputResource?.sourceWorksetResourceIdentity ?? null,
          layoutResourceIdentity,
          boundedExecutionContract: tileReferenceExecutionContract,
          transientBuffers: [
            tileCountsBuffer,
            ...executionPlanResources.transientBuffers,
            ...temporaryParamsBuffers
          ]
        }
      : null,
    contract: buildWebGpuGpuOwnedTileListLayoutContract({
      candidateTileRecordCount: recordCount,
      tileCount,
      tileSize,
      maxRefsPerTile: 0,
      allocatedReferenceCapacity: referenceCapacity,
      requiredPaddedReferenceCapacity: 0,
      recordAndReferenceCapacitySeparated: true,
      compactOffsetsGenerated: true,
      capacityOverflowFailClosed: true,
      silentDropAllowed: false,
      tileReferenceCapacityContract: null,
      boundedExecutionContract: tileReferenceExecutionContract,
      gpuExecutionPlanContract: executionPlanResources.contract,
      offsetCountTableCreated: true,
      splatReferenceListCreated: true,
      referenceListStoresDepthKey: true,
      referenceListStoresSortKey: true,
      gpuOwnedTileListLayoutReady: consumed,
      tileListConsumerReady: consumed,
      tileListConsumerConsumed: consumed,
      tileListConsumerReadbackCompleted: false,
      consumerFollowedOffsetCountTable: true,
      totalTileReferenceCount: 0,
      consumedTileReferenceCount: 0,
      nonEmptyTileCount: 0,
      maxRefsPerTileObserved: 0,
      overflowCount: 0,
      generatedLayoutFields: [
        'gpu-owned-compact-offset-count-padded-count-table',
        'gpu-owned-execution-plan-header',
        'gpu-hillis-steele-offset-span-scan',
        'device-limit-bounded-splat-reference-list',
        'bounded-record-batch-reference-count-and-scatter',
        'reference-depth-key',
        'reference-sort-key'
      ],
      deferredLayoutFields: [
        'chunk-lod-streaming',
        'final-cuda-sort-compositor-parity'
      ],
      tileListLayoutClassification:
        'production-webgpu-compact-gpu-owned-tile-list-layout',
      fullParallelPrefixSumInWgsl: true,
      fullTileListCompactionInWgsl: true,
      fullDepthSortInWgsl: false,
      finalTileCompositorImplemented: false,
      nextDepthSortInputReady: consumed,
      nextTileCompositorInputReady: consumed,
      sourceTileAwareRenderInputContractVersion:
        sourceTileAwareRenderInputContract?.contractVersion ??
        tileInputResource?.contractVersion ?? null,
      reason: consumed ? null : 'webgpu-production-gpu-execution-plan-not-ready'
    })
  };
  const allOwnedBuffers = [
    externalInputBuffer ? null : inputBuffer,
    tileCountsBuffer, tileTableBuffer, referenceListBuffer,
    executionPlanResources.tileChunkTableBuffer,
    executionPlanResources.planBuffer,
    executionPlanResources.compositorIndirectBuffer,
    ...executionPlanResources.transientBuffers,
    ...temporaryParamsBuffers
  ];
  if (!keepGpuResources && typeof device.queue.onSubmittedWorkDone === 'function') {
    device.queue.onSubmittedWorkDone().then(() => {
      for (const buffer of allOwnedBuffers) {
        if (typeof buffer?.destroy === 'function') buffer.destroy();
      }
    });
  }
  return result;
}
