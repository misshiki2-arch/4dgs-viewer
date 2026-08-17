import {
  buildProductionTileExecutionPlanContract,
  buildProductionTileExecutionPlanBindGroupEntries,
  buildProductionTileExecutionPlanBindGroupLayoutEntries,
  buildProductionTileExecutionPlanWgslBindings,
  PRODUCTION_TILE_EXECUTION_PLAN_CHUNK_SIZE,
  PRODUCTION_TILE_EXECUTION_PLAN_MAGIC,
  PRODUCTION_TILE_EXECUTION_PLAN_STATUS,
  PRODUCTION_TILE_EXECUTION_PLAN_WORD,
  PRODUCTION_TILE_EXECUTION_PLAN_WORD_COUNT,
  resolveProductionTileExecutionPlanTopology
} from './common_4dgs_production_tile_execution_plan_contracts.js';

const SCAN_WORD_STRIDE = 8;

function createZeroBuffer(device, wordCount, usage, label) {
  return device.createBuffer({
    label,
    size: Math.max(4, wordCount * Uint32Array.BYTES_PER_ELEMENT),
    usage
  });
}

function createUniformBuffer(device, values, label) {
  const data = new Uint32Array(values);
  const buffer = device.createBuffer({
    label,
    size: Math.max(16, data.byteLength),
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true
  });
  new Uint32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

export function createProductionTileExecutionPlanResources({
  device,
  tileCountsBuffer,
  tileCount,
  recordCount,
  referenceCapacity,
  sourceTileInputResourceIdentity,
  resourceIdentity,
  planIdentity,
  compositorDispatchX = 1,
  compositorDispatchY = 1
} = {}) {
  const topology = resolveProductionTileExecutionPlanTopology({
    tileCount,
    referenceCapacity
  });
  const scanUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const scanWordCount = topology.tileCount * SCAN_WORD_STRIDE;
  const scanBuffers = [
    createZeroBuffer(
      device, scanWordCount, scanUsage,
      'phase3-production-tile-plan-scan-a-buffer'
    ),
    createZeroBuffer(
      device, scanWordCount, scanUsage,
      'phase3-production-tile-plan-scan-b-buffer'
    )
  ];
  const tileTableBuffer = createZeroBuffer(
    device,
    topology.tileCount * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    'phase3-production-gpu-plan-tile-table-buffer'
  );
  const tileChunkTableBuffer = createZeroBuffer(
    device,
    topology.tileCount * 2,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    'phase3-production-gpu-plan-tile-chunk-table-buffer'
  );
  const planBuffer = createZeroBuffer(
    device,
    PRODUCTION_TILE_EXECUTION_PLAN_WORD_COUNT,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    'phase3-production-gpu-execution-plan-buffer'
  );
  const compositorIndirectBuffer = createZeroBuffer(
    device,
    topology.maximumReferenceChunkCount * 3,
    GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
    'phase3-production-gpu-plan-compositor-indirect-buffer'
  );

  const shader = device.createShaderModule({
    label: 'phase3-production-gpu-execution-plan-wgsl',
    code: `
struct Params {
  tileCount: u32,
  recordCount: u32,
  referenceCapacity: u32,
  scanOffset: u32,
  planIdentity: u32,
  chunkSize: u32,
  reserved0: u32,
  reserved1: u32,
};
struct ScanValue {
  sums: vec4u,
  overflow: vec4u,
};

${buildProductionTileExecutionPlanWgslBindings()}

fn nextPowerOfTwo(value: u32) -> u32 {
  if (value <= 1u) { return value; }
  return 1u << (32u - countLeadingZeros(value - 1u));
}

@compute @workgroup_size(64)
fn initializeScan(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.tileCount) { return; }
  let count = atomicLoad(&tileCounts[id.x]);
  let padded = nextPowerOfTwo(count);
  let chunks = (count + params.chunkSize - 1u) / params.chunkSize;
  scanDestination[id.x] = ScanValue(
    vec4u(count, padded, chunks, count),
    vec4u(0u)
  );
}

@compute @workgroup_size(64)
fn scanStep(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.tileCount) { return; }
  let own = scanSource[id.x];
  if (id.x < params.scanOffset) {
    scanDestination[id.x] = own;
    return;
  }
  let previous = scanSource[id.x - params.scanOffset];
  let actual = own.sums.x + previous.sums.x;
  let padded = own.sums.y + previous.sums.y;
  let chunks = own.sums.z + previous.sums.z;
  scanDestination[id.x] = ScanValue(
    vec4u(actual, padded, chunks, max(own.sums.w, previous.sums.w)),
    vec4u(
      own.overflow.x | previous.overflow.x |
        select(0u, 1u, actual < own.sums.x),
      own.overflow.y | previous.overflow.y |
        select(0u, 1u, padded < own.sums.y),
      own.overflow.z | previous.overflow.z |
        select(0u, 1u, chunks < own.sums.z),
      0u
    )
  );
}

@compute @workgroup_size(64)
fn finalizePlan(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.tileCount) { return; }
  let inclusive = scanSource[id.x];
  var previous = ScanValue(vec4u(0u), vec4u(0u));
  if (id.x > 0u) { previous = scanSource[id.x - 1u]; }
  let count = inclusive.sums.x - previous.sums.x;
  let padded = inclusive.sums.y - previous.sums.y;
  let chunks = inclusive.sums.z - previous.sums.z;
  tileTable[id.x] = vec4f(
    f32(previous.sums.y), f32(count), f32(padded), 84.0
  );
  tileChunkTable[id.x] = vec2u(previous.sums.z, chunks);
  if (id.x + 1u != params.tileCount) { return; }
  let arithmeticOverflow = any(inclusive.overflow.xyz != vec3u(0u));
  let overflow = arithmeticOverflow ||
    inclusive.sums.y > params.referenceCapacity;
  atomicStore(&plan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.magic}u], ${PRODUCTION_TILE_EXECUTION_PLAN_MAGIC}u);
  atomicStore(&plan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.status}u], select(
    ${PRODUCTION_TILE_EXECUTION_PLAN_STATUS.ready}u,
    ${PRODUCTION_TILE_EXECUTION_PLAN_STATUS.capacityOverflow}u,
    overflow
  ));
  atomicStore(&plan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.recordCount}u], params.recordCount);
  atomicStore(&plan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.tileCount}u], params.tileCount);
  atomicStore(&plan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.referenceCapacity}u], params.referenceCapacity);
  atomicStore(&plan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.requiredReferenceCount}u], inclusive.sums.x);
  atomicStore(&plan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.requiredPaddedReferenceCapacity}u], inclusive.sums.y);
  atomicStore(&plan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.maxReferencesPerTile}u], inclusive.sums.w);
  atomicStore(&plan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.totalReferenceChunkCount}u], inclusive.sums.z);
  atomicStore(&plan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.maxReferenceChunksPerTile}u],
    (inclusive.sums.w + params.chunkSize - 1u) / params.chunkSize);
  atomicStore(&plan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.overflowReferenceCount}u],
    select(0u, max(inclusive.sums.y, params.referenceCapacity) -
      params.referenceCapacity, overflow));
  atomicStore(&plan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.compactOffsetTableReady}u], 1u);
  atomicStore(&plan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.planIdentity}u], params.planIdentity);
}

@compute @workgroup_size(64)
fn buildCompositorDispatches(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${topology.maximumReferenceChunkCount}u) { return; }
  let status = atomicLoad(&plan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.status}u]);
  let maxChunks = atomicLoad(&plan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.maxReferenceChunksPerTile}u]);
  let dispatchEnabled = status == ${PRODUCTION_TILE_EXECUTION_PLAN_STATUS.ready}u && id.x < maxChunks;
  let base = id.x * 3u;
  compositorDispatches[base + 0u] = select(0u, params.reserved0, dispatchEnabled);
  compositorDispatches[base + 1u] = select(0u, params.reserved1, dispatchEnabled);
  compositorDispatches[base + 2u] = select(0u, 1u, dispatchEnabled);
}
`
  });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: buildProductionTileExecutionPlanBindGroupLayoutEntries(
      GPUShaderStage.COMPUTE
    )
  });
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout]
  });
  const pipelines = new Map();
  const pipeline = (entryPoint) => {
    if (!pipelines.has(entryPoint)) {
      pipelines.set(entryPoint, device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module: shader, entryPoint }
      }));
    }
    return pipelines.get(entryPoint);
  };
  const paramsBuffers = [];
  const submitStage = ({
    entryPoint,
    source,
    destination,
    scanOffset = 0,
    workItemCount = topology.tileCount
  }) => {
    const paramsBuffer = createUniformBuffer(device, [
      topology.tileCount,
      recordCount,
      topology.referenceCapacity,
      scanOffset,
      planIdentity,
      PRODUCTION_TILE_EXECUTION_PLAN_CHUNK_SIZE,
      compositorDispatchX,
      compositorDispatchY
    ], `phase3-production-tile-plan-${entryPoint}-params`);
    paramsBuffers.push(paramsBuffer);
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: buildProductionTileExecutionPlanBindGroupEntries({
        tileCountsBuffer,
        scanSourceBuffer: source,
        scanDestinationBuffer: destination,
        tileTableBuffer,
        tileChunkTableBuffer,
        planBuffer,
        paramsBuffer,
        compositorIndirectBuffer
      })
    });
    const encoder = device.createCommandEncoder({
      label: `phase3-production-tile-plan-${entryPoint}-encoder`
    });
    const pass = encoder.beginComputePass({
      label: `phase3-production-tile-plan-${entryPoint}-pass`
    });
    pass.setPipeline(pipeline(entryPoint));
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(workItemCount / 64));
    pass.end();
    device.queue.submit([encoder.finish()]);
  };

  submitStage({
    entryPoint: 'initializeScan',
    source: scanBuffers[1],
    destination: scanBuffers[0]
  });
  let current = 0;
  for (const scanOffset of topology.scanStageOffsets) {
    const next = current === 0 ? 1 : 0;
    submitStage({
      entryPoint: 'scanStep',
      source: scanBuffers[current],
      destination: scanBuffers[next],
      scanOffset
    });
    current = next;
  }
  submitStage({
    entryPoint: 'finalizePlan',
    source: scanBuffers[current],
    destination: scanBuffers[current === 0 ? 1 : 0]
  });
  submitStage({
    entryPoint: 'buildCompositorDispatches',
    source: scanBuffers[current],
    destination: scanBuffers[current === 0 ? 1 : 0],
    workItemCount: topology.maximumReferenceChunkCount
  });

  const contract = buildProductionTileExecutionPlanContract({
    resourceIdentity,
    sourceTileInputResourceIdentity,
    planIdentity,
    tileCount: topology.tileCount,
    recordCount,
    referenceCapacity: topology.referenceCapacity,
    scanStageCount: topology.scanStageCount,
    maximumReferenceChunkCount: topology.maximumReferenceChunkCount,
    gpuPlanBufferCreated: true,
    gpuTileTableCreated: true,
    gpuChunkTableCreated: true,
    gpuPlanGenerated: true,
    scatterConsumesPlan: true,
    sortConsumesPlan: true,
    compositorConsumesPlan: true,
    productionCriticalReadbackUsed: false,
    intermediateCpuControlRoundTripUsed: false,
    sceneDependentCpuPlanMaterialized: false,
    schedulerContinuationUsed: false,
    capacityOverflowFailClosed: true,
    silentDropAllowed: false
  });
  return {
    topology,
    contract,
    planBuffer,
    compositorIndirectBuffer,
    tileTableBuffer,
    tileChunkTableBuffer,
    transientBuffers: [...scanBuffers, ...paramsBuffers]
  };
}
