import {
  buildProductionGpuExecutionContract,
  countBitonicCompareStages,
  resolveProductionGpuExecutionLimits,
  splitProductionGpuWork
} from './common_4dgs_production_gpu_execution_contracts.js';
import {
  PRODUCTION_TILE_EXECUTION_PLAN_STATUS,
  PRODUCTION_TILE_EXECUTION_PLAN_WORD,
  readProductionTileExecutionPlanObserver
} from './common_4dgs_production_tile_execution_plan_contracts.js';

const ORDERING_SUMMARY_UINT_COUNT = 28;
const COMPOSITOR_SUMMARY_FLOAT_COUNT = 40;

function alignTo(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
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

function submitProductionStage(device, encoder) {
  device.queue.submit([encoder.finish()]);
}

export async function executeBoundedProductionTileSortAndCompositor({
  device,
  resources,
  outputWidth,
  outputHeight,
  outputTextureOverride = null,
  onProductionSubmitted = null
} = {}) {
  const referenceCapacity = Math.max(1, resources.referenceCapacity);
  const requiredPaddedReferenceCapacity = referenceCapacity;
  const requiredReferenceCount = referenceCapacity;
  const sortCapacityLimit = Math.max(
    1,
    2 ** Math.ceil(Math.log2(referenceCapacity))
  );
  const executionLimits = resolveProductionGpuExecutionLimits({
    device,
    tileCount: resources.tileCount
  });
  const referenceBatches = splitProductionGpuWork(
    requiredPaddedReferenceCapacity,
    executionLimits.referenceBatchSize
  );

  const orderedReferenceBufferBytes = Math.max(16, referenceCapacity * 16);
  const orderedReferenceBuffer = device.createBuffer({
    label: 'phase3-bounded-production-depth-sorted-reference-buffer',
    size: orderedReferenceBufferBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  });
  const orderingSummaryData = new Uint32Array(ORDERING_SUMMARY_UINT_COUNT);
  const orderingSummaryBuffer = createBuffer(
    device,
    orderingSummaryData,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    'phase3-bounded-production-ordering-summary-buffer'
  );
  const orderingParamsData = new Uint32Array(8);
  const orderingParamsBuffer = createBuffer(
    device,
    orderingParamsData,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    'phase3-bounded-production-ordering-params-buffer'
  );
  const orderingSummaryReadbackBuffer = device.createBuffer({
    size: orderingSummaryData.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const orderingShader = device.createShaderModule({
    label: 'phase3-bounded-production-global-bitonic-stage-wgsl',
    code: `
struct OrderingParams {
  tileCount: u32,
  referenceCapacity: u32,
  requiredPaddedReferenceCapacity: u32,
  statusCode: u32,
  batchStart: u32,
  batchEnd: u32,
  bitonicK: u32,
  bitonicJ: u32,
};

@group(0) @binding(0) var<storage, read> tileTable: array<vec4f>;
@group(0) @binding(1) var<storage, read> sourceReferences: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> orderedReferences: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> orderingSummary: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: OrderingParams;
@group(0) @binding(5) var<storage, read_write> executionPlan: array<atomic<u32>>;

fn planReady() -> bool {
  return atomicLoad(&executionPlan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.status}u]) ==
    ${PRODUCTION_TILE_EXECUTION_PLAN_STATUS.ready}u;
}
fn requiredPaddedCount() -> u32 {
  return atomicLoad(&executionPlan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.requiredPaddedReferenceCapacity}u]);
}

fn findTile(referenceIndex: u32) -> u32 {
  var low = 0u;
  var high = params.tileCount;
  loop {
    if (low + 1u >= high) { break; }
    let middle = low + (high - low) / 2u;
    if (u32(tileTable[middle].x) <= referenceIndex) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return low;
}

fn currentReferenceIndex(id: vec3u) -> u32 {
  return params.batchStart + id.x;
}

@compute @workgroup_size(64)
fn seedOrderedReferences(@builtin(global_invocation_id) id: vec3u) {
  if (!planReady()) { return; }
  let referenceIndex = currentReferenceIndex(id);
  if (referenceIndex >= params.batchEnd ||
      referenceIndex >= requiredPaddedCount()) { return; }
  let tile = findTile(referenceIndex);
  let table = tileTable[tile];
  let offset = u32(table.x);
  let count = u32(table.y);
  let paddedCount = u32(table.z);
  let localIndex = referenceIndex - offset;
  if (table.w != 84.0 || localIndex >= paddedCount) { return; }
  orderedReferences[referenceIndex] = select(
    vec4f(0.0, 0.0, 0.0, -3.402823e38),
    sourceReferences[referenceIndex],
    localIndex < count
  );
}

@compute @workgroup_size(64)
fn compareSwapBitonicStage(@builtin(global_invocation_id) id: vec3u) {
  if (!planReady()) { return; }
  let referenceIndex = currentReferenceIndex(id);
  if (referenceIndex >= params.batchEnd ||
      referenceIndex >= requiredPaddedCount()) { return; }
  let tile = findTile(referenceIndex);
  let table = tileTable[tile];
  let offset = u32(table.x);
  let paddedCount = u32(table.z);
  let localIndex = referenceIndex - offset;
  if (table.w != 84.0 || localIndex >= paddedCount ||
      params.bitonicK > paddedCount) { return; }
  let partnerLocalIndex = localIndex ^ params.bitonicJ;
  if (partnerLocalIndex <= localIndex || partnerLocalIndex >= paddedCount) { return; }
  let partnerIndex = offset + partnerLocalIndex;
  let selfReference = orderedReferences[referenceIndex];
  let partnerReference = orderedReferences[partnerIndex];
  let descending = (localIndex & params.bitonicK) == 0u;
  let shouldSwap = select(
    selfReference.w > partnerReference.w,
    selfReference.w < partnerReference.w,
    descending
  );
  if (shouldSwap) {
    orderedReferences[referenceIndex] = partnerReference;
    orderedReferences[partnerIndex] = selfReference;
  }
}

@compute @workgroup_size(64)
fn validateOrderedReferences(@builtin(global_invocation_id) id: vec3u) {
  if (!planReady()) { return; }
  let referenceIndex = currentReferenceIndex(id);
  if (referenceIndex >= params.batchEnd ||
      referenceIndex >= requiredPaddedCount()) { return; }
  let tile = findTile(referenceIndex);
  let table = tileTable[tile];
  let offset = u32(table.x);
  let count = u32(table.y);
  let paddedCount = u32(table.z);
  let localIndex = referenceIndex - offset;
  if (table.w != 84.0 || localIndex >= count) { return; }
  let splatRef = orderedReferences[referenceIndex];
  atomicAdd(&orderingSummary[0], 1u);
  atomicAdd(&orderingSummary[4], 1u);
  atomicAdd(&orderingSummary[9], 1u);
  atomicAdd(&executionPlan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.sortedReferenceCount}u], 1u);
  if (splatRef.z != 0.0) { atomicStore(&orderingSummary[2], 1u); }
  if (splatRef.w != 0.0) { atomicStore(&orderingSummary[3], 1u); }
  atomicStore(&orderingSummary[7], 1u);
  if (localIndex + 1u < count &&
      splatRef.w < orderedReferences[referenceIndex + 1u].w) {
    atomicAdd(&orderingSummary[20], 1u);
  }
  if (localIndex == 0u) {
    atomicAdd(&orderingSummary[1], 1u);
    atomicAdd(&orderingSummary[8], 1u);
    atomicAdd(&orderingSummary[12], count);
    atomicMax(&orderingSummary[11], count);
    atomicMax(&orderingSummary[16], paddedCount);
    atomicAdd(&orderingSummary[22], 1u);
    var levels = 0u;
    var levelCapacity = paddedCount;
    loop {
      if (levelCapacity <= 1u) { break; }
      levels = levels + 1u;
      levelCapacity = levelCapacity / 2u;
    }
    let stages = (levels * (levels + 1u)) / 2u;
    atomicMax(&orderingSummary[21], stages);
    atomicMax(&orderingSummary[23], stages);
    let utilization = (count * 100000u) / max(paddedCount, 1u);
    atomicMax(&orderingSummary[17], utilization);
    atomicAdd(&orderingSummary[18], utilization);
    atomicAdd(&orderingSummary[19], 1u);
    atomicStore(&orderingSummary[5], params.referenceCapacity);
    atomicStore(&orderingSummary[6], params.statusCode);
  }
}
`
  });
  const orderingBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
      ,{ binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
    ]
  });
  const orderingPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [orderingBindGroupLayout]
  });
  const orderingPipelines = new Map();
  const orderingPipeline = (entryPoint) => {
    if (!orderingPipelines.has(entryPoint)) {
      orderingPipelines.set(entryPoint, device.createComputePipeline({
        layout: orderingPipelineLayout,
        compute: { module: orderingShader, entryPoint }
      }));
    }
    return orderingPipelines.get(entryPoint);
  };
  const orderingBindGroup = device.createBindGroup({
    layout: orderingBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: resources.tileTableBuffer } },
      { binding: 1, resource: { buffer: resources.referenceListBuffer } },
      { binding: 2, resource: { buffer: orderedReferenceBuffer } },
      { binding: 3, resource: { buffer: orderingSummaryBuffer } },
      { binding: 4, resource: { buffer: orderingParamsBuffer } }
      ,{ binding: 5, resource: { buffer: resources.executionPlanBuffer } }
    ]
  });
  let sortSeedSubmissionCount = 0;
  let sortCompareSubmissionCount = 0;
  let sortValidationSubmissionCount = 0;
  function submitOrdering(entryPoint, batch, bitonicK = 0, bitonicJ = 0) {
    orderingParamsData.set([
      resources.tileCount,
      referenceCapacity,
      requiredPaddedReferenceCapacity,
      94,
      batch.start,
      batch.end,
      bitonicK,
      bitonicJ
    ]);
    device.queue.writeBuffer(orderingParamsBuffer, 0, orderingParamsData);
    const encoder = device.createCommandEncoder({
      label: `phase3-bounded-production-${entryPoint}-encoder`
    });
    const pass = encoder.beginComputePass({
      label: `phase3-bounded-production-${entryPoint}-pass`
    });
    pass.setPipeline(orderingPipeline(entryPoint));
    pass.setBindGroup(0, orderingBindGroup);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil((batch.end - batch.start) / 64)));
    pass.end();
    submitProductionStage(device, encoder);
  }
  for (const batch of referenceBatches) {
    submitOrdering('seedOrderedReferences', batch);
    sortSeedSubmissionCount += 1;
  }
  for (let bitonicK = 2; bitonicK <= sortCapacityLimit; bitonicK *= 2) {
    for (let bitonicJ = bitonicK / 2; bitonicJ >= 1; bitonicJ /= 2) {
      for (const batch of referenceBatches) {
        submitOrdering('compareSwapBitonicStage', batch, bitonicK, bitonicJ);
        sortCompareSubmissionCount += 1;
      }
    }
  }
  for (const batch of referenceBatches) {
    submitOrdering('validateOrderedReferences', batch);
    sortValidationSubmissionCount += 1;
  }

  const outputTexture = outputTextureOverride ?? device.createTexture({
    label: 'phase3-bounded-production-tile-compositor-output-texture',
    size: { width: outputWidth, height: outputHeight },
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING
  });
  const outputTextureReused = outputTextureOverride != null;
  const bytesPerRow = alignTo(outputWidth * 4, 256);
  const textureReadbackBuffer = device.createBuffer({
    size: bytesPerRow * outputHeight,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const summaryData = new Float32Array(COMPOSITOR_SUMMARY_FLOAT_COUNT);
  const summaryBuffer = createBuffer(
    device, summaryData,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const paramsData = new Float32Array([
    resources.tileCount, resources.tileCols, resources.tileRows,
    referenceCapacity, outputWidth, outputHeight, outputWidth, outputHeight
  ]);
  const paramsBuffer = createBuffer(device, paramsData, GPUBufferUsage.UNIFORM);
  const maximumReferenceChunkCount = Math.max(
    1,
    resources.executionPlanTopology?.maximumReferenceChunkCount ??
      Math.ceil(referenceCapacity / executionLimits.compositorReferenceChunkSize)
  );
  const uniformAlignment = Math.max(
    256,
    Number(device?.limits?.minUniformBufferOffsetAlignment) || 256
  );
  const executionParamsData = new Uint8Array(
    maximumReferenceChunkCount * uniformAlignment
  );
  const executionParamsView = new DataView(executionParamsData.buffer);
  for (let chunk = 0; chunk < maximumReferenceChunkCount; chunk += 1) {
    const byteOffset = chunk * uniformAlignment;
    executionParamsView.setUint32(
      byteOffset + 0,
      chunk * executionLimits.compositorReferenceChunkSize,
      true
    );
    executionParamsView.setUint32(
      byteOffset + 4,
      executionLimits.compositorReferenceChunkSize,
      true
    );
    executionParamsView.setUint32(byteOffset + 8, referenceCapacity, true);
    executionParamsView.setUint32(byteOffset + 12, 118, true);
  }
  const executionParamsBuffer = createBuffer(
    device, executionParamsData,
    GPUBufferUsage.UNIFORM
  );
  const accumulationBuffer = device.createBuffer({
    label: 'phase3-bounded-production-pixel-accumulation-buffer',
    size: Math.max(16, outputWidth * outputHeight * 16),
    usage: GPUBufferUsage.STORAGE
  });
  const evidenceData = new Uint32Array(10);
  const evidenceBuffer = createBuffer(
    device, evidenceData,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  );
  const summaryReadbackBuffer = device.createBuffer({
    size: summaryData.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const executionPlanReadbackBuffer = device.createBuffer({
    size: 16 * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const compositorShader = device.createShaderModule({
    label: 'phase3-bounded-production-tile-compositor-wgsl',
    code: `
struct Params {
  tileCount: f32,
  tileCols: f32,
  tileRows: f32,
  referenceCapacity: f32,
  outputWidth: f32,
  outputHeight: f32,
  canvasWidth: f32,
  canvasHeight: f32,
};
struct ExecutionParams {
  referenceChunkStart: u32,
  referenceChunkSize: u32,
  referenceCount: u32,
  statusCode: u32,
};
@group(0) @binding(0) var<storage, read> tileInputs: array<vec4f>;
@group(0) @binding(1) var<storage, read> tileTable: array<vec4f>;
@group(0) @binding(2) var<storage, read> orderedReferences: array<vec4f>;
@group(0) @binding(3) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<storage, read_write> compositorSummary: array<vec4f>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> pixelAccumulation: array<vec4f>;
@group(0) @binding(7) var<uniform> execution: ExecutionParams;
@group(0) @binding(8) var<storage, read_write> evidence: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> executionPlan: array<atomic<u32>>;

fn planReady() -> bool {
  return atomicLoad(&executionPlan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.status}u]) ==
    ${PRODUCTION_TILE_EXECUTION_PLAN_STATUS.ready}u;
}

fn sampleConic(conicAndSort: vec4f, radius: f32) -> vec3f {
  let fallbackConic = 1.0 / max(radius * radius, 1.0);
  return vec3f(
    select(fallbackConic, abs(conicAndSort.x), abs(conicAndSort.x) > 0.0),
    conicAndSort.y,
    select(fallbackConic, abs(conicAndSort.w), abs(conicAndSort.w) > 0.0)
  );
}
fn gaussianWeight(pixel: vec2f, center: vec2f, conic: vec3f) -> f32 {
  let d = pixel - center;
  let power = conic.x * d.x * d.x + 2.0 * conic.y * d.x * d.y + conic.z * d.y * d.y;
  return exp(-0.5 * clamp(power, 0.0, 80.0));
}

@compute @workgroup_size(8, 8)
fn clearProductionAccumulation(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u32(params.outputWidth) || id.y >= u32(params.outputHeight)) { return; }
  let pixelIndex = id.y * u32(params.outputWidth) + id.x;
  pixelAccumulation[pixelIndex] = vec4f(0.0);
  if (planReady()) {
    textureStore(outputTexture, vec2i(id.xy), vec4f(0.0));
  }
}

@compute @workgroup_size(8, 8)
fn compositeReferenceChunk(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_id) localId: vec3u
) {
  if (!planReady()) { return; }
  let tileSizeX = max(params.outputWidth / max(params.tileCols, 1.0), 1.0);
  let tileSizeY = max(params.outputHeight / max(params.tileRows, 1.0), 1.0);
  let subtileCols = u32(max(ceil(tileSizeX / 8.0), 1.0));
  let subtileRows = u32(max(ceil(tileSizeY / 8.0), 1.0));
  let tileX = workgroupId.x / subtileCols;
  let tileY = workgroupId.y / subtileRows;
  if (tileX >= u32(params.tileCols) || tileY >= u32(params.tileRows)) { return; }
  let tile = tileY * u32(params.tileCols) + tileX;
  let table = tileTable[tile];
  if (table.w != 84.0 || table.y <= 0.0) { return; }
  let subtileX = workgroupId.x % subtileCols;
  let subtileY = workgroupId.y % subtileRows;
  let tileStartX = u32(floor(f32(tileX) * tileSizeX));
  let tileStartY = u32(floor(f32(tileY) * tileSizeY));
  let tileEndX = min(u32(ceil(f32(tileX + 1u) * tileSizeX)), u32(params.outputWidth));
  let tileEndY = min(u32(ceil(f32(tileY + 1u) * tileSizeY)), u32(params.outputHeight));
  let pixelX = tileStartX + subtileX * 8u + localId.x;
  let pixelY = tileStartY + subtileY * 8u + localId.y;
  let count = u32(table.y);
  let chunkEnd = min(count, execution.referenceChunkStart + execution.referenceChunkSize);
  if (execution.referenceChunkStart >= chunkEnd) { return; }
  let offset = u32(table.x);
  let evidenceLane = subtileX == 0u && subtileY == 0u &&
    localId.x == 0u && localId.y == 0u;
  if (evidenceLane) {
    if (execution.referenceChunkStart == 0u) {
      atomicAdd(&evidence[8], 1u);
      atomicMax(&evidence[9], count);
    }
    for (var slot = execution.referenceChunkStart; slot < chunkEnd; slot += 1u) {
      let splatRef = orderedReferences[offset + slot];
      let base = u32(max(splatRef.x, 0.0)) * 3u;
      let a = tileInputs[base + 0u];
      let b = tileInputs[base + 1u];
      let c = tileInputs[base + 2u];
      atomicAdd(&evidence[0], 1u);
      atomicAdd(&executionPlan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.compositedReferenceCount}u], 1u);
      if (splatRef.z != 0.0) { atomicStore(&evidence[1], 1u); }
      if (splatRef.w != 0.0) { atomicStore(&evidence[2], 1u); }
      if (c.w > 0.0 || any(c.xyz != vec3f(0.0))) { atomicStore(&evidence[3], 1u); }
      if (a.z > 0.0 || b.x != 0.0 || b.y != 0.0 || b.w != 0.0) {
        atomicStore(&evidence[4], 1u);
      }
      if (a.z > 0.0 && b.x > 0.0 && b.w > 0.0 &&
          (abs(b.x - b.w) > 0.000001 || abs(b.y) > 0.000001)) {
        atomicStore(&evidence[5], 1u);
        atomicAdd(&evidence[6], 1u);
      } else {
        atomicAdd(&evidence[7], 1u);
      }
    }
  }
  if (pixelX >= tileEndX || pixelY >= tileEndY) { return; }
  let pixelIndex = pixelY * u32(params.outputWidth) + pixelX;
  var accumulated = pixelAccumulation[pixelIndex];
  let pixel = vec2f(f32(pixelX) + 0.5, f32(pixelY) + 0.5);
  for (var slot = execution.referenceChunkStart; slot < chunkEnd; slot += 1u) {
    let splatRef = orderedReferences[offset + slot];
    let sampleBase = u32(max(splatRef.x, 0.0)) * 3u;
    let a = tileInputs[sampleBase + 0u];
    let b = tileInputs[sampleBase + 1u];
    let c = tileInputs[sampleBase + 2u];
    let weight = gaussianWeight(pixel, a.xy, sampleConic(b, max(a.z, 1.0)));
    let sampleAlpha = clamp(c.w * weight, 0.0, 0.98);
    let remaining = max(1.0 - accumulated.w, 0.0);
    accumulated = vec4f(
      accumulated.xyz +
        remaining * clamp(c.xyz, vec3f(0.0), vec3f(1.0)) * sampleAlpha,
      accumulated.w + remaining * sampleAlpha
    );
  }
  pixelAccumulation[pixelIndex] = accumulated;
}

@compute @workgroup_size(8, 8)
fn writeProductionOutput(@builtin(global_invocation_id) id: vec3u) {
  if (!planReady()) { return; }
  if (id.x >= u32(params.outputWidth) || id.y >= u32(params.outputHeight)) { return; }
  let pixelIndex = id.y * u32(params.outputWidth) + id.x;
  let accumulated = pixelAccumulation[pixelIndex];
  textureStore(outputTexture, vec2i(id.xy), vec4f(accumulated.xyz, clamp(accumulated.w, 0.0, 1.0)));
}

@compute @workgroup_size(1)
fn finalizeSummary() {
  let planStatus = atomicLoad(&executionPlan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.status}u]);
  let plannedRefs = atomicLoad(&executionPlan[${PRODUCTION_TILE_EXECUTION_PLAN_WORD.requiredReferenceCount}u]);
  let nonEmpty = f32(atomicLoad(&evidence[8]));
  let maxRefs = f32(atomicLoad(&evidence[9]));
  let totalRefs = f32(atomicLoad(&evidence[0]));
  let available = totalRefs > 0.0;
  let depthSeen = f32(atomicLoad(&evidence[1]));
  let sortSeen = f32(atomicLoad(&evidence[2]));
  let attributesSeen = f32(atomicLoad(&evidence[3]));
  let footprintSeen = f32(atomicLoad(&evidence[4]));
  let scaleAwareSeen = f32(atomicLoad(&evidence[5]));
  let anisotropicRefs = f32(atomicLoad(&evidence[6]));
  let fallbackRefs = f32(atomicLoad(&evidence[7]));
  let orderedReady = available && sortSeen == 1.0;
  let attributeReady = available && attributesSeen == 1.0;
  let outputIsCanvasSized = params.outputWidth > params.tileCols || params.outputHeight > params.tileRows;
  let fullScreenPixelWork = max(params.outputWidth * params.outputHeight, 1.0);
  let tileSizeX = max(params.outputWidth / max(params.tileCols, 1.0), 1.0);
  let tileSizeY = max(params.outputHeight / max(params.tileRows, 1.0), 1.0);
  let activeTilePixelWork = nonEmpty * tileSizeX * tileSizeY;
  let avoided = max(fullScreenPixelWork - activeTilePixelWork, 0.0);
  let productionReady = planStatus == ${PRODUCTION_TILE_EXECUTION_PLAN_STATUS.ready}u &&
    totalRefs == f32(plannedRefs) && orderedReady && attributeReady && outputIsCanvasSized;
  compositorSummary[0] = vec4f(params.tileCount, nonEmpty, totalRefs, totalRefs);
  compositorSummary[1] = vec4f(1.0, select(0.0, 1.0, available), select(0.0, 1.0, available), maxRefs);
  compositorSummary[2] = vec4f(0.0, 87.0, totalRefs, totalRefs);
  compositorSummary[3] = vec4f(depthSeen, sortSeen, select(0.0, 1.0, orderedReady), 1.0);
  compositorSummary[4] = vec4f(attributesSeen, footprintSeen, 1.0, select(0.0, 1.0, orderedReady));
  compositorSummary[5] = vec4f(select(0.0, 1.0, attributeReady), select(0.0, 1.0, attributeReady), totalRefs, select(0.0, 1.0, outputIsCanvasSized));
  compositorSummary[6] = vec4f(select(0.0, 1.0, productionReady), select(0.0, 1.0, orderedReady), select(0.0, 1.0, nonEmpty > 0.0), 1.0);
  compositorSummary[7] = vec4f(nonEmpty, params.tileCount - nonEmpty, activeTilePixelWork, avoided);
  compositorSummary[8] = vec4f(avoided / fullScreenPixelWork, select(0.0, 1.0, productionReady), select(0.0, 1.0, productionReady), select(0.0, 1.0, nonEmpty > 0.0));
  compositorSummary[9] = vec4f(scaleAwareSeen, anisotropicRefs, anisotropicRefs / max(totalRefs, 1.0), fallbackRefs);
}
`
  });
  const compositorBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform', hasDynamicOffset: true } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
      ,{ binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
    ]
  });
  const compositorPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [compositorBindGroupLayout]
  });
  const compositorPipelines = new Map();
  const compositorPipeline = (entryPoint) => {
    if (!compositorPipelines.has(entryPoint)) {
      compositorPipelines.set(entryPoint, device.createComputePipeline({
        layout: compositorPipelineLayout,
        compute: { module: compositorShader, entryPoint }
      }));
    }
    return compositorPipelines.get(entryPoint);
  };
  const compositorBindGroup = device.createBindGroup({
    layout: compositorBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: resources.inputBuffer } },
      { binding: 1, resource: { buffer: resources.tileTableBuffer } },
      { binding: 2, resource: { buffer: orderedReferenceBuffer } },
      { binding: 3, resource: outputTexture.createView() },
      { binding: 4, resource: { buffer: summaryBuffer } },
      { binding: 5, resource: { buffer: paramsBuffer } },
      { binding: 6, resource: { buffer: accumulationBuffer } },
      { binding: 7, resource: {
        buffer: executionParamsBuffer, offset: 0, size: 16
      } },
      { binding: 8, resource: { buffer: evidenceBuffer } }
      ,{ binding: 9, resource: { buffer: resources.executionPlanBuffer } }
    ]
  });
  const tileSizeX = Math.max(outputWidth / Math.max(resources.tileCols, 1), 1);
  const tileSizeY = Math.max(outputHeight / Math.max(resources.tileRows, 1), 1);
  const subtileCols = Math.max(1, Math.ceil(tileSizeX / 8));
  const subtileRows = Math.max(1, Math.ceil(tileSizeY / 8));
  const pixelDispatch = [Math.max(1, Math.ceil(outputWidth / 8)),
    Math.max(1, Math.ceil(outputHeight / 8))];
  const tileDispatch = [Math.max(1, resources.tileCols * subtileCols),
    Math.max(1, resources.tileRows * subtileRows)];
  function submitCompositor(entryPoint, dispatch) {
    const encoder = device.createCommandEncoder({
      label: `phase3-bounded-production-${entryPoint}-encoder`
    });
    const pass = encoder.beginComputePass({
      label: `phase3-bounded-production-${entryPoint}-pass`
    });
    pass.setPipeline(compositorPipeline(entryPoint));
    pass.setBindGroup(0, compositorBindGroup, [0]);
    pass.dispatchWorkgroups(...dispatch);
    pass.end();
    submitProductionStage(device, encoder);
  }
  submitCompositor('clearProductionAccumulation', pixelDispatch);
  let compositorChunkSubmissionCount = 0;
  const chunkSize = executionLimits.compositorReferenceChunkSize;
  const compositorWorkgroupsPerChunk = Math.max(
    1, tileDispatch[0] * tileDispatch[1]
  );
  const compositorChunksPerSubmission = Math.max(
    1,
    Math.floor(
      executionLimits.maxInvocationsPerSubmission /
      (compositorWorkgroupsPerChunk * 64)
    )
  );
  for (
    let chunkBase = 0;
    chunkBase < maximumReferenceChunkCount;
    chunkBase += compositorChunksPerSubmission
  ) {
    const encoder = device.createCommandEncoder({
      label: 'phase3-bounded-production-compositeReferenceChunk-encoder'
    });
    const pass = encoder.beginComputePass({
      label: 'phase3-bounded-production-compositeReferenceChunk-pass'
    });
    pass.setPipeline(compositorPipeline('compositeReferenceChunk'));
    const chunkEnd = Math.min(
      maximumReferenceChunkCount,
      chunkBase + compositorChunksPerSubmission
    );
    for (let chunk = chunkBase; chunk < chunkEnd; chunk += 1) {
      pass.setBindGroup(0, compositorBindGroup, [chunk * uniformAlignment]);
      pass.dispatchWorkgroupsIndirect(
        resources.compositorIndirectBuffer,
        chunk * 3 * Uint32Array.BYTES_PER_ELEMENT
      );
      compositorChunkSubmissionCount += 1;
    }
    pass.end();
    submitProductionStage(device, encoder);
  }
  submitCompositor('writeProductionOutput', pixelDispatch);
  const outputPassSubmitted = true;
  {
    const encoder = device.createCommandEncoder({
      label: 'phase3-bounded-production-finalize-and-readback-encoder'
    });
    const pass = encoder.beginComputePass({
      label: 'phase3-bounded-production-finalize-summary-pass'
    });
    pass.setPipeline(compositorPipeline('finalizeSummary'));
    pass.setBindGroup(0, compositorBindGroup, [0]);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyTextureToBuffer(
      { texture: outputTexture },
      { buffer: textureReadbackBuffer, bytesPerRow, rowsPerImage: outputHeight },
      { width: outputWidth, height: outputHeight, depthOrArrayLayers: 1 }
    );
    encoder.copyBufferToBuffer(
      summaryBuffer, 0, summaryReadbackBuffer, 0, summaryData.byteLength
    );
    encoder.copyBufferToBuffer(
      orderingSummaryBuffer, 0, orderingSummaryReadbackBuffer, 0,
      orderingSummaryData.byteLength
    );
    encoder.copyBufferToBuffer(
      resources.executionPlanBuffer, 0, executionPlanReadbackBuffer, 0,
      16 * Uint32Array.BYTES_PER_ELEMENT
    );
    submitProductionStage(device, encoder);
  }
  const productionPresentationPromise =
    typeof onProductionSubmitted === 'function'
      ? Promise.resolve(onProductionSubmitted({
          outputTexture,
          outputTextureReused,
          outputWidth,
          outputHeight,
          executionPlanContract: resources.executionPlanContract
        }))
      : Promise.resolve(null);
  if (typeof device.queue.onSubmittedWorkDone === 'function') {
    await device.queue.onSubmittedWorkDone();
  }
  await summaryReadbackBuffer.mapAsync(GPUMapMode.READ);
  const compositorSummary = new Float32Array(
    summaryReadbackBuffer.getMappedRange().slice(0)
  );
  summaryReadbackBuffer.unmap();
  await orderingSummaryReadbackBuffer.mapAsync(GPUMapMode.READ);
  const orderingSummaryRaw = new Uint32Array(
    orderingSummaryReadbackBuffer.getMappedRange().slice(0)
  );
  orderingSummaryReadbackBuffer.unmap();
  await textureReadbackBuffer.mapAsync(GPUMapMode.READ);
  const textureReadback = new Uint8Array(
    textureReadbackBuffer.getMappedRange().slice(0)
  );
  textureReadbackBuffer.unmap();
  await executionPlanReadbackBuffer.mapAsync(GPUMapMode.READ);
  const executionPlanRaw = new Uint32Array(
    executionPlanReadbackBuffer.getMappedRange().slice(0)
  );
  executionPlanReadbackBuffer.unmap();
  const executionPlanObserver = readProductionTileExecutionPlanObserver(
    executionPlanRaw,
    resources.executionPlanContract
  );
  const productionPresentation = await productionPresentationPromise;

  const sortedReferenceCount = executionPlanObserver.sortedReferenceCount;
  const compositedReferenceCount = executionPlanObserver.compositedReferenceCount;
  const observedRequiredReferenceCount =
    executionPlanObserver.requiredReferenceCount;
  const observedRequiredPaddedReferenceCapacity =
    executionPlanObserver.requiredPaddedReferenceCapacity;
  const outputTextureWriteCompleted =
    outputPassSubmitted &&
    executionPlanObserver.executionCompletionContract
      ?.executionCompletionReady === true;
  const allStagesCompleted =
    resources.executionPlanContract?.gpuExecutionPlanReady === true &&
    resources.recordCount > 0 &&
    executionPlanObserver.executionCompletionContract
      ?.executionCompletionReady === true &&
    sortedReferenceCount === observedRequiredReferenceCount &&
    compositedReferenceCount === observedRequiredReferenceCount &&
    outputTextureWriteCompleted &&
    (orderingSummaryRaw[20] ?? 0) === 0;
  const boundedExecutionContract = buildProductionGpuExecutionContract({
    stage: 'production-tile-reference-sort-and-compositor',
    limits: executionLimits,
    inputRecordCount: resources.recordCount,
    inputReferenceCount: observedRequiredReferenceCount,
    completedRecordCount: resources.recordCount,
    completedReferenceCount: compositedReferenceCount,
    countSubmissionCount:
      resources.boundedExecutionContract?.countSubmissionCount ?? 0,
    scatterSubmissionCount:
      resources.boundedExecutionContract?.scatterSubmissionCount ?? 0,
    sortSeedSubmissionCount,
    sortCompareSubmissionCount,
    sortValidationSubmissionCount,
    compositorChunkSubmissionCount,
    maximumRecordsInSubmission:
      resources.boundedExecutionContract?.maximumRecordsInSubmission ?? 0,
    maximumReferencesInSubmission: Math.min(
      observedRequiredPaddedReferenceCapacity,
      executionLimits.referenceBatchSize
    ),
    maximumReferencesPerPixelInvocation: Math.min(
      executionPlanObserver.maxReferencesPerTile,
      chunkSize
    ),
    gpuResourceLineageMaintained: allStagesCompleted,
    recordReferenceCapacitySeparated: true,
    silentDropAllowed: false,
    schedulerContinuationUsed: false,
    allStagesCompleted,
    reason: allStagesCompleted
      ? null
      : 'production-sort-or-compositor-bounded-execution-incomplete'
  });
  return {
    outputTexture,
    bytesPerRow,
    textureReadback,
    compositorSummary,
    orderingSummaryRaw,
    orderingSummaryData,
    orderedReferenceBuffer,
    orderedReferenceBufferBytes,
    sortCapacityLimit,
    sortSubmissionCount:
      sortSeedSubmissionCount + sortCompareSubmissionCount +
      sortValidationSubmissionCount,
    compositorSubmissionCount: compositorChunkSubmissionCount + 3,
    boundedExecutionContract,
    executionPlanRaw,
    executionPlanObserver,
    outputPassSubmitted,
    outputTextureWriteCompleted,
    productionPresentation,
    outputTextureReused,
    transientBuffers: [
      orderingSummaryBuffer,
      orderingParamsBuffer,
      orderingSummaryReadbackBuffer,
      textureReadbackBuffer,
      summaryBuffer,
      paramsBuffer,
      executionParamsBuffer,
      accumulationBuffer,
      evidenceBuffer,
      summaryReadbackBuffer,
      executionPlanReadbackBuffer
    ]
  };
}
