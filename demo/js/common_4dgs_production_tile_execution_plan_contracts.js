export const PRODUCTION_TILE_EXECUTION_PLAN_CONTRACT_VERSION =
  'phase3-production-tile-execution-plan-v1';

export const PRODUCTION_TILE_EXECUTION_PLAN_OBSERVER_SCHEMA_VERSION =
  'phase3-production-tile-execution-plan-terminal-observer-v1';

export const PRODUCTION_TILE_EXECUTION_PLAN_MAGIC = 1184;
export const PRODUCTION_TILE_EXECUTION_PLAN_STATUS = Object.freeze({
  pending: 0,
  ready: 1,
  capacityOverflow: 2,
  executionFailure: 3
});

export const PRODUCTION_TILE_EXECUTION_PLAN_WORD = Object.freeze({
  magic: 0,
  status: 1,
  recordCount: 2,
  tileCount: 3,
  referenceCapacity: 4,
  requiredReferenceCount: 5,
  requiredPaddedReferenceCapacity: 6,
  maxReferencesPerTile: 7,
  totalReferenceChunkCount: 8,
  maxReferenceChunksPerTile: 9,
  scatteredReferenceCount: 10,
  sortedReferenceCount: 11,
  compositedReferenceCount: 12,
  overflowReferenceCount: 13,
  compactOffsetTableReady: 14,
  planIdentity: 15
});

export const PRODUCTION_TILE_EXECUTION_PLAN_WORD_COUNT = 16;
export const PRODUCTION_TILE_EXECUTION_PLAN_CHUNK_SIZE = 64;
export const PRODUCTION_TILE_EXECUTION_PLAN_PRODUCER_BINDING = Object.freeze({
  tileCounts: 0,
  scanSource: 1,
  scanDestination: 2,
  tileTable: 3,
  tileChunkTable: 4,
  plan: 5,
  params: 6,
  compositorDispatches: 7
});

export function buildProductionTileExecutionPlanWgslBindings() {
  const binding = PRODUCTION_TILE_EXECUTION_PLAN_PRODUCER_BINDING;
  return `
@group(0) @binding(${binding.tileCounts}) var<storage, read_write> tileCounts: array<atomic<u32>>;
@group(0) @binding(${binding.scanSource}) var<storage, read> scanSource: array<ScanValue>;
@group(0) @binding(${binding.scanDestination}) var<storage, read_write> scanDestination: array<ScanValue>;
@group(0) @binding(${binding.tileTable}) var<storage, read_write> tileTable: array<vec4f>;
@group(0) @binding(${binding.tileChunkTable}) var<storage, read_write> tileChunkTable: array<vec2u>;
@group(0) @binding(${binding.plan}) var<storage, read_write> plan: array<atomic<u32>>;
@group(0) @binding(${binding.params}) var<uniform> params: Params;
@group(0) @binding(${binding.compositorDispatches}) var<storage, read_write> compositorDispatches: array<u32>;
`;
}

export function buildProductionTileExecutionPlanBindGroupLayoutEntries(
  computeVisibility
) {
  const binding = PRODUCTION_TILE_EXECUTION_PLAN_PRODUCER_BINDING;
  return [
    { binding: binding.tileCounts, visibility: computeVisibility,
      buffer: { type: 'storage' } },
    { binding: binding.scanSource, visibility: computeVisibility,
      buffer: { type: 'read-only-storage' } },
    { binding: binding.scanDestination, visibility: computeVisibility,
      buffer: { type: 'storage' } },
    { binding: binding.tileTable, visibility: computeVisibility,
      buffer: { type: 'storage' } },
    { binding: binding.tileChunkTable, visibility: computeVisibility,
      buffer: { type: 'storage' } },
    { binding: binding.plan, visibility: computeVisibility,
      buffer: { type: 'storage' } },
    { binding: binding.params, visibility: computeVisibility,
      buffer: { type: 'uniform' } },
    { binding: binding.compositorDispatches, visibility: computeVisibility,
      buffer: { type: 'storage' } }
  ];
}

export function buildProductionTileExecutionPlanBindGroupEntries({
  tileCountsBuffer,
  scanSourceBuffer,
  scanDestinationBuffer,
  tileTableBuffer,
  tileChunkTableBuffer,
  planBuffer,
  paramsBuffer,
  compositorIndirectBuffer
} = {}) {
  const binding = PRODUCTION_TILE_EXECUTION_PLAN_PRODUCER_BINDING;
  return [
    { binding: binding.tileCounts, resource: { buffer: tileCountsBuffer } },
    { binding: binding.scanSource, resource: { buffer: scanSourceBuffer } },
    { binding: binding.scanDestination,
      resource: { buffer: scanDestinationBuffer } },
    { binding: binding.tileTable, resource: { buffer: tileTableBuffer } },
    { binding: binding.tileChunkTable,
      resource: { buffer: tileChunkTableBuffer } },
    { binding: binding.plan, resource: { buffer: planBuffer } },
    { binding: binding.params, resource: { buffer: paramsBuffer } },
    { binding: binding.compositorDispatches,
      resource: { buffer: compositorIndirectBuffer } }
  ];
}

function finiteInteger(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.floor(number)) : fallback;
}

function stringOrNull(value) {
  return typeof value === 'string' && value.length > 0 ? value : null;
}

export function resolveProductionTileExecutionPlanTopology({
  tileCount = 0,
  referenceCapacity = 0,
  chunkSize = PRODUCTION_TILE_EXECUTION_PLAN_CHUNK_SIZE
} = {}) {
  const normalizedTileCount = Math.max(1, finiteInteger(tileCount, 1));
  const normalizedReferenceCapacity = Math.max(
    1, finiteInteger(referenceCapacity, 1)
  );
  const normalizedChunkSize = Math.max(1, finiteInteger(chunkSize, 1));
  const scanStageOffsets = [];
  for (let offset = 1; offset < normalizedTileCount; offset *= 2) {
    scanStageOffsets.push(offset);
  }
  return {
    tileCount: normalizedTileCount,
    referenceCapacity: normalizedReferenceCapacity,
    chunkSize: normalizedChunkSize,
    scanStageOffsets,
    scanStageCount: scanStageOffsets.length,
    maximumReferenceChunkCount: Math.ceil(
      normalizedReferenceCapacity / normalizedChunkSize
    ),
    topologySource:
      'viewport-tile-count-and-device-reference-allocation-capacity'
  };
}

export function buildProductionTileExecutionPlanContract({
  status = 'ok',
  resourceIdentity = null,
  sourceTileInputResourceIdentity = null,
  planIdentity = 0,
  tileCount = 0,
  recordCount = 0,
  referenceCapacity = 0,
  scanStageCount = 0,
  maximumReferenceChunkCount = 0,
  gpuPlanBufferCreated = false,
  gpuTileTableCreated = false,
  gpuChunkTableCreated = false,
  gpuPlanGenerated = false,
  scatterConsumesPlan = false,
  sortConsumesPlan = false,
  compositorConsumesPlan = false,
  productionCriticalReadbackUsed = false,
  intermediateCpuControlRoundTripUsed = false,
  sceneDependentCpuPlanMaterialized = false,
  schedulerContinuationUsed = false,
  capacityOverflowFailClosed = false,
  silentDropAllowed = false,
  reason = null
} = {}) {
  const identitiesReady =
    stringOrNull(resourceIdentity) !== null &&
    stringOrNull(sourceTileInputResourceIdentity) !== null;
  const ready =
    status === 'ok' &&
    identitiesReady &&
    finiteInteger(planIdentity) > 0 &&
    finiteInteger(tileCount) > 0 &&
    finiteInteger(recordCount) > 0 &&
    finiteInteger(referenceCapacity) > 0 &&
    finiteInteger(maximumReferenceChunkCount) > 0 &&
    gpuPlanBufferCreated === true &&
    gpuTileTableCreated === true &&
    gpuChunkTableCreated === true &&
    gpuPlanGenerated === true &&
    scatterConsumesPlan === true &&
    sortConsumesPlan === true &&
    compositorConsumesPlan === true &&
    productionCriticalReadbackUsed === false &&
    intermediateCpuControlRoundTripUsed === false &&
    sceneDependentCpuPlanMaterialized === false &&
    schedulerContinuationUsed === false &&
    capacityOverflowFailClosed === true &&
    silentDropAllowed === false;
  return {
    contractVersion: PRODUCTION_TILE_EXECUTION_PLAN_CONTRACT_VERSION,
    status: ready ? 'ok' : status === 'ok' ? 'blocked' : status,
    gpuExecutionPlanReady: ready,
    resourceIdentity: stringOrNull(resourceIdentity),
    sourceTileInputResourceIdentity:
      stringOrNull(sourceTileInputResourceIdentity),
    planIdentity: finiteInteger(planIdentity),
    tileCount: finiteInteger(tileCount),
    recordCount: finiteInteger(recordCount),
    referenceCapacity: finiteInteger(referenceCapacity),
    scanStageCount: finiteInteger(scanStageCount),
    maximumReferenceChunkCount: finiteInteger(maximumReferenceChunkCount),
    chunkSize: PRODUCTION_TILE_EXECUTION_PLAN_CHUNK_SIZE,
    planWordCount: PRODUCTION_TILE_EXECUTION_PLAN_WORD_COUNT,
    planWordVocabulary: { ...PRODUCTION_TILE_EXECUTION_PLAN_WORD },
    gpuPlanBufferCreated: gpuPlanBufferCreated === true,
    gpuTileTableCreated: gpuTileTableCreated === true,
    gpuChunkTableCreated: gpuChunkTableCreated === true,
    gpuPlanGenerated: gpuPlanGenerated === true,
    scatterConsumesPlan: scatterConsumesPlan === true,
    sortConsumesPlan: sortConsumesPlan === true,
    compositorConsumesPlan: compositorConsumesPlan === true,
    productionCriticalReadbackUsed:
      productionCriticalReadbackUsed === true,
    intermediateCpuControlRoundTripUsed:
      intermediateCpuControlRoundTripUsed === true,
    sceneDependentCpuPlanMaterialized:
      sceneDependentCpuPlanMaterialized === true,
    schedulerContinuationUsed: schedulerContinuationUsed === true,
    capacityOverflowFailClosed: capacityOverflowFailClosed === true,
    silentDropAllowed: silentDropAllowed === true,
    reason: ready ? null : reason ?? 'production-gpu-execution-plan-not-ready'
  };
}

export function readProductionTileExecutionPlanObserver(words) {
  const values = words ?? [];
  const word = PRODUCTION_TILE_EXECUTION_PLAN_WORD;
  const statusCode = finiteInteger(values[word.status]);
  const requiredReferenceCount = finiteInteger(
    values[word.requiredReferenceCount]
  );
  const requiredPaddedReferenceCapacity = finiteInteger(
    values[word.requiredPaddedReferenceCapacity]
  );
  const scatteredReferenceCount = finiteInteger(
    values[word.scatteredReferenceCount]
  );
  const sortedReferenceCount = finiteInteger(values[word.sortedReferenceCount]);
  const compositedReferenceCount = finiteInteger(
    values[word.compositedReferenceCount]
  );
  const ready =
    finiteInteger(values[word.magic]) === PRODUCTION_TILE_EXECUTION_PLAN_MAGIC &&
    statusCode === PRODUCTION_TILE_EXECUTION_PLAN_STATUS.ready &&
    requiredReferenceCount > 0 &&
    requiredPaddedReferenceCapacity >= requiredReferenceCount &&
    requiredPaddedReferenceCapacity <= finiteInteger(
      values[word.referenceCapacity]
    ) &&
    scatteredReferenceCount === requiredReferenceCount &&
    sortedReferenceCount === requiredReferenceCount &&
    compositedReferenceCount === requiredReferenceCount &&
    finiteInteger(values[word.compactOffsetTableReady]) === 1;
  return {
    schemaVersion: PRODUCTION_TILE_EXECUTION_PLAN_OBSERVER_SCHEMA_VERSION,
    evidenceRole: 'terminal-post-production-submission-observer',
    productionControlInput: false,
    rawPlanWordsPublished: false,
    observerReady: ready,
    magic: finiteInteger(values[word.magic]),
    statusCode,
    planIdentity: finiteInteger(values[word.planIdentity]),
    recordCount: finiteInteger(values[word.recordCount]),
    tileCount: finiteInteger(values[word.tileCount]),
    referenceCapacity: finiteInteger(values[word.referenceCapacity]),
    requiredReferenceCount,
    requiredPaddedReferenceCapacity,
    maxReferencesPerTile: finiteInteger(values[word.maxReferencesPerTile]),
    totalReferenceChunkCount: finiteInteger(
      values[word.totalReferenceChunkCount]
    ),
    maxReferenceChunksPerTile: finiteInteger(
      values[word.maxReferenceChunksPerTile]
    ),
    scatteredReferenceCount,
    sortedReferenceCount,
    compositedReferenceCount,
    overflowReferenceCount: finiteInteger(values[word.overflowReferenceCount]),
    compactOffsetTableReady:
      finiteInteger(values[word.compactOffsetTableReady]) === 1,
    capacityOverflowDetected:
      statusCode === PRODUCTION_TILE_EXECUTION_PLAN_STATUS.capacityOverflow,
    capacityOverflowFailClosed: statusCode !== PRODUCTION_TILE_EXECUTION_PLAN_STATUS.ready
      ? scatteredReferenceCount === 0 && sortedReferenceCount === 0 &&
        compositedReferenceCount === 0
      : true
  };
}
