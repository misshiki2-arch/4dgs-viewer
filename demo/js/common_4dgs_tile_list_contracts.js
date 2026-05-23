export const WEBGPU_TILE_LIST_CONTRACT_SCHEMA_VERSION =
  'phase3-step15-tile-list-generation-contract-v1';

export const WEBGPU_TILE_LIST_CAPACITY_CONTRACT_SCHEMA_VERSION =
  'phase3-step16-tile-list-capacity-contract-v1';

export const WEBGPU_TILE_LIST_VALIDATION_CONTRACT_SCHEMA_VERSION =
  'phase3-step16-tile-list-validation-summary-contract-v1';

export const WEBGPU_TILE_LIST_CONTRACT_NAMES = Object.freeze({
  PREFIX_SUM_SCATTER_TILE_LIST:
    'prefix-sum-scatter-tile-list-from-tile-range',
  CAPACITY_OVERFLOW_POLICY:
    'tile-list-capacity-overflow-policy',
  VALIDATION_SUMMARY:
    'tile-list-validation-summary'
});

export const WEBGPU_TILE_LIST_COMPUTE_MODES = Object.freeze({
  DEFERRED_PREFIX_SUM_SCATTER:
    'deferred-prefix-sum-scatter-tile-list-generation',
  DEFERRED_CAPACITY_OVERFLOW_VALIDATION:
    'deferred-capacity-overflow-validation-summary',
  CPU_REFERENCE_TILE_LIST:
    'cpu-reference-tile-list-build'
});

export const WEBGPU_TILE_LIST_CAPACITY_STATUS = Object.freeze({
  NO_OVERFLOW: 'no-overflow',
  OVERFLOW_DETECTED: 'overflow-detected',
  TRUNCATED: 'truncated',
  NEEDS_RESIZE_OR_SECOND_PASS: 'needs-resize-or-second-pass'
});

export const WEBGPU_TILE_LIST_INPUTS = Object.freeze([
  'visible record id / compacted visible index',
  'tileRange',
  'tile grid width / height',
  'tile count',
  'capacity policy',
  'candidate or visible record order'
]);

export const WEBGPU_TILE_LIST_OUTPUT_BUFFERS = Object.freeze({
  TILE_COUNTS: 'tileCounts',
  TILE_OFFSETS: 'tileOffsets',
  TILE_INDICES: 'tileIndices',
  TILE_LIST_METADATA: 'tileListMetadata'
});

export const WEBGPU_TILE_LIST_STAGES = Object.freeze([
  'tile counts pass',
  'tile offsets prefix sum',
  'tile index scatter',
  'tile-list metadata summary'
]);

export const WEBGPU_TILE_LIST_VALIDATION_CHECKS = Object.freeze([
  'tileCounts validation',
  'prefixOffsets validation',
  'scatter output validation',
  'totalTileRefs consistency'
]);

export function createWebGpuTileListCapacityContract({
  computeMode = WEBGPU_TILE_LIST_COMPUTE_MODES.DEFERRED_CAPACITY_OVERFLOW_VALIDATION,
  implementedInWgsl = false
} = {}) {
  return {
    schemaVersion: WEBGPU_TILE_LIST_CAPACITY_CONTRACT_SCHEMA_VERSION,
    contractName: WEBGPU_TILE_LIST_CONTRACT_NAMES.CAPACITY_OVERFLOW_POLICY,
    computeMode,
    implementedInWgsl,
    capacityFields: {
      maxTileRefs: 'maximum allocated tile-list references for tileIndices',
      maxRefsPerTile: 'maximum references observed or allowed for a single tile',
      totalTileRefs: 'sum(tileCounts), also tileOffsets[tileCount]',
      nonEmptyTiles: 'count of tiles with tileCounts[tile] > 0',
      capacityStatus: 'one of no-overflow, overflow-detected, truncated, needs-resize-or-second-pass'
    },
    overflowPolicy: {
      noOverflow: WEBGPU_TILE_LIST_CAPACITY_STATUS.NO_OVERFLOW,
      overflowDetected: WEBGPU_TILE_LIST_CAPACITY_STATUS.OVERFLOW_DETECTED,
      truncated: WEBGPU_TILE_LIST_CAPACITY_STATUS.TRUNCATED,
      needsResizeOrSecondPass: WEBGPU_TILE_LIST_CAPACITY_STATUS.NEEDS_RESIZE_OR_SECOND_PASS
    },
    allocationPolicy: {
      currentMode: 'metadata-only-deferred-explicit-capacity',
      requiredBeforeGpuGeneration: [
        'choose maxTileRefs allocation or growable buffer strategy',
        'classify overflow before scatter writes exceed capacity',
        'preserve validation summary even when tileIndices is incomplete'
      ]
    },
    relationToTileListContract:
      'Capacity status gates whether tileCounts, tileOffsets, and tileIndices can be consumed by later tile composite stages.'
  };
}

export function createWebGpuTileListValidationContract({
  computeMode = WEBGPU_TILE_LIST_COMPUTE_MODES.DEFERRED_CAPACITY_OVERFLOW_VALIDATION,
  implementedInWgsl = false
} = {}) {
  return {
    schemaVersion: WEBGPU_TILE_LIST_VALIDATION_CONTRACT_SCHEMA_VERSION,
    contractName: WEBGPU_TILE_LIST_CONTRACT_NAMES.VALIDATION_SUMMARY,
    computeMode,
    implementedInWgsl,
    checks: [...WEBGPU_TILE_LIST_VALIDATION_CHECKS],
    summarySchema: {
      tileCountsValid: 'boolean, every per-tile count is finite and within capacity policy',
      prefixOffsetsValid: 'boolean, tileOffsets is monotonic and tileOffsets[0] == 0',
      scatterOutputValid: 'boolean, tileIndices writes stay within [0, totalTileRefs)',
      totalTileRefsConsistent: 'boolean, tileOffsets[tileCount] equals sum(tileCounts)',
      capacityStatus: 'overflow classification copied from tile-list capacity contract',
      firstValidationFailures: 'bounded list of field/stage/index/reason entries'
    },
    comparisonRelation:
      'Validation summary is metadata-only in Step16 and does not alter recordComparison until tile-list buffers become compared outputs.',
    downstreamUse:
      'The summary decides whether later WebGPU prefix-sum/scatter output can advance to sort, compaction, and display connection.'
  };
}

export function createWebGpuTileListContract({
  computeMode = WEBGPU_TILE_LIST_COMPUTE_MODES.DEFERRED_PREFIX_SUM_SCATTER,
  referenceMode = WEBGPU_TILE_LIST_COMPUTE_MODES.CPU_REFERENCE_TILE_LIST,
  implementedInWgsl = false
} = {}) {
  return {
    schemaVersion: WEBGPU_TILE_LIST_CONTRACT_SCHEMA_VERSION,
    contractName: WEBGPU_TILE_LIST_CONTRACT_NAMES.PREFIX_SUM_SCATTER_TILE_LIST,
    computeMode,
    referenceMode,
    implementedInWgsl,
    inputs: [...WEBGPU_TILE_LIST_INPUTS],
    stages: [...WEBGPU_TILE_LIST_STAGES],
    outputBuffers: { ...WEBGPU_TILE_LIST_OUTPUT_BUFFERS },
    outputSchema: {
      tileCounts: 'uint32[tileCount], count of visible records touching each tile',
      tileOffsets: 'uint32[tileCount + 1], exclusive prefix sum of tileCounts',
      tileIndices: 'uint32[totalTileRefs], visible record indices scattered by tile',
      tileListMetadata: 'tileCount, totalTileRefs, maxRefsPerTile, nonEmptyTiles, capacity status'
    },
    capacityPolicy: {
      mode: 'deferred-explicit-capacity',
      step16Relation: 'Detailed capacity and overflow status lives in tileListCapacityContract.',
      requiredBeforeGpuGeneration: [
        'max tile refs or growable allocation policy',
        'overflow classification',
        'readback-free validation summary'
      ]
    },
    orderingPolicy: 'preserve incoming visible/record order within each tile until sort/compaction contract changes it',
    downstreamFields: [
      'tile composite input',
      'per-tile splat iteration',
      'tile occupancy diagnostics',
      'future WebGPU display connection'
    ],
    validationRelation:
      'tileListValidationContract defines tileCounts, prefixOffsets, scatter output, and totalTileRefs consistency checks before WebGPU tile-list output is consumed.',
    notes: [
      'Step16 keeps tile-list generation deferred and adds capacity, overflow, and validation summary contracts before any prefix-sum or scatter WGSL implementation.'
    ]
  };
}
