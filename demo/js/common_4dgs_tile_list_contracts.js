export const WEBGPU_TILE_LIST_CONTRACT_SCHEMA_VERSION =
  'phase3-step15-tile-list-generation-contract-v1';

export const WEBGPU_TILE_LIST_CAPACITY_CONTRACT_SCHEMA_VERSION =
  'phase3-step16-tile-list-capacity-contract-v1';

export const WEBGPU_TILE_LIST_VALIDATION_CONTRACT_SCHEMA_VERSION =
  'phase3-step16-tile-list-validation-summary-contract-v1';

export const WEBGPU_TILE_LIST_VALIDATION_UNIT_CONTRACT_SCHEMA_VERSION =
  'phase3-step17-prefix-sum-scatter-validation-unit-contract-v1';

export const WEBGPU_TILE_COUNTS_OFFSETS_COMPARISON_SURFACE_SCHEMA_VERSION =
  'phase3-step19-tile-counts-offsets-comparison-surface-v1';

export const WEBGPU_TILE_LIST_CONTRACT_NAMES = Object.freeze({
  PREFIX_SUM_SCATTER_TILE_LIST:
    'prefix-sum-scatter-tile-list-from-tile-range',
  CAPACITY_OVERFLOW_POLICY:
    'tile-list-capacity-overflow-policy',
  VALIDATION_SUMMARY:
    'tile-list-validation-summary',
  PREFIX_SUM_SCATTER_VALIDATION_UNIT:
    'prefix-sum-scatter-validation-unit',
  TILE_COUNTS_OFFSETS_COMPARISON_SURFACE:
    'tile-counts-offsets-comparison-surface'
});

export const WEBGPU_TILE_LIST_COMPUTE_MODES = Object.freeze({
  DEFERRED_PREFIX_SUM_SCATTER:
    'deferred-prefix-sum-scatter-tile-list-generation',
  DEFERRED_CAPACITY_OVERFLOW_VALIDATION:
    'deferred-capacity-overflow-validation-summary',
  DEFERRED_PREFIX_SUM_SCATTER_VALIDATION_UNIT:
    'deferred-prefix-sum-scatter-validation-unit',
  DEFERRED_TILE_COUNTS_OFFSETS_COMPARISON_SURFACE:
    'deferred-tile-counts-offsets-comparison-surface',
  WEBGPU_TILE_COUNTS_ONLY:
    'webgpu-compute-tile-counts-only',
  CPU_PREFIX_FROM_WEBGPU_TILE_COUNTS:
    'cpu-prefix-from-webgpu-tile-counts-comparison',
  WEBGPU_TILE_OFFSETS_PREFIX_SUM:
    'webgpu-compute-tile-offsets-prefix-sum',
  CPU_REFERENCE_SCATTER_VALIDATION_BOUNDARY:
    'cpu-reference-scatter-write-cursor-validation-boundary',
  CPU_REFERENCE_TILE_INDICES_SELF_COMPARISON:
    'cpu-reference-tile-indices-self-comparison-surface',
  WEBGPU_TILE_INDICES_SCATTER_COMPARISON:
    'webgpu-compute-tile-indices-scatter-comparison',
  WEBGPU_TILE_LIST_SUMMARY_COMPARISON:
    'non-display-webgpu-tile-list-summary-comparison',
  WEBGPU_TILE_LIST_BACKEND_OUTPUT:
    'non-display-webgpu-tile-list-backend-output',
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

export const WEBGPU_TILE_LIST_VALIDATION_UNITS = Object.freeze([
  'tileCounts-from-tileRange',
  'tileOffsets-from-tileCounts',
  'scatter-indices-from-tileRange-and-offsets',
  'tileListMetadata-from-counts-offsets-scatter'
]);

export function createWebGpuTileCountsOffsetsComparisonSurfaceContract({
  computeMode = WEBGPU_TILE_LIST_COMPUTE_MODES.DEFERRED_TILE_COUNTS_OFFSETS_COMPARISON_SURFACE,
  implementedInWgsl = false
} = {}) {
  return {
    schemaVersion: WEBGPU_TILE_COUNTS_OFFSETS_COMPARISON_SURFACE_SCHEMA_VERSION,
    contractName: WEBGPU_TILE_LIST_CONTRACT_NAMES.TILE_COUNTS_OFFSETS_COMPARISON_SURFACE,
    computeMode,
    implementedInWgsl,
    comparedBuffers: {
      tileCounts: 'uint32[tileCount], exact per-tile count comparison',
      tileOffsets: 'uint32[tileCount + 1], exact exclusive prefix comparison',
      capacitySummary: 'totalTileRefs, maxRefsPerTile, nonEmptyTiles, capacityStatus'
    },
    summaryFields: {
      anyMismatch: 'boolean',
      tileCountsMismatchCount: 'number of tileCounts entries that differ',
      tileOffsetsMismatchCount: 'number of tileOffsets entries that differ',
      totalTileRefsMismatch: 'boolean, terminal offset differs from expected totalTileRefs',
      capacityStatusMismatch: 'boolean, capacity classification differs',
      maxAbsCountDelta: 'maximum absolute integer delta across tileCounts',
      maxAbsOffsetDelta: 'maximum absolute integer delta across tileOffsets',
      firstMismatches: 'bounded list of kind/index/expected/actual/delta entries',
      sampleTiles: 'small deterministic sample of zero, non-empty, max-count, and final tiles'
    },
    mismatchClassification: {
      none: 'no mismatch',
      tileCountsMismatch: 'tileRange iteration or tile counting mismatch',
      tileOffsetsMismatch: 'exclusive prefix sum mismatch',
      totalTileRefsMismatch: 'tileOffsets[tileCount] or sum(tileCounts) mismatch',
      capacityStatusMismatch: 'overflow/capacity classification mismatch',
      shapeMismatch: 'tileCount, counts length, or offsets length mismatch'
    },
    failureBoundaries: {
      counts:
        'Check tileRange source, inclusive min/max policy, tile grid clamp, and visible record filtering.',
      offsets:
        'Check exclusive prefix sum, offsets[0], monotonicity, and tileOffsets[i + 1] - tileOffsets[i].',
      capacity:
        'Check totalTileRefs, maxTileRefs, maxRefsPerTile, nonEmptyTiles, and overflow status.',
      sampling:
        'Use deterministic sampleTiles plus firstMismatches to avoid dumping full buffers during early WebGPU tests.'
    },
    recommendedStep20Unit: {
      name: 'cpu-reference-self-comparison-surface',
      scope:
        'Emit comparisonSummary using CPU reference as both expected and actual before adding any WebGPU counts buffer.',
      successCriteria: [
        'comparisonSummary.anyMismatch is false',
        'tileCountsMismatchCount is 0',
        'tileOffsetsMismatchCount is 0',
        'totalTileRefsMismatch is false',
        'capacityStatusMismatch is false',
        'firstMismatches is empty'
      ]
    },
    recommendedStep21Unit: {
      name: 'webgpu-tile-counts-only-comparison',
      scope:
        'Compute tileCounts from tileRange in WebGPU and compare only tileCounts against the CPU reference; keep tileOffsets, prefix sum, and scatter deferred.',
      computeMode: WEBGPU_TILE_LIST_COMPUTE_MODES.WEBGPU_TILE_COUNTS_ONLY,
      successCriteria: [
        'tileCountsMismatchCount is 0',
        'maxAbsCountDelta is 0',
        'totalTileRefsMismatch is false',
        'firstMismatches is empty',
        'tileOffsets remain CPU reference materialized'
      ]
    },
    recommendedStep22Unit: {
      name: 'tile-offsets-prefix-comparison-from-webgpu-counts',
      scope:
        'Build exclusive tileOffsets from WebGPU tileCounts on the CPU as a staged dry-run and compare them against CPU reference tileOffsets; keep WebGPU prefix sum and scatter deferred.',
      computeMode: WEBGPU_TILE_LIST_COMPUTE_MODES.CPU_PREFIX_FROM_WEBGPU_TILE_COUNTS,
      successCriteria: [
        'tileOffsetsMismatchCount is 0',
        'maxAbsOffsetDelta is 0',
        'totalTileRefsMismatch is false',
        'capacityStatusMismatch is false',
        'firstMismatches is empty',
        'WebGPU prefix sum remains deferred'
      ]
    },
    recommendedStep23Unit: {
      name: 'webgpu-tile-offsets-prefix-sum-comparison',
      scope:
        'Compute exclusive tileOffsets from WebGPU tileCounts in a minimal WebGPU prefix-sum dry-run and compare them against CPU reference tileOffsets; keep scatter and tileIndices deferred.',
      computeMode: WEBGPU_TILE_LIST_COMPUTE_MODES.WEBGPU_TILE_OFFSETS_PREFIX_SUM,
      successCriteria: [
        'tileOffsetsMismatchCount is 0',
        'maxAbsOffsetDelta is 0',
        'totalTileRefsMismatch is false',
        'firstMismatches is empty',
        'scatter and tileIndices remain deferred'
      ]
    },
    recommendedStep24Unit: {
      name: 'scatter-write-cursor-capacity-validation-boundary',
      scope:
        'Validate CPU reference scatter write-cursor initialization, final cursor positions, and capacity bounds before promoting any WebGPU scatter/tileIndices pass.',
      computeMode: WEBGPU_TILE_LIST_COMPUTE_MODES.CPU_REFERENCE_SCATTER_VALIDATION_BOUNDARY,
      successCriteria: [
        'writeCursorInitialValid is true',
        'writeCursorFinalValid is true',
        'scatterOutputValid is true',
        'capacityStatus is no-overflow',
        'firstValidationFailures is empty',
        'full WebGPU scatter and tileIndices generation remain deferred'
      ]
    },
    recommendedStep25Unit: {
      name: 'cpu-reference-tile-indices-self-comparison-surface',
      scope:
        'Materialize CPU reference tileIndices for a bounded dry-run comparison summary, compare the reference against itself, and preserve per-tile ordering/capacity samples before WebGPU scatter is promoted.',
      computeMode: WEBGPU_TILE_LIST_COMPUTE_MODES.CPU_REFERENCE_TILE_INDICES_SELF_COMPARISON,
      successCriteria: [
        'tileIndicesMismatchCount is 0',
        'orderingMismatchCount is 0',
        'capacityStatus is no-overflow',
        'firstMismatches is empty',
        'WebGPU scatter remains deferred'
      ]
    },
    recommendedStep26Unit: {
      name: 'webgpu-tile-indices-scatter-comparison',
      scope:
        'Generate tileIndices in a minimal WebGPU scatter dry-run from tileRange and WebGPU tileOffsets, compare against CPU reference tileIndices, and keep full tile-list generation, sort, and display connection deferred.',
      computeMode: WEBGPU_TILE_LIST_COMPUTE_MODES.WEBGPU_TILE_INDICES_SCATTER_COMPARISON,
      successCriteria: [
        'tileIndicesMismatchCount is 0',
        'orderingMismatchCount is 0',
        'capacityStatusMismatch is false',
        'firstMismatches is empty',
        'tileIndices are not stored in JSON as a full buffer',
        'full tile-list generation, sort, and display connection remain deferred'
      ]
    },
    recommendedStep27Unit: {
      name: 'non-display-webgpu-tile-list-summary-comparison',
      scope:
        'Bundle the validated WebGPU tileCounts, tileOffsets, and tileIndices dry-run results into one non-display tile-list summary and compare metadata against the CPU reference without connecting sort or display.',
      computeMode: WEBGPU_TILE_LIST_COMPUTE_MODES.WEBGPU_TILE_LIST_SUMMARY_COMPARISON,
      successCriteria: [
        'countsStatus, offsetsStatus, and indicesStatus are ok',
        'totalTileRefsMismatch is false',
        'capacityStatusMismatch is false',
        'orderingMismatchCount is 0',
        'tileIndices are not stored in JSON as a full buffer',
        'sort and display connection remain deferred'
      ]
    },
    recommendedStep28Unit: {
      name: 'non-display-webgpu-tile-list-backend-output',
      scope:
        'Promote the validated WebGPU tile-list dry-run into a formal non-display backend output with buffer roles, handoff readiness, and explicit blockers before sort or display connection.',
      computeMode: WEBGPU_TILE_LIST_COMPUTE_MODES.WEBGPU_TILE_LIST_BACKEND_OUTPUT,
      successCriteria: [
        'backendOutputReady is true when tileListSummaryComparison is ok',
        'tileCounts, tileOffsets, and tileIndices buffer roles are declared',
        'tileIndicesStoredInJson is false',
        'render handoff blockers are explicit',
        'sort and display connection remain deferred'
      ]
    },
    relationToStep18:
      'Step19 defines how future WebGPU tileCounts/tileOffsets outputs will be compared against tileCountsToOffsetsDryRun without changing the Step18 CPU reference data generation.'
  };
}

export function createWebGpuTileListValidationUnitContract({
  computeMode = WEBGPU_TILE_LIST_COMPUTE_MODES.DEFERRED_PREFIX_SUM_SCATTER_VALIDATION_UNIT,
  implementedInWgsl = false
} = {}) {
  return {
    schemaVersion: WEBGPU_TILE_LIST_VALIDATION_UNIT_CONTRACT_SCHEMA_VERSION,
    contractName: WEBGPU_TILE_LIST_CONTRACT_NAMES.PREFIX_SUM_SCATTER_VALIDATION_UNIT,
    computeMode,
    implementedInWgsl,
    validationUnits: [...WEBGPU_TILE_LIST_VALIDATION_UNITS],
    dependencyOrder: [
      'visible records with tileRange',
      'tile grid dimensions and tileCount',
      'tileCounts pass',
      'tileOffsets exclusive prefix sum',
      'capacity status from totalTileRefs',
      'scatter output using per-tile write cursors',
      'tile-list metadata summary'
    ],
    recommendedStep18Unit: {
      name: 'tileCounts-to-tileOffsets-dry-run',
      scope: 'Generate or compare tileCounts and exclusive tileOffsets only; keep scatter deferred.',
      compareAgainst: 'CPU buildTileLists counts/offsets reference',
      successCriteria: [
        'tileCounts match CPU reference',
        'tileOffsets are monotonic and start at zero',
        'tileOffsets[tileCount] equals sum(tileCounts)',
        'capacityStatus can be derived without writing tileIndices'
      ]
    },
    failureClassification: {
      tileCountsMismatch:
        'tileRange iteration, tile grid clamp, or visible record ordering problem',
      prefixOffsetsMismatch:
        'exclusive prefix sum implementation or tileCount bounds problem',
      capacityMismatch:
        'totalTileRefs, maxTileRefs, or overflow policy problem',
      scatterMismatch:
        'write cursor initialization, per-tile ordering, or index emission problem'
    },
    relationToStep16:
      'Step17 defines the validation units consumed by tileListValidationContract before any WebGPU prefix-sum or scatter implementation is promoted.',
    notes: [
      'Scatter remains deferred. Step18 should begin with tileCounts and tileOffsets so failures can be isolated before tileIndices writes exist.'
    ]
  };
}

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
