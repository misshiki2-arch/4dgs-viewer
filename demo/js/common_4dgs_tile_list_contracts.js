export const WEBGPU_TILE_LIST_CONTRACT_SCHEMA_VERSION =
  'phase3-step15-tile-list-generation-contract-v1';

export const WEBGPU_TILE_LIST_CONTRACT_NAMES = Object.freeze({
  PREFIX_SUM_SCATTER_TILE_LIST:
    'prefix-sum-scatter-tile-list-from-tile-range'
});

export const WEBGPU_TILE_LIST_COMPUTE_MODES = Object.freeze({
  DEFERRED_PREFIX_SUM_SCATTER:
    'deferred-prefix-sum-scatter-tile-list-generation',
  CPU_REFERENCE_TILE_LIST:
    'cpu-reference-tile-list-build'
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
    notes: [
      'Step15 documents the tile-list contract only. Prefix sum and scatter remain CPU/reference concepts until a later WebGPU compute step.'
    ]
  };
}
