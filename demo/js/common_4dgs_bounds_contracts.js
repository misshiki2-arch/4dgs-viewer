export const WEBGPU_BOUNDS_CONTRACT_SCHEMA_VERSION =
  'phase3-step14-bounds-tile-range-contract-v1';

export const WEBGPU_BOUNDS_CONTRACT_NAMES = Object.freeze({
  SCREEN_SPACE_AABB_FROM_CENTER_RADIUS:
    'screen-space-aabb-from-center-radius',
  INCLUSIVE_TILE_RANGE_FROM_AABB:
    'inclusive-tile-range-from-aabb'
});

export const WEBGPU_BOUNDS_COMPUTE_MODES = Object.freeze({
  CPU_MATERIALIZED_AABB_REFERENCE:
    'cpu-materialized-aabb-reference',
  DEFERRED_TILE_RANGE_FROM_AABB:
    'deferred-tile-range-from-aabb',
  DEFERRED_TILE_LIST_GENERATION:
    'deferred-tile-list-generation'
});

export const WEBGPU_AABB_DEPENDENCY_FIELDS = Object.freeze([
  'px',
  'py',
  'radius',
  'canvas width / height',
  'pixel coordinate convention',
  'floor/ceil raster bounds',
  'clamp to canvas bounds'
]);

export const WEBGPU_TILE_RANGE_DEPENDENCY_FIELDS = Object.freeze([
  'aabb',
  'tile size',
  'tile grid width / height',
  'floor(aabb / tileSize)',
  'clamp to tile grid',
  'inclusive tile min/max range'
]);

export const WEBGPU_BOUNDS_DOWNSTREAM_FIELDS = Object.freeze([
  'tileRange',
  'tile counts pass',
  'tile offsets prefix sum',
  'tile index scatter',
  'tile-list generation',
  'tile composite input'
]);

export function createWebGpuAabbContract({
  computeMode = WEBGPU_BOUNDS_COMPUTE_MODES.CPU_MATERIALIZED_AABB_REFERENCE,
  fieldName = 'aabb',
  units = 'screen-pixels',
  implementedInWgsl = false
} = {}) {
  return {
    schemaVersion: WEBGPU_BOUNDS_CONTRACT_SCHEMA_VERSION,
    contractName: WEBGPU_BOUNDS_CONTRACT_NAMES.SCREEN_SPACE_AABB_FROM_CENTER_RADIUS,
    fieldName,
    units,
    computeMode,
    implementedInWgsl,
    inputs: [...WEBGPU_AABB_DEPENDENCY_FIELDS],
    formula: [
      'minX = clamp(floor(px - max(1, radius)), 0, canvasWidth - 1)',
      'minY = clamp(floor(py - max(1, radius)), 0, canvasHeight - 1)',
      'maxX = clamp(ceil(px + max(1, radius)), 0, canvasWidth - 1)',
      'maxY = clamp(ceil(py + max(1, radius)), 0, canvasHeight - 1)'
    ],
    coordinateConvention: 'screen-space pixel coordinates with canvas bounds',
    clampPolicy: 'inclusive canvas bounds [0, width - 1] and [0, height - 1]',
    upstreamFields: ['px', 'py', 'radius', 'screen-space covariance/conic'],
    downstreamFields: [...WEBGPU_BOUNDS_DOWNSTREAM_FIELDS],
    notes: [
      'AABB remains CPU materialized in Step14 because radius is still deferred behind covariance/conic parity.',
      'Known integer-boundary differences should be classified at the comparison layer before AABB becomes a WebGPU display dependency.'
    ]
  };
}

export function createWebGpuTileRangeContract({
  computeMode = WEBGPU_BOUNDS_COMPUTE_MODES.DEFERRED_TILE_RANGE_FROM_AABB,
  fieldName = 'tileRange',
  units = 'tile-indices',
  implementedInWgsl = false
} = {}) {
  return {
    schemaVersion: WEBGPU_BOUNDS_CONTRACT_SCHEMA_VERSION,
    contractName: WEBGPU_BOUNDS_CONTRACT_NAMES.INCLUSIVE_TILE_RANGE_FROM_AABB,
    fieldName,
    units,
    computeMode,
    implementedInWgsl,
    inputs: [...WEBGPU_TILE_RANGE_DEPENDENCY_FIELDS],
    formula: [
      'tminX = clamp(floor(aabb.minX / tileSize), 0, tileCols - 1)',
      'tminY = clamp(floor(aabb.minY / tileSize), 0, tileRows - 1)',
      'tmaxX = clamp(floor(aabb.maxX / tileSize), 0, tileCols - 1)',
      'tmaxY = clamp(floor(aabb.maxY / tileSize), 0, tileRows - 1)'
    ],
    rangePolicy: 'inclusive tile min/max range; tile-list loops use <= max tile',
    clampPolicy: 'inclusive tile grid bounds [0, tileCols - 1] and [0, tileRows - 1]',
    upstreamFields: ['aabb', 'radius', 'conic'],
    downstreamFields: [
      'tile counts pass',
      'tile offsets prefix sum',
      'tile index scatter',
      'tile-list generation',
      'tile composite input'
    ],
    tileListRelation: 'TileRange is the per-record input to tile-list generation counts, prefix offsets, and scatter outputs.',
    notes: [
      'TileRange remains deferred in Step14. It is the contract boundary between visible record generation and future WebGPU tile binning/list generation.'
    ]
  };
}
