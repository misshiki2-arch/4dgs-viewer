export const WEBGPU_VISIBLE_RECORD_DRY_RUN_SCHEMA_VERSION =
  'phase3-step2-webgpu-visible-record-dry-run-v1';

export const WEBGPU_VISIBLE_RECORD_COMPUTE_MODE =
  'webgpu-storage-buffer-compute-fixed-record';

export const WEBGPU_VISIBLE_RECORD_PHASE_STEP = 'phase3-step18';

export const WEBGPU_VISIBLE_RECORD_SCAFFOLD_MODE =
  'wgsl-valid-screen-projection-with-tile-counts-offsets-dry-run';

export const WEBGPU_RADIUS_FIELD_COMPUTE_MODE =
  'deferred-covariance-conic-dependent';

export const WEBGPU_CONIC_FIELD_COMPUTE_MODE =
  'deferred-screen-space-covariance-conic-parity';

export const WEBGPU_AABB_FIELD_COMPUTE_MODE =
  'cpu-materialized-aabb-reference';

export const WEBGPU_TILE_RANGE_FIELD_COMPUTE_MODE =
  'deferred-tile-range-from-aabb';

export const WEBGPU_TILE_LIST_GENERATION_COMPUTE_MODE =
  'deferred-prefix-sum-scatter-tile-list-generation';

export const WEBGPU_TILE_LIST_CAPACITY_VALIDATION_COMPUTE_MODE =
  'deferred-capacity-overflow-validation-summary';

export const WEBGPU_TILE_LIST_VALIDATION_UNIT_COMPUTE_MODE =
  'deferred-prefix-sum-scatter-validation-unit';

export {
  WEBGPU_PROJECTION_CONTRACT_NAMES,
  WEBGPU_PROJECTION_CONTRACT_SCHEMA_VERSION,
  WEBGPU_PROJECTION_PARAM_MODES,
  WEBGPU_PROJECTION_SOURCE_POSITION_MODE
} from './common_4dgs_projection_contracts.js';

export const WEBGPU_VISIBLE_RECORD_FLOATS = 12;

export const WEBGPU_VISIBLE_RECORD_FIELDS = Object.freeze([
  ['srcIndex', 0, 1],
  ['valid', 1, 1],
  ['px', 2, 1],
  ['py', 3, 1],
  ['depth', 4, 1],
  ['aabb', 5, 4],
  ['reserved', 9, 3]
]);

export const WEBGPU_VISIBLE_RECORD_IMPLEMENTED_FIELDS = Object.freeze([
  'srcIndex',
  'valid',
  'px',
  'py',
  'depth',
  'aabb'
]);

export const WEBGPU_VISIBLE_RECORD_WGSL_COMPUTED_FIELDS = Object.freeze([
  'srcIndex',
  'valid',
  'px',
  'py',
  'depth'
]);

export const WEBGPU_VISIBLE_RECORD_REFERENCE_ASSISTED_FIELDS = Object.freeze([]);

export const WEBGPU_VISIBLE_RECORD_CPU_MATERIALIZED_FIELDS = Object.freeze([
  'aabb'
]);

export const WEBGPU_VISIBLE_RECORD_FIELD_COMPUTE_MODES = Object.freeze({
  srcIndex: 'wgsl-candidate-buffer',
  valid: 'wgsl-candidate-bounds-state-position-projection-gate',
  px: 'wgsl-state-position-projection-contract',
  py: 'wgsl-state-position-projection-contract',
  depth: 'wgsl-state-position-projection-contract',
  aabb: WEBGPU_AABB_FIELD_COMPUTE_MODE,
  radius: WEBGPU_RADIUS_FIELD_COMPUTE_MODE,
  conic: WEBGPU_CONIC_FIELD_COMPUTE_MODE,
  tileRange: WEBGPU_TILE_RANGE_FIELD_COMPUTE_MODE,
  tileListGeneration: WEBGPU_TILE_LIST_GENERATION_COMPUTE_MODE,
  tileListCapacityValidation: WEBGPU_TILE_LIST_CAPACITY_VALIDATION_COMPUTE_MODE,
  tileListValidationUnit: WEBGPU_TILE_LIST_VALIDATION_UNIT_COMPUTE_MODE
});

export const WEBGPU_VISIBLE_RECORD_DEFERRED_FIELDS = Object.freeze([
  'radius',
  'conic',
  'alpha',
  'tileRange',
  'colorAlpha.rgb',
  'SH',
  '4D conditional covariance full parity',
  'compaction',
  'depth sort',
  'tile-list GPU generation',
  'tile-list capacity/overflow validation summary',
  'prefix-sum/scatter validation unit',
  'display connection'
]);

export function cloneWebGpuVisibleRecordFieldComputeModes() {
  return { ...WEBGPU_VISIBLE_RECORD_FIELD_COMPUTE_MODES };
}
