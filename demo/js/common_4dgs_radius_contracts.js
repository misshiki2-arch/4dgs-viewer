export const WEBGPU_RADIUS_CONTRACT_SCHEMA_VERSION =
  'phase3-step12-radius-contract-v1';

export const WEBGPU_RADIUS_CONTRACT_NAMES = Object.freeze({
  SCREEN_SPACE_COVARIANCE_EIGEN_RADIUS:
    'screen-space-covariance-eigen-radius'
});

export const WEBGPU_RADIUS_COMPUTE_MODES = Object.freeze({
  DEFERRED_COVARIANCE_CONIC_DEPENDENT:
    'deferred-covariance-conic-dependent',
  CPU_REFERENCE_AABB_DEPENDENCY:
    'cpu-reference-radius-for-aabb-materialization'
});

export const WEBGPU_RADIUS_DEPENDENCY_FIELDS = Object.freeze([
  '4D conditional covariance',
  'camera/view rotation',
  'projection Jacobian',
  'screen-space covariance2D',
  'eigenvalue radius',
  'conic parity'
]);

export const WEBGPU_RADIUS_DOWNSTREAM_FIELDS = Object.freeze([
  'aabb',
  'tileRange',
  'tile-list generation',
  'point-sprite raster bounds',
  'conic/alpha evaluation region'
]);

export function createWebGpuRadiusContract({
  computeMode = WEBGPU_RADIUS_COMPUTE_MODES.DEFERRED_COVARIANCE_CONIC_DEPENDENT,
  referenceMode = WEBGPU_RADIUS_COMPUTE_MODES.CPU_REFERENCE_AABB_DEPENDENCY,
  fieldName = 'radius',
  units = 'screen-pixels',
  formula = 'ceil(3 * sqrt(max(lambda1, lambda2)))',
  implementedInWgsl = false
} = {}) {
  return {
    schemaVersion: WEBGPU_RADIUS_CONTRACT_SCHEMA_VERSION,
    contractName: WEBGPU_RADIUS_CONTRACT_NAMES.SCREEN_SPACE_COVARIANCE_EIGEN_RADIUS,
    fieldName,
    units,
    formula,
    computeMode,
    referenceMode,
    implementedInWgsl,
    dependencies: [...WEBGPU_RADIUS_DEPENDENCY_FIELDS],
    downstreamFields: [...WEBGPU_RADIUS_DOWNSTREAM_FIELDS],
    conicRelation: 'Radius and conic share the same screen-space covariance2D source; radius uses eigenvalues while conic uses the inverse covariance.',
    notes: [
      'Radius depends on screen-space covariance eigenvalues, so it should move with covariance/conic rather than as an isolated scalar shortcut.',
      'Current WebGPU fixed-record dry-run still uses CPU materialized radius indirectly for aabb reference materialization.'
    ]
  };
}
