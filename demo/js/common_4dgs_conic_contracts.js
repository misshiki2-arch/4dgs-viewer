export const WEBGPU_CONIC_CONTRACT_SCHEMA_VERSION =
  'phase3-step13-screen-space-conic-contract-v1';

export const WEBGPU_COVARIANCE_CONTRACT_NAMES = Object.freeze({
  SCREEN_SPACE_COVARIANCE_2D:
    'screen-space-covariance2d-from-4d-conditional-covariance'
});

export const WEBGPU_CONIC_CONTRACT_NAMES = Object.freeze({
  INVERSE_SCREEN_SPACE_COVARIANCE:
    'inverse-screen-space-covariance-conic'
});

export const WEBGPU_CONIC_COMPUTE_MODES = Object.freeze({
  DEFERRED_COVARIANCE_PARITY:
    'deferred-screen-space-covariance-conic-parity',
  CPU_REFERENCE_PACKED_PAYLOAD:
    'cpu-reference-conic-for-packed-payload-validation'
});

export const WEBGPU_COVARIANCE_DEPENDENCY_FIELDS = Object.freeze([
  '4D conditional covariance',
  'camera/view rotation',
  'projection Jacobian',
  'focal length / covariance focal contract',
  'prefilter variance',
  'screen-space covariance2D regularization'
]);

export const WEBGPU_CONIC_DEPENDENCY_FIELDS = Object.freeze([
  'screen-space covariance2D',
  'positive determinant',
  'matrix inverse [c/det, -b/det, a/det]',
  'radius eigenvalue parity',
  'alpha evaluation parity'
]);

export const WEBGPU_CONIC_DOWNSTREAM_FIELDS = Object.freeze([
  'radius',
  'aabb',
  'tileRange',
  'tile-list generation',
  'splat alpha / power evaluation',
  'packed visible payload'
]);

export function createWebGpuCovarianceContract({
  computeMode = WEBGPU_CONIC_COMPUTE_MODES.DEFERRED_COVARIANCE_PARITY,
  fieldName = 'screenSpaceCovariance2D',
  units = 'screen-pixel-squared',
  implementedInWgsl = false
} = {}) {
  return {
    schemaVersion: WEBGPU_CONIC_CONTRACT_SCHEMA_VERSION,
    contractName: WEBGPU_COVARIANCE_CONTRACT_NAMES.SCREEN_SPACE_COVARIANCE_2D,
    fieldName,
    units,
    computeMode,
    implementedInWgsl,
    dependencies: [...WEBGPU_COVARIANCE_DEPENDENCY_FIELDS],
    outputs: ['covariance2D.a', 'covariance2D.b', 'covariance2D.c', 'det', 'lambda1', 'lambda2'],
    downstreamFields: ['conic', 'radius', 'aabb', 'tileRange'],
    notes: [
      'This contract captures the shared source of conic and radius. It must stay aligned with the CPU/CUDA reference before either field becomes a WebGPU display dependency.'
    ]
  };
}

export function createWebGpuConicContract({
  computeMode = WEBGPU_CONIC_COMPUTE_MODES.DEFERRED_COVARIANCE_PARITY,
  referenceMode = WEBGPU_CONIC_COMPUTE_MODES.CPU_REFERENCE_PACKED_PAYLOAD,
  fieldName = 'conic',
  units = 'inverse-screen-pixel-squared',
  implementedInWgsl = false
} = {}) {
  return {
    schemaVersion: WEBGPU_CONIC_CONTRACT_SCHEMA_VERSION,
    contractName: WEBGPU_CONIC_CONTRACT_NAMES.INVERSE_SCREEN_SPACE_COVARIANCE,
    fieldName,
    units,
    formula: '[c / det, -b / det, a / det]',
    computeMode,
    referenceMode,
    implementedInWgsl,
    dependencies: [...WEBGPU_CONIC_DEPENDENCY_FIELDS],
    downstreamFields: [...WEBGPU_CONIC_DOWNSTREAM_FIELDS],
    radiusRelation: 'radius uses the same covariance eigenvalues before conic inversion, so conic and radius should move through WGSL with a shared covariance contract.',
    boundsRelation: 'Conic shares the covariance source with radius; radius feeds AABB and tileRange while conic feeds alpha/power evaluation over those bounds.',
    alphaRelation: 'alpha/power evaluation consumes conic as -0.5 * (conic.x * dx^2 + conic.z * dy^2) - conic.y * dx * dy.',
    notes: [
      'Conic is intentionally deferred in Step13. The dry-run keeps comparing the existing fixed-record fields while documenting the covariance/conic contract needed by later WebGPU stages.'
    ]
  };
}
