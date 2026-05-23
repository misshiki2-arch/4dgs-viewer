export const WEBGPU_INPUT_BUFFER_CONTRACT_SCHEMA_VERSION =
  'phase3-step10-webgpu-input-buffer-contract-v1';

export const WEBGPU_INPUT_BUFFER_NAMES = Object.freeze({
  CANDIDATE_INDICES: 'candidateIndices',
  RAW_XYZ_OPACITY: 'rawXyzOpacity',
  STATE_POSITIONS: 'statePositions',
  CPU_REFERENCE_RECORDS: 'cpuReferenceRecords',
  OUTPUT_FIXED_RECORDS: 'outputFixedRecords',
  PROJECTION_PARAMS: 'projectionParams',
  DISPATCH_PARAMS: 'dispatchParams',
  READBACK: 'readback'
});

export const WEBGPU_INPUT_BUFFER_MODES = Object.freeze({
  CANDIDATE_INDICES: 'screen-coarse-candidate-indices',
  RAW_XYZ_OPACITY: 'candidate-xyz-opacity-step4-fetch-probe',
  STATE_POSITIONS: 'cpu-materialized-4d-state-position',
  CPU_REFERENCE_RECORDS: 'cpu-materialized-fixed-record-reference',
  OUTPUT_FIXED_RECORDS: 'webgpu-compute-fixed-record-output',
  PROJECTION_PARAMS: 'webgpu-projection-contract-params',
  DISPATCH_PARAMS: 'webgpu-dispatch-params',
  READBACK: 'webgpu-fixed-record-readback'
});

export function createWebGpuInputBufferModes() {
  return {
    [WEBGPU_INPUT_BUFFER_NAMES.CANDIDATE_INDICES]: WEBGPU_INPUT_BUFFER_MODES.CANDIDATE_INDICES,
    [WEBGPU_INPUT_BUFFER_NAMES.RAW_XYZ_OPACITY]: WEBGPU_INPUT_BUFFER_MODES.RAW_XYZ_OPACITY,
    [WEBGPU_INPUT_BUFFER_NAMES.STATE_POSITIONS]: WEBGPU_INPUT_BUFFER_MODES.STATE_POSITIONS,
    [WEBGPU_INPUT_BUFFER_NAMES.CPU_REFERENCE_RECORDS]: WEBGPU_INPUT_BUFFER_MODES.CPU_REFERENCE_RECORDS,
    [WEBGPU_INPUT_BUFFER_NAMES.OUTPUT_FIXED_RECORDS]: WEBGPU_INPUT_BUFFER_MODES.OUTPUT_FIXED_RECORDS,
    [WEBGPU_INPUT_BUFFER_NAMES.PROJECTION_PARAMS]: WEBGPU_INPUT_BUFFER_MODES.PROJECTION_PARAMS,
    [WEBGPU_INPUT_BUFFER_NAMES.DISPATCH_PARAMS]: WEBGPU_INPUT_BUFFER_MODES.DISPATCH_PARAMS,
    [WEBGPU_INPUT_BUFFER_NAMES.READBACK]: WEBGPU_INPUT_BUFFER_MODES.READBACK
  };
}

export function createWebGpuInputBufferContract({
  candidateCount = null,
  recordCount = null,
  rawCount = null,
  recordFloats = null,
  outputBufferBytes = null,
  projectionParamMode = null,
  statePositionUploadMode = WEBGPU_INPUT_BUFFER_MODES.STATE_POSITIONS,
  rawBufferUploadMode = WEBGPU_INPUT_BUFFER_MODES.RAW_XYZ_OPACITY
} = {}) {
  return {
    schemaVersion: WEBGPU_INPUT_BUFFER_CONTRACT_SCHEMA_VERSION,
    backend: 'webgpu',
    purpose: 'fixed-record-compute-dry-run',
    inputBufferModes: createWebGpuInputBufferModes(),
    bufferRoles: {
      [WEBGPU_INPUT_BUFFER_NAMES.CANDIDATE_INDICES]: 'candidate index list for compute rows',
      [WEBGPU_INPUT_BUFFER_NAMES.RAW_XYZ_OPACITY]: 'raw xyz/opacity probe data for selected candidates',
      [WEBGPU_INPUT_BUFFER_NAMES.STATE_POSITIONS]: '4D state positions materialized on CPU for this scaffold stage',
      [WEBGPU_INPUT_BUFFER_NAMES.CPU_REFERENCE_RECORDS]: 'CPU fixed-record reference used for validation and assisted fields',
      [WEBGPU_INPUT_BUFFER_NAMES.OUTPUT_FIXED_RECORDS]: 'WGSL fixed-record output buffer',
      [WEBGPU_INPUT_BUFFER_NAMES.PROJECTION_PARAMS]: 'projection contract parameters consumed by WGSL',
      [WEBGPU_INPUT_BUFFER_NAMES.DISPATCH_PARAMS]: 'small uniform buffer for dispatch counts and raw bounds',
      [WEBGPU_INPUT_BUFFER_NAMES.READBACK]: 'mapped buffer used only for dry-run validation'
    },
    counts: {
      candidateCount,
      recordCount,
      rawCount,
      recordFloats,
      outputBufferBytes
    },
    modes: {
      rawBufferUploadMode,
      statePositionUploadMode,
      projectionParamMode
    }
  };
}
