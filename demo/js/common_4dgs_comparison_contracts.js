export const COMPARISON_CONTRACT_SCHEMA_VERSION =
  'phase3-step9-comparison-contract-v1';

export const RECORD_COMPARISON_KEYS = Object.freeze({
  ANY_MISMATCH: 'anyMismatch',
  FIELD_MISMATCH_COUNT: 'fieldMismatchCount',
  MAX_ABS_ERROR: 'maxAbsError',
  FIRST_MISMATCHES: 'firstMismatches'
});

export const RECORD_MISMATCH_KEYS = Object.freeze({
  ROW: 'row',
  FIELD: 'field',
  COMPONENT: 'component',
  EXPECTED: 'expected',
  ACTUAL: 'actual',
  ABS_ERROR: 'absError'
});

export const MISMATCH_CLASSIFICATIONS = Object.freeze({
  NONE: 'none',
  WEBGPU_VISIBLE_RECORD_UNAVAILABLE: 'webgpu-visible-record-unavailable',
  WEBGPU_VISIBLE_RECORD_COMPARE_MISSING: 'webgpu-visible-record-compare-missing',
  WEBGPU_FIXED_RECORD_AABB_MISMATCH: 'webgpu-fixed-record-aabb-mismatch',
  WEBGPU_FIXED_RECORD_FIELD_MISMATCH: 'webgpu-fixed-record-field-mismatch'
});

export const TOLERANCE_MODES = Object.freeze({
  ABSOLUTE_PER_COMPONENT: 'absolute-per-component',
  FIELD_SPECIFIC_ABSOLUTE: 'field-specific-absolute'
});

export const DEFAULT_COMPARISON_EPSILON = 1e-3;
export const DEFAULT_MAX_MISMATCHES = 32;

export const FIXED_RECORD_DEFAULT_FIELD_TOLERANCES = Object.freeze({
  srcIndex: DEFAULT_COMPARISON_EPSILON,
  valid: DEFAULT_COMPARISON_EPSILON,
  px: DEFAULT_COMPARISON_EPSILON,
  py: DEFAULT_COMPARISON_EPSILON,
  depth: DEFAULT_COMPARISON_EPSILON,
  aabb: DEFAULT_COMPARISON_EPSILON
});

export function createComparisonToleranceMetadata({
  epsilon = DEFAULT_COMPARISON_EPSILON,
  maxMismatches = DEFAULT_MAX_MISMATCHES,
  fieldTolerances = FIXED_RECORD_DEFAULT_FIELD_TOLERANCES,
  mode = TOLERANCE_MODES.ABSOLUTE_PER_COMPONENT
} = {}) {
  return {
    schemaVersion: COMPARISON_CONTRACT_SCHEMA_VERSION,
    mode,
    epsilon,
    maxMismatches,
    fieldTolerances: { ...fieldTolerances }
  };
}

export function createRecordComparisonResult({
  anyMismatch,
  fieldMismatchCount,
  maxAbsError,
  firstMismatches
}) {
  return {
    [RECORD_COMPARISON_KEYS.ANY_MISMATCH]: !!anyMismatch,
    [RECORD_COMPARISON_KEYS.FIELD_MISMATCH_COUNT]: fieldMismatchCount,
    [RECORD_COMPARISON_KEYS.MAX_ABS_ERROR]: maxAbsError,
    [RECORD_COMPARISON_KEYS.FIRST_MISMATCHES]: Array.isArray(firstMismatches)
      ? firstMismatches
      : []
  };
}

export function createRecordMismatch({
  row,
  field,
  component,
  expected,
  actual,
  absError
}) {
  return {
    [RECORD_MISMATCH_KEYS.ROW]: row,
    [RECORD_MISMATCH_KEYS.FIELD]: field,
    [RECORD_MISMATCH_KEYS.COMPONENT]: component,
    [RECORD_MISMATCH_KEYS.EXPECTED]: expected,
    [RECORD_MISMATCH_KEYS.ACTUAL]: actual,
    [RECORD_MISMATCH_KEYS.ABS_ERROR]: absError
  };
}
