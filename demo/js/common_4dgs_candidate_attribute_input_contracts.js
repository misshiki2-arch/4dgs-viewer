import {
  cameraWorldPositionFromProjectionParams
} from './common_4dgs_projection_contracts.js';

export const PRODUCTION_CANDIDATE_ATTRIBUTE_INPUT_LAYOUT_VERSION =
  'phase3-step122-production-candidate-attribute-input-layout-v1';

export const PRODUCTION_CANDIDATE_ATTRIBUTE_FLOAT_STRIDE = 32;
export const PRODUCTION_CANDIDATE_ATTRIBUTE_VEC4_STRIDE = 8;
export const PRODUCTION_CANDIDATE_ATTRIBUTE_BYTE_STRIDE =
  PRODUCTION_CANDIDATE_ATTRIBUTE_FLOAT_STRIDE * Float32Array.BYTES_PER_ELEMENT;
export const PRODUCTION_CANDIDATE_ATTRIBUTE_STORAGE_BINDING = 4;
export const PRODUCTION_EVALUATOR_STORAGE_BINDING_COUNT = 8;

export const PRODUCTION_CANDIDATE_ATTRIBUTE_FIELD_OFFSETS = Object.freeze({
  fDcRgb: Object.freeze({ floatOffset: 0, floatCount: 3 }),
  meanScale: Object.freeze({ floatOffset: 3, floatCount: 1 }),
  scaleXyz: Object.freeze({ floatOffset: 4, floatCount: 3 }),
  sourceCode: Object.freeze({ floatOffset: 7, floatCount: 1 }),
  fRestDegree2: Object.freeze({ floatOffset: 8, floatCount: 24 })
});

export const PRODUCTION_DEGREE2_SH_COEFFICIENT_FLOAT_OFFSETS = Object.freeze([
  0, 8, 11, 14, 17, 20, 23, 26, 29
]);

export const PRODUCTION_DEGREE2_SH_CONSTANTS = Object.freeze({
  c0: 0.28209479177387814,
  c1: 0.4886025119029199,
  c2: Object.freeze([
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
  ])
});

const MAX_BLOCKED_REASONS = 16;

function finiteLimit(value) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? number : null;
}

function typedArrayHasRequiredLength(value, requiredLength, Type) {
  return value instanceof Type && value.length >= requiredLength;
}

function boundedReasons(reasons) {
  const unique = [...new Set(reasons)];
  return {
    reasons: unique.slice(0, MAX_BLOCKED_REASONS),
    reasonCount: unique.length,
    truncated: unique.length > MAX_BLOCKED_REASONS
  };
}

export function productionCandidateAttributeByteLength(recordCount) {
  const count = Number(recordCount);
  if (!Number.isSafeInteger(count) || count < 0) return null;
  const byteLength = count * PRODUCTION_CANDIDATE_ATTRIBUTE_BYTE_STRIDE;
  return Number.isSafeInteger(byteLength) ? byteLength : null;
}

function buildFieldContract() {
  return Object.freeze([
    Object.freeze({
      field: 'f_dc.rgb',
      ...PRODUCTION_CANDIDATE_ATTRIBUTE_FIELD_OFFSETS.fDcRgb
    }),
    Object.freeze({
      field: 'meanScale',
      ...PRODUCTION_CANDIDATE_ATTRIBUTE_FIELD_OFFSETS.meanScale
    }),
    Object.freeze({
      field: 'scaleXYZ',
      ...PRODUCTION_CANDIDATE_ATTRIBUTE_FIELD_OFFSETS.scaleXyz
    }),
    Object.freeze({
      field: 'sourceCode',
      ...PRODUCTION_CANDIDATE_ATTRIBUTE_FIELD_OFFSETS.sourceCode
    }),
    Object.freeze({
      field: 'f_rest[0..23]',
      ...PRODUCTION_CANDIDATE_ATTRIBUTE_FIELD_OFFSETS.fRestDegree2
    })
  ]);
}

function buildLayoutContract({
  ready,
  recordCount,
  byteLength,
  activeShDegree,
  activeShDegreeT,
  deviceLimits,
  blocked,
  sourceWorksetResourceIdentity,
  resourceIdentity,
  resourceOwnership,
  cameraWorldPosition
}) {
  const maxStorageBufferBindingSize = finiteLimit(
    deviceLimits?.maxStorageBufferBindingSize
  );
  const maxBufferSize = finiteLimit(deviceLimits?.maxBufferSize);
  const maxStorageBuffersPerShaderStage = finiteLimit(
    deviceLimits?.maxStorageBuffersPerShaderStage
  );
  return {
    schemaVersion: PRODUCTION_CANDIDATE_ATTRIBUTE_INPUT_LAYOUT_VERSION,
    status: ready ? 'ready' : 'blocked',
    ready,
    blockedReasons: blocked.reasons,
    blockedReasonCount: blocked.reasonCount,
    blockedReasonsTruncated: blocked.truncated,
    floatStride: PRODUCTION_CANDIDATE_ATTRIBUTE_FLOAT_STRIDE,
    vec4Stride: PRODUCTION_CANDIDATE_ATTRIBUTE_VEC4_STRIDE,
    byteStride: PRODUCTION_CANDIDATE_ATTRIBUTE_BYTE_STRIDE,
    fields: buildFieldContract(),
    shCoefficientFloatOffsets: [
      ...PRODUCTION_DEGREE2_SH_COEFFICIENT_FLOAT_OFFSETS
    ],
    recordCount,
    byteLength,
    activeShDegree: Number.isFinite(activeShDegree) ? activeShDegree : null,
    activeShDegreeT: Number.isFinite(activeShDegreeT) ? activeShDegreeT : null,
    spatialShEvaluation: 'cuda-aligned-degree-2-original-position-direction',
    temporalShEvaluation: 'not-applied-for-spatial-degree-2',
    storageBinding: PRODUCTION_CANDIDATE_ATTRIBUTE_STORAGE_BINDING,
    storageBindingCount: PRODUCTION_EVALUATOR_STORAGE_BINDING_COUNT,
    evaluatorBindingIndices: [0, 1, 2, 3, 4, 5, 6, 7, 8],
    deviceLimitPreflight: {
      requiredByteLength: byteLength,
      maxStorageBufferBindingSize,
      maxBufferSize,
      requiredStorageBuffersPerShaderStage:
        PRODUCTION_EVALUATOR_STORAGE_BINDING_COUNT,
      maxStorageBuffersPerShaderStage,
      storageBindingSizeReady:
        byteLength != null &&
        maxStorageBufferBindingSize != null &&
        byteLength <= maxStorageBufferBindingSize,
      bufferSizeReady:
        byteLength != null && maxBufferSize != null && byteLength <= maxBufferSize,
      storageBindingCountReady:
        maxStorageBuffersPerShaderStage != null &&
        maxStorageBuffersPerShaderStage >=
          PRODUCTION_EVALUATOR_STORAGE_BINDING_COUNT,
      requiredLimitsRaised: false
    },
    cameraWorldPosition:
      Array.isArray(cameraWorldPosition) ? [...cameraWorldPosition] : null,
    cameraWorldPositionSource: 'projectionParams-row-major-view-minus-r-transpose-t',
    sourcePosition: 'rawXyzOpacity-original-xyz',
    sourceWorksetResourceIdentity:
      typeof sourceWorksetResourceIdentity === 'string'
        ? sourceWorksetResourceIdentity
        : null,
    resourceIdentity:
      ready && typeof resourceIdentity === 'string' ? resourceIdentity : null,
    resourceOwnership:
      typeof resourceOwnership === 'string' ? resourceOwnership : null,
    productionReadbackRequired: false
  };
}

export function buildProductionCandidateAttributeInput({
  raw,
  candidateIndices,
  rawXyzOpacity,
  projectionParams,
  deviceLimits,
  sourceWorksetResourceIdentity = null,
  resourceIdentity = null,
  resourceOwnership = null
} = {}) {
  const reasons = [];
  const recordCount = candidateIndices?.length ?? 0;
  const byteLength = productionCandidateAttributeByteLength(recordCount);
  const sourceRecordCount = Number(raw?.N ?? raw?.count);
  const activeShDegree = Number(raw?.activeShDegree);
  const activeShDegreeT = Number(raw?.activeShDegreeT);
  const fdcDim = Number(raw?.fdcDim);
  const frestDim = Number(raw?.frestDim);
  const scaleXYZDim = Number(raw?.scaleXYZDim);
  const cameraWorldPosition = cameraWorldPositionFromProjectionParams(
    projectionParams
  );

  if (!(candidateIndices instanceof Uint32Array) || recordCount <= 0) {
    reasons.push('candidate-indices-missing-or-invalid');
  }
  if (!Number.isSafeInteger(sourceRecordCount) || sourceRecordCount <= 0) {
    reasons.push('source-record-count-missing-or-invalid');
  }
  if (activeShDegree !== 2) reasons.push('active-spatial-sh-degree-unsupported');
  if (!Number.isInteger(activeShDegreeT) || activeShDegreeT < 0) {
    reasons.push('active-temporal-sh-degree-invalid');
  }
  if (!Number.isInteger(fdcDim) || fdcDim < 3) {
    reasons.push('f-dc-dimension-incomplete');
  }
  if (!Number.isInteger(frestDim) || frestDim < 24) {
    reasons.push('f-rest-degree2-dimension-incomplete');
  }
  if (!Number.isInteger(scaleXYZDim) || scaleXYZDim < 3) {
    reasons.push('scale-xyz-dimension-incomplete');
  }
  if (byteLength == null) reasons.push('candidate-attribute-byte-length-invalid');
  if (!cameraWorldPosition) reasons.push('projection-camera-world-position-invalid');

  const maxStorageBufferBindingSize = finiteLimit(
    deviceLimits?.maxStorageBufferBindingSize
  );
  const maxBufferSize = finiteLimit(deviceLimits?.maxBufferSize);
  const maxStorageBuffersPerShaderStage = finiteLimit(
    deviceLimits?.maxStorageBuffersPerShaderStage
  );
  if (maxStorageBufferBindingSize == null) {
    reasons.push('max-storage-buffer-binding-size-unavailable');
  } else if (byteLength != null && byteLength > maxStorageBufferBindingSize) {
    reasons.push('candidate-attribute-storage-binding-size-exceeded');
  }
  if (maxBufferSize == null) {
    reasons.push('max-buffer-size-unavailable');
  } else if (byteLength != null && byteLength > maxBufferSize) {
    reasons.push('candidate-attribute-buffer-size-exceeded');
  }
  if (maxStorageBuffersPerShaderStage == null) {
    reasons.push('max-storage-buffers-per-shader-stage-unavailable');
  } else if (
    maxStorageBuffersPerShaderStage < PRODUCTION_EVALUATOR_STORAGE_BINDING_COUNT
  ) {
    reasons.push('storage-buffer-binding-limit-insufficient');
  }

  const safeSourceRecordCount = Number.isSafeInteger(sourceRecordCount) &&
    sourceRecordCount > 0
    ? sourceRecordCount
    : 0;
  if (
    !typedArrayHasRequiredLength(
      raw?.f_dc,
      safeSourceRecordCount * Math.max(fdcDim || 0, 0),
      Float32Array
    )
  ) reasons.push('f-dc-array-missing-or-short');
  if (
    !typedArrayHasRequiredLength(
      raw?.f_rest,
      safeSourceRecordCount * Math.max(frestDim || 0, 0),
      Float32Array
    )
  ) reasons.push('f-rest-array-missing-or-short');
  if (
    !typedArrayHasRequiredLength(
      raw?.scale_xyz,
      safeSourceRecordCount * Math.max(scaleXYZDim || 0, 0),
      Float32Array
    )
  ) reasons.push('scale-xyz-array-missing-or-short');
  if (
    !(rawXyzOpacity instanceof Float32Array) ||
    rawXyzOpacity.length < recordCount * 4
  ) reasons.push('raw-xyz-opacity-array-missing-or-short');

  if (reasons.length === 0) {
    for (let row = 0; row < recordCount; row += 1) {
      const srcIndex = candidateIndices[row];
      if (srcIndex >= sourceRecordCount) {
        reasons.push('candidate-src-index-out-of-range');
        break;
      }
      const rawBase = row * 4;
      const originalPosition = [
        rawXyzOpacity[rawBase + 0],
        rawXyzOpacity[rawBase + 1],
        rawXyzOpacity[rawBase + 2]
      ];
      if (!originalPosition.every(Number.isFinite)) {
        reasons.push('candidate-original-position-nonfinite');
        break;
      }
      if (!Number.isFinite(rawXyzOpacity[rawBase + 3])) {
        reasons.push('candidate-opacity-nonfinite');
        break;
      }
      const directionLength = Math.hypot(
        originalPosition[0] - cameraWorldPosition[0],
        originalPosition[1] - cameraWorldPosition[1],
        originalPosition[2] - cameraWorldPosition[2]
      );
      if (!Number.isFinite(directionLength) || directionLength <= 1e-12) {
        reasons.push('candidate-camera-direction-degenerate');
        break;
      }
      const fdcBase = srcIndex * fdcDim;
      const frestBase = srcIndex * frestDim;
      const scaleBase = srcIndex * scaleXYZDim;
      for (let component = 0; component < 3; component += 1) {
        if (!Number.isFinite(raw.f_dc[fdcBase + component])) {
          reasons.push('f-dc-coefficient-nonfinite');
          break;
        }
        if (!Number.isFinite(raw.scale_xyz[scaleBase + component])) {
          reasons.push('scale-xyz-component-nonfinite');
          break;
        }
      }
      if (reasons.length > 0) break;
      for (let component = 0; component < 24; component += 1) {
        if (!Number.isFinite(raw.f_rest[frestBase + component])) {
          reasons.push('f-rest-degree2-coefficient-nonfinite');
          break;
        }
      }
      if (reasons.length > 0) break;
    }
  }

  const blocked = boundedReasons(reasons);
  const ready = blocked.reasonCount === 0;
  const contract = buildLayoutContract({
    ready,
    recordCount,
    byteLength,
    activeShDegree,
    activeShDegreeT,
    deviceLimits,
    blocked,
    sourceWorksetResourceIdentity,
    resourceIdentity,
    resourceOwnership,
    cameraWorldPosition
  });
  if (!ready) return { ready: false, data: null, contract };

  const data = new Float32Array(recordCount * PRODUCTION_CANDIDATE_ATTRIBUTE_FLOAT_STRIDE);
  for (let row = 0; row < recordCount; row += 1) {
    const srcIndex = candidateIndices[row];
    const fdcBase = srcIndex * fdcDim;
    const frestBase = srcIndex * frestDim;
    const scaleBase = srcIndex * scaleXYZDim;
    const outBase = row * PRODUCTION_CANDIDATE_ATTRIBUTE_FLOAT_STRIDE;
    const sx = raw.scale_xyz[scaleBase + 0];
    const sy = raw.scale_xyz[scaleBase + 1];
    const sz = raw.scale_xyz[scaleBase + 2];
    data[outBase + 0] = raw.f_dc[fdcBase + 0];
    data[outBase + 1] = raw.f_dc[fdcBase + 1];
    data[outBase + 2] = raw.f_dc[fdcBase + 2];
    data[outBase + 3] = Math.max(1e-6, (sx + sy + sz) / 3);
    data[outBase + 4] = Math.max(1e-6, sx);
    data[outBase + 5] = Math.max(1e-6, sy);
    data[outBase + 6] = Math.max(1e-6, sz);
    data[outBase + 7] = 111;
    data.set(raw.f_rest.subarray(frestBase, frestBase + 24), outBase + 8);
  }
  return { ready: true, data, contract };
}

export function buildProductionDegree2SpatialShWgsl() {
  const { c0, c1, c2 } = PRODUCTION_DEGREE2_SH_CONSTANTS;
  return `
const PRODUCTION_CANDIDATE_ATTRIBUTE_VEC4_STRIDE: u32 = ${PRODUCTION_CANDIDATE_ATTRIBUTE_VEC4_STRIDE}u;
const SH_C0: f32 = ${c0};
const SH_C1: f32 = ${c1};
const SH_C2_0: f32 = ${c2[0]};
const SH_C2_1: f32 = ${c2[1]};
const SH_C2_2: f32 = ${c2[2]};
const SH_C2_3: f32 = ${c2[3]};
const SH_C2_4: f32 = ${c2[4]};

fn candidateAttributeFloat(row: u32, floatOffset: u32) -> f32 {
  let vec4Offset = floatOffset / 4u;
  let component = floatOffset % 4u;
  return attributeInput[
    row * PRODUCTION_CANDIDATE_ATTRIBUTE_VEC4_STRIDE + vec4Offset
  ][component];
}

fn candidateShCoefficient(row: u32, coefficientIndex: u32) -> vec3f {
  if (coefficientIndex == 0u) {
    return vec3f(
      candidateAttributeFloat(row, 0u),
      candidateAttributeFloat(row, 1u),
      candidateAttributeFloat(row, 2u)
    );
  }
  let floatOffset = 8u + (coefficientIndex - 1u) * 3u;
  return vec3f(
    candidateAttributeFloat(row, floatOffset + 0u),
    candidateAttributeFloat(row, floatOffset + 1u),
    candidateAttributeFloat(row, floatOffset + 2u)
  );
}

fn cameraWorldPositionFromProjectionParams() -> vec3f {
  let view0 = projectionParams[3u];
  let view1 = projectionParams[4u];
  let view2 = projectionParams[5u];
  let translation = vec3f(view0.w, view1.w, view2.w);
  return -(
    view0.xyz * translation.x +
    view1.xyz * translation.y +
    view2.xyz * translation.z
  );
}

fn evaluateProductionDegree2SpatialSh(row: u32, originalPosition: vec3f) -> vec3f {
  let direction = normalize(
    originalPosition - cameraWorldPositionFromProjectionParams()
  );
  let x = direction.x;
  let y = direction.y;
  let z = direction.z;
  let xx = x * x;
  let yy = y * y;
  let zz = z * z;
  var result = SH_C0 * candidateShCoefficient(row, 0u);
  result = result - SH_C1 * y * candidateShCoefficient(row, 1u);
  result = result + SH_C1 * z * candidateShCoefficient(row, 2u);
  result = result - SH_C1 * x * candidateShCoefficient(row, 3u);
  result = result + SH_C2_0 * x * y * candidateShCoefficient(row, 4u);
  result = result + SH_C2_1 * y * z * candidateShCoefficient(row, 5u);
  result = result + SH_C2_2 * (2.0 * zz - xx - yy) * candidateShCoefficient(row, 6u);
  result = result + SH_C2_3 * x * z * candidateShCoefficient(row, 7u);
  result = result + SH_C2_4 * (xx - yy) * candidateShCoefficient(row, 8u);
  return max(result + vec3f(0.5), vec3f(0.0));
}
`;
}
