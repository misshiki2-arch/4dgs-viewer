import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
  buildWebGpuVisibleRecordDiagnosticArtifactBundle
} from '../demo/js/common_4dgs_diagnostic_artifact_contracts.js';
import {
  computeCudaConditionalGaussianState4D
} from '../demo/js/cuda_4d_state.js';

const ERROR_FIELDS = [
  'covarianceBeforeCameraTransformMaxAbs',
  'cameraSpaceCovarianceMaxAbs',
  'jacobianMaxAbs',
  'screenSpaceCovarianceMaxAbs',
  'conicMaxAbs',
  'radiusAbs'
];
const CLASSIFICATION_FIELDS = [
  'conditional4DCovarianceClassification',
  'rotationCovarianceClassification',
  'jacobianProjectionClassification',
  'conicRadiusClassification'
];

const runtimeSource = await readFile(
  new URL('../demo/js/webgpu_visible_record_dry_run_runtime.js', import.meta.url),
  'utf8'
);

function extractNamedFunction(source, name) {
  const start = source.indexOf(`function ${name}`);
  assert.notEqual(start, -1, `${name} must exist`);
  const parametersStart = source.indexOf('(', start);
  let parameterDepth = 0;
  let parametersEnd = -1;
  for (let index = parametersStart; index < source.length; index += 1) {
    if (source[index] === '(') parameterDepth += 1;
    if (source[index] === ')') parameterDepth -= 1;
    if (parameterDepth === 0) {
      parametersEnd = index;
      break;
    }
  }
  const bodyStart = source.indexOf('{', parametersEnd);
  assert.notEqual(bodyStart, -1, `${name} must have a body`);
  let depth = 0;
  for (let index = bodyStart; index < source.length; index += 1) {
    if (source[index] === '{') depth += 1;
    if (source[index] === '}') depth -= 1;
    if (depth === 0) return source.slice(start, index + 1);
  }
  assert.fail(`${name} body must be balanced`);
}

const harnessFunctionNames = [
  'readWebGpuFootprintPayload',
  'finiteNumber',
  'cov3ApplyLocal',
  'cov3BilinearLocal',
  'dot4Local',
  'readProjectionParamRows',
  'resolveCudaConditionalCovarianceInputsLocal',
  'computeStep113CudaAlignedConicReference',
  'maxAbsDeltaArray',
  'readStep113WgslIntermediateReadback',
  'flattenJacobian',
  'getStep113RepresentativeTraits',
  'buildStep113CovarianceJacobianConicEvidence'
];
const harness = Function(
  'computeCudaConditionalGaussianState4D',
  `${harnessFunctionNames
    .map((name) => extractNamedFunction(runtimeSource, name))
    .join('\n\n')}
return {
  buildEvidence: buildStep113CovarianceJacobianConicEvidence,
  computeExpected: computeStep113CudaAlignedConicReference
};`
)(computeCudaConditionalGaussianState4D);

const recordCount = 8;
const candidateIndices = Uint32Array.from(
  { length: recordCount },
  (_, row) => row
);
const raw = {
  scale_xyz: Float32Array.from(
    { length: recordCount * 3 },
    (_, index) => [0.12, 0.24, 0.36][index % 3] + Math.floor(index / 3) * 0.001
  ),
  scaleXYZDim: 3,
  scale_t: Float32Array.from({ length: recordCount }, () => 0.08),
  scaleTDim: 1,
  rotation: Float32Array.from(
    { length: recordCount * 4 },
    (_, index) => [0.92, 0.11, 0.18, 0.31][index % 4]
  ),
  rotationDim: 4,
  rotation_r: Float32Array.from(
    { length: recordCount * 4 },
    (_, index) => [0.97, 0.07, -0.03, 0.12][index % 4]
  ),
  rotationRDim: 4
};
const statePositions = Float32Array.from(
  { length: recordCount * 4 },
  (_, index) => {
    const row = Math.floor(index / 4);
    return [row * 0.4 - 1.4, row * 0.25 - 0.7, 8 + row, 1][index % 4];
  }
);
const projectionParams = new Float32Array(24);
projectionParams[0] = 1;
projectionParams[1] = 1280;
projectionParams[2] = 720;
projectionParams[6] = 1;
projectionParams[8] = 800;
projectionParams[9] = 800;
projectionParams.set([1, 0, 0, 0], 12);
projectionParams.set([0, 1, 0, 0], 16);
projectionParams.set([0, 0, 1, 0], 20);
const buildConfig = { scalingModifier: 1, sigmaScale: 1 };
const footprintPayload = new Float64Array(recordCount * 12);
for (let row = 0; row < recordCount; row += 1) {
  footprintPayload[row * 12 + 8] = 113;
}

const expectedByRow = Array.from({ length: recordCount }, (_, row) =>
  harness.computeExpected({
    raw,
    srcIndex: row,
    statePosition: Array.from(statePositions.slice(row * 4, row * 4 + 3)),
    projectionParams,
    buildConfig
  })
);

function writeValidIntermediate(readback, slot, row, mutate = null) {
  const expected = expectedByRow[row];
  const base = slot * 32;
  const values = [
    row,
    113,
    expected.radius,
    expected.determinant,
    ...expected.covarianceWorldBeforeCameraTransform,
    ...expected.cameraSpaceCovariance,
    ...expected.jacobian.flat(),
    ...expected.screenSpaceCovariance,
    ...expected.conic,
    ...expected.cameraSpacePosition,
    0
  ];
  assert.equal(values.length, 32);
  readback.set(values, base);
  if (typeof mutate === 'function') mutate(readback, base, expected);
}

function buildCase({ count, validSlots = [], readbackSlotCount = count, mutate }) {
  const readback = new Float64Array(readbackSlotCount * 32);
  for (const slot of validSlots) {
    writeValidIntermediate(readback, slot, slot, mutate);
  }
  return harness.buildEvidence({
    raw,
    candidateIndices: candidateIndices.slice(0, count),
    statePositions: statePositions.slice(0, count * 4),
    footprintPayload: footprintPayload.slice(0, count * 12),
    intermediateReadback: readback,
    intermediateReadbackRows: Array.from({ length: count }, (_, row) => row),
    diagnosticBindingEvidence: {},
    projectionParams,
    buildConfig
  });
}

function assertClassifications(evidence, expected) {
  for (const field of CLASSIFICATION_FIELDS) {
    assert.equal(evidence[field], expected[field] ?? expected.default, field);
  }
}

function assertNullErrors(representative) {
  for (const field of ERROR_FIELDS) assert.equal(representative.errors[field], null);
}

// Case A: complete valid evidence within all tolerances.
const complete = buildCase({
  count: recordCount,
  validSlots: Array.from({ length: recordCount }, (_, row) => row)
});
assert.equal(complete.readbackCompletedCount, recordCount);
assert.equal(complete.missingReadbackCount, 0);
assert.equal(complete.invalidReadbackCount, 0);
assert.equal(complete.firstMismatchStage, 'none');
assertClassifications(complete, {
  conditional4DCovarianceClassification:
    'cuda-conditional-4d-to-3d-covariance-matched',
  rotationCovarianceClassification:
    'cuda-conditional-4d-to-3d-covariance-matched',
  jacobianProjectionClassification: 'cuda-aligned-camera-jacobian',
  conicRadiusClassification: 'cuda-aligned-partial'
});
for (const field of ERROR_FIELDS) assert.equal(complete.maxStageErrors[field], 0);

// Case B: Fix1-equivalent one valid row plus five invalid unwritten zero slots.
const partialInvalid = buildCase({ count: 6, validSlots: [0] });
assert.equal(partialInvalid.readbackCompletedCount, 1);
assert.equal(partialInvalid.missingReadbackCount, 0);
assert.equal(partialInvalid.invalidReadbackCount, 5);
assert.equal(
  partialInvalid.firstMismatchStage,
  'wgsl-intermediate-readback-invalid'
);
assertClassifications(partialInvalid, { default: 'missing' });
for (const representative of partialInvalid.representativeGaussians.slice(1)) {
  assertNullErrors(representative);
}
for (const field of ERROR_FIELDS) {
  assert.equal(
    partialInvalid.maxStageErrors[field],
    partialInvalid.representativeGaussians[0].errors[field],
    `${field} must aggregate only the valid subset`
  );
}

// Case C: one valid row and one absent readback slot.
const missing = buildCase({ count: 2, validSlots: [0], readbackSlotCount: 1 });
assert.equal(missing.readbackCompletedCount, 1);
assert.equal(missing.missingReadbackCount, 1);
assert.equal(missing.invalidReadbackCount, 0);
assert.equal(missing.firstMismatchStage, 'wgsl-intermediate-readback-missing');
assertClassifications(missing, { default: 'missing' });
assertNullErrors(missing.representativeGaussians[1]);

// Case D: no valid evidence must publish null max errors, not matching zeroes.
const noValid = buildCase({ count: 2 });
assert.equal(noValid.readbackCompletedCount, 0);
assert.equal(noValid.invalidReadbackCount, 2);
assertClassifications(noValid, { default: 'missing' });
for (const field of ERROR_FIELDS) assert.equal(noValid.maxStageErrors[field], null);
for (const representative of noValid.representativeGaussians) {
  assertNullErrors(representative);
}

// Case E: complete valid evidence with a genuine conditional covariance error.
const semanticMismatch = buildCase({
  count: recordCount,
  validSlots: Array.from({ length: recordCount }, (_, row) => row),
  mutate: (readback, base) => {
    if (base === 0) readback[base + 4] += 0.001;
  }
});
assert.equal(semanticMismatch.readbackCompletedCount, recordCount);
assert.equal(semanticMismatch.missingReadbackCount, 0);
assert.equal(semanticMismatch.invalidReadbackCount, 0);
assert.equal(
  semanticMismatch.firstMismatchStage,
  'conditional-4d-to-3d-covariance'
);
assert.equal(
  semanticMismatch.conditional4DCovarianceClassification,
  'cuda-conditional-4d-to-3d-covariance-mismatch'
);
assert.equal(
  semanticMismatch.rotationCovarianceClassification,
  'cuda-conditional-4d-to-3d-covariance-mismatch'
);
assert.ok(
  semanticMismatch.maxStageErrors.covarianceBeforeCameraTransformMaxAbs > 1e-5
);

// Existing Design C serializer must preserve null errors and fail-closed fields.
const artifactBundle = buildWebGpuVisibleRecordDiagnosticArtifactBundle({
  runtimeResult: {
    schemaVersion: 'phase3-step2-webgpu-visible-record-dry-run-v1',
    status: 'ok',
    reason: 'ok',
    webgpuTileListCompositorContract: {
      step113RepresentativeGaussianComparison: partialInvalid
    }
  },
  detailSelection: null,
  artifactSetIdentity: 'step120-impl2-fix3-smoke'
});
const serializedEvidence = artifactBundle.canonicalDiagnosticResult
  .stageSummaries.tileCompositor.step113SemanticEvidence;
assertClassifications(serializedEvidence, { default: 'missing' });
assert.deepEqual(serializedEvidence.maxStageErrors, partialInvalid.maxStageErrors);
for (const representative of serializedEvidence.representativeGaussians.slice(1)) {
  assertNullErrors(representative);
}

assert.match(
  runtimeSource,
  /const requiredEvidenceComplete =[\s\S]*?readbackCompletedCount === representatives\.length[\s\S]*?missingReadbackCount === 0[\s\S]*?invalidReadbackCount === 0;/
);
assert.doesNotMatch(
  runtimeSource,
  /jacobianProjectionClassification:\s*readbackCompletedCount > 0/
);

console.log('Step120 Impl2 Fix3 classification fail-closed smoke: OK', {
  completeCount: complete.readbackCompletedCount,
  partialInvalid: {
    completed: partialInvalid.readbackCompletedCount,
    invalid: partialInvalid.invalidReadbackCount,
    firstMismatchStage: partialInvalid.firstMismatchStage
  },
  missing: {
    completed: missing.readbackCompletedCount,
    missing: missing.missingReadbackCount,
    firstMismatchStage: missing.firstMismatchStage
  },
  semanticMismatch: semanticMismatch.firstMismatchStage
});
