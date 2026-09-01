import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  POPULATION_SEMANTIC_EVIDENCE_FLOAT_STRIDE,
  POPULATION_SEMANTIC_EVIDENCE_BYTES_PER_RECORD,
  POPULATION_SEMANTIC_EVIDENCE_VEC4_STRIDE,
  POPULATION_RASTER_SEMANTIC_COMPANION_BYTES_PER_RECORD,
  POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE,
  POPULATION_RASTER_SEMANTIC_COMPANION_LAYOUT_SCHEMA_VERSION,
  POPULATION_RASTER_SEMANTIC_COMPANION_VEC4_STRIDE,
  POPULATION_RASTER_SEMANTIC_ACTUAL_PROVENANCE,
  POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE,
  POPULATION_SEMANTIC_LOGICAL_COMBINED_BYTES_PER_RECORD,
  POPULATION_SEMANTIC_LOGICAL_COMBINED_FLOAT_STRIDE,
  POPULATION_SEMANTIC_LOGICAL_COMBINED_VEC4_STRIDE,
  POPULATION_SEMANTIC_COMPARISON_SCHEMA_VERSION,
  POPULATION_SEMANTIC_COMPLETE_STAGE_CLASSIFICATIONS,
  POPULATION_PIXEL_BOUNDARY_PRECISION_CLASSIFICATION_PROVENANCE,
  POPULATION_SEMANTIC_MAX_FIRST_MISMATCHES,
  POPULATION_SEMANTIC_STAGE_LOCAL_CONTEXT_STAGE_KEYS,
  POPULATION_SEMANTIC_STAGE_LOCAL_MAX_REPRESENTATIVE_RECORDS,
  POPULATION_SEMANTIC_STAGE_LOCAL_REPRESENTATIVE_LIMIT,
  POPULATION_SEMANTIC_STAGE_CONTRACTS,
  POPULATION_TILE_INPUT_ALPHA_STAGE_LOCAL_REPRESENTATIVE_LIMIT,
  POPULATION_TILE_INPUT_RGB_STAGE_LOCAL_REPRESENTATIVE_LIMIT,
  PRODUCTION_RESIDENT_RANGE_START,
  buildPopulationRasterSemanticCompanionLayoutContract,
  buildPopulationSemanticComparisonInputContract,
  buildPopulationSemanticDiagnosticWorksetResourceIdentity,
  classifyPopulationSemanticStageEvidence,
  validatePopulationSemanticStageLocalMismatchSummaries
} from '../demo/js/common_4dgs_population_semantic_comparison_contracts.js';
import {
  PRODUCTION_TILE_INPUT_ALPHA_F32_CENTRAL_ORACLE_VERSION,
  buildProductionTileInputAlphaF32Central
} from '../demo/js/common_4dgs_tile_input_alpha_f32_semantic.js';
import {
  buildCudaNormalizedInclusiveTileRect,
  buildCudaDegree2TileInputExpectedRgb,
  buildCudaTileInputExpectedAlpha,
  buildExpectedWebGpuInclusivePixelBounds,
  buildExplicitPopulationChunkIndices,
  buildPopulationAlignedSemanticExpectedRecord,
  comparePopulationAlignedSemanticChunkEvidence
} from '../demo/js/webgpu_population_aligned_semantic_comparison.js';
import {
  buildWebGpu4DStatePositionsForCandidates
} from '../demo/js/webgpu_4d_state_evaluator.js';
import {
  observePopulationRasterSemanticCompanion
} from '../demo/js/webgpu_population_raster_semantic_observer.js';

const validIdentity = {
  sceneInputIdentity: 'scene-fixture',
  spl4InputIdentity: 'spl4-fixture',
  populationContractIdentity: 'production-resident-workset-fixture',
  buildConfig: { timestamp: 23.2, scalingModifier: 1, sigmaScale: 1 },
  cameraIdentity: 'camera-fixture',
  projectionIdentity: 'projection-fixture',
  timeIdentity: 'time-23.2'
};
const representativeActivatedOpacity = 0.1694848388;
const representativeTemporalOpacity = 0.01641350612;

function inputContract(rangeCount, rangeStart = PRODUCTION_RESIDENT_RANGE_START) {
  return buildPopulationSemanticComparisonInputContract({
    rangeStart,
    rangeCount,
    ...validIdentity
  });
}

function expectedRecord(localRow, {
  eligible = true,
  rasterEligible = eligible
} = {}) {
  const base = localRow + 1;
  const values = {
    temporalEligibility: [eligible ? 1 : 0],
    conditionalStatePosition: [base, base + 1, base + 2],
    conditionalWorldCovariance: Array.from({ length: 6 }, (_, index) => base + index),
    cameraSpaceCovariance: Array.from({ length: 6 }, (_, index) => base + 10 + index),
    projectionJacobian: Array.from({ length: 6 }, (_, index) => base + 20 + index),
    screenCovariance: [base + 30, base + 31, base + 32],
    conic: [base + 40, base + 41, base + 42],
    radius: [base + 50],
    productionRasterEligibility: [rasterEligible ? 1 : 0],
    projectedCenter: [base + 60, base + 61],
    cameraDepth: [base + 62],
    webgpuInclusivePixelBounds: [1, 2, 10, 11],
    normalizedInclusiveTileBounds: [0, 0, 1, 1],
    productionTileInputAlpha: [0.5],
    productionTileInputRgb: [0.25, 0.5, 0.75]
  };
  return {
    valid: true,
    temporalEligible: eligible,
    rasterEligible,
    stages: Object.fromEntries(
      POPULATION_SEMANTIC_STAGE_CONTRACTS.map(({ key }) => [
        key,
        {
          valid:
            key === 'temporalEligibility' ||
            key === 'productionRasterEligibility' ||
            (
              [
                'projectedCenter',
                'cameraDepth',
                'webgpuInclusivePixelBounds',
                'normalizedInclusiveTileBounds',
                'productionTileInputAlpha',
                'productionTileInputRgb'
              ].includes(key)
                ? rasterEligible
                : eligible
            ),
          values: values[key]
        }
      ])
    )
  };
}

function packActual(records) {
  const packed = new Float32Array(
    records.length * POPULATION_SEMANTIC_EVIDENCE_FLOAT_STRIDE
  );
  for (let row = 0; row < records.length; row += 1) {
    const record = records[row];
    const base = row * POPULATION_SEMANTIC_EVIDENCE_FLOAT_STRIDE;
    const stage = record.stages;
    packed.set([
      row,
      record.temporalEligible ? 1 : 0,
      ...(record.temporalEligible ? stage.conditionalStatePosition.values : [0, 0, 0]),
      ...(record.temporalEligible ? stage.conditionalWorldCovariance.values : [0, 0, 0, 0, 0, 0]),
      ...(record.temporalEligible ? stage.cameraSpaceCovariance.values : [0, 0, 0, 0, 0, 0]),
      ...(record.temporalEligible ? stage.projectionJacobian.values : [0, 0, 0, 0, 0, 0]),
      ...(record.temporalEligible ? stage.screenCovariance.values : [0, 0, 0]),
      ...(record.temporalEligible ? stage.conic.values : [0, 0, 0]),
      record.temporalEligible ? stage.radius.values[0] : 0,
      record.temporalEligible ? 113 : 0,
      record.temporalEligible ? 0 : 0
    ], base);
  }
  return packed;
}

function packRasterCompanion(records) {
  const packed = new Float32Array(
    records.length * POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE
  );
  for (let row = 0; row < records.length; row += 1) {
    const record = records[row];
    const base = row * POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE;
    const stage = record.stages;
    packed.set(record.rasterEligible
      ? [
          1,
          ...stage.projectedCenter.values,
          stage.cameraDepth.values[0],
          ...stage.webgpuInclusivePixelBounds.values,
          ...stage.normalizedInclusiveTileBounds.values,
          ...stage.productionTileInputRgb.values,
          stage.productionTileInputAlpha.values[0]
        ]
      : new Array(POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE).fill(0), base);
  }
  return packed;
}

function rasterCompanionContract(contract, records, overrides = {}) {
  const evidence = packRasterCompanion(records);
  const sourceWorksetResourceIdentity =
    buildPopulationSemanticDiagnosticWorksetResourceIdentity(contract);
  return buildPopulationRasterSemanticCompanionLayoutContract({
    recordCount: records.length,
    evidenceFloatCount: evidence.length,
    sourceWorksetResourceIdentity,
    sourceStateResourceIdentity: 'state-resource-fixture',
    sourceTileInputResourceIdentity: 'tile-input-resource-fixture',
    canvasWidth: 1280,
    canvasHeight: 720,
    tileSize: 16,
    tileCols: 80,
    tileRows: 45,
    observerDispatchSubmitted: true,
    observerReadbackCompleted: true,
    observerOwnedBuffersDestroyed: true,
    ...overrides
  });
}

function compareFixture({ count, records = null, candidates = null } = {}) {
  const contract = inputContract(count);
  const candidateIndices = candidates ?? buildExplicitPopulationChunkIndices(contract);
  const expectedRecords = records ?? Array.from({ length: count }, (_, row) => expectedRecord(row));
  const rasterEvidence = packRasterCompanion(expectedRecords);
  return comparePopulationAlignedSemanticChunkEvidence({
    inputContract: contract,
    candidateIndices,
    actualPackedEvidence: packActual(expectedRecords),
    actualRasterCompanionEvidence: rasterEvidence,
    actualRasterCompanionContract:
      rasterCompanionContract(contract, expectedRecords),
    expectedRecordForRow: (row) => expectedRecords[row]
  });
}

function comparePacked(
  actualPackedEvidence,
  expectedRecords,
  actualRasterCompanionEvidence = packRasterCompanion(expectedRecords),
  actualRasterCompanionContract = null
) {
  const count = expectedRecords.length;
  const contract = inputContract(count);
  return comparePopulationAlignedSemanticChunkEvidence({
    inputContract: contract,
    candidateIndices: buildExplicitPopulationChunkIndices(contract),
    actualPackedEvidence,
    actualRasterCompanionEvidence,
    actualRasterCompanionContract:
      actualRasterCompanionContract ?? rasterCompanionContract(contract, expectedRecords),
    expectedRecordForRow: (row) => expectedRecords[row]
  });
}

function comparePixelBoundaryFixture({
  expectedCenter,
  actualCenter,
  expectedRadius,
  actualRadius = expectedRadius,
  expectedBounds = null,
  actualBounds = null,
  expectedRasterEligible = true,
  actualRasterEligible = true,
  mutateActualPacked = null,
  rasterContractOverrides = {},
  srcIndex = PRODUCTION_RESIDENT_RANGE_START
}) {
  const expected = expectedRecord(0, {
    eligible: true,
    rasterEligible: expectedRasterEligible
  });
  expected.stages.projectedCenter.values = [...expectedCenter];
  expected.stages.radius.values = [expectedRadius];
  expected.stages.productionRasterEligibility.values = [
    expectedRasterEligible ? 1 : 0
  ];
  expected.stages.webgpuInclusivePixelBounds.values = expectedBounds ??
    buildExpectedWebGpuInclusivePixelBounds({
      centerX: expectedCenter[0],
      centerY: expectedCenter[1],
      radius: expectedRadius,
      canvasWidth: 1280,
      canvasHeight: 720
    });
  const actual = JSON.parse(JSON.stringify(expected));
  actual.rasterEligible = actualRasterEligible;
  actual.stages.productionRasterEligibility.values = [
    actualRasterEligible ? 1 : 0
  ];
  actual.stages.projectedCenter.values = [...actualCenter];
  actual.stages.radius.values = [actualRadius];
  actual.stages.webgpuInclusivePixelBounds.values = actualBounds ??
    buildExpectedWebGpuInclusivePixelBounds({
      centerX: actualCenter[0],
      centerY: actualCenter[1],
      radius: actualRadius,
      canvasWidth: 1280,
      canvasHeight: 720
    });
  const contract = inputContract(1, srcIndex);
  const actualPacked = packActual([actual]);
  if (typeof mutateActualPacked === 'function') mutateActualPacked(actualPacked);
  return comparePopulationAlignedSemanticChunkEvidence({
    inputContract: contract,
    candidateIndices: buildExplicitPopulationChunkIndices(contract),
    actualPackedEvidence: actualPacked,
    actualRasterCompanionEvidence: packRasterCompanion([actual]),
    actualRasterCompanionContract: rasterCompanionContract(
      contract,
      [actual],
      rasterContractOverrides
    ),
    expectedRecordForRow: () => expected
  });
}

for (const count of [1, 65536]) {
  const contract = inputContract(count);
  assert.equal(contract.status, 'ready');
  const indices = buildExplicitPopulationChunkIndices(contract);
  assert.equal(indices.length, count);
  assert.equal(indices[0], PRODUCTION_RESIDENT_RANGE_START);
  assert.equal(indices.at(-1), PRODUCTION_RESIDENT_RANGE_START + count - 1);
}
for (const [start, count] of [
  [PRODUCTION_RESIDENT_RANGE_START, 0],
  [PRODUCTION_RESIDENT_RANGE_START, 65537],
  [PRODUCTION_RESIDENT_RANGE_START - 1, 1],
  [1048576, 1]
]) {
  assert.equal(inputContract(count, start).status, 'blocked');
}

const mapping = buildExplicitPopulationChunkIndices(inputContract(4));
assert.deepEqual(
  Array.from(mapping),
  [524288, 524289, 524290, 524291]
);

assert.equal(
  POPULATION_SEMANTIC_COMPARISON_SCHEMA_VERSION,
  'phase3-population-aligned-semantic-comparison-v6'
);
assert.ok(
  POPULATION_SEMANTIC_COMPLETE_STAGE_CLASSIFICATIONS.includes(
    'precision-aligned'
  )
);
assert.equal(POPULATION_SEMANTIC_STAGE_LOCAL_REPRESENTATIVE_LIMIT, 4);
assert.equal(POPULATION_TILE_INPUT_ALPHA_STAGE_LOCAL_REPRESENTATIVE_LIMIT, 8);
assert.equal(POPULATION_TILE_INPUT_RGB_STAGE_LOCAL_REPRESENTATIVE_LIMIT, 1);
assert.equal(POPULATION_SEMANTIC_STAGE_LOCAL_MAX_REPRESENTATIVE_RECORDS, 61);
assert.equal(POPULATION_SEMANTIC_EVIDENCE_VEC4_STRIDE, 8);
assert.equal(POPULATION_SEMANTIC_EVIDENCE_FLOAT_STRIDE, 32);
assert.equal(POPULATION_SEMANTIC_EVIDENCE_BYTES_PER_RECORD, 128);
assert.equal(POPULATION_RASTER_SEMANTIC_COMPANION_VEC4_STRIDE, 4);
assert.equal(POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE, 16);
assert.equal(POPULATION_RASTER_SEMANTIC_COMPANION_BYTES_PER_RECORD, 64);
assert.equal(POPULATION_SEMANTIC_LOGICAL_COMBINED_VEC4_STRIDE, 12);
assert.equal(POPULATION_SEMANTIC_LOGICAL_COMBINED_FLOAT_STRIDE, 48);
assert.equal(POPULATION_SEMANTIC_LOGICAL_COMBINED_BYTES_PER_RECORD, 192);
assert.deepEqual(
  POPULATION_SEMANTIC_STAGE_CONTRACTS.slice(-7).map(
    ({ key, components, tolerance }) => ({
      key,
      components: [...components],
      tolerance
    })
  ),
  [
    {
      key: 'productionRasterEligibility',
      components: ['eligible'],
      tolerance: 0
    },
    {
      key: 'projectedCenter',
      components: ['px', 'py'],
      tolerance: 1e-3
    },
    {
      key: 'cameraDepth',
      components: ['depth'],
      tolerance: 1e-4
    },
    {
      key: 'webgpuInclusivePixelBounds',
      components: ['minX', 'minY', 'maxX', 'maxY'],
      tolerance: 0
    },
    {
      key: 'normalizedInclusiveTileBounds',
      components: ['minX', 'minY', 'maxX', 'maxY'],
      tolerance: 0
    },
    {
      key: 'productionTileInputAlpha',
      components: ['alpha'],
      tolerance: 1e-5
    },
    {
      key: 'productionTileInputRgb',
      components: ['r', 'g', 'b'],
      tolerance: 1e-5
    }
  ]
);
const companionLayoutFixture = rasterCompanionContract(
  inputContract(1),
  [expectedRecord(0)]
);
assert.equal(
  companionLayoutFixture.schemaVersion,
  POPULATION_RASTER_SEMANTIC_COMPANION_LAYOUT_SCHEMA_VERSION
);
assert.equal(companionLayoutFixture.status, 'ready');
assert.equal(companionLayoutFixture.rowStrideFloats, 16);
assert.equal(companionLayoutFixture.logicalCombinedRowStrideFloats, 48);

const match = compareFixture({ count: 4 });
assert.equal(match.decision, 'match');
assert.equal(match.coverage.coverageComplete, true);
assert.equal(match.coverage.uniqueSrcIndexCount, 4);
assert.equal(match.coverage.completedChunkCount, 1);
assert.ok(match.stageSummaries.every((stage) => stage.classification === 'match'));
assert.deepEqual(
  match.precisionClassificationProvenance,
  POPULATION_PIXEL_BOUNDARY_PRECISION_CLASSIFICATION_PROVENANCE
);
assert.equal(match.firstSemanticMismatchStage, null);
assert.equal(match.firstDownstreamMismatchStage, null);

const alphaMismatchRecord = expectedRecord(0);
alphaMismatchRecord.stages.productionTileInputAlpha.values = [
  representativeTemporalOpacity
];
const alphaMismatchEvidence = packRasterCompanion([alphaMismatchRecord]);
alphaMismatchEvidence[15] = 0.05;
const alphaMismatch = comparePacked(
  packActual([alphaMismatchRecord]),
  [alphaMismatchRecord],
  alphaMismatchEvidence
);
assert.equal(alphaMismatch.decision, 'mismatch');
assert.equal(
  alphaMismatch.firstDownstreamMismatchStage,
  'productionTileInputAlpha'
);
assert.equal(
  alphaMismatch.stageSummaries.find(
    (stage) => stage.stage === 'productionTileInputAlpha'
  ).mismatchCount,
  1
);
assert.equal(
  alphaMismatch.stageSummaries.find(
    (stage) => stage.stage === 'productionTileInputRgb'
  ).classification,
  'match'
);

const upperAlphaMismatchRecord = expectedRecord(0);
upperAlphaMismatchRecord.stages.productionTileInputAlpha.values = [
  buildCudaTileInputExpectedAlpha({ rawOpacityLogit: 10, temporalWeight: 1 })
];
const upperAlphaMismatchEvidence = packRasterCompanion([
  upperAlphaMismatchRecord
]);
upperAlphaMismatchEvidence[15] = 0.99;
const upperAlphaMismatch = comparePacked(
  packActual([upperAlphaMismatchRecord]),
  [upperAlphaMismatchRecord],
  upperAlphaMismatchEvidence
);
assert.equal(upperAlphaMismatch.decision, 'mismatch');
assert.equal(
  upperAlphaMismatch.firstDownstreamMismatchStage,
  'productionTileInputAlpha'
);

const rgbMismatchRecord = expectedRecord(0);
const rgbMismatchEvidence = packRasterCompanion([rgbMismatchRecord]);
rgbMismatchEvidence[12] += 0.25;
const rgbMismatch = comparePacked(
  packActual([rgbMismatchRecord]),
  [rgbMismatchRecord],
  rgbMismatchEvidence
);
assert.equal(rgbMismatch.decision, 'mismatch');
assert.equal(
  rgbMismatch.firstDownstreamMismatchStage,
  'productionTileInputRgb'
);

const upperRgbMatchRecord = expectedRecord(0);
upperRgbMatchRecord.stages.productionTileInputRgb.values = [1.25, 0.5, 0.75];
const upperRgbMatch = comparePacked(
  packActual([upperRgbMatchRecord]),
  [upperRgbMatchRecord]
);
const upperRgbMatchRgb = upperRgbMatch.stageSummaries.find(
  (stage) => stage.stage === 'productionTileInputRgb'
);
const upperRgbMatchAlpha = upperRgbMatch.stageSummaries.find(
  (stage) => stage.stage === 'productionTileInputAlpha'
);
assert.equal(upperRgbMatch.decision, 'match');
assert.equal(upperRgbMatchRgb.validCount, 1);
assert.equal(upperRgbMatchRgb.comparedCount, 1);
assert.equal(upperRgbMatchRgb.invalidCount, 0);
assert.equal(upperRgbMatchRgb.classification, 'match');
assert.equal(upperRgbMatchAlpha.validCount, 1);
assert.equal(upperRgbMatchAlpha.comparedCount, 1);
assert.equal(upperRgbMatchAlpha.invalidCount, 0);
assert.equal(upperRgbMatchAlpha.classification, 'match');

const upperRgbMismatchRecord = expectedRecord(0);
upperRgbMismatchRecord.stages.productionTileInputRgb.values = [1.25, 0.5, 0.75];
const upperRgbMismatchEvidence = packRasterCompanion([upperRgbMismatchRecord]);
upperRgbMismatchEvidence[12] = 1;
const upperRgbMismatch = comparePacked(
  packActual([upperRgbMismatchRecord]),
  [upperRgbMismatchRecord],
  upperRgbMismatchEvidence
);
assert.equal(upperRgbMismatch.decision, 'mismatch');
assert.equal(
  upperRgbMismatch.firstDownstreamMismatchStage,
  'productionTileInputRgb'
);
const upperRgbMismatchRgb = upperRgbMismatch.stageSummaries.find(
  (stage) => stage.stage === 'productionTileInputRgb'
);
const upperRgbMismatchAlpha = upperRgbMismatch.stageSummaries.find(
  (stage) => stage.stage === 'productionTileInputAlpha'
);
assert.equal(upperRgbMismatchRgb.mismatchCount, 1);
assert.equal(upperRgbMismatchRgb.invalidCount, 0);
assert.equal(upperRgbMismatchRgb.classification, 'mismatch');
assert.equal(upperRgbMismatchAlpha.invalidCount, 0);
assert.equal(upperRgbMismatchAlpha.classification, 'match');

const negativeRgbEvidence = packRasterCompanion([expectedRecord(0)]);
negativeRgbEvidence[12] = -0.25;
const negativeRgb = comparePacked(
  packActual([expectedRecord(0)]),
  [expectedRecord(0)],
  negativeRgbEvidence
);
const negativeRgbStage = negativeRgb.stageSummaries.find(
  (stage) => stage.stage === 'productionTileInputRgb'
);
const negativeRgbAlphaStage = negativeRgb.stageSummaries.find(
  (stage) => stage.stage === 'productionTileInputAlpha'
);
assert.equal(negativeRgb.decision, 'blocked');
assert.equal(negativeRgbStage.validCount, 0);
assert.equal(negativeRgbStage.comparedCount, 0);
assert.equal(negativeRgbStage.invalidCount, 1);
assert.equal(negativeRgbStage.mismatchCount, 0);
assert.equal(negativeRgbStage.maxAbsoluteError, null);
assert.equal(negativeRgbStage.classification, 'blocked-incomplete-evidence');
assert.equal(negativeRgbAlphaStage.validCount, 1);
assert.equal(negativeRgbAlphaStage.comparedCount, 1);
assert.equal(negativeRgbAlphaStage.invalidCount, 0);
assert.equal(negativeRgbAlphaStage.classification, 'match');

const invalidAlphaEvidence = packRasterCompanion([expectedRecord(0)]);
invalidAlphaEvidence[15] = 1.25;
const invalidAlpha = comparePacked(
  packActual([expectedRecord(0)]),
  [expectedRecord(0)],
  invalidAlphaEvidence
);
const invalidAlphaStage = invalidAlpha.stageSummaries.find(
  (stage) => stage.stage === 'productionTileInputAlpha'
);
const invalidAlphaRgbStage = invalidAlpha.stageSummaries.find(
  (stage) => stage.stage === 'productionTileInputRgb'
);
assert.equal(invalidAlpha.decision, 'blocked');
assert.equal(invalidAlphaStage.validCount, 0);
assert.equal(invalidAlphaStage.comparedCount, 0);
assert.equal(invalidAlphaStage.invalidCount, 1);
assert.equal(invalidAlphaStage.mismatchCount, 0);
assert.equal(invalidAlphaStage.maxAbsoluteError, null);
assert.equal(invalidAlphaStage.classification, 'blocked-incomplete-evidence');
assert.equal(invalidAlphaRgbStage.validCount, 1);
assert.equal(invalidAlphaRgbStage.comparedCount, 1);
assert.equal(invalidAlphaRgbStage.invalidCount, 0);
assert.equal(invalidAlphaRgbStage.classification, 'match');

const alphaAndRgbMismatchEvidence = packRasterCompanion([alphaMismatchRecord]);
alphaAndRgbMismatchEvidence[12] += 0.25;
alphaAndRgbMismatchEvidence[15] = 0.05;
const alphaAndRgbMismatch = comparePacked(
  packActual([alphaMismatchRecord]),
  [alphaMismatchRecord],
  alphaAndRgbMismatchEvidence
);
assert.equal(
  alphaAndRgbMismatch.firstDownstreamMismatchStage,
  'productionTileInputAlpha'
);
assert.equal(
  alphaAndRgbMismatch.stageLocalMismatchSummaries.find(
    (stage) => stage.stage === 'productionTileInputAlpha'
  ).serializedRepresentativeRecordCount,
  1
);

const fiveAlphaMismatchRecords = Array.from(
  { length: 5 },
  (_, row) => expectedRecord(row)
);
const fiveAlphaMismatchEvidence = packRasterCompanion(
  fiveAlphaMismatchRecords
);
for (let row = 0; row < fiveAlphaMismatchRecords.length; row += 1) {
  const base = row * POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE;
  fiveAlphaMismatchEvidence[base + 15] += 0.01;
}
const fiveAlphaMismatch = comparePacked(
  packActual(fiveAlphaMismatchRecords),
  fiveAlphaMismatchRecords,
  fiveAlphaMismatchEvidence
);
const fiveAlphaStageLocal = fiveAlphaMismatch.stageLocalMismatchSummaries.find(
  (stage) => stage.stage === 'productionTileInputAlpha'
);
assert.equal(fiveAlphaMismatch.decision, 'mismatch');
assert.equal(fiveAlphaStageLocal.sourceMismatchRecordCount, 5);
assert.equal(fiveAlphaStageLocal.sourceComponentMismatchCount, 5);
assert.equal(fiveAlphaStageLocal.serializedRepresentativeRecordCount, 5);
assert.equal(
  fiveAlphaStageLocal.representativeRecordLimit,
  POPULATION_TILE_INPUT_ALPHA_STAGE_LOCAL_REPRESENTATIVE_LIMIT
);
assert.equal(fiveAlphaStageLocal.truncated, false);
assert.deepEqual(
  fiveAlphaStageLocal.representatives.map(({ localRow, srcIndex }) => [
    localRow,
    srcIndex
  ]),
  Array.from({ length: 5 }, (_, row) => [
    row,
    PRODUCTION_RESIDENT_RANGE_START + row
  ])
);
assert.ok(
  fiveAlphaStageLocal.representatives.every(
    (representative) =>
      representative.expectedStageValues.length === 1 &&
      representative.actualStageValues.length === 1 &&
      representative.mismatchComponents.length === 1 &&
      Number.isFinite(representative.mismatchComponents[0].expected) &&
      Number.isFinite(representative.mismatchComponents[0].actual) &&
      Number.isFinite(representative.mismatchComponents[0].absoluteError) &&
      representative.expectedStageProvenance ===
        POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE.productionTileInputAlpha &&
      representative.actualStageProvenance ===
        POPULATION_RASTER_SEMANTIC_ACTUAL_PROVENANCE
  )
);

const overLimitAlphaRecords = Array.from(
  { length: 20 },
  (_, row) => expectedRecord(row)
);
const overLimitAlphaEvidence = packRasterCompanion(overLimitAlphaRecords);
for (let row = 0; row < overLimitAlphaRecords.length; row += 1) {
  const base = row * POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE;
  overLimitAlphaEvidence[base + 15] += 0.01;
}
const overLimitAlpha = comparePacked(
  packActual(overLimitAlphaRecords),
  overLimitAlphaRecords,
  overLimitAlphaEvidence
);
const overLimitAlphaStageLocal =
  overLimitAlpha.stageLocalMismatchSummaries.find(
    (stage) => stage.stage === 'productionTileInputAlpha'
  );
assert.equal(overLimitAlphaStageLocal.sourceMismatchRecordCount, 20);
assert.equal(
  overLimitAlphaStageLocal.serializedRepresentativeRecordCount,
  POPULATION_TILE_INPUT_ALPHA_STAGE_LOCAL_REPRESENTATIVE_LIMIT
);
assert.equal(overLimitAlphaStageLocal.truncated, true);
assert.equal(
  overLimitAlpha.firstMismatches.length,
  POPULATION_SEMANTIC_MAX_FIRST_MISMATCHES
);

const rgbBoundRecords = Array.from(
  { length: 3 },
  (_, row) => expectedRecord(row)
);
const rgbBoundEvidence = packRasterCompanion(rgbBoundRecords);
for (let row = 0; row < rgbBoundRecords.length; row += 1) {
  const base = row * POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE;
  rgbBoundEvidence[base + 12] += 0.01;
}
const rgbBound = comparePacked(
  packActual(rgbBoundRecords),
  rgbBoundRecords,
  rgbBoundEvidence
);
const rgbBoundStageLocal = rgbBound.stageLocalMismatchSummaries.find(
  (stage) => stage.stage === 'productionTileInputRgb'
);
assert.equal(rgbBoundStageLocal.sourceMismatchRecordCount, 3);
assert.equal(
  rgbBoundStageLocal.serializedRepresentativeRecordCount,
  POPULATION_TILE_INPUT_RGB_STAGE_LOCAL_REPRESENTATIVE_LIMIT
);
assert.equal(rgbBoundStageLocal.representativeRecordLimit, 1);
assert.equal(rgbBoundStageLocal.truncated, true);

const alphaProvenanceDrift = JSON.parse(JSON.stringify(fiveAlphaMismatch));
alphaProvenanceDrift.stageLocalMismatchSummaries.find(
  (stage) => stage.stage === 'productionTileInputAlpha'
).representatives[0].expectedStageProvenance = 'drift';
assert.ok(
  validatePopulationSemanticStageLocalMismatchSummaries({
    stageLocalMismatchSummaries:
      alphaProvenanceDrift.stageLocalMismatchSummaries,
    stageSummaries: alphaProvenanceDrift.stageSummaries,
    scope: 'single-chunk',
    rangeStart: PRODUCTION_RESIDENT_RANGE_START,
    rangeCount: fiveAlphaMismatchRecords.length,
    chunkIndex: 0
  }).includes('stage-local-productionTileInputAlpha-provenance-drift')
);

const nonfiniteColorEvidence = packRasterCompanion([expectedRecord(0)]);
nonfiniteColorEvidence[12] = Number.NaN;
const nonfiniteColor = comparePacked(
  packActual([expectedRecord(0)]),
  [expectedRecord(0)],
  nonfiniteColorEvidence
);
assert.equal(nonfiniteColor.decision, 'blocked');
const nonfiniteColorRgb = nonfiniteColor.stageSummaries.find(
  (stage) => stage.stage === 'productionTileInputRgb'
);
const nonfiniteColorAlpha = nonfiniteColor.stageSummaries.find(
  (stage) => stage.stage === 'productionTileInputAlpha'
);
assert.equal(nonfiniteColorRgb.validCount, 0);
assert.equal(nonfiniteColorRgb.comparedCount, 0);
assert.equal(nonfiniteColorRgb.invalidCount, 1);
assert.equal(nonfiniteColorRgb.mismatchCount, 0);
assert.equal(nonfiniteColorRgb.maxAbsoluteError, null);
assert.equal(nonfiniteColorRgb.classification, 'blocked-incomplete-evidence');
assert.equal(nonfiniteColorAlpha.validCount, 1);
assert.equal(nonfiniteColorAlpha.comparedCount, 1);
assert.equal(nonfiniteColorAlpha.invalidCount, 0);
assert.equal(nonfiniteColorAlpha.classification, 'match');
assert.equal(
  nonfiniteColor.stageSummaries.find(
    (stage) => stage.stage === 'productionRasterEligibility'
  ).classification,
  'match'
);
const oldSchemaContract = inputContract(1);
oldSchemaContract.schemaVersion = 'phase3-population-aligned-semantic-comparison-v4';
const oldSchemaRecord = expectedRecord(0);
const oldSchemaResult = comparePopulationAlignedSemanticChunkEvidence({
  inputContract: oldSchemaContract,
  candidateIndices: buildExplicitPopulationChunkIndices(oldSchemaContract),
  actualPackedEvidence: packActual([oldSchemaRecord]),
  actualRasterCompanionEvidence: packRasterCompanion([oldSchemaRecord]),
  actualRasterCompanionContract: rasterCompanionContract(
    oldSchemaContract,
    [oldSchemaRecord]
  ),
  expectedRecordForRow: () => oldSchemaRecord
});
assert.equal(oldSchemaResult.decision, 'blocked');
assert.ok(oldSchemaResult.blockedReasons.includes('input-contract-schema-drift'));

const precisionBoundaryCases = [
  {
    srcIndex: 803621,
    expectedCenter: [852.4109572394634, 275.00000429863246],
    actualCenter: [852.410888671875, 275],
    radius: 15,
    expectedBounds: [837, 260, 868, 291],
    actualBounds: [837, 260, 868, 290]
  },
  {
    srcIndex: 817028,
    expectedCenter: [344.9139184848317, 514.9999438063753],
    actualCenter: [344.9139709472656, 515],
    radius: 16,
    expectedBounds: [328, 498, 361, 531],
    actualBounds: [328, 499, 361, 531]
  },
  {
    srcIndex: 820164,
    expectedCenter: [803.9999882569947, 223.10742235610658],
    actualCenter: [804, 223.107421875],
    radius: 12,
    expectedBounds: [791, 211, 816, 236],
    actualBounds: [792, 211, 816, 236]
  },
  {
    srcIndex: 833130,
    expectedCenter: [619.2979724231383, 381.99999604727986],
    actualCenter: [619.2979736328125, 382],
    radius: 8,
    expectedBounds: [611, 373, 628, 390],
    actualBounds: [611, 374, 628, 390]
  }
];
for (const fixture of precisionBoundaryCases) {
  const result = comparePixelBoundaryFixture({
    expectedCenter: fixture.expectedCenter,
    actualCenter: fixture.actualCenter,
    expectedRadius: fixture.radius,
    expectedBounds: fixture.expectedBounds,
    actualBounds: fixture.actualBounds,
    srcIndex: fixture.srcIndex
  });
  const pixel = result.stageSummaries.find(
    (stage) => stage.stage === 'webgpuInclusivePixelBounds'
  );
  const pixelLocal = result.stageLocalMismatchSummaries.find(
    (stage) => stage.stage === 'webgpuInclusivePixelBounds'
  );
  assert.equal(result.decision, 'match', String(fixture.srcIndex));
  assert.equal(pixel.mismatchCount, 1, String(fixture.srcIndex));
  assert.equal(pixel.componentMismatchCount, 1, String(fixture.srcIndex));
  assert.equal(pixel.precisionAlignedCount, 1, String(fixture.srcIndex));
  assert.equal(pixel.precisionAlignedComponentCount, 1, String(fixture.srcIndex));
  assert.equal(pixel.semanticResidualCount, 0, String(fixture.srcIndex));
  assert.equal(pixel.semanticResidualComponentCount, 0, String(fixture.srcIndex));
  assert.equal(pixel.classification, 'precision-aligned', String(fixture.srcIndex));
  assert.equal(pixelLocal.representatives.length, 1, String(fixture.srcIndex));
  assert.equal(pixelLocal.representatives[0].srcIndex, fixture.srcIndex);
  assert.equal(
    pixelLocal.representatives[0].comparisonClassification,
    'precision-aligned',
    String(fixture.srcIndex)
  );
  assert.equal(result.firstMismatches.length, 0, String(fixture.srcIndex));
}
const groupedPrecisionExpected = precisionBoundaryCases.map((fixture, row) => {
  const record = expectedRecord(row);
  record.stages.projectedCenter.values = [...fixture.expectedCenter];
  record.stages.radius.values = [fixture.radius];
  record.stages.webgpuInclusivePixelBounds.values = [...fixture.expectedBounds];
  return record;
});
const groupedPrecisionActual = precisionBoundaryCases.map((fixture, row) => {
  const record = JSON.parse(JSON.stringify(groupedPrecisionExpected[row]));
  record.stages.projectedCenter.values = [...fixture.actualCenter];
  record.stages.webgpuInclusivePixelBounds.values = [...fixture.actualBounds];
  return record;
});
const groupedPrecisionContract = inputContract(precisionBoundaryCases.length);
const groupedPrecisionResult = comparePopulationAlignedSemanticChunkEvidence({
  inputContract: groupedPrecisionContract,
  candidateIndices: buildExplicitPopulationChunkIndices(groupedPrecisionContract),
  actualPackedEvidence: packActual(groupedPrecisionActual),
  actualRasterCompanionEvidence: packRasterCompanion(groupedPrecisionActual),
  actualRasterCompanionContract: rasterCompanionContract(
    groupedPrecisionContract,
    groupedPrecisionActual
  ),
  expectedRecordForRow: (row) => groupedPrecisionExpected[row]
});
const groupedPrecisionPixel = groupedPrecisionResult.stageSummaries.find(
  (stage) => stage.stage === 'webgpuInclusivePixelBounds'
);
const groupedPrecisionPixelLocal =
  groupedPrecisionResult.stageLocalMismatchSummaries.find(
    (stage) => stage.stage === 'webgpuInclusivePixelBounds'
  );
assert.equal(groupedPrecisionResult.decision, 'match');
assert.equal(groupedPrecisionPixel.mismatchCount, 4);
assert.equal(groupedPrecisionPixel.componentMismatchCount, 4);
assert.equal(groupedPrecisionPixel.precisionAlignedCount, 4);
assert.equal(groupedPrecisionPixel.precisionAlignedComponentCount, 4);
assert.equal(groupedPrecisionPixel.semanticResidualCount, 0);
assert.equal(groupedPrecisionPixel.semanticResidualComponentCount, 0);
assert.equal(groupedPrecisionPixel.classification, 'precision-aligned');
assert.equal(groupedPrecisionPixelLocal.representatives.length, 4);
assert.ok(groupedPrecisionPixelLocal.representatives.every(
  (representative) =>
    representative.comparisonClassification === 'precision-aligned'
));
assert.equal(groupedPrecisionResult.firstMismatches.length, 0);

const exactBoundaryMatch = comparePixelBoundaryFixture({
  expectedCenter: [200.25, 300.75],
  actualCenter: [200.25, 300.75],
  expectedRadius: 8
});
const exactPixel = exactBoundaryMatch.stageSummaries.find(
  (stage) => stage.stage === 'webgpuInclusivePixelBounds'
);
assert.equal(exactPixel.mismatchCount, 0);
assert.equal(exactPixel.precisionAlignedCount, 0);
assert.equal(exactPixel.semanticResidualCount, 0);

const syntheticMaxXBoundary = comparePixelBoundaryFixture({
  expectedCenter: [300.000001, 200.25],
  actualCenter: [300, 200.25],
  expectedRadius: 10,
  expectedBounds: [290, 190, 311, 211],
  actualBounds: [290, 190, 310, 211]
});
assert.equal(
  syntheticMaxXBoundary.stageSummaries.find(
    (stage) => stage.stage === 'webgpuInclusivePixelBounds'
  ).classification,
  'precision-aligned'
);

for (const [name, fixture] of [
  ['center-over-tolerance', {
    expectedCenter: [100.9999, 200.25],
    actualCenter: [101.01, 200.25],
    expectedRadius: 5
  }],
  ['radius-mismatch', {
    expectedCenter: [100.9999, 200.25],
    actualCenter: [101, 200.25],
    expectedRadius: 5,
    actualRadius: 6
  }],
  ['expected-bounds-mutation', {
    expectedCenter: [100.9999, 200.25],
    actualCenter: [101, 200.25],
    expectedRadius: 5,
    expectedBounds: [95, 195, 105, 206]
  }],
  ['actual-bounds-mutation', {
    expectedCenter: [100.9999, 200.25],
    actualCenter: [101, 200.25],
    expectedRadius: 5,
    actualBounds: [96, 195, 107, 206]
  }],
  ['multi-component-partial-explanation', {
    expectedCenter: [100.9999, 200.9999],
    actualCenter: [101, 201],
    expectedRadius: 5,
    actualBounds: [96, 196, 107, 207]
  }]
]) {
  const result = comparePixelBoundaryFixture(fixture);
  const pixel = result.stageSummaries.find(
    (stage) => stage.stage === 'webgpuInclusivePixelBounds'
  );
  assert.equal(pixel.precisionAlignedCount, 0, name);
  if (pixel.mismatchCount > 0) {
    assert.equal(pixel.semanticResidualCount, pixel.mismatchCount, name);
    assert.equal(pixel.classification, 'mismatch', name);
  }
  assert.notEqual(result.decision, 'match', name);
}

const eligibilityMismatchBoundary = comparePixelBoundaryFixture({
  expectedCenter: [100.9999, 200.25],
  actualCenter: [101, 200.25],
  expectedRadius: 5,
  actualRasterEligible: false
});
assert.equal(eligibilityMismatchBoundary.decision, 'mismatch');
assert.equal(
  eligibilityMismatchBoundary.stageSummaries.find(
    (stage) => stage.stage === 'webgpuInclusivePixelBounds'
  ).precisionAlignedCount,
  0
);

for (const [name, fixture] of [
  ['viewport-bounds-invalid', {
    expectedCenter: [0.0001, 200.25],
    actualCenter: [0, 200.25],
    expectedRadius: 5,
    actualBounds: [-1, 195, 5, 206]
  }],
  ['nonfinite-center', {
    expectedCenter: [100.9999, 200.25],
    actualCenter: [Number.NaN, 200.25],
    expectedRadius: 5,
    actualBounds: [96, 195, 106, 206]
  }]
]) {
  const result = comparePixelBoundaryFixture(fixture);
  assert.equal(result.decision, 'blocked', name);
  assert.equal(
    result.stageSummaries.find(
      (stage) => stage.stage === 'webgpuInclusivePixelBounds'
    ).precisionAlignedCount,
    0,
    name
  );
}

const orderedRecords = Array.from({ length: 3 }, (_, row) => expectedRecord(row));
const orderedActual = packActual(orderedRecords);
orderedActual[2] += 1;
orderedActual[3] += 1;
const orderedMismatch = comparePopulationAlignedSemanticChunkEvidence({
  inputContract: inputContract(3),
  candidateIndices: buildExplicitPopulationChunkIndices(inputContract(3)),
  actualPackedEvidence: orderedActual,
  actualRasterCompanionEvidence: packRasterCompanion(orderedRecords),
  actualRasterCompanionContract:
    rasterCompanionContract(inputContract(3), orderedRecords),
  expectedRecordForRow: (row) => orderedRecords[row]
});
assert.equal(orderedMismatch.decision, 'mismatch');
assert.equal(orderedMismatch.firstMismatches[0].localRow, 0);
assert.equal(orderedMismatch.firstMismatches[0].stage, 'conditionalStatePosition');
assert.equal(orderedMismatch.firstMismatches[0].component, 'x');

const manyRecords = Array.from({ length: 32 }, (_, row) => expectedRecord(row));
const manyActual = packActual(manyRecords);
for (let row = 0; row < manyRecords.length; row += 1) {
  manyActual[row * POPULATION_SEMANTIC_EVIDENCE_FLOAT_STRIDE + 2] += 1;
}
const boundedMismatch = comparePopulationAlignedSemanticChunkEvidence({
  inputContract: inputContract(manyRecords.length),
  candidateIndices: buildExplicitPopulationChunkIndices(inputContract(manyRecords.length)),
  actualPackedEvidence: manyActual,
  actualRasterCompanionEvidence: packRasterCompanion(manyRecords),
  actualRasterCompanionContract:
    rasterCompanionContract(inputContract(manyRecords.length), manyRecords),
  expectedRecordForRow: (row) => manyRecords[row]
});
assert.equal(boundedMismatch.firstMismatches.length, POPULATION_SEMANTIC_MAX_FIRST_MISMATCHES);

// Stage-local representatives remain available after an earlier stage fills the
// unchanged global first-16 component list.
const stageLocalRecords = Array.from(
  { length: 32 },
  (_, row) => expectedRecord(row)
);
const stageLocalActual = packActual(stageLocalRecords);
const stageLocalRasterActual = packRasterCompanion(stageLocalRecords);
for (let row = 0; row < 16; row += 1) {
  const base = row * POPULATION_SEMANTIC_EVIDENCE_FLOAT_STRIDE;
  stageLocalActual[base + 2] += 1;
}
stageLocalActual[3] += 2;
for (let row = 20; row < 24; row += 1) {
  const base = row * POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE;
  stageLocalRasterActual[base + 1] += 0.0005;
  stageLocalRasterActual[base + 7] += 1;
}
const stageLocalMismatch = comparePopulationAlignedSemanticChunkEvidence({
  inputContract: inputContract(stageLocalRecords.length),
  candidateIndices: buildExplicitPopulationChunkIndices(
    inputContract(stageLocalRecords.length)
  ),
  actualPackedEvidence: stageLocalActual,
  actualRasterCompanionEvidence: stageLocalRasterActual,
  actualRasterCompanionContract: rasterCompanionContract(
    inputContract(stageLocalRecords.length),
    stageLocalRecords
  ),
  expectedRecordForRow: (row) => stageLocalRecords[row]
});
assert.equal(stageLocalMismatch.decision, 'mismatch');
assert.equal(stageLocalMismatch.firstMismatches.length, 16);
assert.ok(
  stageLocalMismatch.firstMismatches.every(
    (entry) => entry.stage === 'conditionalStatePosition'
  )
);
assert.deepEqual(
  stageLocalMismatch.stageLocalMismatchSummaries.map((summary) => summary.stage),
  POPULATION_SEMANTIC_STAGE_CONTRACTS.map((stage) => stage.key)
);
const positionStageLocal = stageLocalMismatch.stageLocalMismatchSummaries.find(
  (summary) => summary.stage === 'conditionalStatePosition'
);
assert.equal(positionStageLocal.sourceMismatchRecordCount, 16);
assert.equal(positionStageLocal.sourceComponentMismatchCount, 17);
assert.equal(positionStageLocal.serializedRepresentativeRecordCount, 4);
assert.equal(positionStageLocal.representativeRecordLimit, 4);
assert.equal(positionStageLocal.truncated, true);
assert.deepEqual(
  positionStageLocal.representatives[0].mismatchComponents.map(
    ({ component, componentIndex }) => [component, componentIndex]
  ),
  [['x', 0], ['y', 1]]
);
const pixelStageLocal = stageLocalMismatch.stageLocalMismatchSummaries.find(
  (summary) => summary.stage === 'webgpuInclusivePixelBounds'
);
assert.equal(pixelStageLocal.sourceMismatchRecordCount, 4);
assert.equal(pixelStageLocal.sourceComponentMismatchCount, 4);
assert.equal(pixelStageLocal.serializedRepresentativeRecordCount, 4);
assert.equal(pixelStageLocal.truncated, false);
assert.deepEqual(
  pixelStageLocal.representatives.map(({ localRow }) => localRow),
  [20, 21, 22, 23]
);
const pixelRepresentative = pixelStageLocal.representatives[0];
assert.deepEqual(pixelRepresentative.stageComponentNames, ['minX', 'minY', 'maxX', 'maxY']);
assert.deepEqual(pixelRepresentative.expectedStageValues, [1, 2, 10, 11]);
assert.deepEqual(pixelRepresentative.actualStageValues, [1, 2, 10, 12]);
assert.deepEqual(
  pixelRepresentative.mismatchComponents.map(
    ({ component, componentIndex, expected, actual }) =>
      [component, componentIndex, expected, actual]
  ),
  [['maxY', 3, 11, 12]]
);
assert.equal(
  pixelRepresentative.dependencyContext.projectedCenter.expected[0],
  stageLocalRecords[20].stages.projectedCenter.values[0]
);
assert.ok(
  Math.abs(
    pixelRepresentative.dependencyContext.projectedCenter.actual[0] -
    (stageLocalRecords[20].stages.projectedCenter.values[0] + 0.0005)
  ) < 1e-4
);
assert.deepEqual(
  pixelRepresentative.dependencyContext.radius.expected,
  stageLocalRecords[20].stages.radius.values
);
assert.deepEqual(
  pixelRepresentative.dependencyContext.webgpuInclusivePixelBounds.actual,
  [1, 2, 10, 12]
);
assert.deepEqual(
  pixelRepresentative.dependencyContext.normalizedInclusiveTileBounds.actual,
  [0, 0, 1, 1]
);
assert.deepEqual(
  Object.keys(pixelRepresentative.dependencyContext),
  [...POPULATION_SEMANTIC_STAGE_LOCAL_CONTEXT_STAGE_KEYS]
);
assert.equal(pixelRepresentative.actualEvidenceSource, 'diagnostic-gpu-readback');
assert.equal(pixelRepresentative.productionCalculationDependsOnDiagnosticReadback, false);
assert.deepEqual(
  validatePopulationSemanticStageLocalMismatchSummaries({
    stageLocalMismatchSummaries: stageLocalMismatch.stageLocalMismatchSummaries,
    stageSummaries: stageLocalMismatch.stageSummaries,
    scope: 'single-chunk',
    rangeStart: PRODUCTION_RESIDENT_RANGE_START,
    rangeCount: stageLocalRecords.length,
    chunkIndex: 0
  }),
  []
);
assert.ok(
  stageLocalMismatch.stageLocalMismatchSummaries
    .filter((summary) => summary.sourceMismatchRecordCount === 0)
    .every(
      (summary) =>
        summary.representatives.length === 0 && summary.truncated === false
    )
);

function mutateStageLocalResult(result, mutation) {
  const clone = JSON.parse(JSON.stringify(result));
  mutation(clone.stageLocalMismatchSummaries, clone.stageSummaries);
  return validatePopulationSemanticStageLocalMismatchSummaries({
    stageLocalMismatchSummaries: clone.stageLocalMismatchSummaries,
    stageSummaries: clone.stageSummaries,
    scope: 'single-chunk',
    rangeStart: PRODUCTION_RESIDENT_RANGE_START,
    rangeCount: stageLocalRecords.length,
    chunkIndex: 0
  });
}

const stageLocalDrifts = [
  (summaries) => { summaries[1].stage = summaries[0].stage; },
  (summaries) => { summaries[1].representatives.pop(); },
  (summaries) => {
    summaries[1].representatives.push(
      JSON.parse(JSON.stringify(summaries[1].representatives[0]))
    );
    summaries[1].serializedRepresentativeRecordCount += 1;
  },
  (summaries) => {
    summaries[1].representatives[1].localRow =
      summaries[1].representatives[0].localRow;
    summaries[1].representatives[1].srcIndex =
      summaries[1].representatives[0].srcIndex;
    summaries[1].representatives[1].globalResidentRow =
      summaries[1].representatives[0].globalResidentRow;
    summaries[1].representatives[1].scanOrderKey =
      [...summaries[1].representatives[0].scanOrderKey];
  },
  (summaries) => { summaries[11].representatives[0].srcIndex += 1; },
  (summaries) => {
    summaries[1].representatives[0].mismatchComponents.reverse();
  },
  (summaries) => {
    summaries[11].representatives[0].mismatchComponents[0].component = 'invalid';
  },
  (summaries) => {
    summaries[11].representatives[0].mismatchComponents[0].expected = null;
  },
  (summaries) => {
    summaries[11].representatives[0].mismatchComponents[0].absoluteError = 2;
  },
  (summaries) => {
    summaries[11].representatives[0].mismatchComponents[0].tolerance = 1;
  },
  (summaries) => {
    summaries[11].representatives[0].dependencyContext.projectedCenter.actual.pop();
  },
  (summaries) => {
    summaries[11].representatives[0].comparisonClassification = 'invalid';
  },
  (summaries) => { summaries[11].sourcePrecisionAlignedRecordCount += 1; },
  (summaries) => { summaries[1].sourceMismatchRecordCount += 1; },
  (summaries) => { summaries[1].truncated = false; }
];
for (const mutation of stageLocalDrifts) {
  assert.ok(mutateStageLocalResult(stageLocalMismatch, mutation).length > 0);
}
const typedArrayStageLocal = JSON.parse(JSON.stringify(stageLocalMismatch));
typedArrayStageLocal.stageLocalMismatchSummaries[11]
  .representatives[0].dependencyContext.projectedCenter.actual =
    new Float32Array([1, 2]);
assert.ok(validatePopulationSemanticStageLocalMismatchSummaries({
  stageLocalMismatchSummaries: typedArrayStageLocal.stageLocalMismatchSummaries,
  stageSummaries: typedArrayStageLocal.stageSummaries,
  scope: 'single-chunk',
  rangeStart: PRODUCTION_RESIDENT_RANGE_START,
  rangeCount: stageLocalRecords.length,
  chunkIndex: 0
}).some((reason) => reason.includes('typed-array')));

const partial = comparePopulationAlignedSemanticChunkEvidence({
  inputContract: inputContract(2),
  candidateIndices: buildExplicitPopulationChunkIndices(inputContract(2)),
  actualPackedEvidence: packActual([expectedRecord(0)]),
  actualRasterCompanionEvidence:
    packRasterCompanion([expectedRecord(0), expectedRecord(1)]),
  actualRasterCompanionContract:
    rasterCompanionContract(
      inputContract(2),
      [expectedRecord(0), expectedRecord(1)]
    ),
  expectedRecordForRow: (row) => expectedRecord(row)
});
assert.equal(partial.decision, 'blocked');
assert.ok(partial.coverage.missingCount > 0);
const partialPositionStageLocal = partial.stageLocalMismatchSummaries.find(
  (summary) => summary.stage === 'conditionalStatePosition'
);
assert.equal(partialPositionStageLocal.sourceClassification.startsWith('blocked-'), true);
assert.equal(partialPositionStageLocal.evidenceComplete, false);

for (const candidates of [
  new Uint32Array([524288, 524288]),
  new Uint32Array([524288, 1048576]),
  new Uint32Array([524289, 524288])
]) {
  const result = comparePopulationAlignedSemanticChunkEvidence({
    inputContract: inputContract(2),
    candidateIndices: candidates,
    actualPackedEvidence: packActual([expectedRecord(0), expectedRecord(1)]),
    actualRasterCompanionEvidence:
      packRasterCompanion([expectedRecord(0), expectedRecord(1)]),
    actualRasterCompanionContract:
      rasterCompanionContract(
        inputContract(2),
        [expectedRecord(0), expectedRecord(1)]
      ),
    expectedRecordForRow: (row) => expectedRecord(row)
  });
  assert.equal(result.decision, 'blocked');
  assert.equal(result.coverage.coverageComplete, false);
}

const rowMismatchActual = packActual([expectedRecord(0)]);
rowMismatchActual[0] = 1;
const rowMismatch = comparePopulationAlignedSemanticChunkEvidence({
  inputContract: inputContract(1),
  candidateIndices: buildExplicitPopulationChunkIndices(inputContract(1)),
  actualPackedEvidence: rowMismatchActual,
  actualRasterCompanionEvidence: packRasterCompanion([expectedRecord(0)]),
  actualRasterCompanionContract:
    rasterCompanionContract(inputContract(1), [expectedRecord(0)]),
  expectedRecordForRow: () => expectedRecord(0)
});
assert.equal(rowMismatch.decision, 'blocked');
assert.equal(rowMismatch.coverage.orderMismatchCount, 1);

const fractionalRowActual = packActual([expectedRecord(0)]);
fractionalRowActual[0] = 0.25;
const fractionalRow = comparePacked(
  fractionalRowActual,
  [expectedRecord(0)]
);
assert.equal(fractionalRow.decision, 'blocked');
assert.equal(fractionalRow.coverage.coverageComplete, false);

for (const invalidRow of [Number.NaN, Number.POSITIVE_INFINITY, -1, 1]) {
  const invalidRowActual = packActual([expectedRecord(0)]);
  invalidRowActual[0] = invalidRow;
  const invalidRowResult = comparePacked(
    invalidRowActual,
    [expectedRecord(0)]
  );
  assert.equal(invalidRowResult.decision, 'blocked');
  assert.equal(invalidRowResult.coverage.coverageComplete, false);
}

const strictLengthRecords = [expectedRecord(0), expectedRecord(1)];
const strictLengthActual = packActual(strictLengthRecords);
for (const floatCount of [
  strictLengthActual.length - 1,
  strictLengthActual.length + 1,
  strictLengthActual.length - POPULATION_SEMANTIC_EVIDENCE_FLOAT_STRIDE,
  strictLengthActual.length + POPULATION_SEMANTIC_EVIDENCE_FLOAT_STRIDE
]) {
  const wrongLengthActual = new Float32Array(floatCount);
  wrongLengthActual.set(
    strictLengthActual.subarray(0, Math.min(strictLengthActual.length, floatCount))
  );
  const wrongLengthResult = comparePacked(wrongLengthActual, strictLengthRecords);
  assert.equal(wrongLengthResult.decision, 'blocked');
  assert.equal(wrongLengthResult.coverage.coverageComplete, false);
}
const unreadablePackedEvidence = comparePacked(
  { length: strictLengthActual.length },
  strictLengthRecords
);
assert.equal(unreadablePackedEvidence.decision, 'blocked');
assert.equal(unreadablePackedEvidence.coverage.coverageComplete, false);

const companionLengthRecords = [expectedRecord(0), expectedRecord(1)];
const exactCompanion = packRasterCompanion(companionLengthRecords);
for (const floatCount of [
  exactCompanion.length - 1,
  exactCompanion.length + 1,
  exactCompanion.length - POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE,
  exactCompanion.length + POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE
]) {
  const evidence = new Float32Array(floatCount);
  evidence.set(exactCompanion.subarray(0, Math.min(floatCount, exactCompanion.length)));
  const result = comparePacked(
    packActual(companionLengthRecords),
    companionLengthRecords,
    evidence
  );
  assert.equal(result.decision, 'blocked');
  assert.equal(result.rasterCompanionCoverage.evidenceLengthExact, false);
}
const unreadableCompanion = comparePacked(
  packActual(companionLengthRecords),
  companionLengthRecords,
  { length: exactCompanion.length }
);
assert.equal(unreadableCompanion.decision, 'blocked');
assert.ok(
  unreadableCompanion.blockedReasons.includes(
    'raster-companion-evidence-not-readable'
  )
);

for (const [label, mutate] of [
  ['identity', (contract) => {
    contract.sourceWorksetResourceIdentity = 'different-workset';
  }],
  ['row-alignment', (contract) => {
    contract.rowAlignment = 'reordered';
  }]
]) {
  const contract = structuredClone(
    rasterCompanionContract(inputContract(2), companionLengthRecords)
  );
  mutate(contract);
  const result = comparePacked(
    packActual(companionLengthRecords),
    companionLengthRecords,
    exactCompanion,
    contract
  );
  assert.equal(result.decision, 'blocked', label);
  assert.equal(result.rasterCompanionCoverage.rowAlignmentVerified, false, label);
}

for (const [offset, value] of [
  [1, Number.NaN],
  [3, Number.POSITIVE_INFINITY],
  [4, 1.5],
  [8, 0.5]
]) {
  const evidence = new Float32Array(packRasterCompanion([expectedRecord(0)]));
  evidence[offset] = value;
  const result = comparePacked(
    packActual([expectedRecord(0)]),
    [expectedRecord(0)],
    evidence
  );
  assert.equal(result.decision, 'blocked');
  const affectedStages = result.stageSummaries.slice(-4);
  assert.ok(affectedStages.some((stage) => stage.invalidCount > 0));
}

const expectedRasterEligible = expectedRecord(0, {
  eligible: true,
  rasterEligible: true
});
const actualRasterIneligible = new Float32Array(
  POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE
);
const eligibilityMismatch = comparePacked(
  packActual([expectedRasterEligible]),
  [expectedRasterEligible],
  actualRasterIneligible
);
assert.equal(eligibilityMismatch.decision, 'mismatch');
assert.equal(eligibilityMismatch.firstMismatches[0].stage, 'productionRasterEligibility');
assert.equal(
  eligibilityMismatch.stageSummaries.find(
    (stage) => stage.stage === 'productionRasterEligibility'
  ).classification,
  'mismatch'
);
for (const stage of eligibilityMismatch.stageSummaries.slice(-4)) {
  assert.equal(stage.notApplicableCount, 1);
  assert.equal(stage.mismatchCount, 0);
  assert.equal(stage.maxAbsoluteError, null);
}

const expectedCudaIneligible = expectedRecord(0, {
  eligible: true,
  rasterEligible: false
});
const actualWebGpuEligible = packRasterCompanion([expectedRecord(0)]);
const reverseEligibilityMismatch = comparePacked(
  packActual([expectedCudaIneligible]),
  [expectedCudaIneligible],
  actualWebGpuEligible
);
assert.equal(reverseEligibilityMismatch.decision, 'mismatch');
assert.equal(
  reverseEligibilityMismatch.firstMismatches[0].stage,
  'productionRasterEligibility'
);

const exactBoundaryRect = buildCudaNormalizedInclusiveTileRect({
  centerX: 16,
  centerY: 16,
  radius: 1,
  canvasWidth: 1280,
  canvasHeight: 720,
  tileSize: 16
});
assert.deepEqual(exactBoundaryRect.minInclusive, [0, 0]);
assert.deepEqual(exactBoundaryRect.maxExclusive, [2, 2]);
assert.deepEqual(exactBoundaryRect.normalizedInclusive, [0, 0, 1, 1]);
const canvasEdgeBounds = buildExpectedWebGpuInclusivePixelBounds({
  centerX: 1279,
  centerY: 719,
  radius: 4,
  canvasWidth: 1280,
  canvasHeight: 720
});
assert.deepEqual(canvasEdgeBounds, [1275, 715, 1279, 719]);
const partiallyOffscreenBounds = buildExpectedWebGpuInclusivePixelBounds({
  centerX: -0.25,
  centerY: 8.5,
  radius: 3,
  canvasWidth: 1280,
  canvasHeight: 720
});
assert.deepEqual(partiallyOffscreenBounds, [0, 5, 3, 12]);
const emptyCudaRect = buildCudaNormalizedInclusiveTileRect({
  centerX: -32,
  centerY: 4,
  radius: 1,
  canvasWidth: 1280,
  canvasHeight: 720,
  tileSize: 16
});
assert.equal(emptyCudaRect.nonEmpty, false);
assert.equal(emptyCudaRect.normalizedInclusive, null);

function evaluateProductionCudaAlignedTileBounds({
  centerX,
  centerY,
  radius,
  canvasWidth,
  canvasHeight,
  tileSize
}) {
  const tileCols = Math.ceil(canvasWidth / tileSize);
  const tileRows = Math.ceil(canvasHeight / tileSize);
  const truncateAndClamp = (value, maximum) =>
    Math.min(maximum, Math.max(0, Math.trunc(value)));
  const minimum = [
    truncateAndClamp((centerX - radius) / tileSize, tileCols),
    truncateAndClamp((centerY - radius) / tileSize, tileRows)
  ];
  const maximumExclusive = [
    truncateAndClamp(
      (centerX + radius + tileSize - 1) / tileSize,
      tileCols
    ),
    truncateAndClamp(
      (centerY + radius + tileSize - 1) / tileSize,
      tileRows
    )
  ];
  const nonEmpty =
    maximumExclusive[0] > minimum[0] &&
    maximumExclusive[1] > minimum[1];
  return {
    minInclusive: minimum,
    maxExclusive: maximumExclusive,
    normalizedInclusive: nonEmpty
      ? [
          minimum[0],
          minimum[1],
          maximumExclusive[0] - 1,
          maximumExclusive[1] - 1
        ]
      : null,
    nonEmpty
  };
}

function evaluateLegacyPixelDerivedTileBounds({
  centerX,
  centerY,
  radius,
  canvasWidth,
  canvasHeight,
  tileSize
}) {
  const pixelBounds = buildExpectedWebGpuInclusivePixelBounds({
    centerX,
    centerY,
    radius,
    canvasWidth,
    canvasHeight
  });
  return [
    Math.floor(pixelBounds[0] / tileSize),
    Math.floor(pixelBounds[1] / tileSize),
    Math.floor(pixelBounds[2] / tileSize),
    Math.floor(pixelBounds[3] / tileSize)
  ];
}

const tileBoundsFixtures = [
  { name: 'tile-boundary', centerX: 8, centerY: 8, radius: 8 },
  { name: 'tile-boundary-before', centerX: 7.75, centerY: 7.75, radius: 8 },
  { name: 'tile-boundary-after', centerX: 9.25, centerY: 9.25, radius: 8 },
  { name: 'fractional-center', centerX: 24.25, centerY: 31.75, radius: 5 },
  { name: 'partially-off-screen', centerX: -0.25, centerY: 8.5, radius: 3 },
  { name: 'canvas-right-bottom-edge', centerX: 1279, centerY: 719, radius: 4 },
  { name: 'empty-rect', centerX: -32, centerY: 4, radius: 1 },
  { name: 'one-tile-rect', centerX: 4, centerY: 4, radius: 2 },
  { name: 'multiple-tile-rect', centerX: 32, centerY: 32, radius: 20 },
  { name: 'grid-edge-exclusive-max', centerX: 1279, centerY: 719, radius: 32 },
  {
    name: 'non-default-tile-size',
    centerX: 4,
    centerY: 4,
    radius: 4,
    tileSize: 8
  }
];
for (const fixture of tileBoundsFixtures) {
  const input = {
    canvasWidth: 1280,
    canvasHeight: 720,
    tileSize: 16,
    ...fixture
  };
  delete input.name;
  const productionRect = evaluateProductionCudaAlignedTileBounds(input);
  const cudaRect = buildCudaNormalizedInclusiveTileRect(input);
  assert.deepEqual(
    productionRect,
    {
      minInclusive: cudaRect.minInclusive,
      maxExclusive: cudaRect.maxExclusive,
      normalizedInclusive: cudaRect.normalizedInclusive,
      nonEmpty: cudaRect.nonEmpty
    },
    fixture.name
  );
  if (!productionRect.nonEmpty) {
    assert.equal(productionRect.normalizedInclusive, null, fixture.name);
  }
}
const legacyExtraMaximumFixture = {
  centerX: 8,
  centerY: 8,
  radius: 8,
  canvasWidth: 1280,
  canvasHeight: 720,
  tileSize: 16
};
assert.deepEqual(
  evaluateLegacyPixelDerivedTileBounds(legacyExtraMaximumFixture),
  [0, 0, 1, 1]
);
assert.deepEqual(
  evaluateProductionCudaAlignedTileBounds(legacyExtraMaximumFixture)
    .normalizedInclusive,
  [0, 0, 0, 0]
);

function fixedProjectionParams() {
  return new Float32Array([
    1, 1280, 720, 0,
    1, 1, 1, 0,
    500, 500, 639.5, 359.5,
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  ]);
}

function oneRecordRaw(x, y, z) {
  return {
    xyz: new Float32Array([x, y, z]), xyzDim: 3,
    f_dc: new Float32Array([0, 0, 0]), fdcDim: 3,
    f_rest: new Float32Array(45), frestDim: 45,
    activeShDegree: 2,
    activeShDegreeT: 2,
    rot4d: true,
    opacity: new Float32Array([1]), opacityDim: 1,
    scale_xyz: new Float32Array([1, 1, 1]), scaleXYZDim: 3,
    scale_t: new Float32Array([1]), scaleTDim: 1,
    rotation: new Float32Array([1, 0, 0, 0]), rotationDim: 4,
    rotation_r: new Float32Array([1, 0, 0, 0]), rotationRDim: 4,
    t: new Float32Array([0]), tDim: 1
  };
}

assert.equal(buildCudaTileInputExpectedAlpha({
  rawOpacityLogit: 0,
  temporalWeight: 1
}), 0.5);
const representativeLogit = Math.log(
  representativeActivatedOpacity / (1 - representativeActivatedOpacity)
);
assert.ok(Math.abs(buildCudaTileInputExpectedAlpha({
  rawOpacityLogit: representativeLogit,
  temporalWeight: representativeTemporalOpacity / representativeActivatedOpacity
}) - representativeTemporalOpacity) < 1e-12);
assert.notEqual(buildCudaTileInputExpectedAlpha({
  rawOpacityLogit: representativeActivatedOpacity,
  temporalWeight: representativeTemporalOpacity / representativeActivatedOpacity
}), representativeTemporalOpacity);
assert.equal(buildCudaTileInputExpectedAlpha({
  rawOpacityLogit: Number.NaN,
  temporalWeight: 1
}), null);

const f32AlphaFixtureDefaults = Object.freeze({
  timestamp: 23.2,
  scalingModifier: 1,
  sigmaScale: 1
});
const f32AlphaFixtures = Object.freeze([
  Object.freeze({
    srcIndex: 817431,
    rotation: [-0.7209692597389221, 0.42837876081466675, -0.6182104349136353, 0.052885591983795166],
    rotationR: [0.7151115536689758, -0.30974531173706055, 0.05181895196437836, 0.277172327041626],
    sourceScaleXYZ: [0.25277775526046753, 0.16258607804775238, 0.8742130994796753],
    sourceScaleT: 0.04849153757095337,
    rawOpacityLogit: 2.650578260421753,
    tCenter: 22.996662139892578,
    temporalWeight: 0.36620765924453735,
    alpha: 0.34205499291419983
  }),
  Object.freeze({
    srcIndex: 823750,
    rotation: [-0.17102831602096558, -0.9763918519020081, 0.33638453483581543, -0.17507825791835785],
    rotationR: [0.927533745765686, 0.09904562681913376, -0.05579939857125282, -0.16718503832817078],
    sourceScaleXYZ: [0.05201122537255287, 0.17680834233760834, 0.03804946690797806],
    sourceScaleT: 0.0558256059885025,
    rawOpacityLogit: 4.612815856933594,
    tCenter: 23.24584197998047,
    temporalWeight: 0.5941779613494873,
    alpha: 0.5883393883705139,
    browserActual: 0.5883393883705139
  }),
  Object.freeze({
    srcIndex: 826596,
    rotation: [-0.16494120657444, 0.33464086055755615, 0.9835435152053833, 0.37130799889564514],
    rotationR: [0.752812922000885, 0.24305163323879242, -0.10210423916578293, -0.1343451291322708],
    sourceScaleXYZ: [0.1994161307811737, 0.04217933490872383, 0.051031388342380524],
    sourceScaleT: 0.05987834185361862,
    rawOpacityLogit: 5.712879180908203,
    tCenter: 23.249025344848633,
    temporalWeight: 0.5336029529571533,
    alpha: 0.5318461656570435,
    browserActual: 0.5318462252616882
  }),
  Object.freeze({
    srcIndex: 828798,
    rotation: [-0.5192295908927917, -0.43392089009284973, 0.25467661023139954, -0.7996122241020203],
    rotationR: [0.8168833255767822, 0.17581424117088318, 0.20370417833328247, -0.05118275061249733],
    sourceScaleXYZ: [0.03612266853451729, 0.04016087204217911, 0.1419132500886917],
    sourceScaleT: 0.05289372801780701,
    rawOpacityLogit: 5.801218509674072,
    tCenter: 23.240522384643555,
    temporalWeight: 0.6463044881820679,
    alpha: 0.6443560719490051,
    browserActual: 0.6443560123443604
  }),
  Object.freeze({
    srcIndex: 829562,
    rotation: [-0.24845446646213531, -0.28283023834228516, -0.08046253770589828, 0.936150074005127],
    rotationR: [0.32077842950820923, -0.4345601499080658, -0.12950782477855682, -0.5726639032363892],
    sourceScaleXYZ: [0.10413569957017899, 0.03419298306107521, 0.13865502178668976],
    sourceScaleT: 0.049943096935749054,
    rawOpacityLogit: 3.8623690605163574,
    tCenter: 23.24858856201172,
    temporalWeight: 0.5619015097618103,
    alpha: 0.5503345131874084,
    browserActual: 0.5503345727920532
  }),
  Object.freeze({
    srcIndex: 832266,
    rotation: [0.31482407450675964, 0.48717713356018066, 0.02764228545129299, 0.8673467040061951],
    rotationR: [0.7795888185501099, 0.07221396267414093, 0.2869412302970886, 0.0771029144525528],
    sourceScaleXYZ: [0.04127980023622513, 0.17192712426185608, 0.09357568621635437],
    sourceScaleT: 0.05532533675432205,
    rawOpacityLogit: 8.604650497436523,
    tCenter: 23.151559829711914,
    temporalWeight: 0.5735270977020264,
    alpha: 0.5734220743179321,
    browserActual: 0.5734220743179321
  }),
  Object.freeze({
    srcIndex: 834848,
    rotation: [0.3933086097240448, 0.6010604500770569, -0.6753302216529846, -0.2742021381855011],
    rotationR: [0.5370784401893616, 0.23489278554916382, -0.19469054043293, -0.44024309515953064],
    sourceScaleXYZ: [0.1661660075187683, 0.7008224725723267, 0.1818162053823471],
    sourceScaleT: 0.0438644103705883,
    rawOpacityLogit: 4.071102619171143,
    tCenter: 23.54320526123047,
    temporalWeight: 0.05061718076467514,
    alpha: 0.049768202006816864
  }),
  Object.freeze({
    srcIndex: 839237,
    rotation: [0.8870527744293213, 0.45144516229629517, -0.01585867442190647, 0.10149180889129639],
    rotationR: [0.9044269323348999, 0.05455229803919792, 0.029964786022901535, -0.16283413767814636],
    sourceScaleXYZ: [0.18955974280834198, 0.04730967804789543, 0.071531742811203],
    sourceScaleT: 0.05871327221393585,
    rawOpacityLogit: 9.379287719726562,
    tCenter: 23.391206741333008,
    temporalWeight: 0.049987293779850006,
    alpha: 0.04998307302594185
  }),
  Object.freeze({
    srcIndex: 842779,
    rotation: [-0.18988437950611115, 0.8211007714271545, 0.5673241019248962, -0.017591366544365883],
    rotationR: [1.111507773399353, -0.06539185345172882, -0.035201605409383774, 0.04466940835118294],
    sourceScaleXYZ: [0.19313904643058777, 0.15944840013980865, 0.05121394991874695],
    sourceScaleT: 0.0711042582988739,
    rawOpacityLogit: 14.658591270446777,
    tCenter: 23.43128776550293,
    temporalWeight: 0.05000424385070801,
    alpha: 0.05000422149896622
  }),
  Object.freeze({
    srcIndex: 870555,
    rotation: [0.3173673450946808, 0.3990182876586914, -0.9661105871200562, 0.12350577861070633],
    rotationR: [0.9247819185256958, -0.07042082399129868, -0.13608649373054504, 0.001143406261689961],
    sourceScaleXYZ: [0.16049206256866455, 0.03821925073862076, 0.05611656978726387],
    sourceScaleT: 0.05411629006266594,
    rawOpacityLogit: 11.330103874206543,
    tCenter: 23.253604888916016,
    temporalWeight: 0.4775067865848541,
    alpha: 0.4775010347366333,
    browserActual: 0.4775010347366333
  })
]);
const buildF32AlphaFixture = (fixture) =>
  buildProductionTileInputAlphaF32Central({
    ...f32AlphaFixtureDefaults,
    rawOpacityLogit: fixture.rawOpacityLogit,
    sourceScaleXYZ: fixture.sourceScaleXYZ,
    sourceScaleT: fixture.sourceScaleT,
    rotation: fixture.rotation,
    rotationR: fixture.rotationR,
    tCenter: fixture.tCenter
  });
for (const fixture of f32AlphaFixtures) {
  const actual = buildF32AlphaFixture(fixture);
  assert.deepEqual(actual, {
    temporalWeight: fixture.temporalWeight,
    alpha: fixture.alpha
  }, `srcIndex ${fixture.srcIndex}`);
  assert.equal(Object.isFrozen(actual), true, `srcIndex ${fixture.srcIndex}`);
  if (fixture.browserActual != null) {
    assert.ok(
      Math.abs(actual.alpha - fixture.browserActual) <= 1e-5,
      `srcIndex ${fixture.srcIndex}`
    );
  }
}
assert.equal(
  buildF32AlphaFixture(f32AlphaFixtures.find(({ srcIndex }) => srcIndex === 842779))
    .temporalWeight > 0.05,
  true
);
assert.equal(
  buildF32AlphaFixture(f32AlphaFixtures.find(({ srcIndex }) => srcIndex === 839237))
    .temporalWeight > 0.05,
  false
);

const centralInput = {
  rawOpacityLogit: 0,
  sourceScaleXYZ: [1, 1, 1],
  sourceScaleT: 0.5,
  rotation: [1, 0, 0, 0],
  rotationR: [1, 0, 0, 0],
  timestamp: 0.5,
  tCenter: 0,
  scalingModifier: 1,
  sigmaScale: 1
};
assert.deepEqual(buildProductionTileInputAlphaF32Central(centralInput), {
  temporalWeight: 0.6065306663513184,
  alpha: 0.3032653331756592
});
assert.notDeepEqual(
  buildProductionTileInputAlphaF32Central(centralInput),
  buildProductionTileInputAlphaF32Central({
    ...centralInput,
    sourceScaleT: Math.fround(Math.exp(centralInput.sourceScaleT))
  })
);
assert.deepEqual(
  buildProductionTileInputAlphaF32Central({
    ...centralInput,
    actualAlpha: -1,
    actualTemporalWeight: -1
  }),
  buildProductionTileInputAlphaF32Central(centralInput)
);
for (const invalidInput of [
  {},
  { ...centralInput, rawOpacityLogit: Number.NaN },
  { ...centralInput, sourceScaleXYZ: [1, 1] },
  { ...centralInput, sourceScaleT: Number.POSITIVE_INFINITY },
  { ...centralInput, rotation: [1, 0, 0] },
  { ...centralInput, rotationR: [1, 0, 0, Number.NaN] },
  { ...centralInput, timestamp: Number.NaN },
  { ...centralInput, tCenter: Number.NEGATIVE_INFINITY },
  { ...centralInput, scalingModifier: Number.NaN },
  { ...centralInput, sigmaScale: Number.NaN }
]) {
  assert.equal(buildProductionTileInputAlphaF32Central(invalidInput), null);
}
const fixtureAlphaByPartition = (partitionSize) => {
  const values = [];
  for (let start = 0; start < f32AlphaFixtures.length; start += partitionSize) {
    values.push(...f32AlphaFixtures
      .slice(start, start + partitionSize)
      .map((fixture) => buildF32AlphaFixture(fixture).alpha));
  }
  return values;
};
assert.deepEqual(fixtureAlphaByPartition(1), fixtureAlphaByPartition(4));
assert.deepEqual(fixtureAlphaByPartition(4), fixtureAlphaByPartition(7));

assert.match(
  POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE.productionTileInputAlpha,
  new RegExp(PRODUCTION_TILE_INPUT_ALPHA_F32_CENTRAL_ORACLE_VERSION)
);

const degree2RgbRaw = oneRecordRaw(0, 0, 1);
assert.deepEqual(
  buildCudaDegree2TileInputExpectedRgb({
    raw: degree2RgbRaw,
    srcIndex: 0,
    cameraWorldPosition: [0, 0, 0]
  }),
  [0.5, 0.5, 0.5]
);
const dcRgbRaw = oneRecordRaw(0, 0, 1);
dcRgbRaw.f_dc[0] = 1;
const dcRgb = buildCudaDegree2TileInputExpectedRgb({
  raw: dcRgbRaw,
  srcIndex: 0,
  cameraWorldPosition: [0, 0, 0]
});
assert.ok(Math.abs(dcRgb[0] - (0.5 + 0.28209479177387814)) < 1e-12);
degree2RgbRaw.f_rest[3] = 1;
const forwardRgb = buildCudaDegree2TileInputExpectedRgb({
  raw: degree2RgbRaw,
  srcIndex: 0,
  cameraWorldPosition: [0, 0, 0]
});
const reverseRgb = buildCudaDegree2TileInputExpectedRgb({
  raw: degree2RgbRaw,
  srcIndex: 0,
  cameraWorldPosition: [0, 0, 2]
});
assert.ok(forwardRgb[0] > 0.5);
assert.ok(reverseRgb[0] < 0.5);
assert.ok(Math.abs(forwardRgb[0] - (0.5 + 0.4886025119029199)) < 1e-12);
const degree2CoefficientRaw = oneRecordRaw(0, 0, 1);
degree2CoefficientRaw.f_rest[15] = 1;
assert.ok(buildCudaDegree2TileInputExpectedRgb({
  raw: degree2CoefficientRaw,
  srcIndex: 0,
  cameraWorldPosition: [0, 0, 0]
})[0] > 0.5);
const degree2OneHotPosition = [2, 3, 4];
const degree2OneHotLength = Math.hypot(...degree2OneHotPosition);
const [degree2OneHotX, degree2OneHotY, degree2OneHotZ] =
  degree2OneHotPosition.map((value) => value / degree2OneHotLength);
const degree2OneHotScales = [
  0.28209479177387814,
  -0.4886025119029199 * degree2OneHotY,
  0.4886025119029199 * degree2OneHotZ,
  -0.4886025119029199 * degree2OneHotX,
  1.0925484305920792 * degree2OneHotX * degree2OneHotY,
  -1.0925484305920792 * degree2OneHotY * degree2OneHotZ,
  0.31539156525252005 * (
    2 * degree2OneHotZ * degree2OneHotZ -
    degree2OneHotX * degree2OneHotX -
    degree2OneHotY * degree2OneHotY
  ),
  -1.0925484305920792 * degree2OneHotX * degree2OneHotZ,
  0.5462742152960396 * (
    degree2OneHotX * degree2OneHotX -
    degree2OneHotY * degree2OneHotY
  )
];
const degree2OneHotResults = [];
for (let shIndex = 0; shIndex <= 8; shIndex += 1) {
  const component = shIndex % 3;
  const raw = oneRecordRaw(...degree2OneHotPosition);
  if (shIndex === 0) {
    raw.f_dc[component] = 0.5;
  } else {
    raw.f_rest[(shIndex - 1) * 3 + component] = 0.5;
  }
  const actual = buildCudaDegree2TileInputExpectedRgb({
    raw,
    srcIndex: 0,
    cameraWorldPosition: [0, 0, 0]
  });
  const expected = [0.5, 0.5, 0.5];
  expected[component] = Math.max(
    0.5 + 0.5 * degree2OneHotScales[shIndex],
    0
  );
  assert.ok(
    actual.every((value, index) => Math.abs(value - expected[index]) < 1e-12),
    `sh[${shIndex}] coefficient-major RGB triplet`
  );
  degree2OneHotResults.push(actual);
}
assert.ok(degree2OneHotResults[5][2] < 0.5, 'sh[5] yz scale is negative');
assert.ok(degree2OneHotResults[7][1] < 0.5, 'sh[7] xz scale is negative');
const degree2GateRaw = oneRecordRaw(...degree2OneHotPosition);
degree2GateRaw.activeShDegree = 1;
degree2GateRaw.f_rest[12] = 0.5;
assert.deepEqual(buildCudaDegree2TileInputExpectedRgb({
  raw: degree2GateRaw,
  srcIndex: 0,
  cameraWorldPosition: [0, 0, 0]
}), [0.5, 0.5, 0.5]);
const originalPositionRgbRaw = oneRecordRaw(...degree2OneHotPosition);
originalPositionRgbRaw.f_rest[12] = 0.5;
const originalPositionRgb = buildCudaDegree2TileInputExpectedRgb({
  raw: originalPositionRgbRaw,
  srcIndex: 0,
  cameraWorldPosition: [1, 1, 1]
});
originalPositionRgbRaw.conditionalStatePosition = new Float32Array([-20, 30, -40]);
assert.deepEqual(buildCudaDegree2TileInputExpectedRgb({
  raw: originalPositionRgbRaw,
  srcIndex: 0,
  cameraWorldPosition: [1, 1, 1]
}), originalPositionRgb);
degree2RgbRaw.activeShDegreeT = 0;
assert.deepEqual(
  buildCudaDegree2TileInputExpectedRgb({
    raw: degree2RgbRaw,
    srcIndex: 0,
    cameraWorldPosition: [0, 0, 0]
  }),
  forwardRgb
);
const upperClampRaw = oneRecordRaw(0, 0, 1);
upperClampRaw.f_dc[0] = 4;
assert.ok(buildCudaDegree2TileInputExpectedRgb({
  raw: upperClampRaw,
  srcIndex: 0,
  cameraWorldPosition: [0, 0, 0]
})[0] > 1);
upperClampRaw.f_dc[0] = -4;
assert.equal(buildCudaDegree2TileInputExpectedRgb({
  raw: upperClampRaw,
  srcIndex: 0,
  cameraWorldPosition: [0, 0, 0]
})[0], 0);

const cudaNearRejected = buildPopulationAlignedSemanticExpectedRecord({
  raw: oneRecordRaw(0, 0, 0.1),
  srcIndex: 0,
  buildConfig: { timestamp: 0, scalingModifier: 1, sigmaScale: 1 },
  projectionParams: fixedProjectionParams()
});
assert.equal(cudaNearRejected.temporalEligible, true);
assert.equal(cudaNearRejected.rasterEligible, false);
assert.deepEqual(
  cudaNearRejected.stages.productionRasterEligibility.values,
  [0]
);

const cudaViewportEdgeEligible = buildPopulationAlignedSemanticExpectedRecord({
  raw: oneRecordRaw((1280.1 - 639.5) / 500, 0, 1),
  srcIndex: 0,
  buildConfig: { timestamp: 0, scalingModifier: 1, sigmaScale: 1 },
  projectionParams: fixedProjectionParams()
});
assert.equal(cudaViewportEdgeEligible.rasterEligible, true);
assert.ok(cudaViewportEdgeEligible.stages.projectedCenter.values[0] >= 1280);

const routedAlphaRaw = oneRecordRaw(0, 0, 1);
routedAlphaRaw.opacity[0] = centralInput.rawOpacityLogit;
routedAlphaRaw.scale_xyz.set(centralInput.sourceScaleXYZ);
routedAlphaRaw.scale_t[0] = centralInput.sourceScaleT;
routedAlphaRaw.rotation.set(centralInput.rotation);
routedAlphaRaw.rotation_r.set(centralInput.rotationR);
routedAlphaRaw.t[0] = centralInput.tCenter;
const routedAlphaExpected = buildPopulationAlignedSemanticExpectedRecord({
  raw: routedAlphaRaw,
  srcIndex: 0,
  buildConfig: {
    timestamp: centralInput.timestamp,
    scalingModifier: centralInput.scalingModifier,
    sigmaScale: centralInput.sigmaScale
  },
  projectionParams: fixedProjectionParams()
});
assert.deepEqual(
  routedAlphaExpected.stages.productionTileInputAlpha.values,
  [buildProductionTileInputAlphaF32Central(centralInput).alpha]
);
assert.deepEqual(
  routedAlphaExpected.stages.productionTileInputRgb.values,
  buildCudaDegree2TileInputExpectedRgb({
    raw: routedAlphaRaw,
    srcIndex: 0,
    cameraWorldPosition: [0, 0, 0]
  })
);

const ineligible = compareFixture({
  count: 2,
  records: [expectedRecord(0, { eligible: false }), expectedRecord(1, { eligible: false })]
});
assert.equal(ineligible.decision, 'match');
assert.equal(ineligible.evidenceComplete, true);
assert.equal(ineligible.firstMismatches.length, 0);
assert.equal(
  ineligible.stageSummaries.find((stage) => stage.stage === 'temporalEligibility')
    .classification,
  'match'
);
assert.equal(
  ineligible.stageSummaries.find(
    (stage) => stage.stage === 'productionRasterEligibility'
  ).classification,
  'match'
);
assert.equal(
  ineligible.stageSummaries.find((stage) => stage.stage === 'conditionalStatePosition').classification,
  'not-applicable'
);
for (const stageName of [
  'conditionalStatePosition',
  'conditionalWorldCovariance',
  'cameraSpaceCovariance',
  'projectionJacobian',
  'screenCovariance',
  'conic',
  'radius',
  'projectedCenter',
  'cameraDepth',
  'webgpuInclusivePixelBounds',
  'normalizedInclusiveTileBounds',
  'productionTileInputAlpha',
  'productionTileInputRgb'
]) {
  const stage = ineligible.stageSummaries.find((entry) => entry.stage === stageName);
  assert.equal(stage.notApplicableCount, 2);
  assert.equal(stage.validCount, 0);
  assert.equal(stage.missingCount, 0);
  assert.equal(stage.invalidCount, 0);
  assert.equal(stage.classification, 'not-applicable');
  assert.equal(stage.maxAbsoluteError, null);
}

const partiallyApplicableRecords = [
  expectedRecord(0, { eligible: false }),
  expectedRecord(1),
  expectedRecord(2, { eligible: false }),
  expectedRecord(3)
];
const partiallyApplicableMatch = compareFixture({
  count: partiallyApplicableRecords.length,
  records: partiallyApplicableRecords
});
const partiallyApplicablePosition = partiallyApplicableMatch.stageSummaries.find(
  (stage) => stage.stage === 'conditionalStatePosition'
);
assert.equal(partiallyApplicableMatch.decision, 'match');
assert.equal(partiallyApplicablePosition.validCount, 2);
assert.equal(partiallyApplicablePosition.notApplicableCount, 2);
assert.equal(partiallyApplicablePosition.classification, 'match');

const partiallyApplicableActual = packActual(partiallyApplicableRecords);
partiallyApplicableActual[
  3 * POPULATION_SEMANTIC_EVIDENCE_FLOAT_STRIDE + 2
] += 1;
const partiallyApplicableMismatch = comparePacked(
  partiallyApplicableActual,
  partiallyApplicableRecords
);
assert.equal(partiallyApplicableMismatch.decision, 'mismatch');
assert.deepEqual(
  partiallyApplicableMismatch.firstMismatches
    .filter((entry) => entry.stage === 'conditionalStatePosition')
    .map(({ localRow, stage, component }) => ({ localRow, stage, component })),
  [{ localRow: 3, stage: 'conditionalStatePosition', component: 'x' }]
);

function classifyFixture(overrides = {}) {
  return classifyPopulationSemanticStageEvidence({
    requiredRecordCount: 2,
    componentCount: 3,
    comparedCount: 0,
    comparedComponentCount: 0,
    validCount: 0,
    notApplicableCount: 2,
    missingCount: 0,
    invalidCount: 0,
    missingInvalidCount: 0,
    mismatchCount: 0,
    componentMismatchCount: 0,
    precisionAlignedCount: 0,
    precisionAlignedComponentCount: 0,
    semanticResidualCount: 0,
    semanticResidualComponentCount: 0,
    maxAbsoluteError: null,
    ...overrides
  });
}
assert.deepEqual(
  classifyFixture(),
  {
    classification: 'not-applicable',
    evidenceComplete: true,
    accountingComplete: true,
    blockedReasons: []
  }
);
assert.equal(classifyFixture({
  comparedCount: 2,
  comparedComponentCount: 6,
  validCount: 2,
  notApplicableCount: 0,
  mismatchCount: 1,
  componentMismatchCount: 1,
  precisionAlignedCount: 1,
  precisionAlignedComponentCount: 1,
  maxAbsoluteError: 1
}).classification, 'precision-aligned');
assert.equal(classifyFixture({
  comparedCount: 2,
  comparedComponentCount: 6,
  validCount: 2,
  notApplicableCount: 0,
  mismatchCount: 2,
  componentMismatchCount: 2,
  precisionAlignedCount: 1,
  precisionAlignedComponentCount: 1,
  semanticResidualCount: 1,
  semanticResidualComponentCount: 1,
  maxAbsoluteError: 1
}).classification, 'mismatch');
for (const [name, overrides] of [
  ['n-a-and-missing', {
    notApplicableCount: 1,
    missingCount: 1,
    missingInvalidCount: 1
  }],
  ['n-a-and-invalid', {
    notApplicableCount: 1,
    invalidCount: 1,
    missingInvalidCount: 1
  }],
  ['accounting-gap', { notApplicableCount: 1 }],
  ['n-a-count-excess', { notApplicableCount: 3 }],
  ['zero-valid-partial-n-a', { notApplicableCount: 0 }],
  ['n-a-with-mismatch', { mismatchCount: 1, componentMismatchCount: 1 }],
  ['precision-count-exceeds-raw', {
    precisionAlignedCount: 1,
    precisionAlignedComponentCount: 1
  }],
  ['precision-residual-accounting-drift', {
    comparedCount: 2,
    comparedComponentCount: 6,
    validCount: 2,
    notApplicableCount: 0,
    mismatchCount: 1,
    componentMismatchCount: 1,
    precisionAlignedCount: 1,
    precisionAlignedComponentCount: 1,
    semanticResidualCount: 1,
    semanticResidualComponentCount: 1,
    maxAbsoluteError: 1
  }],
  ['n-a-with-max-error', { maxAbsoluteError: 0 }],
  ['fractional-count', { notApplicableCount: 1.5 }],
  ['negative-count', { notApplicableCount: -1 }]
]) {
  const result = classifyFixture(overrides);
  assert.equal(result.evidenceComplete, false, name);
  assert.match(result.classification, /^blocked-/, name);
}

function comparePartitionedFixture(records, partitionSizes, mismatchGlobalRow) {
  const chunks = [];
  let rowOffset = 0;
  for (let chunkIndex = 0; chunkIndex < partitionSizes.length; chunkIndex += 1) {
    const rangeCount = partitionSizes[chunkIndex];
    const rangeStart = PRODUCTION_RESIDENT_RANGE_START + rowOffset;
    const chunkRecords = records.slice(rowOffset, rowOffset + rangeCount);
    const contract = inputContract(rangeCount, rangeStart);
    const actual = packActual(chunkRecords);
    if (
      mismatchGlobalRow >= rowOffset &&
      mismatchGlobalRow < rowOffset + rangeCount
    ) {
      const localRow = mismatchGlobalRow - rowOffset;
      actual[localRow * POPULATION_SEMANTIC_EVIDENCE_FLOAT_STRIDE + 2] += 1;
    }
    chunks.push(comparePopulationAlignedSemanticChunkEvidence({
      inputContract: contract,
      candidateIndices: buildExplicitPopulationChunkIndices(contract),
      actualPackedEvidence: actual,
      actualRasterCompanionEvidence: packRasterCompanion(chunkRecords),
      actualRasterCompanionContract:
        rasterCompanionContract(contract, chunkRecords),
      expectedRecordForRow: (row) => chunkRecords[row],
      chunkIndex
    }));
    rowOffset += rangeCount;
  }
  assert.equal(rowOffset, records.length);
  assert.ok(chunks.every((chunk) => chunk.decision !== 'blocked'));
  const stageSummaries = POPULATION_SEMANTIC_STAGE_CONTRACTS.map(
    (contract, stageIndex) => {
      const stages = chunks.map((chunk) => chunk.stageSummaries[stageIndex]);
      const aggregate = {
        stage: contract.key,
        validCount: stages.reduce((sum, stage) => sum + stage.validCount, 0),
        notApplicableCount: stages.reduce(
          (sum, stage) => sum + stage.notApplicableCount,
          0
        ),
        missingCount: stages.reduce((sum, stage) => sum + stage.missingCount, 0),
        invalidCount: stages.reduce((sum, stage) => sum + stage.invalidCount, 0),
        mismatchCount: stages.reduce((sum, stage) => sum + stage.mismatchCount, 0),
        precisionAlignedCount: stages.reduce(
          (sum, stage) => sum + stage.precisionAlignedCount,
          0
        ),
        precisionAlignedComponentCount: stages.reduce(
          (sum, stage) => sum + stage.precisionAlignedComponentCount,
          0
        ),
        semanticResidualCount: stages.reduce(
          (sum, stage) => sum + stage.semanticResidualCount,
          0
        ),
        semanticResidualComponentCount: stages.reduce(
          (sum, stage) => sum + stage.semanticResidualComponentCount,
          0
        ),
        maxAbsoluteError: stages.reduce(
          (maximum, stage) => stage.maxAbsoluteError == null
            ? maximum
            : Math.max(maximum ?? 0, stage.maxAbsoluteError),
          null
        )
      };
      const componentMismatchCount = stages.reduce(
        (sum, stage) => sum + stage.componentMismatchCount,
        0
      );
      aggregate.classification = classifyPopulationSemanticStageEvidence({
        requiredRecordCount: records.length,
        componentCount: contract.components.length,
        comparedCount: aggregate.validCount,
        comparedComponentCount:
          aggregate.validCount * contract.components.length,
        ...aggregate,
        missingInvalidCount: aggregate.missingCount + aggregate.invalidCount,
        componentMismatchCount
      }).classification;
      return aggregate;
    }
  );
  const firstMismatches = chunks.flatMap((chunk) => chunk.firstMismatches).map(
    ({ srcIndex, stage, component }) => ({ srcIndex, stage, component })
  );
  return {
    decision: stageSummaries.some((stage) => stage.semanticResidualCount > 0)
      ? 'mismatch'
      : 'match',
    coverage: {
      processedCount: records.length,
      firstSrcIndex: PRODUCTION_RESIDENT_RANGE_START,
      lastSrcIndex: PRODUCTION_RESIDENT_RANGE_START + records.length - 1,
      coverageComplete: true
    },
    stageSummaries,
    firstMismatches
  };
}

const partitionedNaOnlyFirst = comparePartitionedFixture(
  partiallyApplicableRecords,
  [2, 2],
  3
);
const partitionedMixedFirst = comparePartitionedFixture(
  partiallyApplicableRecords,
  [3, 1],
  3
);
assert.deepEqual(partitionedNaOnlyFirst, partitionedMixedFirst);

const invalidOnlyRecords = [expectedRecord(0)];
const invalidOnlyActual = packActual(invalidOnlyRecords);
invalidOnlyActual[2] += 100;
invalidOnlyActual[4] = Number.NaN;
const invalidOnlyResult = comparePacked(invalidOnlyActual, invalidOnlyRecords);
const invalidOnlyPositionSummary = invalidOnlyResult.stageSummaries.find(
  (stage) => stage.stage === 'conditionalStatePosition'
);
assert.equal(invalidOnlyPositionSummary.comparedCount, 0);
assert.equal(invalidOnlyPositionSummary.validCount, 0);
assert.equal(invalidOnlyPositionSummary.invalidCount, 1);
assert.equal(invalidOnlyPositionSummary.comparedComponentCount, 0);
assert.equal(invalidOnlyPositionSummary.mismatchCount, 0);
assert.equal(invalidOnlyPositionSummary.componentMismatchCount, 0);
assert.equal(invalidOnlyPositionSummary.maxAbsoluteError, null);
assert.equal(
  invalidOnlyResult.firstMismatches.some(
    (entry) => entry.stage === 'conditionalStatePosition'
  ),
  false
);
assert.equal(invalidOnlyResult.decision, 'blocked');

const maxErrorRecords = [expectedRecord(0), expectedRecord(1)];
const maxErrorActual = packActual(maxErrorRecords);
maxErrorActual[2] += 0.5;
maxErrorActual[POPULATION_SEMANTIC_EVIDENCE_FLOAT_STRIDE + 2] += 100;
maxErrorActual[POPULATION_SEMANTIC_EVIDENCE_FLOAT_STRIDE + 4] = Number.NaN;
const maxErrorResult = comparePacked(maxErrorActual, maxErrorRecords);
const positionSummary = maxErrorResult.stageSummaries.find(
  (stage) => stage.stage === 'conditionalStatePosition'
);
assert.equal(positionSummary.maxAbsoluteError, 0.5);
assert.equal(positionSummary.comparedCount, 1);
assert.equal(positionSummary.validCount, 1);
assert.equal(positionSummary.invalidCount, 1);
assert.equal(positionSummary.comparedComponentCount, 3);
assert.equal(positionSummary.mismatchCount, 1);
assert.equal(positionSummary.componentMismatchCount, 1);
assert.deepEqual(
  maxErrorResult.firstMismatches
    .filter((entry) => entry.stage === 'conditionalStatePosition')
    .map(({ localRow, component, absoluteError }) => ({
      localRow,
      component,
      absoluteError
    })),
  [{ localRow: 0, component: 'x', absoluteError: 0.5 }]
);
assert.equal(maxErrorResult.decision, 'blocked');

for (const key of ['statePositions', 'renderAttributes', 'footprintPayload', 'gpuResources']) {
  assert.equal(Object.hasOwn(match, key), false);
}
const smallSize = JSON.stringify(compareFixture({ count: 1 })).length;
const largeSize = JSON.stringify(compareFixture({ count: 65536 })).length;
const boundedMismatchSize = JSON.stringify(boundedMismatch).length;
assert.ok(largeSize < smallSize * 1.25, `${smallSize} -> ${largeSize}`);

const producerSource = fs.readFileSync(
  new URL('../demo/js/webgpu_population_aligned_semantic_comparison.js', import.meta.url),
  'utf8'
);
const expectedFunctionSource = producerSource.match(
  /export function buildPopulationAlignedSemanticExpectedRecord\([\s\S]*?\n}\n\nexport function buildExplicitPopulationChunkIndices/
)?.[0] ?? '';
assert.ok(expectedFunctionSource.includes('computeCudaConditionalGaussianState4D'));
assert.ok(expectedFunctionSource.includes('buildProductionTileInputAlphaF32Central'));
assert.doesNotMatch(
  expectedFunctionSource,
  /const expectedAlpha\s*=\s*buildCudaTileInputExpectedAlpha/
);
assert.doesNotMatch(expectedFunctionSource, /actualPackedEvidence|actualIntermediate|readback/);
const alphaOracleSource = fs.readFileSync(
  new URL('../demo/js/common_4dgs_tile_input_alpha_f32_semantic.js', import.meta.url),
  'utf8'
);
assert.doesNotMatch(alphaOracleSource, /actualAlpha|actualTemporalWeight|readback/);
assert.doesNotMatch(producerSource, /canonical.*prepend|fraction sampling|deduplicated CPU fallback/i);
assert.match(producerSource, /populationSemanticDiagnostic: true/);
assert.match(producerSource, /readbackPolicy: 'diagnostic'/);
assert.match(producerSource, /readbackPolicy: 'none'/);
assert.match(producerSource, /buildNativeWebGpuProductionTileInput/);
assert.match(producerSource, /observePopulationRasterSemanticCompanion/);
assert.match(producerSource, /finally \{/);

const boundsSource = fs.readFileSync(
  new URL('../demo/js/common_4dgs_bounds_contracts.js', import.meta.url),
  'utf8'
);
const tileListSource = fs.readFileSync(
  new URL('../demo/js/webgpu_gpu_owned_tile_list_layout.js', import.meta.url),
  'utf8'
);
const rasterObserverSource = fs.readFileSync(
  new URL('../demo/js/webgpu_population_raster_semantic_observer.js', import.meta.url),
  'utf8'
);
const nativeTileInputSource = fs.readFileSync(
  new URL('../demo/js/webgpu_production_tile_input.js', import.meta.url),
  'utf8'
);
assert.match(boundsSource, /buildWebGpuProductionInclusiveBoundsWgslHelper/);
assert.match(boundsSource, /floor\(centerRadius\.xy - vec2f\(centerRadius\.z\)\)/);
assert.match(boundsSource, /ceil\(centerRadius\.xy \+ vec2f\(centerRadius\.z\)\)/);
assert.match(boundsSource, /struct ProductionInclusiveTileBounds/);
assert.match(boundsSource, /productionCudaAlignedInclusiveTileBounds/);
assert.match(
  boundsSource,
  /i32\(\(centerRadius\.x - centerRadius\.z\) \/ f32\(tileSize\)\)/
);
assert.match(
  boundsSource,
  /\(\(\(centerRadius\.x \+ centerRadius\.z\) \+ f32\(tileSize\)\) - 1\.0\)/
);
assert.match(boundsSource, /let nonEmpty = all\(maxExclusive > minInclusive\)/);
assert.match(boundsSource, /maxInclusive = maxExclusive - vec2u\(1u\)/);
assert.doesNotMatch(boundsSource, /floor\(pixelBounds\.zw \/ f32\(tileSize\)\)/);
assert.match(tileListSource, /buildWebGpuProductionInclusiveBoundsWgslHelper/);
assert.match(rasterObserverSource, /buildWebGpuProductionInclusiveBoundsWgslHelper/);
assert.equal(
  Array.from(tileListSource.matchAll(/if \(bounds\.nonEmpty == 0u\) \{ return; \}/g))
    .length,
  2
);
assert.match(tileListSource, /ty <= bounds\.maxInclusive\.y/);
assert.match(tileListSource, /tx <= bounds\.maxInclusive\.x/);
assert.doesNotMatch(tileListSource, /productionInclusivePixelBounds\(/);
assert.match(
  rasterObserverSource,
  /productionCudaAlignedInclusiveTileBounds\(\s*centerRadiusDepth,/
);
assert.match(rasterObserverSource, /if \(tileBounds\.nonEmpty == 0u\)/);
assert.doesNotMatch(
  tileListSource,
  /let minimum = clamp\(floor\(a\.xy - vec2f\(a\.z\)\)/
);
assert.match(
  nativeTileInputSource.match(
    /label: 'phase3-native-production-tile-input-resource',[\s\S]*?\n  \}\);/
  )?.[0] ?? '',
  /usage: GPUBufferUsage\.STORAGE\s*\n/
);
assert.doesNotMatch(
  nativeTileInputSource.match(
    /label: 'phase3-native-production-tile-input-resource',[\s\S]*?\n  \}\);/
  )?.[0] ?? '',
  /COPY_SRC/
);
assert.match(
  rasterObserverSource,
  /GPUBufferUsage\.STORAGE \| GPUBufferUsage\.COPY_SRC/
);
assert.match(
  rasterObserverSource,
  /GPUBufferUsage\.COPY_DST \| GPUBufferUsage\.MAP_READ/
);

const previousGpuBufferUsage = globalThis.GPUBufferUsage;
const previousGpuMapMode = globalThis.GPUMapMode;
globalThis.GPUBufferUsage = {
  STORAGE: 1,
  UNIFORM: 2,
  COPY_SRC: 4,
  COPY_DST: 8,
  MAP_READ: 16
};
globalThis.GPUMapMode = { READ: 1 };

function createFakeDevice({ failMap = false } = {}) {
  const buffers = [];
  const bindGroups = [];
  let shaderSource = null;
  const device = {
    limits: {
      maxStorageBufferBindingSize: 1_073_741_824,
      maxBufferSize: 1_073_741_824,
      maxStorageBuffersPerShaderStage: 8
    },
    createBuffer: (descriptor) => {
      const bytes = new ArrayBuffer(descriptor.size);
      const buffer = {
        descriptor,
        destroyed: false,
        getMappedRange: () => bytes,
        unmap: () => {},
        mapAsync: async () => {
          if (failMap) throw new Error('intentional-map-failure');
        },
        destroy() { this.destroyed = true; }
      };
      buffers.push(buffer);
      return buffer;
    },
    createShaderModule: ({ code }) => {
      shaderSource = code;
      return { code };
    },
    createComputePipeline: () => ({
      getBindGroupLayout: () => ({ label: 'step121-layout' })
    }),
    createBindGroup: (descriptor) => {
      bindGroups.push(descriptor);
      return { descriptor };
    },
    createCommandEncoder: () => ({
      beginComputePass: () => ({
        setPipeline: () => {},
        setBindGroup: () => {},
        dispatchWorkgroups: () => {},
        end: () => {}
      }),
      copyBufferToBuffer: () => {},
      finish: () => ({})
    }),
    queue: { submit: () => {}, onSubmittedWorkDone: async () => {} }
  };
  return { device, buffers, bindGroups, getShaderSource: () => shaderSource };
}

const evaluatorRaw = {
  N: 1,
  activeShDegree: 2,
  activeShDegreeT: 2,
  t: new Float32Array([0]), tDim: 1,
  scale_t: new Float32Array([1]), scaleTDim: 1,
  f_dc: new Float32Array([0, 0, 0]), fdcDim: 3,
  f_rest: new Float32Array(45), frestDim: 45,
  scale_xyz: new Float32Array([1, 1, 1]), scaleXYZDim: 3,
  rotation: new Float32Array([1, 0, 0, 0]), rotationDim: 4,
  rotation_r: new Float32Array([1, 0, 0, 0]), rotationRDim: 4
};
const evaluatorInput = {
  raw: evaluatorRaw,
  candidateIndices: new Uint32Array([0]),
  rawXyzOpacity: new Float32Array([0, 0, 1, 0]),
  buildConfig: { timestamp: 0, scalingModifier: 1, sigmaScale: 1 },
  projectionParams: new Float32Array(24)
};

try {
  const successFake = createFakeDevice();
  const diagnosticResult = await buildWebGpu4DStatePositionsForCandidates({
    device: successFake.device,
    ...evaluatorInput,
    populationSemanticDiagnostic: true
  });
  assert.equal(diagnosticResult.populationSemanticDiagnosticEnabled, true);
  assert.equal(
    diagnosticResult.populationSemanticDiagnosticLayout.rowStrideBytes,
    POPULATION_SEMANTIC_EVIDENCE_BYTES_PER_RECORD
  );
  assert.ok(successFake.buffers.every((buffer) => buffer.destroyed));
  assert.deepEqual(
    successFake.bindGroups[0].entries.map((entry) => entry.binding),
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
  );
  assert.match(successFake.getShaderSource(), /POPULATION_SEMANTIC_DIAGNOSTIC_ENABLED: bool = true/);

  const failureFake = createFakeDevice({ failMap: true });
  await assert.rejects(
    buildWebGpu4DStatePositionsForCandidates({
      device: failureFake.device,
      ...evaluatorInput,
      populationSemanticDiagnostic: true
    }),
    /intentional-map-failure/
  );
  assert.ok(failureFake.buffers.every((buffer) => buffer.destroyed));

  const productionFake = createFakeDevice();
  const productionResult = await buildWebGpu4DStatePositionsForCandidates({
    device: productionFake.device,
    ...evaluatorInput,
    readbackPolicy: 'none',
    keepGpuResources: true,
    populationSemanticDiagnostic: true
  });
  assert.equal(productionResult.productionReadbackPerformed, false);
  assert.equal(productionResult.gpuResources.footprintPayloadByteLength, (3 + 64) * 16);
  assert.match(productionFake.getShaderSource(), /POPULATION_SEMANTIC_DIAGNOSTIC_ENABLED: bool = false/);
  for (const buffer of [
    productionResult.gpuResources.statePositionBuffer,
    productionResult.gpuResources.renderAttributeBuffer,
    productionResult.gpuResources.footprintPayloadBuffer
  ]) buffer.destroy();

  const observerFake = createFakeDevice();
  const sourceTileInputBuffer = {
    destroyed: false,
    destroy() { this.destroyed = true; }
  };
  const observerResult = await observePopulationRasterSemanticCompanion({
    device: observerFake.device,
    tileInputResource: {
      buffer: sourceTileInputBuffer,
      recordCount: 1,
      resourceIdentity: 'tile-input-resource-fixture',
      sourceWorksetResourceIdentity:
        buildPopulationSemanticDiagnosticWorksetResourceIdentity(inputContract(1)),
      sourceStateResourceIdentity: 'state-resource-fixture'
    },
    expectedSourceWorksetResourceIdentity:
      buildPopulationSemanticDiagnosticWorksetResourceIdentity(inputContract(1)),
    canvasWidth: 1280,
    canvasHeight: 720,
    tileSize: 16
  });
  assert.equal(observerResult.evidence.length, 16);
  assert.equal(observerResult.contract.status, 'ready');
  assert.equal(observerResult.contract.rowStrideBytes, 64);
  assert.equal(sourceTileInputBuffer.destroyed, false);
  assert.ok(observerFake.buffers.every((buffer) => buffer.destroyed));
  const observerOutput = observerFake.buffers.find(
    (buffer) => buffer.descriptor.label ===
      'phase3-population-raster-semantic-companion-output'
  );
  const observerStaging = observerFake.buffers.find(
    (buffer) => buffer.descriptor.label ===
      'phase3-population-raster-semantic-companion-staging'
  );
  assert.equal(
    observerOutput.descriptor.usage,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  assert.equal(
    observerStaging.descriptor.usage,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  );
  assert.deepEqual(
    observerFake.bindGroups[0].entries.map((entry) => entry.binding),
    [0, 1, 2]
  );
  assert.match(
    observerFake.getShaderSource(),
    /productionInclusivePixelBounds/
  );
  assert.match(
    observerFake.getShaderSource(),
    /companionEvidence\[evidenceBase \+ 3u\] = colorAlpha;/
  );

  const observerFailureFake = createFakeDevice({ failMap: true });
  await assert.rejects(
    observePopulationRasterSemanticCompanion({
      device: observerFailureFake.device,
      tileInputResource: {
        buffer: sourceTileInputBuffer,
        recordCount: 1,
        resourceIdentity: 'tile-input-resource-fixture',
        sourceWorksetResourceIdentity:
          buildPopulationSemanticDiagnosticWorksetResourceIdentity(inputContract(1)),
        sourceStateResourceIdentity: 'state-resource-fixture'
      },
      expectedSourceWorksetResourceIdentity:
        buildPopulationSemanticDiagnosticWorksetResourceIdentity(inputContract(1)),
      canvasWidth: 1280,
      canvasHeight: 720,
      tileSize: 16
    }),
    /intentional-map-failure/
  );
  assert.ok(observerFailureFake.buffers.every((buffer) => buffer.destroyed));

  const observerBlockedFake = createFakeDevice();
  const observerBlocked = await observePopulationRasterSemanticCompanion({
    device: observerBlockedFake.device,
    tileInputResource: null,
    expectedSourceWorksetResourceIdentity: 'expected-workset',
    canvasWidth: 1280,
    canvasHeight: 720
  });
  assert.equal(observerBlocked.contract.status, 'blocked');
  assert.equal(observerBlocked.evidence.length, 0);
  assert.equal(observerBlockedFake.buffers.length, 0);
} finally {
  if (previousGpuBufferUsage === undefined) delete globalThis.GPUBufferUsage;
  else globalThis.GPUBufferUsage = previousGpuBufferUsage;
  if (previousGpuMapMode === undefined) delete globalThis.GPUMapMode;
  else globalThis.GPUMapMode = previousGpuMapMode;
}

console.log('Step121 Impl1 population-aligned single chunk smoke: OK', {
  cases: 52,
  evidenceBytesPerRecord: POPULATION_SEMANTIC_EVIDENCE_BYTES_PER_RECORD,
  maximumChunkRecords: 65536,
  maximumFirstMismatches: POPULATION_SEMANTIC_MAX_FIRST_MISMATCHES,
  smallResultBytes: smallSize,
  largeResultBytes: largeSize,
  boundedMismatchResultBytes: boundedMismatchSize
});
