import assert from 'node:assert/strict';
import {
  POPULATION_RASTER_SEMANTIC_ACTUAL_PROVENANCE,
  POPULATION_RASTER_SEMANTIC_COMPANION_BYTES_PER_RECORD,
  POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE,
  POPULATION_RASTER_SEMANTIC_COMPANION_LAYOUT_SCHEMA_VERSION,
  POPULATION_RASTER_SEMANTIC_COMPANION_VEC4_STRIDE,
  POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE,
  POPULATION_SEMANTIC_ACTUAL_PROVENANCE,
  POPULATION_SEMANTIC_COMPARISON_CONTRACT_NAME,
  POPULATION_SEMANTIC_COMPARISON_SCHEMA_VERSION,
  POPULATION_SEMANTIC_COMPLETE_STAGE_CLASSIFICATIONS,
  POPULATION_SEMANTIC_EXPECTED_PROVENANCE,
  POPULATION_SEMANTIC_LOGICAL_COMBINED_BYTES_PER_RECORD,
  POPULATION_SEMANTIC_LOGICAL_COMBINED_FLOAT_STRIDE,
  POPULATION_SEMANTIC_LOGICAL_COMBINED_VEC4_STRIDE,
  POPULATION_SEMANTIC_MAX_CHUNK_RECORDS,
  POPULATION_SEMANTIC_MAX_FIRST_MISMATCHES,
  POPULATION_PIXEL_BOUNDARY_PRECISION_CLASSIFICATION_PROVENANCE,
  POPULATION_SEMANTIC_STAGE_LOCAL_MAX_REPRESENTATIVE_RECORDS,
  POPULATION_SEMANTIC_STAGE_LOCAL_REPRESENTATIVE_LIMIT,
  POPULATION_SEMANTIC_STAGE_CONTRACTS,
  PRODUCTION_RESIDENT_RANGE_COUNT,
  PRODUCTION_RESIDENT_RANGE_END,
  PRODUCTION_RESIDENT_RANGE_START,
  buildPopulationRasterSemanticCompanionLayoutContract,
  buildPopulationSemanticComparisonInputContract,
  buildPopulationSemanticDiagnosticWorksetResourceIdentity,
  buildPopulationSemanticStageLocalMismatchRepresentative,
  buildPopulationSemanticStageLocalMismatchSummaries,
  classifyPopulationSemanticStageEvidence
} from '../demo/js/common_4dgs_population_semantic_comparison_contracts.js';
import {
  POPULATION_SEMANTIC_FIXED_CHUNK_COUNT,
  POPULATION_SEMANTIC_ORCHESTRATION_CONTRACT_NAME,
  POPULATION_SEMANTIC_ORCHESTRATION_SCHEMA_VERSION,
  buildPopulationAlignedSemanticComparisonChunkPlan,
  runPopulationAlignedSemanticComparisonResidentRange
} from '../demo/js/webgpu_population_aligned_semantic_comparison_orchestrator.js';

const device = { fixture: 'same-diagnostic-device' };
const RASTER_OBSERVER_RESOURCE_OWNERSHIP =
  'observer-call-scoped-destroyed-before-promise-resolution';
const baseInput = {
  device,
  raw: { fixture: 'population-aligned-spl4' },
  buildConfig: { timestamp: 23.2, scalingModifier: 1, sigmaScale: 1 },
  projectionParams: new Float32Array(24),
  sceneInputIdentity: { scene: 'fixture', revision: 1 },
  spl4InputIdentity: { spl4: 'fixture', hash: 'abc' },
  populationContractIdentity: {
    selection: 'explicit-resident-range',
    start: PRODUCTION_RESIDENT_RANGE_START,
    count: PRODUCTION_RESIDENT_RANGE_COUNT
  },
  cameraIdentity: { camera: '000151_v13' },
  projectionIdentity: { projection: 'fixed-reference' },
  timeIdentity: { requested: 23.2, actual: 23.2 }
};

function mismatchEntry(args, localRow, {
  stage = 'conditionalStatePosition',
  component = 'x'
} = {}) {
  const contract = POPULATION_SEMANTIC_STAGE_CONTRACTS.find(
    (candidate) => candidate.key === stage
  );
  const globalResidentRow =
    args.chunkIndex * POPULATION_SEMANTIC_MAX_CHUNK_RECORDS + localRow;
  const rasterStage = Object.hasOwn(
    POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE,
    stage
  );
  return {
    chunkIndex: args.chunkIndex,
    localRow,
    globalResidentRow,
    srcIndex: PRODUCTION_RESIDENT_RANGE_START + globalResidentRow,
    stage,
    component,
    expected: 0,
    actual: 1,
    absoluteError: 1,
    tolerance: contract?.tolerance ?? 0,
    expectedProvenance: POPULATION_SEMANTIC_EXPECTED_PROVENANCE,
    actualProvenance: POPULATION_SEMANTIC_ACTUAL_PROVENANCE,
    expectedStageProvenance: rasterStage
      ? POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE[stage]
      : POPULATION_SEMANTIC_EXPECTED_PROVENANCE,
    actualStageProvenance: rasterStage
      ? POPULATION_RASTER_SEMANTIC_ACTUAL_PROVENANCE
      : POPULATION_SEMANTIC_ACTUAL_PROVENANCE
  };
}

function stageLocalFixtureRecord() {
  return {
    valid: true,
    temporalEligible: true,
    rasterEligible: true,
    stages: Object.fromEntries(
      POPULATION_SEMANTIC_STAGE_CONTRACTS.map((stage) => [
        stage.key,
        {
          valid: true,
          missing: false,
          values: Array.from(
            { length: stage.components.length },
            (_, componentIndex) => componentIndex
          )
        }
      ])
    )
  };
}

function stageLocalFixtureRepresentative(
  args,
  localRow,
  stage,
  componentIndex = 0,
  comparisonClassification = 'mismatch'
) {
  const contract = POPULATION_SEMANTIC_STAGE_CONTRACTS.find(
    (candidate) => candidate.key === stage
  );
  const expectedRecord = stageLocalFixtureRecord();
  const actualRecord = stageLocalFixtureRecord();
  actualRecord.stages[stage].values[componentIndex] += 1;
  return buildPopulationSemanticStageLocalMismatchRepresentative({
    chunkIndex: args.chunkIndex,
    localRow,
    srcIndex: args.rangeStart + localRow,
    stage,
    mismatchComponents: [{
      componentIndex,
      expectedValue: expectedRecord.stages[stage].values[componentIndex],
      actualValue: actualRecord.stages[stage].values[componentIndex],
      absoluteError: 1,
      mismatch: true
    }],
    expectedRecord,
    actualRecord,
    comparisonClassification,
    tolerance: contract.tolerance
  });
}

function makeChunkResult(args, {
  mismatchCount = 0,
  representatives = [],
  matchMaxError = 0,
  notApplicableStages = [],
  stageMismatchCounts = null,
  stagePrecisionAlignedCounts = null
} = {}) {
  const identity = buildPopulationSemanticComparisonInputContract({
    rangeStart: args.rangeStart,
    rangeCount: args.rangeCount,
    sceneInputIdentity: args.sceneInputIdentity,
    spl4InputIdentity: args.spl4InputIdentity,
    populationContractIdentity: args.populationContractIdentity,
    buildConfig: args.buildConfig,
    cameraIdentity: args.cameraIdentity,
    projectionIdentity: args.projectionIdentity,
    timeIdentity: args.timeIdentity
  });
  const expectedRasterFloatCount =
    args.rangeCount * POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE;
  const rasterCompanionLayout =
    buildPopulationRasterSemanticCompanionLayoutContract({
      recordCount: args.rangeCount,
      evidenceFloatCount: expectedRasterFloatCount,
      sourceWorksetResourceIdentity:
        buildPopulationSemanticDiagnosticWorksetResourceIdentity(identity),
      sourceStateResourceIdentity:
        `fixture-state-resource-${args.chunkIndex}`,
      sourceTileInputResourceIdentity:
        `fixture-tile-input-resource-${args.chunkIndex}`,
      canvasWidth: 1280,
      canvasHeight: 720,
      tileSize: 16,
      tileCols: 80,
      tileRows: 45,
      observerDispatchSubmitted: true,
      observerReadbackCompleted: true,
      observerOwnedBuffersDestroyed: true
    });
  const mismatchStage = representatives[0]?.stage ?? 'conditionalStatePosition';
  const mismatchCountsByStage = stageMismatchCounts ?? {
    [mismatchStage]: mismatchCount
  };
  const notApplicableStageSet = new Set(notApplicableStages);
  const stageSummaries = POPULATION_SEMANTIC_STAGE_CONTRACTS.map((contract) => {
    const notApplicable = notApplicableStageSet.has(contract.key);
    const stageMismatchCount = notApplicable
      ? 0
      : Number(mismatchCountsByStage[contract.key] ?? 0);
    const precisionAlignedCount = notApplicable
      ? 0
      : Number(stagePrecisionAlignedCounts?.[contract.key] ?? 0);
    const semanticResidualCount = stageMismatchCount - precisionAlignedCount;
    const summary = {
      stage: contract.key,
      components: [...contract.components],
      comparedCount: notApplicable ? 0 : args.rangeCount,
      comparedComponentCount:
        notApplicable ? 0 : args.rangeCount * contract.components.length,
      validCount: notApplicable ? 0 : args.rangeCount,
      notApplicableCount: notApplicable ? args.rangeCount : 0,
      missingCount: 0,
      invalidCount: 0,
      missingInvalidCount: 0,
      mismatchCount: stageMismatchCount,
      componentMismatchCount: stageMismatchCount,
      precisionAlignedCount,
      precisionAlignedComponentCount: precisionAlignedCount,
      semanticResidualCount,
      semanticResidualComponentCount: semanticResidualCount,
      maxAbsoluteError:
        notApplicable
          ? null
          : stageMismatchCount > 0
          ? 1 + args.chunkIndex
          : contract.key === 'conditionalStatePosition'
            ? matchMaxError
            : 0,
      tolerance: contract.tolerance
    };
    summary.classification = classifyPopulationSemanticStageEvidence({
      ...summary,
      requiredRecordCount: args.rangeCount,
      componentCount: contract.components.length
    }).classification;
    return summary;
  });
  const stageLocalRepresentatives = Object.fromEntries(
    POPULATION_SEMANTIC_STAGE_CONTRACTS.map((stage) => {
      const count = Number(mismatchCountsByStage[stage.key] ?? 0);
      const precisionCount = Number(stagePrecisionAlignedCounts?.[stage.key] ?? 0);
      const firstLegacyRepresentative = representatives.find(
        (entry) => entry.stage === stage.key
      );
      const firstLegacyRow = firstLegacyRepresentative?.localRow ?? 0;
      const componentIndex = Math.max(
        0,
        stage.components.indexOf(firstLegacyRepresentative?.component)
      );
      return [stage.key, Array.from(
        {
          length: Math.min(
            count,
            POPULATION_SEMANTIC_STAGE_LOCAL_REPRESENTATIVE_LIMIT
          )
        },
        (_, index) => stageLocalFixtureRepresentative(
          args,
          firstLegacyRow + index,
          stage.key,
          componentIndex,
          index < precisionCount ? 'precision-aligned' : 'mismatch'
        )
      )];
    })
  );
  const stageLocalMismatchSummaries =
    buildPopulationSemanticStageLocalMismatchSummaries({
      stageSummaries,
      representativesByStage: stageLocalRepresentatives
    });
  const anyMismatch = stageSummaries.some(
    (stage) => stage.semanticResidualCount > 0
  );
  const decision = anyMismatch ? 'mismatch' : 'match';
  return {
    schemaVersion: POPULATION_SEMANTIC_COMPARISON_SCHEMA_VERSION,
    contractName: POPULATION_SEMANTIC_COMPARISON_CONTRACT_NAME,
    decision,
    match: decision === 'match',
    reason: decision === 'mismatch' ? 'semantic-mismatch' : null,
    blockedReasons: [],
    identity,
    coverage: {
      requestedCount: args.rangeCount,
      processedCount: args.rangeCount,
      uniqueSrcIndexCount: args.rangeCount,
      firstSrcIndex: args.rangeStart,
      lastSrcIndex: args.rangeStart + args.rangeCount - 1,
      missingCount: 0,
      extraCount: 0,
      duplicateCount: 0,
      outOfRangeCount: 0,
      orderMismatchCount: 0,
      coverageComplete: true,
      requestedChunkCount: 1,
      completedChunkCount: 1
    },
    stageSummaries,
    stageLocalMismatchSummaries,
    firstMismatches: representatives,
    firstMismatchCount: representatives.length,
    firstMismatchLimit: POPULATION_SEMANTIC_MAX_FIRST_MISMATCHES,
    mismatchScanOrder: 'chunk-local-row-stage-component',
    evidenceComplete: true,
    expectedProvenance: POPULATION_SEMANTIC_EXPECTED_PROVENANCE,
    actualProvenance: POPULATION_SEMANTIC_ACTUAL_PROVENANCE,
    precisionClassificationProvenance:
      POPULATION_PIXEL_BOUNDARY_PRECISION_CLASSIFICATION_PROVENANCE,
    rasterExpectedProvenance: {
      ...POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE
    },
    rasterActualProvenance: POPULATION_RASTER_SEMANTIC_ACTUAL_PROVENANCE,
    rasterCompanionCoverage: {
      status: 'ready',
      reason: null,
      blockedReasons: [],
      requestedRecordCount: args.rangeCount,
      actualRecordCount: args.rangeCount,
      expectedFloatCount: expectedRasterFloatCount,
      actualFloatCount: expectedRasterFloatCount,
      evidenceLengthExact: true,
      rowAlignmentVerified: true
    },
    rasterCompanionEvidenceLayout: rasterCompanionLayout,
    diagnosticRasterCompanionLayout: {
      ...rasterCompanionLayout,
      vec4Layout: [...rasterCompanionLayout.vec4Layout]
    },
    actualEvidenceSameProductionDispatch: false,
    productionCalculationDependsOnDiagnosticReadback: false,
    rawRecordArraysIncluded: false,
    gpuResourcesIncluded: false,
    resultSizePopulationIndependent: true,
    productionBindingCount: 8,
    productionReadbackPolicyChanged: false,
    step113DiagnosticTailChanged: false,
    diagnosticGpuResourceOwnership:
      'evaluator-call-scoped-destroyed-before-promise-resolution',
    rasterObserverGpuResourceOwnership: RASTER_OBSERVER_RESOURCE_OWNERSHIP,
    nativeTileInputBufferUsageChanged: false
  };
}

function createRunner({ behavior = () => ({}), mutate = null } = {}) {
  const stats = {
    calls: [],
    devices: [],
    inputs: [],
    active: 0,
    maximumActive: 0
  };
  const runner = async (args) => {
    stats.calls.push(args.chunkIndex);
    stats.devices.push(args.device);
    stats.inputs.push(args);
    stats.active += 1;
    stats.maximumActive = Math.max(stats.maximumActive, stats.active);
    try {
      await Promise.resolve();
      const action = behavior(args) ?? {};
      if (action.throwError) throw action.throwError;
      const result = makeChunkResult(args, action);
      if (action.blocked) {
        result.decision = 'blocked';
        result.match = false;
        result.reason = action.blockedReason ?? 'fake-chunk-blocked';
        result.blockedReasons = [result.reason];
        result.evidenceComplete = false;
      }
      if (typeof mutate === 'function') mutate(result, args);
      return result;
    } finally {
      stats.active -= 1;
    }
  };
  return { runner, stats };
}

function mutateBothRasterLayouts(result, mutation) {
  mutation(result.rasterCompanionEvidenceLayout);
  mutation(result.diagnosticRasterCompanionLayout);
}

async function runWith(runner, overrides = {}) {
  return runPopulationAlignedSemanticComparisonResidentRange({
    ...baseInput,
    ...overrides,
    chunkRunner: runner
  });
}

const plan = buildPopulationAlignedSemanticComparisonChunkPlan();
assert.equal(
  POPULATION_SEMANTIC_ORCHESTRATION_SCHEMA_VERSION,
  'phase3-population-aligned-semantic-comparison-orchestration-v4'
);
assert.ok(
  POPULATION_SEMANTIC_COMPLETE_STAGE_CLASSIFICATIONS.includes(
    'precision-aligned'
  )
);
assert.equal(POPULATION_SEMANTIC_FIXED_CHUNK_COUNT, 8);
assert.equal(plan.length, 8);
for (let chunkIndex = 0; chunkIndex < plan.length; chunkIndex += 1) {
  const chunk = plan[chunkIndex];
  assert.deepEqual(chunk, {
    chunkIndex,
    rangeStart:
      PRODUCTION_RESIDENT_RANGE_START +
      chunkIndex * POPULATION_SEMANTIC_MAX_CHUNK_RECORDS,
    rangeCount: POPULATION_SEMANTIC_MAX_CHUNK_RECORDS,
    rangeEnd:
      PRODUCTION_RESIDENT_RANGE_START +
      (chunkIndex + 1) * POPULATION_SEMANTIC_MAX_CHUNK_RECORDS
  });
  if (chunkIndex > 0) assert.equal(plan[chunkIndex - 1].rangeEnd, chunk.rangeStart);
}
assert.equal(plan[0].rangeStart, 524288);
assert.equal(plan.at(-1).rangeEnd, 1048576);

const allMatchRunner = createRunner({
  behavior: (args) => ({ matchMaxError: args.chunkIndex * 1e-6 })
});
const allMatch = await runWith(allMatchRunner.runner);
assert.deepEqual(allMatchRunner.stats.calls, [0, 1, 2, 3, 4, 5, 6, 7]);
assert.equal(allMatchRunner.stats.maximumActive, 1);
assert.equal(allMatchRunner.stats.devices.every((value) => value === device), true);
assert.equal(allMatch.decision, 'match');
assert.equal(allMatch.match, true);
assert.equal(allMatch.coverage.completedChunkCount, 8);
assert.equal(allMatch.coverage.processedRecordCount, 524288);
assert.equal(allMatch.coverage.uniqueSrcIndexCount, 524288);
assert.equal(allMatch.coverage.coverageComplete, true);
assert.equal(allMatch.firstMismatches.length, 0);
assert.equal(allMatch.stageLocalMismatchSummaries.length, 13);
assert.ok(
  allMatch.stageLocalMismatchSummaries.every(
    (summary) =>
      summary.sourceMismatchRecordCount === 0 &&
      summary.sourceComponentMismatchCount === 0 &&
      summary.serializedRepresentativeRecordCount === 0 &&
      summary.representatives.length === 0 &&
      summary.truncated === false
  )
);
assert.deepEqual(
  allMatch.stageSummaries.map((stage) => stage.stage),
  POPULATION_SEMANTIC_STAGE_CONTRACTS.map((stage) => stage.key)
);
assert.equal(allMatch.rasterCompanionSummary.validatedChunkCount, 8);
assert.equal(allMatch.rasterCompanionSummary.allChunksReady, true);
assert.equal(
  allMatch.rasterCompanionSummary.allChunksRowAlignmentVerified,
  true
);
assert.equal(
  allMatch.rasterCompanionSummary.layoutContractConsistentAcrossChunks,
  true
);
assert.deepEqual(
  allMatch.rasterCompanionSummary.rasterExpectedProvenance,
  POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE
);
assert.equal(
  allMatch.rasterCompanionSummary.rasterActualProvenance,
  POPULATION_RASTER_SEMANTIC_ACTUAL_PROVENANCE
);
const allMatchPosition = allMatch.stageSummaries.find(
  (stage) => stage.stage === 'conditionalStatePosition'
);
assert.equal(allMatchPosition.comparedCount, 524288);
assert.equal(allMatchPosition.comparedComponentCount, 524288 * 3);
assert.equal(allMatchPosition.maxAbsoluteError, 7e-6);
assert.equal(new Set(allMatchRunner.stats.inputs.map((input) => input.buildConfig)).size, 8);
assert.equal(
  new Set(allMatchRunner.stats.inputs.map((input) => input.projectionParams)).size,
  8
);
assert.equal(
  new Set(allMatchRunner.stats.inputs.map((input) => input.spl4InputIdentity)).size,
  8
);
assert.equal(
  allMatchRunner.stats.inputs.every(
    (input) => input.projectionParams !== baseInput.projectionParams
  ),
  true
);

const eligibilityStageKeys = new Set([
  'temporalEligibility',
  'productionRasterEligibility'
]);
const dependentStageKeys = POPULATION_SEMANTIC_STAGE_CONTRACTS
  .map((stage) => stage.key)
  .filter((key) => !eligibilityStageKeys.has(key));
const neutralFirstChunkRunner = createRunner({
  behavior: (args) => args.chunkIndex === 0
    ? { notApplicableStages: dependentStageKeys }
    : {}
});
const neutralFirstChunk = await runWith(neutralFirstChunkRunner.runner);
assert.deepEqual(
  neutralFirstChunkRunner.stats.calls,
  [0, 1, 2, 3, 4, 5, 6, 7]
);
assert.equal(neutralFirstChunk.decision, 'match');
assert.equal(neutralFirstChunk.evidenceComplete, true);
assert.equal(neutralFirstChunk.coverage.completedChunkCount, 8);
assert.equal(neutralFirstChunk.coverage.processedRecordCount, 524288);
assert.equal(neutralFirstChunk.coverage.uniqueSrcIndexCount, 524288);
assert.equal(neutralFirstChunk.coverage.coverageComplete, true);
assert.equal(neutralFirstChunk.rasterCompanionSummary.validatedChunkCount, 8);
assert.equal(neutralFirstChunk.chunkSummaries[0].decision, 'match');
for (const stageKey of dependentStageKeys) {
  const aggregate = neutralFirstChunk.stageSummaries.find(
    (stage) => stage.stage === stageKey
  );
  assert.equal(
    aggregate.validCount,
    7 * POPULATION_SEMANTIC_MAX_CHUNK_RECORDS,
    stageKey
  );
  assert.equal(
    aggregate.notApplicableCount,
    POPULATION_SEMANTIC_MAX_CHUNK_RECORDS,
    stageKey
  );
  assert.equal(aggregate.classification, 'match', stageKey);
}

const fullyNotApplicableStageRunner = createRunner({
  behavior: () => ({
    notApplicableStages: ['conditionalStatePosition']
  })
});
const fullyNotApplicableStage = await runWith(
  fullyNotApplicableStageRunner.runner
);
const fullyNotApplicablePosition = fullyNotApplicableStage.stageSummaries.find(
  (stage) => stage.stage === 'conditionalStatePosition'
);
assert.equal(fullyNotApplicableStage.decision, 'match');
assert.equal(fullyNotApplicableStage.evidenceComplete, true);
assert.equal(fullyNotApplicablePosition.validCount, 0);
assert.equal(
  fullyNotApplicablePosition.notApplicableCount,
  PRODUCTION_RESIDENT_RANGE_COUNT
);
assert.equal(fullyNotApplicablePosition.maxAbsoluteError, null);
assert.equal(fullyNotApplicablePosition.classification, 'not-applicable');

const neutralThenMismatchRunner = createRunner({
  behavior: (args) => {
    if (args.chunkIndex === 0) {
      return { notApplicableStages: dependentStageKeys };
    }
    if (args.chunkIndex === 4) {
      return {
        mismatchCount: 1,
        representatives: [mismatchEntry(args, 3)]
      };
    }
    return {};
  }
});
const neutralThenMismatch = await runWith(neutralThenMismatchRunner.runner);
assert.equal(neutralThenMismatchRunner.stats.calls.length, 8);
assert.equal(neutralThenMismatch.coverage.coverageComplete, true);
assert.equal(neutralThenMismatch.decision, 'mismatch');
assert.deepEqual(
  neutralThenMismatch.firstMismatches.map(
    ({ srcIndex, stage, component }) => ({ srcIndex, stage, component })
  ),
  [{
    srcIndex:
      PRODUCTION_RESIDENT_RANGE_START +
      4 * POPULATION_SEMANTIC_MAX_CHUNK_RECORDS + 3,
    stage: 'conditionalStatePosition',
    component: 'x'
  }]
);

for (const [name, mutation] of [
  ['missing', (stage) => {
    stage.validCount -= 1;
    stage.comparedCount -= 1;
    stage.comparedComponentCount -= stage.components.length;
    stage.missingCount += 1;
    stage.missingInvalidCount += 1;
  }],
  ['invalid', (stage) => {
    stage.validCount -= 1;
    stage.comparedCount -= 1;
    stage.comparedComponentCount -= stage.components.length;
    stage.invalidCount += 1;
    stage.missingInvalidCount += 1;
  }]
]) {
  const incompleteRunner = createRunner({
    mutate: (result, args) => {
      if (args.chunkIndex !== 2) return;
      const stage = result.stageSummaries.find(
        (candidate) => candidate.stage === 'conditionalStatePosition'
      );
      mutation(stage);
      stage.classification = classifyPopulationSemanticStageEvidence({
        ...stage,
        requiredRecordCount: args.rangeCount,
        componentCount: stage.components.length
      }).classification;
    }
  });
  const incomplete = await runWith(incompleteRunner.runner);
  assert.equal(incomplete.decision, 'blocked', name);
  assert.equal(incompleteRunner.stats.calls.length, 3, name);
  assert.equal(incomplete.coverage.completedChunkCount, 2, name);
  assert.equal(incomplete.coverage.coverageComplete, false, name);
}

const callerBuildConfig = {
  timestamp: 23.2,
  scalingModifier: 1,
  sigmaScale: 1,
  prefilterVar: -1
};
const buildConfigMutationRunner = createRunner({
  behavior: (args) => {
    if (args.chunkIndex === 2) args.buildConfig.timestamp = 99;
    return {};
  }
});
const buildConfigMutation = await runWith(
  buildConfigMutationRunner.runner,
  { buildConfig: callerBuildConfig }
);
assert.equal(buildConfigMutation.decision, 'blocked');
assert.equal(buildConfigMutation.coverage.coverageComplete, false);
assert.equal(buildConfigMutation.coverage.completedChunkCount, 2);
assert.equal(buildConfigMutationRunner.stats.calls.length, 3);
assert.equal(callerBuildConfig.timestamp, 23.2);
assert.equal(
  buildConfigMutation.stageSummaries.find(
    (stage) => stage.stage === 'conditionalStatePosition'
  ).comparedCount,
  2 * POPULATION_SEMANTIC_MAX_CHUNK_RECORDS
);
assert.ok(
  buildConfigMutation.blockedReasons.some(
    (reason) => reason.includes('build-config-mutated')
  )
);

const callerProjectionParams = new Float32Array(baseInput.projectionParams);
const projectionMutationRunner = createRunner({
  behavior: (args) => {
    if (args.chunkIndex === 1) {
      args.projectionParams[8] = 800;
      args.projectionParams[9] = 900;
      args.projectionParams[12] = 12;
    }
    return {};
  }
});
const projectionMutation = await runWith(
  projectionMutationRunner.runner,
  { projectionParams: callerProjectionParams }
);
assert.equal(projectionMutation.decision, 'blocked');
assert.equal(projectionMutation.coverage.completedChunkCount, 1);
assert.equal(projectionMutationRunner.stats.calls.length, 2);
assert.equal(callerProjectionParams[8], 0);
assert.equal(callerProjectionParams[9], 0);
assert.equal(callerProjectionParams[12], 0);
assert.ok(
  projectionMutation.blockedReasons.some(
    (reason) => reason.includes('projection-params-mutated')
  )
);

const nestedIdentityCases = [
  {
    name: 'spl4',
    overrides: { spl4InputIdentity: { source: { hash: 'abc', parts: [1, 2] } } },
    mutate: (args) => { args.spl4InputIdentity.source.parts[1] = 9; },
    original: (overrides) => overrides.spl4InputIdentity.source.parts[1]
  },
  {
    name: 'population',
    overrides: {
      populationContractIdentity: {
        resident: { start: PRODUCTION_RESIDENT_RANGE_START, count: 524288 }
      }
    },
    mutate: (args) => { args.populationContractIdentity.resident.count = 1; },
    original: (overrides) => overrides.populationContractIdentity.resident.count
  },
  {
    name: 'camera-projection-time',
    overrides: {
      cameraIdentity: { fixed: { name: '000151_v13' } },
      projectionIdentity: { fixed: { name: 'reference' } },
      timeIdentity: { fixed: { requested: 23.2, actual: 23.2 } }
    },
    mutate: (args) => {
      args.cameraIdentity.fixed.name = 'drift';
      args.projectionIdentity.fixed.name = 'drift';
      args.timeIdentity.fixed.actual = 99;
    },
    original: (overrides) => [
      overrides.cameraIdentity.fixed.name,
      overrides.projectionIdentity.fixed.name,
      overrides.timeIdentity.fixed.actual
    ]
  }
];
for (const identityCase of nestedIdentityCases) {
  const identityMutationRunner = createRunner({
    behavior: (args) => {
      if (args.chunkIndex === 1) identityCase.mutate(args);
      return {};
    }
  });
  const before = JSON.stringify(identityCase.original(identityCase.overrides));
  const identityMutation = await runWith(
    identityMutationRunner.runner,
    identityCase.overrides
  );
  assert.equal(identityMutation.decision, 'blocked', identityCase.name);
  assert.equal(identityMutation.coverage.completedChunkCount, 1, identityCase.name);
  assert.equal(identityMutationRunner.stats.calls.length, 2, identityCase.name);
  assert.equal(
    JSON.stringify(identityCase.original(identityCase.overrides)),
    before,
    identityCase.name
  );
}

const callerOwnedInput = {
  buildConfig: {
    timestamp: 23.2,
    scalingModifier: 1,
    sigmaScale: 1,
    prefilterVar: -1
  },
  projectionParams: new Float32Array(24),
  sceneInputIdentity: { nested: { revision: 1 } },
  spl4InputIdentity: { nested: { hash: 'abc' } },
  populationContractIdentity: { nested: { count: 524288 } },
  cameraIdentity: { nested: { name: '000151_v13' } },
  projectionIdentity: { nested: { name: 'reference' } },
  timeIdentity: { nested: { actual: 23.2 } }
};
const callerMutationRunner = createRunner({
  behavior: (args) => {
    if (args.chunkIndex === 0) {
      callerOwnedInput.buildConfig.timestamp = 99;
      callerOwnedInput.projectionParams[8] = 999;
      callerOwnedInput.sceneInputIdentity.nested.revision = 9;
      callerOwnedInput.spl4InputIdentity.nested.hash = 'drift';
      callerOwnedInput.populationContractIdentity.nested.count = 1;
      callerOwnedInput.cameraIdentity.nested.name = 'drift';
      callerOwnedInput.projectionIdentity.nested.name = 'drift';
      callerOwnedInput.timeIdentity.nested.actual = 99;
    }
    return {};
  }
});
const callerMutation = await runWith(
  callerMutationRunner.runner,
  callerOwnedInput
);
assert.equal(callerMutation.decision, 'match');
assert.equal(callerMutationRunner.stats.calls.length, 8);
assert.equal(
  callerMutationRunner.stats.inputs.every(
    (input) =>
      input.buildConfig.timestamp === 23.2 &&
      input.projectionParams[8] === 0 &&
      input.sceneInputIdentity.nested.revision === 1 &&
      input.spl4InputIdentity.nested.hash === 'abc' &&
      input.populationContractIdentity.nested.count === 524288 &&
      input.cameraIdentity.nested.name === '000151_v13' &&
      input.projectionIdentity.nested.name === 'reference' &&
      input.timeIdentity.nested.actual === 23.2
  ),
  true
);
assert.equal(callerMutation.identity.buildConfigIdentity.timestamp, 23.2);
assert.deepEqual(callerMutation.identity.sceneInputIdentity, {
  nested: { revision: 1 }
});

const circularIdentity = { name: 'circular' };
circularIdentity.self = circularIdentity;
const snapshotFailureRunner = createRunner();
const snapshotFailure = await runWith(snapshotFailureRunner.runner, {
  spl4InputIdentity: circularIdentity
});
assert.equal(snapshotFailure.decision, 'blocked');
assert.equal(snapshotFailureRunner.stats.calls.length, 0);
assert.equal(snapshotFailure.coverage.coverageComplete, false);
assert.match(snapshotFailure.blockedReasons[0], /execution-snapshot-failed/);
assert.ok(snapshotFailure.blockedReasons[0].length < 220);

const reorderedIdentityRunner = createRunner({
  mutate: (result) => {
    result.identity.sceneInputIdentity = { revision: 1, scene: 'fixture' };
  }
});
const reorderedIdentity = await runWith(reorderedIdentityRunner.runner);
assert.equal(reorderedIdentity.decision, 'match');
assert.equal(reorderedIdentityRunner.stats.calls.length, 8);

const fullMismatchRunner = createRunner({
  behavior: (args) => {
    if (args.chunkIndex === 2) {
      return {
        mismatchCount: 2,
        representatives: [mismatchEntry(args, 5)]
      };
    }
    if (args.chunkIndex === 5) {
      return {
        mismatchCount: 3,
        representatives: [mismatchEntry(args, 7)]
      };
    }
    return {};
  }
});
const fullMismatch = await runWith(fullMismatchRunner.runner);
assert.equal(fullMismatchRunner.stats.calls.length, 8);
assert.equal(fullMismatch.coverage.coverageComplete, true);
assert.equal(fullMismatch.decision, 'mismatch');
assert.equal(
  fullMismatch.stageSummaries.find(
    (stage) => stage.stage === 'conditionalStatePosition'
  ).mismatchCount,
  5
);
assert.equal(
  fullMismatch.stageSummaries.find(
    (stage) => stage.stage === 'conditionalStatePosition'
  ).maxAbsoluteError,
  6
);
assert.equal(
  fullMismatch.firstMismatches[0].expectedStageProvenance,
  POPULATION_SEMANTIC_EXPECTED_PROVENANCE
);
assert.equal(
  fullMismatch.firstMismatches[0].actualStageProvenance,
  POPULATION_SEMANTIC_ACTUAL_PROVENANCE
);
const fullMismatchStageLocal = fullMismatch.stageLocalMismatchSummaries.find(
  (summary) => summary.stage === 'conditionalStatePosition'
);
assert.equal(fullMismatchStageLocal.sourceMismatchRecordCount, 5);
assert.equal(fullMismatchStageLocal.serializedRepresentativeRecordCount, 4);
assert.equal(fullMismatchStageLocal.truncated, true);
assert.deepEqual(
  fullMismatchStageLocal.representatives.map(
    ({ chunkIndex, localRow }) => [chunkIndex, localRow]
  ),
  [[2, 5], [2, 6], [5, 7], [5, 8]]
);

const rasterMismatchRunner = createRunner({
  behavior: (args) => args.chunkIndex === 4
    ? {
        mismatchCount: 1,
        representatives: [mismatchEntry(args, 9, {
          stage: 'webgpuInclusivePixelBounds',
          component: 'minX'
        })]
      }
    : {}
});
const rasterMismatch = await runWith(rasterMismatchRunner.runner);
assert.equal(rasterMismatch.decision, 'mismatch');
assert.equal(rasterMismatchRunner.stats.calls.length, 8);
assert.equal(rasterMismatch.firstMismatches.length, 1);
assert.equal(
  rasterMismatch.firstMismatches[0].expectedStageProvenance,
  POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE.webgpuInclusivePixelBounds
);
assert.equal(
  rasterMismatch.firstMismatches[0].actualStageProvenance,
  POPULATION_RASTER_SEMANTIC_ACTUAL_PROVENANCE
);

function precisionChunkBehavior(distribution) {
  return (args) => {
    const count = distribution[args.chunkIndex] ?? 0;
    return count > 0
      ? {
          stageMismatchCounts: { webgpuInclusivePixelBounds: count },
          stagePrecisionAlignedCounts: { webgpuInclusivePixelBounds: count }
        }
      : {};
  };
}

const distributedPrecisionRunner = createRunner({
  behavior: precisionChunkBehavior({ 0: 1, 2: 1, 4: 1, 6: 1 })
});
const distributedPrecision = await runWith(distributedPrecisionRunner.runner);
const concentratedPrecisionRunner = createRunner({
  behavior: precisionChunkBehavior({ 0: 4 })
});
const concentratedPrecision = await runWith(concentratedPrecisionRunner.runner);
for (const result of [distributedPrecision, concentratedPrecision]) {
  const pixel = result.stageSummaries.find(
    (stage) => stage.stage === 'webgpuInclusivePixelBounds'
  );
  const pixelLocal = result.stageLocalMismatchSummaries.find(
    (stage) => stage.stage === 'webgpuInclusivePixelBounds'
  );
  assert.equal(result.decision, 'match');
  assert.equal(result.firstMismatches.length, 0);
  assert.equal(pixel.mismatchCount, 4);
  assert.equal(pixel.componentMismatchCount, 4);
  assert.equal(pixel.precisionAlignedCount, 4);
  assert.equal(pixel.precisionAlignedComponentCount, 4);
  assert.equal(pixel.semanticResidualCount, 0);
  assert.equal(pixel.semanticResidualComponentCount, 0);
  assert.equal(pixel.classification, 'precision-aligned');
  assert.equal(pixelLocal.representatives.length, 4);
  assert.ok(pixelLocal.representatives.every(
    (representative) =>
      representative.comparisonClassification === 'precision-aligned'
  ));
}
for (const field of [
  'mismatchCount',
  'componentMismatchCount',
  'precisionAlignedCount',
  'precisionAlignedComponentCount',
  'semanticResidualCount',
  'semanticResidualComponentCount',
  'classification'
]) {
  const distributedPixel = distributedPrecision.stageSummaries.find(
    (stage) => stage.stage === 'webgpuInclusivePixelBounds'
  );
  const concentratedPixel = concentratedPrecision.stageSummaries.find(
    (stage) => stage.stage === 'webgpuInclusivePixelBounds'
  );
  assert.equal(distributedPixel[field], concentratedPixel[field], field);
}

const mixedPrecisionRunner = createRunner({
  behavior: (args) => {
    if (args.chunkIndex === 0) {
      return {
        stageMismatchCounts: { webgpuInclusivePixelBounds: 2 },
        stagePrecisionAlignedCounts: { webgpuInclusivePixelBounds: 2 }
      };
    }
    if (args.chunkIndex === 4) {
      return {
        stageMismatchCounts: { webgpuInclusivePixelBounds: 1 },
        representatives: [mismatchEntry(args, 9, {
          stage: 'webgpuInclusivePixelBounds',
          component: 'minX'
        })]
      };
    }
    return {};
  }
});
const mixedPrecision = await runWith(mixedPrecisionRunner.runner);
const mixedPixel = mixedPrecision.stageSummaries.find(
  (stage) => stage.stage === 'webgpuInclusivePixelBounds'
);
assert.equal(mixedPrecision.decision, 'mismatch');
assert.equal(mixedPrecision.firstMismatches.length, 1);
assert.equal(mixedPixel.mismatchCount, 3);
assert.equal(mixedPixel.precisionAlignedCount, 2);
assert.equal(mixedPixel.semanticResidualCount, 1);
assert.equal(mixedPixel.classification, 'mismatch');

const tileIsolationRunner = createRunner({
  behavior: (args) => args.chunkIndex === 3
    ? {
        stageMismatchCounts: {
          webgpuInclusivePixelBounds: 1,
          normalizedInclusiveTileBounds: 2
        },
        stagePrecisionAlignedCounts: { webgpuInclusivePixelBounds: 1 },
        representatives: [mismatchEntry(args, 5, {
          stage: 'normalizedInclusiveTileBounds',
          component: 'maxY'
        })]
      }
    : {}
});
const tileIsolation = await runWith(tileIsolationRunner.runner);
const isolatedPixel = tileIsolation.stageSummaries.find(
  (stage) => stage.stage === 'webgpuInclusivePixelBounds'
);
const isolatedTile = tileIsolation.stageSummaries.find(
  (stage) => stage.stage === 'normalizedInclusiveTileBounds'
);
assert.equal(tileIsolation.decision, 'mismatch');
assert.equal(isolatedPixel.precisionAlignedCount, 1);
assert.equal(isolatedPixel.semanticResidualCount, 0);
assert.equal(isolatedTile.mismatchCount, 2);
assert.equal(isolatedTile.precisionAlignedCount, 0);
assert.equal(isolatedTile.semanticResidualCount, 2);

const boundedMismatchRunner = createRunner({
  behavior: (args) => {
    if (args.chunkIndex > 1) return {};
    return {
      mismatchCount: 12,
      representatives: Array.from(
        { length: 12 },
        (_, localRow) => mismatchEntry(args, localRow)
      )
    };
  }
});
const boundedMismatch = await runWith(boundedMismatchRunner.runner);
assert.equal(boundedMismatch.decision, 'mismatch');
assert.equal(boundedMismatch.firstMismatches.length, 16);
assert.equal(boundedMismatch.firstMismatches[0].chunkIndex, 0);
assert.equal(boundedMismatch.firstMismatches[0].globalResidentRow, 0);
assert.equal(boundedMismatch.firstMismatches[0].srcIndex, 524288);
assert.equal(boundedMismatch.firstMismatches.at(-1).chunkIndex, 1);
assert.equal(boundedMismatch.firstMismatches.at(-1).localRow, 3);
assert.equal(
  boundedMismatch.stageSummaries.find(
    (stage) => stage.stage === 'conditionalStatePosition'
  ).mismatchCount,
  24
);

const starvationRunner = createRunner({
  behavior: (args) => {
    if (args.chunkIndex === 0) {
      return {
        mismatchCount: 20,
        representatives: Array.from({ length: 16 }, (_, localRow) =>
          mismatchEntry(args, localRow, {
            stage: 'normalizedInclusiveTileBounds',
            component: 'maxX'
          }))
      };
    }
    if (args.chunkIndex === 4) {
      return {
        mismatchCount: 4,
        representatives: Array.from({ length: 4 }, (_, index) =>
          mismatchEntry(args, 100 + index, {
            stage: 'webgpuInclusivePixelBounds',
            component: 'maxY'
          }))
      };
    }
    return {};
  }
});
const starvation = await runWith(starvationRunner.runner);
assert.equal(starvationRunner.stats.calls.length, 8);
assert.equal(starvation.coverage.processedRecordCount, 524288);
assert.equal(starvation.coverage.coverageComplete, true);
assert.equal(starvation.decision, 'mismatch');
assert.equal(starvation.firstMismatches.length, 16);
assert.ok(
  starvation.firstMismatches.every(
    (entry) =>
      entry.chunkIndex === 0 &&
      entry.stage === 'normalizedInclusiveTileBounds'
  )
);
const starvationPixel = starvation.stageLocalMismatchSummaries.find(
  (summary) => summary.stage === 'webgpuInclusivePixelBounds'
);
const starvationTile = starvation.stageLocalMismatchSummaries.find(
  (summary) => summary.stage === 'normalizedInclusiveTileBounds'
);
assert.equal(starvationPixel.sourceMismatchRecordCount, 4);
assert.equal(starvationPixel.serializedRepresentativeRecordCount, 4);
assert.equal(starvationPixel.truncated, false);
assert.deepEqual(
  starvationPixel.representatives.map(({ chunkIndex, localRow }) =>
    [chunkIndex, localRow]
  ),
  [[4, 100], [4, 101], [4, 102], [4, 103]]
);
assert.equal(starvationTile.sourceMismatchRecordCount, 20);
assert.equal(starvationTile.serializedRepresentativeRecordCount, 4);
assert.equal(starvationTile.truncated, true);
assert.deepEqual(
  starvationTile.representatives.map(({ chunkIndex, localRow }) =>
    [chunkIndex, localRow]
  ),
  [[0, 0], [0, 1], [0, 2], [0, 3]]
);
assert.deepEqual(
  starvationPixel.representatives[0]
    .dependencyContext.webgpuInclusivePixelBounds.actual,
  [0, 1, 2, 4]
);

const coverageDrifts = [
  ['requested-start', (result) => { result.identity.requestedRangeStart += 1; }],
  ['requested-end', (result) => { result.identity.requestedRangeEnd += 1; }],
  ['requested-count', (result) => { result.identity.requestedRangeCount -= 1; }],
  ['applied-end', (result) => { result.identity.appliedRangeEnd -= 1; }],
  ['applied-count', (result) => { result.identity.appliedRangeCount -= 1; }],
  ['gap', (result) => { result.identity.appliedRangeStart += 1; }],
  ['overlap', (result) => { result.identity.appliedRangeStart -= 1; }],
  ['first-src-index', (result) => { result.coverage.firstSrcIndex += 1; }],
  ['last-src-index', (result) => { result.coverage.lastSrcIndex -= 1; }],
  ['processed-count', (result) => { result.coverage.processedCount -= 1; }],
  ['unique-count', (result) => { result.coverage.uniqueSrcIndexCount -= 1; }],
  ['completed-count', (result) => { result.coverage.completedChunkCount = 0; }]
];
for (const [name, mutation] of coverageDrifts) {
  const driftRunner = createRunner({
    mutate: (result, args) => {
      if (args.chunkIndex === 2) mutation(result);
    }
  });
  const drift = await runWith(driftRunner.runner);
  assert.equal(drift.decision, 'blocked', name);
  assert.equal(drift.coverage.coverageComplete, false, name);
  assert.equal(driftRunner.stats.calls.length, 3, name);
}

const contractDrifts = [
  ['old-schema', (result) => {
    result.schemaVersion = 'phase3-population-aligned-semantic-comparison-v4';
  }],
  ['spl4', (result) => { result.identity.spl4InputIdentity.hash = 'drift'; }],
  ['build-config', (result) => { result.identity.buildConfigIdentity.timestamp = 9; }],
  ['projection', (result) => {
    result.identity.cameraProjectionTimeIdentity.projection = { projection: 'drift' };
  }],
  ['time', (result) => {
    result.identity.cameraProjectionTimeIdentity.time = { requested: 9, actual: 9 };
  }],
  ['expected-provenance', (result) => { result.expectedProvenance = 'drift'; }],
  ['provenance', (result) => { result.actualProvenance = 'drift'; }],
  ['precision-provenance', (result) => {
    result.precisionClassificationProvenance = {
      ...result.precisionClassificationProvenance,
      expectedEnvelope: 'drift'
    };
  }],
  ['same-dispatch', (result) => { result.actualEvidenceSameProductionDispatch = true; }],
  ['stage-key', (result) => { result.stageSummaries[0].stage = 'drift'; }],
  ['component-order', (result) => {
    result.stageSummaries[1].components.reverse();
  }],
  ['tolerance', (result) => { result.stageSummaries[1].tolerance = 1; }],
  ['precision-count', (result) => {
    result.stageSummaries[11].precisionAlignedCount = 1;
  }],
  ['precision-classification', (result) => {
    result.stageSummaries[11].classification = 'precision-aligned';
  }]
];
for (const [name, mutation] of contractDrifts) {
  const driftRunner = createRunner({
    mutate: (result, args) => {
      if (args.chunkIndex === 2) mutation(result);
    }
  });
  const drift = await runWith(driftRunner.runner);
  assert.equal(drift.decision, 'blocked', name);
  assert.equal(drift.coverage.coverageComplete, false, name);
  assert.equal(driftRunner.stats.calls.length, 3, name);
}

const companionCoverageDrifts = [
  ['missing', (result) => { delete result.rasterCompanionCoverage; }],
  ['status', (result) => { result.rasterCompanionCoverage.status = 'blocked'; }],
  ['blocked-reason', (result) => {
    result.rasterCompanionCoverage.blockedReasons = ['fixture-blocked'];
  }],
  ['requested-record-count', (result) => {
    result.rasterCompanionCoverage.requestedRecordCount -= 1;
  }],
  ['actual-record-count', (result) => {
    result.rasterCompanionCoverage.actualRecordCount -= 1;
  }],
  ['expected-float-count', (result) => {
    result.rasterCompanionCoverage.expectedFloatCount -= 1;
  }],
  ['actual-float-count-short', (result) => {
    result.rasterCompanionCoverage.actualFloatCount -= 1;
  }],
  ['actual-float-count-long', (result) => {
    result.rasterCompanionCoverage.actualFloatCount += 1;
  }],
  ['evidence-length', (result) => {
    result.rasterCompanionCoverage.evidenceLengthExact = false;
  }],
  ['row-alignment', (result) => {
    result.rasterCompanionCoverage.rowAlignmentVerified = false;
  }]
];
for (const [name, mutation] of companionCoverageDrifts) {
  const driftRunner = createRunner({
    mutate: (result, args) => {
      if (args.chunkIndex === 2) mutation(result);
    }
  });
  const drift = await runWith(driftRunner.runner);
  assert.equal(drift.decision, 'blocked', `companion-coverage-${name}`);
  assert.equal(drift.coverage.completedChunkCount, 2, name);
  assert.equal(drift.rasterCompanionSummary.validatedChunkCount, 2, name);
  assert.equal(driftRunner.stats.calls.length, 3, name);
}

const companionLayoutDrifts = [
  ['missing', (result) => { delete result.rasterCompanionEvidenceLayout; }],
  ['diagnostic-missing', (result) => {
    delete result.diagnosticRasterCompanionLayout;
  }],
  ['schema', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.schemaVersion = 'drift'; }
  )],
  ['contract-name', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.contractName = 'drift'; }
  )],
  ['status', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.status = 'blocked'; }
  )],
  ['record-count', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.recordCount -= 1; }
  )],
  ['row-stride-vec4', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.rowStrideVec4 += 1; }
  )],
  ['row-stride-floats', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.rowStrideFloats += 1; }
  )],
  ['row-stride-bytes', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.rowStrideBytes += 4; }
  )],
  ['combined-vec4', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.logicalCombinedRowStrideVec4 += 1; }
  )],
  ['combined-floats', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.logicalCombinedRowStrideFloats += 1; }
  )],
  ['combined-bytes', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.logicalCombinedRowStrideBytes += 4; }
  )],
  ['evidence-float-count', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.evidenceFloatCount -= 1; }
  )],
  ['evidence-byte-size', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.evidenceByteSize -= 4; }
  )],
  ['row-alignment', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.rowAlignment = 'drift'; }
  )],
  ['vec4-layout', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.vec4Layout[1] = 'drift'; }
  )],
  ['canvas-tile-dimensions', (result) => mutateBothRasterLayouts(
    result,
    (layout) => {
      layout.canvasWidth = 1296;
      layout.tileCols = 81;
    }
  )],
  ['workset-identity', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.sourceWorksetResourceIdentity = 'drift'; }
  )],
  ['state-identity', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.sourceStateResourceIdentity = ''; }
  )],
  ['tile-input-identity', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.sourceTileInputResourceIdentity = ''; }
  )],
  ['observer-dispatch', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.observerDispatchSubmitted = false; }
  )],
  ['observer-readback', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.observerReadbackCompleted = false; }
  )],
  ['observer-destroy', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.observerOwnedBuffersDestroyed = false; }
  )],
  ['native-usage', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.nativeTileInputBufferUsageChanged = true; }
  )],
  ['output-usage', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.observerOutputUsage = 'storage'; }
  )],
  ['staging-usage', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.observerStagingUsage = 'map-read'; }
  )],
  ['diagnostic-only', (result) => mutateBothRasterLayouts(
    result,
    (layout) => { layout.diagnosticOnly = false; }
  )],
  ['duplicated-layout-drift', (result) => {
    result.diagnosticRasterCompanionLayout.tileRows += 1;
  }]
];
for (const [name, mutation] of companionLayoutDrifts) {
  const driftRunner = createRunner({
    mutate: (result, args) => {
      if (args.chunkIndex === 2) mutation(result);
    }
  });
  const drift = await runWith(driftRunner.runner);
  assert.equal(drift.decision, 'blocked', `companion-layout-${name}`);
  assert.equal(drift.coverage.completedChunkCount, 2, name);
  assert.equal(drift.rasterCompanionSummary.validatedChunkCount, 2, name);
  assert.equal(driftRunner.stats.calls.length, 3, name);
}

const provenanceDrifts = [
  ['raster-map-missing', 'match', (result) => {
    delete result.rasterExpectedProvenance;
  }],
  ['raster-stage-key-missing', 'match', (result) => {
    delete result.rasterExpectedProvenance.projectedCenter;
  }],
  ['raster-stage-key-extra', 'match', (result) => {
    result.rasterExpectedProvenance.extraStage = 'drift';
  }],
  ['raster-stage-value', 'match', (result) => {
    result.rasterExpectedProvenance.cameraDepth = 'drift';
  }],
  ['raster-actual', 'match', (result) => {
    result.rasterActualProvenance = 'drift';
  }],
  ['raster-entry-expected-missing', 'raster', (result) => {
    delete result.firstMismatches[0].expectedStageProvenance;
  }],
  ['raster-entry-expected-wrong', 'raster', (result) => {
    result.firstMismatches[0].expectedStageProvenance =
      POPULATION_SEMANTIC_EXPECTED_PROVENANCE;
  }],
  ['raster-entry-actual-missing', 'raster', (result) => {
    delete result.firstMismatches[0].actualStageProvenance;
  }],
  ['raster-entry-actual-wrong', 'raster', (result) => {
    result.firstMismatches[0].actualStageProvenance =
      POPULATION_SEMANTIC_ACTUAL_PROVENANCE;
  }],
  ['legacy-entry-expected-missing', 'legacy', (result) => {
    delete result.firstMismatches[0].expectedStageProvenance;
  }],
  ['legacy-entry-expected-wrong', 'legacy', (result) => {
    result.firstMismatches[0].expectedStageProvenance =
      POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE.projectedCenter;
  }],
  ['legacy-entry-actual-missing', 'legacy', (result) => {
    delete result.firstMismatches[0].actualStageProvenance;
  }],
  ['legacy-entry-actual-wrong', 'legacy', (result) => {
    result.firstMismatches[0].actualStageProvenance =
      POPULATION_RASTER_SEMANTIC_ACTUAL_PROVENANCE;
  }]
];
for (const [name, mismatchKind, mutation] of provenanceDrifts) {
  const driftRunner = createRunner({
    behavior: (args) => {
      if (args.chunkIndex !== 2 || mismatchKind === 'match') return {};
      return {
        mismatchCount: 1,
        representatives: [mismatchEntry(
          args,
          4,
          mismatchKind === 'raster'
            ? { stage: 'projectedCenter', component: 'px' }
            : undefined
        )]
      };
    },
    mutate: (result, args) => {
      if (args.chunkIndex === 2) mutation(result);
    }
  });
  const drift = await runWith(driftRunner.runner);
  assert.equal(drift.decision, 'blocked', `provenance-${name}`);
  assert.equal(drift.coverage.completedChunkCount, 2, name);
  assert.equal(driftRunner.stats.calls.length, 3, name);
}

const ownershipAndRetentionDrifts = [
  ['legacy-observer-ownership', (result) => {
    result.diagnosticGpuResourceOwnership = 'drift';
  }],
  ['raster-observer-ownership', (result) => {
    result.rasterObserverGpuResourceOwnership = 'drift';
  }],
  ['native-tile-input-usage', (result) => {
    result.nativeTileInputBufferUsageChanged = true;
  }],
  ['production-readback-policy', (result) => {
    result.productionReadbackPolicyChanged = true;
  }],
  ['production-depends-on-readback', (result) => {
    result.productionCalculationDependsOnDiagnosticReadback = true;
  }],
  ['same-production-dispatch', (result) => {
    result.actualEvidenceSameProductionDispatch = true;
  }],
  ['production-binding-count', (result) => {
    result.productionBindingCount = 7;
  }],
  ['step113-tail', (result) => {
    result.step113DiagnosticTailChanged = true;
  }],
  ['raw-retention-flag', (result) => {
    result.rawRecordArraysIncluded = true;
  }],
  ['gpu-retention-flag', (result) => {
    result.gpuResourcesIncluded = true;
  }],
  ['raw-field', (result) => {
    result.actualRasterCompanionEvidence = [1, 2, 3];
  }],
  ['typed-array', (result) => {
    result.retainedPayload = new Float32Array(4);
  }],
  ['gpu-resource', (result) => {
    result.gpuResources = { buffer: {} };
  }],
  ['device', (result) => {
    result.device = device;
  }]
];
for (const [name, mutation] of ownershipAndRetentionDrifts) {
  const driftRunner = createRunner({
    mutate: (result, args) => {
      if (args.chunkIndex === 2) mutation(result);
    }
  });
  const drift = await runWith(driftRunner.runner);
  assert.equal(drift.decision, 'blocked', `ownership-${name}`);
  assert.equal(drift.coverage.completedChunkCount, 2, name);
  assert.equal(drift.rasterCompanionSummary.validatedChunkCount, 2, name);
  assert.equal(driftRunner.stats.calls.length, 3, name);
}

const blockedRunner = createRunner({
  behavior: (args) => {
    if (args.chunkIndex === 1) {
      return {
        mismatchCount: 1,
        representatives: [mismatchEntry(args, 2)]
      };
    }
    if (args.chunkIndex === 3) return { blocked: true };
    return {};
  }
});
const blocked = await runWith(blockedRunner.runner);
assert.equal(blocked.decision, 'blocked');
assert.equal(blocked.match, false);
assert.equal(blockedRunner.stats.calls.length, 4);
assert.equal(blocked.coverage.completedChunkCount, 3);
assert.equal(blocked.coverage.coverageComplete, false);

const exceptionRunner = createRunner({
  behavior: (args) => args.chunkIndex === 4
    ? { throwError: new Error(`intentional-${'x'.repeat(500)}`) }
    : {}
});
const exception = await runWith(exceptionRunner.runner);
assert.equal(exception.decision, 'blocked');
assert.equal(exceptionRunner.stats.calls.length, 5);
assert.equal(exception.coverage.completedChunkCount, 4);
assert.equal(exception.coverage.coverageComplete, false);
assert.match(exception.blockedReasons[0], /chunk-4-runner-exception/);
assert.ok(exception.blockedReasons[0].length < 220);

const representativeDrifts = [
  ['chunk', (entry) => { entry.chunkIndex += 1; }],
  ['row', (entry) => { entry.localRow = -1; }],
  ['global-row', (entry) => { entry.globalResidentRow += 1; }],
  ['src-index', (entry) => { entry.srcIndex += 1; }],
  ['stage', (entry) => { entry.stage = 'unknown'; }],
  ['component', (entry) => { entry.component = 'unknown'; }],
  ['tolerance', (entry) => { entry.tolerance = 1; }],
  ['expected-provenance', (entry) => { entry.expectedProvenance = 'drift'; }],
  ['actual-provenance', (entry) => { entry.actualProvenance = 'drift'; }],
  ['finite-value', (entry) => { entry.actual = Number.NaN; }],
  ['duplicate', (_entry, result, args) => {
    result.firstMismatches = [
      mismatchEntry(args, 4),
      mismatchEntry(args, 4)
    ];
    result.firstMismatchCount = 2;
  }],
  ['component-order', (_entry, result, args) => {
    result.firstMismatches = [
      mismatchEntry(args, 4, { component: 'y' }),
      mismatchEntry(args, 4, { component: 'x' })
    ];
    result.firstMismatchCount = 2;
  }],
  ['stage-order', (_entry, result, args) => {
    result.firstMismatches = [
      mismatchEntry(args, 4, { stage: 'conditionalWorldCovariance', component: 'xx' }),
      mismatchEntry(args, 4, { stage: 'conditionalStatePosition', component: 'x' })
    ];
    result.firstMismatchCount = 2;
  }]
];
for (const [name, mutation] of representativeDrifts) {
  const driftRunner = createRunner({
    behavior: (args) => args.chunkIndex === 2
      ? {
          mismatchCount: 1,
          representatives: [mismatchEntry(args, 4)]
        }
      : {},
    mutate: (result, args) => {
      if (args.chunkIndex === 2) {
        mutation(result.firstMismatches[0], result, args);
      }
    }
  });
  const drift = await runWith(driftRunner.runner);
  assert.equal(drift.decision, 'blocked', name);
  assert.equal(drift.coverage.coverageComplete, false, name);
  assert.equal(driftRunner.stats.calls.length, 3, name);
}

const stageLocalContractDrifts = [
  ['stage-order', (summaries) => { summaries[1].stage = summaries[0].stage; }],
  ['representative-missing', (summaries) => {
    summaries[1].representatives = [];
    summaries[1].serializedRepresentativeRecordCount = 0;
  }],
  ['representative-excess', (summaries) => {
    summaries[1].representatives.push(
      JSON.parse(JSON.stringify(summaries[1].representatives[0]))
    );
    summaries[1].serializedRepresentativeRecordCount += 1;
  }],
  ['row-src-index', (summaries) => {
    summaries[1].representatives[0].srcIndex += 1;
  }],
  ['component-identity', (summaries) => {
    summaries[1].representatives[0].mismatchComponents[0].component = 'invalid';
  }],
  ['absolute-error', (summaries) => {
    summaries[1].representatives[0].mismatchComponents[0].absoluteError = 2;
  }],
  ['tolerance', (summaries) => {
    summaries[1].representatives[0].mismatchComponents[0].tolerance = 1;
  }],
  ['context-length', (summaries) => {
    summaries[1].representatives[0]
      .dependencyContext.projectedCenter.actual.pop();
  }],
  ['comparison-classification', (summaries) => {
    summaries[1].representatives[0].comparisonClassification = 'invalid';
  }],
  ['precision-source-count', (summaries) => {
    summaries[1].sourcePrecisionAlignedRecordCount += 1;
  }],
  ['typed-array', (summaries) => {
    summaries[1].representatives[0]
      .dependencyContext.projectedCenter.actual = new Float32Array([1, 2]);
  }],
  ['raw-object', (summaries) => {
    summaries[1].representatives[0].raw = { retained: true };
  }],
  ['gpu-resource', (summaries) => {
    summaries[1].representatives[0].gpuResources = { retained: true };
  }],
  ['source-count', (summaries) => {
    summaries[1].sourceMismatchRecordCount += 1;
  }],
  ['truncated', (summaries) => { summaries[1].truncated = true; }]
];
for (const [name, mutation] of stageLocalContractDrifts) {
  const driftRunner = createRunner({
    behavior: (args) => args.chunkIndex === 2
      ? {
          mismatchCount: 1,
          representatives: [mismatchEntry(args, 4)]
        }
      : {},
    mutate: (result, args) => {
      if (args.chunkIndex === 2) mutation(result.stageLocalMismatchSummaries);
    }
  });
  const drift = await runWith(driftRunner.runner);
  assert.equal(drift.decision, 'blocked', name);
  assert.equal(drift.coverage.coverageComplete, false, name);
  assert.equal(driftRunner.stats.calls.length, 3, name);
}

assert.equal(boundedMismatch.chunkSummaries.length, 8);
assert.equal(boundedMismatch.firstMismatches.length, 16);
assert.equal(boundedMismatch.singleChunkResultsIncluded, false);
assert.equal(boundedMismatch.rawRecordArraysIncluded, false);
assert.equal(boundedMismatch.typedArraysIncluded, false);
assert.equal(boundedMismatch.gpuResourcesIncluded, false);
assert.equal(boundedMismatch.deviceIncluded, false);
assert.equal(Object.hasOwn(boundedMismatch, 'chunkResults'), false);
assert.equal(Object.hasOwn(boundedMismatch, 'srcIndices'), false);
assert.equal(Object.hasOwn(boundedMismatch, 'device'), false);
assert.equal(
  Object.hasOwn(boundedMismatch.rasterCompanionSummary, 'chunkLayouts'),
  false
);
assert.equal(
  Object.hasOwn(boundedMismatch.rasterCompanionSummary, 'evidence'),
  false
);
assert.equal(
  boundedMismatch.rasterCompanionSummary.schemaVersion,
  POPULATION_RASTER_SEMANTIC_COMPANION_LAYOUT_SCHEMA_VERSION
);
assert.equal(
  boundedMismatch.rasterCompanionSummary.rowStrideVec4,
  POPULATION_RASTER_SEMANTIC_COMPANION_VEC4_STRIDE
);
assert.equal(
  boundedMismatch.rasterCompanionSummary.rowStrideFloats,
  POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE
);
assert.equal(
  boundedMismatch.rasterCompanionSummary.rowStrideBytes,
  POPULATION_RASTER_SEMANTIC_COMPANION_BYTES_PER_RECORD
);
assert.equal(
  boundedMismatch.rasterCompanionSummary.logicalCombinedRowStrideVec4,
  POPULATION_SEMANTIC_LOGICAL_COMBINED_VEC4_STRIDE
);
assert.equal(
  boundedMismatch.rasterCompanionSummary.logicalCombinedRowStrideFloats,
  POPULATION_SEMANTIC_LOGICAL_COMBINED_FLOAT_STRIDE
);
assert.equal(
  boundedMismatch.rasterCompanionSummary.logicalCombinedRowStrideBytes,
  POPULATION_SEMANTIC_LOGICAL_COMBINED_BYTES_PER_RECORD
);
assert.equal(
  boundedMismatch.rasterCompanionSummary.rasterObserverGpuResourceOwnership,
  RASTER_OBSERVER_RESOURCE_OWNERSHIP
);
assert.equal(
  boundedMismatch.rasterCompanionSummary.nativeTileInputBufferUsageChanged,
  false
);
const fullResultBytes = JSON.stringify(boundedMismatch).length;
const worstCaseGlobalRepresentatives = [
  ...POPULATION_SEMANTIC_STAGE_CONTRACTS.map((stage) =>
    mismatchEntry(
      {
        chunkIndex: 0
      },
      0,
      { stage: stage.key, component: stage.components[0] }
    )
  ),
  ...POPULATION_SEMANTIC_STAGE_CONTRACTS.slice(0, 3).map((stage) =>
    mismatchEntry(
      {
        chunkIndex: 0
      },
      1,
      { stage: stage.key, component: stage.components[0] }
    )
  )
];
const worstCaseRunner = createRunner({
  behavior: (args) => args.chunkIndex === 0
    ? {
        representatives: worstCaseGlobalRepresentatives,
        stageMismatchCounts: Object.fromEntries(
          POPULATION_SEMANTIC_STAGE_CONTRACTS.map((stage) => [stage.key, 4])
        )
      }
    : {}
});
const worstCase = await runWith(worstCaseRunner.runner);
assert.equal(worstCase.decision, 'mismatch');
assert.equal(worstCaseRunner.stats.calls.length, 8);
assert.equal(worstCase.firstMismatches.length, 16);
assert.equal(
  worstCase.stageLocalMismatchSummaries.reduce(
    (sum, summary) => sum + summary.representatives.length,
    0
  ),
  POPULATION_SEMANTIC_STAGE_LOCAL_MAX_REPRESENTATIVE_RECORDS
);
assert.ok(
  worstCase.stageLocalMismatchSummaries.every(
    (summary) =>
      summary.serializedRepresentativeRecordCount === 4 &&
      summary.representativeRecordLimit === 4 &&
      summary.truncated === false
  )
);
assert.doesNotThrow(() => JSON.stringify(worstCase));
const worstCaseControllerEnvelope = {
  contractVersion: 'phase3-step121-population-semantic-comparison-controller-v1',
  invocationMode: 'explicit-one-shot-debug-api-request',
  explicitRequest: true,
  automaticExecution: false,
  freshDiagnosticDeviceAcquisition: { status: 'ready' },
  diagnosticDeviceCleanup: { status: 'destroyed' },
  decision: 'mismatch',
  match: false,
  reason: 'semantic-mismatch',
  blockedReasons: [],
  orchestrationResult: worstCase,
  rawObjectIncluded: false,
  rawArraysIncluded: false,
  typedArraysIncluded: false,
  gpuResourcesIncluded: false,
  adapterIncluded: false,
  deviceIncluded: false,
  resultSizePopulationIndependent: true,
  jsonSerializable: true,
  bounded: true
};
const worstCaseControllerBytes = JSON.stringify(worstCaseControllerEnvelope).length;
assert.ok(
  worstCaseControllerBytes < 100000,
  JSON.stringify({ worstCaseControllerBytes })
);
const oneChunkRunner = createRunner({
  behavior: (args) => args.chunkIndex === 0 ? { blocked: true } : {}
});
const oneChunk = await runWith(oneChunkRunner.runner);
const oneChunkResultBytes = JSON.stringify(oneChunk).length;
assert.equal(oneChunk.chunkSummaries.length, 1);
assert.ok(fullResultBytes < oneChunkResultBytes * 4, {
  oneChunkResultBytes,
  fullResultBytes
});
assert.ok(fullResultBytes < 50000);

for (const [name, override] of [
  ['device', { device: null }],
  ['raw', { raw: null }],
  ['projection', { projectionParams: null }],
  ['scene-identity', { sceneInputIdentity: null }],
  ['spl4-identity', { spl4InputIdentity: null }],
  ['population-identity', { populationContractIdentity: null }],
  ['camera-identity', { cameraIdentity: null }],
  ['projection-identity', { projectionIdentity: null }],
  ['time-identity', { timeIdentity: null }],
  ['build-config', { buildConfig: { timestamp: Number.NaN } }]
]) {
  const missingRunner = createRunner();
  const missing = await runWith(missingRunner.runner, override);
  assert.equal(missing.decision, 'blocked', name);
  assert.equal(missingRunner.stats.calls.length, 0, name);
  assert.equal(missing.coverage.coverageComplete, false, name);
}

assert.equal(allMatch.schemaVersion, POPULATION_SEMANTIC_ORCHESTRATION_SCHEMA_VERSION);
assert.equal(allMatch.contractName, POPULATION_SEMANTIC_ORCHESTRATION_CONTRACT_NAME);
assert.equal(allMatch.requestedChunkCount, 8);
assert.equal(allMatch.completedChunkCount, 8);
assert.equal(allMatch.evidenceComplete, true);
assert.equal(allMatch.actualEvidenceSameProductionDispatch, false);
assert.equal(allMatch.productionCalculationDependsOnDiagnosticReadback, false);
assert.equal(
  allMatch.diagnosticGpuResourceOwnership,
  'each-chunk-call-scoped-destroyed-before-next-chunk'
);
assert.equal(allMatch.diagnosticDeviceOwnership, 'caller-owned-reused-not-destroyed');

console.log('Step121 Impl2 population-aligned eight-chunk orchestration smoke: OK', {
  caseGroups: 28,
  snapshotFixCaseGroups: 6,
  coverageDriftCases: coverageDrifts.length,
  contractDriftCases: contractDrifts.length,
  companionCoverageDriftCases: companionCoverageDrifts.length,
  companionLayoutDriftCases: companionLayoutDrifts.length,
  provenanceDriftCases: provenanceDrifts.length,
  ownershipAndRetentionDriftCases: ownershipAndRetentionDrifts.length,
  representativeDriftCases: representativeDrifts.length,
  stageLocalContractDriftCases: stageLocalContractDrifts.length,
  chunkCount: plan.length,
  chunkSize: POPULATION_SEMANTIC_MAX_CHUNK_RECORDS,
  residentRecordCount: PRODUCTION_RESIDENT_RANGE_COUNT,
  maximumFirstMismatches: POPULATION_SEMANTIC_MAX_FIRST_MISMATCHES,
  maximumStageLocalRepresentativeRecords:
    POPULATION_SEMANTIC_STAGE_LOCAL_MAX_REPRESENTATIVE_RECORDS,
  oneChunkResultBytes,
  fullResultBytes,
  worstCaseControllerBytes
});
