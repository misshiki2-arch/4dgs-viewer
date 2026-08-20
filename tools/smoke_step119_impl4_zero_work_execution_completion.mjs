import assert from 'node:assert/strict';
import {
  buildProductionTileExecutionPlanContract,
  PRODUCTION_TILE_EXECUTION_COMPLETION_CONTRACT_VERSION,
  PRODUCTION_TILE_EXECUTION_PLAN_MAGIC,
  PRODUCTION_TILE_EXECUTION_PLAN_STATUS,
  PRODUCTION_TILE_EXECUTION_PLAN_WORD,
  PRODUCTION_TILE_EXECUTION_PLAN_WORD_COUNT,
  readProductionTileExecutionPlanObserver
} from '../demo/js/common_4dgs_production_tile_execution_plan_contracts.js';
import {
  buildProductionTileReferenceCapacityContract
} from '../demo/js/common_4dgs_production_tile_reference_contracts.js';
import {
  buildProductionGpuExecutionContract
} from '../demo/js/common_4dgs_production_gpu_execution_contracts.js';
import {
  buildNativeWebGpuProductionFrameDataPathContract,
  buildProductionResidentWorksetContract
} from '../demo/js/common_4dgs_production_frame_data_contracts.js';
import {
  buildWebGpuTileListCompositorContract
} from '../demo/js/common_4dgs_record_contracts.js';

const limits = {
  workgroupSize: 64,
  maxComputeWorkgroupsPerDimension: 65535,
  maxInvocationsPerSubmission: 4194240,
  maxRecordTileVisitsPerSubmission: 4194240,
  recordBatchSize: 1024,
  referenceBatchSize: 4096,
  compositorReferenceChunkSize: 64,
  executionLimitSource: 'step119-impl4-focused-fixture'
};

function makeStaticPlan({
  planIdentity = 7,
  recordCount = 100,
  tileCount = 3,
  referenceCapacity = 16
} = {}) {
  return buildProductionTileExecutionPlanContract({
    resourceIdentity: 'plan:7',
    sourceTileInputResourceIdentity: 'tile-input:7',
    planIdentity,
    tileCount,
    recordCount,
    referenceCapacity,
    scanStageCount: 2,
    maximumReferenceChunkCount: 1,
    gpuPlanBufferCreated: true,
    gpuTileTableCreated: true,
    gpuChunkTableCreated: true,
    gpuPlanGenerated: true,
    scatterConsumesPlan: true,
    sortConsumesPlan: true,
    compositorConsumesPlan: true,
    capacityOverflowFailClosed: true,
    silentDropAllowed: false
  });
}

function makeWords(overrides = {}) {
  const values = {
    magic: PRODUCTION_TILE_EXECUTION_PLAN_MAGIC,
    status: PRODUCTION_TILE_EXECUTION_PLAN_STATUS.ready,
    recordCount: 100,
    tileCount: 3,
    referenceCapacity: 16,
    requiredReferenceCount: 0,
    requiredPaddedReferenceCapacity: 0,
    maxReferencesPerTile: 0,
    totalReferenceChunkCount: 0,
    maxReferenceChunksPerTile: 0,
    scatteredReferenceCount: 0,
    sortedReferenceCount: 0,
    compositedReferenceCount: 0,
    overflowReferenceCount: 0,
    compactOffsetTableReady: 1,
    planIdentity: 7,
    ...overrides
  };
  const words = new Uint32Array(PRODUCTION_TILE_EXECUTION_PLAN_WORD_COUNT);
  for (const [name, index] of Object.entries(
    PRODUCTION_TILE_EXECUTION_PLAN_WORD
  )) {
    words[index] = values[name] ?? 0;
  }
  return words;
}

function makeCapacity(observer, overrides = {}) {
  return buildProductionTileReferenceCapacityContract({
    recordCount: observer.recordCount,
    tileCount: observer.tileCount,
    allocatedReferenceCapacity: observer.referenceCapacity,
    requiredReferenceCount: observer.requiredReferenceCount,
    requiredPaddedReferenceCapacity: observer.requiredPaddedReferenceCapacity,
    writtenReferenceCount: observer.scatteredReferenceCount,
    maxReferencesPerTile: observer.maxReferencesPerTile,
    compactOffsetsGenerated: observer.compactOffsetTableReady,
    executionPlanCompletionReady:
      observer.executionCompletionContract.executionCompletionReady,
    recordAndReferenceCapacitySeparated: true,
    capacityOverflowDetected: observer.capacityOverflowDetected,
    capacityOverflowFailClosed: observer.capacityOverflowFailClosed,
    silentDropAllowed: false,
    ...overrides
  });
}

function makeBounded(observer, allStagesCompleted = true) {
  return buildProductionGpuExecutionContract({
    limits,
    inputRecordCount: observer.recordCount,
    inputReferenceCount: observer.requiredReferenceCount,
    completedRecordCount: observer.recordCount,
    completedReferenceCount: observer.compositedReferenceCount,
    countSubmissionCount: 1,
    scatterSubmissionCount: 1,
    sortSeedSubmissionCount: observer.requiredReferenceCount > 0 ? 1 : 0,
    sortCompareSubmissionCount: observer.requiredReferenceCount > 1 ? 1 : 0,
    sortValidationSubmissionCount: observer.requiredReferenceCount > 0 ? 1 : 0,
    compositorChunkSubmissionCount: observer.requiredReferenceCount > 0 ? 1 : 0,
    maximumRecordsInSubmission: observer.recordCount,
    maximumReferencesInSubmission: observer.requiredPaddedReferenceCapacity,
    maximumReferencesPerPixelInvocation: Math.min(
      observer.maxReferencesPerTile,
      limits.compositorReferenceChunkSize
    ),
    gpuResourceLineageMaintained: allStagesCompleted,
    recordReferenceCapacitySeparated: true,
    silentDropAllowed: false,
    schedulerContinuationUsed: false,
    allStagesCompleted
  });
}

function makeCompositor(observer, bounded, overrides = {}) {
  const zero = observer.requiredReferenceCount === 0;
  return buildWebGpuTileListCompositorContract({
    tileCompositorReady: true,
    boundedExecutionContract: bounded,
    compositorPassSubmitted: true,
    compositorReadbackCompleted: true,
    compositorReadOffsetCountTable: true,
    compositorTraversedReferenceList: true,
    outputTextureCreated: true,
    outputTextureWritten: true,
    outputTextureReadbackMatchesSummary: true,
    processedTileCount: observer.tileCount,
    compositedTileCount: zero ? 0 : 2,
    nonEmptyCompositedTileCount: zero ? 0 : 2,
    compositedReferenceCount: observer.compositedReferenceCount,
    sourceTotalTileReferenceCount: observer.requiredReferenceCount,
    overflowCount: observer.overflowReferenceCount,
    realTileCompositorOutputReady: !zero,
    debugOutputBypassedForCompositor: !zero,
    gaussianAttributePayloadConsumed: !zero,
    footprintPayloadConsumed: !zero,
    orderedTileReferencesConsumed: !zero,
    depthOrderedAccumulationUsed: !zero,
    alphaAccumulationUsed: !zero,
    colorAccumulationUsed: !zero,
    tileCompositorContributionCount: zero ? 0 : observer.requiredReferenceCount,
    tileCompositorNonzeroOutputRatio: zero ? 0 : 0.25,
    ...overrides
  });
}

const staticPlan = makeStaticPlan();
assert.equal(staticPlan.gpuExecutionPlanReady, true);

const readyZeroObserver = readProductionTileExecutionPlanObserver(
  makeWords(),
  staticPlan
);
assert.equal(readyZeroObserver.observerReady, false);
assert.equal(
  readyZeroObserver.executionCompletionContract.contractVersion,
  PRODUCTION_TILE_EXECUTION_COMPLETION_CONTRACT_VERSION
);
assert.equal(
  readyZeroObserver.executionCompletionContract.executionCompletionReady,
  true
);
assert.equal(
  readyZeroObserver.executionCompletionContract.workClassification,
  'zero-reference'
);
assert.equal(
  readyZeroObserver.executionCompletionContract.staticPlanShapeMatches,
  true
);
assert.equal(
  readyZeroObserver.executionCompletionContract.stageCountsMatch,
  true
);
assert.equal(
  readyZeroObserver.executionCompletionContract.capacityRangeReady,
  true
);
assert.equal(
  readyZeroObserver.executionCompletionContract.workloadShapeReady,
  true
);
assert.equal(
  readProductionTileExecutionPlanObserver(makeWords())
    .executionCompletionContract.executionCompletionReady,
  false
);

const readyZeroCapacity = makeCapacity(readyZeroObserver);
assert.equal(readyZeroCapacity.tileReferenceCapacityReady, true);
assert.equal(readyZeroCapacity.workClassification, 'zero-reference');
assert.equal(readyZeroCapacity.executionPlanCompletionReady, true);

const readyZeroBounded = makeBounded(readyZeroObserver);
assert.equal(readyZeroBounded.boundedExecutionReady, true);
assert.equal(readyZeroBounded.inputReferenceCount, 0);
assert.equal(readyZeroBounded.completedReferenceCount, 0);

const readyZeroCompositor = makeCompositor(
  readyZeroObserver,
  readyZeroBounded
);
assert.equal(readyZeroCompositor.tileCompositorReady, true);
assert.equal(readyZeroCompositor.canonicalOutputCompletionReady, true);
assert.equal(readyZeroCompositor.productionWorkClassification, 'zero-reference');
assert.equal(readyZeroCompositor.realTileCompositorOutputReady, false);
assert.equal(readyZeroCompositor.tileCompositorContributionCount, 0);

const readyZeroWorkset = buildProductionResidentWorksetContract({
  resourceIdentity: 'workset:7',
  sceneResourceIdentity: 'scene:7',
  sceneRecordCount: 100,
  residentRecordCount: 100,
  resourceCapacityRecords: 100
});
const readyZeroFramePath = buildNativeWebGpuProductionFrameDataPathContract({
  worksetContract: readyZeroWorkset,
  stateResourceIdentity: 'state:7',
  attributeResourceIdentity: 'state:7:attributes',
  footprintResourceIdentity: 'state:7:footprint',
  tileInputResourceIdentity: 'tile-input:7',
  tileListInputResourceIdentity: 'tile-list:7',
  compositorInputResourceIdentity: 'tile-list:7',
  stateRecordCount: 100,
  tileInputRecordCount: 100,
  tileReferenceCapacityContract: readyZeroCapacity,
  boundedExecutionContract: readyZeroBounded,
  gpuExecutionPlanContract: staticPlan,
  terminalExecutionPlanObserver: readyZeroObserver,
  gpuResourceLineagePreserved: true,
  capacityOverflowDetected: false,
  capacityOverflowFailClosed: true,
  silentDropAllowed: false,
  compositorSubmitted: true
});
assert.equal(readyZeroFramePath.nativeProductionFrameDataPathReady, true);

const nonzeroWords = makeWords({
  requiredReferenceCount: 9,
  requiredPaddedReferenceCapacity: 10,
  maxReferencesPerTile: 4,
  totalReferenceChunkCount: 3,
  maxReferenceChunksPerTile: 1,
  scatteredReferenceCount: 9,
  sortedReferenceCount: 9,
  compositedReferenceCount: 9
});
const readyNonzeroObserver = readProductionTileExecutionPlanObserver(
  nonzeroWords,
  staticPlan
);
assert.equal(readyNonzeroObserver.observerReady, true);
assert.equal(
  readyNonzeroObserver.executionCompletionContract.executionCompletionReady,
  true
);
assert.equal(
  readyNonzeroObserver.executionCompletionContract.workClassification,
  'nonzero-reference'
);
const readyNonzeroCapacity = makeCapacity(readyNonzeroObserver);
const readyNonzeroBounded = makeBounded(readyNonzeroObserver);
const readyNonzeroCompositor = makeCompositor(
  readyNonzeroObserver,
  readyNonzeroBounded
);
assert.equal(readyNonzeroCapacity.tileReferenceCapacityReady, true);
assert.equal(readyNonzeroBounded.boundedExecutionReady, true);
assert.equal(readyNonzeroCompositor.tileCompositorReady, true);
assert.equal(readyNonzeroCompositor.canonicalOutputCompletionReady, true);
assert.equal(
  readyNonzeroCompositor.productionWorkClassification,
  'nonzero-reference'
);

const invalidObserverFixtures = [
  new Uint32Array(PRODUCTION_TILE_EXECUTION_PLAN_WORD_COUNT),
  makeWords({ magic: 0 }),
  makeWords({ status: PRODUCTION_TILE_EXECUTION_PLAN_STATUS.pending }),
  makeWords({ status: PRODUCTION_TILE_EXECUTION_PLAN_STATUS.capacityOverflow }),
  makeWords({ status: PRODUCTION_TILE_EXECUTION_PLAN_STATUS.executionFailure }),
  makeWords({ planIdentity: 8 }),
  makeWords({ recordCount: 99 }),
  makeWords({ tileCount: 2 }),
  makeWords({ referenceCapacity: 15 }),
  makeWords({ compactOffsetTableReady: 0 }),
  makeWords({ scatteredReferenceCount: 1 }),
  makeWords({ sortedReferenceCount: 1 }),
  makeWords({ compositedReferenceCount: 1 }),
  makeWords({ requiredPaddedReferenceCapacity: 17 }),
  makeWords({ overflowReferenceCount: 1 }),
  makeWords({ requiredPaddedReferenceCapacity: 1 }),
  makeWords({ maxReferencesPerTile: 1 }),
  makeWords({ totalReferenceChunkCount: 1 }),
  makeWords({ maxReferenceChunksPerTile: 1 })
];
for (const words of invalidObserverFixtures) {
  const observer = readProductionTileExecutionPlanObserver(words, staticPlan);
  assert.equal(
    observer.executionCompletionContract.executionCompletionReady,
    false
  );
}

for (const invalidStaticPlan of [
  makeStaticPlan({ recordCount: 0 }),
  makeStaticPlan({ tileCount: 0 }),
  makeStaticPlan({ referenceCapacity: 0 }),
  makeStaticPlan({ planIdentity: 0 })
]) {
  assert.equal(invalidStaticPlan.gpuExecutionPlanReady, false);
  const observer = readProductionTileExecutionPlanObserver(
    makeWords(),
    invalidStaticPlan
  );
  assert.equal(
    observer.executionCompletionContract.executionCompletionReady,
    false
  );
}

const zeroWithoutExecutionProof = makeCapacity(readyZeroObserver, {
  executionPlanCompletionReady: false
});
assert.equal(zeroWithoutExecutionProof.tileReferenceCapacityReady, false);
assert.equal(
  makeCapacity(readyZeroObserver, { allocatedReferenceCapacity: 0 })
    .tileReferenceCapacityReady,
  false
);
assert.equal(
  makeCapacity(readyZeroObserver, { capacityOverflowDetected: true })
    .tileReferenceCapacityReady,
  false
);
assert.equal(
  makeCapacity(readyZeroObserver, { silentDropAllowed: true })
    .tileReferenceCapacityReady,
  false
);

const incompleteBounded = makeBounded(readyZeroObserver, false);
assert.equal(incompleteBounded.boundedExecutionReady, false);
assert.equal(
  makeCompositor(readyZeroObserver, incompleteBounded)
    .canonicalOutputCompletionReady,
  false
);
assert.equal(
  makeCompositor(readyZeroObserver, readyZeroBounded, {
    outputTextureWritten: false
  }).canonicalOutputCompletionReady,
  false
);
assert.equal(
  makeCompositor(readyZeroObserver, readyZeroBounded, {
    outputTextureReadbackMatchesSummary: false
  }).canonicalOutputCompletionReady,
  false
);
assert.equal(
  makeCompositor(readyZeroObserver, readyZeroBounded, {
    compositorTraversedReferenceList: false
  }).canonicalOutputCompletionReady,
  false
);

console.log('Step119 Impl4 canonical zero-work execution completion smoke: OK');
