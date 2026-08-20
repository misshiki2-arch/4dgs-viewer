import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
  buildProductionTileExecutionPlanContract,
  buildProductionTileExecutionPlanBindGroupEntries,
  buildProductionTileExecutionPlanBindGroupLayoutEntries,
  buildProductionTileExecutionPlanWgslBindings,
  PRODUCTION_TILE_EXECUTION_PLAN_MAGIC,
  PRODUCTION_TILE_EXECUTION_PLAN_OBSERVER_SCHEMA_VERSION,
  PRODUCTION_TILE_EXECUTION_PLAN_STATUS,
  PRODUCTION_TILE_EXECUTION_PLAN_WORD,
  PRODUCTION_TILE_EXECUTION_PLAN_WORD_COUNT,
  readProductionTileExecutionPlanObserver,
  resolveProductionTileExecutionPlanTopology
} from '../demo/js/common_4dgs_production_tile_execution_plan_contracts.js';

const topology = resolveProductionTileExecutionPlanTopology({
  tileCount: 3600,
  referenceCapacity: 8388608
});
assert.equal(topology.scanStageCount, 12);
assert.equal(topology.scanStageOffsets.at(-1), 2048);
assert.equal(topology.maximumReferenceChunkCount, 131072);

const contract = buildProductionTileExecutionPlanContract({
  resourceIdentity: 'plan:1',
  sourceTileInputResourceIdentity: 'tile-input:1',
  planIdentity: 1,
  tileCount: 3600,
  recordCount: 3231588,
  referenceCapacity: 8388608,
  scanStageCount: topology.scanStageCount,
  maximumReferenceChunkCount: topology.maximumReferenceChunkCount,
  gpuPlanBufferCreated: true,
  gpuTileTableCreated: true,
  gpuChunkTableCreated: true,
  gpuPlanGenerated: true,
  scatterConsumesPlan: true,
  sortConsumesPlan: true,
  compositorConsumesPlan: true,
  capacityOverflowFailClosed: true
});
assert.equal(contract.gpuExecutionPlanReady, true);
assert.equal(contract.productionCriticalReadbackUsed, false);
assert.equal(contract.intermediateCpuControlRoundTripUsed, false);
assert.equal(contract.sceneDependentCpuPlanMaterialized, false);

const layoutBindings = buildProductionTileExecutionPlanBindGroupLayoutEntries(1)
  .map((entry) => entry.binding);
const descriptorBindings = buildProductionTileExecutionPlanBindGroupEntries({
  tileCountsBuffer: {},
  scanSourceBuffer: {},
  scanDestinationBuffer: {},
  tileTableBuffer: {},
  tileChunkTableBuffer: {},
  planBuffer: {},
  paramsBuffer: {},
  compositorIndirectBuffer: {}
}).map((entry) => entry.binding);
const shaderBindings = [...buildProductionTileExecutionPlanWgslBindings()
  .matchAll(/@binding\((\d+)\)/g)]
  .map((match) => Number(match[1]));
assert.deepEqual(layoutBindings, [0, 1, 2, 3, 4, 5, 6, 7]);
assert.deepEqual(descriptorBindings, layoutBindings);
assert.deepEqual(shaderBindings, layoutBindings);
assert.equal(new Set(layoutBindings).size, layoutBindings.length);

function observerWords({ overflow = false } = {}) {
  const words = new Uint32Array(PRODUCTION_TILE_EXECUTION_PLAN_WORD_COUNT);
  const index = PRODUCTION_TILE_EXECUTION_PLAN_WORD;
  words[index.magic] = PRODUCTION_TILE_EXECUTION_PLAN_MAGIC;
  words[index.status] = overflow
    ? PRODUCTION_TILE_EXECUTION_PLAN_STATUS.capacityOverflow
    : PRODUCTION_TILE_EXECUTION_PLAN_STATUS.ready;
  words[index.recordCount] = 100;
  words[index.tileCount] = 3;
  words[index.referenceCapacity] = 16;
  words[index.requiredReferenceCount] = 9;
  words[index.requiredPaddedReferenceCapacity] = overflow ? 18 : 10;
  words[index.maxReferencesPerTile] = 4;
  words[index.totalReferenceChunkCount] = 3;
  words[index.maxReferenceChunksPerTile] = 1;
  words[index.scatteredReferenceCount] = overflow ? 0 : 9;
  words[index.sortedReferenceCount] = overflow ? 0 : 9;
  words[index.compositedReferenceCount] = overflow ? 0 : 9;
  words[index.overflowReferenceCount] = overflow ? 2 : 0;
  words[index.compactOffsetTableReady] = 1;
  words[index.planIdentity] = 1;
  return words;
}

const readyObserver = readProductionTileExecutionPlanObserver(
  observerWords(),
  {
    ...contract,
    planIdentity: 1,
    recordCount: 100,
    tileCount: 3,
    referenceCapacity: 16
  }
);
assert.equal(
  readyObserver.schemaVersion,
  PRODUCTION_TILE_EXECUTION_PLAN_OBSERVER_SCHEMA_VERSION
);
assert.equal(
  readyObserver.evidenceRole,
  'terminal-post-production-submission-observer'
);
assert.equal(readyObserver.productionControlInput, false);
assert.equal(readyObserver.rawPlanWordsPublished, false);
assert.equal(readyObserver.observerReady, true);
assert.equal(
  readyObserver.executionCompletionContract.executionCompletionReady,
  true
);
assert.equal(
  readyObserver.executionCompletionContract.staticPlanShapeMatches,
  true
);
assert.equal(readyObserver.requiredReferenceCount, 9);
assert.equal(readyObserver.scatteredReferenceCount, 9);
assert.equal(readyObserver.sortedReferenceCount, 9);
assert.equal(readyObserver.compositedReferenceCount, 9);

const overflowObserver = readProductionTileExecutionPlanObserver(
  observerWords({ overflow: true }),
  {
    ...contract,
    planIdentity: 1,
    recordCount: 100,
    tileCount: 3,
    referenceCapacity: 16
  }
);
assert.equal(overflowObserver.observerReady, false);
assert.equal(
  overflowObserver.executionCompletionContract.executionCompletionReady,
  false
);
assert.equal(overflowObserver.capacityOverflowDetected, true);
assert.equal(overflowObserver.capacityOverflowFailClosed, true);

const [layoutSource, planSource, executionSource] = await Promise.all([
  readFile(new URL('../demo/js/webgpu_gpu_owned_tile_list_layout.js', import.meta.url), 'utf8'),
  readFile(new URL('../demo/js/webgpu_production_tile_execution_plan.js', import.meta.url), 'utf8'),
  readFile(new URL('../demo/js/webgpu_bounded_tile_sort_and_compositor.js', import.meta.url), 'utf8')
]);
assert.equal(layoutSource.includes('buildProductionTileReferencePlanFromTileCounts'), false);
assert.equal(layoutSource.includes('mapAsync'), false);
assert.equal(layoutSource.includes('countedTileReferences'), false);
assert.equal(layoutSource.includes('executionPlanResources.planBuffer'), true);
assert.equal(planSource.includes('initializeScan'), true);
assert.equal(planSource.includes('scanStep'), true);
assert.equal(planSource.includes('finalizePlan'), true);
assert.equal(planSource.includes('buildCompositorDispatches'), true);
assert.equal(executionSource.includes('resources.executionPlanBuffer'), true);
assert.equal(executionSource.includes('dispatchWorkgroupsIndirect'), true);
assert.equal(executionSource.includes('resources.maxReferencesPerTile'), false);
assert.equal(executionSource.includes('await submitOrdering'), false);
assert.equal(executionSource.includes('await submitCompositor'), false);
assert.equal(
  executionSource.indexOf('onProductionSubmitted({') <
    executionSource.indexOf('summaryReadbackBuffer.mapAsync'),
  true
);
assert.equal(
  executionSource.slice(
    executionSource.indexOf('summaryReadbackBuffer.mapAsync')
  ).includes('queue.writeBuffer'),
  false
);

console.log('Step118 GPU-native production execution-plan smoke: OK');
