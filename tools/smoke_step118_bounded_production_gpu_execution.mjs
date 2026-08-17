import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
  buildProductionGpuExecutionContract,
  countBitonicCompareStages,
  resolveProductionGpuExecutionLimits,
  splitProductionGpuWork
} from '../demo/js/common_4dgs_production_gpu_execution_contracts.js';

const device = {
  limits: { maxComputeWorkgroupsPerDimension: 65535 }
};
const tileCount = 80 * 45;
const limits = resolveProductionGpuExecutionLimits({ device, tileCount });
assert.equal(limits.workgroupSize, 64);
assert.equal(limits.maxInvocationsPerSubmission, 65535 * 64);
assert.equal(
  limits.recordBatchSize,
  Math.floor((65535 * 64) / tileCount)
);
assert.equal(limits.compositorReferenceChunkSize, limits.workgroupSize);

const sceneRecordCount = 3231588;
const recordBatches = splitProductionGpuWork(
  sceneRecordCount,
  limits.recordBatchSize
);
assert.equal(recordBatches[0].start, 0);
assert.equal(recordBatches.at(-1).end, sceneRecordCount);
assert.equal(
  recordBatches.every(
    ({ start, end }) =>
      end > start && end - start <= limits.recordBatchSize
  ),
  true
);

const referenceCount = 8388608;
const referenceBatches = splitProductionGpuWork(
  referenceCount,
  limits.referenceBatchSize
);
assert.equal(referenceBatches.length, 3);
assert.equal(referenceBatches.at(-1).end, referenceCount);
assert.equal(countBitonicCompareStages(524288), 190);

const ready = buildProductionGpuExecutionContract({
  limits,
  inputRecordCount: sceneRecordCount,
  inputReferenceCount: referenceCount,
  completedRecordCount: sceneRecordCount,
  completedReferenceCount: referenceCount,
  countSubmissionCount: recordBatches.length,
  scatterSubmissionCount: recordBatches.length,
  sortSeedSubmissionCount: referenceBatches.length,
  sortCompareSubmissionCount:
    referenceBatches.length * countBitonicCompareStages(524288),
  sortValidationSubmissionCount: referenceBatches.length,
  compositorChunkSubmissionCount: 524288 / limits.compositorReferenceChunkSize,
  maximumRecordsInSubmission: limits.recordBatchSize,
  maximumReferencesInSubmission: limits.referenceBatchSize,
  maximumReferencesPerPixelInvocation: limits.compositorReferenceChunkSize,
  gpuResourceLineageMaintained: true,
  recordReferenceCapacitySeparated: true,
  silentDropAllowed: false,
  schedulerContinuationUsed: false,
  allStagesCompleted: true
});
assert.equal(ready.boundedExecutionReady, true);
assert.equal(ready.completedReferenceCount, referenceCount);
assert.equal(ready.silentDropAllowed, false);
assert.equal(ready.schedulerContinuationUsed, false);
assert.notEqual(ready.memoryCapacityPolicy, ready.executionCapacityPolicy);

const incomplete = buildProductionGpuExecutionContract({
  ...ready,
  limits,
  completedReferenceCount: referenceCount - 1,
  allStagesCompleted: false
});
assert.equal(incomplete.boundedExecutionReady, false);

const [tileListSource, executionSource, compositorSource, bindingSource] =
  await Promise.all([
    readFile(new URL('../demo/js/webgpu_gpu_owned_tile_list_layout.js', import.meta.url), 'utf8'),
    readFile(new URL('../demo/js/webgpu_bounded_tile_sort_and_compositor.js', import.meta.url), 'utf8'),
    readFile(new URL('../demo/js/webgpu_tile_list_compositor.js', import.meta.url), 'utf8'),
    readFile(new URL('../demo/js/common_4dgs_production_tile_input_binding_contracts.js', import.meta.url), 'utf8')
  ]);
assert.equal(tileListSource.includes('recordBatches'), true);
assert.equal(tileListSource.includes('silentDropAllowed: false'), true);
assert.equal(tileListSource.includes('executionPlanResources.planBuffer'), true);
assert.equal(executionSource.includes('compareSwapBitonicStage'), true);
assert.equal(executionSource.includes('compositeReferenceChunk'), true);
assert.equal(executionSource.includes('dispatchWorkgroupsIndirect'), true);
assert.equal(executionSource.includes('schedulerContinuationUsed: false'), true);
assert.equal(executionSource.includes('for (var k: u32'), false);
assert.equal(executionSource.includes('for (var orderSlot'), false);
assert.equal(executionSource.includes('for (var tile ='), false);
assert.equal(
  compositorSource.includes('executeBoundedProductionTileSortAndCompositor'),
  true
);
assert.equal(bindingSource.includes("binding: 3"), false);

console.log('Step118 bounded production GPU execution smoke: OK');
