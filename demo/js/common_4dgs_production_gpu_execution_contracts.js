export const PRODUCTION_GPU_EXECUTION_CONTRACT_VERSION =
  'phase3-production-gpu-bounded-execution-v1';

export const PRODUCTION_GPU_EXECUTION_WORKGROUP_SIZE = 64;

function finiteInteger(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.floor(number)) : fallback;
}

export function splitProductionGpuWork(totalWorkItems, workItemsPerSubmission) {
  const total = finiteInteger(totalWorkItems);
  const capacity = Math.max(1, finiteInteger(workItemsPerSubmission, 1));
  const batches = [];
  for (let start = 0; start < total; start += capacity) {
    batches.push({ start, end: Math.min(total, start + capacity) });
  }
  return batches;
}

export function countBitonicCompareStages(maxPaddedTileReferenceCount) {
  const padded = finiteInteger(maxPaddedTileReferenceCount);
  if (padded <= 1) return 0;
  const levels = Math.ceil(Math.log2(padded));
  return (levels * (levels + 1)) / 2;
}

export function resolveProductionGpuExecutionLimits({
  device,
  tileCount = 0
} = {}) {
  const normalizedTileCount = Math.max(1, finiteInteger(tileCount, 1));
  const maxComputeWorkgroupsPerDimension = Math.max(
    1,
    finiteInteger(device?.limits?.maxComputeWorkgroupsPerDimension, 65535)
  );
  const workgroupSize = PRODUCTION_GPU_EXECUTION_WORKGROUP_SIZE;
  const maxInvocationsPerSubmission =
    maxComputeWorkgroupsPerDimension * workgroupSize;
  // A record can visit every tile.  This bound therefore limits the worst-case
  // record/tile loop work in a count or scatter submission, not just dispatch X.
  const recordBatchSize = Math.max(
    1,
    Math.floor(maxInvocationsPerSubmission / normalizedTileCount)
  );
  return {
    workgroupSize,
    maxComputeWorkgroupsPerDimension,
    maxInvocationsPerSubmission,
    maxRecordTileVisitsPerSubmission: maxInvocationsPerSubmission,
    recordBatchSize,
    referenceBatchSize: maxInvocationsPerSubmission,
    // One compositor invocation consumes at most one workgroup-width chunk.
    // The value is derived from the shader topology rather than scene policy.
    compositorReferenceChunkSize: workgroupSize,
    executionLimitSource:
      'webgpu-maxComputeWorkgroupsPerDimension-and-production-workgroup-topology'
  };
}

export function buildProductionGpuExecutionContract({
  status = 'ok',
  stage = null,
  limits = null,
  inputRecordCount = 0,
  inputReferenceCount = 0,
  completedRecordCount = 0,
  completedReferenceCount = 0,
  countSubmissionCount = 0,
  scatterSubmissionCount = 0,
  sortSeedSubmissionCount = 0,
  sortCompareSubmissionCount = 0,
  sortValidationSubmissionCount = 0,
  compositorChunkSubmissionCount = 0,
  maximumRecordsInSubmission = 0,
  maximumReferencesInSubmission = 0,
  maximumReferencesPerPixelInvocation = 0,
  gpuResourceLineageMaintained = false,
  recordReferenceCapacitySeparated = false,
  silentDropAllowed = false,
  schedulerContinuationUsed = false,
  allStagesCompleted = false,
  reason = null
} = {}) {
  const inputRecords = finiteInteger(inputRecordCount);
  const inputReferences = finiteInteger(inputReferenceCount);
  const completedRecords = finiteInteger(completedRecordCount);
  const completedReferences = finiteInteger(completedReferenceCount);
  const normalizedLimits = limits ?? {};
  const recordBound = finiteInteger(normalizedLimits.recordBatchSize);
  const referenceBound = finiteInteger(normalizedLimits.referenceBatchSize);
  const compositorBound = finiteInteger(
    normalizedLimits.compositorReferenceChunkSize
  );
  const ready =
    status === 'ok' &&
    recordBound > 0 &&
    referenceBound > 0 &&
    compositorBound > 0 &&
    completedRecords === inputRecords &&
    completedReferences === inputReferences &&
    finiteInteger(maximumRecordsInSubmission) <= recordBound &&
    finiteInteger(maximumReferencesInSubmission) <= referenceBound &&
    finiteInteger(maximumReferencesPerPixelInvocation) <= compositorBound &&
    gpuResourceLineageMaintained === true &&
    recordReferenceCapacitySeparated === true &&
    silentDropAllowed === false &&
    schedulerContinuationUsed === false &&
    allStagesCompleted === true;
  return {
    contractVersion: PRODUCTION_GPU_EXECUTION_CONTRACT_VERSION,
    status: ready ? 'ok' : status === 'ok' ? 'blocked' : status,
    boundedExecutionReady: ready,
    stage,
    memoryCapacityPolicy:
      'device-storage-limit-bounded-production-resource-allocation',
    executionCapacityPolicy:
      'device-dispatch-limit-bounded-in-frame-multi-submission-completion',
    limits: {
      workgroupSize: finiteInteger(normalizedLimits.workgroupSize),
      maxComputeWorkgroupsPerDimension: finiteInteger(
        normalizedLimits.maxComputeWorkgroupsPerDimension
      ),
      maxInvocationsPerSubmission: finiteInteger(
        normalizedLimits.maxInvocationsPerSubmission
      ),
      maxRecordTileVisitsPerSubmission: finiteInteger(
        normalizedLimits.maxRecordTileVisitsPerSubmission
      ),
      recordBatchSize: recordBound,
      referenceBatchSize: referenceBound,
      compositorReferenceChunkSize: compositorBound,
      executionLimitSource: normalizedLimits.executionLimitSource ?? null
    },
    inputRecordCount: inputRecords,
    inputReferenceCount: inputReferences,
    completedRecordCount: completedRecords,
    completedReferenceCount: completedReferences,
    countSubmissionCount: finiteInteger(countSubmissionCount),
    scatterSubmissionCount: finiteInteger(scatterSubmissionCount),
    sortSeedSubmissionCount: finiteInteger(sortSeedSubmissionCount),
    sortCompareSubmissionCount: finiteInteger(sortCompareSubmissionCount),
    sortValidationSubmissionCount: finiteInteger(sortValidationSubmissionCount),
    compositorChunkSubmissionCount: finiteInteger(
      compositorChunkSubmissionCount
    ),
    maximumRecordsInSubmission: finiteInteger(maximumRecordsInSubmission),
    maximumReferencesInSubmission: finiteInteger(maximumReferencesInSubmission),
    maximumReferencesPerPixelInvocation: finiteInteger(
      maximumReferencesPerPixelInvocation
    ),
    gpuResourceLineageMaintained: gpuResourceLineageMaintained === true,
    recordReferenceCapacitySeparated:
      recordReferenceCapacitySeparated === true,
    silentDropAllowed: silentDropAllowed === true,
    schedulerContinuationUsed: schedulerContinuationUsed === true,
    allStagesCompleted: allStagesCompleted === true,
    reason: ready ? null : reason ?? 'production-gpu-bounded-execution-incomplete'
  };
}
