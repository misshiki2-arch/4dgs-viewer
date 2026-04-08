import {
  GPU_SCREEN_TRANSFORM_PATH_CPU,
  GPU_SCREEN_TRANSFORM_PATH_GPU_PREP
} from './gpu_screen_transform_executor.js';

// Step43 stage 1
// Purpose:
// - define a narrow planning layer for splitting source items into execution batches
// - keep batch/chunk policy separate from executor truth-source ownership
// - prepare future mixed GPU/CPU execution without changing current public contracts
//
// Planner responsibilities:
// - read source item count, requested transform path, and max batch size
// - return a deterministic batch plan for later executor use
// - describe each batch with a minimal execution intent
//
// Planner non-responsibilities:
// - deciding actualTransformPath
// - deciding transformFallbackReason
// - generating packed data
// - probing backend capability directly
// - owning upload/debug truth
//
// Executor responsibilities after this stage:
// - choose actual execution path per batch
// - run GPU backend / CPU fallback
// - combine packed outputs
// - own requested/actual/fallback/upload truth fields

function clampBatchSize(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.floor(n));
}

function normalizeSourceItemCount(sourceItemCount) {
  const n = Number(sourceItemCount);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.floor(n));
}

function normalizeRequestedTransformPath(requestedTransformPath) {
  if (requestedTransformPath === GPU_SCREEN_TRANSFORM_PATH_GPU_PREP) {
    return GPU_SCREEN_TRANSFORM_PATH_GPU_PREP;
  }
  return GPU_SCREEN_TRANSFORM_PATH_CPU;
}

function buildBatchIntent(requestedTransformPath, maxBatchItems) {
  if (
    requestedTransformPath === GPU_SCREEN_TRANSFORM_PATH_GPU_PREP &&
    Number.isFinite(maxBatchItems) &&
    maxBatchItems > 0
  ) {
    return {
      intendedBackend: 'gpu',
      intentRole: 'gpu-batch-candidate',
      intentReason: 'requested-gpu-prep-with-batch-capacity'
    };
  }

  return {
    intendedBackend: 'cpu',
    intentRole: 'cpu-direct',
    intentReason: requestedTransformPath === GPU_SCREEN_TRANSFORM_PATH_GPU_PREP
      ? 'requested-gpu-prep-without-batch-capacity'
      : 'requested-cpu'
  };
}

function buildSingleBatchPlan(sourceItemCount, requestedTransformPath, intent) {
  return [{
    batchIndex: 0,
    start: 0,
    end: sourceItemCount,
    itemCount: sourceItemCount,
    requestedTransformPath,
    intendedBackend: intent.intendedBackend,
    intentRole: intent.intentRole,
    intentReason: intent.intentReason
  }];
}

function buildChunkedBatchPlan(sourceItemCount, requestedTransformPath, maxBatchItems, intent) {
  const batches = [];
  let start = 0;
  let batchIndex = 0;

  while (start < sourceItemCount) {
    const end = Math.min(sourceItemCount, start + maxBatchItems);
    batches.push({
      batchIndex,
      start,
      end,
      itemCount: end - start,
      requestedTransformPath,
      intendedBackend: intent.intendedBackend,
      intentRole: intent.intentRole,
      intentReason: intent.intentReason
    });
    start = end;
    batchIndex++;
  }

  return batches;
}

export function planGpuScreenTransformBatches({
  sourceItemCount = 0,
  maxBatchItems = 0,
  requestedTransformPath = GPU_SCREEN_TRANSFORM_PATH_CPU
} = {}) {
  const normalizedSourceItemCount = normalizeSourceItemCount(sourceItemCount);
  const normalizedMaxBatchItems = clampBatchSize(maxBatchItems);
  const normalizedRequestedTransformPath =
    normalizeRequestedTransformPath(requestedTransformPath);
  const intent = buildBatchIntent(
    normalizedRequestedTransformPath,
    normalizedMaxBatchItems
  );

  let batches;
  let planMode;

  if (normalizedSourceItemCount === 0) {
    batches = [];
    planMode = 'empty';
  } else if (
    normalizedRequestedTransformPath === GPU_SCREEN_TRANSFORM_PATH_GPU_PREP &&
    normalizedMaxBatchItems > 0
  ) {
    batches = buildChunkedBatchPlan(
      normalizedSourceItemCount,
      normalizedRequestedTransformPath,
      normalizedMaxBatchItems,
      intent
    );
    planMode = batches.length > 1 ? 'gpu-chunked' : 'gpu-single-batch';
  } else {
    batches = buildSingleBatchPlan(
      normalizedSourceItemCount,
      normalizedRequestedTransformPath,
      intent
    );
    planMode = 'cpu-single-batch';
  }

  return {
    sourceItemCount: normalizedSourceItemCount,
    maxBatchItems: normalizedMaxBatchItems,
    requestedTransformPath: normalizedRequestedTransformPath,
    planMode,
    batchCount: batches.length,
    batches
  };
}

export function summarizeGpuScreenTransformBatchPlan(plan) {
  const batches = Array.isArray(plan?.batches) ? plan.batches : [];
  let largestBatchItemCount = 0;

  for (const batch of batches) {
    const itemCount = normalizeSourceItemCount(batch?.itemCount);
    if (itemCount > largestBatchItemCount) {
      largestBatchItemCount = itemCount;
    }
  }

  return {
    sourceItemCount: normalizeSourceItemCount(plan?.sourceItemCount),
    maxBatchItems: clampBatchSize(plan?.maxBatchItems),
    requestedTransformPath: normalizeRequestedTransformPath(plan?.requestedTransformPath),
    planMode: typeof plan?.planMode === 'string' ? plan.planMode : 'empty',
    batchCount: batches.length,
    largestBatchItemCount
  };
}
