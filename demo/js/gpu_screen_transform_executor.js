import { GPU_VISIBLE_PACK_FLOATS_PER_ITEM } from './gpu_buffer_layout_utils.js';
import { packVisibleItems } from './gpu_visible_pack_utils.js';
import {
  createGpuScreenTransformBackendGpuContext,
  executeGpuScreenTransformBackendGpu,
  summarizeGpuScreenTransformBackendGpuCapability
} from './gpu_screen_transform_backend_gpu.js';
import { planGpuScreenTransformBatches } from './gpu_screen_transform_batch_planner.js';

// Step37 stage 1
// 目的:
// - transform executor を requested / actual / fallback / upload state の唯一の truth source にする
// - packed formal draw contract は維持したまま、将来 GPU transform 実装を差し込める境界を固定する
//
// truth source:
// - requestedTransformPath
// - actualTransformPath
// - transformFallbackReason
// - transformConfigured
// - transformHasBuffers
// - transformUpload*
// - transformStageMs
//
// 非目標:
// - source-item build の変更
// - draw contract の変更
// - UI/state/tile 層の変更
//
// 設計:
// 1. executor の入力は sourceItemsResult と sourcePath / experimental / requestedTransformPath
// 2. sourcePath から requestedTransformPath をここで最終決定する
// 3. 現段階では actualTransformPath は CPU 実装のみ
// 4. GPU_PREP requested 時は fallbackReason を必ず明示する
// 5. builder / renderer / debug はこの summary を読むだけにする

export const GPU_SCREEN_TRANSFORM_PATH_CPU = 'cpu-packed-transform';
export const GPU_SCREEN_TRANSFORM_PATH_GPU_PREP = 'gpu-packed-transform-prep';

export const GPU_SCREEN_TRANSFORM_ROLE_FORMAL = 'formal-transform';
export const GPU_SCREEN_TRANSFORM_ROLE_EXPERIMENTAL = 'experimental-transform';

export const GPU_SCREEN_TRANSFORM_SOURCE_PACKED_CPU = 'packed-cpu';
export const GPU_SCREEN_TRANSFORM_SOURCE_PACKED_GPU_PREP = 'packed-gpu-prep';
export const GPU_SCREEN_TRANSFORM_SOURCE_GPU_SCREEN_EXPERIMENTAL = 'gpu-screen-experimental';

function nowMs() {
  return performance.now();
}

function resolveTransformExecutionInputs(sourceItemsResult, options = {}) {
  const experimental = !!options.experimental || !!sourceItemsResult?.experimental;
  const sourcePath = normalizeSourcePath(options.sourcePath ?? sourceItemsResult?.path, experimental);

  return {
    sourcePath,
    experimental,
    requestedTransformPath: resolveRequestedTransformPath({
      sourcePath,
      experimental,
      requestedTransformPath: options.transformPath ?? null
    })
  };
}

function normalizeSourcePath(path, experimental = false) {
  if (path === GPU_SCREEN_TRANSFORM_SOURCE_PACKED_CPU) {
    return GPU_SCREEN_TRANSFORM_SOURCE_PACKED_CPU;
  }
  if (path === GPU_SCREEN_TRANSFORM_SOURCE_PACKED_GPU_PREP) {
    return GPU_SCREEN_TRANSFORM_SOURCE_PACKED_GPU_PREP;
  }
  if (path === GPU_SCREEN_TRANSFORM_SOURCE_GPU_SCREEN_EXPERIMENTAL) {
    return GPU_SCREEN_TRANSFORM_SOURCE_GPU_SCREEN_EXPERIMENTAL;
  }
  return experimental
    ? GPU_SCREEN_TRANSFORM_SOURCE_PACKED_GPU_PREP
    : GPU_SCREEN_TRANSFORM_SOURCE_PACKED_CPU;
}

function isGpuPrepLikeSourcePath(sourcePath) {
  return (
    sourcePath === GPU_SCREEN_TRANSFORM_SOURCE_PACKED_GPU_PREP ||
    sourcePath === GPU_SCREEN_TRANSFORM_SOURCE_GPU_SCREEN_EXPERIMENTAL
  );
}

function resolveRequestedTransformPath({ sourcePath, experimental = false, requestedTransformPath = null }) {
  if (requestedTransformPath === GPU_SCREEN_TRANSFORM_PATH_CPU) {
    return GPU_SCREEN_TRANSFORM_PATH_CPU;
  }
  if (requestedTransformPath === GPU_SCREEN_TRANSFORM_PATH_GPU_PREP) {
    return GPU_SCREEN_TRANSFORM_PATH_GPU_PREP;
  }

  const normalizedSourcePath = normalizeSourcePath(sourcePath, experimental);
  return isGpuPrepLikeSourcePath(normalizedSourcePath)
    ? GPU_SCREEN_TRANSFORM_PATH_GPU_PREP
    : GPU_SCREEN_TRANSFORM_PATH_CPU;
}

function resolveActualTransformPath(_requestedTransformPath) {
  return GPU_SCREEN_TRANSFORM_PATH_CPU;
}

function shouldPromoteActualTransformPathToGpuPrep(requestedTransformPath, backendResult) {
  if (requestedTransformPath !== GPU_SCREEN_TRANSFORM_PATH_GPU_PREP) return false;
  if (!backendResult?.backendImplemented) return false;
  if (!backendResult?.backendProducedPacked) return false;
  if (!!backendResult?.backendFallbackToCpu) return false;
  if (backendResult?.backendError !== null && backendResult?.backendError !== undefined) return false;
  if (!(backendResult?.packed instanceof Float32Array)) return false;

  const count = backendResult.count;
  const floatsPerItem = backendResult.floatsPerItem;
  if (!Number.isInteger(count) || count < 0) return false;
  if (!Number.isFinite(floatsPerItem) || floatsPerItem !== GPU_VISIBLE_PACK_FLOATS_PER_ITEM) return false;

  const packedLength = backendResult.packed.length;
  if (count === 0) return packedLength === 0;
  return packedLength === count * floatsPerItem;
}

function resolveTransformRole(actualTransformPath) {
  return actualTransformPath === GPU_SCREEN_TRANSFORM_PATH_GPU_PREP
    ? GPU_SCREEN_TRANSFORM_ROLE_EXPERIMENTAL
    : GPU_SCREEN_TRANSFORM_ROLE_FORMAL;
}

function resolveFallbackReason(requestedTransformPath, actualTransformPath) {
  if (requestedTransformPath === actualTransformPath) return 'none';
  if (
    requestedTransformPath === GPU_SCREEN_TRANSFORM_PATH_GPU_PREP &&
    actualTransformPath === GPU_SCREEN_TRANSFORM_PATH_CPU
  ) {
    return 'gpu-transform-not-implemented-use-cpu-pack';
  }
  return 'transform-path-fallback';
}

function safeSourceItems(sourceItemsResult) {
  return Array.isArray(sourceItemsResult?.items) ? sourceItemsResult.items : [];
}

function buildSourceItemsResultSlice(sourceItemsResult, batch) {
  const items = safeSourceItems(sourceItemsResult);
  const start = Number.isFinite(batch?.start) ? Math.max(0, batch.start | 0) : 0;
  const end = Number.isFinite(batch?.end) ? Math.max(start, batch.end | 0) : start;
  const slicedItems = items.slice(start, end);

  return {
    ...sourceItemsResult,
    items: slicedItems,
    itemCount: slicedItems.length
  };
}

function safeSourceItemCount(sourceItemsResult) {
  if (Number.isFinite(sourceItemsResult?.itemCount)) return sourceItemsResult.itemCount;
  return safeSourceItems(sourceItemsResult).length;
}

function safeSourceSchemaVersion(sourceItemsResult) {
  return Number.isFinite(sourceItemsResult?.schemaVersion)
    ? sourceItemsResult.schemaVersion
    : 0;
}

function buildTransformUploadSummary(packed, packedCount) {
  // upload ownership lives here; downstream callers only read the summary fields.
  const packedLength = packed instanceof Float32Array ? packed.length : 0;
  const uploadBytes = packedLength * 4;
  return {
    transformUploadBytes: uploadBytes,
    transformUploadCount: Number.isFinite(packedCount) ? packedCount : 0,
    transformUploadLength: packedLength,
    transformUploadCapacityBytes: uploadBytes,
    transformUploadReusedCapacity: false
  };
}

function resolveTransformBackendDispatch(requestedTransformPath, options = {}) {
  // Step38 seam:
  // backend selection is centralized here so future GPU implementations can be inserted
  // without changing the public transform contract.
  switch (requestedTransformPath) {
    case GPU_SCREEN_TRANSFORM_PATH_GPU_PREP:
      const backendContext = createGpuScreenTransformBackendGpuContext();
      return {
        backendId: 'gpu-prep-requested-gpu-helper',
        backendContext,
        backendCapability: summarizeGpuScreenTransformBackendGpuCapability(backendContext, options.gl ?? null),
        execute: (sourceItemsResult) => executeGpuScreenTransformBackendGpu(
          backendContext,
          sourceItemsResult,
          { gl: options.gl ?? null }
        )
      };
    case GPU_SCREEN_TRANSFORM_PATH_CPU:
    default:
      return {
        backendId: 'cpu-packed',
        execute: runCpuPackedTransform
      };
  }
}

function runCpuPackedTransform(sourceItemsResult) {
  const items = safeSourceItems(sourceItemsResult);
  const packedResult = packVisibleItems(items);
  return {
    packed: packedResult?.packed instanceof Float32Array ? packedResult.packed : null,
    count: Number.isFinite(packedResult?.count) ? packedResult.count : items.length,
    floatsPerItem: Number.isFinite(packedResult?.floatsPerItem)
      ? packedResult.floatsPerItem
      : GPU_VISIBLE_PACK_FLOATS_PER_ITEM
  };
}

function combinePackedBatchResults(batchResults) {
  const results = Array.isArray(batchResults) ? batchResults : [];
  let totalCount = 0;
  let floatsPerItem = GPU_VISIBLE_PACK_FLOATS_PER_ITEM;
  let totalPackedLength = 0;
  let hasPackedArray = false;

  for (const result of results) {
    const count = Number.isFinite(result?.count) ? result.count : 0;
    totalCount += count;

    if (Number.isFinite(result?.floatsPerItem)) {
      floatsPerItem = result.floatsPerItem;
    }

    if (result?.packed instanceof Float32Array) {
      hasPackedArray = true;
      totalPackedLength += result.packed.length;
    }
  }

  const packed = hasPackedArray ? new Float32Array(totalPackedLength) : new Float32Array(0);
  let offset = 0;
  for (const result of results) {
    if (!(result?.packed instanceof Float32Array)) continue;
    packed.set(result.packed, offset);
    offset += result.packed.length;
  }

  return {
    packed,
    count: totalCount,
    floatsPerItem
  };
}

function didAllBatchesPromoteToGpuPrep(requestedTransformPath, batchResults) {
  if (requestedTransformPath !== GPU_SCREEN_TRANSFORM_PATH_GPU_PREP) return false;
  if (!Array.isArray(batchResults) || batchResults.length === 0) return false;

  for (const result of batchResults) {
    if (!shouldPromoteActualTransformPathToGpuPrep(requestedTransformPath, result)) {
      return false;
    }
  }

  return true;
}

export function createGpuScreenTransformContext() {
  return {
    sourcePath: GPU_SCREEN_TRANSFORM_SOURCE_PACKED_CPU,
    requestedTransformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
    actualTransformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
    transformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
    transformRole: GPU_SCREEN_TRANSFORM_ROLE_FORMAL,
    transformConfigured: false,
    transformHasBuffers: false,
    transformFallbackReason: 'none',
    transformUploadBytes: 0,
    transformUploadCount: 0,
    transformUploadLength: 0,
    transformUploadCapacityBytes: 0,
    transformUploadReusedCapacity: false,
    sourceItemCount: 0,
    sourceSchemaVersion: 0,
    packedCount: 0,
    packedLength: 0,
    floatsPerItem: GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
    transformStageMs: 0,
    lastSummary: null
  };
}

// This summary is the truth source for transform state.
// builder / renderer / debug should forward these fields without reinterpretation.
function buildTransformSummary({
  sourcePath,
  sourceItemsResult,
  requestedTransformPath,
  actualTransformPath,
  packed,
  packedCount,
  floatsPerItem,
  transformStageMs
}) {
  const uploadSummary = buildTransformUploadSummary(packed, packedCount);
  const fallbackReason = resolveFallbackReason(requestedTransformPath, actualTransformPath);

  return {
    sourcePath,
    requestedTransformPath,
    actualTransformPath,
    transformPath: actualTransformPath,
    transformRole: resolveTransformRole(actualTransformPath),
    transformConfigured: true,
    transformHasBuffers: false,
    transformFallbackReason: fallbackReason,
    sourceItemCount: safeSourceItemCount(sourceItemsResult),
    sourceSchemaVersion: safeSourceSchemaVersion(sourceItemsResult),
    packedCount: Number.isFinite(packedCount) ? packedCount : 0,
    packedLength: packed instanceof Float32Array ? packed.length : 0,
    floatsPerItem: Number.isFinite(floatsPerItem)
      ? floatsPerItem
      : GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
    transformStageMs: Number.isFinite(transformStageMs) ? transformStageMs : 0,
    ...uploadSummary
  };
}

function updateContext(context, summary) {
  if (!context || !summary) return;
  context.sourcePath = summary.sourcePath ?? GPU_SCREEN_TRANSFORM_SOURCE_PACKED_CPU;
  context.requestedTransformPath = summary.requestedTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU;
  context.actualTransformPath = summary.actualTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU;
  context.transformPath = summary.transformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU;
  context.transformRole = summary.transformRole ?? GPU_SCREEN_TRANSFORM_ROLE_FORMAL;
  context.transformConfigured = !!summary.transformConfigured;
  context.transformHasBuffers = !!summary.transformHasBuffers;
  context.transformFallbackReason = summary.transformFallbackReason ?? 'none';
  context.transformUploadBytes = Number.isFinite(summary.transformUploadBytes) ? summary.transformUploadBytes : 0;
  context.transformUploadCount = Number.isFinite(summary.transformUploadCount) ? summary.transformUploadCount : 0;
  context.transformUploadLength = Number.isFinite(summary.transformUploadLength) ? summary.transformUploadLength : 0;
  context.transformUploadCapacityBytes = Number.isFinite(summary.transformUploadCapacityBytes) ? summary.transformUploadCapacityBytes : 0;
  context.transformUploadReusedCapacity = !!summary.transformUploadReusedCapacity;
  context.sourceItemCount = Number.isFinite(summary.sourceItemCount) ? summary.sourceItemCount : 0;
  context.sourceSchemaVersion = Number.isFinite(summary.sourceSchemaVersion) ? summary.sourceSchemaVersion : 0;
  context.packedCount = Number.isFinite(summary.packedCount) ? summary.packedCount : 0;
  context.packedLength = Number.isFinite(summary.packedLength) ? summary.packedLength : 0;
  context.floatsPerItem = Number.isFinite(summary.floatsPerItem)
    ? summary.floatsPerItem
    : GPU_VISIBLE_PACK_FLOATS_PER_ITEM;
  context.transformStageMs = Number.isFinite(summary.transformStageMs) ? summary.transformStageMs : 0;
  context.lastSummary = summary;
}

export function executeGpuScreenPackedTransform(context, sourceItemsResult, options = {}) {
  const executionInputs = resolveTransformExecutionInputs(sourceItemsResult, options);
  const backendDispatch = resolveTransformBackendDispatch(executionInputs.requestedTransformPath, options);
  const backendCapability = backendDispatch.backendCapability ?? null;
  const batchPlan = planGpuScreenTransformBatches({
    sourceItemCount: safeSourceItemCount(sourceItemsResult),
    maxBatchItems: backendCapability?.maxBatchItems ?? 0,
    requestedTransformPath: executionInputs.requestedTransformPath
  });

  const t0 = nowMs();

  const batchResults = batchPlan.batches.map((batch) => {
    const batchSourceItemsResult = buildSourceItemsResultSlice(sourceItemsResult, batch);
    return backendDispatch.execute(batchSourceItemsResult);
  });
  const combinedBatchResult = combinePackedBatchResults(batchResults);

  const actualTransformPath = didAllBatchesPromoteToGpuPrep(
    executionInputs.requestedTransformPath,
    batchResults
  )
    ? GPU_SCREEN_TRANSFORM_PATH_GPU_PREP
    : resolveActualTransformPath(executionInputs.requestedTransformPath);
  const transformStageMs = nowMs() - t0;

  const summary = buildTransformSummary({
    sourcePath: executionInputs.sourcePath,
    sourceItemsResult,
    requestedTransformPath: executionInputs.requestedTransformPath,
    actualTransformPath,
    packed: combinedBatchResult.packed,
    packedCount: combinedBatchResult.count,
    floatsPerItem: combinedBatchResult.floatsPerItem,
    transformStageMs
  });

  if (backendCapability && backendCapability.isReady) {
    // The backend seam is wired, but actual transform ownership stays in this executor for now.
  }

  updateContext(context, summary);

  return {
    packed: combinedBatchResult.packed,
    count: combinedBatchResult.count,
    floatsPerItem: combinedBatchResult.floatsPerItem,
    sourcePath: summary.sourcePath,
    requestedTransformPath: summary.requestedTransformPath,
    actualTransformPath: summary.actualTransformPath,
    transformPath: summary.transformPath,
    transformRole: summary.transformRole,
    transformConfigured: summary.transformConfigured,
    transformHasBuffers: summary.transformHasBuffers,
    transformFallbackReason: summary.transformFallbackReason,
    transformStageMs: summary.transformStageMs,
    transformUploadBytes: summary.transformUploadBytes,
    transformUploadCount: summary.transformUploadCount,
    transformUploadLength: summary.transformUploadLength,
    transformUploadCapacityBytes: summary.transformUploadCapacityBytes,
    transformUploadReusedCapacity: summary.transformUploadReusedCapacity,
    sourceItems: safeSourceItems(sourceItemsResult),
    sourceItemCount: summary.sourceItemCount,
    sourceSchemaVersion: summary.sourceSchemaVersion,
    summary
  };
}

export function summarizeGpuScreenTransformResult(result) {
  if (!result) {
    return {
      sourcePath: GPU_SCREEN_TRANSFORM_SOURCE_PACKED_CPU,
      requestedTransformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
      actualTransformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
      transformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
      transformRole: GPU_SCREEN_TRANSFORM_ROLE_FORMAL,
      transformConfigured: false,
      transformHasBuffers: false,
      transformFallbackReason: 'none',
      sourceItemCount: 0,
      sourceSchemaVersion: 0,
      packedCount: 0,
      packedLength: 0,
      floatsPerItem: GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
      transformStageMs: 0,
      transformUploadBytes: 0,
      transformUploadCount: 0,
      transformUploadLength: 0,
      transformUploadCapacityBytes: 0,
      transformUploadReusedCapacity: false
    };
  }

  const packed = result.packed;
  return {
    sourcePath: result.sourcePath ?? GPU_SCREEN_TRANSFORM_SOURCE_PACKED_CPU,
    requestedTransformPath: result.requestedTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    actualTransformPath: result.actualTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    transformPath: result.transformPath ?? result.actualTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    transformRole: result.transformRole ?? resolveTransformRole(result.actualTransformPath),
    transformConfigured: !!result.transformConfigured,
    transformHasBuffers: !!result.transformHasBuffers,
    transformFallbackReason: result.transformFallbackReason ?? 'none',
    sourceItemCount: Number.isFinite(result.sourceItemCount) ? result.sourceItemCount : 0,
    sourceSchemaVersion: Number.isFinite(result.sourceSchemaVersion) ? result.sourceSchemaVersion : 0,
    packedCount: Number.isFinite(result.count) ? result.count : 0,
    packedLength: packed instanceof Float32Array ? packed.length : 0,
    floatsPerItem: Number.isFinite(result.floatsPerItem)
      ? result.floatsPerItem
      : GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
    transformStageMs: Number.isFinite(result.transformStageMs) ? result.transformStageMs : 0,
    transformUploadBytes: Number.isFinite(result.transformUploadBytes) ? result.transformUploadBytes : 0,
    transformUploadCount: Number.isFinite(result.transformUploadCount) ? result.transformUploadCount : 0,
    transformUploadLength: Number.isFinite(result.transformUploadLength) ? result.transformUploadLength : 0,
    transformUploadCapacityBytes: Number.isFinite(result.transformUploadCapacityBytes) ? result.transformUploadCapacityBytes : 0,
    transformUploadReusedCapacity: !!result.transformUploadReusedCapacity
  };
}

export function summarizeGpuScreenTransformContext(context) {
  if (!context) {
    return {
      sourcePath: GPU_SCREEN_TRANSFORM_SOURCE_PACKED_CPU,
      requestedTransformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
      actualTransformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
      transformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
      transformRole: GPU_SCREEN_TRANSFORM_ROLE_FORMAL,
      transformConfigured: false,
      transformHasBuffers: false,
      transformFallbackReason: 'none',
      transformUploadBytes: 0,
      transformUploadCount: 0,
      transformUploadLength: 0,
      transformUploadCapacityBytes: 0,
      transformUploadReusedCapacity: false,
      sourceItemCount: 0,
      sourceSchemaVersion: 0,
      packedCount: 0,
      packedLength: 0,
      floatsPerItem: GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
      transformStageMs: 0,
      lastSummary: null
    };
  }

  return {
    sourcePath: context.sourcePath ?? GPU_SCREEN_TRANSFORM_SOURCE_PACKED_CPU,
    requestedTransformPath: context.requestedTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    actualTransformPath: context.actualTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    transformPath: context.transformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    transformRole: context.transformRole ?? GPU_SCREEN_TRANSFORM_ROLE_FORMAL,
    transformConfigured: !!context.transformConfigured,
    transformHasBuffers: !!context.transformHasBuffers,
    transformFallbackReason: context.transformFallbackReason ?? 'none',
    transformUploadBytes: Number.isFinite(context.transformUploadBytes) ? context.transformUploadBytes : 0,
    transformUploadCount: Number.isFinite(context.transformUploadCount) ? context.transformUploadCount : 0,
    transformUploadLength: Number.isFinite(context.transformUploadLength) ? context.transformUploadLength : 0,
    transformUploadCapacityBytes: Number.isFinite(context.transformUploadCapacityBytes) ? context.transformUploadCapacityBytes : 0,
    transformUploadReusedCapacity: !!context.transformUploadReusedCapacity,
    sourceItemCount: Number.isFinite(context.sourceItemCount) ? context.sourceItemCount : 0,
    sourceSchemaVersion: Number.isFinite(context.sourceSchemaVersion) ? context.sourceSchemaVersion : 0,
    packedCount: Number.isFinite(context.packedCount) ? context.packedCount : 0,
    packedLength: Number.isFinite(context.packedLength) ? context.packedLength : 0,
    floatsPerItem: Number.isFinite(context.floatsPerItem)
      ? context.floatsPerItem
      : GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
    transformStageMs: Number.isFinite(context.transformStageMs) ? context.transformStageMs : 0,
    lastSummary: context.lastSummary ?? null
  };
}
