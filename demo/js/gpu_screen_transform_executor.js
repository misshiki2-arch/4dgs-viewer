import { GPU_VISIBLE_PACK_FLOATS_PER_ITEM } from './gpu_buffer_layout_utils.js';
import { packVisibleItems } from './gpu_visible_pack_utils.js';
import {
  GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION
} from './gpu_screen_space_builder.js';

export const GPU_SCREEN_TRANSFORM_PATH_CPU = 'cpu-packed-transform';
export const GPU_SCREEN_TRANSFORM_PATH_GPU_PREP = 'gpu-packed-transform-prep';
export const GPU_SCREEN_TRANSFORM_ROLE_FORMAL = 'formal-transform';
export const GPU_SCREEN_TRANSFORM_ROLE_EXPERIMENTAL = 'experimental-transform';

function nowMs() {
  return performance.now();
}

function normalizeTransformPath(path) {
  if (path === GPU_SCREEN_TRANSFORM_PATH_CPU) return GPU_SCREEN_TRANSFORM_PATH_CPU;
  if (path === GPU_SCREEN_TRANSFORM_PATH_GPU_PREP) return GPU_SCREEN_TRANSFORM_PATH_GPU_PREP;
  return GPU_SCREEN_TRANSFORM_PATH_CPU;
}

function getTransformRole(path) {
  return path === GPU_SCREEN_TRANSFORM_PATH_GPU_PREP
    ? GPU_SCREEN_TRANSFORM_ROLE_EXPERIMENTAL
    : GPU_SCREEN_TRANSFORM_ROLE_FORMAL;
}

function safeSourceItems(sourceItemsResult) {
  return Array.isArray(sourceItemsResult?.items) ? sourceItemsResult.items : [];
}

function safeSourceItemCount(sourceItemsResult) {
  if (Number.isFinite(sourceItemsResult?.itemCount)) return sourceItemsResult.itemCount;
  const items = safeSourceItems(sourceItemsResult);
  return items.length;
}

function safeSchemaVersion(sourceItemsResult) {
  if (Number.isFinite(sourceItemsResult?.schemaVersion)) {
    return sourceItemsResult.schemaVersion;
  }
  return GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION;
}

export function createGpuScreenTransformContext() {
  return {
    lastTransformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
    lastTransformRole: GPU_SCREEN_TRANSFORM_ROLE_FORMAL,
    lastSourceItemCount: 0,
    lastPackedCount: 0,
    lastPackedLength: 0,
    lastTransformStageMs: 0,
    lastFallbackReason: 'none',
    lastSummary: null
  };
}

function buildTransformSummary({
  transformPath,
  sourceItemsResult,
  packed,
  packedCount,
  floatsPerItem,
  transformStageMs,
  fallbackReason
}) {
  return {
    transformPath,
    transformRole: getTransformRole(transformPath),
    transformFallbackReason: fallbackReason ?? 'none',
    sourceItemCount: safeSourceItemCount(sourceItemsResult),
    sourceSchemaVersion: safeSchemaVersion(sourceItemsResult),
    packedCount: Number.isFinite(packedCount) ? packedCount : 0,
    packedLength: packed instanceof Float32Array ? packed.length : 0,
    floatsPerItem: Number.isFinite(floatsPerItem)
      ? floatsPerItem
      : GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
    transformStageMs: Number.isFinite(transformStageMs) ? transformStageMs : 0
  };
}

function updateContext(context, summary) {
  if (!context || !summary) return;
  context.lastTransformPath = summary.transformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU;
  context.lastTransformRole = summary.transformRole ?? GPU_SCREEN_TRANSFORM_ROLE_FORMAL;
  context.lastSourceItemCount = Number.isFinite(summary.sourceItemCount) ? summary.sourceItemCount : 0;
  context.lastPackedCount = Number.isFinite(summary.packedCount) ? summary.packedCount : 0;
  context.lastPackedLength = Number.isFinite(summary.packedLength) ? summary.packedLength : 0;
  context.lastTransformStageMs = Number.isFinite(summary.transformStageMs) ? summary.transformStageMs : 0;
  context.lastFallbackReason = summary.transformFallbackReason ?? 'none';
  context.lastSummary = summary;
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

export function executeGpuScreenPackedTransform(context, sourceItemsResult, options = {}) {
  const requestedTransformPath = normalizeTransformPath(options.transformPath);
  const t0 = nowMs();

  const cpuResult = runCpuPackedTransform(sourceItemsResult);

  const actualTransformPath = requestedTransformPath;
  const fallbackReason =
    requestedTransformPath === GPU_SCREEN_TRANSFORM_PATH_GPU_PREP
      ? 'gpu-transform-not-implemented-use-cpu-pack'
      : 'none';

  const transformStageMs = nowMs() - t0;

  const summary = buildTransformSummary({
    transformPath: actualTransformPath,
    sourceItemsResult,
    packed: cpuResult.packed,
    packedCount: cpuResult.count,
    floatsPerItem: cpuResult.floatsPerItem,
    transformStageMs,
    fallbackReason
  });

  updateContext(context, summary);

  return {
    packed: cpuResult.packed,
    count: cpuResult.count,
    floatsPerItem: cpuResult.floatsPerItem,
    transformPath: summary.transformPath,
    transformRole: summary.transformRole,
    transformFallbackReason: summary.transformFallbackReason,
    transformStageMs: summary.transformStageMs,
    sourceItems: safeSourceItems(sourceItemsResult),
    sourceItemCount: summary.sourceItemCount,
    sourceSchemaVersion: summary.sourceSchemaVersion,
    summary
  };
}

export function summarizeGpuScreenTransformResult(result) {
  if (!result) {
    return {
      transformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
      transformRole: GPU_SCREEN_TRANSFORM_ROLE_FORMAL,
      transformFallbackReason: 'none',
      sourceItemCount: 0,
      sourceSchemaVersion: GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION,
      packedCount: 0,
      packedLength: 0,
      floatsPerItem: GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
      transformStageMs: 0
    };
  }

  const packed = result.packed;
  return {
    transformPath: result.transformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    transformRole: result.transformRole ?? getTransformRole(result.transformPath),
    transformFallbackReason: result.transformFallbackReason ?? 'none',
    sourceItemCount: Number.isFinite(result.sourceItemCount) ? result.sourceItemCount : 0,
    sourceSchemaVersion: Number.isFinite(result.sourceSchemaVersion)
      ? result.sourceSchemaVersion
      : GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION,
    packedCount: Number.isFinite(result.count) ? result.count : 0,
    packedLength: packed instanceof Float32Array ? packed.length : 0,
    floatsPerItem: Number.isFinite(result.floatsPerItem)
      ? result.floatsPerItem
      : GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
    transformStageMs: Number.isFinite(result.transformStageMs) ? result.transformStageMs : 0
  };
}

export function summarizeGpuScreenTransformContext(context) {
  if (!context) {
    return {
      lastTransformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
      lastTransformRole: GPU_SCREEN_TRANSFORM_ROLE_FORMAL,
      lastSourceItemCount: 0,
      lastPackedCount: 0,
      lastPackedLength: 0,
      lastTransformStageMs: 0,
      lastFallbackReason: 'none',
      lastSummary: null
    };
  }

  return {
    lastTransformPath: context.lastTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    lastTransformRole: context.lastTransformRole ?? GPU_SCREEN_TRANSFORM_ROLE_FORMAL,
    lastSourceItemCount: Number.isFinite(context.lastSourceItemCount) ? context.lastSourceItemCount : 0,
    lastPackedCount: Number.isFinite(context.lastPackedCount) ? context.lastPackedCount : 0,
    lastPackedLength: Number.isFinite(context.lastPackedLength) ? context.lastPackedLength : 0,
    lastTransformStageMs: Number.isFinite(context.lastTransformStageMs) ? context.lastTransformStageMs : 0,
    lastFallbackReason: context.lastFallbackReason ?? 'none',
    lastSummary: context.lastSummary ?? null
  };
}
