import {
  getVisiblePackLayout,
  GPU_VISIBLE_PACK_FLOATS_PER_ITEM
} from './gpu_buffer_layout_utils.js';
import {
  GPU_SCREEN_SPACE_SOURCE_PACKED_CPU,
  GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP,
  GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL,
  GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION,
  normalizeScreenSpaceItem as normalizeScreenSpaceItemImpl,
  normalizeScreenSpaceVisible as normalizeScreenSpaceVisibleImpl,
  buildGpuScreenSourceItems as buildGpuScreenSourceItemsImpl
} from './gpu_visible_source_builder.js';
import {
  createGpuScreenTransformContext,
  executeGpuScreenPackedTransform,
  GPU_SCREEN_TRANSFORM_PATH_CPU,
} from './gpu_screen_transform_executor.js';

// Step34 redesign
// 目的:
// - transform executor を唯一の truth source とし、builder はその結果を保持して渡すだけにする
// - packed formal draw contract は維持したまま、source-item build と transform executor の責務分離を固定する
//
// builder の責務:
// - visible -> normalized source items
// - source path の決定
// - transform executor 呼び出し
// - executor が返した summary / comparison summary / transformBatchSummary の保持
//
// builder の非責務:
// - requested / actual transform path の再解釈
// - fallback reason の補完
// - transform state の推測
// - transformBatchSummary の再計算や補正
//
// 重要:
// - requestedTransformPath / actualTransformPath / fallback / configured / upload stats の truth source は
//   gpu_screen_transform_executor.js のみ

export {
  GPU_SCREEN_SPACE_SOURCE_PACKED_CPU,
  GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP,
  GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL,
  GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION
};

function nowMs() {
  return performance.now();
}

function getPackedLogicalLength(packed, packedCount, floatsPerItem) {
  if (packed instanceof Float32Array) return packed.length;
  if (!Number.isFinite(packedCount) || !Number.isFinite(floatsPerItem)) return 0;
  return Math.max(0, (packedCount | 0) * (floatsPerItem | 0));
}

export function normalizeScreenSpaceItem(item) {
  return normalizeScreenSpaceItemImpl(item);
}

export function normalizeScreenSpaceVisible(visible) {
  return normalizeScreenSpaceVisibleImpl(visible);
}

function normalizeSourcePath(path, experimental = false) {
  if (path === GPU_SCREEN_SPACE_SOURCE_PACKED_CPU) return GPU_SCREEN_SPACE_SOURCE_PACKED_CPU;
  if (path === GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP) return GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP;
  if (path === GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL) return GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL;
  return experimental ? GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP : GPU_SCREEN_SPACE_SOURCE_PACKED_CPU;
}

function isExperimentalSourcePath(path) {
  return (
    path === GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP ||
    path === GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL
  );
}

function buildReferenceInfo(path, experimental) {
  const normalizedPath = normalizeSourcePath(path, experimental);
  if (normalizedPath === GPU_SCREEN_SPACE_SOURCE_PACKED_CPU) {
    return {
      sourcePath: GPU_SCREEN_SPACE_SOURCE_PACKED_CPU,
      sourceRole: 'formal-source',
      referencePath: GPU_SCREEN_SPACE_SOURCE_PACKED_CPU,
      referenceRole: 'formal-reference'
    };
  }

  return {
    sourcePath: normalizedPath,
    sourceRole: isExperimentalSourcePath(normalizedPath) ? 'experimental-source' : 'formal-source',
    referencePath: GPU_SCREEN_SPACE_SOURCE_PACKED_CPU,
    referenceRole: 'formal-reference'
  };
}

export function createScreenSpaceBuildContext() {
  return {
    layout: getVisiblePackLayout(),
    transformContext: createGpuScreenTransformContext(),
    lastSourcePath: GPU_SCREEN_SPACE_SOURCE_PACKED_CPU,
    lastRequestedTransformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
    lastActualTransformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
    lastTransformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
    lastTransformRole: 'formal-transform',
    lastTransformConfigured: false,
    lastTransformHasBuffers: false,
    lastTransformFallbackReason: 'none',
    lastTransformUploadBytes: 0,
    lastTransformUploadCount: 0,
    lastTransformUploadLength: 0,
    lastTransformUploadCapacityBytes: 0,
    lastTransformUploadReusedCapacity: false,
    lastInputVisibleCount: 0,
    lastNormalizedVisibleCount: 0,
    lastSourceItemCount: 0,
    lastPackCount: 0,
    lastPackedLength: 0,
    lastPrepStageMs: 0,
    lastPackStageMs: 0,
    lastBuildMs: 0,
    lastTransformBatchSummary: null,
    lastSummary: null,
    lastComparisonSummary: null
  };
}

export function buildGpuScreenSourceItems(normalizedVisible, extra = {}) {
  return buildGpuScreenSourceItemsImpl(normalizedVisible, extra);
}

export function packGpuScreenSourceItems(sourceItemsResult, extra = {}) {
  const transformContext = extra.transformContext ?? createGpuScreenTransformContext();

  return executeGpuScreenPackedTransform(transformContext, sourceItemsResult, {
    gl: extra.gl ?? null,
    sourcePath: extra.path ?? sourceItemsResult?.path,
    experimental: !!extra.experimental || !!sourceItemsResult?.experimental
  });
}

function buildPackedScreenSpaceStateSummary({
  path,
  inputVisible,
  normalizedVisible,
  sourceItems,
  packed,
  packedCount,
  floatsPerItem,
  prepStageMs,
  packStageMs,
  buildMs,
  experimental,
  sourceSchemaVersion,
  transformSummary
}) {
  const normalizedPath = normalizeSourcePath(path, experimental);

  return {
    path: normalizedPath,
    inputVisibleCount: Array.isArray(inputVisible) ? inputVisible.length : 0,
    normalizedVisibleCount: Array.isArray(normalizedVisible) ? normalizedVisible.length : 0,
    sourceItemCount: Array.isArray(sourceItems) ? sourceItems.length : 0,
    sourceSchemaVersion: Number.isFinite(sourceSchemaVersion)
      ? sourceSchemaVersion
      : GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION,
    prepStageMs: Number.isFinite(prepStageMs) ? prepStageMs : 0,
    packStageMs: Number.isFinite(packStageMs) ? packStageMs : 0,

    requestedTransformPath: transformSummary?.requestedTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    actualTransformPath: transformSummary?.actualTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    transformPath: transformSummary?.transformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    transformRole: transformSummary?.transformRole ?? 'formal-transform',
    transformConfigured: !!transformSummary?.transformConfigured,
    transformHasBuffers: !!transformSummary?.transformHasBuffers,
    transformFallbackReason: transformSummary?.transformFallbackReason ?? 'none',
    transformStageMs: Number.isFinite(transformSummary?.transformStageMs) ? transformSummary.transformStageMs : 0,
    transformUploadBytes: Number.isFinite(transformSummary?.transformUploadBytes) ? transformSummary.transformUploadBytes : 0,
    transformUploadCount: Number.isFinite(transformSummary?.transformUploadCount) ? transformSummary.transformUploadCount : 0,
    transformUploadLength: Number.isFinite(transformSummary?.transformUploadLength) ? transformSummary.transformUploadLength : 0,
    transformUploadCapacityBytes: Number.isFinite(transformSummary?.transformUploadCapacityBytes) ? transformSummary.transformUploadCapacityBytes : 0,
    transformUploadReusedCapacity: !!transformSummary?.transformUploadReusedCapacity,
    transformBatchSummary: transformSummary?.transformBatchSummary ?? null,

    packedCount,
    packedLength: getPackedLogicalLength(packed, packedCount, floatsPerItem),
    floatsPerItem,
    alphaSource: 'colorAlpha[3]',
    centerSource: 'centerPx',
    radiusSource: 'radiusPx',
    depthSource: 'depth',
    conicSource: 'conic',
    miscSource: 'misc',
    buildMs: Number.isFinite(buildMs) ? buildMs : 0,
    experimental: !!experimental
  };
}

function buildPackedScreenSpaceComparisonSummary({
  path,
  packedCount,
  packed,
  buildMs,
  experimental,
  sourceSummary,
  transformSummary
}) {
  const ref = buildReferenceInfo(path, experimental);
  return {
    sourcePath: ref.sourcePath,
    sourceRole: ref.sourceRole,
    sourceExperimental: !!experimental,
    sourceBuildMs: Number.isFinite(buildMs) ? buildMs : 0,
    sourcePackedCount: Number.isFinite(packedCount) ? packedCount : 0,
    sourcePackedLength: getPackedLogicalLength(packed, packedCount, transformSummary?.floatsPerItem ?? GPU_VISIBLE_PACK_FLOATS_PER_ITEM),

    sourceItemCount: sourceSummary?.sourceItemCount ?? 0,
    sourceSchemaVersion: sourceSummary?.sourceSchemaVersion ?? GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION,
    sourcePrepStageMs: Number.isFinite(sourceSummary?.prepStageMs) ? sourceSummary.prepStageMs : 0,
    sourcePackStageMs: Number.isFinite(sourceSummary?.packStageMs) ? sourceSummary.packStageMs : 0,

    requestedTransformPath: transformSummary?.requestedTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    actualTransformPath: transformSummary?.actualTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    transformPath: transformSummary?.transformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    transformRole: transformSummary?.transformRole ?? 'formal-transform',
    transformConfigured: !!transformSummary?.transformConfigured,
    transformHasBuffers: !!transformSummary?.transformHasBuffers,
    transformFallbackReason: transformSummary?.transformFallbackReason ?? 'none',
    transformStageMs: Number.isFinite(transformSummary?.transformStageMs) ? transformSummary.transformStageMs : 0,
    transformUploadBytes: Number.isFinite(transformSummary?.transformUploadBytes) ? transformSummary.transformUploadBytes : 0,
    transformUploadCount: Number.isFinite(transformSummary?.transformUploadCount) ? transformSummary.transformUploadCount : 0,
    transformUploadLength: Number.isFinite(transformSummary?.transformUploadLength) ? transformSummary.transformUploadLength : 0,
    transformUploadCapacityBytes: Number.isFinite(transformSummary?.transformUploadCapacityBytes) ? transformSummary.transformUploadCapacityBytes : 0,
    transformUploadReusedCapacity: !!transformSummary?.transformUploadReusedCapacity,
    transformBatchSummary: transformSummary?.transformBatchSummary ?? null,

    referencePath: ref.referencePath,
    referenceRole: ref.referenceRole,
    usesPackedReferenceLayout: ref.referencePath === GPU_SCREEN_SPACE_SOURCE_PACKED_CPU,
    usesPackedReferencePack: ref.referencePath === GPU_SCREEN_SPACE_SOURCE_PACKED_CPU,
    sameLayoutAsReference: true,
    samePackCountAsReference: true
  };
}

function buildPackedScreenSpaceResult(normalizedVisible, sourceItemsResult, packedStageResult, extra = {}) {
  const packed = packedStageResult?.packed instanceof Float32Array ? packedStageResult.packed : null;
  const packedCount = Number.isFinite(packedStageResult?.count) ? packedStageResult.count : 0;
  const floatsPerItem = Number.isFinite(packedStageResult?.floatsPerItem)
    ? packedStageResult.floatsPerItem
    : GPU_VISIBLE_PACK_FLOATS_PER_ITEM;
  const path = normalizeSourcePath(extra.path, !!extra.experimental);
  const experimental = !!extra.experimental || isExperimentalSourcePath(path);
  const buildMs = extra.buildMs;
  const prepStageMs = extra.prepStageMs;
  const packStageMs = extra.packStageMs;
  const sourceItems = Array.isArray(sourceItemsResult?.items) ? sourceItemsResult.items : [];
  const sourceSchemaVersion = Number.isFinite(sourceItemsResult?.schemaVersion)
    ? sourceItemsResult.schemaVersion
    : GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION;
  const transformSummary = packedStageResult?.summary ?? null;
  const gpuPackedPayloads = Array.isArray(packedStageResult?.gpuPackedPayloads)
    ? packedStageResult.gpuPackedPayloads
    : [];

  const stateSummary = buildPackedScreenSpaceStateSummary({
    path,
    inputVisible: extra.inputVisible,
    normalizedVisible,
    sourceItems,
    packed,
    gpuPackedPayloads,
    packedCount,
    floatsPerItem,
    prepStageMs,
    packStageMs,
    buildMs,
    experimental,
    sourceSchemaVersion,
    transformSummary
  });

  const comparisonSummary = buildPackedScreenSpaceComparisonSummary({
    path,
    packedCount,
    packed,
    buildMs,
    experimental,
    sourceSummary: stateSummary,
    transformSummary
  });

  return {
    path,
    visible: normalizedVisible,
    sourceItems,
    packed,
    gpuPackedPayloads,
    packedCount,
    floatsPerItem,
    layout: getVisiblePackLayout(),
    summary: stateSummary,
    comparisonSummary,
    experimental,
    ...extra,
    transformSummary,
    pathNormalized: path
  };
}

function updateContext(context, result, inputVisible) {
  if (!context) return;
  context.layout = context.layout ?? getVisiblePackLayout();
  context.lastSourcePath = result.comparisonSummary?.sourcePath ?? result.path;
  context.lastRequestedTransformPath = result.transformSummary?.requestedTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU;
  context.lastActualTransformPath = result.transformSummary?.actualTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU;
  context.lastTransformPath = result.transformSummary?.transformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU;
  context.lastTransformRole = result.transformSummary?.transformRole ?? 'formal-transform';
  context.lastTransformConfigured = !!result.transformSummary?.transformConfigured;
  context.lastTransformHasBuffers = !!result.transformSummary?.transformHasBuffers;
  context.lastTransformFallbackReason = result.transformSummary?.transformFallbackReason ?? 'none';
  context.lastTransformUploadBytes = Number.isFinite(result.transformSummary?.transformUploadBytes) ? result.transformSummary.transformUploadBytes : 0;
  context.lastTransformUploadCount = Number.isFinite(result.transformSummary?.transformUploadCount) ? result.transformSummary.transformUploadCount : 0;
  context.lastTransformUploadLength = Number.isFinite(result.transformSummary?.transformUploadLength) ? result.transformSummary.transformUploadLength : 0;
  context.lastTransformUploadCapacityBytes = Number.isFinite(result.transformSummary?.transformUploadCapacityBytes) ? result.transformSummary.transformUploadCapacityBytes : 0;
  context.lastTransformUploadReusedCapacity = !!result.transformSummary?.transformUploadReusedCapacity;
  context.lastInputVisibleCount = Array.isArray(inputVisible) ? inputVisible.length : 0;
  context.lastNormalizedVisibleCount = Array.isArray(result.visible) ? result.visible.length : 0;
  context.lastSourceItemCount = Array.isArray(result.sourceItems) ? result.sourceItems.length : 0;
  context.lastPackCount = Number.isFinite(result.packedCount) ? result.packedCount : 0;
  context.lastPackedLength = getPackedLogicalLength(result.packed, result.packedCount, result.floatsPerItem);
  context.lastPrepStageMs = Number.isFinite(result.summary?.prepStageMs) ? result.summary.prepStageMs : 0;
  context.lastPackStageMs = Number.isFinite(result.summary?.packStageMs) ? result.summary.packStageMs : 0;
  context.lastBuildMs = Number.isFinite(result.summary?.buildMs) ? result.summary.buildMs : 0;
  context.lastTransformBatchSummary = result.transformSummary?.transformBatchSummary ?? null;
  context.lastSummary = result.summary;
  context.lastComparisonSummary = result.comparisonSummary ?? null;
}

export function buildPackedScreenSpaceFromVisible(visible, extra = {}) {
  const t0 = nowMs();
  const normalizedVisible = normalizeScreenSpaceVisible(visible);

  const prepT0 = nowMs();
  const sourceItemsResult = buildGpuScreenSourceItems(normalizedVisible, extra);
  const prepStageMs = nowMs() - prepT0;

  const packT0 = nowMs();
  const packedStageResult = packGpuScreenSourceItems(sourceItemsResult, {
    ...extra,
    transformContext: createGpuScreenTransformContext()
  });
  const packStageMs = nowMs() - packT0;

  const buildMs = nowMs() - t0;

  return buildPackedScreenSpaceResult(normalizedVisible, sourceItemsResult, packedStageResult, {
    ...extra,
    inputVisible: visible,
    prepStageMs,
    packStageMs,
    buildMs,
    path: normalizeSourcePath(extra.path, !!extra.experimental),
    experimental: !!extra.experimental
  });
}

export function buildPackedScreenSpaceWithContext(context, visible, extra = {}) {
  const t0 = nowMs();
  const normalizedVisible = normalizeScreenSpaceVisible(visible);

  const prepT0 = nowMs();
  const sourceItemsResult = buildGpuScreenSourceItems(normalizedVisible, extra);
  const prepStageMs = nowMs() - prepT0;

  const packT0 = nowMs();
  const packedStageResult = packGpuScreenSourceItems(sourceItemsResult, {
    ...extra,
    transformContext: context?.transformContext ?? createGpuScreenTransformContext()
  });
  const packStageMs = nowMs() - packT0;

  const buildMs = nowMs() - t0;

  const result = buildPackedScreenSpaceResult(normalizedVisible, sourceItemsResult, packedStageResult, {
    ...extra,
    inputVisible: visible,
    prepStageMs,
    packStageMs,
    buildMs,
    path: normalizeSourcePath(extra.path, !!extra.experimental),
    experimental: !!extra.experimental
  });

  updateContext(context, result, visible);
  return {
    ...result,
    layout: context?.layout ?? getVisiblePackLayout()
  };
}

export function buildPackedCpuScreenSpaceWithContext(context, visible, extra = {}) {
  return buildPackedScreenSpaceWithContext(context, visible, {
    ...extra,
    path: GPU_SCREEN_SPACE_SOURCE_PACKED_CPU,
    experimental: false
  });
}

export function buildPackedGpuPrepScreenSpaceWithContext(context, visible, extra = {}) {
  return buildPackedScreenSpaceWithContext(context, visible, {
    ...extra,
    path: GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP,
    experimental: true
  });
}

export function buildGpuScreenExperimentalSpaceWithContext(context, visible, extra = {}) {
  return buildPackedGpuPrepScreenSpaceWithContext(context, visible, {
    ...extra,
    path: GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL,
    experimental: true
  });
}

export function summarizePackedScreenSpace(result) {
  if (!result) {
    return {
      path: 'none',
      sourceItemCount: 0,
      sourceSchemaVersion: GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION,
      prepStageMs: 0,
      packStageMs: 0,

      requestedTransformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
      actualTransformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
      transformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
      transformRole: 'formal-transform',
      transformConfigured: false,
      transformHasBuffers: false,
      transformFallbackReason: 'none',
      transformStageMs: 0,
      transformUploadBytes: 0,
      transformUploadCount: 0,
      transformUploadLength: 0,
      transformUploadCapacityBytes: 0,
      transformUploadReusedCapacity: false,
      transformBatchSummary: null,

      packedCount: 0,
      packedLength: 0,
      floatsPerItem: GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
      alphaSource: 'colorAlpha[3]',
      centerSource: 'centerPx',
      radiusSource: 'radiusPx',
      depthSource: 'depth',
      conicSource: 'conic',
      miscSource: 'misc',
      buildMs: 0,
      experimental: false
    };
  }

  return {
    path: result.summary?.path ?? result.path ?? 'unknown',
    sourceItemCount: Number.isFinite(result.summary?.sourceItemCount) ? result.summary.sourceItemCount : 0,
    sourceSchemaVersion: Number.isFinite(result.summary?.sourceSchemaVersion)
      ? result.summary.sourceSchemaVersion
      : GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION,
    prepStageMs: Number.isFinite(result.summary?.prepStageMs) ? result.summary.prepStageMs : 0,
    packStageMs: Number.isFinite(result.summary?.packStageMs) ? result.summary.packStageMs : 0,

    requestedTransformPath: result.transformSummary?.requestedTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    actualTransformPath: result.transformSummary?.actualTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    transformPath: result.transformSummary?.transformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    transformRole: result.transformSummary?.transformRole ?? 'formal-transform',
    transformConfigured: !!result.transformSummary?.transformConfigured,
    transformHasBuffers: !!result.transformSummary?.transformHasBuffers,
    transformFallbackReason: result.transformSummary?.transformFallbackReason ?? 'none',
    transformStageMs: Number.isFinite(result.transformSummary?.transformStageMs) ? result.transformSummary.transformStageMs : 0,
    transformUploadBytes: Number.isFinite(result.transformSummary?.transformUploadBytes) ? result.transformSummary.transformUploadBytes : 0,
    transformUploadCount: Number.isFinite(result.transformSummary?.transformUploadCount) ? result.transformSummary.transformUploadCount : 0,
    transformUploadLength: Number.isFinite(result.transformSummary?.transformUploadLength) ? result.transformSummary.transformUploadLength : 0,
    transformUploadCapacityBytes: Number.isFinite(result.transformSummary?.transformUploadCapacityBytes) ? result.transformSummary.transformUploadCapacityBytes : 0,
    transformUploadReusedCapacity: !!result.transformSummary?.transformUploadReusedCapacity,
    transformBatchSummary: result.transformSummary?.transformBatchSummary ?? null,

    packedCount: Number.isFinite(result.packedCount) ? result.packedCount : 0,
    packedLength: getPackedLogicalLength(result.packed, result.packedCount, result.floatsPerItem),
    floatsPerItem: Number.isFinite(result.floatsPerItem)
      ? result.floatsPerItem
      : GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
    alphaSource: result.summary?.alphaSource ?? 'colorAlpha[3]',
    centerSource: result.summary?.centerSource ?? 'centerPx',
    radiusSource: result.summary?.radiusSource ?? 'radiusPx',
    depthSource: result.summary?.depthSource ?? 'depth',
    conicSource: result.summary?.conicSource ?? 'conic',
    miscSource: result.summary?.miscSource ?? 'misc',
    buildMs: Number.isFinite(result.summary?.buildMs) ? result.summary.buildMs : 0,
    experimental: !!result.summary?.experimental
  };
}

export function summarizePackedScreenSpaceComparison(result) {
  if (!result?.comparisonSummary) {
    return {
      sourcePath: 'none',
      sourceRole: 'none',
      sourceExperimental: false,
      sourceBuildMs: 0,
      sourcePackedCount: 0,
      sourcePackedLength: 0,
      sourceItemCount: 0,
      sourceSchemaVersion: GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION,
      sourcePrepStageMs: 0,
      sourcePackStageMs: 0,

      requestedTransformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
      actualTransformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
      transformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
      transformRole: 'formal-transform',
      transformConfigured: false,
      transformHasBuffers: false,
      transformFallbackReason: 'none',
      transformStageMs: 0,
      transformUploadBytes: 0,
      transformUploadCount: 0,
      transformUploadLength: 0,
      transformUploadCapacityBytes: 0,
      transformUploadReusedCapacity: false,
      transformBatchSummary: null,

      referencePath: 'none',
      referenceRole: 'none',
      usesPackedReferenceLayout: false,
      usesPackedReferencePack: false,
      sameLayoutAsReference: false,
      samePackCountAsReference: false
    };
  }

  return {
    sourcePath: result.comparisonSummary.sourcePath ?? 'none',
    sourceRole: result.comparisonSummary.sourceRole ?? 'none',
    sourceExperimental: !!result.comparisonSummary.sourceExperimental,
    sourceBuildMs: Number.isFinite(result.comparisonSummary.sourceBuildMs)
      ? result.comparisonSummary.sourceBuildMs
      : 0,
    sourcePackedCount: Number.isFinite(result.comparisonSummary.sourcePackedCount)
      ? result.comparisonSummary.sourcePackedCount
      : 0,
    sourcePackedLength: Number.isFinite(result.comparisonSummary.sourcePackedLength)
      ? result.comparisonSummary.sourcePackedLength
      : 0,
    sourceItemCount: Number.isFinite(result.comparisonSummary.sourceItemCount)
      ? result.comparisonSummary.sourceItemCount
      : 0,
    sourceSchemaVersion: Number.isFinite(result.comparisonSummary.sourceSchemaVersion)
      ? result.comparisonSummary.sourceSchemaVersion
      : GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION,
    sourcePrepStageMs: Number.isFinite(result.comparisonSummary.sourcePrepStageMs)
      ? result.comparisonSummary.sourcePrepStageMs
      : 0,
    sourcePackStageMs: Number.isFinite(result.comparisonSummary.sourcePackStageMs)
      ? result.comparisonSummary.sourcePackStageMs
      : 0,

    requestedTransformPath: result.comparisonSummary.requestedTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    actualTransformPath: result.comparisonSummary.actualTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    transformPath: result.comparisonSummary.transformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    transformRole: result.comparisonSummary.transformRole ?? 'formal-transform',
    transformConfigured: !!result.comparisonSummary.transformConfigured,
    transformHasBuffers: !!result.comparisonSummary.transformHasBuffers,
    transformFallbackReason: result.comparisonSummary.transformFallbackReason ?? 'none',
    transformStageMs: Number.isFinite(result.comparisonSummary.transformStageMs)
      ? result.comparisonSummary.transformStageMs
      : 0,
    transformUploadBytes: Number.isFinite(result.comparisonSummary.transformUploadBytes)
      ? result.comparisonSummary.transformUploadBytes
      : 0,
    transformUploadCount: Number.isFinite(result.comparisonSummary.transformUploadCount)
      ? result.comparisonSummary.transformUploadCount
      : 0,
    transformUploadLength: Number.isFinite(result.comparisonSummary.transformUploadLength)
      ? result.comparisonSummary.transformUploadLength
      : 0,
    transformUploadCapacityBytes: Number.isFinite(result.comparisonSummary.transformUploadCapacityBytes)
      ? result.comparisonSummary.transformUploadCapacityBytes
      : 0,
    transformUploadReusedCapacity: !!result.comparisonSummary.transformUploadReusedCapacity,
    transformBatchSummary: result.comparisonSummary.transformBatchSummary ?? null,

    referencePath: result.comparisonSummary.referencePath ?? 'none',
    referenceRole: result.comparisonSummary.referenceRole ?? 'none',
    usesPackedReferenceLayout: !!result.comparisonSummary.usesPackedReferenceLayout,
    usesPackedReferencePack: !!result.comparisonSummary.usesPackedReferencePack,
    sameLayoutAsReference: !!result.comparisonSummary.sameLayoutAsReference,
    samePackCountAsReference: !!result.comparisonSummary.samePackCountAsReference
  };
}

export function summarizeScreenSpaceBuildContext(context) {
  if (!context) {
    return {
      lastSourcePath: 'none',
      lastRequestedTransformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
      lastActualTransformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
      lastTransformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
      lastTransformRole: 'formal-transform',
      lastTransformConfigured: false,
      lastTransformHasBuffers: false,
      lastTransformFallbackReason: 'none',
      lastTransformUploadBytes: 0,
      lastTransformUploadCount: 0,
      lastTransformUploadLength: 0,
      lastTransformUploadCapacityBytes: 0,
      lastTransformUploadReusedCapacity: false,
      lastInputVisibleCount: 0,
      lastNormalizedVisibleCount: 0,
      lastSourceItemCount: 0,
      lastPackCount: 0,
      lastPackedLength: 0,
      lastPrepStageMs: 0,
      lastPackStageMs: 0,
      lastBuildMs: 0,
      lastTransformBatchSummary: null,
      lastSummary: null,
      lastComparisonSummary: null
    };
  }

  return {
    lastSourcePath: context.lastSourcePath ?? 'none',
    lastRequestedTransformPath: context.lastRequestedTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    lastActualTransformPath: context.lastActualTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    lastTransformPath: context.lastTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    lastTransformRole: context.lastTransformRole ?? 'formal-transform',
    lastTransformConfigured: !!context.lastTransformConfigured,
    lastTransformHasBuffers: !!context.lastTransformHasBuffers,
    lastTransformFallbackReason: context.lastTransformFallbackReason ?? 'none',
    lastTransformUploadBytes: Number.isFinite(context.lastTransformUploadBytes) ? context.lastTransformUploadBytes : 0,
    lastTransformUploadCount: Number.isFinite(context.lastTransformUploadCount) ? context.lastTransformUploadCount : 0,
    lastTransformUploadLength: Number.isFinite(context.lastTransformUploadLength) ? context.lastTransformUploadLength : 0,
    lastTransformUploadCapacityBytes: Number.isFinite(context.lastTransformUploadCapacityBytes) ? context.lastTransformUploadCapacityBytes : 0,
    lastTransformUploadReusedCapacity: !!context.lastTransformUploadReusedCapacity,
    lastInputVisibleCount: Number.isFinite(context.lastInputVisibleCount)
      ? context.lastInputVisibleCount
      : 0,
    lastNormalizedVisibleCount: Number.isFinite(context.lastNormalizedVisibleCount)
      ? context.lastNormalizedVisibleCount
      : 0,
    lastSourceItemCount: Number.isFinite(context.lastSourceItemCount)
      ? context.lastSourceItemCount
      : 0,
    lastPackCount: Number.isFinite(context.lastPackCount) ? context.lastPackCount : 0,
    lastPackedLength: Number.isFinite(context.lastPackedLength) ? context.lastPackedLength : 0,
    lastPrepStageMs: Number.isFinite(context.lastPrepStageMs) ? context.lastPrepStageMs : 0,
    lastPackStageMs: Number.isFinite(context.lastPackStageMs) ? context.lastPackStageMs : 0,
    lastBuildMs: Number.isFinite(context.lastBuildMs) ? context.lastBuildMs : 0,
    lastTransformBatchSummary: context.lastTransformBatchSummary ?? null,
    lastSummary: context.lastSummary ?? null,
    lastComparisonSummary: context.lastComparisonSummary ?? null
  };
}
