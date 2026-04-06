import {
  getVisiblePackLayout,
  GPU_VISIBLE_PACK_FLOATS_PER_ITEM
} from './gpu_buffer_layout_utils.js';
import {
  createPackedVisibleResult
} from './gpu_visible_pack_utils.js';
import {
  createGpuScreenTransformContext,
  executeGpuScreenPackedTransform,
  summarizeGpuScreenTransformResult,
  GPU_SCREEN_TRANSFORM_PATH_CPU,
  GPU_SCREEN_TRANSFORM_PATH_GPU_PREP
} from './gpu_screen_transform_executor.js';

export const GPU_SCREEN_SPACE_SOURCE_PACKED_CPU = 'packed-cpu';
export const GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP = 'packed-gpu-prep';
export const GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL = 'gpu-screen-experimental';
export const GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION = 1;

function nowMs() {
  return performance.now();
}

function toFiniteOr(value, fallback = 0) {
  return Number.isFinite(value) ? value : fallback;
}

function normalizeCenterPx(item) {
  if (Array.isArray(item?.centerPx) && item.centerPx.length >= 2) {
    return [toFiniteOr(item.centerPx[0], 0), toFiniteOr(item.centerPx[1], 0)];
  }
  return [toFiniteOr(item?.px, 0), toFiniteOr(item?.py, 0)];
}

function normalizeRadiusPx(item) {
  if (Number.isFinite(item?.radiusPx)) return item.radiusPx;
  if (Number.isFinite(item?.radius)) return item.radius;
  return 0;
}

function normalizeDepth(item) {
  return toFiniteOr(item?.depth, 0);
}

function normalizeColorAlpha(item) {
  if (Array.isArray(item?.colorAlpha) && item.colorAlpha.length >= 4) {
    return [
      toFiniteOr(item.colorAlpha[0], 0),
      toFiniteOr(item.colorAlpha[1], 0),
      toFiniteOr(item.colorAlpha[2], 0),
      toFiniteOr(item.colorAlpha[3], 0)
    ];
  }

  const color = Array.isArray(item?.color) ? item.color : [0, 0, 0, 0];
  const alpha = Number.isFinite(item?.opacity) ? item.opacity : toFiniteOr(color[3], 0);
  return [
    toFiniteOr(color[0], 0),
    toFiniteOr(color[1], 0),
    toFiniteOr(color[2], 0),
    toFiniteOr(alpha, 0)
  ];
}

function normalizeConic(item) {
  if (Array.isArray(item?.conic) && item.conic.length >= 3) {
    return [
      toFiniteOr(item.conic[0], 0),
      toFiniteOr(item.conic[1], 0),
      toFiniteOr(item.conic[2], 0)
    ];
  }
  return [0, 0, 0];
}

function normalizeReserved(item) {
  return toFiniteOr(item?.reserved, 0);
}

function normalizeMisc(item) {
  if (Array.isArray(item?.misc) && item.misc.length >= 4) {
    return [
      toFiniteOr(item.misc[0], 0),
      toFiniteOr(item.misc[1], 0),
      toFiniteOr(item.misc[2], 0),
      toFiniteOr(item.misc[3], 0)
    ];
  }

  if (Array.isArray(item?.aabb) && item.aabb.length >= 4) {
    return [
      toFiniteOr(item.aabb[0], 0),
      toFiniteOr(item.aabb[1], 0),
      toFiniteOr(item.aabb[2], 0),
      toFiniteOr(item.aabb[3], 0)
    ];
  }

  return [0, 0, 0, 0];
}

export function normalizeScreenSpaceItem(item) {
  return {
    ...item,
    centerPx: normalizeCenterPx(item),
    radiusPx: normalizeRadiusPx(item),
    depth: normalizeDepth(item),
    colorAlpha: normalizeColorAlpha(item),
    conic: normalizeConic(item),
    reserved: normalizeReserved(item),
    misc: normalizeMisc(item)
  };
}

export function normalizeScreenSpaceVisible(visible) {
  if (!Array.isArray(visible) || visible.length === 0) return [];
  const normalized = new Array(visible.length);
  for (let i = 0; i < visible.length; i++) {
    normalized[i] = normalizeScreenSpaceItem(visible[i]);
  }
  return normalized;
}

export function createScreenSpaceBuildContext() {
  return {
    layout: getVisiblePackLayout(),
    transformContext: createGpuScreenTransformContext(),
    lastSourcePath: GPU_SCREEN_SPACE_SOURCE_PACKED_CPU,
    lastTransformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
    lastTransformRole: 'formal-transform',
    lastTransformFallbackReason: 'none',
    lastInputVisibleCount: 0,
    lastNormalizedVisibleCount: 0,
    lastSourceItemCount: 0,
    lastPackCount: 0,
    lastPackedLength: 0,
    lastPrepStageMs: 0,
    lastPackStageMs: 0,
    lastBuildMs: 0,
    lastSummary: null,
    lastComparisonSummary: null
  };
}

function isExperimentalSourcePath(path) {
  return (
    path === GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP ||
    path === GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL
  );
}

function normalizeSourcePath(path, experimental = false) {
  if (path === GPU_SCREEN_SPACE_SOURCE_PACKED_CPU) return GPU_SCREEN_SPACE_SOURCE_PACKED_CPU;
  if (path === GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP) return GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP;
  if (path === GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL) return GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL;
  return experimental ? GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP : GPU_SCREEN_SPACE_SOURCE_PACKED_CPU;
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

function getRequestedTransformPathForSource(path, experimental = false) {
  const normalizedPath = normalizeSourcePath(path, experimental);
  if (
    normalizedPath === GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP ||
    normalizedPath === GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL
  ) {
    return GPU_SCREEN_TRANSFORM_PATH_GPU_PREP;
  }
  return GPU_SCREEN_TRANSFORM_PATH_CPU;
}

export function buildGpuScreenSourceItems(normalizedVisible, extra = {}) {
  const sourceItems = Array.isArray(normalizedVisible) ? normalizedVisible.slice() : [];
  return {
    path: normalizeSourcePath(extra.path, !!extra.experimental),
    experimental: !!extra.experimental,
    schemaVersion: GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION,
    items: sourceItems,
    itemCount: sourceItems.length
  };
}

export function packGpuScreenSourceItems(sourceItemsResult, extra = {}) {
  const requestedPath = normalizeSourcePath(
    extra.path ?? sourceItemsResult?.path,
    !!extra.experimental || !!sourceItemsResult?.experimental
  );
  const transformContext = extra.transformContext ?? createGpuScreenTransformContext();
  const transformPath = extra.transformPath ?? getRequestedTransformPathForSource(
    requestedPath,
    !!extra.experimental || !!sourceItemsResult?.experimental
  );
  return executeGpuScreenPackedTransform(transformContext, sourceItemsResult, {
    transformPath
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
    transformPath: transformSummary?.transformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    transformRole: transformSummary?.transformRole ?? 'formal-transform',
    transformFallbackReason: transformSummary?.transformFallbackReason ?? 'none',
    transformStageMs: Number.isFinite(transformSummary?.transformStageMs) ? transformSummary.transformStageMs : 0,
    packedCount,
    packedLength: packed instanceof Float32Array ? packed.length : 0,
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
    sourcePackedLength: packed instanceof Float32Array ? packed.length : 0,
    sourceItemCount: sourceSummary?.sourceItemCount ?? 0,
    sourceSchemaVersion: sourceSummary?.sourceSchemaVersion ?? GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION,
    sourcePrepStageMs: Number.isFinite(sourceSummary?.prepStageMs) ? sourceSummary.prepStageMs : 0,
    sourcePackStageMs: Number.isFinite(sourceSummary?.packStageMs) ? sourceSummary.packStageMs : 0,
    transformPath: transformSummary?.transformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    transformRole: transformSummary?.transformRole ?? 'formal-transform',
    transformFallbackReason: transformSummary?.transformFallbackReason ?? 'none',
    transformStageMs: Number.isFinite(transformSummary?.transformStageMs) ? transformSummary.transformStageMs : 0,
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
  const experimental = !!extra.experimental;
  const buildMs = extra.buildMs;
  const prepStageMs = extra.prepStageMs;
  const packStageMs = extra.packStageMs;
  const sourceItems = Array.isArray(sourceItemsResult?.items) ? sourceItemsResult.items : [];
  const sourceSchemaVersion = Number.isFinite(sourceItemsResult?.schemaVersion)
    ? sourceItemsResult.schemaVersion
    : GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION;
  const transformSummary = summarizeGpuScreenTransformResult(packedStageResult);

  const stateSummary = buildPackedScreenSpaceStateSummary({
    path,
    inputVisible: extra.inputVisible,
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
  context.lastTransformPath = result.transformSummary?.transformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU;
  context.lastTransformRole = result.transformSummary?.transformRole ?? 'formal-transform';
  context.lastTransformFallbackReason = result.transformSummary?.transformFallbackReason ?? 'none';
  context.lastInputVisibleCount = Array.isArray(inputVisible) ? inputVisible.length : 0;
  context.lastNormalizedVisibleCount = Array.isArray(result.visible) ? result.visible.length : 0;
  context.lastSourceItemCount = Array.isArray(result.sourceItems) ? result.sourceItems.length : 0;
  context.lastPackCount = Number.isFinite(result.packedCount) ? result.packedCount : 0;
  context.lastPackedLength = result.packed instanceof Float32Array ? result.packed.length : 0;
  context.lastPrepStageMs = Number.isFinite(result.summary?.prepStageMs) ? result.summary.prepStageMs : 0;
  context.lastPackStageMs = Number.isFinite(result.summary?.packStageMs) ? result.summary.packStageMs : 0;
  context.lastBuildMs = Number.isFinite(result.summary?.buildMs) ? result.summary.buildMs : 0;
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
    experimental: false,
    transformPath: GPU_SCREEN_TRANSFORM_PATH_CPU
  });
}

export function buildPackedGpuPrepScreenSpaceWithContext(context, visible, extra = {}) {
  return buildPackedScreenSpaceWithContext(context, visible, {
    ...extra,
    path: GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP,
    experimental: true,
    transformPath: GPU_SCREEN_TRANSFORM_PATH_GPU_PREP
  });
}

export function buildGpuScreenExperimentalSpaceWithContext(context, visible, extra = {}) {
  return buildPackedGpuPrepScreenSpaceWithContext(context, visible, {
    ...extra,
    path: GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL,
    experimental: true,
    transformPath: GPU_SCREEN_TRANSFORM_PATH_GPU_PREP
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
      transformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
      transformRole: 'formal-transform',
      transformFallbackReason: 'none',
      transformStageMs: 0,
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
    transformPath: result.summary?.transformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    transformRole: result.summary?.transformRole ?? 'formal-transform',
    transformFallbackReason: result.summary?.transformFallbackReason ?? 'none',
    transformStageMs: Number.isFinite(result.summary?.transformStageMs) ? result.summary.transformStageMs : 0,
    packedCount: Number.isFinite(result.packedCount) ? result.packedCount : 0,
    packedLength: result.packed instanceof Float32Array ? result.packed.length : 0,
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
      transformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
      transformRole: 'formal-transform',
      transformFallbackReason: 'none',
      transformStageMs: 0,
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
    transformPath: result.comparisonSummary.transformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    transformRole: result.comparisonSummary.transformRole ?? 'formal-transform',
    transformFallbackReason: result.comparisonSummary.transformFallbackReason ?? 'none',
    transformStageMs: Number.isFinite(result.comparisonSummary.transformStageMs)
      ? result.comparisonSummary.transformStageMs
      : 0,
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
      lastTransformPath: GPU_SCREEN_TRANSFORM_PATH_CPU,
      lastTransformRole: 'formal-transform',
      lastTransformFallbackReason: 'none',
      lastInputVisibleCount: 0,
      lastNormalizedVisibleCount: 0,
      lastSourceItemCount: 0,
      lastPackCount: 0,
      lastPackedLength: 0,
      lastPrepStageMs: 0,
      lastPackStageMs: 0,
      lastBuildMs: 0,
      lastSummary: null,
      lastComparisonSummary: null
    };
  }

  return {
    lastSourcePath: context.lastSourcePath ?? 'none',
    lastTransformPath: context.lastTransformPath ?? GPU_SCREEN_TRANSFORM_PATH_CPU,
    lastTransformRole: context.lastTransformRole ?? 'formal-transform',
    lastTransformFallbackReason: context.lastTransformFallbackReason ?? 'none',
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
    lastSummary: context.lastSummary ?? null,
    lastComparisonSummary: context.lastComparisonSummary ?? null
  };
}
