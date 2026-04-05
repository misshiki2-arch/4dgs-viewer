import {
  getVisiblePackLayout,
  GPU_VISIBLE_PACK_FLOATS_PER_ITEM
} from './gpu_buffer_layout_utils.js';
import {
  packVisibleItems,
  createPackedVisibleResult
} from './gpu_visible_pack_utils.js';

// Step32
// 目的:
// - gpu-screen source provider の内部責務を分離する
// - packed formal draw contract は維持したまま、
//   gpu-screen source を
//   (1) source item build
//   (2) packed transform
//   の2段へ分ける
//
// 非目標:
// - draw contract の変更
// - UI/state/tile 層の変更
// - packed formal path の変更
//
// 設計:
// 1. packed-cpu は formal source のまま維持
// 2. packed-gpu-prep は experimental source として維持
// 3. packed-gpu-prep の内部では
//    - gpu-screen source items の生成
//    - source items -> packed formal contract 変換
//    を分離する
// 4. 既存 export 互換は維持する

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
    lastSourcePath: GPU_SCREEN_SPACE_SOURCE_PACKED_CPU,
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
  if (path === GPU_SCREEN_SPACE_SOURCE_PACKED_CPU) {
    return GPU_SCREEN_SPACE_SOURCE_PACKED_CPU;
  }
  if (path === GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP) {
    return GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP;
  }
  if (path === GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL) {
    return GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL;
  }
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
  const items = Array.isArray(sourceItemsResult?.items) ? sourceItemsResult.items : [];
  const packedResult = packVisibleItems(items);
  return {
    packed: packedResult?.packed instanceof Float32Array ? packedResult.packed : null,
    count: Number.isFinite(packedResult?.count) ? packedResult.count : items.length,
    floatsPerItem: Number.isFinite(packedResult?.floatsPerItem)
      ? packedResult.floatsPerItem
      : GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
    path: normalizeSourcePath(extra.path ?? sourceItemsResult?.path, !!extra.experimental || !!sourceItemsResult?.experimental),
    experimental: !!extra.experimental || !!sourceItemsResult?.experimental,
    sourceItems: items,
    sourceItemCount: Number.isFinite(sourceItemsResult?.itemCount) ? sourceItemsResult.itemCount : items.length,
    sourceSchemaVersion: Number.isFinite(sourceItemsResult?.schemaVersion)
      ? sourceItemsResult.schemaVersion
      : GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION
  };
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
  sourceSchemaVersion
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
  experimental
}) {
  const ref = buildReferenceInfo(path, experimental);
  return {
    sourcePath: ref.sourcePath,
    sourceRole: ref.sourceRole,
    sourceExperimental: !!experimental,
    sourceBuildMs: Number.isFinite(buildMs) ? buildMs : 0,
    sourcePackedCount: Number.isFinite(packedCount) ? packedCount : 0,
    sourcePackedLength: packed instanceof Float32Array ? packed.length : 0,
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
    sourceSchemaVersion
  });

  const comparisonSummary = buildPackedScreenSpaceComparisonSummary({
    path,
    packedCount,
    packed,
    buildMs,
    experimental
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
    pathNormalized: path
  };
}

function updateContext(context, result, inputVisible) {
  if (!context) return;
  context.layout = context.layout ?? getVisiblePackLayout();
  context.lastSourcePath = result.comparisonSummary?.sourcePath ?? result.path;
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
  const packedStageResult = packGpuScreenSourceItems(sourceItemsResult, extra);
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
  const packedStageResult = packGpuScreenSourceItems(sourceItemsResult, extra);
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

// formal source
export function buildPackedCpuScreenSpaceWithContext(context, visible, extra = {}) {
  return buildPackedScreenSpaceWithContext(context, visible, {
    ...extra,
    path: GPU_SCREEN_SPACE_SOURCE_PACKED_CPU,
    experimental: false
  });
}

// experimental source prep for future GPU-side generation
export function buildPackedGpuPrepScreenSpaceWithContext(context, visible, extra = {}) {
  return buildPackedScreenSpaceWithContext(context, visible, {
    ...extra,
    path: GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP,
    experimental: true
  });
}

// backward-compatible alias kept for existing callers
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
