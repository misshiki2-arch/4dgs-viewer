import {
  getVisiblePackLayout,
  GPU_VISIBLE_PACK_FLOATS_PER_ITEM
} from './gpu_buffer_layout_utils.js';
import {
  packVisibleItems,
  createPackedVisibleResult
} from './gpu_visible_pack_utils.js';

// Step31
// 目的:
// - Step30 で分離した gpu-screen draw ownership の次として、source ownership separation を始める
// - packed formal draw contract は一切変えず、source provider だけを明示化する
// - packed-cpu を formal source、packed-gpu-prep を experimental source として扱う
//
// 非目標:
// - draw contract の変更
// - UI/state/tile 層の変更
// - renderer 側での意味変更
//
// 設計:
// 1. source provider 名を定数化する
// 2. packed-cpu は formal source のまま維持する
// 3. packed-gpu-prep は experimental source の入口として追加する
// 4. 旧 export buildGpuScreenExperimentalSpaceWithContext は互換 alias として残す

export const GPU_SCREEN_SPACE_SOURCE_PACKED_CPU = 'packed-cpu';
export const GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP = 'packed-gpu-prep';
export const GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL = 'gpu-screen-experimental';

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
    lastPackCount: 0,
    lastPackedLength: 0,
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

function buildPackedScreenSpaceStateSummary({
  path,
  inputVisible,
  normalizedVisible,
  packed,
  packedCount,
  floatsPerItem,
  buildMs,
  experimental
}) {
  const normalizedPath = normalizeSourcePath(path, experimental);
  return {
    path: normalizedPath,
    inputVisibleCount: Array.isArray(inputVisible) ? inputVisible.length : 0,
    normalizedVisibleCount: Array.isArray(normalizedVisible) ? normalizedVisible.length : 0,
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

function buildPackedScreenSpaceResult(normalizedVisible, packedResult, extra = {}) {
  const packed = packedResult?.packed instanceof Float32Array ? packedResult.packed : null;
  const packedCount = Number.isFinite(packedResult?.count) ? packedResult.count : 0;
  const floatsPerItem = Number.isFinite(packedResult?.floatsPerItem)
    ? packedResult.floatsPerItem
    : GPU_VISIBLE_PACK_FLOATS_PER_ITEM;
  const path = normalizeSourcePath(extra.path, !!extra.experimental);
  const experimental = !!extra.experimental;
  const buildMs = extra.buildMs;

  const stateSummary = buildPackedScreenSpaceStateSummary({
    path,
    inputVisible: extra.inputVisible,
    normalizedVisible,
    packed,
    packedCount,
    floatsPerItem,
    buildMs,
    experimental
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
  context.lastPackCount = Number.isFinite(result.packedCount) ? result.packedCount : 0;
  context.lastPackedLength = result.packed instanceof Float32Array ? result.packed.length : 0;
  context.lastBuildMs = Number.isFinite(result.summary?.buildMs) ? result.summary.buildMs : 0;
  context.lastSummary = result.summary;
  context.lastComparisonSummary = result.comparisonSummary ?? null;
}

export function buildPackedScreenSpaceFromVisible(visible, extra = {}) {
  const t0 = nowMs();
  const normalizedVisible = normalizeScreenSpaceVisible(visible);
  const packedResult = createPackedVisibleResult(normalizedVisible, extra);
  const buildMs = nowMs() - t0;

  return buildPackedScreenSpaceResult(normalizedVisible, packedResult, {
    ...extra,
    inputVisible: visible,
    buildMs,
    path: normalizeSourcePath(extra.path, !!extra.experimental),
    experimental: !!extra.experimental
  });
}

export function buildPackedScreenSpaceWithContext(context, visible, extra = {}) {
  const t0 = nowMs();
  const normalizedVisible = normalizeScreenSpaceVisible(visible);
  const packedResult = packVisibleItems(normalizedVisible);
  const buildMs = nowMs() - t0;

  const result = buildPackedScreenSpaceResult(normalizedVisible, packedResult, {
    ...extra,
    inputVisible: visible,
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
      lastPackCount: 0,
      lastPackedLength: 0,
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
    lastPackCount: Number.isFinite(context.lastPackCount) ? context.lastPackCount : 0,
    lastPackedLength: Number.isFinite(context.lastPackedLength) ? context.lastPackedLength : 0,
    lastBuildMs: Number.isFinite(context.lastBuildMs) ? context.lastBuildMs : 0,
    lastSummary: context.lastSummary ?? null,
    lastComparisonSummary: context.lastComparisonSummary ?? null
  };
}
