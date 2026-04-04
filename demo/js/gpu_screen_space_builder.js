import {
  getVisiblePackLayout,
  GPU_VISIBLE_PACK_FLOATS_PER_ITEM
} from './gpu_buffer_layout_utils.js';
import {
  packVisibleItems,
  createPackedVisibleResult
} from './gpu_visible_pack_utils.js';

// Step28:
// screen-space 結果を packed layout の正式契約へ正規化して渡す場所。
// Step27 では packed formal reference / gpu-screen experimental の比較用 summary を持たせた。
// Step28 ではさらに、summary を
// - state 的な基本情報
// - comparison 的な参照関係
// に分けて扱いやすくする。
// packed 内容自体はまだ共通でもよいが、debug 側が重複を減らせる形へ整える。

function nowMs() {
  return performance.now();
}

function toFiniteOr(value, fallback = 0) {
  return Number.isFinite(value) ? value : fallback;
}

function normalizeCenterPx(item) {
  if (Array.isArray(item?.centerPx) && item.centerPx.length >= 2) {
    return [
      toFiniteOr(item.centerPx[0], 0),
      toFiniteOr(item.centerPx[1], 0)
    ];
  }

  return [
    toFiniteOr(item?.px, 0),
    toFiniteOr(item?.py, 0)
  ];
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
  const alpha = Number.isFinite(item?.opacity)
    ? item.opacity
    : toFiniteOr(color[3], 0);

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
  if (!Array.isArray(visible) || visible.length === 0) {
    return [];
  }

  const normalized = new Array(visible.length);
  for (let i = 0; i < visible.length; i++) {
    normalized[i] = normalizeScreenSpaceItem(visible[i]);
  }
  return normalized;
}

export function createScreenSpaceBuildContext() {
  return {
    layout: getVisiblePackLayout(),

    lastPath: 'packed-cpu',
    lastInputVisibleCount: 0,
    lastNormalizedVisibleCount: 0,
    lastPackCount: 0,
    lastPackedLength: 0,
    lastBuildMs: 0,

    lastSummary: null,
    lastComparisonSummary: null
  };
}

function buildReferenceInfo(path, experimental) {
  if (path === 'gpu-screen-experimental') {
    return {
      referencePath: 'packed-cpu',
      referenceRole: 'formal-reference',
      currentRole: experimental ? 'experimental' : 'formal-reference'
    };
  }

  return {
    referencePath: path,
    referenceRole: 'formal-reference',
    currentRole: experimental ? 'experimental' : 'formal-reference'
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
  return {
    path,
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
  const referenceInfo = buildReferenceInfo(path, experimental);

  return {
    referencePath: referenceInfo.referencePath,
    referenceRole: referenceInfo.referenceRole,
    currentPath: path,
    currentRole: referenceInfo.currentRole,
    currentExperimental: !!experimental,
    currentBuildMs: Number.isFinite(buildMs) ? buildMs : 0,
    currentPackedCount: Number.isFinite(packedCount) ? packedCount : 0,
    currentPackedLength: packed instanceof Float32Array ? packed.length : 0,
    usesPackedReferenceLayout: true,
    usesPackedReferencePack: true,
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

  const path = extra.path ?? 'packed-cpu';
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
    ...extra
  };
}

function updateContext(context, result, inputVisible) {
  if (!context) return;

  context.layout = context.layout ?? getVisiblePackLayout();
  context.lastPath = result.path;
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
    path: extra.path ?? 'packed-cpu',
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
    path: extra.path ?? 'packed-cpu',
    experimental: !!extra.experimental
  });

  updateContext(context, result, visible);

  return {
    ...result,
    layout: context?.layout ?? getVisiblePackLayout()
  };
}

export function buildGpuScreenExperimentalSpaceWithContext(context, visible, extra = {}) {
  return buildPackedScreenSpaceWithContext(context, visible, {
    ...extra,
    path: 'gpu-screen-experimental',
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
      referencePath: 'none',
      referenceRole: 'none',
      currentPath: 'none',
      currentRole: 'none',
      currentExperimental: false,
      currentBuildMs: 0,
      currentPackedCount: 0,
      currentPackedLength: 0,
      usesPackedReferenceLayout: false,
      usesPackedReferencePack: false,
      sameLayoutAsReference: false,
      samePackCountAsReference: false
    };
  }

  return {
    referencePath: result.comparisonSummary.referencePath ?? 'none',
    referenceRole: result.comparisonSummary.referenceRole ?? 'none',
    currentPath: result.comparisonSummary.currentPath ?? 'none',
    currentRole: result.comparisonSummary.currentRole ?? 'none',
    currentExperimental: !!result.comparisonSummary.currentExperimental,
    currentBuildMs: Number.isFinite(result.comparisonSummary.currentBuildMs)
      ? result.comparisonSummary.currentBuildMs
      : 0,
    currentPackedCount: Number.isFinite(result.comparisonSummary.currentPackedCount)
      ? result.comparisonSummary.currentPackedCount
      : 0,
    currentPackedLength: Number.isFinite(result.comparisonSummary.currentPackedLength)
      ? result.comparisonSummary.currentPackedLength
      : 0,
    usesPackedReferenceLayout: !!result.comparisonSummary.usesPackedReferenceLayout,
    usesPackedReferencePack: !!result.comparisonSummary.usesPackedReferencePack,
    sameLayoutAsReference: !!result.comparisonSummary.sameLayoutAsReference,
    samePackCountAsReference: !!result.comparisonSummary.samePackCountAsReference
  };
}

export function summarizeScreenSpaceBuildContext(context) {
  if (!context) {
    return {
      lastPath: 'none',
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
    lastPath: context.lastPath ?? 'none',
    lastInputVisibleCount: Number.isFinite(context.lastInputVisibleCount) ? context.lastInputVisibleCount : 0,
    lastNormalizedVisibleCount: Number.isFinite(context.lastNormalizedVisibleCount) ? context.lastNormalizedVisibleCount : 0,
    lastPackCount: Number.isFinite(context.lastPackCount) ? context.lastPackCount : 0,
    lastPackedLength: Number.isFinite(context.lastPackedLength) ? context.lastPackedLength : 0,
    lastBuildMs: Number.isFinite(context.lastBuildMs) ? context.lastBuildMs : 0,
    lastSummary: context.lastSummary ?? null,
    lastComparisonSummary: context.lastComparisonSummary ?? null
  };
}
