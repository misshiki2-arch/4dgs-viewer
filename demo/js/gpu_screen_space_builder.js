import {
  getVisiblePackLayout,
  GPU_VISIBLE_PACK_FLOATS_PER_ITEM
} from './gpu_buffer_layout_utils.js';
import {
  packVisibleItems,
  createPackedVisibleResult
} from './gpu_visible_pack_utils.js';

// Step27:
// screen-space 結果を packed layout の正式契約へ正規化して渡す場所。
// Step26 では packed-cpu と gpu-screen-experimental を同じ summary 形式で扱えるようにした。
// Step27 ではさらに、
// - packed formal reference 側 summary
// - gpu-screen experimental 側 summary
// の比較に使いやすい形へ整える。
// まだ packed 内容自体は共通でもよいが、summary 上は「何を参照し、何を名乗るか」を明示する。

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
    lastSummary: null,
    lastBuildMs: 0,

    // Step27:
    // packed formal reference と experimental path を比較しやすくするため、
    // comparison summary を保持できるようにする。
    lastComparisonSummary: null
  };
}

function buildReferenceInfo(path, experimental) {
  if (path === 'gpu-screen-experimental') {
    return {
      referencePath: 'packed-cpu',
      referenceRole: 'formal-reference',
      currentRole: 'experimental'
    };
  }

  return {
    referencePath: path,
    referenceRole: 'formal-reference',
    currentRole: experimental ? 'experimental' : 'formal-reference'
  };
}

function buildPackedScreenSpaceSummary({
  path,
  inputVisible,
  normalizedVisible,
  packed,
  packedCount,
  floatsPerItem,
  buildMs,
  experimental
}) {
  const referenceInfo = buildReferenceInfo(path, experimental);

  return {
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
    path,
    buildMs: Number.isFinite(buildMs) ? buildMs : 0,
    experimental: !!experimental,
    referencePath: referenceInfo.referencePath,
    referenceRole: referenceInfo.referenceRole,
    currentRole: referenceInfo.currentRole
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

  const summary = buildPackedScreenSpaceSummary({
    path,
    inputVisible: extra.inputVisible,
    normalizedVisible,
    packed,
    packedCount,
    floatsPerItem,
    buildMs: extra.buildMs,
    experimental
  });

  return {
    path,
    visible: normalizedVisible,
    packed,
    packedCount,
    floatsPerItem,
    layout: getVisiblePackLayout(),
    summary,
    experimental,
    ...extra
  };
}

function buildComparisonSummary(result) {
  return {
    currentPath: result?.path ?? 'none',
    currentExperimental: !!result?.experimental,
    currentBuildMs: Number.isFinite(result?.summary?.buildMs) ? result.summary.buildMs : 0,
    currentPackedCount: Number.isFinite(result?.packedCount) ? result.packedCount : 0,
    currentPackedLength: result?.packed instanceof Float32Array ? result.packed.length : 0,

    referencePath: result?.summary?.referencePath ?? 'packed-cpu',
    referenceRole: result?.summary?.referenceRole ?? 'formal-reference',
    currentRole: result?.summary?.currentRole ?? 'formal-reference',

    usesPackedReferenceLayout: true,
    usesPackedReferencePack: true,
    sameLayoutAsReference: true,
    samePackCountAsReference: true
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
  context.lastSummary = result.summary;
  context.lastBuildMs = Number.isFinite(result.summary?.buildMs) ? result.summary.buildMs : 0;
  context.lastComparisonSummary = buildComparisonSummary(result);
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

// Step27:
// gpu-screen 実験経路用の screen-space result を同じ契約で作る。
// packed 内容自体はまだ共通だが、summary 上は
// 「packed formal reference を参照している experimental path」
// であることを明示する。
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
      experimental: false,
      referencePath: 'none',
      referenceRole: 'none',
      currentRole: 'none'
    };
  }

  return {
    path: result.path ?? 'unknown',
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
    experimental: !!result.summary?.experimental,
    referencePath: result.summary?.referencePath ?? 'packed-cpu',
    referenceRole: result.summary?.referenceRole ?? 'formal-reference',
    currentRole: result.summary?.currentRole ?? 'formal-reference'
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
    lastComparisonSummary: context.lastComparisonSummary ?? null
  };
}
