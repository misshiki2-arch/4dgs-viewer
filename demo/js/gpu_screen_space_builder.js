import {
  getVisiblePackLayout,
  GPU_VISIBLE_PACK_FLOATS_PER_ITEM
} from './gpu_buffer_layout_utils.js';
import {
  packVisibleItems,
  createPackedVisibleResult
} from './gpu_visible_pack_utils.js';

// Step24:
// screen-space 結果を packed layout の正式契約へ正規化して渡す場所。
// 現段階では GPU screen-space 本体は未実装であり、CPU 側で構築された
// visible object array を packed 化する。
// ただし Step24 では以下をこのファイルで固定する:
//
// 1. packed layout の唯一の正本は gpu_buffer_layout_utils.js
// 2. packed 書き込みの唯一の正本は gpu_visible_pack_utils.js
// 3. このファイルは screen-space 結果の意味を整理して packed へ橋渡しする
// 4. alpha の正式値は colorAlpha[3]
// 5. path 名は当面 'packed-cpu' を使う
//
// 将来 Step25 以降で GPU screen-space 実装を追加する場合も、
// renderer 側には raw な中間値を直接渡さず、この形式へ正規化して返す。

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

// Step24:
// screen-space builder の出力は「packed に書ける意味」に正規化する。
// これにより、後段の packVisibleItems は canonical 名だけを見ればよい。
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
    lastPackCount: 0,
    lastPackedLength: 0,
    lastPath: 'packed-cpu',
    lastInputVisibleCount: 0,
    lastNormalizedVisibleCount: 0,
    lastSummary: null
  };
}

function buildPackedScreenSpaceResult(normalizedVisible, packedResult, extra = {}) {
  return {
    path: 'packed-cpu',
    visible: normalizedVisible,
    packed: packedResult.packed,
    packedCount: packedResult.count,
    floatsPerItem: packedResult.floatsPerItem,
    layout: getVisiblePackLayout(),
    summary: {
      inputVisibleCount: Array.isArray(normalizedVisible) ? normalizedVisible.length : 0,
      normalizedVisibleCount: packedResult.count,
      packedLength: packedResult.packed instanceof Float32Array ? packedResult.packed.length : 0,
      floatsPerItem: packedResult.floatsPerItem,
      alphaSource: 'colorAlpha[3]',
      centerSource: 'centerPx',
      radiusSource: 'radiusPx',
      conicSource: 'conic',
      miscSource: 'misc'
    },
    ...extra
  };
}

export function buildPackedScreenSpaceFromVisible(visible, extra = {}) {
  const normalizedVisible = normalizeScreenSpaceVisible(visible);
  const packedResult = createPackedVisibleResult(normalizedVisible, extra);

  return buildPackedScreenSpaceResult(normalizedVisible, packedResult, extra);
}

export function buildPackedScreenSpaceWithContext(context, visible, extra = {}) {
  const normalizedVisible = normalizeScreenSpaceVisible(visible);
  const packedResult = packVisibleItems(normalizedVisible);
  const result = buildPackedScreenSpaceResult(normalizedVisible, packedResult, extra);

  if (context) {
    context.layout = context.layout ?? getVisiblePackLayout();
    context.lastPackCount = packedResult.count;
    context.lastPackedLength = packedResult.packed.length;
    context.lastPath = 'packed-cpu';
    context.lastInputVisibleCount = Array.isArray(visible) ? visible.length : 0;
    context.lastNormalizedVisibleCount = normalizedVisible.length;
    context.lastSummary = result.summary;
  }

  return {
    ...result,
    layout: context?.layout ?? getVisiblePackLayout()
  };
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
      conicSource: 'conic',
      miscSource: 'misc'
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
    conicSource: result.summary?.conicSource ?? 'conic',
    miscSource: result.summary?.miscSource ?? 'misc'
  };
}
