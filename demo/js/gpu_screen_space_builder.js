import {
  getVisiblePackLayout,
  GPU_VISIBLE_PACK_FLOATS_PER_ITEM
} from './gpu_buffer_layout_utils.js';
import {
  packVisibleItems,
  createPackedVisibleResult
} from './gpu_visible_pack_utils.js';

function nowMs() { return performance.now(); }
function toFiniteOr(value, fallback = 0) { return Number.isFinite(value) ? value : fallback; }
function normalizeCenterPx(item) {
  if (Array.isArray(item?.centerPx) && item.centerPx.length >= 2) return [toFiniteOr(item.centerPx[0],0), toFiniteOr(item.centerPx[1],0)];
  return [toFiniteOr(item?.px,0), toFiniteOr(item?.py,0)];
}
function normalizeRadiusPx(item) { if (Number.isFinite(item?.radiusPx)) return item.radiusPx; if (Number.isFinite(item?.radius)) return item.radius; return 0; }
function normalizeDepth(item) { return toFiniteOr(item?.depth, 0); }
function normalizeColorAlpha(item) {
  if (Array.isArray(item?.colorAlpha) && item.colorAlpha.length >= 4) return [toFiniteOr(item.colorAlpha[0],0),toFiniteOr(item.colorAlpha[1],0),toFiniteOr(item.colorAlpha[2],0),toFiniteOr(item.colorAlpha[3],0)];
  const color = Array.isArray(item?.color) ? item.color : [0,0,0,0];
  const alpha = Number.isFinite(item?.opacity) ? item.opacity : toFiniteOr(color[3],0);
  return [toFiniteOr(color[0],0),toFiniteOr(color[1],0),toFiniteOr(color[2],0),toFiniteOr(alpha,0)];
}
function normalizeConic(item) { if (Array.isArray(item?.conic) && item.conic.length >= 3) return [toFiniteOr(item.conic[0],0),toFiniteOr(item.conic[1],0),toFiniteOr(item.conic[2],0)]; return [0,0,0]; }
function normalizeReserved(item) { return toFiniteOr(item?.reserved, 0); }
function normalizeMisc(item) {
  if (Array.isArray(item?.misc) && item.misc.length >= 4) return [toFiniteOr(item.misc[0],0),toFiniteOr(item.misc[1],0),toFiniteOr(item.misc[2],0),toFiniteOr(item.misc[3],0)];
  if (Array.isArray(item?.aabb) && item.aabb.length >= 4) return [toFiniteOr(item.aabb[0],0),toFiniteOr(item.aabb[1],0),toFiniteOr(item.aabb[2],0),toFiniteOr(item.aabb[3],0)];
  return [0,0,0,0];
}
export function normalizeScreenSpaceItem(item) { return { ...item, centerPx: normalizeCenterPx(item), radiusPx: normalizeRadiusPx(item), depth: normalizeDepth(item), colorAlpha: normalizeColorAlpha(item), conic: normalizeConic(item), reserved: normalizeReserved(item), misc: normalizeMisc(item) }; }
export function normalizeScreenSpaceVisible(visible) { if (!Array.isArray(visible)||visible.length===0) return []; const normalized=new Array(visible.length); for(let i=0;i<visible.length;i++) normalized[i]=normalizeScreenSpaceItem(visible[i]); return normalized; }
export function createScreenSpaceBuildContext() { return { layout:getVisiblePackLayout(), lastPath:'packed-cpu', lastInputVisibleCount:0, lastNormalizedVisibleCount:0, lastPackCount:0, lastPackedLength:0, lastSummary:null, lastBuildMs:0 }; }
function buildPackedScreenSpaceResult(normalizedVisible, packedResult, extra = {}) {
  const packed = packedResult?.packed instanceof Float32Array ? packedResult.packed : null;
  const packedCount = Number.isFinite(packedResult?.count) ? packedResult.count : 0;
  const floatsPerItem = Number.isFinite(packedResult?.floatsPerItem) ? packedResult.floatsPerItem : GPU_VISIBLE_PACK_FLOATS_PER_ITEM;
  return {
    path: extra.path ?? 'packed-cpu',
    visible: normalizedVisible,
    packed,
    packedCount,
    floatsPerItem,
    layout: getVisiblePackLayout(),
    summary: {
      inputVisibleCount: Array.isArray(extra.inputVisible) ? extra.inputVisible.length : normalizedVisible.length,
      normalizedVisibleCount: normalizedVisible.length,
      packedCount,
      packedLength: packed ? packed.length : 0,
      floatsPerItem,
      alphaSource: 'colorAlpha[3]',
      centerSource: 'centerPx',
      radiusSource: 'radiusPx',
      depthSource: 'depth',
      conicSource: 'conic',
      miscSource: 'misc',
      path: extra.path ?? 'packed-cpu',
      buildMs: Number.isFinite(extra.buildMs) ? extra.buildMs : 0
    },
    ...extra
  };
}
export function buildPackedScreenSpaceFromVisible(visible, extra = {}) {
  const t0 = nowMs();
  const normalizedVisible = normalizeScreenSpaceVisible(visible);
  const packedResult = createPackedVisibleResult(normalizedVisible, extra);
  const buildMs = nowMs() - t0;
  return buildPackedScreenSpaceResult(normalizedVisible, packedResult, { ...extra, inputVisible: visible, buildMs, path: extra.path ?? 'packed-cpu' });
}
export function buildPackedScreenSpaceWithContext(context, visible, extra = {}) {
  const t0 = nowMs();
  const normalizedVisible = normalizeScreenSpaceVisible(visible);
  const packedResult = packVisibleItems(normalizedVisible);
  const buildMs = nowMs() - t0;
  const result = buildPackedScreenSpaceResult(normalizedVisible, packedResult, { ...extra, inputVisible: visible, buildMs, path: extra.path ?? 'packed-cpu' });
  if (context) {
    context.layout = context.layout ?? getVisiblePackLayout();
    context.lastPath = result.path;
    context.lastInputVisibleCount = Array.isArray(visible) ? visible.length : 0;
    context.lastNormalizedVisibleCount = normalizedVisible.length;
    context.lastPackCount = packedResult.count;
    context.lastPackedLength = packedResult.packed.length;
    context.lastSummary = result.summary;
    context.lastBuildMs = buildMs;
  }
  return { ...result, layout: context?.layout ?? getVisiblePackLayout() };
}
export function summarizePackedScreenSpace(result) {
  if (!result) return { path:'none', packedCount:0, packedLength:0, floatsPerItem:GPU_VISIBLE_PACK_FLOATS_PER_ITEM, alphaSource:'colorAlpha[3]', centerSource:'centerPx', radiusSource:'radiusPx', depthSource:'depth', conicSource:'conic', miscSource:'misc', buildMs:0 };
  return {
    path: result.path ?? 'unknown',
    packedCount: Number.isFinite(result.packedCount) ? result.packedCount : 0,
    packedLength: result.packed instanceof Float32Array ? result.packed.length : 0,
    floatsPerItem: Number.isFinite(result.floatsPerItem) ? result.floatsPerItem : GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
    alphaSource: result.summary?.alphaSource ?? 'colorAlpha[3]',
    centerSource: result.summary?.centerSource ?? 'centerPx',
    radiusSource: result.summary?.radiusSource ?? 'radiusPx',
    depthSource: result.summary?.depthSource ?? 'depth',
    conicSource: result.summary?.conicSource ?? 'conic',
    miscSource: result.summary?.miscSource ?? 'misc',
    buildMs: Number.isFinite(result.summary?.buildMs) ? result.summary.buildMs : 0
  };
}
