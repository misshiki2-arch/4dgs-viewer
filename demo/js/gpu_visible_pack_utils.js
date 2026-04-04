import {
  createVisiblePackFloatArray,
  computeVisiblePackFieldFloatOffset,
  GPU_VISIBLE_PACK_FLOATS_PER_ITEM
} from './gpu_buffer_layout_utils.js';

// Step24:
// visible object array -> packed Float32Array への正式変換。
// packed layout の canonical 名に従って書き込む。
// - colorAlpha が正式色/alpha
// - alpha の正式値は colorAlpha[3]
// - reserved / misc は layout 正本の名前で扱う
//
// 旧構造との互換:
// - item.colorAlpha が無い場合のみ item.color + item.opacity から補完
// - item.aabb は misc へ格納
//
// このファイルは「visible -> packed 書き込み」の唯一の実装とし、
// layout 解釈そのものは gpu_buffer_layout_utils.js 側に持たせる。

function writeVec(floatArray, baseOffset, values, expectedLength) {
  for (let i = 0; i < expectedLength; i++) {
    floatArray[baseOffset + i] =
      values && Number.isFinite(values[i]) ? values[i] : 0;
  }
}

function writeScalar(floatArray, offset, value, fallback = 0) {
  floatArray[offset] = Number.isFinite(value) ? value : fallback;
}

function resolveColorAlpha(item) {
  if (Array.isArray(item?.colorAlpha) && item.colorAlpha.length >= 4) {
    return [
      Number.isFinite(item.colorAlpha[0]) ? item.colorAlpha[0] : 0,
      Number.isFinite(item.colorAlpha[1]) ? item.colorAlpha[1] : 0,
      Number.isFinite(item.colorAlpha[2]) ? item.colorAlpha[2] : 0,
      Number.isFinite(item.colorAlpha[3]) ? item.colorAlpha[3] : 0
    ];
  }

  const color = Array.isArray(item?.color) ? item.color : [0, 0, 0, 0];
  const opacity =
    Number.isFinite(item?.opacity)
      ? item.opacity
      : (Number.isFinite(color[3]) ? color[3] : 0);

  return [
    Number.isFinite(color[0]) ? color[0] : 0,
    Number.isFinite(color[1]) ? color[1] : 0,
    Number.isFinite(color[2]) ? color[2] : 0,
    Number.isFinite(opacity) ? opacity : 0
  ];
}

function resolveCenterPx(item) {
  if (Array.isArray(item?.centerPx) && item.centerPx.length >= 2) {
    return [
      Number.isFinite(item.centerPx[0]) ? item.centerPx[0] : 0,
      Number.isFinite(item.centerPx[1]) ? item.centerPx[1] : 0
    ];
  }

  return [
    Number.isFinite(item?.px) ? item.px : 0,
    Number.isFinite(item?.py) ? item.py : 0
  ];
}

function resolveRadiusPx(item) {
  if (Number.isFinite(item?.radiusPx)) return item.radiusPx;
  if (Number.isFinite(item?.radius)) return item.radius;
  return 0;
}

function resolveConic(item) {
  if (Array.isArray(item?.conic) && item.conic.length >= 3) {
    return [
      Number.isFinite(item.conic[0]) ? item.conic[0] : 0,
      Number.isFinite(item.conic[1]) ? item.conic[1] : 0,
      Number.isFinite(item.conic[2]) ? item.conic[2] : 0
    ];
  }
  return [0, 0, 0];
}

function resolveReserved(item) {
  if (Number.isFinite(item?.reserved)) return item.reserved;
  return 0;
}

function resolveMisc(item) {
  if (Array.isArray(item?.misc) && item.misc.length >= 4) {
    return [
      Number.isFinite(item.misc[0]) ? item.misc[0] : 0,
      Number.isFinite(item.misc[1]) ? item.misc[1] : 0,
      Number.isFinite(item.misc[2]) ? item.misc[2] : 0,
      Number.isFinite(item.misc[3]) ? item.misc[3] : 0
    ];
  }

  if (Array.isArray(item?.aabb) && item.aabb.length >= 4) {
    return [
      Number.isFinite(item.aabb[0]) ? item.aabb[0] : 0,
      Number.isFinite(item.aabb[1]) ? item.aabb[1] : 0,
      Number.isFinite(item.aabb[2]) ? item.aabb[2] : 0,
      Number.isFinite(item.aabb[3]) ? item.aabb[3] : 0
    ];
  }

  return [0, 0, 0, 0];
}

export function packVisibleItem(floatArray, index, item) {
  const centerOffset = computeVisiblePackFieldFloatOffset(index, 'centerPx');
  const radiusOffset = computeVisiblePackFieldFloatOffset(index, 'radiusPx');
  const depthOffset = computeVisiblePackFieldFloatOffset(index, 'depth');
  const colorAlphaOffset = computeVisiblePackFieldFloatOffset(index, 'colorAlpha');
  const conicOffset = computeVisiblePackFieldFloatOffset(index, 'conic');
  const reservedOffset = computeVisiblePackFieldFloatOffset(index, 'reserved');
  const miscOffset = computeVisiblePackFieldFloatOffset(index, 'misc');

  const centerPx = resolveCenterPx(item);
  const radiusPx = resolveRadiusPx(item);
  const depth = Number.isFinite(item?.depth) ? item.depth : 0;
  const colorAlpha = resolveColorAlpha(item);
  const conic = resolveConic(item);
  const reserved = resolveReserved(item);
  const misc = resolveMisc(item);

  writeVec(floatArray, centerOffset, centerPx, 2);
  writeScalar(floatArray, radiusOffset, radiusPx, 0);
  writeScalar(floatArray, depthOffset, depth, 0);
  writeVec(floatArray, colorAlphaOffset, colorAlpha, 4);
  writeVec(floatArray, conicOffset, conic, 3);
  writeScalar(floatArray, reservedOffset, reserved, 0);
  writeVec(floatArray, miscOffset, misc, 4);
}

export function packVisibleItems(visible) {
  const count = Array.isArray(visible) ? visible.length : 0;
  const packed = createVisiblePackFloatArray(count);

  for (let i = 0; i < count; i++) {
    packVisibleItem(packed, i, visible[i]);
  }

  return {
    packed,
    count,
    floatsPerItem: GPU_VISIBLE_PACK_FLOATS_PER_ITEM
  };
}

export function repackVisibleItemsInto(packed, visible) {
  const count = Array.isArray(visible) ? visible.length : 0;
  const neededLength = count * GPU_VISIBLE_PACK_FLOATS_PER_ITEM;

  if (!(packed instanceof Float32Array)) {
    throw new Error('repackVisibleItemsInto expects a Float32Array');
  }
  if (packed.length < neededLength) {
    throw new Error(
      `Packed array too small: have ${packed.length}, need at least ${neededLength}`
    );
  }

  for (let i = 0; i < count; i++) {
    packVisibleItem(packed, i, visible[i]);
  }

  return {
    packed,
    count,
    floatsPerItem: GPU_VISIBLE_PACK_FLOATS_PER_ITEM
  };
}

export function createPackedVisibleResult(visible, extra = {}) {
  const packedResult = packVisibleItems(visible);
  return {
    ...extra,
    ...packedResult
  };
}
