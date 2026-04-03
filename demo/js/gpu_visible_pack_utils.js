import {
  createVisiblePackFloatArray,
  computeVisiblePackFieldFloatOffset,
  GPU_VISIBLE_PACK_FLOATS_PER_ITEM
} from './gpu_buffer_layout_utils.js';

// Step20.1:
// visible object array -> packed Float32Array への変換。
// 描画契約の正本を colorAlpha に寄せる。
// 旧構造との互換のため、colorAlpha が無い場合のみ color / opacity へ fallback する。

function writeVec(floatArray, baseOffset, values, expectedLength) {
  for (let i = 0; i < expectedLength; i++) {
    floatArray[baseOffset + i] = values && Number.isFinite(values[i]) ? values[i] : 0;
  }
}

function writeScalar(floatArray, offset, value, fallback = 0) {
  floatArray[offset] = Number.isFinite(value) ? value : fallback;
}

function resolveColorAlpha(item) {
  if (Array.isArray(item?.colorAlpha) && item.colorAlpha.length >= 4) {
    return item.colorAlpha;
  }

  const color = Array.isArray(item?.color) ? item.color : [0, 0, 0, 0];
  const opacity = Number.isFinite(item?.opacity)
    ? item.opacity
    : (Number.isFinite(color[3]) ? color[3] : 0);

  return [
    Number.isFinite(color[0]) ? color[0] : 0,
    Number.isFinite(color[1]) ? color[1] : 0,
    Number.isFinite(color[2]) ? color[2] : 0,
    opacity
  ];
}

export function packVisibleItem(floatArray, index, item) {
  const centerOffset = computeVisiblePackFieldFloatOffset(index, 'centerPx');
  const radiusOffset = computeVisiblePackFieldFloatOffset(index, 'radiusPx');
  const depthOffset = computeVisiblePackFieldFloatOffset(index, 'depth');
  const colorOffset = computeVisiblePackFieldFloatOffset(index, 'color');
  const conicOffset = computeVisiblePackFieldFloatOffset(index, 'conic');
  const opacityOffset = computeVisiblePackFieldFloatOffset(index, 'opacity');
  const aabbOffset = computeVisiblePackFieldFloatOffset(index, 'aabb');

  const colorAlpha = resolveColorAlpha(item);

  writeVec(floatArray, centerOffset, [item?.px, item?.py], 2);
  writeScalar(floatArray, radiusOffset, item?.radius, 0);
  writeScalar(floatArray, depthOffset, item?.depth, 0);
  writeVec(floatArray, colorOffset, colorAlpha, 4);
  writeVec(floatArray, conicOffset, item?.conic, 3);
  writeScalar(floatArray, opacityOffset, colorAlpha[3], 0);
  writeVec(floatArray, aabbOffset, item?.aabb, 4);
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
