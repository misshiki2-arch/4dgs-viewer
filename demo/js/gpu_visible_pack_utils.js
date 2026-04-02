import {
  createVisiblePackFloatArray,
  computeVisiblePackFieldFloatOffset,
  GPU_VISIBLE_PACK_FLOATS_PER_ITEM
} from './gpu_buffer_layout_utils.js';

// Step20:
// visible object array -> packed Float32Array への変換。
// まずは既存の visible 構造をそのまま詰め替える CPU 側ユーティリティとして導入し、
// 後続 Step で GPU upload / screen-space path に接続しやすくする。

function writeVec(floatArray, baseOffset, values, expectedLength) {
  for (let i = 0; i < expectedLength; i++) {
    floatArray[baseOffset + i] = values && Number.isFinite(values[i]) ? values[i] : 0;
  }
}

function writeScalar(floatArray, offset, value, fallback = 0) {
  floatArray[offset] = Number.isFinite(value) ? value : fallback;
}

export function packVisibleItem(floatArray, index, item) {
  const centerOffset = computeVisiblePackFieldFloatOffset(index, 'centerPx');
  const radiusOffset = computeVisiblePackFieldFloatOffset(index, 'radiusPx');
  const depthOffset = computeVisiblePackFieldFloatOffset(index, 'depth');
  const colorOffset = computeVisiblePackFieldFloatOffset(index, 'color');
  const conicOffset = computeVisiblePackFieldFloatOffset(index, 'conic');
  const opacityOffset = computeVisiblePackFieldFloatOffset(index, 'opacity');
  const aabbOffset = computeVisiblePackFieldFloatOffset(index, 'aabb');

  writeVec(floatArray, centerOffset, [item?.px, item?.py], 2);
  writeScalar(floatArray, radiusOffset, item?.radius, 0);
  writeScalar(floatArray, depthOffset, item?.depth, 0);
  writeVec(floatArray, colorOffset, item?.color, 4);
  writeVec(floatArray, conicOffset, item?.conic, 3);
  writeScalar(floatArray, opacityOffset, item?.opacity, 0);
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
