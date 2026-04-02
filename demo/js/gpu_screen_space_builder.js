import {
  getVisiblePackLayout,
  GPU_VISIBLE_PACK_FLOATS_PER_ITEM
} from './gpu_buffer_layout_utils.js';
import { packVisibleItems, createPackedVisibleResult } from './gpu_visible_pack_utils.js';

// Step20:
// 将来の GPU screen-space 化の置き場。
// 現段階では CPU 側で作られた visible object array を packed 化し、
// renderer 側が「object array 経路」と「packed 経路」を比較できるようにする。
// 後続 Step でここに GPU upload / transform feedback / compute 相当処理を寄せる。

export function createScreenSpaceBuildContext() {
  return {
    layout: getVisiblePackLayout(),
    lastPackCount: 0,
    lastPackedLength: 0,
    lastPath: 'packed-cpu'
  };
}

export function buildPackedScreenSpaceFromVisible(visible, extra = {}) {
  const packedResult = createPackedVisibleResult(visible, extra);

  return {
    path: 'packed-cpu',
    visible,
    packed: packedResult.packed,
    packedCount: packedResult.count,
    floatsPerItem: packedResult.floatsPerItem,
    layout: getVisiblePackLayout(),
    ...extra
  };
}

export function buildPackedScreenSpaceWithContext(context, visible, extra = {}) {
  const packedResult = packVisibleItems(visible);

  if (context) {
    context.lastPackCount = packedResult.count;
    context.lastPackedLength = packedResult.packed.length;
    context.lastPath = 'packed-cpu';
  }

  return {
    path: 'packed-cpu',
    visible,
    packed: packedResult.packed,
    packedCount: packedResult.count,
    floatsPerItem: packedResult.floatsPerItem,
    layout: context?.layout ?? getVisiblePackLayout(),
    ...extra
  };
}

export function summarizePackedScreenSpace(result) {
  if (!result) {
    return {
      path: 'none',
      packedCount: 0,
      packedLength: 0,
      floatsPerItem: GPU_VISIBLE_PACK_FLOATS_PER_ITEM
    };
  }

  return {
    path: result.path ?? 'unknown',
    packedCount: Number.isFinite(result.packedCount) ? result.packedCount : 0,
    packedLength: result.packed instanceof Float32Array ? result.packed.length : 0,
    floatsPerItem: Number.isFinite(result.floatsPerItem)
      ? result.floatsPerItem
      : GPU_VISIBLE_PACK_FLOATS_PER_ITEM
  };
}
