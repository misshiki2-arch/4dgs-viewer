import {
  computeVisiblePackFieldByteOffset,
  getVisiblePackField,
  GPU_VISIBLE_PACK_BYTES_PER_ITEM
} from './gpu_buffer_layout_utils.js';
import {
  createManagedArrayBuffer,
  uploadManagedArrayBuffer,
  summarizeManagedArrayBuffer
} from './gpu_gl_utils.js';

// Step23:
// packed visible path 用の upload 補助。
// Step22 では interleaved buffer 管理と upload 統計の取得まで行った。
// Step23 では packed direct draw executor からそのまま使えるように、
// attribute descriptor 群を返す補助を追加する。

export function createPackedUploadState(gl) {
  return {
    interleaved: createManagedArrayBuffer(gl, 0, gl.DYNAMIC_DRAW),
    lastPackedCount: 0,
    lastPackedLength: 0,
    lastUploadBytes: 0,
    reusedCapacity: false
  };
}

export function getPackedInterleavedLayout() {
  const centerField = getVisiblePackField('centerPx');
  const radiusField = getVisiblePackField('radiusPx');
  const colorField = getVisiblePackField('color');
  const conicField = getVisiblePackField('conic');

  return {
    strideBytes: GPU_VISIBLE_PACK_BYTES_PER_ITEM,
    centerPx: {
      size: centerField.components,
      offsetBytes: centerField.byteOffset
    },
    radiusPx: {
      size: radiusField.components,
      offsetBytes: radiusField.byteOffset
    },
    color: {
      size: colorField.components,
      offsetBytes: colorField.byteOffset
    },
    conic: {
      size: conicField.components,
      offsetBytes: conicField.byteOffset
    }
  };
}

export function getPackedInterleavedAttribDescriptors() {
  const layout = getPackedInterleavedLayout();
  return {
    strideBytes: layout.strideBytes,
    attributes: [
      {
        name: 'aCenterPx',
        size: layout.centerPx.size,
        stride: layout.strideBytes,
        offset: layout.centerPx.offsetBytes
      },
      {
        name: 'aRadiusPx',
        size: layout.radiusPx.size,
        stride: layout.strideBytes,
        offset: layout.radiusPx.offsetBytes
      },
      {
        name: 'aColorAlpha',
        size: layout.color.size,
        stride: layout.strideBytes,
        offset: layout.color.offsetBytes
      },
      {
        name: 'aConic',
        size: layout.conic.size,
        stride: layout.strideBytes,
        offset: layout.conic.offsetBytes
      }
    ]
  };
}

export function uploadPackedInterleaved(gl, state, packed, packedCount) {
  if (!(packed instanceof Float32Array)) {
    throw new Error('uploadPackedInterleaved expects packed to be a Float32Array');
  }
  if (!state?.interleaved?.buffer) {
    throw new Error('uploadPackedInterleaved expects a packed upload state');
  }

  const prevCapacity = state.interleaved.capacityBytes | 0;
  uploadManagedArrayBuffer(gl, state.interleaved, packed, gl.DYNAMIC_DRAW);

  state.lastPackedCount = Math.max(0, packedCount | 0);
  state.lastPackedLength = packed.length;
  state.lastUploadBytes = packed.byteLength;
  state.reusedCapacity = (state.interleaved.capacityBytes | 0) === prevCapacity && prevCapacity >= packed.byteLength;

  return state;
}

export function summarizePackedUploadState(state) {
  const interleavedSummary = summarizeManagedArrayBuffer(state?.interleaved);

  return {
    packedUploadBytes: state?.lastUploadBytes ?? 0,
    packedUploadCount: state?.lastPackedCount ?? 0,
    packedUploadLength: state?.lastPackedLength ?? 0,
    packedUploadCapacityBytes: state?.interleaved?.capacityBytes ?? 0,
    packedUploadReusedCapacity: !!state?.reusedCapacity,
    packedUploadManagedCapacityReused: !!interleavedSummary.capacityReused,
    packedUploadManagedCapacityGrown: !!interleavedSummary.capacityGrown,
    packedUploadManagedUploadCount: interleavedSummary.uploadCount ?? 0
  };
}

export function getPackedFieldByteOffset(index, fieldName) {
  return computeVisiblePackFieldByteOffset(index, fieldName);
}
