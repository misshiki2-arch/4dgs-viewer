import {
  computeVisiblePackFieldByteOffset,
  getVisiblePackField,
  GPU_VISIBLE_PACK_BYTES_PER_ITEM
} from './gpu_buffer_layout_utils.js';
import {
  createManagedArrayBuffer,
  uploadManagedArrayBuffer
} from './gpu_gl_utils.js';

// Step22:
// packed visible path 用の upload 補助。
// 目的は、packed Float32Array を毎回 4 本の draw array に再展開する以外の経路を
// 段階的に導入できるようにすること。
// 現段階では interleaved packed buffer を管理し、attribute ごとの offset/stride 情報を
// renderer から参照できる形にする。

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
  return {
    packedUploadBytes: state?.lastUploadBytes ?? 0,
    packedUploadCount: state?.lastPackedCount ?? 0,
    packedUploadLength: state?.lastPackedLength ?? 0,
    packedUploadCapacityBytes: state?.interleaved?.capacityBytes ?? 0,
    packedUploadReusedCapacity: !!state?.reusedCapacity
  };
}

export function getPackedFieldByteOffset(index, fieldName) {
  return computeVisiblePackFieldByteOffset(index, fieldName);
}
