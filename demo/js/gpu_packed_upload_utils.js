import {
  computeVisiblePackFieldByteOffset,
  getVisiblePackField,
  getVisiblePackCanonicalFieldName,
  GPU_VISIBLE_PACK_BYTES_PER_ITEM,
  GPU_VISIBLE_PACK_LAYOUT_VERSION
} from './gpu_buffer_layout_utils.js';
import {
  createManagedArrayBuffer,
  uploadManagedArrayBuffer,
  summarizeManagedArrayBuffer
} from './gpu_gl_utils.js';

// Step24:
// packed visible path 用の upload 補助。
// このファイルは「layout 正本を参照して interleaved upload / attrib descriptor を返す」
// 役割に限定する。
//
// 重要:
// - packed layout の唯一の正本は gpu_buffer_layout_utils.js
// - shader が読む正式 alpha は colorAlpha[3]
// - このファイル自身は独自の field 解釈を持たない
// - 旧 field 名を直接使わず canonical 名へ正規化してから扱う

function getCanonicalField(fieldName) {
  return getVisiblePackField(getVisiblePackCanonicalFieldName(fieldName));
}

function buildAttribDescriptor(attributeName, fieldName) {
  const field = getCanonicalField(fieldName);
  return {
    name: attributeName,
    fieldName: field.name,
    canonicalFieldName: field.name,
    size: field.components,
    stride: GPU_VISIBLE_PACK_BYTES_PER_ITEM,
    offset: field.byteOffset
  };
}

export function createPackedUploadState(gl) {
  return {
    interleaved: createManagedArrayBuffer(gl, 0, gl.DYNAMIC_DRAW),
    lastPackedCount: 0,
    lastPackedLength: 0,
    lastUploadBytes: 0,
    reusedCapacity: false,
    layoutVersion: GPU_VISIBLE_PACK_LAYOUT_VERSION,
    lastStrideBytes: GPU_VISIBLE_PACK_BYTES_PER_ITEM
  };
}

export function getPackedInterleavedLayout() {
  const centerField = getCanonicalField('centerPx');
  const radiusField = getCanonicalField('radiusPx');
  const colorAlphaField = getCanonicalField('colorAlpha');
  const conicField = getCanonicalField('conic');
  const depthField = getCanonicalField('depth');
  const reservedField = getCanonicalField('reserved');
  const miscField = getCanonicalField('misc');

  return {
    layoutVersion: GPU_VISIBLE_PACK_LAYOUT_VERSION,
    strideBytes: GPU_VISIBLE_PACK_BYTES_PER_ITEM,

    centerPx: {
      fieldName: centerField.name,
      size: centerField.components,
      offsetBytes: centerField.byteOffset
    },
    radiusPx: {
      fieldName: radiusField.name,
      size: radiusField.components,
      offsetBytes: radiusField.byteOffset
    },
    depth: {
      fieldName: depthField.name,
      size: depthField.components,
      offsetBytes: depthField.byteOffset
    },
    colorAlpha: {
      fieldName: colorAlphaField.name,
      size: colorAlphaField.components,
      offsetBytes: colorAlphaField.byteOffset
    },
    conic: {
      fieldName: conicField.name,
      size: conicField.components,
      offsetBytes: conicField.byteOffset
    },
    reserved: {
      fieldName: reservedField.name,
      size: reservedField.components,
      offsetBytes: reservedField.byteOffset
    },
    misc: {
      fieldName: miscField.name,
      size: miscField.components,
      offsetBytes: miscField.byteOffset
    }
  };
}

export function getPackedInterleavedAttribDescriptors() {
  const layout = getPackedInterleavedLayout();

  return {
    layoutVersion: layout.layoutVersion,
    strideBytes: layout.strideBytes,
    attributes: [
      buildAttribDescriptor('aCenterPx', 'centerPx'),
      buildAttribDescriptor('aRadiusPx', 'radiusPx'),
      buildAttribDescriptor('aColorAlpha', 'colorAlpha'),
      buildAttribDescriptor('aConic', 'conic')
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
  state.reusedCapacity =
    (state.interleaved.capacityBytes | 0) === prevCapacity &&
    prevCapacity >= packed.byteLength;
  state.layoutVersion = GPU_VISIBLE_PACK_LAYOUT_VERSION;
  state.lastStrideBytes = GPU_VISIBLE_PACK_BYTES_PER_ITEM;

  return state;
}

export function summarizePackedUploadState(state) {
  const interleavedSummary = summarizeManagedArrayBuffer(state?.interleaved);

  return {
    packedUploadLayoutVersion: state?.layoutVersion ?? GPU_VISIBLE_PACK_LAYOUT_VERSION,
    packedUploadStrideBytes: state?.lastStrideBytes ?? GPU_VISIBLE_PACK_BYTES_PER_ITEM,
    packedUploadBytes: state?.lastUploadBytes ?? 0,
    packedUploadCount: state?.lastPackedCount ?? 0,
    packedUploadLength: state?.lastPackedLength ?? 0,
    packedUploadCapacityBytes: state?.interleaved?.capacityBytes ?? 0,
    packedUploadReusedCapacity: !!state?.reusedCapacity,
    packedUploadManagedCapacityReused: !!interleavedSummary.capacityReused,
    packedUploadManagedCapacityGrown: !!interleavedSummary.capacityGrown,
    packedUploadManagedUploadCount: interleavedSummary.uploadCount ?? 0,
    packedUploadAlphaSource: 'colorAlpha[3]'
  };
}

export function getPackedFieldByteOffset(index, fieldName) {
  return computeVisiblePackFieldByteOffset(index, getVisiblePackCanonicalFieldName(fieldName));
}
