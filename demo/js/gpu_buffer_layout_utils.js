// Step20:
// Packed visible path 用の buffer layout 定義。
// 今後 CPU object array から typed-array / GPU upload path へ寄せるため、
// 各 attribute の成分数・byte 幅・offset をここに集約する。

export const GPU_VISIBLE_PACK_FLOAT_BYTES = 4;
export const GPU_VISIBLE_PACK_UINT_BYTES = 4;

export const GPU_VISIBLE_PACK_LAYOUT_VERSION = 1;

// Packed visible 1件あたりの float 構成
// centerPx.xy, radiusPx, depth, color.rgba, conic.xyz, opacity, aabb.xyzw
// 必要に応じて将来拡張する
export const GPU_VISIBLE_PACK_FIELDS = [
  { name: 'centerPx', components: 2, type: 'float32' },
  { name: 'radiusPx', components: 1, type: 'float32' },
  { name: 'depth', components: 1, type: 'float32' },
  { name: 'color', components: 4, type: 'float32' },
  { name: 'conic', components: 3, type: 'float32' },
  { name: 'opacity', components: 1, type: 'float32' },
  { name: 'aabb', components: 4, type: 'float32' }
];

function getTypeByteSize(type) {
  if (type === 'float32') return GPU_VISIBLE_PACK_FLOAT_BYTES;
  if (type === 'uint32') return GPU_VISIBLE_PACK_UINT_BYTES;
  throw new Error(`Unsupported packed type: ${type}`);
}

function buildFieldLayout(fields) {
  let floatOffset = 0;
  let byteOffset = 0;

  const out = [];
  for (const field of fields) {
    const typeBytes = getTypeByteSize(field.type);
    const byteSize = field.components * typeBytes;

    out.push({
      ...field,
      floatOffset,
      byteOffset,
      byteSize
    });

    floatOffset += field.components;
    byteOffset += byteSize;
  }

  return {
    fields: out,
    floatsPerItem: floatOffset,
    bytesPerItem: byteOffset
  };
}

const VISIBLE_PACK_LAYOUT = buildFieldLayout(GPU_VISIBLE_PACK_FIELDS);

export const GPU_VISIBLE_PACK_FLOATS_PER_ITEM = VISIBLE_PACK_LAYOUT.floatsPerItem;
export const GPU_VISIBLE_PACK_BYTES_PER_ITEM = VISIBLE_PACK_LAYOUT.bytesPerItem;

export function getVisiblePackLayout() {
  return VISIBLE_PACK_LAYOUT;
}

export function getVisiblePackField(name) {
  const field = VISIBLE_PACK_LAYOUT.fields.find(f => f.name === name);
  if (!field) {
    throw new Error(`Unknown visible pack field: ${name}`);
  }
  return field;
}

export function createVisiblePackFloatArray(count) {
  const n = Math.max(0, count | 0);
  return new Float32Array(n * GPU_VISIBLE_PACK_FLOATS_PER_ITEM);
}

export function computeVisiblePackBaseFloatOffset(index) {
  return Math.max(0, index | 0) * GPU_VISIBLE_PACK_FLOATS_PER_ITEM;
}

export function computeVisiblePackBaseByteOffset(index) {
  return Math.max(0, index | 0) * GPU_VISIBLE_PACK_BYTES_PER_ITEM;
}

export function computeVisiblePackFieldFloatOffset(index, fieldName) {
  const field = getVisiblePackField(fieldName);
  return computeVisiblePackBaseFloatOffset(index) + field.floatOffset;
}

export function computeVisiblePackFieldByteOffset(index, fieldName) {
  const field = getVisiblePackField(fieldName);
  return computeVisiblePackBaseByteOffset(index) + field.byteOffset;
}

export function createVisiblePackViewMap(floatArray) {
  return {
    raw: floatArray,
    centerPx: { field: getVisiblePackField('centerPx') },
    radiusPx: { field: getVisiblePackField('radiusPx') },
    depth: { field: getVisiblePackField('depth') },
    color: { field: getVisiblePackField('color') },
    conic: { field: getVisiblePackField('conic') },
    opacity: { field: getVisiblePackField('opacity') },
    aabb: { field: getVisiblePackField('aabb') }
  };
}
