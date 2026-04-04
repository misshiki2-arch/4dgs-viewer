// Step24:
// GPU packed layout の正式契約をここに固定する。
// このファイルを packed layout の唯一の正本とし、
// CPU pack / GPU upload / shader 読み出しの全員がこの定義に従う。

export const GPU_VISIBLE_PACK_FLOAT_BYTES = 4;
export const GPU_VISIBLE_PACK_UINT_BYTES = 4;

// Step24 で正式契約を更新したので version を上げる。
export const GPU_VISIBLE_PACK_LAYOUT_VERSION = 2;

// 正式契約:
// 1 splat = 16 float = 64 byte
//
//  0, 1  : centerPx.xy
//  2     : radiusPx
//  3     : depth
//  4-7   : colorAlpha.rgba
//  8-10  : conic.xyz
// 11     : reserved
// 12-15  : misc.xyzw   (現時点では aabb 等を入れてよい拡張領域)
//
// 重要:
// alpha の正式な位置は colorAlpha[3] のみ。
// separate opacity は正式契約から外す。
export const GPU_VISIBLE_PACK_FIELDS = [
  { name: 'centerPx', components: 2, type: 'float32' },
  { name: 'radiusPx', components: 1, type: 'float32' },
  { name: 'depth', components: 1, type: 'float32' },
  { name: 'colorAlpha', components: 4, type: 'float32' },
  { name: 'conic', components: 3, type: 'float32' },
  { name: 'reserved', components: 1, type: 'float32' },
  { name: 'misc', components: 4, type: 'float32' }
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

// 正式 alpha の位置を明示的に公開する。
// alpha = colorAlpha[3]
export const GPU_VISIBLE_PACK_ALPHA_FIELD_NAME = 'colorAlpha';
export const GPU_VISIBLE_PACK_ALPHA_COMPONENT_INDEX = 3;
export const GPU_VISIBLE_PACK_ALPHA_FLOAT_OFFSET_WITHIN_ITEM =
  VISIBLE_PACK_LAYOUT.fields.find(f => f.name === GPU_VISIBLE_PACK_ALPHA_FIELD_NAME).floatOffset +
  GPU_VISIBLE_PACK_ALPHA_COMPONENT_INDEX;
export const GPU_VISIBLE_PACK_ALPHA_BYTE_OFFSET_WITHIN_ITEM =
  GPU_VISIBLE_PACK_ALPHA_FLOAT_OFFSET_WITHIN_ITEM * GPU_VISIBLE_PACK_FLOAT_BYTES;

// 互換用 alias
// Step24 の移行期間だけ旧名を解決できるようにして、1ファイルずつ直しやすくする。
// ただし canonical は必ず colorAlpha / misc / reserved 側。
function buildFieldAliasMap() {
  const colorAlphaField = VISIBLE_PACK_LAYOUT.fields.find(f => f.name === 'colorAlpha');
  const miscField = VISIBLE_PACK_LAYOUT.fields.find(f => f.name === 'misc');

  return {
    color: {
      ...colorAlphaField,
      name: 'color',
      aliasOf: 'colorAlpha'
    },
    opacity: {
      name: 'opacity',
      components: 1,
      type: 'float32',
      floatOffset: colorAlphaField.floatOffset + 3,
      byteOffset: colorAlphaField.byteOffset + 3 * GPU_VISIBLE_PACK_FLOAT_BYTES,
      byteSize: GPU_VISIBLE_PACK_FLOAT_BYTES,
      aliasOf: 'colorAlpha'
    },
    aabb: {
      ...miscField,
      name: 'aabb',
      aliasOf: 'misc'
    }
  };
}

const VISIBLE_PACK_FIELD_ALIASES = buildFieldAliasMap();

export function getVisiblePackLayout() {
  return VISIBLE_PACK_LAYOUT;
}

export function getVisiblePackField(name) {
  const canonicalField = VISIBLE_PACK_LAYOUT.fields.find(f => f.name === name);
  if (canonicalField) return canonicalField;

  const aliasField = VISIBLE_PACK_FIELD_ALIASES[name];
  if (aliasField) return aliasField;

  throw new Error(`Unknown visible pack field: ${name}`);
}

export function getVisiblePackCanonicalFieldName(name) {
  const field = getVisiblePackField(name);
  return field.aliasOf || field.name;
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

export function computeVisiblePackAlphaFloatOffset(index) {
  return computeVisiblePackBaseFloatOffset(index) + GPU_VISIBLE_PACK_ALPHA_FLOAT_OFFSET_WITHIN_ITEM;
}

export function computeVisiblePackAlphaByteOffset(index) {
  return computeVisiblePackBaseByteOffset(index) + GPU_VISIBLE_PACK_ALPHA_BYTE_OFFSET_WITHIN_ITEM;
}

export function createVisiblePackViewMap(floatArray) {
  return {
    raw: floatArray,

    // canonical fields
    centerPx: { field: getVisiblePackField('centerPx') },
    radiusPx: { field: getVisiblePackField('radiusPx') },
    depth: { field: getVisiblePackField('depth') },
    colorAlpha: { field: getVisiblePackField('colorAlpha') },
    conic: { field: getVisiblePackField('conic') },
    reserved: { field: getVisiblePackField('reserved') },
    misc: { field: getVisiblePackField('misc') },

    // compatibility aliases
    color: { field: getVisiblePackField('color') },
    opacity: { field: getVisiblePackField('opacity') },
    aabb: { field: getVisiblePackField('aabb') }
  };
}
