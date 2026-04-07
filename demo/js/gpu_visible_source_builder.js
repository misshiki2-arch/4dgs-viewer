export const GPU_SCREEN_SPACE_SOURCE_PACKED_CPU = 'packed-cpu';
export const GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP = 'packed-gpu-prep';
export const GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL = 'gpu-screen-experimental';
export const GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION = 1;

function toFiniteOr(value, fallback = 0) {
  return Number.isFinite(value) ? value : fallback;
}

function normalizeSourcePath(path, experimental = false) {
  if (path === GPU_SCREEN_SPACE_SOURCE_PACKED_CPU) return GPU_SCREEN_SPACE_SOURCE_PACKED_CPU;
  if (path === GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP) return GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP;
  if (path === GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL) return GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL;
  return experimental ? GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP : GPU_SCREEN_SPACE_SOURCE_PACKED_CPU;
}

function isExperimentalSourcePath(path) {
  return (
    path === GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP ||
    path === GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL
  );
}

function normalizeCenterPx(item) {
  if (Array.isArray(item?.centerPx) && item.centerPx.length >= 2) {
    return [toFiniteOr(item.centerPx[0], 0), toFiniteOr(item.centerPx[1], 0)];
  }
  return [toFiniteOr(item?.px, 0), toFiniteOr(item?.py, 0)];
}

function normalizeRadiusPx(item) {
  if (Number.isFinite(item?.radiusPx)) return item.radiusPx;
  if (Number.isFinite(item?.radius)) return item.radius;
  return 0;
}

function normalizeDepth(item) {
  return toFiniteOr(item?.depth, 0);
}

function normalizeColorAlpha(item) {
  if (Array.isArray(item?.colorAlpha) && item.colorAlpha.length >= 4) {
    return [
      toFiniteOr(item.colorAlpha[0], 0),
      toFiniteOr(item.colorAlpha[1], 0),
      toFiniteOr(item.colorAlpha[2], 0),
      toFiniteOr(item.colorAlpha[3], 0)
    ];
  }

  const color = Array.isArray(item?.color) ? item.color : [0, 0, 0, 0];
  const alpha = Number.isFinite(item?.opacity) ? item.opacity : toFiniteOr(color[3], 0);
  return [
    toFiniteOr(color[0], 0),
    toFiniteOr(color[1], 0),
    toFiniteOr(color[2], 0),
    toFiniteOr(alpha, 0)
  ];
}

function normalizeConic(item) {
  if (Array.isArray(item?.conic) && item.conic.length >= 3) {
    return [
      toFiniteOr(item.conic[0], 0),
      toFiniteOr(item.conic[1], 0),
      toFiniteOr(item.conic[2], 0)
    ];
  }
  return [0, 0, 0];
}

function normalizeReserved(item) {
  return toFiniteOr(item?.reserved, 0);
}

function normalizeMisc(item) {
  if (Array.isArray(item?.misc) && item.misc.length >= 4) {
    return [
      toFiniteOr(item.misc[0], 0),
      toFiniteOr(item.misc[1], 0),
      toFiniteOr(item.misc[2], 0),
      toFiniteOr(item.misc[3], 0)
    ];
  }

  if (Array.isArray(item?.aabb) && item.aabb.length >= 4) {
    return [
      toFiniteOr(item.aabb[0], 0),
      toFiniteOr(item.aabb[1], 0),
      toFiniteOr(item.aabb[2], 0),
      toFiniteOr(item.aabb[3], 0)
    ];
  }

  return [0, 0, 0, 0];
}

export function normalizeScreenSpaceItem(item) {
  return {
    ...item,
    centerPx: normalizeCenterPx(item),
    radiusPx: normalizeRadiusPx(item),
    depth: normalizeDepth(item),
    colorAlpha: normalizeColorAlpha(item),
    conic: normalizeConic(item),
    reserved: normalizeReserved(item),
    misc: normalizeMisc(item)
  };
}

export function normalizeScreenSpaceVisible(visible) {
  if (!Array.isArray(visible) || visible.length === 0) return [];
  const normalized = new Array(visible.length);
  for (let i = 0; i < visible.length; i++) {
    normalized[i] = normalizeScreenSpaceItem(visible[i]);
  }
  return normalized;
}

export function buildGpuScreenSourceItems(visible, extra = {}) {
  const normalizedVisible = normalizeScreenSpaceVisible(visible);
  const normalizedPath = normalizeSourcePath(extra.path, !!extra.experimental);
  const experimental = !!extra.experimental || isExperimentalSourcePath(normalizedPath);

  return {
    path: normalizedPath,
    experimental,
    schemaVersion: GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION,
    items: normalizedVisible,
    itemCount: normalizedVisible.length
  };
}
