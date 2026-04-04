import {
  uploadArrayBuffer,
  clearToGray,
  enableStandardAlphaBlend,
  disableDepth
} from './gpu_gl_utils.js';

// Step24:
// このファイルは legacy draw / per-tile draw / draw 統計の補助に責務を限定する。
// packed 正式契約の解釈はここでは持たない。
// packed path は gpu_packed_* 側へ寄せ、ここは legacy fallback のみを担当する。

function ensureFloat32Array(value) {
  if (value instanceof Float32Array) return value;
  if (Array.isArray(value)) return new Float32Array(value);
  return new Float32Array(0);
}

function ensureUint32Array(value) {
  if (value instanceof Uint32Array) return value;
  if (Array.isArray(value)) return new Uint32Array(value);
  return new Uint32Array(0);
}

function pushCenter(out, item) {
  out.push(
    Number.isFinite(item?.px) ? item.px : 0,
    Number.isFinite(item?.py) ? item.py : 0
  );
}

function pushRadius(out, item) {
  out.push(Number.isFinite(item?.radius) ? item.radius : 0);
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
  const alpha = Number.isFinite(item?.opacity)
    ? item.opacity
    : (Number.isFinite(color[3]) ? color[3] : 0);

  return [
    Number.isFinite(color[0]) ? color[0] : 0,
    Number.isFinite(color[1]) ? color[1] : 0,
    Number.isFinite(color[2]) ? color[2] : 0,
    Number.isFinite(alpha) ? alpha : 0
  ];
}

function pushColorAlpha(out, item) {
  const rgba = resolveColorAlpha(item);
  out.push(rgba[0], rgba[1], rgba[2], rgba[3]);
}

function pushConic(out, item) {
  if (Array.isArray(item?.conic) && item.conic.length >= 3) {
    out.push(
      Number.isFinite(item.conic[0]) ? item.conic[0] : 0,
      Number.isFinite(item.conic[1]) ? item.conic[1] : 0,
      Number.isFinite(item.conic[2]) ? item.conic[2] : 0
    );
    return;
  }
  out.push(0, 0, 0);
}

export function buildDrawArraysFromIndices(visible, drawIndices) {
  const src = Array.isArray(visible) ? visible : [];
  const indices = ensureUint32Array(drawIndices);

  const centers = [];
  const radii = [];
  const colors = [];
  const conics = [];

  for (let i = 0; i < indices.length; i++) {
    const idx = indices[i] | 0;
    if (idx < 0 || idx >= src.length) continue;
    const item = src[idx];
    pushCenter(centers, item);
    pushRadius(radii, item);
    pushColorAlpha(colors, item);
    pushConic(conics, item);
  }

  return {
    centers: new Float32Array(centers),
    radii: new Float32Array(radii),
    colors: new Float32Array(colors),
    conics: new Float32Array(conics),
    nDraw: radii.length
  };
}

export function buildPerTileDrawBatches(visible, tileBatches) {
  const src = Array.isArray(tileBatches) ? tileBatches : [];
  const out = new Array(src.length);

  for (let i = 0; i < src.length; i++) {
    const batch = src[i] || {};
    const drawData = buildDrawArraysFromIndices(visible, batch.indices);
    out[i] = {
      tileId: Number.isInteger(batch.tileId) ? batch.tileId : -1,
      indices: ensureUint32Array(batch.indices),
      drawData
    };
  }

  return out;
}

export function summarizePerTileDrawBatches(perTileDrawBatches) {
  const batches = Array.isArray(perTileDrawBatches) ? perTileDrawBatches : [];
  let totalTileDrawCount = 0;
  let nonEmptyTileBatchCount = 0;
  let maxTileDrawCount = 0;
  let maxTileId = -1;

  for (const batch of batches) {
    const n = batch?.drawData?.nDraw ?? 0;
    totalTileDrawCount += n;
    if (n > 0) {
      nonEmptyTileBatchCount++;
      if (n > maxTileDrawCount) {
        maxTileDrawCount = n;
        maxTileId = Number.isInteger(batch?.tileId) ? batch.tileId : -1;
      }
    }
  }

  return {
    tileBatchCount: batches.length,
    nonEmptyTileBatchCount,
    totalTileDrawCount,
    maxTileDrawCount,
    maxTileId
  };
}

export function uploadAndDraw(gl, gpu, drawData, canvasWidth, canvasHeight) {
  const centers = ensureFloat32Array(drawData?.centers);
  const radii = ensureFloat32Array(drawData?.radii);
  const colors = ensureFloat32Array(drawData?.colors);
  const conics = ensureFloat32Array(drawData?.conics);
  const nDraw = Math.max(0, drawData?.nDraw | 0);

  uploadArrayBuffer(gl, gpu.centerBuffer, centers, gl.DYNAMIC_DRAW);
  uploadArrayBuffer(gl, gpu.radiusBuffer, radii, gl.DYNAMIC_DRAW);
  uploadArrayBuffer(gl, gpu.colorBuffer, colors, gl.DYNAMIC_DRAW);
  uploadArrayBuffer(gl, gpu.conicBuffer, conics, gl.DYNAMIC_DRAW);

  gl.useProgram(gpu.program);
  gl.bindVertexArray(gpu.vao);
  gl.uniform2f(gpu.uViewportPx, canvasWidth, canvasHeight);
  gl.drawArrays(gl.POINTS, 0, nDraw);
  gl.bindVertexArray(null);

  return {
    uploadCount: 4,
    drawCallCount: 1,
    nDraw
  };
}

export function renderPerTileBatches(
  gl,
  gpu,
  perTileDrawBatches,
  canvasWidth,
  canvasHeight,
  hooks = {}
) {
  const batches = Array.isArray(perTileDrawBatches) ? perTileDrawBatches : [];
  let drawCallCount = 0;
  let uploadCount = 0;
  let totalTileDrawCount = 0;
  let nonEmptyTileBatchCount = 0;

  for (const batch of batches) {
    const nDraw = batch?.drawData?.nDraw ?? 0;
    if (nDraw <= 0) continue;

    if (typeof hooks.beforeTile === 'function') {
      hooks.beforeTile(batch);
    }

    const drawSummary = uploadAndDraw(gl, gpu, batch.drawData, canvasWidth, canvasHeight);

    if (typeof hooks.afterTile === 'function') {
      hooks.afterTile(batch, drawSummary);
    }

    drawCallCount += drawSummary.drawCallCount ?? 0;
    uploadCount += drawSummary.uploadCount ?? 0;
    totalTileDrawCount += nDraw;
    nonEmptyTileBatchCount++;
  }

  return {
    tileBatchCount: batches.length,
    nonEmptyTileBatchCount,
    totalTileDrawCount,
    uploadCount,
    drawCallCount
  };
}

export function buildDrawStats({
  visibleCount = 0,
  drawData = null,
  mode = null,
  focusTileId = -1,
  focusTileIds = [],
  tileBatchSummary = null,
  executionSummary = null,
  packedScreenSpace = null,
  packedUploadSummary = null
} = {}) {
  const nDraw = drawData?.nDraw ?? 0;
  const packed = packedScreenSpace?.packed;

  return {
    visibleCount: Number.isFinite(visibleCount) ? visibleCount : 0,
    drawCount: Number.isFinite(nDraw) ? nDraw : 0,

    drawSelectedOnly: !!mode?.drawSelectedOnly,
    showOverlay: !!mode?.showOverlay,
    focusTileId: Number.isInteger(focusTileId) ? focusTileId : -1,
    focusTileIds: Array.isArray(focusTileIds) ? focusTileIds.slice() : [],
    focusTileCount: Array.isArray(focusTileIds) ? focusTileIds.length : 0,

    tileBatchCount: tileBatchSummary?.tileBatchCount ?? executionSummary?.tileBatchCount ?? 0,
    nonEmptyTileBatchCount:
      tileBatchSummary?.nonEmptyTileBatchCount ??
      executionSummary?.nonEmptyTileBatchCount ??
      0,
    totalTileDrawCount:
      tileBatchSummary?.totalTileDrawCount ??
      executionSummary?.totalTileDrawCount ??
      nDraw,
    maxTileDrawCount: tileBatchSummary?.maxTileDrawCount ?? 0,
    maxTileId: tileBatchSummary?.maxTileId ?? -1,

    uploadCount: executionSummary?.uploadCount ?? 0,
    drawCallCount: executionSummary?.drawCallCount ?? 0,
    requestedDrawPath: executionSummary?.requestedDrawPath ?? 'legacy',
    actualDrawPath: executionSummary?.actualDrawPath ?? 'legacy',
    drawPathFallbackReason: executionSummary?.drawPathFallbackReason ?? 'none',

    packedVisiblePath: packedScreenSpace?.path ?? 'none',
    packedVisibleCount: Number.isFinite(packedScreenSpace?.packedCount)
      ? packedScreenSpace.packedCount
      : 0,
    packedVisibleLength: packed instanceof Float32Array ? packed.length : 0,
    packedVisibleFloatsPerItem: Number.isFinite(packedScreenSpace?.floatsPerItem)
      ? packedScreenSpace.floatsPerItem
      : 0,

    packedUploadLayoutVersion: packedUploadSummary?.packedUploadLayoutVersion ?? 0,
    packedUploadStrideBytes: packedUploadSummary?.packedUploadStrideBytes ?? 0,
    packedUploadBytes: packedUploadSummary?.packedUploadBytes ?? 0,
    packedUploadCount: packedUploadSummary?.packedUploadCount ?? 0,
    packedUploadLength: packedUploadSummary?.packedUploadLength ?? 0,
    packedUploadCapacityBytes: packedUploadSummary?.packedUploadCapacityBytes ?? 0,
    packedUploadReusedCapacity: !!packedUploadSummary?.packedUploadReusedCapacity,
    packedUploadManagedCapacityReused:
      !!packedUploadSummary?.packedUploadManagedCapacityReused,
    packedUploadManagedCapacityGrown:
      !!packedUploadSummary?.packedUploadManagedCapacityGrown,
    packedUploadManagedUploadCount:
      packedUploadSummary?.packedUploadManagedUploadCount ?? 0,
    packedUploadAlphaSource: packedUploadSummary?.packedUploadAlphaSource ?? '',

    packedDirectDraw: !!packedUploadSummary?.packedDirectDraw,
    packedDirectConfigured: !!packedUploadSummary?.packedDirectConfigured,
    packedDirectHasVao: !!packedUploadSummary?.packedDirectHasVao,
    packedDirectLayoutVersion: packedUploadSummary?.packedDirectLayoutVersion ?? 0,
    packedDirectStrideBytes: packedUploadSummary?.packedDirectStrideBytes ?? 0,
    packedDirectAttributeCount: packedUploadSummary?.packedDirectAttributeCount ?? 0,
    packedDirectOffsets: packedUploadSummary?.packedDirectOffsets ?? '',
    packedDirectAlphaSource: packedUploadSummary?.packedDirectAlphaSource ?? '',
    packedInterleavedBound: !!packedUploadSummary?.packedInterleavedBound,

    legacyExpandedArraysBuilt: !!packedUploadSummary?.legacyExpandedArraysBuilt
  };
}

export function prepareLegacyDrawFrame(gl, bgGray01) {
  disableDepth(gl);
  enableStandardAlphaBlend(gl);
  clearToGray(gl, bgGray01);
}