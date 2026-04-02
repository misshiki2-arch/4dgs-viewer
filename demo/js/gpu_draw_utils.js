import {
  computeVisiblePackFieldFloatOffset,
  GPU_VISIBLE_PACK_FLOATS_PER_ITEM
} from './gpu_buffer_layout_utils.js';

// Step20:
// 従来の object-array draw 経路を維持しつつ、packed visible path の観測・将来接続に向けた
// 補助関数を追加する。

function ensureFloat32Array(value, name) {
  if (!(value instanceof Float32Array)) {
    throw new Error(`${name} must be a Float32Array`);
  }
}

export function buildDrawArraysFromIndices(visible, drawIndices) {
  const drawCount = drawIndices ? drawIndices.length : 0;

  const centers = new Float32Array(drawCount * 2);
  const radii = new Float32Array(drawCount);
  const colors = new Float32Array(drawCount * 4);
  const conics = new Float32Array(drawCount * 3);

  for (let j = 0; j < drawCount; j++) {
    const src = visible[drawIndices[j]];

    const c2 = j * 2;
    centers[c2 + 0] = src.px;
    centers[c2 + 1] = src.py;

    radii[j] = src.radius;

    const c4 = j * 4;
    colors[c4 + 0] = src.color[0];
    colors[c4 + 1] = src.color[1];
    colors[c4 + 2] = src.color[2];
    colors[c4 + 3] = src.color[3];

    const c3 = j * 3;
    conics[c3 + 0] = src.conic[0];
    conics[c3 + 1] = src.conic[1];
    conics[c3 + 2] = src.conic[2];
  }

  return {
    nDraw: drawCount,
    centers,
    radii,
    colors,
    conics
  };
}

export function buildDrawArraysFromPacked(packed, count) {
  ensureFloat32Array(packed, 'packed');
  const n = Math.max(0, count | 0);

  const centers = new Float32Array(n * 2);
  const radii = new Float32Array(n);
  const colors = new Float32Array(n * 4);
  const conics = new Float32Array(n * 3);

  for (let i = 0; i < n; i++) {
    const centerOffset = computeVisiblePackFieldFloatOffset(i, 'centerPx');
    const radiusOffset = computeVisiblePackFieldFloatOffset(i, 'radiusPx');
    const colorOffset = computeVisiblePackFieldFloatOffset(i, 'color');
    const conicOffset = computeVisiblePackFieldFloatOffset(i, 'conic');

    const c2 = i * 2;
    centers[c2 + 0] = packed[centerOffset + 0];
    centers[c2 + 1] = packed[centerOffset + 1];

    radii[i] = packed[radiusOffset];

    const c4 = i * 4;
    colors[c4 + 0] = packed[colorOffset + 0];
    colors[c4 + 1] = packed[colorOffset + 1];
    colors[c4 + 2] = packed[colorOffset + 2];
    colors[c4 + 3] = packed[colorOffset + 3];

    const c3 = i * 3;
    conics[c3 + 0] = packed[conicOffset + 0];
    conics[c3 + 1] = packed[conicOffset + 1];
    conics[c3 + 2] = packed[conicOffset + 2];
  }

  return {
    nDraw: n,
    centers,
    radii,
    colors,
    conics
  };
}

export function buildPerTileDrawBatches(visible, tileBatches) {
  if (!tileBatches || tileBatches.length === 0) return [];

  return tileBatches.map((batch) => ({
    tileId: batch.tileId,
    tileCount: batch.indices.length,
    drawData: buildDrawArraysFromIndices(visible, batch.indices)
  }));
}

export function summarizePerTileDrawBatches(perTileDrawBatches) {
  const batches = perTileDrawBatches || [];
  let totalTileDrawCount = 0;
  let maxTileDrawCount = 0;
  let maxTileId = -1;

  for (const batch of batches) {
    const n = batch?.drawData?.nDraw || 0;
    totalTileDrawCount += n;
    if (n > maxTileDrawCount) {
      maxTileDrawCount = n;
      maxTileId = batch.tileId;
    }
  }

  return {
    tileBatchCount: batches.length,
    totalTileDrawCount,
    maxTileDrawCount,
    maxTileId
  };
}

export function uploadAndDraw(gl, gpu, drawData, canvasWidth, canvasHeight) {
  gl.useProgram(gpu.program);
  gl.bindVertexArray(gpu.vao);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpu.centerBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, drawData.centers, gl.DYNAMIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpu.radiusBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, drawData.radii, gl.DYNAMIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpu.colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, drawData.colors, gl.DYNAMIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpu.conicBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, drawData.conics, gl.DYNAMIC_DRAW);

  gl.uniform2f(gpu.uViewportPx, canvasWidth, canvasHeight);
  gl.drawArrays(gl.POINTS, 0, drawData.nDraw);

  gl.bindVertexArray(null);
}

export function renderPerTileBatches(
  gl,
  gpu,
  perTileDrawBatches,
  canvasWidth,
  canvasHeight,
  hooks = {}
) {
  let uploadCount = 0;
  let drawCallCount = 0;

  for (const item of perTileDrawBatches) {
    if (hooks.beforeTile) hooks.beforeTile(item);

    uploadAndDraw(gl, gpu, item.drawData, canvasWidth, canvasHeight);
    uploadCount += 1;
    drawCallCount += 1;

    if (hooks.afterTile) hooks.afterTile(item);
  }

  return {
    tileBatchCount: perTileDrawBatches.length,
    uploadCount,
    drawCallCount
  };
}

export function buildPackedDrawStats(packedScreenSpace) {
  const packed = packedScreenSpace?.packed;
  const packedCount = packedScreenSpace?.packedCount ?? 0;
  const packedLength = packed instanceof Float32Array ? packed.length : 0;

  return {
    packedPath: packedScreenSpace?.path ?? 'none',
    packedCount,
    packedLength,
    packedFloatsPerItem: packedScreenSpace?.floatsPerItem ?? GPU_VISIBLE_PACK_FLOATS_PER_ITEM
  };
}

export function buildDrawStats({
  visibleCount,
  drawData,
  mode,
  focusTileId,
  focusTileIds,
  tileBatchSummary,
  executionSummary,
  packedScreenSpace = null
}) {
  const drawCount = drawData?.nDraw || 0;
  const packedStats = buildPackedDrawStats(packedScreenSpace);

  return {
    drawCount,
    visibleCount,
    drawFraction: visibleCount > 0 ? drawCount / visibleCount : 0,
    drawSelectedOnly: !!mode?.drawSelectedOnly,
    showOverlay: !!mode?.showOverlay,
    useMaxTile: !!mode?.useMaxTile,
    selectedTileId: mode?.selectedTileId ?? -1,
    tileRadius: mode?.tileRadius ?? 0,
    focusTileId,
    focusTileIds,
    tileBatchCount: tileBatchSummary?.tileBatchCount ?? executionSummary?.tileBatchCount ?? 0,
    totalTileDrawCount: tileBatchSummary?.totalTileDrawCount ?? drawCount,
    maxTileDrawCount: tileBatchSummary?.maxTileDrawCount ?? drawCount,
    maxTileDrawTileId: tileBatchSummary?.maxTileId ?? focusTileId ?? -1,
    uploadCount: executionSummary?.uploadCount ?? 0,
    drawCallCount: executionSummary?.drawCallCount ?? 0,
    executionTileBatchCount: executionSummary?.tileBatchCount ?? 0,
    packedPath: packedStats.packedPath,
    packedCount: packedStats.packedCount,
    packedLength: packedStats.packedLength,
    packedFloatsPerItem: packedStats.packedFloatsPerItem
  };
}
