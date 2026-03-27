export function clamp01(x) {
  return Math.min(1, Math.max(0, x));
}

export function buildDrawArraysFromIndices(visible, drawIndices) {
  const nDraw = drawIndices.length;
  const centers = new Float32Array(nDraw * 2);
  const radii = new Float32Array(nDraw);
  const colors = new Float32Array(nDraw * 4);
  const conics = new Float32Array(nDraw * 3);

  for (let k = 0; k < nDraw; k++) {
    const s = visible[drawIndices[k]];
    centers[2 * k + 0] = s.px;
    centers[2 * k + 1] = s.py;
    radii[k] = s.radius;
    colors[4 * k + 0] = clamp01(s.color[0]);
    colors[4 * k + 1] = clamp01(s.color[1]);
    colors[4 * k + 2] = clamp01(s.color[2]);
    colors[4 * k + 3] = clamp01(s.opacity);
    conics[3 * k + 0] = s.conic[0];
    conics[3 * k + 1] = s.conic[1];
    conics[3 * k + 2] = s.conic[2];
  }

  return {
    nDraw,
    centers,
    radii,
    colors,
    conics
  };
}

export function buildPerTileDrawArrays(visible, tileBatches) {
  const out = [];
  for (const batch of tileBatches) {
    const drawData = buildDrawArraysFromIndices(visible, batch.drawIndices);
    out.push({
      tileId: batch.tileId,
      drawIndices: batch.drawIndices,
      drawCount: batch.drawCount,
      drawData
    });
  }
  return out;
}

export function summarizePerTileDrawArrays(perTileDrawArrays) {
  let totalTileDrawCount = 0;
  let maxTileDrawCount = 0;
  let maxTileId = -1;

  for (const item of perTileDrawArrays) {
    totalTileDrawCount += item.drawCount;
    if (item.drawCount > maxTileDrawCount) {
      maxTileDrawCount = item.drawCount;
      maxTileId = item.tileId;
    }
  }

  return {
    tileBatchCount: perTileDrawArrays.length,
    totalTileDrawCount,
    maxTileDrawCount,
    maxTileId,
    avgTileDrawCount: perTileDrawArrays.length > 0
      ? (totalTileDrawCount / perTileDrawArrays.length)
      : 0
  };
}

export function uploadDrawArrays(gl, gpu, drawData) {
  gl.bindVertexArray(gpu.vao);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpu.centerBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, drawData.centers, gl.DYNAMIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpu.radiusBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, drawData.radii, gl.DYNAMIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpu.colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, drawData.colors, gl.DYNAMIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpu.conicBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, drawData.conics, gl.DYNAMIC_DRAW);
}

export function drawUploadedArrays(gl, gpu, viewportWidth, viewportHeight, nDraw) {
  gl.useProgram(gpu.program);
  gl.uniform2f(gpu.uViewportPx, viewportWidth, viewportHeight);
  gl.bindVertexArray(gpu.vao);
  gl.drawArrays(gl.POINTS, 0, nDraw);
  gl.bindVertexArray(null);
  gl.useProgram(null);
}

export function uploadAndDraw(gl, gpu, drawData, viewportWidth, viewportHeight) {
  uploadDrawArrays(gl, gpu, drawData);
  drawUploadedArrays(gl, gpu, viewportWidth, viewportHeight, drawData.nDraw);
}

export function uploadAndDrawPerTile(gl, gpu, perTileDrawArrays, viewportWidth, viewportHeight, drawTileFn) {
  for (const item of perTileDrawArrays) {
    if (drawTileFn) drawTileFn(item);
    uploadAndDraw(gl, gpu, item.drawData, viewportWidth, viewportHeight);
  }
}

export function buildDrawStats({
  visibleCount,
  drawData,
  mode = null,
  focusTileId = -1,
  focusTileIds = null,
  tileBatchSummary = null
}) {
  return {
    visibleCount,
    drawCount: drawData ? drawData.nDraw : 0,
    drawFraction: visibleCount > 0 && drawData ? (drawData.nDraw / visibleCount) : 0,
    drawSelectedOnly: mode ? !!mode.drawSelectedOnly : false,
    showOverlay: mode ? !!mode.showOverlay : false,
    useMaxTile: mode ? !!mode.useMaxTile : false,
    selectedTileId: mode ? mode.selectedTileId : -1,
    tileRadius: mode ? mode.tileRadius : 0,
    focusTileId,
    focusTileIds: focusTileIds || [],
    tileBatchSummary
  };
}

export function formatDrawStats(drawStats) {
  const idsText = drawStats.focusTileIds && drawStats.focusTileIds.length > 0
    ? `[${drawStats.focusTileIds.join(', ')}]`
    : 'none';

  const lines = [
    `drawCount=${drawStats.drawCount}`,
    `visibleCount=${drawStats.visibleCount}`,
    `drawFraction=${drawStats.drawFraction.toFixed(3)}`,
    `drawSelectedOnly=${drawStats.drawSelectedOnly}`,
    `showOverlay=${drawStats.showOverlay}`,
    `useMaxTile=${drawStats.useMaxTile}`,
    `selectedTileId=${drawStats.selectedTileId}`,
    `tileRadius=${drawStats.tileRadius}`,
    `focusTileId=${drawStats.focusTileId}`,
    `focusTileIds=${idsText}`
  ];

  if (drawStats.tileBatchSummary) {
    lines.push(`tileBatchCount=${drawStats.tileBatchSummary.tileBatchCount}`);
    lines.push(`totalTileDrawCount=${drawStats.tileBatchSummary.totalTileDrawCount}`);
    lines.push(`maxTileDrawCount=${drawStats.tileBatchSummary.maxTileDrawCount}`);
    lines.push(`maxTileId=${drawStats.tileBatchSummary.maxTileId}`);
    lines.push(`avgTileDrawCount=${drawStats.tileBatchSummary.avgTileDrawCount.toFixed(2)}`);
  }

  return lines.join('  ');
}
