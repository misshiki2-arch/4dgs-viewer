export function clamp01(x) {
  return Math.min(1, Math.max(0, x));
}

export function tileIdToXY(tileId, tileCols) {
  return {
    tx: tileId % tileCols,
    ty: Math.floor(tileId / tileCols),
  };
}

export function tileXYToId(tx, ty, tileCols) {
  return ty * tileCols + tx;
}

export function getTilePixelRect(tx, ty, tileSize, canvasWidth, canvasHeight) {
  const x0 = tx * tileSize;
  const y0 = ty * tileSize;
  const x1 = Math.min(canvasWidth, x0 + tileSize);
  const y1 = Math.min(canvasHeight, y0 + tileSize);
  return [x0, y0, x1, y1];
}

export function findMaxCountTile(tileCounts) {
  let maxCount = -1;
  let maxTileId = -1;
  for (let i = 0; i < tileCounts.length; i++) {
    if (tileCounts[i] > maxCount) {
      maxCount = tileCounts[i];
      maxTileId = i;
    }
  }
  return { maxTileId, maxCount };
}

export function getTileSplatIndices(tileData, tileId, limit = 32) {
  if (!tileData || !tileData.offsets || !tileData.indices) return [];
  if (tileId < 0 || tileId + 1 >= tileData.offsets.length) return [];
  const start = tileData.offsets[tileId];
  const end = tileData.offsets[tileId + 1];
  const out = [];
  for (let i = start; i < Math.min(end, start + limit); i++) {
    out.push(tileData.indices[i]);
  }
  return out;
}

export function buildTileHeatmapImageData({
  tileCounts,
  tileCols,
  tileRows,
  tileSize,
  canvasWidth,
  canvasHeight,
  highlightTileId = -1,
  selectedTileIds = null,
  alpha = 0.65
}) {
  const width = canvasWidth;
  const height = canvasHeight;
  const img = new ImageData(width, height);
  const data = img.data;

  let maxCount = 0;
  for (let i = 0; i < tileCounts.length; i++) {
    if (tileCounts[i] > maxCount) maxCount = tileCounts[i];
  }
  const invMax = maxCount > 0 ? 1.0 / maxCount : 0.0;

  const selected = selectedTileIds ? new Set(selectedTileIds) : null;

  for (let ty = 0; ty < tileRows; ty++) {
    for (let tx = 0; tx < tileCols; tx++) {
      const tileId = ty * tileCols + tx;
      const count = tileCounts[tileId];
      if (count <= 0) continue;

      const [x0, y0, x1, y1] = getTilePixelRect(tx, ty, tileSize, width, height);
      const t = clamp01(count * invMax);

      let r = 0, g = 0, b = 0;
      if (t < 0.33) {
        const u = t / 0.33;
        r = 0;
        g = Math.round(255 * u);
        b = 255;
      } else if (t < 0.66) {
        const u = (t - 0.33) / 0.33;
        r = Math.round(255 * u);
        g = 255;
        b = Math.round(255 * (1.0 - u));
      } else {
        const u = (t - 0.66) / 0.34;
        r = 255;
        g = Math.round(255 * (1.0 - 0.7 * u));
        b = 0;
      }

      const isHighlight = tileId === highlightTileId;
      const isSelected = selected ? selected.has(tileId) : false;
      if (isHighlight) {
        r = 255; g = 255; b = 255;
      } else if (isSelected) {
        r = 255; g = 128; b = 255;
      }

      const a = Math.round(255 * alpha);

      for (let y = y0; y < y1; y++) {
        const rowBase = y * width;
        for (let x = x0; x < x1; x++) {
          const p = (rowBase + x) * 4;
          data[p + 0] = r;
          data[p + 1] = g;
          data[p + 2] = b;
          data[p + 3] = a;
        }
      }

      for (let x = x0; x < x1; x++) {
        let p0 = (y0 * width + x) * 4;
        let p1 = ((y1 - 1) * width + x) * 4;
        data[p0 + 0] = 255; data[p0 + 1] = 255; data[p0 + 2] = 255; data[p0 + 3] = 90;
        data[p1 + 0] = 255; data[p1 + 1] = 255; data[p1 + 2] = 255; data[p1 + 3] = 90;
      }
      for (let y = y0; y < y1; y++) {
        let p0 = (y * width + x0) * 4;
        let p1 = (y * width + (x1 - 1)) * 4;
        data[p0 + 0] = 255; data[p0 + 1] = 255; data[p0 + 2] = 255; data[p0 + 3] = 90;
        data[p1 + 0] = 255; data[p1 + 1] = 255; data[p1 + 2] = 255; data[p1 + 3] = 90;
      }
    }
  }

  return img;
}

export function drawTileHeatmapOverlay(ctx2d, img) {
  if (!ctx2d || !img) return;
  ctx2d.putImageData(img, 0, 0);
}

export function formatTileDebugSummary({
  tileData,
  tileCols,
  tileRows,
  tileSize,
  highlightTileId = -1,
  canvasWidth = 0,
  canvasHeight = 0
}) {
  if (!tileData || !tileData.counts) return 'tile debug: no tile data';

  const { maxTileId, maxCount } = findMaxCountTile(tileData.counts);
  const hiId = highlightTileId >= 0 ? highlightTileId : maxTileId;
  const { tx, ty } = tileIdToXY(hiId, tileCols);
  const rect = getTilePixelRect(tx, ty, tileSize, canvasWidth, canvasHeight);
  const sample = getTileSplatIndices(tileData, hiId, 12);

  return [
    `tile debug`,
    `tileSize=${tileSize}  tileCols=${tileCols}  tileRows=${tileRows}`,
    `maxTileId=${maxTileId}  maxCount=${maxCount}`,
    `focusTile=${hiId} -> (${tx},${ty})`,
    `focusRect=[${rect.join(', ')}]`,
    `focusFirstIndices=[${sample.join(', ')}]`,
  ].join('\n');
}
