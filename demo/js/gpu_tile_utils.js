export function clampInt(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

export function computeTileGrid(width, height, tileSize = 32) {
  return {
    tileSize,
    tileCols: Math.ceil(width / tileSize),
    tileRows: Math.ceil(height / tileSize),
  };
}

export function computeTileRangeFromAABB(aabb, tileCols, tileRows, tileSize = 32) {
  const [minX, minY, maxX, maxY] = aabb;
  const tminX = clampInt(Math.floor(minX / tileSize), 0, tileCols - 1);
  const tmaxX = clampInt(Math.floor(maxX / tileSize), 0, tileCols - 1);
  const tminY = clampInt(Math.floor(minY / tileSize), 0, tileRows - 1);
  const tmaxY = clampInt(Math.floor(maxY / tileSize), 0, tileRows - 1);
  return [tminX, tminY, tmaxX, tmaxY];
}

export function buildTileLists(visible, tileCols, tileRows) {
  const tileCount = tileCols * tileRows;
  const counts = new Uint32Array(tileCount);

  for (let i = 0; i < visible.length; i++) {
    const tr = visible[i].tileRange;
    for (let ty = tr[1]; ty <= tr[3]; ty++) {
      const rowBase = ty * tileCols;
      for (let tx = tr[0]; tx <= tr[2]; tx++) {
        counts[rowBase + tx]++;
      }
    }
  }

  const offsets = new Uint32Array(tileCount + 1);
  for (let i = 0; i < tileCount; i++) {
    offsets[i + 1] = offsets[i] + counts[i];
  }

  const totalRefs = offsets[tileCount];
  const indices = new Uint32Array(totalRefs);
  const write = offsets.slice(0, tileCount);

  for (let i = 0; i < visible.length; i++) {
    const tr = visible[i].tileRange;
    for (let ty = tr[1]; ty <= tr[3]; ty++) {
      const rowBase = ty * tileCols;
      for (let tx = tr[0]; tx <= tr[2]; tx++) {
        const tileId = rowBase + tx;
        indices[write[tileId]++] = i;
      }
    }
  }

  return { counts, offsets, indices, totalRefs };
}

export function summarizeTileLists(tileData, tileCols, tileRows, activeTileBox = null) {
  let maxPerTile = 0;
  let nonEmptyTiles = 0;
  let sumSq = 0;
  let maxTileId = -1;

  for (let i = 0; i < tileData.counts.length; i++) {
    const c = tileData.counts[i];
    if (c > 0) {
      nonEmptyTiles++;
      sumSq += c * c;
      if (c > maxPerTile) {
        maxPerTile = c;
        maxTileId = i;
      }
    }
  }

  let sampleTileText = 'none';
  if (maxTileId >= 0) {
    const start = tileData.offsets[maxTileId];
    const end = tileData.offsets[maxTileId + 1];
    const sample = [];
    for (let i = start; i < Math.min(end, start + 8); i++) {
      sample.push(tileData.indices[i]);
    }
    const tx = maxTileId % tileCols;
    const ty = Math.floor(maxTileId / tileCols);
    sampleTileText = `tile(${tx},${ty}) first=[${sample.join(', ')}]`;
  }

  const avgPerNonEmptyTile = nonEmptyTiles > 0 ? (tileData.totalRefs / nonEmptyTiles) : 0;

  let activeTileBoxText = 'none';
  if (activeTileBox && activeTileBox[2] >= activeTileBox[0] && activeTileBox[3] >= activeTileBox[1]) {
    activeTileBoxText = `${activeTileBox[0]},${activeTileBox[1]} -> ${activeTileBox[2]},${activeTileBox[3]}`;
  }

  return {
    nonEmptyTiles,
    maxPerTile,
    maxTileId,
    avgPerNonEmptyTile,
    countEnergy: sumSq,
    sampleTileText,
    activeTileBoxText,
    offsetsLen: tileData.offsets.length,
    indicesLen: tileData.indices.length,
    totalRefs: tileData.totalRefs,
    tileCols,
    tileRows,
  };
}
