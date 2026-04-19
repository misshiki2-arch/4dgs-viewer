import { packVisibleItems } from './gpu_visible_pack_utils.js';
import { getTilePixelRect } from './gpu_tile_debug.js';

function compareTileCompositeRefs(visible, a, b) {
  const itemA = visible[a];
  const itemB = visible[b];
  const depthA = Number.isFinite(itemA?.depth) ? itemA.depth : Infinity;
  const depthB = Number.isFinite(itemB?.depth) ? itemB.depth : Infinity;
  if (depthA !== depthB) return depthA - depthB;

  const srcA = Number.isFinite(itemA?.srcIndex) ? itemA.srcIndex : a;
  const srcB = Number.isFinite(itemB?.srcIndex) ? itemB.srcIndex : b;
  if (srcA !== srcB) return srcA - srcB;
  return a - b;
}

function buildTileCompositeBatch(visible, tileData, tileGrid, canvasWidth, canvasHeight, tileId) {
  if (tileId < 0 || tileId + 1 >= tileData.offsets.length) return null;

  const start = tileData.offsets[tileId];
  const end = tileData.offsets[tileId + 1];
  const refCount = Math.max(0, end - start);
  if (refCount <= 0) return null;

  const orderedIndices = new Uint32Array(refCount);
  orderedIndices.set(tileData.indices.subarray(start, end));
  orderedIndices.sort((a, b) => compareTileCompositeRefs(visible, a, b));

  const orderedVisible = new Array(refCount);
  const sourceIndices = new Uint32Array(refCount);
  for (let i = 0; i < refCount; i++) {
    const visibleIndex = orderedIndices[i] | 0;
    orderedVisible[i] = visible[visibleIndex];
    sourceIndices[i] = Number.isFinite(visible[visibleIndex]?.srcIndex)
      ? visible[visibleIndex].srcIndex
      : visibleIndex;
  }

  const packedResult = packVisibleItems(orderedVisible);
  const tx = tileId % tileGrid.tileCols;
  const ty = Math.floor(tileId / tileGrid.tileCols);

  return {
    tileId,
    tx,
    ty,
    rect: getTilePixelRect(tx, ty, tileGrid.tileSize, canvasWidth, canvasHeight),
    packed: packedResult.packed,
    packedCount: packedResult.count,
    floatsPerItem: packedResult.floatsPerItem,
    orderedIndices,
    sourceIndices
  };
}

export function buildTileCompositePlan({
  visible,
  tileData,
  tileGrid,
  canvasWidth,
  canvasHeight
}) {
  if (!Array.isArray(visible) || visible.length <= 0 || !tileData || !tileGrid) {
    return {
      batches: [],
      summary: {
        compositingContract: 'tile-local-front-to-back',
        depthOrder: 'ascending-near-to-far',
        tileBatchCount: 0,
        nonEmptyTileBatchCount: 0,
        totalTileDrawCount: 0,
        maxTileDrawCount: 0,
        maxTileId: -1,
        avgTileDrawCount: 0,
        tileCompositeDuplicateRefs: 0,
        tileCompositeOverlapFactor: 0,
        tileCompositeTileSize: tileGrid?.tileSize ?? 0,
        tileCompositeTileCols: tileGrid?.tileCols ?? 0,
        tileCompositeTileRows: tileGrid?.tileRows ?? 0
      }
    };
  }

  const tileCount = tileGrid.tileCols * tileGrid.tileRows;
  const batches = [];
  let totalTileDrawCount = 0;
  let maxTileDrawCount = 0;
  let maxTileId = -1;

  for (let tileId = 0; tileId < tileCount; tileId++) {
    const batch = buildTileCompositeBatch(
      visible,
      tileData,
      tileGrid,
      canvasWidth,
      canvasHeight,
      tileId
    );
    if (!batch) continue;
    batches.push(batch);
    totalTileDrawCount += batch.packedCount;
    if (batch.packedCount > maxTileDrawCount) {
      maxTileDrawCount = batch.packedCount;
      maxTileId = tileId;
    }
  }

  const duplicateRefs = Math.max(0, totalTileDrawCount - visible.length);
  const overlapFactor = visible.length > 0 ? totalTileDrawCount / visible.length : 0;

  return {
    batches,
    summary: {
      compositingContract: 'tile-local-front-to-back',
      depthOrder: 'ascending-near-to-far',
      tileBatchCount: batches.length,
      nonEmptyTileBatchCount: batches.length,
      totalTileDrawCount,
      maxTileDrawCount,
      maxTileId,
      avgTileDrawCount: batches.length > 0 ? (totalTileDrawCount / batches.length) : 0,
      tileCompositeDuplicateRefs: duplicateRefs,
      tileCompositeOverlapFactor: overlapFactor,
      tileCompositeTileSize: tileGrid.tileSize,
      tileCompositeTileCols: tileGrid.tileCols,
      tileCompositeTileRows: tileGrid.tileRows
    }
  };
}
