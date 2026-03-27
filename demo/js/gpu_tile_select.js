import { findMaxCountTile } from './gpu_tile_debug.js';

export function getDrawTileMode(globalObj = window) {
  return {
    showOverlay: !!globalObj.__GPU_TILE_DEBUG_OVERLAY__,
    drawSelectedOnly: !!globalObj.__GPU_TILE_DRAW_SELECTED_ONLY__,
    useMaxTile: globalObj.__GPU_TILE_USE_MAX_TILE__ !== false,
    selectedTileId: Number.isInteger(globalObj.__GPU_TILE_SELECTED_ID__)
      ? globalObj.__GPU_TILE_SELECTED_ID__
      : -1,
  };
}

export function setDefaultDrawTileMode(globalObj = window) {
  if (typeof globalObj.__GPU_TILE_DEBUG_OVERLAY__ === 'undefined') {
    globalObj.__GPU_TILE_DEBUG_OVERLAY__ = false;
  }
  if (typeof globalObj.__GPU_TILE_DRAW_SELECTED_ONLY__ === 'undefined') {
    globalObj.__GPU_TILE_DRAW_SELECTED_ONLY__ = false;
  }
  if (typeof globalObj.__GPU_TILE_USE_MAX_TILE__ === 'undefined') {
    globalObj.__GPU_TILE_USE_MAX_TILE__ = true;
  }
  if (typeof globalObj.__GPU_TILE_SELECTED_ID__ === 'undefined') {
    globalObj.__GPU_TILE_SELECTED_ID__ = -1;
  }
}

export function chooseFocusTileId(tileData, mode) {
  const maxInfo = findMaxCountTile(tileData.counts);

  if (mode.drawSelectedOnly) {
    if (mode.useMaxTile || mode.selectedTileId < 0) {
      return maxInfo.maxTileId;
    }
    if (mode.selectedTileId >= 0 && mode.selectedTileId < tileData.counts.length) {
      return mode.selectedTileId;
    }
  }

  return maxInfo.maxTileId;
}

export function buildDrawIndexList(visible, tileData, focusTileId, drawSelectedOnly) {
  if (!drawSelectedOnly || focusTileId < 0) {
    const out = new Uint32Array(visible.length);
    for (let i = 0; i < visible.length; i++) out[i] = i;
    return out;
  }

  const start = tileData.offsets[focusTileId];
  const end = tileData.offsets[focusTileId + 1];
  const tileIndices = tileData.indices.subarray(start, end);

  const out = new Uint32Array(tileIndices.length);
  out.set(tileIndices);
  return out;
}

export function formatTileSelectionState(mode, focusTileId) {
  return [
    `drawSelectedOnly=${mode.drawSelectedOnly}`,
    `useMaxTile=${mode.useMaxTile}`,
    `selectedTileId=${mode.selectedTileId}`,
    `focusTileId=${focusTileId}`
  ].join('  ');
}
