import { findMaxCountTile } from './gpu_tile_debug.js';

export function getDrawTileMode(globalObj = window, ui = null) {
  const mode = {
    showOverlay: !!globalObj.__GPU_TILE_DEBUG_OVERLAY__,
    drawSelectedOnly: !!globalObj.__GPU_TILE_DRAW_SELECTED_ONLY__,
    useMaxTile: globalObj.__GPU_TILE_USE_MAX_TILE__ !== false,
    selectedTileId: Number.isInteger(globalObj.__GPU_TILE_SELECTED_ID__)
      ? globalObj.__GPU_TILE_SELECTED_ID__
      : -1,
    tileRadius: Number.isInteger(globalObj.__GPU_TILE_RADIUS__)
      ? Math.max(0, globalObj.__GPU_TILE_RADIUS__)
      : 0,
  };

  if (ui) {
    if (ui.showTileDebugCheck) {
      mode.showOverlay = !!ui.showTileDebugCheck.checked;
    }
    if (ui.drawSelectedTileOnlyCheck) {
      mode.drawSelectedOnly = !!ui.drawSelectedTileOnlyCheck.checked;
    }
    if (ui.useMaxTileCheck) {
      mode.useMaxTile = !!ui.useMaxTileCheck.checked;
    }
    if (ui.selectedTileIdInput) {
      const v = Number(ui.selectedTileIdInput.value);
      mode.selectedTileId = Number.isInteger(v) ? v : -1;
    }
    if (ui.tileRadiusInput) {
      const r = Number(ui.tileRadiusInput.value);
      mode.tileRadius = Number.isInteger(r) ? Math.max(0, r) : 0;
    }
  }

  return mode;
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
  if (typeof globalObj.__GPU_TILE_RADIUS__ === 'undefined') {
    globalObj.__GPU_TILE_RADIUS__ = 0;
  }
}

export function syncDrawTileModeToGlobals(mode, globalObj = window) {
  globalObj.__GPU_TILE_DEBUG_OVERLAY__ = !!mode.showOverlay;
  globalObj.__GPU_TILE_DRAW_SELECTED_ONLY__ = !!mode.drawSelectedOnly;
  globalObj.__GPU_TILE_USE_MAX_TILE__ = !!mode.useMaxTile;
  globalObj.__GPU_TILE_SELECTED_ID__ = Number.isInteger(mode.selectedTileId)
    ? mode.selectedTileId
    : -1;
  globalObj.__GPU_TILE_RADIUS__ = Number.isInteger(mode.tileRadius)
    ? Math.max(0, mode.tileRadius)
    : 0;
}

export function syncDrawTileModeFromUI(ui, globalObj = window) {
  const mode = getDrawTileMode(globalObj, ui);
  syncDrawTileModeToGlobals(mode, globalObj);

  if (ui && ui.selectedTileIdInput && ui.selectedTileIdNote) {
    const manualEnabled = mode.drawSelectedOnly && !mode.useMaxTile;
    ui.selectedTileIdInput.disabled = !manualEnabled;
    ui.selectedTileIdNote.textContent = manualEnabled
      ? 'manual tile id'
      : 'used only when max tile is off';
  }

  if (ui && ui.tileRadiusInput && ui.tileRadiusNote) {
    const radiusEnabled = mode.drawSelectedOnly;
    ui.tileRadiusInput.disabled = !radiusEnabled;
    ui.tileRadiusNote.textContent = radiusEnabled
      ? '0=single, 1=3x3, 2=5x5'
      : 'used only when single/multi-tile draw is on';
  }

  return mode;
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

export function buildNeighborTileIds(focusTileId, tileCols, tileRows, radius) {
  if (focusTileId < 0) return [];
  const fx = focusTileId % tileCols;
  const fy = Math.floor(focusTileId / tileCols);
  const ids = [];
  for (let ty = Math.max(0, fy - radius); ty <= Math.min(tileRows - 1, fy + radius); ty++) {
    for (let tx = Math.max(0, fx - radius); tx <= Math.min(tileCols - 1, fx + radius); tx++) {
      ids.push(ty * tileCols + tx);
    }
  }
  return ids;
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

export function buildMultiTileDrawIndexList(visible, tileData, tileIds, drawSelectedOnly) {
  if (!drawSelectedOnly || !tileIds || tileIds.length === 0) {
    const out = new Uint32Array(visible.length);
    for (let i = 0; i < visible.length; i++) out[i] = i;
    return out;
  }

  const mark = new Uint8Array(visible.length);
  const ordered = [];

  for (const tileId of tileIds) {
    if (tileId < 0 || tileId + 1 >= tileData.offsets.length) continue;
    const start = tileData.offsets[tileId];
    const end = tileData.offsets[tileId + 1];
    for (let i = start; i < end; i++) {
      const idx = tileData.indices[i];
      if (!mark[idx]) {
        mark[idx] = 1;
        ordered.push(idx);
      }
    }
  }

  const out = new Uint32Array(ordered.length);
  for (let i = 0; i < ordered.length; i++) out[i] = ordered[i];
  return out;
}

export function formatTileSelectionState(mode, focusTileId, focusTileIds = null) {
  const idsText = focusTileIds && focusTileIds.length > 0
    ? `[${focusTileIds.join(', ')}]`
    : 'none';

  return [
    `showOverlay=${mode.showOverlay}`,
    `drawSelectedOnly=${mode.drawSelectedOnly}`,
    `useMaxTile=${mode.useMaxTile}`,
    `selectedTileId=${mode.selectedTileId}`,
    `tileRadius=${mode.tileRadius}`,
    `focusTileId=${focusTileId}`,
    `focusTileIds=${idsText}`
  ].join('  ');
}
