import { findMaxCountTile } from './gpu_tile_debug.js';

export function getDrawTileMode(globalObj = window, ui = null) {
  const mode = {
    showOverlay: !!globalObj.__GPU_TILE_DEBUG_OVERLAY__,
    drawSelectedOnly: !!globalObj.__GPU_TILE_DRAW_SELECTED_ONLY__,
    useMaxTile: globalObj.__GPU_TILE_USE_MAX_TILE__ !== false,
    selectedTileId: Number.isInteger(globalObj.__GPU_TILE_SELECTED_ID__)
      ? globalObj.__GPU_TILE_SELECTED_ID__
      : -1,
  };

  // If UI exists, UI values take precedence.
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
}

export function syncDrawTileModeToGlobals(mode, globalObj = window) {
  globalObj.__GPU_TILE_DEBUG_OVERLAY__ = !!mode.showOverlay;
  globalObj.__GPU_TILE_DRAW_SELECTED_ONLY__ = !!mode.drawSelectedOnly;
  globalObj.__GPU_TILE_USE_MAX_TILE__ = !!mode.useMaxTile;
  globalObj.__GPU_TILE_SELECTED_ID__ = Number.isInteger(mode.selectedTileId)
    ? mode.selectedTileId
    : -1;
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
    `showOverlay=${mode.showOverlay}`,
    `drawSelectedOnly=${mode.drawSelectedOnly}`,
    `useMaxTile=${mode.useMaxTile}`,
    `selectedTileId=${mode.selectedTileId}`,
    `focusTileId=${focusTileId}`
  ].join('  ');
}
