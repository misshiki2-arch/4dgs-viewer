export function syncTileDebugGlobalsFromUI(ui, win = window) {
  win.__GPU_TILE_DEBUG_OVERLAY__ = !!ui.showTileDebugCheck.checked;
  win.__GPU_TILE_DRAW_SELECTED_ONLY__ = !!ui.drawSelectedTileOnlyCheck.checked;
  win.__GPU_TILE_USE_MAX_TILE__ = !!ui.useMaxTileCheck.checked;

  const tileId = Number(ui.selectedTileIdInput.value);
  win.__GPU_TILE_SELECTED_ID__ = Number.isInteger(tileId) ? tileId : -1;

  const tileRadius = Number(ui.tileRadiusInput.value);
  win.__GPU_TILE_RADIUS__ = Number.isInteger(tileRadius) && tileRadius >= 0 ? tileRadius : 0;

  const manualEnabled = ui.drawSelectedTileOnlyCheck.checked && !ui.useMaxTileCheck.checked;
  ui.selectedTileIdInput.disabled = !manualEnabled;
  ui.selectedTileIdNote.textContent = manualEnabled ? 'manual tile id' : 'used only when max tile is off';

  const radiusEnabled = ui.drawSelectedTileOnlyCheck.checked;
  ui.tileRadiusInput.disabled = !radiusEnabled;
  ui.tileRadiusNote.textContent = radiusEnabled ? '0=single, 1=3x3, 2=5x5' : 'used only when single/multi-tile draw is on';
}

export function syncTemporalIndexUiState(ui) {
  const enabled = !!ui.useTemporalIndexCheck.checked;
  const fixedMode = ui.temporalWindowModeSelect.value === 'fixed';

  ui.useTemporalIndexCacheCheck.disabled = !enabled;
  ui.temporalWindowModeSelect.disabled = !enabled;
  ui.fixedWindowRadiusInput.disabled = !(enabled && fixedMode);

  ui.temporalWindowModeNote.textContent = enabled
    ? 'temporal window policy'
    : 'used only when temporal index is on';

  ui.fixedWindowRadiusNote.textContent = (enabled && fixedMode)
    ? 'used when window mode=fixed'
    : 'used only when temporal index is on and mode=fixed';
}

export function syncTemporalBucketUiState(ui) {
  const enabled = !!ui.useTemporalBucketCheck.checked;

  ui.useTemporalBucketCacheCheck.disabled = !enabled;
  ui.temporalBucketWidthInput.disabled = !enabled;
  ui.temporalBucketRadiusInput.disabled = !enabled;

  ui.temporalBucketWidthNote.textContent = enabled
    ? 'time width per bucket'
    : 'used only when temporal bucket is on';

  ui.temporalBucketRadiusNote.textContent = enabled
    ? 'neighbor bucket count'
    : 'used only when temporal bucket is on';
}

export function initializeViewerUiDefaults(ui) {
  ui.showTileDebugCheck.checked = false;
  ui.drawSelectedTileOnlyCheck.checked = false;
  ui.useMaxTileCheck.checked = true;
  ui.selectedTileIdInput.value = '-1';
  ui.tileRadiusInput.value = '0';

  ui.useTemporalIndexCheck.checked = true;
  ui.useTemporalIndexCacheCheck.checked = true;
  ui.temporalWindowModeSelect.value = 'median';
  ui.fixedWindowRadiusInput.value = '0.50';

  ui.useTemporalBucketCheck.checked = false;
  ui.useTemporalBucketCacheCheck.checked = true;
  ui.temporalBucketWidthInput.value = '0.10';
  ui.temporalBucketRadiusInput.value = '0';
}

export function syncAllViewerUiState(ui, win = window) {
  syncTileDebugGlobalsFromUI(ui, win);
  syncTemporalIndexUiState(ui);
  syncTemporalBucketUiState(ui);
}
