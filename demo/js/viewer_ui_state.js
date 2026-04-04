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

export function syncQualityOverrideUiState(ui) {
  const playbackEnabled = !!ui.usePlaybackOverrideCheck.checked;
  ui.playbackStrideInput.disabled = !playbackEnabled;
  ui.playbackMaxVisibleInput.disabled = !playbackEnabled;
  ui.playbackRenderScaleInput.disabled = !playbackEnabled;

  ui.playbackStrideNote.textContent = playbackEnabled
    ? 'coarser draw sampling while playing'
    : 'used only when playback override is on';
  ui.playbackMaxVisibleNote.textContent = playbackEnabled
    ? 'limit visible splats while playing'
    : 'used only when playback override is on';
  ui.playbackRenderScaleNote.textContent = playbackEnabled
    ? 'lower internal resolution while playing'
    : 'used only when playback override is on';
  ui.usePlaybackOverrideNote.textContent = playbackEnabled
    ? 'GUI-tunable playback degradation'
    : 'playback override is off';

  const interactionEnabled = !!ui.useInteractionOverrideCheck.checked;
  ui.interactionStrideInput.disabled = !interactionEnabled;
  ui.interactionMaxVisibleInput.disabled = !interactionEnabled;
  ui.interactionRenderScaleInput.disabled = !interactionEnabled;

  ui.interactionStrideNote.textContent = interactionEnabled
    ? 'coarser draw sampling while dragging'
    : 'used only when interaction override is on';
  ui.interactionMaxVisibleNote.textContent = interactionEnabled
    ? 'limit visible splats while dragging'
    : 'used only when interaction override is on';
  ui.interactionRenderScaleNote.textContent = interactionEnabled
    ? 'lower internal resolution while dragging'
    : 'used only when interaction override is on';
  ui.useInteractionOverrideNote.textContent = interactionEnabled
    ? 'GUI-tunable drag degradation'
    : 'interaction override is off';
}

export function syncPackedPathUiState(ui) {
  const packedEnabled = !!ui.usePackedVisiblePathCheck.checked;

  ui.usePackedVisiblePathNote.textContent = packedEnabled
    ? 'enable packed visible generation and packed direct draw tracking'
    : 'packed visible path is off';

  if (ui.drawPathSelect) {
    const requested = ui.drawPathSelect.value;
    const packedSelectable = packedEnabled;
    const gpuScreenSelectable = false;

    for (const opt of ui.drawPathSelect.options) {
      if (opt.value === 'packed') {
        opt.disabled = !packedSelectable;
      } else if (opt.value === 'gpu-screen') {
        opt.disabled = !gpuScreenSelectable;
      }
    }

    if (!packedEnabled && requested === 'packed') {
      ui.drawPathSelect.value = 'legacy';
    }
    if (requested === 'gpu-screen') {
      ui.drawPathSelect.value = packedEnabled ? 'packed' : 'legacy';
    }

    ui.drawPathSelectNote.textContent = packedEnabled
      ? 'actual draw path request'
      : 'packed disabled, legacy only';
  }
}

export function syncDebugLogUiState(ui) {
  if (!ui.debugLogArea) return;

  if (ui.debugLogNote && !ui.debugLogArea.value) {
    ui.debugLogNote.textContent = 'latest debug text';
  }
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

  ui.usePlaybackOverrideCheck.checked = true;
  ui.playbackStrideInput.value = '32';
  ui.playbackMaxVisibleInput.value = '30000';
  ui.playbackRenderScaleInput.value = '0.50';

  ui.useInteractionOverrideCheck.checked = true;
  ui.interactionStrideInput.value = '64';
  ui.interactionMaxVisibleInput.value = '10000';
  ui.interactionRenderScaleInput.value = '0.50';

  ui.usePackedVisiblePathCheck.checked = true;
  if (ui.drawPathSelect) {
    ui.drawPathSelect.value = 'packed';
  }

  if (ui.debugLogArea) {
    ui.debugLogArea.value = '';
  }
  if (ui.debugLogNote) {
    ui.debugLogNote.textContent = 'latest debug text';
  }
}

export function syncAllViewerUiState(ui, win = window) {
  syncTileDebugGlobalsFromUI(ui, win);
  syncTemporalIndexUiState(ui);
  syncTemporalBucketUiState(ui);
  syncQualityOverrideUiState(ui);
  syncPackedPathUiState(ui);
  syncDebugLogUiState(ui);
}
