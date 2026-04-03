export function applyInfoWrapStyle(infoEl) {
  if (!infoEl) return;
  infoEl.style.whiteSpace = 'pre-wrap';
  infoEl.style.overflowWrap = 'anywhere';
  infoEl.style.wordBreak = 'break-word';
  infoEl.style.maxWidth = '100%';
}

export function applyPanelResizeStyle(infoEl) {
  if (!infoEl || !infoEl.parentElement) return;
  const panel = infoEl.parentElement;
  panel.style.resize = 'horizontal';
  panel.style.overflow = 'auto';
  panel.style.minWidth = '280px';
  panel.style.maxWidth = '70vw';
  panel.style.width = panel.style.width || '540px';
  panel.style.boxSizing = 'border-box';
}

export function ensureTileDebugControls(ui) {
  const parent = ui.info.parentElement;
  const rows = [
    { id: 'tileDebugRow1', html: '<label>show tile debug</label><input id="showTileDebug" type="checkbox"><span>heatmap overlay</span>' },
    { id: 'tileDebugRow2', html: '<label>draw selected tile only</label><input id="drawSelectedTileOnly" type="checkbox"><span>single/multi-tile draw</span>' },
    { id: 'tileDebugRow3', html: '<label>use max tile</label><input id="useMaxTile" type="checkbox"><span>densest focus tile</span>' },
    { id: 'tileDebugRow4', html: '<label>tile id</label><input id="selectedTileId" type="number" min="-1" step="1" value="-1" style="width:120px;"><span id="selectedTileIdNote">manual tile id</span>' },
    { id: 'tileDebugRow5', html: '<label>tile radius</label><input id="tileRadius" type="number" min="0" step="1" value="0" style="width:120px;"><span id="tileRadiusNote">0=single, 1=3x3, 2=5x5</span>' }
  ];
  for (const rowDef of rows) {
    let row = document.getElementById(rowDef.id);
    if (!row) {
      row = document.createElement('div');
      row.className = 'row';
      row.id = rowDef.id;
      row.innerHTML = rowDef.html;
      parent.insertBefore(row, ui.info);
    }
  }
  ui.showTileDebugCheck = document.getElementById('showTileDebug');
  ui.drawSelectedTileOnlyCheck = document.getElementById('drawSelectedTileOnly');
  ui.useMaxTileCheck = document.getElementById('useMaxTile');
  ui.selectedTileIdInput = document.getElementById('selectedTileId');
  ui.selectedTileIdNote = document.getElementById('selectedTileIdNote');
  ui.tileRadiusInput = document.getElementById('tileRadius');
  ui.tileRadiusNote = document.getElementById('tileRadiusNote');
}

export function ensureTemporalIndexControls(ui) {
  const parent = ui.info.parentElement;
  const rows = [
    { id: 'temporalIndexRow1', html: '<label>use temporal index</label><input id="useTemporalIndex" type="checkbox"><span>candidate narrowing</span>' },
    { id: 'temporalIndexRow2', html: '<label>use index cache</label><input id="useTemporalIndexCache" type="checkbox"><span>sorted/window cache</span>' },
    { id: 'temporalIndexRow3', html: '<label>window mode</label><select id="temporalWindowMode" style="width:120px;"><option value="max">max</option><option value="median">median</option><option value="mean">mean</option><option value="p90">p90</option><option value="fixed">fixed</option></select><span id="temporalWindowModeNote">temporal window policy</span>' },
    { id: 'temporalIndexRow4', html: '<label>fixed window</label><input id="fixedWindowRadius" type="number" min="0" step="0.01" value="0.50" style="width:120px;"><span id="fixedWindowRadiusNote">used when window mode=fixed</span>' }
  ];
  for (const rowDef of rows) {
    let row = document.getElementById(rowDef.id);
    if (!row) {
      row = document.createElement('div');
      row.className = 'row';
      row.id = rowDef.id;
      row.innerHTML = rowDef.html;
      parent.insertBefore(row, ui.info);
    }
  }
  ui.useTemporalIndexCheck = document.getElementById('useTemporalIndex');
  ui.useTemporalIndexCacheCheck = document.getElementById('useTemporalIndexCache');
  ui.temporalWindowModeSelect = document.getElementById('temporalWindowMode');
  ui.temporalWindowModeNote = document.getElementById('temporalWindowModeNote');
  ui.fixedWindowRadiusInput = document.getElementById('fixedWindowRadius');
  ui.fixedWindowRadiusNote = document.getElementById('fixedWindowRadiusNote');
}

export function ensureTemporalBucketControls(ui) {
  const parent = ui.info.parentElement;
  const rows = [
    { id: 'temporalBucketRow1', html: '<label>use temporal bucket</label><input id="useTemporalBucket" type="checkbox"><span>bucket candidate narrowing</span>' },
    { id: 'temporalBucketRow2', html: '<label>use bucket cache</label><input id="useTemporalBucketCache" type="checkbox"><span>bucket cache</span>' },
    { id: 'temporalBucketRow3', html: '<label>bucket width</label><input id="temporalBucketWidth" type="number" min="0.001" step="0.01" value="0.10" style="width:120px;"><span id="temporalBucketWidthNote">time width per bucket</span>' },
    { id: 'temporalBucketRow4', html: '<label>bucket radius</label><input id="temporalBucketRadius" type="number" min="0" step="1" value="0" style="width:120px;"><span id="temporalBucketRadiusNote">neighbor bucket count</span>' }
  ];
  for (const rowDef of rows) {
    let row = document.getElementById(rowDef.id);
    if (!row) {
      row = document.createElement('div');
      row.className = 'row';
      row.id = rowDef.id;
      row.innerHTML = rowDef.html;
      parent.insertBefore(row, ui.info);
    }
  }
  ui.useTemporalBucketCheck = document.getElementById('useTemporalBucket');
  ui.useTemporalBucketCacheCheck = document.getElementById('useTemporalBucketCache');
  ui.temporalBucketWidthInput = document.getElementById('temporalBucketWidth');
  ui.temporalBucketWidthNote = document.getElementById('temporalBucketWidthNote');
  ui.temporalBucketRadiusInput = document.getElementById('temporalBucketRadius');
  ui.temporalBucketRadiusNote = document.getElementById('temporalBucketRadiusNote');
}

export function ensureQualityOverrideControls(ui) {
  const parent = ui.info.parentElement;
  const rows = [
    { id: 'qualityOverrideRow1', html: '<label>use playback override</label><input id="usePlaybackOverride" type="checkbox"><span id="usePlaybackOverrideNote">GUI-tunable playback degradation</span>' },
    { id: 'qualityOverrideRow2', html: '<label>playback stride</label><input id="playbackStride" type="number" min="1" step="1" value="32" style="width:120px;"><span id="playbackStrideNote">coarser draw sampling while playing</span>' },
    { id: 'qualityOverrideRow3', html: '<label>playback max visible</label><input id="playbackMaxVisible" type="number" min="1" step="1000" value="30000" style="width:120px;"><span id="playbackMaxVisibleNote">limit visible splats while playing</span>' },
    { id: 'qualityOverrideRow4', html: '<label>playback render scale</label><input id="playbackRenderScale" type="number" min="0.05" max="1.00" step="0.05" value="0.50" style="width:120px;"><span id="playbackRenderScaleNote">lower internal resolution while playing</span>' },
    { id: 'qualityOverrideRow5', html: '<label>use interaction override</label><input id="useInteractionOverride" type="checkbox"><span id="useInteractionOverrideNote">GUI-tunable drag degradation</span>' },
    { id: 'qualityOverrideRow6', html: '<label>interaction stride</label><input id="interactionStride" type="number" min="1" step="1" value="64" style="width:120px;"><span id="interactionStrideNote">coarser draw sampling while dragging</span>' },
    { id: 'qualityOverrideRow7', html: '<label>interaction max visible</label><input id="interactionMaxVisible" type="number" min="1" step="1000" value="10000" style="width:120px;"><span id="interactionMaxVisibleNote">limit visible splats while dragging</span>' },
    { id: 'qualityOverrideRow8', html: '<label>interaction render scale</label><input id="interactionRenderScale" type="number" min="0.05" max="1.00" step="0.05" value="0.50" style="width:120px;"><span id="interactionRenderScaleNote">lower internal resolution while dragging</span>' }
  ];
  for (const rowDef of rows) {
    let row = document.getElementById(rowDef.id);
    if (!row) {
      row = document.createElement('div');
      row.className = 'row';
      row.id = rowDef.id;
      row.innerHTML = rowDef.html;
      parent.insertBefore(row, ui.info);
    }
  }
  ui.usePlaybackOverrideCheck = document.getElementById('usePlaybackOverride');
  ui.usePlaybackOverrideNote = document.getElementById('usePlaybackOverrideNote');
  ui.playbackStrideInput = document.getElementById('playbackStride');
  ui.playbackStrideNote = document.getElementById('playbackStrideNote');
  ui.playbackMaxVisibleInput = document.getElementById('playbackMaxVisible');
  ui.playbackMaxVisibleNote = document.getElementById('playbackMaxVisibleNote');
  ui.playbackRenderScaleInput = document.getElementById('playbackRenderScale');
  ui.playbackRenderScaleNote = document.getElementById('playbackRenderScaleNote');
  ui.useInteractionOverrideCheck = document.getElementById('useInteractionOverride');
  ui.useInteractionOverrideNote = document.getElementById('useInteractionOverrideNote');
  ui.interactionStrideInput = document.getElementById('interactionStride');
  ui.interactionStrideNote = document.getElementById('interactionStrideNote');
  ui.interactionMaxVisibleInput = document.getElementById('interactionMaxVisible');
  ui.interactionMaxVisibleNote = document.getElementById('interactionMaxVisibleNote');
  ui.interactionRenderScaleInput = document.getElementById('interactionRenderScale');
  ui.interactionRenderScaleNote = document.getElementById('interactionRenderScaleNote');
}

export function ensurePackedPathControls(ui) {
  const parent = ui.info.parentElement;
  const rows = [
    { id: 'packedPathRow1', html: '<label>use packed visible path</label><input id="usePackedVisiblePath" type="checkbox"><span id="usePackedVisiblePathNote">enable packed visible generation and packed upload tracking</span>' },
    { id: 'packedPathRow2', html: '<label>draw path</label><select id="drawPathSelect" style="width:140px;"><option value="legacy">legacy</option><option value="packed">packed</option><option value="gpu-screen">gpu-screen</option></select><span id="drawPathSelectNote">actual draw path request</span>' }
  ];
  for (const rowDef of rows) {
    let row = document.getElementById(rowDef.id);
    if (!row) {
      row = document.createElement('div');
      row.className = 'row';
      row.id = rowDef.id;
      row.innerHTML = rowDef.html;
      parent.insertBefore(row, ui.info);
    }
  }
  ui.usePackedVisiblePathCheck = document.getElementById('usePackedVisiblePath');
  ui.usePackedVisiblePathNote = document.getElementById('usePackedVisiblePathNote');
  ui.drawPathSelect = document.getElementById('drawPathSelect');
  ui.drawPathSelectNote = document.getElementById('drawPathSelectNote');
}

export function ensureDebugLogControls(ui) {
  const parent = ui.info.parentElement;
  const rows = [
    { id: 'debugLogRow1', html: '<label>debug log</label><button id="debugLogBtn" type="button">ログ出力</button><button id="debugLogCopyBtn" type="button">コピー</button><span id="debugLogNote">latest debug text</span>' },
    { id: 'debugLogRow2', html: '<label>debug text</label><textarea id="debugLogArea" rows="10" style="width:100%; box-sizing:border-box; resize:vertical;" spellcheck="false" placeholder="ここに最新のデバッグ情報を出力します。"></textarea>' }
  ];
  for (const rowDef of rows) {
    let row = document.getElementById(rowDef.id);
    if (!row) {
      row = document.createElement('div');
      row.className = 'row';
      row.id = rowDef.id;
      row.innerHTML = rowDef.html;
      parent.insertBefore(row, ui.info);
    }
  }
  ui.debugLogBtn = document.getElementById('debugLogBtn');
  ui.debugLogCopyBtn = document.getElementById('debugLogCopyBtn');
  ui.debugLogNote = document.getElementById('debugLogNote');
  ui.debugLogArea = document.getElementById('debugLogArea');
}

export function setDebugLogText(ui, text) {
  if (!ui || !ui.debugLogArea) return;
  ui.debugLogArea.value = text ?? '';
  if (ui.debugLogNote) {
    ui.debugLogNote.textContent = text ? 'latest debug text' : 'no debug text';
  }
}

export async function copyDebugLogText(ui) {
  const text = ui?.debugLogArea?.value ?? '';
  if (!text) return false;
  try {
    await navigator.clipboard.writeText(text);
    if (ui?.debugLogNote) ui.debugLogNote.textContent = 'copied';
    return true;
  } catch (err) {
    console.warn(err);
    if (ui?.debugLogNote) ui.debugLogNote.textContent = 'copy failed';
    return false;
  }
}
