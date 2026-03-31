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
    {
      id: 'tileDebugRow1',
      html: '<label>show tile debug</label><input id="showTileDebug" type="checkbox"><span>heatmap overlay</span>'
    },
    {
      id: 'tileDebugRow2',
      html: '<label>draw selected tile only</label><input id="drawSelectedTileOnly" type="checkbox"><span>single/multi-tile draw</span>'
    },
    {
      id: 'tileDebugRow3',
      html: '<label>use max tile</label><input id="useMaxTile" type="checkbox"><span>densest focus tile</span>'
    },
    {
      id: 'tileDebugRow4',
      html: '<label>tile id</label><input id="selectedTileId" type="number" min="-1" step="1" value="-1" style="width:120px;"><span id="selectedTileIdNote">manual tile id</span>'
    },
    {
      id: 'tileDebugRow5',
      html: '<label>tile radius</label><input id="tileRadius" type="number" min="0" step="1" value="0" style="width:120px;"><span id="tileRadiusNote">0=single, 1=3x3, 2=5x5</span>'
    }
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
    {
      id: 'temporalIndexRow1',
      html: '<label>use temporal index</label><input id="useTemporalIndex" type="checkbox"><span>candidate narrowing</span>'
    },
    {
      id: 'temporalIndexRow2',
      html: '<label>use index cache</label><input id="useTemporalIndexCache" type="checkbox"><span>sorted/window cache</span>'
    },
    {
      id: 'temporalIndexRow3',
      html: '<label>window mode</label><select id="temporalWindowMode" style="width:120px;"><option value="max">max</option><option value="median">median</option><option value="mean">mean</option><option value="p90">p90</option><option value="fixed">fixed</option></select><span id="temporalWindowModeNote">temporal window policy</span>'
    },
    {
      id: 'temporalIndexRow4',
      html: '<label>fixed window</label><input id="fixedWindowRadius" type="number" min="0" step="0.01" value="0.50" style="width:120px;"><span id="fixedWindowRadiusNote">used when window mode=fixed</span>'
    }
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
    {
      id: 'temporalBucketRow1',
      html: '<label>use temporal bucket</label><input id="useTemporalBucket" type="checkbox"><span>bucket candidate narrowing</span>'
    },
    {
      id: 'temporalBucketRow2',
      html: '<label>use bucket cache</label><input id="useTemporalBucketCache" type="checkbox"><span>bucket cache</span>'
    },
    {
      id: 'temporalBucketRow3',
      html: '<label>bucket width</label><input id="temporalBucketWidth" type="number" min="0.001" step="0.01" value="0.10" style="width:120px;"><span id="temporalBucketWidthNote">time width per bucket</span>'
    },
    {
      id: 'temporalBucketRow4',
      html: '<label>bucket radius</label><input id="temporalBucketRadius" type="number" min="0" step="1" value="0" style="width:120px;"><span id="temporalBucketRadiusNote">neighbor bucket count</span>'
    }
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
