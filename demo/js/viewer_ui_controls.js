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
  panel.style.minWidth = '320px';
  panel.style.maxWidth = '72vw';
  panel.style.width = panel.style.width || '560px';
  panel.style.boxSizing = 'border-box';
}

function ensureRow(ui, rowId) {
  let row = document.getElementById(rowId);
  if (row) return row;

  row = document.createElement('div');
  row.className = 'row';
  row.id = rowId;
  row.style.display = 'flex';
  row.style.alignItems = 'center';
  row.style.flexWrap = 'wrap';
  row.style.gap = '8px';
  row.style.margin = '4px 0';
  row.style.lineHeight = '1.4';
  ui.info.parentElement.insertBefore(row, ui.info);
  return row;
}

function setRowContents(row, nodes) {
  row.replaceChildren(...nodes);
}

function createLabel(text, minWidth = null) {
  const span = document.createElement('span');
  span.textContent = text;
  if (minWidth) span.style.minWidth = minWidth;
  return span;
}

function createCheckbox(id, checked = false) {
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.id = id;
  input.checked = checked;
  return input;
}

function getOrCreateCheckbox(id, checked = false) {
  const existing = document.getElementById(id);
  if (existing) return existing;
  return createCheckbox(id, checked);
}

function createNumberInput(id, value, min = null, max = null, step = null, width = '92px') {
  const input = document.createElement('input');
  input.type = 'number';
  input.id = id;
  input.value = String(value);
  if (min !== null) input.min = String(min);
  if (max !== null) input.max = String(max);
  if (step !== null) input.step = String(step);
  input.style.width = width;
  input.style.boxSizing = 'border-box';
  return input;
}

function getOrCreateNumberInput(id, value, min = null, max = null, step = null, width = '92px') {
  const existing = document.getElementById(id);
  if (existing) return existing;
  return createNumberInput(id, value, min, max, step, width);
}

function createSelect(id, options, value, width = null) {
  const select = document.createElement('select');
  select.id = id;
  if (width) select.style.width = width;
  select.style.boxSizing = 'border-box';
  for (const opt of options) {
    const option = document.createElement('option');
    option.value = opt.value;
    option.textContent = opt.label;
    if (opt.value === value) option.selected = true;
    select.appendChild(option);
  }
  return select;
}

function createNote(id, text = '') {
  const span = document.createElement('span');
  span.id = id;
  span.style.fontSize = '0.9em';
  span.style.opacity = '0.82';
  span.style.marginLeft = '4px';
  span.textContent = text;
  return span;
}

function getOrCreateNote(id, text = '') {
  const existing = document.getElementById(id);
  if (existing) return existing;
  return createNote(id, text);
}

function appendWithSpaces(...nodes) {
  const out = [];
  nodes.forEach((node, i) => {
    if (i > 0) out.push(document.createTextNode(' '));
    out.push(node);
  });
  return out;
}

export function ensureTileDebugControls(ui) {
  const row1 = ensureRow(ui, 'tileDebugRow1');
  setRowContents(row1, appendWithSpaces(
    getOrCreateCheckbox('showTileDebug', false),
    createLabel('show tile debug heatmap overlay')
  ));

  const row2 = ensureRow(ui, 'tileDebugRow2');
  setRowContents(row2, appendWithSpaces(
    getOrCreateCheckbox('drawSelectedTileOnly', false),
    createLabel('draw selected tile only')
  ));

  const row3 = ensureRow(ui, 'tileDebugRow3');
  setRowContents(row3, appendWithSpaces(
    getOrCreateCheckbox('useMaxTile', true),
    createLabel('use max tile'),
    getOrCreateNote('useMaxTileNote', 'densest focus tile')
  ));

  const row4 = ensureRow(ui, 'tileDebugRow4');
  setRowContents(row4, appendWithSpaces(
    createLabel('tile id', '56px'),
    getOrCreateNumberInput('selectedTileId', -1, -1, null, 1),
    getOrCreateNote('selectedTileIdNote', 'manual tile id')
  ));

  const row5 = ensureRow(ui, 'tileDebugRow5');
  setRowContents(row5, appendWithSpaces(
    createLabel('tile radius', '56px'),
    getOrCreateNumberInput('tileRadius', 0, 0, 8, 1),
    getOrCreateNote('tileRadiusNote', '0=single, 1=3x3, 2=5x5')
  ));

  ui.showTileDebugCheck = document.getElementById('showTileDebug');
  ui.drawSelectedTileOnlyCheck = document.getElementById('drawSelectedTileOnly');
  ui.useMaxTileCheck = document.getElementById('useMaxTile');
  ui.useMaxTileNote = document.getElementById('useMaxTileNote');
  ui.selectedTileIdInput = document.getElementById('selectedTileId');
  ui.selectedTileIdNote = document.getElementById('selectedTileIdNote');
  ui.tileRadiusInput = document.getElementById('tileRadius');
  ui.tileRadiusNote = document.getElementById('tileRadiusNote');
}

export function ensureTemporalIndexControls(ui) {
  const row1 = ensureRow(ui, 'temporalIndexRow1');
  setRowContents(row1, appendWithSpaces(
    createCheckbox('useTemporalIndex', false),
    createLabel('use temporal index candidate narrowing')
  ));

  const row2 = ensureRow(ui, 'temporalIndexRow2');
  setRowContents(row2, appendWithSpaces(
    createCheckbox('useTemporalIndexCache', true),
    createLabel('use index cache')
  ));

  const row3 = ensureRow(ui, 'temporalIndexRow3');
  setRowContents(row3, appendWithSpaces(
    createLabel('window mode', '92px'),
    createSelect(
      'temporalWindowMode',
      [
        { value: 'max', label: 'max' },
        { value: 'median', label: 'median' },
        { value: 'mean', label: 'mean' },
        { value: 'p90', label: 'p90' },
        { value: 'fixed', label: 'fixed' }
      ],
      'max',
      '120px'
    ),
    createNote('temporalWindowModeNote', 'temporal window policy')
  ));

  const row4 = ensureRow(ui, 'temporalIndexRow4');
  setRowContents(row4, appendWithSpaces(
    createLabel('fixed window', '92px'),
    createNumberInput('fixedWindowRadius', 0, 0, 9999, 1),
    createNote('fixedWindowRadiusNote', 'used when window mode=fixed')
  ));

  ui.useTemporalIndexCheck = document.getElementById('useTemporalIndex');
  ui.useTemporalIndexCacheCheck = document.getElementById('useTemporalIndexCache');
  ui.temporalWindowModeSelect = document.getElementById('temporalWindowMode');
  ui.temporalWindowModeNote = document.getElementById('temporalWindowModeNote');
  ui.fixedWindowRadiusInput = document.getElementById('fixedWindowRadius');
  ui.fixedWindowRadiusNote = document.getElementById('fixedWindowRadiusNote');
}

export function ensureTemporalBucketControls(ui) {
  const row1 = ensureRow(ui, 'temporalBucketRow1');
  setRowContents(row1, appendWithSpaces(
    createCheckbox('useTemporalBucket', false),
    createLabel('use temporal bucket candidate narrowing')
  ));

  const row2 = ensureRow(ui, 'temporalBucketRow2');
  setRowContents(row2, appendWithSpaces(
    createCheckbox('useTemporalBucketCache', true),
    createLabel('use bucket cache')
  ));

  const row3 = ensureRow(ui, 'temporalBucketRow3');
  setRowContents(row3, appendWithSpaces(
    createLabel('bucket width', '92px'),
    createNumberInput('temporalBucketWidth', 0.05, 0, null, 0.01),
    createNote('temporalBucketWidthNote', 'time width per bucket')
  ));

  const row4 = ensureRow(ui, 'temporalBucketRow4');
  setRowContents(row4, appendWithSpaces(
    createLabel('bucket radius', '92px'),
    createNumberInput('temporalBucketRadius', 1, 0, 9999, 1),
    createNote('temporalBucketRadiusNote', 'neighbor bucket count')
  ));

  ui.useTemporalBucketCheck = document.getElementById('useTemporalBucket');
  ui.useTemporalBucketCacheCheck = document.getElementById('useTemporalBucketCache');
  ui.temporalBucketWidthInput = document.getElementById('temporalBucketWidth');
  ui.temporalBucketWidthNote = document.getElementById('temporalBucketWidthNote');
  ui.temporalBucketRadiusInput = document.getElementById('temporalBucketRadius');
  ui.temporalBucketRadiusNote = document.getElementById('temporalBucketRadiusNote');
}

export function ensureQualityOverrideControls(ui) {
  const row1 = ensureRow(ui, 'qualityOverrideRow1');
  setRowContents(row1, appendWithSpaces(
    createCheckbox('usePlaybackOverride', false),
    createLabel('use playback override')
  ));

  const row2 = ensureRow(ui, 'qualityOverrideRow2');
  setRowContents(row2, appendWithSpaces(
    createLabel('playback stride', '140px'),
    createNumberInput('playbackStride', 1, 1, 9999, 1),
    createNote('playbackStrideNote', 'coarser draw sampling while playing')
  ));

  const row3 = ensureRow(ui, 'qualityOverrideRow3');
  setRowContents(row3, appendWithSpaces(
    createLabel('playback max visible', '140px'),
    createNumberInput('playbackMaxVisible', 0, 0, null, 1),
    createNote('playbackMaxVisibleNote', 'limit visible splats while playing')
  ));

  const row4 = ensureRow(ui, 'qualityOverrideRow4');
  setRowContents(row4, appendWithSpaces(
    createLabel('playback render scale', '140px'),
    createNumberInput('playbackRenderScale', 1.0, 0.05, 1.0, 0.05),
    createNote('playbackRenderScaleNote', 'lower internal resolution while playing')
  ));

  const row5 = ensureRow(ui, 'qualityOverrideRow5');
  setRowContents(row5, appendWithSpaces(
    createCheckbox('useInteractionOverride', false),
    createLabel('use interaction override')
  ));

  const row6 = ensureRow(ui, 'qualityOverrideRow6');
  setRowContents(row6, appendWithSpaces(
    createLabel('interaction stride', '140px'),
    createNumberInput('interactionStride', 1, 1, 9999, 1),
    createNote('interactionStrideNote', 'coarser draw sampling while dragging')
  ));

  const row7 = ensureRow(ui, 'qualityOverrideRow7');
  setRowContents(row7, appendWithSpaces(
    createLabel('interaction max visible', '140px'),
    createNumberInput('interactionMaxVisible', 0, 0, null, 1),
    createNote('interactionMaxVisibleNote', 'limit visible splats while dragging')
  ));

  const row8 = ensureRow(ui, 'qualityOverrideRow8');
  setRowContents(row8, appendWithSpaces(
    createLabel('interaction render scale', '140px'),
    createNumberInput('interactionRenderScale', 1.0, 0.05, 1.0, 0.05),
    createNote('interactionRenderScaleNote', 'lower internal resolution while dragging')
  ));

  ui.usePlaybackOverrideCheck = document.getElementById('usePlaybackOverride');
  ui.playbackStrideInput = document.getElementById('playbackStride');
  ui.playbackStrideNote = document.getElementById('playbackStrideNote');
  ui.playbackMaxVisibleInput = document.getElementById('playbackMaxVisible');
  ui.playbackMaxVisibleNote = document.getElementById('playbackMaxVisibleNote');
  ui.playbackRenderScaleInput = document.getElementById('playbackRenderScale');
  ui.playbackRenderScaleNote = document.getElementById('playbackRenderScaleNote');
  ui.useInteractionOverrideCheck = document.getElementById('useInteractionOverride');
  ui.interactionStrideInput = document.getElementById('interactionStride');
  ui.interactionStrideNote = document.getElementById('interactionStrideNote');
  ui.interactionMaxVisibleInput = document.getElementById('interactionMaxVisible');
  ui.interactionMaxVisibleNote = document.getElementById('interactionMaxVisibleNote');
  ui.interactionRenderScaleInput = document.getElementById('interactionRenderScale');
  ui.interactionRenderScaleNote = document.getElementById('interactionRenderScaleNote');
}

export function ensurePackedPathControls(ui) {
  const row1 = ensureRow(ui, 'packedPathRow1');
  setRowContents(row1, appendWithSpaces(
    createCheckbox('usePackedVisiblePath', true),
    createLabel('use packed visible path'),
    createNote('usePackedVisiblePathNote', 'formal full-frame packed reference path')
  ));

  const row2 = ensureRow(ui, 'packedPathRow2');
  setRowContents(row2, appendWithSpaces(
    createLabel('draw path', '92px'),
    createSelect(
      'drawPathSelect',
      [
        { value: 'packed', label: 'packed (formal reference)' },
        { value: 'gpu-screen', label: 'gpu-screen (experimental compare)' },
        { value: 'legacy', label: 'legacy (fallback)' }
      ],
      'packed',
      '230px'
    ),
    createNote(
      'drawPathSelectNote',
      'full-frame only; gpu-screen debug distinguishes actual, source, and reference'
    )
  ));

  ui.usePackedVisiblePathCheck = document.getElementById('usePackedVisiblePath');
  ui.usePackedVisiblePathNote = document.getElementById('usePackedVisiblePathNote');
  ui.drawPathSelect = document.getElementById('drawPathSelect');
  ui.drawPathSelectNote = document.getElementById('drawPathSelectNote');
}

export function ensureDebugLogControls(ui) {
  const row1 = ensureRow(ui, 'debugLogRow1');

  const logBtn = document.createElement('button');
  logBtn.id = 'debugLogBtn';
  logBtn.type = 'button';
  logBtn.textContent = 'debug log';

  const copyBtn = document.createElement('button');
  copyBtn.id = 'debugLogCopyBtn';
  copyBtn.type = 'button';
  copyBtn.textContent = 'copy';

  setRowContents(row1, appendWithSpaces(logBtn, copyBtn, createNote('debugLogNote', 'latest debug text')));

  const row2 = ensureRow(ui, 'debugLogRow2');
  const area = document.createElement('textarea');
  area.id = 'debugLogArea';
  area.rows = 10;
  area.style.width = '100%';
  area.style.boxSizing = 'border-box';
  area.style.fontFamily = 'monospace';
  area.style.fontSize = '12px';
  area.readOnly = true;
  setRowContents(row2, [area]);

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
