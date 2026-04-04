// Step24:
// UI 状態の読み書きをここに集約する。
// packed を正式経路、legacy を fallback、gpu-screen を future とする
// 文脈を UI state 側でも固定する。
// このファイルは localStorage 永続化と UI 反映の責務だけを持ち、
// renderer 側の draw path 判定ロジックは持たない。

const UI_STATE_STORAGE_KEY = 'gpuViewerUiStateStep24';

function toBool(value, fallback = false) {
  if (typeof value === 'boolean') return value;
  if (value === 'true') return true;
  if (value === 'false') return false;
  return fallback;
}

function toInt(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? (n | 0) : fallback;
}

function toFloat(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function toStringValue(value, fallback = '') {
  return typeof value === 'string' ? value : fallback;
}

function isAllowedDrawPath(value) {
  return value === 'packed' || value === 'legacy' || value === 'gpu-screen';
}

function normalizeDrawPath(value, fallback = 'packed') {
  return isAllowedDrawPath(value) ? value : fallback;
}

function readChecked(el, fallback = false) {
  return el ? !!el.checked : fallback;
}

function readValue(el, fallback = '') {
  return el ? String(el.value) : fallback;
}

function writeChecked(el, value) {
  if (el) el.checked = !!value;
}

function writeValue(el, value) {
  if (el) el.value = String(value);
}

export function createDefaultUiState() {
  return {
    // tile debug
    showTileDebug: false,
    drawSelectedTileOnly: false,
    useMaxTile: true,
    selectedTileId: -1,
    tileRadius: 0,

    // temporal index
    useTemporalIndex: false,
    useTemporalIndexCache: true,
    temporalWindowMode: 'max',
    fixedWindowRadius: 0,

    // temporal bucket
    useTemporalBucket: false,
    useTemporalBucketCache: true,
    temporalBucketWidth: 0.05,
    temporalBucketRadius: 1,

    // quality override
    usePlaybackOverride: false,
    playbackStride: 1,
    playbackMaxVisible: 0,
    playbackRenderScale: 1.0,
    useInteractionOverride: false,
    interactionStride: 1,
    interactionMaxVisible: 0,
    interactionRenderScale: 1.0,

    // Step24 packed path
    usePackedVisiblePath: true,
    drawPath: 'packed',

    // misc/debug
    bgGray: 32,
    debugLogText: ''
  };
}

export function normalizeUiState(input = {}) {
  const defaults = createDefaultUiState();

  return {
    showTileDebug: toBool(input.showTileDebug, defaults.showTileDebug),
    drawSelectedTileOnly: toBool(input.drawSelectedTileOnly, defaults.drawSelectedTileOnly),
    useMaxTile: toBool(input.useMaxTile, defaults.useMaxTile),
    selectedTileId: toInt(input.selectedTileId, defaults.selectedTileId),
    tileRadius: Math.max(0, toInt(input.tileRadius, defaults.tileRadius)),

    useTemporalIndex: toBool(input.useTemporalIndex, defaults.useTemporalIndex),
    useTemporalIndexCache: toBool(input.useTemporalIndexCache, defaults.useTemporalIndexCache),
    temporalWindowMode: toStringValue(input.temporalWindowMode, defaults.temporalWindowMode),
    fixedWindowRadius: Math.max(0, toInt(input.fixedWindowRadius, defaults.fixedWindowRadius)),

    useTemporalBucket: toBool(input.useTemporalBucket, defaults.useTemporalBucket),
    useTemporalBucketCache: toBool(input.useTemporalBucketCache, defaults.useTemporalBucketCache),
    temporalBucketWidth: Math.max(0, toFloat(input.temporalBucketWidth, defaults.temporalBucketWidth)),
    temporalBucketRadius: Math.max(0, toInt(input.temporalBucketRadius, defaults.temporalBucketRadius)),

    usePlaybackOverride: toBool(input.usePlaybackOverride, defaults.usePlaybackOverride),
    playbackStride: Math.max(1, toInt(input.playbackStride, defaults.playbackStride)),
    playbackMaxVisible: Math.max(0, toInt(input.playbackMaxVisible, defaults.playbackMaxVisible)),
    playbackRenderScale: Math.max(0.05, toFloat(input.playbackRenderScale, defaults.playbackRenderScale)),
    useInteractionOverride: toBool(input.useInteractionOverride, defaults.useInteractionOverride),
    interactionStride: Math.max(1, toInt(input.interactionStride, defaults.interactionStride)),
    interactionMaxVisible: Math.max(0, toInt(input.interactionMaxVisible, defaults.interactionMaxVisible)),
    interactionRenderScale: Math.max(0.05, toFloat(input.interactionRenderScale, defaults.interactionRenderScale)),

    // Step24:
    // packed visible path は正式経路なので default=true。
    usePackedVisiblePath: toBool(input.usePackedVisiblePath, defaults.usePackedVisiblePath),

    // Step24:
    // full-frame requested draw path の default は packed。
    // legacy は fallback / 比較用、gpu-screen は future。
    drawPath: normalizeDrawPath(input.drawPath, defaults.drawPath),

    bgGray: Math.min(255, Math.max(0, toInt(input.bgGray, defaults.bgGray))),
    debugLogText: toStringValue(input.debugLogText, defaults.debugLogText)
  };
}

export function loadUiState() {
  try {
    const raw = localStorage.getItem(UI_STATE_STORAGE_KEY);
    if (!raw) return createDefaultUiState();
    const parsed = JSON.parse(raw);
    return normalizeUiState(parsed);
  } catch (err) {
    console.warn('Failed to load UI state:', err);
    return createDefaultUiState();
  }
}

export function saveUiState(state) {
  const normalized = normalizeUiState(state);
  try {
    localStorage.setItem(UI_STATE_STORAGE_KEY, JSON.stringify(normalized));
  } catch (err) {
    console.warn('Failed to save UI state:', err);
  }
  return normalized;
}

export function readUiStateFromControls(ui) {
  return normalizeUiState({
    showTileDebug: readChecked(ui.showTileDebugCheck),
    drawSelectedTileOnly: readChecked(ui.drawSelectedTileOnlyCheck),
    useMaxTile: readChecked(ui.useMaxTileCheck),
    selectedTileId: readValue(ui.selectedTileIdInput, '-1'),
    tileRadius: readValue(ui.tileRadiusInput, '0'),

    useTemporalIndex: readChecked(ui.useTemporalIndexCheck),
    useTemporalIndexCache: readChecked(ui.useTemporalIndexCacheCheck),
    temporalWindowMode: readValue(ui.temporalWindowModeSelect, 'max'),
    fixedWindowRadius: readValue(ui.fixedWindowRadiusInput, '0'),

    useTemporalBucket: readChecked(ui.useTemporalBucketCheck),
    useTemporalBucketCache: readChecked(ui.useTemporalBucketCacheCheck),
    temporalBucketWidth: readValue(ui.temporalBucketWidthInput, '0.05'),
    temporalBucketRadius: readValue(ui.temporalBucketRadiusInput, '1'),

    usePlaybackOverride: readChecked(ui.usePlaybackOverrideCheck),
    playbackStride: readValue(ui.playbackStrideInput, '1'),
    playbackMaxVisible: readValue(ui.playbackMaxVisibleInput, '0'),
    playbackRenderScale: readValue(ui.playbackRenderScaleInput, '1.0'),
    useInteractionOverride: readChecked(ui.useInteractionOverrideCheck),
    interactionStride: readValue(ui.interactionStrideInput, '1'),
    interactionMaxVisible: readValue(ui.interactionMaxVisibleInput, '0'),
    interactionRenderScale: readValue(ui.interactionRenderScaleInput, '1.0'),

    usePackedVisiblePath: readChecked(ui.usePackedVisiblePathCheck, true),
    drawPath: readValue(ui.drawPathSelect, 'packed'),

    bgGray: readValue(ui.bgGraySlider, '32'),
    debugLogText: readValue(ui.debugLogArea, '')
  });
}

export function applyUiStateToControls(ui, state) {
  const s = normalizeUiState(state);

  writeChecked(ui.showTileDebugCheck, s.showTileDebug);
  writeChecked(ui.drawSelectedTileOnlyCheck, s.drawSelectedTileOnly);
  writeChecked(ui.useMaxTileCheck, s.useMaxTile);
  writeValue(ui.selectedTileIdInput, s.selectedTileId);
  writeValue(ui.tileRadiusInput, s.tileRadius);

  writeChecked(ui.useTemporalIndexCheck, s.useTemporalIndex);
  writeChecked(ui.useTemporalIndexCacheCheck, s.useTemporalIndexCache);
  writeValue(ui.temporalWindowModeSelect, s.temporalWindowMode);
  writeValue(ui.fixedWindowRadiusInput, s.fixedWindowRadius);

  writeChecked(ui.useTemporalBucketCheck, s.useTemporalBucket);
  writeChecked(ui.useTemporalBucketCacheCheck, s.useTemporalBucketCache);
  writeValue(ui.temporalBucketWidthInput, s.temporalBucketWidth);
  writeValue(ui.temporalBucketRadiusInput, s.temporalBucketRadius);

  writeChecked(ui.usePlaybackOverrideCheck, s.usePlaybackOverride);
  writeValue(ui.playbackStrideInput, s.playbackStride);
  writeValue(ui.playbackMaxVisibleInput, s.playbackMaxVisible);
  writeValue(ui.playbackRenderScaleInput, s.playbackRenderScale);
  writeChecked(ui.useInteractionOverrideCheck, s.useInteractionOverride);
  writeValue(ui.interactionStrideInput, s.interactionStride);
  writeValue(ui.interactionMaxVisibleInput, s.interactionMaxVisible);
  writeValue(ui.interactionRenderScaleInput, s.interactionRenderScale);

  writeChecked(ui.usePackedVisiblePathCheck, s.usePackedVisiblePath);
  writeValue(ui.drawPathSelect, s.drawPath);

  writeValue(ui.bgGraySlider, s.bgGray);
  writeValue(ui.debugLogArea, s.debugLogText);

  if (ui.usePackedVisiblePathNote) {
    ui.usePackedVisiblePathNote.textContent =
      'Step24 formal path: build packed screen-space data and packed direct draw metadata';
  }

  if (ui.drawPathSelectNote) {
    ui.drawPathSelectNote.textContent =
      'Requested draw path for full-frame rendering. Per-tile mode remains legacy-only.';
  }

  return s;
}

export function loadAndApplyUiState(ui) {
  const state = loadUiState();
  return applyUiStateToControls(ui, state);
}

export function readAndSaveUiState(ui) {
  const state = readUiStateFromControls(ui);
  return saveUiState(state);
}

export function bindUiStatePersistence(ui, options = {}) {
  const onChange = typeof options.onChange === 'function' ? options.onChange : null;

  const controls = [
    ui.showTileDebugCheck,
    ui.drawSelectedTileOnlyCheck,
    ui.useMaxTileCheck,
    ui.selectedTileIdInput,
    ui.tileRadiusInput,

    ui.useTemporalIndexCheck,
    ui.useTemporalIndexCacheCheck,
    ui.temporalWindowModeSelect,
    ui.fixedWindowRadiusInput,

    ui.useTemporalBucketCheck,
    ui.useTemporalBucketCacheCheck,
    ui.temporalBucketWidthInput,
    ui.temporalBucketRadiusInput,

    ui.usePlaybackOverrideCheck,
    ui.playbackStrideInput,
    ui.playbackMaxVisibleInput,
    ui.playbackRenderScaleInput,
    ui.useInteractionOverrideCheck,
    ui.interactionStrideInput,
    ui.interactionMaxVisibleInput,
    ui.interactionRenderScaleInput,

    ui.usePackedVisiblePathCheck,
    ui.drawPathSelect,

    ui.bgGraySlider
  ].filter(Boolean);

  const handler = () => {
    const state = readAndSaveUiState(ui);
    if (onChange) onChange(state);
  };

  for (const control of controls) {
    control.addEventListener('change', handler);
    control.addEventListener('input', handler);
  }

  return () => {
    for (const control of controls) {
      control.removeEventListener('change', handler);
      control.removeEventListener('input', handler);
    }
  };
}

export function summarizeUiState(state) {
  const s = normalizeUiState(state);
  return {
    usePackedVisiblePath: s.usePackedVisiblePath,
    drawPath: s.drawPath,
    drawPathRole:
      s.drawPath === 'packed'
        ? 'Step24 formal'
        : s.drawPath === 'legacy'
          ? 'fallback'
          : 'future',
    drawSelectedTileOnly: s.drawSelectedTileOnly,
    bgGray: s.bgGray
  };
}
