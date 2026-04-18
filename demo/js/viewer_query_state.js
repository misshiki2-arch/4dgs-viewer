import { normalizeUiState } from './viewer_ui_state.js';
import { resolveViewerCameraPreset } from './viewer_camera_presets.js';

const QUERY_DRAW_PATH_VALUES = new Set(['packed', 'gpu-screen', 'legacy']);
const QUERY_FRAME_POLICY_VALUES = new Set([
  'auto',
  'force-transform-throughput',
  'force-draw-throughput'
]);

function parseNumber(value, fallback = null) {
  if (value === null || value === undefined || value === '') return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function parseInteger(value, fallback = null) {
  const n = parseNumber(value, fallback);
  return Number.isFinite(n) ? (n | 0) : fallback;
}

function parseBoolean(value, fallback = null) {
  if (value === null || value === undefined || value === '') return fallback;
  if (typeof value === 'boolean') return value;
  const normalized = String(value).trim().toLowerCase();
  if (normalized === '1' || normalized === 'true' || normalized === 'yes' || normalized === 'on') return true;
  if (normalized === '0' || normalized === 'false' || normalized === 'no' || normalized === 'off') return false;
  return fallback;
}

function setSliderValue(sliderEl, valueEl, value, digits = null) {
  if (!sliderEl || value === null || value === undefined) return;
  sliderEl.value = String(value);
  if (!valueEl) return;
  if (digits === null) {
    valueEl.textContent = String(value);
    return;
  }
  valueEl.textContent = Number(sliderEl.value).toFixed(digits);
}

function setCheckboxValue(el, value) {
  if (!el || value === null || value === undefined) return;
  el.checked = !!value;
}

function setSelectValue(el, value) {
  if (!el || value === null || value === undefined) return;
  el.value = String(value);
}

export function parseViewerQueryState(search = window.location.search) {
  const params = new URLSearchParams(search || '');
  const cameraPresetName = params.get('cameraPreset');
  const cameraPreset = resolveViewerCameraPreset(cameraPresetName);
  const drawPath = params.get('drawPath');
  const gpuFramePolicyOverride = params.get('gpuFramePolicyOverride');

  const state = {
    active: false,
    cameraPresetName: cameraPreset?.name ?? null,
    cameraPreset,
    time: parseNumber(params.get('time'), null),
    drawPath: QUERY_DRAW_PATH_VALUES.has(drawPath) ? drawPath : null,
    gpuFramePolicyOverride: QUERY_FRAME_POLICY_VALUES.has(gpuFramePolicyOverride)
      ? gpuFramePolicyOverride
      : null,
    stride: parseInteger(params.get('stride'), null),
    renderScale: parseNumber(params.get('renderScale'), null),
    sigmaScale: parseNumber(params.get('sigmaScale'), null),
    splatScale: parseNumber(params.get('splatScale'), null),
    prefilterVar: parseNumber(params.get('prefilterVar'), null),
    useSH: parseBoolean(params.get('useSH'), null),
    useRot4d: parseBoolean(params.get('useRot4d'), null),
    useNativeRot4d: parseBoolean(params.get('useNativeRot4d'), null),
    useNativeMarginal: parseBoolean(params.get('useNativeMarginal'), null),
    usePackedVisiblePath: parseBoolean(params.get('usePackedVisiblePath'), null),
    bgGray: parseInteger(params.get('bgGray'), null)
  };

  state.active = [
    'cameraPreset',
    'time',
    'drawPath',
    'gpuFramePolicyOverride',
    'stride',
    'renderScale',
    'sigmaScale',
    'splatScale',
    'prefilterVar',
    'useSH',
    'useRot4d',
    'useNativeRot4d',
    'useNativeMarginal',
    'usePackedVisiblePath',
    'bgGray'
  ].some((key) => params.has(key));

  return state;
}

export function buildViewerDeterministicSummary(queryState) {
  const state = queryState || {};
  return {
    active: !!state.active,
    cameraPresetName: state.cameraPresetName ?? 'none',
    drawPath: state.drawPath ?? 'none',
    gpuFramePolicyOverride: state.gpuFramePolicyOverride ?? 'auto',
    time: Number.isFinite(state.time) ? Number(state.time) : null
  };
}

export function applyViewerQueryStateToUi(ui, queryState) {
  if (!ui || !queryState?.active) {
    return normalizeUiState({});
  }

  setSliderValue(ui.timeSlider, ui.timeVal, queryState.time, 2);
  setSliderValue(ui.splatScaleSlider, ui.splatScaleVal, queryState.splatScale, 2);
  setSliderValue(ui.sigmaScaleSlider, ui.sigmaScaleVal, queryState.sigmaScale, 2);
  setSliderValue(ui.prefilterVarSlider, ui.prefilterVarVal, queryState.prefilterVar, 2);
  setSliderValue(ui.renderScaleSlider, ui.renderScaleVal, queryState.renderScale, 2);
  setSliderValue(ui.strideSlider, ui.strideVal, queryState.stride, null);
  setSliderValue(ui.bgGraySlider, ui.bgGrayVal, queryState.bgGray, null);

  setCheckboxValue(ui.useSHCheck, queryState.useSH);
  setCheckboxValue(ui.useRot4dCheck, queryState.useRot4d);
  setCheckboxValue(ui.useNativeRot4dCheck, queryState.useNativeRot4d);
  setCheckboxValue(ui.useNativeMarginalCheck, queryState.useNativeMarginal);
  setCheckboxValue(ui.usePackedVisiblePathCheck, queryState.usePackedVisiblePath);
  setSelectValue(ui.drawPathSelect, queryState.drawPath);

  const normalizedState = normalizeUiState({
    drawPath: ui.drawPathSelect?.value ?? 'packed',
    usePackedVisiblePath: ui.usePackedVisiblePathCheck?.checked ?? true,
    bgGray: ui.bgGraySlider?.value ?? 32
  });

  return normalizedState;
}
