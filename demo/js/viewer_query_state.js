import { normalizeUiState } from './viewer_ui_state.js';
import { resolveViewerCameraPreset } from './viewer_camera_presets.js';

const QUERY_DRAW_PATH_VALUES = new Set(['packed', 'gpu-screen', 'legacy']);
const QUERY_TILE_COMPOSITE_PATH_VALUES = new Set(['baseline', 'accumulation']);
const QUERY_TILE_COMPOSITE_PRIMITIVE_VALUES = new Set(['point', 'quad']);
const QUERY_INSPECT_SOURCE_VALUES = new Set(['auto', 'actual-draw', 'packed', 'gpu-screen-fallback']);
const QUERY_INSPECT_JSON_MODE_VALUES = new Set(['slim', 'full']);
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

function appendDeterministicQueryParam(params, key, value, formatter = null) {
  if (value === null || value === undefined) return;
  const formattedValue = typeof formatter === 'function' ? formatter(value) : String(value);
  params.set(key, formattedValue);
}

function formatDeterministicBoolean(value) {
  return value ? 'true' : 'false';
}

function formatDeterministicFixed(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function buildDeterministicQueryString(state) {
  const params = new URLSearchParams();
  appendDeterministicQueryParam(params, 'cameraPreset', state?.cameraPresetName);
  appendDeterministicQueryParam(params, 'time', state?.time, (value) => formatDeterministicFixed(value, 2));
  appendDeterministicQueryParam(params, 'drawPath', state?.drawPath);
  appendDeterministicQueryParam(params, 'tileCompositePath', state?.tileCompositePath);
  appendDeterministicQueryParam(params, 'tileCompositePrimitive', state?.tileCompositePrimitive);
  appendDeterministicQueryParam(params, 'inspectSource', state?.inspectSource);
  appendDeterministicQueryParam(params, 'inspectJsonMode', state?.inspectJsonMode);
  appendDeterministicQueryParam(params, 'gpuFramePolicyOverride', state?.gpuFramePolicyOverride);
  appendDeterministicQueryParam(params, 'stride', state?.stride);
  appendDeterministicQueryParam(params, 'renderScale', state?.renderScale, (value) => formatDeterministicFixed(value, 2));
  appendDeterministicQueryParam(params, 'fixedCanvasWidth', state?.fixedCanvasWidth);
  appendDeterministicQueryParam(params, 'fixedCanvasHeight', state?.fixedCanvasHeight);
  appendDeterministicQueryParam(params, 'screenshotProbeX', state?.screenshotProbeX);
  appendDeterministicQueryParam(params, 'screenshotProbeY', state?.screenshotProbeY);
  appendDeterministicQueryParam(params, 'screenshotImageWidth', state?.screenshotImageWidth);
  appendDeterministicQueryParam(params, 'screenshotImageHeight', state?.screenshotImageHeight);
  appendDeterministicQueryParam(params, 'screenshotProbeList', state?.screenshotProbeList);
  appendDeterministicQueryParam(params, 'sigmaScale', state?.sigmaScale, (value) => formatDeterministicFixed(value, 2));
  appendDeterministicQueryParam(params, 'splatScale', state?.splatScale, (value) => formatDeterministicFixed(value, 2));
  appendDeterministicQueryParam(params, 'prefilterVar', state?.prefilterVar, (value) => formatDeterministicFixed(value, 2));
  appendDeterministicQueryParam(params, 'useSH', state?.useSH, formatDeterministicBoolean);
  appendDeterministicQueryParam(params, 'useRot4d', state?.useRot4d, formatDeterministicBoolean);
  appendDeterministicQueryParam(params, 'useNativeRot4d', state?.useNativeRot4d, formatDeterministicBoolean);
  appendDeterministicQueryParam(params, 'useNativeMarginal', state?.useNativeMarginal, formatDeterministicBoolean);
  appendDeterministicQueryParam(params, 'usePackedVisiblePath', state?.usePackedVisiblePath, formatDeterministicBoolean);
  appendDeterministicQueryParam(params, 'bgGray', state?.bgGray);
  return params.toString();
}

export function parseViewerQueryState(search = window.location.search) {
  const params = new URLSearchParams(search || '');
  const cameraPresetName = params.get('cameraPreset');
  const cameraPreset = resolveViewerCameraPreset(cameraPresetName);
  const drawPath = params.get('drawPath');
  const tileCompositePath = params.get('tileCompositePath');
  const tileCompositePrimitive = params.get('tileCompositePrimitive');
  const inspectSource = params.get('inspectSource');
  const inspectJsonMode = params.get('inspectJsonMode');
  const gpuFramePolicyOverride = params.get('gpuFramePolicyOverride');

  const state = {
    active: false,
    cameraPresetName: cameraPreset?.name ?? null,
    cameraPreset,
    time: parseNumber(params.get('time'), null),
    drawPath: QUERY_DRAW_PATH_VALUES.has(drawPath) ? drawPath : null,
    tileCompositePath: QUERY_TILE_COMPOSITE_PATH_VALUES.has(tileCompositePath)
      ? tileCompositePath
      : null,
    tileCompositePrimitive: QUERY_TILE_COMPOSITE_PRIMITIVE_VALUES.has(tileCompositePrimitive)
      ? tileCompositePrimitive
      : null,
    inspectSource: QUERY_INSPECT_SOURCE_VALUES.has(inspectSource)
      ? inspectSource
      : null,
    inspectJsonMode: QUERY_INSPECT_JSON_MODE_VALUES.has(inspectJsonMode)
      ? inspectJsonMode
      : null,
    gpuFramePolicyOverride: QUERY_FRAME_POLICY_VALUES.has(gpuFramePolicyOverride)
      ? gpuFramePolicyOverride
      : null,
    stride: parseInteger(params.get('stride'), null),
    renderScale: parseNumber(params.get('renderScale'), null),
    fixedCanvasWidth: parseInteger(params.get('fixedCanvasWidth'), null),
    fixedCanvasHeight: parseInteger(params.get('fixedCanvasHeight'), null),
    screenshotProbeX: parseNumber(params.get('screenshotProbeX'), null),
    screenshotProbeY: parseNumber(params.get('screenshotProbeY'), null),
    screenshotImageWidth: parseInteger(params.get('screenshotImageWidth'), null),
    screenshotImageHeight: parseInteger(params.get('screenshotImageHeight'), null),
    screenshotProbeList: params.get('screenshotProbeList') ?? null,
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
    'tileCompositePath',
    'tileCompositePrimitive',
    'inspectSource',
    'inspectJsonMode',
    'gpuFramePolicyOverride',
    'stride',
    'renderScale',
    'fixedCanvasWidth',
    'fixedCanvasHeight',
    'screenshotProbeX',
    'screenshotProbeY',
    'screenshotImageWidth',
    'screenshotImageHeight',
    'screenshotProbeList',
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

  state.rawQueryString = String(search || '').replace(/^\?/, '');
  state.deterministicQueryString = state.active ? buildDeterministicQueryString(state) : '';
  state.deterministicUrlSummary = state.active && typeof window !== 'undefined'
    ? `${window.location.pathname}?${state.deterministicQueryString}`
    : '';

  return state;
}

export function buildViewerDeterministicSummary(queryState) {
  const state = queryState || {};
  return {
    active: !!state.active,
    cameraPresetName: state.cameraPresetName ?? 'none',
    drawPath: state.drawPath ?? 'none',
    tileCompositePath: state.tileCompositePath ?? 'baseline',
    tileCompositePrimitive: state.tileCompositePrimitive ?? 'point',
    inspectSource: state.inspectSource ?? 'auto',
    inspectJsonMode: state.inspectJsonMode ?? 'slim',
    gpuFramePolicyOverride: state.gpuFramePolicyOverride ?? 'auto',
    fixedCanvasWidth: Number.isFinite(state.fixedCanvasWidth) ? Number(state.fixedCanvasWidth) : null,
    fixedCanvasHeight: Number.isFinite(state.fixedCanvasHeight) ? Number(state.fixedCanvasHeight) : null,
    screenshotProbeX: Number.isFinite(state.screenshotProbeX) ? Number(state.screenshotProbeX) : null,
    screenshotProbeY: Number.isFinite(state.screenshotProbeY) ? Number(state.screenshotProbeY) : null,
    screenshotImageWidth: Number.isFinite(state.screenshotImageWidth) ? Number(state.screenshotImageWidth) : null,
    screenshotImageHeight: Number.isFinite(state.screenshotImageHeight) ? Number(state.screenshotImageHeight) : null,
    screenshotProbeList: state.screenshotProbeList ?? null,
    time: Number.isFinite(state.time) ? Number(state.time) : null,
    rawQueryString: state.rawQueryString ?? '',
    deterministicQueryString: state.deterministicQueryString ?? '',
    deterministicUrlSummary: state.deterministicUrlSummary ?? ''
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
  setSelectValue(ui.tileCompositePathSelect, queryState.tileCompositePath);
  setSelectValue(ui.tileCompositePrimitiveSelect, queryState.tileCompositePrimitive);

  const normalizedState = normalizeUiState({
    drawPath: ui.drawPathSelect?.value ?? 'packed',
    tileCompositePath: ui.tileCompositePathSelect?.value ?? 'baseline',
    tileCompositePrimitive: ui.tileCompositePrimitiveSelect?.value ?? 'point',
    usePackedVisiblePath: ui.usePackedVisiblePathCheck?.checked ?? true,
    bgGray: ui.bgGraySlider?.value ?? 32
  });

  return normalizedState;
}
