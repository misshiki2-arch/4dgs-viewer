import { normalizeUiState } from './viewer_ui_state.js';
import { resolveViewerCameraPreset } from './viewer_camera_presets.js';

const QUERY_DRAW_PATH_VALUES = new Set(['packed', 'gpu-screen', 'legacy']);
const QUERY_TILE_COMPOSITE_PATH_VALUES = new Set(['baseline', 'accumulation']);
const QUERY_TILE_COMPOSITE_PRIMITIVE_VALUES = new Set(['point', 'quad']);
const QUERY_INSPECT_SOURCE_VALUES = new Set(['auto', 'actual-draw', 'packed', 'gpu-screen-fallback']);
const QUERY_INSPECT_JSON_MODE_VALUES = new Set(['slim', 'full']);
const QUERY_DATASET_VIEW_MATRIX_MODE_VALUES = new Set(['threejs', 'cuda-aligned']);
const QUERY_FRAME_POLICY_VALUES = new Set([
  'auto',
  'force-transform-throughput',
  'force-draw-throughput'
]);
const QUERY_GPU_CANDIDATE_RUNTIME_VALUES = new Set(['off', 'cpu-reference', 'shadow-compare', 'limited-draw']);
const QUERY_GPU_CANDIDATE_FALLBACK_VALUES = new Set(['cpu-on-error', 'cpu-always', 'none']);
const QUERY_GPU_CANDIDATE_SUBSET_MODE_VALUES = new Set([
  'firstN',
  'visibleSrcIndices',
  'fromVisible',
  'visibleReachable'
]);
const QUERY_GPU_CANDIDATE_FILTER_MODE_VALUES = new Set([
  'all-valid',
  'evenIndex'
]);
const QUERY_GPU_CANDIDATE_SOURCE_MODE_VALUES = new Set([
  'visibleSrcIndices',
  'firstN',
  'range',
  'screenCoarse'
]);
const QUERY_GPU_CANDIDATE_PROMOTE_POLICY_VALUES = new Set([
  'never',
  'compare-ok',
  'async-ready',
  'validated-only'
]);
const QUERY_GPU_CANDIDATE_READBACK_MODE_VALUES = new Set([
  'sync-debug',
  'async-fence',
  'none'
]);
const QUERY_GPU_CANDIDATE_SCREEN_COARSE_DEPTH_MODE_VALUES = new Set([
  'positive',
  'any'
]);
const QUERY_WEBGPU_BACKEND_MODE_VALUES = new Set([
  'webgl2-fallback',
  'webgpu-dry-run',
  'webgpu-exclusive'
]);
const QUERY_WEBGPU_BACKEND_IMPLEMENTATION_VALUES = new Set([
  'webgpu-visible-record-dry-run-runtime',
  'webgpu-normal-backend-frame-implementation'
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

function parseVector3(value, fallback = null) {
  if (value === null || value === undefined || value === '') return fallback;
  const parts = String(value).split(',').map((part) => Number(part.trim()));
  if (parts.length < 3 || parts.some((part) => !Number.isFinite(part))) return fallback;
  return [parts[0], parts[1], parts[2]];
}

function parseMatrix4(value, fallback = null) {
  if (value === null || value === undefined || value === '') return fallback;
  const parts = String(value).split(',').map((part) => Number(part.trim()));
  if (parts.length !== 16 || parts.some((part) => !Number.isFinite(part))) return fallback;
  return [
    parts.slice(0, 4),
    parts.slice(4, 8),
    parts.slice(8, 12),
    parts.slice(12, 16)
  ];
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
  appendDeterministicQueryParam(params, 'datasetTransformMatrix', state?.datasetTransformMatrix, (value) => value.flat().join(','));
  appendDeterministicQueryParam(params, 'datasetCameraConvention', state?.datasetCameraConvention);
  appendDeterministicQueryParam(params, 'datasetViewMatrixMode', state?.datasetViewMatrixMode);
  appendDeterministicQueryParam(params, 'cameraControlContract', state?.cameraControlContract);
  appendDeterministicQueryParam(params, 'cameraOrientationPolicy', state?.cameraOrientationPolicy);
  appendDeterministicQueryParam(params, 'datasetPixelXSign', state?.datasetPixelXSign, (value) => String(value));
  appendDeterministicQueryParam(params, 'cameraPosition', state?.datasetCameraPosition, (value) => value.join(','));
  appendDeterministicQueryParam(params, 'cameraTarget', state?.datasetCameraTarget, (value) => value.join(','));
  appendDeterministicQueryParam(params, 'cameraUp', state?.datasetCameraUp, (value) => value.join(','));
  appendDeterministicQueryParam(params, 'cameraFoVyRad', state?.datasetCameraFoVyRad, (value) => String(value));
  appendDeterministicQueryParam(params, 'cameraFoVxRad', state?.datasetCameraFoVxRad, (value) => String(value));
  appendDeterministicQueryParam(params, 'cameraFoVy', state?.datasetCameraFoVy, (value) => String(value));
  appendDeterministicQueryParam(params, 'cameraFoVx', state?.datasetCameraFoVx, (value) => String(value));
  appendDeterministicQueryParam(params, 'datasetCameraLabel', state?.datasetCameraLabel);
  appendDeterministicQueryParam(params, 'datasetImageName', state?.datasetImageName);
  appendDeterministicQueryParam(params, 'datasetFrameNumber', state?.datasetFrameNumber);
  appendDeterministicQueryParam(params, 'datasetViewId', state?.datasetViewId);
  appendDeterministicQueryParam(params, 'datasetTime', state?.datasetTime, (value) => String(value));
  appendDeterministicQueryParam(params, 'datasetFx', state?.datasetFx, (value) => String(value));
  appendDeterministicQueryParam(params, 'datasetFy', state?.datasetFy, (value) => String(value));
  appendDeterministicQueryParam(params, 'datasetCx', state?.datasetCx, (value) => String(value));
  appendDeterministicQueryParam(params, 'datasetCy', state?.datasetCy, (value) => String(value));
  appendDeterministicQueryParam(params, 'cudaReferenceLabel', state?.cudaReferenceLabel);
  appendDeterministicQueryParam(params, 'cudaReferencePath', state?.cudaReferencePath);
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
  appendDeterministicQueryParam(params, 'debugPreserveDrawingBuffer', state?.debugPreserveDrawingBuffer, formatDeterministicBoolean);
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
  appendDeterministicQueryParam(params, 'gpuCandidateRuntime', state?.gpuCandidateRuntime);
  appendDeterministicQueryParam(params, 'gpuCandidateFallback', state?.gpuCandidateFallback);
  appendDeterministicQueryParam(params, 'gpuCandidateRequireCompare', state?.gpuCandidateRequireCompare, formatDeterministicBoolean);
  appendDeterministicQueryParam(params, 'gpuCandidateRequireShadowOk', state?.gpuCandidateRequireShadowOk, formatDeterministicBoolean);
  appendDeterministicQueryParam(params, 'gpuCandidateSubsetMode', state?.gpuCandidateSubsetMode);
  appendDeterministicQueryParam(params, 'gpuCandidateSubsetCount', state?.gpuCandidateSubsetCount);
  appendDeterministicQueryParam(params, 'gpuCandidateFilterMode', state?.gpuCandidateFilterMode);
  appendDeterministicQueryParam(params, 'gpuCandidateSourceMode', state?.gpuCandidateSourceMode);
  appendDeterministicQueryParam(params, 'gpuCandidateRangeStart', state?.gpuCandidateRangeStart);
  appendDeterministicQueryParam(params, 'gpuCandidateRangeCount', state?.gpuCandidateRangeCount);
  appendDeterministicQueryParam(params, 'gpuCandidateScreenCoarseMaxCount', state?.gpuCandidateScreenCoarseMaxCount);
  appendDeterministicQueryParam(params, 'gpuCandidateScreenCoarseMinRadiusPx', state?.gpuCandidateScreenCoarseMinRadiusPx);
  appendDeterministicQueryParam(
    params,
    'gpuCandidateScreenCoarseRequireInViewport',
    state?.gpuCandidateScreenCoarseRequireInViewport,
    formatDeterministicBoolean
  );
  appendDeterministicQueryParam(params, 'gpuCandidateScreenCoarseDepthMode', state?.gpuCandidateScreenCoarseDepthMode);
  appendDeterministicQueryParam(params, 'gpuCandidatePromotePolicy', state?.gpuCandidatePromotePolicy);
  appendDeterministicQueryParam(params, 'gpuCandidateReadbackMode', state?.gpuCandidateReadbackMode);
  appendDeterministicQueryParam(params, 'gpuCandidateCoverageCompare', state?.gpuCandidateCoverageCompare, formatDeterministicBoolean);
  appendDeterministicQueryParam(params, 'gpuCandidateCoverageMaxMisses', state?.gpuCandidateCoverageMaxMisses);
  appendDeterministicQueryParam(params, 'gpuCandidateCompare', state?.gpuCandidateCompare, formatDeterministicBoolean);
  appendDeterministicQueryParam(params, 'gpuCandidateAllowReadbackInDraw', state?.gpuCandidateAllowReadbackInDraw, formatDeterministicBoolean);
  appendDeterministicQueryParam(params, 'gpuCandidateDebugReadback', state?.gpuCandidateDebugReadback, formatDeterministicBoolean);
  appendDeterministicQueryParam(params, 'gpuVisibleRecordDryRun', state?.gpuVisibleRecordDryRun, formatDeterministicBoolean);
  appendDeterministicQueryParam(params, 'gpuVisibleRecordSource', state?.gpuVisibleRecordSource);
  appendDeterministicQueryParam(params, 'gpuVisibleRecordReadback', state?.gpuVisibleRecordReadback);
  appendDeterministicQueryParam(params, 'gpuVisibleRecordMaxCount', state?.gpuVisibleRecordMaxCount);
  appendDeterministicQueryParam(params, 'gpuVisibleRecordCompare', state?.gpuVisibleRecordCompare, formatDeterministicBoolean);
  appendDeterministicQueryParam(params, 'gpuRawVisibleRecordDryRun', state?.gpuRawVisibleRecordDryRun, formatDeterministicBoolean);
  appendDeterministicQueryParam(params, 'gpuRawVisibleRecordMode', state?.gpuRawVisibleRecordMode);
  appendDeterministicQueryParam(params, 'gpuRawVisibleRecordFields', state?.gpuRawVisibleRecordFields);
  appendDeterministicQueryParam(params, 'gpuRawAttributeTexture', state?.gpuRawAttributeTexture, formatDeterministicBoolean);
  appendDeterministicQueryParam(params, 'gpuRawVisibleRecordReadback', state?.gpuRawVisibleRecordReadback);
  appendDeterministicQueryParam(params, 'webgpuVisibleRecordDryRun', state?.webgpuVisibleRecordDryRun, formatDeterministicBoolean);
  appendDeterministicQueryParam(params, 'webgpuVisibleRecordMaxCount', state?.webgpuVisibleRecordMaxCount);
  appendDeterministicQueryParam(params, 'webgpuVisibleRecordFields', state?.webgpuVisibleRecordFields);
  appendDeterministicQueryParam(params, 'webgpuBackendMode', state?.webgpuBackendMode);
  appendDeterministicQueryParam(params, 'webgpuBackendImplementation', state?.webgpuBackendImplementation);
  appendDeterministicQueryParam(params, 'webgpuAllowViewerCanvasPresentation', state?.webgpuAllowViewerCanvasPresentation, formatDeterministicBoolean);
  appendDeterministicQueryParam(params, 'webgpuBackendViewerLoopHook', state?.webgpuBackendViewerLoopHook, formatDeterministicBoolean);
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
  const datasetViewMatrixMode = params.get('datasetViewMatrixMode');
  const cameraControlContract = params.get('cameraControlContract');
  const cameraOrientationPolicy = params.get('cameraOrientationPolicy');
  const datasetPixelXSign = parseInteger(params.get('datasetPixelXSign'), null);
  const gpuCandidateRuntime = params.get('gpuCandidateRuntime');
  const gpuCandidateFallback = params.get('gpuCandidateFallback');
  const gpuCandidateSubsetMode = params.get('gpuCandidateSubsetMode');
  const gpuCandidateFilterMode = params.get('gpuCandidateFilterMode');
  const gpuCandidateSourceMode = params.get('gpuCandidateSourceMode');
  const gpuCandidatePromotePolicy = params.get('gpuCandidatePromotePolicy');
  const gpuCandidateReadbackMode = params.get('gpuCandidateReadbackMode');
  const gpuCandidateScreenCoarseDepthMode = params.get('gpuCandidateScreenCoarseDepthMode');
  const gpuVisibleRecordSource = params.get('gpuVisibleRecordSource');
  const gpuVisibleRecordReadback = params.get('gpuVisibleRecordReadback');
  const gpuRawVisibleRecordReadback = params.get('gpuRawVisibleRecordReadback');
  const webgpuBackendMode = params.get('webgpuBackendMode');
  const webgpuBackendImplementation = params.get('webgpuBackendImplementation');

  const state = {
    active: false,
    cameraPresetName: cameraPreset?.name ?? null,
    cameraPreset,
    datasetTransformMatrix: parseMatrix4(params.get('datasetTransformMatrix'), null),
    datasetCameraConvention: params.get('datasetCameraConvention') ?? null,
    datasetViewMatrixMode: QUERY_DATASET_VIEW_MATRIX_MODE_VALUES.has(datasetViewMatrixMode)
      ? datasetViewMatrixMode
      : null,
    cameraControlContract: cameraControlContract || null,
    cameraOrientationPolicy: cameraOrientationPolicy || null,
    datasetPixelXSign: [-1, 1].includes(datasetPixelXSign)
      ? datasetPixelXSign
      : null,
    datasetCameraPosition: parseVector3(params.get('cameraPosition'), null),
    datasetCameraTarget: parseVector3(params.get('cameraTarget'), null),
    datasetCameraUp: parseVector3(params.get('cameraUp'), null),
    datasetCameraFoVyRad: parseNumber(params.get('cameraFoVyRad'), parseNumber(params.get('cameraFoVy'), null)),
    datasetCameraFoVxRad: parseNumber(params.get('cameraFoVxRad'), parseNumber(params.get('cameraFoVx'), null)),
    datasetCameraFoVy: parseNumber(params.get('cameraFoVy'), null),
    datasetCameraFoVx: parseNumber(params.get('cameraFoVx'), null),
    datasetCameraLabel: params.get('datasetCameraLabel') ?? null,
    datasetImageName: params.get('datasetImageName') ?? null,
    datasetFrameNumber: parseInteger(params.get('datasetFrameNumber'), null),
    datasetViewId: parseInteger(params.get('datasetViewId'), null),
    datasetTime: parseNumber(params.get('datasetTime'), null),
    datasetFx: parseNumber(params.get('datasetFx'), null),
    datasetFy: parseNumber(params.get('datasetFy'), null),
    datasetCx: parseNumber(params.get('datasetCx'), null),
    datasetCy: parseNumber(params.get('datasetCy'), null),
    cudaReferenceLabel: params.get('cudaReferenceLabel') ?? null,
    cudaReferencePath: params.get('cudaReferencePath') ?? null,
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
    debugPreserveDrawingBuffer: parseBoolean(
      params.get('debugPreserveDrawingBuffer'),
      parseBoolean(params.get('preserveDrawingBuffer'), null)
    ),
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
    gpuCandidateRuntime: QUERY_GPU_CANDIDATE_RUNTIME_VALUES.has(gpuCandidateRuntime)
      ? (gpuCandidateRuntime === 'off' ? 'cpu-reference' : gpuCandidateRuntime)
      : null,
    gpuCandidateFallback: QUERY_GPU_CANDIDATE_FALLBACK_VALUES.has(gpuCandidateFallback)
      ? gpuCandidateFallback
      : null,
    gpuCandidateRequireCompare: parseBoolean(params.get('gpuCandidateRequireCompare'), null),
    gpuCandidateRequireShadowOk: parseBoolean(params.get('gpuCandidateRequireShadowOk'), null),
    gpuCandidateSubsetMode: QUERY_GPU_CANDIDATE_SUBSET_MODE_VALUES.has(gpuCandidateSubsetMode)
      ? gpuCandidateSubsetMode
      : null,
    gpuCandidateSubsetCount: parseInteger(params.get('gpuCandidateSubsetCount'), null),
    gpuCandidateFilterMode: QUERY_GPU_CANDIDATE_FILTER_MODE_VALUES.has(gpuCandidateFilterMode)
      ? gpuCandidateFilterMode
      : null,
    gpuCandidateSourceMode: QUERY_GPU_CANDIDATE_SOURCE_MODE_VALUES.has(gpuCandidateSourceMode)
      ? gpuCandidateSourceMode
      : null,
    gpuCandidateRangeStart: parseInteger(params.get('gpuCandidateRangeStart'), null),
    gpuCandidateRangeCount: parseInteger(params.get('gpuCandidateRangeCount'), null),
    gpuCandidateScreenCoarseMaxCount: parseInteger(params.get('gpuCandidateScreenCoarseMaxCount'), null),
    gpuCandidateScreenCoarseMinRadiusPx: parseNumber(params.get('gpuCandidateScreenCoarseMinRadiusPx'), null),
    gpuCandidateScreenCoarseRequireInViewport: parseBoolean(
      params.get('gpuCandidateScreenCoarseRequireInViewport'),
      null
    ),
    gpuCandidateScreenCoarseDepthMode: QUERY_GPU_CANDIDATE_SCREEN_COARSE_DEPTH_MODE_VALUES.has(gpuCandidateScreenCoarseDepthMode)
      ? gpuCandidateScreenCoarseDepthMode
      : null,
    gpuCandidatePromotePolicy: QUERY_GPU_CANDIDATE_PROMOTE_POLICY_VALUES.has(gpuCandidatePromotePolicy)
      ? gpuCandidatePromotePolicy
      : null,
    gpuCandidateReadbackMode: QUERY_GPU_CANDIDATE_READBACK_MODE_VALUES.has(gpuCandidateReadbackMode)
      ? gpuCandidateReadbackMode
      : null,
    gpuCandidateCoverageCompare: parseBoolean(params.get('gpuCandidateCoverageCompare'), null),
    gpuCandidateCoverageMaxMisses: parseInteger(params.get('gpuCandidateCoverageMaxMisses'), null),
    gpuCandidateCompare: parseBoolean(params.get('gpuCandidateCompare'), null),
    gpuCandidateAllowReadbackInDraw: parseBoolean(params.get('gpuCandidateAllowReadbackInDraw'), null),
    gpuCandidateDebugReadback: parseBoolean(params.get('gpuCandidateDebugReadback'), null),
    gpuVisibleRecordDryRun: parseBoolean(params.get('gpuVisibleRecordDryRun'), null),
    gpuVisibleRecordSource: QUERY_GPU_CANDIDATE_SOURCE_MODE_VALUES.has(gpuVisibleRecordSource)
      ? gpuVisibleRecordSource
      : null,
    gpuVisibleRecordReadback: QUERY_GPU_CANDIDATE_READBACK_MODE_VALUES.has(gpuVisibleRecordReadback)
      ? gpuVisibleRecordReadback
      : null,
    gpuVisibleRecordMaxCount: parseInteger(params.get('gpuVisibleRecordMaxCount'), null),
    gpuVisibleRecordCompare: parseBoolean(params.get('gpuVisibleRecordCompare'), null),
    gpuRawVisibleRecordDryRun: parseBoolean(params.get('gpuRawVisibleRecordDryRun'), null),
    gpuRawVisibleRecordMode: params.get('gpuRawVisibleRecordMode') ?? null,
    gpuRawVisibleRecordFields: params.get('gpuRawVisibleRecordFields') ?? null,
    gpuRawAttributeTexture: parseBoolean(params.get('gpuRawAttributeTexture'), null),
    gpuRawVisibleRecordReadback: QUERY_GPU_CANDIDATE_READBACK_MODE_VALUES.has(gpuRawVisibleRecordReadback)
      ? gpuRawVisibleRecordReadback
      : null,
    webgpuVisibleRecordDryRun: parseBoolean(params.get('webgpuVisibleRecordDryRun'), null),
    webgpuVisibleRecordMaxCount: parseInteger(params.get('webgpuVisibleRecordMaxCount'), null),
    webgpuVisibleRecordFields: params.get('webgpuVisibleRecordFields') ?? null,
    webgpuBackendMode: QUERY_WEBGPU_BACKEND_MODE_VALUES.has(webgpuBackendMode)
      ? webgpuBackendMode
      : null,
    webgpuBackendImplementation:
      QUERY_WEBGPU_BACKEND_IMPLEMENTATION_VALUES.has(webgpuBackendImplementation)
        ? webgpuBackendImplementation
        : null,
    webgpuAllowViewerCanvasPresentation: parseBoolean(
      params.get('webgpuAllowViewerCanvasPresentation'),
      null
    ),
    webgpuBackendViewerLoopHook: parseBoolean(
      params.get('webgpuBackendViewerLoopHook'),
      null
    ),
    bgGray: parseInteger(params.get('bgGray'), null)
  };

  state.active = [
    'cameraPreset',
    'datasetTransformMatrix',
    'datasetCameraConvention',
    'datasetViewMatrixMode',
    'cameraControlContract',
    'cameraOrientationPolicy',
    'datasetPixelXSign',
    'cameraPosition',
    'cameraTarget',
    'cameraUp',
    'cameraFoVyRad',
    'cameraFoVxRad',
    'cameraFoVy',
    'cameraFoVx',
    'datasetCameraLabel',
    'datasetImageName',
    'datasetFrameNumber',
    'datasetViewId',
    'datasetTime',
    'datasetFx',
    'datasetFy',
    'datasetCx',
    'datasetCy',
    'cudaReferenceLabel',
    'cudaReferencePath',
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
    'debugPreserveDrawingBuffer',
    'preserveDrawingBuffer',
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
    'gpuCandidateRuntime',
    'gpuCandidateFallback',
    'gpuCandidateRequireCompare',
    'gpuCandidateRequireShadowOk',
    'gpuCandidateSubsetMode',
    'gpuCandidateSubsetCount',
    'gpuCandidateFilterMode',
    'gpuCandidateSourceMode',
    'gpuCandidateRangeStart',
    'gpuCandidateRangeCount',
    'gpuCandidateScreenCoarseMaxCount',
    'gpuCandidateScreenCoarseMinRadiusPx',
    'gpuCandidateScreenCoarseRequireInViewport',
    'gpuCandidateScreenCoarseDepthMode',
    'gpuCandidatePromotePolicy',
    'gpuCandidateReadbackMode',
    'gpuCandidateCoverageCompare',
    'gpuCandidateCoverageMaxMisses',
    'gpuCandidateCompare',
    'gpuCandidateAllowReadbackInDraw',
    'gpuCandidateDebugReadback',
    'gpuVisibleRecordDryRun',
    'gpuVisibleRecordSource',
    'gpuVisibleRecordReadback',
    'gpuVisibleRecordMaxCount',
    'gpuVisibleRecordCompare',
    'gpuRawVisibleRecordDryRun',
    'gpuRawVisibleRecordMode',
    'gpuRawVisibleRecordFields',
    'gpuRawAttributeTexture',
    'gpuRawVisibleRecordReadback',
    'webgpuVisibleRecordDryRun',
    'webgpuVisibleRecordMaxCount',
    'webgpuVisibleRecordFields',
    'webgpuBackendMode',
    'webgpuBackendImplementation',
    'webgpuAllowViewerCanvasPresentation',
    'webgpuBackendViewerLoopHook',
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
    cameraSource: Array.isArray(state.datasetTransformMatrix) ? 'dataset-transform-matrix' :
      (Array.isArray(state.datasetCameraPosition) &&
      Array.isArray(state.datasetCameraTarget))
      ? 'dataset-query-camera'
      : 'camera-preset',
    datasetTransformMatrix: Array.isArray(state.datasetTransformMatrix) ? state.datasetTransformMatrix.map((row) => [...row]) : null,
    datasetCameraConvention: state.datasetCameraConvention ?? null,
    datasetViewMatrixMode: state.datasetViewMatrixMode ?? 'threejs',
    cameraControlContract: state.cameraControlContract ?? null,
    cameraOrientationPolicy: state.cameraOrientationPolicy ?? null,
    datasetPixelXSign: [-1, 1].includes(state.datasetPixelXSign) ? Number(state.datasetPixelXSign) : 1,
    datasetCameraLabel: state.datasetCameraLabel ?? null,
    datasetImageName: state.datasetImageName ?? null,
    datasetFrameNumber: Number.isFinite(state.datasetFrameNumber) ? Number(state.datasetFrameNumber) : null,
    datasetViewId: Number.isFinite(state.datasetViewId) ? Number(state.datasetViewId) : null,
    datasetTime: Number.isFinite(state.datasetTime) ? Number(state.datasetTime) : null,
    datasetCameraPosition: Array.isArray(state.datasetCameraPosition) ? [...state.datasetCameraPosition] : null,
    datasetCameraTarget: Array.isArray(state.datasetCameraTarget) ? [...state.datasetCameraTarget] : null,
    datasetCameraUp: Array.isArray(state.datasetCameraUp) ? [...state.datasetCameraUp] : null,
    datasetCameraFoVyRad: Number.isFinite(state.datasetCameraFoVyRad) ? Number(state.datasetCameraFoVyRad) : null,
    datasetCameraFoVxRad: Number.isFinite(state.datasetCameraFoVxRad) ? Number(state.datasetCameraFoVxRad) : null,
    datasetCameraFoVy: Number.isFinite(state.datasetCameraFoVy) ? Number(state.datasetCameraFoVy) : null,
    datasetCameraFoVx: Number.isFinite(state.datasetCameraFoVx) ? Number(state.datasetCameraFoVx) : null,
    datasetFx: Number.isFinite(state.datasetFx) ? Number(state.datasetFx) : null,
    datasetFy: Number.isFinite(state.datasetFy) ? Number(state.datasetFy) : null,
    datasetCx: Number.isFinite(state.datasetCx) ? Number(state.datasetCx) : null,
    datasetCy: Number.isFinite(state.datasetCy) ? Number(state.datasetCy) : null,
    cudaReferenceLabel: state.cudaReferenceLabel ?? null,
    cudaReferencePath: state.cudaReferencePath ?? null,
    drawPath: state.drawPath ?? 'none',
    tileCompositePath: state.tileCompositePath ?? 'baseline',
    tileCompositePrimitive: state.tileCompositePrimitive ?? 'point',
    inspectSource: state.inspectSource ?? 'auto',
    inspectJsonMode: state.inspectJsonMode ?? 'slim',
    gpuFramePolicyOverride: state.gpuFramePolicyOverride ?? 'auto',
    fixedCanvasWidth: Number.isFinite(state.fixedCanvasWidth) ? Number(state.fixedCanvasWidth) : null,
    fixedCanvasHeight: Number.isFinite(state.fixedCanvasHeight) ? Number(state.fixedCanvasHeight) : null,
    debugPreserveDrawingBuffer: typeof state.debugPreserveDrawingBuffer === 'boolean'
      ? state.debugPreserveDrawingBuffer
      : null,
    screenshotProbeX: Number.isFinite(state.screenshotProbeX) ? Number(state.screenshotProbeX) : null,
    screenshotProbeY: Number.isFinite(state.screenshotProbeY) ? Number(state.screenshotProbeY) : null,
    screenshotImageWidth: Number.isFinite(state.screenshotImageWidth) ? Number(state.screenshotImageWidth) : null,
    screenshotImageHeight: Number.isFinite(state.screenshotImageHeight) ? Number(state.screenshotImageHeight) : null,
    screenshotProbeList: state.screenshotProbeList ?? null,
    stride: Number.isFinite(state.stride) ? Number(state.stride) : null,
    bgGray: Number.isFinite(state.bgGray) ? Number(state.bgGray) : null,
    time: Number.isFinite(state.time) ? Number(state.time) : null,
    gpuCandidateRuntime: state.gpuCandidateRuntime ?? 'cpu-reference',
    gpuCandidateFallback: state.gpuCandidateFallback ?? null,
    gpuCandidateRequireCompare: typeof state.gpuCandidateRequireCompare === 'boolean'
      ? state.gpuCandidateRequireCompare
      : null,
    gpuCandidateRequireShadowOk: typeof state.gpuCandidateRequireShadowOk === 'boolean'
      ? state.gpuCandidateRequireShadowOk
      : null,
    gpuCandidateSubsetMode: state.gpuCandidateSubsetMode ?? null,
    gpuCandidateSubsetCount: Number.isFinite(state.gpuCandidateSubsetCount) ? Number(state.gpuCandidateSubsetCount) : null,
    gpuCandidateFilterMode: state.gpuCandidateFilterMode ?? null,
    gpuCandidateSourceMode: state.gpuCandidateSourceMode ?? null,
    gpuCandidateRangeStart: Number.isFinite(state.gpuCandidateRangeStart) ? Number(state.gpuCandidateRangeStart) : null,
    gpuCandidateRangeCount: Number.isFinite(state.gpuCandidateRangeCount) ? Number(state.gpuCandidateRangeCount) : null,
    gpuCandidateScreenCoarseMaxCount: Number.isFinite(state.gpuCandidateScreenCoarseMaxCount)
      ? Number(state.gpuCandidateScreenCoarseMaxCount)
      : null,
    gpuCandidateScreenCoarseMinRadiusPx: Number.isFinite(state.gpuCandidateScreenCoarseMinRadiusPx)
      ? Number(state.gpuCandidateScreenCoarseMinRadiusPx)
      : null,
    gpuCandidateScreenCoarseRequireInViewport: typeof state.gpuCandidateScreenCoarseRequireInViewport === 'boolean'
      ? state.gpuCandidateScreenCoarseRequireInViewport
      : null,
    gpuCandidateScreenCoarseDepthMode: state.gpuCandidateScreenCoarseDepthMode ?? null,
    gpuCandidatePromotePolicy: state.gpuCandidatePromotePolicy ?? null,
    gpuCandidateReadbackMode: state.gpuCandidateReadbackMode ?? null,
    gpuCandidateCoverageCompare: typeof state.gpuCandidateCoverageCompare === 'boolean'
      ? state.gpuCandidateCoverageCompare
      : null,
    gpuCandidateCoverageMaxMisses: Number.isFinite(state.gpuCandidateCoverageMaxMisses)
      ? Number(state.gpuCandidateCoverageMaxMisses)
      : null,
    gpuCandidateCompare: typeof state.gpuCandidateCompare === 'boolean'
      ? state.gpuCandidateCompare
      : null,
    gpuCandidateAllowReadbackInDraw: typeof state.gpuCandidateAllowReadbackInDraw === 'boolean'
      ? state.gpuCandidateAllowReadbackInDraw
      : null,
    gpuCandidateDebugReadback: typeof state.gpuCandidateDebugReadback === 'boolean'
      ? state.gpuCandidateDebugReadback
      : null,
    gpuVisibleRecordDryRun: typeof state.gpuVisibleRecordDryRun === 'boolean'
      ? state.gpuVisibleRecordDryRun
      : null,
    gpuVisibleRecordSource: state.gpuVisibleRecordSource ?? null,
    gpuVisibleRecordReadback: state.gpuVisibleRecordReadback ?? null,
    gpuVisibleRecordMaxCount: Number.isFinite(state.gpuVisibleRecordMaxCount)
      ? Number(state.gpuVisibleRecordMaxCount)
      : null,
    gpuVisibleRecordCompare: typeof state.gpuVisibleRecordCompare === 'boolean'
      ? state.gpuVisibleRecordCompare
      : null,
    gpuRawVisibleRecordDryRun: typeof state.gpuRawVisibleRecordDryRun === 'boolean'
      ? state.gpuRawVisibleRecordDryRun
      : null,
    gpuRawVisibleRecordMode: state.gpuRawVisibleRecordMode ?? null,
    gpuRawVisibleRecordFields: state.gpuRawVisibleRecordFields ?? null,
    gpuRawAttributeTexture: typeof state.gpuRawAttributeTexture === 'boolean'
      ? state.gpuRawAttributeTexture
      : null,
    gpuRawVisibleRecordReadback: state.gpuRawVisibleRecordReadback ?? null,
    webgpuVisibleRecordDryRun: typeof state.webgpuVisibleRecordDryRun === 'boolean'
      ? state.webgpuVisibleRecordDryRun
      : null,
    webgpuVisibleRecordMaxCount: Number.isFinite(state.webgpuVisibleRecordMaxCount)
      ? Number(state.webgpuVisibleRecordMaxCount)
      : null,
    webgpuVisibleRecordFields: state.webgpuVisibleRecordFields ?? null,
    webgpuBackendMode: state.webgpuBackendMode ?? null,
    webgpuBackendImplementation: state.webgpuBackendImplementation ?? null,
    webgpuAllowViewerCanvasPresentation:
      typeof state.webgpuAllowViewerCanvasPresentation === 'boolean'
        ? state.webgpuAllowViewerCanvasPresentation
        : null,
    webgpuBackendViewerLoopHook:
      typeof state.webgpuBackendViewerLoopHook === 'boolean'
        ? state.webgpuBackendViewerLoopHook
        : null,
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
  const cudaAlignedDefaults = queryState.datasetViewMatrixMode === 'cuda-aligned';
  setCheckboxValue(
    ui.useNativeRot4dCheck,
    queryState.useNativeRot4d ?? (cudaAlignedDefaults ? true : null)
  );
  setCheckboxValue(
    ui.useNativeMarginalCheck,
    queryState.useNativeMarginal ?? (cudaAlignedDefaults ? true : null)
  );
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
