import { parseSplat4DV2 } from './splat4d_parser_v2.js';
import { fitCameraToRaw, computeGaussianDebugState, DEFAULT_SINGLE_SPLAT_COMPARE_INPUT } from './rot4d_math.js';
import { renderGpuFrame } from './gpu_renderer.js';
import {
  inspectGpuPackedPayloadItem,
  inspectPackedInterleavedTileCompositeItem
} from './gpu_packed_payload_draw_shared.js';
import {
  createGpuInteractionState,
  bindGpuDragInteraction
} from './gpu_interaction_utils.js';
import { buildEffectiveGpuQualityConfig } from './gpu_quality_override_utils.js';
import {
  applyInfoWrapStyle,
  applyPanelResizeStyle,
  ensureTileDebugControls,
  ensureTemporalIndexControls,
  ensureTemporalBucketControls,
  ensureQualityOverrideControls,
  ensurePackedPathControls,
  ensureDebugLogControls,
  ensureDeterministicStateNote,
  setDebugLogText,
  copyDebugLogText
} from './viewer_ui_controls.js';
import {
  loadAndApplyUiState,
  readAndSaveUiState,
  bindUiStatePersistence,
  summarizeUiState
} from './viewer_ui_state.js';
import { createRenderScheduler } from './viewer_render_scheduler.js';
import { createViewerPlayback } from './viewer_playback.js';
import { createViewerFileIO } from './viewer_file_io.js';
import { createViewerScene, applyViewerCameraPresetState } from './viewer_scene_setup.js';
import {
  parseViewerQueryState,
  buildViewerDeterministicSummary,
  applyViewerQueryStateToUi
} from './viewer_query_state.js';

const canvas = document.getElementById('glCanvas');

const ui = {
  fileInput: document.getElementById('file'),
  timeSlider: document.getElementById('time'),
  timeVal: document.getElementById('timeVal'),
  splatScaleSlider: document.getElementById('splatScale'),
  splatScaleVal: document.getElementById('splatScaleVal'),
  sigmaScaleSlider: document.getElementById('sigmaScale'),
  sigmaScaleVal: document.getElementById('sigmaScaleVal'),
  prefilterVarSlider: document.getElementById('prefilterVar'),
  prefilterVarVal: document.getElementById('prefilterVarVal'),
  renderScaleSlider: document.getElementById('renderScale'),
  renderScaleVal: document.getElementById('renderScaleVal'),
  strideSlider: document.getElementById('stride'),
  strideVal: document.getElementById('strideVal'),
  maxVisibleSlider: document.getElementById('maxVisible'),
  maxVisibleVal: document.getElementById('maxVisibleVal'),
  bgGraySlider: document.getElementById('bgGray'),
  bgGrayVal: document.getElementById('bgGrayVal'),
  useSHCheck: document.getElementById('useSH'),
  useRot4dCheck: document.getElementById('useRot4d'),
  useNativeRot4dCheck: document.getElementById('useNativeRot4d'),
  useNativeMarginalCheck: document.getElementById('useNativeMarginal'),
  forceSh3dCheck: document.getElementById('forceSh3d'),
  timeDurationSlider: document.getElementById('timeDuration'),
  timeDurationVal: document.getElementById('timeDurationVal'),
  playBtn: document.getElementById('play'),
  renderBtn: document.getElementById('renderBtn'),
  resetCamBtn: document.getElementById('resetCam'),
  info: document.getElementById('info'),
  drop: document.getElementById('drop')
};

ensureTileDebugControls(ui);
ensureTemporalIndexControls(ui);
ensureTemporalBucketControls(ui);
ensureQualityOverrideControls(ui);
ensurePackedPathControls(ui);
ensureDebugLogControls(ui);
ensureDeterministicStateNote(ui);

applyInfoWrapStyle(ui.info);
applyPanelResizeStyle(ui.info);

const scene = createViewerScene(canvas);
const { camera, controls, ensureGpu, getGpu, setCanvasSize } = scene;

let raw = null;
let lastDebugText = '';
let uiUnbindPersistence = null;
const tokenRef = { value: 0 };
const interactionState = createGpuInteractionState();
let playback = null;
let latestRenderResult = null;
const deterministicQueryState = parseViewerQueryState();
let appliedCameraPresetName = deterministicQueryState.cameraPresetName ?? 'none';
let lastSnapshotSummary = {
  available: true,
  source: 'webgl-default-framebuffer-readpixels',
  renderWaitMode: 'direct-render-await',
  status: 'idle',
  reason: 'none'
};

const INSPECT_SOURCE_VALUES = new Set(['auto', 'actual-draw', 'packed', 'gpu-screen-fallback']);
const INSPECT_JSON_MODE_VALUES = new Set(['slim', 'full']);
const SHARED_REPRESENTATIVE_PIXEL_STORAGE_KEY = 'step86.sharedRepresentativePixel';
const SHARED_REPRESENTATIVE_ACCUMULATION_COLOR_STORAGE_KEY = 'step86.sharedRepresentativeAccumulationColor';
const SHARED_REPRESENTATIVE_DEFAULT_PIXEL = [2949, 688];
const SHARED_REPRESENTATIVE_COLOR_MATCH_TOLERANCE = 2.0 / 255.0;
const MAPPED_PROBE_PATCH_SIZES = [7, 21, 31];
const MAPPED_PROBE_PATCH_REFERENCE_STORAGE_PREFIX = 'step87.mappedProbePatchReference.';
const MAPPED_PROBE_PATCH_TOLERANCE = 2.0 / 255.0;

function refreshLatestDebugText(explicitText = null) {
  const text = explicitText ?? lastDebugText ?? ui.info?.textContent ?? '';
  lastDebugText = text;
  return text;
}

function exportLatestDebugTextToArea() {
  setDebugLogText(ui, refreshLatestDebugText());
}

function updateDeterministicStateNote() {
  if (!ui.deterministicStateNote) return;

  if (!deterministicQueryState.active) {
    ui.deterministicStateNote.textContent =
      'URL query can fix cameraPreset/time/drawPath/gpuFramePolicyOverride and window.gpuViewerDebug.captureFrame(...) can save the current canvas';
    return;
  }

  const parts = [];
  parts.push(`query active`);
  parts.push(`cameraPreset=${deterministicQueryState.cameraPresetName ?? 'none'}`);
  parts.push(`drawPath=${deterministicQueryState.drawPath ?? 'default'}`);
  parts.push(`tileCompositePath=${deterministicQueryState.tileCompositePath ?? 'baseline'}`);
  parts.push(`tileCompositePrimitive=${deterministicQueryState.tileCompositePrimitive ?? 'point'}`);
  parts.push(`inspectSource=${deterministicQueryState.inspectSource ?? 'auto'}`);
  parts.push(`inspectJsonMode=${deterministicQueryState.inspectJsonMode ?? 'slim'}`);
  parts.push(`gpuFramePolicyOverride=${deterministicQueryState.gpuFramePolicyOverride ?? 'auto'}`);
  if (Number.isFinite(deterministicQueryState.fixedCanvasWidth) &&
      Number.isFinite(deterministicQueryState.fixedCanvasHeight)) {
    parts.push(`fixedCanvas=${deterministicQueryState.fixedCanvasWidth}x${deterministicQueryState.fixedCanvasHeight}`);
  }
  if (deterministicQueryState.deterministicQueryString) {
    parts.push(`query=${deterministicQueryState.deterministicQueryString}`);
  }
  ui.deterministicStateNote.textContent =
    `${parts.join('  ')}  capture=window.gpuViewerDebug.captureFrame(...)`;
}

function updateStaticUiText() {
  ui.timeVal.textContent = Number(ui.timeSlider.value).toFixed(2);
  ui.splatScaleVal.textContent = Number(ui.splatScaleSlider.value).toFixed(2);
  ui.sigmaScaleVal.textContent = Number(ui.sigmaScaleSlider.value).toFixed(2);
  ui.prefilterVarVal.textContent = Number(ui.prefilterVarSlider.value).toFixed(2);
  ui.renderScaleVal.textContent = Number(ui.renderScaleSlider.value).toFixed(2);
  ui.strideVal.textContent = ui.strideSlider.value;
  ui.maxVisibleVal.textContent = ui.maxVisibleSlider.value;
  ui.bgGrayVal.textContent = ui.bgGraySlider.value;
  ui.timeDurationVal.textContent = Number(ui.timeDurationSlider.value).toFixed(1);
}

function buildRenderOverrides() {
  const quality = buildEffectiveGpuQualityConfig({
    ui,
    interactionState,
    isPlaying: playback ? playback.isPlaying() : false
  });

  return {
    ...quality.effectiveConfig,
    enablePackedVisiblePath: !!ui.usePackedVisiblePathCheck?.checked
  };
}

function buildDeterministicStateSummary() {
  const summary = buildViewerDeterministicSummary(deterministicQueryState);
  return {
    ...summary,
    appliedCameraPresetName,
    deterministicQueryString: summary.deterministicQueryString ?? '',
    deterministicUrlSummary: summary.deterministicUrlSummary ?? '',
    deterministicRawQueryString: summary.rawQueryString ?? '',
    snapshotApiAvailable: true,
    snapshotCaptureSource: lastSnapshotSummary.source,
    snapshotRenderWaitMode: lastSnapshotSummary.renderWaitMode,
    snapshotLastStatus: lastSnapshotSummary.status,
    snapshotLastReason: lastSnapshotSummary.reason
  };
}

function getFixedCanvasSizeOverride() {
  const fixedCanvasWidth = Number.isFinite(deterministicQueryState.fixedCanvasWidth)
    ? Math.max(1, deterministicQueryState.fixedCanvasWidth | 0)
    : null;
  const fixedCanvasHeight = Number.isFinite(deterministicQueryState.fixedCanvasHeight)
    ? Math.max(1, deterministicQueryState.fixedCanvasHeight | 0)
    : null;
  if (fixedCanvasWidth === null || fixedCanvasHeight === null) {
    return null;
  }
  return { fixedCanvasWidth, fixedCanvasHeight };
}

function applyCanvasSize() {
  const fixedSize = getFixedCanvasSizeOverride();
  setCanvasSize(fixedSize ?? {});
}

function buildCanvasSizeSummary() {
  const gpu = getGpu();
  const gl = gpu?.gl ?? null;
  const fixedSize = getFixedCanvasSizeOverride();
  const renderScale = Number(ui.renderScaleSlider?.value);
  return {
    clientWidth: Number.isFinite(canvas?.clientWidth) ? canvas.clientWidth : 0,
    clientHeight: Number.isFinite(canvas?.clientHeight) ? canvas.clientHeight : 0,
    windowInnerWidth: typeof window !== 'undefined' ? window.innerWidth : 0,
    windowInnerHeight: typeof window !== 'undefined' ? window.innerHeight : 0,
    canvasWidth: Number.isFinite(canvas?.width) ? canvas.width : 0,
    canvasHeight: Number.isFinite(canvas?.height) ? canvas.height : 0,
    framebufferWidth: gl ? (gl.drawingBufferWidth | 0) : (Number.isFinite(canvas?.width) ? canvas.width : 0),
    framebufferHeight: gl ? (gl.drawingBufferHeight | 0) : (Number.isFinite(canvas?.height) ? canvas.height : 0),
    devicePixelRatio: typeof window !== 'undefined' ? (window.devicePixelRatio || 1) : 1,
    renderScale: Number.isFinite(renderScale) ? renderScale : 1,
    fixedCanvasWidth: fixedSize?.fixedCanvasWidth ?? null,
    fixedCanvasHeight: fixedSize?.fixedCanvasHeight ?? null,
    fixedCanvasActive: !!fixedSize,
    coordinateSpaceForReadPixels: 'webgl-default-framebuffer-pixels',
    probeCoordinateWidth: gl ? (gl.drawingBufferWidth | 0) : (Number.isFinite(canvas?.width) ? canvas.width : 0),
    probeCoordinateHeight: gl ? (gl.drawingBufferHeight | 0) : (Number.isFinite(canvas?.height) ? canvas.height : 0)
  };
}

function parseScreenshotProbeList(rawProbeList) {
  if (rawProbeList === null || rawProbeList === undefined || rawProbeList === '') {
    return [];
  }

  return String(rawProbeList)
    .split(';')
    .map((entry, index) => {
      const trimmed = String(entry ?? '').trim();
      if (!trimmed) return null;

      const labelMatch = trimmed.match(/^([^:@]+)[:@](.+)$/);
      const label = labelMatch ? labelMatch[1].trim() : null;
      const coordsText = (labelMatch ? labelMatch[2] : trimmed).trim();
      const coords = coordsText.split(',').map((value) => value.trim()).filter(Boolean);
      if (coords.length < 2) {
        return {
          valid: false,
          reason: 'invalid-screenshot-probe-format',
          probeId: label ?? `probe-${index + 1}`,
          screenshotProbeX: null,
          screenshotProbeY: null,
          screenshotProbePixel: [null, null]
        };
      }

      const screenshotProbeX = Number(coords[0]);
      const screenshotProbeY = Number(coords[1]);
      const probeId = label ?? `probe-${index + 1}`;
      if (!Number.isFinite(screenshotProbeX) || !Number.isFinite(screenshotProbeY)) {
        return {
          valid: false,
          reason: 'invalid-screenshot-probe-coordinate',
          probeId,
          screenshotProbeX: Number.isFinite(screenshotProbeX) ? screenshotProbeX : null,
          screenshotProbeY: Number.isFinite(screenshotProbeY) ? screenshotProbeY : null,
          screenshotProbePixel: [
            Number.isFinite(screenshotProbeX) ? screenshotProbeX : null,
            Number.isFinite(screenshotProbeY) ? screenshotProbeY : null
          ]
        };
      }

      return {
        valid: true,
        reason: 'probe-list-item-ok',
        probeId,
        screenshotProbeLabel: label ?? probeId,
        screenshotProbeX,
        screenshotProbeY,
        screenshotProbePixel: [screenshotProbeX, screenshotProbeY]
      };
    })
    .filter(Boolean);
}

function buildDefaultScreenshotProbeInput() {
  const screenshotProbeX = Number.isFinite(deterministicQueryState.screenshotProbeX)
    ? Number(deterministicQueryState.screenshotProbeX)
    : null;
  const screenshotProbeY = Number.isFinite(deterministicQueryState.screenshotProbeY)
    ? Number(deterministicQueryState.screenshotProbeY)
    : null;
  if (!Number.isFinite(screenshotProbeX) || !Number.isFinite(screenshotProbeY)) {
    return null;
  }
  return {
    valid: true,
    reason: 'single-screenshot-probe',
    probeId: 'probe-1',
    screenshotProbeLabel: 'probe-1',
    screenshotProbeX,
    screenshotProbeY,
    screenshotProbePixel: [screenshotProbeX, screenshotProbeY]
  };
}

function buildScreenshotProbeInputList() {
  const parsed = parseScreenshotProbeList(deterministicQueryState.screenshotProbeList);
  const screenshotImageWidth = Number.isFinite(deterministicQueryState.screenshotImageWidth)
    ? Math.max(1, deterministicQueryState.screenshotImageWidth | 0)
    : null;
  const screenshotImageHeight = Number.isFinite(deterministicQueryState.screenshotImageHeight)
    ? Math.max(1, deterministicQueryState.screenshotImageHeight | 0)
    : null;
  if (parsed.length > 0) {
    return parsed.map((entry) => ({
      ...entry,
      screenshotImageWidth,
      screenshotImageHeight
    }));
  }
  const single = buildDefaultScreenshotProbeInput();
  return single ? [{
    ...single,
    screenshotImageWidth,
    screenshotImageHeight
  }] : [];
}

function buildMappedScreenshotProbeSummaryFromInput(probeInput, canvasSizeSummary = buildCanvasSizeSummary()) {
  const screenshotProbeX = Number.isFinite(probeInput?.screenshotProbeX)
    ? Number(probeInput.screenshotProbeX)
    : null;
  const screenshotProbeY = Number.isFinite(probeInput?.screenshotProbeY)
    ? Number(probeInput.screenshotProbeY)
    : null;
  const screenshotImageWidth = Number.isFinite(probeInput?.screenshotImageWidth)
    ? Math.max(1, probeInput.screenshotImageWidth | 0)
    : (Number.isFinite(deterministicQueryState.screenshotImageWidth)
      ? Math.max(1, deterministicQueryState.screenshotImageWidth | 0)
      : null);
  const screenshotImageHeight = Number.isFinite(probeInput?.screenshotImageHeight)
    ? Math.max(1, probeInput.screenshotImageHeight | 0)
    : (Number.isFinite(deterministicQueryState.screenshotImageHeight)
      ? Math.max(1, deterministicQueryState.screenshotImageHeight | 0)
      : null);
  const targetWidth = Number.isFinite(canvasSizeSummary?.probeCoordinateWidth)
    ? Math.max(1, canvasSizeSummary.probeCoordinateWidth | 0)
    : 0;
  const targetHeight = Number.isFinite(canvasSizeSummary?.probeCoordinateHeight)
    ? Math.max(1, canvasSizeSummary.probeCoordinateHeight | 0)
    : 0;
  const usesFixedCanvas = !!canvasSizeSummary?.fixedCanvasActive;
  const coordinateSpace = canvasSizeSummary?.coordinateSpaceForReadPixels ?? 'webgl-default-framebuffer-pixels';

  const sourceSummary = {
    source: 'screenshot-query',
    probeId: probeInput?.probeId ?? 'probe-1',
    screenshotProbeLabel: probeInput?.screenshotProbeLabel ?? probeInput?.probeId ?? 'probe-1',
    screenshotProbePixel: [screenshotProbeX, screenshotProbeY],
    screenshotImageSize: [screenshotImageWidth, screenshotImageHeight],
    targetFramebufferSize: [targetWidth, targetHeight],
    scale: null,
    rounding: 'round-to-nearest-pixel',
    clamped: false,
    targetCoordinateSpace: coordinateSpace,
    fixedCanvasTarget: usesFixedCanvas ? [canvasSizeSummary.fixedCanvasWidth, canvasSizeSummary.fixedCanvasHeight] : null
  };

  if (!Number.isFinite(screenshotProbeX) ||
      !Number.isFinite(screenshotProbeY)) {
    return {
      screenshotProbeX: screenshotProbeX ?? null,
      screenshotProbeY: screenshotProbeY ?? null,
      screenshotProbePixel: [screenshotProbeX ?? null, screenshotProbeY ?? null],
      screenshotImageWidth: screenshotImageWidth ?? null,
      screenshotImageHeight: screenshotImageHeight ?? null,
      screenshotImageSize: [screenshotImageWidth ?? null, screenshotImageHeight ?? null],
      probeId: probeInput?.probeId ?? 'probe-1',
      screenshotProbeLabel: probeInput?.screenshotProbeLabel ?? probeInput?.probeId ?? 'probe-1',
      mappedProbeX: null,
      mappedProbeY: null,
      mappedProbeValid: false,
      mappedProbeReason: 'missing-screenshot-probe-coordinate',
      mappedProbeCoordinateSpace: coordinateSpace,
      mappedProbeUsesFixedCanvas: usesFixedCanvas,
      mappedProbeSourceSummary: sourceSummary
    };
  }

  if (!Number.isFinite(screenshotImageWidth) || !Number.isFinite(screenshotImageHeight)) {
    return {
      screenshotProbeX,
      screenshotProbeY,
      screenshotProbePixel: [screenshotProbeX, screenshotProbeY],
      screenshotImageWidth: screenshotImageWidth ?? null,
      screenshotImageHeight: screenshotImageHeight ?? null,
      screenshotImageSize: [screenshotImageWidth ?? null, screenshotImageHeight ?? null],
      probeId: probeInput?.probeId ?? 'probe-1',
      screenshotProbeLabel: probeInput?.screenshotProbeLabel ?? probeInput?.probeId ?? 'probe-1',
      mappedProbeX: null,
      mappedProbeY: null,
      mappedProbeValid: false,
      mappedProbeReason: 'missing-screenshot-image-size',
      mappedProbeCoordinateSpace: coordinateSpace,
      mappedProbeUsesFixedCanvas: usesFixedCanvas,
      mappedProbeSourceSummary: sourceSummary
    };
  }

  if (!(targetWidth > 0) || !(targetHeight > 0)) {
    return {
      screenshotProbeX,
      screenshotProbeY,
      screenshotProbePixel: [screenshotProbeX, screenshotProbeY],
      screenshotImageWidth,
      screenshotImageHeight,
      screenshotImageSize: [screenshotImageWidth, screenshotImageHeight],
      probeId: probeInput?.probeId ?? 'probe-1',
      screenshotProbeLabel: probeInput?.screenshotProbeLabel ?? probeInput?.probeId ?? 'probe-1',
      mappedProbeX: null,
      mappedProbeY: null,
      mappedProbeValid: false,
      mappedProbeReason: 'invalid-target-framebuffer-size',
      mappedProbeCoordinateSpace: coordinateSpace,
      mappedProbeUsesFixedCanvas: usesFixedCanvas,
      mappedProbeSourceSummary: sourceSummary
    };
  }

  const scaleX = targetWidth / screenshotImageWidth;
  const scaleY = targetHeight / screenshotImageHeight;
  const rawMappedX = screenshotProbeX * scaleX;
  const rawMappedY = screenshotProbeY * scaleY;
  const mappedProbeX = Math.min(targetWidth - 1, Math.max(0, Math.round(rawMappedX)));
  const mappedProbeY = Math.min(targetHeight - 1, Math.max(0, Math.round(rawMappedY)));
  const clamped = mappedProbeX !== Math.round(rawMappedX) || mappedProbeY !== Math.round(rawMappedY);

  return {
    screenshotProbeX,
    screenshotProbeY,
    screenshotProbePixel: [screenshotProbeX, screenshotProbeY],
    screenshotImageWidth,
    screenshotImageHeight,
    screenshotImageSize: [screenshotImageWidth, screenshotImageHeight],
    probeId: probeInput?.probeId ?? 'probe-1',
    screenshotProbeLabel: probeInput?.screenshotProbeLabel ?? probeInput?.probeId ?? 'probe-1',
    mappedProbeX,
    mappedProbeY,
    mappedProbeValid: true,
    mappedProbeReason: 'mapped-screenshot-to-viewer-framebuffer',
    mappedProbeCoordinateSpace: coordinateSpace,
    mappedProbeUsesFixedCanvas: usesFixedCanvas,
    mappedProbeSourceSummary: {
      ...sourceSummary,
      scale: [scaleX, scaleY],
      clamped
    }
  };
}

function readFramebufferPatchAtTopLeftPixel(gl, centerPixel, patchSize) {
  const normalizedCenter = normalizeSharedRepresentativePixel(centerPixel);
  const size = Number.isFinite(patchSize) ? Math.max(1, patchSize | 0) : 0;
  if (!normalizedCenter) {
    return {
      valid: false,
      reason: 'invalid-mapped-probe-center',
      patchSize: size,
      centerPixel: [0, 0],
      topLeft: [0, 0],
      bottomRight: [0, 0],
      pixelCount: 0,
      centerRgba8: [0, 0, 0, 0],
      meanRgb: [0, 0, 0],
      minRgb: [0, 0, 0],
      maxRgb: [0, 0, 0],
      rawRgba8: []
    };
  }

  const width = Number.isFinite(canvas?.width) ? (canvas.width | 0) : 0;
  const height = Number.isFinite(canvas?.height) ? (canvas.height | 0) : 0;
  if (!(size > 0) || (size % 2) === 0) {
    return {
      valid: false,
      reason: 'invalid-patch-size',
      patchSize: size,
      centerPixel: normalizedCenter,
      topLeft: [0, 0],
      bottomRight: [0, 0],
      pixelCount: 0,
      centerRgba8: [0, 0, 0, 0],
      meanRgb: [0, 0, 0],
      minRgb: [0, 0, 0],
      maxRgb: [0, 0, 0],
      rawRgba8: []
    };
  }
  if (width <= 0 || height <= 0) {
    return {
      valid: false,
      reason: 'invalid-canvas-size',
      patchSize: size,
      centerPixel: normalizedCenter,
      topLeft: [0, 0],
      bottomRight: [0, 0],
      pixelCount: 0,
      centerRgba8: [0, 0, 0, 0],
      meanRgb: [0, 0, 0],
      minRgb: [0, 0, 0],
      maxRgb: [0, 0, 0],
      rawRgba8: []
    };
  }

  const half = (size - 1) >> 1;
  const topLeftX = normalizedCenter[0] - half;
  const topLeftY = normalizedCenter[1] - half;
  const bottomRightX = topLeftX + size - 1;
  const bottomRightY = topLeftY + size - 1;
  if (topLeftX < 0 || topLeftY < 0 || bottomRightX >= width || bottomRightY >= height) {
    return {
      valid: false,
      reason: 'patch-out-of-bounds',
      patchSize: size,
      centerPixel: normalizedCenter,
      topLeft: [topLeftX, topLeftY],
      bottomRight: [bottomRightX, bottomRightY],
      pixelCount: 0,
      centerRgba8: [0, 0, 0, 0],
      meanRgb: [0, 0, 0],
      minRgb: [0, 0, 0],
      maxRgb: [0, 0, 0],
      rawRgba8: []
    };
  }

  const glX = topLeftX;
  const glY = height - topLeftY - size;
  const rgba = new Uint8Array(size * size * 4);
  const previousFramebuffer = gl.getParameter(gl.FRAMEBUFFER_BINDING);
  try {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.readPixels(glX, glY, size, size, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
  } catch (error) {
    return {
      valid: false,
      reason: `readpixels-failed:${error?.message ?? 'unknown'}`,
      patchSize: size,
      centerPixel: normalizedCenter,
      topLeft: [topLeftX, topLeftY],
      bottomRight: [bottomRightX, bottomRightY],
      pixelCount: 0,
      centerRgba8: [0, 0, 0, 0],
      meanRgb: [0, 0, 0],
      minRgb: [0, 0, 0],
      maxRgb: [0, 0, 0],
      rawRgba8: []
    };
  } finally {
    gl.bindFramebuffer(gl.FRAMEBUFFER, previousFramebuffer);
  }

  const rawRgba8 = new Array(size * size * 4);
  let pixelIndex = 0;
  let minR = 1;
  let minG = 1;
  let minB = 1;
  let maxR = 0;
  let maxG = 0;
  let maxB = 0;
  let sumR = 0;
  let sumG = 0;
  let sumB = 0;
  for (let row = 0; row < size; row += 1) {
    const sourceRow = size - 1 - row;
    for (let col = 0; col < size; col += 1) {
      const srcIndex = (sourceRow * size + col) * 4;
      const r8 = rgba[srcIndex];
      const g8 = rgba[srcIndex + 1];
      const b8 = rgba[srcIndex + 2];
      const a8 = rgba[srcIndex + 3];
      rawRgba8[pixelIndex++] = r8;
      rawRgba8[pixelIndex++] = g8;
      rawRgba8[pixelIndex++] = b8;
      rawRgba8[pixelIndex++] = a8;
      const r = r8 / 255.0;
      const g = g8 / 255.0;
      const b = b8 / 255.0;
      sumR += r;
      sumG += g;
      sumB += b;
      minR = Math.min(minR, r);
      minG = Math.min(minG, g);
      minB = Math.min(minB, b);
      maxR = Math.max(maxR, r);
      maxG = Math.max(maxG, g);
      maxB = Math.max(maxB, b);
    }
  }

  const centerIndex = half * size + half;
  const centerBase = centerIndex * 4;
  return {
    valid: true,
    reason: 'readback-ok',
    patchSize: size,
    centerPixel: normalizedCenter,
    topLeft: [topLeftX, topLeftY],
    bottomRight: [bottomRightX, bottomRightY],
    pixelCount: size * size,
    centerRgba8: [
      rgba[centerBase],
      rgba[centerBase + 1],
      rgba[centerBase + 2],
      rgba[centerBase + 3]
    ],
    meanRgb: [
      sumR / (size * size),
      sumG / (size * size),
      sumB / (size * size)
    ],
    minRgb: [minR, minG, minB],
    maxRgb: [maxR, maxG, maxB],
    rawRgba8
  };
}

function getMappedProbePatchStorageKey(centerPixel, patchSize, probeKey = null) {
  const normalizedCenter = normalizeSharedRepresentativePixel(centerPixel);
  const size = Number.isFinite(patchSize) ? Math.max(1, patchSize | 0) : 0;
  if (!normalizedCenter || !(size > 0)) return null;
  const keyPrefix = probeKey ? `${probeKey}.` : '';
  return `${MAPPED_PROBE_PATCH_REFERENCE_STORAGE_PREFIX}${keyPrefix}${normalizedCenter[0]}x${normalizedCenter[1]}.${size}`;
}

function readMappedProbePatchReference(centerPixel, patchSize, probeKey = null) {
  const storageKey = getMappedProbePatchStorageKey(centerPixel, patchSize, probeKey);
  if (!storageKey) return null;
  const stored = readStoredJson(storageKey);
  if (!stored || !Array.isArray(stored.rawRgba8)) return null;
  return {
    ...stored,
    valid: stored.valid !== false,
    referenceSource: stored.referenceSource ?? 'stored-accumulation-patch-reference'
  };
}

function writeMappedProbePatchReference(centerPixel, patchSize, patchResult, probeKey = null) {
  const storageKey = getMappedProbePatchStorageKey(centerPixel, patchSize, probeKey);
  if (!storageKey || !patchResult?.valid || !Array.isArray(patchResult.rawRgba8)) return;
  writeStoredJson(storageKey, {
    valid: true,
    patchSize: patchResult.patchSize,
    centerPixel: patchResult.centerPixel,
    topLeft: patchResult.topLeft,
    bottomRight: patchResult.bottomRight,
    pixelCount: patchResult.pixelCount,
    rawRgba8: patchResult.rawRgba8,
    referenceSource: 'stored-accumulation-patch-reference'
  });
}

function buildPatchDeltaSummary(actualPatch, referencePatch) {
  if (!actualPatch?.valid) {
    return {
      valid: false,
      reason: actualPatch?.reason ?? 'patch-readback-invalid',
      comparedAgainstAccumulation: false,
      referenceSource: 'none',
      maxDelta: 0,
      meanDelta: 0,
      mismatchCountAboveTolerance: 0,
      worstPixelOffset: [0, 0],
      worstPixelDelta: [0, 0, 0]
    };
  }

  if (!referencePatch || !Array.isArray(referencePatch.rawRgba8)) {
    return {
      valid: false,
      reason: 'accumulation-reference-unavailable',
      comparedAgainstAccumulation: false,
      referenceSource: 'none',
      maxDelta: 0,
      meanDelta: 0,
      mismatchCountAboveTolerance: 0,
      worstPixelOffset: [0, 0],
      worstPixelDelta: [0, 0, 0]
    };
  }

  const actual = actualPatch.rawRgba8;
  const reference = referencePatch.rawRgba8;
  const pixelCount = Math.min(actual.length, reference.length) / 4;
  const size = actualPatch.patchSize;
  const half = (size - 1) >> 1;
  let maxDelta = 0;
  let sumDelta = 0;
  let mismatchCountAboveTolerance = 0;
  let worstPixelOffset = [0, 0];
  let worstPixelDelta = [0, 0, 0];

  for (let i = 0; i < pixelCount; i += 1) {
    const base = i * 4;
    const dr = actual[base] / 255.0 - reference[base] / 255.0;
    const dg = actual[base + 1] / 255.0 - reference[base + 1] / 255.0;
    const db = actual[base + 2] / 255.0 - reference[base + 2] / 255.0;
    const deltaAbsMax = Math.max(Math.abs(dr), Math.abs(dg), Math.abs(db));
    sumDelta += deltaAbsMax;
    if (deltaAbsMax > maxDelta) {
      maxDelta = deltaAbsMax;
      const row = Math.floor(i / size);
      const col = i - row * size;
      worstPixelOffset = [col - half, row - half];
      worstPixelDelta = [dr, dg, db];
    }
    if (deltaAbsMax > MAPPED_PROBE_PATCH_TOLERANCE) {
      mismatchCountAboveTolerance += 1;
    }
  }

  return {
    valid: true,
    reason: 'patch-delta-ok',
    comparedAgainstAccumulation: true,
    referenceSource: referencePatch.referenceSource ?? 'stored-accumulation-patch-reference',
    maxDelta,
    meanDelta: pixelCount > 0 ? sumDelta / pixelCount : 0,
    mismatchCountAboveTolerance,
    worstPixelOffset,
    worstPixelDelta
  };
}

function buildMappedProbePatchSummary(renderResultSummary, canvasSizeSummary = buildCanvasSizeSummary(), mappedProbeSummary = buildMappedScreenshotProbeSummary(canvasSizeSummary), probeKey = null) {
  const gl = getGpu()?.gl ?? null;
  const tileCompositePath = renderResultSummary?.drawThroughputSummary?.tileCompositePath ?? 'none';
  const mappedProbeX = Number.isFinite(mappedProbeSummary?.mappedProbeX)
    ? Number(mappedProbeSummary.mappedProbeX)
    : null;
  const mappedProbeY = Number.isFinite(mappedProbeSummary?.mappedProbeY)
    ? Number(mappedProbeSummary.mappedProbeY)
    : null;
  const centerPixel = mappedProbeSummary?.mappedProbeValid
    ? (normalizeSharedRepresentativePixel([mappedProbeX, mappedProbeY]) ?? [0, 0])
    : null;

  if (!mappedProbeSummary?.mappedProbeValid || !centerPixel) {
    const patchSizes = [...MAPPED_PROBE_PATCH_SIZES];
    const summaries = {};
    const deltaSummaries = {};
    for (const size of patchSizes) {
      const key = `${size}x${size}`;
      summaries[key] = {
        patchSize: [size, size],
        valid: false,
        reason: mappedProbeSummary?.mappedProbeReason ?? 'mapped-probe-invalid',
        centerPixel: centerPixel ?? [0, 0],
        topLeft: [0, 0],
        bottomRight: [0, 0],
        pixelCount: 0,
        rgba8: [0, 0, 0, 0],
        meanRgb: [0, 0, 0],
        minRgb: [0, 0, 0],
        maxRgb: [0, 0, 0],
        referenceSource: 'none'
      };
      deltaSummaries[key] = {
        valid: false,
        reason: mappedProbeSummary?.mappedProbeReason ?? 'mapped-probe-invalid',
        referenceSource: 'none',
        comparedAgainstAccumulation: false,
        maxDelta: 0,
        meanDelta: 0,
        mismatchCountAboveTolerance: 0,
        worstPixelOffset: [0, 0],
        worstPixelDelta: [0, 0, 0]
      };
    }
    return {
      mappedProbePatchSizes: patchSizes,
      mappedProbePatchReadbackValid: false,
      mappedProbePatchReadbackReason: mappedProbeSummary?.mappedProbeReason ?? 'mapped-probe-invalid',
      mappedProbePatchCenter: centerPixel ?? [0, 0],
      mappedProbePatchSummaries: summaries,
      mappedProbePatchComparedAgainstAccumulation: false,
      mappedProbePatchDeltaSummaries: deltaSummaries
    };
  }

  const patchSizes = [...MAPPED_PROBE_PATCH_SIZES];
  const summaries = {};
  const deltaSummaries = {};
  let allValid = true;
  let firstFailureReason = 'patch-readback-ok';
  let comparedAgainstAccumulation = false;

  for (const size of patchSizes) {
    const key = `${size}x${size}`;
    const actualPatch = gl ? readFramebufferPatchAtTopLeftPixel(gl, centerPixel, size) : {
      valid: false,
      reason: 'webgl-unavailable',
      patchSize: size,
      centerPixel,
      topLeft: [0, 0],
      bottomRight: [0, 0],
      pixelCount: 0,
      centerRgba8: [0, 0, 0, 0],
      meanRgb: [0, 0, 0],
      minRgb: [0, 0, 0],
      maxRgb: [0, 0, 0],
      rawRgba8: []
    };
    if (!actualPatch.valid) {
      allValid = false;
      if (firstFailureReason === 'patch-readback-ok') {
        firstFailureReason = actualPatch.reason;
      }
    }

    let referencePatch = readMappedProbePatchReference(centerPixel, size, probeKey);
    let referenceSource = referencePatch ? 'stored-accumulation-patch-reference' : 'none';
    if (!referencePatch && tileCompositePath === 'accumulation' && actualPatch.valid) {
      referencePatch = {
        valid: true,
        rawRgba8: actualPatch.rawRgba8,
        referenceSource: 'current-accumulation-patch-reference'
      };
      referenceSource = 'current-accumulation-patch-reference';
      writeMappedProbePatchReference(centerPixel, size, actualPatch, probeKey);
    }

    const delta = buildPatchDeltaSummary(actualPatch, referencePatch);
    if (delta.comparedAgainstAccumulation) {
      comparedAgainstAccumulation = true;
    }
    if (tileCompositePath === 'accumulation' && actualPatch.valid) {
      writeMappedProbePatchReference(centerPixel, size, actualPatch);
    }

    summaries[key] = {
      patchSize: [size, size],
      valid: !!actualPatch.valid,
      reason: actualPatch.reason ?? 'none',
      centerPixel: actualPatch.centerPixel ?? centerPixel,
      topLeft: actualPatch.topLeft ?? [0, 0],
      bottomRight: actualPatch.bottomRight ?? [0, 0],
      pixelCount: actualPatch.pixelCount ?? 0,
      rgba8: actualPatch.centerRgba8 ?? [0, 0, 0, 0],
      meanRgb: actualPatch.meanRgb ?? [0, 0, 0],
      minRgb: actualPatch.minRgb ?? [0, 0, 0],
      maxRgb: actualPatch.maxRgb ?? [0, 0, 0],
      referenceSource
    };

    deltaSummaries[key] = {
      valid: delta.valid,
      reason: delta.reason,
      referenceSource: delta.referenceSource,
      comparedAgainstAccumulation: delta.comparedAgainstAccumulation,
      maxDelta: delta.maxDelta,
      meanDelta: delta.meanDelta,
      mismatchCountAboveTolerance: delta.mismatchCountAboveTolerance,
      worstPixelOffset: delta.worstPixelOffset,
      worstPixelDelta: delta.worstPixelDelta
    };
  }

  if (tileCompositePath === 'accumulation' && allValid) {
    for (const size of patchSizes) {
      const key = `${size}x${size}`;
      const actualPatch = summaries[key];
      if (actualPatch?.valid) {
        writeMappedProbePatchReference(centerPixel, size, {
          valid: true,
          patchSize: actualPatch.patchSize?.[0] ?? size,
          centerPixel: actualPatch.centerPixel,
          topLeft: actualPatch.topLeft,
          bottomRight: actualPatch.bottomRight,
          pixelCount: actualPatch.pixelCount,
          rawRgba8: readFramebufferPatchAtTopLeftPixel(
            gl,
            centerPixel,
            size
          ).rawRgba8
        }, probeKey);
      }
    }
  }

  return {
    mappedProbePatchSizes: patchSizes,
    mappedProbePatchReadbackValid: allValid,
    mappedProbePatchReadbackReason: firstFailureReason,
    mappedProbePatchCenter: centerPixel,
    mappedProbePatchSummaries: summaries,
    mappedProbePatchComparedAgainstAccumulation: comparedAgainstAccumulation,
    mappedProbePatchDeltaSummaries: deltaSummaries
  };
}

function buildMappedProbeCollectionsSummary(renderResultSummary, canvasSizeSummary = buildCanvasSizeSummary()) {
  const probeInputs = buildScreenshotProbeInputList();
  const useProbeKeyPrefix = !!deterministicQueryState.screenshotProbeList;
  const fallbackProbeInputs = probeInputs.length > 0
    ? probeInputs
    : [{
      valid: false,
      reason: 'missing-screenshot-probe-coordinate',
      probeId: 'probe-1',
      screenshotProbeLabel: 'probe-1',
      screenshotProbeX: null,
      screenshotProbeY: null,
      screenshotProbePixel: [null, null]
    }];

  const mappedProbeList = fallbackProbeInputs.map((probeInput, index) => {
    const mappedProbeSummary = buildMappedScreenshotProbeSummaryFromInput(probeInput, canvasSizeSummary);
    const probeKey = useProbeKeyPrefix
      ? (mappedProbeSummary?.probeId ?? probeInput?.probeId ?? `probe-${index + 1}`)
      : null;
    const mappedProbePatchSummary = buildMappedProbePatchSummary(
      renderResultSummary,
      canvasSizeSummary,
      mappedProbeSummary,
      probeKey
    );
    return {
      ...mappedProbeSummary,
      ...mappedProbePatchSummary
    };
  });
  const primaryMappedProbe = mappedProbeList[0] ?? null;

  return {
    screenshotProbeList: fallbackProbeInputs,
    mappedProbeList,
    collectionSummary: {
      source: deterministicQueryState.screenshotProbeList ? 'explicit-screenshotProbeList' : 'single-screenshotProbe',
      probeCount: mappedProbeList.length,
      probeIds: mappedProbeList.map((probe) => probe?.probeId ?? 'probe-unknown'),
      probeLabels: mappedProbeList.map((probe) => probe?.screenshotProbeLabel ?? probe?.probeId ?? 'probe-unknown'),
      primaryProbeId: primaryMappedProbe?.probeId ?? 'probe-1',
      primaryProbeLabel: primaryMappedProbe?.screenshotProbeLabel ?? primaryMappedProbe?.probeId ?? 'probe-1',
      primaryProbeSource: primaryMappedProbe?.mappedProbeSourceSummary?.source ?? 'screenshot-query',
      primaryProbeValid: !!primaryMappedProbe?.mappedProbeValid,
      usesFixedCanvas: !!canvasSizeSummary?.fixedCanvasActive,
      coordinateSpaceForReadPixels: canvasSizeSummary?.coordinateSpaceForReadPixels ?? 'webgl-default-framebuffer-pixels'
    },
    firstMappedProbeSummary: primaryMappedProbe,
    firstMappedProbePatchSummary: primaryMappedProbe
  };
}

function normalizeInspectSource(value, fallback = 'auto') {
  return INSPECT_SOURCE_VALUES.has(value) ? value : fallback;
}

function normalizeInspectJsonMode(value, fallback = 'slim') {
  return INSPECT_JSON_MODE_VALUES.has(value) ? value : fallback;
}

function getRequestedTileCompositePrimitive() {
  return ui.tileCompositePrimitiveSelect?.value === 'quad' ? 'quad' : 'point';
}

function getRequestedTileCompositePath() {
  return ui.tileCompositePathSelect?.value === 'accumulation' ? 'accumulation' : 'baseline';
}

function parseNumberTuple(value, expectedLength) {
  if (value === null || value === undefined || value === '') return null;
  const parts = String(value).split(',').map((part) => Number(part.trim()));
  if (parts.length !== expectedLength || parts.some((part) => !Number.isFinite(part))) return null;
  return parts;
}

function getQueryNumberTuple(name, expectedLength) {
  if (typeof window === 'undefined') return null;
  return parseNumberTuple(new URLSearchParams(window.location.search || '').get(name), expectedLength);
}

function readStoredNumberTuple(key, expectedLength) {
  if (typeof window === 'undefined' || !window.localStorage) return null;
  try {
    return parseNumberTuple(window.localStorage.getItem(key), expectedLength);
  } catch {
    return null;
  }
}

function writeStoredNumberTuple(key, values) {
  if (typeof window === 'undefined' || !window.localStorage || !Array.isArray(values)) return;
  try {
    window.localStorage.setItem(key, values.join(','));
  } catch {
    // Storage is best-effort diagnostics only.
  }
}

function readStoredJson(key) {
  if (typeof window === 'undefined' || !window.localStorage) return null;
  try {
    const value = window.localStorage.getItem(key);
    if (!value) return null;
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function writeStoredJson(key, value) {
  if (typeof window === 'undefined' || !window.localStorage) return;
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage is best-effort diagnostics only.
  }
}

function normalizeSharedRepresentativePixel(pixel) {
  if (!Array.isArray(pixel) || pixel.length < 2) return null;
  const x = Number.isFinite(pixel[0]) ? Math.floor(pixel[0]) : -1;
  const y = Number.isFinite(pixel[1]) ? Math.floor(pixel[1]) : -1;
  return x >= 0 && y >= 0 ? [x, y] : null;
}

function clampColor01ForSummary(rgb) {
  const safe = Array.isArray(rgb) ? rgb : [0, 0, 0];
  return [
    Number.isFinite(safe[0]) ? Number(safe[0]) : 0,
    Number.isFinite(safe[1]) ? Number(safe[1]) : 0,
    Number.isFinite(safe[2]) ? Number(safe[2]) : 0
  ];
}

function buildColorDeltaForSummary(referenceColor, color) {
  const reference = clampColor01ForSummary(referenceColor);
  const actual = clampColor01ForSummary(color);
  const delta = [
    actual[0] - reference[0],
    actual[1] - reference[1],
    actual[2] - reference[2]
  ];
  return {
    delta,
    deltaAbsMax: Math.max(Math.abs(delta[0]), Math.abs(delta[1]), Math.abs(delta[2]))
  };
}

function readFramebufferColorAtTopLeftPixel(gl, pixel) {
  const normalizedPixel = normalizeSharedRepresentativePixel(pixel);
  if (!normalizedPixel) {
    return {
      color: [0, 0, 0],
      rgba8: [0, 0, 0, 0],
      valid: false,
      reason: 'invalid-shared-representative-pixel',
      pixel: [0, 0],
      glPixel: [0, 0]
    };
  }
  const width = Number.isFinite(canvas?.width) ? (canvas.width | 0) : 0;
  const height = Number.isFinite(canvas?.height) ? (canvas.height | 0) : 0;
  const [x, yTop] = normalizedPixel;
  if (width <= 0 || height <= 0) {
    return {
      color: [0, 0, 0],
      rgba8: [0, 0, 0, 0],
      valid: false,
      reason: 'invalid-canvas-size',
      pixel: normalizedPixel,
      glPixel: [0, 0]
    };
  }
  if (x >= width || yTop >= height) {
    return {
      color: [0, 0, 0],
      rgba8: [0, 0, 0, 0],
      valid: false,
      reason: 'shared-representative-pixel-out-of-bounds',
      pixel: normalizedPixel,
      glPixel: [0, 0]
    };
  }

  const yGl = height - 1 - yTop;
  const rgba = new Uint8Array(4);
  const previousFramebuffer = gl.getParameter(gl.FRAMEBUFFER_BINDING);
  try {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.readPixels(x, yGl, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
  } catch (error) {
    return {
      color: [0, 0, 0],
      rgba8: Array.from(rgba),
      valid: false,
      reason: `readpixels-failed:${error?.message ?? 'unknown'}`,
      pixel: normalizedPixel,
      glPixel: [x, yGl]
    };
  } finally {
    gl.bindFramebuffer(gl.FRAMEBUFFER, previousFramebuffer);
  }

  return {
    color: [rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0],
    rgba8: Array.from(rgba),
    valid: true,
    reason: 'readback-ok',
    pixel: normalizedPixel,
    glPixel: [x, yGl]
  };
}

function resolveSharedRepresentativePixel(executionSummary) {
  const queryPixel = normalizeSharedRepresentativePixel(getQueryNumberTuple('sharedRepresentativePixel', 2));
  if (queryPixel) return { pixel: queryPixel, source: 'query-sharedRepresentativePixel' };

  const accumulationPixel = normalizeSharedRepresentativePixel(
    executionSummary?.tileAccumulationRepresentativeSamplePixel
  );
  if (accumulationPixel) {
    writeStoredNumberTuple(SHARED_REPRESENTATIVE_PIXEL_STORAGE_KEY, accumulationPixel);
    return { pixel: accumulationPixel, source: 'accumulation-representative-sample' };
  }

  const storedPixel = normalizeSharedRepresentativePixel(
    readStoredNumberTuple(SHARED_REPRESENTATIVE_PIXEL_STORAGE_KEY, 2)
  );
  if (storedPixel) return { pixel: storedPixel, source: 'stored-accumulation-representative-sample' };

  return {
    pixel: SHARED_REPRESENTATIVE_DEFAULT_PIXEL,
    source: 'step86-default-representative-pixel'
  };
}

function resolveSharedRepresentativeAccumulationReference(executionSummary, sharedPixel) {
  const currentColor = executionSummary?.tileAccumulationRepresentativeSampleFramebufferReadbackValid
    ? clampColor01ForSummary(executionSummary.tileAccumulationRepresentativeSampleFramebufferColor)
    : null;
  const currentPixel = normalizeSharedRepresentativePixel(
    executionSummary?.tileAccumulationRepresentativeSampleFramebufferReadbackPixel ??
    executionSummary?.tileAccumulationRepresentativeSamplePixel
  );
  if (currentColor && currentPixel && currentPixel[0] === sharedPixel[0] && currentPixel[1] === sharedPixel[1]) {
    writeStoredNumberTuple(SHARED_REPRESENTATIVE_ACCUMULATION_COLOR_STORAGE_KEY, currentColor);
    return {
      color: currentColor,
      source: 'current-accumulation-framebuffer-readback',
      pixel: currentPixel
    };
  }

  const queryColor = getQueryNumberTuple('sharedRepresentativeAccumulationColor', 3);
  if (queryColor) {
    return {
      color: clampColor01ForSummary(queryColor),
      source: 'query-sharedRepresentativeAccumulationColor',
      pixel: sharedPixel
    };
  }

  const storedColor = readStoredNumberTuple(SHARED_REPRESENTATIVE_ACCUMULATION_COLOR_STORAGE_KEY, 3);
  if (storedColor) {
    return {
      color: clampColor01ForSummary(storedColor),
      source: 'stored-accumulation-framebuffer-readback',
      pixel: sharedPixel
    };
  }

  return {
    color: null,
    source: 'accumulation-reference-unavailable',
    pixel: sharedPixel
  };
}

function buildSharedRepresentativeFramebufferProbe(renderResultSummary) {
  const gpu = getGpu();
  const gl = gpu?.gl;
  const executionSummary = renderResultSummary?.executionSummary ?? null;
  const { pixel, source } = resolveSharedRepresentativePixel(executionSummary);
  if (!gl) {
    return {
      sharedRepresentativePixel: pixel,
      sharedRepresentativePixelSource: source,
      sharedRepresentativeFramebufferColor: [0, 0, 0],
      sharedRepresentativeFramebufferReadbackValid: false,
      sharedRepresentativeFramebufferReadbackReason: 'webgl-unavailable',
      sharedRepresentativeFramebufferRgba8: [0, 0, 0, 0],
      sharedRepresentativeComparedAgainstAccumulationPixel: false,
      sharedRepresentativeAccumulationReferenceColor: [0, 0, 0],
      sharedRepresentativeAccumulationReferenceSource: 'accumulation-reference-unavailable',
      sharedRepresentativeColorDeltaVsAccumulation: [0, 0, 0],
      sharedRepresentativeColorDeltaVsAccumulationAbsMax: 0,
      sharedRepresentativeColorMatchesAccumulation: false,
      sharedRepresentativeColorMatchTolerance: SHARED_REPRESENTATIVE_COLOR_MATCH_TOLERANCE
    };
  }

  const readback = readFramebufferColorAtTopLeftPixel(gl, pixel);
  const reference = resolveSharedRepresentativeAccumulationReference(executionSummary, readback.pixel);
  const hasReference = Array.isArray(reference.color);
  const { delta, deltaAbsMax } = hasReference
    ? buildColorDeltaForSummary(reference.color, readback.color)
    : { delta: [0, 0, 0], deltaAbsMax: 0 };
  const comparedAgainstAccumulationPixel = hasReference &&
    Array.isArray(reference.pixel) &&
    reference.pixel[0] === readback.pixel[0] &&
    reference.pixel[1] === readback.pixel[1];

  return {
    sharedRepresentativePixel: readback.pixel,
    sharedRepresentativePixelSource: source,
    sharedRepresentativeFramebufferColor: readback.color,
    sharedRepresentativeFramebufferReadbackValid: readback.valid,
    sharedRepresentativeFramebufferReadbackReason: readback.reason,
    sharedRepresentativeFramebufferRgba8: readback.rgba8,
    sharedRepresentativeFramebufferReadbackGlPixel: readback.glPixel,
    sharedRepresentativeComparedAgainstAccumulationPixel: comparedAgainstAccumulationPixel,
    sharedRepresentativeAccumulationReferenceColor: hasReference ? reference.color : [0, 0, 0],
    sharedRepresentativeAccumulationReferenceSource: reference.source,
    sharedRepresentativeColorDeltaVsAccumulation: delta,
    sharedRepresentativeColorDeltaVsAccumulationAbsMax: deltaAbsMax,
    sharedRepresentativeColorMatchesAccumulation: readback.valid &&
      comparedAgainstAccumulationPixel &&
      deltaAbsMax <= SHARED_REPRESENTATIVE_COLOR_MATCH_TOLERANCE,
    sharedRepresentativeColorMatchTolerance: SHARED_REPRESENTATIVE_COLOR_MATCH_TOLERANCE
  };
}

function buildSlimDeterministicStateSummary(summary) {
  return {
    active: !!summary?.active,
    cameraPresetName: summary?.cameraPresetName ?? 'none',
    appliedCameraPresetName: summary?.appliedCameraPresetName ?? 'none',
    drawPath: summary?.drawPath ?? 'none',
    tileCompositePath: summary?.tileCompositePath ?? 'baseline',
    tileCompositePrimitive: summary?.tileCompositePrimitive ?? 'point',
    inspectSource: summary?.inspectSource ?? 'auto',
    inspectJsonMode: summary?.inspectJsonMode ?? 'slim',
    gpuFramePolicyOverride: summary?.gpuFramePolicyOverride ?? 'auto',
    time: Number.isFinite(summary?.time) ? Number(summary.time) : null,
    deterministicQueryString: summary?.deterministicQueryString ?? '',
    deterministicUrlSummary: summary?.deterministicUrlSummary ?? ''
  };
}

function buildRenderResultInspectionSummary(renderResult) {
  const executionSummary = renderResult?.executionSummary ?? null;
  const tileCompositeSummary = renderResult?.tileCompositePlan?.summary ?? null;
  const summary = {
    canvasSizeSummary: buildCanvasSizeSummary(),
    actualDrawPath:
      renderResult?.drawThroughputSummary?.actualDrawPath ??
      renderResult?.drawPathSummary?.actualPath ??
      'none',
    drawPathSummary: renderResult?.drawPathSummary ?? null,
    drawThroughputSummary: renderResult?.drawThroughputSummary ?? null,
    gpuFallbackSummary: renderResult?.gpuFallbackSummary ?? null,
    gpuCompatibilityBridgeSummary: renderResult?.gpuCompatibilityBridgeSummary ?? null,
    executionSummary: executionSummary
      ? {
          requestedDrawPath: executionSummary.requestedDrawPath ?? 'none',
          actualDrawPath: executionSummary.actualDrawPath ?? 'none',
          drawPathFallbackReason: executionSummary.drawPathFallbackReason ?? 'none',
          compositingContract: executionSummary.compositingContract ?? 'none',
          tileCompositePath: executionSummary.tileCompositePath ?? 'none',
          tileCompositePrimitive: executionSummary.tileCompositePrimitive ?? 'none',
          tileCompositeRectContract: executionSummary.tileCompositeRectContract ?? 'none',
          tileBatchCount: executionSummary.tileBatchCount ?? 0,
          nonEmptyTileBatchCount: executionSummary.nonEmptyTileBatchCount ?? 0,
          totalTileDrawCount: executionSummary.totalTileDrawCount ?? 0,
          drawCallCount: executionSummary.drawCallCount ?? 0,
          uploadCount: executionSummary.uploadCount ?? 0,
          requestedTextureWidth: executionSummary.requestedTextureWidth ?? 0,
          requestedTextureHeight: executionSummary.requestedTextureHeight ?? 0,
          validatedTextureWidth: executionSummary.validatedTextureWidth ?? 0,
          validatedTextureHeight: executionSummary.validatedTextureHeight ?? 0,
          textureAllocationValid: !!executionSummary.textureAllocationValid,
          textureAllocationFailureReason: executionSummary.textureAllocationFailureReason ?? 'none',
          accumulationMaxItemsPerTile: executionSummary.accumulationMaxItemsPerTile ?? 0,
          accumulationTruncatedBatchCount: executionSummary.accumulationTruncatedBatchCount ?? 0,
          tileAccumulationTruncatedTileCount: executionSummary.tileAccumulationTruncatedTileCount ?? 0,
          tileAccumulationMaxObservedTileItems: executionSummary.tileAccumulationMaxObservedTileItems ?? 0,
          tileAccumulationTotalSkippedItems: executionSummary.tileAccumulationTotalSkippedItems ?? 0,
          tileAccumulationWorstTileId: executionSummary.tileAccumulationWorstTileId ?? -1,
          tileAccumulationWorstTileItemCount: executionSummary.tileAccumulationWorstTileItemCount ?? 0,
          tileAccumulationWorstTileSkippedCount: executionSummary.tileAccumulationWorstTileSkippedCount ?? 0,
          tileAccumulationEarlyOutEnabled: !!executionSummary.tileAccumulationEarlyOutEnabled,
          tileAccumulationEarlyOutThreshold: Number.isFinite(executionSummary.tileAccumulationEarlyOutThreshold)
            ? Number(executionSummary.tileAccumulationEarlyOutThreshold)
            : 0,
          tileAccumulationEarlyOutTriggeredTileCount: executionSummary.tileAccumulationEarlyOutTriggeredTileCount ?? 0,
          tileAccumulationEarlyOutTriggeredPixelEstimate: executionSummary.tileAccumulationEarlyOutTriggeredPixelEstimate ?? 0,
          tileAccumulationWorstEarlyOutTileId: executionSummary.tileAccumulationWorstEarlyOutTileId ?? -1,
          tileAccumulationWorstEarlyOutCount: executionSummary.tileAccumulationWorstEarlyOutCount ?? 0,
          tileAccumulationAverageVisitedItemsPerTile: Number.isFinite(executionSummary.tileAccumulationAverageVisitedItemsPerTile)
            ? Number(executionSummary.tileAccumulationAverageVisitedItemsPerTile)
            : 0,
          tileAccumulationMaxVisitedItemsPerTile: executionSummary.tileAccumulationMaxVisitedItemsPerTile ?? 0,
          tileAccumulationAverageVisitedItemsPerPixelEstimate: Number.isFinite(executionSummary.tileAccumulationAverageVisitedItemsPerPixelEstimate)
            ? Number(executionSummary.tileAccumulationAverageVisitedItemsPerPixelEstimate)
            : 0,
          tileAccumulationVisitedRatioSummary: executionSummary.tileAccumulationVisitedRatioSummary ?? null,
          tileAccumulationObservedTileSummaries: executionSummary.tileAccumulationObservedTileSummaries ?? [],
          tileAccumulationOrderingSummary: executionSummary.tileAccumulationOrderingSummary ?? null,
          tileAccumulationBatchBoundarySummary: executionSummary.tileAccumulationBatchBoundarySummary ?? null,
          tileAccumulationObservedOrderingMismatches: executionSummary.tileAccumulationObservedOrderingMismatches ?? [],
          tileAccumulationHeavyTileSummaries: executionSummary.tileAccumulationHeavyTileSummaries ?? [],
          tileAccumulationRepresentativeTileId: executionSummary.tileAccumulationRepresentativeTileId ?? -1,
          tileAccumulationRepresentativeTileItemCount: executionSummary.tileAccumulationRepresentativeTileItemCount ?? 0,
          tileAccumulationRepresentativeTileOrderPreview: executionSummary.tileAccumulationRepresentativeTileOrderPreview ?? null,
          tileAccumulationRepresentativeTileDepthPreview: executionSummary.tileAccumulationRepresentativeTileDepthPreview ?? null,
          tileAccumulationRepresentativeTileBatchSpan: executionSummary.tileAccumulationRepresentativeTileBatchSpan ?? 1,
          tileAccumulationRepresentativeTileSequenceConsistent: !!executionSummary.tileAccumulationRepresentativeTileSequenceConsistent,
          tileAccumulationContributionSummary: executionSummary.tileAccumulationContributionSummary ?? null,
          tileAccumulationRepresentativeSampleMode: executionSummary.tileAccumulationRepresentativeSampleMode ?? 'none',
          tileAccumulationRepresentativeSampleSelectionMode: executionSummary.tileAccumulationRepresentativeSampleSelectionMode ?? 'none',
          tileAccumulationRepresentativeSampleSelectionReason: executionSummary.tileAccumulationRepresentativeSampleSelectionReason ?? 'none',
          tileAccumulationRepresentativeSamplePixel: executionSummary.tileAccumulationRepresentativeSamplePixel ?? [0, 0],
          tileAccumulationRepresentativeSampleHasContribution: !!executionSummary.tileAccumulationRepresentativeSampleHasContribution,
          tileAccumulationRepresentativeSampleCandidateCount: executionSummary.tileAccumulationRepresentativeSampleCandidateCount ?? 0,
          tileAccumulationRepresentativeSampleEvaluatedCandidateCount: executionSummary.tileAccumulationRepresentativeSampleEvaluatedCandidateCount ?? 0,
          tileAccumulationRepresentativeSampleUsableItemSource: executionSummary.tileAccumulationRepresentativeSampleUsableItemSource ?? 'none',
          tileAccumulationRepresentativeSampleItemReadMode: executionSummary.tileAccumulationRepresentativeSampleItemReadMode ?? 'none',
          tileAccumulationRepresentativeSampleEvaluatedItemCount: executionSummary.tileAccumulationRepresentativeSampleEvaluatedItemCount ?? 0,
          tileAccumulationRepresentativeSampleContributionLog: executionSummary.tileAccumulationRepresentativeSampleContributionLog ?? [],
          tileAccumulationRepresentativeSampleFinalT: Number.isFinite(executionSummary.tileAccumulationRepresentativeSampleFinalT)
            ? Number(executionSummary.tileAccumulationRepresentativeSampleFinalT)
            : 1,
          tileAccumulationRepresentativeSampleAccumColor: executionSummary.tileAccumulationRepresentativeSampleAccumColor ?? [0, 0, 0],
          tileAccumulationRepresentativeSampleResolvedColor: executionSummary.tileAccumulationRepresentativeSampleResolvedColor ?? [0, 0, 0],
          tileAccumulationRepresentativeSampleContributionCount: executionSummary.tileAccumulationRepresentativeSampleContributionCount ?? 0,
          tileAccumulationRepresentativeSampleAlphaSum: Number.isFinite(executionSummary.tileAccumulationRepresentativeSampleAlphaSum)
            ? Number(executionSummary.tileAccumulationRepresentativeSampleAlphaSum)
            : 0,
          tileAccumulationRepresentativeSampleContributionSum: executionSummary.tileAccumulationRepresentativeSampleContributionSum ?? [0, 0, 0],
          tileAccumulationRepresentativeSampleLastContributedLocalOrder: executionSummary.tileAccumulationRepresentativeSampleLastContributedLocalOrder ?? -1,
          tileAccumulationRepresentativeSampleThresholdCrossingCount: executionSummary.tileAccumulationRepresentativeSampleThresholdCrossingCount ?? 0,
          tileAccumulationRepresentativeSampleThresholdSkippedCount: executionSummary.tileAccumulationRepresentativeSampleThresholdSkippedCount ?? 0,
          tileAccumulationRepresentativeSampleFirstThresholdSkipLocalOrder: executionSummary.tileAccumulationRepresentativeSampleFirstThresholdSkipLocalOrder ?? -1,
          tileAccumulationRepresentativeSampleFirstThresholdSkipAlpha: Number.isFinite(executionSummary.tileAccumulationRepresentativeSampleFirstThresholdSkipAlpha)
            ? Number(executionSummary.tileAccumulationRepresentativeSampleFirstThresholdSkipAlpha)
            : 0,
          tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceBefore: Number.isFinite(executionSummary.tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceBefore)
            ? Number(executionSummary.tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceBefore)
            : 1,
          tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceAfter: Number.isFinite(executionSummary.tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceAfter)
            ? Number(executionSummary.tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceAfter)
            : 1,
          tileAccumulationRepresentativeSampleThresholdSkipPreview: executionSummary.tileAccumulationRepresentativeSampleThresholdSkipPreview ?? [],
          tileAccumulationRepresentativeSampleThresholdSemantics: executionSummary.tileAccumulationRepresentativeSampleThresholdSemantics ?? null,
          tileAccumulationRepresentativeSampleFramebufferColor: executionSummary.tileAccumulationRepresentativeSampleFramebufferColor ?? [0, 0, 0],
          tileAccumulationRepresentativeSampleFramebufferReadbackValid: !!executionSummary.tileAccumulationRepresentativeSampleFramebufferReadbackValid,
          tileAccumulationRepresentativeSampleFramebufferReadbackReason: executionSummary.tileAccumulationRepresentativeSampleFramebufferReadbackReason ?? 'not-attempted',
          tileAccumulationRepresentativeSampleFramebufferReadbackPixel: executionSummary.tileAccumulationRepresentativeSampleFramebufferReadbackPixel ?? [0, 0],
          tileAccumulationRepresentativeSampleFramebufferReadbackGlPixel: executionSummary.tileAccumulationRepresentativeSampleFramebufferReadbackGlPixel ?? [0, 0],
          tileAccumulationRepresentativeSampleFramebufferReadbackRgba8: executionSummary.tileAccumulationRepresentativeSampleFramebufferReadbackRgba8 ?? [0, 0, 0, 0],
          tileAccumulationRepresentativeSampleResolvedColorDelta: executionSummary.tileAccumulationRepresentativeSampleResolvedColorDelta ?? [0, 0, 0],
          tileAccumulationRepresentativeSampleResolvedColorDeltaAbsMax: Number.isFinite(executionSummary.tileAccumulationRepresentativeSampleResolvedColorDeltaAbsMax)
            ? Number(executionSummary.tileAccumulationRepresentativeSampleResolvedColorDeltaAbsMax)
            : 0,
          tileAccumulationRepresentativeSampleResolvedColorMatchesFramebuffer: !!executionSummary.tileAccumulationRepresentativeSampleResolvedColorMatchesFramebuffer,
          tileAccumulationRepresentativeSampleResolvedColorMatchTolerance: Number.isFinite(executionSummary.tileAccumulationRepresentativeSampleResolvedColorMatchTolerance)
            ? Number(executionSummary.tileAccumulationRepresentativeSampleResolvedColorMatchTolerance)
            : 0,
          tileAccumulationContractVersion: executionSummary.tileAccumulationContractVersion ?? 'none',
          tileAccumulationTruncationRatio: Number.isFinite(executionSummary.tileAccumulationTruncationRatio)
            ? Number(executionSummary.tileAccumulationTruncationRatio)
            : 0
        }
      : null,
    tileCompositeSummary: tileCompositeSummary
      ? {
          compositingContract: tileCompositeSummary.compositingContract ?? 'none',
          depthOrder: tileCompositeSummary.depthOrder ?? 'none',
          tileBatchCount: tileCompositeSummary.tileBatchCount ?? 0,
          nonEmptyTileBatchCount: tileCompositeSummary.nonEmptyTileBatchCount ?? 0,
          totalTileDrawCount: tileCompositeSummary.totalTileDrawCount ?? 0,
          maxTileDrawCount: tileCompositeSummary.maxTileDrawCount ?? 0,
          tileCompositeDuplicateRefs: tileCompositeSummary.tileCompositeDuplicateRefs ?? 0,
          tileCompositeOverlapFactor: tileCompositeSummary.tileCompositeOverlapFactor ?? 0
        }
      : null,
    packedSummary: renderResult?.packedSummary
      ? {
          path: renderResult.packedSummary.path ?? 'none',
          packedContract: renderResult.packedSummary.packedContract ?? 'none',
          transformPath: renderResult.packedSummary.transformPath ?? 'none',
          transformFallbackReason: renderResult.packedSummary.transformFallbackReason ?? 'none',
          transformFallbackContract: renderResult.packedSummary.transformFallbackContract ?? 'none',
          sourceItemCount: renderResult.packedSummary.sourceItemCount ?? 0,
          packedCount: renderResult.packedSummary.packedCount ?? 0,
          floatsPerItem: renderResult.packedSummary.floatsPerItem ?? 0
        }
      : null,
    gpuScreenSourceSummary: renderResult?.gpuScreenSourceInfo
      ? {
          sourceFallbackReason: renderResult.gpuScreenSourceInfo.sourceFallbackReason ?? 'none',
          sourceContract: renderResult.gpuScreenSourceInfo.sourceContract ?? 'none',
          sourceFallbackContract: renderResult.gpuScreenSourceInfo.sourceFallbackContract ?? 'none'
        }
      : null,
    gpuScreenExecutionSummary: renderResult?.gpuScreenExecutionSummary ?? null,
    tileAccumulationPayloadSummary: renderResult?.tileAccumulationPayloadSummary
      ? {
          payloadContract: renderResult.tileAccumulationPayloadSummary.payloadContract ?? 'none',
          batchCount: renderResult.tileAccumulationPayloadSummary.batchCount ?? 0,
          totalItemCount: renderResult.tileAccumulationPayloadSummary.totalItemCount ?? 0,
          maxBatchItemCount: renderResult.tileAccumulationPayloadSummary.maxBatchItemCount ?? 0,
          payloadFloatCount: renderResult.tileAccumulationPayloadSummary.payloadFloatCount ?? 0,
          payloadTextureWidth: renderResult.tileAccumulationPayloadSummary.payloadTextureWidth ?? 0,
          payloadTextureHeight: renderResult.tileAccumulationPayloadSummary.payloadTextureHeight ?? 0,
          payloadRowsPerColumn: renderResult.tileAccumulationPayloadSummary.payloadRowsPerColumn ?? 0,
          payloadColumnCount: renderResult.tileAccumulationPayloadSummary.payloadColumnCount ?? 0,
          payloadLayoutReason: renderResult.tileAccumulationPayloadSummary.payloadLayoutReason ?? 'none',
          payloadLayoutValid: !!renderResult.tileAccumulationPayloadSummary.payloadLayoutValid,
          payloadLayoutFailureReason: renderResult.tileAccumulationPayloadSummary.payloadLayoutFailureReason ?? 'none',
          maxTextureSize: renderResult.tileAccumulationPayloadSummary.maxTextureSize ?? 0
        }
      : null
  };
  const mappedProbeCollectionsSummary = buildMappedProbeCollectionsSummary(summary, summary.canvasSizeSummary);
  summary.mappedProbeCollectionsSummary = mappedProbeCollectionsSummary;
  summary.mappedProbeCollectionSummary = mappedProbeCollectionsSummary.collectionSummary ?? null;
  summary.screenshotProbeList = mappedProbeCollectionsSummary.screenshotProbeList;
  summary.mappedProbeList = mappedProbeCollectionsSummary.mappedProbeList;
  summary.mappedProbePatchSummary = mappedProbeCollectionsSummary.firstMappedProbePatchSummary;
  if (mappedProbeCollectionsSummary.firstMappedProbeSummary) {
    Object.assign(summary, mappedProbeCollectionsSummary.firstMappedProbeSummary);
  }
  summary.mappedProbeSummaryNote = 'top-level mappedProbe* fields reflect the first probe in mappedProbeList; use mappedProbeList for the full multi-probe comparison';
  return summary;
}

function buildEmptyPayloadSourceSummary(candidate) {
  const payloadArray = candidate?.screenSpace?.gpuPackedPayloads;
  return {
    screenSpacePresent: !!candidate?.screenSpace,
    inspectDataPresent: !!candidate?.inspectData,
    payloadArrayPresent: Array.isArray(payloadArray),
    payloadArrayLength: Array.isArray(payloadArray) ? payloadArray.length : 0,
    candidateCount: 0,
    usableCandidateCount: 0,
    glMatchCandidateCount: 0,
    textureCandidateCount: 0,
    distinctKinds: [],
    failureCounts: {
      noPayloadArray: Array.isArray(payloadArray) ? 0 : 1,
      noTexture: 0,
      noGl: 0,
      glMismatch: 0,
      unusable: 0
    }
  };
}

function buildInspectAttemptRecord(candidate, inspection = null) {
  const successful = !!inspection?.ok;
  return {
    requestedSource: candidate?.requestedSource ?? 'auto',
    source: candidate?.source ?? 'none',
    sourceReason: candidate?.reason ?? 'none',
    drawPath: candidate?.actualDrawPath ?? 'none',
    screenSpacePresent: !!candidate?.screenSpace,
    inspectDataPresent: !!candidate?.inspectData,
    ok: successful,
    failureReason: successful
      ? 'none'
      : (inspection?.failureReason ?? ((candidate?.screenSpace || candidate?.inspectData)
        ? 'inspect-not-attempted'
        : 'inspect-source-unavailable')),
    payloadCandidateCount: inspection?.payloadCandidateCount ?? 0,
    payloadSourceSummary: inspection?.payloadSourceSummary ?? buildEmptyPayloadSourceSummary(candidate)
  };
}

function buildActualDrawInspectCandidate(renderResult, actualDrawPath, requestedSource) {
  if (actualDrawPath === 'packed') {
    if (Array.isArray(renderResult?.tileCompositePlan?.batches) && renderResult.tileCompositePlan.batches.length > 0) {
      return {
        requestedSource,
        source: 'tile-composite-packed-batches',
        inspectMethod: 'tile-composite-packed-batches',
        inspectData: renderResult.tileCompositePlan,
        actualDrawPath,
        reason: 'actual-draw-uses-tile-composite-packed-batches'
      };
    }
    return {
      requestedSource,
      source: 'packed-screen-space',
      inspectMethod: 'gpu-packed-texture',
      screenSpace: renderResult?.packedScreenSpace ?? null,
      actualDrawPath,
      reason: 'actual-draw-uses-packed-screen-space'
    };
  }

  if (actualDrawPath === 'gpu-screen') {
    return {
      requestedSource,
      source: 'gpu-screen-source-space',
      inspectMethod: 'gpu-packed-texture',
      screenSpace: renderResult?.gpuScreenSourceInfo?.sourceSpace ?? null,
      actualDrawPath,
      reason: 'actual-draw-uses-gpu-screen-source-space'
    };
  }

  return {
    requestedSource,
    source: 'none',
    inspectMethod: 'unsupported',
    screenSpace: null,
    actualDrawPath,
    reason: `actual-draw-path-${actualDrawPath}-is-not-inspectable`
  };
}

function buildInspectableScreenSpaceCandidates(renderResult, requestedSource = 'auto') {
  const actualDrawPath =
    renderResult?.drawThroughputSummary?.actualDrawPath ??
    renderResult?.drawPathSummary?.actualPath ??
    'none';
  const packedCandidate = {
    requestedSource,
    source: 'packed-screen-space',
    screenSpace: renderResult?.packedScreenSpace ?? null,
    actualDrawPath
  };
  const gpuScreenFallbackCandidate = {
    requestedSource,
    source: actualDrawPath === 'gpu-screen'
      ? 'gpu-screen-source-space'
      : 'gpu-screen-source-space-fallback',
    screenSpace: renderResult?.gpuScreenSourceInfo?.sourceSpace ?? null,
    actualDrawPath
  };

  if (requestedSource === 'packed') {
    return [{
      ...packedCandidate,
      reason: actualDrawPath === 'packed'
        ? 'explicit-packed-source-matches-actual-draw'
        : 'explicit-packed-source'
    }];
  }

  if (requestedSource === 'gpu-screen-fallback') {
    return [{
      ...gpuScreenFallbackCandidate,
      reason: actualDrawPath === 'gpu-screen'
        ? 'explicit-gpu-screen-source-matches-actual-draw'
        : 'explicit-gpu-screen-fallback-source'
    }];
  }

  if (requestedSource === 'actual-draw') {
    return [buildActualDrawInspectCandidate(renderResult, actualDrawPath, requestedSource)];
  }

  if (actualDrawPath === 'gpu-screen') {
    return [
      {
        ...gpuScreenFallbackCandidate,
        reason: 'auto-prefers-actual-gpu-screen-source-space'
      },
      {
        ...packedCandidate,
        reason: 'auto-fallback-to-packed-screen-space'
      }
    ];
  }

  if (actualDrawPath === 'packed') {
    return [
      {
        ...packedCandidate,
        reason: 'auto-prefers-actual-packed-screen-space'
      },
      {
        ...gpuScreenFallbackCandidate,
        reason: 'auto-fallback-to-gpu-screen-source-space'
      }
    ];
  }

  return [
    {
      ...packedCandidate,
      reason: 'auto-packed-screen-space'
    },
    {
      ...gpuScreenFallbackCandidate,
      reason: 'auto-gpu-screen-source-space'
    }
  ];
}

function buildInspectResultBase({
  inspection,
  renderResult,
  attempts,
  requestedSource,
  outputMode,
  inspectedCandidate
}) {
  const deterministicState = buildDeterministicStateSummary();
  const renderResultSummary = buildRenderResultInspectionSummary(renderResult);
  const actualDrawPath = renderResultSummary.actualDrawPath ?? 'none';
  const executionSummary = renderResultSummary.executionSummary ?? null;
  const gpuFallbackSummary = renderResultSummary.gpuFallbackSummary ?? null;
  const gpuCompatibilityBridgeSummary = renderResultSummary.gpuCompatibilityBridgeSummary ?? null;
  const actualDrawCandidate = buildActualDrawInspectCandidate(renderResult, actualDrawPath, 'actual-draw');
  const actualDrawInspectSupported = actualDrawCandidate?.inspectMethod === 'tile-composite-packed-batches' ||
    actualDrawCandidate?.inspectMethod === 'gpu-packed-texture';
  const actualDrawAttempt = attempts.find((attempt) => attempt?.requestedSource === 'actual-draw') ?? null;
  const actualDrawInspectFailureReason = actualDrawInspectSupported
    ? (actualDrawAttempt?.failureReason ?? (inspection?.ok ? 'none' : inspection?.failureReason ?? 'none'))
    : (actualDrawCandidate?.reason ?? 'actual-draw-inspect-unsupported');
  const sharedRepresentativeProbe = buildSharedRepresentativeFramebufferProbe(renderResultSummary);
  const canvasSizeSummary = buildCanvasSizeSummary();
  const mappedProbeCollectionsSummary = renderResultSummary.mappedProbeCollectionsSummary ??
    buildMappedProbeCollectionsSummary(renderResultSummary, canvasSizeSummary);
  renderResultSummary.mappedProbeCollectionsSummary = mappedProbeCollectionsSummary;
  renderResultSummary.mappedProbeCollectionSummary = mappedProbeCollectionsSummary.collectionSummary ?? null;
  if (renderResultSummary.executionSummary) {
    renderResultSummary.executionSummary.mappedProbeCollectionSummary = mappedProbeCollectionsSummary.collectionSummary ?? null;
    renderResultSummary.executionSummary.screenshotProbeList = mappedProbeCollectionsSummary.screenshotProbeList;
    renderResultSummary.executionSummary.mappedProbeList = mappedProbeCollectionsSummary.mappedProbeList;
  }
  const firstMappedProbeSummary = mappedProbeCollectionsSummary.firstMappedProbeSummary ?? null;
  const firstMappedProbePatchSummary = mappedProbeCollectionsSummary.firstMappedProbePatchSummary ?? null;
  if (firstMappedProbeSummary) {
    Object.assign(renderResultSummary, firstMappedProbeSummary);
  }
  if (firstMappedProbePatchSummary) {
    renderResultSummary.mappedProbePatchSummary = firstMappedProbePatchSummary;
    Object.assign(renderResultSummary, firstMappedProbePatchSummary);
  }

  return {
    ok: !!inspection?.ok,
    failureReason: inspection?.failureReason ?? 'none',
    inspectSourceRequested: requestedSource,
    inspectJsonMode: outputMode,
    actualDrawPath,
    drawPath:
      inspection?.drawPath ??
      actualDrawPath,
    inspectedSourceSpace: inspectedCandidate?.source ?? 'none',
    inspectedSourceReason: inspectedCandidate?.reason ?? 'none',
    tileCompositePathRequested: getRequestedTileCompositePath(),
    tileCompositePathActual: executionSummary?.tileCompositePath ?? 'none',
    tileCompositePrimitiveRequested: getRequestedTileCompositePrimitive(),
    tileCompositePrimitiveActual: executionSummary?.tileCompositePrimitive ?? 'none',
    tileCompositeRectContract: executionSummary?.tileCompositeRectContract ?? 'none',
    tileCompositeContract:
      executionSummary?.compositingContract ??
      renderResultSummary.tileCompositeSummary?.compositingContract ??
      renderResult?.tileCompositePlan?.summary?.compositingContract ??
      'none',
    actualDrawInspectSupported,
    actualDrawInspectDataSource: actualDrawCandidate?.source ?? 'none',
    actualDrawInspectFailureReason,
    gpuFallbackActive: !!gpuFallbackSummary?.active,
    gpuCompatibilityBridgeActive: !!gpuCompatibilityBridgeSummary?.active,
    drawPathSummary: renderResultSummary.drawPathSummary,
    drawThroughputSummary: renderResultSummary.drawThroughputSummary,
    canvasSizeSummary,
    mappedProbeCollectionsSummary,
    mappedProbeCollectionSummary: mappedProbeCollectionsSummary.collectionSummary ?? null,
    screenshotProbeList: mappedProbeCollectionsSummary.screenshotProbeList,
    mappedProbeList: mappedProbeCollectionsSummary.mappedProbeList,
    ...firstMappedProbeSummary,
    ...firstMappedProbePatchSummary,
    deterministicState: buildSlimDeterministicStateSummary(deterministicState),
    attemptedSources: attempts,
    ...sharedRepresentativeProbe,
    lastRenderResultSummary: renderResultSummary
  };
}

function buildSlimInspectResult({
  inspection,
  renderResult,
  attempts,
  requestedSource,
  outputMode,
  inspectedCandidate
}) {
  const base = buildInspectResultBase({
    inspection,
    renderResult,
    attempts,
    requestedSource,
    outputMode,
    inspectedCandidate
  });

  if (!inspection?.ok) {
    return base;
  }

  const executionSummary = base?.lastRenderResultSummary?.executionSummary ?? null;
  const accumulationMaxItems = Number.isFinite(executionSummary?.accumulationMaxItemsPerTile)
    ? Math.max(0, executionSummary.accumulationMaxItemsPerTile | 0)
    : 0;
  const targetTileSplatCount = Number.isFinite(inspection.tileCompositeTileSplatCount)
    ? Math.max(0, inspection.tileCompositeTileSplatCount | 0)
    : 0;
  const targetLocalOrder = Number.isFinite(inspection.tileCompositeLocalOrder)
    ? Math.max(0, inspection.tileCompositeLocalOrder | 0)
    : 0;
  const centerSampleContext = inspection?.sampleContexts?.center ?? null;
  const targetVisitedItems = Number.isFinite(centerSampleContext?.accumulationVisitedItems)
    ? Math.max(0, centerSampleContext.accumulationVisitedItems | 0)
    : (accumulationMaxItems > 0 ? Math.min(targetTileSplatCount, accumulationMaxItems) : 0);
  const targetSkippedItems = Math.max(0, targetTileSplatCount - targetVisitedItems);
  const targetTileTruncated = targetSkippedItems > 0;
  const targetIncludedInLoopWindow = typeof centerSampleContext?.accumulationTargetReached === 'boolean'
    ? !!centerSampleContext.accumulationTargetReached
    : targetLocalOrder < targetVisitedItems;
  const targetSkippedNearerCount = Number.isFinite(centerSampleContext?.accumulationTargetSkippedByEarlyOutCount)
    ? Math.max(0, centerSampleContext.accumulationTargetSkippedByEarlyOutCount | 0)
    : (targetIncludedInLoopWindow ? 0 : Math.max(0, targetLocalOrder + 1 - targetVisitedItems));

  return {
    ...base,
    requestedIndex: inspection.requestedIndex,
    payloadIndex: inspection.payloadIndex,
    localIndex: inspection.localIndex,
    payloadKind: inspection.payloadKind,
    payloadCount: inspection.payloadCount,
    payloadWidth: inspection.payloadWidth,
    payloadHeight: inspection.payloadHeight,
    rowsPerColumn: inspection.rowsPerColumn,
    columnCount: inspection.columnCount,
    tileCompositeBatchIndex: inspection.tileCompositeBatchIndex,
    tileCompositeTileId: inspection.tileCompositeTileId,
    tileCompositeSourceVisibleIndex: inspection.tileCompositeSourceVisibleIndex,
    tileCompositeSourceSplatIndex: inspection.tileCompositeSourceSplatIndex,
    tileCompositeLocalOrder: inspection.tileCompositeLocalOrder,
    tileCompositeTileSplatCount: inspection.tileCompositeTileSplatCount,
    tileCompositeNearerNeighborCount: inspection.tileCompositeNearerNeighborCount,
    tileCompositeFartherNeighborCount: inspection.tileCompositeFartherNeighborCount,
    tileCompositeOverlappingNeighborCount: inspection.tileCompositeOverlappingNeighborCount,
    tileCompositeOverlappingNearerNeighborCount: inspection.tileCompositeOverlappingNearerNeighborCount,
    tileCompositeOverlappingFartherNeighborCount: inspection.tileCompositeOverlappingFartherNeighborCount,
    overlappingNearerAlphaSum: inspection.overlappingNearerAlphaSum,
    overlappingFartherAlphaSum: inspection.overlappingFartherAlphaSum,
    overlappingNearerCountAboveThreshold: inspection.overlappingNearerCountAboveThreshold,
    overlappingFartherCountAboveThreshold: inspection.overlappingFartherCountAboveThreshold,
    overlappingNearerRectOverlapAreaSum: inspection.overlappingNearerRectOverlapAreaSum,
    overlappingFartherRectOverlapAreaSum: inspection.overlappingFartherRectOverlapAreaSum,
    overlappingNearerRectOverlapRatioToTargetSum: inspection.overlappingNearerRectOverlapRatioToTargetSum,
    overlappingFartherRectOverlapRatioToTargetSum: inspection.overlappingFartherRectOverlapRatioToTargetSum,
    overlappingNearerDepthSpread: inspection.overlappingNearerDepthSpread,
    overlappingFartherDepthSpread: inspection.overlappingFartherDepthSpread,
    estimatedNearerTransmittanceAtCenter: inspection.estimatedNearerTransmittanceAtCenter,
    estimatedNearerAlphaCompositeAtCenter: inspection.estimatedNearerAlphaCompositeAtCenter,
    centerSamplePixelIndexPx: inspection.centerSamplePixelIndexPx,
    centerSampleCoordinateSpace: inspection.centerSampleCoordinateSpace,
    centerSampleAlignmentOk: inspection.centerSampleAlignmentOk,
    targetCenterPixelIndexPx: inspection.targetCenterPixelIndexPx,
    targetCenterPixelIndexSpace: inspection.targetCenterPixelIndexSpace,
    targetCenterPixelAlignmentReason: inspection.targetCenterPixelAlignmentReason,
    centerSampleNearerAlphaSum: inspection.centerSampleNearerAlphaSum,
    centerSampleFartherAlphaSum: inspection.centerSampleFartherAlphaSum,
    centerSampleOverlappingNearerAlphaSum: inspection.centerSampleNearerAlphaSum,
    centerSampleOverlappingFartherAlphaSum: inspection.centerSampleFartherAlphaSum,
    centerSampleNearerContributorCount: inspection.centerSampleNearerContributorCount,
    centerSampleFartherContributorCount: inspection.centerSampleFartherContributorCount,
    nearerContributorsAtCenterTopK: inspection.nearerContributorsAtCenterTopK,
    fartherContributorsAtCenterTopK: inspection.fartherContributorsAtCenterTopK,
    sampleContexts: inspection.sampleContexts,
    overlappingNearerTopKSummary: inspection.overlappingNearerTopKSummary,
    overlappingFartherTopKSummary: inspection.overlappingFartherTopKSummary,
    tileCompositeLocalOrderFraction: inspection.tileCompositeLocalOrderFraction,
    tileCompositeOrderBucket: inspection.tileCompositeOrderBucket,
    tileCompositeOverlapContext: inspection.tileCompositeOverlapContext,
    tileCompositeDepthOrderSummary: inspection.tileCompositeDepthOrderSummary,
    tileCompositeTileSummary: inspection.tileCompositeTileSummary,
    tileAccumulationMaxItems: accumulationMaxItems,
    tileAccumulationTruncatedTileCount: executionSummary?.tileAccumulationTruncatedTileCount ?? 0,
    tileAccumulationMaxObservedTileItems: executionSummary?.tileAccumulationMaxObservedTileItems ?? 0,
    tileAccumulationTotalSkippedItems: executionSummary?.tileAccumulationTotalSkippedItems ?? 0,
    tileAccumulationWorstTileId: executionSummary?.tileAccumulationWorstTileId ?? -1,
    tileAccumulationWorstTileItemCount: executionSummary?.tileAccumulationWorstTileItemCount ?? 0,
    tileAccumulationWorstTileSkippedCount: executionSummary?.tileAccumulationWorstTileSkippedCount ?? 0,
    tileAccumulationEarlyOutEnabled: !!executionSummary?.tileAccumulationEarlyOutEnabled,
    tileAccumulationEarlyOutThreshold: executionSummary?.tileAccumulationEarlyOutThreshold ?? 0,
    tileAccumulationEarlyOutTriggeredTileCount: executionSummary?.tileAccumulationEarlyOutTriggeredTileCount ?? 0,
    tileAccumulationEarlyOutTriggeredPixelEstimate: executionSummary?.tileAccumulationEarlyOutTriggeredPixelEstimate ?? 0,
    tileAccumulationWorstEarlyOutTileId: executionSummary?.tileAccumulationWorstEarlyOutTileId ?? -1,
    tileAccumulationWorstEarlyOutCount: executionSummary?.tileAccumulationWorstEarlyOutCount ?? 0,
    tileAccumulationAverageVisitedItemsPerTile: executionSummary?.tileAccumulationAverageVisitedItemsPerTile ?? 0,
    tileAccumulationMaxVisitedItemsPerTile: executionSummary?.tileAccumulationMaxVisitedItemsPerTile ?? 0,
    tileAccumulationAverageVisitedItemsPerPixelEstimate: executionSummary?.tileAccumulationAverageVisitedItemsPerPixelEstimate ?? 0,
    tileAccumulationVisitedRatioSummary: executionSummary?.tileAccumulationVisitedRatioSummary ?? null,
    tileAccumulationObservedTileSummaries: executionSummary?.tileAccumulationObservedTileSummaries ?? [],
    tileAccumulationOrderingSummary: executionSummary?.tileAccumulationOrderingSummary ?? null,
    tileAccumulationBatchBoundarySummary: executionSummary?.tileAccumulationBatchBoundarySummary ?? null,
    tileAccumulationObservedOrderingMismatches: executionSummary?.tileAccumulationObservedOrderingMismatches ?? [],
    tileAccumulationHeavyTileSummaries: executionSummary?.tileAccumulationHeavyTileSummaries ?? [],
    tileAccumulationRepresentativeTileId: executionSummary?.tileAccumulationRepresentativeTileId ?? -1,
    tileAccumulationRepresentativeTileItemCount: executionSummary?.tileAccumulationRepresentativeTileItemCount ?? 0,
    tileAccumulationRepresentativeTileOrderPreview: executionSummary?.tileAccumulationRepresentativeTileOrderPreview ?? null,
    tileAccumulationRepresentativeTileDepthPreview: executionSummary?.tileAccumulationRepresentativeTileDepthPreview ?? null,
    tileAccumulationRepresentativeTileBatchSpan: executionSummary?.tileAccumulationRepresentativeTileBatchSpan ?? 1,
    tileAccumulationRepresentativeTileSequenceConsistent: !!executionSummary?.tileAccumulationRepresentativeTileSequenceConsistent,
    tileAccumulationContributionSummary: executionSummary?.tileAccumulationContributionSummary ?? null,
    tileAccumulationRepresentativeSampleMode: executionSummary?.tileAccumulationRepresentativeSampleMode ?? 'none',
    tileAccumulationRepresentativeSampleSelectionMode: executionSummary?.tileAccumulationRepresentativeSampleSelectionMode ?? 'none',
    tileAccumulationRepresentativeSampleSelectionReason: executionSummary?.tileAccumulationRepresentativeSampleSelectionReason ?? 'none',
    tileAccumulationRepresentativeSamplePixel: executionSummary?.tileAccumulationRepresentativeSamplePixel ?? [0, 0],
    tileAccumulationRepresentativeSampleHasContribution: !!executionSummary?.tileAccumulationRepresentativeSampleHasContribution,
    tileAccumulationRepresentativeSampleCandidateCount: executionSummary?.tileAccumulationRepresentativeSampleCandidateCount ?? 0,
    tileAccumulationRepresentativeSampleEvaluatedCandidateCount: executionSummary?.tileAccumulationRepresentativeSampleEvaluatedCandidateCount ?? 0,
    tileAccumulationRepresentativeSampleUsableItemSource: executionSummary?.tileAccumulationRepresentativeSampleUsableItemSource ?? 'none',
    tileAccumulationRepresentativeSampleItemReadMode: executionSummary?.tileAccumulationRepresentativeSampleItemReadMode ?? 'none',
    tileAccumulationRepresentativeSampleEvaluatedItemCount: executionSummary?.tileAccumulationRepresentativeSampleEvaluatedItemCount ?? 0,
    tileAccumulationRepresentativeSampleContributionLog: executionSummary?.tileAccumulationRepresentativeSampleContributionLog ?? [],
    tileAccumulationRepresentativeSampleFinalT: executionSummary?.tileAccumulationRepresentativeSampleFinalT ?? 1,
    tileAccumulationRepresentativeSampleAccumColor: executionSummary?.tileAccumulationRepresentativeSampleAccumColor ?? [0, 0, 0],
    tileAccumulationRepresentativeSampleResolvedColor: executionSummary?.tileAccumulationRepresentativeSampleResolvedColor ?? [0, 0, 0],
    tileAccumulationRepresentativeSampleContributionCount: executionSummary?.tileAccumulationRepresentativeSampleContributionCount ?? 0,
    tileAccumulationRepresentativeSampleAlphaSum: executionSummary?.tileAccumulationRepresentativeSampleAlphaSum ?? 0,
    tileAccumulationRepresentativeSampleContributionSum: executionSummary?.tileAccumulationRepresentativeSampleContributionSum ?? [0, 0, 0],
    tileAccumulationRepresentativeSampleLastContributedLocalOrder: executionSummary?.tileAccumulationRepresentativeSampleLastContributedLocalOrder ?? -1,
    tileAccumulationRepresentativeSampleThresholdCrossingCount: executionSummary?.tileAccumulationRepresentativeSampleThresholdCrossingCount ?? 0,
    tileAccumulationRepresentativeSampleThresholdSkippedCount: executionSummary?.tileAccumulationRepresentativeSampleThresholdSkippedCount ?? 0,
    tileAccumulationRepresentativeSampleFirstThresholdSkipLocalOrder: executionSummary?.tileAccumulationRepresentativeSampleFirstThresholdSkipLocalOrder ?? -1,
    tileAccumulationRepresentativeSampleFirstThresholdSkipAlpha: executionSummary?.tileAccumulationRepresentativeSampleFirstThresholdSkipAlpha ?? 0,
    tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceBefore: executionSummary?.tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceBefore ?? 1,
    tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceAfter: executionSummary?.tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceAfter ?? 1,
    tileAccumulationRepresentativeSampleThresholdSkipPreview: executionSummary?.tileAccumulationRepresentativeSampleThresholdSkipPreview ?? [],
    tileAccumulationRepresentativeSampleThresholdSemantics: executionSummary?.tileAccumulationRepresentativeSampleThresholdSemantics ?? null,
    tileAccumulationRepresentativeSampleFramebufferColor: executionSummary?.tileAccumulationRepresentativeSampleFramebufferColor ?? [0, 0, 0],
    tileAccumulationRepresentativeSampleFramebufferReadbackValid: !!executionSummary?.tileAccumulationRepresentativeSampleFramebufferReadbackValid,
    tileAccumulationRepresentativeSampleFramebufferReadbackReason: executionSummary?.tileAccumulationRepresentativeSampleFramebufferReadbackReason ?? 'not-attempted',
    tileAccumulationRepresentativeSampleFramebufferReadbackPixel: executionSummary?.tileAccumulationRepresentativeSampleFramebufferReadbackPixel ?? [0, 0],
    tileAccumulationRepresentativeSampleFramebufferReadbackGlPixel: executionSummary?.tileAccumulationRepresentativeSampleFramebufferReadbackGlPixel ?? [0, 0],
    tileAccumulationRepresentativeSampleFramebufferReadbackRgba8: executionSummary?.tileAccumulationRepresentativeSampleFramebufferReadbackRgba8 ?? [0, 0, 0, 0],
    tileAccumulationRepresentativeSampleResolvedColorDelta: executionSummary?.tileAccumulationRepresentativeSampleResolvedColorDelta ?? [0, 0, 0],
    tileAccumulationRepresentativeSampleResolvedColorDeltaAbsMax: executionSummary?.tileAccumulationRepresentativeSampleResolvedColorDeltaAbsMax ?? 0,
    tileAccumulationRepresentativeSampleResolvedColorMatchesFramebuffer: !!executionSummary?.tileAccumulationRepresentativeSampleResolvedColorMatchesFramebuffer,
    tileAccumulationRepresentativeSampleResolvedColorMatchTolerance: executionSummary?.tileAccumulationRepresentativeSampleResolvedColorMatchTolerance ?? 0,
    tileAccumulationContractVersion: executionSummary?.tileAccumulationContractVersion ?? 'none',
    tileAccumulationTruncationRatio: executionSummary?.tileAccumulationTruncationRatio ?? 0,
    tileAccumulationTargetTileTruncated: targetTileTruncated,
    tileAccumulationTargetTileVisitedItems: targetVisitedItems,
    tileAccumulationTargetTileSkippedItems: targetSkippedItems,
    tileAccumulationTargetIncludedInLoopWindow: targetIncludedInLoopWindow,
    tileAccumulationTargetSkippedNearerCount: targetSkippedNearerCount,
    tileAccumulationTargetTileEarlyOutTriggered: !!centerSampleContext?.accumulationEarlyOutTriggered,
    tileAccumulationTargetTileEarlyOutAtItem: centerSampleContext?.accumulationEarlyOutAtItem ?? -1,
    tileAccumulationTargetTileEarlyOutAtTransmittance: centerSampleContext?.accumulationEarlyOutAtTransmittance ?? 1,
    tileAccumulationTargetTileEarlyOutBeforeTarget: !!centerSampleContext?.accumulationEarlyOutBeforeTarget,
    tileAccumulationTargetTileContributingItems: centerSampleContext?.accumulationContributingItems ?? 0,
    tileCompositeTargetTileHeavy: !!inspection.tileCompositeTargetTileHeavy,
    tileCompositeTargetTileBatchSpan: inspection.tileCompositeTargetTileBatchSpan ?? 1,
    tileCompositeTargetSequenceConsistent: !!inspection.tileCompositeTargetSequenceConsistent,
    tileCompositeTargetOrderingMismatchCount: inspection.tileCompositeTargetOrderingMismatchCount ?? 0,
    tileCompositeTargetOrderingFirstMismatch: inspection.tileCompositeTargetOrderingFirstMismatch ?? null,
    tileCompositeTargetSequencePreview: inspection.tileCompositeTargetSequencePreview ?? [],
    textureColumnIndex: inspection.textureColumnIndex,
    textureRowIndex: inspection.textureRowIndex,
    textureXBase: inspection.textureXBase,
    centerPx: inspection.centerPx,
    depth: inspection.depth,
    payloadRadius: inspection.payloadRadius,
    colorAlpha: inspection.colorAlpha,
    conic: inspection.conic,
    unclampedPointSize: inspection.unclampedPointSize,
    clampedPointSize: inspection.clampedPointSize,
    clampApplied: inspection.clampApplied,
    rasterRectMinPx: inspection.rasterRectMinPx,
    rasterRectMaxPxExclusive: inspection.rasterRectMaxPxExclusive,
    rasterWidthPx: inspection.rasterWidthPx,
    rasterHeightPx: inspection.rasterHeightPx,
    rasterPixelArea: inspection.rasterPixelArea,
    rasterCoverageOvershootEstimate: inspection.rasterCoverageOvershootEstimate,
    rasterCoverageOvershootRatio: inspection.rasterCoverageOvershootRatio,
    fragmentSamples: inspection.fragmentSamples
  };
}

function buildFullInspectResult({
  inspection,
  renderResult,
  attempts,
  requestedSource,
  outputMode,
  inspectedCandidate
}) {
  const base = buildInspectResultBase({
    inspection,
    renderResult,
    attempts,
    requestedSource,
    outputMode,
    inspectedCandidate
  });

  return {
    ...inspection,
    ...base,
    debugText: refreshLatestDebugText(),
    lastRenderResult: renderResult
  };
}

function applyDeterministicCameraPreset() {
  if (!raw) return false;

  const preset = deterministicQueryState.cameraPreset;
  if (!preset || preset.name === 'fit') {
    fitCameraToRaw(raw, controls, camera);
    appliedCameraPresetName = preset?.name ?? 'none';
    return !!preset;
  }

  const applied = applyViewerCameraPresetState(camera, controls, preset);
  if (!applied) {
    fitCameraToRaw(raw, controls, camera);
    appliedCameraPresetName = 'fit';
    return false;
  }

  appliedCameraPresetName = preset.name;
  return true;
}

function applyDeterministicUiState() {
  const appliedState = applyViewerQueryStateToUi(ui, deterministicQueryState);
  if (deterministicQueryState.active) {
    updateDrawPathNoteFromState(appliedState);
  }
  updateDeterministicStateNote();
  return appliedState;
}

function sanitizeSnapshotFileName(name) {
  const trimmed = typeof name === 'string' ? name.trim() : '';
  const baseName = trimmed || `gpu-step70-${appliedCameraPresetName || 'view'}`;
  const normalized = baseName.replace(/[^a-zA-Z0-9._-]+/g, '-').replace(/-+/g, '-');
  return normalized.toLowerCase().endsWith('.png') ? normalized : `${normalized}.png`;
}

function downloadBlob(blob, fileName) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function createSnapshotCanvasFromPixels(width, height, pixels) {
  const snapshotCanvas = document.createElement('canvas');
  snapshotCanvas.width = width;
  snapshotCanvas.height = height;
  const ctx = snapshotCanvas.getContext('2d', { willReadFrequently: true });
  const imageData = ctx.createImageData(width, height);
  const rowStride = width * 4;

  for (let y = 0; y < height; y++) {
    const srcOffset = (height - 1 - y) * rowStride;
    const dstOffset = y * rowStride;
    imageData.data.set(pixels.subarray(srcOffset, srcOffset + rowStride), dstOffset);
  }

  ctx.putImageData(imageData, 0, 0);
  return snapshotCanvas;
}

async function captureBlobFromCanvas(sourceCanvas, fileName, download) {
  return await new Promise((resolve, reject) => {
    sourceCanvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error('captureFrame failed: snapshot canvas toBlob returned null'));
        return;
      }

      if (download) {
        downloadBlob(blob, fileName);
      }

      resolve(blob);
    }, 'image/png');
  });
}

function captureSnapshotCanvasFromGpu(gpu) {
  const gl = gpu?.gl;
  if (!gl) {
    throw new Error('captureFrame failed: WebGL renderer is not ready');
  }

  const width = gl.drawingBufferWidth | 0;
  const height = gl.drawingBufferHeight | 0;
  if (width <= 0 || height <= 0) {
    throw new Error('captureFrame failed: drawing buffer is empty');
  }

  const pixels = new Uint8Array(width * height * 4);
  gl.finish();
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  return createSnapshotCanvasFromPixels(width, height, pixels);
}

async function renderCurrentFrame() {
  ensureGpu();
  const renderResult = await renderGpuFrame({
    raw,
    gpu: getGpu(),
    canvas,
    camera,
    controls,
    ui,
    tokenRef,
    infoEl: ui.info,
    interactionOverride: buildRenderOverrides(),
    deterministicStateSummary: buildDeterministicStateSummary()
  });
  latestRenderResult = renderResult;

  if (renderResult && typeof renderResult.debugText === 'string') {
    refreshLatestDebugText(renderResult.debugText);
  } else if (renderResult && typeof renderResult.infoText === 'string') {
    refreshLatestDebugText(renderResult.infoText);
  } else {
    refreshLatestDebugText();
  }

  return renderResult;
}

async function captureFrame(options = {}) {
  const download = options.download !== false;
  const fileName = sanitizeSnapshotFileName(options.name);

  try {
    await renderCurrentFrame();
    const snapshotCanvas = captureSnapshotCanvasFromGpu(getGpu());
    const blob = await captureBlobFromCanvas(snapshotCanvas, fileName, download);
    lastSnapshotSummary = {
      available: true,
      source: 'webgl-default-framebuffer-readpixels',
      renderWaitMode: 'direct-render-await',
      status: 'success',
      reason: 'none'
    };

    return {
      blob,
      fileName,
      source: lastSnapshotSummary.source,
      renderWaitMode: lastSnapshotSummary.renderWaitMode,
      status: lastSnapshotSummary.status,
      reason: lastSnapshotSummary.reason,
      deterministicState: buildDeterministicStateSummary(),
      lastRenderResultSummary: buildRenderResultInspectionSummary(latestRenderResult),
      debugText: refreshLatestDebugText(),
      lastRenderResult: latestRenderResult
    };
  } catch (error) {
    lastSnapshotSummary = {
      available: true,
      source: 'webgl-default-framebuffer-readpixels',
      renderWaitMode: 'direct-render-await',
      status: 'failure',
      reason: error?.message ?? 'unknown-snapshot-error'
    };
    throw error;
  }
}

async function inspectActiveSplat(options = {}) {
  const ensureCurrentFrame = options.ensureCurrentFrame !== false;
  const renderResult = ensureCurrentFrame || !latestRenderResult
    ? await renderCurrentFrame()
    : latestRenderResult;

  const gpu = getGpu();
  const gl = gpu?.gl;
  if (!gl) {
    throw new Error('inspectActiveSplat failed: WebGL renderer is not ready');
  }

  const inspectSource = normalizeInspectSource(
    options.inspectSource ?? deterministicQueryState.inspectSource,
    'auto'
  );
  const outputMode = normalizeInspectJsonMode(
    options.outputMode ?? options.inspectJsonMode ?? deterministicQueryState.inspectJsonMode,
    'slim'
  );
  const candidates = buildInspectableScreenSpaceCandidates(renderResult, inspectSource);
  if (candidates.length <= 0) {
    throw new Error('inspectActiveSplat failed: no inspectable screen-space payloads available');
  }

  const attempts = [];
  for (const candidate of candidates) {
    if (!candidate.screenSpace && !candidate.inspectData) {
      attempts.push(buildInspectAttemptRecord(candidate));
      continue;
    }

    let inspection;
    if (candidate.inspectMethod === 'tile-composite-packed-batches') {
      inspection = inspectPackedInterleavedTileCompositeItem(candidate.inspectData, options);
    } else {
      inspection = inspectGpuPackedPayloadItem(gl, candidate.screenSpace, options);
    }
    const inspectionWithSource = {
      ...inspection,
      drawPath: candidate.actualDrawPath,
      inspectedSourceSpace: candidate.source,
      inspectedSourceReason: candidate.reason
    };
    attempts.push(buildInspectAttemptRecord(candidate, inspection));
    if (inspection.ok) {
      return outputMode === 'full'
        ? buildFullInspectResult({
            inspection: inspectionWithSource,
            renderResult,
            attempts,
            requestedSource: inspectSource,
            outputMode,
            inspectedCandidate: candidate
          })
        : buildSlimInspectResult({
            inspection: inspectionWithSource,
            renderResult,
            attempts,
            requestedSource: inspectSource,
            outputMode,
            inspectedCandidate: candidate
          });
    }
  }

  const failedInspection = {
    ok: false,
    failureReason: 'inspect-no-usable-payload-source'
  };
  return outputMode === 'full'
    ? buildFullInspectResult({
        inspection: failedInspection,
        renderResult,
        attempts,
        requestedSource: inspectSource,
        outputMode,
        inspectedCandidate: null
      })
    : buildSlimInspectResult({
        inspection: failedInspection,
        renderResult,
        attempts,
        requestedSource: inspectSource,
        outputMode,
        inspectedCandidate: null
      });
}

function installViewerDebugApi() {
  window.gpuViewerDebug = {
    captureFrame,
    compareSingleSplat: (input = {}) => computeGaussianDebugState(input),
    inspectActiveSplat,
    inspectActiveSplatSlim: (options = {}) => inspectActiveSplat({
      ...options,
      outputMode: 'slim'
    }),
    inspectActiveSplatFull: (options = {}) => inspectActiveSplat({
      ...options,
      outputMode: 'full'
    }),
    getDefaultSingleSplatCompareInput: () => structuredClone(DEFAULT_SINGLE_SPLAT_COMPARE_INPUT),
    getDeterministicState: () => buildDeterministicStateSummary(),
    getLatestDebugText: () => refreshLatestDebugText(),
    getLastRenderResult: () => latestRenderResult,
    scheduleRender: () => scheduler.scheduleRender()
  };
}

function scheduleRenderAndPersist() {
  const state = readAndSaveUiState(ui);
  updateDrawPathNoteFromState(state);
  scheduler.scheduleRender();
}

function updateDrawPathNoteFromState(stateLike) {
  const summary = summarizeUiState(stateLike);
  if (!ui.drawPathSelectNote) return;

  if (summary.drawPath === 'gpu-screen') {
    ui.drawPathSelectNote.textContent =
      `full-frame only; gpu-screen debug distinguishes actual, source, and reference; tile path=${summary.tileCompositePath}; tile primitive=${summary.tileCompositePrimitive}`;
    return;
  }

  if (summary.drawPath === 'packed') {
    ui.drawPathSelectNote.textContent =
      `full-frame only; packed is the formal reference path; tile path=${summary.tileCompositePath}; tile primitive=${summary.tileCompositePrimitive}`;
    return;
  }

  ui.drawPathSelectNote.textContent =
    `full-frame only; legacy is the fallback path; tile path=${summary.tileCompositePath}; tile primitive=${summary.tileCompositePrimitive}`;
}

function bindSliderTextUpdates() {
  [
    ['timeSlider', 'timeVal', 2],
    ['splatScaleSlider', 'splatScaleVal', 2],
    ['sigmaScaleSlider', 'sigmaScaleVal', 2],
    ['prefilterVarSlider', 'prefilterVarVal', 2],
    ['renderScaleSlider', 'renderScaleVal', 2],
    ['timeDurationSlider', 'timeDurationVal', 1]
  ].forEach(([sliderKey, valueKey, digits]) => {
    ui[sliderKey].addEventListener('input', () => {
      ui[valueKey].textContent = Number(ui[sliderKey].value).toFixed(digits);
      scheduler.scheduleRender();
    });
  });

  ui.strideSlider.addEventListener('input', () => {
    ui.strideVal.textContent = ui.strideSlider.value;
    scheduler.scheduleRender();
  });

  ui.maxVisibleSlider.addEventListener('input', () => {
    ui.maxVisibleVal.textContent = ui.maxVisibleSlider.value;
    scheduler.scheduleRender();
  });

  ui.bgGraySlider.addEventListener('input', () => {
    ui.bgGrayVal.textContent = ui.bgGraySlider.value;
    scheduler.scheduleRender();
  });
}

const scheduler = createRenderScheduler({
  renderFrame: renderCurrentFrame,
  tokenRef,
  isPlaying: () => (playback ? playback.isPlaying() : false)
});

playback = createViewerPlayback({
  ui,
  controls,
  scheduleRender: scheduler.scheduleRender,
  getTimeRange: () => ({ min: parseFloat(ui.timeSlider.min), max: parseFloat(ui.timeSlider.max) }),
  requestNextFrame: (cb) => requestAnimationFrame(cb),
  onPlaybackStateChange: () => {
    scheduler.scheduleRender();
  },
  playbackSpeed: 2.0
});

const fileIO = createViewerFileIO({
  ui,
  parseArrayBuffer: (buf) => parseSplat4DV2(buf),
  onSceneLoaded: async (nextRaw) => {
    raw = nextRaw;
    if (!applyDeterministicCameraPreset()) {
      fitCameraToRaw(raw, controls, camera);
      if (!deterministicQueryState.cameraPreset) {
        appliedCameraPresetName = 'none';
      }
    }
    await scheduler.scheduleRender();
  },
  scheduleRender: scheduler.scheduleRender,
  defaultSceneUrl: './scene_v2.splat4d'
});

function bindPersistentUiState() {
  if (typeof uiUnbindPersistence === 'function') {
    uiUnbindPersistence();
    uiUnbindPersistence = null;
  }

  uiUnbindPersistence = bindUiStatePersistence(ui, {
    onChange: (state) => {
      updateDrawPathNoteFromState(state);
      scheduler.scheduleRender();
    }
  });
}

function bindUiEvents() {
  [
    'useSHCheck',
    'useRot4dCheck',
    'useNativeRot4dCheck',
    'useNativeMarginalCheck',
    'forceSh3dCheck'
  ].forEach((key) => {
    ui[key].addEventListener('change', scheduleRenderAndPersist);
  });

  if (ui.debugLogBtn) {
    ui.debugLogBtn.addEventListener('click', () => {
      exportLatestDebugTextToArea();
    });
  }

  if (ui.debugLogCopyBtn) {
    ui.debugLogCopyBtn.addEventListener('click', async () => {
      if (!ui.debugLogArea?.value) exportLatestDebugTextToArea();
      await copyDebugLogText(ui);
    });
  }

  if (ui.drawPathSelect) {
    ui.drawPathSelect.addEventListener('change', () => {
      const state = readAndSaveUiState(ui);
      updateDrawPathNoteFromState(state);
      scheduler.scheduleRender();
    });
  }

  ui.playBtn.addEventListener('click', () => {
    playback.togglePlaying();
  });

  ui.renderBtn.addEventListener('click', scheduler.scheduleRender);

  ui.resetCamBtn.addEventListener('click', () => {
    if (raw && !applyDeterministicCameraPreset()) {
      fitCameraToRaw(raw, controls, camera);
      if (!deterministicQueryState.cameraPreset) {
        appliedCameraPresetName = 'none';
      }
    }
    scheduler.scheduleRender();
  });

  controls.addEventListener('change', scheduler.scheduleRender);

  bindGpuDragInteraction(canvas, controls, interactionState, () => {
    scheduler.scheduleRender();
  });

  window.addEventListener('resize', () => {
    applyCanvasSize();
    scheduler.scheduleRender();
  });
}

function initializeUiState() {
  const appliedState = loadAndApplyUiState(ui);
  const deterministicState = applyDeterministicUiState();
  updateStaticUiText();
  updateDrawPathNoteFromState(deterministicQueryState.active ? deterministicState : appliedState);
  bindPersistentUiState();
}

function initializeDebugLogArea() {
  setDebugLogText(ui, '');
}

initializeUiState();
initializeDebugLogArea();
bindSliderTextUpdates();
bindUiEvents();
installViewerDebugApi();

applyCanvasSize();
playback.startLoop();
fileIO.bindFileInput();
fileIO.bindDragAndDrop(document);
fileIO.loadDefaultScene();
