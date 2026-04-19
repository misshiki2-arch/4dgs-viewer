import { parseSplat4DV2 } from './splat4d_parser_v2.js';
import { fitCameraToRaw, computeGaussianDebugState, DEFAULT_SINGLE_SPLAT_COMPARE_INPUT } from './rot4d_math.js';
import { renderGpuFrame } from './gpu_renderer.js';
import { inspectGpuPackedPayloadItem } from './gpu_packed_payload_draw_shared.js';
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

function refreshLatestDebugText(explicitText = null) {
  const text = explicitText ?? ui.info?.textContent ?? '';
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
  parts.push(`gpuFramePolicyOverride=${deterministicQueryState.gpuFramePolicyOverride ?? 'auto'}`);
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

  if (renderResult && typeof renderResult.infoText === 'string') {
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

function selectInspectableScreenSpaceCandidates(renderResult) {
  const actualDrawPath =
    renderResult?.drawThroughputSummary?.actualDrawPath ??
    renderResult?.drawPathSummary?.actualPath ??
    'none';
  const candidates = [];

  if (actualDrawPath === 'gpu-screen') {
    candidates.push({
      screenSpace: renderResult?.gpuScreenSourceInfo?.sourceSpace ?? null,
      source: 'gpu-screen-source-space',
      actualDrawPath
    });
    candidates.push({
      screenSpace: renderResult?.packedScreenSpace ?? null,
      source: 'packed-screen-space-fallback',
      actualDrawPath
    });
  } else if (actualDrawPath === 'packed') {
    candidates.push({
      screenSpace: renderResult?.packedScreenSpace ?? null,
      source: 'packed-screen-space',
      actualDrawPath
    });
    candidates.push({
      screenSpace: renderResult?.gpuScreenSourceInfo?.sourceSpace ?? null,
      source: 'gpu-screen-source-space-fallback',
      actualDrawPath
    });
  } else {
    candidates.push({
      screenSpace: renderResult?.packedScreenSpace ?? null,
      source: 'packed-screen-space',
      actualDrawPath
    });
    candidates.push({
      screenSpace: renderResult?.gpuScreenSourceInfo?.sourceSpace ?? null,
      source: 'gpu-screen-source-space',
      actualDrawPath
    });
  }

  return candidates.filter((candidate) => candidate.screenSpace);
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

  const candidates = selectInspectableScreenSpaceCandidates(renderResult);
  if (candidates.length <= 0) {
    throw new Error('inspectActiveSplat failed: no inspectable screen-space payloads available');
  }

  const attempts = [];
  for (const candidate of candidates) {
    const inspection = inspectGpuPackedPayloadItem(gl, candidate.screenSpace, options);
    const inspectionWithSource = {
      ...inspection,
      drawPath: candidate.actualDrawPath,
      inspectedSourceSpace: candidate.source
    };
    attempts.push({
      source: candidate.source,
      drawPath: candidate.actualDrawPath,
      ok: !!inspection.ok,
      failureReason: inspection.failureReason ?? 'none',
      payloadCandidateCount: inspection.payloadCandidateCount ?? 0,
      payloadCandidates: inspection.payloadCandidates ?? []
    });
    if (inspection.ok) {
      return {
        ...inspectionWithSource,
        attemptedSources: attempts,
        deterministicState: buildDeterministicStateSummary(),
        debugText: refreshLatestDebugText(),
        lastRenderResult: renderResult
      };
    }
  }

  return {
    ok: false,
    failureReason: 'inspect-no-usable-payload-source',
    drawPath:
      renderResult?.drawThroughputSummary?.actualDrawPath ??
      renderResult?.drawPathSummary?.actualPath ??
      'none',
    inspectedSourceSpace: 'none',
    attemptedSources: attempts,
    deterministicState: buildDeterministicStateSummary(),
    debugText: refreshLatestDebugText(),
    lastRenderResult: renderResult
  };
}

function installViewerDebugApi() {
  window.gpuViewerDebug = {
    captureFrame,
    compareSingleSplat: (input = {}) => computeGaussianDebugState(input),
    inspectActiveSplat,
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
      'full-frame only; gpu-screen debug distinguishes actual, source, and reference';
    return;
  }

  if (summary.drawPath === 'packed') {
    ui.drawPathSelectNote.textContent = 'full-frame only; packed is the formal reference path';
    return;
  }

  ui.drawPathSelectNote.textContent = 'full-frame only; legacy is the fallback path';
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
    setCanvasSize();
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

setCanvasSize();
playback.startLoop();
fileIO.bindFileInput();
fileIO.bindDragAndDrop(document);
fileIO.loadDefaultScene();
