import { parseSplat4DV2 } from './splat4d_parser_v2.js';
import { fitCameraToRaw } from './rot4d_math.js';
import { renderGpuFrame } from './gpu_renderer.js';
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
  setDebugLogText,
  copyDebugLogText
} from './viewer_ui_controls.js';
import {
  syncTileDebugGlobalsFromUI,
  syncTemporalIndexUiState,
  syncTemporalBucketUiState,
  syncQualityOverrideUiState,
  syncPackedPathUiState,
  initializeViewerUiDefaults,
  syncAllViewerUiState
} from './viewer_ui_state.js';
import { createRenderScheduler } from './viewer_render_scheduler.js';
import { createViewerPlayback } from './viewer_playback.js';
import { createViewerFileIO } from './viewer_file_io.js';
import { createViewerScene } from './viewer_scene_setup.js';

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
applyInfoWrapStyle(ui.info);
applyPanelResizeStyle(ui.info);

const scene = createViewerScene(canvas);
const { camera, controls, ensureGpu, getGpu, setCanvasSize } = scene;

let raw = null;
let lastDebugText = '';
const tokenRef = { value: 0 };
const interactionState = createGpuInteractionState();

let playback = null;

function refreshLatestDebugText(explicitText = null) {
  const text = explicitText ?? ui.info?.textContent ?? '';
  lastDebugText = text;
  return text;
}

function exportLatestDebugTextToArea() {
  setDebugLogText(ui, refreshLatestDebugText());
  if (ui.debugLogNote) {
    ui.debugLogNote.textContent = lastDebugText ? 'latest debug text exported' : 'no debug text';
  }
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

const scheduler = createRenderScheduler({
  renderFrame: async () => {
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
      interactionOverride: buildRenderOverrides()
    });

    if (renderResult && typeof renderResult.infoText === 'string') {
      refreshLatestDebugText(renderResult.infoText);
    } else {
      refreshLatestDebugText();
    }
  },
  tokenRef,
  isPlaying: () => (playback ? playback.isPlaying() : false)
});

playback = createViewerPlayback({
  ui,
  controls,
  scheduleRender: scheduler.scheduleRender,
  getTimeRange: () => ({
    min: parseFloat(ui.timeSlider.min),
    max: parseFloat(ui.timeSlider.max)
  }),
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
    fitCameraToRaw(raw, controls, camera);
    await scheduler.scheduleRender();
  },
  scheduleRender: scheduler.scheduleRender,
  defaultSceneUrl: './scene_v2.splat4d'
});

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

function bindUiEvents() {
  [
    'useSHCheck',
    'useRot4dCheck',
    'useNativeRot4dCheck',
    'useNativeMarginalCheck',
    'forceSh3dCheck'
  ].forEach((key) => {
    ui[key].addEventListener('change', scheduler.scheduleRender);
  });

  [
    'showTileDebugCheck',
    'drawSelectedTileOnlyCheck',
    'useMaxTileCheck'
  ].forEach((key) => {
    ui[key].addEventListener('change', () => {
      syncTileDebugGlobalsFromUI(ui, window);
      scheduler.scheduleRender();
    });
  });

  [
    'selectedTileIdInput',
    'tileRadiusInput'
  ].forEach((key) => {
    ui[key].addEventListener('input', () => {
      syncTileDebugGlobalsFromUI(ui, window);
      scheduler.scheduleRender();
    });
  });

  [
    'useTemporalIndexCheck',
    'useTemporalIndexCacheCheck'
  ].forEach((key) => {
    ui[key].addEventListener('change', () => {
      syncTemporalIndexUiState(ui);
      scheduler.scheduleRender();
    });
  });

  [
    'temporalWindowModeSelect',
    'fixedWindowRadiusInput'
  ].forEach((key) => {
    ui[key].addEventListener('input', () => {
      syncTemporalIndexUiState(ui);
      scheduler.scheduleRender();
    });
  });

  [
    'useTemporalBucketCheck',
    'useTemporalBucketCacheCheck'
  ].forEach((key) => {
    ui[key].addEventListener('change', () => {
      syncTemporalBucketUiState(ui);
      scheduler.scheduleRender();
    });
  });

  [
    'temporalBucketWidthInput',
    'temporalBucketRadiusInput'
  ].forEach((key) => {
    ui[key].addEventListener('input', () => {
      syncTemporalBucketUiState(ui);
      scheduler.scheduleRender();
    });
  });

  [
    'usePlaybackOverrideCheck',
    'useInteractionOverrideCheck'
  ].forEach((key) => {
    ui[key].addEventListener('change', () => {
      syncQualityOverrideUiState(ui);
      scheduler.scheduleRender();
    });
  });

  [
    'playbackStrideInput',
    'playbackMaxVisibleInput',
    'playbackRenderScaleInput',
    'interactionStrideInput',
    'interactionMaxVisibleInput',
    'interactionRenderScaleInput'
  ].forEach((key) => {
    ui[key].addEventListener('input', () => {
      syncQualityOverrideUiState(ui);
      scheduler.scheduleRender();
    });
  });

  if (ui.usePackedVisiblePathCheck) {
    ui.usePackedVisiblePathCheck.addEventListener('change', () => {
      syncPackedPathUiState(ui);
      scheduler.scheduleRender();
    });
  }

  if (ui.drawPathSelect) {
    ui.drawPathSelect.addEventListener('change', () => {
      syncPackedPathUiState(ui);
      scheduler.scheduleRender();
    });
  }

  if (ui.debugLogBtn) {
    ui.debugLogBtn.addEventListener('click', () => {
      exportLatestDebugTextToArea();
    });
  }

  if (ui.debugLogCopyBtn) {
    ui.debugLogCopyBtn.addEventListener('click', async () => {
      if (!ui.debugLogArea?.value) {
        exportLatestDebugTextToArea();
      }
      await copyDebugLogText(ui);
    });
  }

  ui.playBtn.addEventListener('click', () => {
    playback.togglePlaying();
  });

  ui.renderBtn.addEventListener('click', scheduler.scheduleRender);

  ui.resetCamBtn.addEventListener('click', () => {
    if (raw) fitCameraToRaw(raw, controls, camera);
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

function initializeStaticUiText() {
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

initializeViewerUiDefaults(ui);
syncAllViewerUiState(ui, window);
initializeStaticUiText();
bindSliderTextUpdates();
bindUiEvents();
setCanvasSize();
playback.startLoop();
fileIO.bindFileInput();
fileIO.bindDragAndDrop(document);
fileIO.loadDefaultScene();
