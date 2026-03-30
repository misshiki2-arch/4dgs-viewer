import * as THREE from 'three';
import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
import { parseSplat4DV2 } from './splat4d_parser_v2.js';
import { fitCameraToRaw } from './rot4d_math.js';
import { renderGpuFrame, createGpuRenderer } from './gpu_renderer.js';
import {
  createGpuInteractionState,
  bindGpuDragInteraction,
  getGpuInteractionOverride
} from './gpu_interaction_utils.js';

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

function applyInfoWrapStyle() {
  if (!ui.info) return;
  ui.info.style.whiteSpace = 'pre-wrap';
  ui.info.style.overflowWrap = 'anywhere';
  ui.info.style.wordBreak = 'break-word';
  ui.info.style.maxWidth = '100%';
}

function applyPanelResizeStyle() {
  if (!ui.info || !ui.info.parentElement) return;
  const panel = ui.info.parentElement;
  panel.style.resize = 'horizontal';
  panel.style.overflow = 'auto';
  panel.style.minWidth = '280px';
  panel.style.maxWidth = '70vw';
  panel.style.width = panel.style.width || '540px';
  panel.style.boxSizing = 'border-box';
}

function ensureTileDebugControls() {
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

function ensureTemporalIndexControls() {
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

function syncTemporalIndexUiState() {
  const enabled = !!ui.useTemporalIndexCheck.checked;
  const fixedMode = ui.temporalWindowModeSelect.value === 'fixed';

  ui.useTemporalIndexCacheCheck.disabled = !enabled;
  ui.temporalWindowModeSelect.disabled = !enabled;
  ui.fixedWindowRadiusInput.disabled = !(enabled && fixedMode);

  ui.temporalWindowModeNote.textContent = enabled ? 'temporal window policy' : 'used only when temporal index is on';
  ui.fixedWindowRadiusNote.textContent = (enabled && fixedMode)
    ? 'used when window mode=fixed'
    : 'used only when temporal index is on and mode=fixed';
}

ensureTileDebugControls();
ensureTemporalIndexControls();
applyInfoWrapStyle();
applyPanelResizeStyle();

const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 5000);
camera.position.set(40, 20, 20);

const controls = new OrbitControls(camera, canvas);
controls.target.set(35, 20, 2);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

let raw = null;
let gpu = null;
let playing = false;
let lastTime = performance.now();

const state = {
  renderPending: false,
  rendering: false
};

const tokenRef = {
  value: 0
};

const interactionState = createGpuInteractionState();

window.__GPU_TILE_DEBUG_OVERLAY__ = false;
window.__GPU_TILE_DRAW_SELECTED_ONLY__ = false;
window.__GPU_TILE_USE_MAX_TILE__ = true;
window.__GPU_TILE_SELECTED_ID__ = -1;
window.__GPU_TILE_RADIUS__ = 0;

function syncTileDebugGlobalsFromUI() {
  window.__GPU_TILE_DEBUG_OVERLAY__ = !!ui.showTileDebugCheck.checked;
  window.__GPU_TILE_DRAW_SELECTED_ONLY__ = !!ui.drawSelectedTileOnlyCheck.checked;
  window.__GPU_TILE_USE_MAX_TILE__ = !!ui.useMaxTileCheck.checked;

  const tileId = Number(ui.selectedTileIdInput.value);
  window.__GPU_TILE_SELECTED_ID__ = Number.isInteger(tileId) ? tileId : -1;

  const tileRadius = Number(ui.tileRadiusInput.value);
  window.__GPU_TILE_RADIUS__ = Number.isInteger(tileRadius) && tileRadius >= 0 ? tileRadius : 0;

  const manualEnabled = ui.drawSelectedTileOnlyCheck.checked && !ui.useMaxTileCheck.checked;
  ui.selectedTileIdInput.disabled = !manualEnabled;
  ui.selectedTileIdNote.textContent = manualEnabled ? 'manual tile id' : 'used only when max tile is off';

  const radiusEnabled = ui.drawSelectedTileOnlyCheck.checked;
  ui.tileRadiusInput.disabled = !radiusEnabled;
  ui.tileRadiusNote.textContent = radiusEnabled ? '0=single, 1=3x3, 2=5x5' : 'used only when single/multi-tile draw is on';
}

function setCanvasSize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(window.innerWidth * dpr));
  canvas.height = Math.max(1, Math.round(window.innerHeight * dpr));
  canvas.style.width = window.innerWidth + 'px';
  canvas.style.height = window.innerHeight + 'px';

  camera.aspect = canvas.width / canvas.height;
  camera.updateProjectionMatrix();

  if (gpu && gpu.resize) {
    gpu.resize(canvas.width, canvas.height);
  }
}

function ensureGpu() {
  if (!gpu) {
    gpu = createGpuRenderer(canvas);
  }
  return gpu;
}

async function scheduleRender() {
  if (state.rendering) {
    tokenRef.value++;
    return;
  }
  if (state.renderPending) return;

  state.renderPending = true;
  requestAnimationFrame(async () => {
    state.renderPending = false;
    state.rendering = true;

    try {
      ensureGpu();
      const interactionOverride = getGpuInteractionOverride(ui, interactionState);
      await renderGpuFrame({
        raw,
        gpu,
        canvas,
        camera,
        controls,
        ui,
        tokenRef,
        infoEl: ui.info,
        interactionOverride
      });
    } finally {
      state.rendering = false;
    }
  });
}

async function loadArrayBuffer(buf) {
  raw = parseSplat4DV2(buf);
  fitCameraToRaw(raw, controls, camera);
  await scheduleRender();
}

async function loadDefaultScene() {
  try {
    const res = await fetch('./scene_v2.splat4d');
    if (!res.ok) return;
    await loadArrayBuffer(await res.arrayBuffer());
  } catch (e) {
    console.warn(e);
  }
}

ui.fileInput.addEventListener('change', async e => {
  const f = e.target.files[0];
  if (!f) return;
  await loadArrayBuffer(await f.arrayBuffer());
});

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
    scheduleRender();
  });
});

ui.strideSlider.addEventListener('input', () => {
  ui.strideVal.textContent = ui.strideSlider.value;
  scheduleRender();
});

ui.maxVisibleSlider.addEventListener('input', () => {
  ui.maxVisibleVal.textContent = ui.maxVisibleSlider.value;
  scheduleRender();
});

ui.bgGraySlider.addEventListener('input', () => {
  ui.bgGrayVal.textContent = ui.bgGraySlider.value;
  scheduleRender();
});

[
  'useSHCheck',
  'useRot4dCheck',
  'useNativeRot4dCheck',
  'useNativeMarginalCheck',
  'forceSh3dCheck'
].forEach(key => {
  ui[key].addEventListener('change', scheduleRender);
});

[
  'showTileDebugCheck',
  'drawSelectedTileOnlyCheck',
  'useMaxTileCheck'
].forEach(key => {
  ui[key].addEventListener('change', () => {
    syncTileDebugGlobalsFromUI();
    scheduleRender();
  });
});

[
  'selectedTileIdInput',
  'tileRadiusInput'
].forEach(key => {
  ui[key].addEventListener('input', () => {
    syncTileDebugGlobalsFromUI();
    scheduleRender();
  });
});

[
  'useTemporalIndexCheck',
  'useTemporalIndexCacheCheck'
].forEach(key => {
  ui[key].addEventListener('change', () => {
    syncTemporalIndexUiState();
    scheduleRender();
  });
});

[
  'temporalWindowModeSelect',
  'fixedWindowRadiusInput'
].forEach(key => {
  ui[key].addEventListener('input', () => {
    syncTemporalIndexUiState();
    scheduleRender();
  });
});

ui.playBtn.addEventListener('click', () => {
  playing = !playing;
  ui.playBtn.textContent = playing ? '停止' : '再生';
});

ui.renderBtn.addEventListener('click', scheduleRender);

ui.resetCamBtn.addEventListener('click', () => {
  if (raw) fitCameraToRaw(raw, controls, camera);
  scheduleRender();
});

controls.addEventListener('change', scheduleRender);

bindGpuDragInteraction(canvas, controls, interactionState, () => {
  scheduleRender();
});

document.addEventListener('dragover', e => {
  e.preventDefault();
  ui.drop.style.display = 'flex';
});

document.addEventListener('dragleave', e => {
  e.preventDefault();
  ui.drop.style.display = 'none';
});

document.addEventListener('drop', async e => {
  e.preventDefault();
  ui.drop.style.display = 'none';
  const f = e.dataTransfer.files[0];
  if (!f) return;
  await loadArrayBuffer(await f.arrayBuffer());
});

window.addEventListener('resize', () => {
  setCanvasSize();
  scheduleRender();
});

function animate(now) {
  const dt = Math.min((now - lastTime) / 1000, 0.1);
  lastTime = now;

  if (playing) {
    let t = parseFloat(ui.timeSlider.value) + dt * 2;
    if (t > parseFloat(ui.timeSlider.max)) t = parseFloat(ui.timeSlider.min);
    ui.timeSlider.value = t.toFixed(2);
    ui.timeVal.textContent = Number(ui.timeSlider.value).toFixed(2);
    scheduleRender();
  }

  controls.update();
  requestAnimationFrame(animate);
}

ui.timeVal.textContent = Number(ui.timeSlider.value).toFixed(2);
ui.splatScaleVal.textContent = Number(ui.splatScaleSlider.value).toFixed(2);
ui.sigmaScaleVal.textContent = Number(ui.sigmaScaleSlider.value).toFixed(2);
ui.prefilterVarVal.textContent = Number(ui.prefilterVarSlider.value).toFixed(2);
ui.renderScaleVal.textContent = Number(ui.renderScaleSlider.value).toFixed(2);
ui.strideVal.textContent = ui.strideSlider.value;
ui.maxVisibleVal.textContent = ui.maxVisibleSlider.value;
ui.bgGrayVal.textContent = ui.bgGraySlider.value;
ui.timeDurationVal.textContent = Number(ui.timeDurationSlider.value).toFixed(1);

ui.showTileDebugCheck.checked = false;
ui.drawSelectedTileOnlyCheck.checked = false;
ui.useMaxTileCheck.checked = true;
ui.selectedTileIdInput.value = '-1';
ui.tileRadiusInput.value = '0';

ui.useTemporalIndexCheck.checked = true;
ui.useTemporalIndexCacheCheck.checked = true;
ui.temporalWindowModeSelect.value = 'max';
ui.fixedWindowRadiusInput.value = '0.50';

syncTileDebugGlobalsFromUI();
syncTemporalIndexUiState();

setCanvasSize();
requestAnimationFrame(animate);
loadDefaultScene();
