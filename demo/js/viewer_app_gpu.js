import * as THREE from 'three';
import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
import { parseSplat4DV2 } from './splat4d_parser_v2.js';
import { fitCameraToRaw } from './rot4d_math.js';
import { renderGpuFrame, createGpuRenderer } from './gpu_renderer.js';

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

function ensureTileDebugControls() {
  const parent = ui.info.parentElement;

  let row1 = document.getElementById('tileDebugRow1');
  if (!row1) {
    row1 = document.createElement('div');
    row1.className = 'row';
    row1.id = 'tileDebugRow1';
    row1.innerHTML =
      '<label>show tile debug</label><input id="showTileDebug" type="checkbox"><span>heatmap overlay</span>';
    parent.insertBefore(row1, ui.info);
  }

  let row2 = document.getElementById('tileDebugRow2');
  if (!row2) {
    row2 = document.createElement('div');
    row2.className = 'row';
    row2.id = 'tileDebugRow2';
    row2.innerHTML =
      '<label>draw selected tile only</label><input id="drawSelectedTileOnly" type="checkbox"><span>single-tile draw</span>';
    parent.insertBefore(row2, ui.info);
  }

  let row3 = document.getElementById('tileDebugRow3');
  if (!row3) {
    row3 = document.createElement('div');
    row3.className = 'row';
    row3.id = 'tileDebugRow3';
    row3.innerHTML =
      '<label>use max tile</label><input id="useMaxTile" type="checkbox"><span>densest tile</span>';
    parent.insertBefore(row3, ui.info);
  }

  let row4 = document.getElementById('tileDebugRow4');
  if (!row4) {
    row4 = document.createElement('div');
    row4.className = 'row';
    row4.id = 'tileDebugRow4';
    row4.innerHTML =
      '<label>tile id</label><input id="selectedTileId" type="number" min="-1" step="1" value="-1" style="width:120px;"><span id="selectedTileIdNote">manual tile id</span>';
    parent.insertBefore(row4, ui.info);
  }

  ui.showTileDebugCheck = document.getElementById('showTileDebug');
  ui.drawSelectedTileOnlyCheck = document.getElementById('drawSelectedTileOnly');
  ui.useMaxTileCheck = document.getElementById('useMaxTile');
  ui.selectedTileIdInput = document.getElementById('selectedTileId');
  ui.selectedTileIdNote = document.getElementById('selectedTileIdNote');
}

ensureTileDebugControls();

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

window.__GPU_TILE_DEBUG_OVERLAY__ = false;
window.__GPU_TILE_DRAW_SELECTED_ONLY__ = false;
window.__GPU_TILE_USE_MAX_TILE__ = true;
window.__GPU_TILE_SELECTED_ID__ = -1;

function syncTileDebugGlobalsFromUI() {
  window.__GPU_TILE_DEBUG_OVERLAY__ = !!ui.showTileDebugCheck.checked;
  window.__GPU_TILE_DRAW_SELECTED_ONLY__ = !!ui.drawSelectedTileOnlyCheck.checked;
  window.__GPU_TILE_USE_MAX_TILE__ = !!ui.useMaxTileCheck.checked;

  const v = Number(ui.selectedTileIdInput.value);
  window.__GPU_TILE_SELECTED_ID__ = Number.isInteger(v) ? v : -1;

  const manualEnabled = ui.drawSelectedTileOnlyCheck.checked && !ui.useMaxTileCheck.checked;
  ui.selectedTileIdInput.disabled = !manualEnabled;
  ui.selectedTileIdNote.textContent = manualEnabled ? 'manual tile id' : 'used only when max tile is off';
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
      await renderGpuFrame({
        raw,
        gpu,
        canvas,
        camera,
        controls,
        ui,
        tokenRef,
        infoEl: ui.info
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

ui.selectedTileIdInput.addEventListener('input', () => {
  syncTileDebugGlobalsFromUI();
  scheduleRender();
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
syncTileDebugGlobalsFromUI();

setCanvasSize();
requestAnimationFrame(animate);
loadDefaultScene();
