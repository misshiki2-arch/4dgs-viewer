import * as THREE from 'three';
import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
import { parseSplat4DV2 } from './splat4d_parser_v2.js';
import { fitCameraToRaw } from './rot4d_math.js';
import { renderCpuComposite } from './cpu_compositor.js';

const canvas = document.getElementById('cpuCanvas');
const ctx = canvas.getContext('2d', { alpha: false, willReadFrequently: true });

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

const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 5000);
camera.position.set(40, 20, 20);

const controls = new OrbitControls(camera, canvas);
controls.target.set(35, 20, 2);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

let raw = null;
let playing = false;
let lastTime = performance.now();

const state = { renderPending: false, rendering: false };
const tokenRef = { value: 0 };

function setCanvasSize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(window.innerWidth * dpr));
  canvas.height = Math.max(1, Math.round(window.innerHeight * dpr));
  canvas.style.width = window.innerWidth + 'px';
  canvas.style.height = window.innerHeight + 'px';
  camera.aspect = canvas.width / canvas.height;
  camera.updateProjectionMatrix();
}
setCanvasSize();

function scheduleRender() {
  if (state.rendering) {
    tokenRef.value++;
    return;
  }
  if (state.renderPending) return;

  state.renderPending = true;
  requestAnimationFrame(async () => {
    state.renderPending = false;
    await renderCpuComposite({ raw, ctx, canvas, camera, controls, state, ui, tokenRef, infoEl: ui.info });
  });
}

async function loadArrayBuffer(buf) {
  raw = parseSplat4DV2(buf);
  fitCameraToRaw(raw, controls, camera);
  scheduleRender();
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

ui.strideSlider.addEventListener('input', () => { ui.strideVal.textContent = ui.strideSlider.value; scheduleRender(); });
ui.maxVisibleSlider.addEventListener('input', () => { ui.maxVisibleVal.textContent = ui.maxVisibleSlider.value; scheduleRender(); });
ui.bgGraySlider.addEventListener('input', () => { ui.bgGrayVal.textContent = ui.bgGraySlider.value; scheduleRender(); });

[
  'useSHCheck',
  'useRot4dCheck',
  'useNativeRot4dCheck',
  'useNativeMarginalCheck',
  'forceSh3dCheck'
].forEach(key => {
  ui[key].addEventListener('change', scheduleRender);
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

document.addEventListener('dragover', e => { e.preventDefault(); ui.drop.style.display = 'flex'; });
document.addEventListener('dragleave', e => { e.preventDefault(); ui.drop.style.display = 'none'; });
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

requestAnimationFrame(animate);
loadDefaultScene();
