import * as THREE from 'three';
import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
import { createGpuRenderer } from './gpu_renderer.js';

function applyVectorTuple(vec, tuple) {
  if (!vec || !Array.isArray(tuple) || tuple.length < 3) return;
  vec.set(Number(tuple[0]), Number(tuple[1]), Number(tuple[2]));
}

export function applyViewerCameraPresetState(camera, controls, preset) {
  if (!camera || !controls || !preset) return false;
  if (preset.name === 'fit') return false;

  applyVectorTuple(camera.position, preset.position);
  applyVectorTuple(controls.target, preset.target);
  if (Array.isArray(preset.up) && preset.up.length >= 3) {
    applyVectorTuple(camera.up, preset.up);
  } else {
    camera.up.set(0, 1, 0);
  }
  camera.updateProjectionMatrix();
  controls.update();
  return true;
}

export function createViewerScene(canvas) {
  const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 5000);
  camera.position.set(40, 20, 20);

  const controls = new OrbitControls(camera, canvas);
  controls.target.set(35, 20, 2);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.update();

  let gpu = null;

  function ensureGpu() {
    if (!gpu) {
      gpu = createGpuRenderer(canvas);
    }
    return gpu;
  }

  function setCanvasSize(options = {}) {
    const dpr = window.devicePixelRatio || 1;
    const fixedWidth = Number.isFinite(options?.fixedCanvasWidth)
      ? Math.max(1, Math.round(options.fixedCanvasWidth))
      : null;
    const fixedHeight = Number.isFinite(options?.fixedCanvasHeight)
      ? Math.max(1, Math.round(options.fixedCanvasHeight))
      : null;
    const fixedResolutionActive = fixedWidth !== null && fixedHeight !== null;
    canvas.width = fixedResolutionActive
      ? fixedWidth
      : Math.max(1, Math.round(window.innerWidth * dpr));
    canvas.height = fixedResolutionActive
      ? fixedHeight
      : Math.max(1, Math.round(window.innerHeight * dpr));
    canvas.style.width = window.innerWidth + 'px';
    canvas.style.height = window.innerHeight + 'px';

    camera.aspect = canvas.width / canvas.height;
    camera.updateProjectionMatrix();

    if (gpu && gpu.resize) {
      gpu.resize(canvas.width, canvas.height);
    }
  }

  function getGpu() {
    return gpu;
  }

  return {
    camera,
    controls,
    ensureGpu,
    getGpu,
    setCanvasSize
  };
}
