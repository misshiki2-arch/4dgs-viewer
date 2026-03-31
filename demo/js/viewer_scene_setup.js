import * as THREE from 'three';
import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
import { createGpuRenderer } from './gpu_renderer.js';

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
