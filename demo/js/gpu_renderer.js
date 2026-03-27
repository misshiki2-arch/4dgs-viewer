import {
  createProgram,
  createArrayBuffer,
  bindFloatAttrib,
  clearToGray,
  enableStandardAlphaBlend,
  disableDepth
} from './gpu_gl_utils.js';
import {
  GPU_STEP_VERTEX_SHADER,
  GPU_STEP_FRAGMENT_SHADER
} from './gpu_shaders.js';
import {
  computeTileGrid,
  buildTileLists,
  summarizeTileLists
} from './gpu_tile_utils.js';
import {
  buildTileHeatmapImageData,
  drawTileHeatmapOverlay,
  formatTileDebugSummary
} from './gpu_tile_debug.js';
import {
  getDrawTileMode,
  setDefaultDrawTileMode,
  chooseFocusTileId,
  buildDrawIndexList,
  formatTileSelectionState
} from './gpu_tile_select.js';
import { buildVisibleSplats } from './gpu_visible_builder.js';
import { buildDrawArraysFromIndices, uploadAndDraw } from './gpu_draw_utils.js';
import { formatGpuViewerInfo, setInfoText } from './gpu_info_utils.js';

function ensureDebugOverlayCanvas(mainCanvas) {
  let overlay = document.getElementById('gpuTileDebugOverlay');
  if (!overlay) {
    overlay = document.createElement('canvas');
    overlay.id = 'gpuTileDebugOverlay';
    overlay.style.position = 'absolute';
    overlay.style.inset = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.pointerEvents = 'none';
    overlay.style.zIndex = '2';
    overlay.style.display = 'none';
    const parent = mainCanvas.parentElement || document.body;
    parent.appendChild(overlay);
  }
  return overlay;
}

export function createGpuRenderer(canvas) {
  const gl = canvas.getContext('webgl2', {
    alpha: false,
    antialias: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false
  });
  if (!gl) throw new Error('WebGL2 is not available in this browser.');

  const program = createProgram(gl, GPU_STEP_VERTEX_SHADER, GPU_STEP_FRAGMENT_SHADER);

  const emptyF32 = new Float32Array(0);
  const centerBuffer = createArrayBuffer(gl, emptyF32);
  const radiusBuffer = createArrayBuffer(gl, emptyF32);
  const colorBuffer = createArrayBuffer(gl, emptyF32);
  const conicBuffer = createArrayBuffer(gl, emptyF32);

  const vao = gl.createVertexArray();

  bindFloatAttrib(gl, { vao, program, buffer: centerBuffer, name: 'aCenterPx', size: 2 });
  bindFloatAttrib(gl, { vao, program, buffer: radiusBuffer, name: 'aRadiusPx', size: 1 });
  bindFloatAttrib(gl, { vao, program, buffer: colorBuffer, name: 'aColorAlpha', size: 4 });
  bindFloatAttrib(gl, { vao, program, buffer: conicBuffer, name: 'aConic', size: 3 });

  const uViewportPx = gl.getUniformLocation(program, 'uViewportPx');

  const renderer = {
    gl,
    program,
    vao,
    centerBuffer,
    radiusBuffer,
    colorBuffer,
    conicBuffer,
    uViewportPx,
    width: canvas.width,
    height: canvas.height,
    resize(width, height) {
      this.width = width;
      this.height = height;
      gl.viewport(0, 0, width, height);
    }
  };

  renderer.resize(canvas.width, canvas.height);
  return renderer;
}

export async function renderGpuFrame({
  raw,
  gpu,
  canvas,
  camera,
  controls,
  ui,
  tokenRef,
  infoEl
}) {
  const gl = gpu.gl;

  const bg255 = parseInt(ui.bgGraySlider.value, 10);
  const bg = bg255 / 255.0;
  const renderScale = parseFloat(ui.renderScaleSlider.value);
  const stride = parseInt(ui.strideSlider.value, 10);
  const maxVisible = parseInt(ui.maxVisibleSlider.value, 10);
  const timestamp = parseFloat(ui.timeSlider.value);
  const scalingModifier = parseFloat(ui.splatScaleSlider.value);
  const sigmaScale = parseFloat(ui.sigmaScaleSlider.value);
  const prefilterVar = parseFloat(ui.prefilterVarSlider.value);
  const useSH = ui.useSHCheck.checked;
  const useRot4d = ui.useRot4dCheck.checked;
  const useNativeRot4d = ui.useNativeRot4dCheck.checked;
  const useNativeMarginal = ui.useNativeMarginalCheck.checked;
  const forceSh3d = ui.forceSh3dCheck.checked;
  const timeDuration = parseFloat(ui.timeDurationSlider.value);

  controls.update();
  camera.updateMatrixWorld(true);

  setDefaultDrawTileMode(window);

  // UI values win over stale globals when the corresponding widgets exist.
  if (ui.showTileDebugCheck) {
    window.__GPU_TILE_DEBUG_OVERLAY__ = !!ui.showTileDebugCheck.checked;
  }
  if (ui.drawSelectedTileOnlyCheck) {
    window.__GPU_TILE_DRAW_SELECTED_ONLY__ = !!ui.drawSelectedTileOnlyCheck.checked;
  }
  if (ui.useMaxTileCheck) {
    window.__GPU_TILE_USE_MAX_TILE__ = !!ui.useMaxTileCheck.checked;
  }
  if (ui.selectedTileIdInput) {
    const v = Number(ui.selectedTileIdInput.value);
    window.__GPU_TILE_SELECTED_ID__ = Number.isInteger(v) ? v : -1;
  }

  const mode = getDrawTileMode(window);

  const debugOverlayCanvas = ensureDebugOverlayCanvas(canvas);
  const debugCtx = debugOverlayCanvas.getContext('2d');

  if (!raw) {
    gpu.resize(canvas.width, canvas.height);
    gl.disable(gl.BLEND);
    clearToGray(gl, bg);

    debugOverlayCanvas.width = canvas.width;
    debugOverlayCanvas.height = canvas.height;
    debugCtx.clearRect(0, 0, debugOverlayCanvas.width, debugOverlayCanvas.height);
    debugOverlayCanvas.style.display = 'none';

    setInfoText(infoEl, 'GPU Step7 viewer\nNo scene loaded.');
    return;
  }

  const frameToken = ++tokenRef.value;
  const t0 = performance.now();

  const tileGrid = computeTileGrid(canvas.width, canvas.height, 32);
  const camPos = camera.position.clone();

  const visibleResult = await buildVisibleSplats({
    raw,
    camera,
    canvasWidth: canvas.width,
    canvasHeight: canvas.height,
    renderScale,
    stride,
    maxVisible,
    timestamp,
    scalingModifier,
    sigmaScale,
    prefilterVar,
    useSH,
    useRot4d,
    useNativeRot4d,
    useNativeMarginal,
    forceSh3d,
    timeDuration,
    camPos,
    tokenRef,
    frameToken,
    tileGrid
  });

  if (visibleResult === null) return;

  const { visible, activeTileBox } = visibleResult;

  const tileData = buildTileLists(visible, tileGrid.tileCols, tileGrid.tileRows);
  const tileSummary = summarizeTileLists(
    tileData,
    tileGrid.tileCols,
    tileGrid.tileRows,
    activeTileBox
  );

  const focusTileId = chooseFocusTileId(tileData, mode);
  const drawIndices = buildDrawIndexList(visible, tileData, focusTileId, mode.drawSelectedOnly);
  const drawData = buildDrawArraysFromIndices(visible, drawIndices);

  gpu.resize(canvas.width, canvas.height);

  disableDepth(gl);
  enableStandardAlphaBlend(gl);
  clearToGray(gl, bg);

  uploadAndDraw(gl, gpu, drawData, canvas.width, canvas.height);

  debugOverlayCanvas.width = canvas.width;
  debugOverlayCanvas.height = canvas.height;
  debugCtx.clearRect(0, 0, debugOverlayCanvas.width, debugOverlayCanvas.height);

  if (mode.showOverlay) {
    const heatmap = buildTileHeatmapImageData({
      tileCounts: tileData.counts,
      tileCols: tileGrid.tileCols,
      tileRows: tileGrid.tileRows,
      tileSize: tileGrid.tileSize,
      canvasWidth: canvas.width,
      canvasHeight: canvas.height,
      highlightTileId: focusTileId,
      selectedTileIds: mode.drawSelectedOnly && focusTileId >= 0 ? [focusTileId] : null,
      alpha: 0.35
    });
    drawTileHeatmapOverlay(debugCtx, heatmap);
    debugOverlayCanvas.style.display = 'block';
  } else {
    debugOverlayCanvas.style.display = 'none';
  }

  const elapsed = performance.now() - t0;
  const avgRefsPerVisible = visible.length > 0 ? (tileSummary.totalRefs / visible.length) : 0;
  const tileDebugText = formatTileDebugSummary({
    tileData,
    tileCols: tileGrid.tileCols,
    tileRows: tileGrid.tileRows,
    tileSize: tileGrid.tileSize,
    highlightTileId: focusTileId,
    canvasWidth: canvas.width,
    canvasHeight: canvas.height
  });
  const tileSelectionText = formatTileSelectionState(mode, focusTileId);

  const infoText = formatGpuViewerInfo({
    raw,
    visibleCount: visible.length,
    drawCount: drawData.nDraw,
    stride,
    useRot4d,
    useSH,
    useNativeRot4d,
    useNativeMarginal,
    prefilterVar,
    sigmaScale,
    renderScale,
    canvasWidth: canvas.width,
    canvasHeight: canvas.height,
    timestamp,
    splatScale: scalingModifier,
    elapsedMs: elapsed,
    stepLabel: 'GPU Step7',
    stepNotes: [
      'CPU computes screen-space splats + AABB',
      'CPU builds explicit tile->splat lists',
      'GPU can draw all visible splats OR a single tile subset',
      'UI toggles are wired to tile selection globals',
      'draw selected tile only = true で単一tile描画',
      'use max tile = true なら最大密度tileを描画',
      'tile id で任意tile描画',
      'show tile debug = true で heatmap overlay を表示'
    ],
    tileSummary,
    avgRefsPerVisible,
    extraLines: [
      '',
      tileSelectionText,
      '',
      tileDebugText
    ]
  });

  setInfoText(infoEl, infoText);
}
