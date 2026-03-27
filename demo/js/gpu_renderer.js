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
  formatTileDebugSummary,
  getTilePixelRect
} from './gpu_tile_debug.js';
import {
  getDrawTileMode,
  setDefaultDrawTileMode,
  chooseFocusTileId,
  buildDrawIndexList,
  formatTileSelectionState
} from './gpu_tile_select.js';
import { buildVisibleSplats, getVisibleBuildConfig } from './gpu_visible_builder.js';
import { buildDrawArraysFromIndices, uploadAndDraw, buildDrawStats, formatDrawStats } from './gpu_draw_utils.js';
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

function applyUiTileModeToGlobals(ui) {
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
}

function enableTileScissor(gl, canvas, tileRect) {
  const [x0, y0, x1, y1] = tileRect;
  const w = Math.max(0, x1 - x0);
  const h = Math.max(0, y1 - y0);
  // Canvas/UI coordinates are top-left origin, WebGL scissor uses bottom-left origin.
  const scY = canvas.height - y1;
  gl.enable(gl.SCISSOR_TEST);
  gl.scissor(x0, scY, w, h);
}

function disableTileScissor(gl) {
  gl.disable(gl.SCISSOR_TEST);
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

  controls.update();
  camera.updateMatrixWorld(true);

  setDefaultDrawTileMode(window);
  applyUiTileModeToGlobals(ui);
  const mode = getDrawTileMode(window, ui);

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

    setInfoText(infoEl, 'GPU Step8 viewer\nNo scene loaded.');
    return;
  }

  const frameToken = ++tokenRef.value;
  const t0 = performance.now();

  const buildConfig = getVisibleBuildConfig(ui);
  const tileGrid = computeTileGrid(canvas.width, canvas.height, 32);
  const camPos = camera.position.clone();

  const visibleResult = await buildVisibleSplats({
    raw,
    camera,
    canvasWidth: canvas.width,
    canvasHeight: canvas.height,
    camPos,
    tokenRef,
    frameToken,
    tileGrid,
    ...buildConfig
  });

  if (visibleResult === null) return;

  const { visible, activeTileBox, buildStats } = visibleResult;

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

  let focusTileRect = null;
  if (focusTileId >= 0) {
    const tx = focusTileId % tileGrid.tileCols;
    const ty = Math.floor(focusTileId / tileGrid.tileCols);
    focusTileRect = getTilePixelRect(tx, ty, tileGrid.tileSize, canvas.width, canvas.height);
  }

  gpu.resize(canvas.width, canvas.height);

  disableDepth(gl);
  enableStandardAlphaBlend(gl);
  clearToGray(gl, bg);

  const scissorEnabled = !!(mode.drawSelectedOnly && focusTileRect);
  if (scissorEnabled) {
    enableTileScissor(gl, canvas, focusTileRect);
  }

  uploadAndDraw(gl, gpu, drawData, canvas.width, canvas.height);

  if (scissorEnabled) {
    disableTileScissor(gl);
  }

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

  const drawStats = buildDrawStats({
    visibleCount: visible.length,
    drawData,
    mode,
    focusTileId
  });
  const drawStatsText = formatDrawStats(drawStats);

  const extraLines = [
    '',
    `scissorEnabled=${scissorEnabled}`,
    `focusTileRect=${focusTileRect ? '[' + focusTileRect.join(', ') + ']' : 'none'}`,
    `buildAccepted=${buildStats.accepted}  buildProcessed=${buildStats.processed}  buildCulled=${buildStats.culled}`,
    '',
    tileSelectionText,
    '',
    drawStatsText,
    '',
    tileDebugText
  ];

  const infoText = formatGpuViewerInfo({
    raw,
    visibleCount: visible.length,
    drawCount: drawData.nDraw,
    stride: buildConfig.stride,
    useRot4d: buildConfig.useRot4d,
    useSH: buildConfig.useSH,
    useNativeRot4d: buildConfig.useNativeRot4d,
    useNativeMarginal: buildConfig.useNativeMarginal,
    prefilterVar: buildConfig.prefilterVar,
    sigmaScale: buildConfig.sigmaScale,
    renderScale: buildConfig.renderScale,
    canvasWidth: canvas.width,
    canvasHeight: canvas.height,
    timestamp: buildConfig.timestamp,
    splatScale: buildConfig.scalingModifier,
    elapsedMs: elapsed,
    stepLabel: 'GPU Step8',
    stepNotes: [
      'CPU computes screen-space splats + AABB',
      'CPU builds explicit tile->splat lists',
      'GPU can draw all visible splats OR a single tile subset',
      'When single-tile draw is enabled, selected tile rect is clipped by scissor test',
      'show tile debug = true で heatmap overlay を表示'
    ],
    tileSummary,
    avgRefsPerVisible,
    drawStats,
    tileSelectionText,
    tileDebugText,
    extraLines
  });

  setInfoText(infoEl, infoText);
}
