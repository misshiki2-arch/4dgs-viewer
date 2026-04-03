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
  buildNeighborTileIds,
  buildPerTileDrawIndexLists,
  summarizeTileDrawBatches,
  formatTileSelectionState
} from './gpu_tile_select.js';
import { buildVisibleSplats, getVisibleBuildConfig } from './gpu_visible_builder.js';
import {
  buildDrawArraysFromIndices,
  buildDrawArraysFromPacked,
  buildPerTileDrawBatches,
  summarizePerTileDrawBatches,
  uploadAndDraw,
  uploadPackedForStats,
  renderPerTileBatches,
  buildDrawStats
} from './gpu_draw_utils.js';
import { formatGpuViewerInfo, setInfoText } from './gpu_info_utils.js';
import { buildGpuDebugExtraLines } from './gpu_debug_info_builder.js';
import {
  getRequestedDrawPath,
  resolveDrawPath,
  summarizeDrawPathSelection,
  GPU_DRAW_PATH_LEGACY,
  GPU_DRAW_PATH_PACKED
} from './gpu_draw_path_selector.js';

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
  if (ui.showTileDebugCheck) window.__GPU_TILE_DEBUG_OVERLAY__ = !!ui.showTileDebugCheck.checked;
  if (ui.drawSelectedTileOnlyCheck) window.__GPU_TILE_DRAW_SELECTED_ONLY__ = !!ui.drawSelectedTileOnlyCheck.checked;
  if (ui.useMaxTileCheck) window.__GPU_TILE_USE_MAX_TILE__ = !!ui.useMaxTileCheck.checked;
  if (ui.selectedTileIdInput) {
    const v = Number(ui.selectedTileIdInput.value);
    window.__GPU_TILE_SELECTED_ID__ = Number.isInteger(v) ? v : -1;
  }
  if (ui.tileRadiusInput) {
    const r = Number(ui.tileRadiusInput.value);
    window.__GPU_TILE_RADIUS__ = Number.isInteger(r) && r >= 0 ? r : 0;
  }
}

function buildFocusTileRects(tileIds, tileGrid, canvasWidth, canvasHeight) {
  const rects = [];
  for (const tileId of tileIds) {
    const tx = tileId % tileGrid.tileCols;
    const ty = Math.floor(tileId / tileGrid.tileCols);
    rects.push({
      tileId,
      tx,
      ty,
      rect: getTilePixelRect(tx, ty, tileGrid.tileSize, canvasWidth, canvasHeight)
    });
  }
  return rects;
}

function enableTileScissor(gl, canvas, tileRect) {
  const [x0, y0, x1, y1] = tileRect;
  const w = Math.max(0, x1 - x0);
  const h = Math.max(0, y1 - y0);
  const scY = canvas.height - y1;
  gl.enable(gl.SCISSOR_TEST);
  gl.scissor(x0, scY, w, h);
}

function disableTileScissor(gl) {
  gl.disable(gl.SCISSOR_TEST);
}

function buildAllVisibleDrawData(visible) {
  const allDrawIndices = new Uint32Array(visible.length);
  for (let i = 0; i < visible.length; i++) allDrawIndices[i] = i;
  return buildDrawArraysFromIndices(visible, allDrawIndices);
}

function buildPackedVisibleDrawData(packedScreenSpace) {
  if (!packedScreenSpace?.packed || !Number.isFinite(packedScreenSpace?.packedCount)) {
    return null;
  }
  return buildDrawArraysFromPacked(packedScreenSpace.packed, packedScreenSpace.packedCount);
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
  infoEl,
  interactionOverride = null
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
    const emptyInfo = 'GPU Step22 viewer\nNo scene loaded.';
    setInfoText(infoEl, emptyInfo);
    return {
      infoText: emptyInfo,
      visible: [],
      packedScreenSpace: null,
      packedSummary: null,
      buildStats: null,
      drawStats: null
    };
  }

  const frameToken = ++tokenRef.value;
  const t0 = performance.now();

  const buildConfig = getVisibleBuildConfig(ui, interactionOverride);
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
    temporalSigmaThreshold: 3.0,
    enablePackedVisiblePath: !!buildConfig.enablePackedVisiblePath,
    ...buildConfig
  });
  if (visibleResult === null) return null;

  const {
    visible,
    packedScreenSpace,
    packedSummary,
    activeTileBox,
    buildStats
  } = visibleResult;

  const tileData = buildTileLists(visible, tileGrid.tileCols, tileGrid.tileRows);
  const tileSummary = summarizeTileLists(
    tileData,
    tileGrid.tileCols,
    tileGrid.tileRows,
    activeTileBox
  );

  const focusTileId = chooseFocusTileId(tileData, mode);
  const focusTileIds = mode.drawSelectedOnly
    ? buildNeighborTileIds(focusTileId, tileGrid.tileCols, tileGrid.tileRows, mode.tileRadius)
    : [];

  const tileBatches = mode.drawSelectedOnly
    ? buildPerTileDrawIndexLists(visible, tileData, focusTileIds, true)
    : [];

  const perTileDrawBatches = mode.drawSelectedOnly
    ? buildPerTileDrawBatches(visible, tileBatches)
    : [];

  const tileBatchSummary = mode.drawSelectedOnly
    ? summarizeTileDrawBatches(tileBatches)
    : null;

  const perTileDrawSummary = mode.drawSelectedOnly
    ? summarizePerTileDrawBatches(perTileDrawBatches)
    : null;

  const effectiveTileSummary = perTileDrawSummary || tileBatchSummary;

  const focusTileRects = mode.drawSelectedOnly
    ? buildFocusTileRects(focusTileIds, tileGrid, canvas.width, canvas.height)
    : [];

  const requestedPath = getRequestedDrawPath(ui);
  const drawPathSelection = resolveDrawPath({
    requestedPath,
    hasPackedScreenSpace: !!packedScreenSpace?.packed,
    hasGpuScreenPath: false
  });
  const drawPathSummary = summarizeDrawPathSelection(drawPathSelection);

  const legacyDrawData = (!mode.drawSelectedOnly || drawPathSummary.actualPath === GPU_DRAW_PATH_LEGACY)
    ? buildAllVisibleDrawData(visible)
    : null;

  const packedDrawData = (!mode.drawSelectedOnly && drawPathSummary.actualPath === GPU_DRAW_PATH_PACKED)
    ? buildPackedVisibleDrawData(packedScreenSpace)
    : null;

  const finalDrawData =
    drawPathSummary.actualPath === GPU_DRAW_PATH_PACKED
      ? (packedDrawData || legacyDrawData)
      : legacyDrawData;

  gpu.resize(canvas.width, canvas.height);

  disableDepth(gl);
  enableStandardAlphaBlend(gl);
  clearToGray(gl, bg);

  let executionSummary = null;
  let packedUploadSummary = null;

  if (!mode.drawSelectedOnly) {
    if (drawPathSummary.actualPath === GPU_DRAW_PATH_PACKED) {
      packedUploadSummary = uploadPackedForStats(gl, gpu, packedScreenSpace);
    }

    uploadAndDraw(gl, gpu, finalDrawData, canvas.width, canvas.height);
    executionSummary = {
      tileBatchCount: 1,
      uploadCount: 1,
      drawCallCount: 1,
      requestedDrawPath: drawPathSummary.requestedPath,
      actualDrawPath: drawPathSummary.actualPath,
      drawPathFallbackReason: drawPathSummary.fallbackReason
    };
  } else {
    const rectMap = new Map(focusTileRects.map(item => [item.tileId, item.rect]));
    executionSummary = renderPerTileBatches(
      gl,
      gpu,
      perTileDrawBatches,
      canvas.width,
      canvas.height,
      {
        beforeTile: (item) => {
          const rect = rectMap.get(item.tileId);
          if (rect) enableTileScissor(gl, canvas, rect);
          else disableTileScissor(gl);
        },
        afterTile: () => {
          disableTileScissor(gl);
        }
      }
    );
    executionSummary.requestedDrawPath = drawPathSummary.requestedPath;
    executionSummary.actualDrawPath = GPU_DRAW_PATH_LEGACY;
    executionSummary.drawPathFallbackReason =
      drawPathSummary.actualPath === GPU_DRAW_PATH_LEGACY
        ? drawPathSummary.fallbackReason
        : 'per-tile-legacy-only';
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
      selectedTileIds: mode.drawSelectedOnly ? focusTileIds : null,
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

  const tileSelectionText = formatTileSelectionState(
    mode,
    focusTileId,
    focusTileIds,
    effectiveTileSummary
  );

  const effectiveDrawData = mode.drawSelectedOnly
    ? { nDraw: effectiveTileSummary ? effectiveTileSummary.totalTileDrawCount : 0 }
    : finalDrawData;

  const drawStats = buildDrawStats({
    visibleCount: visible.length,
    drawData: effectiveDrawData,
    mode,
    focusTileId,
    focusTileIds,
    tileBatchSummary: effectiveTileSummary,
    executionSummary,
    packedScreenSpace,
    packedUploadSummary
  });

  const extraLines = buildGpuDebugExtraLines({
    buildConfig,
    buildStats,
    drawStats,
    mode,
    focusTileIds,
    focusTileRects,
    ui
  });

  const packedLines = [
    `packedVisiblePathEnabled=${!!buildStats?.packedVisiblePathEnabled}`,
    `packedVisiblePathUsed=${!!buildStats?.packedVisiblePathUsed}`,
    `packedVisiblePath=${buildStats?.packedVisiblePath ?? 'none'}`,
    `packedVisibleCount=${buildStats?.packedVisibleCount ?? 0}`,
    `packedVisibleLength=${buildStats?.packedVisibleLength ?? 0}`,
    `packedVisibleFloatsPerItem=${buildStats?.packedVisibleFloatsPerItem ?? 0}`,
    `requestedDrawPath=${drawPathSummary.requestedPath}`,
    `actualDrawPath=${executionSummary?.actualDrawPath ?? drawPathSummary.actualPath}`,
    `drawPathFallbackReason=${executionSummary?.drawPathFallbackReason ?? drawPathSummary.fallbackReason}`,
    `packedUploadBytes=${drawStats.packedUploadBytes}`,
    `packedUploadCount=${drawStats.packedUploadCount}`,
    `packedUploadLength=${drawStats.packedUploadLength}`,
    `packedUploadCapacityBytes=${drawStats.packedUploadCapacityBytes}`,
    `packedUploadReusedCapacity=${drawStats.packedUploadReusedCapacity}`
  ];

  const infoText = formatGpuViewerInfo({
    raw,
    visibleCount: visible.length,
    drawCount: effectiveDrawData?.nDraw ?? 0,
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
    stepLabel: 'GPU Step22',
    stepNotes: [
      'Packed draw path now records packed-upload statistics for reuse tracking',
      'Draw-path branching remains delegated to gpu_draw_path_selector.js',
      'Per-tile draw still uses legacy batches while full-frame packed draw records upload reuse info',
      'Logs now include packed upload bytes/capacity/reuse flags'
    ],
    tileSummary,
    avgRefsPerVisible,
    drawStats,
    tileSelectionText,
    tileDebugText,
    extraLines: [...extraLines, ...packedLines]
  });

  setInfoText(infoEl, infoText);

  return {
    infoText,
    visible,
    packedScreenSpace,
    packedSummary,
    buildStats,
    drawStats,
    tileSummary,
    drawPathSummary
  };
}
