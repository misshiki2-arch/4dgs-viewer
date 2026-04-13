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
  chooseFocusTileId,
  buildNeighborTileIds,
  buildPerTileDrawIndexLists,
  summarizeTileDrawBatches,
  formatTileSelectionState
} from './gpu_tile_select.js';
import { buildVisibleSplats, getVisibleBuildConfig } from './gpu_visible_builder.js';
import {
  buildDrawArraysFromIndices,
  buildPerTileDrawBatches,
  summarizePerTileDrawBatches,
  buildDrawStats
} from './gpu_draw_utils.js';
import {
  summarizePackedDirectResources
} from './gpu_packed_draw_executor.js';
import {
  summarizeGpuScreenDrawState,
  isGpuScreenDrawReady
} from './gpu_screen_draw_executor.js';
import { formatGpuViewerInfo, setInfoText } from './gpu_info_utils.js';
import {
  buildGpuDebugExtraLines,
  buildLegacySample,
  buildPackedSample
} from './gpu_debug_info_builder.js';
import {
  getRequestedDrawPath,
  resolveDrawPath,
  summarizeDrawPathSelection,
  GPU_DRAW_PATH_LEGACY,
  GPU_DRAW_PATH_PACKED,
  GPU_DRAW_PATH_GPU_SCREEN
} from './gpu_draw_path_selector.js';
import {
  createScreenSpaceBuildContext,
  buildPackedGpuPrepScreenSpaceWithContext,
  summarizePackedScreenSpace,
  summarizePackedScreenSpaceComparison
} from './gpu_screen_space_builder.js';
import { executeFullFrameDrawByPath } from './gpu_draw_execution_router.js';
import { executeSelectedTileLegacyDraw } from './gpu_selected_tile_draw_executor.js';

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

function getDrawTileMode(ui) {
  const getLiveEl = (cachedEl, id) => {
    if (cachedEl && typeof cachedEl.checked !== 'undefined') return cachedEl;
    const liveEl = typeof document !== 'undefined' ? document.getElementById(id) : null;
    return liveEl || cachedEl || null;
  };

  const showTileDebugCheck = getLiveEl(ui?.showTileDebugCheck, 'showTileDebug');
  const drawSelectedTileOnlyCheck = getLiveEl(ui?.drawSelectedTileOnlyCheck, 'drawSelectedTileOnly');
  const useMaxTileCheck = getLiveEl(ui?.useMaxTileCheck, 'useMaxTile');
  const selectedTileIdInput = getLiveEl(ui?.selectedTileIdInput, 'selectedTileId');
  const tileRadiusInput = getLiveEl(ui?.tileRadiusInput, 'tileRadius');

  return {
    showOverlay: !!showTileDebugCheck?.checked,
    drawSelectedOnly: !!drawSelectedTileOnlyCheck?.checked,
    useMaxTile: !!useMaxTileCheck?.checked,
    selectedTileId: Number.isFinite(Number(selectedTileIdInput?.value))
      ? (Number(selectedTileIdInput.value) | 0)
      : -1,
    tileRadius: Number.isFinite(Number(tileRadiusInput?.value))
      ? Math.max(0, Number(tileRadiusInput.value) | 0)
      : 0
  };
}

function ensureGpuScreenSourceContext(gpu) {
  if (!gpu.gpuScreenSourceContext) {
    gpu.gpuScreenSourceContext = createScreenSpaceBuildContext();
  }
  return gpu.gpuScreenSourceContext;
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

function buildAllVisibleDrawData(visible) {
  const allDrawIndices = new Uint32Array(visible.length);
  for (let i = 0; i < visible.length; i++) allDrawIndices[i] = i;
  return buildDrawArraysFromIndices(visible, allDrawIndices);
}

function computeDrawFraction(visibleCount, drawCount) {
  if (!Number.isFinite(visibleCount) || visibleCount <= 0) return 0;
  if (!Number.isFinite(drawCount)) return 0;
  return drawCount / visibleCount;
}

function hasGpuPackedPayloads(screenSpace) {
  return Array.isArray(screenSpace?.gpuPackedPayloads) && screenSpace.gpuPackedPayloads.length > 0;
}

function hasRenderablePackedScreenSpace(screenSpace) {
  if (screenSpace?.packed instanceof Float32Array) return true;
  if (hasGpuPackedPayloads(screenSpace)) return true;
  return !!screenSpace?.summary?.transformHasBuffers || !!screenSpace?.transformSummary?.transformHasBuffers;
}

function getPackedLogicalLength(screenSpace) {
  if (screenSpace?.packed instanceof Float32Array) return screenSpace.packed.length;
  if (Number.isFinite(screenSpace?.packedCount) && Number.isFinite(screenSpace?.floatsPerItem)) {
    return Math.max(0, (screenSpace.packedCount | 0) * (screenSpace.floatsPerItem | 0));
  }
  return 0;
}

function buildSafeBuildStats(rawBuildStats, visible, packedScreenSpace, elapsedMs) {
  const visibleCount = Array.isArray(visible) ? visible.length : 0;

  return {
    ...(rawBuildStats || {}),
    packedVisiblePathEnabled: !!rawBuildStats?.packedVisiblePathEnabled,
    packedVisiblePathUsed: !!rawBuildStats?.packedVisiblePathUsed,
    packedVisiblePath: rawBuildStats?.packedVisiblePath ?? packedScreenSpace?.path ?? 'none',
    packedVisibleCount:
      Number.isFinite(rawBuildStats?.packedVisibleCount)
        ? rawBuildStats.packedVisibleCount
        : (Number.isFinite(packedScreenSpace?.packedCount) ? packedScreenSpace.packedCount : visibleCount),
    packedVisibleLength:
      Number.isFinite(rawBuildStats?.packedVisibleLength)
        ? rawBuildStats.packedVisibleLength
        : getPackedLogicalLength(packedScreenSpace),
    packedVisibleFloatsPerItem:
      Number.isFinite(rawBuildStats?.packedVisibleFloatsPerItem)
        ? rawBuildStats.packedVisibleFloatsPerItem
        : (Number.isFinite(packedScreenSpace?.floatsPerItem) ? packedScreenSpace.floatsPerItem : 0),
    visibleBuildMs: Number.isFinite(rawBuildStats?.visibleBuildMs) ? rawBuildStats.visibleBuildMs : elapsedMs,
    candidateBuildMs: Number.isFinite(rawBuildStats?.candidateBuildMs) ? rawBuildStats.candidateBuildMs : 0,
    screenSpaceBuildMs: Number.isFinite(rawBuildStats?.screenSpaceBuildMs) ? rawBuildStats.screenSpaceBuildMs : 0,
    totalBuildMs: Number.isFinite(rawBuildStats?.totalBuildMs) ? rawBuildStats.totalBuildMs : elapsedMs
  };
}

function buildEffectiveDrawData({
  mode,
  effectiveTileSummary,
  drawPathSelection,
  directPackedDrawInfo,
  packedScreenSpace,
  gpuScreenDrawInfo,
  legacyDrawData
}) {
  if (mode.drawSelectedOnly) {
    return { nDraw: effectiveTileSummary ? effectiveTileSummary.totalTileDrawCount : 0 };
  }

  return {
    nDraw:
      drawPathSelection.actualPath === GPU_DRAW_PATH_PACKED
        ? (directPackedDrawInfo?.drawCount ?? packedScreenSpace?.packedCount ?? 0)
        : drawPathSelection.actualPath === GPU_DRAW_PATH_GPU_SCREEN
          ? (gpuScreenDrawInfo?.drawCount ?? 0)
          : (legacyDrawData?.nDraw ?? 0)
  };
}

function buildGpuScreenResultSummary(gpuScreenSourceInfo, gpuScreenDrawInfo) {
  return {
    gpuScreenComparisonSummary: mergeGpuScreenComparisonSummary(
      gpuScreenSourceInfo?.sourceComparisonSummary ?? null,
      gpuScreenDrawInfo?.gpuScreenComparisonSummary ?? null
    ),
    gpuScreenExecutionSummary: gpuScreenDrawInfo?.gpuScreenExecutionSummary ?? null
  };
}

function buildRendererDrawStats({
  gpu,
  visible,
  mode,
  focusTileId,
  focusTileIds,
  effectiveTileSummary,
  executionSummary,
  packedScreenSpace,
  packedUploadSummary,
  effectiveDrawData
}) {
  const packedDirectResourceSummary = summarizePackedDirectResources(gpu);
  const drawStats = buildDrawStats({
    visibleCount: visible.length,
    drawData: effectiveDrawData,
    mode,
    focusTileId,
    focusTileIds,
    tileBatchSummary: effectiveTileSummary,
    executionSummary,
    packedScreenSpace,
    packedUploadSummary: {
      ...packedDirectResourceSummary,
      ...packedUploadSummary
    }
  });

  drawStats.drawFraction = computeDrawFraction(drawStats.visibleCount, drawStats.drawCount);
  return drawStats;
}

function selectGpuScreenSourceSpace(gpu, visible, packedCpuScreenSpace) {
  const context = ensureGpuScreenSourceContext(gpu);

  try {
    const experimental = buildPackedGpuPrepScreenSpaceWithContext(context, visible, {
      gl: gpu?.gl ?? null
    });
    if (hasRenderablePackedScreenSpace(experimental)) {
      return {
        sourceSpace: experimental,
        sourceSummary: summarizePackedScreenSpace(experimental),
        sourceComparisonSummary: summarizePackedScreenSpaceComparison(experimental),
        sourceFallbackReason: 'none'
      };
    }
  } catch (err) {
    console.warn('gpu-screen source build failed, falling back to packed-cpu source', err);
  }

  return {
    sourceSpace: packedCpuScreenSpace,
    sourceSummary: summarizePackedScreenSpace(packedCpuScreenSpace),
    sourceComparisonSummary: summarizePackedScreenSpaceComparison(packedCpuScreenSpace),
    sourceFallbackReason: 'gpu-source-build-fallback-to-packed-cpu'
  };
}

function mergeGpuScreenComparisonSummary(builderSummary, drawSummary) {
  if (!builderSummary && !drawSummary) return null;
  if (!builderSummary) return drawSummary ?? null;
  if (!drawSummary) return builderSummary;

  return {
    ...builderSummary,
    actualPath: drawSummary.actualPath ?? builderSummary.actualPath,
    actualRole: drawSummary.actualRole ?? builderSummary.actualRole
  };
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

  const mode = getDrawTileMode(ui);

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
    const emptyInfo = 'GPU Step53 viewer\nNo scene loaded.';
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
    buildStats: rawBuildStats
  } = visibleResult;

  const tileData = buildTileLists(visible, tileGrid.tileCols, tileGrid.tileRows);
  const tileSummary = summarizeTileLists(
    tileData,
    tileGrid.tileCols,
    tileGrid.tileRows,
    activeTileBox
  );

  const legacySample = buildLegacySample(visible);
  const packedSample = buildPackedSample(packedScreenSpace);

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
  const gpuScreenSourceInfo = selectGpuScreenSourceSpace(gpu, visible, packedScreenSpace);

  const drawPathSelection = summarizeDrawPathSelection(
    resolveDrawPath({
      requestedPath,
      hasPackedScreenSpace:
        hasRenderablePackedScreenSpace(packedScreenSpace) ||
        hasRenderablePackedScreenSpace(gpuScreenSourceInfo.sourceSpace),
      hasGpuScreenPath: isGpuScreenDrawReady(gpu)
    })
  );

  const needsLegacyExpandedArrays =
    drawPathSelection.actualPath === GPU_DRAW_PATH_LEGACY || mode.drawSelectedOnly;

  const legacyDrawData = needsLegacyExpandedArrays
    ? buildAllVisibleDrawData(visible)
    : null;

  gpu.resize(canvas.width, canvas.height);

  disableDepth(gl);
  enableStandardAlphaBlend(gl);
  clearToGray(gl, bg);

  const executionResult = mode.drawSelectedOnly
    ? executeSelectedTileLegacyDraw({
        gl,
        gpu,
        canvas,
        focusTileRects,
        perTileDrawBatches,
        drawPathSelection
      })
    : executeFullFrameDrawByPath({
        gl,
        gpu,
        canvas,
        drawPathSelection,
        packedScreenSpace,
        gpuScreenSourceSpace: gpuScreenSourceInfo.sourceSpace,
        legacyDrawData
      });

  const {
    executionSummary,
    packedUploadSummary,
    directPackedDrawInfo,
    gpuScreenDrawInfo
  } = executionResult;

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
  const buildStats = buildSafeBuildStats(rawBuildStats, visible, packedScreenSpace, elapsed);
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

  const effectiveDrawData = buildEffectiveDrawData({
    mode,
    effectiveTileSummary,
    drawPathSelection,
    directPackedDrawInfo,
    packedScreenSpace,
    gpuScreenDrawInfo,
    legacyDrawData
  });

  const gpuScreenSummary = summarizeGpuScreenDrawState(gpu);

  const {
    gpuScreenComparisonSummary,
    gpuScreenExecutionSummary
  } = buildGpuScreenResultSummary(gpuScreenSourceInfo, gpuScreenDrawInfo);

  const drawStats = buildRendererDrawStats({
    gpu,
    visible,
    mode,
    focusTileId,
    focusTileIds,
    effectiveTileSummary,
    executionSummary,
    packedScreenSpace,
    packedUploadSummary,
    effectiveDrawData
  });

  const extraLines = buildGpuDebugExtraLines({
    buildConfig,
    buildStats,
    drawStats,
    mode,
    focusTileIds,
    focusTileRects,
    ui,
    gpuScreenSummary,
    gpuScreenSourceSpace: gpuScreenSourceInfo.sourceSpace,
    gpuScreenComparisonSummary,
    drawPathSelection,
    visible,
    packedScreenSpace,
    gpuScreenExecutionSummary,
    legacySample,
    packedSample
  });

  if (gpuScreenSourceInfo?.sourceFallbackReason && gpuScreenSourceInfo.sourceFallbackReason !== 'none') {
    extraLines.push(`gpuScreenSourceFallbackReason=${gpuScreenSourceInfo.sourceFallbackReason}`);
  }

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
    stepLabel: 'GPU Step53',
    stepNotes: [
      'transform executor owns transformBatchSummary and downstream code forwards it without reinterpretation',
      'gpu resident payload pooling now has an upper-bound trim policy, preserving reuse while capping retained backend-owned textures across frames',
      'renderer stays thin and forwards source, transform, lifecycle, pool, and gpu-screen execution summaries to debug output',
      'packed-write backend keeps the offscreen FBO blend-disable fix while preserving existing public draw contracts'
    ],
    tileSummary,
    avgRefsPerVisible,
    drawStats,
    tileSelectionText,
    tileDebugText,
    extraLines: [
      ...extraLines,
    ]
  });

  setInfoText(infoEl, infoText);

  const result = {
    infoText,
    visible,
    packedScreenSpace,
    packedSummary,
    buildStats,
    drawStats,
    tileSummary,
    drawPathSummary: drawPathSelection,
    gpuScreenSummary,
    gpuScreenComparisonSummary,
    gpuScreenExecutionSummary,
    gpuScreenSourceInfo
  };

  return result;
}
