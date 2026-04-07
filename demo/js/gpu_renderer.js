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
  uploadAndDraw,
  renderPerTileBatches,
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
import { buildGpuDebugExtraLines } from './gpu_debug_info_builder.js';
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
  return {
    showOverlay: !!ui?.showTileDebugCheck?.checked,
    drawSelectedOnly: !!ui?.drawSelectedTileOnlyCheck?.checked,
    useMaxTile: !!ui?.useMaxTileCheck?.checked,
    selectedTileId: Number.isFinite(Number(ui?.selectedTileIdInput?.value))
      ? (Number(ui.selectedTileIdInput.value) | 0)
      : -1,
    tileRadius: Number.isFinite(Number(ui?.tileRadiusInput?.value))
      ? Math.max(0, Number(ui.tileRadiusInput.value) | 0)
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

function fmtNum(v, digits = 4) {
  return Number.isFinite(v) ? Number(v).toFixed(digits) : 'NaN';
}

function fmtArr(arr, digits = 4) {
  if (!Array.isArray(arr)) return 'none';
  return '[' + arr.map((v) => fmtNum(v, digits)).join(', ') + ']';
}

function buildLegacySample(visible) {
  if (!Array.isArray(visible) || visible.length === 0) return null;
  const v = visible[0];
  return {
    centerPx: [v.px, v.py],
    radiusPx: v.radius,
    colorAlpha: Array.isArray(v.colorAlpha) ? v.colorAlpha.slice(0, 4) : null,
    conic: Array.isArray(v.conic) ? v.conic.slice(0, 3) : null
  };
}

function buildPackedSample(screenSpace) {
  const packed = screenSpace?.packed;
  if (!(packed instanceof Float32Array) || packed.length < 16) return null;
  return {
    centerPx: [packed[0], packed[1]],
    radiusPx: packed[2],
    depth: packed[3],
    colorAlpha: [packed[4], packed[5], packed[6], packed[7]],
    conic: [packed[8], packed[9], packed[10]],
    reserved: packed[11],
    misc: [packed[12], packed[13], packed[14], packed[15]]
  };
}

function computeDrawFraction(visibleCount, drawCount) {
  if (!Number.isFinite(visibleCount) || visibleCount <= 0) return 0;
  if (!Number.isFinite(drawCount)) return 0;
  return drawCount / visibleCount;
}

function buildSafeBuildStats(rawBuildStats, visible, packedScreenSpace, elapsedMs) {
  const visibleCount = Array.isArray(visible) ? visible.length : 0;
  const packed = packedScreenSpace?.packed;

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
        : (packed instanceof Float32Array ? packed.length : 0),
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

function buildPackedLines(buildStats, drawPathSelection, drawStats) {
  return [
    `packedVisiblePathEnabled=${!!buildStats?.packedVisiblePathEnabled}`,
    `packedVisiblePathUsed=${!!buildStats?.packedVisiblePathUsed}`,
    `packedVisiblePath=${buildStats?.packedVisiblePath ?? 'none'}`,
    `packedVisibleCount=${buildStats?.packedVisibleCount ?? 0}`,
    `packedVisibleLength=${buildStats?.packedVisibleLength ?? 0}`,
    `packedVisibleFloatsPerItem=${buildStats?.packedVisibleFloatsPerItem ?? 0}`,
    `packedAlphaSource=colorAlpha[3]`,
    `requestedDrawPath=${drawPathSelection.requestedPath}`,
    `actualDrawPath=${drawPathSelection.actualPath}`,
    `drawPathFallbackReason=${drawPathSelection.fallbackReason}`,
    `packedUploadBytes=${drawStats?.packedUploadBytes ?? 0}`,
    `packedUploadCount=${drawStats?.packedUploadCount ?? 0}`,
    `packedUploadLength=${drawStats?.packedUploadLength ?? 0}`,
    `packedUploadCapacityBytes=${drawStats?.packedUploadCapacityBytes ?? 0}`,
    `packedUploadReusedCapacity=${!!drawStats?.packedUploadReusedCapacity}`,
    `packedDirectDraw=${!!drawStats?.packedDirectDraw}`,
    `packedDirectLayoutVersion=${drawStats?.packedDirectLayoutVersion ?? 0}`,
    `packedDirectStrideBytes=${drawStats?.packedDirectStrideBytes ?? 0}`,
    `packedDirectAttributeCount=${drawStats?.packedDirectAttributeCount ?? 0}`,
    `packedDirectOffsets=${drawStats?.packedDirectOffsets ?? ''}`,
    `packedInterleavedBound=${!!drawStats?.packedInterleavedBound}`,
    `legacyExpandedArraysBuilt=${!!drawStats?.legacyExpandedArraysBuilt}`
  ];
}

function buildSampleLines(legacySample, packedSample) {
  return [
    `legacySampleCenterPx=${legacySample ? fmtArr(legacySample.centerPx, 3) : 'none'}  packedSampleCenterPx=${packedSample ? fmtArr(packedSample.centerPx, 3) : 'none'}`,
    `legacySampleRadiusPx=${legacySample ? fmtNum(legacySample.radiusPx, 3) : 'none'}  packedSampleRadiusPx=${packedSample ? fmtNum(packedSample.radiusPx, 3) : 'none'}`,
    `legacySampleColorAlpha=${legacySample ? fmtArr(legacySample.colorAlpha, 4) : 'none'}`,
    `packedSampleColorAlpha=${packedSample ? fmtArr(packedSample.colorAlpha, 4) : 'none'}`,
    `legacySampleConic=${legacySample ? fmtArr(legacySample.conic, 4) : 'none'}`,
    `packedSampleConic=${packedSample ? fmtArr(packedSample.conic, 4) : 'none'}`,
    `packedSampleReserved=${packedSample ? fmtNum(packedSample.reserved, 4) : 'none'}`,
    `packedSampleMisc=${packedSample ? fmtArr(packedSample.misc, 2) : 'none'}`
  ];
}

function buildGpuScreenExecutionLines(gpuScreenExecutionSummary) {
  if (!gpuScreenExecutionSummary) return [];
  return [
    `gpuScreenDraw=${!!gpuScreenExecutionSummary.gpuScreenDraw}`,
    `gpuScreenReady=${!!gpuScreenExecutionSummary.gpuScreenReady}`,
    `gpuScreenReason=${gpuScreenExecutionSummary.gpuScreenReason ?? 'unknown'}`,
    `gpuScreenDrawCount=${gpuScreenExecutionSummary.gpuScreenDrawCount ?? 0}`,
    `gpuScreenActualPath=${gpuScreenExecutionSummary.gpuScreenActualPath ?? 'gpu-screen'}`,
    `gpuScreenSourcePath=${gpuScreenExecutionSummary.gpuScreenSourcePath ?? 'none'}`,
    `gpuScreenReferencePath=${gpuScreenExecutionSummary.gpuScreenReferencePath ?? 'packed-cpu'}`
  ];
}

function selectGpuScreenSourceSpace(gpu, visible, packedCpuScreenSpace) {
  const context = ensureGpuScreenSourceContext(gpu);

  try {
    const experimental = buildPackedGpuPrepScreenSpaceWithContext(context, visible, {});
    if (experimental?.packed instanceof Float32Array) {
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

function executePerTileLegacyDraw({
  gl,
  gpu,
  canvas,
  focusTileRects,
  perTileDrawBatches,
  drawPathSelection
}) {
  const rectMap = new Map(focusTileRects.map((item) => [item.tileId, item.rect]));

  const executionSummary = renderPerTileBatches(
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

  disableTileScissor(gl);

  executionSummary.requestedDrawPath = drawPathSelection.requestedPath;
  executionSummary.actualDrawPath = GPU_DRAW_PATH_LEGACY;
  executionSummary.drawPathFallbackReason = drawPathSelection.fallbackReason;

  return {
    executionSummary,
    packedUploadSummary: {
      packedUploadBytes: 0,
      packedUploadCount: 0,
      packedUploadLength: 0,
      packedUploadCapacityBytes: 0,
      packedUploadReusedCapacity: false,
      packedDirectDraw: false,
      packedDirectLayoutVersion: 0,
      packedDirectStrideBytes: 0,
      packedDirectAttributeCount: 0,
      packedDirectOffsets: '',
      packedInterleavedBound: false,
      legacyExpandedArraysBuilt: true
    },
    directPackedDrawInfo: null,
    gpuScreenDrawInfo: null
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
    const emptyInfo = 'GPU Step34 viewer\nNo scene loaded.';
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
      hasPackedScreenSpace: !!packedScreenSpace?.packed,
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
    ? executePerTileLegacyDraw({
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

  const effectiveDrawData = mode.drawSelectedOnly
    ? { nDraw: effectiveTileSummary ? effectiveTileSummary.totalTileDrawCount : 0 }
    : {
        nDraw:
          drawPathSelection.actualPath === GPU_DRAW_PATH_PACKED
            ? (directPackedDrawInfo?.drawCount ?? packedScreenSpace?.packedCount ?? 0)
            : drawPathSelection.actualPath === GPU_DRAW_PATH_GPU_SCREEN
              ? (gpuScreenDrawInfo?.drawCount ?? 0)
              : (legacyDrawData?.nDraw ?? 0)
      };

  const packedDirectResourceSummary = summarizePackedDirectResources(gpu);
  const gpuScreenSummary = summarizeGpuScreenDrawState(gpu);

  const gpuScreenComparisonSummary = mergeGpuScreenComparisonSummary(
    gpuScreenSourceInfo.sourceComparisonSummary ?? null,
    gpuScreenDrawInfo?.gpuScreenComparisonSummary ?? null
  );

  const gpuScreenExecutionSummary =
    gpuScreenDrawInfo?.gpuScreenExecutionSummary ?? null;

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
      ...packedUploadSummary,
      ...packedDirectResourceSummary
    }
  });

  drawStats.drawFraction = computeDrawFraction(drawStats.visibleCount, drawStats.drawCount);

  const extraLines = buildGpuDebugExtraLines({
    buildConfig,
    buildStats,
    drawStats,
    mode,
    focusTileIds,
    focusTileRects,
    ui,
    gpuScreenSummary,
    gpuScreenComparisonSummary
  });

  if (gpuScreenSourceInfo?.sourceFallbackReason && gpuScreenSourceInfo.sourceFallbackReason !== 'none') {
    extraLines.push(`gpuScreenSourceFallbackReason=${gpuScreenSourceInfo.sourceFallbackReason}`);
  }

  const packedLines = buildPackedLines(buildStats, drawPathSelection, drawStats);
  const sampleLines = buildSampleLines(legacySample, packedSample);
  const gpuScreenExecutionLines = buildGpuScreenExecutionLines(gpuScreenExecutionSummary);

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
    stepLabel: 'GPU Step34',
    stepNotes: [
      'transform executor is now the truth source for requested and actual transform paths',
      'screen-space builder keeps transform state without reinterpretation',
      'renderer stays thin and only forwards source and transform summaries to debug output',
      'this step prepares future GPU transform ownership without changing the draw contract'
    ],
    tileSummary,
    avgRefsPerVisible,
    drawStats,
    tileSelectionText,
    tileDebugText,
    extraLines: [
      ...extraLines,
      ...packedLines,
      ...sampleLines,
      ...gpuScreenExecutionLines
    ]
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
    drawPathSummary: drawPathSelection,
    gpuScreenSummary,
    gpuScreenComparisonSummary,
    gpuScreenExecutionSummary,
    gpuScreenSourceInfo
  };
}
