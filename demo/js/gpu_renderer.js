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
  hasExplicitCpuPackedCompatibilityBridge,
  hasGpuResidentPackedScreenSpace,
  resolvePackedScreenSpaceContract,
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

function hasRenderablePackedScreenSpace(screenSpace) {
  return hasGpuResidentPackedScreenSpace(screenSpace) || hasExplicitCpuPackedCompatibilityBridge(screenSpace);
}

function resolveDrawFallbackContract(drawPathSelection) {
  const requestedPath = drawPathSelection?.requestedPath ?? 'none';
  const actualPath = drawPathSelection?.actualPath ?? 'none';
  const fallbackReason = drawPathSelection?.fallbackReason ?? 'none';

  if (fallbackReason === 'none' || requestedPath === actualPath) return 'none';
  if (actualPath === GPU_DRAW_PATH_LEGACY) return 'legacy-draw-fallback';
  if (actualPath === GPU_DRAW_PATH_PACKED) return 'packed-direct-compatibility-fallback';
  if (actualPath === GPU_DRAW_PATH_GPU_SCREEN) return 'gpu-screen-compatibility-fallback';
  return 'draw-path-fallback';
}

function buildFallbackContractSummary({ sourceInfo, comparisonSummary, drawPathSelection }) {
  const sourceContract = sourceInfo?.sourceFallbackContract ?? 'none';
  const transformContract = comparisonSummary?.transformFallbackContract ?? 'none';
  const drawContract = resolveDrawFallbackContract(drawPathSelection);
  const parts = [];
  if (sourceContract !== 'none') parts.push(`source:${sourceContract}`);
  if (transformContract !== 'none') parts.push(`transform:${transformContract}`);
  if (drawContract !== 'none') parts.push(`draw:${drawContract}`);
  return parts.length > 0 ? parts.join(' | ') : 'none';
}

function buildFallbackReasonSummary({ sourceInfo, comparisonSummary, drawPathSelection }) {
  const sourceReason = sourceInfo?.sourceFallbackReason ?? 'none';
  const transformReason = comparisonSummary?.transformFallbackReason ?? 'none';
  const drawReason = drawPathSelection?.fallbackReason ?? 'none';
  const parts = [];
  if (sourceReason !== 'none') parts.push(`source:${sourceReason}`);
  if (transformReason !== 'none') parts.push(`transform:${transformReason}`);
  if (drawReason !== 'none') parts.push(`draw:${drawReason}`);
  return parts.length > 0 ? parts.join(' | ') : 'none';
}

function buildFallbackStageSummary({ sourceInfo, comparisonSummary, drawPathSelection }) {
  const sourceActive = (sourceInfo?.sourceFallbackContract ?? 'none') !== 'none';
  const transformActive = (comparisonSummary?.transformFallbackContract ?? 'none') !== 'none';
  const drawActive = resolveDrawFallbackContract(drawPathSelection) !== 'none';
  const stages = [];
  if (sourceActive) stages.push('source');
  if (transformActive) stages.push('transform');
  if (drawActive) stages.push('draw');
  return stages.length > 0 ? stages.join(' -> ') : 'none';
}

function buildGpuFallbackSummary({ sourceInfo, comparisonSummary, drawPathSelection }) {
  const contractSummary = buildFallbackContractSummary({
    sourceInfo,
    comparisonSummary,
    drawPathSelection
  });
  const reasonSummary = buildFallbackReasonSummary({
    sourceInfo,
    comparisonSummary,
    drawPathSelection
  });
  const stageSummary = buildFallbackStageSummary({
    sourceInfo,
    comparisonSummary,
    drawPathSelection
  });

  return {
    active: contractSummary !== 'none' || reasonSummary !== 'none' || stageSummary !== 'none',
    stageSummary,
    reasonSummary,
    contractSummary
  };
}

function buildGpuCompatibilityBridgeSummary({ sourceInfo, comparisonSummary, drawPathSelection }) {
  const sourceContract = sourceInfo?.sourceFallbackContract ?? 'none';
  const transformContract = comparisonSummary?.transformFallbackContract ?? 'none';
  const drawContract = resolveDrawFallbackContract(drawPathSelection);
  const sourceReason = sourceInfo?.sourceFallbackReason ?? 'none';
  const transformReason = comparisonSummary?.transformFallbackReason ?? 'none';
  const drawReason = drawPathSelection?.fallbackReason ?? 'none';

  const sourceBridge = sourceContract === 'cpu-packed-compatibility-bridge';
  const transformBridge = transformContract === 'cpu-packed-transform-compatibility-fallback';
  const drawBridge =
    drawContract === 'packed-direct-compatibility-fallback' ||
    drawContract === 'gpu-screen-compatibility-fallback';

  const stages = [];
  const reasons = [];
  if (sourceBridge) {
    stages.push('source');
    reasons.push(`source:${sourceReason}`);
  }
  if (transformBridge) {
    stages.push('transform');
    reasons.push(`transform:${transformReason}`);
  }
  if (drawBridge) {
    stages.push('draw');
    reasons.push(`draw:${drawReason}`);
  }

  const gpuRetainedStages = [];
  if (comparisonSummary?.sourceContract === 'gpu-resident-normal') gpuRetainedStages.push('source');
  if (comparisonSummary?.transformFallbackContract === 'none') gpuRetainedStages.push('transform');
  if (drawContract === 'none') gpuRetainedStages.push('draw');

  return {
    active: sourceBridge || transformBridge || drawBridge,
    stageSummary: stages.length > 0 ? stages.join(' -> ') : 'none',
    reasonSummary: reasons.length > 0 ? reasons.join(' | ') : 'none',
    gpuRetainedSummary: gpuRetainedStages.length > 0 ? gpuRetainedStages.join(' -> ') : 'none'
  };
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

function buildFrameGpuThroughputSummary({
  transformThroughputSummary,
  drawThroughputSummary
}) {
  const transformDispatchCount = Number.isFinite(transformThroughputSummary?.totalDispatchCount)
    ? Math.max(0, transformThroughputSummary.totalDispatchCount | 0)
    : 0;
  const drawDispatchCount = Number.isFinite(drawThroughputSummary?.sharedDispatchCount)
    ? Math.max(0, drawThroughputSummary.sharedDispatchCount | 0)
    : 0;
  const transformBatchCount = Number.isFinite(transformThroughputSummary?.batchCount)
    ? Math.max(0, transformThroughputSummary.batchCount | 0)
    : 0;
  const drawCallCount = Number.isFinite(drawThroughputSummary?.drawCallCount)
    ? Math.max(0, drawThroughputSummary.drawCallCount | 0)
    : 0;

  let bottleneckStage = 'balanced-gpu-path';
  if (transformDispatchCount > drawDispatchCount || transformBatchCount > drawCallCount) {
    bottleneckStage = 'transform-throughput-pressure';
  } else if (drawDispatchCount > transformDispatchCount || drawCallCount > transformBatchCount) {
    bottleneckStage = 'draw-throughput-pressure';
  }

  return {
    transformBatchCount,
    transformDispatchCount,
    transformDispatchMode: transformThroughputSummary?.dispatchMode ?? 'none',
    drawCallCount,
    drawDispatchCount,
    drawDispatchMode: drawThroughputSummary?.sharedDispatchMode ?? 'none',
    drawUsesGpuResidentPayload: !!drawThroughputSummary?.usesGpuResidentPayload,
    drawMergePolicySelectedPath: drawThroughputSummary?.sharedMergePolicySelectedPath ?? 'none',
    drawMergePolicyReason: drawThroughputSummary?.sharedMergePolicyReason ?? 'none',
    drawMergePolicyEstimatedCopyCount: drawThroughputSummary?.sharedMergePolicyEstimatedCopyCount ?? 0,
    drawMergePolicyEstimatedDispatchSavings: drawThroughputSummary?.sharedMergePolicyEstimatedDispatchSavings ?? 0,
    bottleneckStage
  };
}

function selectGpuScreenSourceSpace(gpu, visible, packedCpuScreenSpace) {
  const context = ensureGpuScreenSourceContext(gpu);

  try {
    const experimental = buildPackedGpuPrepScreenSpaceWithContext(context, visible, {
      gl: gpu?.gl ?? null
    });
    if (hasGpuResidentPackedScreenSpace(experimental)) {
      return {
        sourceSpace: experimental,
        sourceSummary: summarizePackedScreenSpace(experimental),
        sourceComparisonSummary: summarizePackedScreenSpaceComparison(experimental),
        sourceFallbackReason: 'none',
        sourceContract: resolvePackedScreenSpaceContract(experimental),
        sourceFallbackContract: 'none'
      };
    }
    if (hasExplicitCpuPackedCompatibilityBridge(experimental)) {
      return {
        sourceSpace: packedCpuScreenSpace,
        sourceSummary: summarizePackedScreenSpace(packedCpuScreenSpace),
        sourceComparisonSummary: summarizePackedScreenSpaceComparison(packedCpuScreenSpace),
        sourceFallbackReason: 'gpu-source-produced-cpu-packed-fallback-to-packed-cpu',
        sourceContract: resolvePackedScreenSpaceContract(packedCpuScreenSpace),
        sourceFallbackContract: 'cpu-packed-compatibility-bridge'
      };
    }
  } catch (err) {
    console.warn('gpu-screen source build failed, falling back to packed-cpu source', err);
  }

  return {
    sourceSpace: packedCpuScreenSpace,
    sourceSummary: summarizePackedScreenSpace(packedCpuScreenSpace),
    sourceComparisonSummary: summarizePackedScreenSpaceComparison(packedCpuScreenSpace),
    sourceFallbackReason: 'gpu-source-build-fallback-to-packed-cpu',
    sourceContract: resolvePackedScreenSpaceContract(packedCpuScreenSpace),
    sourceFallbackContract: 'cpu-packed-compatibility-bridge'
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
    const emptyInfo = 'GPU Step66 viewer\nNo scene loaded.';
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
    drawThroughputSummary,
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
  const transformThroughputSummary =
    gpuScreenComparisonSummary?.transformThroughputSummary ??
    gpuScreenSourceInfo?.sourceSpace?.transformSummary?.transformThroughputSummary ??
    packedScreenSpace?.transformSummary?.transformThroughputSummary ??
    null;
  const frameGpuThroughputSummary = buildFrameGpuThroughputSummary({
    transformThroughputSummary,
    drawThroughputSummary
  });

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
    transformThroughputSummary,
    drawThroughputSummary,
    frameGpuThroughputSummary,
    legacySample,
    packedSample
  });

  if (gpuScreenSourceInfo?.sourceFallbackReason && gpuScreenSourceInfo.sourceFallbackReason !== 'none') {
    extraLines.push(`gpuScreenSourceFallbackReason=${gpuScreenSourceInfo.sourceFallbackReason}`);
  }
  if (gpuScreenSourceInfo?.sourceContract) {
    extraLines.push(`gpuScreenSourceContract=${gpuScreenSourceInfo.sourceContract}`);
  }
  if (gpuScreenSourceInfo?.sourceFallbackContract && gpuScreenSourceInfo.sourceFallbackContract !== 'none') {
    extraLines.push(`gpuScreenSourceFallbackContract=${gpuScreenSourceInfo.sourceFallbackContract}`);
  }
  if (
    gpuScreenComparisonSummary?.transformFallbackContract &&
    gpuScreenComparisonSummary.transformFallbackContract !== 'none'
  ) {
    extraLines.push(`gpuScreenTransformFallbackContract=${gpuScreenComparisonSummary.transformFallbackContract}`);
  }
  const drawFallbackContract = resolveDrawFallbackContract(drawPathSelection);
  if (drawFallbackContract !== 'none') {
    extraLines.push(`drawPathFallbackContract=${drawFallbackContract}`);
  }
  const gpuFallbackSummary = buildGpuFallbackSummary({
    sourceInfo: gpuScreenSourceInfo,
    comparisonSummary: gpuScreenComparisonSummary,
    drawPathSelection
  });
  const gpuCompatibilityBridgeSummary = buildGpuCompatibilityBridgeSummary({
    sourceInfo: gpuScreenSourceInfo,
    comparisonSummary: gpuScreenComparisonSummary,
    drawPathSelection
  });
  extraLines.push(`gpuFallbackActive=${gpuFallbackSummary.active}`);
  extraLines.push(`gpuFallbackStageSummary=${gpuFallbackSummary.stageSummary}`);
  extraLines.push(`gpuFallbackReasonSummary=${gpuFallbackSummary.reasonSummary}`);
  extraLines.push(`gpuFallbackContractSummary=${gpuFallbackSummary.contractSummary}`);
  extraLines.push(`gpuCompatibilityBridgeActive=${gpuCompatibilityBridgeSummary.active}`);
  extraLines.push(`gpuCompatibilityBridgeStageSummary=${gpuCompatibilityBridgeSummary.stageSummary}`);
  extraLines.push(`gpuCompatibilityBridgeReasonSummary=${gpuCompatibilityBridgeSummary.reasonSummary}`);
  extraLines.push(`gpuCompatibilityBridgeGpuRetainedSummary=${gpuCompatibilityBridgeSummary.gpuRetainedSummary}`);

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
    stepLabel: 'GPU Step66',
    stepNotes: [
      'transform executor owns transformBatchSummary and downstream code forwards it without reinterpretation',
      'transform truth and draw truth still flow into frame-level GPU throughput summaries so the main-path bottleneck stays readable without reinterpreting executor-owned contracts',
      'transform backend still advertises a preferred GPU batch size based on successful single-texture-copy-pass history, and draw now picks merged-atlas versus multi-payload shared dispatch through a workload-aware atlas policy',
      'debug output now shows transform throughput, draw throughput, frame-level bottleneck hints, merge policy reasons, atlas shape, and shared merge-copy costs while preserving existing truth-source metrics',
      'gpu resident payload draw still shares bind and setup work between gpu-screen and packed direct through the shared texture consumer path, and Step66 makes the merge path conditional on dispatch savings versus copy cost',
      'gpu resident payload remains the explicit normal source contract, while cpu packed stays behind explicit compatibility-bridge contracts without changing public draw contracts',
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
    gpuFallbackSummary,
    gpuCompatibilityBridgeSummary,
    gpuScreenSummary,
    gpuScreenComparisonSummary,
    gpuScreenExecutionSummary,
    gpuScreenSourceInfo,
    transformThroughputSummary,
    drawThroughputSummary,
    frameGpuThroughputSummary
  };

  return result;
}
