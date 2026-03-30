import { computeGaussianState, computeScreenSplat } from './rot4d_math.js';
import { evalSHColor } from './sh_eval.js';
import { clampInt, computeTileRangeFromAABB } from './gpu_tile_utils.js';
import {
  buildTemporalSortedIndex,
  estimateTemporalWindow,
  getTemporalCandidateRange,
  buildTemporalCandidateIndices,
  summarizeTemporalRange,
  getTemporalIndexUiOptions
} from './gpu_temporal_index_utils.js';

export function getVisibleBuildConfig(ui, interactionOverride = null) {
  const temporalIndexOptions = getTemporalIndexUiOptions(ui);

  const baseConfig = {
    renderScale: parseFloat(ui.renderScaleSlider.value),
    stride: parseInt(ui.strideSlider.value, 10),
    maxVisible: parseInt(ui.maxVisibleSlider.value, 10),
    timestamp: parseFloat(ui.timeSlider.value),
    scalingModifier: parseFloat(ui.splatScaleSlider.value),
    sigmaScale: parseFloat(ui.sigmaScaleSlider.value),
    prefilterVar: parseFloat(ui.prefilterVarSlider.value),
    useSH: !!ui.useSHCheck.checked,
    useRot4d: !!ui.useRot4dCheck.checked,
    useNativeRot4d: !!ui.useNativeRot4dCheck.checked,
    useNativeMarginal: !!ui.useNativeMarginalCheck.checked,
    forceSh3d: !!ui.forceSh3dCheck.checked,
    timeDuration: parseFloat(ui.timeDurationSlider.value),
    interactionActive: false,
    ...temporalIndexOptions
  };

  if (!interactionOverride || !interactionOverride.interactionActive) {
    return baseConfig;
  }

  return {
    ...baseConfig,
    interactionActive: true,
    stride: interactionOverride.stride,
    maxVisible: interactionOverride.maxVisible,
    renderScale: interactionOverride.renderScale
  };
}

function getTemporalSigma(raw, i, sigmaScale = 1.0) {
  if (!raw || !raw.scale_t) return Infinity;
  const s = raw.scale_t[i];
  if (!Number.isFinite(s)) return Infinity;
  const sigma = s * sigmaScale;
  return Number.isFinite(sigma) && sigma > 0 ? sigma : Infinity;
}

function passesTemporalCulling(raw, i, timestamp, sigmaScale = 1.0, sigmaThreshold = 3.0) {
  if (!raw || !raw.t || !raw.scale_t) return true;

  const t0 = raw.t[i];
  if (!Number.isFinite(t0)) return true;

  const sigmaT = getTemporalSigma(raw, i, sigmaScale);
  if (!Number.isFinite(sigmaT)) return true;

  const dt = Math.abs(timestamp - t0);
  return dt <= sigmaThreshold * sigmaT;
}

function buildFallbackStrideIndices(raw, stride) {
  if (!raw || !raw.N) return new Uint32Array(0);
  const s = Math.max(1, stride | 0);
  const count = Math.ceil(raw.N / s);
  const out = new Uint32Array(count);
  let w = 0;
  for (let i = 0; i < raw.N; i += s) {
    out[w++] = i;
  }
  return out;
}

function buildCandidateIndices({
  raw,
  timestamp,
  stride,
  sigmaScale,
  temporalSigmaThreshold,
  useTemporalIndex,
  useTemporalIndexCache,
  temporalWindowMode,
  fixedWindowRadius
}) {
  const total = raw ? raw.N : 0;

  if (!raw || !raw.t || !raw.scale_t || !useTemporalIndex) {
    return {
      candidateIndices: buildFallbackStrideIndices(raw, stride),
      temporalWindow: {
        maxSigmaT: Infinity,
        medianSigmaT: Infinity,
        meanSigmaT: Infinity,
        p90SigmaT: Infinity,
        windowRadius: Infinity,
        mode: useTemporalIndex ? (temporalWindowMode || 'max') : 'disabled',
        cacheHit: false,
        builtThisFrame: false
      },
      rangeInfo: {
        start: 0,
        end: total,
        count: total
      },
      rangeSummary: {
        totalCount: total,
        rangeCount: total,
        candidateCount: total > 0 ? Math.ceil(total / Math.max(1, stride | 0)) : 0,
        rangeFraction: 1,
        candidateFraction: 1
      },
      temporalIndexDebug: {
        enabled: !!useTemporalIndex,
        cacheHit: false,
        builtThisFrame: false,
        totalCount: total,
        tMin: NaN,
        tMax: NaN
      }
    };
  }

  const temporalIndexResult = buildTemporalSortedIndex(raw, {
    useCache: useTemporalIndexCache
  });
  const temporalIndex = temporalIndexResult.indexData;

  const temporalWindow = estimateTemporalWindow(
    raw,
    sigmaScale,
    temporalSigmaThreshold,
    {
      useCache: useTemporalIndexCache,
      mode: temporalWindowMode,
      fixedWindowRadius
    }
  );

  const rangeInfo = getTemporalCandidateRange(
    temporalIndex,
    timestamp,
    temporalWindow.windowRadius
  );
  const candidateIndices = buildTemporalCandidateIndices(temporalIndex, rangeInfo, stride);
  const rangeSummary = summarizeTemporalRange(temporalIndex, rangeInfo, candidateIndices);

  return {
    candidateIndices,
    temporalWindow,
    rangeInfo,
    rangeSummary,
    temporalIndexDebug: {
      enabled: true,
      cacheHit: temporalIndexResult.cacheHit,
      builtThisFrame: temporalIndexResult.builtThisFrame,
      totalCount: temporalIndex ? temporalIndex.count : 0,
      tMin: temporalIndex ? temporalIndex.tMin : NaN,
      tMax: temporalIndex ? temporalIndex.tMax : NaN
    }
  };
}

export async function buildVisibleSplats({
  raw,
  camera,
  canvasWidth,
  canvasHeight,
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
  tokenRef = null,
  frameToken = null,
  tileGrid = null,
  temporalSigmaThreshold = 3.0,
  interactionActive = false,
  useTemporalIndex = true,
  useTemporalIndexCache = true,
  temporalWindowMode = 'max',
  fixedWindowRadius = 0.5
}) {
  if (!raw) {
    return {
      visible: [],
      renderW: 0,
      renderH: 0,
      sx: 1,
      sy: 1,
      activeTileBox: null,
      buildStats: {
        accepted: 0,
        processed: 0,
        culled: 0,
        temporalRejected: 0,
        temporalPassed: 0,
        temporalCullRatio: 0,
        temporalIndexEnabled: false,
        temporalIndexCacheEnabled: false,
        temporalIndexCacheHit: false,
        temporalIndexBuiltThisFrame: false,
        temporalIndexTotalCount: 0,
        temporalIndexTMin: NaN,
        temporalIndexTMax: NaN,
        temporalIndexRangeCount: 0,
        temporalIndexCandidateCount: 0,
        temporalIndexRangeFraction: 0,
        temporalIndexCandidateFraction: 0,
        temporalWindowMode: 'disabled',
        temporalWindowRadius: Infinity,
        temporalWindowCacheHit: false,
        temporalWindowBuiltThisFrame: false,
        interactionActive: false
      }
    };
  }

  const renderW = Math.max(1, Math.round(canvasWidth * renderScale));
  const renderH = Math.max(1, Math.round(canvasHeight * renderScale));
  const sx = canvasWidth / renderW;
  const sy = canvasHeight / renderH;

  const flags = {
    nativeRot4d: useNativeRot4d,
    nativeMarginal: useNativeMarginal
  };

  const candidateInfo = buildCandidateIndices({
    raw,
    timestamp,
    stride,
    sigmaScale,
    temporalSigmaThreshold,
    useTemporalIndex,
    useTemporalIndexCache,
    temporalWindowMode,
    fixedWindowRadius
  });

  const candidateIndices = candidateInfo.candidateIndices;

  const visible = [];
  let processed = 0;
  let culled = 0;
  let temporalRejected = 0;
  let temporalPassed = 0;

  let minTileX = tileGrid ? tileGrid.tileCols : 0;
  let minTileY = tileGrid ? tileGrid.tileRows : 0;
  let maxTileX = -1;
  let maxTileY = -1;

  for (let k = 0; k < candidateIndices.length; k++) {
    const i = candidateIndices[k];
    processed++;

    if (!passesTemporalCulling(raw, i, timestamp, sigmaScale, temporalSigmaThreshold)) {
      temporalRejected++;
      culled++;
      continue;
    }
    temporalPassed++;

    const gs = computeGaussianState(
      raw,
      i,
      timestamp,
      scalingModifier,
      sigmaScale,
      prefilterVar,
      useRot4d,
      flags
    );
    if (!gs) {
      culled++;
      continue;
    }

    const color = evalSHColor(
      raw,
      i,
      camPos,
      gs.pos,
      timestamp,
      timeDuration,
      useSH,
      forceSh3d
    );

    const splat = computeScreenSplat(
      camera,
      gs.pos,
      gs.cov3,
      gs.opacity,
      renderW,
      renderH
    );
    if (!splat) {
      culled++;
      continue;
    }

    const px = splat.px * sx;
    const py = splat.py * sy;
    const radius = Math.max(1.0, splat.radius * Math.max(sx, sy));

    const minX = clampInt(Math.floor(px - radius), 0, canvasWidth - 1);
    const maxX = clampInt(Math.ceil(px + radius), 0, canvasWidth - 1);
    const minY = clampInt(Math.floor(py - radius), 0, canvasHeight - 1);
    const maxY = clampInt(Math.ceil(py + radius), 0, canvasHeight - 1);

    let tileRange = null;
    if (tileGrid) {
      tileRange = computeTileRangeFromAABB(
        [minX, minY, maxX, maxY],
        tileGrid.tileCols,
        tileGrid.tileRows,
        tileGrid.tileSize
      );
      minTileX = Math.min(minTileX, tileRange[0]);
      minTileY = Math.min(minTileY, tileRange[1]);
      maxTileX = Math.max(maxTileX, tileRange[2]);
      maxTileY = Math.max(maxTileY, tileRange[3]);
    }

    visible.push({
      srcIndex: i,
      px,
      py,
      radius,
      depth: splat.depth,
      opacity: splat.opacity,
      color,
      conic: [
        splat.conic[0] / (sx * sx),
        splat.conic[1] / (sx * sy),
        splat.conic[2] / (sy * sy)
      ],
      aabb: [minX, minY, maxX, maxY],
      tileRange
    });

    if (visible.length >= maxVisible) break;

    if ((visible.length & 2047) === 0) {
      await new Promise(r => setTimeout(r, 0));
      if (tokenRef && frameToken !== null && frameToken !== tokenRef.value) {
        return null;
      }
    }
  }

  visible.sort((a, b) => b.depth - a.depth);

  let activeTileBox = null;
  if (tileGrid && maxTileX >= minTileX && maxTileY >= minTileY) {
    activeTileBox = [minTileX, minTileY, maxTileX, maxTileY];
  }

  return {
    visible,
    renderW,
    renderH,
    sx,
    sy,
    activeTileBox,
    buildStats: {
      accepted: visible.length,
      processed,
      culled,
      temporalRejected,
      temporalPassed,
      temporalCullRatio: processed > 0 ? (temporalRejected / processed) : 0,
      temporalIndexEnabled: !!useTemporalIndex,
      temporalIndexCacheEnabled: !!useTemporalIndexCache,
      temporalIndexCacheHit: candidateInfo.temporalIndexDebug.cacheHit,
      temporalIndexBuiltThisFrame: candidateInfo.temporalIndexDebug.builtThisFrame,
      temporalIndexTotalCount: candidateInfo.temporalIndexDebug.totalCount,
      temporalIndexTMin: candidateInfo.temporalIndexDebug.tMin,
      temporalIndexTMax: candidateInfo.temporalIndexDebug.tMax,
      temporalIndexRangeCount: candidateInfo.rangeSummary.rangeCount,
      temporalIndexCandidateCount: candidateInfo.rangeSummary.candidateCount,
      temporalIndexRangeFraction: candidateInfo.rangeSummary.rangeFraction,
      temporalIndexCandidateFraction: candidateInfo.rangeSummary.candidateFraction,
      temporalWindowMode: candidateInfo.temporalWindow.mode,
      temporalWindowRadius: candidateInfo.temporalWindow.windowRadius,
      temporalWindowCacheHit: !!candidateInfo.temporalWindow.cacheHit,
      temporalWindowBuiltThisFrame: !!candidateInfo.temporalWindow.builtThisFrame,
      interactionActive
    }
  };
}
