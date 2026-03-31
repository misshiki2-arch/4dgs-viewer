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
import {
  buildTemporalBucketIndex,
  getTemporalBucketCandidateRange,
  buildTemporalBucketCandidateIndices,
  summarizeTemporalBucketRange,
  getTemporalBucketUiOptions
} from './gpu_temporal_bucket_utils.js';

function deriveTemporalPrefilterMode(ui, temporalIndexOptions, temporalBucketOptions) {
  if (ui?.temporalPrefilterModeSelect) return ui.temporalPrefilterModeSelect.value;
  if (temporalBucketOptions.useTemporalBucket && temporalIndexOptions.useTemporalIndex) return 'hybrid';
  if (temporalBucketOptions.useTemporalBucket) return 'bucket';
  if (temporalIndexOptions.useTemporalIndex) return 'sorted';
  return 'off';
}

export function getVisibleBuildConfig(ui, interactionOverride = null) {
  const temporalIndexOptions = getTemporalIndexUiOptions(ui);
  const temporalBucketOptions = getTemporalBucketUiOptions(ui);
  const temporalPrefilterMode = deriveTemporalPrefilterMode(ui, temporalIndexOptions, temporalBucketOptions);

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
    playbackActive: false,
    qualityOverrideActive: false,
    qualityOverrideReason: 'none',
    temporalPrefilterMode,
    ...temporalIndexOptions,
    ...temporalBucketOptions
  };

  if (!interactionOverride) return baseConfig;
  return { ...baseConfig, ...interactionOverride };
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
  return Math.abs(timestamp - t0) <= sigmaThreshold * sigmaT;
}

function buildFallbackStrideIndices(raw, stride) {
  if (!raw || !raw.N) return new Uint32Array(0);
  const s = Math.max(1, stride | 0);
  const count = Math.ceil(raw.N / s);
  const out = new Uint32Array(count);
  let w = 0;
  for (let i = 0; i < raw.N; i += s) out[w++] = i;
  return out;
}

function filterCandidateIndicesByTemporalWindow(raw, candidateIndices, timestamp, windowRadius) {
  if (!raw || !raw.t || !candidateIndices || candidateIndices.length === 0) {
    return new Uint32Array(0);
  }
  if (!Number.isFinite(windowRadius)) return candidateIndices;

  const out = new Uint32Array(candidateIndices.length);
  let w = 0;
  for (let k = 0; k < candidateIndices.length; k++) {
    const i = candidateIndices[k];
    const t = raw.t[i];
    if (!Number.isFinite(t)) continue;
    if (Math.abs(timestamp - t) <= windowRadius) out[w++] = i;
  }
  return out.slice(0, w);
}

function buildFallbackCandidateInfo(raw, stride, reason = 'off') {
  const total = raw ? raw.N : 0;
  return {
    candidateIndices: buildFallbackStrideIndices(raw, stride),
    candidateMode: reason,
    temporalWindow: {
      maxSigmaT: Infinity,
      medianSigmaT: Infinity,
      meanSigmaT: Infinity,
      p90SigmaT: Infinity,
      windowRadius: Infinity,
      mode: reason,
      cacheHit: false,
      builtThisFrame: false
    },
    rangeSummary: {
      totalCount: total,
      rangeCount: total,
      candidateCount: total > 0 ? Math.ceil(total / Math.max(1, stride | 0)) : 0,
      rangeFraction: 1,
      candidateFraction: 1
    },
    temporalIndexDebug: {
      enabled: false,
      cacheEnabled: false,
      cacheHit: false,
      builtThisFrame: false,
      totalCount: total,
      tMin: NaN,
      tMax: NaN
    },
    temporalBucketDebug: {
      enabled: false,
      cacheEnabled: false,
      cacheHit: false,
      builtThisFrame: false,
      bucketWidth: NaN,
      bucketRadius: 0,
      bucketCount: 0,
      bucketStart: 0,
      bucketEnd: -1,
      bucketSourceCount: 0,
      candidateCount: 0,
      bucketSourceFraction: 0,
      candidateFraction: 0,
      postWindowCandidateCount: 0,
      postWindowCandidateFraction: 0,
      tMin: NaN,
      tMax: NaN
    }
  };
}

function buildSortedCandidateInfo(params) {
  const {
    raw,
    timestamp,
    stride,
    sigmaScale,
    temporalSigmaThreshold,
    useTemporalIndexCache,
    temporalWindowMode,
    fixedWindowRadius
  } = params;

  const temporalIndexResult = buildTemporalSortedIndex(raw, { useCache: useTemporalIndexCache });
  const temporalIndex = temporalIndexResult.indexData;

  const temporalWindow = estimateTemporalWindow(
    raw,
    sigmaScale,
    temporalSigmaThreshold,
    { useCache: useTemporalIndexCache, mode: temporalWindowMode, fixedWindowRadius }
  );

  const rangeInfo = getTemporalCandidateRange(temporalIndex, timestamp, temporalWindow.windowRadius);
  const candidateIndices = buildTemporalCandidateIndices(temporalIndex, rangeInfo, stride);
  const rangeSummary = summarizeTemporalRange(temporalIndex, rangeInfo, candidateIndices);

  return {
    candidateIndices,
    candidateMode: 'sorted',
    temporalWindow,
    rangeSummary,
    temporalIndexDebug: {
      enabled: true,
      cacheEnabled: !!useTemporalIndexCache,
      cacheHit: temporalIndexResult.cacheHit,
      builtThisFrame: temporalIndexResult.builtThisFrame,
      totalCount: temporalIndex ? temporalIndex.count : 0,
      tMin: temporalIndex ? temporalIndex.tMin : NaN,
      tMax: temporalIndex ? temporalIndex.tMax : NaN
    },
    temporalBucketDebug: {
      enabled: false,
      cacheEnabled: false,
      cacheHit: false,
      builtThisFrame: false,
      bucketWidth: NaN,
      bucketRadius: 0,
      bucketCount: 0,
      bucketStart: 0,
      bucketEnd: -1,
      bucketSourceCount: 0,
      candidateCount: 0,
      bucketSourceFraction: 0,
      candidateFraction: 0,
      postWindowCandidateCount: 0,
      postWindowCandidateFraction: 0,
      tMin: NaN,
      tMax: NaN
    }
  };
}

function buildBucketCandidateInfo(params) {
  const {
    raw,
    timestamp,
    stride,
    useTemporalBucketCache,
    temporalBucketWidth,
    temporalBucketRadius
  } = params;

  const bucketResult = buildTemporalBucketIndex(raw, {
    bucketWidth: temporalBucketWidth,
    useCache: useTemporalBucketCache
  });
  const bucketData = bucketResult.bucketData;
  const bucketRange = getTemporalBucketCandidateRange(bucketData, timestamp, temporalBucketRadius);
  const candidateIndices = buildTemporalBucketCandidateIndices(bucketData, bucketRange, stride);
  const bucketSummary = summarizeTemporalBucketRange(bucketData, bucketRange, candidateIndices);

  return {
    candidateIndices,
    candidateMode: 'bucket',
    temporalWindow: {
      maxSigmaT: Infinity,
      medianSigmaT: Infinity,
      meanSigmaT: Infinity,
      p90SigmaT: Infinity,
      windowRadius: Infinity,
      mode: 'bucket',
      cacheHit: false,
      builtThisFrame: false
    },
    rangeSummary: {
      totalCount: bucketSummary.totalCount,
      rangeCount: bucketSummary.bucketSourceCount,
      candidateCount: bucketSummary.candidateCount,
      rangeFraction: bucketSummary.bucketSourceFraction,
      candidateFraction: bucketSummary.candidateFraction
    },
    temporalIndexDebug: {
      enabled: false,
      cacheEnabled: false,
      cacheHit: false,
      builtThisFrame: false,
      totalCount: raw ? raw.N : 0,
      tMin: NaN,
      tMax: NaN
    },
    temporalBucketDebug: {
      enabled: true,
      cacheEnabled: !!useTemporalBucketCache,
      cacheHit: bucketResult.cacheHit,
      builtThisFrame: bucketResult.builtThisFrame,
      bucketWidth: bucketData ? bucketData.bucketWidth : NaN,
      bucketRadius: temporalBucketRadius,
      bucketCount: bucketData ? bucketData.bucketCount : 0,
      bucketStart: bucketRange.bucketStart,
      bucketEnd: bucketRange.bucketEnd,
      bucketSourceCount: bucketSummary.bucketSourceCount,
      candidateCount: bucketSummary.candidateCount,
      bucketSourceFraction: bucketSummary.bucketSourceFraction,
      candidateFraction: bucketSummary.candidateFraction,
      postWindowCandidateCount: bucketSummary.candidateCount,
      postWindowCandidateFraction: bucketSummary.candidateFraction,
      tMin: bucketData ? bucketData.tMin : NaN,
      tMax: bucketData ? bucketData.tMax : NaN
    }
  };
}

function buildHybridCandidateInfo(params) {
  const {
    raw,
    timestamp,
    stride,
    sigmaScale,
    temporalSigmaThreshold,
    useTemporalIndexCache,
    temporalWindowMode,
    fixedWindowRadius,
    useTemporalBucketCache,
    temporalBucketWidth,
    temporalBucketRadius
  } = params;

  const bucketResult = buildTemporalBucketIndex(raw, {
    bucketWidth: temporalBucketWidth,
    useCache: useTemporalBucketCache
  });
  const bucketData = bucketResult.bucketData;
  const bucketRange = getTemporalBucketCandidateRange(bucketData, timestamp, temporalBucketRadius);
  const bucketCandidateIndices = buildTemporalBucketCandidateIndices(bucketData, bucketRange, stride);
  const bucketSummary = summarizeTemporalBucketRange(bucketData, bucketRange, bucketCandidateIndices);

  const temporalWindow = estimateTemporalWindow(
    raw,
    sigmaScale,
    temporalSigmaThreshold,
    { useCache: useTemporalIndexCache, mode: temporalWindowMode, fixedWindowRadius }
  );
  const filteredCandidateIndices = filterCandidateIndicesByTemporalWindow(
    raw,
    bucketCandidateIndices,
    timestamp,
    temporalWindow.windowRadius
  );

  const totalCount = raw ? raw.N : 0;
  const filteredCount = filteredCandidateIndices.length;
  const filteredFraction = totalCount > 0 ? filteredCount / totalCount : 0;

  return {
    candidateIndices: filteredCandidateIndices,
    candidateMode: 'hybrid',
    temporalWindow,
    rangeSummary: {
      totalCount,
      rangeCount: bucketSummary.bucketSourceCount,
      candidateCount: filteredCount,
      rangeFraction: bucketSummary.bucketSourceFraction,
      candidateFraction: filteredFraction
    },
    temporalIndexDebug: {
      enabled: true,
      cacheEnabled: !!useTemporalIndexCache,
      cacheHit: !!temporalWindow.cacheHit,
      builtThisFrame: !!temporalWindow.builtThisFrame,
      totalCount,
      tMin: bucketData ? bucketData.tMin : NaN,
      tMax: bucketData ? bucketData.tMax : NaN
    },
    temporalBucketDebug: {
      enabled: true,
      cacheEnabled: !!useTemporalBucketCache,
      cacheHit: bucketResult.cacheHit,
      builtThisFrame: bucketResult.builtThisFrame,
      bucketWidth: bucketData ? bucketData.bucketWidth : NaN,
      bucketRadius: temporalBucketRadius,
      bucketCount: bucketData ? bucketData.bucketCount : 0,
      bucketStart: bucketRange.bucketStart,
      bucketEnd: bucketRange.bucketEnd,
      bucketSourceCount: bucketSummary.bucketSourceCount,
      candidateCount: bucketSummary.candidateCount,
      bucketSourceFraction: bucketSummary.bucketSourceFraction,
      candidateFraction: bucketSummary.candidateFraction,
      postWindowCandidateCount: filteredCount,
      postWindowCandidateFraction: filteredFraction,
      tMin: bucketData ? bucketData.tMin : NaN,
      tMax: bucketData ? bucketData.tMax : NaN
    }
  };
}

function buildCandidateIndices(params) {
  const {
    raw,
    stride,
    temporalPrefilterMode,
    useTemporalIndex,
    useTemporalBucket
  } = params;

  if (!raw || !raw.t || !raw.scale_t) {
    return buildFallbackCandidateInfo(raw, stride, 'no-temporal-data');
  }

  const mode = temporalPrefilterMode || (
    useTemporalBucket && useTemporalIndex ? 'hybrid' :
    useTemporalBucket ? 'bucket' :
    useTemporalIndex ? 'sorted' :
    'off'
  );

  if (mode === 'hybrid' && useTemporalBucket && useTemporalIndex) {
    return buildHybridCandidateInfo(params);
  }
  if (mode === 'bucket' && useTemporalBucket) {
    return buildBucketCandidateInfo(params);
  }
  if (mode === 'sorted' && useTemporalIndex) {
    return buildSortedCandidateInfo(params);
  }
  return buildFallbackCandidateInfo(raw, stride, mode);
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
  playbackActive = false,
  qualityOverrideActive = false,
  qualityOverrideReason = 'none',
  temporalPrefilterMode = 'sorted',
  useTemporalIndex = true,
  useTemporalIndexCache = true,
  temporalWindowMode = 'median',
  fixedWindowRadius = 0.5,
  useTemporalBucket = false,
  useTemporalBucketCache = true,
  temporalBucketWidth = 0.1,
  temporalBucketRadius = 0
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
        candidateMode: 'none',
        qualityOverrideActive: false,
        qualityOverrideReason: 'none',
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
        temporalBucketEnabled: false,
        temporalBucketCacheEnabled: false,
        temporalBucketCacheHit: false,
        temporalBucketBuiltThisFrame: false,
        temporalBucketWidth: NaN,
        temporalBucketRadius: 0,
        temporalBucketCount: 0,
        temporalBucketStart: 0,
        temporalBucketEnd: -1,
        temporalBucketSourceCount: 0,
        temporalBucketCandidateCount: 0,
        temporalBucketSourceFraction: 0,
        temporalBucketCandidateFraction: 0,
        temporalBucketPostWindowCandidateCount: 0,
        temporalBucketPostWindowCandidateFraction: 0,
        temporalBucketTMin: NaN,
        temporalBucketTMax: NaN,
        interactionActive: false,
        playbackActive: false
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
    temporalPrefilterMode,
    useTemporalIndex,
    useTemporalIndexCache,
    temporalWindowMode,
    fixedWindowRadius,
    useTemporalBucket,
    useTemporalBucketCache,
    temporalBucketWidth,
    temporalBucketRadius
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
      candidateMode: candidateInfo.candidateMode,
      qualityOverrideActive,
      qualityOverrideReason,
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
      temporalBucketEnabled: !!useTemporalBucket,
      temporalBucketCacheEnabled: !!useTemporalBucketCache,
      temporalBucketCacheHit: candidateInfo.temporalBucketDebug.cacheHit,
      temporalBucketBuiltThisFrame: candidateInfo.temporalBucketDebug.builtThisFrame,
      temporalBucketWidth: candidateInfo.temporalBucketDebug.bucketWidth,
      temporalBucketRadius: candidateInfo.temporalBucketDebug.bucketRadius,
      temporalBucketCount: candidateInfo.temporalBucketDebug.bucketCount,
      temporalBucketStart: candidateInfo.temporalBucketDebug.bucketStart,
      temporalBucketEnd: candidateInfo.temporalBucketDebug.bucketEnd,
      temporalBucketSourceCount: candidateInfo.temporalBucketDebug.bucketSourceCount,
      temporalBucketCandidateCount: candidateInfo.temporalBucketDebug.candidateCount,
      temporalBucketSourceFraction: candidateInfo.temporalBucketDebug.bucketSourceFraction,
      temporalBucketCandidateFraction: candidateInfo.temporalBucketDebug.candidateFraction,
      temporalBucketPostWindowCandidateCount: candidateInfo.temporalBucketDebug.postWindowCandidateCount,
      temporalBucketPostWindowCandidateFraction: candidateInfo.temporalBucketDebug.postWindowCandidateFraction,
      temporalBucketTMin: candidateInfo.temporalBucketDebug.tMin,
      temporalBucketTMax: candidateInfo.temporalBucketDebug.tMax,
      interactionActive,
      playbackActive
    }
  };
}