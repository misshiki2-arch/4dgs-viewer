import { computeGaussianState, computeScreenSplat } from './rot4d_math.js';
import { evalSHColor } from './sh_eval.js';
import { clampInt, computeTileRangeFromAABB } from './gpu_tile_utils.js';
import { getTemporalIndexUiOptions } from './gpu_temporal_index_utils.js';
import { getTemporalBucketUiOptions } from './gpu_temporal_bucket_utils.js';
import { buildFallbackCandidateInfo } from './gpu_candidate_builder_fallback.js';
import { buildSortedCandidateInfo } from './gpu_candidate_builder_sorted.js';
import { buildBucketCandidateInfo } from './gpu_candidate_builder_bucket.js';
import { buildHybridCandidateInfo } from './gpu_candidate_builder_hybrid.js';
import {
  createScreenSpaceBuildContext,
  buildPackedScreenSpaceWithContext,
  summarizePackedScreenSpace
} from './gpu_screen_space_builder.js';

function deriveTemporalPrefilterMode(ui, temporalIndexOptions, temporalBucketOptions) {
  if (ui?.temporalPrefilterModeSelect) return ui.temporalPrefilterModeSelect.value;
  if (temporalBucketOptions.useTemporalBucket && temporalIndexOptions.useTemporalIndex) return 'hybrid';
  if (temporalBucketOptions.useTemporalBucket) return 'bucket';
  if (temporalIndexOptions.useTemporalIndex) return 'sorted';
  return 'off';
}

export function getVisibleBuildConfig(ui, qualityOverride = null) {
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

  if (!qualityOverride) return baseConfig;
  return { ...baseConfig, ...qualityOverride };
}

function passesTemporalCulling(raw, i, timestamp, sigmaScale = 1.0, sigmaThreshold = 3.0) {
  if (!raw || !raw.t || !raw.scale_t) return true;

  const t0 = raw.t[i];
  if (!Number.isFinite(t0)) return true;

  const s = raw.scale_t[i];
  if (!Number.isFinite(s)) return true;

  const sigmaT = s * sigmaScale;
  if (!Number.isFinite(sigmaT) || sigmaT <= 0) return true;

  return Math.abs(timestamp - t0) <= sigmaThreshold * sigmaT;
}

function buildCandidateInfo({
  raw,
  stride,
  temporalPrefilterMode,
  useTemporalIndex,
  useTemporalIndexCache,
  temporalWindowMode,
  fixedWindowRadius,
  useTemporalBucket,
  useTemporalBucketCache,
  temporalBucketWidth,
  temporalBucketRadius,
  timestamp,
  sigmaScale,
  temporalSigmaThreshold
}) {
  if (!raw || !raw.t || !raw.scale_t) {
    return buildFallbackCandidateInfo({
      raw,
      stride,
      reason: 'no-temporal-data'
    });
  }

  const mode = temporalPrefilterMode || (
    useTemporalBucket && useTemporalIndex ? 'hybrid' :
    useTemporalBucket ? 'bucket' :
    useTemporalIndex ? 'sorted' :
    'off'
  );

  if (mode === 'hybrid' && useTemporalBucket && useTemporalIndex) {
    return buildHybridCandidateInfo({
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
    });
  }

  if (mode === 'bucket' && useTemporalBucket) {
    return buildBucketCandidateInfo({
      raw,
      timestamp,
      stride,
      useTemporalBucketCache,
      temporalBucketWidth,
      temporalBucketRadius
    });
  }

  if (mode === 'sorted' && useTemporalIndex) {
    return buildSortedCandidateInfo({
      raw,
      timestamp,
      stride,
      sigmaScale,
      temporalSigmaThreshold,
      useTemporalIndexCache,
      temporalWindowMode,
      fixedWindowRadius
    });
  }

  return buildFallbackCandidateInfo({
    raw,
    stride,
    reason: mode
  });
}

function buildColorAlpha(color, opacity) {
  const r = Array.isArray(color) && Number.isFinite(color[0]) ? color[0] : 0;
  const g = Array.isArray(color) && Number.isFinite(color[1]) ? color[1] : 0;
  const b = Array.isArray(color) && Number.isFinite(color[2]) ? color[2] : 0;
  const a = Number.isFinite(opacity)
    ? opacity
    : (Array.isArray(color) && Number.isFinite(color[3]) ? color[3] : 0);
  return [r, g, b, a];
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
  temporalBucketRadius = 0,
  enablePackedVisiblePath = true,
  screenSpaceContext = null
}) {
  if (!raw) {
    return {
      visible: [],
      packedScreenSpace: null,
      packedSummary: summarizePackedScreenSpace(null),
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
        packedVisiblePathEnabled: !!enablePackedVisiblePath,
        packedVisiblePathUsed: false,
        packedVisibleCount: 0,
        packedVisibleLength: 0,
        packedVisibleFloatsPerItem: 0,
        packedVisiblePath: 'none',
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

  const candidateInfo = buildCandidateInfo({
    raw,
    stride,
    temporalPrefilterMode,
    useTemporalIndex,
    useTemporalIndexCache,
    temporalWindowMode,
    fixedWindowRadius,
    useTemporalBucket,
    useTemporalBucketCache,
    temporalBucketWidth,
    temporalBucketRadius,
    timestamp,
    sigmaScale,
    temporalSigmaThreshold
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

    const colorAlpha = buildColorAlpha(color, splat.opacity);

    visible.push({
      srcIndex: i,
      px,
      py,
      radius,
      depth: splat.depth,
      colorAlpha,
      conic: [
        splat.conic[0] / (sx * sx),
        splat.conic[1] / (sx * sy),
        splat.conic[2] / (sy * sy)
      ],
      aabb: [minX, minY, maxX, maxY],
      tileRange,
      // backward compatibility only; draw contract should use colorAlpha
      color,
      opacity: splat.opacity
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

  const packedCtx = screenSpaceContext || createScreenSpaceBuildContext();
  const packedScreenSpace = enablePackedVisiblePath
    ? buildPackedScreenSpaceWithContext(packedCtx, visible, {
        renderW,
        renderH,
        sx,
        sy
      })
    : null;
  const packedSummary = summarizePackedScreenSpace(packedScreenSpace);

  return {
    visible,
    packedScreenSpace,
    packedSummary,
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
      packedVisiblePathEnabled: !!enablePackedVisiblePath,
      packedVisiblePathUsed: !!packedScreenSpace,
      packedVisibleCount: packedSummary.packedCount,
      packedVisibleLength: packedSummary.packedLength,
      packedVisibleFloatsPerItem: packedSummary.floatsPerItem,
      packedVisiblePath: packedSummary.path,
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
