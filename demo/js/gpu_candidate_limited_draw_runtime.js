import { buildCandidateInfo } from './gpu_candidate_path_selector.js';
import { buildCandidateSubsetInfo } from './gpu_candidate_builder_gpu_stub.js';
import {
  buildCpuFilteredCandidateInfo,
  buildGpuExplicitCandidateInfo,
  buildGpuFirstNCandidateInfo
} from './gpu_candidate_builder_gpu_firstn.js';
import { buildVisibleItemForCandidate } from './gpu_visible_item_builder.js';
import {
  buildPackedScreenSpaceWithContext,
  createScreenSpaceBuildContext
} from './gpu_screen_space_builder.js';
import {
  buildCandidateComparisonSummary,
  buildVisibleComparisonSummary
} from './gpu_visible_compare_debug.js';
import {
  buildGpuCandidateRuntimeConfig,
  buildGpuCandidateRuntimeSummary
} from './gpu_candidate_runtime_selector.js';
import { buildGpuCandidateRuntimeFallbackSummary } from './gpu_candidate_runtime_fallback.js';
import {
  buildGpuOwnedCandidateSourceComparison,
  isGpuOwnedCandidateSourceMode
} from './gpu_candidate_source_runtime.js';

function clonePlainObject(value) {
  if (!value || typeof value !== 'object') return value;
  if (Array.isArray(value)) return value.map((item) => clonePlainObject(item));
  const out = {};
  for (const [key, item] of Object.entries(value)) out[key] = clonePlainObject(item);
  return out;
}

function collectVisibleSrcIndices(visibleItems, subsetCount) {
  const out = [];
  const seen = new Set();
  const maxCount = Number.isFinite(Number(subsetCount)) ? Math.max(0, Number(subsetCount) | 0) : 1024;
  for (const item of Array.isArray(visibleItems) ? visibleItems : []) {
    const index = Number(item?.srcIndex);
    if (!Number.isFinite(index)) continue;
    const intIndex = index | 0;
    if (intIndex < 0 || seen.has(intIndex)) continue;
    seen.add(intIndex);
    out.push(intIndex);
    if (out.length >= maxCount) break;
  }
  return Uint32Array.from(out);
}

function buildCandidateInfoFromIndices({
  raw = null,
  referenceCandidateInfo = null,
  candidateIndices,
  candidateMode = 'cpu-visible-src-indices-subset',
  subsetMode = 'visibleSrcIndices',
  requestedSubsetCount = 1024,
  sourceVisibleCount = 0
} = {}) {
  const indices = candidateIndices instanceof Uint32Array
    ? candidateIndices
    : (Array.isArray(candidateIndices) ? Uint32Array.from(candidateIndices) : new Uint32Array(0));
  const total = raw ? raw.N : (referenceCandidateInfo?.rangeSummary?.totalCount ?? indices.length);
  return {
    candidateIndices: indices,
    candidateMode,
    temporalWindow: clonePlainObject(referenceCandidateInfo?.temporalWindow) ?? null,
    rangeSummary: {
      ...(clonePlainObject(referenceCandidateInfo?.rangeSummary) ?? {}),
      totalCount: total,
      rangeCount: indices.length,
      candidateCount: indices.length,
      rangeFraction: total > 0 ? indices.length / total : 0,
      candidateFraction: total > 0 ? indices.length / total : 0
    },
    temporalIndexDebug: clonePlainObject(referenceCandidateInfo?.temporalIndexDebug) ?? null,
    temporalBucketDebug: clonePlainObject(referenceCandidateInfo?.temporalBucketDebug) ?? null,
    candidateSubsetSummary: {
      enabled: true,
      contract: 'candidate-info-visible-src-indices-subset',
      subsetMode,
      subsetCount: indices.length,
      selectedCandidateCount: indices.length,
      requestedSubsetCount,
      sourceVisibleCount,
      sourceCandidateMode: referenceCandidateInfo?.candidateMode ?? 'unknown',
      sourceCandidateCount: referenceCandidateInfo?.candidateIndices?.length ?? 0
    }
  };
}

function buildReferenceSubsetCandidateInfo({
  raw,
  referenceCandidateInfo,
  subsetMode,
  subsetCount,
  visibleSourceItems
}) {
  if (subsetMode === 'visibleSrcIndices' || subsetMode === 'fromVisible' || subsetMode === 'visibleReachable') {
    const indices = collectVisibleSrcIndices(visibleSourceItems, subsetCount);
    return buildCandidateInfoFromIndices({
      raw,
      referenceCandidateInfo,
      candidateIndices: indices,
      candidateMode: 'cpu-visible-src-indices-subset',
      subsetMode: 'visibleSrcIndices',
      requestedSubsetCount: subsetCount,
      sourceVisibleCount: Array.isArray(visibleSourceItems) ? visibleSourceItems.length : 0
    });
  }
  return buildCandidateSubsetInfo({
    raw,
    referenceCandidateInfo,
    subsetMode: 'firstN',
    subsetCount,
    explicitIndices: null,
    candidateMode: 'cpu-firstn-subset'
  });
}

function summarizeCandidateInfo(candidateInfo) {
  return {
    candidateMode: candidateInfo?.candidateMode ?? 'none',
    candidateCount: candidateInfo?.candidateIndices?.length ?? 0,
    filterMode: candidateInfo?.filterMode ?? null,
    validCount: Number.isFinite(candidateInfo?.validCount) ? candidateInfo.validCount : null,
    rejectedCount: Number.isFinite(candidateInfo?.rejectedCount) ? candidateInfo.rejectedCount : null,
    rangeSummary: candidateInfo?.rangeSummary ?? null,
    candidateSubsetSummary: candidateInfo?.candidateSubsetSummary ?? null,
    candidateSourceSummary: candidateInfo?.candidateSourceSummary ?? null,
    filterSummary: candidateInfo?.filterSummary ?? null,
    gpuCandidateSummary: candidateInfo?.gpuCandidateSummary ?? null
  };
}

function summarizeCandidateArgs(candidateArgs) {
  return {
    stride: candidateArgs?.stride,
    temporalPrefilterMode: candidateArgs?.temporalPrefilterMode,
    useTemporalIndex: candidateArgs?.useTemporalIndex,
    useTemporalBucket: candidateArgs?.useTemporalBucket,
    timestamp: candidateArgs?.timestamp,
    sigmaScale: candidateArgs?.sigmaScale,
    temporalSigmaThreshold: candidateArgs?.temporalSigmaThreshold,
    temporalWindowMode: candidateArgs?.temporalWindowMode,
    fixedWindowRadius: candidateArgs?.fixedWindowRadius,
    temporalBucketWidth: candidateArgs?.temporalBucketWidth,
    temporalBucketRadius: candidateArgs?.temporalBucketRadius
  };
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

function buildVisibleAndPackedFromCandidateInfo({
  candidateInfo,
  raw,
  camera,
  screenSpaceCamera = null,
  canvasWidth,
  canvasHeight,
  camPos,
  tileGrid = null,
  buildConfig = {},
  temporalSigmaThreshold = 3.0,
  label = 'limited-draw-candidate'
} = {}) {
  const renderScale = Number.isFinite(buildConfig.renderScale) ? buildConfig.renderScale : 1;
  const renderW = Math.max(1, Math.round(canvasWidth * renderScale));
  const renderH = Math.max(1, Math.round(canvasHeight * renderScale));
  const sx = canvasWidth / renderW;
  const sy = canvasHeight / renderH;
  const flags = {
    nativeRot4d: !!buildConfig.useNativeRot4d,
    nativeMarginal: !!buildConfig.useNativeMarginal
  };
  const maxVisible = Number.isFinite(buildConfig.maxVisible) ? buildConfig.maxVisible : Infinity;
  const candidateIndices = candidateInfo?.candidateIndices instanceof Uint32Array
    ? candidateInfo.candidateIndices
    : (Array.isArray(candidateInfo?.candidateIndices)
      ? Uint32Array.from(candidateInfo.candidateIndices)
      : new Uint32Array(0));
  const visible = [];
  let processed = 0;
  let culled = 0;
  let temporalRejected = 0;
  let temporalPassed = 0;

  for (let k = 0; k < candidateIndices.length; k++) {
    const index = candidateIndices[k];
    processed++;
    if (!passesTemporalCulling(raw, index, buildConfig.timestamp, buildConfig.sigmaScale, temporalSigmaThreshold)) {
      temporalRejected++;
      culled++;
      continue;
    }
    temporalPassed++;
    const item = buildVisibleItemForCandidate({
      raw,
      index,
      camera,
      screenSpaceCamera,
      renderW,
      renderH,
      canvasWidth,
      canvasHeight,
      sx,
      sy,
      timestamp: buildConfig.timestamp,
      scalingModifier: buildConfig.scalingModifier,
      sigmaScale: buildConfig.sigmaScale,
      prefilterVar: buildConfig.prefilterVar,
      useRot4d: buildConfig.useRot4d,
      flags,
      camPos,
      timeDuration: buildConfig.timeDuration,
      useSH: buildConfig.useSH,
      forceSh3d: buildConfig.forceSh3d,
      tileGrid
    });
    if (!item) {
      culled++;
      continue;
    }
    visible.push(item);
    if (visible.length >= maxVisible) break;
  }
  visible.sort((a, b) => b.depth - a.depth);
  const packedScreenSpace = buildConfig.enablePackedVisiblePath !== false
    ? buildPackedScreenSpaceWithContext(createScreenSpaceBuildContext(), visible, {
        renderW,
        renderH,
        sx,
        sy
      })
    : null;
  return {
    label,
    candidateInfo,
    visible,
    packedScreenSpace,
    buildStats: {
      label,
      accepted: visible.length,
      processed,
      culled,
      temporalRejected,
      temporalPassed,
      candidateMode: candidateInfo?.candidateMode ?? 'unknown',
      candidateCount: candidateIndices.length,
      packedVisibleCount: packedScreenSpace?.packedCount ?? 0,
      packedVisibleLength: packedScreenSpace?.packed?.length ?? 0,
      packedVisibleFloatsPerItem: packedScreenSpace?.floatsPerItem ?? 0
    }
  };
}

function buildSkippedSummary({
  runtimeConfig,
  runtimeSummary,
  fallback,
  reason,
  candidateArgs,
  referenceCandidateInfo = null
}) {
  return {
    schemaVersion: 'step105-gpu-candidate-limited-draw-summary-v1',
    requestedRuntime: runtimeConfig.requestedRuntime ?? 'cpu-reference',
    effectiveDisplayRuntime: fallback.effectiveRuntime ?? 'cpu-reference',
    displayCandidateSource: fallback.displayCandidateSource ?? 'cpu-reference',
    limitedDrawUsedForCandidateSource: false,
    gpuCandidateUsedForDisplay: false,
    candidateInfoOverrideProvided: false,
    status: 'fallback',
    reason,
    runtimeSummary,
    fallback,
    candidateArgs: summarizeCandidateArgs(candidateArgs),
    referenceCandidateSummary: summarizeCandidateInfo(referenceCandidateInfo),
    gpuCandidateSummary: null,
    candidateSourceSummary: null,
    candidateSourceComparison: null,
    candidateCoverageSummary: null,
    candidateComparison: null
  };
}

function buildLimitedDrawFallbackReasons(
  runtimeConfig,
  gpuCandidateInfo,
  visibleComparison,
  shadowCompare,
  referenceSubsetCandidateInfo,
  candidateSourceComparison
) {
  const reasons = [];
  const sourceMode = runtimeConfig.sourceMode ?? 'visibleSrcIndices';
  const isGpuOwnedSource = isGpuOwnedCandidateSourceMode(sourceMode);
  if (isGpuOwnedSource) {
    reasons.push({
      code: 'source-mode-display-not-allowed',
      message: 'GPU-owned candidate source mode is compare-only in Step103 and is not allowed to replace the normal display candidate source.',
      details: {
        sourceMode,
        promotePolicy: runtimeConfig.promotePolicy ?? 'never',
        rangeStart: runtimeConfig.rangeStart ?? null,
        rangeCount: runtimeConfig.rangeCount ?? null
      }
    });
  } else if (referenceSubsetCandidateInfo?.candidateSubsetSummary?.enabled) {
    reasons.push({
      code: 'subset-display-not-allowed',
      message: 'Subset limited-draw is compare-only and is not allowed to replace the normal display candidate source.',
      details: {
        subsetMode: referenceSubsetCandidateInfo.candidateSubsetSummary.subsetMode ?? runtimeConfig.subsetMode,
        subsetCount: referenceSubsetCandidateInfo.candidateSubsetSummary.subsetCount ?? null,
        requestedSubsetCount: referenceSubsetCandidateInfo.candidateSubsetSummary.requestedSubsetCount ?? runtimeConfig.subsetCount
      }
    });
  }
  if (!isGpuOwnedSource && runtimeConfig.subsetMode !== 'visibleSrcIndices') {
    reasons.push({
      code: 'unsupported-limited-draw-subset',
      message: 'limited-draw compare-only path currently supports visibleSrcIndices.',
      details: { subsetMode: runtimeConfig.subsetMode }
    });
  }
  if (!isGpuOwnedSource && runtimeConfig.subsetCount !== 1024) {
    reasons.push({
      code: 'unsupported-limited-draw-subset-count',
      message: 'limited-draw compare-only path currently supports subsetCount=1024.',
      details: { subsetCount: runtimeConfig.subsetCount }
    });
  }
  if (runtimeConfig.filterMode !== 'all-valid') {
    reasons.push({
      code: 'unsupported-limited-draw-filter',
      message: 'limited-draw compare-only path currently supports filterMode=all-valid.',
      details: { filterMode: runtimeConfig.filterMode }
    });
  }
  if (isGpuOwnedSource && runtimeConfig.promotePolicy !== 'never') {
    reasons.push({
      code: 'unsupported-promote-policy',
      message: 'Step103 keeps GPU-owned candidate source modes compare-only.',
      details: { promotePolicy: runtimeConfig.promotePolicy }
    });
  }
  if (!runtimeConfig.requireCompare) {
    reasons.push({
      code: 'compare-required-for-limited-draw',
      message: 'limited-draw subset path requires candidate/visible comparison.'
    });
  }
  if (!runtimeConfig.requireShadowOk) {
    reasons.push({
      code: 'shadow-ok-required-for-limited-draw',
      message: 'limited-draw subset path requires shadow compare gating.'
    });
  } else if (shadowCompare?.status !== 'ok') {
    reasons.push({
      code: 'shadow-compare-required-for-limited-draw',
      message: 'limited-draw subset path requires a latest shadow compare with status ok.',
      details: { shadowStatus: shadowCompare?.status ?? null }
    });
  }
  if ((gpuCandidateInfo?.candidateIndices?.length ?? 0) <= 0) {
    reasons.push({
      code: 'empty-gpu-candidate',
      message: 'GPU candidate output was empty.'
    });
  }
  if (isGpuOwnedSource && candidateSourceComparison?.status && candidateSourceComparison.status !== 'ok') {
    reasons.push({
      code: 'gpu-error',
      message: 'GPU-owned candidate source comparison did not complete successfully.',
      details: {
        status: candidateSourceComparison.status,
        reason: candidateSourceComparison.reason ?? null
      }
    });
  }
  if (!visibleComparison) {
    reasons.push({
      code: 'visible-comparison-required-for-limited-draw',
      message: 'limited-draw subset path requires visible and packed payload comparison.'
    });
  }
  return reasons;
}

export function resolveGpuCandidateLimitedDrawRuntime({
  gl,
  raw,
  queryState = {},
  overrides = {},
  candidateArgs,
  visibleSourceItems = null,
  shadowCompare = null,
  camera = null,
  screenSpaceCamera = null,
  canvasWidth = 0,
  canvasHeight = 0,
  camPos = null,
  tileGrid = null,
  buildConfig = null
} = {}) {
  const runtimeConfig = buildGpuCandidateRuntimeConfig(queryState, overrides);
  const runtimeSummary = buildGpuCandidateRuntimeSummary(runtimeConfig);
  const preflightFallback = buildGpuCandidateRuntimeFallbackSummary({
    runtimeConfig,
    shadowCompare
  });

  if (runtimeConfig.requestedRuntime !== 'limited-draw') {
    return {
      candidateInfoOverride: null,
      runtimeConfig,
      runtimeSummary,
      fallback: preflightFallback,
      summary: buildSkippedSummary({
        runtimeConfig,
        runtimeSummary,
        fallback: preflightFallback,
        reason: 'runtime-not-limited-draw',
        candidateArgs
      })
    };
  }

  if (preflightFallback.action !== 'use-gpu-candidate') {
    return {
      candidateInfoOverride: null,
      runtimeConfig,
      runtimeSummary,
      fallback: preflightFallback,
      summary: buildSkippedSummary({
        runtimeConfig,
        runtimeSummary,
        fallback: preflightFallback,
        reason: preflightFallback.reason ?? 'fallback',
        candidateArgs
      })
    };
  }

  const subsetMode = runtimeConfig.subsetMode ?? 'visibleSrcIndices';
  const sourceMode = runtimeConfig.sourceMode ?? 'visibleSrcIndices';
  const gpuOwnedSourceMode = isGpuOwnedCandidateSourceMode(sourceMode);
  const needsVisibleSource = subsetMode === 'visibleSrcIndices' ||
    subsetMode === 'fromVisible' ||
    subsetMode === 'visibleReachable';
  if (!gpuOwnedSourceMode && needsVisibleSource && !Array.isArray(visibleSourceItems)) {
    const fallback = buildGpuCandidateRuntimeFallbackSummary({
      runtimeConfig,
      shadowCompare,
      extraReasons: [{
        code: 'visible-src-indices-reference-required-in-draw',
        message: 'limited-draw visibleSrcIndices requires an already-built CPU reference visible list before it can be promoted.',
        details: { subsetMode }
      }]
    });
    return {
      candidateInfoOverride: null,
      runtimeConfig,
      runtimeSummary,
      fallback,
      summary: buildSkippedSummary({
        runtimeConfig,
        runtimeSummary,
        fallback,
        reason: fallback.reason,
        candidateArgs
      })
    };
  }

  const referenceCandidateInfo = buildCandidateInfo(candidateArgs);
  const sourceComparison = gpuOwnedSourceMode
    ? buildGpuOwnedCandidateSourceComparison({
        gl,
        raw,
        runtimeConfig,
        referenceCandidateInfo,
        filterMode: runtimeConfig.filterMode,
        metadata: {
          candidateArgs: summarizeCandidateArgs(candidateArgs)
        }
      })
    : null;
  const referenceSubsetCandidateInfo = gpuOwnedSourceMode
    ? sourceComparison?.cpuMirrorCandidateInfo
    : buildReferenceSubsetCandidateInfo({
        raw,
        referenceCandidateInfo,
        subsetMode,
        subsetCount: runtimeConfig.subsetCount,
        visibleSourceItems
      });
  const referenceFilteredCandidateInfo = gpuOwnedSourceMode
    ? sourceComparison.cpuMirrorCandidateInfo
    : buildCpuFilteredCandidateInfo({
        raw,
        referenceSubsetCandidateInfo,
        filterMode: runtimeConfig.filterMode,
        candidateMode: referenceSubsetCandidateInfo.candidateSubsetSummary?.subsetMode === 'visibleSrcIndices'
          ? 'cpu-visible-src-filter-reference'
          : 'cpu-firstn-filter-reference'
      });
  const gpuCandidateInfo = gpuOwnedSourceMode
    ? sourceComparison.gpuCandidateInfo
    : (referenceSubsetCandidateInfo.candidateSubsetSummary?.subsetMode === 'visibleSrcIndices'
      ? buildGpuExplicitCandidateInfo({
          gl,
          raw,
          referenceSubsetCandidateInfo,
          candidateIndices: referenceSubsetCandidateInfo.candidateIndices,
          filterMode: runtimeConfig.filterMode
        })
      : buildGpuFirstNCandidateInfo({
          gl,
          raw,
          referenceSubsetCandidateInfo,
          subsetCount: runtimeConfig.subsetCount,
          startIndex: runtimeConfig.startIndex,
          filterMode: runtimeConfig.filterMode
        }));
  const candidateComparison = buildCandidateComparisonSummary({
    referenceCandidateInfo: referenceFilteredCandidateInfo,
    candidateCandidateInfo: gpuCandidateInfo,
    referenceLabel: gpuOwnedSourceMode ? 'cpu-range-candidate-source-reference' : 'cpu-filtered-candidate-reference',
    candidateLabel: gpuOwnedSourceMode ? 'gpu-range-candidate-source' : 'gpu-candidate-limited-draw',
    options: { maxMismatches: 16 },
    metadata: {
      comparisonMode: gpuOwnedSourceMode
        ? 'cpu-range-candidate-source-vs-gpu-range-candidate-source'
        : 'cpu-filtered-candidate-vs-gpu-candidate-limited-draw',
      sourceMode,
      sourceConfig: sourceComparison?.sourceConfig ?? null,
      candidateSourceSummary: sourceComparison?.candidateSourceSummary ?? null,
      subset: referenceSubsetCandidateInfo.candidateSubsetSummary ?? null,
      selectedCandidateCount: referenceSubsetCandidateInfo.candidateIndices?.length ?? 0,
      cpuFilterSummary: referenceFilteredCandidateInfo.filterSummary ?? null,
      gpuCandidateSummary: gpuCandidateInfo.gpuCandidateSummary ?? null,
      filterSummary: gpuCandidateInfo.filterSummary ?? null,
      candidateArgs: summarizeCandidateArgs(candidateArgs)
    }
  });
  const canBuildVisibleComparison = raw &&
    camera &&
    Number.isFinite(canvasWidth) &&
    Number.isFinite(canvasHeight) &&
    buildConfig;
  const referenceVisibleBuild = canBuildVisibleComparison
    ? buildVisibleAndPackedFromCandidateInfo({
        candidateInfo: referenceFilteredCandidateInfo,
        raw,
        camera,
        screenSpaceCamera,
        canvasWidth,
        canvasHeight,
        camPos,
        tileGrid,
        buildConfig,
        temporalSigmaThreshold: candidateArgs?.temporalSigmaThreshold ?? 3.0,
        label: 'cpu-filtered-candidate-visible-limited-draw-reference'
      })
    : null;
  const gpuVisibleBuild = canBuildVisibleComparison
    ? buildVisibleAndPackedFromCandidateInfo({
        candidateInfo: gpuCandidateInfo,
        raw,
        camera,
        screenSpaceCamera,
        canvasWidth,
        canvasHeight,
        camPos,
        tileGrid,
        buildConfig,
        temporalSigmaThreshold: candidateArgs?.temporalSigmaThreshold ?? 3.0,
        label: 'gpu-candidate-visible-limited-draw'
      })
    : null;
  const visibleComparison = canBuildVisibleComparison
    ? buildVisibleComparisonSummary({
        referenceItems: referenceVisibleBuild.visible,
        candidateItems: gpuVisibleBuild.visible,
        referencePackedPayload: referenceVisibleBuild.packedScreenSpace,
        candidatePackedPayload: gpuVisibleBuild.packedScreenSpace,
        referenceLabel: 'cpu-filtered-visible-limited-draw-reference',
        candidateLabel: 'gpu-candidate-visible-limited-draw',
        options: { epsilon: 1e-6, maxMismatches: 16 },
        metadata: {
          comparisonMode: 'cpu-filtered-visible-vs-gpu-candidate-limited-draw',
          referenceBuildStats: referenceVisibleBuild.buildStats,
          candidateBuildStats: gpuVisibleBuild.buildStats
        }
      })
    : null;
  const finalFallback = buildGpuCandidateRuntimeFallbackSummary({
    runtimeConfig,
    shadowCompare: {
      status: shadowCompare?.status ?? 'ok',
      candidateComparison,
      visibleComparison,
      summary: { anyMismatch: !!(candidateComparison.anyMismatch || visibleComparison?.anyMismatch) }
    },
    extraReasons: buildLimitedDrawFallbackReasons(
      runtimeConfig,
      gpuCandidateInfo,
      visibleComparison,
      shadowCompare,
      referenceSubsetCandidateInfo,
      sourceComparison
    )
  });
  const useGpuCandidate = finalFallback.action === 'use-gpu-candidate';

  return {
    candidateInfoOverride: useGpuCandidate ? gpuCandidateInfo : null,
    runtimeConfig,
    runtimeSummary,
    fallback: finalFallback,
    candidateSourceComparison: sourceComparison,
    summary: {
      schemaVersion: 'step105-gpu-candidate-limited-draw-summary-v1',
      requestedRuntime: runtimeConfig.requestedRuntime,
      effectiveDisplayRuntime: finalFallback.effectiveRuntime,
      displayCandidateSource: finalFallback.displayCandidateSource,
      limitedDrawUsedForCandidateSource: useGpuCandidate,
      gpuCandidateUsedForDisplay: useGpuCandidate,
      candidateInfoOverrideProvided: useGpuCandidate,
      status: useGpuCandidate ? 'using-gpu-candidate' : 'fallback',
      reason: finalFallback.reason,
      runtimeSummary,
      fallback: finalFallback,
      candidateArgs: summarizeCandidateArgs(candidateArgs),
      candidateSourceSummary: sourceComparison?.candidateSourceSummary ?? null,
      candidateCoverageSummary: null,
      candidateSourceComparison: sourceComparison
        ? {
            schemaVersion: sourceComparison.schemaVersion,
            status: sourceComparison.status,
            reason: sourceComparison.reason,
            sourceConfig: sourceComparison.sourceConfig,
            candidateSourceSummary: sourceComparison.candidateSourceSummary,
            cpuMirrorCandidateSummary: sourceComparison.cpuMirrorCandidateSummary,
            gpuCandidateSummary: sourceComparison.gpuCandidateSummary,
            candidateComparison: sourceComparison.candidateComparison
          }
        : null,
      referenceCandidateSummary: summarizeCandidateInfo(referenceFilteredCandidateInfo),
      gpuCandidateSummary: summarizeCandidateInfo(gpuCandidateInfo),
      candidateComparison,
      visibleComparison,
      referenceVisibleBuildSummary: referenceVisibleBuild?.buildStats ?? null,
      gpuVisibleBuildSummary: gpuVisibleBuild?.buildStats ?? null
    }
  };
}
