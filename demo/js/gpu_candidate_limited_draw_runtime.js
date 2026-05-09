import { buildCandidateInfo } from './gpu_candidate_path_selector.js';
import { buildCandidateSubsetInfo } from './gpu_candidate_builder_gpu_stub.js';
import {
  buildCpuFilteredCandidateInfo,
  buildGpuExplicitCandidateInfo,
  buildGpuFirstNCandidateInfo
} from './gpu_candidate_builder_gpu_firstn.js';
import { buildCandidateComparisonSummary } from './gpu_visible_compare_debug.js';
import {
  buildGpuCandidateRuntimeConfig,
  buildGpuCandidateRuntimeSummary
} from './gpu_candidate_runtime_selector.js';
import { buildGpuCandidateRuntimeFallbackSummary } from './gpu_candidate_runtime_fallback.js';

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

function buildSkippedSummary({
  runtimeConfig,
  runtimeSummary,
  fallback,
  reason,
  candidateArgs,
  referenceCandidateInfo = null
}) {
  return {
    schemaVersion: 'step99-gpu-candidate-limited-draw-summary-v1',
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
    candidateComparison: null
  };
}

export function resolveGpuCandidateLimitedDrawRuntime({
  gl,
  raw,
  queryState = {},
  overrides = {},
  candidateArgs,
  visibleSourceItems = null,
  shadowCompare = null
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
  const needsVisibleSource = subsetMode === 'visibleSrcIndices' ||
    subsetMode === 'fromVisible' ||
    subsetMode === 'visibleReachable';
  if (needsVisibleSource && !Array.isArray(visibleSourceItems)) {
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
  const referenceSubsetCandidateInfo = buildReferenceSubsetCandidateInfo({
    raw,
    referenceCandidateInfo,
    subsetMode,
    subsetCount: runtimeConfig.subsetCount,
    visibleSourceItems
  });
  const referenceFilteredCandidateInfo = buildCpuFilteredCandidateInfo({
    raw,
    referenceSubsetCandidateInfo,
    filterMode: runtimeConfig.filterMode,
    candidateMode: referenceSubsetCandidateInfo.candidateSubsetSummary?.subsetMode === 'visibleSrcIndices'
      ? 'cpu-visible-src-filter-reference'
      : 'cpu-firstn-filter-reference'
  });
  const gpuCandidateInfo = referenceSubsetCandidateInfo.candidateSubsetSummary?.subsetMode === 'visibleSrcIndices'
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
      });
  const candidateComparison = buildCandidateComparisonSummary({
    referenceCandidateInfo: referenceFilteredCandidateInfo,
    candidateCandidateInfo: gpuCandidateInfo,
    referenceLabel: 'cpu-filtered-candidate-reference',
    candidateLabel: 'gpu-candidate-limited-draw',
    options: { maxMismatches: 16 },
    metadata: {
      comparisonMode: 'cpu-filtered-candidate-vs-gpu-candidate-limited-draw',
      subset: referenceSubsetCandidateInfo.candidateSubsetSummary ?? null,
      selectedCandidateCount: referenceSubsetCandidateInfo.candidateIndices?.length ?? 0,
      cpuFilterSummary: referenceFilteredCandidateInfo.filterSummary ?? null,
      gpuCandidateSummary: gpuCandidateInfo.gpuCandidateSummary ?? null,
      filterSummary: gpuCandidateInfo.filterSummary ?? null,
      candidateArgs: summarizeCandidateArgs(candidateArgs)
    }
  });
  const finalFallback = buildGpuCandidateRuntimeFallbackSummary({
    runtimeConfig,
    shadowCompare: {
      status: shadowCompare?.status ?? 'ok',
      candidateComparison,
      visibleComparison: null,
      summary: { anyMismatch: !!candidateComparison.anyMismatch }
    }
  });
  const useGpuCandidate = finalFallback.action === 'use-gpu-candidate';

  return {
    candidateInfoOverride: useGpuCandidate ? gpuCandidateInfo : null,
    runtimeConfig,
    runtimeSummary,
    fallback: finalFallback,
    summary: {
      schemaVersion: 'step99-gpu-candidate-limited-draw-summary-v1',
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
      referenceCandidateSummary: summarizeCandidateInfo(referenceFilteredCandidateInfo),
      gpuCandidateSummary: summarizeCandidateInfo(gpuCandidateInfo),
      candidateComparison
    }
  };
}
