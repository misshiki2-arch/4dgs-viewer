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

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

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
    screenCoarseSummary: candidateInfo?.screenCoarseSummary ?? null,
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
  const totalStartMs = nowMs();
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

  const visibleLoopStartMs = nowMs();
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
  const visibleLoopMs = nowMs() - visibleLoopStartMs;
  const sortStartMs = nowMs();
  visible.sort((a, b) => b.depth - a.depth);
  const visibleSortMs = nowMs() - sortStartMs;
  const packedStartMs = nowMs();
  const packedScreenSpace = buildConfig.enablePackedVisiblePath !== false
    ? buildPackedScreenSpaceWithContext(createScreenSpaceBuildContext(), visible, {
        renderW,
        renderH,
        sx,
        sy
      })
    : null;
  const packedBuildMs = nowMs() - packedStartMs;
  const totalBuildMs = nowMs() - totalStartMs;
  const visibleBuildMs = totalBuildMs - packedBuildMs;
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
      packedVisibleFloatsPerItem: packedScreenSpace?.floatsPerItem ?? 0,
      visibleLoopMs,
      visibleSortMs,
      visibleBuildMs,
      packedBuildMs,
      screenSpaceBuildMs: packedBuildMs,
      totalBuildMs
    }
  };
}

function toUint32Array(value) {
  if (value instanceof Uint32Array) return value;
  if (Array.isArray(value)) return Uint32Array.from(value);
  return new Uint32Array(0);
}

function getVisibleSrcIndex(item) {
  const value = Number(item?.srcIndex);
  return Number.isFinite(value) && value >= 0 ? (value >>> 0) : null;
}

function buildVisibleCoverageAgainstCandidate(visibleItems, candidateInfo) {
  const candidateIndices = toUint32Array(candidateInfo?.candidateIndices);
  const candidateSet = new Set();
  for (let i = 0; i < candidateIndices.length; i++) {
    candidateSet.add(candidateIndices[i] >>> 0);
  }
  const visible = Array.isArray(visibleItems) ? visibleItems : [];
  let hitCount = 0;
  let missCount = 0;
  for (const item of visible) {
    const srcIndex = getVisibleSrcIndex(item);
    if (srcIndex === null) continue;
    if (candidateSet.has(srcIndex)) {
      hitCount++;
    } else {
      missCount++;
    }
  }
  const count = hitCount + missCount;
  return {
    schemaVersion: 'step110-screen-coarse-promotion-coverage-v1',
    sourceMode: 'screenCoarse',
    gpuCandidateCount: candidateIndices.length,
    cpuVisibleCount: count,
    visibleHitCount: hitCount,
    visibleMissCount: missCount,
    visibleCoverageRatio: count > 0 ? hitCount / count : 1
  };
}

function getVisibleItemsMismatch(visibleComparison) {
  return !!visibleComparison?.visibleItems?.anyMismatch;
}

function getVisibleCountMismatch(visibleComparison) {
  return visibleComparison?.visibleItems?.countEqual === false;
}

function getPackedMismatch(visibleComparison) {
  const packedPayload = visibleComparison?.packedPayload;
  if (!packedPayload) return true;
  return !!(packedPayload.anyMismatch ?? packedPayload.packedAnyMismatch);
}

function classifyScreenCoarsePromotionMismatch({
  candidateComparison = null,
  visibleComparison = null,
  coverageSummary = null
} = {}) {
  if (!coverageSummary || !visibleComparison) return 'promotion-validation-missing';
  if ((coverageSummary?.gpuCandidateCount ?? 0) <= 0) return 'empty-gpu-candidate';
  if ((coverageSummary?.visibleMissCount ?? 0) > 0 || coverageSummary?.visibleCoverageRatio !== 1) {
    return 'coverage-not-full';
  }
  if (candidateComparison?.anyMismatch) return 'candidate-mismatch';
  if (getVisibleItemsMismatch(visibleComparison) || getVisibleCountMismatch(visibleComparison)) {
    return 'visible-dryrun-mismatch';
  }
  if (getPackedMismatch(visibleComparison)) return 'packed-dryrun-mismatch';
  return 'none';
}

function buildScreenCoarsePromotionValidation({
  runtimeConfig,
  gpuCandidateInfo,
  candidateComparison,
  sourceComparison,
  visibleComparison,
  referenceVisibleItems
} = {}) {
  const sourceMode = runtimeConfig.sourceMode ?? 'visibleSrcIndices';
  const promotePolicy = runtimeConfig.promotePolicy ?? 'never';
  const hasReferenceVisible = Array.isArray(referenceVisibleItems);
  const coverageSummary = hasReferenceVisible
    ? buildVisibleCoverageAgainstCandidate(referenceVisibleItems, gpuCandidateInfo)
    : null;
  const failureReasons = [];

  if (sourceMode !== 'screenCoarse' || promotePolicy !== 'validated-only') {
    failureReasons.push({
      code: 'promotion-validation-missing',
      message: 'validated-only promotion applies only to screenCoarse candidate source.'
    });
  }
  if (!hasReferenceVisible || !visibleComparison) {
    failureReasons.push({
      code: 'promotion-validation-missing',
      message: 'CPU reference visible/packed comparison was not available for validated-only promotion.'
    });
  }
  if (sourceComparison?.status && sourceComparison.status !== 'ok') {
    failureReasons.push({
      code: 'gpu-error',
      message: 'screenCoarse GPU candidate source did not complete successfully.',
      details: {
        status: sourceComparison.status,
        reason: sourceComparison.reason ?? null
      }
    });
  }

  const mismatchClassification = classifyScreenCoarsePromotionMismatch({
    candidateComparison,
    visibleComparison,
    coverageSummary
  });
  if (mismatchClassification !== 'none') {
    failureReasons.push({
      code: mismatchClassification,
      message: `validated-only promotion rejected: ${mismatchClassification}`
    });
  }

  const promoted = failureReasons.length === 0;
  return {
    schemaVersion: 'step110-screen-coarse-promotion-validation-v1',
    sourceMode,
    promotePolicy,
    promotionDecision: promoted ? 'promoted' : 'fallback',
    promoted,
    displayCandidateSource: promoted ? 'gpu-candidate' : 'cpu-reference',
    gpuCandidateUsedForDisplay: promoted,
    limitedDrawUsedForCandidateSource: promoted,
    candidateInfoOverrideProvided: promoted,
    gpuCandidateCount: coverageSummary?.gpuCandidateCount ?? (gpuCandidateInfo?.candidateIndices?.length ?? 0),
    visibleCoverageRatio: coverageSummary?.visibleCoverageRatio ?? null,
    visibleMissCount: coverageSummary?.visibleMissCount ?? null,
    candidateAnyMismatch: !!candidateComparison?.anyMismatch,
    visibleItemsAnyMismatch: getVisibleItemsMismatch(visibleComparison),
    packedPayloadAnyMismatch: getPackedMismatch(visibleComparison),
    mismatchClassification,
    coverageSummary,
    failureReasons
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

function buildRemainingCpuDependencies({
  runtimeConfig,
  promotionValidation,
  useGpuCandidate
} = {}) {
  const out = [];
  if (runtimeConfig?.readbackMode === 'sync-debug') out.push('sync-debug-readback');
  if (promotionValidation) out.push('cpu-reference-visible-packed-for-validation');
  if (useGpuCandidate) {
    out.push('cpu-visible-build-from-gpu-candidate');
    out.push('cpu-packed-build-from-gpu-candidate');
    out.push('cpu-tile-list-build');
  }
  if (runtimeConfig?.coverageCompare) out.push('cpu-coverage-compare');
  if (runtimeConfig?.requireCompare) out.push('cpu-candidate-visible-packed-compare');
  return out;
}

function buildStep111TimingSummary({
  runtimeConfig,
  finalFallback,
  sourceComparison,
  sourceComparisonMs = null,
  candidateComparisonMs = null,
  referenceVisibleBuild = null,
  referenceDisplayBuildStats = null,
  gpuVisibleBuild = null,
  promotionValidation = null,
  promotionValidationMs = null,
  screenCoarsePromotionCompareMs = null,
  visibleComparisonMs = null,
  totalLimitedDrawMs = null,
  useGpuCandidate = false
} = {}) {
  const gpuTiming = sourceComparison?.gpuCandidateSummary?.timing ?? {};
  const sourceTiming = sourceComparison?.candidateSourceSummary?.timing ?? {};
  const referenceBuildStats = referenceDisplayBuildStats ?? referenceVisibleBuild?.buildStats ?? null;
  const gpuBuildStats = gpuVisibleBuild?.buildStats ?? null;
  const timingUnavailableReasons = [];
  const sourceComparisonTimingMs = Number.isFinite(sourceComparisonMs) ? sourceComparisonMs : null;
  if (sourceComparisonTimingMs === null) {
    timingUnavailableReasons.push('sourceComparisonMs');
  }
  return {
    schemaVersion: 'step111-gpu-candidate-display-timing-summary-v1',
    displayCandidateSource: finalFallback?.displayCandidateSource ?? 'cpu-reference',
    promotionDecision: useGpuCandidate ? 'promoted' : 'fallback',
    gpuCandidateUsedForDisplay: !!useGpuCandidate,
    limitedDrawUsedForCandidateSource: !!useGpuCandidate,
    candidateInfoOverrideProvided: !!useGpuCandidate,
    fallbackReason: finalFallback?.reason ?? 'unknown',
    sourceMode: runtimeConfig?.sourceMode ?? 'unknown',
    promotePolicy: runtimeConfig?.promotePolicy ?? 'never',
    readbackMode: runtimeConfig?.readbackMode ?? null,
    gpuCandidateCount: promotionValidation?.gpuCandidateCount ??
      sourceComparison?.gpuCandidateSummary?.candidateCount ??
      null,
    visibleCoverageRatio: promotionValidation?.visibleCoverageRatio ?? null,
    candidateSourceMs: Number.isFinite(gpuTiming.gpuCandidateTotalMs)
      ? gpuTiming.gpuCandidateTotalMs
      : null,
    cpuCandidateSourceMs: Number.isFinite(sourceTiming.cpuSourceTotalMs)
      ? sourceTiming.cpuSourceTotalMs
      : null,
    uploadAndSetupMs: Number.isFinite(gpuTiming.uploadAndSetupMs) ? gpuTiming.uploadAndSetupMs : null,
    transformFeedbackSetupMs: Number.isFinite(gpuTiming.transformFeedbackSetupMs)
      ? gpuTiming.transformFeedbackSetupMs
      : null,
    transformFeedbackDrawMs: Number.isFinite(gpuTiming.transformFeedbackDrawMs)
      ? gpuTiming.transformFeedbackDrawMs
      : null,
    readbackMs: Number.isFinite(gpuTiming.readbackMs) ? gpuTiming.readbackMs : null,
    collectAcceptedMs: Number.isFinite(gpuTiming.collectAcceptedMs) ? gpuTiming.collectAcceptedMs : null,
    sourceComparisonMs: sourceComparisonTimingMs,
    candidateComparisonMs,
    promotionValidationMs,
    screenCoarsePromotionCompareMs,
    visibleComparisonMs,
    referenceVisibleBuildMs: Number.isFinite(referenceBuildStats?.visibleBuildMs)
      ? referenceBuildStats.visibleBuildMs
      : null,
    referencePackedBuildMs: Number.isFinite(referenceBuildStats?.screenSpaceBuildMs)
      ? referenceBuildStats.screenSpaceBuildMs
      : (Number.isFinite(referenceBuildStats?.packedBuildMs) ? referenceBuildStats.packedBuildMs : null),
    referenceTotalBuildMs: Number.isFinite(referenceBuildStats?.totalBuildMs)
      ? referenceBuildStats.totalBuildMs
      : null,
    visibleBuildMs: Number.isFinite(gpuBuildStats?.visibleBuildMs) ? gpuBuildStats.visibleBuildMs : null,
    packedBuildMs: Number.isFinite(gpuBuildStats?.screenSpaceBuildMs)
      ? gpuBuildStats.screenSpaceBuildMs
      : (Number.isFinite(gpuBuildStats?.packedBuildMs) ? gpuBuildStats.packedBuildMs : null),
    totalVisiblePackedBuildMs: Number.isFinite(gpuBuildStats?.totalBuildMs) ? gpuBuildStats.totalBuildMs : null,
    totalLimitedDrawMs,
    candidateCount: Number.isFinite(gpuBuildStats?.candidateCount) ? gpuBuildStats.candidateCount : null,
    visibleCount: Number.isFinite(gpuBuildStats?.accepted) ? gpuBuildStats.accepted : null,
    packedVisibleCount: Number.isFinite(gpuBuildStats?.packedVisibleCount)
      ? gpuBuildStats.packedVisibleCount
      : null,
    remainingCpuDependencies: buildRemainingCpuDependencies({
      runtimeConfig,
      promotionValidation,
      useGpuCandidate
    }),
    timingUnavailableReasons
  };
}

function buildLimitedDrawFallbackReasons(
  runtimeConfig,
  gpuCandidateInfo,
  visibleComparison,
  shadowCompare,
  referenceSubsetCandidateInfo,
  candidateSourceComparison,
  promotionValidation = null
) {
  const reasons = [];
  const sourceMode = runtimeConfig.sourceMode ?? 'visibleSrcIndices';
  const isGpuOwnedSource = isGpuOwnedCandidateSourceMode(sourceMode);
  const validatedScreenCoarsePromotion = isGpuOwnedSource &&
    sourceMode === 'screenCoarse' &&
    runtimeConfig.promotePolicy === 'validated-only';
  if (isGpuOwnedSource && !validatedScreenCoarsePromotion) {
    reasons.push({
      code: 'source-mode-display-not-allowed',
      message: 'GPU-owned candidate source mode is compare-only in Step103 and is not allowed to replace the normal display candidate source.',
      details: {
        sourceMode,
        promotePolicy: runtimeConfig.promotePolicy ?? 'never',
        rangeStart: runtimeConfig.rangeStart ?? null,
        rangeCount: runtimeConfig.rangeCount ?? null,
        screenCoarseMaxCount: runtimeConfig.screenCoarseMaxCount ?? null,
        screenCoarseMinRadiusPx: runtimeConfig.screenCoarseMinRadiusPx ?? null,
        screenCoarseRequireInViewport: runtimeConfig.screenCoarseRequireInViewport ?? null,
        screenCoarseDepthMode: runtimeConfig.screenCoarseDepthMode ?? null
      }
    });
  } else if (validatedScreenCoarsePromotion && promotionValidation?.promotionDecision !== 'promoted') {
    for (const reason of promotionValidation?.failureReasons ?? []) {
      if (reason?.code) reasons.push(reason);
    }
  } else if (!isGpuOwnedSource && referenceSubsetCandidateInfo?.candidateSubsetSummary?.enabled) {
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
  if (isGpuOwnedSource && runtimeConfig.promotePolicy !== 'never' && !validatedScreenCoarsePromotion) {
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
  if (!runtimeConfig.requireShadowOk && !validatedScreenCoarsePromotion) {
    reasons.push({
      code: 'shadow-ok-required-for-limited-draw',
      message: 'limited-draw subset path requires shadow compare gating.'
    });
  } else if (!validatedScreenCoarsePromotion && shadowCompare?.status !== 'ok') {
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
  referencePackedPayload = null,
  referenceDisplayBuildStats = null,
  shadowCompare = null,
  camera = null,
  screenSpaceCamera = null,
  canvasWidth = 0,
  canvasHeight = 0,
  camPos = null,
  tileGrid = null,
  buildConfig = null
} = {}) {
  const totalStartMs = nowMs();
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
  const sourceCompareStartMs = nowMs();
  const sourceComparison = gpuOwnedSourceMode
    ? buildGpuOwnedCandidateSourceComparison({
        gl,
        raw,
        runtimeConfig,
        referenceCandidateInfo,
        filterMode: runtimeConfig.filterMode,
        camera,
        screenSpaceCamera,
        canvasWidth,
        canvasHeight,
        camPos,
        tileGrid,
        buildConfig,
        metadata: {
          candidateArgs: summarizeCandidateArgs(candidateArgs)
        }
      })
    : null;
  const sourceComparisonMs = nowMs() - sourceCompareStartMs;
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
  const candidateComparisonStartMs = nowMs();
  const candidateComparison = buildCandidateComparisonSummary({
    referenceCandidateInfo: referenceFilteredCandidateInfo,
    candidateCandidateInfo: gpuCandidateInfo,
    referenceLabel: gpuOwnedSourceMode
      ? `cpu-${sourceMode}-candidate-source-reference`
      : 'cpu-filtered-candidate-reference',
    candidateLabel: gpuOwnedSourceMode
      ? `gpu-${sourceMode}-candidate-source`
      : 'gpu-candidate-limited-draw',
    options: { maxMismatches: 16 },
    metadata: {
      comparisonMode: gpuOwnedSourceMode
        ? `cpu-${sourceMode}-candidate-source-vs-gpu-${sourceMode}-candidate-source`
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
  const candidateComparisonMs = nowMs() - candidateComparisonStartMs;
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
  const canBuildScreenCoarsePromotionValidation = gpuOwnedSourceMode &&
    sourceMode === 'screenCoarse' &&
    runtimeConfig.promotePolicy === 'validated-only' &&
    canBuildVisibleComparison &&
    Array.isArray(visibleSourceItems);
  const screenCoarsePromotionCompareStartMs = nowMs();
  const screenCoarsePromotionVisibleComparison = canBuildScreenCoarsePromotionValidation
    ? buildVisibleComparisonSummary({
        referenceItems: visibleSourceItems,
        candidateItems: gpuVisibleBuild.visible,
        referencePackedPayload,
        candidatePackedPayload: gpuVisibleBuild.packedScreenSpace,
        referenceLabel: 'cpu-reference-visible-render-result',
        candidateLabel: 'gpu-screen-coarse-candidate-visible-limited-draw',
        options: { epsilon: 1e-6, maxMismatches: 16 },
        metadata: {
          comparisonMode: 'cpu-reference-visible-vs-gpu-screen-coarse-candidate-limited-draw',
          referenceVisibleCount: visibleSourceItems.length,
          referencePackedCount: Number.isFinite(referencePackedPayload?.packedCount)
            ? referencePackedPayload.packedCount
            : null,
          candidateBuildStats: gpuVisibleBuild.buildStats
        }
      })
    : null;
  const screenCoarsePromotionCompareMs = canBuildScreenCoarsePromotionValidation
    ? nowMs() - screenCoarsePromotionCompareStartMs
    : null;
  const visibleComparisonStartMs = nowMs();
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
  const visibleComparisonMs = canBuildVisibleComparison ? nowMs() - visibleComparisonStartMs : null;
  const promotionValidationStartMs = nowMs();
  const promotionValidation = gpuOwnedSourceMode && sourceMode === 'screenCoarse'
    ? buildScreenCoarsePromotionValidation({
        runtimeConfig,
        gpuCandidateInfo,
        candidateComparison,
        sourceComparison,
        visibleComparison: screenCoarsePromotionVisibleComparison,
        referenceVisibleItems: visibleSourceItems
      })
    : null;
  const promotionValidationMs = promotionValidation ? nowMs() - promotionValidationStartMs : null;
  const visibleComparisonForFallback = promotionValidation
    ? screenCoarsePromotionVisibleComparison
    : visibleComparison;
  const finalFallback = buildGpuCandidateRuntimeFallbackSummary({
    runtimeConfig,
    shadowCompare: {
      status: promotionValidation ? 'ok' : (shadowCompare?.status ?? 'ok'),
      candidateComparison: promotionValidation ? null : candidateComparison,
      visibleComparison: promotionValidation ? null : visibleComparison,
      summary: {
        anyMismatch: promotionValidation
          ? false
          : !!(candidateComparison.anyMismatch || visibleComparison?.anyMismatch)
      }
    },
    extraReasons: buildLimitedDrawFallbackReasons(
      runtimeConfig,
      gpuCandidateInfo,
      visibleComparisonForFallback,
      shadowCompare,
      referenceSubsetCandidateInfo,
      sourceComparison,
      promotionValidation
    )
  });
  const useGpuCandidate = finalFallback.action === 'use-gpu-candidate';
  const step111TimingSummary = buildStep111TimingSummary({
    runtimeConfig,
    finalFallback,
    sourceComparison,
    sourceComparisonMs,
    candidateComparisonMs,
    referenceVisibleBuild,
    referenceDisplayBuildStats,
    gpuVisibleBuild,
    promotionValidation,
    promotionValidationMs,
    screenCoarsePromotionCompareMs,
    visibleComparisonMs,
    totalLimitedDrawMs: nowMs() - totalStartMs,
    useGpuCandidate
  });

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
      promotionDecision: useGpuCandidate ? 'promoted' : 'fallback',
      promotionValidation,
      step111TimingSummary,
      remainingCpuDependencies: step111TimingSummary.remainingCpuDependencies,
      status: useGpuCandidate ? 'using-gpu-candidate' : 'fallback',
      reason: finalFallback.reason,
      runtimeSummary,
      fallback: finalFallback,
      candidateArgs: summarizeCandidateArgs(candidateArgs),
      candidateSourceSummary: sourceComparison?.candidateSourceSummary ?? null,
      candidateCoverageSummary: promotionValidation?.coverageSummary ?? null,
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
      screenCoarsePromotionVisibleComparison,
      referenceVisibleBuildSummary: referenceVisibleBuild?.buildStats ?? null,
      gpuVisibleBuildSummary: gpuVisibleBuild?.buildStats ?? null
    }
  };
}
