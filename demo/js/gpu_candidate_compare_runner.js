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
import { buildGpuOwnedCandidateSourceComparison } from './gpu_candidate_source_runtime.js';

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
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
  renderScale,
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
  tileGrid = null,
  temporalSigmaThreshold = 3.0,
  enablePackedVisiblePath = true,
  label = 'candidate-dry-run'
}) {
  const t0 = nowMs();
  const renderW = Math.max(1, Math.round(canvasWidth * renderScale));
  const renderH = Math.max(1, Math.round(canvasHeight * renderScale));
  const sx = canvasWidth / renderW;
  const sy = canvasHeight / renderH;
  const flags = {
    nativeRot4d: !!useNativeRot4d,
    nativeMarginal: !!useNativeMarginal
  };
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
  let minTileX = tileGrid ? tileGrid.tileCols : 0;
  let minTileY = tileGrid ? tileGrid.tileRows : 0;
  let maxTileX = -1;
  let maxTileY = -1;

  const loopStartMs = nowMs();
  for (let k = 0; k < candidateIndices.length; k++) {
    const index = candidateIndices[k];
    processed++;
    if (!passesTemporalCulling(raw, index, timestamp, sigmaScale, temporalSigmaThreshold)) {
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
      timestamp,
      scalingModifier,
      sigmaScale,
      prefilterVar,
      useRot4d,
      flags,
      camPos,
      timeDuration,
      useSH,
      forceSh3d,
      tileGrid
    });
    if (!item) {
      culled++;
      continue;
    }
    if (item.tileRange) {
      minTileX = Math.min(minTileX, item.tileRange[0]);
      minTileY = Math.min(minTileY, item.tileRange[1]);
      maxTileX = Math.max(maxTileX, item.tileRange[2]);
      maxTileY = Math.max(maxTileY, item.tileRange[3]);
    }
    visible.push(item);
    if (visible.length >= maxVisible) break;
  }
  visible.sort((a, b) => b.depth - a.depth);
  const visibleBuildMs = nowMs() - loopStartMs;
  const activeTileBox = tileGrid && maxTileX >= minTileX && maxTileY >= minTileY
    ? [minTileX, minTileY, maxTileX, maxTileY]
    : null;
  const packedStartMs = nowMs();
  const packedScreenSpace = enablePackedVisiblePath
    ? buildPackedScreenSpaceWithContext(createScreenSpaceBuildContext(), visible, {
        renderW,
        renderH,
        sx,
        sy
      })
    : null;
  const screenSpaceBuildMs = nowMs() - packedStartMs;

  return {
    label,
    candidateInfo,
    visible,
    packedScreenSpace,
    renderW,
    renderH,
    sx,
    sy,
    activeTileBox,
    buildStats: {
      label,
      accepted: visible.length,
      processed,
      culled,
      temporalRejected,
      temporalPassed,
      temporalCullRatio: processed > 0 ? temporalRejected / processed : 0,
      candidateMode: candidateInfo?.candidateMode ?? 'unknown',
      candidateCount: candidateIndices.length,
      visibleBuildMs,
      screenSpaceBuildMs,
      totalBuildMs: nowMs() - t0,
      packedVisiblePathEnabled: !!enablePackedVisiblePath,
      packedVisibleCount: packedScreenSpace?.packedCount ?? 0,
      packedVisibleLength: packedScreenSpace?.packed?.length ?? 0,
      packedVisibleFloatsPerItem: packedScreenSpace?.floatsPerItem ?? 0
    }
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

function toFiniteInteger(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(0, n | 0) : fallback;
}

function toFiniteNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function buildScreenCoarseRuntimeConfig(options = {}) {
  return {
    sourceMode: 'screenCoarse',
    promotePolicy: 'never',
    readbackMode: options.readbackMode ?? 'sync-debug',
    screenCoarseMaxCount: toFiniteInteger(options.maxCount ?? options.screenCoarseMaxCount, 65536),
    screenCoarseMinRadiusPx: Math.max(
      0,
      toFiniteNumber(options.minRadiusPx ?? options.screenCoarseMinRadiusPx, 0.25)
    ),
    screenCoarseRequireInViewport:
      typeof (options.requireInViewport ?? options.screenCoarseRequireInViewport) === 'boolean'
        ? (options.requireInViewport ?? options.screenCoarseRequireInViewport)
        : true,
    screenCoarseDepthMode: options.depthMode ?? options.screenCoarseDepthMode ?? 'positive'
  };
}

function buildVisibleCoverageAgainstCandidate(referenceItems, candidateInfo, maxMissingSamples = 32) {
  const candidateIndices = candidateInfo?.candidateIndices instanceof Uint32Array
    ? candidateInfo.candidateIndices
    : (Array.isArray(candidateInfo?.candidateIndices) ? Uint32Array.from(candidateInfo.candidateIndices) : new Uint32Array(0));
  const candidateSet = new Set();
  for (let i = 0; i < candidateIndices.length; i++) candidateSet.add(candidateIndices[i] >>> 0);

  const visible = Array.isArray(referenceItems) ? referenceItems : [];
  const missing = [];
  let hitCount = 0;
  let missCount = 0;
  const sampleLimit = toFiniteInteger(maxMissingSamples, 32);
  for (const item of visible) {
    const srcIndex = Number(item?.srcIndex);
    if (!Number.isFinite(srcIndex) || srcIndex < 0) continue;
    const index = srcIndex >>> 0;
    if (candidateSet.has(index)) {
      hitCount++;
    } else {
      missCount++;
      if (missing.length < sampleLimit) missing.push(index);
    }
  }
  const count = hitCount + missCount;
  return {
    schemaVersion: 'step108-screen-coarse-dry-run-visible-coverage-v1',
    sourceMode: 'screenCoarse',
    gpuCandidateCount: candidateIndices.length,
    cpuVisibleCount: count,
    visibleHitCount: hitCount,
    visibleMissCount: missCount,
    visibleCoverageRatio: count > 0 ? hitCount / count : 1,
    missingVisibleSrcIndices: missing,
    maxMissingSamples: sampleLimit
  };
}

function classifyDryRunMismatch({
  candidateComparison = null,
  visibleComparison = null,
  coverageSummary = null
} = {}) {
  const candidateIndices = candidateComparison?.candidateIndices ?? {};
  const visibleItems = visibleComparison?.visibleItems ?? {};
  const packedPayload = visibleComparison?.packedPayload ?? {};

  if ((coverageSummary?.visibleMissCount ?? 0) > 0) return 'candidate-shortage';
  if (candidateIndices.anyMismatch) return 'candidate-order-mismatch';
  if (visibleItems.countEqual === false) return 'visible-count-mismatch';
  if ((visibleItems.orderMismatchCount ?? 0) > 0 || (visibleItems.itemMismatchCount ?? 0) > 0) {
    return 'visible-item-field-mismatch';
  }
  if (packedPayload.countEqual === false) return 'packed-count-mismatch';
  if (
    packedPayload.referencePresent === false ||
    packedPayload.candidatePresent === false ||
    packedPayload.lengthEqual === false ||
    packedPayload.referenceFloatsPerItem !== packedPayload.candidateFloatsPerItem
  ) {
    return 'packed-layout-mismatch';
  }
  if ((packedPayload.mismatchCount ?? 0) > 0) return 'packed-value-mismatch';
  return 'none';
}

export function captureGpuCandidateDryRunVisibleComparison({
  gl,
  raw,
  camera,
  screenSpaceCamera = null,
  canvasWidth,
  canvasHeight,
  camPos,
  tileGrid = null,
  buildConfig,
  candidateArgs,
  subsetCount = 1024,
  subsetMode = 'visibleSrcIndices',
  startIndex = 0,
  filterMode = 'all-valid',
  maxMismatches = 16,
  epsilon = 1e-6,
  visibleSourceItems = null,
  metadata = {}
} = {}) {
  const referenceCandidateInfo = buildCandidateInfo(candidateArgs);
  const referenceSubsetCandidateInfo = buildReferenceSubsetCandidateInfo({
    raw,
    referenceCandidateInfo,
    subsetCount,
    subsetMode,
    visibleSourceItems
  });
  const referenceFilteredCandidateInfo = buildCpuFilteredCandidateInfo({
    raw,
    referenceSubsetCandidateInfo,
    filterMode,
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
        filterMode
      })
    : buildGpuFirstNCandidateInfo({
        gl,
        raw,
        referenceSubsetCandidateInfo,
        subsetCount,
        startIndex,
        filterMode
      });
  const dryRunArgs = {
    raw,
    camera,
    screenSpaceCamera,
    canvasWidth,
    canvasHeight,
    camPos,
    tileGrid,
    renderScale: buildConfig.renderScale,
    maxVisible: buildConfig.maxVisible,
    timestamp: buildConfig.timestamp,
    scalingModifier: buildConfig.scalingModifier,
    sigmaScale: buildConfig.sigmaScale,
    prefilterVar: buildConfig.prefilterVar,
    useSH: buildConfig.useSH,
    useRot4d: buildConfig.useRot4d,
    useNativeRot4d: buildConfig.useNativeRot4d,
    useNativeMarginal: buildConfig.useNativeMarginal,
    forceSh3d: buildConfig.forceSh3d,
    timeDuration: buildConfig.timeDuration,
    temporalSigmaThreshold: 3.0,
    enablePackedVisiblePath: !!buildConfig.enablePackedVisiblePath
  };
  const referenceDryRun = buildVisibleAndPackedFromCandidateInfo({
    ...dryRunArgs,
    candidateInfo: referenceFilteredCandidateInfo,
    label: 'cpu-filtered-candidate-visible-dry-run'
  });
  const gpuDryRun = buildVisibleAndPackedFromCandidateInfo({
    ...dryRunArgs,
    candidateInfo: gpuCandidateInfo,
    label: 'gpu-candidate-visible-dry-run'
  });
  const candidateComparison = buildCandidateComparisonSummary({
    referenceCandidateInfo: referenceFilteredCandidateInfo,
    candidateCandidateInfo: gpuCandidateInfo,
    referenceLabel: 'cpu-filtered-candidate-reference',
    candidateLabel: 'gpu-candidate-dry-run',
    options: { maxMismatches },
    metadata: {
      comparisonMode: 'cpu-filtered-candidate-vs-gpu-candidate-dry-run',
      subset: referenceSubsetCandidateInfo.candidateSubsetSummary ?? null,
      selectedCandidateCount: referenceSubsetCandidateInfo.candidateIndices?.length ?? 0,
      cpuFilterSummary: referenceFilteredCandidateInfo.filterSummary ?? null,
      gpuCandidateSummary: gpuCandidateInfo.gpuCandidateSummary ?? null,
      filterSummary: gpuCandidateInfo.filterSummary ?? null,
      candidateArgs: summarizeCandidateArgs(candidateArgs)
    }
  });
  const visibleComparison = buildVisibleComparisonSummary({
    referenceItems: referenceDryRun.visible,
    candidateItems: gpuDryRun.visible,
    referencePackedPayload: referenceDryRun.packedScreenSpace,
    candidatePackedPayload: gpuDryRun.packedScreenSpace,
    referenceLabel: 'cpu-filtered-visible-dry-run',
    candidateLabel: 'gpu-candidate-visible-dry-run',
    options: { epsilon, maxMismatches },
    metadata: {
      comparisonMode: 'cpu-filtered-visible-vs-gpu-candidate-visible-dry-run',
      referenceBuildStats: referenceDryRun.buildStats,
      candidateBuildStats: gpuDryRun.buildStats
    }
  });

  return {
    schemaVersion: 'step95-gpu-candidate-dry-run-visible-comparison-v1',
    timestamp: new Date().toISOString(),
    purpose: 'Build visible items and packed payloads from GPU candidate output in a dry-run path without changing rendering.',
    metadata: {
      ...metadata,
      subsetCount,
      subsetMode: referenceSubsetCandidateInfo.candidateSubsetSummary?.subsetMode ?? subsetMode,
      selectedCandidateCount: referenceSubsetCandidateInfo.candidateIndices?.length ?? 0,
      startIndex,
      filterMode,
      readbackWarning: 'Synchronous transform feedback readback is debug-only; runtime path needs readback/fence-sync design before use.',
      candidateArgs: summarizeCandidateArgs(candidateArgs)
    },
    candidateComparison,
    visibleComparison,
    referenceDryRunSummary: referenceDryRun.buildStats,
    gpuDryRunSummary: gpuDryRun.buildStats,
    anyMismatch: !!(candidateComparison.anyMismatch || visibleComparison.anyMismatch)
  };
}

export function captureGpuCandidateScreenCoarseDryRunVisibleComparison({
  gl,
  raw,
  camera,
  screenSpaceCamera = null,
  canvasWidth,
  canvasHeight,
  camPos,
  tileGrid = null,
  buildConfig,
  candidateArgs,
  referenceVisibleItems = null,
  referencePackedPayload = null,
  maxMismatches = 16,
  maxMissingSamples = 32,
  epsilon = 1e-6,
  filterMode = 'all-valid',
  readbackMode = 'sync-debug',
  maxCount = 65536,
  minRadiusPx = 0.25,
  requireInViewport = true,
  depthMode = 'positive',
  metadata = {}
} = {}) {
  const referenceCandidateInfo = buildCandidateInfo(candidateArgs);
  const runtimeConfig = buildScreenCoarseRuntimeConfig({
    readbackMode,
    maxCount,
    minRadiusPx,
    requireInViewport,
    depthMode
  });
  const sourceComparison = buildGpuOwnedCandidateSourceComparison({
    gl,
    raw,
    runtimeConfig,
    referenceCandidateInfo,
    filterMode,
    camera,
    screenSpaceCamera,
    canvasWidth,
    canvasHeight,
    camPos,
    tileGrid,
    buildConfig,
    metadata: {
      candidateArgs: summarizeCandidateArgs(candidateArgs),
      phase: 'step108-screen-coarse-dry-run-visible'
    }
  });
  const gpuCandidateInfo = sourceComparison?.gpuCandidateInfo ?? null;
  const dryRunArgs = {
    raw,
    camera,
    screenSpaceCamera,
    canvasWidth,
    canvasHeight,
    camPos,
    tileGrid,
    renderScale: buildConfig.renderScale,
    maxVisible: buildConfig.maxVisible,
    timestamp: buildConfig.timestamp,
    scalingModifier: buildConfig.scalingModifier,
    sigmaScale: buildConfig.sigmaScale,
    prefilterVar: buildConfig.prefilterVar,
    useSH: buildConfig.useSH,
    useRot4d: buildConfig.useRot4d,
    useNativeRot4d: buildConfig.useNativeRot4d,
    useNativeMarginal: buildConfig.useNativeMarginal,
    forceSh3d: buildConfig.forceSh3d,
    timeDuration: buildConfig.timeDuration,
    temporalSigmaThreshold: candidateArgs?.temporalSigmaThreshold ?? 3.0,
    enablePackedVisiblePath: !!buildConfig.enablePackedVisiblePath
  };
  const gpuDryRun = buildVisibleAndPackedFromCandidateInfo({
    ...dryRunArgs,
    candidateInfo: gpuCandidateInfo,
    label: 'gpu-screen-coarse-candidate-visible-dry-run'
  });
  const visibleComparison = buildVisibleComparisonSummary({
    referenceItems: Array.isArray(referenceVisibleItems) ? referenceVisibleItems : [],
    candidateItems: gpuDryRun.visible,
    referencePackedPayload,
    candidatePackedPayload: gpuDryRun.packedScreenSpace,
    referenceLabel: 'cpu-reference-visible-render-result',
    candidateLabel: 'gpu-screen-coarse-candidate-visible-dry-run',
    options: { epsilon, maxMismatches },
    metadata: {
      comparisonMode: 'cpu-reference-visible-vs-gpu-screen-coarse-candidate-visible-dry-run',
      referenceVisibleCount: Array.isArray(referenceVisibleItems) ? referenceVisibleItems.length : 0,
      referencePackedCount: Number.isFinite(referencePackedPayload?.packedCount)
        ? referencePackedPayload.packedCount
        : null,
      candidateBuildStats: gpuDryRun.buildStats
    }
  });
  const coverageSummary = buildVisibleCoverageAgainstCandidate(
    referenceVisibleItems,
    gpuCandidateInfo,
    maxMissingSamples
  );
  const mismatchClassification = classifyDryRunMismatch({
    candidateComparison: sourceComparison?.candidateComparison ?? null,
    visibleComparison,
    coverageSummary
  });
  return {
    schemaVersion: 'step108-gpu-candidate-screen-coarse-dryrun-visible-compare-v1',
    timestamp: new Date().toISOString(),
    status: sourceComparison?.status === 'ok' ? 'ok' : 'fallback',
    reason: sourceComparison?.reason ?? 'ok',
    purpose: 'Build visible items and packed payloads from screenCoarse GPU candidate output in a dry-run path without changing rendering.',
    sourceMode: 'screenCoarse',
    displayCandidateSource: 'cpu-reference',
    gpuCandidateUsedForDisplay: false,
    limitedDrawUsedForCandidateSource: false,
    metadata: {
      ...metadata,
      promotePolicy: 'never',
      filterMode,
      readbackMode,
      screenCoarse: {
        maxCount: runtimeConfig.screenCoarseMaxCount,
        minRadiusPx: runtimeConfig.screenCoarseMinRadiusPx,
        requireInViewport: runtimeConfig.screenCoarseRequireInViewport,
        depthMode: runtimeConfig.screenCoarseDepthMode
      },
      candidateArgs: summarizeCandidateArgs(candidateArgs)
    },
    candidateSourceSummary: sourceComparison?.candidateSourceSummary ?? null,
    candidateSourceComparison: sourceComparison,
    candidateComparison: sourceComparison?.candidateComparison ?? null,
    gpuCandidateSummary: sourceComparison?.gpuCandidateSummary ?? gpuCandidateInfo?.gpuCandidateSummary ?? null,
    coverageSummary,
    visibleComparison,
    dryRunVisibleComparison: visibleComparison,
    gpuDryRunSummary: gpuDryRun.buildStats,
    mismatchClassification,
    anyMismatch: mismatchClassification !== 'none'
  };
}
