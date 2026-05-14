import {
  buildCpuFilteredCandidateInfo,
  buildGpuFirstNCandidateInfo
} from './gpu_candidate_builder_gpu_firstn.js';
import {
  buildCpuScreenCoarseCandidateSourceInfo,
  buildGpuScreenCoarseCandidateInfo
} from './gpu_candidate_screen_coarse_runtime.js';
import { buildCandidateComparisonSummary } from './gpu_visible_compare_debug.js';

const SOURCE_MODE_VALUES = new Set(['visibleSrcIndices', 'firstN', 'range', 'screenCoarse']);
const DEFAULT_SOURCE_MODE = 'visibleSrcIndices';
const DEFAULT_RANGE_START = 0;
const DEFAULT_RANGE_COUNT = 65536;
const DEFAULT_SCREEN_COARSE_MAX_COUNT = 65536;
const DEFAULT_SCREEN_COARSE_MIN_RADIUS_PX = 0.25;
const SCREEN_COARSE_DEPTH_MODE_VALUES = new Set(['positive', 'any']);

function clonePlainObject(value) {
  if (!value || typeof value !== 'object') return value;
  if (Array.isArray(value)) return value.map((item) => clonePlainObject(item));
  const out = {};
  for (const [key, item] of Object.entries(value)) out[key] = clonePlainObject(item);
  return out;
}

function toFiniteInteger(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(0, n | 0) : fallback;
}

function toFiniteNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function normalizeScreenCoarseDepthMode(value) {
  return SCREEN_COARSE_DEPTH_MODE_VALUES.has(value) ? value : 'positive';
}

export function normalizeGpuCandidateSourceMode(value) {
  return SOURCE_MODE_VALUES.has(value) ? value : DEFAULT_SOURCE_MODE;
}

export function buildGpuCandidateSourceConfig(runtimeConfig = {}) {
  const sourceMode = normalizeGpuCandidateSourceMode(runtimeConfig.sourceMode);
  return {
    schemaVersion: 'step103-gpu-candidate-source-config-v1',
    sourceMode,
    rangeStart: toFiniteInteger(runtimeConfig.rangeStart, DEFAULT_RANGE_START),
    rangeCount: toFiniteInteger(runtimeConfig.rangeCount, DEFAULT_RANGE_COUNT),
    screenCoarseMaxCount: toFiniteInteger(runtimeConfig.screenCoarseMaxCount, DEFAULT_SCREEN_COARSE_MAX_COUNT),
    screenCoarseMinRadiusPx: Math.max(
      0,
      toFiniteNumber(runtimeConfig.screenCoarseMinRadiusPx, DEFAULT_SCREEN_COARSE_MIN_RADIUS_PX)
    ),
    screenCoarseRequireInViewport: runtimeConfig.screenCoarseRequireInViewport !== false,
    screenCoarseDepthMode: normalizeScreenCoarseDepthMode(runtimeConfig.screenCoarseDepthMode),
    promotePolicy: runtimeConfig.promotePolicy ?? 'never',
    readbackMode: runtimeConfig.readbackMode ?? 'sync-debug'
  };
}

export function isGpuOwnedCandidateSourceMode(sourceMode) {
  const normalized = normalizeGpuCandidateSourceMode(sourceMode);
  return normalized === 'range' || normalized === 'screenCoarse';
}

export function buildCpuRangeCandidateSourceInfo({
  raw = null,
  referenceCandidateInfo = null,
  rangeStart = DEFAULT_RANGE_START,
  rangeCount = DEFAULT_RANGE_COUNT,
  candidateMode = 'cpu-range-candidate-source-reference'
} = {}) {
  const total = raw && Number.isFinite(raw.N)
    ? Math.max(0, raw.N | 0)
    : Math.max(0, referenceCandidateInfo?.rangeSummary?.totalCount ?? 0);
  const start = Math.min(toFiniteInteger(rangeStart, DEFAULT_RANGE_START), total);
  const requestedCount = toFiniteInteger(rangeCount, DEFAULT_RANGE_COUNT);
  const count = Math.max(0, Math.min(requestedCount, total - start));
  const candidateIndices = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    candidateIndices[i] = start + i;
  }
  const rangeSummary = {
    ...(clonePlainObject(referenceCandidateInfo?.rangeSummary) ?? {}),
    totalCount: total,
    rangeStart: start,
    rangeCount: count,
    requestedRangeCount: requestedCount,
    candidateCount: count,
    rangeFraction: total > 0 ? count / total : 0,
    candidateFraction: total > 0 ? count / total : 0
  };
  const sourceSummary = {
    schemaVersion: 'step103-gpu-candidate-source-summary-v1',
    sourceMode: 'range',
    contract: 'gpu-owned-range-candidate-source-v1',
    cpuVisibleDependent: false,
    rangeStart: start,
    rangeCount: count,
    requestedRangeCount: requestedCount,
    candidateCount: count,
    candidateOrder: 'ascending-source-index',
    promotePolicy: 'never'
  };
  return {
    candidateIndices,
    candidateMode,
    temporalWindow: clonePlainObject(referenceCandidateInfo?.temporalWindow) ?? null,
    rangeSummary,
    temporalIndexDebug: clonePlainObject(referenceCandidateInfo?.temporalIndexDebug) ?? null,
    temporalBucketDebug: clonePlainObject(referenceCandidateInfo?.temporalBucketDebug) ?? null,
    candidateSourceSummary: sourceSummary,
    candidateSubsetSummary: {
      enabled: true,
      contract: 'gpu-owned-candidate-source-range-v1',
      subsetMode: 'range',
      sourceMode: 'range',
      subsetCount: count,
      selectedCandidateCount: count,
      rangeStart: start,
      rangeCount: count,
      requestedRangeCount: requestedCount,
      sourceCandidateMode: referenceCandidateInfo?.candidateMode ?? 'unknown',
      sourceCandidateCount: referenceCandidateInfo?.candidateIndices?.length ?? total,
      cpuVisibleDependent: false
    }
  };
}

export function buildGpuOwnedCandidateSourceComparison({
  gl,
  raw = null,
  runtimeConfig = {},
  referenceCandidateInfo = null,
  filterMode = 'all-valid',
  camera = null,
  screenSpaceCamera = null,
  canvasWidth = 0,
  canvasHeight = 0,
  camPos = null,
  tileGrid = null,
  buildConfig = null,
  metadata = {}
} = {}) {
  const sourceConfig = buildGpuCandidateSourceConfig(runtimeConfig);
  if (sourceConfig.sourceMode !== 'range' && sourceConfig.sourceMode !== 'screenCoarse') {
    return {
      schemaVersion: 'step107-gpu-candidate-source-compare-v1',
      status: 'skipped',
      reason: 'unsupported-source-mode-for-gpu-owned-compare',
      sourceConfig,
      candidateSourceSummary: null,
      cpuMirrorCandidateSummary: null,
      gpuCandidateSummary: null,
      candidateComparison: null
    };
  }

  let cpuSourceInfo = null;
  let cpuMirrorInfo = null;
  let gpuCandidateInfo = null;
  let referenceLabel = 'cpu-range-candidate-source-reference';
  let candidateLabel = 'gpu-range-candidate-source';
  let comparisonMode = 'cpu-range-candidate-source-vs-gpu-range-candidate-source';
  if (sourceConfig.sourceMode === 'screenCoarse') {
    cpuSourceInfo = buildCpuScreenCoarseCandidateSourceInfo({
      raw,
      referenceCandidateInfo,
      runtimeConfig: sourceConfig,
      filterMode,
      camera,
      screenSpaceCamera,
      canvasWidth,
      canvasHeight,
      camPos,
      tileGrid,
      buildConfig
    });
    cpuMirrorInfo = cpuSourceInfo;
    gpuCandidateInfo = buildGpuScreenCoarseCandidateInfo({
      gl,
      raw,
      referenceCandidateInfo,
      cpuScreenCoarseSourceInfo: cpuSourceInfo,
      runtimeConfig: sourceConfig,
      filterMode
    });
    referenceLabel = 'cpu-screen-coarse-candidate-source-reference';
    candidateLabel = 'gpu-screen-coarse-candidate-source';
    comparisonMode = 'cpu-screen-coarse-candidate-source-vs-gpu-screen-coarse-candidate-source';
  } else {
    cpuSourceInfo = buildCpuRangeCandidateSourceInfo({
      raw,
      referenceCandidateInfo,
      rangeStart: sourceConfig.rangeStart,
      rangeCount: sourceConfig.rangeCount
    });
    cpuMirrorInfo = buildCpuFilteredCandidateInfo({
      raw,
      referenceSubsetCandidateInfo: cpuSourceInfo,
      filterMode,
      candidateMode: 'cpu-range-candidate-filter-reference'
    });
    gpuCandidateInfo = buildGpuFirstNCandidateInfo({
      gl,
      raw,
      referenceSubsetCandidateInfo: cpuSourceInfo,
      subsetCount: sourceConfig.rangeCount,
      startIndex: sourceConfig.rangeStart,
      filterMode
    });
  }
  const candidateComparison = buildCandidateComparisonSummary({
    referenceCandidateInfo: cpuMirrorInfo,
    candidateCandidateInfo: gpuCandidateInfo,
    referenceLabel,
    candidateLabel,
    options: { maxMismatches: 16 },
    metadata: {
      comparisonMode,
      sourceConfig,
      candidateSourceSummary: cpuSourceInfo.candidateSourceSummary,
      cpuFilterSummary: cpuMirrorInfo.filterSummary ?? null,
      gpuCandidateSummary: gpuCandidateInfo.gpuCandidateSummary ?? null,
      filterSummary: gpuCandidateInfo.filterSummary ?? null,
      readbackPolicy: runtimeConfig.readbackPolicy ?? null,
      ...metadata
    }
  });
  return {
    schemaVersion: 'step107-gpu-candidate-source-compare-v1',
    status: gpuCandidateInfo.gpuCandidateSummary?.status === 'ok' ? 'ok' : 'fallback',
    reason: gpuCandidateInfo.gpuCandidateSummary?.reason ?? 'ok',
    sourceConfig,
    candidateSourceSummary: cpuSourceInfo.candidateSourceSummary,
    cpuMirrorCandidateSummary: {
      candidateMode: cpuMirrorInfo.candidateMode,
      candidateCount: cpuMirrorInfo.candidateIndices?.length ?? 0,
      filterSummary: cpuMirrorInfo.filterSummary ?? null,
      rangeSummary: cpuMirrorInfo.rangeSummary ?? null,
      screenCoarseSummary: cpuMirrorInfo.screenCoarseSummary ?? null
    },
    gpuCandidateSummary: gpuCandidateInfo.gpuCandidateSummary ?? null,
    candidateComparison,
    cpuMirrorCandidateInfo: cpuMirrorInfo,
    gpuCandidateInfo
  };
}
