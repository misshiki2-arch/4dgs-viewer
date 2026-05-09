import { captureGpuCandidateDryRunVisibleComparison } from './gpu_candidate_compare_runner.js';
import {
  buildGpuCandidateRuntimeConfig,
  buildGpuCandidateRuntimeSummary
} from './gpu_candidate_runtime_selector.js';
import { buildGpuCandidateRuntimeFallbackSummary } from './gpu_candidate_runtime_fallback.js';

const RUNTIME_SHADOW_COMPARE = 'shadow-compare';
const DEFAULT_SUBSET_MODE = 'visibleSrcIndices';
const DEFAULT_SUBSET_COUNT = 1024;
const DEFAULT_FILTER_MODE = 'all-valid';

function toFiniteInteger(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(0, n | 0) : fallback;
}

export function buildGpuCandidateShadowOptionsFromQuery(queryState = {}, overrides = {}) {
  const runtimeConfig = buildGpuCandidateRuntimeConfig(queryState, overrides);
  const runtime = runtimeConfig.requestedRuntime === 'cpu-reference'
    ? 'off'
    : runtimeConfig.requestedRuntime;
  const compare = typeof overrides.compare === 'boolean'
    ? overrides.compare
    : (typeof overrides.gpuCandidateCompare === 'boolean'
      ? overrides.gpuCandidateCompare
      : runtimeConfig.compareRequested);
  return {
    runtime,
    compare,
    requestedRuntime: runtimeConfig.requestedRuntime,
    runtimeConfig,
    runtimeSummary: buildGpuCandidateRuntimeSummary(runtimeConfig),
    subsetMode: runtimeConfig.subsetMode ?? DEFAULT_SUBSET_MODE,
    subsetCount: toFiniteInteger(runtimeConfig.subsetCount, DEFAULT_SUBSET_COUNT),
    startIndex: toFiniteInteger(runtimeConfig.startIndex, 0),
    filterMode: runtimeConfig.filterMode ?? DEFAULT_FILTER_MODE,
    maxMismatches: toFiniteInteger(overrides.maxMismatches, 16),
    epsilon: Number.isFinite(Number(overrides.epsilon)) ? Number(overrides.epsilon) : 1e-6
  };
}

export function isGpuCandidateShadowCompareEnabled(options = {}) {
  return (options.runtime === RUNTIME_SHADOW_COMPARE || options.runtime === 'limited-draw') &&
    options.compare === true;
}

function buildShadowFailureResult({ options, metadata, error, status = 'failed' }) {
  const message = error?.message ?? String(error ?? 'unknown error');
  const fallback = buildGpuCandidateRuntimeFallbackSummary({
    runtimeConfig: options.runtimeConfig,
    error
  });
  return {
    schemaVersion: 'step96-gpu-candidate-shadow-compare-v1',
    timestamp: new Date().toISOString(),
    status,
    runtime: options.runtime ?? 'off',
    compareOnly: true,
    gpuCandidateUsedForDisplay: false,
    displayCandidateSource: 'cpu-reference',
    runtimeSummary: options.runtimeSummary ?? buildGpuCandidateRuntimeSummary(options.runtimeConfig),
    fallback,
    readbackWarning: 'GPU candidate shadow compare may use synchronous debug readback; keep it out of normal rendering until readback/fence-sync is designed.',
    options,
    metadata,
    error: {
      message,
      name: error?.name ?? null,
      stack: error?.stack ?? null
    },
    candidateComparison: null,
    visibleComparison: null,
    summary: {
      anyMismatch: null,
      accepted: null,
      packedVisibleCount: null,
      validCount: null,
      rejectedCount: null
    }
  };
}

function buildShadowRuntimeFallback(options, shadowCompare = null, error = null) {
  return buildGpuCandidateRuntimeFallbackSummary({
    runtimeConfig: options.runtimeConfig,
    shadowCompare,
    error
  });
}

export function runGpuCandidateShadowCompare({
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
  renderResult = null,
  visibleSourceItems = null,
  options = {},
  metadata = {}
} = {}) {
  const shadowOptions = buildGpuCandidateShadowOptionsFromQuery({}, options);
  if (!isGpuCandidateShadowCompareEnabled(shadowOptions) && !options.force) {
    return {
      schemaVersion: 'step96-gpu-candidate-shadow-compare-v1',
      timestamp: new Date().toISOString(),
      status: 'skipped',
      runtime: shadowOptions.runtime,
      compareOnly: true,
      gpuCandidateUsedForDisplay: false,
      displayCandidateSource: 'cpu-reference',
      runtimeSummary: shadowOptions.runtimeSummary,
      fallback: buildShadowRuntimeFallback(shadowOptions),
      readbackWarning: 'GPU candidate shadow compare may use synchronous debug readback; keep it out of normal rendering until readback/fence-sync is designed.',
      options: shadowOptions,
      metadata,
      candidateComparison: null,
      visibleComparison: null,
      summary: {
        anyMismatch: null,
        accepted: null,
        packedVisibleCount: null,
        validCount: null,
        rejectedCount: null
      }
    };
  }

  try {
    const dryRun = captureGpuCandidateDryRunVisibleComparison({
      gl,
      raw,
      camera,
      screenSpaceCamera,
      canvasWidth,
      canvasHeight,
      camPos,
      tileGrid,
      buildConfig,
      candidateArgs,
      subsetCount: shadowOptions.subsetCount,
      subsetMode: shadowOptions.subsetMode,
      startIndex: shadowOptions.startIndex,
      filterMode: shadowOptions.filterMode,
      maxMismatches: shadowOptions.maxMismatches,
      epsilon: shadowOptions.epsilon,
      visibleSourceItems: Array.isArray(visibleSourceItems)
        ? visibleSourceItems
        : (Array.isArray(renderResult?.visible) ? renderResult.visible : []),
      metadata: {
        ...metadata,
        comparisonMode: 'gpu-candidate-shadow-compare',
        shadowCompareRuntime: shadowOptions.runtime,
        runtimeSummary: shadowOptions.runtimeSummary,
        displayCandidateSource: 'cpu-reference',
        gpuCandidateUsedForDisplay: false
      }
    });
    const gpuStats = dryRun.gpuDryRunSummary ?? null;
    const filterSummary = dryRun.candidateComparison?.metadata?.filterSummary ??
      dryRun.candidateComparison?.metadata?.gpuCandidateSummary?.filterSummary ??
      null;
    const fallback = buildShadowRuntimeFallback(shadowOptions, {
      status: 'ok',
      candidateComparison: dryRun.candidateComparison,
      visibleComparison: dryRun.visibleComparison,
      summary: { anyMismatch: !!dryRun.anyMismatch },
      shadowCompare: dryRun
    });
    return {
      schemaVersion: 'step96-gpu-candidate-shadow-compare-v1',
      timestamp: new Date().toISOString(),
      status: 'ok',
      runtime: shadowOptions.runtime,
      compareOnly: true,
      gpuCandidateUsedForDisplay: false,
      displayCandidateSource: 'cpu-reference',
      runtimeSummary: shadowOptions.runtimeSummary,
      fallback,
      readbackWarning: 'GPU candidate shadow compare may use synchronous debug readback; keep it out of normal rendering until readback/fence-sync is designed.',
      options: shadowOptions,
      metadata,
      shadowCompare: dryRun,
      candidateComparison: dryRun.candidateComparison,
      visibleComparison: dryRun.visibleComparison,
      referenceDryRunSummary: dryRun.referenceDryRunSummary ?? null,
      gpuDryRunSummary: gpuStats,
      summary: {
        anyMismatch: !!dryRun.anyMismatch,
        accepted: Number.isFinite(gpuStats?.accepted) ? gpuStats.accepted : null,
        packedVisibleCount: Number.isFinite(gpuStats?.packedVisibleCount) ? gpuStats.packedVisibleCount : null,
        candidateCount: Number.isFinite(gpuStats?.candidateCount) ? gpuStats.candidateCount : null,
        filterMode: shadowOptions.filterMode,
        validCount: Number.isFinite(filterSummary?.validCount) ? filterSummary.validCount : null,
        rejectedCount: Number.isFinite(filterSummary?.rejectedCount) ? filterSummary.rejectedCount : null,
        subsetMode: dryRun.metadata?.subsetMode ?? shadowOptions.subsetMode,
        selectedCandidateCount: Number.isFinite(dryRun.metadata?.selectedCandidateCount)
          ? dryRun.metadata.selectedCandidateCount
          : null
      }
    };
  } catch (error) {
    return buildShadowFailureResult({
      options: shadowOptions,
      metadata,
      error
    });
  }
}
