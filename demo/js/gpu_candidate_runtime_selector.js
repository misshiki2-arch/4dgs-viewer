const RUNTIME_VALUES = new Set(['cpu-reference', 'shadow-compare', 'limited-draw']);
const FALLBACK_VALUES = new Set(['cpu-on-error', 'cpu-always', 'none']);
const SUBSET_MODE_VALUES = new Set(['firstN', 'visibleSrcIndices', 'fromVisible', 'visibleReachable']);
const FILTER_MODE_VALUES = new Set(['all-valid', 'evenIndex']);

const DEFAULT_RUNTIME = 'cpu-reference';
const DEFAULT_FALLBACK = 'cpu-on-error';
const DEFAULT_SUBSET_MODE = 'visibleSrcIndices';
const DEFAULT_SUBSET_COUNT = 1024;
const DEFAULT_FILTER_MODE = 'all-valid';

function normalizeRuntime(value) {
  return RUNTIME_VALUES.has(value) ? value : DEFAULT_RUNTIME;
}

function normalizeFallback(value) {
  return FALLBACK_VALUES.has(value) ? value : DEFAULT_FALLBACK;
}

function normalizeSubsetMode(value) {
  return SUBSET_MODE_VALUES.has(value) ? value : DEFAULT_SUBSET_MODE;
}

function normalizeFilterMode(value) {
  return FILTER_MODE_VALUES.has(value) ? value : DEFAULT_FILTER_MODE;
}

function toFiniteInteger(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(0, n | 0) : fallback;
}

function toBoolean(value, fallback) {
  return typeof value === 'boolean' ? value : fallback;
}

export function buildGpuCandidateRuntimeConfig(queryState = {}, overrides = {}) {
  const requestedRuntime = normalizeRuntime(
    overrides.runtime ?? overrides.gpuCandidateRuntime ?? queryState.gpuCandidateRuntime
  );
  const fallbackMode = normalizeFallback(
    overrides.fallback ?? overrides.gpuCandidateFallback ?? queryState.gpuCandidateFallback
  );
  const requireCompare = toBoolean(
    overrides.requireCompare ?? overrides.gpuCandidateRequireCompare ?? queryState.gpuCandidateRequireCompare,
    requestedRuntime === 'limited-draw'
  );
  const requireShadowOk = toBoolean(
    overrides.requireShadowOk ?? overrides.gpuCandidateRequireShadowOk ?? queryState.gpuCandidateRequireShadowOk,
    requestedRuntime === 'limited-draw'
  );
  const allowReadbackInDraw = toBoolean(
    overrides.allowReadbackInDraw ?? overrides.gpuCandidateAllowReadbackInDraw ?? queryState.gpuCandidateAllowReadbackInDraw,
    false
  );
  const debugReadback = toBoolean(
    overrides.debugReadback ?? overrides.gpuCandidateDebugReadback ?? queryState.gpuCandidateDebugReadback,
    true
  );
  const compareRequested = toBoolean(
    overrides.compare ?? overrides.gpuCandidateCompare ?? queryState.gpuCandidateCompare,
    requestedRuntime === 'shadow-compare' || requestedRuntime === 'limited-draw'
  );
  const subsetMode = normalizeSubsetMode(
    overrides.subsetMode ?? overrides.gpuCandidateSubsetMode ?? queryState.gpuCandidateSubsetMode
  );
  const subsetCount = toFiniteInteger(
    overrides.subsetCount ?? overrides.gpuCandidateSubsetCount ?? queryState.gpuCandidateSubsetCount,
    DEFAULT_SUBSET_COUNT
  );
  const filterMode = normalizeFilterMode(
    overrides.filterMode ?? overrides.gpuCandidateFilterMode ?? queryState.gpuCandidateFilterMode
  );
  const startIndex = toFiniteInteger(
    overrides.startIndex ?? overrides.gpuCandidateStartIndex ?? queryState.gpuCandidateStartIndex,
    0
  );

  const limitedDrawRequested = requestedRuntime === 'limited-draw';
  const shadowCompareRequested = requestedRuntime === 'shadow-compare';
  const shadowCompareEnabled = compareRequested && (shadowCompareRequested || limitedDrawRequested);
  const limitedDrawImplemented = false;
  const effectiveDisplayRuntime = limitedDrawRequested && limitedDrawImplemented
    ? 'limited-draw'
    : 'cpu-reference';

  return {
    schemaVersion: 'step97-gpu-candidate-runtime-config-v1',
    requestedRuntime,
    effectiveDisplayRuntime,
    displayCandidateSource: effectiveDisplayRuntime === 'limited-draw'
      ? 'gpu-candidate'
      : 'cpu-reference',
    gpuCandidateUsedForDisplay: effectiveDisplayRuntime === 'limited-draw',
    compareRequested,
    shadowCompareEnabled,
    limitedDrawRequested,
    limitedDrawImplemented,
    fallbackMode,
    requireCompare,
    requireShadowOk,
    allowReadbackInDraw,
    debugReadback,
    subsetMode,
    subsetCount,
    startIndex,
    filterMode,
    readbackPolicy: {
      allowReadbackInDraw,
      debugReadback,
      drawReadbackAllowed: allowReadbackInDraw,
      shadowReadbackAllowed: debugReadback,
      note: 'Synchronous GPU candidate readback is allowed only for debug/shadow unless allowReadbackInDraw is explicitly enabled.'
    }
  };
}

export function buildGpuCandidateRuntimeSummary(runtimeConfig = {}) {
  return {
    schemaVersion: 'step97-gpu-candidate-runtime-summary-v1',
    requestedRuntime: runtimeConfig.requestedRuntime ?? DEFAULT_RUNTIME,
    effectiveDisplayRuntime: runtimeConfig.effectiveDisplayRuntime ?? 'cpu-reference',
    displayCandidateSource: runtimeConfig.displayCandidateSource ?? 'cpu-reference',
    gpuCandidateUsedForDisplay: !!runtimeConfig.gpuCandidateUsedForDisplay,
    compareRequested: !!runtimeConfig.compareRequested,
    shadowCompareEnabled: !!runtimeConfig.shadowCompareEnabled,
    limitedDrawRequested: !!runtimeConfig.limitedDrawRequested,
    limitedDrawImplemented: !!runtimeConfig.limitedDrawImplemented,
    fallbackMode: runtimeConfig.fallbackMode ?? DEFAULT_FALLBACK,
    requireCompare: !!runtimeConfig.requireCompare,
    requireShadowOk: !!runtimeConfig.requireShadowOk,
    allowReadbackInDraw: !!runtimeConfig.allowReadbackInDraw,
    debugReadback: runtimeConfig.debugReadback !== false,
    subsetMode: runtimeConfig.subsetMode ?? DEFAULT_SUBSET_MODE,
    subsetCount: Number.isFinite(runtimeConfig.subsetCount) ? runtimeConfig.subsetCount : DEFAULT_SUBSET_COUNT,
    startIndex: Number.isFinite(runtimeConfig.startIndex) ? runtimeConfig.startIndex : 0,
    filterMode: runtimeConfig.filterMode ?? DEFAULT_FILTER_MODE,
    readbackPolicy: runtimeConfig.readbackPolicy ?? null
  };
}
