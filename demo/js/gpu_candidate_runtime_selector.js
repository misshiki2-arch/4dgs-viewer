const RUNTIME_VALUES = new Set(['cpu-reference', 'shadow-compare', 'limited-draw']);
const FALLBACK_VALUES = new Set(['cpu-on-error', 'cpu-always', 'none']);
const SUBSET_MODE_VALUES = new Set(['firstN', 'visibleSrcIndices', 'fromVisible', 'visibleReachable']);
const FILTER_MODE_VALUES = new Set(['all-valid', 'evenIndex']);
const SOURCE_MODE_VALUES = new Set(['visibleSrcIndices', 'firstN', 'range', 'screenCoarse']);
const PROMOTE_POLICY_VALUES = new Set(['never', 'compare-ok', 'async-ready', 'validated-only']);
const READBACK_MODE_VALUES = new Set(['sync-debug', 'async-fence', 'none']);
const SCREEN_COARSE_DEPTH_MODE_VALUES = new Set(['positive', 'any']);

const DEFAULT_RUNTIME = 'cpu-reference';
const DEFAULT_FALLBACK = 'cpu-on-error';
const DEFAULT_SUBSET_MODE = 'visibleSrcIndices';
const DEFAULT_SUBSET_COUNT = 1024;
const DEFAULT_FILTER_MODE = 'all-valid';
const DEFAULT_SOURCE_MODE = 'visibleSrcIndices';
const DEFAULT_RANGE_START = 0;
const DEFAULT_RANGE_COUNT = 65536;
const DEFAULT_SCREEN_COARSE_MAX_COUNT = 65536;
const DEFAULT_SCREEN_COARSE_MIN_RADIUS_PX = 0.25;
const DEFAULT_SCREEN_COARSE_REQUIRE_IN_VIEWPORT = true;
const DEFAULT_SCREEN_COARSE_DEPTH_MODE = 'positive';
const DEFAULT_PROMOTE_POLICY = 'never';
const DEFAULT_READBACK_MODE = 'sync-debug';
const DEFAULT_COVERAGE_MAX_MISSES = 32;
const DEFAULT_VISIBLE_RECORD_MAX_COUNT = 65536;
const RAW_VISIBLE_RECORD_MODE_VALUES = new Set(['minimal']);

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

function normalizeSourceMode(value) {
  return SOURCE_MODE_VALUES.has(value) ? value : DEFAULT_SOURCE_MODE;
}

function normalizePromotePolicy(value) {
  return PROMOTE_POLICY_VALUES.has(value) ? value : DEFAULT_PROMOTE_POLICY;
}

function normalizeReadbackMode(value) {
  return READBACK_MODE_VALUES.has(value) ? value : DEFAULT_READBACK_MODE;
}

function normalizeScreenCoarseDepthMode(value) {
  return SCREEN_COARSE_DEPTH_MODE_VALUES.has(value) ? value : DEFAULT_SCREEN_COARSE_DEPTH_MODE;
}

function toFiniteInteger(value, fallback) {
  if (value == null || value === '') return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(0, n | 0) : fallback;
}

function toFiniteNumber(value, fallback) {
  if (value == null || value === '') return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
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
  const sourceMode = normalizeSourceMode(
    overrides.sourceMode ?? overrides.gpuCandidateSourceMode ?? queryState.gpuCandidateSourceMode
  );
  const rangeStart = toFiniteInteger(
    overrides.rangeStart ?? overrides.gpuCandidateRangeStart ?? queryState.gpuCandidateRangeStart,
    DEFAULT_RANGE_START
  );
  const rangeCount = toFiniteInteger(
    overrides.rangeCount ?? overrides.gpuCandidateRangeCount ?? queryState.gpuCandidateRangeCount,
    DEFAULT_RANGE_COUNT
  );
  const screenCoarseMaxCount = toFiniteInteger(
    overrides.screenCoarseMaxCount ??
      overrides.gpuCandidateScreenCoarseMaxCount ??
      queryState.gpuCandidateScreenCoarseMaxCount,
    DEFAULT_SCREEN_COARSE_MAX_COUNT
  );
  const screenCoarseMinRadiusPx = Math.max(0, toFiniteNumber(
    overrides.screenCoarseMinRadiusPx ??
      overrides.gpuCandidateScreenCoarseMinRadiusPx ??
      queryState.gpuCandidateScreenCoarseMinRadiusPx,
    DEFAULT_SCREEN_COARSE_MIN_RADIUS_PX
  ));
  const screenCoarseRequireInViewport = toBoolean(
    overrides.screenCoarseRequireInViewport ??
      overrides.gpuCandidateScreenCoarseRequireInViewport ??
      queryState.gpuCandidateScreenCoarseRequireInViewport,
    DEFAULT_SCREEN_COARSE_REQUIRE_IN_VIEWPORT
  );
  const screenCoarseDepthMode = normalizeScreenCoarseDepthMode(
    overrides.screenCoarseDepthMode ??
      overrides.gpuCandidateScreenCoarseDepthMode ??
      queryState.gpuCandidateScreenCoarseDepthMode
  );
  const promotePolicy = normalizePromotePolicy(
    overrides.promotePolicy ?? overrides.gpuCandidatePromotePolicy ?? queryState.gpuCandidatePromotePolicy
  );
  const readbackMode = normalizeReadbackMode(
    overrides.readbackMode ?? overrides.gpuCandidateReadbackMode ?? queryState.gpuCandidateReadbackMode
  );
  const coverageCompare = toBoolean(
    overrides.coverageCompare ?? overrides.gpuCandidateCoverageCompare ?? queryState.gpuCandidateCoverageCompare,
    false
  );
  const coverageMaxMisses = toFiniteInteger(
    overrides.coverageMaxMisses ?? overrides.gpuCandidateCoverageMaxMisses ?? queryState.gpuCandidateCoverageMaxMisses,
    DEFAULT_COVERAGE_MAX_MISSES
  );
  const visibleRecordDryRun = toBoolean(
    overrides.visibleRecordDryRun ?? overrides.gpuVisibleRecordDryRun ?? queryState.gpuVisibleRecordDryRun,
    false
  );
  const visibleRecordSource = normalizeSourceMode(
    overrides.visibleRecordSource ?? overrides.gpuVisibleRecordSource ?? queryState.gpuVisibleRecordSource ?? sourceMode
  );
  const visibleRecordReadback = normalizeReadbackMode(
    overrides.visibleRecordReadback ?? overrides.gpuVisibleRecordReadback ?? queryState.gpuVisibleRecordReadback ?? readbackMode
  );
  const visibleRecordMaxCount = toFiniteInteger(
    overrides.visibleRecordMaxCount ?? overrides.gpuVisibleRecordMaxCount ?? queryState.gpuVisibleRecordMaxCount,
    DEFAULT_VISIBLE_RECORD_MAX_COUNT
  );
  const visibleRecordCompare = toBoolean(
    overrides.visibleRecordCompare ?? overrides.gpuVisibleRecordCompare ?? queryState.gpuVisibleRecordCompare,
    true
  );
  const rawVisibleRecordDryRun = toBoolean(
    overrides.rawVisibleRecordDryRun ?? overrides.gpuRawVisibleRecordDryRun ?? queryState.gpuRawVisibleRecordDryRun,
    false
  );
  const rawVisibleRecordModeValue =
    overrides.rawVisibleRecordMode ?? overrides.gpuRawVisibleRecordMode ?? queryState.gpuRawVisibleRecordMode;
  const rawVisibleRecordMode = RAW_VISIBLE_RECORD_MODE_VALUES.has(rawVisibleRecordModeValue)
    ? rawVisibleRecordModeValue
    : 'minimal';
  const rawVisibleRecordReadback = normalizeReadbackMode(
    overrides.rawVisibleRecordReadback ?? overrides.gpuRawVisibleRecordReadback ?? queryState.gpuRawVisibleRecordReadback ?? readbackMode
  );
  const rawVisibleRecordFields =
    overrides.rawVisibleRecordFields ?? overrides.gpuRawVisibleRecordFields ?? queryState.gpuRawVisibleRecordFields ?? 'srcIndex,valid,px,py,depth,aabb';
  const rawAttributeTexture = toBoolean(
    overrides.rawAttributeTexture ?? overrides.gpuRawAttributeTexture ?? queryState.gpuRawAttributeTexture,
    rawVisibleRecordDryRun
  );

  const limitedDrawRequested = requestedRuntime === 'limited-draw';
  const shadowCompareRequested = requestedRuntime === 'shadow-compare';
  const shadowCompareEnabled = compareRequested && (shadowCompareRequested || limitedDrawRequested);
  const limitedDrawImplemented = true;
  const limitedDrawRequiresReadbackInDraw = limitedDrawRequested;
  const limitedDrawReadbackAllowed = limitedDrawRequiresReadbackInDraw && allowReadbackInDraw;
  const effectiveDisplayRuntime = limitedDrawRequested && limitedDrawImplemented && limitedDrawReadbackAllowed
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
    limitedDrawRequiresReadbackInDraw,
    limitedDrawReadbackAllowed,
    fallbackMode,
    requireCompare,
    requireShadowOk,
    allowReadbackInDraw,
    debugReadback,
    subsetMode,
    subsetCount,
    startIndex,
    filterMode,
    sourceMode,
    rangeStart,
    rangeCount,
    screenCoarseMaxCount,
    screenCoarseMinRadiusPx,
    screenCoarseRequireInViewport,
    screenCoarseDepthMode,
    promotePolicy,
    readbackMode,
    coverageCompare,
    coverageMaxMisses,
    visibleRecordDryRun,
    visibleRecordSource,
    visibleRecordReadback,
    visibleRecordMaxCount,
    visibleRecordCompare,
    rawVisibleRecordDryRun,
    rawVisibleRecordMode,
    rawVisibleRecordReadback,
    rawVisibleRecordFields,
    rawAttributeTexture,
    readbackPolicy: {
      allowReadbackInDraw,
      debugReadback,
      drawReadbackAllowed: allowReadbackInDraw,
      shadowReadbackAllowed: debugReadback,
      readbackMode,
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
    limitedDrawRequiresReadbackInDraw: !!runtimeConfig.limitedDrawRequiresReadbackInDraw,
    limitedDrawReadbackAllowed: !!runtimeConfig.limitedDrawReadbackAllowed,
    fallbackMode: runtimeConfig.fallbackMode ?? DEFAULT_FALLBACK,
    requireCompare: !!runtimeConfig.requireCompare,
    requireShadowOk: !!runtimeConfig.requireShadowOk,
    allowReadbackInDraw: !!runtimeConfig.allowReadbackInDraw,
    debugReadback: runtimeConfig.debugReadback !== false,
    subsetMode: runtimeConfig.subsetMode ?? DEFAULT_SUBSET_MODE,
    subsetCount: Number.isFinite(runtimeConfig.subsetCount) ? runtimeConfig.subsetCount : DEFAULT_SUBSET_COUNT,
    startIndex: Number.isFinite(runtimeConfig.startIndex) ? runtimeConfig.startIndex : 0,
    filterMode: runtimeConfig.filterMode ?? DEFAULT_FILTER_MODE,
    sourceMode: runtimeConfig.sourceMode ?? DEFAULT_SOURCE_MODE,
    rangeStart: Number.isFinite(runtimeConfig.rangeStart) ? runtimeConfig.rangeStart : DEFAULT_RANGE_START,
    rangeCount: Number.isFinite(runtimeConfig.rangeCount) ? runtimeConfig.rangeCount : DEFAULT_RANGE_COUNT,
    screenCoarseMaxCount: Number.isFinite(runtimeConfig.screenCoarseMaxCount)
      ? runtimeConfig.screenCoarseMaxCount
      : DEFAULT_SCREEN_COARSE_MAX_COUNT,
    screenCoarseMinRadiusPx: Number.isFinite(runtimeConfig.screenCoarseMinRadiusPx)
      ? runtimeConfig.screenCoarseMinRadiusPx
      : DEFAULT_SCREEN_COARSE_MIN_RADIUS_PX,
    screenCoarseRequireInViewport: typeof runtimeConfig.screenCoarseRequireInViewport === 'boolean'
      ? runtimeConfig.screenCoarseRequireInViewport
      : DEFAULT_SCREEN_COARSE_REQUIRE_IN_VIEWPORT,
    screenCoarseDepthMode: runtimeConfig.screenCoarseDepthMode ?? DEFAULT_SCREEN_COARSE_DEPTH_MODE,
    promotePolicy: runtimeConfig.promotePolicy ?? DEFAULT_PROMOTE_POLICY,
    readbackMode: runtimeConfig.readbackMode ?? DEFAULT_READBACK_MODE,
    coverageCompare: !!runtimeConfig.coverageCompare,
    coverageMaxMisses: Number.isFinite(runtimeConfig.coverageMaxMisses)
      ? runtimeConfig.coverageMaxMisses
      : DEFAULT_COVERAGE_MAX_MISSES,
    visibleRecordDryRun: !!runtimeConfig.visibleRecordDryRun,
    visibleRecordSource: runtimeConfig.visibleRecordSource ?? runtimeConfig.sourceMode ?? DEFAULT_SOURCE_MODE,
    visibleRecordReadback: runtimeConfig.visibleRecordReadback ?? runtimeConfig.readbackMode ?? DEFAULT_READBACK_MODE,
    visibleRecordMaxCount: Number.isFinite(runtimeConfig.visibleRecordMaxCount)
      ? runtimeConfig.visibleRecordMaxCount
      : DEFAULT_VISIBLE_RECORD_MAX_COUNT,
    visibleRecordCompare: runtimeConfig.visibleRecordCompare !== false,
    rawVisibleRecordDryRun: !!runtimeConfig.rawVisibleRecordDryRun,
    rawVisibleRecordMode: runtimeConfig.rawVisibleRecordMode ?? 'minimal',
    rawVisibleRecordReadback: runtimeConfig.rawVisibleRecordReadback ?? runtimeConfig.readbackMode ?? DEFAULT_READBACK_MODE,
    rawVisibleRecordFields: runtimeConfig.rawVisibleRecordFields ?? 'srcIndex,valid,px,py,depth,aabb',
    rawAttributeTexture: !!runtimeConfig.rawAttributeTexture,
    readbackPolicy: runtimeConfig.readbackPolicy ?? null
  };
}
