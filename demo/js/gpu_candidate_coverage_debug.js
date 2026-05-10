function toUint32Array(value) {
  if (value instanceof Uint32Array) return value;
  if (Array.isArray(value)) return Uint32Array.from(value);
  return new Uint32Array(0);
}

function toFiniteInteger(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(0, n | 0) : fallback;
}

function buildIndexSet(candidateInfo) {
  const indices = toUint32Array(candidateInfo?.candidateIndices);
  const set = new Set();
  for (let i = 0; i < indices.length; i++) {
    set.add(indices[i] >>> 0);
  }
  return { indices, set };
}

function getVisibleSrcIndex(item) {
  const value = Number(item?.srcIndex);
  return Number.isFinite(value) && value >= 0 ? (value >>> 0) : null;
}

function collectVisibleCoverage(visibleItems, candidateSet, maxMissingSamples) {
  const visible = Array.isArray(visibleItems) ? visibleItems : [];
  const missing = [];
  let hitCount = 0;
  let missCount = 0;
  for (const item of visible) {
    const srcIndex = getVisibleSrcIndex(item);
    if (srcIndex === null) continue;
    if (candidateSet.has(srcIndex)) {
      hitCount++;
    } else {
      missCount++;
      if (missing.length < maxMissingSamples) missing.push(srcIndex);
    }
  }
  const count = hitCount + missCount;
  return {
    cpuVisibleCount: count,
    visibleHitCount: hitCount,
    visibleMissCount: missCount,
    visibleCoverageRatio: count > 0 ? hitCount / count : 1,
    missingVisibleSrcIndices: missing
  };
}

function collectPackedCoverage(packedScreenSpace, candidateSet, maxMissingSamples) {
  const sourceIndices = toUint32Array(packedScreenSpace?.sourceIndices);
  if (sourceIndices.length <= 0) {
    return {
      packedSourceIndicesPresent: false,
      packedVisibleCount: Number.isFinite(packedScreenSpace?.packedCount)
        ? packedScreenSpace.packedCount
        : 0,
      packedHitCount: null,
      packedMissCount: null,
      packedCoverageRatio: null,
      missingPackedSrcIndices: []
    };
  }
  const missing = [];
  let hitCount = 0;
  let missCount = 0;
  for (let i = 0; i < sourceIndices.length; i++) {
    const srcIndex = sourceIndices[i] >>> 0;
    if (candidateSet.has(srcIndex)) {
      hitCount++;
    } else {
      missCount++;
      if (missing.length < maxMissingSamples) missing.push(srcIndex);
    }
  }
  const count = hitCount + missCount;
  return {
    packedSourceIndicesPresent: true,
    packedVisibleCount: count,
    packedHitCount: hitCount,
    packedMissCount: missCount,
    packedCoverageRatio: count > 0 ? hitCount / count : 1,
    missingPackedSrcIndices: missing
  };
}

export function buildGpuCandidateCoverageSummary({
  candidateInfo = null,
  candidateSourceSummary = null,
  sourceConfig = null,
  visibleItems = null,
  packedScreenSpace = null,
  maxMissingSamples = 32,
  metadata = {}
} = {}) {
  const sampleLimit = toFiniteInteger(maxMissingSamples, 32);
  const { indices, set } = buildIndexSet(candidateInfo);
  const visibleCoverage = collectVisibleCoverage(visibleItems, set, sampleLimit);
  const packedCoverage = collectPackedCoverage(packedScreenSpace, set, sampleLimit);
  const rangeStart = sourceConfig?.rangeStart ??
    candidateSourceSummary?.rangeStart ??
    candidateInfo?.rangeSummary?.rangeStart ??
    null;
  const rangeCount = sourceConfig?.rangeCount ??
    candidateSourceSummary?.rangeCount ??
    candidateInfo?.rangeSummary?.rangeCount ??
    null;
  return {
    schemaVersion: 'step105-gpu-candidate-coverage-summary-v1',
    status: 'ok',
    reason: 'coverage-compare-only',
    sourceMode: sourceConfig?.sourceMode ?? candidateSourceSummary?.sourceMode ?? 'unknown',
    candidateRangeStart: rangeStart,
    candidateRangeCount: rangeCount,
    gpuCandidateCount: indices.length,
    candidateSourceSummary: candidateSourceSummary ?? null,
    sourceConfig: sourceConfig ?? null,
    acceptedCount: Number.isFinite(visibleCoverage.visibleHitCount)
      ? visibleCoverage.visibleHitCount
      : null,
    ...visibleCoverage,
    ...packedCoverage,
    maxMissingSamples: sampleLimit,
    compareOnly: true,
    displayCandidateSource: 'cpu-reference',
    gpuCandidateUsedForDisplay: false,
    limitedDrawUsedForCandidateSource: false,
    metadata
  };
}
