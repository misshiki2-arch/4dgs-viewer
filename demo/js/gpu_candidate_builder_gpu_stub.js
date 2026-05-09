function cloneCandidateIndices(indices) {
  if (indices instanceof Uint32Array) return new Uint32Array(indices);
  if (Array.isArray(indices)) return Uint32Array.from(indices);
  return new Uint32Array(0);
}

function clonePlainObject(value) {
  if (!value || typeof value !== 'object') return value;
  if (Array.isArray(value)) return value.map((item) => clonePlainObject(item));
  const out = {};
  for (const [key, item] of Object.entries(value)) {
    out[key] = clonePlainObject(item);
  }
  return out;
}

function normalizeSubsetIndices(indices) {
  if (!Array.isArray(indices)) return null;
  const out = [];
  const seen = new Set();
  for (const value of indices) {
    const n = Number(value);
    if (!Number.isFinite(n)) continue;
    const index = n | 0;
    if (index < 0 || seen.has(index)) continue;
    seen.add(index);
    out.push(index);
  }
  return Uint32Array.from(out);
}

function buildSubsetIndices(sourceIndices, {
  subsetMode = 'firstN',
  subsetCount = 1024,
  explicitIndices = null
} = {}) {
  const source = cloneCandidateIndices(sourceIndices);
  if (subsetMode === 'explicitIndices') {
    const explicit = normalizeSubsetIndices(explicitIndices);
    if (!explicit || explicit.length === 0) return new Uint32Array(0);
    const sourceSet = new Set(source);
    const out = [];
    for (const index of explicit) {
      if (sourceSet.has(index)) out.push(index);
    }
    return Uint32Array.from(out);
  }

  const count = Number.isFinite(Number(subsetCount))
    ? Math.max(0, Math.min(source.length, Number(subsetCount) | 0))
    : Math.min(source.length, 1024);
  return source.slice(0, count);
}

function buildSubsetRangeSummary(source, subsetIndices, total) {
  const base = clonePlainObject(source.rangeSummary) ?? {};
  const sourceRangeCount = Number.isFinite(base.rangeCount) ? base.rangeCount : subsetIndices.length;
  return {
    ...base,
    totalCount: Number.isFinite(base.totalCount) ? base.totalCount : total,
    rangeCount: sourceRangeCount,
    candidateCount: subsetIndices.length,
    candidateFraction: total > 0 ? subsetIndices.length / total : 0
  };
}

export function buildCandidateSubsetInfo({
  referenceCandidateInfo = null,
  raw = null,
  subsetMode = 'firstN',
  subsetCount = 1024,
  explicitIndices = null,
  candidateMode = 'cpu-subset'
} = {}) {
  const source = referenceCandidateInfo ?? {};
  const sourceIndices = cloneCandidateIndices(source.candidateIndices);
  const candidateIndices = buildSubsetIndices(sourceIndices, {
    subsetMode,
    subsetCount,
    explicitIndices
  });
  const total = raw ? raw.N : (source.rangeSummary?.totalCount ?? sourceIndices.length);
  return {
    candidateIndices,
    candidateMode,
    temporalWindow: clonePlainObject(source.temporalWindow),
    rangeSummary: buildSubsetRangeSummary(source, candidateIndices, total),
    temporalIndexDebug: clonePlainObject(source.temporalIndexDebug),
    temporalBucketDebug: clonePlainObject(source.temporalBucketDebug),
    candidateSubsetSummary: {
      enabled: true,
      contract: 'candidate-info-subset-adapter',
      subsetMode,
      subsetCount: candidateIndices.length,
      requestedSubsetCount: Number.isFinite(Number(subsetCount)) ? Number(subsetCount) | 0 : null,
      explicitIndexCount: Array.isArray(explicitIndices) ? explicitIndices.length : 0,
      sourceCandidateMode: source.candidateMode ?? 'unknown',
      sourceCandidateCount: sourceIndices.length
    }
  };
}

export function buildGpuStubCandidateInfo({
  referenceCandidateInfo = null,
  raw = null,
  reason = 'gpu-candidate-stub-cpu-adapter'
} = {}) {
  const source = referenceCandidateInfo ?? {};
  const candidateIndices = cloneCandidateIndices(source.candidateIndices);
  const total = raw ? raw.N : (source.rangeSummary?.totalCount ?? candidateIndices.length);

  return {
    candidateIndices,
    candidateMode: 'gpu-stub',
    temporalWindow: clonePlainObject(source.temporalWindow) ?? {
      maxSigmaT: Infinity,
      medianSigmaT: Infinity,
      meanSigmaT: Infinity,
      p90SigmaT: Infinity,
      windowRadius: Infinity,
      mode: 'gpu-stub',
      cacheHit: false,
      builtThisFrame: false
    },
    rangeSummary: clonePlainObject(source.rangeSummary) ?? {
      totalCount: total,
      rangeCount: candidateIndices.length,
      candidateCount: candidateIndices.length,
      rangeFraction: total > 0 ? candidateIndices.length / total : 0,
      candidateFraction: total > 0 ? candidateIndices.length / total : 0
    },
    temporalIndexDebug: clonePlainObject(source.temporalIndexDebug) ?? {
      enabled: false,
      cacheEnabled: false,
      cacheHit: false,
      builtThisFrame: false,
      totalCount: total,
      tMin: NaN,
      tMax: NaN
    },
    temporalBucketDebug: clonePlainObject(source.temporalBucketDebug) ?? {
      enabled: false,
      cacheEnabled: false,
      cacheHit: false,
      builtThisFrame: false,
      bucketWidth: NaN,
      bucketRadius: 0,
      bucketCount: 0,
      bucketStart: 0,
      bucketEnd: -1,
      bucketSourceCount: 0,
      candidateCount: 0,
      bucketSourceFraction: 0,
      candidateFraction: 0,
      postWindowCandidateCount: 0,
      postWindowCandidateFraction: 0,
      tMin: NaN,
      tMax: NaN
    },
    gpuCandidateStubSummary: {
      enabled: true,
      contract: 'cpu-candidate-info-compatible-adapter',
      sourceCandidateMode: source.candidateMode ?? 'unknown',
      candidateCount: candidateIndices.length,
      reason
    }
  };
}

export function buildGpuSubsetCandidateInfo({
  referenceSubsetCandidateInfo = null,
  raw = null,
  reason = 'gpu-candidate-subset-stub-cpu-adapter'
} = {}) {
  const subset = buildGpuStubCandidateInfo({
    referenceCandidateInfo: referenceSubsetCandidateInfo,
    raw,
    reason
  });
  return {
    ...subset,
    candidateMode: 'gpu-subset-stub',
    candidateSubsetSummary: clonePlainObject(referenceSubsetCandidateInfo?.candidateSubsetSummary) ?? null,
    gpuCandidateStubSummary: {
      ...subset.gpuCandidateStubSummary,
      contract: 'cpu-candidate-subset-compatible-adapter',
      sourceCandidateMode: referenceSubsetCandidateInfo?.candidateMode ?? 'unknown',
      reason
    }
  };
}
