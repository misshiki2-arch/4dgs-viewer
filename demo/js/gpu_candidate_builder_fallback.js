function buildFallbackStrideIndices(raw, stride) {
  if (!raw || !raw.N) return new Uint32Array(0);

  const s = Math.max(1, stride | 0);
  const count = Math.ceil(raw.N / s);
  const out = new Uint32Array(count);

  let w = 0;
  for (let i = 0; i < raw.N; i += s) {
    out[w++] = i;
  }
  return out;
}

export function buildFallbackCandidateInfo({
  raw,
  stride,
  reason = 'off'
}) {
  const total = raw ? raw.N : 0;
  const candidateIndices = buildFallbackStrideIndices(raw, stride);

  return {
    candidateIndices,
    candidateMode: reason,
    temporalWindow: {
      maxSigmaT: Infinity,
      medianSigmaT: Infinity,
      meanSigmaT: Infinity,
      p90SigmaT: Infinity,
      windowRadius: Infinity,
      mode: reason,
      cacheHit: false,
      builtThisFrame: false
    },
    rangeSummary: {
      totalCount: total,
      rangeCount: total,
      candidateCount: candidateIndices.length,
      rangeFraction: total > 0 ? 1 : 0,
      candidateFraction: total > 0 ? (candidateIndices.length / total) : 0
    },
    temporalIndexDebug: {
      enabled: false,
      cacheEnabled: false,
      cacheHit: false,
      builtThisFrame: false,
      totalCount: total,
      tMin: NaN,
      tMax: NaN
    },
    temporalBucketDebug: {
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
    }
  };
}
