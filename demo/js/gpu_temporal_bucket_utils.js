const temporalBucketCache = new WeakMap();

function clampInt(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function normalizeBucketWidth(bucketWidth) {
  const w = Number(bucketWidth);
  return Number.isFinite(w) && w > 0 ? w : 0.1;
}

function computeTemporalBucketIndex(raw, bucketWidth = 0.1) {
  if (!raw || !raw.t || !raw.N) {
    return {
      bucketWidth,
      bucketCount: 0,
      tMin: NaN,
      tMax: NaN,
      bucketMin: 0,
      bucketMax: -1,
      bucketOffsets: new Uint32Array(1),
      bucketIndices: new Uint32Array(0)
    };
  }

  const N = raw.N;
  let tMin = Infinity;
  let tMax = -Infinity;
  for (let i = 0; i < N; i++) {
    const t = raw.t[i];
    if (!Number.isFinite(t)) continue;
    if (t < tMin) tMin = t;
    if (t > tMax) tMax = t;
  }

  if (!Number.isFinite(tMin) || !Number.isFinite(tMax)) {
    return {
      bucketWidth,
      bucketCount: 0,
      tMin: NaN,
      tMax: NaN,
      bucketMin: 0,
      bucketMax: -1,
      bucketOffsets: new Uint32Array(1),
      bucketIndices: new Uint32Array(0)
    };
  }

  const bucketMin = Math.floor(tMin / bucketWidth);
  const bucketMax = Math.floor(tMax / bucketWidth);
  const bucketCount = Math.max(0, bucketMax - bucketMin + 1);

  const counts = new Uint32Array(bucketCount);
  for (let i = 0; i < N; i++) {
    const t = raw.t[i];
    if (!Number.isFinite(t)) continue;
    const b = Math.floor(t / bucketWidth) - bucketMin;
    if (b >= 0 && b < bucketCount) {
      counts[b]++;
    }
  }

  const bucketOffsets = new Uint32Array(bucketCount + 1);
  for (let b = 0; b < bucketCount; b++) {
    bucketOffsets[b + 1] = bucketOffsets[b] + counts[b];
  }

  const bucketIndices = new Uint32Array(bucketOffsets[bucketCount]);
  const writeHeads = new Uint32Array(bucketOffsets);

  for (let i = 0; i < N; i++) {
    const t = raw.t[i];
    if (!Number.isFinite(t)) continue;
    const b = Math.floor(t / bucketWidth) - bucketMin;
    if (b >= 0 && b < bucketCount) {
      const w = writeHeads[b]++;
      bucketIndices[w] = i;
    }
  }

  return {
    bucketWidth,
    bucketCount,
    tMin,
    tMax,
    bucketMin,
    bucketMax,
    bucketOffsets,
    bucketIndices
  };
}

function makeBucketCacheKey(bucketWidth) {
  return `w:${bucketWidth.toFixed(6)}`;
}

export function buildTemporalBucketIndex(raw, options = {}) {
  const bucketWidth = normalizeBucketWidth(options.bucketWidth ?? 0.1);
  const useCache = options.useCache !== false;

  if (!raw) {
    return {
      bucketData: computeTemporalBucketIndex(raw, bucketWidth),
      cacheHit: false,
      builtThisFrame: false
    };
  }

  let cacheMap = temporalBucketCache.get(raw);
  if (!cacheMap) {
    cacheMap = new Map();
    temporalBucketCache.set(raw, cacheMap);
  }

  const key = makeBucketCacheKey(bucketWidth);

  if (useCache && cacheMap.has(key)) {
    return {
      bucketData: cacheMap.get(key),
      cacheHit: true,
      builtThisFrame: false
    };
  }

  const bucketData = computeTemporalBucketIndex(raw, bucketWidth);
  if (useCache) {
    cacheMap.set(key, bucketData);
  }

  return {
    bucketData,
    cacheHit: false,
    builtThisFrame: true
  };
}

export function getTemporalBucketCandidateRange(bucketData, timestamp, bucketRadius = 0) {
  if (!bucketData || bucketData.bucketCount <= 0) {
    return {
      bucketStart: 0,
      bucketEnd: -1,
      bucketCount: 0
    };
  }

  const centerBucket = Math.floor(timestamp / bucketData.bucketWidth);
  const localCenter = centerBucket - bucketData.bucketMin;

  const start = clampInt(localCenter - bucketRadius, 0, bucketData.bucketCount - 1);
  const end = clampInt(localCenter + bucketRadius, 0, bucketData.bucketCount - 1);

  return {
    bucketStart: start,
    bucketEnd: end,
    bucketCount: end >= start ? (end - start + 1) : 0
  };
}

export function buildTemporalBucketCandidateIndices(bucketData, bucketRange, stride = 1) {
  if (!bucketData || !bucketRange || bucketRange.bucketCount <= 0) {
    return new Uint32Array(0);
  }

  const s = Math.max(1, stride | 0);

  let total = 0;
  for (let b = bucketRange.bucketStart; b <= bucketRange.bucketEnd; b++) {
    const start = bucketData.bucketOffsets[b];
    const end = bucketData.bucketOffsets[b + 1];
    total += Math.ceil((end - start) / s);
  }

  const out = new Uint32Array(total);
  let w = 0;
  for (let b = bucketRange.bucketStart; b <= bucketRange.bucketEnd; b++) {
    const start = bucketData.bucketOffsets[b];
    const end = bucketData.bucketOffsets[b + 1];
    for (let k = start; k < end; k += s) {
      out[w++] = bucketData.bucketIndices[k];
    }
  }
  return out;
}

export function summarizeTemporalBucketRange(bucketData, bucketRange, candidateIndices) {
  const totalCount = bucketData ? bucketData.bucketIndices.length : 0;
  let bucketSourceCount = 0;

  if (bucketData && bucketRange && bucketRange.bucketCount > 0) {
    for (let b = bucketRange.bucketStart; b <= bucketRange.bucketEnd; b++) {
      bucketSourceCount += (bucketData.bucketOffsets[b + 1] - bucketData.bucketOffsets[b]);
    }
  }

  const candidateCount = candidateIndices ? candidateIndices.length : 0;

  return {
    totalCount,
    bucketSourceCount,
    candidateCount,
    bucketSourceFraction: totalCount > 0 ? bucketSourceCount / totalCount : 0,
    candidateFraction: totalCount > 0 ? candidateCount / totalCount : 0
  };
}

export function getTemporalBucketUiOptions(ui) {
  return {
    useTemporalBucket: !!(ui && ui.useTemporalBucketCheck ? ui.useTemporalBucketCheck.checked : false),
    useTemporalBucketCache: !!(ui && ui.useTemporalBucketCacheCheck ? ui.useTemporalBucketCacheCheck.checked : true),
    temporalBucketWidth: ui && ui.temporalBucketWidthInput ? Number(ui.temporalBucketWidthInput.value) : 0.1,
    temporalBucketRadius: ui && ui.temporalBucketRadiusInput ? Math.max(0, parseInt(ui.temporalBucketRadiusInput.value, 10) || 0) : 0
  };
}
