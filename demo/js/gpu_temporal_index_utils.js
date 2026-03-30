const temporalSortedIndexCache = new WeakMap();
const temporalWindowStatsCache = new WeakMap();

function lowerBound(arr, x) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function upperBound(arr, x) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function computeTemporalSortedIndex(raw) {
  if (!raw || !raw.t || !raw.N) {
    return {
      sortedIndices: new Uint32Array(0),
      sortedT: new Float32Array(0),
      count: 0,
      tMin: NaN,
      tMax: NaN
    };
  }

  const N = raw.N;
  const pairs = new Array(N);
  for (let i = 0; i < N; i++) {
    pairs[i] = { i, t: raw.t[i] };
  }
  pairs.sort((a, b) => a.t - b.t);

  const sortedIndices = new Uint32Array(N);
  const sortedT = new Float32Array(N);
  for (let k = 0; k < N; k++) {
    sortedIndices[k] = pairs[k].i;
    sortedT[k] = pairs[k].t;
  }

  return {
    sortedIndices,
    sortedT,
    count: N,
    tMin: N > 0 ? sortedT[0] : NaN,
    tMax: N > 0 ? sortedT[N - 1] : NaN
  };
}

export function buildTemporalSortedIndex(raw, options = {}) {
  const useCache = options.useCache !== false;

  if (!raw) {
    return {
      indexData: computeTemporalSortedIndex(raw),
      cacheHit: false,
      builtThisFrame: false
    };
  }

  if (useCache) {
    const cached = temporalSortedIndexCache.get(raw);
    if (cached) {
      return {
        indexData: cached,
        cacheHit: true,
        builtThisFrame: false
      };
    }
  }

  const indexData = computeTemporalSortedIndex(raw);
  if (useCache) {
    temporalSortedIndexCache.set(raw, indexData);
  }

  return {
    indexData,
    cacheHit: false,
    builtThisFrame: true
  };
}

function computeTemporalWindowStats(raw, sigmaScale = 1.0) {
  if (!raw || !raw.scale_t || !raw.N) {
    return {
      maxSigmaT: Infinity,
      medianSigmaT: Infinity,
      meanSigmaT: Infinity,
      p90SigmaT: Infinity,
      sampleCount: 0
    };
  }

  const N = raw.N;
  const vals = new Float32Array(N);
  let maxSigmaT = 0;
  let sumSigmaT = 0;

  for (let i = 0; i < N; i++) {
    const s = raw.scale_t[i] * sigmaScale;
    const v = Number.isFinite(s) && s > 0 ? s : 0;
    vals[i] = v;
    if (v > maxSigmaT) maxSigmaT = v;
    sumSigmaT += v;
  }

  const copy = Array.from(vals);
  copy.sort((a, b) => a - b);

  const mid = copy.length > 0 ? ((copy.length / 2) | 0) : 0;
  const p90Idx = copy.length > 0 ? Math.min(copy.length - 1, Math.floor(copy.length * 0.9)) : 0;

  return {
    maxSigmaT,
    medianSigmaT: copy.length > 0 ? copy[mid] : Infinity,
    meanSigmaT: copy.length > 0 ? (sumSigmaT / copy.length) : Infinity,
    p90SigmaT: copy.length > 0 ? copy[p90Idx] : Infinity,
    sampleCount: N
  };
}

export function estimateTemporalWindow(raw, sigmaScale = 1.0, sigmaThreshold = 3.0, options = {}) {
  const useCache = options.useCache !== false;
  const mode = options.mode || 'max'; // max | median | mean | p90 | fixed
  const fixedWindowRadius = Number.isFinite(options.fixedWindowRadius) ? options.fixedWindowRadius : 0.5;

  if (!raw || !raw.scale_t || !raw.N) {
    return {
      maxSigmaT: Infinity,
      medianSigmaT: Infinity,
      meanSigmaT: Infinity,
      p90SigmaT: Infinity,
      windowRadius: Infinity,
      mode,
      cacheHit: false,
      builtThisFrame: false
    };
  }

  let stats = null;
  let cacheHit = false;
  let builtThisFrame = false;

  if (mode === 'fixed') {
    return {
      maxSigmaT: Infinity,
      medianSigmaT: Infinity,
      meanSigmaT: Infinity,
      p90SigmaT: Infinity,
      windowRadius: fixedWindowRadius,
      mode,
      cacheHit: false,
      builtThisFrame: false
    };
  }

  if (useCache) {
    stats = temporalWindowStatsCache.get(raw);
    if (stats) {
      cacheHit = true;
    }
  }

  if (!stats) {
    stats = computeTemporalWindowStats(raw, sigmaScale);
    builtThisFrame = true;
    if (useCache) {
      temporalWindowStatsCache.set(raw, stats);
    }
  }

  let sigmaBase = stats.maxSigmaT;
  if (mode === 'median') sigmaBase = stats.medianSigmaT;
  else if (mode === 'mean') sigmaBase = stats.meanSigmaT;
  else if (mode === 'p90') sigmaBase = stats.p90SigmaT;

  return {
    ...stats,
    windowRadius: sigmaBase * sigmaThreshold,
    mode,
    cacheHit,
    builtThisFrame
  };
}

export function getTemporalCandidateRange(indexData, timestamp, windowRadius) {
  if (!indexData || !indexData.sortedT || indexData.sortedT.length === 0) {
    return {
      start: 0,
      end: 0,
      count: 0
    };
  }

  if (!Number.isFinite(windowRadius)) {
    return {
      start: 0,
      end: indexData.sortedT.length,
      count: indexData.sortedT.length
    };
  }

  const t0 = timestamp - windowRadius;
  const t1 = timestamp + windowRadius;
  const start = lowerBound(indexData.sortedT, t0);
  const end = upperBound(indexData.sortedT, t1);

  return {
    start,
    end,
    count: Math.max(0, end - start)
  };
}

export function buildTemporalCandidateIndices(indexData, rangeInfo, stride = 1) {
  if (!indexData || !indexData.sortedIndices || !rangeInfo) {
    return new Uint32Array(0);
  }

  const s = Math.max(1, stride | 0);
  const count = rangeInfo.count <= 0 ? 0 : Math.ceil(rangeInfo.count / s);
  const out = new Uint32Array(count);

  let w = 0;
  for (let k = rangeInfo.start; k < rangeInfo.end; k += s) {
    out[w++] = indexData.sortedIndices[k];
  }
  return out;
}

export function summarizeTemporalRange(indexData, rangeInfo, candidateIndices) {
  const total = indexData ? indexData.count : 0;
  const rangeCount = rangeInfo ? rangeInfo.count : 0;
  const candidateCount = candidateIndices ? candidateIndices.length : 0;

  return {
    totalCount: total,
    rangeCount,
    candidateCount,
    rangeFraction: total > 0 ? rangeCount / total : 0,
    candidateFraction: total > 0 ? candidateCount / total : 0
  };
}

export function getTemporalIndexUiOptions(ui) {
  return {
    useTemporalIndex: !!(ui && ui.useTemporalIndexCheck ? ui.useTemporalIndexCheck.checked : true),
    useTemporalIndexCache: !!(ui && ui.useTemporalIndexCacheCheck ? ui.useTemporalIndexCacheCheck.checked : true),
    temporalWindowMode: ui && ui.temporalWindowModeSelect ? ui.temporalWindowModeSelect.value : 'max',
    fixedWindowRadius: ui && ui.fixedWindowRadiusInput ? Number(ui.fixedWindowRadiusInput.value) : 0.5
  };
}

export function clearTemporalIndexCaches(raw = null) {
  if (raw) {
    temporalSortedIndexCache.delete(raw);
    temporalWindowStatsCache.delete(raw);
    return;
  }
  // WeakMap cannot be fully cleared, so this only exists for API symmetry.
}
