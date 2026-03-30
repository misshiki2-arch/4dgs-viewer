export function buildTemporalSortedIndex(raw) {
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

export function estimateTemporalWindow(raw, sigmaScale = 1.0, sigmaThreshold = 3.0) {
  if (!raw || !raw.scale_t || !raw.N) {
    return {
      maxSigmaT: Infinity,
      medianSigmaT: Infinity,
      windowRadius: Infinity
    };
  }

  const N = raw.N;
  const vals = new Float32Array(N);
  let maxSigmaT = 0;
  for (let i = 0; i < N; i++) {
    const s = raw.scale_t[i] * sigmaScale;
    vals[i] = Number.isFinite(s) && s > 0 ? s : 0;
    if (vals[i] > maxSigmaT) maxSigmaT = vals[i];
  }

  const copy = Array.from(vals);
  copy.sort((a, b) => a - b);
  const medianSigmaT = copy.length > 0 ? copy[(copy.length / 2) | 0] : Infinity;

  return {
    maxSigmaT,
    medianSigmaT,
    windowRadius: maxSigmaT * sigmaThreshold
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
