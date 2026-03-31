import {
  buildTemporalBucketIndex,
  getTemporalBucketCandidateRange,
  buildTemporalBucketCandidateIndices,
  summarizeTemporalBucketRange
} from './gpu_temporal_bucket_utils.js';
import { estimateTemporalWindow } from './gpu_temporal_index_utils.js';

function filterCandidateIndicesByTemporalWindow(raw, candidateIndices, timestamp, windowRadius) {
  if (!raw || !raw.t || !candidateIndices || candidateIndices.length === 0) {
    return new Uint32Array(0);
  }
  if (!Number.isFinite(windowRadius)) {
    return candidateIndices;
  }

  const out = new Uint32Array(candidateIndices.length);
  let w = 0;
  for (let k = 0; k < candidateIndices.length; k++) {
    const i = candidateIndices[k];
    const t = raw.t[i];
    if (!Number.isFinite(t)) continue;
    if (Math.abs(timestamp - t) <= windowRadius) {
      out[w++] = i;
    }
  }
  return out.slice(0, w);
}

export function buildHybridCandidateInfo({
  raw,
  timestamp,
  stride,
  sigmaScale,
  temporalSigmaThreshold,
  useTemporalIndexCache,
  temporalWindowMode,
  fixedWindowRadius,
  useTemporalBucketCache,
  temporalBucketWidth,
  temporalBucketRadius
}) {
  const total = raw ? raw.N : 0;

  if (!raw || !raw.t || !raw.scale_t) {
    return {
      candidateIndices: new Uint32Array(0),
      candidateMode: 'hybrid-no-temporal-data',
      temporalWindow: {
        maxSigmaT: Infinity,
        medianSigmaT: Infinity,
        meanSigmaT: Infinity,
        p90SigmaT: Infinity,
        windowRadius: Infinity,
        mode: 'hybrid',
        cacheHit: false,
        builtThisFrame: false
      },
      rangeSummary: {
        totalCount: total,
        rangeCount: 0,
        candidateCount: 0,
        rangeFraction: 0,
        candidateFraction: 0
      },
      temporalIndexDebug: {
        enabled: true,
        cacheEnabled: !!useTemporalIndexCache,
        cacheHit: false,
        builtThisFrame: false,
        totalCount: total,
        tMin: NaN,
        tMax: NaN
      },
      temporalBucketDebug: {
        enabled: true,
        cacheEnabled: !!useTemporalBucketCache,
        cacheHit: false,
        builtThisFrame: false,
        bucketWidth: Number.isFinite(Number(temporalBucketWidth)) ? Number(temporalBucketWidth) : NaN,
        bucketRadius: Number.isFinite(Number(temporalBucketRadius)) ? Math.max(0, parseInt(temporalBucketRadius, 10) || 0) : 0,
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

  const bucketResult = buildTemporalBucketIndex(raw, {
    bucketWidth: temporalBucketWidth,
    useCache: useTemporalBucketCache
  });

  const bucketData = bucketResult.bucketData;
  const bucketRange = getTemporalBucketCandidateRange(
    bucketData,
    timestamp,
    temporalBucketRadius
  );

  const bucketCandidateIndices = buildTemporalBucketCandidateIndices(
    bucketData,
    bucketRange,
    stride
  );

  const bucketSummary = summarizeTemporalBucketRange(
    bucketData,
    bucketRange,
    bucketCandidateIndices
  );

  const temporalWindow = estimateTemporalWindow(
    raw,
    sigmaScale,
    temporalSigmaThreshold,
    {
      useCache: useTemporalIndexCache,
      mode: temporalWindowMode,
      fixedWindowRadius
    }
  );

  const filteredCandidateIndices = filterCandidateIndicesByTemporalWindow(
    raw,
    bucketCandidateIndices,
    timestamp,
    temporalWindow.windowRadius
  );

  const filteredCount = filteredCandidateIndices.length;
  const filteredFraction = total > 0 ? filteredCount / total : 0;

  return {
    candidateIndices: filteredCandidateIndices,
    candidateMode: 'hybrid',
    temporalWindow,
    rangeSummary: {
      totalCount: total,
      rangeCount: bucketSummary.bucketSourceCount,
      candidateCount: filteredCount,
      rangeFraction: bucketSummary.bucketSourceFraction,
      candidateFraction: filteredFraction
    },
    temporalIndexDebug: {
      enabled: true,
      cacheEnabled: !!useTemporalIndexCache,
      cacheHit: !!temporalWindow.cacheHit,
      builtThisFrame: !!temporalWindow.builtThisFrame,
      totalCount: total,
      tMin: bucketData ? bucketData.tMin : NaN,
      tMax: bucketData ? bucketData.tMax : NaN
    },
    temporalBucketDebug: {
      enabled: true,
      cacheEnabled: !!useTemporalBucketCache,
      cacheHit: bucketResult.cacheHit,
      builtThisFrame: bucketResult.builtThisFrame,
      bucketWidth: bucketData ? bucketData.bucketWidth : NaN,
      bucketRadius: Number.isFinite(Number(temporalBucketRadius)) ? Math.max(0, parseInt(temporalBucketRadius, 10) || 0) : 0,
      bucketCount: bucketData ? bucketData.bucketCount : 0,
      bucketStart: bucketRange.bucketStart,
      bucketEnd: bucketRange.bucketEnd,
      bucketSourceCount: bucketSummary.bucketSourceCount,
      candidateCount: bucketSummary.candidateCount,
      bucketSourceFraction: bucketSummary.bucketSourceFraction,
      candidateFraction: bucketSummary.candidateFraction,
      postWindowCandidateCount: filteredCount,
      postWindowCandidateFraction: filteredFraction,
      tMin: bucketData ? bucketData.tMin : NaN,
      tMax: bucketData ? bucketData.tMax : NaN
    }
  };
}
