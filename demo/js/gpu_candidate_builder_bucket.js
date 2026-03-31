import {
  buildTemporalBucketIndex,
  getTemporalBucketCandidateRange,
  buildTemporalBucketCandidateIndices,
  summarizeTemporalBucketRange
} from './gpu_temporal_bucket_utils.js';

export function buildBucketCandidateInfo({
  raw,
  timestamp,
  stride,
  useTemporalBucketCache,
  temporalBucketWidth,
  temporalBucketRadius
}) {
  const total = raw ? raw.N : 0;

  if (!raw || !raw.t || !raw.N) {
    return {
      candidateIndices: new Uint32Array(0),
      candidateMode: 'bucket-no-temporal-data',
      temporalWindow: {
        maxSigmaT: Infinity,
        medianSigmaT: Infinity,
        meanSigmaT: Infinity,
        p90SigmaT: Infinity,
        windowRadius: Infinity,
        mode: 'bucket',
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
        enabled: false,
        cacheEnabled: false,
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

  const candidateIndices = buildTemporalBucketCandidateIndices(
    bucketData,
    bucketRange,
    stride
  );

  const bucketSummary = summarizeTemporalBucketRange(
    bucketData,
    bucketRange,
    candidateIndices
  );

  return {
    candidateIndices,
    candidateMode: 'bucket',
    temporalWindow: {
      maxSigmaT: Infinity,
      medianSigmaT: Infinity,
      meanSigmaT: Infinity,
      p90SigmaT: Infinity,
      windowRadius: Infinity,
      mode: 'bucket',
      cacheHit: false,
      builtThisFrame: false
    },
    rangeSummary: {
      totalCount: bucketSummary.totalCount,
      rangeCount: bucketSummary.bucketSourceCount,
      candidateCount: bucketSummary.candidateCount,
      rangeFraction: bucketSummary.bucketSourceFraction,
      candidateFraction: bucketSummary.candidateFraction
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
      postWindowCandidateCount: bucketSummary.candidateCount,
      postWindowCandidateFraction: bucketSummary.candidateFraction,
      tMin: bucketData ? bucketData.tMin : NaN,
      tMax: bucketData ? bucketData.tMax : NaN
    }
  };
}
