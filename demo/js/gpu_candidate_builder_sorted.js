import {
  buildTemporalSortedIndex,
  estimateTemporalWindow,
  getTemporalCandidateRange,
  buildTemporalCandidateIndices,
  summarizeTemporalRange
} from './gpu_temporal_index_utils.js';

export function buildSortedCandidateInfo({
  raw,
  timestamp,
  stride,
  sigmaScale,
  temporalSigmaThreshold,
  useTemporalIndexCache,
  temporalWindowMode,
  fixedWindowRadius
}) {
  const total = raw ? raw.N : 0;

  if (!raw || !raw.t || !raw.scale_t) {
    return {
      candidateIndices: new Uint32Array(0),
      candidateMode: 'sorted-no-temporal-data',
      temporalWindow: {
        maxSigmaT: Infinity,
        medianSigmaT: Infinity,
        meanSigmaT: Infinity,
        p90SigmaT: Infinity,
        windowRadius: Infinity,
        mode: 'sorted',
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

  const temporalIndexResult = buildTemporalSortedIndex(raw, {
    useCache: useTemporalIndexCache
  });
  const temporalIndex = temporalIndexResult.indexData;

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

  const rangeInfo = getTemporalCandidateRange(
    temporalIndex,
    timestamp,
    temporalWindow.windowRadius
  );

  const candidateIndices = buildTemporalCandidateIndices(
    temporalIndex,
    rangeInfo,
    stride
  );

  const rangeSummary = summarizeTemporalRange(
    temporalIndex,
    rangeInfo,
    candidateIndices
  );

  return {
    candidateIndices,
    candidateMode: 'sorted',
    temporalWindow,
    rangeSummary,
    temporalIndexDebug: {
      enabled: true,
      cacheEnabled: !!useTemporalIndexCache,
      cacheHit: temporalIndexResult.cacheHit,
      builtThisFrame: temporalIndexResult.builtThisFrame,
      totalCount: temporalIndex ? temporalIndex.count : 0,
      tMin: temporalIndex ? temporalIndex.tMin : NaN,
      tMax: temporalIndex ? temporalIndex.tMax : NaN
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
