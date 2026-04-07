import { buildFallbackCandidateInfo } from './gpu_candidate_builder_fallback.js';
import { buildSortedCandidateInfo } from './gpu_candidate_builder_sorted.js';
import { buildBucketCandidateInfo } from './gpu_candidate_builder_bucket.js';
import { buildHybridCandidateInfo } from './gpu_candidate_builder_hybrid.js';

export function deriveTemporalPrefilterMode(
  ui,
  temporalIndexOptions = {},
  temporalBucketOptions = {}
) {
  if (ui?.temporalPrefilterModeSelect) {
    return ui.temporalPrefilterModeSelect.value;
  }

  if (temporalBucketOptions.useTemporalBucket && temporalIndexOptions.useTemporalIndex) {
    return 'hybrid';
  }
  if (temporalBucketOptions.useTemporalBucket) {
    return 'bucket';
  }
  if (temporalIndexOptions.useTemporalIndex) {
    return 'sorted';
  }
  return 'off';
}

export function buildCandidateInfo(args) {
  const {
    raw,
    stride,
    temporalPrefilterMode,
    useTemporalIndex,
    useTemporalIndexCache,
    temporalWindowMode,
    fixedWindowRadius,
    useTemporalBucket,
    useTemporalBucketCache,
    temporalBucketWidth,
    temporalBucketRadius,
    timestamp,
    sigmaScale,
    temporalSigmaThreshold
  } = args;

  if (!raw || !raw.t || !raw.scale_t) {
    return buildFallbackCandidateInfo({
      raw,
      stride,
      reason: 'no-temporal-data'
    });
  }

  const mode = temporalPrefilterMode || (
    useTemporalBucket && useTemporalIndex
      ? 'hybrid'
      : useTemporalBucket
        ? 'bucket'
        : useTemporalIndex
          ? 'sorted'
          : 'off'
  );

  if (mode === 'hybrid' && useTemporalBucket && useTemporalIndex) {
    return buildHybridCandidateInfo({
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
    });
  }

  if (mode === 'bucket' && useTemporalBucket) {
    return buildBucketCandidateInfo({
      raw,
      timestamp,
      stride,
      useTemporalBucketCache,
      temporalBucketWidth,
      temporalBucketRadius
    });
  }

  if (mode === 'sorted' && useTemporalIndex) {
    return buildSortedCandidateInfo({
      raw,
      timestamp,
      stride,
      sigmaScale,
      temporalSigmaThreshold,
      useTemporalIndexCache,
      temporalWindowMode,
      fixedWindowRadius
    });
  }

  return buildFallbackCandidateInfo({
    raw,
    stride,
    reason: mode
  });
}
