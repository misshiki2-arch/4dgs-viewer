export function buildInteractionExtraLines(buildConfig, buildStats) {
  const lines = [];

  if (buildConfig) {
    lines.push(`interactionActive=${!!buildConfig.interactionActive}`);
    lines.push(`playbackActive=${!!buildConfig.playbackActive}`);
    lines.push(`qualityOverrideActive=${!!buildConfig.qualityOverrideActive}`);
    lines.push(`qualityOverrideReason=${buildConfig.qualityOverrideReason ?? 'none'}`);
    lines.push(`effectiveStride=${buildConfig.stride}`);
    lines.push(`effectiveMaxVisible=${buildConfig.maxVisible}`);
    lines.push(`effectiveRenderScale=${Number(buildConfig.renderScale).toFixed(2)}`);
  }

  if (buildStats) {
    lines.push(
      `buildAccepted=${buildStats.accepted}  buildProcessed=${buildStats.processed}  buildCulled=${buildStats.culled}`
    );
    lines.push(
      `temporalPassed=${buildStats.temporalPassed}  temporalRejected=${buildStats.temporalRejected}  temporalCullRatio=${Number(buildStats.temporalCullRatio).toFixed(3)}`
    );

    if (typeof buildStats.candidateMode !== 'undefined') {
      lines.push(`candidateMode=${buildStats.candidateMode}`);
    }

    if (typeof buildStats.qualityOverrideActive !== 'undefined') {
      lines.push(
        `statsQualityOverrideActive=${!!buildStats.qualityOverrideActive}  statsQualityOverrideReason=${buildStats.qualityOverrideReason ?? 'none'}`
      );
      lines.push(
        `statsInteractionActive=${!!buildStats.interactionActive}  statsPlaybackActive=${!!buildStats.playbackActive}`
      );
    }

    if (typeof buildStats.temporalIndexRangeCount !== 'undefined') {
      lines.push(
        `temporalIndexEnabled=${!!buildStats.temporalIndexEnabled}  temporalIndexCacheEnabled=${!!buildStats.temporalIndexCacheEnabled}`
      );
      lines.push(
        `temporalIndexRangeCount=${buildStats.temporalIndexRangeCount}  temporalIndexCandidateCount=${buildStats.temporalIndexCandidateCount}`
      );
      lines.push(
        `temporalIndexRangeFraction=${Number(buildStats.temporalIndexRangeFraction).toFixed(3)}  temporalIndexCandidateFraction=${Number(buildStats.temporalIndexCandidateFraction).toFixed(3)}`
      );
      lines.push(
        `temporalWindowMode=${buildStats.temporalWindowMode}  temporalWindowRadius=${Number(buildStats.temporalWindowRadius).toFixed(6)}`
      );
      lines.push(
        `temporalIndexCacheHit=${!!buildStats.temporalIndexCacheHit}  temporalIndexBuiltThisFrame=${!!buildStats.temporalIndexBuiltThisFrame}`
      );
      lines.push(
        `temporalWindowCacheHit=${!!buildStats.temporalWindowCacheHit}  temporalWindowBuiltThisFrame=${!!buildStats.temporalWindowBuiltThisFrame}`
      );
      lines.push(
        `temporalIndexTotalCount=${buildStats.temporalIndexTotalCount}  temporalIndexTMin=${Number(buildStats.temporalIndexTMin).toFixed(6)}  temporalIndexTMax=${Number(buildStats.temporalIndexTMax).toFixed(6)}`
      );
    }

    if (typeof buildStats.temporalBucketEnabled !== 'undefined') {
      lines.push(
        `temporalBucketEnabled=${!!buildStats.temporalBucketEnabled}  temporalBucketCacheEnabled=${!!buildStats.temporalBucketCacheEnabled}`
      );
      lines.push(
        `temporalBucketCacheHit=${!!buildStats.temporalBucketCacheHit}  temporalBucketBuiltThisFrame=${!!buildStats.temporalBucketBuiltThisFrame}`
      );
      lines.push(
        `temporalBucketWidth=${Number(buildStats.temporalBucketWidth).toFixed(6)}  temporalBucketRadius=${buildStats.temporalBucketRadius}  temporalBucketCount=${buildStats.temporalBucketCount}`
      );
      lines.push(
        `temporalBucketStart=${buildStats.temporalBucketStart}  temporalBucketEnd=${buildStats.temporalBucketEnd}`
      );
      lines.push(
        `temporalBucketSourceCount=${buildStats.temporalBucketSourceCount}  temporalBucketCandidateCount=${buildStats.temporalBucketCandidateCount}`
      );
      lines.push(
        `temporalBucketSourceFraction=${Number(buildStats.temporalBucketSourceFraction).toFixed(3)}  temporalBucketCandidateFraction=${Number(buildStats.temporalBucketCandidateFraction).toFixed(3)}`
      );

      if (typeof buildStats.temporalBucketPostWindowCandidateCount !== 'undefined') {
        lines.push(
          `temporalBucketPostWindowCandidateCount=${buildStats.temporalBucketPostWindowCandidateCount}  temporalBucketPostWindowCandidateFraction=${Number(buildStats.temporalBucketPostWindowCandidateFraction).toFixed(3)}`
        );
      }

      lines.push(
        `temporalBucketTMin=${Number(buildStats.temporalBucketTMin).toFixed(6)}  temporalBucketTMax=${Number(buildStats.temporalBucketTMax).toFixed(6)}`
      );
    }
  }

  return lines;
}

export function buildTileExtraLines(mode, focusTileIds, focusTileRects) {
  return [
    `tileRadius=${mode ? mode.tileRadius : 0}`,
    `focusTileIds=${focusTileIds && focusTileIds.length > 0 ? '[' + focusTileIds.join(', ') + ']' : 'none'}`,
    `focusTileRects=${focusTileRects ? focusTileRects.length : 0}`,
    `perTileMode=${!!(mode && mode.drawSelectedOnly)}`
  ];
}

export function buildGpuDebugExtraLines({
  buildConfig,
  buildStats,
  mode,
  focusTileIds,
  focusTileRects
}) {
  return [
    ...buildInteractionExtraLines(buildConfig, buildStats),
    ...buildTileExtraLines(mode, focusTileIds, focusTileRects)
  ];
}
