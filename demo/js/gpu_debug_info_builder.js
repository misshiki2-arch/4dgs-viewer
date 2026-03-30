export function buildInteractionExtraLines(buildConfig, buildStats) {
  const lines = [];

  if (buildConfig) {
    lines.push(`interactionActive=${!!buildConfig.interactionActive}`);
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

    if (typeof buildStats.temporalIndexRangeCount !== 'undefined') {
      lines.push(
        `temporalIndexRangeCount=${buildStats.temporalIndexRangeCount}  temporalIndexCandidateCount=${buildStats.temporalIndexCandidateCount}`
      );
      lines.push(
        `temporalIndexRangeFraction=${Number(buildStats.temporalIndexRangeFraction).toFixed(3)}  temporalIndexCandidateFraction=${Number(buildStats.temporalIndexCandidateFraction).toFixed(3)}`
      );
      lines.push(
        `temporalWindowRadius=${Number(buildStats.temporalWindowRadius).toFixed(6)}`
      );
    }

    if (typeof buildStats.temporalIndexCacheHit !== 'undefined') {
      lines.push(
        `temporalIndexCacheHit=${!!buildStats.temporalIndexCacheHit}  temporalIndexBuiltThisFrame=${!!buildStats.temporalIndexBuiltThisFrame}`
      );
      lines.push(
        `temporalIndexTotalCount=${buildStats.temporalIndexTotalCount}  temporalIndexTMin=${Number(buildStats.temporalIndexTMin).toFixed(6)}  temporalIndexTMax=${Number(buildStats.temporalIndexTMax).toFixed(6)}`
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
