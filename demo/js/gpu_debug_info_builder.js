// Step24:
// debug 情報の組み立てをこのファイルに集約する。
// ここでは packed 正式契約ベースの表示を行い、旧来の曖昧な opacity / aabb 表示は避ける。
// renderer 側は文字列組み立てを持たず、この helper の結果を表示へ流すだけにする。

function toBoolString(value) {
  return value ? 'true' : 'false';
}

function toNumString(value, digits = 4) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : 'NaN';
}

function toIntString(value) {
  return Number.isFinite(value) ? String(value | 0) : '0';
}

function toArrayString(values, digits = 4) {
  if (!Array.isArray(values)) return 'none';
  return '[' + values.map(v => toNumString(v, digits)).join(', ') + ']';
}

function pushIfDefined(lines, label, value) {
  if (value === undefined || value === null || value === '') return;
  lines.push(`${label}=${value}`);
}

function buildBuildConfigLines(buildConfig) {
  if (!buildConfig) return [];
  return [
    `stride=${toIntString(buildConfig.stride)}`,
    `useRot4d=${toBoolString(!!buildConfig.useRot4d)}`,
    `useSH=${toBoolString(!!buildConfig.useSH)}`,
    `useNativeRot4d=${toBoolString(!!buildConfig.useNativeRot4d)}`,
    `useNativeMarginal=${toBoolString(!!buildConfig.useNativeMarginal)}`,
    `prefilterVar=${toBoolString(!!buildConfig.prefilterVar)}`,
    `sigmaScale=${toNumString(buildConfig.sigmaScale, 4)}`,
    `renderScale=${toNumString(buildConfig.renderScale, 4)}`,
    `scalingModifier=${toNumString(buildConfig.scalingModifier, 4)}`,
    `timestamp=${toNumString(buildConfig.timestamp, 4)}`
  ];
}

function buildBuildStatsLines(buildStats) {
  if (!buildStats) return [];
  return [
    `packedVisiblePathEnabled=${toBoolString(!!buildStats.packedVisiblePathEnabled)}`,
    `packedVisiblePathUsed=${toBoolString(!!buildStats.packedVisiblePathUsed)}`,
    `packedVisiblePath=${buildStats.packedVisiblePath ?? 'none'}`,
    `packedVisibleCount=${toIntString(buildStats.packedVisibleCount)}`,
    `packedVisibleLength=${toIntString(buildStats.packedVisibleLength)}`,
    `packedVisibleFloatsPerItem=${toIntString(buildStats.packedVisibleFloatsPerItem)}`,
    `visibleBuildMs=${toNumString(buildStats.visibleBuildMs, 3)}`,
    `candidateBuildMs=${toNumString(buildStats.candidateBuildMs, 3)}`,
    `screenSpaceBuildMs=${toNumString(buildStats.screenSpaceBuildMs, 3)}`,
    `packedAlphaSource=colorAlpha[3]`
  ];
}

function buildDrawStatsLines(drawStats) {
  if (!drawStats) return [];
  return [
    `visibleCount=${toIntString(drawStats.visibleCount)}`,
    `drawCount=${toIntString(drawStats.drawCount)}`,
    `drawSelectedOnly=${toBoolString(!!drawStats.drawSelectedOnly)}`,
    `showOverlay=${toBoolString(!!drawStats.showOverlay)}`,
    `requestedDrawPath=${drawStats.requestedDrawPath ?? 'legacy'}`,
    `actualDrawPath=${drawStats.actualDrawPath ?? 'legacy'}`,
    `drawPathFallbackReason=${drawStats.drawPathFallbackReason ?? 'none'}`,
    `tileBatchCount=${toIntString(drawStats.tileBatchCount)}`,
    `nonEmptyTileBatchCount=${toIntString(drawStats.nonEmptyTileBatchCount)}`,
    `totalTileDrawCount=${toIntString(drawStats.totalTileDrawCount)}`,
    `maxTileDrawCount=${toIntString(drawStats.maxTileDrawCount)}`,
    `maxTileId=${toIntString(drawStats.maxTileId)}`,
    `uploadCount=${toIntString(drawStats.uploadCount)}`,
    `drawCallCount=${toIntString(drawStats.drawCallCount)}`,
    `packedVisiblePath=${drawStats.packedVisiblePath ?? 'none'}`,
    `packedVisibleCount=${toIntString(drawStats.packedVisibleCount)}`,
    `packedVisibleLength=${toIntString(drawStats.packedVisibleLength)}`,
    `packedVisibleFloatsPerItem=${toIntString(drawStats.packedVisibleFloatsPerItem)}`,
    `packedUploadLayoutVersion=${toIntString(drawStats.packedUploadLayoutVersion)}`,
    `packedUploadStrideBytes=${toIntString(drawStats.packedUploadStrideBytes)}`,
    `packedUploadBytes=${toIntString(drawStats.packedUploadBytes)}`,
    `packedUploadCount=${toIntString(drawStats.packedUploadCount)}`,
    `packedUploadLength=${toIntString(drawStats.packedUploadLength)}`,
    `packedUploadCapacityBytes=${toIntString(drawStats.packedUploadCapacityBytes)}`,
    `packedUploadReusedCapacity=${toBoolString(!!drawStats.packedUploadReusedCapacity)}`,
    `packedUploadManagedCapacityReused=${toBoolString(!!drawStats.packedUploadManagedCapacityReused)}`,
    `packedUploadManagedCapacityGrown=${toBoolString(!!drawStats.packedUploadManagedCapacityGrown)}`,
    `packedUploadManagedUploadCount=${toIntString(drawStats.packedUploadManagedUploadCount)}`,
    `packedUploadAlphaSource=${drawStats.packedUploadAlphaSource ?? 'colorAlpha[3]'}`,
    `packedDirectDraw=${toBoolString(!!drawStats.packedDirectDraw)}`,
    `packedDirectConfigured=${toBoolString(!!drawStats.packedDirectConfigured)}`,
    `packedDirectHasVao=${toBoolString(!!drawStats.packedDirectHasVao)}`,
    `packedDirectLayoutVersion=${toIntString(drawStats.packedDirectLayoutVersion)}`,
    `packedDirectStrideBytes=${toIntString(drawStats.packedDirectStrideBytes)}`,
    `packedDirectAttributeCount=${toIntString(drawStats.packedDirectAttributeCount)}`,
    `packedDirectOffsets=${drawStats.packedDirectOffsets ?? ''}`,
    `packedDirectAlphaSource=${drawStats.packedDirectAlphaSource ?? 'aColorAlpha.a -> colorAlpha[3]'}`,
    `packedInterleavedBound=${toBoolString(!!drawStats.packedInterleavedBound)}`,
    `legacyExpandedArraysBuilt=${toBoolString(!!drawStats.legacyExpandedArraysBuilt)}`
  ];
}

function buildModeLines(mode, focusTileIds, focusTileRects) {
  const lines = [];
  if (!mode) return lines;

  lines.push(`tileRadius=${toIntString(mode.tileRadius)}`);
  lines.push(`useMaxTile=${toBoolString(!!mode.useMaxTile)}`);
  lines.push(`selectedTileId=${toIntString(mode.selectedTileId)}`);
  lines.push(`focusTileCount=${Array.isArray(focusTileIds) ? focusTileIds.length : 0}`);

  if (Array.isArray(focusTileIds) && focusTileIds.length > 0) {
    lines.push(`focusTileIds=${focusTileIds.join(',')}`);
  }

  if (Array.isArray(focusTileRects) && focusTileRects.length > 0) {
    const rectText = focusTileRects.map(item => {
      const rect = Array.isArray(item?.rect) ? item.rect.join(',') : 'none';
      return `${item?.tileId ?? -1}:[${rect}]`;
    }).join(' ');
    lines.push(`focusTileRects=${rectText}`);
  }

  return lines;
}

function buildSampleLines(samples) {
  if (!samples) return [];
  const legacy = samples.legacySample ?? null;
  const packed = samples.packedSample ?? null;

  return [
    `legacySampleCenterPx=${legacy ? toArrayString(legacy.centerPx, 3) : 'none'}`,
    `packedSampleCenterPx=${packed ? toArrayString(packed.centerPx, 3) : 'none'}`,
    `legacySampleRadiusPx=${legacy ? toNumString(legacy.radiusPx, 3) : 'none'}`,
    `packedSampleRadiusPx=${packed ? toNumString(packed.radiusPx, 3) : 'none'}`,
    `legacySampleColorAlpha=${legacy ? toArrayString(legacy.colorAlpha, 4) : 'none'}`,
    `packedSampleColorAlpha=${packed ? toArrayString(packed.colorAlpha, 4) : 'none'}`,
    `legacySampleConic=${legacy ? toArrayString(legacy.conic, 4) : 'none'}`,
    `packedSampleConic=${packed ? toArrayString(packed.conic, 4) : 'none'}`,
    `packedSampleReserved=${packed ? toNumString(packed.reserved, 4) : 'none'}`,
    `packedSampleMisc=${packed ? toArrayString(packed.misc, 4) : 'none'}`
  ];
}

export function buildGpuDebugExtraLines({
  buildConfig = null,
  buildStats = null,
  drawStats = null,
  mode = null,
  focusTileIds = [],
  focusTileRects = [],
  ui = null,
  samples = null
} = {}) {
  const lines = [];

  lines.push(...buildBuildConfigLines(buildConfig));
  lines.push(...buildBuildStatsLines(buildStats));
  lines.push(...buildDrawStatsLines(drawStats));
  lines.push(...buildModeLines(mode, focusTileIds, focusTileRects));
  lines.push(...buildSampleLines(samples));

  if (ui) {
    pushIfDefined(lines, 'bgGraySlider', ui.bgGraySlider ? ui.bgGraySlider.value : undefined);
    pushIfDefined(lines, 'drawPathUiValue', ui.drawPathSelect ? ui.drawPathSelect.value : undefined);
    pushIfDefined(
      lines,
      'usePackedVisiblePathUi',
      ui.usePackedVisiblePathCheck ? toBoolString(!!ui.usePackedVisiblePathCheck.checked) : undefined
    );
  }

  return lines;
}
