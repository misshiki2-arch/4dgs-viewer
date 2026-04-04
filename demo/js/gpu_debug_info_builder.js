// Step26:
// debug 情報の重複を抑えつつ、gpu-screen 実験経路の状態を見えるようにする。
// renderer 側ですでに表示している packed/sample 詳細はここでは出さない。
// ここでは主に以下だけを返す。
// - build config の補助情報
// - timing 系
// - tile mode / UI 状態
// - gpu-screen readiness / summary

function isFiniteNumber(v) {
  return Number.isFinite(v);
}

function fmtBool(v) {
  return v ? 'true' : 'false';
}

function fmtNum(v, digits = 4) {
  return isFiniteNumber(v) ? Number(v).toFixed(digits) : null;
}

function fmtInt(v) {
  return isFiniteNumber(v) ? String(v | 0) : null;
}

function fmtArray(arr, digits = 3) {
  if (!Array.isArray(arr)) return null;
  return '[' + arr.map((v) => (isFiniteNumber(v) ? Number(v).toFixed(digits) : 'NaN')).join(', ') + ']';
}

function pushLine(lines, key, value) {
  if (value === null || value === undefined || value === '') return;
  lines.push(`${key}=${value}`);
}

function buildConfigLines(buildConfig) {
  if (!buildConfig) return [];

  const lines = [];
  pushLine(lines, 'stride', fmtInt(buildConfig.stride));
  pushLine(lines, 'useRot4d', fmtBool(!!buildConfig.useRot4d));
  pushLine(lines, 'useSH', fmtBool(!!buildConfig.useSH));
  pushLine(lines, 'useNativeRot4d', fmtBool(!!buildConfig.useNativeRot4d));
  pushLine(lines, 'useNativeMarginal', fmtBool(!!buildConfig.useNativeMarginal));
  pushLine(lines, 'prefilterVarEnabled', fmtBool(!!buildConfig.prefilterVar));
  pushLine(lines, 'sigmaScale', fmtNum(buildConfig.sigmaScale, 4));
  pushLine(lines, 'renderScale', fmtNum(buildConfig.renderScale, 4));
  pushLine(lines, 'scalingModifier', fmtNum(buildConfig.scalingModifier, 4));
  pushLine(lines, 'timestamp', fmtNum(buildConfig.timestamp, 4));
  return lines;
}

function buildTimingLines(buildStats, drawStats, gpuScreenSummary) {
  const lines = [];

  pushLine(lines, 'visibleBuildMs', fmtNum(buildStats?.visibleBuildMs, 3));
  pushLine(lines, 'candidateBuildMs', fmtNum(buildStats?.candidateBuildMs, 3));
  pushLine(lines, 'screenSpaceBuildMs', fmtNum(buildStats?.screenSpaceBuildMs, 3));
  pushLine(lines, 'totalBuildMs', fmtNum(buildStats?.totalBuildMs, 3));

  pushLine(lines, 'drawFraction', fmtNum(drawStats?.drawFraction, 6));
  pushLine(lines, 'gpuScreenLastBuildMs', fmtNum(gpuScreenSummary?.gpuScreenLastBuildMs, 3));

  return lines;
}

function buildModeLines(mode, focusTileIds, focusTileRects) {
  if (!mode) return [];

  const lines = [];
  pushLine(lines, 'tileRadius', fmtInt(mode.tileRadius));
  pushLine(lines, 'useMaxTile', fmtBool(!!mode.useMaxTile));
  pushLine(lines, 'selectedTileId', fmtInt(mode.selectedTileId));
  pushLine(lines, 'focusTileCount', Array.isArray(focusTileIds) ? String(focusTileIds.length) : '0');

  if (Array.isArray(focusTileIds) && focusTileIds.length > 0) {
    pushLine(lines, 'focusTileIds', focusTileIds.join(','));
  }

  if (Array.isArray(focusTileRects) && focusTileRects.length > 0) {
    const rectText = focusTileRects
      .map((item) => {
        const tileId = isFiniteNumber(item?.tileId) ? String(item.tileId | 0) : '-1';
        const rect = fmtArray(item?.rect, 0) ?? 'none';
        return `${tileId}:${rect}`;
      })
      .join(' ');
    pushLine(lines, 'focusTileRects', rectText);
  }

  return lines;
}

function buildUiLines(ui) {
  if (!ui) return [];

  const lines = [];
  pushLine(lines, 'bgGraySlider', ui.bgGraySlider ? String(ui.bgGraySlider.value) : null);
  pushLine(lines, 'drawPathUiValue', ui.drawPathSelect ? String(ui.drawPathSelect.value) : null);
  pushLine(
    lines,
    'usePackedVisiblePathUi',
    ui.usePackedVisiblePathCheck ? fmtBool(!!ui.usePackedVisiblePathCheck.checked) : null
  );
  return lines;
}

function buildGpuScreenLines(gpuScreenSummary) {
  if (!gpuScreenSummary) return [];

  const lines = [];
  pushLine(lines, 'gpuScreenDrawReady', fmtBool(!!gpuScreenSummary.gpuScreenDrawReady));
  pushLine(lines, 'gpuScreenConfigured', fmtBool(!!gpuScreenSummary.gpuScreenConfigured));
  pushLine(lines, 'gpuScreenHasProgram', fmtBool(!!gpuScreenSummary.gpuScreenHasProgram));
  pushLine(lines, 'gpuScreenHasVao', fmtBool(!!gpuScreenSummary.gpuScreenHasVao));
  pushLine(lines, 'gpuScreenHasBuffers', fmtBool(!!gpuScreenSummary.gpuScreenHasBuffers));
  pushLine(lines, 'gpuScreenReason', gpuScreenSummary.gpuScreenReason ?? 'unknown');
  pushLine(lines, 'gpuScreenLayoutVersion', fmtInt(gpuScreenSummary.gpuScreenLayoutVersion));
  pushLine(lines, 'gpuScreenStrideBytes', fmtInt(gpuScreenSummary.gpuScreenStrideBytes));
  pushLine(lines, 'gpuScreenAttributeCount', fmtInt(gpuScreenSummary.gpuScreenAttributeCount));
  pushLine(lines, 'gpuScreenOffsets', gpuScreenSummary.gpuScreenOffsets ?? '');
  pushLine(lines, 'gpuScreenLastPath', gpuScreenSummary.gpuScreenLastPath ?? 'gpu-screen');
  pushLine(lines, 'gpuScreenLastDrawCount', fmtInt(gpuScreenSummary.gpuScreenLastDrawCount));

  const uploadSummary = gpuScreenSummary.gpuScreenUploadSummary;
  if (uploadSummary) {
    pushLine(lines, 'gpuScreenUploadBytes', fmtInt(uploadSummary.packedUploadBytes));
    pushLine(lines, 'gpuScreenUploadCount', fmtInt(uploadSummary.packedUploadCount));
    pushLine(lines, 'gpuScreenUploadLength', fmtInt(uploadSummary.packedUploadLength));
    pushLine(lines, 'gpuScreenUploadCapacityBytes', fmtInt(uploadSummary.packedUploadCapacityBytes));
    pushLine(lines, 'gpuScreenUploadReusedCapacity', fmtBool(!!uploadSummary.packedUploadReusedCapacity));
    pushLine(lines, 'gpuScreenUploadManagedCapacityReused', fmtBool(!!uploadSummary.packedUploadManagedCapacityReused));
    pushLine(lines, 'gpuScreenUploadManagedCapacityGrown', fmtBool(!!uploadSummary.packedUploadManagedCapacityGrown));
    pushLine(lines, 'gpuScreenUploadManagedUploadCount', fmtInt(uploadSummary.packedUploadManagedUploadCount));
  }

  return lines;
}

export function buildGpuDebugExtraLines({
  buildConfig = null,
  buildStats = null,
  drawStats = null,
  mode = null,
  focusTileIds = [],
  focusTileRects = [],
  ui = null,
  gpuScreenSummary = null
} = {}) {
  const lines = [];

  lines.push(...buildConfigLines(buildConfig));
  lines.push(...buildTimingLines(buildStats, drawStats, gpuScreenSummary));
  lines.push(...buildModeLines(mode, focusTileIds, focusTileRects));
  lines.push(...buildUiLines(ui));
  lines.push(...buildGpuScreenLines(gpuScreenSummary));

  return lines;
}
