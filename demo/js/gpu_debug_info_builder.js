// Step28:
// debug 情報の重複を減らし、gpu-screen experimental path の比較表示を整理する。
// renderer 側ですでに表示している packed/sample 詳細はここでは出さない。
// ここでは主に以下だけを返す。
// - build config の補助情報
// - timing 系
// - tile mode / UI 状態
// - gpu-screen state
// - gpu-screen comparison
//
// Step27 で出していた gpu-screen の一部重複項目を削減し、
// 「state」と「comparison」を分けて扱う。

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

function buildGpuScreenStateLines(gpuScreenSummary) {
  if (!gpuScreenSummary) return [];

  const lines = [];
  pushLine(lines, 'gpuScreenDrawReady', fmtBool(!!gpuScreenSummary.gpuScreenDrawReady));
  pushLine(lines, 'gpuScreenConfigured', fmtBool(!!gpuScreenSummary.gpuScreenConfigured));
  pushLine(lines, 'gpuScreenHasProgram', fmtBool(!!gpuScreenSummary.gpuScreenHasProgram));
  pushLine(lines, 'gpuScreenHasVao', fmtBool(!!gpuScreenSummary.gpuScreenHasVao));
  pushLine(lines, 'gpuScreenHasBuffers', fmtBool(!!gpuScreenSummary.gpuScreenHasBuffers));

  pushLine(lines, 'gpuScreenReason', gpuScreenSummary.gpuScreenReason ?? 'unknown');
  pushLine(lines, 'gpuScreenLastPath', gpuScreenSummary.gpuScreenLastPath ?? 'gpu-screen');
  pushLine(lines, 'gpuScreenLastDrawCount', fmtInt(gpuScreenSummary.gpuScreenLastDrawCount));

  pushLine(lines, 'gpuScreenLayoutVersion', fmtInt(gpuScreenSummary.gpuScreenLayoutVersion));
  pushLine(lines, 'gpuScreenStrideBytes', fmtInt(gpuScreenSummary.gpuScreenStrideBytes));
  pushLine(lines, 'gpuScreenAttributeCount', fmtInt(gpuScreenSummary.gpuScreenAttributeCount));
  pushLine(lines, 'gpuScreenOffsets', gpuScreenSummary.gpuScreenOffsets ?? '');

  pushLine(lines, 'gpuScreenUploadBytes', fmtInt(gpuScreenSummary.gpuScreenUploadBytes));
  pushLine(lines, 'gpuScreenUploadCount', fmtInt(gpuScreenSummary.gpuScreenUploadCount));
  pushLine(lines, 'gpuScreenUploadLength', fmtInt(gpuScreenSummary.gpuScreenUploadLength));
  pushLine(lines, 'gpuScreenUploadCapacityBytes', fmtInt(gpuScreenSummary.gpuScreenUploadCapacityBytes));
  pushLine(lines, 'gpuScreenUploadReusedCapacity', fmtBool(!!gpuScreenSummary.gpuScreenUploadReusedCapacity));
  pushLine(lines, 'gpuScreenUploadManagedCapacityReused', fmtBool(!!gpuScreenSummary.gpuScreenUploadManagedCapacityReused));
  pushLine(lines, 'gpuScreenUploadManagedCapacityGrown', fmtBool(!!gpuScreenSummary.gpuScreenUploadManagedCapacityGrown));
  pushLine(lines, 'gpuScreenUploadManagedUploadCount', fmtInt(gpuScreenSummary.gpuScreenUploadManagedUploadCount));

  return lines;
}

function buildGpuScreenComparisonLines(gpuScreenSummary, gpuScreenComparisonSummary) {
  if (!gpuScreenSummary && !gpuScreenComparisonSummary) return [];

  const lines = [];
  const referencePath =
    gpuScreenComparisonSummary?.referencePath ??
    gpuScreenSummary?.gpuScreenReferencePath ??
    'packed-cpu';

  pushLine(lines, 'gpuScreenReferencePath', referencePath);
  pushLine(lines, 'gpuScreenReferenceRole', gpuScreenComparisonSummary?.referenceRole ?? 'formal-reference');
  pushLine(lines, 'gpuScreenCurrentPath', gpuScreenComparisonSummary?.currentPath ?? 'none');
  pushLine(lines, 'gpuScreenCurrentRole', gpuScreenComparisonSummary?.currentRole ?? 'experimental');
  pushLine(lines, 'gpuScreenCurrentExperimental', fmtBool(!!gpuScreenComparisonSummary?.currentExperimental));

  pushLine(lines, 'gpuScreenUsesPackedReferenceLayout', fmtBool(!!gpuScreenSummary?.gpuScreenUsesPackedReferenceLayout));
  pushLine(lines, 'gpuScreenUsesPackedReferenceShader', fmtBool(!!gpuScreenSummary?.gpuScreenUsesPackedReferenceShader));
  pushLine(lines, 'gpuScreenUsesPackedReferenceUpload', fmtBool(!!gpuScreenSummary?.gpuScreenUsesPackedReferenceUpload));

  pushLine(lines, 'gpuScreenCurrentBuildMs', fmtNum(gpuScreenComparisonSummary?.currentBuildMs, 3));
  pushLine(lines, 'gpuScreenCurrentPackedCount', fmtInt(gpuScreenComparisonSummary?.currentPackedCount));
  pushLine(lines, 'gpuScreenCurrentPackedLength', fmtInt(gpuScreenComparisonSummary?.currentPackedLength));

  pushLine(lines, 'gpuScreenSameLayoutAsReference', fmtBool(!!gpuScreenComparisonSummary?.sameLayoutAsReference));
  pushLine(lines, 'gpuScreenSamePackCountAsReference', fmtBool(!!gpuScreenComparisonSummary?.samePackCountAsReference));

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
  gpuScreenSummary = null,
  gpuScreenComparisonSummary = null
} = {}) {
  const lines = [];

  lines.push(...buildConfigLines(buildConfig));
  lines.push(...buildTimingLines(buildStats, drawStats, gpuScreenSummary));
  lines.push(...buildModeLines(mode, focusTileIds, focusTileRects));
  lines.push(...buildUiLines(ui));
  lines.push(...buildGpuScreenStateLines(gpuScreenSummary));
  lines.push(...buildGpuScreenComparisonLines(gpuScreenSummary, gpuScreenComparisonSummary));

  return lines;
}
