// Step34 redesign
// 目的:
// - debug info builder を「整形だけ」の責務に保つ
// - transform executor / screen-space builder / renderer が確定した truth を、そのまま表示する
//
// 非目標:
// - requested / actual / fallback の補完
// - CPU/GPU path の意味づけ変更
// - source / transform / draw state の推測
//
// 設計:
// 1. buildGpuDebugExtraLines() は入力値を文字列化するだけ
// 2. 値が無いものだけを省略する
// 3. build config -> timing -> tile mode -> ui -> gpu-screen state -> gpu-screen comparison の順で安定化
// 4. Step34 では transform truth source の requested / actual / fallback / upload state をそのまま表示する

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
  const lines = [];
  pushLine(lines, 'tileRadius', fmtInt(mode?.tileRadius));
  pushLine(lines, 'useMaxTile', fmtBool(!!mode?.useMaxTile));
  pushLine(lines, 'selectedTileId', fmtInt(mode?.selectedTileId));

  const focusCount = Array.isArray(focusTileIds) ? focusTileIds.length : 0;
  pushLine(lines, 'focusTileCount', String(focusCount));

  if (focusCount > 0) {
    pushLine(lines, 'focusTileIds', focusTileIds.join(','));
  }

  if (Array.isArray(focusTileRects) && focusTileRects.length > 0) {
    const rectText = focusTileRects
      .map((item) => {
        const tileId = isFiniteNumber(item?.tileId) ? String(item.tileId | 0) : '-1';
        const rect = Array.isArray(item?.rect) ? `[${item.rect.join(', ')}]` : 'none';
        return `${tileId}:${rect}`;
      })
      .join(' ');
    pushLine(lines, 'focusTileRects', rectText);
  }

  return lines;
}

function buildUiLines(ui) {
  const lines = [];
  pushLine(lines, 'bgGraySlider', ui?.bgGraySlider ? String(ui.bgGraySlider.value) : null);
  pushLine(lines, 'drawPathUiValue', ui?.drawPathSelect ? String(ui.drawPathSelect.value) : null);
  pushLine(
    lines,
    'usePackedVisiblePathUi',
    ui?.usePackedVisiblePathCheck ? fmtBool(!!ui.usePackedVisiblePathCheck.checked) : null
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

  pushLine(lines, 'gpuScreenLastActualPath', gpuScreenSummary.gpuScreenLastActualPath ?? 'none');
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

  pushLine(lines, 'gpuScreenUsesPackedReferenceLayout', fmtBool(!!gpuScreenSummary.gpuScreenUsesPackedReferenceLayout));
  pushLine(lines, 'gpuScreenUsesPackedReferenceShader', fmtBool(!!gpuScreenSummary.gpuScreenUsesPackedReferenceShader));
  pushLine(lines, 'gpuScreenUsesPackedReferenceUpload', fmtBool(!!gpuScreenSummary.gpuScreenUsesPackedReferenceUpload));

  return lines;
}

function buildGpuScreenComparisonLines(gpuScreenComparisonSummary) {
  if (!gpuScreenComparisonSummary) return [];

  const lines = [];
  pushLine(lines, 'gpuScreenActualPath', gpuScreenComparisonSummary.actualPath ?? 'gpu-screen');
  pushLine(lines, 'gpuScreenActualRole', gpuScreenComparisonSummary.actualRole ?? 'experimental-draw');

  pushLine(lines, 'gpuScreenSourcePath', gpuScreenComparisonSummary.sourcePath ?? 'none');
  pushLine(lines, 'gpuScreenSourceRole', gpuScreenComparisonSummary.sourceRole ?? 'none');
  pushLine(lines, 'gpuScreenSourceExperimental', fmtBool(!!gpuScreenComparisonSummary.sourceExperimental));
  pushLine(lines, 'gpuScreenSourceBuildMs', fmtNum(gpuScreenComparisonSummary.sourceBuildMs, 3));
  pushLine(lines, 'gpuScreenSourcePackedCount', fmtInt(gpuScreenComparisonSummary.sourcePackedCount));
  pushLine(lines, 'gpuScreenSourcePackedLength', fmtInt(gpuScreenComparisonSummary.sourcePackedLength));

  pushLine(lines, 'gpuScreenSourceItemCount', fmtInt(gpuScreenComparisonSummary.sourceItemCount));
  pushLine(lines, 'gpuScreenSourceSchemaVersion', fmtInt(gpuScreenComparisonSummary.sourceSchemaVersion));
  pushLine(lines, 'gpuScreenSourcePrepStageMs', fmtNum(gpuScreenComparisonSummary.sourcePrepStageMs, 3));
  pushLine(lines, 'gpuScreenSourcePackStageMs', fmtNum(gpuScreenComparisonSummary.sourcePackStageMs, 3));

  // Step34 redesign: transform truth source fields
  pushLine(lines, 'gpuScreenRequestedTransformPath', gpuScreenComparisonSummary.requestedTransformPath ?? null);
  pushLine(lines, 'gpuScreenActualTransformPath', gpuScreenComparisonSummary.actualTransformPath ?? null);
  pushLine(lines, 'gpuScreenTransformPath', gpuScreenComparisonSummary.transformPath ?? null);
  pushLine(lines, 'gpuScreenTransformRole', gpuScreenComparisonSummary.transformRole ?? null);
  pushLine(lines, 'gpuScreenTransformConfigured', fmtBool(!!gpuScreenComparisonSummary.transformConfigured));
  pushLine(lines, 'gpuScreenTransformHasBuffers', fmtBool(!!gpuScreenComparisonSummary.transformHasBuffers));
  pushLine(lines, 'gpuScreenTransformFallbackReason', gpuScreenComparisonSummary.transformFallbackReason ?? null);
  pushLine(lines, 'gpuScreenTransformStageMs', fmtNum(gpuScreenComparisonSummary.transformStageMs, 3));
  pushLine(lines, 'gpuScreenTransformUploadBytes', fmtInt(gpuScreenComparisonSummary.transformUploadBytes));
  pushLine(lines, 'gpuScreenTransformUploadCount', fmtInt(gpuScreenComparisonSummary.transformUploadCount));
  pushLine(lines, 'gpuScreenTransformUploadLength', fmtInt(gpuScreenComparisonSummary.transformUploadLength));
  pushLine(lines, 'gpuScreenTransformUploadCapacityBytes', fmtInt(gpuScreenComparisonSummary.transformUploadCapacityBytes));
  pushLine(lines, 'gpuScreenTransformUploadReusedCapacity', fmtBool(!!gpuScreenComparisonSummary.transformUploadReusedCapacity));

  pushLine(lines, 'gpuScreenReferencePath', gpuScreenComparisonSummary.referencePath ?? 'packed-cpu');
  pushLine(lines, 'gpuScreenReferenceRole', gpuScreenComparisonSummary.referenceRole ?? 'formal-reference');

  pushLine(lines, 'gpuScreenSameLayoutAsReference', fmtBool(!!gpuScreenComparisonSummary.sameLayoutAsReference));
  pushLine(lines, 'gpuScreenSamePackCountAsReference', fmtBool(!!gpuScreenComparisonSummary.samePackCountAsReference));

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
  lines.push(...buildGpuScreenComparisonLines(gpuScreenComparisonSummary));

  return lines;
}
