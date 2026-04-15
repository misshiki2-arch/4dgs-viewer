// Step63 display cleanup
// 目的:
// - debug info builder を「整形だけ」の責務に保つ
// - transform executor / screen-space builder / renderer が確定した truth を、そのまま表示する
//
// 非目標:
// - requested / actual / fallback の補完
// - CPU/GPU path の意味づけ変更
// - source / transform / draw state の推測
// - transformBatchSummary の再計算や補正
//
// 設計:
// 1. buildGpuDebugExtraLines() は入力値を文字列化するだけ
// 2. 値が無いものだけを省略する
// 3. build config -> timing -> tile mode -> ui -> gpu-screen state -> gpu-screen comparison の順で安定化
// 4. transform truth source の requested / actual / fallback / upload state をそのまま表示する
// 5. executor-owned transformBatchSummary と gpu-screen draw path の GPU-resident usage も、そのまま表示する

function isFiniteNumber(v) {
  return Number.isFinite(v);
}

function fmtBool(v) {
  return v ? 'true' : 'false';
}

function fmtNum(v, digits = 4) {
  return isFiniteNumber(v) ? Number(v).toFixed(digits) : null;
}

function fmtArr(arr, digits = 4) {
  if (!Array.isArray(arr)) return null;
  return '[' + arr.map((v) => (isFiniteNumber(v) ? Number(v).toFixed(digits) : 'NaN')).join(', ') + ']';
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
  pushLine(lines, 'gpuScreenUsesGpuResidentPayload', fmtBool(!!gpuScreenSummary.gpuScreenUsesGpuResidentPayload));
  pushLine(lines, 'gpuScreenSharedSetupCount', fmtInt(gpuScreenSummary.gpuScreenSharedSetupCount));
  pushLine(lines, 'gpuScreenSharedBindCount', fmtInt(gpuScreenSummary.gpuScreenSharedBindCount));
  pushLine(lines, 'gpuScreenSharedDispatchCount', fmtInt(gpuScreenSummary.gpuScreenSharedDispatchCount));
  pushLine(lines, 'gpuScreenSharedDispatchMode', gpuScreenSummary.gpuScreenSharedDispatchMode ?? 'none');
  pushLine(lines, 'gpuScreenSharedPayloadCount', fmtInt(gpuScreenSummary.gpuScreenSharedPayloadCount));
  pushLine(lines, 'gpuScreenGpuResidentPayloadAvailable', fmtBool(!!gpuScreenSummary.gpuScreenGpuResidentPayloadAvailable));

  return lines;
}

function buildGpuScreenComparisonLines(gpuScreenComparisonSummary) {
  if (!gpuScreenComparisonSummary) return [];

  const lines = [];
  pushLine(lines, 'gpuScreenActualPath', gpuScreenComparisonSummary.actualPath ?? 'gpu-screen');
  pushLine(lines, 'gpuScreenActualRole', gpuScreenComparisonSummary.actualRole ?? 'experimental-draw');

  pushLine(lines, 'gpuScreenSourcePath', gpuScreenComparisonSummary.sourcePath ?? 'none');
  pushLine(lines, 'gpuScreenSourceRole', gpuScreenComparisonSummary.sourceRole ?? 'none');
  pushLine(lines, 'gpuScreenSourceContract', gpuScreenComparisonSummary.sourceContract ?? 'unavailable');
  pushLine(lines, 'gpuScreenSourceExperimental', fmtBool(!!gpuScreenComparisonSummary.sourceExperimental));
  pushLine(lines, 'gpuScreenSourceBuildMs', fmtNum(gpuScreenComparisonSummary.sourceBuildMs, 3));
  pushLine(lines, 'gpuScreenSourcePackedCount', fmtInt(gpuScreenComparisonSummary.sourcePackedCount));
  pushLine(lines, 'gpuScreenSourcePackedLength', fmtInt(gpuScreenComparisonSummary.sourcePackedLength));

  pushLine(lines, 'gpuScreenSourceItemCount', fmtInt(gpuScreenComparisonSummary.sourceItemCount));
  pushLine(lines, 'gpuScreenSourceSchemaVersion', fmtInt(gpuScreenComparisonSummary.sourceSchemaVersion));
  pushLine(lines, 'gpuScreenSourcePrepStageMs', fmtNum(gpuScreenComparisonSummary.sourcePrepStageMs, 3));
  pushLine(lines, 'gpuScreenSourcePackStageMs', fmtNum(gpuScreenComparisonSummary.sourcePackStageMs, 3));

  // Transform truth source fields are forwarded without reinterpretation.
  pushLine(lines, 'gpuScreenRequestedTransformPath', gpuScreenComparisonSummary.requestedTransformPath);
  pushLine(lines, 'gpuScreenActualTransformPath', gpuScreenComparisonSummary.actualTransformPath);
  pushLine(lines, 'gpuScreenTransformPath', gpuScreenComparisonSummary.transformPath);
  pushLine(lines, 'gpuScreenTransformRole', gpuScreenComparisonSummary.transformRole);
  pushLine(lines, 'gpuScreenTransformConfigured', gpuScreenComparisonSummary.transformConfigured);
  pushLine(lines, 'gpuScreenTransformHasBuffers', gpuScreenComparisonSummary.transformHasBuffers);
  pushLine(lines, 'gpuScreenTransformFallbackReason', gpuScreenComparisonSummary.transformFallbackReason);
  pushLine(lines, 'gpuScreenTransformFallbackContract', gpuScreenComparisonSummary.transformFallbackContract);
  pushLine(lines, 'gpuScreenTransformStageMs', gpuScreenComparisonSummary.transformStageMs);
  pushLine(lines, 'gpuScreenTransformUploadBytes', gpuScreenComparisonSummary.transformUploadBytes);
  pushLine(lines, 'gpuScreenTransformUploadCount', gpuScreenComparisonSummary.transformUploadCount);
  pushLine(lines, 'gpuScreenTransformUploadLength', gpuScreenComparisonSummary.transformUploadLength);
  pushLine(lines, 'gpuScreenTransformUploadCapacityBytes', gpuScreenComparisonSummary.transformUploadCapacityBytes);
  pushLine(lines, 'gpuScreenTransformUploadReusedCapacity', gpuScreenComparisonSummary.transformUploadReusedCapacity);

  pushLine(lines, 'gpuScreenReferencePath', gpuScreenComparisonSummary.referencePath ?? 'packed-cpu');
  pushLine(lines, 'gpuScreenReferenceRole', gpuScreenComparisonSummary.referenceRole ?? 'formal-reference');

  pushLine(lines, 'gpuScreenSameLayoutAsReference', fmtBool(!!gpuScreenComparisonSummary.sameLayoutAsReference));
  pushLine(lines, 'gpuScreenSamePackCountAsReference', fmtBool(!!gpuScreenComparisonSummary.samePackCountAsReference));

  return lines;
}

function buildTransformBatchLines(transformBatchSummary) {
  if (!transformBatchSummary) return [];

  const lines = [];
  pushLine(lines, 'transformBatchPlanMode', transformBatchSummary.planMode);
  pushLine(lines, 'transformBatchCount', fmtInt(transformBatchSummary.batchCount));
  pushLine(lines, 'transformBatchMaxItems', fmtInt(transformBatchSummary.maxBatchItems));
  pushLine(lines, 'transformBatchPreferredItems', fmtInt(transformBatchSummary.preferredBatchItems));
  pushLine(lines, 'transformBatchPreferredPolicy', transformBatchSummary.preferredBatchPolicy ?? 'preferred-batch-none');
  pushLine(lines, 'transformBatchLargestItemCount', fmtInt(transformBatchSummary.largestBatchItemCount));
  pushLine(lines, 'transformBatchGpuCount', fmtInt(transformBatchSummary.gpuBatchCount));
  pushLine(lines, 'transformBatchCpuFallbackCount', fmtInt(transformBatchSummary.cpuFallbackBatchCount));
  pushLine(lines, 'transformBatchAllGpuSuccess', fmtBool(!!transformBatchSummary.allBatchesGpuSuccess));
  return lines;
}

function buildTransformLifecycleLines(transformSummary) {
  if (!transformSummary) return [];

  const lines = [];
  pushLine(lines, 'transformPayloadOwner', transformSummary.transformPayloadOwner ?? 'none');
  pushLine(lines, 'transformActivePayloadCount', fmtInt(transformSummary.transformActivePayloadCount));
  pushLine(lines, 'transformReusablePayloadCount', fmtInt(transformSummary.transformReusablePayloadCount));
  pushLine(lines, 'transformReleasedPayloadCount', fmtInt(transformSummary.transformReleasedPayloadCount));
  pushLine(lines, 'transformPayloadPoolReleaseCount', fmtInt(transformSummary.transformPayloadPoolReleaseCount));
  pushLine(lines, 'transformPayloadReuseCount', fmtInt(transformSummary.transformPayloadReuseCount));
  pushLine(lines, 'transformPayloadCreateCount', fmtInt(transformSummary.transformPayloadCreateCount));
  pushLine(lines, 'transformPayloadTrimCount', fmtInt(transformSummary.transformPayloadTrimCount));
  pushLine(lines, 'transformPayloadRetainedCount', fmtInt(transformSummary.transformPayloadRetainedCount));
  pushLine(lines, 'transformPayloadPoolHighWaterCount', fmtInt(transformSummary.transformPayloadPoolHighWaterCount));
  pushLine(lines, 'transformPayloadPoolBaseRetained', fmtInt(transformSummary.transformPayloadPoolBaseRetained));
  pushLine(lines, 'transformPayloadPoolHardMaxRetained', fmtInt(transformSummary.transformPayloadPoolHardMaxRetained));
  pushLine(lines, 'transformPayloadPoolMaxRetained', fmtInt(transformSummary.transformPayloadPoolMaxRetained));
  pushLine(lines, 'transformPayloadPoolPolicy', transformSummary.transformPayloadPoolPolicy ?? 'adaptive-hold-base');
  pushLine(lines, 'transformPayloadResetReason', transformSummary.transformPayloadResetReason ?? 'none');
  pushLine(lines, 'transformPayloadGeneration', fmtInt(transformSummary.transformPayloadGeneration));
  pushLine(lines, 'transformDispatchCount', fmtInt(transformSummary.transformDispatchCount));
  pushLine(lines, 'transformDispatchMode', transformSummary.transformDispatchMode ?? 'none');
  pushLine(lines, 'transformDispatchUploadBytes', fmtInt(transformSummary.transformDispatchUploadBytes));
  pushLine(lines, 'transformDispatchItemCount', fmtInt(transformSummary.transformDispatchItemCount));
  return lines;
}

export function buildLegacySample(visible) {
  if (!Array.isArray(visible) || visible.length === 0) return null;
  const v = visible[0];
  return {
    centerPx: [v.px, v.py],
    radiusPx: v.radius,
    colorAlpha: Array.isArray(v.colorAlpha) ? v.colorAlpha.slice(0, 4) : null,
    conic: Array.isArray(v.conic) ? v.conic.slice(0, 3) : null
  };
}

export function buildPackedSample(screenSpace) {
  const packed = screenSpace?.packed;
  if (!(packed instanceof Float32Array) || packed.length < 16) return null;
  return {
    centerPx: [packed[0], packed[1]],
    radiusPx: packed[2],
    depth: packed[3],
    colorAlpha: [packed[4], packed[5], packed[6], packed[7]],
    conic: [packed[8], packed[9], packed[10]],
    reserved: packed[11],
    misc: [packed[12], packed[13], packed[14], packed[15]]
  };
}

export function buildPackedLines(buildStats, drawPathSelection, drawStats) {
  return [
    `packedVisiblePathEnabled=${!!buildStats?.packedVisiblePathEnabled}`,
    `packedVisiblePathUsed=${!!buildStats?.packedVisiblePathUsed}`,
    `packedVisiblePath=${buildStats?.packedVisiblePath ?? 'none'}`,
    `packedVisibleCount=${buildStats?.packedVisibleCount ?? 0}`,
    `packedVisibleLength=${buildStats?.packedVisibleLength ?? 0}`,
    `packedVisibleFloatsPerItem=${buildStats?.packedVisibleFloatsPerItem ?? 0}`,
    `packedAlphaSource=colorAlpha[3]`,
    `requestedDrawPath=${drawPathSelection?.requestedPath}`,
    `actualDrawPath=${drawPathSelection?.actualPath}`,
    `drawPathFallbackReason=${drawPathSelection?.fallbackReason}`,
    `packedUploadBytes=${drawStats?.packedUploadBytes ?? 0}`,
    `packedUploadCount=${drawStats?.packedUploadCount ?? 0}`,
    `packedUploadLength=${drawStats?.packedUploadLength ?? 0}`,
    `packedUploadCapacityBytes=${drawStats?.packedUploadCapacityBytes ?? 0}`,
    `packedUploadReusedCapacity=${!!drawStats?.packedUploadReusedCapacity}`,
    `packedDirectDraw=${!!drawStats?.packedDirectDraw}`,
    `packedDirectUsesGpuResidentPayload=${!!drawStats?.packedDirectUsesGpuResidentPayload}`,
    `packedDirectSharedSetupCount=${drawStats?.packedDirectSharedSetupCount ?? 0}`,
    `packedDirectSharedBindCount=${drawStats?.packedDirectSharedBindCount ?? 0}`,
    `packedDirectSharedDispatchCount=${drawStats?.packedDirectSharedDispatchCount ?? 0}`,
    `packedDirectSharedDispatchMode=${drawStats?.packedDirectSharedDispatchMode ?? 'none'}`,
    `packedDirectSharedPayloadCount=${drawStats?.packedDirectSharedPayloadCount ?? 0}`,
    `packedDirectGpuResidentPayloadAvailable=${!!drawStats?.packedDirectGpuResidentPayloadAvailable}`,
    `packedDirectLayoutVersion=${drawStats?.packedDirectLayoutVersion ?? 0}`,
    `packedDirectStrideBytes=${drawStats?.packedDirectStrideBytes ?? 0}`,
    `packedDirectAttributeCount=${drawStats?.packedDirectAttributeCount ?? 0}`,
    `packedDirectOffsets=${drawStats?.packedDirectOffsets ?? ''}`,
    `packedInterleavedBound=${!!drawStats?.packedInterleavedBound}`,
    `legacyExpandedArraysBuilt=${!!drawStats?.legacyExpandedArraysBuilt}`
  ];
}

export function buildSampleLines(legacySample, packedSample) {
  return [
    `legacySampleCenterPx=${legacySample ? fmtArr(legacySample.centerPx, 3) : 'none'}  packedSampleCenterPx=${packedSample ? fmtArr(packedSample.centerPx, 3) : 'none'}`,
    `legacySampleRadiusPx=${legacySample ? fmtNum(legacySample.radiusPx, 3) : 'none'}  packedSampleRadiusPx=${packedSample ? fmtNum(packedSample.radiusPx, 3) : 'none'}`,
    `legacySampleColorAlpha=${legacySample ? fmtArr(legacySample.colorAlpha, 4) : 'none'}`,
    `packedSampleColorAlpha=${packedSample ? fmtArr(packedSample.colorAlpha, 4) : 'none'}`,
    `legacySampleConic=${legacySample ? fmtArr(legacySample.conic, 4) : 'none'}`,
    `packedSampleConic=${packedSample ? fmtArr(packedSample.conic, 4) : 'none'}`,
    `packedSampleReserved=${packedSample ? fmtNum(packedSample.reserved, 4) : 'none'}`,
    `packedSampleMisc=${packedSample ? fmtArr(packedSample.misc, 2) : 'none'}`
  ];
}

export function buildGpuScreenExecutionLines(gpuScreenExecutionSummary) {
  if (!gpuScreenExecutionSummary) return [];
  return [
    `gpuScreenDraw=${!!gpuScreenExecutionSummary.gpuScreenDraw}`,
    `gpuScreenReady=${!!gpuScreenExecutionSummary.gpuScreenReady}`,
    `gpuScreenReason=${gpuScreenExecutionSummary.gpuScreenReason ?? 'unknown'}`,
    `gpuScreenDrawCount=${gpuScreenExecutionSummary.gpuScreenDrawCount ?? 0}`,
    `gpuScreenActualPath=${gpuScreenExecutionSummary.gpuScreenActualPath ?? 'gpu-screen'}`,
    `gpuScreenSourcePath=${gpuScreenExecutionSummary.gpuScreenSourcePath ?? 'none'}`,
    `gpuScreenReferencePath=${gpuScreenExecutionSummary.gpuScreenReferencePath ?? 'packed-cpu'}`
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
  gpuScreenSummary = null,
  gpuScreenSourceSpace = null,
  gpuScreenComparisonSummary = null,
  drawPathSelection = null,
  visible = null,
  packedScreenSpace = null,
  gpuScreenExecutionSummary = null,
  legacySample = null,
  packedSample = null
} = {}) {
  const lines = [];
  const transformBatchSummary =
    gpuScreenComparisonSummary?.transformBatchSummary ??
    gpuScreenSourceSpace?.transformSummary?.transformBatchSummary ??
    packedScreenSpace?.transformSummary?.transformBatchSummary ??
    packedScreenSpace?.summary?.transformBatchSummary ??
    null;
  const transformLifecycleSummary =
    gpuScreenSourceSpace?.transformSummary ??
    packedScreenSpace?.transformSummary ??
    null;

  const resolvedLegacySample = legacySample ?? buildLegacySample(visible);
  const resolvedPackedSample = packedSample ?? buildPackedSample(packedScreenSpace);

  lines.push(...buildConfigLines(buildConfig));
  lines.push(...buildTimingLines(buildStats, drawStats, gpuScreenSummary));
  lines.push(...buildModeLines(mode, focusTileIds, focusTileRects));
  lines.push(...buildUiLines(ui));
  lines.push(...buildGpuScreenStateLines(gpuScreenSummary));
  lines.push(...buildGpuScreenComparisonLines(gpuScreenComparisonSummary));
  lines.push(...buildTransformBatchLines(transformBatchSummary));
  lines.push(...buildTransformLifecycleLines(transformLifecycleSummary));
  lines.push(...buildPackedLines(buildStats, drawPathSelection, drawStats));
  lines.push(...buildSampleLines(resolvedLegacySample, resolvedPackedSample));
  lines.push(...buildGpuScreenExecutionLines(gpuScreenExecutionSummary));

  return lines;
}
