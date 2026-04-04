// Step25:
// debug 情報の重複を減らし、timing / metrics を見やすく整理する。
// ここでは renderer 側ですでに表示している packed/draw/sample の重複行は出さない。
// 主に以下だけを返す。
// - build config の補助情報
// - timing 系
// - tile mode / UI 状態
//
// 注意:
// packed 詳細や sample 比較は renderer 側の packedLines / sampleLines に任せる。

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

function buildTimingLines(buildStats, drawStats) {
  const lines = [];
  pushLine(lines, 'visibleBuildMs', fmtNum(buildStats?.visibleBuildMs, 3));
  pushLine(lines, 'candidateBuildMs', fmtNum(buildStats?.candidateBuildMs, 3));
  pushLine(lines, 'screenSpaceBuildMs', fmtNum(buildStats?.screenSpaceBuildMs, 3));
  pushLine(lines, 'drawFraction', fmtNum(drawStats?.drawFraction, 6));
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
    const rectText = focusTileRects.map((item) => {
      const tileId = isFiniteNumber(item?.tileId) ? String(item.tileId | 0) : '-1';
      const rect = fmtArray(item?.rect, 0) ?? 'none';
      return `${tileId}:${rect}`;
    }).join(' ');
    pushLine(lines, 'focusTileRects', rectText);
  }
  return lines;
}

function buildUiLines(ui) {
  if (!ui) return [];
  const lines = [];
  pushLine(lines, 'bgGraySlider', ui.bgGraySlider ? String(ui.bgGraySlider.value) : null);
  pushLine(lines, 'drawPathUiValue', ui.drawPathSelect ? String(ui.drawPathSelect.value) : null);
  pushLine(lines, 'usePackedVisiblePathUi', ui.usePackedVisiblePathCheck ? fmtBool(!!ui.usePackedVisiblePathCheck.checked) : null);
  return lines;
}

export function buildGpuDebugExtraLines({
  buildConfig = null,
  buildStats = null,
  drawStats = null,
  mode = null,
  focusTileIds = [],
  focusTileRects = [],
  ui = null
} = {}) {
  const lines = [];
  lines.push(...buildConfigLines(buildConfig));
  lines.push(...buildTimingLines(buildStats, drawStats));
  lines.push(...buildModeLines(mode, focusTileIds, focusTileRects));
  lines.push(...buildUiLines(ui));
  return lines;
}
