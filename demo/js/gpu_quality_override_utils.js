function clampPositiveInt(v, fallback) {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(1, Math.round(n));
}

function clampPositiveFloat(v, fallback, minValue = 0.01) {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(minValue, n);
}

export function getBaseQualityConfig(ui) {
  return {
    stride: clampPositiveInt(ui?.strideSlider?.value, 1),
    maxVisible: clampPositiveInt(ui?.maxVisibleSlider?.value, 200000),
    renderScale: clampPositiveFloat(ui?.renderScaleSlider?.value, 1.0, 0.05)
  };
}

export function getGpuInteractionOverride(ui, interactionState) {
  const base = getBaseQualityConfig(ui);
  const active = !!(interactionState && interactionState.dragActive);

  if (!active) {
    return {
      active: false,
      reason: 'none',
      stride: base.stride,
      maxVisible: base.maxVisible,
      renderScale: base.renderScale
    };
  }

  // Step18:
  // CPU 側最適化の評価用として、ドラッグ中はかなり強めに品質を落とす。
  // ただし最終目標は stride=1 のまま動かすことなので、これは暫定的な測定用 override。
  return {
    active: true,
    reason: 'interaction',
    stride: Math.max(base.stride, 64),
    maxVisible: Math.min(base.maxVisible, 10000),
    renderScale: Math.min(base.renderScale, 0.50)
  };
}

export function getGpuPlaybackOverride(ui, isPlaying) {
  const base = getBaseQualityConfig(ui);
  const active = !!isPlaying;

  if (!active) {
    return {
      active: false,
      reason: 'none',
      stride: base.stride,
      maxVisible: base.maxVisible,
      renderScale: base.renderScale
    };
  }

  // Step18:
  // 再生時の実効化。interaction よりは少し緩めだが、通常時よりは明確に軽量化する。
  // ここで再生体感と debug 値を比較し、CPU 側でどこまで粘れるかを確認する。
  return {
    active: true,
    reason: 'playback',
    stride: Math.max(base.stride, 32),
    maxVisible: Math.min(base.maxVisible, 30000),
    renderScale: Math.min(base.renderScale, 0.50)
  };
}

export function mergeGpuQualityOverrides(baseConfig, overrides = []) {
  const result = {
    ...baseConfig,
    qualityOverrideActive: false,
    qualityOverrideReasons: [],
    interactionActive: false,
    playbackActive: false
  };

  for (const ov of overrides) {
    if (!ov || !ov.active) continue;

    result.qualityOverrideActive = true;

    if (ov.reason && ov.reason !== 'none') {
      result.qualityOverrideReasons.push(ov.reason);
    }

    if (ov.reason === 'interaction') result.interactionActive = true;
    if (ov.reason === 'playback') result.playbackActive = true;

    // stride は「大きいほど粗い」ので最大値を採用
    if (Number.isFinite(ov.stride)) {
      result.stride = Math.max(result.stride, ov.stride);
    }

    // maxVisible は小さいほど軽いので最小値を採用
    if (Number.isFinite(ov.maxVisible)) {
      result.maxVisible = Math.min(result.maxVisible, ov.maxVisible);
    }

    // renderScale は小さいほど軽いので最小値を採用
    if (Number.isFinite(ov.renderScale)) {
      result.renderScale = Math.min(result.renderScale, ov.renderScale);
    }
  }

  result.qualityOverrideReason =
    result.qualityOverrideReasons.length > 0
      ? result.qualityOverrideReasons.join('+')
      : 'none';

  return result;
}

export function buildEffectiveGpuQualityConfig({
  ui,
  baseConfig,
  interactionState,
  isPlaying
}) {
  const base = baseConfig ? { ...baseConfig } : getBaseQualityConfig(ui);
  const interactionOverride = getGpuInteractionOverride(ui, interactionState);
  const playbackOverride = getGpuPlaybackOverride(ui, isPlaying);

  const effective = mergeGpuQualityOverrides(base, [
    playbackOverride,
    interactionOverride
  ]);

  return {
    baseConfig: base,
    interactionOverride,
    playbackOverride,
    effectiveConfig: effective
  };
}
