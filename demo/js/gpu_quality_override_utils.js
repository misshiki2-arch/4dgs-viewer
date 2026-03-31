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

  return {
    active: true,
    reason: 'playback',
    stride: Math.max(base.stride, 8),
    maxVisible: Math.min(base.maxVisible, 50000),
    renderScale: Math.min(base.renderScale, 0.75)
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

    if (Number.isFinite(ov.stride)) {
      result.stride = Math.max(result.stride, ov.stride);
    }
    if (Number.isFinite(ov.maxVisible)) {
      result.maxVisible = Math.min(result.maxVisible, ov.maxVisible);
    }
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
