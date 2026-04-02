function clampPositiveInt(v, fallback) {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(1, Math.round(n));
}

function clampPositiveFloat(v, fallback, minValue = 0.01, maxValue = Infinity) {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(maxValue, Math.max(minValue, n));
}

export function getBaseQualityConfig(ui) {
  return {
    stride: clampPositiveInt(ui?.strideSlider?.value, 1),
    maxVisible: clampPositiveInt(ui?.maxVisibleSlider?.value, 200000),
    renderScale: clampPositiveFloat(ui?.renderScaleSlider?.value, 1.0, 0.05, 1.0)
  };
}

function getPlaybackOverrideValues(ui, base) {
  return {
    stride: clampPositiveInt(ui?.playbackStrideInput?.value, Math.max(base.stride, 32)),
    maxVisible: clampPositiveInt(ui?.playbackMaxVisibleInput?.value, Math.min(base.maxVisible, 30000)),
    renderScale: clampPositiveFloat(ui?.playbackRenderScaleInput?.value, Math.min(base.renderScale, 0.50), 0.05, 1.0)
  };
}

function getInteractionOverrideValues(ui, base) {
  return {
    stride: clampPositiveInt(ui?.interactionStrideInput?.value, Math.max(base.stride, 64)),
    maxVisible: clampPositiveInt(ui?.interactionMaxVisibleInput?.value, Math.min(base.maxVisible, 10000)),
    renderScale: clampPositiveFloat(ui?.interactionRenderScaleInput?.value, Math.min(base.renderScale, 0.50), 0.05, 1.0)
  };
}

export function getGpuPlaybackOverride(ui, isPlaying) {
  const base = getBaseQualityConfig(ui);
  const enabled = !!ui?.usePlaybackOverrideCheck?.checked;
  const active = !!isPlaying && enabled;

  if (!active) {
    return {
      active: false,
      enabled,
      reason: 'none',
      stride: base.stride,
      maxVisible: base.maxVisible,
      renderScale: base.renderScale
    };
  }

  const values = getPlaybackOverrideValues(ui, base);

  return {
    active: true,
    enabled: true,
    reason: 'playback',
    stride: Math.max(base.stride, values.stride),
    maxVisible: Math.min(base.maxVisible, values.maxVisible),
    renderScale: Math.min(base.renderScale, values.renderScale)
  };
}

export function getGpuInteractionOverride(ui, interactionState) {
  const base = getBaseQualityConfig(ui);
  const enabled = !!ui?.useInteractionOverrideCheck?.checked;
  const dragActive = !!(interactionState && interactionState.dragActive);
  const active = dragActive && enabled;

  if (!active) {
    return {
      active: false,
      enabled,
      reason: 'none',
      stride: base.stride,
      maxVisible: base.maxVisible,
      renderScale: base.renderScale
    };
  }

  const values = getInteractionOverrideValues(ui, base);

  return {
    active: true,
    enabled: true,
    reason: 'interaction',
    stride: Math.max(base.stride, values.stride),
    maxVisible: Math.min(base.maxVisible, values.maxVisible),
    renderScale: Math.min(base.renderScale, values.renderScale)
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
