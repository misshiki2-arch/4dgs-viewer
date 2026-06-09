export const WEBGPU_BOUNDED_COLOR_SAMPLE_CONTRACT_VERSION =
  'phase3-step53-bounded-color-sample-contract-v2';

export const DEFAULT_MAX_BOUNDED_COLOR_SAMPLES = 8;

export function clamp01(value, fallback = 0) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(1, Math.max(0, n));
}

export function normalizeBoundedColorArray(value) {
  if (!Array.isArray(value) && !(value instanceof Float32Array)) return null;
  if (value.length < 3) return null;
  return {
    r: clamp01(value[0], 0),
    g: clamp01(value[1], 0),
    b: clamp01(value[2], 0),
    a: clamp01(value.length >= 4 ? value[3] : 1, 1)
  };
}

export function normalizeBoundedColorObject(value) {
  if (!value || typeof value !== 'object') return null;
  const r = value.r ?? value.red;
  const g = value.g ?? value.green;
  const b = value.b ?? value.blue;
  if (
    !Number.isFinite(Number(r)) ||
    !Number.isFinite(Number(g)) ||
    !Number.isFinite(Number(b))
  ) {
    return null;
  }
  return {
    r: clamp01(r, 0),
    g: clamp01(g, 0),
    b: clamp01(b, 0),
    a: clamp01(value.a ?? value.alpha ?? 1, 1)
  };
}

export function normalizeBoundedColorValue(value) {
  return (
    normalizeBoundedColorArray(value) ??
    normalizeBoundedColorObject(value)
  );
}

export function colorToRgbaFloatArray(color) {
  const normalized = normalizeBoundedColorValue(color);
  if (!normalized) return [0, 0, 0, 0];
  return [normalized.r, normalized.g, normalized.b, normalized.a];
}

export function colorFromBoundedSample(sample) {
  return (
    normalizeBoundedColorValue(sample?.actual?.rgbaFloat) ??
    normalizeBoundedColorValue(sample?.expected?.rgbaFloat) ??
    normalizeBoundedColorValue(sample?.actual?.resolvedColor) ??
    normalizeBoundedColorValue(sample?.expected?.resolvedColor) ??
    normalizeBoundedColorValue(sample?.actual?.resolvedRgb) ??
    normalizeBoundedColorValue(sample?.expected?.resolvedRgb) ??
    normalizeBoundedColorValue(sample?.actual?.colorAlpha) ??
    normalizeBoundedColorValue(sample?.expected?.colorAlpha) ??
    normalizeBoundedColorValue(sample?.colorAlpha) ??
    normalizeBoundedColorValue(sample?.upstreamSample?.actual?.rgbaFloat) ??
    normalizeBoundedColorValue(sample?.upstreamSample?.expected?.rgbaFloat) ??
    normalizeBoundedColorValue(sample?.upstreamSample?.actual?.resolvedColor) ??
    normalizeBoundedColorValue(sample?.upstreamSample?.expected?.resolvedColor) ??
    normalizeBoundedColorValue(sample?.upstreamSample?.actual?.resolvedRgb) ??
    normalizeBoundedColorValue(sample?.upstreamSample?.expected?.resolvedRgb) ??
    normalizeBoundedColorValue(sample?.upstreamSample?.actual?.colorAlpha) ??
    normalizeBoundedColorValue(sample?.upstreamSample?.expected?.colorAlpha) ??
    normalizeBoundedColorValue(sample?.payload?.colorAlpha) ??
    normalizeBoundedColorValue(sample?.referenceAssisted?.colorAlpha) ??
    normalizeBoundedColorValue(sample?.renderPayload?.colorAlpha)
  );
}

export function samplePxFromBoundedSample(sample, fallbackIndex = 0) {
  const px =
    sample?.pixel ??
    sample?.texturePixel ??
    sample?.samplePx ??
    sample?.actual?.pixel ??
    sample?.expected?.pixel ??
    sample?.upstreamSample?.pixel ??
    sample?.upstreamSample?.texturePixel ??
    sample?.upstreamSample?.samplePx ??
    sample?.centerPx;
  if (Array.isArray(px) && px.length >= 2) {
    return { x: Number(px[0]), y: Number(px[1]) };
  }
  if (px && typeof px === 'object') {
    return { x: Number(px.x ?? px[0]), y: Number(px.y ?? px[1]) };
  }
  const offset = fallbackIndex * 24;
  return { x: 32 + offset, y: 32 + offset };
}

export function normalizeBoundedColorSamples({
  list,
  source,
  colorSource,
  maxSamples = DEFAULT_MAX_BOUNDED_COLOR_SAMPLES,
  preserveSampleSource = false,
  includeUpstreamSample = true,
  fallbackIndexOffset = 0
} = {}) {
  const samples = [];
  if (!Array.isArray(list)) return samples;
  for (const sample of list) {
    if (samples.length >= maxSamples) break;
    const color = colorFromBoundedSample(sample);
    if (!color) continue;
    const normalized = {
      source: preserveSampleSource ? sample?.source ?? source : source,
      colorSource: sample?.colorSource ?? colorSource,
      recordIndex: sample?.recordIndex ?? sample?.anchorRecordIndex ?? null,
      sampleKind: sample?.sampleKind ?? null,
      srcIndex: sample?.srcIndex ?? null,
      valid: sample?.valid ?? null,
      samplePx: samplePxFromBoundedSample(
        sample,
        fallbackIndexOffset + samples.length
      ),
      colorAlpha: color
    };
    if (includeUpstreamSample) normalized.upstreamSample = sample;
    samples.push(normalized);
  }
  return samples;
}

export function summarizeBoundedColorSampleContract({
  rawSampleCount = 0,
  presentableSamples = [],
  selectionMode = null,
  fallbackAllowed = false,
  fallbackSuppressedBySelectorSamples = false
} = {}) {
  return {
    contractVersion: WEBGPU_BOUNDED_COLOR_SAMPLE_CONTRACT_VERSION,
    rawSampleCount,
    presentableSampleCount: Array.isArray(presentableSamples)
      ? presentableSamples.length
      : 0,
    selectionMode,
    fallbackAllowed,
    fallbackSuppressedBySelectorSamples,
    sampleSources: Array.isArray(presentableSamples)
      ? [...new Set(presentableSamples.map((sample) => sample.source))]
      : [],
    firstValidationFailures:
      rawSampleCount > 0 && (!Array.isArray(presentableSamples) || presentableSamples.length <= 0)
        ? [
            {
              stage: 'bounded-color-sample-contract',
              reason:
                'raw bounded color samples are present but none satisfy the presentable color/pixel contract'
            }
          ]
        : []
  };
}
