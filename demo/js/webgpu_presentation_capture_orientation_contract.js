const PRESENTATION_CAPTURE_ORIENTATION_CONTRACT = Object.freeze({
  schemaVersion: 'phase3-webgpu-presentation-capture-orientation-v1',
  productionTextureOrigin: 'texture-memory-top-left',
  productionTextureYAxisDirection: 'down',
  presentationUvTransform: 'fullscreen-framebuffer-top-left-to-texture-top-left',
  presentationVerticalFlipApplied: false,
  captureReadbackRowOrder: 'copyTextureToBuffer-texture-memory-top-to-bottom',
  pngEncoderRowOrder: 'canvas-image-data-top-to-bottom',
  canonicalPresentationOrientation: 'production-texture-top-left-y-down',
  savedPngOrientation: 'production-texture-top-left-y-down',
  captureVerticalFlipApplied: false,
  orientationMismatchClassification: 'none'
});

export function getWebGpuPresentationCaptureOrientationContract() {
  return { ...PRESENTATION_CAPTURE_ORIENTATION_CONTRACT };
}

export function buildWebGpuPresentationCaptureOrientationEvidence({
  captureVerticalFlipApplied =
    PRESENTATION_CAPTURE_ORIENTATION_CONTRACT.captureVerticalFlipApplied,
  savedPngMatchesRawProductionOutput = captureVerticalFlipApplied !== true,
  savedPngMatchesPresentedOutput =
    captureVerticalFlipApplied ===
    PRESENTATION_CAPTURE_ORIENTATION_CONTRACT.presentationVerticalFlipApplied
} = {}) {
  const base = getWebGpuPresentationCaptureOrientationContract();
  const captureMatchesCanonicalPresentationOrientation =
    captureVerticalFlipApplied === base.presentationVerticalFlipApplied;
  const orientationConsistencyKnown =
    captureMatchesCanonicalPresentationOrientation &&
    savedPngMatchesPresentedOutput === true;
  const orientationMismatchDetected =
    captureMatchesCanonicalPresentationOrientation !== true;
  return {
    ...base,
    captureVerticalFlipApplied,
    savedPngMatchesRawProductionOutput,
    savedPngMatchesPresentedOutput,
    captureMatchesCanonicalPresentationOrientation,
    orientationConsistencyKnown,
    orientationMismatchDetected,
    orientationMismatchClassification: orientationMismatchDetected
      ? 'capture-presentation-y-transform-mismatch'
      : 'none'
  };
}

function mapProductionTextureRowToOutputRow(
  sourceRow,
  height,
  verticalFlipApplied
) {
  if (!Number.isInteger(height) || height <= 0) {
    throw new RangeError('orientation-row-height-must-be-a-positive-integer');
  }
  if (!Number.isInteger(sourceRow) || sourceRow < 0 || sourceRow >= height) {
    throw new RangeError('orientation-source-row-out-of-bounds');
  }
  return verticalFlipApplied ? height - 1 - sourceRow : sourceRow;
}

export function mapWebGpuProductionTextureRowToPresentedRow(sourceRow, height) {
  return mapProductionTextureRowToOutputRow(
    sourceRow,
    height,
    PRESENTATION_CAPTURE_ORIENTATION_CONTRACT.presentationVerticalFlipApplied
  );
}

export function mapWebGpuProductionTextureRowToPngRow(sourceRow, height) {
  return mapProductionTextureRowToOutputRow(
    sourceRow,
    height,
    PRESENTATION_CAPTURE_ORIENTATION_CONTRACT.captureVerticalFlipApplied
  );
}

export function mapWebGpuPresentationClipYToProductionTextureV(clipY) {
  const normalizedClipY = clipY * 0.5 + 0.5;
  return PRESENTATION_CAPTURE_ORIENTATION_CONTRACT.presentationVerticalFlipApplied
    ? normalizedClipY
    : 1 - normalizedClipY;
}

export function buildWebGpuProductionTexturePresentationUvWgsl(
  positionExpression = 'pos'
) {
  const textureVExpression =
    PRESENTATION_CAPTURE_ORIENTATION_CONTRACT.presentationVerticalFlipApplied
      ? `${positionExpression}.y * 0.5 + 0.5`
      : `0.5 - ${positionExpression}.y * 0.5`;
  return `vec2f(${positionExpression}.x * 0.5 + 0.5, ${textureVExpression})`;
}

export function summarizeWebGpuPresentationFrameIdentity({
  generation = null,
  frameHash = null,
  datasetCameraLabel = null,
  datasetFrameNumber = null,
  datasetTime = null,
  referenceCameraLabel = null,
  outputWidth = null,
  outputHeight = null
} = {}) {
  return {
    generation,
    frameHash,
    datasetCameraLabel,
    datasetFrameNumber,
    datasetTime,
    referenceCameraLabel,
    outputWidth,
    outputHeight
  };
}

export function compareWebGpuPresentationFrameIdentity(a = {}, b = {}, options = {}) {
  const defaultKeys = [
    'generation',
    'datasetCameraLabel',
    'datasetFrameNumber',
    'datasetTime',
    'referenceCameraLabel',
    'outputWidth',
    'outputHeight'
  ];
  const keys = Array.isArray(options.requiredKeys) && options.requiredKeys.length > 0
    ? options.requiredKeys
    : defaultKeys;
  const mismatchedKeys = keys.filter((key) => {
    const left = a?.[key] ?? null;
    const right = b?.[key] ?? null;
    return left !== null && right !== null && left !== right;
  });
  const missingKeys = keys.filter((key) => a?.[key] == null || b?.[key] == null);
  return {
    matches: mismatchedKeys.length === 0 && missingKeys.length === 0,
    mismatchedKeys,
    missingKeys,
    requiredKeys: keys
  };
}
