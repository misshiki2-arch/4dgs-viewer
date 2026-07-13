const PRESENTATION_CAPTURE_ORIENTATION_CONTRACT = Object.freeze({
  schemaVersion: 'phase3-webgpu-presentation-capture-orientation-v1',
  productionTextureOrigin: 'texture-memory-top-left',
  productionTextureYAxisDirection: 'down',
  presentationUvTransform: 'fullscreen-clip-y-up-to-texture-y-down',
  presentationVerticalFlipApplied: true,
  captureReadbackRowOrder: 'copyTextureToBuffer-texture-memory-top-to-bottom',
  pngEncoderRowOrder: 'canvas-image-data-top-to-bottom',
  canonicalPresentationOrientation: 'viewer-currentTexture-presented-orientation',
  savedPngOrientation: 'canonical-presentation-orientation',
  captureVerticalFlipApplied: true,
  orientationMismatchClassification: 'none'
});

export function getWebGpuPresentationCaptureOrientationContract() {
  return { ...PRESENTATION_CAPTURE_ORIENTATION_CONTRACT };
}

export function buildWebGpuPresentationCaptureOrientationEvidence({
  captureVerticalFlipApplied =
    PRESENTATION_CAPTURE_ORIENTATION_CONTRACT.captureVerticalFlipApplied,
  savedPngMatchesRawProductionOutput = false,
  savedPngMatchesPresentedOutput = false
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
