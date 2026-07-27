function normalizeDownloadBaseName(name, fallbackName) {
  const trimmed = typeof name === 'string' ? name.trim() : '';
  const baseName = trimmed || fallbackName;
  return baseName.replace(/[^a-zA-Z0-9._-]+/g, '-').replace(/-+/g, '-');
}

export function sanitizeFileNameWithExtension(name, extension, fallbackName = `debug${extension}`) {
  const ext = typeof extension === 'string' && extension.startsWith('.') ? extension : `.${extension}`;
  const normalized = normalizeDownloadBaseName(name, fallbackName);
  return normalized.toLowerCase().endsWith(ext.toLowerCase()) ? normalized : `${normalized}${ext}`;
}

export function sanitizeJsonFileName(name, fallbackName = 'gpu-viewer-debug.json') {
  return sanitizeFileNameWithExtension(name, '.json', fallbackName);
}

export function sanitizePngFileName(name, fallbackName = 'gpu-viewer-current-canvas.png') {
  return sanitizeFileNameWithExtension(name, '.png', fallbackName);
}

export function downloadBlob(blob, fileName) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

export function downloadJsonDebug(data, fileName = 'gpu-viewer-debug.json') {
  const normalizedFileName = sanitizeJsonFileName(fileName);
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  downloadBlob(blob, normalizedFileName);
  return { fileName: normalizedFileName, byteLength: blob.size };
}

export async function canvasToPngBlob(sourceCanvas) {
  return await new Promise((resolve, reject) => {
    sourceCanvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error('PNG capture failed: canvas toBlob returned null'));
        return;
      }
      resolve(blob);
    }, 'image/png');
  });
}

export function buildOpaqueWebGpuPngAlphaNormalizationEvidence({
  preNormalizationPixelEvidence = {},
  postNormalizationPixelEvidence = {},
  preNormalizationSourceIdentity = null,
  postNormalizationBlobIdentity = null,
  width = null,
  height = null
} = {}) {
  const pixelCount = Number.isFinite(Number(width)) && Number.isFinite(Number(height))
    ? Math.max(1, Math.round(Number(width)) * Math.round(Number(height)))
    : null;
  const rgbInvariant =
    preNormalizationPixelEvidence?.rgbHash != null &&
    postNormalizationPixelEvidence?.rgbHash != null &&
    preNormalizationPixelEvidence.rgbHash ===
      postNormalizationPixelEvidence.rgbHash &&
    preNormalizationPixelEvidence.rgbNonzeroPixelCount ===
      postNormalizationPixelEvidence.rgbNonzeroPixelCount &&
    preNormalizationPixelEvidence.rgbMax ===
      postNormalizationPixelEvidence.rgbMax;
  const alphaOnlyChanged =
    rgbInvariant &&
    postNormalizationPixelEvidence?.alphaMin === 255 &&
    postNormalizationPixelEvidence?.alphaMax === 255 &&
    (
      pixelCount === null ||
      postNormalizationPixelEvidence?.alphaOpaquePixelCount === pixelCount
    );
  return {
    schemaVersion: 'phase3-webgpu-production-png-alpha-normalization-v1',
    scope: 'opaque-webgpu-production-tile-compositor-png-capture-only',
    appliedToOpaqueWebGpuProductionPngCaptureOnly: true,
    genericTransparentPngCaptureUnaffected: true,
    genericCaptureFunction:
      'downloadCanvasPng/canvasToPngBlob-preserves-source-alpha-by-default',
    normalizedChannel: 'alpha-only',
    normalizedAlphaValue: 255,
    preNormalizationSourceIdentity,
    postNormalizationBlobIdentity,
    preNormalizationPixelEvidence,
    postNormalizationPixelEvidence,
    rgbInvariant,
    alphaOnlyChanged,
    viewerOpaquePresentationEquivalent: true,
    productionRenderingChanged: false,
    canvasPresentationChanged: false
  };
}

async function sha256Blob(blob) {
  if (!blob || !globalThis.crypto?.subtle) return null;
  const digest = await globalThis.crypto.subtle.digest(
    'SHA-256',
    await blob.arrayBuffer()
  );
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

async function inspectEncodedPngBlobPixels(blob) {
  const evidence = {
    schemaVersion: 'phase3-encoded-png-blob-pixel-evidence-v1',
    decodeSupported: typeof createImageBitmap === 'function',
    decodeCompleted: false,
    width: null,
    height: null,
    rgbNonzeroPixelCount: null,
    rgbNonblackRatio: null,
    rgbMax: null,
    rgbHash: null,
    alphaHash: null,
    alphaNonzeroPixelCount: null,
    alphaZeroPixelCount: null,
    alphaOpaquePixelCount: null,
    alphaMin: null,
    alphaMax: null,
    nonblackBoundingBox: null,
    pixelClassification: 'unknown',
    decodeError: null
  };
  if (!evidence.decodeSupported) {
    evidence.decodeError = 'createImageBitmap-unavailable';
    return evidence;
  }
  let bitmap = null;
  try {
    bitmap = await createImageBitmap(blob);
    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('2d-context-unavailable-for-png-blob-decode');
    context.drawImage(bitmap, 0, 0);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let rgbNonzeroPixelCount = 0;
    let rgbMax = 0;
    let alphaNonzeroPixelCount = 0;
    let alphaZeroPixelCount = 0;
    let alphaOpaquePixelCount = 0;
    let alphaMin = 255;
    let alphaMax = 0;
    let rgbHash = 2166136261;
    let alphaHash = 2166136261;
    let minX = null;
    let minY = null;
    let maxX = null;
    let maxY = null;
    for (let y = 0; y < canvas.height; y += 1) {
      for (let x = 0; x < canvas.width; x += 1) {
        const offset = (y * canvas.width + x) * 4;
        const pixelRgbMax = Math.max(
          pixels[offset] ?? 0,
          pixels[offset + 1] ?? 0,
          pixels[offset + 2] ?? 0
        );
        const alpha = pixels[offset + 3] ?? 0;
        rgbMax = Math.max(rgbMax, pixelRgbMax);
        alphaMin = Math.min(alphaMin, alpha);
        alphaMax = Math.max(alphaMax, alpha);
        alphaNonzeroPixelCount += alpha > 0 ? 1 : 0;
        alphaZeroPixelCount += alpha === 0 ? 1 : 0;
        alphaOpaquePixelCount += alpha === 255 ? 1 : 0;
        rgbHash ^= pixels[offset] ?? 0;
        rgbHash = Math.imul(rgbHash, 16777619) >>> 0;
        rgbHash ^= pixels[offset + 1] ?? 0;
        rgbHash = Math.imul(rgbHash, 16777619) >>> 0;
        rgbHash ^= pixels[offset + 2] ?? 0;
        rgbHash = Math.imul(rgbHash, 16777619) >>> 0;
        alphaHash ^= alpha;
        alphaHash = Math.imul(alphaHash, 16777619) >>> 0;
        if (pixelRgbMax > 0) {
          rgbNonzeroPixelCount += 1;
          minX = minX === null ? x : Math.min(minX, x);
          minY = minY === null ? y : Math.min(minY, y);
          maxX = maxX === null ? x : Math.max(maxX, x);
          maxY = maxY === null ? y : Math.max(maxY, y);
        }
      }
    }
    const pixelCount = Math.max(1, canvas.width * canvas.height);
    Object.assign(evidence, {
      decodeCompleted: true,
      width: canvas.width,
      height: canvas.height,
      rgbNonzeroPixelCount,
      rgbNonblackRatio: rgbNonzeroPixelCount / pixelCount,
      rgbMax,
      rgbHash: rgbHash.toString(16).padStart(8, '0'),
      alphaHash: alphaHash.toString(16).padStart(8, '0'),
      alphaNonzeroPixelCount,
      alphaZeroPixelCount,
      alphaOpaquePixelCount,
      alphaMin,
      alphaMax,
      nonblackBoundingBox:
        rgbNonzeroPixelCount > 0 ? [minX, minY, maxX, maxY] : null,
      pixelClassification: rgbNonzeroPixelCount > 0 ? 'nonblank' : 'black'
    });
  } catch (error) {
    evidence.decodeError = error?.message ?? String(error);
  } finally {
    if (bitmap && typeof bitmap.close === 'function') bitmap.close();
  }
  return evidence;
}

export async function downloadCanvasPng(sourceCanvas, fileName = 'gpu-viewer-current-canvas.png', options = {}) {
  const normalizedFileName = sanitizePngFileName(fileName);
  const blob = await canvasToPngBlob(sourceCanvas);
  const captureBlobIdentity = {
    schemaVersion: 'phase3-capture-png-blob-identity-v1',
    fileName: normalizedFileName,
    mimeType: blob.type || 'image/png',
    sizeBytes: blob.size,
    sha256: await sha256Blob(blob)
  };
  const encodedPngPixelEvidence = await inspectEncodedPngBlobPixels(blob);
  if (options.download !== false) {
    downloadBlob(blob, normalizedFileName);
  }
  return {
    blob,
    fileName: normalizedFileName,
    captureBlobIdentity,
    encodedPngPixelEvidence
  };
}
