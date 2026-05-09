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

export async function downloadCanvasPng(sourceCanvas, fileName = 'gpu-viewer-current-canvas.png', options = {}) {
  const normalizedFileName = sanitizePngFileName(fileName);
  const blob = await canvasToPngBlob(sourceCanvas);
  if (options.download !== false) {
    downloadBlob(blob, normalizedFileName);
  }
  return { blob, fileName: normalizedFileName };
}
