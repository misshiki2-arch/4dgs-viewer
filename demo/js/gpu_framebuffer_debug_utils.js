import { downloadCanvasPng } from './debug_download_utils.js';

function normalizeFramebufferSampleOptions(options = {}) {
  const gridWidth = Number.isFinite(Number(options.gridWidth))
    ? Math.max(1, Math.min(512, Number(options.gridWidth) | 0))
    : 64;
  const gridHeight = Number.isFinite(Number(options.gridHeight))
    ? Math.max(1, Math.min(512, Number(options.gridHeight) | 0))
    : 36;
  const nonBlackThreshold = Number.isFinite(Number(options.nonBlackThreshold))
    ? Math.max(0, Number(options.nonBlackThreshold))
    : 5;
  const brightestSampleCount = Number.isFinite(Number(options.brightestSampleCount))
    ? Math.max(0, Math.min(64, Number(options.brightestSampleCount) | 0))
    : 12;
  return { gridWidth, gridHeight, nonBlackThreshold, brightestSampleCount };
}

function sampleFramebufferPixel(gl, x, y) {
  const pixel = new Uint8Array(4);
  gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
  return [pixel[0], pixel[1], pixel[2], pixel[3]];
}

function pushBrightestSample(samples, sample, maxCount) {
  if (maxCount <= 0) return;
  samples.push(sample);
  samples.sort((a, b) => b.rgbSum - a.rgbSum);
  if (samples.length > maxCount) samples.length = maxCount;
}

export function sampleCurrentFramebufferStats(gl, canvas, options = {}, canvasSummary = {}) {
  if (!gl) {
    return {
      available: false,
      reason: 'webgl-not-initialized',
      ...canvasSummary
    };
  }

  const opts = normalizeFramebufferSampleOptions(options);
  const width = gl.drawingBufferWidth | 0;
  const height = gl.drawingBufferHeight | 0;
  if (width <= 0 || height <= 0) {
    return {
      available: false,
      reason: 'empty-framebuffer',
      ...canvasSummary
    };
  }

  gl.finish();

  const sampleCount = opts.gridWidth * opts.gridHeight;
  const sumRgb = [0, 0, 0];
  const minRgb = [255, 255, 255];
  const maxRgb = [0, 0, 0];
  let alphaSum = 0;
  let maxAlpha = 0;
  let nonBlackCount = 0;
  let nonTransparentCount = 0;
  let representativeNonBlackPixel = null;
  const brightestSamples = [];

  for (let gy = 0; gy < opts.gridHeight; gy++) {
    const y = opts.gridHeight === 1
      ? Math.floor((height - 1) * 0.5)
      : Math.round((gy / (opts.gridHeight - 1)) * (height - 1));
    for (let gx = 0; gx < opts.gridWidth; gx++) {
      const x = opts.gridWidth === 1
        ? Math.floor((width - 1) * 0.5)
        : Math.round((gx / (opts.gridWidth - 1)) * (width - 1));
      const rgba8 = sampleFramebufferPixel(gl, x, y);
      const rgbSum = rgba8[0] + rgba8[1] + rgba8[2];

      for (let axis = 0; axis < 3; axis++) {
        sumRgb[axis] += rgba8[axis];
        minRgb[axis] = Math.min(minRgb[axis], rgba8[axis]);
        maxRgb[axis] = Math.max(maxRgb[axis], rgba8[axis]);
      }
      alphaSum += rgba8[3];
      maxAlpha = Math.max(maxAlpha, rgba8[3]);

      if (rgbSum > opts.nonBlackThreshold) {
        nonBlackCount++;
        if (!representativeNonBlackPixel) {
          representativeNonBlackPixel = { x, y, rgba8, rgbSum };
        }
      }
      if (rgba8[3] > 0) nonTransparentCount++;

      pushBrightestSample(brightestSamples, { x, y, rgba8, rgbSum }, opts.brightestSampleCount);
    }
  }

  const centerX = Math.floor((width - 1) * 0.5);
  const centerY = Math.floor((height - 1) * 0.5);
  const centerPixel = {
    x: centerX,
    y: centerY,
    rgba8: sampleFramebufferPixel(gl, centerX, centerY)
  };
  centerPixel.rgbSum = centerPixel.rgba8[0] + centerPixel.rgba8[1] + centerPixel.rgba8[2];

  return {
    available: true,
    reason: 'ok',
    canvasWidth: Number.isFinite(canvas?.width) ? canvas.width : 0,
    canvasHeight: Number.isFinite(canvas?.height) ? canvas.height : 0,
    clientWidth: Number.isFinite(canvas?.clientWidth) ? canvas.clientWidth : 0,
    clientHeight: Number.isFinite(canvas?.clientHeight) ? canvas.clientHeight : 0,
    framebufferWidth: width,
    framebufferHeight: height,
    gridWidth: opts.gridWidth,
    gridHeight: opts.gridHeight,
    nonBlackThreshold: opts.nonBlackThreshold,
    sampleCount,
    nonBlackCount,
    nonTransparentCount,
    meanRgb: sumRgb.map((value) => value / sampleCount),
    maxRgb,
    minRgb,
    maxAlpha,
    meanAlpha: alphaSum / sampleCount,
    nonBlackRatio: nonBlackCount / sampleCount,
    nonTransparentRatio: nonTransparentCount / sampleCount,
    brightestSamples,
    centerPixel,
    representativeNonBlackPixel,
    likelyBlackFrame: nonBlackCount === 0
  };
}

export function readFramebufferPixelRgb(gl, canvas, pixel) {
  const width = Number.isFinite(canvas?.width) ? canvas.width | 0 : 0;
  const height = Number.isFinite(canvas?.height) ? canvas.height | 0 : 0;
  const x = Math.floor(pixel?.[0] ?? -1);
  const y = Math.floor(pixel?.[1] ?? -1);
  if (!gl || x < 0 || y < 0 || x >= width || y >= height) {
    return {
      valid: false,
      reason: 'pixel-out-of-bounds-or-missing-gl',
      pixel: [x, y],
      glPixel: [x, height - 1 - y],
      rgba8: [0, 0, 0, 0],
      rgb: [0, 0, 0]
    };
  }
  const yGl = height - 1 - y;
  const rgba = new Uint8Array(4);
  try {
    gl.readPixels(x, yGl, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
  } catch (error) {
    return {
      valid: false,
      reason: `readpixels-failed:${error?.message ?? 'unknown'}`,
      pixel: [x, y],
      glPixel: [x, yGl],
      rgba8: [0, 0, 0, 0],
      rgb: [0, 0, 0]
    };
  }
  return {
    valid: true,
    reason: 'readback-ok',
    pixel: [x, y],
    glPixel: [x, yGl],
    rgba8: Array.from(rgba),
    rgb: [rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0]
  };
}

export function computeFramebufferDeltaSummary(framebufferRgb, finalRgb) {
  if (!Array.isArray(framebufferRgb) || !Array.isArray(finalRgb)) {
    return { delta: null, deltaAbsMax: null };
  }
  const delta = [
    framebufferRgb[0] - finalRgb[0],
    framebufferRgb[1] - finalRgb[1],
    framebufferRgb[2] - finalRgb[2]
  ];
  return {
    delta,
    deltaAbsMax: Math.max(...delta.map((value) => Math.abs(value)))
  };
}

export function createSnapshotCanvasFromPixels(width, height, pixels) {
  const snapshotCanvas = document.createElement('canvas');
  snapshotCanvas.width = width;
  snapshotCanvas.height = height;
  const ctx = snapshotCanvas.getContext('2d', { willReadFrequently: true });
  const imageData = ctx.createImageData(width, height);
  const rowStride = width * 4;

  for (let y = 0; y < height; y++) {
    const srcOffset = (height - 1 - y) * rowStride;
    const dstOffset = y * rowStride;
    imageData.data.set(pixels.subarray(srcOffset, srcOffset + rowStride), dstOffset);
  }

  ctx.putImageData(imageData, 0, 0);
  return snapshotCanvas;
}

export function captureSnapshotCanvasFromGpu(gpu, errorPrefix = 'captureFrame failed') {
  const gl = gpu?.gl;
  if (!gl) {
    throw new Error(`${errorPrefix}: WebGL renderer is not ready`);
  }

  const width = gl.drawingBufferWidth | 0;
  const height = gl.drawingBufferHeight | 0;
  if (width <= 0 || height <= 0) {
    throw new Error(`${errorPrefix}: drawing buffer is empty`);
  }

  const pixels = new Uint8Array(width * height * 4);
  gl.finish();
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  return createSnapshotCanvasFromPixels(width, height, pixels);
}

export async function captureCanvasPngBlob(sourceCanvas, fileName, download) {
  const result = await downloadCanvasPng(sourceCanvas, fileName, { download });
  return result.blob;
}
