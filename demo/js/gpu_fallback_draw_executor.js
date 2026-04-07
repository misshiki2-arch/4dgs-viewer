import { uploadAndDraw } from './gpu_draw_utils.js';

function buildFallbackPackedUploadSummary() {
  return {
    packedUploadBytes: 0,
    packedUploadCount: 0,
    packedUploadLength: 0,
    packedUploadCapacityBytes: 0,
    packedUploadReusedCapacity: false,
    packedDirectDraw: false,
    packedDirectLayoutVersion: 0,
    packedDirectStrideBytes: 0,
    packedDirectAttributeCount: 0,
    packedDirectOffsets: '',
    packedInterleavedBound: false,
    legacyExpandedArraysBuilt: true
  };
}

function buildFallbackExecutionSummary(drawPathSelection, legacyDrawData) {
  return {
    tileBatchCount: 1,
    nonEmptyTileBatchCount: 1,
    totalTileDrawCount: legacyDrawData?.nDraw ?? 0,
    uploadCount: 1,
    drawCallCount: 1,
    requestedDrawPath: drawPathSelection?.requestedPath ?? 'legacy',
    actualDrawPath: drawPathSelection?.actualPath ?? 'legacy',
    drawPathFallbackReason: drawPathSelection?.fallbackReason ?? 'none'
  };
}

export function executeFallbackFullFrameDraw({
  gl,
  gpu,
  canvas,
  legacyDrawData,
  drawPathSelection
}) {
  uploadAndDraw(gl, gpu, legacyDrawData, canvas.width, canvas.height);

  return {
    executionSummary: buildFallbackExecutionSummary(drawPathSelection, legacyDrawData),
    packedUploadSummary: buildFallbackPackedUploadSummary(),
    directPackedDrawInfo: null,
    gpuScreenDrawInfo: null
  };
}
