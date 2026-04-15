import {
  GPU_DRAW_PATH_LEGACY,
  GPU_DRAW_PATH_PACKED,
  GPU_DRAW_PATH_GPU_SCREEN
} from './gpu_draw_path_selector.js';
import { uploadAndDrawPackedDirect } from './gpu_packed_draw_executor.js';
import { uploadAndDrawGpuScreen } from './gpu_screen_draw_executor.js';
import { executeFallbackFullFrameDraw } from './gpu_fallback_draw_executor.js';

function buildPackedExecutionSummary(drawPathSelection, directPackedDrawInfo) {
  return {
    tileBatchCount: 1,
    nonEmptyTileBatchCount: 0,
    totalTileDrawCount: directPackedDrawInfo?.drawCount ?? 0,
    uploadCount: directPackedDrawInfo?.uploadCount ?? 1,
    drawCallCount: directPackedDrawInfo?.drawCallCount ?? 1,
    requestedDrawPath: drawPathSelection?.requestedPath ?? GPU_DRAW_PATH_PACKED,
    actualDrawPath: drawPathSelection?.actualPath ?? GPU_DRAW_PATH_PACKED,
    drawPathFallbackReason: drawPathSelection?.fallbackReason ?? 'none'
  };
}

function buildPackedUploadSummary(directPackedDrawInfo) {
  return {
    ...(directPackedDrawInfo?.uploadSummary || {}),
    packedDirectDraw: !!directPackedDrawInfo?.packedDirectDraw,
    packedDirectUsesGpuResidentPayload: !!directPackedDrawInfo?.packedDirectUsesGpuResidentPayload,
    packedDirectSharedSetupCount: directPackedDrawInfo?.packedDirectSharedSetupCount ?? 0,
    packedDirectSharedBindCount: directPackedDrawInfo?.packedDirectSharedBindCount ?? 0,
    packedDirectSharedDispatchCount: directPackedDrawInfo?.packedDirectSharedDispatchCount ?? 0,
    packedDirectSharedDispatchMode: directPackedDrawInfo?.packedDirectSharedDispatchMode ?? 'none',
    packedDirectSharedPayloadCount: directPackedDrawInfo?.packedDirectSharedPayloadCount ?? 0,
    packedDirectLayoutVersion: directPackedDrawInfo?.packedDirectLayoutVersion ?? 0,
    packedDirectStrideBytes: directPackedDrawInfo?.packedDirectStrideBytes ?? 0,
    packedDirectAttributeCount: directPackedDrawInfo?.packedDirectAttributeCount ?? 0,
    packedDirectOffsets: directPackedDrawInfo?.packedDirectOffsets ?? '',
    packedInterleavedBound: !!directPackedDrawInfo?.packedInterleavedBound,
    legacyExpandedArraysBuilt: false
  };
}

function executePackedFullFrameDraw({
  gl,
  gpu,
  canvas,
  packedScreenSpace,
  gpuScreenSourceSpace,
  drawPathSelection
}) {
  const packedDirectSourceSpace = Array.isArray(gpuScreenSourceSpace?.gpuPackedPayloads) &&
    gpuScreenSourceSpace.gpuPackedPayloads.length > 0
    ? gpuScreenSourceSpace
    : packedScreenSpace;

  const directPackedDrawInfo = uploadAndDrawPackedDirect(
    gl,
    gpu,
    packedDirectSourceSpace,
    canvas.width,
    canvas.height
  );

  return {
    executionSummary: buildPackedExecutionSummary(
      drawPathSelection,
      directPackedDrawInfo
    ),
    packedUploadSummary: buildPackedUploadSummary(directPackedDrawInfo),
    directPackedDrawInfo,
    gpuScreenDrawInfo: null
  };
}

function buildGpuScreenExecutionSummary(drawPathSelection, gpuScreenDrawInfo) {
  return {
    tileBatchCount: 1,
    nonEmptyTileBatchCount: 0,
    totalTileDrawCount: gpuScreenDrawInfo?.drawCount ?? 0,
    uploadCount: 1,
    drawCallCount: gpuScreenDrawInfo?.drawCallCount ?? 1,
    requestedDrawPath: drawPathSelection?.requestedPath ?? GPU_DRAW_PATH_GPU_SCREEN,
    actualDrawPath: drawPathSelection?.actualPath ?? GPU_DRAW_PATH_GPU_SCREEN,
    drawPathFallbackReason: drawPathSelection?.fallbackReason ?? 'none'
  };
}

function buildGpuScreenPackedUploadSummary() {
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
    legacyExpandedArraysBuilt: false
  };
}

function executeGpuScreenFullFrameDraw({
  gl,
  gpu,
  canvas,
  gpuScreenSourceSpace,
  drawPathSelection
}) {
  const gpuScreenDrawInfo = uploadAndDrawGpuScreen(
    gl,
    gpu,
    gpuScreenSourceSpace,
    canvas.width,
    canvas.height
  );

  return {
    executionSummary: buildGpuScreenExecutionSummary(drawPathSelection, gpuScreenDrawInfo),
    packedUploadSummary: buildGpuScreenPackedUploadSummary(),
    directPackedDrawInfo: null,
    gpuScreenDrawInfo
  };
}

export function executeFullFrameDrawByPath({
  gl,
  gpu,
  canvas,
  drawPathSelection,
  packedScreenSpace,
  gpuScreenSourceSpace,
  legacyDrawData
}) {
  const actualPath = drawPathSelection?.actualPath ?? GPU_DRAW_PATH_LEGACY;

  if (actualPath === GPU_DRAW_PATH_GPU_SCREEN) {
    return executeGpuScreenFullFrameDraw({
      gl,
      gpu,
      canvas,
      gpuScreenSourceSpace,
      drawPathSelection
    });
  }

  if (actualPath === GPU_DRAW_PATH_PACKED) {
    return executePackedFullFrameDraw({
      gl,
      gpu,
      canvas,
      packedScreenSpace,
      gpuScreenSourceSpace,
      drawPathSelection
    });
  }

  if (actualPath === GPU_DRAW_PATH_LEGACY) {
    return executeFallbackFullFrameDraw({
      gl,
      gpu,
      canvas,
      legacyDrawData,
      drawPathSelection
    });
  }

  throw new Error(`Unsupported full-frame draw path: ${actualPath}`);
}
