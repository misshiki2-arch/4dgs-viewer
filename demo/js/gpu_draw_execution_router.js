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
    packedDirectSharedMergeCopyCount: directPackedDrawInfo?.packedDirectSharedMergeCopyCount ?? 0,
    packedDirectSharedMergeAttempted: !!directPackedDrawInfo?.packedDirectSharedMergeAttempted,
    packedDirectSharedMergeFailureReason: directPackedDrawInfo?.packedDirectSharedMergeFailureReason ?? 'none',
    packedDirectSharedMergeTextureWidth: directPackedDrawInfo?.packedDirectSharedMergeTextureWidth ?? 0,
    packedDirectSharedMergeTextureHeight: directPackedDrawInfo?.packedDirectSharedMergeTextureHeight ?? 0,
    packedDirectSharedMergeRowCount: directPackedDrawInfo?.packedDirectSharedMergeRowCount ?? 0,
    packedDirectSharedMergeRowsPerColumn: directPackedDrawInfo?.packedDirectSharedMergeRowsPerColumn ?? 0,
    packedDirectSharedMergeColumnCount: directPackedDrawInfo?.packedDirectSharedMergeColumnCount ?? 0,
    packedDirectSharedMergePolicySelectedPath: directPackedDrawInfo?.packedDirectSharedMergePolicySelectedPath ?? 'none',
    packedDirectSharedMergePolicyReason: directPackedDrawInfo?.packedDirectSharedMergePolicyReason ?? 'none',
    packedDirectSharedMergePolicyEstimatedCopyCount: directPackedDrawInfo?.packedDirectSharedMergePolicyEstimatedCopyCount ?? 0,
    packedDirectSharedMergePolicyEstimatedDispatchSavings: directPackedDrawInfo?.packedDirectSharedMergePolicyEstimatedDispatchSavings ?? 0,
    packedDirectSharedMergePolicyAtlasArea: directPackedDrawInfo?.packedDirectSharedMergePolicyAtlasArea ?? 0,
    packedDirectLayoutVersion: directPackedDrawInfo?.packedDirectLayoutVersion ?? 0,
    packedDirectStrideBytes: directPackedDrawInfo?.packedDirectStrideBytes ?? 0,
    packedDirectAttributeCount: directPackedDrawInfo?.packedDirectAttributeCount ?? 0,
    packedDirectOffsets: directPackedDrawInfo?.packedDirectOffsets ?? '',
    packedInterleavedBound: !!directPackedDrawInfo?.packedInterleavedBound,
    legacyExpandedArraysBuilt: false
  };
}

function buildDrawThroughputSummary({
  drawPathSelection,
  actualDrawPath,
  drawCallCount = 0,
  uploadCount = 0,
  uploadBytes = 0,
  sharedSetupCount = 0,
  sharedBindCount = 0,
  sharedDispatchCount = 0,
  sharedDispatchMode = 'none',
  sharedPayloadCount = 0,
  sharedMergeCopyCount = 0,
  sharedMergeAttempted = false,
  sharedMergeFailureReason = 'none',
  sharedMergeTextureWidth = 0,
  sharedMergeTextureHeight = 0,
  sharedMergeRowCount = 0,
  sharedMergeRowsPerColumn = 0,
  sharedMergeColumnCount = 0,
  sharedMergePolicySelectedPath = 'none',
  sharedMergePolicyReason = 'none',
  sharedMergePolicyEstimatedCopyCount = 0,
  sharedMergePolicyEstimatedDispatchSavings = 0,
  sharedMergePolicyAtlasArea = 0,
  usesGpuResidentPayload = false
}) {
  const dispatchPressure = sharedDispatchCount > 0
    ? 'shared-dispatch-bound'
    : drawCallCount > 0
      ? 'draw-call-bound'
      : 'no-draw-work';

  return {
    requestedDrawPath: drawPathSelection?.requestedPath ?? actualDrawPath,
    actualDrawPath: actualDrawPath ?? drawPathSelection?.actualPath ?? 'legacy',
    drawPathFallbackReason: drawPathSelection?.fallbackReason ?? 'none',
    drawCallCount: Number.isFinite(drawCallCount) ? Math.max(0, drawCallCount | 0) : 0,
    uploadCount: Number.isFinite(uploadCount) ? Math.max(0, uploadCount | 0) : 0,
    uploadBytes: Number.isFinite(uploadBytes) ? Math.max(0, uploadBytes | 0) : 0,
    sharedSetupCount: Number.isFinite(sharedSetupCount) ? Math.max(0, sharedSetupCount | 0) : 0,
    sharedBindCount: Number.isFinite(sharedBindCount) ? Math.max(0, sharedBindCount | 0) : 0,
    sharedDispatchCount: Number.isFinite(sharedDispatchCount) ? Math.max(0, sharedDispatchCount | 0) : 0,
    sharedDispatchMode: sharedDispatchMode ?? 'none',
    sharedPayloadCount: Number.isFinite(sharedPayloadCount) ? Math.max(0, sharedPayloadCount | 0) : 0,
    sharedMergeCopyCount: Number.isFinite(sharedMergeCopyCount) ? Math.max(0, sharedMergeCopyCount | 0) : 0,
    sharedMergeAttempted: !!sharedMergeAttempted,
    sharedMergeFailureReason: sharedMergeFailureReason ?? 'none',
    sharedMergeTextureWidth: Number.isFinite(sharedMergeTextureWidth) ? Math.max(0, sharedMergeTextureWidth | 0) : 0,
    sharedMergeTextureHeight: Number.isFinite(sharedMergeTextureHeight) ? Math.max(0, sharedMergeTextureHeight | 0) : 0,
    sharedMergeRowCount: Number.isFinite(sharedMergeRowCount) ? Math.max(0, sharedMergeRowCount | 0) : 0,
    sharedMergeRowsPerColumn: Number.isFinite(sharedMergeRowsPerColumn) ? Math.max(0, sharedMergeRowsPerColumn | 0) : 0,
    sharedMergeColumnCount: Number.isFinite(sharedMergeColumnCount) ? Math.max(0, sharedMergeColumnCount | 0) : 0,
    sharedMergePolicySelectedPath: sharedMergePolicySelectedPath ?? 'none',
    sharedMergePolicyReason: sharedMergePolicyReason ?? 'none',
    sharedMergePolicyEstimatedCopyCount: Number.isFinite(sharedMergePolicyEstimatedCopyCount) ? Math.max(0, sharedMergePolicyEstimatedCopyCount | 0) : 0,
    sharedMergePolicyEstimatedDispatchSavings: Number.isFinite(sharedMergePolicyEstimatedDispatchSavings) ? Math.max(0, sharedMergePolicyEstimatedDispatchSavings | 0) : 0,
    sharedMergePolicyAtlasArea: Number.isFinite(sharedMergePolicyAtlasArea) ? Math.max(0, sharedMergePolicyAtlasArea | 0) : 0,
    usesGpuResidentPayload: !!usesGpuResidentPayload,
    throughputPressure: dispatchPressure
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
    drawThroughputSummary: buildDrawThroughputSummary({
      drawPathSelection,
      actualDrawPath: GPU_DRAW_PATH_PACKED,
      drawCallCount: directPackedDrawInfo?.drawCallCount ?? 1,
      uploadCount: directPackedDrawInfo?.uploadCount ?? 0,
      uploadBytes: directPackedDrawInfo?.uploadSummary?.packedUploadBytes ?? 0,
      sharedSetupCount: directPackedDrawInfo?.packedDirectSharedSetupCount ?? 0,
      sharedBindCount: directPackedDrawInfo?.packedDirectSharedBindCount ?? 0,
      sharedDispatchCount: directPackedDrawInfo?.packedDirectSharedDispatchCount ?? 0,
      sharedDispatchMode: directPackedDrawInfo?.packedDirectSharedDispatchMode ?? 'none',
      sharedPayloadCount: directPackedDrawInfo?.packedDirectSharedPayloadCount ?? 0,
      sharedMergeCopyCount: directPackedDrawInfo?.packedDirectSharedMergeCopyCount ?? 0,
      sharedMergeAttempted: !!directPackedDrawInfo?.packedDirectSharedMergeAttempted,
      sharedMergeFailureReason: directPackedDrawInfo?.packedDirectSharedMergeFailureReason ?? 'none',
      sharedMergeTextureWidth: directPackedDrawInfo?.packedDirectSharedMergeTextureWidth ?? 0,
      sharedMergeTextureHeight: directPackedDrawInfo?.packedDirectSharedMergeTextureHeight ?? 0,
      sharedMergeRowCount: directPackedDrawInfo?.packedDirectSharedMergeRowCount ?? 0,
      sharedMergeRowsPerColumn: directPackedDrawInfo?.packedDirectSharedMergeRowsPerColumn ?? 0,
      sharedMergeColumnCount: directPackedDrawInfo?.packedDirectSharedMergeColumnCount ?? 0,
      sharedMergePolicySelectedPath: directPackedDrawInfo?.packedDirectSharedMergePolicySelectedPath ?? 'none',
      sharedMergePolicyReason: directPackedDrawInfo?.packedDirectSharedMergePolicyReason ?? 'none',
      sharedMergePolicyEstimatedCopyCount: directPackedDrawInfo?.packedDirectSharedMergePolicyEstimatedCopyCount ?? 0,
      sharedMergePolicyEstimatedDispatchSavings: directPackedDrawInfo?.packedDirectSharedMergePolicyEstimatedDispatchSavings ?? 0,
      sharedMergePolicyAtlasArea: directPackedDrawInfo?.packedDirectSharedMergePolicyAtlasArea ?? 0,
      usesGpuResidentPayload: !!directPackedDrawInfo?.packedDirectUsesGpuResidentPayload
    }),
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
    drawThroughputSummary: buildDrawThroughputSummary({
      drawPathSelection,
      actualDrawPath: GPU_DRAW_PATH_GPU_SCREEN,
      drawCallCount: gpuScreenDrawInfo?.drawCallCount ?? 1,
      uploadCount: gpuScreenDrawInfo?.sharedDispatchCount > 0 ? 0 : 1,
      uploadBytes: gpuScreenDrawInfo?.gpuScreenSummary?.gpuScreenUploadBytes
        ?? gpuScreenDrawInfo?.gpuScreenUploadSummary?.packedUploadBytes
        ?? 0,
      sharedSetupCount: gpuScreenDrawInfo?.sharedSetupCount ?? 0,
      sharedBindCount: gpuScreenDrawInfo?.sharedBindCount ?? 0,
      sharedDispatchCount: gpuScreenDrawInfo?.sharedDispatchCount ?? 0,
      sharedDispatchMode: gpuScreenDrawInfo?.sharedDispatchMode ?? 'none',
      sharedPayloadCount: gpuScreenDrawInfo?.sharedPayloadCount ?? 0,
      sharedMergeCopyCount: gpuScreenDrawInfo?.sharedMergeCopyCount ?? 0,
      sharedMergeAttempted: !!gpuScreenDrawInfo?.sharedMergeAttempted,
      sharedMergeFailureReason: gpuScreenDrawInfo?.sharedMergeFailureReason ?? 'none',
      sharedMergeTextureWidth: gpuScreenDrawInfo?.sharedMergeTextureWidth ?? 0,
      sharedMergeTextureHeight: gpuScreenDrawInfo?.sharedMergeTextureHeight ?? 0,
      sharedMergeRowCount: gpuScreenDrawInfo?.sharedMergeRowCount ?? 0,
      sharedMergeRowsPerColumn: gpuScreenDrawInfo?.sharedMergeRowsPerColumn ?? 0,
      sharedMergeColumnCount: gpuScreenDrawInfo?.sharedMergeColumnCount ?? 0,
      sharedMergePolicySelectedPath: gpuScreenDrawInfo?.sharedMergePolicySelectedPath ?? 'none',
      sharedMergePolicyReason: gpuScreenDrawInfo?.sharedMergePolicyReason ?? 'none',
      sharedMergePolicyEstimatedCopyCount: gpuScreenDrawInfo?.sharedMergePolicyEstimatedCopyCount ?? 0,
      sharedMergePolicyEstimatedDispatchSavings: gpuScreenDrawInfo?.sharedMergePolicyEstimatedDispatchSavings ?? 0,
      sharedMergePolicyAtlasArea: gpuScreenDrawInfo?.sharedMergePolicyAtlasArea ?? 0,
      usesGpuResidentPayload: !!(
        gpuScreenDrawInfo?.gpuScreenSummary?.gpuScreenUsesGpuResidentPayload
      )
    }),
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
    const fallbackResult = executeFallbackFullFrameDraw({
      gl,
      gpu,
      canvas,
      legacyDrawData,
      drawPathSelection
    });
    return {
      ...fallbackResult,
      drawThroughputSummary: buildDrawThroughputSummary({
        drawPathSelection,
        actualDrawPath: GPU_DRAW_PATH_LEGACY,
        drawCallCount: fallbackResult?.executionSummary?.drawCallCount ?? 1,
        uploadCount: fallbackResult?.executionSummary?.uploadCount ?? 1,
        uploadBytes: 0,
        sharedSetupCount: 0,
        sharedBindCount: 0,
        sharedDispatchCount: 0,
        sharedDispatchMode: 'none',
        sharedPayloadCount: 0,
        sharedMergeCopyCount: 0,
        usesGpuResidentPayload: false
      })
    };
  }

  throw new Error(`Unsupported full-frame draw path: ${actualPath}`);
}
