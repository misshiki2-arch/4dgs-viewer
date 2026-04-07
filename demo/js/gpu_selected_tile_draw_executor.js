import { renderPerTileBatches } from './gpu_draw_utils.js';
import { GPU_DRAW_PATH_LEGACY } from './gpu_draw_path_selector.js';

function enableTileScissor(gl, canvas, tileRect) {
  const [x0, y0, x1, y1] = tileRect;
  const w = Math.max(0, x1 - x0);
  const h = Math.max(0, y1 - y0);
  const scY = canvas.height - y1;
  gl.enable(gl.SCISSOR_TEST);
  gl.scissor(x0, scY, w, h);
}

function disableTileScissor(gl) {
  gl.disable(gl.SCISSOR_TEST);
}

function buildRectMap(focusTileRects) {
  return new Map((Array.isArray(focusTileRects) ? focusTileRects : []).map((item) => [item.tileId, item.rect]));
}

function buildPackedUploadSummary() {
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

export function executeSelectedTileLegacyDraw({
  gl,
  gpu,
  canvas,
  focusTileRects,
  perTileDrawBatches,
  drawPathSelection,
  hooks = {}
}) {
  const rectMap = buildRectMap(focusTileRects);

  const executionSummary = renderPerTileBatches(
    gl,
    gpu,
    perTileDrawBatches,
    canvas.width,
    canvas.height,
    {
      beforeTile: (item) => {
        const rect = rectMap.get(item.tileId);
        if (rect) enableTileScissor(gl, canvas, rect);
        else disableTileScissor(gl);
        if (typeof hooks.beforeTile === 'function') hooks.beforeTile(item, rect);
      },
      afterTile: (item, drawSummary) => {
        disableTileScissor(gl);
        if (typeof hooks.afterTile === 'function') hooks.afterTile(item, drawSummary);
      }
    }
  );

  disableTileScissor(gl);

  executionSummary.requestedDrawPath = drawPathSelection?.requestedPath ?? GPU_DRAW_PATH_LEGACY;
  executionSummary.actualDrawPath = GPU_DRAW_PATH_LEGACY;
  executionSummary.drawPathFallbackReason = drawPathSelection?.fallbackReason ?? 'none';

  return {
    executionSummary,
    packedUploadSummary: buildPackedUploadSummary(),
    directPackedDrawInfo: null,
    gpuScreenDrawInfo: null
  };
}
