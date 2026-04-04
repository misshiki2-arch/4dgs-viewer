// Step25:
// draw path 選択の責務をこのファイルに集約する。
// packed を正式 full-frame 経路、legacy を fallback、gpu-screen を future とする。
// renderer 側はこの結果を受け取って実行するだけにする。

export const GPU_DRAW_PATH_LEGACY = 'legacy';
export const GPU_DRAW_PATH_PACKED = 'packed';
export const GPU_DRAW_PATH_GPU_SCREEN = 'gpu-screen';

const ALLOWED_DRAW_PATHS = new Set([
  GPU_DRAW_PATH_LEGACY,
  GPU_DRAW_PATH_PACKED,
  GPU_DRAW_PATH_GPU_SCREEN
]);

function normalizeRequestedPath(path) {
  if (typeof path !== 'string') return GPU_DRAW_PATH_PACKED;
  return ALLOWED_DRAW_PATHS.has(path) ? path : GPU_DRAW_PATH_PACKED;
}

export function getRequestedDrawPath(ui) {
  const uiValue = ui?.drawPathSelect?.value;
  return normalizeRequestedPath(uiValue);
}

export function resolveDrawPath({ requestedPath, hasPackedScreenSpace = false, hasGpuScreenPath = false } = {}) {
  const normalizedRequestedPath = normalizeRequestedPath(requestedPath);

  if (normalizedRequestedPath === GPU_DRAW_PATH_PACKED) {
    if (hasPackedScreenSpace) {
      return { requestedPath: GPU_DRAW_PATH_PACKED, actualPath: GPU_DRAW_PATH_PACKED, fallbackReason: 'none' };
    }
    return { requestedPath: GPU_DRAW_PATH_PACKED, actualPath: GPU_DRAW_PATH_LEGACY, fallbackReason: 'missing-packed-screen-space' };
  }

  if (normalizedRequestedPath === GPU_DRAW_PATH_GPU_SCREEN) {
    if (hasGpuScreenPath) {
      return { requestedPath: GPU_DRAW_PATH_GPU_SCREEN, actualPath: GPU_DRAW_PATH_GPU_SCREEN, fallbackReason: 'none' };
    }
    if (hasPackedScreenSpace) {
      return { requestedPath: GPU_DRAW_PATH_GPU_SCREEN, actualPath: GPU_DRAW_PATH_PACKED, fallbackReason: 'gpu-screen-not-ready' };
    }
    return { requestedPath: GPU_DRAW_PATH_GPU_SCREEN, actualPath: GPU_DRAW_PATH_LEGACY, fallbackReason: 'gpu-screen-not-ready-and-missing-packed-screen-space' };
  }

  return { requestedPath: GPU_DRAW_PATH_LEGACY, actualPath: GPU_DRAW_PATH_LEGACY, fallbackReason: 'none' };
}

export function summarizeDrawPathSelection(selection) {
  const requestedPath = normalizeRequestedPath(selection?.requestedPath);
  const actualPath = normalizeRequestedPath(selection?.actualPath);
  const fallbackReason = typeof selection?.fallbackReason === 'string' && selection.fallbackReason.length > 0 ? selection.fallbackReason : 'none';
  return {
    requestedPath,
    actualPath,
    fallbackReason,
    usedFallback: requestedPath !== actualPath || fallbackReason !== 'none',
    packedFormalPath: actualPath === GPU_DRAW_PATH_PACKED,
    legacyFallbackPath: actualPath === GPU_DRAW_PATH_LEGACY,
    gpuScreenPath: actualPath === GPU_DRAW_PATH_GPU_SCREEN
  };
}

export function formatDrawPathSelection(selection) {
  const summary = summarizeDrawPathSelection(selection);
  return [
    `requestedDrawPath=${summary.requestedPath}`,
    `actualDrawPath=${summary.actualPath}`,
    `drawPathFallbackReason=${summary.fallbackReason}`,
    `usedFallback=${summary.usedFallback}`
  ].join('  ');
}
