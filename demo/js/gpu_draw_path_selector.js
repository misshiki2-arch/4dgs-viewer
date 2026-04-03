// Step21:
// draw path の分岐司令塔。
// 今後 legacy / packed / gpu-screen-space などが増える前提で、
// 分岐ロジックを renderer 本体から分離する。

export const GPU_DRAW_PATH_LEGACY = 'legacy';
export const GPU_DRAW_PATH_PACKED = 'packed';
export const GPU_DRAW_PATH_GPU_SCREEN = 'gpu-screen';

export function getRequestedDrawPath(ui = null) {
  if (ui?.drawPathSelect?.value) {
    return ui.drawPathSelect.value;
  }

  if (ui?.usePackedVisiblePathCheck?.checked) {
    return GPU_DRAW_PATH_PACKED;
  }

  return GPU_DRAW_PATH_LEGACY;
}

export function normalizeDrawPath(path) {
  if (
    path === GPU_DRAW_PATH_LEGACY ||
    path === GPU_DRAW_PATH_PACKED ||
    path === GPU_DRAW_PATH_GPU_SCREEN
  ) {
    return path;
  }
  return GPU_DRAW_PATH_LEGACY;
}

export function resolveDrawPath({
  requestedPath,
  hasPackedScreenSpace = false,
  hasGpuScreenPath = false
}) {
  const requested = normalizeDrawPath(requestedPath);

  if (requested === GPU_DRAW_PATH_GPU_SCREEN) {
    if (hasGpuScreenPath) {
      return {
        requestedPath: requested,
        actualPath: GPU_DRAW_PATH_GPU_SCREEN,
        fallbackReason: 'none'
      };
    }
    if (hasPackedScreenSpace) {
      return {
        requestedPath: requested,
        actualPath: GPU_DRAW_PATH_PACKED,
        fallbackReason: 'gpu-screen-unavailable'
      };
    }
    return {
      requestedPath: requested,
      actualPath: GPU_DRAW_PATH_LEGACY,
      fallbackReason: 'gpu-screen-unavailable'
    };
  }

  if (requested === GPU_DRAW_PATH_PACKED) {
    if (hasPackedScreenSpace) {
      return {
        requestedPath: requested,
        actualPath: GPU_DRAW_PATH_PACKED,
        fallbackReason: 'none'
      };
    }
    return {
      requestedPath: requested,
      actualPath: GPU_DRAW_PATH_LEGACY,
      fallbackReason: 'packed-unavailable'
    };
  }

  return {
    requestedPath: requested,
    actualPath: GPU_DRAW_PATH_LEGACY,
    fallbackReason: 'none'
  };
}

export function summarizeDrawPathSelection(selection) {
  return {
    requestedPath: selection?.requestedPath ?? GPU_DRAW_PATH_LEGACY,
    actualPath: selection?.actualPath ?? GPU_DRAW_PATH_LEGACY,
    fallbackReason: selection?.fallbackReason ?? 'none'
  };
}
