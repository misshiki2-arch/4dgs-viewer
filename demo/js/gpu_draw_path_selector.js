// Step27:
// draw path 選択の責務をこのファイルに集約する。
// packed は formal reference path、legacy は fallback、gpu-screen は experimental path。
// Step26 では gpu-screen を実行可能経路にした。
// Step27 ではさらに、packed 参照の experimental path であることを
// summary 上でも見えるようにする。

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
  return normalizeRequestedPath(ui?.drawPathSelect?.value);
}

export function resolveDrawPath({
  requestedPath,
  hasPackedScreenSpace = false,
  hasGpuScreenPath = false
} = {}) {
  const normalizedRequestedPath = normalizeRequestedPath(requestedPath);

  if (normalizedRequestedPath === GPU_DRAW_PATH_GPU_SCREEN) {
    if (hasGpuScreenPath) {
      return {
        requestedPath: GPU_DRAW_PATH_GPU_SCREEN,
        actualPath: GPU_DRAW_PATH_GPU_SCREEN,
        fallbackReason: 'none'
      };
    }

    if (hasPackedScreenSpace) {
      return {
        requestedPath: GPU_DRAW_PATH_GPU_SCREEN,
        actualPath: GPU_DRAW_PATH_PACKED,
        fallbackReason: 'gpu-screen-not-ready'
      };
    }

    return {
      requestedPath: GPU_DRAW_PATH_GPU_SCREEN,
      actualPath: GPU_DRAW_PATH_LEGACY,
      fallbackReason: 'gpu-screen-not-ready-and-missing-packed-screen-space'
    };
  }

  if (normalizedRequestedPath === GPU_DRAW_PATH_PACKED) {
    if (hasPackedScreenSpace) {
      return {
        requestedPath: GPU_DRAW_PATH_PACKED,
        actualPath: GPU_DRAW_PATH_PACKED,
        fallbackReason: 'none'
      };
    }

    return {
      requestedPath: GPU_DRAW_PATH_PACKED,
      actualPath: GPU_DRAW_PATH_LEGACY,
      fallbackReason: 'missing-packed-screen-space'
    };
  }

  return {
    requestedPath: GPU_DRAW_PATH_LEGACY,
    actualPath: GPU_DRAW_PATH_LEGACY,
    fallbackReason: 'none'
  };
}

function getPathRole(path) {
  if (path === GPU_DRAW_PATH_PACKED) return 'formal-reference';
  if (path === GPU_DRAW_PATH_GPU_SCREEN) return 'experimental';
  return 'fallback';
}

export function summarizeDrawPathSelection(selection) {
  const requestedPath = normalizeRequestedPath(selection?.requestedPath);
  const actualPath = normalizeRequestedPath(selection?.actualPath);
  const fallbackReason =
    typeof selection?.fallbackReason === 'string' && selection.fallbackReason.length > 0
      ? selection.fallbackReason
      : 'none';

  const usedFallback = requestedPath !== actualPath || fallbackReason !== 'none';

  return {
    requestedPath,
    actualPath,
    fallbackReason,
    usedFallback,

    requestedRole: getPathRole(requestedPath),
    actualRole: getPathRole(actualPath),

    packedFormalPath: actualPath === GPU_DRAW_PATH_PACKED,
    legacyFallbackPath: actualPath === GPU_DRAW_PATH_LEGACY,
    gpuScreenExperimentalPath: actualPath === GPU_DRAW_PATH_GPU_SCREEN,

    requestedPacked: requestedPath === GPU_DRAW_PATH_PACKED,
    requestedLegacy: requestedPath === GPU_DRAW_PATH_LEGACY,
    requestedGpuScreen: requestedPath === GPU_DRAW_PATH_GPU_SCREEN,

    // Step27:
    // gpu-screen は packed formal reference を参照する experimental path とみなす。
    actualReferencePath:
      actualPath === GPU_DRAW_PATH_GPU_SCREEN
        ? GPU_DRAW_PATH_PACKED
        : actualPath,
    actualReferenceRole:
      actualPath === GPU_DRAW_PATH_GPU_SCREEN
        ? 'formal-reference'
        : getPathRole(actualPath)
  };
}

export function formatDrawPathSelection(selection) {
  const summary = summarizeDrawPathSelection(selection);
  return [
    `requestedDrawPath=${summary.requestedPath}`,
    `actualDrawPath=${summary.actualPath}`,
    `drawPathFallbackReason=${summary.fallbackReason}`,
    `usedFallback=${summary.usedFallback}`,
    `requestedRole=${summary.requestedRole}`,
    `actualRole=${summary.actualRole}`,
    `actualReferencePath=${summary.actualReferencePath}`
  ].join('  ');
}
