// Step29:
// draw path 選択の責務をこのファイルに集約する。
// packed は formal reference path、legacy は fallback、gpu-screen は experimental compare path。
// Step28 では actualReferencePath を持たせた。
// Step29 では、comparison debug が actual / source / reference を明示できるように、
// selector 側でも actual path と reference path の意味を整理する。
// ここで扱うのは draw path 自体なので、source path は持たない。
// source path は gpu_screen_space_builder.js 側で決まる。

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

function getPathRole(path) {
  if (path === GPU_DRAW_PATH_PACKED) return 'formal-reference';
  if (path === GPU_DRAW_PATH_GPU_SCREEN) return 'experimental-draw';
  return 'fallback';
}

function getReferencePathForActual(actualPath) {
  if (actualPath === GPU_DRAW_PATH_GPU_SCREEN) return GPU_DRAW_PATH_PACKED;
  return actualPath;
}

function getReferenceRoleForActual(actualPath) {
  if (actualPath === GPU_DRAW_PATH_GPU_SCREEN) return 'formal-reference';
  return getPathRole(actualPath);
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

export function summarizeDrawPathSelection(selection) {
  const requestedPath = normalizeRequestedPath(selection?.requestedPath);
  const actualPath = normalizeRequestedPath(selection?.actualPath);
  const fallbackReason =
    typeof selection?.fallbackReason === 'string' && selection?.fallbackReason.length > 0
      ? selection.fallbackReason
      : 'none';

  const usedFallback = requestedPath !== actualPath || fallbackReason !== 'none';
  const actualReferencePath = getReferencePathForActual(actualPath);
  const actualReferenceRole = getReferenceRoleForActual(actualPath);

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
    actualReferencePath,
    actualReferenceRole,
    comparisonSummary: {
      requestedPath,
      requestedRole: getPathRole(requestedPath),
      actualPath,
      actualRole: getPathRole(actualPath),
      referencePath: actualReferencePath,
      referenceRole: actualReferenceRole,
      usedFallback,
      fallbackReason
    }
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
    `referencePath=${summary.actualReferencePath}`,
    `referenceRole=${summary.actualReferenceRole}`
  ].join('  ');
}
