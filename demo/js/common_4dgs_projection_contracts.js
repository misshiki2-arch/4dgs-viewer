export const WEBGPU_PROJECTION_CONTRACT_SCHEMA_VERSION =
  'phase3-step4-webgpu-projection-contract-v1';

export const WEBGPU_PROJECTION_PARAM_MODES = Object.freeze({
  CUDA_ALIGNED: 'cuda-aligned',
  THREEJS: 'threejs'
});

export const WEBGPU_PROJECTION_CONTRACT_NAMES = Object.freeze({
  CUDA_ALIGNED: 'cuda-plus-z-forward-fx-fy-cx-cy',
  THREEJS: 'threejs-view-projection-ndc'
});

export const WEBGPU_PROJECTION_SOURCE_POSITION_MODE =
  'cpu-materialized-4d-state-position';

export function matrixToRows4(matrix) {
  const e = matrix?.elements;
  if (!e || e.length < 16) return null;
  return [
    [e[0], e[4], e[8], e[12]],
    [e[1], e[5], e[9], e[13]],
    [e[2], e[6], e[10], e[14]],
    [e[3], e[7], e[11], e[15]]
  ];
}

export function flattenRows4(rows, fallbackIdentity = false) {
  const source = Array.isArray(rows) && rows.length >= 4
    ? rows
    : (fallbackIdentity ? [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]
    ] : null);
  if (!source) return null;
  const out = [];
  for (let r = 0; r < 4; r += 1) {
    for (let c = 0; c < 4; c += 1) {
      const value = Number(source[r]?.[c]);
      out.push(Number.isFinite(value) ? value : (r === c ? 1 : 0));
    }
  }
  return out;
}

export function buildWebGpuProjectionContract({
  camera,
  screenSpaceCamera,
  renderW,
  renderH,
  sx,
  sy
}) {
  if (typeof camera?.updateMatrixWorld === 'function') camera.updateMatrixWorld(true);
  const override = screenSpaceCamera?.screenSpaceTransformOverride;
  const isCudaAligned = override?.mode === WEBGPU_PROJECTION_PARAM_MODES.CUDA_ALIGNED &&
    Array.isArray(override.viewMatrix);
  const mode = isCudaAligned ? 1 : 0;
  const viewRows = isCudaAligned
    ? flattenRows4(override.viewMatrix, true)
    : flattenRows4(matrixToRows4(camera?.matrixWorldInverse), true);
  const projectionRows = isCudaAligned
    ? flattenRows4(null, true)
    : flattenRows4(matrixToRows4(camera?.projectionMatrix), true);
  const fov = Number.isFinite(camera?.fov) ? Number(camera.fov) : 60;
  const aspect = Number.isFinite(camera?.aspect) ? Number(camera.aspect) : (renderW / Math.max(1, renderH));
  const tanFovY = Math.tan((fov * Math.PI / 180) * 0.5);
  const tanFovX = tanFovY * aspect;
  const fx = isCudaAligned && Number.isFinite(override?.intrinsics?.fx)
    ? Number(override.intrinsics.fx)
    : renderW / (2 * tanFovX);
  const fy = isCudaAligned && Number.isFinite(override?.intrinsics?.fy)
    ? Number(override.intrinsics.fy)
    : renderH / (2 * tanFovY);
  const cx = isCudaAligned && Number.isFinite(override?.intrinsics?.cx)
    ? Number(override.intrinsics.cx) - 0.5
    : (renderW - 1) * 0.5;
  const cy = isCudaAligned && Number.isFinite(override?.intrinsics?.cy)
    ? Number(override.intrinsics.cy) - 0.5
    : (renderH - 1) * 0.5;
  const pixelXSign = isCudaAligned && [-1, 1].includes(override?.pixelXSign)
    ? Number(override.pixelXSign)
    : 1;
  const data = new Float32Array([
    mode, renderW, renderH, 0,
    sx, sy, pixelXSign, 0,
    fx, fy, cx, cy,
    ...viewRows,
    ...projectionRows
  ]);
  return {
    data,
    summary: {
      schemaVersion: WEBGPU_PROJECTION_CONTRACT_SCHEMA_VERSION,
      mode: isCudaAligned
        ? WEBGPU_PROJECTION_PARAM_MODES.CUDA_ALIGNED
        : WEBGPU_PROJECTION_PARAM_MODES.THREEJS,
      projectionContract: isCudaAligned
        ? WEBGPU_PROJECTION_CONTRACT_NAMES.CUDA_ALIGNED
        : WEBGPU_PROJECTION_CONTRACT_NAMES.THREEJS,
      sourcePositionMode: WEBGPU_PROJECTION_SOURCE_POSITION_MODE,
      renderW,
      renderH,
      sx,
      sy,
      pixelXSign,
      intrinsics: { fx, fy, cx, cy },
      viewMatrixSource: isCudaAligned
        ? (override.viewMatrixSource ?? 'cuda-aligned-view-matrix')
        : 'threejs-camera-matrixWorldInverse',
      projectionMatrixSource: isCudaAligned
        ? 'intrinsics-fx-fy-cx-cy'
        : 'threejs-camera-projectionMatrix'
    }
  };
}
