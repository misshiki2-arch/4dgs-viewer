import { computeGaussianState, computeScreenSplat } from './rot4d_math.js';
import { evalSHColor } from './sh_eval.js';
import { clampInt, computeTileRangeFromAABB } from './gpu_tile_utils.js';

function buildColorAlpha(color, opacity) {
  const r = Array.isArray(color) && Number.isFinite(color[0]) ? color[0] : 0;
  const g = Array.isArray(color) && Number.isFinite(color[1]) ? color[1] : 0;
  const b = Array.isArray(color) && Number.isFinite(color[2]) ? color[2] : 0;
  const a = Number.isFinite(opacity)
    ? opacity
    : (Array.isArray(color) && Number.isFinite(color[3]) ? color[3] : 0);
  return [r, g, b, a];
}

export function buildVisibleItemForCandidate({
  raw,
  index,
  camera,
  screenSpaceCamera = null,
  renderW,
  renderH,
  canvasWidth,
  canvasHeight,
  sx,
  sy,
  timestamp,
  scalingModifier,
  sigmaScale,
  prefilterVar,
  useRot4d,
  flags,
  camPos,
  timeDuration,
  useSH,
  forceSh3d,
  tileGrid = null
}) {
  const gs = computeGaussianState(
    raw,
    index,
    timestamp,
    scalingModifier,
    sigmaScale,
    prefilterVar,
    useRot4d,
    flags
  );
  if (!gs) return null;

  const color = evalSHColor(raw, index, camPos, gs.pos, timestamp, timeDuration, useSH, forceSh3d);
  const splat = computeScreenSplat(screenSpaceCamera || camera, gs.pos, gs.cov3, gs.opacity, renderW, renderH);
  if (!splat) return null;

  const px = splat.px * sx;
  const py = splat.py * sy;
  const drawRadius = splat.radius * Math.max(sx, sy);
  const coverageRadius = Math.max(1.0, drawRadius);
  const minX = clampInt(Math.floor(px - coverageRadius), 0, canvasWidth - 1);
  const maxX = clampInt(Math.ceil(px + coverageRadius), 0, canvasWidth - 1);
  const minY = clampInt(Math.floor(py - coverageRadius), 0, canvasHeight - 1);
  const maxY = clampInt(Math.ceil(py + coverageRadius), 0, canvasHeight - 1);
  const aabb = [minX, minY, maxX, maxY];
  const tileRange = tileGrid
    ? computeTileRangeFromAABB(aabb, tileGrid.tileCols, tileGrid.tileRows, tileGrid.tileSize)
    : null;
  const colorAlpha = buildColorAlpha(color, splat.opacity);

  return {
    srcIndex: index,
    px,
    py,
    radius: drawRadius,
    depth: splat.depth,
    colorAlpha,
    conic: [
      splat.conic[0] / (sx * sx),
      splat.conic[1] / (sx * sy),
      splat.conic[2] / (sy * sy)
    ],
    aabb,
    tileRange,
    color,
    opacity: splat.opacity,
    stateConvention: gs.stateConvention ?? 'unknown',
    usedCuda4DStateHelper: !!gs.usedCuda4DStateHelper,
    stateHelperVersion: gs.helperVersion ?? null
  };
}
