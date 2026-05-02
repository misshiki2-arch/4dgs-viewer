#!/usr/bin/env node
import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

const DEFAULTS = {
  splat: path.resolve(projectRoot, 'demo', 'scene_v2.splat4d'),
  meta: '/home/demo/work/outputs/sph_scene_4dgs/cuda_reference_named/iter_012000/000195_v26_meta.json',
  forwardJson: '/home/demo/work/json/step90_pc1_projection_debug.json',
  alignedJson: '/home/demo/work/json/step90_pc1_cuda_aligned_matrix_debug_fix1.json',
  output: '/home/demo/work/json/step90_representative_splat_projection_covariance_debug.json',
  time: 32.0,
  width: 1280,
  height: 720,
  sampleCount: 32,
  includeIndex: 123
};

function parseArgs(argv) {
  const args = { ...DEFAULTS };
  const rest = [...argv];
  while (rest.length) {
    const key = rest.shift();
    if (!key) continue;
    if (key === '--splat') args.splat = path.resolve(process.cwd(), rest.shift());
    else if (key === '--meta') args.meta = path.resolve(process.cwd(), rest.shift());
    else if (key === '--forward-json') args.forwardJson = path.resolve(process.cwd(), rest.shift());
    else if (key === '--aligned-json') args.alignedJson = path.resolve(process.cwd(), rest.shift());
    else if (key === '--output') args.output = path.resolve(process.cwd(), rest.shift());
    else if (key === '--time') args.time = Number(rest.shift());
    else if (key === '--sample-count') args.sampleCount = Math.max(1, Number.parseInt(rest.shift(), 10));
    else if (key === '--include-index') args.includeIndex = Number.parseInt(rest.shift(), 10);
    else throw new Error(`Unknown argument: ${key}`);
  }
  return args;
}

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function nestedGet(obj, dotted, fallback = null) {
  let cur = obj;
  for (const key of dotted.split('.')) {
    if (cur && typeof cur === 'object' && key in cur) cur = cur[key];
    else return fallback;
  }
  return cur;
}

function transpose4(m) {
  return m[0].map((_, c) => m.map((row) => row[c]));
}

function matVec4(m, v) {
  return m.map((row) => row.reduce((sum, x, i) => sum + x * v[i], 0));
}

function rowVecMat4(v, m) {
  return [0, 1, 2, 3].map((c) => v.reduce((sum, x, r) => sum + x * m[r][c], 0));
}

function mat3Mul(a, b) {
  const out = Array.from({ length: 3 }, () => Array(3).fill(0));
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      for (let k = 0; k < 3; k++) out[r][c] += a[r][k] * b[k][c];
    }
  }
  return out;
}

function mat3Transpose(m) {
  return [
    [m[0][0], m[1][0], m[2][0]],
    [m[0][1], m[1][1], m[2][1]],
    [m[0][2], m[1][2], m[2][2]]
  ];
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function ndc2Pix(v, size) {
  return ((v + 1.0) * size - 1.0) * 0.5;
}

function projectFull(point, fullProj, width, height) {
  const clip = rowVecMat4([point[0], point[1], point[2], 1], fullProj);
  const invW = 1 / (clip[3] + 1e-7);
  const ndc = [clip[0] * invW, clip[1] * invW, clip[2] * invW];
  return {
    clip,
    ndc,
    pixel: [ndc2Pix(ndc[0], width), ndc2Pix(ndc[1], height)]
  };
}

function projectThree(point, view, projection, width, height) {
  const viewPos = matVec4(view, [point[0], point[1], point[2], 1]);
  const clip = matVec4(projection, viewPos);
  const invW = 1 / (clip[3] + 1e-7);
  const ndc = [clip[0] * invW, clip[1] * invW, clip[2] * invW];
  return {
    view: viewPos,
    clip,
    ndc,
    pixel: [ndc2Pix(ndc[0], width), ndc2Pix(ndc[1], height)]
  };
}

function computeCov2DLikeCuda({ point, cov3, opacity, viewMatrix, fx, fy, tanFovX, tanFovY, width, height, fullProj = null }) {
  const view = matVec4(viewMatrix, [point[0], point[1], point[2], 1]);
  const z = view[2];
  if (!Number.isFinite(z) || z <= 0.2) {
    return { culled: true, cullReason: 'z-near-or-invalid', viewSpace: view };
  }

  const txtzRaw = view[0] / z;
  const tytzRaw = view[1] / z;
  const txtz = Math.min(1.3 * tanFovX, Math.max(-1.3 * tanFovX, txtzRaw));
  const tytz = Math.min(1.3 * tanFovY, Math.max(-1.3 * tanFovY, tytzRaw));
  const tx = txtz * z;
  const ty = tytz * z;
  const j = [
    [fx / z, 0, -(fx * tx) / (z * z)],
    [0, fy / z, -(fy * ty) / (z * z)],
    [0, 0, 0]
  ];
  const w = [
    [viewMatrix[0][0], viewMatrix[0][1], viewMatrix[0][2]],
    [viewMatrix[1][0], viewMatrix[1][1], viewMatrix[1][2]],
    [viewMatrix[2][0], viewMatrix[2][1], viewMatrix[2][2]]
  ];
  const tm = mat3Mul(w, j);
  const cov = mat3Mul(mat3Transpose(tm), mat3Mul(cov3, tm));
  cov[0][0] += 0.3;
  cov[1][1] += 0.3;
  const a = cov[0][0];
  const b = cov[0][1];
  const c = cov[1][1];
  const det = a * c - b * b;
  if (!Number.isFinite(det) || det <= 0) {
    return { culled: true, cullReason: 'projected-covariance-singular', viewSpace: view, covariance2D: [[a, b], [b, c]], det };
  }
  const inv = 1 / det;
  const conic = [c * inv, -b * inv, a * inv];
  const mid = 0.5 * (a + c);
  const lambdaRoot = Math.sqrt(Math.max(0.1, mid * mid - det));
  const lambda1 = mid + lambdaRoot;
  const lambda2 = mid - lambdaRoot;
  const radius = Math.ceil(3 * Math.sqrt(Math.max(lambda1, lambda2)));
  if (radius <= 0.4 || radius > 4096) {
    return { culled: true, cullReason: radius > 4096 ? 'radius-too-large' : 'radius-too-small', viewSpace: view, covariance2D: [[a, b], [b, c]], conic, det, radius };
  }

  const center = fullProj
    ? projectFull(point, fullProj, width, height).pixel
    : [fx * (view[0] / z) + width * 0.5, fy * (view[1] / z) + height * 0.5];
  return {
    culled: false,
    cullReason: 'none',
    viewSpace: view,
    txtzRaw,
    tytzRaw,
    txtz,
    tytz,
    jacobian: j,
    viewRotation: w,
    covariance2D: [[a, b], [b, c]],
    conic,
    det,
    mid,
    lambda1,
    lambda2,
    radius,
    depth: z,
    opacity,
    centerPx: center,
    rasterRect: [
      Math.max(0, Math.floor(center[0] - Math.max(1, radius))),
      Math.max(0, Math.floor(center[1] - Math.max(1, radius))),
      Math.min(width - 1, Math.ceil(center[0] + Math.max(1, radius))),
      Math.min(height - 1, Math.ceil(center[1] + Math.max(1, radius)))
    ]
  };
}

function computeThreeLikeViewer({ point, cov3, opacity, viewMatrix, projectionMatrix, fovDeg, aspect, width, height }) {
  const view = matVec4(viewMatrix, [point[0], point[1], point[2], 1]);
  const z = -view[2];
  if (!Number.isFinite(z) || z <= 1e-6) {
    return { culled: true, cullReason: 'three-z-not-positive', viewSpace: view };
  }
  const tanFovY = Math.tan((fovDeg * Math.PI / 180) * 0.5);
  const tanFovX = tanFovY * aspect;
  const txtzRaw = view[0] / z;
  const tytzRaw = view[1] / z;
  const txtz = Math.min(1.3 * tanFovX, Math.max(-1.3 * tanFovX, txtzRaw));
  const tytz = Math.min(1.3 * tanFovY, Math.max(-1.3 * tanFovY, tytzRaw));
  const tx = txtz * z;
  const ty = tytz * z;
  const fy = height / (2 * tanFovY);
  const fx = width / (2 * tanFovX);
  const j = [
    [fx / z, 0, -(fx * tx) / (z * z)],
    [0, fy / z, -(fy * ty) / (z * z)],
    [0, 0, 0]
  ];
  const w = [
    [viewMatrix[0][0], viewMatrix[0][1], viewMatrix[0][2]],
    [viewMatrix[1][0], viewMatrix[1][1], viewMatrix[1][2]],
    [viewMatrix[2][0], viewMatrix[2][1], viewMatrix[2][2]]
  ];
  const tm = mat3Mul(w, j);
  const cov = mat3Mul(mat3Transpose(tm), mat3Mul(cov3, tm));
  cov[0][0] += 0.3;
  cov[1][1] += 0.3;
  const a = cov[0][0];
  const b = cov[0][1];
  const c = cov[1][1];
  const det = a * c - b * b;
  if (!Number.isFinite(det) || det <= 0) {
    return { culled: true, cullReason: 'three-projected-covariance-singular', viewSpace: view, covariance2D: [[a, b], [b, c]], det };
  }
  const inv = 1 / det;
  const conic = [c * inv, -b * inv, a * inv];
  const mid = 0.5 * (a + c);
  const lambdaRoot = Math.sqrt(Math.max(0.1, mid * mid - det));
  const lambda1 = mid + lambdaRoot;
  const lambda2 = mid - lambdaRoot;
  const radius = Math.ceil(3 * Math.sqrt(Math.max(lambda1, lambda2)));
  if (radius <= 0.4 || radius > 4096) {
    return { culled: true, cullReason: radius > 4096 ? 'three-radius-too-large' : 'three-radius-too-small', viewSpace: view, covariance2D: [[a, b], [b, c]], conic, det, radius };
  }
  const center = projectThree(point, viewMatrix, projectionMatrix, width, height).pixel;
  return {
    culled: false,
    cullReason: 'none',
    viewSpace: view,
    txtzRaw,
    tytzRaw,
    txtz,
    tytz,
    jacobian: j,
    viewRotation: w,
    covariance2D: [[a, b], [b, c]],
    conic,
    det,
    mid,
    lambda1,
    lambda2,
    radius,
    depth: z,
    opacity,
    centerPx: center,
    rasterRect: [
      Math.max(0, Math.floor(center[0] - Math.max(1, radius))),
      Math.max(0, Math.floor(center[1] - Math.max(1, radius))),
      Math.min(width - 1, Math.ceil(center[0] + Math.max(1, radius))),
      Math.min(height - 1, Math.ceil(center[1] + Math.max(1, radius)))
    ]
  };
}

function pixelDelta(a, b) {
  if (!a?.centerPx || !b?.centerPx) return null;
  return [b.centerPx[0] - a.centerPx[0], b.centerPx[1] - a.centerPx[1]];
}

function covSummaryDelta(a, b) {
  if (!a?.covariance2D || !b?.covariance2D) return null;
  return [
    [b.covariance2D[0][0] - a.covariance2D[0][0], b.covariance2D[0][1] - a.covariance2D[0][1]],
    [b.covariance2D[1][0] - a.covariance2D[1][0], b.covariance2D[1][1] - a.covariance2D[1][1]]
  ];
}

function rawSplatFields(raw, i) {
  const take = (arr, dim) => Array.from(arr.subarray(i * dim, i * dim + dim));
  return {
    xyz: take(raw.xyz, raw.xyzDim),
    rotation: take(raw.rotation, raw.rotationDim),
    rotation_r: take(raw.rotation_r, raw.rotationRDim),
    scale_xyz: take(raw.scale_xyz, raw.scaleXYZDim),
    f_dc: take(raw.f_dc, raw.fdcDim),
    opacityRaw: take(raw.opacity, raw.opacityDim),
    t: take(raw.t, raw.tDim),
    scale_t: take(raw.scale_t, raw.scaleTDim)
  };
}

function temporalPass(raw, i, timestamp, sigmaScale = 1, threshold = 3) {
  if (!raw.t || !raw.scale_t || raw.tDim <= 0 || raw.scaleTDim <= 0) return true;
  const t = raw.t[i * raw.tDim];
  const s = raw.scale_t[i * raw.scaleTDim] * sigmaScale;
  if (!Number.isFinite(t) || !Number.isFinite(s) || s <= 0) return true;
  return Math.abs(timestamp - t) <= threshold * s;
}

function upsertBest(map, key, candidate, compare) {
  const prev = map.get(key);
  if (!prev || compare(candidate, prev) < 0) map.set(key, candidate);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const parserMod = await import(pathToFileURL(path.resolve(projectRoot, 'demo/js/splat4d_parser_v2.js')).href);
  const mathMod = await import(pathToFileURL(path.resolve(projectRoot, 'demo/js/rot4d_math.js')).href);
  const shMod = await import(pathToFileURL(path.resolve(projectRoot, 'demo/js/sh_eval.js')).href);

  const rawBuffer = fs.readFileSync(args.splat);
  const raw = parserMod.parseSplat4DV2(rawBuffer.buffer.slice(rawBuffer.byteOffset, rawBuffer.byteOffset + rawBuffer.byteLength));
  const meta = readJson(args.meta);
  const forwardJson = readJson(args.forwardJson);
  const alignedJson = readJson(args.alignedJson);

  const width = Number(meta.width || args.width);
  const height = Number(meta.height || args.height);
  const fx = Number(meta.fx);
  const fy = Number(meta.fy);
  const tanFovX = Math.tan(Number(meta.FoVx) * 0.5);
  const tanFovY = Math.tan(Number(meta.FoVy) * 0.5);
  const fullProj = meta.full_proj_transform;
  const cudaView = transpose4(meta.world_view_transform);
  const forwardView = nestedGet(forwardJson, 'cameraDebugState.camera.matrixWorldInverse');
  const forwardProj = nestedGet(forwardJson, 'cameraDebugState.camera.projectionMatrix');
  const forwardFov = nestedGet(forwardJson, 'cameraDebugState.camera.fov', Number(meta.FoVy) * 180 / Math.PI);
  const forwardAspect = nestedGet(forwardJson, 'cameraDebugState.camera.aspect', width / height);
  const alignedView = nestedGet(alignedJson, 'bundle.deterministicState.cudaAlignedScreenSpaceCamera.cudaAlignedViewMatrix')
    || nestedGet(alignedJson, 'cameraDebugState.deterministicState.cudaAlignedScreenSpaceCamera.cudaAlignedViewMatrix')
    || cudaView;
  const cameraCenter = meta.camera_center;
  const camPos = { x: cameraCenter[0], y: cameraCenter[1], z: cameraCenter[2] };
  const flagsViewerCurrent = { nativeRot4d: false, nativeMarginal: false };
  const flagsCudaNative = { nativeRot4d: true, nativeMarginal: true };
  const common = {
    timestamp: args.time,
    scalingModifier: 1.0,
    sigmaScale: 1.0,
    prefilterVar: 0.0,
    useRot4d: true,
    useSH: true,
    forceSh3d: false,
    timeDuration: 33.0
  };

  const candidates = [];
  let temporalPassed = 0;
  let nativeVisible = 0;
  let viewerVisible = 0;
  const best = new Map();
  const center = [width * 0.5, height * 0.5];
  for (let i = 0; i < raw.N; i++) {
    if (!temporalPass(raw, i, args.time, common.sigmaScale, 3.0)) continue;
    temporalPassed++;
    const cudaState = mathMod.computeGaussianState(raw, i, args.time, common.scalingModifier, common.sigmaScale, common.prefilterVar, common.useRot4d, flagsCudaNative);
    if (!cudaState) continue;
    const cudaScreen = computeCov2DLikeCuda({
      point: cudaState.pos,
      cov3: cudaState.cov3,
      opacity: cudaState.opacity,
      viewMatrix: cudaView,
      fx,
      fy,
      tanFovX,
      tanFovY,
      width,
      height,
      fullProj
    });
    if (cudaScreen.culled) continue;
    nativeVisible++;
    const viewerState = mathMod.computeGaussianState(raw, i, args.time, common.scalingModifier, common.sigmaScale, common.prefilterVar, common.useRot4d, flagsViewerCurrent);
    const viewerScreen = viewerState
      ? computeThreeLikeViewer({
          point: viewerState.pos,
          cov3: viewerState.cov3,
          opacity: viewerState.opacity,
          viewMatrix: forwardView,
          projectionMatrix: forwardProj,
          fovDeg: forwardFov,
          aspect: forwardAspect,
          width,
          height
        })
      : { culled: true, cullReason: 'viewer-current-gaussian-state-null' };
    if (!viewerScreen.culled) viewerVisible++;
    const alignedScreen = viewerState
      ? computeCov2DLikeCuda({
          point: viewerState.pos,
          cov3: viewerState.cov3,
          opacity: viewerState.opacity,
          viewMatrix: alignedView,
          fx,
          fy,
          tanFovX,
          tanFovY,
          width,
          height,
          fullProj: null
        })
      : { culled: true, cullReason: 'viewer-current-gaussian-state-null' };
    const color = shMod.evalSHColor(raw, i, camPos, cudaState.pos, args.time, common.timeDuration, common.useSH, common.forceSh3d);
    const distCenter = Math.hypot(cudaScreen.centerPx[0] - center[0], cudaScreen.centerPx[1] - center[1]);
    const item = {
      index: i,
      selectionScores: {
        distanceToScreenCenter: distCenter,
        cudaRadius: cudaScreen.radius,
        cudaOpacity: cudaScreen.opacity,
        cudaDepth: cudaScreen.depth
      },
      raw: rawSplatFields(raw, i),
      color,
      cudaNativeState: cudaState,
      viewerCurrentState: viewerState,
      projections: {
        cudaNative: cudaScreen,
        viewerForwardCurrent: viewerScreen,
        viewerCudaAlignedCurrent: alignedScreen
      }
    };
    item.deltas = {
      cudaToViewerForwardCenterPx: pixelDelta(cudaScreen, viewerScreen),
      cudaToViewerCudaAlignedCenterPx: pixelDelta(cudaScreen, alignedScreen),
      cudaToViewerForwardCovariance2D: covSummaryDelta(cudaScreen, viewerScreen),
      cudaToViewerCudaAlignedCovariance2D: covSummaryDelta(cudaScreen, alignedScreen),
      radius: {
        viewerForwardMinusCuda: Number.isFinite(viewerScreen.radius) ? viewerScreen.radius - cudaScreen.radius : null,
        viewerCudaAlignedMinusCuda: Number.isFinite(alignedScreen.radius) ? alignedScreen.radius - cudaScreen.radius : null
      }
    };
    candidates.push(item);
    upsertBest(best, 'nearest_center', item, (a, b) => a.selectionScores.distanceToScreenCenter - b.selectionScores.distanceToScreenCenter);
    upsertBest(best, 'largest_radius', item, (a, b) => b.selectionScores.cudaRadius - a.selectionScores.cudaRadius);
    upsertBest(best, 'highest_opacity', item, (a, b) => b.selectionScores.cudaOpacity - a.selectionScores.cudaOpacity);
  }

  candidates.sort((a, b) => a.selectionScores.cudaDepth - b.selectionScores.cudaDepth);
  const selected = new Map();
  const add = (reason, item) => {
    if (!item) return;
    const existing = selected.get(item.index);
    selected.set(item.index, {
      ...item,
      selectionReasons: existing ? [...new Set([...existing.selectionReasons, reason])] : [reason]
    });
  };

  const fixedState = mathMod.computeGaussianState(raw, args.includeIndex, args.time, common.scalingModifier, common.sigmaScale, common.prefilterVar, common.useRot4d, flagsCudaNative);
  if (fixedState) {
    const point = fixedState.pos;
    const cudaScreen = computeCov2DLikeCuda({ point, cov3: fixedState.cov3, opacity: fixedState.opacity, viewMatrix: cudaView, fx, fy, tanFovX, tanFovY, width, height, fullProj });
    add('fixed-index-123', candidates.find((x) => x.index === args.includeIndex) ?? {
      index: args.includeIndex,
      selectionScores: { distanceToScreenCenter: null, cudaRadius: cudaScreen.radius ?? null, cudaOpacity: fixedState.opacity, cudaDepth: cudaScreen.depth ?? null },
      raw: rawSplatFields(raw, args.includeIndex),
      color: shMod.evalSHColor(raw, args.includeIndex, camPos, point, args.time, common.timeDuration, common.useSH, common.forceSh3d),
      cudaNativeState: fixedState,
      viewerCurrentState: mathMod.computeGaussianState(raw, args.includeIndex, args.time, common.scalingModifier, common.sigmaScale, common.prefilterVar, common.useRot4d, flagsViewerCurrent),
      projections: { cudaNative: cudaScreen },
      deltas: {}
    });
  }
  for (const [reason, item] of best) add(reason, item);
  const quantiles = [0.1, 0.25, 0.5, 0.75, 0.9];
  for (const q of quantiles) add(`depth_quantile_${q}`, candidates[Math.max(0, Math.min(candidates.length - 1, Math.floor((candidates.length - 1) * q)))]);
  for (const item of [...candidates].sort((a, b) => a.selectionScores.distanceToScreenCenter - b.selectionScores.distanceToScreenCenter).slice(0, 8)) add('screen-center-top8', item);
  for (const item of [...candidates].sort((a, b) => b.selectionScores.cudaRadius - a.selectionScores.cudaRadius).slice(0, 8)) add('radius-top8', item);
  for (const item of [...candidates].sort((a, b) => b.selectionScores.cudaOpacity - a.selectionScores.cudaOpacity).slice(0, 8)) add('opacity-top8', item);

  const selectedItems = [...selected.values()].slice(0, args.sampleCount);
  const summary = {
    selectedCount: selectedItems.length,
    rawCount: raw.N,
    temporalPassed,
    cudaNativeVisible: nativeVisible,
    viewerCurrentVisible: viewerVisible,
    selectionPolicy: [
      'include fixed index 123 when available',
      'include nearest screen-center, largest radius, highest opacity',
      'include depth quantiles and top screen-center/radius/opacity samples',
      'CUDA/native 4D state is used as the selection reference because CUDA Reference uses rasterizer-native rot_4d'
    ],
    formulaNotes: {
      cudaReference: 'CUDA rasterizer uses native 4D rotation in computeCov3D_conditional, conditional mean offset, cov2D, conic, radius.',
      viewerCurrent: 'Viewer default uses useRot4d=true but nativeRot4d=false unless toggled, so this report compares viewer-current old rot4d against CUDA/native.',
      centerProjectionStatus: 'Earlier Step90 projection debug showed center projection is aligned for CUDA view+intrinsics and Viewer cuda-aligned intrinsics; this report focuses on covariance/conic/radius.'
    }
  };
  const result = {
    schemaVersion: 'step90-representative-splat-projection-covariance-v1',
    inputs: args,
    camera: {
      width,
      height,
      fx,
      fy,
      FoVx: meta.FoVx,
      FoVy: meta.FoVy,
      tanFovX,
      tanFovY,
      cameraCenter,
      cudaViewMatrix: cudaView,
      cudaFullProjTransform: fullProj,
      viewerForwardViewMatrix: forwardView,
      viewerForwardProjectionMatrix: forwardProj,
      viewerCudaAlignedViewMatrix: alignedView
    },
    summary,
    selectedIndices: selectedItems.map((x) => ({ index: x.index, reasons: x.selectionReasons })),
    splats: selectedItems
  };
  fs.mkdirSync(path.dirname(args.output), { recursive: true });
  fs.writeFileSync(args.output, JSON.stringify(result, null, 2));
  console.log(JSON.stringify({
    output: args.output,
    selectedCount: selectedItems.length,
    selectedIndices: result.selectedIndices,
    summary
  }, null, 2));
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
