import * as THREE from 'three';

export const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
export const sigmoid = x => 1 / (1 + Math.exp(-x));

export function fitCameraToRaw(raw, controls, camera) {
  if (!raw || raw.N === 0) return;

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (let i = 0; i < raw.N; i++) {
    const x = raw.xyz[i * raw.xyzDim + 0];
    const y = raw.xyz[i * raw.xyzDim + 1];
    const z = raw.xyz[i * raw.xyzDim + 2];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }

  const cx = 0.5 * (minX + maxX);
  const cy = 0.5 * (minY + maxY);
  const cz = 0.5 * (minZ + maxZ);
  const r = Math.max(maxX - minX, maxY - minY, maxZ - minZ);

  controls.target.set(cx, cy, cz);
  camera.position.set(cx + 1.5 * r, cy + 0.8 * r, cz + 0.8 * r);
  camera.near = Math.max(0.001, r / 1000);
  camera.far = Math.max(1000, r * 10);
  camera.updateProjectionMatrix();
  controls.update();
}

export function normalizeQuat4(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

export function buildRotation3(qIn) {
  const q = normalizeQuat4(qIn);
  const r = q[0], x = q[1], y = q[2], z = q[3];
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y)],
    [2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x)],
    [2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)]
  ];
}

export function mat4Mul(A, B) {
  const C = Array.from({ length: 4 }, () => Array(4).fill(0));
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      for (let k = 0; k < 4; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return C;
}

export function mat3Mul(A, B) {
  const C = Array.from({ length: 3 }, () => Array(3).fill(0));
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return C;
}

export function mat3Transpose(A) {
  return [[A[0][0], A[1][0], A[2][0]], [A[0][1], A[1][1], A[2][1]], [A[0][2], A[1][2], A[2][2]]];
}

export function mat4Transpose(A) {
  const R = Array.from({ length: 4 }, () => Array(4).fill(0));
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      R[i][j] = A[j][i];
    }
  }
  return R;
}

export function outer3(a, b) {
  return [
    [a[0] * b[0], a[0] * b[1], a[0] * b[2]],
    [a[1] * b[0], a[1] * b[1], a[1] * b[2]],
    [a[2] * b[0], a[2] * b[1], a[2] * b[2]]
  ];
}

export function sub3(A, B) {
  return [
    [A[0][0] - B[0][0], A[0][1] - B[0][1], A[0][2] - B[0][2]],
    [A[1][0] - B[1][0], A[1][1] - B[1][1], A[1][2] - B[1][2]],
    [A[2][0] - B[2][0], A[2][1] - B[2][1], A[2][2] - B[2][2]]
  ];
}

export function mulScalar3(A, s) {
  return [
    [A[0][0] * s, A[0][1] * s, A[0][2] * s],
    [A[1][0] * s, A[1][1] * s, A[1][2] * s],
    [A[2][0] * s, A[2][1] * s, A[2][2] * s]
  ];
}

function flip2D(M) {
  return [
    [M[3][3], M[3][2], M[3][1], M[3][0]],
    [M[2][3], M[2][2], M[2][1], M[2][0]],
    [M[1][3], M[1][2], M[1][1], M[1][0]],
    [M[0][3], M[0][2], M[0][1], M[0][0]]
  ];
}

function buildRotation4DOld(qLIn, qRIn) {
  const [a, b, c, d] = normalizeQuat4(qLIn);
  const [p, q, r, s] = normalizeQuat4(qRIn);

  const Ml = [
    [a, -b, -c, -d],
    [b,  a, -d,  c],
    [c,  d,  a, -b],
    [d, -c,  b,  a]
  ];
  const Mr = [
    [ p,  q,  r,  s],
    [-q,  p, -s,  r],
    [-r,  s,  p, -q],
    [-s, -r,  q,  p]
  ];

  return flip2D(mat4Mul(Ml, Mr));
}

function buildRotation4DNative(qLIn, qRIn) {
  const [a, b, c, d] = normalizeQuat4(qLIn);
  const [p, q, r, s] = normalizeQuat4(qRIn);

  const Ml = [
    [ a,  b, -c,  d],
    [-b,  a,  d,  c],
    [ c, -d,  a,  b],
    [-d, -c, -b,  a]
  ];
  const Mr = [
    [ p,  q, -r, -s],
    [-q,  p,  s, -r],
    [ r, -s,  p, -q],
    [ s,  r,  q,  p]
  ];

  return mat4Mul(Mr, Ml);
}

export function getViewRotation3x3(camera) {
  camera.updateMatrixWorld(true);
  const e = camera.matrixWorldInverse.elements;
  return [[e[0], e[4], e[8]], [e[1], e[5], e[9]], [e[2], e[6], e[10]]];
}

export function computeGaussianState(raw, i, timestamp, scalingModifier, sigmaScale, prefilterVar, useRot4d, flags) {
  const pos0 = [raw.xyz[i * raw.xyzDim], raw.xyz[i * raw.xyzDim + 1], raw.xyz[i * raw.xyzDim + 2]];
  let opacity = sigmoid(raw.opacity[i * raw.opacityDim]);

  const scale = [
    Math.max(raw.scale_xyz[i * raw.scaleXYZDim] * scalingModifier, 1e-6),
    Math.max(raw.scale_xyz[i * raw.scaleXYZDim + 1] * scalingModifier, 1e-6),
    Math.max(raw.scale_xyz[i * raw.scaleXYZDim + 2] * scalingModifier, 1e-6)
  ];

  const tCenter = raw.tDim > 0 ? raw.t[i * raw.tDim] : 0;
  const scaleT = raw.scaleTDim > 0 ? Math.max(raw.scale_t[i * raw.scaleTDim] * scalingModifier * sigmaScale, 1e-6) : 1e-6;
  const dt = timestamp - tCenter;

  if (useRot4d && raw.rot4d && raw.rotationRDim >= 4 && raw.scaleTDim > 0) {
    const qL = [raw.rotation[i * raw.rotationDim], raw.rotation[i * raw.rotationDim + 1], raw.rotation[i * raw.rotationDim + 2], raw.rotation[i * raw.rotationDim + 3]];
    const qR = [raw.rotation_r[i * raw.rotationRDim], raw.rotation_r[i * raw.rotationRDim + 1], raw.rotation_r[i * raw.rotationRDim + 2], raw.rotation_r[i * raw.rotationRDim + 3]];

    const R4 = flags.nativeRot4d ? buildRotation4DNative(qL, qR) : buildRotation4DOld(qL, qR);

    const S = [
      [scale[0], 0, 0, 0],
      [0, scale[1], 0, 0],
      [0, 0, scale[2], 0],
      [0, 0, 0, scaleT]
    ];

    const M = mat4Mul(R4, S);
    const Sigma = mat4Mul(M, mat4Transpose(M));

    const cov_t = Sigma[3][3];
    const denom = flags.nativeMarginal
      ? ((prefilterVar > 0) ? (prefilterVar + cov_t) : cov_t)
      : cov_t;

    const marginal_t = Math.exp(-0.5 * dt * dt / Math.max(1e-8, denom));
    if (marginal_t <= 0.05) return null;
    opacity *= marginal_t;

    const cov11 = [
      [Sigma[0][0], Sigma[0][1], Sigma[0][2]],
      [Sigma[1][0], Sigma[1][1], Sigma[1][2]],
      [Sigma[2][0], Sigma[2][1], Sigma[2][2]]
    ];
    const cov12 = [Sigma[0][3], Sigma[1][3], Sigma[2][3]];
    const cond = sub3(cov11, mulScalar3(outer3(cov12, cov12), 1 / Math.max(1e-8, cov_t)));
    const delta = [cov12[0] / cov_t * dt, cov12[1] / cov_t * dt, cov12[2] / cov_t * dt];

    return {
      pos: [pos0[0] + delta[0], pos0[1] + delta[1], pos0[2] + delta[2]],
      cov3: cond,
      opacity
    };
  }

  const q = [raw.rotation[i * raw.rotationDim], raw.rotation[i * raw.rotationDim + 1], raw.rotation[i * raw.rotationDim + 2], raw.rotation[i * raw.rotationDim + 3]];
  const R = buildRotation3(q);
  const M = [
    [scale[0] * R[0][0], scale[0] * R[0][1], scale[0] * R[0][2]],
    [scale[1] * R[1][0], scale[1] * R[1][1], scale[1] * R[1][2]],
    [scale[2] * R[2][0], scale[2] * R[2][1], scale[2] * R[2][2]]
  ];
  const Sigma = mat3Mul(M, mat3Transpose(M));

  if (raw.scaleTDim > 0) {
    const denom = flags.nativeMarginal
      ? ((prefilterVar > 0) ? (prefilterVar + scaleT) : scaleT)
      : scaleT;
    const marginal_t = Math.exp(-0.5 * dt * dt / Math.max(1e-8, denom));
    if (marginal_t <= 0.05) return null;
    opacity *= marginal_t;
  }

  return { pos: pos0, cov3: Sigma, opacity };
}

export function computeScreenSplat(camera, pos, cov3, opacity, renderW, renderH) {
  const mv = new THREE.Vector3(pos[0], pos[1], pos[2]).applyMatrix4(camera.matrixWorldInverse);
  const zc = -mv.z;
  if (zc <= 1e-6) return null;

  const tanFovY = Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));
  const tanFovX = tanFovY * camera.aspect;
  let txtz = mv.x / zc;
  let tytz = mv.y / zc;
  txtz = clamp(txtz, -1.3 * tanFovX, 1.3 * tanFovX);
  tytz = clamp(tytz, -1.3 * tanFovY, 1.3 * tanFovY);

  const tx = txtz * zc;
  const ty = tytz * zc;
  const fy = renderH / (2 * tanFovY);
  const fx = renderW / (2 * tanFovX);

  const J = [
    [fx / zc, 0.0, -(fx * tx) / (zc * zc)],
    [0.0, fy / zc, -(fy * ty) / (zc * zc)],
    [0.0, 0.0, 0.0]
  ];

  const W = getViewRotation3x3(camera);
  const Tm = mat3Mul(W, J);
  let cov = mat3Mul(mat3Transpose(Tm), mat3Mul(cov3, Tm));
  cov[0][0] += 0.3;
  cov[1][1] += 0.3;

  const a = cov[0][0], b = cov[0][1], c = cov[1][1];
  const det = a * c - b * b;
  if (!isFinite(det) || det <= 0) return null;

  const inv = 1 / det;
  const conic = [c * inv, -b * inv, a * inv];

  const mid = 0.5 * (a + c);
  const lambda1 = mid + Math.sqrt(Math.max(0.1, mid * mid - det));
  const lambda2 = mid - Math.sqrt(Math.max(0.1, mid * mid - det));
  const radius = Math.ceil(3 * Math.sqrt(Math.max(lambda1, lambda2)));
  if (radius <= 0.4 || radius > 4096) return null;

  const clip = new THREE.Vector4(pos[0], pos[1], pos[2], 1.0)
    .applyMatrix4(camera.matrixWorldInverse)
    .applyMatrix4(camera.projectionMatrix);

  const p_w = 1 / (clip.w + 1e-7);
  const p_proj_x = clip.x * p_w;
  const p_proj_y = clip.y * p_w;
  const px = ((p_proj_x + 1.0) * renderW - 1.0) * 0.5;
  const py = ((p_proj_y + 1.0) * renderH - 1.0) * 0.5;

  return { px, py, conic, radius, depth: zc, opacity };
}
