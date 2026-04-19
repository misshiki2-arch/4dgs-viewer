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

function vecSub3(a, b) {
  return [
    a[0] - b[0],
    a[1] - b[1],
    a[2] - b[2]
  ];
}

function vecDot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function vecCross3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  ];
}

function vecLength3(v) {
  return Math.hypot(v[0], v[1], v[2]);
}

function normalizeVec3(v, fallback = [0, 0, 1]) {
  const len = vecLength3(v);
  if (!Number.isFinite(len) || len <= 1e-8) return fallback.slice();
  return [v[0] / len, v[1] / len, v[2] / len];
}

function cloneMat3(A) {
  return [
    [A[0][0], A[0][1], A[0][2]],
    [A[1][0], A[1][1], A[1][2]],
    [A[2][0], A[2][1], A[2][2]]
  ];
}

function cloneMat4(A) {
  return [
    [A[0][0], A[0][1], A[0][2], A[0][3]],
    [A[1][0], A[1][1], A[1][2], A[1][3]],
    [A[2][0], A[2][1], A[2][2], A[2][3]],
    [A[3][0], A[3][1], A[3][2], A[3][3]]
  ];
}

function cloneVec3(v) {
  return [v[0], v[1], v[2]];
}

function pickNumber(value, fallback) {
  return Number.isFinite(value) ? value : fallback;
}

function normalizeCompareInput(input = {}) {
  const camera = input?.camera ?? {};
  return {
    mean3D: Array.isArray(input?.mean3D) && input.mean3D.length >= 3
      ? [
          pickNumber(input.mean3D[0], 0.18),
          pickNumber(input.mean3D[1], -0.06),
          pickNumber(input.mean3D[2], -2.4)
        ]
      : [0.18, -0.06, -2.4],
    scale_xyz: Array.isArray(input?.scale_xyz) && input.scale_xyz.length >= 3
      ? [
          Math.max(pickNumber(input.scale_xyz[0], 0.03), 1e-6),
          Math.max(pickNumber(input.scale_xyz[1], 0.28), 1e-6),
          Math.max(pickNumber(input.scale_xyz[2], 0.012), 1e-6)
        ]
      : [0.03, 0.28, 0.012],
    rotation: Array.isArray(input?.rotation) && input.rotation.length >= 4
      ? [
          pickNumber(input.rotation[0], 0.94),
          pickNumber(input.rotation[1], 0.12),
          pickNumber(input.rotation[2], -0.22),
          pickNumber(input.rotation[3], 0.24)
        ]
      : [0.94, 0.12, -0.22, 0.24],
    rotation_r: Array.isArray(input?.rotation_r) && input.rotation_r.length >= 4
      ? [
          pickNumber(input.rotation_r[0], 0.88),
          pickNumber(input.rotation_r[1], -0.19),
          pickNumber(input.rotation_r[2], 0.31),
          pickNumber(input.rotation_r[3], -0.27)
        ]
      : [0.88, -0.19, 0.31, -0.27],
    t_center: pickNumber(input?.t_center, 8.0),
    timestamp: pickNumber(input?.timestamp, 8.65),
    scale_t: Math.max(pickNumber(input?.scale_t, 0.18), 1e-6),
    prefilter_var: Math.max(pickNumber(input?.prefilter_var, 0.12), 0.0),
    opacity: clamp(pickNumber(input?.opacity, 0.35), 0.0, 0.99),
    scalingModifier: Math.max(pickNumber(input?.scalingModifier, 1.0), 1e-6),
    sigmaScale: Math.max(pickNumber(input?.sigmaScale, 1.0), 1e-6),
    useRot4d: input?.useRot4d !== false,
    nativeRot4d: !!input?.nativeRot4d,
    nativeMarginal: input?.nativeMarginal !== false,
    renderW: Math.max(1, pickNumber(input?.renderW, 1280) | 0),
    renderH: Math.max(1, pickNumber(input?.renderH, 720) | 0),
    camera: {
      fovDeg: pickNumber(camera?.fovDeg, 50),
      aspect: pickNumber(camera?.aspect, 1280 / 720),
      near: Math.max(pickNumber(camera?.near, 0.01), 1e-6),
      far: Math.max(pickNumber(camera?.far, 100.0), 1e-3),
      position: Array.isArray(camera?.position) && camera.position.length >= 3
        ? [
            pickNumber(camera.position[0], 0),
            pickNumber(camera.position[1], 0),
            pickNumber(camera.position[2], 0)
          ]
        : [0, 0, 0],
      target: Array.isArray(camera?.target) && camera.target.length >= 3
        ? [
            pickNumber(camera.target[0], 0),
            pickNumber(camera.target[1], 0),
            pickNumber(camera.target[2], -1)
          ]
        : [0, 0, -1],
      up: Array.isArray(camera?.up) && camera.up.length >= 3
        ? [
            pickNumber(camera.up[0], 0),
            pickNumber(camera.up[1], 1),
            pickNumber(camera.up[2], 0)
          ]
        : [0, 1, 0]
    }
  };
}

export const DEFAULT_SINGLE_SPLAT_COMPARE_INPUT = Object.freeze(normalizeCompareInput());

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

function buildDebugCamera(compareInput) {
  const camera = new THREE.PerspectiveCamera(
    compareInput.camera.fovDeg,
    compareInput.camera.aspect,
    compareInput.camera.near,
    compareInput.camera.far
  );
  camera.position.set(
    compareInput.camera.position[0],
    compareInput.camera.position[1],
    compareInput.camera.position[2]
  );
  camera.up.set(
    compareInput.camera.up[0],
    compareInput.camera.up[1],
    compareInput.camera.up[2]
  );
  camera.lookAt(
    compareInput.camera.target[0],
    compareInput.camera.target[1],
    compareInput.camera.target[2]
  );
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  return camera;
}

function buildScalingMatrix4(scaleXYZ, scaleT) {
  return [
    [scaleXYZ[0], 0, 0, 0],
    [0, scaleXYZ[1], 0, 0],
    [0, 0, scaleXYZ[2], 0],
    [0, 0, 0, scaleT]
  ];
}

function extractCov11FromSigma4(Sigma) {
  return [
    [Sigma[0][0], Sigma[0][1], Sigma[0][2]],
    [Sigma[1][0], Sigma[1][1], Sigma[1][2]],
    [Sigma[2][0], Sigma[2][1], Sigma[2][2]]
  ];
}

function extractCov12FromSigma4(Sigma) {
  return [Sigma[0][3], Sigma[1][3], Sigma[2][3]];
}

function computeGaussianStateDebugFromInput(compareInput) {
  const pos0 = cloneVec3(compareInput.mean3D);
  const baseOpacity = compareInput.opacity;
  const scale = [
    Math.max(compareInput.scale_xyz[0] * compareInput.scalingModifier, 1e-6),
    Math.max(compareInput.scale_xyz[1] * compareInput.scalingModifier, 1e-6),
    Math.max(compareInput.scale_xyz[2] * compareInput.scalingModifier, 1e-6)
  ];
  const scaleT = Math.max(compareInput.scale_t * compareInput.scalingModifier * compareInput.sigmaScale, 1e-6);
  const dt = compareInput.timestamp - compareInput.t_center;
  const qLNorm = normalizeQuat4(compareInput.rotation);
  const qRNorm = normalizeQuat4(compareInput.rotation_r);

  const baseResult = {
    qL_norm: qLNorm,
    qR_norm: qRNorm,
    dt,
    scale_xyz: cloneVec3(scale),
    scale_t: scaleT,
    opacity_base: baseOpacity,
    cov_t: null,
    marginal_denom: null,
    marginal_t: null,
    cov12: null,
    mean_offset: [0, 0, 0],
    conditional_cov3x3: null,
    pos_conditional: cloneVec3(pos0),
    opacity_after_marginal: baseOpacity,
    R4: null,
    Sigma4: null,
    culled: false,
    cullReason: 'none'
  };

  if (compareInput.useRot4d) {
    const R4 = compareInput.nativeRot4d
      ? buildRotation4DNative(qLNorm, qRNorm)
      : buildRotation4DOld(qLNorm, qRNorm);
    const S = buildScalingMatrix4(scale, scaleT);
    // CUDA builds the 4D covariance from M = S * R and Sigma = M^T * M.
    const M = mat4Mul(S, R4);
    const Sigma = mat4Mul(mat4Transpose(M), M);
    const cov_t = Sigma[3][3];
    const marginal_denom = compareInput.nativeMarginal
      ? ((compareInput.prefilter_var > 0) ? (compareInput.prefilter_var + cov_t) : cov_t)
      : cov_t;
    const marginal_t = Math.exp(-0.5 * dt * dt / Math.max(1e-8, marginal_denom));
    const cov12 = extractCov12FromSigma4(Sigma);
    const meanOffset = [
      cov12[0] / Math.max(1e-8, cov_t) * dt,
      cov12[1] / Math.max(1e-8, cov_t) * dt,
      cov12[2] / Math.max(1e-8, cov_t) * dt
    ];
    const cov11 = extractCov11FromSigma4(Sigma);
    const cond = sub3(cov11, mulScalar3(outer3(cov12, cov12), 1 / Math.max(1e-8, cov_t)));

    return {
      ...baseResult,
      cov_t,
      marginal_denom,
      marginal_t,
      cov12,
      mean_offset: meanOffset,
      conditional_cov3x3: cloneMat3(cond),
      pos_conditional: [
        pos0[0] + meanOffset[0],
        pos0[1] + meanOffset[1],
        pos0[2] + meanOffset[2]
      ],
      opacity_after_marginal: baseOpacity * marginal_t,
      R4: cloneMat4(R4),
      Sigma4: cloneMat4(Sigma),
      culled: marginal_t <= 0.05,
      cullReason: marginal_t <= 0.05 ? 'temporal-marginal-below-threshold' : 'none'
    };
  }

  const R3 = buildRotation3(qLNorm);
  const M = [
    [scale[0] * R3[0][0], scale[0] * R3[0][1], scale[0] * R3[0][2]],
    [scale[1] * R3[1][0], scale[1] * R3[1][1], scale[1] * R3[1][2]],
    [scale[2] * R3[2][0], scale[2] * R3[2][1], scale[2] * R3[2][2]]
  ];
  const Sigma = mat3Mul(M, mat3Transpose(M));
  const marginal_denom = compareInput.nativeMarginal
    ? ((compareInput.prefilter_var > 0) ? (compareInput.prefilter_var + scaleT) : scaleT)
    : scaleT;
  const marginal_t = Math.exp(-0.5 * dt * dt / Math.max(1e-8, marginal_denom));
  return {
    ...baseResult,
    cov_t: scaleT,
    marginal_denom,
    marginal_t,
    conditional_cov3x3: cloneMat3(Sigma),
    opacity_after_marginal: baseOpacity * marginal_t,
    culled: marginal_t <= 0.05,
    cullReason: marginal_t <= 0.05 ? 'temporal-marginal-below-threshold' : 'none'
  };
}

function computeScreenSplatDebugDetails(camera, pos, cov3, opacity, renderW, renderH) {
  const worldPoint = new THREE.Vector3(pos[0], pos[1], pos[2]);
  const mv = worldPoint.clone().applyMatrix4(camera.matrixWorldInverse);
  const zc = -mv.z;
  if (zc <= 1e-6) {
    return {
      culled: true,
      cullReason: 'camera-space-z-too-small',
      view_space_pos: [mv.x, mv.y, mv.z],
      projected_cov2x2: null,
      conic: null,
      det: null,
      mid: null,
      lambda1: null,
      lambda2: null,
      radius: null,
      opacity
    };
  }

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
  const cov = mat3Mul(mat3Transpose(Tm), mat3Mul(cov3, Tm));
  cov[0][0] += 0.3;
  cov[1][1] += 0.3;

  const a = cov[0][0];
  const b = cov[0][1];
  const c = cov[1][1];
  const det = a * c - b * b;
  if (!Number.isFinite(det) || det <= 0) {
    return {
      culled: true,
      cullReason: 'projected-covariance-singular',
      view_space_pos: [mv.x, mv.y, mv.z],
      projected_cov2x2: [[a, b], [b, c]],
      conic: null,
      det,
      mid: null,
      lambda1: null,
      lambda2: null,
      radius: null,
      opacity
    };
  }

  const inv = 1 / det;
  const conic = [c * inv, -b * inv, a * inv];
  const mid = 0.5 * (a + c);
  const lambda1 = mid + Math.sqrt(Math.max(0.1, mid * mid - det));
  const lambda2 = mid - Math.sqrt(Math.max(0.1, mid * mid - det));
  const radius = Math.ceil(3 * Math.sqrt(Math.max(lambda1, lambda2)));
  if (radius <= 0.4 || radius > 4096) {
    return {
      culled: true,
      cullReason: radius > 4096 ? 'radius-too-large' : 'radius-too-small',
      view_space_pos: [mv.x, mv.y, mv.z],
      projected_cov2x2: [[a, b], [b, c]],
      conic,
      det,
      mid,
      lambda1,
      lambda2,
      radius,
      opacity
    };
  }

  return {
    culled: false,
    cullReason: 'none',
    view_space_pos: [mv.x, mv.y, mv.z],
    projected_cov2x2: [[a, b], [b, c]],
    conic,
    det,
    mid,
    lambda1,
    lambda2,
    radius,
    opacity
  };
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

    const S = buildScalingMatrix4(scale, scaleT);
    const M = mat4Mul(S, R4);
    const Sigma = mat4Mul(mat4Transpose(M), M);

    const cov_t = Sigma[3][3];
    const denom = flags.nativeMarginal
      ? ((prefilterVar > 0) ? (prefilterVar + cov_t) : cov_t)
      : cov_t;

    const marginal_t = Math.exp(-0.5 * dt * dt / Math.max(1e-8, denom));
    if (marginal_t <= 0.05) return null;
    opacity *= marginal_t;

    const cov11 = extractCov11FromSigma4(Sigma);
    const cov12 = extractCov12FromSigma4(Sigma);
    const cond = sub3(cov11, mulScalar3(outer3(cov12, cov12), 1 / Math.max(1e-8, cov_t)));
    const delta = [
      cov12[0] / Math.max(1e-8, cov_t) * dt,
      cov12[1] / Math.max(1e-8, cov_t) * dt,
      cov12[2] / Math.max(1e-8, cov_t) * dt
    ];

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

export function computeGaussianDebugState(input = {}) {
  const compareInput = normalizeCompareInput(input);
  const camera = buildDebugCamera(compareInput);
  const gaussian = computeGaussianStateDebugFromInput(compareInput);
  const screen = gaussian.culled
    ? {
        culled: true,
        cullReason: gaussian.cullReason,
        view_space_pos: null,
        projected_cov2x2: null,
        conic: null,
        det: null,
        mid: null,
        lambda1: null,
        lambda2: null,
        radius: null,
        opacity: gaussian.opacity_after_marginal
      }
    : computeScreenSplatDebugDetails(
        camera,
        gaussian.pos_conditional,
        gaussian.conditional_cov3x3,
        gaussian.opacity_after_marginal,
        compareInput.renderW,
        compareInput.renderH
      );

  return {
    schemaVersion: 'step74-single-splat-compare-v1',
    source: 'viewer',
    compareInput,
    qL_norm: gaussian.qL_norm,
    qR_norm: gaussian.qR_norm,
    R4: gaussian.R4,
    dt: gaussian.dt,
    cov_t: gaussian.cov_t,
    marginal_denom: gaussian.marginal_denom,
    marginal_t: gaussian.marginal_t,
    cov12: gaussian.cov12,
    mean_offset: gaussian.mean_offset,
    conditional_cov3x3: gaussian.conditional_cov3x3,
    pos_conditional: gaussian.pos_conditional,
    projected_cov2x2: screen.projected_cov2x2,
    conic: screen.conic,
    det: screen.det,
    mid: screen.mid,
    lambda1: screen.lambda1,
    lambda2: screen.lambda2,
    radius: screen.radius,
    culled: !!(gaussian.culled || screen.culled),
    cullReason: gaussian.culled ? gaussian.cullReason : screen.cullReason,
    opacity_base: gaussian.opacity_base,
    opacity_after_marginal: gaussian.opacity_after_marginal,
    view_space_pos: screen.view_space_pos,
    notes: {
      rotationMode: compareInput.nativeRot4d ? 'viewer-native-rot4d' : 'viewer-buildRotation4DOld',
      marginalMode: compareInput.nativeMarginal
        ? 'viewer-prefilter-var-enabled-when-positive'
        : 'viewer-scale-t-only-marginal',
      nativeRot4dApplied: !!compareInput.nativeRot4d,
      nativeMarginalApplied: !!compareInput.nativeMarginal
    }
  };
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
