const EPS = 1e-8;
export const CUDA_4D_STATE_HELPER_VERSION = 'step91-cuda-glm-v1';

function normalizeQuat4(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

function mat4FromGlmConstructor(args) {
  return [
    [args[0], args[4], args[8], args[12]],
    [args[1], args[5], args[9], args[13]],
    [args[2], args[6], args[10], args[14]],
    [args[3], args[7], args[11], args[15]]
  ];
}

function mat4Mul(A, B) {
  const C = Array.from({ length: 4 }, () => Array(4).fill(0));
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      for (let k = 0; k < 4; k++) C[i][j] += A[i][k] * B[k][j];
    }
  }
  return C;
}

function mat4Transpose(A) {
  const T = Array.from({ length: 4 }, () => Array(4).fill(0));
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) T[i][j] = A[j][i];
  }
  return T;
}

function buildScalingMatrix4(scaleXYZ, scaleT) {
  return [
    [scaleXYZ[0], 0, 0, 0],
    [0, scaleXYZ[1], 0, 0],
    [0, 0, scaleXYZ[2], 0],
    [0, 0, 0, scaleT]
  ];
}

export function buildRotation4DCudaGlm(qLIn, qRIn) {
  const [a, b, c, d] = normalizeQuat4(qLIn);
  const [p, q, r, s] = normalizeQuat4(qRIn);

  const Ml = mat4FromGlmConstructor([
     a,  b, -c,  d,
    -b,  a,  d,  c,
     c, -d,  a,  b,
    -d, -c, -b,  a
  ]);
  const Mr = mat4FromGlmConstructor([
     p,  q, -r, -s,
    -q,  p,  s, -r,
     r, -s,  p, -q,
     s,  r,  q,  p
  ]);

  return mat4Mul(Mr, Ml);
}

export function computeCudaConditionalGaussianState4D({
  position,
  opacity,
  scaleXYZ,
  scaleT,
  rotation,
  rotationR,
  timestamp,
  tCenter,
  prefilterVar = -1
}) {
  const R4 = buildRotation4DCudaGlm(rotation, rotationR);
  const S = buildScalingMatrix4(scaleXYZ, scaleT);
  const M = mat4Mul(S, R4);
  const Sigma = mat4Mul(mat4Transpose(M), M);
  const covT = Sigma[3][3];
  const dt = timestamp - tCenter;
  const marginalDenom = prefilterVar > 0 ? prefilterVar + covT : covT;
  const marginalT = Math.exp(-0.5 * dt * dt / Math.max(EPS, marginalDenom));
  const cov12 = [Sigma[0][3], Sigma[1][3], Sigma[2][3]];
  const invCovT = 1 / Math.max(EPS, covT);
  const meanOffset = [
    cov12[0] * invCovT * dt,
    cov12[1] * invCovT * dt,
    cov12[2] * invCovT * dt
  ];
  const cov11 = [
    [Sigma[0][0], Sigma[0][1], Sigma[0][2]],
    [Sigma[1][0], Sigma[1][1], Sigma[1][2]],
    [Sigma[2][0], Sigma[2][1], Sigma[2][2]]
  ];
  const cond = [
    [
      cov11[0][0] - cov12[0] * cov12[0] * invCovT,
      cov11[0][1] - cov12[0] * cov12[1] * invCovT,
      cov11[0][2] - cov12[0] * cov12[2] * invCovT
    ],
    [
      cov11[1][0] - cov12[1] * cov12[0] * invCovT,
      cov11[1][1] - cov12[1] * cov12[1] * invCovT,
      cov11[1][2] - cov12[1] * cov12[2] * invCovT
    ],
    [
      cov11[2][0] - cov12[2] * cov12[0] * invCovT,
      cov11[2][1] - cov12[2] * cov12[1] * invCovT,
      cov11[2][2] - cov12[2] * cov12[2] * invCovT
    ]
  ];

  return {
    pos: [
      position[0] + meanOffset[0],
      position[1] + meanOffset[1],
      position[2] + meanOffset[2]
    ],
    cov3: cond,
    opacity: opacity * marginalT,
    debug: {
      helperVersion: CUDA_4D_STATE_HELPER_VERSION,
      stateConvention: 'cuda-glm',
      usedCuda4DStateHelper: true,
      qL_norm: normalizeQuat4(rotation),
      qR_norm: normalizeQuat4(rotationR),
      R4,
      Sigma4: Sigma,
      cov_t: covT,
      marginal_denom: marginalDenom,
      marginal_t: marginalT,
      cov12,
      cov11,
      mean_offset: meanOffset,
      dt,
      culled: marginalT <= 0.05,
      cullReason: marginalT <= 0.05 ? 'temporal-marginal-below-threshold' : 'none'
    }
  };
}
