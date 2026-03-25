function parseArray(text, expectedLength = null) {
  const value = JSON.parse(text);
  if (!Array.isArray(value)) throw new Error('配列ではありません: ' + text);
  if (expectedLength !== null && value.length !== expectedLength) {
    throw new Error(`要素数が ${expectedLength} ではありません: ${text}`);
  }
  return value.map(Number);
}

function parseNumber(text) {
  const v = Number(text);
  if (!Number.isFinite(v)) throw new Error('数値ではありません: ' + text);
  return v;
}

function normalizeQuat4(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

function mat4Mul(A, B) {
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

function mat4Transpose(A) {
  const R = Array.from({ length: 4 }, () => Array(4).fill(0));
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      R[i][j] = A[j][i];
    }
  }
  return R;
}

function outer3(a, b) {
  return [
    [a[0] * b[0], a[0] * b[1], a[0] * b[2]],
    [a[1] * b[0], a[1] * b[1], a[1] * b[2]],
    [a[2] * b[0], a[2] * b[1], a[2] * b[2]]
  ];
}

function sub3(A, B) {
  return [
    [A[0][0] - B[0][0], A[0][1] - B[0][1], A[0][2] - B[0][2]],
    [A[1][0] - B[1][0], A[1][1] - B[1][1], A[1][2] - B[1][2]],
    [A[2][0] - B[2][0], A[2][1] - B[2][1], A[2][2] - B[2][2]]
  ];
}

function mulScalar3(A, s) {
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

function makeScale4(scaleXYZ, scaleT) {
  return [
    [scaleXYZ[0], 0, 0, 0],
    [0, scaleXYZ[1], 0, 0],
    [0, 0, scaleXYZ[2], 0],
    [0, 0, 0, scaleT]
  ];
}

function sigmaFromM(M, useMTM) {
  return useMTM ? mat4Mul(mat4Transpose(M), M) : mat4Mul(M, mat4Transpose(M));
}

function computeDerived(Sigma, dt, prefilterVar) {
  const cov_t = Sigma[3][3];
  const cov12 = [Sigma[0][3], Sigma[1][3], Sigma[2][3]];
  const cov11 = [
    [Sigma[0][0], Sigma[0][1], Sigma[0][2]],
    [Sigma[1][0], Sigma[1][1], Sigma[1][2]],
    [Sigma[2][0], Sigma[2][1], Sigma[2][2]]
  ];

  const cond = sub3(
    cov11,
    mulScalar3(outer3(cov12, cov12), 1 / Math.max(1e-8, cov_t))
  );

  const delta = [
    cov12[0] / cov_t * dt,
    cov12[1] / cov_t * dt,
    cov12[2] / cov_t * dt
  ];

  const marginalOld = Math.exp(-0.5 * dt * dt / Math.max(1e-8, cov_t));
  const marginalNative = Math.exp(
    -0.5 * dt * dt / Math.max(1e-8, prefilterVar > 0 ? (prefilterVar + cov_t) : cov_t)
  );

  return { cov_t, cov12, cond, delta, marginalOld, marginalNative };
}

function fmtNumber(v) {
  return (Math.abs(v) < 1e-12 ? 0 : v).toFixed(6);
}

function fmtVec(v) {
  return '[' + v.map(fmtNumber).join(', ') + ']';
}

function fmtMat(M) {
  return M.map(row => '[' + row.map(fmtNumber).join(', ') + ']').join('\n');
}

function block(title, value) {
  return `${title}\n${value}\n`;
}

function computeAndRender() {
  const qL = parseArray(document.getElementById('qL').value, 4);
  const qR = parseArray(document.getElementById('qR').value, 4);
  const scaleXYZ = parseArray(document.getElementById('scale').value, 3);
  const scaleT = parseNumber(document.getElementById('scaleT').value);
  const dt = parseNumber(document.getElementById('dt').value);
  const prefilterVar = parseNumber(document.getElementById('prefilterVar').value);

  const S = makeScale4(scaleXYZ, scaleT);
  const Rold = buildRotation4DOld(qL, qR);
  const Rnative = buildRotation4DNative(qL, qR);

  const cases = [
    { name: 'OLD / M = R*S / Sigma = M*M^T', R: Rold, useOrderSR: false, useSigmaMTM: false },
    { name: 'OLD / M = S*R / Sigma = M*M^T', R: Rold, useOrderSR: true, useSigmaMTM: false },
    { name: 'OLD / M = R*S / Sigma = M^T*M', R: Rold, useOrderSR: false, useSigmaMTM: true },
    { name: 'OLD / M = S*R / Sigma = M^T*M', R: Rold, useOrderSR: true, useSigmaMTM: true },
    { name: 'NATIVE / M = R*S / Sigma = M*M^T', R: Rnative, useOrderSR: false, useSigmaMTM: false },
    { name: 'NATIVE / M = S*R / Sigma = M*M^T', R: Rnative, useOrderSR: true, useSigmaMTM: false },
    { name: 'NATIVE / M = R*S / Sigma = M^T*M', R: Rnative, useOrderSR: false, useSigmaMTM: true },
    { name: 'NATIVE / M = S*R / Sigma = M^T*M', R: Rnative, useOrderSR: true, useSigmaMTM: true }
  ];

  let out = '';
  out += block('qL', fmtVec(normalizeQuat4(qL)));
  out += block('qR', fmtVec(normalizeQuat4(qR)));
  out += block('S', fmtMat(S));
  out += block('R_old', fmtMat(Rold));
  out += block('R_native', fmtMat(Rnative));

  for (const c of cases) {
    const M = c.useOrderSR ? mat4Mul(S, c.R) : mat4Mul(c.R, S);
    const Sigma = sigmaFromM(M, c.useSigmaMTM);
    const d = computeDerived(Sigma, dt, prefilterVar);

    out += '============================================================\n';
    out += block(c.name, '');
    out += block('M', fmtMat(M));
    out += block('Sigma', fmtMat(Sigma));
    out += block('cov_t', fmtNumber(d.cov_t));
    out += block('cov12', fmtVec(d.cov12));
    out += block('delta', fmtVec(d.delta));
    out += block('cond', fmtMat(d.cond));
    out += block('marginal(old)', fmtNumber(d.marginalOld));
    out += block('marginal(native)', fmtNumber(d.marginalNative));
  }

  document.getElementById('out').textContent = out;
}

document.getElementById('runBtn').addEventListener('click', () => {
  try {
    computeAndRender();
  } catch (e) {
    document.getElementById('out').textContent = 'エラー: ' + e.message;
  }
});

document.getElementById('presetBtn').addEventListener('click', () => {
  document.getElementById('qL').value = "[0.8660254, 0.3535534, 0.3535534, 0.0]";
  document.getElementById('qR').value = "[0.9238795, 0.0, 0.2705981, 0.2705981]";
  document.getElementById('scale').value = "[1.4, 0.6, 0.3]";
  document.getElementById('scaleT').value = "0.45";
  document.getElementById('dt').value = "0.15";
  document.getElementById('prefilterVar').value = "0.10";
  try {
    computeAndRender();
  } catch (e) {
    document.getElementById('out').textContent = 'エラー: ' + e.message;
  }
});

computeAndRender();
