function parseArray(text) {
  return JSON.parse(text);
}

function parseNumber(text) {
  const v = Number(text);
  if (!Number.isFinite(v)) throw new Error("数値ではありません: " + text);
  return v;
}

function clamp(x, a, b) {
  return Math.min(b, Math.max(a, x));
}

function mat3Mul(A, B) {
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

function mat3Transpose(A) {
  return [
    [A[0][0], A[1][0], A[2][0]],
    [A[0][1], A[1][1], A[2][1]],
    [A[0][2], A[1][2], A[2][2]]
  ];
}

function mat4VecMul(A, v) {
  return [
    A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2] + A[0][3] * v[3],
    A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2] + A[1][3] * v[3],
    A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2] + A[2][3] * v[3],
    A[3][0] * v[0] + A[3][1] * v[1] + A[3][2] * v[2] + A[3][3] * v[3]
  ];
}

function addDiag2(cov, blurPad) {
  return [
    [cov[0][0] + blurPad, cov[0][1]],
    [cov[1][0], cov[1][1] + blurPad]
  ];
}

function fmtNumber(v) {
  return (Math.abs(v) < 1e-12 ? 0 : v).toFixed(6);
}

function fmtVec(v) {
  return "[" + v.map(fmtNumber).join(", ") + "]";
}

function fmtMat(M) {
  return M.map(row => "[" + row.map(fmtNumber).join(", ") + "]").join("\n");
}

function block(title, value) {
  return `${title}\n${value}\n`;
}

function extractWFromView(view4) {
  return [
    [view4[0][0], view4[0][1], view4[0][2]],
    [view4[1][0], view4[1][1], view4[1][2]],
    [view4[2][0], view4[2][1], view4[2][2]]
  ];
}

function perspectiveClip(posCam, fovYDeg, aspect) {
  const fovY = fovYDeg * Math.PI / 180.0;
  const tanFovY = Math.tan(fovY * 0.5);
  const tanFovX = tanFovY * aspect;
  const x = posCam[0], y = posCam[1], z = posCam[2];
  const zc = -z;
  const ndcX = (x / zc) / tanFovX;
  const ndcY = (y / zc) / tanFovY;
  return { ndcX, ndcY, zc, tanFovX, tanFovY };
}

function toScreen(ndcX, ndcY, renderW, renderH) {
  return [
    ((ndcX + 1.0) * renderW - 1.0) * 0.5,
    ((ndcY + 1.0) * renderH - 1.0) * 0.5
  ];
}

function conicRadiusFromCov2(cov2, blurPad) {
  const cov = addDiag2(cov2, blurPad);
  const a = cov[0][0];
  const b = cov[0][1];
  const c = cov[1][1];
  const det = a * c - b * b;
  if (!Number.isFinite(det) || det <= 0) {
    return { cov2Pad: cov, conic: null, radius: null, det };
  }
  const inv = 1.0 / det;
  const conic = [c * inv, -b * inv, a * inv];
  const mid = 0.5 * (a + c);
  const delta = Math.max(0.1, mid * mid - det);
  const lambda1 = mid + Math.sqrt(delta);
  const lambda2 = mid - Math.sqrt(delta);
  const radius = Math.ceil(3.0 * Math.sqrt(Math.max(lambda1, lambda2)));
  return { cov2Pad: cov, conic, radius, det };
}

function computeCPUCurrent(pos, cov3, view4, fovYDeg, aspect, renderW, renderH, blurPad) {
  const posCam4 = mat4VecMul(view4, [pos[0], pos[1], pos[2], 1.0]);
  const posCam = [posCam4[0], posCam4[1], posCam4[2]];
  const { zc, tanFovX, tanFovY, ndcX, ndcY } = perspectiveClip(posCam, fovYDeg, aspect);

  const txtz = clamp(posCam[0] / zc, -1.3 * tanFovX, 1.3 * tanFovX);
  const tytz = clamp(posCam[1] / zc, -1.3 * tanFovY, 1.3 * tanFovY);
  const tx = txtz * zc;
  const ty = tytz * zc;
  const fx = renderW / (2.0 * tanFovX);
  const fy = renderH / (2.0 * tanFovY);

  const J = [
    [fx / zc, 0.0, -(fx * tx) / (zc * zc)],
    [0.0, fy / zc, -(fy * ty) / (zc * zc)],
    [0.0, 0.0, 0.0]
  ];

  const W = extractWFromView(view4);
  const T = mat3Mul(W, J);
  const cov2full = mat3Mul(mat3Transpose(T), mat3Mul(cov3, T));
  const cov2 = [
    [cov2full[0][0], cov2full[0][1]],
    [cov2full[1][0], cov2full[1][1]]
  ];

  return {
    posCam, zc, J, W, T, cov2,
    center: toScreen(ndcX, ndcY, renderW, renderH),
    ...conicRadiusFromCov2(cov2, blurPad)
  };
}

function computeNativeCandidate(pos, cov3, view4, fovYDeg, aspect, renderW, renderH, blurPad) {
  const posCam4 = mat4VecMul(view4, [pos[0], pos[1], pos[2], 1.0]);
  const posCam = [posCam4[0], posCam4[1], posCam4[2]];
  const { zc, tanFovX, tanFovY, ndcX, ndcY } = perspectiveClip(posCam, fovYDeg, aspect);

  const txtz = clamp(posCam[0] / zc, -1.3 * tanFovX, 1.3 * tanFovX);
  const tytz = clamp(posCam[1] / zc, -1.3 * tanFovY, 1.3 * tanFovY);
  const tx = txtz * zc;
  const ty = tytz * zc;
  const fx = renderW / (2.0 * tanFovX);
  const fy = renderH / (2.0 * tanFovY);

  const J = [
    [fx / zc, 0.0, -(fx * tx) / (zc * zc)],
    [0.0, fy / zc, -(fy * ty) / (zc * zc)],
    [0.0, 0.0, 0.0]
  ];

  const W = extractWFromView(view4);
  const T = mat3Mul(W, J);
  const cov2full = mat3Mul(T, mat3Mul(cov3, mat3Transpose(T)));
  const cov2 = [
    [cov2full[0][0], cov2full[0][1]],
    [cov2full[1][0], cov2full[1][1]]
  ];

  return {
    posCam, zc, J, W, T, cov2,
    center: toScreen(ndcX, ndcY, renderW, renderH),
    ...conicRadiusFromCov2(cov2, blurPad)
  };
}

function diffMat2(A, B) {
  return [
    [A[0][0] - B[0][0], A[0][1] - B[0][1]],
    [A[1][0] - B[1][0], A[1][1] - B[1][1]]
  ];
}

function diffVec(a, b) {
  return a.map((v, i) => v - b[i]);
}

function run() {
  const pos = parseArray(document.getElementById("pos").value);
  const cov3 = parseArray(document.getElementById("cov3").value);
  const view4 = parseArray(document.getElementById("view").value);
  const fovYDeg = parseNumber(document.getElementById("fovYDeg").value);
  const aspect = parseNumber(document.getElementById("aspect").value);
  const renderW = parseNumber(document.getElementById("renderW").value);
  const renderH = parseNumber(document.getElementById("renderH").value);
  const blurPad = parseNumber(document.getElementById("blurPad").value);

  const cpu = computeCPUCurrent(pos, cov3, view4, fovYDeg, aspect, renderW, renderH, blurPad);
  const native = computeNativeCandidate(pos, cov3, view4, fovYDeg, aspect, renderW, renderH, blurPad);

  let out = "";
  out += block("pos", fmtVec(pos));
  out += block("cov3", fmtMat(cov3));
  out += block("view4", fmtMat(view4));

  out += "================ CPU current =================\n";
  out += block("posCam", fmtVec(cpu.posCam));
  out += block("J", fmtMat(cpu.J));
  out += block("W", fmtMat(cpu.W));
  out += block("T", fmtMat(cpu.T));
  out += block("cov2", fmtMat(cpu.cov2));
  out += block("cov2 (+ blurPad)", fmtMat(cpu.cov2Pad));
  out += block("conic", cpu.conic ? fmtVec(cpu.conic) : "null");
  out += block("radius", cpu.radius === null ? "null" : String(cpu.radius));
  out += block("screen center", fmtVec(cpu.center));

  out += "================ native candidate =================\n";
  out += block("posCam", fmtVec(native.posCam));
  out += block("J", fmtMat(native.J));
  out += block("W", fmtMat(native.W));
  out += block("T", fmtMat(native.T));
  out += block("cov2", fmtMat(native.cov2));
  out += block("cov2 (+ blurPad)", fmtMat(native.cov2Pad));
  out += block("conic", native.conic ? fmtVec(native.conic) : "null");
  out += block("radius", native.radius === null ? "null" : String(native.radius));
  out += block("screen center", fmtVec(native.center));

  out += "================ diff (CPU - native) =================\n";
  out += block("cov2 diff", fmtMat(diffMat2(cpu.cov2, native.cov2)));
  out += block("center diff", fmtVec(diffVec(cpu.center, native.center)));
  if (cpu.conic && native.conic) {
    out += block("conic diff", fmtVec(diffVec(cpu.conic, native.conic)));
  }
  out += block("radius diff", (cpu.radius === null || native.radius === null) ? "null" : String(cpu.radius - native.radius));

  document.getElementById("out").textContent = out;
}

document.getElementById("runBtn").addEventListener("click", () => {
  try {
    run();
  } catch (e) {
    document.getElementById("out").textContent = "エラー: " + e.message;
  }
});

document.getElementById("presetBtn").addEventListener("click", () => {
  document.getElementById("pos").value = "[0.18, -0.09, 3.2]";
  document.getElementById("cov3").value = "[[0.64,0.11,-0.05],[0.11,0.29,0.04],[-0.05,0.04,0.18]]";
  document.getElementById("view").value = "[[0.9659258,0,0.2588190,0],[0,1,0,0],[-0.2588190,0,0.9659258,0],[0,0,0,1]]";
  document.getElementById("fovYDeg").value = "60";
  document.getElementById("aspect").value = "1.7777778";
  document.getElementById("renderW").value = "1280";
  document.getElementById("renderH").value = "720";
  document.getElementById("blurPad").value = "0.3";
  try {
    run();
  } catch (e) {
    document.getElementById("out").textContent = "エラー: " + e.message;
  }
});

run();
