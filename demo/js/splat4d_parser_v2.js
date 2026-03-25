export function parseSplat4DV2(arrayBuffer) {
  const dv = new DataView(arrayBuffer);

  const magic = String.fromCharCode(
    dv.getUint8(0),
    dv.getUint8(1),
    dv.getUint8(2),
    dv.getUint8(3)
  );
  if (magic !== 'SPL4') {
    throw new Error('Not SPL4');
  }

  const version = dv.getUint32(4, true);
  if (version !== 2) {
    throw new Error('Need SPL4 v2');
  }

  const N = dv.getUint32(8, true);
  const activeShDegree = dv.getInt32(12, true);
  const activeShDegreeT = dv.getInt32(16, true);
  const rot4d = !!dv.getUint8(20);
  const storeScaleLog = !!dv.getUint8(21);

  const xyzDim = dv.getUint32(24, true);
  const rotationDim = dv.getUint32(28, true);
  const rotationRDim = dv.getUint32(32, true);
  const scaleXYZDim = dv.getUint32(36, true);
  const fdcDim = dv.getUint32(40, true);
  const frestDim = dv.getUint32(44, true);
  const opacityDim = dv.getUint32(48, true);
  const tDim = dv.getUint32(52, true);
  const scaleTDim = dv.getUint32(56, true);

  const data = new Float32Array(arrayBuffer, 128);

  const out = {
    version,
    N,
    activeShDegree,
    activeShDegreeT,
    rot4d,
    storeScaleLog,
    xyzDim,
    rotationDim,
    rotationRDim,
    scaleXYZDim,
    fdcDim,
    frestDim,
    opacityDim,
    tDim,
    scaleTDim,
    xyz: new Float32Array(N * xyzDim),
    rotation: new Float32Array(N * rotationDim),
    rotation_r: new Float32Array(N * rotationRDim),
    scale_xyz: new Float32Array(N * scaleXYZDim),
    f_dc: new Float32Array(N * fdcDim),
    f_rest: new Float32Array(N * frestDim),
    opacity: new Float32Array(N * opacityDim),
    t: new Float32Array(N * tDim),
    scale_t: new Float32Array(N * scaleTDim)
  };

  let p = 0;
  for (let i = 0; i < N; i++) {
    out.xyz.set(data.subarray(p, p + xyzDim), i * xyzDim);
    p += xyzDim;

    out.rotation.set(data.subarray(p, p + rotationDim), i * rotationDim);
    p += rotationDim;

    out.rotation_r.set(data.subarray(p, p + rotationRDim), i * rotationRDim);
    p += rotationRDim;

    out.scale_xyz.set(data.subarray(p, p + scaleXYZDim), i * scaleXYZDim);
    p += scaleXYZDim;

    out.f_dc.set(data.subarray(p, p + fdcDim), i * fdcDim);
    p += fdcDim;

    out.f_rest.set(data.subarray(p, p + frestDim), i * frestDim);
    p += frestDim;

    out.opacity.set(data.subarray(p, p + opacityDim), i * opacityDim);
    p += opacityDim;

    out.t.set(data.subarray(p, p + tDim), i * tDim);
    p += tDim;

    out.scale_t.set(data.subarray(p, p + scaleTDim), i * scaleTDim);
    p += scaleTDim;
  }

  if (storeScaleLog) {
    for (let i = 0; i < out.scale_xyz.length; i++) {
      out.scale_xyz[i] = Math.exp(out.scale_xyz[i]);
    }
    for (let i = 0; i < out.scale_t.length; i++) {
      out.scale_t[i] = Math.exp(out.scale_t[i]);
    }
  }

  return out;
}
