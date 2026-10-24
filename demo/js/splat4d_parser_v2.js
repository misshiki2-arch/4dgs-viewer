export function parseSplat4DV2(arrayBuffer) {
  if (!(arrayBuffer instanceof ArrayBuffer) || arrayBuffer.byteLength < 128) {
    throw new Error('SPL4 header missing');
  }
  const dv = new DataView(arrayBuffer);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== 'SPL4') throw new Error('Not SPL4');
  const version = dv.getUint32(4, true);
  if (version !== 2) throw new Error('Need SPL4 v2');

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

  const flags = [20, 21, 22, 23].map((offset) => dv.getUint8(offset));
  if (flags.some((flag) => flag !== 0 && flag !== 1) || flags[2] !== 1 || flags[3] !== 1) {
    throw new Error('SPL4 unsupported raw flags');
  }
  for (let i = 60; i < 128; i++) {
    if (dv.getUint8(i) !== 0) throw new Error('SPL4 reserved bytes nonzero');
  }
  const dimensions = [xyzDim, rotationDim, rotationRDim, scaleXYZDim, fdcDim,
    frestDim, opacityDim, tDim, scaleTDim];
  if (!N || xyzDim !== 3 || rotationDim !== 4 || scaleXYZDim !== 3 || fdcDim !== 3 ||
      opacityDim !== 1 || ![0, 4].includes(rotationRDim) ||
      ![0, 1].includes(tDim) || scaleTDim !== tDim ||
      (rot4d && (rotationRDim !== 4 || tDim !== 1)) ||
      ![0, 9, 24, 45, 93, 141].includes(frestDim) ||
      activeShDegree < 0 || activeShDegree > 3 || activeShDegreeT < 0 || activeShDegreeT > 2 ||
      frestDim < ((activeShDegree + 1) ** 2 - 1) * 3) {
    throw new Error('SPL4 unsupported dimensions/degrees');
  }
  const stride = dimensions.reduce((a, b) => a + b, 0);
  const expectedBytes = 128 + N * stride * 4;
  if (!Number.isSafeInteger(expectedBytes) || arrayBuffer.byteLength !== expectedBytes) {
    throw new Error('SPL4 payload length mismatch');
  }

  const data = new Float32Array(arrayBuffer, 128);
  // Validate before SoA allocation/publication, without changing raw SH/opacity/quaternions.
  for (let i = 0; i < data.length; i++) {
    if (!Number.isFinite(data[i])) throw new Error('SPL4 nonfinite payload');
  }
  for (let row = 0; row < N; row++) {
    const start = row * stride;
    for (const offset of (rotationRDim ? [3, 7] : [3])) {
      if (!(Math.hypot(...data.subarray(start + offset, start + offset + 4)) > 0)) {
        throw new Error('SPL4 zero quaternion');
      }
    }
    const scaleStart = start + xyzDim + rotationDim + rotationRDim;
    const scaleOffsets = [scaleStart, scaleStart + 1, scaleStart + 2];
    if (scaleTDim) scaleOffsets.push(start + stride - 1);
    for (const offset of scaleOffsets) {
      const linear = storeScaleLog ? Math.fround(Math.exp(data[offset])) : data[offset];
      if (!Number.isFinite(linear) || linear <= 0) throw new Error('SPL4 invalid linear scale');
    }
  }
  const out = {
    version, N, activeShDegree, activeShDegreeT, rot4d, storeScaleLog,
    rawSh: true, rawOpacity: true,
    runtimeScaleRepresentation: 'linear', runtimeOpacityRepresentation: 'logit',
    runtimeQuaternionRepresentation: 'wxyz',
    xyzDim, rotationDim, rotationRDim, scaleXYZDim, fdcDim, frestDim, opacityDim, tDim, scaleTDim,
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
    out.xyz.set(data.subarray(p, p + xyzDim), i * xyzDim); p += xyzDim;
    out.rotation.set(data.subarray(p, p + rotationDim), i * rotationDim); p += rotationDim;
    out.rotation_r.set(data.subarray(p, p + rotationRDim), i * rotationRDim); p += rotationRDim;
    out.scale_xyz.set(data.subarray(p, p + scaleXYZDim), i * scaleXYZDim); p += scaleXYZDim;
    out.f_dc.set(data.subarray(p, p + fdcDim), i * fdcDim); p += fdcDim;
    out.f_rest.set(data.subarray(p, p + frestDim), i * frestDim); p += frestDim;
    out.opacity.set(data.subarray(p, p + opacityDim), i * opacityDim); p += opacityDim;
    out.t.set(data.subarray(p, p + tDim), i * tDim); p += tDim;
    out.scale_t.set(data.subarray(p, p + scaleTDim), i * scaleTDim); p += scaleTDim;
  }

  if (storeScaleLog) {
    for (let i = 0; i < out.scale_xyz.length; i++) out.scale_xyz[i] = Math.exp(out.scale_xyz[i]);
    for (let i = 0; i < out.scale_t.length; i++) out.scale_t[i] = Math.exp(out.scale_t[i]);
  }
  return out;
}
