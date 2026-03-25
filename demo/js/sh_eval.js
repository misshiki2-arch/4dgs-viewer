import * as THREE from 'three';
import { clamp } from './rot4d_math.js';

const SH_C0 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = [
  1.0925484305920792,
  1.0925484305920792,
  0.31539156525252005,
  1.0925484305920792,
  0.5462742152960396
];
const SH_C3 = [
  0.5900435899266435,
  2.890611442640554,
  0.4570457994644658,
  0.3731763325901154,
  0.4570457994644658,
  1.445305721320277,
  0.5900435899266435
];

function coeffVec(raw, i, k) {
  const base0 = i * raw.fdcDim;
  const restBase = i * raw.frestDim;

  if (k === 0) {
    return [
      raw.f_dc[base0 + 0] || 0,
      raw.f_dc[base0 + 1] || 0,
      raw.f_dc[base0 + 2] || 0
    ];
  }

  const j = (k - 1) * 3;
  return [
    raw.f_rest[restBase + j + 0] || 0,
    raw.f_rest[restBase + j + 1] || 0,
    raw.f_rest[restBase + j + 2] || 0
  ];
}

function addScaled(out, coeff, s) {
  out[0] += coeff[0] * s;
  out[1] += coeff[1] * s;
  out[2] += coeff[2] * s;
}

export function evalSHColor(raw, i, camPos, pos, timestamp, timeDuration, useSH, forceSh3d) {
  if (!useSH) {
    return [
      clamp((raw.f_dc[i * raw.fdcDim + 0] || 0) + 0.5, 0, 1),
      clamp((raw.f_dc[i * raw.fdcDim + 1] || 0) + 0.5, 0, 1),
      clamp((raw.f_dc[i * raw.fdcDim + 2] || 0) + 0.5, 0, 1)
    ];
  }

  const deg = raw.activeShDegree;
  const degT = raw.activeShDegreeT;
  const gaussianDim4 = raw.tDim > 0 && raw.scaleTDim > 0;

  const dir = new THREE.Vector3(
    pos[0] - camPos.x,
    pos[1] - camPos.y,
    pos[2] - camPos.z
  );
  if (dir.lengthSq() < 1e-12) dir.set(0, 0, 1);
  dir.normalize();

  const x = dir.x;
  const y = dir.y;
  const z = dir.z;

  let result = [0, 0, 0];

  const l0m0 = SH_C0;
  addScaled(result, coeffVec(raw, i, 0), l0m0);

  let l1m1 = 0, l1m0 = 0, l1p1 = 0;
  let l2m2 = 0, l2m1 = 0, l2m0 = 0, l2p1 = 0, l2p2 = 0;
  let l3m3 = 0, l3m2 = 0, l3m1 = 0, l3m0 = 0, l3p1 = 0, l3p2 = 0, l3p3 = 0;

  if (deg > 0) {
    l1m1 = -SH_C1 * y;
    l1m0 =  SH_C1 * z;
    l1p1 = -SH_C1 * x;

    addScaled(result, coeffVec(raw, i, 1), l1m1);
    addScaled(result, coeffVec(raw, i, 2), l1m0);
    addScaled(result, coeffVec(raw, i, 3), l1p1);

    if (deg > 1) {
      const xx = x * x;
      const yy = y * y;
      const zz = z * z;
      const xy = x * y;
      const yz = y * z;
      const xz = x * z;

      l2m2 = SH_C2[0] * xy;
      l2m1 = SH_C2[1] * yz;
      l2m0 = SH_C2[2] * (2 * zz - xx - yy);
      l2p1 = SH_C2[3] * xz;
      l2p2 = SH_C2[4] * (xx - yy);

      addScaled(result, coeffVec(raw, i, 4), l2m2);
      addScaled(result, coeffVec(raw, i, 5), l2m1);
      addScaled(result, coeffVec(raw, i, 6), l2m0);
      addScaled(result, coeffVec(raw, i, 7), l2p1);
      addScaled(result, coeffVec(raw, i, 8), l2p2);

      if (deg > 2) {
        l3m3 = SH_C3[0] * y * (3 * xx - yy);
        l3m2 = SH_C3[1] * xy * z;
        l3m1 = SH_C3[2] * y * (4 * zz - xx - yy);
        l3m0 = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
        l3p1 = SH_C3[4] * x * (4 * zz - xx - yy);
        l3p2 = SH_C3[5] * z * (xx - yy);
        l3p3 = SH_C3[6] * x * (xx - 3 * yy);

        addScaled(result, coeffVec(raw, i, 9),  l3m3);
        addScaled(result, coeffVec(raw, i, 10), l3m2);
        addScaled(result, coeffVec(raw, i, 11), l3m1);
        addScaled(result, coeffVec(raw, i, 12), l3m0);
        addScaled(result, coeffVec(raw, i, 13), l3p1);
        addScaled(result, coeffVec(raw, i, 14), l3p2);
        addScaled(result, coeffVec(raw, i, 15), l3p3);

        if (gaussianDim4 && !forceSh3d && degT > 0) {
          const dir_t = (raw.t[i * raw.tDim] || 0) - timestamp;
          const terms = [
            l0m0, l1m1, l1m0, l1p1,
            l2m2, l2m1, l2m0, l2p1, l2p2,
            l3m3, l3m2, l3m1, l3m0, l3p1, l3p2, l3p3
          ];

          const t1 = Math.cos(2 * Math.PI * dir_t / Math.max(1e-6, timeDuration));
          for (let k = 0; k < terms.length; k++) {
            addScaled(result, coeffVec(raw, i, 16 + k), t1 * terms[k]);
          }

          if (degT > 1) {
            const t2 = Math.cos(4 * Math.PI * dir_t / Math.max(1e-6, timeDuration));
            for (let k = 0; k < terms.length; k++) {
              addScaled(result, coeffVec(raw, i, 32 + k), t2 * terms[k]);
            }
          }
        }
      }
    }
  }

  result[0] = Math.max(result[0] + 0.5, 0);
  result[1] = Math.max(result[1] + 0.5, 0);
  result[2] = Math.max(result[2] + 0.5, 0);

  return [
    clamp(result[0], 0, 1),
    clamp(result[1], 0, 1),
    clamp(result[2], 0, 1)
  ];
}
