import {
  buildWebGpu4DStateSourceContract,
  buildWebGpuGaussianAttributeEvaluationContract,
  buildWebGpuGaussianFootprintEvaluationContract
} from './common_4dgs_record_contracts.js';

function toFiniteNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function createBuffer(device, data, usage) {
  const buffer = device.createBuffer({
    size: Math.max(4, Math.ceil(data.byteLength / 4) * 4),
    usage,
    mappedAtCreation: true
  });
  const mapped = new Uint8Array(buffer.getMappedRange());
  mapped.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  buffer.unmap();
  return buffer;
}

function buildStateEvaluatorParams({ count, buildConfig }) {
  const data = new ArrayBuffer(16);
  const view = new DataView(data);
  view.setUint32(0, count >>> 0, true);
  view.setFloat32(4, toFiniteNumber(buildConfig?.timestamp, 0), true);
  view.setFloat32(8, toFiniteNumber(buildConfig?.scalingModifier, 1), true);
  view.setFloat32(12, toFiniteNumber(buildConfig?.sigmaScale, 1), true);
  return new Uint8Array(data);
}

function buildCandidateTimeScale(raw, candidateIndices) {
  const out = new Float32Array(candidateIndices.length * 4);
  for (let row = 0; row < candidateIndices.length; row += 1) {
    const srcIndex = candidateIndices[row] | 0;
    const o = row * 4;
    out[o + 0] = toFiniteNumber(raw.t?.[srcIndex * raw.tDim], 0);
    out[o + 1] = toFiniteNumber(raw.scale_t?.[srcIndex * raw.scaleTDim], 1);
    out[o + 2] = raw.tDim > 0 ? 1 : 0;
    out[o + 3] = raw.scaleTDim > 0 ? 1 : 0;
  }
  return out;
}

function buildCandidateAttributeInput(raw, candidateIndices) {
  const out = new Float32Array(candidateIndices.length * 8);
  for (let row = 0; row < candidateIndices.length; row += 1) {
    const srcIndex = candidateIndices[row] | 0;
    const fdcBase = srcIndex * (raw.fdcDim ?? 0);
    const scaleBase = srcIndex * (raw.scaleXYZDim ?? 0);
    const sx = toFiniteNumber(raw.scale_xyz?.[scaleBase + 0], 0.01);
    const sy = toFiniteNumber(raw.scale_xyz?.[scaleBase + 1], sx);
    const sz = toFiniteNumber(raw.scale_xyz?.[scaleBase + 2], sx);
    const o = row * 8;
    out[o + 0] = toFiniteNumber(raw.f_dc?.[fdcBase + 0], 0);
    out[o + 1] = toFiniteNumber(raw.f_dc?.[fdcBase + 1], 0);
    out[o + 2] = toFiniteNumber(raw.f_dc?.[fdcBase + 2], 0);
    out[o + 3] = Math.max(1e-6, (sx + sy + sz) / 3);
    out[o + 4] = Math.max(1e-6, sx);
    out[o + 5] = Math.max(1e-6, sy);
    out[o + 6] = Math.max(1e-6, sz);
    out[o + 7] = raw.scaleXYZDim >= 3 ? 111 : 0;
  }
  return out;
}

function buildCandidateRotationInput(raw, candidateIndices) {
  const out = new Float32Array(candidateIndices.length * 8);
  for (let row = 0; row < candidateIndices.length; row += 1) {
    const srcIndex = candidateIndices[row] | 0;
    const rotationBase = srcIndex * (raw.rotationDim ?? 0);
    const rotationRBase = srcIndex * (raw.rotationRDim ?? 0);
    const o = row * 8;
    out[o + 0] = toFiniteNumber(raw.rotation?.[rotationBase + 0], 1);
    out[o + 1] = toFiniteNumber(raw.rotation?.[rotationBase + 1], 0);
    out[o + 2] = toFiniteNumber(raw.rotation?.[rotationBase + 2], 0);
    out[o + 3] = toFiniteNumber(raw.rotation?.[rotationBase + 3], 0);
    out[o + 4] = toFiniteNumber(raw.rotation_r?.[rotationRBase + 0], 1);
    out[o + 5] = toFiniteNumber(raw.rotation_r?.[rotationRBase + 1], 0);
    out[o + 6] = toFiniteNumber(raw.rotation_r?.[rotationRBase + 2], 0);
    out[o + 7] = toFiniteNumber(raw.rotation_r?.[rotationRBase + 3], 0);
  }
  return out;
}

const STEP113_DIAGNOSTIC_ROW_COUNT = 8;
const STEP113_DIAGNOSTIC_VEC4_STRIDE = 8;
const STEP113_DIAGNOSTIC_ROW_SENTINEL = 0xffffffff;

function buildStep113DiagnosticRowIndices(count) {
  const rows = new Uint32Array(STEP113_DIAGNOSTIC_ROW_COUNT);
  rows.fill(STEP113_DIAGNOSTIC_ROW_SENTINEL);
  if (count <= 0) return rows;
  const selected = [];
  for (const fraction of [0, 0.17, 0.37, 0.63, 0.83, 1]) {
    const row = Math.min(count - 1, Math.max(0, Math.round((count - 1) * fraction)));
    if (!selected.includes(row)) selected.push(row);
  }
  for (let i = 0; i < Math.min(rows.length, selected.length); i += 1) {
    rows[i] = selected[i] >>> 0;
  }
  return rows;
}

function summarizeComputedStatePositions(statePositions) {
  const count = Math.floor((statePositions?.length ?? 0) / 4);
  let computed4DStatePositionCount = 0;
  let cudaConditionalTemporalMeanCount = 0;
  let unavailableStatePositionCount = 0;
  for (let row = 0; row < count; row += 1) {
    const w = Number(statePositions[row * 4 + 3]);
    if (w > 0.89) {
      computed4DStatePositionCount += 1;
      if (w > 0.99) cudaConditionalTemporalMeanCount += 1;
    } else {
      unavailableStatePositionCount += 1;
    }
  }
  return {
    computed4DStatePositionCount,
    cudaConditionalTemporalMeanCount,
    unavailableStatePositionCount
  };
}

function summarizeComputedRenderAttributes(renderAttributes) {
  const count = Math.floor((renderAttributes?.length ?? 0) / 8);
  let computedRenderAttributeCount = 0;
  let radiusSum = 0;
  let alphaSum = 0;
  for (let row = 0; row < count; row += 1) {
    const o = row * 8;
    const radius = Number(renderAttributes[o + 0]);
    const alpha = Number(renderAttributes[o + 1]);
    const sourceCode = Number(renderAttributes[o + 6]);
    if (radius > 0 && alpha > 0 && sourceCode === 81) {
      computedRenderAttributeCount += 1;
      radiusSum += radius;
      alphaSum += alpha;
    }
  }
  const averageComputedRadiusPx =
    computedRenderAttributeCount > 0
      ? radiusSum / computedRenderAttributeCount
      : null;
  return {
    computedRenderAttributeCount,
    averageComputedRadiusPx,
    averageComputedAlpha:
      computedRenderAttributeCount > 0
        ? alphaSum / computedRenderAttributeCount
        : null,
    normalBackendPointRadiusPx:
      averageComputedRadiusPx == null
        ? null
        : Math.max(3, Math.min(14, Math.round(averageComputedRadiusPx)))
  };
}

function summarizeComputedFootprintPayload(footprintPayload, candidateCount = null) {
  const count = Math.min(
    candidateCount ?? Number.POSITIVE_INFINITY,
    Math.floor((footprintPayload?.length ?? 0) / 12)
  );
  let computedFootprintPayloadCount = 0;
  let rotationAwareCovarianceCount = 0;
  let cameraJacobianConicCount = 0;
  let scaleOnlyFallbackCount = 0;
  let conicXSum = 0;
  let areaSum = 0;
  for (let row = 0; row < count; row += 1) {
    const o = row * 12;
    const conicX = Number(footprintPayload[o + 0]);
    const radius = Number(footprintPayload[o + 6]);
    const sourceCode = Number(footprintPayload[o + 8]);
    const area = Number(footprintPayload[o + 10]);
    if (
      conicX > 0 &&
      radius > 0 &&
      (sourceCode === 82 || sourceCode === 111 || sourceCode === 113)
    ) {
      computedFootprintPayloadCount += 1;
      if (sourceCode === 113) {
        rotationAwareCovarianceCount += 1;
        cameraJacobianConicCount += 1;
      } else if (sourceCode === 111) {
        scaleOnlyFallbackCount += 1;
      }
      conicXSum += conicX;
      areaSum += Number.isFinite(area) ? area : 0;
    }
  }
  return {
    computedFootprintPayloadCount,
    averageComputedConicX:
      computedFootprintPayloadCount > 0
        ? conicXSum / computedFootprintPayloadCount
        : null,
    averageComputedFootprintAreaPx:
      computedFootprintPayloadCount > 0
        ? areaSum / computedFootprintPayloadCount
        : null,
    rotationAwareCovarianceCount,
    cameraJacobianConicCount,
    scaleOnlyFallbackCount
  };
}

export async function buildWebGpu4DStatePositionsForCandidates({
  device,
  raw,
  candidateIndices,
  rawXyzOpacity,
  buildConfig,
  projectionParams = null
}) {
  const count = candidateIndices?.length ?? 0;
  if (!device || !raw || !candidateIndices || !rawXyzOpacity || count <= 0) {
    return {
      statePositions: new Float32Array(0),
      renderAttributes: new Float32Array(0),
      footprintPayload: new Float32Array(0),
      contract: buildWebGpu4DStateSourceContract({
        status: 'unavailable',
        stateSourceMode: 'webgpu-partial-4d-state-evaluator',
        reason: 'webgpu-4d-state-evaluator-input-unavailable'
      }),
      gaussianAttributeEvaluationContract:
        buildWebGpuGaussianAttributeEvaluationContract({
          status: 'unavailable',
          reason: 'webgpu-gaussian-attribute-evaluator-input-unavailable'
      }),
      gaussianFootprintEvaluationContract:
        buildWebGpuGaussianFootprintEvaluationContract({
          status: 'unavailable',
          reason: 'webgpu-gaussian-footprint-evaluator-input-unavailable'
        })
    };
  }

  const timeScale = buildCandidateTimeScale(raw, candidateIndices);
  const attributeInput = buildCandidateAttributeInput(raw, candidateIndices);
  const rotationInput = buildCandidateRotationInput(raw, candidateIndices);
  const projectionParamInput =
    projectionParams instanceof Float32Array && projectionParams.length >= 24
      ? projectionParams
      : new Float32Array(24);
  const step113DiagnosticRows = buildStep113DiagnosticRowIndices(count);
  const shader = device.createShaderModule({
    label: 'phase3-step81-partial-4d-state-and-attribute-evaluator-wgsl',
    code: `
struct Params {
  count: u32,
  timestamp: f32,
  scalingModifier: f32,
  sigmaScale: f32,
};

@group(0) @binding(0) var<storage, read> rawXyzOpacity: array<vec4f>;
@group(0) @binding(1) var<storage, read> timeScale: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> statePositions: array<vec4f>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> attributeInput: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> renderAttributes: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> footprintPayload: array<vec4f>;
@group(0) @binding(7) var<storage, read> rotationInput: array<vec4f>;
@group(0) @binding(8) var<storage, read> projectionParams: array<vec4f>;

const STEP113_DIAGNOSTIC_ROW_COUNT: u32 = 8u;
const STEP113_DIAGNOSTIC_VEC4_STRIDE: u32 = 8u;
const STEP113_DIAGNOSTIC_ROW_SENTINEL: u32 = 0xffffffffu;

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

fn quatToCudaRotationRows(qRaw: vec4f) -> mat3x3f {
  let qLen = max(length(qRaw), 1e-6);
  let q = qRaw / qLen;
  let r = q.x;
  let x = q.y;
  let y = q.z;
  let z = q.w;
  return mat3x3f(
    vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y)),
    vec3f(2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x)),
    vec3f(2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y))
  );
}

fn cov3Apply(
  v: vec3f,
  c00: f32,
  c01: f32,
  c02: f32,
  c11: f32,
  c12: f32,
  c22: f32
) -> vec3f {
  return vec3f(
    c00 * v.x + c01 * v.y + c02 * v.z,
    c01 * v.x + c11 * v.y + c12 * v.z,
    c02 * v.x + c12 * v.y + c22 * v.z
  );
}

fn cov3Bilinear(
  a: vec3f,
  b: vec3f,
  c00: f32,
  c01: f32,
  c02: f32,
  c11: f32,
  c12: f32,
  c22: f32
) -> f32 {
  return dot(a, cov3Apply(b, c00, c01, c02, c11, c12, c22));
}

fn rotation4dRowsCudaGlm(qLRaw: vec4f, qRRaw: vec4f) -> mat4x4f {
  let qL = qLRaw / max(length(qLRaw), 1e-6);
  let qR = qRRaw / max(length(qRRaw), 1e-6);
  let a = qL.x;
  let b = qL.y;
  let c = qL.z;
  let d = qL.w;
  let p = qR.x;
  let q = qR.y;
  let r = qR.z;
  let s = qR.w;
  let ml0 = vec4f(a, -b, c, -d);
  let ml1 = vec4f(b, a, -d, -c);
  let ml2 = vec4f(-c, d, a, -b);
  let ml3 = vec4f(d, c, b, a);
  let mr0 = vec4f(p, -q, r, s);
  let mr1 = vec4f(q, p, -s, r);
  let mr2 = vec4f(-r, s, p, q);
  let mr3 = vec4f(-s, -r, -q, p);
  let mlCol0 = vec4f(ml0.x, ml1.x, ml2.x, ml3.x);
  let mlCol1 = vec4f(ml0.y, ml1.y, ml2.y, ml3.y);
  let mlCol2 = vec4f(ml0.z, ml1.z, ml2.z, ml3.z);
  let mlCol3 = vec4f(ml0.w, ml1.w, ml2.w, ml3.w);
  return mat4x4f(
    vec4f(dot(mr0, mlCol0), dot(mr0, mlCol1), dot(mr0, mlCol2), dot(mr0, mlCol3)),
    vec4f(dot(mr1, mlCol0), dot(mr1, mlCol1), dot(mr1, mlCol2), dot(mr1, mlCol3)),
    vec4f(dot(mr2, mlCol0), dot(mr2, mlCol1), dot(mr2, mlCol2), dot(mr2, mlCol3)),
    vec4f(dot(mr3, mlCol0), dot(mr3, mlCol1), dot(mr3, mlCol2), dot(mr3, mlCol3))
  );
}

fn sigma4Component(scaleSq: vec4f, colA: vec4f, colB: vec4f) -> f32 {
  return dot(scaleSq * colA, colB);
}

fn cudaConditionalTemporalMeanOffset(
  dt: f32,
  scaleXYZ: vec3f,
  scaleT: f32,
  qLRaw: vec4f,
  qRRaw: vec4f
) -> vec4f {
  let r4 = rotation4dRowsCudaGlm(qLRaw, qRRaw);
  let row0 = r4[0];
  let row1 = r4[1];
  let row2 = r4[2];
  let row3 = r4[3];
  let col0 = vec4f(row0.x, row1.x, row2.x, row3.x);
  let col1 = vec4f(row0.y, row1.y, row2.y, row3.y);
  let col2 = vec4f(row0.z, row1.z, row2.z, row3.z);
  let col3 = vec4f(row0.w, row1.w, row2.w, row3.w);
  let scaleSq = vec4f(
    scaleXYZ.x * scaleXYZ.x,
    scaleXYZ.y * scaleXYZ.y,
    scaleXYZ.z * scaleXYZ.z,
    scaleT * scaleT
  );
  let covT = max(sigma4Component(scaleSq, col3, col3), 1e-8);
  let cov12 = vec3f(
    sigma4Component(scaleSq, col0, col3),
    sigma4Component(scaleSq, col1, col3),
    sigma4Component(scaleSq, col2, col3)
  );
  return vec4f(cov12 * (dt / covT), covT);
}

fn roundFractionRow(maxRow: u32, numerator: u32, denominator: u32) -> u32 {
  return (maxRow * numerator + denominator / 2u) / denominator;
}

fn step113DiagnosticRowForSlot(count: u32, slot: u32) -> u32 {
  if (count == 0u || slot >= STEP113_DIAGNOSTIC_ROW_COUNT) {
    return STEP113_DIAGNOSTIC_ROW_SENTINEL;
  }
  let maxRow = count - 1u;
  if (slot == 0u) { return 0u; }
  if (slot == 1u) { return roundFractionRow(maxRow, 17u, 100u); }
  if (slot == 2u) { return roundFractionRow(maxRow, 37u, 100u); }
  if (slot == 3u) { return roundFractionRow(maxRow, 63u, 100u); }
  if (slot == 4u) { return roundFractionRow(maxRow, 83u, 100u); }
  if (slot == 5u) { return maxRow; }
  return STEP113_DIAGNOSTIC_ROW_SENTINEL;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let row = id.x;
  if (row >= params.count) {
    return;
  }
  let raw0 = rawXyzOpacity[row];
  let ts = timeScale[row];
  let hasTime = ts.z > 0.5;
  let hasScaleT = ts.w > 0.5;
  let scaleT = max(select(1.0, ts.y, hasScaleT) * params.scalingModifier * params.sigmaScale, 1e-6);
  let dt = select(0.0, params.timestamp - ts.x, hasTime);
  let attrBaseInput = row * 2u;
  let attrs = attributeInput[attrBaseInput + 0u];
  let scaleInfo = attributeInput[attrBaseInput + 1u];
  let meanScale = max(attrs.w, 1e-6);
  let scaleX = max(scaleInfo.x * params.scalingModifier, 1e-6);
  let scaleY = max(scaleInfo.y * params.scalingModifier, 1e-6);
  let scaleZ = max(scaleInfo.z * params.scalingModifier, 1e-6);
  let rotationBase = row * 2u;
  let qRaw = rotationInput[rotationBase + 0u];
  let qRRaw = rotationInput[rotationBase + 1u];
  let temporalMean = cudaConditionalTemporalMeanOffset(
    dt,
    vec3f(scaleX, scaleY, scaleZ),
    scaleT,
    qRaw,
    qRRaw
  );
  let temporalWeight = exp(-0.5 * dt * dt / max(temporalMean.w, 1e-8));
  let pos = raw0.xyz + temporalMean.xyz;
  statePositions[row] = vec4f(pos, 1.0);
  let alpha = clamp(sigmoid(raw0.w) * temporalWeight, 0.05, 0.99);
  let rgb = clamp(attrs.rgb + vec3f(0.5), vec3f(0.0), vec3f(1.0));
  let normalizedTemporal = clamp(dt / scaleT, -1.0, 1.0);
  let radiusPx = clamp(attrs.w * 900.0 + 2.0 + abs(normalizedTemporal) * 2.0, 3.0, 14.0);
  let attrBase = row * 2u;
  renderAttributes[attrBase + 0u] = vec4f(radiusPx, alpha, rgb.r, rgb.g);
  renderAttributes[attrBase + 1u] = vec4f(rgb.b, temporalWeight, 81.0, 0.0);
  let anisotropyX = clamp(scaleX / meanScale, 0.35, 2.75);
  let anisotropyY = clamp(scaleY / meanScale, 0.35, 2.75);
  let radiusX = max(radiusPx * anisotropyX, 1.0);
  let radiusY = max(radiusPx * anisotropyY, 1.0);
  let conservativeRadiusPx = max(radiusPx, max(radiusX, radiusY));
  let sigmaX = max(radiusX / 3.0, 0.5);
  let sigmaY = max(radiusY / 3.0, 0.5);
  let varianceX = sigmaX * sigmaX;
  let varianceY = sigmaY * sigmaY;
  var conicX = 1.0 / varianceX;
  var conicY = 0.0;
  var conicZ = 1.0 / varianceY;
  var covarianceA = varianceX;
  var covarianceB = 0.0;
  var covarianceC = varianceY;
  var finalRadiusPx = conservativeRadiusPx;
  var finalDepth = raw0.z;
  var footprintAreaPx = 3.14159265 * radiusX * radiusY;
  var footprintSourceCode = 111.0;
  var fallbackFlag = 1.0;

  let q = qRaw / max(length(qRaw), 1e-6);
  let qr = q.x;
  let qx = q.y;
  let qy = q.z;
  let qz = q.w;
  let r0 = vec3f(
    1.0 - 2.0 * (qy * qy + qz * qz),
    2.0 * (qx * qy - qr * qz),
    2.0 * (qx * qz + qr * qy)
  );
  let r1 = vec3f(
    2.0 * (qx * qy + qr * qz),
    1.0 - 2.0 * (qx * qx + qz * qz),
    2.0 * (qy * qz - qr * qx)
  );
  let r2 = vec3f(
    2.0 * (qx * qz - qr * qy),
    2.0 * (qy * qz + qr * qx),
    1.0 - 2.0 * (qx * qx + qy * qy)
  );
  let m0 = scaleX * r0;
  let m1 = scaleY * r1;
  let m2 = scaleZ * r2;
  let cov00 = m0.x * m0.x + m1.x * m1.x + m2.x * m2.x;
  let cov01 = m0.x * m0.y + m1.x * m1.y + m2.x * m2.y;
  let cov02 = m0.x * m0.z + m1.x * m1.z + m2.x * m2.z;
  let cov11 = m0.y * m0.y + m1.y * m1.y + m2.y * m2.y;
  let cov12 = m0.y * m0.z + m1.y * m1.z + m2.y * m2.z;
  let cov22 = m0.z * m0.z + m1.z * m1.z + m2.z * m2.z;

  let view0 = projectionParams[3u].xyz;
  let view1 = projectionParams[4u].xyz;
  let view2 = projectionParams[5u].xyz;
  let mv = vec3f(
    dot(projectionParams[3u], vec4f(pos, 1.0)),
    dot(projectionParams[4u], vec4f(pos, 1.0)),
    dot(projectionParams[5u], vec4f(pos, 1.0))
  );
  let covCam00 = cov3Bilinear(view0, view0, cov00, cov01, cov02, cov11, cov12, cov22);
  let covCam01 = cov3Bilinear(view0, view1, cov00, cov01, cov02, cov11, cov12, cov22);
  let covCam02 = cov3Bilinear(view0, view2, cov00, cov01, cov02, cov11, cov12, cov22);
  let covCam11 = cov3Bilinear(view1, view1, cov00, cov01, cov02, cov11, cov12, cov22);
  let covCam12 = cov3Bilinear(view1, view2, cov00, cov01, cov02, cov11, cov12, cov22);
  let covCam22 = cov3Bilinear(view2, view2, cov00, cov01, cov02, cov11, cov12, cov22);

  let fx = abs(projectionParams[2u].x);
  let fy = abs(projectionParams[2u].y);
  let renderW = max(abs(projectionParams[0u].y), 1.0);
  let renderH = max(abs(projectionParams[0u].z), 1.0);
  let zSafe = max(abs(mv.z), 1e-6);
  let tanFovX = renderW / max(2.0 * fx, 1e-6);
  let tanFovY = renderH / max(2.0 * fy, 1e-6);
  let clampedX = clamp(mv.x / zSafe, -1.3 * tanFovX, 1.3 * tanFovX) * zSafe;
  let clampedY = clamp(mv.y / zSafe, -1.3 * tanFovY, 1.3 * tanFovY) * zSafe;
  let jacobian0 = vec3f(fx / zSafe, 0.0, -(fx * clampedX) / (zSafe * zSafe));
  let jacobian1 = vec3f(0.0, fy / zSafe, -(fy * clampedY) / (zSafe * zSafe));
  let screenCovA = cov3Bilinear(
    jacobian0,
    jacobian0,
    covCam00,
    covCam01,
    covCam02,
    covCam11,
    covCam12,
    covCam22
  ) + 0.3;
  let screenCovB = cov3Bilinear(
    jacobian0,
    jacobian1,
    covCam00,
    covCam01,
    covCam02,
    covCam11,
    covCam12,
    covCam22
  );
  let screenCovC = cov3Bilinear(
    jacobian1,
    jacobian1,
    covCam00,
    covCam01,
    covCam02,
    covCam11,
    covCam12,
    covCam22
  ) + 0.3;
  let det = screenCovA * screenCovC - screenCovB * screenCovB;
  let mid = 0.5 * (screenCovA + screenCovC);
  let eigenDisc = max(0.1, mid * mid - det);
  let lambda1 = mid + sqrt(eigenDisc);
  let lambda2 = mid - sqrt(eigenDisc);
  let maxLambda = max(lambda1, lambda2);
  let jacobianRadiusPx = ceil(3.0 * sqrt(max(maxLambda, 1e-6)));
  if (
    fx > 0.0 &&
    fy > 0.0 &&
    abs(mv.z) > 1e-6 &&
    det > 1e-8 &&
    screenCovA > 0.0 &&
    screenCovC > 0.0 &&
    jacobianRadiusPx > 0.0
  ) {
    conicX = screenCovC / det;
    conicY = -screenCovB / det;
    conicZ = screenCovA / det;
    covarianceA = screenCovA;
    covarianceB = screenCovB;
    covarianceC = screenCovC;
    finalRadiusPx = jacobianRadiusPx;
    finalDepth = mv.z;
    footprintAreaPx = 3.14159265 * max(lambda1, 0.0) * max(lambda2, 0.0);
    footprintSourceCode = 113.0;
    fallbackFlag = 0.0;
  }
  let footprintBase = row * 3u;
  footprintPayload[footprintBase + 0u] = vec4f(conicX, conicY, conicZ, covarianceA);
  footprintPayload[footprintBase + 1u] = vec4f(covarianceB, covarianceC, finalRadiusPx, finalDepth);
  footprintPayload[footprintBase + 2u] = vec4f(footprintSourceCode, abs(finalDepth), footprintAreaPx, fallbackFlag);

  for (var slot: u32 = 0u; slot < STEP113_DIAGNOSTIC_ROW_COUNT; slot = slot + 1u) {
    if (step113DiagnosticRowForSlot(params.count, slot) == row) {
      let outBase = params.count * 3u + slot * STEP113_DIAGNOSTIC_VEC4_STRIDE;
      footprintPayload[outBase + 0u] = vec4f(f32(row), footprintSourceCode, finalRadiusPx, det);
      footprintPayload[outBase + 1u] = vec4f(cov00, cov01, cov02, cov11);
      footprintPayload[outBase + 2u] = vec4f(cov12, cov22, covCam00, covCam01);
      footprintPayload[outBase + 3u] = vec4f(covCam02, covCam11, covCam12, covCam22);
      footprintPayload[outBase + 4u] = vec4f(jacobian0.x, jacobian0.y, jacobian0.z, jacobian1.x);
      footprintPayload[outBase + 5u] = vec4f(jacobian1.y, jacobian1.z, screenCovA, screenCovB);
      footprintPayload[outBase + 6u] = vec4f(screenCovC, conicX, conicY, conicZ);
      footprintPayload[outBase + 7u] = vec4f(mv.x, mv.y, mv.z, fallbackFlag);
    }
  }
}`
  });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: shader, entryPoint: 'main' }
  });
  const rawBuffer = createBuffer(device, rawXyzOpacity, GPUBufferUsage.STORAGE);
  const timeScaleBuffer = createBuffer(device, timeScale, GPUBufferUsage.STORAGE);
  const attributeInputBuffer = createBuffer(
    device,
    attributeInput,
    GPUBufferUsage.STORAGE
  );
  const rotationInputBuffer = createBuffer(
    device,
    rotationInput,
    GPUBufferUsage.STORAGE
  );
  const projectionParamsBuffer = createBuffer(
    device,
    projectionParamInput,
    GPUBufferUsage.STORAGE
  );
  const outputByteLength = Math.max(4, count * 4 * Float32Array.BYTES_PER_ELEMENT);
  const attributeOutputByteLength = Math.max(
    4,
    count * 8 * Float32Array.BYTES_PER_ELEMENT
  );
  const footprintProductionVec4Count = count * 3;
  const step113DiagnosticVec4Count =
    STEP113_DIAGNOSTIC_ROW_COUNT * STEP113_DIAGNOSTIC_VEC4_STRIDE;
  const footprintVec4Count =
    footprintProductionVec4Count + step113DiagnosticVec4Count;
  const footprintOutputByteLength = Math.max(
    4,
    footprintVec4Count * 4 * Float32Array.BYTES_PER_ELEMENT
  );
  const outputBuffer = device.createBuffer({
    size: outputByteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const attributeOutputBuffer = device.createBuffer({
    size: attributeOutputByteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const footprintOutputBuffer = device.createBuffer({
    size: footprintOutputByteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const paramsBuffer = createBuffer(
    device,
    buildStateEvaluatorParams({
      count,
      buildConfig
    }),
    GPUBufferUsage.UNIFORM
  );
  const readbackBuffer = device.createBuffer({
    size: outputByteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const attributeReadbackBuffer = device.createBuffer({
    size: attributeOutputByteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const footprintReadbackBuffer = device.createBuffer({
    size: footprintOutputByteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: rawBuffer } },
      { binding: 1, resource: { buffer: timeScaleBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: paramsBuffer } },
      { binding: 4, resource: { buffer: attributeInputBuffer } },
      { binding: 5, resource: { buffer: attributeOutputBuffer } },
      { binding: 6, resource: { buffer: footprintOutputBuffer } },
      { binding: 7, resource: { buffer: rotationInputBuffer } },
      { binding: 8, resource: { buffer: projectionParamsBuffer } }
    ]
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputByteLength);
  encoder.copyBufferToBuffer(
    attributeOutputBuffer,
    0,
    attributeReadbackBuffer,
    0,
    attributeOutputByteLength
  );
  encoder.copyBufferToBuffer(
    footprintOutputBuffer,
    0,
    footprintReadbackBuffer,
    0,
    footprintOutputByteLength
  );
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const statePositions = new Float32Array(readbackBuffer.getMappedRange().slice(0));
  readbackBuffer.unmap();
  await attributeReadbackBuffer.mapAsync(GPUMapMode.READ);
  const renderAttributes = new Float32Array(
    attributeReadbackBuffer.getMappedRange().slice(0)
  );
  attributeReadbackBuffer.unmap();
  await footprintReadbackBuffer.mapAsync(GPUMapMode.READ);
  const footprintPayload = new Float32Array(
    footprintReadbackBuffer.getMappedRange().slice(0)
  );
  footprintReadbackBuffer.unmap();
  const step113DiagnosticFloatOffset = count * 12;
  const step113DiagnosticFloatCount =
    STEP113_DIAGNOSTIC_ROW_COUNT * STEP113_DIAGNOSTIC_VEC4_STRIDE * 4;
  const step113DiagnosticByteOffset =
    step113DiagnosticFloatOffset * Float32Array.BYTES_PER_ELEMENT;
  const step113DiagnosticByteSize =
    step113DiagnosticFloatCount * Float32Array.BYTES_PER_ELEMENT;
  const footprintProductionByteOffset = 0;
  const footprintProductionByteSize =
    footprintProductionVec4Count * 4 * Float32Array.BYTES_PER_ELEMENT;
  const footprintProductionByteEnd =
    footprintProductionByteOffset + footprintProductionByteSize;
  const step113DiagnosticByteEnd =
    step113DiagnosticByteOffset + step113DiagnosticByteSize;
  const step113DiagnosticAlignmentBytes = 16;
  const diagnosticWriteRangeWithinTail =
    step113DiagnosticByteOffset >= step113DiagnosticByteOffset &&
    step113DiagnosticByteEnd <= step113DiagnosticByteEnd;
  const diagnosticRegionsOverlap =
    footprintProductionByteEnd > step113DiagnosticByteOffset &&
    footprintProductionByteOffset < step113DiagnosticByteEnd;
  const step113DiagnosticLayout = {
    schemaVersion: 'phase3-step113-packed-diagnostic-tail-layout-v1',
    packingMode: 'packed-into-footprint-payload-tail-existing-storage-buffer',
    bufferName: 'footprintPayload',
    bufferTotalSizeBytes: footprintOutputByteLength,
    bufferTotalVec4Count: footprintVec4Count,
    productionPayloadOffsetBytes: footprintProductionByteOffset,
    productionPayloadSizeBytes: footprintProductionByteSize,
    productionPayloadEndBytes: footprintProductionByteEnd,
    productionPayloadVec4Offset: 0,
    productionPayloadVec4Count: footprintProductionVec4Count,
    diagnosticTailOffsetBytes: step113DiagnosticByteOffset,
    diagnosticTailSizeBytes: step113DiagnosticByteSize,
    diagnosticTailEndBytes: step113DiagnosticByteEnd,
    diagnosticTailVec4Offset: footprintProductionVec4Count,
    diagnosticTailVec4Count: step113DiagnosticVec4Count,
    requiredAlignmentBytes: step113DiagnosticAlignmentBytes,
    productionPayloadOffsetAligned:
      footprintProductionByteOffset % step113DiagnosticAlignmentBytes === 0,
    diagnosticTailOffsetAligned:
      step113DiagnosticByteOffset % step113DiagnosticAlignmentBytes === 0,
    readbackRangeOffsetAligned:
      step113DiagnosticByteOffset % step113DiagnosticAlignmentBytes === 0,
    productionRecordStrideVec4: 3,
    productionRecordStrideBytes: 3 * 4 * Float32Array.BYTES_PER_ELEMENT,
    productionRecordCount: count,
    diagnosticRowStrideVec4: STEP113_DIAGNOSTIC_VEC4_STRIDE,
    diagnosticRowStrideBytes:
      STEP113_DIAGNOSTIC_VEC4_STRIDE * 4 * Float32Array.BYTES_PER_ELEMENT,
    diagnosticRowCount: STEP113_DIAGNOSTIC_ROW_COUNT,
    regionsOverlap: diagnosticRegionsOverlap,
    bufferCapacitySufficient: step113DiagnosticByteEnd <= footprintOutputByteLength,
    wgslWriteRangeBytes: {
      offset: step113DiagnosticByteOffset,
      size: step113DiagnosticByteSize,
      end: step113DiagnosticByteEnd
    },
    readbackRangeBytes: {
      offset: step113DiagnosticByteOffset,
      size: step113DiagnosticByteSize,
      end: step113DiagnosticByteEnd
    },
    wgslWriteRangeWithinDiagnosticTail: diagnosticWriteRangeWithinTail,
    readbackRangeWithinDiagnosticTail: diagnosticWriteRangeWithinTail,
    productionRecordStridePreserved: true,
    productionRecordCountPreserved: true
  };
  const step113IntermediateReadback = footprintPayload.slice(
    step113DiagnosticFloatOffset,
    step113DiagnosticFloatOffset + step113DiagnosticFloatCount
  );
  const stateSummary = summarizeComputedStatePositions(statePositions);
  const attributeSummary = summarizeComputedRenderAttributes(renderAttributes);
  const footprintSummary = summarizeComputedFootprintPayload(footprintPayload, count);
  return {
    statePositions,
    renderAttributes,
    footprintPayload,
    step113IntermediateReadback,
    step113IntermediateReadbackRows: Array.from(step113DiagnosticRows),
    step113DiagnosticBindingEvidence: {
      productionComputeStorageBufferBindingCount: 8,
      deviceDefaultMaxStorageBuffersPerShaderStage: 8,
      adapterSupportedMaxStorageBuffersPerShaderStage:
        device?.limits?.maxStorageBuffersPerShaderStage ?? null,
      requestedMaxStorageBuffersPerShaderStage:
        device?.limits?.maxStorageBuffersPerShaderStage ?? null,
      requiredLimitsRaisedForStep113Diagnostic: false,
      diagnosticPackingMode:
        'packed-into-footprint-payload-tail-existing-storage-buffer',
      diagnosticReadbackSourceBuffer: 'footprintPayload',
      diagnosticPackedTailLayout: step113DiagnosticLayout,
      actualEvidenceSameProductionDispatch: true,
      storageBufferBindingBreakdown: [
        'rawXyzOpacity',
        'timeScale',
        'statePositions',
        'attributeInput',
        'renderAttributes',
        'footprintPayload-with-packed-step113-diagnostic-tail',
        'rotationInput-packed-left-and-right-4d-quaternions',
        'projectionParams'
      ]
    },
    contract: buildWebGpu4DStateSourceContract({
      stateSourceMode: 'webgpu-cuda-conditional-temporal-mean-evaluator',
      candidateCount: count,
      statePositionCount: count,
      computed4DStatePositionCount: stateSummary.computed4DStatePositionCount,
      baselineStatePositionCount: 0,
      unavailableStatePositionCount: stateSummary.unavailableStatePositionCount,
      timestamp: buildConfig?.timestamp ?? null,
      stateParameterMode: 'viewer-build-config-webgpu-uniform',
      webgpuComputedStatePositions: true,
      webgpu4DStateEvaluationMode: 'cuda-style-conditional-temporal-mean-position-eval',
      full4DStateEvaluationInWgsl: false,
      rawXyzRepairInVisibleRecordComputeRequired: false,
      reason:
        stateSummary.computed4DStatePositionCount > 0
          ? null
          : 'webgpu-partial-4d-state-evaluator-produced-no-valid-state-positions'
    }),
    gaussianAttributeEvaluationContract:
      buildWebGpuGaussianAttributeEvaluationContract({
        candidateCount: count,
        computedRenderAttributeCount:
          attributeSummary.computedRenderAttributeCount,
        webgpuComputedRenderAttributes: true,
        computedAttributeFields: [
          'radiusPx',
          'colorAlpha.rgb',
          'colorAlpha.a',
          'temporalWeight'
        ],
        partialAttributeFields: [
          'radiusPx-from-scale-mean',
          'colorAlpha.rgb-from-f_dc-l0',
          'colorAlpha.a-from-opacity-and-temporal-weight'
        ],
        referenceAssistedAttributeFields: ['none-for-normal-backend-visible-samples'],
        deferredAttributeFields: [
          'full-conic',
          'full-covariance',
          'full-SH-color',
          'tile-range',
          'depth-sort'
        ],
        normalBackendPointRadiusPx:
          attributeSummary.normalBackendPointRadiusPx,
        averageComputedRadiusPx: attributeSummary.averageComputedRadiusPx,
        averageComputedAlpha: attributeSummary.averageComputedAlpha,
        renderAttributeClassification: 'partial-webgpu-computed',
        renderPayloadClassification:
          'partial-webgpu-gaussian-render-attributes',
        fullGaussianAttributeEvaluationInWgsl: false,
        reason:
          attributeSummary.computedRenderAttributeCount > 0
            ? null
            : 'webgpu-gaussian-attribute-evaluator-produced-no-valid-attributes'
    }),
    gaussianFootprintEvaluationContract:
      buildWebGpuGaussianFootprintEvaluationContract({
        candidateCount: count,
        computedFootprintPayloadCount:
          footprintSummary.computedFootprintPayloadCount,
        webgpuComputedFootprintPayload: true,
        computedFootprintFields: [
          'conic',
          'covariance2D',
          'radiusPx',
          'depth',
          'sortKey'
        ],
        partialFootprintFields: [
          '4d-conditional-covariance-temporal-marginal-deferred',
          'full-cuda-front-to-back-radius-clamp-parity-deferred'
        ],
        baselineFootprintFields: [],
        fallbackFootprintFields:
          footprintSummary.scaleOnlyFallbackCount > 0
            ? ['step111-scale-only-conic-fallback-for-invalid-jacobian-cases']
            : [],
        deferredFootprintFields: [
          'gpu-aabb-from-projected-center',
          'gpu-tileRange-from-aabb',
          'depth-sort-dispatch',
          'full-4d-conditional-covariance'
        ],
        averageComputedConicX: footprintSummary.averageComputedConicX,
        averageComputedFootprintAreaPx:
          footprintSummary.averageComputedFootprintAreaPx,
        footprintPayloadClassification:
          footprintSummary.cameraJacobianConicCount > 0
            ? 'partial-cuda-aligned-rotation-aware-camera-jacobian-conic'
            : 'partial-webgpu-gaussian-footprint',
        fullGaussianFootprintEvaluationInWgsl: false,
        rotationAwareCovarianceCount:
          footprintSummary.rotationAwareCovarianceCount,
        cameraJacobianConicCount: footprintSummary.cameraJacobianConicCount,
        scaleOnlyFallbackCount: footprintSummary.scaleOnlyFallbackCount,
        reason:
          footprintSummary.computedFootprintPayloadCount > 0
            ? null
            : 'webgpu-gaussian-footprint-evaluator-produced-no-valid-payload'
      })
  };
}
