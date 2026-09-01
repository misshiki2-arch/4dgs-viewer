export const PRODUCTION_TILE_INPUT_ALPHA_F32_CENTRAL_ORACLE_VERSION =
  'phase3-production-tile-input-alpha-f32-central-v1';

const SCALE_FLOOR = Math.fround(1e-6);
const COVARIANCE_FLOOR = Math.fround(1e-8);

function finiteF32(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return null;
  const rounded = Math.fround(number);
  return Number.isFinite(rounded) ? rounded : null;
}

function finiteF32Vector(value, length) {
  if (
    (!Array.isArray(value) && !ArrayBuffer.isView(value)) ||
    value.length !== length
  ) return null;
  const result = Array.from(value, finiteF32);
  return result.every((entry) => entry != null) ? result : null;
}

const addF32 = (left, right) => Math.fround(left + right);
const subtractF32 = (left, right) => Math.fround(left - right);
const multiplyF32 = (left, right) => Math.fround(left * right);
const divideF32 = (left, right) => Math.fround(left / right);

function dot4F32(left, right) {
  let result = multiplyF32(left[0], right[0]);
  for (let index = 1; index < 4; index += 1) {
    result = addF32(result, multiplyF32(left[index], right[index]));
  }
  return result;
}

function normalizeQuaternionF32(value) {
  const lengthSquared = dot4F32(value, value);
  const length = Math.fround(Math.sqrt(lengthSquared));
  const denominator = Math.max(length, SCALE_FLOOR);
  return value.map((component) => divideF32(component, denominator));
}

function buildRotation4dColumnsF32(qLRaw, qRRaw) {
  const [a, b, c, d] = normalizeQuaternionF32(qLRaw);
  const [p, q, r, s] = normalizeQuaternionF32(qRRaw);
  const mlRows = [
    [a, -b, c, -d],
    [b, a, -d, -c],
    [-c, d, a, -b],
    [d, c, b, a]
  ];
  const mrRows = [
    [p, -q, r, s],
    [q, p, -s, r],
    [-r, s, p, q],
    [-s, -r, -q, p]
  ];
  const mlColumns = Array.from(
    { length: 4 },
    (_, column) => mlRows.map((row) => Math.fround(row[column]))
  );
  return mrRows.map((row) =>
    mlColumns.map((column) => dot4F32(row, column))
  );
}

function sigma4ComponentF32(scaleSquared, columnA, columnB) {
  const scaledColumn = scaleSquared.map((scale, index) =>
    multiplyF32(scale, columnA[index])
  );
  return dot4F32(scaledColumn, columnB);
}

function sigmoidF32(rawOpacityLogit) {
  const exponent = Math.fround(-rawOpacityLogit);
  const expNegative = Math.fround(Math.exp(exponent));
  const denominator = addF32(Math.fround(1), expNegative);
  return divideF32(Math.fround(1), denominator);
}

export function buildProductionTileInputAlphaF32Central({
  rawOpacityLogit,
  sourceScaleXYZ,
  sourceScaleT,
  rotation,
  rotationR,
  timestamp,
  tCenter,
  scalingModifier = 1,
  sigmaScale = 1
} = {}) {
  const opacityLogitF32 = finiteF32(rawOpacityLogit);
  const sourceScaleXYZF32 = finiteF32Vector(sourceScaleXYZ, 3);
  const sourceScaleTF32 = finiteF32(sourceScaleT);
  const rotationF32 = finiteF32Vector(rotation, 4);
  const rotationRF32 = finiteF32Vector(rotationR, 4);
  const timestampF32 = finiteF32(timestamp);
  const tCenterF32 = finiteF32(tCenter);
  const scalingModifierF32 = finiteF32(scalingModifier);
  const sigmaScaleF32 = finiteF32(sigmaScale);
  if (
    opacityLogitF32 == null || !sourceScaleXYZF32 ||
    sourceScaleTF32 == null || !rotationF32 || !rotationRF32 ||
    timestampF32 == null || tCenterF32 == null ||
    scalingModifierF32 == null || sigmaScaleF32 == null
  ) return null;

  const scaleXYZ = sourceScaleXYZF32.map((scale) =>
    Math.max(multiplyF32(scale, scalingModifierF32), SCALE_FLOOR)
  );
  const scaleT = Math.max(
    multiplyF32(
      multiplyF32(sourceScaleTF32, scalingModifierF32),
      sigmaScaleF32
    ),
    SCALE_FLOOR
  );
  const rotationColumns = buildRotation4dColumnsF32(rotationF32, rotationRF32);
  const temporalColumn = rotationColumns.map((column) => column[3]);
  const scaleSquared = [...scaleXYZ, scaleT].map((scale) =>
    multiplyF32(scale, scale)
  );
  const covarianceT = Math.max(
    sigma4ComponentF32(scaleSquared, temporalColumn, temporalColumn),
    COVARIANCE_FLOOR
  );
  const dt = subtractF32(timestampF32, tCenterF32);
  const exponent = divideF32(
    multiplyF32(multiplyF32(Math.fround(-0.5), dt), dt),
    Math.max(covarianceT, COVARIANCE_FLOOR)
  );
  const temporalWeight = Math.fround(Math.exp(exponent));
  const activatedOpacity = sigmoidF32(opacityLogitF32);
  const alpha = multiplyF32(activatedOpacity, temporalWeight);
  if (
    !Number.isFinite(temporalWeight) || temporalWeight < 0 ||
    !Number.isFinite(alpha)
  ) return null;
  return Object.freeze({
    temporalWeight,
    alpha
  });
}
