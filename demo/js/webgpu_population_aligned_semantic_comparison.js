import {
  POPULATION_SEMANTIC_ACTUAL_PROVENANCE,
  POPULATION_SEMANTIC_COMPARISON_CONTRACT_NAME,
  POPULATION_SEMANTIC_COMPARISON_SCHEMA_VERSION,
  POPULATION_SEMANTIC_EVIDENCE_FLOAT_STRIDE,
  POPULATION_SEMANTIC_EXPECTED_PROVENANCE,
  POPULATION_SEMANTIC_MAX_FIRST_MISMATCHES,
  POPULATION_SEMANTIC_STAGE_LOCAL_REPRESENTATIVE_LIMIT,
  POPULATION_RASTER_SEMANTIC_ACTUAL_PROVENANCE,
  POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE,
  POPULATION_RASTER_SEMANTIC_COMPANION_LAYOUT_SCHEMA_VERSION,
  POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE,
  POPULATION_SEMANTIC_COMPLETE_STAGE_CLASSIFICATIONS,
  POPULATION_PIXEL_BOUNDARY_PRECISION_CLASSIFICATION_PROVENANCE,
  POPULATION_SEMANTIC_STAGE_CONTRACTS,
  PRODUCTION_RESIDENT_RANGE_START,
  buildPopulationSemanticComparisonInputContract,
  buildPopulationSemanticCoverageContract,
  buildPopulationSemanticDiagnosticWorksetResourceIdentity,
  buildPopulationSemanticStageLocalMismatchRepresentative,
  buildPopulationSemanticStageLocalMismatchSummaries,
  classifyPopulationSemanticStageEvidence,
  validatePopulationSemanticStageLocalMismatchSummaries
} from './common_4dgs_population_semantic_comparison_contracts.js';
import {
  computeCudaConditionalGaussianState4D
} from './cuda_4d_state.js';
import {
  buildWebGpu4DStatePositionsForCandidates
} from './webgpu_4d_state_evaluator.js';
import {
  buildNativeWebGpuProductionTileInput
} from './webgpu_production_tile_input.js';
import {
  observePopulationRasterSemanticCompanion
} from './webgpu_population_raster_semantic_observer.js';

const TEMPORAL_ELIGIBILITY_THRESHOLD = 0.05;
const PRODUCTION_TILE_SIZE = 16;
const RASTER_STAGE_KEYS = new Set([
  'productionRasterEligibility',
  'projectedCenter',
  'cameraDepth',
  'webgpuInclusivePixelBounds',
  'normalizedInclusiveTileBounds'
]);
const RASTER_VALUE_STAGE_KEYS = new Set([
  'projectedCenter',
  'cameraDepth',
  'webgpuInclusivePixelBounds',
  'normalizedInclusiveTileBounds'
]);

function finite(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function readRawVector(array, base, count) {
  if (!array || base < 0 || base + count > array.length) return null;
  const values = Array.from({ length: count }, (_, index) =>
    finite(array[base + index])
  );
  return values.every((value) => value != null) ? values : null;
}

function cov3Apply(covariance, vector) {
  return [
    covariance[0] * vector[0] + covariance[1] * vector[1] + covariance[2] * vector[2],
    covariance[1] * vector[0] + covariance[3] * vector[1] + covariance[4] * vector[2],
    covariance[2] * vector[0] + covariance[4] * vector[1] + covariance[5] * vector[2]
  ];
}

function cov3Bilinear(a, b, covariance) {
  const applied = cov3Apply(covariance, b);
  return a[0] * applied[0] + a[1] * applied[1] + a[2] * applied[2];
}

function readProjectionRows(projectionParams) {
  if (!(projectionParams instanceof Float32Array) || projectionParams.length < 24) {
    return null;
  }
  const row = (index) => Array.from(
    projectionParams.subarray(index * 4, index * 4 + 4),
    Number
  );
  const rows = [row(3), row(4), row(5)];
  const values = {
    mode: Number(projectionParams[0]),
    renderWidth: Math.max(Math.abs(Number(projectionParams[1])), 1),
    renderHeight: Math.max(Math.abs(Number(projectionParams[2])), 1),
    sx: Number(projectionParams[4]),
    sy: Number(projectionParams[5]),
    pixelXSign: Number(projectionParams[6]),
    fx: Math.abs(Number(projectionParams[8])),
    fy: Math.abs(Number(projectionParams[9])),
    cx: Number(projectionParams[10]),
    cy: Number(projectionParams[11]),
    viewRows: rows
  };
  return [
    values.mode,
    values.renderWidth,
    values.renderHeight,
    values.sx,
    values.sy,
    values.pixelXSign,
    values.fx,
    values.fy,
    values.cx,
    values.cy,
    ...rows.flat()
  ].every(Number.isFinite)
    ? {
        ...values,
        canvasWidth: Math.max(1, Math.round(values.renderWidth * values.sx)),
        canvasHeight: Math.max(1, Math.round(values.renderHeight * values.sy))
      }
    : null;
}

function dotViewRow(row, position) {
  return row[0] * position[0] + row[1] * position[1] +
    row[2] * position[2] + row[3];
}

function clampInteger(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

export function buildCudaNormalizedInclusiveTileRect({
  centerX,
  centerY,
  radius,
  canvasWidth,
  canvasHeight,
  tileSize = PRODUCTION_TILE_SIZE
} = {}) {
  const values = [centerX, centerY, radius, canvasWidth, canvasHeight, tileSize]
    .map(Number);
  if (
    !values.every(Number.isFinite) ||
    !Number.isInteger(values[2]) || values[2] < 0 ||
    !Number.isInteger(values[3]) || values[3] <= 0 ||
    !Number.isInteger(values[4]) || values[4] <= 0 ||
    !Number.isInteger(values[5]) || values[5] <= 0
  ) return null;
  const [px, py, integerRadius, width, height, size] = values;
  const tileCols = Math.ceil(width / size);
  const tileRows = Math.ceil(height / size);
  const minX = clampInteger(Math.trunc((px - integerRadius) / size), 0, tileCols);
  const minY = clampInteger(Math.trunc((py - integerRadius) / size), 0, tileRows);
  const maxExclusiveX = clampInteger(
    Math.trunc((px + integerRadius + size - 1) / size),
    0,
    tileCols
  );
  const maxExclusiveY = clampInteger(
    Math.trunc((py + integerRadius + size - 1) / size),
    0,
    tileRows
  );
  const nonEmpty = maxExclusiveX > minX && maxExclusiveY > minY;
  return {
    minInclusive: [minX, minY],
    maxExclusive: [maxExclusiveX, maxExclusiveY],
    normalizedInclusive: nonEmpty
      ? [minX, minY, maxExclusiveX - 1, maxExclusiveY - 1]
      : null,
    nonEmpty,
    tileCols,
    tileRows
  };
}

export function buildExpectedWebGpuInclusivePixelBounds({
  centerX,
  centerY,
  radius,
  canvasWidth,
  canvasHeight
} = {}) {
  const values = [centerX, centerY, radius, canvasWidth, canvasHeight].map(Number);
  if (
    !values.every(Number.isFinite) ||
    !Number.isInteger(values[3]) || values[3] <= 0 ||
    !Number.isInteger(values[4]) || values[4] <= 0
  ) return null;
  const [px, py, radiusPixels, width, height] = values;
  return [
    Math.min(width - 1, Math.max(0, Math.floor(px - radiusPixels))),
    Math.min(height - 1, Math.max(0, Math.floor(py - radiusPixels))),
    Math.min(width - 1, Math.max(0, Math.ceil(px + radiusPixels))),
    Math.min(height - 1, Math.max(0, Math.ceil(py + radiusPixels)))
  ];
}

function pixelBoundaryPrecisionClassification({
  expectedRecord,
  actualRecord,
  actualRasterCompanionContract,
  stageMismatches,
  rowIdentityValid
} = {}) {
  if (!Array.isArray(stageMismatches) || stageMismatches.length === 0) {
    return 'mismatch';
  }
  const centerContract = POPULATION_SEMANTIC_STAGE_CONTRACTS.find(
    ({ key }) => key === 'projectedCenter'
  );
  const tolerance = centerContract?.tolerance;
  const expectedCenter = Array.from(
    expectedRecord?.stages?.projectedCenter?.values ?? [],
    Number
  );
  const actualCenter = Array.from(
    actualRecord?.stages?.projectedCenter?.values ?? [],
    Number
  );
  const expectedRadiusValues = Array.from(
    expectedRecord?.stages?.radius?.values ?? [],
    Number
  );
  const actualRadiusValues = Array.from(
    actualRecord?.stages?.radius?.values ?? [],
    Number
  );
  const expectedBounds = Array.from(
    expectedRecord?.stages?.webgpuInclusivePixelBounds?.values ?? [],
    Number
  );
  const actualBounds = Array.from(
    actualRecord?.stages?.webgpuInclusivePixelBounds?.values ?? [],
    Number
  );
  const expectedEligibility = Array.from(
    expectedRecord?.stages?.productionRasterEligibility?.values ?? [],
    Number
  );
  const actualEligibility = Array.from(
    actualRecord?.stages?.productionRasterEligibility?.values ?? [],
    Number
  );
  const width = Number(actualRasterCompanionContract?.canvasWidth);
  const height = Number(actualRasterCompanionContract?.canvasHeight);
  if (
    !Number.isFinite(tolerance) || tolerance < 0 ||
    expectedCenter.length !== 2 || actualCenter.length !== 2 ||
    !expectedCenter.every(Number.isFinite) ||
    !actualCenter.every(Number.isFinite) ||
    expectedRadiusValues.length !== 1 || actualRadiusValues.length !== 1 ||
    expectedBounds.length !== 4 || actualBounds.length !== 4 ||
    !expectedBounds.every(Number.isInteger) ||
    !actualBounds.every(Number.isInteger) ||
    expectedEligibility.length !== 1 || actualEligibility.length !== 1 ||
    expectedEligibility[0] !== 1 || actualEligibility[0] !== 1 ||
    expectedRecord?.stages?.projectedCenter?.valid !== true ||
    actualRecord?.stages?.projectedCenter?.valid !== true ||
    expectedRecord?.stages?.radius?.valid !== true ||
    actualRecord?.stages?.radius?.valid !== true ||
    expectedRecord?.stages?.productionRasterEligibility?.valid !== true ||
    actualRecord?.stages?.productionRasterEligibility?.valid !== true ||
    expectedRecord?.stages?.webgpuInclusivePixelBounds?.valid !== true ||
    actualRecord?.stages?.webgpuInclusivePixelBounds?.valid !== true ||
    expectedRecord?.rasterEligible !== true || actualRecord?.rasterEligible !== true ||
    rowIdentityValid !== true ||
    !Number.isInteger(width) || width <= 0 ||
    !Number.isInteger(height) || height <= 0
  ) return 'mismatch';
  const expectedRadius = expectedRadiusValues[0];
  const actualRadius = actualRadiusValues[0];
  if (
    !Number.isInteger(expectedRadius) || expectedRadius <= 0 ||
    actualRadius !== expectedRadius ||
    expectedCenter.some(
      (value, axis) => Math.abs(actualCenter[axis] - value) > tolerance
    )
  ) return 'mismatch';

  const buildBounds = (center) => buildExpectedWebGpuInclusivePixelBounds({
    centerX: center[0],
    centerY: center[1],
    radius: expectedRadius,
    canvasWidth: width,
    canvasHeight: height
  });
  const expectedDependencyBounds = buildBounds(expectedCenter);
  const actualDependencyBounds = buildExpectedWebGpuInclusivePixelBounds({
    centerX: actualCenter[0],
    centerY: actualCenter[1],
    radius: actualRadius,
    canvasWidth: width,
    canvasHeight: height
  });
  if (
    !expectedDependencyBounds || !actualDependencyBounds ||
    expectedBounds.some((value, index) => value !== expectedDependencyBounds[index]) ||
    actualBounds.some((value, index) => value !== actualDependencyBounds[index])
  ) return 'mismatch';

  const envelopeBounds = [
    [expectedCenter[0] - tolerance, expectedCenter[1] - tolerance],
    [expectedCenter[0] - tolerance, expectedCenter[1] + tolerance],
    [expectedCenter[0] + tolerance, expectedCenter[1] - tolerance],
    [expectedCenter[0] + tolerance, expectedCenter[1] + tolerance]
  ].map(buildBounds);
  if (envelopeBounds.some((bounds) => bounds == null)) return 'mismatch';
  for (let componentIndex = 0; componentIndex < 4; componentIndex += 1) {
    const allowedValues = envelopeBounds.map((bounds) => bounds[componentIndex]);
    if (
      actualBounds[componentIndex] < Math.min(...allowedValues) ||
      actualBounds[componentIndex] > Math.max(...allowedValues)
    ) return 'mismatch';
  }
  if (stageMismatches.some(({ componentIndex }) =>
    !Number.isSafeInteger(componentIndex) || componentIndex < 0 || componentIndex >= 4
  )) return 'mismatch';
  return 'precision-aligned';
}

function invalidExpectedRecord(reason) {
  return {
    valid: false,
    reason,
    temporalEligible: null,
    stages: Object.fromEntries(
      POPULATION_SEMANTIC_STAGE_CONTRACTS.map(({ key }) => [
        key,
        { valid: false, values: null }
      ])
    )
  };
}

export function buildPopulationAlignedSemanticExpectedRecord({
  raw,
  srcIndex,
  buildConfig,
  projectionParams
} = {}) {
  const xyzDim = Number(raw?.xyzDim);
  const opacityDim = Number(raw?.opacityDim);
  const scaleXYZDim = Number(raw?.scaleXYZDim);
  const scaleTDim = Number(raw?.scaleTDim);
  const rotationDim = Number(raw?.rotationDim);
  const rotationRDim = Number(raw?.rotationRDim);
  const tDim = Number(raw?.tDim);
  if (
    !Number.isInteger(srcIndex) ||
    ![xyzDim, opacityDim, scaleXYZDim, scaleTDim, rotationDim, rotationRDim, tDim]
      .every(Number.isInteger) ||
    xyzDim < 3 || opacityDim < 1 || scaleXYZDim < 3 || scaleTDim < 1 ||
    rotationDim < 4 || rotationRDim < 4 || tDim < 1
  ) return invalidExpectedRecord('population-aligned-spl4-layout-invalid');

  const position = readRawVector(raw.xyz, srcIndex * xyzDim, 3);
  const opacity = finite(raw.opacity?.[srcIndex * opacityDim]);
  const sourceScale = readRawVector(raw.scale_xyz, srcIndex * scaleXYZDim, 3);
  const sourceScaleT = finite(raw.scale_t?.[srcIndex * scaleTDim]);
  const rotation = readRawVector(raw.rotation, srcIndex * rotationDim, 4);
  const rotationR = readRawVector(raw.rotation_r, srcIndex * rotationRDim, 4);
  const tCenter = finite(raw.t?.[srcIndex * tDim]);
  const timestamp = finite(buildConfig?.timestamp);
  const projection = readProjectionRows(projectionParams);
  if (
    !position || opacity == null || !sourceScale || sourceScaleT == null ||
    !rotation || !rotationR || tCenter == null || timestamp == null || !projection
  ) return invalidExpectedRecord('population-aligned-semantic-input-missing');
  if (projection.mode <= 0.5) {
    return invalidExpectedRecord('cuda-aligned-projection-contract-required');
  }

  const scalingModifier = finite(buildConfig?.scalingModifier) ?? 1;
  const sigmaScale = finite(buildConfig?.sigmaScale) ?? 1;
  const scaleXYZ = sourceScale.map((value) =>
    Math.max(Math.max(value, 1e-6) * scalingModifier, 1e-6)
  );
  const scaleT = Math.max(
    sourceScaleT * scalingModifier * sigmaScale,
    1e-6
  );
  const state = computeCudaConditionalGaussianState4D({
    position,
    opacity,
    scaleXYZ,
    scaleT,
    rotation,
    rotationR,
    timestamp,
    tCenter,
    prefilterVar: -1
  });
  const temporalEligible =
    state.debug.marginal_t > TEMPORAL_ELIGIBILITY_THRESHOLD;
  const covarianceWorld = [
    state.cov3[0][0], state.cov3[0][1], state.cov3[0][2],
    state.cov3[1][1], state.cov3[1][2], state.cov3[2][2]
  ];
  const view = projection.viewRows.map((row) => row.slice(0, 3));
  const covarianceCamera = [
    cov3Bilinear(view[0], view[0], covarianceWorld),
    cov3Bilinear(view[0], view[1], covarianceWorld),
    cov3Bilinear(view[0], view[2], covarianceWorld),
    cov3Bilinear(view[1], view[1], covarianceWorld),
    cov3Bilinear(view[1], view[2], covarianceWorld),
    cov3Bilinear(view[2], view[2], covarianceWorld)
  ];
  const cameraPosition = projection.viewRows.map((row) =>
    dotViewRow(row, state.pos)
  );
  const zSafe = Math.max(Math.abs(cameraPosition[2]), 1e-6);
  const tanFovX = projection.renderWidth / Math.max(2 * projection.fx, 1e-6);
  const tanFovY = projection.renderHeight / Math.max(2 * projection.fy, 1e-6);
  const clampedX = Math.max(
    -1.3 * tanFovX,
    Math.min(1.3 * tanFovX, cameraPosition[0] / zSafe)
  ) * zSafe;
  const clampedY = Math.max(
    -1.3 * tanFovY,
    Math.min(1.3 * tanFovY, cameraPosition[1] / zSafe)
  ) * zSafe;
  const jacobian = [
    projection.fx / zSafe,
    0,
    -(projection.fx * clampedX) / (zSafe * zSafe),
    0,
    projection.fy / zSafe,
    -(projection.fy * clampedY) / (zSafe * zSafe)
  ];
  const jacobian0 = jacobian.slice(0, 3);
  const jacobian1 = jacobian.slice(3, 6);
  const screenCovariance = [
    cov3Bilinear(jacobian0, jacobian0, covarianceCamera) + 0.3,
    cov3Bilinear(jacobian0, jacobian1, covarianceCamera),
    cov3Bilinear(jacobian1, jacobian1, covarianceCamera) + 0.3
  ];
  const determinant =
    screenCovariance[0] * screenCovariance[2] -
    screenCovariance[1] * screenCovariance[1];
  const mid = 0.5 * (screenCovariance[0] + screenCovariance[2]);
  const eigenDisc = Math.max(0.1, mid * mid - determinant);
  const lambda1 = mid + Math.sqrt(eigenDisc);
  const lambda2 = mid - Math.sqrt(eigenDisc);
  const radius = Math.ceil(
    3 * Math.sqrt(Math.max(lambda1, lambda2, 1e-6))
  );
  const projectedCenter = [
    (
      projection.pixelXSign * projection.fx *
        (cameraPosition[0] / cameraPosition[2]) + projection.cx
    ) * projection.sx,
    (
      projection.fy * (cameraPosition[1] / cameraPosition[2]) + projection.cy
    ) * projection.sy
  ];
  const cudaTileRect = buildCudaNormalizedInclusiveTileRect({
    centerX: projectedCenter[0],
    centerY: projectedCenter[1],
    radius,
    canvasWidth: projection.canvasWidth,
    canvasHeight: projection.canvasHeight
  });
  const webgpuPixelBounds = buildExpectedWebGpuInclusivePixelBounds({
    centerX: projectedCenter[0],
    centerY: projectedCenter[1],
    radius,
    canvasWidth: projection.canvasWidth,
    canvasHeight: projection.canvasHeight
  });
  const rasterEligibilityEvidenceValid =
    temporalEligible === false ||
    (
      [
        ...cameraPosition,
        ...projectedCenter,
        determinant,
        radius
      ].every(Number.isFinite) &&
      cudaTileRect != null
    );
  const rasterEligible = rasterEligibilityEvidenceValid && temporalEligible &&
    cameraPosition[2] > 0.2 &&
    determinant !== 0 &&
    Number.isInteger(radius) && radius > 0 &&
    cudaTileRect?.nonEmpty === true;
  const projectedEvidenceValid =
    projection.fx > 0 &&
    projection.fy > 0 &&
    Math.abs(cameraPosition[2]) > 1e-6 &&
    determinant > 1e-8 &&
    screenCovariance[0] > 0 &&
    screenCovariance[2] > 0 &&
    radius > 0;
  const conic = projectedEvidenceValid
    ? [
        screenCovariance[2] / determinant,
        -screenCovariance[1] / determinant,
        screenCovariance[0] / determinant
      ]
    : null;
  const eligibleStageValid = temporalEligible;
  const finiteStage = (values) => values.every(Number.isFinite);
  return {
    valid: true,
    reason: null,
    temporalEligible,
    stages: {
      temporalEligibility: { valid: true, values: [temporalEligible ? 1 : 0] },
      conditionalStatePosition: {
        valid: eligibleStageValid && finiteStage(state.pos),
        values: state.pos
      },
      conditionalWorldCovariance: {
        valid: eligibleStageValid && finiteStage(covarianceWorld),
        values: covarianceWorld
      },
      cameraSpaceCovariance: {
        valid: eligibleStageValid && finiteStage(covarianceCamera),
        values: covarianceCamera
      },
      projectionJacobian: {
        valid: eligibleStageValid && finiteStage(jacobian),
        values: jacobian
      },
      screenCovariance: {
        valid: eligibleStageValid && finiteStage(screenCovariance),
        values: screenCovariance
      },
      conic: {
        valid: eligibleStageValid && projectedEvidenceValid && finiteStage(conic ?? []),
        values: conic
      },
      radius: {
        valid: eligibleStageValid && projectedEvidenceValid && Number.isFinite(radius),
        values: [radius]
      },
      productionRasterEligibility: {
        valid: rasterEligibilityEvidenceValid,
        values: [rasterEligible ? 1 : 0]
      },
      projectedCenter: {
        valid: rasterEligible && finiteStage(projectedCenter),
        values: projectedCenter
      },
      cameraDepth: {
        valid: rasterEligible && Number.isFinite(cameraPosition[2]),
        values: [cameraPosition[2]]
      },
      webgpuInclusivePixelBounds: {
        valid:
          rasterEligible &&
          Array.isArray(webgpuPixelBounds) &&
          webgpuPixelBounds.every(Number.isInteger),
        values: webgpuPixelBounds
      },
      normalizedInclusiveTileBounds: {
        valid:
          rasterEligible &&
          Array.isArray(cudaTileRect?.normalizedInclusive) &&
          cudaTileRect.normalizedInclusive.every(Number.isInteger),
        values: cudaTileRect?.normalizedInclusive ?? null
      }
    },
    rasterEligible
  };
}

export function buildExplicitPopulationChunkIndices(inputContract) {
  if (inputContract?.status !== 'ready') return new Uint32Array(0);
  return Uint32Array.from(
    { length: inputContract.appliedRangeCount },
    (_, localRow) => inputContract.appliedRangeStart + localRow
  );
}

function buildPackedRawXyzOpacity(raw, candidateIndices) {
  const result = new Float32Array(candidateIndices.length * 4);
  for (let localRow = 0; localRow < candidateIndices.length; localRow += 1) {
    const srcIndex = candidateIndices[localRow];
    const xyz = readRawVector(raw?.xyz, srcIndex * Number(raw?.xyzDim), 3);
    const opacity = finite(raw?.opacity?.[srcIndex * Number(raw?.opacityDim)]);
    if (!xyz || opacity == null) return null;
    result.set([...xyz, opacity], localRow * 4);
  }
  return result;
}

function destroyBuffers(buffers) {
  for (const buffer of buffers) {
    if (typeof buffer?.destroy === 'function') buffer.destroy();
  }
}

function readActualRecord(actualPackedEvidence, localRow) {
  const base = localRow * POPULATION_SEMANTIC_EVIDENCE_FLOAT_STRIDE;
  if (!actualPackedEvidence || base + 31 >= actualPackedEvidence.length) {
    return { missing: true, row: null, temporalEligible: null, stages: {} };
  }
  const values = Array.from(
    actualPackedEvidence.slice(base, base + 32),
    Number
  );
  const row = Number.isSafeInteger(values[0]) ? values[0] : null;
  const eligibilityValue = values[1];
  const temporalEligibilityValid =
    eligibilityValue === 0 || eligibilityValue === 1;
  const temporalEligible = temporalEligibilityValid
    ? eligibilityValue === 1
    : null;
  const stage = (slice, valid = temporalEligible === true) => ({
    valid: valid && slice.every(Number.isFinite),
    values: slice
  });
  const sourceCode = values[30];
  const fallbackFlag = values[31];
  const projectionOutputValid =
    temporalEligible === true && sourceCode === 113 && fallbackFlag === 0;
  return {
    missing: false,
    row,
    temporalEligible,
    stages: {
      temporalEligibility: {
        valid: temporalEligibilityValid,
        values: [temporalEligible ? 1 : 0]
      },
      conditionalStatePosition: stage(values.slice(2, 5)),
      conditionalWorldCovariance: stage(values.slice(5, 11)),
      cameraSpaceCovariance: stage(values.slice(11, 17)),
      projectionJacobian: stage(values.slice(17, 23)),
      screenCovariance: stage(values.slice(23, 26)),
      conic: stage(values.slice(26, 29), projectionOutputValid),
      radius: stage(values.slice(29, 30), projectionOutputValid)
    }
  };
}

function readablePackedEvidence(value) {
  return Array.isArray(value) || (
    ArrayBuffer.isView(value) &&
    typeof value?.length === 'number' &&
    typeof value?.slice === 'function'
  );
}

function missingRasterActualRecord() {
  return {
    rasterEligible: null,
    stages: Object.fromEntries(
      [...RASTER_STAGE_KEYS].map((key) => [
        key,
        { valid: false, missing: true, values: null }
      ])
    )
  };
}

function readActualRasterRecord(actualPackedEvidence, localRow, layout) {
  const base = localRow * POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE;
  if (
    !readablePackedEvidence(actualPackedEvidence) ||
    base + POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE >
      actualPackedEvidence.length
  ) return missingRasterActualRecord();
  const values = Array.from(
    actualPackedEvidence.slice(
      base,
      base + POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE
    ),
    Number
  );
  const allFinite = values.every(Number.isFinite);
  const eligibilityValid = values[0] === 0 || values[0] === 1;
  const rasterEligible = eligibilityValid ? values[0] === 1 : null;
  const pixelBounds = values.slice(4, 8);
  const tileBounds = values.slice(8, 12);
  const pixelBoundsValid =
    allFinite &&
    pixelBounds.every(Number.isInteger) &&
    pixelBounds[0] >= 0 && pixelBounds[1] >= 0 &&
    pixelBounds[2] >= pixelBounds[0] &&
    pixelBounds[3] >= pixelBounds[1] &&
    pixelBounds[2] < layout.canvasWidth &&
    pixelBounds[3] < layout.canvasHeight;
  const tileBoundsValid =
    allFinite &&
    tileBounds.every(Number.isInteger) &&
    tileBounds[0] >= 0 && tileBounds[1] >= 0 &&
    tileBounds[2] >= tileBounds[0] &&
    tileBounds[3] >= tileBounds[1] &&
    tileBounds[2] < layout.tileCols &&
    tileBounds[3] < layout.tileRows;
  const centerDepthValid =
    allFinite &&
    values.slice(1, 4).every(Number.isFinite) &&
    (!rasterEligible || values[3] > 0);
  return {
    rasterEligible,
    stages: {
      productionRasterEligibility: {
        valid: allFinite && eligibilityValid,
        missing: false,
        values: eligibilityValid ? [rasterEligible ? 1 : 0] : null
      },
      projectedCenter: {
        valid: rasterEligible === true && centerDepthValid,
        missing: false,
        values: values.slice(1, 3)
      },
      cameraDepth: {
        valid: rasterEligible === true && centerDepthValid,
        missing: false,
        values: values.slice(3, 4)
      },
      webgpuInclusivePixelBounds: {
        valid: rasterEligible === true && pixelBoundsValid,
        missing: false,
        values: pixelBounds
      },
      normalizedInclusiveTileBounds: {
        valid: rasterEligible === true && tileBoundsValid,
        missing: false,
        values: tileBounds
      }
    }
  };
}

function validateRasterCompanionEvidence({
  inputContract,
  actualRasterCompanionEvidence,
  actualRasterCompanionContract
}) {
  const requestedCount = inputContract?.appliedRangeCount ?? 0;
  const expectedFloatCount =
    requestedCount * POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE;
  const readable = readablePackedEvidence(actualRasterCompanionEvidence);
  const actualFloatCount = readable ? actualRasterCompanionEvidence.length : 0;
  const expectedSourceWorksetResourceIdentity =
    buildPopulationSemanticDiagnosticWorksetResourceIdentity(inputContract);
  const reasons = [];
  if (!readable) reasons.push('raster-companion-evidence-not-readable');
  if (actualFloatCount !== expectedFloatCount) {
    reasons.push('raster-companion-evidence-length-mismatch');
  }
  if (
    actualRasterCompanionContract?.schemaVersion !==
      POPULATION_RASTER_SEMANTIC_COMPANION_LAYOUT_SCHEMA_VERSION
  ) reasons.push('raster-companion-schema-mismatch');
  if (actualRasterCompanionContract?.status !== 'ready') {
    reasons.push('raster-companion-contract-not-ready');
  }
  if (actualRasterCompanionContract?.recordCount !== requestedCount) {
    reasons.push('raster-companion-record-count-mismatch');
  }
  if (
    actualRasterCompanionContract?.rowStrideFloats !==
      POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE ||
    actualRasterCompanionContract?.evidenceFloatCount !== expectedFloatCount
  ) reasons.push('raster-companion-layout-mismatch');
  if (
    actualRasterCompanionContract?.rowAlignment !==
      'local-row-matches-explicit-candidate-index-order'
  ) reasons.push('raster-companion-row-alignment-mismatch');
  if (
    actualRasterCompanionContract?.sourceWorksetResourceIdentity !==
      expectedSourceWorksetResourceIdentity
  ) reasons.push('raster-companion-workset-identity-mismatch');
  for (const field of [
    'sourceStateResourceIdentity',
    'sourceTileInputResourceIdentity'
  ]) {
    if (
      typeof actualRasterCompanionContract?.[field] !== 'string' ||
      actualRasterCompanionContract[field].length <= 0
    ) reasons.push(`raster-companion-${field}-missing`);
  }
  for (const field of ['canvasWidth', 'canvasHeight', 'tileSize', 'tileCols', 'tileRows']) {
    if (
      !Number.isInteger(actualRasterCompanionContract?.[field]) ||
      actualRasterCompanionContract[field] <= 0
    ) reasons.push(`raster-companion-${field}-invalid`);
  }
  if (
    Number.isInteger(actualRasterCompanionContract?.canvasWidth) &&
    Number.isInteger(actualRasterCompanionContract?.tileSize) &&
    actualRasterCompanionContract?.tileCols !== Math.ceil(
      actualRasterCompanionContract.canvasWidth /
      actualRasterCompanionContract.tileSize
    )
  ) reasons.push('raster-companion-tile-cols-mismatch');
  if (
    Number.isInteger(actualRasterCompanionContract?.canvasHeight) &&
    Number.isInteger(actualRasterCompanionContract?.tileSize) &&
    actualRasterCompanionContract?.tileRows !== Math.ceil(
      actualRasterCompanionContract.canvasHeight /
      actualRasterCompanionContract.tileSize
    )
  ) reasons.push('raster-companion-tile-rows-mismatch');
  return {
    status: reasons.length === 0 ? 'ready' : 'blocked',
    reason: reasons[0] ?? null,
    blockedReasons: [...new Set(reasons)],
    requestedRecordCount: requestedCount,
    actualRecordCount:
      readable && actualFloatCount % POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE === 0
        ? actualFloatCount / POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE
        : 0,
    expectedFloatCount,
    actualFloatCount,
    evidenceLengthExact: readable && actualFloatCount === expectedFloatCount,
    rowAlignmentVerified: reasons.every(
      (reason) => !reason.includes('identity') && !reason.includes('row-alignment')
    )
  };
}

function createStageSummary(stageContract) {
  return {
    stage: stageContract.key,
    components: [...stageContract.components],
    comparedCount: 0,
    comparedComponentCount: 0,
    validCount: 0,
    notApplicableCount: 0,
    missingCount: 0,
    invalidCount: 0,
    missingInvalidCount: 0,
    mismatchCount: 0,
    componentMismatchCount: 0,
    precisionAlignedCount: 0,
    precisionAlignedComponentCount: 0,
    semanticResidualCount: 0,
    semanticResidualComponentCount: 0,
    maxAbsoluteError: null,
    tolerance: stageContract.tolerance,
    classification: 'blocked-no-valid-evidence'
  };
}

export function comparePopulationAlignedSemanticChunkEvidence({
  inputContract,
  candidateIndices,
  actualPackedEvidence,
  actualRasterCompanionEvidence,
  actualRasterCompanionContract,
  expectedRecordForRow,
  chunkIndex = 0
} = {}) {
  const requestedCount = inputContract?.appliedRangeCount ?? 0;
  const packedEvidenceReadable = readablePackedEvidence(actualPackedEvidence);
  const actualFloatCount = packedEvidenceReadable
    ? actualPackedEvidence.length
    : 0;
  const expectedFloatCount =
    requestedCount * POPULATION_SEMANTIC_EVIDENCE_FLOAT_STRIDE;
  const packedEvidenceLengthExact =
    packedEvidenceReadable && actualFloatCount === expectedFloatCount;
  const completeActualRecordCount = packedEvidenceReadable
    ? Math.floor(
        actualFloatCount / POPULATION_SEMANTIC_EVIDENCE_FLOAT_STRIDE
      )
    : 0;
  const actualRecordCount = packedEvidenceLengthExact
    ? completeActualRecordCount
    : !packedEvidenceReadable
      ? 0
      : actualFloatCount < expectedFloatCount
        ? completeActualRecordCount
        : Math.ceil(
            actualFloatCount / POPULATION_SEMANTIC_EVIDENCE_FLOAT_STRIDE
          );
  const actualRecords = Array.from(
    { length: Math.min(completeActualRecordCount, requestedCount) },
    (_, localRow) => readActualRecord(actualPackedEvidence, localRow)
  );
  const rasterCompanionCoverage = validateRasterCompanionEvidence({
    inputContract,
    actualRasterCompanionEvidence,
    actualRasterCompanionContract
  });
  const actualRasterRecords = rasterCompanionCoverage.status === 'ready'
    ? Array.from(
        { length: requestedCount },
        (_, localRow) => readActualRasterRecord(
          actualRasterCompanionEvidence,
          localRow,
          actualRasterCompanionContract
        )
      )
    : [];
  const coverage = buildPopulationSemanticCoverageContract({
    inputContract,
    candidateIndices,
    actualRecordCount,
    actualRows: actualRecords.map((record) => record.row)
  });
  const summaries = Object.fromEntries(
    POPULATION_SEMANTIC_STAGE_CONTRACTS.map((stage) => [
      stage.key,
      createStageSummary(stage)
    ])
  );
  const firstMismatches = [];
  const stageLocalRepresentatives = Object.fromEntries(
    POPULATION_SEMANTIC_STAGE_CONTRACTS.map(({ key }) => [key, []])
  );
  for (let localRow = 0; localRow < requestedCount; localRow += 1) {
    const srcIndex = Number(candidateIndices?.[localRow]);
    const expected = typeof expectedRecordForRow === 'function'
      ? expectedRecordForRow(localRow, srcIndex)
      : invalidExpectedRecord('expected-record-provider-missing');
    const legacyActual = actualRecords[localRow] ?? readActualRecord(null, localRow);
    const rasterActual = actualRasterRecords[localRow] ?? missingRasterActualRecord();
    const actual = {
      ...legacyActual,
      rasterEligible: rasterActual.rasterEligible,
      stages: {
        ...legacyActual.stages,
        ...rasterActual.stages
      }
    };
    for (const stageContract of POPULATION_SEMANTIC_STAGE_CONTRACTS) {
      const summary = summaries[stageContract.key];
      const expectedStage = expected?.stages?.[stageContract.key];
      const actualStage = actual?.stages?.[stageContract.key];
      if (
        !RASTER_STAGE_KEYS.has(stageContract.key) &&
        stageContract.key !== 'temporalEligibility' &&
        expected?.temporalEligible === false &&
        actual?.temporalEligible === false
      ) {
        summary.notApplicableCount += 1;
        continue;
      }
      if (
        RASTER_VALUE_STAGE_KEYS.has(stageContract.key) &&
        typeof expected?.rasterEligible === 'boolean' &&
        typeof actual?.rasterEligible === 'boolean' &&
        (expected.rasterEligible !== true || actual.rasterEligible !== true)
      ) {
        summary.notApplicableCount += 1;
        continue;
      }
      if (
        (!RASTER_STAGE_KEYS.has(stageContract.key) && actual?.missing === true) ||
        actualStage?.missing === true
      ) {
        summary.missingCount += 1;
        continue;
      }
      if (
        expected?.valid !== true ||
        expectedStage?.valid !== true ||
        actualStage?.valid !== true
      ) {
        summary.invalidCount += 1;
        continue;
      }
      const expectedValues = Array.from(expectedStage.values ?? [], Number);
      const actualValues = Array.from(actualStage.values ?? [], Number);
      const componentCount = stageContract.components.length;
      const stageValuesValid =
        expectedValues.length === componentCount &&
        actualValues.length === componentCount &&
        expectedValues.every(Number.isFinite) &&
        actualValues.every(Number.isFinite);
      if (!stageValuesValid) {
        summary.invalidCount += 1;
        continue;
      }
      const stageComponentResults = [];
      for (let componentIndex = 0; componentIndex < componentCount; componentIndex += 1) {
        const expectedValue = expectedValues[componentIndex];
        const actualValue = actualValues[componentIndex];
        const absoluteError = Math.abs(expectedValue - actualValue);
        stageComponentResults.push({
          componentIndex,
          expectedValue,
          actualValue,
          absoluteError,
          mismatch: absoluteError > stageContract.tolerance
        });
      }
      const stageMismatches = stageComponentResults.filter(
        (component) => component.mismatch
      );
      summary.comparedCount += 1;
      summary.validCount += 1;
      summary.comparedComponentCount += componentCount;
      summary.maxAbsoluteError = Math.max(
        summary.maxAbsoluteError ?? 0,
        ...stageComponentResults.map((component) => component.absoluteError)
      );
      if (stageMismatches.length > 0) {
        summary.mismatchCount += 1;
        summary.componentMismatchCount += stageMismatches.length;
        const comparisonClassification =
          stageContract.key === 'webgpuInclusivePixelBounds'
            ? pixelBoundaryPrecisionClassification({
                expectedRecord: expected,
                actualRecord: actual,
                actualRasterCompanionContract,
                stageMismatches,
                rowIdentityValid:
                  coverage.coverageComplete === true &&
                  srcIndex === inputContract?.appliedRangeStart + localRow &&
                  legacyActual?.row === localRow
              })
            : 'mismatch';
        if (comparisonClassification === 'precision-aligned') {
          summary.precisionAlignedCount += 1;
          summary.precisionAlignedComponentCount += stageMismatches.length;
        } else {
          summary.semanticResidualCount += 1;
          summary.semanticResidualComponentCount += stageMismatches.length;
          for (const component of stageMismatches) {
            if (firstMismatches.length >= POPULATION_SEMANTIC_MAX_FIRST_MISMATCHES) {
              break;
            }
            firstMismatches.push({
              chunkIndex,
              localRow,
              globalResidentRow: srcIndex - PRODUCTION_RESIDENT_RANGE_START,
              srcIndex,
              stage: stageContract.key,
              component: stageContract.components[component.componentIndex],
              expected: component.expectedValue,
              actual: component.actualValue,
              absoluteError: component.absoluteError,
              tolerance: stageContract.tolerance,
              expectedProvenance: POPULATION_SEMANTIC_EXPECTED_PROVENANCE,
              actualProvenance: POPULATION_SEMANTIC_ACTUAL_PROVENANCE,
              expectedStageProvenance: RASTER_STAGE_KEYS.has(stageContract.key)
                ? POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE[stageContract.key]
                : POPULATION_SEMANTIC_EXPECTED_PROVENANCE,
              actualStageProvenance: RASTER_STAGE_KEYS.has(stageContract.key)
                ? POPULATION_RASTER_SEMANTIC_ACTUAL_PROVENANCE
                : POPULATION_SEMANTIC_ACTUAL_PROVENANCE
            });
          }
        }
        const stageRepresentatives = stageLocalRepresentatives[stageContract.key];
        if (
          stageRepresentatives.length <
            POPULATION_SEMANTIC_STAGE_LOCAL_REPRESENTATIVE_LIMIT
        ) {
          const representative =
            buildPopulationSemanticStageLocalMismatchRepresentative({
              chunkIndex,
              localRow,
              srcIndex,
              stage: stageContract.key,
              mismatchComponents: stageMismatches,
              expectedRecord: expected,
              actualRecord: actual,
              comparisonClassification
            });
          if (representative) stageRepresentatives.push(representative);
        }
      }
    }
  }

  for (const stageContract of POPULATION_SEMANTIC_STAGE_CONTRACTS) {
    const summary = summaries[stageContract.key];
    summary.missingInvalidCount = summary.missingCount + summary.invalidCount;
    summary.classification = classifyPopulationSemanticStageEvidence({
      ...summary,
      requiredRecordCount: requestedCount,
      componentCount: stageContract.components.length
    }).classification;
  }
  const stageSummaries = POPULATION_SEMANTIC_STAGE_CONTRACTS.map(
    ({ key }) => summaries[key]
  );
  const stageLocalMismatchSummaries =
    buildPopulationSemanticStageLocalMismatchSummaries({
      stageSummaries,
      representativesByStage: stageLocalRepresentatives
    });
  const anyMismatch = stageSummaries.some(
    (summary) => summary.semanticResidualCount > 0
  );
  const evidenceComplete = stageSummaries.every(
    (summary) => POPULATION_SEMANTIC_COMPLETE_STAGE_CLASSIFICATIONS.includes(
      summary.classification
    )
  );
  const blockedReasons = [];
  if (inputContract?.schemaVersion !== POPULATION_SEMANTIC_COMPARISON_SCHEMA_VERSION) {
    blockedReasons.push('input-contract-schema-drift');
  }
  if (inputContract?.contractName !== POPULATION_SEMANTIC_COMPARISON_CONTRACT_NAME) {
    blockedReasons.push('input-contract-name-drift');
  }
  if (inputContract?.status !== 'ready') {
    blockedReasons.push(...(inputContract?.blockedReasons ?? ['input-contract-blocked']));
  }
  if (!coverage.coverageComplete) blockedReasons.push('coverage-incomplete');
  if (rasterCompanionCoverage.status !== 'ready') {
    blockedReasons.push(...rasterCompanionCoverage.blockedReasons);
  }
  blockedReasons.push(
    ...validatePopulationSemanticStageLocalMismatchSummaries({
      stageLocalMismatchSummaries,
      stageSummaries,
      scope: 'single-chunk',
      rangeStart: inputContract?.appliedRangeStart,
      rangeCount: requestedCount,
      chunkIndex
    })
  );
  for (const summary of stageSummaries) {
    if (summary.classification.startsWith('blocked-')) {
      blockedReasons.push(`${summary.stage}:${summary.classification}`);
    }
  }
  const decision =
    blockedReasons.length > 0
      ? 'blocked'
      : anyMismatch
        ? 'mismatch'
        : 'match';
  return {
    schemaVersion: inputContract?.schemaVersion ?? null,
    contractName: inputContract?.contractName ?? null,
    decision,
    match: decision === 'match',
    reason: blockedReasons[0] ?? (anyMismatch ? 'semantic-mismatch' : null),
    blockedReasons: [...new Set(blockedReasons)],
    identity: inputContract,
    coverage,
    rasterCompanionCoverage,
    rasterCompanionEvidenceLayout: actualRasterCompanionContract ?? null,
    stageSummaries,
    stageLocalMismatchSummaries,
    firstMismatches,
    firstMismatchCount: firstMismatches.length,
    firstMismatchLimit: POPULATION_SEMANTIC_MAX_FIRST_MISMATCHES,
    mismatchScanOrder: 'chunk-local-row-stage-component',
    evidenceComplete,
    expectedProvenance: POPULATION_SEMANTIC_EXPECTED_PROVENANCE,
    actualProvenance: POPULATION_SEMANTIC_ACTUAL_PROVENANCE,
    rasterExpectedProvenance: POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE,
    rasterActualProvenance: POPULATION_RASTER_SEMANTIC_ACTUAL_PROVENANCE,
    precisionClassificationProvenance:
      POPULATION_PIXEL_BOUNDARY_PRECISION_CLASSIFICATION_PROVENANCE,
    actualEvidenceSameProductionDispatch: false,
    productionCalculationDependsOnDiagnosticReadback: false,
    rawRecordArraysIncluded: false,
    gpuResourcesIncluded: false,
    resultSizePopulationIndependent: true
  };
}

export async function runPopulationAlignedSemanticComparisonChunk({
  device,
  raw,
  rangeStart,
  rangeCount,
  buildConfig,
  projectionParams,
  sceneInputIdentity,
  spl4InputIdentity,
  populationContractIdentity,
  cameraIdentity,
  projectionIdentity,
  timeIdentity,
  chunkIndex = 0
} = {}) {
  const inputContract = buildPopulationSemanticComparisonInputContract({
    rangeStart,
    rangeCount,
    sceneInputIdentity,
    spl4InputIdentity,
    populationContractIdentity,
    buildConfig,
    cameraIdentity,
    projectionIdentity,
    timeIdentity
  });
  if (inputContract.status !== 'ready' || !device || !raw) {
    return {
      schemaVersion: inputContract.schemaVersion,
      contractName: inputContract.contractName,
      decision: 'blocked',
      match: false,
      reason: inputContract.reason ?? 'diagnostic-device-or-raw-input-missing',
      blockedReasons: inputContract.blockedReasons.length > 0
        ? inputContract.blockedReasons
        : ['diagnostic-device-or-raw-input-missing'],
      identity: inputContract,
      coverage: buildPopulationSemanticCoverageContract({ inputContract }),
      stageSummaries: [],
      firstMismatches: [],
      firstMismatchCount: 0,
      firstMismatchLimit: POPULATION_SEMANTIC_MAX_FIRST_MISMATCHES,
      rawRecordArraysIncluded: false,
      gpuResourcesIncluded: false,
      resultSizePopulationIndependent: true
    };
  }
  const candidateIndices = buildExplicitPopulationChunkIndices(inputContract);
  const rawXyzOpacity = buildPackedRawXyzOpacity(raw, candidateIndices);
  if (!rawXyzOpacity) {
    return {
      schemaVersion: inputContract.schemaVersion,
      contractName: inputContract.contractName,
      decision: 'blocked',
      match: false,
      reason: 'population-aligned-spl4-xyz-opacity-input-missing',
      blockedReasons: ['population-aligned-spl4-xyz-opacity-input-missing'],
      identity: inputContract,
      coverage: buildPopulationSemanticCoverageContract({
        inputContract,
        candidateIndices,
        actualRecordCount: 0
      }),
      stageSummaries: [],
      firstMismatches: [],
      firstMismatchCount: 0,
      firstMismatchLimit: POPULATION_SEMANTIC_MAX_FIRST_MISMATCHES,
      rawRecordArraysIncluded: false,
      gpuResourcesIncluded: false,
      resultSizePopulationIndependent: true
    };
  }

  const sourceWorksetResourceIdentity =
    buildPopulationSemanticDiagnosticWorksetResourceIdentity(inputContract);
  const projection = readProjectionRows(projectionParams);
  let stateGpuResources = null;
  let tileInputGpuResource = null;
  try {
    const actual = await buildWebGpu4DStatePositionsForCandidates({
      device,
      raw,
      candidateIndices,
      rawXyzOpacity,
      buildConfig,
      projectionParams,
      readbackPolicy: 'diagnostic',
      keepGpuResources: false,
      populationSemanticDiagnostic: true
    });
    const productionState = await buildWebGpu4DStatePositionsForCandidates({
      device,
      raw,
      candidateIndices,
      rawXyzOpacity,
      buildConfig,
      projectionParams,
      readbackPolicy: 'none',
      keepGpuResources: true,
      populationSemanticDiagnostic: false,
      sourceWorksetResourceIdentity
    });
    stateGpuResources = productionState.gpuResources;
    const tileInput = await buildNativeWebGpuProductionTileInput({
      device,
      workset: {
        candidateIndices,
        contract: { resourceIdentity: sourceWorksetResourceIdentity }
      },
      stateGpuResources,
      projectionParams
    });
    tileInputGpuResource = tileInput.gpuResource;
    const rasterCompanion = await observePopulationRasterSemanticCompanion({
      device,
      tileInputResource: tileInputGpuResource,
      expectedSourceWorksetResourceIdentity: sourceWorksetResourceIdentity,
      canvasWidth: projection?.canvasWidth,
      canvasHeight: projection?.canvasHeight,
      tileSize: PRODUCTION_TILE_SIZE
    });
    const result = comparePopulationAlignedSemanticChunkEvidence({
      inputContract,
      candidateIndices,
      actualPackedEvidence: actual.populationSemanticIntermediateReadback,
      actualRasterCompanionEvidence: rasterCompanion.evidence,
      actualRasterCompanionContract: rasterCompanion.contract,
      expectedRecordForRow: (_localRow, srcIndex) =>
        buildPopulationAlignedSemanticExpectedRecord({
          raw,
          srcIndex,
          buildConfig,
          projectionParams
        }),
      chunkIndex
    });
    return {
      ...result,
      diagnosticEvidenceLayout: actual.populationSemanticDiagnosticLayout,
      diagnosticRasterCompanionLayout: rasterCompanion.contract,
      diagnosticGpuResourceOwnership: actual.diagnosticGpuResourceOwnership,
      rasterObserverGpuResourceOwnership:
        'observer-call-scoped-destroyed-before-promise-resolution',
      productionBindingCount: 8,
      productionReadbackPolicyChanged: false,
      step113DiagnosticTailChanged: false,
      nativeTileInputBufferUsageChanged: false
    };
  } finally {
    destroyBuffers([
      stateGpuResources?.statePositionBuffer,
      stateGpuResources?.renderAttributeBuffer,
      stateGpuResources?.footprintPayloadBuffer,
      tileInputGpuResource?.buffer
    ]);
  }
}
