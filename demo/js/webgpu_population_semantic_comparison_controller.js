import {
  PRODUCTION_EVALUATION_INPUT_CONTRACT_VERSION
} from './common_4dgs_production_frame_data_contracts.js';
import {
  POPULATION_RASTER_SEMANTIC_ACTUAL_DEVICE_SCOPE,
  POPULATION_SEMANTIC_CONTROLLER_MAX_RESULT_JSON_BYTES,
  POPULATION_SEMANTIC_MAX_FIRST_MISMATCHES,
  POPULATION_SEMANTIC_STAGE_CONTRACTS,
  PRODUCTION_RESIDENT_RANGE_COUNT,
  PRODUCTION_RESIDENT_RANGE_END,
  PRODUCTION_RESIDENT_RANGE_START
} from './common_4dgs_population_semantic_comparison_contracts.js';
import {
  POPULATION_SEMANTIC_FIXED_CHUNK_COUNT,
  POPULATION_SEMANTIC_ORCHESTRATION_CONTRACT_NAME,
  POPULATION_SEMANTIC_ORCHESTRATION_SCHEMA_VERSION,
  runPopulationAlignedSemanticComparisonResidentRange
} from './webgpu_population_aligned_semantic_comparison_orchestrator.js';

export const POPULATION_SEMANTIC_COMPARISON_CONTROLLER_CONTRACT_VERSION =
  'phase3-population-semantic-comparison-controller-v1';

const MAX_BLOCKED_REASONS = 16;
const MAX_REASON_LENGTH = 192;
const MAX_JSON_DEPTH = 20;
const MAX_JSON_ARRAY_LENGTH = 64;
const MAX_JSON_OBJECT_KEYS = 128;
const MAX_JSON_STRING_LENGTH = 2048;
const RAW_DIMENSION_FIELDS = Object.freeze([
  ['xyz', 'xyzDim'],
  ['rotation', 'rotationDim'],
  ['rotationR', 'rotationRDim'],
  ['scaleXyz', 'scaleXYZDim'],
  ['fDc', 'fdcDim'],
  ['fRest', 'frestDim'],
  ['opacity', 'opacityDim'],
  ['time', 'tDim'],
  ['scaleTime', 'scaleTDim']
]);
const FORBIDDEN_RESULT_KEYS = new Set([
  'adapter',
  'buffer',
  'candidateIndices',
  'device',
  'gpuResources',
  'raw',
  'rawXyzOpacity'
]);

function isPlainObject(value) {
  if (value == null || typeof value !== 'object') return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function nonEmptyString(value) {
  return typeof value === 'string' && value.length > 0;
}

function positiveSafeInteger(value) {
  return Number.isSafeInteger(value) && value > 0;
}

function nonNegativeSafeInteger(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

function sameFloat32Value(left, right) {
  return Number.isFinite(left) &&
    Number.isFinite(right) &&
    Math.fround(left) === Math.fround(right);
}

function boundedReason(value) {
  return String(value ?? 'unknown')
    .replace(/\s+/gu, ' ')
    .slice(0, MAX_REASON_LENGTH);
}

function addBlockedReason(reasons, reason) {
  if (reasons.length >= MAX_BLOCKED_REASONS) return;
  const normalized = boundedReason(reason);
  if (!reasons.includes(normalized)) reasons.push(normalized);
}

function cloneBoundedJsonValue(value, path = 'value', depth = 0, ancestors = new WeakSet()) {
  if (depth > MAX_JSON_DEPTH) throw new TypeError(`${path}-maximum-depth-exceeded`);
  if (value === null || typeof value === 'boolean') return value;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new TypeError(`${path}-finite-number-required`);
    return value;
  }
  if (typeof value === 'string') {
    if (value.length > MAX_JSON_STRING_LENGTH) {
      throw new TypeError(`${path}-string-too-long`);
    }
    return value;
  }
  if (typeof value !== 'object') throw new TypeError(`${path}-unsupported-value`);
  if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) {
    throw new TypeError(`${path}-typed-array-or-buffer-not-allowed`);
  }
  if (ancestors.has(value)) throw new TypeError(`${path}-circular-value`);
  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      if (value.length > MAX_JSON_ARRAY_LENGTH) {
        throw new TypeError(`${path}-array-too-long`);
      }
      return value.map((item, index) =>
        cloneBoundedJsonValue(item, `${path}[${index}]`, depth + 1, ancestors)
      );
    }
    if (!isPlainObject(value)) throw new TypeError(`${path}-plain-object-required`);
    const keys = Object.keys(value);
    if (keys.length > MAX_JSON_OBJECT_KEYS) {
      throw new TypeError(`${path}-too-many-fields`);
    }
    const clone = {};
    for (const key of keys) {
      if (FORBIDDEN_RESULT_KEYS.has(key)) {
        throw new TypeError(`${path}.${key}-forbidden-result-field`);
      }
      clone[key] = cloneBoundedJsonValue(
        value[key],
        `${path}.${key}`,
        depth + 1,
        ancestors
      );
    }
    return clone;
  } finally {
    ancestors.delete(value);
  }
}

function deepFreeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}

function validateProductionEvaluationInputContract(contract) {
  const reasons = [];
  if (!isPlainObject(contract)) {
    return ['production-evaluation-input-contract-missing'];
  }
  if (contract.contractVersion !== PRODUCTION_EVALUATION_INPUT_CONTRACT_VERSION) {
    addBlockedReason(reasons, 'production-evaluation-input-contract-version-unsupported');
  }
  if (contract.status !== 'ready') {
    addBlockedReason(reasons, 'production-evaluation-input-contract-not-ready');
  }
  if (contract.reason !== null) {
    addBlockedReason(reasons, 'production-evaluation-input-contract-reason-not-null');
  }
  if (!Array.isArray(contract.blockedReasons) || contract.blockedReasons.length !== 0) {
    addBlockedReason(reasons, 'production-evaluation-input-blocked-reasons-not-empty');
  }
  for (const field of ['bounded', 'immutable', 'jsonSerializable']) {
    if (contract[field] !== true) {
      addBlockedReason(reasons, `production-evaluation-input-${field}-not-ready`);
    }
  }
  for (const field of [
    'rawObjectIncluded',
    'rawArraysIncluded',
    'candidateIndicesIncluded',
    'projectionTypedArrayIncluded',
    'typedArraysIncluded',
    'gpuResourcesIncluded',
    'populationScaledArraysIncluded'
  ]) {
    if (contract[field] !== false) {
      addBlockedReason(reasons, `production-evaluation-input-${field}-drift`);
    }
  }
  if (contract.provenance !== 'production-frame-exact-local-input-snapshot') {
    addBlockedReason(reasons, 'production-evaluation-input-provenance-drift');
  }
  if (
    contract.publicationMode !==
      'additive-observer-metadata-not-production-control'
  ) addBlockedReason(reasons, 'production-evaluation-input-publication-mode-drift');

  const asset = contract.assetIdentity;
  const assetValues = asset?.spl4AssetIdentity;
  const workset = contract.productionWorksetIdentity;
  const selection = workset?.selection;
  if (
    !nonEmptyString(asset?.sceneResourceIdentity) ||
    !isPlainObject(assetValues) ||
    !/^[0-9a-fA-F]{64}$/u.test(assetValues.sha256 ?? '') ||
    !positiveSafeInteger(assetValues.sizeBytes) ||
    !nonEmptyString(assetValues.sourceKind) ||
    assetValues.format !== 'SPL4-v2' ||
    assetValues.formatVersion !== 2 ||
    !positiveSafeInteger(assetValues.recordCount) ||
    !isPlainObject(assetValues.dimensions) ||
    RAW_DIMENSION_FIELDS.some(([field]) =>
      field === 'fRest'
        ? !nonNegativeSafeInteger(assetValues.dimensions?.[field])
        : !positiveSafeInteger(assetValues.dimensions?.[field])
    )
  ) addBlockedReason(reasons, 'production-evaluation-asset-identity-invalid');
  if (
    !isPlainObject(workset) ||
    !nonEmptyString(workset.resourceIdentity) ||
    workset.sceneResourceIdentity !== asset?.sceneResourceIdentity ||
    workset.sceneRecordCount !== assetValues?.recordCount ||
    workset.residentStart !== PRODUCTION_RESIDENT_RANGE_START ||
    workset.residentRecordCount !== PRODUCTION_RESIDENT_RANGE_COUNT ||
    workset.residentEndExclusive !== PRODUCTION_RESIDENT_RANGE_END ||
    workset.productionResidentSelectionReady !== true ||
    workset.residentSelectionMatches !== true ||
    workset.residentWorksetReady !== true ||
    workset.diagnosticCandidateSourceUsed !== false ||
    workset.diagnosticMaxRecordsUsed !== false ||
    workset.nonResidentRecordsExplicit !== true ||
    workset.silentOmissionAllowed !== false ||
    workset.sourceIndexSpace !== 'spl4-original-source-index' ||
    workset.residentRowSpace !== 'active-resident-workset-local-row' ||
    workset.overflowPolicy !== 'fail-closed-before-compositor-promotion' ||
    workset.candidateOrder !==
      'resident-row-ascending-original-source-index' ||
    workset.sourceIndexMapping !== 'srcIndex=residentStart+residentRow'
  ) addBlockedReason(reasons, 'production-evaluation-fixed-workset-invalid');
  if (
    !isPlainObject(selection) ||
    selection.requestProvided !== true ||
    selection.requestMode !== 'range' ||
    selection.requestedStart !== PRODUCTION_RESIDENT_RANGE_START ||
    selection.requestedRecordCount !== PRODUCTION_RESIDENT_RANGE_COUNT ||
    selection.requestedEndExclusive !== PRODUCTION_RESIDENT_RANGE_END ||
    selection.appliedStart !== PRODUCTION_RESIDENT_RANGE_START ||
    selection.appliedRecordCount !== PRODUCTION_RESIDENT_RANGE_COUNT ||
    selection.appliedEndExclusive !== PRODUCTION_RESIDENT_RANGE_END ||
    selection.sourceIndexSpace !== 'spl4-original-source-index' ||
    selection.residentRowSpace !== 'active-resident-workset-local-row' ||
    selection.selectionPolicy !== workset.selectionPolicy ||
    selection.productionResidentSelectionReady !== true
  ) addBlockedReason(reasons, 'production-evaluation-fixed-selection-invalid');

  const appliedConfig = contract.appliedConfig;
  if (
    !isPlainObject(appliedConfig) ||
    !Number.isFinite(appliedConfig.timestamp) ||
    !Number.isFinite(appliedConfig.scalingModifier) ||
    appliedConfig.scalingModifier <= 0 ||
    !Number.isFinite(appliedConfig.sigmaScale) ||
    appliedConfig.sigmaScale <= 0 ||
    !Number.isFinite(appliedConfig.prefilterVar) ||
    !Number.isFinite(appliedConfig.configuredRenderScale) ||
    appliedConfig.configuredRenderScale <= 0 ||
    appliedConfig.effectiveRenderScale !== appliedConfig.configuredRenderScale ||
    !positiveSafeInteger(appliedConfig.renderWidth) ||
    !positiveSafeInteger(appliedConfig.renderHeight) ||
    !positiveSafeInteger(appliedConfig.canvasWidth) ||
    !positiveSafeInteger(appliedConfig.canvasHeight) ||
    appliedConfig.renderWidth !== Math.max(
      1,
      Math.round(
        appliedConfig.canvasWidth * appliedConfig.effectiveRenderScale
      )
    ) ||
    appliedConfig.renderHeight !== Math.max(
      1,
      Math.round(
        appliedConfig.canvasHeight * appliedConfig.effectiveRenderScale
      )
    )
  ) addBlockedReason(reasons, 'production-evaluation-applied-config-invalid');

  const projection = contract.projection;
  if (
    !isPlainObject(projection) ||
    projection.floatCount !== 44 ||
    !Array.isArray(projection.values) ||
    projection.values.length !== 44 ||
    !projection.values.every(Number.isFinite) ||
    !isPlainObject(projection.summary) ||
    !nonEmptyString(projection.summary.schemaVersion) ||
    !['cuda-aligned', 'threejs'].includes(projection.summary.mode) ||
    !nonEmptyString(projection.summary.projectionContract) ||
    !nonEmptyString(projection.summary.sourcePositionMode) ||
    !nonEmptyString(projection.summary.viewMatrixSource) ||
    !nonEmptyString(projection.summary.projectionMatrixSource) ||
    !isPlainObject(projection.summary.intrinsics) ||
    projection.summary.renderW !== appliedConfig?.renderWidth ||
    projection.summary.renderH !== appliedConfig?.renderHeight ||
    projection.values[0] !== (projection.summary.mode === 'cuda-aligned' ? 1 : 0) ||
    projection.values[1] !== appliedConfig?.renderWidth ||
    projection.values[2] !== appliedConfig?.renderHeight ||
    !sameFloat32Value(projection.summary.sx, projection.values[4]) ||
    !sameFloat32Value(projection.summary.sy, projection.values[5]) ||
    !sameFloat32Value(projection.summary.pixelXSign, projection.values[6]) ||
    !sameFloat32Value(projection.summary.intrinsics?.fx, projection.values[8]) ||
    !sameFloat32Value(projection.summary.intrinsics?.fy, projection.values[9]) ||
    !sameFloat32Value(projection.summary.intrinsics?.cx, projection.values[10]) ||
    !sameFloat32Value(projection.summary.intrinsics?.cy, projection.values[11])
  ) addBlockedReason(reasons, 'production-evaluation-projection-invalid');

  const camera = contract.cameraIdentity;
  const time = contract.timeIdentity;
  const request = contract.requestIdentity;
  const production = contract.productionIdentity;
  const orientation = contract.presentationOrientationIdentity;
  if (
    !isPlainObject(camera) ||
    !nonEmptyString(camera.cameraLabel) ||
    !nonEmptyString(camera.referenceCameraLabel) ||
    !nonNegativeSafeInteger(camera.datasetFrameNumber) ||
    !nonNegativeSafeInteger(camera.datasetViewId) ||
    !nonEmptyString(camera.cameraSource) ||
    !nonEmptyString(camera.datasetViewMatrixMode) ||
    !nonEmptyString(camera.cameraControlContract) ||
    !nonEmptyString(camera.cameraOrientationPolicy)
  ) addBlockedReason(reasons, 'production-evaluation-camera-identity-invalid');
  if (
    !isPlainObject(time) ||
    !Number.isFinite(time.requestedDatasetTime) ||
    !Number.isFinite(time.actualAppliedTimestamp) ||
    time.requestedDatasetTime !== appliedConfig?.timestamp ||
    time.actualAppliedTimestamp !== appliedConfig?.timestamp ||
    time.requestedDatasetTimeMatchesAppliedTimestamp !== true
  ) addBlockedReason(reasons, 'production-evaluation-time-identity-invalid');
  if (
    !isPlainObject(request) ||
    !nonEmptyString(request.schedulerRequestIdentity) ||
    !nonEmptyString(request.schedulerRequestSource) ||
    request.sourceRequestIdentity !== request.schedulerRequestIdentity ||
    !nonNegativeSafeInteger(request.schedulerFrameIndex)
  ) addBlockedReason(reasons, 'production-evaluation-request-identity-invalid');
  if (
    !isPlainObject(production) ||
    !nonNegativeSafeInteger(production.productionGeneration) ||
    production.presentedGeneration !== production.productionGeneration ||
    !isPlainObject(production.productionFrameIdentity) ||
    !isPlainObject(production.presentedFrameIdentity) ||
    production.productionFrameIdentity.generation !==
      production.productionGeneration ||
    production.presentedFrameIdentity.generation !==
      production.productionGeneration ||
    production.productionFrameIdentity.datasetCameraLabel !==
      production.presentedFrameIdentity.datasetCameraLabel ||
    production.productionFrameIdentity.datasetFrameNumber !==
      production.presentedFrameIdentity.datasetFrameNumber ||
    production.productionFrameIdentity.referenceCameraLabel !==
      production.presentedFrameIdentity.referenceCameraLabel ||
    production.productionFrameIdentity.outputWidth !==
      production.presentedFrameIdentity.outputWidth ||
    production.productionFrameIdentity.outputHeight !==
      production.presentedFrameIdentity.outputHeight ||
    production.productionFrameIdentity.datasetCameraLabel !== camera?.cameraLabel ||
    production.productionFrameIdentity.datasetFrameNumber !==
      camera?.datasetFrameNumber ||
    production.productionFrameIdentity.referenceCameraLabel !==
      camera?.referenceCameraLabel ||
    production.productionFrameIdentity.datasetTime !== appliedConfig?.timestamp ||
    production.presentedFrameIdentity.datasetTime !== appliedConfig?.timestamp ||
    production.productionFrameIdentity.outputWidth !== appliedConfig?.canvasWidth ||
    production.productionFrameIdentity.outputHeight !== appliedConfig?.canvasHeight
  ) addBlockedReason(reasons, 'production-evaluation-production-identity-invalid');
  if (
    !isPlainObject(orientation) ||
    !nonEmptyString(orientation.schemaVersion) ||
    orientation.productionTextureOrigin !== 'texture-memory-top-left' ||
    orientation.productionTextureYAxisDirection !== 'down' ||
    orientation.presentationVerticalFlipApplied !== false ||
    orientation.captureVerticalFlipApplied !== false ||
    orientation.orientationMismatchClassification !== 'none'
  ) addBlockedReason(reasons, 'production-evaluation-orientation-identity-invalid');
  return reasons;
}

function validateRawAssetIdentity(raw, contract) {
  const reasons = [];
  const asset = contract?.assetIdentity?.spl4AssetIdentity;
  const rawRecordCount = Number.isSafeInteger(raw?.N)
    ? raw.N
    : Number.isSafeInteger(raw?.count)
      ? raw.count
      : null;
  if (!isPlainObject(raw)) addBlockedReason(reasons, 'parsed-spl4-raw-missing');
  if (raw?.version !== 2) addBlockedReason(reasons, 'parsed-spl4-version-mismatch');
  if (rawRecordCount !== asset?.recordCount) {
    addBlockedReason(reasons, 'parsed-spl4-record-count-mismatch');
  }
  if (
    !nonEmptyString(raw?.assetSha256) ||
    !nonEmptyString(asset?.sha256) ||
    raw.assetSha256.toLowerCase() !== asset.sha256.toLowerCase()
  ) addBlockedReason(reasons, 'parsed-spl4-sha256-mismatch');
  if (raw?.assetSizeBytes !== asset?.sizeBytes) {
    addBlockedReason(reasons, 'parsed-spl4-size-mismatch');
  }
  if (raw?.assetSourceKind !== asset?.sourceKind) {
    addBlockedReason(reasons, 'parsed-spl4-source-kind-mismatch');
  }
  for (const [contractField, rawField] of RAW_DIMENSION_FIELDS) {
    if (raw?.[rawField] !== asset?.dimensions?.[contractField]) {
      addBlockedReason(reasons, `parsed-spl4-${contractField}-dimension-mismatch`);
    }
  }
  if (rawRecordCount !== contract?.productionWorksetIdentity?.sceneRecordCount) {
    addBlockedReason(reasons, 'parsed-spl4-workset-scene-count-mismatch');
  }
  return reasons;
}

function buildOrchestratorInput(raw, contract) {
  const projectionValues = [...contract.projection.values];
  return {
    raw,
    buildConfig: {
      timestamp: contract.appliedConfig.timestamp,
      scalingModifier: contract.appliedConfig.scalingModifier,
      sigmaScale: contract.appliedConfig.sigmaScale,
      prefilterVar: contract.appliedConfig.prefilterVar
    },
    projectionParams: new Float32Array(projectionValues),
    sceneInputIdentity: {
      sceneResourceIdentity: contract.assetIdentity.sceneResourceIdentity
    },
    spl4InputIdentity: cloneBoundedJsonValue(
      contract.assetIdentity.spl4AssetIdentity,
      'spl4-input-identity'
    ),
    populationContractIdentity: cloneBoundedJsonValue(
      contract.productionWorksetIdentity,
      'population-contract-identity'
    ),
    cameraIdentity: cloneBoundedJsonValue(
      contract.cameraIdentity,
      'camera-identity'
    ),
    projectionIdentity: {
      floatCount: contract.projection.floatCount,
      values: projectionValues,
      summary: cloneBoundedJsonValue(
        contract.projection.summary,
        'projection-summary'
      )
    },
    timeIdentity: {
      time: cloneBoundedJsonValue(contract.timeIdentity, 'time-identity'),
      request: cloneBoundedJsonValue(contract.requestIdentity, 'request-identity'),
      production: cloneBoundedJsonValue(
        contract.productionIdentity,
        'production-identity'
      ),
      orientation: cloneBoundedJsonValue(
        contract.presentationOrientationIdentity,
        'orientation-identity'
      )
    }
  };
}

function validateOrchestrationResult(result) {
  const reasons = [];
  if (!isPlainObject(result)) return ['orchestration-result-missing'];
  if (result.schemaVersion !== POPULATION_SEMANTIC_ORCHESTRATION_SCHEMA_VERSION) {
    addBlockedReason(reasons, 'orchestration-schema-version-drift');
  }
  if (result.contractName !== POPULATION_SEMANTIC_ORCHESTRATION_CONTRACT_NAME) {
    addBlockedReason(reasons, 'orchestration-contract-name-drift');
  }
  if (!['match', 'mismatch', 'blocked'].includes(result.decision)) {
    addBlockedReason(reasons, 'orchestration-decision-invalid');
  }
  if (!Array.isArray(result.blockedReasons)) {
    addBlockedReason(reasons, 'orchestration-blocked-reasons-invalid');
  }
  for (const [field, expected] of [
    ['singleChunkResultsIncluded', false],
    ['rawRecordArraysIncluded', false],
    ['typedArraysIncluded', false],
    ['gpuResourcesIncluded', false],
    ['deviceIncluded', false],
    ['resultSizePopulationIndependent', true],
    ['actualEvidenceSameProductionDispatch', false],
    ['productionCalculationDependsOnDiagnosticReadback', false],
    ['expectedGenerationDependsOnActual', false]
  ]) {
    if (result[field] !== expected) {
      addBlockedReason(reasons, `orchestration-${field}-drift`);
    }
  }
  if (
    result.actualGpuDeviceScope !==
      POPULATION_RASTER_SEMANTIC_ACTUAL_DEVICE_SCOPE
  ) addBlockedReason(reasons, 'orchestration-actual-gpu-device-scope-drift');
  if (result.diagnosticDeviceOwnership !== 'caller-owned-reused-not-destroyed') {
    addBlockedReason(reasons, 'orchestration-diagnostic-device-ownership-drift');
  }
  if (result.decision === 'match' || result.decision === 'mismatch') {
    if (
      result.requestedChunkCount !== POPULATION_SEMANTIC_FIXED_CHUNK_COUNT ||
      result.completedChunkCount !== POPULATION_SEMANTIC_FIXED_CHUNK_COUNT ||
      result.coverage?.requestedRangeStart !== PRODUCTION_RESIDENT_RANGE_START ||
      result.coverage?.requestedRangeCount !== PRODUCTION_RESIDENT_RANGE_COUNT ||
      result.coverage?.requestedRangeEnd !== PRODUCTION_RESIDENT_RANGE_END ||
      result.coverage?.processedRecordCount !== PRODUCTION_RESIDENT_RANGE_COUNT ||
      result.coverage?.coverageComplete !== true ||
      result.coverage?.firstSrcIndex !== PRODUCTION_RESIDENT_RANGE_START ||
      result.coverage?.lastSrcIndex !== PRODUCTION_RESIDENT_RANGE_END - 1 ||
      result.evidenceComplete !== true
    ) addBlockedReason(reasons, 'orchestration-complete-coverage-invalid');
    if (
      !Array.isArray(result.stageSummaries) ||
      result.stageSummaries.length !== POPULATION_SEMANTIC_STAGE_CONTRACTS.length ||
      result.stageSummaries.some(
        (stage, index) => stage?.stage !== POPULATION_SEMANTIC_STAGE_CONTRACTS[index].key
      )
    ) addBlockedReason(reasons, 'orchestration-stage-contract-invalid');
    if (
      !Array.isArray(result.firstMismatches) ||
      result.firstMismatches.length > POPULATION_SEMANTIC_MAX_FIRST_MISMATCHES ||
      result.firstMismatchCount !== result.firstMismatches.length
    ) addBlockedReason(reasons, 'orchestration-first-mismatch-contract-invalid');
    if (
      result.rasterCompanionSummary?.validatedChunkCount !==
        POPULATION_SEMANTIC_FIXED_CHUNK_COUNT ||
      result.rasterCompanionSummary?.allChunksReady !== true ||
      result.rasterCompanionSummary?.allChunksRowAlignmentVerified !== true ||
      result.rasterCompanionSummary?.nativeTileInputBufferUsageChanged !== false
    ) addBlockedReason(reasons, 'orchestration-raster-companion-incomplete');
  }
  return reasons;
}

function buildControllerResult({
  productionEvaluationInputContract,
  acquisition,
  cleanup,
  orchestrationResult,
  decision,
  blockedReasons
}) {
  const result = {
    contractVersion: POPULATION_SEMANTIC_COMPARISON_CONTROLLER_CONTRACT_VERSION,
    invocationMode: 'explicit-one-shot-debug-api-request',
    explicitRequest: true,
    automaticExecution: false,
    productionEvaluationInputContractVersion:
      productionEvaluationInputContract?.contractVersion ?? null,
    inputSource:
      'completed-production-frame-productionEvaluationInputContract',
    productionDeviceAcceptedAsInput: false,
    productionDeviceUsed: false,
    freshDiagnosticDeviceAcquisition: acquisition,
    diagnosticDeviceCleanup: cleanup,
    decision,
    match: decision === 'match',
    reason:
      blockedReasons[0] ?? (decision === 'mismatch' ? 'semantic-mismatch' : null),
    blockedReasons,
    orchestrationResult,
    rawObjectIncluded: false,
    rawArraysIncluded: false,
    typedArraysIncluded: false,
    gpuResourcesIncluded: false,
    adapterIncluded: false,
    deviceIncluded: false,
    resultSizePopulationIndependent: true,
    jsonSerializable: true,
    bounded: true
  };
  const serialized = JSON.stringify(result);
  if (
    serialized.length <= POPULATION_SEMANTIC_CONTROLLER_MAX_RESULT_JSON_BYTES
  ) return deepFreeze(result);
  return deepFreeze({
    ...result,
    decision: 'blocked',
    match: false,
    reason: 'controller-result-size-limit-exceeded',
    blockedReasons: ['controller-result-size-limit-exceeded'],
    orchestrationResult: null
  });
}

function defaultRequestAdapter() {
  if (
    typeof navigator === 'undefined' ||
    !navigator.gpu ||
    typeof navigator.gpu.requestAdapter !== 'function'
  ) return Promise.resolve(null);
  return navigator.gpu.requestAdapter();
}

export function createPopulationSemanticComparisonController({
  requestAdapter = defaultRequestAdapter,
  runOrchestrator = runPopulationAlignedSemanticComparisonResidentRange
} = {}) {
  if (typeof requestAdapter !== 'function') {
    throw new TypeError('population-semantic-controller-request-adapter-invalid');
  }
  if (typeof runOrchestrator !== 'function') {
    throw new TypeError('population-semantic-controller-orchestrator-invalid');
  }
  let running = false;
  let lastResult = null;

  async function runPopulationAlignedSemanticComparison({
    raw,
    productionEvaluationInputContract
  } = {}) {
    if (running) {
      return buildControllerResult({
        productionEvaluationInputContract,
        acquisition: {
          status: 'not-requested-already-running',
          adapterRequestAttempted: false,
          adapterAcquired: false,
          deviceRequestAttempted: false,
          deviceAcquired: false,
          freshDiagnosticDevice: false
        },
        cleanup: {
          status: 'not-required',
          required: false,
          attempted: false,
          completed: false
        },
        orchestrationResult: null,
        decision: 'blocked',
        blockedReasons: ['already-running']
      });
    }

    running = true;
    const blockedReasons = [];
    const acquisition = {
      status: 'not-requested',
      adapterRequestAttempted: false,
      adapterAcquired: false,
      deviceRequestAttempted: false,
      deviceAcquired: false,
      freshDiagnosticDevice: false
    };
    const cleanup = {
      status: 'not-required',
      required: false,
      attempted: false,
      completed: false
    };
    let adapter = null;
    let device = null;
    let orchestrationResult = null;
    let decision = 'blocked';

    try {
      for (const reason of validateProductionEvaluationInputContract(
        productionEvaluationInputContract
      )) addBlockedReason(blockedReasons, reason);
      if (blockedReasons.length === 0) {
        for (const reason of validateRawAssetIdentity(
          raw,
          productionEvaluationInputContract
        )) addBlockedReason(blockedReasons, reason);
      }

      let orchestratorInput = null;
      if (blockedReasons.length === 0) {
        try {
          orchestratorInput = buildOrchestratorInput(
            raw,
            productionEvaluationInputContract
          );
        } catch (error) {
          addBlockedReason(
            blockedReasons,
            `orchestrator-input-reconstruction-failed:${boundedReason(error?.message ?? error)}`
          );
        }
      }

      if (blockedReasons.length === 0) {
        acquisition.adapterRequestAttempted = true;
        acquisition.status = 'adapter-requested';
        try {
          adapter = await requestAdapter();
        } catch (error) {
          addBlockedReason(
            blockedReasons,
            `diagnostic-adapter-request-failed:${boundedReason(error?.message ?? error)}`
          );
          acquisition.status = 'adapter-request-failed';
        }
      }
      if (blockedReasons.length === 0) {
        if (!adapter || typeof adapter.requestDevice !== 'function') {
          addBlockedReason(blockedReasons, 'diagnostic-adapter-unavailable');
          acquisition.status = 'adapter-unavailable';
        } else {
          acquisition.adapterAcquired = true;
          acquisition.deviceRequestAttempted = true;
          acquisition.status = 'device-requested';
          try {
            device = await adapter.requestDevice();
          } catch (error) {
            addBlockedReason(
              blockedReasons,
              `diagnostic-device-request-failed:${boundedReason(error?.message ?? error)}`
            );
            acquisition.status = 'device-request-failed';
          }
        }
      }
      if (blockedReasons.length === 0) {
        if (!device) {
          addBlockedReason(blockedReasons, 'diagnostic-device-unavailable');
          acquisition.status = 'device-unavailable';
        } else {
          acquisition.deviceAcquired = true;
          acquisition.freshDiagnosticDevice = true;
          acquisition.status = 'ready';
          cleanup.required = true;
        }
      }

      if (blockedReasons.length === 0) {
        try {
          const rawOrchestrationResult = await runOrchestrator({
            ...orchestratorInput,
            device
          });
          const orchestrationReasons = validateOrchestrationResult(
            rawOrchestrationResult
          );
          try {
            orchestrationResult = cloneBoundedJsonValue(
              rawOrchestrationResult,
              'orchestration-result'
            );
          } catch (error) {
            addBlockedReason(
              blockedReasons,
              `orchestration-result-not-bounded:${boundedReason(error?.message ?? error)}`
            );
          }
          for (const reason of orchestrationReasons) {
            addBlockedReason(blockedReasons, reason);
          }
          if (blockedReasons.length === 0) {
            if (rawOrchestrationResult.decision === 'blocked') {
              addBlockedReason(
                blockedReasons,
                `orchestration-blocked:${boundedReason(rawOrchestrationResult.reason)}`
              );
            } else {
              decision = rawOrchestrationResult.decision;
            }
          }
        } catch (error) {
          addBlockedReason(
            blockedReasons,
            `orchestration-exception:${boundedReason(error?.message ?? error)}`
          );
        }
      }
    } finally {
      adapter = null;
      if (device) {
        cleanup.attempted = true;
        cleanup.status = 'destroying';
        try {
          if (typeof device.destroy !== 'function') {
            throw new TypeError('diagnostic-device-destroy-unavailable');
          }
          device.destroy();
          cleanup.completed = true;
          cleanup.status = 'destroyed';
        } catch (error) {
          cleanup.status = 'destroy-failed';
          addBlockedReason(
            blockedReasons,
            `diagnostic-device-destroy-failed:${boundedReason(error?.message ?? error)}`
          );
        }
        device = null;
      }
      running = false;
    }

    if (blockedReasons.length > 0) decision = 'blocked';
    lastResult = buildControllerResult({
      productionEvaluationInputContract,
      acquisition,
      cleanup,
      orchestrationResult,
      decision,
      blockedReasons
    });
    return lastResult;
  }

  return Object.freeze({
    runPopulationAlignedSemanticComparison,
    getLastPopulationAlignedSemanticComparisonResult: () => lastResult
  });
}
