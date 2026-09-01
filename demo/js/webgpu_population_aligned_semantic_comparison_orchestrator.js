import {
  POPULATION_RASTER_SEMANTIC_ACTUAL_PROVENANCE,
  POPULATION_RASTER_SEMANTIC_ACTUAL_DEVICE_SCOPE,
  POPULATION_RASTER_SEMANTIC_COMPANION_BYTES_PER_RECORD,
  POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE,
  POPULATION_RASTER_SEMANTIC_COMPANION_LAYOUT_SCHEMA_VERSION,
  POPULATION_RASTER_SEMANTIC_COMPANION_VEC4_STRIDE,
  POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE,
  POPULATION_SEMANTIC_ACTUAL_PROVENANCE,
  POPULATION_SEMANTIC_COMPARISON_CONTRACT_NAME,
  POPULATION_SEMANTIC_COMPARISON_SCHEMA_VERSION,
  POPULATION_SEMANTIC_COMPLETE_STAGE_CLASSIFICATIONS,
  POPULATION_SEMANTIC_EXPECTED_PROVENANCE,
  POPULATION_SEMANTIC_LOGICAL_COMBINED_BYTES_PER_RECORD,
  POPULATION_SEMANTIC_LOGICAL_COMBINED_FLOAT_STRIDE,
  POPULATION_SEMANTIC_LOGICAL_COMBINED_VEC4_STRIDE,
  POPULATION_SEMANTIC_MAX_CHUNK_RECORDS,
  POPULATION_SEMANTIC_MAX_FIRST_MISMATCHES,
  POPULATION_PIXEL_BOUNDARY_PRECISION_CLASSIFICATION_PROVENANCE,
  POPULATION_SEMANTIC_STAGE_CONTRACTS,
  PRODUCTION_RESIDENT_RANGE_COUNT,
  PRODUCTION_RESIDENT_RANGE_END,
  PRODUCTION_RESIDENT_RANGE_START,
  buildPopulationRasterSemanticCompanionLayoutContract,
  buildPopulationSemanticComparisonInputContract,
  buildPopulationSemanticDiagnosticWorksetResourceIdentity,
  buildPopulationSemanticStageLocalMismatchSummaries,
  classifyPopulationSemanticStageEvidence,
  populationSemanticStageLocalRepresentativeLimit,
  validatePopulationSemanticStageLocalMismatchSummaries
} from './common_4dgs_population_semantic_comparison_contracts.js';
import {
  runPopulationAlignedSemanticComparisonChunk
} from './webgpu_population_aligned_semantic_comparison.js';

export const POPULATION_SEMANTIC_ORCHESTRATION_SCHEMA_VERSION =
  'phase3-population-aligned-semantic-comparison-orchestration-v5';
export const POPULATION_SEMANTIC_ORCHESTRATION_CONTRACT_NAME =
  'full-production-resident-range-eight-chunk-semantic-comparison';
export const POPULATION_SEMANTIC_FIXED_CHUNK_COUNT =
  PRODUCTION_RESIDENT_RANGE_COUNT / POPULATION_SEMANTIC_MAX_CHUNK_RECORDS;

const CHUNK_RESOURCE_OWNERSHIP =
  'evaluator-call-scoped-destroyed-before-promise-resolution';
const RASTER_OBSERVER_RESOURCE_OWNERSHIP =
  'observer-call-scoped-destroyed-before-promise-resolution';
const RASTER_COMPANION_ROW_ALIGNMENT =
  'local-row-matches-explicit-candidate-index-order';
const RASTER_STAGE_KEYS = new Set(
  Object.keys(POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE)
);
const RETAINED_RESULT_FIELD_NAMES = new Set([
  'actualPackedEvidence',
  'actualRasterCompanionEvidence',
  'candidateIndices',
  'device',
  'footprintPayload',
  'gpuResources',
  'populationSemanticIntermediateReadback',
  'raw',
  'renderAttributes',
  'statePositions',
  'tileInputGpuResource'
]);

function nonNegativeSafeInteger(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

function boundedReason(value) {
  return String(value ?? 'unknown')
    .replace(/\s+/g, ' ')
    .slice(0, 160);
}

function canonicalize(value) {
  if (value == null || typeof value !== 'object') return value;
  if (Array.isArray(value)) return value.map(canonicalize);
  return Object.fromEntries(
    Object.keys(value)
      .sort()
      .map((key) => [key, canonicalize(value[key])])
  );
}

function semanticallyEqual(left, right) {
  return JSON.stringify(canonicalize(left)) === JSON.stringify(canonicalize(right));
}

function retainedRuntimePayloadReason(value, path = 'chunk-result', ancestors = new WeakSet()) {
  if (value == null || typeof value !== 'object') return null;
  if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) {
    return `${path}-typed-array-or-buffer-retained`;
  }
  if (
    typeof SharedArrayBuffer !== 'undefined' &&
    value instanceof SharedArrayBuffer
  ) return `${path}-shared-buffer-retained`;
  if (ancestors.has(value)) return `${path}-circular-reference-retained`;
  const prototype = Object.getPrototypeOf(value);
  if (
    !Array.isArray(value) &&
    prototype !== Object.prototype &&
    prototype !== null
  ) return `${path}-runtime-resource-retained`;
  ancestors.add(value);
  try {
    for (const key of Object.keys(value)) {
      if (RETAINED_RESULT_FIELD_NAMES.has(key)) {
        return `${path}-${key}-retained`;
      }
      if (key === 'stageLocalMismatchSummaries') continue;
      if (
        Array.isArray(value[key]) &&
        /(?:raw|evidence|record|sample|indices|position|attribute|payload)/iu.test(key)
      ) return `${path}-${key}-raw-array-retained`;
      if (
        value[key] != null &&
        typeof value[key] === 'object' &&
        /(?:device|gpu|buffer|resource)/iu.test(key)
      ) return `${path}-${key}-runtime-resource-retained`;
      const nestedReason = retainedRuntimePayloadReason(
        value[key],
        `${path}-${key}`,
        ancestors
      );
      if (nestedReason) return nestedReason;
    }
  } finally {
    ancestors.delete(value);
  }
  return null;
}

function validateRasterProvenance(result) {
  const reasons = [];
  if (
    !semanticallyEqual(
      result.rasterExpectedProvenance,
      POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE
    )
  ) reasons.push('chunk-raster-expected-provenance-drift');
  if (result.rasterActualProvenance !== POPULATION_RASTER_SEMANTIC_ACTUAL_PROVENANCE) {
    reasons.push('chunk-raster-actual-provenance-drift');
  }
  if (result.actualGpuDeviceScope !== POPULATION_RASTER_SEMANTIC_ACTUAL_DEVICE_SCOPE) {
    reasons.push('chunk-actual-gpu-device-scope-drift');
  }
  if (result.expectedGenerationDependsOnActual !== false) {
    reasons.push('chunk-expected-actual-dependency-drift');
  }
  return reasons;
}

function validateRasterCompanionCoverage(coverage, plan) {
  const reasons = [];
  const expectedFloatCount =
    plan.rangeCount * POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE;
  if (!coverage || typeof coverage !== 'object' || Array.isArray(coverage)) {
    return ['raster-companion-coverage-missing'];
  }
  if (coverage.status !== 'ready') reasons.push('raster-companion-coverage-not-ready');
  if (coverage.reason !== null) reasons.push('raster-companion-coverage-reason-not-null');
  if (!Array.isArray(coverage.blockedReasons) || coverage.blockedReasons.length !== 0) {
    reasons.push('raster-companion-coverage-blocked-reasons-not-empty');
  }
  if (coverage.requestedRecordCount !== plan.rangeCount) {
    reasons.push('raster-companion-requested-record-count-drift');
  }
  if (coverage.actualRecordCount !== plan.rangeCount) {
    reasons.push('raster-companion-actual-record-count-drift');
  }
  if (coverage.expectedFloatCount !== expectedFloatCount) {
    reasons.push('raster-companion-expected-float-count-drift');
  }
  if (coverage.actualFloatCount !== expectedFloatCount) {
    reasons.push('raster-companion-actual-float-count-drift');
  }
  if (coverage.evidenceLengthExact !== true) {
    reasons.push('raster-companion-evidence-length-not-exact');
  }
  if (coverage.rowAlignmentVerified !== true) {
    reasons.push('raster-companion-row-alignment-not-verified');
  }
  return reasons;
}

function buildCompanionInvariantSummary(layout) {
  return {
    schemaVersion: layout.schemaVersion,
    contractName: layout.contractName,
    recordCountPerChunk: layout.recordCount,
    rowStrideVec4: layout.rowStrideVec4,
    rowStrideFloats: layout.rowStrideFloats,
    rowStrideBytes: layout.rowStrideBytes,
    logicalCombinedRowStrideVec4: layout.logicalCombinedRowStrideVec4,
    logicalCombinedRowStrideFloats: layout.logicalCombinedRowStrideFloats,
    logicalCombinedRowStrideBytes: layout.logicalCombinedRowStrideBytes,
    rowAlignment: layout.rowAlignment,
    canvasWidth: layout.canvasWidth,
    canvasHeight: layout.canvasHeight,
    tileSize: layout.tileSize,
    tileCols: layout.tileCols,
    tileRows: layout.tileRows,
    observerOutputUsage: layout.observerOutputUsage,
    observerStagingUsage: layout.observerStagingUsage,
    actualGpuDeviceScope: layout.actualGpuDeviceScope,
    expectedGenerationDependsOnActual: layout.expectedGenerationDependsOnActual,
    diagnosticOnly: layout.diagnosticOnly
  };
}

function validateRasterCompanionLayouts(result, plan, expectedIdentity) {
  const reasons = [];
  const layout = result.rasterCompanionEvidenceLayout;
  const diagnosticLayout = result.diagnosticRasterCompanionLayout;
  if (!layout || typeof layout !== 'object' || Array.isArray(layout)) {
    reasons.push('raster-companion-layout-missing');
  }
  if (
    !diagnosticLayout ||
    typeof diagnosticLayout !== 'object' ||
    Array.isArray(diagnosticLayout)
  ) reasons.push('diagnostic-raster-companion-layout-missing');
  if (reasons.length > 0) return { reasons, summary: null };
  if (!semanticallyEqual(layout, diagnosticLayout)) {
    reasons.push('raster-companion-layout-publication-drift');
  }
  const expectedWorksetResourceIdentity =
    buildPopulationSemanticDiagnosticWorksetResourceIdentity(expectedIdentity);
  if (
    layout.sourceWorksetResourceIdentity !== expectedWorksetResourceIdentity ||
    typeof expectedWorksetResourceIdentity !== 'string'
  ) reasons.push('raster-companion-workset-identity-drift');
  for (const field of [
    'sourceStateResourceIdentity',
    'sourceTileInputResourceIdentity'
  ]) {
    if (typeof layout[field] !== 'string' || layout[field].length === 0) {
      reasons.push(`raster-companion-${field}-missing`);
    }
  }
  const dimensionFields = [
    'canvasWidth',
    'canvasHeight',
    'tileSize',
    'tileCols',
    'tileRows'
  ];
  for (const field of dimensionFields) {
    if (!Number.isSafeInteger(layout[field]) || layout[field] <= 0) {
      reasons.push(`raster-companion-${field}-invalid`);
    }
  }
  if (
    Number.isSafeInteger(layout.canvasWidth) &&
    Number.isSafeInteger(layout.tileSize) &&
    layout.tileSize > 0 &&
    layout.tileCols !== Math.ceil(layout.canvasWidth / layout.tileSize)
  ) reasons.push('raster-companion-tile-cols-drift');
  if (
    Number.isSafeInteger(layout.canvasHeight) &&
    Number.isSafeInteger(layout.tileSize) &&
    layout.tileSize > 0 &&
    layout.tileRows !== Math.ceil(layout.canvasHeight / layout.tileSize)
  ) reasons.push('raster-companion-tile-rows-drift');

  const canonicalLayout = buildPopulationRasterSemanticCompanionLayoutContract({
    recordCount: plan.rangeCount,
    evidenceFloatCount:
      plan.rangeCount * POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE,
    sourceWorksetResourceIdentity: expectedWorksetResourceIdentity,
    sourceStateResourceIdentity: layout.sourceStateResourceIdentity,
    sourceTileInputResourceIdentity: layout.sourceTileInputResourceIdentity,
    canvasWidth: layout.canvasWidth,
    canvasHeight: layout.canvasHeight,
    tileSize: layout.tileSize,
    tileCols: layout.tileCols,
    tileRows: layout.tileRows,
    observerDispatchSubmitted: true,
    observerReadbackCompleted: true,
    observerOwnedBuffersDestroyed: true
  });
  if (!semanticallyEqual(layout, canonicalLayout)) {
    reasons.push('raster-companion-layout-contract-drift');
  }
  if (layout.schemaVersion !== POPULATION_RASTER_SEMANTIC_COMPANION_LAYOUT_SCHEMA_VERSION) {
    reasons.push('raster-companion-layout-schema-drift');
  }
  if (layout.status !== 'ready') reasons.push('raster-companion-layout-not-ready');
  if (layout.recordCount !== plan.rangeCount) {
    reasons.push('raster-companion-layout-record-count-drift');
  }
  if (
    layout.rowStrideVec4 !== POPULATION_RASTER_SEMANTIC_COMPANION_VEC4_STRIDE ||
    layout.rowStrideFloats !== POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE ||
    layout.rowStrideBytes !== POPULATION_RASTER_SEMANTIC_COMPANION_BYTES_PER_RECORD
  ) reasons.push('raster-companion-row-stride-drift');
  if (
    layout.logicalCombinedRowStrideVec4 !==
      POPULATION_SEMANTIC_LOGICAL_COMBINED_VEC4_STRIDE ||
    layout.logicalCombinedRowStrideFloats !==
      POPULATION_SEMANTIC_LOGICAL_COMBINED_FLOAT_STRIDE ||
    layout.logicalCombinedRowStrideBytes !==
      POPULATION_SEMANTIC_LOGICAL_COMBINED_BYTES_PER_RECORD
  ) reasons.push('raster-companion-logical-combined-stride-drift');
  const expectedFloatCount =
    plan.rangeCount * POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE;
  if (
    layout.evidenceFloatCount !== expectedFloatCount ||
    layout.evidenceByteSize !== expectedFloatCount * Float32Array.BYTES_PER_ELEMENT
  ) reasons.push('raster-companion-layout-evidence-size-drift');
  if (layout.rowAlignment !== RASTER_COMPANION_ROW_ALIGNMENT) {
    reasons.push('raster-companion-layout-row-alignment-drift');
  }
  if (layout.observerDispatchSubmitted !== true) {
    reasons.push('raster-companion-observer-dispatch-incomplete');
  }
  if (layout.observerReadbackCompleted !== true) {
    reasons.push('raster-companion-observer-readback-incomplete');
  }
  if (layout.observerOwnedBuffersDestroyed !== true) {
    reasons.push('raster-companion-observer-buffer-destruction-incomplete');
  }
  if (layout.nativeTileInputBufferUsageChanged !== false) {
    reasons.push('raster-companion-native-tile-input-usage-drift');
  }
  if (layout.observerOutputUsage !== 'storage-copy-src') {
    reasons.push('raster-companion-observer-output-usage-drift');
  }
  if (layout.observerStagingUsage !== 'copy-dst-map-read') {
    reasons.push('raster-companion-observer-staging-usage-drift');
  }
  if (layout.actualGpuDeviceScope !== POPULATION_RASTER_SEMANTIC_ACTUAL_DEVICE_SCOPE) {
    reasons.push('raster-companion-actual-gpu-device-scope-drift');
  }
  if (layout.expectedGenerationDependsOnActual !== false) {
    reasons.push('raster-companion-expected-actual-dependency-drift');
  }
  if (layout.diagnosticOnly !== true) {
    reasons.push('raster-companion-diagnostic-only-drift');
  }
  return {
    reasons: [...new Set(reasons)],
    summary: buildCompanionInvariantSummary(layout)
  };
}

function cloneSnapshotValue(value, path, ancestors = new WeakSet()) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new Error(`${path}-number-not-finite`);
    return value;
  }
  if (typeof value !== 'object') throw new Error(`${path}-type-unsupported`);
  if (ancestors.has(value)) throw new Error(`${path}-circular-reference`);
  if (Object.getOwnPropertySymbols(value).length > 0) {
    throw new Error(`${path}-symbol-key-unsupported`);
  }
  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      const result = [];
      for (let index = 0; index < value.length; index += 1) {
        if (!Object.hasOwn(value, index)) {
          throw new Error(`${path}-${index}-sparse-array-unsupported`);
        }
        result.push(cloneSnapshotValue(value[index], `${path}-${index}`, ancestors));
      }
      return result;
    }
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new Error(`${path}-object-prototype-unsupported`);
    }
    return Object.fromEntries(
      Object.keys(value).map((key) => [
        key,
        cloneSnapshotValue(value[key], `${path}-${key}`, ancestors)
      ])
    );
  } finally {
    ancestors.delete(value);
  }
}

function snapshotBuildConfig(buildConfig) {
  const timestamp = Number(buildConfig?.timestamp);
  const scalingModifier = buildConfig?.scalingModifier == null
    ? 1
    : Number(buildConfig.scalingModifier);
  const sigmaScale = buildConfig?.sigmaScale == null
    ? 1
    : Number(buildConfig.sigmaScale);
  const prefilterVar = buildConfig?.prefilterVar == null
    ? -1
    : Number(buildConfig.prefilterVar);
  if (!Number.isFinite(timestamp)) throw new Error('buildConfig-timestamp-invalid');
  if (!Number.isFinite(scalingModifier)) {
    throw new Error('buildConfig-scalingModifier-invalid');
  }
  if (!Number.isFinite(sigmaScale)) throw new Error('buildConfig-sigmaScale-invalid');
  if (!Number.isFinite(prefilterVar)) throw new Error('buildConfig-prefilterVar-invalid');
  return { timestamp, scalingModifier, sigmaScale, prefilterVar };
}

function snapshotProjectionParams(projectionParams) {
  if (!(projectionParams instanceof Float32Array) || projectionParams.length < 24) {
    throw new Error('projectionParams-missing-or-invalid');
  }
  const snapshot = new Float32Array(projectionParams);
  if (!Array.from(snapshot).every(Number.isFinite)) {
    throw new Error('projectionParams-nonfinite');
  }
  return snapshot;
}

function buildAuthoritativeExecutionSnapshot(input) {
  try {
    return {
      snapshot: {
        buildConfig: snapshotBuildConfig(input.buildConfig),
        projectionParams: snapshotProjectionParams(input.projectionParams),
        sceneInputIdentity:
          cloneSnapshotValue(input.sceneInputIdentity, 'sceneInputIdentity'),
        spl4InputIdentity:
          cloneSnapshotValue(input.spl4InputIdentity, 'spl4InputIdentity'),
        populationContractIdentity:
          cloneSnapshotValue(
            input.populationContractIdentity,
            'populationContractIdentity'
          ),
        cameraIdentity:
          cloneSnapshotValue(input.cameraIdentity, 'cameraIdentity'),
        projectionIdentity:
          cloneSnapshotValue(input.projectionIdentity, 'projectionIdentity'),
        timeIdentity:
          cloneSnapshotValue(input.timeIdentity, 'timeIdentity')
      },
      blockedReasons: []
    };
  } catch (error) {
    return {
      snapshot: null,
      blockedReasons: [
        `execution-snapshot-failed:${boundedReason(error?.message ?? error)}`
      ]
    };
  }
}

function buildChunkExecutionInput(snapshot, device, raw, plan) {
  return {
    device,
    raw,
    rangeStart: plan.rangeStart,
    rangeCount: plan.rangeCount,
    buildConfig: { ...snapshot.buildConfig },
    projectionParams: new Float32Array(snapshot.projectionParams),
    sceneInputIdentity:
      cloneSnapshotValue(snapshot.sceneInputIdentity, 'sceneInputIdentity'),
    spl4InputIdentity:
      cloneSnapshotValue(snapshot.spl4InputIdentity, 'spl4InputIdentity'),
    populationContractIdentity:
      cloneSnapshotValue(
        snapshot.populationContractIdentity,
        'populationContractIdentity'
      ),
    cameraIdentity:
      cloneSnapshotValue(snapshot.cameraIdentity, 'cameraIdentity'),
    projectionIdentity:
      cloneSnapshotValue(snapshot.projectionIdentity, 'projectionIdentity'),
    timeIdentity:
      cloneSnapshotValue(snapshot.timeIdentity, 'timeIdentity'),
    chunkIndex: plan.chunkIndex
  };
}

function validateChunkExecutionInput(chunkInput, snapshot, device, raw, plan) {
  const reasons = [];
  if (chunkInput.device !== device) reasons.push('diagnostic-device-reference-mutated');
  if (chunkInput.raw !== raw) reasons.push('raw-spl4-reference-mutated');
  if (chunkInput.chunkIndex !== plan.chunkIndex) reasons.push('chunk-index-mutated');
  if (chunkInput.rangeStart !== plan.rangeStart) reasons.push('range-start-mutated');
  if (chunkInput.rangeCount !== plan.rangeCount) reasons.push('range-count-mutated');
  try {
    const buildConfig = cloneSnapshotValue(chunkInput.buildConfig, 'buildConfig');
    if (!semanticallyEqual(buildConfig, snapshot.buildConfig)) {
      reasons.push('build-config-mutated');
    }
  } catch {
    reasons.push('build-config-mutated-or-unsupported');
  }
  const projectionParams = chunkInput.projectionParams;
  if (
    !(projectionParams instanceof Float32Array) ||
    projectionParams.length !== snapshot.projectionParams.length
  ) {
    reasons.push('projection-params-mutated');
  } else {
    for (let index = 0; index < projectionParams.length; index += 1) {
      if (!Object.is(projectionParams[index], snapshot.projectionParams[index])) {
        reasons.push('projection-params-mutated');
        break;
      }
    }
  }
  for (const [field, reason] of [
    ['sceneInputIdentity', 'scene-input-identity-mutated'],
    ['spl4InputIdentity', 'spl4-input-identity-mutated'],
    ['populationContractIdentity', 'population-contract-identity-mutated'],
    ['cameraIdentity', 'camera-identity-mutated'],
    ['projectionIdentity', 'projection-identity-mutated'],
    ['timeIdentity', 'time-identity-mutated']
  ]) {
    try {
      const value = cloneSnapshotValue(chunkInput[field], field);
      if (!semanticallyEqual(value, snapshot[field])) reasons.push(reason);
    } catch {
      reasons.push(`${reason}-or-unsupported`);
    }
  }
  return [...new Set(reasons)];
}

function createStageAggregate(stageContract) {
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

function firstSemanticMismatchStage(stageSummaries, downstreamOnly = false) {
  const downstreamStages = new Set([
    'productionTileInputAlpha',
    'productionTileInputRgb'
  ]);
  return stageSummaries.find((stage) =>
    (!downstreamOnly || downstreamStages.has(stage.stage)) &&
    stage.semanticResidualCount > 0
  )?.stage ?? null;
}

export function buildPopulationAlignedSemanticComparisonChunkPlan() {
  if (
    !Number.isSafeInteger(POPULATION_SEMANTIC_FIXED_CHUNK_COUNT) ||
    POPULATION_SEMANTIC_FIXED_CHUNK_COUNT !== 8
  ) return [];
  return Array.from(
    { length: POPULATION_SEMANTIC_FIXED_CHUNK_COUNT },
    (_, chunkIndex) => {
      const rangeStart =
        PRODUCTION_RESIDENT_RANGE_START +
        chunkIndex * POPULATION_SEMANTIC_MAX_CHUNK_RECORDS;
      return Object.freeze({
        chunkIndex,
        rangeStart,
        rangeCount: POPULATION_SEMANTIC_MAX_CHUNK_RECORDS,
        rangeEnd: rangeStart + POPULATION_SEMANTIC_MAX_CHUNK_RECORDS
      });
    }
  );
}

function buildExpectedChunkIdentity(plan, input) {
  return buildPopulationSemanticComparisonInputContract({
    rangeStart: plan.rangeStart,
    rangeCount: plan.rangeCount,
    sceneInputIdentity: input.sceneInputIdentity,
    spl4InputIdentity: input.spl4InputIdentity,
    populationContractIdentity: input.populationContractIdentity,
    buildConfig: input.buildConfig,
    cameraIdentity: input.cameraIdentity,
    projectionIdentity: input.projectionIdentity,
    timeIdentity: input.timeIdentity
  });
}

function buildOverallIdentity(snapshot, status, blockedReasons) {
  const base = snapshot
    ? buildExpectedChunkIdentity({
        rangeStart: PRODUCTION_RESIDENT_RANGE_START,
        rangeCount: POPULATION_SEMANTIC_MAX_CHUNK_RECORDS
      }, snapshot)
    : null;
  return {
    schemaVersion: POPULATION_SEMANTIC_ORCHESTRATION_SCHEMA_VERSION,
    contractName: POPULATION_SEMANTIC_ORCHESTRATION_CONTRACT_NAME,
    status,
    reason: blockedReasons[0] ?? null,
    blockedReasons: [...blockedReasons],
    requestedRangeStart: PRODUCTION_RESIDENT_RANGE_START,
    requestedRangeCount: PRODUCTION_RESIDENT_RANGE_COUNT,
    requestedRangeEnd: PRODUCTION_RESIDENT_RANGE_END,
    appliedRangeStart:
      status === 'ready' ? PRODUCTION_RESIDENT_RANGE_START : null,
    appliedRangeCount:
      status === 'ready' ? PRODUCTION_RESIDENT_RANGE_COUNT : 0,
    appliedRangeEnd:
      status === 'ready' ? PRODUCTION_RESIDENT_RANGE_END : null,
    requestedChunkCount: POPULATION_SEMANTIC_FIXED_CHUNK_COUNT,
    chunkSize: POPULATION_SEMANTIC_MAX_CHUNK_RECORDS,
    singleChunkSchemaVersion: POPULATION_SEMANTIC_COMPARISON_SCHEMA_VERSION,
    singleChunkContractName: POPULATION_SEMANTIC_COMPARISON_CONTRACT_NAME,
    sceneInputIdentity: base?.sceneInputIdentity ?? null,
    spl4InputIdentity: base?.spl4InputIdentity ?? null,
    populationContractIdentity: base?.populationContractIdentity ?? null,
    buildConfigIdentity: base?.buildConfigIdentity ?? null,
    cameraProjectionTimeIdentity: base?.cameraProjectionTimeIdentity ?? null,
    expectedProvenance: POPULATION_SEMANTIC_EXPECTED_PROVENANCE,
    actualProvenance: POPULATION_SEMANTIC_ACTUAL_PROVENANCE,
    actualEvidenceSameProductionDispatch: false,
    productionCalculationDependsOnDiagnosticReadback: false,
    mappingContract: base?.mappingContract ?? null,
    populationSelectionPolicy: base?.populationSelectionPolicy ?? null,
    chunkExecutionPolicy: 'fixed-eight-contiguous-chunks-sequential-await'
  };
}

function validateOrchestratorInput(input, plan, snapshotResult) {
  const reasons = [...snapshotResult.blockedReasons];
  if (plan.length !== 8) reasons.push('fixed-eight-chunk-plan-invalid');
  if (!input.device) reasons.push('diagnostic-device-missing');
  if (!input.raw) reasons.push('raw-spl4-input-missing');
  if (input.chunkRunner != null && typeof input.chunkRunner !== 'function') {
    reasons.push('chunk-runner-invalid');
  }
  if (snapshotResult.snapshot) {
    const identity = buildExpectedChunkIdentity(
      plan[0] ?? {
        rangeStart: PRODUCTION_RESIDENT_RANGE_START,
        rangeCount: POPULATION_SEMANTIC_MAX_CHUNK_RECORDS
      },
      snapshotResult.snapshot
    );
    if (identity.status !== 'ready') reasons.push(...identity.blockedReasons);
  }
  return [...new Set(reasons)];
}

function validateStageSummaries(stageSummaries, decision) {
  const reasons = [];
  if (
    !Array.isArray(stageSummaries) ||
    stageSummaries.length !== POPULATION_SEMANTIC_STAGE_CONTRACTS.length
  ) return { reasons: ['stage-contract-count-drift'], stages: [] };
  const stages = [];
  for (let index = 0; index < POPULATION_SEMANTIC_STAGE_CONTRACTS.length; index += 1) {
    const contract = POPULATION_SEMANTIC_STAGE_CONTRACTS[index];
    const stage = stageSummaries[index];
    if (stage?.stage !== contract.key) {
      reasons.push(`stage-${index}-key-drift`);
      continue;
    }
    if (!semanticallyEqual(stage.components, [...contract.components])) {
      reasons.push(`stage-${contract.key}-components-drift`);
    }
    if (stage.tolerance !== contract.tolerance) {
      reasons.push(`stage-${contract.key}-tolerance-drift`);
    }
    const countFields = [
      'comparedCount',
      'comparedComponentCount',
      'validCount',
      'notApplicableCount',
      'missingCount',
      'invalidCount',
      'missingInvalidCount',
      'mismatchCount',
      'componentMismatchCount',
      'precisionAlignedCount',
      'precisionAlignedComponentCount',
      'semanticResidualCount',
      'semanticResidualComponentCount'
    ];
    for (const field of countFields) {
      if (!nonNegativeSafeInteger(stage?.[field])) {
        reasons.push(`stage-${contract.key}-${field}-invalid`);
      }
    }
    if (
      nonNegativeSafeInteger(stage?.missingCount) &&
      nonNegativeSafeInteger(stage?.invalidCount) &&
      stage.missingInvalidCount !== stage.missingCount + stage.invalidCount
    ) reasons.push(`stage-${contract.key}-missing-invalid-total-drift`);
    if (stage?.validCount !== stage?.comparedCount) {
      reasons.push(`stage-${contract.key}-valid-compared-count-drift`);
    }
    if (
      nonNegativeSafeInteger(stage?.validCount) &&
      stage.comparedComponentCount !== stage.validCount * contract.components.length
    ) reasons.push(`stage-${contract.key}-component-count-drift`);
    if (
      nonNegativeSafeInteger(stage?.validCount) &&
      nonNegativeSafeInteger(stage?.notApplicableCount) &&
      nonNegativeSafeInteger(stage?.missingCount) &&
      nonNegativeSafeInteger(stage?.invalidCount) &&
      stage.validCount + stage.notApplicableCount +
        stage.missingCount + stage.invalidCount !==
        POPULATION_SEMANTIC_MAX_CHUNK_RECORDS
    ) reasons.push(`stage-${contract.key}-row-accounting-drift`);
    if (
      !nonNegativeSafeInteger(stage?.mismatchCount) ||
      stage.mismatchCount > stage.comparedCount
    ) reasons.push(`stage-${contract.key}-mismatch-count-invalid`);
    if (
      !nonNegativeSafeInteger(stage?.componentMismatchCount) ||
      stage.componentMismatchCount > stage.comparedComponentCount
    ) reasons.push(`stage-${contract.key}-component-mismatch-count-invalid`);
    if (
      contract.key !== 'webgpuInclusivePixelBounds' &&
      (
        stage?.precisionAlignedCount !== 0 ||
        stage?.precisionAlignedComponentCount !== 0
      )
    ) reasons.push(`stage-${contract.key}-precision-classification-stage-invalid`);
    if (
      stage?.validCount > 0 &&
      (!Number.isFinite(stage.maxAbsoluteError) || stage.maxAbsoluteError < 0)
    ) reasons.push(`stage-${contract.key}-max-error-invalid`);
    if (stage?.validCount === 0 && stage.maxAbsoluteError !== null) {
      reasons.push(`stage-${contract.key}-empty-max-error-not-null`);
    }
    const classification = classifyPopulationSemanticStageEvidence({
      ...stage,
      requiredRecordCount: POPULATION_SEMANTIC_MAX_CHUNK_RECORDS,
      componentCount: contract.components.length
    });
    for (const reason of classification.blockedReasons) {
      reasons.push(`stage-${contract.key}-${reason}`);
    }
    if (stage?.classification !== classification.classification) {
      reasons.push(`stage-${contract.key}-classification-drift`);
    }
    if (!classification.evidenceComplete) {
      reasons.push(`stage-${contract.key}-evidence-incomplete`);
    }
    stages.push({
      stage: contract.key,
      components: [...contract.components],
      comparedCount: stage?.comparedCount,
      comparedComponentCount: stage?.comparedComponentCount,
      validCount: stage?.validCount,
      notApplicableCount: stage?.notApplicableCount,
      missingCount: stage?.missingCount,
      invalidCount: stage?.invalidCount,
      missingInvalidCount: stage?.missingInvalidCount,
      mismatchCount: stage?.mismatchCount,
      componentMismatchCount: stage?.componentMismatchCount,
      precisionAlignedCount: stage?.precisionAlignedCount,
      precisionAlignedComponentCount: stage?.precisionAlignedComponentCount,
      semanticResidualCount: stage?.semanticResidualCount,
      semanticResidualComponentCount: stage?.semanticResidualComponentCount,
      maxAbsoluteError: stage?.maxAbsoluteError,
      tolerance: contract.tolerance,
      classification: stage?.classification
    });
  }
  const anyMismatch = stages.some((stage) => stage.semanticResidualCount > 0);
  if ((decision === 'mismatch') !== anyMismatch) {
    reasons.push('chunk-decision-stage-mismatch-drift');
  }
  return { reasons, stages };
}

function validateRepresentatives(firstMismatches, plan) {
  const reasons = [];
  if (!Array.isArray(firstMismatches)) {
    return { reasons: ['first-mismatches-not-array'], entries: [] };
  }
  if (firstMismatches.length > POPULATION_SEMANTIC_MAX_FIRST_MISMATCHES) {
    reasons.push('first-mismatch-limit-exceeded');
  }
  const stageIndices = new Map(
    POPULATION_SEMANTIC_STAGE_CONTRACTS.map((stage, index) => [stage.key, index])
  );
  const entries = [];
  let previousOrder = null;
  for (const entry of firstMismatches) {
    const stageIndex = stageIndices.get(entry?.stage);
    const stageContract = stageIndex == null
      ? null
      : POPULATION_SEMANTIC_STAGE_CONTRACTS[stageIndex];
    const componentIndex = stageContract
      ? stageContract.components.indexOf(entry?.component)
      : -1;
    const localRow = entry?.localRow;
    const globalResidentRow =
      plan.chunkIndex * POPULATION_SEMANTIC_MAX_CHUNK_RECORDS + localRow;
    const srcIndex = PRODUCTION_RESIDENT_RANGE_START + globalResidentRow;
    if (entry?.chunkIndex !== plan.chunkIndex) {
      reasons.push('first-mismatch-chunk-index-invalid');
    }
    if (
      !Number.isSafeInteger(localRow) ||
      localRow < 0 ||
      localRow >= POPULATION_SEMANTIC_MAX_CHUNK_RECORDS
    ) reasons.push('first-mismatch-local-row-invalid');
    if (entry?.globalResidentRow !== globalResidentRow) {
      reasons.push('first-mismatch-global-row-invalid');
    }
    if (entry?.srcIndex !== srcIndex) reasons.push('first-mismatch-src-index-invalid');
    if (!stageContract) reasons.push('first-mismatch-stage-invalid');
    if (componentIndex < 0) reasons.push('first-mismatch-component-invalid');
    for (const field of ['expected', 'actual', 'absoluteError', 'tolerance']) {
      if (!Number.isFinite(entry?.[field])) {
        reasons.push(`first-mismatch-${field}-invalid`);
      }
    }
    if (Number.isFinite(entry?.absoluteError) && entry.absoluteError < 0) {
      reasons.push('first-mismatch-absolute-error-negative');
    }
    if (stageContract && entry?.tolerance !== stageContract.tolerance) {
      reasons.push('first-mismatch-tolerance-drift');
    }
    if (entry?.expectedProvenance !== POPULATION_SEMANTIC_EXPECTED_PROVENANCE) {
      reasons.push('first-mismatch-expected-provenance-drift');
    }
    if (entry?.actualProvenance !== POPULATION_SEMANTIC_ACTUAL_PROVENANCE) {
      reasons.push('first-mismatch-actual-provenance-drift');
    }
    const rasterStage = stageContract && RASTER_STAGE_KEYS.has(stageContract.key);
    const expectedStageProvenance = rasterStage
      ? POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE[stageContract.key]
      : POPULATION_SEMANTIC_EXPECTED_PROVENANCE;
    const actualStageProvenance = rasterStage
      ? POPULATION_RASTER_SEMANTIC_ACTUAL_PROVENANCE
      : POPULATION_SEMANTIC_ACTUAL_PROVENANCE;
    if (entry?.expectedStageProvenance !== expectedStageProvenance) {
      reasons.push('first-mismatch-expected-stage-provenance-drift');
    }
    if (entry?.actualStageProvenance !== actualStageProvenance) {
      reasons.push('first-mismatch-actual-stage-provenance-drift');
    }
    if (
      Number.isFinite(entry?.expected) &&
      Number.isFinite(entry?.actual) &&
      Number.isFinite(entry?.absoluteError) &&
      entry.absoluteError !== Math.abs(entry.expected - entry.actual)
    ) reasons.push('first-mismatch-absolute-error-drift');
    if (
      stageContract &&
      Number.isFinite(entry?.absoluteError) &&
      entry.absoluteError <= stageContract.tolerance
    ) reasons.push('first-mismatch-not-over-tolerance');
    const order = [localRow, stageIndex ?? -1, componentIndex];
    if (
      previousOrder &&
      (order[0] < previousOrder[0] ||
        (order[0] === previousOrder[0] && order[1] < previousOrder[1]) ||
        (order[0] === previousOrder[0] && order[1] === previousOrder[1] &&
          order[2] <= previousOrder[2]))
    ) reasons.push('first-mismatch-order-invalid');
    previousOrder = order;
    entries.push({
      chunkIndex: plan.chunkIndex,
      localRow,
      globalResidentRow: entry?.globalResidentRow,
      srcIndex: entry?.srcIndex,
      stage: entry?.stage,
      component: entry?.component,
      expected: entry?.expected,
      actual: entry?.actual,
      absoluteError: entry?.absoluteError,
      tolerance: entry?.tolerance,
      expectedProvenance: entry?.expectedProvenance,
      actualProvenance: entry?.actualProvenance,
      expectedStageProvenance: entry?.expectedStageProvenance,
      actualStageProvenance: entry?.actualStageProvenance
    });
  }
  return { reasons: [...new Set(reasons)], entries };
}

function validateChunkResult(result, plan, expectedIdentity) {
  const reasons = [];
  if (!result || typeof result !== 'object') {
    return {
      reasons: ['chunk-result-missing'],
      stages: [],
      representatives: [],
      companionSummary: null
    };
  }
  const retainedPayloadReason = retainedRuntimePayloadReason(result);
  if (retainedPayloadReason) reasons.push(retainedPayloadReason);
  if (result.schemaVersion !== POPULATION_SEMANTIC_COMPARISON_SCHEMA_VERSION) {
    reasons.push('single-chunk-schema-drift');
  }
  if (result.contractName !== POPULATION_SEMANTIC_COMPARISON_CONTRACT_NAME) {
    reasons.push('single-chunk-contract-name-drift');
  }
  if (!semanticallyEqual(result.identity, expectedIdentity)) {
    reasons.push('single-chunk-identity-drift');
  }
  if (result.decision !== 'match' && result.decision !== 'mismatch') {
    reasons.push('chunk-decision-blocked-or-invalid');
  }
  if (result.match !== (result.decision === 'match')) {
    reasons.push('chunk-match-boolean-drift');
  }
  if (result.evidenceComplete !== true) reasons.push('chunk-evidence-incomplete');
  if (!Array.isArray(result.blockedReasons) || result.blockedReasons.length !== 0) {
    reasons.push('chunk-blocked-reasons-not-empty');
  }
  if (result.expectedProvenance !== POPULATION_SEMANTIC_EXPECTED_PROVENANCE) {
    reasons.push('chunk-expected-provenance-drift');
  }
  if (result.actualProvenance !== POPULATION_SEMANTIC_ACTUAL_PROVENANCE) {
    reasons.push('chunk-actual-provenance-drift');
  }
  if (
    !semanticallyEqual(
      result.precisionClassificationProvenance,
      POPULATION_PIXEL_BOUNDARY_PRECISION_CLASSIFICATION_PROVENANCE
    )
  ) reasons.push('chunk-precision-classification-provenance-drift');
  reasons.push(...validateRasterProvenance(result));
  if (result.actualEvidenceSameProductionDispatch !== false) {
    reasons.push('chunk-same-dispatch-flag-drift');
  }
  if (result.productionCalculationDependsOnDiagnosticReadback !== false) {
    reasons.push('chunk-production-readback-dependency-drift');
  }
  if (result.rawRecordArraysIncluded !== false) {
    reasons.push('chunk-raw-record-contract-drift');
  }
  if (result.gpuResourcesIncluded !== false) {
    reasons.push('chunk-gpu-resource-contract-drift');
  }
  if (result.resultSizePopulationIndependent !== true) {
    reasons.push('chunk-result-size-contract-drift');
  }
  if (result.productionBindingCount !== 8) {
    reasons.push('chunk-production-binding-count-drift');
  }
  if (result.productionReadbackPolicyChanged !== false) {
    reasons.push('chunk-production-readback-policy-drift');
  }
  if (result.step113DiagnosticTailChanged !== false) {
    reasons.push('chunk-step113-diagnostic-tail-drift');
  }
  if (result.diagnosticGpuResourceOwnership !== CHUNK_RESOURCE_OWNERSHIP) {
    reasons.push('chunk-resource-ownership-drift');
  }
  if (
    result.rasterObserverGpuResourceOwnership !==
      RASTER_OBSERVER_RESOURCE_OWNERSHIP
  ) reasons.push('chunk-raster-observer-resource-ownership-drift');
  if (result.nativeTileInputBufferUsageChanged !== false) {
    reasons.push('chunk-native-tile-input-buffer-usage-drift');
  }
  const coverage = result.coverage;
  const expectedCoverage = {
    requestedCount: plan.rangeCount,
    processedCount: plan.rangeCount,
    uniqueSrcIndexCount: plan.rangeCount,
    firstSrcIndex: plan.rangeStart,
    lastSrcIndex: plan.rangeEnd - 1,
    missingCount: 0,
    extraCount: 0,
    duplicateCount: 0,
    outOfRangeCount: 0,
    orderMismatchCount: 0,
    coverageComplete: true,
    requestedChunkCount: 1,
    completedChunkCount: 1
  };
  if (!semanticallyEqual(coverage, expectedCoverage)) {
    reasons.push('single-chunk-coverage-drift');
  }
  reasons.push(
    ...validateRasterCompanionCoverage(result.rasterCompanionCoverage, plan)
  );
  const companionLayoutValidation = validateRasterCompanionLayouts(
    result,
    plan,
    expectedIdentity
  );
  reasons.push(...companionLayoutValidation.reasons);
  const stageValidation = validateStageSummaries(
    result.stageSummaries,
    result.decision
  );
  reasons.push(...stageValidation.reasons);
  const expectedFirstSemanticMismatchStage = firstSemanticMismatchStage(
    stageValidation.stages
  );
  const expectedFirstDownstreamMismatchStage = firstSemanticMismatchStage(
    stageValidation.stages,
    true
  );
  if (result.firstSemanticMismatchStage !== expectedFirstSemanticMismatchStage) {
    reasons.push('first-semantic-mismatch-stage-drift');
  }
  if (
    result.firstDownstreamMismatchStage !== expectedFirstDownstreamMismatchStage
  ) reasons.push('first-downstream-mismatch-stage-drift');
  reasons.push(
    ...validatePopulationSemanticStageLocalMismatchSummaries({
      stageLocalMismatchSummaries: result.stageLocalMismatchSummaries,
      stageSummaries: stageValidation.stages,
      scope: 'single-chunk',
      rangeStart: plan.rangeStart,
      rangeCount: plan.rangeCount,
      chunkIndex: plan.chunkIndex
    })
  );
  if (!Array.isArray(result.firstMismatches)) {
    reasons.push('first-mismatches-not-array');
  }
  if (result.firstMismatchCount !== result.firstMismatches?.length) {
    reasons.push('first-mismatch-count-drift');
  }
  if (result.firstMismatchLimit !== POPULATION_SEMANTIC_MAX_FIRST_MISMATCHES) {
    reasons.push('first-mismatch-limit-drift');
  }
  if (result.mismatchScanOrder !== 'chunk-local-row-stage-component') {
    reasons.push('first-mismatch-scan-order-drift');
  }
  const representativeValidation = validateRepresentatives(
    result.firstMismatches,
    plan
  );
  reasons.push(...representativeValidation.reasons);
  for (const entry of representativeValidation.entries) {
    const stage = stageValidation.stages.find(
      (candidate) => candidate.stage === entry.stage
    );
    if (!stage || stage.semanticResidualCount <= 0) {
      reasons.push('first-mismatch-stage-has-no-mismatch');
    }
  }
  const semanticResidualCount = stageValidation.stages.reduce(
    (sum, stage) => sum + (stage.semanticResidualCount ?? 0),
    0
  );
  if (result.decision === 'match' && result.firstMismatches?.length !== 0) {
    reasons.push('match-chunk-has-first-mismatch');
  }
  if (result.decision === 'mismatch' && semanticResidualCount > 0 &&
      result.firstMismatches?.length === 0) {
    reasons.push('mismatch-chunk-missing-representative');
  }
  return {
    reasons: [...new Set(reasons)],
    stages: stageValidation.stages,
    stageLocalMismatchSummaries: result.stageLocalMismatchSummaries,
    representatives: representativeValidation.entries,
    mismatchCount: semanticResidualCount,
    companionSummary: companionLayoutValidation.summary
  };
}

function commitStageAggregates(aggregates, stages) {
  for (let index = 0; index < aggregates.length; index += 1) {
    const aggregate = aggregates[index];
    const stage = stages[index];
    for (const field of [
      'comparedCount',
      'comparedComponentCount',
      'validCount',
      'notApplicableCount',
      'missingCount',
      'invalidCount',
      'mismatchCount',
      'componentMismatchCount',
      'precisionAlignedCount',
      'precisionAlignedComponentCount',
      'semanticResidualCount',
      'semanticResidualComponentCount'
    ]) aggregate[field] += stage[field];
    if (stage.maxAbsoluteError != null) {
      aggregate.maxAbsoluteError = Math.max(
        aggregate.maxAbsoluteError ?? 0,
        stage.maxAbsoluteError
      );
    }
  }
}

function observeRangeDrift(coverageState, identity, plan) {
  const start = identity?.appliedRangeStart;
  const end = identity?.appliedRangeEnd;
  if (Number.isSafeInteger(start)) {
    if (start > plan.rangeStart) coverageState.gapCount += 1;
    if (start < plan.rangeStart) coverageState.overlapCount += 1;
  }
  if (Number.isSafeInteger(end)) {
    if (end < plan.rangeEnd) coverageState.gapCount += 1;
    if (end > plan.rangeEnd) coverageState.overlapCount += 1;
  }
}

function observeCoverageErrors(coverageState, coverage) {
  for (const field of [
    'missingCount',
    'extraCount',
    'duplicateCount',
    'outOfRangeCount',
    'orderMismatchCount'
  ]) {
    if (nonNegativeSafeInteger(coverage?.[field])) {
      coverageState[field] += coverage[field];
    }
  }
}

function buildChunkSummary(plan, result, validationReasons) {
  const mismatchCount = Array.isArray(result?.stageSummaries)
    ? result.stageSummaries.reduce(
        (sum, stage) => sum +
          (nonNegativeSafeInteger(stage?.mismatchCount) ? stage.mismatchCount : 0),
        0
      )
    : 0;
  const accepted = validationReasons.length === 0;
  const semanticResidualCount = Array.isArray(result?.stageSummaries)
    ? result.stageSummaries.reduce(
        (sum, stage) => sum +
          (nonNegativeSafeInteger(stage?.semanticResidualCount)
            ? stage.semanticResidualCount
            : 0),
        0
      )
    : 0;
  return {
    chunkIndex: plan.chunkIndex,
    rangeStart: plan.rangeStart,
    rangeCount: plan.rangeCount,
    rangeEnd: plan.rangeEnd,
    decision: accepted ? result.decision : 'blocked',
    processedRecordCount:
      nonNegativeSafeInteger(result?.coverage?.processedCount)
        ? result.coverage.processedCount
        : 0,
    uniqueSrcIndexCount:
      nonNegativeSafeInteger(result?.coverage?.uniqueSrcIndexCount)
        ? result.coverage.uniqueSrcIndexCount
        : 0,
    coverageComplete: accepted && result.coverage?.coverageComplete === true,
    mismatchCount,
    semanticResidualCount,
    firstSemanticMismatchStage: accepted
      ? result.firstSemanticMismatchStage ?? null
      : null,
    firstDownstreamMismatchStage: accepted
      ? result.firstDownstreamMismatchStage ?? null
      : null,
    blockedReason: validationReasons[0] ?? null
  };
}

async function runDefaultChunk(input) {
  const result = await runPopulationAlignedSemanticComparisonChunk(input);
  if (!Array.isArray(result?.firstMismatches)) return result;
  return {
    ...result,
    firstMismatches: result.firstMismatches.map((entry) =>
      entry?.chunkIndex === 0
        ? { ...entry, chunkIndex: input.chunkIndex }
        : entry
    )
  };
}

export async function runPopulationAlignedSemanticComparisonResidentRange({
  device,
  raw,
  buildConfig,
  projectionParams,
  sceneInputIdentity,
  spl4InputIdentity,
  populationContractIdentity,
  cameraIdentity,
  projectionIdentity,
  timeIdentity,
  chunkRunner = null
} = {}) {
  const callerInput = {
    device,
    raw,
    buildConfig,
    projectionParams,
    sceneInputIdentity,
    spl4InputIdentity,
    populationContractIdentity,
    cameraIdentity,
    projectionIdentity,
    timeIdentity,
    chunkRunner
  };
  const plan = buildPopulationAlignedSemanticComparisonChunkPlan();
  const snapshotResult = buildAuthoritativeExecutionSnapshot(callerInput);
  const authoritativeSnapshot = snapshotResult.snapshot;
  const inputBlockedReasons = validateOrchestratorInput(
    callerInput,
    plan,
    snapshotResult
  );
  const inputReady = inputBlockedReasons.length === 0;
  const expectedChunkIdentities = inputReady
    ? plan.map((chunk) =>
        buildExpectedChunkIdentity(chunk, authoritativeSnapshot)
      )
    : [];
  const blockedReasons = [...inputBlockedReasons];
  const stageAggregates = POPULATION_SEMANTIC_STAGE_CONTRACTS.map(
    createStageAggregate
  );
  const chunkSummaries = [];
  const firstMismatches = [];
  const stageLocalRepresentatives = Object.fromEntries(
    POPULATION_SEMANTIC_STAGE_CONTRACTS.map(({ key }) => [key, []])
  );
  const companionState = {
    validatedChunkCount: 0,
    invariantSummary: null
  };
  const coverageState = {
    processedRecordCount: 0,
    uniqueSrcIndexCount: 0,
    firstSrcIndex: null,
    lastSrcIndex: null,
    missingCount: 0,
    extraCount: 0,
    duplicateCount: 0,
    outOfRangeCount: 0,
    orderMismatchCount: 0,
    gapCount: 0,
    overlapCount: 0,
    completedChunkCount: 0
  };
  const runner = chunkRunner ?? runDefaultChunk;

  if (inputReady) {
    for (const chunk of plan) {
      let chunkInput;
      try {
        chunkInput = buildChunkExecutionInput(
          authoritativeSnapshot,
          device,
          raw,
          chunk
        );
      } catch (error) {
        const reason =
          `chunk-${chunk.chunkIndex}-input-copy-failed:${boundedReason(error?.message ?? error)}`;
        blockedReasons.push(reason);
        chunkSummaries.push({
          chunkIndex: chunk.chunkIndex,
          rangeStart: chunk.rangeStart,
          rangeCount: chunk.rangeCount,
          rangeEnd: chunk.rangeEnd,
          decision: 'blocked',
          processedRecordCount: 0,
          uniqueSrcIndexCount: 0,
          coverageComplete: false,
          mismatchCount: 0,
          blockedReason: reason
        });
        break;
      }
      let result;
      try {
        result = await runner(chunkInput);
      } catch (error) {
        const reason =
          `chunk-${chunk.chunkIndex}-runner-exception:${boundedReason(error?.message ?? error)}`;
        blockedReasons.push(reason);
        chunkSummaries.push({
          chunkIndex: chunk.chunkIndex,
          rangeStart: chunk.rangeStart,
          rangeCount: chunk.rangeCount,
          rangeEnd: chunk.rangeEnd,
          decision: 'blocked',
          processedRecordCount: 0,
          uniqueSrcIndexCount: 0,
          coverageComplete: false,
          mismatchCount: 0,
          blockedReason: reason
        });
        break;
      }
      const mutationReasons = validateChunkExecutionInput(
        chunkInput,
        authoritativeSnapshot,
        device,
        raw,
        chunk
      );
      const validation = mutationReasons.length === 0
        ? validateChunkResult(
            result,
            chunk,
            expectedChunkIdentities[chunk.chunkIndex]
          )
        : {
            reasons: [],
            stages: [],
            representatives: [],
            stageLocalMismatchSummaries: [],
            companionSummary: null
          };
      if (
        mutationReasons.length === 0 &&
        validation.reasons.length === 0 &&
        companionState.invariantSummary &&
        !semanticallyEqual(
          validation.companionSummary,
          companionState.invariantSummary
        )
      ) validation.reasons.push('raster-companion-cross-chunk-layout-drift');
      const chunkReasons = [...mutationReasons, ...validation.reasons].map(
        (reason) => `chunk-${chunk.chunkIndex}-${reason}`
      );
      chunkSummaries.push(buildChunkSummary(chunk, result, chunkReasons));
      if (chunkReasons.length > 0) {
        blockedReasons.push(...chunkReasons);
        break;
      }
      observeRangeDrift(coverageState, result.identity, chunk);
      observeCoverageErrors(coverageState, result.coverage);
      coverageState.completedChunkCount += 1;
      coverageState.processedRecordCount += result.coverage.processedCount;
      coverageState.uniqueSrcIndexCount += result.coverage.uniqueSrcIndexCount;
      if (coverageState.firstSrcIndex == null) {
        coverageState.firstSrcIndex = result.coverage.firstSrcIndex;
      }
      coverageState.lastSrcIndex = result.coverage.lastSrcIndex;
      if (!companionState.invariantSummary) {
        companionState.invariantSummary = validation.companionSummary;
      }
      companionState.validatedChunkCount += 1;
      commitStageAggregates(stageAggregates, validation.stages);
      for (const entry of validation.representatives) {
        if (firstMismatches.length >= POPULATION_SEMANTIC_MAX_FIRST_MISMATCHES) {
          break;
        }
        firstMismatches.push(entry);
      }
      for (const stageSummary of validation.stageLocalMismatchSummaries) {
        const aggregateRepresentatives =
          stageLocalRepresentatives[stageSummary.stage];
        for (const representative of stageSummary.representatives) {
          if (
            aggregateRepresentatives.length >=
              populationSemanticStageLocalRepresentativeLimit(
                stageSummary.stage
              )
          ) break;
          aggregateRepresentatives.push(representative);
        }
      }
    }
  }

  const unprocessedRecordCount = Math.max(
    0,
    PRODUCTION_RESIDENT_RANGE_COUNT - coverageState.processedRecordCount
  );
  coverageState.missingCount = Math.max(
    coverageState.missingCount,
    unprocessedRecordCount
  );
  const coverageComplete =
    inputReady &&
    coverageState.completedChunkCount === POPULATION_SEMANTIC_FIXED_CHUNK_COUNT &&
    coverageState.processedRecordCount === PRODUCTION_RESIDENT_RANGE_COUNT &&
    coverageState.uniqueSrcIndexCount === PRODUCTION_RESIDENT_RANGE_COUNT &&
    coverageState.firstSrcIndex === PRODUCTION_RESIDENT_RANGE_START &&
    coverageState.lastSrcIndex === PRODUCTION_RESIDENT_RANGE_END - 1 &&
    coverageState.missingCount === 0 &&
    coverageState.extraCount === 0 &&
    coverageState.duplicateCount === 0 &&
    coverageState.outOfRangeCount === 0 &&
    coverageState.orderMismatchCount === 0 &&
    coverageState.gapCount === 0 &&
    coverageState.overlapCount === 0;
  if (!coverageComplete) blockedReasons.push('full-population-coverage-incomplete');
  const allCompanionChunksReady =
    companionState.validatedChunkCount === POPULATION_SEMANTIC_FIXED_CHUNK_COUNT;
  if (!allCompanionChunksReady) {
    blockedReasons.push('raster-companion-full-chunk-validation-incomplete');
  }

  for (const aggregate of stageAggregates) {
    if (unprocessedRecordCount > 0) {
      aggregate.missingCount += unprocessedRecordCount;
    }
    aggregate.missingInvalidCount = aggregate.missingCount + aggregate.invalidCount;
    aggregate.classification = classifyPopulationSemanticStageEvidence({
      ...aggregate,
      requiredRecordCount: PRODUCTION_RESIDENT_RANGE_COUNT,
      componentCount: aggregate.components.length
    }).classification;
    if (aggregate.classification.startsWith('blocked-')) {
      blockedReasons.push(
        `${aggregate.stage}:${aggregate.classification}`
      );
    }
  }
  const stageLocalMismatchSummaries =
    buildPopulationSemanticStageLocalMismatchSummaries({
      stageSummaries: stageAggregates,
      representativesByStage: stageLocalRepresentatives
    });
  blockedReasons.push(
    ...validatePopulationSemanticStageLocalMismatchSummaries({
      stageLocalMismatchSummaries,
      stageSummaries: stageAggregates,
      scope: 'orchestration',
      rangeStart: PRODUCTION_RESIDENT_RANGE_START,
      rangeCount: PRODUCTION_RESIDENT_RANGE_COUNT
    })
  );
  const evidenceComplete =
    coverageComplete &&
    stageAggregates.every(
      (stage) => POPULATION_SEMANTIC_COMPLETE_STAGE_CLASSIFICATIONS.includes(
        stage.classification
      )
    );
  if (!evidenceComplete) blockedReasons.push('full-population-evidence-incomplete');
  const uniqueBlockedReasons = [...new Set(blockedReasons)];
  const anyMismatch = stageAggregates.some(
    (stage) => stage.semanticResidualCount > 0
  );
  const decision =
    uniqueBlockedReasons.length > 0
      ? 'blocked'
      : anyMismatch
        ? 'mismatch'
        : 'match';
  const identityStatus = inputReady ? 'ready' : 'blocked';
  return {
    schemaVersion: POPULATION_SEMANTIC_ORCHESTRATION_SCHEMA_VERSION,
    contractName: POPULATION_SEMANTIC_ORCHESTRATION_CONTRACT_NAME,
    decision,
    match: decision === 'match',
    reason:
      uniqueBlockedReasons[0] ?? (anyMismatch ? 'semantic-mismatch' : null),
    blockedReasons: uniqueBlockedReasons,
    identity: buildOverallIdentity(
      authoritativeSnapshot,
      identityStatus,
      inputBlockedReasons
    ),
    coverage: {
      requestedRangeStart: PRODUCTION_RESIDENT_RANGE_START,
      requestedRangeCount: PRODUCTION_RESIDENT_RANGE_COUNT,
      requestedRangeEnd: PRODUCTION_RESIDENT_RANGE_END,
      appliedRangeStart:
        inputReady ? PRODUCTION_RESIDENT_RANGE_START : null,
      appliedRangeCount:
        inputReady ? PRODUCTION_RESIDENT_RANGE_COUNT : 0,
      appliedRangeEnd:
        inputReady ? PRODUCTION_RESIDENT_RANGE_END : null,
      requestedChunkCount: POPULATION_SEMANTIC_FIXED_CHUNK_COUNT,
      completedChunkCount: coverageState.completedChunkCount,
      requestedRecordCount: PRODUCTION_RESIDENT_RANGE_COUNT,
      processedRecordCount: coverageState.processedRecordCount,
      uniqueSrcIndexCount: coverageState.uniqueSrcIndexCount,
      firstSrcIndex: coverageState.firstSrcIndex,
      lastSrcIndex: coverageState.lastSrcIndex,
      missingCount: coverageState.missingCount,
      extraCount: coverageState.extraCount,
      duplicateCount: coverageState.duplicateCount,
      outOfRangeCount: coverageState.outOfRangeCount,
      orderMismatchCount: coverageState.orderMismatchCount,
      gapCount: coverageState.gapCount,
      overlapCount: coverageState.overlapCount,
      coverageComplete
    },
    stageSummaries: stageAggregates,
    stageLocalMismatchSummaries,
    firstSemanticMismatchStage: firstSemanticMismatchStage(stageAggregates),
    firstDownstreamMismatchStage:
      firstSemanticMismatchStage(stageAggregates, true),
    firstMismatches,
    firstMismatchCount: firstMismatches.length,
    firstMismatchLimit: POPULATION_SEMANTIC_MAX_FIRST_MISMATCHES,
    mismatchScanOrder: 'chunk-local-row-stage-component',
    requestedChunkCount: POPULATION_SEMANTIC_FIXED_CHUNK_COUNT,
    completedChunkCount: coverageState.completedChunkCount,
    chunkPlan: {
      rangeStart: PRODUCTION_RESIDENT_RANGE_START,
      rangeCount: PRODUCTION_RESIDENT_RANGE_COUNT,
      rangeEnd: PRODUCTION_RESIDENT_RANGE_END,
      chunkCount: POPULATION_SEMANTIC_FIXED_CHUNK_COUNT,
      chunkSize: POPULATION_SEMANTIC_MAX_CHUNK_RECORDS,
      contiguous: true,
      ordered: true,
      executionMode: 'sequential-await'
    },
    chunkSummaries,
    rasterCompanionSummary: {
      schemaVersion:
        companionState.invariantSummary?.schemaVersion ??
        POPULATION_RASTER_SEMANTIC_COMPANION_LAYOUT_SCHEMA_VERSION,
      contractName:
        companionState.invariantSummary?.contractName ??
        'native-production-raster-semantic-companion-evidence',
      recordCountPerChunk: POPULATION_SEMANTIC_MAX_CHUNK_RECORDS,
      rowStrideVec4: POPULATION_RASTER_SEMANTIC_COMPANION_VEC4_STRIDE,
      rowStrideFloats: POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE,
      rowStrideBytes: POPULATION_RASTER_SEMANTIC_COMPANION_BYTES_PER_RECORD,
      logicalCombinedRowStrideVec4:
        POPULATION_SEMANTIC_LOGICAL_COMBINED_VEC4_STRIDE,
      logicalCombinedRowStrideFloats:
        POPULATION_SEMANTIC_LOGICAL_COMBINED_FLOAT_STRIDE,
      logicalCombinedRowStrideBytes:
        POPULATION_SEMANTIC_LOGICAL_COMBINED_BYTES_PER_RECORD,
      validatedChunkCount: companionState.validatedChunkCount,
      allChunksReady: allCompanionChunksReady,
      allChunksRowAlignmentVerified: allCompanionChunksReady,
      layoutContractConsistentAcrossChunks: allCompanionChunksReady,
      rowAlignment: RASTER_COMPANION_ROW_ALIGNMENT,
      canvasWidth: companionState.invariantSummary?.canvasWidth ?? null,
      canvasHeight: companionState.invariantSummary?.canvasHeight ?? null,
      tileSize: companionState.invariantSummary?.tileSize ?? null,
      tileCols: companionState.invariantSummary?.tileCols ?? null,
      tileRows: companionState.invariantSummary?.tileRows ?? null,
      rasterExpectedProvenance: POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE,
      rasterActualProvenance: POPULATION_RASTER_SEMANTIC_ACTUAL_PROVENANCE,
      actualGpuDeviceScope: POPULATION_RASTER_SEMANTIC_ACTUAL_DEVICE_SCOPE,
      expectedGenerationDependsOnActual: false,
      rasterObserverGpuResourceOwnership: RASTER_OBSERVER_RESOURCE_OWNERSHIP,
      nativeTileInputBufferUsageChanged: false,
      diagnosticOnly: true
    },
    evidenceComplete,
    expectedProvenance: POPULATION_SEMANTIC_EXPECTED_PROVENANCE,
    actualProvenance: POPULATION_SEMANTIC_ACTUAL_PROVENANCE,
    actualGpuDeviceScope: POPULATION_RASTER_SEMANTIC_ACTUAL_DEVICE_SCOPE,
    expectedGenerationDependsOnActual: false,
    precisionClassificationProvenance:
      POPULATION_PIXEL_BOUNDARY_PRECISION_CLASSIFICATION_PROVENANCE,
    actualEvidenceSameProductionDispatch: false,
    productionCalculationDependsOnDiagnosticReadback: false,
    diagnosticGpuResourceOwnership:
      'each-chunk-call-scoped-destroyed-before-next-chunk',
    diagnosticDeviceOwnership: 'caller-owned-reused-not-destroyed',
    singleChunkResultsIncluded: false,
    rawRecordArraysIncluded: false,
    typedArraysIncluded: false,
    gpuResourcesIncluded: false,
    deviceIncluded: false,
    resultSizePopulationIndependent: true
  };
}
