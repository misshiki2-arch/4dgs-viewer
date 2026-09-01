import {
  PRODUCTION_TILE_INPUT_ALPHA_F32_CENTRAL_ORACLE_VERSION
} from './common_4dgs_tile_input_alpha_f32_semantic.js';

export const POPULATION_SEMANTIC_COMPARISON_SCHEMA_VERSION =
  'phase3-population-aligned-semantic-comparison-v6';
export const POPULATION_SEMANTIC_COMPARISON_CONTRACT_NAME =
  'single-contiguous-resident-chunk-semantic-comparison';
export const POPULATION_SEMANTIC_EXPECTED_PROVENANCE =
  'CUDA-formula reconstruction from population-aligned SPL4 input';
export const POPULATION_SEMANTIC_ACTUAL_PROVENANCE =
  'separate-diagnostic-dispatch';
export const POPULATION_RASTER_SEMANTIC_ACTUAL_PROVENANCE =
  'native-production-tile-input-storage-buffer-observed-by-diagnostic-wgsl';
export const POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE = Object.freeze({
  productionRasterEligibility:
    'independent CUDA raster eligibility reconstruction from SPL4 and projection contracts',
  projectedCenter:
    'independent CUDA ndc2Pix-equivalent reconstruction from SPL4 and projection contracts',
  cameraDepth:
    'independent CUDA view-space reconstruction from SPL4 and projection contracts',
  webgpuInclusivePixelBounds:
    'independent CPU reconstruction of the WebGPU inclusive bounds contract',
  normalizedInclusiveTileBounds:
    'independent CUDA getRect exclusive-max reconstruction normalized to inclusive max',
  productionTileInputAlpha:
    `independent production-aligned f32 central alpha reconstruction from SPL4 input (${PRODUCTION_TILE_INPUT_ALPHA_F32_CENTRAL_ORACLE_VERSION})`,
  productionTileInputRgb:
    'independent CUDA degree-2 spatial SH reconstruction from SPL4 input and original-position camera direction'
});
export const POPULATION_RASTER_SEMANTIC_ACTUAL_DEVICE_SCOPE =
  'fresh-separate-diagnostic-device-reexecuting-production-evaluator-and-tile-input-path';
export const POPULATION_PIXEL_BOUNDARY_PRECISION_CLASSIFICATION_PROVENANCE =
  Object.freeze({
    stage: 'webgpuInclusivePixelBounds',
    dependencyStage: 'projectedCenter',
    projectedCenterToleranceSource:
      'population-semantic-stage-contract:projectedCenter',
    expectedEnvelope:
      'independent-expected-center-plus-minus-tolerance',
    dependencyConsistency:
      'paired-expected-and-actual-center-radius-viewport-bounds',
    actualUsedForExpectedGeneration: false,
    pixelBoundsToleranceChanged: false
  });

export const PRODUCTION_RESIDENT_RANGE_START = 524288;
export const PRODUCTION_RESIDENT_RANGE_END = 1048576;
export const PRODUCTION_RESIDENT_RANGE_COUNT =
  PRODUCTION_RESIDENT_RANGE_END - PRODUCTION_RESIDENT_RANGE_START;
export const POPULATION_SEMANTIC_MAX_CHUNK_RECORDS = 65536;
export const POPULATION_SEMANTIC_MAX_FIRST_MISMATCHES = 16;
export const POPULATION_SEMANTIC_STAGE_LOCAL_REPRESENTATIVE_LIMIT = 4;
export const POPULATION_TILE_INPUT_ALPHA_STAGE_LOCAL_REPRESENTATIVE_LIMIT = 8;
export const POPULATION_TILE_INPUT_RGB_STAGE_LOCAL_REPRESENTATIVE_LIMIT = 1;
export const POPULATION_SEMANTIC_CONTROLLER_MAX_RESULT_JSON_BYTES = 110000;

export const POPULATION_SEMANTIC_EVIDENCE_VEC4_STRIDE = 8;
export const POPULATION_SEMANTIC_EVIDENCE_FLOAT_STRIDE =
  POPULATION_SEMANTIC_EVIDENCE_VEC4_STRIDE * 4;
export const POPULATION_SEMANTIC_EVIDENCE_BYTES_PER_RECORD =
  POPULATION_SEMANTIC_EVIDENCE_FLOAT_STRIDE * Float32Array.BYTES_PER_ELEMENT;

export const POPULATION_RASTER_SEMANTIC_COMPANION_VEC4_STRIDE = 4;
export const POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE =
  POPULATION_RASTER_SEMANTIC_COMPANION_VEC4_STRIDE * 4;
export const POPULATION_RASTER_SEMANTIC_COMPANION_BYTES_PER_RECORD =
  POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE *
  Float32Array.BYTES_PER_ELEMENT;
export const POPULATION_SEMANTIC_LOGICAL_COMBINED_VEC4_STRIDE =
  POPULATION_SEMANTIC_EVIDENCE_VEC4_STRIDE +
  POPULATION_RASTER_SEMANTIC_COMPANION_VEC4_STRIDE;
export const POPULATION_SEMANTIC_LOGICAL_COMBINED_FLOAT_STRIDE =
  POPULATION_SEMANTIC_EVIDENCE_FLOAT_STRIDE +
  POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE;
export const POPULATION_SEMANTIC_LOGICAL_COMBINED_BYTES_PER_RECORD =
  POPULATION_SEMANTIC_EVIDENCE_BYTES_PER_RECORD +
  POPULATION_RASTER_SEMANTIC_COMPANION_BYTES_PER_RECORD;

export const POPULATION_RASTER_SEMANTIC_COMPANION_LAYOUT_SCHEMA_VERSION =
  'phase3-population-raster-semantic-companion-layout-v2';

export const POPULATION_SEMANTIC_STAGE_CONTRACTS = Object.freeze([
  Object.freeze({
    key: 'temporalEligibility',
    components: Object.freeze(['eligible']),
    tolerance: 0
  }),
  Object.freeze({
    key: 'conditionalStatePosition',
    components: Object.freeze(['x', 'y', 'z']),
    tolerance: 1e-5
  }),
  Object.freeze({
    key: 'conditionalWorldCovariance',
    components: Object.freeze(['xx', 'xy', 'xz', 'yy', 'yz', 'zz']),
    tolerance: 1e-5
  }),
  Object.freeze({
    key: 'cameraSpaceCovariance',
    components: Object.freeze(['xx', 'xy', 'xz', 'yy', 'yz', 'zz']),
    tolerance: 1e-5
  }),
  Object.freeze({
    key: 'projectionJacobian',
    components: Object.freeze(['j00', 'j01', 'j02', 'j10', 'j11', 'j12']),
    tolerance: 1e-4
  }),
  Object.freeze({
    key: 'screenCovariance',
    components: Object.freeze(['xx', 'xy', 'yy']),
    tolerance: 1e-2
  }),
  Object.freeze({
    key: 'conic',
    components: Object.freeze(['x', 'y', 'z']),
    tolerance: 1e-4
  }),
  Object.freeze({
    key: 'radius',
    components: Object.freeze(['pixels']),
    tolerance: 0
  }),
  Object.freeze({
    key: 'productionRasterEligibility',
    components: Object.freeze(['eligible']),
    tolerance: 0
  }),
  Object.freeze({
    key: 'projectedCenter',
    components: Object.freeze(['px', 'py']),
    tolerance: 1e-3
  }),
  Object.freeze({
    key: 'cameraDepth',
    components: Object.freeze(['depth']),
    tolerance: 1e-4
  }),
  Object.freeze({
    key: 'webgpuInclusivePixelBounds',
    components: Object.freeze(['minX', 'minY', 'maxX', 'maxY']),
    tolerance: 0
  }),
  Object.freeze({
    key: 'normalizedInclusiveTileBounds',
    components: Object.freeze(['minX', 'minY', 'maxX', 'maxY']),
    tolerance: 0
  }),
  Object.freeze({
    key: 'productionTileInputAlpha',
    components: Object.freeze(['alpha']),
    tolerance: 1e-5
  }),
  Object.freeze({
    key: 'productionTileInputRgb',
    components: Object.freeze(['r', 'g', 'b']),
    tolerance: 1e-5
  })
]);

export const POPULATION_SEMANTIC_STAGE_LOCAL_MAX_REPRESENTATIVE_RECORDS =
  POPULATION_SEMANTIC_STAGE_CONTRACTS.reduce(
    (sum, stage) => sum + populationSemanticStageLocalRepresentativeLimit(
      stage.key
    ),
    0
  );

export const POPULATION_SEMANTIC_STAGE_LOCAL_CONTEXT_STAGE_KEYS = Object.freeze([
  'temporalEligibility',
  'productionRasterEligibility',
  'projectedCenter',
  'cameraDepth',
  'radius',
  'webgpuInclusivePixelBounds',
  'normalizedInclusiveTileBounds'
]);
const POPULATION_TILE_INPUT_STAGE_LOCAL_CONTEXT_STAGE_KEYS = Object.freeze([
  'temporalEligibility',
  'productionRasterEligibility'
]);

function stageLocalContextKeys(stageKey) {
  return stageKey === 'productionTileInputAlpha' ||
    stageKey === 'productionTileInputRgb'
    ? POPULATION_TILE_INPUT_STAGE_LOCAL_CONTEXT_STAGE_KEYS
    : POPULATION_SEMANTIC_STAGE_LOCAL_CONTEXT_STAGE_KEYS;
}

export function populationSemanticStageLocalRepresentativeLimit(stageKey) {
  if (stageKey === 'productionTileInputAlpha') {
    return POPULATION_TILE_INPUT_ALPHA_STAGE_LOCAL_REPRESENTATIVE_LIMIT;
  }
  if (stageKey === 'productionTileInputRgb') {
    return POPULATION_TILE_INPUT_RGB_STAGE_LOCAL_REPRESENTATIVE_LIMIT;
  }
  return POPULATION_SEMANTIC_STAGE_LOCAL_REPRESENTATIVE_LIMIT;
}

const POPULATION_SEMANTIC_STAGE_LOCAL_ACTUAL_EVIDENCE_SOURCE =
  'diagnostic-gpu-readback';
const POPULATION_SEMANTIC_STAGE_LOCAL_FORBIDDEN_FIELDS = new Set([
  'adapter',
  'candidateIndices',
  'device',
  'gpuResources',
  'raw',
  'rawArrays',
  'rawObject'
]);

export const POPULATION_SEMANTIC_COMPLETE_STAGE_CLASSIFICATIONS =
  Object.freeze(['match', 'precision-aligned', 'mismatch', 'not-applicable']);

function stageContractForKey(stageKey) {
  return POPULATION_SEMANTIC_STAGE_CONTRACTS.find(
    (stage) => stage.key === stageKey
  ) ?? null;
}

function fixedStageEvidence(stageEvidence, stageContract) {
  const values = Array.from(stageEvidence?.values ?? [], Number);
  const valuesReady =
    values.length === stageContract.components.length &&
    values.every(Number.isFinite);
  return {
    valid: stageEvidence?.valid === true && valuesReady,
    missing: stageEvidence?.missing === true,
    values: valuesReady
      ? values
      : Array.from({ length: stageContract.components.length }, () => null)
  };
}

function stageLocalExpectedProvenance(stageKey) {
  return Object.hasOwn(POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE, stageKey)
    ? POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE[stageKey]
    : POPULATION_SEMANTIC_EXPECTED_PROVENANCE;
}

function stageLocalActualProvenance(stageKey) {
  return Object.hasOwn(POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE, stageKey)
    ? POPULATION_RASTER_SEMANTIC_ACTUAL_PROVENANCE
    : POPULATION_SEMANTIC_ACTUAL_PROVENANCE;
}

export function buildPopulationSemanticStageLocalMismatchRepresentative({
  chunkIndex,
  localRow,
  srcIndex,
  stage,
  mismatchComponents,
  expectedRecord,
  actualRecord,
  comparisonClassification = 'mismatch'
} = {}) {
  const stageIndex = POPULATION_SEMANTIC_STAGE_CONTRACTS.findIndex(
    (contract) => contract.key === stage
  );
  const stageContract = stageIndex >= 0
    ? POPULATION_SEMANTIC_STAGE_CONTRACTS[stageIndex]
    : null;
  if (!stageContract) return null;
  const expectedStage = fixedStageEvidence(
    expectedRecord?.stages?.[stage],
    stageContract
  );
  const actualStage = fixedStageEvidence(
    actualRecord?.stages?.[stage],
    stageContract
  );
  const components = Array.from(mismatchComponents ?? [])
    .map((component) => ({
      component: stageContract.components[component.componentIndex] ?? null,
      componentIndex: Number(component.componentIndex),
      expected: Number(component.expectedValue),
      actual: Number(component.actualValue),
      absoluteError: Number(component.absoluteError),
      tolerance: stageContract.tolerance
    }))
    .sort((left, right) => left.componentIndex - right.componentIndex);
  const dependencyContext = Object.fromEntries(
    stageLocalContextKeys(stageContract.key).map((contextStageKey) => {
      const contextContract = stageContractForKey(contextStageKey);
      const expectedContext = fixedStageEvidence(
        expectedRecord?.stages?.[contextStageKey],
        contextContract
      );
      const actualContext = fixedStageEvidence(
        actualRecord?.stages?.[contextStageKey],
        contextContract
      );
      return [contextStageKey, {
        expected: expectedContext.values,
        actual: actualContext.values
      }];
    })
  );
  const normalizedChunkIndex = Number(chunkIndex);
  const normalizedLocalRow = Number(localRow);
  const normalizedSrcIndex = Number(srcIndex);
  return {
    chunkIndex: normalizedChunkIndex,
    localRow: normalizedLocalRow,
    globalResidentRow: normalizedSrcIndex - PRODUCTION_RESIDENT_RANGE_START,
    srcIndex: normalizedSrcIndex,
    stage: stageContract.key,
    stageIndex,
    scanOrderKey: [normalizedChunkIndex, normalizedLocalRow, stageIndex],
    comparisonClassification,
    mismatchComponents: components,
    stageComponentNames: [...stageContract.components],
    expectedStageValues: expectedStage.values,
    actualStageValues: actualStage.values,
    expectedValid: expectedStage.valid,
    actualValid: actualStage.valid,
    actualMissing: actualStage.missing,
    dependencyContext,
    expectedProvenance: POPULATION_SEMANTIC_EXPECTED_PROVENANCE,
    actualProvenance: POPULATION_SEMANTIC_ACTUAL_PROVENANCE,
    expectedStageProvenance: stageLocalExpectedProvenance(stageContract.key),
    actualStageProvenance: stageLocalActualProvenance(stageContract.key),
    actualEvidenceSource: POPULATION_SEMANTIC_STAGE_LOCAL_ACTUAL_EVIDENCE_SOURCE,
    actualEvidenceSameProductionDispatch: false,
    productionCalculationDependsOnDiagnosticReadback: false
  };
}

export function buildPopulationSemanticStageLocalMismatchSummaries({
  stageSummaries,
  representativesByStage = {}
} = {}) {
  const summariesByStage = new Map(
    Array.from(stageSummaries ?? []).map((summary) => [summary?.stage, summary])
  );
  return POPULATION_SEMANTIC_STAGE_CONTRACTS.map((stageContract) => {
    const source = summariesByStage.get(stageContract.key) ?? {};
    const representatives = Array.from(
      representativesByStage?.[stageContract.key] ?? []
    ).slice(0, populationSemanticStageLocalRepresentativeLimit(
      stageContract.key
    ));
    const representativeRecordLimit =
      populationSemanticStageLocalRepresentativeLimit(stageContract.key);
    const sourceMismatchRecordCount = Number(source.mismatchCount ?? 0);
    const sourceComponentMismatchCount = Number(
      source.componentMismatchCount ?? 0
    );
    const sourcePrecisionAlignedRecordCount = Number(
      source.precisionAlignedCount ?? 0
    );
    const sourcePrecisionAlignedComponentCount = Number(
      source.precisionAlignedComponentCount ?? 0
    );
    const sourceSemanticResidualRecordCount = Number(
      source.semanticResidualCount ?? 0
    );
    const sourceSemanticResidualComponentCount = Number(
      source.semanticResidualComponentCount ?? 0
    );
    const sourceClassification = String(
      source.classification ?? 'blocked-incomplete-evidence'
    );
    return {
      stage: stageContract.key,
      sourceMismatchRecordCount,
      sourceComponentMismatchCount,
      sourcePrecisionAlignedRecordCount,
      sourcePrecisionAlignedComponentCount,
      sourceSemanticResidualRecordCount,
      sourceSemanticResidualComponentCount,
      serializedRepresentativeRecordCount: representatives.length,
      representativeRecordLimit,
      truncated:
        sourceMismatchRecordCount > representativeRecordLimit,
      sourceClassification,
      evidenceComplete:
        POPULATION_SEMANTIC_COMPLETE_STAGE_CLASSIFICATIONS.includes(
          sourceClassification
        ),
      representatives
    };
  });
}

function validateFixedContextValues(values, componentCount, path, reasons) {
  if (!Array.isArray(values) || values.length !== componentCount) {
    reasons.push(`${path}-length-drift`);
    return;
  }
  if (!values.every((value) => value === null || Number.isFinite(value))) {
    reasons.push(`${path}-value-invalid`);
  }
}

function stageLocalForbiddenValueReason(
  value,
  path = 'stage-local-summaries',
  ancestors = new WeakSet()
) {
  if (
    value == null ||
    typeof value === 'string' ||
    typeof value === 'boolean' ||
    (typeof value === 'number' && Number.isFinite(value))
  ) return null;
  if (typeof value !== 'object') return `${path}-unsupported-value`;
  if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) {
    return `${path}-typed-array-or-buffer-retained`;
  }
  if (ancestors.has(value)) return `${path}-circular-reference-retained`;
  const prototype = Object.getPrototypeOf(value);
  if (!Array.isArray(value) && prototype !== Object.prototype && prototype !== null) {
    return `${path}-runtime-resource-retained`;
  }
  ancestors.add(value);
  try {
    for (const key of Object.keys(value)) {
      if (POPULATION_SEMANTIC_STAGE_LOCAL_FORBIDDEN_FIELDS.has(key)) {
        return `${path}-${key}-retained`;
      }
      const nested = stageLocalForbiddenValueReason(
        value[key],
        `${path}-${key}`,
        ancestors
      );
      if (nested) return nested;
    }
  } finally {
    ancestors.delete(value);
  }
  return null;
}

export function validatePopulationSemanticStageLocalMismatchSummaries({
  stageLocalMismatchSummaries,
  stageSummaries,
  scope,
  rangeStart,
  rangeCount,
  chunkIndex = null
} = {}) {
  const reasons = [];
  if (
    !Array.isArray(stageLocalMismatchSummaries) ||
    stageLocalMismatchSummaries.length !== POPULATION_SEMANTIC_STAGE_CONTRACTS.length
  ) return ['stage-local-summary-count-drift'];
  const forbiddenReason = stageLocalForbiddenValueReason(
    stageLocalMismatchSummaries
  );
  if (forbiddenReason) reasons.push(forbiddenReason);
  try {
    JSON.stringify(stageLocalMismatchSummaries);
  } catch {
    reasons.push('stage-local-summary-not-json-serializable');
  }
  const sourceByStage = new Map(
    Array.from(stageSummaries ?? []).map((summary) => [summary?.stage, summary])
  );
  let totalRepresentativeRecords = 0;
  for (let stageIndex = 0;
    stageIndex < POPULATION_SEMANTIC_STAGE_CONTRACTS.length;
    stageIndex += 1) {
    const contract = POPULATION_SEMANTIC_STAGE_CONTRACTS[stageIndex];
    const summary = stageLocalMismatchSummaries[stageIndex];
    const source = sourceByStage.get(contract.key);
    const prefix = `stage-local-${contract.key}`;
    if (summary?.stage !== contract.key) reasons.push(`${prefix}-order-drift`);
    if (!source) reasons.push(`${prefix}-source-summary-missing`);
    if (summary?.sourceMismatchRecordCount !== source?.mismatchCount) {
      reasons.push(`${prefix}-source-mismatch-count-drift`);
    }
    if (summary?.sourceComponentMismatchCount !== source?.componentMismatchCount) {
      reasons.push(`${prefix}-source-component-mismatch-count-drift`);
    }
    for (const [summaryField, sourceField] of [
      ['sourcePrecisionAlignedRecordCount', 'precisionAlignedCount'],
      ['sourcePrecisionAlignedComponentCount', 'precisionAlignedComponentCount'],
      ['sourceSemanticResidualRecordCount', 'semanticResidualCount'],
      ['sourceSemanticResidualComponentCount', 'semanticResidualComponentCount']
    ]) {
      if (summary?.[summaryField] !== source?.[sourceField]) {
        reasons.push(`${prefix}-${summaryField}-drift`);
      }
    }
    if (
      summary?.sourceClassification !== source?.classification ||
      summary?.evidenceComplete !==
        POPULATION_SEMANTIC_COMPLETE_STAGE_CLASSIFICATIONS.includes(
          source?.classification
        )
    ) reasons.push(`${prefix}-source-classification-drift`);
    if (
      summary?.representativeRecordLimit !==
        populationSemanticStageLocalRepresentativeLimit(contract.key)
    ) reasons.push(`${prefix}-limit-drift`);
    const representatives = summary?.representatives;
    if (!Array.isArray(representatives)) {
      reasons.push(`${prefix}-representatives-not-array`);
      continue;
    }
    totalRepresentativeRecords += representatives.length;
    const expectedRepresentativeCount = Math.min(
      source?.mismatchCount ?? 0,
      populationSemanticStageLocalRepresentativeLimit(contract.key)
    );
    if (
      representatives.length !== expectedRepresentativeCount ||
      summary.serializedRepresentativeRecordCount !== representatives.length
    ) reasons.push(`${prefix}-representative-count-drift`);
    if (
      summary?.truncated !==
        ((source?.mismatchCount ?? 0) >
          populationSemanticStageLocalRepresentativeLimit(contract.key))
    ) reasons.push(`${prefix}-truncated-drift`);
    const seenRows = new Set();
    let previousOrder = null;
    let serializedMismatchComponentCount = 0;
    let serializedPrecisionAlignedRecordCount = 0;
    let serializedPrecisionAlignedComponentCount = 0;
    let serializedSemanticResidualRecordCount = 0;
    let serializedSemanticResidualComponentCount = 0;
    for (const representative of representatives) {
      const rowKey = `${representative?.chunkIndex}:${representative?.localRow}`;
      if (seenRows.has(rowKey)) reasons.push(`${prefix}-duplicate-row`);
      seenRows.add(rowKey);
      if (representative?.stage !== contract.key ||
          representative?.stageIndex !== stageIndex) {
        reasons.push(`${prefix}-representative-stage-drift`);
      }
      if (
        representative?.comparisonClassification !== 'precision-aligned' &&
        representative?.comparisonClassification !== 'mismatch'
      ) {
        reasons.push(`${prefix}-representative-classification-invalid`);
      } else if (representative.comparisonClassification === 'precision-aligned') {
        serializedPrecisionAlignedRecordCount += 1;
        if (contract.key !== 'webgpuInclusivePixelBounds') {
          reasons.push(`${prefix}-precision-classification-stage-invalid`);
        }
      } else {
        serializedSemanticResidualRecordCount += 1;
      }
      const currentOrder = [representative?.chunkIndex, representative?.localRow];
      if (
        previousOrder &&
        (currentOrder[0] < previousOrder[0] ||
          (currentOrder[0] === previousOrder[0] &&
            currentOrder[1] <= previousOrder[1]))
      ) reasons.push(`${prefix}-representative-order-drift`);
      previousOrder = currentOrder;
      const local = representative?.localRow;
      const chunk = representative?.chunkIndex;
      const global = representative?.globalResidentRow;
      const srcIndex = representative?.srcIndex;
      if (scope === 'single-chunk') {
        if (chunk !== chunkIndex) reasons.push(`${prefix}-chunk-index-drift`);
        if (!Number.isSafeInteger(local) || local < 0 || local >= rangeCount) {
          reasons.push(`${prefix}-local-row-invalid`);
        }
        if (srcIndex !== rangeStart + local) reasons.push(`${prefix}-src-index-drift`);
      } else if (scope === 'orchestration') {
        if (
          !Number.isSafeInteger(chunk) ||
          chunk < 0 ||
          chunk >= PRODUCTION_RESIDENT_RANGE_COUNT /
            POPULATION_SEMANTIC_MAX_CHUNK_RECORDS
        ) reasons.push(`${prefix}-chunk-index-drift`);
        if (
          !Number.isSafeInteger(local) ||
          local < 0 ||
          local >= POPULATION_SEMANTIC_MAX_CHUNK_RECORDS
        ) reasons.push(`${prefix}-local-row-invalid`);
        if (
          global !== chunk * POPULATION_SEMANTIC_MAX_CHUNK_RECORDS + local
        ) reasons.push(`${prefix}-global-row-drift`);
        if (srcIndex !== PRODUCTION_RESIDENT_RANGE_START + global) {
          reasons.push(`${prefix}-src-index-drift`);
        }
      } else {
        reasons.push('stage-local-validation-scope-invalid');
      }
      if (global !== srcIndex - PRODUCTION_RESIDENT_RANGE_START) {
        reasons.push(`${prefix}-global-row-drift`);
      }
      if (
        !Array.isArray(representative?.scanOrderKey) ||
        representative.scanOrderKey.length !== 3 ||
        representative.scanOrderKey[0] !== chunk ||
        representative.scanOrderKey[1] !== local ||
        representative.scanOrderKey[2] !== stageIndex
      ) reasons.push(`${prefix}-scan-order-key-drift`);
      const mismatchComponents = representative?.mismatchComponents;
      if (
        !Array.isArray(mismatchComponents) ||
        mismatchComponents.length <= 0 ||
        mismatchComponents.length > contract.components.length
      ) {
        reasons.push(`${prefix}-mismatch-components-invalid`);
      } else {
        serializedMismatchComponentCount += mismatchComponents.length;
        if (representative?.comparisonClassification === 'precision-aligned') {
          serializedPrecisionAlignedComponentCount += mismatchComponents.length;
        } else if (representative?.comparisonClassification === 'mismatch') {
          serializedSemanticResidualComponentCount += mismatchComponents.length;
        }
        let previousComponentIndex = -1;
        for (const component of mismatchComponents) {
          const componentIndex = component?.componentIndex;
          if (
            !Number.isSafeInteger(componentIndex) ||
            componentIndex <= previousComponentIndex ||
            componentIndex >= contract.components.length ||
            component?.component !== contract.components[componentIndex]
          ) reasons.push(`${prefix}-component-order-or-identity-drift`);
          previousComponentIndex = componentIndex;
          for (const field of ['expected', 'actual', 'absoluteError', 'tolerance']) {
            if (!Number.isFinite(component?.[field])) {
              reasons.push(`${prefix}-component-${field}-invalid`);
            }
          }
          if (component?.tolerance !== contract.tolerance) {
            reasons.push(`${prefix}-component-tolerance-drift`);
          }
          if (
            Number.isFinite(component?.expected) &&
            Number.isFinite(component?.actual) &&
            component?.absoluteError !==
              Math.abs(component.expected - component.actual)
          ) reasons.push(`${prefix}-component-absolute-error-drift`);
          if (component?.absoluteError <= contract.tolerance) {
            reasons.push(`${prefix}-non-mismatch-component-retained`);
          }
          if (
            Number.isSafeInteger(componentIndex) &&
            componentIndex >= 0 &&
            componentIndex < contract.components.length &&
            (
              component?.expected !==
                representative?.expectedStageValues?.[componentIndex] ||
              component?.actual !==
                representative?.actualStageValues?.[componentIndex]
            )
          ) reasons.push(`${prefix}-component-stage-value-drift`);
        }
      }
      if (
        !Array.isArray(representative?.stageComponentNames) ||
        representative.stageComponentNames.length !== contract.components.length ||
        representative.stageComponentNames.some(
          (component, index) => component !== contract.components[index]
        )
      ) reasons.push(`${prefix}-stage-component-names-drift`);
      for (const [field, values] of [
        ['expected-stage-values', representative?.expectedStageValues],
        ['actual-stage-values', representative?.actualStageValues]
      ]) validateFixedContextValues(
        values,
        contract.components.length,
        `${prefix}-${field}`,
        reasons
      );
      if (
        representative?.expectedValid !== true ||
        representative?.actualValid !== true ||
        representative?.actualMissing !== false
      ) reasons.push(`${prefix}-stage-validity-drift`);
      const context = representative?.dependencyContext;
      if (!context || typeof context !== 'object' || Array.isArray(context)) {
        reasons.push(`${prefix}-dependency-context-invalid`);
      } else {
        const contextKeys = Object.keys(context);
        const expectedContextKeys = stageLocalContextKeys(contract.key);
        if (
          contextKeys.length !== expectedContextKeys.length ||
          contextKeys.some(
            (key, index) =>
              key !== expectedContextKeys[index]
          )
        ) reasons.push(`${prefix}-dependency-context-order-drift`);
        for (const contextStageKey of expectedContextKeys) {
          const contextContract = stageContractForKey(contextStageKey);
          const entry = context[contextStageKey];
          if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
            reasons.push(`${prefix}-context-${contextStageKey}-missing`);
            continue;
          }
          validateFixedContextValues(
            entry.expected,
            contextContract.components.length,
            `${prefix}-context-${contextStageKey}-expected`,
            reasons
          );
          validateFixedContextValues(
            entry.actual,
            contextContract.components.length,
            `${prefix}-context-${contextStageKey}-actual`,
            reasons
          );
          if (
            contextStageKey === contract.key &&
            (
              JSON.stringify(entry.expected) !==
                JSON.stringify(representative.expectedStageValues) ||
              JSON.stringify(entry.actual) !==
                JSON.stringify(representative.actualStageValues)
            )
          ) reasons.push(`${prefix}-context-stage-value-drift`);
        }
      }
      if (
        representative?.expectedProvenance !==
          POPULATION_SEMANTIC_EXPECTED_PROVENANCE ||
        representative?.actualProvenance !==
          POPULATION_SEMANTIC_ACTUAL_PROVENANCE ||
        representative?.expectedStageProvenance !==
          stageLocalExpectedProvenance(contract.key) ||
        representative?.actualStageProvenance !==
          stageLocalActualProvenance(contract.key) ||
        representative?.actualEvidenceSource !==
          POPULATION_SEMANTIC_STAGE_LOCAL_ACTUAL_EVIDENCE_SOURCE ||
        representative?.actualEvidenceSameProductionDispatch !== false ||
        representative?.productionCalculationDependsOnDiagnosticReadback !== false
      ) reasons.push(`${prefix}-provenance-drift`);
    }
    if (
      (source?.mismatchCount ?? 0) <=
        populationSemanticStageLocalRepresentativeLimit(contract.key) &&
      serializedMismatchComponentCount !== (source?.componentMismatchCount ?? 0)
    ) reasons.push(`${prefix}-serialized-component-count-drift`);
    if (
      (source?.mismatchCount ?? 0) <=
        populationSemanticStageLocalRepresentativeLimit(contract.key) &&
      (
        serializedPrecisionAlignedRecordCount !==
          (source?.precisionAlignedCount ?? 0) ||
        serializedPrecisionAlignedComponentCount !==
          (source?.precisionAlignedComponentCount ?? 0) ||
        serializedSemanticResidualRecordCount !==
          (source?.semanticResidualCount ?? 0) ||
        serializedSemanticResidualComponentCount !==
          (source?.semanticResidualComponentCount ?? 0)
      )
    ) reasons.push(`${prefix}-serialized-classification-count-drift`);
    if (
      serializedMismatchComponentCount > (source?.componentMismatchCount ?? 0)
    ) reasons.push(`${prefix}-serialized-component-count-out-of-range`);
    if (
      serializedPrecisionAlignedRecordCount >
        (source?.precisionAlignedCount ?? 0) ||
      serializedPrecisionAlignedComponentCount >
        (source?.precisionAlignedComponentCount ?? 0) ||
      serializedSemanticResidualRecordCount >
        (source?.semanticResidualCount ?? 0) ||
      serializedSemanticResidualComponentCount >
        (source?.semanticResidualComponentCount ?? 0)
    ) reasons.push(`${prefix}-serialized-classification-count-out-of-range`);
  }
  if (
    totalRepresentativeRecords >
      POPULATION_SEMANTIC_STAGE_LOCAL_MAX_REPRESENTATIVE_RECORDS
  ) reasons.push('stage-local-total-representative-limit-exceeded');
  return [...new Set(reasons)];
}

function nonNegativeSafeInteger(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

export function classifyPopulationSemanticStageEvidence({
  requiredRecordCount,
  componentCount,
  comparedCount,
  comparedComponentCount,
  validCount,
  notApplicableCount,
  missingCount,
  invalidCount,
  missingInvalidCount,
  mismatchCount,
  componentMismatchCount,
  precisionAlignedCount,
  precisionAlignedComponentCount,
  semanticResidualCount,
  semanticResidualComponentCount,
  maxAbsoluteError
} = {}) {
  const reasons = [];
  const requiredCountReady =
    Number.isSafeInteger(requiredRecordCount) && requiredRecordCount > 0;
  const componentCountReady =
    Number.isSafeInteger(componentCount) && componentCount > 0;
  if (!requiredCountReady) reasons.push('required-record-count-invalid');
  if (!componentCountReady) reasons.push('component-count-invalid');

  const counts = {
    comparedCount,
    comparedComponentCount,
    validCount,
    notApplicableCount,
    missingCount,
    invalidCount,
    missingInvalidCount,
    mismatchCount,
    componentMismatchCount,
    precisionAlignedCount,
    precisionAlignedComponentCount,
    semanticResidualCount,
    semanticResidualComponentCount
  };
  for (const [field, value] of Object.entries(counts)) {
    if (!nonNegativeSafeInteger(value)) reasons.push(`${field}-invalid`);
  }

  const rowCountsReady = [
    validCount,
    notApplicableCount,
    missingCount,
    invalidCount
  ].every(nonNegativeSafeInteger);
  if (
    requiredCountReady &&
    rowCountsReady &&
    validCount + notApplicableCount + missingCount + invalidCount !==
      requiredRecordCount
  ) reasons.push('row-accounting-incomplete');
  if (
    nonNegativeSafeInteger(missingCount) &&
    nonNegativeSafeInteger(invalidCount) &&
    nonNegativeSafeInteger(missingInvalidCount) &&
    missingInvalidCount !== missingCount + invalidCount
  ) reasons.push('missing-invalid-total-drift');
  if (
    nonNegativeSafeInteger(validCount) &&
    nonNegativeSafeInteger(comparedCount) &&
    comparedCount !== validCount
  ) reasons.push('valid-compared-count-drift');
  if (
    nonNegativeSafeInteger(validCount) &&
    nonNegativeSafeInteger(comparedComponentCount) &&
    componentCountReady &&
    comparedComponentCount !== validCount * componentCount
  ) reasons.push('compared-component-count-drift');
  if (
    nonNegativeSafeInteger(mismatchCount) &&
    nonNegativeSafeInteger(validCount) &&
    mismatchCount > validCount
  ) reasons.push('mismatch-count-out-of-range');
  if (
    nonNegativeSafeInteger(componentMismatchCount) &&
    nonNegativeSafeInteger(comparedComponentCount) &&
    componentMismatchCount > comparedComponentCount
  ) reasons.push('component-mismatch-count-out-of-range');
  if (
    nonNegativeSafeInteger(mismatchCount) &&
    nonNegativeSafeInteger(componentMismatchCount) &&
    (
      (mismatchCount === 0 && componentMismatchCount !== 0) ||
      (mismatchCount > 0 && componentMismatchCount < mismatchCount)
    )
  ) reasons.push('row-component-mismatch-count-drift');
  if (
    nonNegativeSafeInteger(mismatchCount) &&
    nonNegativeSafeInteger(precisionAlignedCount) &&
    nonNegativeSafeInteger(semanticResidualCount) &&
    mismatchCount !== precisionAlignedCount + semanticResidualCount
  ) reasons.push('raw-precision-residual-record-count-drift');
  if (
    nonNegativeSafeInteger(componentMismatchCount) &&
    nonNegativeSafeInteger(precisionAlignedComponentCount) &&
    nonNegativeSafeInteger(semanticResidualComponentCount) &&
    componentMismatchCount !==
      precisionAlignedComponentCount + semanticResidualComponentCount
  ) reasons.push('raw-precision-residual-component-count-drift');
  for (const [name, recordCount, componentTotal] of [
    ['precision-aligned', precisionAlignedCount, precisionAlignedComponentCount],
    ['semantic-residual', semanticResidualCount, semanticResidualComponentCount]
  ]) {
    if (
      nonNegativeSafeInteger(recordCount) &&
      nonNegativeSafeInteger(componentTotal) &&
      (
        (recordCount === 0 && componentTotal !== 0) ||
        (recordCount > 0 && componentTotal < recordCount)
      )
    ) reasons.push(`${name}-row-component-count-drift`);
  }
  if (
    nonNegativeSafeInteger(validCount) &&
    (
      (validCount > 0 &&
        (!Number.isFinite(maxAbsoluteError) || maxAbsoluteError < 0)) ||
      (validCount === 0 && maxAbsoluteError !== null)
    )
  ) reasons.push('max-absolute-error-invalid');
  if (
    nonNegativeSafeInteger(missingCount) &&
    nonNegativeSafeInteger(invalidCount) &&
    (missingCount > 0 || invalidCount > 0)
  ) reasons.push('missing-or-invalid-evidence');

  let classification = 'blocked-incomplete-evidence';
  if (reasons.length === 0) {
    if (validCount > 0) {
      classification = semanticResidualCount > 0
        ? 'mismatch'
        : precisionAlignedCount > 0
          ? 'precision-aligned'
          : 'match';
    } else if (
      notApplicableCount === requiredRecordCount &&
      mismatchCount === 0 &&
      componentMismatchCount === 0 &&
      precisionAlignedCount === 0 &&
      precisionAlignedComponentCount === 0 &&
      semanticResidualCount === 0 &&
      semanticResidualComponentCount === 0 &&
      maxAbsoluteError === null
    ) {
      classification = 'not-applicable';
    } else {
      classification = 'blocked-no-valid-evidence';
    }
  }
  return {
    classification,
    evidenceComplete:
      POPULATION_SEMANTIC_COMPLETE_STAGE_CLASSIFICATIONS.includes(
        classification
      ),
    accountingComplete: reasons.length === 0,
    blockedReasons: reasons
  };
}

function finiteInteger(value) {
  const number = Number(value);
  return Number.isInteger(number) ? number : null;
}

function identityPresent(value) {
  if (typeof value === 'string') return value.length > 0;
  if (typeof value === 'number' || typeof value === 'boolean') return true;
  return value != null && typeof value === 'object';
}

function cloneIdentity(value) {
  if (value == null || typeof value !== 'object') return value;
  return JSON.parse(JSON.stringify(value));
}

export function buildPopulationSemanticComparisonInputContract({
  rangeStart,
  rangeCount,
  sceneInputIdentity,
  spl4InputIdentity,
  populationContractIdentity,
  buildConfig = {},
  cameraIdentity,
  projectionIdentity,
  timeIdentity
} = {}) {
  const requestedStart = finiteInteger(rangeStart);
  const requestedCount = finiteInteger(rangeCount);
  const requestedEnd =
    requestedStart == null || requestedCount == null
      ? null
      : requestedStart + requestedCount;
  const reasons = [];
  if (requestedStart == null) reasons.push('range-start-invalid');
  if (
    requestedCount == null ||
    requestedCount <= 0 ||
    requestedCount > POPULATION_SEMANTIC_MAX_CHUNK_RECORDS
  ) reasons.push('range-count-out-of-bounds');
  if (
    requestedStart != null &&
    requestedEnd != null &&
    (requestedStart < PRODUCTION_RESIDENT_RANGE_START ||
      requestedEnd > PRODUCTION_RESIDENT_RANGE_END)
  ) reasons.push('range-outside-production-resident-population');
  for (const [name, value] of [
    ['scene-input-identity', sceneInputIdentity],
    ['spl4-input-identity', spl4InputIdentity],
    ['population-contract-identity', populationContractIdentity],
    ['camera-identity', cameraIdentity],
    ['projection-identity', projectionIdentity],
    ['time-identity', timeIdentity]
  ]) {
    if (!identityPresent(value)) reasons.push(`${name}-missing`);
  }
  const timestamp = Number(buildConfig?.timestamp);
  if (!Number.isFinite(timestamp)) reasons.push('build-config-timestamp-invalid');

  const ready = reasons.length === 0;
  return {
    schemaVersion: POPULATION_SEMANTIC_COMPARISON_SCHEMA_VERSION,
    contractName: POPULATION_SEMANTIC_COMPARISON_CONTRACT_NAME,
    status: ready ? 'ready' : 'blocked',
    reason: reasons[0] ?? null,
    blockedReasons: reasons,
    requestedRangeStart: requestedStart,
    requestedRangeCount: requestedCount,
    requestedRangeEnd: requestedEnd,
    appliedRangeStart: ready ? requestedStart : null,
    appliedRangeCount: ready ? requestedCount : 0,
    appliedRangeEnd: ready ? requestedEnd : null,
    productionResidentRangeStart: PRODUCTION_RESIDENT_RANGE_START,
    productionResidentRangeCount: PRODUCTION_RESIDENT_RANGE_COUNT,
    productionResidentRangeEnd: PRODUCTION_RESIDENT_RANGE_END,
    requestedChunkCount: 1,
    sceneInputIdentity: cloneIdentity(sceneInputIdentity),
    spl4InputIdentity: cloneIdentity(spl4InputIdentity),
    populationContractIdentity: cloneIdentity(populationContractIdentity),
    buildConfigIdentity: {
      timestamp: Number.isFinite(timestamp) ? timestamp : null,
      scalingModifier: Number.isFinite(Number(buildConfig?.scalingModifier))
        ? Number(buildConfig.scalingModifier)
        : 1,
      sigmaScale: Number.isFinite(Number(buildConfig?.sigmaScale))
        ? Number(buildConfig.sigmaScale)
        : 1,
      prefilterVar: Number.isFinite(Number(buildConfig?.prefilterVar))
        ? Number(buildConfig.prefilterVar)
        : -1
    },
    cameraProjectionTimeIdentity: {
      camera: cloneIdentity(cameraIdentity),
      projection: cloneIdentity(projectionIdentity),
      time: cloneIdentity(timeIdentity)
    },
    expectedProvenance: POPULATION_SEMANTIC_EXPECTED_PROVENANCE,
    actualProvenance: POPULATION_SEMANTIC_ACTUAL_PROVENANCE,
    actualEvidenceSameProductionDispatch: false,
    productionCalculationDependsOnDiagnosticReadback: false,
    mappingContract: {
      localRowToSrcIndex: 'srcIndex = rangeStart + localRow',
      srcIndexToGlobalResidentRow:
        'globalResidentRow = srcIndex - productionResidentRangeStart',
      localRowOrderMatchesSrcIndexOrder: true
    },
    populationSelectionPolicy:
      'explicit-contiguous-original-source-range-no-prepend-no-fallback-no-dedup-no-fraction'
  };
}

export function buildPopulationSemanticEvidenceLayoutContract({
  recordCount,
  productionVec4Count,
  step113DiagnosticVec4Count,
  totalVec4Count
} = {}) {
  const count = finiteInteger(recordCount) ?? 0;
  const semanticVec4Count = count * POPULATION_SEMANTIC_EVIDENCE_VEC4_STRIDE;
  const semanticVec4Offset =
    (finiteInteger(productionVec4Count) ?? 0) +
    (finiteInteger(step113DiagnosticVec4Count) ?? 0);
  return {
    schemaVersion: 'phase3-population-semantic-packed-evidence-layout-v1',
    packingMode:
      'diagnostic-only-tail-after-production-payload-and-step113-tail',
    bufferName: 'footprintPayload',
    recordCount: count,
    rowStrideVec4: POPULATION_SEMANTIC_EVIDENCE_VEC4_STRIDE,
    rowStrideFloats: POPULATION_SEMANTIC_EVIDENCE_FLOAT_STRIDE,
    rowStrideBytes: POPULATION_SEMANTIC_EVIDENCE_BYTES_PER_RECORD,
    evidenceVec4Offset: semanticVec4Offset,
    evidenceFloatOffset: semanticVec4Offset * 4,
    evidenceByteOffset:
      semanticVec4Offset * 4 * Float32Array.BYTES_PER_ELEMENT,
    evidenceVec4Count: semanticVec4Count,
    evidenceByteSize:
      semanticVec4Count * 4 * Float32Array.BYTES_PER_ELEMENT,
    totalVec4Count: finiteInteger(totalVec4Count) ?? 0,
    storageBindingAdded: false,
    productionPayloadLayoutChanged: false,
    step113DiagnosticTailLayoutChanged: false,
    diagnosticOnly: true
  };
}

export function buildPopulationSemanticDiagnosticWorksetResourceIdentity(
  inputContract
) {
  const start = finiteInteger(inputContract?.appliedRangeStart);
  const count = finiteInteger(inputContract?.appliedRangeCount);
  const schemaVersion = String(inputContract?.schemaVersion ?? 'unknown');
  if (inputContract?.status !== 'ready' || start == null || count == null || count <= 0) {
    return null;
  }
  const comparisonIdentity = JSON.stringify({
    schemaVersion,
    start,
    count,
    sceneInputIdentity: inputContract.sceneInputIdentity,
    spl4InputIdentity: inputContract.spl4InputIdentity,
    populationContractIdentity: inputContract.populationContractIdentity,
    buildConfigIdentity: inputContract.buildConfigIdentity,
    cameraProjectionTimeIdentity: inputContract.cameraProjectionTimeIdentity
  });
  return `population-semantic-workset:${comparisonIdentity}`;
}

export function buildPopulationRasterSemanticCompanionLayoutContract({
  recordCount,
  evidenceFloatCount,
  sourceWorksetResourceIdentity,
  sourceStateResourceIdentity,
  sourceTileInputResourceIdentity,
  canvasWidth,
  canvasHeight,
  tileSize,
  tileCols,
  tileRows,
  observerDispatchSubmitted = false,
  observerReadbackCompleted = false,
  observerOwnedBuffersDestroyed = false,
  reason = null
} = {}) {
  const count = finiteInteger(recordCount) ?? 0;
  const floatCount = finiteInteger(evidenceFloatCount) ?? 0;
  const width = finiteInteger(canvasWidth) ?? 0;
  const height = finiteInteger(canvasHeight) ?? 0;
  const size = finiteInteger(tileSize) ?? 0;
  const cols = finiteInteger(tileCols) ?? 0;
  const rows = finiteInteger(tileRows) ?? 0;
  const expectedFloatCount =
    count * POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE;
  const ready =
    count > 0 &&
    floatCount === expectedFloatCount &&
    typeof sourceWorksetResourceIdentity === 'string' &&
    sourceWorksetResourceIdentity.length > 0 &&
    typeof sourceStateResourceIdentity === 'string' &&
    sourceStateResourceIdentity.length > 0 &&
    typeof sourceTileInputResourceIdentity === 'string' &&
    sourceTileInputResourceIdentity.length > 0 &&
    width > 0 && height > 0 && size > 0 && cols > 0 && rows > 0 &&
    cols === Math.ceil(width / size) &&
    rows === Math.ceil(height / size) &&
    observerDispatchSubmitted === true &&
    observerReadbackCompleted === true &&
    observerOwnedBuffersDestroyed === true;
  return {
    schemaVersion: POPULATION_RASTER_SEMANTIC_COMPANION_LAYOUT_SCHEMA_VERSION,
    contractName: 'native-production-raster-semantic-companion-evidence',
    status: ready ? 'ready' : 'blocked',
    reason: ready ? null : reason ?? 'raster-semantic-companion-evidence-incomplete',
    recordCount: count,
    rowStrideVec4: POPULATION_RASTER_SEMANTIC_COMPANION_VEC4_STRIDE,
    rowStrideFloats: POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE,
    rowStrideBytes: POPULATION_RASTER_SEMANTIC_COMPANION_BYTES_PER_RECORD,
    evidenceFloatCount: floatCount,
    evidenceByteSize: floatCount * Float32Array.BYTES_PER_ELEMENT,
    logicalCombinedRowStrideVec4:
      POPULATION_SEMANTIC_LOGICAL_COMBINED_VEC4_STRIDE,
    logicalCombinedRowStrideFloats:
      POPULATION_SEMANTIC_LOGICAL_COMBINED_FLOAT_STRIDE,
    logicalCombinedRowStrideBytes:
      POPULATION_SEMANTIC_LOGICAL_COMBINED_BYTES_PER_RECORD,
    vec4Layout: [
      'eligibility-px-py-depth',
      'pixel-minX-minY-maxX-maxY',
      'tile-minX-minY-maxX-maxY',
      'production-tile-input-r-g-b-alpha'
    ],
    rowAlignment: 'local-row-matches-explicit-candidate-index-order',
    sourceWorksetResourceIdentity,
    sourceStateResourceIdentity,
    sourceTileInputResourceIdentity,
    canvasWidth: width,
    canvasHeight: height,
    tileSize: size,
    tileCols: cols,
    tileRows: rows,
    observerDispatchSubmitted: observerDispatchSubmitted === true,
    observerReadbackCompleted: observerReadbackCompleted === true,
    observerOwnedBuffersDestroyed: observerOwnedBuffersDestroyed === true,
    nativeTileInputBufferUsageChanged: false,
    observerOutputUsage: 'storage-copy-src',
    observerStagingUsage: 'copy-dst-map-read',
    actualGpuDeviceScope: POPULATION_RASTER_SEMANTIC_ACTUAL_DEVICE_SCOPE,
    expectedGenerationDependsOnActual: false,
    diagnosticOnly: true
  };
}

export function buildPopulationSemanticCoverageContract({
  inputContract,
  candidateIndices,
  actualRecordCount,
  actualRows = []
} = {}) {
  const requestedCount = inputContract?.appliedRangeCount ?? 0;
  const rangeStart = inputContract?.appliedRangeStart ?? 0;
  const processedRecordCount = finiteInteger(actualRecordCount) ?? 0;
  const indices = Array.from(candidateIndices ?? [], Number);
  const expectedCount = Math.min(requestedCount, indices.length);
  const seen = new Set();
  let duplicateCount = 0;
  let outOfRangeCount = 0;
  let orderMismatchCount = 0;
  for (let row = 0; row < indices.length; row += 1) {
    const srcIndex = indices[row];
    if (seen.has(srcIndex)) duplicateCount += 1;
    seen.add(srcIndex);
    if (
      !Number.isInteger(srcIndex) ||
      srcIndex < rangeStart ||
      srcIndex >= rangeStart + requestedCount
    ) outOfRangeCount += 1;
    if (srcIndex !== rangeStart + row) orderMismatchCount += 1;
  }
  const rowCount = Math.min(actualRows.length, requestedCount);
  for (let localRow = 0; localRow < rowCount; localRow += 1) {
    if (actualRows[localRow] !== localRow) orderMismatchCount += 1;
  }
  const missingCount = Math.max(
    Math.max(0, requestedCount - expectedCount),
    Math.max(0, requestedCount - processedRecordCount)
  );
  const extraCount = Math.max(
    Math.max(0, indices.length - requestedCount),
    Math.max(0, processedRecordCount - requestedCount)
  );
  const coverageComplete =
    inputContract?.status === 'ready' &&
    processedRecordCount === requestedCount &&
    indices.length === requestedCount &&
    missingCount === 0 &&
    extraCount === 0 &&
    duplicateCount === 0 &&
    outOfRangeCount === 0 &&
    orderMismatchCount === 0 &&
    seen.size === requestedCount;
  return {
    requestedCount,
    processedCount: Math.min(processedRecordCount, requestedCount),
    uniqueSrcIndexCount: seen.size,
    firstSrcIndex: indices.length > 0 ? indices[0] : null,
    lastSrcIndex: indices.length > 0 ? indices[indices.length - 1] : null,
    missingCount,
    extraCount,
    duplicateCount,
    outOfRangeCount,
    orderMismatchCount,
    coverageComplete,
    requestedChunkCount: 1,
    completedChunkCount: coverageComplete ? 1 : 0
  };
}
