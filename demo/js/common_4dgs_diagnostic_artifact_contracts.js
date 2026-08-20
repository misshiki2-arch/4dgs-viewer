export const WEBGPU_VISIBLE_RECORD_DIAGNOSTIC_SCHEMA_VERSION =
  'phase3-webgpu-visible-record-diagnostic-result-v2';

export const WEBGPU_VISIBLE_RECORD_LINEAGE_SCHEMA_VERSION =
  'phase3-webgpu-visible-record-detailed-lineage-v1';

export const WEBGPU_VISIBLE_RECORD_DIAGNOSTIC_BUNDLE_SCHEMA_VERSION =
  'phase3-webgpu-visible-record-diagnostic-artifact-bundle-v1';

export const WEBGPU_DIAGNOSTIC_DETAIL_DEFAULT_LIMIT = 8;
export const WEBGPU_DIAGNOSTIC_DETAIL_HARD_LIMIT = 32;
export const WEBGPU_DIAGNOSTIC_REPRESENTATIVE_LIMIT = 16;

const WEBGPU_DIAGNOSTIC_DETAIL_SELECTION_SCHEMA_VERSION =
  'phase3-webgpu-diagnostic-detail-selection-v1';

const DETAIL_SELECTION_MODES = new Set([
  'none',
  'explicit-src-indices',
  'first-mismatch',
  'explicit-and-first-mismatch'
]);

const DIAGNOSTIC_STAGE_FIELDS = Object.freeze([
  'radiusContract',
  'covarianceContract',
  'conicContract',
  'aabbContract',
  'tileRangeContract',
  'tileListContract',
  'tileListCapacityContract',
  'tileListValidationContract',
  'tileListValidationUnitContract',
  'tileCountsOffsetsComparisonSurfaceContract',
  'tileCountsToOffsetsDryRun',
  'tileCountsOffsetsSelfComparison',
  'webgpuTileCountsDryRun',
  'tileCountsWebGpuComparison',
  'tileOffsetsFromWebGpuCountsDryRun',
  'tileOffsetsPrefixComparison',
  'webgpuTileOffsetsPrefixDryRun',
  'tileOffsetsWebGpuPrefixComparison',
  'scatterValidationBoundary',
  'tileIndicesSelfComparison',
  'tileIndicesWebGpuScatterComparison',
  'tileListSummaryComparison',
  'webgpuTileListBackendOutput',
  'renderPayloadSortReadiness',
  'depthSortComparison',
  'webgpuRenderHandoffStub',
  'webgpuTileCompositeHandoffStub',
  'webgpuTileCompositeShaderHandoff',
  'webgpuTileCompositeShaderDryRunComparison',
  'webgpuTileCompositeAccumulationDryRunComparison',
  'webgpuFramebufferFreeTileOutputDryRunComparison',
  'webgpuRenderTargetHandoffDryRunComparison',
  'webgpuConstrainedDisplayAdapterDryRunComparison'
]);

const COMPACT_NESTED_SUMMARY_FIELDS = new Set([
  'capacity',
  'comparison',
  'comparisonSummary',
  'metadata',
  'recordCounts',
  'summary',
  'validationSummary'
]);

const BOUNDED_EVIDENCE_FIELDS = new Set([
  'firstMismatches',
  'firstSortDifferences',
  'firstValidationFailures'
]);

function toNonNegativeInteger(value) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? Math.floor(number) : null;
}

function boundedInteger(value, fallback, hardMax) {
  const number = toNonNegativeInteger(value);
  return Math.min(number ?? fallback, hardMax);
}

function uniqueIndices(values) {
  const result = [];
  const seen = new Set();
  for (const value of Array.isArray(values) || ArrayBuffer.isView(values)
    ? values
    : []) {
    const index = toNonNegativeInteger(value);
    if (index === null || seen.has(index)) continue;
    seen.add(index);
    result.push(index);
  }
  return result;
}

function compactArray(values, limit = WEBGPU_DIAGNOSTIC_REPRESENTATIVE_LIMIT) {
  return Array.isArray(values) ? values.slice(0, limit) : [];
}

function pick(source, names) {
  const result = {};
  if (!source || typeof source !== 'object') return result;
  for (const name of names) {
    const value = source[name];
    if (value !== undefined) result[name] = value;
  }
  return result;
}

function compactStageSummary(source) {
  return pick(source, [
    'schemaVersion',
    'contractVersion',
    'status',
    'reason',
    'computeMode',
    'source',
    'classification',
    'fourDStateSourceReady',
    'gaussianAttributeEvaluationReady',
    'gaussianFootprintEvaluationReady',
    'tileAwareRenderInputReady',
    'gpuOwnedTileListLayoutReady',
    'tileCompositorReady',
    'frameImplementationReady',
    'phase3BackendBoundaryReady',
    'candidateCount',
    'recordCount',
    'validRecordCount',
    'computed4DStatePositionCount',
    'computedRenderAttributeCount',
    'computedFootprintPayloadCount',
    'generatedTileRecordCount',
    'totalTileReferenceCount',
    'compositedReferenceCount',
    'activeTileCount',
    'overflowCount',
    'requiredStorageBindingCount',
    'storageBindingCount'
  ]);
}

function compactScalarObject(source) {
  const result = {};
  if (!source || typeof source !== 'object') return result;
  for (const [name, value] of Object.entries(source)) {
    if (
      value === null ||
      typeof value === 'string' ||
      typeof value === 'number' ||
      typeof value === 'boolean'
    ) {
      result[name] = value;
    } else if (BOUNDED_EVIDENCE_FIELDS.has(name) && Array.isArray(value)) {
      result[name] = value.slice(0, WEBGPU_DIAGNOSTIC_REPRESENTATIVE_LIMIT);
      result[`${name}SerializedCount`] = result[name].length;
      result[`${name}Truncated`] = value.length > result[name].length;
    } else if (
      Array.isArray(value) &&
      value.length <= WEBGPU_DIAGNOSTIC_REPRESENTATIVE_LIMIT &&
      value.every((item) =>
        item === null || ['string', 'number', 'boolean'].includes(typeof item)
      )
    ) {
      result[name] = [...value];
    } else if (
      COMPACT_NESTED_SUMMARY_FIELDS.has(name) &&
      value &&
      typeof value === 'object' &&
      !Array.isArray(value)
    ) {
      result[name] = compactScalarObject(value);
    }
  }
  return result;
}

function buildDiagnosticStageAggregates(result) {
  const summaries = {};
  for (const field of DIAGNOSTIC_STAGE_FIELDS) {
    if (result?.[field] == null) continue;
    summaries[field] = compactScalarObject(result[field]);
  }
  return summaries;
}

function buildStageSummaries(result) {
  return {
    stateSource: compactStageSummary(result?.webgpu4DStateSourceContract),
    gaussianAttributes: compactStageSummary(
      result?.webgpuGaussianAttributeEvaluationContract
    ),
    gaussianFootprint: compactStageSummary(
      result?.webgpuGaussianFootprintEvaluationContract
    ),
    tileInput: compactStageSummary(result?.webgpuTileAwareRenderInputContract),
    tileList: compactStageSummary(result?.webgpuGpuOwnedTileListLayoutContract),
    tileCompositor: compactStageSummary(result?.webgpuTileListCompositorContract),
    backendBoundary: compactStageSummary(result?.webgpuPhase3BackendBoundaryContract),
    visibleRecordGate: compactStageSummary(result?.webgpuVisibleRecordGateSummary)
  };
}

export function normalizeWebGpuDiagnosticDetailSelection(input = null) {
  const source = input && typeof input === 'object' ? input : {};
  const canonicalInput =
    source.schemaVersion === WEBGPU_DIAGNOSTIC_DETAIL_SELECTION_SCHEMA_VERSION;
  const rawExplicitSrcIndices =
    source.srcIndices ?? source.selectedSrcIndices ?? source.indices;
  const canonicalSelectionInput =
    rawExplicitSrcIndices == null && canonicalInput;
  const canonicalExplicitSrcIndices =
    canonicalSelectionInput ? source.explicitSrcIndices : null;
  const explicitSrcIndices = uniqueIndices(
    rawExplicitSrcIndices ?? canonicalExplicitSrcIndices
  );
  const requestedMode = String(
    source.mode ?? (explicitSrcIndices.length > 0 ? 'explicit-src-indices' : 'none')
  );
  const mode = DETAIL_SELECTION_MODES.has(requestedMode)
    ? requestedMode
    : 'none';
  const rawConfiguredLimit = source.limit ?? source.maxRecords;
  const canonicalConfiguredLimit =
    canonicalSelectionInput && source.configuredLimit != null
      ? source.configuredLimit
      : undefined;
  const canonicalEffectiveLimit =
    canonicalSelectionInput && source.effectiveLimit != null
      ? source.effectiveLimit
      : undefined;
  const configuredLimit = toNonNegativeInteger(
    rawConfiguredLimit ?? canonicalConfiguredLimit
  );
  const limit = boundedInteger(
    rawConfiguredLimit ?? canonicalEffectiveLimit,
    WEBGPU_DIAGNOSTIC_DETAIL_DEFAULT_LIMIT,
    WEBGPU_DIAGNOSTIC_DETAIL_HARD_LIMIT
  );
  const boundedExplicitSrcIndices = explicitSrcIndices.slice(0, limit);
  const canonicalRequestedCount =
    canonicalSelectionInput
      ? toNonNegativeInteger(source.requestedExplicitSrcIndexCount)
      : null;
  const requestedExplicitSrcIndexCount = Math.max(
    explicitSrcIndices.length,
    canonicalRequestedCount ?? 0
  );
  return {
    schemaVersion: WEBGPU_DIAGNOSTIC_DETAIL_SELECTION_SCHEMA_VERSION,
    mode,
    requestedExplicitSrcIndexCount,
    explicitSrcIndices: boundedExplicitSrcIndices,
    configuredLimit,
    effectiveLimit: limit,
    hardLimit: WEBGPU_DIAGNOSTIC_DETAIL_HARD_LIMIT,
    selectionTruncated:
      (canonicalSelectionInput && source.selectionTruncated === true) ||
      requestedExplicitSrcIndexCount > boundedExplicitSrcIndices.length,
    computeRecordLimitIndependent: true
  };
}

export function resolveWebGpuDiagnosticDetailRows({
  candidateIndices = [],
  selection = null,
  firstMismatches = []
} = {}) {
  const normalized = normalizeWebGpuDiagnosticDetailSelection(selection);
  const rowsBySrcIndex = new Map();
  for (let row = 0; row < candidateIndices.length; row += 1) {
    const srcIndex = toNonNegativeInteger(candidateIndices[row]);
    if (srcIndex !== null && !rowsBySrcIndex.has(srcIndex)) {
      rowsBySrcIndex.set(srcIndex, row);
    }
  }
  const requestedRows = [];
  const requestedSrcIndices = [];
  const appendRow = (rowValue, requestedSrcIndex = null) => {
    const row = toNonNegativeInteger(rowValue);
    if (row === null || row >= candidateIndices.length || requestedRows.includes(row)) {
      return;
    }
    requestedRows.push(row);
    requestedSrcIndices.push(
      requestedSrcIndex ?? toNonNegativeInteger(candidateIndices[row])
    );
  };
  if (
    normalized.mode === 'explicit-src-indices' ||
    normalized.mode === 'explicit-and-first-mismatch'
  ) {
    for (const srcIndex of normalized.explicitSrcIndices) {
      appendRow(rowsBySrcIndex.get(srcIndex), srcIndex);
    }
  }
  if (
    normalized.mode === 'first-mismatch' ||
    normalized.mode === 'explicit-and-first-mismatch'
  ) {
    for (const mismatch of firstMismatches) appendRow(mismatch?.row);
  }
  const rows = requestedRows.slice(0, normalized.effectiveLimit);
  const selectedSrcIndices = rows.map((row) => toNonNegativeInteger(candidateIndices[row]));
  const missingExplicitSrcIndices = normalized.explicitSrcIndices.filter(
    (srcIndex) => !rowsBySrcIndex.has(srcIndex)
  );
  return {
    ...normalized,
    requestedRowCount: requestedRows.length,
    selectedRowCount: rows.length,
    selectedSrcIndices,
    missingExplicitSrcIndices,
    rows,
    selectionTruncated:
      normalized.selectionTruncated || requestedRows.length > rows.length
  };
}

function buildComparison(result) {
  const comparison = result?.recordComparison ?? {};
  const firstMismatches = compactArray(
    comparison.firstMismatches ?? result?.firstMismatches
  );
  return {
    contract: result?.comparisonContract ?? null,
    tolerance: result?.comparisonTolerance ?? null,
    anyMismatch: comparison.anyMismatch ?? result?.anyMismatch ?? null,
    fieldMismatchCount:
      comparison.fieldMismatchCount ?? result?.fieldMismatchCount ?? null,
    maxAbsError: comparison.maxAbsError ?? null,
    mismatchClassification: result?.mismatchClassification ?? null,
    firstMismatches,
    firstMismatchCount: firstMismatches.length,
    firstMismatchLimit: WEBGPU_DIAGNOSTIC_REPRESENTATIVE_LIMIT,
    firstMismatchesTruncated:
      Array.isArray(comparison.firstMismatches) &&
      comparison.firstMismatches.length > firstMismatches.length
  };
}

export function buildWebGpuVisibleRecordDetailedLineageArtifact({
  runtimeResult,
  selection = null,
  artifactSetIdentity = null,
  artifactProvenance = null
} = {}) {
  const normalizedSelection = normalizeWebGpuDiagnosticDetailSelection(selection);
  if (normalizedSelection.mode === 'none') return null;
  const preCullEvidence =
    runtimeResult?.step114PreCullDirectEvidence ??
    runtimeResult?.webgpuPreCullDirectGaussianEvidence ??
    null;
  const sourceRecords = Array.isArray(preCullEvidence?.records)
    ? preCullEvidence.records
    : [];
  const selectedRecords = sourceRecords.slice(0, normalizedSelection.effectiveLimit);
  const selectedSrcIndices = selectedRecords
    .map((record) => toNonNegativeInteger(record?.srcIndex ?? record?.sourceIndex))
    .filter((value) => value !== null);
  return {
    schemaVersion: WEBGPU_VISIBLE_RECORD_LINEAGE_SCHEMA_VERSION,
    artifactRole: 'optional-bounded-detailed-lineage',
    artifactSetIdentity,
    artifactProvenance,
    sourceDiagnosticSchemaVersion: runtimeResult?.schemaVersion ?? null,
    selection: {
      ...normalizedSelection,
      selectedRecordCount: selectedRecords.length,
      selectedSrcIndices,
      missingExplicitSrcIndices: normalizedSelection.explicitSrcIndices.filter(
        (index) => !selectedSrcIndices.includes(index)
      ),
      selectionTruncated:
        normalizedSelection.selectionTruncated ||
        sourceRecords.length > selectedRecords.length
    },
    actualEvidenceSource: preCullEvidence?.actualEvidenceSource ?? null,
    actualEvidenceDispatch: preCullEvidence?.actualEvidenceDispatch ?? null,
    directEvidenceLayout: preCullEvidence?.directEvidenceLayout ?? null,
    fieldAvailabilitySummary: preCullEvidence?.fieldAvailabilitySummary ?? null,
    recordCount: selectedRecords.length,
    records: selectedRecords,
    productionDiagnosticSeparation: {
      productionOutputDependsOnDetailedLineage: false,
      diagnosticReadbackIsProductionDependency: false
    }
  };
}

export function buildWebGpuVisibleRecordCanonicalDiagnosticResult({
  runtimeResult,
  detailSelection = null,
  detailArtifact = null,
  artifactSetIdentity = null,
  artifactProvenance = null
} = {}) {
  const selection = normalizeWebGpuDiagnosticDetailSelection(detailSelection);
  const comparison = buildComparison(runtimeResult);
  const metadata = runtimeResult?.metadata ?? {};
  return {
    schemaVersion: WEBGPU_VISIBLE_RECORD_DIAGNOSTIC_SCHEMA_VERSION,
    artifactRole: 'canonical-compact-diagnostic-result',
    artifactSetIdentity,
    artifactProvenance,
    sourceRuntimeResultSchemaVersion: runtimeResult?.schemaVersion ?? null,
    phaseStep: runtimeResult?.phaseStep ?? metadata?.phase ?? null,
    status: runtimeResult?.status ?? 'unavailable',
    reason: runtimeResult?.reason ?? null,
    input: {
      computeMode: runtimeResult?.computeMode ?? null,
      scaffoldMode: runtimeResult?.scaffoldMode ?? null,
      recordFloats: runtimeResult?.recordFloats ?? null,
      recordLayout: runtimeResult?.recordLayout ?? null,
      inputContract: runtimeResult?.inputContract ?? null,
      inputBufferModes: runtimeResult?.inputBufferModes ?? null,
      comparisonMode: metadata?.comparisonMode ?? null,
      candidateInputSource: metadata?.candidateInputSource ?? null,
      candidateInputReason: metadata?.candidateInputReason ?? null,
      deterministicState: metadata?.deterministicState ?? null
    },
    cardinality: {
      candidateCount: runtimeResult?.candidateCount ?? null,
      computedRecordCount: runtimeResult?.recordCount ?? null,
      validRecordCount: runtimeResult?.validRecordCount ?? null,
      cpuReferenceValidRecordCount:
        runtimeResult?.cpuReferenceValidRecordCount ?? null,
      comparedRecordCount: runtimeResult?.recordCount ?? null,
      serializedFirstMismatchCount: comparison.firstMismatchCount,
      serializedDetailedLineageRecordCount: 0,
      computeCountIndependentFromSerializedDetail: true
    },
    execution: {
      adapterInfoAvailable: runtimeResult?.webgpu?.adapterInfoAvailable ?? null,
      rawBufferUploadMode: runtimeResult?.webgpu?.rawBufferUploadMode ?? null,
      statePositionUploadMode:
        runtimeResult?.webgpu?.statePositionUploadMode ?? null,
      projectionParamMode: runtimeResult?.webgpu?.projectionParamMode ?? null,
      candidateBufferCount: runtimeResult?.webgpu?.candidateBufferCount ?? null,
      outputBufferBytes: runtimeResult?.webgpu?.outputBufferBytes ?? null
    },
    comparison,
    stageSummaries: buildStageSummaries(runtimeResult),
    diagnosticStageAggregates: buildDiagnosticStageAggregates(runtimeResult),
    validation: {
      status: runtimeResult?.status === 'ok' ? 'completed' : 'blocked',
      diagnosticValidationSemanticsPreserved: true,
      comparisonMismatchIsExecutionFailure: false
    },
    timing: runtimeResult?.timing ?? null,
    serializationPolicy: {
      canonicalCardinalityMode: 'aggregate-plus-fixed-bounded-evidence',
      computeRecordCountIndependent: true,
      firstMismatchLimit: WEBGPU_DIAGNOSTIC_REPRESENTATIVE_LIMIT,
      detailedLineageHardLimit: WEBGPU_DIAGNOSTIC_DETAIL_HARD_LIMIT,
      fullBackendSubresultsSerialized: false,
      runtimeHistorySerialized: false,
      captureOrchestrationSerialized: false,
      legacyPayloadAliasesSerialized: false
    },
    detailedLineageArtifact: {
      requested: selection.mode !== 'none',
      required: selection.mode !== 'none',
      present: detailArtifact !== null,
      schemaVersion: detailArtifact?.schemaVersion ?? null,
      selectedRecordCount: detailArtifact?.recordCount ?? 0,
      selectedSrcIndices: detailArtifact?.selection?.selectedSrcIndices ?? [],
      suggestedSuffix: '_webgpu_visible_record_lineage.json'
    }
  };
}

export function buildWebGpuVisibleRecordDiagnosticArtifactBundle({
  runtimeResult,
  detailSelection = null,
  artifactSetIdentity = null,
  artifactProvenance = null
} = {}) {
  const detailedLineageArtifact = buildWebGpuVisibleRecordDetailedLineageArtifact({
    runtimeResult,
    selection: detailSelection,
    artifactSetIdentity,
    artifactProvenance
  });
  const canonicalDiagnosticResult =
    buildWebGpuVisibleRecordCanonicalDiagnosticResult({
      runtimeResult,
      detailSelection,
      detailArtifact: detailedLineageArtifact,
      artifactSetIdentity,
      artifactProvenance
    });
  canonicalDiagnosticResult.cardinality.serializedDetailedLineageRecordCount = 0;
  return {
    schemaVersion: WEBGPU_VISIBLE_RECORD_DIAGNOSTIC_BUNDLE_SCHEMA_VERSION,
    artifactRole: 'producer-only-diagnostic-artifact-bundle',
    artifactSetIdentity,
    artifactProvenance,
    canonicalDiagnosticResult,
    detailedLineageArtifact
  };
}
