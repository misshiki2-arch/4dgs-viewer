import assert from 'node:assert/strict';
import {
  WEBGPU_DIAGNOSTIC_DETAIL_HARD_LIMIT,
  WEBGPU_VISIBLE_RECORD_DIAGNOSTIC_SCHEMA_VERSION,
  WEBGPU_VISIBLE_RECORD_LINEAGE_SCHEMA_VERSION,
  buildWebGpuVisibleRecordDiagnosticArtifactBundle,
  normalizeWebGpuDiagnosticDetailSelection,
  resolveWebGpuDiagnosticDetailRows
} from '../demo/js/common_4dgs_diagnostic_artifact_contracts.js';

function buildLineageRecord(srcIndex) {
  return {
    srcIndex,
    sourceIndex: srcIndex,
    available: true,
    actualEvidenceSource:
      'webgpu-production-wgsl-visible-record-pre-cull-readback',
    actualEvidenceDispatch:
      'same-webgpu-visible-record-production-dispatch-diagnostic-buffer',
    temporalEvaluation: {
      temporalMotionDelta: [0.1, 0.2, 0.3],
      postTemporalWorldPosition: [1, 2, 3]
    },
    clipPosition: [0, 0, 0, 1],
    ndc: [0, 0, 0],
    screenCenter: [640, 360],
    radius: 2
  };
}

function buildRuntimeResult(recordCount) {
  const firstMismatches = Array.from({ length: 40 }, (_, row) => ({
    row,
    field: 'px',
    component: null,
    expected: row,
    actual: row + 1,
    absError: 1
  }));
  const lineageRecords = Array.from({ length: 100 }, (_, index) =>
    buildLineageRecord(index)
  );
  return {
    schemaVersion: 'phase3-step2-webgpu-visible-record-dry-run-v1',
    phaseStep: 'generic-phase',
    status: 'ok',
    reason: 'ok',
    computeMode: 'webgpu-storage-buffer-compute-fixed-record',
    scaffoldMode: 'diagnostic',
    candidateCount: recordCount,
    recordCount,
    validRecordCount: recordCount - 1,
    cpuReferenceValidRecordCount: recordCount - 1,
    recordFloats: 12,
    recordLayout: [['srcIndex', 0, 1]],
    inputContract: { schemaVersion: 'input-v1' },
    inputBufferModes: { raw: 'storage' },
    recordComparison: {
      anyMismatch: true,
      fieldMismatchCount: 40,
      maxAbsError: 1,
      firstMismatches
    },
    mismatchClassification: 'webgpu-fixed-record-field-mismatch',
    comparisonTolerance: { epsilon: 1e-3, maxMismatches: 40 },
    webgpu4DStateSourceContract: {
      status: 'ok',
      fourDStateSourceReady: true,
      computed4DStatePositionCount: recordCount,
      records: Array.from({ length: recordCount }, () => 1)
    },
    webgpuTileListCompositorContract: {
      status: 'ok',
      tileCompositorReady: true,
      totalTileReferenceCount: recordCount * 2,
      tileIndices: Array.from({ length: recordCount }, (_, index) => index)
    },
    tileCountsWebGpuComparison: {
      status: 'ok',
      anyMismatch: false,
      mismatchCount: 0,
      validationSummary: {
        comparisonReady: true,
        firstValidationFailures: []
      },
      tileCounts: Array.from({ length: recordCount }, () => 1)
    },
    step114PreCullDirectEvidence: {
      actualEvidenceSource:
        'webgpu-production-wgsl-visible-record-pre-cull-readback',
      actualEvidenceDispatch:
        'same-webgpu-visible-record-production-dispatch-diagnostic-buffer',
      records: lineageRecords
    },
    webgpuPreCullDirectGaussianEvidence: {
      records: lineageRecords
    },
    webgpu: {
      adapterInfoAvailable: true,
      candidateBufferCount: recordCount,
      outputBufferBytes: recordCount * 48,
      fullMirror: Array.from({ length: recordCount }, (_, index) => index)
    },
    metadata: {
      phase: 'generic-phase',
      comparisonMode: 'generic-comparison',
      candidateInputSource: 'synthetic',
      deterministicState: { frame: 151, time: 23.2 }
    },
    timing: { totalMs: 1 }
  };
}

const detailSelection = {
  mode: 'explicit-src-indices',
  srcIndices: Array.from({ length: 100 }, (_, index) => index),
  limit: 8
};

const impl8SrcIndices = [
  658947,
  771007,
  788034,
  826401,
  835183,
  852955,
  863505,
  906711
];
const immutableImpl8SrcIndices = Object.freeze([...impl8SrcIndices]);
const immutableRawSelection = Object.freeze({
  mode: 'explicit-src-indices',
  srcIndices: immutableImpl8SrcIndices,
  limit: 8
});
const canonicalImpl8Selection = normalizeWebGpuDiagnosticDetailSelection(
  immutableRawSelection
);
assert.equal(canonicalImpl8Selection.mode, 'explicit-src-indices');
assert.equal(canonicalImpl8Selection.requestedExplicitSrcIndexCount, 8);
assert.deepEqual(canonicalImpl8Selection.explicitSrcIndices, impl8SrcIndices);
assert.deepEqual(immutableRawSelection.srcIndices, impl8SrcIndices);

const immutableCanonicalImpl8Selection = Object.freeze({
  ...canonicalImpl8Selection,
  explicitSrcIndices: Object.freeze([
    ...canonicalImpl8Selection.explicitSrcIndices
  ])
});
const renormalizedImpl8Selection = normalizeWebGpuDiagnosticDetailSelection(
  immutableCanonicalImpl8Selection
);
assert.deepEqual(renormalizedImpl8Selection, canonicalImpl8Selection);
assert.deepEqual(
  immutableCanonicalImpl8Selection.explicitSrcIndices,
  impl8SrcIndices
);

for (const [alias, expected] of [
  ['srcIndices', [11, 12]],
  ['selectedSrcIndices', [21, 22]],
  ['indices', [31, 32]]
]) {
  assert.deepEqual(
    normalizeWebGpuDiagnosticDetailSelection({
      mode: 'explicit-src-indices',
      [alias]: expected,
      limit: 8
    }).explicitSrcIndices,
    expected
  );
}
const rawAliasPrioritySelection = normalizeWebGpuDiagnosticDetailSelection({
  schemaVersion: canonicalImpl8Selection.schemaVersion,
  mode: 'explicit-src-indices',
  srcIndices: [41],
  selectedSrcIndices: [42],
  indices: [43],
  explicitSrcIndices: [44],
  requestedExplicitSrcIndexCount: 99,
  configuredLimit: 0,
  effectiveLimit: 0,
  selectionTruncated: true
});
assert.deepEqual(rawAliasPrioritySelection.explicitSrcIndices, [41]);
assert.equal(rawAliasPrioritySelection.requestedExplicitSrcIndexCount, 1);
assert.equal(rawAliasPrioritySelection.configuredLimit, null);
assert.equal(rawAliasPrioritySelection.effectiveLimit, 8);
assert.equal(rawAliasPrioritySelection.selectionTruncated, false);
assert.deepEqual(
  normalizeWebGpuDiagnosticDetailSelection({
    mode: 'explicit-src-indices',
    srcIndices: [51, -1, Number.NaN, 51, 52]
  }).explicitSrcIndices,
  [51, 52]
);

const impl8RuntimeResult = buildRuntimeResult(65536);
const impl8LineageRecords = impl8SrcIndices.map(buildLineageRecord);
impl8RuntimeResult.step114PreCullDirectEvidence.records = impl8LineageRecords;
impl8RuntimeResult.webgpuPreCullDirectGaussianEvidence.records =
  impl8LineageRecords;
const impl8Bundle = buildWebGpuVisibleRecordDiagnosticArtifactBundle({
  runtimeResult: impl8RuntimeResult,
  detailSelection: renormalizedImpl8Selection,
  artifactSetIdentity: 'phase3-step119-investigation6-fix1'
});
assert.equal(
  impl8Bundle.detailedLineageArtifact.selection
    .requestedExplicitSrcIndexCount,
  8
);
assert.equal(
  impl8Bundle.detailedLineageArtifact.selection.selectedRecordCount,
  8
);
assert.equal(impl8Bundle.detailedLineageArtifact.recordCount, 8);
assert.deepEqual(
  impl8Bundle.detailedLineageArtifact.selection.explicitSrcIndices,
  impl8SrcIndices
);
assert.deepEqual(
  impl8Bundle.detailedLineageArtifact.selection.selectedSrcIndices,
  impl8SrcIndices
);
assert.deepEqual(
  impl8Bundle.detailedLineageArtifact.records.map((record) => record.srcIndex),
  impl8SrcIndices
);
assert.deepEqual(
  impl8Bundle.detailedLineageArtifact.productionDiagnosticSeparation,
  {
    productionOutputDependsOnDetailedLineage: false,
    diagnosticReadbackIsProductionDependency: false
  }
);

const serializedSizes = [];
for (const recordCount of [8, 4096, 65536]) {
  const bundle = buildWebGpuVisibleRecordDiagnosticArtifactBundle({
    runtimeResult: buildRuntimeResult(recordCount),
    detailSelection,
    artifactSetIdentity: `synthetic-${recordCount}`
  });
  const canonical = bundle.canonicalDiagnosticResult;
  const detail = bundle.detailedLineageArtifact;
  assert.equal(
    canonical.schemaVersion,
    WEBGPU_VISIBLE_RECORD_DIAGNOSTIC_SCHEMA_VERSION
  );
  assert.equal(canonical.cardinality.computedRecordCount, recordCount);
  assert.equal(canonical.cardinality.serializedDetailedLineageRecordCount, 0);
  assert.equal(canonical.comparison.firstMismatches.length, 16);
  assert.equal(
    canonical.diagnosticStageAggregates.tileCountsWebGpuComparison.anyMismatch,
    false
  );
  assert.equal(detail.schemaVersion, WEBGPU_VISIBLE_RECORD_LINEAGE_SCHEMA_VERSION);
  assert.equal(detail.records.length, 8);
  assert.deepEqual(
    detail.selection.selectedSrcIndices,
    Array.from({ length: 8 }, (_, index) => index)
  );
  const canonicalJson = JSON.stringify(canonical);
  assert(!canonicalJson.includes('step114PreCullDirectEvidence'));
  assert(!canonicalJson.includes('webgpuPreCullDirectGaussianEvidence'));
  assert(!canonicalJson.includes('fullMirror'));
  assert(!canonicalJson.includes('tileIndices'));
  serializedSizes.push(canonicalJson.length);
}

assert(
  Math.max(...serializedSizes) - Math.min(...serializedSizes) < 256,
  `canonical JSON grew with compute count: ${serializedSizes.join(',')}`
);

const hardBoundSelection = normalizeWebGpuDiagnosticDetailSelection({
  mode: 'explicit-src-indices',
  srcIndices: Array.from({ length: 100 }, (_, index) => index),
  limit: 1000
});
assert.equal(hardBoundSelection.effectiveLimit, WEBGPU_DIAGNOSTIC_DETAIL_HARD_LIMIT);
assert.equal(
  hardBoundSelection.explicitSrcIndices.length,
  WEBGPU_DIAGNOSTIC_DETAIL_HARD_LIMIT
);
assert.equal(hardBoundSelection.selectionTruncated, true);
assert.deepEqual(
  normalizeWebGpuDiagnosticDetailSelection(hardBoundSelection),
  hardBoundSelection
);

const resolved = resolveWebGpuDiagnosticDetailRows({
  candidateIndices: Uint32Array.from([99, 10, 20, 30]),
  selection: {
    mode: 'explicit-and-first-mismatch',
    srcIndices: [20],
    limit: 2
  },
  firstMismatches: [{ row: 3 }]
});
assert.deepEqual(resolved.rows, [2, 3]);
assert.deepEqual(resolved.selectedSrcIndices, [20, 30]);

console.log('WebGPU diagnostic artifact contract smoke tests passed');
