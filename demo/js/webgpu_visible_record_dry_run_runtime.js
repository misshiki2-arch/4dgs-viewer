import { buildVisibleItemForCandidate } from './gpu_visible_item_builder.js';
import { clampInt } from './gpu_tile_utils.js';
import { computeGaussianState } from './rot4d_math.js';
import {
  createWebGpuAabbContract,
  createWebGpuTileRangeContract
} from './common_4dgs_bounds_contracts.js';
import {
  createWebGpuTileListCapacityContract,
  createWebGpuTileListContract,
  createWebGpuTileListValidationUnitContract,
  createWebGpuTileListValidationContract,
  createWebGpuTileCountsOffsetsComparisonSurfaceContract
} from './common_4dgs_tile_list_contracts.js';
import {
  COMPARISON_CONTRACT_SCHEMA_VERSION,
  DEFAULT_COMPARISON_EPSILON,
  DEFAULT_MAX_MISMATCHES,
  MISMATCH_CLASSIFICATIONS,
  RECORD_COMPARISON_KEYS,
  createComparisonToleranceMetadata,
  createRecordComparisonResult,
  createRecordMismatch
} from './common_4dgs_comparison_contracts.js';
import {
  createWebGpuConicContract,
  createWebGpuCovarianceContract
} from './common_4dgs_conic_contracts.js';
import { buildWebGpuProjectionContract } from './common_4dgs_projection_contracts.js';
import { createWebGpuRadiusContract } from './common_4dgs_radius_contracts.js';
import {
  WEBGPU_VISIBLE_RECORD_COMPUTE_MODE,
  WEBGPU_VISIBLE_RECORD_CPU_MATERIALIZED_FIELDS,
  WEBGPU_VISIBLE_RECORD_DEFERRED_FIELDS,
  WEBGPU_VISIBLE_RECORD_FIELDS,
  WEBGPU_VISIBLE_RECORD_FLOATS,
  WEBGPU_VISIBLE_RECORD_IMPLEMENTED_FIELDS,
  WEBGPU_VISIBLE_RECORD_PHASE_STEP,
  WEBGPU_VISIBLE_RECORD_REFERENCE_ASSISTED_FIELDS,
  WEBGPU_VISIBLE_RECORD_SCAFFOLD_MODE,
  WEBGPU_VISIBLE_RECORD_DRY_RUN_SCHEMA_VERSION,
  WEBGPU_VISIBLE_RECORD_WGSL_COMPUTED_FIELDS,
  cloneWebGpuVisibleRecordFieldComputeModes
} from './common_4dgs_record_contracts.js';
import {
  WEBGPU_INPUT_BUFFER_MODES,
  createWebGpuInputBufferContract,
  createWebGpuInputBufferModes
} from './common_4dgs_webgpu_input_contracts.js';
import { buildWebGpuRenderHandoffStub } from './webgpu_render_handoff_stub.js';
import { buildWebGpuTileCompositeHandoffStub } from './webgpu_tile_composite_handoff_stub.js';
import { buildWebGpuTileCompositeShaderHandoff } from './webgpu_tile_composite_shader_handoff.js';
import { buildWebGpuTileCompositeShaderDryRunComparison } from './webgpu_tile_composite_shader_dry_run.js';
import { buildWebGpuTileCompositeAccumulationDryRunComparison } from './webgpu_tile_composite_accumulation_dry_run.js';
import { buildWebGpuFramebufferFreeTileOutputDryRunComparison } from './webgpu_framebuffer_free_tile_output_dry_run.js';
import { buildWebGpuRenderTargetHandoffDryRunComparison } from './webgpu_render_target_handoff_dry_run.js';
import { buildWebGpuConstrainedDisplayAdapterDryRunComparison } from './webgpu_constrained_display_adapter_dry_run.js';
import { buildWebGpuGuardedFirstDisplayExperiment } from './webgpu_guarded_first_display_experiment.js';
import { buildWebGpuCanvasPresentationAdapterDryRunComparison } from './webgpu_canvas_presentation_adapter_dry_run.js';
import { buildWebGpuExclusiveCanvasHandoffReadiness } from './webgpu_exclusive_canvas_handoff.js';
import { buildWebGpuExclusiveFrameLifecycleSwitch } from './webgpu_exclusive_frame_lifecycle_switch.js';
import { buildWebGpuBackendFramePrototype } from './webgpu_backend_frame_prototype.js';
import { buildWebGpuBackendFrameLifecyclePrototype } from './webgpu_backend_frame_lifecycle_prototype.js';
import { buildWebGpuBackendViewerLoopAdapter } from './webgpu_backend_viewer_loop_adapter.js';
import {
  buildWebGpuBackendViewerLifecycleControlledExecution,
  buildWebGpuBackendViewerLifecycleIntegrationBoundary
} from './webgpu_backend_viewer_lifecycle_integration.js';
import {
  buildUnavailableSchedulerFramePresentationBoundaryContract
} from './common_4dgs_backend_output_contracts.js';

const DEFAULT_MAX_RECORDS = 65536;
const DEFAULT_EPSILON = DEFAULT_COMPARISON_EPSILON;
const RECORD_FLOATS = WEBGPU_VISIBLE_RECORD_FLOATS;
const IMPLEMENTED_FIELDS = WEBGPU_VISIBLE_RECORD_IMPLEMENTED_FIELDS;
const WGSL_COMPUTED_FIELDS = WEBGPU_VISIBLE_RECORD_WGSL_COMPUTED_FIELDS;
const WGSL_REFERENCE_ASSISTED_FIELDS = WEBGPU_VISIBLE_RECORD_REFERENCE_ASSISTED_FIELDS;
const CPU_MATERIALIZED_FIELDS = WEBGPU_VISIBLE_RECORD_CPU_MATERIALIZED_FIELDS;
const DEFERRED_FIELDS = WEBGPU_VISIBLE_RECORD_DEFERRED_FIELDS;

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function toUint32Array(value) {
  if (value instanceof Uint32Array) return value;
  if (Array.isArray(value)) return Uint32Array.from(value);
  return new Uint32Array(0);
}

function toFiniteInteger(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(0, Math.floor(n)) : fallback;
}

function createTileCountsOffsetsUnavailable(reason) {
  return {
    mode: 'cpu-reference-materialized-tile-counts-to-offsets-dry-run',
    status: 'unavailable',
    reason,
    implementedInWgsl: false,
    scatterImplemented: false,
    tileCountsValid: false,
    prefixOffsetsValid: false,
    totalTileRefsConsistent: false,
    capacityStatus: 'needs-resize-or-second-pass',
    firstValidationFailures: [{ stage: 'input', reason }]
  };
}

function createTileCountsOffsetsSelfComparisonUnavailable(reason) {
  return {
    mode: 'cpu-reference-self-comparison-surface',
    status: 'unavailable',
    reason,
    expectedSource: 'tileCountsToOffsetsDryRun',
    actualSource: 'tileCountsToOffsetsDryRun',
    implementedInWgsl: false,
    webgpuComputed: false,
    scatterCompared: false,
    anyMismatch: true,
    mismatchClassification: 'tileCountsOffsetsSelfComparisonUnavailable',
    tileCountsMismatchCount: null,
    tileOffsetsMismatchCount: null,
    totalTileRefsMismatch: null,
    capacityStatusMismatch: null,
    maxAbsCountDelta: null,
    maxAbsOffsetDelta: null,
    firstMismatches: [{ kind: 'input', reason }],
    sampleTiles: []
  };
}

function createTileCountsWebGpuComparisonUnavailable(reason) {
  return {
    mode: 'webgpu-tile-counts-only-comparison',
    status: 'unavailable',
    reason,
    expectedSource: 'tileCountsToOffsetsDryRun.tileCounts',
    actualSource: 'webgpuTileCountsDryRun.tileCounts',
    implementedInWgsl: false,
    webgpuComputed: false,
    tileOffsetsCompared: false,
    prefixSumImplemented: false,
    scatterCompared: false,
    anyMismatch: true,
    mismatchClassification: 'tileCountsWebGpuComparisonUnavailable',
    tileCountsMismatchCount: null,
    tileOffsetsMismatchCount: null,
    totalTileRefsMismatch: null,
    capacityStatusMismatch: null,
    maxAbsCountDelta: null,
    maxAbsOffsetDelta: null,
    firstMismatches: [{ kind: 'input', reason }],
    sampleTiles: []
  };
}

function createWebGpuTileCountsUnavailable(reason) {
  return {
    mode: 'webgpu-compute-tile-counts-only',
    status: 'unavailable',
    reason,
    implementedInWgsl: true,
    tileOffsetsComputed: false,
    prefixSumImplemented: false,
    scatterImplemented: false,
    tileCounts: []
  };
}

function createTileOffsetsFromWebGpuCountsUnavailable(reason) {
  return {
    mode: 'cpu-prefix-from-webgpu-tile-counts-dry-run',
    status: 'unavailable',
    reason,
    source: 'webgpuTileCountsDryRun.tileCounts',
    implementedInWgsl: false,
    webgpuPrefixComputed: false,
    scatterImplemented: false,
    tileOffsets: []
  };
}

function createTileOffsetsPrefixComparisonUnavailable(reason) {
  return {
    mode: 'tile-offsets-prefix-comparison-from-webgpu-counts',
    status: 'unavailable',
    reason,
    expectedSource: 'tileCountsToOffsetsDryRun.tileOffsets',
    actualSource: 'tileOffsetsFromWebGpuCountsDryRun.tileOffsets',
    implementedInWgsl: false,
    webgpuComputed: false,
    webgpuPrefixComputed: false,
    scatterCompared: false,
    anyMismatch: true,
    mismatchClassification: 'tileOffsetsPrefixComparisonUnavailable',
    tileCountsMismatchCount: null,
    tileOffsetsMismatchCount: null,
    totalTileRefsMismatch: null,
    capacityStatusMismatch: null,
    maxAbsCountDelta: null,
    maxAbsOffsetDelta: null,
    prefixOffsetsValid: false,
    totalTileRefsConsistent: false,
    firstMismatches: [{ kind: 'input', reason }],
    sampleTiles: []
  };
}

function createWebGpuTileOffsetsPrefixUnavailable(reason) {
  return {
    mode: 'webgpu-compute-tile-offsets-prefix-sum',
    status: 'unavailable',
    reason,
    source: 'webgpuTileCountsDryRun.tileCounts',
    implementedInWgsl: true,
    webgpuPrefixComputed: true,
    scatterImplemented: false,
    tileOffsets: []
  };
}

function createTileOffsetsWebGpuPrefixComparisonUnavailable(reason) {
  return {
    mode: 'webgpu-tile-offsets-prefix-sum-comparison',
    status: 'unavailable',
    reason,
    expectedSource: 'tileCountsToOffsetsDryRun.tileOffsets',
    actualSource: 'webgpuTileOffsetsPrefixDryRun.tileOffsets',
    implementedInWgsl: true,
    webgpuComputed: true,
    webgpuPrefixComputed: true,
    scatterCompared: false,
    anyMismatch: true,
    mismatchClassification: 'tileOffsetsWebGpuPrefixComparisonUnavailable',
    tileCountsMismatchCount: null,
    tileOffsetsMismatchCount: null,
    totalTileRefsMismatch: null,
    capacityStatusMismatch: null,
    maxAbsCountDelta: null,
    maxAbsOffsetDelta: null,
    prefixOffsetsValid: false,
    totalTileRefsConsistent: false,
    firstMismatches: [{ kind: 'input', reason }],
    sampleTiles: []
  };
}

function createScatterValidationBoundaryUnavailable(reason) {
  return {
    mode: 'scatter-write-cursor-capacity-validation-boundary',
    status: 'unavailable',
    reason,
    source: 'tileRange + tileCountsToOffsetsDryRun.tileOffsets',
    implementedInWgsl: false,
    webgpuScatterComputed: false,
    tileIndicesMaterialized: false,
    scatterCompared: false,
    writeCursorInitialValid: false,
    writeCursorFinalValid: false,
    scatterOutputValid: false,
    capacityStatus: 'needs-resize-or-second-pass',
    firstValidationFailures: [{ stage: 'input', reason }],
    sampleTiles: []
  };
}

function createTileIndicesSelfComparisonUnavailable(reason) {
  return {
    mode: 'cpu-reference-tile-indices-self-comparison-surface',
    status: 'unavailable',
    reason,
    expectedSource: 'cpu-reference-tileIndices',
    actualSource: 'cpu-reference-tileIndices',
    implementedInWgsl: false,
    webgpuScatterComputed: false,
    tileIndicesMaterialized: false,
    scatterCompared: false,
    anyMismatch: true,
    mismatchClassification: 'tileIndicesSelfComparisonUnavailable',
    tileIndicesMismatchCount: null,
    orderingMismatchCount: null,
    capacityStatusMismatch: null,
    maxAbsIndexDelta: null,
    firstMismatches: [{ kind: 'input', reason }],
    sampleTiles: []
  };
}

function createTileIndicesWebGpuScatterComparisonUnavailable(reason) {
  return {
    mode: 'webgpu-tile-indices-scatter-comparison',
    status: 'unavailable',
    reason,
    expectedSource: 'cpu-reference-tileIndices',
    actualSource: 'webgpu-scatter-readback-tileIndices',
    implementedInWgsl: true,
    webgpuScatterComputed: true,
    tileIndicesMaterialized: true,
    tileIndicesStoredInJson: false,
    scatterCompared: true,
    anyMismatch: true,
    mismatchClassification: 'tileIndicesWebGpuScatterComparisonUnavailable',
    tileIndicesMismatchCount: null,
    orderingMismatchCount: null,
    capacityStatusMismatch: null,
    maxAbsIndexDelta: null,
    firstMismatches: [{ kind: 'input', reason }],
    sampleTiles: []
  };
}

function createTileListSummaryComparisonUnavailable(reason) {
  return {
    mode: 'non-display-webgpu-tile-list-summary-comparison',
    status: 'unavailable',
    reason,
    source:
      'webgpuTileCountsDryRun + webgpuTileOffsetsPrefixDryRun + tileIndicesWebGpuScatterComparison',
    implementedInWgsl: false,
    nonDisplayOnly: true,
    fullTileListGeneration: false,
    sortImplemented: false,
    displayConnectionImplemented: false,
    tileIndicesStoredInJson: false,
    anyMismatch: true,
    mismatchClassification: 'tileListSummaryComparisonUnavailable',
    firstMismatches: [{ kind: 'input', reason }],
    sampleTiles: []
  };
}

function createWebGpuTileListBackendOutputUnavailable(reason) {
  return {
    mode: 'non-display-webgpu-tile-list-backend-output',
    status: 'unavailable',
    reason,
    source: 'tileListSummaryComparison',
    backendOutputReady: false,
    nonDisplayOnly: true,
    fullTileListGeneration: false,
    sortImplemented: false,
    displayConnectionImplemented: false,
    tileIndicesStoredInJson: false,
    handoffReadiness: {
      status: 'blocked',
      displayConnectionAllowed: false,
      satisfied: [],
      unresolved: ['tileListSummaryComparison'],
      blocked: [{ stage: 'input', reason }]
    }
  };
}

function createRenderPayloadSortReadinessUnavailable(reason) {
  return {
    mode: 'render-payload-and-sort-readiness-summary',
    status: 'unavailable',
    reason,
    source: 'webgpuTileListBackendOutput',
    backendOutputReady: false,
    displayConnectionAllowed: false,
    sortImplemented: false,
    compactionImplemented: false,
    renderPayloadGpuImplemented: false,
    tileCompositeImplemented: false,
    firstValidationFailures: [{ stage: 'input', reason }]
  };
}

function createDepthSortComparisonUnavailable(reason) {
  return {
    mode: 'cpu-staged-webgpu-tile-list-depth-sort-comparison',
    status: 'unavailable',
    reason,
    source: 'webgpuTileOffsetsPrefixDryRun + tileIndicesWebGpuScatterComparison + webgpu fixed-record depth',
    implementedInWgsl: false,
    webgpuSortComputed: false,
    cpuStagedSortComputed: false,
    nonDisplayOnly: true,
    displayConnectionAllowed: false,
    tileCompositeImplemented: false,
    sortedIndicesStoredInJson: false,
    anyMismatch: true,
    mismatchClassification: 'depthSortComparisonUnavailable',
    sortMismatchCount: null,
    exactSortDifferenceCount: null,
    nearTieSortDifferenceCount: null,
    orderingMismatchCount: null,
    depthKeyMismatchCount: null,
    firstMismatches: [{ kind: 'input', reason }],
    firstSortDifferences: [],
    sampleTiles: []
  };
}

function createWebGpuRenderHandoffStubUnavailable(reason) {
  return {
    mode: 'webgpu-render-handoff-stub-partial-payload',
    status: 'unavailable',
    reason,
    source: 'webgpu fixed-record output + webgpuTileListBackendOutput',
    backendOutputReady: false,
    depthSortReady: false,
    renderHandoffStubReady: false,
    displayConnectionAllowed: false,
    tileCompositeImplemented: false,
    renderPayloadGpuImplemented: false,
    partialPayloadMaterialized: false,
    referenceAssistedPayloadFields: [],
    payloadStoredInJson: false,
    firstValidationFailures: [{ stage: 'input', reason }],
    sampleRecords: []
  };
}

function createWebGpuTileCompositeHandoffStubUnavailable(reason) {
  return {
    mode: 'webgpu-tile-composite-handoff-stub',
    status: 'unavailable',
    reason,
    source: 'webgpuRenderHandoffStub + webgpu tile-list output + depthSortComparison',
    nonDisplayOnly: true,
    tileCompositeHandoffStubReady: false,
    tileCompositeImplemented: false,
    framebufferImplemented: false,
    displayConnectionAllowed: false,
    firstValidationFailures: [{ stage: 'input', reason }],
    sampleTiles: []
  };
}

function createWebGpuTileCompositeShaderHandoffUnavailable(reason) {
  return {
    mode: 'webgpu-tile-composite-shader-handoff-non-display',
    status: 'unavailable',
    reason,
    source: 'webgpuTileCompositeHandoffStub',
    nonDisplayOnly: true,
    tileCompositeShaderHandoffImplemented: true,
    tileCompositeShaderHandoffReady: false,
    tileCompositeShaderImplemented: false,
    tileCompositeImplemented: false,
    framebufferImplemented: false,
    displayConnectionAllowed: false,
    orderedTileIndicesStoredInJson: false,
    firstValidationFailures: [{ stage: 'input', reason }],
    sampleTiles: []
  };
}

function createWebGpuTileCompositeShaderDryRunComparisonUnavailable(reason) {
  return {
    mode: 'webgpu-tile-composite-shader-dry-run-comparison',
    status: 'unavailable',
    reason,
    source: 'webgpuTileCompositeShaderHandoff',
    nonDisplayOnly: true,
    implementedInWgsl: true,
    tileCompositeShaderComputed: false,
    tileCompositeImplemented: false,
    framebufferImplemented: false,
    displayConnectionAllowed: false,
    anyMismatch: true,
    mismatchClassification: 'tileCompositeShaderDryRunUnavailable',
    firstMismatches: [{ kind: 'input', reason }],
    sampleEvaluations: []
  };
}

function createWebGpuTileCompositeAccumulationDryRunComparisonUnavailable(reason) {
  return {
    mode: 'webgpu-tile-composite-accumulation-dry-run-comparison',
    status: 'unavailable',
    reason,
    source: 'webgpuTileCompositeShaderHandoff + webgpuRenderHandoffStub',
    nonDisplayOnly: true,
    implementedInWgsl: true,
    tileCompositeAccumulationComputed: false,
    tileCompositeImplemented: false,
    framebufferImplemented: false,
    displayConnectionAllowed: false,
    anyMismatch: true,
    mismatchClassification: 'tileCompositeAccumulationDryRunUnavailable',
    firstMismatches: [{ kind: 'input', reason }],
    sampleTileAccumulations: []
  };
}

function createWebGpuFramebufferFreeTileOutputDryRunComparisonUnavailable(reason) {
  return {
    mode: 'webgpu-framebuffer-free-tile-output-dry-run-comparison',
    status: 'unavailable',
    reason,
    source: 'webgpuTileCompositeAccumulationDryRunComparison',
    nonDisplayOnly: true,
    implementedInWgsl: true,
    tileOutputPackingComputed: false,
    framebufferFreeOutputComputed: false,
    productionTileCompositeImplemented: false,
    framebufferImplemented: false,
    displayConnectionAllowed: false,
    anyMismatch: true,
    mismatchClassification: 'framebufferFreeTileOutputDryRunUnavailable',
    firstMismatches: [{ kind: 'input', reason }],
    sampleTileOutputs: []
  };
}

function createWebGpuRenderTargetHandoffDryRunComparisonUnavailable(reason) {
  return {
    mode: 'webgpu-render-target-handoff-dry-run-comparison',
    status: 'unavailable',
    reason,
    source: 'webgpuFramebufferFreeTileOutputDryRunComparison',
    nonDisplayOnly: true,
    implementedInWgsl: true,
    renderTargetSamplePackingComputed: false,
    renderTargetHandoffReady: false,
    productionTileCompositeImplemented: false,
    framebufferImplemented: false,
    displayConnectionImplemented: false,
    displayConnectionAllowed: false,
    anyMismatch: true,
    mismatchClassification: 'renderTargetHandoffDryRunUnavailable',
    firstMismatches: [{ kind: 'input', reason }],
    sampleRenderTargetPixels: []
  };
}

function createWebGpuConstrainedDisplayAdapterDryRunComparisonUnavailable(reason) {
  return {
    mode: 'webgpu-constrained-display-adapter-dry-run-comparison',
    status: 'unavailable',
    reason,
    source: 'webgpuRenderTargetHandoffDryRunComparison',
    nonDisplayOnly: true,
    implementedInWgsl: true,
    constrainedDisplayExperiment: true,
    displayAdapterDryRunComputed: false,
    renderTargetTextureWritten: false,
    textureReadbackCompared: false,
    framebufferAdapterImplemented: false,
    framebufferImplemented: false,
    canvasPresentationImplemented: false,
    displayConnectionImplemented: false,
    displayConnectionAllowed: false,
    anyMismatch: true,
    mismatchClassification: 'constrainedDisplayAdapterDryRunUnavailable',
    firstMismatches: [{ kind: 'input', reason }],
    sampleTexturePixels: []
  };
}

function makeTileCountsOffsetsSampleTiles(reference, {
  actualTileCounts = null,
  actualTileOffsets = null
} = {}) {
  const tileCounts = Array.isArray(reference?.tileCounts) ? reference.tileCounts : [];
  const tileOffsets = Array.isArray(reference?.tileOffsets) ? reference.tileOffsets : [];
  const actualCounts = Array.isArray(actualTileCounts) ? actualTileCounts : tileCounts;
  const actualOffsets = Array.isArray(actualTileOffsets) ? actualTileOffsets : tileOffsets;
  const tileCount = toFiniteInteger(reference?.tileGrid?.tileCount, tileCounts.length);
  if (tileCount <= 0) return [];

  const sampleIds = [];
  const addSampleId = (tileId) => {
    if (tileId < 0 || tileId >= tileCount || sampleIds.includes(tileId)) return;
    sampleIds.push(tileId);
  };

  addSampleId(0);
  addSampleId(tileCounts.findIndex((count) => count > 0));
  let maxCountTileId = 0;
  for (let i = 1; i < Math.min(tileCounts.length, tileCount); i += 1) {
    if ((tileCounts[i] ?? 0) > (tileCounts[maxCountTileId] ?? 0)) {
      maxCountTileId = i;
    }
  }
  addSampleId(maxCountTileId);
  addSampleId(tileCount - 1);

  return sampleIds.map((tileId) => ({
    tileId,
    expectedCount: tileCounts[tileId] ?? 0,
    actualCount: actualCounts[tileId] ?? 0,
    countDelta: (actualCounts[tileId] ?? 0) - (tileCounts[tileId] ?? 0),
    expectedOffset: tileOffsets[tileId] ?? null,
    actualOffset: actualOffsets[tileId] ?? null,
    offsetDelta: (actualOffsets[tileId] ?? 0) - (tileOffsets[tileId] ?? 0),
    expectedNextOffset: tileOffsets[tileId + 1] ?? null,
    actualNextOffset: actualOffsets[tileId + 1] ?? null
  }));
}

function buildTileCountsOffsetsSelfComparison(tileCountsToOffsetsDryRun) {
  const startMs = nowMs();
  if (!tileCountsToOffsetsDryRun || tileCountsToOffsetsDryRun.status !== 'ok') {
    const reason = tileCountsToOffsetsDryRun?.reason ??
      tileCountsToOffsetsDryRun?.status ??
      'tile-counts-to-offsets-dry-run-unavailable';
    return createTileCountsOffsetsSelfComparisonUnavailable(reason);
  }

  const expectedCounts = Array.isArray(tileCountsToOffsetsDryRun.tileCounts)
    ? tileCountsToOffsetsDryRun.tileCounts
    : [];
  const actualCounts = expectedCounts;
  const expectedOffsets = Array.isArray(tileCountsToOffsetsDryRun.tileOffsets)
    ? tileCountsToOffsetsDryRun.tileOffsets
    : [];
  const actualOffsets = expectedOffsets;
  const expectedCapacity = tileCountsToOffsetsDryRun.capacity ?? {};
  const actualCapacity = expectedCapacity;
  const tileCount = toFiniteInteger(tileCountsToOffsetsDryRun.tileGrid?.tileCount, expectedCounts.length);
  const firstMismatches = [];
  let tileCountsMismatchCount = 0;
  let tileOffsetsMismatchCount = 0;
  let maxAbsCountDelta = 0;
  let maxAbsOffsetDelta = 0;

  const countLengthMismatch = expectedCounts.length !== actualCounts.length || expectedCounts.length !== tileCount;
  const offsetLengthMismatch =
    expectedOffsets.length !== actualOffsets.length || expectedOffsets.length !== tileCount + 1;
  if (countLengthMismatch) {
    firstMismatches.push({
      kind: 'shapeMismatch',
      field: 'tileCounts',
      expectedLength: expectedCounts.length,
      actualLength: actualCounts.length,
      tileCount
    });
  }
  if (offsetLengthMismatch && firstMismatches.length < 8) {
    firstMismatches.push({
      kind: 'shapeMismatch',
      field: 'tileOffsets',
      expectedLength: expectedOffsets.length,
      actualLength: actualOffsets.length,
      expectedTileCountPlusOne: tileCount + 1
    });
  }

  for (let i = 0; i < Math.min(expectedCounts.length, actualCounts.length, tileCount); i += 1) {
    const expected = expectedCounts[i] ?? 0;
    const actual = actualCounts[i] ?? 0;
    const delta = actual - expected;
    maxAbsCountDelta = Math.max(maxAbsCountDelta, Math.abs(delta));
    if (delta !== 0) {
      tileCountsMismatchCount += 1;
      if (firstMismatches.length < 8) {
        firstMismatches.push({ kind: 'tileCountsMismatch', tileId: i, expected, actual, delta });
      }
    }
  }

  for (let i = 0; i < Math.min(expectedOffsets.length, actualOffsets.length, tileCount + 1); i += 1) {
    const expected = expectedOffsets[i] ?? 0;
    const actual = actualOffsets[i] ?? 0;
    const delta = actual - expected;
    maxAbsOffsetDelta = Math.max(maxAbsOffsetDelta, Math.abs(delta));
    if (delta !== 0) {
      tileOffsetsMismatchCount += 1;
      if (firstMismatches.length < 8) {
        firstMismatches.push({ kind: 'tileOffsetsMismatch', offsetIndex: i, expected, actual, delta });
      }
    }
  }

  const expectedTotalTileRefs = expectedCapacity.totalTileRefs ??
    tileCountsToOffsetsDryRun.metadata?.totalTileRefs ??
    expectedOffsets[tileCount] ??
    null;
  const actualTotalTileRefs = actualCapacity.totalTileRefs ??
    tileCountsToOffsetsDryRun.metadata?.totalTileRefs ??
    actualOffsets[tileCount] ??
    null;
  const totalTileRefsMismatch = expectedTotalTileRefs !== actualTotalTileRefs;
  if (totalTileRefsMismatch && firstMismatches.length < 8) {
    firstMismatches.push({
      kind: 'totalTileRefsMismatch',
      expected: expectedTotalTileRefs,
      actual: actualTotalTileRefs
    });
  }

  const expectedCapacityStatus = expectedCapacity.capacityStatus ??
    tileCountsToOffsetsDryRun.validationSummary?.capacityStatus ??
    null;
  const actualCapacityStatus = actualCapacity.capacityStatus ??
    tileCountsToOffsetsDryRun.validationSummary?.capacityStatus ??
    null;
  const capacityStatusMismatch = expectedCapacityStatus !== actualCapacityStatus;
  if (capacityStatusMismatch && firstMismatches.length < 8) {
    firstMismatches.push({
      kind: 'capacityStatusMismatch',
      expected: expectedCapacityStatus,
      actual: actualCapacityStatus
    });
  }

  const anyMismatch =
    countLengthMismatch ||
    offsetLengthMismatch ||
    tileCountsMismatchCount > 0 ||
    tileOffsetsMismatchCount > 0 ||
    totalTileRefsMismatch ||
    capacityStatusMismatch;

  return {
    mode: 'cpu-reference-self-comparison-surface',
    status: anyMismatch ? 'mismatch' : 'ok',
    expectedSource: 'tileCountsToOffsetsDryRun',
    actualSource: 'tileCountsToOffsetsDryRun',
    implementedInWgsl: false,
    webgpuComputed: false,
    scatterCompared: false,
    anyMismatch,
    mismatchClassification: anyMismatch ? 'cpuReferenceSelfComparisonMismatch' : 'none',
    tileCountsMismatchCount,
    tileOffsetsMismatchCount,
    totalTileRefsMismatch,
    capacityStatusMismatch,
    maxAbsCountDelta,
    maxAbsOffsetDelta,
    firstMismatches,
    sampleTiles: makeTileCountsOffsetsSampleTiles(tileCountsToOffsetsDryRun),
    timing: {
      tileCountsOffsetsSelfComparisonMs: nowMs() - startMs
    }
  };
}

function buildTileCountsWebGpuComparison(tileCountsToOffsetsDryRun, webgpuTileCountsDryRun) {
  const startMs = nowMs();
  if (!tileCountsToOffsetsDryRun || tileCountsToOffsetsDryRun.status !== 'ok') {
    const reason = tileCountsToOffsetsDryRun?.reason ??
      tileCountsToOffsetsDryRun?.status ??
      'tile-counts-to-offsets-dry-run-unavailable';
    return createTileCountsWebGpuComparisonUnavailable(reason);
  }
  if (!webgpuTileCountsDryRun || webgpuTileCountsDryRun.status !== 'ok') {
    const reason = webgpuTileCountsDryRun?.reason ??
      webgpuTileCountsDryRun?.status ??
      'webgpu-tile-counts-dry-run-unavailable';
    return createTileCountsWebGpuComparisonUnavailable(reason);
  }

  const expectedCounts = Array.isArray(tileCountsToOffsetsDryRun.tileCounts)
    ? tileCountsToOffsetsDryRun.tileCounts
    : [];
  const actualCounts = Array.isArray(webgpuTileCountsDryRun.tileCounts)
    ? webgpuTileCountsDryRun.tileCounts
    : [];
  const tileCount = toFiniteInteger(tileCountsToOffsetsDryRun.tileGrid?.tileCount, expectedCounts.length);
  const firstMismatches = [];
  let tileCountsMismatchCount = 0;
  let maxAbsCountDelta = 0;
  let expectedTotalTileRefs = 0;
  let actualTotalTileRefs = 0;

  const shapeMismatch = expectedCounts.length !== tileCount || actualCounts.length !== tileCount;
  if (shapeMismatch) {
    firstMismatches.push({
      kind: 'shapeMismatch',
      field: 'tileCounts',
      expectedLength: expectedCounts.length,
      actualLength: actualCounts.length,
      tileCount
    });
  }

  for (let i = 0; i < Math.min(expectedCounts.length, actualCounts.length, tileCount); i += 1) {
    const expected = expectedCounts[i] ?? 0;
    const actual = actualCounts[i] ?? 0;
    const delta = actual - expected;
    expectedTotalTileRefs += expected;
    actualTotalTileRefs += actual;
    maxAbsCountDelta = Math.max(maxAbsCountDelta, Math.abs(delta));
    if (delta !== 0) {
      tileCountsMismatchCount += 1;
      if (firstMismatches.length < 8) {
        firstMismatches.push({ kind: 'tileCountsMismatch', tileId: i, expected, actual, delta });
      }
    }
  }

  const expectedTerminalTotal = tileCountsToOffsetsDryRun.metadata?.totalTileRefs ??
    tileCountsToOffsetsDryRun.tileOffsets?.[tileCount] ??
    expectedTotalTileRefs;
  const actualReportedTotal = webgpuTileCountsDryRun.metadata?.totalTileRefs ?? actualTotalTileRefs;
  const totalTileRefsMismatch = expectedTerminalTotal !== actualReportedTotal ||
    expectedTotalTileRefs !== actualTotalTileRefs;
  if (totalTileRefsMismatch && firstMismatches.length < 8) {
    firstMismatches.push({
      kind: 'totalTileRefsMismatch',
      expected: expectedTerminalTotal,
      actual: actualReportedTotal
    });
  }

  const expectedCapacityStatus = tileCountsToOffsetsDryRun.capacity?.capacityStatus ??
    tileCountsToOffsetsDryRun.validationSummary?.capacityStatus ??
    null;
  const actualCapacityStatus = webgpuTileCountsDryRun.capacity?.capacityStatus ?? expectedCapacityStatus;
  const capacityStatusMismatch = expectedCapacityStatus !== actualCapacityStatus;
  if (capacityStatusMismatch && firstMismatches.length < 8) {
    firstMismatches.push({
      kind: 'capacityStatusMismatch',
      expected: expectedCapacityStatus,
      actual: actualCapacityStatus
    });
  }

  const anyMismatch =
    shapeMismatch ||
    tileCountsMismatchCount > 0 ||
    totalTileRefsMismatch ||
    capacityStatusMismatch;

  return {
    mode: 'webgpu-tile-counts-only-comparison',
    status: anyMismatch ? 'mismatch' : 'ok',
    expectedSource: 'tileCountsToOffsetsDryRun.tileCounts',
    actualSource: 'webgpuTileCountsDryRun.tileCounts',
    implementedInWgsl: true,
    webgpuComputed: true,
    tileOffsetsCompared: false,
    prefixSumImplemented: false,
    scatterCompared: false,
    anyMismatch,
    mismatchClassification: anyMismatch ? 'tileCountsMismatch' : 'none',
    tileCountsMismatchCount,
    tileOffsetsMismatchCount: null,
    totalTileRefsMismatch,
    capacityStatusMismatch,
    maxAbsCountDelta,
    maxAbsOffsetDelta: null,
    expectedTotalTileRefs,
    actualTotalTileRefs,
    firstMismatches,
    sampleTiles: makeTileCountsOffsetsSampleTiles(tileCountsToOffsetsDryRun, {
      actualTileCounts: actualCounts
    }),
    timing: {
      tileCountsWebGpuComparisonMs: nowMs() - startMs
    }
  };
}

function buildTileOffsetsFromWebGpuCountsDryRun(webgpuTileCountsDryRun, tileGrid) {
  const startMs = nowMs();
  if (!webgpuTileCountsDryRun || webgpuTileCountsDryRun.status !== 'ok') {
    const reason = webgpuTileCountsDryRun?.reason ??
      webgpuTileCountsDryRun?.status ??
      'webgpu-tile-counts-dry-run-unavailable';
    return createTileOffsetsFromWebGpuCountsUnavailable(reason);
  }

  const tileCounts = Array.isArray(webgpuTileCountsDryRun.tileCounts)
    ? webgpuTileCountsDryRun.tileCounts
    : [];
  const tileCols = toFiniteInteger(tileGrid?.tileCols ?? webgpuTileCountsDryRun.tileGrid?.tileCols, 0);
  const tileRows = toFiniteInteger(tileGrid?.tileRows ?? webgpuTileCountsDryRun.tileGrid?.tileRows, 0);
  const tileSize = toFiniteInteger(tileGrid?.tileSize ?? webgpuTileCountsDryRun.tileGrid?.tileSize, 32);
  const tileCount = toFiniteInteger(webgpuTileCountsDryRun.tileGrid?.tileCount, tileCols * tileRows);
  if (tileCount <= 0 || tileCounts.length !== tileCount) {
    return createTileOffsetsFromWebGpuCountsUnavailable('tile-counts-shape-unavailable');
  }

  const tileOffsets = new Uint32Array(tileCount + 1);
  let maxRefsPerTile = 0;
  let nonEmptyTiles = 0;
  let prefixOffsetsValid = tileOffsets[0] === 0;
  const firstValidationFailures = [];
  for (let i = 0; i < tileCount; i += 1) {
    const count = toFiniteInteger(tileCounts[i], 0);
    tileOffsets[i + 1] = tileOffsets[i] + count;
    if (count > 0) nonEmptyTiles += 1;
    if (count > maxRefsPerTile) maxRefsPerTile = count;
    if (tileOffsets[i + 1] < tileOffsets[i] || tileOffsets[i + 1] - tileOffsets[i] !== count) {
      prefixOffsetsValid = false;
      if (firstValidationFailures.length < 8) {
        firstValidationFailures.push({ stage: 'tileOffsets', tileId: i, reason: 'exclusive-prefix-mismatch' });
      }
    }
  }

  const totalTileRefs = tileOffsets[tileCount] ?? 0;
  const expectedTotalTileRefs = webgpuTileCountsDryRun.metadata?.totalTileRefs ?? totalTileRefs;
  const totalTileRefsConsistent = totalTileRefs === expectedTotalTileRefs;
  if (!totalTileRefsConsistent && firstValidationFailures.length < 8) {
    firstValidationFailures.push({ stage: 'totalTileRefs', reason: 'offset-terminal-does-not-match-webgpu-count-sum' });
  }

  return {
    mode: 'cpu-prefix-from-webgpu-tile-counts-dry-run',
    status: firstValidationFailures.length === 0 ? 'ok' : 'validation-failed',
    source: 'webgpuTileCountsDryRun.tileCounts',
    implementedInWgsl: false,
    webgpuPrefixComputed: false,
    scatterImplemented: false,
    outputSchema: {
      tileOffsets: 'uint32[tileCount + 1], exclusive prefix sum of webgpuTileCountsDryRun.tileCounts'
    },
    tileGrid: { tileCols, tileRows, tileCount, tileSize },
    recordCounts: {
      tileCountsLength: tileCounts.length,
      tileOffsetsLength: tileOffsets.length
    },
    metadata: {
      tileOffsetsType: 'uint32',
      tileOffsetsPolicy: 'exclusive-prefix-sum',
      tileOffsetsInitialValue: tileOffsets[0] ?? null,
      tileOffsetsTerminalValue: totalTileRefs,
      totalTileRefs,
      maxRefsPerTile,
      nonEmptyTiles
    },
    capacity: {
      totalTileRefs,
      maxRefsPerTile,
      nonEmptyTiles,
      capacityStatus: 'no-overflow'
    },
    validationSummary: {
      prefixOffsetsValid,
      totalTileRefsConsistent,
      capacityStatus: 'no-overflow',
      firstValidationFailures
    },
    tileOffsets: Array.from(tileOffsets),
    timing: {
      tileOffsetsFromWebGpuCountsDryRunMs: nowMs() - startMs
    }
  };
}

function buildTileOffsetsPrefixComparison(tileCountsToOffsetsDryRun, tileOffsetsFromWebGpuCountsDryRun) {
  const startMs = nowMs();
  if (!tileCountsToOffsetsDryRun || tileCountsToOffsetsDryRun.status !== 'ok') {
    const reason = tileCountsToOffsetsDryRun?.reason ??
      tileCountsToOffsetsDryRun?.status ??
      'tile-counts-to-offsets-dry-run-unavailable';
    return createTileOffsetsPrefixComparisonUnavailable(reason);
  }
  if (!tileOffsetsFromWebGpuCountsDryRun || tileOffsetsFromWebGpuCountsDryRun.status !== 'ok') {
    const reason = tileOffsetsFromWebGpuCountsDryRun?.reason ??
      tileOffsetsFromWebGpuCountsDryRun?.status ??
      'tile-offsets-from-webgpu-counts-dry-run-unavailable';
    return createTileOffsetsPrefixComparisonUnavailable(reason);
  }

  const expectedOffsets = Array.isArray(tileCountsToOffsetsDryRun.tileOffsets)
    ? tileCountsToOffsetsDryRun.tileOffsets
    : [];
  const actualOffsets = Array.isArray(tileOffsetsFromWebGpuCountsDryRun.tileOffsets)
    ? tileOffsetsFromWebGpuCountsDryRun.tileOffsets
    : [];
  const tileCount = toFiniteInteger(tileCountsToOffsetsDryRun.tileGrid?.tileCount, expectedOffsets.length - 1);
  const firstMismatches = [];
  let tileOffsetsMismatchCount = 0;
  let maxAbsOffsetDelta = 0;

  const shapeMismatch = expectedOffsets.length !== tileCount + 1 || actualOffsets.length !== tileCount + 1;
  if (shapeMismatch) {
    firstMismatches.push({
      kind: 'shapeMismatch',
      field: 'tileOffsets',
      expectedLength: expectedOffsets.length,
      actualLength: actualOffsets.length,
      expectedTileCountPlusOne: tileCount + 1
    });
  }

  for (let i = 0; i < Math.min(expectedOffsets.length, actualOffsets.length, tileCount + 1); i += 1) {
    const expected = expectedOffsets[i] ?? 0;
    const actual = actualOffsets[i] ?? 0;
    const delta = actual - expected;
    maxAbsOffsetDelta = Math.max(maxAbsOffsetDelta, Math.abs(delta));
    if (delta !== 0) {
      tileOffsetsMismatchCount += 1;
      if (firstMismatches.length < 8) {
        firstMismatches.push({ kind: 'tileOffsetsMismatch', offsetIndex: i, expected, actual, delta });
      }
    }
  }

  const expectedTotalTileRefs = tileCountsToOffsetsDryRun.metadata?.totalTileRefs ??
    expectedOffsets[tileCount] ??
    null;
  const actualTotalTileRefs = tileOffsetsFromWebGpuCountsDryRun.metadata?.totalTileRefs ??
    actualOffsets[tileCount] ??
    null;
  const totalTileRefsMismatch = expectedTotalTileRefs !== actualTotalTileRefs;
  if (totalTileRefsMismatch && firstMismatches.length < 8) {
    firstMismatches.push({
      kind: 'totalTileRefsMismatch',
      expected: expectedTotalTileRefs,
      actual: actualTotalTileRefs
    });
  }

  const expectedCapacityStatus = tileCountsToOffsetsDryRun.capacity?.capacityStatus ??
    tileCountsToOffsetsDryRun.validationSummary?.capacityStatus ??
    null;
  const actualCapacityStatus = tileOffsetsFromWebGpuCountsDryRun.capacity?.capacityStatus ??
    tileOffsetsFromWebGpuCountsDryRun.validationSummary?.capacityStatus ??
    null;
  const capacityStatusMismatch = expectedCapacityStatus !== actualCapacityStatus;
  if (capacityStatusMismatch && firstMismatches.length < 8) {
    firstMismatches.push({
      kind: 'capacityStatusMismatch',
      expected: expectedCapacityStatus,
      actual: actualCapacityStatus
    });
  }

  const prefixOffsetsValid =
    tileOffsetsFromWebGpuCountsDryRun.validationSummary?.prefixOffsetsValid === true;
  const totalTileRefsConsistent =
    tileOffsetsFromWebGpuCountsDryRun.validationSummary?.totalTileRefsConsistent === true;
  const anyMismatch =
    shapeMismatch ||
    tileOffsetsMismatchCount > 0 ||
    totalTileRefsMismatch ||
    capacityStatusMismatch ||
    !prefixOffsetsValid ||
    !totalTileRefsConsistent;

  return {
    mode: 'tile-offsets-prefix-comparison-from-webgpu-counts',
    status: anyMismatch ? 'mismatch' : 'ok',
    expectedSource: 'tileCountsToOffsetsDryRun.tileOffsets',
    actualSource: 'tileOffsetsFromWebGpuCountsDryRun.tileOffsets',
    implementedInWgsl: false,
    webgpuComputed: false,
    webgpuPrefixComputed: false,
    scatterCompared: false,
    anyMismatch,
    mismatchClassification: anyMismatch ? 'tileOffsetsMismatch' : 'none',
    tileCountsMismatchCount: null,
    tileOffsetsMismatchCount,
    totalTileRefsMismatch,
    capacityStatusMismatch,
    maxAbsCountDelta: null,
    maxAbsOffsetDelta,
    prefixOffsetsValid,
    totalTileRefsConsistent,
    firstMismatches,
    sampleTiles: makeTileCountsOffsetsSampleTiles(tileCountsToOffsetsDryRun, {
      actualTileOffsets: actualOffsets
    }),
    timing: {
      tileOffsetsPrefixComparisonMs: nowMs() - startMs
    }
  };
}

function buildTileOffsetsWebGpuPrefixComparison(tileCountsToOffsetsDryRun, webgpuTileOffsetsPrefixDryRun) {
  const startMs = nowMs();
  if (!tileCountsToOffsetsDryRun || tileCountsToOffsetsDryRun.status !== 'ok') {
    const reason = tileCountsToOffsetsDryRun?.reason ??
      tileCountsToOffsetsDryRun?.status ??
      'tile-counts-to-offsets-dry-run-unavailable';
    return createTileOffsetsWebGpuPrefixComparisonUnavailable(reason);
  }
  if (!webgpuTileOffsetsPrefixDryRun || webgpuTileOffsetsPrefixDryRun.status !== 'ok') {
    const reason = webgpuTileOffsetsPrefixDryRun?.reason ??
      webgpuTileOffsetsPrefixDryRun?.status ??
      'webgpu-tile-offsets-prefix-dry-run-unavailable';
    return createTileOffsetsWebGpuPrefixComparisonUnavailable(reason);
  }

  const expectedOffsets = Array.isArray(tileCountsToOffsetsDryRun.tileOffsets)
    ? tileCountsToOffsetsDryRun.tileOffsets
    : [];
  const actualOffsets = Array.isArray(webgpuTileOffsetsPrefixDryRun.tileOffsets)
    ? webgpuTileOffsetsPrefixDryRun.tileOffsets
    : [];
  const tileCount = toFiniteInteger(tileCountsToOffsetsDryRun.tileGrid?.tileCount, expectedOffsets.length - 1);
  const firstMismatches = [];
  let tileOffsetsMismatchCount = 0;
  let maxAbsOffsetDelta = 0;

  const shapeMismatch = expectedOffsets.length !== tileCount + 1 || actualOffsets.length !== tileCount + 1;
  if (shapeMismatch) {
    firstMismatches.push({
      kind: 'shapeMismatch',
      field: 'tileOffsets',
      expectedLength: expectedOffsets.length,
      actualLength: actualOffsets.length,
      expectedTileCountPlusOne: tileCount + 1
    });
  }

  for (let i = 0; i < Math.min(expectedOffsets.length, actualOffsets.length, tileCount + 1); i += 1) {
    const expected = expectedOffsets[i] ?? 0;
    const actual = actualOffsets[i] ?? 0;
    const delta = actual - expected;
    maxAbsOffsetDelta = Math.max(maxAbsOffsetDelta, Math.abs(delta));
    if (delta !== 0) {
      tileOffsetsMismatchCount += 1;
      if (firstMismatches.length < 8) {
        firstMismatches.push({ kind: 'tileOffsetsMismatch', offsetIndex: i, expected, actual, delta });
      }
    }
  }

  const expectedTotalTileRefs = tileCountsToOffsetsDryRun.metadata?.totalTileRefs ??
    expectedOffsets[tileCount] ??
    null;
  const actualTotalTileRefs = webgpuTileOffsetsPrefixDryRun.metadata?.totalTileRefs ??
    actualOffsets[tileCount] ??
    null;
  const totalTileRefsMismatch = expectedTotalTileRefs !== actualTotalTileRefs;
  if (totalTileRefsMismatch && firstMismatches.length < 8) {
    firstMismatches.push({
      kind: 'totalTileRefsMismatch',
      expected: expectedTotalTileRefs,
      actual: actualTotalTileRefs
    });
  }

  const expectedCapacityStatus = tileCountsToOffsetsDryRun.capacity?.capacityStatus ??
    tileCountsToOffsetsDryRun.validationSummary?.capacityStatus ??
    null;
  const actualCapacityStatus = webgpuTileOffsetsPrefixDryRun.capacity?.capacityStatus ??
    webgpuTileOffsetsPrefixDryRun.validationSummary?.capacityStatus ??
    null;
  const capacityStatusMismatch = expectedCapacityStatus !== actualCapacityStatus;
  if (capacityStatusMismatch && firstMismatches.length < 8) {
    firstMismatches.push({
      kind: 'capacityStatusMismatch',
      expected: expectedCapacityStatus,
      actual: actualCapacityStatus
    });
  }

  const prefixOffsetsValid =
    webgpuTileOffsetsPrefixDryRun.validationSummary?.prefixOffsetsValid === true;
  const totalTileRefsConsistent =
    webgpuTileOffsetsPrefixDryRun.validationSummary?.totalTileRefsConsistent === true;
  const anyMismatch =
    shapeMismatch ||
    tileOffsetsMismatchCount > 0 ||
    totalTileRefsMismatch ||
    capacityStatusMismatch ||
    !prefixOffsetsValid ||
    !totalTileRefsConsistent;

  return {
    mode: 'webgpu-tile-offsets-prefix-sum-comparison',
    status: anyMismatch ? 'mismatch' : 'ok',
    expectedSource: 'tileCountsToOffsetsDryRun.tileOffsets',
    actualSource: 'webgpuTileOffsetsPrefixDryRun.tileOffsets',
    implementedInWgsl: true,
    webgpuComputed: true,
    webgpuPrefixComputed: true,
    scatterCompared: false,
    anyMismatch,
    mismatchClassification: anyMismatch ? 'tileOffsetsMismatch' : 'none',
    tileCountsMismatchCount: null,
    tileOffsetsMismatchCount,
    totalTileRefsMismatch,
    capacityStatusMismatch,
    maxAbsCountDelta: null,
    maxAbsOffsetDelta,
    prefixOffsetsValid,
    totalTileRefsConsistent,
    firstMismatches,
    sampleTiles: makeTileCountsOffsetsSampleTiles(tileCountsToOffsetsDryRun, {
      actualTileOffsets: actualOffsets
    }),
    timing: {
      tileOffsetsWebGpuPrefixComparisonMs: nowMs() - startMs
    }
  };
}

function buildScatterValidationBoundaryDryRun({ tileRanges, tileCountsToOffsetsDryRun, webgpuTileOffsetsPrefixDryRun }) {
  const startMs = nowMs();
  if (!tileCountsToOffsetsDryRun || tileCountsToOffsetsDryRun.status !== 'ok') {
    const reason = tileCountsToOffsetsDryRun?.reason ??
      tileCountsToOffsetsDryRun?.status ??
      'tile-counts-to-offsets-dry-run-unavailable';
    return createScatterValidationBoundaryUnavailable(reason);
  }

  const tileCounts = toUint32Array(tileCountsToOffsetsDryRun.tileCounts);
  const tileOffsets = toUint32Array(tileCountsToOffsetsDryRun.tileOffsets);
  const tileCount = toFiniteInteger(tileCountsToOffsetsDryRun.tileGrid?.tileCount, tileCounts.length);
  if (tileCount <= 0 || tileCounts.length !== tileCount || tileOffsets.length !== tileCount + 1) {
    return createScatterValidationBoundaryUnavailable('tile-counts-or-offsets-shape-unavailable');
  }

  const firstValidationFailures = [];
  const writeCursors = new Uint32Array(tileOffsets);
  const initialCursors = new Uint32Array(tileOffsets);
  let writeCursorInitialValid = tileOffsets[0] === 0;
  for (let tileId = 0; tileId < tileCount; tileId += 1) {
    if (tileOffsets[tileId + 1] < tileOffsets[tileId]) {
      writeCursorInitialValid = false;
      if (firstValidationFailures.length < 8) {
        firstValidationFailures.push({ stage: 'writeCursorInit', tileId, reason: 'offsets-not-monotonic' });
      }
    }
  }

  let totalWrites = 0;
  let capacityOverflowCount = 0;
  const firstScatterWrites = [];
  for (let recordIndex = 0; recordIndex < tileRanges.length; recordIndex += 1) {
    const tr = tileRanges[recordIndex];
    if (!tr || tr.length < 4) {
      if (firstValidationFailures.length < 8) {
        firstValidationFailures.push({ stage: 'scatterInput', recordIndex, reason: 'tileRange-unavailable' });
      }
      continue;
    }
    for (let ty = tr[1]; ty <= tr[3]; ty += 1) {
      const rowBase = ty * (tileCountsToOffsetsDryRun.tileGrid?.tileCols ?? 0);
      for (let tx = tr[0]; tx <= tr[2]; tx += 1) {
        const tileId = rowBase + tx;
        if (tileId < 0 || tileId >= tileCount) {
          if (firstValidationFailures.length < 8) {
            firstValidationFailures.push({ stage: 'scatterRange', recordIndex, tileId, reason: 'tileId-out-of-range' });
          }
          continue;
        }
        const writeIndex = writeCursors[tileId];
        const tileEnd = tileOffsets[tileId + 1];
        if (writeIndex < tileOffsets[tileId] || writeIndex >= tileEnd) {
          capacityOverflowCount += 1;
          if (firstValidationFailures.length < 8) {
            firstValidationFailures.push({
              stage: 'scatterWrite',
              recordIndex,
              tileId,
              writeIndex,
              tileStart: tileOffsets[tileId],
              tileEnd,
              reason: 'write-index-out-of-tile-capacity'
            });
          }
        } else if (firstScatterWrites.length < 8) {
          firstScatterWrites.push({ recordIndex, tileId, writeIndex });
        }
        writeCursors[tileId] += 1;
        totalWrites += 1;
      }
    }
  }

  let writeCursorFinalValid = true;
  let maxRefsPerTile = 0;
  let nonEmptyTiles = 0;
  for (let tileId = 0; tileId < tileCount; tileId += 1) {
    const count = tileCounts[tileId] ?? 0;
    if (count > 0) nonEmptyTiles += 1;
    if (count > maxRefsPerTile) maxRefsPerTile = count;
    if (writeCursors[tileId] !== tileOffsets[tileId + 1]) {
      writeCursorFinalValid = false;
      if (firstValidationFailures.length < 8) {
        firstValidationFailures.push({
          stage: 'writeCursorFinal',
          tileId,
          expected: tileOffsets[tileId + 1],
          actual: writeCursors[tileId],
          reason: 'cursor-does-not-match-next-offset'
        });
      }
    }
  }

  const totalTileRefs = tileOffsets[tileCount] ?? 0;
  const totalTileRefsConsistent = totalWrites === totalTileRefs;
  if (!totalTileRefsConsistent && firstValidationFailures.length < 8) {
    firstValidationFailures.push({
      stage: 'totalTileRefs',
      expected: totalTileRefs,
      actual: totalWrites,
      reason: 'scatter-write-count-does-not-match-terminal-offset'
    });
  }

  const webgpuPrefixTerminal = webgpuTileOffsetsPrefixDryRun?.metadata?.tileOffsetsTerminalValue ?? null;
  const webgpuPrefixMatchesReference = webgpuPrefixTerminal === null || webgpuPrefixTerminal === totalTileRefs;
  if (!webgpuPrefixMatchesReference && firstValidationFailures.length < 8) {
    firstValidationFailures.push({
      stage: 'webgpuPrefixRelation',
      expected: totalTileRefs,
      actual: webgpuPrefixTerminal,
      reason: 'webgpu-prefix-terminal-does-not-match-reference'
    });
  }

  const sampleTiles = makeTileCountsOffsetsSampleTiles(tileCountsToOffsetsDryRun).map((sample) => {
    const tileId = sample.tileId;
    return {
      ...sample,
      initialCursor: initialCursors[tileId] ?? null,
      finalCursor: writeCursors[tileId] ?? null,
      expectedFinalCursor: tileOffsets[tileId + 1] ?? null,
      cursorDelta: (writeCursors[tileId] ?? 0) - (initialCursors[tileId] ?? 0),
      cursorCapacityOk: writeCursors[tileId] === tileOffsets[tileId + 1]
    };
  });
  const scatterOutputValid =
    writeCursorInitialValid &&
    writeCursorFinalValid &&
    totalTileRefsConsistent &&
    capacityOverflowCount === 0 &&
    webgpuPrefixMatchesReference;

  return {
    mode: 'scatter-write-cursor-capacity-validation-boundary',
    status: scatterOutputValid ? 'ok' : 'validation-failed',
    source: 'tileRange + tileCountsToOffsetsDryRun.tileOffsets',
    implementedInWgsl: false,
    webgpuScatterComputed: false,
    tileIndicesMaterialized: false,
    scatterCompared: false,
    outputSchema: {
      writeCursors: 'uint32[tileCount + 1] initialized from tileOffsets; final cursors are validated against tileOffsets[i + 1]',
      tileIndices: 'deferred, not materialized in Step24'
    },
    tileGrid: tileCountsToOffsetsDryRun.tileGrid ?? {},
    recordCounts: {
      tileRangeCount: tileRanges.length,
      tileCountsLength: tileCounts.length,
      tileOffsetsLength: tileOffsets.length,
      tileIndicesLength: totalTileRefs
    },
    writeCursorPolicy: {
      initialSource: 'tileCountsToOffsetsDryRun.tileOffsets',
      incrementPolicy: 'one cursor increment per inclusive tileRange touch',
      finalCursorRule: 'writeCursor[tile] == tileOffsets[tile + 1]',
      perTileCapacityRule: 'tileOffsets[tile] <= writeIndex < tileOffsets[tile + 1]'
    },
    capacity: {
      totalTileRefs,
      maxRefsPerTile,
      nonEmptyTiles,
      capacityOverflowCount,
      capacityStatus: capacityOverflowCount === 0 ? 'no-overflow' : 'overflow-detected'
    },
    validationSummary: {
      writeCursorInitialValid,
      writeCursorFinalValid,
      scatterOutputValid,
      totalTileRefsConsistent,
      capacityStatus: capacityOverflowCount === 0 ? 'no-overflow' : 'overflow-detected',
      webgpuPrefixMatchesReference,
      firstValidationFailures
    },
    firstScatterWrites,
    sampleTiles,
    timing: {
      scatterValidationBoundaryMs: nowMs() - startMs
    }
  };
}

function materializeCpuReferenceTileIndices({ tileRanges, tileCountsToOffsetsDryRun }) {
  const tileOffsets = toUint32Array(tileCountsToOffsetsDryRun.tileOffsets);
  const tileCounts = toUint32Array(tileCountsToOffsetsDryRun.tileCounts);
  const tileCount = toFiniteInteger(tileCountsToOffsetsDryRun.tileGrid?.tileCount, tileCounts.length);
  const tileCols = toFiniteInteger(tileCountsToOffsetsDryRun.tileGrid?.tileCols, 0);
  if (tileCount <= 0 || tileCols <= 0 || tileCounts.length !== tileCount || tileOffsets.length !== tileCount + 1) {
    return {
      status: 'unavailable',
      reason: 'tile-counts-or-offsets-shape-unavailable',
      tileIndices: new Uint32Array(0),
      firstMismatches: []
    };
  }

  const totalTileRefs = tileOffsets[tileCount] ?? 0;
  const tileIndices = new Uint32Array(totalTileRefs);
  const writeCursors = new Uint32Array(tileOffsets);
  const firstMismatches = [];
  let capacityOverflowCount = 0;
  for (let recordIndex = 0; recordIndex < tileRanges.length; recordIndex += 1) {
    const tr = tileRanges[recordIndex];
    if (!tr || tr.length < 4) {
      if (firstMismatches.length < 8) {
        firstMismatches.push({ kind: 'tileRangeUnavailable', recordIndex });
      }
      continue;
    }
    for (let ty = tr[1]; ty <= tr[3]; ty += 1) {
      const rowBase = ty * tileCols;
      for (let tx = tr[0]; tx <= tr[2]; tx += 1) {
        const tileId = rowBase + tx;
        if (tileId < 0 || tileId >= tileCount) {
          if (firstMismatches.length < 8) {
            firstMismatches.push({ kind: 'tileIdOutOfRange', recordIndex, tileId });
          }
          continue;
        }
        const writeIndex = writeCursors[tileId];
        if (writeIndex < tileOffsets[tileId] || writeIndex >= tileOffsets[tileId + 1]) {
          capacityOverflowCount += 1;
          if (firstMismatches.length < 8) {
            firstMismatches.push({
              kind: 'capacityMismatch',
              recordIndex,
              tileId,
              writeIndex,
              tileStart: tileOffsets[tileId],
              tileEnd: tileOffsets[tileId + 1]
            });
          }
        } else {
          tileIndices[writeIndex] = recordIndex;
        }
        writeCursors[tileId] += 1;
      }
    }
  }

  return {
    status: 'ok',
    tileOffsets,
    tileCounts,
    tileCount,
    tileCols,
    totalTileRefs,
    tileIndices,
    writeCursors,
    firstMismatches,
    capacityOverflowCount
  };
}

function buildTileIndicesSelfComparison({ tileRanges, tileCountsToOffsetsDryRun, scatterValidationBoundary }) {
  const startMs = nowMs();
  if (!tileCountsToOffsetsDryRun || tileCountsToOffsetsDryRun.status !== 'ok') {
    const reason = tileCountsToOffsetsDryRun?.reason ??
      tileCountsToOffsetsDryRun?.status ??
      'tile-counts-to-offsets-dry-run-unavailable';
    return createTileIndicesSelfComparisonUnavailable(reason);
  }
  if (!scatterValidationBoundary || scatterValidationBoundary.status !== 'ok') {
    const reason = scatterValidationBoundary?.reason ??
      scatterValidationBoundary?.status ??
      'scatter-validation-boundary-unavailable';
    return createTileIndicesSelfComparisonUnavailable(reason);
  }

  const reference = materializeCpuReferenceTileIndices({ tileRanges, tileCountsToOffsetsDryRun });
  if (reference.status !== 'ok') {
    return createTileIndicesSelfComparisonUnavailable(reference.reason);
  }
  const {
    tileOffsets,
    tileCount,
    totalTileRefs,
    tileIndices,
    firstMismatches,
    capacityOverflowCount
  } = reference;

  let tileIndicesMismatchCount = 0;
  let orderingMismatchCount = 0;
  let maxAbsIndexDelta = 0;
  for (let tileId = 0; tileId < tileCount; tileId += 1) {
    const start = tileOffsets[tileId];
    const end = tileOffsets[tileId + 1];
    let previous = null;
    for (let index = start; index < end; index += 1) {
      const expected = tileIndices[index];
      const actual = tileIndices[index];
      const delta = actual - expected;
      maxAbsIndexDelta = Math.max(maxAbsIndexDelta, Math.abs(delta));
      if (delta !== 0) {
        tileIndicesMismatchCount += 1;
        if (firstMismatches.length < 8) {
          firstMismatches.push({ kind: 'tileIndicesMismatch', tileId, index, expected, actual, delta });
        }
      }
      if (previous !== null && actual < previous) {
        orderingMismatchCount += 1;
        if (firstMismatches.length < 8) {
          firstMismatches.push({ kind: 'orderingMismatch', tileId, index, previous, actual });
        }
      }
      previous = actual;
    }
  }

  const capacityStatus = capacityOverflowCount === 0 ? 'no-overflow' : 'overflow-detected';
  const capacityStatusMismatch =
    (scatterValidationBoundary.capacity?.capacityStatus ?? 'no-overflow') !== capacityStatus;
  const sampleTiles = makeTileCountsOffsetsSampleTiles(tileCountsToOffsetsDryRun).map((sample) => {
    const tileId = sample.tileId;
    const start = tileOffsets[tileId] ?? 0;
    const end = tileOffsets[tileId + 1] ?? start;
    const sampleLimit = Math.min(end, start + 4);
    return {
      ...sample,
      tileIndexStart: start,
      tileIndexEnd: end,
      tileIndexCount: Math.max(0, end - start),
      firstTileIndices: Array.from(tileIndices.slice(start, sampleLimit))
    };
  });
  const anyMismatch =
    tileIndicesMismatchCount > 0 ||
    orderingMismatchCount > 0 ||
    capacityOverflowCount > 0 ||
    capacityStatusMismatch ||
    firstMismatches.length > 0;

  return {
    mode: 'cpu-reference-tile-indices-self-comparison-surface',
    status: anyMismatch ? 'mismatch' : 'ok',
    expectedSource: 'cpu-reference-tileIndices',
    actualSource: 'cpu-reference-tileIndices',
    implementedInWgsl: false,
    webgpuScatterComputed: false,
    tileIndicesMaterialized: true,
    tileIndicesStoredInJson: false,
    scatterCompared: false,
    anyMismatch,
    mismatchClassification: anyMismatch ? 'tileIndicesSelfComparisonMismatch' : 'none',
    tileIndicesMismatchCount,
    orderingMismatchCount,
    capacityStatusMismatch,
    maxAbsIndexDelta,
    recordCounts: {
      tileRangeCount: tileRanges.length,
      tileIndicesLength: tileIndices.length,
      tileOffsetsLength: tileOffsets.length
    },
    capacity: {
      totalTileRefs,
      capacityOverflowCount,
      capacityStatus
    },
    orderingPolicy: {
      sourceOrder: 'record-index-order',
      perTileOrder: 'preserve incoming record order within each tile',
      orderingValidated: orderingMismatchCount === 0
    },
    firstMismatches,
    sampleTiles,
    timing: {
      tileIndicesSelfComparisonMs: nowMs() - startMs
    }
  };
}

function buildTileCountsOffsetsDryRun({ tileRanges, tileGrid }) {
  const startMs = nowMs();
  const tileCols = toFiniteInteger(tileGrid?.tileCols, 0);
  const tileRows = toFiniteInteger(tileGrid?.tileRows, 0);
  const tileSize = toFiniteInteger(tileGrid?.tileSize, 32);
  const tileCount = tileCols * tileRows;
  if (tileCount <= 0) {
    return createTileCountsOffsetsUnavailable('tile-grid-unavailable');
  }

  const countsStartMs = nowMs();
  const tileCounts = new Uint32Array(tileCount);
  const firstValidationFailures = [];
  for (let i = 0; i < tileRanges.length; i += 1) {
    const tr = tileRanges[i];
    if (!tr || tr.length < 4) {
      if (firstValidationFailures.length < 8) {
        firstValidationFailures.push({ stage: 'tileCounts', index: i, reason: 'tileRange-unavailable' });
      }
      continue;
    }
    for (let ty = tr[1]; ty <= tr[3]; ty += 1) {
      const rowBase = ty * tileCols;
      for (let tx = tr[0]; tx <= tr[2]; tx += 1) {
        const tileId = rowBase + tx;
        if (tileId < 0 || tileId >= tileCount) {
          if (firstValidationFailures.length < 8) {
            firstValidationFailures.push({ stage: 'tileCounts', index: i, tileId, reason: 'tileId-out-of-range' });
          }
          continue;
        }
        tileCounts[tileId] += 1;
      }
    }
  }
  const tileCountsBuildMs = nowMs() - countsStartMs;

  const offsetsStartMs = nowMs();
  const tileOffsets = new Uint32Array(tileCount + 1);
  for (let i = 0; i < tileCount; i += 1) {
    tileOffsets[i + 1] = tileOffsets[i] + tileCounts[i];
  }
  const tileOffsetsBuildMs = nowMs() - offsetsStartMs;

  let sumTileCounts = 0;
  let maxRefsPerTile = 0;
  let nonEmptyTiles = 0;
  let tileCountsValid = firstValidationFailures.length === 0 && tileCounts.length === tileCount;
  let prefixOffsetsValid = tileOffsets.length === tileCount + 1 && tileOffsets[0] === 0;
  for (let i = 0; i < tileCount; i += 1) {
    const count = tileCounts[i];
    sumTileCounts += count;
    if (count > 0) nonEmptyTiles += 1;
    if (count > maxRefsPerTile) maxRefsPerTile = count;
    if (tileOffsets[i + 1] < tileOffsets[i] || tileOffsets[i + 1] - tileOffsets[i] !== count) {
      prefixOffsetsValid = false;
      if (firstValidationFailures.length < 8) {
        firstValidationFailures.push({ stage: 'tileOffsets', tileId: i, reason: 'exclusive-prefix-mismatch' });
      }
    }
  }

  const totalTileRefs = tileOffsets[tileCount] ?? 0;
  const totalTileRefsConsistent = totalTileRefs === sumTileCounts;
  if (!totalTileRefsConsistent && firstValidationFailures.length < 8) {
    firstValidationFailures.push({ stage: 'totalTileRefs', reason: 'offset-terminal-does-not-match-count-sum' });
  }

  return {
    mode: 'cpu-reference-materialized-tile-counts-to-offsets-dry-run',
    status: firstValidationFailures.length === 0 ? 'ok' : 'validation-failed',
    computeMode: 'cpu-reference-materialized',
    implementedInWgsl: false,
    scatterImplemented: false,
    source: 'cpu-reference-visible-record-tileRange',
    outputSchema: {
      tileCounts: 'uint32[tileCount]',
      tileOffsets: 'uint32[tileCount + 1], exclusive prefix sum of tileCounts'
    },
    tileGrid: { tileCols, tileRows, tileCount, tileSize },
    recordCounts: {
      tileRangeCount: tileRanges.length,
      tileCountsLength: tileCounts.length,
      tileOffsetsLength: tileOffsets.length
    },
    metadata: {
      tileCountsType: 'uint32',
      tileOffsetsType: 'uint32',
      tileOffsetsPolicy: 'exclusive-prefix-sum',
      tileOffsetsInitialValue: tileOffsets[0] ?? null,
      tileOffsetsTerminalValue: totalTileRefs,
      maxRefsPerTile,
      nonEmptyTiles,
      totalTileRefs
    },
    capacity: {
      maxTileRefs: totalTileRefs,
      maxRefsPerTile,
      totalTileRefs,
      nonEmptyTiles,
      capacityStatus: 'no-overflow'
    },
    validationSummary: {
      tileCountsValid,
      prefixOffsetsValid,
      totalTileRefsConsistent,
      capacityStatus: 'no-overflow',
      firstValidationFailures
    },
    tileCounts: Array.from(tileCounts),
    tileOffsets: Array.from(tileOffsets),
    timing: {
      tileCountsBuildMs,
      tileOffsetsBuildMs,
      tileCountsToOffsetsDryRunMs: nowMs() - startMs
    }
  };
}

function makeFallback(reason, extra = {}) {
  const radiusContract = extra.radiusContract ?? createWebGpuRadiusContract();
  const covarianceContract = extra.covarianceContract ?? createWebGpuCovarianceContract();
  const conicContract = extra.conicContract ?? createWebGpuConicContract();
  const aabbContract = extra.aabbContract ?? createWebGpuAabbContract();
  const tileRangeContract = extra.tileRangeContract ?? createWebGpuTileRangeContract();
  const tileListContract = extra.tileListContract ?? createWebGpuTileListContract();
  const tileListCapacityContract =
    extra.tileListCapacityContract ?? createWebGpuTileListCapacityContract();
  const tileListValidationContract =
    extra.tileListValidationContract ?? createWebGpuTileListValidationContract();
  const tileListValidationUnitContract =
    extra.tileListValidationUnitContract ?? createWebGpuTileListValidationUnitContract();
  const tileCountsOffsetsComparisonSurfaceContract =
    extra.tileCountsOffsetsComparisonSurfaceContract ??
    createWebGpuTileCountsOffsetsComparisonSurfaceContract();
  const tileCountsToOffsetsDryRun =
    extra.tileCountsToOffsetsDryRun ?? createTileCountsOffsetsUnavailable(reason);
  const tileCountsOffsetsSelfComparison =
    extra.tileCountsOffsetsSelfComparison ??
    createTileCountsOffsetsSelfComparisonUnavailable(reason);
  const webgpuTileCountsDryRun =
    extra.webgpuTileCountsDryRun ?? createWebGpuTileCountsUnavailable(reason);
  const tileCountsWebGpuComparison =
    extra.tileCountsWebGpuComparison ?? createTileCountsWebGpuComparisonUnavailable(reason);
  const tileOffsetsFromWebGpuCountsDryRun =
    extra.tileOffsetsFromWebGpuCountsDryRun ??
    createTileOffsetsFromWebGpuCountsUnavailable(reason);
  const tileOffsetsPrefixComparison =
    extra.tileOffsetsPrefixComparison ?? createTileOffsetsPrefixComparisonUnavailable(reason);
  const webgpuTileOffsetsPrefixDryRun =
    extra.webgpuTileOffsetsPrefixDryRun ?? createWebGpuTileOffsetsPrefixUnavailable(reason);
  const tileOffsetsWebGpuPrefixComparison =
    extra.tileOffsetsWebGpuPrefixComparison ??
    createTileOffsetsWebGpuPrefixComparisonUnavailable(reason);
  const scatterValidationBoundary =
    extra.scatterValidationBoundary ?? createScatterValidationBoundaryUnavailable(reason);
  const tileIndicesSelfComparison =
    extra.tileIndicesSelfComparison ?? createTileIndicesSelfComparisonUnavailable(reason);
  const tileIndicesWebGpuScatterComparison =
    extra.tileIndicesWebGpuScatterComparison ??
    createTileIndicesWebGpuScatterComparisonUnavailable(reason);
  const tileListSummaryComparison =
    extra.tileListSummaryComparison ?? createTileListSummaryComparisonUnavailable(reason);
  const webgpuTileListBackendOutput =
    extra.webgpuTileListBackendOutput ?? createWebGpuTileListBackendOutputUnavailable(reason);
  const renderPayloadSortReadiness =
    extra.renderPayloadSortReadiness ?? createRenderPayloadSortReadinessUnavailable(reason);
  const depthSortComparison =
    extra.depthSortComparison ?? createDepthSortComparisonUnavailable(reason);
  const webgpuRenderHandoffStub =
    extra.webgpuRenderHandoffStub ?? createWebGpuRenderHandoffStubUnavailable(reason);
  const webgpuTileCompositeHandoffStub =
    extra.webgpuTileCompositeHandoffStub ??
    createWebGpuTileCompositeHandoffStubUnavailable(reason);
  const webgpuTileCompositeShaderHandoff =
    extra.webgpuTileCompositeShaderHandoff ??
    createWebGpuTileCompositeShaderHandoffUnavailable(reason);
  const webgpuTileCompositeShaderDryRunComparison =
    extra.webgpuTileCompositeShaderDryRunComparison ??
    createWebGpuTileCompositeShaderDryRunComparisonUnavailable(reason);
  const webgpuTileCompositeAccumulationDryRunComparison =
    extra.webgpuTileCompositeAccumulationDryRunComparison ??
    createWebGpuTileCompositeAccumulationDryRunComparisonUnavailable(reason);
  const webgpuFramebufferFreeTileOutputDryRunComparison =
    extra.webgpuFramebufferFreeTileOutputDryRunComparison ??
    createWebGpuFramebufferFreeTileOutputDryRunComparisonUnavailable(reason);
  const webgpuRenderTargetHandoffDryRunComparison =
    extra.webgpuRenderTargetHandoffDryRunComparison ??
    createWebGpuRenderTargetHandoffDryRunComparisonUnavailable(reason);
  const webgpuConstrainedDisplayAdapterDryRunComparison =
    extra.webgpuConstrainedDisplayAdapterDryRunComparison ??
    createWebGpuConstrainedDisplayAdapterDryRunComparisonUnavailable(reason);
  return {
    schemaVersion: WEBGPU_VISIBLE_RECORD_DRY_RUN_SCHEMA_VERSION,
    phaseStep: WEBGPU_VISIBLE_RECORD_PHASE_STEP,
    status: 'fallback',
    reason,
    computeMode: WEBGPU_VISIBLE_RECORD_COMPUTE_MODE,
    scaffoldMode: WEBGPU_VISIBLE_RECORD_SCAFFOLD_MODE,
    implementedFields: IMPLEMENTED_FIELDS,
    wgslComputedFields: WGSL_COMPUTED_FIELDS,
    wgslReferenceAssistedFields: WGSL_REFERENCE_ASSISTED_FIELDS,
    cpuMaterializedFields: CPU_MATERIALIZED_FIELDS,
    fieldComputeModes: cloneWebGpuVisibleRecordFieldComputeModes(),
    deferredFields: DEFERRED_FIELDS,
    candidateCount: extra.candidateCount ?? null,
    recordCount: extra.recordCount ?? null,
    validRecordCount: extra.validRecordCount ?? null,
    recordComparison: {
      [RECORD_COMPARISON_KEYS.ANY_MISMATCH]: true,
      [RECORD_COMPARISON_KEYS.FIELD_MISMATCH_COUNT]: null,
      [RECORD_COMPARISON_KEYS.MAX_ABS_ERROR]: null,
      [RECORD_COMPARISON_KEYS.FIRST_MISMATCHES]: extra.firstMismatches ?? []
    },
    comparisonContract: {
      schemaVersion: COMPARISON_CONTRACT_SCHEMA_VERSION,
      recordComparisonKeys: RECORD_COMPARISON_KEYS,
      mismatchClassifications: MISMATCH_CLASSIFICATIONS
    },
    comparisonTolerance: createComparisonToleranceMetadata(),
    radiusContract,
    radiusComputeMode: radiusContract.computeMode,
    covarianceContract,
    conicContract,
    conicComputeMode: conicContract.computeMode,
    aabbContract,
    tileRangeContract,
    boundsComputeMode: {
      aabb: aabbContract.computeMode,
      tileRange: tileRangeContract.computeMode
    },
    tileListContract,
    tileListComputeMode: tileListContract.computeMode,
    tileListCapacityContract,
    tileListCapacityComputeMode: tileListCapacityContract.computeMode,
    tileListValidationContract,
    tileListValidationComputeMode: tileListValidationContract.computeMode,
    tileListValidationUnitContract,
    tileListValidationUnitComputeMode: tileListValidationUnitContract.computeMode,
    tileCountsOffsetsComparisonSurfaceContract,
    tileCountsOffsetsComparisonSurfaceComputeMode:
      tileCountsOffsetsComparisonSurfaceContract.computeMode,
    tileCountsToOffsetsDryRun,
    tileCountsOffsetsSelfComparison,
    webgpuTileCountsDryRun,
    tileCountsWebGpuComparison,
    tileOffsetsFromWebGpuCountsDryRun,
    tileOffsetsPrefixComparison,
    webgpuTileOffsetsPrefixDryRun,
    tileOffsetsWebGpuPrefixComparison,
    scatterValidationBoundary,
    tileIndicesSelfComparison,
    tileIndicesWebGpuScatterComparison,
    tileListSummaryComparison,
    webgpuTileListBackendOutput,
    renderPayloadSortReadiness,
    depthSortComparison,
    webgpuRenderHandoffStub,
    webgpuTileCompositeHandoffStub,
    webgpuTileCompositeShaderHandoff,
    webgpuTileCompositeShaderDryRunComparison,
    webgpuTileCompositeAccumulationDryRunComparison,
    webgpuFramebufferFreeTileOutputDryRunComparison,
    webgpuRenderTargetHandoffDryRunComparison,
    webgpuConstrainedDisplayAdapterDryRunComparison,
    fieldMismatchCount: null,
    firstMismatches: extra.firstMismatches ?? [],
    mismatchClassification: extra.mismatchClassification ??
      MISMATCH_CLASSIFICATIONS.WEBGPU_VISIBLE_RECORD_UNAVAILABLE,
    timing: extra.timing ?? null,
    inputContract: extra.inputContract ?? null,
    bufferContract: extra.bufferContract ?? null,
    inputBufferModes: extra.inputBufferModes ?? createWebGpuInputBufferModes(),
    webgpu: extra.webgpu ?? null,
    metadata: extra.metadata ?? null
  };
}

function writeRecord(out, row, item, srcIndex) {
  const base = row * RECORD_FLOATS;
  out[base + 0] = srcIndex;
  out[base + 1] = item ? 1 : 0;
  out[base + 2] = item?.px ?? 0;
  out[base + 3] = item?.py ?? 0;
  out[base + 4] = item?.depth ?? 0;
  out[base + 5] = item?.aabb?.[0] ?? 0;
  out[base + 6] = item?.aabb?.[1] ?? 0;
  out[base + 7] = item?.aabb?.[2] ?? 0;
  out[base + 8] = item?.aabb?.[3] ?? 0;
  out[base + 9] = 0;
  out[base + 10] = 0;
  out[base + 11] = 0;
}

function buildCpuReferenceRecords({
  candidateInfo,
  raw,
  camera,
  screenSpaceCamera,
  canvasWidth,
  canvasHeight,
  camPos,
  tileGrid,
  buildConfig,
  maxRecords
}) {
  const startMs = nowMs();
  const candidateIndices = toUint32Array(candidateInfo?.candidateIndices);
  const count = Math.min(candidateIndices.length, toFiniteInteger(maxRecords, DEFAULT_MAX_RECORDS));
  const renderScale = Number.isFinite(buildConfig.renderScale) ? buildConfig.renderScale : 1;
  const renderW = Math.max(1, Math.round(canvasWidth * renderScale));
  const renderH = Math.max(1, Math.round(canvasHeight * renderScale));
  const sx = canvasWidth / renderW;
  const sy = canvasHeight / renderH;
  const flags = {
    nativeRot4d: !!buildConfig.useNativeRot4d,
    nativeMarginal: !!buildConfig.useNativeMarginal
  };
  const records = new Float32Array(count * RECORD_FLOATS);
  const renderPayloadReference = new Float32Array(count * 8);
  const tileRanges = [];
  let validCount = 0;
  for (let i = 0; i < count; i += 1) {
    const srcIndex = candidateIndices[i];
    const item = buildVisibleItemForCandidate({
      raw,
      index: srcIndex,
      camera,
      screenSpaceCamera,
      renderW,
      renderH,
      canvasWidth,
      canvasHeight,
      sx,
      sy,
      timestamp: buildConfig.timestamp,
      scalingModifier: buildConfig.scalingModifier,
      sigmaScale: buildConfig.sigmaScale,
      prefilterVar: buildConfig.prefilterVar,
      useRot4d: buildConfig.useRot4d,
      flags,
      camPos,
      timeDuration: buildConfig.timeDuration,
      useSH: buildConfig.useSH,
      forceSh3d: buildConfig.forceSh3d,
      tileGrid
    });
    if (item) validCount += 1;
    if (item) {
      const recordPx = Math.fround(item.px);
      const recordPy = Math.fround(item.py);
      const recordCoverageRadius = Math.max(1.0, Math.fround(item.radius));
      const recordAabb = [
        clampInt(Math.floor(recordPx - recordCoverageRadius), 0, canvasWidth - 1),
        clampInt(Math.floor(recordPy - recordCoverageRadius), 0, canvasHeight - 1),
        clampInt(Math.ceil(recordPx + recordCoverageRadius), 0, canvasWidth - 1),
        clampInt(Math.ceil(recordPy + recordCoverageRadius), 0, canvasHeight - 1)
      ];
      const tileCols = toFiniteInteger(tileGrid?.tileCols, 0);
      const tileRows = toFiniteInteger(tileGrid?.tileRows, 0);
      const tileSize = toFiniteInteger(tileGrid?.tileSize, 32);
      if (tileCols > 0 && tileRows > 0) {
        tileRanges.push(item.tileRange ?? [
          clampInt(Math.floor(recordAabb[0] / tileSize), 0, tileCols - 1),
          clampInt(Math.floor(recordAabb[1] / tileSize), 0, tileRows - 1),
          clampInt(Math.floor(recordAabb[2] / tileSize), 0, tileCols - 1),
          clampInt(Math.floor(recordAabb[3] / tileSize), 0, tileRows - 1)
        ]);
      }
      writeRecord(records, i, {
        ...item,
        px: recordPx,
        py: recordPy,
        depth: Math.fround(item.depth),
        aabb: recordAabb
      }, srcIndex);
      const payloadRefBase = i * 8;
      renderPayloadReference[payloadRefBase + 0] = Math.fround(item.radius);
      renderPayloadReference[payloadRefBase + 1] = Math.fround(item.conic?.[0] ?? 0);
      renderPayloadReference[payloadRefBase + 2] = Math.fround(item.conic?.[1] ?? 0);
      renderPayloadReference[payloadRefBase + 3] = Math.fround(item.conic?.[2] ?? 0);
      renderPayloadReference[payloadRefBase + 4] = Math.fround(item.colorAlpha?.[3] ?? item.opacity ?? 0);
      renderPayloadReference[payloadRefBase + 5] = Math.fround(item.colorAlpha?.[0] ?? 0);
      renderPayloadReference[payloadRefBase + 6] = Math.fround(item.colorAlpha?.[1] ?? 0);
      renderPayloadReference[payloadRefBase + 7] = Math.fround(item.colorAlpha?.[2] ?? 0);
    } else {
      writeRecord(records, i, null, srcIndex);
    }
  }
  return {
    candidateIndices: candidateIndices.slice(0, count),
    candidateCount: candidateIndices.length,
    count,
    validCount,
    records,
    renderPayloadReference,
    renderPayloadReferenceLayout: {
      floatsPerRecord: 8,
      fields: {
        radiusPx: { offset: 0, components: 1 },
        conic: { offset: 1, components: 3 },
        alpha: { offset: 4, components: 1 },
        colorAlphaRgb: { offset: 5, components: 3 }
      },
      source: 'buildVisibleItemForCandidate CPU reference render payload fields'
    },
    tileRanges,
    timing: {
      cpuReferenceBuildMs: nowMs() - startMs
    }
  };
}

function buildRawXyzOpacityForCandidates(raw, candidateIndices) {
  const count = candidateIndices.length;
  const out = new Float32Array(count * 4);
  for (let i = 0; i < count; i += 1) {
    const srcIndex = candidateIndices[i];
    const o = i * 4;
    out[o + 0] = raw.xyz?.[srcIndex * raw.xyzDim + 0] ?? 0;
    out[o + 1] = raw.xyz?.[srcIndex * raw.xyzDim + 1] ?? 0;
    out[o + 2] = raw.xyz?.[srcIndex * raw.xyzDim + 2] ?? 0;
    out[o + 3] = raw.opacity?.[srcIndex * raw.opacityDim + 0] ?? 0;
  }
  return out;
}

function buildStatePositionsForCandidates(raw, candidateIndices, buildConfig) {
  const count = candidateIndices.length;
  const out = new Float32Array(count * 4);
  const flags = {
    nativeRot4d: !!buildConfig.useNativeRot4d,
    nativeMarginal: !!buildConfig.useNativeMarginal
  };
  for (let i = 0; i < count; i += 1) {
    const srcIndex = candidateIndices[i];
    const state = computeGaussianState(
      raw,
      srcIndex,
      buildConfig.timestamp,
      buildConfig.scalingModifier,
      buildConfig.sigmaScale,
      buildConfig.prefilterVar,
      buildConfig.useRot4d,
      flags
    );
    const o = i * 4;
    if (state?.pos) {
      out[o + 0] = state.pos[0] ?? 0;
      out[o + 1] = state.pos[1] ?? 0;
      out[o + 2] = state.pos[2] ?? 0;
      out[o + 3] = 1;
    } else {
      out[o + 0] = 0;
      out[o + 1] = 0;
      out[o + 2] = 0;
      out[o + 3] = 0;
    }
  }
  return out;
}

function compareRecords(reference, candidate, count, { epsilon, maxMismatches }) {
  const firstMismatches = [];
  let fieldMismatchCount = 0;
  let maxAbsError = 0;
  const fields = WEBGPU_VISIBLE_RECORD_FIELDS.filter(([field]) => field !== 'reserved');
  for (let row = 0; row < count; row += 1) {
    const base = row * RECORD_FLOATS;
    for (const [field, offset, components] of fields) {
      for (let c = 0; c < components; c += 1) {
        const ref = reference[base + offset + c];
        const got = candidate[base + offset + c];
        const diff = Math.abs(ref - got);
        maxAbsError = Math.max(maxAbsError, diff);
        if (diff > epsilon) {
          fieldMismatchCount += 1;
          if (firstMismatches.length < maxMismatches) {
            firstMismatches.push(createRecordMismatch({
              row,
              field,
              component: components > 1 ? c : null,
              expected: ref,
              actual: got,
              absError: diff
            }));
          }
        }
      }
    }
  }
  return createRecordComparisonResult({
    anyMismatch: fieldMismatchCount > 0,
    fieldMismatchCount,
    maxAbsError,
    firstMismatches
  });
}

function classifyComparison(comparison) {
  if (!comparison) return MISMATCH_CLASSIFICATIONS.WEBGPU_VISIBLE_RECORD_COMPARE_MISSING;
  if (!comparison.anyMismatch) return MISMATCH_CLASSIFICATIONS.NONE;
  const fields = new Set((comparison.firstMismatches ?? []).map((item) => item.field));
  if (fields.size === 1 && fields.has('aabb')) {
    return MISMATCH_CLASSIFICATIONS.WEBGPU_FIXED_RECORD_AABB_MISMATCH;
  }
  return MISMATCH_CLASSIFICATIONS.WEBGPU_FIXED_RECORD_FIELD_MISMATCH;
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

async function runCompute({ device, cpuReference, rawXyzOpacity, statePositions, projectionParams, rawCount }) {
  const shader = device.createShaderModule({
    label: 'phase3-step4-visible-record-projection-wgsl',
    code: `
struct Params {
  count: u32,
  recordFloats: u32,
  rawCount: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> candidates: array<u32>;
@group(0) @binding(1) var<storage, read> rawXyzOpacity: array<vec4f>;
@group(0) @binding(2) var<storage, read> referenceRecords: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> outputRecords: array<vec4f>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> statePositions: array<vec4f>;
@group(0) @binding(6) var<storage, read> projectionParams: array<vec4f>;

fn rowDot(row: vec4f, value: vec4f) -> f32 {
  return dot(row, value);
}

fn viewRow(index: u32) -> vec4f {
  return projectionParams[3u + index];
}

fn projectionRow(index: u32) -> vec4f {
  return projectionParams[7u + index];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let row = id.x;
  if (row >= params.count) {
    return;
  }
  let base = row * 3u;
  let srcIndex = candidates[row];
  let raw0 = rawXyzOpacity[row];
  let statePos = statePositions[row];
  var r0 = referenceRecords[base + 0u];
  var r1 = referenceRecords[base + 1u];
  var r2 = referenceRecords[base + 2u];

  // Phase 3 Step11: srcIndex, valid, and minimal screen projection fields are
  // produced in WGSL. The 4D Gaussian state and AABB remain CPU materialized.
  r0.x = f32(srcIndex);

  let header = projectionParams[0u];
  let scale = projectionParams[1u];
  let intrinsics = projectionParams[2u];
  let mode = header.x;
  let renderW = header.y;
  let renderH = header.z;
  let sx = scale.x;
  let sy = scale.y;
  let pixelXSign = scale.z;
  let pos4 = vec4f(statePos.x, statePos.y, statePos.z, 1.0);
  let mv4 = vec4f(
    rowDot(viewRow(0u), pos4),
    rowDot(viewRow(1u), pos4),
    rowDot(viewRow(2u), pos4),
    rowDot(viewRow(3u), pos4)
  );

  var projectedPx = 0.0;
  var projectedPy = 0.0;
  var projectedDepth = 0.0;
  var projectionOk = false;
  if (mode > 0.5) {
    projectedDepth = mv4.z;
    projectionOk = projectedDepth > 1e-6;
    projectedPx = (pixelXSign * intrinsics.x * (mv4.x / max(projectedDepth, 1e-8)) + intrinsics.z) * sx;
    projectedPy = (intrinsics.y * (mv4.y / max(projectedDepth, 1e-8)) + intrinsics.w) * sy;
  } else {
    projectedDepth = -mv4.z;
    let clip = vec4f(
      rowDot(projectionRow(0u), mv4),
      rowDot(projectionRow(1u), mv4),
      rowDot(projectionRow(2u), mv4),
      rowDot(projectionRow(3u), mv4)
    );
    let invW = 1.0 / (clip.w + 1e-7);
    let ndcX = clip.x * invW;
    let ndcY = clip.y * invW;
    projectionOk = projectedDepth > 1e-6;
    projectedPx = (((ndcX + 1.0) * renderW - 1.0) * 0.5) * sx;
    projectedPy = (((ndcY + 1.0) * renderH - 1.0) * 0.5) * sy;
  }

  let rawIndexInBounds = srcIndex < params.rawCount;
  let outputValid = rawIndexInBounds && statePos.w > 0.5 && projectionOk;
  r0.y = select(0.0, 1.0, outputValid);
  r0.z = select(0.0, projectedPx, outputValid);
  r0.w = select(0.0, projectedPy, outputValid);
  r1.x = select(0.0, projectedDepth, outputValid);

  // Reserved lanes carry a tiny raw-buffer fetch probe for future diagnostics.
  // They are outside the compared fixed-record fields.
  r2.y = raw0.x;
  r2.z = raw0.y;
  r2.w = raw0.z;

  outputRecords[base + 0u] = r0;
  outputRecords[base + 1u] = r1;
  outputRecords[base + 2u] = r2;
}`
  });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shader,
      entryPoint: 'main'
    }
  });
  const outputByteLength = cpuReference.records.byteLength;
  const candidateBuffer = createBuffer(device, cpuReference.candidateIndices, GPUBufferUsage.STORAGE);
  const rawBuffer = createBuffer(device, rawXyzOpacity, GPUBufferUsage.STORAGE);
  const statePositionBuffer = createBuffer(device, statePositions, GPUBufferUsage.STORAGE);
  const projectionParamsBuffer = createBuffer(device, projectionParams, GPUBufferUsage.STORAGE);
  const referenceBuffer = createBuffer(device, cpuReference.records, GPUBufferUsage.STORAGE);
  const outputBuffer = device.createBuffer({
    size: outputByteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const paramsBuffer = createBuffer(
    device,
    new Uint32Array([cpuReference.count, RECORD_FLOATS, rawCount, 0]),
    GPUBufferUsage.UNIFORM
  );
  const readbackBuffer = device.createBuffer({
    size: outputByteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: candidateBuffer } },
      { binding: 1, resource: { buffer: rawBuffer } },
      { binding: 2, resource: { buffer: referenceBuffer } },
      { binding: 3, resource: { buffer: outputBuffer } },
      { binding: 4, resource: { buffer: paramsBuffer } },
      { binding: 5, resource: { buffer: statePositionBuffer } },
      { binding: 6, resource: { buffer: projectionParamsBuffer } }
    ]
  });
  const dispatchStartMs = nowMs();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(cpuReference.count / 64));
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputByteLength);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const computeDispatchMs = nowMs() - dispatchStartMs;
  const readbackStartMs = nowMs();
  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(readbackBuffer.getMappedRange().slice(0));
  readbackBuffer.unmap();
  return {
    records: out,
    timing: {
      computeDispatchMs,
      readbackMs: nowMs() - readbackStartMs
    }
  };
}

async function runTileCountsCompute({ device, tileRanges, tileGrid }) {
  const startMs = nowMs();
  const tileCols = toFiniteInteger(tileGrid?.tileCols, 0);
  const tileRows = toFiniteInteger(tileGrid?.tileRows, 0);
  const tileCount = tileCols * tileRows;
  if (tileCount <= 0) {
    return createWebGpuTileCountsUnavailable('tile-grid-unavailable');
  }

  const tileRangeCount = tileRanges.length;
  const tileRangeData = new Uint32Array(Math.max(4, tileRangeCount * 4));
  for (let i = 0; i < tileRangeCount; i += 1) {
    const tr = tileRanges[i] ?? [0, 0, 0, 0];
    const o = i * 4;
    tileRangeData[o + 0] = toFiniteInteger(tr[0], 0);
    tileRangeData[o + 1] = toFiniteInteger(tr[1], 0);
    tileRangeData[o + 2] = toFiniteInteger(tr[2], 0);
    tileRangeData[o + 3] = toFiniteInteger(tr[3], 0);
  }

  const shader = device.createShaderModule({
    label: 'phase3-step21-tile-counts-only-wgsl',
    code: `
struct Params {
  tileRangeCount: u32,
  tileCols: u32,
  tileRows: u32,
  tileCount: u32,
};

@group(0) @binding(0) var<storage, read> tileRanges: array<vec4u>;
@group(0) @binding(1) var<storage, read_write> tileCounts: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let row = id.x;
  if (row >= params.tileRangeCount) {
    return;
  }
  let tr = tileRanges[row];
  var ty = tr.y;
  loop {
    if (ty > tr.w) {
      break;
    }
    if (ty < params.tileRows) {
      var tx = tr.x;
      loop {
        if (tx > tr.z) {
          break;
        }
        if (tx < params.tileCols) {
          let tileId = ty * params.tileCols + tx;
          if (tileId < params.tileCount) {
            atomicAdd(&tileCounts[tileId], 1u);
          }
        }
        tx = tx + 1u;
      }
    }
    ty = ty + 1u;
  }
}`
  });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shader,
      entryPoint: 'main'
    }
  });

  const tileRangeBuffer = createBuffer(device, tileRangeData, GPUBufferUsage.STORAGE);
  const tileCountsByteLength = Math.max(4, tileCount * Uint32Array.BYTES_PER_ELEMENT);
  const tileCountsBuffer = createBuffer(
    device,
    new Uint32Array(tileCount),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const paramsBuffer = createBuffer(
    device,
    new Uint32Array([tileRangeCount, tileCols, tileRows, tileCount]),
    GPUBufferUsage.UNIFORM
  );
  const readbackBuffer = device.createBuffer({
    size: tileCountsByteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: tileRangeBuffer } },
      { binding: 1, resource: { buffer: tileCountsBuffer } },
      { binding: 2, resource: { buffer: paramsBuffer } }
    ]
  });

  const dispatchStartMs = nowMs();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(tileRangeCount / 64)));
  pass.end();
  encoder.copyBufferToBuffer(tileCountsBuffer, 0, readbackBuffer, 0, tileCountsByteLength);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const computeMs = nowMs() - dispatchStartMs;

  const readbackStartMs = nowMs();
  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const tileCounts = Array.from(new Uint32Array(readbackBuffer.getMappedRange().slice(0), 0, tileCount));
  readbackBuffer.unmap();
  const readbackMs = nowMs() - readbackStartMs;

  let totalTileRefs = 0;
  let maxRefsPerTile = 0;
  let nonEmptyTiles = 0;
  for (const count of tileCounts) {
    totalTileRefs += count;
    if (count > 0) nonEmptyTiles += 1;
    if (count > maxRefsPerTile) maxRefsPerTile = count;
  }

  return {
    mode: 'webgpu-compute-tile-counts-only',
    status: 'ok',
    source: 'cpu-reference-visible-record-tileRange-buffer-upload',
    implementedInWgsl: true,
    tileOffsetsComputed: false,
    prefixSumImplemented: false,
    scatterImplemented: false,
    outputSchema: {
      tileCounts: 'uint32[tileCount]'
    },
    tileGrid: { tileCols, tileRows, tileCount, tileSize: toFiniteInteger(tileGrid?.tileSize, 32) },
    recordCounts: {
      tileRangeCount,
      tileCountsLength: tileCounts.length
    },
    metadata: {
      tileCountsType: 'uint32',
      totalTileRefs,
      maxRefsPerTile,
      nonEmptyTiles
    },
    capacity: {
      totalTileRefs,
      maxRefsPerTile,
      nonEmptyTiles,
      capacityStatus: 'no-overflow'
    },
    tileCounts,
    timing: {
      tileCountsWebGpuComputeMs: computeMs,
      tileCountsWebGpuReadbackMs: readbackMs,
      tileCountsWebGpuTotalMs: nowMs() - startMs
    }
  };
}

async function runTileOffsetsPrefixCompute({ device, webgpuTileCountsDryRun, tileGrid }) {
  const startMs = nowMs();
  if (!webgpuTileCountsDryRun || webgpuTileCountsDryRun.status !== 'ok') {
    const reason = webgpuTileCountsDryRun?.reason ??
      webgpuTileCountsDryRun?.status ??
      'webgpu-tile-counts-dry-run-unavailable';
    return createWebGpuTileOffsetsPrefixUnavailable(reason);
  }

  const tileCounts = toUint32Array(webgpuTileCountsDryRun.tileCounts);
  const tileCols = toFiniteInteger(tileGrid?.tileCols ?? webgpuTileCountsDryRun.tileGrid?.tileCols, 0);
  const tileRows = toFiniteInteger(tileGrid?.tileRows ?? webgpuTileCountsDryRun.tileGrid?.tileRows, 0);
  const tileSize = toFiniteInteger(tileGrid?.tileSize ?? webgpuTileCountsDryRun.tileGrid?.tileSize, 32);
  const tileCount = toFiniteInteger(webgpuTileCountsDryRun.tileGrid?.tileCount, tileCols * tileRows);
  if (tileCount <= 0 || tileCounts.length !== tileCount) {
    return createWebGpuTileOffsetsPrefixUnavailable('tile-counts-shape-unavailable');
  }

  const shader = device.createShaderModule({
    label: 'phase3-step23-tile-offsets-prefix-sum-wgsl',
    code: `
struct Params {
  tileCount: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> tileCounts: array<u32>;
@group(0) @binding(1) var<storage, read_write> tileOffsets: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x != 0u) {
    return;
  }
  var sum = 0u;
  tileOffsets[0u] = 0u;
  var i = 0u;
  loop {
    if (i >= params.tileCount) {
      break;
    }
    sum = sum + tileCounts[i];
    tileOffsets[i + 1u] = sum;
    i = i + 1u;
  }
}`
  });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shader,
      entryPoint: 'main'
    }
  });

  const tileOffsetsLength = tileCount + 1;
  const tileOffsetsByteLength = Math.max(4, tileOffsetsLength * Uint32Array.BYTES_PER_ELEMENT);
  const tileCountsBuffer = createBuffer(device, tileCounts, GPUBufferUsage.STORAGE);
  const tileOffsetsBuffer = createBuffer(
    device,
    new Uint32Array(tileOffsetsLength),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const paramsBuffer = createBuffer(
    device,
    new Uint32Array([tileCount, 0, 0, 0]),
    GPUBufferUsage.UNIFORM
  );
  const readbackBuffer = device.createBuffer({
    size: tileOffsetsByteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: tileCountsBuffer } },
      { binding: 1, resource: { buffer: tileOffsetsBuffer } },
      { binding: 2, resource: { buffer: paramsBuffer } }
    ]
  });

  const dispatchStartMs = nowMs();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyBufferToBuffer(tileOffsetsBuffer, 0, readbackBuffer, 0, tileOffsetsByteLength);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const computeMs = nowMs() - dispatchStartMs;

  const readbackStartMs = nowMs();
  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const tileOffsets = Array.from(new Uint32Array(readbackBuffer.getMappedRange().slice(0), 0, tileOffsetsLength));
  readbackBuffer.unmap();
  const readbackMs = nowMs() - readbackStartMs;

  let totalTileRefs = 0;
  let maxRefsPerTile = 0;
  let nonEmptyTiles = 0;
  let prefixOffsetsValid = tileOffsets[0] === 0;
  const firstValidationFailures = [];
  for (let i = 0; i < tileCount; i += 1) {
    const count = tileCounts[i] ?? 0;
    totalTileRefs += count;
    if (count > 0) nonEmptyTiles += 1;
    if (count > maxRefsPerTile) maxRefsPerTile = count;
    if (tileOffsets[i + 1] < tileOffsets[i] || tileOffsets[i + 1] - tileOffsets[i] !== count) {
      prefixOffsetsValid = false;
      if (firstValidationFailures.length < 8) {
        firstValidationFailures.push({ stage: 'tileOffsets', tileId: i, reason: 'exclusive-prefix-mismatch' });
      }
    }
  }
  const terminalTotalTileRefs = tileOffsets[tileCount] ?? 0;
  const reportedTotalTileRefs = webgpuTileCountsDryRun.metadata?.totalTileRefs ?? totalTileRefs;
  const totalTileRefsConsistent =
    terminalTotalTileRefs === totalTileRefs &&
    terminalTotalTileRefs === reportedTotalTileRefs;
  if (!totalTileRefsConsistent && firstValidationFailures.length < 8) {
    firstValidationFailures.push({ stage: 'totalTileRefs', reason: 'offset-terminal-does-not-match-webgpu-count-sum' });
  }

  return {
    mode: 'webgpu-compute-tile-offsets-prefix-sum',
    status: firstValidationFailures.length === 0 ? 'ok' : 'validation-failed',
    source: 'webgpuTileCountsDryRun.tileCounts',
    implementedInWgsl: true,
    webgpuPrefixComputed: true,
    scatterImplemented: false,
    outputSchema: {
      tileOffsets: 'uint32[tileCount + 1], exclusive prefix sum of webgpuTileCountsDryRun.tileCounts'
    },
    tileGrid: { tileCols, tileRows, tileCount, tileSize },
    recordCounts: {
      tileCountsLength: tileCounts.length,
      tileOffsetsLength: tileOffsets.length
    },
    metadata: {
      tileOffsetsType: 'uint32',
      tileOffsetsPolicy: 'exclusive-prefix-sum',
      tileOffsetsInitialValue: tileOffsets[0] ?? null,
      tileOffsetsTerminalValue: terminalTotalTileRefs,
      totalTileRefs: terminalTotalTileRefs,
      maxRefsPerTile,
      nonEmptyTiles
    },
    capacity: {
      totalTileRefs: terminalTotalTileRefs,
      maxRefsPerTile,
      nonEmptyTiles,
      capacityStatus: 'no-overflow'
    },
    validationSummary: {
      prefixOffsetsValid,
      totalTileRefsConsistent,
      capacityStatus: 'no-overflow',
      firstValidationFailures
    },
    tileOffsets,
    timing: {
      tileOffsetsWebGpuPrefixComputeMs: computeMs,
      tileOffsetsWebGpuPrefixReadbackMs: readbackMs,
      tileOffsetsWebGpuPrefixTotalMs: nowMs() - startMs
    }
  };
}

async function runTileIndicesScatterComparison({
  device,
  tileRanges,
  tileCountsToOffsetsDryRun,
  webgpuTileOffsetsPrefixDryRun,
  tileIndicesSelfComparison
}) {
  const startMs = nowMs();
  if (!tileCountsToOffsetsDryRun || tileCountsToOffsetsDryRun.status !== 'ok') {
    const reason = tileCountsToOffsetsDryRun?.reason ??
      tileCountsToOffsetsDryRun?.status ??
      'tile-counts-to-offsets-dry-run-unavailable';
    return createTileIndicesWebGpuScatterComparisonUnavailable(reason);
  }
  if (!webgpuTileOffsetsPrefixDryRun || webgpuTileOffsetsPrefixDryRun.status !== 'ok') {
    const reason = webgpuTileOffsetsPrefixDryRun?.reason ??
      webgpuTileOffsetsPrefixDryRun?.status ??
      'webgpu-tile-offsets-prefix-dry-run-unavailable';
    return createTileIndicesWebGpuScatterComparisonUnavailable(reason);
  }
  if (!tileIndicesSelfComparison || tileIndicesSelfComparison.status !== 'ok') {
    const reason = tileIndicesSelfComparison?.reason ??
      tileIndicesSelfComparison?.status ??
      'tile-indices-self-comparison-unavailable';
    return createTileIndicesWebGpuScatterComparisonUnavailable(reason);
  }

  const reference = materializeCpuReferenceTileIndices({ tileRanges, tileCountsToOffsetsDryRun });
  if (reference.status !== 'ok') {
    return createTileIndicesWebGpuScatterComparisonUnavailable(reference.reason);
  }

  const tileOffsets = toUint32Array(webgpuTileOffsetsPrefixDryRun.tileOffsets);
  const expectedTileIndices = reference.tileIndices;
  const tileCount = toFiniteInteger(webgpuTileOffsetsPrefixDryRun.tileGrid?.tileCount, reference.tileCount);
  const tileCols = toFiniteInteger(webgpuTileOffsetsPrefixDryRun.tileGrid?.tileCols, 0);
  const tileRows = toFiniteInteger(webgpuTileOffsetsPrefixDryRun.tileGrid?.tileRows, 0);
  const tileSize = toFiniteInteger(webgpuTileOffsetsPrefixDryRun.tileGrid?.tileSize, 32);
  const totalTileRefs = toFiniteInteger(
    webgpuTileOffsetsPrefixDryRun.metadata?.tileOffsetsTerminalValue ??
      webgpuTileOffsetsPrefixDryRun.metadata?.totalTileRefs ??
      tileOffsets[tileCount],
    0
  );
  if (
    tileCount <= 0 ||
    tileCols <= 0 ||
    tileRows <= 0 ||
    tileOffsets.length !== tileCount + 1 ||
    totalTileRefs !== expectedTileIndices.length
  ) {
    return createTileIndicesWebGpuScatterComparisonUnavailable('scatter-input-shape-unavailable');
  }

  const tileRangeCount = tileRanges.length;
  const tileRangeData = new Uint32Array(Math.max(4, tileRangeCount * 4));
  for (let i = 0; i < tileRangeCount; i += 1) {
    const tr = tileRanges[i] ?? [0, 0, 0, 0];
    const o = i * 4;
    tileRangeData[o + 0] = toFiniteInteger(tr[0], 0);
    tileRangeData[o + 1] = toFiniteInteger(tr[1], 0);
    tileRangeData[o + 2] = toFiniteInteger(tr[2], 0);
    tileRangeData[o + 3] = toFiniteInteger(tr[3], 0);
  }

  const shader = device.createShaderModule({
    label: 'phase3-step26-tile-indices-scatter-wgsl',
    code: `
struct Params {
  tileRangeCount: u32,
  tileCols: u32,
  tileRows: u32,
  tileCount: u32,
};

@group(0) @binding(0) var<storage, read> tileRanges: array<vec4u>;
@group(0) @binding(1) var<storage, read> tileOffsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> writeCursors: array<u32>;
@group(0) @binding(3) var<storage, read_write> tileIndices: array<u32>;
@group(0) @binding(4) var<storage, read_write> stats: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x != 0u) {
    return;
  }
  var recordIndex = 0u;
  loop {
    if (recordIndex >= params.tileRangeCount) {
      break;
    }
    let tr = tileRanges[recordIndex];
    var ty = tr.y;
    loop {
      if (ty > tr.w) {
        break;
      }
      if (ty < params.tileRows) {
        var tx = tr.x;
        loop {
          if (tx > tr.z) {
            break;
          }
          if (tx < params.tileCols) {
            let tileId = ty * params.tileCols + tx;
            if (tileId < params.tileCount) {
              let writeIndex = writeCursors[tileId];
              if (writeIndex >= tileOffsets[tileId] && writeIndex < tileOffsets[tileId + 1u]) {
                tileIndices[writeIndex] = recordIndex;
              } else {
                stats[0u] = stats[0u] + 1u;
              }
              writeCursors[tileId] = writeIndex + 1u;
            }
          }
          tx = tx + 1u;
        }
      }
      ty = ty + 1u;
    }
    recordIndex = recordIndex + 1u;
  }
}`
  });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shader,
      entryPoint: 'main'
    }
  });

  const tileIndicesByteLength = Math.max(4, totalTileRefs * Uint32Array.BYTES_PER_ELEMENT);
  const writeCursorsByteLength = Math.max(4, tileCount * Uint32Array.BYTES_PER_ELEMENT);
  const tileRangeBuffer = createBuffer(device, tileRangeData, GPUBufferUsage.STORAGE);
  const tileOffsetsBuffer = createBuffer(device, tileOffsets, GPUBufferUsage.STORAGE);
  const writeCursorsBuffer = createBuffer(
    device,
    tileOffsets.slice(0, tileCount),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const tileIndicesBuffer = createBuffer(
    device,
    new Uint32Array(totalTileRefs),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const statsBuffer = createBuffer(
    device,
    new Uint32Array(4),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const paramsBuffer = createBuffer(
    device,
    new Uint32Array([tileRangeCount, tileCols, tileRows, tileCount]),
    GPUBufferUsage.UNIFORM
  );
  const tileIndicesReadbackBuffer = device.createBuffer({
    size: tileIndicesByteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const writeCursorsReadbackBuffer = device.createBuffer({
    size: writeCursorsByteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const statsReadbackBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: tileRangeBuffer } },
      { binding: 1, resource: { buffer: tileOffsetsBuffer } },
      { binding: 2, resource: { buffer: writeCursorsBuffer } },
      { binding: 3, resource: { buffer: tileIndicesBuffer } },
      { binding: 4, resource: { buffer: statsBuffer } },
      { binding: 5, resource: { buffer: paramsBuffer } }
    ]
  });

  const dispatchStartMs = nowMs();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyBufferToBuffer(tileIndicesBuffer, 0, tileIndicesReadbackBuffer, 0, tileIndicesByteLength);
  encoder.copyBufferToBuffer(writeCursorsBuffer, 0, writeCursorsReadbackBuffer, 0, writeCursorsByteLength);
  encoder.copyBufferToBuffer(statsBuffer, 0, statsReadbackBuffer, 0, 16);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const computeMs = nowMs() - dispatchStartMs;

  const readbackStartMs = nowMs();
  await Promise.all([
    tileIndicesReadbackBuffer.mapAsync(GPUMapMode.READ),
    writeCursorsReadbackBuffer.mapAsync(GPUMapMode.READ),
    statsReadbackBuffer.mapAsync(GPUMapMode.READ)
  ]);
  const actualTileIndices = new Uint32Array(
    tileIndicesReadbackBuffer.getMappedRange().slice(0),
    0,
    totalTileRefs
  );
  const finalWriteCursors = new Uint32Array(
    writeCursorsReadbackBuffer.getMappedRange().slice(0),
    0,
    tileCount
  );
  const stats = new Uint32Array(statsReadbackBuffer.getMappedRange().slice(0), 0, 4);
  const readbackMs = nowMs() - readbackStartMs;

  let tileIndicesMismatchCount = 0;
  let orderingMismatchCount = 0;
  let writeCursorMismatchCount = 0;
  let maxAbsIndexDelta = 0;
  const firstMismatches = [];
  for (let tileId = 0; tileId < tileCount; tileId += 1) {
    const start = tileOffsets[tileId];
    const end = tileOffsets[tileId + 1];
    if (finalWriteCursors[tileId] !== end) {
      writeCursorMismatchCount += 1;
      if (firstMismatches.length < 8) {
        firstMismatches.push({
          kind: 'writeCursorMismatch',
          tileId,
          expected: end,
          actual: finalWriteCursors[tileId]
        });
      }
    }
    let previous = null;
    for (let index = start; index < end; index += 1) {
      const expected = expectedTileIndices[index];
      const actual = actualTileIndices[index];
      const delta = actual - expected;
      maxAbsIndexDelta = Math.max(maxAbsIndexDelta, Math.abs(delta));
      if (delta !== 0) {
        tileIndicesMismatchCount += 1;
        if (firstMismatches.length < 8) {
          firstMismatches.push({ kind: 'tileIndicesMismatch', tileId, index, expected, actual, delta });
        }
      }
      if (previous !== null && actual < previous) {
        orderingMismatchCount += 1;
        if (firstMismatches.length < 8) {
          firstMismatches.push({ kind: 'orderingMismatch', tileId, index, previous, actual });
        }
      }
      previous = actual;
    }
  }

  const capacityOverflowCount = stats[0] ?? 0;
  const capacityStatus = capacityOverflowCount === 0 ? 'no-overflow' : 'overflow-detected';
  const expectedCapacityStatus = tileIndicesSelfComparison.capacity?.capacityStatus ?? 'no-overflow';
  const capacityStatusMismatch = expectedCapacityStatus !== capacityStatus;
  const sampleTiles = makeTileCountsOffsetsSampleTiles(tileCountsToOffsetsDryRun).map((sample) => {
    const tileId = sample.tileId;
    const start = tileOffsets[tileId] ?? 0;
    const end = tileOffsets[tileId + 1] ?? start;
    const sampleLimit = Math.min(end, start + 4);
    return {
      ...sample,
      tileIndexStart: start,
      tileIndexEnd: end,
      tileIndexCount: Math.max(0, end - start),
      expectedFirstTileIndices: Array.from(expectedTileIndices.slice(start, sampleLimit)),
      actualFirstTileIndices: Array.from(actualTileIndices.slice(start, sampleLimit))
    };
  });
  const anyMismatch =
    tileIndicesMismatchCount > 0 ||
    orderingMismatchCount > 0 ||
    writeCursorMismatchCount > 0 ||
    capacityStatusMismatch ||
    capacityOverflowCount > 0;
  const transientTileIndices = new Uint32Array(actualTileIndices);

  tileIndicesReadbackBuffer.unmap();
  writeCursorsReadbackBuffer.unmap();
  statsReadbackBuffer.unmap();

  const result = {
    mode: 'webgpu-tile-indices-scatter-comparison',
    status: anyMismatch ? 'mismatch' : 'ok',
    expectedSource: 'cpu-reference-tileIndices',
    actualSource: 'webgpu-scatter-readback-tileIndices',
    source: 'tileRange + webgpuTileOffsetsPrefixDryRun.tileOffsets',
    implementedInWgsl: true,
    webgpuScatterComputed: true,
    tileIndicesMaterialized: true,
    tileIndicesStoredInJson: false,
    scatterCompared: true,
    fullTileListGeneration: false,
    sortImplemented: false,
    displayConnectionImplemented: false,
    anyMismatch,
    mismatchClassification: anyMismatch ? 'tileIndicesWebGpuScatterMismatch' : 'none',
    tileIndicesMismatchCount,
    orderingMismatchCount,
    writeCursorMismatchCount,
    capacityStatusMismatch,
    maxAbsIndexDelta,
    recordCounts: {
      tileRangeCount,
      tileIndicesLength: totalTileRefs,
      tileOffsetsLength: tileOffsets.length
    },
    capacity: {
      totalTileRefs,
      capacityOverflowCount,
      capacityStatus
    },
    orderingPolicy: {
      sourceOrder: 'record-index-order',
      perTileOrder: 'preserve incoming record order within each tile',
      orderingValidated: orderingMismatchCount === 0
    },
    validationSummary: {
      writeCursorFinalValid: writeCursorMismatchCount === 0,
      scatterOutputValid: tileIndicesMismatchCount === 0 && orderingMismatchCount === 0,
      capacityStatus,
      capacityOverflowCount,
      firstValidationFailures: firstMismatches
    },
    firstMismatches,
    sampleTiles,
    timing: {
      tileIndicesWebGpuScatterComputeMs: computeMs,
      tileIndicesWebGpuScatterReadbackMs: readbackMs,
      tileIndicesWebGpuScatterComparisonMs: nowMs() - startMs
    }
  };
  Object.defineProperty(result, 'transientTileIndices', {
    value: transientTileIndices,
    enumerable: false
  });
  return result;
}

function buildTileListSummaryComparison({
  tileCountsToOffsetsDryRun,
  webgpuTileCountsDryRun,
  tileCountsWebGpuComparison,
  webgpuTileOffsetsPrefixDryRun,
  tileOffsetsWebGpuPrefixComparison,
  tileIndicesSelfComparison,
  tileIndicesWebGpuScatterComparison
}) {
  const startMs = nowMs();
  const required = [
    ['tileCountsToOffsetsDryRun', tileCountsToOffsetsDryRun],
    ['webgpuTileCountsDryRun', webgpuTileCountsDryRun],
    ['tileCountsWebGpuComparison', tileCountsWebGpuComparison],
    ['webgpuTileOffsetsPrefixDryRun', webgpuTileOffsetsPrefixDryRun],
    ['tileOffsetsWebGpuPrefixComparison', tileOffsetsWebGpuPrefixComparison],
    ['tileIndicesSelfComparison', tileIndicesSelfComparison],
    ['tileIndicesWebGpuScatterComparison', tileIndicesWebGpuScatterComparison]
  ];
  for (const [name, value] of required) {
    if (!value || value.status !== 'ok') {
      const reason = value?.reason ?? value?.status ?? `${name}-unavailable`;
      return createTileListSummaryComparisonUnavailable(reason);
    }
  }

  const expectedCapacity = tileCountsToOffsetsDryRun.capacity ?? {};
  const countsCapacity = webgpuTileCountsDryRun.capacity ?? {};
  const offsetsCapacity = webgpuTileOffsetsPrefixDryRun.capacity ?? {};
  const indicesCapacity = tileIndicesWebGpuScatterComparison.capacity ?? {};
  const tileGrid = tileCountsToOffsetsDryRun.tileGrid ?? {};
  const recordCounts = {
    tileRangeCount: tileCountsToOffsetsDryRun.recordCounts?.tileRangeCount ?? null,
    tileCount: tileGrid.tileCount ?? null,
    tileCountsLength: webgpuTileCountsDryRun.recordCounts?.tileCountsLength ?? null,
    tileOffsetsLength: webgpuTileOffsetsPrefixDryRun.recordCounts?.tileOffsetsLength ?? null,
    tileIndicesLength: tileIndicesWebGpuScatterComparison.recordCounts?.tileIndicesLength ?? null
  };
  const expectedTotalTileRefs = expectedCapacity.totalTileRefs ??
    tileCountsToOffsetsDryRun.metadata?.totalTileRefs ??
    null;
  const actualTotalTileRefs = indicesCapacity.totalTileRefs ??
    offsetsCapacity.totalTileRefs ??
    countsCapacity.totalTileRefs ??
    null;
  const totalTileRefsMismatch = expectedTotalTileRefs !== actualTotalTileRefs;
  const expectedCapacityStatus = expectedCapacity.capacityStatus ??
    tileCountsToOffsetsDryRun.validationSummary?.capacityStatus ??
    'no-overflow';
  const actualCapacityStatus = indicesCapacity.capacityStatus ??
    offsetsCapacity.capacityStatus ??
    countsCapacity.capacityStatus ??
    'no-overflow';
  const capacityStatusMismatch = expectedCapacityStatus !== actualCapacityStatus;
  const firstMismatches = [];
  if (totalTileRefsMismatch) {
    firstMismatches.push({
      kind: 'totalTileRefsMismatch',
      expected: expectedTotalTileRefs,
      actual: actualTotalTileRefs
    });
  }
  if (capacityStatusMismatch) {
    firstMismatches.push({
      kind: 'capacityStatusMismatch',
      expected: expectedCapacityStatus,
      actual: actualCapacityStatus
    });
  }

  const countsOk =
    tileCountsWebGpuComparison.anyMismatch === false &&
    tileCountsWebGpuComparison.tileCountsMismatchCount === 0;
  const offsetsOk =
    tileOffsetsWebGpuPrefixComparison.anyMismatch === false &&
    tileOffsetsWebGpuPrefixComparison.tileOffsetsMismatchCount === 0;
  const indicesOk =
    tileIndicesWebGpuScatterComparison.anyMismatch === false &&
    tileIndicesWebGpuScatterComparison.tileIndicesMismatchCount === 0 &&
    tileIndicesWebGpuScatterComparison.orderingMismatchCount === 0 &&
    tileIndicesWebGpuScatterComparison.writeCursorMismatchCount === 0;
  const anyMismatch =
    !countsOk ||
    !offsetsOk ||
    !indicesOk ||
    totalTileRefsMismatch ||
    capacityStatusMismatch ||
    firstMismatches.length > 0;

  return {
    mode: 'non-display-webgpu-tile-list-summary-comparison',
    status: anyMismatch ? 'mismatch' : 'ok',
    source:
      'webgpuTileCountsDryRun + webgpuTileOffsetsPrefixDryRun + tileIndicesWebGpuScatterComparison',
    expectedSource: 'cpu-reference tileCounts/tileOffsets/tileIndices metadata',
    actualSource: 'webgpu tileCounts/tileOffsets/tileIndices dry-run summaries',
    implementedInWgsl: false,
    nonDisplayOnly: true,
    fullTileListGeneration: false,
    sortImplemented: false,
    displayConnectionImplemented: false,
    tileIndicesStoredInJson: false,
    anyMismatch,
    mismatchClassification: anyMismatch ? 'tileListSummaryMismatch' : 'none',
    stageStatuses: {
      countsStatus: countsOk ? 'ok' : 'mismatch',
      offsetsStatus: offsetsOk ? 'ok' : 'mismatch',
      indicesStatus: indicesOk ? 'ok' : 'mismatch'
    },
    mismatchCounts: {
      tileCountsMismatchCount: tileCountsWebGpuComparison.tileCountsMismatchCount ?? null,
      tileOffsetsMismatchCount: tileOffsetsWebGpuPrefixComparison.tileOffsetsMismatchCount ?? null,
      tileIndicesMismatchCount: tileIndicesWebGpuScatterComparison.tileIndicesMismatchCount ?? null,
      orderingMismatchCount: tileIndicesWebGpuScatterComparison.orderingMismatchCount ?? null,
      writeCursorMismatchCount: tileIndicesWebGpuScatterComparison.writeCursorMismatchCount ?? null
    },
    metadataComparison: {
      expectedTotalTileRefs,
      actualTotalTileRefs,
      totalTileRefsMismatch,
      expectedCapacityStatus,
      actualCapacityStatus,
      capacityStatusMismatch,
      expectedNonEmptyTiles: expectedCapacity.nonEmptyTiles ?? null,
      actualNonEmptyTiles: countsCapacity.nonEmptyTiles ?? offsetsCapacity.nonEmptyTiles ?? null,
      expectedMaxRefsPerTile: expectedCapacity.maxRefsPerTile ?? null,
      actualMaxRefsPerTile: countsCapacity.maxRefsPerTile ?? offsetsCapacity.maxRefsPerTile ?? null
    },
    recordCounts,
    tileGrid,
    orderingPolicy: tileIndicesWebGpuScatterComparison.orderingPolicy ?? {},
    capacity: {
      totalTileRefs: actualTotalTileRefs,
      capacityStatus: actualCapacityStatus,
      capacityOverflowCount: indicesCapacity.capacityOverflowCount ?? 0,
      maxRefsPerTile: countsCapacity.maxRefsPerTile ?? offsetsCapacity.maxRefsPerTile ?? null,
      nonEmptyTiles: countsCapacity.nonEmptyTiles ?? offsetsCapacity.nonEmptyTiles ?? null
    },
    validationSummary: {
      countsValid: countsOk,
      offsetsValid: offsetsOk,
      indicesValid: indicesOk,
      orderingValid:
        tileIndicesWebGpuScatterComparison.orderingPolicy?.orderingValidated === true,
      writeCursorFinalValid:
        tileIndicesWebGpuScatterComparison.validationSummary?.writeCursorFinalValid === true,
      scatterOutputValid:
        tileIndicesWebGpuScatterComparison.validationSummary?.scatterOutputValid === true,
      totalTileRefsConsistent: !totalTileRefsMismatch,
      capacityStatus: actualCapacityStatus,
      firstValidationFailures: firstMismatches
    },
    firstMismatches,
    sampleTiles: tileIndicesWebGpuScatterComparison.sampleTiles ?? [],
    timing: {
      tileListSummaryComparisonMs: nowMs() - startMs
    }
  };
}

function buildWebGpuTileListBackendOutput({
  tileListSummaryComparison,
  tileCountsToOffsetsDryRun,
  webgpuTileCountsDryRun,
  webgpuTileOffsetsPrefixDryRun,
  tileIndicesWebGpuScatterComparison
}) {
  const startMs = nowMs();
  if (!tileListSummaryComparison || tileListSummaryComparison.status !== 'ok') {
    const reason = tileListSummaryComparison?.reason ??
      tileListSummaryComparison?.status ??
      'tile-list-summary-comparison-unavailable';
    return createWebGpuTileListBackendOutputUnavailable(reason);
  }

  const tileGrid = tileListSummaryComparison.tileGrid ?? tileCountsToOffsetsDryRun?.tileGrid ?? {};
  const recordCounts = tileListSummaryComparison.recordCounts ?? {};
  const capacity = tileListSummaryComparison.capacity ?? {};
  const validation = tileListSummaryComparison.validationSummary ?? {};
  const backendOutputReady =
    tileListSummaryComparison.anyMismatch === false &&
    validation.countsValid === true &&
    validation.offsetsValid === true &&
    validation.indicesValid === true &&
    validation.scatterOutputValid === true;
  const unresolved = [
    'depth sort / per-tile ordering for final blending',
    'render payload fields: conic, alpha, colorAlpha.rgb, SH',
    'tile composite shader handoff',
    'display connection'
  ];
  return {
    mode: 'non-display-webgpu-tile-list-backend-output',
    status: backendOutputReady ? 'ok' : 'blocked',
    source: 'tileListSummaryComparison',
    backendOutputReady,
    nonDisplayOnly: true,
    fullTileListGeneration: false,
    sortImplemented: false,
    displayConnectionImplemented: false,
    tileIndicesStoredInJson: false,
    backendStage: 'tile-list-output-ready-for-render-handoff-planning',
    outputBuffers: {
      tileCounts: {
        source: 'webgpuTileCountsDryRun.tileCounts',
        type: 'uint32[tileCount]',
        length: webgpuTileCountsDryRun?.recordCounts?.tileCountsLength ?? recordCounts.tileCountsLength ?? null,
        storedInJson: Array.isArray(webgpuTileCountsDryRun?.tileCounts)
      },
      tileOffsets: {
        source: 'webgpuTileOffsetsPrefixDryRun.tileOffsets',
        type: 'uint32[tileCount + 1]',
        length: webgpuTileOffsetsPrefixDryRun?.recordCounts?.tileOffsetsLength ?? recordCounts.tileOffsetsLength ?? null,
        policy: 'exclusive-prefix-sum',
        storedInJson: Array.isArray(webgpuTileOffsetsPrefixDryRun?.tileOffsets)
      },
      tileIndices: {
        source: 'webgpu scatter readback validation buffer',
        type: 'uint32[totalTileRefs]',
        length: tileIndicesWebGpuScatterComparison?.recordCounts?.tileIndicesLength ?? recordCounts.tileIndicesLength ?? null,
        storedInJson: false
      },
      tileListMetadata: {
        source: 'tileListSummaryComparison',
        fields: [
          'tileCount',
          'totalTileRefs',
          'maxRefsPerTile',
          'nonEmptyTiles',
          'capacityStatus',
          'orderingPolicy'
        ],
        materializedInJson: true
      }
    },
    tileGrid,
    recordCounts,
    capacity,
    validationSummary: {
      countsValid: validation.countsValid === true,
      offsetsValid: validation.offsetsValid === true,
      indicesValid: validation.indicesValid === true,
      orderingValid: validation.orderingValid === true,
      writeCursorFinalValid: validation.writeCursorFinalValid === true,
      scatterOutputValid: validation.scatterOutputValid === true,
      totalTileRefsConsistent: validation.totalTileRefsConsistent === true,
      capacityStatus: validation.capacityStatus ?? capacity.capacityStatus ?? 'unknown'
    },
    handoffReadiness: {
      status: backendOutputReady ? 'ready-for-render-payload-planning' : 'blocked',
      displayConnectionAllowed: false,
      satisfied: [
        'webgpu tileCounts validated',
        'webgpu tileOffsets validated',
        'webgpu tileIndices scatter validated',
        'non-display tile-list summary validated'
      ],
      unresolved,
      blocked: unresolved.map((reason) => ({ stage: 'render-handoff', reason }))
    },
    nextBackendPrototypeStep: 'render-payload-and-sort-readiness',
    timing: {
      webgpuTileListBackendOutputMs: nowMs() - startMs
    }
  };
}

function buildRenderPayloadSortReadiness({
  webgpuTileListBackendOutput,
  conicContract,
  radiusContract,
  covarianceContract
}) {
  const startMs = nowMs();
  if (!webgpuTileListBackendOutput || webgpuTileListBackendOutput.status !== 'ok') {
    const reason = webgpuTileListBackendOutput?.reason ??
      webgpuTileListBackendOutput?.status ??
      'webgpu-tile-list-backend-output-unavailable';
    return createRenderPayloadSortReadinessUnavailable(reason);
  }

  const handoff = webgpuTileListBackendOutput.handoffReadiness ?? {};
  const outputBuffers = webgpuTileListBackendOutput.outputBuffers ?? {};
  const sortInputsReady =
    webgpuTileListBackendOutput.backendOutputReady === true &&
    !!outputBuffers.tileIndices &&
    !!outputBuffers.tileOffsets;
  const depthAvailable = WGSL_COMPUTED_FIELDS.includes('depth');
  const sortPrototypeReady = sortInputsReady && depthAvailable;
  const renderPayloadFields = {
    conic: {
      requiredForDisplay: true,
      computeMode: conicContract?.computeMode ?? 'deferred-screen-space-covariance-conic-parity',
      implementedInWgsl: conicContract?.implementedInWgsl === true,
      referenceSource: 'CPU/CUDA conic reference',
      dependency: 'screen-space covariance2D'
    },
    alpha: {
      requiredForDisplay: true,
      computeMode: 'deferred-alpha-power-evaluation-parity',
      implementedInWgsl: false,
      referenceSource: 'CPU/CUDA alpha evaluation reference',
      dependency: 'conic + opacity + per-pixel delta'
    },
    colorAlphaRgb: {
      requiredForDisplay: true,
      computeMode: 'deferred-color-sh-evaluation-parity',
      implementedInWgsl: false,
      referenceSource: 'CPU/CUDA colorAlpha.rgb reference',
      dependency: 'SH/color pipeline'
    },
    sh: {
      requiredForDisplay: true,
      computeMode: 'deferred-sh-evaluation-parity',
      implementedInWgsl: false,
      referenceSource: 'CPU/CUDA SH reference',
      dependency: 'view direction + SH coefficients'
    },
    radius: {
      requiredForTileList: true,
      computeMode: radiusContract?.computeMode ?? 'deferred-covariance-conic-dependent',
      implementedInWgsl: radiusContract?.implementedInWgsl === true,
      referenceSource: 'CPU radius reference for AABB/tileRange',
      dependency: 'screen-space covariance eigen radius'
    }
  };
  const payloadReady = Object.values(renderPayloadFields)
    .filter((field) => field.requiredForDisplay)
    .every((field) => field.implementedInWgsl === true);
  const payloadMissingFields = Object.entries(renderPayloadFields)
    .filter(([, field]) => field.requiredForDisplay && field.implementedInWgsl !== true)
    .map(([name]) => name);
  const sortBlockers = sortPrototypeReady
    ? ['depth sort display handoff not implemented']
    : ['tile-list output or depth field unavailable'];
  const payloadBlockers = payloadMissingFields.map(
    (field) => `render payload field not implemented: ${field}`
  );
  const blockers = [
    ...payloadBlockers.map((reason) => ({ stage: 'render-payload', reason })),
    ...sortBlockers.map((reason) => ({ stage: 'depth-sort', reason })),
    { stage: 'tile-composite', reason: 'tile composite shader handoff not implemented' },
    { stage: 'display-connection', reason: 'display connection intentionally deferred' }
  ];

  return {
    mode: 'render-payload-and-sort-readiness-summary',
    status: 'ok',
    source: 'webgpuTileListBackendOutput',
    backendOutputReady: webgpuTileListBackendOutput.backendOutputReady === true,
    displayConnectionAllowed: false,
    sortImplemented: false,
    compactionImplemented: false,
    renderPayloadGpuImplemented: false,
    tileCompositeImplemented: false,
    tileListBackendStage: webgpuTileListBackendOutput.backendStage ?? null,
    payloadReadiness: {
      status: payloadReady ? 'ready' : 'blocked',
      requiredFields: Object.keys(renderPayloadFields).filter(
        (name) => renderPayloadFields[name].requiredForDisplay
      ),
      missingFields: payloadMissingFields,
      fields: renderPayloadFields,
      comparisonReference: 'CPU/CUDA render payload reference before display handoff'
    },
    sortReadiness: {
      status: sortPrototypeReady ? 'ready-for-minimal-sort-comparison' : 'blocked',
      sortKey: 'depth',
      currentOrdering:
        'record-index-order within each tile from WebGPU scatter validation',
      requiredOrdering:
        'per-tile depth order for final alpha blending before display connection',
      inputs: {
        tileOffsets: outputBuffers.tileOffsets ?? null,
        tileIndices: outputBuffers.tileIndices ?? null,
        depth: {
          source: 'webgpu visible record depth',
          implementedInWgsl: depthAvailable
        }
      },
      comparisonReference: 'CPU/CUDA sorted visible order or CPU reference per-tile depth order',
      blockers: sortBlockers
    },
    readinessSummary: {
      tileListBackendReady: webgpuTileListBackendOutput.backendOutputReady === true,
      renderPayloadReady: payloadReady,
      sortPrototypeReady,
      displayConnectionAllowed: false,
      nextRecommendedUnit: sortPrototypeReady
        ? 'webgpu-render-handoff-stub-partial-payload'
        : 'render-payload-field-contract-summary'
    },
    inheritedHandoffReadiness: {
      status: handoff.status ?? null,
      satisfied: handoff.satisfied ?? [],
      unresolved: handoff.unresolved ?? []
    },
    blockers,
    nextBackendPrototypeStep: sortPrototypeReady
      ? 'webgpu-render-handoff-stub-partial-payload'
      : 'render-payload-field-contract-summary',
    timing: {
      renderPayloadSortReadinessMs: nowMs() - startMs
    },
    covarianceDependency: {
      covarianceComputeMode:
        covarianceContract?.computeMode ?? 'deferred-screen-space-covariance-conic-parity',
      conicComputeMode:
        conicContract?.computeMode ?? 'deferred-screen-space-covariance-conic-parity',
      radiusComputeMode:
        radiusContract?.computeMode ?? 'deferred-covariance-conic-dependent'
    }
  };
}

function getRecordDepth(records, recordIndex) {
  const base = recordIndex * RECORD_FLOATS;
  const depth = records?.[base + 4];
  return Number.isFinite(depth) ? depth : Number.POSITIVE_INFINITY;
}

function sortTileIndicesByDepth(tileIndices, depths) {
  return Array.from(tileIndices).sort((a, b) => {
    const da = depths[a] ?? Number.POSITIVE_INFINITY;
    const db = depths[b] ?? Number.POSITIVE_INFINITY;
    if (da !== db) return da - db;
    return a - b;
  });
}

function buildDepthSortComparison({
  tileRanges,
  tileCountsToOffsetsDryRun,
  webgpuTileOffsetsPrefixDryRun,
  tileIndicesWebGpuScatterComparison,
  renderPayloadSortReadiness,
  cpuReferenceRecords,
  webgpuRecords,
  epsilon = DEFAULT_EPSILON
}) {
  const startMs = nowMs();
  if (!renderPayloadSortReadiness || renderPayloadSortReadiness.status !== 'ok') {
    const reason = renderPayloadSortReadiness?.reason ??
      renderPayloadSortReadiness?.status ??
      'render-payload-sort-readiness-unavailable';
    return createDepthSortComparisonUnavailable(reason);
  }
  if (!tileCountsToOffsetsDryRun || tileCountsToOffsetsDryRun.status !== 'ok') {
    const reason = tileCountsToOffsetsDryRun?.reason ??
      tileCountsToOffsetsDryRun?.status ??
      'tile-counts-to-offsets-dry-run-unavailable';
    return createDepthSortComparisonUnavailable(reason);
  }
  if (!webgpuTileOffsetsPrefixDryRun || webgpuTileOffsetsPrefixDryRun.status !== 'ok') {
    const reason = webgpuTileOffsetsPrefixDryRun?.reason ??
      webgpuTileOffsetsPrefixDryRun?.status ??
      'webgpu-tile-offsets-prefix-dry-run-unavailable';
    return createDepthSortComparisonUnavailable(reason);
  }
  if (!tileIndicesWebGpuScatterComparison || tileIndicesWebGpuScatterComparison.status !== 'ok') {
    const reason = tileIndicesWebGpuScatterComparison?.reason ??
      tileIndicesWebGpuScatterComparison?.status ??
      'tile-indices-webgpu-scatter-comparison-unavailable';
    return createDepthSortComparisonUnavailable(reason);
  }

  const reference = materializeCpuReferenceTileIndices({ tileRanges, tileCountsToOffsetsDryRun });
  if (reference.status !== 'ok') {
    return createDepthSortComparisonUnavailable(reference.reason);
  }

  const tileOffsets = toUint32Array(webgpuTileOffsetsPrefixDryRun.tileOffsets);
  const actualTileIndices = tileIndicesWebGpuScatterComparison.transientTileIndices;
  const expectedTileIndices = reference.tileIndices;
  const tileCount = toFiniteInteger(webgpuTileOffsetsPrefixDryRun.tileGrid?.tileCount, reference.tileCount);
  const totalTileRefs = toFiniteInteger(tileOffsets[tileCount], 0);
  if (
    tileCount <= 0 ||
    tileOffsets.length !== tileCount + 1 ||
    !(actualTileIndices instanceof Uint32Array) ||
    actualTileIndices.length !== totalTileRefs ||
    expectedTileIndices.length !== totalTileRefs
  ) {
    return createDepthSortComparisonUnavailable('depth-sort-input-shape-unavailable');
  }

  const cpuDepths = new Float32Array(tileRanges.length);
  const webgpuDepths = new Float32Array(tileRanges.length);
  let depthKeyMismatchCount = 0;
  let maxAbsDepthDelta = 0;
  const firstMismatches = [];
  for (let recordIndex = 0; recordIndex < tileRanges.length; recordIndex += 1) {
    const expectedDepth = getRecordDepth(cpuReferenceRecords, recordIndex);
    const actualDepth = getRecordDepth(webgpuRecords, recordIndex);
    cpuDepths[recordIndex] = expectedDepth;
    webgpuDepths[recordIndex] = actualDepth;
    const depthDelta = Math.abs(actualDepth - expectedDepth);
    maxAbsDepthDelta = Math.max(maxAbsDepthDelta, depthDelta);
    if (depthDelta > epsilon) {
      depthKeyMismatchCount += 1;
      if (firstMismatches.length < 8) {
        firstMismatches.push({
          kind: 'depthKeyMismatch',
          recordIndex,
          expectedDepth,
          actualDepth,
          depthDelta
        });
      }
    }
  }

  let sortMismatchCount = 0;
  let exactSortDifferenceCount = 0;
  let nearTieSortDifferenceCount = 0;
  let orderingMismatchCount = 0;
  let sortedTileMismatchCount = 0;
  let exactSortedTileDifferenceCount = 0;
  let maxTileSortMismatchCount = 0;
  let maxTileExactSortDifferenceCount = 0;
  const firstSortDifferences = [];
  for (let tileId = 0; tileId < tileCount; tileId += 1) {
    const start = tileOffsets[tileId];
    const end = tileOffsets[tileId + 1];
    const expectedSorted = sortTileIndicesByDepth(expectedTileIndices.slice(start, end), cpuDepths);
    const actualSorted = sortTileIndicesByDepth(actualTileIndices.slice(start, end), webgpuDepths);
    let tileMismatchCount = 0;
    let tileExactDifferenceCount = 0;
    let previousDepth = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < actualSorted.length; i += 1) {
      const expected = expectedSorted[i] ?? null;
      const actual = actualSorted[i] ?? null;
      if (expected !== actual) {
        const expectedDepth = expected === null ? null : cpuDepths[expected];
        const actualDepth = actual === null ? null : webgpuDepths[actual];
        const sortDepthDelta =
          Number.isFinite(expectedDepth) && Number.isFinite(actualDepth)
            ? Math.abs(actualDepth - expectedDepth)
            : Number.POSITIVE_INFINITY;
        const isNearTieSortDifference = sortDepthDelta <= epsilon;
        exactSortDifferenceCount += 1;
        tileExactDifferenceCount += 1;
        if (isNearTieSortDifference) {
          nearTieSortDifferenceCount += 1;
        } else {
          sortMismatchCount += 1;
          tileMismatchCount += 1;
          if (firstMismatches.length < 8) {
            firstMismatches.push({
              kind: 'sortMismatch',
              tileId,
              localIndex: i,
              expected,
              actual,
              expectedDepth,
              actualDepth,
              sortDepthDelta
            });
          }
        }
        if (firstSortDifferences.length < 8) {
          firstSortDifferences.push({
            kind: isNearTieSortDifference ? 'nearTieSortDifference' : 'sortMismatch',
            tileId,
            localIndex: i,
            expected,
            actual,
            expectedDepth,
            actualDepth,
            sortDepthDelta
          });
        }
      }
      if (actual !== null) {
        const depth = webgpuDepths[actual];
        if (depth + epsilon < previousDepth) {
          orderingMismatchCount += 1;
          if (firstMismatches.length < 8) {
            firstMismatches.push({
              kind: 'orderingMismatch',
              tileId,
              localIndex: i,
              previousDepth,
              actualDepth: depth,
              actual
            });
          }
        }
        previousDepth = Math.max(previousDepth, depth);
      }
    }
    if (tileMismatchCount > 0) sortedTileMismatchCount += 1;
    if (tileExactDifferenceCount > 0) exactSortedTileDifferenceCount += 1;
    maxTileSortMismatchCount = Math.max(maxTileSortMismatchCount, tileMismatchCount);
    maxTileExactSortDifferenceCount =
      Math.max(maxTileExactSortDifferenceCount, tileExactDifferenceCount);
  }

  const sampleTiles = makeTileCountsOffsetsSampleTiles(tileCountsToOffsetsDryRun).map((sample) => {
    const tileId = sample.tileId;
    const start = tileOffsets[tileId] ?? 0;
    const end = tileOffsets[tileId + 1] ?? start;
    const expectedSorted = sortTileIndicesByDepth(expectedTileIndices.slice(start, end), cpuDepths);
    const actualSorted = sortTileIndicesByDepth(actualTileIndices.slice(start, end), webgpuDepths);
    const limit = Math.min(4, expectedSorted.length, actualSorted.length);
    return {
      ...sample,
      tileIndexStart: start,
      tileIndexEnd: end,
      tileIndexCount: Math.max(0, end - start),
      expectedFirstSortedIndices: expectedSorted.slice(0, limit),
      actualFirstSortedIndices: actualSorted.slice(0, limit),
      expectedFirstDepths: expectedSorted.slice(0, limit).map((recordIndex) => cpuDepths[recordIndex]),
      actualFirstDepths: actualSorted.slice(0, limit).map((recordIndex) => webgpuDepths[recordIndex])
    };
  });
  const anyMismatch =
    sortMismatchCount > 0 ||
    orderingMismatchCount > 0 ||
    depthKeyMismatchCount > 0;

  return {
    mode: 'cpu-staged-webgpu-tile-list-depth-sort-comparison',
    status: anyMismatch ? 'mismatch' : 'ok',
    source: 'webgpuTileOffsetsPrefixDryRun + tileIndicesWebGpuScatterComparison + webgpu fixed-record depth',
    expectedSource: 'cpu-reference tileIndices sorted by CPU reference depth',
    actualSource: 'WebGPU tileIndices sorted by WebGPU fixed-record depth on CPU staging path',
    implementedInWgsl: false,
    webgpuSortComputed: false,
    cpuStagedSortComputed: true,
    nonDisplayOnly: true,
    displayConnectionAllowed: false,
    tileCompositeImplemented: false,
    sortedIndicesStoredInJson: false,
    anyMismatch,
    mismatchClassification: anyMismatch ? 'depthSortComparisonMismatch' : 'none',
    sortMismatchCount,
    exactSortDifferenceCount,
    nearTieSortDifferenceCount,
    orderingMismatchCount,
    depthKeyMismatchCount,
    sortedTileMismatchCount,
    exactSortedTileDifferenceCount,
    maxTileSortMismatchCount,
    maxTileExactSortDifferenceCount,
    maxAbsDepthDelta,
    depthKeyPolicy: {
      key: 'depth',
      order: 'ascending-depth-near-to-far-front-to-back',
      tieBreak: 'recordIndex ascending for exact depth equality',
      comparisonTolerance: epsilon,
      comparisonToleranceAppliesTo:
        'depth-key validation and near-tie difference classification, not sort comparator ordering',
      sortMismatchDefinition:
        'semantic mismatch after excluding exact-index differences whose depth delta is within comparisonTolerance'
    },
    recordCounts: {
      tileRangeCount: tileRanges.length,
      tileCount,
      tileOffsetsLength: tileOffsets.length,
      tileIndicesLength: totalTileRefs
    },
    validationSummary: {
      sortOutputValid: sortMismatchCount === 0,
      orderingValid: orderingMismatchCount === 0,
      depthKeysValid: depthKeyMismatchCount === 0,
      displayConnectionAllowed: false,
      firstValidationFailures: firstMismatches
    },
    firstMismatches,
    firstSortDifferences,
    sampleTiles,
    timing: {
      depthSortComparisonMs: nowMs() - startMs
    }
  };
}

export async function runWebGpuVisibleRecordDryRun({
  candidateInfo,
  raw,
  camera,
  screenSpaceCamera = null,
  canvasWidth = 0,
  canvasHeight = 0,
  camPos = null,
  tileGrid = null,
  buildConfig = {},
  maxRecords = DEFAULT_MAX_RECORDS,
  epsilon = DEFAULT_EPSILON,
  maxMismatches = DEFAULT_MAX_MISMATCHES,
  viewerCanvasState = null,
  metadata = null
} = {}) {
  const totalStartMs = nowMs();
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    return makeFallback('webgpu-unavailable', { metadata });
  }
  if (!raw || !candidateInfo || !camera || !buildConfig) {
    return makeFallback('webgpu-visible-record-input-unavailable', { metadata });
  }
  const adapterStartMs = nowMs();
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return makeFallback('webgpu-adapter-unavailable', { metadata });
  const device = await adapter.requestDevice();
  const adapterDeviceMs = nowMs() - adapterStartMs;
  const cpuReference = buildCpuReferenceRecords({
    candidateInfo,
    raw,
    camera,
    screenSpaceCamera,
    canvasWidth,
    canvasHeight,
    camPos,
    tileGrid,
    buildConfig,
    maxRecords
  });
  if (cpuReference.candidateCount <= 0 || cpuReference.count <= 0) {
    return makeFallback('webgpu-visible-record-candidate-input-empty', {
      candidateCount: cpuReference.candidateCount,
      recordCount: cpuReference.count,
      validRecordCount: cpuReference.validCount,
      metadata
    });
  }
  const uploadStartMs = nowMs();
  const rawXyzOpacity = buildRawXyzOpacityForCandidates(raw, cpuReference.candidateIndices);
  const statePositions = buildStatePositionsForCandidates(raw, cpuReference.candidateIndices, buildConfig);
  const renderScale = Number.isFinite(buildConfig.renderScale) ? buildConfig.renderScale : 1;
  const renderW = Math.max(1, Math.round(canvasWidth * renderScale));
  const renderH = Math.max(1, Math.round(canvasHeight * renderScale));
  const sx = canvasWidth / renderW;
  const sy = canvasHeight / renderH;
  const projectionContract = buildWebGpuProjectionContract({ camera, screenSpaceCamera, renderW, renderH, sx, sy });
  const radiusContract = createWebGpuRadiusContract();
  const covarianceContract = createWebGpuCovarianceContract();
  const conicContract = createWebGpuConicContract();
  const aabbContract = createWebGpuAabbContract();
  const tileRangeContract = createWebGpuTileRangeContract();
  const tileListContract = createWebGpuTileListContract();
  const tileListCapacityContract = createWebGpuTileListCapacityContract();
  const tileListValidationContract = createWebGpuTileListValidationContract();
  const tileListValidationUnitContract = createWebGpuTileListValidationUnitContract();
  const tileCountsOffsetsComparisonSurfaceContract =
    createWebGpuTileCountsOffsetsComparisonSurfaceContract();
  const tileCountsToOffsetsDryRun = buildTileCountsOffsetsDryRun({
    tileRanges: cpuReference.tileRanges,
    tileGrid
  });
  const tileCountsOffsetsSelfComparison =
    buildTileCountsOffsetsSelfComparison(tileCountsToOffsetsDryRun);
  const bufferUploadPrepareMs = nowMs() - uploadStartMs;
  const webgpuTileCountsDryRun = await runTileCountsCompute({
    device,
    tileRanges: cpuReference.tileRanges,
    tileGrid
  });
  const tileCountsWebGpuComparison = buildTileCountsWebGpuComparison(
    tileCountsToOffsetsDryRun,
    webgpuTileCountsDryRun
  );
  const tileOffsetsFromWebGpuCountsDryRun = buildTileOffsetsFromWebGpuCountsDryRun(
    webgpuTileCountsDryRun,
    tileGrid
  );
  const tileOffsetsPrefixComparison = buildTileOffsetsPrefixComparison(
    tileCountsToOffsetsDryRun,
    tileOffsetsFromWebGpuCountsDryRun
  );
  const webgpuTileOffsetsPrefixDryRun = await runTileOffsetsPrefixCompute({
    device,
    webgpuTileCountsDryRun,
    tileGrid
  });
  const tileOffsetsWebGpuPrefixComparison = buildTileOffsetsWebGpuPrefixComparison(
    tileCountsToOffsetsDryRun,
    webgpuTileOffsetsPrefixDryRun
  );
  const scatterValidationBoundary = buildScatterValidationBoundaryDryRun({
    tileRanges: cpuReference.tileRanges,
    tileCountsToOffsetsDryRun,
    webgpuTileOffsetsPrefixDryRun
  });
  const tileIndicesSelfComparison = buildTileIndicesSelfComparison({
    tileRanges: cpuReference.tileRanges,
    tileCountsToOffsetsDryRun,
    scatterValidationBoundary
  });
  const tileIndicesWebGpuScatterComparison = await runTileIndicesScatterComparison({
    device,
    tileRanges: cpuReference.tileRanges,
    tileCountsToOffsetsDryRun,
    webgpuTileOffsetsPrefixDryRun,
    tileIndicesSelfComparison
  });
  const tileListSummaryComparison = buildTileListSummaryComparison({
    tileCountsToOffsetsDryRun,
    webgpuTileCountsDryRun,
    tileCountsWebGpuComparison,
    webgpuTileOffsetsPrefixDryRun,
    tileOffsetsWebGpuPrefixComparison,
    tileIndicesSelfComparison,
    tileIndicesWebGpuScatterComparison
  });
  const webgpuTileListBackendOutput = buildWebGpuTileListBackendOutput({
    tileListSummaryComparison,
    tileCountsToOffsetsDryRun,
    webgpuTileCountsDryRun,
    webgpuTileOffsetsPrefixDryRun,
    tileIndicesWebGpuScatterComparison
  });
  const renderPayloadSortReadiness = buildRenderPayloadSortReadiness({
    webgpuTileListBackendOutput,
    conicContract,
    radiusContract,
    covarianceContract
  });
  const rawCount = toFiniteInteger(raw.count ?? raw.N ?? (raw.xyz?.length / Math.max(1, raw.xyzDim || 3)), 0);
  const computeResult = await runCompute({
    device,
    cpuReference,
    rawXyzOpacity,
    statePositions,
    projectionParams: projectionContract.data,
    rawCount
  });
  const depthSortComparison = buildDepthSortComparison({
    tileRanges: cpuReference.tileRanges,
    tileCountsToOffsetsDryRun,
    webgpuTileOffsetsPrefixDryRun,
    tileIndicesWebGpuScatterComparison,
    renderPayloadSortReadiness,
    cpuReferenceRecords: cpuReference.records,
    webgpuRecords: computeResult.records,
    epsilon
  });
  const webgpuRenderHandoffStub = buildWebGpuRenderHandoffStub({
    webgpuTileListBackendOutput,
    depthSortComparison,
    renderPayloadSortReadiness,
    webgpuRecords: computeResult.records,
    renderPayloadReference: cpuReference.renderPayloadReference,
    renderPayloadReferenceLayout: cpuReference.renderPayloadReferenceLayout,
    epsilon,
    recordFloats: RECORD_FLOATS
  });
  const webgpuTileCompositeHandoffStub = buildWebGpuTileCompositeHandoffStub({
    webgpuRenderHandoffStub,
    webgpuTileListBackendOutput,
    depthSortComparison,
    tileIndicesWebGpuScatterComparison,
    webgpuTileOffsetsPrefixDryRun,
    webgpuRecords: computeResult.records,
    recordFloats: RECORD_FLOATS
  });
  const webgpuTileCompositeShaderHandoff = buildWebGpuTileCompositeShaderHandoff({
    webgpuTileCompositeHandoffStub,
    webgpuRenderHandoffStub,
    tileIndicesWebGpuScatterComparison,
    webgpuTileOffsetsPrefixDryRun,
    webgpuRecords: computeResult.records,
    recordFloats: RECORD_FLOATS
  });
  const webgpuTileCompositeShaderDryRunComparison =
    await buildWebGpuTileCompositeShaderDryRunComparison({
      device,
      webgpuTileCompositeShaderHandoff,
      webgpuRenderHandoffStub,
      epsilon
    });
  const webgpuTileCompositeAccumulationDryRunComparison =
    await buildWebGpuTileCompositeAccumulationDryRunComparison({
      device,
      webgpuTileCompositeShaderHandoff,
      webgpuRenderHandoffStub,
      epsilon
    });
  const webgpuFramebufferFreeTileOutputDryRunComparison =
    await buildWebGpuFramebufferFreeTileOutputDryRunComparison({
      device,
      webgpuTileCompositeAccumulationDryRunComparison,
      epsilon,
      bgGray01: 0
    });
  const webgpuRenderTargetHandoffDryRunComparison =
    await buildWebGpuRenderTargetHandoffDryRunComparison({
      device,
      webgpuFramebufferFreeTileOutputDryRunComparison,
      projectionContract: projectionContract.summary,
      canvasWidth,
      canvasHeight,
      epsilon
    });
  const webgpuConstrainedDisplayAdapterDryRunComparison =
    await buildWebGpuConstrainedDisplayAdapterDryRunComparison({
      device,
      webgpuRenderTargetHandoffDryRunComparison,
      epsilon
    });
  const webgpuGuardedFirstDisplayExperiment =
    await buildWebGpuGuardedFirstDisplayExperiment({
      device,
      webgpuConstrainedDisplayAdapterDryRunComparison,
      epsilon
    });
  const webgpuCanvasPresentationAdapterDryRunComparison =
    await buildWebGpuCanvasPresentationAdapterDryRunComparison({
      device,
      webgpuGuardedFirstDisplayExperiment,
      epsilon,
      viewerCanvasState
    });
  const webgpuExclusiveCanvasHandoffReadiness =
    buildWebGpuExclusiveCanvasHandoffReadiness({
      webgpuCanvasPresentationAdapterDryRunComparison,
      viewerCanvasState
    });
  const webgpuBackendFramePrototype =
    await buildWebGpuBackendFramePrototype({
      device,
      viewerCanvasState,
      webgpuRenderHandoffStub,
      webgpuFramebufferFreeTileOutputDryRunComparison,
      webgpuRenderTargetHandoffDryRunComparison,
      webgpuConstrainedDisplayAdapterDryRunComparison,
      canvasWidth,
      canvasHeight
    });
  const webgpuBackendFrameLifecyclePrototype =
    buildWebGpuBackendFrameLifecyclePrototype({
      webgpuBackendFramePrototype,
      repeatedFrameCount: 3
    });
  const webgpuBackendViewerLoopAdapter =
    await buildWebGpuBackendViewerLoopAdapter({
      initialBackendFramePrototype: webgpuBackendFramePrototype,
      repeatedFrameCount: 3,
      cameraSnapshot: {
        projectionContract: projectionContract.summary,
        canvasWidth,
        canvasHeight
      },
      executeBackendFrame: async ({ frameIndex, previousBackendFramePrototype }) =>
        buildWebGpuBackendFramePrototype({
          device,
          viewerCanvasState,
          webgpuRenderHandoffStub,
          webgpuFramebufferFreeTileOutputDryRunComparison,
          webgpuRenderTargetHandoffDryRunComparison,
          webgpuConstrainedDisplayAdapterDryRunComparison,
          canvasWidth,
          canvasHeight,
          frameIndex,
          previousBackendFramePrototype
        })
    });
  const {
    webgpuBackendFrameControlledRepeatedExecution
  } = webgpuBackendViewerLoopAdapter;
  const viewerLifecycleIntegrationRequest =
    metadata?.viewerLifecycleIntegrationRequest ?? {};
  const webgpuBackendViewerLifecycleIntegrationBoundary =
    buildWebGpuBackendViewerLifecycleIntegrationBoundary({
      requestedBackendMode:
        viewerLifecycleIntegrationRequest.requestedBackendMode ??
        viewerCanvasState?.requestedBackendMode ??
        'webgl2-fallback',
      allowViewerCanvasPresentation:
        viewerLifecycleIntegrationRequest.allowViewerCanvasPresentation === true ||
        viewerCanvasState?.allowViewerCanvasPresentation === true,
      enableViewerLoopHook:
        viewerLifecycleIntegrationRequest.webgpuBackendViewerLoopHook === true,
      renderLifecycleStage: 'webgpu-visible-record-dry-run-runtime',
      viewerCanvasState,
      cameraSnapshot: {
        deterministicState: metadata?.deterministicState ?? null,
        projectionContract: projectionContract.summary,
        canvasWidth,
        canvasHeight
      },
      adapterResult: webgpuBackendViewerLoopAdapter,
      adapterInvocationSource:
        viewerLifecycleIntegrationRequest.lastRenderLifecycleIntegrationBoundary
          ? 'viewer-render-lifecycle-hook-and-dry-run-adapter'
          : 'webgpu-visible-record-dry-run-adapter'
    });
  const webgpuBackendViewerLifecycleControlledExecution =
    buildWebGpuBackendViewerLifecycleControlledExecution({
      integrationBoundary: webgpuBackendViewerLifecycleIntegrationBoundary,
      adapterResult: webgpuBackendViewerLoopAdapter,
      invocationRequested:
        viewerLifecycleIntegrationRequest.webgpuBackendViewerLoopHook === true &&
        webgpuBackendViewerLifecycleIntegrationBoundary.integrationBoundaryReady === true,
      invocationSource:
        viewerLifecycleIntegrationRequest.lastRenderLifecycleControlledExecution
          ? 'viewer-render-lifecycle-hook-and-dry-run-controlled-execution'
          : viewerLifecycleIntegrationRequest.invocationSource ??
            'webgpu-visible-record-dry-run-controlled-execution',
      webgl2FrameLifecycleSuppressed:
        viewerCanvasState?.webgl2FrameLifecycleSuppressed === true,
      cameraSnapshot: {
        deterministicState: metadata?.deterministicState ?? null,
        projectionContract: projectionContract.summary,
        canvasWidth,
        canvasHeight
      }
    });
  const webgpuBackendViewerFrameExecutor =
    viewerLifecycleIntegrationRequest.lastRenderBackendFrameExecutor
      ? {
          ...viewerLifecycleIntegrationRequest.lastRenderBackendFrameExecutor,
          recorderObservation: {
            observedBy: 'webgpu-visible-record-dry-run-recorder',
            recorderRole: 'validation-oracle-json-recorder',
            executionBoundarySource:
              'viewer_app_gpu.renderCurrentFrame webgpu backend frame executor',
            captureDebugFunctionIsExecutionBoundary: false
          }
        }
      : {
          mode: 'webgpu-backend-viewer-frame-executor-boundary',
          status: 'unavailable',
          source:
            'Phase 3 Step73 viewer backend frame executor boundary was not observed in the latest render result',
          contractVersion:
            'phase3-step73-backend-viewer-frame-executor-boundary-v1',
          executorImplemented: true,
          executorReady: false,
          productionDisplayConnectionImplemented: false,
          displayConnectionAllowed: false,
          webgl2HybridRenderingAllowed: false,
          recorderObservation: {
            observedBy: 'webgpu-visible-record-dry-run-recorder',
            recorderRole: 'validation-oracle-json-recorder',
            executionBoundarySource: 'not-observed',
            captureDebugFunctionIsExecutionBoundary: false
          },
          validationSummary: {
            executorReady: false,
            firstValidationFailures: [
              {
                stage: 'viewer-backend-frame-executor-observation',
                reason:
                  'run scheduleRender with webgpuBackendViewerLoopHook=true before capture so the viewer lifecycle executor summary is available'
              }
            ]
          },
          firstValidationFailures: [
            {
              stage: 'viewer-backend-frame-executor-observation',
              reason:
                'run scheduleRender with webgpuBackendViewerLoopHook=true before capture so the viewer lifecycle executor summary is available'
            }
          ],
          timing: {
            webgpuBackendViewerFrameExecutorMs: 0
          }
        };
  const webgpuBackendRuntimeRunner =
    webgpuBackendViewerFrameExecutor?.runtimeRunner
      ? {
          ...webgpuBackendViewerFrameExecutor.runtimeRunner,
          recorderObservation: {
            observedBy: 'webgpu-visible-record-dry-run-recorder',
            recorderRole: 'validation-oracle-json-recorder',
            executionBoundarySource:
              'viewer_app_gpu.renderCurrentFrame webgpu backend runtime runner',
            captureDebugFunctionIsExecutionBoundary: false
          }
        }
      : {
          mode: 'webgpu-backend-runtime-runner',
          status: 'unavailable',
          source:
            'Phase 3 Step73 backend runtime runner was not observed in the latest render executor summary',
          contractVersion: 'phase3-step73-backend-runtime-runner-contract-v1',
          runtimeRunnerImplemented: true,
          runtimeRunnerReady: false,
          productionDisplayConnectionImplemented: false,
          displayConnectionAllowed: false,
          webgl2HybridRenderingAllowed: false,
          recorderObservation: {
            observedBy: 'webgpu-visible-record-dry-run-recorder',
            recorderRole: 'validation-oracle-json-recorder',
            executionBoundarySource: 'not-observed',
            captureDebugFunctionIsExecutionBoundary: false
          },
          validationSummary: {
            runtimeRunnerReady: false,
            firstValidationFailures: [
              {
                stage: 'runtime-runner-observation',
                reason:
                  'run scheduleRender with webgpuBackendViewerLoopHook=true before capture so the runtime runner summary is available'
              }
            ]
          },
          firstValidationFailures: [
            {
              stage: 'runtime-runner-observation',
              reason:
                'run scheduleRender with webgpuBackendViewerLoopHook=true before capture so the runtime runner summary is available'
            }
          ],
          timing: {
            webgpuBackendRuntimeRunnerMs: 0
          }
        };
  const webgpuBackendViewerFramePresentationPass =
    webgpuBackendViewerFrameExecutor?.webgpuBackendViewerFramePresentationPass ??
    webgpuBackendViewerFrameExecutor?.viewerFramePresentationPassContract ??
    null;
  const webgpuSchedulerFramePresentationBoundary =
    viewerLifecycleIntegrationRequest.lastRenderSchedulerFramePresentationBoundary ??
    buildUnavailableSchedulerFramePresentationBoundaryContract(
      'scheduler-owned guarded WebGPU frame presentation boundary was not observed in the latest render result',
      {
        validationOracleRole:
          'capture/dry-run observes scheduler-owned presentation boundary output but does not own it'
      }
    );
  const webgpuNormalBackendFrameImplementation =
    webgpuBackendRuntimeRunner?.webgpuNormalBackendFrameImplementation
      ? webgpuBackendRuntimeRunner.webgpuNormalBackendFrameImplementation
      : {
          mode: 'webgpu-normal-backend-frame-implementation',
          status: 'unavailable',
          source:
            'Phase 3 Step72 normal WebGPU backend implementation was not selected or observed in the latest runtime runner summary',
          contractVersion:
            'phase3-step72-normal-backend-frame-implementation-v1',
          implementationKind: 'webgpu-normal-backend-frame-implementation',
          normalBackendImplementationImplemented: true,
          normalBackendImplementationReady: false,
          productionDisplayConnectionImplemented: false,
          displayConnectionAllowed: false,
          webgl2HybridRenderingAllowed: false,
          validationSummary: {
            normalBackendImplementationReady: false,
            firstValidationFailures: [
              {
                stage: 'normal-backend-implementation-observation',
                reason:
                  'run with webgpuBackendImplementation=webgpu-normal-backend-frame-implementation to select the first normal WebGPU backend implementation path'
              }
            ]
          },
          firstValidationFailures: [
            {
              stage: 'normal-backend-implementation-observation',
              reason:
                'run with webgpuBackendImplementation=webgpu-normal-backend-frame-implementation to select the first normal WebGPU backend implementation path'
            }
          ],
          timing: {
            webgpuNormalBackendFrameImplementationMs: 0
          }
        };
  const {
    webgpuViewerCanvasCurrentTexturePath,
    webgpuViewerCanvasBoundedFirstPresent,
    webgpuViewerCanvasNativeBoundedColorSamples,
    webgpuViewerCanvasBoundedColorSourceSelector,
    webgpuViewerCanvasBoundedColorPresent
  } = webgpuBackendFramePrototype;
  const webgpuExclusiveFrameLifecycleSwitch =
    buildWebGpuExclusiveFrameLifecycleSwitch({
      webgpuCanvasPresentationAdapterDryRunComparison,
      webgpuExclusiveCanvasHandoffReadiness,
      webgpuViewerCanvasCurrentTexturePath,
      viewerCanvasState
    });
  const inputContract = createWebGpuInputBufferContract({
    candidateCount: cpuReference.candidateCount,
    recordCount: cpuReference.count,
    rawCount,
    recordFloats: RECORD_FLOATS,
    outputBufferBytes: cpuReference.records.byteLength,
    projectionParamMode: projectionContract.summary.mode
  });
  const compareStartMs = nowMs();
  const comparisonTolerance = createComparisonToleranceMetadata({ epsilon, maxMismatches });
  const recordComparison = compareRecords(cpuReference.records, computeResult.records, cpuReference.count, {
    epsilon,
    maxMismatches
  });
  const compareMs = nowMs() - compareStartMs;
  const mismatchClassification = classifyComparison(recordComparison);
  return {
    schemaVersion: WEBGPU_VISIBLE_RECORD_DRY_RUN_SCHEMA_VERSION,
    phaseStep: WEBGPU_VISIBLE_RECORD_PHASE_STEP,
    status: 'ok',
    reason: 'ok',
    computeMode: WEBGPU_VISIBLE_RECORD_COMPUTE_MODE,
    scaffoldMode: WEBGPU_VISIBLE_RECORD_SCAFFOLD_MODE,
    scaffoldNote: 'Phase 3 Step74 lets the scheduler/frame loop own the guarded WebGPU frame presentation boundary while preserving Step67/68/69/70/71/72/73 validation.',
    implementedFields: IMPLEMENTED_FIELDS,
    wgslComputedFields: WGSL_COMPUTED_FIELDS,
    wgslReferenceAssistedFields: WGSL_REFERENCE_ASSISTED_FIELDS,
    cpuMaterializedFields: CPU_MATERIALIZED_FIELDS,
    fieldComputeModes: cloneWebGpuVisibleRecordFieldComputeModes(),
    deferredFields: DEFERRED_FIELDS,
    candidateCount: cpuReference.candidateCount,
    recordCount: cpuReference.count,
    validRecordCount: cpuReference.validCount,
    recordFloats: RECORD_FLOATS,
    recordLayout: WEBGPU_VISIBLE_RECORD_FIELDS,
    comparisonContract: {
      schemaVersion: comparisonTolerance.schemaVersion,
      recordComparisonKeys: RECORD_COMPARISON_KEYS,
      mismatchClassifications: MISMATCH_CLASSIFICATIONS
    },
    comparisonTolerance,
    radiusContract,
    radiusComputeMode: radiusContract.computeMode,
    covarianceContract,
    conicContract,
    conicComputeMode: conicContract.computeMode,
    aabbContract,
    tileRangeContract,
    boundsComputeMode: {
      aabb: aabbContract.computeMode,
      tileRange: tileRangeContract.computeMode
    },
    tileListContract,
    tileListComputeMode: tileListContract.computeMode,
    tileListCapacityContract,
    tileListCapacityComputeMode: tileListCapacityContract.computeMode,
    tileListValidationContract,
    tileListValidationComputeMode: tileListValidationContract.computeMode,
    tileListValidationUnitContract,
    tileListValidationUnitComputeMode: tileListValidationUnitContract.computeMode,
    tileCountsOffsetsComparisonSurfaceContract,
    tileCountsOffsetsComparisonSurfaceComputeMode:
      tileCountsOffsetsComparisonSurfaceContract.computeMode,
    tileCountsToOffsetsDryRun,
    tileCountsOffsetsSelfComparison,
    webgpuTileCountsDryRun,
    tileCountsWebGpuComparison,
    tileOffsetsFromWebGpuCountsDryRun,
    tileOffsetsPrefixComparison,
    webgpuTileOffsetsPrefixDryRun,
    tileOffsetsWebGpuPrefixComparison,
    scatterValidationBoundary,
    tileIndicesSelfComparison,
    tileIndicesWebGpuScatterComparison,
    tileListSummaryComparison,
    webgpuTileListBackendOutput,
    renderPayloadSortReadiness,
    depthSortComparison,
    webgpuRenderHandoffStub,
    webgpuTileCompositeHandoffStub,
    webgpuTileCompositeShaderHandoff,
    webgpuTileCompositeShaderDryRunComparison,
    webgpuTileCompositeAccumulationDryRunComparison,
    webgpuFramebufferFreeTileOutputDryRunComparison,
    webgpuRenderTargetHandoffDryRunComparison,
    webgpuConstrainedDisplayAdapterDryRunComparison,
    webgpuGuardedFirstDisplayExperiment,
    webgpuCanvasPresentationAdapterDryRunComparison,
    webgpuExclusiveCanvasHandoffReadiness,
    webgpuViewerCanvasCurrentTexturePath,
    webgpuViewerCanvasBoundedFirstPresent,
    webgpuViewerCanvasNativeBoundedColorSamples,
    webgpuViewerCanvasBoundedColorSourceSelector,
    webgpuViewerCanvasBoundedColorPresent,
    webgpuBackendFramePrototype,
    webgpuBackendFrameLifecyclePrototype,
    webgpuBackendFrameControlledRepeatedExecution,
    webgpuBackendViewerLoopAdapter,
    webgpuBackendViewerLifecycleIntegrationBoundary,
    webgpuBackendViewerLifecycleControlledExecution,
    webgpuBackendViewerFrameExecutor,
    webgpuBackendViewerFramePresentationPass,
    webgpuSchedulerFramePresentationBoundary,
    webgpuBackendRuntimeRunner,
    webgpuNormalBackendFrameImplementation,
    webgpuExclusiveFrameLifecycleSwitch,
    inputContract,
    bufferContract: inputContract,
    inputBufferModes: inputContract.inputBufferModes,
    recordComparison,
    fieldMismatchCount: recordComparison.fieldMismatchCount,
    firstMismatches: recordComparison.firstMismatches,
    mismatchClassification,
    anyMismatch: !!recordComparison.anyMismatch,
    timing: {
      adapterDeviceMs,
      bufferUploadMs: bufferUploadPrepareMs,
      ...cpuReference.timing,
      ...tileCountsToOffsetsDryRun.timing,
      ...tileCountsOffsetsSelfComparison.timing,
      ...webgpuTileCountsDryRun.timing,
      ...tileCountsWebGpuComparison.timing,
      ...tileOffsetsFromWebGpuCountsDryRun.timing,
      ...tileOffsetsPrefixComparison.timing,
      ...webgpuTileOffsetsPrefixDryRun.timing,
      ...tileOffsetsWebGpuPrefixComparison.timing,
      ...scatterValidationBoundary.timing,
      ...tileIndicesSelfComparison.timing,
      ...tileIndicesWebGpuScatterComparison.timing,
      ...tileListSummaryComparison.timing,
      ...webgpuTileListBackendOutput.timing,
      ...renderPayloadSortReadiness.timing,
      ...depthSortComparison.timing,
      ...webgpuRenderHandoffStub.timing,
      ...webgpuTileCompositeHandoffStub.timing,
      ...webgpuTileCompositeShaderHandoff.timing,
      ...webgpuTileCompositeShaderDryRunComparison.timing,
      ...webgpuTileCompositeAccumulationDryRunComparison.timing,
      ...webgpuFramebufferFreeTileOutputDryRunComparison.timing,
      ...webgpuRenderTargetHandoffDryRunComparison.timing,
      ...webgpuConstrainedDisplayAdapterDryRunComparison.timing,
      ...webgpuGuardedFirstDisplayExperiment.timing,
      ...webgpuCanvasPresentationAdapterDryRunComparison.timing,
      ...webgpuExclusiveCanvasHandoffReadiness.timing,
      ...webgpuViewerCanvasCurrentTexturePath.timing,
      ...webgpuViewerCanvasBoundedFirstPresent.timing,
      ...webgpuViewerCanvasNativeBoundedColorSamples.timing,
      ...webgpuViewerCanvasBoundedColorSourceSelector.timing,
      ...webgpuViewerCanvasBoundedColorPresent.timing,
      ...webgpuBackendFramePrototype.timing,
      ...webgpuBackendFrameLifecyclePrototype.timing,
      ...webgpuBackendFrameControlledRepeatedExecution.timing,
      ...webgpuBackendViewerLoopAdapter.timing,
      ...webgpuBackendViewerLifecycleIntegrationBoundary.timing,
      ...webgpuBackendViewerLifecycleControlledExecution.timing,
      ...webgpuBackendViewerFrameExecutor.timing,
      ...webgpuBackendRuntimeRunner.timing,
      ...webgpuNormalBackendFrameImplementation.timing,
      ...webgpuExclusiveFrameLifecycleSwitch.timing,
      ...computeResult.timing,
      compareMs,
      totalMs: nowMs() - totalStartMs
    },
    webgpu: {
      adapterInfoAvailable: typeof adapter.requestAdapterInfo === 'function',
      rawBufferUploadMode: WEBGPU_INPUT_BUFFER_MODES.RAW_XYZ_OPACITY,
      statePositionUploadMode: WEBGPU_INPUT_BUFFER_MODES.STATE_POSITIONS,
      projectionParamMode: projectionContract.summary.mode,
      projectionContract: projectionContract.summary,
      radiusContract,
      radiusComputeMode: radiusContract.computeMode,
      covarianceContract,
      conicContract,
      conicComputeMode: conicContract.computeMode,
      aabbContract,
      tileRangeContract,
      boundsComputeMode: {
        aabb: aabbContract.computeMode,
        tileRange: tileRangeContract.computeMode
      },
      tileListContract,
      tileListComputeMode: tileListContract.computeMode,
      tileListCapacityContract,
      tileListCapacityComputeMode: tileListCapacityContract.computeMode,
      tileListValidationContract,
      tileListValidationComputeMode: tileListValidationContract.computeMode,
      tileListValidationUnitContract,
      tileListValidationUnitComputeMode: tileListValidationUnitContract.computeMode,
      tileCountsOffsetsComparisonSurfaceContract,
      tileCountsOffsetsComparisonSurfaceComputeMode:
        tileCountsOffsetsComparisonSurfaceContract.computeMode,
      tileCountsToOffsetsDryRun,
      tileCountsOffsetsSelfComparison,
      webgpuTileCountsDryRun,
      tileCountsWebGpuComparison,
      tileOffsetsFromWebGpuCountsDryRun,
      tileOffsetsPrefixComparison,
      webgpuTileOffsetsPrefixDryRun,
      tileOffsetsWebGpuPrefixComparison,
      scatterValidationBoundary,
      tileIndicesSelfComparison,
      tileIndicesWebGpuScatterComparison,
      tileListSummaryComparison,
      webgpuTileListBackendOutput,
      renderPayloadSortReadiness,
      depthSortComparison,
      webgpuRenderHandoffStub,
      webgpuTileCompositeHandoffStub,
      webgpuTileCompositeShaderHandoff,
      webgpuTileCompositeShaderDryRunComparison,
      webgpuTileCompositeAccumulationDryRunComparison,
      webgpuFramebufferFreeTileOutputDryRunComparison,
      webgpuRenderTargetHandoffDryRunComparison,
      webgpuConstrainedDisplayAdapterDryRunComparison,
      webgpuGuardedFirstDisplayExperiment,
      webgpuCanvasPresentationAdapterDryRunComparison,
      webgpuExclusiveCanvasHandoffReadiness,
      webgpuViewerCanvasCurrentTexturePath,
      webgpuViewerCanvasBoundedFirstPresent,
      webgpuViewerCanvasNativeBoundedColorSamples,
      webgpuViewerCanvasBoundedColorSourceSelector,
      webgpuViewerCanvasBoundedColorPresent,
      webgpuBackendFramePrototype,
      webgpuBackendFrameLifecyclePrototype,
      webgpuBackendFrameControlledRepeatedExecution,
      webgpuBackendViewerLoopAdapter,
      webgpuBackendViewerLifecycleIntegrationBoundary,
      webgpuBackendViewerLifecycleControlledExecution,
      webgpuBackendViewerFrameExecutor,
      webgpuBackendViewerFramePresentationPass,
      webgpuSchedulerFramePresentationBoundary,
      webgpuBackendRuntimeRunner,
      webgpuExclusiveFrameLifecycleSwitch,
      inputContract,
      bufferContract: inputContract,
      inputBufferModes: inputContract.inputBufferModes,
      candidateBufferCount: cpuReference.count,
      rawCount,
      outputBufferBytes: cpuReference.records.byteLength
    },
    metadata
  };
}
