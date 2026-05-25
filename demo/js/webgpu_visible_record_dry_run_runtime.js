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
  const tileRangeData = new Uint32Array(Math.max(1, tileRangeCount * 4));
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
  const tileRangeData = new Uint32Array(Math.max(1, tileRangeCount * 4));
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

  tileIndicesReadbackBuffer.unmap();
  writeCursorsReadbackBuffer.unmap();
  statsReadbackBuffer.unmap();

  return {
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
  const rawCount = toFiniteInteger(raw.count ?? raw.N ?? (raw.xyz?.length / Math.max(1, raw.xyzDim || 3)), 0);
  const computeResult = await runCompute({
    device,
    cpuReference,
    rawXyzOpacity,
    statePositions,
    projectionParams: projectionContract.data,
    rawCount
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
    scaffoldNote: 'Phase 3 Step26 adds a minimal WebGPU tileIndices scatter comparison while keeping full tile-list generation deferred.',
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
