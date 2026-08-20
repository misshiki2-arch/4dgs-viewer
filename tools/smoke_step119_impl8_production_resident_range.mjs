import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import {
  buildViewerDeterministicSummary,
  parseViewerQueryState
} from '../demo/js/viewer_query_state.js';
import {
  runNativeWebGpuProductionFrameDataPath
} from '../demo/js/webgpu_production_frame_data_path.js';
import {
  selectActiveProductionResidentWorkset
} from '../demo/js/webgpu_production_workset_owner.js';

const SCENE_RECORD_COUNT = 1_048_576;
const RANGE_START = 524_288;
const RANGE_COUNT = 524_288;
const RANGE_END = RANGE_START + RANGE_COUNT;
const device = {
  limits: {
    maxStorageBufferBindingSize: 128 * 1024 * 1024,
    maxBufferSize: 128 * 1024 * 1024,
    maxComputeWorkgroupsPerDimension: 65_535
  }
};
const raw = {
  count: SCENE_RECORD_COUNT,
  xyzDim: 3,
  opacityDim: 1,
  xyz: new Float32Array(SCENE_RECORD_COUNT * 3),
  opacity: new Float32Array(SCENE_RECORD_COUNT)
};
const middleSourceIndex = RANGE_START + Math.floor(RANGE_COUNT / 2);
const sourceSamples = [
  [RANGE_START, [11, 12, 13], 0.11],
  [middleSourceIndex, [21, 22, 23], 0.22],
  [RANGE_END - 1, [31, 32, 33], 0.33]
];
for (const [sourceIndex, xyz, opacity] of sourceSamples) {
  raw.xyz.set(xyz, sourceIndex * raw.xyzDim);
  raw.opacity[sourceIndex] = opacity;
}

const defaultWorkset = selectActiveProductionResidentWorkset({ raw, device });
assert.equal(defaultWorkset.contract.residentWorksetReady, true);
assert.equal(defaultWorkset.contract.residentStart, 0);
assert.equal(defaultWorkset.contract.residentRecordCount, RANGE_COUNT);
assert.equal(defaultWorkset.contract.residentEndExclusive, RANGE_COUNT);
assert.equal(
  defaultWorkset.contract.selectionPolicy,
  'scene-owner-single-active-resource-bounded-resident-range'
);
assert.equal(defaultWorkset.candidateIndices[0], 0);
assert.equal(defaultWorkset.candidateIndices[RANGE_COUNT - 1], RANGE_COUNT - 1);

const productionSelectionRequest = {
  mode: 'range',
  rangeStart: RANGE_START,
  rangeCount: RANGE_COUNT
};
const visibleRangeWorkset = selectActiveProductionResidentWorkset({
  raw,
  device,
  productionResidentSelectionRequest: productionSelectionRequest
});
assert.equal(visibleRangeWorkset.contract.residentWorksetReady, true);
assert.equal(visibleRangeWorkset.contract.residentStart, RANGE_START);
assert.equal(visibleRangeWorkset.contract.residentRecordCount, RANGE_COUNT);
assert.equal(visibleRangeWorkset.contract.residentEndExclusive, RANGE_END);
assert.equal(visibleRangeWorkset.candidateIndices.length, RANGE_COUNT);
assert.equal(visibleRangeWorkset.rawXyzOpacity.length, RANGE_COUNT * 4);
assert.equal(visibleRangeWorkset.contract.nonResidentRecordCount, RANGE_COUNT);
assert.equal(visibleRangeWorkset.contract.residentRangeInBounds, true);
assert.equal(visibleRangeWorkset.contract.diagnosticMaxRecordsUsed, false);
assert.equal(visibleRangeWorkset.contract.diagnosticCandidateSourceUsed, false);
assert.equal(
  visibleRangeWorkset.contract.residentSelectionContract
    .productionResidentSelectionReady,
  true
);
assert.equal(
  visibleRangeWorkset.contract.residentSelectionContract.requestedEndExclusive,
  RANGE_END
);
assert.equal(visibleRangeWorkset.candidateIndices[0], RANGE_START);
assert.equal(
  visibleRangeWorkset.candidateIndices[Math.floor(RANGE_COUNT / 2)],
  middleSourceIndex
);
assert.equal(visibleRangeWorkset.candidateIndices[RANGE_COUNT - 1], RANGE_END - 1);

for (const [sourceIndex, xyz, opacity] of sourceSamples) {
  const residentRow = sourceIndex - RANGE_START;
  const packedOffset = residentRow * 4;
  assert.deepEqual(
    Array.from(visibleRangeWorkset.rawXyzOpacity.slice(
      packedOffset,
      packedOffset + 3
    )),
    xyz
  );
  assert.ok(
    Math.abs(
      visibleRangeWorkset.rawXyzOpacity[packedOffset + 3] - opacity
    ) < 1e-6
  );
}

const cachedVisibleRangeWorkset = selectActiveProductionResidentWorkset({
  raw,
  device,
  productionResidentSelectionRequest: productionSelectionRequest
});
assert.equal(cachedVisibleRangeWorkset, visibleRangeWorkset);
assert.equal(
  defaultWorkset.contract.sceneResourceIdentity,
  visibleRangeWorkset.contract.sceneResourceIdentity
);
assert.notEqual(
  defaultWorkset.contract.resourceIdentity,
  visibleRangeWorkset.contract.resourceIdentity
);
const defaultAfterRange = selectActiveProductionResidentWorkset({ raw, device });
assert.equal(defaultAfterRange.contract.residentStart, 0);
assert.equal(defaultAfterRange.candidateIndices[0], 0);
assert.notEqual(
  defaultAfterRange.contract.resourceIdentity,
  visibleRangeWorkset.contract.resourceIdentity
);

const smallRaw = {
  count: 10,
  xyzDim: 3,
  opacityDim: 1,
  xyz: new Float32Array(30),
  opacity: new Float32Array(10)
};
const smallDevice = {
  limits: {
    maxStorageBufferBindingSize: 1024,
    maxBufferSize: 1024,
    maxComputeWorkgroupsPerDimension: 65_535
  }
};
const invalidRequests = [
  { request: { mode: 'range', rangeStart: 0 }, reason: 'production-resident-range-start-count-required' },
  { request: { mode: 'range', rangeCount: 1 }, reason: 'production-resident-range-start-count-required' },
  { request: { mode: 'range', rangeStart: -1, rangeCount: 1 }, reason: 'production-resident-range-start-negative' },
  { request: { mode: 'range', rangeStart: 0, rangeCount: 0 }, reason: 'production-resident-range-count-not-positive' },
  { request: { mode: 'range', rangeStart: 0.5, rangeCount: 1 }, reason: 'production-resident-range-finite-safe-integers-required' },
  { request: { mode: 'range', rangeStart: 0, rangeCount: Number.POSITIVE_INFINITY }, reason: 'production-resident-range-finite-safe-integers-required' },
  { request: { mode: 'range', rangeStart: 8, rangeCount: 3 }, reason: 'production-resident-range-out-of-scene-bounds' },
  { request: { mode: 'range', rangeStart: 0, rangeCount: 5 }, reason: 'production-resident-range-exceeds-resource-capacity' },
  { request: { mode: 'other', rangeStart: 0, rangeCount: 1 }, reason: 'production-resident-selection-mode-range-required' },
  { request: { rangeStart: 0, rangeCount: 1 }, reason: 'production-resident-selection-mode-range-required' }
];
for (const { request, reason } of invalidRequests) {
  const blocked = selectActiveProductionResidentWorkset({
    raw: smallRaw,
    device: smallDevice,
    productionResidentSelectionRequest: request
  });
  assert.equal(blocked.contract.residentWorksetReady, false);
  assert.equal(blocked.contract.reason, reason);
  assert.equal(blocked.candidateIndices.length, 0);
}

const querySearch = [
  '?gpuCandidateSourceMode=range',
  'gpuCandidateRangeStart=7',
  'gpuCandidateRangeCount=8',
  'webgpuProductionResidentSelectionMode=range',
  `webgpuProductionResidentRangeStart=${RANGE_START}`,
  `webgpuProductionResidentRangeCount=${RANGE_COUNT}`
].join('&');
const queryState = parseViewerQueryState(querySearch);
const deterministicSummary = buildViewerDeterministicSummary(queryState);
assert.equal(deterministicSummary.gpuCandidateRangeStart, 7);
assert.equal(deterministicSummary.gpuCandidateRangeCount, 8);
assert.equal(deterministicSummary.webgpuProductionResidentSelectionMode, 'range');
assert.equal(deterministicSummary.webgpuProductionResidentRangeStart, RANGE_START);
assert.equal(deterministicSummary.webgpuProductionResidentRangeCount, RANGE_COUNT);
const deterministicParams = new URLSearchParams(
  deterministicSummary.deterministicQueryString
);
assert.equal(deterministicParams.get('gpuCandidateRangeStart'), '7');
assert.equal(
  deterministicParams.get('webgpuProductionResidentRangeStart'),
  String(RANGE_START)
);

const changedDiagnosticState = buildViewerDeterministicSummary(
  parseViewerQueryState(querySearch
    .replace('gpuCandidateRangeStart=7', 'gpuCandidateRangeStart=17')
    .replace('gpuCandidateRangeCount=8', 'gpuCandidateRangeCount=18'))
);
assert.equal(
  changedDiagnosticState.webgpuProductionResidentRangeStart,
  deterministicSummary.webgpuProductionResidentRangeStart
);
assert.equal(
  changedDiagnosticState.webgpuProductionResidentRangeCount,
  deterministicSummary.webgpuProductionResidentRangeCount
);
const malformedRangeState = buildViewerDeterministicSummary(
  parseViewerQueryState(
    '?webgpuProductionResidentSelectionMode=range&' +
    'webgpuProductionResidentRangeStart=not-a-number&' +
    'webgpuProductionResidentRangeCount=1'
  )
);
assert.equal(
  malformedRangeState.webgpuProductionResidentRangeStart,
  'not-a-number'
);
const malformedRangeWorkset = selectActiveProductionResidentWorkset({
  raw: smallRaw,
  device: smallDevice,
  productionResidentSelectionRequest: {
    mode: malformedRangeState.webgpuProductionResidentSelectionMode,
    rangeStart: malformedRangeState.webgpuProductionResidentRangeStart,
    rangeCount: malformedRangeState.webgpuProductionResidentRangeCount
  }
});
assert.equal(malformedRangeWorkset.contract.residentWorksetReady, false);
assert.equal(
  malformedRangeWorkset.contract.reason,
  'production-resident-range-finite-safe-integers-required'
);
const worksetFromChangedDiagnostic = selectActiveProductionResidentWorkset({
  raw,
  device,
  productionResidentSelectionRequest: {
    mode: changedDiagnosticState.webgpuProductionResidentSelectionMode,
    rangeStart: changedDiagnosticState.webgpuProductionResidentRangeStart,
    rangeCount: changedDiagnosticState.webgpuProductionResidentRangeCount
  }
});
assert.equal(worksetFromChangedDiagnostic.contract.residentStart, RANGE_START);
assert.equal(
  worksetFromChangedDiagnostic.contract.residentRecordCount,
  RANGE_COUNT
);

const previousNavigatorDescriptor = Object.getOwnPropertyDescriptor(
  globalThis,
  'navigator'
);
Object.defineProperty(globalThis, 'navigator', {
  configurable: true,
  value: {
    gpu: {
      requestAdapter: async () => ({
        requestDevice: async () => smallDevice
      })
    }
  }
});
try {
  const forwardedBlockedResult = await runNativeWebGpuProductionFrameDataPath({
    raw: smallRaw,
    camera: {},
    canvasWidth: 1280,
    canvasHeight: 720,
    viewerCanvasState: { canvas: {} },
    productionResidentSelectionRequest: {
      mode: 'range',
      rangeStart: 0,
      rangeCount: 5
    }
  });
  assert.equal(forwardedBlockedResult.status, 'blocked');
  assert.equal(
    forwardedBlockedResult.reason,
    'production-resident-range-exceeds-resource-capacity'
  );
  assert.equal(
    forwardedBlockedResult.productionResidentWorksetContract
      .residentSelectionContract.requestedRecordCount,
    5
  );
} finally {
  if (previousNavigatorDescriptor) {
    Object.defineProperty(
      globalThis,
      'navigator',
      previousNavigatorDescriptor
    );
  } else {
    delete globalThis.navigator;
  }
}

const urlResult = spawnSync('python3', [
  '-B',
  'tools/make_step_url.py',
  '--host',
  '127.0.0.1:8080',
  '--source-mode',
  'range',
  '--range-start',
  '7',
  '--range-count',
  '8',
  '--webgpu-production-resident-selection-mode',
  'range',
  '--webgpu-production-resident-range-start',
  String(RANGE_START),
  '--webgpu-production-resident-range-count',
  String(RANGE_COUNT)
], { encoding: 'utf8' });
assert.equal(urlResult.status, 0, urlResult.stderr);
const generatedUrl = new URL(urlResult.stdout.trim());
assert.equal(generatedUrl.searchParams.get('gpuCandidateRangeStart'), '7');
assert.equal(
  generatedUrl.searchParams.get('webgpuProductionResidentRangeStart'),
  String(RANGE_START)
);
assert.equal(
  generatedUrl.searchParams.get('webgpuProductionResidentRangeCount'),
  String(RANGE_COUNT)
);

const defaultUrlResult = spawnSync('python3', [
  '-B',
  'tools/make_step_url.py',
  '--host',
  '127.0.0.1:8080'
], { encoding: 'utf8' });
assert.equal(defaultUrlResult.status, 0, defaultUrlResult.stderr);
assert.equal(
  new URL(defaultUrlResult.stdout.trim()).searchParams.has(
    'webgpuProductionResidentSelectionMode'
  ),
  false
);

const partialUrlResult = spawnSync('python3', [
  '-B',
  'tools/make_step_url.py',
  '--webgpu-production-resident-selection-mode',
  'range'
], { encoding: 'utf8' });
assert.notEqual(partialUrlResult.status, 0);

console.log('Step119 Impl8 production resident range smoke passed.');
