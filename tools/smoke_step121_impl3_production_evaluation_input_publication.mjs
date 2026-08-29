import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
  buildNativeWebGpuProductionFrameDataPathContract,
  buildProductionEvaluationInputContract,
  buildProductionResidentSelectionContract,
  buildProductionResidentWorksetContract,
  PRODUCTION_EVALUATION_INPUT_CONTRACT_VERSION
} from '../demo/js/common_4dgs_production_frame_data_contracts.js';

const cloneJson = (value) => JSON.parse(JSON.stringify(value));

function buildProjectionFixture({ renderWidth = 640, renderHeight = 360 } = {}) {
  const values = new Float32Array(44);
  values.set([
    1, renderWidth, renderHeight, 0,
    2, 2, -1, 0,
    800, 810, 639.5, 359.5
  ]);
  values.set([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  ], 12);
  values.set([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  ], 28);
  return {
    data: values,
    summary: {
      schemaVersion: 'phase3-step4-webgpu-projection-contract-v1',
      mode: 'cuda-aligned',
      projectionContract: 'cuda-plus-z-forward-fx-fy-cx-cy',
      sourcePositionMode: 'cpu-materialized-4d-state-position',
      renderW: renderWidth,
      renderH: renderHeight,
      sx: 2,
      sy: 2,
      pixelXSign: -1,
      intrinsics: { fx: 800, fy: 810, cx: 639.5, cy: 359.5 },
      viewMatrixSource: 'cuda-aligned-view-matrix',
      projectionMatrixSource: 'intrinsics-fx-fy-cx-cy'
    }
  };
}

function buildWorksetFixture({
  sceneRecordCount = 3231588,
  residentStart = 524288,
  residentRecordCount = 524288
} = {}) {
  const selection = buildProductionResidentSelectionContract({
    request: {
      mode: 'range',
      rangeStart: residentStart,
      rangeCount: residentRecordCount
    },
    sceneRecordCount,
    resourceCapacityRecords: Math.max(residentRecordCount, 524288)
  });
  return buildProductionResidentWorksetContract({
    resourceIdentity: 'production-resident-workset-7',
    sceneResourceIdentity: 'production-scene-resource-4',
    sceneRecordCount,
    residentStart,
    residentRecordCount,
    resourceCapacityRecords: Math.max(residentRecordCount, 524288),
    residentSelectionContract: selection,
    selectionPolicy: selection.selectionPolicy,
    diagnosticMaxRecordsUsed: false,
    diagnosticCandidateSourceUsed: false,
    nonResidentRecordsExplicit: true,
    overflowPolicy: 'fail-closed-before-compositor-promotion'
  });
}

function buildReadyInputs(options = {}) {
  const sceneRecordCount = options.sceneRecordCount ?? 3231588;
  const residentStart = options.residentStart ?? 524288;
  const residentRecordCount = options.residentRecordCount ?? 524288;
  const projection = buildProjectionFixture();
  const frameIdentity = {
    generation: 7,
    frameHash: 'production-frame-hash-7',
    datasetCameraLabel: '000151_v13',
    datasetFrameNumber: 151,
    datasetTime: 23.2,
    referenceCameraLabel: '000151_v13',
    outputWidth: 1280,
    outputHeight: 720
  };
  return {
    assetIdentity: {
      sceneResourceIdentity: 'production-scene-resource-4',
      assetSha256: 'a'.repeat(64),
      assetSizeBytes: 987654321,
      assetSourceKind: 'default-scene-url',
      assetFormat: 'SPL4-v2',
      formatVersion: 2,
      recordCount: sceneRecordCount,
      dimensions: {
        xyz: 3,
        rotation: 4,
        rotationR: 4,
        scaleXyz: 3,
        fDc: 3,
        fRest: 45,
        opacity: 1,
        time: 1,
        scaleTime: 1
      }
    },
    worksetContract: buildWorksetFixture({
      sceneRecordCount,
      residentStart,
      residentRecordCount
    }),
    buildConfig: {
      timestamp: 23.2,
      scalingModifier: 1.25,
      sigmaScale: 3,
      prefilterVar: 0.3,
      renderScale: 0.5
    },
    effectiveRenderScale: 0.5,
    renderWidth: 640,
    renderHeight: 360,
    canvasWidth: 1280,
    canvasHeight: 720,
    projectionData: projection.data,
    projectionSummary: projection.summary,
    cameraIdentity: {
      cameraLabel: '000151_v13',
      referenceCameraLabel: '000151_v13',
      datasetFrameNumber: 151,
      datasetViewId: 13,
      cameraSource: 'camera-preset',
      datasetViewMatrixMode: 'cuda-aligned',
      fixedReferenceCameraActivationMode: 'fixed-reference-camera',
      cameraControlContract: 'fixed-reference-read-only',
      cameraOrientationPolicy: 'cuda-reference-camera-orientation'
    },
    timeIdentity: {
      requestedDatasetTime: 23.2,
      requestedViewerTime: 23.2,
      actualAppliedTimestamp: 23.2
    },
    requestIdentity: {
      schedulerRequestIdentity: 'viewer-render-request:9',
      schedulerRequestSource: 'capture-fresh-production-frame',
      sourceRequestIdentity: 'viewer-render-request:9',
      schedulerFrameIndex: 9
    },
    productionIdentity: {
      productionGeneration: 7,
      presentedGeneration: 7,
      productionFrameIdentity: { ...frameIdentity },
      presentedFrameIdentity: { ...frameIdentity }
    },
    orientationIdentity: {
      schemaVersion: 'phase3-webgpu-presentation-capture-orientation-v1',
      productionTextureOrigin: 'texture-memory-top-left',
      productionTextureYAxisDirection: 'down',
      presentationUvTransform:
        'fullscreen-framebuffer-top-left-to-texture-top-left',
      presentationVerticalFlipApplied: false,
      captureReadbackRowOrder:
        'copyTextureToBuffer-texture-memory-top-to-bottom',
      pngEncoderRowOrder: 'canvas-image-data-top-to-bottom',
      canonicalPresentationOrientation:
        'production-texture-top-left-y-down',
      savedPngOrientation: 'production-texture-top-left-y-down',
      captureVerticalFlipApplied: false,
      orientationMismatchClassification: 'none'
    }
  };
}

function assertBlocked(inputs, expectedReason = null) {
  const contract = buildProductionEvaluationInputContract(inputs);
  assert.equal(contract.status, 'blocked');
  assert.ok(contract.reason);
  if (expectedReason) assert.ok(contract.blockedReasons.includes(expectedReason));
  assert.doesNotThrow(() => JSON.stringify(contract));
  return contract;
}

function buildReadyParent(evaluationInputContract) {
  const workset = buildWorksetFixture();
  return buildNativeWebGpuProductionFrameDataPathContract({
    worksetContract: workset,
    stateResourceIdentity: 'state-resource',
    attributeResourceIdentity: 'attribute-resource',
    footprintResourceIdentity: 'footprint-resource',
    tileInputResourceIdentity: 'tile-input-resource',
    tileListInputResourceIdentity: 'tile-list-resource',
    compositorInputResourceIdentity: 'tile-list-resource',
    stateRecordCount: workset.residentRecordCount,
    tileInputRecordCount: workset.residentRecordCount,
    tileReferenceCapacityContract: { tileReferenceCapacityReady: true },
    boundedExecutionContract: { boundedExecutionReady: true },
    productionEvaluationInputContract: evaluationInputContract,
    gpuResourceLineagePreserved: true,
    capacityOverflowDetected: false,
    capacityOverflowFailClosed: true,
    silentDropAllowed: false,
    compositorSubmitted: true
  });
}

// Case A/B: valid bounded input is ready and preserves exact config/projection.
const readyInputs = buildReadyInputs();
const ready = buildProductionEvaluationInputContract(readyInputs);
assert.equal(ready.contractVersion, PRODUCTION_EVALUATION_INPUT_CONTRACT_VERSION);
assert.equal(ready.status, 'ready');
assert.equal(ready.reason, null);
assert.deepEqual(ready.blockedReasons, []);
assert.deepEqual(ready.appliedConfig, {
  timestamp: 23.2,
  scalingModifier: 1.25,
  sigmaScale: 3,
  prefilterVar: 0.3,
  configuredRenderScale: 0.5,
  effectiveRenderScale: 0.5,
  renderWidth: 640,
  renderHeight: 360,
  canvasWidth: 1280,
  canvasHeight: 720
});
assert.equal(ready.projection.floatCount, 44);
assert.deepEqual(ready.projection.values, Array.from(readyInputs.projectionData));
assert.equal(ArrayBuffer.isView(ready.projection.values), false);
assert.equal(ready.timeIdentity.requestedDatasetTimeMatchesAppliedTimestamp, true);
assert.equal(
  ready.productionIdentity.productionFrameIdentity.datasetTime,
  ready.appliedConfig.timestamp
);
assert.equal(
  ready.productionIdentity.presentedFrameIdentity.datasetTime,
  ready.appliedConfig.timestamp
);
assert.match(ready.assetIdentity.spl4AssetIdentity.sha256, /^[0-9a-fA-F]{64}$/);

// Case C: all published data is detached from inputs and deeply immutable.
const immutableInputs = buildReadyInputs();
const immutable = buildProductionEvaluationInputContract(immutableInputs);
const immutableJson = JSON.stringify(immutable);
immutableInputs.buildConfig.timestamp = 999;
immutableInputs.projectionData[8] = 999;
immutableInputs.worksetContract.residentSelectionContract.appliedStart = 0;
immutableInputs.assetIdentity.dimensions.xyz = 99;
immutableInputs.cameraIdentity.cameraLabel = 'mutated';
immutableInputs.timeIdentity.requestedDatasetTime = 999;
immutableInputs.requestIdentity.schedulerRequestIdentity = 'mutated';
immutableInputs.productionIdentity.productionFrameIdentity.datasetTime = 999;
immutableInputs.orientationIdentity.productionTextureYAxisDirection = 'up';
assert.equal(JSON.stringify(immutable), immutableJson);
assert.equal(Object.isFrozen(immutable), true);
assert.equal(Object.isFrozen(immutable.projection.values), true);
assert.equal(Object.isFrozen(immutable.productionWorksetIdentity.selection), true);
assert.throws(() => {
  immutable.appliedConfig.timestamp = 1;
}, TypeError);

// Case D: missing/nonfinite config and invalid effective scale fail closed.
for (const mutate of [
  (inputs) => { delete inputs.buildConfig.timestamp; },
  (inputs) => { inputs.buildConfig.scalingModifier = Number.NaN; },
  (inputs) => { inputs.buildConfig.sigmaScale = Number.POSITIVE_INFINITY; },
  (inputs) => { inputs.buildConfig.prefilterVar = Number.NEGATIVE_INFINITY; },
  (inputs) => { inputs.buildConfig.renderScale = 0; },
  (inputs) => { inputs.effectiveRenderScale = Number.NaN; }
]) {
  const inputs = buildReadyInputs();
  mutate(inputs);
  assertBlocked(inputs, 'production-evaluation-applied-config-invalid');
}

// Case D2: every authoritative dataset/applied/frame time must be identical.
const requestedTimeDriftInputs = buildReadyInputs();
requestedTimeDriftInputs.timeIdentity.requestedDatasetTime = 99;
const requestedTimeDrift = assertBlocked(
  requestedTimeDriftInputs,
  'production-evaluation-time-identity-missing-or-invalid'
);
assert.equal(
  requestedTimeDrift.timeIdentity.requestedDatasetTimeMatchesAppliedTimestamp,
  false
);

const appliedTimeDriftInputs = buildReadyInputs();
appliedTimeDriftInputs.timeIdentity.requestedDatasetTime = 99;
appliedTimeDriftInputs.productionIdentity.productionFrameIdentity.datasetTime = 99;
appliedTimeDriftInputs.productionIdentity.presentedFrameIdentity.datasetTime = 99;
const appliedTimeDrift = assertBlocked(
  appliedTimeDriftInputs,
  'production-evaluation-time-identity-missing-or-invalid'
);
assert.equal(
  appliedTimeDrift.timeIdentity.requestedDatasetTimeMatchesAppliedTimestamp,
  false
);

const productionTimeDriftInputs = buildReadyInputs();
productionTimeDriftInputs.productionIdentity.productionFrameIdentity.datasetTime = 99;
const productionTimeDrift = assertBlocked(
  productionTimeDriftInputs,
  'production-evaluation-frame-identity-missing-or-mismatched'
);
assert.equal(productionTimeDrift.productionIdentity.productionFrameIdentity.datasetTime, 99);

const presentedTimeDriftInputs = buildReadyInputs();
presentedTimeDriftInputs.productionIdentity.presentedFrameIdentity.datasetTime = 99;
const presentedTimeDrift = assertBlocked(
  presentedTimeDriftInputs,
  'production-evaluation-frame-identity-missing-or-mismatched'
);
assert.equal(presentedTimeDrift.productionIdentity.presentedFrameIdentity.datasetTime, 99);

// Case E: exact Float32Array length/values and summary agreement are required.
for (const mutate of [
  (inputs) => { inputs.projectionData = new Float32Array(43); },
  (inputs) => { inputs.projectionData = new Float32Array(45); },
  (inputs) => { inputs.projectionData[14] = Number.NaN; },
  (inputs) => { inputs.projectionData[14] = Number.POSITIVE_INFINITY; },
  (inputs) => { inputs.projectionData = Array.from(inputs.projectionData); },
  (inputs) => { inputs.projectionSummary.renderW = 641; },
  (inputs) => { inputs.projectionSummary.intrinsics.fx = 801; }
]) {
  const inputs = buildReadyInputs();
  mutate(inputs);
  assertBlocked(inputs);
}

// Case G2: SHA-256 identity is exactly 64 hexadecimal characters.
const shortShaInputs = buildReadyInputs();
shortShaInputs.assetIdentity.assetSha256 = 'abc123';
const shortSha = assertBlocked(
  shortShaInputs,
  'production-evaluation-asset-identity-missing-or-unsupported'
);
assert.equal(shortSha.assetIdentity.spl4AssetIdentity.sha256, 'abc123');

const nonHexShaInputs = buildReadyInputs();
nonHexShaInputs.assetIdentity.assetSha256 = 'g'.repeat(64);
const nonHexSha = assertBlocked(
  nonHexShaInputs,
  'production-evaluation-asset-identity-missing-or-unsupported'
);
assert.equal(nonHexSha.assetIdentity.spl4AssetIdentity.sha256, 'g'.repeat(64));

const uppercaseShaInputs = buildReadyInputs();
uppercaseShaInputs.assetIdentity.assetSha256 = 'A'.repeat(64);
const uppercaseSha = buildProductionEvaluationInputContract(uppercaseShaInputs);
assert.equal(uppercaseSha.status, 'ready');
assert.equal(uppercaseSha.assetIdentity.spl4AssetIdentity.sha256, 'A'.repeat(64));

// Case F: range, scene identity, ordering, readiness and diagnostic drift block.
for (const mutate of [
  (inputs) => {
    inputs.worksetContract.residentSelectionContract.requestedStart += 1;
  },
  (inputs) => { inputs.worksetContract.residentStart += 1; },
  (inputs) => { inputs.assetIdentity.sceneResourceIdentity = 'other-scene'; },
  (inputs) => {
    inputs.worksetContract.residentSelectionContract
      .productionResidentSelectionReady = false;
  },
  (inputs) => {
    inputs.worksetContract.residentSelectionContract.sourceIndexSpace =
      'resident-row';
  },
  (inputs) => { inputs.worksetContract.diagnosticCandidateSourceUsed = true; },
  (inputs) => { inputs.worksetContract.diagnosticMaxRecordsUsed = true; }
]) {
  const inputs = buildReadyInputs();
  mutate(inputs);
  assertBlocked(inputs);
}

// Case G: missing or unsafe identities never escape into the contract.
for (const mutate of [
  (inputs) => { inputs.assetIdentity = null; },
  (inputs) => {
    inputs.assetIdentity = Object.assign(Object.create({ inherited: true }),
      inputs.assetIdentity);
  },
  (inputs) => { inputs.assetIdentity.unsupported = undefined; },
  (inputs) => { inputs.assetIdentity.unsupported = Number.NaN; },
  (inputs) => { inputs.assetIdentity.self = inputs.assetIdentity; },
  (inputs) => { inputs.cameraIdentity = null; },
  (inputs) => { inputs.timeIdentity = null; },
  (inputs) => { inputs.requestIdentity = null; },
  (inputs) => { inputs.cameraIdentity.unsupported = new Date(); },
  (inputs) => { inputs.requestIdentity.unsupported = () => {}; }
]) {
  const inputs = buildReadyInputs();
  mutate(inputs);
  assertBlocked(inputs);
}

// Case H: output is JSON-safe and bounded independently of population size.
const serialized = JSON.stringify(ready);
assert.ok(serialized.length < 10000);
const disallowedKeys = new Set([
  'raw',
  'candidateIndices',
  'rawXyzOpacity',
  'device',
  'adapter',
  'buffer',
  'gpuResources'
]);
function inspectBounded(value, path = 'contract') {
  assert.equal(ArrayBuffer.isView(value), false, `${path} must not be a TypedArray`);
  if (Array.isArray(value)) {
    assert.ok(value.length <= 64, `${path} must remain bounded`);
    value.forEach((item, index) => inspectBounded(item, `${path}[${index}]`));
    return;
  }
  if (value && typeof value === 'object') {
    for (const [key, child] of Object.entries(value)) {
      assert.equal(disallowedKeys.has(key), false, `${path}.${key} is forbidden`);
      inspectBounded(child, `${path}.${key}`);
    }
  }
}
inspectBounded(ready);
const smallPopulation = buildProductionEvaluationInputContract(buildReadyInputs({
  sceneRecordCount: 32,
  residentStart: 8,
  residentRecordCount: 16
}));
assert.equal(smallPopulation.status, 'ready');
assert.ok(Math.abs(JSON.stringify(smallPopulation).length - serialized.length) < 100);

// Case I: nested publication readiness is observational and parent-neutral.
const blockedEvaluation = assertBlocked({});
const parentReady = buildReadyParent(ready);
const parentBlockedNested = buildReadyParent(blockedEvaluation);
const parentNullNested = buildReadyParent(null);
for (const parent of [parentReady, parentBlockedNested, parentNullNested]) {
  assert.equal(parent.nativeProductionFrameDataPathReady, true);
  assert.equal(parent.status, 'ok');
  assert.equal(parent.reason, null);
}
assert.equal(parentReady.productionEvaluationInputContract, ready);
assert.equal(parentBlockedNested.productionEvaluationInputContract, blockedEvaluation);
assert.equal(parentNullNested.productionEvaluationInputContract, null);

const parentTimeDrift = buildReadyParent(requestedTimeDrift);
const parentInvalidSha = buildReadyParent(shortSha);
for (const parent of [parentTimeDrift, parentInvalidSha]) {
  assert.equal(parent.nativeProductionFrameDataPathReady, true);
  assert.equal(parent.status, 'ok');
  assert.equal(parent.reason, null);
}

// Case J: the runtime publishes from the same locals without feeding metadata
// back into GPU execution or changing existing blocked-path reasons.
const source = await readFile(
  new URL('../demo/js/webgpu_production_frame_data_path.js', import.meta.url),
  'utf8'
);
const compositorIndex = source.indexOf(
  'const compositor = await buildWebGpuTileListCompositor'
);
const publicationIndex = source.indexOf(
  'buildProductionEvaluationInputContract({'
);
const parentIndex = source.indexOf(
  'buildNativeWebGpuProductionFrameDataPathContract({',
  publicationIndex
);
assert.ok(compositorIndex >= 0 && compositorIndex < publicationIndex);
assert.ok(publicationIndex < parentIndex);
assert.match(source, /buildConfig,\n\s+effectiveRenderScale: renderScale,/);
assert.match(source, /projectionData: projectionContract\.data,/);
assert.match(source, /projectionSummary: projectionContract\.summary,/);
assert.match(source, /projectionParams: projectionContract\.data,/);
assert.match(source, /readbackPolicy: 'none'/);
assert.match(source, /productionEvaluationInputContract,\n\s+cpuReferenceUsedAsProductionInput/);
assert.match(source, /projectionContract: projectionContract\.summary,/);
assert.match(source, /unavailableResult\(reason, workset\.contract\)/);
assert.doesNotMatch(
  source,
  /productionEvaluationInputContract\?\.(?:status|reason)|productionEvaluationInputContract\.status/
);

console.log('Step121 Impl3 production evaluation-input publication smoke passed');
