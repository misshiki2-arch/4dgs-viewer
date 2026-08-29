import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
  buildProductionEvaluationInputContract,
  buildProductionResidentSelectionContract,
  buildProductionResidentWorksetContract,
  PRODUCTION_EVALUATION_INPUT_CONTRACT_VERSION
} from '../demo/js/common_4dgs_production_frame_data_contracts.js';
import {
  POPULATION_RASTER_SEMANTIC_ACTUAL_PROVENANCE,
  POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE,
  POPULATION_SEMANTIC_ACTUAL_PROVENANCE,
  POPULATION_SEMANTIC_EXPECTED_PROVENANCE,
  POPULATION_SEMANTIC_MAX_CHUNK_RECORDS,
  POPULATION_SEMANTIC_STAGE_CONTRACTS,
  PRODUCTION_RESIDENT_RANGE_COUNT,
  PRODUCTION_RESIDENT_RANGE_END,
  PRODUCTION_RESIDENT_RANGE_START
} from '../demo/js/common_4dgs_population_semantic_comparison_contracts.js';
import {
  POPULATION_SEMANTIC_FIXED_CHUNK_COUNT,
  POPULATION_SEMANTIC_ORCHESTRATION_CONTRACT_NAME,
  POPULATION_SEMANTIC_ORCHESTRATION_SCHEMA_VERSION
} from '../demo/js/webgpu_population_aligned_semantic_comparison_orchestrator.js';
import {
  createPopulationSemanticComparisonController,
  POPULATION_SEMANTIC_COMPARISON_CONTROLLER_CONTRACT_VERSION
} from '../demo/js/webgpu_population_semantic_comparison_controller.js';

const ASSET_SHA = 'a'.repeat(64);
const ASSET_SIZE = 987654321;
const SCENE_RECORD_COUNT = 3231588;
const SCENE_RESOURCE_IDENTITY = 'production-scene-resource-4';

function buildProjectionFixture() {
  const values = new Float32Array(44);
  values.set([
    1, 640, 360, 0,
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
    values,
    summary: {
      schemaVersion: 'phase3-step4-webgpu-projection-contract-v1',
      mode: 'cuda-aligned',
      projectionContract: 'cuda-plus-z-forward-fx-fy-cx-cy',
      sourcePositionMode: 'cpu-materialized-4d-state-position',
      renderW: 640,
      renderH: 360,
      sx: 2,
      sy: 2,
      pixelXSign: -1,
      intrinsics: { fx: 800, fy: 810, cx: 639.5, cy: 359.5 },
      viewMatrixSource: 'cuda-aligned-view-matrix',
      projectionMatrixSource: 'intrinsics-fx-fy-cx-cy'
    }
  };
}

function buildReadyEvaluationContract({
  residentStart = PRODUCTION_RESIDENT_RANGE_START,
  residentRecordCount = PRODUCTION_RESIDENT_RANGE_COUNT,
  assetSha = ASSET_SHA
} = {}) {
  const projection = buildProjectionFixture();
  const selection = buildProductionResidentSelectionContract({
    request: {
      mode: 'range',
      rangeStart: residentStart,
      rangeCount: residentRecordCount
    },
    sceneRecordCount: SCENE_RECORD_COUNT,
    resourceCapacityRecords: Math.max(residentRecordCount, PRODUCTION_RESIDENT_RANGE_COUNT)
  });
  const workset = buildProductionResidentWorksetContract({
    resourceIdentity: 'production-resident-workset-7',
    sceneResourceIdentity: SCENE_RESOURCE_IDENTITY,
    sceneRecordCount: SCENE_RECORD_COUNT,
    residentStart,
    residentRecordCount,
    resourceCapacityRecords: Math.max(
      residentRecordCount,
      PRODUCTION_RESIDENT_RANGE_COUNT
    ),
    residentSelectionContract: selection,
    selectionPolicy: selection.selectionPolicy,
    diagnosticMaxRecordsUsed: false,
    diagnosticCandidateSourceUsed: false,
    nonResidentRecordsExplicit: true,
    overflowPolicy: 'fail-closed-before-compositor-promotion'
  });
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
  return buildProductionEvaluationInputContract({
    assetIdentity: {
      sceneResourceIdentity: SCENE_RESOURCE_IDENTITY,
      assetSha256: assetSha,
      assetSizeBytes: ASSET_SIZE,
      assetSourceKind: 'default-scene-url',
      assetFormat: 'SPL4-v2',
      formatVersion: 2,
      recordCount: SCENE_RECORD_COUNT,
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
    worksetContract: workset,
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
    projectionData: projection.values,
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
  });
}

function buildRawFixture({ sha = ASSET_SHA } = {}) {
  return {
    version: 2,
    N: SCENE_RECORD_COUNT,
    xyzDim: 3,
    rotationDim: 4,
    rotationRDim: 4,
    scaleXYZDim: 3,
    fdcDim: 3,
    frestDim: 45,
    opacityDim: 1,
    tDim: 1,
    scaleTDim: 1,
    assetSha256: sha,
    assetSizeBytes: ASSET_SIZE,
    assetSourceKind: 'default-scene-url',
    xyz: new Float32Array(1),
    rotation: new Float32Array(1),
    rotation_r: new Float32Array(1),
    scale_xyz: new Float32Array(1),
    f_dc: new Float32Array(1),
    f_rest: new Float32Array(1),
    opacity: new Float32Array(1),
    t: new Float32Array(1),
    scale_t: new Float32Array(1)
  };
}

function mismatchEntry() {
  return {
    chunkIndex: 0,
    localRow: 3,
    globalResidentRow: 3,
    srcIndex: PRODUCTION_RESIDENT_RANGE_START + 3,
    stage: 'webgpuInclusivePixelBounds',
    component: 'minX',
    expected: 1,
    actual: 2,
    absoluteError: 1,
    tolerance: 0,
    expectedProvenance: POPULATION_SEMANTIC_EXPECTED_PROVENANCE,
    actualProvenance: POPULATION_SEMANTIC_ACTUAL_PROVENANCE,
    expectedStageProvenance:
      POPULATION_RASTER_SEMANTIC_EXPECTED_PROVENANCE.webgpuInclusivePixelBounds,
    actualStageProvenance: POPULATION_RASTER_SEMANTIC_ACTUAL_PROVENANCE
  };
}

function buildOrchestrationResult(decision = 'match') {
  const mismatch = decision === 'mismatch';
  const blocked = decision === 'blocked';
  const completedChunkCount = blocked ? 3 : POPULATION_SEMANTIC_FIXED_CHUNK_COUNT;
  const processedRecordCount = completedChunkCount * POPULATION_SEMANTIC_MAX_CHUNK_RECORDS;
  const stageSummaries = POPULATION_SEMANTIC_STAGE_CONTRACTS.map((stage) => ({
    stage: stage.key,
    components: [...stage.components],
    comparedCount: blocked ? processedRecordCount : PRODUCTION_RESIDENT_RANGE_COUNT,
    mismatchCount:
      mismatch && stage.key === 'webgpuInclusivePixelBounds' ? 1 : 0,
    maxAbsoluteError:
      mismatch && stage.key === 'webgpuInclusivePixelBounds' ? 1 : 0,
    tolerance: stage.tolerance,
    classification:
      blocked
        ? 'blocked-incomplete-evidence'
        : mismatch && stage.key === 'webgpuInclusivePixelBounds'
          ? 'mismatch'
          : 'match'
  }));
  const firstMismatches = mismatch ? [mismatchEntry()] : [];
  return {
    schemaVersion: POPULATION_SEMANTIC_ORCHESTRATION_SCHEMA_VERSION,
    contractName: POPULATION_SEMANTIC_ORCHESTRATION_CONTRACT_NAME,
    decision,
    match: decision === 'match',
    reason: blocked
      ? 'fixture-orchestration-blocked'
      : mismatch
        ? 'semantic-mismatch'
        : null,
    blockedReasons: blocked ? ['fixture-orchestration-blocked'] : [],
    coverage: {
      requestedRangeStart: PRODUCTION_RESIDENT_RANGE_START,
      requestedRangeCount: PRODUCTION_RESIDENT_RANGE_COUNT,
      requestedRangeEnd: PRODUCTION_RESIDENT_RANGE_END,
      completedChunkCount,
      processedRecordCount,
      firstSrcIndex: completedChunkCount > 0
        ? PRODUCTION_RESIDENT_RANGE_START
        : null,
      lastSrcIndex: blocked
        ? PRODUCTION_RESIDENT_RANGE_START + processedRecordCount - 1
        : PRODUCTION_RESIDENT_RANGE_END - 1,
      coverageComplete: !blocked
    },
    stageSummaries,
    firstMismatches,
    firstMismatchCount: firstMismatches.length,
    firstMismatchLimit: 16,
    requestedChunkCount: POPULATION_SEMANTIC_FIXED_CHUNK_COUNT,
    completedChunkCount,
    chunkSummaries: Array.from(
      { length: completedChunkCount },
      (_, chunkIndex) => ({
        chunkIndex,
        rangeStart:
          PRODUCTION_RESIDENT_RANGE_START +
          chunkIndex * POPULATION_SEMANTIC_MAX_CHUNK_RECORDS,
        rangeCount: POPULATION_SEMANTIC_MAX_CHUNK_RECORDS,
        decision: mismatch && chunkIndex === 0 ? 'mismatch' : 'match'
      })
    ),
    rasterCompanionSummary: {
      validatedChunkCount: completedChunkCount,
      allChunksReady: !blocked,
      allChunksRowAlignmentVerified: !blocked,
      nativeTileInputBufferUsageChanged: false
    },
    evidenceComplete: !blocked,
    expectedProvenance: POPULATION_SEMANTIC_EXPECTED_PROVENANCE,
    actualProvenance: POPULATION_SEMANTIC_ACTUAL_PROVENANCE,
    actualEvidenceSameProductionDispatch: false,
    productionCalculationDependsOnDiagnosticReadback: false,
    diagnosticDeviceOwnership: 'caller-owned-reused-not-destroyed',
    singleChunkResultsIncluded: false,
    rawRecordArraysIncluded: false,
    typedArraysIncluded: false,
    gpuResourcesIncluded: false,
    deviceIncluded: false,
    resultSizePopulationIndependent: true
  };
}

function createHarness({
  orchestrationDecision = 'match',
  adapterAvailable = true,
  requestAdapterError = null,
  requestDeviceError = null,
  orchestratorError = null,
  destroyError = null,
  beforeReturn = null,
  mutateOrchestrationResult = null
} = {}) {
  const stats = {
    adapterRequests: 0,
    deviceRequests: 0,
    orchestratorCalls: 0,
    destroyCalls: 0,
    inputs: [],
    devicesUsedByChunks: []
  };
  const device = {
    fixture: 'fresh-diagnostic-device',
    destroy() {
      stats.destroyCalls += 1;
      if (destroyError) throw destroyError;
    }
  };
  const adapter = {
    async requestDevice() {
      stats.deviceRequests += 1;
      if (requestDeviceError) throw requestDeviceError;
      return device;
    }
  };
  const requestAdapter = async () => {
    stats.adapterRequests += 1;
    if (requestAdapterError) throw requestAdapterError;
    return adapterAvailable ? adapter : null;
  };
  const runOrchestrator = async (input) => {
    stats.orchestratorCalls += 1;
    stats.inputs.push(input);
    stats.devicesUsedByChunks = Array.from(
      { length: POPULATION_SEMANTIC_FIXED_CHUNK_COUNT },
      () => input.device
    );
    if (beforeReturn) await beforeReturn(input);
    if (orchestratorError) throw orchestratorError;
    const result = buildOrchestrationResult(orchestrationDecision);
    if (mutateOrchestrationResult) mutateOrchestrationResult(result);
    return result;
  };
  return {
    controller: createPopulationSemanticComparisonController({
      requestAdapter,
      runOrchestrator
    }),
    stats,
    device
  };
}

function inspectBoundedResult(value, path = 'result') {
  assert.equal(ArrayBuffer.isView(value), false, `${path} retained TypedArray`);
  assert.equal(value instanceof ArrayBuffer, false, `${path} retained ArrayBuffer`);
  if (Array.isArray(value)) {
    assert.ok(value.length <= 64, `${path} array is not bounded`);
    value.forEach((item, index) => inspectBoundedResult(item, `${path}[${index}]`));
    return;
  }
  if (value && typeof value === 'object') {
    for (const [key, child] of Object.entries(value)) {
      assert.equal(
        ['raw', 'adapter', 'device', 'buffer', 'gpuResources'].includes(key),
        false,
        `${path}.${key} retained runtime input/resource`
      );
      inspectBoundedResult(child, `${path}.${key}`);
    }
  }
}

const readyContract = buildReadyEvaluationContract();
const raw = buildRawFixture();
assert.equal(readyContract.status, 'ready');
assert.equal(readyContract.contractVersion, PRODUCTION_EVALUATION_INPUT_CONTRACT_VERSION);

// Ready mapping, independent projection copy, diagnostic-device separation,
// bounded result, getter, and no import-time execution.
const matchHarness = createHarness({
  beforeReturn: (input) => {
    assert.equal(input.projectionParams instanceof Float32Array, true);
    assert.notEqual(input.projectionParams, readyContract.projection.values);
    assert.equal(input.projectionParams.length, 44);
    assert.equal(input.projectionParams[8], readyContract.projection.values[8]);
    input.projectionParams[8] = 999;
    assert.equal(readyContract.projection.values[8], 800);
  }
});
assert.equal(matchHarness.stats.adapterRequests, 0);
assert.equal(
  matchHarness.controller.getLastPopulationAlignedSemanticComparisonResult(),
  null
);
const ignoredProductionDevice = { fixture: 'must-not-be-used' };
const match = await matchHarness.controller.runPopulationAlignedSemanticComparison({
  raw,
  productionEvaluationInputContract: readyContract,
  productionDevice: ignoredProductionDevice
});
assert.equal(match.decision, 'match');
assert.equal(match.contractVersion, POPULATION_SEMANTIC_COMPARISON_CONTROLLER_CONTRACT_VERSION);
assert.equal(match.explicitRequest, true);
assert.equal(match.automaticExecution, false);
assert.equal(match.productionDeviceAcceptedAsInput, false);
assert.equal(match.productionDeviceUsed, false);
assert.equal(match.freshDiagnosticDeviceAcquisition.status, 'ready');
assert.equal(match.diagnosticDeviceCleanup.status, 'destroyed');
assert.equal(matchHarness.stats.adapterRequests, 1);
assert.equal(matchHarness.stats.deviceRequests, 1);
assert.equal(matchHarness.stats.orchestratorCalls, 1);
assert.equal(matchHarness.stats.destroyCalls, 1);
assert.equal(matchHarness.stats.inputs[0].raw, raw);
assert.equal(matchHarness.stats.inputs[0].device, matchHarness.device);
assert.equal(matchHarness.stats.inputs[0].device === ignoredProductionDevice, false);
assert.equal(new Set(matchHarness.stats.devicesUsedByChunks).size, 1);
assert.deepEqual(matchHarness.stats.inputs[0].buildConfig, {
  timestamp: 23.2,
  scalingModifier: 1.25,
  sigmaScale: 3,
  prefilterVar: 0.3
});
assert.deepEqual(matchHarness.stats.inputs[0].sceneInputIdentity, {
  sceneResourceIdentity: SCENE_RESOURCE_IDENTITY
});
assert.equal(
  matchHarness.stats.inputs[0].populationContractIdentity.residentStart,
  PRODUCTION_RESIDENT_RANGE_START
);
assert.equal(
  matchHarness.stats.inputs[0].timeIdentity.request.schedulerRequestIdentity,
  'viewer-render-request:9'
);
assert.equal(
  matchHarness.controller.getLastPopulationAlignedSemanticComparisonResult(),
  match
);
assert.equal(Object.isFrozen(match), true);
assert.doesNotThrow(() => JSON.stringify(match));
assert.ok(JSON.stringify(match).length < 100000);
inspectBoundedResult(match);

// Upper/lower hexadecimal SHA text denotes the same already-computed asset hash.
const uppercaseContract = buildReadyEvaluationContract({ assetSha: ASSET_SHA.toUpperCase() });
const uppercaseHarness = createHarness();
const uppercase = await uppercaseHarness.controller
  .runPopulationAlignedSemanticComparison({
    raw,
    productionEvaluationInputContract: uppercaseContract
  });
assert.equal(uppercase.decision, 'match');
assert.equal(raw.assetSha256, ASSET_SHA);

// Complete mismatch remains a successful diagnostic result and preserves all
// eight completed chunks and the bounded representative.
const mismatchHarness = createHarness({ orchestrationDecision: 'mismatch' });
const mismatch = await mismatchHarness.controller
  .runPopulationAlignedSemanticComparison({
    raw,
    productionEvaluationInputContract: readyContract
  });
assert.equal(mismatch.decision, 'mismatch');
assert.equal(mismatch.blockedReasons.length, 0);
assert.equal(mismatch.orchestrationResult.completedChunkCount, 8);
assert.equal(mismatch.orchestrationResult.coverage.processedRecordCount, 524288);
assert.equal(mismatch.orchestrationResult.firstMismatches.length, 1);
assert.equal(mismatchHarness.stats.destroyCalls, 1);

// Contract/range/raw drift blocks before adapter acquisition.
const preflightCases = [
  ['blocked-contract', buildProductionEvaluationInputContract({}), raw],
  ['wrong-range', buildReadyEvaluationContract({
    residentStart: 0,
    residentRecordCount: PRODUCTION_RESIDENT_RANGE_COUNT
  }), raw],
  ['raw-sha', readyContract, buildRawFixture({ sha: 'b'.repeat(64) })],
  ['raw-version', readyContract, { ...raw, version: 1 }],
  ['raw-count', readyContract, { ...raw, N: SCENE_RECORD_COUNT - 1 }],
  ['raw-size', readyContract, { ...raw, assetSizeBytes: ASSET_SIZE - 1 }],
  ['raw-dimension', readyContract, { ...raw, rotationRDim: 3 }]
];
for (const [name, contract, rawInput] of preflightCases) {
  const harness = createHarness();
  const result = await harness.controller.runPopulationAlignedSemanticComparison({
    raw: rawInput,
    productionEvaluationInputContract: contract
  });
  assert.equal(result.decision, 'blocked', name);
  assert.equal(harness.stats.adapterRequests, 0, name);
  assert.equal(harness.stats.orchestratorCalls, 0, name);
  assert.equal(harness.stats.destroyCalls, 0, name);
  assert.equal(result.freshDiagnosticDeviceAcquisition.status, 'not-requested', name);
}

// Adapter absence and device request failure fail closed without orchestration.
const noAdapterHarness = createHarness({ adapterAvailable: false });
const noAdapter = await noAdapterHarness.controller
  .runPopulationAlignedSemanticComparison({
    raw,
    productionEvaluationInputContract: readyContract
  });
assert.equal(noAdapter.decision, 'blocked');
assert.match(noAdapter.reason, /diagnostic-adapter-unavailable/u);
assert.equal(noAdapterHarness.stats.deviceRequests, 0);
assert.equal(noAdapterHarness.stats.destroyCalls, 0);

const adapterExceptionHarness = createHarness({
  requestAdapterError: new Error(`adapter-request-${'z'.repeat(500)}`)
});
const adapterException = await adapterExceptionHarness.controller
  .runPopulationAlignedSemanticComparison({
    raw,
    productionEvaluationInputContract: readyContract
  });
assert.equal(adapterException.decision, 'blocked');
assert.match(adapterException.reason, /diagnostic-adapter-request-failed/u);
assert.ok(adapterException.reason.length < 240);
assert.equal(adapterExceptionHarness.stats.deviceRequests, 0);

const deviceExceptionHarness = createHarness({
  requestDeviceError: new Error(`device-request-${'x'.repeat(500)}`)
});
const deviceException = await deviceExceptionHarness.controller
  .runPopulationAlignedSemanticComparison({
    raw,
    productionEvaluationInputContract: readyContract
  });
assert.equal(deviceException.decision, 'blocked');
assert.match(deviceException.reason, /diagnostic-device-request-failed/u);
assert.ok(deviceException.reason.length < 240);
assert.equal(deviceExceptionHarness.stats.destroyCalls, 0);

// Orchestrator exception/blocked and cleanup failure remain bounded blocked
// results. Acquired devices are destroyed on every path where destruction is
// possible.
const orchestratorExceptionHarness = createHarness({
  orchestratorError: new Error(`orchestrator-${'y'.repeat(500)}`)
});
const orchestratorException = await orchestratorExceptionHarness.controller
  .runPopulationAlignedSemanticComparison({
    raw,
    productionEvaluationInputContract: readyContract
  });
assert.equal(orchestratorException.decision, 'blocked');
assert.match(orchestratorException.reason, /orchestration-exception/u);
assert.ok(orchestratorException.reason.length < 240);
assert.equal(orchestratorExceptionHarness.stats.destroyCalls, 1);

const orchestratorBlockedHarness = createHarness({
  orchestrationDecision: 'blocked'
});
const orchestratorBlocked = await orchestratorBlockedHarness.controller
  .runPopulationAlignedSemanticComparison({
    raw,
    productionEvaluationInputContract: readyContract
  });
assert.equal(orchestratorBlocked.decision, 'blocked');
assert.match(orchestratorBlocked.reason, /orchestration-blocked/u);
assert.equal(orchestratorBlocked.orchestrationResult.decision, 'blocked');
assert.equal(orchestratorBlockedHarness.stats.destroyCalls, 1);

const destroyFailureHarness = createHarness({
  destroyError: new Error('fixture-device-destroy-failure')
});
const destroyFailure = await destroyFailureHarness.controller
  .runPopulationAlignedSemanticComparison({
    raw,
    productionEvaluationInputContract: readyContract
  });
assert.equal(destroyFailure.decision, 'blocked');
assert.match(destroyFailure.reason, /diagnostic-device-destroy-failed/u);
assert.equal(destroyFailure.diagnosticDeviceCleanup.status, 'destroy-failed');
assert.equal(destroyFailureHarness.stats.destroyCalls, 1);

const retainedRuntimeResultHarness = createHarness({
  mutateOrchestrationResult: (result) => {
    result.retainedPayload = new Float32Array(4);
    result.device = { fixture: 'must-not-escape' };
  }
});
const retainedRuntimeResult = await retainedRuntimeResultHarness.controller
  .runPopulationAlignedSemanticComparison({
    raw,
    productionEvaluationInputContract: readyContract
  });
assert.equal(retainedRuntimeResult.decision, 'blocked');
assert.match(retainedRuntimeResult.reason, /orchestration-result-not-bounded/u);
assert.equal(retainedRuntimeResult.orchestrationResult, null);
assert.equal(retainedRuntimeResultHarness.stats.destroyCalls, 1);

// Concurrent requests never acquire a second adapter/device and the transient
// already-running response does not replace the primary final result.
let releaseOrchestrator;
let signalOrchestratorStarted;
const orchestratorStarted = new Promise((resolve) => {
  signalOrchestratorStarted = resolve;
});
const release = new Promise((resolve) => {
  releaseOrchestrator = resolve;
});
const concurrentHarness = createHarness({
  beforeReturn: async () => {
    signalOrchestratorStarted();
    await release;
  }
});
const primaryPromise = concurrentHarness.controller
  .runPopulationAlignedSemanticComparison({
    raw,
    productionEvaluationInputContract: readyContract
  });
await orchestratorStarted;
const duplicate = await concurrentHarness.controller
  .runPopulationAlignedSemanticComparison({
    raw,
    productionEvaluationInputContract: readyContract
  });
assert.equal(duplicate.decision, 'blocked');
assert.equal(duplicate.reason, 'already-running');
assert.equal(concurrentHarness.stats.adapterRequests, 1);
assert.equal(concurrentHarness.stats.deviceRequests, 1);
assert.equal(concurrentHarness.stats.orchestratorCalls, 1);
assert.equal(
  concurrentHarness.controller.getLastPopulationAlignedSemanticComparisonResult(),
  null
);
releaseOrchestrator();
const primary = await primaryPromise;
assert.equal(primary.decision, 'match');
assert.equal(concurrentHarness.stats.destroyCalls, 1);
assert.equal(
  concurrentHarness.controller.getLastPopulationAlignedSemanticComparisonResult(),
  primary
);

// Source-level boundary: import/Viewer startup only creates the controller;
// routing reads exactly raw + completed production snapshot at invocation and
// does not connect to scheduler, capture, RAF, timer, UI, or URL state.
const controllerSource = await readFile(
  new URL(
    '../demo/js/webgpu_population_semantic_comparison_controller.js',
    import.meta.url
  ),
  'utf8'
);
const viewerSource = await readFile(
  new URL('../demo/js/viewer_app_gpu.js', import.meta.url),
  'utf8'
);
for (const forbidden of [
  'latestRenderResult',
  'requestAnimationFrame',
  'setTimeout',
  'localStorage',
  'scheduleRender',
  'captureFrame'
]) assert.equal(controllerSource.includes(forbidden), false, forbidden);
assert.match(
  viewerSource,
  /createPopulationSemanticComparisonController\(\)/u
);
assert.match(
  viewerSource,
  /latestRenderResult\?\.webgpuProductionFrameDataPathContract\s*\?\.productionEvaluationInputContract/u
);
assert.match(viewerSource, /runPopulationAlignedSemanticComparison,/u);
assert.match(
  viewerSource,
  /getLastPopulationAlignedSemanticComparisonResult:/u
);
const routeStart = viewerSource.indexOf(
  'const runPopulationAlignedSemanticComparison = () => {'
);
const routeEnd = viewerSource.indexOf('window.gpuViewerDebug = {', routeStart);
assert.ok(routeStart >= 0 && routeEnd > routeStart);
const routeSource = viewerSource.slice(routeStart, routeEnd);
assert.match(routeSource, /\braw\b/u);
assert.match(routeSource, /productionEvaluationInputContract/u);
assert.doesNotMatch(
  routeSource,
  /scheduleRender|capture|requestAnimationFrame|setTimeout|\bui\b/u
);

console.log('Step121 Impl4 population semantic controller smoke: OK', {
  caseGroups: 16,
  preflightCases: preflightCases.length,
  fixedRange: [PRODUCTION_RESIDENT_RANGE_START, PRODUCTION_RESIDENT_RANGE_END],
  chunkCount: POPULATION_SEMANTIC_FIXED_CHUNK_COUNT,
  recordsPerChunk: POPULATION_SEMANTIC_MAX_CHUNK_RECORDS,
  controllerResultBytes: JSON.stringify(match).length,
  browserApi: [
    'runPopulationAlignedSemanticComparison',
    'getLastPopulationAlignedSemanticComparisonResult'
  ]
});
