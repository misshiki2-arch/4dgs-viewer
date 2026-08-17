import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
  buildNativeWebGpuProductionFrameDataPathContract,
  buildNativeWebGpuProductionTileInputContract
} from '../demo/js/common_4dgs_production_frame_data_contracts.js';
import {
  buildProductionTileReferencePlanFromTileCounts
} from '../demo/js/common_4dgs_production_tile_reference_contracts.js';
import {
  buildProductionGpuExecutionContract,
  resolveProductionGpuExecutionLimits
} from '../demo/js/common_4dgs_production_gpu_execution_contracts.js';
import {
  buildNativeWebGpuProductionTileInputBindGroupEntries,
  buildNativeWebGpuProductionTileInputBindGroupLayoutEntries,
  buildNativeWebGpuProductionTileInputWgslBindings,
  validateNativeWebGpuProductionTileInputBindingContract
} from '../demo/js/common_4dgs_production_tile_input_binding_contracts.js';
import {
  selectActiveProductionResidentWorkset
} from '../demo/js/webgpu_production_workset_owner.js';
import {
  buildNativeWebGpuProductionTileInput
} from '../demo/js/webgpu_production_tile_input.js';
import {
  runWebGpuBackendRuntimeFrame
} from '../demo/js/webgpu_backend_runtime_runner.js';
import {
  executeWebGpuBackendViewerFrame
} from '../demo/js/webgpu_backend_viewer_frame_executor.js';

const raw = {
  count: 10,
  xyzDim: 3,
  opacityDim: 1,
  xyz: new Float32Array(30),
  opacity: new Float32Array(10)
};
const device = {
  limits: {
    maxStorageBufferBindingSize: 128 * 1024 * 1024,
    maxBufferSize: 128 * 1024 * 1024
  }
};
const workset = selectActiveProductionResidentWorkset({
  raw,
  device,
  canvasWidth: 1280,
  canvasHeight: 720
});
assert.equal(workset.contract.residentWorksetReady, true);
assert.equal(workset.contract.residentRecordCount, 10);
assert.equal(workset.contract.diagnosticMaxRecordsUsed, false);
assert.equal(workset.contract.diagnosticCandidateSourceUsed, false);
assert.equal(workset.contract.nonResidentRecordsExplicit, true);
assert.equal(
  workset.contract.tileReferenceCapacityCoupledToRecordSelection,
  false
);
assert.equal(
  workset.contract.overflowPolicy,
  'fail-closed-before-compositor-promotion'
);

const tileInputContract = buildNativeWebGpuProductionTileInputContract({
  sourceWorksetResourceIdentity: workset.contract.resourceIdentity,
  sourceStateResourceIdentity: 'state-1',
  resourceIdentity: 'tile-input-1',
  recordCount: 10,
  dispatchSubmitted: true,
  productionReadbackPerformed: false,
  javascriptVisibleSamplesMaterialized: false
});
assert.equal(tileInputContract.tileAwareRenderInputReady, true);

const fakeBindingResources = {
  statePositions: { label: 'state-positions' },
  renderAttributes: { label: 'render-attributes' },
  footprintPayload: { label: 'footprint-payload' },
  projectionParams: { label: 'projection-params' },
  tileInputs: { label: 'tile-inputs' },
  params: { label: 'params' }
};
const tileInputShaderBindings = `
struct Params {
  count: u32,
};
${buildNativeWebGpuProductionTileInputWgslBindings()}
`;
const tileInputLayoutEntries =
  buildNativeWebGpuProductionTileInputBindGroupLayoutEntries({
    computeVisibility: 4
  });
const tileInputBindGroupEntries =
  buildNativeWebGpuProductionTileInputBindGroupEntries(fakeBindingResources);
const bindingContract =
  validateNativeWebGpuProductionTileInputBindingContract({
    shaderSource: tileInputShaderBindings,
    layoutEntries: tileInputLayoutEntries,
    bindGroupEntries: tileInputBindGroupEntries
  });
assert.equal(bindingContract.ready, true);
assert.deepEqual(bindingContract.expectedBindings, [0, 1, 2, 4, 5, 6]);
assert.equal(bindingContract.expectedBindings.includes(3), false);

const extraBindingContract =
  validateNativeWebGpuProductionTileInputBindingContract({
    shaderSource: tileInputShaderBindings,
    layoutEntries: tileInputLayoutEntries,
    bindGroupEntries: [
      ...tileInputBindGroupEntries,
      { binding: 3, resource: { buffer: { label: 'stale-candidates' } } }
    ]
  });
assert.equal(extraBindingContract.ready, false);
assert.deepEqual(extraBindingContract.extraBindings.descriptor, [3]);

const missingBindingContract =
  validateNativeWebGpuProductionTileInputBindingContract({
    shaderSource: tileInputShaderBindings,
    layoutEntries: tileInputLayoutEntries,
    bindGroupEntries: tileInputBindGroupEntries.filter(
      ({ binding }) => binding !== 6
    )
  });
assert.equal(missingBindingContract.ready, false);
assert.deepEqual(missingBindingContract.missingBindings.descriptor, [6]);

const duplicateBindingContract =
  validateNativeWebGpuProductionTileInputBindingContract({
    shaderSource: tileInputShaderBindings,
    layoutEntries: tileInputLayoutEntries,
    bindGroupEntries: [
      ...tileInputBindGroupEntries,
      tileInputBindGroupEntries[0]
    ]
  });
assert.equal(duplicateBindingContract.ready, false);
assert.deepEqual(duplicateBindingContract.duplicateBindings.descriptor, [0]);

const incompatibleLayoutContract =
  validateNativeWebGpuProductionTileInputBindingContract({
    shaderSource: tileInputShaderBindings,
    layoutEntries: tileInputLayoutEntries.map((entry) =>
      entry.binding === 0
        ? { ...entry, buffer: { type: 'storage' } }
        : entry
    ),
    bindGroupEntries: tileInputBindGroupEntries
  });
assert.equal(incompatibleLayoutContract.ready, false);
assert.deepEqual(
  incompatibleLayoutContract.layoutDefinitionMismatches,
  ['statePositions']
);
assert.throws(
  () => buildNativeWebGpuProductionTileInputBindGroupEntries({}),
  /Missing production tile-input binding resource/
);

const previousGpuBufferUsage = globalThis.GPUBufferUsage;
const previousGpuShaderStage = globalThis.GPUShaderStage;
const capturedTileInputGpuDescriptors = {};
globalThis.GPUBufferUsage = { STORAGE: 1, UNIFORM: 2 };
globalThis.GPUShaderStage = { COMPUTE: 4 };
try {
  const createFakeBuffer = (descriptor) => {
    const mappedRange = new ArrayBuffer(descriptor.size);
    return {
      descriptor,
      getMappedRange: () => mappedRange,
      unmap: () => {},
      destroy: () => {}
    };
  };
  const fakePass = {
    setPipeline: () => {},
    setBindGroup: () => {},
    dispatchWorkgroups: () => {},
    end: () => {}
  };
  const fakeTileInputDevice = {
    createBuffer: createFakeBuffer,
    createShaderModule: (descriptor) => {
      capturedTileInputGpuDescriptors.shaderModule = descriptor;
      return { descriptor };
    },
    createBindGroupLayout: (descriptor) => {
      capturedTileInputGpuDescriptors.bindGroupLayout = descriptor;
      return { descriptor };
    },
    createPipelineLayout: (descriptor) => {
      capturedTileInputGpuDescriptors.pipelineLayout = descriptor;
      return { descriptor };
    },
    createComputePipeline: (descriptor) => {
      capturedTileInputGpuDescriptors.pipeline = descriptor;
      return { descriptor };
    },
    createBindGroup: (descriptor) => {
      capturedTileInputGpuDescriptors.bindGroup = descriptor;
      return { descriptor };
    },
    createCommandEncoder: () => ({
      beginComputePass: () => fakePass,
      finish: () => ({ label: 'fake-command-buffer' })
    }),
    queue: {
      submit: () => {},
      onSubmittedWorkDone: async () => {}
    }
  };
  const fakeStatePositionBuffer = { label: 'state-position-buffer' };
  const fakeRenderAttributeBuffer = { label: 'render-attribute-buffer' };
  const fakeFootprintPayloadBuffer = { label: 'footprint-payload-buffer' };
  const tileInputResult = await buildNativeWebGpuProductionTileInput({
    device: fakeTileInputDevice,
    workset: {
      candidateIndices: new Uint32Array([0, 1]),
      contract: { resourceIdentity: 'workset-smoke' }
    },
    stateGpuResources: {
      resourceIdentity: 'state-smoke',
      statePositionBuffer: fakeStatePositionBuffer,
      renderAttributeBuffer: fakeRenderAttributeBuffer,
      footprintPayloadBuffer: fakeFootprintPayloadBuffer
    },
    projectionParams: new Float32Array(48)
  });
  assert.equal(tileInputResult.contract.tileAwareRenderInputReady, true);
  assert.deepEqual(
    capturedTileInputGpuDescriptors.bindGroupLayout.entries.map(
      ({ binding }) => binding
    ),
    [0, 1, 2, 4, 5, 6]
  );
  assert.deepEqual(
    capturedTileInputGpuDescriptors.bindGroup.entries.map(
      ({ binding }) => binding
    ),
    [0, 1, 2, 4, 5, 6]
  );
  assert.equal(
    capturedTileInputGpuDescriptors.pipeline.layout.descriptor,
    capturedTileInputGpuDescriptors.pipelineLayout
  );
  assert.equal(
    capturedTileInputGpuDescriptors.shaderModule.code.includes('candidates'),
    false
  );
  assert.equal(
    capturedTileInputGpuDescriptors.shaderModule.code.includes('@binding(3)'),
    false
  );
} finally {
  if (previousGpuBufferUsage === undefined) {
    delete globalThis.GPUBufferUsage;
  } else {
    globalThis.GPUBufferUsage = previousGpuBufferUsage;
  }
  if (previousGpuShaderStage === undefined) {
    delete globalThis.GPUShaderStage;
  } else {
    globalThis.GPUShaderStage = previousGpuShaderStage;
  }
}

const readyContract = buildNativeWebGpuProductionFrameDataPathContract({
  worksetContract: workset.contract,
  stateResourceIdentity: 'state-1',
  attributeResourceIdentity: 'state-1:attributes',
  footprintResourceIdentity: 'state-1:footprint',
  tileInputResourceIdentity: 'tile-input-1',
  tileListInputResourceIdentity: 'tile-list-1',
  compositorInputResourceIdentity: 'tile-list-1',
  stateRecordCount: 10,
  tileInputRecordCount: 10,
  tileReferenceCapacityContract:
    buildProductionTileReferencePlanFromTileCounts({
      tileReferenceCounts: [5, 5],
      allocatedReferenceCapacity: 16,
      recordCount: 10
    }).contract,
  boundedExecutionContract: buildProductionGpuExecutionContract({
    limits: resolveProductionGpuExecutionLimits({ device, tileCount: 2 }),
    inputRecordCount: 10,
    inputReferenceCount: 10,
    completedRecordCount: 10,
    completedReferenceCount: 10,
    maximumRecordsInSubmission: 10,
    maximumReferencesInSubmission: 10,
    maximumReferencesPerPixelInvocation: 10,
    gpuResourceLineageMaintained: true,
    recordReferenceCapacitySeparated: true,
    silentDropAllowed: false,
    schedulerContinuationUsed: false,
    allStagesCompleted: true
  }),
  terminalExecutionPlanObserver: {
    schemaVersion: 'phase3-production-tile-execution-plan-terminal-observer-v1',
    evidenceRole: 'terminal-post-production-submission-observer',
    productionControlInput: false,
    rawPlanWordsPublished: false,
    observerReady: true,
    planIdentity: 7,
    requiredReferenceCount: 10,
    scatteredReferenceCount: 10,
    sortedReferenceCount: 10,
    compositedReferenceCount: 10
  },
  cpuReferenceUsedAsProductionInput: false,
  diagnosticReadbackUsedAsProductionInput: false,
  javascriptVisibleSamplesUsedAsProductionInput: false,
  diagnosticMaxRecordsUsedAsProductionLimit: false,
  gpuResourceLineagePreserved: true,
  capacityOverflowDetected: false,
  capacityOverflowFailClosed: true,
  silentDropAllowed: false,
  compositorSubmitted: true
});
assert.equal(readyContract.nativeProductionFrameDataPathReady, true);
assert.equal(readyContract.diagnosticIndependent, true);
assert.equal(readyContract.countsMatch, true);
assert.equal(readyContract.tileReferenceCapacityReady, true);
assert.equal(readyContract.terminalExecutionPlanObserver.observerReady, true);
assert.equal(readyContract.terminalExecutionPlanObserver.planIdentity, 7);
assert.equal(
  readyContract.terminalExecutionPlanObserver.sortedReferenceCount,
  10
);
assert.equal(
  readyContract.terminalExecutionPlanObserver.productionControlInput,
  false
);
const serializedReadyContract = JSON.parse(JSON.stringify(readyContract));
assert.equal(
  serializedReadyContract.terminalExecutionPlanObserver.schemaVersion,
  'phase3-production-tile-execution-plan-terminal-observer-v1'
);
assert.equal(
  serializedReadyContract.terminalExecutionPlanObserver.sortedReferenceCount,
  10
);
assert.equal(
  readyContract.tileReferenceCapacityContract.requiredReferenceCount,
  10
);

const overflowContract = buildNativeWebGpuProductionFrameDataPathContract({
  status: 'blocked',
  worksetContract: workset.contract,
  stateResourceIdentity: 'state-1',
  attributeResourceIdentity: 'state-1:attributes',
  footprintResourceIdentity: 'state-1:footprint',
  tileInputResourceIdentity: 'tile-input-1',
  tileListInputResourceIdentity: 'tile-list-1',
  compositorInputResourceIdentity: 'tile-list-1',
  stateRecordCount: 10,
  tileInputRecordCount: 10,
  tileReferenceCapacityContract:
    buildProductionTileReferencePlanFromTileCounts({
      tileReferenceCounts: [10, 10],
      allocatedReferenceCapacity: 8,
      recordCount: 10
    }).contract,
  gpuResourceLineagePreserved: true,
  capacityOverflowDetected: true,
  capacityOverflowFailClosed: true,
  silentDropAllowed: false,
  compositorSubmitted: false,
  reason: 'native-production-tile-list-capacity-overflow-fail-closed'
});
assert.equal(overflowContract.nativeProductionFrameDataPathReady, false);
assert.equal(overflowContract.capacityOverflowFailClosed, true);
assert.equal(overflowContract.silentDropAllowed, false);

const tileCompositorContract = {
  contractVersion: 'phase3-step85-webgpu-tile-list-compositor-v1',
  compositorPassSubmitted: true,
  tileCompositorOutputPresentedToCurrentTexture: true,
  compositorCurrentTextureRenderPassSubmitted: true,
  compositorCurrentTextureReadbackCompleted: true,
  compositorCurrentTextureReadbackNonZero: true,
  outputFormat: 'rgba8unorm'
};
const syntheticProductionFrame = {
  webgpuProductionFrameDataPathContract: readyContract,
  webgpuTileListCompositorContract: tileCompositorContract
};
const runtimeFrame = await runWebGpuBackendRuntimeFrame({
  requestedBackendMode: 'webgpu-exclusive',
  allowViewerCanvasPresentation: true,
  enableViewerLoopHook: true,
  invocationSource: 'renderCurrentFrame-step118-smoke',
  backendImplementationKind: 'webgpu-tile-compositor-frame-implementation',
  viewerCanvasState: { webgl2FrameLifecycleSuppressed: true },
  runBackendFrame: async () => syntheticProductionFrame
});
assert.equal(runtimeFrame.summary.runtimeRunnerReady, true);
assert.equal(
  runtimeFrame.summary.canonicalPresentSummary.selectedSourceKind,
  'webgpu-production-tile-compositor'
);

const executorFrame = await executeWebGpuBackendViewerFrame({
  requestedBackendMode: 'webgpu-exclusive',
  allowViewerCanvasPresentation: true,
  enableViewerLoopHook: true,
  invocationSource: 'renderCurrentFrame-step118-smoke',
  backendImplementationKind: 'webgpu-tile-compositor-frame-implementation',
  viewerCanvasState: { webgl2FrameLifecycleSuppressed: true },
  integrationBoundary: {
    integrationBoundaryReady: true,
    validationSummary: { webgl2HybridRenderingPrevented: true }
  },
  runBackendFrame: async () => syntheticProductionFrame
});
assert.equal(executorFrame.summary.executorReady, true);
assert.equal(
  executorFrame.summary.viewerFramePresentationPassContract
    .viewerFramePresentationPassReady,
  true
);

const [runnerSource, tileInputSource, evaluatorSource, layoutSource, viewerSource] =
  await Promise.all([
    readFile(
      new URL('../demo/js/webgpu_production_frame_data_path.js', import.meta.url),
      'utf8'
    ),
    readFile(
      new URL('../demo/js/webgpu_production_tile_input.js', import.meta.url),
      'utf8'
    ),
    readFile(
      new URL('../demo/js/webgpu_4d_state_evaluator.js', import.meta.url),
      'utf8'
    ),
    readFile(
      new URL('../demo/js/webgpu_gpu_owned_tile_list_layout.js', import.meta.url),
      'utf8'
    ),
    readFile(
      new URL('../demo/js/viewer_app_gpu.js', import.meta.url),
      'utf8'
    )
  ]);
assert.equal(runnerSource.includes('webgpu_visible_record_dry_run_runtime'), false);
assert.equal(runnerSource.includes('buildCpuReferenceRecords'), false);
assert.equal(runnerSource.includes("readbackPolicy: 'none'"), true);
assert.equal(runnerSource.includes('javascriptVisibleSamplesUsedAsProductionInput: false'), true);
assert.equal(
  runnerSource.includes(
    'terminalExecutionPlanObserver: compositor.executionPlanObserver ?? null'
  ),
  true
);
assert.equal(tileInputSource.includes("layout: 'auto'"), false);
assert.equal(tileInputSource.includes('candidateBuffer'), false);
assert.equal(tileInputSource.includes('binding: 3'), false);
assert.equal(
  tileInputSource.includes('buildNativeWebGpuProductionTileInputWgslBindings'),
  true
);
assert.equal(
  tileInputSource.includes(
    'buildNativeWebGpuProductionTileInputBindGroupLayoutEntries'
  ),
  true
);
assert.equal(
  tileInputSource.includes('buildNativeWebGpuProductionTileInputBindGroupEntries'),
  true
);
assert.equal(evaluatorSource.includes("const productionGpuOnly = readbackPolicy === 'none'"), true);
assert.equal(layoutSource.includes('tileInputResource = null'), true);
assert.equal(layoutSource.includes('capacityOverflowFailClosed: true'), true);
assert.equal(layoutSource.includes('executionPlanResources.planBuffer'), true);
assert.equal(
  viewerSource.includes('runNativeWebGpuProductionFrameFromViewerState'),
  true
);
assert.equal(
  viewerSource.includes('async function captureWebGpuVisibleRecordDryRunDebug'),
  true
);

console.log('Step118 native production frame data path smoke: OK');
