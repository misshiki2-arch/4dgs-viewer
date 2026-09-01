import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
  PRODUCTION_CANDIDATE_ATTRIBUTE_BYTE_STRIDE,
  PRODUCTION_CANDIDATE_ATTRIBUTE_FIELD_OFFSETS,
  PRODUCTION_CANDIDATE_ATTRIBUTE_FLOAT_STRIDE,
  PRODUCTION_CANDIDATE_ATTRIBUTE_INPUT_LAYOUT_VERSION,
  PRODUCTION_CANDIDATE_ATTRIBUTE_VEC4_STRIDE,
  PRODUCTION_DEGREE2_SH_COEFFICIENT_FLOAT_OFFSETS,
  PRODUCTION_EVALUATOR_STORAGE_BINDING_COUNT,
  buildProductionCandidateAttributeInput,
  buildProductionDegree2SpatialShWgsl,
  productionCandidateAttributeByteLength
} from '../demo/js/common_4dgs_candidate_attribute_input_contracts.js';
import {
  cameraWorldPositionFromProjectionParams
} from '../demo/js/common_4dgs_projection_contracts.js';
import {
  WEBGPU_GAUSSIAN_ATTRIBUTE_EVALUATION_CONTRACT_VERSION
} from '../demo/js/common_4dgs_record_contracts.js';
import {
  buildWebGpu4DStatePositionsForCandidates
} from '../demo/js/webgpu_4d_state_evaluator.js';

const previousGpuBufferUsage = globalThis.GPUBufferUsage;
const previousGpuMapMode = globalThis.GPUMapMode;
globalThis.GPUBufferUsage = {
  STORAGE: 1,
  UNIFORM: 2,
  COPY_SRC: 4,
  COPY_DST: 8,
  MAP_READ: 16
};
globalThis.GPUMapMode = { READ: 1 };

const DEVICE_LIMITS = Object.freeze({
  maxStorageBufferBindingSize: 1_073_741_824,
  maxBufferSize: 1_073_741_824,
  maxStorageBuffersPerShaderStage: 8
});

function projectionParams({
  viewRows = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]
  ]
} = {}) {
  const values = new Float32Array(44);
  values.set([1, 1280, 720, 0, 1, 1, 1, 0, 800, 800, 639.5, 359.5]);
  for (let row = 0; row < 3; row += 1) {
    values.set(viewRows[row], 12 + row * 4);
  }
  values.set([0, 0, 0, 1], 24);
  values.set([1, 0, 0, 0], 28);
  values.set([0, 1, 0, 0], 32);
  values.set([0, 0, 1, 0], 36);
  values.set([0, 0, 0, 1], 40);
  return values;
}

function rawFixture(recordCount = 1) {
  return {
    N: recordCount,
    activeShDegree: 2,
    activeShDegreeT: 2,
    xyz: new Float32Array(recordCount * 3),
    xyzDim: 3,
    t: new Float32Array(recordCount),
    tDim: 1,
    scale_t: new Float32Array(recordCount).fill(1),
    scaleTDim: 1,
    f_dc: new Float32Array(recordCount * 3),
    fdcDim: 3,
    f_rest: new Float32Array(recordCount * 45),
    frestDim: 45,
    scale_xyz: new Float32Array(recordCount * 3).fill(1),
    scaleXYZDim: 3,
    rotation: Float32Array.from(
      { length: recordCount * 4 },
      (_, index) => index % 4 === 0 ? 1 : 0
    ),
    rotationDim: 4,
    rotation_r: Float32Array.from(
      { length: recordCount * 4 },
      (_, index) => index % 4 === 0 ? 1 : 0
    ),
    rotationRDim: 4
  };
}

function candidateXyzOpacity(recordCount = 1) {
  const values = new Float32Array(recordCount * 4);
  for (let row = 0; row < recordCount; row += 1) {
    values.set([1, 2, 3, 0], row * 4);
  }
  return values;
}

function commonBuildInput(overrides = {}) {
  const raw = overrides.raw ?? rawFixture(1);
  const candidateIndices = overrides.candidateIndices ?? new Uint32Array([0]);
  return {
    raw,
    candidateIndices,
    rawXyzOpacity:
      overrides.rawXyzOpacity ?? candidateXyzOpacity(candidateIndices.length),
    projectionParams: overrides.projectionParams ?? projectionParams(),
    deviceLimits: overrides.deviceLimits ?? DEVICE_LIMITS,
    sourceWorksetResourceIdentity: 'impl3-focused-workset',
    resourceIdentity: 'impl3-focused-candidate-attribute-resource',
    resourceOwnership:
      'production-evaluator-input-destroyed-after-submitted-work-completion'
  };
}

function createFakeDevice({ limits = DEVICE_LIMITS, failMap = false } = {}) {
  const buffers = [];
  const bindGroups = [];
  let shaderSource = null;
  let shaderModuleCount = 0;
  const device = {
    limits,
    createBuffer(descriptor) {
      const bytes = new ArrayBuffer(descriptor.size);
      const buffer = {
        descriptor,
        bytes,
        destroyed: false,
        getMappedRange: () => bytes,
        mapAsync: async () => {
          if (failMap) throw new Error('intentional-map-failure');
        },
        unmap: () => {},
        destroy() { this.destroyed = true; }
      };
      buffers.push(buffer);
      return buffer;
    },
    createShaderModule({ code }) {
      shaderModuleCount += 1;
      shaderSource = code;
      return { code };
    },
    createComputePipeline: () => ({
      getBindGroupLayout: () => ({ label: 'impl3-focused-layout' })
    }),
    createBindGroup(descriptor) {
      bindGroups.push(descriptor);
      return { descriptor };
    },
    createCommandEncoder: () => ({
      beginComputePass: () => ({
        setPipeline: () => {},
        setBindGroup: () => {},
        dispatchWorkgroups: () => {},
        end: () => {}
      }),
      copyBufferToBuffer: () => {},
      finish: () => ({ label: 'impl3-focused-command-buffer' })
    }),
    queue: {
      submit: () => {},
      onSubmittedWorkDone: async () => {}
    }
  };
  return {
    device,
    buffers,
    bindGroups,
    getShaderSource: () => shaderSource,
    getShaderModuleCount: () => shaderModuleCount
  };
}

function f32(value) {
  return Math.fround(value);
}

function sourceOrderedOneHotValue(shIndex, direction) {
  const [x, y, z] = direction.map(f32);
  const xx = f32(x * x);
  const yy = f32(y * y);
  const zz = f32(z * z);
  const scales = [
    f32(0.28209479177387814),
    f32(f32(-0.4886025119029199) * y),
    f32(f32(0.4886025119029199) * z),
    f32(f32(-0.4886025119029199) * x),
    f32(f32(f32(1.0925484305920792) * x) * y),
    f32(f32(f32(-1.0925484305920792) * y) * z),
    f32(f32(0.31539156525252005) * f32(f32(2 * zz) - xx - yy)),
    f32(f32(f32(-1.0925484305920792) * x) * z),
    f32(f32(0.5462742152960396) * f32(xx - yy))
  ];
  return f32(Math.max(f32(scales[shIndex] + f32(0.5)), 0));
}

function blockedReason(overrides) {
  const result = buildProductionCandidateAttributeInput(commonBuildInput(overrides));
  assert.equal(result.ready, false);
  return result.contract.blockedReasons;
}

try {
  assert.equal(
    PRODUCTION_CANDIDATE_ATTRIBUTE_INPUT_LAYOUT_VERSION,
    'phase3-step122-production-candidate-attribute-input-layout-v1'
  );
  assert.equal(PRODUCTION_CANDIDATE_ATTRIBUTE_FLOAT_STRIDE, 32);
  assert.equal(PRODUCTION_CANDIDATE_ATTRIBUTE_VEC4_STRIDE, 8);
  assert.equal(PRODUCTION_CANDIDATE_ATTRIBUTE_BYTE_STRIDE, 128);
  assert.deepEqual(PRODUCTION_CANDIDATE_ATTRIBUTE_FIELD_OFFSETS, {
    fDcRgb: { floatOffset: 0, floatCount: 3 },
    meanScale: { floatOffset: 3, floatCount: 1 },
    scaleXyz: { floatOffset: 4, floatCount: 3 },
    sourceCode: { floatOffset: 7, floatCount: 1 },
    fRestDegree2: { floatOffset: 8, floatCount: 24 }
  });
  assert.deepEqual(
    PRODUCTION_DEGREE2_SH_COEFFICIENT_FLOAT_OFFSETS,
    [0, 8, 11, 14, 17, 20, 23, 26, 29]
  );
  assert.equal(productionCandidateAttributeByteLength(65_536), 8 * 1024 * 1024);
  assert.equal(productionCandidateAttributeByteLength(524_288), 64 * 1024 * 1024);

  const packingRaw = rawFixture(1);
  packingRaw.f_dc.set([10, 11, 12]);
  packingRaw.scale_xyz.set([2, 3, 4]);
  for (let index = 0; index < 45; index += 1) packingRaw.f_rest[index] = index + 20;
  const packed = buildProductionCandidateAttributeInput(
    commonBuildInput({ raw: packingRaw })
  );
  assert.equal(packed.ready, true);
  assert.deepEqual(Array.from(packed.data.slice(0, 3)), [10, 11, 12]);
  assert.equal(packed.data[3], 3);
  assert.deepEqual(Array.from(packed.data.slice(4, 8)), [2, 3, 4, 111]);
  assert.deepEqual(
    Array.from(packed.data.slice(8, 32)),
    Array.from({ length: 24 }, (_, index) => index + 20)
  );
  assert.deepEqual(Array.from(packed.data.slice(11, 14)), [23, 24, 25]);
  assert.deepEqual(Array.from(packed.data.slice(23, 26)), [35, 36, 37]);
  assert.equal(packed.contract.deviceLimitPreflight.storageBindingSizeReady, true);
  assert.equal(packed.contract.deviceLimitPreflight.bufferSizeReady, true);
  assert.equal(packed.contract.deviceLimitPreflight.storageBindingCountReady, true);
  assert.equal(packed.contract.storageBindingCount, 8);
  assert.equal(packed.contract.sourceWorksetResourceIdentity, 'impl3-focused-workset');

  const translatedProjection = projectionParams({
    viewRows: [
      [0, -1, 0, 3],
      [1, 0, 0, -2],
      [0, 0, 1, -4]
    ]
  });
  assert.deepEqual(
    cameraWorldPositionFromProjectionParams(translatedProjection),
    [2, 3, 4]
  );

  const oneHotRaw = rawFixture(9);
  for (let shIndex = 0; shIndex < 9; shIndex += 1) {
    if (shIndex === 0) {
      oneHotRaw.f_dc[shIndex * oneHotRaw.fdcDim] = 1;
    } else {
      oneHotRaw.f_rest[
        shIndex * oneHotRaw.frestDim + (shIndex - 1) * 3
      ] = 1;
    }
  }
  const oneHotPacked = buildProductionCandidateAttributeInput(commonBuildInput({
    raw: oneHotRaw,
    candidateIndices: Uint32Array.from({ length: 9 }, (_, index) => index),
    rawXyzOpacity: candidateXyzOpacity(9)
  }));
  assert.equal(oneHotPacked.ready, true);
  for (let shIndex = 0; shIndex < 9; shIndex += 1) {
    const recordBase = shIndex * PRODUCTION_CANDIDATE_ATTRIBUTE_FLOAT_STRIDE;
    const coefficientBase =
      recordBase + PRODUCTION_DEGREE2_SH_COEFFICIENT_FLOAT_OFFSETS[shIndex];
    assert.deepEqual(
      Array.from(oneHotPacked.data.slice(coefficientBase, coefficientBase + 3)),
      [1, 0, 0],
      `sh[${shIndex}] must retain its coefficient-major RGB triplet`
    );
  }

  const directionLength = Math.hypot(1, 2, 3);
  const direction = [1, 2, 3].map((value) => f32(value / directionLength));
  const oneHotValues = Array.from({ length: 9 }, (_, shIndex) =>
    sourceOrderedOneHotValue(shIndex, direction)
  );
  assert.ok(oneHotValues[0] > 0.5);
  assert.ok(oneHotValues[1] < 0.5);
  assert.ok(oneHotValues[2] > 0.5);
  assert.ok(oneHotValues[3] < 0.5);
  assert.ok(oneHotValues[4] > 0.5);
  assert.ok(oneHotValues[5] < 0.5, 'sh[5] yz must use the negative C2 sign');
  assert.ok(oneHotValues[6] > 0.5);
  assert.ok(oneHotValues[7] < 0.5, 'sh[7] xz must use the negative C2 sign');
  assert.ok(oneHotValues[8] < 0.5);
  assert.ok(sourceOrderedOneHotValue(0, direction) < 1);
  const aboveOne = f32(f32(f32(4) * f32(0.28209479177387814)) + f32(0.5));
  assert.ok(aboveOne > 1, 'lower-only clamp must preserve valid RGB above 1');

  const wgsl = buildProductionDegree2SpatialShWgsl();
  assert.match(wgsl, /let floatOffset = 8u \+ \(coefficientIndex - 1u\) \* 3u;/);
  assert.match(wgsl, /const SH_C2_1: f32 = -1\.0925484305920792;/);
  assert.match(wgsl, /const SH_C2_3: f32 = -1\.0925484305920792;/);
  assert.match(wgsl, /originalPosition - cameraWorldPositionFromProjectionParams\(\)/);
  assert.match(wgsl, /return max\(result \+ vec3f\(0\.5\), vec3f\(0\.0\)\);/);
  assert.doesNotMatch(wgsl, /min\(|clamp\(/);
  assert.doesNotMatch(wgsl, /timestamp|temporal|timeScale/);

  const unsupportedDegreeRaw = rawFixture(1);
  unsupportedDegreeRaw.activeShDegree = 1;
  assert.ok(blockedReason({ raw: unsupportedDegreeRaw }).includes(
    'active-spatial-sh-degree-unsupported'
  ));
  const invalidTemporalDegreeRaw = rawFixture(1);
  invalidTemporalDegreeRaw.activeShDegreeT = -1;
  assert.ok(blockedReason({ raw: invalidTemporalDegreeRaw }).includes(
    'active-temporal-sh-degree-invalid'
  ));
  const shortFRestRaw = rawFixture(1);
  shortFRestRaw.f_rest = new Float32Array(23);
  assert.ok(blockedReason({ raw: shortFRestRaw }).includes(
    'f-rest-array-missing-or-short'
  ));
  const incompleteFRestRaw = rawFixture(1);
  incompleteFRestRaw.frestDim = 23;
  assert.ok(blockedReason({ raw: incompleteFRestRaw }).includes(
    'f-rest-degree2-dimension-incomplete'
  ));
  const nonfiniteDcRaw = rawFixture(1);
  nonfiniteDcRaw.f_dc[0] = Number.NaN;
  assert.ok(blockedReason({ raw: nonfiniteDcRaw }).includes(
    'f-dc-coefficient-nonfinite'
  ));
  const nonfiniteRestRaw = rawFixture(1);
  nonfiniteRestRaw.f_rest[5] = Number.POSITIVE_INFINITY;
  assert.ok(blockedReason({ raw: nonfiniteRestRaw }).includes(
    'f-rest-degree2-coefficient-nonfinite'
  ));
  assert.ok(blockedReason({ candidateIndices: new Uint32Array([1]) }).includes(
    'candidate-src-index-out-of-range'
  ));
  const invalidProjection = projectionParams();
  invalidProjection[12] = Number.NaN;
  assert.ok(blockedReason({ projectionParams: invalidProjection }).includes(
    'projection-camera-world-position-invalid'
  ));
  assert.ok(blockedReason({
    deviceLimits: { ...DEVICE_LIMITS, maxStorageBufferBindingSize: 127 }
  }).includes('candidate-attribute-storage-binding-size-exceeded'));
  assert.ok(blockedReason({
    deviceLimits: { ...DEVICE_LIMITS, maxBufferSize: 127 }
  }).includes('candidate-attribute-buffer-size-exceeded'));
  assert.ok(blockedReason({
    deviceLimits: { ...DEVICE_LIMITS, maxStorageBuffersPerShaderStage: 7 }
  }).includes('storage-buffer-binding-limit-insufficient'));

  const failClosedFake = createFakeDevice({
    limits: { ...DEVICE_LIMITS, maxBufferSize: 127 }
  });
  const failClosedResult = await buildWebGpu4DStatePositionsForCandidates({
    device: failClosedFake.device,
    raw: rawFixture(1),
    candidateIndices: new Uint32Array([0]),
    rawXyzOpacity: candidateXyzOpacity(1),
    buildConfig: { timestamp: 0, scalingModifier: 1, sigmaScale: 1 },
    projectionParams: projectionParams(),
    readbackPolicy: 'none',
    keepGpuResources: true
  });
  assert.equal(failClosedResult.gaussianAttributeEvaluationContract.status, 'blocked');
  assert.equal(failClosedFake.buffers.length, 0);
  assert.equal(failClosedFake.getShaderModuleCount(), 0);

  const productionFake = createFakeDevice();
  const productionResult = await buildWebGpu4DStatePositionsForCandidates({
    device: productionFake.device,
    raw: rawFixture(1),
    candidateIndices: new Uint32Array([0]),
    rawXyzOpacity: candidateXyzOpacity(1),
    buildConfig: { timestamp: 0, scalingModifier: 1, sigmaScale: 1 },
    projectionParams: projectionParams(),
    readbackPolicy: 'none',
    keepGpuResources: true,
    sourceWorksetResourceIdentity: 'impl3-production-workset'
  });
  assert.equal(productionResult.productionReadbackPerformed, false);
  assert.equal(
    productionResult.gaussianAttributeEvaluationContract.contractVersion,
    WEBGPU_GAUSSIAN_ATTRIBUTE_EVALUATION_CONTRACT_VERSION
  );
  assert.equal(
    productionResult.gaussianAttributeEvaluationContract.spatialShDegree,
    2
  );
  assert.equal(
    productionResult.gaussianAttributeEvaluationContract.fullGaussianAttributeEvaluationInWgsl,
    false
  );
  assert.deepEqual(
    productionFake.bindGroups[0].entries.map((entry) => entry.binding),
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
  );
  const attributeInputBuffer = productionFake.bindGroups[0].entries.find(
    (entry) => entry.binding === 4
  ).resource.buffer;
  assert.equal(attributeInputBuffer.descriptor.size, 128);
  assert.equal(attributeInputBuffer.destroyed, true);
  for (const binding of [2, 5, 6]) {
    assert.equal(
      productionFake.bindGroups[0].entries.find((entry) => entry.binding === binding)
        .resource.buffer.destroyed,
      false
    );
  }
  assert.ok(productionFake.getShaderSource().includes(wgsl));
  assert.match(
    productionFake.getShaderSource(),
    /let attrBaseInput = row \* 8u;/
  );
  assert.match(
    productionFake.getShaderSource(),
    /let rgb = evaluateProductionDegree2SpatialSh\(row, raw0\.xyz\);/
  );
  assert.doesNotMatch(
    productionFake.getShaderSource(),
    /evaluateProductionDegree2SpatialSh\(row, pos\)/
  );
  assert.equal(productionResult.gpuResources.renderAttributeByteLength, 32);
  for (const buffer of [
    productionResult.gpuResources.statePositionBuffer,
    productionResult.gpuResources.renderAttributeBuffer,
    productionResult.gpuResources.footprintPayloadBuffer
  ]) buffer.destroy();

  const diagnosticFake = createFakeDevice();
  const diagnosticResult = await buildWebGpu4DStatePositionsForCandidates({
    device: diagnosticFake.device,
    raw: rawFixture(1),
    candidateIndices: new Uint32Array([0]),
    rawXyzOpacity: candidateXyzOpacity(1),
    buildConfig: { timestamp: 0, scalingModifier: 1, sigmaScale: 1 },
    projectionParams: projectionParams()
  });
  assert.equal(
    diagnosticResult.diagnosticGpuResourceOwnership,
    'evaluator-call-scoped-destroyed-before-promise-resolution'
  );
  assert.ok(diagnosticFake.buffers.every((buffer) => buffer.destroyed));

  const tileInputSource = await readFile(
    new URL('../demo/js/webgpu_production_tile_input.js', import.meta.url),
    'utf8'
  );
  const rasterObserverSource = await readFile(
    new URL('../demo/js/webgpu_population_raster_semantic_observer.js', import.meta.url),
    'utf8'
  );
  assert.match(tileInputSource, /let tileBase = row \* 3u;/);
  assert.match(tileInputSource, /let attributeBase = row \* 2u;/);
  assert.match(rasterObserverSource, /POPULATION_RASTER_SEMANTIC_COMPANION_FLOAT_STRIDE/);
  assert.equal(PRODUCTION_EVALUATOR_STORAGE_BINDING_COUNT, 8);

  console.log('Step122 Impl3 production degree-2 spatial SH smoke: OK', {
    layoutVersion: PRODUCTION_CANDIDATE_ATTRIBUTE_INPUT_LAYOUT_VERSION,
    floatStride: PRODUCTION_CANDIDATE_ATTRIBUTE_FLOAT_STRIDE,
    byteStride: PRODUCTION_CANDIDATE_ATTRIBUTE_BYTE_STRIDE,
    fixedChunkBytes: productionCandidateAttributeByteLength(65_536),
    fixedRangeBytes: productionCandidateAttributeByteLength(524_288),
    storageBindingCount: PRODUCTION_EVALUATOR_STORAGE_BINDING_COUNT,
    oneHotValues
  });
} finally {
  globalThis.GPUBufferUsage = previousGpuBufferUsage;
  globalThis.GPUMapMode = previousGpuMapMode;
}
