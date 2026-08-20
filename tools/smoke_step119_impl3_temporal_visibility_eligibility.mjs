import assert from 'node:assert/strict';

import {
  buildWebGpu4DStatePositionsForCandidates
} from '../demo/js/webgpu_4d_state_evaluator.js';

const previousGpuBufferUsage = globalThis.GPUBufferUsage;
globalThis.GPUBufferUsage = {
  STORAGE: 1,
  UNIFORM: 2,
  COPY_SRC: 4
};

let evaluatorShaderSource = null;
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
const fakeDevice = {
  createBuffer: createFakeBuffer,
  createShaderModule: ({ code }) => {
    evaluatorShaderSource = code;
    return { code };
  },
  createComputePipeline: () => ({
    getBindGroupLayout: () => ({ label: 'fake-evaluator-bind-group-layout' })
  }),
  createBindGroup: () => ({ label: 'fake-evaluator-bind-group' }),
  createCommandEncoder: () => ({
    beginComputePass: () => fakePass,
    finish: () => ({ label: 'fake-evaluator-command-buffer' })
  }),
  queue: {
    submit: () => {},
    onSubmittedWorkDone: async () => {}
  }
};

try {
  const result = await buildWebGpu4DStatePositionsForCandidates({
    device: fakeDevice,
    raw: {
      t: new Float32Array([0]),
      tDim: 1,
      scale_t: new Float32Array([1]),
      scaleTDim: 1,
      f_dc: new Float32Array([0, 0, 0]),
      fdcDim: 3,
      scale_xyz: new Float32Array([1, 1, 1]),
      scaleXYZDim: 3,
      rotation: new Float32Array([1, 0, 0, 0]),
      rotationDim: 4,
      rotation_r: new Float32Array([1, 0, 0, 0]),
      rotationRDim: 4
    },
    candidateIndices: new Uint32Array([0]),
    rawXyzOpacity: new Float32Array([0, 0, 1, 0]),
    buildConfig: {
      timestamp: 0,
      scalingModifier: 1,
      sigmaScale: 1
    },
    projectionParams: new Float32Array(24),
    readbackPolicy: 'none',
    keepGpuResources: true,
    sourceWorksetResourceIdentity: 'step119-impl3-smoke-workset'
  });

  assert.ok(result.gpuResources);
  assert.equal(result.productionReadbackPerformed, false);
  assert.equal(
    result.gpuResources.sourceWorksetResourceIdentity,
    'step119-impl3-smoke-workset'
  );
  assert.equal(typeof evaluatorShaderSource, 'string');

  const thresholdMatch = evaluatorShaderSource.match(
    /const CUDA_TEMPORAL_VISIBILITY_THRESHOLD: f32 = ([0-9.]+);/
  );
  assert.ok(thresholdMatch);
  const threshold = Number(thresholdMatch[1]);
  assert.equal(threshold, 0.05);

  const predicateMatch = evaluatorShaderSource.match(
    /let temporalEligible = temporalWeight ([><]=?) CUDA_TEMPORAL_VISIBILITY_THRESHOLD;/
  );
  assert.ok(predicateMatch);
  assert.equal(predicateMatch[1], '>');
  const temporalEligible = (temporalWeight) => temporalWeight > threshold;
  assert.equal(temporalEligible(0.050001), true);
  assert.equal(temporalEligible(0.05), false);
  assert.equal(temporalEligible(0.049999), false);

  const predicateIndex = evaluatorShaderSource.indexOf(
    'let temporalEligible = temporalWeight > CUDA_TEMPORAL_VISIBILITY_THRESHOLD;'
  );
  const invalidBranchIndex = evaluatorShaderSource.indexOf(
    'if (!temporalEligible) {'
  );
  const invalidStateIndex = evaluatorShaderSource.indexOf(
    'statePositions[row] = vec4f(0.0);'
  );
  const invalidAttributeIndex = evaluatorShaderSource.indexOf(
    'renderAttributes[invalidAttributeBase + 0u] = vec4f(0.0);'
  );
  const invalidFootprintIndex = evaluatorShaderSource.indexOf(
    'footprintPayload[invalidFootprintBase + 0u] = vec4f(0.0);'
  );
  const validPositionIndex = evaluatorShaderSource.indexOf(
    'let pos = raw0.xyz + temporalMean.xyz;'
  );
  const validAlphaIndex = evaluatorShaderSource.indexOf(
    'let alpha = clamp(sigmoid(raw0.w) * temporalWeight, 0.05, 0.99);'
  );
  const validRadiusIndex = evaluatorShaderSource.indexOf(
    'let radiusPx = clamp(attrs.w * 900.0 + 2.0 + abs(normalizedTemporal) * 2.0, 3.0, 14.0);'
  );

  assert.ok(predicateIndex >= 0);
  assert.ok(invalidBranchIndex > predicateIndex);
  assert.ok(invalidStateIndex > invalidBranchIndex);
  assert.ok(invalidAttributeIndex > invalidStateIndex);
  assert.ok(invalidFootprintIndex > invalidAttributeIndex);
  assert.ok(validPositionIndex > invalidFootprintIndex);
  assert.ok(validAlphaIndex > validPositionIndex);
  assert.ok(validRadiusIndex > validAlphaIndex);
  assert.match(
    evaluatorShaderSource.slice(invalidBranchIndex, validPositionIndex),
    /footprintPayload\[invalidFootprintBase \+ 2u\] = vec4f\(0\.0\);\s+return;\s+}/
  );

  for (const buffer of [
    result.gpuResources.statePositionBuffer,
    result.gpuResources.renderAttributeBuffer,
    result.gpuResources.footprintPayloadBuffer
  ]) {
    buffer.destroy();
  }
} finally {
  if (previousGpuBufferUsage === undefined) {
    delete globalThis.GPUBufferUsage;
  } else {
    globalThis.GPUBufferUsage = previousGpuBufferUsage;
  }
}

console.log('Step119 Impl3 temporal visibility eligibility smoke: OK');
