import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
  buildWebGpu4DStatePositionsForCandidates
} from '../demo/js/webgpu_4d_state_evaluator.js';
import {
  buildCudaTileInputExpectedAlpha
} from '../demo/js/webgpu_population_aligned_semantic_comparison.js';

const tileInputSource = await readFile(
  new URL('../demo/js/webgpu_production_tile_input.js', import.meta.url),
  'utf8'
);
const productionCompositorSource = await readFile(
  new URL('../demo/js/webgpu_bounded_tile_sort_and_compositor.js', import.meta.url),
  'utf8'
);
const cudaAlignedAccumulationSource = await readFile(
  new URL('../demo/js/webgpu_tile_composite_accumulation_dry_run.js', import.meta.url),
  'utf8'
);
const observerSource = await readFile(
  new URL('../demo/js/webgpu_population_raster_semantic_observer.js', import.meta.url),
  'utf8'
);
const comparisonContractSource = await readFile(
  new URL(
    '../demo/js/common_4dgs_population_semantic_comparison_contracts.js',
    import.meta.url
  ),
  'utf8'
);

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
  limits: {
    maxStorageBufferBindingSize: 1_073_741_824,
    maxBufferSize: 1_073_741_824,
    maxStorageBuffersPerShaderStage: 8
  },
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
      N: 1,
      activeShDegree: 2,
      activeShDegreeT: 2,
      t: new Float32Array([0]),
      tDim: 1,
      scale_t: new Float32Array([1]),
      scaleTDim: 1,
      f_dc: new Float32Array([0, 0, 0]),
      fdcDim: 3,
      f_rest: new Float32Array(45),
      frestDim: 45,
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
    'let alpha = sigmoid(raw0.w) * temporalWeight;'
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

  const alphaAssignment = evaluatorShaderSource.match(/let alpha = ([^;]+);/u);
  assert.ok(alphaAssignment);
  assert.equal(alphaAssignment[1].trim(), 'sigmoid(raw0.w) * temporalWeight');
  assert.equal(
    (alphaAssignment[1].match(/sigmoid\(raw0\.w\)/gu) ?? []).length,
    1
  );
  assert.equal((alphaAssignment[1].match(/temporalWeight/gu) ?? []).length, 1);
  assert.doesNotMatch(alphaAssignment[1], /clamp|min|max/u);

  const representativeExpectedAlpha = 0.017547628968024982;
  const representativeTemporalWeight = 0.1;
  const representativeActivatedOpacity =
    representativeExpectedAlpha / representativeTemporalWeight;
  const representativeRawLogit = Math.log(
    representativeActivatedOpacity / (1 - representativeActivatedOpacity)
  );
  const lowAlpha = buildCudaTileInputExpectedAlpha({
    rawOpacityLogit: representativeRawLogit,
    temporalWeight: representativeTemporalWeight
  });
  assert.ok(Math.abs(lowAlpha - representativeExpectedAlpha) < 1e-12);
  assert.ok(lowAlpha < 0.05);
  assert.notEqual(lowAlpha, 0.05);

  const highAlpha = buildCudaTileInputExpectedAlpha({
    rawOpacityLogit: 10,
    temporalWeight: 1
  });
  assert.ok(Number.isFinite(highAlpha));
  assert.ok(highAlpha > 0.99 && highAlpha < 1);

  assert.match(
    evaluatorShaderSource,
    /let rgb = evaluateProductionDegree2SpatialSh\(row, raw0\.xyz\);/u
  );
  assert.doesNotMatch(
    evaluatorShaderSource,
    /let rgb = clamp\(attrs\.rgb \+ vec3f\(0\.5\), vec3f\(0\.0\), vec3f\(1\.0\)\);/u
  );
  assert.match(
    evaluatorShaderSource,
    /renderAttributes\[attrBase \+ 0u\] = vec4f\(radiusPx, alpha, rgb\.r, rgb\.g\);/u
  );
  assert.match(
    tileInputSource,
    /tileInputs\[tileBase \+ 2u\] = vec4f\([\s\S]*?attribute0\.z,[\s\S]*?attribute0\.w,[\s\S]*?attribute1\.x,[\s\S]*?attribute0\.y[\s\S]*?\);/u
  );

  // Neither compositor belongs to this implementation responsibility.
  assert.match(
    cudaAlignedAccumulationSource,
    /let alpha = min\(0\.99, colorAlpha\.a \* exp\(power\)\);/u
  );
  assert.match(
    productionCompositorSource,
    /let sampleAlpha = clamp\(c\.w \* weight, 0\.0, 0\.98\);/u
  );
  assert.match(
    observerSource,
    /companionEvidence\[evidenceBase \+ 3u\] = colorAlpha;/u
  );
  assert.match(comparisonContractSource, /key: 'productionTileInputAlpha'/u);
  assert.match(comparisonContractSource, /key: 'productionTileInputRgb'/u);
  assert.match(
    comparisonContractSource,
    /fresh-separate-diagnostic-device-reexecuting-production-evaluator-and-tile-input-path/u
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

console.log('Step119 Impl3 / Step122 Impl2 temporal visibility and record-local alpha smoke: OK');
