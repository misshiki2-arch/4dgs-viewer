export const WEBGPU_NORMAL_BACKEND_UNIFORM_RESOURCE_LIFECYCLE_CONTRACT_VERSION =
  'phase3-step67-normal-backend-uniform-resource-lifecycle-v1';

export const WEBGPU_NORMAL_BACKEND_UNIFORM_SHADER_CONSUMPTION_CONTRACT_VERSION =
  'phase3-step67-normal-backend-uniform-sample-shader-consumption-v1';

export const WEBGPU_NORMAL_BACKEND_SAMPLE_RESOURCE_CONTRACT_VERSION =
  'phase3-step67-normal-backend-sample-storage-resource-v1';

const SAMPLE_FLOAT_STRIDE = 8;

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function getMatrixValues(matrixSummary) {
  return Array.isArray(matrixSummary?.values) &&
    matrixSummary.values.length === 16 &&
    matrixSummary.values.every(isFiniteNumber)
    ? matrixSummary.values
    : null;
}

function createUnavailableSummary(reason, extra = {}) {
  return {
    contractVersion:
      WEBGPU_NORMAL_BACKEND_UNIFORM_RESOURCE_LIFECYCLE_CONTRACT_VERSION,
    resourceMode: 'normal-webgpu-backend-uniform-buffer-resource-lifecycle',
    status: 'unavailable',
    normalBackendOwnsGpuUniformResource: true,
    uniformBufferCreated: false,
    uniformBufferWriteSubmitted: false,
    uniformBufferWriteCompleted: false,
    uniformBufferDestroyed: false,
    sampleBufferCreated: false,
    sampleBufferWriteSubmitted: false,
    sampleBufferDestroyed: false,
    queueWriteBufferUsed: false,
    bindGroupReadyBoundary: false,
    bindGroupLayoutCreated: false,
    bindGroupCreatedThisStep: false,
    shaderConsumptionImplemented: false,
    uniformShaderConsumptionContract: createUnavailableConsumptionSummary(reason),
    reason,
    ...extra
  };
}

function createUnavailableConsumptionSummary(reason, extra = {}) {
  return {
    contractVersion:
      WEBGPU_NORMAL_BACKEND_UNIFORM_SHADER_CONSUMPTION_CONTRACT_VERSION,
    consumptionMode:
      'normal-webgpu-backend-minimal-uniform-and-sample-compute-consumption',
    status: 'unavailable',
    bindGroupLayoutCreated: false,
    bindGroupCreated: false,
    computePipelineCreated: false,
    uniformReadDispatchSubmitted: false,
    uniformReadbackCompleted: false,
    uniformReadMatchesPackedFrameConstants: false,
    sampleStorageBufferReadbackCompleted: false,
    sampleReadMatchesPackedSelectedSamples: false,
    reason,
    ...extra
  };
}

export function packNormalBackendFrameUniformData(frameConstantsContract) {
  const matrices = frameConstantsContract?.matrices ?? {};
  const viewMatrix = getMatrixValues(matrices.viewMatrix);
  const projectionMatrix = getMatrixValues(matrices.projectionMatrix);
  const viewProjectionMatrix = getMatrixValues(matrices.viewProjectionMatrix);
  const viewport = frameConstantsContract?.viewport ?? {};
  if (!viewMatrix || !projectionMatrix || !viewProjectionMatrix) {
    return null;
  }
  const data = new Float32Array(52);
  data.set(
    [
      isFiniteNumber(frameConstantsContract?.frameIndexUniformValue)
        ? frameConstantsContract.frameIndexUniformValue
        : 0,
      isFiniteNumber(frameConstantsContract?.timeSeconds)
        ? frameConstantsContract.timeSeconds
        : 0,
      isFiniteNumber(viewport.width) ? viewport.width : 0,
      isFiniteNumber(viewport.height) ? viewport.height : 0
    ],
    0
  );
  data.set(viewMatrix, 4);
  data.set(projectionMatrix, 20);
  data.set(viewProjectionMatrix, 36);
  return data;
}

function sourceKindToCode(source) {
  if (source === 'webgpuConstrainedDisplayAdapterDryRunComparison.sampleTexturePixels') {
    return 40;
  }
  if (source === 'webgpuRenderTargetHandoffDryRunComparison.sampleTexturePixels') {
    return 39;
  }
  if (source === 'webgpuFramebufferFreeTileOutputDryRunComparison.sampleTexturePixels') {
    return 38;
  }
  if (source === 'webgpuRenderHandoffStub.sampleRecords') {
    return -1;
  }
  return 0;
}

function finiteNumberOr(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

export function packNormalBackendSelectedSampleData(samples) {
  if (!Array.isArray(samples) || samples.length <= 0) {
    return {
      data: null,
      summary: {
        contractVersion: WEBGPU_NORMAL_BACKEND_SAMPLE_RESOURCE_CONTRACT_VERSION,
        status: 'unavailable',
        reason: 'selected-samples-empty',
        sampleCount: 0,
        containsRenderHandoffFallback: false
      }
    };
  }
  const containsRenderHandoffFallback = samples.some(
    (sample) => sample?.source === 'webgpuRenderHandoffStub.sampleRecords'
  );
  if (containsRenderHandoffFallback) {
    return {
      data: null,
      summary: {
        contractVersion: WEBGPU_NORMAL_BACKEND_SAMPLE_RESOURCE_CONTRACT_VERSION,
        status: 'blocked',
        reason: 'render-handoff-fallback-sample-present-in-selected-samples',
        sampleCount: samples.length,
        containsRenderHandoffFallback: true
      }
    };
  }
  const data = new Float32Array(samples.length * SAMPLE_FLOAT_STRIDE);
  samples.forEach((sample, index) => {
    const offset = index * SAMPLE_FLOAT_STRIDE;
    const samplePx = sample?.samplePx ?? {};
    const color = sample?.colorAlpha ?? {};
    data[offset + 0] = finiteNumberOr(samplePx.x, 0);
    data[offset + 1] = finiteNumberOr(samplePx.y, 0);
    data[offset + 2] = finiteNumberOr(color.r, 0);
    data[offset + 3] = finiteNumberOr(color.g, 0);
    data[offset + 4] = finiteNumberOr(color.b, 0);
    data[offset + 5] = finiteNumberOr(color.a, 0);
    data[offset + 6] = sourceKindToCode(sample?.source);
    data[offset + 7] = finiteNumberOr(sample?.recordIndex, -1);
  });
  return {
    data,
    summary: {
      contractVersion: WEBGPU_NORMAL_BACKEND_SAMPLE_RESOURCE_CONTRACT_VERSION,
      status: 'ok',
      sampleResourceMode: 'normal-webgpu-backend-selected-sample-storage-buffer',
      sampleCount: samples.length,
      sampleFloatStride: SAMPLE_FLOAT_STRIDE,
      packedFloat32Count: data.length,
      packedByteLength: data.byteLength,
      containsRenderHandoffFallback: false,
      packedFields: [
        'samplePx.x',
        'samplePx.y',
        'colorAlpha.r',
        'colorAlpha.g',
        'colorAlpha.b',
        'colorAlpha.a',
        'sourceKindCode',
        'recordIndex'
      ],
      sampleSources: [...new Set(samples.map((sample) => sample?.source ?? null))]
    }
  };
}

async function consumeUniformWithMinimalCompute({
  device,
  uniformBuffer,
  sampleBuffer,
  uniformData,
  sampleData,
  minBindingSizeBytes
}) {
  if (typeof GPUBufferUsage === 'undefined' || typeof GPUMapMode === 'undefined') {
    return createUnavailableConsumptionSummary('webgpu-buffer-usage-unavailable');
  }
  const expected = Array.from(uniformData.slice(0, 4));
  const expectedSample = Array.from(sampleData.slice(0, SAMPLE_FLOAT_STRIDE));
  const outputFloatCount = 4 + SAMPLE_FLOAT_STRIDE;
  const outputByteLength = outputFloatCount * 4;
  const outputBuffer = device.createBuffer({
    label: 'phase3-step67-normal-backend-uniform-sample-consumption-output',
    size: outputByteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const readbackBuffer = device.createBuffer({
    label: 'phase3-step67-normal-backend-uniform-sample-consumption-readback',
    size: outputByteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: 'phase3-step67-normal-backend-uniform-sample-bind-group-layout',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'uniform',
          minBindingSize: minBindingSizeBytes
        }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'read-only-storage',
          minBindingSize: sampleData.byteLength
        }
      }
    ]
  });
  const bindGroup = device.createBindGroup({
    label: 'phase3-step67-normal-backend-uniform-sample-bind-group',
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
      { binding: 2, resource: { buffer: sampleBuffer } }
    ]
  });
  const shader = device.createShaderModule({
    label: 'phase3-step67-normal-backend-uniform-sample-consumption-wgsl',
    code: `
struct FrameUniforms {
  frame: vec4<f32>,
  view: mat4x4<f32>,
  projection: mat4x4<f32>,
  viewProjection: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> selectedSamples: array<f32>;

@compute @workgroup_size(1)
fn main() {
  output[0] = frameUniforms.frame.x;
  output[1] = frameUniforms.frame.y;
  output[2] = frameUniforms.frame.z;
  output[3] = frameUniforms.frame.w;
  for (var i: u32 = 0u; i < 8u; i = i + 1u) {
    output[4u + i] = selectedSamples[i];
  }
}
`
  });
  const pipelineLayout = device.createPipelineLayout({
    label: 'phase3-step67-normal-backend-uniform-sample-pipeline-layout',
    bindGroupLayouts: [bindGroupLayout]
  });
  const pipeline = device.createComputePipeline({
    label: 'phase3-step67-normal-backend-uniform-sample-consumption-pipeline',
    layout: pipelineLayout,
    compute: {
      module: shader,
      entryPoint: 'main'
    }
  });
  const encoder = device.createCommandEncoder({
    label: 'phase3-step67-normal-backend-uniform-sample-consumption-encoder'
  });
  const pass = encoder.beginComputePass({
    label: 'phase3-step67-normal-backend-uniform-sample-consumption-pass'
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputByteLength);
  device.queue.submit([encoder.finish()]);
  let submittedWorkDone = false;
  if (typeof device.queue.onSubmittedWorkDone === 'function') {
    await device.queue.onSubmittedWorkDone();
    submittedWorkDone = true;
  }
  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const actualOutput = Array.from(
    new Float32Array(readbackBuffer.getMappedRange()).slice(0, outputFloatCount)
  );
  const actual = actualOutput.slice(0, 4);
  const actualSample = actualOutput.slice(4, 4 + SAMPLE_FLOAT_STRIDE);
  readbackBuffer.unmap();
  if (typeof outputBuffer.destroy === 'function') {
    outputBuffer.destroy();
  }
  if (typeof readbackBuffer.destroy === 'function') {
    readbackBuffer.destroy();
  }
  const epsilon = 1e-6;
  const maxAbsDiff = actual.reduce((maxDiff, value, index) => {
    return Math.max(maxDiff, Math.abs(value - expected[index]));
  }, 0);
  const sampleMaxAbsDiff = actualSample.reduce((maxDiff, value, index) => {
    return Math.max(maxDiff, Math.abs(value - expectedSample[index]));
  }, 0);
  const uniformReadMatchesPackedFrameConstants = maxAbsDiff <= epsilon;
  const sampleReadMatchesPackedSelectedSamples = sampleMaxAbsDiff <= epsilon;
  return {
    contractVersion:
      WEBGPU_NORMAL_BACKEND_UNIFORM_SHADER_CONSUMPTION_CONTRACT_VERSION,
    consumptionMode:
      'normal-webgpu-backend-minimal-uniform-and-sample-compute-consumption',
    status:
      uniformReadMatchesPackedFrameConstants && sampleReadMatchesPackedSelectedSamples
        ? 'ok'
        : 'mismatch',
    bindGroupLayoutCreated: true,
    bindGroupCreated: true,
    computePipelineCreated: true,
    uniformReadDispatchSubmitted: true,
    uniformReadbackCompleted: true,
    sampleStorageBufferReadbackCompleted: true,
    submittedWorkDone,
    uniformReadMatchesPackedFrameConstants,
    sampleReadMatchesPackedSelectedSamples,
    expectedFrameUniformPrefix: expected,
    readbackFrameUniformPrefix: actual,
    expectedFirstSamplePackedFields: expectedSample,
    readbackFirstSamplePackedFields: actualSample,
    maxAbsDiff,
    sampleMaxAbsDiff,
    epsilon,
    shaderEntryPoint: 'main',
    workgroupCount: 1,
    outputByteLength,
    validationField:
      'frameIndex_time_viewportWidth_viewportHeight_and_first_selected_sample',
    bindGroupLayoutEntries: [
      {
        binding: 0,
        visibility: 'compute',
        bufferType: 'uniform',
        minBindingSize: minBindingSizeBytes
      },
      {
        binding: 1,
        visibility: 'compute',
        bufferType: 'storage'
      },
      {
        binding: 2,
        visibility: 'compute',
        bufferType: 'read-only-storage',
        minBindingSize: sampleData.byteLength
      }
    ],
    sampleStorageBufferConsumed: true,
    samplePackedFields: [
      'samplePx.x',
      'samplePx.y',
      'colorAlpha.r',
      'colorAlpha.g',
      'colorAlpha.b',
      'colorAlpha.a',
      'sourceKindCode',
      'recordIndex'
    ],
    productionShaderImplemented: false,
    shColorParityImplemented: false,
    fullDatasetGpuResidencyRequired: false
  };
}

export async function prepareNormalBackendUniformResources({
  frameConstantsContract,
  uniformResourcePreparationContract,
  selectedSamples = [],
  device = null
} = {}) {
  const startMs =
    typeof performance !== 'undefined' && typeof performance.now === 'function'
      ? performance.now()
      : Date.now();
  if (uniformResourcePreparationContract?.uniformResourcePreparationReady !== true) {
    return createUnavailableSummary('uniform-resource-preparation-not-ready');
  }
  const uniformData = packNormalBackendFrameUniformData(frameConstantsContract);
  if (!uniformData) {
    return createUnavailableSummary('frame-constants-pack-failed');
  }
  if (typeof GPUBufferUsage === 'undefined') {
    return createUnavailableSummary('webgpu-buffer-usage-unavailable', {
      packedFloat32Count: uniformData.length,
      packedByteLength: uniformData.byteLength
    });
  }
  const packedSamples = packNormalBackendSelectedSampleData(selectedSamples);
  if (!packedSamples?.data) {
    return createUnavailableSummary(
      packedSamples?.summary?.reason ?? 'selected-sample-pack-failed',
      {
        sampleResourceLifecycleContract: packedSamples?.summary ?? null,
        packedFloat32Count: uniformData.length,
        packedByteLength: uniformData.byteLength
      }
    );
  }
  let adapter = null;
  let ownedDevice = false;
  let effectiveDevice = device;
  const lifecycleEvents = [];
  try {
    if (!effectiveDevice) {
      if (typeof navigator === 'undefined' || !navigator.gpu) {
        return createUnavailableSummary('webgpu-unavailable', {
          packedFloat32Count: uniformData.length,
          packedByteLength: uniformData.byteLength
        });
      }
      adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        return createUnavailableSummary('webgpu-adapter-unavailable', {
          packedFloat32Count: uniformData.length,
          packedByteLength: uniformData.byteLength
        });
      }
      effectiveDevice = await adapter.requestDevice();
      ownedDevice = true;
      lifecycleEvents.push('transient-device-created');
    }
    const paddedByteLength =
      uniformResourcePreparationContract?.layout?.paddedUniformByteLength ??
      uniformData.byteLength;
    const uniformBuffer = effectiveDevice.createBuffer({
      label: 'phase3-step67-normal-backend-frame-constants-uniform-buffer',
      size: Math.max(paddedByteLength, uniformData.byteLength),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    lifecycleEvents.push('uniform-buffer-created');
    effectiveDevice.queue.writeBuffer(uniformBuffer, 0, uniformData);
    lifecycleEvents.push('queue-write-buffer');
    const sampleBuffer = effectiveDevice.createBuffer({
      label: 'phase3-step67-normal-backend-selected-sample-storage-buffer',
      size: packedSamples.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    lifecycleEvents.push('sample-storage-buffer-created');
    effectiveDevice.queue.writeBuffer(sampleBuffer, 0, packedSamples.data);
    lifecycleEvents.push('queue-write-sample-buffer');
    const uniformShaderConsumptionContract =
      await consumeUniformWithMinimalCompute({
        device: effectiveDevice,
        uniformBuffer,
        sampleBuffer,
        uniformData,
        sampleData: packedSamples.data,
        minBindingSizeBytes: paddedByteLength
      });
    lifecycleEvents.push('bind-group-layout-created');
    lifecycleEvents.push('bind-group-created');
    lifecycleEvents.push('minimal-uniform-sample-compute-dispatched');
    const submittedWorkDone =
      uniformShaderConsumptionContract?.submittedWorkDone === true;
    if (submittedWorkDone) {
      lifecycleEvents.push('queue-submitted-work-done');
    }
    if (typeof uniformBuffer.destroy === 'function') {
      uniformBuffer.destroy();
      lifecycleEvents.push('uniform-buffer-destroyed');
    }
    if (typeof sampleBuffer.destroy === 'function') {
      sampleBuffer.destroy();
      lifecycleEvents.push('sample-storage-buffer-destroyed');
    }
    if (ownedDevice && typeof effectiveDevice.destroy === 'function') {
      effectiveDevice.destroy();
      lifecycleEvents.push('transient-device-destroyed');
    }
    const elapsedMs =
      (typeof performance !== 'undefined' && typeof performance.now === 'function'
        ? performance.now()
        : Date.now()) - startMs;
    return {
      contractVersion:
        WEBGPU_NORMAL_BACKEND_UNIFORM_RESOURCE_LIFECYCLE_CONTRACT_VERSION,
      resourceMode: 'normal-webgpu-backend-uniform-buffer-resource-lifecycle',
      status:
        uniformShaderConsumptionContract?.status === 'ok'
          ? 'ok'
          : 'blocked',
      normalBackendOwnsGpuUniformResource: true,
      deviceOwnershipMode: ownedDevice
        ? 'normal-backend-created-transient-device'
        : 'runner-provided-device',
      uniformBufferCreated: true,
      uniformBufferWriteSubmitted: true,
      uniformBufferWriteCompleted: submittedWorkDone,
      uniformBufferDestroyed: true,
      sampleBufferCreated: true,
      sampleBufferWriteSubmitted: true,
      sampleBufferWriteCompleted: submittedWorkDone,
      sampleBufferDestroyed: true,
      queueWriteBufferUsed: true,
      bindGroupReadyBoundary: true,
      bindGroupLayoutCreated:
        uniformShaderConsumptionContract?.bindGroupLayoutCreated === true,
      bindGroupCreatedThisStep:
        uniformShaderConsumptionContract?.bindGroupCreated === true,
      shaderConsumptionImplemented:
        uniformShaderConsumptionContract?.uniformReadDispatchSubmitted === true,
      uniformReadbackCompleted:
        uniformShaderConsumptionContract?.uniformReadbackCompleted === true,
      uniformReadMatchesPackedFrameConstants:
        uniformShaderConsumptionContract?.uniformReadMatchesPackedFrameConstants === true,
      sampleStorageBufferReadbackCompleted:
        uniformShaderConsumptionContract?.sampleStorageBufferReadbackCompleted === true,
      sampleReadMatchesPackedSelectedSamples:
        uniformShaderConsumptionContract?.sampleReadMatchesPackedSelectedSamples === true,
      uniformShaderConsumptionContract,
      sampleResourceLifecycleContract: {
        ...packedSamples.summary,
        normalBackendOwnsGpuSampleResource: true,
        sampleBufferCreated: true,
        sampleBufferWriteSubmitted: true,
        sampleBufferWriteCompleted: submittedWorkDone,
        sampleBufferDestroyed: true,
        sampleStorageBufferConsumed:
          uniformShaderConsumptionContract?.sampleStorageBufferConsumed === true,
        sampleReadMatchesPackedSelectedSamples:
          uniformShaderConsumptionContract?.sampleReadMatchesPackedSelectedSamples === true
      },
      packedFloat32Count: uniformData.length,
      packedByteLength: uniformData.byteLength,
      paddedUniformByteLength:
        uniformResourcePreparationContract?.layout?.paddedUniformByteLength ??
        null,
      minBindingSizeBytes:
        uniformResourcePreparationContract?.layout?.minBindingSizeBytes ?? null,
      samplePackedFloat32Count: packedSamples.data.length,
      samplePackedByteLength: packedSamples.data.byteLength,
      usage: ['UNIFORM', 'STORAGE', 'COPY_DST'],
      layoutMode:
        uniformResourcePreparationContract?.layout?.layoutMode ?? null,
      reusePolicy:
        'future runner-owned device/resource cache; Step67 uses per-call transient uniform and sample buffers with one bind group',
      disposePolicy:
        'destroy transient uniform and sample buffers after minimal shader consumption validation',
      lifecycleEvents,
      validationOracleOwnsResource: false,
      productionLoopConnected: false,
      interactiveCameraImplemented: false,
      streamingImplemented: false,
      fullDatasetGpuResidencyRequired: false,
      timing: {
        uniformResourceLifecycleMs: elapsedMs
      }
    };
  } catch (error) {
    return createUnavailableSummary('uniform-resource-lifecycle-error', {
      error: {
        name: error?.name ?? 'Error',
        message: error?.message ?? String(error)
      },
      packedFloat32Count: uniformData.length,
      packedByteLength: uniformData.byteLength,
      lifecycleEvents
    });
  }
}
