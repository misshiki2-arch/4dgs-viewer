export const WEBGPU_NORMAL_BACKEND_UNIFORM_RESOURCE_LIFECYCLE_CONTRACT_VERSION =
  'phase3-step66-normal-backend-uniform-resource-lifecycle-v1';

export const WEBGPU_NORMAL_BACKEND_UNIFORM_SHADER_CONSUMPTION_CONTRACT_VERSION =
  'phase3-step66-normal-backend-uniform-shader-consumption-v1';

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
    consumptionMode: 'normal-webgpu-backend-minimal-uniform-compute-consumption',
    status: 'unavailable',
    bindGroupLayoutCreated: false,
    bindGroupCreated: false,
    computePipelineCreated: false,
    uniformReadDispatchSubmitted: false,
    uniformReadbackCompleted: false,
    uniformReadMatchesPackedFrameConstants: false,
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

async function consumeUniformWithMinimalCompute({
  device,
  uniformBuffer,
  uniformData,
  minBindingSizeBytes
}) {
  if (typeof GPUBufferUsage === 'undefined' || typeof GPUMapMode === 'undefined') {
    return createUnavailableConsumptionSummary('webgpu-buffer-usage-unavailable');
  }
  const expected = Array.from(uniformData.slice(0, 4));
  const outputBuffer = device.createBuffer({
    label: 'phase3-step66-normal-backend-uniform-consumption-output',
    size: 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const readbackBuffer = device.createBuffer({
    label: 'phase3-step66-normal-backend-uniform-consumption-readback',
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: 'phase3-step66-normal-backend-uniform-bind-group-layout',
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
      }
    ]
  });
  const bindGroup = device.createBindGroup({
    label: 'phase3-step66-normal-backend-uniform-bind-group',
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } }
    ]
  });
  const shader = device.createShaderModule({
    label: 'phase3-step66-normal-backend-uniform-consumption-wgsl',
    code: `
struct FrameUniforms {
  frame: vec4<f32>,
  view: mat4x4<f32>,
  projection: mat4x4<f32>,
  viewProjection: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
@group(0) @binding(1) var<storage, read_write> output: array<f32, 4>;

@compute @workgroup_size(1)
fn main() {
  output[0] = frameUniforms.frame.x;
  output[1] = frameUniforms.frame.y;
  output[2] = frameUniforms.frame.z;
  output[3] = frameUniforms.frame.w;
}
`
  });
  const pipelineLayout = device.createPipelineLayout({
    label: 'phase3-step66-normal-backend-uniform-pipeline-layout',
    bindGroupLayouts: [bindGroupLayout]
  });
  const pipeline = device.createComputePipeline({
    label: 'phase3-step66-normal-backend-uniform-consumption-pipeline',
    layout: pipelineLayout,
    compute: {
      module: shader,
      entryPoint: 'main'
    }
  });
  const encoder = device.createCommandEncoder({
    label: 'phase3-step66-normal-backend-uniform-consumption-encoder'
  });
  const pass = encoder.beginComputePass({
    label: 'phase3-step66-normal-backend-uniform-consumption-pass'
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, 16);
  device.queue.submit([encoder.finish()]);
  let submittedWorkDone = false;
  if (typeof device.queue.onSubmittedWorkDone === 'function') {
    await device.queue.onSubmittedWorkDone();
    submittedWorkDone = true;
  }
  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const actual = Array.from(new Float32Array(readbackBuffer.getMappedRange()).slice(0, 4));
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
  const uniformReadMatchesPackedFrameConstants = maxAbsDiff <= epsilon;
  return {
    contractVersion:
      WEBGPU_NORMAL_BACKEND_UNIFORM_SHADER_CONSUMPTION_CONTRACT_VERSION,
    consumptionMode: 'normal-webgpu-backend-minimal-uniform-compute-consumption',
    status: uniformReadMatchesPackedFrameConstants ? 'ok' : 'mismatch',
    bindGroupLayoutCreated: true,
    bindGroupCreated: true,
    computePipelineCreated: true,
    uniformReadDispatchSubmitted: true,
    uniformReadbackCompleted: true,
    submittedWorkDone,
    uniformReadMatchesPackedFrameConstants,
    expectedFrameUniformPrefix: expected,
    readbackFrameUniformPrefix: actual,
    maxAbsDiff,
    epsilon,
    shaderEntryPoint: 'main',
    workgroupCount: 1,
    outputByteLength: 16,
    validationField: 'frameIndex_time_viewportWidth_viewportHeight',
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
      }
    ],
    productionShaderImplemented: false,
    shColorParityImplemented: false,
    fullDatasetGpuResidencyRequired: false
  };
}

export async function prepareNormalBackendUniformResources({
  frameConstantsContract,
  uniformResourcePreparationContract,
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
      label: 'phase3-step66-normal-backend-frame-constants-uniform-buffer',
      size: Math.max(paddedByteLength, uniformData.byteLength),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    lifecycleEvents.push('uniform-buffer-created');
    effectiveDevice.queue.writeBuffer(uniformBuffer, 0, uniformData);
    lifecycleEvents.push('queue-write-buffer');
    const uniformShaderConsumptionContract =
      await consumeUniformWithMinimalCompute({
        device: effectiveDevice,
        uniformBuffer,
        uniformData,
        minBindingSizeBytes: paddedByteLength
      });
    lifecycleEvents.push('bind-group-layout-created');
    lifecycleEvents.push('bind-group-created');
    lifecycleEvents.push('minimal-uniform-compute-dispatched');
    const submittedWorkDone =
      uniformShaderConsumptionContract?.submittedWorkDone === true;
    if (submittedWorkDone) {
      lifecycleEvents.push('queue-submitted-work-done');
    }
    if (typeof uniformBuffer.destroy === 'function') {
      uniformBuffer.destroy();
      lifecycleEvents.push('uniform-buffer-destroyed');
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
      uniformShaderConsumptionContract,
      packedFloat32Count: uniformData.length,
      packedByteLength: uniformData.byteLength,
      paddedUniformByteLength:
        uniformResourcePreparationContract?.layout?.paddedUniformByteLength ??
        null,
      minBindingSizeBytes:
        uniformResourcePreparationContract?.layout?.minBindingSizeBytes ?? null,
      usage: ['UNIFORM', 'COPY_DST'],
      layoutMode:
        uniformResourcePreparationContract?.layout?.layoutMode ?? null,
      reusePolicy:
        'future runner-owned device/resource cache; Step66 uses a per-call transient buffer and bind group',
      disposePolicy: 'destroy transient uniform buffer after minimal shader consumption validation',
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
