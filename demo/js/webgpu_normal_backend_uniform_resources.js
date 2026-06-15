export const WEBGPU_NORMAL_BACKEND_UNIFORM_RESOURCE_LIFECYCLE_CONTRACT_VERSION =
  'phase3-step65-normal-backend-uniform-resource-lifecycle-v1';

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
      label: 'phase3-step65-normal-backend-frame-constants-uniform-buffer',
      size: Math.max(paddedByteLength, uniformData.byteLength),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    lifecycleEvents.push('uniform-buffer-created');
    effectiveDevice.queue.writeBuffer(uniformBuffer, 0, uniformData);
    lifecycleEvents.push('queue-write-buffer');
    let submittedWorkDone = false;
    if (typeof effectiveDevice.queue.onSubmittedWorkDone === 'function') {
      await effectiveDevice.queue.onSubmittedWorkDone();
      submittedWorkDone = true;
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
      status: 'ok',
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
      bindGroupCreatedThisStep: false,
      shaderConsumptionImplemented: false,
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
        'future runner-owned device/resource cache; Step65 uses a per-call transient buffer',
      disposePolicy: 'destroy transient uniform buffer after write validation',
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
