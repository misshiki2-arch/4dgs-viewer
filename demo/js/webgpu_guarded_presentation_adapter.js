import {
  buildGuardedPresentationAdapterContract,
  buildPresentationBridgeContract,
  buildUnavailableGuardedPresentationAdapterContract
} from './common_4dgs_backend_output_contracts.js';

function clamp01(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

function buildExpectedRgba8Surface(floatSurfaceData) {
  const source = floatSurfaceData instanceof Float32Array
    ? floatSurfaceData
    : new Float32Array(0);
  const bytes = new Uint8Array(source.length);
  for (let i = 0; i < source.length; i++) {
    bytes[i] = Math.round(clamp01(source[i]) * 255);
  }
  return bytes;
}

function readTextureRows(readbackBytes, width, height, bytesPerRow) {
  const rowBytes = width * 4;
  const compact = new Uint8Array(rowBytes * height);
  for (let y = 0; y < height; y++) {
    const sourceOffset = y * bytesPerRow;
    const targetOffset = y * rowBytes;
    compact.set(
      readbackBytes.slice(sourceOffset, sourceOffset + rowBytes),
      targetOffset
    );
  }
  return compact;
}

export async function runWebGpuOnlyGuardedPresentationAdapter({
  device,
  handoffBuffer,
  expectedSurfaceData,
  normalBackendOutputContract,
  presentationHandoffContract,
  targetFormat = 'rgba8unorm'
} = {}) {
  if (!device || !handoffBuffer) {
    return buildUnavailableGuardedPresentationAdapterContract(
      'presentation-adapter-input-resource-unavailable'
    );
  }
  if (typeof GPUTextureUsage === 'undefined' || typeof GPUMapMode === 'undefined') {
    return buildUnavailableGuardedPresentationAdapterContract(
      'webgpu-texture-usage-unavailable'
    );
  }
  const width = normalBackendOutputContract?.outputWidth ?? 0;
  const height = normalBackendOutputContract?.outputHeight ?? 0;
  if (width <= 0 || height <= 0) {
    return buildUnavailableGuardedPresentationAdapterContract(
      'presentation-target-extent-unavailable'
    );
  }
  const expectedBytes = buildExpectedRgba8Surface(expectedSurfaceData);
  const bytesPerRow = Math.max(256, Math.ceil((width * 4) / 256) * 256);
  const readbackBufferSize = bytesPerRow * height;
  const targetTexture = device.createTexture({
    label: 'phase3-step70-webgpu-only-guarded-presentation-target-texture',
    size: { width, height },
    format: targetFormat,
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC
  });
  const bridgeTexture = device.createTexture({
    label: 'phase3-step71-viewer-presentation-render-target-bridge-texture',
    size: { width, height },
    format: targetFormat,
    usage:
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.RENDER_ATTACHMENT
  });
  const readbackBuffer = device.createBuffer({
    label: 'phase3-step70-webgpu-only-guarded-presentation-target-readback',
    size: readbackBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const bridgeReadbackBuffer = device.createBuffer({
    label: 'phase3-step71-viewer-presentation-render-target-bridge-readback',
    size: readbackBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: 'phase3-step70-webgpu-only-guarded-presentation-bind-group-layout',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'read-only-storage',
          minBindingSize: expectedSurfaceData.byteLength
        }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: {
          access: 'write-only',
          format: targetFormat,
          viewDimension: '2d'
        }
      }
    ]
  });
  const bindGroup = device.createBindGroup({
    label: 'phase3-step70-webgpu-only-guarded-presentation-bind-group',
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: handoffBuffer } },
      { binding: 1, resource: targetTexture.createView() }
    ]
  });
  const shader = device.createShaderModule({
    label: 'phase3-step70-webgpu-only-guarded-presentation-wgsl',
    code: `
@group(0) @binding(0) var<storage, read> handoffSurface: array<f32>;
@group(0) @binding(1) var presentationTarget: texture_storage_2d<${targetFormat}, write>;

const targetWidth: u32 = ${width}u;
const targetHeight: u32 = ${height}u;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= targetWidth || globalId.y >= targetHeight) {
    return;
  }
  let pixelIndex = (globalId.y * targetWidth) + globalId.x;
  let base = pixelIndex * 4u;
  let rgba = vec4<f32>(
    handoffSurface[base + 0u],
    handoffSurface[base + 1u],
    handoffSurface[base + 2u],
    handoffSurface[base + 3u]
  );
  textureStore(presentationTarget, vec2<i32>(globalId.xy), rgba);
}
`
  });
  const pipeline = device.createComputePipeline({
    label: 'phase3-step70-webgpu-only-guarded-presentation-pipeline',
    layout: device.createPipelineLayout({
      label: 'phase3-step70-webgpu-only-guarded-presentation-pipeline-layout',
      bindGroupLayouts: [bindGroupLayout]
    }),
    compute: {
      module: shader,
      entryPoint: 'main'
    }
  });
  const encoder = device.createCommandEncoder({
    label: 'phase3-step70-webgpu-only-guarded-presentation-encoder'
  });
  const pass = encoder.beginComputePass({
    label: 'phase3-step70-webgpu-only-guarded-presentation-pass'
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(width, height);
  pass.end();
  encoder.copyTextureToBuffer(
    { texture: targetTexture },
    { buffer: readbackBuffer, bytesPerRow, rowsPerImage: height },
    { width, height }
  );
  encoder.copyTextureToTexture(
    { texture: targetTexture },
    { texture: bridgeTexture },
    { width, height }
  );
  encoder.copyTextureToBuffer(
    { texture: bridgeTexture },
    { buffer: bridgeReadbackBuffer, bytesPerRow, rowsPerImage: height },
    { width, height }
  );
  device.queue.submit([encoder.finish()]);
  let submittedWorkDone = false;
  if (typeof device.queue.onSubmittedWorkDone === 'function') {
    await device.queue.onSubmittedWorkDone();
    submittedWorkDone = true;
  }
  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const paddedReadback = new Uint8Array(readbackBuffer.getMappedRange()).slice(
    0,
    readbackBufferSize
  );
  const compactReadback = readTextureRows(paddedReadback, width, height, bytesPerRow);
  readbackBuffer.unmap();
  await bridgeReadbackBuffer.mapAsync(GPUMapMode.READ);
  const paddedBridgeReadback = new Uint8Array(
    bridgeReadbackBuffer.getMappedRange()
  ).slice(0, readbackBufferSize);
  const compactBridgeReadback = readTextureRows(
    paddedBridgeReadback,
    width,
    height,
    bytesPerRow
  );
  bridgeReadbackBuffer.unmap();
  const guardedPresentationAdapterContract =
    buildGuardedPresentationAdapterContract({
      normalBackendOutputContract,
      presentationHandoffContract,
      targetFormat,
      targetWidth: width,
      targetHeight: height,
      expectedBytes,
      readbackBytes: compactReadback,
      gpuWriteSubmitted: true,
      readbackCompleted: true,
      submittedWorkDone,
      epsilon: 0
    });
  const presentationBridgeContract = buildPresentationBridgeContract({
    guardedPresentationAdapterContract,
    normalBackendOutputContract,
    presentationHandoffContract,
    targetFormat,
    targetKind: 'render-target-texture-presentation-bridge',
    targetWidth: width,
    targetHeight: height,
    expectedBytes: compactReadback,
    readbackBytes: compactBridgeReadback,
    renderTargetCreated: true,
    gpuCopySubmitted: true,
    readbackCompleted: true,
    currentTextureConnectionAttempted: true,
    currentTextureConnected: false,
    currentTextureBlockedReason:
      'viewer-canvas-currentTexture direct connection requires a viewer canvas WebGPU context owned by the guarded viewer lifecycle; Step71 uses a render-target bridge until that context is passed into the adapter boundary',
    submittedWorkDone,
    epsilon: 0
  });
  if (typeof readbackBuffer.destroy === 'function') {
    readbackBuffer.destroy();
  }
  if (typeof bridgeReadbackBuffer.destroy === 'function') {
    bridgeReadbackBuffer.destroy();
  }
  if (typeof targetTexture.destroy === 'function') {
    targetTexture.destroy();
  }
  if (typeof bridgeTexture.destroy === 'function') {
    bridgeTexture.destroy();
  }
  return {
    ...guardedPresentationAdapterContract,
    presentationBridgeContract,
    viewerPresentationBridgeReady:
      presentationBridgeContract.viewerPresentationBridgeReady === true,
    currentTextureConnectionAttempted:
      presentationBridgeContract.currentTextureConnectionAttempted === true,
    currentTextureConnected:
      presentationBridgeContract.currentTextureConnected === true,
    currentTextureBlockedReason:
      presentationBridgeContract.currentTextureBlockedReason ?? null,
    renderTargetBridgeReady:
      presentationBridgeContract.renderTargetBridgeReady === true,
    renderTargetTextureConnected:
      presentationBridgeContract.renderTargetTextureConnected === true,
    bridgeGpuCommandPath: presentationBridgeContract.gpuCommandPath
  };
}
