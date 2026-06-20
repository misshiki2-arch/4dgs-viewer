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

function buildExpectedTextureReadbackBytes(rgbaBytes, textureFormat) {
  if (textureFormat !== 'bgra8unorm') {
    return rgbaBytes;
  }
  const bgraBytes = new Uint8Array(rgbaBytes.length);
  for (let i = 0; i < rgbaBytes.length; i += 4) {
    bgraBytes[i] = rgbaBytes[i + 2];
    bgraBytes[i + 1] = rgbaBytes[i + 1];
    bgraBytes[i + 2] = rgbaBytes[i];
    bgraBytes[i + 3] = rgbaBytes[i + 3];
  }
  return bgraBytes;
}

function bytesMatch(actual, expected) {
  if (actual.length !== expected.length) return false;
  for (let i = 0; i < expected.length; i++) {
    if (actual[i] !== expected[i]) return false;
  }
  return true;
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
  viewerCanvasState = null,
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
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC
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
  const viewerCanvas = viewerCanvasState?.canvas ?? null;
  const currentTextureConnectionAttempted = true;
  const currentTextureGuardAllowed =
    viewerCanvasState?.requestedBackendMode === 'webgpu-exclusive' &&
    viewerCanvasState?.allowViewerCanvasPresentation === true &&
    viewerCanvasState?.webgl2FrameLifecycleSuppressed === true &&
    viewerCanvasState?.provided === true &&
    !!viewerCanvas;
  const preferredCanvasFormat =
    typeof navigator !== 'undefined' && navigator.gpu?.getPreferredCanvasFormat
      ? navigator.gpu.getPreferredCanvasFormat()
      : 'rgba8unorm';
  const currentTextureFormat =
    preferredCanvasFormat === 'bgra8unorm' ? 'bgra8unorm' : 'rgba8unorm';
  let currentTextureContext = null;
  let currentTexture = null;
  let currentTextureReadbackBuffer = null;
  let currentTextureConfigured = false;
  let currentTextureAcquired = false;
  let currentTextureRenderPassSubmitted = false;
  let currentTextureReadbackCompleted = false;
  let currentTextureReadbackMatchesAdapterOutput = false;
  let currentTextureReadback = new Uint8Array(0);
  let currentTextureBlockedReason = currentTextureGuardAllowed
    ? 'viewer-canvas-currentTexture-render-pass-not-submitted'
    : 'viewer-canvas-currentTexture guard requires webgpu-exclusive mode, presentation permission, WebGL2 lifecycle suppression, and a provided viewer canvas';
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
  if (currentTextureGuardAllowed) {
    try {
      currentTextureContext = viewerCanvas.getContext?.('webgpu') ?? null;
      if (currentTextureContext) {
        currentTextureContext.configure({
          device,
          format: currentTextureFormat,
          usage:
            GPUTextureUsage.RENDER_ATTACHMENT |
            GPUTextureUsage.COPY_SRC |
            GPUTextureUsage.COPY_DST,
          alphaMode: 'premultiplied'
        });
        currentTextureConfigured = true;
        currentTexture = currentTextureContext.getCurrentTexture();
        currentTextureAcquired = !!currentTexture;
        currentTextureReadbackBuffer = device.createBuffer({
          label: 'phase3-step72-viewer-current-texture-readback',
          size: readbackBufferSize,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const sampler = device.createSampler({
          label: 'phase3-step72-viewer-current-texture-bridge-sampler',
          magFilter: 'nearest',
          minFilter: 'nearest'
        });
        const currentTextureBindGroupLayout = device.createBindGroupLayout({
          label: 'phase3-step72-viewer-current-texture-bridge-bind-group-layout',
          entries: [
            {
              binding: 0,
              visibility: GPUShaderStage.FRAGMENT,
              sampler: {}
            },
            {
              binding: 1,
              visibility: GPUShaderStage.FRAGMENT,
              texture: { sampleType: 'float', viewDimension: '2d' }
            }
          ]
        });
        const currentTextureBindGroup = device.createBindGroup({
          label: 'phase3-step72-viewer-current-texture-bridge-bind-group',
          layout: currentTextureBindGroupLayout,
          entries: [
            { binding: 0, resource: sampler },
            { binding: 1, resource: targetTexture.createView() }
          ]
        });
        const currentTextureShader = device.createShaderModule({
          label: 'phase3-step72-viewer-current-texture-bridge-wgsl',
          code: `
struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0)
  );
  var uvs = array<vec2f, 3>(
    vec2f(0.0, 1.0),
    vec2f(2.0, 1.0),
    vec2f(0.0, -1.0)
  );
  var out: VertexOut;
  out.position = vec4f(positions[vertexIndex], 0.0, 1.0);
  out.uv = uvs[vertexIndex];
  return out;
}

@group(0) @binding(0) var bridgeSampler: sampler;
@group(0) @binding(1) var bridgeTexture: texture_2d<f32>;

@fragment
fn fsMain(in: VertexOut) -> @location(0) vec4f {
  return textureSample(bridgeTexture, bridgeSampler, in.uv);
}
`
        });
        const currentTexturePipeline = device.createRenderPipeline({
          label: 'phase3-step72-viewer-current-texture-bridge-pipeline',
          layout: device.createPipelineLayout({
            label: 'phase3-step72-viewer-current-texture-bridge-pipeline-layout',
            bindGroupLayouts: [currentTextureBindGroupLayout]
          }),
          vertex: {
            module: currentTextureShader,
            entryPoint: 'vsMain'
          },
          fragment: {
            module: currentTextureShader,
            entryPoint: 'fsMain',
            targets: [{ format: currentTextureFormat }]
          },
          primitive: { topology: 'triangle-list' }
        });
        const currentTexturePass = encoder.beginRenderPass({
          label: 'phase3-step72-viewer-current-texture-bridge-pass',
          colorAttachments: [
            {
              view: currentTexture.createView(),
              clearValue: { r: 0, g: 0, b: 0, a: 0 },
              loadOp: 'clear',
              storeOp: 'store'
            }
          ]
        });
        currentTexturePass.setPipeline(currentTexturePipeline);
        currentTexturePass.setBindGroup(0, currentTextureBindGroup);
        currentTexturePass.setViewport(0, 0, width, height, 0, 1);
        currentTexturePass.setScissorRect(0, 0, width, height);
        currentTexturePass.draw(3);
        currentTexturePass.end();
        currentTextureRenderPassSubmitted = true;
        encoder.copyTextureToBuffer(
          { texture: currentTexture },
          {
            buffer: currentTextureReadbackBuffer,
            bytesPerRow,
            rowsPerImage: height
          },
          { width, height }
        );
      } else {
        currentTextureBlockedReason =
          'viewer canvas WebGPU context was unavailable at the guarded adapter boundary';
      }
    } catch (error) {
      currentTextureBlockedReason = error?.message ?? String(error);
    }
  }
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
  if (currentTextureReadbackBuffer) {
    await currentTextureReadbackBuffer.mapAsync(GPUMapMode.READ);
    const paddedCurrentTextureReadback = new Uint8Array(
      currentTextureReadbackBuffer.getMappedRange()
    ).slice(0, readbackBufferSize);
    currentTextureReadback = readTextureRows(
      paddedCurrentTextureReadback,
      width,
      height,
      bytesPerRow
    );
    currentTextureReadbackBuffer.unmap();
    currentTextureReadbackCompleted = true;
    const expectedCurrentTextureBytes = buildExpectedTextureReadbackBytes(
      compactReadback,
      currentTextureFormat
    );
    currentTextureReadbackMatchesAdapterOutput = bytesMatch(
      currentTextureReadback,
      expectedCurrentTextureBytes
    );
  }
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
    currentTextureConnected:
      currentTextureAcquired &&
      currentTextureRenderPassSubmitted &&
      currentTextureReadbackCompleted &&
      currentTextureReadbackMatchesAdapterOutput,
    currentTextureContextProvided: !!currentTextureContext,
    currentTextureConfigured,
    currentTextureAcquired,
    currentTextureRenderPassSubmitted,
    currentTextureReadbackCompleted,
    currentTextureReadbackMatchesAdapterOutput,
    currentTextureFormat,
    currentTextureReadbackBytes: currentTextureReadback,
    currentTextureBlockedReason,
    submittedWorkDone,
    epsilon: 0
  });
  if (typeof readbackBuffer.destroy === 'function') {
    readbackBuffer.destroy();
  }
  if (typeof bridgeReadbackBuffer.destroy === 'function') {
    bridgeReadbackBuffer.destroy();
  }
  if (
    currentTextureReadbackBuffer &&
    typeof currentTextureReadbackBuffer.destroy === 'function'
  ) {
    currentTextureReadbackBuffer.destroy();
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
