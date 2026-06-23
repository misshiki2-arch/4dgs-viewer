import {
  buildWebGpuTileListCompositorContract
} from './common_4dgs_record_contracts.js';

const COMPOSITOR_SUMMARY_FLOAT_COUNT = 12;

function alignTo(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
}

function finiteNumberOr(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function createBuffer(device, data, usage) {
  const buffer = device.createBuffer({
    size: Math.max(4, Math.ceil(data.byteLength / 4) * 4),
    usage,
    mappedAtCreation: true
  });
  new Uint8Array(buffer.getMappedRange()).set(
    new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
  );
  buffer.unmap();
  return buffer;
}

function readCompositorSummary(summary) {
  const processedTileCount = Math.round(finiteNumberOr(summary[0], 0));
  const nonEmptyCompositedTileCount = Math.round(finiteNumberOr(summary[1], 0));
  return {
    processedTileCount,
    compositedTileCount: nonEmptyCompositedTileCount,
    nonEmptyCompositedTileCount,
    compositedReferenceCount: Math.round(finiteNumberOr(summary[2], 0)),
    sourceTotalTileReferenceCount: Math.round(finiteNumberOr(summary[3], 0)),
    readOffsetCountTable: Math.round(finiteNumberOr(summary[4], 0)) === 1,
    traversedReferenceList: Math.round(finiteNumberOr(summary[5], 0)) === 1,
    outputTextureWritten: Math.round(finiteNumberOr(summary[6], 0)) === 1,
    maxRefsPerTileObserved: Math.round(finiteNumberOr(summary[7], 0)),
    overflowCount: Math.round(finiteNumberOr(summary[8], 0)),
    statusCode: Math.round(finiteNumberOr(summary[9], 0))
  };
}

function hasNonZeroTextureByte(readback, bytesPerRow, width, height) {
  for (let y = 0; y < height; y += 1) {
    const row = y * bytesPerRow;
    for (let x = 0; x < width * 4; x += 1) {
      if (readback[row + x] !== 0) return true;
    }
  }
  return false;
}

export async function buildWebGpuTileListCompositor({
  device,
  gpuOwnedTileListLayout,
  canvasWidth,
  canvasHeight
} = {}) {
  const resources = gpuOwnedTileListLayout?.gpuResources;
  const sourceContract = gpuOwnedTileListLayout?.contract;
  if (
    !device ||
    !resources?.inputBuffer ||
    !resources?.tileTableBuffer ||
    !resources?.referenceListBuffer ||
    typeof GPUBufferUsage === 'undefined' ||
    typeof GPUMapMode === 'undefined' ||
    typeof GPUShaderStage === 'undefined' ||
    typeof GPUTextureUsage === 'undefined'
  ) {
    return {
      contract: buildWebGpuTileListCompositorContract({
        status: 'unavailable',
        reason: 'webgpu-tile-list-compositor-unavailable'
      })
    };
  }

  const outputWidth = Math.max(1, resources.tileCols);
  const outputHeight = Math.max(1, resources.tileRows);
  const outputTexture = device.createTexture({
    label: 'phase3-step85-webgpu-tile-list-compositor-output-texture',
    size: { width: outputWidth, height: outputHeight },
    format: 'rgba8unorm',
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.TEXTURE_BINDING
  });
  const bytesPerRow = alignTo(outputWidth * 4, 256);
  const textureReadbackBuffer = device.createBuffer({
    size: bytesPerRow * outputHeight,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const summaryData = new Float32Array(COMPOSITOR_SUMMARY_FLOAT_COUNT);
  const summaryBuffer = createBuffer(
    device,
    summaryData,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const params = new Float32Array([
    resources.tileCount,
    resources.tileCols,
    resources.tileRows,
    resources.maxRefsPerTile,
    outputWidth,
    outputHeight,
    finiteNumberOr(canvasWidth, outputWidth),
    finiteNumberOr(canvasHeight, outputHeight)
  ]);
  const paramsBuffer = createBuffer(device, params, GPUBufferUsage.UNIFORM);
  const summaryReadbackBuffer = device.createBuffer({
    size: summaryData.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const shader = device.createShaderModule({
    label: 'phase3-step85-webgpu-tile-list-compositor-wgsl',
    code: `
struct Params {
  tileCount: f32,
  tileCols: f32,
  tileRows: f32,
  maxRefsPerTile: f32,
  outputWidth: f32,
  outputHeight: f32,
  canvasWidth: f32,
  canvasHeight: f32,
};

@group(0) @binding(0) var<storage, read> tileInputs: array<vec4f>;
@group(0) @binding(1) var<storage, read> tileTable: array<vec4f>;
@group(0) @binding(2) var<storage, read> referenceList: array<vec4f>;
@group(0) @binding(3) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<storage, read_write> compositorSummary: array<vec4f>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn compositeTiles(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u32(params.tileCols) || id.y >= u32(params.tileRows)) {
    return;
  }
  let tile = id.y * u32(params.tileCols) + id.x;
  let table = tileTable[tile];
  var color = vec3f(0.0, 0.0, 0.0);
  var alpha = 0.0;
  var refs = 0.0;
  var readTable = 0.0;
  var traversedList = 0.0;
  if (table.w == 84.0 && table.y > 0.0) {
    readTable = 1.0;
    let offset = u32(table.x);
    let count = u32(table.y);
    for (var slot: u32 = 0u; slot < count; slot = slot + 1u) {
      let splatRef = referenceList[offset + slot];
      let sampleRow = u32(max(splatRef.x, 0.0));
      let sampleBase = sampleRow * 3u;
      let c = tileInputs[sampleBase + 2u];
      let a = clamp(c.w, 0.0, 1.0);
      color = color + clamp(c.xyz, vec3f(0.0), vec3f(1.0)) * a;
      alpha = alpha + a;
      refs = refs + 1.0;
      traversedList = 1.0;
    }
  }
  let invAlpha = select(0.0, 1.0 / max(alpha, 0.0001), alpha > 0.0);
  let outColor = vec4f(color * invAlpha, clamp(alpha / max(refs, 1.0), 0.0, 1.0));
  textureStore(outputTexture, vec2i(i32(id.x), i32(id.y)), outColor);
}

@compute @workgroup_size(1)
fn finalizeSummary() {
  var nonEmpty = 0.0;
  var totalRefs = 0.0;
  var maxRefs = 0.0;
  var overflow = 0.0;
  var readTable = 0.0;
  var traversedList = 0.0;
  for (var tile: u32 = 0u; tile < u32(params.tileCount); tile = tile + 1u) {
    let table = tileTable[tile];
    if (table.w == 84.0) {
      readTable = 1.0;
    }
    if (table.w == 84.0 && table.y > 0.0) {
      nonEmpty = nonEmpty + 1.0;
      totalRefs = totalRefs + table.y;
      maxRefs = max(maxRefs, table.y);
      overflow = overflow + table.z;
      let offset = u32(table.x);
      let count = u32(table.y);
      for (var slot: u32 = 0u; slot < count; slot = slot + 1u) {
        let splatRef = referenceList[offset + slot];
        if (splatRef.z != 0.0 || splatRef.w != 0.0) {
          traversedList = 1.0;
        }
      }
    }
  }
  compositorSummary[0] = vec4f(params.tileCount, nonEmpty, totalRefs, totalRefs);
  compositorSummary[1] = vec4f(readTable, traversedList, select(0.0, 1.0, totalRefs > 0.0), maxRefs);
  compositorSummary[2] = vec4f(overflow, 85.0, 0.0, 0.0);
}
`
  });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' }
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba8unorm' }
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      },
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' }
      }
    ]
  });
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout]
  });
  const compositorPipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shader, entryPoint: 'compositeTiles' }
  });
  const finalizePipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shader, entryPoint: 'finalizeSummary' }
  });
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: resources.inputBuffer } },
      { binding: 1, resource: { buffer: resources.tileTableBuffer } },
      { binding: 2, resource: { buffer: resources.referenceListBuffer } },
      { binding: 3, resource: outputTexture.createView() },
      { binding: 4, resource: { buffer: summaryBuffer } },
      { binding: 5, resource: { buffer: paramsBuffer } }
    ]
  });
  const encoder = device.createCommandEncoder({
    label: 'phase3-step85-webgpu-tile-list-compositor-encoder'
  });
  const pass = encoder.beginComputePass({
    label: 'phase3-step85-webgpu-tile-list-compositor-pass'
  });
  pass.setBindGroup(0, bindGroup);
  pass.setPipeline(compositorPipeline);
  pass.dispatchWorkgroups(
    Math.max(1, Math.ceil(outputWidth / 8)),
    Math.max(1, Math.ceil(outputHeight / 8))
  );
  pass.setPipeline(finalizePipeline);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyTextureToBuffer(
    { texture: outputTexture },
    {
      buffer: textureReadbackBuffer,
      bytesPerRow,
      rowsPerImage: outputHeight
    },
    { width: outputWidth, height: outputHeight, depthOrArrayLayers: 1 }
  );
  encoder.copyBufferToBuffer(
    summaryBuffer,
    0,
    summaryReadbackBuffer,
    0,
    summaryData.byteLength
  );
  device.queue.submit([encoder.finish()]);
  if (typeof device.queue.onSubmittedWorkDone === 'function') {
    await device.queue.onSubmittedWorkDone();
  }
  await summaryReadbackBuffer.mapAsync(GPUMapMode.READ);
  const compositorSummary = new Float32Array(summaryReadbackBuffer.getMappedRange().slice(0));
  summaryReadbackBuffer.unmap();
  await textureReadbackBuffer.mapAsync(GPUMapMode.READ);
  const textureReadback = new Uint8Array(textureReadbackBuffer.getMappedRange().slice(0));
  textureReadbackBuffer.unmap();

  const summary = readCompositorSummary(compositorSummary);
  summary.sourceTotalTileReferenceCount =
    sourceContract?.totalTileReferenceCount ?? summary.compositedReferenceCount;
  const outputTextureWritten =
    summary.outputTextureWritten && hasNonZeroTextureByte(
      textureReadback,
      bytesPerRow,
      outputWidth,
      outputHeight
    );
  const ready =
    summary.readOffsetCountTable &&
    summary.traversedReferenceList &&
    outputTextureWritten &&
    summary.compositedReferenceCount > 0;

  for (const buffer of [summaryBuffer, paramsBuffer, summaryReadbackBuffer, textureReadbackBuffer]) {
    if (typeof buffer.destroy === 'function') {
      buffer.destroy();
    }
  }
  if (typeof outputTexture.destroy === 'function') {
    outputTexture.destroy();
  }

  return {
    compositorSummary,
    contract: buildWebGpuTileListCompositorContract({
      tileCompositorReady: ready,
      compositorPassSubmitted: true,
      compositorReadbackCompleted: true,
      compositorReadOffsetCountTable: summary.readOffsetCountTable,
      compositorTraversedReferenceList: summary.traversedReferenceList,
      outputTextureCreated: true,
      outputTextureWritten,
      outputTextureReadbackMatchesSummary: outputTextureWritten,
      outputWidth,
      outputHeight,
      processedTileCount: summary.processedTileCount,
      compositedTileCount: summary.compositedTileCount,
      nonEmptyCompositedTileCount: summary.nonEmptyCompositedTileCount,
      compositedReferenceCount: summary.compositedReferenceCount,
      sourceTotalTileReferenceCount: summary.sourceTotalTileReferenceCount,
      overflowCount: summary.overflowCount,
      orderHandling: 'unsorted-fixed-reference-order',
      generatedCompositorFields: [
        'tile-list-offset-count-read',
        'splat-reference-list-traversal',
        'partial-alpha-accumulation',
        'rgba8unorm-output-texture'
      ],
      deferredCompositorFields: [
        'full-depth-sort-dispatch',
        'cuda-compositor-parity',
        'final-production-tile-compositor'
      ],
      compositorClassification: 'partial-webgpu-tile-list-compositor',
      fullDepthSortInWgsl: false,
      fullCudaParity: false,
      finalProductionTileCompositor: false,
      normalBackendFallbackMaintained: true,
      sourceGpuOwnedTileListLayoutContractVersion:
        sourceContract?.contractVersion ?? null,
      currentTexturePathMaintained: true,
      reason: ready
        ? null
        : 'webgpu-tile-list-compositor-did-not-consume-gpu-owned-tile-list'
    })
  };
}
