import {
  buildPopulationRasterSemanticCompanionLayoutContract
} from './common_4dgs_population_semantic_comparison_contracts.js';
import {
  buildWebGpuProductionInclusiveBoundsWgslHelper
} from './common_4dgs_bounds_contracts.js';

function positiveInteger(value) {
  const number = Number(value);
  return Number.isInteger(number) && number > 0 ? number : null;
}

function createUploadBuffer(device, data, usage, label) {
  const buffer = device.createBuffer({
    label,
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

export async function observePopulationRasterSemanticCompanion({
  device,
  tileInputResource,
  expectedSourceWorksetResourceIdentity,
  canvasWidth,
  canvasHeight,
  tileSize = 16
} = {}) {
  const recordCount = positiveInteger(tileInputResource?.recordCount) ?? 0;
  const width = positiveInteger(canvasWidth) ?? 0;
  const height = positiveInteger(canvasHeight) ?? 0;
  const size = positiveInteger(tileSize) ?? 0;
  const tileCols = size > 0 ? Math.ceil(width / size) : 0;
  const tileRows = size > 0 ? Math.ceil(height / size) : 0;
  const sourceWorksetResourceIdentity =
    tileInputResource?.sourceWorksetResourceIdentity ?? null;
  const sourceStateResourceIdentity =
    tileInputResource?.sourceStateResourceIdentity ?? null;
  const sourceTileInputResourceIdentity =
    tileInputResource?.resourceIdentity ?? null;
  const inputsReady =
    device &&
    tileInputResource?.buffer &&
    recordCount > 0 && width > 0 && height > 0 && size > 0 &&
    typeof expectedSourceWorksetResourceIdentity === 'string' &&
    expectedSourceWorksetResourceIdentity.length > 0 &&
    sourceWorksetResourceIdentity === expectedSourceWorksetResourceIdentity &&
    typeof sourceStateResourceIdentity === 'string' &&
    sourceStateResourceIdentity.length > 0 &&
    typeof sourceTileInputResourceIdentity === 'string' &&
    sourceTileInputResourceIdentity.length > 0 &&
    typeof GPUBufferUsage !== 'undefined' &&
    typeof GPUMapMode !== 'undefined';
  if (!inputsReady) {
    return {
      evidence: new Float32Array(0),
      contract: buildPopulationRasterSemanticCompanionLayoutContract({
        recordCount,
        evidenceFloatCount: 0,
        sourceWorksetResourceIdentity,
        sourceStateResourceIdentity,
        sourceTileInputResourceIdentity,
        canvasWidth: width,
        canvasHeight: height,
        tileSize: size,
        tileCols,
        tileRows,
        observerOwnedBuffersDestroyed: true,
        reason: 'population-raster-semantic-observer-input-unavailable'
      })
    };
  }

  const evidenceFloatCount = recordCount * 12;
  const evidenceByteSize = evidenceFloatCount * Float32Array.BYTES_PER_ELEMENT;
  const ownedBuffers = [];
  let stagingBuffer = null;
  let stagingMapped = false;
  let evidence = null;
  let observerDispatchSubmitted = false;
  let observerReadbackCompleted = false;
  try {
    const shader = device.createShaderModule({
      label: 'phase3-population-raster-semantic-observer-wgsl',
      code: `
struct Params {
  count: u32,
  canvasWidth: u32,
  canvasHeight: u32,
  tileSize: u32,
  tileCols: u32,
  tileRows: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> tileInputs: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> companionEvidence: array<vec4f>;
@group(0) @binding(2) var<uniform> params: Params;

${buildWebGpuProductionInclusiveBoundsWgslHelper()}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let row = id.x;
  if (row >= params.count) { return; }
  let tileBase = row * 3u;
  let evidenceBase = row * 3u;
  let centerRadiusDepth = tileInputs[tileBase + 0u];
  let colorAlpha = tileInputs[tileBase + 2u];
  let rasterEligible = centerRadiusDepth.z > 0.0 && colorAlpha.w > 0.0;
  if (!rasterEligible) {
    companionEvidence[evidenceBase + 0u] = vec4f(0.0);
    companionEvidence[evidenceBase + 1u] = vec4f(0.0);
    companionEvidence[evidenceBase + 2u] = vec4f(0.0);
    return;
  }
  let pixelBounds = productionInclusivePixelBounds(
    centerRadiusDepth,
    params.canvasWidth,
    params.canvasHeight
  );
  let tileBounds = productionCudaAlignedInclusiveTileBounds(
    centerRadiusDepth,
    params.tileSize,
    params.tileCols,
    params.tileRows
  );
  if (tileBounds.nonEmpty == 0u) {
    companionEvidence[evidenceBase + 0u] = vec4f(0.0);
    companionEvidence[evidenceBase + 1u] = vec4f(0.0);
    companionEvidence[evidenceBase + 2u] = vec4f(0.0);
    return;
  }
  companionEvidence[evidenceBase + 0u] = vec4f(
    1.0,
    centerRadiusDepth.x,
    centerRadiusDepth.y,
    centerRadiusDepth.w
  );
  companionEvidence[evidenceBase + 1u] = pixelBounds;
  companionEvidence[evidenceBase + 2u] = vec4f(
    vec2f(tileBounds.minInclusive),
    vec2f(tileBounds.maxInclusive)
  );
}`
    });
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: shader, entryPoint: 'main' }
    });
    const outputBuffer = device.createBuffer({
      label: 'phase3-population-raster-semantic-companion-output',
      size: evidenceByteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    stagingBuffer = device.createBuffer({
      label: 'phase3-population-raster-semantic-companion-staging',
      size: evidenceByteSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    const paramsBuffer = createUploadBuffer(
      device,
      new Uint32Array([
        recordCount, width, height, size, tileCols, tileRows, 0, 0
      ]),
      GPUBufferUsage.UNIFORM,
      'phase3-population-raster-semantic-observer-params'
    );
    ownedBuffers.push(outputBuffer, stagingBuffer, paramsBuffer);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tileInputResource.buffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } }
      ]
    });
    const encoder = device.createCommandEncoder({
      label: 'phase3-population-raster-semantic-observer-encoder'
    });
    const pass = encoder.beginComputePass({
      label: 'phase3-population-raster-semantic-observer-pass'
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(recordCount / 64)));
    pass.end();
    encoder.copyBufferToBuffer(
      outputBuffer,
      0,
      stagingBuffer,
      0,
      evidenceByteSize
    );
    device.queue.submit([encoder.finish()]);
    observerDispatchSubmitted = true;
    if (typeof device.queue.onSubmittedWorkDone === 'function') {
      await device.queue.onSubmittedWorkDone();
    }
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    stagingMapped = true;
    evidence = new Float32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();
    stagingMapped = false;
    observerReadbackCompleted = true;
  } finally {
    if (stagingMapped && typeof stagingBuffer?.unmap === 'function') {
      stagingBuffer.unmap();
    }
    for (const buffer of ownedBuffers) {
      if (typeof buffer?.destroy === 'function') buffer.destroy();
    }
  }

  return {
    evidence,
    contract: buildPopulationRasterSemanticCompanionLayoutContract({
      recordCount,
      evidenceFloatCount: evidence?.length ?? 0,
      sourceWorksetResourceIdentity,
      sourceStateResourceIdentity,
      sourceTileInputResourceIdentity,
      canvasWidth: width,
      canvasHeight: height,
      tileSize: size,
      tileCols,
      tileRows,
      observerDispatchSubmitted,
      observerReadbackCompleted,
      observerOwnedBuffersDestroyed: true
    })
  };
}
