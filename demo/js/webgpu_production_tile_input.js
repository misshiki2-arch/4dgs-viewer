import {
  buildNativeWebGpuProductionTileInputContract
} from './common_4dgs_production_frame_data_contracts.js';
import {
  buildNativeWebGpuProductionTileInputBindGroupEntries,
  buildNativeWebGpuProductionTileInputBindGroupLayoutEntries,
  buildNativeWebGpuProductionTileInputWgslBindings
} from './common_4dgs_production_tile_input_binding_contracts.js';

let nextProductionTileInputResourceIdentity = 1;

function createBuffer(device, data, usage, label) {
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

export async function buildNativeWebGpuProductionTileInput({
  device,
  workset,
  stateGpuResources,
  projectionParams
} = {}) {
  const recordCount = workset?.candidateIndices?.length ?? 0;
  const sourceWorksetResourceIdentity = workset?.contract?.resourceIdentity ?? null;
  const sourceStateResourceIdentity = stateGpuResources?.resourceIdentity ?? null;
  if (
    !device ||
    recordCount <= 0 ||
    !stateGpuResources?.statePositionBuffer ||
    !stateGpuResources?.renderAttributeBuffer ||
    !stateGpuResources?.footprintPayloadBuffer ||
    !(projectionParams instanceof Float32Array) ||
    typeof GPUBufferUsage === 'undefined' ||
    typeof GPUShaderStage === 'undefined'
  ) {
    return {
      gpuResource: null,
      contract: buildNativeWebGpuProductionTileInputContract({
        status: 'unavailable',
        sourceWorksetResourceIdentity,
        sourceStateResourceIdentity,
        recordCount,
        reason: 'native-production-tile-input-source-unavailable'
      })
    };
  }

  const shaderSource = `
struct Params {
  count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

${buildNativeWebGpuProductionTileInputWgslBindings()}

fn rowDot(row: vec4f, value: vec4f) -> f32 {
  return dot(row, value);
}

fn viewRow(index: u32) -> vec4f {
  return projectionParams[3u + index];
}

fn projectionRow(index: u32) -> vec4f {
  return projectionParams[7u + index];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let row = id.x;
  if (row >= params.count) {
    return;
  }
  let tileBase = row * 3u;
  let statePos = statePositions[row];
  let attributeBase = row * 2u;
  let attribute0 = renderAttributes[attributeBase + 0u];
  let attribute1 = renderAttributes[attributeBase + 1u];
  let footprintBase = row * 3u;
  let footprint0 = footprintPayload[footprintBase + 0u];
  let footprint1 = footprintPayload[footprintBase + 1u];
  let header = projectionParams[0u];
  let scale = projectionParams[1u];
  let intrinsics = projectionParams[2u];
  let renderW = header.y;
  let renderH = header.z;
  let sx = scale.x;
  let sy = scale.y;
  let pixelXSign = scale.z;
  let mv4 = vec4f(
    rowDot(viewRow(0u), statePos),
    rowDot(viewRow(1u), statePos),
    rowDot(viewRow(2u), statePos),
    rowDot(viewRow(3u), statePos)
  );
  var px = 0.0;
  var py = 0.0;
  var depth = 0.0;
  var projectionOk = false;
  if (header.x > 0.5) {
    depth = mv4.z;
    projectionOk = depth > 1e-6;
    px = (pixelXSign * intrinsics.x * (mv4.x / max(depth, 1e-8)) + intrinsics.z) * sx;
    py = (intrinsics.y * (mv4.y / max(depth, 1e-8)) + intrinsics.w) * sy;
  } else {
    depth = -mv4.z;
    let clip = vec4f(
      rowDot(projectionRow(0u), mv4),
      rowDot(projectionRow(1u), mv4),
      rowDot(projectionRow(2u), mv4),
      rowDot(projectionRow(3u), mv4)
    );
    let invW = 1.0 / (clip.w + 1e-7);
    let ndcX = clip.x * invW;
    let ndcY = clip.y * invW;
    projectionOk = depth > 1e-6;
    px = (((ndcX + 1.0) * renderW - 1.0) * 0.5) * sx;
    py = (((ndcY + 1.0) * renderH - 1.0) * 0.5) * sy;
  }
  let projectedInBounds =
    px >= 0.0 && py >= 0.0 && px < renderW * sx && py < renderH * sy;
  let valid =
    statePos.w > 0.5 &&
    projectionOk &&
    projectedInBounds &&
    footprint1.z > 0.0 &&
    attribute0.y > 0.0;
  if (!valid) {
    tileInputs[tileBase + 0u] = vec4f(0.0, 0.0, -1.0, 0.0);
    tileInputs[tileBase + 1u] = vec4f(0.0);
    tileInputs[tileBase + 2u] = vec4f(0.0);
    return;
  }
  tileInputs[tileBase + 0u] = vec4f(px, py, footprint1.z, depth);
  tileInputs[tileBase + 1u] = vec4f(
    footprint0.x,
    footprint0.y,
    depth,
    footprint0.z
  );
  tileInputs[tileBase + 2u] = vec4f(
    attribute0.z,
    attribute0.w,
    attribute1.x,
    attribute0.y
  );
}`;
  const shader = device.createShaderModule({
    label: 'phase3-native-production-tile-input-wgsl',
    code: shaderSource
  });
  const bindGroupLayoutEntries =
    buildNativeWebGpuProductionTileInputBindGroupLayoutEntries({
      computeVisibility: GPUShaderStage.COMPUTE
    });
  const bindGroupLayout = device.createBindGroupLayout({
    label: 'phase3-native-production-tile-input-bind-group-layout',
    entries: bindGroupLayoutEntries
  });
  const pipelineLayout = device.createPipelineLayout({
    label: 'phase3-native-production-tile-input-pipeline-layout',
    bindGroupLayouts: [bindGroupLayout]
  });
  const pipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shader, entryPoint: 'main' }
  });
  const projectionBuffer = createBuffer(
    device,
    projectionParams,
    GPUBufferUsage.STORAGE,
    'phase3-native-production-projection-params'
  );
  const outputBuffer = device.createBuffer({
    label: 'phase3-native-production-tile-input-resource',
    size: Math.max(16, recordCount * 12 * Float32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE
  });
  const paramsBuffer = createBuffer(
    device,
    new Uint32Array([recordCount, 0, 0, 0]),
    GPUBufferUsage.UNIFORM,
    'phase3-native-production-tile-input-params'
  );
  const bindGroupEntries =
    buildNativeWebGpuProductionTileInputBindGroupEntries({
      statePositions: stateGpuResources.statePositionBuffer,
      renderAttributes: stateGpuResources.renderAttributeBuffer,
      footprintPayload: stateGpuResources.footprintPayloadBuffer,
      projectionParams: projectionBuffer,
      tileInputs: outputBuffer,
      params: paramsBuffer
    });
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: bindGroupEntries
  });
  const encoder = device.createCommandEncoder({
    label: 'phase3-native-production-tile-input-encoder'
  });
  const pass = encoder.beginComputePass({
    label: 'phase3-native-production-tile-input-pass'
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(recordCount / 64)));
  pass.end();
  device.queue.submit([encoder.finish()]);
  if (typeof device.queue.onSubmittedWorkDone === 'function') {
    await device.queue.onSubmittedWorkDone();
  }
  for (const buffer of [projectionBuffer, paramsBuffer]) {
    if (typeof buffer.destroy === 'function') buffer.destroy();
  }

  const resourceIdentity =
    `production-tile-input-resource-${nextProductionTileInputResourceIdentity++}`;
  const contract = buildNativeWebGpuProductionTileInputContract({
    sourceWorksetResourceIdentity,
    sourceStateResourceIdentity,
    resourceIdentity,
    recordCount,
    dispatchSubmitted: true,
    productionReadbackPerformed: false,
    javascriptVisibleSamplesMaterialized: false
  });
  return {
    gpuResource: {
      buffer: outputBuffer,
      resourceIdentity,
      sourceWorksetResourceIdentity,
      sourceStateResourceIdentity,
      recordCount,
      contractVersion: contract.contractVersion
    },
    contract
  };
}
