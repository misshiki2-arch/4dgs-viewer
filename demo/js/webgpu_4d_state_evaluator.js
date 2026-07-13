import {
  buildWebGpu4DStateSourceContract,
  buildWebGpuGaussianAttributeEvaluationContract,
  buildWebGpuGaussianFootprintEvaluationContract
} from './common_4dgs_record_contracts.js';

function toFiniteNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function createBuffer(device, data, usage) {
  const buffer = device.createBuffer({
    size: Math.max(4, Math.ceil(data.byteLength / 4) * 4),
    usage,
    mappedAtCreation: true
  });
  const mapped = new Uint8Array(buffer.getMappedRange());
  mapped.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  buffer.unmap();
  return buffer;
}

function buildStateEvaluatorParams({ count, buildConfig }) {
  const data = new ArrayBuffer(16);
  const view = new DataView(data);
  view.setUint32(0, count >>> 0, true);
  view.setFloat32(4, toFiniteNumber(buildConfig?.timestamp, 0), true);
  view.setFloat32(8, toFiniteNumber(buildConfig?.scalingModifier, 1), true);
  view.setFloat32(12, toFiniteNumber(buildConfig?.sigmaScale, 1), true);
  return new Uint8Array(data);
}

function buildCandidateTimeScale(raw, candidateIndices) {
  const out = new Float32Array(candidateIndices.length * 4);
  for (let row = 0; row < candidateIndices.length; row += 1) {
    const srcIndex = candidateIndices[row] | 0;
    const o = row * 4;
    out[o + 0] = toFiniteNumber(raw.t?.[srcIndex * raw.tDim], 0);
    out[o + 1] = toFiniteNumber(raw.scale_t?.[srcIndex * raw.scaleTDim], 1);
    out[o + 2] = raw.tDim > 0 ? 1 : 0;
    out[o + 3] = raw.scaleTDim > 0 ? 1 : 0;
  }
  return out;
}

function buildCandidateAttributeInput(raw, candidateIndices) {
  const out = new Float32Array(candidateIndices.length * 8);
  for (let row = 0; row < candidateIndices.length; row += 1) {
    const srcIndex = candidateIndices[row] | 0;
    const fdcBase = srcIndex * (raw.fdcDim ?? 0);
    const scaleBase = srcIndex * (raw.scaleXYZDim ?? 0);
    const sx = toFiniteNumber(raw.scale_xyz?.[scaleBase + 0], 0.01);
    const sy = toFiniteNumber(raw.scale_xyz?.[scaleBase + 1], sx);
    const sz = toFiniteNumber(raw.scale_xyz?.[scaleBase + 2], sx);
    const o = row * 8;
    out[o + 0] = toFiniteNumber(raw.f_dc?.[fdcBase + 0], 0);
    out[o + 1] = toFiniteNumber(raw.f_dc?.[fdcBase + 1], 0);
    out[o + 2] = toFiniteNumber(raw.f_dc?.[fdcBase + 2], 0);
    out[o + 3] = Math.max(1e-6, (sx + sy + sz) / 3);
    out[o + 4] = Math.max(1e-6, sx);
    out[o + 5] = Math.max(1e-6, sy);
    out[o + 6] = Math.max(1e-6, sz);
    out[o + 7] = raw.scaleXYZDim >= 3 ? 111 : 0;
  }
  return out;
}

function summarizeComputedStatePositions(statePositions) {
  const count = Math.floor((statePositions?.length ?? 0) / 4);
  let computed4DStatePositionCount = 0;
  let unavailableStatePositionCount = 0;
  for (let row = 0; row < count; row += 1) {
    const w = Number(statePositions[row * 4 + 3]);
    if (w > 0.89 && w < 0.99) {
      computed4DStatePositionCount += 1;
    } else {
      unavailableStatePositionCount += 1;
    }
  }
  return { computed4DStatePositionCount, unavailableStatePositionCount };
}

function summarizeComputedRenderAttributes(renderAttributes) {
  const count = Math.floor((renderAttributes?.length ?? 0) / 8);
  let computedRenderAttributeCount = 0;
  let radiusSum = 0;
  let alphaSum = 0;
  for (let row = 0; row < count; row += 1) {
    const o = row * 8;
    const radius = Number(renderAttributes[o + 0]);
    const alpha = Number(renderAttributes[o + 1]);
    const sourceCode = Number(renderAttributes[o + 6]);
    if (radius > 0 && alpha > 0 && sourceCode === 81) {
      computedRenderAttributeCount += 1;
      radiusSum += radius;
      alphaSum += alpha;
    }
  }
  const averageComputedRadiusPx =
    computedRenderAttributeCount > 0
      ? radiusSum / computedRenderAttributeCount
      : null;
  return {
    computedRenderAttributeCount,
    averageComputedRadiusPx,
    averageComputedAlpha:
      computedRenderAttributeCount > 0
        ? alphaSum / computedRenderAttributeCount
        : null,
    normalBackendPointRadiusPx:
      averageComputedRadiusPx == null
        ? null
        : Math.max(3, Math.min(14, Math.round(averageComputedRadiusPx)))
  };
}

function summarizeComputedFootprintPayload(footprintPayload) {
  const count = Math.floor((footprintPayload?.length ?? 0) / 12);
  let computedFootprintPayloadCount = 0;
  let conicXSum = 0;
  let areaSum = 0;
  for (let row = 0; row < count; row += 1) {
    const o = row * 12;
    const conicX = Number(footprintPayload[o + 0]);
    const radius = Number(footprintPayload[o + 6]);
    const sourceCode = Number(footprintPayload[o + 8]);
    const area = Number(footprintPayload[o + 10]);
    if (conicX > 0 && radius > 0 && (sourceCode === 82 || sourceCode === 111)) {
      computedFootprintPayloadCount += 1;
      conicXSum += conicX;
      areaSum += Number.isFinite(area) ? area : 0;
    }
  }
  return {
    computedFootprintPayloadCount,
    averageComputedConicX:
      computedFootprintPayloadCount > 0
        ? conicXSum / computedFootprintPayloadCount
        : null,
    averageComputedFootprintAreaPx:
      computedFootprintPayloadCount > 0
        ? areaSum / computedFootprintPayloadCount
        : null
  };
}

export async function buildWebGpu4DStatePositionsForCandidates({
  device,
  raw,
  candidateIndices,
  rawXyzOpacity,
  buildConfig
}) {
  const count = candidateIndices?.length ?? 0;
  if (!device || !raw || !candidateIndices || !rawXyzOpacity || count <= 0) {
    return {
      statePositions: new Float32Array(0),
      renderAttributes: new Float32Array(0),
      footprintPayload: new Float32Array(0),
      contract: buildWebGpu4DStateSourceContract({
        status: 'unavailable',
        stateSourceMode: 'webgpu-partial-4d-state-evaluator',
        reason: 'webgpu-4d-state-evaluator-input-unavailable'
      }),
      gaussianAttributeEvaluationContract:
        buildWebGpuGaussianAttributeEvaluationContract({
          status: 'unavailable',
          reason: 'webgpu-gaussian-attribute-evaluator-input-unavailable'
      }),
      gaussianFootprintEvaluationContract:
        buildWebGpuGaussianFootprintEvaluationContract({
          status: 'unavailable',
          reason: 'webgpu-gaussian-footprint-evaluator-input-unavailable'
        })
    };
  }

  const timeScale = buildCandidateTimeScale(raw, candidateIndices);
  const attributeInput = buildCandidateAttributeInput(raw, candidateIndices);
  const shader = device.createShaderModule({
    label: 'phase3-step81-partial-4d-state-and-attribute-evaluator-wgsl',
    code: `
struct Params {
  count: u32,
  timestamp: f32,
  scalingModifier: f32,
  sigmaScale: f32,
};

@group(0) @binding(0) var<storage, read> rawXyzOpacity: array<vec4f>;
@group(0) @binding(1) var<storage, read> timeScale: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> statePositions: array<vec4f>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> attributeInput: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> renderAttributes: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> footprintPayload: array<vec4f>;

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let row = id.x;
  if (row >= params.count) {
    return;
  }
  let raw0 = rawXyzOpacity[row];
  let ts = timeScale[row];
  let hasTime = ts.z > 0.5;
  let hasScaleT = ts.w > 0.5;
  let scaleT = max(select(1.0, ts.y, hasScaleT) * params.scalingModifier * params.sigmaScale, 1e-6);
  let dt = select(0.0, params.timestamp - ts.x, hasTime);
  let temporal = clamp(dt / scaleT, -1.0, 1.0);

  // Step80 intentionally implements a partial state evaluation boundary:
  // time parameters affect the emitted position, while full covariance/rotation
  // parity remains deferred to later WebGPU 4D state work.
  let pos = raw0.xyz + vec3f(0.015 * temporal, -0.010 * temporal, 0.005 * temporal);
  statePositions[row] = vec4f(pos, 0.9);

  let attrBaseInput = row * 2u;
  let attrs = attributeInput[attrBaseInput + 0u];
  let scaleInfo = attributeInput[attrBaseInput + 1u];
  let temporalWeight = exp(-0.5 * temporal * temporal);
  let alpha = clamp(sigmoid(raw0.w) * temporalWeight, 0.05, 0.99);
  let rgb = clamp(attrs.rgb + vec3f(0.5), vec3f(0.0), vec3f(1.0));
  let radiusPx = clamp(attrs.w * 900.0 + 2.0 + abs(temporal) * 2.0, 3.0, 14.0);
  let attrBase = row * 2u;
  renderAttributes[attrBase + 0u] = vec4f(radiusPx, alpha, rgb.r, rgb.g);
  renderAttributes[attrBase + 1u] = vec4f(rgb.b, temporalWeight, 81.0, 0.0);

  let meanScale = max(attrs.w, 1e-6);
  let scaleX = max(scaleInfo.x, 1e-6);
  let scaleY = max(scaleInfo.y, 1e-6);
  let anisotropyX = clamp(scaleX / meanScale, 0.35, 2.75);
  let anisotropyY = clamp(scaleY / meanScale, 0.35, 2.75);
  let radiusX = max(radiusPx * anisotropyX, 1.0);
  let radiusY = max(radiusPx * anisotropyY, 1.0);
  let conservativeRadiusPx = max(radiusPx, max(radiusX, radiusY));
  let sigmaX = max(radiusX / 3.0, 0.5);
  let sigmaY = max(radiusY / 3.0, 0.5);
  let varianceX = sigmaX * sigmaX;
  let varianceY = sigmaY * sigmaY;
  let conicX = 1.0 / varianceX;
  let conicZ = 1.0 / varianceY;
  let footprintAreaPx = 3.14159265 * radiusX * radiusY;
  let footprintBase = row * 3u;
  footprintPayload[footprintBase + 0u] = vec4f(conicX, 0.0, conicZ, varianceX);
  footprintPayload[footprintBase + 1u] = vec4f(0.0, varianceY, conservativeRadiusPx, raw0.z);
  footprintPayload[footprintBase + 2u] = vec4f(111.0, abs(raw0.z), footprintAreaPx, 0.0);
}`
  });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: shader, entryPoint: 'main' }
  });
  const rawBuffer = createBuffer(device, rawXyzOpacity, GPUBufferUsage.STORAGE);
  const timeScaleBuffer = createBuffer(device, timeScale, GPUBufferUsage.STORAGE);
  const attributeInputBuffer = createBuffer(
    device,
    attributeInput,
    GPUBufferUsage.STORAGE
  );
  const outputByteLength = Math.max(4, count * 4 * Float32Array.BYTES_PER_ELEMENT);
  const attributeOutputByteLength = Math.max(
    4,
    count * 8 * Float32Array.BYTES_PER_ELEMENT
  );
  const footprintOutputByteLength = Math.max(
    4,
    count * 12 * Float32Array.BYTES_PER_ELEMENT
  );
  const outputBuffer = device.createBuffer({
    size: outputByteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const attributeOutputBuffer = device.createBuffer({
    size: attributeOutputByteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const footprintOutputBuffer = device.createBuffer({
    size: footprintOutputByteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const paramsBuffer = createBuffer(
    device,
    buildStateEvaluatorParams({
      count,
      buildConfig
    }),
    GPUBufferUsage.UNIFORM
  );
  const readbackBuffer = device.createBuffer({
    size: outputByteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const attributeReadbackBuffer = device.createBuffer({
    size: attributeOutputByteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const footprintReadbackBuffer = device.createBuffer({
    size: footprintOutputByteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: rawBuffer } },
      { binding: 1, resource: { buffer: timeScaleBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: paramsBuffer } },
      { binding: 4, resource: { buffer: attributeInputBuffer } },
      { binding: 5, resource: { buffer: attributeOutputBuffer } },
      { binding: 6, resource: { buffer: footprintOutputBuffer } }
    ]
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputByteLength);
  encoder.copyBufferToBuffer(
    attributeOutputBuffer,
    0,
    attributeReadbackBuffer,
    0,
    attributeOutputByteLength
  );
  encoder.copyBufferToBuffer(
    footprintOutputBuffer,
    0,
    footprintReadbackBuffer,
    0,
    footprintOutputByteLength
  );
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const statePositions = new Float32Array(readbackBuffer.getMappedRange().slice(0));
  readbackBuffer.unmap();
  await attributeReadbackBuffer.mapAsync(GPUMapMode.READ);
  const renderAttributes = new Float32Array(
    attributeReadbackBuffer.getMappedRange().slice(0)
  );
  attributeReadbackBuffer.unmap();
  await footprintReadbackBuffer.mapAsync(GPUMapMode.READ);
  const footprintPayload = new Float32Array(
    footprintReadbackBuffer.getMappedRange().slice(0)
  );
  footprintReadbackBuffer.unmap();
  const stateSummary = summarizeComputedStatePositions(statePositions);
  const attributeSummary = summarizeComputedRenderAttributes(renderAttributes);
  const footprintSummary = summarizeComputedFootprintPayload(footprintPayload);
  return {
    statePositions,
    renderAttributes,
    footprintPayload,
    contract: buildWebGpu4DStateSourceContract({
      stateSourceMode: 'webgpu-partial-4d-state-evaluator',
      candidateCount: count,
      statePositionCount: count,
      computed4DStatePositionCount: stateSummary.computed4DStatePositionCount,
      baselineStatePositionCount: 0,
      unavailableStatePositionCount: stateSummary.unavailableStatePositionCount,
      timestamp: buildConfig?.timestamp ?? null,
      stateParameterMode: 'viewer-build-config-webgpu-uniform',
      webgpuComputedStatePositions: true,
      webgpu4DStateEvaluationMode: 'partial-time-parameter-position-eval',
      full4DStateEvaluationInWgsl: false,
      rawXyzRepairInVisibleRecordComputeRequired: false,
      reason:
        stateSummary.computed4DStatePositionCount > 0
          ? null
          : 'webgpu-partial-4d-state-evaluator-produced-no-valid-state-positions'
    }),
    gaussianAttributeEvaluationContract:
      buildWebGpuGaussianAttributeEvaluationContract({
        candidateCount: count,
        computedRenderAttributeCount:
          attributeSummary.computedRenderAttributeCount,
        webgpuComputedRenderAttributes: true,
        computedAttributeFields: [
          'radiusPx',
          'colorAlpha.rgb',
          'colorAlpha.a',
          'temporalWeight'
        ],
        partialAttributeFields: [
          'radiusPx-from-scale-mean',
          'colorAlpha.rgb-from-f_dc-l0',
          'colorAlpha.a-from-opacity-and-temporal-weight'
        ],
        referenceAssistedAttributeFields: ['none-for-normal-backend-visible-samples'],
        deferredAttributeFields: [
          'full-conic',
          'full-covariance',
          'full-SH-color',
          'tile-range',
          'depth-sort'
        ],
        normalBackendPointRadiusPx:
          attributeSummary.normalBackendPointRadiusPx,
        averageComputedRadiusPx: attributeSummary.averageComputedRadiusPx,
        averageComputedAlpha: attributeSummary.averageComputedAlpha,
        renderAttributeClassification: 'partial-webgpu-computed',
        renderPayloadClassification:
          'partial-webgpu-gaussian-render-attributes',
        fullGaussianAttributeEvaluationInWgsl: false,
        reason:
          attributeSummary.computedRenderAttributeCount > 0
            ? null
            : 'webgpu-gaussian-attribute-evaluator-produced-no-valid-attributes'
    }),
    gaussianFootprintEvaluationContract:
      buildWebGpuGaussianFootprintEvaluationContract({
        candidateCount: count,
        computedFootprintPayloadCount:
          footprintSummary.computedFootprintPayloadCount,
        webgpuComputedFootprintPayload: true,
        computedFootprintFields: [
          'conic',
          'covariance2D',
          'radiusPx',
          'depth',
          'sortKey'
        ],
        partialFootprintFields: [
          'conic-scale-anisotropic-from-raw-scale-xy',
          'covariance2D-scale-anisotropic-from-raw-scale-xy',
          'sortKey-from-state-depth'
        ],
        baselineFootprintFields: [],
        fallbackFootprintFields: [],
        deferredFootprintFields: [
          'full-4d-covariance-projection',
          'rotation-aware-anisotropic-conic-parity',
          'camera-jacobian-screen-space-conic-parity',
          'gpu-aabb-from-projected-center',
          'gpu-tileRange-from-aabb',
          'depth-sort-dispatch'
        ],
        averageComputedConicX: footprintSummary.averageComputedConicX,
        averageComputedFootprintAreaPx:
          footprintSummary.averageComputedFootprintAreaPx,
        footprintPayloadClassification:
          'partial-webgpu-gaussian-footprint',
        fullGaussianFootprintEvaluationInWgsl: false,
        reason:
          footprintSummary.computedFootprintPayloadCount > 0
            ? null
            : 'webgpu-gaussian-footprint-evaluator-produced-no-valid-payload'
      })
  };
}
