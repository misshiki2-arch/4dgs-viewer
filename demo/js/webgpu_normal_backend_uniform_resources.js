import {
  buildCameraAwareVisibleOutputContract,
  buildNormalBackendOutputContract,
  buildUnavailableNormalBackendOutputContracts,
  validateNormalBackendOutputContracts
} from './common_4dgs_backend_output_contracts.js';
import {
  runWebGpuOnlyGuardedPresentationAdapter
} from './webgpu_guarded_presentation_adapter.js';

export const WEBGPU_NORMAL_BACKEND_UNIFORM_RESOURCE_LIFECYCLE_CONTRACT_VERSION =
  'phase3-step72-normal-backend-uniform-resource-lifecycle-v1';

export const WEBGPU_NORMAL_BACKEND_UNIFORM_SHADER_CONSUMPTION_CONTRACT_VERSION =
  'phase3-step72-normal-backend-uniform-sample-color-output-consumption-v1';

export const WEBGPU_NORMAL_BACKEND_SAMPLE_RESOURCE_CONTRACT_VERSION =
  'phase3-step69-normal-backend-sample-storage-resource-v1';

export const WEBGPU_NORMAL_BACKEND_COLOR_OUTPUT_SURFACE_CONTRACT_VERSION =
  'phase3-step69-normal-backend-color-output-surface-resource-v1';

const SAMPLE_FLOAT_STRIDE = 20;

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
  const outputContracts = buildUnavailableNormalBackendOutputContracts(reason);
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
    sampleStorageBufferReadbackCompleted: false,
    sampleReadMatchesPackedSelectedSamples: false,
    sampleReadMatchesPackedNormalBackendSamples: false,
    colorOutputSurfaceCreated: false,
    colorOutputSurfaceWriteSubmitted: false,
    colorOutputSurfaceReadbackCompleted: false,
    colorOutputSurfaceDestroyed: false,
    normalBackendOutputResourceCreated: false,
    normalBackendOutputHandoffCopySubmitted: false,
    normalBackendOutputReadbackCompleted: false,
    normalBackendOutputResourceDestroyed: false,
    guardedPresentationAdapterCalled: false,
    guardedPresentationAdapterReady: false,
    viewerPresentationBridgeReady: false,
    renderTargetBridgeReady: false,
    currentTextureConnectionAttempted: false,
    currentTextureConnected: false,
    queueWriteBufferUsed: false,
    bindGroupReadyBoundary: false,
    bindGroupLayoutCreated: false,
    bindGroupCreatedThisStep: false,
    shaderConsumptionImplemented: false,
    uniformShaderConsumptionContract: createUnavailableConsumptionSummary(reason),
    ...outputContracts,
    reason,
    ...extra
  };
}

function createUnavailableConsumptionSummary(reason, extra = {}) {
  const outputContracts = buildUnavailableNormalBackendOutputContracts(reason);
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
    sampleReadMatchesPackedNormalBackendSamples: false,
    colorOutputSurfaceWritten: false,
    colorOutputSurfaceReadbackCompleted: false,
    colorOutputSurfaceMatchesExpected: false,
    normalBackendOutputHandoffCopySubmitted: false,
    normalBackendOutputReadbackCompleted: false,
    normalBackendOutputMatchesExpected: false,
    guardedPresentationAdapterCalled: false,
    guardedPresentationAdapterReady: false,
    viewerPresentationBridgeReady: false,
    renderTargetBridgeReady: false,
    currentTextureConnectionAttempted: false,
    currentTextureConnected: false,
    ...outputContracts,
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
  if (source === 'webgpuVisibleRecordDryRun.cameraAwareVisibleRecords') {
    return 75;
  }
  if (
    source ===
    'validationAssistedScreenCoarseBridge.cameraProjectionCandidateSamples'
  ) {
    return 76;
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
  const webgpuComputedRenderAttributeSampleCount = samples.filter(
    (sample) => sample?.renderAttributeSource === 'webgpu-gaussian-attribute-evaluator'
  ).length;
  const computedRenderPayloadConsumed =
    samples.length > 0 &&
    webgpuComputedRenderAttributeSampleCount === samples.length;
  const computedRadiusConsumed = samples.every(
    (sample) =>
      sample?.renderAttributeSource === 'webgpu-gaussian-attribute-evaluator' &&
      Number.isFinite(sample?.renderAttribute?.radiusPx)
  );
  const computedColorAlphaConsumed = samples.every(
    (sample) =>
      sample?.renderAttributeSource === 'webgpu-gaussian-attribute-evaluator' &&
      sample?.renderAttribute?.colorAlpha &&
      ['r', 'g', 'b', 'a'].every((field) =>
        Number.isFinite(sample.renderAttribute.colorAlpha[field])
      )
  );
  const computedTemporalWeightAvailable = samples.every(
    (sample) =>
      sample?.renderAttributeSource === 'webgpu-gaussian-attribute-evaluator' &&
      Number.isFinite(sample?.renderAttribute?.temporalWeight)
  );
  const webgpuComputedFootprintPayloadSampleCount = samples.filter(
    (sample) => sample?.footprintPayloadSource === 'webgpu-gaussian-footprint-evaluator'
  ).length;
  const computedFootprintPayloadConsumed =
    samples.length > 0 &&
    webgpuComputedFootprintPayloadSampleCount === samples.length;
  const computedConicConsumed = samples.every(
    (sample) =>
      sample?.footprintPayloadSource === 'webgpu-gaussian-footprint-evaluator' &&
      Array.isArray(sample?.conic) &&
      sample.conic.length >= 3 &&
      sample.conic.every((value) => Number.isFinite(value))
  );
  const computedAabbConsumed = samples.every(
    (sample) =>
      sample?.footprintPayloadSource === 'webgpu-gaussian-footprint-evaluator' &&
      Array.isArray(sample?.aabb) &&
      sample.aabb.length >= 4 &&
      sample.aabb.every((value) => Number.isFinite(value))
  );
  const computedTileRangeConsumed = samples.every(
    (sample) =>
      sample?.footprintPayloadSource === 'webgpu-gaussian-footprint-evaluator' &&
      Array.isArray(sample?.tileRange) &&
      sample.tileRange.length >= 4 &&
      sample.tileRange.every((value) => Number.isFinite(value))
  );
  const partialAabbDerivedConsumed = computedAabbConsumed;
  const partialTileRangeDerivedConsumed = computedTileRangeConsumed;
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
    const conic = Array.isArray(sample?.conic) ? sample.conic : [];
    const aabb = Array.isArray(sample?.aabb) ? sample.aabb : [];
    const tileRange = Array.isArray(sample?.tileRange) ? sample.tileRange : [];
    data[offset + 8] = finiteNumberOr(conic[0], 0);
    data[offset + 9] = finiteNumberOr(conic[1], 0);
    data[offset + 10] = finiteNumberOr(conic[2], 0);
    data[offset + 11] = finiteNumberOr(aabb[0], -1);
    data[offset + 12] = finiteNumberOr(aabb[1], -1);
    data[offset + 13] = finiteNumberOr(aabb[2], -1);
    data[offset + 14] = finiteNumberOr(aabb[3], -1);
    data[offset + 15] = finiteNumberOr(tileRange[0], -1);
    data[offset + 16] = finiteNumberOr(tileRange[1], -1);
    data[offset + 17] = finiteNumberOr(tileRange[2], -1);
    data[offset + 18] = finiteNumberOr(tileRange[3], -1);
    data[offset + 19] = finiteNumberOr(sample?.sortKey, 0);
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
        'recordIndex',
        'conic.x',
        'conic.y',
        'conic.z',
        'aabb.minX',
        'aabb.minY',
        'aabb.maxX',
        'aabb.maxY',
        'tileRange.minX',
        'tileRange.minY',
        'tileRange.maxX',
        'tileRange.maxY',
        'sortKey'
      ],
      sampleSources: [...new Set(samples.map((sample) => sample?.source ?? null))],
      renderAttributeSources: [
        ...new Set(samples.map((sample) => sample?.renderAttributeSource ?? null))
      ].filter((source) => source != null),
      webgpuComputedRenderAttributeSampleCount,
      computedRenderAttributeSampleCount:
        webgpuComputedRenderAttributeSampleCount,
      computedRenderPayloadConsumed,
      computedRadiusConsumed,
      computedColorAlphaConsumed,
      computedTemporalWeightAvailable,
      computedTemporalWeightUsage:
        computedTemporalWeightAvailable
          ? 'applied by WebGPU evaluator to colorAlpha.a before normal backend packing'
          : 'not-available-on-all-samples',
      normalBackendConsumedComputedRenderAttributes:
        computedRenderPayloadConsumed &&
        computedRadiusConsumed &&
        computedColorAlphaConsumed,
      footprintPayloadSources: [
        ...new Set(samples.map((sample) => sample?.footprintPayloadSource ?? null))
      ].filter((source) => source != null),
      webgpuComputedFootprintPayloadSampleCount,
      computedFootprintPayloadSampleCount:
        webgpuComputedFootprintPayloadSampleCount,
      computedFootprintPayloadConsumed,
      computedConicConsumed,
      computedAabbConsumed,
      computedTileRangeConsumed,
      partialAabbDerivedConsumed,
      partialTileRangeDerivedConsumed,
      aabbConsumptionMode:
        partialAabbDerivedConsumed
          ? 'partial-derived-from-webgpu-projected-px-py-and-computed-radius'
          : 'not-consumed',
      tileRangeConsumptionMode:
        partialTileRangeDerivedConsumed
          ? 'partial-derived-from-webgpu-projected-px-py-and-computed-radius'
          : 'not-consumed',
      fullGpuNativeAabbParityDeferred: true,
      fullGpuNativeTileRangeParityDeferred: true,
      normalBackendConsumedComputedFootprintPayload:
        computedFootprintPayloadConsumed &&
        computedConicConsumed &&
        computedAabbConsumed &&
        computedTileRangeConsumed
    }
  };
}

function buildExpectedColorOutputSurface(sampleData, sampleCount, {
  pointRadiusPx = 0
} = {}) {
  const radius = Math.max(0, Math.min(64, Math.floor(pointRadiusPx)));
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = 0;
  let maxY = 0;
  for (let i = 0; i < sampleCount; i++) {
    const offset = i * SAMPLE_FLOAT_STRIDE;
    const x = Math.floor(finiteNumberOr(sampleData[offset + 0], 0));
    const y = Math.floor(finiteNumberOr(sampleData[offset + 1], 0));
    minX = Math.min(minX, x - radius);
    minY = Math.min(minY, y - radius);
    maxX = Math.max(maxX, x + radius);
    maxY = Math.max(maxY, y + radius);
  }
  const originX = Number.isFinite(minX) ? minX : 0;
  const originY = Number.isFinite(minY) ? minY : 0;
  const boundedWidth = Math.max(1, maxX - originX + 1);
  const boundedHeight = Math.max(1, maxY - originY + 1);
  const data = new Float32Array(boundedWidth * boundedHeight * 4);
  let writtenPixelCount = 0;
  const writtenPixels = [];
  for (let i = 0; i < sampleCount; i++) {
    const sampleOffset = i * SAMPLE_FLOAT_STRIDE;
    const sourceX = Math.floor(finiteNumberOr(sampleData[sampleOffset + 0], 0));
    const sourceY = Math.floor(finiteNumberOr(sampleData[sampleOffset + 1], 0));
    for (let dy = -radius; dy <= radius; dy += 1) {
      for (let dx = -radius; dx <= radius; dx += 1) {
        const x = sourceX + dx - originX;
        const y = sourceY + dy - originY;
        if (x < 0 || y < 0 || x >= boundedWidth || y >= boundedHeight) {
          continue;
        }
        const pixelOffset = (y * boundedWidth + x) * 4;
        data[pixelOffset + 0] = finiteNumberOr(sampleData[sampleOffset + 2], 0);
        data[pixelOffset + 1] = finiteNumberOr(sampleData[sampleOffset + 3], 0);
        data[pixelOffset + 2] = finiteNumberOr(sampleData[sampleOffset + 4], 0);
        data[pixelOffset + 3] = finiteNumberOr(sampleData[sampleOffset + 5], 0);
        writtenPixelCount += 1;
      }
    }
    writtenPixels.push({ sourceX, sourceY, sampleIndex: i, radius });
  }
  return {
    data,
    summary: {
      contractVersion: WEBGPU_NORMAL_BACKEND_COLOR_OUTPUT_SURFACE_CONTRACT_VERSION,
      status: writtenPixelCount > 0 ? 'ok' : 'blocked',
      outputSurfaceMode: 'normal-backend-minimal-rgba-storage-buffer-surface',
      outputResourceKind: 'storage-buffer-rgba-float-surface',
      outputFormat: 'rgba32float',
      presentationReadyBoundary: true,
      connectedToViewerCanvasPresentation: false,
      surfaceWidth: boundedWidth,
      surfaceHeight: boundedHeight,
      outputExtent: { width: boundedWidth, height: boundedHeight },
      surfaceOriginPx: { x: originX, y: originY },
      coordinateOrigin: 'top-left-bounded-sample-pixel-origin',
      coordinateMapping:
        'surface pixels = floor(samplePx.xy) plus point radius, then rgba is copied from colorAlpha',
      outputPointRadiusPx: radius,
      surfacePixelCount: boundedWidth * boundedHeight,
      colorChannels: 4,
      packedFloat32Count: data.length,
      packedByteLength: data.byteLength,
      sampleCount,
      writtenPixelCount,
      writtenPixels,
      packedFieldInputs: [
        'samplePx.x',
        'samplePx.y',
        'colorAlpha.r',
        'colorAlpha.g',
        'colorAlpha.b',
        'colorAlpha.a'
      ],
      futurePresentationTargets: [
        'viewer-canvas-current-texture',
        'render-target-texture',
        'storage-texture-copy'
      ],
      productionPresentationConnected: false
    }
  };
}

async function consumeUniformWithMinimalCompute({
  device,
  uniformBuffer,
  sampleBuffer,
  uniformData,
  sampleData,
  minBindingSizeBytes,
  viewerCanvasState = null,
  cameraAwareVisibleOutputContract = null
}) {
  if (typeof GPUBufferUsage === 'undefined' || typeof GPUMapMode === 'undefined') {
    return createUnavailableConsumptionSummary('webgpu-buffer-usage-unavailable');
  }
  const expected = Array.from(uniformData.slice(0, 4));
  const expectedSample = Array.from(sampleData.slice(0, SAMPLE_FLOAT_STRIDE));
  const sampleCount = Math.floor(sampleData.length / SAMPLE_FLOAT_STRIDE);
  const expectedColorOutputSurface = buildExpectedColorOutputSurface(
    sampleData,
    sampleCount,
    {
      pointRadiusPx:
        cameraAwareVisibleOutputContract?.outputPointRadiusPx ?? 0
    }
  );
  const outputFloatCount = 4 + SAMPLE_FLOAT_STRIDE;
  const outputByteLength = outputFloatCount * 4;
  const outputBuffer = device.createBuffer({
    label: 'phase3-step68-normal-backend-uniform-sample-consumption-output',
    size: outputByteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const readbackBuffer = device.createBuffer({
    label: 'phase3-step68-normal-backend-uniform-sample-consumption-readback',
    size: outputByteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const colorOutputSurfaceBuffer = device.createBuffer({
    label: 'phase3-step68-normal-backend-color-output-surface-buffer',
    size: expectedColorOutputSurface.data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const colorOutputSurfaceReadbackBuffer = device.createBuffer({
    label: 'phase3-step68-normal-backend-color-output-surface-readback',
    size: expectedColorOutputSurface.data.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const normalBackendOutputHandoffBuffer = device.createBuffer({
    label: 'phase3-step69-normal-backend-output-handoff-buffer',
    size: expectedColorOutputSurface.data.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST
  });
  const normalBackendOutputHandoffReadbackBuffer = device.createBuffer({
    label: 'phase3-step69-normal-backend-output-handoff-readback',
    size: expectedColorOutputSurface.data.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: 'phase3-step68-normal-backend-uniform-sample-color-output-bind-group-layout',
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
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'storage',
          minBindingSize: expectedColorOutputSurface.data.byteLength
        }
      }
    ]
  });
  const bindGroup = device.createBindGroup({
    label: 'phase3-step68-normal-backend-uniform-sample-color-output-bind-group',
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
      { binding: 2, resource: { buffer: sampleBuffer } },
      { binding: 3, resource: { buffer: colorOutputSurfaceBuffer } }
    ]
  });
  const shader = device.createShaderModule({
    label: 'phase3-step68-normal-backend-uniform-sample-color-output-wgsl',
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
@group(0) @binding(3) var<storage, read_write> colorOutputSurface: array<f32>;

const sampleFloatStride: u32 = ${SAMPLE_FLOAT_STRIDE}u;
const sampleCount: u32 = ${sampleCount}u;
const surfaceWidth: u32 = ${expectedColorOutputSurface.summary.surfaceWidth}u;
const surfaceHeight: u32 = ${expectedColorOutputSurface.summary.surfaceHeight}u;
const surfaceOriginX: f32 = ${expectedColorOutputSurface.summary.surfaceOriginPx.x}.0;
const surfaceOriginY: f32 = ${expectedColorOutputSurface.summary.surfaceOriginPx.y}.0;
const pointRadius: i32 = ${expectedColorOutputSurface.summary.outputPointRadiusPx};

@compute @workgroup_size(1)
fn main() {
  output[0] = frameUniforms.frame.x;
  output[1] = frameUniforms.frame.y;
  output[2] = frameUniforms.frame.z;
  output[3] = frameUniforms.frame.w;
  for (var i: u32 = 0u; i < sampleFloatStride; i = i + 1u) {
    output[4u + i] = selectedSamples[i];
  }
  for (var sampleIndex: u32 = 0u; sampleIndex < sampleCount; sampleIndex = sampleIndex + 1u) {
    let base = sampleIndex * sampleFloatStride;
    let centerX = i32(floor(selectedSamples[base + 0u] - surfaceOriginX));
    let centerY = i32(floor(selectedSamples[base + 1u] - surfaceOriginY));
    for (var dy: i32 = -pointRadius; dy <= pointRadius; dy = dy + 1) {
      for (var dx: i32 = -pointRadius; dx <= pointRadius; dx = dx + 1) {
        let px = centerX + dx;
        let py = centerY + dy;
        if (px >= 0 && py >= 0 && u32(px) < surfaceWidth && u32(py) < surfaceHeight) {
          let pixelBase = ((u32(py) * surfaceWidth) + u32(px)) * 4u;
          colorOutputSurface[pixelBase + 0u] = selectedSamples[base + 2u];
          colorOutputSurface[pixelBase + 1u] = selectedSamples[base + 3u];
          colorOutputSurface[pixelBase + 2u] = selectedSamples[base + 4u];
          colorOutputSurface[pixelBase + 3u] = selectedSamples[base + 5u];
        }
      }
    }
  }
}
`
  });
  const pipelineLayout = device.createPipelineLayout({
    label: 'phase3-step68-normal-backend-uniform-sample-color-output-pipeline-layout',
    bindGroupLayouts: [bindGroupLayout]
  });
  const pipeline = device.createComputePipeline({
    label: 'phase3-step68-normal-backend-uniform-sample-color-output-pipeline',
    layout: pipelineLayout,
    compute: {
      module: shader,
      entryPoint: 'main'
    }
  });
  const encoder = device.createCommandEncoder({
    label: 'phase3-step68-normal-backend-uniform-sample-color-output-encoder'
  });
  const pass = encoder.beginComputePass({
    label: 'phase3-step68-normal-backend-uniform-sample-color-output-pass'
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputByteLength);
  encoder.copyBufferToBuffer(
    colorOutputSurfaceBuffer,
    0,
    colorOutputSurfaceReadbackBuffer,
    0,
    expectedColorOutputSurface.data.byteLength
  );
  encoder.copyBufferToBuffer(
    colorOutputSurfaceBuffer,
    0,
    normalBackendOutputHandoffBuffer,
    0,
    expectedColorOutputSurface.data.byteLength
  );
  encoder.copyBufferToBuffer(
    normalBackendOutputHandoffBuffer,
    0,
    normalBackendOutputHandoffReadbackBuffer,
    0,
    expectedColorOutputSurface.data.byteLength
  );
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
  await colorOutputSurfaceReadbackBuffer.mapAsync(GPUMapMode.READ);
  const actualColorOutputSurface = Array.from(
    new Float32Array(colorOutputSurfaceReadbackBuffer.getMappedRange()).slice(
      0,
      expectedColorOutputSurface.data.length
    )
  );
  colorOutputSurfaceReadbackBuffer.unmap();
  await normalBackendOutputHandoffReadbackBuffer.mapAsync(GPUMapMode.READ);
  const actualNormalBackendOutputHandoff = Array.from(
    new Float32Array(
      normalBackendOutputHandoffReadbackBuffer.getMappedRange()
    ).slice(0, expectedColorOutputSurface.data.length)
  );
  normalBackendOutputHandoffReadbackBuffer.unmap();
  const epsilon = 1e-6;
  const maxAbsDiff = actual.reduce((maxDiff, value, index) => {
    return Math.max(maxDiff, Math.abs(value - expected[index]));
  }, 0);
  const sampleMaxAbsDiff = actualSample.reduce((maxDiff, value, index) => {
    return Math.max(maxDiff, Math.abs(value - expectedSample[index]));
  }, 0);
  const expectedColorOutputArray = Array.from(expectedColorOutputSurface.data);
  const colorOutputSurfaceMaxAbsDiff = actualColorOutputSurface.reduce(
    (maxDiff, value, index) => {
      return Math.max(
        maxDiff,
        Math.abs(value - expectedColorOutputArray[index])
      );
    },
    0
  );
  const uniformReadMatchesPackedFrameConstants = maxAbsDiff <= epsilon;
  const sampleReadMatchesPackedSelectedSamples = sampleMaxAbsDiff <= epsilon;
  const sampleReadMatchesPackedNormalBackendSamples =
    sampleReadMatchesPackedSelectedSamples;
  const colorOutputSurfaceMatchesExpected =
    colorOutputSurfaceMaxAbsDiff <= epsilon;
  const {
    normalBackendOutputContract,
    presentationHandoffContract
  } = buildNormalBackendOutputContract({
    expectedSurfaceSummary: expectedColorOutputSurface.summary,
    expectedSurfaceData: expectedColorOutputSurface.data,
    colorOutputSurfaceReadback: actualColorOutputSurface,
    handoffReadback: actualNormalBackendOutputHandoff,
    sampleCount,
    sourceKind: 'step40-constrained-display-adapter',
    fallbackSamplesMixed: false,
    epsilon
  });
  const normalBackendOutputValidation = validateNormalBackendOutputContracts({
    normalBackendOutputContract,
    presentationHandoffContract
  });
  const normalBackendOutputMatchesExpected =
    normalBackendOutputContract.normalBackendOutputMatchesExpected === true;
  const handoffReadbackMatchesColorOutputSurface =
    normalBackendOutputContract.handoffReadbackMatchesColorOutputSurface === true;
  const guardedPresentationAdapterContract =
    await runWebGpuOnlyGuardedPresentationAdapter({
      device,
      handoffBuffer: normalBackendOutputHandoffBuffer,
      expectedSurfaceData: expectedColorOutputSurface.data,
      normalBackendOutputContract,
      presentationHandoffContract,
      viewerCanvasState
    });
  const guardedPresentationAdapterReady =
    guardedPresentationAdapterContract?.guardedPresentationAdapterReady === true;
  const presentationBridgeContract =
    guardedPresentationAdapterContract?.presentationBridgeContract ?? null;
  const viewerPresentationBridgeReady =
    presentationBridgeContract?.viewerPresentationBridgeReady === true;
  const cameraAwareVisibleOutputFinalContract =
    buildCameraAwareVisibleOutputContract({
      ...(cameraAwareVisibleOutputContract ?? {}),
      status:
        cameraAwareVisibleOutputContract?.sampleCount > 0 &&
        presentationBridgeContract?.currentTextureConnected === true
          ? 'ok'
          : 'blocked',
      currentTextureConnected:
        presentationBridgeContract?.currentTextureConnected === true,
      currentTextureRenderPassSubmitted:
        presentationBridgeContract?.currentTextureRenderPassSubmitted === true,
      currentTextureReadbackMatchesAdapterOutput:
        presentationBridgeContract?.currentTextureReadbackMatchesAdapterOutput === true,
      webgl2HybridRenderingAllowed:
        presentationBridgeContract?.webgl2HybridRenderingAllowed === true,
      fallbackSamplesMixed:
        cameraAwareVisibleOutputContract?.fallbackSamplesMixed === true,
      reason:
        cameraAwareVisibleOutputContract?.sampleCount > 0
          ? presentationBridgeContract?.currentTextureConnected === true
            ? null
            : 'camera-aware-visible-output-currentTexture-not-connected'
          : cameraAwareVisibleOutputContract?.reason ??
            'camera-aware-visible-output-samples-empty'
    });
  if (typeof outputBuffer.destroy === 'function') {
    outputBuffer.destroy();
  }
  if (typeof readbackBuffer.destroy === 'function') {
    readbackBuffer.destroy();
  }
  if (typeof colorOutputSurfaceBuffer.destroy === 'function') {
    colorOutputSurfaceBuffer.destroy();
  }
  if (typeof colorOutputSurfaceReadbackBuffer.destroy === 'function') {
    colorOutputSurfaceReadbackBuffer.destroy();
  }
  if (typeof normalBackendOutputHandoffBuffer.destroy === 'function') {
    normalBackendOutputHandoffBuffer.destroy();
  }
  if (typeof normalBackendOutputHandoffReadbackBuffer.destroy === 'function') {
    normalBackendOutputHandoffReadbackBuffer.destroy();
  }
  return {
    contractVersion:
      WEBGPU_NORMAL_BACKEND_UNIFORM_SHADER_CONSUMPTION_CONTRACT_VERSION,
    consumptionMode:
      'normal-webgpu-backend-minimal-uniform-and-sample-compute-consumption',
    status:
      uniformReadMatchesPackedFrameConstants &&
      sampleReadMatchesPackedSelectedSamples &&
      colorOutputSurfaceMatchesExpected &&
      normalBackendOutputMatchesExpected &&
      handoffReadbackMatchesColorOutputSurface &&
      guardedPresentationAdapterReady &&
      viewerPresentationBridgeReady
        ? 'ok'
        : 'mismatch',
    bindGroupLayoutCreated: true,
    bindGroupCreated: true,
    computePipelineCreated: true,
    uniformReadDispatchSubmitted: true,
    uniformReadbackCompleted: true,
    sampleStorageBufferReadbackCompleted: true,
    colorOutputSurfaceWritten: true,
    colorOutputSurfaceReadbackCompleted: true,
    normalBackendOutputHandoffCopySubmitted: true,
    normalBackendOutputReadbackCompleted: true,
    guardedPresentationAdapterCalled: true,
    guardedPresentationAdapterReady,
    viewerPresentationBridgeReady,
    renderTargetBridgeReady:
      presentationBridgeContract?.renderTargetBridgeReady === true,
    currentTextureConnectionAttempted:
      presentationBridgeContract?.currentTextureConnectionAttempted === true,
    currentTextureConnected:
      presentationBridgeContract?.currentTextureConnected === true,
    submittedWorkDone,
    uniformReadMatchesPackedFrameConstants,
    sampleReadMatchesPackedSelectedSamples,
    sampleReadMatchesPackedNormalBackendSamples,
    colorOutputSurfaceMatchesExpected,
    normalBackendOutputMatchesExpected,
    handoffReadbackMatchesColorOutputSurface,
    normalBackendOutputValidation,
    cameraAwareVisibleOutputContract: cameraAwareVisibleOutputFinalContract,
    guardedPresentationAdapterContract,
    presentationBridgeContract,
    expectedFrameUniformPrefix: expected,
    readbackFrameUniformPrefix: actual,
    expectedFirstSamplePackedFields: expectedSample,
    readbackFirstSamplePackedFields: actualSample,
    expectedColorOutputSurfaceFirstPixels: expectedColorOutputArray.slice(0, 16),
    readbackColorOutputSurfaceFirstPixels: actualColorOutputSurface.slice(0, 16),
    maxAbsDiff,
    sampleMaxAbsDiff,
    colorOutputSurfaceMaxAbsDiff,
    normalBackendOutputMaxAbsDiff:
      normalBackendOutputContract.normalBackendOutputMaxAbsDiff ?? null,
    epsilon,
    shaderEntryPoint: 'main',
    workgroupCount: 1,
    outputByteLength,
    validationField:
      'frameIndex_time_viewportWidth_viewportHeight_first_selected_sample_and_color_output_surface',
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
      },
      {
        binding: 3,
        visibility: 'compute',
        bufferType: 'storage',
        minBindingSize: expectedColorOutputSurface.data.byteLength
      }
    ],
    sampleStorageBufferConsumed: true,
    colorOutputSurfaceContract: {
      ...expectedColorOutputSurface.summary,
      visibleOutputMode:
        cameraAwareVisibleOutputFinalContract.outputMode ?? null,
      cameraAwareVisibleOutputReady:
        cameraAwareVisibleOutputFinalContract.cameraAwareVisibleOutputReady === true,
      colorOutputSurfaceCreated: true,
      colorOutputSurfaceWriteSubmitted: true,
      colorOutputSurfaceReadbackCompleted: true,
      colorOutputSurfaceDestroyed: true,
      colorOutputSurfaceMatchesExpected,
      colorOutputSurfaceMaxAbsDiff,
      expectedFirstPixels: expectedColorOutputArray.slice(0, 16),
      readbackFirstPixels: actualColorOutputSurface.slice(0, 16)
    },
    normalBackendOutputContract,
    presentationHandoffContract,
    guardedPresentationAdapterContract,
    presentationBridgeContract,
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
  visibleOutputSamples = [],
  cameraAwareVisibleOutputContract = null,
  device = null,
  viewerCanvasState = null
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
  const outputSamples =
    Array.isArray(visibleOutputSamples) && visibleOutputSamples.length > 0
      ? visibleOutputSamples
      : selectedSamples;
  const packedSamples = packNormalBackendSelectedSampleData(outputSamples);
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
      label: 'phase3-step68-normal-backend-frame-constants-uniform-buffer',
      size: Math.max(paddedByteLength, uniformData.byteLength),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    lifecycleEvents.push('uniform-buffer-created');
    effectiveDevice.queue.writeBuffer(uniformBuffer, 0, uniformData);
    lifecycleEvents.push('queue-write-buffer');
    const sampleBuffer = effectiveDevice.createBuffer({
      label: 'phase3-step68-normal-backend-selected-sample-storage-buffer',
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
        minBindingSizeBytes: paddedByteLength,
        viewerCanvasState,
        cameraAwareVisibleOutputContract
      });
    lifecycleEvents.push('bind-group-layout-created');
    lifecycleEvents.push('bind-group-created');
    lifecycleEvents.push('color-output-surface-buffer-created');
    lifecycleEvents.push('minimal-uniform-sample-color-output-compute-dispatched');
    if (uniformShaderConsumptionContract?.guardedPresentationAdapterCalled === true) {
      lifecycleEvents.push('guarded-presentation-adapter-called');
    }
    if (uniformShaderConsumptionContract?.guardedPresentationAdapterReady === true) {
      lifecycleEvents.push('guarded-presentation-adapter-consumed-handoff-output');
    }
    if (uniformShaderConsumptionContract?.viewerPresentationBridgeReady === true) {
      lifecycleEvents.push('viewer-presentation-render-target-bridge-ready');
    }
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
    lifecycleEvents.push('color-output-surface-buffer-destroyed');
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
      sampleReadMatchesPackedNormalBackendSamples:
        uniformShaderConsumptionContract
          ?.sampleReadMatchesPackedNormalBackendSamples === true,
      colorOutputSurfaceCreated:
        uniformShaderConsumptionContract?.colorOutputSurfaceWritten === true,
      colorOutputSurfaceWriteSubmitted:
        uniformShaderConsumptionContract?.colorOutputSurfaceWritten === true,
      colorOutputSurfaceReadbackCompleted:
        uniformShaderConsumptionContract?.colorOutputSurfaceReadbackCompleted === true,
      colorOutputSurfaceDestroyed: true,
      colorOutputSurfaceMatchesExpected:
        uniformShaderConsumptionContract?.colorOutputSurfaceMatchesExpected === true,
      normalBackendOutputResourceCreated:
        uniformShaderConsumptionContract?.normalBackendOutputHandoffCopySubmitted === true,
      normalBackendOutputHandoffCopySubmitted:
        uniformShaderConsumptionContract?.normalBackendOutputHandoffCopySubmitted === true,
      normalBackendOutputReadbackCompleted:
        uniformShaderConsumptionContract?.normalBackendOutputReadbackCompleted === true,
      normalBackendOutputResourceDestroyed: true,
      normalBackendOutputMatchesExpected:
        uniformShaderConsumptionContract?.normalBackendOutputMatchesExpected === true,
      handoffReadbackMatchesColorOutputSurface:
        uniformShaderConsumptionContract?.handoffReadbackMatchesColorOutputSurface === true,
      guardedPresentationAdapterCalled:
        uniformShaderConsumptionContract?.guardedPresentationAdapterCalled === true,
      guardedPresentationAdapterReady:
        uniformShaderConsumptionContract?.guardedPresentationAdapterReady === true,
      viewerPresentationBridgeReady:
        uniformShaderConsumptionContract?.viewerPresentationBridgeReady === true,
      renderTargetBridgeReady:
        uniformShaderConsumptionContract?.renderTargetBridgeReady === true,
      currentTextureConnectionAttempted:
        uniformShaderConsumptionContract?.currentTextureConnectionAttempted === true,
      currentTextureConnected:
        uniformShaderConsumptionContract?.currentTextureConnected === true,
      normalBackendOutputValidation:
        uniformShaderConsumptionContract?.normalBackendOutputValidation ?? null,
      uniformShaderConsumptionContract,
      sampleResourceLifecycleContract: {
        ...packedSamples.summary,
        outputSampleSourceMode:
          Array.isArray(visibleOutputSamples) && visibleOutputSamples.length > 0
            ? 'camera-aware-visible-output-samples'
            : 'selected-present-samples',
        normalBackendOwnsGpuSampleResource: true,
        sampleBufferCreated: true,
        sampleBufferWriteSubmitted: true,
        sampleBufferWriteCompleted: submittedWorkDone,
        sampleBufferDestroyed: true,
        sampleStorageBufferConsumed:
          uniformShaderConsumptionContract?.sampleStorageBufferConsumed === true,
        sampleReadMatchesPackedSelectedSamples:
          uniformShaderConsumptionContract?.sampleReadMatchesPackedSelectedSamples === true,
        sampleReadMatchesPackedNormalBackendSamples:
          uniformShaderConsumptionContract
            ?.sampleReadMatchesPackedNormalBackendSamples === true
      },
      colorOutputSurfaceLifecycleContract:
        uniformShaderConsumptionContract?.colorOutputSurfaceContract ?? null,
      normalBackendOutputContract:
        uniformShaderConsumptionContract?.normalBackendOutputContract ?? null,
      presentationHandoffContract:
        uniformShaderConsumptionContract?.presentationHandoffContract ?? null,
      guardedPresentationAdapterContract:
        uniformShaderConsumptionContract?.guardedPresentationAdapterContract ?? null,
      presentationBridgeContract:
        uniformShaderConsumptionContract?.presentationBridgeContract ?? null,
      cameraAwareVisibleOutputContract:
        uniformShaderConsumptionContract?.cameraAwareVisibleOutputContract ?? null,
      packedFloat32Count: uniformData.length,
      packedByteLength: uniformData.byteLength,
      paddedUniformByteLength:
        uniformResourcePreparationContract?.layout?.paddedUniformByteLength ??
        null,
      minBindingSizeBytes:
        uniformResourcePreparationContract?.layout?.minBindingSizeBytes ?? null,
      samplePackedFloat32Count: packedSamples.data.length,
      samplePackedByteLength: packedSamples.data.byteLength,
      colorOutputSurfacePackedByteLength:
        uniformShaderConsumptionContract?.colorOutputSurfaceContract
          ?.packedByteLength ?? null,
      normalBackendOutputPackedByteLength:
        uniformShaderConsumptionContract?.normalBackendOutputContract
          ?.packedByteLength ?? null,
      usage: ['UNIFORM', 'STORAGE', 'COPY_DST', 'COPY_SRC'],
      layoutMode:
        uniformResourcePreparationContract?.layout?.layoutMode ?? null,
      reusePolicy:
        'future runner-owned device/resource cache; Step72 uses per-call transient uniform, sample, color output, handoff, guarded presentation adapter, render-target bridge, and currentTexture handoff resources',
      disposePolicy:
        'destroy transient uniform, sample, color output, handoff, offscreen presentation target, render-target bridge, and readback resources after validation; currentTexture lifecycle remains canvas-owned',
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
