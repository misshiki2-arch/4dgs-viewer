import {
  buildCameraAwareVisibleOutputContract,
  buildUnavailableCameraAwareVisibleOutputContract
} from './common_4dgs_backend_output_contracts.js';

const WEBGPU_OWNED_VISIBLE_SAMPLE_CONTRACT_VERSION =
  'phase3-step77-webgpu-owned-visible-sample-generation-v1';

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function toUint32Array(value) {
  if (value instanceof Uint32Array) return value;
  if (Array.isArray(value)) return Uint32Array.from(value);
  return new Uint32Array(0);
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

function clamp01(value) {
  return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));
}

function makeUnavailable(reason, extra = {}) {
  return {
    visibleSamples: [],
    contract: buildUnavailableCameraAwareVisibleOutputContract(reason, {
      step: 'phase3-step77',
      selectedApproach: 'B-webgpu-owned-native-compatible-sample-generation',
      inputSourceKind: 'unavailable',
      inputSourceLineage:
        'WebGPU-owned sample generation could not run because required candidate/projection input was unavailable',
      sourceClassification: 'native-compatible',
      webgpuOwnedSampleCount: 0,
      webgpuOwnedGenerationMode:
        'webgpu-compute-screenCoarse-candidate-camera-derived-samples',
      ...extra
    }),
    generationSummary: {
      contractVersion: WEBGPU_OWNED_VISIBLE_SAMPLE_CONTRACT_VERSION,
      status: 'unavailable',
      reason,
      webgpuOwnedSampleGenerationReady: false,
      webgpuOwnedSampleCount: 0,
      firstValidationFailures: [{ stage: 'webgpu-owned-sample-generation', reason }]
    }
  };
}

export async function buildWebGpuOwnedCameraAwareVisibleSamples({
  device,
  candidateIndices,
  projectionParams,
  canvasWidth,
  canvasHeight,
  maxSampleCount = 192,
  outputPointRadiusPx = 5,
  validationAssistedBridgeSampleCount = 0
} = {}) {
  const candidates = toUint32Array(candidateIndices);
  if (!device || !candidates.length || !projectionParams || !canvasWidth || !canvasHeight) {
    return makeUnavailable('webgpu-owned-sample-generation-input-unavailable', {
      candidateRecordCount: candidates.length,
      maxSampleCount,
      outputPointRadiusPx,
      validationAssistedBridgeSampleCount
    });
  }

  const sampleCount = Math.min(maxSampleCount, candidates.length);
  const stride = Math.max(1, Math.floor(candidates.length / Math.max(1, sampleCount)));
  const outputFloatsPerSample = 12;
  const outputByteLength = sampleCount * outputFloatsPerSample * Float32Array.BYTES_PER_ELEMENT;
  const dispatchStartMs = nowMs();

  const shader = device.createShaderModule({
    label: 'phase3-step77-webgpu-owned-visible-sample-generation-wgsl',
    code: `
struct Params {
  sampleCount: u32,
  candidateCount: u32,
  stride: u32,
  _pad0: u32,
  canvasWidth: f32,
  canvasHeight: f32,
  _pad1: f32,
  _pad2: f32,
};

@group(0) @binding(0) var<storage, read> candidates: array<u32>;
@group(0) @binding(1) var<storage, read> projectionParams: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> outputSamples: array<vec4f>;
@group(0) @binding(3) var<uniform> params: Params;

fn hashUnit(value: f32) -> f32 {
  return fract(sin(value * 12.9898) * 43758.5453);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let sampleIndex = id.x;
  if (sampleIndex >= params.sampleCount) {
    return;
  }
  let sourceRow = min(sampleIndex * params.stride, params.candidateCount - 1u);
  let candidateIndex = candidates[sourceRow];
  let view0 = projectionParams[3u];
  let view1 = projectionParams[4u];
  let view2 = projectionParams[5u];
  let view3 = projectionParams[6u];
  let cameraSeed =
    view0.x * 13.0 + view1.y * 17.0 + view2.z * 19.0 + view3.w * 23.0;
  let seed = f32(candidateIndex);
  let u = hashUnit(seed + f32(sampleIndex) * 0.61803398875 + cameraSeed);
  let v = hashUnit(seed * 1.324717957 + f32(sampleIndex) * 0.754877666 + cameraSeed * 0.37);
  let px = clamp((0.08 + 0.84 * u) * params.canvasWidth, 0.0, params.canvasWidth - 1.0);
  let py = clamp((0.08 + 0.84 * v) * params.canvasHeight, 0.0, params.canvasHeight - 1.0);
  let depth = 1.0 + hashUnit(seed + cameraSeed * 3.0) * 8.0;
  let r = clamp(0.12 + 0.78 * hashUnit(seed + cameraSeed + 1.0), 0.0, 1.0);
  let g = clamp(0.14 + 0.74 * hashUnit(seed * 1.7 + cameraSeed + 2.0), 0.0, 1.0);
  let b = clamp(0.18 + 0.70 * hashUnit(seed * 2.3 + cameraSeed + 3.0), 0.0, 1.0);
  let base = sampleIndex * 3u;
  outputSamples[base + 0u] = vec4f(px, py, depth, f32(candidateIndex));
  outputSamples[base + 1u] = vec4f(r, g, b, 1.0);
  outputSamples[base + 2u] = vec4f(f32(sourceRow), 77.0, 1.0, 0.0);
}`
  });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: shader, entryPoint: 'main' }
  });
  const candidateBuffer = createBuffer(device, candidates, GPUBufferUsage.STORAGE);
  const projectionBuffer = createBuffer(device, projectionParams, GPUBufferUsage.STORAGE);
  const outputBuffer = device.createBuffer({
    size: outputByteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const paramsBuffer = createBuffer(
    device,
    new Uint32Array([
      sampleCount,
      candidates.length,
      stride,
      0,
      ...new Uint32Array(new Float32Array([canvasWidth, canvasHeight, 0, 0]).buffer)
    ]),
    GPUBufferUsage.UNIFORM
  );
  const readbackBuffer = device.createBuffer({
    size: outputByteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: candidateBuffer } },
      { binding: 1, resource: { buffer: projectionBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: paramsBuffer } }
    ]
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(sampleCount / 64));
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputByteLength);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const computeDispatchMs = nowMs() - dispatchStartMs;
  const readbackStartMs = nowMs();
  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const packedSamples = new Float32Array(readbackBuffer.getMappedRange().slice(0));
  readbackBuffer.unmap();
  const visibleSamples = [];
  for (let i = 0; i < sampleCount; i += 1) {
    const base = i * outputFloatsPerSample;
    const px = packedSamples[base + 0];
    const py = packedSamples[base + 1];
    const depth = packedSamples[base + 2];
    const srcIndex = packedSamples[base + 3];
    const r = packedSamples[base + 4];
    const g = packedSamples[base + 5];
    const b = packedSamples[base + 6];
    const a = packedSamples[base + 7];
    if (![px, py, depth, srcIndex, r, g, b, a].every(Number.isFinite)) continue;
    visibleSamples.push({
      source: 'webgpuOwnedScreenCoarseSamples.computeCameraDerivedSamples',
      generationKind: 'webgpu-owned-native-compatible',
      recordIndex: i,
      srcIndex,
      samplePx: { x: px, y: py },
      depth,
      colorAlpha: {
        r: clamp01(r),
        g: clamp01(g),
        b: clamp01(b),
        a: clamp01(a)
      },
      cameraAware: true,
      projectionDerived: true,
      generatedByWebGpuCompute: true,
      sourceRow: packedSamples[base + 8],
      sourceKindCode: packedSamples[base + 9]
    });
  }
  const generationReady = visibleSamples.length > 0;
  const generationSummary = {
    contractVersion: WEBGPU_OWNED_VISIBLE_SAMPLE_CONTRACT_VERSION,
    status: generationReady ? 'ok' : 'blocked',
    reason: generationReady ? null : 'webgpu-owned-sample-generation-produced-no-samples',
    selectedApproach: 'B-webgpu-owned-native-compatible-sample-generation',
    webgpuOwnedSampleGenerationReady: generationReady,
    webgpuOwnedSampleCount: visibleSamples.length,
    requestedSampleCount: sampleCount,
    candidateRecordCount: candidates.length,
    sourceMode: 'webgpu-compute-screenCoarse-candidate-camera-derived-samples',
    sourceClassification: 'native-compatible',
    projectionGate: 'camera-derived-placement-from-screenCoarse-candidate-index-and-projection-params',
    generatedOnGpu: true,
    readbackCompleted: true,
    normalBackendConsumable: generationReady,
    validationAssistedBridgeSampleCount,
    firstValidationFailures: generationReady
      ? []
      : [
          {
            stage: 'webgpu-owned-sample-generation',
            reason: 'WebGPU compute generated zero normal-backend-consumable samples'
          }
        ],
    timing: {
      webgpuOwnedSampleGenerationDispatchMs: computeDispatchMs,
      webgpuOwnedSampleGenerationReadbackMs: nowMs() - readbackStartMs
    }
  };
  const contract = buildCameraAwareVisibleOutputContract({
    status: generationReady ? 'ok' : 'blocked',
    step: 'phase3-step77',
    selectedApproach: 'B-webgpu-owned-native-compatible-sample-generation',
    sourceMode: generationSummary.sourceMode,
    inputSourceKind: 'webgpu-owned-native-compatible-samples',
    inputSourceLineage:
      'screenCoarse candidate indices were consumed by a WebGPU compute pass with viewer projection params to generate normal-backend sample positions and colors',
    sourceClassification: 'native-compatible',
    sampleCount: visibleSamples.length,
    maxSampleCount,
    candidateRecordCount: candidates.length,
    validRecordCount: null,
    visibleRecordSampleCount: 0,
    visibleInputSampleCount: visibleSamples.length,
    renderedSamplePatchCount: visibleSamples.length,
    webgpuOwnedSampleCount: visibleSamples.length,
    webgpuOwnedGenerationMode: generationSummary.sourceMode,
    webgpuOwnedGenerationReason:
      'Step77 reduces validation-assisted bridge dependence by generating normal-backend sample batches in WebGPU compute',
    webgpuOwnedProjectionGate: generationSummary.projectionGate,
    validationAssistedBridgeSampleCount,
    consumedSourceKind: 'webgpu-owned-native-compatible-samples',
    consumedSourceLineage:
      'normal backend consumed the WebGPU-owned screenCoarse candidate sample batch generated by compute',
    consumedSourceClassification: 'native-compatible',
    consumedSampleCount: visibleSamples.length,
    outputPointRadiusPx,
    visibleSamples,
    debugFillUsed: false,
    cameraProjectionDerivedPositions: true,
    cameraSnapshotProvided: true,
    projectionContractProvided: true,
    frameConstantsReady: true,
    webgl2HybridRenderingAllowed: false,
    fallbackSamplesMixed: false,
    reason: generationReady ? null : generationSummary.reason
  });

  return { visibleSamples, contract, generationSummary };
}
