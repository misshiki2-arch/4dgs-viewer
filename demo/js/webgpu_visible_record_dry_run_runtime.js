import { buildVisibleItemForCandidate } from './gpu_visible_item_builder.js';
import { clampInt } from './gpu_tile_utils.js';

const DEFAULT_MAX_RECORDS = 65536;
const DEFAULT_EPSILON = 1e-3;
const RECORD_FLOATS = 12;
const IMPLEMENTED_FIELDS = ['srcIndex', 'valid', 'px', 'py', 'depth', 'aabb'];
const DEFERRED_FIELDS = [
  'radius',
  'conic',
  'alpha',
  'tileRange',
  'colorAlpha.rgb',
  'SH',
  '4D conditional covariance full parity',
  'compaction',
  'depth sort',
  'tile-list GPU generation',
  'display connection'
];

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

function toFiniteInteger(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(0, Math.floor(n)) : fallback;
}

function makeFallback(reason, extra = {}) {
  return {
    schemaVersion: 'phase3-step2-webgpu-visible-record-dry-run-v1',
    status: 'fallback',
    reason,
    computeMode: 'webgpu-storage-buffer-compute-fixed-record',
    implementedFields: IMPLEMENTED_FIELDS,
    deferredFields: DEFERRED_FIELDS,
    candidateCount: extra.candidateCount ?? null,
    recordCount: extra.recordCount ?? null,
    validRecordCount: extra.validRecordCount ?? null,
    recordComparison: {
      anyMismatch: true,
      fieldMismatchCount: null,
      maxAbsError: null,
      firstMismatches: extra.firstMismatches ?? []
    },
    fieldMismatchCount: null,
    firstMismatches: extra.firstMismatches ?? [],
    mismatchClassification: extra.mismatchClassification ?? 'webgpu-visible-record-unavailable',
    timing: extra.timing ?? null,
    webgpu: extra.webgpu ?? null,
    metadata: extra.metadata ?? null
  };
}

function writeRecord(out, row, item, srcIndex) {
  const base = row * RECORD_FLOATS;
  out[base + 0] = srcIndex;
  out[base + 1] = item ? 1 : 0;
  out[base + 2] = item?.px ?? 0;
  out[base + 3] = item?.py ?? 0;
  out[base + 4] = item?.depth ?? 0;
  out[base + 5] = item?.aabb?.[0] ?? 0;
  out[base + 6] = item?.aabb?.[1] ?? 0;
  out[base + 7] = item?.aabb?.[2] ?? 0;
  out[base + 8] = item?.aabb?.[3] ?? 0;
  out[base + 9] = 0;
  out[base + 10] = 0;
  out[base + 11] = 0;
}

function buildCpuReferenceRecords({
  candidateInfo,
  raw,
  camera,
  screenSpaceCamera,
  canvasWidth,
  canvasHeight,
  camPos,
  tileGrid,
  buildConfig,
  maxRecords
}) {
  const startMs = nowMs();
  const candidateIndices = toUint32Array(candidateInfo?.candidateIndices);
  const count = Math.min(candidateIndices.length, toFiniteInteger(maxRecords, DEFAULT_MAX_RECORDS));
  const renderScale = Number.isFinite(buildConfig.renderScale) ? buildConfig.renderScale : 1;
  const renderW = Math.max(1, Math.round(canvasWidth * renderScale));
  const renderH = Math.max(1, Math.round(canvasHeight * renderScale));
  const sx = canvasWidth / renderW;
  const sy = canvasHeight / renderH;
  const flags = {
    nativeRot4d: !!buildConfig.useNativeRot4d,
    nativeMarginal: !!buildConfig.useNativeMarginal
  };
  const records = new Float32Array(count * RECORD_FLOATS);
  let validCount = 0;
  for (let i = 0; i < count; i += 1) {
    const srcIndex = candidateIndices[i];
    const item = buildVisibleItemForCandidate({
      raw,
      index: srcIndex,
      camera,
      screenSpaceCamera,
      renderW,
      renderH,
      canvasWidth,
      canvasHeight,
      sx,
      sy,
      timestamp: buildConfig.timestamp,
      scalingModifier: buildConfig.scalingModifier,
      sigmaScale: buildConfig.sigmaScale,
      prefilterVar: buildConfig.prefilterVar,
      useRot4d: buildConfig.useRot4d,
      flags,
      camPos,
      timeDuration: buildConfig.timeDuration,
      useSH: buildConfig.useSH,
      forceSh3d: buildConfig.forceSh3d,
      tileGrid
    });
    if (item) validCount += 1;
    if (item) {
      const recordPx = Math.fround(item.px);
      const recordPy = Math.fround(item.py);
      const recordCoverageRadius = Math.max(1.0, Math.fround(item.radius));
      writeRecord(records, i, {
        ...item,
        px: recordPx,
        py: recordPy,
        depth: Math.fround(item.depth),
        aabb: [
          clampInt(Math.floor(recordPx - recordCoverageRadius), 0, canvasWidth - 1),
          clampInt(Math.floor(recordPy - recordCoverageRadius), 0, canvasHeight - 1),
          clampInt(Math.ceil(recordPx + recordCoverageRadius), 0, canvasWidth - 1),
          clampInt(Math.ceil(recordPy + recordCoverageRadius), 0, canvasHeight - 1)
        ]
      }, srcIndex);
    } else {
      writeRecord(records, i, null, srcIndex);
    }
  }
  return {
    candidateIndices: candidateIndices.slice(0, count),
    candidateCount: candidateIndices.length,
    count,
    validCount,
    records,
    timing: {
      cpuReferenceBuildMs: nowMs() - startMs
    }
  };
}

function buildRawXyzOpacityForCandidates(raw, candidateIndices) {
  const count = candidateIndices.length;
  const out = new Float32Array(count * 4);
  for (let i = 0; i < count; i += 1) {
    const srcIndex = candidateIndices[i];
    const o = i * 4;
    out[o + 0] = raw.xyz?.[srcIndex * raw.xyzDim + 0] ?? 0;
    out[o + 1] = raw.xyz?.[srcIndex * raw.xyzDim + 1] ?? 0;
    out[o + 2] = raw.xyz?.[srcIndex * raw.xyzDim + 2] ?? 0;
    out[o + 3] = raw.opacity?.[srcIndex * raw.opacityDim + 0] ?? 0;
  }
  return out;
}

function compareRecords(reference, candidate, count, { epsilon, maxMismatches }) {
  const firstMismatches = [];
  let fieldMismatchCount = 0;
  let maxAbsError = 0;
  const fields = [
    ['srcIndex', 0, 1],
    ['valid', 1, 1],
    ['px', 2, 1],
    ['py', 3, 1],
    ['depth', 4, 1],
    ['aabb', 5, 4]
  ];
  for (let row = 0; row < count; row += 1) {
    const base = row * RECORD_FLOATS;
    for (const [field, offset, components] of fields) {
      for (let c = 0; c < components; c += 1) {
        const ref = reference[base + offset + c];
        const got = candidate[base + offset + c];
        const diff = Math.abs(ref - got);
        maxAbsError = Math.max(maxAbsError, diff);
        if (diff > epsilon) {
          fieldMismatchCount += 1;
          if (firstMismatches.length < maxMismatches) {
            firstMismatches.push({
              row,
              field,
              component: components > 1 ? c : null,
              expected: ref,
              actual: got,
              absError: diff
            });
          }
        }
      }
    }
  }
  return {
    anyMismatch: fieldMismatchCount > 0,
    fieldMismatchCount,
    maxAbsError,
    firstMismatches
  };
}

function classifyComparison(comparison) {
  if (!comparison) return 'webgpu-visible-record-compare-missing';
  if (!comparison.anyMismatch) return 'none';
  const fields = new Set((comparison.firstMismatches ?? []).map((item) => item.field));
  if (fields.size === 1 && fields.has('aabb')) return 'webgpu-fixed-record-aabb-mismatch';
  return 'webgpu-fixed-record-field-mismatch';
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

async function runCompute({ device, cpuReference, rawXyzOpacity }) {
  const shader = device.createShaderModule({
    label: 'phase3-step2-visible-record-copy-scaffold',
    code: `
struct Params {
  count: u32,
  recordFloats: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> candidates: array<u32>;
@group(0) @binding(1) var<storage, read> rawXyzOpacity: array<vec4f>;
@group(0) @binding(2) var<storage, read> referenceRecords: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> outputRecords: array<vec4f>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let row = id.x;
  if (row >= params.count) {
    return;
  }
  let base = row * 3u;
  let srcIndex = candidates[row];
  let raw0 = rawXyzOpacity[row];
  var r0 = referenceRecords[base + 0u];
  let r1 = referenceRecords[base + 1u];
  let r2 = referenceRecords[base + 2u];
  r0.x = f32(srcIndex);
  r0.y = r0.y + raw0.x * 0.0;
  outputRecords[base + 0u] = r0;
  outputRecords[base + 1u] = r1;
  outputRecords[base + 2u] = r2;
}`
  });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shader,
      entryPoint: 'main'
    }
  });
  const outputByteLength = cpuReference.records.byteLength;
  const candidateBuffer = createBuffer(device, cpuReference.candidateIndices, GPUBufferUsage.STORAGE);
  const rawBuffer = createBuffer(device, rawXyzOpacity, GPUBufferUsage.STORAGE);
  const referenceBuffer = createBuffer(device, cpuReference.records, GPUBufferUsage.STORAGE);
  const outputBuffer = device.createBuffer({
    size: outputByteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const paramsBuffer = createBuffer(
    device,
    new Uint32Array([cpuReference.count, RECORD_FLOATS, 0, 0]),
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
      { binding: 1, resource: { buffer: rawBuffer } },
      { binding: 2, resource: { buffer: referenceBuffer } },
      { binding: 3, resource: { buffer: outputBuffer } },
      { binding: 4, resource: { buffer: paramsBuffer } }
    ]
  });
  const dispatchStartMs = nowMs();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(cpuReference.count / 64));
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputByteLength);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const computeDispatchMs = nowMs() - dispatchStartMs;
  const readbackStartMs = nowMs();
  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(readbackBuffer.getMappedRange().slice(0));
  readbackBuffer.unmap();
  return {
    records: out,
    timing: {
      computeDispatchMs,
      readbackMs: nowMs() - readbackStartMs
    }
  };
}

export async function runWebGpuVisibleRecordDryRun({
  candidateInfo,
  raw,
  camera,
  screenSpaceCamera = null,
  canvasWidth = 0,
  canvasHeight = 0,
  camPos = null,
  tileGrid = null,
  buildConfig = {},
  maxRecords = DEFAULT_MAX_RECORDS,
  epsilon = DEFAULT_EPSILON,
  maxMismatches = 32,
  metadata = null
} = {}) {
  const totalStartMs = nowMs();
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    return makeFallback('webgpu-unavailable', { metadata });
  }
  if (!raw || !candidateInfo || !camera || !buildConfig) {
    return makeFallback('webgpu-visible-record-input-unavailable', { metadata });
  }
  const adapterStartMs = nowMs();
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return makeFallback('webgpu-adapter-unavailable', { metadata });
  const device = await adapter.requestDevice();
  const adapterDeviceMs = nowMs() - adapterStartMs;
  const cpuReference = buildCpuReferenceRecords({
    candidateInfo,
    raw,
    camera,
    screenSpaceCamera,
    canvasWidth,
    canvasHeight,
    camPos,
    tileGrid,
    buildConfig,
    maxRecords
  });
  if (cpuReference.candidateCount <= 0 || cpuReference.count <= 0) {
    return makeFallback('webgpu-visible-record-candidate-input-empty', {
      candidateCount: cpuReference.candidateCount,
      recordCount: cpuReference.count,
      validRecordCount: cpuReference.validCount,
      metadata
    });
  }
  const uploadStartMs = nowMs();
  const rawXyzOpacity = buildRawXyzOpacityForCandidates(raw, cpuReference.candidateIndices);
  const bufferUploadPrepareMs = nowMs() - uploadStartMs;
  const computeResult = await runCompute({ device, cpuReference, rawXyzOpacity });
  const compareStartMs = nowMs();
  const recordComparison = compareRecords(cpuReference.records, computeResult.records, cpuReference.count, {
    epsilon,
    maxMismatches
  });
  const compareMs = nowMs() - compareStartMs;
  const mismatchClassification = classifyComparison(recordComparison);
  return {
    schemaVersion: 'phase3-step2-webgpu-visible-record-dry-run-v1',
    status: 'ok',
    reason: 'ok',
    computeMode: 'webgpu-storage-buffer-compute-fixed-record',
    scaffoldMode: 'storage-buffer-fixed-record-copy-with-raw-buffer-bind',
    scaffoldNote: 'Phase 3 Step2 validates WebGPU storage buffer, compute dispatch, readback, and fixed-record comparison plumbing. Full raw 4DGS math in WGSL is deferred.',
    implementedFields: IMPLEMENTED_FIELDS,
    deferredFields: DEFERRED_FIELDS,
    candidateCount: cpuReference.candidateCount,
    recordCount: cpuReference.count,
    validRecordCount: cpuReference.validCount,
    recordFloats: RECORD_FLOATS,
    recordLayout: [
      ['srcIndex', 0, 1],
      ['valid', 1, 1],
      ['px', 2, 1],
      ['py', 3, 1],
      ['depth', 4, 1],
      ['aabb', 5, 4],
      ['reserved', 9, 3]
    ],
    recordComparison,
    fieldMismatchCount: recordComparison.fieldMismatchCount,
    firstMismatches: recordComparison.firstMismatches,
    mismatchClassification,
    anyMismatch: !!recordComparison.anyMismatch,
    timing: {
      adapterDeviceMs,
      bufferUploadMs: bufferUploadPrepareMs,
      ...cpuReference.timing,
      ...computeResult.timing,
      compareMs,
      totalMs: nowMs() - totalStartMs
    },
    webgpu: {
      adapterInfoAvailable: typeof adapter.requestAdapterInfo === 'function',
      rawBufferUploadMode: 'candidate-xyz-opacity-scaffold',
      candidateBufferCount: cpuReference.count,
      outputBufferBytes: cpuReference.records.byteLength
    },
    metadata
  };
}
