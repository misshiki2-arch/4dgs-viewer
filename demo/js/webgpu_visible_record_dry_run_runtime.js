import { buildVisibleItemForCandidate } from './gpu_visible_item_builder.js';
import { clampInt } from './gpu_tile_utils.js';
import { computeGaussianState } from './rot4d_math.js';
import {
  COMPARISON_CONTRACT_SCHEMA_VERSION,
  DEFAULT_COMPARISON_EPSILON,
  DEFAULT_MAX_MISMATCHES,
  MISMATCH_CLASSIFICATIONS,
  RECORD_COMPARISON_KEYS,
  createComparisonToleranceMetadata,
  createRecordComparisonResult,
  createRecordMismatch
} from './common_4dgs_comparison_contracts.js';
import { buildWebGpuProjectionContract } from './common_4dgs_projection_contracts.js';
import {
  WEBGPU_VISIBLE_RECORD_COMPUTE_MODE,
  WEBGPU_VISIBLE_RECORD_CPU_MATERIALIZED_FIELDS,
  WEBGPU_VISIBLE_RECORD_DEFERRED_FIELDS,
  WEBGPU_VISIBLE_RECORD_FIELDS,
  WEBGPU_VISIBLE_RECORD_FLOATS,
  WEBGPU_VISIBLE_RECORD_IMPLEMENTED_FIELDS,
  WEBGPU_VISIBLE_RECORD_PHASE_STEP,
  WEBGPU_VISIBLE_RECORD_REFERENCE_ASSISTED_FIELDS,
  WEBGPU_VISIBLE_RECORD_SCAFFOLD_MODE,
  WEBGPU_VISIBLE_RECORD_DRY_RUN_SCHEMA_VERSION,
  WEBGPU_VISIBLE_RECORD_WGSL_COMPUTED_FIELDS,
  cloneWebGpuVisibleRecordFieldComputeModes
} from './common_4dgs_record_contracts.js';
import {
  WEBGPU_INPUT_BUFFER_MODES,
  createWebGpuInputBufferContract,
  createWebGpuInputBufferModes
} from './common_4dgs_webgpu_input_contracts.js';

const DEFAULT_MAX_RECORDS = 65536;
const DEFAULT_EPSILON = DEFAULT_COMPARISON_EPSILON;
const RECORD_FLOATS = WEBGPU_VISIBLE_RECORD_FLOATS;
const IMPLEMENTED_FIELDS = WEBGPU_VISIBLE_RECORD_IMPLEMENTED_FIELDS;
const WGSL_COMPUTED_FIELDS = WEBGPU_VISIBLE_RECORD_WGSL_COMPUTED_FIELDS;
const WGSL_REFERENCE_ASSISTED_FIELDS = WEBGPU_VISIBLE_RECORD_REFERENCE_ASSISTED_FIELDS;
const CPU_MATERIALIZED_FIELDS = WEBGPU_VISIBLE_RECORD_CPU_MATERIALIZED_FIELDS;
const DEFERRED_FIELDS = WEBGPU_VISIBLE_RECORD_DEFERRED_FIELDS;

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
    schemaVersion: WEBGPU_VISIBLE_RECORD_DRY_RUN_SCHEMA_VERSION,
    phaseStep: WEBGPU_VISIBLE_RECORD_PHASE_STEP,
    status: 'fallback',
    reason,
    computeMode: WEBGPU_VISIBLE_RECORD_COMPUTE_MODE,
    scaffoldMode: WEBGPU_VISIBLE_RECORD_SCAFFOLD_MODE,
    implementedFields: IMPLEMENTED_FIELDS,
    wgslComputedFields: WGSL_COMPUTED_FIELDS,
    wgslReferenceAssistedFields: WGSL_REFERENCE_ASSISTED_FIELDS,
    cpuMaterializedFields: CPU_MATERIALIZED_FIELDS,
    fieldComputeModes: cloneWebGpuVisibleRecordFieldComputeModes(),
    deferredFields: DEFERRED_FIELDS,
    candidateCount: extra.candidateCount ?? null,
    recordCount: extra.recordCount ?? null,
    validRecordCount: extra.validRecordCount ?? null,
    recordComparison: {
      [RECORD_COMPARISON_KEYS.ANY_MISMATCH]: true,
      [RECORD_COMPARISON_KEYS.FIELD_MISMATCH_COUNT]: null,
      [RECORD_COMPARISON_KEYS.MAX_ABS_ERROR]: null,
      [RECORD_COMPARISON_KEYS.FIRST_MISMATCHES]: extra.firstMismatches ?? []
    },
    comparisonContract: {
      schemaVersion: COMPARISON_CONTRACT_SCHEMA_VERSION,
      recordComparisonKeys: RECORD_COMPARISON_KEYS,
      mismatchClassifications: MISMATCH_CLASSIFICATIONS
    },
    comparisonTolerance: createComparisonToleranceMetadata(),
    fieldMismatchCount: null,
    firstMismatches: extra.firstMismatches ?? [],
    mismatchClassification: extra.mismatchClassification ??
      MISMATCH_CLASSIFICATIONS.WEBGPU_VISIBLE_RECORD_UNAVAILABLE,
    timing: extra.timing ?? null,
    inputContract: extra.inputContract ?? null,
    bufferContract: extra.bufferContract ?? null,
    inputBufferModes: extra.inputBufferModes ?? createWebGpuInputBufferModes(),
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

function buildStatePositionsForCandidates(raw, candidateIndices, buildConfig) {
  const count = candidateIndices.length;
  const out = new Float32Array(count * 4);
  const flags = {
    nativeRot4d: !!buildConfig.useNativeRot4d,
    nativeMarginal: !!buildConfig.useNativeMarginal
  };
  for (let i = 0; i < count; i += 1) {
    const srcIndex = candidateIndices[i];
    const state = computeGaussianState(
      raw,
      srcIndex,
      buildConfig.timestamp,
      buildConfig.scalingModifier,
      buildConfig.sigmaScale,
      buildConfig.prefilterVar,
      buildConfig.useRot4d,
      flags
    );
    const o = i * 4;
    if (state?.pos) {
      out[o + 0] = state.pos[0] ?? 0;
      out[o + 1] = state.pos[1] ?? 0;
      out[o + 2] = state.pos[2] ?? 0;
      out[o + 3] = 1;
    } else {
      out[o + 0] = 0;
      out[o + 1] = 0;
      out[o + 2] = 0;
      out[o + 3] = 0;
    }
  }
  return out;
}

function compareRecords(reference, candidate, count, { epsilon, maxMismatches }) {
  const firstMismatches = [];
  let fieldMismatchCount = 0;
  let maxAbsError = 0;
  const fields = WEBGPU_VISIBLE_RECORD_FIELDS.filter(([field]) => field !== 'reserved');
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
            firstMismatches.push(createRecordMismatch({
              row,
              field,
              component: components > 1 ? c : null,
              expected: ref,
              actual: got,
              absError: diff
            }));
          }
        }
      }
    }
  }
  return createRecordComparisonResult({
    anyMismatch: fieldMismatchCount > 0,
    fieldMismatchCount,
    maxAbsError,
    firstMismatches
  });
}

function classifyComparison(comparison) {
  if (!comparison) return MISMATCH_CLASSIFICATIONS.WEBGPU_VISIBLE_RECORD_COMPARE_MISSING;
  if (!comparison.anyMismatch) return MISMATCH_CLASSIFICATIONS.NONE;
  const fields = new Set((comparison.firstMismatches ?? []).map((item) => item.field));
  if (fields.size === 1 && fields.has('aabb')) {
    return MISMATCH_CLASSIFICATIONS.WEBGPU_FIXED_RECORD_AABB_MISMATCH;
  }
  return MISMATCH_CLASSIFICATIONS.WEBGPU_FIXED_RECORD_FIELD_MISMATCH;
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

async function runCompute({ device, cpuReference, rawXyzOpacity, statePositions, projectionParams, rawCount }) {
  const shader = device.createShaderModule({
    label: 'phase3-step4-visible-record-projection-wgsl',
    code: `
struct Params {
  count: u32,
  recordFloats: u32,
  rawCount: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> candidates: array<u32>;
@group(0) @binding(1) var<storage, read> rawXyzOpacity: array<vec4f>;
@group(0) @binding(2) var<storage, read> referenceRecords: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> outputRecords: array<vec4f>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> statePositions: array<vec4f>;
@group(0) @binding(6) var<storage, read> projectionParams: array<vec4f>;

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
  let base = row * 3u;
  let srcIndex = candidates[row];
  let raw0 = rawXyzOpacity[row];
  let statePos = statePositions[row];
  var r0 = referenceRecords[base + 0u];
  var r1 = referenceRecords[base + 1u];
  var r2 = referenceRecords[base + 2u];

  // Phase 3 Step4: srcIndex and the minimal screen projection fields are
  // produced in WGSL. The 4D Gaussian state and AABB remain CPU materialized.
  r0.x = f32(srcIndex);

  let header = projectionParams[0u];
  let scale = projectionParams[1u];
  let intrinsics = projectionParams[2u];
  let mode = header.x;
  let renderW = header.y;
  let renderH = header.z;
  let sx = scale.x;
  let sy = scale.y;
  let pixelXSign = scale.z;
  let pos4 = vec4f(statePos.x, statePos.y, statePos.z, 1.0);
  let mv4 = vec4f(
    rowDot(viewRow(0u), pos4),
    rowDot(viewRow(1u), pos4),
    rowDot(viewRow(2u), pos4),
    rowDot(viewRow(3u), pos4)
  );

  var projectedPx = 0.0;
  var projectedPy = 0.0;
  var projectedDepth = 0.0;
  var projectionOk = false;
  if (mode > 0.5) {
    projectedDepth = mv4.z;
    projectionOk = projectedDepth > 1e-6;
    projectedPx = (pixelXSign * intrinsics.x * (mv4.x / max(projectedDepth, 1e-8)) + intrinsics.z) * sx;
    projectedPy = (intrinsics.y * (mv4.y / max(projectedDepth, 1e-8)) + intrinsics.w) * sy;
  } else {
    projectedDepth = -mv4.z;
    let clip = vec4f(
      rowDot(projectionRow(0u), mv4),
      rowDot(projectionRow(1u), mv4),
      rowDot(projectionRow(2u), mv4),
      rowDot(projectionRow(3u), mv4)
    );
    let invW = 1.0 / (clip.w + 1e-7);
    let ndcX = clip.x * invW;
    let ndcY = clip.y * invW;
    projectionOk = projectedDepth > 1e-6;
    projectedPx = (((ndcX + 1.0) * renderW - 1.0) * 0.5) * sx;
    projectedPy = (((ndcY + 1.0) * renderH - 1.0) * 0.5) * sy;
  }

  let referenceValid = r0.y > 0.5;
  let rawIndexInBounds = srcIndex < params.rawCount;
  let outputValid = referenceValid && rawIndexInBounds && statePos.w > 0.5 && projectionOk;
  r0.y = select(0.0, 1.0, outputValid);
  r0.z = select(0.0, projectedPx, outputValid);
  r0.w = select(0.0, projectedPy, outputValid);
  r1.x = select(0.0, projectedDepth, outputValid);

  // Reserved lanes carry a tiny raw-buffer fetch probe for future diagnostics.
  // They are outside the compared fixed-record fields.
  r2.y = raw0.x;
  r2.z = raw0.y;
  r2.w = raw0.z;

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
  const statePositionBuffer = createBuffer(device, statePositions, GPUBufferUsage.STORAGE);
  const projectionParamsBuffer = createBuffer(device, projectionParams, GPUBufferUsage.STORAGE);
  const referenceBuffer = createBuffer(device, cpuReference.records, GPUBufferUsage.STORAGE);
  const outputBuffer = device.createBuffer({
    size: outputByteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const paramsBuffer = createBuffer(
    device,
    new Uint32Array([cpuReference.count, RECORD_FLOATS, rawCount, 0]),
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
      { binding: 4, resource: { buffer: paramsBuffer } },
      { binding: 5, resource: { buffer: statePositionBuffer } },
      { binding: 6, resource: { buffer: projectionParamsBuffer } }
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
  maxMismatches = DEFAULT_MAX_MISMATCHES,
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
  const statePositions = buildStatePositionsForCandidates(raw, cpuReference.candidateIndices, buildConfig);
  const renderScale = Number.isFinite(buildConfig.renderScale) ? buildConfig.renderScale : 1;
  const renderW = Math.max(1, Math.round(canvasWidth * renderScale));
  const renderH = Math.max(1, Math.round(canvasHeight * renderScale));
  const sx = canvasWidth / renderW;
  const sy = canvasHeight / renderH;
  const projectionContract = buildWebGpuProjectionContract({ camera, screenSpaceCamera, renderW, renderH, sx, sy });
  const bufferUploadPrepareMs = nowMs() - uploadStartMs;
  const rawCount = toFiniteInteger(raw.count ?? raw.N ?? (raw.xyz?.length / Math.max(1, raw.xyzDim || 3)), 0);
  const computeResult = await runCompute({
    device,
    cpuReference,
    rawXyzOpacity,
    statePositions,
    projectionParams: projectionContract.data,
    rawCount
  });
  const inputContract = createWebGpuInputBufferContract({
    candidateCount: cpuReference.candidateCount,
    recordCount: cpuReference.count,
    rawCount,
    recordFloats: RECORD_FLOATS,
    outputBufferBytes: cpuReference.records.byteLength,
    projectionParamMode: projectionContract.summary.mode
  });
  const compareStartMs = nowMs();
  const comparisonTolerance = createComparisonToleranceMetadata({ epsilon, maxMismatches });
  const recordComparison = compareRecords(cpuReference.records, computeResult.records, cpuReference.count, {
    epsilon,
    maxMismatches
  });
  const compareMs = nowMs() - compareStartMs;
  const mismatchClassification = classifyComparison(recordComparison);
  return {
    schemaVersion: WEBGPU_VISIBLE_RECORD_DRY_RUN_SCHEMA_VERSION,
    phaseStep: WEBGPU_VISIBLE_RECORD_PHASE_STEP,
    status: 'ok',
    reason: 'ok',
    computeMode: WEBGPU_VISIBLE_RECORD_COMPUTE_MODE,
    scaffoldMode: WEBGPU_VISIBLE_RECORD_SCAFFOLD_MODE,
    scaffoldNote: 'Phase 3 Step4 computes srcIndex and minimal screen-space projection fields (px/py/depth) in WGSL from CPU-materialized 4D state positions. valid is still reference-assisted; aabb remains CPU-materialized until radius/covariance moves to WGSL.',
    implementedFields: IMPLEMENTED_FIELDS,
    wgslComputedFields: WGSL_COMPUTED_FIELDS,
    wgslReferenceAssistedFields: WGSL_REFERENCE_ASSISTED_FIELDS,
    cpuMaterializedFields: CPU_MATERIALIZED_FIELDS,
    fieldComputeModes: cloneWebGpuVisibleRecordFieldComputeModes(),
    deferredFields: DEFERRED_FIELDS,
    candidateCount: cpuReference.candidateCount,
    recordCount: cpuReference.count,
    validRecordCount: cpuReference.validCount,
    recordFloats: RECORD_FLOATS,
    recordLayout: WEBGPU_VISIBLE_RECORD_FIELDS,
    comparisonContract: {
      schemaVersion: comparisonTolerance.schemaVersion,
      recordComparisonKeys: RECORD_COMPARISON_KEYS,
      mismatchClassifications: MISMATCH_CLASSIFICATIONS
    },
    comparisonTolerance,
    inputContract,
    bufferContract: inputContract,
    inputBufferModes: inputContract.inputBufferModes,
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
      rawBufferUploadMode: WEBGPU_INPUT_BUFFER_MODES.RAW_XYZ_OPACITY,
      statePositionUploadMode: WEBGPU_INPUT_BUFFER_MODES.STATE_POSITIONS,
      projectionParamMode: projectionContract.summary.mode,
      projectionContract: projectionContract.summary,
      inputContract,
      bufferContract: inputContract,
      inputBufferModes: inputContract.inputBufferModes,
      candidateBufferCount: cpuReference.count,
      rawCount,
      outputBufferBytes: cpuReference.records.byteLength
    },
    metadata
  };
}
