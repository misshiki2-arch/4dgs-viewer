import { bindInterleavedFloatAttribs } from './gpu_gl_utils.js';
import {
  createPackedUploadState,
  getPackedInterleavedAttribDescriptors,
  uploadPackedInterleaved,
  summarizePackedUploadState
} from './gpu_packed_upload_utils.js';

// Step28:
// gpu-screen experimental path の state / comparison / execution summary を整理する。
// Step27 では packed 参照依存を明示したが、debug 上で state と comparison が少し重複していた。
// Step28 では
// - state: 実行器自身の現在状態
// - comparison: packed formal reference との比較情報
// - execution: 今回の draw 実行結果
// を分ける。

function createDefaultGpuScreenState() {
  return {
    isReady: true,
    configured: false,

    hasProgram: false,
    hasVao: false,
    hasBuffers: false,

    usesPackedReferenceLayout: true,
    usesPackedReferenceShader: true,
    usesPackedReferenceUpload: true,

    layoutVersion: 0,
    strideBytes: 0,
    attributeCount: 0,
    lastOffsets: '',

    lastPath: 'gpu-screen',
    lastBuildMs: 0,
    lastDrawCount: 0,
    lastReason: 'ready',

    lastUploadBytes: 0,
    lastUploadCount: 0,
    lastUploadLength: 0,
    lastUploadCapacityBytes: 0,
    lastUploadReusedCapacity: false,
    lastUploadManagedCapacityReused: false,
    lastUploadManagedCapacityGrown: false,
    lastUploadManagedUploadCount: 0
  };
}

function ensureGpuScreenState(gpu) {
  if (!gpu.gpuScreenDrawState) {
    gpu.gpuScreenDrawState = createDefaultGpuScreenState();
  }
  return gpu.gpuScreenDrawState;
}

function ensureGpuScreenUploadState(gl, gpu) {
  if (gpu.gpuScreenUploadState?.interleaved?.buffer) return gpu.gpuScreenUploadState;
  gpu.gpuScreenUploadState = createPackedUploadState(gl);
  return gpu.gpuScreenUploadState;
}

function ensureGpuScreenVao(gl, gpu) {
  if (gpu.gpuScreenVao) return gpu.gpuScreenVao;
  gpu.gpuScreenVao = gl.createVertexArray();
  return gpu.gpuScreenVao;
}

function buildOffsetsText(attributes) {
  if (!Array.isArray(attributes)) return '';
  return attributes.map((attr) => `${attr.name}:${attr.offset}`).join(', ');
}

function configureGpuScreenVao(gl, gpu, vao, uploadState, state) {
  const desc = getPackedInterleavedAttribDescriptors();

  bindInterleavedFloatAttribs(gl, {
    vao,
    program: gpu.program,
    buffer: uploadState.interleaved.buffer,
    attributes: desc.attributes
  });

  state.hasProgram = !!gpu.program;
  state.hasVao = !!vao;
  state.hasBuffers = !!uploadState?.interleaved?.buffer;

  state.layoutVersion = desc.layoutVersion ?? 0;
  state.strideBytes = desc.strideBytes ?? 0;
  state.attributeCount = Array.isArray(desc.attributes) ? desc.attributes.length : 0;
  state.lastOffsets = buildOffsetsText(desc.attributes);

  state.configured = true;
  state.lastReason = 'ready';

  gpu.gpuScreenLayout = desc;
  return desc;
}

function updateGpuScreenUploadStateFromSummary(state, uploadSummary) {
  if (!state || !uploadSummary) return;

  state.lastUploadBytes = uploadSummary.packedUploadBytes ?? 0;
  state.lastUploadCount = uploadSummary.packedUploadCount ?? 0;
  state.lastUploadLength = uploadSummary.packedUploadLength ?? 0;
  state.lastUploadCapacityBytes = uploadSummary.packedUploadCapacityBytes ?? 0;
  state.lastUploadReusedCapacity = !!uploadSummary.packedUploadReusedCapacity;
  state.lastUploadManagedCapacityReused = !!uploadSummary.packedUploadManagedCapacityReused;
  state.lastUploadManagedCapacityGrown = !!uploadSummary.packedUploadManagedCapacityGrown;
  state.lastUploadManagedUploadCount = uploadSummary.packedUploadManagedUploadCount ?? 0;
}

function buildGpuScreenComparisonSummary(state, gpuScreenSpace) {
  return {
    referencePath: 'packed-cpu',
    referenceRole: 'formal-reference',
    currentPath: gpuScreenSpace?.path ?? 'none',
    currentRole: gpuScreenSpace?.experimental ? 'experimental' : 'formal-reference',
    currentExperimental: !!gpuScreenSpace?.experimental,
    currentBuildMs: Number.isFinite(gpuScreenSpace?.summary?.buildMs)
      ? gpuScreenSpace.summary.buildMs
      : 0,
    currentPackedCount: Number.isFinite(gpuScreenSpace?.packedCount)
      ? gpuScreenSpace.packedCount
      : 0,
    currentPackedLength: gpuScreenSpace?.packed instanceof Float32Array
      ? gpuScreenSpace.packed.length
      : 0,
    usesPackedReferenceLayout: !!state.usesPackedReferenceLayout,
    usesPackedReferenceShader: !!state.usesPackedReferenceShader,
    usesPackedReferenceUpload: !!state.usesPackedReferenceUpload,
    sameLayoutAsReference: !!state.usesPackedReferenceLayout,
    samePackCountAsReference: true
  };
}

function buildGpuScreenExecutionSummary({
  drawCount,
  reason,
  ready,
  comparisonSummary
}) {
  return {
    gpuScreenDraw: ready && reason === 'ready',
    gpuScreenReady: !!ready,
    gpuScreenReason: reason ?? 'unknown',
    gpuScreenDrawCount: drawCount ?? 0,
    gpuScreenReferencePath: comparisonSummary?.referencePath ?? 'packed-cpu',
    gpuScreenSourcePath: comparisonSummary?.currentPath ?? 'none'
  };
}

export function createGpuScreenDrawState() {
  return createDefaultGpuScreenState();
}

export function ensureGpuScreenDrawResources(gl, gpu) {
  const state = ensureGpuScreenState(gpu);
  const uploadState = ensureGpuScreenUploadState(gl, gpu);
  const vao = ensureGpuScreenVao(gl, gpu);
  const layout = configureGpuScreenVao(gl, gpu, vao, uploadState, state);

  return {
    state,
    uploadState,
    vao,
    layout,
    ready: !!state.isReady,
    reason: state.lastReason
  };
}

export function isGpuScreenDrawReady(gpu) {
  return !!ensureGpuScreenState(gpu).isReady;
}

export function summarizeGpuScreenDrawState(gpu) {
  const state = ensureGpuScreenState(gpu);
  const uploadSummary = summarizePackedUploadState(gpu?.gpuScreenUploadState);

  return {
    gpuScreenDrawReady: !!state.isReady,
    gpuScreenConfigured: !!state.configured,
    gpuScreenHasProgram: !!state.hasProgram,
    gpuScreenHasVao: !!state.hasVao,
    gpuScreenHasBuffers: !!state.hasBuffers,

    gpuScreenLayoutVersion: state.layoutVersion ?? 0,
    gpuScreenStrideBytes: state.strideBytes ?? 0,
    gpuScreenAttributeCount: state.attributeCount ?? 0,
    gpuScreenOffsets: state.lastOffsets ?? '',

    gpuScreenLastPath: state.lastPath ?? 'gpu-screen',
    gpuScreenLastBuildMs: state.lastBuildMs ?? 0,
    gpuScreenLastDrawCount: state.lastDrawCount ?? 0,
    gpuScreenReason: state.lastReason ?? 'unknown',

    gpuScreenUploadBytes: state.lastUploadBytes ?? 0,
    gpuScreenUploadCount: state.lastUploadCount ?? 0,
    gpuScreenUploadLength: state.lastUploadLength ?? 0,
    gpuScreenUploadCapacityBytes: state.lastUploadCapacityBytes ?? 0,
    gpuScreenUploadReusedCapacity: !!state.lastUploadReusedCapacity,
    gpuScreenUploadManagedCapacityReused: !!state.lastUploadManagedCapacityReused,
    gpuScreenUploadManagedCapacityGrown: !!state.lastUploadManagedCapacityGrown,
    gpuScreenUploadManagedUploadCount: state.lastUploadManagedUploadCount ?? 0,

    gpuScreenUploadSummary: uploadSummary
  };
}

export function uploadAndDrawGpuScreen(gl, gpu, gpuScreenSpace, canvasWidth, canvasHeight) {
  const state = ensureGpuScreenState(gpu);

  if (!(gpuScreenSpace?.packed instanceof Float32Array)) {
    state.lastPath = 'gpu-screen';
    state.lastBuildMs = 0;
    state.lastDrawCount = 0;
    state.lastReason = 'missing-gpu-screen-source';

    const comparisonSummary = buildGpuScreenComparisonSummary(state, gpuScreenSpace);
    return {
      drawCount: 0,
      gpuScreenSummary: summarizeGpuScreenDrawState(gpu),
      gpuScreenComparisonSummary: comparisonSummary,
      gpuScreenExecutionSummary: buildGpuScreenExecutionSummary({
        drawCount: 0,
        reason: 'missing-gpu-screen-source',
        ready: false,
        comparisonSummary
      }),
      packedScreenSpacePath: gpuScreenSpace?.path ?? 'none',
      packedScreenSpaceSummary: gpuScreenSpace?.summary ?? null
    };
  }

  const drawCount = Number.isFinite(gpuScreenSpace?.packedCount)
    ? Math.max(0, gpuScreenSpace.packedCount | 0)
    : 0;

  const { state: ensuredState, uploadState, vao } = ensureGpuScreenDrawResources(gl, gpu);

  uploadPackedInterleaved(gl, uploadState, gpuScreenSpace.packed, drawCount);
  configureGpuScreenVao(gl, gpu, vao, uploadState, ensuredState);

  gl.useProgram(gpu.program);
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, uploadState.interleaved.buffer);
  gl.uniform2f(gpu.uViewportPx, canvasWidth, canvasHeight);
  gl.drawArrays(gl.POINTS, 0, drawCount);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  gl.bindVertexArray(null);

  ensuredState.lastPath = 'gpu-screen';
  ensuredState.lastBuildMs = Number.isFinite(gpuScreenSpace?.summary?.buildMs)
    ? gpuScreenSpace.summary.buildMs
    : 0;
  ensuredState.lastDrawCount = drawCount;
  ensuredState.lastReason = 'ready';

  const uploadSummary = summarizePackedUploadState(uploadState);
  updateGpuScreenUploadStateFromSummary(ensuredState, uploadSummary);

  const comparisonSummary = buildGpuScreenComparisonSummary(ensuredState, gpuScreenSpace);

  return {
    drawCount,
    gpuScreenSummary: summarizeGpuScreenDrawState(gpu),
    gpuScreenComparisonSummary: comparisonSummary,
    gpuScreenExecutionSummary: buildGpuScreenExecutionSummary({
      drawCount,
      reason: 'ready',
      ready: true,
      comparisonSummary
    }),
    packedScreenSpacePath: gpuScreenSpace?.path ?? 'none',
    packedScreenSpaceSummary: gpuScreenSpace?.summary ?? null
  };
}
