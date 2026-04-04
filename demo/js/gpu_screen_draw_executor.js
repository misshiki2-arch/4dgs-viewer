import { bindInterleavedFloatAttribs } from './gpu_gl_utils.js';
import {
  createPackedUploadState,
  getPackedInterleavedAttribDescriptors,
  uploadPackedInterleaved,
  summarizePackedUploadState
} from './gpu_packed_upload_utils.js';

// Step27:
// gpu-screen path を packed の単なる別名から一段分離する。
// まだ本格的な専用 shader / 専用 layout には進まないが、少なくとも
// - gpu-screen 独自 state
// - gpu-screen 独自 summary
// - packed 参照であることの明示
// を持たせ、比較実験経路として育てられる形にする。
//
// 現段階の方針:
// 1. draw は引き続き interleaved packed buffer を流用する
// 2. ただし state / summary / reason は gpu-screen 名義で独立させる
// 3. 将来 shader/layout/program を差し替えられるよう API を固定する

function createDefaultGpuScreenState() {
  return {
    isReady: true,
    configured: false,

    hasProgram: false,
    hasVao: false,
    hasBuffers: false,

    // 現段階では packed formal path を参照していることを明示する
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

    lastReferencePath: 'packed-cpu',
    lastReferenceLayoutVersion: 0,
    lastReferenceStrideBytes: 0,
    lastReferenceAttributeCount: 0,

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

function buildGpuScreenLayoutSummary(desc) {
  return {
    gpuScreenLayoutVersion: desc?.layoutVersion ?? 0,
    gpuScreenStrideBytes: desc?.strideBytes ?? 0,
    gpuScreenAttributeCount: Array.isArray(desc?.attributes) ? desc.attributes.length : 0,
    gpuScreenOffsets: buildOffsetsText(desc?.attributes),
    gpuScreenUsesPackedReferenceLayout: true,
    gpuScreenUsesPackedReferenceShader: true,
    gpuScreenUsesPackedReferenceUpload: true
  };
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

  state.lastReferenceLayoutVersion = desc.layoutVersion ?? 0;
  state.lastReferenceStrideBytes = desc.strideBytes ?? 0;
  state.lastReferenceAttributeCount = Array.isArray(desc.attributes) ? desc.attributes.length : 0;
  state.lastReferencePath = 'packed-cpu';

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
    gpuScreenReferencePath: state.lastReferencePath ?? 'packed-cpu',
    gpuScreenReferenceLayoutVersion: state.lastReferenceLayoutVersion ?? 0,
    gpuScreenReferenceStrideBytes: state.lastReferenceStrideBytes ?? 0,
    gpuScreenReferenceAttributeCount: state.lastReferenceAttributeCount ?? 0,
    gpuScreenSourcePath: gpuScreenSpace?.path ?? 'none',
    gpuScreenSourceExperimental: !!gpuScreenSpace?.experimental,
    gpuScreenSourceBuildMs: Number.isFinite(gpuScreenSpace?.summary?.buildMs)
      ? gpuScreenSpace.summary.buildMs
      : 0
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

    gpuScreenUsesPackedReferenceLayout: !!state.usesPackedReferenceLayout,
    gpuScreenUsesPackedReferenceShader: !!state.usesPackedReferenceShader,
    gpuScreenUsesPackedReferenceUpload: !!state.usesPackedReferenceUpload,

    gpuScreenLayoutVersion: state.layoutVersion ?? 0,
    gpuScreenStrideBytes: state.strideBytes ?? 0,
    gpuScreenAttributeCount: state.attributeCount ?? 0,
    gpuScreenOffsets: state.lastOffsets ?? '',

    gpuScreenLastPath: state.lastPath ?? 'gpu-screen',
    gpuScreenLastBuildMs: state.lastBuildMs ?? 0,
    gpuScreenLastDrawCount: state.lastDrawCount ?? 0,
    gpuScreenReason: state.lastReason ?? 'unknown',

    gpuScreenReferencePath: state.lastReferencePath ?? 'packed-cpu',
    gpuScreenReferenceLayoutVersion: state.lastReferenceLayoutVersion ?? 0,
    gpuScreenReferenceStrideBytes: state.lastReferenceStrideBytes ?? 0,
    gpuScreenReferenceAttributeCount: state.lastReferenceAttributeCount ?? 0,

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

    return {
      drawCount: 0,
      gpuScreenDraw: false,
      gpuScreenReady: false,
      gpuScreenFallbackSuggested: true,
      gpuScreenReason: 'missing-gpu-screen-source',
      gpuScreenSummary: summarizeGpuScreenDrawState(gpu),
      gpuScreenComparisonSummary: buildGpuScreenComparisonSummary(state, gpuScreenSpace),
      packedScreenSpacePath: gpuScreenSpace?.path ?? 'none',
      packedScreenSpaceSummary: gpuScreenSpace?.summary ?? null
    };
  }

  const drawCount = Number.isFinite(gpuScreenSpace?.packedCount)
    ? Math.max(0, gpuScreenSpace.packedCount | 0)
    : 0;

  const { uploadState, vao, layout } = ensureGpuScreenDrawResources(gl, gpu);

  // Step27:
  // draw 実体はまだ packed formal reference を流用する。
  // ただし、gpu-screen 側の state / summary は独立名義で更新し、
  // packed を参照していることを明示的に残す。
  uploadPackedInterleaved(gl, uploadState, gpuScreenSpace.packed, drawCount);
  configureGpuScreenVao(gl, gpu, vao, uploadState, state);

  gl.useProgram(gpu.program);
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, uploadState.interleaved.buffer);
  gl.uniform2f(gpu.uViewportPx, canvasWidth, canvasHeight);
  gl.drawArrays(gl.POINTS, 0, drawCount);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  gl.bindVertexArray(null);

  state.lastPath = 'gpu-screen';
  state.lastBuildMs = Number.isFinite(gpuScreenSpace?.summary?.buildMs)
    ? gpuScreenSpace.summary.buildMs
    : 0;
  state.lastDrawCount = drawCount;
  state.lastReason = 'ready';

  const uploadSummary = summarizePackedUploadState(uploadState);
  updateGpuScreenUploadStateFromSummary(state, uploadSummary);

  return {
    drawCount,
    gpuScreenDraw: true,
    gpuScreenReady: true,
    gpuScreenFallbackSuggested: false,
    gpuScreenReason: 'ready',

    ...buildGpuScreenLayoutSummary(layout),

    gpuScreenSummary: summarizeGpuScreenDrawState(gpu),
    gpuScreenComparisonSummary: buildGpuScreenComparisonSummary(state, gpuScreenSpace),

    packedScreenSpacePath: gpuScreenSpace?.path ?? 'none',
    packedScreenSpaceSummary: gpuScreenSpace?.summary ?? null
  };
}
