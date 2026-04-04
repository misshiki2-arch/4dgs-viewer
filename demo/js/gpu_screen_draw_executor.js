import { bindInterleavedFloatAttribs } from './gpu_gl_utils.js';
import {
  createPackedUploadState,
  getPackedInterleavedAttribDescriptors,
  uploadPackedInterleaved,
  summarizePackedUploadState
} from './gpu_packed_upload_utils.js';

// Step26:
// gpu-screen path を「future の名前」から「比較可能な実験 draw path」へ引き上げる。
// まだ本物の GPU screen-space 計算ではないが、少なくとも
// requestedDrawPath=gpu-screen -> actualDrawPath=gpu-screen
// で通せる executor を用意する。
//
// 現段階の方針:
// - 入力は packedScreenSpace をそのまま使う
// - packed direct draw と同じ interleaved buffer / descriptor を使う
// - ただし state / summary / debug 上は gpu-screen として独立扱いにする
// - 将来ここに専用 shader / buffer / build path を差し込めるよう、API を固定する

function createDefaultGpuScreenState() {
  return {
    isReady: true,
    hasProgram: false,
    hasVao: false,
    hasBuffers: false,
    layoutVersion: 0,
    strideBytes: 0,
    attributeCount: 0,
    lastPath: 'gpu-screen',
    lastBuildMs: 0,
    lastDrawCount: 0,
    lastReason: 'ready',
    lastOffsets: '',
    configured: false
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
    gpuScreenUploadSummary: uploadSummary
  };
}

export function uploadAndDrawGpuScreen(gl, gpu, gpuScreenSpace, canvasWidth, canvasHeight) {
  if (!(gpuScreenSpace?.packed instanceof Float32Array)) {
    const state = ensureGpuScreenState(gpu);
    state.lastPath = 'gpu-screen';
    state.lastBuildMs = 0;
    state.lastDrawCount = 0;
    state.lastReason = 'missing-packed-screen-space';

    return {
      drawCount: 0,
      gpuScreenDraw: false,
      gpuScreenReady: false,
      gpuScreenFallbackSuggested: true,
      gpuScreenReason: 'missing-packed-screen-space',
      gpuScreenSummary: summarizeGpuScreenDrawState(gpu),
      packedScreenSpacePath: gpuScreenSpace?.path ?? 'none',
      packedScreenSpaceSummary: gpuScreenSpace?.summary ?? null
    };
  }

  const drawCount = Number.isFinite(gpuScreenSpace?.packedCount)
    ? Math.max(0, gpuScreenSpace.packedCount | 0)
    : 0;

  const { state, uploadState, vao, layout } = ensureGpuScreenDrawResources(gl, gpu);

  // Step26:
  // 現時点では packed interleaved をそのまま流用して gpu-screen 経路を成立させる。
  // これにより selector / renderer / debug からは gpu-screen として扱える。
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

  return {
    drawCount,
    gpuScreenDraw: true,
    gpuScreenReady: true,
    gpuScreenFallbackSuggested: false,
    gpuScreenReason: 'ready',
    gpuScreenLayoutVersion: layout?.layoutVersion ?? 0,
    gpuScreenStrideBytes: layout?.strideBytes ?? 0,
    gpuScreenAttributeCount: Array.isArray(layout?.attributes) ? layout.attributes.length : 0,
    gpuScreenOffsets: buildOffsetsText(layout?.attributes),
    gpuScreenUploadSummary: summarizePackedUploadState(uploadState),
    gpuScreenSummary: summarizeGpuScreenDrawState(gpu),
    packedScreenSpacePath: gpuScreenSpace?.path ?? 'none',
    packedScreenSpaceSummary: gpuScreenSpace?.summary ?? null
  };
}
