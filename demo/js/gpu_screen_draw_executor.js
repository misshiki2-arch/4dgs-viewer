import { bindInterleavedFloatAttribs } from './gpu_gl_utils.js';
import {
  createPackedUploadState,
  getPackedInterleavedAttribDescriptors,
  uploadPackedInterleaved,
  summarizePackedUploadState
} from './gpu_packed_upload_utils.js';
import { ensurePackedDirectDrawResources } from './gpu_packed_draw_executor.js';
import {
  GPU_SCREEN_SPACE_SOURCE_PACKED_CPU,
  GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP,
  GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL,
  GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION
} from './gpu_screen_space_builder.js';

// Step32
// 目的:
// - Step30/31 の draw ownership separation を維持する
// - Step32 で追加した source internal stages を gpu-screen executor の summary に反映する
// - packed formal draw contract は変えず、comparison / execution summary だけを source-stage aware にする
//
// 設計:
// 1. upload state / VAO / draw ownership は Step31 と同じ
// 2. attribute descriptor は packed formal contract をそのまま使う
// 3. source path / role / internal stage metrics は input gpuScreenSpace.summary から読む
// 4. readiness の公開挙動は Step30/31 互換を維持する

function createDefaultGpuScreenState() {
  return {
    isReady: true,
    configured: false,

    hasProgram: false,
    hasVao: false,
    hasBuffers: false,

    usesPackedReferenceLayout: true,
    usesPackedReferenceShader: true,
    usesPackedReferenceUpload: false,

    layoutVersion: 0,
    strideBytes: 0,
    attributeCount: 0,
    lastOffsets: '',

    lastActualPath: 'none',
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

function prepareFullFrameGpuScreenDraw(gl, canvasWidth, canvasHeight) {
  // Full-frame gpu-screen draw must not inherit tile-only state.
  // The renderer owns draw-path selection and blend/depth policy; this executor only
  // restores the minimal framebuffer/scissor/viewport state required for a stable
  // full-frame draw.
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  gl.disable(gl.SCISSOR_TEST);
  gl.disable(gl.CULL_FACE);
  gl.disable(gl.DEPTH_TEST);
  gl.depthMask(false);
  gl.enable(gl.BLEND);
  gl.blendFuncSeparate(
    gl.SRC_ALPHA,
    gl.ONE_MINUS_SRC_ALPHA,
    gl.ONE,
    gl.ONE_MINUS_SRC_ALPHA
  );
  gl.viewport(0, 0, canvasWidth, canvasHeight);
}

function normalizeSourcePath(gpuScreenSpace) {
  const path = gpuScreenSpace?.path;
  if (path === GPU_SCREEN_SPACE_SOURCE_PACKED_CPU) return GPU_SCREEN_SPACE_SOURCE_PACKED_CPU;
  if (path === GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP) return GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP;
  if (path === GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL) return GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL;
  return 'none';
}

function getSourceRole(path, experimental) {
  if (path === GPU_SCREEN_SPACE_SOURCE_PACKED_CPU) return 'formal-source';
  if (experimental || path === GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP || path === GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL) {
    return 'experimental-source';
  }
  return 'formal-source';
}

function getSourceStageMetrics(gpuScreenSpace) {
  const summary = gpuScreenSpace?.summary ?? null;
  return {
    sourceItemCount: Number.isFinite(summary?.sourceItemCount) ? summary.sourceItemCount : 0,
    sourceSchemaVersion: Number.isFinite(summary?.sourceSchemaVersion)
      ? summary.sourceSchemaVersion
      : GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION,
    sourcePrepStageMs: Number.isFinite(summary?.prepStageMs) ? summary.prepStageMs : 0,
    sourcePackStageMs: Number.isFinite(summary?.packStageMs) ? summary.packStageMs : 0
  };
}

function buildGpuScreenComparisonSummary(state, gpuScreenSpace) {
  const sourcePath = normalizeSourcePath(gpuScreenSpace);
  const sourceExperimental =
    !!gpuScreenSpace?.experimental ||
    sourcePath === GPU_SCREEN_SPACE_SOURCE_PACKED_GPU_PREP ||
    sourcePath === GPU_SCREEN_SPACE_SOURCE_GPU_SCREEN_EXPERIMENTAL;
  const sourceRole = getSourceRole(sourcePath, sourceExperimental);
  const stageMetrics = getSourceStageMetrics(gpuScreenSpace);

  return {
    actualPath: 'gpu-screen',
    actualRole: 'experimental-draw',

    sourcePath,
    sourceRole,
    sourceExperimental,
    sourceBuildMs: Number.isFinite(gpuScreenSpace?.summary?.buildMs)
      ? gpuScreenSpace.summary.buildMs
      : 0,
    sourcePackedCount: Number.isFinite(gpuScreenSpace?.packedCount)
      ? gpuScreenSpace.packedCount
      : 0,
    sourcePackedLength: gpuScreenSpace?.packed instanceof Float32Array
      ? gpuScreenSpace.packed.length
      : 0,

    sourceItemCount: stageMetrics.sourceItemCount,
    sourceSchemaVersion: stageMetrics.sourceSchemaVersion,
    sourcePrepStageMs: stageMetrics.sourcePrepStageMs,
    sourcePackStageMs: stageMetrics.sourcePackStageMs,

    referencePath: GPU_SCREEN_SPACE_SOURCE_PACKED_CPU,
    referenceRole: 'formal-reference',

    usesPackedReferenceLayout: !!state.usesPackedReferenceLayout,
    usesPackedReferenceShader: !!state.usesPackedReferenceShader,
    usesPackedReferenceUpload: !!state.usesPackedReferenceUpload,

    sameLayoutAsReference: !!state.usesPackedReferenceLayout,
    samePackCountAsReference: true
  };
}

function buildGpuScreenExecutionSummary({ drawCount, reason, ready, comparisonSummary }) {
  return {
    gpuScreenDraw: ready && reason === 'ready',
    gpuScreenReady: !!ready,
    gpuScreenReason: reason ?? 'unknown',
    gpuScreenDrawCount: drawCount ?? 0,
    gpuScreenActualPath: comparisonSummary?.actualPath ?? 'gpu-screen',
    gpuScreenSourcePath: comparisonSummary?.sourcePath ?? 'none',
    gpuScreenReferencePath: comparisonSummary?.referencePath ?? GPU_SCREEN_SPACE_SOURCE_PACKED_CPU,
    gpuScreenSourceItemCount: comparisonSummary?.sourceItemCount ?? 0,
    gpuScreenSourceSchemaVersion: comparisonSummary?.sourceSchemaVersion ?? GPU_SCREEN_SPACE_SOURCE_SCHEMA_VERSION,
    gpuScreenSourcePrepStageMs: comparisonSummary?.sourcePrepStageMs ?? 0,
    gpuScreenSourcePackStageMs: comparisonSummary?.sourcePackStageMs ?? 0
  };
}

function buildFailureResult(gpu, gpuScreenSpace, reason) {
  const state = ensureGpuScreenState(gpu);
  state.lastActualPath = 'gpu-screen';
  state.lastBuildMs = Number.isFinite(gpuScreenSpace?.summary?.buildMs)
    ? gpuScreenSpace.summary.buildMs
    : 0;
  state.lastDrawCount = 0;
  state.lastReason = reason;

  const comparisonSummary = buildGpuScreenComparisonSummary(state, gpuScreenSpace);

  return {
    drawCount: 0,
    gpuScreenSummary: summarizeGpuScreenDrawState(gpu),
    gpuScreenComparisonSummary: comparisonSummary,
    gpuScreenExecutionSummary: buildGpuScreenExecutionSummary({
      drawCount: 0,
      reason,
      ready: false,
      comparisonSummary
    }),
    packedScreenSpacePath: gpuScreenSpace?.path ?? 'none',
    packedScreenSpaceSummary: gpuScreenSpace?.summary ?? null
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

    gpuScreenLastActualPath: state.lastActualPath ?? 'none',
    gpuScreenLastBuildMs: state.lastBuildMs ?? 0,
    gpuScreenLastDrawCount: state.lastDrawCount ?? 0,
    gpuScreenReason: state.lastReason ?? 'unknown',

    gpuScreenUsesPackedReferenceLayout: !!state.usesPackedReferenceLayout,
    gpuScreenUsesPackedReferenceShader: !!state.usesPackedReferenceShader,
    gpuScreenUsesPackedReferenceUpload: !!state.usesPackedReferenceUpload,

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
    return buildFailureResult(gpu, gpuScreenSpace, 'missing-gpu-screen-source');
  }

  const drawCount = Number.isFinite(gpuScreenSpace?.packedCount)
    ? Math.max(0, gpuScreenSpace.packedCount | 0)
    : 0;

  const { state: ensuredState } = ensureGpuScreenDrawResources(gl, gpu);
  // Step45 fix:
  // full-frame gpu-screen draw reuses the known-good packed direct draw resources.
  // This keeps the gpu-screen path on the same formal packed upload/VAO contract
  // as the packed full-frame path, while leaving comparison/execution summaries here.
  const {
    vao,
    uploadState,
    layout
  } = ensurePackedDirectDrawResources(gl, gpu);

  uploadPackedInterleaved(gl, uploadState, gpuScreenSpace.packed, drawCount);
  ensuredState.hasProgram = !!gpu.program;
  ensuredState.hasVao = !!vao;
  ensuredState.hasBuffers = !!uploadState?.interleaved?.buffer;
  ensuredState.layoutVersion = layout?.layoutVersion ?? 0;
  ensuredState.strideBytes = layout?.strideBytes ?? 0;
  ensuredState.attributeCount = Array.isArray(layout?.attributes) ? layout.attributes.length : 0;
  ensuredState.lastOffsets = buildOffsetsText(layout?.attributes);
  ensuredState.configured = true;
  ensuredState.lastReason = 'ready';

  prepareFullFrameGpuScreenDraw(gl, canvasWidth, canvasHeight);
  gl.useProgram(gpu.program);
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, uploadState.interleaved.buffer);
  gl.uniform2f(gpu.uViewportPx, canvasWidth, canvasHeight);
  gl.drawArrays(gl.POINTS, 0, drawCount);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  gl.bindVertexArray(null);

  ensuredState.lastActualPath = 'gpu-screen';
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
