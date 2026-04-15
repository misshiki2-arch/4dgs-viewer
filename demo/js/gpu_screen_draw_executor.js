import { bindInterleavedFloatAttribs } from './gpu_gl_utils.js';
import {
  createPackedUploadState,
  getPackedInterleavedAttribDescriptors,
  uploadPackedInterleaved,
  summarizePackedUploadState
} from './gpu_packed_upload_utils.js';
import { packVisibleItems } from './gpu_visible_pack_utils.js';
import { drawGpuPackedPayloads, ensureGpuPackedPayloadTextureDrawResources } from './gpu_packed_payload_draw_shared.js';
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
    lastUploadManagedUploadCount: 0,
    usesGpuResidentPayload: false,
    lastSharedSetupCount: 0,
    lastSharedBindCount: 0,
    lastSharedDispatchCount: 0,
    lastSharedDispatchMode: 'none',
    lastSharedPayloadCount: 0
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

function ensureGpuScreenTextureDrawResources(gl, gpu) {
  return ensureGpuPackedPayloadTextureDrawResources(gl, gpu, 'gpuScreenTextureDrawResources');
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

function resetGpuScreenUploadState(state) {
  updateGpuScreenUploadStateFromSummary(state, {
    packedUploadBytes: 0,
    packedUploadCount: 0,
    packedUploadLength: 0,
    packedUploadCapacityBytes: 0,
    packedUploadReusedCapacity: false,
    packedUploadManagedCapacityReused: false,
    packedUploadManagedCapacityGrown: false,
    packedUploadManagedUploadCount: 0
  });
}

function resetGpuScreenSharedDrawState(state) {
  if (!state) return;
  state.lastSharedSetupCount = 0;
  state.lastSharedBindCount = 0;
  state.lastSharedDispatchCount = 0;
  state.lastSharedDispatchMode = 'none';
  state.lastSharedPayloadCount = 0;
}

function prepareFullFrameGpuScreenDraw(gl, canvasWidth, canvasHeight) {
  // Full-frame gpu-screen draw must not inherit tile-only state.
  // The renderer owns draw-path selection and blend/depth policy.
  // This executor now only keeps the seam for any future gpu-screen-specific
  // preparation without forcing additional GL state.
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
    sourcePackedLength:
      gpuScreenSpace?.packed instanceof Float32Array
        ? gpuScreenSpace.packed.length
        : (
          Number.isFinite(gpuScreenSpace?.packedCount) && Number.isFinite(gpuScreenSpace?.floatsPerItem)
            ? (gpuScreenSpace.packedCount | 0) * (gpuScreenSpace.floatsPerItem | 0)
            : 0
        ),

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

function buildCpuFallbackGpuScreenSpace(gpuScreenSpace) {
  const items = Array.isArray(gpuScreenSpace?.sourceItems) ? gpuScreenSpace.sourceItems : [];
  const packedResult = packVisibleItems(items);
  return {
    ...gpuScreenSpace,
    packed: packedResult?.packed instanceof Float32Array ? packedResult.packed : null,
    packedCount: Number.isFinite(packedResult?.count) ? packedResult.count : items.length,
    floatsPerItem: Number.isFinite(packedResult?.floatsPerItem)
      ? packedResult.floatsPerItem
      : 16
  };
}

function drawGpuScreenWithPackedUpload(gl, gpu, gpuScreenSpace, canvasWidth, canvasHeight, state) {
  const drawCount = Number.isFinite(gpuScreenSpace?.packedCount)
    ? Math.max(0, gpuScreenSpace.packedCount | 0)
    : 0;

  const {
    state: ensuredState,
    vao,
    uploadState,
    layout
  } = ensureGpuScreenDrawResources(gl, gpu);

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
  ensuredState.usesGpuResidentPayload = false;
  ensuredState.usesPackedReferenceLayout = true;
  ensuredState.usesPackedReferenceShader = true;
  ensuredState.usesPackedReferenceUpload = true;
  resetGpuScreenSharedDrawState(ensuredState);

  prepareFullFrameGpuScreenDraw(gl, canvasWidth, canvasHeight);
  gl.useProgram(gpu.program);
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, uploadState.interleaved.buffer);
  gl.uniform2f(gpu.uViewportPx, canvasWidth, canvasHeight);
  gl.drawArrays(gl.POINTS, 0, drawCount);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  gl.bindVertexArray(null);

  const uploadSummary = summarizePackedUploadState(uploadState);
  updateGpuScreenUploadStateFromSummary(ensuredState, uploadSummary);
  return drawCount;
}

function drawGpuScreenWithGpuResidentPayloads(gl, gpu, gpuScreenSpace, canvasWidth, canvasHeight, state) {
  const resources = ensureGpuScreenTextureDrawResources(gl, gpu);
  const drawResult = drawGpuPackedPayloads(gl, gpu, gpuScreenSpace, canvasWidth, canvasHeight, {
    storageKey: 'gpuScreenTextureDrawResources',
    resources
  });
  if (!drawResult) return null;

  state.hasProgram = !!resources.program;
  state.hasVao = !!resources.vao;
  state.hasBuffers = true;
  state.layoutVersion = 0;
  state.strideBytes = 0;
  state.attributeCount = 0;
  state.lastOffsets = 'gpu-packed-texture';
  state.configured = true;
  state.lastReason = 'ready';
  state.usesGpuResidentPayload = true;
  state.usesPackedReferenceLayout = false;
  state.usesPackedReferenceShader = false;
  state.usesPackedReferenceUpload = false;
  state.lastSharedSetupCount = drawResult.setupCount ?? 0;
  state.lastSharedBindCount = drawResult.bindCount ?? 0;
  state.lastSharedDispatchCount = drawResult.dispatchCount ?? 0;
  state.lastSharedDispatchMode = drawResult.dispatchMode ?? 'none';
  state.lastSharedPayloadCount = drawResult.payloadCount ?? 0;
  resetGpuScreenUploadState(state);

  prepareFullFrameGpuScreenDraw(gl, canvasWidth, canvasHeight);
  return drawResult;
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
    gpuScreenUsesGpuResidentPayload: !!state.usesGpuResidentPayload,
    gpuScreenSharedSetupCount: state.lastSharedSetupCount ?? 0,
    gpuScreenSharedBindCount: state.lastSharedBindCount ?? 0,
    gpuScreenSharedDispatchCount: state.lastSharedDispatchCount ?? 0,
    gpuScreenSharedDispatchMode: state.lastSharedDispatchMode ?? 'none',
    gpuScreenSharedPayloadCount: state.lastSharedPayloadCount ?? 0,
    gpuScreenGpuResidentPayloadAvailable:
      !!gpu?.gpuScreenTextureDrawResources?.program &&
      !!gpu?.gpuScreenTextureDrawResources?.vao,

    gpuScreenUploadSummary: uploadSummary
  };
}

export function uploadAndDrawGpuScreen(gl, gpu, gpuScreenSpace, canvasWidth, canvasHeight) {
  const state = ensureGpuScreenState(gpu);
  let textureDrawResult = drawGpuScreenWithGpuResidentPayloads(
    gl,
    gpu,
    gpuScreenSpace,
    canvasWidth,
    canvasHeight,
    state
  );
  let drawCount = textureDrawResult?.drawCount ?? null;
  let drawInput = gpuScreenSpace;

  if (drawCount === null) {
    drawInput = gpuScreenSpace?.packed instanceof Float32Array
      ? gpuScreenSpace
      : buildCpuFallbackGpuScreenSpace(gpuScreenSpace);

    if (!(drawInput?.packed instanceof Float32Array)) {
      return buildFailureResult(gpu, gpuScreenSpace, 'missing-gpu-screen-source');
    }

    drawCount = drawGpuScreenWithPackedUpload(
      gl,
      gpu,
      drawInput,
      canvasWidth,
      canvasHeight,
      state
    );
  }

  state.lastActualPath = 'gpu-screen';
  state.lastBuildMs = Number.isFinite(gpuScreenSpace?.summary?.buildMs)
    ? gpuScreenSpace.summary.buildMs
    : 0;
  state.lastDrawCount = drawCount;
  state.lastReason = 'ready';

  const comparisonSummary = buildGpuScreenComparisonSummary(state, gpuScreenSpace);

  return {
    drawCount,
    drawCallCount: textureDrawResult?.drawCallCount ?? 1,
    sharedSetupCount: textureDrawResult?.setupCount ?? 0,
    sharedBindCount: textureDrawResult?.bindCount ?? 0,
    sharedDispatchCount: textureDrawResult?.dispatchCount ?? 0,
    sharedDispatchMode: textureDrawResult?.dispatchMode ?? 'none',
    sharedPayloadCount: textureDrawResult?.payloadCount ?? 0,
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
