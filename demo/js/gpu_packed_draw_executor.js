import { bindInterleavedFloatAttribs } from './gpu_gl_utils.js';
import {
  createPackedUploadState,
  getPackedInterleavedAttribDescriptors,
  uploadPackedInterleaved,
  summarizePackedUploadState
} from './gpu_packed_upload_utils.js';
import { drawGpuPackedPayloads, ensureGpuPackedPayloadTextureDrawResources } from './gpu_packed_payload_draw_shared.js';

// Step24:
// packed interleaved buffer を直接使う draw 実行部。
// Step23 では「表示されない」問題に対して VAO/buffer/attribute の結び直しを明確化した。
// Step24 ではさらに、次を正式化する。
//
// 1. attribute 定義は gpu_packed_upload_utils.js から来る descriptor のみを見る
// 2. renderer 側は packed layout を再解釈しない
// 3. draw executor は upload -> VAO 再設定 -> draw の順を固定
// 4. debug/summarize は正式契約ベースで返す
//
// ここでは packed direct draw だけを担当し、legacy / fallback の判断は持たない。

function buildZeroPackedUploadSummary() {
  return {
    packedUploadBytes: 0,
    packedUploadCount: 0,
    packedUploadLength: 0,
    packedUploadCapacityBytes: 0,
    packedUploadReusedCapacity: false,
    packedUploadManagedCapacityReused: false,
    packedUploadManagedCapacityGrown: false,
    packedUploadManagedUploadCount: 0
  };
}

function ensurePackedDirectVao(gl, gpu) {
  if (gpu.packedDirectVao) return gpu.packedDirectVao;
  gpu.packedDirectVao = gl.createVertexArray();
  return gpu.packedDirectVao;
}

function ensurePackedUploadStateLocal(gl, gpu) {
  if (gpu.packedUploadState?.interleaved?.buffer) return gpu.packedUploadState;
  gpu.packedUploadState = createPackedUploadState(gl);
  return gpu.packedUploadState;
}

function ensurePackedDirectTextureDrawResources(gl, gpu) {
  return ensureGpuPackedPayloadTextureDrawResources(gl, gpu, 'packedDirectTextureDrawResources');
}

function configurePackedDirectVao(gl, gpu, vao, uploadState) {
  const desc = getPackedInterleavedAttribDescriptors();

  bindInterleavedFloatAttribs(gl, {
    vao,
    program: gpu.program,
    buffer: uploadState.interleaved.buffer,
    attributes: desc.attributes
  });

  gpu.packedDirectLayout = desc;
  gpu.packedDirectConfigured = true;

  return desc;
}

function summarizePackedDirectLayout(layout) {
  return {
    packedDirectLayoutVersion: layout?.layoutVersion ?? 0,
    packedDirectStrideBytes: layout?.strideBytes ?? 0,
    packedDirectAttributeCount: Array.isArray(layout?.attributes)
      ? layout.attributes.length
      : 0,
    packedDirectAttributes: Array.isArray(layout?.attributes)
      ? layout.attributes.map(attr => ({
          name: attr.name,
          fieldName: attr.fieldName ?? '',
          canonicalFieldName: attr.canonicalFieldName ?? '',
          size: attr.size ?? 0,
          stride: attr.stride ?? 0,
          offset: attr.offset ?? 0
        }))
      : [],
    packedDirectOffsets: Array.isArray(layout?.attributes)
      ? layout.attributes.map(attr => `${attr.name}:${attr.offset}`).join(', ')
      : '',
    packedDirectAlphaSource: 'aColorAlpha.a -> colorAlpha[3]'
  };
}

export function ensurePackedDirectDrawResources(gl, gpu) {
  const uploadState = ensurePackedUploadStateLocal(gl, gpu);
  const vao = ensurePackedDirectVao(gl, gpu);
  const layout = getPackedInterleavedAttribDescriptors();

  return {
    vao,
    uploadState,
    layout
  };
}

export function uploadAndDrawPackedDirect(gl, gpu, packedScreenSpace, canvasWidth, canvasHeight) {
  const textureResources = ensurePackedDirectTextureDrawResources(gl, gpu);
  const drawResult = drawGpuPackedPayloads(gl, gpu, packedScreenSpace, canvasWidth, canvasHeight, {
    storageKey: 'packedDirectTextureDrawResources',
    resources: textureResources
  });

  if (drawResult) {
    gpu.packedDirectConfigured = true;
    gpu.packedDirectUsesGpuResidentPayload = true;

    return {
      drawCount: drawResult.drawCount,
      drawCallCount: drawResult.drawCallCount,
      packedDirectSharedSetupCount: drawResult.setupCount ?? 0,
      packedDirectSharedBindCount: drawResult.bindCount ?? 0,
      packedDirectSharedDispatchCount: drawResult.dispatchCount ?? 0,
      packedDirectSharedDispatchMode: drawResult.dispatchMode ?? 'none',
      packedDirectSharedPayloadCount: drawResult.payloadCount ?? 0,
      packedDirectSharedMergeCopyCount: drawResult.mergeCopyCount ?? 0,
      packedDirectSharedMergeAttempted: !!drawResult.mergeAttempted,
      packedDirectSharedMergeFailureReason: drawResult.mergeFailureReason ?? 'none',
      packedDirectSharedMergeTextureWidth: drawResult.mergeTextureWidth ?? 0,
      packedDirectSharedMergeTextureHeight: drawResult.mergeTextureHeight ?? 0,
      packedDirectSharedMergeRowCount: drawResult.mergeRowCount ?? 0,
      packedDirectSharedMergeRowsPerColumn: drawResult.mergeRowsPerColumn ?? 0,
      packedDirectSharedMergeColumnCount: drawResult.mergeColumnCount ?? 0,
      uploadCount: 0,
      packedDirectDraw: true,
      packedDirectUsesGpuResidentPayload: true,
      packedInterleavedBound: false,
      ...summarizePackedDirectLayout(null),
      uploadSummary: buildZeroPackedUploadSummary(),
      packedScreenSpacePath: packedScreenSpace?.path ?? 'packed-cpu',
      packedScreenSpaceSummary: packedScreenSpace?.summary ?? null
    };
  }

  if (!(packedScreenSpace?.packed instanceof Float32Array)) {
    throw new Error('uploadAndDrawPackedDirect requires packedScreenSpace.packed as Float32Array');
  }
  if (!Number.isFinite(packedScreenSpace?.packedCount)) {
    throw new Error('uploadAndDrawPackedDirect requires packedScreenSpace.packedCount');
  }

  const drawCount = Math.max(0, packedScreenSpace.packedCount | 0);
  const { vao, uploadState } = ensurePackedDirectDrawResources(gl, gpu);

  // Step24:
  // 先に interleaved buffer を upload し、その後に同じ buffer で VAO を再設定する。
  // これにより、buffer 再確保や容量拡張が起きても VAO が古い buffer を参照しない。
  uploadPackedInterleaved(gl, uploadState, packedScreenSpace.packed, drawCount);

  const layout = configurePackedDirectVao(gl, gpu, vao, uploadState);

  gl.useProgram(gpu.program);
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, uploadState.interleaved.buffer);

  gl.uniform2f(gpu.uViewportPx, canvasWidth, canvasHeight);
  gl.drawArrays(gl.POINTS, 0, drawCount);

  const uploadSummary = summarizePackedUploadState(uploadState);
  const layoutSummary = summarizePackedDirectLayout(layout);
  gpu.packedDirectUsesGpuResidentPayload = false;

  return {
    drawCount,
    drawCallCount: 1,
    packedDirectSharedSetupCount: 0,
    packedDirectSharedBindCount: 0,
    packedDirectSharedDispatchCount: 0,
    packedDirectSharedDispatchMode: 'none',
    packedDirectSharedPayloadCount: 0,
    packedDirectSharedMergeCopyCount: 0,
    packedDirectSharedMergeAttempted: false,
    packedDirectSharedMergeFailureReason: 'none',
    packedDirectSharedMergeTextureWidth: 0,
    packedDirectSharedMergeTextureHeight: 0,
    packedDirectSharedMergeRowCount: 0,
    packedDirectSharedMergeRowsPerColumn: 0,
    packedDirectSharedMergeColumnCount: 0,
    uploadCount: 1,
    packedDirectDraw: true,
    packedDirectUsesGpuResidentPayload: false,
    packedInterleavedBound: true,
    ...layoutSummary,
    uploadSummary,
    packedScreenSpacePath: packedScreenSpace?.path ?? 'packed-cpu',
    packedScreenSpaceSummary: packedScreenSpace?.summary ?? null
  };
}

export function summarizePackedDirectResources(gpu) {
  const layoutSummary = summarizePackedDirectLayout(gpu?.packedDirectLayout);
  const uploadSummary = summarizePackedUploadState(gpu?.packedUploadState);

  return {
    packedDirectConfigured: !!gpu?.packedDirectConfigured,
    packedDirectHasVao: !!gpu?.packedDirectVao,
    packedDirectGpuResidentPayloadAvailable:
      !!gpu?.packedDirectTextureDrawResources?.program &&
      !!gpu?.packedDirectTextureDrawResources?.vao,
    packedDirectSharedSetupCount: 0,
    packedDirectSharedBindCount: 0,
    packedDirectSharedDispatchCount: 0,
    packedDirectSharedDispatchMode: 'none',
    packedDirectSharedPayloadCount: 0,
    packedDirectSharedMergeCopyCount: 0,
    packedDirectSharedMergeAttempted: false,
    packedDirectSharedMergeFailureReason: 'none',
    packedDirectSharedMergeTextureWidth: 0,
    packedDirectSharedMergeTextureHeight: 0,
    packedDirectSharedMergeRowCount: 0,
    packedDirectSharedMergeRowsPerColumn: 0,
    packedDirectSharedMergeColumnCount: 0,
    ...layoutSummary,
    uploadSummary
  };
}
