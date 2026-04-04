import { bindInterleavedFloatAttribs } from './gpu_gl_utils.js';
import {
  createPackedUploadState,
  getPackedInterleavedAttribDescriptors,
  uploadPackedInterleaved,
  summarizePackedUploadState
} from './gpu_packed_upload_utils.js';

// Step23 fix:
// packed interleaved buffer を直接使う draw 実行部。
// packed で何も表示されない症状に対して、attribute bind を descriptor ベースへ統一し、
// VAO 構築時と draw 前の buffer/VAO 状態を明確化する。

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

function configurePackedDirectVao(gl, gpu, vao, uploadState) {
  const desc = getPackedInterleavedAttribDescriptors();

  // VAO に interleaved buffer の attribute 割当を一括設定する。
  bindInterleavedFloatAttribs(gl, {
    vao,
    program: gpu.program,
    buffer: uploadState.interleaved.buffer,
    attributes: desc.attributes
  });

  gpu.packedDirectLayout = desc;
  gpu.packedDirectConfigured = true;
  return desc
}

export function ensurePackedDirectDrawResources(gl, gpu) {
  const uploadState = ensurePackedUploadStateLocal(gl, gpu);
  const vao = ensurePackedDirectVao(gl, gpu);
  const layout = configurePackedDirectVao(gl, gpu, vao, uploadState);

  return {
    vao,
    uploadState,
    layout
  };
}

export function uploadAndDrawPackedDirect(gl, gpu, packedScreenSpace, canvasWidth, canvasHeight) {
  if (!packedScreenSpace?.packed || !Number.isFinite(packedScreenSpace?.packedCount)) {
    throw new Error('uploadAndDrawPackedDirect requires packedScreenSpace with packed/packedCount');
  }

  const { vao, uploadState, layout } = ensurePackedDirectDrawResources(gl, gpu);

  // 先に interleaved buffer を upload
  uploadPackedInterleaved(
    gl,
    uploadState,
    packedScreenSpace.packed,
    packedScreenSpace.packedCount
  );

  // upload 後に同じ buffer を改めて VAO に結び直して、状態の食い違いを避ける。
  configurePackedDirectVao(gl, gpu, vao, uploadState);

  gl.useProgram(gpu.program);
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, uploadState.interleaved.buffer);
  gl.uniform2f(gpu.uViewportPx, canvasWidth, canvasHeight);
  gl.drawArrays(gl.POINTS, 0, packedScreenSpace.packedCount);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  gl.bindVertexArray(null);

  const uploadSummary = summarizePackedUploadState(uploadState);

  return {
    drawCount: packedScreenSpace.packedCount,
    packedDirectDraw: true,
    packedInterleavedStrideBytes: layout.strideBytes,
    packedInterleavedBound: true,
    packedInterleavedAttributeCount: Array.isArray(layout.attributes) ? layout.attributes.length : 0,
    packedInterleavedOffsets: Array.isArray(layout.attributes)
      ? layout.attributes.map(a => `${a.name}:${a.offset}`).join(', ')
      : '',
    uploadSummary
  };
}
