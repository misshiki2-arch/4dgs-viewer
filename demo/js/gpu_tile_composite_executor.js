import { bindInterleavedFloatAttribs, createProgram, enableFrontToBackAlphaBlend } from './gpu_gl_utils.js';
import {
  createPackedUploadState,
  getPackedInterleavedAttribDescriptors,
  uploadPackedInterleaved
} from './gpu_packed_upload_utils.js';
import {
  GPU_STEP_VERTEX_SHADER,
  GPU_STEP_FRAGMENT_SHADER_FRONT_TO_BACK
} from './gpu_shaders.js';

const TILE_COMPOSITE_RESOLVE_VERTEX_SHADER = `#version 300 es
precision highp float;

out vec2 vUv;

void main() {
  vec2 pos;
  if (gl_VertexID == 0) {
    pos = vec2(-1.0, -1.0);
  } else if (gl_VertexID == 1) {
    pos = vec2(3.0, -1.0);
  } else {
    pos = vec2(-1.0, 3.0);
  }
  vUv = pos * 0.5 + 0.5;
  gl_Position = vec4(pos, 0.0, 1.0);
}
`;

const TILE_COMPOSITE_RESOLVE_FRAGMENT_SHADER = `#version 300 es
precision highp float;

uniform sampler2D uAccumTexture;
uniform vec3 uBackgroundRgb;

in vec2 vUv;
out vec4 outColor;

void main() {
  vec4 accum = texture(uAccumTexture, vUv);
  vec3 resolved = accum.rgb + (1.0 - accum.a) * uBackgroundRgb;
  outColor = vec4(resolved, 1.0);
}
`;

function ensureTileCompositeState(gl, gpu) {
  if (gpu.tileCompositeDrawState?.program && gpu.tileCompositeDrawState?.vao && gpu.tileCompositeDrawState?.uploadState) {
    return gpu.tileCompositeDrawState;
  }

  const program = createProgram(gl, GPU_STEP_VERTEX_SHADER, GPU_STEP_FRAGMENT_SHADER_FRONT_TO_BACK);
  const vao = gl.createVertexArray();
  const uploadState = createPackedUploadState(gl);
  const layout = getPackedInterleavedAttribDescriptors();
  const uniformViewportPx = gl.getUniformLocation(program, 'uViewportPx');

  gpu.tileCompositeDrawState = {
    program,
    vao,
    uploadState,
    layout,
    uniformViewportPx
  };
  return gpu.tileCompositeDrawState;
}

function ensureTileCompositeResolveState(gl, gpu) {
  if (gpu.tileCompositeResolveState?.program && gpu.tileCompositeResolveState?.vao) {
    return gpu.tileCompositeResolveState;
  }

  const program = createProgram(
    gl,
    TILE_COMPOSITE_RESOLVE_VERTEX_SHADER,
    TILE_COMPOSITE_RESOLVE_FRAGMENT_SHADER
  );
  const vao = gl.createVertexArray();

  gpu.tileCompositeResolveState = {
    program,
    vao,
    uniformAccumTexture: gl.getUniformLocation(program, 'uAccumTexture'),
    uniformBackgroundRgb: gl.getUniformLocation(program, 'uBackgroundRgb')
  };
  return gpu.tileCompositeResolveState;
}

function ensureTileCompositeAccumulationTarget(gl, gpu, width, height) {
  const safeWidth = Math.max(1, width | 0);
  const safeHeight = Math.max(1, height | 0);
  const current = gpu.tileCompositeAccumulationTarget;

  if (
    current?.framebuffer &&
    current?.texture &&
    current.width === safeWidth &&
    current.height === safeHeight
  ) {
    return current;
  }

  if (current?.framebuffer) gl.deleteFramebuffer(current.framebuffer);
  if (current?.texture) gl.deleteTexture(current.texture);

  const texture = gl.createTexture();
  const framebuffer = gl.createFramebuffer();

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    safeWidth,
    safeHeight,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    null
  );

  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(
    gl.FRAMEBUFFER,
    gl.COLOR_ATTACHMENT0,
    gl.TEXTURE_2D,
    texture,
    0
  );

  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);

  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error(`Tile composite accumulation framebuffer incomplete (${status})`);
  }

  const target = {
    framebuffer,
    texture,
    width: safeWidth,
    height: safeHeight
  };
  gpu.tileCompositeAccumulationTarget = target;
  return target;
}

function configureTileCompositeVao(gl, state) {
  bindInterleavedFloatAttribs(gl, {
    vao: state.vao,
    program: state.program,
    buffer: state.uploadState.interleaved.buffer,
    attributes: state.layout.attributes
  });
}

function enableTileScissor(gl, canvasHeight, rect) {
  const [x0, y0, x1, y1] = rect;
  const width = Math.max(0, x1 - x0);
  const height = Math.max(0, y1 - y0);
  const scissorY = canvasHeight - y1;
  gl.scissor(x0, scissorY, width, height);
}

function clearTileCompositeAccumulationTarget(gl, target) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
  gl.viewport(0, 0, target.width, target.height);
  gl.disable(gl.SCISSOR_TEST);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
}

function resolveTileCompositeToCanvas(gl, gpu, target, canvasWidth, canvasHeight, bgGray01) {
  const state = ensureTileCompositeResolveState(gl, gpu);
  const bg = Number.isFinite(bgGray01) ? Math.min(1, Math.max(0, bgGray01)) : 0;

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvasWidth, canvasHeight);
  gl.disable(gl.SCISSOR_TEST);
  gl.disable(gl.BLEND);
  gl.useProgram(state.program);
  gl.bindVertexArray(state.vao);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, target.texture);
  gl.uniform1i(state.uniformAccumTexture, 0);
  gl.uniform3f(state.uniformBackgroundRgb, bg, bg, bg);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.bindVertexArray(null);
}

function buildUploadSummary(state, aggregate) {
  return {
    packedUploadLayoutVersion: state.layout?.layoutVersion ?? 0,
    packedUploadStrideBytes: state.layout?.strideBytes ?? 0,
    packedUploadBytes: aggregate.uploadBytes,
    packedUploadCount: aggregate.uploadCountItems,
    packedUploadLength: aggregate.uploadFloatLength,
    packedUploadCapacityBytes: state.uploadState?.interleaved?.capacityBytes ?? 0,
    packedUploadReusedCapacity: !!aggregate.reusedCapacity,
    packedUploadManagedCapacityReused: !!state.uploadState?.interleaved?.capacityReused,
    packedUploadManagedCapacityGrown: !!state.uploadState?.interleaved?.capacityGrown,
    packedUploadManagedUploadCount: state.uploadState?.interleaved?.uploadCount ?? 0,
    packedUploadAlphaSource: 'colorAlpha[3]',
    packedDirectDraw: true,
    packedDirectUsesGpuResidentPayload: false,
    packedDirectLayoutVersion: state.layout?.layoutVersion ?? 0,
    packedDirectStrideBytes: state.layout?.strideBytes ?? 0,
    packedDirectAttributeCount: Array.isArray(state.layout?.attributes) ? state.layout.attributes.length : 0,
    packedDirectOffsets: Array.isArray(state.layout?.attributes)
      ? state.layout.attributes.map((attr) => `${attr.name}:${attr.offset}`).join(', ')
      : '',
    packedInterleavedBound: true,
    packedDirectTileComposite: true,
    packedDirectCompositingContract: 'tile-local-front-to-back'
  };
}

export function executeTileCompositeDraw({
  gl,
  gpu,
  canvas,
  tileCompositePlan,
  drawPathSelection,
  bgGray01 = 0
}) {
  const batches = Array.isArray(tileCompositePlan?.batches) ? tileCompositePlan.batches : [];
  const state = ensureTileCompositeState(gl, gpu);
  const accumulationTarget = ensureTileCompositeAccumulationTarget(gl, gpu, canvas.width, canvas.height);
  const previousScissorEnabled = gl.isEnabled(gl.SCISSOR_TEST);
  const previousBlendEnabled = gl.isEnabled(gl.BLEND);
  const previousFramebuffer = gl.getParameter(gl.FRAMEBUFFER_BINDING);
  const aggregate = {
    uploadBytes: 0,
    uploadCountItems: 0,
    uploadFloatLength: 0,
    reusedCapacity: false
  };

  let drawCallCount = 0;
  let nonEmptyTileBatchCount = 0;
  let totalTileDrawCount = 0;

  clearTileCompositeAccumulationTarget(gl, accumulationTarget);
  enableFrontToBackAlphaBlend(gl);
  gl.enable(gl.SCISSOR_TEST);

  try {
    for (const batch of batches) {
      const packedCount = Number.isFinite(batch?.packedCount) ? Math.max(0, batch.packedCount | 0) : 0;
      if (!(batch?.packed instanceof Float32Array) || packedCount <= 0) continue;

      uploadPackedInterleaved(gl, state.uploadState, batch.packed, packedCount);
      configureTileCompositeVao(gl, state);

      gl.useProgram(state.program);
      gl.bindVertexArray(state.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, state.uploadState.interleaved.buffer);
      gl.uniform2f(state.uniformViewportPx, canvas.width, canvas.height);
      enableTileScissor(gl, canvas.height, batch.rect);
      gl.drawArrays(gl.POINTS, 0, packedCount);

      aggregate.uploadBytes += batch.packed.byteLength;
      aggregate.uploadCountItems += packedCount;
      aggregate.uploadFloatLength += batch.packed.length;
      aggregate.reusedCapacity = aggregate.reusedCapacity || !!state.uploadState?.reusedCapacity;
      drawCallCount++;
      nonEmptyTileBatchCount++;
      totalTileDrawCount += packedCount;
    }

    resolveTileCompositeToCanvas(gl, gpu, accumulationTarget, canvas.width, canvas.height, bgGray01);
  } finally {
    gl.bindFramebuffer(gl.FRAMEBUFFER, previousFramebuffer);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    if (previousScissorEnabled) gl.enable(gl.SCISSOR_TEST);
    else gl.disable(gl.SCISSOR_TEST);
    if (previousBlendEnabled) gl.enable(gl.BLEND);
    else gl.disable(gl.BLEND);
  }

  const uploadSummary = buildUploadSummary(state, aggregate);

  return {
    executionSummary: {
      tileBatchCount: batches.length,
      nonEmptyTileBatchCount,
      totalTileDrawCount,
      uploadCount: nonEmptyTileBatchCount,
      drawCallCount,
      requestedDrawPath: drawPathSelection?.requestedPath ?? 'packed',
      actualDrawPath: drawPathSelection?.actualPath ?? 'packed',
      drawPathFallbackReason: drawPathSelection?.fallbackReason ?? 'none',
      compositingContract: 'tile-local-front-to-back'
    },
    packedUploadSummary: uploadSummary,
    tileCompositeDrawInfo: {
      drawCount: totalTileDrawCount,
      drawCallCount,
      uploadCount: nonEmptyTileBatchCount
    }
  };
}
