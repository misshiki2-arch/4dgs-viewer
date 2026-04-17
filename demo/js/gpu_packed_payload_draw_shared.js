import { createProgram } from './gpu_gl_utils.js';
import { GPU_STEP_FRAGMENT_SHADER } from './gpu_shaders.js';

const GPU_PACKED_PAYLOAD_WIDTH = 4;

const GPU_PACKED_TEXTURE_VERTEX_SHADER = `#version 300 es
precision highp float;

uniform sampler2D uPackedTexture;
uniform vec2 uViewportPx;
uniform int uPackedRowsPerColumn;

out vec4 vColorAlpha;
out float vRadiusPx;
out vec3 vConic;

void main() {
  int itemIndex = gl_VertexID;
  int rowsPerColumn = max(1, uPackedRowsPerColumn);
  int columnIndex = itemIndex / rowsPerColumn;
  int rowIndex = itemIndex - (columnIndex * rowsPerColumn);
  int xBase = columnIndex * 4;
  vec4 row0 = texelFetch(uPackedTexture, ivec2(xBase + 0, rowIndex), 0);
  vec4 row1 = texelFetch(uPackedTexture, ivec2(xBase + 1, rowIndex), 0);
  vec4 row2 = texelFetch(uPackedTexture, ivec2(xBase + 2, rowIndex), 0);

  float x = (row0.x / uViewportPx.x) * 2.0 - 1.0;
  float y = 1.0 - (row0.y / uViewportPx.y) * 2.0;
  gl_Position = vec4(x, y, 0.0, 1.0);
  gl_PointSize = max(1.0, row0.z * 2.0);

  vColorAlpha = row1;
  vRadiusPx = row0.z;
  vConic = row2.xyz;
}
`;

export function getValidGpuPackedPayloads(gl, screenSpace) {
  const payloads = Array.isArray(screenSpace?.gpuPackedPayloads) ? screenSpace.gpuPackedPayloads : [];
  return payloads.filter((payload) => payload?.texture && payload?.gl === gl);
}

export function ensureGpuPackedPayloadTextureDrawResources(gl, gpu, storageKey) {
  const key = storageKey || 'gpuPackedPayloadTextureDrawResources';
  if (
    gpu[key]?.program &&
    gpu[key]?.vao &&
    gpu[key]?.mergeTexture &&
    gpu[key]?.readFramebuffer &&
    gpu[key]?.drawFramebuffer
  ) {
    return gpu[key];
  }

  try {
    const program = createProgram(gl, GPU_PACKED_TEXTURE_VERTEX_SHADER, GPU_STEP_FRAGMENT_SHADER);
    const vao = gl.createVertexArray();
    const resources = {
      program,
      vao,
      uniformViewportPx: gl.getUniformLocation(program, 'uViewportPx'),
      uniformPackedTexture: gl.getUniformLocation(program, 'uPackedTexture'),
      uniformPackedRowsPerColumn: gl.getUniformLocation(program, 'uPackedRowsPerColumn'),
      mergeTexture: gl.createTexture(),
      readFramebuffer: gl.createFramebuffer(),
      drawFramebuffer: gl.createFramebuffer(),
      mergeTextureWidth: 0,
      mergeTextureHeight: 0,
      mergeRowsPerColumn: 0,
      mergeColumnCount: 0
    };
    gpu[key] = resources;
    return resources;
  } catch (_error) {
    return null;
  }
}

function getPayloadDrawCount(payloads) {
  let totalCount = 0;
  for (const payload of payloads) {
    const count = Number.isFinite(payload?.count) ? Math.max(0, payload.count | 0) : 0;
    totalCount += count;
  }
  return totalCount;
}

function buildMergeAtlasLayout(gl, totalCount) {
  const maxTextureSize = Number.isFinite(gl?.getParameter?.(gl.MAX_TEXTURE_SIZE))
    ? Math.max(0, gl.getParameter(gl.MAX_TEXTURE_SIZE) | 0)
    : 0;
  if (totalCount <= 0) {
    return {
      ok: false,
      failureReason: 'merge-no-rows',
      textureWidth: 0,
      textureHeight: 0,
      rowsPerColumn: 0,
      columnCount: 0,
      maxTextureSize
    };
  }
  if (maxTextureSize < GPU_PACKED_PAYLOAD_WIDTH) {
    return {
      ok: false,
      failureReason: 'merge-max-texture-too-small',
      textureWidth: 0,
      textureHeight: 0,
      rowsPerColumn: 0,
      columnCount: 0,
      maxTextureSize
    };
  }

  const maxColumns = Math.max(1, Math.floor(maxTextureSize / GPU_PACKED_PAYLOAD_WIDTH));
  const rowsPerColumn = Math.max(1, Math.min(maxTextureSize, Math.ceil(totalCount / maxColumns)));
  const columnCount = Math.max(1, Math.ceil(totalCount / rowsPerColumn));
  const textureWidth = columnCount * GPU_PACKED_PAYLOAD_WIDTH;
  const textureHeight = rowsPerColumn;

  if (textureWidth > maxTextureSize || textureHeight > maxTextureSize) {
    return {
      ok: false,
      failureReason: 'merge-atlas-exceeds-max-texture-size',
      textureWidth,
      textureHeight,
      rowsPerColumn,
      columnCount,
      maxTextureSize
    };
  }

  return {
    ok: true,
    failureReason: 'none',
    textureWidth,
    textureHeight,
    rowsPerColumn,
    columnCount,
    maxTextureSize
  };
}

function ensureMergedPackedPayloadTexture(gl, resources, layout) {
  if (!resources?.mergeTexture || !resources?.drawFramebuffer || !layout?.ok) {
    return {
      ok: false,
      failureReason: layout?.failureReason ?? 'merge-layout-invalid'
    };
  }
  if (
    resources.mergeTextureWidth === layout.textureWidth &&
    resources.mergeTextureHeight === layout.textureHeight &&
    resources.mergeRowsPerColumn === layout.rowsPerColumn &&
    resources.mergeColumnCount === layout.columnCount
  ) {
    return {
      ok: true,
      failureReason: 'none'
    };
  }

  gl.bindTexture(gl.TEXTURE_2D, resources.mergeTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA32F,
    layout.textureWidth,
    layout.textureHeight,
    0,
    gl.RGBA,
    gl.FLOAT,
    null
  );
  gl.bindFramebuffer(gl.FRAMEBUFFER, resources.drawFramebuffer);
  gl.framebufferTexture2D(
    gl.FRAMEBUFFER,
    gl.COLOR_ATTACHMENT0,
    gl.TEXTURE_2D,
    resources.mergeTexture,
    0
  );
  if (typeof gl.drawBuffers === 'function') {
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
  }
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    return {
      ok: false,
      failureReason: `merge-draw-framebuffer-incomplete-${status}`
    };
  }
  resources.mergeTextureWidth = layout.textureWidth;
  resources.mergeTextureHeight = layout.textureHeight;
  resources.mergeRowsPerColumn = layout.rowsPerColumn;
  resources.mergeColumnCount = layout.columnCount;
  return {
    ok: true,
    failureReason: 'none'
  };
}

function mergeGpuPackedPayloads(gl, resources, payloads, totalCount) {
  const layout = buildMergeAtlasLayout(gl, totalCount);
  const textureState = ensureMergedPackedPayloadTexture(gl, resources, layout);
  if (!resources?.readFramebuffer || !resources?.drawFramebuffer || !layout.ok || !textureState.ok) {
    return {
      attempted: true,
      merged: false,
      copyCount: 0,
      failureReason: textureState.failureReason ?? layout.failureReason ?? 'merge-resources-unavailable',
      textureWidth: layout.textureWidth ?? 0,
      textureHeight: layout.textureHeight ?? 0,
      rowCount: totalCount,
      rowsPerColumn: layout.rowsPerColumn ?? 0,
      columnCount: layout.columnCount ?? 0
    };
  }

  let copyCount = 0;
  let itemOffset = 0;
  const previousViewport = gl.getParameter(gl.VIEWPORT);
  const blendEnabled = gl.isEnabled(gl.BLEND);
  const depthTestEnabled = gl.isEnabled(gl.DEPTH_TEST);
  const scissorTestEnabled = gl.isEnabled(gl.SCISSOR_TEST);
  let failureReason = 'none';

  try {
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, resources.drawFramebuffer);
    if (typeof gl.drawBuffers === 'function') {
      gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    }
    if (gl.checkFramebufferStatus(gl.DRAW_FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      return { merged: false, copyCount: 0 };
    }

    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.SCISSOR_TEST);
    gl.viewport(0, 0, layout.textureWidth, layout.textureHeight);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    for (const payload of payloads) {
      const count = Number.isFinite(payload?.count) ? Math.max(0, payload.count | 0) : 0;
      if (count <= 0 || !payload?.texture) continue;

      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, resources.readFramebuffer);
      gl.framebufferTexture2D(
        gl.READ_FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        payload.texture,
        0
      );
      if (typeof gl.readBuffer === 'function') {
        gl.readBuffer(gl.COLOR_ATTACHMENT0);
      }
      if (gl.checkFramebufferStatus(gl.READ_FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        failureReason = 'merge-read-framebuffer-incomplete';
        return {
          attempted: true,
          merged: false,
          copyCount: 0,
          failureReason,
          textureWidth: layout.textureWidth,
          textureHeight: layout.textureHeight,
          rowCount: totalCount,
          rowsPerColumn: layout.rowsPerColumn,
          columnCount: layout.columnCount
        };
      }
      let payloadOffset = 0;
      while (payloadOffset < count) {
        const absoluteItemOffset = itemOffset + payloadOffset;
        const destColumn = Math.floor(absoluteItemOffset / layout.rowsPerColumn);
        const destRow = absoluteItemOffset % layout.rowsPerColumn;
        const remainingInColumn = layout.rowsPerColumn - destRow;
        const chunkCount = Math.min(remainingInColumn, count - payloadOffset);
        const destX0 = destColumn * GPU_PACKED_PAYLOAD_WIDTH;
        const destY0 = destRow;
        gl.blitFramebuffer(
          0,
          payloadOffset,
          GPU_PACKED_PAYLOAD_WIDTH,
          payloadOffset + chunkCount,
          destX0,
          destY0,
          destX0 + GPU_PACKED_PAYLOAD_WIDTH,
          destY0 + chunkCount,
          gl.COLOR_BUFFER_BIT,
          gl.NEAREST
        );
        payloadOffset += chunkCount;
      }
      itemOffset += count;
      copyCount++;
    }
  } catch (_error) {
    failureReason = 'merge-blit-exception';
    return {
      attempted: true,
      merged: false,
      copyCount: 0,
      failureReason,
      textureWidth: layout.textureWidth,
      textureHeight: layout.textureHeight,
      rowCount: totalCount,
      rowsPerColumn: layout.rowsPerColumn,
      columnCount: layout.columnCount
    };
  } finally {
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    gl.viewport(
      previousViewport[0],
      previousViewport[1],
      previousViewport[2],
      previousViewport[3]
    );
    if (blendEnabled) gl.enable(gl.BLEND);
    else gl.disable(gl.BLEND);
    if (depthTestEnabled) gl.enable(gl.DEPTH_TEST);
    else gl.disable(gl.DEPTH_TEST);
    if (scissorTestEnabled) gl.enable(gl.SCISSOR_TEST);
    else gl.disable(gl.SCISSOR_TEST);
  }

  return {
    attempted: true,
    merged: copyCount > 0 && itemOffset === totalCount,
    copyCount,
    failureReason: copyCount > 0 && itemOffset === totalCount ? 'none' : (failureReason || 'merge-row-layout-mismatch'),
    textureWidth: layout.textureWidth,
    textureHeight: layout.textureHeight,
    rowCount: totalCount,
    rowsPerColumn: layout.rowsPerColumn,
    columnCount: layout.columnCount
  };
}

export function drawGpuPackedPayloads(gl, gpu, screenSpace, canvasWidth, canvasHeight, options = {}) {
  const payloads = getValidGpuPackedPayloads(gl, screenSpace);
  if (payloads.length <= 0) return null;

  try {
    const resources = options.resources || ensureGpuPackedPayloadTextureDrawResources(gl, gpu, options.storageKey);
    if (!resources?.program || !resources?.vao) return null;

    const totalCount = getPayloadDrawCount(payloads);
    const mergeResult = payloads.length > 1 && totalCount > 0
      ? mergeGpuPackedPayloads(gl, resources, payloads, totalCount)
      : {
          attempted: false,
          merged: false,
          copyCount: 0,
          failureReason: payloads.length > 1 ? 'merge-not-attempted' : 'merge-single-payload-not-needed',
          textureWidth: 0,
          textureHeight: 0,
          rowCount: totalCount,
          rowsPerColumn: 0,
          columnCount: 0
        };
    let drawCount = 0;
    let drawCallCount = 0;
    let bindCount = 0;

    gl.useProgram(resources.program);
    gl.bindVertexArray(resources.vao);
    gl.uniform2f(resources.uniformViewportPx, canvasWidth, canvasHeight);
    gl.uniform1i(resources.uniformPackedTexture, 0);

    if (mergeResult.merged) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, resources.mergeTexture);
      gl.uniform1i(resources.uniformPackedRowsPerColumn, Math.max(1, mergeResult.rowsPerColumn | 0));
      bindCount++;
      gl.drawArrays(gl.POINTS, 0, totalCount);
      drawCount = totalCount;
      drawCallCount = 1;
    } else {
      for (const payload of payloads) {
        const count = Number.isFinite(payload?.count) ? Math.max(0, payload.count | 0) : 0;
        if (count <= 0) continue;
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, payload.texture);
        gl.uniform1i(resources.uniformPackedRowsPerColumn, Math.max(1, count));
        bindCount++;
        gl.drawArrays(gl.POINTS, 0, count);
        drawCount += count;
        drawCallCount++;
      }
    }

    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindVertexArray(null);

    return {
      drawCount,
      drawCallCount,
      bindCount,
      setupCount: 1,
      dispatchCount: drawCallCount,
      dispatchMode: mergeResult.merged
        ? 'shared-texture-merged-payloads'
        : payloads.length > 1
          ? 'shared-texture-multi-payload'
          : 'shared-texture-single-payload',
      resources,
      payloadCount: payloads.length,
      mergeAttempted: !!mergeResult.attempted,
      mergeCopyCount: mergeResult.copyCount,
      mergeFailureReason: mergeResult.failureReason ?? 'none',
      mergeTextureWidth: mergeResult.textureWidth ?? 0,
      mergeTextureHeight: mergeResult.textureHeight ?? 0,
      mergeRowCount: mergeResult.rowCount ?? totalCount,
      mergeRowsPerColumn: mergeResult.rowsPerColumn ?? 0,
      mergeColumnCount: mergeResult.columnCount ?? 0
    };
  } catch (_error) {
    try {
      gl.bindTexture(gl.TEXTURE_2D, null);
      gl.bindVertexArray(null);
    } catch (_cleanupError) {
      // ignore cleanup failure in guarded fallback path
    }
    return null;
  }
}
