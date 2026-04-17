import { createProgram } from './gpu_gl_utils.js';
import { GPU_STEP_FRAGMENT_SHADER } from './gpu_shaders.js';

const GPU_PACKED_PAYLOAD_WIDTH = 4;
const GPU_MERGE_MIN_PAYLOAD_COUNT = 4;
const GPU_MERGE_MIN_TOTAL_ROWS = 128;

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
      mergeTextureCapacityWidth: 0,
      mergeTextureCapacityHeight: 0,
      mergeRowsPerColumn: 0,
      mergeColumnCount: 0,
      lastMergeSucceeded: false,
      lastMergeFailureReason: 'none',
      lastMergeRowCount: 0,
      lastMergePayloadCount: 0,
      lastMergeAtlasReused: false,
      lastMergeAtlasRebuilt: false,
      lastMergeAtlasChurnReason: 'none'
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

function buildMergePolicyDecision(gl, resources, payloads, totalCount) {
  const payloadCount = Array.isArray(payloads) ? payloads.length : 0;
  const dispatchSavings = Math.max(0, payloadCount - 1);
  const estimatedCopyCount = payloadCount;
  const layout = buildMergeAtlasLayout(gl, totalCount);
  const atlasArea = layout.ok ? layout.textureWidth * layout.textureHeight : 0;
  const copyPerSavedDispatch = dispatchSavings > 0
    ? estimatedCopyCount / dispatchSavings
    : estimatedCopyCount;
  const previousMergeSucceeded = !!resources?.lastMergeSucceeded;
  const previousMergeFailureReason = resources?.lastMergeFailureReason ?? 'none';

  if (payloadCount <= 1) {
    return {
      shouldMerge: false,
      policySelectedPath: 'single-payload',
      policyReason: 'merge-single-payload-not-needed',
      estimatedDispatchSavings: dispatchSavings,
      estimatedCopyCount,
      atlasArea,
      layout
    };
  }

  if (totalCount <= 0) {
    return {
      shouldMerge: false,
      policySelectedPath: 'multi-payload',
      policyReason: 'merge-no-rows',
      estimatedDispatchSavings: dispatchSavings,
      estimatedCopyCount,
      atlasArea,
      layout
    };
  }

  if (!layout.ok) {
    return {
      shouldMerge: false,
      policySelectedPath: 'multi-payload',
      policyReason: layout.failureReason ?? 'merge-layout-invalid',
      estimatedDispatchSavings: dispatchSavings,
      estimatedCopyCount,
      atlasArea,
      layout
    };
  }

  if (payloadCount < GPU_MERGE_MIN_PAYLOAD_COUNT && totalCount < GPU_MERGE_MIN_TOTAL_ROWS) {
    return {
      shouldMerge: false,
      policySelectedPath: 'multi-payload',
      policyReason: 'merge-policy-small-workload',
      estimatedDispatchSavings: dispatchSavings,
      estimatedCopyCount,
      atlasArea,
      layout
    };
  }

  if (copyPerSavedDispatch > 12 && !previousMergeSucceeded && totalCount < (GPU_MERGE_MIN_TOTAL_ROWS * 4)) {
    return {
      shouldMerge: false,
      policySelectedPath: 'multi-payload',
      policyReason: 'merge-policy-copy-cost-dominates',
      estimatedDispatchSavings: dispatchSavings,
      estimatedCopyCount,
      atlasArea,
      layout
    };
  }

  if (previousMergeSucceeded) {
    return {
      shouldMerge: true,
      policySelectedPath: 'merged-atlas',
      policyReason: 'merge-policy-hold-success',
      estimatedDispatchSavings: dispatchSavings,
      estimatedCopyCount,
      atlasArea,
      layout
    };
  }

  if (payloadCount >= 16 || totalCount >= 1024 || dispatchSavings >= 8) {
    return {
      shouldMerge: true,
      policySelectedPath: 'merged-atlas',
      policyReason: 'merge-policy-dispatch-savings-favored',
      estimatedDispatchSavings: dispatchSavings,
      estimatedCopyCount,
      atlasArea,
      layout
    };
  }

  return {
    shouldMerge: true,
    policySelectedPath: 'merged-atlas',
    policyReason: previousMergeFailureReason !== 'none'
      ? 'merge-policy-retry-after-failure'
      : 'merge-policy-balanced-atlas',
    estimatedDispatchSavings: dispatchSavings,
    estimatedCopyCount,
    atlasArea,
    layout
  };
}

function chooseMergeTextureCapacity(currentSize, requestedSize, maxTextureSize) {
  if (requestedSize <= 0) return 0;
  if (currentSize >= requestedSize) return currentSize;
  let nextSize = Math.max(requestedSize, currentSize > 0 ? currentSize : 1);
  while (nextSize < requestedSize) {
    nextSize = Math.min(maxTextureSize, Math.max(nextSize + 1, nextSize * 2));
    if (nextSize === maxTextureSize) break;
  }
  return Math.min(maxTextureSize, Math.max(requestedSize, nextSize));
}

function ensureMergedPackedPayloadTexture(gl, resources, layout) {
  if (!resources?.mergeTexture || !resources?.drawFramebuffer || !layout?.ok) {
    return {
      ok: false,
      failureReason: layout?.failureReason ?? 'merge-layout-invalid',
      atlasReused: false,
      atlasRebuilt: false,
      atlasChurnReason: layout?.failureReason ?? 'merge-layout-invalid',
      atlasCapacityWidth: resources?.mergeTextureCapacityWidth ?? 0,
      atlasCapacityHeight: resources?.mergeTextureCapacityHeight ?? 0,
      atlasAllocationBytes: 0,
      atlasSavedAllocationBytes: 0
    };
  }

  const currentCapacityWidth = resources.mergeTextureCapacityWidth ?? 0;
  const currentCapacityHeight = resources.mergeTextureCapacityHeight ?? 0;
  const canReuseExistingAtlas =
    currentCapacityWidth >= layout.textureWidth &&
    currentCapacityHeight >= layout.textureHeight;

  if (canReuseExistingAtlas) {
    resources.mergeTextureWidth = layout.textureWidth;
    resources.mergeTextureHeight = layout.textureHeight;
    resources.mergeRowsPerColumn = layout.rowsPerColumn;
    resources.mergeColumnCount = layout.columnCount;
    return {
      ok: true,
      failureReason: 'none',
      atlasReused: true,
      atlasRebuilt: false,
      atlasChurnReason: 'atlas-reuse-capacity',
      atlasCapacityWidth: currentCapacityWidth,
      atlasCapacityHeight: currentCapacityHeight,
      atlasAllocationBytes: 0,
      atlasSavedAllocationBytes: layout.textureWidth * layout.textureHeight * 16
    };
  }

  const targetWidth = chooseMergeTextureCapacity(
    currentCapacityWidth,
    layout.textureWidth,
    layout.maxTextureSize ?? layout.textureWidth
  );
  const targetHeight = chooseMergeTextureCapacity(
    currentCapacityHeight,
    layout.textureHeight,
    layout.maxTextureSize ?? layout.textureHeight
  );

  if (targetWidth < layout.textureWidth || targetHeight < layout.textureHeight) {
    return {
      ok: false,
      failureReason: 'merge-atlas-capacity-growth-failed',
      atlasReused: false,
      atlasRebuilt: false,
      atlasChurnReason: 'atlas-growth-failed',
      atlasCapacityWidth: currentCapacityWidth,
      atlasCapacityHeight: currentCapacityHeight,
      atlasAllocationBytes: 0,
      atlasSavedAllocationBytes: 0
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
    targetWidth,
    targetHeight,
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
      failureReason: `merge-draw-framebuffer-incomplete-${status}`,
      atlasReused: false,
      atlasRebuilt: false,
      atlasChurnReason: 'atlas-framebuffer-incomplete',
      atlasCapacityWidth: currentCapacityWidth,
      atlasCapacityHeight: currentCapacityHeight,
      atlasAllocationBytes: 0,
      atlasSavedAllocationBytes: 0
    };
  }
  resources.mergeTextureWidth = layout.textureWidth;
  resources.mergeTextureHeight = layout.textureHeight;
  resources.mergeTextureCapacityWidth = targetWidth;
  resources.mergeTextureCapacityHeight = targetHeight;
  resources.mergeRowsPerColumn = layout.rowsPerColumn;
  resources.mergeColumnCount = layout.columnCount;
  return {
    ok: true,
    failureReason: 'none',
    atlasReused: false,
    atlasRebuilt: true,
    atlasChurnReason: currentCapacityWidth > 0 || currentCapacityHeight > 0
      ? 'atlas-grow-capacity'
      : 'atlas-initial-allocation',
    atlasCapacityWidth: targetWidth,
    atlasCapacityHeight: targetHeight,
    atlasAllocationBytes: targetWidth * targetHeight * 16,
    atlasSavedAllocationBytes: 0
  };
}

function mergeGpuPackedPayloads(gl, resources, payloads, totalCount, layout) {
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
      columnCount: layout.columnCount ?? 0,
      atlasReused: !!textureState.atlasReused,
      atlasRebuilt: !!textureState.atlasRebuilt,
      atlasChurnReason: textureState.atlasChurnReason ?? 'merge-resources-unavailable',
      atlasCapacityWidth: textureState.atlasCapacityWidth ?? 0,
      atlasCapacityHeight: textureState.atlasCapacityHeight ?? 0,
      atlasAllocationBytes: textureState.atlasAllocationBytes ?? 0,
      atlasSavedAllocationBytes: textureState.atlasSavedAllocationBytes ?? 0
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
          columnCount: layout.columnCount,
          atlasReused: !!textureState.atlasReused,
          atlasRebuilt: !!textureState.atlasRebuilt,
          atlasChurnReason: textureState.atlasChurnReason ?? failureReason,
          atlasCapacityWidth: textureState.atlasCapacityWidth ?? 0,
          atlasCapacityHeight: textureState.atlasCapacityHeight ?? 0,
          atlasAllocationBytes: textureState.atlasAllocationBytes ?? 0,
          atlasSavedAllocationBytes: textureState.atlasSavedAllocationBytes ?? 0
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
      columnCount: layout.columnCount,
      atlasReused: !!textureState.atlasReused,
      atlasRebuilt: !!textureState.atlasRebuilt,
      atlasChurnReason: textureState.atlasChurnReason ?? failureReason,
      atlasCapacityWidth: textureState.atlasCapacityWidth ?? 0,
      atlasCapacityHeight: textureState.atlasCapacityHeight ?? 0,
      atlasAllocationBytes: textureState.atlasAllocationBytes ?? 0,
      atlasSavedAllocationBytes: textureState.atlasSavedAllocationBytes ?? 0
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
    columnCount: layout.columnCount,
    atlasReused: !!textureState.atlasReused,
    atlasRebuilt: !!textureState.atlasRebuilt,
    atlasChurnReason: textureState.atlasChurnReason ?? 'none',
    atlasCapacityWidth: textureState.atlasCapacityWidth ?? 0,
    atlasCapacityHeight: textureState.atlasCapacityHeight ?? 0,
    atlasAllocationBytes: textureState.atlasAllocationBytes ?? 0,
    atlasSavedAllocationBytes: textureState.atlasSavedAllocationBytes ?? 0
  };
}

export function drawGpuPackedPayloads(gl, gpu, screenSpace, canvasWidth, canvasHeight, options = {}) {
  const payloads = getValidGpuPackedPayloads(gl, screenSpace);
  if (payloads.length <= 0) return null;

  try {
    const resources = options.resources || ensureGpuPackedPayloadTextureDrawResources(gl, gpu, options.storageKey);
    if (!resources?.program || !resources?.vao) return null;

    const totalCount = getPayloadDrawCount(payloads);
    const mergePolicy = buildMergePolicyDecision(gl, resources, payloads, totalCount);
    const mergeResult = mergePolicy.shouldMerge
      ? mergeGpuPackedPayloads(gl, resources, payloads, totalCount, mergePolicy.layout)
      : {
          attempted: false,
          merged: false,
          copyCount: 0,
          failureReason: mergePolicy.policyReason,
          textureWidth: mergePolicy.layout?.textureWidth ?? 0,
          textureHeight: mergePolicy.layout?.textureHeight ?? 0,
          rowCount: totalCount,
          rowsPerColumn: mergePolicy.layout?.rowsPerColumn ?? 0,
          columnCount: mergePolicy.layout?.columnCount ?? 0
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

    resources.lastMergeSucceeded = !!mergeResult.merged;
    resources.lastMergeFailureReason = mergeResult.failureReason ?? 'none';
    resources.lastMergeRowCount = mergeResult.rowCount ?? totalCount;
    resources.lastMergePayloadCount = payloads.length;
    resources.lastMergeAtlasReused = !!mergeResult.atlasReused;
    resources.lastMergeAtlasRebuilt = !!mergeResult.atlasRebuilt;
    resources.lastMergeAtlasChurnReason = mergeResult.atlasChurnReason ?? 'none';

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
      mergeColumnCount: mergeResult.columnCount ?? 0,
      mergePolicySelectedPath: mergeResult.merged
        ? 'merged-atlas'
        : (
          mergeResult.attempted
            ? 'multi-payload-fallback'
            : mergePolicy.policySelectedPath
        ),
      mergePolicyReason: mergeResult.merged
        ? mergePolicy.policyReason
        : (
          mergeResult.attempted
            ? `merge-policy-fallback:${mergeResult.failureReason ?? 'unknown'}`
            : mergePolicy.policyReason
        ),
      mergePolicyEstimatedCopyCount: mergePolicy.estimatedCopyCount ?? payloads.length,
      mergePolicyEstimatedDispatchSavings: mergePolicy.estimatedDispatchSavings ?? Math.max(0, payloads.length - 1),
      mergePolicyAtlasArea: mergePolicy.atlasArea ?? 0,
      mergeAtlasReused: !!mergeResult.atlasReused,
      mergeAtlasRebuilt: !!mergeResult.atlasRebuilt,
      mergeAtlasChurnReason: mergeResult.atlasChurnReason ?? 'none',
      mergeAtlasCapacityWidth: mergeResult.atlasCapacityWidth ?? 0,
      mergeAtlasCapacityHeight: mergeResult.atlasCapacityHeight ?? 0,
      mergeAtlasAllocationBytes: mergeResult.atlasAllocationBytes ?? 0,
      mergeAtlasSavedAllocationBytes: mergeResult.atlasSavedAllocationBytes ?? 0
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
