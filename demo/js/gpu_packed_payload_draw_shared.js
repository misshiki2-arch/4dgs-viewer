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
out vec2 vCenterPx;

void main() {
  int itemIndex = gl_VertexID;
  int rowsPerColumn = max(1, uPackedRowsPerColumn);
  int columnIndex = itemIndex / rowsPerColumn;
  int rowIndex = itemIndex - (columnIndex * rowsPerColumn);
  int xBase = columnIndex * 4;
  vec4 row0 = texelFetch(uPackedTexture, ivec2(xBase + 0, rowIndex), 0);
  vec4 row1 = texelFetch(uPackedTexture, ivec2(xBase + 1, rowIndex), 0);
  vec4 row2 = texelFetch(uPackedTexture, ivec2(xBase + 2, rowIndex), 0);

  float x = ((row0.x + 0.5) / uViewportPx.x) * 2.0 - 1.0;
  float y = 1.0 - ((row0.y + 0.5) / uViewportPx.y) * 2.0;
  gl_Position = vec4(x, y, 0.0, 1.0);
  gl_PointSize = max(1.0, row0.z * 2.0);

  vColorAlpha = row1;
  vRadiusPx = row0.z;
  vConic = row2.xyz;
  vCenterPx = row0.xy;
}
`;

function getPayloadRowsPerColumn(payload) {
  if (Number.isFinite(payload?.rowsPerColumn)) {
    return Math.max(1, payload.rowsPerColumn | 0);
  }
  if (Number.isFinite(payload?.height)) {
    return Math.max(1, payload.height | 0);
  }
  if (Number.isFinite(payload?.count)) {
    return Math.max(1, payload.count | 0);
  }
  return 1;
}

function isBackendAtlasPayload(payload) {
  return payload?.kind === 'gpu-packed-texture-atlas' ||
    (Number.isFinite(payload?.columnCount) && (payload.columnCount | 0) > 1);
}

export function getValidGpuPackedPayloads(gl, screenSpace) {
  const payloads = Array.isArray(screenSpace?.gpuPackedPayloads) ? screenSpace.gpuPackedPayloads : [];
  return payloads.filter((payload) => payload?.texture && payload?.gl === gl);
}

function buildPackedPayloadInspectionCandidates(gl, screenSpace) {
  const payloads = Array.isArray(screenSpace?.gpuPackedPayloads) ? screenSpace.gpuPackedPayloads : [];
  return payloads.map((payload, index) => {
    const hasTexture = !!payload?.texture;
    const hasGl = !!payload?.gl;
    const glMatches = !gl || !hasGl ? false : payload.gl === gl;
    const usable = hasTexture;
    return {
      index,
      payload,
      hasTexture,
      hasGl,
      glMatches,
      usable,
      kind: payload?.kind ?? 'gpu-packed-texture',
      count: Number.isFinite(payload?.count) ? Math.max(0, payload.count | 0) : 0,
      width: Number.isFinite(payload?.width) ? Math.max(0, payload.width | 0) : 0,
      height: Number.isFinite(payload?.height) ? Math.max(0, payload.height | 0) : 0
    };
  });
}

function resolvePayloadSelection(payloads, requestedIndex) {
  const absoluteIndex = Number.isFinite(requestedIndex) ? Math.max(0, requestedIndex | 0) : 0;
  let runningCount = 0;
  for (const payload of payloads) {
    const count = Number.isFinite(payload?.count) ? Math.max(0, payload.count | 0) : 0;
    if (absoluteIndex < runningCount + count) {
      return {
        ok: true,
        absoluteIndex,
        payload,
        payloadIndex: payloads.indexOf(payload),
        localIndex: absoluteIndex - runningCount,
        count
      };
    }
    runningCount += count;
  }
  return {
    ok: false,
    absoluteIndex,
    failureReason: runningCount <= 0 ? 'inspect-no-payload-items' : 'inspect-item-index-out-of-range',
    availableCount: runningCount
  };
}

function readPackedPayloadItemRows(gl, payload, localIndex) {
  const rowsPerColumn = getPayloadRowsPerColumn(payload);
  const columnIndex = Math.floor(localIndex / rowsPerColumn);
  const rowIndex = localIndex - (columnIndex * rowsPerColumn);
  const xBase = columnIndex * GPU_PACKED_PAYLOAD_WIDTH;
  const previousFramebuffer = gl.getParameter(gl.FRAMEBUFFER_BINDING);
  const framebuffer = gl.createFramebuffer();
  if (!framebuffer) {
    throw new Error('inspectActiveSplat failed: could not create framebuffer');
  }

  try {
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      payload.texture,
      0
    );
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`inspectActiveSplat failed: framebuffer incomplete (${status})`);
    }
    const values = new Float32Array(GPU_PACKED_PAYLOAD_WIDTH * 4);
    gl.readPixels(xBase, rowIndex, GPU_PACKED_PAYLOAD_WIDTH, 1, gl.RGBA, gl.FLOAT, values);
    return {
      localIndex,
      rowIndex,
      columnIndex,
      xBase,
      rowsPerColumn,
      values,
      row0: Array.from(values.subarray(0, 4)),
      row1: Array.from(values.subarray(4, 8)),
      row2: Array.from(values.subarray(8, 12)),
      row3: Array.from(values.subarray(12, 16))
    };
  } finally {
    gl.bindFramebuffer(gl.FRAMEBUFFER, previousFramebuffer);
    gl.deleteFramebuffer(framebuffer);
  }
}

function evaluatePackedFragmentSample(centerPx, radiusPx, conic, colorAlpha, pointCoord, pointSizePx) {
  const pointSize = Math.max(1.0, pointSizePx);
  const localPixelIndex = [
    Math.min(pointSize - 1.0, Math.max(0.0, Math.floor(pointCoord[0] * pointSize))),
    Math.min(pointSize - 1.0, Math.max(0.0, Math.floor(pointCoord[1] * pointSize)))
  ];
  const localPixelCenter = [
    localPixelIndex[0] + 0.5,
    localPixelIndex[1] + 0.5
  ];
  const spriteMinPx = [
    Math.floor(centerPx[0] - pointSize * 0.5),
    Math.floor(centerPx[1] - pointSize * 0.5)
  ];
  const pixelIndexPx = [
    spriteMinPx[0] + localPixelIndex[0],
    spriteMinPx[1] + localPixelIndex[1]
  ];
  const d = [
    pixelIndexPx[0] - centerPx[0],
    pixelIndexPx[1] - centerPx[1]
  ];
  const dx = d[0];
  const dy = d[1];
  const power =
    -0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) -
    conic[1] * dx * dy;
  const packedAlpha = colorAlpha[3];
  const gaussianAlpha = packedAlpha * Math.exp(power);
  const finalAlpha = Math.min(0.99, Math.max(0.0, gaussianAlpha));
  const discardedByPositivePower = power > 0.0;
  const discardedByAlphaCutoff = finalAlpha < (1.0 / 255.0);
  return {
    pointCoord,
    localPixelIndex,
    localPixelCenter,
    pixelIndexPx,
    d,
    power,
    packedAlpha,
    gaussianAlpha,
    finalAlpha,
    discardedByPositivePower,
    discardedByAlphaCutoff,
    survivesFragment:
      !discardedByPositivePower &&
      !discardedByAlphaCutoff
  };
}

export function inspectGpuPackedPayloadItem(gl, screenSpace, options = {}) {
  const payloadCandidates = buildPackedPayloadInspectionCandidates(gl, screenSpace);
  const payloads = payloadCandidates.filter((candidate) => candidate.usable).map((candidate) => candidate.payload);
  if (payloads.length <= 0) {
    return {
      ok: false,
      failureReason: 'inspect-no-valid-gpu-packed-payloads',
      payloadCandidateCount: payloadCandidates.length,
      payloadCandidates: payloadCandidates.map((candidate) => ({
        index: candidate.index,
        kind: candidate.kind,
        count: candidate.count,
        width: candidate.width,
        height: candidate.height,
        hasTexture: candidate.hasTexture,
        hasGl: candidate.hasGl,
        glMatches: candidate.glMatches,
        usable: candidate.usable
      }))
    };
  }

  const selection = resolvePayloadSelection(payloads, options.index ?? 0);
  if (!selection.ok) {
    return {
      ok: false,
      failureReason: selection.failureReason,
      requestedIndex: selection.absoluteIndex,
      availableCount: selection.availableCount ?? 0,
      payloadCandidateCount: payloadCandidates.length,
      payloadCandidates: payloadCandidates.map((candidate) => ({
        index: candidate.index,
        kind: candidate.kind,
        count: candidate.count,
        width: candidate.width,
        height: candidate.height,
        hasTexture: candidate.hasTexture,
        hasGl: candidate.hasGl,
        glMatches: candidate.glMatches,
        usable: candidate.usable
      }))
    };
  }

  const rows = readPackedPayloadItemRows(gl, selection.payload, selection.localIndex);
  const centerPx = [rows.row0[0], rows.row0[1]];
  const payloadRadius = rows.row0[2];
  const depth = rows.row0[3];
  const colorAlpha = rows.row1.slice(0, 4);
  const conic = rows.row2.slice(0, 3);
  const reserved = rows.row2[3];
  const misc = rows.row3.slice(0, 4);
  const unclampedPointSize = payloadRadius * 2.0;
  const clampedPointSize = Math.max(1.0, unclampedPointSize);
  const clampApplied = clampedPointSize !== unclampedPointSize;
  const spriteHalfExtentPx = clampedPointSize * 0.5;
  const drawRadiusBbox = [
    centerPx[0] - payloadRadius,
    centerPx[1] - payloadRadius,
    centerPx[0] + payloadRadius,
    centerPx[1] + payloadRadius
  ];
  const spriteBbox = [
    centerPx[0] - spriteHalfExtentPx,
    centerPx[1] - spriteHalfExtentPx,
    centerPx[0] + spriteHalfExtentPx,
    centerPx[1] + spriteHalfExtentPx
  ];
  const spritePixelArea = clampedPointSize * clampedPointSize;
  const gaussianEffectiveAreaEstimate = Math.PI * payloadRadius * payloadRadius;
  const coverageOvershootEstimate = spritePixelArea - gaussianEffectiveAreaEstimate;
  const coverageOvershootRatio = gaussianEffectiveAreaEstimate > 1e-8
    ? spritePixelArea / gaussianEffectiveAreaEstimate
    : Infinity;

  const fragmentSamples = {
    center: evaluatePackedFragmentSample(centerPx, payloadRadius, conic, colorAlpha, [0.5, 0.5], clampedPointSize),
    midX: evaluatePackedFragmentSample(centerPx, payloadRadius, conic, colorAlpha, [0.75, 0.5], clampedPointSize),
    edgeX: evaluatePackedFragmentSample(centerPx, payloadRadius, conic, colorAlpha, [1.0, 0.5], clampedPointSize),
    edgeY: evaluatePackedFragmentSample(centerPx, payloadRadius, conic, colorAlpha, [0.5, 1.0], clampedPointSize),
    corner: evaluatePackedFragmentSample(centerPx, payloadRadius, conic, colorAlpha, [1.0, 1.0], clampedPointSize)
  };

  return {
    ok: true,
    requestedIndex: selection.absoluteIndex,
    payloadIndex: selection.payloadIndex,
    localIndex: selection.localIndex,
    payloadCandidateCount: payloadCandidates.length,
    payloadKind: selection.payload?.kind ?? 'gpu-packed-texture',
    payloadCount: selection.payload?.count ?? 0,
    payloadWidth: selection.payload?.width ?? 0,
    payloadHeight: selection.payload?.height ?? 0,
    rowsPerColumn: rows.rowsPerColumn,
    columnCount: selection.payload?.columnCount ?? 1,
    textureColumnIndex: rows.columnIndex,
    textureRowIndex: rows.rowIndex,
    textureXBase: rows.xBase,
    centerPx,
    depth,
    payloadRadius,
    colorAlpha,
    conic,
    reserved,
    misc,
    unclampedPointSize,
    clampedPointSize,
    clampApplied,
    drawRadiusBbox,
    spriteBbox,
    spriteHalfExtentPx,
    spritePixelArea,
    gaussianEffectiveAreaEstimate,
    coverageOvershootEstimate,
    coverageOvershootRatio,
    fragmentEquation: {
      pointCoordToPixelCenter: 'localPixelIndex = clamp(floor(gl_PointCoord * pointSizePx), 0.0, pointSizePx - 1.0)',
      spritePixelIndex: 'pixelIndexPx = floor(centerPx - pointSizePx * 0.5) + localPixelIndex',
      displacement: 'd = pixelIndexPx - centerPx',
      power: '-0.5 * (conic.x * dx^2 + conic.z * dy^2) - conic.y * dx * dy',
      gaussianAlpha: 'colorAlpha.a * exp(power)',
      finalAlpha: 'clamp(gaussianAlpha, 0.0, 0.99)',
      alphaCutoff: 'discard when finalAlpha < 1.0 / 255.0'
    },
    fragmentSamples
  };
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

export function buildGpuPackedPayloadAtlas(gl, gpu, payloads, options = {}) {
  const validPayloads = Array.isArray(payloads)
    ? payloads.filter((payload) => payload?.texture && payload?.gl === gl)
    : [];
  if (validPayloads.length <= 1) {
    return {
      atlasPayload: validPayloads.length === 1 ? validPayloads[0] : null,
      payloadCount: validPayloads.length,
      totalCount: getPayloadDrawCount(validPayloads),
      mergeAttempted: false,
      mergeSucceeded: false,
      mergeCopyCount: 0,
      mergeFailureReason: validPayloads.length === 1 ? 'single-payload-not-merged' : 'no-valid-payloads',
      mergePolicySelectedPath: validPayloads.length === 1 && isBackendAtlasPayload(validPayloads[0])
        ? 'backend-atlas-payload'
        : 'single-payload',
      mergePolicyReason: validPayloads.length === 1 && isBackendAtlasPayload(validPayloads[0])
        ? 'backend-atlas-already-ready'
        : (validPayloads.length === 1 ? 'single-payload-not-needed' : 'no-valid-payloads'),
      mergePolicyEstimatedCopyCount: 0,
      mergePolicyEstimatedDispatchSavings: 0,
      mergePolicyAtlasArea: 0,
      mergePolicyOverrideMode: options.policyOverride?.mode ?? 'none',
      mergePolicyOverrideReason: options.policyOverride?.reason ?? 'none',
      mergeAtlasReused: false,
      mergeAtlasRebuilt: false,
      mergeAtlasChurnReason: 'none',
      mergeAtlasCapacityWidth: 0,
      mergeAtlasCapacityHeight: 0,
      mergeAtlasAllocationBytes: 0,
      mergeAtlasSavedAllocationBytes: 0,
      mergeTextureWidth: validPayloads[0]?.width ?? 0,
      mergeTextureHeight: validPayloads[0]?.height ?? 0,
      mergeRowCount: getPayloadDrawCount(validPayloads),
      mergeRowsPerColumn: validPayloads.length === 1 ? getPayloadRowsPerColumn(validPayloads[0]) : 0,
      mergeColumnCount: validPayloads[0]?.columnCount ?? 1
    };
  }

  const resources = options.resources || ensureGpuPackedPayloadTextureDrawResources(
    gl,
    gpu,
    options.storageKey
  );
  if (!resources) {
    return {
      atlasPayload: null,
      payloadCount: validPayloads.length,
      totalCount: getPayloadDrawCount(validPayloads),
      mergeAttempted: false,
      mergeSucceeded: false,
      mergeCopyCount: 0,
      mergeFailureReason: 'merge-resources-unavailable',
      mergePolicySelectedPath: 'multi-payload',
      mergePolicyReason: 'merge-resources-unavailable',
      mergePolicyEstimatedCopyCount: 0,
      mergePolicyEstimatedDispatchSavings: 0,
      mergePolicyAtlasArea: 0,
      mergePolicyOverrideMode: options.policyOverride?.mode ?? 'none',
      mergePolicyOverrideReason: options.policyOverride?.reason ?? 'none',
      mergeAtlasReused: false,
      mergeAtlasRebuilt: false,
      mergeAtlasChurnReason: 'merge-resources-unavailable',
      mergeAtlasCapacityWidth: 0,
      mergeAtlasCapacityHeight: 0,
      mergeAtlasAllocationBytes: 0,
      mergeAtlasSavedAllocationBytes: 0,
      mergeTextureWidth: 0,
      mergeTextureHeight: 0,
      mergeRowCount: 0,
      mergeRowsPerColumn: 0,
      mergeColumnCount: 0
    };
  }

  const totalCount = getPayloadDrawCount(validPayloads);
  const mergePolicy = buildMergePolicyDecision(
    gl,
    resources,
    validPayloads,
    totalCount,
    options.policyOverride ?? null
  );
  const mergeResult = mergePolicy.shouldMerge
    ? mergeGpuPackedPayloads(gl, resources, validPayloads, totalCount, mergePolicy.layout)
    : {
        attempted: false,
        merged: false,
        copyCount: 0,
        failureReason: mergePolicy.policyReason,
        textureWidth: mergePolicy.layout?.textureWidth ?? 0,
        textureHeight: mergePolicy.layout?.textureHeight ?? 0,
        rowCount: totalCount,
        rowsPerColumn: mergePolicy.layout?.rowsPerColumn ?? 0,
        columnCount: mergePolicy.layout?.columnCount ?? 0,
        atlasReused: false,
        atlasRebuilt: false,
        atlasChurnReason: 'none',
        atlasCapacityWidth: resources.mergeTextureCapacityWidth ?? 0,
        atlasCapacityHeight: resources.mergeTextureCapacityHeight ?? 0,
        atlasAllocationBytes: 0,
        atlasSavedAllocationBytes: 0
      };

  const atlasPayload = mergeResult.merged
    ? {
        kind: 'gpu-packed-texture-atlas',
        gl,
        texture: resources.mergeTexture,
        width: mergeResult.textureWidth ?? 0,
        height: mergeResult.textureHeight ?? 0,
        count: totalCount,
        rowsPerColumn: mergeResult.rowsPerColumn ?? 0,
        columnCount: mergeResult.columnCount ?? 0
      }
    : null;

  return {
    atlasPayload,
    payloadCount: validPayloads.length,
    totalCount,
    mergeAttempted: !!mergeResult.attempted,
    mergeSucceeded: !!mergeResult.merged,
    mergeCopyCount: mergeResult.copyCount ?? 0,
    mergeFailureReason: mergeResult.failureReason ?? 'none',
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
    mergePolicyEstimatedCopyCount: mergePolicy.estimatedCopyCount ?? validPayloads.length,
    mergePolicyEstimatedDispatchSavings: mergePolicy.estimatedDispatchSavings ?? Math.max(0, validPayloads.length - 1),
    mergePolicyAtlasArea: mergePolicy.atlasArea ?? 0,
    mergePolicyOverrideMode: mergePolicy.overrideMode ?? 'none',
    mergePolicyOverrideReason: mergePolicy.overrideReason ?? 'none',
    mergeAtlasReused: !!mergeResult.atlasReused,
    mergeAtlasRebuilt: !!mergeResult.atlasRebuilt,
    mergeAtlasChurnReason: mergeResult.atlasChurnReason ?? 'none',
    mergeAtlasCapacityWidth: mergeResult.atlasCapacityWidth ?? 0,
    mergeAtlasCapacityHeight: mergeResult.atlasCapacityHeight ?? 0,
    mergeAtlasAllocationBytes: mergeResult.atlasAllocationBytes ?? 0,
    mergeAtlasSavedAllocationBytes: mergeResult.atlasSavedAllocationBytes ?? 0,
    mergeTextureWidth: mergeResult.textureWidth ?? 0,
    mergeTextureHeight: mergeResult.textureHeight ?? 0,
    mergeRowCount: mergeResult.rowCount ?? totalCount,
    mergeRowsPerColumn: mergeResult.rowsPerColumn ?? 0,
    mergeColumnCount: mergeResult.columnCount ?? 0
  };
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

function buildMergePolicyDecision(gl, resources, payloads, totalCount, policyOverride = null) {
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
  const canReuseExistingAtlas =
    layout.ok &&
    (resources?.mergeTextureCapacityWidth ?? 0) >= layout.textureWidth &&
    (resources?.mergeTextureCapacityHeight ?? 0) >= layout.textureHeight;

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

  if (policyOverride?.mode === 'favor-merged-atlas') {
    return {
      shouldMerge: true,
      policySelectedPath: 'merged-atlas',
      policyReason: policyOverride?.reason ?? 'merge-policy-override-draw-throughput',
      estimatedDispatchSavings: dispatchSavings,
      estimatedCopyCount,
      atlasArea,
      layout,
      overrideMode: policyOverride.mode,
      overrideReason: policyOverride?.reason ?? 'merge-policy-override-draw-throughput'
    };
  }

  if (policyOverride?.mode === 'favor-atlas-reuse') {
    if (canReuseExistingAtlas || previousMergeSucceeded) {
      return {
        shouldMerge: true,
        policySelectedPath: 'merged-atlas',
        policyReason: policyOverride?.reason ?? 'merge-policy-override-transform-throughput-reuse',
        estimatedDispatchSavings: dispatchSavings,
        estimatedCopyCount,
        atlasArea,
        layout,
        overrideMode: policyOverride.mode,
        overrideReason: policyOverride?.reason ?? 'merge-policy-override-transform-throughput-reuse'
      };
    }
    if (dispatchSavings <= 4) {
      return {
        shouldMerge: false,
        policySelectedPath: 'multi-payload',
        policyReason: policyOverride?.reason ?? 'merge-policy-override-transform-throughput-avoid-rebuild',
        estimatedDispatchSavings: dispatchSavings,
        estimatedCopyCount,
        atlasArea,
        layout,
        overrideMode: policyOverride.mode,
        overrideReason: policyOverride?.reason ?? 'merge-policy-override-transform-throughput-avoid-rebuild'
      };
    }
  }

  if (payloadCount < GPU_MERGE_MIN_PAYLOAD_COUNT && totalCount < GPU_MERGE_MIN_TOTAL_ROWS) {
    return {
      shouldMerge: false,
      policySelectedPath: 'multi-payload',
      policyReason: 'merge-policy-small-workload',
      estimatedDispatchSavings: dispatchSavings,
      estimatedCopyCount,
      atlasArea,
      layout,
      overrideMode: policyOverride?.mode ?? 'none',
      overrideReason: policyOverride?.reason ?? 'none'
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
      layout,
      overrideMode: policyOverride?.mode ?? 'none',
      overrideReason: policyOverride?.reason ?? 'none'
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
      layout,
      overrideMode: policyOverride?.mode ?? 'none',
      overrideReason: policyOverride?.reason ?? 'none'
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
      layout,
      overrideMode: policyOverride?.mode ?? 'none',
      overrideReason: policyOverride?.reason ?? 'none'
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
    layout,
    overrideMode: policyOverride?.mode ?? 'none',
    overrideReason: policyOverride?.reason ?? 'none'
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
    const atlasResult = buildGpuPackedPayloadAtlas(gl, gpu, payloads, {
      resources,
      storageKey: options.storageKey,
      policyOverride: options.policyOverride ?? null
    });
    let drawCount = 0;
    let drawCallCount = 0;
    let bindCount = 0;

    gl.useProgram(resources.program);
    gl.bindVertexArray(resources.vao);
    gl.uniform2f(resources.uniformViewportPx, canvasWidth, canvasHeight);
    gl.uniform1i(resources.uniformPackedTexture, 0);

    if (atlasResult.mergeSucceeded && atlasResult.atlasPayload?.texture) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, atlasResult.atlasPayload.texture);
      gl.uniform1i(resources.uniformPackedRowsPerColumn, Math.max(1, getPayloadRowsPerColumn(atlasResult.atlasPayload)));
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
        gl.uniform1i(resources.uniformPackedRowsPerColumn, getPayloadRowsPerColumn(payload));
        bindCount++;
        gl.drawArrays(gl.POINTS, 0, count);
        drawCount += count;
        drawCallCount++;
      }
    }

    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindVertexArray(null);

    resources.lastMergeSucceeded = !!atlasResult.mergeSucceeded;
    resources.lastMergeFailureReason = atlasResult.mergeFailureReason ?? 'none';
    resources.lastMergeRowCount = atlasResult.mergeRowCount ?? totalCount;
    resources.lastMergePayloadCount = payloads.length;
    resources.lastMergeAtlasReused = !!atlasResult.mergeAtlasReused;
    resources.lastMergeAtlasRebuilt = !!atlasResult.mergeAtlasRebuilt;
    resources.lastMergeAtlasChurnReason = atlasResult.mergeAtlasChurnReason ?? 'none';

    return {
      drawCount,
      drawCallCount,
      bindCount,
      setupCount: 1,
      dispatchCount: drawCallCount,
      dispatchMode: atlasResult.mergeSucceeded
        ? 'shared-texture-merged-payloads'
        : payloads.length > 1
          ? 'shared-texture-multi-payload'
          : (isBackendAtlasPayload(payloads[0])
            ? 'shared-texture-backend-atlas-payload'
            : 'shared-texture-single-payload'),
      resources,
      payloadCount: payloads.length,
      mergeAttempted: !!atlasResult.mergeAttempted,
      mergeCopyCount: atlasResult.mergeCopyCount,
      mergeFailureReason: atlasResult.mergeFailureReason ?? 'none',
      mergeTextureWidth: atlasResult.mergeTextureWidth ?? 0,
      mergeTextureHeight: atlasResult.mergeTextureHeight ?? 0,
      mergeRowCount: atlasResult.mergeRowCount ?? totalCount,
      mergeRowsPerColumn: atlasResult.mergeRowsPerColumn ?? 0,
      mergeColumnCount: atlasResult.mergeColumnCount ?? 0,
      mergePolicySelectedPath: atlasResult.mergePolicySelectedPath ?? 'none',
      mergePolicyReason: atlasResult.mergePolicyReason ?? 'none',
      mergePolicyEstimatedCopyCount: atlasResult.mergePolicyEstimatedCopyCount ?? payloads.length,
      mergePolicyEstimatedDispatchSavings: atlasResult.mergePolicyEstimatedDispatchSavings ?? Math.max(0, payloads.length - 1),
      mergePolicyAtlasArea: atlasResult.mergePolicyAtlasArea ?? 0,
      mergePolicyOverrideMode: atlasResult.mergePolicyOverrideMode ?? 'none',
      mergePolicyOverrideReason: atlasResult.mergePolicyOverrideReason ?? 'none',
      mergeAtlasReused: !!atlasResult.mergeAtlasReused,
      mergeAtlasRebuilt: !!atlasResult.mergeAtlasRebuilt,
      mergeAtlasChurnReason: atlasResult.mergeAtlasChurnReason ?? 'none',
      mergeAtlasCapacityWidth: atlasResult.mergeAtlasCapacityWidth ?? 0,
      mergeAtlasCapacityHeight: atlasResult.mergeAtlasCapacityHeight ?? 0,
      mergeAtlasAllocationBytes: atlasResult.mergeAtlasAllocationBytes ?? 0,
      mergeAtlasSavedAllocationBytes: atlasResult.mergeAtlasSavedAllocationBytes ?? 0
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
