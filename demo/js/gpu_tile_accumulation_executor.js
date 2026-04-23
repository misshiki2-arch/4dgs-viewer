import { createProgram, clearToGray, disableDepth } from './gpu_gl_utils.js';

const TILE_ACCUMULATION_MAX_ITEMS = 2048;
const TILE_ACCUMULATION_EARLY_OUT_ENABLED = true;
const TILE_ACCUMULATION_EARLY_OUT_THRESHOLD = 0.0001;
const TILE_ACCUMULATION_CONTRACT_VERSION = 'step86-explicit-ftb-v4';
const TILE_ACCUMULATION_CONTRIBUTION_LOG_LIMIT = 5;
const TILE_ACCUMULATION_REPRESENTATIVE_CANDIDATE_LIMIT = 12;
const TILE_ACCUMULATION_FRAMEBUFFER_MATCH_TOLERANCE = 2.0 / 255.0;

const TILE_ACCUMULATION_VERTEX_SHADER = `#version 300 es
precision highp float;

uniform vec4 uTileRectPx;
uniform vec2 uViewportPx;

void main() {
  vec2 unitCorner;
  if (gl_VertexID == 0) {
    unitCorner = vec2(0.0, 0.0);
  } else if (gl_VertexID == 1) {
    unitCorner = vec2(1.0, 0.0);
  } else if (gl_VertexID == 2) {
    unitCorner = vec2(0.0, 1.0);
  } else if (gl_VertexID == 3) {
    unitCorner = vec2(0.0, 1.0);
  } else if (gl_VertexID == 4) {
    unitCorner = vec2(1.0, 0.0);
  } else {
    unitCorner = vec2(1.0, 1.0);
  }

  vec2 pixelEdgePx = mix(uTileRectPx.xy, uTileRectPx.zw, unitCorner);
  float x = (pixelEdgePx.x / uViewportPx.x) * 2.0 - 1.0;
  float y = 1.0 - (pixelEdgePx.y / uViewportPx.y) * 2.0;
  gl_Position = vec4(x, y, 0.0, 1.0);
}
`;

const TILE_ACCUMULATION_FRAGMENT_SHADER = `#version 300 es
precision highp float;
precision highp sampler2D;

uniform sampler2D uPayloadTexture;
uniform vec2 uViewportPx;
uniform int uBatchStartItem;
uniform int uBatchCount;
uniform int uRowsPerColumn;
uniform int uTexelsPerItem;
uniform float uBackgroundGray;

out vec4 outColor;

vec4 fetchPayloadRow(int itemIndex, int rowIndex) {
  int rowsPerColumn = max(1, uRowsPerColumn);
  int texelsPerItem = max(1, uTexelsPerItem);
  int columnIndex = itemIndex / rowsPerColumn;
  int rowIndexY = itemIndex - (columnIndex * rowsPerColumn);
  int texelX = columnIndex * texelsPerItem + rowIndex;
  return texelFetch(uPayloadTexture, ivec2(texelX, rowIndexY), 0);
}

void main() {
  vec2 pixelIndexPx = vec2(
    gl_FragCoord.x - 0.5,
    uViewportPx.y - gl_FragCoord.y - 0.5
  );

  vec3 accumColor = vec3(0.0);
  float T = 1.0;
  int clampedCount = min(uBatchCount, ${TILE_ACCUMULATION_MAX_ITEMS});

  for (int i = 0; i < ${TILE_ACCUMULATION_MAX_ITEMS}; ++i) {
    if (i >= clampedCount) break;

    int itemIndex = uBatchStartItem + i;
    vec4 row0 = fetchPayloadRow(itemIndex, 0);
    vec4 row1 = fetchPayloadRow(itemIndex, 1);
    vec4 row2 = fetchPayloadRow(itemIndex, 2);

    vec2 centerPx = row0.xy;
    vec3 conic = vec3(row2.x, row2.y, row2.z);
    vec2 d = centerPx - pixelIndexPx;
    float dx = d.x;
    float dy = d.y;
    float power =
      -0.5 * (conic.x * dx * dx + conic.z * dy * dy)
      - conic.y * dx * dy;
    if (power > 0.0) continue;

    float alpha = min(0.99, row1.a * exp(power));
    if (alpha < (1.0 / 255.0)) continue;

    float testT = T * (1.0 - alpha);
    if (testT < ${TILE_ACCUMULATION_EARLY_OUT_THRESHOLD}) break;
    accumColor += row1.rgb * alpha * T;
    T = testT;
  }

  vec3 resolved = accumColor + T * vec3(uBackgroundGray);
  outColor = vec4(resolved, 1.0);
}
`;

function ensureTileAccumulationState(gl, gpu) {
  if (gpu.tileAccumulationState?.program && gpu.tileAccumulationState?.vao && gpu.tileAccumulationState?.texture) {
    return gpu.tileAccumulationState;
  }

  const program = createProgram(gl, TILE_ACCUMULATION_VERTEX_SHADER, TILE_ACCUMULATION_FRAGMENT_SHADER);
  const vao = gl.createVertexArray();
  const texture = gl.createTexture();

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_2D, null);

  gpu.tileAccumulationState = {
    program,
    vao,
    texture,
    textureWidth: 0,
    textureHeight: 0,
    uniformTileRectPx: gl.getUniformLocation(program, 'uTileRectPx'),
    uniformViewportPx: gl.getUniformLocation(program, 'uViewportPx'),
    uniformBatchStartItem: gl.getUniformLocation(program, 'uBatchStartItem'),
    uniformBatchCount: gl.getUniformLocation(program, 'uBatchCount'),
    uniformRowsPerColumn: gl.getUniformLocation(program, 'uRowsPerColumn'),
    uniformTexelsPerItem: gl.getUniformLocation(program, 'uTexelsPerItem'),
    uniformPayloadTexture: gl.getUniformLocation(program, 'uPayloadTexture'),
    uniformBackgroundGray: gl.getUniformLocation(program, 'uBackgroundGray')
  };
  return gpu.tileAccumulationState;
}

function validateTextureAllocation(gl, payload) {
  const maxTextureSize = Number(gl.getParameter(gl.MAX_TEXTURE_SIZE)) | 0;
  const requestedWidth = Number(payload?.textureWidth);
  const requestedHeight = Number(payload?.textureHeight);
  const rowsPerColumn = Number(payload?.rowsPerColumn);
  const columnCount = Number(payload?.columnCount);
  const texelsPerItem = Number(payload?.texelsPerItem);
  const dataLength = payload?.packedFloats instanceof Float32Array ? payload.packedFloats.length : 0;

  const failure = (reason) => ({
    requestedTextureWidth: Number.isFinite(requestedWidth) ? requestedWidth : 0,
    requestedTextureHeight: Number.isFinite(requestedHeight) ? requestedHeight : 0,
    validatedTextureWidth: 0,
    validatedTextureHeight: 0,
    textureAllocationValid: false,
    textureAllocationFailureReason: reason,
    maxTextureSize,
    payloadRowsPerColumn: Number.isFinite(rowsPerColumn) ? rowsPerColumn : 0,
    payloadColumnCount: Number.isFinite(columnCount) ? columnCount : 0,
    payloadTexelsPerItem: Number.isFinite(texelsPerItem) ? texelsPerItem : 0,
    payloadFloatCount: dataLength
  });
  if (!Number.isFinite(requestedWidth) || !Number.isFinite(requestedHeight)) {
    return failure('non-finite-texture-dimension');
  }
  if (!Number.isFinite(rowsPerColumn) || rowsPerColumn <= 0) {
    return failure('invalid-rows-per-column');
  }
  if (!Number.isFinite(columnCount) || columnCount <= 0) {
    return failure('invalid-column-count');
  }
  if (!Number.isFinite(texelsPerItem) || texelsPerItem <= 0) {
    return failure('invalid-texels-per-item');
  }

  const width = requestedWidth | 0;
  const height = requestedHeight | 0;
  if (width <= 0 || height <= 0) {
    return failure('non-positive-texture-dimension');
  }
  if (maxTextureSize <= 0) {
    return failure('max-texture-size-unavailable');
  }
  if (width > maxTextureSize || height > maxTextureSize) {
    return failure('texture-dimension-exceeds-max-texture-size');
  }
  const expectedFloatCount = width * height * 4;
  if (dataLength !== expectedFloatCount) {
    return failure('payload-float-count-mismatch');
  }

  return {
    requestedTextureWidth: requestedWidth,
    requestedTextureHeight: requestedHeight,
    validatedTextureWidth: width,
    validatedTextureHeight: height,
    textureAllocationValid: true,
    textureAllocationFailureReason: 'none',
    maxTextureSize,
    payloadRowsPerColumn: rowsPerColumn | 0,
    payloadColumnCount: columnCount | 0,
    payloadTexelsPerItem: texelsPerItem | 0,
    payloadFloatCount: dataLength
  };
}

function uploadAccumulationPayloadTexture(gl, state, payload, validation) {
  const width = validation.validatedTextureWidth;
  const height = validation.validatedTextureHeight;
  const data = payload?.packedFloats instanceof Float32Array ? payload.packedFloats : new Float32Array(width * height * 4);
  gl.bindTexture(gl.TEXTURE_2D, state.texture);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA32F,
    width,
    height,
    0,
    gl.RGBA,
    gl.FLOAT,
    data
  );
  gl.bindTexture(gl.TEXTURE_2D, null);

  state.textureWidth = width;
  state.textureHeight = height;
  state.rowsPerColumn = validation.payloadRowsPerColumn;
  state.columnCount = validation.payloadColumnCount;
  state.texelsPerItem = validation.payloadTexelsPerItem;
}

function enableTileScissor(gl, canvasHeight, rect) {
  const x0 = rect?.[0] ?? 0;
  const y0 = rect?.[1] ?? 0;
  const x1 = rect?.[2] ?? x0;
  const y1 = rect?.[3] ?? y0;
  const width = Math.max(0, x1 - x0);
  const height = Math.max(0, y1 - y0);
  gl.scissor(x0, canvasHeight - y1, width, height);
}

function drawAccumulationBatch(gl, state, canvas, batch, bgGray01) {
  gl.useProgram(state.program);
  gl.bindVertexArray(state.vao);
  gl.uniform4f(
    state.uniformTileRectPx,
    batch.rect[0] ?? 0,
    batch.rect[1] ?? 0,
    batch.rect[2] ?? 0,
    batch.rect[3] ?? 0
  );
  gl.uniform2f(state.uniformViewportPx, canvas.width, canvas.height);
  gl.uniform1i(state.uniformBatchStartItem, batch.startItemIndex | 0);
  gl.uniform1i(state.uniformBatchCount, batch.packedCount | 0);
  gl.uniform1i(state.uniformRowsPerColumn, state.rowsPerColumn | 0);
  gl.uniform1i(state.uniformTexelsPerItem, state.texelsPerItem | 0);
  gl.uniform1f(state.uniformBackgroundGray, bgGray01);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, state.texture);
  gl.uniform1i(state.uniformPayloadTexture, 0);
  enableTileScissor(gl, canvas.height, batch.rect);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
}

function buildBatchRepresentativePixelIndex(batch) {
  const rect = Array.isArray(batch?.rect) ? batch.rect : null;
  if (!rect || rect.length < 4) return [0, 0];
  const minX = Number.isFinite(rect[0]) ? (rect[0] | 0) : 0;
  const minY = Number.isFinite(rect[1]) ? (rect[1] | 0) : 0;
  const maxXExclusive = Number.isFinite(rect[2]) ? (rect[2] | 0) : minX + 1;
  const maxYExclusive = Number.isFinite(rect[3]) ? (rect[3] | 0) : minY + 1;
  return [
    Math.max(minX, Math.floor((minX + Math.max(minX + 1, maxXExclusive) - 1) * 0.5)),
    Math.max(minY, Math.floor((minY + Math.max(minY + 1, maxYExclusive) - 1) * 0.5))
  ];
}

function getPayloadPackedFloat(payload, globalItemIndex, floatOffset) {
  const packedFloats = payload?.packedFloats;
  const rowsPerColumn = Number.isFinite(payload?.rowsPerColumn) ? Math.max(1, payload.rowsPerColumn | 0) : 0;
  const texelsPerItem = Number.isFinite(payload?.texelsPerItem) ? Math.max(1, payload.texelsPerItem | 0) : 0;
  const textureWidth = Number.isFinite(payload?.textureWidth) ? Math.max(1, payload.textureWidth | 0) : 0;
  if (!(packedFloats instanceof Float32Array) || rowsPerColumn <= 0 || texelsPerItem <= 0 || textureWidth <= 0) {
    return null;
  }
  const safeGlobalItemIndex = Number.isFinite(globalItemIndex) ? Math.max(0, globalItemIndex | 0) : -1;
  const safeFloatOffset = Number.isFinite(floatOffset) ? Math.max(0, floatOffset | 0) : -1;
  if (safeGlobalItemIndex < 0 || safeFloatOffset < 0) return null;
  const columnIndex = Math.floor(safeGlobalItemIndex / rowsPerColumn);
  const rowIndex = safeGlobalItemIndex - columnIndex * rowsPerColumn;
  const texelXBase = columnIndex * texelsPerItem;
  const texelOffset = Math.floor(safeFloatOffset / 4);
  const componentOffset = safeFloatOffset - texelOffset * 4;
  const index = ((rowIndex * textureWidth) + texelXBase + texelOffset) * 4 + componentOffset;
  if (index < 0 || index >= packedFloats.length) return null;
  return packedFloats[index];
}

function readPackedBatchItem(batch, itemIndex, payload = null) {
  const floatsPerItem = Number.isFinite(batch?.floatsPerItem) ? Math.max(16, batch.floatsPerItem | 0) : 16;
  const base = itemIndex * floatsPerItem;
  let readFloat = null;
  let readMode = 'none';
  if (batch?.packed instanceof Float32Array && base >= 0 && (base + 16) <= batch.packed.length) {
    readMode = 'batch-packed';
    readFloat = (offset) => batch.packed[base + offset];
  } else if (
    payload?.packedFloats instanceof Float32Array &&
    Number.isFinite(batch?.startItemIndex) &&
    Number.isFinite(itemIndex)
  ) {
    const globalItemIndex = (batch.startItemIndex | 0) + (itemIndex | 0);
    readMode = 'payload-packed-floats';
    readFloat = (offset) => getPayloadPackedFloat(payload, globalItemIndex, offset);
  }
  if (!readFloat) return null;
  const centerX = readFloat(0);
  const centerY = readFloat(1);
  const radiusPx = readFloat(2);
  const depth = readFloat(3);
  const r = readFloat(4);
  const g = readFloat(5);
  const b = readFloat(6);
  const alpha = readFloat(7);
  const conic0 = readFloat(8);
  const conic1 = readFloat(9);
  const conic2 = readFloat(10);
  const rectMinX = readFloat(12);
  const rectMinY = readFloat(13);
  const rectMaxXInclusive = readFloat(14);
  const rectMaxYInclusive = readFloat(15);
  if (
    !Number.isFinite(centerX) ||
    !Number.isFinite(centerY) ||
    !Number.isFinite(radiusPx) ||
    !Number.isFinite(r) ||
    !Number.isFinite(g) ||
    !Number.isFinite(b) ||
    !Number.isFinite(alpha) ||
    !Number.isFinite(conic0) ||
    !Number.isFinite(conic1) ||
    !Number.isFinite(conic2) ||
    !Number.isFinite(rectMinX) ||
    !Number.isFinite(rectMinY) ||
    !Number.isFinite(rectMaxXInclusive) ||
    !Number.isFinite(rectMaxYInclusive)
  ) {
    return null;
  }
  return {
    itemReadMode: readMode,
    centerPx: [centerX, centerY],
    radiusPx,
    depth: Number.isFinite(depth) ? depth : null,
    rgb: [r, g, b],
    colorAlpha: [
      r,
      g,
      b,
      alpha
    ],
    conic: [
      conic0,
      conic1,
      conic2
    ],
    rectMinPx: [rectMinX, rectMinY],
    rectMaxPxExclusive: [rectMaxXInclusive + 1, rectMaxYInclusive + 1]
  };
}

function pixelInsideRectExclusive(pixelIndexPx, item) {
  return (
    pixelIndexPx[0] >= item.rectMinPx[0] &&
    pixelIndexPx[0] < item.rectMaxPxExclusive[0] &&
    pixelIndexPx[1] >= item.rectMinPx[1] &&
    pixelIndexPx[1] < item.rectMaxPxExclusive[1]
  );
}

function evaluateBatchItemAtPixel(item, pixelIndexPx) {
  if (!item || !pixelInsideRectExclusive(pixelIndexPx, item)) {
    return { survivesFragment: false, finalAlpha: 0, rgb: item?.rgb ?? [0, 0, 0], depth: item?.depth ?? null };
  }
  const dx = item.centerPx[0] - pixelIndexPx[0];
  const dy = item.centerPx[1] - pixelIndexPx[1];
  const power =
    -0.5 * (item.conic[0] * dx * dx + item.conic[2] * dy * dy) -
    item.conic[1] * dx * dy;
  if (power > 0.0) {
    return { survivesFragment: false, finalAlpha: 0, rgb: item.rgb, depth: item.depth };
  }
  const finalAlpha = Math.min(0.99, item.colorAlpha[3] * Math.exp(power));
  return {
    survivesFragment: finalAlpha >= (1.0 / 255.0),
    finalAlpha,
    rgb: item.rgb,
    depth: item.depth
  };
}

function clampColor01(rgb) {
  const safe = Array.isArray(rgb) ? rgb : [0, 0, 0];
  return [
    Number.isFinite(safe[0]) ? Number(safe[0]) : 0,
    Number.isFinite(safe[1]) ? Number(safe[1]) : 0,
    Number.isFinite(safe[2]) ? Number(safe[2]) : 0
  ];
}

function addColor3(a, b) {
  return [
    (a?.[0] ?? 0) + (b?.[0] ?? 0),
    (a?.[1] ?? 0) + (b?.[1] ?? 0),
    (a?.[2] ?? 0) + (b?.[2] ?? 0)
  ];
}

function scaleColor3(rgb, scalar) {
  const safeRgb = clampColor01(rgb);
  const s = Number.isFinite(scalar) ? Number(scalar) : 0;
  return [safeRgb[0] * s, safeRgb[1] * s, safeRgb[2] * s];
}

function buildFramebufferReadbackSummary(reason = 'not-attempted') {
  return {
    tileAccumulationRepresentativeSampleFramebufferColor: [0, 0, 0],
    tileAccumulationRepresentativeSampleFramebufferReadbackValid: false,
    tileAccumulationRepresentativeSampleFramebufferReadbackReason: reason,
    tileAccumulationRepresentativeSampleFramebufferReadbackPixel: [0, 0],
    tileAccumulationRepresentativeSampleFramebufferReadbackGlPixel: [0, 0],
    tileAccumulationRepresentativeSampleFramebufferReadbackRgba8: [0, 0, 0, 0],
    tileAccumulationRepresentativeSampleResolvedColorDelta: [0, 0, 0],
    tileAccumulationRepresentativeSampleResolvedColorDeltaAbsMax: 0,
    tileAccumulationRepresentativeSampleResolvedColorMatchesFramebuffer: false,
    tileAccumulationRepresentativeSampleResolvedColorMatchTolerance: TILE_ACCUMULATION_FRAMEBUFFER_MATCH_TOLERANCE
  };
}

function buildFramebufferResolvedColorDelta(resolvedColor, framebufferColor) {
  const resolved = clampColor01(resolvedColor);
  const framebuffer = clampColor01(framebufferColor);
  const delta = [
    framebuffer[0] - resolved[0],
    framebuffer[1] - resolved[1],
    framebuffer[2] - resolved[2]
  ];
  return {
    delta,
    deltaAbsMax: Math.max(Math.abs(delta[0]), Math.abs(delta[1]), Math.abs(delta[2]))
  };
}

function readRepresentativeSampleFramebufferColor(gl, canvas, accumulationStats) {
  const pixel = accumulationStats?.tileAccumulationRepresentativeSamplePixel;
  const resolvedColor = accumulationStats?.tileAccumulationRepresentativeSampleResolvedColor;
  if (!Array.isArray(pixel) || pixel.length < 2) {
    return buildFramebufferReadbackSummary('missing-representative-sample-pixel');
  }
  if (!Array.isArray(resolvedColor) || resolvedColor.length < 3) {
    return buildFramebufferReadbackSummary('missing-representative-resolved-color');
  }

  const width = Number.isFinite(canvas?.width) ? (canvas.width | 0) : 0;
  const height = Number.isFinite(canvas?.height) ? (canvas.height | 0) : 0;
  if (width <= 0 || height <= 0) {
    return buildFramebufferReadbackSummary('invalid-canvas-size');
  }

  const x = Number.isFinite(pixel[0]) ? Math.floor(pixel[0]) : -1;
  const yTop = Number.isFinite(pixel[1]) ? Math.floor(pixel[1]) : -1;
  if (x < 0 || yTop < 0 || x >= width || yTop >= height) {
    return {
      ...buildFramebufferReadbackSummary('representative-sample-pixel-out-of-bounds'),
      tileAccumulationRepresentativeSampleFramebufferReadbackPixel: [x, yTop]
    };
  }

  const yGl = height - 1 - yTop;
  const rgba = new Uint8Array(4);
  try {
    gl.readPixels(x, yGl, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
  } catch (error) {
    return {
      ...buildFramebufferReadbackSummary(`readpixels-failed:${error?.message ?? 'unknown'}`),
      tileAccumulationRepresentativeSampleFramebufferReadbackPixel: [x, yTop],
      tileAccumulationRepresentativeSampleFramebufferReadbackGlPixel: [x, yGl]
    };
  }

  const framebufferColor = [
    rgba[0] / 255.0,
    rgba[1] / 255.0,
    rgba[2] / 255.0
  ];
  const { delta, deltaAbsMax } = buildFramebufferResolvedColorDelta(resolvedColor, framebufferColor);
  return {
    tileAccumulationRepresentativeSampleFramebufferColor: framebufferColor,
    tileAccumulationRepresentativeSampleFramebufferReadbackValid: true,
    tileAccumulationRepresentativeSampleFramebufferReadbackReason: 'readback-ok',
    tileAccumulationRepresentativeSampleFramebufferReadbackPixel: [x, yTop],
    tileAccumulationRepresentativeSampleFramebufferReadbackGlPixel: [x, yGl],
    tileAccumulationRepresentativeSampleFramebufferReadbackRgba8: Array.from(rgba),
    tileAccumulationRepresentativeSampleResolvedColorDelta: delta,
    tileAccumulationRepresentativeSampleResolvedColorDeltaAbsMax: deltaAbsMax,
    tileAccumulationRepresentativeSampleResolvedColorMatchesFramebuffer: deltaAbsMax <= TILE_ACCUMULATION_FRAMEBUFFER_MATCH_TOLERANCE,
    tileAccumulationRepresentativeSampleResolvedColorMatchTolerance: TILE_ACCUMULATION_FRAMEBUFFER_MATCH_TOLERANCE
  };
}

function colorLuma(rgb) {
  const safe = clampColor01(rgb);
  return safe[0] * 0.2126 + safe[1] * 0.7152 + safe[2] * 0.0722;
}

function clampPixelToRect(pixelIndexPx, rect) {
  if (!Array.isArray(rect) || rect.length < 4) return [0, 0];
  const minX = Number.isFinite(rect[0]) ? (rect[0] | 0) : 0;
  const minY = Number.isFinite(rect[1]) ? (rect[1] | 0) : 0;
  const maxXExclusive = Number.isFinite(rect[2]) ? (rect[2] | 0) : minX + 1;
  const maxYExclusive = Number.isFinite(rect[3]) ? (rect[3] | 0) : minY + 1;
  return [
    Math.min(Math.max(minX, pixelIndexPx[0] | 0), Math.max(minX, maxXExclusive - 1)),
    Math.min(Math.max(minY, pixelIndexPx[1] | 0), Math.max(minY, maxYExclusive - 1))
  ];
}

function buildItemCenterPixelIndex(item, batchRect) {
  return clampPixelToRect([
    Math.floor(item?.centerPx?.[0] ?? 0),
    Math.floor(item?.centerPx?.[1] ?? 0)
  ], batchRect);
}

function buildItemRectCenterPixelIndex(item, batchRect) {
  return clampPixelToRect([
    Math.floor((((item?.rectMinPx?.[0] ?? 0) + (item?.rectMaxPxExclusive?.[0] ?? 1) - 1) * 0.5)),
    Math.floor((((item?.rectMinPx?.[1] ?? 0) + (item?.rectMaxPxExclusive?.[1] ?? 1) - 1) * 0.5))
  ], batchRect);
}

function scoreRepresentativeCandidateItem(item) {
  const alpha = Number.isFinite(item?.colorAlpha?.[3]) ? Math.max(0, item.colorAlpha[3]) : 0;
  const radius = Number.isFinite(item?.radiusPx) ? Math.max(0, item.radiusPx) : 0;
  return alpha * Math.max(1, radius);
}

function buildRepresentativeSampleCandidates(batch, payload = null) {
  const packedCount = Number.isFinite(batch?.packedCount) ? Math.max(0, batch.packedCount | 0) : 0;
  const candidates = [];
  const candidateItems = [];
  const seenCandidateKeys = new Set();
  const seenItemPixels = new Set();
  const seenRectPixels = new Set();
  let readableItemCount = 0;
  let itemReadMode = 'none';
  const addCandidate = (pixel, mode, reason, localOrder = -1, score = 0) => {
    const clampedPixel = clampPixelToRect(pixel, batch?.rect);
    const key = `${clampedPixel[0]},${clampedPixel[1]}`;
    if (seenCandidateKeys.has(key)) return;
    seenCandidateKeys.add(key);
    candidates.push({
      pixelIndexPx: clampedPixel,
      selectionMode: mode,
      selectionReason: reason,
      localOrder,
      score
    });
  };

  addCandidate(
    buildBatchRepresentativePixelIndex(batch),
    'tile-center-fallback',
    'fallback tile center pixel'
  );

  for (let i = 0; i < packedCount; i++) {
    const item = readPackedBatchItem(batch, i, payload);
    if (!item) continue;
    readableItemCount++;
    if (itemReadMode === 'none') itemReadMode = item.itemReadMode ?? 'unknown';
    candidateItems.push({
      localOrder: i,
      score: scoreRepresentativeCandidateItem(item),
      item
    });
  }

  candidateItems.sort((a, b) => b.score - a.score);
  for (const candidateItem of candidateItems.slice(0, TILE_ACCUMULATION_REPRESENTATIVE_CANDIDATE_LIMIT)) {
    const centerPixel = buildItemCenterPixelIndex(candidateItem.item, batch?.rect);
    const centerKey = `${centerPixel[0]},${centerPixel[1]}`;
    if (!seenItemPixels.has(centerKey)) {
      seenItemPixels.add(centerKey);
      addCandidate(
        centerPixel,
        'max-item-center-contribution',
        `top scored item center from localOrder=${candidateItem.localOrder}`,
        candidateItem.localOrder,
        candidateItem.score
      );
    }
    const rectCenterPixel = buildItemRectCenterPixelIndex(candidateItem.item, batch?.rect);
    const rectKey = `${rectCenterPixel[0]},${rectCenterPixel[1]}`;
    if (!seenRectPixels.has(rectKey)) {
      seenRectPixels.add(rectKey);
      addCandidate(
        rectCenterPixel,
        'max-item-rect-center-contribution',
        `top scored item rect center from localOrder=${candidateItem.localOrder}`,
        candidateItem.localOrder,
        candidateItem.score
      );
    }
  }

  return {
    candidates,
    readableItemCount,
    itemReadMode,
    usableItemSource: itemReadMode === 'payload-packed-floats'
      ? 'accumulationPayload.packedFloats'
      : (itemReadMode === 'batch-packed' ? 'batch.packed' : 'none')
  };
}

function scoreRepresentativeContributionSummary(summary) {
  if (!summary) return -Infinity;
  const count = Number.isFinite(summary.representativeSampleContributionCount)
    ? summary.representativeSampleContributionCount
    : 0;
  const alphaSum = Number.isFinite(summary.representativeSampleAlphaSum)
    ? summary.representativeSampleAlphaSum
    : 0;
  const contributionLuma = colorLuma(summary.representativeSampleContributionSum);
  const finalT = Number.isFinite(summary.representativeSampleFinalT)
    ? summary.representativeSampleFinalT
    : 1;
  return count * 1000 + alphaSum * 100 + contributionLuma * 10 + (1 - finalT);
}

function buildContributionSummaryAtPixel(batch, bgGray01, representativePixelIndexPx, selectionMeta = null, payload = null) {
  const packedCount = Number.isFinite(batch?.packedCount) ? Math.max(0, batch.packedCount | 0) : 0;
  const clampedCount = Math.min(packedCount, TILE_ACCUMULATION_MAX_ITEMS);
  const safePixelIndexPx = Array.isArray(representativePixelIndexPx) ? representativePixelIndexPx : buildBatchRepresentativePixelIndex(batch);
  const bg = Number.isFinite(bgGray01) ? Math.max(0, Math.min(1, Number(bgGray01))) : 0;
  let transmittance = 1.0;
  let accumColor = [0, 0, 0];
  let contributionCount = 0;
  let alphaSum = 0;
  let contributionSum = [0, 0, 0];
  let earlyOutTriggered = false;
  let earlyOutAtItem = -1;
  const contributionLog = [];
  const thresholdSkipPreview = [];
  let evaluatedItemCount = 0;
  let itemReadMode = 'none';
  let thresholdCrossingCount = 0;
  let thresholdSkippedCount = 0;
  let firstThresholdSkipLocalOrder = -1;
  let firstThresholdSkipAlpha = 0;
  let firstThresholdSkipTransmittanceBefore = 1;
  let firstThresholdSkipTransmittanceAfter = 1;
  let lastContributedLocalOrder = -1;

  for (let i = 0; i < clampedCount; i++) {
    const item = readPackedBatchItem(batch, i, payload);
    if (!item) continue;
    evaluatedItemCount++;
    if (itemReadMode === 'none') itemReadMode = item.itemReadMode ?? 'unknown';
    const evaluation = evaluateBatchItemAtPixel(item, safePixelIndexPx);
    if (!evaluation.survivesFragment) continue;
    const alpha = Number.isFinite(evaluation.finalAlpha) ? Math.max(0, Math.min(0.99, Number(evaluation.finalAlpha))) : 0;
    const transmittanceBefore = transmittance;
    const testT = transmittanceBefore * (1.0 - alpha);
    if (TILE_ACCUMULATION_EARLY_OUT_ENABLED && testT < TILE_ACCUMULATION_EARLY_OUT_THRESHOLD) {
      earlyOutTriggered = true;
      earlyOutAtItem = i;
      thresholdCrossingCount++;
      thresholdSkippedCount = Math.max(1, clampedCount - i);
      firstThresholdSkipLocalOrder = i;
      firstThresholdSkipAlpha = alpha;
      firstThresholdSkipTransmittanceBefore = transmittanceBefore;
      firstThresholdSkipTransmittanceAfter = testT;
      thresholdSkipPreview.push({
        localOrder: i,
        depth: evaluation.depth,
        alpha,
        transmittanceBefore,
        transmittanceAfter: testT,
        skippedRemainingItemCount: Math.max(0, clampedCount - i - 1)
      });
      break;
    }
    const contributionRgb = scaleColor3(evaluation.rgb, alpha * transmittanceBefore);
    accumColor = addColor3(accumColor, contributionRgb);
    contributionSum = addColor3(contributionSum, contributionRgb);
    alphaSum += alpha;
    contributionCount++;
    transmittance = testT;
    lastContributedLocalOrder = i;
    if (contributionLog.length < TILE_ACCUMULATION_CONTRIBUTION_LOG_LIMIT) {
      contributionLog.push({
        localOrder: i,
        depth: evaluation.depth,
        alpha,
        transmittanceBefore,
        transmittanceAfter: transmittance,
        contributionRgb
      });
    }
  }

  const resolvedColor = addColor3(accumColor, [bg * transmittance, bg * transmittance, bg * transmittance]);
  return {
    representativeSampleSelectionMode: selectionMeta?.selectionMode ?? 'tile-center-fallback',
    representativeSampleSelectionReason: selectionMeta?.selectionReason ?? 'fallback tile center pixel',
    representativeSamplePixel: safePixelIndexPx,
    representativeSampleHasContribution: contributionCount > 0,
    representativeSampleCandidateCount: Number.isFinite(selectionMeta?.candidateCount) ? selectionMeta.candidateCount : 1,
    representativeSampleEvaluatedCandidateCount: Number.isFinite(selectionMeta?.evaluatedCandidateCount)
      ? selectionMeta.evaluatedCandidateCount
      : 1,
    representativeSampleUsableItemSource: selectionMeta?.usableItemSource ?? (itemReadMode === 'payload-packed-floats'
      ? 'accumulationPayload.packedFloats'
      : (itemReadMode === 'batch-packed' ? 'batch.packed' : 'none')),
    representativeSampleItemReadMode: selectionMeta?.itemReadMode ?? itemReadMode,
    representativeSampleEvaluatedItemCount: evaluatedItemCount,
    representativeSampleMode: 'tile-selected-pixel',
    representativeSampleContributionLog: contributionLog,
    representativeSampleFinalT: transmittance,
    representativeSampleAccumColor: accumColor,
    representativeSampleResolvedColor: resolvedColor,
    representativeSampleContributionCount: contributionCount,
    representativeSampleAlphaSum: alphaSum,
    representativeSampleContributionSum: contributionSum,
    representativeSampleLastContributedLocalOrder: lastContributedLocalOrder,
    representativeSampleThresholdCrossingCount: thresholdCrossingCount,
    representativeSampleThresholdSkippedCount: thresholdSkippedCount,
    representativeSampleFirstThresholdSkipLocalOrder: firstThresholdSkipLocalOrder,
    representativeSampleFirstThresholdSkipAlpha: firstThresholdSkipAlpha,
    representativeSampleFirstThresholdSkipTransmittanceBefore: firstThresholdSkipTransmittanceBefore,
    representativeSampleFirstThresholdSkipTransmittanceAfter: firstThresholdSkipTransmittanceAfter,
    representativeSampleThresholdSkipPreview: thresholdSkipPreview,
    representativeSampleThresholdSemantics: {
      contract: 'cuda-like-testT-skip-and-stop',
      threshold: TILE_ACCUMULATION_EARLY_OUT_THRESHOLD,
      crossingSplatContributes: false,
      stopsAfterFirstCrossing: true
    },
    representativeSampleEarlyOutTriggered: earlyOutTriggered,
    representativeSampleEarlyOutAtItem: earlyOutAtItem
  };
}

function buildRepresentativeSampleContributionSummary(batch, bgGray01, payload = null) {
  const candidateResult = buildRepresentativeSampleCandidates(batch, payload);
  const candidates = candidateResult.candidates;
  let bestSummary = null;
  let bestScore = -Infinity;
  let evaluatedCandidateCount = 0;

  for (const candidate of candidates) {
    evaluatedCandidateCount++;
    const summary = buildContributionSummaryAtPixel(batch, bgGray01, candidate.pixelIndexPx, {
      ...candidate,
      candidateCount: candidates.length,
      evaluatedCandidateCount,
      usableItemSource: candidateResult.usableItemSource,
      itemReadMode: candidateResult.itemReadMode
    }, payload);
    const score = scoreRepresentativeContributionSummary(summary);
    if (score > bestScore) {
      bestScore = score;
      bestSummary = summary;
    }
  }

  if (bestSummary) {
    bestSummary.representativeSampleEvaluatedCandidateCount = evaluatedCandidateCount;
    return bestSummary;
  }

  return buildContributionSummaryAtPixel(batch, bgGray01, buildBatchRepresentativePixelIndex(batch), {
    selectionMode: 'tile-center-fallback',
    selectionReason: 'no valid candidate pixels',
    candidateCount: 1,
    evaluatedCandidateCount: 1,
    usableItemSource: candidateResult.usableItemSource,
    itemReadMode: candidateResult.itemReadMode
  }, payload);
}

function estimateBatchAccumulationProgress(batch, payload = null) {
  const packedCount = Number.isFinite(batch?.packedCount) ? Math.max(0, batch.packedCount | 0) : 0;
  const clampedCount = Math.min(packedCount, TILE_ACCUMULATION_MAX_ITEMS);
  const representativePixelIndexPx = buildBatchRepresentativePixelIndex(batch);
  const tilePixelAreaEstimate = Array.isArray(batch?.rect) && batch.rect.length >= 4
    ? Math.max(0, ((batch.rect[2] | 0) - (batch.rect[0] | 0)) * ((batch.rect[3] | 0) - (batch.rect[1] | 0)))
    : 0;
  let visitedItems = 0;
  let contributingItems = 0;
  let transmittance = 1.0;
  let earlyOutTriggered = false;
  let earlyOutAtItem = -1;
  let earlyOutAtTransmittance = 1.0;

  for (let i = 0; i < clampedCount; i++) {
    visitedItems++;
    const evaluation = evaluateBatchItemAtPixel(readPackedBatchItem(batch, i, payload), representativePixelIndexPx);
    if (!evaluation.survivesFragment) continue;
    contributingItems++;
    transmittance *= (1.0 - evaluation.finalAlpha);
    if (TILE_ACCUMULATION_EARLY_OUT_ENABLED && transmittance < TILE_ACCUMULATION_EARLY_OUT_THRESHOLD) {
      earlyOutTriggered = true;
      earlyOutAtItem = i;
      earlyOutAtTransmittance = transmittance;
      break;
    }
  }

  return {
    tileId: Number.isFinite(batch?.tileId) ? (batch.tileId | 0) : -1,
    packedCount,
    representativePixelIndexPx,
    tilePixelAreaEstimate,
    visitedItems,
    contributingItems,
    visitedRatio: packedCount > 0 ? visitedItems / packedCount : 0,
    earlyOutTriggered,
    earlyOutAtItem,
    earlyOutAtTransmittance
  };
}

function summarizeBatchSequence(batch) {
  const packedCount = Number.isFinite(batch?.packedCount) ? Math.max(0, batch.packedCount | 0) : 0;
  return {
    tileId: Number.isFinite(batch?.tileId) ? (batch.tileId | 0) : -1,
    itemCount: packedCount,
    batchSpan: Number.isFinite(batch?.batchSpan) ? Math.max(1, batch.batchSpan | 0) : 1,
    sequenceConsistent: batch?.sequenceConsistent !== false,
    orderingMismatchCount: Number.isFinite(batch?.orderingMismatchCount) ? Math.max(0, batch.orderingMismatchCount | 0) : 0,
    firstMismatch: batch?.firstMismatch ?? null,
    orderPreviewHead: Array.isArray(batch?.orderPreviewHead) ? batch.orderPreviewHead : [],
    orderPreviewTail: Array.isArray(batch?.orderPreviewTail) ? batch.orderPreviewTail : [],
    depthPreviewHead: Array.isArray(batch?.depthPreviewHead) ? batch.depthPreviewHead : [],
    depthPreviewTail: Array.isArray(batch?.depthPreviewTail) ? batch.depthPreviewTail : [],
    firstDepth: Number.isFinite(batch?.firstDepth) ? Number(batch.firstDepth) : null,
    lastDepth: Number.isFinite(batch?.lastDepth) ? Number(batch.lastDepth) : null
  };
}

function summarizeAccumulationBatches(batches, bgGray01 = 0, payload = null) {
  let truncatedTileCount = 0;
  let maxObservedTileItems = 0;
  let totalSkippedItems = 0;
  let worstTileId = -1;
  let worstTileItemCount = 0;
  let worstTileSkippedCount = 0;
  let totalTileItems = 0;
  let earlyOutTriggeredTileCount = 0;
  let earlyOutTriggeredPixelEstimate = 0;
  let worstEarlyOutTileId = -1;
  let worstEarlyOutCount = 0;
  let visitedItemsSum = 0;
  let maxVisitedItemsPerTile = 0;
  let weightedVisitedItemsSum = 0;
  let weightedTilePixelArea = 0;
  let visitedRatioSum = 0;
  let minVisitedRatio = 1;
  let maxVisitedRatio = 0;
  let observedTileCount = 0;
  const observedTileSummaries = [];
  const heavyTileSummaries = [];
  const orderingMismatches = [];
  let multiBatchTileCount = 0;
  let maxBatchSpan = 1;
  let representativeTile = null;
  let representativeBatch = null;

  for (const batch of batches) {
    const packedCount = Number.isFinite(batch?.packedCount) ? Math.max(0, batch.packedCount | 0) : 0;
    if (packedCount <= 0) continue;
    observedTileCount++;
    totalTileItems += packedCount;
    if (packedCount > maxObservedTileItems) {
      maxObservedTileItems = packedCount;
    }
    const sequenceSummary = summarizeBatchSequence(batch);
    if (sequenceSummary.batchSpan > 1) multiBatchTileCount++;
    if (sequenceSummary.batchSpan > maxBatchSpan) maxBatchSpan = sequenceSummary.batchSpan;
    if (!representativeTile || sequenceSummary.itemCount > representativeTile.itemCount) {
      representativeTile = sequenceSummary;
      representativeBatch = batch;
    }
    heavyTileSummaries.push(sequenceSummary);
    if (sequenceSummary.orderingMismatchCount > 0) {
      orderingMismatches.push({
        tileId: sequenceSummary.tileId,
        itemCount: sequenceSummary.itemCount,
        batchSpan: sequenceSummary.batchSpan,
        orderingMismatchCount: sequenceSummary.orderingMismatchCount,
        firstMismatch: sequenceSummary.firstMismatch
      });
    }
    const progress = estimateBatchAccumulationProgress(batch, payload);
    visitedItemsSum += progress.visitedItems;
    visitedRatioSum += progress.visitedRatio;
    minVisitedRatio = Math.min(minVisitedRatio, progress.visitedRatio);
    maxVisitedRatio = Math.max(maxVisitedRatio, progress.visitedRatio);
    if (progress.visitedItems > maxVisitedItemsPerTile) {
      maxVisitedItemsPerTile = progress.visitedItems;
    }
    if (progress.tilePixelAreaEstimate > 0) {
      weightedVisitedItemsSum += progress.visitedItems * progress.tilePixelAreaEstimate;
      weightedTilePixelArea += progress.tilePixelAreaEstimate;
    }
    if (progress.earlyOutTriggered) {
      const earlyOutCount = Math.max(0, progress.packedCount - progress.visitedItems);
      earlyOutTriggeredTileCount++;
      earlyOutTriggeredPixelEstimate += progress.tilePixelAreaEstimate;
      if (earlyOutCount > worstEarlyOutCount) {
        worstEarlyOutCount = earlyOutCount;
        worstEarlyOutTileId = progress.tileId;
      }
      observedTileSummaries.push({
        tileId: progress.tileId,
        packedCount: progress.packedCount,
        visitedItems: progress.visitedItems,
        visitedRatio: progress.visitedRatio,
        earlyOutTriggered: true,
        earlyOutAtItem: progress.earlyOutAtItem,
        earlyOutAtTransmittance: progress.earlyOutAtTransmittance
      });
    }
    const skippedCount = Math.max(0, packedCount - TILE_ACCUMULATION_MAX_ITEMS);
    if (skippedCount > 0) {
      truncatedTileCount++;
      totalSkippedItems += skippedCount;
      if (skippedCount > worstTileSkippedCount) {
        worstTileSkippedCount = skippedCount;
        worstTileItemCount = packedCount;
        worstTileId = Number.isFinite(batch?.tileId) ? (batch.tileId | 0) : -1;
      }
    }
  }

  observedTileSummaries.sort((a, b) => {
    const aSkipped = (a.packedCount ?? 0) - (a.visitedItems ?? 0);
    const bSkipped = (b.packedCount ?? 0) - (b.visitedItems ?? 0);
    return bSkipped - aSkipped;
  });
  heavyTileSummaries.sort((a, b) => b.itemCount - a.itemCount);
  orderingMismatches.sort((a, b) => b.orderingMismatchCount - a.orderingMismatchCount);
  const representativeTileSummary = representativeTile ?? summarizeBatchSequence(null);
  const representativeContributionSummary = representativeBatch
    ? buildRepresentativeSampleContributionSummary(representativeBatch, bgGray01, payload)
    : buildRepresentativeSampleContributionSummary(null, bgGray01, payload);
  const inconsistentTileCount = orderingMismatches.length;
  const totalOrderingMismatchCount = orderingMismatches.reduce((sum, item) => sum + (item.orderingMismatchCount ?? 0), 0);

  return {
    accumulationMaxItems: TILE_ACCUMULATION_MAX_ITEMS,
    tileAccumulationTruncatedTileCount: truncatedTileCount,
    tileAccumulationMaxObservedTileItems: maxObservedTileItems,
    tileAccumulationTotalSkippedItems: totalSkippedItems,
    tileAccumulationWorstTileId: worstTileId,
    tileAccumulationWorstTileItemCount: worstTileItemCount,
    tileAccumulationWorstTileSkippedCount: worstTileSkippedCount,
    tileAccumulationEarlyOutEnabled: TILE_ACCUMULATION_EARLY_OUT_ENABLED,
    tileAccumulationEarlyOutThreshold: TILE_ACCUMULATION_EARLY_OUT_THRESHOLD,
    tileAccumulationEarlyOutTriggeredTileCount: earlyOutTriggeredTileCount,
    tileAccumulationEarlyOutTriggeredPixelEstimate: earlyOutTriggeredPixelEstimate,
    tileAccumulationWorstEarlyOutTileId: worstEarlyOutTileId,
    tileAccumulationWorstEarlyOutCount: worstEarlyOutCount,
    tileAccumulationAverageVisitedItemsPerTile: observedTileCount > 0 ? visitedItemsSum / observedTileCount : 0,
    tileAccumulationMaxVisitedItemsPerTile: maxVisitedItemsPerTile,
    tileAccumulationAverageVisitedItemsPerPixelEstimate: weightedTilePixelArea > 0
      ? weightedVisitedItemsSum / weightedTilePixelArea
      : 0,
    tileAccumulationVisitedRatioSummary: {
      min: observedTileCount > 0 ? minVisitedRatio : 0,
      avg: observedTileCount > 0 ? visitedRatioSum / observedTileCount : 0,
      max: observedTileCount > 0 ? maxVisitedRatio : 0
    },
    tileAccumulationObservedTileSummaries: observedTileSummaries.slice(0, 3),
    tileAccumulationOrderingSummary: {
      expectedDepthOrder: 'ascending-near-to-far',
      tilesWithSequenceMetadata: observedTileCount,
      inconsistentTileCount,
      totalOrderingMismatchCount,
      orderingGuaranteedByTilePlanSort: true
    },
    tileAccumulationBatchBoundarySummary: {
      totalTileCount: observedTileCount,
      oneBatchPerTile: multiBatchTileCount === 0,
      multiBatchTileCount,
      maxBatchSpan
    },
    tileAccumulationObservedOrderingMismatches: orderingMismatches.slice(0, 3),
    tileAccumulationHeavyTileSummaries: heavyTileSummaries.slice(0, 3),
    tileAccumulationRepresentativeTileId: representativeTileSummary.tileId,
    tileAccumulationRepresentativeTileItemCount: representativeTileSummary.itemCount,
    tileAccumulationRepresentativeTileOrderPreview: {
      head: representativeTileSummary.orderPreviewHead,
      tail: representativeTileSummary.orderPreviewTail
    },
    tileAccumulationRepresentativeTileDepthPreview: {
      head: representativeTileSummary.depthPreviewHead,
      tail: representativeTileSummary.depthPreviewTail
    },
    tileAccumulationRepresentativeTileBatchSpan: representativeTileSummary.batchSpan,
    tileAccumulationRepresentativeTileSequenceConsistent: representativeTileSummary.sequenceConsistent,
    tileAccumulationContributionSummary: {
      alphaSource: 'colorAlpha[3] * exp(power)',
      contributionContract: 'rgb * alpha * T_before',
      transmittanceUpdateContract: 'T_after = T_before * (1 - alpha)',
      resolveContract: 'accumColor + finalT * backgroundGray',
      thresholdedContributionSkipped: true
    },
    tileAccumulationRepresentativeSampleMode: representativeContributionSummary.representativeSampleMode,
    tileAccumulationRepresentativeSampleSelectionMode: representativeContributionSummary.representativeSampleSelectionMode,
    tileAccumulationRepresentativeSampleSelectionReason: representativeContributionSummary.representativeSampleSelectionReason,
    tileAccumulationRepresentativeSamplePixel: representativeContributionSummary.representativeSamplePixel,
    tileAccumulationRepresentativeSampleHasContribution: representativeContributionSummary.representativeSampleHasContribution,
    tileAccumulationRepresentativeSampleCandidateCount: representativeContributionSummary.representativeSampleCandidateCount,
    tileAccumulationRepresentativeSampleEvaluatedCandidateCount: representativeContributionSummary.representativeSampleEvaluatedCandidateCount,
    tileAccumulationRepresentativeSampleUsableItemSource: representativeContributionSummary.representativeSampleUsableItemSource,
    tileAccumulationRepresentativeSampleItemReadMode: representativeContributionSummary.representativeSampleItemReadMode,
    tileAccumulationRepresentativeSampleEvaluatedItemCount: representativeContributionSummary.representativeSampleEvaluatedItemCount,
    tileAccumulationRepresentativeSampleContributionLog: representativeContributionSummary.representativeSampleContributionLog,
    tileAccumulationRepresentativeSampleFinalT: representativeContributionSummary.representativeSampleFinalT,
    tileAccumulationRepresentativeSampleAccumColor: representativeContributionSummary.representativeSampleAccumColor,
    tileAccumulationRepresentativeSampleResolvedColor: representativeContributionSummary.representativeSampleResolvedColor,
    tileAccumulationRepresentativeSampleContributionCount: representativeContributionSummary.representativeSampleContributionCount,
    tileAccumulationRepresentativeSampleAlphaSum: representativeContributionSummary.representativeSampleAlphaSum,
    tileAccumulationRepresentativeSampleContributionSum: representativeContributionSummary.representativeSampleContributionSum,
    tileAccumulationRepresentativeSampleLastContributedLocalOrder: representativeContributionSummary.representativeSampleLastContributedLocalOrder,
    tileAccumulationRepresentativeSampleThresholdCrossingCount: representativeContributionSummary.representativeSampleThresholdCrossingCount,
    tileAccumulationRepresentativeSampleThresholdSkippedCount: representativeContributionSummary.representativeSampleThresholdSkippedCount,
    tileAccumulationRepresentativeSampleFirstThresholdSkipLocalOrder: representativeContributionSummary.representativeSampleFirstThresholdSkipLocalOrder,
    tileAccumulationRepresentativeSampleFirstThresholdSkipAlpha: representativeContributionSummary.representativeSampleFirstThresholdSkipAlpha,
    tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceBefore: representativeContributionSummary.representativeSampleFirstThresholdSkipTransmittanceBefore,
    tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceAfter: representativeContributionSummary.representativeSampleFirstThresholdSkipTransmittanceAfter,
    tileAccumulationRepresentativeSampleThresholdSkipPreview: representativeContributionSummary.representativeSampleThresholdSkipPreview,
    tileAccumulationRepresentativeSampleThresholdSemantics: representativeContributionSummary.representativeSampleThresholdSemantics,
    tileAccumulationContractVersion: TILE_ACCUMULATION_CONTRACT_VERSION,
    tileAccumulationTruncationRatio: totalTileItems > 0 ? totalSkippedItems / totalTileItems : 0
  };
}

function buildAccumulationUploadSummary(payload, accumulationStats) {
  return {
    packedUploadLayoutVersion: 0,
    packedUploadStrideBytes: 0,
    packedUploadBytes: payload?.packedFloats?.byteLength ?? 0,
    packedUploadCount: payload?.totalItemCount ?? 0,
    packedUploadLength: payload?.packedFloats?.length ?? 0,
    packedUploadCapacityBytes: payload?.packedFloats?.byteLength ?? 0,
    packedUploadReusedCapacity: false,
    packedUploadManagedCapacityReused: false,
    packedUploadManagedCapacityGrown: false,
    packedUploadManagedUploadCount: 1,
    packedUploadAlphaSource: 'colorAlpha[3]',
    packedDirectDraw: false,
    packedDirectUsesGpuResidentPayload: false,
    packedInterleavedBound: false,
    packedDirectTileComposite: false,
    packedDirectCompositingContract: 'tile-local-explicit-front-to-back',
    tileCompositePrimitive: 'tile-rect-explicit-accumulation',
    tileCompositeRectContract: 'tile-rect-screen-space',
    tileAccumulationPayloadContract: payload?.summary?.payloadContract ?? 'tile-plan-to-accumulation-texture',
    tileAccumulationMaxItemsPerTile: accumulationStats?.accumulationMaxItems ?? TILE_ACCUMULATION_MAX_ITEMS,
    tileAccumulationTruncatedBatchCount: accumulationStats?.tileAccumulationTruncatedTileCount ?? 0,
    tileAccumulationTotalSkippedItems: accumulationStats?.tileAccumulationTotalSkippedItems ?? 0,
    tileAccumulationWorstTileId: accumulationStats?.tileAccumulationWorstTileId ?? -1,
    tileAccumulationWorstTileSkippedCount: accumulationStats?.tileAccumulationWorstTileSkippedCount ?? 0,
    tileAccumulationEarlyOutTriggeredTileCount: accumulationStats?.tileAccumulationEarlyOutTriggeredTileCount ?? 0,
    tileAccumulationWorstEarlyOutTileId: accumulationStats?.tileAccumulationWorstEarlyOutTileId ?? -1,
    tileAccumulationWorstEarlyOutCount: accumulationStats?.tileAccumulationWorstEarlyOutCount ?? 0,
    tileAccumulationOrderingSummary: accumulationStats?.tileAccumulationOrderingSummary ?? null,
    tileAccumulationBatchBoundarySummary: accumulationStats?.tileAccumulationBatchBoundarySummary ?? null,
    tileAccumulationObservedOrderingMismatches: accumulationStats?.tileAccumulationObservedOrderingMismatches ?? [],
    tileAccumulationHeavyTileSummaries: accumulationStats?.tileAccumulationHeavyTileSummaries ?? [],
    tileAccumulationRepresentativeTileId: accumulationStats?.tileAccumulationRepresentativeTileId ?? -1,
    tileAccumulationRepresentativeTileItemCount: accumulationStats?.tileAccumulationRepresentativeTileItemCount ?? 0,
    tileAccumulationRepresentativeTileOrderPreview: accumulationStats?.tileAccumulationRepresentativeTileOrderPreview ?? null,
    tileAccumulationRepresentativeTileDepthPreview: accumulationStats?.tileAccumulationRepresentativeTileDepthPreview ?? null,
    tileAccumulationRepresentativeTileBatchSpan: accumulationStats?.tileAccumulationRepresentativeTileBatchSpan ?? 1,
    tileAccumulationRepresentativeTileSequenceConsistent: accumulationStats?.tileAccumulationRepresentativeTileSequenceConsistent ?? false,
    tileAccumulationContributionSummary: accumulationStats?.tileAccumulationContributionSummary ?? null,
    tileAccumulationRepresentativeSampleMode: accumulationStats?.tileAccumulationRepresentativeSampleMode ?? 'none',
    tileAccumulationRepresentativeSampleSelectionMode: accumulationStats?.tileAccumulationRepresentativeSampleSelectionMode ?? 'none',
    tileAccumulationRepresentativeSampleSelectionReason: accumulationStats?.tileAccumulationRepresentativeSampleSelectionReason ?? 'none',
    tileAccumulationRepresentativeSamplePixel: accumulationStats?.tileAccumulationRepresentativeSamplePixel ?? [0, 0],
    tileAccumulationRepresentativeSampleHasContribution: !!accumulationStats?.tileAccumulationRepresentativeSampleHasContribution,
    tileAccumulationRepresentativeSampleCandidateCount: accumulationStats?.tileAccumulationRepresentativeSampleCandidateCount ?? 0,
    tileAccumulationRepresentativeSampleEvaluatedCandidateCount: accumulationStats?.tileAccumulationRepresentativeSampleEvaluatedCandidateCount ?? 0,
    tileAccumulationRepresentativeSampleUsableItemSource: accumulationStats?.tileAccumulationRepresentativeSampleUsableItemSource ?? 'none',
    tileAccumulationRepresentativeSampleItemReadMode: accumulationStats?.tileAccumulationRepresentativeSampleItemReadMode ?? 'none',
    tileAccumulationRepresentativeSampleEvaluatedItemCount: accumulationStats?.tileAccumulationRepresentativeSampleEvaluatedItemCount ?? 0,
    tileAccumulationRepresentativeSampleContributionLog: accumulationStats?.tileAccumulationRepresentativeSampleContributionLog ?? [],
    tileAccumulationRepresentativeSampleFinalT: accumulationStats?.tileAccumulationRepresentativeSampleFinalT ?? 1,
    tileAccumulationRepresentativeSampleAccumColor: accumulationStats?.tileAccumulationRepresentativeSampleAccumColor ?? [0, 0, 0],
    tileAccumulationRepresentativeSampleResolvedColor: accumulationStats?.tileAccumulationRepresentativeSampleResolvedColor ?? [0, 0, 0],
    tileAccumulationRepresentativeSampleContributionCount: accumulationStats?.tileAccumulationRepresentativeSampleContributionCount ?? 0,
    tileAccumulationRepresentativeSampleAlphaSum: accumulationStats?.tileAccumulationRepresentativeSampleAlphaSum ?? 0,
    tileAccumulationRepresentativeSampleContributionSum: accumulationStats?.tileAccumulationRepresentativeSampleContributionSum ?? [0, 0, 0],
    tileAccumulationRepresentativeSampleLastContributedLocalOrder: accumulationStats?.tileAccumulationRepresentativeSampleLastContributedLocalOrder ?? -1,
    tileAccumulationRepresentativeSampleThresholdCrossingCount: accumulationStats?.tileAccumulationRepresentativeSampleThresholdCrossingCount ?? 0,
    tileAccumulationRepresentativeSampleThresholdSkippedCount: accumulationStats?.tileAccumulationRepresentativeSampleThresholdSkippedCount ?? 0,
    tileAccumulationRepresentativeSampleFirstThresholdSkipLocalOrder: accumulationStats?.tileAccumulationRepresentativeSampleFirstThresholdSkipLocalOrder ?? -1,
    tileAccumulationRepresentativeSampleFirstThresholdSkipAlpha: accumulationStats?.tileAccumulationRepresentativeSampleFirstThresholdSkipAlpha ?? 0,
    tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceBefore: accumulationStats?.tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceBefore ?? 1,
    tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceAfter: accumulationStats?.tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceAfter ?? 1,
    tileAccumulationRepresentativeSampleThresholdSkipPreview: accumulationStats?.tileAccumulationRepresentativeSampleThresholdSkipPreview ?? [],
    tileAccumulationRepresentativeSampleThresholdSemantics: accumulationStats?.tileAccumulationRepresentativeSampleThresholdSemantics ?? null,
    tileAccumulationRepresentativeSampleFramebufferColor: accumulationStats?.tileAccumulationRepresentativeSampleFramebufferColor ?? [0, 0, 0],
    tileAccumulationRepresentativeSampleFramebufferReadbackValid: !!accumulationStats?.tileAccumulationRepresentativeSampleFramebufferReadbackValid,
    tileAccumulationRepresentativeSampleFramebufferReadbackReason: accumulationStats?.tileAccumulationRepresentativeSampleFramebufferReadbackReason ?? 'not-attempted',
    tileAccumulationRepresentativeSampleFramebufferReadbackPixel: accumulationStats?.tileAccumulationRepresentativeSampleFramebufferReadbackPixel ?? [0, 0],
    tileAccumulationRepresentativeSampleFramebufferReadbackGlPixel: accumulationStats?.tileAccumulationRepresentativeSampleFramebufferReadbackGlPixel ?? [0, 0],
    tileAccumulationRepresentativeSampleFramebufferReadbackRgba8: accumulationStats?.tileAccumulationRepresentativeSampleFramebufferReadbackRgba8 ?? [0, 0, 0, 0],
    tileAccumulationRepresentativeSampleResolvedColorDelta: accumulationStats?.tileAccumulationRepresentativeSampleResolvedColorDelta ?? [0, 0, 0],
    tileAccumulationRepresentativeSampleResolvedColorDeltaAbsMax: accumulationStats?.tileAccumulationRepresentativeSampleResolvedColorDeltaAbsMax ?? 0,
    tileAccumulationRepresentativeSampleResolvedColorMatchesFramebuffer: !!accumulationStats?.tileAccumulationRepresentativeSampleResolvedColorMatchesFramebuffer,
    tileAccumulationRepresentativeSampleResolvedColorMatchTolerance: accumulationStats?.tileAccumulationRepresentativeSampleResolvedColorMatchTolerance ?? TILE_ACCUMULATION_FRAMEBUFFER_MATCH_TOLERANCE,
    tileAccumulationContractVersion: accumulationStats?.tileAccumulationContractVersion ?? TILE_ACCUMULATION_CONTRACT_VERSION
  };
}

export function executeTileAccumulationDraw({
  gl,
  gpu,
  canvas,
  accumulationPayload,
  drawPathSelection,
  bgGray01 = 0
}) {
  const state = ensureTileAccumulationState(gl, gpu);
  const batches = Array.isArray(accumulationPayload?.batchMetadata) ? accumulationPayload.batchMetadata : [];
  const previousScissorEnabled = gl.isEnabled(gl.SCISSOR_TEST);
  const previousBlendEnabled = gl.isEnabled(gl.BLEND);
  const previousFramebuffer = gl.getParameter(gl.FRAMEBUFFER_BINDING);

  let drawCallCount = 0;
  let nonEmptyTileBatchCount = 0;
  let totalTileDrawCount = 0;
  const accumulationStats = summarizeAccumulationBatches(batches, bgGray01, accumulationPayload);
  const textureValidation = validateTextureAllocation(gl, accumulationPayload);

  disableDepth(gl);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.disable(gl.BLEND);
  gl.enable(gl.SCISSOR_TEST);
  clearToGray(gl, bgGray01);
  if (textureValidation.textureAllocationValid) {
    uploadAccumulationPayloadTexture(gl, state, accumulationPayload, textureValidation);
  }

  try {
    if (textureValidation.textureAllocationValid) {
      for (const batch of batches) {
        if (!Number.isFinite(batch?.packedCount) || batch.packedCount <= 0) continue;
        drawAccumulationBatch(gl, state, canvas, batch, bgGray01);
        drawCallCount++;
        nonEmptyTileBatchCount++;
        totalTileDrawCount += batch.packedCount | 0;
      }
      Object.assign(
        accumulationStats,
        readRepresentativeSampleFramebufferColor(gl, canvas, accumulationStats)
      );
    } else {
      Object.assign(
        accumulationStats,
        buildFramebufferReadbackSummary(`texture-allocation-invalid:${textureValidation.textureAllocationFailureReason}`)
      );
    }
  } finally {
    gl.bindFramebuffer(gl.FRAMEBUFFER, previousFramebuffer);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindVertexArray(null);
    if (previousScissorEnabled) gl.enable(gl.SCISSOR_TEST);
    else gl.disable(gl.SCISSOR_TEST);
    if (previousBlendEnabled) gl.enable(gl.BLEND);
    else gl.disable(gl.BLEND);
  }

  return {
    executionSummary: {
      tileBatchCount: batches.length,
      nonEmptyTileBatchCount,
      totalTileDrawCount,
      uploadCount: accumulationPayload?.totalItemCount > 0 ? 1 : 0,
      drawCallCount,
      requestedDrawPath: drawPathSelection?.requestedPath ?? 'packed',
      actualDrawPath: drawPathSelection?.actualPath ?? 'packed',
      drawPathFallbackReason: drawPathSelection?.fallbackReason ?? 'none',
      compositingContract: 'tile-local-explicit-front-to-back',
      tileCompositePath: 'accumulation',
      tileCompositePrimitive: 'tile-rect-explicit-accumulation',
      tileCompositeRectContract: 'tile-rect-screen-space',
      accumulationPayloadContract: accumulationPayload?.summary?.payloadContract ?? 'tile-plan-to-accumulation-texture',
      accumulationMaxItemsPerTile: accumulationStats.accumulationMaxItems,
      accumulationTruncatedBatchCount: accumulationStats.tileAccumulationTruncatedTileCount,
      tileAccumulationTruncatedTileCount: accumulationStats.tileAccumulationTruncatedTileCount,
      tileAccumulationMaxObservedTileItems: accumulationStats.tileAccumulationMaxObservedTileItems,
      tileAccumulationTotalSkippedItems: accumulationStats.tileAccumulationTotalSkippedItems,
      tileAccumulationWorstTileId: accumulationStats.tileAccumulationWorstTileId,
      tileAccumulationWorstTileItemCount: accumulationStats.tileAccumulationWorstTileItemCount,
      tileAccumulationWorstTileSkippedCount: accumulationStats.tileAccumulationWorstTileSkippedCount,
      tileAccumulationEarlyOutEnabled: accumulationStats.tileAccumulationEarlyOutEnabled,
      tileAccumulationEarlyOutThreshold: accumulationStats.tileAccumulationEarlyOutThreshold,
      tileAccumulationEarlyOutTriggeredTileCount: accumulationStats.tileAccumulationEarlyOutTriggeredTileCount,
      tileAccumulationEarlyOutTriggeredPixelEstimate: accumulationStats.tileAccumulationEarlyOutTriggeredPixelEstimate,
      tileAccumulationWorstEarlyOutTileId: accumulationStats.tileAccumulationWorstEarlyOutTileId,
      tileAccumulationWorstEarlyOutCount: accumulationStats.tileAccumulationWorstEarlyOutCount,
      tileAccumulationAverageVisitedItemsPerTile: accumulationStats.tileAccumulationAverageVisitedItemsPerTile,
      tileAccumulationMaxVisitedItemsPerTile: accumulationStats.tileAccumulationMaxVisitedItemsPerTile,
      tileAccumulationAverageVisitedItemsPerPixelEstimate: accumulationStats.tileAccumulationAverageVisitedItemsPerPixelEstimate,
      tileAccumulationVisitedRatioSummary: accumulationStats.tileAccumulationVisitedRatioSummary,
      tileAccumulationObservedTileSummaries: accumulationStats.tileAccumulationObservedTileSummaries,
      tileAccumulationOrderingSummary: accumulationStats.tileAccumulationOrderingSummary,
      tileAccumulationBatchBoundarySummary: accumulationStats.tileAccumulationBatchBoundarySummary,
      tileAccumulationObservedOrderingMismatches: accumulationStats.tileAccumulationObservedOrderingMismatches,
      tileAccumulationHeavyTileSummaries: accumulationStats.tileAccumulationHeavyTileSummaries,
      tileAccumulationRepresentativeTileId: accumulationStats.tileAccumulationRepresentativeTileId,
      tileAccumulationRepresentativeTileItemCount: accumulationStats.tileAccumulationRepresentativeTileItemCount,
      tileAccumulationRepresentativeTileOrderPreview: accumulationStats.tileAccumulationRepresentativeTileOrderPreview,
      tileAccumulationRepresentativeTileDepthPreview: accumulationStats.tileAccumulationRepresentativeTileDepthPreview,
      tileAccumulationRepresentativeTileBatchSpan: accumulationStats.tileAccumulationRepresentativeTileBatchSpan,
      tileAccumulationRepresentativeTileSequenceConsistent: accumulationStats.tileAccumulationRepresentativeTileSequenceConsistent,
      tileAccumulationContributionSummary: accumulationStats.tileAccumulationContributionSummary,
      tileAccumulationRepresentativeSampleMode: accumulationStats.tileAccumulationRepresentativeSampleMode,
      tileAccumulationRepresentativeSampleSelectionMode: accumulationStats.tileAccumulationRepresentativeSampleSelectionMode,
      tileAccumulationRepresentativeSampleSelectionReason: accumulationStats.tileAccumulationRepresentativeSampleSelectionReason,
      tileAccumulationRepresentativeSamplePixel: accumulationStats.tileAccumulationRepresentativeSamplePixel,
      tileAccumulationRepresentativeSampleHasContribution: accumulationStats.tileAccumulationRepresentativeSampleHasContribution,
      tileAccumulationRepresentativeSampleCandidateCount: accumulationStats.tileAccumulationRepresentativeSampleCandidateCount,
      tileAccumulationRepresentativeSampleEvaluatedCandidateCount: accumulationStats.tileAccumulationRepresentativeSampleEvaluatedCandidateCount,
      tileAccumulationRepresentativeSampleUsableItemSource: accumulationStats.tileAccumulationRepresentativeSampleUsableItemSource,
      tileAccumulationRepresentativeSampleItemReadMode: accumulationStats.tileAccumulationRepresentativeSampleItemReadMode,
      tileAccumulationRepresentativeSampleEvaluatedItemCount: accumulationStats.tileAccumulationRepresentativeSampleEvaluatedItemCount,
      tileAccumulationRepresentativeSampleContributionLog: accumulationStats.tileAccumulationRepresentativeSampleContributionLog,
      tileAccumulationRepresentativeSampleFinalT: accumulationStats.tileAccumulationRepresentativeSampleFinalT,
      tileAccumulationRepresentativeSampleAccumColor: accumulationStats.tileAccumulationRepresentativeSampleAccumColor,
      tileAccumulationRepresentativeSampleResolvedColor: accumulationStats.tileAccumulationRepresentativeSampleResolvedColor,
      tileAccumulationRepresentativeSampleContributionCount: accumulationStats.tileAccumulationRepresentativeSampleContributionCount,
      tileAccumulationRepresentativeSampleAlphaSum: accumulationStats.tileAccumulationRepresentativeSampleAlphaSum,
      tileAccumulationRepresentativeSampleContributionSum: accumulationStats.tileAccumulationRepresentativeSampleContributionSum,
      tileAccumulationRepresentativeSampleLastContributedLocalOrder: accumulationStats.tileAccumulationRepresentativeSampleLastContributedLocalOrder,
      tileAccumulationRepresentativeSampleThresholdCrossingCount: accumulationStats.tileAccumulationRepresentativeSampleThresholdCrossingCount,
      tileAccumulationRepresentativeSampleThresholdSkippedCount: accumulationStats.tileAccumulationRepresentativeSampleThresholdSkippedCount,
      tileAccumulationRepresentativeSampleFirstThresholdSkipLocalOrder: accumulationStats.tileAccumulationRepresentativeSampleFirstThresholdSkipLocalOrder,
      tileAccumulationRepresentativeSampleFirstThresholdSkipAlpha: accumulationStats.tileAccumulationRepresentativeSampleFirstThresholdSkipAlpha,
      tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceBefore: accumulationStats.tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceBefore,
      tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceAfter: accumulationStats.tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceAfter,
      tileAccumulationRepresentativeSampleThresholdSkipPreview: accumulationStats.tileAccumulationRepresentativeSampleThresholdSkipPreview,
      tileAccumulationRepresentativeSampleThresholdSemantics: accumulationStats.tileAccumulationRepresentativeSampleThresholdSemantics,
      tileAccumulationRepresentativeSampleFramebufferColor: accumulationStats.tileAccumulationRepresentativeSampleFramebufferColor,
      tileAccumulationRepresentativeSampleFramebufferReadbackValid: accumulationStats.tileAccumulationRepresentativeSampleFramebufferReadbackValid,
      tileAccumulationRepresentativeSampleFramebufferReadbackReason: accumulationStats.tileAccumulationRepresentativeSampleFramebufferReadbackReason,
      tileAccumulationRepresentativeSampleFramebufferReadbackPixel: accumulationStats.tileAccumulationRepresentativeSampleFramebufferReadbackPixel,
      tileAccumulationRepresentativeSampleFramebufferReadbackGlPixel: accumulationStats.tileAccumulationRepresentativeSampleFramebufferReadbackGlPixel,
      tileAccumulationRepresentativeSampleFramebufferReadbackRgba8: accumulationStats.tileAccumulationRepresentativeSampleFramebufferReadbackRgba8,
      tileAccumulationRepresentativeSampleResolvedColorDelta: accumulationStats.tileAccumulationRepresentativeSampleResolvedColorDelta,
      tileAccumulationRepresentativeSampleResolvedColorDeltaAbsMax: accumulationStats.tileAccumulationRepresentativeSampleResolvedColorDeltaAbsMax,
      tileAccumulationRepresentativeSampleResolvedColorMatchesFramebuffer: accumulationStats.tileAccumulationRepresentativeSampleResolvedColorMatchesFramebuffer,
      tileAccumulationRepresentativeSampleResolvedColorMatchTolerance: accumulationStats.tileAccumulationRepresentativeSampleResolvedColorMatchTolerance,
      tileAccumulationContractVersion: accumulationStats.tileAccumulationContractVersion,
      tileAccumulationTruncationRatio: accumulationStats.tileAccumulationTruncationRatio,
      requestedTextureWidth: textureValidation.requestedTextureWidth,
      requestedTextureHeight: textureValidation.requestedTextureHeight,
      validatedTextureWidth: textureValidation.validatedTextureWidth,
      validatedTextureHeight: textureValidation.validatedTextureHeight,
      textureAllocationValid: textureValidation.textureAllocationValid,
      textureAllocationFailureReason: textureValidation.textureAllocationFailureReason
    },
    packedUploadSummary: buildAccumulationUploadSummary(accumulationPayload, accumulationStats),
    tileCompositeDrawInfo: {
      drawCount: totalTileDrawCount,
      drawCallCount,
      uploadCount: accumulationPayload?.totalItemCount > 0 ? 1 : 0,
      primitive: 'tile-rect-explicit-accumulation',
      rectContract: 'tile-rect-screen-space'
    }
  };
}
