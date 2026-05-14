import { buildVisibleItemForCandidate } from './gpu_visible_item_builder.js';

const stateByGl = new WeakMap();
const DEFAULT_MAX_COUNT = 65536;
const DEFAULT_MIN_RADIUS_PX = 0.25;
const DEPTH_MODE_VALUES = new Set(['positive', 'any']);

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) || 'unknown shader compile error';
    gl.deleteShader(shader);
    throw new Error(log);
  }
  return shader;
}

function createScreenCoarseState(gl) {
  const vertexSource = `#version 300 es
precision highp float;
in uint aCandidateIndex;
in uint aItemValid;
in vec4 aCoarse;
flat out uint vCandidateIndex;
flat out uint vValidFlag;
uniform float uCanvasWidth;
uniform float uCanvasHeight;
uniform float uMinRadiusPx;
uniform uint uRequireInViewport;
uniform uint uDepthMode;
uniform uint uFilterMode;
bool passesViewport(float px, float py, float radiusPx) {
  return px + radiusPx >= 0.0 &&
    px - radiusPx <= uCanvasWidth - 1.0 &&
    py + radiusPx >= 0.0 &&
    py - radiusPx <= uCanvasHeight - 1.0;
}
void main() {
  float px = aCoarse.x;
  float py = aCoarse.y;
  float radiusPx = aCoarse.z;
  float depth = aCoarse.w;
  bool ok = aItemValid != 0u;
  ok = ok && radiusPx >= uMinRadiusPx;
  ok = ok && (uRequireInViewport == 0u || passesViewport(px, py, radiusPx));
  ok = ok && (uDepthMode != 1u || depth > 0.0);
  ok = ok && (uFilterMode != 1u || (aCandidateIndex & 1u) == 0u);
  vCandidateIndex = aCandidateIndex;
  vValidFlag = ok ? 1u : 0u;
  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}`;
  const fragmentSource = `#version 300 es
precision highp float;
out vec4 outColor;
void main() {
  outColor = vec4(0.0);
}`;
  const vertex = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.transformFeedbackVaryings(program, ['vCandidateIndex', 'vValidFlag'], gl.INTERLEAVED_ATTRIBS);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) || 'unknown screenCoarse candidate program link error';
    gl.deleteProgram(program);
    throw new Error(log);
  }
  return {
    program,
    vao: gl.createVertexArray(),
    candidateBuffer: gl.createBuffer(),
    itemValidBuffer: gl.createBuffer(),
    coarseBuffer: gl.createBuffer(),
    outputBuffer: gl.createBuffer(),
    transformFeedback: gl.createTransformFeedback(),
    aCandidateIndex: gl.getAttribLocation(program, 'aCandidateIndex'),
    aItemValid: gl.getAttribLocation(program, 'aItemValid'),
    aCoarse: gl.getAttribLocation(program, 'aCoarse'),
    uCanvasWidth: gl.getUniformLocation(program, 'uCanvasWidth'),
    uCanvasHeight: gl.getUniformLocation(program, 'uCanvasHeight'),
    uMinRadiusPx: gl.getUniformLocation(program, 'uMinRadiusPx'),
    uRequireInViewport: gl.getUniformLocation(program, 'uRequireInViewport'),
    uDepthMode: gl.getUniformLocation(program, 'uDepthMode'),
    uFilterMode: gl.getUniformLocation(program, 'uFilterMode')
  };
}

function getScreenCoarseState(gl) {
  let state = stateByGl.get(gl);
  if (!state) {
    state = createScreenCoarseState(gl);
    stateByGl.set(gl, state);
  }
  return state;
}

function clonePlainObject(value) {
  if (!value || typeof value !== 'object') return value;
  if (Array.isArray(value)) return value.map((item) => clonePlainObject(item));
  const out = {};
  for (const [key, item] of Object.entries(value)) out[key] = clonePlainObject(item);
  return out;
}

function toUint32Array(value) {
  if (value instanceof Uint32Array) return value;
  if (Array.isArray(value)) return Uint32Array.from(value);
  return new Uint32Array(0);
}

function toFiniteInteger(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(0, n | 0) : fallback;
}

function toFiniteNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function normalizeDepthMode(value) {
  return DEPTH_MODE_VALUES.has(value) ? value : 'positive';
}

function normalizeFilterMode(value) {
  return value === 'evenIndex' ? 'evenIndex' : 'all-valid';
}

function filterModeToShaderCode(filterMode) {
  return normalizeFilterMode(filterMode) === 'evenIndex' ? 1 : 0;
}

function depthModeToShaderCode(depthMode) {
  return normalizeDepthMode(depthMode) === 'positive' ? 1 : 0;
}

function isFilterAccepted(candidateIndex, filterMode) {
  if (normalizeFilterMode(filterMode) === 'evenIndex') return (candidateIndex & 1) === 0;
  return true;
}

function getWebGlErrorName(gl, errorCode) {
  if (errorCode === gl.NO_ERROR) return 'NO_ERROR';
  if (errorCode === gl.INVALID_ENUM) return 'INVALID_ENUM';
  if (errorCode === gl.INVALID_VALUE) return 'INVALID_VALUE';
  if (errorCode === gl.INVALID_OPERATION) return 'INVALID_OPERATION';
  if (errorCode === gl.INVALID_FRAMEBUFFER_OPERATION) return 'INVALID_FRAMEBUFFER_OPERATION';
  if (errorCode === gl.OUT_OF_MEMORY) return 'OUT_OF_MEMORY';
  if (errorCode === gl.CONTEXT_LOST_WEBGL) return 'CONTEXT_LOST_WEBGL';
  return `WEBGL_ERROR_${errorCode}`;
}

function drainWebGlErrors(gl) {
  if (!gl || typeof gl.getError !== 'function') return;
  let guard = 0;
  while (gl.getError() !== gl.NO_ERROR && guard < 16) guard++;
}

function throwForWebGlErrorCode(gl, phase, errorCode) {
  if (errorCode !== gl.NO_ERROR) {
    throw new Error(`${phase}:${getWebGlErrorName(gl, errorCode)}`);
  }
}

function passesViewport(item, canvasWidth, canvasHeight) {
  return item.px + item.radius >= 0 &&
    item.px - item.radius <= canvasWidth - 1 &&
    item.py + item.radius >= 0 &&
    item.py - item.radius <= canvasHeight - 1;
}

function passesDepth(item, depthMode) {
  if (normalizeDepthMode(depthMode) === 'positive') return item.depth > 0;
  return true;
}

function passesScreenCoarsePredicate(item, config) {
  if (!item) return false;
  if (!(item.radius >= config.minRadiusPx)) return false;
  if (config.requireInViewport && !passesViewport(item, config.canvasWidth, config.canvasHeight)) return false;
  if (!passesDepth(item, config.depthMode)) return false;
  return true;
}

function buildFilterSummary({
  filterMode = 'all-valid',
  requestedCount = 0,
  emittedCount = 0,
  validCount = 0,
  rejectedCount = 0,
  generatedOnGpu = false,
  status = 'ok',
  reason = 'ok'
} = {}) {
  return {
    schemaVersion: 'step94-gpu-candidate-filter-summary-v1',
    filterMode: normalizeFilterMode(filterMode),
    predicate: normalizeFilterMode(filterMode) === 'evenIndex'
      ? {
          type: 'evenIndex',
          expression: 'candidateIndex % 2 === 0',
          divisor: 2,
          remainder: 0
        }
      : {
          type: 'all-valid',
          expression: 'true'
        },
    contract: 'candidate-index-plus-valid-flag',
    status,
    generatedOnGpu,
    requestedCount,
    emittedCount,
    validCount,
    rejectedCount,
    reason
  };
}

function buildRangeSummary(referenceCandidateInfo, count, rawTotal) {
  const base = clonePlainObject(referenceCandidateInfo?.rangeSummary) ?? {};
  const total = Number.isFinite(base.totalCount) ? base.totalCount : rawTotal;
  return {
    ...base,
    totalCount: total,
    candidateCount: count,
    rangeCount: count,
    candidateFraction: total > 0 ? count / total : 0,
    rangeFraction: total > 0 ? count / total : 0
  };
}

function buildFailureCandidateInfo({
  raw = null,
  referenceCandidateInfo = null,
  reason,
  screenCoarseSummary = null,
  filterMode = 'all-valid',
  candidateMode = 'gpu-screen-coarse-unavailable'
} = {}) {
  return {
    candidateIndices: new Uint32Array(0),
    candidateMode,
    validCount: 0,
    rejectedCount: 0,
    filterMode: normalizeFilterMode(filterMode),
    filterSummary: buildFilterSummary({
      filterMode,
      requestedCount: 0,
      emittedCount: 0,
      validCount: 0,
      rejectedCount: 0,
      generatedOnGpu: candidateMode.startsWith('gpu-'),
      status: 'fallback',
      reason
    }),
    rangeSummary: buildRangeSummary(referenceCandidateInfo, 0, raw?.N ?? 0),
    screenCoarseSummary,
    gpuCandidateSummary: candidateMode.startsWith('gpu-')
      ? {
          schemaVersion: 'step107-gpu-screen-coarse-candidate-summary-v1',
          status: 'fallback',
          reason,
          sourceMode: 'screenCoarse',
          generatedOnGpu: true,
          candidateCount: 0,
          screenCoarseSummary
        }
      : null
  };
}

export function buildScreenCoarseSourceConfig(runtimeConfig = {}) {
  return {
    maxCount: toFiniteInteger(runtimeConfig.screenCoarseMaxCount, DEFAULT_MAX_COUNT),
    minRadiusPx: Math.max(0, toFiniteNumber(runtimeConfig.screenCoarseMinRadiusPx, DEFAULT_MIN_RADIUS_PX)),
    requireInViewport: runtimeConfig.screenCoarseRequireInViewport !== false,
    depthMode: normalizeDepthMode(runtimeConfig.screenCoarseDepthMode),
    canvasWidth: toFiniteNumber(runtimeConfig.screenCoarseCanvasWidth, 0),
    canvasHeight: toFiniteNumber(runtimeConfig.screenCoarseCanvasHeight, 0)
  };
}

export function buildCpuScreenCoarseCandidateSourceInfo({
  raw = null,
  referenceCandidateInfo = null,
  runtimeConfig = {},
  filterMode = 'all-valid',
  camera = null,
  screenSpaceCamera = null,
  canvasWidth = 0,
  canvasHeight = 0,
  camPos = null,
  tileGrid = null,
  buildConfig = {},
  candidateMode = 'cpu-screen-coarse-candidate-source-reference'
} = {}) {
  if (!raw || !camera || !buildConfig) {
    return buildFailureCandidateInfo({
      raw,
      referenceCandidateInfo,
      reason: 'screen-coarse-input-unavailable',
      filterMode,
      candidateMode
    });
  }
  const renderScale = Number.isFinite(buildConfig.renderScale) ? buildConfig.renderScale : 1;
  const renderW = Math.max(1, Math.round(canvasWidth * renderScale));
  const renderH = Math.max(1, Math.round(canvasHeight * renderScale));
  const sx = canvasWidth / renderW;
  const sy = canvasHeight / renderH;
  const config = buildScreenCoarseSourceConfig({
    ...runtimeConfig,
    screenCoarseCanvasWidth: canvasWidth,
    screenCoarseCanvasHeight: canvasHeight
  });
  const sourceIndices = toUint32Array(referenceCandidateInfo?.candidateIndices);
  const flags = {
    nativeRot4d: !!buildConfig.useNativeRot4d,
    nativeMarginal: !!buildConfig.useNativeMarginal
  };
  const emittedCandidateIndices = [];
  const emittedItemValid = [];
  const emittedCoarse = [];
  const accepted = [];
  let itemRejectedCount = 0;
  let coarseRejectedCount = 0;
  let filterRejectedCount = 0;

  for (let k = 0; k < sourceIndices.length; k++) {
    const index = sourceIndices[k] >>> 0;
    const item = buildVisibleItemForCandidate({
      raw,
      index,
      camera,
      screenSpaceCamera,
      renderW,
      renderH,
      canvasWidth,
      canvasHeight,
      sx,
      sy,
      timestamp: buildConfig.timestamp,
      scalingModifier: buildConfig.scalingModifier,
      sigmaScale: buildConfig.sigmaScale,
      prefilterVar: buildConfig.prefilterVar,
      useRot4d: buildConfig.useRot4d,
      flags,
      camPos,
      timeDuration: buildConfig.timeDuration,
      useSH: buildConfig.useSH,
      forceSh3d: buildConfig.forceSh3d,
      tileGrid
    });
    const itemValid = item ? 1 : 0;
    const coarseItem = item
      ? {
          px: Math.fround(item.px),
          py: Math.fround(item.py),
          radius: Math.fround(item.radius),
          depth: Math.fround(item.depth)
        }
      : null;
    const coarseAccepted = itemValid === 1 && passesScreenCoarsePredicate(coarseItem, config);
    const filterAccepted = coarseAccepted && isFilterAccepted(index, filterMode);
    emittedCandidateIndices.push(index);
    emittedItemValid.push(itemValid);
    emittedCoarse.push(coarseItem?.px ?? 0, coarseItem?.py ?? 0, coarseItem?.radius ?? -1, coarseItem?.depth ?? 0);
    if (!item) {
      itemRejectedCount++;
    } else if (!coarseAccepted) {
      coarseRejectedCount++;
    } else if (!filterAccepted) {
      filterRejectedCount++;
    } else {
      accepted.push(index);
      if (accepted.length >= config.maxCount) break;
    }
  }

  const candidateIndices = Uint32Array.from(accepted);
  const emittedCount = emittedCandidateIndices.length;
  const screenCoarseSummary = {
    schemaVersion: 'step107-screen-coarse-candidate-source-summary-v1',
    sourceMode: 'screenCoarse',
    contract: 'gpu-owned-screen-coarse-candidate-source-v1',
    cpuVisibleDependent: false,
    maxCount: config.maxCount,
    minRadiusPx: config.minRadiusPx,
    requireInViewport: config.requireInViewport,
    depthMode: config.depthMode,
    candidateCount: candidateIndices.length,
    inspectedCount: emittedCount,
    emittedCount,
    itemRejectedCount,
    coarseRejectedCount,
    filterRejectedCount,
    sourceCandidateMode: referenceCandidateInfo?.candidateMode ?? 'unknown',
    sourceCandidateCount: sourceIndices.length,
    candidateOrder: 'source-candidate-order',
    promotePolicy: 'never'
  };
  const filterSummary = buildFilterSummary({
    filterMode,
    requestedCount: emittedCount,
    emittedCount,
    validCount: candidateIndices.length,
    rejectedCount: itemRejectedCount + coarseRejectedCount + filterRejectedCount,
    generatedOnGpu: false,
    status: 'ok',
    reason: 'ok'
  });
  const result = {
    candidateIndices,
    candidateMode,
    validCount: candidateIndices.length,
    rejectedCount: itemRejectedCount + coarseRejectedCount + filterRejectedCount,
    filterMode: normalizeFilterMode(filterMode),
    filterSummary,
    temporalWindow: clonePlainObject(referenceCandidateInfo?.temporalWindow) ?? null,
    rangeSummary: buildRangeSummary(referenceCandidateInfo, candidateIndices.length, raw?.N ?? sourceIndices.length),
    temporalIndexDebug: clonePlainObject(referenceCandidateInfo?.temporalIndexDebug) ?? null,
    temporalBucketDebug: clonePlainObject(referenceCandidateInfo?.temporalBucketDebug) ?? null,
    candidateSourceSummary: screenCoarseSummary,
    screenCoarseSummary,
    candidateSubsetSummary: {
      enabled: true,
      contract: 'gpu-owned-candidate-source-screen-coarse-v1',
      subsetMode: 'screenCoarse',
      sourceMode: 'screenCoarse',
      subsetCount: candidateIndices.length,
      selectedCandidateCount: candidateIndices.length,
      maxCount: config.maxCount,
      sourceCandidateMode: referenceCandidateInfo?.candidateMode ?? 'unknown',
      sourceCandidateCount: sourceIndices.length,
      cpuVisibleDependent: false
    },
  };
  Object.defineProperty(result, 'screenCoarsePayload', {
    value: {
      candidateIndices: Uint32Array.from(emittedCandidateIndices),
      itemValid: Uint32Array.from(emittedItemValid),
      coarse: Float32Array.from(emittedCoarse),
      config
    },
    enumerable: false
  });
  return result;
}

export function buildGpuScreenCoarseCandidateInfo({
  gl,
  raw = null,
  referenceCandidateInfo = null,
  cpuScreenCoarseSourceInfo = null,
  runtimeConfig = {},
  filterMode = 'all-valid',
  candidateMode = 'gpu-screen-coarse-candidate-source'
} = {}) {
  const payload = cpuScreenCoarseSourceInfo?.screenCoarsePayload;
  const emittedCandidateIndices = toUint32Array(payload?.candidateIndices);
  const itemValid = toUint32Array(payload?.itemValid);
  const coarse = payload?.coarse instanceof Float32Array
    ? payload.coarse
    : new Float32Array(0);
  if (!gl || emittedCandidateIndices.length <= 0 || itemValid.length !== emittedCandidateIndices.length || coarse.length !== emittedCandidateIndices.length * 4) {
    return buildFailureCandidateInfo({
      raw,
      referenceCandidateInfo,
      reason: emittedCandidateIndices.length <= 0 ? 'empty-screen-coarse-input' : 'screen-coarse-input-invalid',
      screenCoarseSummary: cpuScreenCoarseSourceInfo?.screenCoarseSummary ?? null,
      filterMode,
      candidateMode
    });
  }
  const config = payload.config ?? buildScreenCoarseSourceConfig(runtimeConfig);
  const emittedCount = emittedCandidateIndices.length;
  try {
    const state = getScreenCoarseState(gl);
    const bytesPerOutput = 2 * Uint32Array.BYTES_PER_ELEMENT;
    const outputBytes = Math.max(bytesPerOutput, emittedCount * bytesPerOutput);
    const cleanupTransformFeedback = () => {
      gl.disable(gl.RASTERIZER_DISCARD);
      gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
      gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
      gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
      gl.bindVertexArray(null);
      gl.useProgram(null);
    };
    drainWebGlErrors(gl);
    gl.bindVertexArray(state.vao);
    gl.useProgram(state.program);

    gl.bindBuffer(gl.ARRAY_BUFFER, state.candidateBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, emittedCandidateIndices, gl.STREAM_DRAW);
    gl.enableVertexAttribArray(state.aCandidateIndex);
    gl.vertexAttribIPointer(state.aCandidateIndex, 1, gl.UNSIGNED_INT, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, state.itemValidBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, itemValid, gl.STREAM_DRAW);
    gl.enableVertexAttribArray(state.aItemValid);
    gl.vertexAttribIPointer(state.aItemValid, 1, gl.UNSIGNED_INT, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, state.coarseBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, coarse, gl.STREAM_DRAW);
    gl.enableVertexAttribArray(state.aCoarse);
    gl.vertexAttribPointer(state.aCoarse, 4, gl.FLOAT, false, 0, 0);

    gl.uniform1f(state.uCanvasWidth, config.canvasWidth);
    gl.uniform1f(state.uCanvasHeight, config.canvasHeight);
    gl.uniform1f(state.uMinRadiusPx, config.minRadiusPx);
    gl.uniform1ui(state.uRequireInViewport, config.requireInViewport ? 1 : 0);
    gl.uniform1ui(state.uDepthMode, depthModeToShaderCode(config.depthMode));
    gl.uniform1ui(state.uFilterMode, filterModeToShaderCode(filterMode));

    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, state.outputBuffer);
    gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER, outputBytes, gl.DYNAMIC_READ);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, state.transformFeedback);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, state.outputBuffer);
    const bindError = gl.getError();
    if (bindError !== gl.NO_ERROR) {
      cleanupTransformFeedback();
      throwForWebGlErrorCode(gl, 'screen-coarse-transform-feedback-bind', bindError);
    }
    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, emittedCount);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    const drawError = gl.getError();
    if (drawError !== gl.NO_ERROR) {
      cleanupTransformFeedback();
      throwForWebGlErrorCode(gl, 'screen-coarse-transform-feedback-draw', drawError);
    }
    const rawOut = new Uint32Array(emittedCount * 2);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, state.outputBuffer);
    gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER, 0, rawOut);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);
    const readbackError = gl.getError();
    if (readbackError !== gl.NO_ERROR) {
      cleanupTransformFeedback();
      throwForWebGlErrorCode(gl, 'screen-coarse-transform-feedback-readback', readbackError);
    }
    const accepted = [];
    let rejectedCount = 0;
    for (let i = 0; i < emittedCount; i++) {
      const candidateIndex = rawOut[i * 2] >>> 0;
      const validFlag = rawOut[i * 2 + 1] >>> 0;
      if (validFlag) {
        accepted.push(candidateIndex);
      } else {
        rejectedCount++;
      }
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindVertexArray(null);
    gl.useProgram(null);

    const candidateIndices = Uint32Array.from(accepted);
    const screenCoarseSummary = {
      ...(clonePlainObject(cpuScreenCoarseSourceInfo?.screenCoarseSummary) ?? {}),
      generatedOnGpu: true,
      emittedCount,
      candidateCount: candidateIndices.length,
      rejectedCount
    };
    const filterSummary = buildFilterSummary({
      filterMode,
      requestedCount: emittedCount,
      emittedCount,
      validCount: candidateIndices.length,
      rejectedCount,
      generatedOnGpu: true,
      status: 'ok',
      reason: 'ok'
    });
    return {
      candidateIndices,
      candidateMode,
      validCount: candidateIndices.length,
      rejectedCount,
      filterMode: normalizeFilterMode(filterMode),
      filterSummary,
      temporalWindow: clonePlainObject(referenceCandidateInfo?.temporalWindow) ?? null,
      rangeSummary: buildRangeSummary(referenceCandidateInfo, candidateIndices.length, raw?.N ?? 0),
      temporalIndexDebug: clonePlainObject(referenceCandidateInfo?.temporalIndexDebug) ?? null,
      temporalBucketDebug: clonePlainObject(referenceCandidateInfo?.temporalBucketDebug) ?? null,
      candidateSourceSummary: screenCoarseSummary,
      screenCoarseSummary,
      gpuCandidateSummary: {
        schemaVersion: 'step107-gpu-screen-coarse-candidate-summary-v1',
        status: 'ok',
        reason: 'ok',
        sourceMode: 'screenCoarse',
        contract: 'webgl2-transform-feedback-screen-coarse-candidate-filter-v1',
        generatedOnGpu: true,
        readbackMode: runtimeConfig.readbackMode ?? 'sync-debug',
        candidateCount: candidateIndices.length,
        emittedCount,
        rejectedCount,
        screenCoarseSummary,
        filterSummary
      }
    };
  } catch (error) {
    return buildFailureCandidateInfo({
      raw,
      referenceCandidateInfo,
      reason: error?.message ?? 'screen-coarse-gpu-error',
      screenCoarseSummary: cpuScreenCoarseSourceInfo?.screenCoarseSummary ?? null,
      filterMode,
      candidateMode
    });
  }
}
