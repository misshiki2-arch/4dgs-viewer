const stateByGl = new WeakMap();

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

function createFirstNState(gl) {
  const vertexSource = `#version 300 es
precision highp float;
flat out uint vCandidateIndex;
uniform uint uStartIndex;
void main() {
  vCandidateIndex = uint(gl_VertexID) + uStartIndex;
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
  gl.transformFeedbackVaryings(program, ['vCandidateIndex'], gl.INTERLEAVED_ATTRIBS);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) || 'unknown program link error';
    gl.deleteProgram(program);
    throw new Error(log);
  }
  return {
    program,
    vao: gl.createVertexArray(),
    transformFeedback: gl.createTransformFeedback(),
    buffer: gl.createBuffer(),
    uStartIndex: gl.getUniformLocation(program, 'uStartIndex')
  };
}

function getFirstNState(gl) {
  let state = stateByGl.get(gl);
  if (!state) {
    state = createFirstNState(gl);
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

function buildRangeSummary(referenceSubsetCandidateInfo, count, total) {
  const base = clonePlainObject(referenceSubsetCandidateInfo?.rangeSummary) ?? {};
  return {
    ...base,
    totalCount: Number.isFinite(base.totalCount) ? base.totalCount : total,
    candidateCount: count,
    candidateFraction: total > 0 ? count / total : 0
  };
}

function buildFailureCandidateInfo({ raw = null, referenceSubsetCandidateInfo = null, reason }) {
  const total = raw ? raw.N : (referenceSubsetCandidateInfo?.rangeSummary?.totalCount ?? 0);
  return {
    candidateIndices: new Uint32Array(0),
    candidateMode: 'gpu-firstn-unavailable',
    temporalWindow: clonePlainObject(referenceSubsetCandidateInfo?.temporalWindow) ?? null,
    rangeSummary: buildRangeSummary(referenceSubsetCandidateInfo, 0, total),
    temporalIndexDebug: clonePlainObject(referenceSubsetCandidateInfo?.temporalIndexDebug) ?? null,
    temporalBucketDebug: clonePlainObject(referenceSubsetCandidateInfo?.temporalBucketDebug) ?? null,
    candidateSubsetSummary: clonePlainObject(referenceSubsetCandidateInfo?.candidateSubsetSummary) ?? null,
    gpuCandidateSummary: {
      enabled: true,
      status: 'failure',
      contract: 'webgl2-transform-feedback-firstn-candidate-indices',
      generatedOnGpu: false,
      candidateCount: 0,
      reason
    }
  };
}

export function buildGpuFirstNCandidateInfo({
  gl,
  raw = null,
  referenceSubsetCandidateInfo = null,
  subsetCount = 1024,
  startIndex = 0
} = {}) {
  if (!gl || typeof gl.createProgram !== 'function' || typeof gl.beginTransformFeedback !== 'function') {
    return buildFailureCandidateInfo({
      raw,
      referenceSubsetCandidateInfo,
      reason: 'webgl2-transform-feedback-unavailable'
    });
  }

  const requestedCount = Number.isFinite(Number(subsetCount)) ? Math.max(0, Number(subsetCount) | 0) : 1024;
  const rawCount = raw && Number.isFinite(raw.N) ? Math.max(0, raw.N | 0) : requestedCount;
  const referenceCount = referenceSubsetCandidateInfo?.candidateIndices instanceof Uint32Array
    ? referenceSubsetCandidateInfo.candidateIndices.length
    : requestedCount;
  const count = Math.max(0, Math.min(requestedCount, referenceCount, rawCount));
  const start = Number.isFinite(Number(startIndex)) ? Math.max(0, Number(startIndex) | 0) : 0;

  try {
    const state = getFirstNState(gl);
    const byteCount = count * Uint32Array.BYTES_PER_ELEMENT;
    const out = new Uint32Array(count);

    gl.bindVertexArray(state.vao);
    gl.useProgram(state.program);
    gl.uniform1ui(state.uStartIndex, start);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, state.buffer);
    gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER, byteCount, gl.DYNAMIC_READ);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, state.transformFeedback);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, state.buffer);
    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, count);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, state.buffer);
    gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER, 0, out);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);
    gl.bindVertexArray(null);

    const total = raw ? raw.N : (referenceSubsetCandidateInfo?.rangeSummary?.totalCount ?? count);
    return {
      candidateIndices: out,
      candidateMode: 'gpu-firstn-debug',
      temporalWindow: clonePlainObject(referenceSubsetCandidateInfo?.temporalWindow) ?? null,
      rangeSummary: buildRangeSummary(referenceSubsetCandidateInfo, out.length, total),
      temporalIndexDebug: clonePlainObject(referenceSubsetCandidateInfo?.temporalIndexDebug) ?? null,
      temporalBucketDebug: clonePlainObject(referenceSubsetCandidateInfo?.temporalBucketDebug) ?? null,
      candidateSubsetSummary: clonePlainObject(referenceSubsetCandidateInfo?.candidateSubsetSummary) ?? {
        enabled: true,
        contract: 'candidate-info-subset-adapter',
        subsetMode: 'firstN',
        subsetCount: out.length,
        requestedSubsetCount: requestedCount,
        explicitIndexCount: 0,
        sourceCandidateMode: 'unknown',
        sourceCandidateCount: total
      },
      gpuCandidateSummary: {
        enabled: true,
        status: 'ok',
        contract: 'webgl2-transform-feedback-firstn-candidate-indices',
        generatedOnGpu: true,
        candidateCount: out.length,
        requestedSubsetCount: requestedCount,
        startIndex: start,
        reason: 'ok'
      }
    };
  } catch (error) {
    return buildFailureCandidateInfo({
      raw,
      referenceSubsetCandidateInfo,
      reason: error?.message ?? 'unknown-gpu-firstn-candidate-error'
    });
  }
}
