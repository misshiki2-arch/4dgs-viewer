const stateByGl = new WeakMap();
const explicitStateByGl = new WeakMap();

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
flat out uint vValidFlag;
uniform uint uStartIndex;
uniform uint uFilterMode;
void main() {
  vCandidateIndex = uint(gl_VertexID) + uStartIndex;
  vValidFlag = (uFilterMode == 1u && (vCandidateIndex & 1u) != 0u) ? 0u : 1u;
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
    const log = gl.getProgramInfoLog(program) || 'unknown program link error';
    gl.deleteProgram(program);
    throw new Error(log);
  }
  return {
    program,
    vao: gl.createVertexArray(),
    transformFeedback: gl.createTransformFeedback(),
    buffer: gl.createBuffer(),
    uStartIndex: gl.getUniformLocation(program, 'uStartIndex'),
    uFilterMode: gl.getUniformLocation(program, 'uFilterMode')
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

function createExplicitState(gl) {
  const vertexSource = `#version 300 es
precision highp float;
in uint aCandidateIndex;
flat out uint vCandidateIndex;
flat out uint vValidFlag;
uniform uint uFilterMode;
void main() {
  vCandidateIndex = aCandidateIndex;
  vValidFlag = (uFilterMode == 1u && (vCandidateIndex & 1u) != 0u) ? 0u : 1u;
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
    const log = gl.getProgramInfoLog(program) || 'unknown explicit candidate program link error';
    gl.deleteProgram(program);
    throw new Error(log);
  }
  return {
    program,
    vao: gl.createVertexArray(),
    inputBuffer: gl.createBuffer(),
    outputBuffer: gl.createBuffer(),
    transformFeedback: gl.createTransformFeedback(),
    aCandidateIndex: gl.getAttribLocation(program, 'aCandidateIndex'),
    uFilterMode: gl.getUniformLocation(program, 'uFilterMode')
  };
}

function getExplicitState(gl) {
  let state = explicitStateByGl.get(gl);
  if (!state) {
    state = createExplicitState(gl);
    explicitStateByGl.set(gl, state);
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

function normalizeFilterMode(filterMode) {
  return filterMode === 'evenIndex' ? 'evenIndex' : 'all-valid';
}

function filterModeToShaderCode(filterMode) {
  return normalizeFilterMode(filterMode) === 'evenIndex' ? 1 : 0;
}

function isCandidateValidForFilter(candidateIndex, filterMode) {
  if (normalizeFilterMode(filterMode) === 'evenIndex') return (candidateIndex & 1) === 0;
  return true;
}

function buildFilterPredicateSummary(filterMode) {
  const mode = normalizeFilterMode(filterMode);
  if (mode === 'evenIndex') {
    return {
      type: 'evenIndex',
      expression: 'candidateIndex % 2 === 0',
      divisor: 2,
      remainder: 0
    };
  }
  return {
    type: 'all-valid',
    expression: 'true'
  };
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
    filterMode,
    predicate: buildFilterPredicateSummary(filterMode),
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

export function buildCpuFilteredCandidateInfo({
  referenceSubsetCandidateInfo = null,
  raw = null,
  filterMode = 'all-valid',
  candidateMode = 'cpu-firstn-filter-reference'
} = {}) {
  const normalizedFilterMode = normalizeFilterMode(filterMode);
  const source = referenceSubsetCandidateInfo ?? {};
  const sourceIndices = source.candidateIndices instanceof Uint32Array
    ? source.candidateIndices
    : (Array.isArray(source.candidateIndices) ? Uint32Array.from(source.candidateIndices) : new Uint32Array(0));
  const accepted = [];
  let rejectedCount = 0;
  for (const candidateIndex of sourceIndices) {
    if (isCandidateValidForFilter(candidateIndex, normalizedFilterMode)) {
      accepted.push(candidateIndex);
    } else {
      rejectedCount++;
    }
  }
  const candidateIndices = Uint32Array.from(accepted);
  const filterSummary = buildFilterSummary({
    filterMode: normalizedFilterMode,
    requestedCount: sourceIndices.length,
    emittedCount: sourceIndices.length,
    validCount: candidateIndices.length,
    rejectedCount,
    generatedOnGpu: false,
    status: 'ok',
    reason: 'ok'
  });
  const total = raw ? raw.N : (source.rangeSummary?.totalCount ?? sourceIndices.length);
  return {
    candidateIndices,
    candidateMode,
    validCount: candidateIndices.length,
    rejectedCount,
    filterMode: normalizedFilterMode,
    filterSummary,
    temporalWindow: clonePlainObject(source.temporalWindow) ?? null,
    rangeSummary: buildRangeSummary(source, candidateIndices.length, total),
    temporalIndexDebug: clonePlainObject(source.temporalIndexDebug) ?? null,
    temporalBucketDebug: clonePlainObject(source.temporalBucketDebug) ?? null,
    candidateSubsetSummary: source.candidateSubsetSummary
      ? {
          ...clonePlainObject(source.candidateSubsetSummary),
          preFilterSubsetCount: sourceIndices.length,
          subsetCount: candidateIndices.length,
          filterApplied: true,
          filterMode: normalizedFilterMode
        }
      : null
  };
}

function buildFailureCandidateInfo({
  raw = null,
  referenceSubsetCandidateInfo = null,
  reason,
  filterMode = 'all-valid',
  requestedCount = 0,
  candidateMode = 'gpu-firstn-unavailable'
}) {
  const total = raw ? raw.N : (referenceSubsetCandidateInfo?.rangeSummary?.totalCount ?? 0);
  const filterSummary = buildFilterSummary({
    filterMode,
    requestedCount,
    emittedCount: 0,
    validCount: 0,
    rejectedCount: 0,
    generatedOnGpu: false,
    status: 'failure',
    reason
  });
  return {
    candidateIndices: new Uint32Array(0),
    candidateMode,
    validCount: 0,
    rejectedCount: 0,
    filterMode,
    filterSummary,
    temporalWindow: clonePlainObject(referenceSubsetCandidateInfo?.temporalWindow) ?? null,
    rangeSummary: buildRangeSummary(referenceSubsetCandidateInfo, 0, total),
    temporalIndexDebug: clonePlainObject(referenceSubsetCandidateInfo?.temporalIndexDebug) ?? null,
    temporalBucketDebug: clonePlainObject(referenceSubsetCandidateInfo?.temporalBucketDebug) ?? null,
    candidateSubsetSummary: clonePlainObject(referenceSubsetCandidateInfo?.candidateSubsetSummary) ?? null,
    gpuCandidateSummary: {
      enabled: true,
      status: 'failure',
      contract: 'webgl2-transform-feedback-firstn-candidate-filter-v1',
      generatedOnGpu: false,
      candidateCount: 0,
      validCount: 0,
      rejectedCount: 0,
      filterMode,
      filterSummary,
      reason
    }
  };
}

function collectAcceptedFromTransformFeedback(rawOut, count) {
  const fieldsPerCandidate = 2;
  const accepted = [];
  let validCount = 0;
  let rejectedCount = 0;
  for (let i = 0; i < count; i++) {
    const candidateIndex = rawOut[i * fieldsPerCandidate];
    const validFlag = rawOut[i * fieldsPerCandidate + 1];
    if (validFlag !== 0) {
      accepted.push(candidateIndex);
      validCount++;
    } else {
      rejectedCount++;
    }
  }
  return {
    candidateIndices: Uint32Array.from(accepted),
    validCount,
    rejectedCount
  };
}

export function buildGpuFirstNCandidateInfo({
  gl,
  raw = null,
  referenceSubsetCandidateInfo = null,
  subsetCount = 1024,
  startIndex = 0,
  filterMode = 'all-valid'
} = {}) {
  const normalizedFilterMode = normalizeFilterMode(filterMode);
  const requestedCount = Number.isFinite(Number(subsetCount)) ? Math.max(0, Number(subsetCount) | 0) : 1024;
  if (!gl || typeof gl.createProgram !== 'function' || typeof gl.beginTransformFeedback !== 'function') {
    return buildFailureCandidateInfo({
      raw,
      referenceSubsetCandidateInfo,
      reason: 'webgl2-transform-feedback-unavailable',
      filterMode: normalizedFilterMode,
      requestedCount
    });
  }

  const rawCount = raw && Number.isFinite(raw.N) ? Math.max(0, raw.N | 0) : requestedCount;
  const referenceCount = referenceSubsetCandidateInfo?.candidateIndices instanceof Uint32Array
    ? referenceSubsetCandidateInfo.candidateIndices.length
    : requestedCount;
  const count = Math.max(0, Math.min(requestedCount, referenceCount, rawCount));
  const start = Number.isFinite(Number(startIndex)) ? Math.max(0, Number(startIndex) | 0) : 0;

  try {
    const state = getFirstNState(gl);
    const fieldsPerCandidate = 2;
    const byteCount = count * fieldsPerCandidate * Uint32Array.BYTES_PER_ELEMENT;
    const rawOut = new Uint32Array(count * fieldsPerCandidate);

    gl.bindVertexArray(state.vao);
    gl.useProgram(state.program);
    gl.uniform1ui(state.uStartIndex, start);
    gl.uniform1ui(state.uFilterMode, filterModeToShaderCode(normalizedFilterMode));
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
    gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER, 0, rawOut);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);
    gl.bindVertexArray(null);

    const { candidateIndices: out, validCount, rejectedCount } = collectAcceptedFromTransformFeedback(rawOut, count);
    const filterSummary = buildFilterSummary({
      filterMode: normalizedFilterMode,
      requestedCount,
      emittedCount: count,
      validCount,
      rejectedCount,
      generatedOnGpu: true,
      status: 'ok',
      reason: 'ok'
    });
    const total = raw ? raw.N : (referenceSubsetCandidateInfo?.rangeSummary?.totalCount ?? count);
    return {
      candidateIndices: out,
      candidateMode: 'gpu-firstn-debug',
      validCount,
      rejectedCount,
      filterMode: normalizedFilterMode,
      filterSummary,
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
        contract: 'webgl2-transform-feedback-firstn-candidate-filter-v1',
        generatedOnGpu: true,
        candidateCount: out.length,
        requestedSubsetCount: requestedCount,
        startIndex: start,
        validCount,
        rejectedCount,
        filterMode: normalizedFilterMode,
        filterSummary,
        reason: 'ok'
      }
    };
  } catch (error) {
    return buildFailureCandidateInfo({
      raw,
      referenceSubsetCandidateInfo,
      reason: error?.message ?? 'unknown-gpu-firstn-candidate-error',
      filterMode: normalizedFilterMode,
      requestedCount
    });
  }
}

export function buildGpuExplicitCandidateInfo({
  gl,
  raw = null,
  referenceSubsetCandidateInfo = null,
  candidateIndices = null,
  filterMode = 'all-valid'
} = {}) {
  const normalizedFilterMode = normalizeFilterMode(filterMode);
  const sourceIndices = candidateIndices instanceof Uint32Array
    ? candidateIndices
    : (Array.isArray(candidateIndices) ? Uint32Array.from(candidateIndices) : (
      referenceSubsetCandidateInfo?.candidateIndices instanceof Uint32Array
        ? referenceSubsetCandidateInfo.candidateIndices
        : new Uint32Array(0)
    ));
  const requestedCount = sourceIndices.length;
  if (!gl || typeof gl.createProgram !== 'function' || typeof gl.beginTransformFeedback !== 'function') {
    return buildFailureCandidateInfo({
      raw,
      referenceSubsetCandidateInfo,
      reason: 'webgl2-transform-feedback-unavailable',
      filterMode: normalizedFilterMode,
      requestedCount,
      candidateMode: 'gpu-explicit-unavailable'
    });
  }

  try {
    const state = getExplicitState(gl);
    const fieldsPerCandidate = 2;
    const count = sourceIndices.length;
    const rawOut = new Uint32Array(count * fieldsPerCandidate);

    gl.bindVertexArray(state.vao);
    gl.useProgram(state.program);
    gl.uniform1ui(state.uFilterMode, filterModeToShaderCode(normalizedFilterMode));
    gl.bindBuffer(gl.ARRAY_BUFFER, state.inputBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, sourceIndices, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(state.aCandidateIndex);
    gl.vertexAttribIPointer(state.aCandidateIndex, 1, gl.UNSIGNED_INT, 0, 0);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, state.outputBuffer);
    gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER, rawOut.byteLength, gl.DYNAMIC_READ);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, state.transformFeedback);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, state.outputBuffer);
    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, count);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, state.outputBuffer);
    gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER, 0, rawOut);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.disableVertexAttribArray(state.aCandidateIndex);
    gl.bindVertexArray(null);

    const { candidateIndices: out, validCount, rejectedCount } = collectAcceptedFromTransformFeedback(rawOut, count);
    const filterSummary = buildFilterSummary({
      filterMode: normalizedFilterMode,
      requestedCount,
      emittedCount: count,
      validCount,
      rejectedCount,
      generatedOnGpu: true,
      status: 'ok',
      reason: 'ok'
    });
    const total = raw ? raw.N : (referenceSubsetCandidateInfo?.rangeSummary?.totalCount ?? count);
    return {
      candidateIndices: out,
      candidateMode: 'gpu-explicit-debug',
      validCount,
      rejectedCount,
      filterMode: normalizedFilterMode,
      filterSummary,
      temporalWindow: clonePlainObject(referenceSubsetCandidateInfo?.temporalWindow) ?? null,
      rangeSummary: buildRangeSummary(referenceSubsetCandidateInfo, out.length, total),
      temporalIndexDebug: clonePlainObject(referenceSubsetCandidateInfo?.temporalIndexDebug) ?? null,
      temporalBucketDebug: clonePlainObject(referenceSubsetCandidateInfo?.temporalBucketDebug) ?? null,
      candidateSubsetSummary: clonePlainObject(referenceSubsetCandidateInfo?.candidateSubsetSummary) ?? null,
      gpuCandidateSummary: {
        enabled: true,
        status: 'ok',
        contract: 'webgl2-transform-feedback-explicit-candidate-filter-v1',
        generatedOnGpu: true,
        candidateCount: out.length,
        requestedSubsetCount: requestedCount,
        validCount,
        rejectedCount,
        filterMode: normalizedFilterMode,
        filterSummary,
        reason: 'ok'
      }
    };
  } catch (error) {
    return buildFailureCandidateInfo({
      raw,
      referenceSubsetCandidateInfo,
      reason: error?.message ?? 'unknown-gpu-explicit-candidate-error',
      filterMode: normalizedFilterMode,
      requestedCount,
      candidateMode: 'gpu-explicit-unavailable'
    });
  }
}
