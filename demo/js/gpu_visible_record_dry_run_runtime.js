import { buildVisibleItemForCandidate } from './gpu_visible_item_builder.js';

const stateByGl = new WeakMap();
const RECORD_FLOATS = 21;
const DEFAULT_MAX_RECORDS = 65536;
const DEFAULT_EPSILON = 1e-6;
const FIELD_LAYOUT = [
  ['srcIndex', 0, 1],
  ['valid', 1, 1],
  ['px', 2, 1],
  ['py', 3, 1],
  ['radius', 4, 1],
  ['depth', 5, 1],
  ['conic', 6, 3],
  ['colorAlpha', 9, 4],
  ['aabb', 13, 4],
  ['tileRange', 17, 4]
];

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

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

function createVisibleRecordState(gl) {
  const vertexSource = `#version 300 es
precision highp float;
in vec2 aSrcValid;
in vec4 aScreen;
in vec3 aConic;
in vec4 aColorAlpha;
in vec4 aAabb;
in vec4 aTileRange;
out vec2 vSrcValid;
out vec4 vScreen;
out vec3 vConic;
out vec4 vColorAlpha;
out vec4 vAabb;
out vec4 vTileRange;
void main() {
  vSrcValid = aSrcValid;
  vScreen = aScreen;
  vConic = aConic;
  vColorAlpha = aColorAlpha;
  vAabb = aAabb;
  vTileRange = aTileRange;
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
  gl.transformFeedbackVaryings(program, [
    'vSrcValid',
    'vScreen',
    'vConic',
    'vColorAlpha',
    'vAabb',
    'vTileRange'
  ], gl.INTERLEAVED_ATTRIBS);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) || 'unknown visible record program link error';
    gl.deleteProgram(program);
    throw new Error(log);
  }
  return {
    program,
    vao: gl.createVertexArray(),
    inputBuffer: gl.createBuffer(),
    outputBuffer: gl.createBuffer(),
    transformFeedback: gl.createTransformFeedback(),
    aSrcValid: gl.getAttribLocation(program, 'aSrcValid'),
    aScreen: gl.getAttribLocation(program, 'aScreen'),
    aConic: gl.getAttribLocation(program, 'aConic'),
    aColorAlpha: gl.getAttribLocation(program, 'aColorAlpha'),
    aAabb: gl.getAttribLocation(program, 'aAabb'),
    aTileRange: gl.getAttribLocation(program, 'aTileRange')
  };
}

function getVisibleRecordState(gl) {
  let state = stateByGl.get(gl);
  if (!state) {
    state = createVisibleRecordState(gl);
    stateByGl.set(gl, state);
  }
  return state;
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

function passesTemporalCulling(raw, i, timestamp, sigmaScale = 1.0, sigmaThreshold = 3.0) {
  if (!raw || !raw.t || !raw.scale_t) return true;
  const t0 = raw.t[i];
  if (!Number.isFinite(t0)) return true;
  const s = raw.scale_t[i];
  if (!Number.isFinite(s)) return true;
  const sigmaT = s * sigmaScale;
  if (!Number.isFinite(sigmaT) || sigmaT <= 0) return true;
  return Math.abs(timestamp - t0) <= sigmaThreshold * sigmaT;
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

function writeItemRecord(records, row, srcIndex, item) {
  const base = row * RECORD_FLOATS;
  records[base + 0] = srcIndex;
  records[base + 1] = item ? 1 : 0;
  if (!item) return;
  records[base + 2] = Math.fround(item.px);
  records[base + 3] = Math.fround(item.py);
  records[base + 4] = Math.fround(item.radius);
  records[base + 5] = Math.fround(item.depth);
  for (let i = 0; i < 3; i++) records[base + 6 + i] = Math.fround(item.conic?.[i] ?? 0);
  for (let i = 0; i < 4; i++) records[base + 9 + i] = Math.fround(item.colorAlpha?.[i] ?? 0);
  for (let i = 0; i < 4; i++) records[base + 13 + i] = Math.fround(item.aabb?.[i] ?? 0);
  for (let i = 0; i < 4; i++) records[base + 17 + i] = Math.fround(item.tileRange?.[i] ?? -1);
}

function buildCpuVisibleRecords({
  candidateInfo,
  raw,
  camera,
  screenSpaceCamera = null,
  canvasWidth,
  canvasHeight,
  camPos,
  tileGrid = null,
  buildConfig = {},
  temporalSigmaThreshold = 3.0,
  maxRecords = DEFAULT_MAX_RECORDS
} = {}) {
  const totalStartMs = nowMs();
  const renderScale = Number.isFinite(buildConfig.renderScale) ? buildConfig.renderScale : 1;
  const renderW = Math.max(1, Math.round(canvasWidth * renderScale));
  const renderH = Math.max(1, Math.round(canvasHeight * renderScale));
  const sx = canvasWidth / renderW;
  const sy = canvasHeight / renderH;
  const flags = {
    nativeRot4d: !!buildConfig.useNativeRot4d,
    nativeMarginal: !!buildConfig.useNativeMarginal
  };
  const sourceIndices = toUint32Array(candidateInfo?.candidateIndices);
  const count = Math.min(sourceIndices.length, toFiniteInteger(maxRecords, DEFAULT_MAX_RECORDS));
  const records = new Float32Array(count * RECORD_FLOATS);
  const validSrcIndices = [];
  const validRecordBySrcIndex = new Map();
  let temporalRejected = 0;
  let invalidItemCount = 0;

  const loopStartMs = nowMs();
  for (let row = 0; row < count; row++) {
    const srcIndex = sourceIndices[row] >>> 0;
    let item = null;
    if (!passesTemporalCulling(raw, srcIndex, buildConfig.timestamp, buildConfig.sigmaScale, temporalSigmaThreshold)) {
      temporalRejected++;
    } else {
      item = buildVisibleItemForCandidate({
        raw,
        index: srcIndex,
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
      if (!item) invalidItemCount++;
    }
    writeItemRecord(records, row, srcIndex, item);
    if (item) {
      validSrcIndices.push(srcIndex);
      validRecordBySrcIndex.set(srcIndex, row);
    }
  }
  const loopMs = nowMs() - loopStartMs;
  return {
    records,
    count,
    candidateCount: sourceIndices.length,
    validCount: validSrcIndices.length,
    validSrcIndices,
    validRecordBySrcIndex,
    temporalRejected,
    invalidItemCount,
    timing: {
      cpuRecordBuildMs: nowMs() - totalStartMs,
      cpuRecordLoopMs: loopMs
    }
  };
}

function runTransformFeedback(gl, inputRecords, recordCount) {
  const state = getVisibleRecordState(gl);
  const outputBytes = Math.max(
    RECORD_FLOATS * Float32Array.BYTES_PER_ELEMENT,
    recordCount * RECORD_FLOATS * Float32Array.BYTES_PER_ELEMENT
  );
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
  const uploadStartMs = nowMs();
  gl.bindVertexArray(state.vao);
  gl.useProgram(state.program);
  gl.bindBuffer(gl.ARRAY_BUFFER, state.inputBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, inputRecords, gl.STREAM_DRAW);
  const strideBytes = RECORD_FLOATS * Float32Array.BYTES_PER_ELEMENT;
  gl.enableVertexAttribArray(state.aSrcValid);
  gl.vertexAttribPointer(state.aSrcValid, 2, gl.FLOAT, false, strideBytes, 0);
  gl.enableVertexAttribArray(state.aScreen);
  gl.vertexAttribPointer(state.aScreen, 4, gl.FLOAT, false, strideBytes, 2 * Float32Array.BYTES_PER_ELEMENT);
  gl.enableVertexAttribArray(state.aConic);
  gl.vertexAttribPointer(state.aConic, 3, gl.FLOAT, false, strideBytes, 6 * Float32Array.BYTES_PER_ELEMENT);
  gl.enableVertexAttribArray(state.aColorAlpha);
  gl.vertexAttribPointer(state.aColorAlpha, 4, gl.FLOAT, false, strideBytes, 9 * Float32Array.BYTES_PER_ELEMENT);
  gl.enableVertexAttribArray(state.aAabb);
  gl.vertexAttribPointer(state.aAabb, 4, gl.FLOAT, false, strideBytes, 13 * Float32Array.BYTES_PER_ELEMENT);
  gl.enableVertexAttribArray(state.aTileRange);
  gl.vertexAttribPointer(state.aTileRange, 4, gl.FLOAT, false, strideBytes, 17 * Float32Array.BYTES_PER_ELEMENT);
  const uploadAndSetupMs = nowMs() - uploadStartMs;

  const transformFeedbackSetupStartMs = nowMs();
  gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, state.outputBuffer);
  gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER, outputBytes, gl.DYNAMIC_READ);
  gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);
  gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, state.transformFeedback);
  gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, state.outputBuffer);
  const bindError = gl.getError();
  if (bindError !== gl.NO_ERROR) {
    cleanupTransformFeedback();
    throwForWebGlErrorCode(gl, 'visible-record-transform-feedback-bind', bindError);
  }
  const transformFeedbackSetupMs = nowMs() - transformFeedbackSetupStartMs;

  const transformFeedbackDrawStartMs = nowMs();
  gl.enable(gl.RASTERIZER_DISCARD);
  gl.beginTransformFeedback(gl.POINTS);
  gl.drawArrays(gl.POINTS, 0, recordCount);
  gl.endTransformFeedback();
  gl.disable(gl.RASTERIZER_DISCARD);
  gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
  gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
  const drawError = gl.getError();
  if (drawError !== gl.NO_ERROR) {
    cleanupTransformFeedback();
    throwForWebGlErrorCode(gl, 'visible-record-transform-feedback-draw', drawError);
  }
  const transformFeedbackDrawMs = nowMs() - transformFeedbackDrawStartMs;

  const out = new Float32Array(recordCount * RECORD_FLOATS);
  const readbackStartMs = nowMs();
  gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, state.outputBuffer);
  gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER, 0, out);
  gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);
  const readbackError = gl.getError();
  if (readbackError !== gl.NO_ERROR) {
    cleanupTransformFeedback();
    throwForWebGlErrorCode(gl, 'visible-record-transform-feedback-readback', readbackError);
  }
  const readbackMs = nowMs() - readbackStartMs;

  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  gl.bindVertexArray(null);
  gl.useProgram(null);
  return {
    records: out,
    timing: {
      uploadAndSetupMs,
      transformFeedbackSetupMs,
      transformFeedbackDrawMs,
      readbackMs,
      outputBytes
    }
  };
}

function readField(records, row, offset, length) {
  const base = row * RECORD_FLOATS + offset;
  if (length === 1) return records[base];
  const out = [];
  for (let i = 0; i < length; i++) out.push(records[base + i]);
  return out;
}

function compareScalar(a, b, epsilon) {
  if (!Number.isFinite(a) && !Number.isFinite(b)) return { mismatch: false, abs: 0 };
  const abs = Math.abs(Number(a) - Number(b));
  return { mismatch: !(abs <= epsilon), abs };
}

function compareRecords(referenceRecords, candidateRecords, recordCount, options = {}) {
  const epsilon = toFiniteNumber(options.epsilon, DEFAULT_EPSILON);
  const maxMismatches = toFiniteInteger(options.maxMismatches, 32);
  const firstMismatches = [];
  let fieldMismatchCount = 0;
  let maxAbsError = 0;

  for (let row = 0; row < recordCount; row++) {
    for (const [field, offset, length] of FIELD_LAYOUT) {
      for (let i = 0; i < length; i++) {
        const ref = referenceRecords[row * RECORD_FLOATS + offset + i];
        const got = candidateRecords[row * RECORD_FLOATS + offset + i];
        const { mismatch, abs } = compareScalar(ref, got, epsilon);
        maxAbsError = Math.max(maxAbsError, abs);
        if (mismatch) {
          fieldMismatchCount++;
          if (firstMismatches.length < maxMismatches) {
            firstMismatches.push({
              row,
              field,
              component: length > 1 ? i : null,
              reference: ref,
              candidate: got,
              absError: abs
            });
          }
        }
      }
    }
  }
  return {
    anyMismatch: fieldMismatchCount > 0,
    fieldMismatchCount,
    maxAbsError,
    firstMismatches
  };
}

function compareReferenceVisibleItems(referenceVisibleItems, gpuRecords, recordBySrcIndex, options = {}) {
  const epsilon = toFiniteNumber(options.epsilon, DEFAULT_EPSILON);
  const maxMismatches = toFiniteInteger(options.maxMismatches, 32);
  const firstMismatches = [];
  let missingRecordCount = 0;
  let invalidRecordCount = 0;
  let fieldMismatchCount = 0;
  let maxAbsError = 0;
  const visible = Array.isArray(referenceVisibleItems) ? referenceVisibleItems : [];

  const pushMismatch = (entry) => {
    if (firstMismatches.length < maxMismatches) firstMismatches.push(entry);
  };

  const compareVisibleField = (item, row, field, offset, length, values) => {
    for (let i = 0; i < length; i++) {
      const ref = values[i];
      const got = gpuRecords[row * RECORD_FLOATS + offset + i];
      const { mismatch, abs } = compareScalar(ref, got, epsilon);
      maxAbsError = Math.max(maxAbsError, abs);
      if (mismatch) {
        fieldMismatchCount++;
        pushMismatch({
          srcIndex: item.srcIndex,
          field,
          component: length > 1 ? i : null,
          reference: ref,
          candidate: got,
          absError: abs
        });
      }
    }
  };

  for (const item of visible) {
    const srcIndex = Number(item?.srcIndex);
    if (!Number.isFinite(srcIndex)) continue;
    const row = recordBySrcIndex.get(srcIndex >>> 0);
    if (!Number.isFinite(row)) {
      missingRecordCount++;
      pushMismatch({ srcIndex, field: 'record', reason: 'missing-record' });
      continue;
    }
    const valid = gpuRecords[row * RECORD_FLOATS + 1];
    if (valid < 0.5) {
      invalidRecordCount++;
      pushMismatch({ srcIndex, field: 'valid', reference: 1, candidate: valid });
      continue;
    }
    compareVisibleField(item, row, 'px', 2, 1, [Math.fround(item.px)]);
    compareVisibleField(item, row, 'py', 3, 1, [Math.fround(item.py)]);
    compareVisibleField(item, row, 'radius', 4, 1, [Math.fround(item.radius)]);
    compareVisibleField(item, row, 'depth', 5, 1, [Math.fround(item.depth)]);
    compareVisibleField(item, row, 'conic', 6, 3, (item.conic ?? []).map((v) => Math.fround(v ?? 0)));
    compareVisibleField(item, row, 'colorAlpha', 9, 4, (item.colorAlpha ?? []).map((v) => Math.fround(v ?? 0)));
    compareVisibleField(item, row, 'aabb', 13, 4, (item.aabb ?? []).map((v) => Math.fround(v ?? 0)));
    compareVisibleField(item, row, 'tileRange', 17, 4, (item.tileRange ?? [-1, -1, -1, -1]).map((v) => Math.fround(v ?? -1)));
  }
  const anyMismatch = missingRecordCount > 0 || invalidRecordCount > 0 || fieldMismatchCount > 0;
  return {
    anyMismatch,
    referenceVisibleCount: visible.length,
    missingRecordCount,
    invalidRecordCount,
    fieldMismatchCount,
    maxAbsError,
    firstMismatches
  };
}

function classifyMismatch({ status, recordComparison, referenceVisibleComparison }) {
  if (status !== 'ok') return 'gpu-visible-record-unavailable';
  if (recordComparison?.anyMismatch) return 'visible-record-field-mismatch';
  if (referenceVisibleComparison?.missingRecordCount > 0) return 'reference-visible-missing-record';
  if (referenceVisibleComparison?.invalidRecordCount > 0) return 'reference-visible-invalid-record';
  if (referenceVisibleComparison?.fieldMismatchCount > 0) return 'reference-visible-field-mismatch';
  return 'none';
}

function buildFallbackSummary(reason, extra = {}) {
  return {
    schemaVersion: 'step114-gpu-visible-record-dry-run-v1',
    status: 'fallback',
    reason,
    sourceMode: extra.sourceMode ?? 'screenCoarse',
    recordLayout: {
      floatsPerRecord: RECORD_FLOATS,
      fields: FIELD_LAYOUT.map(([name, offset, length]) => ({ name, offset, length }))
    },
    mismatchClassification: 'gpu-visible-record-unavailable',
    displayCandidateSource: extra.displayCandidateSource ?? 'cpu-reference',
    gpuCandidateUsedForDisplay: !!extra.gpuCandidateUsedForDisplay,
    limitedDrawUsedForCandidateSource: !!extra.limitedDrawUsedForCandidateSource
  };
}

export function runGpuVisibleRecordDryRun({
  gl,
  candidateInfo,
  raw,
  camera,
  screenSpaceCamera = null,
  canvasWidth = 0,
  canvasHeight = 0,
  camPos = null,
  tileGrid = null,
  buildConfig = {},
  temporalSigmaThreshold = 3.0,
  referenceVisibleItems = null,
  maxRecords = DEFAULT_MAX_RECORDS,
  epsilon = DEFAULT_EPSILON,
  maxMismatches = 32,
  sourceMode = 'screenCoarse',
  readbackMode = 'sync-debug',
  displayCandidateSource = 'cpu-reference',
  gpuCandidateUsedForDisplay = false,
  limitedDrawUsedForCandidateSource = false,
  metadata = null
} = {}) {
  const totalStartMs = nowMs();
  if (!gl) {
    return buildFallbackSummary('webgl-unavailable', {
      sourceMode,
      displayCandidateSource,
      gpuCandidateUsedForDisplay,
      limitedDrawUsedForCandidateSource
    });
  }
  if (!raw || !camera || !candidateInfo) {
    return buildFallbackSummary('visible-record-input-unavailable', {
      sourceMode,
      displayCandidateSource,
      gpuCandidateUsedForDisplay,
      limitedDrawUsedForCandidateSource
    });
  }
  if (readbackMode === 'none') {
    return buildFallbackSummary('debug-readback-disabled', {
      sourceMode,
      displayCandidateSource,
      gpuCandidateUsedForDisplay,
      limitedDrawUsedForCandidateSource
    });
  }

  try {
    const cpuRecordBuild = buildCpuVisibleRecords({
      candidateInfo,
      raw,
      camera,
      screenSpaceCamera,
      canvasWidth,
      canvasHeight,
      camPos,
      tileGrid,
      buildConfig,
      temporalSigmaThreshold,
      maxRecords
    });
    const tfResult = runTransformFeedback(gl, cpuRecordBuild.records, cpuRecordBuild.count);
    const compareStartMs = nowMs();
    const recordComparison = compareRecords(cpuRecordBuild.records, tfResult.records, cpuRecordBuild.count, {
      epsilon,
      maxMismatches
    });
    const referenceVisibleComparison = compareReferenceVisibleItems(
      referenceVisibleItems,
      tfResult.records,
      cpuRecordBuild.validRecordBySrcIndex,
      { epsilon, maxMismatches }
    );
    const compareMs = nowMs() - compareStartMs;
    const status = 'ok';
    const mismatchClassification = classifyMismatch({
      status,
      recordComparison,
      referenceVisibleComparison
    });
    return {
      schemaVersion: 'step114-gpu-visible-record-dry-run-v1',
      status,
      reason: 'ok',
      sourceMode,
      readbackMode,
      contract: 'webgl2-transform-feedback-visible-fixed-record-dry-run-v1',
      computeMode: 'cpu-visible-record-upload-tf-materialize',
      note: 'Step114 validates the WebGL2 fixed-record TF/readback boundary; full raw gaussian visible evaluation remains CPU-side.',
      recordLayout: {
        floatsPerRecord: RECORD_FLOATS,
        fields: FIELD_LAYOUT.map(([name, offset, length]) => ({ name, offset, length }))
      },
      candidateCount: cpuRecordBuild.candidateCount,
      recordCount: cpuRecordBuild.count,
      validRecordCount: cpuRecordBuild.validCount,
      temporalRejected: cpuRecordBuild.temporalRejected,
      invalidItemCount: cpuRecordBuild.invalidItemCount,
      recordComparison,
      referenceVisibleComparison,
      mismatchClassification,
      anyMismatch: mismatchClassification !== 'none',
      displayCandidateSource,
      gpuCandidateUsedForDisplay: !!gpuCandidateUsedForDisplay,
      limitedDrawUsedForCandidateSource: !!limitedDrawUsedForCandidateSource,
      timing: {
        ...cpuRecordBuild.timing,
        ...tfResult.timing,
        compareMs,
        totalMs: nowMs() - totalStartMs
      },
      metadata
    };
  } catch (error) {
    return buildFallbackSummary(error?.message ?? 'gpu-visible-record-error', {
      sourceMode,
      displayCandidateSource,
      gpuCandidateUsedForDisplay,
      limitedDrawUsedForCandidateSource
    });
  }
}

