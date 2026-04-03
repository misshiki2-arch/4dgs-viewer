// Step22:
// WebGL ユーティリティ。
// Step20.2 の通常 alpha 合成は維持しつつ、
// Step22 では managed buffer の再利用状況を追いやすくする。

export function createShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) || 'Unknown shader compile error';
    gl.deleteShader(shader);
    throw new Error(log);
  }

  return shader;
}

export function createProgram(gl, vertexSource, fragmentSource) {
  const vs = createShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fs = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);

  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);

  gl.deleteShader(vs);
  gl.deleteShader(fs);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) || 'Unknown program link error';
    gl.deleteProgram(program);
    throw new Error(log);
  }

  return program;
}

export function createArrayBuffer(gl, data = null, usage = gl.DYNAMIC_DRAW) {
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data || 0, usage);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  return buffer;
}

export function orphanArrayBuffer(gl, buffer, byteLength, usage = gl.DYNAMIC_DRAW) {
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, Math.max(0, byteLength | 0), usage);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
}

export function uploadArrayBuffer(gl, buffer, data, usage = gl.DYNAMIC_DRAW) {
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data, usage);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
}

export function updateArrayBufferSubData(gl, buffer, data, dstByteOffset = 0) {
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferSubData(gl.ARRAY_BUFFER, Math.max(0, dstByteOffset | 0), data);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
}

export function ensureArrayBufferCapacity(gl, state, requiredByteLength, usage = gl.DYNAMIC_DRAW) {
  if (!state || !state.buffer) {
    throw new Error('ensureArrayBufferCapacity requires { buffer, capacityBytes } state');
  }

  const need = Math.max(0, requiredByteLength | 0);
  state.lastRequiredBytes = need;

  if ((state.capacityBytes | 0) >= need) {
    state.capacityReused = true;
    state.capacityGrown = false;
    return state;
  }

  let nextCapacity = Math.max(256, state.capacityBytes | 0);
  while (nextCapacity < need) {
    nextCapacity *= 2;
  }

  orphanArrayBuffer(gl, state.buffer, nextCapacity, usage);
  state.capacityBytes = nextCapacity;
  state.usage = usage;
  state.capacityReused = false;
  state.capacityGrown = true;
  return state;
}

export function createManagedArrayBuffer(gl, initialByteLength = 0, usage = gl.DYNAMIC_DRAW) {
  const buffer = gl.createBuffer();
  const state = {
    buffer,
    capacityBytes: 0,
    usage,
    lastUploadBytes: 0,
    lastRequiredBytes: 0,
    capacityReused: false,
    capacityGrown: false,
    uploadCount: 0
  };
  ensureArrayBufferCapacity(gl, state, initialByteLength, usage);
  return state;
}

export function uploadManagedArrayBuffer(gl, state, data, usage = state?.usage ?? gl.DYNAMIC_DRAW) {
  if (!state || !state.buffer) {
    throw new Error('uploadManagedArrayBuffer requires managed buffer state');
  }
  const requiredByteLength = data?.byteLength ?? 0;
  ensureArrayBufferCapacity(gl, state, requiredByteLength, usage);

  gl.bindBuffer(gl.ARRAY_BUFFER, state.buffer);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);

  state.usage = usage;
  state.lastUploadBytes = requiredByteLength;
  state.uploadCount = (state.uploadCount | 0) + 1;
  return state;
}

export function summarizeManagedArrayBuffer(state) {
  return {
    capacityBytes: state?.capacityBytes ?? 0,
    lastUploadBytes: state?.lastUploadBytes ?? 0,
    lastRequiredBytes: state?.lastRequiredBytes ?? 0,
    capacityReused: !!state?.capacityReused,
    capacityGrown: !!state?.capacityGrown,
    uploadCount: state?.uploadCount ?? 0
  };
}

export function bindFloatAttrib(gl, {
  vao,
  program,
  buffer,
  name,
  size,
  stride = 0,
  offset = 0,
  normalized = false
}) {
  const loc = gl.getAttribLocation(program, name);
  if (loc < 0) return;

  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, size, gl.FLOAT, normalized, stride, offset);
  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
}

export function clearToGray(gl, gray) {
  gl.clearColor(gray, gray, gray, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT);
}

export function enableStandardAlphaBlend(gl) {
  gl.enable(gl.BLEND);
  gl.blendFuncSeparate(
    gl.SRC_ALPHA,
    gl.ONE_MINUS_SRC_ALPHA,
    gl.ONE,
    gl.ONE_MINUS_SRC_ALPHA
  );
}

export function disableDepth(gl) {
  gl.disable(gl.DEPTH_TEST);
}
