export function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) || '';
    gl.deleteShader(shader);
    throw new Error('Shader compile error: ' + log);
  }
  return shader;
}

export function createProgram(gl, vsSource, fsSource) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSource);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSource);

  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) || '';
    gl.deleteProgram(program);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    throw new Error('Program link error: ' + log);
  }

  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return program;
}

export function createArrayBuffer(gl, data, usage = gl.DYNAMIC_DRAW) {
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data, usage);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  return buffer;
}

export function updateArrayBuffer(gl, buffer, data, usage = gl.DYNAMIC_DRAW) {
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data, usage);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
}

export function bindFloatAttrib(gl, {
  vao,
  program,
  buffer,
  name,
  size,
  type = gl.FLOAT,
  normalized = false,
  stride = 0,
  offset = 0
}) {
  const loc = gl.getAttribLocation(program, name);
  if (loc < 0) {
    throw new Error(`Attribute not found: ${name}`);
  }

  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, size, type, normalized, stride, offset);
  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);

  return loc;
}

export function clearToGray(gl, gray01) {
  gl.clearColor(gray01, gray01, gray01, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT);
}

export function enableStandardAlphaBlend(gl) {
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
}

export function disableDepth(gl) {
  gl.disable(gl.DEPTH_TEST);
}
