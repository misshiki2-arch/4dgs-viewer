export function clamp01(x) {
  return Math.min(1, Math.max(0, x));
}

export function buildDrawArraysFromIndices(visible, drawIndices) {
  const nDraw = drawIndices.length;
  const centers = new Float32Array(nDraw * 2);
  const radii = new Float32Array(nDraw);
  const colors = new Float32Array(nDraw * 4);
  const conics = new Float32Array(nDraw * 3);

  for (let k = 0; k < nDraw; k++) {
    const s = visible[drawIndices[k]];
    centers[2 * k + 0] = s.px;
    centers[2 * k + 1] = s.py;
    radii[k] = s.radius;
    colors[4 * k + 0] = clamp01(s.color[0]);
    colors[4 * k + 1] = clamp01(s.color[1]);
    colors[4 * k + 2] = clamp01(s.color[2]);
    colors[4 * k + 3] = clamp01(s.opacity);
    conics[3 * k + 0] = s.conic[0];
    conics[3 * k + 1] = s.conic[1];
    conics[3 * k + 2] = s.conic[2];
  }

  return {
    nDraw,
    centers,
    radii,
    colors,
    conics
  };
}

export function uploadDrawArrays(gl, gpu, drawData) {
  gl.bindVertexArray(gpu.vao);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpu.centerBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, drawData.centers, gl.DYNAMIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpu.radiusBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, drawData.radii, gl.DYNAMIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpu.colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, drawData.colors, gl.DYNAMIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpu.conicBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, drawData.conics, gl.DYNAMIC_DRAW);
}

export function drawUploadedArrays(gl, gpu, viewportWidth, viewportHeight, nDraw) {
  gl.useProgram(gpu.program);
  gl.uniform2f(gpu.uViewportPx, viewportWidth, viewportHeight);
  gl.bindVertexArray(gpu.vao);
  gl.drawArrays(gl.POINTS, 0, nDraw);
  gl.bindVertexArray(null);
  gl.useProgram(null);
}

export function uploadAndDraw(gl, gpu, drawData, viewportWidth, viewportHeight) {
  uploadDrawArrays(gl, gpu, drawData);
  drawUploadedArrays(gl, gpu, viewportWidth, viewportHeight, drawData.nDraw);
}
