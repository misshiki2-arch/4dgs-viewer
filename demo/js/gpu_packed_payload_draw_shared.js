import { createProgram } from './gpu_gl_utils.js';
import { GPU_STEP_FRAGMENT_SHADER } from './gpu_shaders.js';

const GPU_PACKED_TEXTURE_VERTEX_SHADER = `#version 300 es
precision highp float;

uniform sampler2D uPackedTexture;
uniform vec2 uViewportPx;

out vec4 vColorAlpha;
out float vRadiusPx;
out vec3 vConic;

void main() {
  int itemIndex = gl_VertexID;
  vec4 row0 = texelFetch(uPackedTexture, ivec2(0, itemIndex), 0);
  vec4 row1 = texelFetch(uPackedTexture, ivec2(1, itemIndex), 0);
  vec4 row2 = texelFetch(uPackedTexture, ivec2(2, itemIndex), 0);

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
  if (gpu[key]?.program && gpu[key]?.vao) return gpu[key];

  const program = createProgram(gl, GPU_PACKED_TEXTURE_VERTEX_SHADER, GPU_STEP_FRAGMENT_SHADER);
  const vao = gl.createVertexArray();
  const resources = {
    program,
    vao,
    uniformViewportPx: gl.getUniformLocation(program, 'uViewportPx'),
    uniformPackedTexture: gl.getUniformLocation(program, 'uPackedTexture')
  };
  gpu[key] = resources;
  return resources;
}

export function drawGpuPackedPayloads(gl, gpu, screenSpace, canvasWidth, canvasHeight, options = {}) {
  const payloads = getValidGpuPackedPayloads(gl, screenSpace);
  if (payloads.length <= 0) return null;

  const resources = ensureGpuPackedPayloadTextureDrawResources(gl, gpu, options.storageKey);
  let drawCount = 0;
  let drawCallCount = 0;

  gl.useProgram(resources.program);
  gl.bindVertexArray(resources.vao);
  gl.uniform2f(resources.uniformViewportPx, canvasWidth, canvasHeight);
  gl.uniform1i(resources.uniformPackedTexture, 0);

  for (const payload of payloads) {
    const count = Number.isFinite(payload?.count) ? Math.max(0, payload.count | 0) : 0;
    if (count <= 0) continue;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, payload.texture);
    gl.drawArrays(gl.POINTS, 0, count);
    drawCount += count;
    drawCallCount++;
  }

  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.bindVertexArray(null);

  return {
    drawCount,
    drawCallCount,
    resources,
    payloadCount: payloads.length
  };
}
