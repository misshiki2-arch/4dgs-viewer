import { computeGaussianState, computeScreenSplat } from './rot4d_math.js';
import { evalSHColor } from './sh_eval.js';

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error('Shader compile error: ' + log);
  }
  return shader;
}

function createProgram(gl, vsSource, fsSource) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSource);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSource);
  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    throw new Error('Program link error: ' + log);
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return program;
}

function clamp01(x) {
  return Math.min(1, Math.max(0, x));
}

export function createGpuRenderer(canvas) {
  const gl = canvas.getContext('webgl2', {
    alpha: false,
    antialias: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false
  });
  if (!gl) {
    throw new Error('WebGL2 is not available in this browser.');
  }

  const vsSource = `#version 300 es
  precision highp float;

  in vec2 aCenterPx;
  in float aRadiusPx;
  in vec4 aColorAlpha;

  uniform vec2 uViewportPx;

  out vec4 vColorAlpha;
  out float vRadiusPx;

  void main() {
    vec2 ndc = vec2(
      (aCenterPx.x / uViewportPx.x) * 2.0 - 1.0,
      1.0 - (aCenterPx.y / uViewportPx.y) * 2.0
    );
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = max(1.0, aRadiusPx * 2.0);
    vColorAlpha = aColorAlpha;
    vRadiusPx = aRadiusPx;
  }`;

  const fsSource = `#version 300 es
  precision highp float;

  in vec4 vColorAlpha;
  in float vRadiusPx;

  out vec4 outColor;

  void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;

    // Minimal Step1 approximation:
    // point sprite circular Gaussian-like falloff
    float alpha = vColorAlpha.a * exp(-2.0 * r2);
    outColor = vec4(vColorAlpha.rgb * alpha, alpha);
  }`;

  const program = createProgram(gl, vsSource, fsSource);

  const centerBuffer = gl.createBuffer();
  const radiusBuffer = gl.createBuffer();
  const colorBuffer = gl.createBuffer();
  const vao = gl.createVertexArray();

  gl.bindVertexArray(vao);

  const aCenterPx = gl.getAttribLocation(program, 'aCenterPx');
  gl.bindBuffer(gl.ARRAY_BUFFER, centerBuffer);
  gl.enableVertexAttribArray(aCenterPx);
  gl.vertexAttribPointer(aCenterPx, 2, gl.FLOAT, false, 0, 0);

  const aRadiusPx = gl.getAttribLocation(program, 'aRadiusPx');
  gl.bindBuffer(gl.ARRAY_BUFFER, radiusBuffer);
  gl.enableVertexAttribArray(aRadiusPx);
  gl.vertexAttribPointer(aRadiusPx, 1, gl.FLOAT, false, 0, 0);

  const aColorAlpha = gl.getAttribLocation(program, 'aColorAlpha');
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.enableVertexAttribArray(aColorAlpha);
  gl.vertexAttribPointer(aColorAlpha, 4, gl.FLOAT, false, 0, 0);

  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);

  const uViewportPx = gl.getUniformLocation(program, 'uViewportPx');

  const renderer = {
    gl,
    program,
    vao,
    centerBuffer,
    radiusBuffer,
    colorBuffer,
    uViewportPx,
    width: canvas.width,
    height: canvas.height,
    resize(width, height) {
      this.width = width;
      this.height = height;
      gl.viewport(0, 0, width, height);
    }
  };

  renderer.resize(canvas.width, canvas.height);
  return renderer;
}

export async function renderGpuFrame({
  raw,
  gpu,
  canvas,
  camera,
  controls,
  ui,
  tokenRef,
  infoEl
}) {
  const gl = gpu.gl;

  const bg255 = parseInt(ui.bgGraySlider.value, 10);
  const bg = bg255 / 255.0;
  const renderScale = parseFloat(ui.renderScaleSlider.value);
  const renderW = Math.max(1, Math.round(canvas.width * renderScale));
  const renderH = Math.max(1, Math.round(canvas.height * renderScale));
  const stride = parseInt(ui.strideSlider.value, 10);
  const maxVisible = parseInt(ui.maxVisibleSlider.value, 10);
  const timestamp = parseFloat(ui.timeSlider.value);
  const scalingModifier = parseFloat(ui.splatScaleSlider.value);
  const sigmaScale = parseFloat(ui.sigmaScaleSlider.value);
  const prefilterVar = parseFloat(ui.prefilterVarSlider.value);
  const useSH = ui.useSHCheck.checked;
  const useRot4d = ui.useRot4dCheck.checked;
  const useNativeRot4d = ui.useNativeRot4dCheck.checked;
  const useNativeMarginal = ui.useNativeMarginalCheck.checked;
  const forceSh3d = ui.forceSh3dCheck.checked;
  const timeDuration = parseFloat(ui.timeDurationSlider.value);

  controls.update();
  camera.updateMatrixWorld(true);

  if (!raw) {
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.disable(gl.BLEND);
    gl.clearColor(bg, bg, bg, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    infoEl.textContent = 'GPU Step1 viewer\nNo scene loaded.';
    return;
  }

  const frameToken = ++tokenRef.value;
  const t0 = performance.now();

  const flags = {
    nativeRot4d: useNativeRot4d,
    nativeMarginal: useNativeMarginal
  };

  const visible = [];
  const camPos = camera.position.clone();

  for (let i = 0; i < raw.N; i += stride) {
    const gs = computeGaussianState(
      raw,
      i,
      timestamp,
      scalingModifier,
      sigmaScale,
      prefilterVar,
      useRot4d,
      flags
    );
    if (!gs) continue;

    const color = evalSHColor(
      raw,
      i,
      camPos,
      gs.pos,
      timestamp,
      timeDuration,
      useSH,
      forceSh3d
    );

    const splat = computeScreenSplat(
      camera,
      gs.pos,
      gs.cov3,
      gs.opacity,
      renderW,
      renderH
    );
    if (!splat) continue;

    visible.push({
      px: splat.px,
      py: splat.py,
      radius: splat.radius,
      depth: splat.depth,
      opacity: splat.opacity,
      color
    });

    if (visible.length >= maxVisible) break;

    if ((visible.length & 2047) === 0) {
      await new Promise(r => setTimeout(r, 0));
      if (frameToken !== tokenRef.value) return;
    }
  }

  // Front-to-back approximation:
  // draw far to near with standard alpha blending
  visible.sort((a, b) => b.depth - a.depth);

  const n = visible.length;
  const centers = new Float32Array(n * 2);
  const radii = new Float32Array(n);
  const colors = new Float32Array(n * 4);

  for (let k = 0; k < n; k++) {
    const s = visible[k];
    centers[2 * k + 0] = s.px / renderW * canvas.width;
    centers[2 * k + 1] = s.py / renderH * canvas.height;
    radii[k] = Math.max(1.0, s.radius / renderW * canvas.width);
    colors[4 * k + 0] = clamp01(s.color[0]);
    colors[4 * k + 1] = clamp01(s.color[1]);
    colors[4 * k + 2] = clamp01(s.color[2]);
    colors[4 * k + 3] = clamp01(s.opacity);
  }

  gpu.resize(canvas.width, canvas.height);

  gl.disable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

  gl.clearColor(bg, bg, bg, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT);

  gl.useProgram(gpu.program);
  gl.uniform2f(gpu.uViewportPx, canvas.width, canvas.height);
  gl.bindVertexArray(gpu.vao);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpu.centerBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, centers, gl.DYNAMIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpu.radiusBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, radii, gl.DYNAMIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpu.colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, colors, gl.DYNAMIC_DRAW);

  gl.drawArrays(gl.POINTS, 0, n);

  gl.bindVertexArray(null);
  gl.useProgram(null);

  const elapsed = performance.now() - t0;
  infoEl.textContent =
`format=v2
N=${raw.N.toLocaleString()}  visible=${visible.length.toLocaleString()}  stride=${stride}
active_sh_degree=${raw.activeShDegree}  active_sh_degree_t=${raw.activeShDegreeT}
rot_4d(file)=${raw.rot4d}  useRot4d=${useRot4d}  useSH=${useSH}
nativeRot4d=${useNativeRot4d}  nativeMarginal=${useNativeMarginal}
prefilterVar=${prefilterVar.toFixed(2)}  sigmaScale=${sigmaScale.toFixed(2)}
renderScale=${renderScale.toFixed(2)}  canvas=${canvas.width}x${canvas.height}
time=${timestamp.toFixed(2)}  splatScale=${scalingModifier.toFixed(2)}
GPU Step1 render=${elapsed.toFixed(1)} ms

Step1 note:
- CPU computes screen-space splats
- GPU draws circular point-sprite splats
- This is an approximate first GPU stage, not the final native conic compositor.`;
}
