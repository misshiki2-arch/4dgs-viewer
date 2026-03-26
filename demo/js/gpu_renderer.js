import { computeGaussianState, computeScreenSplat } from './rot4d_math.js';
import { evalSHColor } from './sh_eval.js';
import {
  createProgram,
  createArrayBuffer,
  updateArrayBuffer,
  bindFloatAttrib,
  clearToGray,
  enableStandardAlphaBlend,
  disableDepth
} from './gpu_gl_utils.js';
import {
  GPU_STEP_VERTEX_SHADER,
  GPU_STEP_FRAGMENT_SHADER
} from './gpu_shaders.js';
import {
  clampInt,
  computeTileGrid,
  computeTileRangeFromAABB,
  buildTileLists,
  summarizeTileLists
} from './gpu_tile_utils.js';

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

  const program = createProgram(gl, GPU_STEP_VERTEX_SHADER, GPU_STEP_FRAGMENT_SHADER);

  const emptyF32 = new Float32Array(0);
  const centerBuffer = createArrayBuffer(gl, emptyF32);
  const radiusBuffer = createArrayBuffer(gl, emptyF32);
  const colorBuffer = createArrayBuffer(gl, emptyF32);
  const conicBuffer = createArrayBuffer(gl, emptyF32);

  const vao = gl.createVertexArray();

  bindFloatAttrib(gl, {
    vao,
    program,
    buffer: centerBuffer,
    name: 'aCenterPx',
    size: 2
  });

  bindFloatAttrib(gl, {
    vao,
    program,
    buffer: radiusBuffer,
    name: 'aRadiusPx',
    size: 1
  });

  bindFloatAttrib(gl, {
    vao,
    program,
    buffer: colorBuffer,
    name: 'aColorAlpha',
    size: 4
  });

  bindFloatAttrib(gl, {
    vao,
    program,
    buffer: conicBuffer,
    name: 'aConic',
    size: 3
  });

  const uViewportPx = gl.getUniformLocation(program, 'uViewportPx');

  const renderer = {
    gl,
    program,
    vao,
    centerBuffer,
    radiusBuffer,
    colorBuffer,
    conicBuffer,
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
    gpu.resize(canvas.width, canvas.height);
    gl.disable(gl.BLEND);
    clearToGray(gl, bg);
    infoEl.textContent = 'GPU Step5 viewer\nNo scene loaded.';
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

  const sx = canvas.width / renderW;
  const sy = canvas.height / renderH;

  const tileGrid = computeTileGrid(canvas.width, canvas.height, 32);
  let minTileX = tileGrid.tileCols, minTileY = tileGrid.tileRows, maxTileX = -1, maxTileY = -1;

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

    const px = splat.px * sx;
    const py = splat.py * sy;
    const radius = Math.max(1.0, splat.radius * Math.max(sx, sy));

    const minX = clampInt(Math.floor(px - radius), 0, canvas.width - 1);
    const maxX = clampInt(Math.ceil(px + radius), 0, canvas.width - 1);
    const minY = clampInt(Math.floor(py - radius), 0, canvas.height - 1);
    const maxY = clampInt(Math.ceil(py + radius), 0, canvas.height - 1);

    const tileRange = computeTileRangeFromAABB(
      [minX, minY, maxX, maxY],
      tileGrid.tileCols,
      tileGrid.tileRows,
      tileGrid.tileSize
    );

    minTileX = Math.min(minTileX, tileRange[0]);
    minTileY = Math.min(minTileY, tileRange[1]);
    maxTileX = Math.max(maxTileX, tileRange[2]);
    maxTileY = Math.max(maxTileY, tileRange[3]);

    visible.push({
      px,
      py,
      radius,
      depth: splat.depth,
      opacity: splat.opacity,
      color,
      conic: [
        splat.conic[0] / (sx * sx),
        splat.conic[1] / (sx * sy),
        splat.conic[2] / (sy * sy)
      ],
      aabb: [minX, minY, maxX, maxY],
      tileRange
    });

    if (visible.length >= maxVisible) break;

    if ((visible.length & 2047) === 0) {
      await new Promise(r => setTimeout(r, 0));
      if (frameToken !== tokenRef.value) return;
    }
  }

  visible.sort((a, b) => b.depth - a.depth);

  const tileData = buildTileLists(visible, tileGrid.tileCols, tileGrid.tileRows);
  const tileSummary = summarizeTileLists(
    tileData,
    tileGrid.tileCols,
    tileGrid.tileRows,
    [minTileX, minTileY, maxTileX, maxTileY]
  );

  const n = visible.length;
  const centers = new Float32Array(n * 2);
  const radii = new Float32Array(n);
  const colors = new Float32Array(n * 4);
  const conics = new Float32Array(n * 3);

  for (let k = 0; k < n; k++) {
    const s = visible[k];
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

  gpu.resize(canvas.width, canvas.height);

  disableDepth(gl);
  enableStandardAlphaBlend(gl);
  clearToGray(gl, bg);

  gl.useProgram(gpu.program);
  gl.uniform2f(gpu.uViewportPx, canvas.width, canvas.height);
  gl.bindVertexArray(gpu.vao);

  updateArrayBuffer(gl, gpu.centerBuffer, centers);
  updateArrayBuffer(gl, gpu.radiusBuffer, radii);
  updateArrayBuffer(gl, gpu.colorBuffer, colors);
  updateArrayBuffer(gl, gpu.conicBuffer, conics);

  gl.drawArrays(gl.POINTS, 0, n);

  gl.bindVertexArray(null);
  gl.useProgram(null);

  const elapsed = performance.now() - t0;
  const avgRefsPerVisible = n > 0 ? (tileSummary.totalRefs / n) : 0;

  infoEl.textContent =
`format=v2
N=${raw.N.toLocaleString()}  visible=${visible.length.toLocaleString()}  stride=${stride}
active_sh_degree=${raw.activeShDegree}  active_sh_degree_t=${raw.activeShDegreeT}
rot_4d(file)=${raw.rot4d}  useRot4d=${useRot4d}  useSH=${useSH}
nativeRot4d=${useNativeRot4d}  nativeMarginal=${useNativeMarginal}
prefilterVar=${prefilterVar.toFixed(2)}  sigmaScale=${sigmaScale.toFixed(2)}
renderScale=${renderScale.toFixed(2)}  canvas=${canvas.width}x${canvas.height}
time=${timestamp.toFixed(2)}  splatScale=${scalingModifier.toFixed(2)}
GPU Step5 render=${elapsed.toFixed(1)} ms

Step5 note:
- CPU computes screen-space splats + AABB
- CPU builds explicit tile->splat lists
- GPU still draws ordered anisotropic conic splats

tileCols=${tileSummary.tileCols}  tileRows=${tileSummary.tileRows}  nonEmptyTiles=${tileSummary.nonEmptyTiles}
totalTileRefs=${tileSummary.totalRefs.toLocaleString()}  avgRefsPerVisible=${avgRefsPerVisible.toFixed(2)}
avgPerNonEmptyTile=${tileSummary.avgPerNonEmptyTile.toFixed(2)}  maxPerTile=${tileSummary.maxPerTile}
activeTileBox=${tileSummary.activeTileBoxText}
offsetsLen=${tileSummary.offsetsLen.toLocaleString()}  indicesLen=${tileSummary.indicesLen.toLocaleString()}
countEnergy=${tileSummary.countEnergy.toLocaleString()}  ${tileSummary.sampleTileText}`;
}
