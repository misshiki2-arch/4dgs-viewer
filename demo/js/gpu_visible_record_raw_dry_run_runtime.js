import { buildVisibleItemForCandidate } from './gpu_visible_item_builder.js';
import { computeGaussianState } from './rot4d_math.js';
import { ensureRawAttributeTextures } from './gpu_raw_attribute_texture_runtime.js';

const stateByGl = new WeakMap();
const RAW_RECORD_FLOATS = 48;
const DEFAULT_MAX_RECORDS = 65536;
const DEFAULT_EPSILON = 1e-3;
const DEFAULT_DEBUG_SAMPLE_COUNT = 4;
const IMPLEMENTED_FIELDS = ['srcIndex', 'valid', 'px', 'py', 'depth', 'aabb', 'radius', 'conic', 'alpha', 'tileRange'];
const DEFERRED_FIELDS = ['colorRgb', 'colorAlpha.rgb', 'SH', '4D conditional covariance full parity', 'depth sort', 'variable packing', 'tile-list GPU generation'];
const PACKED_LIKE_FLOATS_PER_ITEM = 16;
const PACKED_LIKE_IMPLEMENTED_FIELDS = [
  'centerPx',
  'radiusPx',
  'depth',
  'colorAlpha.a',
  'conic',
  'misc.aabb'
];
const PACKED_LIKE_DEFERRED_FIELDS = [
  'colorAlpha.rgb',
  'sorted-visible-order',
  'variable packing',
  'tile-list GPU generation',
  'tile composite connection'
];
const PACKED_LIKE_FIELD_LAYOUT = [
  ['centerPx', 0, 2],
  ['radiusPx', 2, 1],
  ['depth', 3, 1],
  ['colorAlpha.rgb', 4, 3],
  ['colorAlpha.a', 7, 1],
  ['conic', 8, 3],
  ['reserved', 11, 1],
  ['misc.aabb', 12, 4]
];
const DEBUG_LAYOUT = {
  rawTexCoord: 18,
  rawXyzOpacity: 20,
  rawScaleTime: 24,
  rawTimeScale: 28,
  rawRotation: 30,
  rawRotationR: 34,
  computedPosition: 38,
  marginalT: 41,
  viewSpacePosition: 42,
  radius: 45,
  shaderRejectBits: 46
};
const FIELD_LAYOUT = [
  ['srcIndex', 0, 1],
  ['valid', 1, 1],
  ['px', 2, 1],
  ['py', 3, 1],
  ['depth', 4, 1],
  ['aabb', 5, 4],
  ['radius', 9, 1],
  ['conic', 10, 3],
  ['alpha', 13, 1],
  ['tileRange', 14, 4]
];
const MINIMAL_FETCH_RECORD_FLOATS = 8;
const FIXED_RECORD_COMPARISON_REFERENCE = Object.freeze({
  cpuReferenceMode: 'f32-fixed-record-materialized',
  aabbReferenceMode: 'recomputed-from-f32-px-py-radius',
  canonicalCpuAabbMode: 'buildVisibleItemForCandidate.aabb',
  note: 'CPU canonical aabb and GPU/f32 fixed-record aabb can differ by 1px on integer rounding boundaries.'
});
const PACKED_LIKE_COMPARISON_REFERENCE = Object.freeze({
  cpuReferenceMode: 'packed-like-f32-fixed-record-materialized',
  packedLayoutVersion: 2,
  packedFloatsPerItem: PACKED_LIKE_FLOATS_PER_ITEM,
  orderMode: 'candidate-order-unsorted',
  colorRgbMode: 'deferred-zero-filled',
  alphaMode: 'colorAlpha.a',
  miscMode: 'aabb',
  note: 'Packed-like dry-run follows the v2 16-float layout shape in candidate order; RGB/SH, sorting, compaction, and tile-list generation remain deferred.'
});

const DISPLAY_CONNECTION_SATISFIED_ITEMS = [
  'raw-attribute-texture-fetch',
  'transform-feedback-fixed-record-output',
  'packed-layout-v2-shape',
  'centerPx-radius-depth-alpha-conic-miscAabb-fields',
  'validated-only-gpu-candidate-display-source',
  'cpu-fallback-available'
];

const DISPLAY_CONNECTION_UNRESOLVED_ITEMS = [
  'colorAlpha.rgb-deferred',
  'SH-color-parity-deferred',
  'cpu-tile-list-index-contract',
  'known-aabb-rounding-boundary-diff'
];

const DISPLAY_CONNECTION_BLOCKED_ITEMS = [
  'candidate-order-unsorted',
  'variable-packing-not-implemented',
  'gpu-tile-list-generation-not-implemented',
  'tile-composite-input-contract-not-switched'
];

const DISPLAY_CONNECTION_WEBGL2_LIMIT_CANDIDATES = [
  'depth-sort',
  'variable-packing-compaction',
  'prefix-sum-tile-list-build',
  'tile-composite-rewire'
];

const DISPLAY_CONNECTION_WEBGPU_MIGRATION_SIGNALS = [
  'storage-buffer-visible-records',
  'compute-prefix-sum-compaction',
  'gpu-depth-sort',
  'gpu-tile-binning',
  'compute-driven-tile-list-generation'
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

function createRawVisibleRecordState(gl) {
  const vertexSource = `#version 300 es
precision highp float;
in float aSrcIndex;
in vec2 aRawTexCoord;
uniform sampler2D uXyzOpacityTex;
uniform sampler2D uScaleTimeTex;
uniform sampler2D uTimeScaleTex;
uniform sampler2D uRotationTex;
uniform sampler2D uRotationRTex;
uniform float uTextureWidth;
uniform float uTextureHeight;
uniform float uRawCount;
uniform float uTimestamp;
uniform float uScalingModifier;
uniform float uSigmaScale;
uniform float uPrefilterVar;
uniform float uTemporalSigmaThreshold;
uniform float uRenderW;
uniform float uRenderH;
uniform float uCanvasW;
uniform float uCanvasH;
uniform float uSx;
uniform float uSy;
uniform float uFx;
uniform float uFy;
uniform float uCx;
uniform float uCy;
uniform float uCovFocalX;
uniform float uCovFocalY;
uniform float uCovTanFovX;
uniform float uCovTanFovY;
uniform float uPixelXSign;
uniform vec4 uViewRow0;
uniform vec4 uViewRow1;
uniform vec4 uViewRow2;
uniform vec3 uViewRotRow0;
uniform vec3 uViewRotRow1;
uniform vec3 uViewRotRow2;
uniform float uTileRangeEnabled;
uniform float uTileCols;
uniform float uTileRows;
uniform float uTileSize;
out vec4 vRecord0;
out vec4 vRecord1;
out vec4 vRecord2;
out vec4 vRecord3;
out vec4 vRecord4;
out vec4 vDebug0;
out vec4 vDebug1;
out vec4 vDebug2;
out vec4 vDebug3;
out vec4 vDebug4;
out vec4 vDebug5;
out vec4 vDebug6;

vec4 fetchRaw(sampler2D tex, vec2 coord) {
  ivec2 size = ivec2(max(1, int(uTextureWidth)), max(1, int(uTextureHeight)));
  ivec2 xy = ivec2(
    int(floor(coord.x + 0.5)),
    int(floor(coord.y + 0.5))
  );
  xy = clamp(xy, ivec2(0), size - ivec2(1));
  return texelFetch(tex, xy, 0);
}

vec4 normalizeQuat(vec4 q) {
  float n = max(length(q), 1e-8);
  return q / n;
}

float sigmoid(float x) {
  return 1.0 / (1.0 + exp(-x));
}

void buildRotationRows(vec4 qLIn, vec4 qRIn, out vec4 r0, out vec4 r1, out vec4 r2, out vec4 r3) {
  vec4 qL = normalizeQuat(qLIn);
  vec4 qR = normalizeQuat(qRIn);
  float a = qL.x, b = qL.y, c = qL.z, d = qL.w;
  float p = qR.x, q = qR.y, r = qR.z, s = qR.w;
  vec4 ml0 = vec4( a, -b,  c, -d);
  vec4 ml1 = vec4( b,  a, -d, -c);
  vec4 ml2 = vec4(-c,  d,  a, -b);
  vec4 ml3 = vec4( d,  c,  b,  a);
  vec4 mr0 = vec4( p, -q,  r,  s);
  vec4 mr1 = vec4( q,  p, -s,  r);
  vec4 mr2 = vec4(-r,  s,  p,  q);
  vec4 mr3 = vec4(-s, -r, -q,  p);
  r0 = mr0.x * ml0 + mr0.y * ml1 + mr0.z * ml2 + mr0.w * ml3;
  r1 = mr1.x * ml0 + mr1.y * ml1 + mr1.z * ml2 + mr1.w * ml3;
  r2 = mr2.x * ml0 + mr2.y * ml1 + mr2.z * ml2 + mr2.w * ml3;
  r3 = mr3.x * ml0 + mr3.y * ml1 + mr3.z * ml2 + mr3.w * ml3;
}

mat3 computeCov3(vec3 scaleXYZ, float scaleT, vec4 qL, vec4 qR, out float covTOut) {
  vec4 r0, r1, r2, r3;
  buildRotationRows(qL, qR, r0, r1, r2, r3);
  vec4 m0 = scaleXYZ.x * r0;
  vec4 m1 = scaleXYZ.y * r1;
  vec4 m2 = scaleXYZ.z * r2;
  vec4 m3 = scaleT * r3;
  vec4 col0 = vec4(m0.x, m1.x, m2.x, m3.x);
  vec4 col1 = vec4(m0.y, m1.y, m2.y, m3.y);
  vec4 col2 = vec4(m0.z, m1.z, m2.z, m3.z);
  vec4 col3 = vec4(m0.w, m1.w, m2.w, m3.w);
  float covT = max(dot(col3, col3), 1e-8);
  covTOut = covT;
  vec3 cov12 = vec3(dot(col0, col3), dot(col1, col3), dot(col2, col3));
  mat3 cov11 = mat3(
    dot(col0, col0), dot(col0, col1), dot(col0, col2),
    dot(col1, col0), dot(col1, col1), dot(col1, col2),
    dot(col2, col0), dot(col2, col1), dot(col2, col2)
  );
  return cov11 - outerProduct(cov12, cov12) / covT;
}

vec3 computeConditionalPos(vec3 pos0, vec3 scaleXYZ, float scaleT, vec4 qL, vec4 qR, float dt) {
  vec4 r0, r1, r2, r3;
  buildRotationRows(qL, qR, r0, r1, r2, r3);
  vec4 m0 = scaleXYZ.x * r0;
  vec4 m1 = scaleXYZ.y * r1;
  vec4 m2 = scaleXYZ.z * r2;
  vec4 m3 = scaleT * r3;
  vec4 col0 = vec4(m0.x, m1.x, m2.x, m3.x);
  vec4 col1 = vec4(m0.y, m1.y, m2.y, m3.y);
  vec4 col2 = vec4(m0.z, m1.z, m2.z, m3.z);
  vec4 col3 = vec4(m0.w, m1.w, m2.w, m3.w);
  float covT = max(dot(col3, col3), 1e-8);
  vec3 cov12 = vec3(dot(col0, col3), dot(col1, col3), dot(col2, col3));
  return pos0 + cov12 / covT * dt;
}

vec3 mulCov(mat3 cov3, vec3 v) {
  return vec3(
    dot(vec3(cov3[0][0], cov3[1][0], cov3[2][0]), v),
    dot(vec3(cov3[0][1], cov3[1][1], cov3[2][1]), v),
    dot(vec3(cov3[0][2], cov3[1][2], cov3[2][2]), v)
  );
}

void main() {
  float srcIndex = aSrcIndex;
  float valid = 0.0;
  float px = 0.0;
  float py = 0.0;
  float depth = 0.0;
  vec4 aabb = vec4(0.0);
  float recordRadius = 0.0;
  vec3 recordConic = vec3(0.0);
  float recordAlpha = 0.0;
  vec4 recordTileRange = vec4(0.0);
  vec4 debugXyzOpacity = vec4(0.0);
  vec4 debugScaleTime = vec4(0.0);
  vec4 debugTimeScale = vec4(0.0);
  vec4 debugQL = vec4(0.0);
  vec4 debugQR = vec4(0.0);
  vec3 debugPos = vec3(0.0);
  vec3 debugView = vec3(0.0);
  float debugMarginalT = 0.0;
  float debugRadius = 0.0;
  float debugRejectBits = 0.0;
  if (srcIndex >= 0.0 && srcIndex < uRawCount) {
    vec4 xyzOpacity = fetchRaw(uXyzOpacityTex, aRawTexCoord);
    vec4 scaleTime = fetchRaw(uScaleTimeTex, aRawTexCoord);
    vec4 timeScale = fetchRaw(uTimeScaleTex, aRawTexCoord);
    vec4 qL = fetchRaw(uRotationTex, aRawTexCoord);
    vec4 qR = fetchRaw(uRotationRTex, aRawTexCoord);
    debugXyzOpacity = xyzOpacity;
    debugScaleTime = scaleTime;
    debugTimeScale = timeScale;
    debugQL = qL;
    debugQR = qR;
    vec3 scaleXYZ = max(scaleTime.xyz * uScalingModifier, vec3(1e-6));
    float scaleTForCull = max(timeScale.y * uSigmaScale, 1e-6);
    float dt = uTimestamp - timeScale.x;
    bool ok = true;
    float scaleT = max(timeScale.y * uScalingModifier * uSigmaScale, 1e-6);
    vec3 pos = computeConditionalPos(xyzOpacity.xyz, scaleXYZ, scaleT, qL, qR, dt);
    float covT = 0.0;
    mat3 cov3 = computeCov3(scaleXYZ, scaleT, qL, qR, covT);
    float marginalDenom = uPrefilterVar > 0.0 ? uPrefilterVar + covT : covT;
    float marginalT = exp(-0.5 * dt * dt / max(1e-8, marginalDenom));
    ok = ok && marginalT > 0.05;
    if (!(marginalT > 0.05)) debugRejectBits += 2.0;
    debugPos = pos;
    debugMarginalT = marginalT;

    vec4 mv4 = vec4(
      dot(uViewRow0, vec4(pos, 1.0)),
      dot(uViewRow1, vec4(pos, 1.0)),
      dot(uViewRow2, vec4(pos, 1.0)),
      1.0
    );
    depth = mv4.z;
    ok = ok && depth > 1e-6;
    if (!(depth > 1e-6)) debugRejectBits += 4.0;
    debugView = mv4.xyz;
    float txtz = clamp(mv4.x / max(depth, 1e-8), -1.3 * uCovTanFovX, 1.3 * uCovTanFovX);
    float tytz = clamp(mv4.y / max(depth, 1e-8), -1.3 * uCovTanFovY, 1.3 * uCovTanFovY);
    float tx = txtz * depth;
    float ty = tytz * depth;
    vec3 j0 = vec3(uCovFocalX / depth, 0.0, -(uCovFocalX * tx) / (depth * depth));
    vec3 j1 = vec3(0.0, uCovFocalY / depth, -(uCovFocalY * ty) / (depth * depth));
    vec3 t0 = j0.x * uViewRotRow0 + j0.z * uViewRotRow2;
    vec3 t1 = j1.y * uViewRotRow1 + j1.z * uViewRotRow2;
    float covA = dot(t0, mulCov(cov3, t0)) + 0.3;
    float covB = dot(t0, mulCov(cov3, t1));
    float covC = dot(t1, mulCov(cov3, t1)) + 0.3;
    float det = covA * covC - covB * covB;
    ok = ok && det > 0.0 && abs(det) < 1.0e30;
    if (!(det > 0.0 && abs(det) < 1.0e30)) debugRejectBits += 8.0;
    float mid = 0.5 * (covA + covC);
    float invDet = 1.0 / max(det, 1.0e-30);
    recordConic = vec3(
      covC * invDet / max(uSx * uSx, 1.0e-30),
      -covB * invDet / max(uSx * uSy, 1.0e-30),
      covA * invDet / max(uSy * uSy, 1.0e-30)
    );
    float lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    float radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));
    ok = ok && radius > 0.4 && radius <= 4096.0;
    if (!(radius > 0.4 && radius <= 4096.0)) debugRejectBits += 16.0;
    debugRadius = radius;
    px = (uPixelXSign * uFx * (mv4.x / depth) + uCx) * uSx;
    py = (uFy * (mv4.y / depth) + uCy) * uSy;
    float drawRadius = radius * max(uSx, uSy);
    float coverageRadius = max(1.0, drawRadius);
    recordRadius = drawRadius;
    recordAlpha = sigmoid(xyzOpacity.w) * marginalT;
    aabb = vec4(
      clamp(floor(px - coverageRadius), 0.0, uCanvasW - 1.0),
      clamp(floor(py - coverageRadius), 0.0, uCanvasH - 1.0),
      clamp(ceil(px + coverageRadius), 0.0, uCanvasW - 1.0),
      clamp(ceil(py + coverageRadius), 0.0, uCanvasH - 1.0)
    );
    if (uTileRangeEnabled > 0.5) {
      float safeTileSize = max(1.0, uTileSize);
      recordTileRange = vec4(
        clamp(floor(aabb.x / safeTileSize), 0.0, uTileCols - 1.0),
        clamp(floor(aabb.y / safeTileSize), 0.0, uTileRows - 1.0),
        clamp(floor(aabb.z / safeTileSize), 0.0, uTileCols - 1.0),
        clamp(floor(aabb.w / safeTileSize), 0.0, uTileRows - 1.0)
      );
    }
    valid = ok ? 1.0 : 0.0;
  }
  vRecord0 = vec4(srcIndex, valid, px, py);
  vRecord1 = vec4(depth, aabb.xyz);
  vRecord2 = vec4(aabb.w, recordRadius, recordConic.xy);
  vRecord3 = vec4(recordConic.z, recordAlpha, recordTileRange.xy);
  vRecord4 = vec4(recordTileRange.zw, aRawTexCoord);
  vDebug0 = debugXyzOpacity;
  vDebug1 = debugScaleTime;
  vDebug2 = vec4(debugTimeScale.xy, debugQL.xy);
  vDebug3 = vec4(debugQL.zw, debugQR.xy);
  vDebug4 = vec4(debugQR.zw, debugPos.xy);
  vDebug5 = vec4(debugPos.z, debugMarginalT, debugView.xy);
  vDebug6 = vec4(debugView.z, debugRadius, debugRejectBits, 0.0);
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
    'vRecord0',
    'vRecord1',
    'vRecord2',
    'vRecord3',
    'vRecord4',
    'vDebug0',
    'vDebug1',
    'vDebug2',
    'vDebug3',
    'vDebug4',
    'vDebug5',
    'vDebug6'
  ], gl.INTERLEAVED_ATTRIBS);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) || 'unknown raw visible record program link error';
    gl.deleteProgram(program);
    throw new Error(log);
  }
  const uniforms = {};
  for (const name of [
    'uXyzOpacityTex', 'uScaleTimeTex', 'uTimeScaleTex', 'uRotationTex', 'uRotationRTex',
    'uTextureWidth', 'uTextureHeight', 'uRawCount',
    'uTimestamp', 'uScalingModifier', 'uSigmaScale', 'uPrefilterVar', 'uTemporalSigmaThreshold',
    'uRenderW', 'uRenderH', 'uCanvasW', 'uCanvasH', 'uSx', 'uSy', 'uFx', 'uFy', 'uCx', 'uCy',
    'uCovFocalX', 'uCovFocalY', 'uCovTanFovX', 'uCovTanFovY', 'uPixelXSign',
    'uViewRow0', 'uViewRow1', 'uViewRow2', 'uViewRotRow0', 'uViewRotRow1', 'uViewRotRow2',
    'uTileRangeEnabled', 'uTileCols', 'uTileRows', 'uTileSize'
  ]) {
    uniforms[name] = gl.getUniformLocation(program, name);
  }
  return {
    program,
    uniforms,
    vao: gl.createVertexArray(),
    srcIndexBuffer: gl.createBuffer(),
    rawTexCoordBuffer: gl.createBuffer(),
    outputBuffer: gl.createBuffer(),
    transformFeedback: gl.createTransformFeedback(),
    aSrcIndex: gl.getAttribLocation(program, 'aSrcIndex'),
    aRawTexCoord: gl.getAttribLocation(program, 'aRawTexCoord'),
    minimalFetchProbe: createMinimalFetchProbeState(gl)
  };
}

function createMinimalFetchProbeState(gl) {
  const vertexSource = `#version 300 es
precision highp float;
in vec2 aRawTexCoord;
uniform sampler2D uXyzOpacityTex;
uniform float uTextureWidth;
uniform float uTextureHeight;
out vec4 vFetch;
out vec4 vCoord;
void main() {
  ivec2 size = ivec2(max(1, int(uTextureWidth)), max(1, int(uTextureHeight)));
  ivec2 xy = ivec2(
    int(floor(aRawTexCoord.x + 0.5)),
    int(floor(aRawTexCoord.y + 0.5))
  );
  xy = clamp(xy, ivec2(0), size - ivec2(1));
  vFetch = texelFetch(uXyzOpacityTex, xy, 0);
  vCoord = vec4(float(xy.x), float(xy.y), uTextureWidth, uTextureHeight);
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
  gl.transformFeedbackVaryings(program, ['vFetch', 'vCoord'], gl.INTERLEAVED_ATTRIBS);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) || 'unknown raw minimal fetch probe program link error';
    gl.deleteProgram(program);
    throw new Error(log);
  }
  return {
    program,
    vao: gl.createVertexArray(),
    rawTexCoordBuffer: gl.createBuffer(),
    outputBuffer: gl.createBuffer(),
    transformFeedback: gl.createTransformFeedback(),
    aRawTexCoord: gl.getAttribLocation(program, 'aRawTexCoord'),
    uniforms: {
      uXyzOpacityTex: gl.getUniformLocation(program, 'uXyzOpacityTex'),
      uTextureWidth: gl.getUniformLocation(program, 'uTextureWidth'),
      uTextureHeight: gl.getUniformLocation(program, 'uTextureHeight')
    }
  };
}

function getRawVisibleRecordState(gl) {
  let state = stateByGl.get(gl);
  if (!state) {
    state = createRawVisibleRecordState(gl);
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
  if (value == null || value === '') return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(0, n | 0) : fallback;
}

function toFiniteNumber(value, fallback) {
  if (value == null || value === '') return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function finiteOrNull(value) {
  return Number.isFinite(value) ? Number(value) : null;
}

function clampInt(value, min, max) {
  return Math.min(max, Math.max(min, value | 0));
}

function readRawComponents(array, index, dim, count, fallback = 0) {
  const out = [];
  for (let i = 0; i < count; i++) {
    const value = array && dim > i ? array[index * dim + i] : fallback;
    out.push(finiteOrNull(value) ?? fallback);
  }
  return out;
}

function readRecordVec(records, row, offset, count) {
  const base = row * RAW_RECORD_FLOATS + offset;
  const out = [];
  for (let i = 0; i < count; i++) out.push(finiteOrNull(records[base + i]));
  return out;
}

function applyRows4(rows, vector4) {
  if (!Array.isArray(rows) || rows.length < 3) return null;
  return rows.slice(0, 3).map((row) => {
    if (!Array.isArray(row) || row.length < 4) return null;
    return row[0] * vector4[0] + row[1] * vector4[1] + row[2] * vector4[2] + row[3] * vector4[3];
  });
}

function buildCpuDebugForRow({
  raw,
  srcIndex,
  row,
  rawTexCoord,
  records,
  screenSpaceCamera,
  buildConfig,
  flags
}) {
  const state = computeGaussianState(
    raw,
    srcIndex,
    buildConfig.timestamp,
    buildConfig.scalingModifier,
    buildConfig.sigmaScale,
    buildConfig.prefilterVar,
    buildConfig.useRot4d,
    flags
  );
  const viewRows = screenSpaceCamera?.screenSpaceTransformOverride?.viewMatrix ?? null;
  const viewSpace = state?.pos
    ? applyRows4(viewRows, [state.pos[0], state.pos[1], state.pos[2], 1])
    : null;
  return {
    row,
    srcIndex,
    rawTexCoord,
    raw: {
      xyz: readRawComponents(raw.xyz, srcIndex, raw.xyzDim, 3),
      opacity: readRawComponents(raw.opacity, srcIndex, raw.opacityDim, 1),
      scale: readRawComponents(raw.scale_xyz, srcIndex, raw.scaleXYZDim, 3, 1),
      time: readRawComponents(raw.t, srcIndex, raw.tDim, 1),
      scaleT: readRawComponents(raw.scale_t, srcIndex, raw.scaleTDim, 1, 1),
      rotation: readRawComponents(raw.rotation, srcIndex, raw.rotationDim, 4, 0),
      rotationR: readRawComponents(raw.rotation_r, srcIndex, raw.rotationRDim, 4, 0)
    },
    state: state
      ? {
          position: state.pos.map(finiteOrNull),
          viewSpacePosition: Array.isArray(viewSpace) ? viewSpace.map(finiteOrNull) : null,
          opacity: finiteOrNull(state.opacity),
          stateConvention: state.stateConvention ?? null,
          usedCuda4DStateHelper: !!state.usedCuda4DStateHelper
        }
      : {
          position: null,
          viewSpacePosition: null,
          opacity: null,
          cullReason: 'computeGaussianState-returned-null'
        },
    record: {
      srcIndex: finiteOrNull(records[row * RAW_RECORD_FLOATS + 0]),
      valid: finiteOrNull(records[row * RAW_RECORD_FLOATS + 1]),
      px: finiteOrNull(records[row * RAW_RECORD_FLOATS + 2]),
      py: finiteOrNull(records[row * RAW_RECORD_FLOATS + 3]),
      depth: finiteOrNull(records[row * RAW_RECORD_FLOATS + 4]),
      aabb: readRecordVec(records, row, 5, 4),
      radius: finiteOrNull(records[row * RAW_RECORD_FLOATS + 9]),
      conic: readRecordVec(records, row, 10, 3),
      alpha: finiteOrNull(records[row * RAW_RECORD_FLOATS + 13]),
      tileRange: readRecordVec(records, row, 14, 4)
    }
  };
}

function buildGpuDebugForRow(records, row) {
  const recordBase = row * RAW_RECORD_FLOATS;
  return {
    record: {
      srcIndex: finiteOrNull(records[recordBase + 0]),
      valid: finiteOrNull(records[recordBase + 1]),
      px: finiteOrNull(records[recordBase + 2]),
      py: finiteOrNull(records[recordBase + 3]),
      depth: finiteOrNull(records[recordBase + 4]),
      aabb: readRecordVec(records, row, 5, 4),
      radius: finiteOrNull(records[recordBase + 9]),
      conic: readRecordVec(records, row, 10, 3),
      alpha: finiteOrNull(records[recordBase + 13]),
      tileRange: readRecordVec(records, row, 14, 4)
    },
    rawTexCoord: readRecordVec(records, row, DEBUG_LAYOUT.rawTexCoord, 2),
    raw: {
      xyz: [
        ...readRecordVec(records, row, DEBUG_LAYOUT.rawXyzOpacity, 3)
      ],
      opacity: [finiteOrNull(records[recordBase + DEBUG_LAYOUT.rawXyzOpacity + 3])],
      scale: readRecordVec(records, row, DEBUG_LAYOUT.rawScaleTime, 3),
      time: [finiteOrNull(records[recordBase + DEBUG_LAYOUT.rawScaleTime + 3])],
      scaleT: [finiteOrNull(records[recordBase + DEBUG_LAYOUT.rawTimeScale + 1])],
      rotation: readRecordVec(records, row, DEBUG_LAYOUT.rawRotation, 4),
      rotationR: readRecordVec(records, row, DEBUG_LAYOUT.rawRotationR, 4)
    },
    state: {
      position: readRecordVec(records, row, DEBUG_LAYOUT.computedPosition, 3),
      marginalT: finiteOrNull(records[recordBase + DEBUG_LAYOUT.marginalT]),
      viewSpacePosition: readRecordVec(records, row, DEBUG_LAYOUT.viewSpacePosition, 3),
      radius: finiteOrNull(records[recordBase + DEBUG_LAYOUT.radius]),
      shaderOk: finiteOrNull(records[recordBase + 1]),
      rejectBits: finiteOrNull(records[recordBase + DEBUG_LAYOUT.shaderRejectBits])
    }
  };
}

function buildRawVisibleDebugSamples({
  raw,
  cpuRecords,
  gpuRecords,
  screenSpaceCamera,
  buildConfig,
  textureWidth,
  maxSamples = DEFAULT_DEBUG_SAMPLE_COUNT
}) {
  const flags = {
    nativeRot4d: !!buildConfig.useNativeRot4d,
    nativeMarginal: !!buildConfig.useNativeMarginal
  };
  const sampleCount = Math.min(cpuRecords.count, Math.max(0, maxSamples | 0));
  const samples = [];
  for (let row = 0; row < sampleCount; row++) {
    const srcIndex = cpuRecords.sourceIndices[row] >>> 0;
    const rawTexCoord = [
      srcIndex % Math.max(1, textureWidth | 0),
      Math.floor(srcIndex / Math.max(1, textureWidth | 0))
    ];
    samples.push({
      row,
      srcIndex,
      cpu: buildCpuDebugForRow({
        raw,
        srcIndex,
        row,
        rawTexCoord,
        records: cpuRecords.records,
        screenSpaceCamera,
        buildConfig,
        flags
      }),
      gpu: buildGpuDebugForRow(gpuRecords, row)
    });
  }
  return {
    schemaVersion: 'step116-raw-visible-record-debug-samples-v1',
    sampleCount,
    fields: [
      'srcIndex',
      'rawTexCoord',
      'raw.xyz',
      'raw.opacity',
      'raw.scale',
      'raw.time',
      'raw.scaleT',
      'raw.rotation',
      'raw.rotationR',
      'state.position',
      'state.viewSpacePosition',
      'record.px',
      'record.py',
      'record.depth',
      'record.valid',
      'record.aabb',
      'record.radius',
      'record.conic',
      'record.alpha',
      'record.tileRange'
    ],
    samples
  };
}

function classifyFirstDebugDivergence(debugSamples, epsilon = DEFAULT_EPSILON) {
  const samples = Array.isArray(debugSamples?.samples) ? debugSamples.samples : [];
  for (const sample of samples) {
    const cpuRaw = sample.cpu?.raw ?? {};
    const gpuRaw = sample.gpu?.raw ?? {};
    for (const field of ['xyz', 'opacity', 'scale', 'time', 'scaleT', 'rotation', 'rotationR']) {
      const a = Array.isArray(cpuRaw[field]) ? cpuRaw[field] : [];
      const b = Array.isArray(gpuRaw[field]) ? gpuRaw[field] : [];
      const n = Math.max(a.length, b.length);
      for (let i = 0; i < n; i++) {
        const abs = Math.abs(Number(a[i]) - Number(b[i]));
        if (!(abs <= epsilon)) {
          return {
            stage: 'raw-attribute-fetch',
            row: sample.row,
            srcIndex: sample.srcIndex,
            field,
            component: n > 1 ? i : null,
            cpu: finiteOrNull(a[i]),
            gpu: finiteOrNull(b[i]),
            absError: finiteOrNull(abs)
          };
        }
      }
    }

    const cpuState = sample.cpu?.state ?? {};
    const gpuState = sample.gpu?.state ?? {};
    for (const field of ['position', 'viewSpacePosition']) {
      const a = Array.isArray(cpuState[field]) ? cpuState[field] : [];
      const b = Array.isArray(gpuState[field]) ? gpuState[field] : [];
      const n = Math.max(a.length, b.length);
      for (let i = 0; i < n; i++) {
        const abs = Math.abs(Number(a[i]) - Number(b[i]));
        if (!(abs <= epsilon)) {
          return {
            stage: field === 'position' ? '4d-state' : 'view-transform',
            row: sample.row,
            srcIndex: sample.srcIndex,
            field,
            component: n > 1 ? i : null,
            cpu: finiteOrNull(a[i]),
            gpu: finiteOrNull(b[i]),
            absError: finiteOrNull(abs)
          };
        }
      }
    }

    const cpuRecord = sample.cpu?.record ?? {};
    const gpuRecord = sample.gpu?.record ?? {};
    for (const field of ['valid', 'px', 'py', 'depth', 'aabb', 'radius', 'conic', 'alpha', 'tileRange']) {
      const a = ['aabb', 'conic', 'tileRange'].includes(field) ? cpuRecord[field] : [cpuRecord[field]];
      const b = ['aabb', 'conic', 'tileRange'].includes(field) ? gpuRecord[field] : [gpuRecord[field]];
      const n = Math.max(a?.length ?? 0, b?.length ?? 0);
      for (let i = 0; i < n; i++) {
        const abs = Math.abs(Number(a[i]) - Number(b[i]));
        if (!(abs <= epsilon)) {
          return {
            stage: ['aabb', 'tileRange'].includes(field) ? field : 'screen-record',
            row: sample.row,
            srcIndex: sample.srcIndex,
            field,
            component: n > 1 ? i : null,
            cpu: finiteOrNull(a[i]),
            gpu: finiteOrNull(b[i]),
            absError: finiteOrNull(abs)
          };
        }
      }
    }
  }
  return null;
}

function buildCpuMinimalRecords({
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
}) {
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
  const records = new Float32Array(count * RAW_RECORD_FLOATS);
  let validCount = 0;
  const startMs = nowMs();
  for (let row = 0; row < count; row++) {
    const srcIndex = sourceIndices[row] >>> 0;
    const base = row * RAW_RECORD_FLOATS;
    records[base + 0] = srcIndex;
    const item = buildVisibleItemForCandidate({
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
    records[base + 1] = item ? 1 : 0;
    if (!item) continue;
    validCount++;
    const recordPx = Math.fround(item.px);
    const recordPy = Math.fround(item.py);
    const recordDepth = Math.fround(item.depth);
    const recordCoverageRadius = Math.max(1.0, Math.fround(item.radius));
    records[base + 2] = recordPx;
    records[base + 3] = recordPy;
    records[base + 4] = recordDepth;
    // This is the CPU reference materialized for GPU/f32 fixed-record comparison,
    // not the canonical buildVisibleItemForCandidate().aabb value.
    records[base + 5] = clampInt(Math.floor(recordPx - recordCoverageRadius), 0, canvasWidth - 1);
    records[base + 6] = clampInt(Math.floor(recordPy - recordCoverageRadius), 0, canvasHeight - 1);
    records[base + 7] = clampInt(Math.ceil(recordPx + recordCoverageRadius), 0, canvasWidth - 1);
    records[base + 8] = clampInt(Math.ceil(recordPy + recordCoverageRadius), 0, canvasHeight - 1);
    records[base + 9] = Math.fround(item.radius);
    records[base + 10] = Math.fround(Array.isArray(item.conic) ? item.conic[0] : 0);
    records[base + 11] = Math.fround(Array.isArray(item.conic) ? item.conic[1] : 0);
    records[base + 12] = Math.fround(Array.isArray(item.conic) ? item.conic[2] : 0);
    const alpha = Array.isArray(item.colorAlpha) && Number.isFinite(item.colorAlpha[3])
      ? item.colorAlpha[3]
      : (Number.isFinite(item.opacity) ? item.opacity : 0);
    records[base + 13] = Math.fround(alpha);
    if (Array.isArray(item.tileRange) && item.tileRange.length >= 4) {
      records[base + 14] = item.tileRange[0];
      records[base + 15] = item.tileRange[1];
      records[base + 16] = item.tileRange[2];
      records[base + 17] = item.tileRange[3];
    }
  }
  return {
    records,
    count,
    candidateCount: sourceIndices.length,
    validCount,
    comparisonReference: FIXED_RECORD_COMPARISON_REFERENCE,
    sourceIndices: sourceIndices.slice(0, count),
    timing: {
      cpuMinimalRecordBuildMs: nowMs() - startMs
    }
  };
}

function flattenRows4(rows) {
  return new Float32Array([
    rows[0][0], rows[0][1], rows[0][2], rows[0][3],
    rows[1][0], rows[1][1], rows[1][2], rows[1][3],
    rows[2][0], rows[2][1], rows[2][2], rows[2][3]
  ]);
}

function getCudaAlignedUniforms(screenSpaceCamera, renderW, renderH) {
  const s = screenSpaceCamera?.screenSpaceTransformOverride;
  if (!s || s.mode !== 'cuda-aligned' || !Array.isArray(s.viewMatrix)) return null;
  const tanFovY = Math.tan((Number(screenSpaceCamera.fov ?? s.fov ?? 60) * Math.PI / 180) * 0.5);
  const aspect = Number(screenSpaceCamera.aspect ?? s.aspect ?? (renderW / Math.max(1, renderH)));
  const tanFovX = tanFovY * aspect;
  const covarianceTanFovX = Number.isFinite(s.covarianceTanFovX) ? Number(s.covarianceTanFovX) : tanFovX;
  const covarianceTanFovY = Number.isFinite(s.covarianceTanFovY) ? Number(s.covarianceTanFovY) : tanFovY;
  return {
    viewRows: s.viewMatrix,
    viewRotRows: [
      [s.viewMatrix[0][0], s.viewMatrix[0][1], s.viewMatrix[0][2]],
      [s.viewMatrix[1][0], s.viewMatrix[1][1], s.viewMatrix[1][2]],
      [s.viewMatrix[2][0], s.viewMatrix[2][1], s.viewMatrix[2][2]]
    ],
    fx: Number.isFinite(s.intrinsics?.fx) ? Number(s.intrinsics.fx) : renderW / (2 * tanFovX),
    fy: Number.isFinite(s.intrinsics?.fy) ? Number(s.intrinsics.fy) : renderH / (2 * tanFovY),
    cx: Number.isFinite(s.intrinsics?.cx) ? Number(s.intrinsics.cx) - 0.5 : (renderW - 1) * 0.5,
    cy: Number.isFinite(s.intrinsics?.cy) ? Number(s.intrinsics.cy) - 0.5 : (renderH - 1) * 0.5,
    covFocalX: -renderW / (2 * covarianceTanFovX),
    covFocalY: -renderH / (2 * covarianceTanFovY),
    covarianceTanFovX,
    covarianceTanFovY,
    pixelXSign: [-1, 1].includes(s.pixelXSign) ? Number(s.pixelXSign) : 1
  };
}

function bindTexture(gl, unit, texture, location, name) {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  if (location) gl.uniform1i(location, unit);
  const error = gl.getError();
  return {
    name,
    unit,
    texturePresent: !!texture,
    uniformLocationPresent: !!location,
    error: error === gl.NO_ERROR ? null : error
  };
}

function buildExpectedXyzOpacity(raw, srcIndex) {
  return [
    readRawComponents(raw.xyz, srcIndex, raw.xyzDim, 3)[0],
    readRawComponents(raw.xyz, srcIndex, raw.xyzDim, 3)[1],
    readRawComponents(raw.xyz, srcIndex, raw.xyzDim, 3)[2],
    readRawComponents(raw.opacity, srcIndex, raw.opacityDim, 1)[0]
  ];
}

function runMinimalFetchProbe({
  gl,
  state,
  rawTextureResult,
  cpuRecords,
  raw,
  rawTexCoords,
  textureWidth,
  textureHeight,
  epsilon = DEFAULT_EPSILON
}) {
  const probe = state.minimalFetchProbe;
  const sampleCount = Math.min(DEFAULT_DEBUG_SAMPLE_COUNT, cpuRecords.count);
  if (!probe || sampleCount <= 0) {
    return {
      status: 'skipped',
      reason: 'minimal-fetch-probe-no-samples',
      sampleCount
    };
  }
  const sampleCoords = new Float32Array(sampleCount * 2);
  const samples = [];
  for (let row = 0; row < sampleCount; row++) {
    sampleCoords[row * 2 + 0] = rawTexCoords[row * 2 + 0];
    sampleCoords[row * 2 + 1] = rawTexCoords[row * 2 + 1];
  }
  const output = new Float32Array(sampleCount * MINIMAL_FETCH_RECORD_FLOATS);
  const outputBytes = output.byteLength;
  const cleanup = () => {
    gl.disable(gl.RASTERIZER_DISCARD);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.activeTexture(gl.TEXTURE0 + 0);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindVertexArray(null);
    gl.useProgram(null);
  };
  const startMs = nowMs();
  gl.bindVertexArray(probe.vao);
  gl.useProgram(probe.program);
  gl.bindBuffer(gl.ARRAY_BUFFER, probe.rawTexCoordBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, sampleCoords, gl.STREAM_DRAW);
  gl.enableVertexAttribArray(probe.aRawTexCoord);
  gl.vertexAttribPointer(probe.aRawTexCoord, 2, gl.FLOAT, false, 0, 0);
  const samplerBinding = bindTexture(
    gl,
    0,
    rawTextureResult.textures.xyzOpacity,
    probe.uniforms.uXyzOpacityTex,
    'minimalFetch.uXyzOpacityTex'
  );
  gl.uniform1f(probe.uniforms.uTextureWidth, textureWidth);
  gl.uniform1f(probe.uniforms.uTextureHeight, textureHeight);
  gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, probe.outputBuffer);
  gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER, outputBytes, gl.DYNAMIC_READ);
  gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);
  gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, probe.transformFeedback);
  gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, probe.outputBuffer);
  const bindError = gl.getError();
  if (bindError !== gl.NO_ERROR) {
    cleanup();
    return {
      status: 'error',
      reason: `minimal-fetch-probe-bind-${bindError}`,
      samplerBinding,
      sampleCount
    };
  }
  gl.enable(gl.RASTERIZER_DISCARD);
  gl.beginTransformFeedback(gl.POINTS);
  gl.drawArrays(gl.POINTS, 0, sampleCount);
  gl.endTransformFeedback();
  gl.disable(gl.RASTERIZER_DISCARD);
  const drawError = gl.getError();
  if (drawError !== gl.NO_ERROR) {
    cleanup();
    return {
      status: 'error',
      reason: `minimal-fetch-probe-draw-${drawError}`,
      samplerBinding,
      sampleCount
    };
  }
  gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
  gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
  gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, probe.outputBuffer);
  gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER, 0, output);
  gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);
  const readbackError = gl.getError();
  if (readbackError !== gl.NO_ERROR) {
    cleanup();
    return {
      status: 'error',
      reason: `minimal-fetch-probe-readback-${readbackError}`,
      samplerBinding,
      sampleCount
    };
  }
  let anyMismatch = false;
  let firstMismatch = null;
  for (let row = 0; row < sampleCount; row++) {
    const srcIndex = cpuRecords.sourceIndices[row] >>> 0;
    const base = row * MINIMAL_FETCH_RECORD_FLOATS;
    const gpuFetch = Array.from(output.subarray(base, base + 4)).map(finiteOrNull);
    const gpuCoord = Array.from(output.subarray(base + 4, base + 8)).map(finiteOrNull);
    const expected = buildExpectedXyzOpacity(raw, srcIndex).map(finiteOrNull);
    let maxAbsError = 0;
    for (let i = 0; i < 4; i++) {
      const abs = Math.abs(Number(expected[i]) - Number(gpuFetch[i]));
      maxAbsError = Math.max(maxAbsError, abs);
      if (!firstMismatch && !(abs <= epsilon)) {
        firstMismatch = {
          row,
          srcIndex,
          field: 'xyzOpacity',
          component: i,
          cpu: expected[i],
          gpu: gpuFetch[i],
          absError: finiteOrNull(abs)
        };
      }
    }
    anyMismatch = anyMismatch || maxAbsError > epsilon;
    samples.push({
      row,
      srcIndex,
      rawTexCoord: [sampleCoords[row * 2 + 0], sampleCoords[row * 2 + 1]],
      expectedXyzOpacity: expected,
      gpuXyzOpacity: gpuFetch,
      gpuFetchCoordAndSize: gpuCoord,
      maxAbsError: finiteOrNull(maxAbsError)
    });
  }
  cleanup();
  return {
    schemaVersion: 'step116-minimal-texel-fetch-probe-v1',
    status: 'ok',
    reason: 'ok',
    computeMode: 'minimal-xyz-opacity-texture-tf-fetch',
    sampleCount,
    anyMismatch,
    firstMismatch,
    samplerBinding,
    textureWidth,
    textureHeight,
    outputBytes,
    probeMs: nowMs() - startMs,
    samples
  };
}

function runTransformFeedback({
  gl,
  rawTextureResult,
  cpuRecords,
  raw,
  screenSpaceCamera,
  canvasWidth,
  canvasHeight,
  tileGrid,
  buildConfig,
  temporalSigmaThreshold
}) {
  const renderScale = Number.isFinite(buildConfig.renderScale) ? buildConfig.renderScale : 1;
  const renderW = Math.max(1, Math.round(canvasWidth * renderScale));
  const renderH = Math.max(1, Math.round(canvasHeight * renderScale));
  const sx = canvasWidth / renderW;
  const sy = canvasHeight / renderH;
  const cudaUniforms = getCudaAlignedUniforms(screenSpaceCamera, renderW, renderH);
  if (!cudaUniforms) throw new Error('cuda-aligned-screen-space-camera-required');

  const state = getRawVisibleRecordState(gl);
  const srcIndices = new Float32Array(cpuRecords.sourceIndices.length);
  const rawTexCoords = new Float32Array(cpuRecords.sourceIndices.length * 2);
  const textureWidth = Math.max(1, rawTextureResult.summary.width | 0);
  const textureHeight = Math.max(1, rawTextureResult.summary.height | 0);
  for (let i = 0; i < srcIndices.length; i++) {
    const srcIndex = cpuRecords.sourceIndices[i] >>> 0;
    srcIndices[i] = srcIndex;
    rawTexCoords[i * 2 + 0] = srcIndex % textureWidth;
    rawTexCoords[i * 2 + 1] = Math.floor(srcIndex / textureWidth);
  }
  const outputBytes = Math.max(
    RAW_RECORD_FLOATS * Float32Array.BYTES_PER_ELEMENT,
    cpuRecords.count * RAW_RECORD_FLOATS * Float32Array.BYTES_PER_ELEMENT
  );
  const cleanup = () => {
    gl.disable(gl.RASTERIZER_DISCARD);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    for (let i = 0; i < 5; i++) {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, null);
    }
    gl.bindVertexArray(null);
    gl.useProgram(null);
  };
  const minimalFetchProbe = runMinimalFetchProbe({
    gl,
    state,
    rawTextureResult,
    cpuRecords,
    raw,
    rawTexCoords,
    textureWidth,
    textureHeight
  });

  const setupStartMs = nowMs();
  gl.bindVertexArray(state.vao);
  gl.useProgram(state.program);
  gl.bindBuffer(gl.ARRAY_BUFFER, state.srcIndexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, srcIndices, gl.STREAM_DRAW);
  gl.enableVertexAttribArray(state.aSrcIndex);
  gl.vertexAttribPointer(state.aSrcIndex, 1, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, state.rawTexCoordBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, rawTexCoords, gl.STREAM_DRAW);
  gl.enableVertexAttribArray(state.aRawTexCoord);
  gl.vertexAttribPointer(state.aRawTexCoord, 2, gl.FLOAT, false, 0, 0);
  const samplerBindingSummary = [
    bindTexture(gl, 0, rawTextureResult.textures.xyzOpacity, state.uniforms.uXyzOpacityTex, 'uXyzOpacityTex'),
    bindTexture(gl, 1, rawTextureResult.textures.scaleTime, state.uniforms.uScaleTimeTex, 'uScaleTimeTex'),
    bindTexture(gl, 2, rawTextureResult.textures.timeScale, state.uniforms.uTimeScaleTex, 'uTimeScaleTex'),
    bindTexture(gl, 3, rawTextureResult.textures.rotation, state.uniforms.uRotationTex, 'uRotationTex'),
    bindTexture(gl, 4, rawTextureResult.textures.rotationR, state.uniforms.uRotationRTex, 'uRotationRTex')
  ];
  const samplerBindFailure = samplerBindingSummary.find((entry) => (
    entry.error != null || !entry.texturePresent || !entry.uniformLocationPresent
  ));
  if (samplerBindFailure) {
    cleanup();
    throw new Error(`raw-visible-record-sampler-bind:${samplerBindFailure.name}`);
  }
  gl.uniform1f(state.uniforms.uTextureWidth, rawTextureResult.summary.width);
  gl.uniform1f(state.uniforms.uTextureHeight, rawTextureResult.summary.height);
  gl.uniform1f(state.uniforms.uRawCount, raw.N);
  gl.uniform1f(state.uniforms.uTimestamp, buildConfig.timestamp);
  gl.uniform1f(state.uniforms.uScalingModifier, buildConfig.scalingModifier);
  gl.uniform1f(state.uniforms.uSigmaScale, buildConfig.sigmaScale);
  gl.uniform1f(state.uniforms.uPrefilterVar, buildConfig.prefilterVar);
  gl.uniform1f(state.uniforms.uTemporalSigmaThreshold, temporalSigmaThreshold);
  gl.uniform1f(state.uniforms.uRenderW, renderW);
  gl.uniform1f(state.uniforms.uRenderH, renderH);
  gl.uniform1f(state.uniforms.uCanvasW, canvasWidth);
  gl.uniform1f(state.uniforms.uCanvasH, canvasHeight);
  gl.uniform1f(state.uniforms.uSx, sx);
  gl.uniform1f(state.uniforms.uSy, sy);
  gl.uniform1f(state.uniforms.uFx, cudaUniforms.fx);
  gl.uniform1f(state.uniforms.uFy, cudaUniforms.fy);
  gl.uniform1f(state.uniforms.uCx, cudaUniforms.cx);
  gl.uniform1f(state.uniforms.uCy, cudaUniforms.cy);
  gl.uniform1f(state.uniforms.uCovFocalX, cudaUniforms.covFocalX);
  gl.uniform1f(state.uniforms.uCovFocalY, cudaUniforms.covFocalY);
  gl.uniform1f(state.uniforms.uCovTanFovX, cudaUniforms.covarianceTanFovX);
  gl.uniform1f(state.uniforms.uCovTanFovY, cudaUniforms.covarianceTanFovY);
  gl.uniform1f(state.uniforms.uPixelXSign, cudaUniforms.pixelXSign);
  const rows = flattenRows4(cudaUniforms.viewRows);
  gl.uniform4fv(state.uniforms.uViewRow0, rows.subarray(0, 4));
  gl.uniform4fv(state.uniforms.uViewRow1, rows.subarray(4, 8));
  gl.uniform4fv(state.uniforms.uViewRow2, rows.subarray(8, 12));
  gl.uniform3fv(state.uniforms.uViewRotRow0, cudaUniforms.viewRotRows[0]);
  gl.uniform3fv(state.uniforms.uViewRotRow1, cudaUniforms.viewRotRows[1]);
  gl.uniform3fv(state.uniforms.uViewRotRow2, cudaUniforms.viewRotRows[2]);
  const tileRangeEnabled = tileGrid &&
    Number.isFinite(tileGrid.tileCols) &&
    Number.isFinite(tileGrid.tileRows) &&
    Number.isFinite(tileGrid.tileSize);
  gl.uniform1f(state.uniforms.uTileRangeEnabled, tileRangeEnabled ? 1 : 0);
  gl.uniform1f(state.uniforms.uTileCols, tileRangeEnabled ? tileGrid.tileCols : 1);
  gl.uniform1f(state.uniforms.uTileRows, tileRangeEnabled ? tileGrid.tileRows : 1);
  gl.uniform1f(state.uniforms.uTileSize, tileRangeEnabled ? tileGrid.tileSize : 1);
  const setupMs = nowMs() - setupStartMs;

  const tfSetupStartMs = nowMs();
  gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, state.outputBuffer);
  gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER, outputBytes, gl.DYNAMIC_READ);
  gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);
  gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, state.transformFeedback);
  gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, state.outputBuffer);
  const bindError = gl.getError();
  if (bindError !== gl.NO_ERROR) {
    cleanup();
    throw new Error(`raw-visible-record-transform-feedback-bind:${bindError}`);
  }
  const transformFeedbackSetupMs = nowMs() - tfSetupStartMs;

  const drawStartMs = nowMs();
  gl.enable(gl.RASTERIZER_DISCARD);
  gl.beginTransformFeedback(gl.POINTS);
  gl.drawArrays(gl.POINTS, 0, cpuRecords.count);
  gl.endTransformFeedback();
  gl.disable(gl.RASTERIZER_DISCARD);
  gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
  gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
  const drawError = gl.getError();
  if (drawError !== gl.NO_ERROR) {
    cleanup();
    throw new Error(`raw-visible-record-transform-feedback-draw:${drawError}`);
  }
  const transformFeedbackDrawMs = nowMs() - drawStartMs;

  const records = new Float32Array(cpuRecords.count * RAW_RECORD_FLOATS);
  const readbackStartMs = nowMs();
  gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, state.outputBuffer);
  gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER, 0, records);
  gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);
  const readbackError = gl.getError();
  if (readbackError !== gl.NO_ERROR) {
    cleanup();
    throw new Error(`raw-visible-record-transform-feedback-readback:${readbackError}`);
  }
  const readbackMs = nowMs() - readbackStartMs;
  cleanup();
  return {
    records,
    minimalFetchProbe,
    samplerBindingSummary,
    fetchInputSummary: {
      textureWidth,
      textureHeight,
      rawCount: raw.N,
      recordCount: cpuRecords.count,
      tileRangeEnabled: !!(tileGrid &&
        Number.isFinite(tileGrid.tileCols) &&
        Number.isFinite(tileGrid.tileRows) &&
        Number.isFinite(tileGrid.tileSize)),
      tileGrid: tileGrid
        ? {
            tileCols: tileGrid.tileCols ?? null,
            tileRows: tileGrid.tileRows ?? null,
            tileSize: tileGrid.tileSize ?? null
          }
        : null,
      firstSourceIndices: Array.from(cpuRecords.sourceIndices.slice(0, Math.min(4, cpuRecords.sourceIndices.length))).map((value) => value >>> 0),
      firstRawTexCoords: Array.from(rawTexCoords.slice(0, Math.min(8, rawTexCoords.length)))
    },
    timing: {
      uploadAndSetupMs: setupMs,
      transformFeedbackSetupMs,
      transformFeedbackDrawMs,
      readbackMs,
      outputBytes
    }
  };
}

function compareRecords(referenceRecords, candidateRecords, count, options = {}) {
  const epsilon = toFiniteNumber(options.epsilon, DEFAULT_EPSILON);
  const maxMismatches = toFiniteInteger(options.maxMismatches, 32);
  let fieldMismatchCount = 0;
  let maxAbsError = 0;
  const firstMismatches = [];
  for (let row = 0; row < count; row++) {
    for (const [field, offset, length] of FIELD_LAYOUT) {
      for (let i = 0; i < length; i++) {
        const ref = referenceRecords[row * RAW_RECORD_FLOATS + offset + i];
        const got = candidateRecords[row * RAW_RECORD_FLOATS + offset + i];
        const abs = Math.abs(Number(ref) - Number(got));
        maxAbsError = Math.max(maxAbsError, abs);
        if (!(abs <= epsilon)) {
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

function buildPackedLikeRecords(sourceRecords, count) {
  const packed = new Float32Array(Math.max(0, count | 0) * PACKED_LIKE_FLOATS_PER_ITEM);
  for (let row = 0; row < count; row++) {
    const srcBase = row * RAW_RECORD_FLOATS;
    const dstBase = row * PACKED_LIKE_FLOATS_PER_ITEM;
    packed[dstBase + 0] = sourceRecords[srcBase + 2]; // centerPx.x
    packed[dstBase + 1] = sourceRecords[srcBase + 3]; // centerPx.y
    packed[dstBase + 2] = sourceRecords[srcBase + 9]; // radiusPx
    packed[dstBase + 3] = sourceRecords[srcBase + 4]; // depth
    packed[dstBase + 4] = 0; // colorAlpha.r deferred
    packed[dstBase + 5] = 0; // colorAlpha.g deferred
    packed[dstBase + 6] = 0; // colorAlpha.b deferred
    packed[dstBase + 7] = sourceRecords[srcBase + 13]; // colorAlpha.a
    packed[dstBase + 8] = sourceRecords[srcBase + 10]; // conic.x
    packed[dstBase + 9] = sourceRecords[srcBase + 11]; // conic.y
    packed[dstBase + 10] = sourceRecords[srcBase + 12]; // conic.z
    packed[dstBase + 11] = 0; // reserved
    packed[dstBase + 12] = sourceRecords[srcBase + 5]; // misc/aabb.x
    packed[dstBase + 13] = sourceRecords[srcBase + 6]; // misc/aabb.y
    packed[dstBase + 14] = sourceRecords[srcBase + 7]; // misc/aabb.z
    packed[dstBase + 15] = sourceRecords[srcBase + 8]; // misc/aabb.w
  }
  return packed;
}

function comparePackedLikeRecords(referenceRecords, candidateRecords, count, options = {}) {
  const epsilon = toFiniteNumber(options.epsilon, DEFAULT_EPSILON);
  const maxMismatches = toFiniteInteger(options.maxMismatches, 32);
  let fieldMismatchCount = 0;
  let maxAbsError = 0;
  const firstMismatches = [];
  for (let row = 0; row < count; row++) {
    for (const [field, offset, length] of PACKED_LIKE_FIELD_LAYOUT) {
      for (let i = 0; i < length; i++) {
        const ref = referenceRecords[row * PACKED_LIKE_FLOATS_PER_ITEM + offset + i];
        const got = candidateRecords[row * PACKED_LIKE_FLOATS_PER_ITEM + offset + i];
        const abs = Math.abs(Number(ref) - Number(got));
        maxAbsError = Math.max(maxAbsError, abs);
        if (!(abs <= epsilon)) {
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
    schemaVersion: 'step120a-packed-like-fixed-record-comparison-v1',
    packedLayoutVersion: 2,
    floatsPerItem: PACKED_LIKE_FLOATS_PER_ITEM,
    orderMode: 'candidate-order-unsorted',
    implementedFields: PACKED_LIKE_IMPLEMENTED_FIELDS,
    deferredFields: PACKED_LIKE_DEFERRED_FIELDS,
    anyMismatch: fieldMismatchCount > 0,
    fieldMismatchCount,
    maxAbsError,
    firstMismatches
  };
}

function classifyRecordComparison(recordComparison) {
  if (!recordComparison?.anyMismatch) return 'none';
  const mismatches = Array.isArray(recordComparison.firstMismatches)
    ? recordComparison.firstMismatches
    : [];
  const fieldMismatchCount = toFiniteInteger(recordComparison.fieldMismatchCount, -1);
  const allKnownAabbBoundaryDiff = fieldMismatchCount > 0 &&
    mismatches.length > 0 &&
    mismatches.every((mismatch) => (
      mismatch?.field === 'aabb' &&
      Math.abs(Number(mismatch?.absError)) === 1
    ));
  const allKnownTileRangeBoundaryDiff = fieldMismatchCount > 0 &&
    mismatches.length > 0 &&
    mismatches.every((mismatch) => (
      mismatch?.field === 'tileRange' &&
      Math.abs(Number(mismatch?.absError)) === 1
    ));
  const allKnownAabbOrTileBoundaryDiff = fieldMismatchCount > 0 &&
    mismatches.length > 0 &&
    mismatches.every((mismatch) => (
      ['aabb', 'tileRange'].includes(mismatch?.field) &&
      Math.abs(Number(mismatch?.absError)) === 1
    ));
  return allKnownAabbBoundaryDiff
    ? 'known-aabb-rounding-boundary-diff'
    : (allKnownTileRangeBoundaryDiff
      ? 'known-tile-range-rounding-boundary-diff'
      : (allKnownAabbOrTileBoundaryDiff ? 'known-aabb-tile-range-rounding-boundary-diff' : 'raw-visible-record-field-mismatch'));
}

function classifyPackedLikeComparison(packedLikeComparison) {
  if (!packedLikeComparison?.anyMismatch) return 'none';
  const mismatches = Array.isArray(packedLikeComparison.firstMismatches)
    ? packedLikeComparison.firstMismatches
    : [];
  const fieldMismatchCount = toFiniteInteger(packedLikeComparison.fieldMismatchCount, -1);
  const allKnownMiscAabbBoundaryDiff = fieldMismatchCount > 0 &&
    mismatches.length > 0 &&
    mismatches.every((mismatch) => (
      mismatch?.field === 'misc.aabb' &&
      Math.abs(Number(mismatch?.absError)) === 1
    ));
  return allKnownMiscAabbBoundaryDiff
    ? 'known-packed-like-aabb-rounding-boundary-diff'
    : 'packed-like-field-mismatch';
}

function buildDisplayConnectionReadinessSummary({
  status = 'ok',
  reason = 'ok',
  recordMode = 'richer',
  recordComparison = null,
  packedLikeComparison = null,
  mismatchClassification = null,
  packedLikeMismatchClassification = null,
  displayCandidateSource = 'cpu-reference',
  gpuCandidateUsedForDisplay = false,
  limitedDrawUsedForCandidateSource = false,
  fallbackReason = 'none',
  candidateCount = null,
  recordCount = null,
  validRecordCount = null
} = {}) {
  const isPackedLike = recordMode === 'packed-like';
  const hasRecords = (toFiniteInteger(candidateCount, 0) > 0) &&
    (toFiniteInteger(recordCount, 0) > 0) &&
    (toFiniteInteger(validRecordCount, 0) > 0);
  const rawMismatchKnown = !recordComparison?.anyMismatch ||
    mismatchClassification === 'known-aabb-rounding-boundary-diff' ||
    mismatchClassification === 'known-aabb-tile-range-rounding-boundary-diff';
  const packedMismatchKnown = !isPackedLike ||
    !packedLikeComparison?.anyMismatch ||
    packedLikeMismatchClassification === 'known-packed-like-aabb-rounding-boundary-diff';
  const fixedRecordReady = status === 'ok' && isPackedLike && hasRecords && rawMismatchKnown && packedMismatchKnown;

  const satisfied = fixedRecordReady ? DISPLAY_CONNECTION_SATISFIED_ITEMS : [];
  const unresolved = fixedRecordReady ? DISPLAY_CONNECTION_UNRESOLVED_ITEMS : [
    'packed-like-fixed-record-not-validated'
  ];
  const blocked = fixedRecordReady ? DISPLAY_CONNECTION_BLOCKED_ITEMS : [
    'raw-visible-record-dry-run-not-ready'
  ];

  return {
    schemaVersion: 'step122a-display-connection-readiness-v1',
    status: fixedRecordReady ? 'not-ready' : 'blocked',
    reason: fixedRecordReady
      ? 'packed-like-fixed-record-ready-but-display-connection-blocked-by-order-color-and-tile-list-contracts'
      : reason,
    displayConnectionAllowed: false,
    displayConnectionClassification: fixedRecordReady ? 'webgl2-fixed-record-ready-display-not-ready' : 'dry-run-not-ready',
    recordMode,
    packedLikeFixedRecordReady: fixedRecordReady,
    packedLikeComparisonUsable: isPackedLike && !!packedLikeComparison,
    currentDisplayCandidateSource: displayCandidateSource,
    gpuCandidateUsedForDisplay: !!gpuCandidateUsedForDisplay,
    limitedDrawUsedForCandidateSource: !!limitedDrawUsedForCandidateSource,
    fallbackRequired: true,
    fallbackMode: 'cpu-reference-display-path',
    fallbackReason,
    satisfied,
    unresolved,
    blocked,
    webgl2LimitCandidates: DISPLAY_CONNECTION_WEBGL2_LIMIT_CANDIDATES,
    webgpuMigrationSignals: DISPLAY_CONNECTION_WEBGPU_MIGRATION_SIGNALS,
    notes: [
      'Step122A does not connect GPU packed-like records to display.',
      'Candidate-order fixed records are not equivalent to sorted visible order.',
      'CPU tile-list remains valid only while display continues using the existing CPU packed/visible path.',
      'AABB 1px boundary diffs are classified and should not be used as a visual correction.'
    ]
  };
}

function resolveRecordMode(mode) {
  return mode === 'packed-like' ? 'packed-like' : (mode === 'minimal' ? 'minimal' : 'richer');
}

function computeModeForRecordMode(recordMode) {
  return recordMode === 'packed-like'
    ? 'raw-attribute-texture-tf-packed-like-fixed-record'
    : (recordMode === 'minimal'
      ? 'raw-attribute-texture-tf-minimal-visible-record'
      : 'raw-attribute-texture-tf-richer-visible-record');
}

function fallback(reason, extra = {}) {
  const recordMode = resolveRecordMode(extra.recordMode);
  const displayConnectionReadiness = buildDisplayConnectionReadinessSummary({
    status: 'fallback',
    reason,
    recordMode,
    recordComparison: null,
    packedLikeComparison: extra.packedLikeComparison ?? null,
    mismatchClassification: extra.mismatchClassification ?? 'raw-visible-record-unavailable',
    packedLikeMismatchClassification: extra.packedLikeMismatchClassification ?? null,
    displayCandidateSource: extra.displayCandidateSource ?? 'cpu-reference',
    gpuCandidateUsedForDisplay: !!extra.gpuCandidateUsedForDisplay,
    limitedDrawUsedForCandidateSource: !!extra.limitedDrawUsedForCandidateSource,
    fallbackReason: reason,
    candidateCount: extra.candidateCount,
    recordCount: extra.recordCount,
    validRecordCount: extra.validRecordCount
  });
  return {
    schemaVersion: 'step116-raw-visible-record-dry-run-v1',
    status: 'fallback',
    reason,
    computeMode: computeModeForRecordMode(recordMode),
    recordMode,
    rawVisibleRecordMode: recordMode,
    implementedFields: IMPLEMENTED_FIELDS,
    deferredFields: DEFERRED_FIELDS,
    packedLikeImplementedFields: recordMode === 'packed-like' ? PACKED_LIKE_IMPLEMENTED_FIELDS : [],
    packedLikeDeferredFields: recordMode === 'packed-like' ? PACKED_LIKE_DEFERRED_FIELDS : [],
    candidateCount: Number.isFinite(extra.candidateCount) ? extra.candidateCount : null,
    recordCount: Number.isFinite(extra.recordCount) ? extra.recordCount : null,
    validRecordCount: Number.isFinite(extra.validRecordCount) ? extra.validRecordCount : null,
    recordComparison: {
      anyMismatch: true,
      fieldMismatchCount: null,
      maxAbsError: null,
      firstMismatches: Array.isArray(extra.firstMismatches) ? extra.firstMismatches : []
    },
    mismatchClassification: extra.mismatchClassification ?? 'raw-visible-record-unavailable',
    packedLikeComparison: extra.packedLikeComparison ?? null,
    packedLikeMismatchClassification: extra.packedLikeMismatchClassification ?? null,
    displayConnectionReadiness,
    anyMismatch: true,
    displayCandidateSource: extra.displayCandidateSource ?? 'cpu-reference',
    gpuCandidateUsedForDisplay: !!extra.gpuCandidateUsedForDisplay,
    limitedDrawUsedForCandidateSource: !!extra.limitedDrawUsedForCandidateSource,
    fallbackReason: reason,
    metadata: extra.metadata ?? null
  };
}

export function runGpuRawVisibleRecordDryRun({
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
  maxRecords = DEFAULT_MAX_RECORDS,
  epsilon = DEFAULT_EPSILON,
  maxMismatches = 32,
  readbackMode = 'sync-debug',
  recordMode = 'richer',
  displayCandidateSource = 'cpu-reference',
  gpuCandidateUsedForDisplay = false,
  limitedDrawUsedForCandidateSource = false,
  metadata = null
} = {}) {
  const totalStartMs = nowMs();
  const resolvedRecordMode = resolveRecordMode(recordMode ?? metadata?.rawVisibleRecordMode);
  const fallbackExtra = {
    displayCandidateSource,
    gpuCandidateUsedForDisplay,
    limitedDrawUsedForCandidateSource,
    recordMode: resolvedRecordMode,
    metadata
  };
  if (!gl) return fallback('webgl-unavailable', fallbackExtra);
  if (!raw || !candidateInfo || !camera || !buildConfig) {
    return fallback('raw-visible-record-input-unavailable', fallbackExtra);
  }
  if (readbackMode === 'none') return fallback('debug-readback-disabled', fallbackExtra);
  if (!buildConfig.useNativeRot4d || !raw.rot4d) {
    return fallback('native-rot4d-required-for-step116-minimal-shader', fallbackExtra);
  }
  try {
    const rawTextureResult = ensureRawAttributeTextures(gl, raw);
    if (rawTextureResult.status !== 'ok') {
      return fallback(rawTextureResult.reason ?? 'raw-texture-unavailable', fallbackExtra);
    }
    const cpuRecords = buildCpuMinimalRecords({
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
    const countSummary = {
      ...fallbackExtra,
      candidateCount: cpuRecords.candidateCount,
      recordCount: cpuRecords.count,
      validRecordCount: cpuRecords.validCount
    };
    if (cpuRecords.candidateCount <= 0) {
      return fallback('raw-visible-record-candidate-input-empty', {
        ...countSummary,
        mismatchClassification: 'raw-visible-record-empty-candidate-input',
        firstMismatches: [{
          reason: 'candidateInfo.candidateIndices was empty or unavailable'
        }]
      });
    }
    if (cpuRecords.count <= 0) {
      return fallback('raw-visible-record-count-zero', {
        ...countSummary,
        mismatchClassification: 'raw-visible-record-count-zero',
        firstMismatches: [{
          reason: 'maxRecords limited the non-empty candidate input to zero records',
          candidateCount: cpuRecords.candidateCount,
          maxRecords: toFiniteInteger(maxRecords, DEFAULT_MAX_RECORDS)
        }]
      });
    }
    if (cpuRecords.validCount <= 0) {
      return fallback('raw-visible-record-valid-count-zero', {
        ...countSummary,
        mismatchClassification: 'raw-visible-record-valid-count-zero',
        firstMismatches: [{
          reason: 'CPU reference minimal visible build produced no valid records for the candidate input',
          candidateCount: cpuRecords.candidateCount,
          recordCount: cpuRecords.count
        }]
      });
    }
    const tfResult = runTransformFeedback({
      gl,
      rawTextureResult,
      cpuRecords,
      raw,
      screenSpaceCamera,
      canvasWidth,
      canvasHeight,
      tileGrid,
      buildConfig,
      temporalSigmaThreshold
    });
    const compareStartMs = nowMs();
    const recordComparison = compareRecords(cpuRecords.records, tfResult.records, cpuRecords.count, {
      epsilon,
      maxMismatches
    });
    const compareMs = nowMs() - compareStartMs;
    let packedLikeComparison = null;
    let packedLikeMismatchClassification = null;
    let packedLikeCompareMs = 0;
    if (resolvedRecordMode === 'packed-like') {
      const packedLikeCompareStartMs = nowMs();
      const packedLikeReference = buildPackedLikeRecords(cpuRecords.records, cpuRecords.count);
      const packedLikeCandidate = buildPackedLikeRecords(tfResult.records, cpuRecords.count);
      packedLikeComparison = comparePackedLikeRecords(packedLikeReference, packedLikeCandidate, cpuRecords.count, {
        epsilon,
        maxMismatches
      });
      packedLikeMismatchClassification = classifyPackedLikeComparison(packedLikeComparison);
      packedLikeCompareMs = nowMs() - packedLikeCompareStartMs;
    }
    const debugSamples = buildRawVisibleDebugSamples({
      raw,
      cpuRecords,
      gpuRecords: tfResult.records,
      screenSpaceCamera,
      buildConfig,
      textureWidth: rawTextureResult.summary.width,
      maxSamples: DEFAULT_DEBUG_SAMPLE_COUNT
    });
    const firstDebugDivergence = classifyFirstDebugDivergence(debugSamples, epsilon);
    const mismatchClassification = classifyRecordComparison(recordComparison);
    const displayConnectionReadiness = buildDisplayConnectionReadinessSummary({
      status: 'ok',
      reason: 'ok',
      recordMode: resolvedRecordMode,
      recordComparison,
      packedLikeComparison,
      mismatchClassification,
      packedLikeMismatchClassification,
      displayCandidateSource,
      gpuCandidateUsedForDisplay,
      limitedDrawUsedForCandidateSource,
      fallbackReason: 'none',
      candidateCount: cpuRecords.candidateCount,
      recordCount: cpuRecords.count,
      validRecordCount: cpuRecords.validCount
    });
    return {
      schemaVersion: 'step116-raw-visible-record-dry-run-v1',
      status: 'ok',
      reason: 'ok',
      computeMode: computeModeForRecordMode(resolvedRecordMode),
      recordMode: resolvedRecordMode,
      rawVisibleRecordMode: resolvedRecordMode,
      implementedFields: IMPLEMENTED_FIELDS,
      deferredFields: DEFERRED_FIELDS,
      packedLikeImplementedFields: resolvedRecordMode === 'packed-like' ? PACKED_LIKE_IMPLEMENTED_FIELDS : [],
      packedLikeDeferredFields: resolvedRecordMode === 'packed-like' ? PACKED_LIKE_DEFERRED_FIELDS : [],
      candidateCount: cpuRecords.candidateCount,
      recordCount: cpuRecords.count,
      validRecordCount: cpuRecords.validCount,
      recordComparison,
      comparisonReference: cpuRecords.comparisonReference,
      packedLikeComparisonReference: resolvedRecordMode === 'packed-like' ? PACKED_LIKE_COMPARISON_REFERENCE : null,
      packedLikeComparison,
      mismatchClassification,
      packedLikeMismatchClassification,
      displayConnectionReadiness,
      anyMismatch: !!recordComparison.anyMismatch,
      rawTextureSummary: rawTextureResult.summary,
      minimalFetchProbe: tfResult.minimalFetchProbe,
      samplerBindingSummary: tfResult.samplerBindingSummary,
      fetchInputSummary: tfResult.fetchInputSummary,
      debugSamples,
      firstDebugDivergence,
      displayCandidateSource,
      gpuCandidateUsedForDisplay: !!gpuCandidateUsedForDisplay,
      limitedDrawUsedForCandidateSource: !!limitedDrawUsedForCandidateSource,
      fallbackReason: 'none',
      timing: {
        ...cpuRecords.timing,
        rawTextureUploadMs: rawTextureResult.summary.rawTextureUploadMs ?? 0,
        rawTextureEnsureMs: rawTextureResult.summary.rawTextureEnsureMs ?? 0,
        minimalFetchProbeMs: tfResult.minimalFetchProbe?.probeMs ?? 0,
        ...tfResult.timing,
        compareMs,
        packedLikeCompareMs,
        totalMs: nowMs() - totalStartMs
      },
      metadata
    };
  } catch (error) {
    return fallback(error?.message ?? 'raw-visible-record-error', fallbackExtra);
  }
}
