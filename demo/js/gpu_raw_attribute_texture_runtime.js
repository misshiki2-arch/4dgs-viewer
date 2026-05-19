const stateByGl = new WeakMap();

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function getRawTextureState(gl) {
  let state = stateByGl.get(gl);
  if (!state) {
    state = {
      raw: null,
      width: 0,
      height: 0,
      textures: new Map(),
      summary: null
    };
    stateByGl.set(gl, state);
  }
  return state;
}

function makeTexture(gl) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_BASE_LEVEL, 0);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL, 0);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return texture;
}

function uploadTexture(gl, state, name, data) {
  let texture = state.textures.get(name);
  if (!texture) {
    texture = makeTexture(gl);
    state.textures.set(name, texture);
  }
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA32F,
    state.width,
    state.height,
    0,
    gl.RGBA,
    gl.FLOAT,
    data
  );
  gl.bindTexture(gl.TEXTURE_2D, null);
  return texture;
}

function readTextureDataSample(data, index) {
  if (!data || !Number.isFinite(index) || index < 0) return null;
  const base = (index | 0) * 4;
  if (base + 3 >= data.length) return null;
  return [
    Number(data[base + 0]),
    Number(data[base + 1]),
    Number(data[base + 2]),
    Number(data[base + 3])
  ];
}

function summarizeTextureData(data, rawCount, width, sampleIndices) {
  const samples = {};
  for (const index of sampleIndices) {
    if (!Number.isFinite(index) || index < 0 || index >= rawCount) continue;
    samples[String(index | 0)] = {
      texCoord: [(index | 0) % width, Math.floor((index | 0) / width)],
      value: readTextureDataSample(data, index)
    };
  }
  return {
    floatLength: data?.length ?? 0,
    texelCount: data ? Math.floor(data.length / 4) : 0,
    samples
  };
}

function getRawValue(array, index, dim, component, fallback = 0) {
  if (!array || dim <= component) return fallback;
  const value = array[index * dim + component];
  return Number.isFinite(value) ? value : fallback;
}

function buildRawTextureData(raw, width, height) {
  const texelCount = width * height;
  const xyzOpacity = new Float32Array(texelCount * 4);
  const scaleTime = new Float32Array(texelCount * 4);
  const timeScale = new Float32Array(texelCount * 4);
  const rotation = new Float32Array(texelCount * 4);
  const rotationR = new Float32Array(texelCount * 4);

  for (let i = 0; i < raw.N; i++) {
    const o = i * 4;
    xyzOpacity[o + 0] = getRawValue(raw.xyz, i, raw.xyzDim, 0);
    xyzOpacity[o + 1] = getRawValue(raw.xyz, i, raw.xyzDim, 1);
    xyzOpacity[o + 2] = getRawValue(raw.xyz, i, raw.xyzDim, 2);
    xyzOpacity[o + 3] = getRawValue(raw.opacity, i, raw.opacityDim, 0);

    scaleTime[o + 0] = getRawValue(raw.scale_xyz, i, raw.scaleXYZDim, 0, 1);
    scaleTime[o + 1] = getRawValue(raw.scale_xyz, i, raw.scaleXYZDim, 1, 1);
    scaleTime[o + 2] = getRawValue(raw.scale_xyz, i, raw.scaleXYZDim, 2, 1);
    scaleTime[o + 3] = getRawValue(raw.t, i, raw.tDim, 0);
    timeScale[o + 0] = getRawValue(raw.t, i, raw.tDim, 0);
    timeScale[o + 1] = getRawValue(raw.scale_t, i, raw.scaleTDim, 0, 1);

    rotation[o + 0] = getRawValue(raw.rotation, i, raw.rotationDim, 0, 1);
    rotation[o + 1] = getRawValue(raw.rotation, i, raw.rotationDim, 1);
    rotation[o + 2] = getRawValue(raw.rotation, i, raw.rotationDim, 2);
    rotation[o + 3] = getRawValue(raw.rotation, i, raw.rotationDim, 3);

    rotationR[o + 0] = getRawValue(raw.rotation_r, i, raw.rotationRDim, 0, 1);
    rotationR[o + 1] = getRawValue(raw.rotation_r, i, raw.rotationRDim, 1);
    rotationR[o + 2] = getRawValue(raw.rotation_r, i, raw.rotationRDim, 2);
    rotationR[o + 3] = getRawValue(raw.rotation_r, i, raw.rotationRDim, 3);
  }

  return {
    xyzOpacity,
    scaleTime,
    timeScale,
    rotation,
    rotationR
  };
}

export function ensureRawAttributeTextures(gl, raw) {
  const totalStartMs = nowMs();
  if (!gl || !raw || !Number.isFinite(raw.N) || raw.N <= 0) {
    return {
      status: 'fallback',
      reason: 'raw-texture-input-unavailable',
      textures: null,
      summary: {
        status: 'fallback',
        reason: 'raw-texture-input-unavailable'
      }
    };
  }

  const state = getRawTextureState(gl);
  if (state.raw === raw && state.summary?.status === 'ok') {
    return {
      status: 'ok',
      reason: 'ok',
      textures: Object.fromEntries(state.textures),
      summary: {
        ...state.summary,
        reused: true,
        rawTextureEnsureMs: nowMs() - totalStartMs
      }
    };
  }

  const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) || 4096;
  const width = Math.min(maxTextureSize, Math.max(1, Math.ceil(Math.sqrt(raw.N))));
  const height = Math.ceil(raw.N / width);
  if (height > maxTextureSize) {
    return {
      status: 'fallback',
      reason: 'raw-texture-size-exceeds-webgl-limit',
      textures: null,
      summary: {
        status: 'fallback',
        reason: 'raw-texture-size-exceeds-webgl-limit',
        rawCount: raw.N,
        width,
        height,
        maxTextureSize
      }
    };
  }

  const buildStartMs = nowMs();
  const data = buildRawTextureData(raw, width, height);
  const buildMs = nowMs() - buildStartMs;
  const sampleIndices = Array.from(new Set([
    0,
    Math.min(Math.max(0, raw.N - 1), 658947)
  ].filter((index) => Number.isFinite(index) && index >= 0 && index < raw.N)));

  state.raw = raw;
  state.width = width;
  state.height = height;
  const uploadStartMs = nowMs();
  uploadTexture(gl, state, 'xyzOpacity', data.xyzOpacity);
  uploadTexture(gl, state, 'scaleTime', data.scaleTime);
  uploadTexture(gl, state, 'timeScale', data.timeScale);
  uploadTexture(gl, state, 'rotation', data.rotation);
  uploadTexture(gl, state, 'rotationR', data.rotationR);
  const uploadMs = nowMs() - uploadStartMs;
  const error = gl.getError();
  if (error !== gl.NO_ERROR) {
    return {
      status: 'fallback',
      reason: `raw-texture-upload-webgl-error-${error}`,
      textures: null,
      summary: {
        status: 'fallback',
        reason: `raw-texture-upload-webgl-error-${error}`,
        rawCount: raw.N,
        width,
        height,
        maxTextureSize
      }
    };
  }

  state.summary = {
    schemaVersion: 'step116-raw-attribute-texture-summary-v1',
    status: 'ok',
    reason: 'ok',
    reused: false,
    rawCount: raw.N,
    width,
    height,
    maxTextureSize,
    texelCount: width * height,
    expectedFloatLength: width * height * 4,
    capabilities: {
      maxTextureSize,
      maxTextureImageUnits: gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS),
      maxVertexTextureImageUnits: gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS),
      maxCombinedTextureImageUnits: gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS)
    },
    textureNames: ['xyzOpacity', 'scaleTime', 'timeScale', 'rotation', 'rotationR'],
    format: 'RGBA32F',
    formatDetail: {
      internalFormat: 'RGBA32F',
      format: 'RGBA',
      type: 'FLOAT',
      minFilter: 'NEAREST',
      magFilter: 'NEAREST',
      wrapS: 'CLAMP_TO_EDGE',
      wrapT: 'CLAMP_TO_EDGE',
      baseLevel: 0,
      maxLevel: 0,
      unpackAlignment: 1
    },
    uploadDataSummary: {
      sampleIndices,
      xyzOpacity: summarizeTextureData(data.xyzOpacity, raw.N, width, sampleIndices),
      scaleTime: summarizeTextureData(data.scaleTime, raw.N, width, sampleIndices),
      timeScale: summarizeTextureData(data.timeScale, raw.N, width, sampleIndices),
      rotation: summarizeTextureData(data.rotation, raw.N, width, sampleIndices),
      rotationR: summarizeTextureData(data.rotationR, raw.N, width, sampleIndices)
    },
    buildRawTextureDataMs: buildMs,
    rawTextureUploadMs: uploadMs,
    rawTextureEnsureMs: nowMs() - totalStartMs
  };

  return {
    status: 'ok',
    reason: 'ok',
    textures: Object.fromEntries(state.textures),
    summary: state.summary
  };
}
