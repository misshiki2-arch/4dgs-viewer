// Step38 stage 2
// GPU transform backend helper.
//
// Purpose:
// - provide a narrow helper layer for a future GPU-backed transform implementation
// - keep backend capability / readiness probing and backend state in one place
// - expose a single execution entry that executor code can import later
//
// Non-goals for this stage:
// - deciding requested / actual transform paths
// - deciding fallback reasons
// - owning transform upload truth
// - changing public transform contracts
//
// This file is intentionally a thin stub. The executor remains the truth source.

import { GPU_VISIBLE_PACK_FLOATS_PER_ITEM } from './gpu_buffer_layout_utils.js';
import { packVisibleItems } from './gpu_visible_pack_utils.js';

const GPU_BACKEND_ID = 'gpu-transform-backend';
const GPU_BACKEND_REASON_NOT_IMPLEMENTED = 'gpu-backend-not-implemented';
const GPU_BACKEND_REASON_MISSING_WEBGL2 = 'gpu-backend-missing-webgl2';
const GPU_BACKEND_REASON_MISSING_FLOAT_COLOR = 'gpu-backend-missing-float-color';
const GPU_BACKEND_REASON_TRIAL_FAILED = 'gpu-backend-trial-failed';
const GPU_BACKEND_REASON_CPU_FALLBACK = 'gpu-backend-cpu-fallback';
const GPU_BACKEND_SMALL_BATCH_MAX_ITEMS = 64;
const GPU_BACKEND_IMPLEMENTATION_STATE_STUB = 'stub';
const GPU_BACKEND_IMPLEMENTATION_STATE_MINIMAL_GPU_PACK = 'minimal-gpu-pack';
const GPU_BACKEND_EXECUTION_MODE_CPU_FALLBACK = 'cpu-fallback-stub';
const GPU_BACKEND_EXECUTION_MODE_GPU_TRIAL_CPU_FALLBACK = 'gpu-trial-cpu-fallback';
const GPU_BACKEND_EXECUTION_MODE_GPU_SMALL_BATCH_PACK = 'gpu-small-batch-pack';
const GPU_BACKEND_DISABLE_PACKED_WRITE_REUSE_DIAGNOSTIC = false;

function nowMs() {
  return performance.now();
}

function probeWebGL2Support(gl) {
  return !!gl && typeof gl.createVertexArray === 'function';
}

function probeFloatColorSupport(gl) {
  if (!probeWebGL2Support(gl) || typeof gl.getExtension !== 'function') return false;
  return !!gl.getExtension('EXT_color_buffer_float');
}

function getMaxTextureSize(gl) {
  if (!probeWebGL2Support(gl) || typeof gl.getParameter !== 'function') return 0;
  const value = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  return Number.isFinite(value) ? value : 0;
}

function getMaxSupportedBatchItems(gl) {
  const maxTextureSize = getMaxTextureSize(gl);
  if (!Number.isFinite(maxTextureSize) || maxTextureSize <= 0) return 0;
  return Math.max(1, Math.min(GPU_BACKEND_SMALL_BATCH_MAX_ITEMS, maxTextureSize));
}

function buildBackendError(error) {
  if (!error) return null;
  if (typeof error === 'string') return error;
  if (typeof error?.message === 'string' && error.message.length > 0) return error.message;
  return String(error);
}

function normalizeSourceCount(sourceItemsResult) {
  if (Number.isFinite(sourceItemsResult?.itemCount)) return sourceItemsResult.itemCount;
  if (Array.isArray(sourceItemsResult?.items)) return sourceItemsResult.items.length;
  return 0;
}

function normalizeSourceItems(sourceItemsResult) {
  return Array.isArray(sourceItemsResult?.items) ? sourceItemsResult.items : [];
}

function toFiniteOrZero(value) {
  return Number.isFinite(value) ? value : 0;
}

function normalizeVec(itemValue, fallbackLength) {
  const out = new Array(fallbackLength);
  for (let i = 0; i < fallbackLength; i++) {
    out[i] = Array.isArray(itemValue) ? toFiniteOrZero(itemValue[i]) : 0;
  }
  return out;
}

function buildPackedRowsFromSourceItem(item) {
  const centerPx = normalizeVec(item?.centerPx, 2);
  const colorAlpha = normalizeVec(item?.colorAlpha, 4);
  const conic = normalizeVec(item?.conic, 3);
  const misc = normalizeVec(item?.misc, 4);

  return [
    new Float32Array([
      centerPx[0],
      centerPx[1],
      toFiniteOrZero(item?.radiusPx),
      toFiniteOrZero(item?.depth)
    ]),
    new Float32Array(colorAlpha),
    new Float32Array([
      conic[0],
      conic[1],
      conic[2],
      toFiniteOrZero(item?.reserved)
    ]),
    new Float32Array(misc)
  ];
}

function runCpuFallbackPack(sourceItemsResult) {
  const items = normalizeSourceItems(sourceItemsResult);
  const packedResult = packVisibleItems(items);
  return {
    packed: packedResult?.packed instanceof Float32Array ? packedResult.packed : null,
    count: Number.isFinite(packedResult?.count) ? packedResult.count : items.length,
    floatsPerItem: Number.isFinite(packedResult?.floatsPerItem)
      ? packedResult.floatsPerItem
      : GPU_VISIBLE_PACK_FLOATS_PER_ITEM
  };
}

function isSameWebGlContext(contextGl, gl) {
  return !!contextGl && !!gl && contextGl === gl;
}

function hasReusableTrialResources(context, gl) {
  return (
    isSameWebGlContext(context?.trialResourceGl, gl) &&
    !!context?.trialResources?.buffer &&
    !!context?.trialResources?.vao
  );
}

function hasReusablePackedWriteResources(context, gl, targetHeight, maxSupportedBatchItems) {
  if (GPU_BACKEND_DISABLE_PACKED_WRITE_REUSE_DIAGNOSTIC) return false;
  return (
    isSameWebGlContext(context?.packedWriteResourceGl, gl) &&
    !!context?.packedWriteResources?.program &&
    !!context?.packedWriteResources?.framebuffer &&
    !!context?.packedWriteResources?.texture &&
    !!context?.packedWriteResources?.vao &&
    Number.isFinite(context?.packedWriteAllocatedHeight) &&
    context.packedWriteAllocatedHeight >= targetHeight &&
    Number.isFinite(context?.packedWriteMaxItems) &&
    context.packedWriteMaxItems === maxSupportedBatchItems
  );
}

function disposePackedWriteResources(context, gl) {
  const resources = context?.packedWriteResources;
  if (!resources || !gl || !isSameWebGlContext(context?.packedWriteResourceGl, gl)) return;

  if (resources.program) gl.deleteProgram(resources.program);
  if (resources.texture) gl.deleteTexture(resources.texture);
  if (resources.framebuffer) gl.deleteFramebuffer(resources.framebuffer);
  if (resources.vao) gl.deleteVertexArray(resources.vao);

  context.packedWriteResources = null;
  context.packedWriteResourceGl = null;
  context.packedWriteAllocatedHeight = 0;
}

function ensureGpuTrialResources(context, gl) {
  if (!gl || !probeWebGL2Support(gl)) {
    return {
      ok: false,
      stage: 'probe-failed',
      reason: GPU_BACKEND_REASON_MISSING_WEBGL2,
      error: null
    };
  }

  if (hasReusableTrialResources(context, gl)) {
    return {
      ok: true,
      stage: 'resources-ready',
      reason: 'ready',
      error: null
    };
  }

  try {
    const buffer = gl.createBuffer();
    const vao = gl.createVertexArray();
    if (!buffer || !vao) {
      return {
        ok: false,
        stage: 'resource-init-failed',
        reason: GPU_BACKEND_REASON_TRIAL_FAILED,
        error: 'failed-to-create-gpu-trial-resources'
      };
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, 4, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    if (context) {
      context.trialResources = { buffer, vao };
      context.trialResourceGl = gl;
    }

    return {
      ok: true,
      stage: 'resources-initialized',
      reason: 'ready',
      error: null
    };
  } catch (error) {
    return {
      ok: false,
      stage: 'resource-init-exception',
      reason: GPU_BACKEND_REASON_TRIAL_FAILED,
      error: buildBackendError(error)
    };
  }
}

function ensureGpuPackedWriteResources(context, gl) {
  if (!gl || !probeWebGL2Support(gl)) {
    return {
      ok: false,
      stage: 'gpu-pack-missing-webgl2',
      reason: GPU_BACKEND_REASON_MISSING_WEBGL2,
      error: null
    };
  }

  if (!probeFloatColorSupport(gl)) {
    return {
      ok: false,
      stage: 'gpu-pack-missing-float-color',
      reason: GPU_BACKEND_REASON_MISSING_FLOAT_COLOR,
      error: null
    };
  }

  const maxSupportedBatchItems = getMaxSupportedBatchItems(gl);
  const targetHeight = Math.max(1, Math.min(
    maxSupportedBatchItems,
    Number.isFinite(context?.packedWriteHeight) ? context.packedWriteHeight : 1
  ));

  if (hasReusablePackedWriteResources(context, gl, targetHeight, maxSupportedBatchItems)) {
    return {
      ok: true,
      stage: 'gpu-pack-resources-ready',
      reason: 'ready',
      error: null
    };
  }

  const vertexShaderSource = `#version 300 es
void main() {
  vec2 pos;
  if (gl_VertexID == 0) pos = vec2(-1.0, -1.0);
  else if (gl_VertexID == 1) pos = vec2(3.0, -1.0);
  else pos = vec2(-1.0, 3.0);
  gl_Position = vec4(pos, 0.0, 1.0);
}`;

  const fragmentShaderSource = `#version 300 es
precision highp float;
uniform vec4 uPackedRow0;
uniform vec4 uPackedRow1;
uniform vec4 uPackedRow2;
uniform vec4 uPackedRow3;
out vec4 outColor;
void main() {
  int column = int(floor(gl_FragCoord.x));
  if (column == 0) outColor = uPackedRow0;
  else if (column == 1) outColor = uPackedRow1;
  else if (column == 2) outColor = uPackedRow2;
  else outColor = uPackedRow3;
}`;

  try {
    // Step44/45 diagnostic rollback stage 2:
    // allow packed-write resource reuse again for the same GL/size contract,
    // while still keeping the safer per-frame target initialization below.
    if (GPU_BACKEND_DISABLE_PACKED_WRITE_REUSE_DIAGNOSTIC) {
      disposePackedWriteResources(context, gl);
    }

    if (
      isSameWebGlContext(context?.packedWriteResourceGl, gl) &&
      context?.packedWriteResources?.texture &&
      context?.packedWriteResources?.framebuffer &&
      context?.packedWriteResources?.program &&
      context?.packedWriteResources?.vao &&
      Number.isFinite(context?.packedWriteAllocatedHeight) &&
      Number.isFinite(context?.packedWriteMaxItems) &&
      context.packedWriteMaxItems === maxSupportedBatchItems
    ) {
      gl.bindTexture(gl.TEXTURE_2D, context.packedWriteResources.texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 4, targetHeight, 0, gl.RGBA, gl.FLOAT, null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, context.packedWriteResources.framebuffer);
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        context.packedWriteResources.texture,
        0
      );
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.bindTexture(gl.TEXTURE_2D, null);

      if (status !== gl.FRAMEBUFFER_COMPLETE) {
        return {
          ok: false,
          stage: 'gpu-pack-framebuffer-incomplete',
          reason: GPU_BACKEND_REASON_TRIAL_FAILED,
          error: `framebuffer-status-${status}`
        };
      }

      if (context) {
        context.packedWriteHeight = targetHeight;
        context.packedWriteAllocatedHeight = targetHeight;
      }

      return {
        ok: true,
        stage: 'gpu-pack-resources-resized',
        reason: 'ready',
        error: null
      };
    }

    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    const program = gl.createProgram();
    const vao = gl.createVertexArray();
    const texture = gl.createTexture();
    const framebuffer = gl.createFramebuffer();

    if (!vertexShader || !fragmentShader || !program || !vao || !texture || !framebuffer) {
      return {
        ok: false,
        stage: 'gpu-pack-resource-create-failed',
        reason: GPU_BACKEND_REASON_TRIAL_FAILED,
        error: 'failed-to-create-gpu-pack-resources'
      };
    }

    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);
    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
      return {
        ok: false,
        stage: 'gpu-pack-vertex-compile-failed',
        reason: GPU_BACKEND_REASON_TRIAL_FAILED,
        error: gl.getShaderInfoLog(vertexShader) || 'vertex-shader-compile-failed'
      };
    }

    gl.shaderSource(fragmentShader, fragmentShaderSource);
    gl.compileShader(fragmentShader);
    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
      return {
        ok: false,
        stage: 'gpu-pack-fragment-compile-failed',
        reason: GPU_BACKEND_REASON_TRIAL_FAILED,
        error: gl.getShaderInfoLog(fragmentShader) || 'fragment-shader-compile-failed'
      };
    }

    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      return {
        ok: false,
        stage: 'gpu-pack-link-failed',
        reason: GPU_BACKEND_REASON_TRIAL_FAILED,
        error: gl.getProgramInfoLog(program) || 'gpu-pack-link-failed'
      };
    }

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 4, targetHeight, 0, gl.RGBA, gl.FLOAT, null);

    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);

    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      return {
        ok: false,
        stage: 'gpu-pack-framebuffer-incomplete',
        reason: GPU_BACKEND_REASON_TRIAL_FAILED,
        error: `framebuffer-status-${status}`
      };
    }

    if (context) {
      context.packedWriteResources = {
        program,
        vao,
        texture,
        framebuffer,
        uniformRow0: gl.getUniformLocation(program, 'uPackedRow0'),
        uniformRow1: gl.getUniformLocation(program, 'uPackedRow1'),
        uniformRow2: gl.getUniformLocation(program, 'uPackedRow2'),
        uniformRow3: gl.getUniformLocation(program, 'uPackedRow3')
      };
      context.packedWriteHeight = targetHeight;
      context.packedWriteAllocatedHeight = targetHeight;
      context.packedWriteMaxItems = maxSupportedBatchItems;
      context.packedWriteResourceGl = gl;
    }

    return {
      ok: true,
      stage: 'gpu-pack-resources-initialized',
      reason: 'ready',
      error: null
    };
  } catch (error) {
    return {
      ok: false,
      stage: 'gpu-pack-resource-exception',
      reason: GPU_BACKEND_REASON_TRIAL_FAILED,
      error: buildBackendError(error)
    };
  }
}

function tryGenerateGpuPackedSmallBatch(context, sourceItemsResult, gl) {
  const items = normalizeSourceItems(sourceItemsResult);
  const maxSupportedBatchItems = getMaxSupportedBatchItems(gl);
  if (items.length <= 0 || items.length > maxSupportedBatchItems) {
    return {
      producedPacked: false,
      packed: null,
      count: 0,
      floatsPerItem: GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
      stage: 'gpu-pack-small-batch-size-unsupported',
      reason: GPU_BACKEND_REASON_CPU_FALLBACK,
      error: null
    };
  }

  if (context) {
    context.packedWriteHeight = items.length;
  }

  const resourceState = ensureGpuPackedWriteResources(context, gl);
  if (!resourceState.ok) {
    return {
      producedPacked: false,
      packed: null,
      count: 0,
      floatsPerItem: GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
      stage: resourceState.stage,
      reason: resourceState.reason,
      error: resourceState.error
    };
  }

  const resources = context?.packedWriteResources;
  const out = new Float32Array(items.length * GPU_VISIBLE_PACK_FLOATS_PER_ITEM);

  try {
    gl.bindFramebuffer(gl.FRAMEBUFFER, resources.framebuffer);
    gl.useProgram(resources.program);
    gl.bindVertexArray(resources.vao);
    // Packed write pass contract:
    // this pass writes formal packed payload rows into an offscreen float target.
    // It is a data-write pass, not a visual composition pass, so BLEND must be
    // disabled here. Otherwise previous framebuffer contents can be mixed into
    // the new payload and silently corrupt later full-frame draws.
    gl.disable(gl.SCISSOR_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.viewport(0, 0, 4, items.length);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    for (let i = 0; i < items.length; i++) {
      const rows = buildPackedRowsFromSourceItem(items[i]);
      gl.viewport(0, i, 4, 1);
      gl.uniform4fv(resources.uniformRow0, rows[0]);
      gl.uniform4fv(resources.uniformRow1, rows[1]);
      gl.uniform4fv(resources.uniformRow2, rows[2]);
      gl.uniform4fv(resources.uniformRow3, rows[3]);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    // Step44/45 diagnostic rollback stage 1:
    // keep the safer per-frame target initialization and packed-write reuse disable,
    // but remove the heaviest global GPU/CPU sync first so we can isolate whether
    // readback correctness really depends on gl.finish().
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, resources.framebuffer);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    gl.readPixels(0, 0, 4, items.length, gl.RGBA, gl.FLOAT, out);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(null);

    return {
      producedPacked: true,
      packed: out,
      count: items.length,
      floatsPerItem: GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
      stage: 'gpu-pack-small-batch-generated',
      reason: 'gpu-packed-generated',
      error: null
    };
  } catch (error) {
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(null);
    return {
      producedPacked: false,
      packed: null,
      count: 0,
      floatsPerItem: GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
      stage: 'gpu-pack-small-batch-failed',
      reason: GPU_BACKEND_REASON_TRIAL_FAILED,
      error: buildBackendError(error)
    };
  }
}

function executeGpuTransformTrial(context, sourceItemsResult, gl) {
  const resourceState = ensureGpuTrialResources(context, gl);
  if (!resourceState.ok) {
    return {
      triedGpu: false,
      stage: resourceState.stage,
      reason: resourceState.reason,
      error: resourceState.error
    };
  }

  try {
    // Minimal Step39 trial:
    // touch GPU-owned resources and prove that the backend can initialize safely.
    gl.bindVertexArray(context?.trialResources?.vao ?? null);
    gl.bindBuffer(gl.ARRAY_BUFFER, context?.trialResources?.buffer ?? null);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, new Float32Array([normalizeSourceCount(sourceItemsResult)]));
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindVertexArray(null);

    return {
      triedGpu: true,
      stage: 'gpu-trial-executed',
      reason: GPU_BACKEND_REASON_CPU_FALLBACK,
      error: null
    };
  } catch (error) {
    return {
      triedGpu: true,
      stage: 'gpu-trial-failed',
      reason: GPU_BACKEND_REASON_TRIAL_FAILED,
      error: buildBackendError(error)
    };
  }
}

export function createGpuScreenTransformBackendGpuContext(initialState = {}) {
  return {
    backendId: GPU_BACKEND_ID,
    implementationState: GPU_BACKEND_IMPLEMENTATION_STATE_STUB,
    executionMode: GPU_BACKEND_EXECUTION_MODE_CPU_FALLBACK,
    isProbed: false,
    isAvailable: false,
    isReady: false,
    isImplemented: false,
    reason: GPU_BACKEND_REASON_NOT_IMPLEMENTED,
    supportsWebGL2: false,
    supportsTransformPath: false,
    lastProbeMs: 0,
    lastExecutionMs: 0,
    lastSourceItemCount: 0,
    lastCount: 0,
    lastStageName: 'idle',
    lastError: null,
    trialResources: null,
    trialResourceGl: null,
    packedWriteResources: null,
    packedWriteResourceGl: null,
    packedWriteHeight: 1,
    packedWriteAllocatedHeight: 0,
    packedWriteMaxItems: GPU_BACKEND_SMALL_BATCH_MAX_ITEMS,
    ...initialState
  };
}

export function summarizeGpuScreenTransformBackendGpuCapability(context, gl = null) {
  const supportsWebGL2 = probeWebGL2Support(gl);
  const isReady = !!context?.isReady && supportsWebGL2;
  const maxTextureSize = getMaxTextureSize(gl);
  const maxBatchItems = supportsWebGL2 ? getMaxSupportedBatchItems(gl) : 0;

  return {
    backendId: context?.backendId ?? GPU_BACKEND_ID,
    implementationState: context?.implementationState ?? GPU_BACKEND_IMPLEMENTATION_STATE_STUB,
    executionMode: context?.executionMode ?? GPU_BACKEND_EXECUTION_MODE_CPU_FALLBACK,
    isProbed: !!context?.isProbed,
    isAvailable: !!context?.isAvailable,
    isReady,
    isImplemented: !!context?.isImplemented,
    reason: context?.reason ?? GPU_BACKEND_REASON_NOT_IMPLEMENTED,
    supportsWebGL2,
    supportsTransformPath: !!context?.supportsTransformPath,
    maxTextureSize,
    maxBatchItems,
    lastProbeMs: Number.isFinite(context?.lastProbeMs) ? context.lastProbeMs : 0,
    lastExecutionMs: Number.isFinite(context?.lastExecutionMs) ? context.lastExecutionMs : 0,
    lastSourceItemCount: Number.isFinite(context?.lastSourceItemCount) ? context.lastSourceItemCount : 0,
    lastCount: Number.isFinite(context?.lastCount) ? context.lastCount : 0,
    lastStageName: context?.lastStageName ?? 'idle',
    lastError: context?.lastError ?? null
  };
}

export function executeGpuScreenTransformBackendGpu(context, sourceItemsResult, options = {}) {
  const t0 = nowMs();
  const sourceItemCount = normalizeSourceCount(sourceItemsResult);
  const gl = options?.gl ?? null;
  const supportsWebGL2 = probeWebGL2Support(gl);
  const cpuFallback = runCpuFallbackPack(sourceItemsResult);
  const trialResult = executeGpuTransformTrial(context, sourceItemsResult, gl);
  const gpuPackedResult = trialResult.triedGpu
    ? tryGenerateGpuPackedSmallBatch(context, sourceItemsResult, gl)
    : {
        producedPacked: false,
        packed: null,
        count: 0,
        floatsPerItem: GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
        stage: trialResult.stage,
        reason: trialResult.reason,
        error: trialResult.error
      };
  const producedPacked = !!gpuPackedResult.producedPacked;
  const ready = producedPacked;
  const implementationState = producedPacked
    ? GPU_BACKEND_IMPLEMENTATION_STATE_MINIMAL_GPU_PACK
    : (trialResult.triedGpu ? 'trial-only' : GPU_BACKEND_IMPLEMENTATION_STATE_STUB);
  const executionMode = producedPacked
    ? GPU_BACKEND_EXECUTION_MODE_GPU_SMALL_BATCH_PACK
    : trialResult.triedGpu
      ? GPU_BACKEND_EXECUTION_MODE_GPU_TRIAL_CPU_FALLBACK
      : GPU_BACKEND_EXECUTION_MODE_CPU_FALLBACK;
  const reason = producedPacked
    ? gpuPackedResult.reason
    : (gpuPackedResult.reason ?? (supportsWebGL2 ? GPU_BACKEND_REASON_CPU_FALLBACK : GPU_BACKEND_REASON_MISSING_WEBGL2));
  const stageMs = nowMs() - t0;
  const finalPacked = producedPacked ? gpuPackedResult.packed : cpuFallback.packed;
  const finalCount = producedPacked ? gpuPackedResult.count : cpuFallback.count;
  const finalFloatsPerItem = producedPacked ? gpuPackedResult.floatsPerItem : cpuFallback.floatsPerItem;

  if (context) {
    context.implementationState = implementationState;
    context.executionMode = executionMode;
    context.isProbed = true;
    context.supportsWebGL2 = supportsWebGL2;
    context.isAvailable = supportsWebGL2;
    context.isReady = ready;
    context.isImplemented = !!trialResult.triedGpu;
    context.supportsTransformPath = !!trialResult.triedGpu;
    context.reason = reason;
    context.lastProbeMs = stageMs;
    context.lastExecutionMs = stageMs;
    context.lastSourceItemCount = sourceItemCount;
    context.lastCount = finalCount;
    context.lastStageName = producedPacked ? gpuPackedResult.stage : trialResult.stage;
    context.lastError = producedPacked ? null : (gpuPackedResult.error ?? trialResult.error);
    if (Number.isFinite(sourceItemCount) && sourceItemCount > 0) {
      context.packedWriteHeight = Math.min(
        Number.isFinite(context.packedWriteAllocatedHeight) && context.packedWriteAllocatedHeight > 0
          ? context.packedWriteAllocatedHeight
          : sourceItemCount,
        Number.isFinite(context.packedWriteMaxItems) && context.packedWriteMaxItems > 0
          ? context.packedWriteMaxItems
          : sourceItemCount
      );
    }
  }

  return {
    backendId: GPU_BACKEND_ID,
    implementationState,
    executionMode,
    ready,
    reason,
    stageMs,
    packed: finalPacked,
    count: finalCount,
    floatsPerItem: finalFloatsPerItem,
    sourceItemCount,
    backendStage: producedPacked ? gpuPackedResult.stage : trialResult.stage,
    backendFallback: !producedPacked,
    backendImplemented: !!trialResult.triedGpu,
    backendProducedPacked: producedPacked,
    backendFallbackToCpu: !producedPacked,
    backendError: producedPacked ? null : (gpuPackedResult.error ?? trialResult.error),
    backendContext: summarizeGpuScreenTransformBackendGpuCapability(context, gl)
  };
}
