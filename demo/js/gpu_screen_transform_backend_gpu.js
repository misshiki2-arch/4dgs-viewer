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
import {
  buildGpuPackedPayloadAtlas,
  ensureGpuPackedPayloadTextureDrawResources
} from './gpu_packed_payload_draw_shared.js';

const GPU_BACKEND_ID = 'gpu-transform-backend';
const GPU_BACKEND_REASON_NOT_IMPLEMENTED = 'gpu-backend-not-implemented';
const GPU_BACKEND_REASON_MISSING_WEBGL2 = 'gpu-backend-missing-webgl2';
const GPU_BACKEND_REASON_MISSING_FLOAT_COLOR = 'gpu-backend-missing-float-color';
const GPU_BACKEND_REASON_TRIAL_FAILED = 'gpu-backend-trial-failed';
const GPU_BACKEND_REASON_CPU_FALLBACK = 'gpu-backend-cpu-fallback';
const GPU_BACKEND_SMALL_BATCH_MAX_ITEMS = 256;
const GPU_BACKEND_IMPLEMENTATION_STATE_STUB = 'stub';
const GPU_BACKEND_IMPLEMENTATION_STATE_MINIMAL_GPU_PACK = 'minimal-gpu-pack';
const GPU_BACKEND_EXECUTION_MODE_CPU_FALLBACK = 'cpu-fallback-stub';
const GPU_BACKEND_EXECUTION_MODE_GPU_SMALL_BATCH_PACK = 'gpu-small-batch-pack';
const GPU_PACKED_PAYLOAD_WIDTH = 4;
const GPU_BACKEND_BASE_REUSABLE_PAYLOAD_TEXTURES = 2;
const GPU_BACKEND_HARD_MAX_REUSABLE_PAYLOAD_TEXTURES = 16;

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

function resolvePreferredBatchItems(context, maxBatchItems) {
  const maxItems = Number.isFinite(maxBatchItems) ? Math.max(0, maxBatchItems | 0) : 0;
  if (maxItems <= 0) return 0;

  const lastSuccessfulDispatchItemCount = Number.isFinite(context?.lastSuccessfulDispatchItemCount)
    ? Math.max(0, context.lastSuccessfulDispatchItemCount | 0)
    : 0;
  const lastSuccessfulDispatchMode = context?.lastSuccessfulDispatchMode ?? 'none';

  if (lastSuccessfulDispatchMode === 'single-texture-copy-pass' && lastSuccessfulDispatchItemCount > 0) {
    const successFloor = Math.max(64, lastSuccessfulDispatchItemCount);
    const exploratoryStep = successFloor >= 128 ? successFloor : successFloor * 2;
    return Math.min(maxItems, exploratoryStep);
  }

  return Math.min(maxItems, 64);
}

function resolvePreferredBatchPolicy(context, preferredBatchItems, maxBatchItems) {
  const preferredItems = Number.isFinite(preferredBatchItems) ? Math.max(0, preferredBatchItems | 0) : 0;
  const maxItems = Number.isFinite(maxBatchItems) ? Math.max(0, maxBatchItems | 0) : 0;
  const lastSuccessfulDispatchItemCount = Number.isFinite(context?.lastSuccessfulDispatchItemCount)
    ? Math.max(0, context.lastSuccessfulDispatchItemCount | 0)
    : 0;
  const lastSuccessfulDispatchMode = context?.lastSuccessfulDispatchMode ?? 'none';

  if (preferredItems <= 0) return 'preferred-batch-none';
  if (lastSuccessfulDispatchMode !== 'single-texture-copy-pass' || lastSuccessfulDispatchItemCount <= 0) {
    return preferredItems >= maxItems ? 'preferred-batch-clamp-hard-cap' : 'preferred-batch-bootstrap-64';
  }
  if (preferredItems > lastSuccessfulDispatchItemCount) {
    return preferredItems >= maxItems
      ? 'preferred-batch-explore-hit-hard-cap'
      : 'preferred-batch-explore-grow';
  }
  if (preferredItems >= maxItems) return 'preferred-batch-clamp-hard-cap';
  return 'preferred-batch-hold-success';
}

function buildBackendError(error) {
  if (!error) return null;
  if (typeof error === 'string') return error;
  if (typeof error?.message === 'string' && error.message.length > 0) return error.message;
  return String(error);
}

function createGpuPackedPayloadRecord(gl, texture, count) {
  return {
    kind: 'gpu-packed-texture',
    gl,
    texture,
    width: GPU_PACKED_PAYLOAD_WIDTH,
    height: Math.max(0, count | 0),
    count: Math.max(0, count | 0),
    rowsPerColumn: Math.max(1, count | 0),
    columnCount: 1,
    floatsPerItem: GPU_VISIBLE_PACK_FLOATS_PER_ITEM
  };
}

function countGpuResidentPackedPayloads(context) {
  const residentCount = Array.isArray(context?.gpuResidentPackedPayloads)
    ? context.gpuResidentPackedPayloads.length
    : 0;
  return residentCount + (context?.atlasGpuPackedPayload?.texture ? 1 : 0);
}

function countReusableGpuResidentPayloadTextures(context) {
  return Array.isArray(context?.reusableGpuResidentPayloadTextures)
    ? context.reusableGpuResidentPayloadTextures.length
    : 0;
}

function getReusableGpuResidentPayloadTextureLimit(context) {
  const configuredLimit = context?.payloadPoolMaxRetained;
  if (Number.isFinite(configuredLimit)) {
    return Math.max(0, configuredLimit | 0);
  }
  return GPU_BACKEND_BASE_REUSABLE_PAYLOAD_TEXTURES;
}

function getReusableGpuResidentPayloadTextureBaseLimit(context) {
  const configuredLimit = context?.payloadPoolBaseRetained;
  if (Number.isFinite(configuredLimit)) {
    return Math.max(0, configuredLimit | 0);
  }
  return GPU_BACKEND_BASE_REUSABLE_PAYLOAD_TEXTURES;
}

function getReusableGpuResidentPayloadTextureHardLimit(context) {
  const configuredLimit = context?.payloadPoolHardMaxRetained;
  if (Number.isFinite(configuredLimit)) {
    return Math.max(0, configuredLimit | 0);
  }
  return GPU_BACKEND_HARD_MAX_REUSABLE_PAYLOAD_TEXTURES;
}

function resolveAdaptiveReusablePayloadTextureLimit(context) {
  const baseLimit = getReusableGpuResidentPayloadTextureBaseLimit(context);
  const hardLimit = Math.max(baseLimit, getReusableGpuResidentPayloadTextureHardLimit(context));
  const activeCount = countGpuResidentPackedPayloads(context);
  const recentReuseCount = Number.isFinite(context?.lastPayloadReuseCount)
    ? Math.max(0, context.lastPayloadReuseCount | 0)
    : 0;
  const recentCreateCount = Number.isFinite(context?.lastPayloadCreateCount)
    ? Math.max(0, context.lastPayloadCreateCount | 0)
    : 0;

  const burstRetain = Math.ceil(recentCreateCount / 8);
  const reuseRetain = recentReuseCount + (recentCreateCount > 0 ? 1 : 0);
  const releaseCap = activeCount > 0 ? activeCount : baseLimit;
  const adaptiveDemand = Math.max(baseLimit, burstRetain, reuseRetain);

  return Math.max(
    baseLimit,
    Math.min(hardLimit, adaptiveDemand, Math.max(baseLimit, releaseCap))
  );
}

function resolveAdaptiveReusablePayloadTexturePolicy(context) {
  const baseLimit = getReusableGpuResidentPayloadTextureBaseLimit(context);
  const hardLimit = Math.max(baseLimit, getReusableGpuResidentPayloadTextureHardLimit(context));
  const nextLimit = resolveAdaptiveReusablePayloadTextureLimit(context);
  const recentReuseCount = Number.isFinite(context?.lastPayloadReuseCount)
    ? Math.max(0, context.lastPayloadReuseCount | 0)
    : 0;
  const recentCreateCount = Number.isFinite(context?.lastPayloadCreateCount)
    ? Math.max(0, context.lastPayloadCreateCount | 0)
    : 0;

  let reason = 'adaptive-hold-base';
  if (nextLimit >= hardLimit && nextLimit > baseLimit) {
    reason = 'adaptive-hit-hard-cap';
  } else if (nextLimit > baseLimit && recentCreateCount > recentReuseCount) {
    reason = 'adaptive-grow-create-pressure';
  } else if (nextLimit > baseLimit && recentReuseCount > 0) {
    reason = 'adaptive-grow-reuse-pressure';
  } else if (nextLimit === baseLimit && (recentReuseCount > 0 || recentCreateCount > 0)) {
    reason = 'adaptive-clamp-base';
  }

  return {
    nextLimit,
    reason
  };
}

function trimReusableGpuResidentPayloadTextures(context, limit = getReusableGpuResidentPayloadTextureLimit(context)) {
  if (!context || !Array.isArray(context.reusableGpuResidentPayloadTextures)) return 0;

  const normalizedLimit = Math.max(0, limit | 0);
  let trimmedCount = 0;

  while (context.reusableGpuResidentPayloadTextures.length > normalizedLimit) {
    const entry = context.reusableGpuResidentPayloadTextures.shift();
    if (entry?.texture && entry?.gl && typeof entry.gl.deleteTexture === 'function') {
      entry.gl.deleteTexture(entry.texture);
      trimmedCount++;
    }
  }

  context.reusablePayloadCount = context.reusableGpuResidentPayloadTextures.length;
  context.lastPayloadTrimCount = trimmedCount;
  context.lastPayloadRetainedCount = context.reusablePayloadCount;
  return trimmedCount;
}

function releaseGpuResidentPackedPayloads(context) {
  if (!context || !Array.isArray(context.gpuResidentPackedPayloads)) return 0;

  context.reusableGpuResidentPayloadTextures = context.reusableGpuResidentPayloadTextures ?? [];
  let releasedCount = 0;
  let pooledCount = 0;
  for (const payload of context.gpuResidentPackedPayloads) {
    if (!payload?.texture || !payload?.gl || typeof payload.gl.deleteTexture !== 'function') continue;
    if (probeWebGL2Support(payload.gl)) {
      context.reusableGpuResidentPayloadTextures.push({
        gl: payload.gl,
        texture: payload.texture
      });
      pooledCount++;
    } else {
      payload.gl.deleteTexture(payload.texture);
    }
    releasedCount++;
  }

  context.gpuResidentPackedPayloads = [];
  context.activePayloadCount = 0;
  context.reusablePayloadCount = countReusableGpuResidentPayloadTextures(context);
  context.payloadPoolHighWaterCount = Math.max(
    Number.isFinite(context.payloadPoolHighWaterCount) ? context.payloadPoolHighWaterCount : 0,
    context.reusablePayloadCount
  );
  context.lastPayloadPoolReleaseCount = pooledCount;
  trimReusableGpuResidentPayloadTextures(context);
  return releasedCount;
}

function releaseSpecificGpuResidentPackedPayloads(context, payloads) {
  if (!context || !Array.isArray(context.gpuResidentPackedPayloads) || !Array.isArray(payloads) || payloads.length <= 0) {
    return 0;
  }

  const releaseSet = new Set(payloads);
  const retainedPayloads = [];
  let releasedCount = 0;
  let pooledCount = 0;

  context.reusableGpuResidentPayloadTextures = context.reusableGpuResidentPayloadTextures ?? [];

  for (const payload of context.gpuResidentPackedPayloads) {
    if (!releaseSet.has(payload)) {
      retainedPayloads.push(payload);
      continue;
    }
    if (payload?.texture && payload?.gl && typeof payload.gl.deleteTexture === 'function' && probeWebGL2Support(payload.gl)) {
      context.reusableGpuResidentPayloadTextures.push({
        gl: payload.gl,
        texture: payload.texture
      });
      pooledCount++;
    } else if (payload?.texture && payload?.gl && typeof payload.gl.deleteTexture === 'function') {
      payload.gl.deleteTexture(payload.texture);
    }
    releasedCount++;
  }

  context.gpuResidentPackedPayloads = retainedPayloads;
  context.activePayloadCount = countGpuResidentPackedPayloads(context);
  context.reusablePayloadCount = countReusableGpuResidentPayloadTextures(context);
  context.lastPayloadPoolReleaseCount = (context.lastPayloadPoolReleaseCount ?? 0) + pooledCount;
  trimReusableGpuResidentPayloadTextures(context);
  return releasedCount;
}

export function resetGpuScreenTransformBackendGpuPayloads(context, reason = 'manual-reset') {
  if (!context) return 0;
  const adaptivePolicy = resolveAdaptiveReusablePayloadTexturePolicy(context);
  context.payloadPoolMaxRetained = adaptivePolicy.nextLimit;
  context.lastPayloadPoolPolicy = adaptivePolicy.reason;
  context.lastPayloadPoolReleaseCount = 0;
  context.lastPayloadReuseCount = 0;
  context.lastPayloadCreateCount = 0;
  context.lastPayloadTrimCount = 0;
  context.lastPayloadRetainedCount = 0;
  context.atlasGpuPackedPayload = null;
  context.lastAtlasPayloadBuilt = false;
  context.lastAtlasPayloadBatchCount = 0;
  context.lastAtlasPayloadCopyCount = 0;
  context.lastAtlasPayloadPolicySelectedPath = 'none';
  context.lastAtlasPayloadPolicyReason = 'none';
  context.lastAtlasPayloadReused = false;
  context.lastAtlasPayloadRebuilt = false;
  context.lastAtlasPayloadChurnReason = 'none';
  context.lastAtlasPayloadCapacityWidth = 0;
  context.lastAtlasPayloadCapacityHeight = 0;
  context.lastAtlasPayloadAllocationBytes = 0;
  context.lastAtlasPayloadSavedAllocationBytes = 0;
  context.lastAtlasPayloadWidth = 0;
  context.lastAtlasPayloadHeight = 0;
  const releasedCount = releaseGpuResidentPackedPayloads(context);
  context.lastReleasedPayloadCount = releasedCount;
  context.lastPayloadResetReason = reason;
  context.payloadGeneration = Number.isFinite(context.payloadGeneration)
    ? (context.payloadGeneration + 1)
    : 1;
  context.lastPayloadOwner = releasedCount > 0
    ? 'backend-gpu-pool'
    : (context.lastPayloadOwner ?? 'backend-gpu-resident');
  context.lastDispatchCount = 0;
  context.lastDispatchMode = 'none';
  context.lastDispatchUploadBytes = 0;
  context.lastDispatchItemCount = 0;
  return releasedCount;
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

function buildPackedSourceUpload(items) {
  const normalizedItems = Array.isArray(items) ? items : [];
  const upload = new Float32Array(normalizedItems.length * GPU_VISIBLE_PACK_FLOATS_PER_ITEM);
  let offset = 0;

  for (let i = 0; i < normalizedItems.length; i++) {
    const rows = buildPackedRowsFromSourceItem(normalizedItems[i]);
    for (let rowIndex = 0; rowIndex < rows.length; rowIndex++) {
      upload.set(rows[rowIndex], offset);
      offset += 4;
    }
  }

  return upload;
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

function hasReusablePackedWriteResources(context, gl) {
  return (
    isSameWebGlContext(context?.packedWriteResourceGl, gl) &&
    !!context?.packedWriteResources?.program &&
    !!context?.packedWriteResources?.framebuffer &&
    !!context?.packedWriteResources?.texture &&
    !!context?.packedWriteResources?.sourceTexture &&
    !!context?.packedWriteResources?.uniformSourceTexture &&
    !!context?.packedWriteResources?.vao
  );
}

function acquireReusableGpuResidentPackedPayloadTexture(context, gl) {
  const pool = Array.isArray(context?.reusableGpuResidentPayloadTextures)
    ? context.reusableGpuResidentPayloadTextures
    : null;

  if (pool) {
    for (let i = pool.length - 1; i >= 0; i--) {
      const entry = pool[i];
      if (!entry?.texture || !isSameWebGlContext(entry?.gl, gl)) continue;
      pool.splice(i, 1);
      if (context) {
        context.reusablePayloadCount = pool.length;
        context.lastPayloadReuseCount = (context.lastPayloadReuseCount ?? 0) + 1;
      }
      return entry.texture;
    }
  }

  const texture = gl.createTexture();
  if (texture && context) {
    context.lastPayloadCreateCount = (context.lastPayloadCreateCount ?? 0) + 1;
  }
  return texture;
}

function cloneGpuResidentPackedPayload(context, gl, count) {
  const resources = context?.packedWriteResources;
  if (!resources?.texture || !resources?.framebuffer) {
    throw new Error('missing-packed-write-resources-for-gpu-payload-clone');
  }

  const texture = acquireReusableGpuResidentPackedPayloadTexture(context, gl);
  if (!texture) {
    throw new Error('failed-to-create-gpu-packed-payload-texture');
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA32F,
    GPU_PACKED_PAYLOAD_WIDTH,
    count,
    0,
    gl.RGBA,
    gl.FLOAT,
    null
  );

  gl.bindFramebuffer(gl.READ_FRAMEBUFFER, resources.framebuffer);
  gl.copyTexSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 0, 0, GPU_PACKED_PAYLOAD_WIDTH, count);
  gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);

  const payload = createGpuPackedPayloadRecord(gl, texture, count);
  if (context) {
    context.gpuResidentPackedPayloads = context.gpuResidentPackedPayloads ?? [];
    context.gpuResidentPackedPayloads.push(payload);
    context.activePayloadCount = context.gpuResidentPackedPayloads.length;
    context.reusablePayloadCount = countReusableGpuResidentPayloadTextures(context);
    context.payloadPoolHighWaterCount = Math.max(
      Number.isFinite(context.payloadPoolHighWaterCount) ? context.payloadPoolHighWaterCount : 0,
      context.reusablePayloadCount
    );
    context.lastPayloadOwner = 'backend-gpu-resident';
  }
  return payload;
}

function ensureBackendAtlasResources(context, gl) {
  return ensureGpuPackedPayloadTextureDrawResources(gl, context, 'backendPackedAtlasResources');
}

function consolidateGpuResidentPackedPayloadAtlas(context, gl, payloads, options = {}) {
  const validPayloads = Array.isArray(payloads)
    ? payloads.filter((payload) => payload?.texture && payload?.gl === gl)
    : [];
  if (!context || validPayloads.length <= 1) {
    return {
      atlasPayload: validPayloads.length === 1 ? validPayloads[0] : null,
      usedBackendAtlas: false,
      avoidedDrawTimeMerge: false
    };
  }

  const resources = ensureBackendAtlasResources(context, gl);
  const atlasResult = buildGpuPackedPayloadAtlas(gl, context, validPayloads, {
    resources,
    storageKey: 'backendPackedAtlasResources',
    policyOverride: options.drawPolicyOverride ?? null
  });

  context.lastAtlasPayloadBuilt = !!atlasResult.atlasPayload?.texture;
  context.lastAtlasPayloadBatchCount = validPayloads.length;
  context.lastAtlasPayloadCopyCount = atlasResult.mergeCopyCount ?? 0;
  context.lastAtlasPayloadPolicySelectedPath = atlasResult.mergePolicySelectedPath ?? 'none';
  context.lastAtlasPayloadPolicyReason = atlasResult.mergePolicyReason ?? 'none';
  context.lastAtlasPayloadReused = !!atlasResult.mergeAtlasReused;
  context.lastAtlasPayloadRebuilt = !!atlasResult.mergeAtlasRebuilt;
  context.lastAtlasPayloadChurnReason = atlasResult.mergeAtlasChurnReason ?? 'none';
  context.lastAtlasPayloadCapacityWidth = atlasResult.mergeAtlasCapacityWidth ?? 0;
  context.lastAtlasPayloadCapacityHeight = atlasResult.mergeAtlasCapacityHeight ?? 0;
  context.lastAtlasPayloadAllocationBytes = atlasResult.mergeAtlasAllocationBytes ?? 0;
  context.lastAtlasPayloadSavedAllocationBytes = atlasResult.mergeAtlasSavedAllocationBytes ?? 0;
  context.lastAtlasPayloadWidth = atlasResult.mergeTextureWidth ?? 0;
  context.lastAtlasPayloadHeight = atlasResult.mergeTextureHeight ?? 0;

  if (!atlasResult.atlasPayload?.texture) {
    context.atlasGpuPackedPayload = null;
    return {
      atlasPayload: null,
      usedBackendAtlas: false,
      avoidedDrawTimeMerge: false
    };
  }

  const atlasPayload = {
    ...atlasResult.atlasPayload,
    atlasReady: true,
    owner: 'backend-transform-atlas',
    mergePolicySelectedPath: atlasResult.mergePolicySelectedPath ?? 'none',
    mergePolicyReason: atlasResult.mergePolicyReason ?? 'none',
    mergeCopyCount: atlasResult.mergeCopyCount ?? 0,
    atlasReused: !!atlasResult.mergeAtlasReused,
    atlasRebuilt: !!atlasResult.mergeAtlasRebuilt,
    atlasChurnReason: atlasResult.mergeAtlasChurnReason ?? 'none',
    atlasAllocationBytes: atlasResult.mergeAtlasAllocationBytes ?? 0,
    atlasSavedAllocationBytes: atlasResult.mergeAtlasSavedAllocationBytes ?? 0
  };

  context.atlasGpuPackedPayload = atlasPayload;
  releaseSpecificGpuResidentPackedPayloads(context, validPayloads);
  context.activePayloadCount = countGpuResidentPackedPayloads(context);
  context.lastPayloadOwner = 'backend-transform-atlas';

  return {
    atlasPayload,
    usedBackendAtlas: true,
    avoidedDrawTimeMerge: true
  };
}

function ensureGpuPackedWriteResources(context, gl, targetHeight) {
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
  const normalizedTargetHeight = Math.max(1, Math.min(maxSupportedBatchItems, targetHeight));

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
uniform sampler2D uPackedSourceTexture;
out vec4 outColor;
void main() {
  ivec2 coord = ivec2(int(floor(gl_FragCoord.x)), int(floor(gl_FragCoord.y)));
  outColor = texelFetch(uPackedSourceTexture, coord, 0);
}`;

  try {
    if (hasReusablePackedWriteResources(context, gl)) {
      gl.bindTexture(gl.TEXTURE_2D, context.packedWriteResources.texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 4, normalizedTargetHeight, 0, gl.RGBA, gl.FLOAT, null);
      gl.bindTexture(gl.TEXTURE_2D, context.packedWriteResources.sourceTexture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 4, normalizedTargetHeight, 0, gl.RGBA, gl.FLOAT, null);
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
    const sourceTexture = gl.createTexture();
    const framebuffer = gl.createFramebuffer();

    if (!vertexShader || !fragmentShader || !program || !vao || !texture || !sourceTexture || !framebuffer) {
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
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 4, normalizedTargetHeight, 0, gl.RGBA, gl.FLOAT, null);

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
      gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 4, normalizedTargetHeight, 0, gl.RGBA, gl.FLOAT, null);
      gl.bindTexture(gl.TEXTURE_2D, null);

      context.packedWriteResources = {
        program,
        vao,
        texture,
        sourceTexture,
        framebuffer,
        uniformSourceTexture: gl.getUniformLocation(program, 'uPackedSourceTexture')
      };
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

function tryGenerateGpuPackedSmallBatch(context, sourceItemsResult, gl, options = {}) {
  const items = normalizeSourceItems(sourceItemsResult);
  const maxSupportedBatchItems = getMaxSupportedBatchItems(gl);
  if (items.length <= 0 || items.length > maxSupportedBatchItems) {
    return {
      producedPacked: false,
      packed: null,
      gpuPackedPayload: null,
      count: 0,
      floatsPerItem: GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
      stage: 'gpu-pack-small-batch-size-unsupported',
      reason: GPU_BACKEND_REASON_CPU_FALLBACK,
      error: null
    };
  }

  const resourceState = ensureGpuPackedWriteResources(context, gl, items.length);
  if (!resourceState.ok) {
    return {
      producedPacked: false,
      packed: null,
      gpuPackedPayload: null,
      count: 0,
      floatsPerItem: GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
      stage: resourceState.stage,
      reason: resourceState.reason,
      error: resourceState.error
    };
  }

  const resources = context?.packedWriteResources;
  try {
    const packedSourceUpload = buildPackedSourceUpload(items);

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
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, resources.sourceTexture);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      GPU_PACKED_PAYLOAD_WIDTH,
      items.length,
      gl.RGBA,
      gl.FLOAT,
      packedSourceUpload
    );
    gl.uniform1i(resources.uniformSourceTexture, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    const packed = null;
    const gpuPackedPayload = cloneGpuResidentPackedPayload(context, gl, items.length);
    context.lastDispatchCount = 1;
    context.lastDispatchMode = 'single-texture-copy-pass';
    context.lastDispatchUploadBytes = packedSourceUpload.byteLength;
    context.lastDispatchItemCount = items.length;
    context.lastSuccessfulDispatchMode = 'single-texture-copy-pass';
    context.lastSuccessfulDispatchItemCount = items.length;

    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(null);

    return {
      producedPacked: true,
      packed,
      gpuPackedPayload,
      count: items.length,
      floatsPerItem: GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
      stage: 'gpu-pack-small-batch-generated',
      reason: 'gpu-packed-generated',
      error: null
    };
  } catch (error) {
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(null);
    return {
      producedPacked: false,
      packed: null,
      gpuPackedPayload: null,
      count: 0,
      floatsPerItem: GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
      stage: 'gpu-pack-small-batch-failed',
      reason: GPU_BACKEND_REASON_TRIAL_FAILED,
      error: buildBackendError(error)
    };
  }
}

function supportsGpuPackedWriteBackend(gl) {
  return probeWebGL2Support(gl) && probeFloatColorSupport(gl);
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
    activePayloadCount: 0,
    reusablePayloadCount: 0,
    lastReleasedPayloadCount: 0,
    lastPayloadPoolReleaseCount: 0,
    lastPayloadReuseCount: 0,
    lastPayloadCreateCount: 0,
    lastPayloadTrimCount: 0,
    lastPayloadRetainedCount: 0,
    payloadPoolHighWaterCount: 0,
    payloadPoolBaseRetained: GPU_BACKEND_BASE_REUSABLE_PAYLOAD_TEXTURES,
    payloadPoolHardMaxRetained: GPU_BACKEND_HARD_MAX_REUSABLE_PAYLOAD_TEXTURES,
    payloadPoolMaxRetained: GPU_BACKEND_BASE_REUSABLE_PAYLOAD_TEXTURES,
    lastPayloadPoolPolicy: 'adaptive-hold-base',
    lastPayloadResetReason: 'none',
    payloadGeneration: 0,
    lastPayloadOwner: 'backend-gpu-resident',
    lastDispatchCount: 0,
    lastDispatchMode: 'none',
    lastDispatchUploadBytes: 0,
    lastDispatchItemCount: 0,
    lastSuccessfulDispatchMode: 'none',
    lastSuccessfulDispatchItemCount: 0,
    lastAtlasPayloadBuilt: false,
    lastAtlasPayloadBatchCount: 0,
    lastAtlasPayloadCopyCount: 0,
    lastAtlasPayloadPolicySelectedPath: 'none',
    lastAtlasPayloadPolicyReason: 'none',
    lastAtlasPayloadReused: false,
    lastAtlasPayloadRebuilt: false,
    lastAtlasPayloadChurnReason: 'none',
    lastAtlasPayloadCapacityWidth: 0,
    lastAtlasPayloadCapacityHeight: 0,
    lastAtlasPayloadAllocationBytes: 0,
    lastAtlasPayloadSavedAllocationBytes: 0,
    lastAtlasPayloadWidth: 0,
    lastAtlasPayloadHeight: 0,
    packedWriteResources: null,
    packedWriteResourceGl: null,
    gpuResidentPackedPayloads: [],
    atlasGpuPackedPayload: null,
    reusableGpuResidentPayloadTextures: [],
    ...initialState
  };
}

export function summarizeGpuScreenTransformBackendGpuCapability(context, gl = null) {
  const supportsWebGL2 = probeWebGL2Support(gl);
  const isReady = !!context?.isReady && supportsWebGL2;
  const maxTextureSize = getMaxTextureSize(gl);
  const maxBatchItems = supportsWebGL2 ? getMaxSupportedBatchItems(gl) : 0;
  const preferredBatchItems = supportsWebGL2 ? resolvePreferredBatchItems(context, maxBatchItems) : 0;
  const preferredBatchPolicy = supportsWebGL2
    ? resolvePreferredBatchPolicy(context, preferredBatchItems, maxBatchItems)
    : 'preferred-batch-none';

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
    preferredBatchItems,
    preferredBatchPolicy,
    lastProbeMs: Number.isFinite(context?.lastProbeMs) ? context.lastProbeMs : 0,
    lastExecutionMs: Number.isFinite(context?.lastExecutionMs) ? context.lastExecutionMs : 0,
    lastSourceItemCount: Number.isFinite(context?.lastSourceItemCount) ? context.lastSourceItemCount : 0,
    lastCount: Number.isFinite(context?.lastCount) ? context.lastCount : 0,
    lastStageName: context?.lastStageName ?? 'idle',
    lastError: context?.lastError ?? null,
    activePayloadCount: Number.isFinite(context?.activePayloadCount)
      ? context.activePayloadCount
      : countGpuResidentPackedPayloads(context),
    reusablePayloadCount: Number.isFinite(context?.reusablePayloadCount)
      ? context.reusablePayloadCount
      : countReusableGpuResidentPayloadTextures(context),
    lastReleasedPayloadCount: Number.isFinite(context?.lastReleasedPayloadCount)
      ? context.lastReleasedPayloadCount
      : 0,
    lastPayloadPoolReleaseCount: Number.isFinite(context?.lastPayloadPoolReleaseCount)
      ? context.lastPayloadPoolReleaseCount
      : 0,
    lastPayloadReuseCount: Number.isFinite(context?.lastPayloadReuseCount)
      ? context.lastPayloadReuseCount
      : 0,
    lastPayloadCreateCount: Number.isFinite(context?.lastPayloadCreateCount)
      ? context.lastPayloadCreateCount
      : 0,
    lastPayloadTrimCount: Number.isFinite(context?.lastPayloadTrimCount)
      ? context.lastPayloadTrimCount
      : 0,
    lastPayloadRetainedCount: Number.isFinite(context?.lastPayloadRetainedCount)
      ? context.lastPayloadRetainedCount
      : countReusableGpuResidentPayloadTextures(context),
    payloadPoolHighWaterCount: Number.isFinite(context?.payloadPoolHighWaterCount)
      ? context.payloadPoolHighWaterCount
      : 0,
    payloadPoolBaseRetained: getReusableGpuResidentPayloadTextureBaseLimit(context),
    payloadPoolHardMaxRetained: getReusableGpuResidentPayloadTextureHardLimit(context),
    payloadPoolMaxRetained: getReusableGpuResidentPayloadTextureLimit(context),
    lastPayloadPoolPolicy: context?.lastPayloadPoolPolicy ?? 'adaptive-hold-base',
    lastPayloadResetReason: context?.lastPayloadResetReason ?? 'none',
    payloadGeneration: Number.isFinite(context?.payloadGeneration) ? context.payloadGeneration : 0,
    lastPayloadOwner: context?.lastPayloadOwner ?? 'backend-gpu-resident',
    lastDispatchCount: Number.isFinite(context?.lastDispatchCount) ? context.lastDispatchCount : 0,
    lastDispatchMode: context?.lastDispatchMode ?? 'none',
    lastDispatchUploadBytes: Number.isFinite(context?.lastDispatchUploadBytes)
      ? context.lastDispatchUploadBytes
      : 0,
    lastDispatchItemCount: Number.isFinite(context?.lastDispatchItemCount)
      ? context.lastDispatchItemCount
      : 0,
    lastSuccessfulDispatchMode: context?.lastSuccessfulDispatchMode ?? 'none',
    lastSuccessfulDispatchItemCount: Number.isFinite(context?.lastSuccessfulDispatchItemCount)
      ? context.lastSuccessfulDispatchItemCount
      : 0,
    lastAtlasPayloadBuilt: !!context?.lastAtlasPayloadBuilt,
    lastAtlasPayloadBatchCount: Number.isFinite(context?.lastAtlasPayloadBatchCount)
      ? context.lastAtlasPayloadBatchCount
      : 0,
    lastAtlasPayloadCopyCount: Number.isFinite(context?.lastAtlasPayloadCopyCount)
      ? context.lastAtlasPayloadCopyCount
      : 0,
    lastAtlasPayloadPolicySelectedPath: context?.lastAtlasPayloadPolicySelectedPath ?? 'none',
    lastAtlasPayloadPolicyReason: context?.lastAtlasPayloadPolicyReason ?? 'none',
    lastAtlasPayloadReused: !!context?.lastAtlasPayloadReused,
    lastAtlasPayloadRebuilt: !!context?.lastAtlasPayloadRebuilt,
    lastAtlasPayloadChurnReason: context?.lastAtlasPayloadChurnReason ?? 'none',
    lastAtlasPayloadCapacityWidth: Number.isFinite(context?.lastAtlasPayloadCapacityWidth)
      ? context.lastAtlasPayloadCapacityWidth
      : 0,
    lastAtlasPayloadCapacityHeight: Number.isFinite(context?.lastAtlasPayloadCapacityHeight)
      ? context.lastAtlasPayloadCapacityHeight
      : 0,
    lastAtlasPayloadAllocationBytes: Number.isFinite(context?.lastAtlasPayloadAllocationBytes)
      ? context.lastAtlasPayloadAllocationBytes
      : 0,
    lastAtlasPayloadSavedAllocationBytes: Number.isFinite(context?.lastAtlasPayloadSavedAllocationBytes)
      ? context.lastAtlasPayloadSavedAllocationBytes
      : 0,
    lastAtlasPayloadWidth: Number.isFinite(context?.lastAtlasPayloadWidth)
      ? context.lastAtlasPayloadWidth
      : 0,
    lastAtlasPayloadHeight: Number.isFinite(context?.lastAtlasPayloadHeight)
      ? context.lastAtlasPayloadHeight
      : 0
  };
}

export function finalizeGpuScreenTransformBackendGpuAtlas(context, payloads, options = {}) {
  const gl = options?.gl ?? null;
  if (!probeWebGL2Support(gl)) {
    return {
      atlasPayload: null,
      usedBackendAtlas: false,
      avoidedDrawTimeMerge: false
    };
  }
  return consolidateGpuResidentPackedPayloadAtlas(context, gl, payloads, options);
}

export function executeGpuScreenTransformBackendGpu(context, sourceItemsResult, options = {}) {
  const t0 = nowMs();
  const sourceItemCount = normalizeSourceCount(sourceItemsResult);
  const gl = options?.gl ?? null;
  const supportsWebGL2 = probeWebGL2Support(gl);
  const backendImplemented = supportsGpuPackedWriteBackend(gl);
  if (options?.resetGpuResidentPayloads) {
    resetGpuScreenTransformBackendGpuPayloads(context, options.resetGpuResidentPayloadsReason ?? 'batch-reset');
  }
  const cpuFallback = runCpuFallbackPack(sourceItemsResult);
  const gpuPackedResult = tryGenerateGpuPackedSmallBatch(context, sourceItemsResult, gl, options);
  const producedPacked = !!gpuPackedResult.producedPacked;
  const ready = producedPacked;
  const implementationState = backendImplemented
    ? GPU_BACKEND_IMPLEMENTATION_STATE_MINIMAL_GPU_PACK
    : GPU_BACKEND_IMPLEMENTATION_STATE_STUB;
  const executionMode = producedPacked
    ? GPU_BACKEND_EXECUTION_MODE_GPU_SMALL_BATCH_PACK
    : GPU_BACKEND_EXECUTION_MODE_CPU_FALLBACK;
  const reason = producedPacked
    ? gpuPackedResult.reason
    : (gpuPackedResult.reason ?? (supportsWebGL2 ? GPU_BACKEND_REASON_CPU_FALLBACK : GPU_BACKEND_REASON_MISSING_WEBGL2));
  const stageMs = nowMs() - t0;
  const finalPacked = producedPacked ? null : cpuFallback.packed;
  const finalCount = producedPacked ? gpuPackedResult.count : cpuFallback.count;
  const finalFloatsPerItem = producedPacked ? gpuPackedResult.floatsPerItem : cpuFallback.floatsPerItem;

  if (context) {
    context.implementationState = implementationState;
    context.executionMode = executionMode;
    context.isProbed = true;
    context.supportsWebGL2 = supportsWebGL2;
    context.isAvailable = supportsWebGL2;
    context.isReady = ready;
    context.isImplemented = backendImplemented;
    context.supportsTransformPath = backendImplemented;
    context.reason = reason;
    context.lastProbeMs = stageMs;
    context.lastExecutionMs = stageMs;
    context.lastSourceItemCount = sourceItemCount;
    context.lastCount = finalCount;
    context.lastStageName = gpuPackedResult.stage;
    context.lastError = producedPacked ? null : gpuPackedResult.error;
    context.activePayloadCount = countGpuResidentPackedPayloads(context);
    context.reusablePayloadCount = countReusableGpuResidentPayloadTextures(context);
    if (producedPacked) {
      context.lastPayloadOwner = 'backend-gpu-resident';
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
    gpuPackedPayload: producedPacked ? (gpuPackedResult.gpuPackedPayload ?? null) : null,
    count: finalCount,
    floatsPerItem: finalFloatsPerItem,
    sourceItemCount,
    backendStage: gpuPackedResult.stage,
    backendFallback: !producedPacked,
    backendImplemented,
    backendProducedPacked: producedPacked,
    backendFallbackToCpu: !producedPacked,
    backendError: producedPacked ? null : gpuPackedResult.error,
    backendContext: summarizeGpuScreenTransformBackendGpuCapability(context, gl),
    backendDispatchCount: Number.isFinite(context?.lastDispatchCount) ? context.lastDispatchCount : 0,
    backendDispatchMode: context?.lastDispatchMode ?? 'none',
    backendDispatchUploadBytes: Number.isFinite(context?.lastDispatchUploadBytes)
      ? context.lastDispatchUploadBytes
      : 0,
    backendDispatchItemCount: Number.isFinite(context?.lastDispatchItemCount)
      ? context.lastDispatchItemCount
      : 0
  };
}
