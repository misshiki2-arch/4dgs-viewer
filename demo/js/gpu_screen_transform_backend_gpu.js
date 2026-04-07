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
const GPU_BACKEND_REASON_TRIAL_FAILED = 'gpu-backend-trial-failed';
const GPU_BACKEND_REASON_CPU_FALLBACK = 'gpu-backend-cpu-fallback';
const GPU_BACKEND_IMPLEMENTATION_STATE_STUB = 'stub';
const GPU_BACKEND_EXECUTION_MODE_CPU_FALLBACK = 'cpu-fallback-stub';
const GPU_BACKEND_EXECUTION_MODE_GPU_TRIAL_CPU_FALLBACK = 'gpu-trial-cpu-fallback';

function nowMs() {
  return performance.now();
}

function probeWebGL2Support(gl) {
  return !!gl && typeof gl.createVertexArray === 'function';
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

function runCpuFallbackPack(sourceItemsResult) {
  const items = Array.isArray(sourceItemsResult?.items) ? sourceItemsResult.items : [];
  const packedResult = packVisibleItems(items);
  return {
    packed: packedResult?.packed instanceof Float32Array ? packedResult.packed : null,
    count: Number.isFinite(packedResult?.count) ? packedResult.count : items.length,
    floatsPerItem: Number.isFinite(packedResult?.floatsPerItem)
      ? packedResult.floatsPerItem
      : GPU_VISIBLE_PACK_FLOATS_PER_ITEM
  };
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

  if (context?.trialResources?.buffer && context?.trialResources?.vao) {
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
    ...initialState
  };
}

export function summarizeGpuScreenTransformBackendGpuCapability(context, gl = null) {
  const supportsWebGL2 = probeWebGL2Support(gl);
  const isReady = !!context?.isReady && supportsWebGL2;

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
  const ready = !!trialResult.triedGpu && trialResult.error === null;
  const implementationState = ready ? 'trial-only' : GPU_BACKEND_IMPLEMENTATION_STATE_STUB;
  const executionMode = trialResult.triedGpu
    ? GPU_BACKEND_EXECUTION_MODE_GPU_TRIAL_CPU_FALLBACK
    : GPU_BACKEND_EXECUTION_MODE_CPU_FALLBACK;
  const reason = trialResult.reason ?? (supportsWebGL2 ? GPU_BACKEND_REASON_CPU_FALLBACK : GPU_BACKEND_REASON_MISSING_WEBGL2);
  const stageMs = nowMs() - t0;

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
    context.lastCount = cpuFallback.count;
    context.lastStageName = trialResult.stage;
    context.lastError = trialResult.error;
  }

  return {
    backendId: GPU_BACKEND_ID,
    implementationState,
    executionMode,
    ready,
    reason,
    stageMs,
    packed: cpuFallback.packed,
    count: cpuFallback.count,
    floatsPerItem: cpuFallback.floatsPerItem,
    sourceItemCount,
    backendStage: trialResult.stage,
    backendFallback: true,
    backendImplemented: !!trialResult.triedGpu,
    backendProducedPacked: false,
    backendFallbackToCpu: true,
    backendError: trialResult.error,
    backendContext: summarizeGpuScreenTransformBackendGpuCapability(context, gl)
  };
}
