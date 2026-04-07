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

const GPU_BACKEND_ID = 'gpu-transform-backend';
const GPU_BACKEND_REASON_NOT_IMPLEMENTED = 'gpu-backend-not-implemented';
const GPU_BACKEND_IMPLEMENTATION_STATE_STUB = 'stub';
const GPU_BACKEND_EXECUTION_MODE_CPU_FALLBACK = 'cpu-fallback-stub';

function nowMs() {
  return performance.now();
}

function probeWebGL2Support(gl) {
  return !!gl && typeof gl.createVertexArray === 'function';
}

function normalizeSourceCount(sourceItemsResult) {
  if (Number.isFinite(sourceItemsResult?.itemCount)) return sourceItemsResult.itemCount;
  if (Array.isArray(sourceItemsResult?.items)) return sourceItemsResult.items.length;
  return 0;
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
  const ready = !!context?.isReady && supportsWebGL2;

  if (context) {
    context.implementationState = GPU_BACKEND_IMPLEMENTATION_STATE_STUB;
    context.executionMode = GPU_BACKEND_EXECUTION_MODE_CPU_FALLBACK;
    context.isProbed = true;
    context.supportsWebGL2 = supportsWebGL2;
    context.isAvailable = supportsWebGL2;
    context.isReady = ready;
    context.isImplemented = false;
    context.reason = ready ? 'ready' : GPU_BACKEND_REASON_NOT_IMPLEMENTED;
    context.lastProbeMs = nowMs() - t0;
    context.lastExecutionMs = 0;
    context.lastSourceItemCount = sourceItemCount;
    context.lastCount = 0;
    context.lastStageName = 'probe-only';
    context.lastError = null;
  }

  return {
    backendId: GPU_BACKEND_ID,
    implementationState: GPU_BACKEND_IMPLEMENTATION_STATE_STUB,
    executionMode: GPU_BACKEND_EXECUTION_MODE_CPU_FALLBACK,
    ready,
    reason: ready ? 'ready' : GPU_BACKEND_REASON_NOT_IMPLEMENTED,
    stageMs: 0,
    packed: null,
    count: 0,
    floatsPerItem: 0,
    sourceItemCount,
    backendStage: 'stub',
    backendFallback: true,
    backendImplemented: false,
    backendFallbackToCpu: true,
    backendError: null,
    backendContext: summarizeGpuScreenTransformBackendGpuCapability(context, gl)
  };
}
