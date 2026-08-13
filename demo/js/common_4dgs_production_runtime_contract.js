export const PRODUCTION_RUNTIME_CONTRACT_SCHEMA_VERSION =
  'phase3-production-runtime-selection-contract-v1';

export const PRODUCTION_WEBGPU_RUNTIME = 'webgpu';
export const PRODUCTION_WEBGPU_EFFECTIVE_DISPLAY_RUNTIME = 'webgpu-production';
export const PRODUCTION_WEBGPU_BACKEND_MODE = 'webgpu-exclusive';
export const PRODUCTION_WEBGPU_TILE_COMPOSITOR_IMPLEMENTATION =
  'webgpu-tile-compositor-frame-implementation';
export const PRODUCTION_PRESENTATION_OWNER_ROLE =
  'production-presentation-owner';
export const PRODUCTION_DIAGNOSTIC_OBSERVER_ROLE =
  'production-diagnostic-observer';

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function normalizeBoolean(value) {
  return value === true ? true : value === false ? false : null;
}

export function buildProductionPresentationMutationPolicy({
  executionRole = PRODUCTION_PRESENTATION_OWNER_ROLE
} = {}) {
  const productionPresentationOwner =
    executionRole === PRODUCTION_PRESENTATION_OWNER_ROLE;
  return {
    schemaVersion: 'phase3-production-presentation-mutation-policy-v1',
    source: 'common-production-runtime-selection-contract',
    executionRole,
    diagnosticComputationAllowed: true,
    diagnosticReadbackAllowed: true,
    liveProductionCanvasMutationAllowed: productionPresentationOwner,
    productionOutputMutationAllowed: productionPresentationOwner,
    lastValidProductionCacheMutationAllowed: productionPresentationOwner,
    productionRequestMutationAllowed: productionPresentationOwner,
    productionGenerationMutationAllowed: productionPresentationOwner,
    compositorGenerationMutationAllowed: productionPresentationOwner,
    presentedGenerationMutationAllowed: productionPresentationOwner,
    finalProductionWriterMutationAllowed: productionPresentationOwner
  };
}

export function canMutateProductionPresentationState(policy = null) {
  if (!policy) return true;
  return (
    policy.executionRole === PRODUCTION_PRESENTATION_OWNER_ROLE &&
    policy.liveProductionCanvasMutationAllowed === true &&
    policy.productionOutputMutationAllowed === true &&
    policy.lastValidProductionCacheMutationAllowed === true &&
    policy.productionRequestMutationAllowed === true &&
    policy.productionGenerationMutationAllowed === true &&
    policy.compositorGenerationMutationAllowed === true &&
    policy.presentedGenerationMutationAllowed === true &&
    policy.finalProductionWriterMutationAllowed === true
  );
}

export function buildProductionRuntimeSelectionContract({
  queryState = {},
  latestFinalCanvasEvent = null,
  snapshotTakenAtMs = nowMs()
} = {}) {
  const requestedRuntime = queryState.viewerRuntime ?? null;
  const backendMode = queryState.webgpuBackendMode ?? null;
  const backendImplementation = queryState.webgpuBackendImplementation ?? null;
  const canvasPresentationEnabled = normalizeBoolean(
    queryState.webgpuAllowViewerCanvasPresentation
  );
  const viewerLoopHookEnabled = normalizeBoolean(
    queryState.webgpuBackendViewerLoopHook
  );
  const productionSelectionReady =
    requestedRuntime === PRODUCTION_WEBGPU_RUNTIME &&
    backendMode === PRODUCTION_WEBGPU_BACKEND_MODE &&
    backendImplementation === PRODUCTION_WEBGPU_TILE_COMPOSITOR_IMPLEMENTATION &&
    canvasPresentationEnabled === true &&
    viewerLoopHookEnabled === true;

  return {
    schemaVersion: PRODUCTION_RUNTIME_CONTRACT_SCHEMA_VERSION,
    source: 'common-production-runtime-selection-contract',
    snapshotTakenAtMs,
    readOnlySnapshot: true,
    runtimeEvidenceCurrent: true,
    requestedRuntime,
    effectiveDisplayRuntime: productionSelectionReady
      ? PRODUCTION_WEBGPU_EFFECTIVE_DISPLAY_RUNTIME
      : requestedRuntime === 'webgpu'
        ? 'webgpu-not-production-ready'
        : requestedRuntime ?? 'unspecified',
    backendMode,
    backendImplementation,
    canvasPresentationEnabled,
    viewerLoopHookEnabled,
    productionSelectionReady,
    actualProductionPresentationPath:
      latestFinalCanvasEvent?.presentationPathIdentity ?? null,
    actualPresentationSource:
      latestFinalCanvasEvent?.presentationSource ?? null,
    actualPresentationEventIdentity:
      latestFinalCanvasEvent?.eventIdentity ?? null,
    actualPresentationSourceRequestIdentity:
      latestFinalCanvasEvent?.sourceRequestIdentity ?? null,
    actualPresentedGeneration:
      Number.isFinite(Number(latestFinalCanvasEvent?.presentedGeneration))
        ? Number(latestFinalCanvasEvent.presentedGeneration)
        : null,
    gpuCandidateRuntime: queryState.gpuCandidateRuntime ?? null,
    gpuCandidateRuntimeIsProductionDisplayRuntime: false
  };
}

export function validateExpectedProductionRuntimeContract(actual, expected = {}) {
  const required = {
    requestedRuntime: expected.requestedRuntime ?? null,
    effectiveDisplayRuntime: expected.effectiveDisplayRuntime ?? null,
    backendMode: expected.backendMode ?? null,
    backendImplementation: expected.backendImplementation ?? null,
    canvasPresentationEnabled: normalizeBoolean(
      expected.canvasPresentationEnabled
    ),
    viewerLoopHookEnabled: normalizeBoolean(expected.viewerLoopHookEnabled)
  };
  const mismatches = [];
  if (actual?.schemaVersion !== PRODUCTION_RUNTIME_CONTRACT_SCHEMA_VERSION) {
    mismatches.push('runtime-contract-schema');
  }
  if (actual?.readOnlySnapshot !== true || actual?.runtimeEvidenceCurrent !== true) {
    mismatches.push('runtime-contract-current-read-only-evidence');
  }
  for (const [field, expectedValue] of Object.entries(required)) {
    if (expectedValue !== null && actual?.[field] !== expectedValue) {
      mismatches.push(field);
    }
  }
  return {
    schemaVersion: 'phase3-expected-production-runtime-validation-v1',
    source: 'common-production-runtime-selection-contract',
    expected: required,
    actual: actual ?? null,
    mismatchFields: mismatches,
    ready: mismatches.length === 0,
    blockedReason: mismatches.length > 0
      ? 'production-runtime-selection-mismatch'
      : null
  };
}
