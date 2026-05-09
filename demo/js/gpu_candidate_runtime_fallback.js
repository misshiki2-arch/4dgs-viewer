function addReason(reasons, code, message, details = null) {
  reasons.push({ code, message, details });
}

function getPackedMismatch(visibleComparison) {
  return !!(
    visibleComparison?.packedPayload?.anyMismatch ??
    visibleComparison?.packedPayload?.packedAnyMismatch
  );
}

export function buildGpuCandidateRuntimeFallbackSummary({
  runtimeConfig = {},
  shadowCompare = null,
  error = null,
  extraReasons = []
} = {}) {
  const reasons = [];
  const requestedRuntime = runtimeConfig.requestedRuntime ?? 'cpu-reference';
  const shadowStatus = shadowCompare?.status ?? null;
  const candidateMismatch = !!shadowCompare?.candidateComparison?.anyMismatch;
  const visibleMismatch = !!shadowCompare?.visibleComparison?.visibleItems?.anyMismatch;
  const packedMismatch = getPackedMismatch(shadowCompare?.visibleComparison);
  const explicitAnyMismatch = shadowCompare?.summary?.anyMismatch ??
    shadowCompare?.shadowCompare?.anyMismatch;
  const shadowAnyMismatch = typeof explicitAnyMismatch === 'boolean'
    ? explicitAnyMismatch
    : (candidateMismatch || visibleMismatch || packedMismatch);

  if (error) {
    addReason(reasons, 'gpu-error', error.message ?? String(error), {
      name: error.name ?? null
    });
  }
  if (requestedRuntime === 'limited-draw' && !runtimeConfig.limitedDrawImplemented) {
    addReason(
      reasons,
      'limited-draw-not-implemented',
      'limited-draw is recognized by the selector but is not connected to rendering in Step97.'
    );
  }
  if (requestedRuntime === 'limited-draw' && !runtimeConfig.allowReadbackInDraw) {
    addReason(
      reasons,
      'readback-required-in-draw',
      'Current GPU candidate debug builders require readback; draw runtime keeps CPU reference until async/fence-sync design exists.'
    );
  }
  for (const reason of Array.isArray(extraReasons) ? extraReasons : []) {
    if (!reason?.code) continue;
    addReason(
      reasons,
      reason.code,
      reason.message ?? reason.code,
      reason.details ?? null
    );
  }
  if (runtimeConfig.requireShadowOk && shadowStatus && shadowStatus !== 'ok') {
    addReason(reasons, 'shadow-not-ok', `shadow compare status is ${shadowStatus}`, { shadowStatus });
  }
  if (runtimeConfig.requireCompare && candidateMismatch) {
    addReason(reasons, 'candidate-mismatch', 'candidate comparison reported mismatch');
  }
  if (runtimeConfig.requireCompare && visibleMismatch) {
    addReason(reasons, 'visible-mismatch', 'visible item comparison reported mismatch');
  }
  if (runtimeConfig.requireCompare && packedMismatch) {
    addReason(reasons, 'packed-mismatch', 'packed payload comparison reported mismatch');
  }
  if (runtimeConfig.requireCompare && shadowAnyMismatch && !candidateMismatch && !visibleMismatch && !packedMismatch) {
    addReason(reasons, 'shadow-mismatch', 'shadow comparison reported mismatch');
  }

  const shouldUseCpu = requestedRuntime !== 'limited-draw' ||
    reasons.length > 0 ||
    runtimeConfig.fallbackMode === 'cpu-always';
  const action = shouldUseCpu
    ? (requestedRuntime === 'limited-draw' ? 'use-cpu-reference' : 'keep-cpu-render-result')
    : 'use-gpu-candidate';
  return {
    schemaVersion: 'step97-gpu-candidate-runtime-fallback-v1',
    action,
    displayCandidateSource: shouldUseCpu ? 'cpu-reference' : 'gpu-candidate',
    gpuCandidateUsedForDisplay: !shouldUseCpu,
    fallbackMode: runtimeConfig.fallbackMode ?? 'cpu-on-error',
    requestedRuntime,
    effectiveRuntime: shouldUseCpu ? 'cpu-reference' : 'limited-draw',
    reason: reasons.length > 0 ? reasons.map((item) => item.code).join(',') : 'none',
    reasons,
    shadowStatus,
    candidateMismatch,
    visibleMismatch,
    packedMismatch,
    anyMismatch: shadowAnyMismatch,
    limitedDrawUsedForCandidateSource: requestedRuntime === 'limited-draw' && !shouldUseCpu,
    readbackPolicy: runtimeConfig.readbackPolicy ?? null
  };
}
