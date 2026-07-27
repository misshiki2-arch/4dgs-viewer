export const FINAL_CANVAS_PRESENTATION_TRACE_SCHEMA_VERSION =
  'phase3-final-canvas-presentation-trace-v1';

export const FINAL_CANVAS_PRESENTATION_BOUNDARY_SCHEMA_VERSION =
  'phase3-final-canvas-presentation-boundary-v1';

const DEFAULT_HISTORY_LIMIT = 256;
const DEFAULT_STEADY_STATE_EVENT_COUNT = 8;

export const FINAL_CANVAS_PRESENTATION_PATHS = Object.freeze({
  TILE_COMPOSITOR: 'webgpu-tile-compositor-current-texture',
  BOUNDED_FIRST_PRESENT: 'webgpu-viewer-canvas-bounded-first-present',
  BOUNDED_COLOR_PRESENT: 'webgpu-viewer-canvas-bounded-color-present',
  GUARDED_PRESENTATION: 'webgpu-guarded-presentation-adapter-current-texture'
});

export const FINAL_CANVAS_PRESENTATION_EVENT_KINDS = Object.freeze([
  'production-presentation',
  'cached-production-presentation',
  'clear',
  'black-fallback',
  'diagnostic-presentation',
  'presentation-failure'
]);

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function cloneValue(value) {
  if (value === null || value === undefined) return value;
  if (typeof structuredClone === 'function') return structuredClone(value);
  return JSON.parse(JSON.stringify(value));
}

function finiteNumberOrNull(value) {
  if (value === null || value === undefined || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function booleanOrNull(value) {
  return value === true ? true : value === false ? false : null;
}

function normalizePixelResult(value) {
  if (value === true || value === 'nonblank') return 'nonblank';
  if (value === false || value === 'black') return 'black';
  return 'unknown';
}

function normalizedIdentity(value) {
  return value && typeof value === 'object' ? cloneValue(value) : null;
}

function identityMatches(actual, expected) {
  if (!expected || typeof expected !== 'object') {
    return { known: true, matches: true, mismatchedKeys: [], missingKeys: [] };
  }
  if (!actual || typeof actual !== 'object') {
    return {
      known: false,
      matches: false,
      mismatchedKeys: [],
      missingKeys: Object.keys(expected).filter((key) => expected[key] != null)
    };
  }
  const requiredKeys = Object.keys(expected).filter((key) => expected[key] != null);
  const missingKeys = requiredKeys.filter((key) => actual[key] == null);
  const mismatchedKeys = requiredKeys.filter(
    (key) => actual[key] != null && actual[key] !== expected[key]
  );
  return {
    known: missingKeys.length === 0,
    matches: missingKeys.length === 0 && mismatchedKeys.length === 0,
    mismatchedKeys,
    missingKeys
  };
}

function eventMatchesExpectedSource(event, {
  expectedRequestIdentity = null,
  expectedGeneration = null,
  expectedFrameIdentity = null
} = {}) {
  const requestKnown = expectedRequestIdentity == null ||
    event?.sourceRequestIdentity != null;
  const requestMatches = expectedRequestIdentity == null ||
    event?.sourceRequestIdentity === expectedRequestIdentity;
  const actualGeneration = finiteNumberOrNull(
    event?.presentedGeneration ?? event?.productionGeneration
  );
  const generationKnown = expectedGeneration == null || actualGeneration != null;
  const generationMatches = expectedGeneration == null ||
    actualGeneration === finiteNumberOrNull(expectedGeneration);
  const frameIdentityComparison = identityMatches(
    event?.frameIdentity,
    expectedFrameIdentity
  );
  return {
    known: requestKnown && generationKnown && frameIdentityComparison.known,
    matches: requestMatches && generationMatches && frameIdentityComparison.matches,
    requestKnown,
    requestMatches,
    generationKnown,
    generationMatches,
    frameIdentityComparison
  };
}

function isSuccessfulCanvasWrite(event) {
  return event?.canvasWriteAttempted === true &&
    event?.canvasWriteSubmitted === true &&
    event?.canvasWriteCompleted === true &&
    event?.presentationFailed !== true;
}

function eventOverwritesCandidate(event, candidate) {
  if (!isSuccessfulCanvasWrite(event)) return false;
  if (event?.staleSource === true) return true;
  if (event?.eventKind === 'clear' || event?.eventKind === 'black-fallback') {
    return true;
  }
  if (normalizePixelResult(event?.sourcePixelResult) === 'black') return true;
  const candidateGeneration = finiteNumberOrNull(
    candidate?.presentedGeneration ?? candidate?.productionGeneration
  );
  const eventGeneration = finiteNumberOrNull(
    event?.presentedGeneration ?? event?.productionGeneration
  );
  if (candidateGeneration !== null && eventGeneration !== candidateGeneration) {
    return true;
  }
  if (
    candidate?.sourceRequestIdentity != null &&
    event?.sourceRequestIdentity !== candidate.sourceRequestIdentity
  ) {
    return true;
  }
  const frameComparison = identityMatches(
    event?.frameIdentity,
    candidate?.frameIdentity
  );
  return frameComparison.known && frameComparison.matches === false;
}

function classifyBrowserVisibleResult({
  finalCanvasEvent,
  finalSourceIdentity,
  laterOverwriteDetected,
  steadyStateConfirmed,
  coverageComplete = true,
  quiescenceConfirmed = true
}) {
  if (!finalCanvasEvent) {
    return {
      result: null,
      classification: 'unknown-no-successful-final-canvas-write',
      blockedReason: 'successful-final-canvas-write-not-observed-before-boundary'
    };
  }
  const pixelResult = normalizePixelResult(finalCanvasEvent.sourcePixelResult);
  if (pixelResult === 'black') {
    return {
      result: false,
      classification: 'browser-visible-final-canvas-state-black',
      blockedReason: 'final-canvas-write-source-pixel-evidence-black'
    };
  }
  if (pixelResult !== 'nonblank') {
    return {
      result: null,
      classification: 'unknown-final-canvas-source-pixels',
      blockedReason: 'final-canvas-write-source-pixel-evidence-unknown'
    };
  }
  if (coverageComplete !== true) {
    return {
      result: null,
      classification: 'unknown-final-canvas-write-path-coverage',
      blockedReason: 'active-current-texture-write-path-coverage-incomplete'
    };
  }
  if (finalSourceIdentity.known !== true || finalSourceIdentity.matches !== true) {
    return {
      result: null,
      classification: 'unknown-final-canvas-source-identity',
      blockedReason: finalSourceIdentity.known === true
        ? 'final-canvas-write-source-identity-mismatch'
        : 'final-canvas-write-source-identity-incomplete'
    };
  }
  if (laterOverwriteDetected === true) {
    return {
      result: false,
      classification: 'browser-visible-nonblank-source-overwritten',
      blockedReason: 'later-clear-black-or-stale-source-overwrite'
    };
  }
  if (steadyStateConfirmed !== true) {
    return {
      result: null,
      classification: 'unknown-final-canvas-steady-state',
      blockedReason: 'steady-state-final-canvas-persistence-not-confirmed'
    };
  }
  if (quiescenceConfirmed !== true) {
    return {
      result: null,
      classification: 'unknown-final-canvas-quiescence',
      blockedReason: 'scheduler-presentation-quiescence-not-confirmed'
    };
  }
  return {
    result: true,
    classification: 'browser-visible-final-canvas-state-nonblank-and-stable',
    blockedReason: null
  };
}

export function buildFinalCanvasPresentationBoundaryEvidence({
  events = [],
  boundaryKind = 'unspecified',
  boundaryTimestampMs = nowMs(),
  expectedRequestIdentity = null,
  expectedGeneration = null,
  expectedFrameIdentity = null,
  requiredSteadyStateEventCount = DEFAULT_STEADY_STATE_EVENT_COUNT,
  canvasWritePathCoverage = null,
  quiescenceEvidence = null,
  requireCanvasWritePathCoverage = false,
  requireQuiescence = false
} = {}) {
  const timestamp = finiteNumberOrNull(boundaryTimestampMs) ?? nowMs();
  const eventHistory = (Array.isArray(events) ? events : [])
    .filter((event) => finiteNumberOrNull(event?.timestampMs) <= timestamp)
    .map((event) => cloneValue(event));
  const successfulWrites = eventHistory.filter(isSuccessfulCanvasWrite);
  const expectedSource = {
    expectedRequestIdentity,
    expectedGeneration: finiteNumberOrNull(expectedGeneration),
    expectedFrameIdentity: normalizedIdentity(expectedFrameIdentity)
  };
  const matchingWrites = successfulWrites.filter(
    (event) =>
      eventMatchesExpectedSource(event, expectedSource).matches === true &&
      normalizePixelResult(event?.sourcePixelResult) === 'nonblank' &&
      event?.eventKind !== 'clear' &&
      event?.eventKind !== 'black-fallback'
  );
  const candidate = matchingWrites.length > 0
    ? matchingWrites[matchingWrites.length - 1]
    : successfulWrites.length > 0
      ? successfulWrites[successfulWrites.length - 1]
      : null;
  const candidateSequence = finiteNumberOrNull(candidate?.eventSequence);
  const writesAfterCandidate = candidateSequence == null
    ? []
    : successfulWrites.filter(
        (event) => finiteNumberOrNull(event?.eventSequence) > candidateSequence
      );
  const overwriteEvents = candidate
    ? writesAfterCandidate.filter((event) => eventOverwritesCandidate(event, candidate))
    : [];
  const finalCanvasEvent = successfulWrites.length > 0
    ? successfulWrites[successfulWrites.length - 1]
    : null;
  const finalSourceIdentity = eventMatchesExpectedSource(
    finalCanvasEvent,
    expectedSource
  );
  const requiredCount = Math.max(
    1,
    Math.round(finiteNumberOrNull(requiredSteadyStateEventCount) ??
      DEFAULT_STEADY_STATE_EVENT_COUNT)
  );
  const steadyStateEvents = successfulWrites.slice(-requiredCount);
  const steadyStateIdentityComparisons = steadyStateEvents.map((event) =>
    eventMatchesExpectedSource(event, expectedSource)
  );
  const steadyStateConfirmed =
    steadyStateEvents.length >= requiredCount &&
    steadyStateEvents.every(
      (event) => normalizePixelResult(event?.sourcePixelResult) === 'nonblank'
    ) &&
    steadyStateIdentityComparisons.every(
      (comparison) => comparison.known === true && comparison.matches === true
    ) &&
    steadyStateEvents.every(
      (event) => event?.eventKind !== 'clear' && event?.eventKind !== 'black-fallback'
    );
  const classification = classifyBrowserVisibleResult({
    finalCanvasEvent,
    finalSourceIdentity,
    laterOverwriteDetected: overwriteEvents.length > 0,
    steadyStateConfirmed,
    coverageComplete:
      requireCanvasWritePathCoverage === true
        ? canvasWritePathCoverage?.coverageComplete === true
        : true,
    quiescenceConfirmed:
      requireQuiescence === true
        ? quiescenceEvidence?.quiescent === true
        : true
  });
  return {
    schemaVersion: FINAL_CANVAS_PRESENTATION_BOUNDARY_SCHEMA_VERSION,
    source: 'common-final-canvas-presentation-trace',
    boundaryKind,
    boundaryIdentity: `${boundaryKind}:${eventHistory.length}:${timestamp}`,
    boundaryTimestampMs: timestamp,
    readOnlySnapshot: {
      getterMutatesRuntimeState: false,
      renderScheduledByGetter: false,
      sceneRetryPerformed: false,
      gpuReadbackPerformedByGetter: false,
      canvasWritePerformedByGetter: false,
      cloneReturnedToCaller: true
    },
    expectedSource,
    eventCountAtBoundary: eventHistory.length,
    successfulCanvasWriteCount: successfulWrites.length,
    matchingSourceCanvasWriteCount: matchingWrites.length,
    finalCanvasEventIdentity: finalCanvasEvent?.eventIdentity ?? null,
    finalCanvasEvent: finalCanvasEvent ? cloneValue(finalCanvasEvent) : null,
    finalPresentationSource: finalCanvasEvent?.presentationSource ?? null,
    finalSourceRequestIdentity: finalCanvasEvent?.sourceRequestIdentity ?? null,
    finalPresentingRequestIdentity:
      finalCanvasEvent?.presentingRequestIdentity ?? null,
    finalProductionGeneration:
      finiteNumberOrNull(finalCanvasEvent?.productionGeneration),
    finalCompositorGeneration:
      finiteNumberOrNull(finalCanvasEvent?.compositorGeneration),
    finalPresentedGeneration:
      finiteNumberOrNull(finalCanvasEvent?.presentedGeneration),
    finalFrameIdentity: normalizedIdentity(finalCanvasEvent?.frameIdentity),
    finalSourcePixelEvidenceIdentity:
      finalCanvasEvent?.sourcePixelEvidenceIdentity ?? null,
    finalSourcePixelResult:
      finalCanvasEvent
        ? normalizePixelResult(finalCanvasEvent.sourcePixelResult)
        : 'unknown',
    finalCanvasWriteAttempted:
      booleanOrNull(finalCanvasEvent?.canvasWriteAttempted),
    finalCanvasWriteSubmitted:
      booleanOrNull(finalCanvasEvent?.canvasWriteSubmitted),
    finalCanvasWriteCompleted:
      booleanOrNull(finalCanvasEvent?.canvasWriteCompleted),
    finalSourceIdentityKnown: finalSourceIdentity.known,
    finalSourceIdentityMatchesExpected: finalSourceIdentity.matches,
    finalSourceIdentityComparison: finalSourceIdentity,
    laterCanvasWriteCount: writesAfterCandidate.length,
    laterOverwriteDetected: overwriteEvents.length > 0,
    laterClearDetected: overwriteEvents.some(
      (event) => event?.eventKind === 'clear'
    ),
    laterBlackDetected: overwriteEvents.some(
      (event) => normalizePixelResult(event?.sourcePixelResult) === 'black'
    ),
    laterDifferentGenerationDetected: overwriteEvents.some(
      (event) => finiteNumberOrNull(event?.presentedGeneration) !==
        finiteNumberOrNull(candidate?.presentedGeneration)
    ),
    laterStaleSourceDetected: overwriteEvents.some(
      (event) => event?.staleSource === true
    ),
    overwriteEventIdentities: overwriteEvents.map(
      (event) => event?.eventIdentity ?? null
    ),
    requiredSteadyStateEventCount: requiredCount,
    steadyStateObservedEventCount: steadyStateEvents.length,
    steadyStateObservationMutatedRuntime: false,
    steadyStateConfirmed,
    steadyStateEventIdentities: steadyStateEvents.map(
      (event) => event?.eventIdentity ?? null
    ),
    canvasWritePathCoverage: normalizedIdentity(canvasWritePathCoverage),
    quiescenceEvidence: normalizedIdentity(quiescenceEvidence),
    requireCanvasWritePathCoverage: requireCanvasWritePathCoverage === true,
    requireQuiescence: requireQuiescence === true,
    browserVisibleResult: classification.result,
    classification: classification.classification,
    unknownOrBlockedReason: classification.blockedReason,
    eventHistory
  };
}

export function createFinalCanvasPresentationTraceRecorder({
  historyLimit = DEFAULT_HISTORY_LIMIT,
  requiredSteadyStateEventCount = DEFAULT_STEADY_STATE_EVENT_COUNT,
  activePathIdentities = []
} = {}) {
  const installedAtMs = nowMs();
  const events = [];
  let nextEventSequence = 1;
  let nextWriteTokenSequence = 1;
  const registeredPaths = new Map();
  const observedPathIdentities = new Set();
  const inFlightWrites = new Map();
  let quiescenceObservation = null;

  function registerPath(path = {}) {
    const pathIdentity = path.pathIdentity ?? null;
    if (!pathIdentity) return null;
    const normalized = {
      pathIdentity,
      source: path.source ?? null,
      supportedEventKinds: Array.isArray(path.supportedEventKinds)
        ? [...path.supportedEventKinds]
        : [],
      traceRegistrationActive: true,
      registeredAtMs: nowMs()
    };
    registeredPaths.set(pathIdentity, normalized);
    return cloneValue(normalized);
  }

  function buildCoverageSnapshot() {
    const expected = [...new Set([
      ...activePathIdentities.filter(Boolean),
      ...observedPathIdentities
    ])];
    const registered = [...registeredPaths.keys()];
    const unregistered = expected.filter((path) => !registeredPaths.has(path));
    const supportedEventKinds = [...new Set(
      [...registeredPaths.values()].flatMap(
        (path) => path.supportedEventKinds ?? []
      )
    )];
    const missingEventKinds = FINAL_CANVAS_PRESENTATION_EVENT_KINDS.filter(
      (kind) => !supportedEventKinds.includes(kind)
    );
    return {
      schemaVersion: 'phase3-final-canvas-write-path-coverage-v1',
      source: 'common-final-canvas-presentation-registry',
      activePathIdentities: expected,
      observedPathIdentities: [...observedPathIdentities],
      registeredPathIdentities: registered,
      registeredPaths: [...registeredPaths.values()].map(cloneValue),
      unregisteredPathIdentities: unregistered,
      unregisteredWritePathCount: unregistered.length,
      supportedEventKinds,
      missingRequiredEventKinds: missingEventKinds,
      coverageComplete:
        expected.length > 0 &&
        unregistered.length === 0,
      eventVocabularyCoverageComplete: missingEventKinds.length === 0
    };
  }

  function beginWrite(event = {}) {
    const writeToken = `final-canvas-write:${nextWriteTokenSequence}`;
    nextWriteTokenSequence += 1;
    inFlightWrites.set(writeToken, {
      writeToken,
      pathIdentity: event.pathIdentity ?? null,
      sourceRequestIdentity: event.sourceRequestIdentity ?? null,
      productionGeneration: finiteNumberOrNull(event.productionGeneration),
      startedAtMs: nowMs()
    });
    return writeToken;
  }

  function recordEvent(event = {}) {
    const eventSequence = nextEventSequence;
    nextEventSequence += 1;
    const normalized = {
      schemaVersion: 'phase3-final-canvas-presentation-event-v1',
      eventSequence,
      eventIdentity: `final-canvas-event:${eventSequence}`,
      timestampMs: finiteNumberOrNull(event.timestampMs) ?? nowMs(),
      eventKind: event.eventKind ?? 'presentation',
      presentationPathIdentity: event.presentationPathIdentity ?? null,
      presentationSource: event.presentationSource ?? null,
      sourceRequestIdentity: event.sourceRequestIdentity ?? null,
      presentingRequestIdentity: event.presentingRequestIdentity ?? null,
      scheduleSource: event.scheduleSource ?? null,
      productionGeneration: finiteNumberOrNull(event.productionGeneration),
      compositorGeneration: finiteNumberOrNull(event.compositorGeneration),
      presentedGeneration: finiteNumberOrNull(event.presentedGeneration),
      frameIdentity: normalizedIdentity(event.frameIdentity),
      sourcePixelEvidenceIdentity:
        event.sourcePixelEvidenceIdentity ?? null,
      sourcePixelResult: normalizePixelResult(event.sourcePixelResult),
      sourcePixelStats: event.sourcePixelStats ?? null,
      canvasWriteAttempted: booleanOrNull(event.canvasWriteAttempted),
      canvasWriteSubmitted: booleanOrNull(event.canvasWriteSubmitted),
      canvasWriteCompleted: booleanOrNull(event.canvasWriteCompleted),
      staleSource: booleanOrNull(event.staleSource),
      presentationFailed: booleanOrNull(event.presentationFailed),
      error: event.error ?? null,
      blockedReason: event.blockedReason ?? null
    };
    if (normalized.presentationPathIdentity) {
      observedPathIdentities.add(normalized.presentationPathIdentity);
    }
    if (event.writeToken) inFlightWrites.delete(event.writeToken);
    events.push(normalized);
    while (events.length > Math.max(1, historyLimit | 0)) events.shift();
    return cloneValue(normalized);
  }

  function getRuntimeStateSnapshot() {
    return cloneValue({
      schemaVersion: 'phase3-final-canvas-presentation-runtime-state-v1',
      snapshotTakenAtMs: nowMs(),
      latestEventSequence: nextEventSequence - 1,
      latestEventIdentity: events.at(-1)?.eventIdentity ?? null,
      latestEvent: events.at(-1) ?? null,
      presentationInFlightCount: inFlightWrites.size,
      inFlightWrites: [...inFlightWrites.values()],
      canvasWritePathCoverage: buildCoverageSnapshot(),
      quiescenceObservation,
      readOnlySnapshot: true
    });
  }

  function observeQuiescence({ schedulerSnapshot = null, requiredConsecutive = 3 } = {}) {
    const latestEvent = events.at(-1) ?? null;
    const observationIdentity = JSON.stringify({
      eventSequence: nextEventSequence - 1,
      sourceRequestIdentity: latestEvent?.sourceRequestIdentity ?? null,
      generation:
        finiteNumberOrNull(latestEvent?.presentedGeneration) ??
        finiteNumberOrNull(latestEvent?.productionGeneration),
      presentationSource: latestEvent?.presentationSource ?? null
    });
    const schedulerIdle =
      schedulerSnapshot?.pendingRequestCount === 0 &&
      schedulerSnapshot?.renderPending === false &&
      schedulerSnapshot?.productionFrameInFlight === false &&
      schedulerSnapshot?.needsRenderAgain === false;
    const presentationIdle = inFlightWrites.size === 0;
    const sameAsPrevious =
      quiescenceObservation?.observationIdentity === observationIdentity;
    const consecutiveStableObservationCount =
      schedulerIdle && presentationIdle
        ? sameAsPrevious
          ? (quiescenceObservation?.consecutiveStableObservationCount ?? 0) + 1
          : 1
        : 0;
    const required = Math.max(2, Math.round(Number(requiredConsecutive) || 3));
    quiescenceObservation = {
      schemaVersion: 'phase3-final-canvas-quiescence-observation-v1',
      source: 'passive-scheduler-and-presentation-state-monitor',
      observedAtMs: nowMs(),
      observationIdentity,
      latestEventSequence: nextEventSequence - 1,
      latestEventIdentity: latestEvent?.eventIdentity ?? null,
      finalSourceRequestIdentity: latestEvent?.sourceRequestIdentity ?? null,
      finalGeneration:
        finiteNumberOrNull(latestEvent?.presentedGeneration) ??
        finiteNumberOrNull(latestEvent?.productionGeneration),
      finalPresentationSource: latestEvent?.presentationSource ?? null,
      schedulerPendingRequestCount: schedulerSnapshot?.pendingRequestCount ?? null,
      productionFrameInFlight:
        schedulerSnapshot?.productionFrameInFlight ?? null,
      presentationInFlightCount: inFlightWrites.size,
      schedulerIdle,
      presentationIdle,
      eventSequenceStable: sameAsPrevious,
      consecutiveStableObservationCount,
      requiredConsecutiveStableObservationCount: required,
      observationRequestedFrame: false,
      observationMutatedScheduler: false,
      observationMutatedProductionState: false,
      observationWroteCanvas: false,
      observationPerformedGpuReadback: false,
      quiescent:
        schedulerIdle &&
        presentationIdle &&
        consecutiveStableObservationCount >= required
    };
    return cloneValue(quiescenceObservation);
  }

  function getSnapshot(options = {}) {
    const boundary = buildFinalCanvasPresentationBoundaryEvidence({
      ...options,
      events,
      canvasWritePathCoverage:
        options.canvasWritePathCoverage ?? buildCoverageSnapshot(),
      quiescenceEvidence:
        options.quiescenceEvidence ?? quiescenceObservation,
      requiredSteadyStateEventCount:
        options.requiredSteadyStateEventCount ?? requiredSteadyStateEventCount
    });
    return cloneValue({
      ...boundary,
      traceSchemaVersion: FINAL_CANVAS_PRESENTATION_TRACE_SCHEMA_VERSION,
      traceInstalledAtMs: installedAtMs,
      traceActiveFromPageLoad: true,
      traceHistoryLimit: Math.max(1, historyLimit | 0),
      productionRuntimeBehaviorChangedByTrace: false
    });
  }

  return {
    registerPath,
    beginWrite,
    recordEvent,
    getSnapshot,
    getRuntimeStateSnapshot,
    observeQuiescence
  };
}

export function registerFinalCanvasPresentationPath(viewerCanvasState, path) {
  const recorder = viewerCanvasState?.finalCanvasPresentationTraceRecorder;
  return typeof recorder?.registerPath === 'function'
    ? recorder.registerPath(path)
    : null;
}

export function beginFinalCanvasPresentationWrite(viewerCanvasState, event) {
  const recorder = viewerCanvasState?.finalCanvasPresentationTraceRecorder;
  return typeof recorder?.beginWrite === 'function'
    ? recorder.beginWrite(event)
    : null;
}

export function recordFinalCanvasPresentationEvent(
  viewerCanvasState,
  event
) {
  const recorder = viewerCanvasState?.finalCanvasPresentationTraceRecorder;
  if (!recorder || typeof recorder.recordEvent !== 'function') return null;
  try {
    return recorder.recordEvent(event);
  } catch (error) {
    console.warn('Final canvas presentation trace recorder failed', error);
    return null;
  }
}
