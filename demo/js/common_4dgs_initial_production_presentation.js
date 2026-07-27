export const INITIAL_PRODUCTION_PRESENTATION_TRACE_SCHEMA_VERSION =
  'phase3-initial-production-presentation-trace-v2';

const INITIAL_SCENE_SCHEDULE_SOURCE = 'default-scene-loaded';
const TRACE_HISTORY_LIMIT = 48;

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function cloneValue(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

function booleanOrNull(value) {
  return value === true ? true : value === false ? false : null;
}

function finiteNumberOrNull(value) {
  if (value === null || value === undefined || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function normalizeFrameIdentity(value) {
  return value && typeof value === 'object' ? cloneValue(value) : null;
}

function classifyInitialPresentation({ request, frame }) {
  if (!request) {
    return 'initial-schedule-request-not-observed';
  }
  if (!frame || frame.productionFrameCompleted !== true) {
    return 'initial-scheduled-production-frame-not-completed';
  }
  if (frame.compositorOutputGenerated === false) {
    return 'initial-production-frame-completed-without-compositor-output';
  }
  if (frame.compositorOutputGenerated !== true) {
    return 'initial-compositor-output-evidence-unavailable';
  }
  if (frame.viewerCanvasPresented === false) {
    return 'initial-compositor-output-not-presented-to-current-texture';
  }
  if (frame.viewerCanvasPresented !== true) {
    return 'initial-canvas-presentation-evidence-unavailable';
  }
  if (frame.browserVisibleFinalPresentationKnown !== true) {
    return 'initial-browser-visible-final-presentation-evidence-unavailable';
  }
  if (frame.browserVisiblePixelNonblank === false) {
    return 'initial-presentation-completed-but-black';
  }
  if (frame.browserVisiblePixelNonblank !== true) {
    return 'initial-browser-visible-pixel-evidence-unavailable';
  }
  return 'url-only-initial-production-presentation-succeeded';
}

function firstBlockedReason({ request, frame, classification }) {
  if (classification === 'url-only-initial-production-presentation-succeeded') {
    return null;
  }
  return (
    frame?.blockedReason ??
    frame?.runtimeError?.message ??
    request?.error?.message ??
    classification
  );
}

export function buildWebGpuProductionPresentationFrameEvidence({
  backendFrameResult = null,
  renderResult = null,
  completedAtMs = null
} = {}) {
  const tileContract =
    backendFrameResult?.webgpuTileListCompositorContract ?? null;
  const implementationContract =
    backendFrameResult?.webgpuTileCompositorFrameImplementation ?? null;
  const freshness = tileContract?.captureFreshnessEvidence ?? {};
  const productionGeneration = finiteNumberOrNull(
    freshness.productionOutputGeneration
  );
  const presentedGeneration = finiteNumberOrNull(
    freshness.presentedOutputGeneration
  );
  const outputReady = booleanOrNull(tileContract?.tileCompositorReady);
  const outputGenerated =
    productionGeneration !== null
      ? true
      : outputReady === false
        ? false
        : null;
  const presentationSubmitted = booleanOrNull(
    tileContract?.tileCompositorOutputPresentedToCurrentTexture ??
      implementationContract?.compositorOutputPresentedToCurrentTexture
  );
  const presentationReadbackCompleted = booleanOrNull(
    tileContract?.compositorCurrentTextureReadbackCompleted
  );
  const presentationReadbackNonzero = booleanOrNull(
    tileContract?.compositorCurrentTextureReadbackNonZero
  );
  const knownNonblank =
    presentationReadbackCompleted === true
      ? presentationReadbackNonzero
      : null;
  const finalPresentSourceTracingReady = booleanOrNull(
    implementationContract?.finalPresentSourceTracingReady
  );
  const finalPresentSourceStable = booleanOrNull(
    implementationContract?.finalPresentSourceStable
  );
  const finalPresentSourceAlternates = booleanOrNull(
    implementationContract?.finalPresentSourceAlternates
  );
  const tileCompositorOwnsFinalPresentation = booleanOrNull(
    implementationContract?.tileCompositorOwnsFinalPresentation
  );
  const steadyStateSamplingReady = booleanOrNull(
    implementationContract?.steadyStateSamplingReady
  );
  const presentationAllSampledFramesNonBlank = booleanOrNull(
    implementationContract?.presentationAllSampledFramesNonBlank
  );
  const browserVisibleFinalPresentationKnown =
    finalPresentSourceTracingReady === true &&
    finalPresentSourceStable === true &&
    finalPresentSourceAlternates === false &&
    tileCompositorOwnsFinalPresentation === true &&
    steadyStateSamplingReady === true;
  const browserVisiblePixelNonblank =
    browserVisibleFinalPresentationKnown &&
    presentationReadbackCompleted === true
      ? presentationAllSampledFramesNonBlank === true && knownNonblank === true
      : null;
  const runtimeError =
    renderResult?.webgpuBackendViewerFrameExecutor?.executionError ??
    implementationContract?.viewerLoopRuntimeFatalError ??
    null;
  const blockedReason =
    runtimeError?.message ??
    implementationContract?.reason ??
    tileContract?.presentationErrorMessage ??
    null;
  return {
    schemaVersion: 'phase3-production-presentation-frame-evidence-v1',
    productionFrameCompleted: true,
    compositorOutputGenerated: outputGenerated,
    compositorOutputReady: outputReady,
    productionGeneration,
    compositorGeneration: productionGeneration,
    presentedGeneration,
    productionSourceRequestIdentity:
      freshness.productionSourceRequestIdentity ?? null,
    productionFrameIdentity: normalizeFrameIdentity(
      freshness.productionFrameIdentity
    ),
    presentedFrameIdentity: normalizeFrameIdentity(
      freshness.presentedFrameIdentity
    ),
    productionOutputCreatedAtMs: finiteNumberOrNull(
      freshness.productionOutputCreatedAtMs
    ),
    presentationTimestampMs: finiteNumberOrNull(
      freshness.presentationTimestampMs
    ),
    viewerCanvasPresented: presentationSubmitted,
    logicalPresentationSucceeded:
      outputGenerated === true && presentationSubmitted === true,
    presentationReadbackCompleted,
    knownNonblank,
    currentTextureRgbPixelEvidenceKnown:
      presentationReadbackCompleted === true,
    finalPresentSourceTracingReady,
    finalPresentSourceStable,
    finalPresentSourceAlternates,
    finalPresentSourceSequence:
      implementationContract?.finalPresentSourceSequence ?? [],
    tileCompositorOwnsFinalPresentation,
    steadyStateSamplingReady,
    steadyStateSampledRafCount:
      finiteNumberOrNull(implementationContract?.steadyStateSampledRafCount),
    presentationAllSampledFramesNonBlank,
    browserVisibleFinalPresentationKnown,
    browserVisiblePixelNonblank,
    presentationSource:
      tileContract?.currentTextureSource ??
      implementationContract?.currentTextureSource ??
      null,
    runtimeError: runtimeError ? cloneValue(runtimeError) : null,
    blockedReason,
    completedAtMs: finiteNumberOrNull(completedAtMs) ?? nowMs()
  };
}

export function createInitialProductionPresentationRecorder({
  initialScheduleSource = INITIAL_SCENE_SCHEDULE_SOURCE
} = {}) {
  const installedAtMs = nowMs();
  const requests = [];
  const frames = [];
  let initialRequestIdentity = null;
  let initialRequestRecord = null;
  let initialFrameRecord = null;
  let firstProductionFrameRecord = null;

  function upsertRequest(request = {}) {
    const requestIdentity = request.requestIdentity ?? null;
    if (!requestIdentity) return null;
    const existingIndex = requests.findIndex(
      (item) => item.requestIdentity === requestIdentity
    );
    const normalized = {
      ...(existingIndex >= 0 ? requests[existingIndex] : {}),
      ...cloneValue(request)
    };
    if (existingIndex >= 0) {
      requests[existingIndex] = normalized;
    } else {
      requests.push(normalized);
    }
    while (requests.length > TRACE_HISTORY_LIMIT) requests.shift();
    if (
      initialRequestIdentity === null &&
      normalized.source === initialScheduleSource
    ) {
      initialRequestIdentity = requestIdentity;
    }
    if (requestIdentity === initialRequestIdentity) {
      initialRequestRecord = cloneValue(normalized);
    }
    return normalized;
  }

  function recordScheduleRequest(request = {}) {
    return upsertRequest(request);
  }

  function recordFrameStarted(schedulerFrameState = {}) {
    return upsertRequest({
      requestIdentity: schedulerFrameState.requestIdentity ?? null,
      source: schedulerFrameState.requestSource ?? null,
      schedulerFrameIndex: schedulerFrameState.schedulerFrameIndex ?? null,
      frameStartedAtMs: schedulerFrameState.frameStartedAtMs ?? nowMs(),
      status: 'running'
    });
  }

  function recordFrameCompleted({ schedulerFrameState = {}, frameEvidence = {} } = {}) {
    const requestIdentity = schedulerFrameState.requestIdentity ?? null;
    upsertRequest({
      requestIdentity,
      source: schedulerFrameState.requestSource ?? null,
      schedulerFrameIndex: schedulerFrameState.schedulerFrameIndex ?? null,
      frameCompletedAtMs: schedulerFrameState.frameCompletedAtMs ?? nowMs(),
      status: 'completed'
    });
    const normalizedFrame = {
      schemaVersion: 'phase3-production-presentation-request-frame-v1',
      requestIdentity,
      requestSource: schedulerFrameState.requestSource ?? null,
      schedulerFrameIndex: schedulerFrameState.schedulerFrameIndex ?? null,
      ...cloneValue(frameEvidence)
    };
    const existingIndex = frames.findIndex(
      (item) => item.requestIdentity === requestIdentity
    );
    if (existingIndex >= 0) {
      frames[existingIndex] = normalizedFrame;
    } else {
      frames.push(normalizedFrame);
    }
    if (requestIdentity === initialRequestIdentity) {
      initialFrameRecord = cloneValue(normalizedFrame);
    }
    if (
      firstProductionFrameRecord === null &&
      finiteNumberOrNull(normalizedFrame.productionGeneration) !== null
    ) {
      firstProductionFrameRecord = cloneValue(normalizedFrame);
    }
    while (frames.length > TRACE_HISTORY_LIMIT) frames.shift();
    return normalizedFrame;
  }

  function recordFrameError({ schedulerFrameState = {}, error = null } = {}) {
    const normalizedError = error
      ? {
          name: error.name ?? 'Error',
          message: error.message ?? String(error),
          stack: error.stack ?? null
        }
      : { name: 'Error', message: 'unknown-scheduler-frame-error', stack: null };
    return upsertRequest({
      requestIdentity: schedulerFrameState.requestIdentity ?? null,
      source: schedulerFrameState.requestSource ?? null,
      schedulerFrameIndex: schedulerFrameState.schedulerFrameIndex ?? null,
      frameCompletedAtMs: schedulerFrameState.frameCompletedAtMs ?? nowMs(),
      status: 'error',
      error: normalizedError
    });
  }

  function getSnapshot() {
    const snapshotTakenAtMs = nowMs();
    const initialRequest = initialRequestRecord;
    const initialFrame = initialFrameRecord;
    const latestFrame = frames.length > 0 ? frames[frames.length - 1] : null;
    const classification = classifyInitialPresentation({
      request: initialRequest,
      frame: initialFrame
    });
    const initialSucceeded =
      classification === 'url-only-initial-production-presentation-succeeded';
    const snapshot = {
      schemaVersion: INITIAL_PRODUCTION_PRESENTATION_TRACE_SCHEMA_VERSION,
      source: 'common-initial-production-presentation-recorder',
      activeFromPageLoad: true,
      recorderInstalledAtMs: installedAtMs,
      snapshotTakenAtMs,
      initialScheduleSourceExpected: initialScheduleSource,
      readOnlySnapshot: {
        getterMutatesRuntimeState: false,
        sceneRetryPerformed: false,
        renderScheduledByGetter: false,
        gpuReadbackPerformed: false,
        cloneReturnedToCaller: true,
        callerCanMutateRecorderState: false
      },
      initialScheduleRequestObserved: initialRequest ? true : false,
      initialScheduleSource: initialRequest?.source ?? null,
      initialRequestIdentity: initialRequest?.requestIdentity ?? null,
      initialRequest: initialRequest ? cloneValue(initialRequest) : null,
      firstProductionRequestIdentity:
        firstProductionFrameRecord?.requestIdentity ?? null,
      firstProductionRequestSource:
        firstProductionFrameRecord?.requestSource ?? null,
      firstProductionSourceRequestIdentity:
        firstProductionFrameRecord?.productionSourceRequestIdentity ?? null,
      initialProductionSourceRequestIdentity:
        initialFrame?.productionSourceRequestIdentity ?? null,
      initialProductionFrameCompleted:
        initialFrame?.productionFrameCompleted === true
          ? true
          : initialRequest
            ? false
            : null,
      initialProductionGeneration:
        finiteNumberOrNull(initialFrame?.productionGeneration),
      initialCompositorGeneration:
        finiteNumberOrNull(initialFrame?.compositorGeneration),
      initialPresentedGeneration:
        finiteNumberOrNull(initialFrame?.presentedGeneration),
      initialFrameIdentity: normalizeFrameIdentity(
        initialFrame?.productionFrameIdentity
      ),
      initialPresentedFrameIdentity: normalizeFrameIdentity(
        initialFrame?.presentedFrameIdentity
      ),
      compositorOutputGenerated: booleanOrNull(
        initialFrame?.compositorOutputGenerated
      ),
      viewerCanvasPresented: booleanOrNull(
        initialFrame?.viewerCanvasPresented
      ),
      logicalPresentationSucceeded: booleanOrNull(
        initialFrame?.logicalPresentationSucceeded
      ),
      knownNonblank: booleanOrNull(initialFrame?.knownNonblank),
      currentTextureRgbPixelEvidenceKnown: booleanOrNull(
        initialFrame?.currentTextureRgbPixelEvidenceKnown
      ),
      browserVisibleFinalPresentationKnown: booleanOrNull(
        initialFrame?.browserVisibleFinalPresentationKnown
      ),
      browserVisiblePixelNonblank: booleanOrNull(
        initialFrame?.browserVisiblePixelNonblank
      ),
      finalPresentSourceTracingReady: booleanOrNull(
        initialFrame?.finalPresentSourceTracingReady
      ),
      finalPresentSourceStable: booleanOrNull(
        initialFrame?.finalPresentSourceStable
      ),
      finalPresentSourceAlternates: booleanOrNull(
        initialFrame?.finalPresentSourceAlternates
      ),
      finalPresentSourceSequence:
        initialFrame?.finalPresentSourceSequence ?? [],
      tileCompositorOwnsFinalPresentation: booleanOrNull(
        initialFrame?.tileCompositorOwnsFinalPresentation
      ),
      steadyStateSamplingReady: booleanOrNull(
        initialFrame?.steadyStateSamplingReady
      ),
      steadyStateSampledRafCount:
        finiteNumberOrNull(initialFrame?.steadyStateSampledRafCount),
      presentationSource: initialFrame?.presentationSource ?? null,
      presentationTimestampMs:
        finiteNumberOrNull(initialFrame?.presentationTimestampMs) ??
        finiteNumberOrNull(initialFrame?.completedAtMs),
      urlLoadAloneGaussianVisible: initialSucceeded ? true : (
        initialFrame?.browserVisibleFinalPresentationKnown === true &&
        initialFrame?.browserVisiblePixelNonblank === false
          ? false
          : null
      ),
      captureCommandDependencyKnown: false,
      captureCommandDependencyRemaining: null,
      classification,
      blockedReason: firstBlockedReason({
        request: initialRequest,
        frame: initialFrame,
        classification
      }),
      runtimeError: initialFrame?.runtimeError ?? initialRequest?.error ?? null,
      latestProductionFrame: latestFrame ? cloneValue(latestFrame) : null,
      latestProductionGeneration:
        finiteNumberOrNull(latestFrame?.productionGeneration),
      latestCompositorGeneration:
        finiteNumberOrNull(latestFrame?.compositorGeneration),
      latestPresentedGeneration:
        finiteNumberOrNull(latestFrame?.presentedGeneration),
      latestProductionRequestIdentity:
        latestFrame?.requestIdentity ?? null,
      latestProductionSourceRequestIdentity:
        latestFrame?.productionSourceRequestIdentity ?? null,
      requestHistory: cloneValue(requests),
      frameHistory: cloneValue(frames),
      productionRuntimeBehaviorChangedByRecorder: false
    };
    return cloneValue(snapshot);
  }

  return {
    recordScheduleRequest,
    recordFrameStarted,
    recordFrameCompleted,
    recordFrameError,
    getSnapshot
  };
}
