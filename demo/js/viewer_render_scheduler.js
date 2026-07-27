export function createRenderScheduler({
  renderFrame,
  tokenRef,
  isPlaying,
  buildFramePresentationBoundary,
  onScheduleRequest,
  onFrameStarted,
  onFrameCompleted,
  onFrameError
}) {
  function summarizeRenderError(error) {
    return {
      name: error?.name ?? 'Error',
      message: error?.message ?? String(error),
      stack: error?.stack ?? null,
      string: String(error)
    };
  }

  const state = {
    renderPending: false,
    rendering: false,
    needsRenderAgain: false,
    lastRenderError: null,
    lastRenderFailed: false,
    scheduledFrameCount: 0,
    completedFrameCount: 0,
    nextRequestSequence: 1,
    queuedRequest: null,
    activeRequest: null,
    activeFrameState: null,
    lastScheduledRequest: null,
    lastCompletedRequest: null
  };

  function cloneValue(value) {
    if (value === null || value === undefined) return value;
    if (typeof structuredClone === 'function') return structuredClone(value);
    return JSON.parse(JSON.stringify(value));
  }

  function getSynchronousSnapshot() {
    const pendingRequestIdentities = [
      state.renderPending ? state.lastScheduledRequest?.requestIdentity : null,
      state.queuedRequest?.requestIdentity ?? null
    ].filter((value, index, values) => value && values.indexOf(value) === index);
    return cloneValue({
      schemaVersion: 'phase3-viewer-render-scheduler-synchronous-snapshot-v1',
      source: 'common-viewer-render-scheduler',
      snapshotTakenAtMs: nowMs(),
      latestRequestSequence: state.nextRequestSequence - 1,
      latestRequestIdentity: state.lastScheduledRequest?.requestIdentity ?? null,
      latestRequestSource: state.lastScheduledRequest?.source ?? null,
      pendingRequestCount: pendingRequestIdentities.length,
      pendingRequestIdentities,
      renderPending: state.renderPending,
      productionFrameInFlight: state.rendering,
      needsRenderAgain: state.needsRenderAgain,
      activeRequest: state.activeRequest,
      activeFrameState: state.activeFrameState,
      queuedRequest: state.queuedRequest,
      lastScheduledRequest: state.lastScheduledRequest,
      lastCompletedRequest: state.lastCompletedRequest,
      scheduledFrameCount: state.scheduledFrameCount,
      completedFrameCount: state.completedFrameCount,
      readOnlySnapshot: true
    });
  }

  function nowMs() {
    return typeof performance !== 'undefined' && typeof performance.now === 'function'
      ? performance.now()
      : Date.now();
  }

  function notify(callback, payload) {
    if (typeof callback !== 'function') return;
    try {
      callback(payload);
    } catch (error) {
      console.warn('Viewer render scheduler observer failed', error);
    }
  }

  function buildRequest(input = {}) {
    const metadata =
      input &&
      typeof input === 'object' &&
      typeof input.source === 'string'
        ? input
        : {};
    const sequence = state.nextRequestSequence;
    state.nextRequestSequence += 1;
    return {
      requestIdentity: `viewer-render-request:${sequence}`,
      source: metadata.source ?? 'viewer-render-request',
      requestedAtMs: nowMs(),
      forceProductionUpdate: metadata.forceProductionUpdate === true,
      metadata: metadata.metadata && typeof metadata.metadata === 'object'
        ? { ...metadata.metadata }
        : null,
      disposition: 'pending'
    };
  }

  function startRequest(request) {
    const activeRequest = {
      ...request,
      disposition:
        request.disposition === 'coalesced-next-frame'
          ? 'scheduled-after-coalescing'
          : 'scheduled'
    };
    if (state.rendering || state.renderPending) {
      state.needsRenderAgain = true;
      const queuedRequest = {
        ...activeRequest,
        disposition: 'coalesced-next-frame'
      };
      // Preserve one coherent request identity/source/metadata tuple. A later
      // forced request may upgrade weaker queued work, but cannot replace an
      // already-forced scene-ready request with unrelated metadata.
      if (
        !state.queuedRequest ||
        (
          state.queuedRequest.forceProductionUpdate !== true &&
          queuedRequest.forceProductionUpdate === true
        )
      ) {
        state.queuedRequest = queuedRequest;
      }
      state.lastScheduledRequest = queuedRequest;
      notify(onScheduleRequest, { ...queuedRequest });
      return { ...queuedRequest };
    }

    state.renderPending = true;
    state.activeRequest = { ...activeRequest };
    state.scheduledFrameCount += 1;
    state.lastScheduledRequest = activeRequest;
    notify(onScheduleRequest, { ...activeRequest });
    const schedulerFrameState = {
      calledFromSchedulerFrameLoop: true,
      frameRequestIssued: true,
      requestAnimationFrameCallbackEntered: false,
      renderFrameInvoked: false,
      renderFrameCompleted: false,
      schedulerFrameIndex: state.scheduledFrameCount,
      schedulerFrameCount: state.scheduledFrameCount,
      requestIdentity: activeRequest.requestIdentity,
      requestSource: activeRequest.source,
      requestIssuedAtMs: activeRequest.requestedAtMs,
      requestDisposition: activeRequest.disposition,
      forceProductionUpdate: activeRequest.forceProductionUpdate === true,
      requestMetadata: activeRequest.metadata
    };
    state.activeFrameState = { ...schedulerFrameState };

    requestAnimationFrame(async () => {
      state.renderPending = false;
      state.rendering = true;
      schedulerFrameState.requestAnimationFrameCallbackEntered = true;
      schedulerFrameState.frameStartedAtMs = nowMs();
      state.activeFrameState = { ...schedulerFrameState };
      notify(onFrameStarted, { ...schedulerFrameState });

      try {
        schedulerFrameState.renderFrameInvoked = true;
        const renderResult = await renderFrame({
          schedulerFrameState
        });
        state.lastRenderError = null;
        state.lastRenderFailed = false;
        schedulerFrameState.renderFrameCompleted = true;
        schedulerFrameState.frameCompletedAtMs = nowMs();
        state.completedFrameCount += 1;
        schedulerFrameState.schedulerCompletedFrameCount =
          state.completedFrameCount;
        if (
          renderResult &&
          typeof buildFramePresentationBoundary === 'function'
        ) {
          renderResult.webgpuSchedulerFramePresentationBoundary =
            buildFramePresentationBoundary({
              schedulerFrameState,
              renderResult
            });
        }
        state.lastCompletedRequest = {
          ...activeRequest,
          schedulerFrameIndex: schedulerFrameState.schedulerFrameIndex,
          completedAtMs: schedulerFrameState.frameCompletedAtMs,
          status: 'completed'
        };
        notify(onFrameCompleted, {
          schedulerFrameState: { ...schedulerFrameState },
          renderResult
        });
      } catch (error) {
        state.lastRenderError = summarizeRenderError(error);
        state.lastRenderFailed = true;
        schedulerFrameState.renderFrameError = state.lastRenderError;
        schedulerFrameState.frameCompletedAtMs = nowMs();
        state.lastCompletedRequest = {
          ...activeRequest,
          schedulerFrameIndex: schedulerFrameState.schedulerFrameIndex,
          completedAtMs: schedulerFrameState.frameCompletedAtMs,
          status: 'error',
          error: state.lastRenderError
        };
        notify(onFrameError, {
          schedulerFrameState: { ...schedulerFrameState },
          error: state.lastRenderError
        });
        console.error('Viewer render scheduler frame failed', error);
      } finally {
        state.rendering = false;
        state.activeRequest = null;
        state.activeFrameState = null;

        if (
          state.queuedRequest ||
          state.needsRenderAgain ||
          (typeof isPlaying === 'function' && isPlaying())
        ) {
          const queuedRequest = state.queuedRequest;
          state.queuedRequest = null;
          state.needsRenderAgain = false;
          if (queuedRequest) {
            startRequest(queuedRequest);
          } else {
            startRequest(buildRequest({ source: 'viewer-loop-continuation' }));
          }
        }
      }
    });
    return { ...activeRequest };
  }

  async function scheduleRender(input = {}) {
    return startRequest(buildRequest(input));
  }

  return {
    state,
    scheduleRender,
    getSynchronousSnapshot
  };
}
