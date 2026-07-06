export function createRenderScheduler({
  renderFrame,
  tokenRef,
  isPlaying,
  buildFramePresentationBoundary
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
    completedFrameCount: 0
  };

  async function scheduleRender() {
    if (state.rendering || state.renderPending) {
      state.needsRenderAgain = true;
      return;
    }

    state.renderPending = true;
    state.scheduledFrameCount += 1;
    const schedulerFrameState = {
      calledFromSchedulerFrameLoop: true,
      frameRequestIssued: true,
      requestAnimationFrameCallbackEntered: false,
      renderFrameInvoked: false,
      renderFrameCompleted: false,
      schedulerFrameIndex: state.scheduledFrameCount,
      schedulerFrameCount: state.scheduledFrameCount
    };

    requestAnimationFrame(async () => {
      state.renderPending = false;
      state.rendering = true;
      state.needsRenderAgain = false;
      schedulerFrameState.requestAnimationFrameCallbackEntered = true;

      try {
        schedulerFrameState.renderFrameInvoked = true;
        const renderResult = await renderFrame({
          schedulerFrameState
        });
        state.lastRenderError = null;
        state.lastRenderFailed = false;
        schedulerFrameState.renderFrameCompleted = true;
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
      } catch (error) {
        state.lastRenderError = summarizeRenderError(error);
        state.lastRenderFailed = true;
        schedulerFrameState.renderFrameError = state.lastRenderError;
        console.error('Viewer render scheduler frame failed', error);
      } finally {
        state.rendering = false;

        if (state.needsRenderAgain || (typeof isPlaying === 'function' && isPlaying())) {
          state.needsRenderAgain = false;
          scheduleRender();
        }
      }
    });
  }

  return {
    state,
    scheduleRender
  };
}
