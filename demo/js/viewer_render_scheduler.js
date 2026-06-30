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
    lastRenderFailed: false
  };

  async function scheduleRender() {
    if (state.rendering || state.renderPending) {
      state.needsRenderAgain = true;
      return;
    }

    state.renderPending = true;
    const schedulerFrameState = {
      calledFromSchedulerFrameLoop: true,
      frameRequestIssued: true,
      requestAnimationFrameCallbackEntered: false,
      renderFrameInvoked: false,
      renderFrameCompleted: false
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
