export function createRenderScheduler({
  renderFrame,
  tokenRef,
  isPlaying,
  buildFramePresentationBoundary
}) {
  const state = {
    renderPending: false,
    rendering: false,
    needsRenderAgain: false
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
