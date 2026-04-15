export function createRenderScheduler({ renderFrame, tokenRef, isPlaying }) {
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

    requestAnimationFrame(async () => {
      state.renderPending = false;
      state.rendering = true;
      state.needsRenderAgain = false;

      try {
        await renderFrame();
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
