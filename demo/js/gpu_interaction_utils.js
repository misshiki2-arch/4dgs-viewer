export function createGpuInteractionState() {
  return {
    dragActive: false,
    pointerDown: false,
    pointerButton: -1,
    lastDragStartMs: 0,
    lastDragEndMs: 0
  };
}

export function isDragButton(button) {
  return button === 0 || button === 1 || button === 2;
}

export function beginGpuDragInteraction(state, button, nowMs = performance.now()) {
  if (!state) return false;
  if (!isDragButton(button)) return false;

  state.pointerDown = true;
  state.dragActive = true;
  state.pointerButton = button;
  state.lastDragStartMs = nowMs;
  return true;
}

export function endGpuDragInteraction(state, nowMs = performance.now()) {
  if (!state) return false;

  const wasActive = !!state.dragActive || !!state.pointerDown;
  state.pointerDown = false;
  state.dragActive = false;
  state.pointerButton = -1;
  state.lastDragEndMs = nowMs;
  return wasActive;
}

export function cancelGpuDragInteraction(state, nowMs = performance.now()) {
  return endGpuDragInteraction(state, nowMs);
}

export function isGpuInteractionActive(state) {
  return !!(state && state.dragActive);
}

export function getGpuInteractionOverride(ui, state) {
  const active = isGpuInteractionActive(state);

  const baseStride = parseInt(ui.strideSlider.value, 10);
  const baseMaxVisible = parseInt(ui.maxVisibleSlider.value, 10);
  const baseRenderScale = parseFloat(ui.renderScaleSlider.value);

  if (!active) {
    return {
      interactionActive: false,
      stride: baseStride,
      maxVisible: baseMaxVisible,
      renderScale: baseRenderScale
    };
  }

  // Step14 tuned:
  // Make dragging much lighter than the initial trial.
  const interactionStride = Math.max(baseStride, 64);
  const interactionMaxVisible = Math.min(baseMaxVisible, 10000);
  const interactionRenderScale = Math.min(baseRenderScale, 0.50);

  return {
    interactionActive: true,
    stride: interactionStride,
    maxVisible: interactionMaxVisible,
    renderScale: interactionRenderScale
  };
}

export function applyGpuInteractionOverride(baseConfig, interactionOverride) {
  if (!interactionOverride || !interactionOverride.interactionActive) {
    return {
      ...baseConfig,
      interactionActive: false
    };
  }

  return {
    ...baseConfig,
    interactionActive: true,
    stride: interactionOverride.stride,
    maxVisible: interactionOverride.maxVisible,
    renderScale: interactionOverride.renderScale
  };
}

export function bindGpuDragInteraction(canvas, controls, state, onInteractionChange) {
  const notify = () => {
    if (typeof onInteractionChange === 'function') {
      onInteractionChange(isGpuInteractionActive(state));
    }
  };

  const onPointerDown = (e) => {
    if (beginGpuDragInteraction(state, e.button)) {
      notify();
    }
  };

  const onPointerUp = () => {
    if (endGpuDragInteraction(state)) {
      notify();
    }
  };

  const onPointerCancel = () => {
    if (cancelGpuDragInteraction(state)) {
      notify();
    }
  };

  const onContextMenu = (e) => {
    e.preventDefault();
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointerup', onPointerUp);
  window.addEventListener('pointercancel', onPointerCancel);
  canvas.addEventListener('contextmenu', onContextMenu);

  return function unbindGpuDragInteraction() {
    canvas.removeEventListener('pointerdown', onPointerDown);
    window.removeEventListener('pointerup', onPointerUp);
    window.removeEventListener('pointercancel', onPointerCancel);
    canvas.removeEventListener('contextmenu', onContextMenu);
  };
}
