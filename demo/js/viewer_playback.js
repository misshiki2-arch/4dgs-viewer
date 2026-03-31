export function createViewerPlayback({
  ui,
  controls,
  scheduleRender,
  getTimeRange,
  requestNextFrame
}) {
  let playing = false;
  let lastTime = performance.now();

  function isPlaying() {
    return playing;
  }

  function setPlaying(nextPlaying) {
    playing = !!nextPlaying;
    if (ui && ui.playBtn) {
      ui.playBtn.textContent = playing ? '停止' : '再生';
    }
    if (playing) {
      scheduleRender();
    }
  }

  function togglePlaying() {
    setPlaying(!playing);
  }

  function animate(now) {
    const dt = Math.min((now - lastTime) / 1000, 0.1);
    lastTime = now;

    if (playing) {
      const range = getTimeRange();
      let t = parseFloat(ui.timeSlider.value) + dt * 2.0;
      if (t > range.max) t = range.min;
      ui.timeSlider.value = t.toFixed(2);
      if (ui.timeVal) {
        ui.timeVal.textContent = Number(ui.timeSlider.value).toFixed(2);
      }
      scheduleRender();
    }

    if (controls && typeof controls.update === 'function') {
      controls.update();
    }

    requestNextFrame(animate);
  }

  function startLoop() {
    requestNextFrame(animate);
  }

  function resetTimeClock(now = performance.now()) {
    lastTime = now;
  }

  return {
    isPlaying,
    setPlaying,
    togglePlaying,
    startLoop,
    resetTimeClock
  };
}
