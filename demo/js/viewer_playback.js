export function createViewerPlayback({
  ui,
  controls,
  scheduleRender,
  getTimeRange,
  requestNextFrame,
  onPlaybackStateChange = null,
  playbackSpeed = 2.0
}) {
  let playing = false;
  let lastTime = performance.now();

  function notifyPlaybackStateChange() {
    if (typeof onPlaybackStateChange === 'function') {
      onPlaybackStateChange(playing);
    }
  }

  function isPlaying() {
    return playing;
  }

  function setPlaying(nextPlaying) {
    const newValue = !!nextPlaying;
    if (playing === newValue) return;

    playing = newValue;

    if (ui && ui.playBtn) {
      ui.playBtn.textContent = playing ? '停止' : '再生';
    }

    notifyPlaybackStateChange();

    // 再生開始・停止のどちらでも、quality override や表示更新のために再描画する
    if (typeof scheduleRender === 'function') {
      scheduleRender();
    }
  }

  function togglePlaying() {
    setPlaying(!playing);
  }

  function stepPlayback(dtSeconds) {
    const range = getTimeRange();
    let t = parseFloat(ui.timeSlider.value) + dtSeconds * playbackSpeed;
    if (t > range.max) t = range.min;
    if (t < range.min) t = range.max;

    ui.timeSlider.value = t.toFixed(2);
    if (ui.timeVal) {
      ui.timeVal.textContent = Number(ui.timeSlider.value).toFixed(2);
    }
  }

  function animate(now) {
    const dt = Math.min((now - lastTime) / 1000, 0.1);
    lastTime = now;

    if (playing) {
      stepPlayback(dt);
      if (typeof scheduleRender === 'function') {
        scheduleRender();
      }
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
    resetTimeClock,
    stepPlayback
  };
}
