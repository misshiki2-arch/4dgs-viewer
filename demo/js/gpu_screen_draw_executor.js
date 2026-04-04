// Step25:
// future path である gpu-screen 用の受け皿。
// まだ本実装は入れず、renderer / draw path selector / debug が
// 一貫して扱えるための最小インターフェースだけを定義する。
//
// 目的:
// - gpu-screen 経路を「未実装だが設計上は存在する経路」として独立させる
// - renderer が将来ここを呼べるようにする
// - packed への fallback 理由を明示しやすくする
//
// 現段階では draw 自体は行わない。

export function createGpuScreenDrawState() {
  return {
    isReady: false,
    hasProgram: false,
    hasVao: false,
    hasBuffers: false,
    layoutVersion: 0,
    strideBytes: 0,
    attributeCount: 0,
    lastPath: 'gpu-screen',
    lastBuildMs: 0,
    lastDrawCount: 0,
    lastReason: 'not-initialized'
  };
}

export function ensureGpuScreenDrawResources(gl, gpu) {
  if (!gpu.gpuScreenDrawState) {
    gpu.gpuScreenDrawState = createGpuScreenDrawState();
  }
  return { state: gpu.gpuScreenDrawState, ready: false, reason: 'gpu-screen-not-implemented' };
}

export function isGpuScreenDrawReady(gpu) {
  return !!gpu?.gpuScreenDrawState?.isReady;
}

export function summarizeGpuScreenDrawState(gpu) {
  const state = gpu?.gpuScreenDrawState ?? createGpuScreenDrawState();
  return {
    gpuScreenDrawReady: !!state.isReady,
    gpuScreenHasProgram: !!state.hasProgram,
    gpuScreenHasVao: !!state.hasVao,
    gpuScreenHasBuffers: !!state.hasBuffers,
    gpuScreenLayoutVersion: state.layoutVersion ?? 0,
    gpuScreenStrideBytes: state.strideBytes ?? 0,
    gpuScreenAttributeCount: state.attributeCount ?? 0,
    gpuScreenLastPath: state.lastPath ?? 'gpu-screen',
    gpuScreenLastBuildMs: state.lastBuildMs ?? 0,
    gpuScreenLastDrawCount: state.lastDrawCount ?? 0,
    gpuScreenReason: state.lastReason ?? 'not-initialized'
  };
}

export function uploadAndDrawGpuScreen(gl, gpu, gpuScreenSpace, canvasWidth, canvasHeight) {
  const ensured = ensureGpuScreenDrawResources(gl, gpu);
  const state = ensured.state;
  state.lastPath = 'gpu-screen';
  state.lastBuildMs = Number.isFinite(gpuScreenSpace?.summary?.buildMs) ? gpuScreenSpace.summary.buildMs : 0;
  state.lastDrawCount = Number.isFinite(gpuScreenSpace?.packedCount) ? gpuScreenSpace.packedCount : 0;
  state.lastReason = 'gpu-screen-not-implemented';
  return {
    drawCount: 0,
    gpuScreenDraw: false,
    gpuScreenReady: false,
    gpuScreenFallbackSuggested: true,
    gpuScreenReason: 'gpu-screen-not-implemented',
    gpuScreenSummary: summarizeGpuScreenDrawState(gpu),
    packedScreenSpacePath: gpuScreenSpace?.path ?? 'none',
    packedScreenSpaceSummary: gpuScreenSpace?.summary ?? null,
    canvasWidth,
    canvasHeight
  };
}
