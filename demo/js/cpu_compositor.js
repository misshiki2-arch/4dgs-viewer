import { computeGaussianState, computeScreenSplat, clamp } from './rot4d_math.js';
import { evalSHColor } from './sh_eval.js';

export async function renderCpuComposite({
  raw,
  ctx,
  canvas,
  camera,
  controls,
  state,
  ui,
  tokenRef,
  infoEl
}) {
  if (!raw) return;

  state.rendering = true;
  const myToken = ++tokenRef.value;
  const t0 = performance.now();

  const renderScale = parseFloat(ui.renderScaleSlider.value);
  const renderW = Math.max(1, Math.round(canvas.width * renderScale));
  const renderH = Math.max(1, Math.round(canvas.height * renderScale));

  const bg = parseInt(ui.bgGraySlider.value, 10) / 255.0;
  const bg255 = parseInt(ui.bgGraySlider.value, 10);
  const stride = parseInt(ui.strideSlider.value, 10);
  const maxVisible = parseInt(ui.maxVisibleSlider.value, 10);
  const timestamp = parseFloat(ui.timeSlider.value);
  const scalingModifier = parseFloat(ui.splatScaleSlider.value);
  const useSH = ui.useSHCheck.checked;
  const useRot4d = ui.useRot4dCheck.checked;
  const forceSh3d = ui.forceSh3dCheck.checked;
  const timeDuration = parseFloat(ui.timeDurationSlider.value);

  controls.update();
  camera.updateMatrixWorld(true);

  const visible = [];
  const camPos = camera.position.clone();

  for (let i = 0; i < raw.N; i += stride) {
    const gs = computeGaussianState(raw, i, timestamp, scalingModifier, useRot4d);
    if (!gs) continue;

    const color = evalSHColor(
      raw,
      i,
      camPos,
      gs.pos,
      timestamp,
      timeDuration,
      useSH,
      forceSh3d
    );

    const splat = computeScreenSplat(camera, gs.pos, gs.cov3, gs.opacity, renderW, renderH);
    if (!splat) continue;

    visible.push({
      px: splat.px,
      py: splat.py,
      conic: splat.conic,
      radius: splat.radius,
      depth: splat.depth,
      opacity: splat.opacity,
      color
    });

    if (visible.length >= maxVisible) break;

    if ((visible.length & 1023) === 0 && performance.now() - t0 > 20) {
      await new Promise(r => setTimeout(r, 0));
      if (myToken !== tokenRef.value) {
        state.rendering = false;
        return;
      }
    }
  }

  visible.sort((a, b) => a.depth - b.depth);

  const pixelCount = renderW * renderH;
  const Tbuf = new Float32Array(pixelCount);
  Tbuf.fill(1.0);
  const Cbuf = new Float32Array(pixelCount * 3);

  let processed = 0;
  for (const s of visible) {
    const minX = Math.max(0, Math.floor(s.px - s.radius));
    const maxX = Math.min(renderW - 1, Math.ceil(s.px + s.radius));
    const minY = Math.max(0, Math.floor(s.py - s.radius));
    const maxY = Math.min(renderH - 1, Math.ceil(s.py + s.radius));
    const [cx, cy, cz] = s.conic;

    for (let y = minY; y <= maxY; y++) {
      const dy = s.py - y;
      let idx = y * renderW + minX;

      for (let x = minX; x <= maxX; x++, idx++) {
        const T = Tbuf[idx];
        if (T < 1e-4) continue;

        const dx = s.px - x;
        const power = -0.5 * (cx * dx * dx + cz * dy * dy) - cy * dx * dy;
        if (power > 0.0) continue;

        const alpha = Math.min(0.99, s.opacity * Math.exp(power));
        if (alpha < 1.0 / 255.0) continue;

        const w = alpha * T;
        const base = idx * 3;
        Cbuf[base + 0] += s.color[0] * w;
        Cbuf[base + 1] += s.color[1] * w;
        Cbuf[base + 2] += s.color[2] * w;
        Tbuf[idx] = T * (1.0 - alpha);
      }
    }

    processed++;
    if ((processed & 255) === 0) {
      await new Promise(r => setTimeout(r, 0));
      if (myToken !== tokenRef.value) {
        state.rendering = false;
        return;
      }
    }
  }

  const img = ctx.createImageData(renderW, renderH);
  const data = img.data;

  for (let p = 0; p < pixelCount; p++) {
    const T = Tbuf[p];
    const base = p * 3;
    const q = p * 4;

    data[q + 0] = Math.round(clamp(Cbuf[base + 0] + T * bg, 0, 1) * 255);
    data[q + 1] = Math.round(clamp(Cbuf[base + 1] + T * bg, 0, 1) * 255);
    data[q + 2] = Math.round(clamp(Cbuf[base + 2] + T * bg, 0, 1) * 255);
    data[q + 3] = 255;
  }

  const tmp = document.createElement('canvas');
  tmp.width = renderW;
  tmp.height = renderH;
  tmp.getContext('2d').putImageData(img, 0, 0);

  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = `rgb(${bg255},${bg255},${bg255})`;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
  ctx.restore();

  const elapsed = performance.now() - t0;
  infoEl.textContent =
`format=v2
N=${raw.N.toLocaleString()}  visible=${visible.length.toLocaleString()}  stride=${stride}
active_sh_degree=${raw.activeShDegree}  active_sh_degree_t=${raw.activeShDegreeT}
rot_4d(file)=${raw.rot4d}  useRot4d=${useRot4d}  useSH=${useSH}
renderScale=${renderScale.toFixed(2)}  canvas=${renderW}x${renderH}
time=${timestamp.toFixed(2)}  splatScale=${scalingModifier.toFixed(2)}
CPU T-composite render=${elapsed.toFixed(1)} ms
本家の renderCUDA にある front-to-back の T 合成に寄せた CPU 版です。`;

  state.rendering = false;
}
