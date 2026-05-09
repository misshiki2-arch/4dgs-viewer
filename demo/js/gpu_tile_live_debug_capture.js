import {
  computeFramebufferDeltaSummary,
  readFramebufferPixelRgb as readFramebufferPixelRgbForCanvas
} from './gpu_framebuffer_debug_utils.js';

function toFiniteNumberOrNull(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function cloneNumberArray(values, length = null) {
  if (!Array.isArray(values) && !(values instanceof Float32Array)) return null;
  const count = Number.isFinite(length) ? Math.min(values.length, length | 0) : values.length;
  const out = [];
  for (let i = 0; i < count; i++) out.push(toFiniteNumberOrNull(values[i]));
  return out;
}

function clampIntForDebug(value, min, max) {
  const n = Number(value);
  if (!Number.isFinite(n)) return min;
  return Math.min(max, Math.max(min, n | 0));
}

function computeRasterRectFromCenterRadius(centerPx, radiusPx) {
  if (!Array.isArray(centerPx) || centerPx.length < 2 || !Number.isFinite(radiusPx)) return null;
  const minX = Math.floor(centerPx[0] - radiusPx);
  const minY = Math.floor(centerPx[1] - radiusPx);
  const maxXExclusive = Math.floor(centerPx[0] + radiusPx) + 1;
  const maxYExclusive = Math.floor(centerPx[1] + radiusPx) + 1;
  return [minX, minY, maxXExclusive, maxYExclusive];
}

function decodePackedPayloadItem(packed, itemIndex, floatsPerItem = 16) {
  const stride = Number.isFinite(floatsPerItem) ? Math.max(16, floatsPerItem | 0) : 16;
  const index = Number.isFinite(itemIndex) ? itemIndex | 0 : -1;
  const base = index * stride;
  if (!(packed instanceof Float32Array) || index < 0 || base + 16 > packed.length) return null;
  const centerPx = [packed[base + 0], packed[base + 1]];
  const radius = packed[base + 2];
  const depth = packed[base + 3];
  const colorAlpha = [packed[base + 4], packed[base + 5], packed[base + 6], packed[base + 7]];
  const conic = [packed[base + 8], packed[base + 9], packed[base + 10]];
  const misc = [packed[base + 12], packed[base + 13], packed[base + 14], packed[base + 15]];
  const miscLooksLikeRect = misc.every(Number.isFinite) && misc[2] >= misc[0] && misc[3] >= misc[1];
  return {
    centerPx,
    radius,
    radiusPx: radius,
    depth,
    colorAlpha,
    color: colorAlpha.slice(0, 3),
    alpha: colorAlpha[3],
    opacity: colorAlpha[3],
    conic,
    reserved: packed[base + 11],
    misc,
    rasterRect: miscLooksLikeRect ? misc : computeRasterRectFromCenterRadius(centerPx, radius),
    rasterRectFromCenterRadius: computeRasterRectFromCenterRadius(centerPx, radius),
    packBaseFloatOffset: base,
    floatsPerItem: stride
  };
}

function normalizeDebugPixelList(inputPixels) {
  if (!Array.isArray(inputPixels)) return [];
  const pixels = [];
  const seen = new Set();
  for (const entry of inputPixels) {
    const x = Array.isArray(entry) ? Number(entry[0]) : Number(entry?.x ?? entry?.[0]);
    const y = Array.isArray(entry) ? Number(entry[1]) : Number(entry?.y ?? entry?.[1]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const pixel = [Math.floor(x), Math.floor(y)];
    const key = `${pixel[0]},${pixel[1]}`;
    if (seen.has(key)) continue;
    seen.add(key);
    pixels.push({
      pixel,
      source: entry?.source ?? 'requested',
      representativeIndex: Number.isFinite(Number(entry?.representativeIndex))
        ? Number(entry.representativeIndex) | 0
        : null
    });
  }
  return pixels;
}

function buildRepresentativeCompareMap(compareResult) {
  const map = new Map();
  for (const item of Array.isArray(compareResult?.items) ? compareResult.items : []) {
    const index = Number(item?.index);
    if (Number.isFinite(index)) map.set(index | 0, item);
  }
  return map;
}

function selectDefaultAccumulationDebugPixels(compareResult, maxPixels = 3) {
  const items = Array.isArray(compareResult?.items) ? compareResult.items : [];
  return items
    .filter((item) => item?.found && Array.isArray(item.actualPayload?.centerPx))
    .map((item) => {
      const radius = Number.isFinite(item.actualPayload?.radius) ? Math.max(1, Number(item.actualPayload.radius)) : 1;
      const alpha = Number.isFinite(item.actualPayload?.alpha) ? Math.max(0, Number(item.actualPayload.alpha)) : 0;
      const occurrenceCount = Number.isFinite(item.accumulationOccurrenceCount) ? Math.max(1, item.accumulationOccurrenceCount | 0) : 1;
      return { item, score: radius * alpha * occurrenceCount };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.max(1, maxPixels | 0))
    .map(({ item }) => ({
      pixel: [Math.floor(item.actualPayload.centerPx[0]), Math.floor(item.actualPayload.centerPx[1])],
      source: 'representative-high-overlap-alpha-radius',
      representativeIndex: Number(item.index) | 0
    }));
}

function clonePayloadForAssociationDebug(payload) {
  if (!payload) return null;
  return {
    centerPx: cloneNumberArray(payload.centerPx, 2),
    depth: toFiniteNumberOrNull(payload.depth),
    conic: cloneNumberArray(payload.conic, 3),
    radius: toFiniteNumberOrNull(payload.radius),
    radiusPx: toFiniteNumberOrNull(payload.radiusPx),
    rasterRect: cloneNumberArray(payload.rasterRect, 4),
    opacity: toFiniteNumberOrNull(payload.opacity),
    alpha: toFiniteNumberOrNull(payload.alpha),
    color: cloneNumberArray(payload.color, 3),
    colorAlpha: cloneNumberArray(payload.colorAlpha, 4),
    stateConvention: typeof payload.stateConvention === 'string' ? payload.stateConvention : null,
    usedCuda4DStateHelper: typeof payload.usedCuda4DStateHelper === 'boolean' ? payload.usedCuda4DStateHelper : null,
    stateHelperVersion: typeof payload.stateHelperVersion === 'string'
      ? payload.stateHelperVersion
      : (typeof payload.helperVersion === 'string' ? payload.helperVersion : null)
  };
}

function computeMaxAbsDelta(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return null;
  let maxAbs = 0;
  for (let i = 0; i < a.length; i++) {
    const av = Number(a[i]);
    const bv = Number(b[i]);
    if (!Number.isFinite(av) || !Number.isFinite(bv)) return null;
    maxAbs = Math.max(maxAbs, Math.abs(av - bv));
  }
  return maxAbs;
}

function buildAssociationDelta(packedPayload, recomputedPayload) {
  if (!packedPayload || !recomputedPayload) {
    return {
      valid: false,
      reason: packedPayload ? 'recomputed-payload-missing' : 'packed-payload-missing'
    };
  }
  const centerPxMaxAbs = computeMaxAbsDelta(packedPayload.centerPx, recomputedPayload.centerPx);
  const conicMaxAbs = computeMaxAbsDelta(packedPayload.conic, recomputedPayload.conic);
  const colorMaxAbs = computeMaxAbsDelta(packedPayload.color, recomputedPayload.color);
  const rasterRectMaxAbs = computeMaxAbsDelta(packedPayload.rasterRect, recomputedPayload.rasterRect);
  const depthAbs = Number.isFinite(packedPayload.depth) && Number.isFinite(recomputedPayload.depth)
    ? Math.abs(packedPayload.depth - recomputedPayload.depth)
    : null;
  const radiusAbs = Number.isFinite(packedPayload.radius) && Number.isFinite(recomputedPayload.radius)
    ? Math.abs(packedPayload.radius - recomputedPayload.radius)
    : null;
  const opacityAbs = Number.isFinite(packedPayload.alpha) && Number.isFinite(recomputedPayload.alpha)
    ? Math.abs(packedPayload.alpha - recomputedPayload.alpha)
    : null;
  const mismatches = {
    centerPx: Number.isFinite(centerPxMaxAbs) && centerPxMaxAbs > 1e-3,
    depth: Number.isFinite(depthAbs) && depthAbs > 1e-3,
    conic: Number.isFinite(conicMaxAbs) && conicMaxAbs > 1e-5,
    radius: Number.isFinite(radiusAbs) && radiusAbs > 0,
    opacity: Number.isFinite(opacityAbs) && opacityAbs > 1e-5,
    color: Number.isFinite(colorMaxAbs) && colorMaxAbs > 1e-5,
    rasterRect: Number.isFinite(rasterRectMaxAbs) && rasterRectMaxAbs > 0
  };
  return {
    valid: true,
    centerPxMaxAbs,
    depthAbs,
    conicMaxAbs,
    radiusAbs,
    opacityAbs,
    colorMaxAbs,
    rasterRectMaxAbs,
    mismatches,
    anyMismatch: Object.values(mismatches).some(Boolean)
  };
}

function computeDepthOrderingSummaryForBatch(batch) {
  const packedCount = Number.isFinite(batch?.packedCount) ? Math.max(0, batch.packedCount | 0) : 0;
  let previousDepth = -Infinity;
  let mismatchCount = 0;
  let firstMismatch = null;
  const depthPreviewHead = [];
  const depthPreviewTail = [];
  for (let i = 0; i < packedCount; i++) {
    const payload = decodePackedPayloadItem(batch?.packed, i, batch?.floatsPerItem);
    const depth = Number(payload?.depth);
    if (i < 8) depthPreviewHead.push(depth);
    if (i >= Math.max(0, packedCount - 8)) depthPreviewTail.push(depth);
    if (Number.isFinite(depth) && Number.isFinite(previousDepth) && depth < previousDepth) {
      mismatchCount++;
      if (!firstMismatch) firstMismatch = { localOrder: i, previousDepth, currentDepth: depth };
    }
    if (Number.isFinite(depth)) previousDepth = depth;
  }
  return {
    depthOrder: 'ascending-near-to-far',
    sequenceConsistent: mismatchCount === 0,
    orderingMismatchCount: mismatchCount,
    firstMismatch,
    depthPreviewHead,
    depthPreviewTail
  };
}

function evaluateAccumulationPayloadAtPixel(payload, pixel) {
  const dx = payload.centerPx[0] - pixel[0];
  const dy = payload.centerPx[1] - pixel[1];
  const power = -0.5 * (payload.conic[0] * dx * dx + payload.conic[2] * dy * dy) - payload.conic[1] * dx * dy;
  const rawAlpha = payload.alpha * Math.exp(power);
  const computedAlpha = Math.min(0.99, rawAlpha);
  let skipReason = 'none';
  if (power > 0.0) skipReason = 'power-positive';
  else if (computedAlpha < (1.0 / 255.0)) skipReason = 'alpha-below-1-over-255';
  return {
    dx,
    dy,
    power,
    rawAlpha,
    computedAlpha,
    survivesPower: power <= 0.0,
    survivesAlphaThreshold: computedAlpha >= (1.0 / 255.0),
    skipReason
  };
}

function summarizeAssociationMismatchReasons(mismatches) {
  const out = {};
  for (const mismatch of mismatches) {
    let reason = 'value-delta';
    if (mismatch?.recomputedFromOriginalSplatIndex?.ok === false) {
      reason = mismatch.recomputedFromOriginalSplatIndex.reason ?? 'recompute-failed';
    } else if (mismatch?.delta?.mismatches) {
      reason = Object.entries(mismatch.delta.mismatches)
        .filter(([, value]) => !!value)
        .map(([key]) => key)
        .join('+') || 'value-delta';
    }
    out[reason] = (out[reason] ?? 0) + 1;
  }
  return out;
}

function getVisibleCountFromDebugResult(debugResult) {
  const retention = debugResult?.actualPayloadRetentionSummary;
  if (Number.isFinite(retention?.visibleCount)) return retention.visibleCount;
  const attempts = Array.isArray(debugResult?.debugRenderAttempts) ? debugResult.debugRenderAttempts : [];
  for (let i = attempts.length - 1; i >= 0; i--) {
    const visibleCount = attempts[i]?.retentionSummary?.visibleCount;
    if (Number.isFinite(visibleCount)) return visibleCount;
  }
  return null;
}

function findOriginalSplatOccurrencesInTileDebug(tileDebug, index) {
  const entries = tileDebug?.pixels?.[0]?.accumulation?.entries;
  if (!Array.isArray(entries)) return [];
  return entries
    .filter((entry) => Number.isFinite(entry?.originalSplatIndex) && (entry.originalSplatIndex | 0) === (index | 0))
    .map((entry) => ({
      localOrder: entry.localOrder ?? null,
      slot: entry.localOrder ?? null,
      packedIndex: entry.packedIndex ?? null,
      sourceVisibleIndex: entry.sourceVisibleIndex ?? null,
      originalSplatIndex: entry.originalSplatIndex ?? null,
      depth: entry.depth ?? null,
      centerPx: cloneNumberArray(entry.centerPx, 2),
      conic: cloneNumberArray(entry.conic, 3),
      radius: entry.radius ?? null,
      opacity: entry.opacity ?? null,
      computedAlpha: entry.computedAlpha ?? null,
      skipReason: entry.skipReason ?? null
    }));
}

function summarizeSlotIndexAgreement(tileDebug, associationDebug) {
  const tileEntries = tileDebug?.pixels?.[0]?.accumulation?.entries;
  const associationEntries = associationDebug?.entries;
  if (!Array.isArray(tileEntries) || !Array.isArray(associationEntries)) {
    return { comparable: false, reason: 'entries-missing' };
  }
  const associationByLocalOrder = new Map();
  for (const entry of associationEntries) {
    if (Number.isFinite(entry?.localOrder)) associationByLocalOrder.set(entry.localOrder | 0, entry);
  }
  let comparedCount = 0;
  let mismatchCount = 0;
  let firstMismatch = null;
  for (const tileEntry of tileEntries) {
    if (!Number.isFinite(tileEntry?.localOrder)) continue;
    const associationEntry = associationByLocalOrder.get(tileEntry.localOrder | 0);
    if (!associationEntry) continue;
    comparedCount++;
    const tileOriginal = Number.isFinite(tileEntry?.originalSplatIndex) ? (tileEntry.originalSplatIndex | 0) : null;
    const associationOriginal = Number.isFinite(associationEntry?.originalSplatIndex) ? (associationEntry.originalSplatIndex | 0) : null;
    if (tileOriginal !== associationOriginal) {
      mismatchCount++;
      if (!firstMismatch) {
        firstMismatch = {
          localOrder: tileEntry.localOrder | 0,
          tileOriginalSplatIndex: tileOriginal,
          associationOriginalSplatIndex: associationOriginal,
          tileSourceVisibleIndex: tileEntry.sourceVisibleIndex ?? null,
          associationSourceVisibleIndex: associationEntry.sourceVisibleIndex ?? null
        };
      }
    }
  }
  return { comparable: true, comparedCount, mismatchCount, firstMismatch };
}

export function createGpuTileLiveDebugCapture(deps) {
  const getRaw = deps.getRaw;
  const getCanvas = deps.getCanvas;
  const getCamera = deps.getCamera;
  const getUi = deps.getUi;
  const getGpu = deps.getGpu;
  const getLatestRenderResult = deps.getLatestRenderResult;

  function getTileGridSummaryFromRenderResult(renderResult) {
    const canvas = getCanvas();
    const summary = renderResult?.tileCompositePlan?.summary ?? renderResult?.tileSummary ?? null;
    const tileSize = Number.isFinite(summary?.tileCompositeTileSize)
      ? Math.max(1, summary.tileCompositeTileSize | 0)
      : (Number.isFinite(summary?.tileSize) ? Math.max(1, summary.tileSize | 0) : 32);
    const canvasWidth = Number.isFinite(canvas?.width) ? canvas.width | 0 : 0;
    const canvasHeight = Number.isFinite(canvas?.height) ? canvas.height | 0 : 0;
    const tileCols = Number.isFinite(summary?.tileCompositeTileCols)
      ? Math.max(1, summary.tileCompositeTileCols | 0)
      : Math.max(1, Math.ceil(canvasWidth / tileSize));
    const tileRows = Number.isFinite(summary?.tileCompositeTileRows)
      ? Math.max(1, summary.tileCompositeTileRows | 0)
      : Math.max(1, Math.ceil(canvasHeight / tileSize));
    return { tileSize, tileCols, tileRows, canvasWidth, canvasHeight };
  }

  function findTileBatchForPixel(renderResult, pixel) {
    const grid = getTileGridSummaryFromRenderResult(renderResult);
    const x = Math.floor(pixel?.[0] ?? -1);
    const y = Math.floor(pixel?.[1] ?? -1);
    if (x < 0 || y < 0 || x >= grid.canvasWidth || y >= grid.canvasHeight) {
      return { ok: false, reason: 'pixel-out-of-bounds', grid, tileId: -1, tile: [-1, -1], batch: null };
    }
    const tx = Math.min(grid.tileCols - 1, Math.floor(x / grid.tileSize));
    const ty = Math.min(grid.tileRows - 1, Math.floor(y / grid.tileSize));
    const tileId = ty * grid.tileCols + tx;
    const batches = Array.isArray(renderResult?.tileCompositePlan?.batches) ? renderResult.tileCompositePlan.batches : [];
    const batch = batches.find((candidate) => (candidate?.tileId | 0) === tileId) ?? null;
    return {
      ok: !!batch,
      reason: batch ? 'ok' : 'tile-batch-not-found',
      grid,
      tileId,
      tile: [tx, ty],
      tileBounds: batch?.rect ? cloneNumberArray(batch.rect, 4) : [
        tx * grid.tileSize,
        ty * grid.tileSize,
        Math.min(grid.canvasWidth, (tx + 1) * grid.tileSize),
        Math.min(grid.canvasHeight, (ty + 1) * grid.tileSize)
      ],
      batch
    };
  }

  function buildDebugScreenSpaceCameraProxy() {
    const camera = getCamera();
    const deterministicState = deps.buildDeterministicStateSummary();
    const cudaAligned = deterministicState?.cudaAlignedScreenSpaceCamera;
    if (!cudaAligned?.enabled || !Array.isArray(cudaAligned.cudaAlignedViewMatrix)) return null;
    return {
      fov: camera.fov,
      aspect: camera.aspect,
      matrixWorldInverse: camera.matrixWorldInverse,
      projectionMatrix: camera.projectionMatrix,
      updateMatrixWorld: () => {},
      screenSpaceTransformOverride: {
        mode: 'cuda-aligned',
        viewMatrixSource: cudaAligned.viewMatrixSource ?? 'dataset-transform-cuda-reader-c2w-inverse',
        viewMatrix: cudaAligned.cudaAlignedViewMatrix,
        fov: camera.fov,
        aspect: camera.aspect,
        intrinsics: cudaAligned.intrinsics ?? null,
        covarianceTanFovX: Number.isFinite(cudaAligned.covarianceTanFovX) ? Number(cudaAligned.covarianceTanFovX) : null,
        covarianceTanFovY: Number.isFinite(cudaAligned.covarianceTanFovY) ? Number(cudaAligned.covarianceTanFovY) : null,
        covarianceFocalContract: cudaAligned.covarianceFocalContract ?? null,
        pixelXSign: [-1, 1].includes(cudaAligned.pixelXSign) ? Number(cudaAligned.pixelXSign) : 1,
        signConversionMatrix: cudaAligned.signConversionMatrix ?? null,
        basis: cudaAligned.cudaAlignedCameraBasis ?? null,
        projectionContract: cudaAligned.projectionContract ?? 'cuda-plus-z-forward-fx-fy-cx-cy',
        screenYSign: Number.isFinite(cudaAligned.screenYSign) ? Number(cudaAligned.screenYSign) : 1,
        depthSign: Number.isFinite(cudaAligned.depthSign) ? Number(cudaAligned.depthSign) : 1
      }
    };
  }

  function recomputePayloadForOriginalSplatIndex(originalSplatIndex) {
    const raw = getRaw();
    const canvas = getCanvas();
    const camera = getCamera();
    const ui = getUi();
    const index = Number(originalSplatIndex);
    if (!raw || !Number.isFinite(index) || index < 0 || index >= raw.N) {
      return {
        ok: false,
        reason: 'raw-missing-or-index-out-of-range',
        originalSplatIndex: Number.isFinite(index) ? (index | 0) : null
      };
    }

    const renderScale = Number.isFinite(Number(ui.renderScaleSlider?.value)) ? Number(ui.renderScaleSlider.value) : 1;
    const renderW = Math.max(1, Math.round(canvas.width * renderScale));
    const renderH = Math.max(1, Math.round(canvas.height * renderScale));
    const sx = canvas.width / renderW;
    const sy = canvas.height / renderH;
    const timestamp = Number.isFinite(Number(ui.timeSlider?.value)) ? Number(ui.timeSlider.value) : 0;
    const scalingModifier = Number.isFinite(Number(ui.splatScaleSlider?.value)) ? Number(ui.splatScaleSlider.value) : 1;
    const sigmaScale = Number.isFinite(Number(ui.sigmaScaleSlider?.value)) ? Number(ui.sigmaScaleSlider.value) : 1;
    const prefilterVar = Number.isFinite(Number(ui.prefilterVarSlider?.value)) ? Number(ui.prefilterVarSlider.value) : 0;
    const useRot4d = !!ui.useRot4dCheck?.checked;
    const flags = {
      nativeRot4d: !!ui.useNativeRot4dCheck?.checked,
      nativeMarginal: !!ui.useNativeMarginalCheck?.checked
    };

    const gs = deps.computeGaussianState(raw, index | 0, timestamp, scalingModifier, sigmaScale, prefilterVar, useRot4d, flags);
    if (!gs) {
      return { ok: false, reason: 'computeGaussianState-culled-or-null', originalSplatIndex: index | 0, timestamp, flags, useRot4d };
    }

    const color = deps.evalSHColor(
      raw,
      index | 0,
      camera.position.clone(),
      gs.pos,
      timestamp,
      Number.isFinite(Number(ui.timeDurationSlider?.value)) ? Number(ui.timeDurationSlider.value) : 33,
      !!ui.useSHCheck?.checked,
      !!ui.forceSh3dCheck?.checked
    );
    const screenSpaceCamera = buildDebugScreenSpaceCameraProxy();
    const splat = deps.computeScreenSplat(screenSpaceCamera || camera, gs.pos, gs.cov3, gs.opacity, renderW, renderH);
    if (!splat) {
      return { ok: false, reason: 'computeScreenSplat-culled-or-null', originalSplatIndex: index | 0, timestamp, flags, useRot4d };
    }

    const px = splat.px * sx;
    const py = splat.py * sy;
    const radius = splat.radius * Math.max(sx, sy);
    const coverageRadius = Math.max(1, radius);
    const rasterRect = [
      clampIntForDebug(Math.floor(px - coverageRadius), 0, canvas.width - 1),
      clampIntForDebug(Math.floor(py - coverageRadius), 0, canvas.height - 1),
      clampIntForDebug(Math.ceil(px + coverageRadius), 0, canvas.width - 1),
      clampIntForDebug(Math.ceil(py + coverageRadius), 0, canvas.height - 1)
    ];
    const conic = [splat.conic[0] / (sx * sx), splat.conic[1] / (sx * sy), splat.conic[2] / (sy * sy)];
    return {
      ok: true,
      reason: 'ok',
      originalSplatIndex: index | 0,
      timestamp,
      flags,
      useRot4d,
      source: 'raw-recompute-current-viewer-settings',
      stateConvention: gs.stateConvention ?? null,
      usedCuda4DStateHelper: !!gs.usedCuda4DStateHelper,
      stateHelperVersion: gs.helperVersion ?? null,
      centerPx: [px, py],
      depth: splat.depth,
      conic,
      radius,
      radiusPx: radius,
      rasterRect,
      opacity: splat.opacity,
      alpha: splat.opacity,
      color,
      colorAlpha: [color[0], color[1], color[2], splat.opacity]
    };
  }

  function simulateTileAccumulationAtPixel({ batch, visibleItems = [], pixel, bgGray01, representativeCompareMap, maxItems = 2048 }) {
    const packedCount = Number.isFinite(batch?.packedCount) ? Math.max(0, batch.packedCount | 0) : 0;
    const clampedCount = Math.min(packedCount, Math.max(1, Math.min(2048, maxItems | 0)));
    const sourceIndices = batch?.sourceIndices instanceof Uint32Array ? batch.sourceIndices : null;
    const orderedIndices = batch?.orderedIndices instanceof Uint32Array ? batch.orderedIndices : null;
    const entries = [];
    let T = 1.0;
    let accumColor = [0, 0, 0];
    let accumDepth = 0;
    let contributor = 0;
    let lastContributingLocalOrder = -1;
    let earlyOutTriggered = false;
    let earlyOutAtLocalOrder = -1;
    let alphaSum = 0;
    let contributionCount = 0;
    let powerSkipCount = 0;
    let alphaSkipCount = 0;

    for (let localOrder = 0; localOrder < clampedCount; localOrder++) {
      contributor++;
      const payload = decodePackedPayloadItem(batch?.packed, localOrder, batch?.floatsPerItem);
      if (!payload) {
        entries.push({ localOrder, skipReason: 'payload-decode-failed' });
        continue;
      }
      const originalSplatIndex = sourceIndices ? (sourceIndices[localOrder] | 0) : null;
      const representative = Number.isFinite(originalSplatIndex) ? representativeCompareMap.get(originalSplatIndex) : null;
      const evaluation = evaluateAccumulationPayloadAtPixel(payload, pixel);
      const TBefore = T;
      const testT = TBefore * (1.0 - evaluation.computedAlpha);
      let contributionRgb = [0, 0, 0];
      let accumColorAfter = [...accumColor];
      let TAfter = TBefore;
      let contributes = false;
      let skipReason = evaluation.skipReason;

      if (skipReason === 'power-positive') {
        powerSkipCount++;
      } else if (skipReason === 'alpha-below-1-over-255') {
        alphaSkipCount++;
      } else if (testT < 0.0001) {
        skipReason = 'early-out-testT-below-0.0001';
        earlyOutTriggered = true;
        earlyOutAtLocalOrder = localOrder;
      } else {
        contributionRgb = payload.color.map((value) => value * evaluation.computedAlpha * TBefore);
        accumColor = [
          accumColor[0] + contributionRgb[0],
          accumColor[1] + contributionRgb[1],
          accumColor[2] + contributionRgb[2]
        ];
        accumDepth += payload.depth * evaluation.computedAlpha * TBefore;
        accumColorAfter = [...accumColor];
        T = testT;
        TAfter = T;
        contributes = true;
        contributionCount++;
        alphaSum += evaluation.computedAlpha;
        lastContributingLocalOrder = localOrder;
      }

      const sourceVisibleIndex = orderedIndices ? (orderedIndices[localOrder] | 0) : null;
      const visibleItem = Number.isFinite(sourceVisibleIndex) ? visibleItems[sourceVisibleIndex] : null;
      entries.push({
        localOrder,
        packedIndex: localOrder,
        sourceVisibleIndex,
        originalSplatIndex,
        representative: !!representative,
        representativeReferenceMatchesActualPayload: representative?.summary?.payloadMatchesDebug ?? representative?.found ?? null,
        representativeAccumulationOccurrenceCount: representative?.accumulationOccurrenceCount ?? null,
        depth: payload.depth,
        centerPx: payload.centerPx,
        radius: payload.radius,
        rasterRect: payload.rasterRect,
        conic: payload.conic,
        opacity: payload.alpha,
        alpha: payload.alpha,
        color: payload.color,
        stateConvention: visibleItem?.stateConvention ?? null,
        usedCuda4DStateHelper: Number.isFinite(sourceVisibleIndex) ? !!visibleItem?.usedCuda4DStateHelper : null,
        stateHelperVersion: visibleItem?.stateHelperVersion ?? null,
        dx: evaluation.dx,
        dy: evaluation.dy,
        power: evaluation.power,
        rawAlpha: evaluation.rawAlpha,
        computedAlpha: evaluation.computedAlpha,
        skipReason,
        contributes,
        TBefore,
        test_T: testT,
        TAfter,
        contributionRgb,
        accumColorAfter,
        accumDepthAfter: accumDepth
      });

      if (earlyOutTriggered) break;
    }

    const bg = Number.isFinite(bgGray01) ? Math.max(0, Math.min(1, Number(bgGray01))) : 0;
    const finalRgb = [accumColor[0] + T * bg, accumColor[1] + T * bg, accumColor[2] + T * bg];
    return {
      pixel,
      packedCount,
      clampedCount,
      maxItemsEvaluated: clampedCount,
      sortOrderContract: 'ascending-depth-near-to-far-front-to-back',
      shaderEquationContract: 'matches gpu_tile_accumulation_executor fragment shader and CUDA forward.cu render loop',
      depthOrderingSummary: computeDepthOrderingSummaryForBatch(batch),
      contributorCounter: contributor,
      contributionCount,
      powerSkipCount,
      alphaSkipCount,
      alphaSum,
      lastContributingLocalOrder,
      earlyOutTriggered,
      earlyOutAtLocalOrder,
      finalT: T,
      accumColor,
      finalRgb,
      accumDepth,
      entries
    };
  }

  async function resolveDebugRender(options) {
    const renderResultOverride = deps.hasRetainedActualPayload(options.renderResultOverride) ? options.renderResultOverride : null;
    if (renderResultOverride) {
      return {
        renderResult: renderResultOverride,
        attempts: Array.isArray(options.sharedDebugRenderAttempts)
          ? options.sharedDebugRenderAttempts
          : [{ stage: 'shared-render-result-override', retentionSummary: deps.buildActualPayloadRetentionSummary(renderResultOverride) }]
      };
    }
    const latestRenderResult = getLatestRenderResult();
    if (options.ensureCurrentFrame === false && deps.hasRetainedActualPayload(latestRenderResult)) {
      return {
        renderResult: latestRenderResult,
        attempts: [{ stage: 'reuse-latest-render-result', retentionSummary: deps.buildActualPayloadRetentionSummary(latestRenderResult) }]
      };
    }
    return await deps.renderCurrentFrameForDebugPayload(options);
  }

  async function captureTileAccumulationDebug(input = {}) {
    const options = input ?? {};
    const representativeCompare = options.representativeCompare ?? options.compareResult ?? null;
    const requestedPixels = normalizeDebugPixelList(options.pixels);
    const pixels = requestedPixels.length > 0
      ? requestedPixels
      : selectDefaultAccumulationDebugPixels(representativeCompare, options.maxPixels ?? 3);
    const debugRender = await resolveDebugRender(options);
    const renderResult = debugRender.renderResult;
    const gl = getGpu()?.gl ?? null;
    const ui = getUi();
    const canvas = getCanvas();
    const bgGray01 = Number.isFinite(Number(ui.bgGraySlider?.value)) ? Number(ui.bgGraySlider.value) / 255.0 : 0;
    const representativeCompareMap = buildRepresentativeCompareMap(representativeCompare);

    const pixelResults = pixels.map((pixelSpec) => {
      const tileInfo = findTileBatchForPixel(renderResult, pixelSpec.pixel);
      if (!tileInfo.ok) {
        return {
          pixel: pixelSpec.pixel,
          pixelSource: pixelSpec.source,
          representativeIndex: pixelSpec.representativeIndex,
          ok: false,
          reason: tileInfo.reason,
          tileId: tileInfo.tileId,
          tile: tileInfo.tile,
          tileBounds: tileInfo.tileBounds ?? null,
          grid: tileInfo.grid
        };
      }
      const simulation = simulateTileAccumulationAtPixel({
        batch: tileInfo.batch,
        visibleItems: Array.isArray(renderResult?.visible) ? renderResult.visible : [],
        pixel: pixelSpec.pixel,
        bgGray01,
        representativeCompareMap,
        maxItems: options.maxItems ?? 2048
      });
      const framebuffer = readFramebufferPixelRgbForCanvas(gl, canvas, pixelSpec.pixel);
      const framebufferDeltaSummary = framebuffer.valid
        ? computeFramebufferDeltaSummary(framebuffer.rgb, simulation.finalRgb)
        : { delta: null, deltaAbsMax: null };
      return {
        pixel: pixelSpec.pixel,
        pixelSource: pixelSpec.source,
        representativeIndex: pixelSpec.representativeIndex,
        ok: true,
        tileId: tileInfo.tileId,
        tile: tileInfo.tile,
        tileBounds: tileInfo.tileBounds,
        grid: tileInfo.grid,
        tilePayloadCount: Number.isFinite(tileInfo.batch?.packedCount) ? tileInfo.batch.packedCount : 0,
        batchPackedCount: Number.isFinite(tileInfo.batch?.packedCount) ? tileInfo.batch.packedCount : 0,
        batchFloatsPerItem: Number.isFinite(tileInfo.batch?.floatsPerItem) ? tileInfo.batch.floatsPerItem : 16,
        batchHasSourceIndices: tileInfo.batch?.sourceIndices instanceof Uint32Array,
        batchHasOrderedIndices: tileInfo.batch?.orderedIndices instanceof Uint32Array,
        accumulation: simulation,
        framebuffer,
        framebufferDelta: framebufferDeltaSummary.delta,
        framebufferDeltaAbsMax: framebufferDeltaSummary.deltaAbsMax
      };
    });

    return {
      schemaVersion: 'step90-tile-accumulation-debug-v1',
      timestamp: new Date().toISOString(),
      purpose: 'CPU/JS replay of Viewer tile accumulation shader inputs/order/equations for selected pixels',
      cudaEquationSummary: {
        order: 'per tile sorted by ascending depth key, front-to-back',
        power: '-0.5 * (conic.x * dx^2 + conic.z * dy^2) - conic.y * dx * dy',
        alpha: 'min(0.99, opacity * exp(power))',
        skip: 'continue if power > 0 or alpha < 1/255',
        testT: 'T * (1 - alpha)',
        earlyOut: 'if testT < 0.0001, stop without contributing crossing splat',
        color: 'C += rgb * alpha * T; final = C + T * bg'
      },
      viewerEquationSummary: {
        order: 'tileCompositePlan.batches[].packed sorted by ascending depth',
        bgGray01,
        premultipliedAlpha: false,
        outputAlpha: 1
      },
      selectedPixels: pixels,
      actualPayloadRetentionSummary: deps.buildActualPayloadRetentionSummary(renderResult),
      debugRenderAttempts: debugRender.attempts,
      lastRenderResultSummary: deps.buildRenderResultInspectionSummary(renderResult),
      representativeCompareSummary: representativeCompare?.summary ?? null,
      pixels: pixelResults
    };
  }

  async function captureViewerPayloadIndexAssociationDebug(input = {}) {
    const options = input ?? {};
    const targetPixel = Array.isArray(options.pixel) && options.pixel.length >= 2
      ? [Number(options.pixel[0]) | 0, Number(options.pixel[1]) | 0]
      : [655, 363];
    const targetIndices = Array.isArray(options.indices) && options.indices.length > 0
      ? options.indices.map((value) => Number(value)).filter(Number.isFinite).map((value) => value | 0)
      : [2765070, 1182029, 2718004];
    const targetIndexSet = new Set(targetIndices);
    const maxEntries = Number.isFinite(Number(options.maxEntries)) ? Math.max(1, Number(options.maxEntries) | 0) : 2048;
    const debugRender = await resolveDebugRender(options);
    const renderResult = debugRender.renderResult;
    const tileInfo = findTileBatchForPixel(renderResult, targetPixel);
    const visible = Array.isArray(renderResult?.visible) ? renderResult.visible : [];
    const entries = [];
    const mismatches = [];
    const targetOccurrences = new Map(targetIndices.map((index) => [index, []]));

    if (tileInfo.ok) {
      const batch = tileInfo.batch;
      const packedCount = Number.isFinite(batch?.packedCount) ? Math.max(0, batch.packedCount | 0) : 0;
      const clampedCount = Math.min(packedCount, maxEntries);
      const sourceIndices = batch?.sourceIndices instanceof Uint32Array ? batch.sourceIndices : null;
      const orderedIndices = batch?.orderedIndices instanceof Uint32Array ? batch.orderedIndices : null;

      for (let localOrder = 0; localOrder < clampedCount; localOrder++) {
        const sourceVisibleIndex = orderedIndices ? (orderedIndices[localOrder] | 0) : null;
        const originalSplatIndex = sourceIndices ? (sourceIndices[localOrder] | 0) : null;
        const visibleItem = Number.isFinite(sourceVisibleIndex) ? visible[sourceVisibleIndex] : null;
        const packedPayload = decodePackedPayloadItem(batch?.packed, localOrder, batch?.floatsPerItem);
        const recomputed = Number.isFinite(originalSplatIndex)
          ? recomputePayloadForOriginalSplatIndex(originalSplatIndex)
          : { ok: false, reason: 'missing-originalSplatIndex' };
        const recomputedPayload = recomputed?.ok ? clonePayloadForAssociationDebug(recomputed) : null;
        const packedSlim = clonePayloadForAssociationDebug(packedPayload);
        const delta = buildAssociationDelta(packedSlim, recomputedPayload);
        const sourceVisibleConsistency = {
          hasVisibleItem: !!visibleItem,
          visibleItemSrcIndex: Number.isFinite(visibleItem?.srcIndex) ? (visibleItem.srcIndex | 0) : null,
          visibleItemStateConvention: typeof visibleItem?.stateConvention === 'string' ? visibleItem.stateConvention : null,
          visibleItemUsedCuda4DStateHelper: typeof visibleItem?.usedCuda4DStateHelper === 'boolean' ? visibleItem.usedCuda4DStateHelper : null,
          visibleItemStateHelperVersion: typeof visibleItem?.stateHelperVersion === 'string' ? visibleItem.stateHelperVersion : null,
          matchesOriginalSplatIndex:
            Number.isFinite(visibleItem?.srcIndex) &&
            Number.isFinite(originalSplatIndex) &&
            (visibleItem.srcIndex | 0) === (originalSplatIndex | 0)
        };
        const evaluation = packedPayload ? evaluateAccumulationPayloadAtPixel(packedPayload, targetPixel) : null;
        const entry = {
          localOrder,
          slot: localOrder,
          packedIndex: localOrder,
          sourceVisibleIndex,
          originalSplatIndex,
          targetIndex: Number.isFinite(originalSplatIndex) && targetIndexSet.has(originalSplatIndex),
          sourceVisibleConsistency,
          packedPayload: packedSlim,
          recomputedFromOriginalSplatIndex: recomputed?.ok
            ? recomputedPayload
            : {
                ok: false,
                reason: recomputed?.reason ?? 'recompute-failed',
                originalSplatIndex,
                timestamp: recomputed?.timestamp ?? null,
                flags: recomputed?.flags ?? null,
                useRot4d: recomputed?.useRot4d ?? null
              },
          delta,
          pixelEvaluation: evaluation
            ? {
                dx: evaluation.dx,
                dy: evaluation.dy,
                power: evaluation.power,
                rawAlpha: evaluation.rawAlpha,
                computedAlpha: evaluation.computedAlpha,
                skipReason: evaluation.skipReason
              }
            : null,
          cullShouldNotContribute: recomputed && !recomputed.ok && evaluation?.skipReason === 'none',
          associationMismatch: !!delta?.anyMismatch || (recomputed && !recomputed.ok)
        };
        if (entry.associationMismatch) mismatches.push(entry);
        if (Number.isFinite(originalSplatIndex) && targetOccurrences.has(originalSplatIndex)) {
          targetOccurrences.get(originalSplatIndex).push(entry);
        }
        if (options.includeAllEntries === true || localOrder < 64 || entry.targetIndex || entry.associationMismatch) entries.push(entry);
      }
    }

    for (const index of targetIndices) {
      if ((targetOccurrences.get(index)?.length ?? 0) > 0) continue;
      const recomputed = recomputePayloadForOriginalSplatIndex(index);
      targetOccurrences.set(index, [{
        localOrder: null,
        slot: null,
        packedIndex: null,
        sourceVisibleIndex: null,
        originalSplatIndex: index,
        targetIndex: true,
        sourceVisibleConsistency: {
          hasVisibleItem: false,
          visibleItemSrcIndex: null,
          matchesOriginalSplatIndex: false
        },
        packedPayload: null,
        recomputedFromOriginalSplatIndex: recomputed?.ok
          ? clonePayloadForAssociationDebug(recomputed)
          : {
              ok: false,
              reason: recomputed?.reason ?? 'recompute-failed',
              originalSplatIndex: index,
              timestamp: recomputed?.timestamp ?? null,
              flags: recomputed?.flags ?? null,
              useRot4d: recomputed?.useRot4d ?? null
            },
        delta: { valid: false, reason: 'target-index-not-present-in-target-tile-batch' },
        pixelEvaluation: null,
        cullShouldNotContribute: false,
        associationMismatch: false
      }]);
    }

    return {
      schemaVersion: 'step90-viewer-payload-index-association-debug-v1',
      timestamp: new Date().toISOString(),
      purpose: 'Validate that each tile accumulation packed payload slot matches its source/original index metadata.',
      target: { pixel: targetPixel, indices: targetIndices, maxEntries },
      debugRenderAttempts: debugRender.attempts,
      deterministicState: deps.buildSlimDeterministicStateSummary(deps.buildDeterministicStateSummary()),
      lastRenderResultSummary: deps.buildRenderResultInspectionSummary(renderResult),
      tile: {
        ok: tileInfo.ok,
        reason: tileInfo.reason,
        tileId: tileInfo.tileId,
        tile: tileInfo.tile,
        tileBounds: tileInfo.tileBounds ?? null,
        grid: tileInfo.grid,
        packedCount: Number.isFinite(tileInfo.batch?.packedCount) ? tileInfo.batch.packedCount : 0,
        floatsPerItem: Number.isFinite(tileInfo.batch?.floatsPerItem) ? tileInfo.batch.floatsPerItem : 16,
        hasSourceIndices: tileInfo.batch?.sourceIndices instanceof Uint32Array,
        hasOrderedIndices: tileInfo.batch?.orderedIndices instanceof Uint32Array
      },
      summary: {
        checkedEntryCount: tileInfo.ok ? Math.min(Number.isFinite(tileInfo.batch?.packedCount) ? Math.max(0, tileInfo.batch.packedCount | 0) : 0, maxEntries) : 0,
        emittedEntryCount: entries.length,
        mismatchCount: mismatches.length,
        targetOccurrenceCounts: Object.fromEntries(
          targetIndices.map((index) => [String(index), targetOccurrences.get(index)?.filter((entry) => entry.localOrder !== null).length ?? 0])
        ),
        associationMismatchLikely: mismatches.length > 0,
        mismatchReasons: summarizeAssociationMismatchReasons(mismatches)
      },
      targetOccurrences: Object.fromEntries(targetIndices.map((index) => [String(index), targetOccurrences.get(index) ?? []])),
      mismatches: mismatches.slice(0, 128),
      entries
    };
  }

  function buildLiveSameStateConsistencySummary(tileDebug, associationDebug, indices) {
    const tilePixel = tileDebug?.pixels?.[0] ?? null;
    const tileAccumulation = tilePixel?.accumulation ?? null;
    const tileVisibleCount = getVisibleCountFromDebugResult(tileDebug);
    const associationVisibleCount = getVisibleCountFromDebugResult(associationDebug);
    const tilePayloadCount = Number.isFinite(tilePixel?.tilePayloadCount) ? tilePixel.tilePayloadCount : null;
    const associationPackedCount = Number.isFinite(associationDebug?.tile?.packedCount) ? associationDebug.tile.packedCount : null;
    const targetIndices = Array.isArray(indices) ? indices : [];
    const targetSummary = {};
    for (const index of targetIndices) {
      const key = String(index | 0);
      const associationOccurrences = associationDebug?.targetOccurrences?.[key] ?? [];
      const tileOccurrences = findOriginalSplatOccurrencesInTileDebug(tileDebug, index);
      targetSummary[key] = {
        tileAccumulationOccurrences: tileOccurrences,
        associationOccurrences,
        tileAccumulationOccurrenceCount: tileOccurrences.length,
        associationOccurrenceCount: associationOccurrences.filter((entry) => entry?.localOrder !== null).length
      };
    }
    return {
      schemaVersion: 'step90-live-same-state-consistency-summary-v1',
      timestamp: new Date().toISOString(),
      visibleCount: {
        tileAccumulation: tileVisibleCount,
        association: associationVisibleCount,
        equal: tileVisibleCount !== null && tileVisibleCount === associationVisibleCount
      },
      tilePayloadCount: {
        tileAccumulation: tilePayloadCount,
        associationPackedCount,
        equal: tilePayloadCount !== null && tilePayloadCount === associationPackedCount
      },
      tile: {
        tileAccumulationTileId: tilePixel?.tileId ?? null,
        associationTileId: associationDebug?.tile?.tileId ?? null,
        tileAccumulationTile: tilePixel?.tile ?? null,
        associationTile: associationDebug?.tile?.tile ?? null,
        equal:
          tilePixel?.tileId === associationDebug?.tile?.tileId &&
          JSON.stringify(tilePixel?.tile ?? null) === JSON.stringify(associationDebug?.tile?.tile ?? null)
      },
      accumulation: {
        contributorCounter: tileAccumulation?.contributorCounter ?? null,
        contributionCount: tileAccumulation?.contributionCount ?? null,
        alphaSkipCount: tileAccumulation?.alphaSkipCount ?? null,
        earlyOutTriggered: tileAccumulation?.earlyOutTriggered ?? null,
        earlyOutAtLocalOrder: tileAccumulation?.earlyOutAtLocalOrder ?? null,
        finalRgb: tileAccumulation?.finalRgb ?? null
      },
      association: {
        mismatchCount: associationDebug?.summary?.mismatchCount ?? null,
        associationMismatchLikely: associationDebug?.summary?.associationMismatchLikely ?? null,
        targetOccurrenceCounts: associationDebug?.summary?.targetOccurrenceCounts ?? null
      },
      slotIndexAgreement: summarizeSlotIndexAgreement(tileDebug, associationDebug),
      targetSummary
    };
  }

  async function captureLiveSameStateTileAndAssociationDebug(input = {}) {
    const options = input ?? {};
    const pixel = Array.isArray(options.pixel) && options.pixel.length >= 2
      ? [Number(options.pixel[0]) | 0, Number(options.pixel[1]) | 0]
      : [655, 363];
    const indices = Array.isArray(options.indices) && options.indices.length > 0
      ? options.indices.map((value) => Number(value)).filter(Number.isFinite).map((value) => value | 0)
      : [2765070, 1182029, 2718004];
    const debugRender = await deps.renderCurrentFrameForDebugPayload(options);
    const sharedOptions = {
      ...options,
      ensureCurrentFrame: false,
      renderResultOverride: debugRender.renderResult,
      sharedDebugRenderAttempts: debugRender.attempts
    };
    const tileAccumulationDebug = await captureTileAccumulationDebug({
      ...sharedOptions,
      pixels: [{ x: pixel[0], y: pixel[1], source: 'live-same-state-target' }],
      maxItems: Number.isFinite(Number(options.maxItems)) ? Number(options.maxItems) : 2048
    });
    const associationDebug = await captureViewerPayloadIndexAssociationDebug({
      ...sharedOptions,
      pixel,
      indices,
      maxEntries: Number.isFinite(Number(options.maxEntries)) ? Number(options.maxEntries) : 2048,
      includeAllEntries: options.includeAllEntries !== false
    });
    const consistencySummary = buildLiveSameStateConsistencySummary(tileAccumulationDebug, associationDebug, indices);
    return {
      schemaVersion: 'step90-live-same-state-tile-and-association-debug-v1',
      timestamp: new Date().toISOString(),
      purpose: 'Capture tile accumulation debug and payload-index association debug from one shared Viewer renderResult.',
      target: { pixel, indices },
      sharedDebugRenderAttempts: debugRender.attempts,
      consistencySummary,
      tileAccumulationDebug,
      associationDebug
    };
  }

  async function downloadLiveSameStateTileAndAssociationDebugJson(input = {}, fileNames = {}) {
    const result = await captureLiveSameStateTileAndAssociationDebug(input);
    const tileFileName = fileNames.tileAccumulation ?? 'step90_tile_accumulation_debug_live_same_state.json';
    const associationFileName = fileNames.association ?? 'step90_viewer_payload_index_association_debug_live_same_state.json';
    const summaryFileName = fileNames.summary ?? 'step90_live_same_state_consistency_summary.json';
    return {
      result,
      downloads: {
        tileAccumulation: deps.downloadJsonDebug(result.tileAccumulationDebug, tileFileName),
        association: deps.downloadJsonDebug(result.associationDebug, associationFileName),
        summary: deps.downloadJsonDebug(result.consistencySummary, summaryFileName)
      }
    };
  }

  return {
    captureTileAccumulationDebug,
    captureViewerPayloadIndexAssociationDebug,
    captureLiveSameStateTileAndAssociationDebug,
    downloadLiveSameStateTileAndAssociationDebugJson,
    buildLiveSameStateConsistencySummary
  };
}
