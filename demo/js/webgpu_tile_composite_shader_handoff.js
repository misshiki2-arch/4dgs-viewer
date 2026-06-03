import { computeVisiblePackBaseFloatOffset } from './gpu_buffer_layout_utils.js';

export const WEBGPU_TILE_COMPOSITE_SHADER_HANDOFF_MODE =
  'webgpu-tile-composite-shader-handoff-non-display';

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function finiteOrZero(value) {
  return Number.isFinite(value) ? value : 0;
}

function buildUnavailable(reason) {
  return {
    mode: WEBGPU_TILE_COMPOSITE_SHADER_HANDOFF_MODE,
    status: 'unavailable',
    reason,
    source: 'webgpuTileCompositeHandoffStub',
    nonDisplayOnly: true,
    tileCompositeShaderHandoffImplemented: true,
    tileCompositeShaderHandoffReady: false,
    tileCompositeShaderImplemented: false,
    tileCompositeImplemented: false,
    framebufferImplemented: false,
    displayConnectionAllowed: false,
    orderedTileIndicesStoredInJson: false,
    firstValidationFailures: [{ stage: 'input', reason }],
    sampleTiles: []
  };
}

function getRecordDepth(records, recordFloats, recordIndex) {
  const base = recordIndex * recordFloats;
  return finiteOrZero(records?.[base + 4]);
}

function readPayloadPreview(payload, recordIndex) {
  const base = computeVisiblePackBaseFloatOffset(recordIndex);
  return {
    centerPx: [payload[base + 0], payload[base + 1]],
    radiusPx: payload[base + 2],
    depth: payload[base + 3],
    colorAlpha: [
      payload[base + 4],
      payload[base + 5],
      payload[base + 6],
      payload[base + 7]
    ],
    conic: [payload[base + 8], payload[base + 9], payload[base + 10]],
    miscAabb: [
      payload[base + 12],
      payload[base + 13],
      payload[base + 14],
      payload[base + 15]
    ]
  };
}

function makeSampleTileIds(tileCompositeHandoffStub, tileCount) {
  const ids = new Set();
  for (const sample of tileCompositeHandoffStub?.sampleTiles ?? []) {
    if (Number.isFinite(sample?.tileId)) ids.add(sample.tileId | 0);
  }
  if (tileCount > 0) {
    ids.add(0);
    ids.add(tileCount - 1);
  }
  return Array.from(ids)
    .filter((tileId) => tileId >= 0 && tileId < tileCount)
    .sort((a, b) => a - b)
    .slice(0, 8);
}

export function buildWebGpuTileCompositeShaderHandoff({
  webgpuTileCompositeHandoffStub,
  webgpuRenderHandoffStub,
  tileIndicesWebGpuScatterComparison,
  webgpuTileOffsetsPrefixDryRun,
  webgpuRecords,
  recordFloats
}) {
  const startMs = nowMs();
  if (!webgpuTileCompositeHandoffStub || webgpuTileCompositeHandoffStub.status !== 'ok') {
    const reason = webgpuTileCompositeHandoffStub?.reason ??
      webgpuTileCompositeHandoffStub?.status ??
      'webgpu-tile-composite-handoff-stub-unavailable';
    return buildUnavailable(reason);
  }
  if (!webgpuRenderHandoffStub || webgpuRenderHandoffStub.status !== 'ok') {
    const reason = webgpuRenderHandoffStub?.reason ??
      webgpuRenderHandoffStub?.status ??
      'webgpu-render-handoff-stub-unavailable';
    return buildUnavailable(reason);
  }

  const payload = webgpuRenderHandoffStub.transientRenderPayload;
  const tileIndices = tileIndicesWebGpuScatterComparison?.transientTileIndices;
  const tileOffsets = webgpuTileOffsetsPrefixDryRun?.tileOffsets;
  if (!(payload instanceof Float32Array)) {
    return buildUnavailable('transient-render-payload-unavailable');
  }
  if (!(tileIndices instanceof Uint32Array)) {
    return buildUnavailable('transient-tile-indices-unavailable');
  }
  if (!Array.isArray(tileOffsets) && !(tileOffsets instanceof Uint32Array)) {
    return buildUnavailable('tile-offsets-unavailable');
  }
  if (!(webgpuRecords instanceof Float32Array) || !Number.isFinite(recordFloats) || recordFloats <= 0) {
    return buildUnavailable('webgpu-records-unavailable');
  }

  const recordCounts = webgpuTileCompositeHandoffStub.recordCounts ?? {};
  const tileCount = recordCounts.tileCount ?? Math.max(0, tileOffsets.length - 1);
  const totalTileRefs = recordCounts.totalTileRefs ?? tileIndices.length;
  const orderedTileIndices = new Uint32Array(tileIndices.length);
  const firstValidationFailures = [];
  const tileOffsetsShapeValid = tileOffsets.length === tileCount + 1;
  const tileIndicesShapeValid = tileIndices.length === totalTileRefs;
  const payloadShapeValid =
    payload.length === (webgpuRenderHandoffStub.outputBuffer?.floatCount ?? payload.length);

  if (!tileOffsetsShapeValid) {
    firstValidationFailures.push({
      stage: 'tile-offsets',
      reason: 'tile-offsets-length-mismatch',
      actualLength: tileOffsets.length,
      expectedLength: tileCount + 1
    });
  }
  if (!tileIndicesShapeValid) {
    firstValidationFailures.push({
      stage: 'tile-indices',
      reason: 'tile-indices-length-mismatch',
      actualLength: tileIndices.length,
      expectedLength: totalTileRefs
    });
  }
  if (!payloadShapeValid) {
    firstValidationFailures.push({
      stage: 'payload',
      reason: 'payload-float-count-mismatch'
    });
  }

  let orderingViolationCount = 0;
  let maxRefsPerTile = 0;
  let nonEmptyTiles = 0;
  for (let tileId = 0; tileId < tileCount; tileId += 1) {
    const start = tileOffsets[tileId] ?? 0;
    const end = tileOffsets[tileId + 1] ?? start;
    const count = Math.max(0, end - start);
    if (count > 0) nonEmptyTiles += 1;
    maxRefsPerTile = Math.max(maxRefsPerTile, count);
    const sorted = Array.from(tileIndices.slice(start, end)).sort((a, b) => {
      const da = getRecordDepth(webgpuRecords, recordFloats, a);
      const db = getRecordDepth(webgpuRecords, recordFloats, b);
      if (da !== db) return da - db;
      return a - b;
    });
    for (let i = 0; i < sorted.length; i += 1) {
      orderedTileIndices[start + i] = sorted[i];
      if (i > 0) {
        const prev = getRecordDepth(webgpuRecords, recordFloats, sorted[i - 1]);
        const cur = getRecordDepth(webgpuRecords, recordFloats, sorted[i]);
        if (cur < prev) {
          orderingViolationCount += 1;
          if (firstValidationFailures.length < 8) {
            firstValidationFailures.push({
              stage: 'ordered-tile-indices',
              reason: 'depth-order-violation',
              tileId,
              localIndex: i,
              previousDepth: prev,
              currentDepth: cur
            });
          }
        }
      }
    }
  }

  const sampleTiles = makeSampleTileIds(webgpuTileCompositeHandoffStub, tileCount).map((tileId) => {
    const start = tileOffsets[tileId] ?? 0;
    const end = tileOffsets[tileId + 1] ?? start;
    const preview = Array.from(orderedTileIndices.slice(start, Math.min(end, start + 4)));
    return {
      tileId,
      tileIndexStart: start,
      tileIndexEnd: end,
      tileRefCount: Math.max(0, end - start),
      previewCount: preview.length,
      firstShaderPackets: preview.map((recordIndex, localIndex) => ({
        localIndex,
        recordIndex,
        depth: getRecordDepth(webgpuRecords, recordFloats, recordIndex),
        payload: readPayloadPreview(payload, recordIndex)
      }))
    };
  });

  const tileCompositeShaderHandoffReady =
    payloadShapeValid &&
    tileOffsetsShapeValid &&
    tileIndicesShapeValid &&
    orderingViolationCount === 0 &&
    webgpuTileCompositeHandoffStub.tileCompositeHandoffStubReady === true &&
    webgpuRenderHandoffStub.renderHandoffStubReady === true;

  const result = {
    mode: WEBGPU_TILE_COMPOSITE_SHADER_HANDOFF_MODE,
    status: tileCompositeShaderHandoffReady ? 'ok' : 'blocked',
    source: 'webgpuTileCompositeHandoffStub + transient render payload + transient tile-list buffers',
    nonDisplayOnly: true,
    tileCompositeShaderHandoffImplemented: true,
    tileCompositeShaderHandoffReady,
    tileCompositeShaderImplemented: false,
    tileCompositeImplemented: false,
    framebufferImplemented: false,
    displayConnectionAllowed: false,
    orderedTileIndicesStoredInJson: false,
    renderPayloadStoredInJson: false,
    shPolicy: {
      requiredForThisHandoff: false,
      status: 'deferred',
      fallbackColorSource: 'reference-assisted colorAlpha.rgb payload',
      unresolved: 'WGSL SH/color evaluation parity'
    },
    shaderInputBuffers: {
      tileOffsets: {
        source: 'webgpuTileOffsetsPrefixDryRun.tileOffsets',
        type: 'uint32[tileCount + 1]',
        length: tileOffsets.length,
        storedInJson: true
      },
      orderedTileIndices: {
        source: 'CPU-staged sorted transient tileIndices from WebGPU scatter output',
        type: 'uint32[totalTileRefs]',
        length: orderedTileIndices.length,
        storedInJson: false
      },
      renderPayload: {
        source: 'webgpuRenderHandoffStub.transientRenderPayload',
        type: 'float32[recordCount * 16]',
        length: payload.length,
        storedInJson: false
      }
    },
    shaderPacketLayout: {
      order: 'per-tile ascending depth, recordIndex ascending tie-break',
      payloadFields: ['centerPx', 'radiusPx', 'depth', 'colorAlpha', 'conic', 'misc.aabb'],
      compositeEquationStatus: 'not-executed',
      intendedCompositeEquation: 'front-to-back alpha accumulation over ordered tile refs'
    },
    recordCounts: {
      recordCount: recordCounts.recordCount ?? null,
      validRecordCount: recordCounts.validRecordCount ?? null,
      tileCount,
      tileIndicesLength: tileIndices.length,
      orderedTileIndicesLength: orderedTileIndices.length,
      totalTileRefs,
      maxRefsPerTile,
      nonEmptyTiles
    },
    validationSummary: {
      payloadShapeValid,
      tileOffsetsShapeValid,
      tileIndicesShapeValid,
      orderedTileIndicesShapeValid: orderedTileIndices.length === totalTileRefs,
      depthOrderingValid: orderingViolationCount === 0,
      orderingViolationCount,
      renderHandoffReady: webgpuRenderHandoffStub.renderHandoffStubReady === true,
      tileCompositeHandoffReady:
        webgpuTileCompositeHandoffStub.tileCompositeHandoffStubReady === true,
      firstValidationFailures
    },
    blockers: [
      { stage: 'tile-composite-shader', reason: 'WebGPU tile composite shader execution remains deferred' },
      { stage: 'sh-color-evaluation', reason: 'WGSL SH/color evaluation parity remains deferred' },
      { stage: 'framebuffer', reason: 'WebGPU framebuffer/display connection intentionally deferred' }
    ],
    nextBackendPrototypeStep: 'webgpu-tile-composite-shader-non-display-comparison-or-sh-color-parity',
    sampleTiles,
    timing: {
      webgpuTileCompositeShaderHandoffMs: nowMs() - startMs
    }
  };
  Object.defineProperty(result, 'transientOrderedTileIndices', {
    value: orderedTileIndices,
    enumerable: false
  });
  return result;
}
