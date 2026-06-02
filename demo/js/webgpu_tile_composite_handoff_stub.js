import { computeVisiblePackBaseFloatOffset } from './gpu_buffer_layout_utils.js';

export const WEBGPU_TILE_COMPOSITE_HANDOFF_STUB_MODE =
  'webgpu-tile-composite-handoff-stub';

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
    mode: WEBGPU_TILE_COMPOSITE_HANDOFF_STUB_MODE,
    status: 'unavailable',
    reason,
    source: 'webgpuRenderHandoffStub + webgpu tile-list output + depthSortComparison',
    nonDisplayOnly: true,
    tileCompositeHandoffStubReady: false,
    tileCompositeImplemented: false,
    framebufferImplemented: false,
    displayConnectionAllowed: false,
    firstValidationFailures: [{ stage: 'input', reason }],
    sampleTiles: []
  };
}

function getRecordDepth(records, recordFloats, recordIndex) {
  const base = recordIndex * recordFloats;
  return finiteOrZero(records?.[base + 4]);
}

function makeSampleTileIds(depthSortComparison, tileCount) {
  const ids = new Set();
  for (const sample of depthSortComparison?.sampleTiles ?? []) {
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

export function buildWebGpuTileCompositeHandoffStub({
  webgpuRenderHandoffStub,
  webgpuTileListBackendOutput,
  depthSortComparison,
  tileIndicesWebGpuScatterComparison,
  webgpuTileOffsetsPrefixDryRun,
  webgpuRecords,
  recordFloats
}) {
  const startMs = nowMs();
  if (!webgpuRenderHandoffStub || webgpuRenderHandoffStub.status !== 'ok') {
    const reason = webgpuRenderHandoffStub?.reason ??
      webgpuRenderHandoffStub?.status ??
      'webgpu-render-handoff-stub-unavailable';
    return buildUnavailable(reason);
  }
  if (!webgpuTileListBackendOutput || webgpuTileListBackendOutput.status !== 'ok') {
    const reason = webgpuTileListBackendOutput?.reason ??
      webgpuTileListBackendOutput?.status ??
      'webgpu-tile-list-backend-output-unavailable';
    return buildUnavailable(reason);
  }
  if (!depthSortComparison || depthSortComparison.status !== 'ok') {
    const reason = depthSortComparison?.reason ??
      depthSortComparison?.status ??
      'depth-sort-comparison-unavailable';
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

  const tileCount = webgpuTileListBackendOutput.recordCounts?.tileCount ?? (tileOffsets.length - 1);
  const totalTileRefs = webgpuTileListBackendOutput.capacity?.totalTileRefs ?? tileIndices.length;
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

  const sampleTiles = makeSampleTileIds(depthSortComparison, tileCount).map((tileId) => {
    const start = tileOffsets[tileId] ?? 0;
    const end = tileOffsets[tileId + 1] ?? start;
    const sortedRecords = Array.from(tileIndices.slice(start, end)).sort((a, b) => {
      const da = getRecordDepth(webgpuRecords, recordFloats, a);
      const db = getRecordDepth(webgpuRecords, recordFloats, b);
      if (da !== db) return da - db;
      return a - b;
    });
    return {
      tileId,
      tileIndexStart: start,
      tileIndexEnd: end,
      tileRefCount: Math.max(0, end - start),
      previewCount: Math.min(4, sortedRecords.length),
      firstCompositePackets: sortedRecords.slice(0, 4).map((recordIndex, localIndex) => ({
        localIndex,
        recordIndex,
        depth: getRecordDepth(webgpuRecords, recordFloats, recordIndex),
        payload: readPayloadPreview(payload, recordIndex)
      }))
    };
  });

  const tileCompositeHandoffStubReady =
    firstValidationFailures.length === 0 &&
    webgpuRenderHandoffStub.renderHandoffStubReady === true &&
    webgpuTileListBackendOutput.backendOutputReady === true &&
    depthSortComparison.status === 'ok';

  return {
    mode: WEBGPU_TILE_COMPOSITE_HANDOFF_STUB_MODE,
    status: tileCompositeHandoffStubReady ? 'ok' : 'blocked',
    source: 'webgpuRenderHandoffStub + webgpu tile-list output + depthSortComparison',
    nonDisplayOnly: true,
    tileCompositeHandoffStubImplemented: true,
    tileCompositeHandoffStubReady,
    tileCompositeImplemented: false,
    framebufferImplemented: false,
    displayConnectionAllowed: false,
    renderPayloadSource: 'webgpuRenderHandoffStub.transientRenderPayload',
    tileListSource: 'webgpuTileOffsetsPrefixDryRun.tileOffsets + tileIndicesWebGpuScatterComparison.transientTileIndices',
    orderingSource: 'depthSortComparison CPU-staged depth order',
    shPolicy: {
      requiredForThisStub: false,
      status: 'deferred',
      fallbackColorSource: 'reference-assisted colorAlpha.rgb payload',
      unresolved: 'WGSL SH/color evaluation parity'
    },
    payloadFieldsConsumed: ['centerPx', 'radiusPx', 'depth', 'colorAlpha.rgb', 'colorAlpha.a', 'conic', 'misc.aabb'],
    outputSchema: {
      tileOffsets: 'uint32[tileCount + 1]',
      tileIndices: 'uint32[totalTileRefs]',
      renderPayload: 'float32[recordCount * 16]',
      order: 'per-tile ascending depth order for preview packets'
    },
    recordCounts: {
      recordCount: webgpuRenderHandoffStub.recordCounts?.recordCount ?? null,
      validRecordCount: webgpuRenderHandoffStub.recordCounts?.validRecordCount ?? null,
      tileCount,
      tileIndicesLength: tileIndices.length,
      totalTileRefs
    },
    validationSummary: {
      payloadShapeValid,
      tileOffsetsShapeValid,
      tileIndicesShapeValid,
      depthSortReady: depthSortComparison.status === 'ok',
      renderHandoffReady: webgpuRenderHandoffStub.renderHandoffStubReady === true,
      tileListBackendReady: webgpuTileListBackendOutput.backendOutputReady === true,
      firstValidationFailures
    },
    blockers: [
      { stage: 'sh-color-evaluation', reason: 'WGSL SH/color evaluation parity remains deferred' },
      { stage: 'tile-composite-shader', reason: 'tile composite shader is not implemented in WebGPU' },
      { stage: 'framebuffer', reason: 'WebGPU framebuffer/display connection intentionally deferred' }
    ],
    nextBackendPrototypeStep: 'webgpu-tile-composite-shader-or-sh-color-evaluation-parity',
    sampleTiles,
    timing: {
      webgpuTileCompositeHandoffStubMs: nowMs() - startMs
    }
  };
}
