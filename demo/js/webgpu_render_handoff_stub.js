import {
  GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
  createVisiblePackFloatArray,
  computeVisiblePackBaseFloatOffset
} from './gpu_buffer_layout_utils.js';

export const WEBGPU_RENDER_HANDOFF_STUB_MODE =
  'webgpu-render-handoff-stub-partial-payload';

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
    mode: WEBGPU_RENDER_HANDOFF_STUB_MODE,
    status: 'unavailable',
    reason,
    source: 'webgpu fixed-record output + webgpuTileListBackendOutput',
    backendOutputReady: false,
    depthSortReady: false,
    renderHandoffStubReady: false,
    displayConnectionAllowed: false,
    tileCompositeImplemented: false,
    renderPayloadGpuImplemented: false,
    partialPayloadMaterialized: false,
    payloadStoredInJson: false,
    firstValidationFailures: [{ stage: 'input', reason }],
    sampleRecords: []
  };
}

function makeSampleRecordIndices(records, recordFloats) {
  const recordCount = Math.max(0, Math.floor((records?.length ?? 0) / recordFloats));
  if (recordCount <= 0) return [];
  const samples = new Set([0, recordCount - 1]);
  for (let i = 0; i < recordCount; i += 1) {
    const base = i * recordFloats;
    if ((records[base + 1] ?? 0) > 0.5) {
      samples.add(i);
      break;
    }
  }
  return Array.from(samples).sort((a, b) => a - b).slice(0, 4);
}

export function buildWebGpuRenderHandoffStub({
  webgpuTileListBackendOutput,
  depthSortComparison,
  renderPayloadSortReadiness,
  webgpuRecords,
  recordFloats
}) {
  const startMs = nowMs();
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
  if (!(webgpuRecords instanceof Float32Array) || !Number.isFinite(recordFloats) || recordFloats <= 0) {
    return buildUnavailable('webgpu-fixed-record-buffer-unavailable');
  }

  const recordCount = Math.floor(webgpuRecords.length / recordFloats);
  const payload = createVisiblePackFloatArray(recordCount);
  let validRecordCount = 0;
  let populatedFieldMismatchCount = 0;
  const firstValidationFailures = [];

  for (let recordIndex = 0; recordIndex < recordCount; recordIndex += 1) {
    const srcBase = recordIndex * recordFloats;
    const dstBase = computeVisiblePackBaseFloatOffset(recordIndex);
    const valid = (webgpuRecords[srcBase + 1] ?? 0) > 0.5;
    if (valid) validRecordCount += 1;

    const px = finiteOrZero(webgpuRecords[srcBase + 2]);
    const py = finiteOrZero(webgpuRecords[srcBase + 3]);
    const depth = finiteOrZero(webgpuRecords[srcBase + 4]);
    payload[dstBase + 0] = px;
    payload[dstBase + 1] = py;
    payload[dstBase + 2] = 0;
    payload[dstBase + 3] = depth;
    payload[dstBase + 12] = finiteOrZero(webgpuRecords[srcBase + 5]);
    payload[dstBase + 13] = finiteOrZero(webgpuRecords[srcBase + 6]);
    payload[dstBase + 14] = finiteOrZero(webgpuRecords[srcBase + 7]);
    payload[dstBase + 15] = finiteOrZero(webgpuRecords[srcBase + 8]);

    if (payload[dstBase + 0] !== px || payload[dstBase + 1] !== py || payload[dstBase + 3] !== depth) {
      populatedFieldMismatchCount += 1;
      if (firstValidationFailures.length < 8) {
        firstValidationFailures.push({
          stage: 'payload-materialization',
          recordIndex,
          reason: 'center/depth payload copy mismatch'
        });
      }
    }
  }

  const sampleRecords = makeSampleRecordIndices(webgpuRecords, recordFloats).map((recordIndex) => {
    const srcBase = recordIndex * recordFloats;
    const dstBase = computeVisiblePackBaseFloatOffset(recordIndex);
    return {
      recordIndex,
      srcIndex: webgpuRecords[srcBase + 0],
      valid: webgpuRecords[srcBase + 1],
      centerPx: [payload[dstBase + 0], payload[dstBase + 1]],
      radiusPx: payload[dstBase + 2],
      depth: payload[dstBase + 3],
      colorAlpha: [
        payload[dstBase + 4],
        payload[dstBase + 5],
        payload[dstBase + 6],
        payload[dstBase + 7]
      ],
      conic: [payload[dstBase + 8], payload[dstBase + 9], payload[dstBase + 10]],
      miscAabb: [
        payload[dstBase + 12],
        payload[dstBase + 13],
        payload[dstBase + 14],
        payload[dstBase + 15]
      ]
    };
  });

  const payloadShapeValid =
    payload.length === recordCount * GPU_VISIBLE_PACK_FLOATS_PER_ITEM;
  const renderHandoffStubReady =
    payloadShapeValid &&
    populatedFieldMismatchCount === 0 &&
    webgpuTileListBackendOutput.backendOutputReady === true &&
    depthSortComparison.status === 'ok';
  const result = {
    mode: WEBGPU_RENDER_HANDOFF_STUB_MODE,
    status: renderHandoffStubReady ? 'ok' : 'blocked',
    source: 'webgpu fixed-record output + webgpuTileListBackendOutput + depthSortComparison',
    backendOutputReady: webgpuTileListBackendOutput.backendOutputReady === true,
    depthSortReady: depthSortComparison.status === 'ok',
    renderPayloadReadinessStatus:
      renderPayloadSortReadiness?.payloadReadiness?.status ?? null,
    renderHandoffStubReady,
    displayConnectionAllowed: false,
    tileCompositeImplemented: false,
    renderPayloadGpuImplemented: false,
    partialPayloadMaterialized: true,
    payloadStoredInJson: false,
    payloadLayout: {
      layoutVersion: 2,
      floatsPerItem: GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
      materializedFields: ['centerPx', 'depth', 'misc.aabb'],
      zeroFilledFields: ['radiusPx', 'colorAlpha', 'conic', 'reserved'],
      missingDisplayFields: ['radiusPx', 'conic', 'alpha', 'colorAlpha.rgb', 'SH']
    },
    outputBuffer: {
      source: 'transient WebGPU render handoff payload',
      type: `float32[recordCount * ${GPU_VISIBLE_PACK_FLOATS_PER_ITEM}]`,
      recordCount,
      floatCount: payload.length,
      storedInJson: false
    },
    tileListInputs: {
      tileOffsets: webgpuTileListBackendOutput.outputBuffers?.tileOffsets ?? null,
      tileIndices: webgpuTileListBackendOutput.outputBuffers?.tileIndices ?? null
    },
    recordCounts: {
      recordCount,
      validRecordCount,
      tileCount: webgpuTileListBackendOutput.recordCounts?.tileCount ?? null,
      tileIndicesLength: webgpuTileListBackendOutput.recordCounts?.tileIndicesLength ?? null
    },
    validationSummary: {
      payloadShapeValid,
      populatedFieldsValid: populatedFieldMismatchCount === 0,
      populatedFieldMismatchCount,
      firstValidationFailures,
      displayConnectionAllowed: false
    },
    blockers: [
      { stage: 'render-payload', reason: 'radius/conic/colorAlpha/SH still require GPU parity before display' },
      { stage: 'tile-composite', reason: 'tile composite shader handoff not implemented' },
      { stage: 'display-connection', reason: 'display connection intentionally deferred' }
    ],
    nextBackendPrototypeStep: 'render-payload-field-gpu-parity',
    sampleRecords,
    timing: {
      webgpuRenderHandoffStubMs: nowMs() - startMs
    }
  };
  Object.defineProperty(result, 'transientRenderPayload', {
    value: payload,
    enumerable: false
  });
  return result;
}
