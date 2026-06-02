import {
  GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
  createVisiblePackFloatArray,
  computeVisiblePackBaseFloatOffset
} from './gpu_buffer_layout_utils.js';

export const WEBGPU_RENDER_HANDOFF_STUB_MODE =
  'webgpu-render-handoff-stub-partial-payload';

const RENDER_PAYLOAD_REFERENCE_FLOATS = 5;
const RENDER_PAYLOAD_REFERENCE_FIELDS = Object.freeze({
  radiusPx: { offset: 0, components: 1 },
  conic: { offset: 1, components: 3 },
  alpha: { offset: 4, components: 1 }
});

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function finiteOrZero(value) {
  return Number.isFinite(value) ? value : 0;
}

function referenceBaseOffset(recordIndex, layout) {
  const floatsPerRecord = Number.isFinite(layout?.floatsPerRecord)
    ? Math.max(1, Math.floor(layout.floatsPerRecord))
    : RENDER_PAYLOAD_REFERENCE_FLOATS;
  return recordIndex * floatsPerRecord;
}

function comparePayloadField({
  field,
  payload,
  renderPayloadReference,
  renderPayloadReferenceLayout,
  recordCount,
  payloadOffsets,
  referenceField,
  epsilon
}) {
  const refField =
    renderPayloadReferenceLayout?.fields?.[referenceField] ??
    RENDER_PAYLOAD_REFERENCE_FIELDS[referenceField];
  const components = Math.min(payloadOffsets.length, refField?.components ?? payloadOffsets.length);
  const firstMismatches = [];
  let mismatchCount = 0;
  let maxAbsDelta = 0;

  for (let recordIndex = 0; recordIndex < recordCount; recordIndex += 1) {
    const payloadBase = computeVisiblePackBaseFloatOffset(recordIndex);
    const refBase = referenceBaseOffset(recordIndex, renderPayloadReferenceLayout);
    for (let component = 0; component < components; component += 1) {
      const actual = finiteOrZero(payload[payloadBase + payloadOffsets[component]]);
      const expected = finiteOrZero(
        renderPayloadReference[refBase + (refField?.offset ?? 0) + component]
      );
      const absDelta = Math.abs(actual - expected);
      maxAbsDelta = Math.max(maxAbsDelta, absDelta);
      if (absDelta > epsilon) {
        mismatchCount += 1;
        if (firstMismatches.length < 8) {
          firstMismatches.push({
            recordIndex,
            field,
            component: components > 1 ? component : null,
            expected,
            actual,
            absDelta
          });
        }
      }
    }
  }

  return {
    status: mismatchCount === 0 ? 'ok' : 'mismatch',
    field,
    expectedSource: 'cpu-reference-render-payload-field',
    actualSource: 'webgpu-render-handoff-transient-payload',
    referenceField,
    componentCount: components,
    mismatchCount,
    maxAbsDelta,
    firstMismatches
  };
}

function summarizePayloadFieldComparisons(comparisons) {
  const firstValidationFailures = [];
  let totalMismatchCount = 0;
  let maxAbsDelta = 0;
  for (const comparison of Object.values(comparisons)) {
    totalMismatchCount += comparison?.mismatchCount ?? 0;
    maxAbsDelta = Math.max(maxAbsDelta, comparison?.maxAbsDelta ?? 0);
    if (comparison?.status !== 'ok' && firstValidationFailures.length < 8) {
      firstValidationFailures.push({
        stage: 'render-payload-field-comparison',
        field: comparison?.field,
        reason: comparison?.status ?? 'unavailable',
        firstMismatch: comparison?.firstMismatches?.[0] ?? null
      });
    }
  }
  return {
    allComparedFieldsValid: totalMismatchCount === 0,
    totalMismatchCount,
    maxAbsDelta,
    firstValidationFailures
  };
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
    referenceAssistedPayloadFields: [],
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
  renderPayloadReference,
  renderPayloadReferenceLayout,
  epsilon = 1e-4,
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
  if (!(renderPayloadReference instanceof Float32Array)) {
    return buildUnavailable('render-payload-reference-buffer-unavailable');
  }

  const recordCount = Math.floor(webgpuRecords.length / recordFloats);
  const referenceFloatsPerRecord = Number.isFinite(renderPayloadReferenceLayout?.floatsPerRecord)
    ? Math.max(1, Math.floor(renderPayloadReferenceLayout.floatsPerRecord))
    : RENDER_PAYLOAD_REFERENCE_FLOATS;
  if (Math.floor(renderPayloadReference.length / referenceFloatsPerRecord) < recordCount) {
    return buildUnavailable('render-payload-reference-buffer-too-small');
  }
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
    payload[dstBase + 3] = depth;
    const refBase = referenceBaseOffset(recordIndex, renderPayloadReferenceLayout);
    payload[dstBase + 2] = finiteOrZero(renderPayloadReference[refBase + 0]);
    payload[dstBase + 7] = finiteOrZero(renderPayloadReference[refBase + 4]);
    payload[dstBase + 8] = finiteOrZero(renderPayloadReference[refBase + 1]);
    payload[dstBase + 9] = finiteOrZero(renderPayloadReference[refBase + 2]);
    payload[dstBase + 10] = finiteOrZero(renderPayloadReference[refBase + 3]);
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

  const payloadFieldComparisons = {
    radiusPx: comparePayloadField({
      field: 'radiusPx',
      payload,
      renderPayloadReference,
      renderPayloadReferenceLayout,
      recordCount,
      payloadOffsets: [2],
      referenceField: 'radiusPx',
      epsilon
    }),
    conic: comparePayloadField({
      field: 'conic',
      payload,
      renderPayloadReference,
      renderPayloadReferenceLayout,
      recordCount,
      payloadOffsets: [8, 9, 10],
      referenceField: 'conic',
      epsilon
    }),
    alpha: comparePayloadField({
      field: 'alpha',
      payload,
      renderPayloadReference,
      renderPayloadReferenceLayout,
      recordCount,
      payloadOffsets: [7],
      referenceField: 'alpha',
      epsilon
    })
  };
  const payloadFieldComparisonSummary =
    summarizePayloadFieldComparisons(payloadFieldComparisons);
  for (const failure of payloadFieldComparisonSummary.firstValidationFailures) {
    if (firstValidationFailures.length < 8) {
      firstValidationFailures.push(failure);
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
      ],
      payloadFieldDeltas: {
        radiusPx:
          payload[dstBase + 2] -
          finiteOrZero(renderPayloadReference[referenceBaseOffset(recordIndex, renderPayloadReferenceLayout) + 0]),
        conic: [
          payload[dstBase + 8] -
            finiteOrZero(renderPayloadReference[referenceBaseOffset(recordIndex, renderPayloadReferenceLayout) + 1]),
          payload[dstBase + 9] -
            finiteOrZero(renderPayloadReference[referenceBaseOffset(recordIndex, renderPayloadReferenceLayout) + 2]),
          payload[dstBase + 10] -
            finiteOrZero(renderPayloadReference[referenceBaseOffset(recordIndex, renderPayloadReferenceLayout) + 3])
        ],
        alpha:
          payload[dstBase + 7] -
          finiteOrZero(renderPayloadReference[referenceBaseOffset(recordIndex, renderPayloadReferenceLayout) + 4])
      }
    };
  });

  const payloadShapeValid =
    payload.length === recordCount * GPU_VISIBLE_PACK_FLOATS_PER_ITEM;
  const renderHandoffStubReady =
    payloadShapeValid &&
    populatedFieldMismatchCount === 0 &&
    payloadFieldComparisonSummary.allComparedFieldsValid &&
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
    referenceAssistedPayloadFields: ['radiusPx', 'conic', 'alpha'],
    payloadStoredInJson: false,
    payloadLayout: {
      layoutVersion: 2,
      floatsPerItem: GPU_VISIBLE_PACK_FLOATS_PER_ITEM,
      materializedFields: ['centerPx', 'radiusPx', 'depth', 'colorAlpha.a', 'conic', 'misc.aabb'],
      zeroFilledFields: ['colorAlpha.rgb', 'reserved'],
      missingDisplayFields: ['colorAlpha.rgb', 'SH'],
      referenceAssistedFields: ['radiusPx', 'conic', 'alpha']
    },
    payloadFieldComparisons,
    payloadFieldComparisonSummary,
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
      referenceAssistedFieldsValid: payloadFieldComparisonSummary.allComparedFieldsValid,
      referenceAssistedFieldMismatchCount: payloadFieldComparisonSummary.totalMismatchCount,
      referenceAssistedFieldMaxAbsDelta: payloadFieldComparisonSummary.maxAbsDelta,
      firstValidationFailures,
      displayConnectionAllowed: false
    },
    blockers: [
      { stage: 'render-payload', reason: 'colorAlpha.rgb/SH still require payload parity before display' },
      { stage: 'render-payload-gpu-parity', reason: 'radius/conic/alpha are reference-assisted in the handoff stub, not WGSL payload compute yet' },
      { stage: 'tile-composite', reason: 'tile composite shader handoff not implemented' },
      { stage: 'display-connection', reason: 'display connection intentionally deferred' }
    ],
    nextBackendPrototypeStep: 'color-alpha-rgb-sh-payload-parity',
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
