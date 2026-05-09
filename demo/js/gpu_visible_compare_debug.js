const DEFAULT_EPSILON = 1e-6;
const DEFAULT_MAX_MISMATCHES = 16;

function getCandidateIndices(candidateInfoOrIndices) {
  if (candidateInfoOrIndices instanceof Uint32Array) return candidateInfoOrIndices;
  if (Array.isArray(candidateInfoOrIndices)) return Uint32Array.from(candidateInfoOrIndices);
  if (candidateInfoOrIndices?.candidateIndices instanceof Uint32Array) return candidateInfoOrIndices.candidateIndices;
  if (Array.isArray(candidateInfoOrIndices?.candidateIndices)) {
    return Uint32Array.from(candidateInfoOrIndices.candidateIndices);
  }
  return new Uint32Array(0);
}

function finiteOrNull(value) {
  return Number.isFinite(value) ? Number(value) : null;
}

function normalizeVector(value, length) {
  const out = new Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = Array.isArray(value) && Number.isFinite(value[i]) ? Number(value[i]) : null;
  }
  return out;
}

export function compareCandidateIndices(referenceCandidateInfo, candidateCandidateInfo, options = {}) {
  const maxMismatches = Number.isFinite(options.maxMismatches)
    ? Math.max(0, options.maxMismatches | 0)
    : DEFAULT_MAX_MISMATCHES;
  const reference = getCandidateIndices(referenceCandidateInfo);
  const candidate = getCandidateIndices(candidateCandidateInfo);
  const comparedCount = Math.min(reference.length, candidate.length);
  const summary = {
    schemaVersion: 'step94-candidate-indices-comparison-v1',
    referenceMode: referenceCandidateInfo?.candidateMode ?? 'unknown',
    candidateMode: candidateCandidateInfo?.candidateMode ?? 'unknown',
    referenceCount: reference.length,
    candidateCount: candidate.length,
    countEqual: reference.length === candidate.length,
    comparedCount,
    orderMismatchCount: 0,
    firstMismatches: []
  };

  for (let i = 0; i < comparedCount; i++) {
    if (reference[i] !== candidate[i]) {
      summary.orderMismatchCount++;
      if (summary.firstMismatches.length < maxMismatches) {
        summary.firstMismatches.push({
          candidateOrder: i,
          referenceIndex: reference[i],
          candidateIndex: candidate[i]
        });
      }
    }
  }

  summary.anyMismatch = !summary.countEqual || summary.orderMismatchCount > 0;
  return summary;
}

export function buildCandidateComparisonSummary({
  referenceCandidateInfo = null,
  candidateCandidateInfo = null,
  referenceLabel = 'cpu-candidate-reference',
  candidateLabel = 'gpu-candidate-stub',
  metadata = {},
  options = {}
} = {}) {
  const candidateIndices = compareCandidateIndices(referenceCandidateInfo, candidateCandidateInfo, options);
  return {
    schemaVersion: 'step94-candidate-comparison-debug-v1',
    timestamp: new Date().toISOString(),
    purpose: 'Compare CPU reference candidate indices against a candidate/GPU-stub path without changing rendering.',
    referenceLabel,
    candidateLabel,
    metadata,
    candidateIndices,
    referenceSummary: {
      candidateMode: referenceCandidateInfo?.candidateMode ?? 'unknown',
      rangeSummary: referenceCandidateInfo?.rangeSummary ?? null,
      temporalWindow: referenceCandidateInfo?.temporalWindow ?? null,
      temporalIndexDebug: referenceCandidateInfo?.temporalIndexDebug ?? null,
      temporalBucketDebug: referenceCandidateInfo?.temporalBucketDebug ?? null
    },
    candidateSummary: {
      candidateMode: candidateCandidateInfo?.candidateMode ?? 'unknown',
      rangeSummary: candidateCandidateInfo?.rangeSummary ?? null,
      temporalWindow: candidateCandidateInfo?.temporalWindow ?? null,
      temporalIndexDebug: candidateCandidateInfo?.temporalIndexDebug ?? null,
      temporalBucketDebug: candidateCandidateInfo?.temporalBucketDebug ?? null,
      gpuCandidateStubSummary: candidateCandidateInfo?.gpuCandidateStubSummary ?? null
    },
    anyMismatch: !!candidateIndices.anyMismatch
  };
}

function normalizeVisibleItem(item) {
  const centerPx = Array.isArray(item?.centerPx)
    ? normalizeVector(item.centerPx, 2)
    : [finiteOrNull(item?.px), finiteOrNull(item?.py)];
  const radius = Number.isFinite(item?.radiusPx) ? Number(item.radiusPx) : finiteOrNull(item?.radius);
  return {
    srcIndex: Number.isFinite(item?.srcIndex) ? (item.srcIndex | 0) : null,
    centerPx,
    radius,
    radiusPx: radius,
    depth: finiteOrNull(item?.depth),
    conic: normalizeVector(item?.conic, 3),
    opacity: Number.isFinite(item?.opacity) ? Number(item.opacity) : (
      Array.isArray(item?.colorAlpha) && Number.isFinite(item.colorAlpha[3]) ? Number(item.colorAlpha[3]) : null
    ),
    color: normalizeVector(item?.color, 3),
    colorAlpha: normalizeVector(item?.colorAlpha, 4),
    aabb: normalizeVector(item?.aabb, 4),
    tileRange: normalizeVector(item?.tileRange, 4),
    stateConvention: typeof item?.stateConvention === 'string' ? item.stateConvention : null,
    usedCuda4DStateHelper: typeof item?.usedCuda4DStateHelper === 'boolean' ? item.usedCuda4DStateHelper : null,
    stateHelperVersion: typeof item?.stateHelperVersion === 'string' ? item.stateHelperVersion : null
  };
}

function compareScalars(a, b, epsilon) {
  if (a === null || b === null) {
    return {
      comparable: a === b,
      mismatch: a !== b,
      abs: a === b ? 0 : Infinity
    };
  }
  const abs = Math.abs(a - b);
  return {
    comparable: true,
    mismatch: abs > epsilon,
    abs
  };
}

function compareVectors(a, b, epsilon) {
  const length = Math.max(Array.isArray(a) ? a.length : 0, Array.isArray(b) ? b.length : 0);
  let maxAbs = 0;
  let mismatch = false;
  for (let i = 0; i < length; i++) {
    const result = compareScalars(a?.[i] ?? null, b?.[i] ?? null, epsilon);
    maxAbs = Math.max(maxAbs, result.abs);
    mismatch = mismatch || result.mismatch;
  }
  return { mismatch, maxAbs };
}

function noteField(summary, fieldName, mismatch, maxAbs) {
  if (!summary.fieldMismatchCounts[fieldName]) summary.fieldMismatchCounts[fieldName] = 0;
  if (!summary.maxAbsByField[fieldName]) summary.maxAbsByField[fieldName] = 0;
  if (mismatch) summary.fieldMismatchCounts[fieldName]++;
  summary.maxAbsByField[fieldName] = Math.max(summary.maxAbsByField[fieldName], maxAbs);
}

function compareNormalizedItems(reference, candidate, epsilon) {
  const fieldResults = {};
  const scalarFields = ['srcIndex', 'radius', 'radiusPx', 'depth', 'opacity'];
  const vectorFields = ['centerPx', 'conic', 'color', 'colorAlpha', 'aabb', 'tileRange'];
  const exactFields = ['stateConvention', 'usedCuda4DStateHelper', 'stateHelperVersion'];

  for (const field of scalarFields) {
    const result = compareScalars(reference[field], candidate[field], epsilon);
    fieldResults[field] = { mismatch: result.mismatch, maxAbs: result.abs };
  }
  for (const field of vectorFields) {
    fieldResults[field] = compareVectors(reference[field], candidate[field], epsilon);
  }
  for (const field of exactFields) {
    fieldResults[field] = {
      mismatch: reference[field] !== candidate[field],
      maxAbs: reference[field] === candidate[field] ? 0 : Infinity
    };
  }
  return fieldResults;
}

export function summarizeVisibleItemDelta(referenceItem, candidateItem, options = {}) {
  const epsilon = Number.isFinite(options.epsilon) ? Number(options.epsilon) : DEFAULT_EPSILON;
  const reference = normalizeVisibleItem(referenceItem);
  const candidate = normalizeVisibleItem(candidateItem);
  const fields = compareNormalizedItems(reference, candidate, epsilon);
  const mismatchedFields = Object.entries(fields)
    .filter(([, result]) => result.mismatch)
    .map(([field]) => field);
  return {
    mismatch: mismatchedFields.length > 0,
    mismatchedFields,
    fields,
    reference,
    candidate
  };
}

export function compareVisibleItems(referenceItems, candidateItems, options = {}) {
  const epsilon = Number.isFinite(options.epsilon) ? Number(options.epsilon) : DEFAULT_EPSILON;
  const maxMismatches = Number.isFinite(options.maxMismatches)
    ? Math.max(0, options.maxMismatches | 0)
    : DEFAULT_MAX_MISMATCHES;
  const reference = Array.isArray(referenceItems) ? referenceItems : [];
  const candidate = Array.isArray(candidateItems) ? candidateItems : [];
  const comparedCount = Math.min(reference.length, candidate.length);
  const summary = {
    schemaVersion: 'step94-visible-items-comparison-v1',
    epsilon,
    referenceCount: reference.length,
    candidateCount: candidate.length,
    countEqual: reference.length === candidate.length,
    comparedCount,
    orderMismatchCount: 0,
    itemMismatchCount: 0,
    fieldMismatchCounts: {},
    maxAbsByField: {},
    firstMismatches: []
  };

  for (let i = 0; i < comparedCount; i++) {
    const delta = summarizeVisibleItemDelta(reference[i], candidate[i], { epsilon });
    const refIndex = Number.isFinite(reference[i]?.srcIndex) ? (reference[i].srcIndex | 0) : null;
    const candidateIndex = Number.isFinite(candidate[i]?.srcIndex) ? (candidate[i].srcIndex | 0) : null;
    const orderMismatch = refIndex !== candidateIndex;
    if (orderMismatch) summary.orderMismatchCount++;
    for (const [fieldName, fieldResult] of Object.entries(delta.fields)) {
      noteField(summary, fieldName, fieldResult.mismatch, fieldResult.maxAbs);
    }
    if (delta.mismatch || orderMismatch) {
      summary.itemMismatchCount++;
      if (summary.firstMismatches.length < maxMismatches) {
        summary.firstMismatches.push({
          itemIndex: i,
          orderMismatch,
          referenceSrcIndex: refIndex,
          candidateSrcIndex: candidateIndex,
          mismatchedFields: delta.mismatchedFields,
          reference: delta.reference,
          candidate: delta.candidate
        });
      }
    }
  }

  summary.anyMismatch =
    !summary.countEqual ||
    summary.orderMismatchCount > 0 ||
    summary.itemMismatchCount > 0;
  return summary;
}

function getPackedFloatArray(payload) {
  if (payload instanceof Float32Array) return payload;
  if (payload?.packed instanceof Float32Array) return payload.packed;
  if (payload?.packedFloats instanceof Float32Array) return payload.packedFloats;
  return null;
}

export function comparePackedPayloads(referencePayload, candidatePayload, options = {}) {
  const epsilon = Number.isFinite(options.epsilon) ? Number(options.epsilon) : DEFAULT_EPSILON;
  const maxMismatches = Number.isFinite(options.maxMismatches)
    ? Math.max(0, options.maxMismatches | 0)
    : DEFAULT_MAX_MISMATCHES;
  const reference = getPackedFloatArray(referencePayload);
  const candidate = getPackedFloatArray(candidatePayload);
  const referenceLength = reference ? reference.length : 0;
  const candidateLength = candidate ? candidate.length : 0;
  const comparedLength = Math.min(referenceLength, candidateLength);
  const summary = {
    schemaVersion: 'step94-packed-payload-comparison-v1',
    epsilon,
    referencePresent: !!reference,
    candidatePresent: !!candidate,
    referenceCount: Number.isFinite(referencePayload?.packedCount) ? referencePayload.packedCount : null,
    candidateCount: Number.isFinite(candidatePayload?.packedCount) ? candidatePayload.packedCount : null,
    referenceFloatsPerItem: Number.isFinite(referencePayload?.floatsPerItem) ? referencePayload.floatsPerItem : null,
    candidateFloatsPerItem: Number.isFinite(candidatePayload?.floatsPerItem) ? candidatePayload.floatsPerItem : null,
    referenceLength,
    candidateLength,
    lengthEqual: referenceLength === candidateLength,
    comparedLength,
    mismatchCount: 0,
    maxAbs: 0,
    firstMismatches: []
  };

  for (let i = 0; i < comparedLength; i++) {
    const abs = Math.abs(reference[i] - candidate[i]);
    summary.maxAbs = Math.max(summary.maxAbs, abs);
    if (abs > epsilon) {
      summary.mismatchCount++;
      if (summary.firstMismatches.length < maxMismatches) {
        summary.firstMismatches.push({
          floatIndex: i,
          reference: reference[i],
          candidate: candidate[i],
          abs
        });
      }
    }
  }

  summary.countEqual =
    summary.referenceCount === summary.candidateCount &&
    summary.referenceFloatsPerItem === summary.candidateFloatsPerItem;
  summary.anyMismatch =
    !summary.referencePresent ||
    !summary.candidatePresent ||
    !summary.lengthEqual ||
    !summary.countEqual ||
    summary.mismatchCount > 0;
  return summary;
}

export function buildVisibleComparisonSummary({
  referenceItems,
  candidateItems,
  referencePackedPayload = null,
  candidatePackedPayload = null,
  referenceLabel = 'cpu-reference',
  candidateLabel = 'candidate',
  metadata = {},
  options = {}
} = {}) {
  const visibleItems = compareVisibleItems(referenceItems, candidateItems, options);
  const packedPayload = comparePackedPayloads(referencePackedPayload, candidatePackedPayload, options);
  return {
    schemaVersion: 'step94-visible-comparison-debug-v1',
    timestamp: new Date().toISOString(),
    purpose: 'Compare CPU reference visible items and packed payloads against a candidate path without changing rendering.',
    referenceLabel,
    candidateLabel,
    metadata,
    visibleItems,
    packedPayload,
    anyMismatch: !!(visibleItems.anyMismatch || packedPayload.anyMismatch)
  };
}
