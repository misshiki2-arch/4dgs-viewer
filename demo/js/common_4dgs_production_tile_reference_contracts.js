export const PRODUCTION_TILE_REFERENCE_CAPACITY_CONTRACT_VERSION =
  'phase3-production-tile-reference-capacity-v2';

export const PRODUCTION_TILE_REFERENCE_FLOAT_STRIDE = 4;

function finiteInteger(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.floor(number)) : fallback;
}

export function nextPowerOfTwo(value) {
  const normalized = finiteInteger(value);
  if (normalized <= 1) return normalized;
  return 2 ** Math.ceil(Math.log2(normalized));
}

export function resolveProductionTileReferenceAllocation({
  device,
  recordCount = 0,
  tileCount = 0
} = {}) {
  const normalizedRecordCount = finiteInteger(recordCount);
  const normalizedTileCount = finiteInteger(tileCount);
  const maxStorageBufferBindingSize = finiteInteger(
    device?.limits?.maxStorageBufferBindingSize,
    128 * 1024 * 1024
  );
  const maxBufferSize = finiteInteger(
    device?.limits?.maxBufferSize,
    maxStorageBufferBindingSize
  );
  const limitingBytes = Math.min(maxStorageBufferBindingSize, maxBufferSize);
  const deviceReferenceCapacity = Math.floor(
    limitingBytes /
      (PRODUCTION_TILE_REFERENCE_FLOAT_STRIDE * Float32Array.BYTES_PER_ELEMENT)
  );
  const theoreticalReferenceCount = Math.min(
    Number.MAX_SAFE_INTEGER,
    normalizedRecordCount * normalizedTileCount
  );
  const allocatedReferenceCapacity = Math.max(
    1,
    Math.min(deviceReferenceCapacity, theoreticalReferenceCount)
  );
  return {
    allocatedReferenceCapacity,
    allocatedReferenceBytes:
      allocatedReferenceCapacity *
      PRODUCTION_TILE_REFERENCE_FLOAT_STRIDE *
      Float32Array.BYTES_PER_ELEMENT,
    deviceReferenceCapacity,
    maxStorageBufferBindingSize,
    maxBufferSize,
    allocationPolicy:
      'device-storage-limit-bounded-compact-production-tile-reference-buffer'
  };
}

export function buildProductionTileReferenceCapacityContract({
  status = 'ok',
  recordCount = 0,
  tileCount = 0,
  allocatedReferenceCapacity = 0,
  requiredReferenceCount = 0,
  requiredPaddedReferenceCapacity = 0,
  writtenReferenceCount = 0,
  maxReferencesPerTile = 0,
  compactOffsetsGenerated = false,
  executionPlanCompletionReady = false,
  recordAndReferenceCapacitySeparated = false,
  capacityOverflowDetected = false,
  capacityOverflowFailClosed = false,
  silentDropAllowed = false,
  allocationPolicy = null,
  reason = null
} = {}) {
  const normalizedRecordCount = finiteInteger(recordCount);
  const normalizedTileCount = finiteInteger(tileCount);
  const normalizedAllocatedCapacity = finiteInteger(allocatedReferenceCapacity);
  const normalizedRequiredCount = finiteInteger(requiredReferenceCount);
  const normalizedPaddedCapacity = finiteInteger(requiredPaddedReferenceCapacity);
  const normalizedWrittenCount = finiteInteger(writtenReferenceCount);
  const overflowDetected =
    capacityOverflowDetected === true ||
    normalizedPaddedCapacity > normalizedAllocatedCapacity;
  const zeroReferenceWorkload = normalizedRequiredCount === 0;
  const countsReady =
    normalizedWrittenCount === normalizedRequiredCount &&
    (
      zeroReferenceWorkload === false ||
      executionPlanCompletionReady === true
    );
  const ready =
    status === 'ok' &&
    normalizedRecordCount > 0 &&
    normalizedTileCount > 0 &&
    normalizedAllocatedCapacity > 0 &&
    normalizedPaddedCapacity >= normalizedRequiredCount &&
    normalizedPaddedCapacity <= normalizedAllocatedCapacity &&
    countsReady &&
    compactOffsetsGenerated === true &&
    recordAndReferenceCapacitySeparated === true &&
    overflowDetected === false &&
    capacityOverflowFailClosed === true &&
    silentDropAllowed === false;
  return {
    contractVersion: PRODUCTION_TILE_REFERENCE_CAPACITY_CONTRACT_VERSION,
    status: ready ? 'ok' : status === 'ok' ? 'blocked' : status,
    tileReferenceCapacityReady: ready,
    recordCount: normalizedRecordCount,
    tileCount: normalizedTileCount,
    allocatedReferenceCapacity: normalizedAllocatedCapacity,
    requiredReferenceCount: normalizedRequiredCount,
    requiredPaddedReferenceCapacity: normalizedPaddedCapacity,
    writtenReferenceCount: normalizedWrittenCount,
    maxReferencesPerTile: finiteInteger(maxReferencesPerTile),
    compactOffsetsGenerated: compactOffsetsGenerated === true,
    executionPlanCompletionReady: executionPlanCompletionReady === true,
    workClassification:
      zeroReferenceWorkload ? 'zero-reference' : 'nonzero-reference',
    recordAndReferenceCapacitySeparated:
      recordAndReferenceCapacitySeparated === true,
    capacityOverflowDetected: overflowDetected,
    capacityOverflowFailClosed: capacityOverflowFailClosed === true,
    silentDropAllowed: silentDropAllowed === true,
    allocationPolicy,
    reason: ready
      ? null
      : reason ??
        (overflowDetected
          ? 'production-tile-reference-capacity-overflow-fail-closed'
          : 'production-tile-reference-capacity-not-ready')
  };
}

// Pure contract oracle used by focused tests. Runtime counts and scatters on GPU.
export function buildProductionTileReferencePlanFromTileCounts({
  tileReferenceCounts = [],
  allocatedReferenceCapacity = 0,
  recordCount = 0
} = {}) {
  const counts = Array.from(tileReferenceCounts, (value) => finiteInteger(value));
  let requiredReferenceCount = 0;
  let requiredPaddedReferenceCapacity = 0;
  let maxReferencesPerTile = 0;
  const tiles = counts.map((count) => {
    const offset = requiredPaddedReferenceCapacity;
    const paddedCount = nextPowerOfTwo(count);
    requiredReferenceCount += count;
    requiredPaddedReferenceCapacity += paddedCount;
    maxReferencesPerTile = Math.max(maxReferencesPerTile, count);
    return { offset, count, paddedCount };
  });
  const capacityOverflowDetected =
    requiredPaddedReferenceCapacity > finiteInteger(allocatedReferenceCapacity);
  const contract = buildProductionTileReferenceCapacityContract({
    recordCount,
    tileCount: counts.length,
    allocatedReferenceCapacity,
    requiredReferenceCount,
    requiredPaddedReferenceCapacity,
    writtenReferenceCount: capacityOverflowDetected ? 0 : requiredReferenceCount,
    maxReferencesPerTile,
    compactOffsetsGenerated: true,
    recordAndReferenceCapacitySeparated: true,
    capacityOverflowDetected,
    capacityOverflowFailClosed: true,
    silentDropAllowed: false,
    allocationPolicy: 'test-oracle-explicit-reference-capacity'
  });
  return { tiles, contract };
}
