import {
  buildProductionResidentSelectionContract,
  buildProductionResidentWorksetContract
} from './common_4dgs_production_frame_data_contracts.js';

const ESTIMATED_PRODUCTION_GPU_BYTES_PER_RECORD = 256;
const worksetCache = new WeakMap();
const sceneIdentityCache = new WeakMap();
let nextSceneResourceIdentity = 1;
let nextWorksetResourceIdentity = 1;

function finiteInteger(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.floor(number)) : fallback;
}

function resolveSceneRecordCount(raw) {
  return finiteInteger(
    raw?.count ?? raw?.N ??
      ((raw?.xyz?.length ?? 0) / Math.max(1, finiteInteger(raw?.xyzDim, 3)))
  );
}

function resolveResourceCapacityRecords(device) {
  const maxStorageBufferBindingSize = finiteInteger(
    device?.limits?.maxStorageBufferBindingSize,
    128 * 1024 * 1024
  );
  const maxBufferSize = finiteInteger(
    device?.limits?.maxBufferSize,
    maxStorageBufferBindingSize
  );
  const maxComputeWorkgroupsPerDimension = finiteInteger(
    device?.limits?.maxComputeWorkgroupsPerDimension,
    65535
  );
  const limitingBytes = Math.min(maxStorageBufferBindingSize, maxBufferSize);
  return Math.max(
    1,
    Math.min(
      Math.floor(limitingBytes / ESTIMATED_PRODUCTION_GPU_BYTES_PER_RECORD),
      maxComputeWorkgroupsPerDimension * 64
    )
  );
}

function buildPackedRawXyzOpacity(raw, candidateIndices) {
  const result = new Float32Array(candidateIndices.length * 4);
  for (let row = 0; row < candidateIndices.length; row += 1) {
    const sourceIndex = candidateIndices[row];
    const xyzOffset = sourceIndex * Math.max(1, finiteInteger(raw?.xyzDim, 3));
    const outputOffset = row * 4;
    result[outputOffset + 0] = Number(raw?.xyz?.[xyzOffset + 0]) || 0;
    result[outputOffset + 1] = Number(raw?.xyz?.[xyzOffset + 1]) || 0;
    result[outputOffset + 2] = Number(raw?.xyz?.[xyzOffset + 2]) || 0;
    result[outputOffset + 3] = Number(
      raw?.opacity?.[
        sourceIndex * Math.max(1, finiteInteger(raw?.opacityDim, 1))
      ]
    ) || 0;
  }
  return result;
}

export function selectActiveProductionResidentWorkset({
  raw,
  device,
  canvasWidth,
  canvasHeight,
  productionResidentSelectionRequest = null
} = {}) {
  const sceneRecordCount = resolveSceneRecordCount(raw);
  const resourceCapacityRecords = resolveResourceCapacityRecords(device);
  const residentSelectionContract = buildProductionResidentSelectionContract({
    request: productionResidentSelectionRequest,
    sceneRecordCount,
    resourceCapacityRecords
  });
  const residentStart = residentSelectionContract.appliedStart ?? 0;
  const residentRecordCount = residentSelectionContract.appliedRecordCount;
  const sceneResourceIdentity = raw
    ? sceneIdentityCache.get(raw) ??
      `production-scene-resource-${nextSceneResourceIdentity++}`
    : `production-scene-resource-${nextSceneResourceIdentity++}`;
  if (raw && !sceneIdentityCache.has(raw)) {
    sceneIdentityCache.set(raw, sceneResourceIdentity);
  }
  const cacheKey = [
    sceneResourceIdentity,
    sceneRecordCount,
    residentStart,
    residentRecordCount,
    resourceCapacityRecords,
    residentSelectionContract.selectionPolicy
  ].join(':');
  const cached = raw && worksetCache.get(raw);
  if (
    residentSelectionContract.productionResidentSelectionReady === true &&
    cached?.cacheKey === cacheKey
  ) return cached.workset;

  const resourceIdentity =
    `production-resident-workset-${nextWorksetResourceIdentity++}`;
  const candidateIndices = new Uint32Array(residentRecordCount);
  for (let index = 0; index < residentRecordCount; index += 1) {
    candidateIndices[index] = residentStart + index;
  }
  const rawXyzOpacity = buildPackedRawXyzOpacity(raw, candidateIndices);
  const contract = buildProductionResidentWorksetContract({
    resourceIdentity,
    sceneResourceIdentity,
    sceneRecordCount,
    residentStart,
    residentRecordCount,
    resourceCapacityRecords,
    residentSelectionContract,
    selectionPolicy: residentSelectionContract.selectionPolicy,
    diagnosticMaxRecordsUsed: false,
    diagnosticCandidateSourceUsed: false,
    nonResidentRecordsExplicit: true,
    overflowPolicy: 'fail-closed-before-compositor-promotion',
    reason: residentSelectionContract.reason
  });
  const workset = {
    candidateIndices,
    rawXyzOpacity,
    contract
  };
  if (
    raw &&
    residentSelectionContract.productionResidentSelectionReady === true
  ) worksetCache.set(raw, { cacheKey, workset });
  return workset;
}
