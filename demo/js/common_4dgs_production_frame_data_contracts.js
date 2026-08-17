export const PRODUCTION_WORKSET_CONTRACT_VERSION =
  'phase3-production-resident-workset-v1';

export const PRODUCTION_FRAME_DATA_PATH_CONTRACT_VERSION =
  'phase3-native-webgpu-production-frame-data-path-v1';

export const PRODUCTION_TILE_INPUT_CONTRACT_VERSION =
  'phase3-native-webgpu-production-tile-input-v1';

function finiteInteger(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.floor(number)) : fallback;
}

function stringOrNull(value) {
  return typeof value === 'string' && value.length > 0 ? value : null;
}

export function buildProductionResidentWorksetContract({
  status = 'ok',
  resourceIdentity = null,
  sceneResourceIdentity = null,
  sceneRecordCount = 0,
  residentStart = 0,
  residentRecordCount = 0,
  resourceCapacityRecords = 0,
  selectionPolicy = 'scene-resource-capacity-bounded-resident-range',
  diagnosticMaxRecordsUsed = false,
  diagnosticCandidateSourceUsed = false,
  nonResidentRecordsExplicit = true,
  overflowPolicy = 'fail-closed-before-compositor-promotion',
  reason = null
} = {}) {
  const normalizedSceneCount = finiteInteger(sceneRecordCount);
  const normalizedResidentStart = finiteInteger(residentStart);
  const normalizedResidentCount = finiteInteger(residentRecordCount);
  const normalizedResourceCapacity = finiteInteger(resourceCapacityRecords);
  const residentRangeInBounds =
    normalizedResidentStart + normalizedResidentCount <= normalizedSceneCount;
  const ready =
    status === 'ok' &&
    stringOrNull(resourceIdentity) !== null &&
    stringOrNull(sceneResourceIdentity) !== null &&
    normalizedResidentCount > 0 &&
    normalizedResidentCount <= normalizedResourceCapacity &&
    residentRangeInBounds &&
    diagnosticMaxRecordsUsed === false &&
    diagnosticCandidateSourceUsed === false &&
    nonResidentRecordsExplicit === true &&
    overflowPolicy === 'fail-closed-before-compositor-promotion';
  return {
    contractVersion: PRODUCTION_WORKSET_CONTRACT_VERSION,
    status: ready ? 'ok' : status === 'ok' ? 'blocked' : status,
    residentWorksetReady: ready,
    resourceIdentity: stringOrNull(resourceIdentity),
    sceneResourceIdentity: stringOrNull(sceneResourceIdentity),
    sceneRecordCount: normalizedSceneCount,
    residentStart: normalizedResidentStart,
    residentRecordCount: normalizedResidentCount,
    nonResidentRecordCount: Math.max(
      0,
      normalizedSceneCount - normalizedResidentCount
    ),
    sceneFullyResident: normalizedResidentCount === normalizedSceneCount,
    resourceCapacityRecords: normalizedResourceCapacity,
    tileReferenceCapacityCoupledToRecordSelection: false,
    residentRangeInBounds,
    selectionPolicy,
    diagnosticMaxRecordsUsed: diagnosticMaxRecordsUsed === true,
    diagnosticCandidateSourceUsed: diagnosticCandidateSourceUsed === true,
    nonResidentRecordsExplicit: nonResidentRecordsExplicit === true,
    overflowPolicy,
    streamingImplemented: false,
    lodImplemented: false,
    reason: ready ? null : reason ?? 'production-resident-workset-not-ready'
  };
}

export function buildNativeWebGpuProductionFrameDataPathContract({
  status = 'ok',
  worksetContract = null,
  stateResourceIdentity = null,
  attributeResourceIdentity = null,
  footprintResourceIdentity = null,
  tileInputResourceIdentity = null,
  tileListInputResourceIdentity = null,
  compositorInputResourceIdentity = null,
  stateRecordCount = 0,
  tileInputRecordCount = 0,
  tileReferenceCapacityContract = null,
  boundedExecutionContract = null,
  gpuExecutionPlanContract = null,
  terminalExecutionPlanObserver = null,
  cpuReferenceUsedAsProductionInput = false,
  diagnosticReadbackUsedAsProductionInput = false,
  javascriptVisibleSamplesUsedAsProductionInput = false,
  diagnosticMaxRecordsUsedAsProductionLimit = false,
  gpuResourceLineagePreserved = false,
  capacityOverflowDetected = false,
  capacityOverflowFailClosed = false,
  silentDropAllowed = false,
  compositorSubmitted = false,
  reason = null
} = {}) {
  const worksetCount = finiteInteger(worksetContract?.residentRecordCount);
  const normalizedStateCount = finiteInteger(stateRecordCount);
  const normalizedTileInputCount = finiteInteger(tileInputRecordCount);
  const resourceIdentities = {
    workset: stringOrNull(worksetContract?.resourceIdentity),
    state: stringOrNull(stateResourceIdentity),
    attributes: stringOrNull(attributeResourceIdentity),
    footprint: stringOrNull(footprintResourceIdentity),
    tileInput: stringOrNull(tileInputResourceIdentity),
    tileListInput: stringOrNull(tileListInputResourceIdentity),
    compositorInput: stringOrNull(compositorInputResourceIdentity)
  };
  const allResourceIdentitiesPresent = Object.values(resourceIdentities).every(
    (value) => value !== null
  );
  const diagnosticIndependent =
    cpuReferenceUsedAsProductionInput === false &&
    diagnosticReadbackUsedAsProductionInput === false &&
    javascriptVisibleSamplesUsedAsProductionInput === false &&
    diagnosticMaxRecordsUsedAsProductionLimit === false;
  const countsMatch =
    worksetCount > 0 &&
    normalizedStateCount === worksetCount &&
    normalizedTileInputCount === worksetCount;
  const capacityReady =
    (
      tileReferenceCapacityContract?.tileReferenceCapacityReady === true ||
      gpuExecutionPlanContract?.gpuExecutionPlanReady === true
    ) &&
    (capacityOverflowDetected === false || capacityOverflowFailClosed === true);
  const executionReady =
    boundedExecutionContract?.boundedExecutionReady === true &&
    (
      gpuExecutionPlanContract == null ||
      gpuExecutionPlanContract?.gpuExecutionPlanReady === true
    );
  const ready =
    status === 'ok' &&
    worksetContract?.residentWorksetReady === true &&
    allResourceIdentitiesPresent &&
    diagnosticIndependent &&
    countsMatch &&
    gpuResourceLineagePreserved === true &&
    capacityReady &&
    executionReady &&
    silentDropAllowed === false &&
    compositorSubmitted === true;
  return {
    contractVersion: PRODUCTION_FRAME_DATA_PATH_CONTRACT_VERSION,
    status: ready ? 'ok' : status === 'ok' ? 'blocked' : status,
    nativeProductionFrameDataPathReady: ready,
    worksetContractVersion: worksetContract?.contractVersion ?? null,
    worksetResourceIdentity: resourceIdentities.workset,
    resourceIdentities,
    allResourceIdentitiesPresent,
    stateRecordCount: normalizedStateCount,
    tileInputRecordCount: normalizedTileInputCount,
    tileReferenceCapacityContract,
    boundedExecutionContract,
    gpuExecutionPlanContract,
    terminalExecutionPlanObserver,
    tileReferenceCapacityReady:
      tileReferenceCapacityContract?.tileReferenceCapacityReady === true,
    worksetRecordCount: worksetCount,
    countsMatch,
    cpuReferenceUsedAsProductionInput:
      cpuReferenceUsedAsProductionInput === true,
    diagnosticReadbackUsedAsProductionInput:
      diagnosticReadbackUsedAsProductionInput === true,
    javascriptVisibleSamplesUsedAsProductionInput:
      javascriptVisibleSamplesUsedAsProductionInput === true,
    diagnosticMaxRecordsUsedAsProductionLimit:
      diagnosticMaxRecordsUsedAsProductionLimit === true,
    diagnosticIndependent,
    gpuResourceLineagePreserved: gpuResourceLineagePreserved === true,
    capacityOverflowDetected: capacityOverflowDetected === true,
    capacityOverflowFailClosed: capacityOverflowFailClosed === true,
    capacityReady,
    executionReady,
    silentDropAllowed: silentDropAllowed === true,
    compositorSubmitted: compositorSubmitted === true,
    streamingImplemented: false,
    lodImplemented: false,
    reason: ready ? null : reason ?? 'native-production-frame-data-path-not-ready'
  };
}

export function buildNativeWebGpuProductionTileInputContract({
  status = 'ok',
  sourceWorksetResourceIdentity = null,
  sourceStateResourceIdentity = null,
  resourceIdentity = null,
  recordCount = 0,
  dispatchSubmitted = false,
  productionReadbackPerformed = false,
  javascriptVisibleSamplesMaterialized = false,
  reason = null
} = {}) {
  const normalizedRecordCount = finiteInteger(recordCount);
  const ready =
    status === 'ok' &&
    stringOrNull(sourceWorksetResourceIdentity) !== null &&
    stringOrNull(sourceStateResourceIdentity) !== null &&
    stringOrNull(resourceIdentity) !== null &&
    normalizedRecordCount > 0 &&
    dispatchSubmitted === true &&
    productionReadbackPerformed === false &&
    javascriptVisibleSamplesMaterialized === false;
  return {
    contractVersion: PRODUCTION_TILE_INPUT_CONTRACT_VERSION,
    status: ready ? 'ok' : status === 'ok' ? 'blocked' : status,
    tileAwareRenderInputReady: ready,
    tileAwareConsumerReady: ready,
    tileAwareConsumerConsumed: ready,
    generationMode: 'native-webgpu-production-resource-handoff',
    sourceWorksetResourceIdentity:
      stringOrNull(sourceWorksetResourceIdentity),
    sourceStateResourceIdentity: stringOrNull(sourceStateResourceIdentity),
    resourceIdentity: stringOrNull(resourceIdentity),
    generatedTileRecordCount: normalizedRecordCount,
    recordCount: normalizedRecordCount,
    dispatchSubmitted: dispatchSubmitted === true,
    productionReadbackPerformed: productionReadbackPerformed === true,
    javascriptVisibleSamplesMaterialized:
      javascriptVisibleSamplesMaterialized === true,
    tilePayloadClassification:
      'native-webgpu-production-tile-input-storage-buffer',
    generatedPayloadFields: [
      'screen-center',
      'radius',
      'depth',
      'conic',
      'sort-key',
      'color-alpha'
    ],
    reason: ready ? null : reason ?? 'native-production-tile-input-not-ready'
  };
}
