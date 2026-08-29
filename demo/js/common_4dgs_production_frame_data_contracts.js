export const PRODUCTION_WORKSET_CONTRACT_VERSION =
  'phase3-production-resident-workset-v1';

export const PRODUCTION_RESIDENT_SELECTION_CONTRACT_VERSION =
  'phase3-production-resident-selection-v1';

export const PRODUCTION_FRAME_DATA_PATH_CONTRACT_VERSION =
  'phase3-native-webgpu-production-frame-data-path-v1';

export const PRODUCTION_EVALUATION_INPUT_CONTRACT_VERSION =
  'phase3-production-evaluation-input-v1';

export const PRODUCTION_TILE_INPUT_CONTRACT_VERSION =
  'phase3-native-webgpu-production-tile-input-v1';

const PRODUCTION_EVALUATION_PROJECTION_FLOAT_COUNT = 44;
const PRODUCTION_EVALUATION_SNAPSHOT_MAX_DEPTH = 8;
const PRODUCTION_EVALUATION_SNAPSHOT_MAX_ARRAY_LENGTH = 64;
const PRODUCTION_EVALUATION_SNAPSHOT_MAX_OBJECT_KEYS = 64;
const PRODUCTION_EVALUATION_SNAPSHOT_MAX_STRING_LENGTH = 1024;

function finiteInteger(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.floor(number)) : fallback;
}

function stringOrNull(value) {
  return typeof value === 'string' && value.length > 0 ? value : null;
}

function isProvided(value) {
  return value !== null && value !== undefined;
}

function safeIntegerOrNull(value) {
  if (!isProvided(value) || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) && Number.isSafeInteger(number)
    ? number
    : null;
}

function isPlainObject(value) {
  if (value === null || typeof value !== 'object') return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function cloneBoundedPlainValue(
  value,
  path,
  seen = new Set(),
  depth = 0
) {
  if (depth > PRODUCTION_EVALUATION_SNAPSHOT_MAX_DEPTH) {
    throw new TypeError(`${path}-maximum-depth-exceeded`);
  }
  if (value === null || typeof value === 'boolean') return value;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new TypeError(`${path}-finite-number-required`);
    return value;
  }
  if (typeof value === 'string') {
    if (value.length > PRODUCTION_EVALUATION_SNAPSHOT_MAX_STRING_LENGTH) {
      throw new TypeError(`${path}-string-too-long`);
    }
    return value;
  }
  if (typeof value !== 'object') {
    throw new TypeError(`${path}-unsupported-value`);
  }
  if (seen.has(value)) throw new TypeError(`${path}-circular-value`);
  seen.add(value);
  try {
    if (Array.isArray(value)) {
      if (value.length > PRODUCTION_EVALUATION_SNAPSHOT_MAX_ARRAY_LENGTH) {
        throw new TypeError(`${path}-array-too-long`);
      }
      return value.map((item, index) =>
        cloneBoundedPlainValue(item, `${path}[${index}]`, seen, depth + 1)
      );
    }
    if (!isPlainObject(value)) {
      throw new TypeError(`${path}-plain-object-required`);
    }
    const keys = Reflect.ownKeys(value);
    if (keys.some((key) => typeof key !== 'string')) {
      throw new TypeError(`${path}-symbol-key-not-supported`);
    }
    if (keys.length > PRODUCTION_EVALUATION_SNAPSHOT_MAX_OBJECT_KEYS) {
      throw new TypeError(`${path}-too-many-fields`);
    }
    const clone = {};
    for (const key of keys) {
      clone[key] = cloneBoundedPlainValue(
        value[key],
        `${path}.${key}`,
        seen,
        depth + 1
      );
    }
    return clone;
  } finally {
    seen.delete(value);
  }
}

function deepFreeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}

function addBlockedReason(blockedReasons, condition, reason) {
  if (condition && !blockedReasons.includes(reason)) blockedReasons.push(reason);
}

function finiteNumberOrNull(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function positiveSafeIntegerOrNull(value) {
  return Number.isSafeInteger(value) && value > 0 ? value : null;
}

function nonNegativeSafeIntegerOrNull(value) {
  return Number.isSafeInteger(value) && value >= 0 ? value : null;
}

function isSha256Hex(value) {
  return typeof value === 'string' && /^[0-9a-fA-F]{64}$/.test(value);
}

function sameFloat32Value(left, right) {
  return Number.isFinite(left) &&
    Number.isFinite(right) &&
    Math.fround(left) === Math.fround(right);
}

function pickProductionFrameIdentity(identity) {
  if (!identity) return null;
  return {
    generation: nonNegativeSafeIntegerOrNull(identity.generation),
    frameHash: stringOrNull(identity.frameHash),
    datasetCameraLabel: stringOrNull(identity.datasetCameraLabel),
    datasetFrameNumber: nonNegativeSafeIntegerOrNull(identity.datasetFrameNumber),
    datasetTime: finiteNumberOrNull(identity.datasetTime),
    referenceCameraLabel: stringOrNull(identity.referenceCameraLabel),
    outputWidth: positiveSafeIntegerOrNull(identity.outputWidth),
    outputHeight: positiveSafeIntegerOrNull(identity.outputHeight)
  };
}

/**
 * Builds observer-only evaluation lineage from the exact local values used by
 * a completed production frame. This contract is deliberately bounded and
 * never participates in production readiness or GPU control.
 */
export function buildProductionEvaluationInputContract({
  assetIdentity = null,
  worksetContract = null,
  buildConfig = null,
  effectiveRenderScale = null,
  renderWidth = null,
  renderHeight = null,
  canvasWidth = null,
  canvasHeight = null,
  projectionData = null,
  projectionSummary = null,
  cameraIdentity = null,
  timeIdentity = null,
  requestIdentity = null,
  productionIdentity = null,
  orientationIdentity = null
} = {}) {
  const blockedReasons = [];
  let asset = null;
  let workset = null;
  let projection = null;
  let camera = null;
  let time = null;
  let request = null;
  let production = null;
  let orientation = null;

  try {
    const source = cloneBoundedPlainValue(
      assetIdentity,
      'production-evaluation-asset-identity'
    );
    asset = {
      sceneResourceIdentity: stringOrNull(source.sceneResourceIdentity),
      spl4AssetIdentity: {
        sha256: stringOrNull(source.assetSha256),
        sizeBytes: positiveSafeIntegerOrNull(source.assetSizeBytes),
        sourceKind: stringOrNull(source.assetSourceKind),
        format: stringOrNull(source.assetFormat),
        formatVersion: positiveSafeIntegerOrNull(source.formatVersion),
        recordCount: positiveSafeIntegerOrNull(source.recordCount),
        dimensions: {
          xyz: positiveSafeIntegerOrNull(source.dimensions?.xyz),
          rotation: positiveSafeIntegerOrNull(source.dimensions?.rotation),
          rotationR: positiveSafeIntegerOrNull(source.dimensions?.rotationR),
          scaleXyz: positiveSafeIntegerOrNull(source.dimensions?.scaleXyz),
          fDc: positiveSafeIntegerOrNull(source.dimensions?.fDc),
          fRest: nonNegativeSafeIntegerOrNull(source.dimensions?.fRest),
          opacity: positiveSafeIntegerOrNull(source.dimensions?.opacity),
          time: positiveSafeIntegerOrNull(source.dimensions?.time),
          scaleTime: positiveSafeIntegerOrNull(source.dimensions?.scaleTime)
        }
      }
    };
  } catch (_error) {
    blockedReasons.push('production-evaluation-asset-identity-not-bounded-plain-data');
  }
  const assetValues = asset?.spl4AssetIdentity ?? {};
  addBlockedReason(
    blockedReasons,
    asset == null ||
      asset.sceneResourceIdentity == null ||
      !isSha256Hex(assetValues.sha256) ||
      assetValues.sizeBytes == null ||
      assetValues.sourceKind == null ||
      assetValues.format !== 'SPL4-v2' ||
      assetValues.formatVersion !== 2 ||
      assetValues.recordCount == null ||
      Object.values(assetValues.dimensions ?? {}).some((value) => value == null),
    'production-evaluation-asset-identity-missing-or-unsupported'
  );

  try {
    const source = cloneBoundedPlainValue(
      worksetContract,
      'production-evaluation-workset-identity'
    );
    const selection = source.residentSelectionContract ?? {};
    workset = {
      resourceIdentity: stringOrNull(source.resourceIdentity),
      sceneResourceIdentity: stringOrNull(source.sceneResourceIdentity),
      sceneRecordCount: positiveSafeIntegerOrNull(source.sceneRecordCount),
      residentStart: nonNegativeSafeIntegerOrNull(source.residentStart),
      residentRecordCount: positiveSafeIntegerOrNull(source.residentRecordCount),
      residentEndExclusive:
        positiveSafeIntegerOrNull(source.residentEndExclusive),
      selection: {
        requestProvided: selection.requestProvided === true,
        requestMode: stringOrNull(selection.requestMode),
        requestedStart: nonNegativeSafeIntegerOrNull(selection.requestedStart),
        requestedRecordCount:
          positiveSafeIntegerOrNull(selection.requestedRecordCount),
        requestedEndExclusive:
          positiveSafeIntegerOrNull(selection.requestedEndExclusive),
        appliedStart: nonNegativeSafeIntegerOrNull(selection.appliedStart),
        appliedRecordCount:
          positiveSafeIntegerOrNull(selection.appliedRecordCount),
        appliedEndExclusive:
          positiveSafeIntegerOrNull(selection.appliedEndExclusive),
        selectionPolicy: stringOrNull(selection.selectionPolicy),
        sourceIndexSpace: stringOrNull(selection.sourceIndexSpace),
        residentRowSpace: stringOrNull(selection.residentRowSpace),
        productionResidentSelectionReady:
          selection.productionResidentSelectionReady === true
      },
      selectionPolicy: stringOrNull(source.selectionPolicy),
      sourceIndexSpace: stringOrNull(selection.sourceIndexSpace),
      residentRowSpace: stringOrNull(selection.residentRowSpace),
      productionResidentSelectionReady:
        selection.productionResidentSelectionReady === true,
      residentSelectionMatches: source.residentSelectionMatches === true,
      residentWorksetReady: source.residentWorksetReady === true,
      diagnosticCandidateSourceUsed:
        source.diagnosticCandidateSourceUsed === true,
      diagnosticMaxRecordsUsed: source.diagnosticMaxRecordsUsed === true,
      nonResidentRecordsExplicit: source.nonResidentRecordsExplicit === true,
      overflowPolicy: stringOrNull(source.overflowPolicy),
      candidateOrder: 'resident-row-ascending-original-source-index',
      sourceIndexMapping: 'srcIndex=residentStart+residentRow',
      silentOmissionAllowed: false
    };
  } catch (_error) {
    blockedReasons.push('production-evaluation-workset-identity-not-bounded-plain-data');
  }
  const selection = workset?.selection ?? {};
  const topLevelRangeMatches = workset != null &&
    workset.residentStart === selection.appliedStart &&
    workset.residentRecordCount === selection.appliedRecordCount &&
    workset.residentEndExclusive === selection.appliedEndExclusive;
  const requestedRangeMatches = workset != null &&
    (
      selection.requestProvided === false ||
      (
        selection.requestMode === 'range' &&
        selection.requestedStart === selection.appliedStart &&
        selection.requestedRecordCount === selection.appliedRecordCount &&
        selection.requestedEndExclusive === selection.appliedEndExclusive
      )
    );
  addBlockedReason(
    blockedReasons,
    workset == null ||
      workset.resourceIdentity == null ||
      workset.sceneResourceIdentity == null ||
      workset.sceneRecordCount == null ||
      workset.residentStart == null ||
      workset.residentRecordCount == null ||
      workset.residentEndExclusive == null ||
      workset.residentEndExclusive !==
        workset.residentStart + workset.residentRecordCount ||
      workset.residentEndExclusive > workset.sceneRecordCount ||
      workset.productionResidentSelectionReady !== true ||
      workset.residentSelectionMatches !== true ||
      workset.residentWorksetReady !== true ||
      workset.selectionPolicy !== selection.selectionPolicy ||
      topLevelRangeMatches !== true ||
      requestedRangeMatches !== true,
    'production-evaluation-workset-range-or-readiness-mismatch'
  );
  addBlockedReason(
    blockedReasons,
    workset != null &&
      (
        workset.sourceIndexSpace !== 'spl4-original-source-index' ||
        workset.residentRowSpace !== 'active-resident-workset-local-row' ||
        workset.selection.sourceIndexSpace !== 'spl4-original-source-index' ||
        workset.selection.residentRowSpace !==
          'active-resident-workset-local-row'
      ),
    'production-evaluation-workset-index-space-or-ordering-mismatch'
  );
  addBlockedReason(
    blockedReasons,
    workset != null &&
      (
        workset.diagnosticCandidateSourceUsed !== false ||
        workset.diagnosticMaxRecordsUsed !== false ||
        workset.nonResidentRecordsExplicit !== true ||
        workset.overflowPolicy !==
          'fail-closed-before-compositor-promotion'
      ),
    'production-evaluation-workset-production-semantics-mismatch'
  );
  addBlockedReason(
    blockedReasons,
    asset != null && workset != null &&
      (
        asset.sceneResourceIdentity !== workset.sceneResourceIdentity ||
        asset.spl4AssetIdentity.recordCount !== workset.sceneRecordCount
      ),
    'production-evaluation-asset-workset-identity-mismatch'
  );

  const timestamp = finiteNumberOrNull(buildConfig?.timestamp);
  const scalingModifier = finiteNumberOrNull(buildConfig?.scalingModifier);
  const sigmaScale = finiteNumberOrNull(buildConfig?.sigmaScale);
  const prefilterVar = finiteNumberOrNull(buildConfig?.prefilterVar);
  const configuredRenderScale = finiteNumberOrNull(buildConfig?.renderScale);
  const normalizedEffectiveRenderScale = finiteNumberOrNull(effectiveRenderScale);
  const normalizedRenderWidth = positiveSafeIntegerOrNull(renderWidth);
  const normalizedRenderHeight = positiveSafeIntegerOrNull(renderHeight);
  const normalizedCanvasWidth = positiveSafeIntegerOrNull(canvasWidth);
  const normalizedCanvasHeight = positiveSafeIntegerOrNull(canvasHeight);
  const appliedConfig = {
    timestamp,
    scalingModifier,
    sigmaScale,
    prefilterVar,
    configuredRenderScale,
    effectiveRenderScale: normalizedEffectiveRenderScale,
    renderWidth: normalizedRenderWidth,
    renderHeight: normalizedRenderHeight,
    canvasWidth: normalizedCanvasWidth,
    canvasHeight: normalizedCanvasHeight
  };
  addBlockedReason(
    blockedReasons,
    timestamp == null ||
      scalingModifier == null ||
      sigmaScale == null ||
      prefilterVar == null ||
      configuredRenderScale == null ||
      configuredRenderScale <= 0 ||
      normalizedEffectiveRenderScale == null ||
      normalizedEffectiveRenderScale <= 0 ||
      configuredRenderScale !== normalizedEffectiveRenderScale ||
      normalizedRenderWidth == null ||
      normalizedRenderHeight == null ||
      normalizedCanvasWidth == null ||
      normalizedCanvasHeight == null ||
      normalizedRenderWidth !== Math.max(
        1,
        Math.round(normalizedCanvasWidth * normalizedEffectiveRenderScale)
      ) ||
      normalizedRenderHeight !== Math.max(
        1,
        Math.round(normalizedCanvasHeight * normalizedEffectiveRenderScale)
      ),
    'production-evaluation-applied-config-invalid'
  );

  if (
    projectionData instanceof Float32Array &&
    projectionData.length === PRODUCTION_EVALUATION_PROJECTION_FLOAT_COUNT &&
    Array.from(projectionData).every(Number.isFinite)
  ) {
    try {
      const summary = cloneBoundedPlainValue(
        projectionSummary,
        'production-evaluation-projection-summary'
      );
      projection = {
        floatCount: PRODUCTION_EVALUATION_PROJECTION_FLOAT_COUNT,
        values: Array.from(projectionData),
        summary: {
          schemaVersion: stringOrNull(summary.schemaVersion),
          mode: stringOrNull(summary.mode),
          projectionContract: stringOrNull(summary.projectionContract),
          sourcePositionMode: stringOrNull(summary.sourcePositionMode),
          renderW: positiveSafeIntegerOrNull(summary.renderW),
          renderH: positiveSafeIntegerOrNull(summary.renderH),
          sx: finiteNumberOrNull(summary.sx),
          sy: finiteNumberOrNull(summary.sy),
          pixelXSign: finiteNumberOrNull(summary.pixelXSign),
          intrinsics: {
            fx: finiteNumberOrNull(summary.intrinsics?.fx),
            fy: finiteNumberOrNull(summary.intrinsics?.fy),
            cx: finiteNumberOrNull(summary.intrinsics?.cx),
            cy: finiteNumberOrNull(summary.intrinsics?.cy)
          },
          viewMatrixSource: stringOrNull(summary.viewMatrixSource),
          projectionMatrixSource: stringOrNull(summary.projectionMatrixSource)
        },
        viewMatrixValues: { offset: 12, count: 16 },
        projectionMatrixValues: { offset: 28, count: 16 }
      };
    } catch (_error) {
      blockedReasons.push('production-evaluation-projection-summary-not-bounded-plain-data');
    }
  } else {
    blockedReasons.push('production-evaluation-projection-data-invalid');
  }
  const projectionValues = projection?.values ?? [];
  const projectionContractMode = projection?.summary?.mode === 'cuda-aligned'
    ? 1
    : projection?.summary?.mode === 'threejs'
      ? 0
      : null;
  addBlockedReason(
    blockedReasons,
    projection == null ||
      projectionContractMode == null ||
      projection.summary.schemaVersion == null ||
      projection.summary.projectionContract == null ||
      projection.summary.sourcePositionMode == null ||
      projection.summary.viewMatrixSource == null ||
      projection.summary.projectionMatrixSource == null ||
      projection.summary.renderW !== normalizedRenderWidth ||
      projection.summary.renderH !== normalizedRenderHeight ||
      projectionValues[0] !== projectionContractMode ||
      projectionValues[1] !== normalizedRenderWidth ||
      projectionValues[2] !== normalizedRenderHeight ||
      !sameFloat32Value(
        projection.summary.sx,
        normalizedCanvasWidth / normalizedRenderWidth
      ) ||
      !sameFloat32Value(
        projection.summary.sy,
        normalizedCanvasHeight / normalizedRenderHeight
      ) ||
      !sameFloat32Value(projection.summary.sx, projectionValues[4]) ||
      !sameFloat32Value(projection.summary.sy, projectionValues[5]) ||
      !sameFloat32Value(projection.summary.pixelXSign, projectionValues[6]) ||
      !sameFloat32Value(projection.summary.intrinsics.fx, projectionValues[8]) ||
      !sameFloat32Value(projection.summary.intrinsics.fy, projectionValues[9]) ||
      !sameFloat32Value(projection.summary.intrinsics.cx, projectionValues[10]) ||
      !sameFloat32Value(projection.summary.intrinsics.cy, projectionValues[11]),
    'production-evaluation-projection-summary-mismatch'
  );

  try {
    const source = cloneBoundedPlainValue(
      cameraIdentity,
      'production-evaluation-camera-identity'
    );
    camera = {
      cameraLabel: stringOrNull(source.cameraLabel),
      referenceCameraLabel: stringOrNull(source.referenceCameraLabel),
      datasetFrameNumber: nonNegativeSafeIntegerOrNull(source.datasetFrameNumber),
      datasetViewId: nonNegativeSafeIntegerOrNull(source.datasetViewId),
      cameraSource: stringOrNull(source.cameraSource),
      datasetViewMatrixMode: stringOrNull(source.datasetViewMatrixMode),
      fixedReferenceCameraActivationMode:
        stringOrNull(source.fixedReferenceCameraActivationMode),
      cameraControlContract: stringOrNull(source.cameraControlContract),
      cameraOrientationPolicy: stringOrNull(source.cameraOrientationPolicy),
      projectionViewMatrixValues: { offset: 12, count: 16 }
    };
  } catch (_error) {
    blockedReasons.push('production-evaluation-camera-identity-not-bounded-plain-data');
  }
  addBlockedReason(
    blockedReasons,
    camera == null ||
      camera.cameraLabel == null ||
      camera.referenceCameraLabel == null ||
      camera.datasetFrameNumber == null ||
      camera.datasetViewId == null ||
      camera.cameraSource == null ||
      camera.datasetViewMatrixMode == null ||
      camera.cameraControlContract == null ||
      camera.cameraOrientationPolicy == null,
    'production-evaluation-camera-identity-missing-or-invalid'
  );

  try {
    const source = cloneBoundedPlainValue(
      timeIdentity,
      'production-evaluation-time-identity'
    );
    time = {
      requestedDatasetTime: finiteNumberOrNull(source.requestedDatasetTime),
      requestedViewerTime: finiteNumberOrNull(source.requestedViewerTime),
      actualAppliedTimestamp: finiteNumberOrNull(source.actualAppliedTimestamp),
      requestedDatasetTimeMatchesAppliedTimestamp: false
    };
    time.requestedDatasetTimeMatchesAppliedTimestamp =
      time.requestedDatasetTime === time.actualAppliedTimestamp;
  } catch (_error) {
    blockedReasons.push('production-evaluation-time-identity-not-bounded-plain-data');
  }
  addBlockedReason(
    blockedReasons,
    time == null ||
      time.requestedDatasetTime == null ||
      time.actualAppliedTimestamp == null ||
      time.actualAppliedTimestamp !== timestamp ||
      time.requestedDatasetTimeMatchesAppliedTimestamp !== true,
    'production-evaluation-time-identity-missing-or-invalid'
  );

  try {
    const source = cloneBoundedPlainValue(
      requestIdentity,
      'production-evaluation-request-identity'
    );
    request = {
      schedulerRequestIdentity: stringOrNull(source.schedulerRequestIdentity),
      schedulerRequestSource: stringOrNull(source.schedulerRequestSource),
      sourceRequestIdentity: stringOrNull(source.sourceRequestIdentity),
      schedulerFrameIndex:
        nonNegativeSafeIntegerOrNull(source.schedulerFrameIndex)
    };
  } catch (_error) {
    blockedReasons.push('production-evaluation-request-identity-not-bounded-plain-data');
  }
  addBlockedReason(
    blockedReasons,
    request == null ||
      request.schedulerRequestIdentity == null ||
      request.schedulerRequestSource == null ||
      request.sourceRequestIdentity == null ||
      request.schedulerFrameIndex == null ||
      request.schedulerRequestIdentity !== request.sourceRequestIdentity,
    'production-evaluation-request-identity-missing-or-mismatched'
  );

  try {
    const source = cloneBoundedPlainValue(
      productionIdentity,
      'production-evaluation-production-identity'
    );
    production = {
      productionGeneration:
        nonNegativeSafeIntegerOrNull(source.productionGeneration),
      presentedGeneration:
        nonNegativeSafeIntegerOrNull(source.presentedGeneration),
      productionFrameIdentity:
        pickProductionFrameIdentity(source.productionFrameIdentity),
      presentedFrameIdentity:
        pickProductionFrameIdentity(source.presentedFrameIdentity)
    };
  } catch (_error) {
    blockedReasons.push('production-evaluation-production-identity-not-bounded-plain-data');
  }
  const productionFrame = production?.productionFrameIdentity;
  const presentedFrame = production?.presentedFrameIdentity;
  const identityFields = [
    'generation',
    'datasetCameraLabel',
    'datasetFrameNumber',
    'datasetTime',
    'referenceCameraLabel',
    'outputWidth',
    'outputHeight'
  ];
  const productionAndPresentedMatch = productionFrame != null &&
    presentedFrame != null &&
    identityFields.every((field) => productionFrame[field] === presentedFrame[field]);
  addBlockedReason(
    blockedReasons,
    production == null ||
      production.productionGeneration == null ||
      production.presentedGeneration == null ||
      productionFrame == null ||
      presentedFrame == null ||
      identityFields.some((field) => productionFrame?.[field] == null) ||
      identityFields.some((field) => presentedFrame?.[field] == null) ||
      production.productionGeneration !== productionFrame?.generation ||
      production.presentedGeneration !== presentedFrame?.generation ||
      productionAndPresentedMatch !== true ||
      productionFrame?.datasetCameraLabel !== camera?.cameraLabel ||
      productionFrame?.referenceCameraLabel !== camera?.referenceCameraLabel ||
      productionFrame?.datasetFrameNumber !== camera?.datasetFrameNumber ||
      productionFrame?.datasetTime !== time?.requestedDatasetTime ||
      productionFrame?.datasetTime !== time?.actualAppliedTimestamp ||
      presentedFrame?.datasetTime !== time?.actualAppliedTimestamp ||
      productionFrame?.outputWidth !== normalizedCanvasWidth ||
      productionFrame?.outputHeight !== normalizedCanvasHeight,
    'production-evaluation-frame-identity-missing-or-mismatched'
  );

  try {
    const source = cloneBoundedPlainValue(
      orientationIdentity,
      'production-evaluation-orientation-identity'
    );
    orientation = {
      schemaVersion: stringOrNull(source.schemaVersion),
      productionTextureOrigin: stringOrNull(source.productionTextureOrigin),
      productionTextureYAxisDirection:
        stringOrNull(source.productionTextureYAxisDirection),
      presentationUvTransform: stringOrNull(source.presentationUvTransform),
      presentationVerticalFlipApplied:
        typeof source.presentationVerticalFlipApplied === 'boolean'
          ? source.presentationVerticalFlipApplied
          : null,
      captureReadbackRowOrder: stringOrNull(source.captureReadbackRowOrder),
      pngEncoderRowOrder: stringOrNull(source.pngEncoderRowOrder),
      canonicalPresentationOrientation:
        stringOrNull(source.canonicalPresentationOrientation),
      savedPngOrientation: stringOrNull(source.savedPngOrientation),
      captureVerticalFlipApplied:
        typeof source.captureVerticalFlipApplied === 'boolean'
          ? source.captureVerticalFlipApplied
          : null,
      orientationMismatchClassification:
        stringOrNull(source.orientationMismatchClassification)
    };
  } catch (_error) {
    blockedReasons.push('production-evaluation-orientation-identity-not-bounded-plain-data');
  }
  addBlockedReason(
    blockedReasons,
    orientation == null ||
      orientation.schemaVersion == null ||
      orientation.productionTextureOrigin !== 'texture-memory-top-left' ||
      orientation.productionTextureYAxisDirection !== 'down' ||
      orientation.presentationVerticalFlipApplied !== false ||
      orientation.canonicalPresentationOrientation !==
        'production-texture-top-left-y-down' ||
      orientation.savedPngOrientation !==
        'production-texture-top-left-y-down' ||
      orientation.captureVerticalFlipApplied !== false ||
      orientation.orientationMismatchClassification !== 'none',
    'production-evaluation-orientation-identity-missing-or-mismatched'
  );

  const ready = blockedReasons.length === 0;
  const contract = {
    contractVersion: PRODUCTION_EVALUATION_INPUT_CONTRACT_VERSION,
    status: ready ? 'ready' : 'blocked',
    reason: ready ? null : blockedReasons[0],
    blockedReasons,
    provenance: 'production-frame-exact-local-input-snapshot',
    publicationMode: 'additive-observer-metadata-not-production-control',
    bounded: true,
    immutable: true,
    jsonSerializable: true,
    rawObjectIncluded: false,
    rawArraysIncluded: false,
    candidateIndicesIncluded: false,
    projectionTypedArrayIncluded: false,
    typedArraysIncluded: false,
    gpuResourcesIncluded: false,
    populationScaledArraysIncluded: false,
    assetIdentity: asset,
    productionWorksetIdentity: workset,
    appliedConfig,
    projection,
    cameraIdentity: camera,
    timeIdentity: time,
    requestIdentity: request,
    productionIdentity: production,
    presentationOrientationIdentity: orientation
  };
  try {
    JSON.stringify(contract);
  } catch (_error) {
    contract.status = 'blocked';
    contract.reason = 'production-evaluation-contract-not-json-serializable';
    contract.blockedReasons.push(contract.reason);
    contract.jsonSerializable = false;
  }
  return deepFreeze(contract);
}

export function buildProductionResidentSelectionContract({
  request = null,
  sceneRecordCount = 0,
  resourceCapacityRecords = 0
} = {}) {
  const normalizedSceneCount = finiteInteger(sceneRecordCount);
  const normalizedResourceCapacity = finiteInteger(resourceCapacityRecords);
  const requestObject = request && typeof request === 'object' ? request : {};
  const nonObjectRequestProvided =
    isProvided(request) && typeof request !== 'object';
  const modeProvided = isProvided(requestObject.mode);
  const startProvided = isProvided(requestObject.rangeStart);
  const countProvided = isProvided(requestObject.rangeCount);
  const requestProvided =
    nonObjectRequestProvided || modeProvided || startProvided || countProvided;
  const requestMode = modeProvided
    ? String(requestObject.mode)
    : nonObjectRequestProvided
      ? String(request)
      : null;
  const requestedStart = safeIntegerOrNull(requestObject.rangeStart);
  const requestedRecordCount = safeIntegerOrNull(requestObject.rangeCount);
  const defaultSelection = requestProvided === false;
  const explicitRangeSelection = requestMode === 'range';
  const rangeValuesPresent = startProvided && countProvided;
  const rangeValuesAreSafeIntegers =
    requestedStart !== null && requestedRecordCount !== null;
  const requestedEndExclusive = rangeValuesAreSafeIntegers
    ? requestedStart + requestedRecordCount
    : null;
  const requestedEndIsSafe =
    requestedEndExclusive !== null &&
    Number.isSafeInteger(requestedEndExclusive) &&
    requestedEndExclusive <= 0x100000000;
  const requestShapeValid = defaultSelection || (
    explicitRangeSelection &&
    rangeValuesPresent &&
    rangeValuesAreSafeIntegers &&
    requestedStart >= 0 &&
    requestedRecordCount > 0 &&
    requestedEndIsSafe
  );
  const requestedRangeInBounds = defaultSelection || (
    requestShapeValid && requestedEndExclusive <= normalizedSceneCount
  );
  const requestedRangeWithinCapacity = defaultSelection || (
    requestShapeValid &&
    requestedRecordCount <= normalizedResourceCapacity
  );
  const appliedStart = defaultSelection
    ? 0
    : requestShapeValid && requestedRangeInBounds && requestedRangeWithinCapacity
      ? requestedStart
      : null;
  const appliedRecordCount = defaultSelection
    ? Math.min(normalizedSceneCount, normalizedResourceCapacity)
    : requestShapeValid && requestedRangeInBounds && requestedRangeWithinCapacity
      ? requestedRecordCount
      : 0;
  const appliedEndExclusive = appliedStart === null
    ? null
    : appliedStart + appliedRecordCount;
  const selectionReady =
    requestShapeValid &&
    requestedRangeInBounds &&
    requestedRangeWithinCapacity &&
    appliedRecordCount > 0;

  let reason = null;
  if (!selectionReady) {
    if (requestProvided && requestMode !== 'range') {
      reason = 'production-resident-selection-mode-range-required';
    } else if (explicitRangeSelection && !rangeValuesPresent) {
      reason = 'production-resident-range-start-count-required';
    } else if (explicitRangeSelection && !rangeValuesAreSafeIntegers) {
      reason = 'production-resident-range-finite-safe-integers-required';
    } else if (explicitRangeSelection && requestedStart < 0) {
      reason = 'production-resident-range-start-negative';
    } else if (explicitRangeSelection && requestedRecordCount <= 0) {
      reason = 'production-resident-range-count-not-positive';
    } else if (explicitRangeSelection && !requestedEndIsSafe) {
      reason = 'production-resident-range-end-not-representable';
    } else if (!requestedRangeInBounds) {
      reason = 'production-resident-range-out-of-scene-bounds';
    } else if (!requestedRangeWithinCapacity) {
      reason = 'production-resident-range-exceeds-resource-capacity';
    } else {
      reason = 'production-scene-has-no-resident-records';
    }
  }

  return {
    contractVersion: PRODUCTION_RESIDENT_SELECTION_CONTRACT_VERSION,
    status: selectionReady ? 'ok' : 'blocked',
    productionResidentSelectionReady: selectionReady,
    requestProvided,
    requestMode,
    requestedStart,
    requestedRecordCount,
    requestedEndExclusive: requestedEndIsSafe
      ? requestedEndExclusive
      : null,
    requestShapeValid,
    requestedRangeInBounds,
    requestedRangeWithinCapacity,
    appliedStart,
    appliedRecordCount,
    appliedEndExclusive,
    sceneRecordCount: normalizedSceneCount,
    resourceCapacityRecords: normalizedResourceCapacity,
    selectionPolicy: defaultSelection
      ? 'scene-owner-single-active-resource-bounded-resident-range'
      : 'scene-owner-explicit-contiguous-original-source-range',
    sourceIndexSpace: 'spl4-original-source-index',
    residentRowSpace: 'active-resident-workset-local-row',
    reason
  };
}

export function buildProductionResidentWorksetContract({
  status = 'ok',
  resourceIdentity = null,
  sceneResourceIdentity = null,
  sceneRecordCount = 0,
  residentStart = 0,
  residentRecordCount = 0,
  resourceCapacityRecords = 0,
  residentSelectionContract = null,
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
  const normalizedResidentSelection = residentSelectionContract ??
    buildProductionResidentSelectionContract({
      request:
        normalizedResidentStart === 0 &&
        normalizedResidentCount === Math.min(
          normalizedSceneCount,
          normalizedResourceCapacity
        )
          ? null
          : {
              mode: 'range',
              rangeStart: normalizedResidentStart,
              rangeCount: normalizedResidentCount
            },
      sceneRecordCount: normalizedSceneCount,
      resourceCapacityRecords: normalizedResourceCapacity
    });
  const residentRangeInBounds =
    normalizedResidentStart + normalizedResidentCount <= normalizedSceneCount;
  const residentSelectionMatches =
    normalizedResidentSelection?.contractVersion ===
      PRODUCTION_RESIDENT_SELECTION_CONTRACT_VERSION &&
    normalizedResidentSelection?.productionResidentSelectionReady === true &&
    normalizedResidentSelection?.appliedStart === normalizedResidentStart &&
    normalizedResidentSelection?.appliedRecordCount === normalizedResidentCount;
  const ready =
    status === 'ok' &&
    stringOrNull(resourceIdentity) !== null &&
    stringOrNull(sceneResourceIdentity) !== null &&
    normalizedResidentCount > 0 &&
    normalizedResidentCount <= normalizedResourceCapacity &&
    residentRangeInBounds &&
    residentSelectionMatches &&
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
    residentEndExclusive: normalizedResidentStart + normalizedResidentCount,
    nonResidentRecordCount: Math.max(
      0,
      normalizedSceneCount - normalizedResidentCount
    ),
    sceneFullyResident: normalizedResidentCount === normalizedSceneCount,
    resourceCapacityRecords: normalizedResourceCapacity,
    tileReferenceCapacityCoupledToRecordSelection: false,
    residentRangeInBounds,
    residentSelectionContract: normalizedResidentSelection,
    residentSelectionMatches,
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
  productionEvaluationInputContract = null,
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
    productionEvaluationInputContract,
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
