import {
  compareWebGpuPresentationFrameIdentity
} from './webgpu_presentation_capture_orientation_contract.js';

const RENDER_STATE_MANIFEST_SCHEMA_VERSION = 'phase3-render-state-manifest-v2';

function unknown(reason, source = 'webgpu-viewer-runtime') {
  return { status: 'unknown', source, reason, value: null };
}

function available(value, source = 'webgpu-viewer-runtime') {
  return { status: 'available', source, value };
}

function buildStateIdentity(deterministicState = {}) {
  return {
    cameraMetadataName:
      deterministicState.datasetCameraLabel ??
      deterministicState.imageName ??
      deterministicState.cudaReferenceLabel ??
      null,
    cameraLabel:
      deterministicState.datasetCameraLabel ??
      deterministicState.imageName ??
      deterministicState.cudaReferenceLabel ??
      null,
    frameNumber:
      deterministicState.datasetFrameNumber ??
      deterministicState.frameNumber ??
      null,
    viewId: deterministicState.datasetViewId ?? deterministicState.viewId ?? null,
    datasetTime: deterministicState.datasetTime ?? deterministicState.time ?? null,
    outputWidth:
      deterministicState.canvasWidth ??
      deterministicState.outputWidth ??
      deterministicState.width ??
      null,
    outputHeight:
      deterministicState.canvasHeight ??
      deterministicState.outputHeight ??
      deterministicState.height ??
      null,
    referenceCameraLabel:
      deterministicState.cudaReferenceLabel ??
      deterministicState.datasetCameraLabel ??
      deterministicState.imageName ??
      null
  };
}

function finiteNumberOrNull(value) {
  if (value === null || value === undefined || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function uniqueFiniteNumbers(values, tolerance = 1e-6) {
  const out = [];
  for (const value of values) {
    const number = finiteNumberOrNull(value);
    if (number === null) continue;
    if (!out.some((item) => Math.abs(item - number) <= tolerance)) {
      out.push(number);
    }
  }
  return out;
}

function collectDirectEvidenceTimes(directGaussianEvidence) {
  const records = Array.isArray(directGaussianEvidence?.records)
    ? directGaussianEvidence.records
    : [];
  const requested = [];
  const actual = [];
  for (const record of records) {
    const temporal =
      record?.preCullDirectEvidence?.temporalEvaluation ??
      record?.temporalEvaluation ??
      null;
    requested.push(temporal?.requestedTimestamp);
    actual.push(temporal?.actualEvaluatedTimestamp);
  }
  return {
    requestedTimestamps: uniqueFiniteNumbers(requested),
    actualEvaluatedTimestamps: uniqueFiniteNumbers(actual),
    recordCount: records.length
  };
}

function sourceLooksLikeWebGpuProductionCompositor(source) {
  const text = String(source ?? '');
  return (
    text === 'last-valid-webgpu-tile-compositor-output' ||
    text.includes('last-valid-webgpu-tile-compositor-output') ||
    text.includes('webgpu-tile-compositor') ||
    text.includes('production-tile-compositor')
  );
}

function countRecordsWithTemporalField(directGaussianEvidence, fieldName) {
  const records = Array.isArray(directGaussianEvidence?.records)
    ? directGaussianEvidence.records
    : [];
  return records.filter((record) => {
    const temporal =
      record?.preCullDirectEvidence?.temporalEvaluation ??
      record?.temporalEvaluation ??
      null;
    const value = temporal?.[fieldName];
    return Array.isArray(value)
      ? value.every((component) => Number.isFinite(Number(component)))
      : Number.isFinite(Number(value));
  }).length;
}

function buildRotationInputPackingContract(directGaussianEvidence) {
  const records = Array.isArray(directGaussianEvidence?.records)
    ? directGaussianEvidence.records
    : [];
  const recordCount = records.length;
  const leftAvailableCount = countRecordsWithTemporalField(
    directGaussianEvidence,
    'rotation'
  );
  const rightAvailableCount = countRecordsWithTemporalField(
    directGaussianEvidence,
    'rotationR'
  );
  const ready =
    recordCount > 0 &&
    leftAvailableCount === recordCount &&
    rightAvailableCount === recordCount;
  return {
    schemaVersion: 'phase3-step114-fix10-rotation-input-packing-contract-v1',
    source: 'common-webgpu-render-state-manifest-builder',
    contractReady: ready,
    checkpointFieldSource:
      'CUDA checkpoint 4D rotation fields exported into SPL4-v2 asset',
    assetParserFieldNames: {
      leftQuaternion: 'rotation',
      rightQuaternion: 'rotation_r'
    },
    leftQuaternionComponentOrder: ['x', 'y', 'z', 'w'],
    rightQuaternionComponentOrder: ['x', 'y', 'z', 'w'],
    quaternionMathConvention:
      'CUDA 4D left/right quaternion pair consumed by conditional covariance temporal mean',
    uploadType: 'Float32Array',
    storageBufferName: 'rotationInput',
    bufferRecordLayout: {
      vec4sPerRecord: 2,
      bytesPerVec4: 16,
      recordStrideBytes: 32,
      leftQuaternionOffsetBytes: 0,
      rightQuaternionOffsetBytes: 16,
      totalRecordSizeBytes: 32,
      requiredAlignmentBytes: 16,
      alignmentValid: true
    },
    storagePlacement:
      'rotationInput[record*2+0] stores left quaternion; rotationInput[record*2+1] stores right quaternion',
    wgslReadContract: {
      leftQuaternion: 'rotationInput[row * 2u + 0u].xyzw',
      rightQuaternion: 'rotationInput[row * 2u + 1u].xyzw',
      componentOrder: 'xyzw'
    },
    productionConsumption:
      'webgpu_4d_state_evaluator production WGSL consumes both quaternions for cudaConditionalTemporalMeanOffset',
    diagnosticReadback:
      'directGaussianEvidence.records[].temporalEvaluation.rotation/rotationR',
    existing3dRotationCollision: {
      collisionDetected: false,
      reason:
        'left and right 4D quaternion rows are separate vec4 lanes in rotationInput and do not overwrite packed spatial scale or output payload fields'
    },
    specialCaseBranching: {
      stepSpecific: false,
      cameraSpecific: false,
      srcIndexSpecific: false
    },
    availability: {
      recordCount,
      leftQuaternionAvailableCount: leftAvailableCount,
      rightQuaternionAvailableCount: rightAvailableCount,
      missingCount: Math.max(0, recordCount - Math.min(leftAvailableCount, rightAvailableCount))
    },
    blockedReason: ready ? null : 'rotation-input-left-right-quaternion-evidence-missing'
  };
}

export function buildFixedReferenceCameraContract(deterministicState = {}) {
  const activationMode =
    deterministicState.fixedReferenceCameraActivationMode ??
    deterministicState.referenceCameraMode ??
    null;
  const active =
    deterministicState.fixedReferenceCameraMode === true ||
    activationMode === 'cuda-aligned-fixed-reference-camera' ||
    deterministicState.datasetViewMatrixMode === 'cuda-aligned';
  return {
    schemaVersion: 'phase3-fixed-reference-camera-contract-v1',
    sourceOfTruth: 'normalized-viewer-deterministic-query-runtime-state',
    active,
    fixedReferenceCameraMode: active,
    activationMode:
      activationMode ??
      (active ? 'cuda-aligned-fixed-reference-camera' : null),
    datasetViewMatrixMode: deterministicState.datasetViewMatrixMode ?? null,
    cameraMathChangedByContractNormalization: false
  };
}

function buildInitialProductionPresentationContract({
  deterministicState = {},
  renderResultSummary = {},
  pngStatus = {},
  captureStateContract = null
} = {}) {
  const fixedReferenceCameraContract =
    buildFixedReferenceCameraContract(deterministicState);
  const snapshot =
    captureStateContract?.preCaptureSnapshot ??
    captureStateContract?.initialPresentation ??
    null;
  const capturedObservation = captureStateContract?.initialPresentation ?? null;
  const captureFrame = captureStateContract?.captureFrame ?? null;
  const boundary = captureStateContract?.captureCommandBoundary ?? null;
  const urlOnlyFinalPresentationBoundary =
    captureStateContract?.urlOnlyFinalPresentationBoundary ??
    capturedObservation?.finalCanvasPresentationBoundary ??
    null;
  const captureFinalPresentationBoundary =
    captureStateContract?.captureFinalPresentationBoundary ?? null;
  const snapshotReadOnly =
    snapshot?.readOnlySnapshot?.getterMutatesRuntimeState === false &&
    snapshot?.readOnlySnapshot?.renderScheduledByGetter === false &&
    snapshot?.readOnlySnapshot?.sceneRetryPerformed === false &&
    snapshot?.readOnlySnapshot?.gpuReadbackPerformed === false &&
    snapshot?.readOnlySnapshot?.cloneReturnedToCaller === true &&
    snapshot?.readOnlySnapshot?.callerCanMutateRecorderState === false;
  const snapshotCapturedBeforeMutation =
    boundary?.snapshotWasFirstRuntimeOperation === true &&
    boundary?.snapshotTakenBeforeReadinessWait === true &&
    boundary?.snapshotTakenBeforeSceneRetry === true &&
    boundary?.snapshotTakenBeforeScheduleRender === true &&
    boundary?.snapshotTakenBeforeGpuReadback === true;
  const initialSucceeded =
    snapshot?.viewerCanvasPresented === true &&
    snapshot?.logicalPresentationSucceeded === true &&
    urlOnlyFinalPresentationBoundary?.browserVisibleResult === true;
  const initialLogicalPresentationSucceeded =
    snapshot?.logicalPresentationSucceeded === true;
  const captureSucceeded =
    captureFrame?.productionFrameCompleted === true &&
    captureFrame?.freshGenerationObserved === true &&
    captureFrame?.viewerCanvasPresented === true &&
    captureFrame?.logicalPresentationSucceeded === true &&
    captureFinalPresentationBoundary?.browserVisibleResult === true;
  const captureLogicalPresentationSucceeded =
    captureFrame?.productionFrameCompleted === true &&
    captureFrame?.freshGenerationObserved === true &&
    captureFrame?.logicalPresentationSucceeded === true;
  const initialGeneration = finiteNumberOrNull(
    snapshot?.initialProductionGeneration
  );
  const captureGeneration = finiteNumberOrNull(captureFrame?.productionGeneration);
  const generationSeparated =
    initialGeneration !== null && captureGeneration !== null
      ? initialGeneration !== captureGeneration
      : null;
  const captureArtifactFrameIdentityComparison =
    captureFrame?.productionFrameIdentity &&
    captureFrame?.captureArtifactFrameIdentity
      ? compareWebGpuPresentationFrameIdentity(
          captureFrame.productionFrameIdentity,
          captureFrame.captureArtifactFrameIdentity
        )
      : null;
  const captureDependencyRemaining = initialSucceeded
    ? false
    : urlOnlyFinalPresentationBoundary?.browserVisibleResult === false &&
        captureSucceeded
      ? true
      : null;
  const classification = initialSucceeded
    ? 'url-only-initial-production-presentation-succeeded'
    : urlOnlyFinalPresentationBoundary?.browserVisibleResult === false &&
        captureSucceeded
      ? 'capture-command-or-retry-generated-first-observed-presentation'
      : snapshot?.classification ?? 'initial-presentation-evidence-insufficient';
  const genericPolicy =
    deterministicState.webgpuBackendViewerLoopHook === true &&
    deterministicState.webgpuBackendMode === 'webgpu-exclusive';
  return {
    schemaVersion: 'phase3-initial-production-presentation-contract-v3',
    source: 'common-webgpu-render-state-manifest-builder',
    policy:
      'webgpu-exclusive viewer loop should schedule and present the first production frame without requiring capture commands',
    policySelectedBy: {
      webgpuBackendMode: deterministicState.webgpuBackendMode ?? null,
      webgpuBackendViewerLoopHook: deterministicState.webgpuBackendViewerLoopHook ?? null,
      fixedReferenceCameraMode:
        fixedReferenceCameraContract.fixedReferenceCameraMode
    },
    traceActiveFromPageLoad: snapshot?.activeFromPageLoad ?? null,
    preCaptureSnapshotAvailable: !!snapshot,
    preCaptureSnapshotReadOnly: snapshotReadOnly,
    preCaptureSnapshotCapturedBeforeMutation: snapshotCapturedBeforeMutation,
    preCaptureSnapshotTimestampMs: snapshot?.snapshotTakenAtMs ?? null,
    recorderInstalledAtMs: snapshot?.recorderInstalledAtMs ?? null,
    viewerInitializationComplete:
      snapshot?.viewerInitializationComplete ?? null,
    webgpuDeviceReady: null,
    assetUploadComplete: null,
    fixedCameraApplied:
      fixedReferenceCameraContract.fixedReferenceCameraMode,
    fixedReferenceCameraContract,
    fixedTimeApplied:
      Number.isFinite(Number(deterministicState.datasetTime ?? deterministicState.time)),
    initialProductionFrameScheduled:
      snapshot?.initialScheduleRequestObserved ?? null,
    initialScheduleSource: snapshot?.initialScheduleSource ?? null,
    initialRequestIdentity: snapshot?.initialRequestIdentity ?? null,
    initialProductionFrameCompleted:
      snapshot?.initialProductionFrameCompleted ?? null,
    compositorOutputGenerated: snapshot?.compositorOutputGenerated ?? null,
    viewerCanvasPresented: snapshot?.viewerCanvasPresented ?? null,
    knownNonblank: snapshot?.knownNonblank ?? null,
    urlOnlyLogicalPresentationResult:
      initialLogicalPresentationSucceeded ? true : (
        snapshot?.logicalPresentationSucceeded === false ? false : null
      ),
    urlOnlyPixelBackedPresentationResult:
      urlOnlyFinalPresentationBoundary?.browserVisibleResult ?? null,
    browserVisibleFinalPresentationKnown:
      urlOnlyFinalPresentationBoundary?.browserVisibleResult === true ||
      urlOnlyFinalPresentationBoundary?.browserVisibleResult === false,
    currentTextureRgbPixelEvidenceKnown:
      urlOnlyFinalPresentationBoundary?.finalSourcePixelResult === 'nonblank' ||
      urlOnlyFinalPresentationBoundary?.finalSourcePixelResult === 'black',
    finalPresentSourceTracingReady:
      urlOnlyFinalPresentationBoundary?.finalCanvasEventIdentity != null,
    finalPresentSourceStable:
      urlOnlyFinalPresentationBoundary?.steadyStateConfirmed ?? null,
    finalPresentSourceAlternates:
      urlOnlyFinalPresentationBoundary?.laterOverwriteDetected ?? null,
    finalPresentSourceSequence:
      urlOnlyFinalPresentationBoundary?.eventHistory?.map(
        (event) => event?.presentationSource ?? 'unknown'
      ) ?? [],
    tileCompositorOwnsFinalPresentation:
      urlOnlyFinalPresentationBoundary?.finalPresentationSource == null
        ? null
        : sourceLooksLikeWebGpuProductionCompositor(
            urlOnlyFinalPresentationBoundary.finalPresentationSource
          ),
    steadyStateSamplingReady:
      urlOnlyFinalPresentationBoundary?.steadyStateConfirmed ?? null,
    steadyStateSampledRafCount:
      finiteNumberOrNull(
        urlOnlyFinalPresentationBoundary?.steadyStateObservedEventCount
      ),
    initialProductionGeneration: initialGeneration,
    initialCompositorGeneration:
      finiteNumberOrNull(snapshot?.initialCompositorGeneration),
    initialPresentedGeneration:
      finiteNumberOrNull(snapshot?.initialPresentedGeneration),
    initialFrameIdentity: snapshot?.initialFrameIdentity ?? null,
    initialPresentedFrameIdentity:
      snapshot?.initialPresentedFrameIdentity ?? null,
    initialPresentationTimestampMs:
      snapshot?.presentationTimestampMs ?? null,
    urlLoadAloneGaussianVisible:
      urlOnlyFinalPresentationBoundary?.browserVisibleResult ?? null,
    captureCommandDependencyKnown:
      capturedObservation?.captureCommandDependencyKnown ??
      (initialSucceeded || captureSucceeded ? true : null),
    captureCommandDependencyRemaining: captureDependencyRemaining,
    synchronousCommandStartFence:
      captureStateContract?.synchronousCommandStartFence ?? null,
    commandEraCausalTrace:
      captureStateContract?.commandEraCausalTrace ?? null,
    urlOnlyQuiescenceEvidence:
      urlOnlyFinalPresentationBoundary?.quiescenceEvidence ?? null,
    captureQuiescenceEvidence:
      captureFinalPresentationBoundary?.quiescenceEvidence ??
      captureFinalPresentationBoundary?.quiescenceObservation ?? null,
    urlOnlyCanvasWritePathCoverage:
      urlOnlyFinalPresentationBoundary?.canvasWritePathCoverage ?? null,
    captureCanvasWritePathCoverage:
      captureFinalPresentationBoundary?.canvasWritePathCoverage ?? null,
    initialRequestPresentationIdentityChain:
      captureStateContract?.synchronousCommandStartFence
        ?.initialRequestPresentationIdentityChain ?? null,
    startupRuntimeCorrection: {
      applied: true,
      scope: 'common-scene-ready-scheduler-transition',
      source: 'viewer-file-io-onSceneLoaded',
      forceProductionUpdateAfterSceneReady: true,
      stepNameDependent: false,
      reason:
        'scene-ready request must own a fresh production generation rather than reuse a pre-load or cached frame'
    },
    captureCommandStartedAtMs: boundary?.startedAtMs ?? null,
    captureBaselineGeneration:
      finiteNumberOrNull(boundary?.baselineProductionGeneration),
    captureRequestIdentity: captureFrame?.requestIdentity ?? null,
    captureProductionGeneration: captureGeneration,
    captureCompositorGeneration:
      finiteNumberOrNull(captureFrame?.compositorGeneration),
    capturePresentedGeneration:
      finiteNumberOrNull(captureFrame?.presentedGeneration),
    captureProductionFrameIdentity:
      captureFrame?.productionFrameIdentity ?? null,
    capturePresentedFrameIdentity:
      captureFrame?.presentedFrameIdentity ?? null,
    captureArtifactFrameIdentity:
      captureFrame?.captureArtifactFrameIdentity ?? null,
    captureArtifactFrameIdentityComparison,
    captureArtifactMatchesCaptureProductionFrame:
      captureArtifactFrameIdentityComparison?.matches ?? null,
    captureFreshGenerationObserved:
      captureFrame?.freshGenerationObserved ?? null,
    captureLogicalPresentationResult:
      captureLogicalPresentationSucceeded ? true : (
        captureFrame?.logicalPresentationSucceeded === false ? false : null
      ),
    capturePixelBackedPresentationResult:
      captureFinalPresentationBoundary?.browserVisibleResult ?? null,
    captureBrowserVisibleFinalPresentationKnown:
      captureFinalPresentationBoundary?.browserVisibleResult === true ||
      captureFinalPresentationBoundary?.browserVisibleResult === false,
    initialAndCaptureGenerationSeparated: generationSeparated,
    presentationSource:
      urlOnlyFinalPresentationBoundary?.finalPresentationSource ?? null,
    presentationSourceIsWebGpuProductionCompositor:
      urlOnlyFinalPresentationBoundary?.finalPresentationSource == null
        ? null
        : sourceLooksLikeWebGpuProductionCompositor(
            urlOnlyFinalPresentationBoundary.finalPresentationSource
          ),
    classification,
    initialObservationClassification:
      capturedObservation?.classification ?? snapshot?.classification ?? null,
    evidenceSeparation: {
      productionOutputNonblank: snapshot?.knownNonblank ?? null,
      browserVisiblePresentationNonblank:
        urlOnlyFinalPresentationBoundary?.browserVisibleResult ?? null,
      encodedPngBlobNonblank:
        pngStatus?.encodedPngPixelEvidence?.pixelClassification === 'nonblank'
          ? true
          : pngStatus?.encodedPngPixelEvidence?.pixelClassification === 'black'
            ? false
            : null,
      savedPngFileNonblank: null,
      savedPngFileEvidenceSource:
        'tool-side-exact-file-pixel-decode-required'
    },
    captureBlobIdentity: pngStatus?.captureBlobIdentity ?? null,
    encodedPngPixelEvidence:
      pngStatus?.encodedPngPixelEvidence ?? null,
    finalCanvasPresentationEvidence: {
      schemaVersion: 'phase3-final-canvas-presentation-evidence-v1',
      source: 'common-webgpu-render-state-manifest-builder',
      urlOnlyBoundary: urlOnlyFinalPresentationBoundary,
      captureBoundary: captureFinalPresentationBoundary,
      boundariesSeparated:
        urlOnlyFinalPresentationBoundary?.boundaryIdentity != null &&
        captureFinalPresentationBoundary?.boundaryIdentity != null &&
        urlOnlyFinalPresentationBoundary.boundaryIdentity !==
          captureFinalPresentationBoundary.boundaryIdentity,
      independentPredicates: {
        productionOutput: true,
        finalBrowserPresentation: true,
        encodedPngBlob: true,
        savedPngFile: true
      },
      productionRuntimeBehaviorChanged: false
    },
    alphaNormalizationEvidence:
      pngStatus?.alphaNormalizationEvidence ?? null,
    runtimeError: snapshot?.runtimeError ?? null,
    runtimeBehaviorChanged:
      captureStateContract?.productionRuntimeBehaviorChanged === true
        ? true
        : captureStateContract?.productionRuntimeBehaviorChanged === false
          ? false
          : null,
    blockedReason:
      initialSucceeded
        ? null
        : urlOnlyFinalPresentationBoundary?.unknownOrBlockedReason ??
          snapshot?.blockedReason ??
          (genericPolicy
            ? 'initial-production-presentation-not-confirmed-before-capture'
            : 'initial-production-presentation-policy-not-active')
  };
}

function buildFixedTimeCaptureStateContract({
  deterministicState = {},
  stateIdentity = {},
  pngStatus = {},
  directGaussianEvidence = null,
  inputContract = null,
  generatedAtUtc = null
} = {}) {
  const directEvidenceTimes = collectDirectEvidenceTimes(directGaussianEvidence);
  const requestedTime =
    finiteNumberOrNull(inputContract?.requestedTimeFromUrl) ??
    finiteNumberOrNull(stateIdentity.datasetTime) ??
    finiteNumberOrNull(deterministicState.datasetTime) ??
    finiteNumberOrNull(deterministicState.time);
  const manifestTime =
    finiteNumberOrNull(stateIdentity.datasetTime) ??
    finiteNumberOrNull(deterministicState.datasetTime) ??
    finiteNumberOrNull(deterministicState.time);
  const presentedTime = finiteNumberOrNull(
    pngStatus.presentedFrameIdentity?.datasetTime ??
    pngStatus.presentedStateIdentity?.datasetTime
  );
  const capturedTime = finiteNumberOrNull(
    pngStatus.capturedFrameIdentity?.datasetTime ??
    pngStatus.capturedStateIdentity?.datasetTime ??
    pngStatus.requestedStateIdentity?.datasetTime
  );
  const directActualTimes = directEvidenceTimes.actualEvaluatedTimestamps;
  const directRequestedTimes = directEvidenceTimes.requestedTimestamps;
  const expected = requestedTime;
  const mismatches = [];
  const checkTime = (name, value) => {
    if (expected === null || value === null) {
      mismatches.push({ field: name, expected, actual: value, reason: 'missing-fixed-time-value' });
      return;
    }
    if (Math.abs(value - expected) > 1e-5) {
      mismatches.push({ field: name, expected, actual: value, reason: 'time-value-differs-from-requested-fixed-time' });
    }
  };
  checkTime('manifestTime', manifestTime);
  checkTime('presentedFrameTime', presentedTime);
  checkTime('capturedFrameTime', capturedTime);
  for (const value of directRequestedTimes) {
    checkTime('directEvidenceRequestedTimestamp', value);
  }
  for (const value of directActualTimes) {
    checkTime('directEvidenceActualEvaluatedTimestamp', value);
  }

  const probeMutationDetected =
    inputContract?.schedulerProbe?.executedInCaptureCommand === true ||
    inputContract?.cameraDirtyProbe?.executedInCaptureCommand === true;
  const artifactsShareFixedTime =
    expected !== null &&
    mismatches.length === 0 &&
    directActualTimes.length > 0 &&
    directRequestedTimes.length > 0;
  return {
    schemaVersion: 'phase3-step114-fix8-fixed-time-capture-state-contract-v1',
    source: 'webgpu-render-state-manifest-builder',
    generatedAtUtc,
    isolationPolicy:
      inputContract?.policy ?? 'unknown-step114-capture-isolation-policy',
    selectedIsolationMode: inputContract?.selectedIsolationMode ?? null,
    requestedTime,
    requestedDatasetTimeFromUrl:
      finiteNumberOrNull(inputContract?.requestedDatasetTimeFromUrl),
    probeBeforeTime:
      finiteNumberOrNull(inputContract?.schedulerProbe?.beforeTime) ?? null,
    probeAfterTime:
      finiteNumberOrNull(inputContract?.schedulerProbe?.afterTime) ?? null,
    schedulerProbeExecutedInCaptureCommand:
      inputContract?.schedulerProbe?.executedInCaptureCommand === true,
    cameraDirtyProbeExecutedInCaptureCommand:
      inputContract?.cameraDirtyProbe?.executedInCaptureCommand === true,
    probeStateMutationDetected: probeMutationDetected,
    stateRestorationPerformed:
      inputContract?.stateRestoration?.performed === true,
    stateRestorationCompletedFrame:
      inputContract?.stateRestoration?.completedFrame ?? null,
    manifestTime,
    presentedFrameTime: presentedTime,
    capturedFrameTime: capturedTime,
    directEvidenceRequestedTimestamps: directRequestedTimes,
    directEvidenceActualEvaluatedTimestamps: directActualTimes,
    generationIdentity: {
      productionOutputGeneration: pngStatus.productionOutputGeneration ?? null,
      presentedOutputGeneration: pngStatus.presentedOutputGeneration ?? null,
      capturedOutputGeneration: pngStatus.capturedOutputGeneration ?? null
    },
    frameIdentity: {
      requestedStateIdentity: pngStatus.requestedStateIdentity ?? stateIdentity,
      presentedFrameIdentity: pngStatus.presentedFrameIdentity ?? null,
      capturedFrameIdentity: pngStatus.capturedFrameIdentity ?? null
    },
    captureOrder: Array.isArray(inputContract?.captureOrder)
      ? inputContract.captureOrder
      : [],
    artifactsShareFixedTime,
    artifactsShareFixedFrame:
      pngStatus.captureMatchesPresentedFrame === true &&
      pngStatus.captureMatchesRequestedState === true,
    mismatchedFields: mismatches,
    classification:
      artifactsShareFixedTime && !probeMutationDetected
        ? 'fixed-time-capture-isolated'
        : (probeMutationDetected
            ? 'state-changing-probe-ran-during-step114-capture'
            : 'fixed-time-artifact-state-mismatch')
  };
}

function buildArtifactIdentity(
  path,
  {
    status = 'unknown',
    reason = 'browser runtime cannot hash local filesystem artifact',
    sha256 = null,
    sizeBytes = null
  } = {}
) {
  return {
    absolutePath: path ?? null,
    exists: status === 'available' ? true : null,
    sizeBytes,
    sha256,
    status,
    reason: status === 'available' ? null : reason
  };
}

function pickFixedReferenceScreenSpaceCamera(deterministicState = {}) {
  return deterministicState.fixedReferenceScreenSpaceCamera ??
    deterministicState.cudaAlignedScreenSpaceCamera ??
    null;
}

function buildProjectionContractEvidence(screenSpaceCamera, viewport) {
  if (!screenSpaceCamera || !screenSpaceCamera.intrinsics) {
    return unknown('fixed-reference WebGPU projection contract not emitted by viewer deterministic state');
  }
  return available(
    {
      matrixRepresentation: 'intrinsics-plus-viewport-camera-constants',
      projectionContract: screenSpaceCamera.projectionContract ?? null,
      intrinsics: screenSpaceCamera.intrinsics,
      viewport,
      covarianceTanFovX: screenSpaceCamera.covarianceTanFovX ?? null,
      covarianceTanFovY: screenSpaceCamera.covarianceTanFovY ?? null,
      covarianceFocalContract: screenSpaceCamera.covarianceFocalContract ?? null,
      screenYSign: screenSpaceCamera.screenYSign ?? null,
      depthSign: screenSpaceCamera.depthSign ?? null,
      pixelXSign: screenSpaceCamera.pixelXSign ?? null
    },
    'webgpu-fixed-reference-camera-constants-contract'
  );
}

function pickCameraConstantsSource(deterministicState = {}) {
  if (buildFixedReferenceCameraContract(deterministicState).active) {
    return 'cuda-reference-camera-evidence-via-fixed-reference-camera-mode';
  }
  return 'interactive-viewer-camera';
}

function normalizeSrcIndexList(values) {
  const out = [];
  const seen = new Set();
  if (!Array.isArray(values)) return out;
  for (const item of values) {
    const value = Number.isFinite(Number(item))
      ? Number(item)
      : (Number.isFinite(Number(item?.srcIndex)) ? Number(item.srcIndex) : NaN);
    if (!Number.isFinite(value)) continue;
    const index = value | 0;
    if (seen.has(index)) continue;
    seen.add(index);
    out.push(index);
  }
  return out;
}

function buildCanonicalComparisonIndexSet(indices, directGaussianEvidence) {
  const normalized = normalizeSrcIndexList(indices);
  const evidenceIndices = normalizeSrcIndexList(directGaussianEvidence?.records);
  const selected = normalized.length > 0 ? normalized : evidenceIndices;
  return {
    source: normalized.length > 0
      ? 'capture-command-cuda-direct-rasterizer-selected-src-indices'
      : (evidenceIndices.length > 0
          ? 'webgpu-direct-gaussian-evidence-records'
          : 'missing-canonical-index-set'),
    selectedSrcIndices: selected,
    selectedCount: selected.length,
    missing: selected.length <= 0,
    missingReason: selected.length <= 0
      ? 'cuda-direct-rasterizer-selected-src-indices-not-provided-to-webgpu-capture'
      : null
  };
}

function buildSrcIndexSemantics(rawSummary, directGaussianEvidence) {
  const mappingArtifact =
    rawSummary.indexMappingArtifact ??
    rawSummary.checkpointToAssetMappingArtifact ??
    rawSummary.mappingArtifact ??
    rawSummary.assetIndexMapping ??
    null;
  const mappingAvailable = !!(
    mappingArtifact &&
    (mappingArtifact.sha256 || mappingArtifact.hash || mappingArtifact.absolutePath || mappingArtifact.path)
  );
  const originalIndexPreserved =
    directGaussianEvidence?.indexSemantics?.runtimeOriginalIndexPreserved === true ||
    directGaussianEvidence?.indexSemantics?.preCullRuntimeOriginalIndexPreserved === true ||
    directGaussianEvidence?.recordSummary?.allRecordsUseRequestedOriginalIndex === true;
  const preCullEvidenceAvailable =
    directGaussianEvidence?.preCullEvidenceSummary?.available === true;
  const sourcePositionUniquenessEvidence =
    directGaussianEvidence?.indexSemantics?.sourcePositionUniquenessEvidence ??
    directGaussianEvidence?.sourcePositionUniquenessEvidence ??
    null;
  const sourcePositionUniqueMappingReady =
    sourcePositionUniquenessEvidence?.ready === true;
  const checkpointToAssetLineageReady =
    mappingAvailable || sourcePositionUniqueMappingReady;
  return {
    cudaIndexRole: 'checkpoint-gaussian-index',
    conversionInputIndexRole: rawSummary.conversionInputIndexRole ?? 'unknown',
    webgpuAssetIndexRole: rawSummary.webgpuAssetIndexRole ?? 'SPL4-v2 asset record index',
    runtimeBufferIndexRole: 'original srcIndex preserved in pre-cull visible-record dispatch and visible/packed/tile payload when evidence record is found',
    sortedOrCompactedIndexRole: 'tile/compositor order may reorder records but preserves originalSplatIndex/sourceIndices',
    originalIndexPreserved,
    preCullEvidenceAvailable,
    assetRecordIndexPreservedByParser:
      rawSummary.assetRecordIndexPreservedByParser === true,
    assetRecordIndexPreservationSource:
      rawSummary.assetRecordIndexPreservationSource ?? null,
    indexMappingAvailable: mappingAvailable,
    sourcePositionUniqueMappingReady,
    sourcePositionUniquenessEvidence,
    conversionReorderKnown: rawSummary.conversionReorderKnown ?? null,
    conversionReorderApplied: rawSummary.conversionReorderApplied ?? null,
    mappingArtifact,
    mappingEvidence: {
      checkpointToAssetMappingArtifact: mappingArtifact,
      assetToRuntimeIndexPreserved:
        rawSummary.assetRecordIndexPreservedByParser === true,
      assetToRuntimeEvidenceSource:
        rawSummary.assetRecordIndexPreservationSource ?? null,
      runtimeToProductionOriginalIndexPreserved: originalIndexPreserved,
      checkpointToAssetEvidenceAvailable: mappingAvailable,
      sourcePositionUniqueMappingEvidenceAvailable: sourcePositionUniqueMappingReady,
      sourcePositionUniquenessEvidence,
      gaussianCountOnlyUsedForMapping: false
    },
    mappingDecision: checkpointToAssetLineageReady
      ? 'ready'
      : (originalIndexPreserved
          ? (preCullEvidenceAvailable
              ? 'pre-cull-runtime-original-index-preserved-but-checkpoint-to-asset-mapping-artifact-missing'
              : 'runtime-original-index-preserved-but-checkpoint-to-asset-mapping-artifact-missing')
          : 'blocked-index-mapping-unknown'),
    status: mappingAvailable
      ? 'verified-direct'
      : (sourcePositionUniqueMappingReady
          ? 'verified-derived'
          : (originalIndexPreserved ? 'inferred' : 'unknown')),
    unknownReason: checkpointToAssetLineageReady
      ? null
      : 'browser-loaded asset exposes asset-to-runtime preservation but not checkpoint-to-asset mapping artifact'
  };
}

export function buildWebGpuRenderStateManifest({
  deterministicState = {},
  canvasSizeSummary = {},
  renderResultSummary = {},
  pngCaptureStatus = null,
  rawSummary = {},
  canonicalComparisonSrcIndices = [],
  directGaussianEvidence = null,
  captureStateContract = null,
  phaseStep = 'phase3-step114',
  comparisonMode = 'phase3-step114-cuda-reference-provenance-render-state-audit',
  generatedAtUtc = new Date().toISOString()
} = {}) {
  const stateIdentity = buildStateIdentity(deterministicState);
  const cameraConstantsSource = pickCameraConstantsSource(deterministicState);
  const fixedReferenceCameraActive =
    cameraConstantsSource === 'cuda-reference-camera-evidence-via-fixed-reference-camera-mode';
  const fixedReferenceCameraContract =
    buildFixedReferenceCameraContract(deterministicState);
  const screenSpaceCamera = pickFixedReferenceScreenSpaceCamera(deterministicState);
  const pngStatus = pngCaptureStatus && typeof pngCaptureStatus === 'object'
    ? pngCaptureStatus
    : {};
  const capturedIdentity =
    pngStatus.capturedStateIdentity ??
    pngStatus.captureStateIdentity ??
    pngStatus.requestedStateIdentity ??
    null;
  const presentedIdentity =
    pngStatus.presentedStateIdentity ??
    pngStatus.presentedFrameIdentity ??
    null;
  const assetPath = rawSummary.assetPath ?? rawSummary.sourcePath ?? null;
  const assetSha256 = rawSummary.assetSha256 ?? rawSummary.sha256 ?? null;
  const assetSizeBytes = rawSummary.assetSizeBytes ?? rawSummary.sizeBytes ?? null;
  const assetIdentity = buildArtifactIdentity(assetPath, {
    status: assetSha256 ? 'available' : 'unknown',
    sha256: assetSha256,
    sizeBytes: Number.isFinite(Number(assetSizeBytes)) ? Number(assetSizeBytes) : null,
    reason: assetPath
      ? 'browser runtime has asset path but asset SHA-256 was not emitted'
      : 'loaded Gaussian asset path and hash are not emitted by viewer runtime'
  });
  const viewport = {
    width:
      canvasSizeSummary.width ??
      canvasSizeSummary.canvasWidth ??
      stateIdentity.outputWidth,
    height:
      canvasSizeSummary.height ??
      canvasSizeSummary.canvasHeight ??
      stateIdentity.outputHeight
  };
  const canonicalComparisonIndexSet = buildCanonicalComparisonIndexSet(
    canonicalComparisonSrcIndices,
    directGaussianEvidence
  );
  const srcIndexSemantics = buildSrcIndexSemantics(rawSummary, directGaussianEvidence);
  const fixedTimeCaptureState = buildFixedTimeCaptureStateContract({
    deterministicState,
    stateIdentity,
    pngStatus,
    directGaussianEvidence,
    inputContract: captureStateContract,
    generatedAtUtc
  });
  const presentationSource =
    pngStatus.source ??
    pngStatus.captureSource ??
    'last-valid-webgpu-tile-compositor-output';
  const rotationInputPackingContract =
    buildRotationInputPackingContract(directGaussianEvidence);
  const initialProductionPresentation =
    buildInitialProductionPresentationContract({
      deterministicState,
      renderResultSummary,
      pngStatus,
      captureStateContract
    });
  return {
    schemaVersion: RENDER_STATE_MANIFEST_SCHEMA_VERSION,
    manifestKind: 'webgpu-production-render-state',
    phaseStep,
    comparisonMode,
    generatedAtUtc,
    execution: {
      browserUserAgent:
        typeof navigator !== 'undefined' ? navigator.userAgent : null,
      url: typeof window !== 'undefined' ? window.location.href : null,
      deterministicQueryString:
        deterministicState.deterministicQueryString ??
        deterministicState.deterministicRawQueryString ??
        null
    },
    lineage: {
      gaussianAsset: {
        path: assetPath,
        identity: assetIdentity,
        gaussianCount: rawSummary.rawCount ?? rawSummary.count ?? null,
        assetSourceKind: rawSummary.assetSourceKind ?? null,
        assetFormat: rawSummary.assetFormat ?? null,
        assetRecordIndexPreservedByParser:
          rawSummary.assetRecordIndexPreservedByParser ?? null,
        assetRecordIndexPreservationSource:
          rawSummary.assetRecordIndexPreservationSource ?? null,
        lineageStatus:
          assetSha256
            ? 'browser-loaded-asset-hash-recorded'
            : assetPath
              ? 'partial-path-no-browser-hash'
            : 'unknown',
        unknownReason:
          assetSha256
            ? null
            : assetPath
              ? 'browser runtime did not hash Gaussian asset'
            : 'loaded Gaussian asset path and hash are not emitted by viewer runtime'
      }
    },
    canonicalComparisonIndexSet,
    srcIndexSemantics,
    directGaussianEvidence: directGaussianEvidence ?? {
      available: false,
      actualEvidenceSource: 'missing-webgpu-direct-gaussian-evidence',
      recordCount: 0,
      records: [],
      requestedSrcIndices: canonicalComparisonIndexSet.selectedSrcIndices,
      blockedReason: canonicalComparisonIndexSet.missing
        ? canonicalComparisonIndexSet.missingReason
        : 'webgpu-direct-gaussian-evidence-not-built'
    },
    runtimeResponsibilitySeparation: {
      gpuCandidateRuntimeRole:
        deterministicState.gpuCandidateRuntime === 'limited-draw'
          ? 'auxiliary-validation-candidate-generation'
          : 'not-limited-draw-or-not-reported',
      gpuCandidateRuntime: deterministicState.gpuCandidateRuntime ?? null,
      presentationSource:
        presentationSource,
      presentationPath:
        'webgpu-exclusive-production-tile-compositor',
      presentationDerivedFromWebGpuProductionCompositor:
        sourceLooksLikeWebGpuProductionCompositor(presentationSource),
      directGaussianEvidenceSource:
        directGaussianEvidence?.actualEvidenceSource ??
        directGaussianEvidence?.preCullEvidenceSummary?.source ??
        'missing-webgpu-direct-gaussian-evidence',
      directGaussianEvidenceFromProductionEvaluator:
        directGaussianEvidence?.preCullEvidenceSummary?.available === true ||
        String(directGaussianEvidence?.actualEvidenceSource ?? '').includes('webgpu-production'),
      limitedDrawUsedAsDirectActual: false,
      cpuRecalculationUsedAsDirectActual: false,
      fallbackUsedAsDirectActual: false,
      webgl2MixedWithWebGpuPresentation: false
    },
    renderState: {
      ...stateIdentity,
      fixedTimeCaptureState,
      requestedStateIdentity: pngStatus.requestedStateIdentity ?? stateIdentity,
      presentedStateIdentity: presentedIdentity,
      capturedStateIdentity: capturedIdentity,
      fixedReferenceCameraMode: fixedReferenceCameraActive,
      fixedReferenceCameraActivationMode:
        deterministicState.fixedReferenceCameraActivationMode ??
        deterministicState.referenceCameraMode ??
        (fixedReferenceCameraActive ? 'cuda-aligned-fixed-reference-camera' : null),
      webgpuCameraConstantsSource: cameraConstantsSource,
      interactiveCameraExcludedFromReferenceComparison: fixedReferenceCameraActive,
      fixedReferenceCameraContract,
      backgroundPolicy: {
        bgGray: deterministicState.bgGray ?? null,
        rgb: Number.isFinite(Number(deterministicState.bgGray))
          ? [Number(deterministicState.bgGray), Number(deterministicState.bgGray), Number(deterministicState.bgGray)]
          : null
      },
      viewport,
      sourceMode: deterministicState.gpuCandidateSourceMode ?? 'screenCoarse'
    },
    rotationInputPackingContract,
    initialProductionPresentation,
    camera: {
      source: fixedReferenceCameraActive
        ? 'CUDA-aligned fixed reference camera query state'
        : 'interactive viewer camera state',
      intrinsics: deterministicState.intrinsics ?? {
        fx: deterministicState.datasetFx ?? null,
        fy: deterministicState.datasetFy ?? null,
        cx: deterministicState.datasetCx ?? null,
        cy: deterministicState.datasetCy ?? null
      },
      cameraCenter: deterministicState.cameraPosition ?? null,
      worldViewTransform:
        screenSpaceCamera?.cudaAlignedViewMatrix ??
        screenSpaceCamera?.worldViewMatrix ??
        unknown('world-view matrix not emitted by viewer deterministic state'),
      fullProjTransform:
        screenSpaceCamera?.projectionMatrix ??
        buildProjectionContractEvidence(screenSpaceCamera, viewport),
      transformMatrix: deterministicState.rawTransformMatrix ?? null,
      matrixConvention: {
        layout: 'viewer-query-state-and-WebGPU-camera-constant-contract',
        multiplicationOrder: 'WebGPU backend consumes routed camera constants',
        status: fixedReferenceCameraActive ? 'fixed-reference-active' : 'interactive-camera-active'
      }
    },
    imageSpaceConvention: {
      pixelOrigin: available('top-left-for-saved-PNG-canonical-presentation-orientation'),
      xDirection: available('increasing-column-index-right'),
      yDirection: available('increasing-row-index-down-after-capture-row-conversion'),
      ndcToPixel:
        screenSpaceCamera?.ndcToPixel ??
        available('((ndc + 1) * size - 1) * 0.5', 'webgpu-fixed-reference-camera-constants-contract'),
      halfPixelConvention:
        screenSpaceCamera?.halfPixelConvention ??
        available('cuda-ndc2Pix-minus-one-half-pixel-center-convention', 'webgpu-fixed-reference-camera-constants-contract'),
      pngRowOrder: available('top-to-bottom-row-major'),
      transposeOrFlipApplied: available(
        pngStatus.verticalFlipApplied ??
        pngStatus.orientationContract?.captureVerticalFlipApplied ??
        null,
        'png-capture-status'
      ),
      captureSource:
        pngStatus.source ??
        pngStatus.captureSource ??
        pngStatus.requestedCaptureSource ??
        null
    },
    presentationAndCapture: {
      pngCaptureStatus: pngStatus,
      renderResultSummary,
      canvasSizeSummary,
      runtimeOutputNonblank:
        pngStatus.captureOutputNonblank ??
        renderResultSummary?.executionSummary?.outputTextureNonblank ??
        null,
      presentationOutputNonblank: pngStatus.presentationOutputNonblank ?? null,
      savedPngNonblank:
        null,
      savedPngNonblankEvidenceSource:
        'tool-side-exact-saved-file-pixel-decode-required',
      captureBlobIdentity: pngStatus.captureBlobIdentity ?? null,
      encodedPngBlobPixelEvidence:
        pngStatus.encodedPngPixelEvidence ?? null,
      finalCanvasPresentationEvidence:
        initialProductionPresentation.finalCanvasPresentationEvidence ?? null,
      alphaNormalizationEvidence:
        pngStatus.alphaNormalizationEvidence ?? null,
      encodedPngBlobNonblank:
        pngStatus.encodedPngPixelEvidence?.pixelClassification === 'nonblank'
          ? true
          : pngStatus.encodedPngPixelEvidence?.pixelClassification === 'black'
            ? false
            : null,
      savedPngMatchesRuntimeOutput: pngStatus.savedPngMatchesRuntimeOutput ?? null,
      savedPngPresentationOrientationMatchesPresentedOutput:
        pngStatus.savedPngMatchesPresentedOutput ??
        pngStatus.savedPngMatchesRuntimeOutput ??
        null,
      savedPngMatchesPresentedOutput: null,
      savedPngMatchesPresentedOutputEvidenceSource:
        'tool-side-blob-saved-file-hash-and-pixel-verification-required'
    },
    artifacts: {
      savedPng: {
        fileName: pngStatus.fileName ?? null,
        source: pngStatus.source ?? pngStatus.captureSource ?? null,
        captureBlobSha256:
          pngStatus.captureBlobIdentity?.sha256 ?? null,
        sha256: null,
        savedFileSha256: null,
        blobSavedFileIdentityMatch: null,
        status:
          pngStatus.captureBlobIdentity?.sha256
            ? 'capture-blob-available-saved-file-verification-required'
            : 'unknown'
      }
    },
    comparisonReadiness: {
      visualComparisonReady: false,
      blockedReasons: [
        ...(
          fixedReferenceCameraActive
            ? []
            : ['webgpu-fixed-reference-camera-not-active']
        ),
        ...(
          assetSha256
            ? []
            : assetPath
              ? ['webgpu-gaussian-asset-hash-missing']
            : ['webgpu-gaussian-asset-lineage-missing']
        ),
        ...(
          cameraConstantsSource === 'cuda-reference-camera-evidence-via-fixed-reference-camera-mode'
            ? []
            : ['webgpu-camera-constants-not-cuda-reference-derived']
        )
      ]
    }
  };
}
