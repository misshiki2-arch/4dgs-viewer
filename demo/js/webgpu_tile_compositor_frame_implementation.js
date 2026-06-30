import {
  buildWebGpuTileCompositorFrameImplementationContract
} from './common_4dgs_record_contracts.js';

export const WEBGPU_TILE_COMPOSITOR_FRAME_IMPLEMENTATION_MODE =
  'webgpu-tile-compositor-frame-implementation';

export function buildWebGpuTileCompositorFrameImplementation({
  backendImplementationKind =
    WEBGPU_TILE_COMPOSITOR_FRAME_IMPLEMENTATION_MODE,
  webgpu4DStateSourceContract = null,
  webgpuGaussianAttributeEvaluationContract = null,
  webgpuGaussianFootprintEvaluationContract = null,
  webgpuTileAwareRenderInputContract = null,
  webgpuGpuOwnedTileListLayoutContract = null,
  webgpuTileListCompositorContract = null,
  webgpuPhase3BackendBoundaryContract = null,
  viewerLoopPersistenceContract = null,
  viewerLoopRuntimeFatalError = null
} = {}) {
  const depthOrderingContract =
    webgpuTileListCompositorContract?.tileDepthOrderingContract ?? null;
  const frameImplementationSelected =
    backendImplementationKind === WEBGPU_TILE_COMPOSITOR_FRAME_IMPLEMENTATION_MODE;
  const statePassReady =
    webgpu4DStateSourceContract?.fourDStateSourceReady === true;
  const attributePassReady =
    webgpuGaussianAttributeEvaluationContract
      ?.gaussianAttributeEvaluationReady === true;
  const footprintPassReady =
    webgpuGaussianFootprintEvaluationContract
      ?.gaussianFootprintEvaluationReady === true;
  const tileInputPassReady =
    webgpuTileAwareRenderInputContract?.tileAwareRenderInputReady === true;
  const tileListPassReady =
    webgpuGpuOwnedTileListLayoutContract
      ?.gpuOwnedTileListLayoutReady === true;
  const depthOrderingPassReady =
    depthOrderingContract?.tileDepthOrderingReady === true;
  const tileCompositorPassReady =
    webgpuTileListCompositorContract?.tileCompositorReady === true;
  const compositorOutputPresentedToCurrentTexture =
    webgpuTileListCompositorContract
      ?.tileCompositorOutputPresentedToCurrentTexture === true ||
    (
      viewerLoopPersistenceContract
        ?.currentTextureUsesWebGpuTileCompositorOutput === true &&
      viewerLoopPersistenceContract
        ?.currentTextureReadbackMatchesCompositorOutput === true
    );
  const currentTextureConnectionReady =
    compositorOutputPresentedToCurrentTexture &&
    (
      webgpuTileListCompositorContract
        ?.compositorCurrentTextureRenderPassSubmitted === true ||
      viewerLoopPersistenceContract
        ?.currentTextureUsesWebGpuTileCompositorOutput === true
    );
  const currentTextureReadbackMatchesCompositorOutput =
    compositorOutputPresentedToCurrentTexture &&
    (
      (
        webgpuTileListCompositorContract
          ?.compositorCurrentTextureReadbackCompleted === true &&
        webgpuTileListCompositorContract
          ?.compositorCurrentTextureReadbackNonZero === true
      ) ||
      viewerLoopPersistenceContract
        ?.currentTextureReadbackMatchesCompositorOutput === true
    );
  const currentTextureUsesWebGpuTileCompositorOutput =
    webgpuTileListCompositorContract
      ?.currentTextureUsesWebGpuTileCompositorOutput === true ||
    viewerLoopPersistenceContract
      ?.currentTextureUsesWebGpuTileCompositorOutput === true;
  const presentationStableUntilCapture =
    webgpuTileListCompositorContract
      ?.presentationStableUntilCapture === true;
  const viewerLoopFrameImplementationActive =
    viewerLoopPersistenceContract?.viewerLoopFrameImplementationActive === true;
  const frameImplementationRegisteredWithViewerLoop =
    viewerLoopPersistenceContract?.frameImplementationRegisteredWithViewerLoop === true;
  const compositorOutputPresentedByViewerLoop =
    viewerLoopPersistenceContract?.compositorOutputPresentedByViewerLoop === true;
  const presentationPersistsAfterDelay =
    viewerLoopPersistenceContract?.presentationPersistsAfterDelay === true;
  const presentationPersistsAcrossAnimationFrames =
    viewerLoopPersistenceContract?.presentationPersistsAcrossAnimationFrames === true;
  const animationFramePresentationCount =
    viewerLoopPersistenceContract?.animationFramePresentationCount ?? 0;
  const presentationSampleFrameCount =
    viewerLoopPersistenceContract?.presentationSampleFrameCount ??
    webgpuTileListCompositorContract?.presentationSampleFrameCount ??
    0;
  const presentationNonBlankFrameCount =
    viewerLoopPersistenceContract?.presentationNonBlankFrameCount ??
    webgpuTileListCompositorContract?.presentationNonBlankFrameCount ??
    0;
  const presentationBlankFrameCount =
    viewerLoopPersistenceContract?.presentationBlankFrameCount ??
    webgpuTileListCompositorContract?.presentationBlankFrameCount ??
    0;
  const presentationAllSampledFramesNonBlank =
    viewerLoopPersistenceContract
      ? viewerLoopPersistenceContract.presentationAllSampledFramesNonBlank === true
      : webgpuTileListCompositorContract
        ?.presentationAllSampledFramesNonBlank === true;
  const presentationAlternatingBlankDetected =
    viewerLoopPersistenceContract
      ? viewerLoopPersistenceContract.presentationAlternatingBlankDetected === true
      : webgpuTileListCompositorContract
        ?.presentationAlternatingBlankDetected === true;
  const presentationStableVisualOutput =
    viewerLoopPersistenceContract
      ? viewerLoopPersistenceContract.presentationStableVisualOutput === true
      : webgpuTileListCompositorContract?.presentationStableVisualOutput === true;
  const presentationNonzeroPixelRatioMin =
    viewerLoopPersistenceContract?.presentationNonzeroPixelRatioMin ??
    webgpuTileListCompositorContract?.presentationNonzeroPixelRatioMin ??
    0;
  const presentationNonzeroPixelRatioMax =
    viewerLoopPersistenceContract?.presentationNonzeroPixelRatioMax ??
    webgpuTileListCompositorContract?.presentationNonzeroPixelRatioMax ??
    0;
  const presentationFrameHashChanges =
    viewerLoopPersistenceContract?.presentationFrameHashChanges ??
    webgpuTileListCompositorContract?.presentationFrameHashChanges ??
    0;
  const compositorOutputPresentedEverySampledFrame =
    viewerLoopPersistenceContract
      ? viewerLoopPersistenceContract.compositorOutputPresentedEverySampledFrame === true
      : webgpuTileListCompositorContract
        ?.compositorOutputPresentedEverySampledFrame === true;
  const canvasClearBetweenCompositorFramesDetected =
    viewerLoopPersistenceContract
      ? viewerLoopPersistenceContract.canvasClearBetweenCompositorFramesDetected === true
      : webgpuTileListCompositorContract
        ?.canvasClearBetweenCompositorFramesDetected === true;
  const viewerLoopPresentationCadenceStable =
    viewerLoopPersistenceContract?.viewerLoopPresentationCadenceStable === true;
  const presentationHeartbeatReady =
    viewerLoopPersistenceContract?.presentationHeartbeatReady === true;
  const presentationHeartbeatRunsEveryViewerRaf =
    viewerLoopPersistenceContract?.presentationHeartbeatRunsEveryViewerRaf === true;
  const presentationDecoupledFromCompositorUpdate =
    viewerLoopPersistenceContract
      ?.presentationDecoupledFromCompositorUpdate === true;
  const lastValidCompositorOutputCached =
    viewerLoopPersistenceContract?.lastValidCompositorOutputCached === true;
  const lastValidCompositorOutputPresentedOnCleanFrames =
    viewerLoopPersistenceContract
      ?.lastValidCompositorOutputPresentedOnCleanFrames === true;
  const compositorUpdateFrameCount =
    viewerLoopPersistenceContract?.compositorUpdateFrameCount ??
    webgpuTileListCompositorContract?.compositorUpdateFrameCount ??
    0;
  const presentationHeartbeatFrameCount =
    viewerLoopPersistenceContract?.presentationHeartbeatFrameCount ??
    webgpuTileListCompositorContract?.presentationHeartbeatFrameCount ??
    0;
  const presentationHeartbeatFrameCountMatchesSampledRaf =
    viewerLoopPersistenceContract
      ?.presentationHeartbeatFrameCountMatchesSampledRaf === true;
  const dirtySkippedCompositorUpdateButPresentedCachedOutput =
    viewerLoopPersistenceContract
      ?.dirtySkippedCompositorUpdateButPresentedCachedOutput === true;
  const noBlankFrameBetweenHeartbeatPresentations =
    viewerLoopPersistenceContract
      ?.noBlankFrameBetweenHeartbeatPresentations === true;
  const canvasVisibleOutputStableAcrossRaf =
    viewerLoopPersistenceContract?.canvasVisibleOutputStableAcrossRaf === true;
  const visualFlickerDetected =
    viewerLoopPersistenceContract?.visualFlickerDetected === true;
  const finalPresentSourceTracingReady =
    viewerLoopPersistenceContract?.finalPresentSourceTracingReady === true;
  const sampledRafCount = viewerLoopPersistenceContract?.sampledRafCount ?? 0;
  const tileCompositorFinalPresentFrameCount =
    viewerLoopPersistenceContract?.tileCompositorFinalPresentFrameCount ?? 0;
  const heartbeatFinalPresentFrameCount =
    viewerLoopPersistenceContract?.heartbeatFinalPresentFrameCount ?? 0;
  const normalBackendFinalPresentFrameCount =
    viewerLoopPersistenceContract?.normalBackendFinalPresentFrameCount ?? 0;
  const webgl2FallbackFinalPresentFrameCount =
    viewerLoopPersistenceContract?.webgl2FallbackFinalPresentFrameCount ?? 0;
  const debugClearFinalPresentFrameCount =
    viewerLoopPersistenceContract?.debugClearFinalPresentFrameCount ?? 0;
  const canvasClearFinalPresentFrameCount =
    viewerLoopPersistenceContract?.canvasClearFinalPresentFrameCount ?? 0;
  const noOpFinalPresentFrameCount =
    viewerLoopPersistenceContract?.noOpFinalPresentFrameCount ?? 0;
  const unknownFinalPresentFrameCount =
    viewerLoopPersistenceContract?.unknownFinalPresentFrameCount ?? 0;
  const finalPresentSourceStable =
    viewerLoopPersistenceContract?.finalPresentSourceStable === true;
  const finalPresentSourceAlternates =
    viewerLoopPersistenceContract?.finalPresentSourceAlternates === true;
  const finalPresentSourceSequence =
    Array.isArray(viewerLoopPersistenceContract?.finalPresentSourceSequence)
      ? viewerLoopPersistenceContract.finalPresentSourceSequence
      : [];
  const tileCompositorOwnsFinalPresentation =
    viewerLoopPersistenceContract?.tileCompositorOwnsFinalPresentation === true;
  const summaryCanDetectObservedFlicker =
    viewerLoopPersistenceContract?.summaryCanDetectObservedFlicker === true;
  const rafTraceRingBufferReady =
    viewerLoopPersistenceContract?.rafTraceRingBufferReady === true;
  const rafTraceRecordedFromViewerLoopStart =
    viewerLoopPersistenceContract?.rafTraceRecordedFromViewerLoopStart === true;
  const rafTraceCapturedBeforeCommandStart =
    viewerLoopPersistenceContract?.rafTraceCapturedBeforeCommandStart === true;
  const rafTraceRingBufferFrameCount =
    viewerLoopPersistenceContract?.rafTraceRingBufferFrameCount ?? 0;
  const requiredSteadyStateRafCount =
    viewerLoopPersistenceContract?.requiredSteadyStateRafCount ?? 0;
  const startupTransientObserved =
    viewerLoopPersistenceContract?.startupTransientObserved === true;
  const startupTransientFrameCount =
    viewerLoopPersistenceContract?.startupTransientFrameCount ?? 0;
  const startupTransientFinalPresentSourceSequence =
    Array.isArray(
      viewerLoopPersistenceContract?.startupTransientFinalPresentSourceSequence
    )
      ? viewerLoopPersistenceContract.startupTransientFinalPresentSourceSequence
      : [];
  const firstValidCompositorOutputFrame =
    viewerLoopPersistenceContract?.firstValidCompositorOutputFrame ?? -1;
  const steadyStateSamplingReady =
    viewerLoopPersistenceContract?.steadyStateSamplingReady === true;
  const steadyStateSampledRafCount =
    viewerLoopPersistenceContract?.steadyStateSampledRafCount ?? 0;
  const steadyStateSamplingWindowStartFrame =
    viewerLoopPersistenceContract?.steadyStateSamplingWindowStartFrame ?? -1;
  const steadyStateSamplingWindowEndFrame =
    viewerLoopPersistenceContract?.steadyStateSamplingWindowEndFrame ?? -1;
  const steadyStateFinalPresentSourceSequence =
    Array.isArray(
      viewerLoopPersistenceContract?.steadyStateFinalPresentSourceSequence
    )
      ? viewerLoopPersistenceContract.steadyStateFinalPresentSourceSequence
      : [];
  const steadyStateTileCompositorOwnsFinalPresentation =
    viewerLoopPersistenceContract
      ?.steadyStateTileCompositorOwnsFinalPresentation === true;
  const steadyStateFinalPresentSourceStable =
    viewerLoopPersistenceContract?.steadyStateFinalPresentSourceStable === true;
  const steadyStateFinalPresentSourceAlternates =
    viewerLoopPersistenceContract
      ?.steadyStateFinalPresentSourceAlternates === true;
  const steadyStateBlankFrameCount =
    viewerLoopPersistenceContract?.steadyStateBlankFrameCount ?? 0;
  const steadyStateNoOpFrameCount =
    viewerLoopPersistenceContract?.steadyStateNoOpFrameCount ?? 0;
  const steadyStateClearFrameCount =
    viewerLoopPersistenceContract?.steadyStateClearFrameCount ?? 0;
  const steadyStateUnknownFrameCount =
    viewerLoopPersistenceContract?.steadyStateUnknownFrameCount ?? 0;
  const steadyStateNormalBackendFrameCount =
    viewerLoopPersistenceContract?.steadyStateNormalBackendFrameCount ?? 0;
  const steadyStateWebgl2FallbackFrameCount =
    viewerLoopPersistenceContract?.steadyStateWebgl2FallbackFrameCount ?? 0;
  const steadyStateVisualFlickerDetected =
    viewerLoopPersistenceContract?.steadyStateVisualFlickerDetected === true;
  const summaryCanDetectStartupTransient =
    viewerLoopPersistenceContract?.summaryCanDetectStartupTransient === true;
  const summaryCanDetectSteadyStateFlicker =
    viewerLoopPersistenceContract?.summaryCanDetectSteadyStateFlicker === true;
  const presentationPersistsAfterStartup =
    viewerLoopPersistenceContract?.presentationPersistsAfterStartup === true;
  const presentationPersistsAcrossSteadyStateRaf =
    viewerLoopPersistenceContract
      ?.presentationPersistsAcrossSteadyStateRaf === true;
  const captureWaitedForSteadyStateRaf =
    viewerLoopPersistenceContract?.captureWaitedForSteadyStateRaf === true;
  const captureSteadyStateWaitTimedOut =
    viewerLoopPersistenceContract?.captureSteadyStateWaitTimedOut === true;
  const webgpuDeviceConsistencyReady =
    viewerLoopPersistenceContract
      ? viewerLoopPersistenceContract.webgpuDeviceConsistencyReady === true
      : webgpuTileListCompositorContract?.webgpuDeviceConsistencyReady === true;
  const presentationDeviceMatchesCompositorDevice =
    viewerLoopPersistenceContract
      ? viewerLoopPersistenceContract.presentationDeviceMatchesCompositorDevice === true
      : webgpuTileListCompositorContract
        ?.presentationDeviceMatchesCompositorDevice === true;
  const currentTextureViewFreshPerPresentation =
    viewerLoopPersistenceContract
      ? viewerLoopPersistenceContract.currentTextureViewFreshPerPresentation === true
      : webgpuTileListCompositorContract
        ?.currentTextureViewFreshPerPresentation === true;
  const currentTextureViewReusedAcrossFrames =
    viewerLoopPersistenceContract
      ? viewerLoopPersistenceContract.currentTextureViewReusedAcrossFrames === true
      : webgpuTileListCompositorContract
        ?.currentTextureViewReusedAcrossFrames === true;
  const staleTextureViewReuseDetected =
    viewerLoopPersistenceContract?.staleTextureViewReuseDetected === true ||
    webgpuTileListCompositorContract?.staleTextureViewReuseDetected === true;
  const crossDeviceTextureViewUseDetected =
    viewerLoopPersistenceContract?.crossDeviceTextureViewUseDetected === true ||
    webgpuTileListCompositorContract?.crossDeviceTextureViewUseDetected === true;
  const contextReconfiguredOnDeviceChange =
    viewerLoopPersistenceContract?.contextReconfiguredOnDeviceChange === true ||
    webgpuTileListCompositorContract?.contextReconfiguredOnDeviceChange === true;
  const compositorOutputCacheInvalidatedOnDeviceChange =
    viewerLoopPersistenceContract
      ?.compositorOutputCacheInvalidatedOnDeviceChange === true ||
    webgpuTileListCompositorContract
      ?.compositorOutputCacheInvalidatedOnDeviceChange === true;
  const webgpuValidationErrorDetected =
    viewerLoopPersistenceContract?.webgpuValidationErrorDetected === true ||
    webgpuTileListCompositorContract?.webgpuValidationErrorDetected === true;
  const invalidCommandBufferDetected =
    viewerLoopPersistenceContract?.invalidCommandBufferDetected === true ||
    webgpuTileListCompositorContract?.invalidCommandBufferDetected === true;
  const queueSubmitFailureDetected =
    viewerLoopPersistenceContract?.queueSubmitFailureDetected === true ||
    webgpuTileListCompositorContract?.queueSubmitFailureDetected === true;
  const presentationErrorName =
    viewerLoopPersistenceContract?.presentationErrorName ??
    webgpuTileListCompositorContract?.presentationErrorName ??
    null;
  const presentationErrorMessage =
    viewerLoopPersistenceContract?.presentationErrorMessage ??
    webgpuTileListCompositorContract?.presentationErrorMessage ??
    null;
  const canvasOverwriteAfterCompositorPresentationDetected =
    viewerLoopPersistenceContract
      ?.canvasOverwriteAfterCompositorPresentationDetected === true;
  const normalBackendOverwriteAfterCompositorPresentationDetected =
    viewerLoopPersistenceContract
      ?.normalBackendOverwriteAfterCompositorPresentationDetected === true;
  const fallbackOverwriteAfterCompositorPresentationDetected =
    viewerLoopPersistenceContract
      ?.fallbackOverwriteAfterCompositorPresentationDetected === true;
  const viewerLoopRuntimeFatalErrorDetected = !!viewerLoopRuntimeFatalError;
  const step86BoundaryContractPreserved =
    webgpuPhase3BackendBoundaryContract?.phase3BackendBoundaryReady === true &&
    webgpuPhase3BackendBoundaryContract?.dirtyUpdateContractReady === true &&
    webgpuPhase3BackendBoundaryContract?.toolsDoNotOwnRuntimeBackend === true &&
    webgpuPhase3BackendBoundaryContract?.backendRecordFormatShared === true;
  const fallbackMixingPrevented =
    webgpuPhase3BackendBoundaryContract
      ?.webgpuWebgl2SameFramePresentationMixed === false &&
    webgpuPhase3BackendBoundaryContract?.backendRecordFormatShared === true;
  return buildWebGpuTileCompositorFrameImplementationContract({
    selectedFrameImplementation:
      WEBGPU_TILE_COMPOSITOR_FRAME_IMPLEMENTATION_MODE,
    frameImplementationSelected,
    frameImplementationExecuted: frameImplementationSelected,
    webgpuOwnsFramePassChain: true,
    statePassReady,
    attributePassReady,
    footprintPassReady,
    tileInputPassReady,
    tileListPassReady,
    depthOrderingPassReady,
    tileCompositorPassReady,
    presentationPassReady: currentTextureConnectionReady,
    compositorOutputPresentedToCurrentTexture,
    depthAwareCompositorPresented:
      depthOrderingPassReady && compositorOutputPresentedToCurrentTexture,
    currentTextureConnectionReady,
    currentTextureSource:
      viewerLoopPersistenceContract?.currentTextureSource ??
      webgpuTileListCompositorContract?.currentTextureSource ??
      null,
    currentTextureUsesWebGpuTileCompositorOutput,
    currentTextureReadbackMatchesCompositorOutput,
    presentationFrameCount:
      webgpuTileListCompositorContract?.presentationFrameCount ?? 0,
    compositorPresentationFrameCount:
      webgpuTileListCompositorContract?.compositorPresentationFrameCount ?? 0,
    presentationStableUntilCapture,
    viewerLoopFrameImplementationActive,
    frameImplementationRegisteredWithViewerLoop,
    compositorOutputPresentedByViewerLoop,
    presentationPersistsAfterDelay,
    presentationPersistenceDelayMs:
      viewerLoopPersistenceContract?.presentationPersistenceDelayMs ?? 0,
    presentationPersistsAcrossAnimationFrames,
    animationFramePresentationCount,
    presentationSampleFrameCount,
    presentationNonBlankFrameCount,
    presentationBlankFrameCount,
    presentationAllSampledFramesNonBlank,
    presentationAlternatingBlankDetected,
    presentationStableVisualOutput,
    presentationNonzeroPixelRatioMin,
    presentationNonzeroPixelRatioMax,
    presentationFrameHashChanges,
    compositorOutputPresentedEverySampledFrame,
    canvasClearBetweenCompositorFramesDetected,
    viewerLoopPresentationCadenceStable,
    presentationHeartbeatReady,
    presentationHeartbeatRunsEveryViewerRaf,
    presentationDecoupledFromCompositorUpdate,
    lastValidCompositorOutputCached,
    lastValidCompositorOutputPresentedOnCleanFrames,
    compositorUpdateFrameCount,
    presentationHeartbeatFrameCount,
    presentationHeartbeatFrameCountMatchesSampledRaf,
    dirtySkippedCompositorUpdateButPresentedCachedOutput,
    noBlankFrameBetweenHeartbeatPresentations,
    canvasVisibleOutputStableAcrossRaf,
    visualFlickerDetected,
    finalPresentSourceTracingReady,
    sampledRafCount,
    tileCompositorFinalPresentFrameCount,
    heartbeatFinalPresentFrameCount,
    normalBackendFinalPresentFrameCount,
    webgl2FallbackFinalPresentFrameCount,
    debugClearFinalPresentFrameCount,
    canvasClearFinalPresentFrameCount,
    noOpFinalPresentFrameCount,
    unknownFinalPresentFrameCount,
    finalPresentSourceStable,
    finalPresentSourceAlternates,
    finalPresentSourceSequence,
    tileCompositorOwnsFinalPresentation,
    summaryCanDetectObservedFlicker,
    rafTraceRingBufferReady,
    rafTraceRecordedFromViewerLoopStart,
    rafTraceCapturedBeforeCommandStart,
    rafTraceRingBufferFrameCount,
    requiredSteadyStateRafCount,
    startupTransientObserved,
    startupTransientFrameCount,
    startupTransientFinalPresentSourceSequence,
    firstValidCompositorOutputFrame,
    steadyStateSamplingReady,
    steadyStateSampledRafCount,
    steadyStateSamplingWindowStartFrame,
    steadyStateSamplingWindowEndFrame,
    steadyStateFinalPresentSourceSequence,
    steadyStateTileCompositorOwnsFinalPresentation,
    steadyStateFinalPresentSourceStable,
    steadyStateFinalPresentSourceAlternates,
    steadyStateBlankFrameCount,
    steadyStateNoOpFrameCount,
    steadyStateClearFrameCount,
    steadyStateUnknownFrameCount,
    steadyStateNormalBackendFrameCount,
    steadyStateWebgl2FallbackFrameCount,
    steadyStateVisualFlickerDetected,
    summaryCanDetectStartupTransient,
    summaryCanDetectSteadyStateFlicker,
    presentationPersistsAfterStartup,
    presentationPersistsAcrossSteadyStateRaf,
    captureWaitedForSteadyStateRaf,
    captureSteadyStateWaitTimedOut,
    webgpuDeviceConsistencyReady,
    presentationDeviceMatchesCompositorDevice,
    currentTextureViewFreshPerPresentation,
    currentTextureViewReusedAcrossFrames,
    staleTextureViewReuseDetected,
    crossDeviceTextureViewUseDetected,
    contextReconfiguredOnDeviceChange,
    compositorOutputCacheInvalidatedOnDeviceChange,
    webgpuValidationErrorDetected,
    invalidCommandBufferDetected,
    queueSubmitFailureDetected,
    presentationErrorName,
    presentationErrorMessage,
    canvasOverwriteAfterCompositorPresentationDetected,
    normalBackendOverwriteAfterCompositorPresentationDetected,
    fallbackOverwriteAfterCompositorPresentationDetected,
    viewerLoopRuntimeFatalErrorDetected,
    viewerLoopRuntimeFatalError,
    normalBackendFrameImplementationUsed: false,
    normalBackendPresentationUsed: false,
    normalBackendPresentationBypassed: frameImplementationSelected,
    normalBackendDependencyReduced: frameImplementationSelected,
    step85TileCompositorPathPreserved: tileCompositorPassReady,
    step86BoundaryContractPreserved,
    step87DepthOrderingPreserved: depthOrderingPassReady,
    webgpuWebgl2SameFramePresentationMixed:
      webgpuPhase3BackendBoundaryContract
        ?.webgpuWebgl2SameFramePresentationMixed === true,
    fallbackMixingPrevented,
    fullCudaParity: false,
    finalProductionCompositor: false,
    fullRendererSuccessClaimed: false,
    sourceTileCompositorContractVersion:
      webgpuTileListCompositorContract?.contractVersion ?? null,
    sourceDepthOrderingContractVersion:
      depthOrderingContract?.contractVersion ?? null,
    sourceBoundaryContractVersion:
      webgpuPhase3BackendBoundaryContract?.contractVersion ?? null,
    reason:
      frameImplementationSelected
        ? presentationPersistsAfterDelay
          ? viewerLoopRuntimeFatalErrorDetected
            ? 'viewer-loop-runtime-fatal-error-detected'
            : webgpuValidationErrorDetected
              ? 'webgpu-validation-error-detected'
            : invalidCommandBufferDetected
              ? 'invalid-command-buffer-detected'
            : queueSubmitFailureDetected
              ? 'queue-submit-failure-detected'
            : crossDeviceTextureViewUseDetected
              ? 'cross-device-texture-view-use-detected'
            : presentationHeartbeatReady &&
                presentationDecoupledFromCompositorUpdate &&
                lastValidCompositorOutputPresentedOnCleanFrames &&
                visualFlickerDetected === false &&
                steadyStateSamplingReady &&
                steadyStateTileCompositorOwnsFinalPresentation &&
                steadyStateVisualFlickerDetected === false &&
                captureSteadyStateWaitTimedOut === false &&
                presentationPersistsAfterStartup &&
                presentationPersistsAcrossSteadyStateRaf
              ? null
              : captureSteadyStateWaitTimedOut === true
                ? 'steady-state-sampling-wait-timed-out'
              : steadyStateSamplingReady !== true
                ? 'steady-state-sampling-not-ready'
              : steadyStateTileCompositorOwnsFinalPresentation !== true
                ? 'steady-state-final-presentation-not-owned-by-tile-compositor'
              : steadyStateVisualFlickerDetected === true
                ? 'steady-state-visual-flicker-detected'
              : presentationPersistsAfterStartup !== true
                ? 'presentation-does-not-persist-after-startup'
              : presentationPersistsAcrossSteadyStateRaf !== true
                ? 'presentation-does-not-persist-across-steady-state-raf'
              : 'viewer-loop-tile-compositor-heartbeat-not-stable'
          : 'viewer-loop-persistent-tile-compositor-presentation-not-observed'
        : 'webgpu-tile-compositor-frame-implementation-not-selected'
  });
}
