import {
  buildNativeWebGpuProductionFrameDataPathContract
} from './common_4dgs_production_frame_data_contracts.js';
import {
  buildWebGpuPhase3BackendBoundaryContract
} from './common_4dgs_record_contracts.js';
import {
  buildWebGpuProjectionContract
} from './common_4dgs_projection_contracts.js';
import {
  buildWebGpu4DStatePositionsForCandidates
} from './webgpu_4d_state_evaluator.js';
import {
  buildWebGpuGpuOwnedTileListLayout
} from './webgpu_gpu_owned_tile_list_layout.js';
import {
  buildWebGpuTileListCompositor
} from './webgpu_tile_list_compositor.js';
import {
  buildWebGpuTileCompositorFrameImplementation,
  WEBGPU_TILE_COMPOSITOR_FRAME_IMPLEMENTATION_MODE
} from './webgpu_tile_compositor_frame_implementation.js';
import {
  selectActiveProductionResidentWorkset
} from './webgpu_production_workset_owner.js';
import {
  buildNativeWebGpuProductionTileInput
} from './webgpu_production_tile_input.js';

export const NATIVE_WEBGPU_PRODUCTION_FRAME_DATA_PATH_MODE =
  'native-webgpu-production-frame-data-path';

const deviceCache = new WeakMap();

async function acquireProductionDevice(viewerCanvasState) {
  const canvas = viewerCanvasState?.canvas ?? null;
  const cached = canvas ? deviceCache.get(canvas) : null;
  if (cached && cached.lost !== true) return cached;
  if (typeof navigator === 'undefined' || !navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  const resource = { adapter, device, lost: false };
  if (canvas) {
    deviceCache.set(canvas, resource);
    if (device.lost && typeof device.lost.then === 'function') {
      device.lost.then(() => {
        resource.lost = true;
        if (deviceCache.get(canvas) === resource) deviceCache.delete(canvas);
      });
    }
  }
  return resource;
}

function destroyBuffers(buffers) {
  for (const buffer of buffers) {
    if (buffer && typeof buffer.destroy === 'function') buffer.destroy();
  }
}

function unavailableResult(reason, worksetContract = null) {
  return {
    status: 'blocked',
    reason,
    computeMode: NATIVE_WEBGPU_PRODUCTION_FRAME_DATA_PATH_MODE,
    productionResidentWorksetContract: worksetContract,
    webgpuProductionFrameDataPathContract:
      buildNativeWebGpuProductionFrameDataPathContract({
        status: 'blocked',
        worksetContract,
        reason
      })
  };
}

export async function runNativeWebGpuProductionFrameDataPath({
  raw,
  camera,
  screenSpaceCamera = null,
  canvasWidth = 0,
  canvasHeight = 0,
  buildConfig = {},
  viewerCanvasState = null,
  productionResidentSelectionRequest = null,
  metadata = null
} = {}) {
  if (!raw || !camera || canvasWidth <= 0 || canvasHeight <= 0) {
    return unavailableResult('native-production-frame-input-unavailable');
  }
  const deviceResource = await acquireProductionDevice(viewerCanvasState);
  if (!deviceResource?.device) {
    return unavailableResult('native-production-webgpu-device-unavailable');
  }
  const { device } = deviceResource;
  const renderScale = Number.isFinite(buildConfig?.renderScale)
    ? Number(buildConfig.renderScale)
    : 1;
  const renderW = Math.max(1, Math.round(canvasWidth * renderScale));
  const renderH = Math.max(1, Math.round(canvasHeight * renderScale));
  const sx = canvasWidth / renderW;
  const sy = canvasHeight / renderH;
  const projectionContract = buildWebGpuProjectionContract({
    camera,
    screenSpaceCamera,
    renderW,
    renderH,
    sx,
    sy
  });
  const workset = selectActiveProductionResidentWorkset({
    raw,
    device,
    canvasWidth,
    canvasHeight,
    productionResidentSelectionRequest
  });
  if (workset.contract?.residentWorksetReady !== true) {
    return unavailableResult(
      workset.contract?.reason ?? 'production-resident-workset-not-ready',
      workset.contract
    );
  }

  const stateResult = await buildWebGpu4DStatePositionsForCandidates({
    device,
    raw,
    candidateIndices: workset.candidateIndices,
    rawXyzOpacity: workset.rawXyzOpacity,
    buildConfig,
    projectionParams: projectionContract.data,
    readbackPolicy: 'none',
    keepGpuResources: true,
    sourceWorksetResourceIdentity: workset.contract.resourceIdentity
  });
  if (!stateResult.gpuResources) {
    return unavailableResult(
      'native-production-state-resource-not-ready',
      workset.contract
    );
  }
  const tileInput = await buildNativeWebGpuProductionTileInput({
    device,
    workset,
    stateGpuResources: stateResult.gpuResources,
    projectionParams: projectionContract.data
  });
  if (tileInput.contract?.tileAwareRenderInputReady !== true) {
    destroyBuffers([
      stateResult.gpuResources.statePositionBuffer,
      stateResult.gpuResources.renderAttributeBuffer,
      stateResult.gpuResources.footprintPayloadBuffer
    ]);
    return unavailableResult(
      tileInput.contract?.reason ?? 'native-production-tile-input-not-ready',
      workset.contract
    );
  }

  const tileList = await buildWebGpuGpuOwnedTileListLayout({
    device,
    tileInputResource: tileInput.gpuResource,
    canvasWidth,
    canvasHeight,
    tileSize: 16,
    sourceTileAwareRenderInputContract: tileInput.contract,
    keepGpuResources: true
  });
  let capacityOverflowDetected =
    tileList.contract?.tileReferenceCapacityContract
      ?.capacityOverflowDetected === true;
  if (
    capacityOverflowDetected ||
    tileList.contract?.gpuOwnedTileListLayoutReady !== true
  ) {
    destroyBuffers([
      stateResult.gpuResources.statePositionBuffer,
      stateResult.gpuResources.renderAttributeBuffer,
      stateResult.gpuResources.footprintPayloadBuffer,
      tileInput.gpuResource?.buffer,
      tileList.gpuResources?.tileTableBuffer,
      tileList.gpuResources?.referenceListBuffer,
      tileList.gpuResources?.tileChunkTableBuffer,
      tileList.gpuResources?.executionPlanBuffer,
      tileList.gpuResources?.compositorIndirectBuffer,
      ...(tileList.gpuResources?.transientBuffers ?? [])
    ]);
    const reason = capacityOverflowDetected
      ? 'native-production-tile-list-capacity-overflow-fail-closed'
      : tileList.contract?.reason ?? 'native-production-tile-list-not-ready';
    return {
      ...unavailableResult(reason, workset.contract),
      webgpu4DStateSourceContract: stateResult.contract,
      webgpuGaussianAttributeEvaluationContract:
        stateResult.gaussianAttributeEvaluationContract,
      webgpuGaussianFootprintEvaluationContract:
        stateResult.gaussianFootprintEvaluationContract,
      webgpuTileAwareRenderInputContract: tileInput.contract,
      webgpuGpuOwnedTileListLayoutContract: tileList.contract,
      webgpuProductionFrameDataPathContract:
        buildNativeWebGpuProductionFrameDataPathContract({
          status: 'blocked',
          worksetContract: workset.contract,
          stateResourceIdentity: stateResult.gpuResources.resourceIdentity,
          attributeResourceIdentity:
            `${stateResult.gpuResources.resourceIdentity}:attributes`,
          footprintResourceIdentity:
            `${stateResult.gpuResources.resourceIdentity}:footprint`,
          tileInputResourceIdentity: tileInput.gpuResource.resourceIdentity,
          tileListInputResourceIdentity:
            tileList.gpuResources?.layoutResourceIdentity ?? null,
          compositorInputResourceIdentity:
            tileList.gpuResources?.layoutResourceIdentity ?? null,
          stateRecordCount: stateResult.gpuResources.recordCount,
          tileInputRecordCount: tileInput.gpuResource.recordCount,
          tileReferenceCapacityContract:
            tileList.contract?.tileReferenceCapacityContract ?? null,
          boundedExecutionContract:
            tileList.contract?.boundedExecutionContract ?? null,
          gpuExecutionPlanContract:
            tileList.gpuResources?.executionPlanContract ?? null,
          gpuResourceLineagePreserved: true,
          capacityOverflowDetected,
          capacityOverflowFailClosed: true,
          silentDropAllowed: false,
          compositorSubmitted: false,
          reason
        })
    };
  }

  const compositor = await buildWebGpuTileListCompositor({
    device,
    gpuOwnedTileListLayout: tileList,
    canvasWidth,
    canvasHeight,
    viewerCanvasState,
    metadata: {
      ...(metadata ?? {}),
      projectionContract: projectionContract.summary
    }
  });
  capacityOverflowDetected =
    capacityOverflowDetected ||
    compositor.executionPlanObserver?.capacityOverflowDetected === true;
  const compositorSubmitted =
    compositor.contract?.compositorPassSubmitted === true;
  const productionFrameDataPathContract =
    buildNativeWebGpuProductionFrameDataPathContract({
      worksetContract: workset.contract,
      stateResourceIdentity: stateResult.gpuResources.resourceIdentity,
      attributeResourceIdentity:
        `${stateResult.gpuResources.resourceIdentity}:attributes`,
      footprintResourceIdentity:
        `${stateResult.gpuResources.resourceIdentity}:footprint`,
      tileInputResourceIdentity: tileInput.gpuResource.resourceIdentity,
      tileListInputResourceIdentity:
        tileList.gpuResources.layoutResourceIdentity,
      compositorInputResourceIdentity:
        tileList.gpuResources.layoutResourceIdentity,
      stateRecordCount: stateResult.gpuResources.recordCount,
      tileInputRecordCount: tileInput.gpuResource.recordCount,
      tileReferenceCapacityContract:
        compositor.tileReferenceCapacityContract ??
        tileList.contract.tileReferenceCapacityContract,
      boundedExecutionContract:
        compositor.contract?.boundedExecutionContract ?? null,
      gpuExecutionPlanContract:
        tileList.gpuResources.executionPlanContract,
      terminalExecutionPlanObserver: compositor.executionPlanObserver ?? null,
      cpuReferenceUsedAsProductionInput: false,
      diagnosticReadbackUsedAsProductionInput: false,
      javascriptVisibleSamplesUsedAsProductionInput: false,
      diagnosticMaxRecordsUsedAsProductionLimit: false,
      gpuResourceLineagePreserved:
        tileList.gpuResources.inputResourceIdentity ===
          tileInput.gpuResource.resourceIdentity &&
        tileList.gpuResources.sourceWorksetResourceIdentity ===
          workset.contract.resourceIdentity,
      capacityOverflowDetected,
      capacityOverflowFailClosed: true,
      silentDropAllowed: false,
      compositorSubmitted
    });
  const boundaryContract = buildWebGpuPhase3BackendBoundaryContract({
    dirtyContractReady: true,
    viewerAppGpuNewWebGpuPassResponsibilitiesAdded: false,
    step85RuntimePathPreserved:
      compositor.contract?.tileCompositorReady === true,
    step85TileCompositorPathPreserved:
      compositor.contract?.tileCompositorReady === true,
    step85CurrentTexturePathMaintained:
      compositor.contract?.currentTexturePathMaintained === true,
    step85CurrentTextureConnectionReady:
      compositor.contract?.currentTexturePathMaintained === true,
    step85CurrentTextureReadbackMatchesAdapterOutput:
      compositor.contract?.outputTextureReadbackMatchesSummary === true,
    step85CurrentTexturePreservationSource:
      'native-production-frame-data-path-tile-compositor-contract',
    nextDepthSortBoundaryReady:
      tileList.contract?.nextDepthSortInputReady === true,
    nextFinalCompositorBoundaryReady:
      compositor.contract?.tileCompositorReady === true,
    nextChunkLodStreamingBoundaryReady: false,
    reason: productionFrameDataPathContract.nativeProductionFrameDataPathReady
      ? null
      : productionFrameDataPathContract.reason
  });
  const viewerLifecycle = metadata?.viewerLifecycleIntegrationRequest ?? {};
  const frameImplementation = buildWebGpuTileCompositorFrameImplementation({
    backendImplementationKind:
      viewerLifecycle.backendImplementationKind ??
      WEBGPU_TILE_COMPOSITOR_FRAME_IMPLEMENTATION_MODE,
    webgpu4DStateSourceContract: stateResult.contract,
    webgpuGaussianAttributeEvaluationContract:
      stateResult.gaussianAttributeEvaluationContract,
    webgpuGaussianFootprintEvaluationContract:
      stateResult.gaussianFootprintEvaluationContract,
    webgpuTileAwareRenderInputContract: tileInput.contract,
    webgpuGpuOwnedTileListLayoutContract: tileList.contract,
    webgpuTileListCompositorContract: compositor.contract,
    webgpuPhase3BackendBoundaryContract: boundaryContract,
    viewerLoopPersistenceContract:
      viewerLifecycle.lastRenderTileCompositorViewerLoopPersistence ?? null,
    viewerLoopRuntimeFatalError:
      viewerLifecycle.lastRenderSchedulerFatalError ?? null
  });

  destroyBuffers([
    stateResult.gpuResources.statePositionBuffer,
    stateResult.gpuResources.renderAttributeBuffer,
    stateResult.gpuResources.footprintPayloadBuffer,
    tileInput.gpuResource.buffer,
    tileList.gpuResources.tileTableBuffer,
    tileList.gpuResources.referenceListBuffer,
    tileList.gpuResources.tileChunkTableBuffer,
    tileList.gpuResources.executionPlanBuffer,
    tileList.gpuResources.compositorIndirectBuffer,
    ...(tileList.gpuResources.transientBuffers ?? [])
  ]);

  return {
    status: productionFrameDataPathContract.nativeProductionFrameDataPathReady
      ? 'ok'
      : 'blocked',
    reason: productionFrameDataPathContract.reason ?? 'ok',
    computeMode: NATIVE_WEBGPU_PRODUCTION_FRAME_DATA_PATH_MODE,
    productionResidentWorksetContract: workset.contract,
    webgpuProductionFrameDataPathContract: productionFrameDataPathContract,
    webgpu4DStateSourceContract: stateResult.contract,
    webgpuGaussianAttributeEvaluationContract:
      stateResult.gaussianAttributeEvaluationContract,
    webgpuGaussianFootprintEvaluationContract:
      stateResult.gaussianFootprintEvaluationContract,
    webgpuTileAwareRenderInputContract: tileInput.contract,
    webgpuGpuOwnedTileListLayoutContract: tileList.contract,
    webgpuTileListCompositorContract: compositor.contract,
    webgpuTileListCompositorSummary: {
      compositorSummary: compositor.compositorSummary
    },
    webgpuPhase3BackendBoundaryContract: boundaryContract,
    webgpuTileCompositorFrameImplementation: frameImplementation,
    projectionContract: projectionContract.summary,
    metadata
  };
}
