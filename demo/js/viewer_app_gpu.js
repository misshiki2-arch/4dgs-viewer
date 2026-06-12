import { parseSplat4DV2 } from './splat4d_parser_v2.js';
import {
  fitCameraToRaw,
  computeGaussianDebugState,
  computeGaussianState,
  computeScreenSplat,
  DEFAULT_SINGLE_SPLAT_COMPARE_INPUT
} from './rot4d_math.js';
import { evalSHColor } from './sh_eval.js';
import {
  buildScreenSpaceCameraProxy,
  renderGpuFrame
} from './gpu_renderer.js';
import { getVisibleBuildConfig } from './gpu_visible_builder.js';
import { computeTileGrid } from './gpu_tile_utils.js';
import { buildCandidateInfo } from './gpu_candidate_path_selector.js';
import {
  buildCandidateSubsetInfo,
  buildGpuStubCandidateInfo,
  buildGpuSubsetCandidateInfo
} from './gpu_candidate_builder_gpu_stub.js';
import {
  buildCpuFilteredCandidateInfo,
  buildGpuFirstNCandidateInfo
} from './gpu_candidate_builder_gpu_firstn.js';
import {
  inspectGpuPackedPayloadItem,
  inspectPackedInterleavedTileCompositeItem
} from './gpu_packed_payload_draw_shared.js';
import {
  createGpuInteractionState,
  bindGpuDragInteraction
} from './gpu_interaction_utils.js';
import { buildEffectiveGpuQualityConfig } from './gpu_quality_override_utils.js';
import {
  applyInfoWrapStyle,
  applyPanelResizeStyle,
  ensureTileDebugControls,
  ensureTemporalIndexControls,
  ensureTemporalBucketControls,
  ensureQualityOverrideControls,
  ensurePackedPathControls,
  ensureDebugLogControls,
  ensureDeterministicStateNote,
  setDebugLogText,
  copyDebugLogText
} from './viewer_ui_controls.js';
import {
  loadAndApplyUiState,
  readAndSaveUiState,
  bindUiStatePersistence,
  summarizeUiState
} from './viewer_ui_state.js';
import { createRenderScheduler } from './viewer_render_scheduler.js';
import { createViewerPlayback } from './viewer_playback.js';
import { createViewerFileIO } from './viewer_file_io.js';
import { createViewerScene, applyViewerCameraPresetState } from './viewer_scene_setup.js';
import {
  parseViewerQueryState,
  buildViewerDeterministicSummary,
  applyViewerQueryStateToUi
} from './viewer_query_state.js';
import {
  downloadJsonDebug,
  sanitizePngFileName
} from './debug_download_utils.js';
import {
  sampleCurrentFramebufferStats as sampleFramebufferStatsForCanvas,
  captureSnapshotCanvasFromGpu,
  captureCanvasPngBlob
} from './gpu_framebuffer_debug_utils.js';
import { createGpuTileLiveDebugCapture } from './gpu_tile_live_debug_capture.js';
import {
  buildCandidateComparisonSummary,
  buildVisibleComparisonSummary
} from './gpu_visible_compare_debug.js';
import {
  captureGpuCandidateDryRunVisibleComparison,
  captureGpuCandidateScreenCoarseDryRunVisibleComparison,
  captureGpuCandidateScreenCoarseSweepComparison
} from './gpu_candidate_compare_runner.js';
import { runGpuVisibleRecordDryRun } from './gpu_visible_record_dry_run_runtime.js';
import { runGpuRawVisibleRecordDryRun } from './gpu_visible_record_raw_dry_run_runtime.js';
import { runWebGpuVisibleRecordDryRun } from './webgpu_visible_record_dry_run_runtime.js';
import { shouldUseWebGpuExclusiveFrameLifecycle } from './webgpu_exclusive_frame_lifecycle_switch.js';
import {
  buildWebGpuBackendViewerLifecycleControlledExecution,
  buildWebGpuBackendViewerLifecycleIntegrationBoundary
} from './webgpu_backend_viewer_lifecycle_integration.js';
import {
  executeWebGpuBackendViewerFrame
} from './webgpu_backend_viewer_frame_executor.js';
import {
  buildGpuCandidateShadowOptionsFromQuery,
  isGpuCandidateShadowCompareEnabled,
  runGpuCandidateShadowCompare
} from './gpu_candidate_shadow_compare_runner.js';
import {
  buildGpuCandidateRuntimeConfig,
  buildGpuCandidateRuntimeSummary
} from './gpu_candidate_runtime_selector.js';
import { buildGpuCandidateRuntimeFallbackSummary } from './gpu_candidate_runtime_fallback.js';
import { buildGpuOwnedCandidateSourceComparison } from './gpu_candidate_source_runtime.js';

const canvas = document.getElementById('glCanvas');

const ui = {
  fileInput: document.getElementById('file'),
  timeSlider: document.getElementById('time'),
  timeVal: document.getElementById('timeVal'),
  splatScaleSlider: document.getElementById('splatScale'),
  splatScaleVal: document.getElementById('splatScaleVal'),
  sigmaScaleSlider: document.getElementById('sigmaScale'),
  sigmaScaleVal: document.getElementById('sigmaScaleVal'),
  prefilterVarSlider: document.getElementById('prefilterVar'),
  prefilterVarVal: document.getElementById('prefilterVarVal'),
  renderScaleSlider: document.getElementById('renderScale'),
  renderScaleVal: document.getElementById('renderScaleVal'),
  strideSlider: document.getElementById('stride'),
  strideVal: document.getElementById('strideVal'),
  maxVisibleSlider: document.getElementById('maxVisible'),
  maxVisibleVal: document.getElementById('maxVisibleVal'),
  bgGraySlider: document.getElementById('bgGray'),
  bgGrayVal: document.getElementById('bgGrayVal'),
  useSHCheck: document.getElementById('useSH'),
  useRot4dCheck: document.getElementById('useRot4d'),
  useNativeRot4dCheck: document.getElementById('useNativeRot4d'),
  useNativeMarginalCheck: document.getElementById('useNativeMarginal'),
  forceSh3dCheck: document.getElementById('forceSh3d'),
  timeDurationSlider: document.getElementById('timeDuration'),
  timeDurationVal: document.getElementById('timeDurationVal'),
  playBtn: document.getElementById('play'),
  renderBtn: document.getElementById('renderBtn'),
  resetCamBtn: document.getElementById('resetCam'),
  info: document.getElementById('info'),
  drop: document.getElementById('drop')
};

ensureTileDebugControls(ui);
ensureTemporalIndexControls(ui);
ensureTemporalBucketControls(ui);
ensureQualityOverrideControls(ui);
ensurePackedPathControls(ui);
ensureDebugLogControls(ui);
ensureDeterministicStateNote(ui);

applyInfoWrapStyle(ui.info);
applyPanelResizeStyle(ui.info);

const deterministicQueryState = parseViewerQueryState();
const scene = createViewerScene(canvas, {
  debugPreserveDrawingBuffer: !!deterministicQueryState.debugPreserveDrawingBuffer
});
const { camera, controls, ensureGpu, getGpu, setCanvasSize } = scene;

let raw = null;
let lastDebugText = '';
let uiUnbindPersistence = null;
const tokenRef = { value: 0 };
const interactionState = createGpuInteractionState();
let playback = null;
let latestRenderResult = null;
let latestGpuCandidateShadowCompare = null;
let appliedCameraPresetName = deterministicQueryState.cameraPresetName ?? 'none';
let lastSnapshotSummary = {
  available: true,
  source: 'webgl-default-framebuffer-readpixels',
  renderWaitMode: 'direct-render-await',
  status: 'idle',
  reason: 'none'
};

const INSPECT_SOURCE_VALUES = new Set(['auto', 'actual-draw', 'packed', 'gpu-screen-fallback']);
const INSPECT_JSON_MODE_VALUES = new Set(['slim', 'full']);
const SHARED_REPRESENTATIVE_PIXEL_STORAGE_KEY = 'step86.sharedRepresentativePixel';
const SHARED_REPRESENTATIVE_ACCUMULATION_COLOR_STORAGE_KEY = 'step86.sharedRepresentativeAccumulationColor';
const SHARED_REPRESENTATIVE_DEFAULT_PIXEL = [2949, 688];
const SHARED_REPRESENTATIVE_COLOR_MATCH_TOLERANCE = 2.0 / 255.0;
const MAPPED_PROBE_PATCH_SIZES = [7, 21, 31];
const MAPPED_PROBE_PATCH_REFERENCE_STORAGE_PREFIX = 'step87.mappedProbePatchReference.';
const MAPPED_PROBE_PATCH_TOLERANCE = 2.0 / 255.0;

function refreshLatestDebugText(explicitText = null) {
  const text = explicitText ?? lastDebugText ?? ui.info?.textContent ?? '';
  lastDebugText = text;
  return text;
}

function exportLatestDebugTextToArea() {
  setDebugLogText(ui, refreshLatestDebugText());
}

function updateDeterministicStateNote() {
  if (!ui.deterministicStateNote) return;

  if (!deterministicQueryState.active) {
    ui.deterministicStateNote.textContent =
      'URL query can fix cameraPreset/time/drawPath/gpuFramePolicyOverride and window.gpuViewerDebug.captureFrame(...) can save the current canvas';
    return;
  }

  const parts = [];
  parts.push(`query active`);
  if (Array.isArray(deterministicQueryState.datasetCameraPosition) &&
      Array.isArray(deterministicQueryState.datasetCameraTarget)) {
    parts.push(`cameraSource=dataset-query-camera`);
    parts.push(`datasetImage=${deterministicQueryState.datasetImageName ?? 'none'}`);
    parts.push(`datasetTime=${Number.isFinite(deterministicQueryState.datasetTime) ? deterministicQueryState.datasetTime : 'none'}`);
  } else {
    parts.push(`cameraPreset=${deterministicQueryState.cameraPresetName ?? 'none'}`);
  }
  parts.push(`drawPath=${deterministicQueryState.drawPath ?? 'default'}`);
  parts.push(`tileCompositePath=${deterministicQueryState.tileCompositePath ?? 'baseline'}`);
  parts.push(`tileCompositePrimitive=${deterministicQueryState.tileCompositePrimitive ?? 'point'}`);
  parts.push(`inspectSource=${deterministicQueryState.inspectSource ?? 'auto'}`);
  parts.push(`inspectJsonMode=${deterministicQueryState.inspectJsonMode ?? 'slim'}`);
  parts.push(`gpuFramePolicyOverride=${deterministicQueryState.gpuFramePolicyOverride ?? 'auto'}`);
  if (Number.isFinite(deterministicQueryState.fixedCanvasWidth) &&
      Number.isFinite(deterministicQueryState.fixedCanvasHeight)) {
    parts.push(`fixedCanvas=${deterministicQueryState.fixedCanvasWidth}x${deterministicQueryState.fixedCanvasHeight}`);
  }
  if (deterministicQueryState.deterministicQueryString) {
    parts.push(`query=${deterministicQueryState.deterministicQueryString}`);
  }
  ui.deterministicStateNote.textContent =
    `${parts.join('  ')}  capture=window.gpuViewerDebug.captureFrame(...)`;
}

function updateStaticUiText() {
  ui.timeVal.textContent = Number(ui.timeSlider.value).toFixed(2);
  ui.splatScaleVal.textContent = Number(ui.splatScaleSlider.value).toFixed(2);
  ui.sigmaScaleVal.textContent = Number(ui.sigmaScaleSlider.value).toFixed(2);
  ui.prefilterVarVal.textContent = Number(ui.prefilterVarSlider.value).toFixed(2);
  ui.renderScaleVal.textContent = Number(ui.renderScaleSlider.value).toFixed(2);
  ui.strideVal.textContent = ui.strideSlider.value;
  ui.maxVisibleVal.textContent = ui.maxVisibleSlider.value;
  ui.bgGrayVal.textContent = ui.bgGraySlider.value;
  ui.timeDurationVal.textContent = Number(ui.timeDurationSlider.value).toFixed(1);
}

function buildRenderOverrides() {
  const quality = buildEffectiveGpuQualityConfig({
    ui,
    interactionState,
    isPlaying: playback ? playback.isPlaying() : false
  });

  return {
    ...quality.effectiveConfig,
    enablePackedVisiblePath: !!ui.usePackedVisiblePathCheck?.checked
  };
}

function buildDeterministicStateSummary() {
  const summary = buildViewerDeterministicSummary(deterministicQueryState);
  const referencePose = convertDatasetTransformMatrixToViewerPose(
    summary.datasetTransformMatrix,
    summary.datasetCameraConvention ?? 'nerf-blender-c2w'
  );
  const convertedPose = summary.cameraControlContract === 'interactive-from-reference' &&
    summary.datasetViewMatrixMode !== 'cuda-aligned'
    ? buildInteractiveCameraPoseFromReferencePose(
        referencePose,
        summary.cameraOrientationPolicy ?? 'roll-free-reference-screen-up'
      )
    : referencePose;
  const cameraFoVyRad = Number.isFinite(summary.datasetCameraFoVyRad)
    ? Number(summary.datasetCameraFoVyRad)
    : (Number.isFinite(summary.datasetCameraFoVy) ? Number(summary.datasetCameraFoVy) : null);
  const cameraFoVxRad = Number.isFinite(summary.datasetCameraFoVxRad)
    ? Number(summary.datasetCameraFoVxRad)
    : (Number.isFinite(summary.datasetCameraFoVx) ? Number(summary.datasetCameraFoVx) : null);
  const cameraFoVyDeg = Number.isFinite(cameraFoVyRad) ? (cameraFoVyRad * 180 / Math.PI) : null;
  const cameraFoVxDeg = Number.isFinite(cameraFoVxRad) ? (cameraFoVxRad * 180 / Math.PI) : null;
  const cudaAlignedScreenSpaceCamera = buildCudaAlignedScreenSpaceCameraSummary(summary, convertedPose);
  const fixedReferenceScreenSpaceCamera = buildCudaAlignedScreenSpaceCameraSummary(
    { ...summary, datasetViewMatrixMode: 'cuda-aligned' },
    referencePose
  );
  return {
    ...summary,
    appliedCameraPresetName,
    deterministicQueryString: summary.deterministicQueryString ?? '',
    deterministicUrlSummary: summary.deterministicUrlSummary ?? '',
    deterministicRawQueryString: summary.rawQueryString ?? '',
    cameraSource: summary.cameraSource ?? 'camera-preset',
    datasetCameraConvention: summary.datasetCameraConvention ?? null,
    datasetViewMatrixMode: summary.datasetViewMatrixMode ?? 'threejs',
    cameraControlContract: summary.cameraControlContract ?? null,
    cameraOrientationPolicy: summary.cameraOrientationPolicy ?? null,
    datasetPixelXSign: [-1, 1].includes(summary.datasetPixelXSign) ? Number(summary.datasetPixelXSign) : 1,
    datasetCameraLabel: summary.datasetCameraLabel ?? null,
    imageName: summary.datasetImageName ?? null,
    frameNumber: Number.isFinite(summary.datasetFrameNumber) ? Number(summary.datasetFrameNumber) : null,
    viewId: Number.isFinite(summary.datasetViewId) ? Number(summary.datasetViewId) : null,
    datasetTime: Number.isFinite(summary.datasetTime) ? Number(summary.datasetTime) : null,
    rawTransformMatrix: Array.isArray(summary.datasetTransformMatrix) ? summary.datasetTransformMatrix.map((row) => [...row]) : null,
    convertedCameraPose: convertedPose
      ? {
          position: [...convertedPose.position],
          target: [...convertedPose.target],
          up: [...convertedPose.up],
          forward: [...convertedPose.forward],
          targetDistance: convertedPose.targetDistance,
          convertedMatrix: convertedPose.convertedMatrix.map((row) => [...row])
        }
      : null,
    referenceCameraPose: referencePose
      ? {
          position: [...referencePose.position],
          target: [...referencePose.target],
          up: [...referencePose.up],
          forward: [...referencePose.forward],
          targetDistance: referencePose.targetDistance
        }
      : null,
    screenAxisMapping: convertedPose?.screenAxisMapping ?? null,
    cameraPosition: convertedPose
      ? [...convertedPose.position]
      : (Array.isArray(summary.datasetCameraPosition) ? [...summary.datasetCameraPosition] : null),
    cameraTarget: convertedPose
      ? [...convertedPose.target]
      : (Array.isArray(summary.datasetCameraTarget) ? [...summary.datasetCameraTarget] : null),
    cameraUp: convertedPose
      ? [...convertedPose.up]
      : (Array.isArray(summary.datasetCameraUp) ? [...summary.datasetCameraUp] : null),
    cameraFoVyRad,
    cameraFoVxRad,
    cameraFoVyDeg,
    cameraFoVxDeg,
    cameraFoVy: Number.isFinite(summary.datasetCameraFoVy) ? Number(summary.datasetCameraFoVy) : null,
    cameraFoVx: Number.isFinite(summary.datasetCameraFoVx) ? Number(summary.datasetCameraFoVx) : null,
    appliedCameraFovDeg: Number.isFinite(camera?.fov) ? Number(camera.fov) : null,
    intrinsics: {
      fx: Number.isFinite(summary.datasetFx) ? Number(summary.datasetFx) : null,
      fy: Number.isFinite(summary.datasetFy) ? Number(summary.datasetFy) : null,
      cx: Number.isFinite(summary.datasetCx) ? Number(summary.datasetCx) : null,
      cy: Number.isFinite(summary.datasetCy) ? Number(summary.datasetCy) : null
    },
    stride: Number.isFinite(summary.stride) ? Number(summary.stride) : null,
    bgGray: Number.isFinite(summary.bgGray) ? Number(summary.bgGray) : null,
    debugPreserveDrawingBuffer: typeof summary.debugPreserveDrawingBuffer === 'boolean'
      ? summary.debugPreserveDrawingBuffer
      : null,
    cudaReferenceLabel: summary.cudaReferenceLabel ?? null,
    cudaReferencePath: summary.cudaReferencePath ?? null,
    actualCameraPosition: vector3ToArray(camera?.position),
    actualCameraQuaternion: quaternionToArray(camera?.quaternion),
    actualCameraUp: vector3ToArray(camera?.up),
    actualControlsTarget: vector3ToArray(controls?.target),
    actualCameraFov: Number.isFinite(camera?.fov) ? Number(camera.fov) : null,
    actualCameraNear: Number.isFinite(camera?.near) ? Number(camera.near) : null,
    actualCameraFar: Number.isFinite(camera?.far) ? Number(camera.far) : null,
    actualCameraMatrixWorld: matrix4ToRows(camera?.matrixWorld),
    actualCameraRight: column3(matrix4ToRows(camera?.matrixWorld), 0),
    cudaAlignedScreenSpaceCamera,
    fixedReferenceScreenSpaceCamera,
    snapshotApiAvailable: true,
    snapshotCaptureSource: lastSnapshotSummary.source,
    snapshotRenderWaitMode: lastSnapshotSummary.renderWaitMode,
    snapshotLastStatus: lastSnapshotSummary.status,
    snapshotLastReason: lastSnapshotSummary.reason
  };
}

function buildGpuCandidateRuntimeDebugSummary(options = {}) {
  const runtimeConfig = buildGpuCandidateRuntimeConfig(deterministicQueryState, options);
  const runtimeSummary = buildGpuCandidateRuntimeSummary(runtimeConfig);
  const limitedDrawSummary = latestRenderResult?.limitedDrawRuntimeSummary ?? null;
  const fallback = buildGpuCandidateRuntimeFallbackSummary({
    runtimeConfig,
    shadowCompare: latestGpuCandidateShadowCompare
  });
  const effectiveFallback = limitedDrawSummary?.fallback ?? fallback;
  return {
    schemaVersion: 'step105-gpu-candidate-runtime-summary-debug-v1',
    timestamp: new Date().toISOString(),
    purpose: 'Summarize GPU candidate runtime selector, limited-draw candidate source promotion, and fallback decisions.',
    runtimeSummary,
    fallback: effectiveFallback,
    limitedDrawSummary,
    candidateSourceSummary: limitedDrawSummary?.candidateSourceSummary ?? null,
    candidateSourceComparison: limitedDrawSummary?.candidateSourceComparison ?? null,
    candidateCoverageSummary: limitedDrawSummary?.candidateCoverageSummary ?? null,
    displayCandidateSource: limitedDrawSummary?.displayCandidateSource ?? effectiveFallback.displayCandidateSource ?? runtimeSummary.displayCandidateSource,
    gpuCandidateUsedForDisplay: !!(limitedDrawSummary?.gpuCandidateUsedForDisplay ?? effectiveFallback.gpuCandidateUsedForDisplay),
    limitedDrawConnectedToRendering: true,
    limitedDrawUsedForCandidateSource: !!(limitedDrawSummary?.limitedDrawUsedForCandidateSource ?? effectiveFallback.limitedDrawUsedForCandidateSource),
    latestShadowCompareStatus: latestGpuCandidateShadowCompare?.status ?? null,
    latestShadowCompareAnyMismatch: latestGpuCandidateShadowCompare?.summary?.anyMismatch ?? null,
    deterministicState: buildSlimDeterministicStateSummary(buildDeterministicStateSummary()),
    lastRenderResultSummary: buildRenderResultInspectionSummary(latestRenderResult),
    readbackPolicy: runtimeSummary.readbackPolicy
  };
}

function getFixedCanvasSizeOverride() {
  const fixedCanvasWidth = Number.isFinite(deterministicQueryState.fixedCanvasWidth)
    ? Math.max(1, deterministicQueryState.fixedCanvasWidth | 0)
    : null;
  const fixedCanvasHeight = Number.isFinite(deterministicQueryState.fixedCanvasHeight)
    ? Math.max(1, deterministicQueryState.fixedCanvasHeight | 0)
    : null;
  if (fixedCanvasWidth === null || fixedCanvasHeight === null) {
    return null;
  }
  return { fixedCanvasWidth, fixedCanvasHeight };
}

function applyCanvasSize() {
  const fixedSize = getFixedCanvasSizeOverride();
  setCanvasSize(fixedSize ?? {});
}

function buildCanvasSizeSummary() {
  const gpu = getGpu();
  const gl = gpu?.gl ?? null;
  const contextAttributes = gl && typeof gl.getContextAttributes === 'function'
    ? gl.getContextAttributes()
    : null;
  const fixedSize = getFixedCanvasSizeOverride();
  const renderScale = Number(ui.renderScaleSlider?.value);
  return {
    clientWidth: Number.isFinite(canvas?.clientWidth) ? canvas.clientWidth : 0,
    clientHeight: Number.isFinite(canvas?.clientHeight) ? canvas.clientHeight : 0,
    windowInnerWidth: typeof window !== 'undefined' ? window.innerWidth : 0,
    windowInnerHeight: typeof window !== 'undefined' ? window.innerHeight : 0,
    canvasWidth: Number.isFinite(canvas?.width) ? canvas.width : 0,
    canvasHeight: Number.isFinite(canvas?.height) ? canvas.height : 0,
    framebufferWidth: gl ? (gl.drawingBufferWidth | 0) : (Number.isFinite(canvas?.width) ? canvas.width : 0),
    framebufferHeight: gl ? (gl.drawingBufferHeight | 0) : (Number.isFinite(canvas?.height) ? canvas.height : 0),
    devicePixelRatio: typeof window !== 'undefined' ? (window.devicePixelRatio || 1) : 1,
    renderScale: Number.isFinite(renderScale) ? renderScale : 1,
    fixedCanvasWidth: fixedSize?.fixedCanvasWidth ?? null,
    fixedCanvasHeight: fixedSize?.fixedCanvasHeight ?? null,
    fixedCanvasActive: !!fixedSize,
    contextAttributes,
    preserveDrawingBuffer: typeof contextAttributes?.preserveDrawingBuffer === 'boolean'
      ? contextAttributes.preserveDrawingBuffer
      : null,
    coordinateSpaceForReadPixels: 'webgl-default-framebuffer-pixels',
    probeCoordinateWidth: gl ? (gl.drawingBufferWidth | 0) : (Number.isFinite(canvas?.width) ? canvas.width : 0),
    probeCoordinateHeight: gl ? (gl.drawingBufferHeight | 0) : (Number.isFinite(canvas?.height) ? canvas.height : 0)
  };
}

function vector3ToArray(v) {
  return v ? [Number(v.x), Number(v.y), Number(v.z)] : null;
}

function eulerToArray(euler) {
  return euler ? [Number(euler.x), Number(euler.y), Number(euler.z), euler.order ?? 'XYZ'] : null;
}

function quaternionToArray(q) {
  return q ? [Number(q.x), Number(q.y), Number(q.z), Number(q.w)] : null;
}

function matrix4ToRows(matrix) {
  if (!matrix || !Array.isArray(matrix.elements)) return null;
  const e = matrix.elements;
  return [
    [e[0], e[4], e[8], e[12]],
    [e[1], e[5], e[9], e[13]],
    [e[2], e[6], e[10], e[14]],
    [e[3], e[7], e[11], e[15]]
  ].map((row) => row.map(Number));
}

function cloneMatrixRows(matrixRows) {
  return Array.isArray(matrixRows)
    ? matrixRows.map((row) => Array.isArray(row) ? row.map(Number) : [])
    : null;
}

function multiplyMatrix4Rows(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== 4 || b.length !== 4) return null;
  const out = Array.from({ length: 4 }, () => [0, 0, 0, 0]);
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        const av = Number(a[row]?.[k]);
        const bv = Number(b[k]?.[col]);
        if (!Number.isFinite(av) || !Number.isFinite(bv)) return null;
        sum += av * bv;
      }
      out[row][col] = sum;
    }
  }
  return out;
}

function invertRigidC2wMatrixRows(c2w) {
  if (!Array.isArray(c2w) || c2w.length !== 4) return null;
  const out = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
  ];
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      const value = Number(c2w[col]?.[row]);
      if (!Number.isFinite(value)) return null;
      out[row][col] = value;
    }
  }
  const t = [Number(c2w[0]?.[3]), Number(c2w[1]?.[3]), Number(c2w[2]?.[3])];
  if (t.some((value) => !Number.isFinite(value))) return null;
  for (let row = 0; row < 3; row++) {
    out[row][3] = -(out[row][0] * t[0] + out[row][1] * t[1] + out[row][2] * t[2]);
  }
  out[3][3] = 1;
  return out;
}

function column3(matrixRows, column) {
  return Array.isArray(matrixRows) && matrixRows.length >= 3
    ? [Number(matrixRows[0]?.[column]), Number(matrixRows[1]?.[column]), Number(matrixRows[2]?.[column])]
    : null;
}

function buildCudaAlignedScreenSpaceCameraSummary(summary, convertedPose) {
  const mode = summary?.datasetViewMatrixMode ?? 'threejs';
  const enabled = mode === 'cuda-aligned';
  const c2w = cloneMatrixRows(convertedPose?.convertedMatrix);
  const cudaAlignedViewMatrix = enabled && c2w ? invertRigidC2wMatrixRows(c2w) : null;
  const signConversionMatrix = [
    [-1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
  ];
  const threeJsViewMatrix = matrix4ToRows(camera?.matrixWorldInverse);
  const threeJsToCudaViewMatrix = threeJsViewMatrix
    ? multiplyMatrix4Rows(signConversionMatrix, threeJsViewMatrix)
    : null;
  const fx = Number.isFinite(summary?.datasetFx) ? Number(summary.datasetFx) : null;
  const fy = Number.isFinite(summary?.datasetFy) ? Number(summary.datasetFy) : null;
  const cx = Number.isFinite(summary?.datasetCx) ? Number(summary.datasetCx) : null;
  const cy = Number.isFinite(summary?.datasetCy) ? Number(summary.datasetCy) : null;
  const covarianceTanFov = -Math.tan(0.5);
  const pixelXSign = [-1, 1].includes(summary?.datasetPixelXSign)
    ? Number(summary.datasetPixelXSign)
    : 1;

  return {
    mode,
    enabled: enabled && !!cudaAlignedViewMatrix,
    viewMatrixSource: enabled ? 'dataset-transform-cuda-reader-c2w-inverse' : 'threejs-camera',
    signConversionMatrix,
    threeJsViewMatrix,
    threeJsToCudaViewMatrix,
    cudaAlignedViewMatrix,
    cudaAlignedCameraBasis: c2w
      ? {
          right: column3(c2w, 0),
          up: column3(c2w, 1),
          forward: column3(c2w, 2)
        }
      : null,
    intrinsics: { fx, fy, cx, cy },
    covarianceTanFovX: covarianceTanFov,
    covarianceTanFovY: covarianceTanFov,
    covarianceFocalContract: 'current-cuda-render-computeCov2D-uses-negative-tan-fov-0.5-rad',
    pixelXSign,
    pixelSignContract: 'debug-ablation-applied-to-center-projection-and-covariance-jacobian-x',
    projectionContract: enabled
      ? 'cuda-plus-z-forward-fx-fy-cx-cy'
      : 'threejs-camera-projection-matrix',
    screenYSign: 1,
    depthSign: 1
  };
}

function getCurrentUiDebugSummary() {
  return {
    drawPath: ui.drawPathSelect?.value ?? null,
    tileCompositePath: ui.tileCompositePathSelect?.value ?? null,
    tileCompositePrimitive: ui.tileCompositePrimitiveSelect?.value ?? null,
    time: Number.isFinite(Number(ui.timeSlider?.value)) ? Number(ui.timeSlider.value) : null,
    stride: Number.isFinite(Number(ui.strideSlider?.value)) ? Number(ui.strideSlider.value) : null,
    bgGray: Number.isFinite(Number(ui.bgGraySlider?.value)) ? Number(ui.bgGraySlider.value) : null
  };
}

function buildRawBoundsSummary() {
  if (!raw || !raw.xyz || raw.N <= 0) {
    return {
      available: false,
      count: raw?.N ?? 0,
      reason: raw ? 'missing-xyz-or-empty-scene' : 'scene-not-loaded'
    };
  }

  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  const dim = raw.xyzDim || 3;
  for (let i = 0; i < raw.N; i++) {
    for (let axis = 0; axis < 3; axis++) {
      const value = raw.xyz[i * dim + axis];
      if (!Number.isFinite(value)) continue;
      min[axis] = Math.min(min[axis], value);
      max[axis] = Math.max(max[axis], value);
    }
  }

  if (min.some((value) => !Number.isFinite(value)) || max.some((value) => !Number.isFinite(value))) {
    return {
      available: false,
      count: raw.N,
      reason: 'non-finite-bounds'
    };
  }

  const center = min.map((value, axis) => 0.5 * (value + max[axis]));
  const size = max.map((value, axis) => value - min[axis]);
  return {
    available: true,
    count: raw.N,
    min,
    max,
    center,
    size
  };
}

function buildActualCameraSummary() {
  if (!camera) return null;
  camera.updateMatrixWorld(true);
  const forwardTarget = typeof camera.position?.clone === 'function' ? camera.position.clone() : null;
  const forward = camera.getWorldDirection && forwardTarget
    ? vector3ToArray(camera.getWorldDirection(forwardTarget))
    : null;
  return {
    position: vector3ToArray(camera.position),
    rotation: eulerToArray(camera.rotation),
    quaternion: quaternionToArray(camera.quaternion),
    up: vector3ToArray(camera.up),
    forward,
    fov: Number(camera.fov),
    aspect: Number(camera.aspect),
    near: Number(camera.near),
    far: Number(camera.far),
    matrix: matrix4ToRows(camera.matrix),
    matrixWorld: matrix4ToRows(camera.matrixWorld),
    matrixWorldInverse: matrix4ToRows(camera.matrixWorldInverse),
    projectionMatrix: matrix4ToRows(camera.projectionMatrix)
  };
}

function subtract3(a, b) {
  return Array.isArray(a) && Array.isArray(b) && a.length >= 3 && b.length >= 3
    ? [Number(a[0]) - Number(b[0]), Number(a[1]) - Number(b[1]), Number(a[2]) - Number(b[2])]
    : null;
}

function vectorLengthSummary(v) {
  return Array.isArray(v) && v.length >= 3 ? length3(v.map(Number)) : null;
}

function referenceRightFromPose(pose) {
  const forward = normalize3(pose?.forward ?? []);
  const up = normalize3(pose?.up ?? []);
  return forward && up ? normalize3([
    forward[1] * up[2] - forward[2] * up[1],
    forward[2] * up[0] - forward[0] * up[2],
    forward[0] * up[1] - forward[1] * up[0]
  ]) : null;
}

function buildCameraControlContractComparison({ deterministicState, actualCamera, controlsSummary }) {
  const referencePose = deterministicState?.referenceCameraPose ?? null;
  const interactivePose = deterministicState?.convertedCameraPose ?? null;
  const actualForward = normalize3(actualCamera?.forward ?? []);
  const actualUp = normalize3(actualCamera?.up ?? []);
  const actualRight = normalize3(deterministicState?.actualCameraRight ?? []);
  const referenceForward = normalize3(referencePose?.forward ?? []);
  const referenceUp = normalize3(referencePose?.up ?? []);
  const referenceRight = referenceRightFromPose(referencePose);
  const interactiveUp = normalize3(interactivePose?.up ?? []);
  const positionDelta = subtract3(actualCamera?.position, referencePose?.position);
  const targetDelta = subtract3(controlsSummary?.target, referencePose?.target);
  return {
    schemaVersion: 'step131-camera-control-contract-comparison-v1',
    purpose: 'Compare fixed CUDA reference camera contract against current interactive Three.js/OrbitControls state without changing rendering.',
    cameraControlContract: deterministicState?.cameraControlContract ?? null,
    cameraOrientationPolicy: deterministicState?.cameraOrientationPolicy ?? null,
    activeDatasetViewMatrixMode: deterministicState?.datasetViewMatrixMode ?? 'threejs',
    fixedReference: {
      cameraSource: 'dataset-transform-matrix',
      datasetViewMatrixMode: 'cuda-aligned',
      referencePose,
      screenSpaceCamera: deterministicState?.fixedReferenceScreenSpaceCamera ?? null
    },
    interactive: {
      cameraSource: deterministicState?.cameraSource ?? null,
      datasetViewMatrixMode: deterministicState?.datasetViewMatrixMode ?? 'threejs',
      pose: interactivePose,
      actualCamera,
      controls: controlsSummary,
      screenSpaceCamera: deterministicState?.cudaAlignedScreenSpaceCamera ?? null
    },
    basisDotProducts: {
      actualForwardVsReferenceForward: actualForward && referenceForward ? dot3(actualForward, referenceForward) : null,
      actualUpVsReferenceUp: actualUp && referenceUp ? dot3(actualUp, referenceUp) : null,
      actualRightVsReferenceRight: actualRight && referenceRight ? dot3(actualRight, referenceRight) : null,
      interactiveUpVsReferenceUp: interactiveUp && referenceUp ? dot3(interactiveUp, referenceUp) : null
    },
    deltas: {
      actualPositionMinusReferencePosition: positionDelta,
      actualPositionDeltaLength: vectorLengthSummary(positionDelta),
      controlsTargetMinusReferenceTarget: targetDelta,
      controlsTargetDeltaLength: vectorLengthSummary(targetDelta)
    },
    screenAxisMapping: deterministicState?.screenAxisMapping ?? null,
    interpretationHints: {
      nearOneDotMeansAligned: true,
      nearMinusOneDotMeansFlipped: true,
      positionAndTargetShouldMatchBeforeManualInteraction: true,
      fixedReferenceUsesCudaAlignedProjectionSigns: true,
      interactiveUsesThreeJsCameraProjection: true
    }
  };
}

function getCameraDebugState() {
  const deterministicState = buildDeterministicStateSummary();
  const actualCamera = buildActualCameraSummary();
  const controlsSummary = {
    target: vector3ToArray(controls?.target),
    enabled: typeof controls?.enabled === 'boolean' ? controls.enabled : null,
    enableDamping: typeof controls?.enableDamping === 'boolean' ? controls.enableDamping : null,
    screenSpacePanning: typeof controls?.screenSpacePanning === 'boolean' ? controls.screenSpacePanning : null,
    panSpeed: Number.isFinite(controls?.panSpeed) ? Number(controls.panSpeed) : null,
    rotateSpeed: Number.isFinite(controls?.rotateSpeed) ? Number(controls.rotateSpeed) : null
  };
  return {
    timestamp: new Date().toISOString(),
    locationHref: window.location.href,
    rawQueryString: window.location.search.replace(/^\?/, ''),
    camera: actualCamera,
    controls: controlsSummary,
    deterministicState,
    cameraControlContractComparison: buildCameraControlContractComparison({
      deterministicState,
      actualCamera,
      controlsSummary
    }),
    convertedCameraPose: deterministicState.convertedCameraPose ?? null,
    lastRenderResultSummary: buildRenderResultInspectionSummary(latestRenderResult),
    canvasSizeSummary: buildCanvasSizeSummary(),
    sceneBounds: buildRawBoundsSummary(),
    uiState: getCurrentUiDebugSummary()
  };
}

function sampleCurrentFramebufferStats(options = {}) {
  const gpu = getGpu();
  return sampleFramebufferStatsForCanvas(gpu?.gl ?? null, canvas, options, buildCanvasSizeSummary());
}

async function captureCurrentDebugBundle(options = {}) {
  const includeInspect = !!options.includeInspect;
  let inspectActiveSplatResult = null;
  let inspectActiveSplatError = null;
  if (includeInspect) {
    try {
      inspectActiveSplatResult = await inspectActiveSplat({
        ensureCurrentFrame: options.ensureCurrentFrame !== false,
        inspectSource: options.inspectSource ?? deterministicQueryState.inspectSource ?? 'actual-draw',
        outputMode: options.outputMode ?? deterministicQueryState.inspectJsonMode ?? 'slim',
        index: Number.isFinite(Number(options.index)) ? Number(options.index) : 0
      });
    } catch (error) {
      inspectActiveSplatError = error?.message ?? 'unknown-inspect-error';
    }
  }

  const cameraDebugState = getCameraDebugState();
  const framebufferStats = sampleCurrentFramebufferStats(options.framebufferStats ?? {});
  const lastRenderResultSummary = buildRenderResultInspectionSummary(latestRenderResult);
  return {
    timestamp: new Date().toISOString(),
    locationHref: window.location.href,
    deterministicState: buildDeterministicStateSummary(),
    cameraDebugState,
    framebufferStats,
    lastRenderResultSummary,
    drawPathSummary: latestRenderResult?.drawPathSummary ?? null,
    canvasSizeSummary: buildCanvasSizeSummary(),
    inspectActiveSplatIncluded: includeInspect,
    inspectActiveSplat: inspectActiveSplatResult,
    inspectActiveSplatError
  };
}

function normalizeRepresentativeIndexList(input, fallbackReference = null) {
  const referenceLikeInput = Array.isArray(input?.selectedIndices) || Array.isArray(input?.splats)
    ? input
    : null;
  const source = Array.isArray(input)
    ? input
    : (Array.isArray(input?.indices)
        ? input.indices
        : (referenceLikeInput?.selectedIndices ?? fallbackReference?.selectedIndices));
  if (!Array.isArray(source)) return [];
  const out = [];
  const seen = new Set();
  for (const entry of source) {
    const value = Number.isFinite(Number(entry))
      ? Number(entry)
      : (Number.isFinite(Number(entry?.index)) ? Number(entry.index) : NaN);
    if (!Number.isFinite(value)) continue;
    const index = value | 0;
    if (seen.has(index)) continue;
    seen.add(index);
    out.push(index);
  }
  return out;
}

function toFiniteNumberOrNull(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function cloneNumberArray(values, length = null) {
  if (!Array.isArray(values) && !(values instanceof Float32Array)) return null;
  const count = Number.isFinite(length) ? Math.min(values.length, length | 0) : values.length;
  const out = [];
  for (let i = 0; i < count; i++) {
    out.push(toFiniteNumberOrNull(values[i]));
  }
  return out;
}

function computeRasterRectFromCenterRadius(centerPx, radiusPx) {
  if (!Array.isArray(centerPx) || centerPx.length < 2 || !Number.isFinite(radiusPx)) return null;
  const minX = Math.floor(centerPx[0] - radiusPx);
  const minY = Math.floor(centerPx[1] - radiusPx);
  const maxXExclusive = Math.floor(centerPx[0] + radiusPx) + 1;
  const maxYExclusive = Math.floor(centerPx[1] + radiusPx) + 1;
  return [minX, minY, maxXExclusive, maxYExclusive];
}

function computeTileCoverageFromTileRange(tileRange) {
  if (!Array.isArray(tileRange) || tileRange.length < 4) return null;
  const minX = Number(tileRange[0]);
  const minY = Number(tileRange[1]);
  const maxX = Number(tileRange[2]);
  const maxY = Number(tileRange[3]);
  if (![minX, minY, maxX, maxY].every(Number.isFinite)) return null;
  return Math.max(0, (maxX - minX + 1) * (maxY - minY + 1));
}

function decodePackedPayloadItem(packed, itemIndex, floatsPerItem = 16) {
  const stride = Number.isFinite(floatsPerItem) ? Math.max(16, floatsPerItem | 0) : 16;
  const index = Number.isFinite(itemIndex) ? itemIndex | 0 : -1;
  const base = index * stride;
  if (!(packed instanceof Float32Array) || index < 0 || base + 16 > packed.length) return null;
  const centerPx = [packed[base + 0], packed[base + 1]];
  const radius = packed[base + 2];
  const depth = packed[base + 3];
  const colorAlpha = [
    packed[base + 4],
    packed[base + 5],
    packed[base + 6],
    packed[base + 7]
  ];
  const conic = [
    packed[base + 8],
    packed[base + 9],
    packed[base + 10]
  ];
  const misc = [
    packed[base + 12],
    packed[base + 13],
    packed[base + 14],
    packed[base + 15]
  ];
  const miscLooksLikeRect = misc.every(Number.isFinite) && misc[2] >= misc[0] && misc[3] >= misc[1];
  return {
    centerPx,
    radius,
    radiusPx: radius,
    depth,
    colorAlpha,
    color: colorAlpha.slice(0, 3),
    alpha: colorAlpha[3],
    opacity: colorAlpha[3],
    conic,
    reserved: packed[base + 11],
    misc,
    rasterRect: miscLooksLikeRect ? misc : computeRasterRectFromCenterRadius(centerPx, radius),
    rasterRectFromCenterRadius: computeRasterRectFromCenterRadius(centerPx, radius),
    packBaseFloatOffset: base,
    floatsPerItem: stride
  };
}

function buildActualPayloadFromVisibleItem(item, visibleIndex) {
  if (!item) return null;
  const centerPx = Array.isArray(item.centerPx)
    ? cloneNumberArray(item.centerPx, 2)
    : [toFiniteNumberOrNull(item.px), toFiniteNumberOrNull(item.py)];
  const radius = Number.isFinite(item.radiusPx) ? Number(item.radiusPx) : Number(item.radius);
  const colorAlpha = Array.isArray(item.colorAlpha)
    ? cloneNumberArray(item.colorAlpha, 4)
    : [
        ...(Array.isArray(item.color) ? cloneNumberArray(item.color, 3) : [0, 0, 0]),
        toFiniteNumberOrNull(item.opacity)
      ];
  const tileCoverage = computeTileCoverageFromTileRange(item.tileRange);
  return {
    payloadSource: 'visible',
    sourceItemIndex: visibleIndex,
    visibleIndex,
    originalSplatIndex: Number.isFinite(item.srcIndex) ? item.srcIndex | 0 : null,
    centerPx,
    depth: toFiniteNumberOrNull(item.depth),
    radius,
    radiusPx: radius,
    conic: cloneNumberArray(item.conic, 3),
    colorAlpha,
    color: colorAlpha ? colorAlpha.slice(0, 3) : null,
    alpha: colorAlpha ? colorAlpha[3] : null,
    opacity: colorAlpha ? colorAlpha[3] : null,
    rasterRect: cloneNumberArray(item.aabb, 4) ?? computeRasterRectFromCenterRadius(centerPx, radius),
    tileRange: cloneNumberArray(item.tileRange, 4),
    tileCoverage,
    stateConvention: typeof item.stateConvention === 'string' ? item.stateConvention : null,
    usedCuda4DStateHelper: typeof item.usedCuda4DStateHelper === 'boolean' ? item.usedCuda4DStateHelper : null,
    stateHelperVersion: typeof item.stateHelperVersion === 'string' ? item.stateHelperVersion : null
  };
}

function findPackedScreenSpacePayloadForIndex(packedScreenSpace, originalIndex) {
  const sourceItems = Array.isArray(packedScreenSpace?.sourceItems)
    ? packedScreenSpace.sourceItems
    : (Array.isArray(packedScreenSpace?.visible) ? packedScreenSpace.visible : []);
  let sourceItemIndex = -1;
  for (let i = 0; i < sourceItems.length; i++) {
    const srcIndex = Number(sourceItems[i]?.srcIndex);
    if (Number.isFinite(srcIndex) && (srcIndex | 0) === originalIndex) {
      sourceItemIndex = i;
      break;
    }
  }
  if (sourceItemIndex < 0) return null;

  const decoded = decodePackedPayloadItem(
    packedScreenSpace?.packed,
    sourceItemIndex,
    packedScreenSpace?.floatsPerItem
  );
  const sourceItem = sourceItems[sourceItemIndex];
  return {
    ...(decoded ?? buildActualPayloadFromVisibleItem(sourceItem, sourceItemIndex)),
    payloadSource: decoded ? 'packedScreenSpace.packed' : 'packedScreenSpace.sourceItems',
    packedIndex: sourceItemIndex,
    sourceItemIndex,
    visibleIndex: sourceItemIndex,
    originalSplatIndex: originalIndex,
    sourcePath: packedScreenSpace?.path ?? 'unknown',
    packedContract: packedScreenSpace?.packedContract ?? 'unknown',
    transformPath: packedScreenSpace?.transformSummary?.actualTransformPath ?? 'unknown',
    tileRange: cloneNumberArray(sourceItem?.tileRange, 4),
    tileCoverage: computeTileCoverageFromTileRange(sourceItem?.tileRange)
  };
}

function findAccumulationPayloadOccurrencesForIndex(tileCompositePlan, originalIndex) {
  const batches = Array.isArray(tileCompositePlan?.batches) ? tileCompositePlan.batches : [];
  const occurrences = [];
  let globalItemIndex = 0;
  for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
    const batch = batches[batchIndex];
    const packedCount = Number.isFinite(batch?.packedCount) ? Math.max(0, batch.packedCount | 0) : 0;
    const sourceIndices = batch?.sourceIndices instanceof Uint32Array ? batch.sourceIndices : null;
    const orderedIndices = batch?.orderedIndices instanceof Uint32Array ? batch.orderedIndices : null;
    for (let localIndex = 0; localIndex < packedCount; localIndex++) {
      const sourceSplatIndex = sourceIndices ? (sourceIndices[localIndex] | 0) : null;
      if (sourceSplatIndex !== originalIndex) continue;
      const decoded = decodePackedPayloadItem(batch.packed, localIndex, batch.floatsPerItem);
      occurrences.push({
        ...(decoded ?? {}),
        payloadSource: 'tileCompositePlan.batches[].packed',
        batchIndex,
        tileId: Number.isFinite(batch?.tileId) ? batch.tileId | 0 : -1,
        tile: [
          Number.isFinite(batch?.tx) ? batch.tx | 0 : 0,
          Number.isFinite(batch?.ty) ? batch.ty | 0 : 0
        ],
        tileRect: cloneNumberArray(batch?.rect, 4),
        localIndex,
        packedIndex: localIndex,
        accumulationGlobalItemIndex: globalItemIndex + localIndex,
        sourceVisibleIndex: orderedIndices ? (orderedIndices[localIndex] | 0) : null,
        originalSplatIndex: sourceSplatIndex,
        batchPackedCount: packedCount
      });
    }
    globalItemIndex += packedCount;
  }
  return occurrences;
}

function summarizeAccumulationOccurrenceConsistency(occurrences) {
  if (!Array.isArray(occurrences) || occurrences.length <= 1) {
    return { consistent: true, maxCenterDelta: 0, maxDepthDelta: 0, maxConicDelta: 0, maxRadiusDelta: 0 };
  }
  const first = occurrences[0];
  let maxCenterDelta = 0;
  let maxDepthDelta = 0;
  let maxConicDelta = 0;
  let maxRadiusDelta = 0;
  for (const occurrence of occurrences.slice(1)) {
    maxCenterDelta = Math.max(maxCenterDelta, vectorDistance(first.centerPx, occurrence.centerPx));
    maxDepthDelta = Math.max(maxDepthDelta, scalarAbsDelta(first.depth, occurrence.depth));
    maxConicDelta = Math.max(maxConicDelta, maxVectorAbsDelta(first.conic, occurrence.conic));
    maxRadiusDelta = Math.max(maxRadiusDelta, scalarAbsDelta(first.radius, occurrence.radius));
  }
  return {
    consistent: maxCenterDelta <= 1e-6 && maxDepthDelta <= 1e-6 && maxConicDelta <= 1e-9 && maxRadiusDelta <= 1e-6,
    maxCenterDelta,
    maxDepthDelta,
    maxConicDelta,
    maxRadiusDelta
  };
}

function scalarAbsDelta(a, b) {
  const av = Number(a);
  const bv = Number(b);
  return Number.isFinite(av) && Number.isFinite(bv) ? Math.abs(av - bv) : null;
}

function vectorDelta(a, b, length = null) {
  if (!Array.isArray(a) || !Array.isArray(b)) return null;
  const count = Number.isFinite(length) ? length : Math.min(a.length, b.length);
  const out = [];
  for (let i = 0; i < count; i++) {
    const av = Number(a[i]);
    const bv = Number(b[i]);
    out.push(Number.isFinite(av) && Number.isFinite(bv) ? bv - av : null);
  }
  return out;
}

function maxVectorAbsDelta(a, b, length = null) {
  const delta = vectorDelta(a, b, length);
  if (!Array.isArray(delta)) return null;
  let maxDelta = 0;
  for (const value of delta) {
    if (!Number.isFinite(value)) return null;
    maxDelta = Math.max(maxDelta, Math.abs(value));
  }
  return maxDelta;
}

function vectorDistance(a, b, length = null) {
  const delta = vectorDelta(a, b, length);
  if (!Array.isArray(delta)) return null;
  let sum = 0;
  for (const value of delta) {
    if (!Number.isFinite(value)) return null;
    sum += value * value;
  }
  return Math.sqrt(sum);
}

function getRepresentativeReferenceByIndex(referenceDebug) {
  const splats = Array.isArray(referenceDebug?.splats) ? referenceDebug.splats : [];
  const map = new Map();
  for (const splat of splats) {
    const index = Number(splat?.index);
    if (Number.isFinite(index)) map.set(index | 0, splat);
  }
  return map;
}

function buildReferencePayloadSummary(referenceEntry) {
  const projection = referenceEntry?.projections?.viewerCudaAlignedCurrent ?? null;
  if (!projection) return null;
  const color = Array.isArray(referenceEntry?.color) ? cloneNumberArray(referenceEntry.color, 3) : null;
  const alpha = Number.isFinite(projection.opacity)
    ? Number(projection.opacity)
    : (Number.isFinite(referenceEntry?.cudaNativeState?.opacity) ? Number(referenceEntry.cudaNativeState.opacity) : null);
  return {
    centerPx: cloneNumberArray(projection.centerPx, 2),
    depth: toFiniteNumberOrNull(projection.depth),
    conic: cloneNumberArray(projection.conic, 3),
    radius: toFiniteNumberOrNull(projection.radius),
    radiusPx: toFiniteNumberOrNull(projection.radius),
    rasterRect: cloneNumberArray(projection.rasterRect, 4),
    color,
    alpha,
    opacity: alpha,
    colorAlpha: color ? [...color, alpha] : null,
    culled: !!projection.culled,
    cullReason: projection.cullReason ?? 'none'
  };
}

function buildPayloadComparison(referencePayload, actualPayload) {
  if (!referencePayload || !actualPayload) return null;
  const centerDelta = vectorDelta(referencePayload.centerPx, actualPayload.centerPx, 2);
  const conicDelta = vectorDelta(referencePayload.conic, actualPayload.conic, 3);
  const rasterRectDelta = vectorDelta(referencePayload.rasterRect, actualPayload.rasterRect, 4);
  const colorDelta = vectorDelta(referencePayload.color, actualPayload.color, 3);
  return {
    centerDelta,
    centerDistance: vectorDistance(referencePayload.centerPx, actualPayload.centerPx, 2),
    depthDelta: scalarAbsDelta(referencePayload.depth, actualPayload.depth),
    conicDelta,
    conicMaxAbsDelta: maxVectorAbsDelta(referencePayload.conic, actualPayload.conic, 3),
    radiusDelta: scalarAbsDelta(referencePayload.radius, actualPayload.radius),
    rasterRectDelta,
    rasterRectMaxAbsDelta: maxVectorAbsDelta(referencePayload.rasterRect, actualPayload.rasterRect, 4),
    alphaDelta: scalarAbsDelta(referencePayload.alpha, actualPayload.alpha),
    colorDelta,
    colorMaxAbsDelta: maxVectorAbsDelta(referencePayload.color, actualPayload.color, 3)
  };
}

function summarizeRepresentativePayloadComparisons(items) {
  const found = items.filter((item) => item.found);
  const missing = items.filter((item) => !item.found);
  const stats = {};
  const fields = [
    ['centerDistance', (item) => item.compare?.centerDistance],
    ['depthAbsDelta', (item) => item.compare?.depthDelta],
    ['conicMaxAbsDelta', (item) => item.compare?.conicMaxAbsDelta],
    ['radiusAbsDelta', (item) => item.compare?.radiusDelta],
    ['rasterRectMaxAbsDelta', (item) => item.compare?.rasterRectMaxAbsDelta],
    ['alphaAbsDelta', (item) => item.compare?.alphaDelta],
    ['colorMaxAbsDelta', (item) => item.compare?.colorMaxAbsDelta]
  ];
  for (const [name, getter] of fields) {
    const values = found.map(getter).filter(Number.isFinite).sort((a, b) => a - b);
    stats[name] = values.length > 0
      ? {
          count: values.length,
          mean: values.reduce((sum, value) => sum + value, 0) / values.length,
          median: values[Math.floor(values.length / 2)],
          max: values[values.length - 1]
        }
      : { count: 0, mean: null, median: null, max: null };
  }
  const missingReasons = {};
  for (const item of missing) {
    const reason = item.missingReason ?? 'unknown';
    missingReasons[reason] = (missingReasons[reason] ?? 0) + 1;
  }
  const allCompared = found.length > 0 && found.every((item) => !!item.compare);
  const payloadMatchesDebug =
    allCompared &&
    stats.centerDistance.max !== null && stats.centerDistance.max <= 1e-3 &&
    stats.depthAbsDelta.max !== null && stats.depthAbsDelta.max <= 1e-4 &&
    stats.conicMaxAbsDelta.max !== null && stats.conicMaxAbsDelta.max <= 1e-6 &&
    stats.radiusAbsDelta.max !== null && stats.radiusAbsDelta.max <= 1e-6;
  return {
    selectedCount: items.length,
    foundCount: found.length,
    missingCount: missing.length,
    missingReasons,
    stats,
    indexMappingProblemLikely: missing.length > 0 || found.some((item) => !Number.isFinite(item.actualPayload?.originalSplatIndex)),
    payloadMatchesDebug,
    classification: missing.length > 0
      ? 'C'
      : (payloadMatchesDebug ? 'A' : 'B'),
    classificationReason: missing.length > 0
      ? 'one-or-more-representative-indices-missing-from-actual-payload'
      : (payloadMatchesDebug
          ? 'debug-viewerCudaAlignedCurrent-values-match-actual-payload-within-tolerance'
          : 'debug-viewerCudaAlignedCurrent-values-differ-from-actual-payload')
  };
}

function inspectActualPayloadForOriginalIndex(renderResult, originalIndex) {
  if (!renderResult) {
    return {
      found: false,
      missingReason: 'payload-not-retained',
      visiblePayload: null,
      packedPayload: null,
      accumulationOccurrences: []
    };
  }
  const visible = Array.isArray(renderResult.visible) ? renderResult.visible : [];
  const visibleIndex = visible.findIndex((item) => {
    const srcIndex = Number(item?.srcIndex);
    return Number.isFinite(srcIndex) && (srcIndex | 0) === originalIndex;
  });
  const visiblePayload = visibleIndex >= 0
    ? buildActualPayloadFromVisibleItem(visible[visibleIndex], visibleIndex)
    : null;
  const packedPayload = findPackedScreenSpacePayloadForIndex(renderResult.packedScreenSpace, originalIndex);
  const accumulationOccurrences = findAccumulationPayloadOccurrencesForIndex(renderResult.tileCompositePlan, originalIndex);
  const accumulationFirst = accumulationOccurrences.length > 0 ? accumulationOccurrences[0] : null;
  const actualPayload = accumulationFirst ?? packedPayload ?? visiblePayload;
  const found = !!actualPayload;
  let missingReason = 'none';
  if (!found) {
    if (visible.length > 0 && visible.every((item) => !Number.isFinite(item?.srcIndex))) {
      missingReason = 'source-index-not-preserved';
    } else if (!renderResult.packedScreenSpace && !renderResult.tileCompositePlan) {
      missingReason = 'payload-not-retained';
    } else {
      missingReason = 'not-visible-or-culled';
    }
  }
  return {
    found,
    missingReason,
    actualPayload,
    visiblePayload,
    packedPayload,
    accumulationPayload: accumulationFirst,
    accumulationOccurrences,
    accumulationOccurrenceCount: accumulationOccurrences.length,
    accumulationOccurrenceConsistency: summarizeAccumulationOccurrenceConsistency(accumulationOccurrences)
  };
}

function hasRetainedActualPayload(renderResult) {
  if (!renderResult) return false;
  if (Array.isArray(renderResult.visible) && renderResult.visible.length > 0) return true;
  if (renderResult.packedScreenSpace?.packed instanceof Float32Array) return true;
  if (Array.isArray(renderResult.packedScreenSpace?.sourceItems) && renderResult.packedScreenSpace.sourceItems.length > 0) return true;
  if (Array.isArray(renderResult.tileCompositePlan?.batches) && renderResult.tileCompositePlan.batches.length > 0) return true;
  return false;
}

function buildActualPayloadRetentionSummary(renderResult) {
  const visible = Array.isArray(renderResult?.visible) ? renderResult.visible : [];
  const packedScreenSpace = renderResult?.packedScreenSpace ?? null;
  const sourceItems = Array.isArray(packedScreenSpace?.sourceItems) ? packedScreenSpace.sourceItems : [];
  const batches = Array.isArray(renderResult?.tileCompositePlan?.batches) ? renderResult.tileCompositePlan.batches : [];
  const batchWithPackedCount = batches.filter((batch) => batch?.packed instanceof Float32Array).length;
  const batchWithSourceIndicesCount = batches.filter((batch) => batch?.sourceIndices instanceof Uint32Array).length;
  const visibleWithSrcIndexCount = visible.filter((item) => Number.isFinite(Number(item?.srcIndex))).length;
  const sourceItemsWithSrcIndexCount = sourceItems.filter((item) => Number.isFinite(Number(item?.srcIndex))).length;
  return {
    renderResultPresent: !!renderResult,
    retainedActualPayload: hasRetainedActualPayload(renderResult),
    visibleCount: visible.length,
    visibleWithSrcIndexCount,
    visibleSrcIndexPreserved: visible.length > 0 && visibleWithSrcIndexCount === visible.length,
    packedScreenSpacePresent: !!packedScreenSpace,
    packedScreenSpacePackedPresent: packedScreenSpace?.packed instanceof Float32Array,
    packedScreenSpacePackedCount: Number.isFinite(packedScreenSpace?.packedCount) ? packedScreenSpace.packedCount : 0,
    packedScreenSpaceFloatsPerItem: Number.isFinite(packedScreenSpace?.floatsPerItem) ? packedScreenSpace.floatsPerItem : 0,
    packedScreenSpaceSourceItemCount: sourceItems.length,
    packedScreenSpaceSourceItemsWithSrcIndexCount: sourceItemsWithSrcIndexCount,
    packedScreenSpaceSrcIndexPreserved: sourceItems.length > 0 && sourceItemsWithSrcIndexCount === sourceItems.length,
    tileCompositePlanPresent: !!renderResult?.tileCompositePlan,
    tileCompositeBatchCount: batches.length,
    tileCompositeBatchWithPackedCount: batchWithPackedCount,
    tileCompositeBatchWithSourceIndicesCount: batchWithSourceIndicesCount,
    tileCompositeSourceIndicesPreserved: batches.length > 0 && batchWithSourceIndicesCount === batches.length,
    tileCompositeTotalPackedCount: batches.reduce((sum, batch) => (
      sum + (Number.isFinite(batch?.packedCount) ? Math.max(0, batch.packedCount | 0) : 0)
    ), 0),
    actualDrawPath:
      renderResult?.drawThroughputSummary?.actualDrawPath ??
      renderResult?.drawPathSummary?.actualPath ??
      'none',
    tileCompositePath: renderResult?.executionSummary?.tileCompositePath ?? 'none',
    tileCompositePrimitive: renderResult?.executionSummary?.tileCompositePrimitive ?? 'none'
  };
}

function delayMs(ms) {
  return new Promise((resolve) => setTimeout(resolve, Math.max(0, ms | 0)));
}

async function waitForRenderSchedulerIdle(timeoutMs = 2000) {
  const start = performance.now();
  while (
    scheduler?.state &&
    (scheduler.state.rendering || scheduler.state.renderPending) &&
    performance.now() - start < timeoutMs
  ) {
    await delayMs(16);
  }
  return {
    idle: !scheduler?.state || (!scheduler.state.rendering && !scheduler.state.renderPending),
    waitedMs: performance.now() - start,
    schedulerState: scheduler?.state
      ? {
          rendering: !!scheduler.state.rendering,
          renderPending: !!scheduler.state.renderPending,
          needsRenderAgain: !!scheduler.state.needsRenderAgain
        }
      : null
  };
}

async function renderCurrentFrameForDebugPayload(options = {}) {
  const attempts = [];
  const idleBefore = await waitForRenderSchedulerIdle(options.schedulerIdleTimeoutMs ?? 2000);
  attempts.push({ stage: 'wait-for-scheduler-idle', ...idleBefore });

  const maxAttempts = Number.isFinite(options.maxAttempts)
    ? Math.max(1, Math.min(5, options.maxAttempts | 0))
    : 3;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const result = await renderCurrentFrame({
      preservePreviousOnNull: true,
      isolatedTokenRef: true
    });
    const retentionSummary = buildActualPayloadRetentionSummary(result);
    attempts.push({
      stage: 'render-current-frame',
      attempt,
      resultPresent: !!result,
      retentionSummary
    });
    if (hasRetainedActualPayload(result)) {
      return { renderResult: result, attempts };
    }
    if (hasRetainedActualPayload(latestRenderResult)) {
      attempts.push({
        stage: 'fallback-latest-render-result',
        resultPresent: true,
        retentionSummary: buildActualPayloadRetentionSummary(latestRenderResult)
      });
      return { renderResult: latestRenderResult, attempts };
    }
    await delayMs(32);
  }

  return {
    renderResult: hasRetainedActualPayload(latestRenderResult) ? latestRenderResult : null,
    attempts
  };
}

let tileLiveDebugCaptureApi = null;

function getTileLiveDebugCaptureApi() {
  if (!tileLiveDebugCaptureApi) {
    tileLiveDebugCaptureApi = createGpuTileLiveDebugCapture({
      getRaw: () => raw,
      getCanvas: () => canvas,
      getCamera: () => camera,
      getUi: () => ui,
      getGpu,
      getLatestRenderResult: () => latestRenderResult,
      hasRetainedActualPayload,
      buildActualPayloadRetentionSummary,
      buildRenderResultInspectionSummary,
      buildDeterministicStateSummary,
      buildSlimDeterministicStateSummary,
      renderCurrentFrameForDebugPayload,
      computeGaussianState,
      computeScreenSplat,
      evalSHColor,
      downloadJsonDebug
    });
  }
  return tileLiveDebugCaptureApi;
}

async function captureTileAccumulationDebug(input = {}) {
  return await getTileLiveDebugCaptureApi().captureTileAccumulationDebug(input);
}

async function captureViewerPayloadIndexAssociationDebug(input = {}) {
  return await getTileLiveDebugCaptureApi().captureViewerPayloadIndexAssociationDebug(input);
}

async function captureLiveSameStateTileAndAssociationDebug(input = {}) {
  return await getTileLiveDebugCaptureApi().captureLiveSameStateTileAndAssociationDebug(input);
}

async function downloadLiveSameStateTileAndAssociationDebugJson(input = {}, fileNames = {}) {
  return await getTileLiveDebugCaptureApi().downloadLiveSameStateTileAndAssociationDebugJson(input, fileNames);
}

async function captureVisibleComparisonDebug(options = {}) {
  const ensureCurrentFrame = options.ensureCurrentFrame !== false;
  const debugRender = ensureCurrentFrame || !latestRenderResult
    ? await renderCurrentFrameForDebugPayload(options)
    : {
        renderResult: latestRenderResult,
        attempts: [{ stage: 'reuse-latest-render-result' }]
      };
  const renderResult = debugRender.renderResult;
  const referenceItems = Array.isArray(options.referenceItems)
    ? options.referenceItems
    : (Array.isArray(renderResult?.visible) ? renderResult.visible : []);
  const candidateItems = Array.isArray(options.candidateItems)
    ? options.candidateItems
    : (Array.isArray(renderResult?.visible) ? renderResult.visible : []);
  const referencePackedPayload =
    options.referencePackedPayload ??
    options.referencePackedScreenSpace ??
    renderResult?.packedScreenSpace ??
    null;
  const candidatePackedPayload =
    options.candidatePackedPayload ??
    options.candidatePackedScreenSpace ??
    renderResult?.packedScreenSpace ??
    null;
  return buildVisibleComparisonSummary({
    referenceItems,
    candidateItems,
    referencePackedPayload,
    candidatePackedPayload,
    referenceLabel: options.referenceLabel ?? 'cpu-visible-reference',
    candidateLabel: options.candidateLabel ?? 'cpu-visible-self',
    options: {
      epsilon: options.epsilon,
      maxMismatches: options.maxMismatches
    },
    metadata: {
      comparisonMode: options.comparisonMode ?? 'latest-render-result-self-compare',
      renderAttempts: debugRender.attempts,
      deterministicState: buildSlimDeterministicStateSummary(buildDeterministicStateSummary()),
      lastRenderResultSummary: buildRenderResultInspectionSummary(renderResult),
      buildStats: renderResult?.buildStats ?? null,
      packedSummary: renderResult?.packedSummary ?? null
    }
  });
}

function buildCandidateComparisonArgs(options = {}) {
  const buildConfig = getVisibleBuildConfig(ui, buildRenderOverrides());
  return {
    raw,
    stride: Number.isFinite(options.stride) ? options.stride : buildConfig.stride,
    temporalPrefilterMode: options.temporalPrefilterMode ?? buildConfig.temporalPrefilterMode,
    useTemporalIndex: typeof options.useTemporalIndex === 'boolean'
      ? options.useTemporalIndex
      : buildConfig.useTemporalIndex,
    useTemporalIndexCache: typeof options.useTemporalIndexCache === 'boolean'
      ? options.useTemporalIndexCache
      : buildConfig.useTemporalIndexCache,
    temporalWindowMode: options.temporalWindowMode ?? buildConfig.temporalWindowMode,
    fixedWindowRadius: Number.isFinite(options.fixedWindowRadius)
      ? options.fixedWindowRadius
      : buildConfig.fixedWindowRadius,
    useTemporalBucket: typeof options.useTemporalBucket === 'boolean'
      ? options.useTemporalBucket
      : buildConfig.useTemporalBucket,
    useTemporalBucketCache: typeof options.useTemporalBucketCache === 'boolean'
      ? options.useTemporalBucketCache
      : buildConfig.useTemporalBucketCache,
    temporalBucketWidth: Number.isFinite(options.temporalBucketWidth)
      ? options.temporalBucketWidth
      : buildConfig.temporalBucketWidth,
    temporalBucketRadius: Number.isFinite(options.temporalBucketRadius)
      ? options.temporalBucketRadius
      : buildConfig.temporalBucketRadius,
    timestamp: Number.isFinite(options.timestamp) ? options.timestamp : buildConfig.timestamp,
    sigmaScale: Number.isFinite(options.sigmaScale) ? options.sigmaScale : buildConfig.sigmaScale,
    temporalSigmaThreshold: Number.isFinite(options.temporalSigmaThreshold)
      ? options.temporalSigmaThreshold
      : 3.0
  };
}

async function captureCandidateComparisonDebug(options = {}) {
  const candidateArgs = buildCandidateComparisonArgs(options);
  const referenceCandidateInfo = buildCandidateInfo(candidateArgs);
  const candidateCandidateInfo = buildGpuStubCandidateInfo({
    raw,
    referenceCandidateInfo,
    reason: options.reason ?? 'step94-3-gpu-candidate-stub'
  });
  return buildCandidateComparisonSummary({
    referenceCandidateInfo,
    candidateCandidateInfo,
    referenceLabel: options.referenceLabel ?? 'cpu-candidate-reference',
    candidateLabel: options.candidateLabel ?? 'gpu-candidate-stub',
    options: {
      maxMismatches: options.maxMismatches
    },
    metadata: {
      comparisonMode: options.comparisonMode ?? 'cpu-candidate-vs-gpu-stub',
      deterministicState: buildSlimDeterministicStateSummary(buildDeterministicStateSummary()),
      candidateArgs: {
        stride: candidateArgs.stride,
        temporalPrefilterMode: candidateArgs.temporalPrefilterMode,
        useTemporalIndex: candidateArgs.useTemporalIndex,
        useTemporalBucket: candidateArgs.useTemporalBucket,
        timestamp: candidateArgs.timestamp,
        sigmaScale: candidateArgs.sigmaScale,
        temporalSigmaThreshold: candidateArgs.temporalSigmaThreshold,
        temporalWindowMode: candidateArgs.temporalWindowMode,
        fixedWindowRadius: candidateArgs.fixedWindowRadius,
        temporalBucketWidth: candidateArgs.temporalBucketWidth,
        temporalBucketRadius: candidateArgs.temporalBucketRadius
      }
    }
  });
}

async function captureCandidateSubsetComparisonDebug(options = {}) {
  const candidateArgs = buildCandidateComparisonArgs(options);
  const referenceCandidateInfo = buildCandidateInfo(candidateArgs);
  const referenceSubsetCandidateInfo = buildCandidateSubsetInfo({
    raw,
    referenceCandidateInfo,
    subsetMode: options.subsetMode ?? 'firstN',
    subsetCount: Number.isFinite(options.subsetCount) ? options.subsetCount : 1024,
    explicitIndices: options.indices ?? options.explicitIndices ?? null,
    candidateMode: options.referenceSubsetMode ?? 'cpu-subset'
  });
  const candidateCandidateInfo = buildGpuSubsetCandidateInfo({
    raw,
    referenceSubsetCandidateInfo,
    reason: options.reason ?? 'step94-4-gpu-candidate-subset-stub'
  });
  return buildCandidateComparisonSummary({
    referenceCandidateInfo: referenceSubsetCandidateInfo,
    candidateCandidateInfo,
    referenceLabel: options.referenceLabel ?? 'cpu-candidate-subset-reference',
    candidateLabel: options.candidateLabel ?? 'gpu-candidate-subset-stub',
    options: {
      maxMismatches: options.maxMismatches
    },
    metadata: {
      comparisonMode: options.comparisonMode ?? 'cpu-candidate-subset-vs-gpu-subset-stub',
      deterministicState: buildSlimDeterministicStateSummary(buildDeterministicStateSummary()),
      fullReferenceCandidateMode: referenceCandidateInfo.candidateMode ?? 'unknown',
      fullReferenceCandidateCount: referenceCandidateInfo.candidateIndices?.length ?? 0,
      subset: referenceSubsetCandidateInfo.candidateSubsetSummary ?? null,
      candidateArgs: {
        stride: candidateArgs.stride,
        temporalPrefilterMode: candidateArgs.temporalPrefilterMode,
        useTemporalIndex: candidateArgs.useTemporalIndex,
        useTemporalBucket: candidateArgs.useTemporalBucket,
        timestamp: candidateArgs.timestamp,
        sigmaScale: candidateArgs.sigmaScale,
        temporalSigmaThreshold: candidateArgs.temporalSigmaThreshold,
        temporalWindowMode: candidateArgs.temporalWindowMode,
        fixedWindowRadius: candidateArgs.fixedWindowRadius,
        temporalBucketWidth: candidateArgs.temporalBucketWidth,
        temporalBucketRadius: candidateArgs.temporalBucketRadius
      }
    }
  });
}

async function captureGpuCandidateSubsetComparisonDebug(options = {}) {
  const subsetCount = Number.isFinite(options.subsetCount) ? options.subsetCount : 1024;
  const startIndex = Number.isFinite(options.startIndex) ? options.startIndex : 0;
  const candidateArgs = buildCandidateComparisonArgs(options);
  const referenceCandidateInfo = buildCandidateInfo(candidateArgs);
  const referenceSubsetCandidateInfo = buildCandidateSubsetInfo({
    raw,
    referenceCandidateInfo,
    subsetMode: 'firstN',
    subsetCount,
    explicitIndices: null,
    candidateMode: options.referenceSubsetMode ?? 'cpu-firstn-subset'
  });
  ensureGpu();
  const candidateCandidateInfo = buildGpuFirstNCandidateInfo({
    gl: getGpu()?.gl,
    raw,
    referenceSubsetCandidateInfo,
    subsetCount,
    startIndex,
    filterMode: options.filterMode ?? 'all-valid'
  });
  return buildCandidateComparisonSummary({
    referenceCandidateInfo: referenceSubsetCandidateInfo,
    candidateCandidateInfo,
    referenceLabel: options.referenceLabel ?? 'cpu-firstn-candidate-reference',
    candidateLabel: options.candidateLabel ?? 'gpu-firstn-candidate-debug',
    options: {
      maxMismatches: options.maxMismatches
    },
    metadata: {
      comparisonMode: options.comparisonMode ?? 'cpu-firstn-subset-vs-gpu-firstn-debug',
      deterministicState: buildSlimDeterministicStateSummary(buildDeterministicStateSummary()),
      fullReferenceCandidateMode: referenceCandidateInfo.candidateMode ?? 'unknown',
      fullReferenceCandidateCount: referenceCandidateInfo.candidateIndices?.length ?? 0,
      subset: referenceSubsetCandidateInfo.candidateSubsetSummary ?? null,
      gpuCandidateSummary: candidateCandidateInfo.gpuCandidateSummary ?? null,
      filterSummary: candidateCandidateInfo.filterSummary ?? null,
      candidateArgs: {
        stride: candidateArgs.stride,
        temporalPrefilterMode: candidateArgs.temporalPrefilterMode,
        useTemporalIndex: candidateArgs.useTemporalIndex,
        useTemporalBucket: candidateArgs.useTemporalBucket,
        timestamp: candidateArgs.timestamp,
        sigmaScale: candidateArgs.sigmaScale,
        temporalSigmaThreshold: candidateArgs.temporalSigmaThreshold,
        temporalWindowMode: candidateArgs.temporalWindowMode,
        fixedWindowRadius: candidateArgs.fixedWindowRadius,
        temporalBucketWidth: candidateArgs.temporalBucketWidth,
        temporalBucketRadius: candidateArgs.temporalBucketRadius
      }
    }
  });
}

async function captureGpuCandidateFilterComparisonDebug(options = {}) {
  const subsetCount = Number.isFinite(options.subsetCount) ? options.subsetCount : 1024;
  const startIndex = Number.isFinite(options.startIndex) ? options.startIndex : 0;
  const filterMode = options.filterMode ?? 'evenIndex';
  const candidateArgs = buildCandidateComparisonArgs(options);
  const referenceCandidateInfo = buildCandidateInfo(candidateArgs);
  const referenceSubsetCandidateInfo = buildCandidateSubsetInfo({
    raw,
    referenceCandidateInfo,
    subsetMode: 'firstN',
    subsetCount,
    explicitIndices: null,
    candidateMode: options.referenceSubsetMode ?? 'cpu-firstn-subset'
  });
  const referenceFilteredCandidateInfo = buildCpuFilteredCandidateInfo({
    raw,
    referenceSubsetCandidateInfo,
    filterMode,
    candidateMode: options.referenceFilterMode ?? 'cpu-firstn-filter-reference'
  });
  ensureGpu();
  const candidateCandidateInfo = buildGpuFirstNCandidateInfo({
    gl: getGpu()?.gl,
    raw,
    referenceSubsetCandidateInfo,
    subsetCount,
    startIndex,
    filterMode
  });
  return buildCandidateComparisonSummary({
    referenceCandidateInfo: referenceFilteredCandidateInfo,
    candidateCandidateInfo,
    referenceLabel: options.referenceLabel ?? 'cpu-firstn-candidate-filter-reference',
    candidateLabel: options.candidateLabel ?? 'gpu-firstn-candidate-filter-debug',
    options: {
      maxMismatches: options.maxMismatches
    },
    metadata: {
      comparisonMode: options.comparisonMode ?? 'cpu-firstn-filter-vs-gpu-firstn-filter-debug',
      deterministicState: buildSlimDeterministicStateSummary(buildDeterministicStateSummary()),
      fullReferenceCandidateMode: referenceCandidateInfo.candidateMode ?? 'unknown',
      fullReferenceCandidateCount: referenceCandidateInfo.candidateIndices?.length ?? 0,
      subset: referenceSubsetCandidateInfo.candidateSubsetSummary ?? null,
      cpuFilterSummary: referenceFilteredCandidateInfo.filterSummary ?? null,
      gpuCandidateSummary: candidateCandidateInfo.gpuCandidateSummary ?? null,
      filterSummary: candidateCandidateInfo.filterSummary ?? null,
      candidateArgs: {
        stride: candidateArgs.stride,
        temporalPrefilterMode: candidateArgs.temporalPrefilterMode,
        useTemporalIndex: candidateArgs.useTemporalIndex,
        useTemporalBucket: candidateArgs.useTemporalBucket,
        timestamp: candidateArgs.timestamp,
        sigmaScale: candidateArgs.sigmaScale,
        temporalSigmaThreshold: candidateArgs.temporalSigmaThreshold,
        temporalWindowMode: candidateArgs.temporalWindowMode,
        fixedWindowRadius: candidateArgs.fixedWindowRadius,
        temporalBucketWidth: candidateArgs.temporalBucketWidth,
        temporalBucketRadius: candidateArgs.temporalBucketRadius
      }
    }
  });
}

async function captureGpuCandidateDryRunVisibleComparisonDebug(options = {}) {
  const ensureCurrentFrame = options.ensureCurrentFrame !== false;
  const debugRender = ensureCurrentFrame || !latestRenderResult
    ? await renderCurrentFrameForDebugPayload(options)
    : {
        renderResult: latestRenderResult,
        attempts: [{ stage: 'reuse-latest-render-result' }]
      };
  ensureGpu();
  camera.updateMatrixWorld(true);
  const deterministicState = buildDeterministicStateSummary();
  const buildConfig = getVisibleBuildConfig(ui, buildRenderOverrides());
  const candidateArgs = buildCandidateComparisonArgs(options);
  const tileGrid = computeTileGrid(canvas.width, canvas.height, 32);
  const screenSpaceCamera = buildScreenSpaceCameraProxy(camera, deterministicState);
  return captureGpuCandidateDryRunVisibleComparison({
    gl: getGpu()?.gl,
    raw,
    camera,
    screenSpaceCamera,
    canvasWidth: canvas.width,
    canvasHeight: canvas.height,
    camPos: camera.position.clone(),
    tileGrid,
    buildConfig,
    candidateArgs,
    subsetCount: Number.isFinite(options.subsetCount) ? options.subsetCount : 1024,
    subsetMode: options.subsetMode ?? 'visibleSrcIndices',
    startIndex: Number.isFinite(options.startIndex) ? options.startIndex : 0,
    filterMode: options.filterMode ?? 'all-valid',
    maxMismatches: options.maxMismatches,
    epsilon: options.epsilon,
    visibleSourceItems: Array.isArray(debugRender.renderResult?.visible) ? debugRender.renderResult.visible : [],
    metadata: {
      comparisonMode: options.comparisonMode ?? 'gpu-candidate-dry-run-visible-comparison',
      deterministicState: buildSlimDeterministicStateSummary(deterministicState),
      renderAttempts: debugRender.attempts,
      lastRenderResultSummary: buildRenderResultInspectionSummary(debugRender.renderResult)
    }
  });
}

async function captureGpuCandidateScreenCoarseDryRunVisibleComparisonDebug(options = {}) {
  const ensureCurrentFrame = options.ensureCurrentFrame !== false;
  const debugRender = ensureCurrentFrame || !latestRenderResult
    ? await renderCurrentFrameForDebugPayload(options)
    : {
        renderResult: latestRenderResult,
        attempts: [{ stage: 'reuse-latest-render-result' }]
      };
  ensureGpu();
  camera.updateMatrixWorld(true);
  const deterministicState = buildDeterministicStateSummary();
  const buildConfig = getVisibleBuildConfig(ui, buildRenderOverrides());
  const candidateArgs = buildCandidateComparisonArgs(options);
  const tileGrid = computeTileGrid(canvas.width, canvas.height, 32);
  const screenSpaceCamera = buildScreenSpaceCameraProxy(camera, deterministicState);
  return captureGpuCandidateScreenCoarseDryRunVisibleComparison({
    gl: getGpu()?.gl,
    raw,
    camera,
    screenSpaceCamera,
    canvasWidth: canvas.width,
    canvasHeight: canvas.height,
    camPos: camera.position.clone(),
    tileGrid,
    buildConfig,
    candidateArgs,
    referenceVisibleItems: Array.isArray(debugRender.renderResult?.visible) ? debugRender.renderResult.visible : [],
    referencePackedPayload: debugRender.renderResult?.packedScreenSpace ?? null,
    maxMismatches: options.maxMismatches,
    maxMissingSamples: options.maxMisses,
    epsilon: options.epsilon,
    filterMode: options.filterMode ?? 'all-valid',
    readbackMode: options.readbackMode ?? 'sync-debug',
    maxCount: Number.isFinite(options.maxCount) ? options.maxCount : 65536,
    minRadiusPx: Number.isFinite(options.minRadiusPx) ? options.minRadiusPx : 0.25,
    requireInViewport: typeof options.requireInViewport === 'boolean' ? options.requireInViewport : true,
    depthMode: options.depthMode ?? 'positive',
    metadata: {
      comparisonMode: options.comparisonMode ?? 'gpu-screen-coarse-candidate-dry-run-visible-comparison',
      deterministicState: buildSlimDeterministicStateSummary(deterministicState),
      renderAttempts: debugRender.attempts,
      lastRenderResultSummary: buildRenderResultInspectionSummary(debugRender.renderResult)
    }
  });
}

async function captureGpuCandidateScreenCoarseSweepComparisonDebug(options = {}) {
  const ensureCurrentFrame = options.ensureCurrentFrame !== false;
  const debugRender = ensureCurrentFrame || !latestRenderResult
    ? await renderCurrentFrameForDebugPayload(options)
    : {
        renderResult: latestRenderResult,
        attempts: [{ stage: 'reuse-latest-render-result' }]
      };
  ensureGpu();
  camera.updateMatrixWorld(true);
  const deterministicState = buildDeterministicStateSummary();
  const buildConfig = getVisibleBuildConfig(ui, buildRenderOverrides());
  const candidateArgs = buildCandidateComparisonArgs(options);
  const tileGrid = computeTileGrid(canvas.width, canvas.height, 32);
  const screenSpaceCamera = buildScreenSpaceCameraProxy(camera, deterministicState);
  return captureGpuCandidateScreenCoarseSweepComparison({
    gl: getGpu()?.gl,
    raw,
    camera,
    screenSpaceCamera,
    canvasWidth: canvas.width,
    canvasHeight: canvas.height,
    camPos: camera.position.clone(),
    tileGrid,
    buildConfig,
    candidateArgs,
    referenceVisibleItems: Array.isArray(debugRender.renderResult?.visible) ? debugRender.renderResult.visible : [],
    referencePackedPayload: debugRender.renderResult?.packedScreenSpace ?? null,
    cases: Array.isArray(options.cases) ? options.cases : null,
    maxCounts: Array.isArray(options.maxCounts) ? options.maxCounts : null,
    minRadiusPxValues: Array.isArray(options.minRadiusPxValues) ? options.minRadiusPxValues : null,
    requireInViewportValues: Array.isArray(options.requireInViewportValues) ? options.requireInViewportValues : null,
    depthModes: Array.isArray(options.depthModes) ? options.depthModes : null,
    maxMismatches: options.maxMismatches,
    maxMissingSamples: options.maxMisses,
    epsilon: options.epsilon,
    filterMode: options.filterMode ?? 'all-valid',
    readbackMode: options.readbackMode ?? 'sync-debug',
    includeCaseResults: options.includeCaseResults === true,
    metadata: {
      comparisonMode: options.comparisonMode ?? 'gpu-screen-coarse-candidate-sweep-comparison',
      deterministicState: buildSlimDeterministicStateSummary(deterministicState),
      renderAttempts: debugRender.attempts,
      lastRenderResultSummary: buildRenderResultInspectionSummary(debugRender.renderResult)
    }
  });
}

async function captureGpuVisibleRecordDryRunDebug(options = {}) {
  const ensureCurrentFrame = options.ensureCurrentFrame !== false;
  const debugRender = ensureCurrentFrame || !latestRenderResult
    ? await renderCurrentFrameForDebugPayload(options)
    : {
        renderResult: latestRenderResult,
        attempts: [{ stage: 'reuse-latest-render-result' }]
      };
  const existingSummary = debugRender.renderResult?.limitedDrawRuntimeSummary?.gpuVisibleRecordDryRunSummary;
  if (existingSummary && options.forceRebuild !== true) {
    return {
      ...existingSummary,
      metadata: {
        ...(existingSummary.metadata ?? {}),
        renderAttempts: debugRender.attempts,
        captureSource: 'latest-render-result'
      }
    };
  }
  ensureGpu();
  camera.updateMatrixWorld(true);
  const deterministicState = buildDeterministicStateSummary();
  const buildConfig = getVisibleBuildConfig(ui, buildRenderOverrides());
  const candidateArgs = buildCandidateComparisonArgs(options);
  const tileGrid = computeTileGrid(canvas.width, canvas.height, 32);
  const screenSpaceCamera = buildScreenSpaceCameraProxy(camera, deterministicState);
  const runtimeSummary = debugRender.renderResult?.limitedDrawRuntimeSummary ?? {};
  const candidateInfo = runtimeSummary?.candidateSourceComparison?.gpuCandidateInfo ?? null;
  return runGpuVisibleRecordDryRun({
    gl: getGpu()?.gl,
    candidateInfo,
    raw,
    camera,
    screenSpaceCamera,
    canvasWidth: canvas.width,
    canvasHeight: canvas.height,
    camPos: camera.position.clone(),
    tileGrid,
    buildConfig,
    temporalSigmaThreshold: candidateArgs.temporalSigmaThreshold ?? 3.0,
    referenceVisibleItems: Array.isArray(debugRender.renderResult?.visible) ? debugRender.renderResult.visible : [],
    maxRecords: Number.isFinite(options.maxRecords) ? options.maxRecords : 65536,
    epsilon: Number.isFinite(options.epsilon) ? options.epsilon : 1e-6,
    maxMismatches: Number.isFinite(options.maxMismatches) ? options.maxMismatches : 32,
    sourceMode: options.sourceMode ?? 'screenCoarse',
    readbackMode: options.readbackMode ?? 'sync-debug',
    displayCandidateSource: runtimeSummary.displayCandidateSource ?? 'cpu-reference',
    gpuCandidateUsedForDisplay: !!runtimeSummary.gpuCandidateUsedForDisplay,
    limitedDrawUsedForCandidateSource: !!runtimeSummary.limitedDrawUsedForCandidateSource,
    metadata: {
      comparisonMode: options.comparisonMode ?? 'gpu-visible-fixed-record-dry-run-vs-cpu-visible-record',
      deterministicState: buildSlimDeterministicStateSummary(deterministicState),
      renderAttempts: debugRender.attempts,
      lastRenderResultSummary: buildRenderResultInspectionSummary(debugRender.renderResult),
      captureSource: 'forced-rebuild'
    }
  });
}

async function captureGpuRawVisibleRecordDryRunDebug(options = {}) {
  const ensureCurrentFrame = options.ensureCurrentFrame !== false;
  const debugRender = ensureCurrentFrame || !latestRenderResult
    ? await renderCurrentFrameForDebugPayload(options)
    : {
        renderResult: latestRenderResult,
        attempts: [{ stage: 'reuse-latest-render-result' }]
      };
  const existingSummary = debugRender.renderResult?.limitedDrawRuntimeSummary?.gpuRawVisibleRecordDryRunSummary;
  if (existingSummary && options.forceRebuild !== true) {
    return {
      ...existingSummary,
      metadata: {
        ...(existingSummary.metadata ?? {}),
        renderAttempts: debugRender.attempts,
        captureSource: 'latest-render-result'
      }
    };
  }
  ensureGpu();
  camera.updateMatrixWorld(true);
  const deterministicState = buildDeterministicStateSummary();
  const buildConfig = getVisibleBuildConfig(ui, buildRenderOverrides());
  const candidateArgs = buildCandidateComparisonArgs(options);
  const tileGrid = computeTileGrid(canvas.width, canvas.height, 32);
  const screenSpaceCamera = buildScreenSpaceCameraProxy(camera, deterministicState);
  const runtimeSummary = debugRender.renderResult?.limitedDrawRuntimeSummary ?? {};
  const candidateInfo = runtimeSummary?.candidateSourceComparison?.gpuCandidateInfo ?? null;
  return runGpuRawVisibleRecordDryRun({
    gl: getGpu()?.gl,
    candidateInfo,
    raw,
    camera,
    screenSpaceCamera,
    canvasWidth: canvas.width,
    canvasHeight: canvas.height,
    camPos: camera.position.clone(),
    tileGrid,
    buildConfig,
    temporalSigmaThreshold: candidateArgs.temporalSigmaThreshold ?? 3.0,
    maxRecords: Number.isFinite(options.maxRecords) ? options.maxRecords : 65536,
    epsilon: Number.isFinite(options.epsilon) ? options.epsilon : 1e-3,
    maxMismatches: Number.isFinite(options.maxMismatches) ? options.maxMismatches : 32,
    readbackMode: options.readbackMode ?? 'sync-debug',
    recordMode: options.recordMode ?? options.rawVisibleRecordMode ?? 'packed-like',
    displayCandidateSource: runtimeSummary.displayCandidateSource ?? 'cpu-reference',
    gpuCandidateUsedForDisplay: !!runtimeSummary.gpuCandidateUsedForDisplay,
    limitedDrawUsedForCandidateSource: !!runtimeSummary.limitedDrawUsedForCandidateSource,
    metadata: {
      comparisonMode: options.comparisonMode ?? 'gpu-raw-attribute-texture-packed-like-fixed-record-vs-cpu-packed-like-record',
      rawVisibleRecordMode: options.recordMode ?? options.rawVisibleRecordMode ?? 'packed-like',
      deterministicState: buildSlimDeterministicStateSummary(deterministicState),
      renderAttempts: debugRender.attempts,
      lastRenderResultSummary: buildRenderResultInspectionSummary(debugRender.renderResult),
      captureSource: 'forced-rebuild'
    }
  });
}

function buildScreenCoarseRuntimeConfigForDebug(options = {}) {
  return {
    sourceMode: 'screenCoarse',
    promotePolicy: 'never',
    readbackMode: options.readbackMode ?? 'sync-debug',
    screenCoarseMaxCount: Number.isFinite(options.maxCount) ? options.maxCount : 65536,
    screenCoarseMinRadiusPx: Number.isFinite(options.minRadiusPx) ? options.minRadiusPx : 0.25,
    screenCoarseRequireInViewport: typeof options.requireInViewport === 'boolean' ? options.requireInViewport : true,
    screenCoarseDepthMode: options.depthMode ?? 'positive'
  };
}

function resolveCandidateInfoForWebGpuDryRun({
  existingCandidateInfo = null,
  gl = null,
  raw = null,
  camera = null,
  screenSpaceCamera = null,
  canvasWidth = 0,
  canvasHeight = 0,
  camPos = null,
  tileGrid = null,
  buildConfig = null,
  candidateArgs = null,
  options = {}
} = {}) {
  if (existingCandidateInfo?.candidateIndices?.length > 0) {
    return {
      candidateInfo: existingCandidateInfo,
      source: 'latest-render-result-limited-draw-summary'
    };
  }
  if (!gl || !raw || !camera || !buildConfig || !candidateArgs) {
    if (
      options.allowCpuReferenceCandidateFallback === true &&
      raw &&
      camera &&
      buildConfig &&
      candidateArgs
    ) {
      return {
        candidateInfo: buildCandidateInfo(candidateArgs),
        source: 'cpu-reference-candidate-fallback-for-webgpu-exclusive-lifecycle',
        reason:
          'WebGL2 frame lifecycle was suppressed for guarded webgpu-exclusive mode'
      };
    }
    return {
      candidateInfo: null,
      source: 'unavailable',
      reason: [
        !gl ? 'webgl-context-missing-for-screen-coarse-candidate-rebuild' : null,
        !raw ? 'raw-missing' : null,
        !camera ? 'camera-missing' : null,
        !buildConfig ? 'build-config-missing' : null,
        !candidateArgs ? 'candidate-args-missing' : null
      ].filter(Boolean).join(',')
    };
  }
  const sourceComparison = buildGpuOwnedCandidateSourceComparison({
    gl,
    raw,
    runtimeConfig: buildScreenCoarseRuntimeConfigForDebug(options),
    referenceCandidateInfo: buildCandidateInfo(candidateArgs),
    filterMode: options.filterMode ?? 'all-valid',
    camera,
    screenSpaceCamera,
    canvasWidth,
    canvasHeight,
    camPos,
    tileGrid,
    buildConfig,
    metadata: {
      comparisonMode: 'phase3-step2-webgpu-dry-run-screen-coarse-candidate-input-rebuild'
    }
  });
  return {
    candidateInfo: sourceComparison?.gpuCandidateInfo ?? null,
    source: 'rebuilt-screen-coarse-candidate-source',
    sourceComparisonSummary: sourceComparison
      ? {
          status: sourceComparison.status,
          reason: sourceComparison.reason,
          sourceConfig: sourceComparison.sourceConfig,
          gpuCandidateSummary: sourceComparison.gpuCandidateSummary,
          candidateComparison: sourceComparison.candidateComparison
            ? {
                anyMismatch: !!sourceComparison.candidateComparison.anyMismatch,
                mismatchCount: sourceComparison.candidateComparison.mismatchCount ?? null,
                countEqual: sourceComparison.candidateComparison.countEqual ?? null
              }
            : null
        }
      : null
  };
}

async function runWebGpuVisibleRecordDryRunFromViewerState({
  options = {},
  requestedWebGpuBackendMode = 'webgl2-fallback',
  allowViewerCanvasPresentation = false,
  enableViewerLoopHook = false,
  useExclusiveWebGpuFrameLifecycle = false,
  debugRender = { renderResult: null, attempts: [] },
  metadataOverrides = {}
} = {}) {
  camera.updateMatrixWorld(true);
  const deterministicState = buildDeterministicStateSummary();
  const buildConfig = getVisibleBuildConfig(ui, buildRenderOverrides());
  const tileGrid = computeTileGrid(canvas.width, canvas.height, 32);
  const screenSpaceCamera = buildScreenSpaceCameraProxy(camera, deterministicState);
  const runtimeSummary = debugRender.renderResult?.limitedDrawRuntimeSummary ?? {};
  const candidateInput = resolveCandidateInfoForWebGpuDryRun({
    existingCandidateInfo: runtimeSummary?.candidateSourceComparison?.gpuCandidateInfo ?? null,
    gl: getGpu()?.gl,
    raw,
    camera,
    screenSpaceCamera,
    canvasWidth: canvas.width,
    canvasHeight: canvas.height,
    camPos: camera.position.clone(),
    tileGrid,
    buildConfig,
    candidateArgs: buildCandidateComparisonArgs(options),
    options: {
      ...options,
      allowCpuReferenceCandidateFallback: useExclusiveWebGpuFrameLifecycle
    }
  });
  const gpu = getGpu();
  const viewerCanvasContextMode = gpu?.gl
    ? 'webgl2-active'
    : useExclusiveWebGpuFrameLifecycle
      ? 'webgpu-exclusive-lifecycle-requested'
      : 'unknown';
  return runWebGpuVisibleRecordDryRun({
    candidateInfo: candidateInput.candidateInfo,
    raw,
    camera,
    screenSpaceCamera,
    canvasWidth: canvas.width,
    canvasHeight: canvas.height,
    camPos: camera.position.clone(),
    tileGrid,
    buildConfig,
    maxRecords: Number.isFinite(options.maxRecords) ? options.maxRecords : 65536,
    epsilon: Number.isFinite(options.epsilon) ? options.epsilon : 1e-3,
    maxMismatches: Number.isFinite(options.maxMismatches) ? options.maxMismatches : 32,
    viewerCanvasState: {
      provided: true,
      canvas,
      contextMode: viewerCanvasContextMode,
      requestedBackendMode: requestedWebGpuBackendMode,
      allowViewerCanvasPresentation,
      webgl2FrameLifecycleSuppressed: useExclusiveWebGpuFrameLifecycle
    },
    metadata: {
      comparisonMode: options.comparisonMode ?? 'webgpu-storage-buffer-compute-fixed-record-vs-cpu-fixed-record',
      deterministicState: buildSlimDeterministicStateSummary(deterministicState),
      renderAttempts: debugRender.attempts,
      lastRenderResultSummary: buildRenderResultInspectionSummary(debugRender.renderResult),
      captureSource: metadataOverrides.captureSource ?? 'forced-rebuild',
      phase: metadataOverrides.phase ?? 'phase3-step4',
      candidateInputSource: candidateInput.source,
      candidateInputReason: candidateInput.reason ?? null,
      candidateInputSummary: candidateInput.sourceComparisonSummary ?? null,
      viewerLifecycleIntegrationRequest: {
        webgpuBackendViewerLoopHook: enableViewerLoopHook,
        requestedBackendMode: requestedWebGpuBackendMode,
        allowViewerCanvasPresentation,
        renderLifecycleStage:
          metadataOverrides.renderLifecycleStage ??
          'captureWebGpuVisibleRecordDryRunDebug',
        invocationSource:
          metadataOverrides.invocationSource ??
          options.webgpuBackendViewerLifecycleInvocationSource ??
          'captureWebGpuVisibleRecordDryRunDebug',
        controlledExecutionRequested:
          metadataOverrides.controlledExecutionRequested === true ||
          options.webgpuBackendViewerLifecycleControlledExecution === true,
        backendExecutorRequest:
          metadataOverrides.backendExecutorRequest ?? null,
        lastRenderLifecycleIntegrationBoundary:
          debugRender.renderResult?.webgpuBackendViewerLifecycleIntegrationBoundary ?? null,
        lastRenderLifecycleControlledExecution:
          debugRender.renderResult?.webgpuBackendViewerLifecycleControlledExecution ?? null,
        lastRenderBackendFrameExecutor:
          debugRender.renderResult?.webgpuBackendViewerFrameExecutor ?? null
      }
    }
  });
}

async function captureWebGpuVisibleRecordDryRunDebug(options = {}) {
  const requestedWebGpuBackendMode =
    options.webgpuBackendMode ??
    deterministicQueryState.webgpuBackendMode ??
    'webgl2-fallback';
  const allowViewerCanvasPresentation =
    options.webgpuAllowViewerCanvasPresentation === true ||
    deterministicQueryState.webgpuAllowViewerCanvasPresentation === true;
  const enableViewerLoopHook =
    options.webgpuBackendViewerLoopHook === true ||
    deterministicQueryState.webgpuBackendViewerLoopHook === true;
  const useExclusiveWebGpuFrameLifecycle = shouldUseWebGpuExclusiveFrameLifecycle({
    requestedBackendMode: requestedWebGpuBackendMode,
    allowViewerCanvasPresentation
  });
  const ensureCurrentFrame =
    options.ensureCurrentFrame !== false && !useExclusiveWebGpuFrameLifecycle;
  const debugRender = useExclusiveWebGpuFrameLifecycle
    ? {
        renderResult: latestRenderResult,
        attempts: [
          {
            stage: 'skip-webgl2-render-current-frame',
            reason:
              'guarded webgpu-exclusive mode suppresses WebGL2 frame lifecycle before WebGPU viewer canvas ownership'
          }
        ]
      }
    : ensureCurrentFrame || !latestRenderResult
      ? await renderCurrentFrameForDebugPayload(options)
      : {
          renderResult: latestRenderResult,
          attempts: [{ stage: 'reuse-latest-render-result' }]
        };
  return runWebGpuVisibleRecordDryRunFromViewerState({
    options,
    requestedWebGpuBackendMode,
    allowViewerCanvasPresentation,
    enableViewerLoopHook,
    useExclusiveWebGpuFrameLifecycle,
    debugRender
  });
}

async function runGpuCandidateShadowCompareForRender(renderResult, options = {}) {
  const shadowOptions = buildGpuCandidateShadowOptionsFromQuery(deterministicQueryState, options);
  if (!options.force && !isGpuCandidateShadowCompareEnabled(shadowOptions)) {
    latestGpuCandidateShadowCompare = null;
    return null;
  }
  ensureGpu();
  camera.updateMatrixWorld(true);
  const deterministicState = buildDeterministicStateSummary();
  const buildConfig = getVisibleBuildConfig(ui, buildRenderOverrides());
  const candidateArgs = buildCandidateComparisonArgs(options);
  const tileGrid = computeTileGrid(canvas.width, canvas.height, 32);
  const screenSpaceCamera = buildScreenSpaceCameraProxy(camera, deterministicState);
  latestGpuCandidateShadowCompare = runGpuCandidateShadowCompare({
    gl: getGpu()?.gl,
    raw,
    camera,
    screenSpaceCamera,
    canvasWidth: canvas.width,
    canvasHeight: canvas.height,
    camPos: camera.position.clone(),
    tileGrid,
    buildConfig,
    candidateArgs,
    renderResult,
    visibleSourceItems: Array.isArray(renderResult?.visible) ? renderResult.visible : [],
    options: {
      ...shadowOptions,
      force: !!options.force
    },
    metadata: {
      comparisonMode: options.comparisonMode ?? 'gpu-candidate-shadow-compare',
      deterministicState: buildSlimDeterministicStateSummary(deterministicState),
      trigger: options.trigger ?? 'manual',
      lastRenderResultSummary: buildRenderResultInspectionSummary(renderResult)
    }
  });
  return latestGpuCandidateShadowCompare;
}

async function captureGpuCandidateShadowCompareDebug(options = {}) {
  const ensureCurrentFrame = options.ensureCurrentFrame !== false;
  const debugRender = ensureCurrentFrame || !latestRenderResult
    ? await renderCurrentFrameForDebugPayload(options)
    : {
        renderResult: latestRenderResult,
        attempts: [{ stage: 'reuse-latest-render-result' }]
      };
  const result = await runGpuCandidateShadowCompareForRender(debugRender.renderResult, {
    ...options,
    force: options.force !== false,
    trigger: options.trigger ?? 'debug-api',
    comparisonMode: options.comparisonMode ?? 'gpu-candidate-shadow-compare-debug-api'
  });
  if (result?.metadata) {
    result.metadata.renderAttempts = debugRender.attempts;
  }
  return result;
}

async function captureGpuCandidateSourceCompareDebug(options = {}) {
  const ensureCurrentFrame = options.ensureCurrentFrame !== false;
  const debugRender = ensureCurrentFrame || !latestRenderResult
    ? await renderCurrentFrameForDebugPayload(options)
    : {
        renderResult: latestRenderResult,
        attempts: [{ stage: 'reuse-latest-render-result' }]
      };
  const limitedDrawSummary = debugRender.renderResult?.limitedDrawRuntimeSummary ?? null;
  return {
    schemaVersion: 'step103-gpu-candidate-source-compare-debug-v1',
    timestamp: new Date().toISOString(),
    status: limitedDrawSummary?.candidateSourceComparison ? 'ok' : 'missing',
    reason: limitedDrawSummary?.candidateSourceComparison ? 'ok' : 'candidate-source-comparison-unavailable',
    candidateSourceSummary: limitedDrawSummary?.candidateSourceSummary ?? null,
    candidateSourceComparison: limitedDrawSummary?.candidateSourceComparison ?? null,
    candidateCoverageSummary: limitedDrawSummary?.candidateCoverageSummary ?? null,
    candidateComparison: limitedDrawSummary?.candidateSourceComparison?.candidateComparison ??
      limitedDrawSummary?.candidateComparison ??
      null,
    displayCandidateSource: limitedDrawSummary?.displayCandidateSource ?? 'cpu-reference',
    gpuCandidateUsedForDisplay: !!limitedDrawSummary?.gpuCandidateUsedForDisplay,
    limitedDrawUsedForCandidateSource: !!limitedDrawSummary?.limitedDrawUsedForCandidateSource,
    deterministicState: buildSlimDeterministicStateSummary(buildDeterministicStateSummary()),
    renderAttempts: debugRender.attempts,
    lastRenderResultSummary: buildRenderResultInspectionSummary(debugRender.renderResult)
  };
}

async function captureGpuCandidateScreenCoarseCompareDebug(options = {}) {
  const result = await captureGpuCandidateSourceCompareDebug(options);
  return {
    ...result,
    schemaVersion: 'step107-gpu-candidate-screen-coarse-compare-debug-v1',
    sourceMode: result?.candidateSourceComparison?.sourceConfig?.sourceMode ??
      result?.candidateSourceSummary?.sourceMode ??
      null
  };
}

async function captureGpuCandidateCoverageDebug(options = {}) {
  const ensureCurrentFrame = options.ensureCurrentFrame !== false;
  const debugRender = ensureCurrentFrame || !latestRenderResult
    ? await renderCurrentFrameForDebugPayload(options)
    : {
        renderResult: latestRenderResult,
        attempts: [{ stage: 'reuse-latest-render-result' }]
      };
  const limitedDrawSummary = debugRender.renderResult?.limitedDrawRuntimeSummary ?? null;
  return {
    schemaVersion: 'step105-gpu-candidate-coverage-debug-v1',
    timestamp: new Date().toISOString(),
    status: limitedDrawSummary?.candidateCoverageSummary ? 'ok' : 'missing',
    reason: limitedDrawSummary?.candidateCoverageSummary ? 'ok' : 'candidate-coverage-summary-unavailable',
    candidateCoverageSummary: limitedDrawSummary?.candidateCoverageSummary ?? null,
    candidateSourceSummary: limitedDrawSummary?.candidateSourceSummary ?? null,
    candidateSourceComparison: limitedDrawSummary?.candidateSourceComparison ?? null,
    displayCandidateSource: limitedDrawSummary?.displayCandidateSource ?? 'cpu-reference',
    gpuCandidateUsedForDisplay: !!limitedDrawSummary?.gpuCandidateUsedForDisplay,
    limitedDrawUsedForCandidateSource: !!limitedDrawSummary?.limitedDrawUsedForCandidateSource,
    deterministicState: buildSlimDeterministicStateSummary(buildDeterministicStateSummary()),
    renderAttempts: debugRender.attempts,
    lastRenderResultSummary: buildRenderResultInspectionSummary(debugRender.renderResult)
  };
}

async function captureRepresentativeActualPayloadDebug(input = {}, maybeOptions = {}) {
  const options = Array.isArray(input) || Array.isArray(input?.selectedIndices) || Array.isArray(input?.splats)
    ? maybeOptions
    : (input ?? {});
  const referenceDebug = options.referenceDebug ?? options.reference ?? (
    Array.isArray(input?.selectedIndices) || Array.isArray(input?.splats) ? input : null
  );
  const indices = normalizeRepresentativeIndexList(input, referenceDebug);
  const ensureCurrentFrame = options.ensureCurrentFrame !== false;
  const debugRender = ensureCurrentFrame || !hasRetainedActualPayload(latestRenderResult)
    ? await renderCurrentFrameForDebugPayload(options)
    : { renderResult: latestRenderResult, attempts: [{ stage: 'reuse-latest-render-result', retentionSummary: buildActualPayloadRetentionSummary(latestRenderResult) }] };
  const renderResult = debugRender.renderResult;
  const referenceByIndex = getRepresentativeReferenceByIndex(referenceDebug);
  const items = indices.map((index) => {
    const referenceEntry = referenceByIndex.get(index) ?? null;
    const referencePayload = buildReferencePayloadSummary(referenceEntry);
    const actual = inspectActualPayloadForOriginalIndex(renderResult, index);
    return {
      index,
      referenceAvailable: !!referencePayload,
      referencePayload,
      found: actual.found,
      missingReason: actual.missingReason,
      actualPayload: actual.actualPayload,
      visiblePayload: actual.visiblePayload,
      packedPayload: actual.packedPayload,
      accumulationPayload: actual.accumulationPayload,
      accumulationOccurrenceCount: actual.accumulationOccurrenceCount,
      accumulationOccurrenceConsistency: actual.accumulationOccurrenceConsistency,
      accumulationOccurrences: options.includeAllAccumulationOccurrences === true
        ? actual.accumulationOccurrences
        : actual.accumulationOccurrences.slice(0, 8),
      compare: buildPayloadComparison(referencePayload, actual.actualPayload)
    };
  });
  const summary = summarizeRepresentativePayloadComparisons(items);
  return {
    schemaVersion: 'step90-representative-actual-payload-compare-v1',
    timestamp: new Date().toISOString(),
    selectedIndices: indices,
    summary,
    actualPayloadSource: {
      preferredPayload: 'tileCompositePlan.batches[].packed when accumulation occurrences exist; otherwise packedScreenSpace.packed; otherwise visible',
      visibleSource: 'latestRenderResult.visible[] from buildVisibleSplats',
      packedSource: 'latestRenderResult.packedScreenSpace.packed/sourceItems',
      accumulationSource: 'latestRenderResult.tileCompositePlan.batches[].packed/sourceIndices before buildTileAccumulationPayload texture upload'
    },
    actualPayloadRetentionSummary: buildActualPayloadRetentionSummary(renderResult),
    debugRenderAttempts: debugRender.attempts,
    deterministicState: buildSlimDeterministicStateSummary(buildDeterministicStateSummary()),
    lastRenderResultSummary: buildRenderResultInspectionSummary(renderResult),
    items
  };
}

async function saveRepresentativeActualPayloadOverlayPng(compareResult, options = {}) {
  const width = Number.isFinite(options.width) ? Math.max(1, options.width | 0) : (canvas?.width ?? 1280);
  const height = Number.isFinite(options.height) ? Math.max(1, options.height | 0) : (canvas?.height ?? 720);
  const overlay = document.createElement('canvas');
  overlay.width = width;
  overlay.height = height;
  const ctx = overlay.getContext('2d');
  ctx.fillStyle = 'rgba(0,0,0,1)';
  ctx.fillRect(0, 0, width, height);
  ctx.font = '11px sans-serif';
  ctx.lineWidth = 1.5;

  const drawPoint = (center, radius, color, label, labelOffsetY = 0) => {
    if (!Array.isArray(center) || center.length < 2 || !center.every(Number.isFinite)) return;
    const r = Number.isFinite(radius) ? Math.max(2, Math.min(80, radius)) : 4;
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.arc(center[0], center[1], r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.fillRect(center[0] - 2, center[1] - 2, 4, 4);
    if (label) ctx.fillText(label, center[0] + 5, center[1] - 5 + labelOffsetY);
  };

  for (const item of compareResult?.items ?? []) {
    drawPoint(
      item.referencePayload?.centerPx,
      item.referencePayload?.radius,
      'rgba(0,200,255,0.9)',
      `${item.index} ref`,
      0
    );
    drawPoint(
      item.actualPayload?.centerPx,
      item.actualPayload?.radius,
      'rgba(255,210,0,0.9)',
      `${item.index} actual`,
      12
    );
  }

  ctx.fillStyle = 'rgba(0,200,255,0.9)';
  ctx.fillText('cyan: viewerCudaAlignedCurrent reference', 12, 18);
  ctx.fillStyle = 'rgba(255,210,0,0.9)';
  ctx.fillText('yellow: actual gpu-screen / accumulation payload', 12, 34);

  const fileName = sanitizeSnapshotFileName(options.name ?? 'step90_representative_actual_payload_overlay.png');
  const blob = await captureBlobFromCanvas(overlay, fileName, options.download !== false);
  return { blob, fileName, width, height, source: 'generated-debug-overlay-canvas' };
}

async function saveTileAccumulationDebugOverlayPng(debugResult, options = {}) {
  const width = Number.isFinite(options.width) ? Math.max(1, options.width | 0) : (canvas?.width ?? 1280);
  const height = Number.isFinite(options.height) ? Math.max(1, options.height | 0) : (canvas?.height ?? 720);
  const overlay = document.createElement('canvas');
  overlay.width = width;
  overlay.height = height;
  const ctx = overlay.getContext('2d');
  ctx.fillStyle = 'rgba(0,0,0,1)';
  ctx.fillRect(0, 0, width, height);
  ctx.font = '12px sans-serif';
  ctx.lineWidth = 1.5;
  for (const result of debugResult?.pixels ?? []) {
    const [x, y] = result.pixel ?? [0, 0];
    ctx.strokeStyle = result.ok ? 'rgba(255,210,0,0.95)' : 'rgba(255,80,80,0.95)';
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.stroke();
    if (Array.isArray(result.tileBounds)) {
      ctx.strokeStyle = 'rgba(0,200,255,0.65)';
      ctx.strokeRect(
        result.tileBounds[0],
        result.tileBounds[1],
        result.tileBounds[2] - result.tileBounds[0],
        result.tileBounds[3] - result.tileBounds[1]
      );
    }
    ctx.fillStyle = 'rgba(255,210,0,0.95)';
    ctx.fillText(
      `${x},${y} tile=${result.tileId ?? -1} contrib=${result.accumulation?.contributionCount ?? 0}`,
      x + 10,
      y - 10
    );
  }
  const fileName = sanitizeSnapshotFileName(options.name ?? 'step90_tile_accumulation_debug_overlay.png');
  const blob = await captureBlobFromCanvas(overlay, fileName, options.download !== false);
  return { blob, fileName, width, height, source: 'generated-tile-accumulation-debug-overlay' };
}

function parseScreenshotProbeList(rawProbeList) {
  if (rawProbeList === null || rawProbeList === undefined || rawProbeList === '') {
    return [];
  }

  return String(rawProbeList)
    .split(';')
    .map((entry, index) => {
      const trimmed = String(entry ?? '').trim();
      if (!trimmed) return null;

      const labelMatch = trimmed.match(/^([^:@]+)[:@](.+)$/);
      const label = labelMatch ? labelMatch[1].trim() : null;
      const coordsText = (labelMatch ? labelMatch[2] : trimmed).trim();
      const coords = coordsText.split(',').map((value) => value.trim()).filter(Boolean);
      if (coords.length < 2) {
        return {
          valid: false,
          reason: 'invalid-screenshot-probe-format',
          probeId: label ?? `probe-${index + 1}`,
          screenshotProbeX: null,
          screenshotProbeY: null,
          screenshotProbePixel: [null, null]
        };
      }

      const screenshotProbeX = Number(coords[0]);
      const screenshotProbeY = Number(coords[1]);
      const probeId = label ?? `probe-${index + 1}`;
      if (!Number.isFinite(screenshotProbeX) || !Number.isFinite(screenshotProbeY)) {
        return {
          valid: false,
          reason: 'invalid-screenshot-probe-coordinate',
          probeId,
          screenshotProbeX: Number.isFinite(screenshotProbeX) ? screenshotProbeX : null,
          screenshotProbeY: Number.isFinite(screenshotProbeY) ? screenshotProbeY : null,
          screenshotProbePixel: [
            Number.isFinite(screenshotProbeX) ? screenshotProbeX : null,
            Number.isFinite(screenshotProbeY) ? screenshotProbeY : null
          ]
        };
      }

      return {
        valid: true,
        reason: 'probe-list-item-ok',
        probeId,
        screenshotProbeLabel: label ?? probeId,
        screenshotProbeX,
        screenshotProbeY,
        screenshotProbePixel: [screenshotProbeX, screenshotProbeY]
      };
    })
    .filter(Boolean);
}

function buildDefaultScreenshotProbeInput() {
  const screenshotProbeX = Number.isFinite(deterministicQueryState.screenshotProbeX)
    ? Number(deterministicQueryState.screenshotProbeX)
    : null;
  const screenshotProbeY = Number.isFinite(deterministicQueryState.screenshotProbeY)
    ? Number(deterministicQueryState.screenshotProbeY)
    : null;
  if (!Number.isFinite(screenshotProbeX) || !Number.isFinite(screenshotProbeY)) {
    return null;
  }
  return {
    valid: true,
    reason: 'single-screenshot-probe',
    probeId: 'probe-1',
    screenshotProbeLabel: 'probe-1',
    screenshotProbeX,
    screenshotProbeY,
    screenshotProbePixel: [screenshotProbeX, screenshotProbeY]
  };
}

function buildScreenshotProbeInputList() {
  const parsed = parseScreenshotProbeList(deterministicQueryState.screenshotProbeList);
  const screenshotImageWidth = Number.isFinite(deterministicQueryState.screenshotImageWidth)
    ? Math.max(1, deterministicQueryState.screenshotImageWidth | 0)
    : null;
  const screenshotImageHeight = Number.isFinite(deterministicQueryState.screenshotImageHeight)
    ? Math.max(1, deterministicQueryState.screenshotImageHeight | 0)
    : null;
  if (parsed.length > 0) {
    return parsed.map((entry) => ({
      ...entry,
      screenshotImageWidth,
      screenshotImageHeight
    }));
  }
  const single = buildDefaultScreenshotProbeInput();
  return single ? [{
    ...single,
    screenshotImageWidth,
    screenshotImageHeight
  }] : [];
}

function buildMappedScreenshotProbeSummaryFromInput(probeInput, canvasSizeSummary = buildCanvasSizeSummary()) {
  const screenshotProbeX = Number.isFinite(probeInput?.screenshotProbeX)
    ? Number(probeInput.screenshotProbeX)
    : null;
  const screenshotProbeY = Number.isFinite(probeInput?.screenshotProbeY)
    ? Number(probeInput.screenshotProbeY)
    : null;
  const screenshotImageWidth = Number.isFinite(probeInput?.screenshotImageWidth)
    ? Math.max(1, probeInput.screenshotImageWidth | 0)
    : (Number.isFinite(deterministicQueryState.screenshotImageWidth)
      ? Math.max(1, deterministicQueryState.screenshotImageWidth | 0)
      : null);
  const screenshotImageHeight = Number.isFinite(probeInput?.screenshotImageHeight)
    ? Math.max(1, probeInput.screenshotImageHeight | 0)
    : (Number.isFinite(deterministicQueryState.screenshotImageHeight)
      ? Math.max(1, deterministicQueryState.screenshotImageHeight | 0)
      : null);
  const targetWidth = Number.isFinite(canvasSizeSummary?.probeCoordinateWidth)
    ? Math.max(1, canvasSizeSummary.probeCoordinateWidth | 0)
    : 0;
  const targetHeight = Number.isFinite(canvasSizeSummary?.probeCoordinateHeight)
    ? Math.max(1, canvasSizeSummary.probeCoordinateHeight | 0)
    : 0;
  const usesFixedCanvas = !!canvasSizeSummary?.fixedCanvasActive;
  const coordinateSpace = canvasSizeSummary?.coordinateSpaceForReadPixels ?? 'webgl-default-framebuffer-pixels';

  const sourceSummary = {
    source: 'screenshot-query',
    probeId: probeInput?.probeId ?? 'probe-1',
    screenshotProbeLabel: probeInput?.screenshotProbeLabel ?? probeInput?.probeId ?? 'probe-1',
    screenshotProbePixel: [screenshotProbeX, screenshotProbeY],
    screenshotImageSize: [screenshotImageWidth, screenshotImageHeight],
    targetFramebufferSize: [targetWidth, targetHeight],
    scale: null,
    rounding: 'round-to-nearest-pixel',
    clamped: false,
    targetCoordinateSpace: coordinateSpace,
    fixedCanvasTarget: usesFixedCanvas ? [canvasSizeSummary.fixedCanvasWidth, canvasSizeSummary.fixedCanvasHeight] : null
  };

  if (!Number.isFinite(screenshotProbeX) ||
      !Number.isFinite(screenshotProbeY)) {
    return {
      screenshotProbeX: screenshotProbeX ?? null,
      screenshotProbeY: screenshotProbeY ?? null,
      screenshotProbePixel: [screenshotProbeX ?? null, screenshotProbeY ?? null],
      screenshotImageWidth: screenshotImageWidth ?? null,
      screenshotImageHeight: screenshotImageHeight ?? null,
      screenshotImageSize: [screenshotImageWidth ?? null, screenshotImageHeight ?? null],
      probeId: probeInput?.probeId ?? 'probe-1',
      screenshotProbeLabel: probeInput?.screenshotProbeLabel ?? probeInput?.probeId ?? 'probe-1',
      mappedProbeX: null,
      mappedProbeY: null,
      mappedProbeValid: false,
      mappedProbeReason: 'missing-screenshot-probe-coordinate',
      mappedProbeCoordinateSpace: coordinateSpace,
      mappedProbeUsesFixedCanvas: usesFixedCanvas,
      mappedProbeSourceSummary: sourceSummary
    };
  }

  if (!Number.isFinite(screenshotImageWidth) || !Number.isFinite(screenshotImageHeight)) {
    return {
      screenshotProbeX,
      screenshotProbeY,
      screenshotProbePixel: [screenshotProbeX, screenshotProbeY],
      screenshotImageWidth: screenshotImageWidth ?? null,
      screenshotImageHeight: screenshotImageHeight ?? null,
      screenshotImageSize: [screenshotImageWidth ?? null, screenshotImageHeight ?? null],
      probeId: probeInput?.probeId ?? 'probe-1',
      screenshotProbeLabel: probeInput?.screenshotProbeLabel ?? probeInput?.probeId ?? 'probe-1',
      mappedProbeX: null,
      mappedProbeY: null,
      mappedProbeValid: false,
      mappedProbeReason: 'missing-screenshot-image-size',
      mappedProbeCoordinateSpace: coordinateSpace,
      mappedProbeUsesFixedCanvas: usesFixedCanvas,
      mappedProbeSourceSummary: sourceSummary
    };
  }

  if (!(targetWidth > 0) || !(targetHeight > 0)) {
    return {
      screenshotProbeX,
      screenshotProbeY,
      screenshotProbePixel: [screenshotProbeX, screenshotProbeY],
      screenshotImageWidth,
      screenshotImageHeight,
      screenshotImageSize: [screenshotImageWidth, screenshotImageHeight],
      probeId: probeInput?.probeId ?? 'probe-1',
      screenshotProbeLabel: probeInput?.screenshotProbeLabel ?? probeInput?.probeId ?? 'probe-1',
      mappedProbeX: null,
      mappedProbeY: null,
      mappedProbeValid: false,
      mappedProbeReason: 'invalid-target-framebuffer-size',
      mappedProbeCoordinateSpace: coordinateSpace,
      mappedProbeUsesFixedCanvas: usesFixedCanvas,
      mappedProbeSourceSummary: sourceSummary
    };
  }

  const scaleX = targetWidth / screenshotImageWidth;
  const scaleY = targetHeight / screenshotImageHeight;
  const rawMappedX = screenshotProbeX * scaleX;
  const rawMappedY = screenshotProbeY * scaleY;
  const mappedProbeX = Math.min(targetWidth - 1, Math.max(0, Math.round(rawMappedX)));
  const mappedProbeY = Math.min(targetHeight - 1, Math.max(0, Math.round(rawMappedY)));
  const clamped = mappedProbeX !== Math.round(rawMappedX) || mappedProbeY !== Math.round(rawMappedY);

  return {
    screenshotProbeX,
    screenshotProbeY,
    screenshotProbePixel: [screenshotProbeX, screenshotProbeY],
    screenshotImageWidth,
    screenshotImageHeight,
    screenshotImageSize: [screenshotImageWidth, screenshotImageHeight],
    probeId: probeInput?.probeId ?? 'probe-1',
    screenshotProbeLabel: probeInput?.screenshotProbeLabel ?? probeInput?.probeId ?? 'probe-1',
    mappedProbeX,
    mappedProbeY,
    mappedProbeValid: true,
    mappedProbeReason: 'mapped-screenshot-to-viewer-framebuffer',
    mappedProbeCoordinateSpace: coordinateSpace,
    mappedProbeUsesFixedCanvas: usesFixedCanvas,
    mappedProbeSourceSummary: {
      ...sourceSummary,
      scale: [scaleX, scaleY],
      clamped
    }
  };
}

function readFramebufferPatchAtTopLeftPixel(gl, centerPixel, patchSize) {
  const normalizedCenter = normalizeSharedRepresentativePixel(centerPixel);
  const size = Number.isFinite(patchSize) ? Math.max(1, patchSize | 0) : 0;
  if (!normalizedCenter) {
    return {
      valid: false,
      reason: 'invalid-mapped-probe-center',
      patchSize: size,
      centerPixel: [0, 0],
      topLeft: [0, 0],
      bottomRight: [0, 0],
      pixelCount: 0,
      centerRgba8: [0, 0, 0, 0],
      meanRgb: [0, 0, 0],
      minRgb: [0, 0, 0],
      maxRgb: [0, 0, 0],
      rawRgba8: []
    };
  }

  const width = Number.isFinite(canvas?.width) ? (canvas.width | 0) : 0;
  const height = Number.isFinite(canvas?.height) ? (canvas.height | 0) : 0;
  if (!(size > 0) || (size % 2) === 0) {
    return {
      valid: false,
      reason: 'invalid-patch-size',
      patchSize: size,
      centerPixel: normalizedCenter,
      topLeft: [0, 0],
      bottomRight: [0, 0],
      pixelCount: 0,
      centerRgba8: [0, 0, 0, 0],
      meanRgb: [0, 0, 0],
      minRgb: [0, 0, 0],
      maxRgb: [0, 0, 0],
      rawRgba8: []
    };
  }
  if (width <= 0 || height <= 0) {
    return {
      valid: false,
      reason: 'invalid-canvas-size',
      patchSize: size,
      centerPixel: normalizedCenter,
      topLeft: [0, 0],
      bottomRight: [0, 0],
      pixelCount: 0,
      centerRgba8: [0, 0, 0, 0],
      meanRgb: [0, 0, 0],
      minRgb: [0, 0, 0],
      maxRgb: [0, 0, 0],
      rawRgba8: []
    };
  }

  const half = (size - 1) >> 1;
  const topLeftX = normalizedCenter[0] - half;
  const topLeftY = normalizedCenter[1] - half;
  const bottomRightX = topLeftX + size - 1;
  const bottomRightY = topLeftY + size - 1;
  if (topLeftX < 0 || topLeftY < 0 || bottomRightX >= width || bottomRightY >= height) {
    return {
      valid: false,
      reason: 'patch-out-of-bounds',
      patchSize: size,
      centerPixel: normalizedCenter,
      topLeft: [topLeftX, topLeftY],
      bottomRight: [bottomRightX, bottomRightY],
      pixelCount: 0,
      centerRgba8: [0, 0, 0, 0],
      meanRgb: [0, 0, 0],
      minRgb: [0, 0, 0],
      maxRgb: [0, 0, 0],
      rawRgba8: []
    };
  }

  const glX = topLeftX;
  const glY = height - topLeftY - size;
  const rgba = new Uint8Array(size * size * 4);
  const previousFramebuffer = gl.getParameter(gl.FRAMEBUFFER_BINDING);
  try {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.readPixels(glX, glY, size, size, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
  } catch (error) {
    return {
      valid: false,
      reason: `readpixels-failed:${error?.message ?? 'unknown'}`,
      patchSize: size,
      centerPixel: normalizedCenter,
      topLeft: [topLeftX, topLeftY],
      bottomRight: [bottomRightX, bottomRightY],
      pixelCount: 0,
      centerRgba8: [0, 0, 0, 0],
      meanRgb: [0, 0, 0],
      minRgb: [0, 0, 0],
      maxRgb: [0, 0, 0],
      rawRgba8: []
    };
  } finally {
    gl.bindFramebuffer(gl.FRAMEBUFFER, previousFramebuffer);
  }

  const rawRgba8 = new Array(size * size * 4);
  let pixelIndex = 0;
  let minR = 1;
  let minG = 1;
  let minB = 1;
  let maxR = 0;
  let maxG = 0;
  let maxB = 0;
  let sumR = 0;
  let sumG = 0;
  let sumB = 0;
  for (let row = 0; row < size; row += 1) {
    const sourceRow = size - 1 - row;
    for (let col = 0; col < size; col += 1) {
      const srcIndex = (sourceRow * size + col) * 4;
      const r8 = rgba[srcIndex];
      const g8 = rgba[srcIndex + 1];
      const b8 = rgba[srcIndex + 2];
      const a8 = rgba[srcIndex + 3];
      rawRgba8[pixelIndex++] = r8;
      rawRgba8[pixelIndex++] = g8;
      rawRgba8[pixelIndex++] = b8;
      rawRgba8[pixelIndex++] = a8;
      const r = r8 / 255.0;
      const g = g8 / 255.0;
      const b = b8 / 255.0;
      sumR += r;
      sumG += g;
      sumB += b;
      minR = Math.min(minR, r);
      minG = Math.min(minG, g);
      minB = Math.min(minB, b);
      maxR = Math.max(maxR, r);
      maxG = Math.max(maxG, g);
      maxB = Math.max(maxB, b);
    }
  }

  const centerIndex = half * size + half;
  const centerBase = centerIndex * 4;
  return {
    valid: true,
    reason: 'readback-ok',
    patchSize: size,
    centerPixel: normalizedCenter,
    topLeft: [topLeftX, topLeftY],
    bottomRight: [bottomRightX, bottomRightY],
    pixelCount: size * size,
    centerRgba8: [
      rgba[centerBase],
      rgba[centerBase + 1],
      rgba[centerBase + 2],
      rgba[centerBase + 3]
    ],
    meanRgb: [
      sumR / (size * size),
      sumG / (size * size),
      sumB / (size * size)
    ],
    minRgb: [minR, minG, minB],
    maxRgb: [maxR, maxG, maxB],
    rawRgba8
  };
}

function getMappedProbePatchStorageKey(centerPixel, patchSize, probeKey = null) {
  const normalizedCenter = normalizeSharedRepresentativePixel(centerPixel);
  const size = Number.isFinite(patchSize) ? Math.max(1, patchSize | 0) : 0;
  if (!normalizedCenter || !(size > 0)) return null;
  const keyPrefix = probeKey ? `${probeKey}.` : '';
  return `${MAPPED_PROBE_PATCH_REFERENCE_STORAGE_PREFIX}${keyPrefix}${normalizedCenter[0]}x${normalizedCenter[1]}.${size}`;
}

function readMappedProbePatchReference(centerPixel, patchSize, probeKey = null) {
  const storageKey = getMappedProbePatchStorageKey(centerPixel, patchSize, probeKey);
  if (!storageKey) return null;
  const stored = readStoredJson(storageKey);
  if (!stored || !Array.isArray(stored.rawRgba8)) return null;
  return {
    ...stored,
    valid: stored.valid !== false,
    referenceSource: stored.referenceSource ?? 'stored-accumulation-patch-reference'
  };
}

function writeMappedProbePatchReference(centerPixel, patchSize, patchResult, probeKey = null) {
  const storageKey = getMappedProbePatchStorageKey(centerPixel, patchSize, probeKey);
  if (!storageKey || !patchResult?.valid || !Array.isArray(patchResult.rawRgba8)) return;
  writeStoredJson(storageKey, {
    valid: true,
    patchSize: patchResult.patchSize,
    centerPixel: patchResult.centerPixel,
    topLeft: patchResult.topLeft,
    bottomRight: patchResult.bottomRight,
    pixelCount: patchResult.pixelCount,
    rawRgba8: patchResult.rawRgba8,
    referenceSource: 'stored-accumulation-patch-reference'
  });
}

function buildPatchDeltaSummary(actualPatch, referencePatch) {
  if (!actualPatch?.valid) {
    return {
      valid: false,
      reason: actualPatch?.reason ?? 'patch-readback-invalid',
      comparedAgainstAccumulation: false,
      referenceSource: 'none',
      maxDelta: 0,
      meanDelta: 0,
      mismatchCountAboveTolerance: 0,
      worstPixelOffset: [0, 0],
      worstPixelDelta: [0, 0, 0]
    };
  }

  if (!referencePatch || !Array.isArray(referencePatch.rawRgba8)) {
    return {
      valid: false,
      reason: 'accumulation-reference-unavailable',
      comparedAgainstAccumulation: false,
      referenceSource: 'none',
      maxDelta: 0,
      meanDelta: 0,
      mismatchCountAboveTolerance: 0,
      worstPixelOffset: [0, 0],
      worstPixelDelta: [0, 0, 0]
    };
  }

  const actual = actualPatch.rawRgba8;
  const reference = referencePatch.rawRgba8;
  const pixelCount = Math.min(actual.length, reference.length) / 4;
  const size = actualPatch.patchSize;
  const half = (size - 1) >> 1;
  let maxDelta = 0;
  let sumDelta = 0;
  let mismatchCountAboveTolerance = 0;
  let worstPixelOffset = [0, 0];
  let worstPixelDelta = [0, 0, 0];

  for (let i = 0; i < pixelCount; i += 1) {
    const base = i * 4;
    const dr = actual[base] / 255.0 - reference[base] / 255.0;
    const dg = actual[base + 1] / 255.0 - reference[base + 1] / 255.0;
    const db = actual[base + 2] / 255.0 - reference[base + 2] / 255.0;
    const deltaAbsMax = Math.max(Math.abs(dr), Math.abs(dg), Math.abs(db));
    sumDelta += deltaAbsMax;
    if (deltaAbsMax > maxDelta) {
      maxDelta = deltaAbsMax;
      const row = Math.floor(i / size);
      const col = i - row * size;
      worstPixelOffset = [col - half, row - half];
      worstPixelDelta = [dr, dg, db];
    }
    if (deltaAbsMax > MAPPED_PROBE_PATCH_TOLERANCE) {
      mismatchCountAboveTolerance += 1;
    }
  }

  return {
    valid: true,
    reason: 'patch-delta-ok',
    comparedAgainstAccumulation: true,
    referenceSource: referencePatch.referenceSource ?? 'stored-accumulation-patch-reference',
    maxDelta,
    meanDelta: pixelCount > 0 ? sumDelta / pixelCount : 0,
    mismatchCountAboveTolerance,
    worstPixelOffset,
    worstPixelDelta
  };
}

function buildMappedProbePatchSummary(renderResultSummary, canvasSizeSummary = buildCanvasSizeSummary(), mappedProbeSummary = buildMappedScreenshotProbeSummary(canvasSizeSummary), probeKey = null) {
  const gl = getGpu()?.gl ?? null;
  const tileCompositePath = renderResultSummary?.drawThroughputSummary?.tileCompositePath ?? 'none';
  const mappedProbeX = Number.isFinite(mappedProbeSummary?.mappedProbeX)
    ? Number(mappedProbeSummary.mappedProbeX)
    : null;
  const mappedProbeY = Number.isFinite(mappedProbeSummary?.mappedProbeY)
    ? Number(mappedProbeSummary.mappedProbeY)
    : null;
  const centerPixel = mappedProbeSummary?.mappedProbeValid
    ? (normalizeSharedRepresentativePixel([mappedProbeX, mappedProbeY]) ?? [0, 0])
    : null;

  if (!mappedProbeSummary?.mappedProbeValid || !centerPixel) {
    const patchSizes = [...MAPPED_PROBE_PATCH_SIZES];
    const summaries = {};
    const deltaSummaries = {};
    for (const size of patchSizes) {
      const key = `${size}x${size}`;
      summaries[key] = {
        patchSize: [size, size],
        valid: false,
        reason: mappedProbeSummary?.mappedProbeReason ?? 'mapped-probe-invalid',
        centerPixel: centerPixel ?? [0, 0],
        topLeft: [0, 0],
        bottomRight: [0, 0],
        pixelCount: 0,
        rgba8: [0, 0, 0, 0],
        meanRgb: [0, 0, 0],
        minRgb: [0, 0, 0],
        maxRgb: [0, 0, 0],
        referenceSource: 'none'
      };
      deltaSummaries[key] = {
        valid: false,
        reason: mappedProbeSummary?.mappedProbeReason ?? 'mapped-probe-invalid',
        referenceSource: 'none',
        comparedAgainstAccumulation: false,
        maxDelta: 0,
        meanDelta: 0,
        mismatchCountAboveTolerance: 0,
        worstPixelOffset: [0, 0],
        worstPixelDelta: [0, 0, 0]
      };
    }
    return {
      mappedProbePatchSizes: patchSizes,
      mappedProbePatchReadbackValid: false,
      mappedProbePatchReadbackReason: mappedProbeSummary?.mappedProbeReason ?? 'mapped-probe-invalid',
      mappedProbePatchCenter: centerPixel ?? [0, 0],
      mappedProbePatchSummaries: summaries,
      mappedProbePatchComparedAgainstAccumulation: false,
      mappedProbePatchDeltaSummaries: deltaSummaries
    };
  }

  const patchSizes = [...MAPPED_PROBE_PATCH_SIZES];
  const summaries = {};
  const deltaSummaries = {};
  let allValid = true;
  let firstFailureReason = 'patch-readback-ok';
  let comparedAgainstAccumulation = false;

  for (const size of patchSizes) {
    const key = `${size}x${size}`;
    const actualPatch = gl ? readFramebufferPatchAtTopLeftPixel(gl, centerPixel, size) : {
      valid: false,
      reason: 'webgl-unavailable',
      patchSize: size,
      centerPixel,
      topLeft: [0, 0],
      bottomRight: [0, 0],
      pixelCount: 0,
      centerRgba8: [0, 0, 0, 0],
      meanRgb: [0, 0, 0],
      minRgb: [0, 0, 0],
      maxRgb: [0, 0, 0],
      rawRgba8: []
    };
    if (!actualPatch.valid) {
      allValid = false;
      if (firstFailureReason === 'patch-readback-ok') {
        firstFailureReason = actualPatch.reason;
      }
    }

    let referencePatch = readMappedProbePatchReference(centerPixel, size, probeKey);
    let referenceSource = referencePatch ? 'stored-accumulation-patch-reference' : 'none';
    if (!referencePatch && tileCompositePath === 'accumulation' && actualPatch.valid) {
      referencePatch = {
        valid: true,
        rawRgba8: actualPatch.rawRgba8,
        referenceSource: 'current-accumulation-patch-reference'
      };
      referenceSource = 'current-accumulation-patch-reference';
      writeMappedProbePatchReference(centerPixel, size, actualPatch, probeKey);
    }

    const delta = buildPatchDeltaSummary(actualPatch, referencePatch);
    if (delta.comparedAgainstAccumulation) {
      comparedAgainstAccumulation = true;
    }
    if (tileCompositePath === 'accumulation' && actualPatch.valid) {
      writeMappedProbePatchReference(centerPixel, size, actualPatch);
    }

    summaries[key] = {
      patchSize: [size, size],
      valid: !!actualPatch.valid,
      reason: actualPatch.reason ?? 'none',
      centerPixel: actualPatch.centerPixel ?? centerPixel,
      topLeft: actualPatch.topLeft ?? [0, 0],
      bottomRight: actualPatch.bottomRight ?? [0, 0],
      pixelCount: actualPatch.pixelCount ?? 0,
      rgba8: actualPatch.centerRgba8 ?? [0, 0, 0, 0],
      meanRgb: actualPatch.meanRgb ?? [0, 0, 0],
      minRgb: actualPatch.minRgb ?? [0, 0, 0],
      maxRgb: actualPatch.maxRgb ?? [0, 0, 0],
      referenceSource
    };

    deltaSummaries[key] = {
      valid: delta.valid,
      reason: delta.reason,
      referenceSource: delta.referenceSource,
      comparedAgainstAccumulation: delta.comparedAgainstAccumulation,
      maxDelta: delta.maxDelta,
      meanDelta: delta.meanDelta,
      mismatchCountAboveTolerance: delta.mismatchCountAboveTolerance,
      worstPixelOffset: delta.worstPixelOffset,
      worstPixelDelta: delta.worstPixelDelta
    };
  }

  if (tileCompositePath === 'accumulation' && allValid) {
    for (const size of patchSizes) {
      const key = `${size}x${size}`;
      const actualPatch = summaries[key];
      if (actualPatch?.valid) {
        writeMappedProbePatchReference(centerPixel, size, {
          valid: true,
          patchSize: actualPatch.patchSize?.[0] ?? size,
          centerPixel: actualPatch.centerPixel,
          topLeft: actualPatch.topLeft,
          bottomRight: actualPatch.bottomRight,
          pixelCount: actualPatch.pixelCount,
          rawRgba8: readFramebufferPatchAtTopLeftPixel(
            gl,
            centerPixel,
            size
          ).rawRgba8
        }, probeKey);
      }
    }
  }

  return {
    mappedProbePatchSizes: patchSizes,
    mappedProbePatchReadbackValid: allValid,
    mappedProbePatchReadbackReason: firstFailureReason,
    mappedProbePatchCenter: centerPixel,
    mappedProbePatchSummaries: summaries,
    mappedProbePatchComparedAgainstAccumulation: comparedAgainstAccumulation,
    mappedProbePatchDeltaSummaries: deltaSummaries
  };
}

function buildMappedProbeCollectionsSummary(renderResultSummary, canvasSizeSummary = buildCanvasSizeSummary()) {
  const probeInputs = buildScreenshotProbeInputList();
  const useProbeKeyPrefix = !!deterministicQueryState.screenshotProbeList;
  const fallbackProbeInputs = probeInputs.length > 0
    ? probeInputs
    : [{
      valid: false,
      reason: 'missing-screenshot-probe-coordinate',
      probeId: 'probe-1',
      screenshotProbeLabel: 'probe-1',
      screenshotProbeX: null,
      screenshotProbeY: null,
      screenshotProbePixel: [null, null]
    }];

  const mappedProbeList = fallbackProbeInputs.map((probeInput, index) => {
    const mappedProbeSummary = buildMappedScreenshotProbeSummaryFromInput(probeInput, canvasSizeSummary);
    const probeKey = useProbeKeyPrefix
      ? (mappedProbeSummary?.probeId ?? probeInput?.probeId ?? `probe-${index + 1}`)
      : null;
    const mappedProbePatchSummary = buildMappedProbePatchSummary(
      renderResultSummary,
      canvasSizeSummary,
      mappedProbeSummary,
      probeKey
    );
    return {
      ...mappedProbeSummary,
      ...mappedProbePatchSummary
    };
  });
  const primaryMappedProbe = mappedProbeList[0] ?? null;

  return {
    screenshotProbeList: fallbackProbeInputs,
    mappedProbeList,
    collectionSummary: {
      source: deterministicQueryState.screenshotProbeList ? 'explicit-screenshotProbeList' : 'single-screenshotProbe',
      probeCount: mappedProbeList.length,
      probeIds: mappedProbeList.map((probe) => probe?.probeId ?? 'probe-unknown'),
      probeLabels: mappedProbeList.map((probe) => probe?.screenshotProbeLabel ?? probe?.probeId ?? 'probe-unknown'),
      primaryProbeId: primaryMappedProbe?.probeId ?? 'probe-1',
      primaryProbeLabel: primaryMappedProbe?.screenshotProbeLabel ?? primaryMappedProbe?.probeId ?? 'probe-1',
      primaryProbeSource: primaryMappedProbe?.mappedProbeSourceSummary?.source ?? 'screenshot-query',
      primaryProbeValid: !!primaryMappedProbe?.mappedProbeValid,
      usesFixedCanvas: !!canvasSizeSummary?.fixedCanvasActive,
      coordinateSpaceForReadPixels: canvasSizeSummary?.coordinateSpaceForReadPixels ?? 'webgl-default-framebuffer-pixels'
    },
    firstMappedProbeSummary: primaryMappedProbe,
    firstMappedProbePatchSummary: primaryMappedProbe
  };
}

function normalizeInspectSource(value, fallback = 'auto') {
  return INSPECT_SOURCE_VALUES.has(value) ? value : fallback;
}

function normalizeInspectJsonMode(value, fallback = 'slim') {
  return INSPECT_JSON_MODE_VALUES.has(value) ? value : fallback;
}

function getRequestedTileCompositePrimitive() {
  return ui.tileCompositePrimitiveSelect?.value === 'quad' ? 'quad' : 'point';
}

function getRequestedTileCompositePath() {
  return ui.tileCompositePathSelect?.value === 'accumulation' ? 'accumulation' : 'baseline';
}

function parseNumberTuple(value, expectedLength) {
  if (value === null || value === undefined || value === '') return null;
  const parts = String(value).split(',').map((part) => Number(part.trim()));
  if (parts.length !== expectedLength || parts.some((part) => !Number.isFinite(part))) return null;
  return parts;
}

function getQueryNumberTuple(name, expectedLength) {
  if (typeof window === 'undefined') return null;
  return parseNumberTuple(new URLSearchParams(window.location.search || '').get(name), expectedLength);
}

function readStoredNumberTuple(key, expectedLength) {
  if (typeof window === 'undefined' || !window.localStorage) return null;
  try {
    return parseNumberTuple(window.localStorage.getItem(key), expectedLength);
  } catch {
    return null;
  }
}

function writeStoredNumberTuple(key, values) {
  if (typeof window === 'undefined' || !window.localStorage || !Array.isArray(values)) return;
  try {
    window.localStorage.setItem(key, values.join(','));
  } catch {
    // Storage is best-effort diagnostics only.
  }
}

function readStoredJson(key) {
  if (typeof window === 'undefined' || !window.localStorage) return null;
  try {
    const value = window.localStorage.getItem(key);
    if (!value) return null;
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function writeStoredJson(key, value) {
  if (typeof window === 'undefined' || !window.localStorage) return;
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage is best-effort diagnostics only.
  }
}

function normalizeSharedRepresentativePixel(pixel) {
  if (!Array.isArray(pixel) || pixel.length < 2) return null;
  const x = Number.isFinite(pixel[0]) ? Math.floor(pixel[0]) : -1;
  const y = Number.isFinite(pixel[1]) ? Math.floor(pixel[1]) : -1;
  return x >= 0 && y >= 0 ? [x, y] : null;
}

function clampColor01ForSummary(rgb) {
  const safe = Array.isArray(rgb) ? rgb : [0, 0, 0];
  return [
    Number.isFinite(safe[0]) ? Number(safe[0]) : 0,
    Number.isFinite(safe[1]) ? Number(safe[1]) : 0,
    Number.isFinite(safe[2]) ? Number(safe[2]) : 0
  ];
}

function buildColorDeltaForSummary(referenceColor, color) {
  const reference = clampColor01ForSummary(referenceColor);
  const actual = clampColor01ForSummary(color);
  const delta = [
    actual[0] - reference[0],
    actual[1] - reference[1],
    actual[2] - reference[2]
  ];
  return {
    delta,
    deltaAbsMax: Math.max(Math.abs(delta[0]), Math.abs(delta[1]), Math.abs(delta[2]))
  };
}

function readFramebufferColorAtTopLeftPixel(gl, pixel) {
  const normalizedPixel = normalizeSharedRepresentativePixel(pixel);
  if (!normalizedPixel) {
    return {
      color: [0, 0, 0],
      rgba8: [0, 0, 0, 0],
      valid: false,
      reason: 'invalid-shared-representative-pixel',
      pixel: [0, 0],
      glPixel: [0, 0]
    };
  }
  const width = Number.isFinite(canvas?.width) ? (canvas.width | 0) : 0;
  const height = Number.isFinite(canvas?.height) ? (canvas.height | 0) : 0;
  const [x, yTop] = normalizedPixel;
  if (width <= 0 || height <= 0) {
    return {
      color: [0, 0, 0],
      rgba8: [0, 0, 0, 0],
      valid: false,
      reason: 'invalid-canvas-size',
      pixel: normalizedPixel,
      glPixel: [0, 0]
    };
  }
  if (x >= width || yTop >= height) {
    return {
      color: [0, 0, 0],
      rgba8: [0, 0, 0, 0],
      valid: false,
      reason: 'shared-representative-pixel-out-of-bounds',
      pixel: normalizedPixel,
      glPixel: [0, 0]
    };
  }

  const yGl = height - 1 - yTop;
  const rgba = new Uint8Array(4);
  const previousFramebuffer = gl.getParameter(gl.FRAMEBUFFER_BINDING);
  try {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.readPixels(x, yGl, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
  } catch (error) {
    return {
      color: [0, 0, 0],
      rgba8: Array.from(rgba),
      valid: false,
      reason: `readpixels-failed:${error?.message ?? 'unknown'}`,
      pixel: normalizedPixel,
      glPixel: [x, yGl]
    };
  } finally {
    gl.bindFramebuffer(gl.FRAMEBUFFER, previousFramebuffer);
  }

  return {
    color: [rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0],
    rgba8: Array.from(rgba),
    valid: true,
    reason: 'readback-ok',
    pixel: normalizedPixel,
    glPixel: [x, yGl]
  };
}

function resolveSharedRepresentativePixel(executionSummary) {
  const queryPixel = normalizeSharedRepresentativePixel(getQueryNumberTuple('sharedRepresentativePixel', 2));
  if (queryPixel) return { pixel: queryPixel, source: 'query-sharedRepresentativePixel' };

  const accumulationPixel = normalizeSharedRepresentativePixel(
    executionSummary?.tileAccumulationRepresentativeSamplePixel
  );
  if (accumulationPixel) {
    writeStoredNumberTuple(SHARED_REPRESENTATIVE_PIXEL_STORAGE_KEY, accumulationPixel);
    return { pixel: accumulationPixel, source: 'accumulation-representative-sample' };
  }

  const storedPixel = normalizeSharedRepresentativePixel(
    readStoredNumberTuple(SHARED_REPRESENTATIVE_PIXEL_STORAGE_KEY, 2)
  );
  if (storedPixel) return { pixel: storedPixel, source: 'stored-accumulation-representative-sample' };

  return {
    pixel: SHARED_REPRESENTATIVE_DEFAULT_PIXEL,
    source: 'step86-default-representative-pixel'
  };
}

function resolveSharedRepresentativeAccumulationReference(executionSummary, sharedPixel) {
  const currentColor = executionSummary?.tileAccumulationRepresentativeSampleFramebufferReadbackValid
    ? clampColor01ForSummary(executionSummary.tileAccumulationRepresentativeSampleFramebufferColor)
    : null;
  const currentPixel = normalizeSharedRepresentativePixel(
    executionSummary?.tileAccumulationRepresentativeSampleFramebufferReadbackPixel ??
    executionSummary?.tileAccumulationRepresentativeSamplePixel
  );
  if (currentColor && currentPixel && currentPixel[0] === sharedPixel[0] && currentPixel[1] === sharedPixel[1]) {
    writeStoredNumberTuple(SHARED_REPRESENTATIVE_ACCUMULATION_COLOR_STORAGE_KEY, currentColor);
    return {
      color: currentColor,
      source: 'current-accumulation-framebuffer-readback',
      pixel: currentPixel
    };
  }

  const queryColor = getQueryNumberTuple('sharedRepresentativeAccumulationColor', 3);
  if (queryColor) {
    return {
      color: clampColor01ForSummary(queryColor),
      source: 'query-sharedRepresentativeAccumulationColor',
      pixel: sharedPixel
    };
  }

  const storedColor = readStoredNumberTuple(SHARED_REPRESENTATIVE_ACCUMULATION_COLOR_STORAGE_KEY, 3);
  if (storedColor) {
    return {
      color: clampColor01ForSummary(storedColor),
      source: 'stored-accumulation-framebuffer-readback',
      pixel: sharedPixel
    };
  }

  return {
    color: null,
    source: 'accumulation-reference-unavailable',
    pixel: sharedPixel
  };
}

function buildSharedRepresentativeFramebufferProbe(renderResultSummary) {
  const gpu = getGpu();
  const gl = gpu?.gl;
  const executionSummary = renderResultSummary?.executionSummary ?? null;
  const { pixel, source } = resolveSharedRepresentativePixel(executionSummary);
  if (!gl) {
    return {
      sharedRepresentativePixel: pixel,
      sharedRepresentativePixelSource: source,
      sharedRepresentativeFramebufferColor: [0, 0, 0],
      sharedRepresentativeFramebufferReadbackValid: false,
      sharedRepresentativeFramebufferReadbackReason: 'webgl-unavailable',
      sharedRepresentativeFramebufferRgba8: [0, 0, 0, 0],
      sharedRepresentativeComparedAgainstAccumulationPixel: false,
      sharedRepresentativeAccumulationReferenceColor: [0, 0, 0],
      sharedRepresentativeAccumulationReferenceSource: 'accumulation-reference-unavailable',
      sharedRepresentativeColorDeltaVsAccumulation: [0, 0, 0],
      sharedRepresentativeColorDeltaVsAccumulationAbsMax: 0,
      sharedRepresentativeColorMatchesAccumulation: false,
      sharedRepresentativeColorMatchTolerance: SHARED_REPRESENTATIVE_COLOR_MATCH_TOLERANCE
    };
  }

  const readback = readFramebufferColorAtTopLeftPixel(gl, pixel);
  const reference = resolveSharedRepresentativeAccumulationReference(executionSummary, readback.pixel);
  const hasReference = Array.isArray(reference.color);
  const { delta, deltaAbsMax } = hasReference
    ? buildColorDeltaForSummary(reference.color, readback.color)
    : { delta: [0, 0, 0], deltaAbsMax: 0 };
  const comparedAgainstAccumulationPixel = hasReference &&
    Array.isArray(reference.pixel) &&
    reference.pixel[0] === readback.pixel[0] &&
    reference.pixel[1] === readback.pixel[1];

  return {
    sharedRepresentativePixel: readback.pixel,
    sharedRepresentativePixelSource: source,
    sharedRepresentativeFramebufferColor: readback.color,
    sharedRepresentativeFramebufferReadbackValid: readback.valid,
    sharedRepresentativeFramebufferReadbackReason: readback.reason,
    sharedRepresentativeFramebufferRgba8: readback.rgba8,
    sharedRepresentativeFramebufferReadbackGlPixel: readback.glPixel,
    sharedRepresentativeComparedAgainstAccumulationPixel: comparedAgainstAccumulationPixel,
    sharedRepresentativeAccumulationReferenceColor: hasReference ? reference.color : [0, 0, 0],
    sharedRepresentativeAccumulationReferenceSource: reference.source,
    sharedRepresentativeColorDeltaVsAccumulation: delta,
    sharedRepresentativeColorDeltaVsAccumulationAbsMax: deltaAbsMax,
    sharedRepresentativeColorMatchesAccumulation: readback.valid &&
      comparedAgainstAccumulationPixel &&
      deltaAbsMax <= SHARED_REPRESENTATIVE_COLOR_MATCH_TOLERANCE,
    sharedRepresentativeColorMatchTolerance: SHARED_REPRESENTATIVE_COLOR_MATCH_TOLERANCE
  };
}

function buildSlimDeterministicStateSummary(summary) {
  return {
    active: !!summary?.active,
    cameraPresetName: summary?.cameraPresetName ?? 'none',
    appliedCameraPresetName: summary?.appliedCameraPresetName ?? 'none',
    cameraSource: summary?.cameraSource ?? 'camera-preset',
    datasetCameraConvention: summary?.datasetCameraConvention ?? null,
    datasetViewMatrixMode: summary?.datasetViewMatrixMode ?? 'threejs',
    cameraControlContract: summary?.cameraControlContract ?? null,
    cameraOrientationPolicy: summary?.cameraOrientationPolicy ?? null,
    datasetPixelXSign: [-1, 1].includes(summary?.datasetPixelXSign) ? Number(summary.datasetPixelXSign) : 1,
    datasetCameraLabel: summary?.datasetCameraLabel ?? null,
    imageName: summary?.imageName ?? null,
    frameNumber: Number.isFinite(summary?.frameNumber) ? Number(summary.frameNumber) : null,
    viewId: Number.isFinite(summary?.viewId) ? Number(summary.viewId) : null,
    datasetTime: Number.isFinite(summary?.datasetTime) ? Number(summary.datasetTime) : null,
    rawTransformMatrix: Array.isArray(summary?.rawTransformMatrix) ? summary.rawTransformMatrix.map((row) => [...row]) : null,
    convertedCameraPose: summary?.convertedCameraPose ?? null,
    referenceCameraPose: summary?.referenceCameraPose ?? null,
    screenAxisMapping: summary?.screenAxisMapping ?? null,
    cameraPosition: Array.isArray(summary?.cameraPosition) ? [...summary.cameraPosition] : null,
    cameraTarget: Array.isArray(summary?.cameraTarget) ? [...summary.cameraTarget] : null,
    cameraUp: Array.isArray(summary?.cameraUp) ? [...summary.cameraUp] : null,
    cameraFoVyRad: Number.isFinite(summary?.cameraFoVyRad) ? Number(summary.cameraFoVyRad) : null,
    cameraFoVxRad: Number.isFinite(summary?.cameraFoVxRad) ? Number(summary.cameraFoVxRad) : null,
    cameraFoVyDeg: Number.isFinite(summary?.cameraFoVyDeg) ? Number(summary.cameraFoVyDeg) : null,
    cameraFoVxDeg: Number.isFinite(summary?.cameraFoVxDeg) ? Number(summary.cameraFoVxDeg) : null,
    cameraFoVy: Number.isFinite(summary?.cameraFoVy) ? Number(summary.cameraFoVy) : null,
    cameraFoVx: Number.isFinite(summary?.cameraFoVx) ? Number(summary.cameraFoVx) : null,
    appliedCameraFovDeg: Number.isFinite(summary?.appliedCameraFovDeg) ? Number(summary.appliedCameraFovDeg) : null,
    intrinsics: summary?.intrinsics ?? null,
    stride: Number.isFinite(summary?.stride) ? Number(summary.stride) : null,
    bgGray: Number.isFinite(summary?.bgGray) ? Number(summary.bgGray) : null,
    debugPreserveDrawingBuffer: typeof summary?.debugPreserveDrawingBuffer === 'boolean'
      ? summary.debugPreserveDrawingBuffer
      : null,
    cudaReferenceLabel: summary?.cudaReferenceLabel ?? null,
    cudaReferencePath: summary?.cudaReferencePath ?? null,
    actualCameraPosition: Array.isArray(summary?.actualCameraPosition) ? [...summary.actualCameraPosition] : null,
    actualCameraQuaternion: Array.isArray(summary?.actualCameraQuaternion) ? [...summary.actualCameraQuaternion] : null,
    actualCameraUp: Array.isArray(summary?.actualCameraUp) ? [...summary.actualCameraUp] : null,
    actualControlsTarget: Array.isArray(summary?.actualControlsTarget) ? [...summary.actualControlsTarget] : null,
    actualCameraFov: Number.isFinite(summary?.actualCameraFov) ? Number(summary.actualCameraFov) : null,
    actualCameraNear: Number.isFinite(summary?.actualCameraNear) ? Number(summary.actualCameraNear) : null,
    actualCameraFar: Number.isFinite(summary?.actualCameraFar) ? Number(summary.actualCameraFar) : null,
    actualCameraMatrixWorld: Array.isArray(summary?.actualCameraMatrixWorld)
      ? summary.actualCameraMatrixWorld.map((row) => Array.isArray(row) ? [...row] : row)
      : null,
    actualCameraRight: Array.isArray(summary?.actualCameraRight) ? [...summary.actualCameraRight] : null,
    cudaAlignedScreenSpaceCamera: summary?.cudaAlignedScreenSpaceCamera ?? null,
    fixedReferenceScreenSpaceCamera: summary?.fixedReferenceScreenSpaceCamera ?? null,
    drawPath: summary?.drawPath ?? 'none',
    tileCompositePath: summary?.tileCompositePath ?? 'baseline',
    tileCompositePrimitive: summary?.tileCompositePrimitive ?? 'point',
    inspectSource: summary?.inspectSource ?? 'auto',
    inspectJsonMode: summary?.inspectJsonMode ?? 'slim',
    gpuFramePolicyOverride: summary?.gpuFramePolicyOverride ?? 'auto',
    gpuCandidateRuntime: summary?.gpuCandidateRuntime ?? 'off',
    gpuCandidateFallback: summary?.gpuCandidateFallback ?? null,
    gpuCandidateRequireCompare: typeof summary?.gpuCandidateRequireCompare === 'boolean'
      ? summary.gpuCandidateRequireCompare
      : null,
    gpuCandidateRequireShadowOk: typeof summary?.gpuCandidateRequireShadowOk === 'boolean'
      ? summary.gpuCandidateRequireShadowOk
      : null,
    gpuCandidateSubsetMode: summary?.gpuCandidateSubsetMode ?? null,
    gpuCandidateSubsetCount: Number.isFinite(summary?.gpuCandidateSubsetCount)
      ? Number(summary.gpuCandidateSubsetCount)
      : null,
    gpuCandidateFilterMode: summary?.gpuCandidateFilterMode ?? null,
    gpuCandidateSourceMode: summary?.gpuCandidateSourceMode ?? null,
    gpuCandidateRangeStart: Number.isFinite(summary?.gpuCandidateRangeStart)
      ? Number(summary.gpuCandidateRangeStart)
      : null,
    gpuCandidateRangeCount: Number.isFinite(summary?.gpuCandidateRangeCount)
      ? Number(summary.gpuCandidateRangeCount)
      : null,
    gpuCandidatePromotePolicy: summary?.gpuCandidatePromotePolicy ?? null,
    gpuCandidateReadbackMode: summary?.gpuCandidateReadbackMode ?? null,
    gpuCandidateCoverageCompare: typeof summary?.gpuCandidateCoverageCompare === 'boolean'
      ? summary.gpuCandidateCoverageCompare
      : null,
    gpuCandidateCoverageMaxMisses: Number.isFinite(summary?.gpuCandidateCoverageMaxMisses)
      ? Number(summary.gpuCandidateCoverageMaxMisses)
      : null,
    gpuCandidateCompare: typeof summary?.gpuCandidateCompare === 'boolean'
      ? summary.gpuCandidateCompare
      : null,
    gpuCandidateAllowReadbackInDraw: typeof summary?.gpuCandidateAllowReadbackInDraw === 'boolean'
      ? summary.gpuCandidateAllowReadbackInDraw
      : null,
    gpuCandidateDebugReadback: typeof summary?.gpuCandidateDebugReadback === 'boolean'
      ? summary.gpuCandidateDebugReadback
      : null,
    webgpuBackendMode: summary?.webgpuBackendMode ?? null,
    webgpuAllowViewerCanvasPresentation:
      typeof summary?.webgpuAllowViewerCanvasPresentation === 'boolean'
        ? summary.webgpuAllowViewerCanvasPresentation
        : null,
    webgpuBackendViewerLoopHook:
      typeof summary?.webgpuBackendViewerLoopHook === 'boolean'
        ? summary.webgpuBackendViewerLoopHook
        : null,
    time: Number.isFinite(summary?.time) ? Number(summary.time) : null,
    deterministicQueryString: summary?.deterministicQueryString ?? '',
    deterministicUrlSummary: summary?.deterministicUrlSummary ?? ''
  };
}

function buildRenderResultInspectionSummary(renderResult) {
  const executionSummary = renderResult?.executionSummary ?? null;
  const tileCompositeSummary = renderResult?.tileCompositePlan?.summary ?? null;
  const summary = {
    canvasSizeSummary: buildCanvasSizeSummary(),
    actualDrawPath:
      renderResult?.drawThroughputSummary?.actualDrawPath ??
      renderResult?.drawPathSummary?.actualPath ??
      'none',
    drawPathSummary: renderResult?.drawPathSummary ?? null,
    drawThroughputSummary: renderResult?.drawThroughputSummary ?? null,
    gpuFallbackSummary: renderResult?.gpuFallbackSummary ?? null,
    gpuCompatibilityBridgeSummary: renderResult?.gpuCompatibilityBridgeSummary ?? null,
    limitedDrawRuntimeSummary: renderResult?.limitedDrawRuntimeSummary ?? null,
    gpuCandidateRuntimeSummary: renderResult?.gpuCandidateRuntimeSummary ?? null,
    gpuCandidateRuntimeFallback: renderResult?.gpuCandidateRuntimeFallback ?? null,
    limitedDrawUsedForCandidateSource: !!renderResult?.limitedDrawRuntimeSummary?.limitedDrawUsedForCandidateSource,
    displayCandidateSource: renderResult?.limitedDrawRuntimeSummary?.displayCandidateSource ?? 'cpu-reference',
    executionSummary: executionSummary
      ? {
          requestedDrawPath: executionSummary.requestedDrawPath ?? 'none',
          actualDrawPath: executionSummary.actualDrawPath ?? 'none',
          drawPathFallbackReason: executionSummary.drawPathFallbackReason ?? 'none',
          compositingContract: executionSummary.compositingContract ?? 'none',
          tileCompositePath: executionSummary.tileCompositePath ?? 'none',
          tileCompositePrimitive: executionSummary.tileCompositePrimitive ?? 'none',
          tileCompositePrimitivePolicy: executionSummary.tileCompositePrimitivePolicy ?? 'none',
          tileCompositePrimitivePolicyReason: executionSummary.tileCompositePrimitivePolicyReason ?? 'none',
          tileCompositePrimitiveResolved: executionSummary.tileCompositePrimitiveResolved ?? 'none',
          tileCompositeRectContract: executionSummary.tileCompositeRectContract ?? 'none',
          tileCompositeAccumulationTargetFormat: executionSummary.tileCompositeAccumulationTargetFormat ?? 'none',
          tileCompositeAccumulationTargetType: executionSummary.tileCompositeAccumulationTargetType ?? 'none',
          tileCompositeAccumulationTargetPrecisionContract: executionSummary.tileCompositeAccumulationTargetPrecisionContract ?? 'none',
          tileCompositeAccumulationTargetFallbackReason: executionSummary.tileCompositeAccumulationTargetFallbackReason ?? 'none',
          tileCompositeResolveContract: executionSummary.tileCompositeResolveContract ?? 'none',
          tileBatchCount: executionSummary.tileBatchCount ?? 0,
          nonEmptyTileBatchCount: executionSummary.nonEmptyTileBatchCount ?? 0,
          totalTileDrawCount: executionSummary.totalTileDrawCount ?? 0,
          drawCallCount: executionSummary.drawCallCount ?? 0,
          uploadCount: executionSummary.uploadCount ?? 0,
          requestedTextureWidth: executionSummary.requestedTextureWidth ?? 0,
          requestedTextureHeight: executionSummary.requestedTextureHeight ?? 0,
          validatedTextureWidth: executionSummary.validatedTextureWidth ?? 0,
          validatedTextureHeight: executionSummary.validatedTextureHeight ?? 0,
          textureAllocationValid: !!executionSummary.textureAllocationValid,
          textureAllocationFailureReason: executionSummary.textureAllocationFailureReason ?? 'none',
          accumulationMaxItemsPerTile: executionSummary.accumulationMaxItemsPerTile ?? 0,
          accumulationTruncatedBatchCount: executionSummary.accumulationTruncatedBatchCount ?? 0,
          tileAccumulationTruncatedTileCount: executionSummary.tileAccumulationTruncatedTileCount ?? 0,
          tileAccumulationMaxObservedTileItems: executionSummary.tileAccumulationMaxObservedTileItems ?? 0,
          tileAccumulationTotalSkippedItems: executionSummary.tileAccumulationTotalSkippedItems ?? 0,
          tileAccumulationWorstTileId: executionSummary.tileAccumulationWorstTileId ?? -1,
          tileAccumulationWorstTileItemCount: executionSummary.tileAccumulationWorstTileItemCount ?? 0,
          tileAccumulationWorstTileSkippedCount: executionSummary.tileAccumulationWorstTileSkippedCount ?? 0,
          tileAccumulationEarlyOutEnabled: !!executionSummary.tileAccumulationEarlyOutEnabled,
          tileAccumulationEarlyOutThreshold: Number.isFinite(executionSummary.tileAccumulationEarlyOutThreshold)
            ? Number(executionSummary.tileAccumulationEarlyOutThreshold)
            : 0,
          tileAccumulationEarlyOutTriggeredTileCount: executionSummary.tileAccumulationEarlyOutTriggeredTileCount ?? 0,
          tileAccumulationEarlyOutTriggeredPixelEstimate: executionSummary.tileAccumulationEarlyOutTriggeredPixelEstimate ?? 0,
          tileAccumulationWorstEarlyOutTileId: executionSummary.tileAccumulationWorstEarlyOutTileId ?? -1,
          tileAccumulationWorstEarlyOutCount: executionSummary.tileAccumulationWorstEarlyOutCount ?? 0,
          tileAccumulationAverageVisitedItemsPerTile: Number.isFinite(executionSummary.tileAccumulationAverageVisitedItemsPerTile)
            ? Number(executionSummary.tileAccumulationAverageVisitedItemsPerTile)
            : 0,
          tileAccumulationMaxVisitedItemsPerTile: executionSummary.tileAccumulationMaxVisitedItemsPerTile ?? 0,
          tileAccumulationAverageVisitedItemsPerPixelEstimate: Number.isFinite(executionSummary.tileAccumulationAverageVisitedItemsPerPixelEstimate)
            ? Number(executionSummary.tileAccumulationAverageVisitedItemsPerPixelEstimate)
            : 0,
          tileAccumulationVisitedRatioSummary: executionSummary.tileAccumulationVisitedRatioSummary ?? null,
          tileAccumulationObservedTileSummaries: executionSummary.tileAccumulationObservedTileSummaries ?? [],
          tileAccumulationOrderingSummary: executionSummary.tileAccumulationOrderingSummary ?? null,
          tileAccumulationBatchBoundarySummary: executionSummary.tileAccumulationBatchBoundarySummary ?? null,
          tileAccumulationObservedOrderingMismatches: executionSummary.tileAccumulationObservedOrderingMismatches ?? [],
          tileAccumulationHeavyTileSummaries: executionSummary.tileAccumulationHeavyTileSummaries ?? [],
          tileAccumulationRepresentativeTileId: executionSummary.tileAccumulationRepresentativeTileId ?? -1,
          tileAccumulationRepresentativeTileItemCount: executionSummary.tileAccumulationRepresentativeTileItemCount ?? 0,
          tileAccumulationRepresentativeTileOrderPreview: executionSummary.tileAccumulationRepresentativeTileOrderPreview ?? null,
          tileAccumulationRepresentativeTileDepthPreview: executionSummary.tileAccumulationRepresentativeTileDepthPreview ?? null,
          tileAccumulationRepresentativeTileBatchSpan: executionSummary.tileAccumulationRepresentativeTileBatchSpan ?? 1,
          tileAccumulationRepresentativeTileSequenceConsistent: !!executionSummary.tileAccumulationRepresentativeTileSequenceConsistent,
          tileAccumulationContributionSummary: executionSummary.tileAccumulationContributionSummary ?? null,
          tileAccumulationRepresentativeSampleMode: executionSummary.tileAccumulationRepresentativeSampleMode ?? 'none',
          tileAccumulationRepresentativeSampleSelectionMode: executionSummary.tileAccumulationRepresentativeSampleSelectionMode ?? 'none',
          tileAccumulationRepresentativeSampleSelectionReason: executionSummary.tileAccumulationRepresentativeSampleSelectionReason ?? 'none',
          tileAccumulationRepresentativeSamplePixel: executionSummary.tileAccumulationRepresentativeSamplePixel ?? [0, 0],
          tileAccumulationRepresentativeSampleHasContribution: !!executionSummary.tileAccumulationRepresentativeSampleHasContribution,
          tileAccumulationRepresentativeSampleCandidateCount: executionSummary.tileAccumulationRepresentativeSampleCandidateCount ?? 0,
          tileAccumulationRepresentativeSampleEvaluatedCandidateCount: executionSummary.tileAccumulationRepresentativeSampleEvaluatedCandidateCount ?? 0,
          tileAccumulationRepresentativeSampleUsableItemSource: executionSummary.tileAccumulationRepresentativeSampleUsableItemSource ?? 'none',
          tileAccumulationRepresentativeSampleItemReadMode: executionSummary.tileAccumulationRepresentativeSampleItemReadMode ?? 'none',
          tileAccumulationRepresentativeSampleEvaluatedItemCount: executionSummary.tileAccumulationRepresentativeSampleEvaluatedItemCount ?? 0,
          tileAccumulationRepresentativeSampleContributionLog: executionSummary.tileAccumulationRepresentativeSampleContributionLog ?? [],
          tileAccumulationRepresentativeSampleFinalT: Number.isFinite(executionSummary.tileAccumulationRepresentativeSampleFinalT)
            ? Number(executionSummary.tileAccumulationRepresentativeSampleFinalT)
            : 1,
          tileAccumulationRepresentativeSampleAccumColor: executionSummary.tileAccumulationRepresentativeSampleAccumColor ?? [0, 0, 0],
          tileAccumulationRepresentativeSampleResolvedColor: executionSummary.tileAccumulationRepresentativeSampleResolvedColor ?? [0, 0, 0],
          tileAccumulationRepresentativeSampleContributionCount: executionSummary.tileAccumulationRepresentativeSampleContributionCount ?? 0,
          tileAccumulationRepresentativeSampleAlphaSum: Number.isFinite(executionSummary.tileAccumulationRepresentativeSampleAlphaSum)
            ? Number(executionSummary.tileAccumulationRepresentativeSampleAlphaSum)
            : 0,
          tileAccumulationRepresentativeSampleContributionSum: executionSummary.tileAccumulationRepresentativeSampleContributionSum ?? [0, 0, 0],
          tileAccumulationRepresentativeSampleLastContributedLocalOrder: executionSummary.tileAccumulationRepresentativeSampleLastContributedLocalOrder ?? -1,
          tileAccumulationRepresentativeSampleThresholdCrossingCount: executionSummary.tileAccumulationRepresentativeSampleThresholdCrossingCount ?? 0,
          tileAccumulationRepresentativeSampleThresholdSkippedCount: executionSummary.tileAccumulationRepresentativeSampleThresholdSkippedCount ?? 0,
          tileAccumulationRepresentativeSampleFirstThresholdSkipLocalOrder: executionSummary.tileAccumulationRepresentativeSampleFirstThresholdSkipLocalOrder ?? -1,
          tileAccumulationRepresentativeSampleFirstThresholdSkipAlpha: Number.isFinite(executionSummary.tileAccumulationRepresentativeSampleFirstThresholdSkipAlpha)
            ? Number(executionSummary.tileAccumulationRepresentativeSampleFirstThresholdSkipAlpha)
            : 0,
          tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceBefore: Number.isFinite(executionSummary.tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceBefore)
            ? Number(executionSummary.tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceBefore)
            : 1,
          tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceAfter: Number.isFinite(executionSummary.tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceAfter)
            ? Number(executionSummary.tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceAfter)
            : 1,
          tileAccumulationRepresentativeSampleThresholdSkipPreview: executionSummary.tileAccumulationRepresentativeSampleThresholdSkipPreview ?? [],
          tileAccumulationRepresentativeSampleThresholdSemantics: executionSummary.tileAccumulationRepresentativeSampleThresholdSemantics ?? null,
          tileAccumulationRepresentativeSampleFramebufferColor: executionSummary.tileAccumulationRepresentativeSampleFramebufferColor ?? [0, 0, 0],
          tileAccumulationRepresentativeSampleFramebufferReadbackValid: !!executionSummary.tileAccumulationRepresentativeSampleFramebufferReadbackValid,
          tileAccumulationRepresentativeSampleFramebufferReadbackReason: executionSummary.tileAccumulationRepresentativeSampleFramebufferReadbackReason ?? 'not-attempted',
          tileAccumulationRepresentativeSampleFramebufferReadbackPixel: executionSummary.tileAccumulationRepresentativeSampleFramebufferReadbackPixel ?? [0, 0],
          tileAccumulationRepresentativeSampleFramebufferReadbackGlPixel: executionSummary.tileAccumulationRepresentativeSampleFramebufferReadbackGlPixel ?? [0, 0],
          tileAccumulationRepresentativeSampleFramebufferReadbackRgba8: executionSummary.tileAccumulationRepresentativeSampleFramebufferReadbackRgba8 ?? [0, 0, 0, 0],
          tileAccumulationRepresentativeSampleResolvedColorDelta: executionSummary.tileAccumulationRepresentativeSampleResolvedColorDelta ?? [0, 0, 0],
          tileAccumulationRepresentativeSampleResolvedColorDeltaAbsMax: Number.isFinite(executionSummary.tileAccumulationRepresentativeSampleResolvedColorDeltaAbsMax)
            ? Number(executionSummary.tileAccumulationRepresentativeSampleResolvedColorDeltaAbsMax)
            : 0,
          tileAccumulationRepresentativeSampleResolvedColorMatchesFramebuffer: !!executionSummary.tileAccumulationRepresentativeSampleResolvedColorMatchesFramebuffer,
          tileAccumulationRepresentativeSampleResolvedColorMatchTolerance: Number.isFinite(executionSummary.tileAccumulationRepresentativeSampleResolvedColorMatchTolerance)
            ? Number(executionSummary.tileAccumulationRepresentativeSampleResolvedColorMatchTolerance)
            : 0,
          tileAccumulationContractVersion: executionSummary.tileAccumulationContractVersion ?? 'none',
          tileAccumulationTruncationRatio: Number.isFinite(executionSummary.tileAccumulationTruncationRatio)
            ? Number(executionSummary.tileAccumulationTruncationRatio)
            : 0
        }
      : null,
    tileCompositeSummary: tileCompositeSummary
      ? {
          compositingContract: tileCompositeSummary.compositingContract ?? 'none',
          depthOrder: tileCompositeSummary.depthOrder ?? 'none',
          tileBatchCount: tileCompositeSummary.tileBatchCount ?? 0,
          nonEmptyTileBatchCount: tileCompositeSummary.nonEmptyTileBatchCount ?? 0,
          totalTileDrawCount: tileCompositeSummary.totalTileDrawCount ?? 0,
          maxTileDrawCount: tileCompositeSummary.maxTileDrawCount ?? 0,
          tileCompositeDuplicateRefs: tileCompositeSummary.tileCompositeDuplicateRefs ?? 0,
          tileCompositeOverlapFactor: tileCompositeSummary.tileCompositeOverlapFactor ?? 0
        }
      : null,
    packedSummary: renderResult?.packedSummary
      ? {
          path: renderResult.packedSummary.path ?? 'none',
          packedContract: renderResult.packedSummary.packedContract ?? 'none',
          transformPath: renderResult.packedSummary.transformPath ?? 'none',
          transformFallbackReason: renderResult.packedSummary.transformFallbackReason ?? 'none',
          transformFallbackContract: renderResult.packedSummary.transformFallbackContract ?? 'none',
          sourceItemCount: renderResult.packedSummary.sourceItemCount ?? 0,
          packedCount: renderResult.packedSummary.packedCount ?? 0,
          floatsPerItem: renderResult.packedSummary.floatsPerItem ?? 0
        }
      : null,
    gpuScreenSourceSummary: renderResult?.gpuScreenSourceInfo
      ? {
          sourceFallbackReason: renderResult.gpuScreenSourceInfo.sourceFallbackReason ?? 'none',
          sourceContract: renderResult.gpuScreenSourceInfo.sourceContract ?? 'none',
          sourceFallbackContract: renderResult.gpuScreenSourceInfo.sourceFallbackContract ?? 'none'
        }
      : null,
    gpuScreenExecutionSummary: renderResult?.gpuScreenExecutionSummary ?? null,
    webgpuBackendViewerLifecycleIntegrationBoundary:
      renderResult?.webgpuBackendViewerLifecycleIntegrationBoundary ?? null,
    webgpuBackendViewerLifecycleControlledExecution:
      renderResult?.webgpuBackendViewerLifecycleControlledExecution ?? null,
    webgpuBackendViewerFrameExecutor:
      renderResult?.webgpuBackendViewerFrameExecutor ?? null,
    tileAccumulationPayloadSummary: renderResult?.tileAccumulationPayloadSummary
      ? {
          payloadContract: renderResult.tileAccumulationPayloadSummary.payloadContract ?? 'none',
          batchCount: renderResult.tileAccumulationPayloadSummary.batchCount ?? 0,
          totalItemCount: renderResult.tileAccumulationPayloadSummary.totalItemCount ?? 0,
          maxBatchItemCount: renderResult.tileAccumulationPayloadSummary.maxBatchItemCount ?? 0,
          payloadFloatCount: renderResult.tileAccumulationPayloadSummary.payloadFloatCount ?? 0,
          payloadTextureWidth: renderResult.tileAccumulationPayloadSummary.payloadTextureWidth ?? 0,
          payloadTextureHeight: renderResult.tileAccumulationPayloadSummary.payloadTextureHeight ?? 0,
          payloadRowsPerColumn: renderResult.tileAccumulationPayloadSummary.payloadRowsPerColumn ?? 0,
          payloadColumnCount: renderResult.tileAccumulationPayloadSummary.payloadColumnCount ?? 0,
          payloadLayoutReason: renderResult.tileAccumulationPayloadSummary.payloadLayoutReason ?? 'none',
          payloadLayoutValid: !!renderResult.tileAccumulationPayloadSummary.payloadLayoutValid,
          payloadLayoutFailureReason: renderResult.tileAccumulationPayloadSummary.payloadLayoutFailureReason ?? 'none',
          maxTextureSize: renderResult.tileAccumulationPayloadSummary.maxTextureSize ?? 0
        }
      : null
  };
  const mappedProbeCollectionsSummary = buildMappedProbeCollectionsSummary(summary, summary.canvasSizeSummary);
  summary.mappedProbeCollectionsSummary = mappedProbeCollectionsSummary;
  summary.mappedProbeCollectionSummary = mappedProbeCollectionsSummary.collectionSummary ?? null;
  summary.screenshotProbeList = mappedProbeCollectionsSummary.screenshotProbeList;
  summary.mappedProbeList = mappedProbeCollectionsSummary.mappedProbeList;
  summary.mappedProbePatchSummary = mappedProbeCollectionsSummary.firstMappedProbePatchSummary;
  if (mappedProbeCollectionsSummary.firstMappedProbeSummary) {
    Object.assign(summary, mappedProbeCollectionsSummary.firstMappedProbeSummary);
  }
  summary.mappedProbeSummaryNote = 'top-level mappedProbe* fields reflect the first probe in mappedProbeList; use mappedProbeList for the full multi-probe comparison';
  return summary;
}

function buildEmptyPayloadSourceSummary(candidate) {
  const payloadArray = candidate?.screenSpace?.gpuPackedPayloads;
  return {
    screenSpacePresent: !!candidate?.screenSpace,
    inspectDataPresent: !!candidate?.inspectData,
    payloadArrayPresent: Array.isArray(payloadArray),
    payloadArrayLength: Array.isArray(payloadArray) ? payloadArray.length : 0,
    candidateCount: 0,
    usableCandidateCount: 0,
    glMatchCandidateCount: 0,
    textureCandidateCount: 0,
    distinctKinds: [],
    failureCounts: {
      noPayloadArray: Array.isArray(payloadArray) ? 0 : 1,
      noTexture: 0,
      noGl: 0,
      glMismatch: 0,
      unusable: 0
    }
  };
}

function buildInspectAttemptRecord(candidate, inspection = null) {
  const successful = !!inspection?.ok;
  return {
    requestedSource: candidate?.requestedSource ?? 'auto',
    source: candidate?.source ?? 'none',
    sourceReason: candidate?.reason ?? 'none',
    drawPath: candidate?.actualDrawPath ?? 'none',
    screenSpacePresent: !!candidate?.screenSpace,
    inspectDataPresent: !!candidate?.inspectData,
    ok: successful,
    failureReason: successful
      ? 'none'
      : (inspection?.failureReason ?? ((candidate?.screenSpace || candidate?.inspectData)
        ? 'inspect-not-attempted'
        : 'inspect-source-unavailable')),
    payloadCandidateCount: inspection?.payloadCandidateCount ?? 0,
    payloadSourceSummary: inspection?.payloadSourceSummary ?? buildEmptyPayloadSourceSummary(candidate)
  };
}

function buildActualDrawInspectCandidate(renderResult, actualDrawPath, requestedSource) {
  if (actualDrawPath === 'packed') {
    if (Array.isArray(renderResult?.tileCompositePlan?.batches) && renderResult.tileCompositePlan.batches.length > 0) {
      return {
        requestedSource,
        source: 'tile-composite-packed-batches',
        inspectMethod: 'tile-composite-packed-batches',
        inspectData: renderResult.tileCompositePlan,
        actualDrawPath,
        reason: 'actual-draw-uses-tile-composite-packed-batches'
      };
    }
    return {
      requestedSource,
      source: 'packed-screen-space',
      inspectMethod: 'gpu-packed-texture',
      screenSpace: renderResult?.packedScreenSpace ?? null,
      actualDrawPath,
      reason: 'actual-draw-uses-packed-screen-space'
    };
  }

  if (actualDrawPath === 'gpu-screen') {
    return {
      requestedSource,
      source: 'gpu-screen-source-space',
      inspectMethod: 'gpu-packed-texture',
      screenSpace: renderResult?.gpuScreenSourceInfo?.sourceSpace ?? null,
      actualDrawPath,
      reason: 'actual-draw-uses-gpu-screen-source-space'
    };
  }

  return {
    requestedSource,
    source: 'none',
    inspectMethod: 'unsupported',
    screenSpace: null,
    actualDrawPath,
    reason: `actual-draw-path-${actualDrawPath}-is-not-inspectable`
  };
}

function buildInspectableScreenSpaceCandidates(renderResult, requestedSource = 'auto') {
  const actualDrawPath =
    renderResult?.drawThroughputSummary?.actualDrawPath ??
    renderResult?.drawPathSummary?.actualPath ??
    'none';
  const packedCandidate = {
    requestedSource,
    source: 'packed-screen-space',
    screenSpace: renderResult?.packedScreenSpace ?? null,
    actualDrawPath
  };
  const gpuScreenFallbackCandidate = {
    requestedSource,
    source: actualDrawPath === 'gpu-screen'
      ? 'gpu-screen-source-space'
      : 'gpu-screen-source-space-fallback',
    screenSpace: renderResult?.gpuScreenSourceInfo?.sourceSpace ?? null,
    actualDrawPath
  };

  if (requestedSource === 'packed') {
    return [{
      ...packedCandidate,
      reason: actualDrawPath === 'packed'
        ? 'explicit-packed-source-matches-actual-draw'
        : 'explicit-packed-source'
    }];
  }

  if (requestedSource === 'gpu-screen-fallback') {
    return [{
      ...gpuScreenFallbackCandidate,
      reason: actualDrawPath === 'gpu-screen'
        ? 'explicit-gpu-screen-source-matches-actual-draw'
        : 'explicit-gpu-screen-fallback-source'
    }];
  }

  if (requestedSource === 'actual-draw') {
    return [buildActualDrawInspectCandidate(renderResult, actualDrawPath, requestedSource)];
  }

  if (actualDrawPath === 'gpu-screen') {
    return [
      {
        ...gpuScreenFallbackCandidate,
        reason: 'auto-prefers-actual-gpu-screen-source-space'
      },
      {
        ...packedCandidate,
        reason: 'auto-fallback-to-packed-screen-space'
      }
    ];
  }

  if (actualDrawPath === 'packed') {
    return [
      {
        ...packedCandidate,
        reason: 'auto-prefers-actual-packed-screen-space'
      },
      {
        ...gpuScreenFallbackCandidate,
        reason: 'auto-fallback-to-gpu-screen-source-space'
      }
    ];
  }

  return [
    {
      ...packedCandidate,
      reason: 'auto-packed-screen-space'
    },
    {
      ...gpuScreenFallbackCandidate,
      reason: 'auto-gpu-screen-source-space'
    }
  ];
}

function buildInspectResultBase({
  inspection,
  renderResult,
  attempts,
  requestedSource,
  outputMode,
  inspectedCandidate
}) {
  const deterministicState = buildDeterministicStateSummary();
  const renderResultSummary = buildRenderResultInspectionSummary(renderResult);
  const actualDrawPath = renderResultSummary.actualDrawPath ?? 'none';
  const executionSummary = renderResultSummary.executionSummary ?? null;
  const gpuFallbackSummary = renderResultSummary.gpuFallbackSummary ?? null;
  const gpuCompatibilityBridgeSummary = renderResultSummary.gpuCompatibilityBridgeSummary ?? null;
  const actualDrawCandidate = buildActualDrawInspectCandidate(renderResult, actualDrawPath, 'actual-draw');
  const actualDrawInspectSupported = actualDrawCandidate?.inspectMethod === 'tile-composite-packed-batches' ||
    actualDrawCandidate?.inspectMethod === 'gpu-packed-texture';
  const actualDrawAttempt = attempts.find((attempt) => attempt?.requestedSource === 'actual-draw') ?? null;
  const actualDrawInspectFailureReason = actualDrawInspectSupported
    ? (actualDrawAttempt?.failureReason ?? (inspection?.ok ? 'none' : inspection?.failureReason ?? 'none'))
    : (actualDrawCandidate?.reason ?? 'actual-draw-inspect-unsupported');
  const sharedRepresentativeProbe = buildSharedRepresentativeFramebufferProbe(renderResultSummary);
  const canvasSizeSummary = buildCanvasSizeSummary();
  const mappedProbeCollectionsSummary = renderResultSummary.mappedProbeCollectionsSummary ??
    buildMappedProbeCollectionsSummary(renderResultSummary, canvasSizeSummary);
  renderResultSummary.mappedProbeCollectionsSummary = mappedProbeCollectionsSummary;
  renderResultSummary.mappedProbeCollectionSummary = mappedProbeCollectionsSummary.collectionSummary ?? null;
  if (renderResultSummary.executionSummary) {
    renderResultSummary.executionSummary.mappedProbeCollectionSummary = mappedProbeCollectionsSummary.collectionSummary ?? null;
    renderResultSummary.executionSummary.screenshotProbeList = mappedProbeCollectionsSummary.screenshotProbeList;
    renderResultSummary.executionSummary.mappedProbeList = mappedProbeCollectionsSummary.mappedProbeList;
  }
  const firstMappedProbeSummary = mappedProbeCollectionsSummary.firstMappedProbeSummary ?? null;
  const firstMappedProbePatchSummary = mappedProbeCollectionsSummary.firstMappedProbePatchSummary ?? null;
  if (firstMappedProbeSummary) {
    Object.assign(renderResultSummary, firstMappedProbeSummary);
  }
  if (firstMappedProbePatchSummary) {
    renderResultSummary.mappedProbePatchSummary = firstMappedProbePatchSummary;
    Object.assign(renderResultSummary, firstMappedProbePatchSummary);
  }

  return {
    ok: !!inspection?.ok,
    failureReason: inspection?.failureReason ?? 'none',
    inspectSourceRequested: requestedSource,
    inspectJsonMode: outputMode,
    actualDrawPath,
    drawPath:
      inspection?.drawPath ??
      actualDrawPath,
    inspectedSourceSpace: inspectedCandidate?.source ?? 'none',
    inspectedSourceReason: inspectedCandidate?.reason ?? 'none',
    tileCompositePathRequested: getRequestedTileCompositePath(),
    tileCompositePathActual: executionSummary?.tileCompositePath ?? 'none',
    tileCompositePrimitiveRequested: getRequestedTileCompositePrimitive(),
    tileCompositePrimitiveActual: executionSummary?.tileCompositePrimitive ?? 'none',
    tileCompositePrimitivePolicy: executionSummary?.tileCompositePrimitivePolicy ?? 'none',
    tileCompositePrimitivePolicyReason: executionSummary?.tileCompositePrimitivePolicyReason ?? 'none',
    tileCompositePrimitiveResolved: executionSummary?.tileCompositePrimitiveResolved ?? 'none',
    tileCompositeRectContract: executionSummary?.tileCompositeRectContract ?? 'none',
    tileCompositeAccumulationTargetFormat: executionSummary?.tileCompositeAccumulationTargetFormat ?? 'none',
    tileCompositeAccumulationTargetType: executionSummary?.tileCompositeAccumulationTargetType ?? 'none',
    tileCompositeAccumulationTargetPrecisionContract: executionSummary?.tileCompositeAccumulationTargetPrecisionContract ?? 'none',
    tileCompositeAccumulationTargetFallbackReason: executionSummary?.tileCompositeAccumulationTargetFallbackReason ?? 'none',
    tileCompositeResolveContract: executionSummary?.tileCompositeResolveContract ?? 'none',
    tileCompositeContract:
      executionSummary?.compositingContract ??
      renderResultSummary.tileCompositeSummary?.compositingContract ??
      renderResult?.tileCompositePlan?.summary?.compositingContract ??
      'none',
    actualDrawInspectSupported,
    actualDrawInspectDataSource: actualDrawCandidate?.source ?? 'none',
    actualDrawInspectFailureReason,
    gpuFallbackActive: !!gpuFallbackSummary?.active,
    gpuCompatibilityBridgeActive: !!gpuCompatibilityBridgeSummary?.active,
    drawPathSummary: renderResultSummary.drawPathSummary,
    drawThroughputSummary: renderResultSummary.drawThroughputSummary,
    canvasSizeSummary,
    mappedProbeCollectionsSummary,
    mappedProbeCollectionSummary: mappedProbeCollectionsSummary.collectionSummary ?? null,
    screenshotProbeList: mappedProbeCollectionsSummary.screenshotProbeList,
    mappedProbeList: mappedProbeCollectionsSummary.mappedProbeList,
    ...firstMappedProbeSummary,
    ...firstMappedProbePatchSummary,
    deterministicState: buildSlimDeterministicStateSummary(deterministicState),
    attemptedSources: attempts,
    ...sharedRepresentativeProbe,
    lastRenderResultSummary: renderResultSummary
  };
}

function buildSlimInspectResult({
  inspection,
  renderResult,
  attempts,
  requestedSource,
  outputMode,
  inspectedCandidate
}) {
  const base = buildInspectResultBase({
    inspection,
    renderResult,
    attempts,
    requestedSource,
    outputMode,
    inspectedCandidate
  });

  if (!inspection?.ok) {
    return base;
  }

  const executionSummary = base?.lastRenderResultSummary?.executionSummary ?? null;
  const accumulationMaxItems = Number.isFinite(executionSummary?.accumulationMaxItemsPerTile)
    ? Math.max(0, executionSummary.accumulationMaxItemsPerTile | 0)
    : 0;
  const targetTileSplatCount = Number.isFinite(inspection.tileCompositeTileSplatCount)
    ? Math.max(0, inspection.tileCompositeTileSplatCount | 0)
    : 0;
  const targetLocalOrder = Number.isFinite(inspection.tileCompositeLocalOrder)
    ? Math.max(0, inspection.tileCompositeLocalOrder | 0)
    : 0;
  const centerSampleContext = inspection?.sampleContexts?.center ?? null;
  const targetVisitedItems = Number.isFinite(centerSampleContext?.accumulationVisitedItems)
    ? Math.max(0, centerSampleContext.accumulationVisitedItems | 0)
    : (accumulationMaxItems > 0 ? Math.min(targetTileSplatCount, accumulationMaxItems) : 0);
  const targetSkippedItems = Math.max(0, targetTileSplatCount - targetVisitedItems);
  const targetTileTruncated = targetSkippedItems > 0;
  const targetIncludedInLoopWindow = typeof centerSampleContext?.accumulationTargetReached === 'boolean'
    ? !!centerSampleContext.accumulationTargetReached
    : targetLocalOrder < targetVisitedItems;
  const targetSkippedNearerCount = Number.isFinite(centerSampleContext?.accumulationTargetSkippedByEarlyOutCount)
    ? Math.max(0, centerSampleContext.accumulationTargetSkippedByEarlyOutCount | 0)
    : (targetIncludedInLoopWindow ? 0 : Math.max(0, targetLocalOrder + 1 - targetVisitedItems));

  return {
    ...base,
    requestedIndex: inspection.requestedIndex,
    payloadIndex: inspection.payloadIndex,
    localIndex: inspection.localIndex,
    payloadKind: inspection.payloadKind,
    payloadCount: inspection.payloadCount,
    payloadWidth: inspection.payloadWidth,
    payloadHeight: inspection.payloadHeight,
    rowsPerColumn: inspection.rowsPerColumn,
    columnCount: inspection.columnCount,
    tileCompositeBatchIndex: inspection.tileCompositeBatchIndex,
    tileCompositeTileId: inspection.tileCompositeTileId,
    tileCompositeSourceVisibleIndex: inspection.tileCompositeSourceVisibleIndex,
    tileCompositeSourceSplatIndex: inspection.tileCompositeSourceSplatIndex,
    tileCompositeLocalOrder: inspection.tileCompositeLocalOrder,
    tileCompositeTileSplatCount: inspection.tileCompositeTileSplatCount,
    tileCompositeNearerNeighborCount: inspection.tileCompositeNearerNeighborCount,
    tileCompositeFartherNeighborCount: inspection.tileCompositeFartherNeighborCount,
    tileCompositeOverlappingNeighborCount: inspection.tileCompositeOverlappingNeighborCount,
    tileCompositeOverlappingNearerNeighborCount: inspection.tileCompositeOverlappingNearerNeighborCount,
    tileCompositeOverlappingFartherNeighborCount: inspection.tileCompositeOverlappingFartherNeighborCount,
    overlappingNearerAlphaSum: inspection.overlappingNearerAlphaSum,
    overlappingFartherAlphaSum: inspection.overlappingFartherAlphaSum,
    overlappingNearerCountAboveThreshold: inspection.overlappingNearerCountAboveThreshold,
    overlappingFartherCountAboveThreshold: inspection.overlappingFartherCountAboveThreshold,
    overlappingNearerRectOverlapAreaSum: inspection.overlappingNearerRectOverlapAreaSum,
    overlappingFartherRectOverlapAreaSum: inspection.overlappingFartherRectOverlapAreaSum,
    overlappingNearerRectOverlapRatioToTargetSum: inspection.overlappingNearerRectOverlapRatioToTargetSum,
    overlappingFartherRectOverlapRatioToTargetSum: inspection.overlappingFartherRectOverlapRatioToTargetSum,
    overlappingNearerDepthSpread: inspection.overlappingNearerDepthSpread,
    overlappingFartherDepthSpread: inspection.overlappingFartherDepthSpread,
    estimatedNearerTransmittanceAtCenter: inspection.estimatedNearerTransmittanceAtCenter,
    estimatedNearerAlphaCompositeAtCenter: inspection.estimatedNearerAlphaCompositeAtCenter,
    centerSamplePixelIndexPx: inspection.centerSamplePixelIndexPx,
    centerSampleCoordinateSpace: inspection.centerSampleCoordinateSpace,
    centerSampleAlignmentOk: inspection.centerSampleAlignmentOk,
    targetCenterPixelIndexPx: inspection.targetCenterPixelIndexPx,
    targetCenterPixelIndexSpace: inspection.targetCenterPixelIndexSpace,
    targetCenterPixelAlignmentReason: inspection.targetCenterPixelAlignmentReason,
    centerSampleNearerAlphaSum: inspection.centerSampleNearerAlphaSum,
    centerSampleFartherAlphaSum: inspection.centerSampleFartherAlphaSum,
    centerSampleOverlappingNearerAlphaSum: inspection.centerSampleNearerAlphaSum,
    centerSampleOverlappingFartherAlphaSum: inspection.centerSampleFartherAlphaSum,
    centerSampleNearerContributorCount: inspection.centerSampleNearerContributorCount,
    centerSampleFartherContributorCount: inspection.centerSampleFartherContributorCount,
    nearerContributorsAtCenterTopK: inspection.nearerContributorsAtCenterTopK,
    fartherContributorsAtCenterTopK: inspection.fartherContributorsAtCenterTopK,
    sampleContexts: inspection.sampleContexts,
    overlappingNearerTopKSummary: inspection.overlappingNearerTopKSummary,
    overlappingFartherTopKSummary: inspection.overlappingFartherTopKSummary,
    tileCompositeLocalOrderFraction: inspection.tileCompositeLocalOrderFraction,
    tileCompositeOrderBucket: inspection.tileCompositeOrderBucket,
    tileCompositeOverlapContext: inspection.tileCompositeOverlapContext,
    tileCompositeDepthOrderSummary: inspection.tileCompositeDepthOrderSummary,
    tileCompositeTileSummary: inspection.tileCompositeTileSummary,
    tileAccumulationMaxItems: accumulationMaxItems,
    tileAccumulationTruncatedTileCount: executionSummary?.tileAccumulationTruncatedTileCount ?? 0,
    tileAccumulationMaxObservedTileItems: executionSummary?.tileAccumulationMaxObservedTileItems ?? 0,
    tileAccumulationTotalSkippedItems: executionSummary?.tileAccumulationTotalSkippedItems ?? 0,
    tileAccumulationWorstTileId: executionSummary?.tileAccumulationWorstTileId ?? -1,
    tileAccumulationWorstTileItemCount: executionSummary?.tileAccumulationWorstTileItemCount ?? 0,
    tileAccumulationWorstTileSkippedCount: executionSummary?.tileAccumulationWorstTileSkippedCount ?? 0,
    tileAccumulationEarlyOutEnabled: !!executionSummary?.tileAccumulationEarlyOutEnabled,
    tileAccumulationEarlyOutThreshold: executionSummary?.tileAccumulationEarlyOutThreshold ?? 0,
    tileAccumulationEarlyOutTriggeredTileCount: executionSummary?.tileAccumulationEarlyOutTriggeredTileCount ?? 0,
    tileAccumulationEarlyOutTriggeredPixelEstimate: executionSummary?.tileAccumulationEarlyOutTriggeredPixelEstimate ?? 0,
    tileAccumulationWorstEarlyOutTileId: executionSummary?.tileAccumulationWorstEarlyOutTileId ?? -1,
    tileAccumulationWorstEarlyOutCount: executionSummary?.tileAccumulationWorstEarlyOutCount ?? 0,
    tileAccumulationAverageVisitedItemsPerTile: executionSummary?.tileAccumulationAverageVisitedItemsPerTile ?? 0,
    tileAccumulationMaxVisitedItemsPerTile: executionSummary?.tileAccumulationMaxVisitedItemsPerTile ?? 0,
    tileAccumulationAverageVisitedItemsPerPixelEstimate: executionSummary?.tileAccumulationAverageVisitedItemsPerPixelEstimate ?? 0,
    tileAccumulationVisitedRatioSummary: executionSummary?.tileAccumulationVisitedRatioSummary ?? null,
    tileAccumulationObservedTileSummaries: executionSummary?.tileAccumulationObservedTileSummaries ?? [],
    tileAccumulationOrderingSummary: executionSummary?.tileAccumulationOrderingSummary ?? null,
    tileAccumulationBatchBoundarySummary: executionSummary?.tileAccumulationBatchBoundarySummary ?? null,
    tileAccumulationObservedOrderingMismatches: executionSummary?.tileAccumulationObservedOrderingMismatches ?? [],
    tileAccumulationHeavyTileSummaries: executionSummary?.tileAccumulationHeavyTileSummaries ?? [],
    tileAccumulationRepresentativeTileId: executionSummary?.tileAccumulationRepresentativeTileId ?? -1,
    tileAccumulationRepresentativeTileItemCount: executionSummary?.tileAccumulationRepresentativeTileItemCount ?? 0,
    tileAccumulationRepresentativeTileOrderPreview: executionSummary?.tileAccumulationRepresentativeTileOrderPreview ?? null,
    tileAccumulationRepresentativeTileDepthPreview: executionSummary?.tileAccumulationRepresentativeTileDepthPreview ?? null,
    tileAccumulationRepresentativeTileBatchSpan: executionSummary?.tileAccumulationRepresentativeTileBatchSpan ?? 1,
    tileAccumulationRepresentativeTileSequenceConsistent: !!executionSummary?.tileAccumulationRepresentativeTileSequenceConsistent,
    tileAccumulationContributionSummary: executionSummary?.tileAccumulationContributionSummary ?? null,
    tileAccumulationRepresentativeSampleMode: executionSummary?.tileAccumulationRepresentativeSampleMode ?? 'none',
    tileAccumulationRepresentativeSampleSelectionMode: executionSummary?.tileAccumulationRepresentativeSampleSelectionMode ?? 'none',
    tileAccumulationRepresentativeSampleSelectionReason: executionSummary?.tileAccumulationRepresentativeSampleSelectionReason ?? 'none',
    tileAccumulationRepresentativeSamplePixel: executionSummary?.tileAccumulationRepresentativeSamplePixel ?? [0, 0],
    tileAccumulationRepresentativeSampleHasContribution: !!executionSummary?.tileAccumulationRepresentativeSampleHasContribution,
    tileAccumulationRepresentativeSampleCandidateCount: executionSummary?.tileAccumulationRepresentativeSampleCandidateCount ?? 0,
    tileAccumulationRepresentativeSampleEvaluatedCandidateCount: executionSummary?.tileAccumulationRepresentativeSampleEvaluatedCandidateCount ?? 0,
    tileAccumulationRepresentativeSampleUsableItemSource: executionSummary?.tileAccumulationRepresentativeSampleUsableItemSource ?? 'none',
    tileAccumulationRepresentativeSampleItemReadMode: executionSummary?.tileAccumulationRepresentativeSampleItemReadMode ?? 'none',
    tileAccumulationRepresentativeSampleEvaluatedItemCount: executionSummary?.tileAccumulationRepresentativeSampleEvaluatedItemCount ?? 0,
    tileAccumulationRepresentativeSampleContributionLog: executionSummary?.tileAccumulationRepresentativeSampleContributionLog ?? [],
    tileAccumulationRepresentativeSampleFinalT: executionSummary?.tileAccumulationRepresentativeSampleFinalT ?? 1,
    tileAccumulationRepresentativeSampleAccumColor: executionSummary?.tileAccumulationRepresentativeSampleAccumColor ?? [0, 0, 0],
    tileAccumulationRepresentativeSampleResolvedColor: executionSummary?.tileAccumulationRepresentativeSampleResolvedColor ?? [0, 0, 0],
    tileAccumulationRepresentativeSampleContributionCount: executionSummary?.tileAccumulationRepresentativeSampleContributionCount ?? 0,
    tileAccumulationRepresentativeSampleAlphaSum: executionSummary?.tileAccumulationRepresentativeSampleAlphaSum ?? 0,
    tileAccumulationRepresentativeSampleContributionSum: executionSummary?.tileAccumulationRepresentativeSampleContributionSum ?? [0, 0, 0],
    tileAccumulationRepresentativeSampleLastContributedLocalOrder: executionSummary?.tileAccumulationRepresentativeSampleLastContributedLocalOrder ?? -1,
    tileAccumulationRepresentativeSampleThresholdCrossingCount: executionSummary?.tileAccumulationRepresentativeSampleThresholdCrossingCount ?? 0,
    tileAccumulationRepresentativeSampleThresholdSkippedCount: executionSummary?.tileAccumulationRepresentativeSampleThresholdSkippedCount ?? 0,
    tileAccumulationRepresentativeSampleFirstThresholdSkipLocalOrder: executionSummary?.tileAccumulationRepresentativeSampleFirstThresholdSkipLocalOrder ?? -1,
    tileAccumulationRepresentativeSampleFirstThresholdSkipAlpha: executionSummary?.tileAccumulationRepresentativeSampleFirstThresholdSkipAlpha ?? 0,
    tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceBefore: executionSummary?.tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceBefore ?? 1,
    tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceAfter: executionSummary?.tileAccumulationRepresentativeSampleFirstThresholdSkipTransmittanceAfter ?? 1,
    tileAccumulationRepresentativeSampleThresholdSkipPreview: executionSummary?.tileAccumulationRepresentativeSampleThresholdSkipPreview ?? [],
    tileAccumulationRepresentativeSampleThresholdSemantics: executionSummary?.tileAccumulationRepresentativeSampleThresholdSemantics ?? null,
    tileAccumulationRepresentativeSampleFramebufferColor: executionSummary?.tileAccumulationRepresentativeSampleFramebufferColor ?? [0, 0, 0],
    tileAccumulationRepresentativeSampleFramebufferReadbackValid: !!executionSummary?.tileAccumulationRepresentativeSampleFramebufferReadbackValid,
    tileAccumulationRepresentativeSampleFramebufferReadbackReason: executionSummary?.tileAccumulationRepresentativeSampleFramebufferReadbackReason ?? 'not-attempted',
    tileAccumulationRepresentativeSampleFramebufferReadbackPixel: executionSummary?.tileAccumulationRepresentativeSampleFramebufferReadbackPixel ?? [0, 0],
    tileAccumulationRepresentativeSampleFramebufferReadbackGlPixel: executionSummary?.tileAccumulationRepresentativeSampleFramebufferReadbackGlPixel ?? [0, 0],
    tileAccumulationRepresentativeSampleFramebufferReadbackRgba8: executionSummary?.tileAccumulationRepresentativeSampleFramebufferReadbackRgba8 ?? [0, 0, 0, 0],
    tileAccumulationRepresentativeSampleResolvedColorDelta: executionSummary?.tileAccumulationRepresentativeSampleResolvedColorDelta ?? [0, 0, 0],
    tileAccumulationRepresentativeSampleResolvedColorDeltaAbsMax: executionSummary?.tileAccumulationRepresentativeSampleResolvedColorDeltaAbsMax ?? 0,
    tileAccumulationRepresentativeSampleResolvedColorMatchesFramebuffer: !!executionSummary?.tileAccumulationRepresentativeSampleResolvedColorMatchesFramebuffer,
    tileAccumulationRepresentativeSampleResolvedColorMatchTolerance: executionSummary?.tileAccumulationRepresentativeSampleResolvedColorMatchTolerance ?? 0,
    tileAccumulationContractVersion: executionSummary?.tileAccumulationContractVersion ?? 'none',
    tileAccumulationTruncationRatio: executionSummary?.tileAccumulationTruncationRatio ?? 0,
    tileAccumulationTargetTileTruncated: targetTileTruncated,
    tileAccumulationTargetTileVisitedItems: targetVisitedItems,
    tileAccumulationTargetTileSkippedItems: targetSkippedItems,
    tileAccumulationTargetIncludedInLoopWindow: targetIncludedInLoopWindow,
    tileAccumulationTargetSkippedNearerCount: targetSkippedNearerCount,
    tileAccumulationTargetTileEarlyOutTriggered: !!centerSampleContext?.accumulationEarlyOutTriggered,
    tileAccumulationTargetTileEarlyOutAtItem: centerSampleContext?.accumulationEarlyOutAtItem ?? -1,
    tileAccumulationTargetTileEarlyOutAtTransmittance: centerSampleContext?.accumulationEarlyOutAtTransmittance ?? 1,
    tileAccumulationTargetTileEarlyOutBeforeTarget: !!centerSampleContext?.accumulationEarlyOutBeforeTarget,
    tileAccumulationTargetTileContributingItems: centerSampleContext?.accumulationContributingItems ?? 0,
    tileCompositeTargetTileHeavy: !!inspection.tileCompositeTargetTileHeavy,
    tileCompositeTargetTileBatchSpan: inspection.tileCompositeTargetTileBatchSpan ?? 1,
    tileCompositeTargetSequenceConsistent: !!inspection.tileCompositeTargetSequenceConsistent,
    tileCompositeTargetOrderingMismatchCount: inspection.tileCompositeTargetOrderingMismatchCount ?? 0,
    tileCompositeTargetOrderingFirstMismatch: inspection.tileCompositeTargetOrderingFirstMismatch ?? null,
    tileCompositeTargetSequencePreview: inspection.tileCompositeTargetSequencePreview ?? [],
    textureColumnIndex: inspection.textureColumnIndex,
    textureRowIndex: inspection.textureRowIndex,
    textureXBase: inspection.textureXBase,
    centerPx: inspection.centerPx,
    depth: inspection.depth,
    payloadRadius: inspection.payloadRadius,
    colorAlpha: inspection.colorAlpha,
    conic: inspection.conic,
    unclampedPointSize: inspection.unclampedPointSize,
    clampedPointSize: inspection.clampedPointSize,
    clampApplied: inspection.clampApplied,
    rasterRectMinPx: inspection.rasterRectMinPx,
    rasterRectMaxPxExclusive: inspection.rasterRectMaxPxExclusive,
    rasterWidthPx: inspection.rasterWidthPx,
    rasterHeightPx: inspection.rasterHeightPx,
    rasterPixelArea: inspection.rasterPixelArea,
    rasterCoverageOvershootEstimate: inspection.rasterCoverageOvershootEstimate,
    rasterCoverageOvershootRatio: inspection.rasterCoverageOvershootRatio,
    fragmentSamples: inspection.fragmentSamples
  };
}

function buildFullInspectResult({
  inspection,
  renderResult,
  attempts,
  requestedSource,
  outputMode,
  inspectedCandidate
}) {
  const base = buildInspectResultBase({
    inspection,
    renderResult,
    attempts,
    requestedSource,
    outputMode,
    inspectedCandidate
  });

  return {
    ...inspection,
    ...base,
    debugText: refreshLatestDebugText(),
    lastRenderResult: renderResult
  };
}

function convertDatasetTransformMatrixToViewerPose(rawMatrix, convention = 'nerf-blender-c2w') {
  if (!Array.isArray(rawMatrix) || rawMatrix.length !== 4) return null;
  const c2w = rawMatrix.map((row) => (Array.isArray(row) ? row.slice(0, 4).map(Number) : []));
  if (c2w.some((row) => row.length !== 4 || row.some((value) => !Number.isFinite(value)))) {
    return null;
  }

  if (convention === 'nerf-blender-c2w') {
    for (let row = 0; row < 3; row++) {
      c2w[row][1] *= -1;
      c2w[row][2] *= -1;
    }
  }

  const position = [c2w[0][3], c2w[1][3], c2w[2][3]];
  const forward = [c2w[0][2], c2w[1][2], c2w[2][2]];
  const up = [c2w[0][1], c2w[1][1], c2w[2][1]];
  const targetDistance = 10.0;
  const target = [
    position[0] + forward[0] * targetDistance,
    position[1] + forward[1] * targetDistance,
    position[2] + forward[2] * targetDistance
  ];

  return {
    position,
    target,
    up,
    forward,
    targetDistance,
    convertedMatrix: c2w
  };
}

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function length3(v) {
  return Math.hypot(v[0], v[1], v[2]);
}

function normalize3(v) {
  const len = length3(v);
  return len > 1e-8 ? [v[0] / len, v[1] / len, v[2] / len] : null;
}

function projectOntoViewPlane(axis, forward) {
  const d = dot3(axis, forward);
  return normalize3([
    axis[0] - forward[0] * d,
    axis[1] - forward[1] * d,
    axis[2] - forward[2] * d
  ]);
}

function buildInteractiveCameraPoseFromReferencePose(referencePose, orientationPolicy = 'roll-free-reference-screen-up') {
  if (!referencePose || !Array.isArray(referencePose.forward) || !Array.isArray(referencePose.up)) {
    return null;
  }
  if (orientationPolicy !== 'roll-free-reference-screen-up') {
    return referencePose;
  }

  const forward = normalize3(referencePose.forward);
  const referenceScreenUp = normalize3(referencePose.up);
  if (!forward || !referenceScreenUp) return referencePose;

  const candidateAxes = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1]
  ];
  let best = null;
  for (const axis of candidateAxes) {
    const projected = projectOntoViewPlane(axis, forward);
    if (!projected) continue;
    const alignment = dot3(projected, referenceScreenUp);
    if (!best || alignment > best.alignment) {
      best = { axis, projected, alignment };
    }
  }
  if (!best) return referencePose;

  return {
    ...referencePose,
    up: [...best.axis],
    referenceUp: [...referencePose.up],
    orientationPolicy,
    screenAxisMapping: {
      source: 'reference-camera-pose',
      selectedWorldUpAxis: [...best.axis],
      projectedScreenUp: [...best.projected],
      referenceScreenUp: [...referenceScreenUp],
      alignment: best.alignment
    }
  };
}

function applyDeterministicDatasetCameraPose() {
  if (!raw) return false;
  const referencePose = convertDatasetTransformMatrixToViewerPose(
    deterministicQueryState.datasetTransformMatrix,
    deterministicQueryState.datasetCameraConvention ?? 'nerf-blender-c2w'
  );
  const useInteractiveReferenceContract =
    deterministicQueryState.cameraControlContract === 'interactive-from-reference' &&
    deterministicQueryState.datasetViewMatrixMode !== 'cuda-aligned';
  const convertedPose = useInteractiveReferenceContract
    ? buildInteractiveCameraPoseFromReferencePose(
        referencePose,
        deterministicQueryState.cameraOrientationPolicy ?? 'roll-free-reference-screen-up'
      )
    : referencePose;
  const position = Array.isArray(convertedPose?.position)
    ? convertedPose.position
    : deterministicQueryState.datasetCameraPosition;
  const target = Array.isArray(convertedPose?.target)
    ? convertedPose.target
    : deterministicQueryState.datasetCameraTarget;
  if (!Array.isArray(position) || position.length < 3 || !Array.isArray(target) || target.length < 3) {
    return false;
  }

  camera.position.set(Number(position[0]), Number(position[1]), Number(position[2]));
  controls.target.set(Number(target[0]), Number(target[1]), Number(target[2]));

  const up = Array.isArray(convertedPose?.up)
    ? convertedPose.up
    : deterministicQueryState.datasetCameraUp;
  if (Array.isArray(up) && up.length >= 3) {
    camera.up.set(Number(up[0]), Number(up[1]), Number(up[2]));
  } else {
    camera.up.set(0, 1, 0);
  }

  const fovYRadians = Number.isFinite(deterministicQueryState.datasetCameraFoVyRad)
    ? Number(deterministicQueryState.datasetCameraFoVyRad)
    : Number(deterministicQueryState.datasetCameraFoVy);
  if (Number.isFinite(fovYRadians) && fovYRadians > 0) {
    camera.fov = fovYRadians * 180 / Math.PI;
  }

  camera.lookAt(Number(target[0]), Number(target[1]), Number(target[2]));
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  controls.update();
  appliedCameraPresetName = deterministicQueryState.datasetCameraLabel ?? deterministicQueryState.datasetImageName ?? 'dataset-camera';
  return true;
}

function applyDeterministicCameraPreset() {
  if (!raw) return false;

  if (applyDeterministicDatasetCameraPose()) {
    return true;
  }

  const preset = deterministicQueryState.cameraPreset;
  if (!preset || preset.name === 'fit') {
    fitCameraToRaw(raw, controls, camera);
    appliedCameraPresetName = preset?.name ?? 'none';
    return !!preset;
  }

  const applied = applyViewerCameraPresetState(camera, controls, preset);
  if (!applied) {
    fitCameraToRaw(raw, controls, camera);
    appliedCameraPresetName = 'fit';
    return false;
  }

  appliedCameraPresetName = preset.name;
  return true;
}

function applyDeterministicUiState() {
  const appliedState = applyViewerQueryStateToUi(ui, deterministicQueryState);
  if (deterministicQueryState.active) {
    updateDrawPathNoteFromState(appliedState);
  }
  updateDeterministicStateNote();
  return appliedState;
}

function sanitizeSnapshotFileName(name) {
  const trimmed = typeof name === 'string' ? name.trim() : '';
  const baseName = trimmed || `gpu-step70-${appliedCameraPresetName || 'view'}`;
  return sanitizePngFileName(baseName);
}

async function captureBlobFromCanvas(sourceCanvas, fileName, download) {
  return await captureCanvasPngBlob(sourceCanvas, fileName, download);
}

function isWebGpuExclusiveViewerLifecycleRequested(options = {}) {
  const requestedBackendMode =
    options.webgpuBackendMode ??
    deterministicQueryState.webgpuBackendMode ??
    'webgl2-fallback';
  const allowViewerCanvasPresentation =
    options.webgpuAllowViewerCanvasPresentation === true ||
    deterministicQueryState.webgpuAllowViewerCanvasPresentation === true;
  return shouldUseWebGpuExclusiveFrameLifecycle({
    requestedBackendMode,
    allowViewerCanvasPresentation
  });
}

async function saveCurrentCanvasPng(options = {}) {
  const input = typeof options === 'string' ? { name: options } : (options ?? {});
  const download = input.download !== false;
  const fileName = sanitizeSnapshotFileName(input.name ?? 'gpu-viewer-current-canvas.png');
  if (input.renderBeforeCapture) {
    await renderCurrentFrame();
  }
  const framebufferStats = sampleCurrentFramebufferStats(input.framebufferStats ?? {});
  console.info('[gpuViewerDebug.saveCurrentCanvasPng] framebufferStats', {
    likelyBlackFrame: framebufferStats?.likelyBlackFrame ?? null,
    nonBlackRatio: framebufferStats?.nonBlackRatio ?? null,
    nonBlackCount: framebufferStats?.nonBlackCount ?? null,
    maxRgb: framebufferStats?.maxRgb ?? null,
    preserveDrawingBuffer: buildCanvasSizeSummary().preserveDrawingBuffer
  });
  const blob = await captureBlobFromCanvas(canvas, fileName, download);
  return {
    blob,
    fileName,
    source: 'canvas-toBlob-current-default-framebuffer',
    framebufferStats,
    canvasSizeSummary: buildCanvasSizeSummary(),
    deterministicState: buildDeterministicStateSummary(),
    lastRenderResultSummary: buildRenderResultInspectionSummary(latestRenderResult)
  };
}

async function renderCurrentFrame(options = {}) {
  if (isWebGpuExclusiveViewerLifecycleRequested(options)) {
    const requestedBackendMode =
      options.webgpuBackendMode ??
      deterministicQueryState.webgpuBackendMode ??
      'webgl2-fallback';
    const allowViewerCanvasPresentation =
      options.webgpuAllowViewerCanvasPresentation === true ||
      deterministicQueryState.webgpuAllowViewerCanvasPresentation === true;
    const enableViewerLoopHook =
      options.webgpuBackendViewerLoopHook === true ||
      deterministicQueryState.webgpuBackendViewerLoopHook === true;
    const backendImplementationKind =
      options.webgpuBackendImplementation ??
      deterministicQueryState.webgpuBackendImplementation ??
      'webgpu-visible-record-dry-run-runtime';
    camera.updateMatrixWorld(true);
    const deterministicState = buildDeterministicStateSummary();
    const cameraSnapshot = {
      deterministicState: buildSlimDeterministicStateSummary(deterministicState),
      canvasWidth: canvas.width,
      canvasHeight: canvas.height,
      cameraPosition: camera.position.toArray(),
      cameraQuaternion: camera.quaternion.toArray(),
      controlsTarget: controls.target.toArray()
    };
    const viewerCanvasState = {
      provided: true,
      canvas,
      contextMode: 'webgpu-exclusive-lifecycle-requested',
      requestedBackendMode,
      allowViewerCanvasPresentation,
      webgl2FrameLifecycleSuppressed: true
    };
    const webgpuBackendViewerLifecycleIntegrationBoundary = enableViewerLoopHook
      ? buildWebGpuBackendViewerLifecycleIntegrationBoundary({
          requestedBackendMode,
          allowViewerCanvasPresentation,
          enableViewerLoopHook,
          renderLifecycleStage: 'renderCurrentFrame',
          viewerCanvasState,
          cameraSnapshot,
          adapterResult: null,
          adapterInvocationSource: 'viewer-render-lifecycle-hook-boundary'
        })
      : null;
    const backendFrameExecutorResult =
      webgpuBackendViewerLifecycleIntegrationBoundary
        ? await executeWebGpuBackendViewerFrame({
            requestedBackendMode,
            allowViewerCanvasPresentation,
            enableViewerLoopHook,
            invocationSource: 'renderCurrentFrame-viewer-backend-executor',
            frameIndex: Number.isFinite(latestRenderResult?.executionSummary?.frameIndex)
              ? latestRenderResult.executionSummary.frameIndex + 1
              : 0,
            integrationBoundary: webgpuBackendViewerLifecycleIntegrationBoundary,
            cameraSnapshot,
            viewerCanvasState,
            backendImplementationKind,
            runBackendFrame: ({
              executorContract,
              runnerContract,
              backendImplementationKind: selectedBackendImplementationKind,
              implementationContract,
              frameInputContract
            }) =>
              runWebGpuVisibleRecordDryRunFromViewerState({
                options: {
                  ...options,
                  ensureCurrentFrame: false,
                  webgpuBackendMode: requestedBackendMode,
                  webgpuBackendImplementation: backendImplementationKind,
                  webgpuAllowViewerCanvasPresentation: allowViewerCanvasPresentation,
                  webgpuBackendViewerLoopHook: true,
                  webgpuBackendViewerLifecycleInvocationSource:
                    'renderCurrentFrame-viewer-backend-executor',
                  webgpuBackendViewerLifecycleControlledExecution: true,
                  comparisonMode:
                    options.comparisonMode ??
                    (backendImplementationKind ===
                    'webgpu-normal-backend-frame-implementation'
                      ? 'phase3-step63-normal-webgpu-backend-contracts'
                      : 'phase3-step61-viewer-backend-runtime-runner')
                },
                requestedWebGpuBackendMode: requestedBackendMode,
                allowViewerCanvasPresentation,
                enableViewerLoopHook: true,
                useExclusiveWebGpuFrameLifecycle: true,
                debugRender: {
                  renderResult: latestRenderResult,
                  attempts: [
                    {
                      stage: 'viewer-backend-frame-executor-direct-run',
                      reason:
                        'renderCurrentFrame invokes the WebGPU backend runtime runner through the executor boundary'
                    }
                  ]
                },
                metadataOverrides: {
                  captureSource: 'viewer-backend-frame-executor',
                  phase:
                    backendImplementationKind ===
                    'webgpu-normal-backend-frame-implementation'
                      ? 'phase3-step63'
                      : 'phase3-step61',
                  renderLifecycleStage: 'renderCurrentFrame',
                  invocationSource: 'renderCurrentFrame-viewer-backend-executor',
                  controlledExecutionRequested: true,
                  selectedBackendImplementationKind,
                  backendExecutorRequest: {
                    executorContract,
                    runnerContract,
                    implementationContract,
                    frameInputContract
                  }
                }
              })
          })
        : null;
    const webgpuBackendViewerFrameExecutor =
      backendFrameExecutorResult?.summary ?? null;
    const webgpuBackendViewerLifecycleControlledExecution =
      backendFrameExecutorResult?.controlledExecution ??
      (webgpuBackendViewerLifecycleIntegrationBoundary
        ? buildWebGpuBackendViewerLifecycleControlledExecution({
            integrationBoundary: webgpuBackendViewerLifecycleIntegrationBoundary,
            adapterResult: null,
            invocationRequested:
              webgpuBackendViewerLifecycleIntegrationBoundary.integrationBoundaryReady === true,
            invocationSource: 'renderCurrentFrame-viewer-backend-executor-unavailable',
            webgl2FrameLifecycleSuppressed: true,
            cameraSnapshot
          })
        : null);
    const renderResult = {
      status: 'webgpu-exclusive-frame-lifecycle-pending',
      reason:
        'webgpu-exclusive mode owns the viewer canvas lifecycle; WebGL2 render frame is suppressed',
      webgpuExclusiveFrameLifecyclePending: true,
      webgpuBackendViewerLifecycleIntegrationBoundary,
      webgpuBackendViewerLifecycleControlledExecution,
      webgpuBackendViewerFrameExecutor,
      drawPathSummary: {
        requestedPath: 'webgpu-exclusive',
        actualPath:
          webgpuBackendViewerFrameExecutor?.executorReady
            ? 'webgpu-exclusive-viewer-backend-frame-executor'
            : webgpuBackendViewerLifecycleControlledExecution?.controlledExecutionReady
              ? 'webgpu-exclusive-controlled-backend-frame-executed'
              : webgpuBackendViewerLifecycleIntegrationBoundary?.integrationBoundaryReady
                ? 'webgpu-exclusive-viewer-loop-hook-ready'
                : 'webgpu-exclusive-pending'
      },
      executionSummary: {
        backendMode: 'webgpu-exclusive',
        webgl2FrameSuppressed: true,
        displayConnectionImplemented: false,
        frameIndex: webgpuBackendViewerFrameExecutor?.executorContract?.frameIndex ?? 0,
        webgpuBackendViewerLoopHookEnabled: enableViewerLoopHook,
        webgpuBackendViewerLoopHookReady:
          webgpuBackendViewerLifecycleIntegrationBoundary?.integrationBoundaryReady === true,
        webgpuBackendViewerFrameExecutorReady:
          webgpuBackendViewerFrameExecutor?.executorReady === true,
        webgpuBackendImplementation: backendImplementationKind,
        webgpuBackendViewerLifecycleControlledExecutionReady:
          webgpuBackendViewerLifecycleControlledExecution?.controlledExecutionReady === true,
        webgpuBackendViewerLifecycleControlledInvocationCount:
          webgpuBackendViewerLifecycleControlledExecution?.invocationCount ?? 0,
        webgpuBackendViewerLifecycleControlledSubmittedFrameCount:
          webgpuBackendViewerLifecycleControlledExecution?.submittedFrameCount ?? 0
      },
      limitedDrawRuntimeSummary: null
    };
    latestRenderResult = renderResult;
    latestGpuCandidateShadowCompare = null;
    return renderResult;
  }
  ensureGpu();
  const renderResult = await renderGpuFrame({
    raw,
    gpu: getGpu(),
    canvas,
    camera,
    controls,
    ui,
    tokenRef: options.isolatedTokenRef ? { value: 0 } : tokenRef,
    infoEl: ui.info,
    interactionOverride: buildRenderOverrides(),
    deterministicStateSummary: buildDeterministicStateSummary(),
    latestGpuCandidateShadowCompare
  });
  if (renderResult || !options.preservePreviousOnNull) {
    latestRenderResult = renderResult;
  }
  if (renderResult) {
    const shadowOptions = buildGpuCandidateShadowOptionsFromQuery(deterministicQueryState);
    if (isGpuCandidateShadowCompareEnabled(shadowOptions)) {
      latestGpuCandidateShadowCompare = await runGpuCandidateShadowCompareForRender(renderResult, {
        ...shadowOptions,
        trigger: 'render-current-frame',
        comparisonMode: 'gpu-candidate-shadow-compare-runtime-query'
      });
    } else {
      latestGpuCandidateShadowCompare = null;
    }
  } else if (!options.preservePreviousOnNull) {
    latestGpuCandidateShadowCompare = null;
  }

  if (renderResult && typeof renderResult.debugText === 'string') {
    refreshLatestDebugText(renderResult.debugText);
  } else if (renderResult && typeof renderResult.infoText === 'string') {
    refreshLatestDebugText(renderResult.infoText);
  } else if (!options.preservePreviousOnNull) {
    refreshLatestDebugText();
  }

  return renderResult;
}

async function captureFrame(options = {}) {
  const download = options.download !== false;
  const fileName = sanitizeSnapshotFileName(options.name);

  try {
    await renderCurrentFrame();
    const snapshotCanvas = captureSnapshotCanvasFromGpu(getGpu());
    const blob = await captureBlobFromCanvas(snapshotCanvas, fileName, download);
    lastSnapshotSummary = {
      available: true,
      source: 'webgl-default-framebuffer-readpixels',
      renderWaitMode: 'direct-render-await',
      status: 'success',
      reason: 'none'
    };

    return {
      blob,
      fileName,
      source: lastSnapshotSummary.source,
      renderWaitMode: lastSnapshotSummary.renderWaitMode,
      status: lastSnapshotSummary.status,
      reason: lastSnapshotSummary.reason,
      deterministicState: buildDeterministicStateSummary(),
      lastRenderResultSummary: buildRenderResultInspectionSummary(latestRenderResult),
      debugText: refreshLatestDebugText(),
      lastRenderResult: latestRenderResult
    };
  } catch (error) {
    lastSnapshotSummary = {
      available: true,
      source: 'webgl-default-framebuffer-readpixels',
      renderWaitMode: 'direct-render-await',
      status: 'failure',
      reason: error?.message ?? 'unknown-snapshot-error'
    };
    throw error;
  }
}

async function inspectActiveSplat(options = {}) {
  const ensureCurrentFrame = options.ensureCurrentFrame !== false;
  const renderResult = ensureCurrentFrame || !latestRenderResult
    ? await renderCurrentFrame()
    : latestRenderResult;

  const gpu = getGpu();
  const gl = gpu?.gl;
  if (!gl) {
    throw new Error('inspectActiveSplat failed: WebGL renderer is not ready');
  }

  const inspectSource = normalizeInspectSource(
    options.inspectSource ?? deterministicQueryState.inspectSource,
    'auto'
  );
  const outputMode = normalizeInspectJsonMode(
    options.outputMode ?? options.inspectJsonMode ?? deterministicQueryState.inspectJsonMode,
    'slim'
  );
  const candidates = buildInspectableScreenSpaceCandidates(renderResult, inspectSource);
  if (candidates.length <= 0) {
    throw new Error('inspectActiveSplat failed: no inspectable screen-space payloads available');
  }

  const attempts = [];
  for (const candidate of candidates) {
    if (!candidate.screenSpace && !candidate.inspectData) {
      attempts.push(buildInspectAttemptRecord(candidate));
      continue;
    }

    let inspection;
    if (candidate.inspectMethod === 'tile-composite-packed-batches') {
      inspection = inspectPackedInterleavedTileCompositeItem(candidate.inspectData, options);
    } else {
      inspection = inspectGpuPackedPayloadItem(gl, candidate.screenSpace, options);
    }
    const inspectionWithSource = {
      ...inspection,
      drawPath: candidate.actualDrawPath,
      inspectedSourceSpace: candidate.source,
      inspectedSourceReason: candidate.reason
    };
    attempts.push(buildInspectAttemptRecord(candidate, inspection));
    if (inspection.ok) {
      return outputMode === 'full'
        ? buildFullInspectResult({
            inspection: inspectionWithSource,
            renderResult,
            attempts,
            requestedSource: inspectSource,
            outputMode,
            inspectedCandidate: candidate
          })
        : buildSlimInspectResult({
            inspection: inspectionWithSource,
            renderResult,
            attempts,
            requestedSource: inspectSource,
            outputMode,
            inspectedCandidate: candidate
          });
    }
  }

  const failedInspection = {
    ok: false,
    failureReason: 'inspect-no-usable-payload-source'
  };
  return outputMode === 'full'
    ? buildFullInspectResult({
        inspection: failedInspection,
        renderResult,
        attempts,
        requestedSource: inspectSource,
        outputMode,
        inspectedCandidate: null
      })
    : buildSlimInspectResult({
        inspection: failedInspection,
        renderResult,
        attempts,
        requestedSource: inspectSource,
        outputMode,
        inspectedCandidate: null
      });
}

function installViewerDebugApi() {
  window.gpuViewerDebug = {
    captureFrame,
    compareSingleSplat: (input = {}) => computeGaussianDebugState(input),
    inspectActiveSplat,
    inspectActiveSplatSlim: (options = {}) => inspectActiveSplat({
      ...options,
      outputMode: 'slim'
    }),
    inspectActiveSplatFull: (options = {}) => inspectActiveSplat({
      ...options,
      outputMode: 'full'
    }),
    getDefaultSingleSplatCompareInput: () => structuredClone(DEFAULT_SINGLE_SPLAT_COMPARE_INPUT),
    getDeterministicState: () => buildDeterministicStateSummary(),
    getCameraDebugState,
    sampleCurrentFramebufferStats,
    captureCurrentDebugBundle,
    captureRepresentativeActualPayloadDebug,
    inspectPackedPayloadForIndices: captureRepresentativeActualPayloadDebug,
    downloadJsonDebug,
    downloadRepresentativeActualPayloadCompareJson: async (input = {}, fileName = 'step90_representative_actual_payload_compare.json') => {
      const result = await captureRepresentativeActualPayloadDebug(input);
      return {
        result,
        download: downloadJsonDebug(result, fileName)
      };
    },
    saveRepresentativeActualPayloadOverlayPng,
    captureTileAccumulationDebug,
    captureRepresentativePixelAccumulationDebug: captureTileAccumulationDebug,
    saveTileAccumulationDebugOverlayPng,
    captureViewerPayloadIndexAssociationDebug,
    downloadViewerPayloadIndexAssociationDebugJson: async (input = {}, fileName = 'step90_viewer_payload_index_association_debug.json') => {
      const result = await captureViewerPayloadIndexAssociationDebug(input);
      return {
        result,
        download: downloadJsonDebug(result, fileName)
      };
    },
    captureVisibleComparisonDebug,
    captureCandidateComparisonDebug,
    captureCandidateSubsetComparisonDebug,
    captureGpuCandidateSubsetComparisonDebug,
    captureGpuCandidateFilterComparisonDebug,
    captureGpuCandidateDryRunVisibleComparisonDebug,
    captureGpuCandidateScreenCoarseDryRunVisibleComparisonDebug,
    captureGpuCandidateScreenCoarseSweepComparisonDebug,
    captureGpuVisibleRecordDryRunDebug,
    captureGpuRawVisibleRecordDryRunDebug,
    captureWebGpuVisibleRecordDryRunDebug,
    captureGpuCandidateShadowCompareDebug,
    captureGpuCandidateSourceCompareDebug,
    captureGpuCandidateScreenCoarseCompareDebug,
    captureGpuCandidateCoverageDebug,
    captureGpuCandidateRuntimeSummaryDebug: buildGpuCandidateRuntimeDebugSummary,
    captureLiveSameStateTileAndAssociationDebug,
    downloadLiveSameStateTileAndAssociationDebugJson,
    saveCurrentCanvasPng,
    captureCurrentCanvasPng: saveCurrentCanvasPng,
    getLatestDebugText: () => refreshLatestDebugText(),
    getLastRenderResult: () => latestRenderResult,
    getLastGpuCandidateShadowCompare: () => latestGpuCandidateShadowCompare,
    scheduleRender: () => scheduler.scheduleRender()
  };
}

function scheduleRenderAndPersist() {
  const state = readAndSaveUiState(ui);
  updateDrawPathNoteFromState(state);
  scheduler.scheduleRender();
}

function updateDrawPathNoteFromState(stateLike) {
  const summary = summarizeUiState(stateLike);
  if (!ui.drawPathSelectNote) return;

  if (summary.drawPath === 'gpu-screen') {
    ui.drawPathSelectNote.textContent =
      `full-frame only; gpu-screen debug distinguishes actual, source, and reference; tile path=${summary.tileCompositePath}; tile primitive=${summary.tileCompositePrimitive}`;
    return;
  }

  if (summary.drawPath === 'packed') {
    ui.drawPathSelectNote.textContent =
      `full-frame only; packed is the formal reference path; tile path=${summary.tileCompositePath}; tile primitive=${summary.tileCompositePrimitive}`;
    return;
  }

  ui.drawPathSelectNote.textContent =
    `full-frame only; legacy is the fallback path; tile path=${summary.tileCompositePath}; tile primitive=${summary.tileCompositePrimitive}`;
}

function bindSliderTextUpdates() {
  [
    ['timeSlider', 'timeVal', 2],
    ['splatScaleSlider', 'splatScaleVal', 2],
    ['sigmaScaleSlider', 'sigmaScaleVal', 2],
    ['prefilterVarSlider', 'prefilterVarVal', 2],
    ['renderScaleSlider', 'renderScaleVal', 2],
    ['timeDurationSlider', 'timeDurationVal', 1]
  ].forEach(([sliderKey, valueKey, digits]) => {
    ui[sliderKey].addEventListener('input', () => {
      ui[valueKey].textContent = Number(ui[sliderKey].value).toFixed(digits);
      scheduler.scheduleRender();
    });
  });

  ui.strideSlider.addEventListener('input', () => {
    ui.strideVal.textContent = ui.strideSlider.value;
    scheduler.scheduleRender();
  });

  ui.maxVisibleSlider.addEventListener('input', () => {
    ui.maxVisibleVal.textContent = ui.maxVisibleSlider.value;
    scheduler.scheduleRender();
  });

  ui.bgGraySlider.addEventListener('input', () => {
    ui.bgGrayVal.textContent = ui.bgGraySlider.value;
    scheduler.scheduleRender();
  });
}

const scheduler = createRenderScheduler({
  renderFrame: renderCurrentFrame,
  tokenRef,
  isPlaying: () => (playback ? playback.isPlaying() : false)
});

playback = createViewerPlayback({
  ui,
  controls,
  scheduleRender: scheduler.scheduleRender,
  getTimeRange: () => ({ min: parseFloat(ui.timeSlider.min), max: parseFloat(ui.timeSlider.max) }),
  requestNextFrame: (cb) => requestAnimationFrame(cb),
  onPlaybackStateChange: () => {
    scheduler.scheduleRender();
  },
  playbackSpeed: 2.0
});

const fileIO = createViewerFileIO({
  ui,
  parseArrayBuffer: (buf) => parseSplat4DV2(buf),
  onSceneLoaded: async (nextRaw) => {
    raw = nextRaw;
    if (!applyDeterministicCameraPreset()) {
      fitCameraToRaw(raw, controls, camera);
      if (!deterministicQueryState.cameraPreset) {
        appliedCameraPresetName = 'none';
      }
    }
    await scheduler.scheduleRender();
  },
  scheduleRender: scheduler.scheduleRender,
  defaultSceneUrl: './scene_v2.splat4d'
});

function bindPersistentUiState() {
  if (typeof uiUnbindPersistence === 'function') {
    uiUnbindPersistence();
    uiUnbindPersistence = null;
  }

  uiUnbindPersistence = bindUiStatePersistence(ui, {
    onChange: (state) => {
      updateDrawPathNoteFromState(state);
      scheduler.scheduleRender();
    }
  });
}

function bindUiEvents() {
  [
    'useSHCheck',
    'useRot4dCheck',
    'useNativeRot4dCheck',
    'useNativeMarginalCheck',
    'forceSh3dCheck'
  ].forEach((key) => {
    ui[key].addEventListener('change', scheduleRenderAndPersist);
  });

  if (ui.debugLogBtn) {
    ui.debugLogBtn.addEventListener('click', () => {
      exportLatestDebugTextToArea();
    });
  }

  if (ui.debugLogCopyBtn) {
    ui.debugLogCopyBtn.addEventListener('click', async () => {
      if (!ui.debugLogArea?.value) exportLatestDebugTextToArea();
      await copyDebugLogText(ui);
    });
  }

  if (ui.drawPathSelect) {
    ui.drawPathSelect.addEventListener('change', () => {
      const state = readAndSaveUiState(ui);
      updateDrawPathNoteFromState(state);
      scheduler.scheduleRender();
    });
  }

  ui.playBtn.addEventListener('click', () => {
    playback.togglePlaying();
  });

  ui.renderBtn.addEventListener('click', scheduler.scheduleRender);

  ui.resetCamBtn.addEventListener('click', () => {
    if (raw && !applyDeterministicCameraPreset()) {
      fitCameraToRaw(raw, controls, camera);
      if (!deterministicQueryState.cameraPreset) {
        appliedCameraPresetName = 'none';
      }
    }
    scheduler.scheduleRender();
  });

  controls.addEventListener('change', scheduler.scheduleRender);

  bindGpuDragInteraction(canvas, controls, interactionState, () => {
    scheduler.scheduleRender();
  });

  window.addEventListener('resize', () => {
    applyCanvasSize();
    scheduler.scheduleRender();
  });
}

function initializeUiState() {
  const appliedState = loadAndApplyUiState(ui);
  const deterministicState = applyDeterministicUiState();
  updateStaticUiText();
  updateDrawPathNoteFromState(deterministicQueryState.active ? deterministicState : appliedState);
  bindPersistentUiState();
}

function initializeDebugLogArea() {
  setDebugLogText(ui, '');
}

initializeUiState();
initializeDebugLogArea();
bindSliderTextUpdates();
bindUiEvents();
installViewerDebugApi();

applyCanvasSize();
playback.startLoop();
fileIO.bindFileInput();
fileIO.bindDragAndDrop(document);
fileIO.loadDefaultScene();
