import { parseSplat4DV2 } from './splat4d_parser_v2.js';
import {
  fitCameraToRaw,
  computeGaussianDebugState,
  computeGaussianState,
  computeScreenSplat,
  DEFAULT_SINGLE_SPLAT_COMPARE_INPUT
} from './rot4d_math.js';
import { evalSHColor } from './sh_eval.js';
import { renderGpuFrame } from './gpu_renderer.js';
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
  const convertedPose = convertDatasetTransformMatrixToViewerPose(
    summary.datasetTransformMatrix,
    summary.datasetCameraConvention ?? 'nerf-blender-c2w'
  );
  const cameraFoVyRad = Number.isFinite(summary.datasetCameraFoVyRad)
    ? Number(summary.datasetCameraFoVyRad)
    : (Number.isFinite(summary.datasetCameraFoVy) ? Number(summary.datasetCameraFoVy) : null);
  const cameraFoVxRad = Number.isFinite(summary.datasetCameraFoVxRad)
    ? Number(summary.datasetCameraFoVxRad)
    : (Number.isFinite(summary.datasetCameraFoVx) ? Number(summary.datasetCameraFoVx) : null);
  const cameraFoVyDeg = Number.isFinite(cameraFoVyRad) ? (cameraFoVyRad * 180 / Math.PI) : null;
  const cameraFoVxDeg = Number.isFinite(cameraFoVxRad) ? (cameraFoVxRad * 180 / Math.PI) : null;
  const cudaAlignedScreenSpaceCamera = buildCudaAlignedScreenSpaceCameraSummary(summary, convertedPose);
  return {
    ...summary,
    appliedCameraPresetName,
    deterministicQueryString: summary.deterministicQueryString ?? '',
    deterministicUrlSummary: summary.deterministicUrlSummary ?? '',
    deterministicRawQueryString: summary.rawQueryString ?? '',
    cameraSource: summary.cameraSource ?? 'camera-preset',
    datasetCameraConvention: summary.datasetCameraConvention ?? null,
    datasetViewMatrixMode: summary.datasetViewMatrixMode ?? 'threejs',
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
    snapshotApiAvailable: true,
    snapshotCaptureSource: lastSnapshotSummary.source,
    snapshotRenderWaitMode: lastSnapshotSummary.renderWaitMode,
    snapshotLastStatus: lastSnapshotSummary.status,
    snapshotLastReason: lastSnapshotSummary.reason
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

function getCameraDebugState() {
  const deterministicState = buildDeterministicStateSummary();
  return {
    timestamp: new Date().toISOString(),
    locationHref: window.location.href,
    rawQueryString: window.location.search.replace(/^\?/, ''),
    camera: buildActualCameraSummary(),
    controls: {
      target: vector3ToArray(controls?.target),
      enabled: typeof controls?.enabled === 'boolean' ? controls.enabled : null,
      enableDamping: typeof controls?.enableDamping === 'boolean' ? controls.enableDamping : null
    },
    deterministicState,
    convertedCameraPose: deterministicState.convertedCameraPose ?? null,
    lastRenderResultSummary: buildRenderResultInspectionSummary(latestRenderResult),
    canvasSizeSummary: buildCanvasSizeSummary(),
    sceneBounds: buildRawBoundsSummary(),
    uiState: getCurrentUiDebugSummary()
  };
}

function normalizeFramebufferSampleOptions(options = {}) {
  const gridWidth = Number.isFinite(Number(options.gridWidth))
    ? Math.max(1, Math.min(512, Number(options.gridWidth) | 0))
    : 64;
  const gridHeight = Number.isFinite(Number(options.gridHeight))
    ? Math.max(1, Math.min(512, Number(options.gridHeight) | 0))
    : 36;
  const nonBlackThreshold = Number.isFinite(Number(options.nonBlackThreshold))
    ? Math.max(0, Number(options.nonBlackThreshold))
    : 5;
  const brightestSampleCount = Number.isFinite(Number(options.brightestSampleCount))
    ? Math.max(0, Math.min(64, Number(options.brightestSampleCount) | 0))
    : 12;
  return { gridWidth, gridHeight, nonBlackThreshold, brightestSampleCount };
}

function sampleFramebufferPixel(gl, x, y) {
  const pixel = new Uint8Array(4);
  gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
  return [pixel[0], pixel[1], pixel[2], pixel[3]];
}

function pushBrightestSample(samples, sample, maxCount) {
  if (maxCount <= 0) return;
  samples.push(sample);
  samples.sort((a, b) => b.rgbSum - a.rgbSum);
  if (samples.length > maxCount) samples.length = maxCount;
}

function sampleCurrentFramebufferStats(options = {}) {
  const gpu = getGpu();
  const gl = gpu?.gl ?? null;
  const canvasSummary = buildCanvasSizeSummary();
  if (!gl) {
    return {
      available: false,
      reason: 'webgl-not-initialized',
      ...canvasSummary
    };
  }

  const opts = normalizeFramebufferSampleOptions(options);
  const width = gl.drawingBufferWidth | 0;
  const height = gl.drawingBufferHeight | 0;
  if (width <= 0 || height <= 0) {
    return {
      available: false,
      reason: 'empty-framebuffer',
      ...canvasSummary
    };
  }

  gl.finish();

  const sampleCount = opts.gridWidth * opts.gridHeight;
  const sumRgb = [0, 0, 0];
  const minRgb = [255, 255, 255];
  const maxRgb = [0, 0, 0];
  let alphaSum = 0;
  let maxAlpha = 0;
  let nonBlackCount = 0;
  let nonTransparentCount = 0;
  let representativeNonBlackPixel = null;
  const brightestSamples = [];

  for (let gy = 0; gy < opts.gridHeight; gy++) {
    const y = opts.gridHeight === 1
      ? Math.floor((height - 1) * 0.5)
      : Math.round((gy / (opts.gridHeight - 1)) * (height - 1));
    for (let gx = 0; gx < opts.gridWidth; gx++) {
      const x = opts.gridWidth === 1
        ? Math.floor((width - 1) * 0.5)
        : Math.round((gx / (opts.gridWidth - 1)) * (width - 1));
      const rgba8 = sampleFramebufferPixel(gl, x, y);
      const rgbSum = rgba8[0] + rgba8[1] + rgba8[2];

      for (let axis = 0; axis < 3; axis++) {
        sumRgb[axis] += rgba8[axis];
        minRgb[axis] = Math.min(minRgb[axis], rgba8[axis]);
        maxRgb[axis] = Math.max(maxRgb[axis], rgba8[axis]);
      }
      alphaSum += rgba8[3];
      maxAlpha = Math.max(maxAlpha, rgba8[3]);

      if (rgbSum > opts.nonBlackThreshold) {
        nonBlackCount++;
        if (!representativeNonBlackPixel) {
          representativeNonBlackPixel = { x, y, rgba8, rgbSum };
        }
      }
      if (rgba8[3] > 0) nonTransparentCount++;

      pushBrightestSample(brightestSamples, { x, y, rgba8, rgbSum }, opts.brightestSampleCount);
    }
  }

  const centerX = Math.floor((width - 1) * 0.5);
  const centerY = Math.floor((height - 1) * 0.5);
  const centerPixel = {
    x: centerX,
    y: centerY,
    rgba8: sampleFramebufferPixel(gl, centerX, centerY)
  };
  centerPixel.rgbSum = centerPixel.rgba8[0] + centerPixel.rgba8[1] + centerPixel.rgba8[2];

  return {
    available: true,
    reason: 'ok',
    canvasWidth: Number.isFinite(canvas?.width) ? canvas.width : 0,
    canvasHeight: Number.isFinite(canvas?.height) ? canvas.height : 0,
    clientWidth: Number.isFinite(canvas?.clientWidth) ? canvas.clientWidth : 0,
    clientHeight: Number.isFinite(canvas?.clientHeight) ? canvas.clientHeight : 0,
    framebufferWidth: width,
    framebufferHeight: height,
    gridWidth: opts.gridWidth,
    gridHeight: opts.gridHeight,
    nonBlackThreshold: opts.nonBlackThreshold,
    sampleCount,
    nonBlackCount,
    nonTransparentCount,
    meanRgb: sumRgb.map((value) => value / sampleCount),
    maxRgb,
    minRgb,
    maxAlpha,
    meanAlpha: alphaSum / sampleCount,
    nonBlackRatio: nonBlackCount / sampleCount,
    nonTransparentRatio: nonTransparentCount / sampleCount,
    brightestSamples,
    centerPixel,
    representativeNonBlackPixel,
    likelyBlackFrame: nonBlackCount === 0
  };
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

function downloadJsonDebug(data, fileName = 'gpu-viewer-debug.json') {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = sanitizeSnapshotFileName(fileName);
  a.click();
  URL.revokeObjectURL(url);
  return { fileName: a.download, byteLength: blob.size };
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

function normalizeDebugPixelList(inputPixels) {
  if (!Array.isArray(inputPixels)) return [];
  const pixels = [];
  const seen = new Set();
  for (const entry of inputPixels) {
    const x = Array.isArray(entry) ? Number(entry[0]) : Number(entry?.x ?? entry?.[0]);
    const y = Array.isArray(entry) ? Number(entry[1]) : Number(entry?.y ?? entry?.[1]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const pixel = [Math.floor(x), Math.floor(y)];
    const key = `${pixel[0]},${pixel[1]}`;
    if (seen.has(key)) continue;
    seen.add(key);
    pixels.push({
      pixel,
      source: entry?.source ?? 'requested',
      representativeIndex: Number.isFinite(Number(entry?.representativeIndex))
        ? Number(entry.representativeIndex) | 0
        : null
    });
  }
  return pixels;
}

function buildRepresentativeCompareMap(compareResult) {
  const map = new Map();
  for (const item of Array.isArray(compareResult?.items) ? compareResult.items : []) {
    const index = Number(item?.index);
    if (Number.isFinite(index)) map.set(index | 0, item);
  }
  return map;
}

function selectDefaultAccumulationDebugPixels(compareResult, maxPixels = 3) {
  const items = Array.isArray(compareResult?.items) ? compareResult.items : [];
  return items
    .filter((item) => item?.found && Array.isArray(item.actualPayload?.centerPx))
    .map((item) => {
      const radius = Number.isFinite(item.actualPayload?.radius) ? Math.max(1, Number(item.actualPayload.radius)) : 1;
      const alpha = Number.isFinite(item.actualPayload?.alpha) ? Math.max(0, Number(item.actualPayload.alpha)) : 0;
      const occurrenceCount = Number.isFinite(item.accumulationOccurrenceCount) ? Math.max(1, item.accumulationOccurrenceCount | 0) : 1;
      return {
        item,
        score: radius * alpha * occurrenceCount
      };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.max(1, maxPixels | 0))
    .map(({ item }) => ({
      pixel: [
        Math.floor(item.actualPayload.centerPx[0]),
        Math.floor(item.actualPayload.centerPx[1])
      ],
      source: 'representative-high-overlap-alpha-radius',
      representativeIndex: Number(item.index) | 0
    }));
}

function getTileGridSummaryFromRenderResult(renderResult) {
  const summary = renderResult?.tileCompositePlan?.summary ?? renderResult?.tileSummary ?? null;
  const tileSize = Number.isFinite(summary?.tileCompositeTileSize)
    ? Math.max(1, summary.tileCompositeTileSize | 0)
    : (Number.isFinite(summary?.tileSize) ? Math.max(1, summary.tileSize | 0) : 32);
  const canvasWidth = Number.isFinite(canvas?.width) ? canvas.width | 0 : 0;
  const canvasHeight = Number.isFinite(canvas?.height) ? canvas.height | 0 : 0;
  const tileCols = Number.isFinite(summary?.tileCompositeTileCols)
    ? Math.max(1, summary.tileCompositeTileCols | 0)
    : Math.max(1, Math.ceil(canvasWidth / tileSize));
  const tileRows = Number.isFinite(summary?.tileCompositeTileRows)
    ? Math.max(1, summary.tileCompositeTileRows | 0)
    : Math.max(1, Math.ceil(canvasHeight / tileSize));
  return { tileSize, tileCols, tileRows, canvasWidth, canvasHeight };
}

function findTileBatchForPixel(renderResult, pixel) {
  const grid = getTileGridSummaryFromRenderResult(renderResult);
  const x = Math.floor(pixel?.[0] ?? -1);
  const y = Math.floor(pixel?.[1] ?? -1);
  if (x < 0 || y < 0 || x >= grid.canvasWidth || y >= grid.canvasHeight) {
    return {
      ok: false,
      reason: 'pixel-out-of-bounds',
      grid,
      tileId: -1,
      tile: [-1, -1],
      batch: null
    };
  }
  const tx = Math.min(grid.tileCols - 1, Math.floor(x / grid.tileSize));
  const ty = Math.min(grid.tileRows - 1, Math.floor(y / grid.tileSize));
  const tileId = ty * grid.tileCols + tx;
  const batches = Array.isArray(renderResult?.tileCompositePlan?.batches)
    ? renderResult.tileCompositePlan.batches
    : [];
  const batch = batches.find((candidate) => (candidate?.tileId | 0) === tileId) ?? null;
  return {
    ok: !!batch,
    reason: batch ? 'ok' : 'tile-batch-not-found',
    grid,
    tileId,
    tile: [tx, ty],
    tileBounds: batch?.rect ? cloneNumberArray(batch.rect, 4) : [
      tx * grid.tileSize,
      ty * grid.tileSize,
      Math.min(grid.canvasWidth, (tx + 1) * grid.tileSize),
      Math.min(grid.canvasHeight, (ty + 1) * grid.tileSize)
    ],
    batch
  };
}

function buildDebugScreenSpaceCameraProxy() {
  const deterministicState = buildDeterministicStateSummary();
  const cudaAligned = deterministicState?.cudaAlignedScreenSpaceCamera;
  if (!cudaAligned?.enabled || !Array.isArray(cudaAligned.cudaAlignedViewMatrix)) return null;
  return {
    fov: camera.fov,
    aspect: camera.aspect,
    matrixWorldInverse: camera.matrixWorldInverse,
    projectionMatrix: camera.projectionMatrix,
    updateMatrixWorld: () => {},
    screenSpaceTransformOverride: {
      mode: 'cuda-aligned',
      viewMatrixSource: cudaAligned.viewMatrixSource ?? 'dataset-transform-cuda-reader-c2w-inverse',
      viewMatrix: cudaAligned.cudaAlignedViewMatrix,
      fov: camera.fov,
      aspect: camera.aspect,
      intrinsics: cudaAligned.intrinsics ?? null,
      covarianceTanFovX: Number.isFinite(cudaAligned.covarianceTanFovX) ? Number(cudaAligned.covarianceTanFovX) : null,
      covarianceTanFovY: Number.isFinite(cudaAligned.covarianceTanFovY) ? Number(cudaAligned.covarianceTanFovY) : null,
      covarianceFocalContract: cudaAligned.covarianceFocalContract ?? null,
      pixelXSign: [-1, 1].includes(cudaAligned.pixelXSign) ? Number(cudaAligned.pixelXSign) : 1,
      signConversionMatrix: cudaAligned.signConversionMatrix ?? null,
      basis: cudaAligned.cudaAlignedCameraBasis ?? null,
      projectionContract: cudaAligned.projectionContract ?? 'cuda-plus-z-forward-fx-fy-cx-cy',
      screenYSign: Number.isFinite(cudaAligned.screenYSign) ? Number(cudaAligned.screenYSign) : 1,
      depthSign: Number.isFinite(cudaAligned.depthSign) ? Number(cudaAligned.depthSign) : 1
    }
  };
}

function clonePayloadForAssociationDebug(payload) {
  if (!payload) return null;
  return {
    centerPx: cloneNumberArray(payload.centerPx, 2),
    depth: toFiniteNumberOrNull(payload.depth),
    conic: cloneNumberArray(payload.conic, 3),
    radius: toFiniteNumberOrNull(payload.radius),
    radiusPx: toFiniteNumberOrNull(payload.radiusPx),
    rasterRect: cloneNumberArray(payload.rasterRect, 4),
    opacity: toFiniteNumberOrNull(payload.opacity),
    alpha: toFiniteNumberOrNull(payload.alpha),
    color: cloneNumberArray(payload.color, 3),
    colorAlpha: cloneNumberArray(payload.colorAlpha, 4),
    stateConvention: typeof payload.stateConvention === 'string' ? payload.stateConvention : null,
    usedCuda4DStateHelper: typeof payload.usedCuda4DStateHelper === 'boolean' ? payload.usedCuda4DStateHelper : null,
    stateHelperVersion: typeof payload.stateHelperVersion === 'string'
      ? payload.stateHelperVersion
      : (typeof payload.helperVersion === 'string' ? payload.helperVersion : null)
  };
}

function computeMaxAbsDelta(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return null;
  let maxAbs = 0;
  for (let i = 0; i < a.length; i++) {
    const av = Number(a[i]);
    const bv = Number(b[i]);
    if (!Number.isFinite(av) || !Number.isFinite(bv)) return null;
    maxAbs = Math.max(maxAbs, Math.abs(av - bv));
  }
  return maxAbs;
}

function buildAssociationDelta(packedPayload, recomputedPayload) {
  if (!packedPayload || !recomputedPayload) {
    return {
      valid: false,
      reason: packedPayload ? 'recomputed-payload-missing' : 'packed-payload-missing'
    };
  }
  const centerPxMaxAbs = computeMaxAbsDelta(packedPayload.centerPx, recomputedPayload.centerPx);
  const conicMaxAbs = computeMaxAbsDelta(packedPayload.conic, recomputedPayload.conic);
  const colorMaxAbs = computeMaxAbsDelta(packedPayload.color, recomputedPayload.color);
  const rasterRectMaxAbs = computeMaxAbsDelta(packedPayload.rasterRect, recomputedPayload.rasterRect);
  const depthAbs = Number.isFinite(packedPayload.depth) && Number.isFinite(recomputedPayload.depth)
    ? Math.abs(packedPayload.depth - recomputedPayload.depth)
    : null;
  const radiusAbs = Number.isFinite(packedPayload.radius) && Number.isFinite(recomputedPayload.radius)
    ? Math.abs(packedPayload.radius - recomputedPayload.radius)
    : null;
  const opacityAbs = Number.isFinite(packedPayload.alpha) && Number.isFinite(recomputedPayload.alpha)
    ? Math.abs(packedPayload.alpha - recomputedPayload.alpha)
    : null;
  const mismatches = {
    centerPx: Number.isFinite(centerPxMaxAbs) && centerPxMaxAbs > 1e-3,
    depth: Number.isFinite(depthAbs) && depthAbs > 1e-3,
    conic: Number.isFinite(conicMaxAbs) && conicMaxAbs > 1e-5,
    radius: Number.isFinite(radiusAbs) && radiusAbs > 0,
    opacity: Number.isFinite(opacityAbs) && opacityAbs > 1e-5,
    color: Number.isFinite(colorMaxAbs) && colorMaxAbs > 1e-5,
    rasterRect: Number.isFinite(rasterRectMaxAbs) && rasterRectMaxAbs > 0
  };
  return {
    valid: true,
    centerPxMaxAbs,
    depthAbs,
    conicMaxAbs,
    radiusAbs,
    opacityAbs,
    colorMaxAbs,
    rasterRectMaxAbs,
    mismatches,
    anyMismatch: Object.values(mismatches).some(Boolean)
  };
}

function recomputePayloadForOriginalSplatIndex(originalSplatIndex) {
  const index = Number(originalSplatIndex);
  if (!raw || !Number.isFinite(index) || index < 0 || index >= raw.N) {
    return {
      ok: false,
      reason: 'raw-missing-or-index-out-of-range',
      originalSplatIndex: Number.isFinite(index) ? (index | 0) : null
    };
  }

  const renderScale = Number.isFinite(Number(ui.renderScaleSlider?.value))
    ? Number(ui.renderScaleSlider.value)
    : 1;
  const renderW = Math.max(1, Math.round(canvas.width * renderScale));
  const renderH = Math.max(1, Math.round(canvas.height * renderScale));
  const sx = canvas.width / renderW;
  const sy = canvas.height / renderH;
  const timestamp = Number.isFinite(Number(ui.timeSlider?.value)) ? Number(ui.timeSlider.value) : 0;
  const scalingModifier = Number.isFinite(Number(ui.splatScaleSlider?.value)) ? Number(ui.splatScaleSlider.value) : 1;
  const sigmaScale = Number.isFinite(Number(ui.sigmaScaleSlider?.value)) ? Number(ui.sigmaScaleSlider.value) : 1;
  const prefilterVar = Number.isFinite(Number(ui.prefilterVarSlider?.value)) ? Number(ui.prefilterVarSlider.value) : 0;
  const useRot4d = !!ui.useRot4dCheck?.checked;
  const flags = {
    nativeRot4d: !!ui.useNativeRot4dCheck?.checked,
    nativeMarginal: !!ui.useNativeMarginalCheck?.checked
  };

  const gs = computeGaussianState(
    raw,
    index | 0,
    timestamp,
    scalingModifier,
    sigmaScale,
    prefilterVar,
    useRot4d,
    flags
  );
  if (!gs) {
    return {
      ok: false,
      reason: 'computeGaussianState-culled-or-null',
      originalSplatIndex: index | 0,
      timestamp,
      flags,
      useRot4d
    };
  }

  const color = evalSHColor(
    raw,
    index | 0,
    camera.position.clone(),
    gs.pos,
    timestamp,
    Number.isFinite(Number(ui.timeDurationSlider?.value)) ? Number(ui.timeDurationSlider.value) : 33,
    !!ui.useSHCheck?.checked,
    !!ui.forceSh3dCheck?.checked
  );
  const screenSpaceCamera = buildDebugScreenSpaceCameraProxy();
  const splat = computeScreenSplat(screenSpaceCamera || camera, gs.pos, gs.cov3, gs.opacity, renderW, renderH);
  if (!splat) {
    return {
      ok: false,
      reason: 'computeScreenSplat-culled-or-null',
      originalSplatIndex: index | 0,
      timestamp,
      flags,
      useRot4d
    };
  }

  const px = splat.px * sx;
  const py = splat.py * sy;
  const radius = splat.radius * Math.max(sx, sy);
  const coverageRadius = Math.max(1, radius);
  const rasterRect = [
    clampIntForDebug(Math.floor(px - coverageRadius), 0, canvas.width - 1),
    clampIntForDebug(Math.floor(py - coverageRadius), 0, canvas.height - 1),
    clampIntForDebug(Math.ceil(px + coverageRadius), 0, canvas.width - 1),
    clampIntForDebug(Math.ceil(py + coverageRadius), 0, canvas.height - 1)
  ];
  const conic = [
    splat.conic[0] / (sx * sx),
    splat.conic[1] / (sx * sy),
    splat.conic[2] / (sy * sy)
  ];
  return {
    ok: true,
    reason: 'ok',
    originalSplatIndex: index | 0,
    timestamp,
    flags,
    useRot4d,
    source: 'raw-recompute-current-viewer-settings',
    stateConvention: gs.stateConvention ?? null,
    usedCuda4DStateHelper: !!gs.usedCuda4DStateHelper,
    stateHelperVersion: gs.helperVersion ?? null,
    centerPx: [px, py],
    depth: splat.depth,
    conic,
    radius,
    radiusPx: radius,
    rasterRect,
    opacity: splat.opacity,
    alpha: splat.opacity,
    color,
    colorAlpha: [color[0], color[1], color[2], splat.opacity]
  };
}

function clampIntForDebug(value, min, max) {
  const n = Number(value);
  if (!Number.isFinite(n)) return min;
  return Math.min(max, Math.max(min, n | 0));
}

function computeDepthOrderingSummaryForBatch(batch) {
  const packedCount = Number.isFinite(batch?.packedCount) ? Math.max(0, batch.packedCount | 0) : 0;
  let previousDepth = -Infinity;
  let mismatchCount = 0;
  let firstMismatch = null;
  const depthPreviewHead = [];
  const depthPreviewTail = [];
  for (let i = 0; i < packedCount; i++) {
    const payload = decodePackedPayloadItem(batch?.packed, i, batch?.floatsPerItem);
    const depth = Number(payload?.depth);
    if (i < 8) depthPreviewHead.push(depth);
    if (i >= Math.max(0, packedCount - 8)) depthPreviewTail.push(depth);
    if (Number.isFinite(depth) && Number.isFinite(previousDepth) && depth < previousDepth) {
      mismatchCount++;
      if (!firstMismatch) firstMismatch = { localOrder: i, previousDepth, currentDepth: depth };
    }
    if (Number.isFinite(depth)) previousDepth = depth;
  }
  return {
    depthOrder: 'ascending-near-to-far',
    sequenceConsistent: mismatchCount === 0,
    orderingMismatchCount: mismatchCount,
    firstMismatch,
    depthPreviewHead,
    depthPreviewTail
  };
}

function evaluateAccumulationPayloadAtPixel(payload, pixel) {
  const dx = payload.centerPx[0] - pixel[0];
  const dy = payload.centerPx[1] - pixel[1];
  const power =
    -0.5 * (payload.conic[0] * dx * dx + payload.conic[2] * dy * dy) -
    payload.conic[1] * dx * dy;
  const rawAlpha = payload.alpha * Math.exp(power);
  const computedAlpha = Math.min(0.99, rawAlpha);
  let skipReason = 'none';
  if (power > 0.0) {
    skipReason = 'power-positive';
  } else if (computedAlpha < (1.0 / 255.0)) {
    skipReason = 'alpha-below-1-over-255';
  }
  return {
    dx,
    dy,
    power,
    rawAlpha,
    computedAlpha,
    survivesPower: power <= 0.0,
    survivesAlphaThreshold: computedAlpha >= (1.0 / 255.0),
    skipReason
  };
}

function readFramebufferPixelRgb(gl, pixel) {
  const width = Number.isFinite(canvas?.width) ? canvas.width | 0 : 0;
  const height = Number.isFinite(canvas?.height) ? canvas.height | 0 : 0;
  const x = Math.floor(pixel?.[0] ?? -1);
  const y = Math.floor(pixel?.[1] ?? -1);
  if (!gl || x < 0 || y < 0 || x >= width || y >= height) {
    return {
      valid: false,
      reason: 'pixel-out-of-bounds-or-missing-gl',
      pixel: [x, y],
      glPixel: [x, height - 1 - y],
      rgba8: [0, 0, 0, 0],
      rgb: [0, 0, 0]
    };
  }
  const yGl = height - 1 - y;
  const rgba = new Uint8Array(4);
  try {
    gl.readPixels(x, yGl, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
  } catch (error) {
    return {
      valid: false,
      reason: `readpixels-failed:${error?.message ?? 'unknown'}`,
      pixel: [x, y],
      glPixel: [x, yGl],
      rgba8: [0, 0, 0, 0],
      rgb: [0, 0, 0]
    };
  }
  return {
    valid: true,
    reason: 'readback-ok',
    pixel: [x, y],
    glPixel: [x, yGl],
    rgba8: Array.from(rgba),
    rgb: [rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0]
  };
}

function simulateTileAccumulationAtPixel({
  batch,
  visibleItems = [],
  pixel,
  bgGray01,
  representativeCompareMap,
  maxItems = 2048
}) {
  const packedCount = Number.isFinite(batch?.packedCount) ? Math.max(0, batch.packedCount | 0) : 0;
  const clampedCount = Math.min(packedCount, Math.max(1, Math.min(2048, maxItems | 0)));
  const sourceIndices = batch?.sourceIndices instanceof Uint32Array ? batch.sourceIndices : null;
  const orderedIndices = batch?.orderedIndices instanceof Uint32Array ? batch.orderedIndices : null;
  const entries = [];
  let T = 1.0;
  let accumColor = [0, 0, 0];
  let accumDepth = 0;
  let contributor = 0;
  let lastContributingLocalOrder = -1;
  let earlyOutTriggered = false;
  let earlyOutAtLocalOrder = -1;
  let alphaSum = 0;
  let contributionCount = 0;
  let powerSkipCount = 0;
  let alphaSkipCount = 0;

  for (let localOrder = 0; localOrder < clampedCount; localOrder++) {
    contributor++;
    const payload = decodePackedPayloadItem(batch?.packed, localOrder, batch?.floatsPerItem);
    if (!payload) {
      entries.push({ localOrder, skipReason: 'payload-decode-failed' });
      continue;
    }
    const originalSplatIndex = sourceIndices ? (sourceIndices[localOrder] | 0) : null;
    const representative = Number.isFinite(originalSplatIndex)
      ? representativeCompareMap.get(originalSplatIndex)
      : null;
    const evaluation = evaluateAccumulationPayloadAtPixel(payload, pixel);
    const TBefore = T;
    const testT = TBefore * (1.0 - evaluation.computedAlpha);
    let contributionRgb = [0, 0, 0];
    let accumColorAfter = [...accumColor];
    let TAfter = TBefore;
    let contributes = false;
    let skipReason = evaluation.skipReason;

    if (skipReason === 'power-positive') {
      powerSkipCount++;
    } else if (skipReason === 'alpha-below-1-over-255') {
      alphaSkipCount++;
    } else if (testT < 0.0001) {
      skipReason = 'early-out-testT-below-0.0001';
      earlyOutTriggered = true;
      earlyOutAtLocalOrder = localOrder;
    } else {
      contributionRgb = payload.color.map((value) => value * evaluation.computedAlpha * TBefore);
      accumColor = [
        accumColor[0] + contributionRgb[0],
        accumColor[1] + contributionRgb[1],
        accumColor[2] + contributionRgb[2]
      ];
      accumDepth += payload.depth * evaluation.computedAlpha * TBefore;
      accumColorAfter = [...accumColor];
      T = testT;
      TAfter = T;
      contributes = true;
      contributionCount++;
      alphaSum += evaluation.computedAlpha;
      lastContributingLocalOrder = localOrder;
    }

    entries.push({
      localOrder,
      packedIndex: localOrder,
      sourceVisibleIndex: orderedIndices ? (orderedIndices[localOrder] | 0) : null,
      originalSplatIndex,
      representative: !!representative,
      representativeReferenceMatchesActualPayload: representative?.summary?.payloadMatchesDebug ?? representative?.found ?? null,
      representativeAccumulationOccurrenceCount: representative?.accumulationOccurrenceCount ?? null,
      depth: payload.depth,
      centerPx: payload.centerPx,
      radius: payload.radius,
      rasterRect: payload.rasterRect,
      conic: payload.conic,
      opacity: payload.alpha,
      alpha: payload.alpha,
      color: payload.color,
      stateConvention: Number.isFinite(orderedIndices ? (orderedIndices[localOrder] | 0) : null)
        ? (visibleItems[orderedIndices[localOrder] | 0]?.stateConvention ?? null)
        : null,
      usedCuda4DStateHelper: Number.isFinite(orderedIndices ? (orderedIndices[localOrder] | 0) : null)
        ? !!visibleItems[orderedIndices[localOrder] | 0]?.usedCuda4DStateHelper
        : null,
      stateHelperVersion: Number.isFinite(orderedIndices ? (orderedIndices[localOrder] | 0) : null)
        ? (visibleItems[orderedIndices[localOrder] | 0]?.stateHelperVersion ?? null)
        : null,
      dx: evaluation.dx,
      dy: evaluation.dy,
      power: evaluation.power,
      rawAlpha: evaluation.rawAlpha,
      computedAlpha: evaluation.computedAlpha,
      skipReason,
      contributes,
      TBefore,
      test_T: testT,
      TAfter,
      contributionRgb,
      accumColorAfter,
      accumDepthAfter: accumDepth
    });

    if (earlyOutTriggered) break;
  }

  const bg = Number.isFinite(bgGray01) ? Math.max(0, Math.min(1, Number(bgGray01))) : 0;
  const finalRgb = [
    accumColor[0] + T * bg,
    accumColor[1] + T * bg,
    accumColor[2] + T * bg
  ];
  return {
    pixel,
    packedCount,
    clampedCount,
    maxItemsEvaluated: clampedCount,
    sortOrderContract: 'ascending-depth-near-to-far-front-to-back',
    shaderEquationContract: 'matches gpu_tile_accumulation_executor fragment shader and CUDA forward.cu render loop',
    depthOrderingSummary: computeDepthOrderingSummaryForBatch(batch),
    contributorCounter: contributor,
    contributionCount,
    powerSkipCount,
    alphaSkipCount,
    alphaSum,
    lastContributingLocalOrder,
    earlyOutTriggered,
    earlyOutAtLocalOrder,
    finalT: T,
    accumColor,
    finalRgb,
    accumDepth,
    entries
  };
}

async function captureTileAccumulationDebug(input = {}) {
  const options = input ?? {};
  const representativeCompare = options.representativeCompare ?? options.compareResult ?? null;
  const requestedPixels = normalizeDebugPixelList(options.pixels);
  const pixels = requestedPixels.length > 0
    ? requestedPixels
    : selectDefaultAccumulationDebugPixels(representativeCompare, options.maxPixels ?? 3);
  const renderResultOverride = hasRetainedActualPayload(options.renderResultOverride) ? options.renderResultOverride : null;
  const debugRender = renderResultOverride
    ? {
        renderResult: renderResultOverride,
        attempts: Array.isArray(options.sharedDebugRenderAttempts)
          ? options.sharedDebugRenderAttempts
          : [{ stage: 'shared-render-result-override', retentionSummary: buildActualPayloadRetentionSummary(renderResultOverride) }]
      }
    : options.ensureCurrentFrame === false && hasRetainedActualPayload(latestRenderResult)
    ? { renderResult: latestRenderResult, attempts: [{ stage: 'reuse-latest-render-result', retentionSummary: buildActualPayloadRetentionSummary(latestRenderResult) }] }
    : await renderCurrentFrameForDebugPayload(options);
  const renderResult = debugRender.renderResult;
  const gl = getGpu()?.gl ?? null;
  const bgGray01 = Number.isFinite(Number(ui.bgGraySlider?.value))
    ? Number(ui.bgGraySlider.value) / 255.0
    : 0;
  const representativeCompareMap = buildRepresentativeCompareMap(representativeCompare);

  const pixelResults = pixels.map((pixelSpec) => {
    const tileInfo = findTileBatchForPixel(renderResult, pixelSpec.pixel);
    if (!tileInfo.ok) {
      return {
        pixel: pixelSpec.pixel,
        pixelSource: pixelSpec.source,
        representativeIndex: pixelSpec.representativeIndex,
        ok: false,
        reason: tileInfo.reason,
        tileId: tileInfo.tileId,
        tile: tileInfo.tile,
        tileBounds: tileInfo.tileBounds ?? null,
        grid: tileInfo.grid
      };
    }
    const simulation = simulateTileAccumulationAtPixel({
      batch: tileInfo.batch,
      visibleItems: Array.isArray(renderResult?.visible) ? renderResult.visible : [],
      pixel: pixelSpec.pixel,
      bgGray01,
      representativeCompareMap,
      maxItems: options.maxItems ?? 2048
    });
    const framebuffer = readFramebufferPixelRgb(gl, pixelSpec.pixel);
    const framebufferDelta = framebuffer.valid
      ? [
          framebuffer.rgb[0] - simulation.finalRgb[0],
          framebuffer.rgb[1] - simulation.finalRgb[1],
          framebuffer.rgb[2] - simulation.finalRgb[2]
        ]
      : null;
    return {
      pixel: pixelSpec.pixel,
      pixelSource: pixelSpec.source,
      representativeIndex: pixelSpec.representativeIndex,
      ok: true,
      tileId: tileInfo.tileId,
      tile: tileInfo.tile,
      tileBounds: tileInfo.tileBounds,
      grid: tileInfo.grid,
      tilePayloadCount: Number.isFinite(tileInfo.batch?.packedCount) ? tileInfo.batch.packedCount : 0,
      batchPackedCount: Number.isFinite(tileInfo.batch?.packedCount) ? tileInfo.batch.packedCount : 0,
      batchFloatsPerItem: Number.isFinite(tileInfo.batch?.floatsPerItem) ? tileInfo.batch.floatsPerItem : 16,
      batchHasSourceIndices: tileInfo.batch?.sourceIndices instanceof Uint32Array,
      batchHasOrderedIndices: tileInfo.batch?.orderedIndices instanceof Uint32Array,
      accumulation: simulation,
      framebuffer,
      framebufferDelta,
      framebufferDeltaAbsMax: Array.isArray(framebufferDelta)
        ? Math.max(...framebufferDelta.map((value) => Math.abs(value)))
        : null
    };
  });

  return {
    schemaVersion: 'step90-tile-accumulation-debug-v1',
    timestamp: new Date().toISOString(),
    purpose: 'CPU/JS replay of Viewer tile accumulation shader inputs/order/equations for selected pixels',
    cudaEquationSummary: {
      order: 'per tile sorted by ascending depth key, front-to-back',
      power: '-0.5 * (conic.x * dx^2 + conic.z * dy^2) - conic.y * dx * dy',
      alpha: 'min(0.99, opacity * exp(power))',
      skip: 'continue if power > 0 or alpha < 1/255',
      testT: 'T * (1 - alpha)',
      earlyOut: 'if testT < 0.0001, stop without contributing crossing splat',
      color: 'C += rgb * alpha * T; final = C + T * bg'
    },
    viewerEquationSummary: {
      order: 'tileCompositePlan.batches[].packed sorted by ascending depth',
      bgGray01,
      premultipliedAlpha: false,
      outputAlpha: 1
    },
    selectedPixels: pixels,
    actualPayloadRetentionSummary: buildActualPayloadRetentionSummary(renderResult),
    debugRenderAttempts: debugRender.attempts,
    lastRenderResultSummary: buildRenderResultInspectionSummary(renderResult),
    representativeCompareSummary: representativeCompare?.summary ?? null,
    pixels: pixelResults
  };
}

async function captureViewerPayloadIndexAssociationDebug(input = {}) {
  const options = input ?? {};
  const targetPixel = Array.isArray(options.pixel) && options.pixel.length >= 2
    ? [Number(options.pixel[0]) | 0, Number(options.pixel[1]) | 0]
    : [655, 363];
  const targetIndices = Array.isArray(options.indices) && options.indices.length > 0
    ? options.indices.map((value) => Number(value)).filter(Number.isFinite).map((value) => value | 0)
    : [2765070, 1182029, 2718004];
  const targetIndexSet = new Set(targetIndices);
  const maxEntries = Number.isFinite(Number(options.maxEntries))
    ? Math.max(1, Number(options.maxEntries) | 0)
    : 2048;
  const renderResultOverride = hasRetainedActualPayload(options.renderResultOverride) ? options.renderResultOverride : null;
  const debugRender = renderResultOverride
    ? {
        renderResult: renderResultOverride,
        attempts: Array.isArray(options.sharedDebugRenderAttempts)
          ? options.sharedDebugRenderAttempts
          : [{ stage: 'shared-render-result-override', retentionSummary: buildActualPayloadRetentionSummary(renderResultOverride) }]
      }
    : options.ensureCurrentFrame === false && hasRetainedActualPayload(latestRenderResult)
    ? { renderResult: latestRenderResult, attempts: [{ stage: 'reuse-latest-render-result', retentionSummary: buildActualPayloadRetentionSummary(latestRenderResult) }] }
    : await renderCurrentFrameForDebugPayload(options);
  const renderResult = debugRender.renderResult;
  const tileInfo = findTileBatchForPixel(renderResult, targetPixel);
  const visible = Array.isArray(renderResult?.visible) ? renderResult.visible : [];
  const entries = [];
  const mismatches = [];
  const targetOccurrences = new Map(targetIndices.map((index) => [index, []]));

  if (tileInfo.ok) {
    const batch = tileInfo.batch;
    const packedCount = Number.isFinite(batch?.packedCount) ? Math.max(0, batch.packedCount | 0) : 0;
    const clampedCount = Math.min(packedCount, maxEntries);
    const sourceIndices = batch?.sourceIndices instanceof Uint32Array ? batch.sourceIndices : null;
    const orderedIndices = batch?.orderedIndices instanceof Uint32Array ? batch.orderedIndices : null;

    for (let localOrder = 0; localOrder < clampedCount; localOrder++) {
      const sourceVisibleIndex = orderedIndices ? (orderedIndices[localOrder] | 0) : null;
      const originalSplatIndex = sourceIndices ? (sourceIndices[localOrder] | 0) : null;
      const visibleItem = Number.isFinite(sourceVisibleIndex) ? visible[sourceVisibleIndex] : null;
      const packedPayload = decodePackedPayloadItem(batch?.packed, localOrder, batch?.floatsPerItem);
      const recomputed = Number.isFinite(originalSplatIndex)
        ? recomputePayloadForOriginalSplatIndex(originalSplatIndex)
        : { ok: false, reason: 'missing-originalSplatIndex' };
      const recomputedPayload = recomputed?.ok ? clonePayloadForAssociationDebug(recomputed) : null;
      const packedSlim = clonePayloadForAssociationDebug(packedPayload);
      const delta = buildAssociationDelta(packedSlim, recomputedPayload);
      const sourceVisibleConsistency = {
        hasVisibleItem: !!visibleItem,
        visibleItemSrcIndex: Number.isFinite(visibleItem?.srcIndex) ? (visibleItem.srcIndex | 0) : null,
        visibleItemStateConvention: typeof visibleItem?.stateConvention === 'string'
          ? visibleItem.stateConvention
          : null,
        visibleItemUsedCuda4DStateHelper: typeof visibleItem?.usedCuda4DStateHelper === 'boolean'
          ? visibleItem.usedCuda4DStateHelper
          : null,
        visibleItemStateHelperVersion: typeof visibleItem?.stateHelperVersion === 'string'
          ? visibleItem.stateHelperVersion
          : null,
        matchesOriginalSplatIndex:
          Number.isFinite(visibleItem?.srcIndex) &&
          Number.isFinite(originalSplatIndex) &&
          (visibleItem.srcIndex | 0) === (originalSplatIndex | 0)
      };
      const evaluation = packedPayload ? evaluateAccumulationPayloadAtPixel(packedPayload, targetPixel) : null;
      const entry = {
        localOrder,
        slot: localOrder,
        packedIndex: localOrder,
        sourceVisibleIndex,
        originalSplatIndex,
        targetIndex: Number.isFinite(originalSplatIndex) && targetIndexSet.has(originalSplatIndex),
        sourceVisibleConsistency,
        packedPayload: packedSlim,
        recomputedFromOriginalSplatIndex: recomputed?.ok
          ? recomputedPayload
          : {
              ok: false,
              reason: recomputed?.reason ?? 'recompute-failed',
              originalSplatIndex,
              timestamp: recomputed?.timestamp ?? null,
              flags: recomputed?.flags ?? null,
              useRot4d: recomputed?.useRot4d ?? null
            },
        delta,
        pixelEvaluation: evaluation
          ? {
              dx: evaluation.dx,
              dy: evaluation.dy,
              power: evaluation.power,
              rawAlpha: evaluation.rawAlpha,
              computedAlpha: evaluation.computedAlpha,
              skipReason: evaluation.skipReason
            }
          : null,
        cullShouldNotContribute:
          recomputed && !recomputed.ok && evaluation?.skipReason === 'none',
        associationMismatch: !!delta?.anyMismatch || (recomputed && !recomputed.ok)
      };
      if (entry.associationMismatch) mismatches.push(entry);
      if (Number.isFinite(originalSplatIndex) && targetOccurrences.has(originalSplatIndex)) {
        targetOccurrences.get(originalSplatIndex).push(entry);
      }
      if (options.includeAllEntries === true || localOrder < 64 || entry.targetIndex || entry.associationMismatch) entries.push(entry);
    }
  }

  for (const index of targetIndices) {
    if ((targetOccurrences.get(index)?.length ?? 0) > 0) continue;
    const recomputed = recomputePayloadForOriginalSplatIndex(index);
    targetOccurrences.set(index, [{
      localOrder: null,
      slot: null,
      packedIndex: null,
      sourceVisibleIndex: null,
      originalSplatIndex: index,
      targetIndex: true,
      sourceVisibleConsistency: {
        hasVisibleItem: false,
        visibleItemSrcIndex: null,
        matchesOriginalSplatIndex: false
      },
      packedPayload: null,
      recomputedFromOriginalSplatIndex: recomputed?.ok
        ? clonePayloadForAssociationDebug(recomputed)
        : {
            ok: false,
            reason: recomputed?.reason ?? 'recompute-failed',
            originalSplatIndex: index,
            timestamp: recomputed?.timestamp ?? null,
            flags: recomputed?.flags ?? null,
            useRot4d: recomputed?.useRot4d ?? null
          },
      delta: {
        valid: false,
        reason: 'target-index-not-present-in-target-tile-batch'
      },
      pixelEvaluation: null,
      cullShouldNotContribute: false,
      associationMismatch: false
    }]);
  }

  return {
    schemaVersion: 'step90-viewer-payload-index-association-debug-v1',
    timestamp: new Date().toISOString(),
    purpose: 'Validate that each tile accumulation packed payload slot matches its source/original index metadata.',
    target: {
      pixel: targetPixel,
      indices: targetIndices,
      maxEntries
    },
    debugRenderAttempts: debugRender.attempts,
    deterministicState: buildSlimDeterministicStateSummary(buildDeterministicStateSummary()),
    lastRenderResultSummary: buildRenderResultInspectionSummary(renderResult),
    tile: {
      ok: tileInfo.ok,
      reason: tileInfo.reason,
      tileId: tileInfo.tileId,
      tile: tileInfo.tile,
      tileBounds: tileInfo.tileBounds ?? null,
      grid: tileInfo.grid,
      packedCount: Number.isFinite(tileInfo.batch?.packedCount) ? tileInfo.batch.packedCount : 0,
      floatsPerItem: Number.isFinite(tileInfo.batch?.floatsPerItem) ? tileInfo.batch.floatsPerItem : 16,
      hasSourceIndices: tileInfo.batch?.sourceIndices instanceof Uint32Array,
      hasOrderedIndices: tileInfo.batch?.orderedIndices instanceof Uint32Array
    },
    summary: {
      checkedEntryCount: tileInfo.ok
        ? Math.min(Number.isFinite(tileInfo.batch?.packedCount) ? Math.max(0, tileInfo.batch.packedCount | 0) : 0, maxEntries)
        : 0,
      emittedEntryCount: entries.length,
      mismatchCount: mismatches.length,
      targetOccurrenceCounts: Object.fromEntries(
        targetIndices.map((index) => [String(index), targetOccurrences.get(index)?.filter((entry) => entry.localOrder !== null).length ?? 0])
      ),
      associationMismatchLikely: mismatches.length > 0,
      mismatchReasons: summarizeAssociationMismatchReasons(mismatches)
    },
    targetOccurrences: Object.fromEntries(
      targetIndices.map((index) => [String(index), targetOccurrences.get(index) ?? []])
    ),
    mismatches: mismatches.slice(0, 128),
    entries
  };
}

function summarizeAssociationMismatchReasons(mismatches) {
  const out = {};
  for (const mismatch of mismatches) {
    let reason = 'value-delta';
    if (mismatch?.recomputedFromOriginalSplatIndex?.ok === false) {
      reason = mismatch.recomputedFromOriginalSplatIndex.reason ?? 'recompute-failed';
    } else if (mismatch?.delta?.mismatches) {
      reason = Object.entries(mismatch.delta.mismatches)
        .filter(([, value]) => !!value)
        .map(([key]) => key)
        .join('+') || 'value-delta';
    }
    out[reason] = (out[reason] ?? 0) + 1;
  }
  return out;
}

function getVisibleCountFromDebugResult(debugResult) {
  const retention = debugResult?.actualPayloadRetentionSummary;
  if (Number.isFinite(retention?.visibleCount)) return retention.visibleCount;
  const attempts = Array.isArray(debugResult?.debugRenderAttempts) ? debugResult.debugRenderAttempts : [];
  for (let i = attempts.length - 1; i >= 0; i--) {
    const visibleCount = attempts[i]?.retentionSummary?.visibleCount;
    if (Number.isFinite(visibleCount)) return visibleCount;
  }
  return null;
}

function findOriginalSplatOccurrencesInTileDebug(tileDebug, index) {
  const entries = tileDebug?.pixels?.[0]?.accumulation?.entries;
  if (!Array.isArray(entries)) return [];
  return entries
    .filter((entry) => Number.isFinite(entry?.originalSplatIndex) && (entry.originalSplatIndex | 0) === (index | 0))
    .map((entry) => ({
      localOrder: entry.localOrder ?? null,
      slot: entry.localOrder ?? null,
      packedIndex: entry.packedIndex ?? null,
      sourceVisibleIndex: entry.sourceVisibleIndex ?? null,
      originalSplatIndex: entry.originalSplatIndex ?? null,
      depth: entry.depth ?? null,
      centerPx: cloneNumberArray(entry.centerPx, 2),
      conic: cloneNumberArray(entry.conic, 3),
      radius: entry.radius ?? null,
      opacity: entry.opacity ?? null,
      computedAlpha: entry.computedAlpha ?? null,
      skipReason: entry.skipReason ?? null
    }));
}

function summarizeSlotIndexAgreement(tileDebug, associationDebug) {
  const tileEntries = tileDebug?.pixels?.[0]?.accumulation?.entries;
  const associationEntries = associationDebug?.entries;
  if (!Array.isArray(tileEntries) || !Array.isArray(associationEntries)) {
    return {
      comparable: false,
      reason: 'entries-missing'
    };
  }
  const associationByLocalOrder = new Map();
  for (const entry of associationEntries) {
    if (Number.isFinite(entry?.localOrder)) associationByLocalOrder.set(entry.localOrder | 0, entry);
  }
  let comparedCount = 0;
  let mismatchCount = 0;
  let firstMismatch = null;
  for (const tileEntry of tileEntries) {
    if (!Number.isFinite(tileEntry?.localOrder)) continue;
    const associationEntry = associationByLocalOrder.get(tileEntry.localOrder | 0);
    if (!associationEntry) continue;
    comparedCount++;
    const tileOriginal = Number.isFinite(tileEntry?.originalSplatIndex) ? (tileEntry.originalSplatIndex | 0) : null;
    const associationOriginal = Number.isFinite(associationEntry?.originalSplatIndex) ? (associationEntry.originalSplatIndex | 0) : null;
    if (tileOriginal !== associationOriginal) {
      mismatchCount++;
      if (!firstMismatch) {
        firstMismatch = {
          localOrder: tileEntry.localOrder | 0,
          tileOriginalSplatIndex: tileOriginal,
          associationOriginalSplatIndex: associationOriginal,
          tileSourceVisibleIndex: tileEntry.sourceVisibleIndex ?? null,
          associationSourceVisibleIndex: associationEntry.sourceVisibleIndex ?? null
        };
      }
    }
  }
  return {
    comparable: true,
    comparedCount,
    mismatchCount,
    firstMismatch
  };
}

function buildLiveSameStateConsistencySummary(tileDebug, associationDebug, indices) {
  const tilePixel = tileDebug?.pixels?.[0] ?? null;
  const tileAccumulation = tilePixel?.accumulation ?? null;
  const tileVisibleCount = getVisibleCountFromDebugResult(tileDebug);
  const associationVisibleCount = getVisibleCountFromDebugResult(associationDebug);
  const tilePayloadCount = Number.isFinite(tilePixel?.tilePayloadCount) ? tilePixel.tilePayloadCount : null;
  const associationPackedCount = Number.isFinite(associationDebug?.tile?.packedCount) ? associationDebug.tile.packedCount : null;
  const targetIndices = Array.isArray(indices) ? indices : [];
  const targetSummary = {};
  for (const index of targetIndices) {
    const key = String(index | 0);
    const associationOccurrences = associationDebug?.targetOccurrences?.[key] ?? [];
    targetSummary[key] = {
      tileAccumulationOccurrences: findOriginalSplatOccurrencesInTileDebug(tileDebug, index),
      associationOccurrences,
      tileAccumulationOccurrenceCount: findOriginalSplatOccurrencesInTileDebug(tileDebug, index).length,
      associationOccurrenceCount: associationOccurrences.filter((entry) => entry?.localOrder !== null).length
    };
  }
  return {
    schemaVersion: 'step90-live-same-state-consistency-summary-v1',
    timestamp: new Date().toISOString(),
    visibleCount: {
      tileAccumulation: tileVisibleCount,
      association: associationVisibleCount,
      equal: tileVisibleCount !== null && tileVisibleCount === associationVisibleCount
    },
    tilePayloadCount: {
      tileAccumulation: tilePayloadCount,
      associationPackedCount,
      equal: tilePayloadCount !== null && tilePayloadCount === associationPackedCount
    },
    tile: {
      tileAccumulationTileId: tilePixel?.tileId ?? null,
      associationTileId: associationDebug?.tile?.tileId ?? null,
      tileAccumulationTile: tilePixel?.tile ?? null,
      associationTile: associationDebug?.tile?.tile ?? null,
      equal:
        tilePixel?.tileId === associationDebug?.tile?.tileId &&
        JSON.stringify(tilePixel?.tile ?? null) === JSON.stringify(associationDebug?.tile?.tile ?? null)
    },
    accumulation: {
      contributorCounter: tileAccumulation?.contributorCounter ?? null,
      contributionCount: tileAccumulation?.contributionCount ?? null,
      alphaSkipCount: tileAccumulation?.alphaSkipCount ?? null,
      earlyOutTriggered: tileAccumulation?.earlyOutTriggered ?? null,
      earlyOutAtLocalOrder: tileAccumulation?.earlyOutAtLocalOrder ?? null,
      finalRgb: tileAccumulation?.finalRgb ?? null
    },
    association: {
      mismatchCount: associationDebug?.summary?.mismatchCount ?? null,
      associationMismatchLikely: associationDebug?.summary?.associationMismatchLikely ?? null,
      targetOccurrenceCounts: associationDebug?.summary?.targetOccurrenceCounts ?? null
    },
    slotIndexAgreement: summarizeSlotIndexAgreement(tileDebug, associationDebug),
    targetSummary
  };
}

async function captureLiveSameStateTileAndAssociationDebug(input = {}) {
  const options = input ?? {};
  const pixel = Array.isArray(options.pixel) && options.pixel.length >= 2
    ? [Number(options.pixel[0]) | 0, Number(options.pixel[1]) | 0]
    : [655, 363];
  const indices = Array.isArray(options.indices) && options.indices.length > 0
    ? options.indices.map((value) => Number(value)).filter(Number.isFinite).map((value) => value | 0)
    : [2765070, 1182029, 2718004];
  const debugRender = await renderCurrentFrameForDebugPayload(options);
  const sharedOptions = {
    ...options,
    ensureCurrentFrame: false,
    renderResultOverride: debugRender.renderResult,
    sharedDebugRenderAttempts: debugRender.attempts
  };
  const tileAccumulationDebug = await captureTileAccumulationDebug({
    ...sharedOptions,
    pixels: [{ x: pixel[0], y: pixel[1], source: 'live-same-state-target' }],
    maxItems: Number.isFinite(Number(options.maxItems)) ? Number(options.maxItems) : 2048
  });
  const associationDebug = await captureViewerPayloadIndexAssociationDebug({
    ...sharedOptions,
    pixel,
    indices,
    maxEntries: Number.isFinite(Number(options.maxEntries)) ? Number(options.maxEntries) : 2048,
    includeAllEntries: options.includeAllEntries !== false
  });
  const consistencySummary = buildLiveSameStateConsistencySummary(tileAccumulationDebug, associationDebug, indices);
  return {
    schemaVersion: 'step90-live-same-state-tile-and-association-debug-v1',
    timestamp: new Date().toISOString(),
    purpose: 'Capture tile accumulation debug and payload-index association debug from one shared Viewer renderResult.',
    target: {
      pixel,
      indices
    },
    sharedDebugRenderAttempts: debugRender.attempts,
    consistencySummary,
    tileAccumulationDebug,
    associationDebug
  };
}

async function downloadLiveSameStateTileAndAssociationDebugJson(input = {}, fileNames = {}) {
  const result = await captureLiveSameStateTileAndAssociationDebug(input);
  const tileFileName = fileNames.tileAccumulation ?? 'step90_tile_accumulation_debug_live_same_state.json';
  const associationFileName = fileNames.association ?? 'step90_viewer_payload_index_association_debug_live_same_state.json';
  const summaryFileName = fileNames.summary ?? 'step90_live_same_state_consistency_summary.json';
  return {
    result,
    downloads: {
      tileAccumulation: downloadJsonDebug(result.tileAccumulationDebug, tileFileName),
      association: downloadJsonDebug(result.associationDebug, associationFileName),
      summary: downloadJsonDebug(result.consistencySummary, summaryFileName)
    }
  };
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
    datasetPixelXSign: [-1, 1].includes(summary?.datasetPixelXSign) ? Number(summary.datasetPixelXSign) : 1,
    datasetCameraLabel: summary?.datasetCameraLabel ?? null,
    imageName: summary?.imageName ?? null,
    frameNumber: Number.isFinite(summary?.frameNumber) ? Number(summary.frameNumber) : null,
    viewId: Number.isFinite(summary?.viewId) ? Number(summary.viewId) : null,
    datasetTime: Number.isFinite(summary?.datasetTime) ? Number(summary.datasetTime) : null,
    rawTransformMatrix: Array.isArray(summary?.rawTransformMatrix) ? summary.rawTransformMatrix.map((row) => [...row]) : null,
    convertedCameraPose: summary?.convertedCameraPose ?? null,
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
    drawPath: summary?.drawPath ?? 'none',
    tileCompositePath: summary?.tileCompositePath ?? 'baseline',
    tileCompositePrimitive: summary?.tileCompositePrimitive ?? 'point',
    inspectSource: summary?.inspectSource ?? 'auto',
    inspectJsonMode: summary?.inspectJsonMode ?? 'slim',
    gpuFramePolicyOverride: summary?.gpuFramePolicyOverride ?? 'auto',
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

function applyDeterministicDatasetCameraPose() {
  if (!raw) return false;
  const convertedPose = convertDatasetTransformMatrixToViewerPose(
    deterministicQueryState.datasetTransformMatrix,
    deterministicQueryState.datasetCameraConvention ?? 'nerf-blender-c2w'
  );
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
  const normalized = baseName.replace(/[^a-zA-Z0-9._-]+/g, '-').replace(/-+/g, '-');
  return normalized.toLowerCase().endsWith('.png') ? normalized : `${normalized}.png`;
}

function downloadBlob(blob, fileName) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function createSnapshotCanvasFromPixels(width, height, pixels) {
  const snapshotCanvas = document.createElement('canvas');
  snapshotCanvas.width = width;
  snapshotCanvas.height = height;
  const ctx = snapshotCanvas.getContext('2d', { willReadFrequently: true });
  const imageData = ctx.createImageData(width, height);
  const rowStride = width * 4;

  for (let y = 0; y < height; y++) {
    const srcOffset = (height - 1 - y) * rowStride;
    const dstOffset = y * rowStride;
    imageData.data.set(pixels.subarray(srcOffset, srcOffset + rowStride), dstOffset);
  }

  ctx.putImageData(imageData, 0, 0);
  return snapshotCanvas;
}

async function captureBlobFromCanvas(sourceCanvas, fileName, download) {
  return await new Promise((resolve, reject) => {
    sourceCanvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error('captureFrame failed: snapshot canvas toBlob returned null'));
        return;
      }

      if (download) {
        downloadBlob(blob, fileName);
      }

      resolve(blob);
    }, 'image/png');
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

function captureSnapshotCanvasFromGpu(gpu) {
  const gl = gpu?.gl;
  if (!gl) {
    throw new Error('captureFrame failed: WebGL renderer is not ready');
  }

  const width = gl.drawingBufferWidth | 0;
  const height = gl.drawingBufferHeight | 0;
  if (width <= 0 || height <= 0) {
    throw new Error('captureFrame failed: drawing buffer is empty');
  }

  const pixels = new Uint8Array(width * height * 4);
  gl.finish();
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  return createSnapshotCanvasFromPixels(width, height, pixels);
}

async function renderCurrentFrame(options = {}) {
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
    deterministicStateSummary: buildDeterministicStateSummary()
  });
  if (renderResult || !options.preservePreviousOnNull) {
    latestRenderResult = renderResult;
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
    captureLiveSameStateTileAndAssociationDebug,
    downloadLiveSameStateTileAndAssociationDebugJson,
    saveCurrentCanvasPng,
    captureCurrentCanvasPng: saveCurrentCanvasPng,
    getLatestDebugText: () => refreshLatestDebugText(),
    getLastRenderResult: () => latestRenderResult,
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
