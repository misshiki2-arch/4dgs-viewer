import { buildWebGpuProjectionContract } from './common_4dgs_projection_contracts.js';
import { compareWebGpuPresentationFrameIdentity } from './webgpu_presentation_capture_orientation_contract.js';

export const ADOPTED_B_PROFILE = 'adopted-b-sh0-bundle-v1';
const verifiedScenes = new WeakMap();
const rawFields = ['xyz', 'rotation', 'rotation_r', 'scale_xyz', 'f_dc', 'f_rest', 'opacity', 't', 'scale_t'];
const check = (ok, reason) => { if (!ok) throw new Error(`adopted-bundle:${reason}`); };
function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map((k) => [k, canonical(value[k])]));
  return value;
}
const equal = (a, b) => JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));
const clone = (value) => JSON.parse(JSON.stringify(value));
function freeze(value) {
  if (value && typeof value === 'object') { Object.values(value).forEach(freeze); Object.freeze(value); }
  return value;
}
function requireIdentity(value) {
  check(value && Number.isSafeInteger(value.size) && value.size > 0 &&
    /^[a-f0-9]{64}$/.test(value.sha256), 'identity-required');
}
export async function hashAdoptedBytes(bytes) {
  check(bytes instanceof ArrayBuffer && globalThis.crypto?.subtle, 'hash-unavailable');
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), (v) => v.toString(16).padStart(2, '0')).join('');
}
function memberPath(path) {
  check(typeof path === 'string' && /^[A-Za-z0-9_.-]+(?:\/[A-Za-z0-9_.-]+)*$/.test(path) &&
    !path.split('/').some((v) => v === '.' || v === '..'), 'unsafe-member-path');
  return path;
}
// Only explicit relative members are resolved. Absolute provenance paths are never fetched.
export function createAdoptedBundleUrlReader(baseUrl) {
  const base = new URL(baseUrl, globalThis.location?.href);
  check(['http:', 'https:'].includes(base.protocol) && base.pathname.endsWith('/') &&
    !base.search && !base.hash && !base.username && !base.password, 'base-url');
  return async (path, expectedSize) => {
    const url = new URL(memberPath(path), base);
    const response = await fetch(url, { redirect: 'error' });
    check(response.ok && response.body, 'fetch-failed');
    const length = response.headers.get('content-length');
    check(length === null || Number(length) === expectedSize, 'fetch-size');
    const reader = response.body.getReader();
    const bytes = new Uint8Array(expectedSize);
    let offset = 0;
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        check(offset + value.length <= bytes.length, 'fetch-overflow');
        bytes.set(value, offset); offset += value.length;
      }
      check(offset === bytes.length, 'fetch-short');
    } finally { await reader.cancel(); reader.releaseLock(); }
    return bytes.buffer;
  };
}

// authority comes from the independently reviewed external publication evidence,
// not from an unverified member selected alongside the bundle.
export async function loadVerifiedAdoptedBundle({ authority, selection, readResource, parseArrayBuffer }) {
  check(authority && equal(Object.keys(authority).sort(), ['index','manifest','receipt','result','spl4','verification']), 'authority-fields');
  check(selection && equal(Object.keys(selection).sort(), ['count','representatives','start','view']), 'selection-fields');
  const expected = freeze(clone(authority));
  const chosen = freeze(clone(selection));
  for (const key of ['result', 'index', 'manifest', 'verification', 'spl4']) requireIdentity(expected[key]);
  check(typeof readResource === 'function' && typeof parseArrayBuffer === 'function', 'reader-required');
  const bytes = async (path, id) => {
    memberPath(path); requireIdentity(id);
    const buffer = await readResource(path, id.size);
    check(buffer instanceof ArrayBuffer && buffer.byteLength === id.size, 'member-size');
    check(await hashAdoptedBytes(buffer) === id.sha256, 'member-hash');
    return buffer;
  };
  const json = async (path, id) => {
    check(id.size <= 16 * 1024 * 1024, 'json-size');
    return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(await bytes(path, id)));
  };
  const result = await json('result.json', expected.result);
  const receipt = result.receipt;
  check(result.status === 'verified_requires_review' && result.gate_c_passed === false &&
    receipt?.schema === 'adopted-reference-receipt-v1' && receipt.status === result.status &&
    receipt.gate_c_passed === false && equal(receipt.index, expected.index) &&
    equal(receipt.verification, expected.verification), 'publication');
  // The external result contains the exact receipt. Bound its separate file as well.
  // Read size from explicit external evidence; no unbounded locator read.
  requireIdentity(expected.receipt);
  check(equal(await json('receipt.json', expected.receipt), receipt), 'receipt-mismatch');
  for (const name of ['worker_exit', 'reader_exit']) {
    requireIdentity(result[name]);
    const exit = await json(name.replace('_', '-') + '.json', result[name]);
    check(exit.exit_code === 0 && exit.primary === null && exit.stop_confirmed === true, 'child-exit');
  }
  const index = await json('bundle/index.json', expected.index);
  check(index.schema === 'adopted-bundle-index-v1' && index.members && !Array.isArray(index.members), 'index-schema');
  const members = index.members;
  const names = ['manifest.json', 'population.json', 'population.splat4d', 'verification.json'];
  for (let v = 0; v < 14; v++) for (const name of [
    'render.npy', 'alpha.npy', 'depth.npy', 'radii.npy', 'visibility_filter.npy', 'gt.npy',
    'render.png', 'alpha.png', 'depth.png', 'gt.png', 'runtime.json'
  ]) names.push(`references/${String(v).padStart(2, '0')}/${name}`);
  check(equal(Object.keys(members).sort(), names.sort()), 'member-set');
  Object.entries(members).forEach(([p, id]) => { memberPath(p); requireIdentity(id); });
  check(equal(members['manifest.json'], expected.manifest) &&
    equal(members['verification.json'], expected.verification) &&
    equal(members['population.splat4d'], expected.spl4), 'external-index-binding');
  const manifest = await json('bundle/manifest.json', expected.manifest);
  const population = await json('bundle/population.json', members['population.json']);
  const verification = await json('bundle/verification.json', expected.verification);
  check(manifest.schema === 'adopted-b-reference-bundle-v1' && manifest.version === 1 &&
    population.schema === 'adopted-population-v1' && equal(population, manifest.population), 'manifest-schema');
  check(Number.isSafeInteger(population.rows) && population.rows > 0 && population.header_bytes === 128 &&
    population.stride === 260 && population.physical_sh === 16 && equal(population.active_sh, [0, 0]) &&
    equal(population.maximum_sh_labels, [3, 2]) && equal(population.identity, expected.spl4) &&
    population.mapping === 'sourceIndex = assetRecordIndex = CUDA localIndex' &&
    population.quaternion === 'raw wxyz' && population.scales === 'raw log' &&
    population.opacity === 'raw logit' && population.sh === 'raw coefficient-major RGB', 'population-profile');
  check(verification.schema === 'adopted-reference-verification-v1' &&
    verification.status === 'verified_requires_review' && verification.gate_c_passed === false &&
    ['all_rows', 'raw14_exact', 'gt14_raw_exact', 'png14_exact', 'separate_cpu_process'].every((k) => verification[k] === true) &&
    equal(verification.selected, manifest.input?.selected), 'verification-binding');
  for (const k of ['marker', 'metadata', 'state']) requireIdentity(manifest.input.selected[k]);
  const renderer = manifest.input.profile?.renderer;
  check(renderer?.scaling_modifier === 1 && renderer.temporal_prefilter === 'disabled' &&
    renderer.compute_cov3D_python === false && renderer.convert_SHs_python === false &&
    renderer.env_map_res === 0 && renderer.override_color === null, 'renderer-profile');
  check(Array.isArray(manifest.references) && manifest.references.length === 14, 'reference-count');
  for (let view = 0; view < 14; view++) {
    const ref = manifest.references[view];
    const c = ref?.camera, n = ref?.native, s = n?.scalars;
    check(ref?.schema === 'adopted-view-runtime-v1' && ref.invocation === view && n?.call_count === 1 &&
      n.population_rows === population.rows && equal(n.tensors?.['19']?.shape, [population.rows, 16, 3]) &&
      s?.['20'] === 0 && s['21'] === 0 && s['10'] === 1 &&
      c?.matrix_layout === 'row-major storage of transposed transforms' && c.time_divisor === 1 &&
      c.time_duration === 33 && c.values.timestamp === ref.frame?.[3] && s['23'] === ref.frame[3] && s['24'] === 33 &&
      c.values.width === s['18'] && c.values.height === s['17'] &&
      equal(c.values.world_view_transform, n.matrices?.['13']) &&
      equal(c.values.full_proj_transform, n.matrices?.['14']) &&
      equal(c.values.camera_center, n.matrices?.['22']) &&
      equal(ref.frame, manifest.input.selected.representatives?.[view]), 'reference-camera-binding');
  }
  // Sequential verification releases other view payloads before the SPL4/SoA allocation.
  for (const path of names) {
    if (['manifest.json', 'population.json', 'population.splat4d', 'verification.json'].includes(path)) continue;
    if (path.endsWith('/runtime.json')) {
      const view = Number(path.split('/')[1]);
      check(equal(await json('bundle/' + path, members[path]), manifest.references[view]), 'runtime-binding');
    } else await bytes('bundle/' + path, members[path]);
  }
  check(Number.isSafeInteger(chosen.view) && chosen.view >= 0 && chosen.view < 14 &&
    Number.isSafeInteger(chosen.start) && chosen.start >= 0 && Number.isSafeInteger(chosen.count) &&
    chosen.count > 0 && chosen.start + chosen.count <= population.rows &&
    Array.isArray(chosen.representatives) && chosen.representatives.length > 0 && chosen.representatives.length <= 32 &&
    chosen.representatives.every((i) => Number.isSafeInteger(i) && i >= chosen.start && i < chosen.start + chosen.count), 'selection');
  const buffer = await bytes('bundle/population.splat4d', expected.spl4);
  const raw = parseArrayBuffer(buffer);
  check(raw.N === population.rows && raw.activeShDegree === 0 && raw.activeShDegreeT === 0 &&
    raw.rot4d && raw.storeScaleLog && raw.rawSh && raw.rawOpacity && raw.frestDim === 45, 'parsed-profile');
  raw.assetSha256 = expected.spl4.sha256; raw.assetSizeBytes = expected.spl4.size;
  raw.assetSourceKind = ADOPTED_B_PROFILE;
  raw.handoffProfile = ADOPTED_B_PROFILE;
  const runtime = manifest.references[chosen.view];
  const context = freeze({ profile: ADOPTED_B_PROFILE, external: expected, selection: chosen,
    checkpoint: { marker: manifest.input.selected.marker, metadata: manifest.input.selected.metadata, state: manifest.input.selected.state },
    rows: raw.N, activeSh: [0, 0], maximumSh: [3, 2], physicalSh: 16,
    runtimeIdentity: members[`references/${String(chosen.view).padStart(2, '0')}/runtime.json`],
    camera: runtime.camera.values, frame: runtime.frame,
    timeDuration: runtime.camera.time_duration, timeDivisor: runtime.camera.time_divisor,
    coverage: 'all-indexed-members', fileScales: 'log', runtimeScales: 'linear' });
  verifiedScenes.set(raw, { context, arrays: rawFields.map((k) => raw[k]) });
  return raw;
}

export function getAdoptedBundleContext(raw) {
  const record = raw && verifiedScenes.get(raw);
  if (!record) { check(!raw?.handoffProfile, 'unverified-scene'); return null; }
  const c = record.context;
  check(raw.handoffProfile === c.profile && raw.N === c.rows && raw.activeShDegree === 0 &&
    raw.activeShDegreeT === 0 && raw.assetSha256 === c.external.spl4.sha256 &&
    raw.assetSizeBytes === c.external.spl4.size && raw.assetSourceKind === c.profile &&
    raw.rot4d && raw.storeScaleLog && raw.rawSh && raw.rawOpacity &&
    equal([raw.xyzDim,raw.rotationDim,raw.rotationRDim,raw.scaleXYZDim,raw.fdcDim,raw.frestDim,raw.opacityDim,raw.tDim,raw.scaleTDim], [3,4,4,3,3,45,1,1,1]) &&
    rawFields.every((k, i) => raw[k] === record.arrays[i] && raw[k].length === [3,4,4,3,3,45,1,1,1][i]*c.rows), 'scene-replaced');
  return c;
}

export function adoptedFrameArguments(raw) {
  const c = getAdoptedBundleContext(raw);
  if (!c) return null;
  const m = c.camera.world_view_transform;
  check(Array.isArray(m) && m.length === 4 && m.every((row) => Array.isArray(row) && row.length === 4 && row.every(Number.isFinite)), 'view-matrix');
  const viewMatrix = m.map((_, i) => m.map((row) => row[i]));
  return { canvasWidth: c.camera.width, canvasHeight: c.camera.height,
    screenSpaceCamera: { screenSpaceTransformOverride: { mode: 'cuda-aligned', viewMatrix,
      intrinsics: { fx: c.camera.fx, fy: c.camera.fy, cx: c.camera.cx, cy: c.camera.cy },
      pixelXSign: 1, viewMatrixSource: 'verified-adopted-runtime' } },
    buildConfig: { timestamp: c.camera.timestamp, timeDuration: c.timeDuration,
      scalingModifier: 1, sigmaScale: 1, prefilterVar: 0, renderScale: 1, stride: 1, useSH: true },
    productionResidentSelectionRequest: { mode: 'range', rangeStart: c.selection.start, rangeCount: c.selection.count } };
}

// Explicit fixed-Reference mode; UI/orbit settings are not evidence for this mode.
export function adoptedDeterministicState(raw, fallback = {}) {
  const c = getAdoptedBundleContext(raw);
  if (!c) return fallback;
  const a = adoptedFrameArguments(raw);
  const screen = { mode:'cuda-aligned', enabled:true,
    cudaAlignedViewMatrix:a.screenSpaceCamera.screenSpaceTransformOverride.viewMatrix,
    intrinsics:a.screenSpaceCamera.screenSpaceTransformOverride.intrinsics,
    pixelXSign:1,screenYSign:1,depthSign:1,
    viewMatrixSource:'verified-adopted-runtime',projectionContract:'cuda-plus-z-forward-fx-fy-cx-cy' };
  const position = [...c.camera.camera_center];
  const forward = screen.cudaAlignedViewMatrix[2].slice(0,3);
  const up = screen.cudaAlignedViewMatrix[1].slice(0,3).map(v => -v);
  const target = position.map((v,i) => v + forward[i]);
  return { ...fallback, active: true, cameraSource: 'verified-adopted-runtime',
    datasetCameraLabel: c.frame[2], imageName: c.frame[2], cudaReferenceLabel: c.frame[2],
    frameNumber: c.frame[1], datasetFrameNumber: c.frame[1], viewId: c.selection.view,
    datasetViewId: c.selection.view, datasetTime: c.frame[3], time: c.frame[3],
    datasetViewMatrixMode: 'cuda-aligned', cameraControlContract: 'fixed-reference-read-only',
    cameraOrientationPolicy: 'cuda-reference-camera-orientation',
    fixedReferenceCameraActivationMode: 'fixed-reference-camera',
    cameraPosition: position, referenceCameraPose: { position, target, up, forward, targetDistance:1 },
    actualCameraPosition: position, actualControlsTarget: target, actualCameraUp: up,
    cameraFoVyDeg: 2 * Math.atan(c.camera.height / (2*c.camera.fy)) * 180 / Math.PI,
    intrinsics: a.screenSpaceCamera.screenSpaceTransformOverride.intrinsics,
    webgpuProductionResidentSelectionMode: 'range',
    webgpuProductionResidentRangeStart: c.selection.start,
    webgpuProductionResidentRangeCount: c.selection.count,
    outputWidth: c.camera.width, outputHeight: c.camera.height,
    bgGray: 0, stride: 1, cudaAlignedScreenSpaceCamera: screen, fixedReferenceScreenSpaceCamera: screen,
    adoptedBundle: c };
}

export function applyAdoptedUiDefaults(raw, ui) {
  const a = adoptedFrameArguments(raw);
  if (!a) return;
  for (const [key, field] of Object.entries({ timestamp: 'timeSlider', timeDuration: 'timeDurationSlider',
    scalingModifier: 'splatScaleSlider', sigmaScale: 'sigmaScaleSlider', prefilterVar: 'prefilterVarSlider',
    renderScale: 'renderScaleSlider', stride: 'strideSlider' })) {
    check(ui[field], 'ui-' + field);
    if (key === 'timestamp') ui[field].max = String(a.buildConfig.timeDuration);
    ui[field].value = String(a.buildConfig[key]);
  }
  ui.useSHCheck.checked = true;
  ui.bgGraySlider.value = '0';
}

export function validateAdoptedFrame({ raw, buildConfig, projectionData, canvasWidth, canvasHeight,
  productionResidentSelectionRequest } = {}) {
  const c = getAdoptedBundleContext(raw);
  if (!c) return null;
  const expected = adoptedFrameArguments(raw);
  check(canvasWidth === expected.canvasWidth && canvasHeight === expected.canvasHeight, 'viewport-mismatch');
  for (const [key, value] of Object.entries(expected.buildConfig)) check(buildConfig?.[key] === value, 'config-' + key);
  check(equal(productionResidentSelectionRequest, expected.productionResidentSelectionRequest), 'range-mismatch');
  const projection = buildWebGpuProjectionContract({ camera: {}, screenSpaceCamera: expected.screenSpaceCamera,
    renderW: canvasWidth, renderH: canvasHeight, sx: 1, sy: 1 });
  check(projectionData instanceof Float32Array && equal(Array.from(projectionData), Array.from(projection.data)), 'projection-mismatch');
  return c;
}

// Stable frame identity is mandatory; a readback frameHash is an optional
// observation of that frame, not a payload or encoded-PNG SHA. Compare every
// observed hash so an unobserved intermediate endpoint cannot hide a conflict.
// Do not fill, clear or otherwise mutate the original observations.
function sameAdoptedFrameObservations(...frames) {
  return frames.every(frame => compareWebGpuPresentationFrameIdentity(frames[0], frame).matches) &&
    new Set(frames.map(frame => frame?.frameHash).filter(hash => hash != null)).size <= 1;
}

export function verifyAdoptedSnapshot(raw, snapshot) {
  const c = getAdoptedBundleContext(raw);
  if (!c) { check(!snapshot?.adoptedBundle, 'snapshot-without-scene'); return null; }
  check(snapshot?.status === 'ready' && equal(snapshot.adoptedBundle, c), 'snapshot-binding');
  const asset = snapshot.assetIdentity?.spl4AssetIdentity;
  check(asset?.sha256 === c.external.spl4.sha256 && asset.sizeBytes === c.external.spl4.size &&
    asset.recordCount === c.rows, 'snapshot-asset');
  const p = snapshot.productionIdentity;
  check(p?.productionGeneration != null && p.productionGeneration === p.presentedGeneration &&
    p.productionFrameIdentity?.generation === p.productionGeneration &&
    sameAdoptedFrameObservations(p.productionFrameIdentity, p.presentedFrameIdentity), 'snapshot-unpresented');
  const frame = p.productionFrameIdentity;
  check(frame.datasetTime === c.frame[3] && frame.datasetCameraLabel === c.frame[2] &&
    frame.datasetFrameNumber === c.frame[1] && frame.outputWidth === c.camera.width &&
    frame.outputHeight === c.camera.height, 'snapshot-frame');
  const a = adoptedFrameArguments(raw);
  for (const key of ['timestamp','scalingModifier','sigmaScale','prefilterVar']) {
    check(snapshot.appliedConfig?.[key] === a.buildConfig[key], 'snapshot-config-' + key);
  }
  check(snapshot.appliedConfig?.configuredRenderScale === a.buildConfig.renderScale &&
    snapshot.appliedConfig?.effectiveRenderScale === a.buildConfig.renderScale, 'snapshot-render-scale');
  const projection = buildWebGpuProjectionContract({camera:{},screenSpaceCamera:a.screenSpaceCamera,
    renderW:a.canvasWidth,renderH:a.canvasHeight,sx:1,sy:1});
  check(equal(snapshot.projection?.values, Array.from(projection.data)), 'snapshot-projection');
  const workset = snapshot.productionWorksetIdentity;
  check(workset?.residentStart === c.selection.start && workset.residentRecordCount === c.selection.count &&
    workset.residentEndExclusive === c.selection.start + c.selection.count, 'snapshot-range');
  return c;
}

export function adoptedCaptureBinding(raw, snapshot, pngStatus = null) {
  const context = verifyAdoptedSnapshot(raw, snapshot);
  if (!context) return null;
  if (pngStatus) {
    const presented = pngStatus.presentedStateIdentity ?? pngStatus.presentedFrameIdentity;
    const captured = pngStatus.capturedStateIdentity ?? pngStatus.captureStateIdentity;
    check(sameAdoptedFrameObservations(snapshot.productionIdentity.productionFrameIdentity,
      snapshot.productionIdentity.presentedFrameIdentity, presented, captured), 'capture-stale-or-unpresented');
  }
  return { status: 'ready', context, productionIdentity: snapshot.productionIdentity,
    pngBound: pngStatus != null, imageAgreementVerified: false };
}
