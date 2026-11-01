// Normal-entry wiring only. The verified loader and frame validator own input semantics.
import { NORMAL_ADOPTED_SCOPE as scope, NORMAL_SCOPE_IDENTITY } from './common_4dgs_normal_adopted_scope.js';
import { getAdoptedBundleContext, validateAdoptedFrame, adoptedFrameArguments } from './common_4dgs_adopted_bundle.js';
import { buildWebGpuProjectionContract } from './common_4dgs_projection_contracts.js';
import { PRODUCTION_DEPTH_ORDER, PRODUCTION_SORT_ALGORITHM } from './common_4dgs_record_contracts.js';

export const NORMAL_POLICY_VERSION = 'adopted-normal-compositing-policy-v2';
const scenes = new WeakSet(), tokens = new WeakSet();
const canonical = x => Array.isArray(x) ? x.map(canonical) : x && typeof x === 'object'
  ? Object.fromEntries(Object.keys(x).sort().map(k => [k, canonical(x[k])])) : x;
const equal = (a,b) => JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));
const check = (ok,reason) => { if (!ok) throw new Error('normal-on:'+reason); };
export const normalScopeIdentity = () => NORMAL_SCOPE_IDENTITY;
export const isNormalAdoptedScene = raw => !!raw && scenes.has(raw);

export function normalStartupParameters(search) {
  const params = new URLSearchParams(search || '');
  if (!params.has('adoptedProfile')) return params;
  check(params.getAll('adoptedProfile').length === 1 && params.get('adoptedProfile') === scope.id, 'unknown-profile');
  const values = { adoptedProfile:scope.id, viewerRuntime:'webgpu',
    webgpuBackendMode:'webgpu-exclusive', webgpuBackendImplementation:'webgpu-tile-compositor-frame-implementation',
    webgpuAllowViewerCanvasPresentation:'true', webgpuBackendViewerLoopHook:'true',
    fixedCanvasWidth:'1280', fixedCanvasHeight:'720', time:String(scope.frame[3]),
    stride:'1', renderScale:'1', splatScale:'1', sigmaScale:'1', prefilterVar:'0', bgGray:'0',
    useSH:'true', useRot4d:'true', useNativeRot4d:'true', useNativeMarginal:'true',
    webgpuProductionResidentSelectionMode:'range',
    webgpuProductionResidentRangeStart:String(scope.input.selection.start),
    webgpuProductionResidentRangeCount:String(scope.input.selection.count) };
  for (const [key,value] of params) check(Object.hasOwn(values,key) && values[key] === value &&
    params.getAll(key).length === 1, 'query-conflict:'+key);
  return new URLSearchParams(values);
}

export function normalStartupInput(profile) {
  check(profile === scope.id, 'unknown-profile');
  return { ...scope.input, normalProfile:profile };
}

export function activateNormalAdoptedScene(raw, profile, input) {
  check(profile === scope.id && equal(input, scope.input), 'loader-authority');
  const c = getAdoptedBundleContext(raw);
  check(c && equal(c.external,scope.input.authority) && equal(c.selection,scope.input.selection) &&
    c.rows === scope.rows && equal(c.frame,scope.frame) && equal(c.activeSh,[0,0]) &&
    equal(c.maximumSh,[3,2]) && c.physicalSh === 16, 'scope');
  scenes.add(raw);
}

export function normalFrameScope(input) {
  if (!isNormalAdoptedScene(input.raw)) return null;
  const c = validateAdoptedFrame(input);
  check(c && equal(c.external,scope.input.authority) && equal(c.selection,scope.input.selection), 'scope');
  check(input.backgroundGray === 0, 'background');
  check(scope.mathVersion === 'cuda-aligned-composite-v1' && scope.depthOrder === PRODUCTION_DEPTH_ORDER &&
    scope.sortAlgorithm === PRODUCTION_SORT_ALGORITHM, 'math-sort-contract');
  const token = Object.freeze({ schema:NORMAL_POLICY_VERSION, scopeIdentity:normalScopeIdentity(),
    profile:scope.id, raw:input.raw });
  tokens.add(token);
  return token;
}

export function assertNormalFrameScope(token) {
  check(tokens.has(token) && isNormalAdoptedScene(token.raw), 'unverified-frame');
  getAdoptedBundleContext(token.raw);
  return token;
}

// Same fixed-reference projection used by the native path; no UI/orbit-label shortcut.
export function normalViewerScope(raw, buildConfig, canvas, backgroundGray) {
  if (!isNormalAdoptedScene(raw)) return null;
  const a = adoptedFrameArguments(raw), w = canvas.width, h = canvas.height;
  const projection = buildWebGpuProjectionContract({camera:{},screenSpaceCamera:a.screenSpaceCamera,
    renderW:Math.max(1,Math.round(w*buildConfig.renderScale)),
    renderH:Math.max(1,Math.round(h*buildConfig.renderScale)),
    sx:1/buildConfig.renderScale,sy:1/buildConfig.renderScale});
  return normalFrameScope({raw,buildConfig,canvasWidth:w,canvasHeight:h,backgroundGray,
    projectionData:projection.data,productionResidentSelectionRequest:a.productionResidentSelectionRequest});
}

export function normalCacheMatches(policy, token, rawMatches = true) {
  if (!token) return policy?.schema !== NORMAL_POLICY_VERSION;
  assertNormalFrameScope(token);
  return rawMatches && policy?.schema === NORMAL_POLICY_VERSION && policy.purpose === 'normal' &&
    policy.appliedMode === 'NEW-ON' && policy.scopeIdentity === token.scopeIdentity && policy.executed === true;
}

export function normalCaptureFields(token, snapshot, currentRequest, options = {}) {
  assertNormalFrameScope(token);
  const p = snapshot?.compositingPolicy, identity = snapshot?.productionIdentity;
  check(normalCacheMatches(p,token) && p.requestIdentity === currentRequest && snapshot.status === 'ready' &&
    identity?.productionGeneration > 0 && identity.productionGeneration === identity.presentedGeneration,
    'capture-stale');
  const fields = {compositingMathVersion:p.mathVersion,compositingMode:'NEW-ON',
    compositingRequestIdentity:currentRequest,compositingPurpose:'normal',compositingScopeIdentity:token.scopeIdentity};
  for (const [key,value] of Object.entries(fields)) {
    check(identity.productionFrameIdentity?.[key] === value && identity.presentedFrameIdentity?.[key] === value,
      'capture-frame:'+key);
    check(options[key] === undefined || options[key] === value, 'capture-request:'+key);
  }
  return fields;
}
