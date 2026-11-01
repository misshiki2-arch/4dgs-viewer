// Input authority, not observation-derived control. No GPU/browser imports.
import { assertNormalFrameScope, NORMAL_POLICY_VERSION } from './common_4dgs_normal_on.js';
export const COMPOSITING_MATH_VERSION = 'cuda-aligned-composite-v1';
export const COMPOSITING_POLICY_VERSION = 'adopted-range-compositing-policy-v1';
const verified = new WeakSet();
const equal = (a, b) => JSON.stringify(order(a)) === JSON.stringify(order(b));
function order(x) {
  if (Array.isArray(x)) return x.map(order);
  return x && typeof x === 'object'
    ? Object.fromEntries(Object.keys(x).sort().map(k => [k, order(x[k])])) : x;
}
function check(ok, reason) { if (!ok) throw new Error('compositing-policy:' + reason); }
function freeze(x) {
  if (x && typeof x === 'object') { Object.values(x).forEach(freeze); Object.freeze(x); }
  return x;
}

export function resolveCompositingPolicy({ request = null, adoptedBundle = null,
  requestIdentity = null, normalScope = null } = {}) {
  let purpose = 'normal', mode = 'NEW-OFF', candidateIdentity = null;
  let normalFields = {};
  if (request !== null) {
    check(equal(Object.keys(request).sort(), ['authorization','authority','mathVersion','mode','schema'].sort()), 'fields');
    check(request.schema === COMPOSITING_POLICY_VERSION && request.mathVersion === COMPOSITING_MATH_VERSION &&
      ['NEW-OFF', 'NEW-ON'].includes(request.mode), 'unknown-contract');
    const c = adoptedBundle, a = request.authorization;
    check(c?.profile === 'adopted-b-sh0-bundle-v1' && c.rows === 3248901 &&
      c.selection?.view === 4 && c.selection.start === 524288 && c.selection.count === 65536 &&
      equal(c.frame, ['train', 2588, 'images/000118_v16.png', 16.6]) &&
      c.camera?.width === 1280 && c.camera.height === 720 &&
      equal(c.activeSh, [0,0]) && equal(c.maximumSh, [3,2]) && c.physicalSh === 16 &&
      equal(request.authority, c.external), 'scope');
    check(a?.schema === 'adopted-range-execution-authorization-v1' && a.execution_authorized === true &&
      a.purpose === 'validation-only' && a.mode === request.mode &&
      /^[a-f0-9]{64}$/.test(a.candidate_sha256) &&
      typeof requestIdentity === 'string' && requestIdentity.length > 0, 'authorization');
    purpose = 'validation-only'; mode = request.mode; candidateIdentity = a.candidate_sha256;
  } else if (normalScope !== null) {
    const token = assertNormalFrameScope(normalScope);
    check(typeof requestIdentity === 'string' && requestIdentity.length > 0, 'normal-request');
    mode = 'NEW-ON';
    normalFields = {schema:NORMAL_POLICY_VERSION,scopeIdentity:token.scopeIdentity,
      normalApplicationAuthorized:true,profile:token.profile};
  }
  const value = freeze({ schema: COMPOSITING_POLICY_VERSION, mathVersion: COMPOSITING_MATH_VERSION,
    purpose, requestedMode: mode, appliedMode: mode, requestIdentity, candidateIdentity,
    normalOnAccepted: false, alphaCap: Math.fround(0.99),
    alphaSkipBelow: Math.fround(1/255), terminationBelow: Math.fround(1e-4), ...normalFields });
  verified.add(value);
  return value;
}

export function assertCompositingPolicy(policy) {
  check(verified.has(policy), 'unverified-policy');
  return policy;
}

export function compositingFrameFields(policy) {
  assertCompositingPolicy(policy);
  return { compositingMathVersion: policy.mathVersion, compositingMode: policy.appliedMode,
    compositingRequestIdentity: policy.requestIdentity,
    ...(policy.schema === NORMAL_POLICY_VERSION ? {compositingPurpose:policy.purpose,
      compositingScopeIdentity:policy.scopeIdentity} : {}) };
}

export function validateCompositingResources(device, width, height) {
  check(Number.isSafeInteger(width) && width > 0 && Number.isSafeInteger(height) && height > 0,
    'viewport');
  const limits = device?.limits;
  check(Number.isFinite(limits?.maxStorageBuffersPerShaderStage) && limits.maxStorageBuffersPerShaderStage >= 8,
    'storage-bindings');
  for (const bytes of [width*height*16, width*height*4]) {
    check(Number.isSafeInteger(bytes) && bytes <= limits?.maxStorageBufferBindingSize && bytes <= limits?.maxBufferSize,
      'storage-capacity');
  }
}

// Bounded, versioned observer. Legacy Step104 fields are historical, not this
// execution's pixel counters. Production never consumes this result as policy.
export function compositingExecutionEvidence(policy, completed, terminatedPixelCount, pixels) {
  assertCompositingPolicy(policy);
  check(typeof completed === 'boolean' && Number.isSafeInteger(terminatedPixelCount) &&
    terminatedPixelCount >= 0 && terminatedPixelCount <= pixels &&
    (policy.appliedMode === 'NEW-ON' || terminatedPixelCount === 0), 'termination-count');
  return { ...policy, executed: completed, onPathExecuted: completed && policy.appliedMode === 'NEW-ON',
    terminatedPixelCount, terminationObserved: terminatedPixelCount > 0,
    evaluatedPixelReferenceCount: null, skippedPixelReferenceCount: null,
    unmeasuredReason: 'pixel-reference-total-not-measured',
    coverageUnit: 'tile-reference', offOnComparisonComplete: false, accepted: false };
}
