// App-local capture admission. Scheduler, rendering and last-valid display are unchanged.
import { assertCompositingPolicy, compositingFrameFields } from './common_4dgs_compositing_policy.js';
import { normalCaptureFields, NORMAL_POLICY_VERSION } from './common_4dgs_normal_on.js';
const check = (ok, reason) => { if (!ok) throw new Error('capture-admission:' + reason); };
const heartbeat = r => r?.source === 'viewer-loop-continuation' &&
  r.forceProductionUpdate !== true && r.metadata == null;
const snapshotOf = result => result?.webgpuProductionFrameDataPathContract?.productionEvaluationInputContract;

export function createCaptureAdmission() {
  let current = null;
  function pending(id, raw, loadGeneration) {
    current = {id, raw, loadGeneration, status:'pending', policy:null, generation:null};
  }
  function matches(snapshot, policy) {
    const p = snapshot?.compositingPolicy, i = snapshot?.productionIdentity;
    if (!policy || snapshot?.status !== 'ready' || p?.executed !== true ||
        !Number.isSafeInteger(i?.productionGeneration) || i.productionGeneration <= 0 ||
        i.productionGeneration !== i.presentedGeneration) return false;
    for (const key of ['schema','mathVersion','purpose','appliedMode','requestIdentity',
      'candidateIdentity','scopeIdentity','normalApplicationAuthorized','profile'])
      if (p[key] !== policy[key]) return false;
    for (const frame of [i.productionFrameIdentity, i.presentedFrameIdentity]) {
      if (frame?.generation !== i.productionGeneration) return false;
      for (const [key,value] of Object.entries(compositingFrameFields(policy)))
        if (frame[key] !== value) return false;
    }
    return true;
  }
  return {
    reset() { current = null; },
    scheduled(request, raw, generation) {
      if (!heartbeat(request)) pending(request.requestIdentity, raw, generation);
    },
    prepared(frame, policy, raw, generation) {
      assertCompositingPolicy(policy);
      // A clean continuation is observation, not a new content authority. A
      // policy transition from a completed frame must produce its own frame.
      if (!current || (current.status === 'completed' &&
          (current.raw !== raw || current.loadGeneration !== generation ||
           current.policy.purpose !== policy.purpose || current.policy.appliedMode !== policy.appliedMode)))
        pending(frame?.requestIdentity, raw, generation);
      if (current.id === frame?.requestIdentity && current.raw === raw && current.loadGeneration === generation)
        current.policy = policy;
    },
    completed(frame, result, raw, generation) {
      if (!current || current.id !== frame?.requestIdentity) return;
      const snapshot = snapshotOf(result);
      current.status = current.raw === raw && current.loadGeneration === generation &&
        matches(snapshot, current.policy) ? 'completed' : 'failed';
      if (current.status === 'completed') current.generation = snapshot.productionIdentity.productionGeneration;
    },
    failed(frame) {
      if (current && (current.id === frame?.requestIdentity || current.status === 'completed'))
        current.status = 'failed';
    },
    admit({raw, loadGeneration, token, snapshot, options = {}}) {
      const record = current;
      check(record?.status === 'completed' && record.raw === raw && record.loadGeneration === loadGeneration &&
        matches(snapshot, record.policy) && snapshot.productionIdentity.productionGeneration === record.generation,
      'current-content-not-presented');
      const fields = record.policy.schema === NORMAL_POLICY_VERSION
        ? normalCaptureFields(token, snapshot, record.id, options)
        : compositingFrameFields(record.policy);
      for (const [key,value] of Object.entries({...fields, compositingPurpose:record.policy.purpose,
        compositingScopeIdentity:record.policy.scopeIdentity ?? null}))
        check(options[key] === undefined || options[key] === value, 'option-conflict:' + key);
      // The closure is retained across readback/PNG awaits. A new request,
      // failed heartbeat or scene/load swap cannot publish the old capture.
      return {fields, generation:record.generation, assertCurrent(nextRaw, nextGeneration) {
        check(current === record && record.status === 'completed' && record.raw === nextRaw &&
          record.loadGeneration === nextGeneration, 'changed-during-capture');
      }};
    }
  };
}
