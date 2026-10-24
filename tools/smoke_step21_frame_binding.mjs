import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { webcrypto } from 'node:crypto';
import { bundleFixture, readySnapshot } from './step21_adopted_bundle_fixture.mjs';
import { parseSplat4DV2 } from '../demo/js/splat4d_parser_v2.js';
import { loadVerifiedAdoptedBundle, verifyAdoptedSnapshot, adoptedCaptureBinding,
  adoptedDeterministicState } from '../demo/js/common_4dgs_adopted_bundle.js';
import { compareWebGpuPresentationFrameIdentity } from '../demo/js/webgpu_presentation_capture_orientation_contract.js';
import { buildWebGpuRenderStateManifest } from '../demo/js/common_4dgs_render_state_manifest.js';
import { createPopulationSemanticComparisonController } from '../demo/js/webgpu_population_semantic_comparison_controller.js';

// Read-only real failure evidence is supplied explicitly. No browser/GPU/network.
const [browserPath, pngPath, mode] = process.argv.slice(2);
const browser = JSON.parse(await readFile(browserPath, 'utf8'));
const png = JSON.parse(await readFile(pngPath, 'utf8'));
const copy = value => JSON.parse(JSON.stringify(value));
globalThis.crypto ??= webcrypto;
const actual = browser.productionSnapshot.productionIdentity;
const fields = ['generation', 'datasetCameraLabel', 'datasetFrameNumber', 'datasetTime',
  'referenceCameraLabel', 'outputWidth', 'outputHeight'];
assert.equal(browser.failure.message, 'adopted-bundle:snapshot-unpresented');
assert.equal(browser.phase, 'existing-fresh-production-capture');
assert.equal(browser.semanticComparison, null);
assert.equal(actual.productionFrameIdentity.frameHash, null);
assert.equal(png.capturedFrameIdentity.frameHash, null);
assert.equal(typeof actual.presentedFrameIdentity.frameHash, 'string');
assert.ok(actual.presentedFrameIdentity.frameHash.length > 0);
assert.deepEqual(png.presentedFrameIdentity, actual.presentedFrameIdentity);
assert.equal(png.captureMatchesPresentedFrame, true);
assert.equal(png.staleCaptureDetected, false);
for (const field of fields) {
  assert.notEqual(actual.productionFrameIdentity[field], null);
  assert.equal(actual.productionFrameIdentity[field], actual.presentedFrameIdentity[field]);
  assert.equal(png.capturedFrameIdentity[field], actual.presentedFrameIdentity[field]);
}

// Synthetic geometry/publication, real saved frame tuples without changing their
// values. Keep the approved range so the actual comparison controller is reached.
// This is not the real B payload nor an image/SH oracle.
const selection = copy(browser.inputContext.selection);
const fixture = bundleFixture(({ manifest }) => {
  const r = manifest.references[selection.view];
  r.frame.splice(0, 4, ...browser.inputContext.frame);
  r.camera.values.timestamp = r.frame[3];
  r.camera.values.key = r.frame.slice(0, 3);
  r.native.scalars[23] = r.frame[3];
}, selection.start + selection.count);
const raw = await loadVerifiedAdoptedBundle({ ...fixture, selection, parseArrayBuffer: parseSplat4DV2 });
fixture.files.clear();
const snapshot = copy(readySnapshot(raw));
assert.equal(snapshot.status, 'ready');
snapshot.productionIdentity = copy(actual);
const capture = { presentedStateIdentity: copy(png.presentedFrameIdentity),
  capturedStateIdentity: copy(png.capturedFrameIdentity) };
const manifest = (s = snapshot, p = capture) => buildWebGpuRenderStateManifest({ raw,
  rawSummary: { assetSha256: raw.assetSha256 }, productionEvaluationInputContract: s,
  pngCaptureStatus: p, deterministicState: adoptedDeterministicState(raw),
  canvasSizeSummary: { width: 1280, height: 720 } });
async function controller(s) {
  let received = null, adapters = 0, destroyed = false;
  const c = createPopulationSemanticComparisonController({
    requestAdapter: async () => { adapters++; return { requestDevice: async () => ({
      queue: { onSubmittedWorkDone: async () => {} }, destroy() { destroyed = true; }
    }) }; },
    // Explicit external boundary, not a fabricated semantic match result.
    runOrchestrator: async input => { received = input; throw new Error('CPU-boundary-no-GPU-execution'); }
  });
  const result = await c.runPopulationAlignedSemanticComparison({ raw, productionEvaluationInputContract: s });
  return { result, received, adapters, destroyed };
}

if (mode === '--expect-old-rejection') {
  assert.throws(() => verifyAdoptedSnapshot(raw, snapshot), /snapshot-unpresented/);
  assert.throws(() => adoptedCaptureBinding(raw, snapshot, capture), /snapshot-unpresented/);
  // Independently expose the downstream whole-object capture comparison.
  const s = copy(snapshot); s.productionIdentity.presentedFrameIdentity.frameHash = null;
  assert.ok(verifyAdoptedSnapshot(raw, s));
  assert.throws(() => adoptedCaptureBinding(raw, s, capture), /capture-stale-or-unpresented/);
  assert.equal(manifest().adoptedHandoff.status, 'blocked');
  const c = await controller(snapshot);
  assert.equal(c.adapters, 0); assert.equal(c.received, null);
  assert.match(JSON.stringify(c.result), /snapshot-unpresented/);
  console.log(JSON.stringify({ status: 'pass', mode, oldSnapshotCaptureAndControllerRejectionReproduced: true }));
} else {
  assert.equal(mode, undefined);
  const before = JSON.stringify({ snapshot, capture });
  assert.ok(verifyAdoptedSnapshot(raw, snapshot));
  const binding = adoptedCaptureBinding(raw, snapshot, capture);
  assert.equal(binding.status, 'ready'); assert.equal(binding.pngBound, true);
  assert.equal(binding.imageAgreementVerified, false);
  assert.equal(binding.productionIdentity, snapshot.productionIdentity);
  assert.equal(manifest().adoptedHandoff.status, 'ready');
  assert.equal(manifest().srcIndexSemantics.status, 'verified-direct');
  assert.equal(manifest().adoptedHandoff.imageAgreementVerified, false);
  const c = await controller(snapshot);
  assert.ok(c.received, JSON.stringify(c.result));
  assert.equal(c.received.raw, raw); assert.equal(c.adapters, 1); assert.equal(c.destroyed, true);
  assert.equal(c.received.populationContractIdentity.residentRecordCount, selection.count);
  assert.equal(c.received.buildConfig.timestamp, actual.productionFrameIdentity.datasetTime);
  assert.notEqual(c.result.decision, 'match'); // GPU work was deliberately not run.
  assert.equal(JSON.stringify({ snapshot, capture }), before);
  let rejected = 0, accepted = 1;
  const endpoints = (s, p) => [s.productionIdentity.productionFrameIdentity,
    s.productionIdentity.presentedFrameIdentity, p.presentedStateIdentity, p.capturedStateIdentity];
  for (let endpoint = 0; endpoint < 4; endpoint++) for (const field of fields) {
    for (const missing of [false, true]) {
      const s = copy(snapshot), p = copy(capture), f = endpoints(s, p)[endpoint];
      if (missing) delete f[field];
      else f[field] = typeof f[field] === 'number' ? f[field] + 1 : f[field] + '-different';
      assert.throws(() => adoptedCaptureBinding(raw, s, p), /adopted-bundle:/);
      assert.equal(manifest(s, p).adoptedHandoff.status, 'blocked'); rejected++;
    }
  }
  // All pairs of observed hashes must agree, even if intervening endpoints have
  // no observation. The PNG blob SHA is intentionally not a frameHash oracle.
  for (let left = 0; left < 4; left++) for (let right = left + 1; right < 4; right++) {
    const s = copy(snapshot), p = copy(capture), fs = endpoints(s, p);
    fs.forEach(f => { f.frameHash = null; });
    fs[left].frameHash = 'cpu-observation-one'; fs[right].frameHash = 'cpu-observation-two';
    assert.throws(() => adoptedCaptureBinding(raw, s, p), /adopted-bundle:/); rejected++;
    fs[right].frameHash = fs[left].frameHash;
    assert.equal(adoptedCaptureBinding(raw, s, p).status, 'ready'); accepted++;
  }
  for (let observed = -1; observed < 4; observed++) {
    const s = copy(snapshot), p = copy(capture), fs = endpoints(s, p);
    fs.forEach((f, i) => { f.frameHash = i === observed ? 'cpu-only-observation' : null; });
    const original = JSON.stringify({ s, p });
    assert.equal(adoptedCaptureBinding(raw, s, p).status, 'ready'); accepted++;
    assert.equal(JSON.stringify({ s, p }), original);
  }
  const corruptions = [
    s => { s.productionIdentity.presentedGeneration = null; },
    s => { s.productionIdentity.presentedGeneration++; },
    s => { s.assetIdentity.spl4AssetIdentity.sha256 = 'wrong-asset'; },
    s => { s.adoptedBundle.selection.start++; },
    s => { s.appliedConfig.timestamp++; },
    s => { s.appliedConfig.scalingModifier++; },
    s => { s.appliedConfig.sigmaScale++; },
    s => { s.appliedConfig.prefilterVar++; },
    s => { s.appliedConfig.effectiveRenderScale = 0.5; },
    s => { s.projection.values[12]++; },
    s => { s.productionWorksetIdentity.residentStart++; }
  ];
  for (const corrupt of corruptions) {
    const s = copy(snapshot); corrupt(s);
    assert.throws(() => verifyAdoptedSnapshot(raw, s), /adopted-bundle:/);
    assert.equal(manifest(s).adoptedHandoff.status, 'blocked');
    const blocked = await controller(s);
    assert.equal(blocked.adapters, 0); assert.equal(blocked.received, null); rejected++;
  }
  for (const corrupt of [
    s => { s.productionIdentity.presentedFrameIdentity.frameHash = 'conflicting-observation';
      s.productionIdentity.productionFrameIdentity.frameHash = 'other-observation'; },
    s => { delete s.productionIdentity.presentedFrameIdentity.referenceCameraLabel; }
  ]) {
    const s = copy(snapshot); corrupt(s); const blocked = await controller(s);
    assert.equal(blocked.adapters, 0); assert.equal(blocked.received, null); rejected++;
  }
  assert.deepEqual(compareWebGpuPresentationFrameIdentity(actual.productionFrameIdentity,
    actual.presentedFrameIdentity).requiredKeys, fields);
  console.log(JSON.stringify({ status: 'pass', accepted, rejected,
    savedEvidenceFrameValuesPreserved: true, requiredFieldMutations: 56,
    observedHashContradictionPairs: 6, actualConsumers: ['snapshot', 'capture', 'manifest', 'comparison-controller'],
    externalBoundary: 'fake diagnostic adapter/device; orchestrator input captured then explicit throw; no GPU or semantic match claimed',
    geometry: 'synthetic publication using approved range; not real B payload', imageAgreementVerified: false }));
}
