import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { resolve } from 'node:path';

// All module execution is CPU-only, in a virtual-clock VM. No server or GPU.
const [reportsRoot] = process.argv.slice(2);
const root = resolve(reportsRoot);
const jsonRoot = resolve(root, '../../../../..', 'json');
const read = p => readFile(p, 'utf8');
const copy = x => JSON.parse(JSON.stringify(x));
const oldSource = await read(resolve(root, '../step21-validation2/step21-validation2-capture.js'));
const newSource = await read(resolve(root, 'step21-fix2-capture-candidate.mjs'));
const wrapper = await read(resolve(root, 'step21-fix2-browser-candidate.mjs'));
const diagnostic = JSON.parse(await read(resolve(jsonRoot, 'step21_validation2_000118_v16_presentation_diagnostic.json')));
const failure = JSON.parse(await read(resolve(jsonRoot, 'step21_validation2_000118_v16_browser_result.json')));
const input = JSON.parse(await read(resolve(root, '../step21-validation2/step21-validation2-input.json')));
const savedRequest = diagnostic.requestHistory.find(r => r.source === 'generic-fresh-production-artifact-capture');
const savedFrame = diagnostic.frameHistory.find(f => f.requestIdentity === savedRequest.requestIdentity);
const queueMs = savedRequest.frameStartedAtMs - savedRequest.requestedAtMs;
const renderMs = savedRequest.frameCompletedAtMs - savedRequest.frameStartedAtMs;
assert.ok(queueMs < 10000 && renderMs < 10000 && queueMs + renderMs > 10000);
assert.equal(failure.failure.message, 'capture-blocked-fresh-production-frame-incomplete');
assert.equal(failure.semanticComparison, null);
assert.equal(savedFrame.runtimeError, null);

async function execute({ source = newSource, queue = queueMs, duration = renderMs,
  mutateFrame = () => {}, mutateRequest = () => {}, runtimeError = false,
  missing = false, stop = false, drift = false, finalFailure = false,
  wrapperMode = false, importFailure = false, realScheduler = false } = {}) {
  let time = savedRequest.requestedAtMs, schedules = 0, finalCalls = 0, diagnostics = 0,
    pngCalls = 0, semantics = 0, downloads = [], settled = false, error = null;
  const start = time, tasks = [];
  const context = vm.createContext({ URLSearchParams, URL, console: { log() {}, error() {}, warn() {} },
    performance: { now: () => time }, structuredClone, setTimeout: (fn, delay) => { tasks.push({ at: time + delay, fn }); },
    requestAnimationFrame: fn => { tasks.push({ at: time, fn }); }
  });
  const CaptureError = vm.runInContext('Error', context);
  const frame = copy(savedFrame); mutateFrame(frame);
  const req = copy(savedRequest); req.requestedAtMs = start;
  req.frameStartedAtMs = start + queue; req.frameCompletedAtMs = start + queue + duration;
  mutateRequest(req);
  let getTrace = () => {
    const elapsed = time - start;
    let observed = null;
    if (!missing) {
      observed = copy(req);
      if (elapsed < queue) { observed.status = 'pending'; delete observed.frameStartedAtMs; delete observed.frameCompletedAtMs; }
      else if (stop || elapsed < queue + duration) { observed.status = 'running'; delete observed.frameCompletedAtMs; }
      else if (runtimeError) { observed.status = 'error'; observed.error = { name: 'RendererFailure', message: 'original-render-failure' }; }
      if (drift && elapsed > queue + 100) observed.frameStartedAtMs += 100;
    }
    return { snapshotTakenAtMs: time, latestProductionGeneration: elapsed >= queue + duration ? 4 : 3,
      latestProductionRequestIdentity: elapsed >= queue + duration ? req.requestIdentity : 'prior-fixture',
      requestHistory: observed ? [observed] : [], frameHistory: [
        { requestIdentity: 'prior-fixture', productionGeneration: 3, productionFrameCompleted: true },
        ...(!missing && !stop && !runtimeError && elapsed >= queue + duration ? [copy(frame)] : [])] };
  };
  let schedule = async () => { schedules++; return { requestIdentity: req.requestIdentity, requestedAtMs: start }; };
  if (realScheduler) {
    const schedulerModule = new vm.SourceTextModule(await read(new URL('../demo/js/viewer_render_scheduler.js', import.meta.url)), { context });
    const recorderModule = new vm.SourceTextModule(await read(new URL('../demo/js/common_4dgs_initial_production_presentation.js', import.meta.url)), { context });
    await schedulerModule.link(() => {}); await recorderModule.link(() => {});
    await schedulerModule.evaluate(); await recorderModule.evaluate();
    const recorder = recorderModule.namespace.createInitialProductionPresentationRecorder();
    const scheduler = schedulerModule.namespace.createRenderScheduler({
      renderFrame: ({ schedulerFrameState: s }) => new Promise(resolve => {
        tasks.push({ at: time + (s.requestSource === 'prior-fixture' ? queue : duration), fn: () => resolve({}) });
      }),
      onScheduleRequest: r => recorder.recordScheduleRequest(r),
      onFrameStarted: s => recorder.recordFrameStarted(s),
      onFrameCompleted: ({ schedulerFrameState: s }) => recorder.recordFrameCompleted({ schedulerFrameState: s,
        frameEvidence: { ...frame, productionGeneration: s.requestSource === 'prior-fixture' ? 3 : 4,
          requestIdentity: s.requestIdentity, productionSourceRequestIdentity: s.requestIdentity,
          completedAtMs: time } }),
      onFrameError: e => recorder.recordFrameError(e)
    });
    await scheduler.scheduleRender({ source: 'prior-fixture', forceProductionUpdate: true });
    getTrace = () => recorder.getSnapshot();
    schedule = async options => { schedules++; return scheduler.scheduleRender(options); };
  }
  const debug = {
    getSynchronousCommandStartFence: () => ({ synchronousReadOnlyFence: true,
      initialPresentationSnapshot: { latestProductionGeneration: 2 } }),
    validateExpectedProductionRuntimeContract: () => ({ ready: true }),
    waitForViewerDebugDataReady: async () => ({ ready: true }),
    scheduleRender: schedule, getInitialProductionPresentationSnapshot: () => getTrace(),
    waitForFinalCanvasPresentationQuiescence: async expected => {
      finalCalls++;
      assert.ok(time >= start + queue + duration, 'no final check before completion');
      assert.equal(expected.expectedGeneration, 4);
      if (finalFailure) throw new CaptureError('original-final-presentation-error');
      return { browserVisibleResult: true, quiescenceObservation: { quiescent: true },
        finalSourceRequestIdentity: expected.expectedRequestIdentity };
    },
    getLastRenderResult: () => ({}),
    captureWebGpuVisibleRecordDryRunDebug: async () => {
      diagnostics++; assert.ok(time >= start + queue + duration);
      throw new CaptureError('CPU-boundary-no-diagnostic-GPU');
    },
    captureGpuCandidateRuntimeSummaryDebug: async () => {
      throw new CaptureError('CPU-boundary-no-diagnostic-GPU');
    },
    saveCurrentCanvasPng: async () => { pngCalls++; throw new Error('unexpected-PNG'); },
    runPopulationAlignedSemanticComparison: async () => { semantics++; throw new Error('unexpected-semantic'); },
    loadAdoptedBundle: async () => ({ fixture: true }),
    downloadJsonDebug: async result => { downloads.push(copy(result)); }
  };
  context.window = { gpuViewerDebug: debug, location: { search: '?time=16.6&datasetTime=16.6&datasetCameraLabel=images/000118_v16.png&datasetFrameNumber=2588&fixedCanvasWidth=1280&fixedCanvasHeight=720' } };
  context.fetch = async () => ({ ok: true, json: async () => copy(input) });
  const captureModule = new vm.SourceTextModule(source, { context });
  await captureModule.link(() => {});
  let work;
  if (wrapperMode) {
    const binding = new vm.SyntheticModule(['getAdoptedBundleContext', 'adoptedCaptureBinding'], function () {
      this.setExport('getAdoptedBundleContext', () => copy(failure.inputContext));
      this.setExport('adoptedCaptureBinding', () => { throw new Error('unexpected-binding-before-capture'); });
    }, { context });
    const wrapperModule = new vm.SourceTextModule(wrapper, { context,
      importModuleDynamically: async () => {
        if (importFailure) throw new Error('original-import-error');
        await captureModule.evaluate(); return captureModule;
      } });
    await wrapperModule.link(() => binding); await wrapperModule.evaluate();
    work = wrapperModule.namespace.run();
  } else work = captureModule.evaluate();
  work.then(() => { settled = true; }, e => { error = e; settled = true; });
  for (let i = 0; !settled && i < 4000; i++) {
    await new Promise(resolve => setImmediate(resolve));
    if (settled) break;
    tasks.sort((a, b) => a.at - b.at);
    const task = tasks.shift(); assert.ok(task, 'unbounded await/no pending CPU event');
    time = task.at; task.fn();
  }
  assert.ok(settled, 'wait must be bounded');
  assert.equal(schedules, importFailure ? 0 : 1, 'exactly one fresh request, no retry');
  assert.equal(pngCalls, 0); assert.equal(semantics, 0);
  return { error: error && { name: error.name, message: error.message, captureFailure: copy(error.captureFailure ?? null) },
    schedules, finalCalls, diagnostics, elapsedMs: time - start, downloads };
}

const old = await execute({ source: oldSource });
assert.match(old.error.message, /fresh-production-frame-incomplete/);
assert.equal(old.finalCalls, 0); assert.equal(old.diagnostics, 0);
const passed = [];
for (const options of [{}, { queue: 0 }, { realScheduler: true }]) {
  const r = await execute(options);
  assert.match(r.error.message, /CPU-boundary-no-diagnostic-GPU/, JSON.stringify(options));
  assert.equal(r.finalCalls, 1); assert.equal(r.diagnostics, 1);
  passed.push(options.realScheduler ? 'actual-scheduler-recorder' : options.queue === 0 ? 'no-prior-wait' : 'saved-timing-relation');
}
const negatives = {
  'queue-timeout': { queue: 10001 }, 'execution-timeout': { queue: 0, duration: 10001 },
  'missing-request': { missing: true }, 'stopped-render': { stop: true },
  'clock-origin-drift': { drift: true }, 'runtime-error': { runtimeError: true },
  'old-generation': { mutateFrame: f => { f.productionGeneration = 2; } },
  'other-request': { mutateFrame: f => { f.requestIdentity = 'other-fixture'; } },
  'other-source': { mutateFrame: f => { f.productionSourceRequestIdentity = 'other-fixture'; } },
  'incomplete': { mutateFrame: f => { f.productionFrameCompleted = false; } },
  'no-output': { mutateFrame: f => { f.compositorOutputGenerated = false; } },
  'not-presented': { mutateFrame: f => { f.viewerCanvasPresented = false; } },
  'logical-failure': { mutateFrame: f => { f.logicalPresentationSucceeded = false; } }
};
const failures = {};
for (const [name, options] of Object.entries(negatives)) {
  const r = await execute(options);
  assert.ok(r.error, name); assert.equal(r.finalCalls, 0, name); assert.equal(r.diagnostics, 0, name);
  assert.ok(r.error.captureFailure.waitObservation, name);
  assert.equal(r.error.captureFailure.request.requestIdentity, savedRequest.requestIdentity);
  assert.ok(r.elapsedMs <= 20016, name);
  failures[name] = r;
}
for (const options of [{ runtimeError: true }, { stop: true }, { finalFailure: true }, { importFailure: true }]) {
  const r = await execute({ ...options, wrapperMode: true });
  assert.equal(r.error, null); assert.equal(r.downloads.length, 1);
  const result = r.downloads[0];
  assert.ok(result.failure); assert.equal(result.semanticComparison, null);
  assert.equal(result.postFailureObservation.takenAfterFailure, true);
  if (options.importFailure) assert.equal(result.failure.message, 'original-import-error');
  else {
    assert.ok(result.captureFailure.waitObservation);
    assert.equal(result.captureFailure.originalError.message, result.failure.message);
  }
  failures['wrapper-' + Object.keys(options)[0]] = result;
}
assert.equal(failures['runtime-error'].error.message, 'original-render-failure');
assert.equal(failures['wrapper-finalFailure'].failure.message, 'original-final-presentation-error');
console.log(JSON.stringify({ status: 'pass', oldEarlyStopReproduced: true,
  savedTimingMs: { queueMs, renderMs }, passed, negativeCases: Object.keys(negatives),
  wrapperCases: 4, boundary: 'real generated module/wrapper, actual scheduler+recorder case; fake clock and GPU/capture APIs; no actual capture/SH match',
  failureExamples: { runtime: failures['wrapper-runtimeError'], stopped: failures['wrapper-stop'] } }, null, 2));
