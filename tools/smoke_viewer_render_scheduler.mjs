import assert from 'node:assert/strict';

import { createRenderScheduler } from '../demo/js/viewer_render_scheduler.js';

function createRafHarness() {
  const callbacks = [];
  return {
    callbacks,
    requestAnimationFrame(callback) {
      callbacks.push(callback);
      return callbacks.length;
    },
    async runNext() {
      assert.ok(callbacks.length > 0, 'expected a queued animation frame');
      await callbacks.shift()(0);
    }
  };
}

function createDeferred() {
  let resolve;
  const promise = new Promise((nextResolve) => {
    resolve = nextResolve;
  });
  return { promise, resolve };
}

async function testQueuedRequestDuringActiveFrame() {
  const raf = createRafHarness();
  globalThis.requestAnimationFrame = raf.requestAnimationFrame;
  const firstFrameGate = createDeferred();
  const executions = [];
  let productionGeneration = 0;
  const scheduler = createRenderScheduler({
    renderFrame: async ({ schedulerFrameState }) => {
      if (executions.length === 0) await firstFrameGate.promise;
      productionGeneration += 1;
      executions.push({
        requestIdentity: schedulerFrameState.requestIdentity,
        source: schedulerFrameState.requestSource,
        metadata: schedulerFrameState.requestMetadata,
        forceProductionUpdate:
          schedulerFrameState.forceProductionUpdate === true,
        productionGeneration
      });
      return { productionGeneration };
    },
    isPlaying: () => false
  });

  const initialRequest = await scheduler.scheduleRender({
    source: 'viewer-render-request'
  });
  const firstFrame = raf.callbacks.shift()(0);
  await Promise.resolve();
  assert.equal(scheduler.state.rendering, true);

  const sceneRequest = await scheduler.scheduleRender({
    source: 'default-scene-loaded',
    forceProductionUpdate: true,
    metadata: {
      transition: 'scene-ready-full-production-update',
      sourceKind: 'default-scene-url'
    }
  });
  await scheduler.scheduleRender({ source: 'camera-change' });
  await scheduler.scheduleRender({
    source: 'later-force-request',
    forceProductionUpdate: true,
    metadata: { transition: 'later-force-update' }
  });

  assert.equal(
    scheduler.state.queuedRequest.requestIdentity,
    sceneRequest.requestIdentity,
    'the first queued forced request must retain ownership'
  );
  assert.equal(scheduler.state.queuedRequest.forceProductionUpdate, true);
  assert.deepEqual(scheduler.state.queuedRequest.metadata, {
    transition: 'scene-ready-full-production-update',
    sourceKind: 'default-scene-url'
  });

  firstFrameGate.resolve();
  await firstFrame;
  assert.equal(raf.callbacks.length, 1, 'queued request must schedule one RAF');
  await raf.runNext();

  assert.deepEqual(
    executions.map(({ requestIdentity }) => requestIdentity),
    [initialRequest.requestIdentity, sceneRequest.requestIdentity]
  );
  assert.equal(executions[1].source, 'default-scene-loaded');
  assert.equal(executions[1].forceProductionUpdate, true);
  assert.equal(executions[1].productionGeneration, 2);
  assert.equal(scheduler.state.queuedRequest, null);
  assert.equal(scheduler.state.renderPending, false);
  assert.equal(scheduler.state.rendering, false);
  assert.equal(scheduler.state.needsRenderAgain, false);
  assert.equal(scheduler.state.scheduledFrameCount, 2);
  assert.equal(scheduler.state.completedFrameCount, 2);
  assert.equal(raf.callbacks.length, 0, 'scheduler must become quiescent');

  const captureRequest = await scheduler.scheduleRender({
    source: 'fixed-time-artifact-capture',
    forceProductionUpdate: true
  });
  assert.notEqual(captureRequest.requestIdentity, sceneRequest.requestIdentity);
  await raf.runNext();
  assert.equal(executions[2].requestIdentity, captureRequest.requestIdentity);
  assert.equal(executions[2].productionGeneration, 3);
  assert.equal(raf.callbacks.length, 0);
}

async function testQueuedRequestBeforeRafEntryAndForceEscalation() {
  const raf = createRafHarness();
  globalThis.requestAnimationFrame = raf.requestAnimationFrame;
  const executions = [];
  const scheduler = createRenderScheduler({
    renderFrame: async ({ schedulerFrameState }) => {
      executions.push({ ...schedulerFrameState });
      return {};
    },
    isPlaying: () => false
  });

  await scheduler.scheduleRender({ source: 'initial' });
  await scheduler.scheduleRender({
    source: 'weak-coalesced-request',
    metadata: { transition: 'weak' }
  });
  const forcedRequest = await scheduler.scheduleRender({
    source: 'default-scene-loaded',
    forceProductionUpdate: true,
    metadata: { transition: 'scene-ready-full-production-update' }
  });

  assert.equal(
    scheduler.state.queuedRequest.requestIdentity,
    forcedRequest.requestIdentity,
    'a forced request must replace a weaker queued request'
  );
  await raf.runNext();
  assert.equal(raf.callbacks.length, 1, 'pre-RAF queued work must survive RAF entry');
  await raf.runNext();

  assert.equal(executions.length, 2);
  assert.equal(executions[1].requestIdentity, forcedRequest.requestIdentity);
  assert.equal(executions[1].requestSource, 'default-scene-loaded');
  assert.equal(executions[1].forceProductionUpdate, true);
  assert.deepEqual(executions[1].requestMetadata, {
    transition: 'scene-ready-full-production-update'
  });
  assert.equal(scheduler.state.queuedRequest, null);
  assert.equal(scheduler.state.completedFrameCount, 2);
  assert.equal(raf.callbacks.length, 0);
}

async function testPlaybackContinuationStopsWithoutLooping() {
  const raf = createRafHarness();
  globalThis.requestAnimationFrame = raf.requestAnimationFrame;
  let playing = true;
  const executedSources = [];
  const scheduler = createRenderScheduler({
    renderFrame: async ({ schedulerFrameState }) => {
      executedSources.push(schedulerFrameState.requestSource);
      return {};
    },
    isPlaying: () => playing
  });

  await scheduler.scheduleRender({ source: 'playback-start' });
  await raf.runNext();
  assert.equal(raf.callbacks.length, 1);
  playing = false;
  await raf.runNext();

  assert.deepEqual(executedSources, [
    'playback-start',
    'viewer-loop-continuation'
  ]);
  assert.equal(scheduler.state.completedFrameCount, 2);
  assert.equal(raf.callbacks.length, 0);
}

const originalRequestAnimationFrame = globalThis.requestAnimationFrame;
try {
  await testQueuedRequestDuringActiveFrame();
  await testQueuedRequestBeforeRafEntryAndForceEscalation();
  await testPlaybackContinuationStopsWithoutLooping();
  console.log('viewer render scheduler smoke: ok');
} finally {
  globalThis.requestAnimationFrame = originalRequestAnimationFrame;
}
