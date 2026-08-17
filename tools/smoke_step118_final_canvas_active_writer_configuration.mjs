import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
  createFinalCanvasPresentationTraceRecorder,
  FINAL_CANVAS_PRESENTATION_PATHS
} from '../demo/js/common_4dgs_final_canvas_presentation.js';

const viewerSource = await readFile(
  new URL('../demo/js/viewer_app_gpu.js', import.meta.url),
  'utf8'
);
const activePathConfiguration = viewerSource.match(
  /activePathIdentities:\s*deterministicQueryState\.webgpuBackendImplementation\s*===\s*WEBGPU_TILE_COMPOSITOR_FRAME_IMPLEMENTATION\s*&&\s*deterministicQueryState\.webgpuBackendViewerLoopHook\s*===\s*true\s*&&\s*deterministicQueryState\.webgpuAllowViewerCanvasPresentation\s*===\s*true\s*\?\s*\[(?<production>[^\]]*)\]\s*:\s*\[(?<other>[^\]]*)\]/s
);
assert.ok(
  activePathConfiguration?.groups,
  'viewer final-canvas active writer configuration must remain structurally inspectable'
);

function configuredPathNames(source) {
  return [...source.matchAll(/FINAL_CANVAS_PRESENTATION_PATHS\.([A-Z_]+)/g)]
    .map((match) => match[1]);
}

const productionPathNames = configuredPathNames(
  activePathConfiguration.groups.production
);
const otherBackendPathNames = configuredPathNames(
  activePathConfiguration.groups.other
);
assert.deepEqual(productionPathNames, ['TILE_COMPOSITOR']);
assert.deepEqual(otherBackendPathNames, ['GUARDED_PRESENTATION']);

const productionRecorder = createFinalCanvasPresentationTraceRecorder({
  activePathIdentities: productionPathNames.map(
    (name) => FINAL_CANVAS_PRESENTATION_PATHS[name]
  ),
  requiredSteadyStateEventCount: 1
});
productionRecorder.registerPath({
  pathIdentity: FINAL_CANVAS_PRESENTATION_PATHS.TILE_COMPOSITOR,
  source: 'step118-active-writer-configuration-smoke',
  supportedEventKinds: [
    'production-presentation',
    'cached-production-presentation',
    'black-fallback',
    'presentation-failure'
  ]
});
productionRecorder.recordEvent({
  presentationPathIdentity: FINAL_CANVAS_PRESENTATION_PATHS.TILE_COMPOSITOR,
  eventKind: 'production-presentation',
  presentationSource: 'webgpu-tile-compositor-output-texture',
  sourceRequestIdentity: 'viewer-render-request:3',
  productionGeneration: 3,
  compositorGeneration: 3,
  presentedGeneration: 3,
  sourcePixelResult: 'nonblank',
  canvasWriteAttempted: true,
  canvasWriteSubmitted: true,
  canvasWriteCompleted: true,
  staleSource: false,
  presentationFailed: false
});
const productionBoundary = productionRecorder.getSnapshot({
  boundaryKind: 'step118-active-writer-configuration-smoke',
  expectedRequestIdentity: 'viewer-render-request:3',
  expectedGeneration: 3,
  requiredSteadyStateEventCount: 1,
  requireCanvasWritePathCoverage: true
});
assert.equal(productionBoundary.canvasWritePathCoverage.coverageComplete, true);
assert.deepEqual(
  productionBoundary.canvasWritePathCoverage.unregisteredPathIdentities,
  []
);
assert.equal(productionBoundary.browserVisibleResult, true);
assert.equal(
  productionBoundary.canvasWritePathCoverage.eventVocabularyCoverageComplete,
  false
);

productionRecorder.recordEvent({
  presentationPathIdentity:
    FINAL_CANVAS_PRESENTATION_PATHS.BOUNDED_FIRST_PRESENT,
  eventKind: 'diagnostic-presentation',
  presentationSource: 'unexpected-observed-writer-smoke',
  sourcePixelResult: 'unknown',
  canvasWriteAttempted: true,
  canvasWriteSubmitted: true,
  canvasWriteCompleted: true,
  staleSource: false,
  presentationFailed: false
});
const observedPathCoverage = productionRecorder
  .getRuntimeStateSnapshot()
  .canvasWritePathCoverage;
assert.equal(observedPathCoverage.coverageComplete, false);
assert.deepEqual(observedPathCoverage.unregisteredPathIdentities, [
  FINAL_CANVAS_PRESENTATION_PATHS.BOUNDED_FIRST_PRESENT
]);

console.log('Step118 final-canvas active writer configuration smoke test passed');
