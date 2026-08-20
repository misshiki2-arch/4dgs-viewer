import assert from 'node:assert/strict';
import fs from 'node:fs';

import {
  buildWebGpuProductionTexturePresentationUvWgsl,
  buildWebGpuPresentationCaptureOrientationEvidence,
  compareWebGpuPresentationFrameIdentity,
  getWebGpuPresentationCaptureOrientationContract,
  mapWebGpuPresentationClipYToProductionTextureV,
  mapWebGpuProductionTextureRowToPngRow,
  mapWebGpuProductionTextureRowToPresentedRow,
  summarizeWebGpuPresentationFrameIdentity
} from '../demo/js/webgpu_presentation_capture_orientation_contract.js';

function applyRowMapping(rows, mapRow) {
  const output = new Array(rows.length);
  rows.forEach((row, sourceRow) => {
    output[mapRow(sourceRow, rows.length)] = row;
  });
  return output;
}

const asymmetricRows = [
  [0xf1, 0x11, 0x01, 0xff],
  [0x22, 0xe2, 0x02, 0x80],
  [0x03, 0x33, 0xd3, 0x40]
];
const presentedRows = applyRowMapping(
  asymmetricRows,
  mapWebGpuProductionTextureRowToPresentedRow
);
const pngRows = applyRowMapping(
  asymmetricRows,
  mapWebGpuProductionTextureRowToPngRow
);

assert.deepEqual(presentedRows, asymmetricRows);
assert.deepEqual(pngRows, asymmetricRows);
assert.deepEqual(presentedRows, pngRows);
assert.equal(mapWebGpuProductionTextureRowToPresentedRow(0, 3), 0);
assert.equal(mapWebGpuProductionTextureRowToPresentedRow(2, 3), 2);
assert.equal(mapWebGpuProductionTextureRowToPngRow(0, 3), 0);
assert.equal(mapWebGpuProductionTextureRowToPngRow(2, 3), 2);

assert.equal(mapWebGpuPresentationClipYToProductionTextureV(1), 0);
assert.equal(mapWebGpuPresentationClipYToProductionTextureV(-1), 1);
assert.equal(
  buildWebGpuProductionTexturePresentationUvWgsl('pos'),
  'vec2f(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5)'
);

const contract = getWebGpuPresentationCaptureOrientationContract();
assert.equal(contract.productionTextureOrigin, 'texture-memory-top-left');
assert.equal(contract.productionTextureYAxisDirection, 'down');
assert.equal(contract.presentationVerticalFlipApplied, false);
assert.equal(contract.captureVerticalFlipApplied, false);
assert.equal(
  contract.canonicalPresentationOrientation,
  'production-texture-top-left-y-down'
);
assert.equal(
  contract.savedPngOrientation,
  contract.canonicalPresentationOrientation
);

const evidence = buildWebGpuPresentationCaptureOrientationEvidence();
assert.equal(evidence.savedPngMatchesRawProductionOutput, true);
assert.equal(evidence.savedPngMatchesPresentedOutput, true);
assert.equal(evidence.captureMatchesCanonicalPresentationOrientation, true);
assert.equal(evidence.orientationConsistencyKnown, true);
assert.equal(evidence.orientationMismatchDetected, false);

const frameIdentity = summarizeWebGpuPresentationFrameIdentity({
  generation: 9,
  frameHash: 'asymmetric-row-fixture',
  datasetCameraLabel: 'camera-fixture',
  datasetFrameNumber: 151,
  datasetTime: 23.2,
  referenceCameraLabel: 'camera-fixture',
  outputWidth: 1,
  outputHeight: asymmetricRows.length
});
assert.deepEqual(
  compareWebGpuPresentationFrameIdentity(frameIdentity, { ...frameIdentity }),
  {
    matches: true,
    mismatchedKeys: [],
    missingKeys: [],
    requiredKeys: [
      'generation',
      'datasetCameraLabel',
      'datasetFrameNumber',
      'datasetTime',
      'referenceCameraLabel',
      'outputWidth',
      'outputHeight'
    ]
  }
);

assert.throws(
  () => mapWebGpuProductionTextureRowToPresentedRow(-1, 3),
  /orientation-source-row-out-of-bounds/
);
assert.throws(
  () => mapWebGpuProductionTextureRowToPngRow(3, 3),
  /orientation-source-row-out-of-bounds/
);

const compositorSource = fs.readFileSync(
  new URL('../demo/js/webgpu_tile_list_compositor.js', import.meta.url),
  'utf8'
);
assert.match(
  compositorSource,
  /out\.uv = \$\{presentationUvExpression\};/,
  'live and cached currentTexture presentation must consume the common UV policy'
);
assert.match(
  compositorSource,
  /mapWebGpuProductionTextureRowToPngRow\(y, height\)/,
  'PNG row copy must consume the common row policy'
);
assert.doesNotMatch(compositorSource, /\{\s*flipY:/);

console.log('Step119 Investigation6 Fix2 orientation mapping smoke passed.');
