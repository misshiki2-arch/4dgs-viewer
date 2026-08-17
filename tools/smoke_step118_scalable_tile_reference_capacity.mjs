import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
  buildProductionTileReferencePlanFromTileCounts,
  resolveProductionTileReferenceAllocation
} from '../demo/js/common_4dgs_production_tile_reference_contracts.js';
import {
  selectActiveProductionResidentWorkset
} from '../demo/js/webgpu_production_workset_owner.js';

const canonical = buildProductionTileReferencePlanFromTileCounts({
  // Four Gaussian records generate nine references across three tiles.
  tileReferenceCounts: [4, 3, 2],
  allocatedReferenceCapacity: 16,
  recordCount: 4
});
assert.equal(canonical.contract.tileReferenceCapacityReady, true);
assert.equal(canonical.contract.recordCount, 4);
assert.equal(canonical.contract.requiredReferenceCount, 9);
assert.equal(canonical.contract.requiredPaddedReferenceCapacity, 10);
assert.equal(canonical.contract.writtenReferenceCount, 9);
assert.equal(canonical.contract.recordAndReferenceCapacitySeparated, true);
assert.equal(canonical.contract.silentDropAllowed, false);
assert.deepEqual(canonical.tiles, [
  { offset: 0, count: 4, paddedCount: 4 },
  { offset: 4, count: 3, paddedCount: 4 },
  { offset: 8, count: 2, paddedCount: 2 }
]);

const overflow = buildProductionTileReferencePlanFromTileCounts({
  tileReferenceCounts: [4, 3, 2],
  allocatedReferenceCapacity: 8,
  recordCount: 4
});
assert.equal(overflow.contract.tileReferenceCapacityReady, false);
assert.equal(overflow.contract.capacityOverflowDetected, true);
assert.equal(overflow.contract.capacityOverflowFailClosed, true);
assert.equal(overflow.contract.writtenReferenceCount, 0);
assert.equal(overflow.contract.silentDropAllowed, false);

const device = {
  limits: {
    maxStorageBufferBindingSize: 1024,
    maxBufferSize: 2048
  }
};
const allocation = resolveProductionTileReferenceAllocation({
  device,
  recordCount: 10,
  tileCount: 20
});
assert.equal(allocation.allocatedReferenceCapacity, 64);
assert.equal(allocation.allocatedReferenceBytes, 1024);

const raw = {
  count: 100,
  xyzDim: 3,
  opacityDim: 1,
  xyz: new Float32Array(300),
  opacity: new Float32Array(100)
};
const workset = selectActiveProductionResidentWorkset({
  raw,
  device: {
    limits: {
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxBufferSize: 128 * 1024 * 1024
    }
  },
  canvasWidth: 16,
  canvasHeight: 16
});
assert.equal(workset.contract.residentRecordCount, 100);
assert.equal(
  workset.contract.tileReferenceCapacityCoupledToRecordSelection,
  false
);

const [layoutSource, compositorSource, frameSource] = await Promise.all([
  readFile(
    new URL('../demo/js/webgpu_gpu_owned_tile_list_layout.js', import.meta.url),
    'utf8'
  ),
  readFile(
    new URL('../demo/js/webgpu_tile_list_compositor.js', import.meta.url),
    'utf8'
  ),
  readFile(
    new URL('../demo/js/webgpu_production_frame_data_path.js', import.meta.url),
    'utf8'
  )
]);
assert.equal(layoutSource.includes('splitProductionGpuWork'), true);
assert.equal(layoutSource.includes("submitRecordBatch('scatterReferences'"), true);
assert.equal(layoutSource.includes('createProductionTileExecutionPlanResources'), true);
assert.equal(layoutSource.includes('buildProductionTileReferencePlanFromTileCounts'), false);
assert.equal(layoutSource.includes('mapAsync'), false);
assert.equal(layoutSource.includes('recordAndReferenceCapacitySeparated: true'), true);
assert.equal(layoutSource.includes('slot < maxRefs'), false);
assert.equal(layoutSource.includes('tile * maxRefs'), false);
assert.equal(compositorSource.includes('min(u32(params.maxRefsPerTile), 64u)'), false);
assert.equal(
  compositorSource.includes('complete-reference-sort-or-frame-fail-closed'),
  true
);
assert.equal(
  compositorSource.includes('executeBoundedProductionTileSortAndCompositor'),
  true
);
assert.equal(frameSource.includes('maxRefsPerTile: 64'), false);
assert.equal(frameSource.includes("readbackPolicy: 'none'"), true);

console.log('Step118 scalable production tile-reference capacity smoke: OK');
