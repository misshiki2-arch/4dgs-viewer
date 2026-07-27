import assert from 'node:assert/strict';

import {
  buildProductionRuntimeSelectionContract,
  validateExpectedProductionRuntimeContract
} from '../demo/js/common_4dgs_production_runtime_contract.js';
import {
  buildViewerDeterministicSummary,
  parseViewerQueryState
} from '../demo/js/viewer_query_state.js';

const productionQuery = [
  'viewerRuntime=webgpu',
  'gpuCandidateRuntime=cpu-reference',
  'webgpuBackendMode=webgpu-exclusive',
  'webgpuBackendImplementation=webgpu-tile-compositor-frame-implementation',
  'webgpuAllowViewerCanvasPresentation=true',
  'webgpuBackendViewerLoopHook=true'
].join('&');
const state = parseViewerQueryState(`?${productionQuery}`);
assert.equal(state.viewerRuntime, 'webgpu');
assert.equal(state.gpuCandidateRuntime, 'cpu-reference');
const deterministic = buildViewerDeterministicSummary(state);
const roundTrip = parseViewerQueryState(`?${deterministic.deterministicQueryString}`);
assert.equal(roundTrip.viewerRuntime, 'webgpu');
assert.equal(roundTrip.webgpuBackendMode, 'webgpu-exclusive');
assert.equal(
  roundTrip.webgpuBackendImplementation,
  'webgpu-tile-compositor-frame-implementation'
);

const contract = buildProductionRuntimeSelectionContract({ queryState: state });
assert.equal(contract.requestedRuntime, 'webgpu');
assert.equal(contract.effectiveDisplayRuntime, 'webgpu-production');
assert.equal(contract.productionSelectionReady, true);
assert.equal(contract.gpuCandidateRuntimeIsProductionDisplayRuntime, false);

const expected = {
  requestedRuntime: 'webgpu',
  effectiveDisplayRuntime: 'webgpu-production',
  backendMode: 'webgpu-exclusive',
  backendImplementation: 'webgpu-tile-compositor-frame-implementation',
  canvasPresentationEnabled: true,
  viewerLoopHookEnabled: true
};
assert.equal(validateExpectedProductionRuntimeContract(contract, expected).ready, true);

const cpuContract = buildProductionRuntimeSelectionContract({
  queryState: { ...state, viewerRuntime: 'webgl2' }
});
assert.equal(validateExpectedProductionRuntimeContract(cpuContract, expected).ready, false);
const backendMismatch = buildProductionRuntimeSelectionContract({
  queryState: {
    ...state,
    webgpuBackendImplementation: 'webgpu-normal-backend-frame-implementation'
  }
});
assert.equal(
  validateExpectedProductionRuntimeContract(backendMismatch, expected).ready,
  false
);
assert.equal(
  validateExpectedProductionRuntimeContract(
    { ...contract, runtimeEvidenceCurrent: false },
    expected
  ).ready,
  false
);

console.log('production runtime selection smoke tests passed');
