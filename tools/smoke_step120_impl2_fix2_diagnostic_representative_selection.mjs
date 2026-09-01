import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
  normalizeWebGpuDiagnosticDetailSelection,
  resolveWebGpuDiagnosticDetailRows
} from '../demo/js/common_4dgs_diagnostic_artifact_contracts.js';
import {
  buildWebGpu4DStatePositionsForCandidates
} from '../demo/js/webgpu_4d_state_evaluator.js';

const SENTINEL = 0xffffffff;
const CANONICAL_SRC_INDICES = [
  658947,
  771007,
  788034,
  826401,
  835183,
  852955,
  863505,
  906711
];
const canonicalEligibilityFixture = [
  { srcIndex: 658947, row: 0, temporalWeight: 0.6619040317663506, productionPayloadSourceCode: 113 },
  { srcIndex: 771007, row: 1, temporalWeight: 0.09684425382284878, productionPayloadSourceCode: 113 },
  { srcIndex: 788034, row: 2, temporalWeight: 0.7423464737334201, productionPayloadSourceCode: 113 },
  { srcIndex: 826401, row: 3, temporalWeight: 0.4944480275451095, productionPayloadSourceCode: 113 },
  { srcIndex: 835183, row: 4, temporalWeight: 0.5222365727962414, productionPayloadSourceCode: 113 },
  { srcIndex: 852955, row: 5, temporalWeight: 0.0671115505869993, productionPayloadSourceCode: 113 },
  { srcIndex: 863505, row: 6, temporalWeight: 0.7833088094678182, productionPayloadSourceCode: 113 },
  { srcIndex: 906711, row: 7, temporalWeight: 0.32300241398750507, productionPayloadSourceCode: 113 }
];

const runtimeSource = await readFile(
  new URL('../demo/js/webgpu_visible_record_dry_run_runtime.js', import.meta.url),
  'utf8'
);
const evaluatorSource = await readFile(
  new URL('../demo/js/webgpu_4d_state_evaluator.js', import.meta.url),
  'utf8'
);

const resolverSourceMatch = runtimeSource.match(
  /function resolveStep113DiagnosticRowsBeforeDispatch\([\s\S]*?\n}\n\nexport async function runWebGpuVisibleRecordDryRun/
);
assert.ok(resolverSourceMatch, 'pre-dispatch Step113 resolver must exist');
const resolveStep113DiagnosticRowsBeforeDispatch = Function(
  'resolveWebGpuDiagnosticDetailRows',
  `${resolverSourceMatch[0].replace(
    /\n\nexport async function runWebGpuVisibleRecordDryRun$/,
    ''
  )}\nreturn resolveStep113DiagnosticRowsBeforeDispatch;`
)(resolveWebGpuDiagnosticDetailRows);

function normalizedSelection(mode, srcIndices = [], limit = 8) {
  return normalizeWebGpuDiagnosticDetailSelection({
    mode,
    srcIndices,
    limit
  });
}

const candidateIndices = Uint32Array.from([
  ...CANONICAL_SRC_INDICES,
  100,
  101,
  102,
  103
]);
const explicitSelection = normalizedSelection(
  'explicit-src-indices',
  CANONICAL_SRC_INDICES
);
const explicitRows = resolveStep113DiagnosticRowsBeforeDispatch({
  candidateIndices,
  detailedLineageSelection: explicitSelection
});
assert.deepEqual(explicitRows, [0, 1, 2, 3, 4, 5, 6, 7]);
assert.deepEqual(
  explicitRows.map((row) => candidateIndices[row]),
  CANONICAL_SRC_INDICES,
  'explicit srcIndex order must be preserved when resolving candidate rows'
);

const explicitAndFirstRows = resolveStep113DiagnosticRowsBeforeDispatch({
  candidateIndices,
  detailedLineageSelection: normalizedSelection(
    'explicit-and-first-mismatch',
    CANONICAL_SRC_INDICES
  )
});
assert.deepEqual(
  explicitAndFirstRows,
  explicitRows,
  'pre-dispatch explicit-and-first-mismatch may contain only its explicit rows'
);
assert.equal(
  resolveStep113DiagnosticRowsBeforeDispatch({
    candidateIndices,
    detailedLineageSelection: normalizedSelection('first-mismatch')
  }),
  null,
  'first-mismatch-only must not fabricate a pre-dispatch row override'
);
assert.equal(
  resolveStep113DiagnosticRowsBeforeDispatch({
    candidateIndices,
    detailedLineageSelection: normalizedSelection('none')
  }),
  null,
  'selection-free execution must retain the evaluator fallback'
);

const moreThanTailCapacity = Array.from(
  { length: 12 },
  (_, index) => 2000 + index
);
assert.deepEqual(
  resolveStep113DiagnosticRowsBeforeDispatch({
    candidateIndices: Uint32Array.from(moreThanTailCapacity),
    detailedLineageSelection: normalizedSelection(
      'explicit-src-indices',
      moreThanTailCapacity,
      32
    )
  }),
  [0, 1, 2, 3, 4, 5, 6, 7],
  'pre-dispatch explicit rows must be bounded to the eight-slot tail'
);

for (const fixture of canonicalEligibilityFixture) {
  assert.equal(fixture.row, canonicalEligibilityFixture.indexOf(fixture));
  assert.ok(fixture.temporalWeight > 0.05);
  assert.equal(fixture.productionPayloadSourceCode, 113);
}

function parseStep113RowsFromShader(shaderSource) {
  const match = shaderSource.match(
    /const STEP113_DIAGNOSTIC_ROWS: array<u32, 8> = array<u32, 8>\(\s*([^)]*?)\s*\);/
  );
  assert.ok(match, 'generated WGSL must contain the authoritative row array');
  return match[1]
    .split(',')
    .map((value) => Number(value.trim().replace(/u$/, '')));
}

const previousGpuBufferUsage = globalThis.GPUBufferUsage;
const previousGpuMapMode = globalThis.GPUMapMode;
globalThis.GPUBufferUsage = {
  STORAGE: 1,
  UNIFORM: 2,
  COPY_SRC: 4,
  COPY_DST: 8,
  MAP_READ: 16
};
globalThis.GPUMapMode = { READ: 1 };

let lastShaderSource = null;
const createdBindGroups = [];
const createFakeBuffer = (descriptor) => {
  const mappedRange = new ArrayBuffer(descriptor.size);
  return {
    descriptor,
    getMappedRange: () => mappedRange,
    mapAsync: async () => {},
    unmap: () => {},
    destroy: () => {}
  };
};
const fakePass = {
  setPipeline: () => {},
  setBindGroup: () => {},
  dispatchWorkgroups: () => {},
  end: () => {}
};
const fakeDevice = {
  limits: {
    maxStorageBufferBindingSize: 1_073_741_824,
    maxBufferSize: 1_073_741_824,
    maxStorageBuffersPerShaderStage: 8
  },
  createBuffer: createFakeBuffer,
  createShaderModule: ({ code }) => {
    lastShaderSource = code;
    return { code };
  },
  createComputePipeline: () => ({
    getBindGroupLayout: () => ({ label: 'step120-fix2-fake-layout' })
  }),
  createBindGroup: (descriptor) => {
    createdBindGroups.push(descriptor);
    return { descriptor };
  },
  createCommandEncoder: () => ({
    beginComputePass: () => fakePass,
    copyBufferToBuffer: () => {},
    finish: () => ({ label: 'step120-fix2-fake-command-buffer' })
  }),
  queue: {
    submit: () => {},
    onSubmittedWorkDone: async () => {}
  }
};

async function runEvaluator({ count, explicitRows: rowOverride }) {
  const rawXyzOpacity = new Float32Array(count * 4);
  for (let row = 0; row < count; row += 1) rawXyzOpacity[row * 4 + 2] = 1;
  const result = await buildWebGpu4DStatePositionsForCandidates({
    device: fakeDevice,
    raw: {
      N: count,
      activeShDegree: 2,
      activeShDegreeT: 2,
      t: new Float32Array(count),
      tDim: 1,
      scale_t: new Float32Array(count).fill(1),
      scaleTDim: 1,
      f_dc: new Float32Array(count * 3),
      fdcDim: 3,
      f_rest: new Float32Array(count * 45),
      frestDim: 45,
      scale_xyz: new Float32Array(count * 3).fill(1),
      scaleXYZDim: 3,
      rotation: Float32Array.from(
        { length: count * 4 },
        (_, index) => index % 4 === 0 ? 1 : 0
      ),
      rotationDim: 4,
      rotation_r: Float32Array.from(
        { length: count * 4 },
        (_, index) => index % 4 === 0 ? 1 : 0
      ),
      rotationRDim: 4
    },
    candidateIndices: Uint32Array.from({ length: count }, (_, row) => row),
    rawXyzOpacity,
    buildConfig: { timestamp: 0, scalingModifier: 1, sigmaScale: 1 },
    projectionParams: new Float32Array(24),
    step113DiagnosticRowIndices: rowOverride
  });
  const jsRows = result.step113IntermediateReadbackRows;
  const wgslRows = parseStep113RowsFromShader(lastShaderSource);
  assert.deepEqual(wgslRows, jsRows, 'JS and WGSL must consume one authoritative row list');
  return { result, jsRows, wgslRows };
}

try {
  const canonical = await runEvaluator({ count: 16, explicitRows });
  assert.deepEqual(canonical.jsRows, [0, 1, 2, 3, 4, 5, 6, 7]);

  const shortExplicit = await runEvaluator({
    count: 16,
    explicitRows: [5, 2, 7]
  });
  assert.deepEqual(
    shortExplicit.jsRows,
    [5, 2, 7, SENTINEL, SENTINEL, SENTINEL, SENTINEL, SENTINEL],
    'short explicit selection must preserve order without fraction padding'
  );

  const bounded = await runEvaluator({
    count: 16,
    explicitRows: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
  });
  assert.deepEqual(bounded.jsRows, [9, 8, 7, 6, 5, 4, 3, 2]);

  const fallback = await runEvaluator({ count: 65536, explicitRows: null });
  assert.deepEqual(fallback.jsRows, [0, 11141, 24248, 41287, 54394, 65535, SENTINEL, SENTINEL]);

  const lastBindGroup = createdBindGroups.at(-1);
  assert.deepEqual(
    lastBindGroup.entries.map((entry) => entry.binding),
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
  );
  assert.equal(fallback.result.step113DiagnosticBindingEvidence.productionComputeStorageBufferBindingCount, 8);
  const layout = fallback.result.step113DiagnosticBindingEvidence.diagnosticPackedTailLayout;
  assert.equal(layout.diagnosticRowCount, 8);
  assert.equal(layout.diagnosticRowStrideVec4, 8);
  assert.equal(layout.productionRecordStrideVec4, 3);
  assert.equal(layout.diagnosticTailVec4Offset, 65536 * 3);
  assert.equal(layout.regionsOverlap, false);
  assert.equal(layout.productionRecordCountPreserved, true);
} finally {
  if (previousGpuBufferUsage === undefined) {
    delete globalThis.GPUBufferUsage;
  } else {
    globalThis.GPUBufferUsage = previousGpuBufferUsage;
  }
  if (previousGpuMapMode === undefined) {
    delete globalThis.GPUMapMode;
  } else {
    globalThis.GPUMapMode = previousGpuMapMode;
  }
}

const bindings = [...evaluatorSource.matchAll(/@group\(0\) @binding\((\d+)\)/g)]
  .map((match) => Number(match[1]));
assert.deepEqual([...new Set(bindings)], [0, 1, 2, 3, 4, 5, 6, 7, 8]);
assert.equal(
  [...evaluatorSource.matchAll(/var<storage/g)].length,
  8,
  'the evaluator storage binding count must remain eight'
);
assert.match(evaluatorSource, /const STEP113_DIAGNOSTIC_ROW_COUNT: u32 = 8u;/);
assert.match(evaluatorSource, /const STEP113_DIAGNOSTIC_VEC4_STRIDE: u32 = 8u;/);
assert.match(evaluatorSource, /let outBase = params\.count \* 3u \+ slot \* STEP113_DIAGNOSTIC_VEC4_STRIDE;/);
assert.ok(
  evaluatorSource.indexOf('footprintPayload[footprintBase + 2u]') <
    evaluatorSource.indexOf('for (var slot: u32 = 0u; slot < STEP113_DIAGNOSTIC_ROW_COUNT'),
  'diagnostic selection must remain downstream of production footprint output'
);
assert.match(
  runtimeSource,
  /const step113DiagnosticRowIndices =[\s\S]*?resolveStep113DiagnosticRowsBeforeDispatch\([\s\S]*?const webgpu4DStateSource = await buildWebGpu4DStatePositionsForCandidates\([\s\S]*?step113DiagnosticRowIndices/
);

console.log('Step120 Impl2 Fix2 diagnostic representative selection smoke: OK', {
  canonicalSrcIndices: CANONICAL_SRC_INDICES,
  canonicalRows: explicitRows,
  tailCapacity: 8,
  fallbackRows: [0, 11141, 24248, 41287, 54394, 65535]
});
