import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
  WEBGPU_VISIBLE_RECORD_CPU_MATERIALIZED_FIELDS,
  WEBGPU_VISIBLE_RECORD_FIELD_COMPUTE_MODES,
  WEBGPU_VISIBLE_RECORD_FIELDS,
  WEBGPU_VISIBLE_RECORD_FLOATS,
  WEBGPU_VISIBLE_RECORD_REFERENCE_ASSISTED_FIELDS
} from '../demo/js/common_4dgs_record_contracts.js';
import {
  FIXED_RECORD_DEFAULT_FIELD_TOLERANCES
} from '../demo/js/common_4dgs_comparison_contracts.js';

const runtimeSource = await readFile(
  new URL('../demo/js/webgpu_visible_record_dry_run_runtime.js', import.meta.url),
  'utf8'
);
const evaluatorSource = await readFile(
  new URL('../demo/js/webgpu_4d_state_evaluator.js', import.meta.url),
  'utf8'
);

function extractShaderSource(source, label) {
  const labelOffset = source.indexOf(`label: '${label}'`);
  assert.notEqual(labelOffset, -1, `${label} shader label must exist`);
  const marker = 'code: `';
  const codeOffset = source.indexOf(marker, labelOffset);
  assert.notEqual(codeOffset, -1, `${label} shader source must exist`);
  const codeStart = codeOffset + marker.length;
  const codeEnd = source.indexOf('`', codeStart);
  assert.notEqual(codeEnd, -1, `${label} shader source must terminate`);
  return source.slice(codeStart, codeEnd);
}

function bindingVocabulary(shaderSource) {
  return [...shaderSource.matchAll(/@group\(0\) @binding\((\d+)\)/g)].map(
    (match) => Number(match[1])
  );
}

const fixedRecordShader = extractShaderSource(
  runtimeSource,
  'phase3-step4-visible-record-projection-wgsl'
);
const evaluatorShader = extractShaderSource(
  evaluatorSource,
  'phase3-step81-partial-4d-state-and-attribute-evaluator-wgsl'
);

const outputValidMatch = fixedRecordShader.match(
  /let outputValid =\s*([^;]+);/
);
assert.ok(outputValidMatch, 'fixed-record WGSL must define outputValid');
const outputValidFactors = outputValidMatch[1]
  .split('&&')
  .map((factor) => factor.trim());
assert.deepEqual(
  outputValidFactors,
  [
    'statePositionAvailable',
    'rawIndexInBounds',
    'projectionOk',
    'projectedInBounds'
  ],
  'fixed-record valid must require state availability and all existing projection gates'
);

assert.match(
  fixedRecordShader,
  /var sourcePos = raw0\.xyz;\s*if \(statePositionAvailable\) \{\s*sourcePos = statePos\.xyz;\s*\}/,
  'state XYZ must remain authoritative when statePositionAvailable is true'
);
assert.match(
  fixedRecordShader,
  /r0\.y = select\(0\.0, 1\.0, outputValid\);\s*r0\.z = select\(0\.0, projectedPx, outputValid\);\s*r0\.w = select\(0\.0, projectedPy, outputValid\);\s*r1\.x = select\(0\.0, projectedDepth, outputValid\);/,
  'valid, px, py, and depth must all zero through the same outputValid gate'
);

function runFixedRecordCase({
  statePositionAvailable,
  rawPosition,
  statePosition,
  rawIndexInBounds = true,
  viewportWidth = 100,
  viewportHeight = 100
}) {
  const sourcePosition = statePositionAvailable ? statePosition : rawPosition;
  const projectionOk = sourcePosition[2] > 0;
  const projectedPx = sourcePosition[0];
  const projectedPy = sourcePosition[1];
  const projectedDepth = sourcePosition[2];
  const projectedInBounds =
    projectedPx >= 0 &&
    projectedPy >= 0 &&
    projectedPx < viewportWidth &&
    projectedPy < viewportHeight;
  const gates = {
    statePositionAvailable,
    rawIndexInBounds,
    projectionOk,
    projectedInBounds
  };
  const outputValid = outputValidFactors.every((factor) => gates[factor] === true);
  return {
    sourcePosition,
    valid: outputValid ? 1 : 0,
    px: outputValid ? projectedPx : 0,
    py: outputValid ? projectedPy : 0,
    depth: outputValid ? projectedDepth : 0
  };
}

// Case A: raw XYZ projects, but unavailable state must not be promoted.
const unavailableRawProjectable = runFixedRecordCase({
  statePositionAvailable: false,
  rawPosition: [32, 24, 2],
  statePosition: [0, 0, 0]
});
assert.deepEqual(unavailableRawProjectable, {
  sourcePosition: [32, 24, 2],
  valid: 0,
  px: 0,
  py: 0,
  depth: 0
});

// Case B: available state XYZ, rather than raw XYZ, owns valid projection.
const availableStateProjectable = runFixedRecordCase({
  statePositionAvailable: true,
  rawPosition: [999, 999, -1],
  statePosition: [40, 30, 4]
});
assert.deepEqual(availableStateProjectable, {
  sourcePosition: [40, 30, 4],
  valid: 1,
  px: 40,
  py: 30,
  depth: 4
});

// Case C: unavailable state and unprojectable raw XYZ remain invalid.
const unavailableRawUnprojectable = runFixedRecordCase({
  statePositionAvailable: false,
  rawPosition: [32, 24, -2],
  statePosition: [0, 0, 0]
});
assert.deepEqual(unavailableRawUnprojectable, {
  sourcePosition: [32, 24, -2],
  valid: 0,
  px: 0,
  py: 0,
  depth: 0
});

// Case D: available state still requires the existing viewport gate.
const availableStateOutOfBounds = runFixedRecordCase({
  statePositionAvailable: true,
  rawPosition: [20, 20, 2],
  statePosition: [120, 30, 4]
});
assert.deepEqual(availableStateOutOfBounds, {
  sourcePosition: [120, 30, 4],
  valid: 0,
  px: 0,
  py: 0,
  depth: 0
});

assert.equal(WEBGPU_VISIBLE_RECORD_FLOATS, 12);
assert.deepEqual(WEBGPU_VISIBLE_RECORD_FIELDS, [
  ['srcIndex', 0, 1],
  ['valid', 1, 1],
  ['px', 2, 1],
  ['py', 3, 1],
  ['depth', 4, 1],
  ['aabb', 5, 4],
  ['reserved', 9, 3]
]);
assert.deepEqual(Object.keys(FIXED_RECORD_DEFAULT_FIELD_TOLERANCES), [
  'srcIndex',
  'valid',
  'px',
  'py',
  'depth',
  'aabb'
]);
assert.deepEqual(WEBGPU_VISIBLE_RECORD_REFERENCE_ASSISTED_FIELDS, []);
assert.deepEqual(WEBGPU_VISIBLE_RECORD_CPU_MATERIALIZED_FIELDS, ['aabb']);
assert.equal(
  WEBGPU_VISIBLE_RECORD_FIELD_COMPUTE_MODES.valid,
  'wgsl-candidate-bounds-state-position-projection-gate'
);
assert.deepEqual(bindingVocabulary(fixedRecordShader), [0, 1, 2, 3, 4, 5, 6, 7]);
assert.deepEqual(bindingVocabulary(evaluatorShader), [0, 1, 2, 3, 4, 5, 6, 7, 8]);

assert.doesNotMatch(fixedRecordShader, /\br1\.[yzw]\s*=/);
assert.doesNotMatch(fixedRecordShader, /\br2\.x\s*=/);
assert.match(fixedRecordShader, /r2\.y = raw0\.x;/);
assert.match(fixedRecordShader, /r2\.w = raw0\.z;/);
assert.match(
  fixedRecordShader,
  /directEvidenceRecords\[directBase \+ 1u\] = vec4f\(raw0\.xyz, raw0\.w\);/
);
assert.match(evaluatorShader, /statePositions\[row\] = vec4f\(0\.0\);/);
assert.match(evaluatorShader, /let footprintBase = row \* 3u;/);
assert.match(evaluatorSource, /const footprintProductionVec4Count = count \* 3;/);
assert.match(evaluatorSource, /const STEP113_DIAGNOSTIC_VEC4_STRIDE = 8;/);
assert.match(evaluatorSource, /const STEP113_DIAGNOSTIC_VEC4_STRIDE: u32 = 8u;/);
assert.match(runtimeSource, /function buildCpuReferenceRecords\(/);
assert.match(
  runtimeSource,
  /out\[base \+ 1\] = item \? 1 : 0;[\s\S]*?out\[base \+ 4\] = item\?\.depth \?\? 0;/
);

console.log('Step120 Impl2 Fix4 fixed-record state validity smoke: OK', {
  outputValidFactors,
  fixedRecordBindings: bindingVocabulary(fixedRecordShader),
  evaluatorBindings: bindingVocabulary(evaluatorShader)
});
