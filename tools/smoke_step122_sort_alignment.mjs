import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { fake } from './smoke_step122_compositing_diagnostic.mjs';
import { sortFixtures, expectedSortedFixture, compareSortedDiagnostic } from './step122_sort_fixture.mjs';
import { runCompositingDiagnostic, validateCompositingDiagnosticInput } from '../demo/js/webgpu_compositing_diagnostic.js';
import { buildWebGpuTileDepthOrderingContract, buildWebGpuTileListCompositorContract,
  PRODUCTION_DEPTH_ORDER, PRODUCTION_SORT_ALGORITHM } from '../demo/js/common_4dgs_record_contracts.js';
import { PRODUCTION_TILE_EXECUTION_PLAN_WORD as WORD } from '../demo/js/common_4dgs_production_tile_execution_plan_contracts.js';

// CPU model of the scheduled network; not WGSL execution. It consumes actual
// API-submitted batch/K/J parameters and unsorted source buffers. Expected
// sequences below are independently specified, not produced by this model.
function scheduledModel(dispatches) {
  const states=new Map();
  const before=(a,b)=>a[0]<0?false:b[0]<0?true:a[3]<b[3]||(a[3]===b[3]&&a[0]<b[0]);
  for(const d of dispatches.filter(d=>['seedOrderedReferences','compareSwapBitonicStage','validateOrderedReferences'].includes(d.entry))) {
    const group=d.group.entries,table=new Float32Array(group[0].resource.buffer.bytes);
    const source=new Float32Array(group[1].resource.buffer.bytes),out=group[2].resource.buffer;
    const [,capacity,, ,start,end,k,j]=new Uint32Array(d.params);
    if(!states.has(out))states.set(out,Array.from({length:capacity},()=>[0,0,0,0]));
    const values=states.get(out);
    for(let t=0;t<table.length;t+=4) {
      const [offset,count,padded]=table.slice(t,t+3);
      for(let local=0;local<padded;local++) {
        const ix=offset+local;if(ix<start||ix>=end)continue;
        if(d.entry==='seedOrderedReferences')values[ix]=local<count?Array.from(source.slice(ix*4,ix*4+4)):[-1,0,0,0];
        if(d.entry==='compareSwapBitonicStage'&&k<=padded) {
          const peer=local^j;if(peer<=local||peer>=padded)continue;
          const swap=(local&k)===0?before(values[offset+peer],values[ix]):before(values[ix],values[offset+peer]);
          if(swap)[values[ix],values[offset+peer]]=[values[offset+peer],values[ix]];
        }
      }
    }
  }
  return states;
}

// Strict saved-result consumer must reject wrong order or missing references,
// even if C/T happened to match. Synthetic readback tests are not GPU evidence.
function synthetic(fixture) {
  const e=expectedSortedFixture(fixture),s=fixture.input;
  return {schema:'small-compositing-diagnostic-result-v2',status:'observed-requires-independent-comparison',
    productionDeviceUsed:false,productionDeviceAcceptedAsInput:false,cleanupCompleted:true,
    accepted:false,normalOnAccepted:false,comparisonComplete:false,frames:e.frames.map(w=>{
      const plan=Array(16).fill(0),summary=Array(44).fill(0),ordering=Array(28).fill(0),bytesPerRow=256;
      plan[WORD.recordCount]=s.records.length;
      for(const k of [WORD.requiredReferenceCount,WORD.scatteredReferenceCount,WORD.sortedReferenceCount,WORD.compositedReferenceCount])plan[k]=w.coverage;
      for(const k of [2,3,10,11])summary[k]=w.coverage;
      summary[40]=w.terminated;ordering[0]=w.coverage;
      const rgbaPadded=Array(bytesPerRow*s.height).fill(0);
      for(let y=0;y<s.height;y++)rgbaPadded.splice(y*bytesPerRow,s.width*4,...w.rgba.slice(y*s.width*4,(y+1)*s.width*4));
      return {mode:w.mode,width:s.width,height:s.height,coverageUnit:'tile-reference',
        orderContract:PRODUCTION_DEPTH_ORDER,sortAlgorithm:PRODUCTION_SORT_ALGORITHM,productionSortSubmitted:true,
        tileTable:e.table,orderedReferences:e.refs.length?e.refs:[0,0,0,0],orderingSummary:ordering,
        executionPlan:plan,summary,accumulation:w.accumulation,done:w.done,bytesPerRow,rgbaPadded};
    })};
}
const results=[];
for(const fixture of sortFixtures()) {
  const expected=expectedSortedFixture(fixture),input={...fixture.input,executionAuthorized:true};
  const f=fake(),raw=await runCompositingDiagnostic({input,requestAdapter:f.requestAdapter});
  assert.equal(raw.status,'observed-requires-independent-comparison');
  assert.equal(f.destroyed,true);assert.ok(f.objects.every(o=>o.destroyed));
  assert.equal(raw.schema,'small-compositing-diagnostic-result-v2');
  assert.equal(raw.frames.length,3);assert.equal(f.requests,1);
  const model=scheduledModel(f.dispatches);assert.equal(model.size,3);
  for(const refs of model.values())assert.deepEqual(refs.flat().slice(0,expected.refs.length),expected.refs,fixture.name);
  for(const d of f.dispatches.filter(d=>d.entry==='compositeReferenceChunk'))assert.ok(model.has(d.group.entries[2].resource.buffer),'actual sorted buffer connected');
  const sortShader=f.shaders.find(s=>s.includes('fn referenceBefore'));
  assert.ok(sortShader.includes('return a.w < b.w || (a.w == b.w && a.x < b.x);'));
  assert.ok(sortShader.includes('vec4f(-1.0, 0.0, 0.0, 0.0)'));
  assert.ok(sortShader.includes('referenceBefore(partnerReference, selfReference),\n    ascending'));
  assert.ok(sortShader.includes('referenceBefore(next, splatRef) || splatRef.x == next.x'));
  assert.ok(sortShader.includes('if (splatRef.x != -1.0)'));
  assert.ok(sortShader.includes('splatRef.x == floor(splatRef.x)'));
  const sim=synthetic(fixture);assert.equal(compareSortedDiagnostic(sim,fixture).frames,3);
  if(expected.refs.length) {
    assert.throws(()=>compareSortedDiagnostic(raw,fixture)); // fake API is not numerical evidence
    const missing=structuredClone(sim);missing.frames[0].orderedReferences.pop();
    assert.throws(()=>compareSortedDiagnostic(missing,fixture),/ordered-references/);
    const counts=structuredClone(sim);counts.frames[0].summary[2]--;
    assert.throws(()=>compareSortedDiagnostic(counts,fixture),/coverage/);
  }
  results.push({name:fixture.name,cpuModel:true,apiConnected:true,staticShader:true,savedConsumer:true,gpuExecuted:false});
}
const two=sortFixtures()[0],reversed=synthetic(two);
reversed.frames[0].orderedReferences=[0,0,4,4,1,0,1,1];
assert.throws(()=>compareSortedDiagnostic(reversed,two),/ordered-references/);
assert.notDeepEqual([.25,0,.375,.375],two.center); // far red first differs analytically
for(const mutate of [s=>s.tiles[0].push(0),s=>s.records[0][6]=0,s=>s.tiles[0][0]=99,s=>s.tileCols=0,s=>s.extra=1]) {
  const spec=structuredClone({...two.input,executionAuthorized:true});mutate(spec);
  assert.throws(()=>validateCompositingDiagnosticInput(spec),/compositing-diagnostic:/);
}
assert.equal(buildWebGpuTileDepthOrderingContract().orderHandling,PRODUCTION_DEPTH_ORDER);
assert.equal(buildWebGpuTileListCompositorContract().parallelSortAlgorithm,PRODUCTION_SORT_ALGORITHM);
assert.equal(buildWebGpuTileDepthOrderingContract({orderHandling:'depth-aware-compositor-sort-key-descending'}).orderHandling,
  'depth-aware-compositor-sort-key-descending'); // explicit old schema/fields remain readable
const production=await readFile(new URL('../demo/js/webgpu_bounded_tile_sort_and_compositor.js',import.meta.url),'utf8');
assert.match(production,/\} = executeBoundedProductionTileSort\(\{ device, resources \}\)/);
assert.ok(production.includes('(orderingSummaryRaw[20] ?? 0) === 0'));
const consumer=await readFile(new URL('../demo/js/webgpu_tile_list_compositor.js',import.meta.url),'utf8');
assert.ok(!consumer.includes('descending-sort-key'));assert.match(consumer,/parallelSortAlgorithm: PRODUCTION_SORT_ALGORITHM/);
// Existing 13 compositing inputs remain V1 and exercise the same API/shader.
const oldRoot=new URL('../../reports/webgpu-4dgs-viewer/phase3/step122/step122-new-prepare-validation1/gpu-fixtures/',import.meta.url);
const manifest=JSON.parse(await readFile(new URL('manifest.json',oldRoot),'utf8'));
assert.equal(manifest.cases.length,13);
for(const name of manifest.cases) {
  const input=JSON.parse(await readFile(new URL(name+'-input.json',oldRoot),'utf8'));
  const f=fake(),raw=await runCompositingDiagnostic({input:{...input,executionAuthorized:true},requestAdapter:f.requestAdapter});
  assert.equal(raw.schema,'small-compositing-diagnostic-result-v1');assert.equal(raw.frames.length,3);
  assert.ok(!f.dispatches.some(d=>d.entry==='seedOrderedReferences'));
}
if(process.argv[2])await writeFile(process.argv[2],JSON.stringify({schema:'step122-sort-fixture-candidates-v1',
  execution_authorized:false,review_required:true,gpu_executed:false,
  entry:'runCompositingDiagnostic',consumer:'compareSortedDiagnostic',
  cases:sortFixtures().map(fixture=>({...fixture,expected:expectedSortedFixture(fixture)}))},null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({status:'pass',results,legacyCompositingCases:13,legacyFrames:39,
  numericalEvidence:'CPU only; fake API and shader static checks are not WGSL execution',gpuExecuted:false}));
