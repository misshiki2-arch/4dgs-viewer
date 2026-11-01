import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolveCompositingPolicy, compositingFrameFields, compositingExecutionEvidence,
  validateCompositingResources, COMPOSITING_MATH_VERSION as MATH } from '../demo/js/common_4dgs_compositing_policy.js';
import { buildAlignedCompositorWgsl, executeBoundedProductionTileSortAndCompositor }
  from '../demo/js/webgpu_bounded_tile_sort_and_compositor.js';
import { compareWebGpuPresentationFrameIdentity as compareFrame, summarizeWebGpuPresentationFrameIdentity as frame }
  from '../demo/js/webgpu_presentation_capture_orientation_contract.js';

// Independent, CPU-only f32 reference. Not a runtime fallback and not WGSL execution.
const f = Math.fround, half = f(1/255), tau=f(1e-4);
function adjacent(value, delta) {
  const bytes=new ArrayBuffer(4),a=new Float32Array(bytes),u=new Uint32Array(bytes);
  a[0]=value;u[0]+=delta;return a[0];
}
function sample(C,T,done,alpha,rgb,on) {
  if(done) return {C,T,done};
  const a=f(Math.max(0,Math.min(f(alpha),f(0.99))));
  if(a<half) return {C,T,done};
  const next=f(T*f(1-a));
  if(on && next<tau) return {C,T,done:true};
  return {C:C.map((v,i)=>f(v+f(f(Math.max(rgb[i],0)*a)*T))),T:next,done:false};
}
const init=()=>({C:[0,0,0],T:1,done:false});
const row=(alpha,rgb=[1,1,1])=>({alpha,rgb});
function chunks(groups,on) {
  let state=init(),coverage=0,terminated=0;
  for(const group of groups) for(const x of group) {
    coverage++;const old=state;
    state=sample(state.C,state.T,state.done,x.alpha,x.rgb,on);
    if(!old.done&&state.done)terminated++;
  }
  return {...state,coverage,terminated};
}
const passed=[];
for(const delta of [-1,0,1]) {
  const result=sample([0,0,0],1,false,adjacent(half,delta),[1,1,1],true);
  assert.equal(result.C[0]===0,delta<0);
  const T=adjacent(f(tau*2),delta);
  const cut=sample([0.2,0.3,0.4],T,false,0.5,[2,1,0],true);
  assert.equal(cut.done,delta<0);
  if(delta<0){assert.deepEqual(cut.C,[0.2,0.3,0.4]);assert.equal(cut.T,T);}
}
passed.push('f32-neighbors-skip-and-test-T-equality');
for(const delta of [-1,0,1]) {
  const a=adjacent(f(.99),delta), result=sample([0,0,0],1,false,a,[1,1,1],false);
  assert.equal(result.C[0],Math.min(a,f(.99)));
}
assert.deepEqual(sample([0,0,0],1,false,.25,[2,-1,8],false).C,[.5,0,2]);
assert.deepEqual(sample([0,0,0],1,false,.25,[2,-1,8],false).C.map(x=>Math.min(x,1)),[.5,0,1]);
passed.push('cap-lower-only-RGB-final-quantization-separate');
const sequence=[...Array.from({length:7},()=>row(.5,[0,0,0])),row(.99,[2,2,2]),row(.5,[1,0,0])];
for(const on of [false,true,false,true]) {
  const all=chunks([sequence],on);
  for(let split=0;split<=sequence.length;split++)assert.deepEqual(chunks([sequence.slice(0,split),sequence.slice(split)],on),all);
  assert.equal(all.coverage,9);assert.equal(all.terminated,on?1:0);
  if(on){assert.deepEqual(all.C,[0,0,0]);assert.equal(all.T,1/128);}
  else assert.ok(all.C[0]>1/255);
}
assert.deepEqual(chunks([[]],true),{...init(),coverage:0,terminated:0});
assert.equal(chunks([[row(half/2)]],true).terminated,0);
passed.push('crossing-exclusion-chunks-clear-mode-toggle-coverage-count-once-empty-black');
const weight=(x,y,c,q)=>Math.exp(-.5*(q[0]*(x-c[0])**2+2*q[1]*(x-c[0])*(y-c[1])+q[2]*(y-c[1])**2));
for(const center of [[8,8],[0,0]])for(const q of [[1,0,1],[2,.25,1]]) {
  assert.equal(weight(...center,center,q),1);
  assert.equal(weight(center[0]+1,center[1]+2,center,q),weight(center[0]-1,center[1]-2,center,q));
}
passed.push('integer-center-symmetric-nondiagonal-edge-independent-oracle');
const shaderOff=buildAlignedCompositorWgsl(false),shaderOn=buildAlignedCompositorWgsl(true);
assert.equal(shaderOff.replace('TERMINATION_ENABLED: bool = false','TERMINATION_ENABLED: bool = true'),shaderOn);
for(const text of ['vec2f(f32(pixelX), f32(pixelY))','clamp(c.w * weight, 0.0, 0.99)',
  'sampleAlpha < (1.0 / 255.0)','testT < 0.0001','max(c.xyz, vec3f(0.0))',
  'clamp(1.0 - accumulated.w, 0.0, 1.0)','pixelDone[pixelIndex] = 0u;',
  'vec4f(0.0, 0.0, 0.0, 1.0)','atomicAdd(&evidence[10], 1u)'])assert.ok(shaderOn.includes(text),text);
assert.ok(shaderOn.indexOf('atomicAdd(&evidence[0]')<shaderOn.indexOf('if (pixelDone[pixelIndex]'));
assert.ok(shaderOn.indexOf('break;',shaderOn.indexOf('testT <'))<shaderOn.indexOf('accumulated = vec4f(',shaderOn.indexOf('testT <')));
passed.push('actual-WGSL-generated-production-code-static-not-GPU');
const authority={sha256:createHash('sha256').update('synthetic authority').digest('hex')};
const context={profile:'adopted-b-sh0-bundle-v1',rows:3248901,selection:{view:4,start:524288,count:65536},
  frame:['train',2588,'images/000118_v16.png',16.6],camera:{width:1280,height:720},
  activeSh:[0,0],maximumSh:[3,2],physicalSh:16,external:authority};
const grant={schema:'adopted-range-execution-authorization-v1',execution_authorized:true,purpose:'validation-only',
  candidate_sha256:createHash('sha256').update('synthetic unexecuted candidate').digest('hex')};
function policy(mode='NEW-ON',change={}) {
  return resolveCompositingPolicy({request:{schema:'adopted-range-compositing-policy-v1',mathVersion:MATH,mode,
    authority,authorization:{...grant,mode},...change},adoptedBundle:context,requestIdentity:'test-'+mode});
}
const on=policy(),off=policy('NEW-OFF');
assert.equal(on.normalOnAccepted,false);assert.equal(resolveCompositingPolicy().appliedMode,'NEW-OFF');
for(const change of [{mode:'ON'}, {mathVersion:'wrong'}, {authorization:{...grant,mode:'NEW-ON',execution_authorized:false}},
  {authority:{}}])assert.throws(()=>policy('NEW-ON',change),/compositing-policy:/);
assert.throws(()=>resolveCompositingPolicy({request:{schema:'adopted-range-compositing-policy-v1',mathVersion:MATH,mode:'NEW-ON',authority,
  authorization:{...grant,mode:'NEW-ON'}},adoptedBundle:{...context,rows:2},requestIdentity:'test'}),/scope/);
const a=frame({generation:1,datasetCameraLabel:'view',datasetFrameNumber:1,datasetTime:0,
  referenceCameraLabel:'view',outputWidth:16,outputHeight:16,...compositingFrameFields(off)});
assert.equal(compareFrame(a,{...a,frameHash:null}).matches,true);
assert.equal(compareFrame(a,{...a,...compositingFrameFields(on)}).matches,false);
assert.equal(compareFrame(a,{...a,generation:2}).matches,false);
for(const k of ['compositingMathVersion','compositingMode','compositingRequestIdentity']) {
  const bad={...a};delete bad[k];assert.equal(compareFrame(a,bad).matches,false);
}
assert.equal(compositingExecutionEvidence(on,true,0,256).terminationObserved,false);
assert.equal(compositingExecutionEvidence(on,true,1,256).onPathExecuted,true);
assert.throws(()=>compositingExecutionEvidence(off,true,1,256));
passed.push('real-policy-authorization-scope-frame-version-mode-request-negative-and-count');
const limits={maxStorageBuffersPerShaderStage:8,maxStorageBufferBindingSize:2**24,maxBufferSize:2**24};
validateCompositingResources({limits},1280,720);
for(const bad of [{...limits,maxStorageBuffersPerShaderStage:7},{...limits,maxBufferSize:16},{}]) {
  let touched=false;
  await assert.rejects(executeBoundedProductionTileSortAndCompositor({device:{limits:bad,createBuffer(){touched=true;}},
    resources:{referenceCapacity:1},outputWidth:1280,outputHeight:720,compositingPolicy:off}),/compositing-policy:/);
  assert.equal(touched,false);
}
passed.push('real-dispatch-entry-resource-rejection-before-GPU-allocation');
// Real production wiring is inspected here; run selected snapshot/capture smoke
// separately. These assertions do not simulate presentation or shader numbers.
const source=await readFile(new URL('../demo/js/webgpu_production_frame_data_path.js',import.meta.url),'utf8');
assert.ok(source.indexOf('resolveCompositingPolicy({')<source.indexOf('await acquireProductionDevice('));
assert.ok(source.indexOf('validateCompositingResources(device')<source.indexOf('await buildWebGpu4DStatePositionsForCandidates'));
assert.ok(source.includes('requestMetadata?.compositingPolicyRequest'));
console.log(JSON.stringify({passed,cpuOnly:true,shaderExecuted:false,normalOnAccepted:false},null,2));
