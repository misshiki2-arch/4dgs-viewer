import assert from 'node:assert/strict';
import { runCompositingDiagnostic } from '../demo/js/webgpu_compositing_diagnostic.js';
import { buildAlignedCompositorWgsl } from '../demo/js/webgpu_bounded_tile_sort_and_compositor.js';

globalThis.GPUBufferUsage={STORAGE:1,COPY_SRC:2,COPY_DST:4,UNIFORM:8,MAP_READ:16};
globalThis.GPUTextureUsage={STORAGE_BINDING:1,COPY_SRC:2};
globalThis.GPUShaderStage={COMPUTE:1};globalThis.GPUMapMode={READ:1};
const input={schema:'small-compositing-diagnostic-v1',executionAuthorized:true,width:16,height:16,chunkSize:2,
  records:Array.from({length:9},(_,i)=>[8,8,3,0,2,.25,9-i,1,2,-1,.5,i===7?.99:.5]),modes:['NEW-OFF','NEW-ON','NEW-OFF']};
export function fake({failMap=false,bindings=8}={}) {
  const objects=[],shaders=[],entries=[],dispatches=[];let destroyed=false,requests=0;
  const device={limits:{maxStorageBuffersPerShaderStage:bindings,maxStorageBufferBindingSize:2**24,maxBufferSize:2**24},
    pushErrorScope(){},async popErrorScope(){return null;},
    createBuffer(d){const bytes=new ArrayBuffer(d.size);const b={d,bytes,destroyed:false,getMappedRange:()=>bytes,
      async mapAsync(){if(failMap)throw new Error('synthetic-map-failure');},unmap(){},destroy(){this.destroyed=true;}};objects.push(b);return b;},
    createTexture(d){const t={d,destroyed:false,createView:()=>({texture:t}),destroy(){this.destroyed=true;}};objects.push(t);return t;},
    createShaderModule(d){shaders.push(d.code);return d;},createBindGroupLayout:d=>d,createPipelineLayout:d=>d,
    createBindGroup(d){entries.push(d.entries);return d;},createComputePipeline:d=>d,
    createCommandEncoder(){return {beginComputePass(){let pipeline,group;return {setPipeline(p){pipeline=p;},setBindGroup(_,g){group=g;},
      dispatchWorkgroups(...grid){dispatches.push({entry:pipeline.compute.entryPoint,grid,group,
        params:group.entries[4]?.resource.buffer?.bytes?.slice(0)});},end(){}};},
      copyBufferToBuffer(a,ao,b,bo,n){new Uint8Array(b.bytes,bo,n).set(new Uint8Array(a.bytes,ao,n));},
      copyTextureToBuffer(){},finish:()=>({})};},
    queue:{writeBuffer(b,offset,data){new Uint8Array(b.bytes,offset,data.byteLength).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));},
      submit(){},async onSubmittedWorkDone(){}},destroy(){destroyed=true;}};
  return {objects,shaders,entries,dispatches,get destroyed(){return destroyed;},get requests(){return requests;},
    requestAdapter:async()=>({requestDevice:async options=>{requests++;assert.equal(options.requiredLimits.maxStorageBuffersPerShaderStage,8);return device;}})};
}
let touched=false;
for(const altered of [{...input,executionAuthorized:false},{...input,width:33},{...input,modes:['OLD']},
  {...input,records:[Array(11).fill(0)]},{...input,chunkSize:0}])
  await assert.rejects(runCompositingDiagnostic({input:altered,requestAdapter:()=>{touched=true;}}),/compositing-diagnostic:/);
await assert.rejects(runCompositingDiagnostic({input,device:{}}),/no-production-resource/);
assert.equal(touched,false);
for(const options of [{},{failMap:true},{bindings:7}]) {
  const f=fake(options),result=await runCompositingDiagnostic({input,requestAdapter:f.requestAdapter});
  assert.equal(f.requests,1);assert.equal(f.destroyed,true);assert.ok(f.objects.every(x=>x.destroyed));
  assert.equal(result.productionDeviceUsed,false);assert.equal(result.accepted,false);assert.equal(result.cleanupCompleted,true);
  if(options.failMap)assert.match(result.failure.message,/synthetic-map/);
  else if(options.bindings){assert.match(result.failure.message,/storage-bindings/);assert.equal(f.objects.length,0);}
  else {
    assert.equal(result.status,'observed-requires-independent-comparison');assert.equal(result.frames.length,3);
    assert.deepEqual(f.shaders,[false,true,false].map(buildAlignedCompositorWgsl));
    assert.equal(f.entries[0].length,11); // eight storage, two uniform, one texture
    assert.equal(f.dispatches.filter(x=>x.entry==='clearProductionAccumulation').length,3);
    assert.equal(f.dispatches.filter(x=>x.entry==='compositeReferenceChunk').length,15);
    assert.ok(f.dispatches.filter(x=>x.entry==='compositeReferenceChunk').every(x=>x.grid[0]===2&&x.grid[1]===2));
  }
}
console.log(JSON.stringify({status:'pass',authorizationNegatives:6,isolatedCases:3,
  actualProductionShader:true,cpuOnly:true,shaderExecuted:false,numericalAcceptance:false}));
