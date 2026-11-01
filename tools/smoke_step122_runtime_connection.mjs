// Real production JS call graph, with only GPU API execution replaced by a
// canonical zero-reference observation. This is not rendered-image validation.
import assert from 'node:assert/strict';
import { PerspectiveCamera } from '../build/demo/lib/three.module.js';
import { spl4Fixture } from './step21_adopted_bundle_fixture.mjs';
import { parseSplat4DV2 } from '../demo/js/splat4d_parser_v2.js';
import { runNativeWebGpuProductionFrameDataPath } from '../demo/js/webgpu_production_frame_data_path.js';
import { buildWebGpuProjectionContract } from '../demo/js/common_4dgs_projection_contracts.js';
import { getVisibleBuildConfig } from '../demo/js/gpu_visible_builder.js';
import { PRODUCTION_TILE_EXECUTION_PLAN_WORD as W, PRODUCTION_TILE_EXECUTION_PLAN_MAGIC as MAGIC,
  PRODUCTION_TILE_EXECUTION_PLAN_STATUS as STATUS } from '../demo/js/common_4dgs_production_tile_execution_plan_contracts.js';

globalThis.GPUBufferUsage={STORAGE:1,UNIFORM:2,COPY_SRC:4,COPY_DST:8,MAP_READ:16,INDIRECT:32};
globalThis.GPUTextureUsage={STORAGE_BINDING:1,COPY_SRC:2,TEXTURE_BINDING:4,RENDER_ATTACHMENT:8};
globalThis.GPUShaderStage={COMPUTE:4};globalThis.GPUMapMode={READ:1};
const buffers=[],shaders=[],dispatched=[];
const gpu={limits:{maxStorageBufferBindingSize:4096,maxBufferSize:4096,maxStorageBuffersPerShaderStage:8,
  maxComputeWorkgroupsPerDimension:65535,minUniformBufferOffsetAlignment:256},
  createBuffer(d){const data=new ArrayBuffer(d.size);const b={d,data,destroyed:false,
    getMappedRange:()=>data,unmap(){},async mapAsync(){},destroy(){this.destroyed=true;}};buffers.push(b);return b;},
  createTexture(d){return {d,createView:()=>({}),destroy(){}};},
  createShaderModule(d){shaders.push(d);return d;},
  createBindGroupLayout:d=>d,createPipelineLayout:d=>d,createBindGroup:d=>d,
  createComputePipeline(d){return {...d,getBindGroupLayout:()=>({})};},
  createCommandEncoder(){const operations=[];return {
    beginComputePass(){let pipeline,group;return {setPipeline(p){pipeline=p;},setBindGroup(_,g){group=g;},
      dispatchWorkgroups(){const p=pipeline,g=group;operations.push(()=>kernel(p,g));},
      dispatchWorkgroupsIndirect(){const p=pipeline,g=group;operations.push(()=>kernel(p,g));},end(){}};},
    copyBufferToBuffer(a,ao,b,bo,n){operations.push(()=>new Uint8Array(b.data,bo,n).set(new Uint8Array(a.data,ao,n)));},
    clearBuffer(b,offset=0,size=b.data.byteLength-offset){operations.push(()=>new Uint8Array(b.data,offset,size).fill(0));},
    copyTextureToBuffer(){},finish:()=>operations};},
  queue:{submit(commands){for(const commandsOfEncoder of commands)for(const op of commandsOfEncoder)op();},
    async onSubmittedWorkDone(){},writeBuffer(b,offset,data){new Uint8Array(b.data,offset,data.byteLength).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));}}
};
function kernel(p,g) {
  const entry=p.compute.entryPoint;dispatched.push(entry);
  const b=index=>g.entries.find(x=>x.binding===index)?.resource.buffer;
  // Shader boundary fixture: all records invisible, therefore empty tile work.
  // Populate observer words, not application contracts or success flags.
  if(entry==='finalizePlan') {
    const params=new Uint32Array(b(6).data),plan=new Uint32Array(b(5).data);
    plan.fill(0);
    for(const [key,value] of Object.entries({magic:MAGIC,status:STATUS.ready,recordCount:params[1],tileCount:params[0],
      referenceCapacity:params[2],compactOffsetTableReady:1,planIdentity:params[4]}))plan[W[key]]=value;
  }
  if(entry==='finalizeSummary') {
    const summary=new Float32Array(b(4).data),params=new Float32Array(b(5).data);
    summary.fill(0);summary[0]=params[0];summary[4]=1;summary[9]=87;summary[15]=1;
  }
}
let requests=0;
Object.defineProperty(globalThis,'navigator',{configurable:true,value:{gpu:{requestAdapter:async()=>({requestDevice:async()=>{requests++;return gpu;}})}}});
const camera=new PerspectiveCamera(60,1,0.1,100);camera.position.set(1,2,3);camera.updateMatrixWorld(true);
// The ordinary Three.js path is legacy SH2. SH0 requires the verified adopted
// loader and is tested separately; never mark an unverified SH0 scene as ready.
const bytes=spl4Fixture();new DataView(bytes).setInt32(12,2,true);
const raw=parseSplat4DV2(bytes);
const originalScales=Array.from(raw.scale_xyz);
const ui=Object.fromEntries(['timeSlider','timeDurationSlider','splatScaleSlider','sigmaScaleSlider','prefilterVarSlider',
  'renderScaleSlider','strideSlider','useSHCheck','maxVisibleSlider','useRot4dCheck','useNativeRot4dCheck',
  'useNativeMarginalCheck','forceSh3dCheck'].map(k=>[k,{}]));
for(const [key,value] of Object.entries({timeSlider:.5,timeDurationSlider:33,splatScaleSlider:1.5,sigmaScaleSlider:2,
  prefilterVarSlider:0,renderScaleSlider:.5,strideSlider:1,maxVisibleSlider:3}))ui[key].value=String(value);
ui.useSHCheck.checked=true;ui.useRot4dCheck.checked=true;
const buildConfig=getVisibleBuildConfig(ui);
assert.equal(buildConfig.scalingModifier,1.5);assert.equal(buildConfig.sigmaScale,2);assert.equal(buildConfig.renderScale,.5);
const before=buildWebGpuProjectionContract({camera,renderW:8,renderH:8,sx:2,sy:2});
assert.equal(before.data[0],0);assert.equal(before.data[10],3.5);assert.equal(before.data[11],3.5);
assert.ok(Math.abs(before.data[8]-Math.fround(8/(2*Math.tan(Math.PI/6))))<1e-6);
const result=await runNativeWebGpuProductionFrameDataPath({raw,camera,canvasWidth:16,canvasHeight:16,buildConfig,
  viewerCanvasState:{schedulerFrameState:{requestIdentity:'cpu-runtime-request'}},
  metadata:{deterministicState:{datasetCameraLabel:'synthetic',datasetFrameNumber:1,datasetTime:.5}}});
assert.equal(result.status,'ok',JSON.stringify(result));
assert.equal(requests,1);
const snap=result.webgpuProductionFrameDataPathContract.productionEvaluationInputContract;
assert.equal(snap.compositingPolicy.appliedMode,'NEW-OFF');assert.equal(snap.compositingPolicy.executed,true);
assert.equal(snap.compositingPolicy.requestIdentity,'cpu-runtime-request');
assert.equal(snap.compositingPolicy.normalOnAccepted,false);
assert.equal(result.webgpuTileListCompositorContract.productionWorkClassification,'zero-reference');
assert.deepEqual(snap.projection.values,Array.from(before.data));
assert.deepEqual(raw.scale_xyz,Float32Array.from(originalScales));
assert.deepEqual(buildWebGpuProjectionContract({camera,renderW:8,renderH:8,sx:2,sy:2}).data,before.data);
assert.ok(shaders.some(x=>x.code.includes('TERMINATION_ENABLED: bool = false')));
assert.ok(dispatched.includes('compositeReferenceChunk'));assert.ok(dispatched.includes('finalizeSummary'));
assert.ok(buffers.some(b=>b.d.size===16&&new Uint32Array(b.data)[0]===raw.N&&new Float32Array(b.data)[1]===.5&&
  new Float32Array(b.data)[2]===1.5&&new Float32Array(b.data)[3]===2),'actual evaluator scale/time uniforms');
delete globalThis.navigator;
console.log(JSON.stringify({status:'pass',runtimeStatus:result.status,requests,canonicalZeroReference:true,
  mathVersion:snap.compositingPolicy.mathVersion,productionSourceReadbackPolicy:'none',projectionUnchanged:true,
  actualThreeCamera:true,scaleInputsPreserved:true,shaderExecuted:false,presentationTested:false}));
