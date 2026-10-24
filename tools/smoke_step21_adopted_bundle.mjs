import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { webcrypto } from 'node:crypto';
import { resolve, join } from 'node:path';
import { bundleFixture, spl4Fixture, readySnapshot } from './step21_adopted_bundle_fixture.mjs';
import { parseSplat4DV2 } from '../demo/js/splat4d_parser_v2.js';
import { createViewerFileIO } from '../demo/js/viewer_file_io.js';
import { getAdoptedBundleContext, adoptedFrameArguments, adoptedDeterministicState,
  validateAdoptedFrame, applyAdoptedUiDefaults } from '../demo/js/common_4dgs_adopted_bundle.js';
import { buildWebGpuProjectionContract } from '../demo/js/common_4dgs_projection_contracts.js';
import { buildWebGpuRenderStateManifest } from '../demo/js/common_4dgs_render_state_manifest.js';
import { runNativeWebGpuProductionFrameDataPath } from '../demo/js/webgpu_production_frame_data_path.js';
import { createPopulationSemanticComparisonController } from '../demo/js/webgpu_population_semantic_comparison_controller.js';
import { buildPopulationAlignedSemanticComparisonChunkPlan } from '../demo/js/webgpu_population_aligned_semantic_comparison_orchestrator.js';
import { buildProductionCandidateAttributeInput } from '../demo/js/common_4dgs_candidate_attribute_input_contracts.js';
import { buildCudaDegree2TileInputExpectedRgb } from '../demo/js/webgpu_population_aligned_semantic_comparison.js';
import { buildWebGpu4DStatePositionsForCandidates } from '../demo/js/webgpu_4d_state_evaluator.js';
import { getVisibleBuildConfig } from '../demo/js/gpu_visible_builder.js';

globalThis.crypto ??= webcrypto;
globalThis.GPUBufferUsage = { STORAGE:1,UNIFORM:2,COPY_SRC:4,COPY_DST:8,MAP_READ:16 };
globalThis.GPUMapMode = {READ:1};
const passed=[];
const limits={maxStorageBufferBindingSize:1073741824,maxBufferSize:1073741824,maxStorageBuffersPerShaderStage:8};
const copy = (x)=>JSON.parse(JSON.stringify(x));
const projection = (a)=>buildWebGpuProjectionContract({camera:{},screenSpaceCamera:a.screenSpaceCamera,renderW:a.canvasWidth,renderH:a.canvasHeight,sx:1,sy:1});
function fileIO(onSceneLoaded=()=>{}, parser=parseSplat4DV2) {
  return createViewerFileIO({ui:{},parseArrayBuffer:parser,onSceneLoaded});
}
function fakeEvaluatorDevice() {
  const shaders=[],buffers=[];
  return {shaders,buffers,device:{limits,
    createBuffer(descriptor){const bytes=new ArrayBuffer(descriptor.size); const b={descriptor,bytes,destroyed:false,
      getMappedRange:()=>bytes,unmap(){},destroy(){this.destroyed=true;}};buffers.push(b);return b;},
    createShaderModule({code}){shaders.push(code);return {code};},
    createComputePipeline:()=>({getBindGroupLayout:()=>({})}),createBindGroup:(x)=>x,
    createCommandEncoder:()=>({beginComputePass:()=>({setPipeline(){},setBindGroup(){},dispatchWorkgroups(){},end(){}}),finish:()=>({})}),
    queue:{submit(){},onSubmittedWorkDone:async()=>{}}}};
}
async function checkScene(raw, { real=false }={}) {
  const c=getAdoptedBundleContext(raw), a=adoptedFrameArguments(raw), p=projection(a);
  assert.equal(validateAdoptedFrame({raw,...a,projectionData:p.data}),c);
  assert.equal(p.data[10],Math.fround(c.camera.cx-0.5));
  assert.equal(p.data[11],Math.fround(c.camera.cy-0.5));
  assert.deepEqual(Array.from(p.data.slice(12,16)),c.camera.world_view_transform.map(row=>Math.fround(row[0])));
  const state=adoptedDeterministicState(raw);
  assert.deepEqual(state.referenceCameraPose.position,c.camera.camera_center);
  assert.deepEqual(state.referenceCameraPose.forward,Array.from(p.data.slice(20,23)));
  assert.deepEqual(state.referenceCameraPose.up,Array.from(p.data.slice(16,19)).map(x=>-x));
  for(const [key,value] of [['timestamp',-1],['useSH',false],['stride',2],['renderScale',0.5]]) {
    assert.throws(()=>validateAdoptedFrame({raw,...a,projectionData:p.data,buildConfig:{...a.buildConfig,[key]:value}}));
  }
  assert.throws(()=>validateAdoptedFrame({raw,...a,projectionData:p.data,productionResidentSelectionRequest:{...a.productionResidentSelectionRequest,rangeCount:a.productionResidentSelectionRequest.rangeCount+1}}));
  let adapters=0;
  Object.defineProperty(globalThis,'navigator',{configurable:true,value:{gpu:{requestAdapter:async()=>{adapters++;return null;}}}});
  const bad=await runNativeWebGpuProductionFrameDataPath({raw,...a,camera:{},buildConfig:{...a.buildConfig,timestamp:-1}});
  assert.equal(adapters,0); assert.match(JSON.stringify(bad),/config-timestamp/);
  const noDevice=await runNativeWebGpuProductionFrameDataPath({raw,...a,camera:{}});
  assert.equal(adapters,1);assert.match(JSON.stringify(noDevice),/device-unavailable/);
  delete globalThis.navigator;
  const snapshot=readySnapshot(raw);
  assert.equal(snapshot.status,'ready',JSON.stringify(snapshot.blockedReasons));
  const capture=(s=snapshot,png=null)=>buildWebGpuRenderStateManifest({raw,rawSummary:{assetSha256:raw.assetSha256},
    productionEvaluationInputContract:s,pngCaptureStatus:png,deterministicState:adoptedDeterministicState(raw),
    canvasSizeSummary:{width:a.canvasWidth,height:a.canvasHeight}});
  assert.equal(capture().adoptedHandoff.status,'ready');
  assert.equal(capture().srcIndexSemantics.status,'verified-direct');
  assert.deepEqual(capture().camera.worldViewTransform,a.screenSpaceCamera.screenSpaceTransformOverride.viewMatrix);
  const frame=snapshot.productionIdentity.presentedFrameIdentity;
  assert.equal(capture(snapshot,{capturedStateIdentity:frame,presentedStateIdentity:frame}).adoptedHandoff.pngBound,true);
  assert.equal(capture(snapshot,{capturedStateIdentity:{...frame,generation:6},presentedStateIdentity:frame}).adoptedHandoff.status,'blocked');
  const stale=copy(snapshot);stale.productionIdentity.presentedGeneration=6;
  assert.equal(capture(stale).adoptedHandoff.status,'blocked');
  const wrong=copy(snapshot);wrong.adoptedBundle.selection.start++;
  assert.equal(capture(wrong).adoptedHandoff.status,'blocked');
  const wrongProjection=copy(snapshot);wrongProjection.projection.values[12]+=1;
  assert.equal(capture(wrongProjection).adoptedHandoff.status,'blocked');
  const wrongTime=copy(snapshot);wrongTime.appliedConfig.timestamp++;
  assert.equal(capture(wrongTime).adoptedHandoff.status,'blocked');
  const layout=buildProductionCandidateAttributeInput({raw,candidateIndices:Uint32Array.from(c.selection.representatives),
    rawXyzOpacity:new Float32Array(c.selection.representatives.length*4),projectionParams:p.data,deviceLimits:limits});
  assert.equal(layout.ready,true,JSON.stringify(layout.contract.blockedReasons));
  assert.equal(layout.data.length,c.selection.representatives.length*32);
  const numerical=[];
  c.selection.representatives.forEach((index,row)=>{
    const dc=Array.from(raw.f_dc.subarray(index*3,index*3+3));
    const expected=dc.map(x=>Math.max(x/(2*Math.sqrt(Math.PI))+0.5,0)); // independent analytical C0
    const actual=buildCudaDegree2TileInputExpectedRgb({raw,srcIndex:index,cameraWorldPosition:Array.from(raw.xyz.subarray(index*3,index*3+3))});
    actual.forEach((v,i)=>assert.ok(Math.abs(v-expected[i])<1e-12));
    assert.deepEqual(Array.from(layout.data.subarray(row*32,row*32+3)),dc);
    assert.deepEqual(Array.from(layout.data.subarray(row*32+8,row*32+32)),Array.from(raw.f_rest.subarray(index*45,index*45+24)));
    numerical.push({index,dc,expected});
  });
  assert.ok(numerical.some(x=>x.expected.some(v=>v>1)));
  assert.ok(numerical.some(x=>x.expected.some(v=>v===0)));
  assert.equal(raw.frestDim,45);
  const fake=fakeEvaluatorDevice();
  for(const scene of [raw,parseSplat4DV2((()=>{const b=spl4Fixture();new DataView(b).setInt32(12,2,true);return b;})()),raw]) {
    const result=await buildWebGpu4DStatePositionsForCandidates({device:fake.device,raw:scene,candidateIndices:new Uint32Array([scene===raw?c.selection.representatives[0]:1]),
      rawXyzOpacity:new Float32Array([0,0,1,0]),buildConfig:a.buildConfig,projectionParams:p.data,readbackPolicy:'none',keepGpuResources:true});
    assert.equal(result.productionReadbackPerformed,false);
    assert.equal(result.gaussianAttributeEvaluationContract.spatialShDegree,scene.activeShDegree);
    assert.ok(fake.shaders.at(-1).includes(`let rgb = evaluateProduction${scene.activeShDegree===0?'Sh0':'Degree2SpatialSh'}(row, raw0.xyz)`));
    Object.values(result.gpuResources??{}).forEach(v=>v?.destroy?.());
  }
  assert.equal(fake.shaders.length,3); // no cross-degree cached shader reuse
  if(real) {
    assert.equal(buildPopulationAlignedSemanticComparisonChunkPlan(raw).length,1);
    let received=null, destroyed=false;
    const controller=createPopulationSemanticComparisonController({requestAdapter:async()=>({requestDevice:async()=>({queue:{onSubmittedWorkDone:async()=>{}},destroy(){destroyed=true;}})}),
      runOrchestrator:async(input)=>{received=input;throw new Error('test-boundary-no-real-GPU');}});
    const result=await controller.runPopulationAlignedSemanticComparison({raw,productionEvaluationInputContract:snapshot});
    assert.ok(received,JSON.stringify(result));assert.equal(received.raw,raw);assert.equal(received.populationContractIdentity.residentRecordCount,c.selection.count);
    assert.equal(received.buildConfig.timestamp,c.frame[3]);assert.equal(destroyed,true);
  }
  passed.push(real?'real-fileIO-parser-camera-snapshot-capture-controller-sh0':'fixture-fileIO-parser-camera-snapshot-capture-sh0');
  return {rows:raw.N,profile:c.profile,selection:c.selection,coverage:c.coverage,external:c.external,numerical,
    productionGpu:'adapter-null CPU boundary only',evaluatorGpu:'fake API; WGSL not executed',snapshotPresentation:'CPU fixture; not browser acceptance'};
}

if(process.argv[2]==='--real') {
  const reports=resolve(process.argv[3]);
  const json=async(p)=>JSON.parse(await readFile(join(reports,p),'utf8'));
  const ev=await json('phase4/step20/step20-validation1/step20-validation1-evidence.json');
  const instruction=await json('phase5/step21/step21-implement1/step21-implement1-instruction-evidence.json');
  const design=await json('phase5/step21/step21-design1/step21-design1-design-evidence.json');
  const root=join(reports,'phase4/step20/step20-implement1/reference-run');
  const authority={...ev.external_digest,receipt:instruction.references_before[join(root,'receipt.json')]};
  const manifest=await json('phase4/step20/step20-implement1/reference-run/bundle/manifest.json');
  const s=design.selection_candidate;
  const selection={view:manifest.references.findIndex(r=>JSON.stringify(r.frame)===JSON.stringify(s.view)),
    start:s.chosen.start,count:s.chosen.count,representatives:[...new Set(s.representatives.map(x=>x.source_index))]};
  let published=0,reads=0;
  const raw=await fileIO(()=>published++).loadAdoptedBundle({authority,selection,readResource:async(path)=>{
    reads++;const b=await readFile(join(root,path));return b.byteOffset===0&&b.byteLength===b.buffer.byteLength?b.buffer:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);
  }});
  assert.equal(published,1);assert.equal(reads,163);
  console.log(JSON.stringify({passed,reads,...await checkScene(raw,{real:true})},null,2));
} else {
  const f=bundleFixture();let parsedBytes,scene;
  const io=fileIO((raw)=>{scene=raw;},b=>{parsedBytes=b;return parseSplat4DV2(b);});
  const raw=await io.loadAdoptedBundle(f);assert.equal(raw,scene);assert.deepEqual(parsedBytes,f.files.get('bundle/population.splat4d'));
  assert.deepEqual(Array.from(raw.scale_xyz.slice(0,3)),[2,3,4]);
  await checkScene(raw);
  const ui=Object.fromEntries(['timeSlider','timeDurationSlider','splatScaleSlider','sigmaScaleSlider','prefilterVarSlider','renderScaleSlider','strideSlider','useSHCheck','bgGraySlider',
    'maxVisibleSlider','useRot4dCheck','useNativeRot4dCheck','useNativeMarginalCheck','forceSh3dCheck'].map(k=>[k,{}]));
  applyAdoptedUiDefaults(raw,ui);assert.equal(ui.useSHCheck.checked,true);assert.equal(ui.timeSlider.value,'0.5');
  const frameArgs=adoptedFrameArguments(raw);
  assert.ok(validateAdoptedFrame({raw,...frameArgs,projectionData:projection(frameArgs).data,buildConfig:getVisibleBuildConfig(ui)}));
  ui.timeSlider.value='1';
  assert.throws(()=>validateAdoptedFrame({raw,...frameArgs,projectionData:projection(frameArgs).data,buildConfig:getVisibleBuildConfig(ui)}),/config-timestamp/);
  const invalid=[b=>b.slice(0,127),b=>b.slice(0,b.byteLength-1),b=>{const n=new Uint8Array(b.byteLength+4);n.set(new Uint8Array(b));return n.buffer;},
    b=>{new DataView(b).setUint32(24,4,true);return b;}, b=>{new DataView(b).setUint8(20,2);return b;},
    b=>{new DataView(b).setUint8(23,0);return b;}, b=>{new DataView(b).setUint8(60,1);return b;},
    b=>{new DataView(b).setFloat32(128,NaN,true);return b;},b=>{new Float32Array(b,128).fill(0,3,7);return b;},
    b=>{new DataView(b).setFloat32(128+11*4,1000,true);return b;},b=>{new DataView(b).setFloat32(128+11*4,-1000,true);return b;}];
  invalid.forEach(change=>assert.throws(()=>parseSplat4DV2(change(spl4Fixture()))));passed.push('strict-parser-11-negative-cases');
  for(const mutator of [({manifest})=>manifest.version=2,({population})=>population.active_sh=[1,0],
    ({manifest})=>manifest.references[0].native.scalars[23]=9,({verification})=>verification.raw14_exact=false]) {
    let publications=0;await assert.rejects(fileIO(()=>publications++).loadAdoptedBundle(bundleFixture(mutator)));assert.equal(publications,0);
  }
  const wrong=bundleFixture();new Uint8Array(wrong.files.get('bundle/references/00/render.npy'))[0]^=1;
  await assert.rejects(fileIO().loadAdoptedBundle(wrong),/member-hash/);
  const missing=bundleFixture();missing.files.delete('bundle/references/01/gt.npy');await assert.rejects(fileIO().loadAdoptedBundle(missing));
  const stale=bundleFixture();stale.authority.index.sha256='0'.repeat(64);await assert.rejects(fileIO().loadAdoptedBundle(stale));
  const unknown=bundleFixture();unknown.authority.profile='unknown';await assert.rejects(fileIO().loadAdoptedBundle(unknown),/authority-fields/);
  const legacy=await fileIO().loadArrayBuffer(spl4Fixture(),{verified:true});assert.equal(getAdoptedBundleContext(legacy),null);
  assert.throws(()=>getAdoptedBundleContext({...raw}),/unverified-scene/);
  const saved=raw.f_dc;raw.f_dc=saved.slice();assert.throws(()=>getAdoptedBundleContext(raw),/scene-replaced/);raw.f_dc=saved;
  let release;const held=new Promise(r=>release=r);const race=bundleFixture();let last;
  const racing=fileIO(x=>last=x);const pending=racing.loadAdoptedBundle({...race,readResource:async(p)=>{if(p==='result.json')await held;return race.readResource(p);}});
  const newest=await racing.loadArrayBuffer(spl4Fixture());release();await assert.rejects(pending,/superseded/);assert.equal(last,newest);
  passed.push('publication-negative-8-and-legacy-ownership-load-race');
  console.log(JSON.stringify({passed,tests:'CPU only',fakeBoundaries:['GPU adapter/device','presentation observations'],browser:false},null,2));
}
