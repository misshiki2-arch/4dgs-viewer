// CPU-only generated-wrapper/capture integration. Synthetic GPU observations,
// actual parser, bundle verifier, binding and policy; never opens a browser.
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { webcrypto } from 'node:crypto';
import { bundleFixture, readySnapshot, identity } from './step21_adopted_bundle_fixture.mjs';
import { parseSplat4DV2 } from '../demo/js/splat4d_parser_v2.js';
import * as adopted from '../demo/js/common_4dgs_adopted_bundle.js';
import { resolveCompositingPolicy, compositingFrameFields, compositingExecutionEvidence }
  from '../demo/js/common_4dgs_compositing_policy.js';

globalThis.crypto ??= webcrypto;
const root=resolve(process.argv[2]);
assert.ok(['step122-numerical-alignment-fix1','step122-sort-alignment-fix1','step122-normal-on-implement1','step122-normal-on-fix1','step122-normal-on-fix2','step122-normal-on-png-fix1','step122-normal-on-png-fix2'].some(task=>
  root.startsWith('/home/demo/work/reports/webgpu-4dgs-viewer/phase3/step122/'+task+'/')));
await mkdir(root); // exclusive, keep every fixture and failure result
const copy=x=>JSON.parse(JSON.stringify(x));
const external=process.argv[3]?JSON.parse(await readFile(process.argv[3],'utf8')):null;
const generated=external?.generated??root;
const fixture=external?{authority:external.input.authority,selection:external.input.selection,
  readResource:async name=>{const b=await readFile(join(external.publication,name));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);}}:bundleFixture();
const raw=await adopted.loadVerifiedAdoptedBundle({...fixture,parseArrayBuffer:parseSplat4DV2});
const input=external?.input??{authority:fixture.authority,selection:fixture.selection,baseUrl:'/synthetic-bundle/'};
const normalCandidate=external?.normal?JSON.parse(await readFile(join(generated,'candidate.json'),'utf8')):null;
let normalRuntime=null;
if(normalCandidate){
  // Trusted data boundary is synthetic only in this CPU VM. Real scope/frame
  // validation and policy code are evaluated; no production override is added.
  const ctx=vm.createContext({URLSearchParams}),cache=new Map();
  async function module(path){
    if(cache.has(path))return cache.get(path);
    const m=new vm.SourceTextModule(await readFile(path,'utf8'),{context:ctx,identifier:path});cache.set(path,m);return m;
  }
  const normalPath=resolve('demo/js/common_4dgs_normal_on.js');
  async function link(spec,parent){
    const path=resolve(parent.identifier,'..',spec);
    if(path===normalPath)return module(path);
    const api=spec.endsWith('common_4dgs_normal_adopted_scope.js')
      ? {NORMAL_ADOPTED_SCOPE:normalCandidate.normal_scope.data,NORMAL_SCOPE_IDENTITY:normalCandidate.normal_scope.identity}
      : await import(path);
    return new vm.SyntheticModule(Object.keys(api),function(){for(const k of Object.keys(api))this.setExport(k,api[k]);},{context:ctx});
  }
  const pm=await module(resolve('demo/js/common_4dgs_compositing_policy.js'));await pm.link(link);await pm.evaluate();
  const nm=cache.get(normalPath).namespace;
  nm.activateNormalAdoptedScene(raw,normalCandidate.normal_scope.profile,input);
  const a=adopted.adoptedFrameArguments(raw),c=adopted.getAdoptedBundleContext(raw);
  normalRuntime={policy:pm.namespace,token:nm.normalViewerScope(raw,a.buildConfig,c.camera,0)};
}
const sourceBytes=new TextEncoder().encode('synthetic served-byte binding only');
const candidates={};
for(const mode of ['NEW-OFF','NEW-ON']) {
  const prefix='step122_'+mode.toLowerCase().replace('-','_')+'_000118_v16';
  candidates[mode]={schema:'viewer-aligned-capture-candidate-v1',execution_authorized:false,review_required:true,
    prefix,mode:{schema:'viewer-range-policy-evidence-v1',mode,math_version:'cuda-aligned-composite-v1',executed:false,terminated_pixel_count:null},
    range_input:{fixed:{population:{rows:raw.N}}},input,source_delivery:{'/js/synthetic.js':identity(sourceBytes)}};
}
if(!external) {
await writeFile(join(root,'fixture.json'),JSON.stringify({candidates,input}),{flag:'wx'});
// Same production generator functions; not a prepared real-data candidate.
execFileSync('/home/demo/miniconda3/envs/4dgs310/bin/python',['-B','-c',`
import json,sys
from pathlib import Path
sys.path.insert(0,'tools')
import aligned_range_capture as a
import adopted_range_contract as c
import make_capture_commands as g
p=Path(sys.argv[1]); v=json.loads((p/'fixture.json').read_text())
for mode,candidate in v['candidates'].items():
 d=p/mode.lower(); d.mkdir(); name='step122-'+mode.lower()
 for filename,data in [('candidate.json',c.canonical(candidate)),(name+'-input.json',c.canonical(v['input'])),(name+'-browser.js',a.wrapper(candidate).encode())]:
  with (d/filename).open('xb') as f:f.write(data)
 sys.argv=['make_capture_commands.py',*a.generator_args(mode)]
 command=g.build_commands(g.parse_args())+'\\nexport { genericProductionCaptureResult, genericProductionPngCaptureStatus, genericProductionPngCaptureResult };\\n'
 with (d/(name+'-capture.js')).open('x') as f:f.write(command)
with (p/'run-off-on.js').open('xb') as f:f.write(a.pair_wrapper())
`,root],{cwd:process.cwd(),env:{...process.env,PYTHONDONTWRITEBYTECODE:'1'}});
}

async function runCase(name, fault=null, order=['NEW-OFF','NEW-ON']) {
  const dest=join(root,name);await mkdir(dest);
  let generation=6, scheduled=[], downloads=[], snapshot=null, request=null, mode=null, loads=0;
  let frame=null, captureCount=0, blobDownloads=0, pngCalls=0, pendingBlob=null, candidateReads=0;
  const writes=[];
  const contextRecord=adopted.getAdoptedBundleContext(raw),width=contextRecord.camera.width,height=contextRecord.camera.height;
  const logs=[];
  const debug={
    getNormalAdoptedScene(){return normalCandidate?raw:null;},
    getNormalStartupEvidence(){return normalCandidate&&fault!=='initial-missing'?copy(initialFrame):null;},
    async loadAdoptedBundle(value) {assert.deepEqual(copy(value),input);loads++;return raw;},
    async waitForViewerDebugDataReady(){return {ready:true};},
    getSynchronousCommandStartFence(){return {synchronousReadOnlyFence:true,initialPresentationSnapshot:{latestProductionGeneration:generation}};},
    validateExpectedProductionRuntimeContract(){return {ready:true};},
    async scheduleRender(options) {
      const r=options.metadata?.compositingPolicyRequest;
      if(normalCandidate)assert.equal(r,undefined,'normal wrapper must not inject validation policy');
      mode=normalCandidate?'NEW-ON':r.mode;generation++;
      request={requestIdentity:'cpu-'+name+'-'+generation,requestedAtMs:1,source:options.source,forceProductionUpdate:true};
      scheduled.push(copy(options));
      // Test the actual strict policy with independently constructed fixed-scope
      // metadata. The parsed three-row bundle tests binding separately, not the
      // real 65,536-row runtime or GPU numerical execution.
      const scope={profile:'adopted-b-sh0-bundle-v1',rows:3248901,selection:{view:4,start:524288,count:65536},
        frame:['train',2588,'images/000118_v16.png',16.6],camera:{width:1280,height:720},
        activeSh:[0,0],maximumSh:[3,2],physicalSh:16,external:input.authority};
      const api=normalRuntime?.policy??{resolveCompositingPolicy,compositingExecutionEvidence,compositingFrameFields};
      const policy=api.resolveCompositingPolicy(normalCandidate?{normalScope:normalRuntime.token,requestIdentity:request.requestIdentity}
        :{request:r,adoptedBundle:scope,requestIdentity:request.requestIdentity});
      snapshot=copy(readySnapshot(raw));
      snapshot.compositingPolicy=api.compositingExecutionEvidence(policy,true,mode==='NEW-ON'?1:0,1280*720);
      const id=snapshot.productionIdentity;
      id.productionGeneration=id.presentedGeneration=generation;
      for(const key of ['productionFrameIdentity','presentedFrameIdentity'])
        Object.assign(id[key],{generation},api.compositingFrameFields(policy));
      Object.assign(snapshot.requestIdentity,{schedulerRequestIdentity:request.requestIdentity,sourceRequestIdentity:request.requestIdentity});
      frame={requestIdentity:request.requestIdentity,productionSourceRequestIdentity:request.requestIdentity,
        productionGeneration:generation,compositorGeneration:generation,presentedGeneration:generation,
        productionFrameIdentity:copy(id.productionFrameIdentity),productionFrameCompleted:true,
        compositorOutputGenerated:true,viewerCanvasPresented:true,logicalPresentationSucceeded:true};
      if(fault==='stale') frame.productionGeneration=6;
      const actualField={'actual-camera':'datasetCameraLabel','actual-frame':'datasetFrameNumber',
        'actual-time':'datasetTime','actual-width':'outputWidth','actual-height':'outputHeight',
        'actual-scope':'compositingScopeIdentity'}[fault];
      if(actualField)frame.productionFrameIdentity[actualField]=typeof frame.productionFrameIdentity[actualField]==='number'
        ?frame.productionFrameIdentity[actualField]+1:'wrong-identity';
      return request;
    },
    getInitialProductionPresentationSnapshot(){return {snapshotTakenAtMs:4,latestProductionGeneration:generation,
      latestProductionRequestIdentity:request?.requestIdentity,requestHistory:request?[{...request,status:'completed',frameStartedAtMs:2,frameCompletedAtMs:3}]:[],frameHistory:frame?[frame]:[]};},
    async waitForFinalCanvasPresentationQuiescence(){return {browserVisibleResult:true,quiescenceObservation:{quiescent:true},
      finalSourceRequestIdentity:request.requestIdentity,finalProductionGeneration:generation,
      finalCompositorGeneration:generation,finalPresentedGeneration:generation,finalSourceIdentityKnown:true,finalSourceIdentityMatchesExpected:true,
      policyNeutralPresentationFacts:{presentationFailed:false,webgpuValidationErrorDetected:false,invalidCommandBufferDetected:false,queueSubmitFailureDetected:false}};},
    getLastRenderResult(){return {webgpuProductionFrameDataPathContract:{productionEvaluationInputContract:snapshot}};},
    async captureWebGpuVisibleRecordDryRunDebug(options){captureCount++;return {
      canonicalDiagnosticResult:{schemaVersion:'synthetic-gpu-diagnostic',artifactProvenance:options.artifactProvenance,
        artifactSetIdentity:options.artifactSetIdentity,comparison:{anyMismatch:false}},
      detailedLineageArtifact:{schemaVersion:'synthetic-gpu-lineage',artifactProvenance:options.artifactProvenance,artifactSetIdentity:options.artifactSetIdentity}};},
    async captureGpuCandidateRuntimeSummaryDebug(){return {limitedDrawSummary:{synthetic:true}};},
    async saveCurrentCanvasPng(options){
      pngCalls++;
      if(normalCandidate){
        assert.equal(options.compositingMode,undefined);assert.equal(options.compositingRequestIdentity,undefined);
      } else {
        assert.equal(options.compositingMode,mode);assert.equal(options.compositingRequestIdentity,request.requestIdentity);
      }
      const bytes=external?new Uint8Array(await readFile(join(external.pngInputs,options.name))):new Uint8Array([137,80,78,71]);
      const fid=snapshot.productionIdentity.productionFrameIdentity;
      return {blob:new Blob([bytes],{type:'image/png'}),fileName:options.name,status:'success',
        captureBlobIdentity:{fileName:options.name,sha256:identity(bytes).sha256,sizeBytes:bytes.length,mimeType:'image/png'},
        encodedPngPixelEvidence:{decodeCompleted:true,pixelClassification:'nonblank',width:fault==='png-size'?width-1:width,height,rgbNonzeroPixelCount:1,rgbMax:127},
        outputStats:{rgbNonzeroPixelCount:1,rgbMax:127},outputWidth:width,outputHeight:height,
        productionOutputGeneration:generation,presentedOutputGeneration:generation,capturedOutputGeneration:generation,
        captureFreshnessKnown:true,staleCaptureDetected:fault==='png-stale',captureMatchesPresentedFrame:true,captureMatchesRequestedState:true,
        capturedFrameIdentity:fid,presentedFrameIdentity:fid,requestedStateIdentity:fid,
        orientationEvidence:{savedPngOrientation:'production-texture-top-left-y-down',captureVerticalFlipApplied:false,
          presentationVerticalFlipApplied:false,orientationMismatchClassification:'none'},
        captureAlphaPolicy:'force-opaque-to-match-webgpu-canvas-presentation'};
    },
    async runPopulationAlignedSemanticComparison(){return {decision:'match',productionDeviceUsed:false,productionDeviceAcceptedAsInput:false,
      orchestrationResult:{decision:'match',coverage:{coverageComplete:true,requestedChunkCount:1,completedChunkCount:1,
        processedRecordCount:input.selection.count,uniqueSrcIndexCount:input.selection.count,
        firstSrcIndex:input.selection.start,lastSrcIndex:input.selection.start+input.selection.count-1},
        expectedGenerationDependsOnActual:false,productionCalculationDependsOnDiagnosticReadback:false},
      diagnosticDeviceCleanup:{status:'destroyed',completed:true}};},
    async downloadJsonDebug(value,fileName){
      const bytes=new TextEncoder().encode(JSON.stringify(value,null,2));
      await writeFile(join(dest,fileName),bytes,{flag:'wx'});downloads.push(fileName);
      assert.deepEqual(new Uint8Array(await readFile(join(dest,fileName))),bytes);
      return {fileName,byteLength:bytes.length+(fault==='download'&&!fileName.endsWith('_browser_result.json')?1:0)};
    }
  };
  let initialFrame=null;
  if(normalCandidate){
    // Represents a successful ordinary startup before the capture wrapper.
    // The loader/scheduler/app path itself is separately exercised by normal_on.
    const api=normalRuntime.policy,p=api.resolveCompositingPolicy({normalScope:normalRuntime.token,requestIdentity:'cpu-startup'});
    const i=copy(readySnapshot(raw).productionIdentity);i.productionGeneration=i.presentedGeneration=generation;
    for(const k of ['productionFrameIdentity','presentedFrameIdentity'])Object.assign(i[k],{generation},api.compositingFrameFields(p));
    initialFrame={schema:'adopted-normal-initial-frame-v1',profile:normalCandidate.normal_scope.profile,
      scopeIdentity:normalCandidate.normal_scope.identity,requestIdentity:p.requestIdentity,
      productionIdentity:i,compositingPolicy:api.compositingExecutionEvidence(p,true,0,width*height)};
  }
  const original={schedule:debug.scheduleRender,png:debug.saveCurrentCanvasPng,download:debug.downloadJsonDebug};
  const context=vm.createContext({console:{log:x=>logs.push(x),error:x=>logs.push(x),warn:x=>logs.push(x)},
    TextEncoder,TextDecoder,crypto:webcrypto,Blob,URLSearchParams,performance:{now:()=>4},setTimeout,
    URL:{createObjectURL:blob=> {pendingBlob=blob;return 'blob:cpu';},revokeObjectURL(){}},
    document:{body:{appendChild(){}},createElement:()=>({click(){blobDownloads++;
      if(external)writes.push(pendingBlob.arrayBuffer().then(b=>writeFile(join(dest,this.download),new Uint8Array(b),{flag:'wx'})));
    },remove(){}})},
    window:{gpuViewerDebug:debug,location:{search:'?'+new URLSearchParams(normalCandidate?
      {adoptedProfile:fault==='profile'?'wrong-profile':normalCandidate.normal_scope.profile}:
      {time:contextRecord.frame[3],datasetTime:contextRecord.frame[3],datasetCameraLabel:contextRecord.frame[2],
        datasetFrameNumber:contextRecord.frame[1],fixedCanvasWidth:width,fixedCanvasHeight:height})}},
    fetch:async url=>{
      if(url==='/js/synthetic.js')return {ok:true,arrayBuffer:async()=>sourceBytes.buffer};
      if(external&&url.startsWith('/js/')){const b=await readFile(join(external.sourceRoot,url.slice(1)));return {ok:true,arrayBuffer:async()=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)};}
      const [folder,file]=url.slice(1).split('/');const m=folder==='step122-new-off'?'new-off':'new-on';
      let data=await readFile(normalCandidate?join(generated,file):join(generated,m,file));
      if(normalCandidate&&file==='candidate.json'&&++candidateReads===2&&fault==='candidate')data=new TextEncoder().encode('{}');
      return {ok:true,json:async()=>JSON.parse(data),arrayBuffer:async()=>data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength)};
    }});
  const binding=new vm.SyntheticModule(['getAdoptedBundleContext','adoptedCaptureBinding'],function(){
    this.setExport('getAdoptedBundleContext',adopted.getAdoptedBundleContext);this.setExport('adoptedCaptureBinding',adopted.adoptedCaptureBinding);
  },{context});
  const cache=new Map();
  async function module(path){
    if(cache.has(path))return cache.get(path);
    const captureOverride=normalCandidate&&path.endsWith('-capture.js')?external.captureVariants?.[fault]:null;
    const m=new vm.SourceTextModule(await readFile(captureOverride??path,'utf8'),{context,identifier:path,
      importModuleDynamically:async(spec,parent)=>{const child=await module(resolve(parent.identifier,'..',spec));await child.evaluate();return child;}});
    cache.set(path,m);await m.link(()=>binding);return m;
  }
  const authorizations=normalCandidate?{normal:{execution_authorized:true,mode:'NEW-ON',purpose:'normal-capture',
    candidate_sha256:identity(await readFile(join(generated,'candidate.json'))).sha256}}:
    Object.fromEntries(await Promise.all(['NEW-OFF','NEW-ON'].map(async m=>[m==='NEW-OFF'?'off':'on',{
    execution_authorized:true,mode:m,candidate_sha256:identity(await readFile(join(generated,m.toLowerCase(),'candidate.json'))).sha256}])));
  let results;
  if(normalCandidate){
    const m=await module(join(generated,'step122-normal-on-browser.js'));await m.evaluate();
    results=[JSON.parse(await m.namespace.run(authorizations.normal))];
    await assert.rejects(()=>m.namespace.run(authorizations.normal),/Step21 already started in this tab/);
  } else if(order[0]==='NEW-OFF') {
    const m=await module(join(generated,'run-off-on.js'));await m.evaluate();results=await m.namespace.run(authorizations);
    await assert.rejects(()=>m.namespace.run(authorizations),/Already started/);
  } else {
    results=[];let shared=null;
    for(const mode of order){const m=await module(join(generated,mode.toLowerCase(),'step122-'+mode.toLowerCase()+'-browser.js'));
      await m.evaluate();const r=await m.namespace.run(authorizations[mode==='NEW-ON'?'on':'off'],shared);shared=r.raw;results.push(r.summary);}
  }
  await Promise.all(writes);
  assert.equal(debug.scheduleRender,original.schedule);assert.equal(debug.saveCurrentCanvasPng,original.png);assert.equal(debug.downloadJsonDebug,original.download);
  await writeFile(join(dest,'vm-observation.json'),JSON.stringify({results,scheduled,downloads,loads,captureCount,pngCalls,blobDownloads,
    query:context.window.location.search,logs},null,2),{flag:'wx'});
  if(fault&&normalCandidate){
    assert.equal(results[0].executionChecksPassed,false,JSON.stringify(results));
    assert.equal(blobDownloads,0);assert.equal(loads,0);
    await assert.rejects(readFile(join(dest,normalCandidate.prefix+'_canvas.png')),{code:'ENOENT'});
    const preflight=['initial-missing','profile','candidate','expected-missing','expected-type','expected-dimension','expected-scope','expected-camera'].includes(fault);
    assert.equal(scheduled.length,preflight?0:1);
    assert.equal(pngCalls,['before','png-size','png-stale'].includes(fault)?1:0);
    if(fault==='before'){
      const status=JSON.parse(await readFile(join(dest,normalCandidate.prefix+'_png_capture_status.json')));
      assert.equal(status.captureExpectationContract.freshIdentity.targetIdentity.datasetCameraLabel,null);
      assert.equal(status.captureExpectationContract.freshIdentity.targetIdentity.outputWidth,0);
    }
  }
  else if(fault){assert.equal(results.stopped,true);assert.equal(scheduled.length,1);assert.equal(loads,1);}
  else {
    assert.equal(scheduled.length,normalCandidate?1:2);assert.equal(loads,normalCandidate?0:1);
    assert.equal(downloads.length,normalCandidate?7:14);
    assert.equal(captureCount,normalCandidate?1:2);assert.equal(blobDownloads,normalCandidate?1:2);
    const summaries=Array.isArray(results)?results:[results.off,results.on];
    for(const s of summaries)assert.equal(s.executionChecksPassed,true,JSON.stringify(s));
  }
  return {name,fault,scheduled:scheduled.length,downloads:downloads.length,restored:true};
}
const results=[];
const normalFaults=['initial-missing','profile','candidate','expected-missing','expected-type','expected-dimension','expected-scope','expected-camera',
  'actual-camera','actual-frame','actual-time','actual-width','actual-height','actual-scope','stale','png-size','png-stale'];
if(external?.captureVariants?.before)normalFaults.unshift('before');
const cases=normalCandidate?[['off-on'],...normalFaults.map(f=>[f,f])]:external?[['off-on']]:
  [['off-on'],['on-off',null,['NEW-ON','NEW-OFF']],['stale','stale'],['download','download']];
for(const spec of cases)results.push(await runCase(...spec));
console.log(JSON.stringify({status:'pass',results,boundary:'synthetic GPU observations; actual generated wrapper/capture, policy, parser, binding and saved JSON; no GPU/browser/image parity'},null,2));
