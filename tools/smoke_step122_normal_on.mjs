// CPU-only module graph; synthetic trusted data, real parser/loader/UI/policy/runtime.
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {webcrypto,createHash} from 'node:crypto';
import {bundleFixture} from './step21_adopted_bundle_fixture.mjs';
const f=bundleFixture(),root=resolve('demo/js'),copy=x=>JSON.parse(JSON.stringify(x));
const input={baseUrl:'/synthetic-bundle/',authority:f.authority,selection:f.selection};
const scope={id:'synthetic-fixed',input,rows:3,frame:['train',4,'images/view4.png',.5],
  mathVersion:'cuda-aligned-composite-v1',depthOrder:'positive-depth-ascending-source-index-ascending-v2',
  sortAlgorithm:'bounded-global-bitonic-stage-v2-depth-ascending-source-index'};
const scopeId=createHash('sha256').update(JSON.stringify(scope)).digest('hex');
let reads=0,defaultReads=0,failLoad=false;
const context=vm.createContext({console,crypto:webcrypto,TextEncoder,TextDecoder,URL,URLSearchParams,
  ArrayBuffer,DataView,Float32Array,Float64Array,Uint32Array,Int32Array,Uint8Array,Uint8ClampedArray,
  setTimeout,clearTimeout,structuredClone,performance,
  fetch:async url=>{
    reads++;if(!String(url).includes('/synthetic-bundle/'))defaultReads++;
    if(failLoad)throw new Error('synthetic-load-failure');
    const path=new URL(url,'http://cpu.invalid').pathname.split('/synthetic-bundle/')[1];
    return new Response(f.files.get(path) ?? null,{status:f.files.has(path)?200:404});
  },location:{href:'http://cpu.invalid/'},window:{location:{href:'http://cpu.invalid/',search:''}},document:{baseURI:'http://cpu.invalid/'}});
const modules=new Map();
async function getModule(path){
  if(modules.has(path))return modules.get(path);
  let source=await readFile(path,'utf8');
  if(path===resolve(root,'common_4dgs_normal_adopted_scope.js'))
    source=`export const NORMAL_ADOPTED_SCOPE=${JSON.stringify(scope)};export const NORMAL_SCOPE_IDENTITY=${JSON.stringify(scopeId)};`;
  const m=new vm.SourceTextModule(source,{context,identifier:path});modules.set(path,m);return m;
}
async function linker(name,ref){
  if(name==='three')return getModule(resolve('build/demo/lib/three.module.js'));
  if(name.startsWith('node:')) {const api=await import(name);return new vm.SyntheticModule(Object.keys(api),function(){
    for(const key of Object.keys(api))this.setExport(key,api[key]);},{context});}
  return getModule(resolve(dirname(ref.identifier),name));
}
async function load(name){const m=await getModule(resolve(root,name));if(m.status==='unlinked')await m.link(linker);
  if(m.status==='linked')await m.evaluate();return m.namespace;}
const query=await load('viewer_query_state.js'),normal=await load('common_4dgs_normal_on.js');
const file=await load('viewer_file_io.js'),parser=await load('splat4d_parser_v2.js');
const adopted=await load('common_4dgs_adopted_bundle.js'),policy=await load('common_4dgs_compositing_policy.js');
const projection=await load('common_4dgs_projection_contracts.js');
const frame=await load('webgpu_presentation_capture_orientation_contract.js');
assert.equal(query.parseViewerQueryState('').adoptedProfile,null);
const q=query.parseViewerQueryState('?adoptedProfile=synthetic-fixed');
assert.equal(q.fixedCanvasWidth,1280);assert.equal(q.webgpuBackendMode,'webgpu-exclusive');
for(const s of ['?adoptedProfile=unknown','?adoptedProfile=synthetic-fixed&time=9',
  '?adoptedProfile=synthetic-fixed&adoptedProfile=synthetic-fixed','?adoptedProfile=synthetic-fixed&authority=anything'])
  assert.throws(()=>query.parseViewerQueryState(s));
const ui=Object.fromEntries(['timeSlider','timeDurationSlider','splatScaleSlider','sigmaScaleSlider','prefilterVarSlider',
  'renderScaleSlider','strideSlider','useSHCheck','bgGraySlider'].map(k=>[k,{}]));
let raw,published=0,firstPolicy;
const io=file.createViewerFileIO({ui,parseArrayBuffer:parser.parseSplat4DV2,
  onSceneLoaded:async r=>{raw=r;published++;adopted.applyAdoptedUiDefaults(r,ui);
    const a=adopted.adoptedFrameArguments(r);
    const token=normal.normalViewerScope(r,a.buildConfig,{width:1280,height:720},0);
    firstPolicy=policy.resolveCompositingPolicy({normalScope:token,requestIdentity:'first-normal'});
  },scheduleRender:()=>{throw new Error('unexpected secondary load');}});
await io.loadStartup(q.adoptedProfile);
assert.equal(defaultReads,0);assert.equal(published,1);assert.equal(firstPolicy.purpose,'normal');
assert.equal(firstPolicy.appliedMode,'NEW-ON');assert.equal(firstPolicy.normalOnAccepted,false);
const a=adopted.adoptedFrameArguments(raw),p=projection.buildWebGpuProjectionContract({camera:{},
  screenSpaceCamera:a.screenSpaceCamera,renderW:1280,renderH:720,sx:1,sy:1});
const args={raw,...a,projectionData:p.data,backgroundGray:0};
const token=normal.normalFrameScope(args);
assert.equal(normal.normalCacheMatches({...firstPolicy,executed:true},token),true);
assert.equal(normal.normalCacheMatches({...firstPolicy,executed:true,purpose:'validation-only'},token),false);
assert.equal(normal.normalCacheMatches({...firstPolicy,executed:true},token,false),false);
assert.equal(normal.normalCacheMatches({...firstPolicy,executed:true,appliedMode:'NEW-OFF'},token),false);
assert.throws(()=>policy.resolveCompositingPolicy({normalScope:{...token},requestIdentity:'forged'}));
assert.throws(()=>policy.resolveCompositingPolicy({normalScope:token,request:{},requestIdentity:'invalid'}));
assert.equal(policy.resolveCompositingPolicy().appliedMode,'NEW-OFF');
for(const [name,change] of Object.entries({time:x=>x.buildConfig.timestamp++,scale:x=>x.buildConfig.scalingModifier=2,
  range:x=>x.productionResidentSelectionRequest.rangeStart++,viewport:x=>x.canvasWidth--,
  background:x=>x.backgroundGray=1,projection:x=>x.projectionData[3]++})) {
  const bad={...args,buildConfig:{...args.buildConfig},productionResidentSelectionRequest:{...args.productionResidentSelectionRequest},
    projectionData:new Float32Array(args.projectionData)};change(bad);
  assert.throws(()=>normal.normalFrameScope(bad),undefined,name);
}
const oldSh=raw.activeShDegree;raw.activeShDegree=2;assert.throws(()=>normal.normalFrameScope(args));raw.activeShDegree=oldSh;
const oldHash=raw.assetSha256;raw.assetSha256='0'.repeat(64);assert.throws(()=>normal.normalFrameScope(args));raw.assetSha256=oldHash;
for(const key of ['mathVersion','sortAlgorithm','depthOrder']){
  const old=scope[key]; // VM data is separate: mutate only test trusted module, never a production override.
  const data=(await load('common_4dgs_normal_adopted_scope.js')).NORMAL_ADOPTED_SCOPE;
  data[key]='invalid';assert.throws(()=>normal.normalFrameScope(args));data[key]=old;
}
const fields=policy.compositingFrameFields(firstPolicy),identity={generation:7,...fields};
assert.equal(frame.compareWebGpuPresentationFrameIdentity(identity,{...identity,compositingPurpose:'validation-only'},
  {requiredKeys:['generation']}).matches,false);
assert.equal(frame.compareWebGpuPresentationFrameIdentity(identity,{generation:7},
  {requiredKeys:['generation']}).matches,false);
const snapshot={status:'ready',compositingPolicy:{...firstPolicy,executed:true},productionIdentity:{
  productionGeneration:7,presentedGeneration:7,productionFrameIdentity:identity,presentedFrameIdentity:{...identity}}};
assert.equal(normal.normalCaptureFields(token,snapshot,'first-normal').compositingPurpose,'normal');
for(const options of [{compositingMode:'NEW-OFF'},{compositingPurpose:'validation-only'}])
  assert.throws(()=>normal.normalCaptureFields(token,snapshot,'first-normal',options));
assert.throws(()=>normal.normalCaptureFields(token,snapshot,'stale'));
snapshot.productionIdentity.presentedGeneration=6;assert.throws(()=>normal.normalCaptureFields(token,snapshot,'first-normal'));
failLoad=true;await assert.rejects(io.loadStartup(q.adoptedProfile),/synthetic-load-failure/);
assert.equal(published,1);assert.equal(defaultReads,0);failLoad=false;
const concurrent=[io.loadStartup(q.adoptedProfile),io.loadStartup(q.adoptedProfile)];
const results=await Promise.allSettled(concurrent);
assert.equal(results[0].status,'rejected');assert.equal(results[1].status,'fulfilled');assert.equal(published,2);
// Actual native runtime graph, with the existing canonical-zero GPU fixture only.
let fixture=await readFile(resolve('tools/smoke_step122_runtime_connection.mjs'),'utf8');
fixture=fixture.slice(0,fixture.indexOf('const camera=new PerspectiveCamera'))+'\nexport {gpu,requests,shaders,buffers};';
const gpuModule=new vm.SourceTextModule(fixture,{context,identifier:resolve('tools/smoke_step122_runtime_connection.mjs')});
await gpuModule.link(async(name,ref)=>{
  if(name.startsWith('node:')){const api=await import(name);return new vm.SyntheticModule(Object.keys(api),function(){
    for(const key of Object.keys(api))this.setExport(key,api[key]);},{context});}
  return linker(name,ref);
});await gpuModule.evaluate();
gpuModule.namespace.gpu.limits.maxStorageBufferBindingSize=128*1024**2;
gpuModule.namespace.gpu.limits.maxBufferSize=128*1024**2;
const runtime=await load('webgpu_production_frame_data_path.js');
const actual=await runtime.runNativeWebGpuProductionFrameDataPath({raw,...adopted.adoptedFrameArguments(raw),camera:{},
  viewerCanvasState:{backgroundGray:0,schedulerFrameState:{requestIdentity:'normal-native'}},
  metadata:{deterministicState:adopted.adoptedDeterministicState(raw)}});
const applied=actual.webgpuProductionFrameDataPathContract.productionEvaluationInputContract.compositingPolicy;
assert.equal(actual.status,'ok');assert.equal(applied.purpose,'normal');assert.equal(applied.appliedMode,'NEW-ON');
assert.equal(applied.executed,true);assert.equal(applied.terminatedPixelCount,0);
assert.ok(gpuModule.namespace.shaders.some(s=>s.code.includes('TERMINATION_ENABLED: bool = true')));
const app=await readFile(resolve(root,'viewer_app_gpu.js'),'utf8');
assert.ok(app.includes('fileIO.loadStartup(deterministicQueryState.adoptedProfile)'));
assert.ok(!app.includes('fileIO.loadDefaultScene()'));
assert.ok(app.includes('normalHeartbeatMatches && calledFromSchedulerFrameLoop'));
assert.ok(app.includes('isCurrentScene:() => raw === frameRaw'));
console.log(JSON.stringify({status:'pass',queryLoaderUiPolicyRuntime:true,reads,defaultReads,
  sceneCompetition:true,normalCacheAndCapture:true,syntheticZeroReference:true,browserExecuted:false,shaderExecuted:false}));

// Separately selectable application-boundary negative case. Execute the exact
// saved app function; replace only browser/GPU I/O, never the admission checks.
if(process.argv.includes('--before-app')) {
  const before=await readFile(process.argv[process.argv.indexOf('--before-app')+1],'utf8');
  const source=before.slice(before.indexOf('async function saveCurrentCanvasPng('),before.indexOf('async function renderCurrentFrame('));
  snapshot.productionIdentity.presentedGeneration=7;
  const reached=[];
  Object.assign(context,{
    currentNormalScope:()=>token,normalCaptureFields:normal.normalCaptureFields,
    currentNormalPublication:()=>true,normalProduction:{requestIdentity:'first-normal'},
    latestRenderResult:{webgpuProductionFrameDataPathContract:{productionEvaluationInputContract:snapshot}},
    latestViewerConnectedSchedulerRuntimeState:{schedulerFrameState:{requestIdentity:'failed-new-request'}},
    scheduler:{state:{lastRenderFailed:true,lastRenderError:{message:'synthetic request failure'},
      lastScheduledRequest:{requestIdentity:'failed-new-request'}}},
    COMPOSITING_MATH_VERSION:'cuda-aligned-composite-v1',canvas:{width:1280,height:720},
    sanitizeSnapshotFileName:x=>x,buildDeterministicStateSummary:()=>({}),
    buildRenderResultInspectionSummary:()=>({}),buildCanvasSizeSummary:()=>({}),
    captureCachedWebGpuTileCompositorOutputPng:async options=>{
      reached.push({boundary:'cached-PNG-I/O',requested:copy(options.requestedStateIdentity)});return {status:'success'};},
    sampleCurrentFramebufferStats:()=>{reached.push({boundary:'default-framebuffer-I/O'});throw new Error('CPU I/O boundary');}
  });
  vm.runInContext(source,context);
  const failedRequestCapture=await context.saveCurrentCanvasPng({captureSource:'last-valid-webgpu-tile-compositor-output',download:false});
  await assert.rejects(context.saveCurrentCanvasPng({download:false}),/CPU I\/O boundary/);
  console.log(JSON.stringify({schema:'normal-app-capture-negative-evidence-v1',failedRequestCapture,
    schedulerFailure:true,currentRequest:'failed-new-request',reached,browserExecuted:false,pngGenerated:false}));
  assert.equal(reached.length,2,'known A/B reproduction, not a successful admission test');
  snapshot.compositingPolicy={...snapshot.compositingPolicy,schema:'adopted-range-compositing-policy-v1',
    purpose:'validation-only',normalApplicationAuthorized:undefined};
  await assert.rejects(context.saveCurrentCanvasPng({captureSource:'last-valid-webgpu-tile-compositor-output',download:false}),
    /normal-on:capture-stale/,'known C: valid v1 snapshot rejected solely by normal-scene branch');
  console.log(JSON.stringify({knownPreFixABC:true,unexpectedIO:2,validationV1Rejected:true}));
} else {
  // Actual app save + policy-preparation + scheduler observers, with only the
  // presentation/GPU/DOM boundary replaced. Keep scheduler state transitions real.
  const admissionModule=await load('viewer_capture_admission.js');
  const schedulerModule=await load('viewer_render_scheduler.js');
  const raf=[],ioEvents=[],downloads=[];
  let generation=0,loadGeneration=1,cacheFailure=false,duringCapture=null,failFrame=false;
  const fixedScope={profile:'adopted-b-sh0-bundle-v1',rows:3248901,
    selection:{view:4,start:524288,count:65536},frame:['train',2588,'images/000118_v16.png',16.6],
    camera:{width:1280,height:720},activeSh:[0,0],maximumSh:[3,2],physicalSh:16,external:f.authority};
  const validation=mode=>({schema:policy.COMPOSITING_POLICY_VERSION,mathVersion:policy.COMPOSITING_MATH_VERSION,
    authority:f.authority,mode,authorization:{schema:'adopted-range-execution-authorization-v1',
      execution_authorized:true,purpose:'validation-only',mode,candidate_sha256:scopeId}});
  Object.assign(context,{
    raw:token.raw, isNormalAdoptedScene:normal.isNormalAdoptedScene,
    currentNormalScope:()=>context.raw===token.raw?token:null,
    captureAdmission:admissionModule.createCaptureAdmission(),fileIO:{getLoadGeneration:()=>loadGeneration},
    resolveCompositingPolicy:policy.resolveCompositingPolicy,getAdoptedBundleContext:()=>fixedScope,
    createRenderScheduler:schedulerModule.createRenderScheduler,requestAnimationFrame:fn=>raf.push(fn),
    initialProductionPresentationRecorder:{recordScheduleRequest(){},recordFrameStarted(){},recordFrameCompleted(){},recordFrameError(){}},
    tokenRef:{value:0},playback:null,deterministicQueryState:{},buildSchedulerFramePresentationBoundaryContract:()=>null,
    getNowMs:()=>0,latestRenderResult:null,latestViewerConnectedSchedulerRuntimeState:null,
    COMPOSITING_MATH_VERSION:policy.COMPOSITING_MATH_VERSION,canvas:{width:1280,height:720},
    sanitizeSnapshotFileName:x=>x,buildDeterministicStateSummary:()=>({}),
    buildRenderResultInspectionSummary:()=>({}),buildCanvasSizeSummary:()=>({}),
    downloadBlob:(blob,name)=>downloads.push(name),
    captureCachedWebGpuTileCompositorOutputPng:async options=>{
      ioEvents.push('cached'); assert.equal(options.download,false);
      const s=context.latestRenderResult.webgpuProductionFrameDataPathContract.productionEvaluationInputContract;
      const i=copy(s.productionIdentity);
      if(duringCapture)await duringCapture();
      return {status:cacheFailure?'unavailable':'success',blob:{},fileName:options.name,captureFreshnessKnown:true,
        capturedOutputGeneration:i.productionGeneration,presentedOutputGeneration:i.presentedGeneration,
        capturedFrameIdentity:i.productionFrameIdentity,presentedFrameIdentity:i.presentedFrameIdentity};
    },
    sampleCurrentFramebufferStats:()=>{ioEvents.push('framebuffer');return {};},captureBlobFromCanvas:async()=>({})
  });
  const prepare=app.slice(app.indexOf('    const normalToken = currentNormalScope();',app.indexOf('async function renderCurrentFrame(')),
    app.indexOf('    const cachedPolicy =',app.indexOf('async function renderCurrentFrame(')));
  context.renderCurrentFrame=async ({schedulerFrameState})=>{
    context.schedulerFrameState=schedulerFrameState;
    const actualPolicy=vm.runInContext('(function(){'+prepare+';return requestedPolicy;})()',context);
    if(failFrame)throw new Error('synthetic-content-failure');
    const previous=context.latestRenderResult?.webgpuProductionFrameDataPathContract?.productionEvaluationInputContract;
    const old=previous?.compositingPolicy;
    if(schedulerFrameState.requestSource!=='viewer-loop-continuation' || !old ||
        old.purpose!==actualPolicy.purpose || old.appliedMode!==actualPolicy.appliedMode){
      generation++;
      const fields=policy.compositingFrameFields(actualPolicy),fid={generation,...fields};
      context.latestRenderResult={webgpuProductionFrameDataPathContract:{productionEvaluationInputContract:{
        status:'ready',compositingPolicy:policy.compositingExecutionEvidence(actualPolicy,true,0,1280*720),
        productionIdentity:{productionGeneration:generation,presentedGeneration:generation,
          productionFrameIdentity:fid,presentedFrameIdentity:{...fid}}}}};
    }
    return context.latestRenderResult;
  };
  vm.runInContext(app.slice(app.indexOf('const scheduler = createRenderScheduler({'),app.indexOf('\nplayback = createViewerPlayback({'))+
    '\nglobalThis.appScheduler=scheduler;',context);
  vm.runInContext(app.slice(app.indexOf('async function saveCurrentCanvasPng('),app.indexOf('async function renderCurrentFrame(')),context);
  const step=async()=>{assert.equal(raf.length,1);await raf.shift()();};
  const schedule=async(metadata=null,source='cpu-content')=>context.appScheduler.scheduleRender({source,
    forceProductionUpdate:source!=='viewer-loop-continuation',metadata});
  const fresh=async metadata=>{await schedule(metadata);await step();};
  const reject=async(options={})=>{
    const before=ioEvents.length;
    await assert.rejects(context.saveCurrentCanvasPng({...options,download:false}),/capture-admission|normal-on/);
    assert.equal(ioEvents.length,before,'rejection must precede PNG I/O');
  };
  await reject();await fresh();
  await context.saveCurrentCanvasPng();assert.equal(downloads.length,1);
  await context.saveCurrentCanvasPng({captureSource:'last-valid-webgpu-tile-compositor-output',download:false});
  await schedule();await reject();await reject({captureSource:'last-valid-webgpu-tile-compositor-output'});
  failFrame=true;await step();failFrame=false;
  await reject({captureSource:'last-valid-webgpu-tile-compositor-output'}); // A independent
  await reject(); // B independent
  await schedule(null,'viewer-loop-continuation');await step();
  assert.equal(context.appScheduler.state.lastRenderFailed,false);await reject();
  await fresh();await context.saveCurrentCanvasPng({download:false});
  const beforeHeartbeat=generation;
  await schedule(null,'viewer-loop-continuation');await step();
  assert.equal(generation,beforeHeartbeat);await context.saveCurrentCanvasPng({download:false});
  for(const options of [{source:'viewer-canvas-current-default-framebuffer'},
    {captureSource:'last-valid-webgpu-tile-compositor-output',source:'canvas'},
    {useLastValidWebGpuTileCompositorOutput:false},{compositingMode:'NEW-OFF'},
    {compositingPurpose:'validation-only'},{compositingRequestIdentity:'stale'}])await reject(options);
  cacheFailure=true;const beforeFallback=ioEvents.length;
  await assert.rejects(context.saveCurrentCanvasPng({fallbackToCanvasOnCaptureFailure:true}),/capture-not-current/);
  assert.equal(ioEvents.length,beforeFallback+1);assert.equal(ioEvents.at(-1),'cached');cacheFailure=false;
  const i=context.latestRenderResult.webgpuProductionFrameDataPathContract.productionEvaluationInputContract.productionIdentity;
  i.presentedGeneration--;await reject();i.presentedGeneration++;
  loadGeneration++;await reject();loadGeneration--;
  context.raw={};const savedRaw=context.raw;context.raw=token.raw;
  duringCapture=async()=>{context.raw=savedRaw;loadGeneration++;};const d=downloads.length;
  await assert.rejects(context.saveCurrentCanvasPng(),/changed-during-capture/);
  assert.equal(downloads.length,d);duringCapture=null;context.raw=token.raw;await reject();await fresh();
  for(const mode of ['NEW-OFF','NEW-ON']){
    await fresh({compositingPolicyRequest:validation(mode)});
    await context.saveCurrentCanvasPng({download:false,compositingMode:mode,compositingPurpose:'validation-only'});
    await reject({compositingPurpose:'normal'});
    await schedule();await reject({compositingPurpose:'validation-only',compositingMode:mode});
    await step();await context.saveCurrentCanvasPng({download:false,compositingPurpose:'normal'});
  }
  await fresh({compositingPolicyRequest:{}});await reject();await fresh();
  duringCapture=async()=>{await schedule();};
  await assert.rejects(context.saveCurrentCanvasPng(),/changed-during-capture/);
  duringCapture=null;await step();
  context.raw={};context.captureAdmission.reset();
  await context.saveCurrentCanvasPng({download:false});assert.equal(ioEvents.at(-1),'framebuffer');
  // Legacy cached caller retains explicit OLD/OFF semantics, without normal authority.
  context.latestViewerConnectedSchedulerRuntimeState={schedulerFrameState:{requestIdentity:'legacy'}};
  await context.saveCurrentCanvasPng({download:false,captureSource:'last-valid-webgpu-tile-compositor-output',compositingMode:'NEW-OFF'});
  console.log(JSON.stringify({status:'pass',applicationBoundary:true,independentABRejections:true,
    normalValidationBothDirections:true,pendingFailureHeartbeatRecovery:true,sceneAwaitAndStale:true,
    sourceConflictAndNoFallback:true,legacy:true,browserExecuted:false,pngIO:'mocked-only'}));
}
