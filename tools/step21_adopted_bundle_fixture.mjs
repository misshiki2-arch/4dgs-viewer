import { createHash } from 'node:crypto';
import { buildProductionEvaluationInputContract, buildProductionResidentSelectionContract,
  buildProductionResidentWorksetContract } from '../demo/js/common_4dgs_production_frame_data_contracts.js';
import { adoptedFrameArguments, adoptedDeterministicState, getAdoptedBundleContext } from '../demo/js/common_4dgs_adopted_bundle.js';
import { buildWebGpuProjectionContract } from '../demo/js/common_4dgs_projection_contracts.js';

export const identity = (bytes) => ({ size: bytes.byteLength, sha256: createHash('sha256').update(new Uint8Array(bytes)).digest('hex') });
const encode = (value) => new TextEncoder().encode(JSON.stringify(value)).buffer;
export function spl4Fixture(N = 3) {
  const bytes = new ArrayBuffer(128 + N * 260), d = new DataView(bytes);
  new Uint8Array(bytes).set(new TextEncoder().encode('SPL4'));
  d.setUint32(4, 2, true); d.setUint32(8, N, true);
  [20,21,22,23].forEach((i) => d.setUint8(i, 1));
  [3,4,4,3,3,45,1,1,1].forEach((n, i) => d.setUint32(24 + i*4, n, true));
  const values = new Float32Array(bytes, 128);
  for (let row = 0; row < N; row++) {
    const s = row*65;
    values.set([0,0,row, 1,0,0,0, 1,0,0,0, Math.log(2),Math.log(3),Math.log(4), 4,-4,0.5],s);
    values.fill(0.25, s+17, s+62); // deliberately nonzero inactive coefficients
    values[s+62] = 0.25; values[s+63] = 0.5; values[s+64] = Math.log(2);
  }
  return bytes;
}

// A complete small publication with independently hashed bytes, not ready flags.
export function bundleFixture(mutator = () => {}, N = 3) {
  const files = new Map();
  const put = (path, value) => { const b = value instanceof ArrayBuffer ? value : encode(value); files.set(path,b); return identity(b); };
  const spl4 = put('bundle/population.splat4d', spl4Fixture(N));
  const population = { schema:'adopted-population-v1', rows:N, header_bytes:128, stride:260, physical_sh:16,
    active_sh:[0,0], maximum_sh_labels:[3,2], identity:spl4,
    mapping:'sourceIndex = assetRecordIndex = CUDA localIndex', quaternion:'raw wxyz', scales:'raw log', opacity:'raw logit', sh:'raw coefficient-major RGB' };
  const unit = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  const refs = Array.from({length:14},(_,v)=>{
    const frame = ['train',v,`images/view${v}.png`,0.5];
    const values = { width:1280,height:720,fx:800,fy:800,cx:640,cy:360,timestamp:0.5,
      world_view_transform:unit,full_proj_transform:unit,projection_matrix:unit,camera_center:[0,0,0], key:frame.slice(0,3) };
    return { schema:'adopted-view-runtime-v1', invocation:v, frame,
      camera:{matrix_layout:'row-major storage of transposed transforms',time_divisor:1,time_duration:33,values},
      native:{call_count:1,population_rows:N,tensors:{19:{shape:[N,16,3]}},
        scalars:{10:1,17:720,18:1280,20:0,21:0,23:0.5,24:33},matrices:{13:unit,14:unit,22:[0,0,0]}} };
  });
  const checkpoint = identity(encode('checkpoint identity only; never loaded'));
  const selected = {marker:checkpoint,metadata:checkpoint,state:checkpoint,representatives:refs.map(r=>r.frame)};
  const manifest = {schema:'adopted-b-reference-bundle-v1',version:1,population,references:refs,
    input:{selected,profile:{renderer:{scaling_modifier:1,temporal_prefilter:'disabled',compute_cov3D_python:false,
      convert_SHs_python:false,env_map_res:0,override_color:null}}}};
  const verification = {schema:'adopted-reference-verification-v1',status:'verified_requires_review',gate_c_passed:false,
    all_rows:true,raw14_exact:true,gt14_raw_exact:true,png14_exact:true,separate_cpu_process:true,selected};
  mutator({manifest,population,verification,files});
  const members = {'population.splat4d':spl4};
  for (const [name,obj] of Object.entries({'manifest.json':manifest,'population.json':population,'verification.json':verification})) {
    members[name] = put('bundle/'+name,obj);
  }
  for (let v=0;v<14;v++) for (const name of ['render.npy','alpha.npy','depth.npy','radii.npy','visibility_filter.npy','gt.npy','render.png','alpha.png','depth.png','gt.png','runtime.json']) {
    const path=`references/${String(v).padStart(2,'0')}/${name}`;
    members[path]=put('bundle/'+path,name==='runtime.json'? refs[v] : new Uint8Array([v,1,2,3]).buffer);
  }
  const index = put('bundle/index.json',{schema:'adopted-bundle-index-v1',members});
  const receipt = {schema:'adopted-reference-receipt-v1',status:'verified_requires_review',gate_c_passed:false,index,verification:members['verification.json']};
  const receiptId=put('receipt.json',receipt);
  const exit={exit_code:0,primary:null,stop_confirmed:true};
  const result=put('result.json',{status:receipt.status,gate_c_passed:false,receipt,
    worker_exit:put('worker-exit.json',exit),reader_exit:put('reader-exit.json',exit)});
  return {files,authority:{result,index,manifest:members['manifest.json'],verification:members['verification.json'],spl4,receipt:receiptId},
    selection:{view:4,start:0,count:N,representatives:[0,N-1]},
    readResource:async(path)=>{if(!files.has(path))throw new Error('fixture-missing:'+path);return files.get(path).slice(0);} };
}

// Uses the real snapshot builder. Only external presentation/device observations are fixtures.
export function readySnapshot(raw) {
  const c=getAdoptedBundleContext(raw), a=adoptedFrameArguments(raw), d=adoptedDeterministicState(raw);
  const projection=buildWebGpuProjectionContract({camera:{},screenSpaceCamera:a.screenSpaceCamera,
    renderW:a.canvasWidth,renderH:a.canvasHeight,sx:1,sy:1});
  const selection=buildProductionResidentSelectionContract({request:a.productionResidentSelectionRequest,
    sceneRecordCount:raw.N,resourceCapacityRecords:c.selection.count});
  const workset=buildProductionResidentWorksetContract({resourceIdentity:'fixture-workset',sceneResourceIdentity:'fixture-scene',sceneRecordCount:raw.N,
    residentStart:c.selection.start,residentRecordCount:c.selection.count,resourceCapacityRecords:c.selection.count,
    residentSelectionContract:selection,selectionPolicy:selection.selectionPolicy,diagnosticMaxRecordsUsed:false,
    diagnosticCandidateSourceUsed:false,nonResidentRecordsExplicit:true,overflowPolicy:'fail-closed-before-compositor-promotion'});
  const frame={generation:7,frameHash:'fixture-frame',datasetCameraLabel:c.frame[2],datasetFrameNumber:c.frame[1],datasetTime:c.frame[3],
    referenceCameraLabel:c.frame[2],outputWidth:a.canvasWidth,outputHeight:a.canvasHeight};
  return buildProductionEvaluationInputContract({adoptedBundle:c,
    assetIdentity:{sceneResourceIdentity:'fixture-scene',assetSha256:raw.assetSha256,assetSizeBytes:raw.assetSizeBytes,assetSourceKind:raw.assetSourceKind,
      assetFormat:'SPL4-v2',formatVersion:2,recordCount:raw.N,dimensions:{xyz:3,rotation:4,rotationR:4,scaleXyz:3,fDc:3,fRest:45,opacity:1,time:1,scaleTime:1}},
    worksetContract:workset,buildConfig:a.buildConfig,effectiveRenderScale:1,renderWidth:a.canvasWidth,renderHeight:a.canvasHeight,
    canvasWidth:a.canvasWidth,canvasHeight:a.canvasHeight,projectionData:projection.data,projectionSummary:projection.summary,
    cameraIdentity:{cameraLabel:c.frame[2],referenceCameraLabel:c.frame[2],datasetFrameNumber:c.frame[1],datasetViewId:c.selection.view,
      cameraSource:d.cameraSource,datasetViewMatrixMode:d.datasetViewMatrixMode,fixedReferenceCameraActivationMode:d.fixedReferenceCameraActivationMode,
      cameraControlContract:d.cameraControlContract,cameraOrientationPolicy:d.cameraOrientationPolicy},
    timeIdentity:{requestedDatasetTime:c.frame[3],requestedViewerTime:c.frame[3],actualAppliedTimestamp:c.frame[3]},
    requestIdentity:{schedulerRequestIdentity:'fixture-request',sourceRequestIdentity:'fixture-request',schedulerRequestSource:'fixture-capture',schedulerFrameIndex:1},
    productionIdentity:{productionGeneration:7,presentedGeneration:7,productionFrameIdentity:frame,presentedFrameIdentity:{...frame}},
    orientationIdentity:{schemaVersion:'phase3-webgpu-presentation-capture-orientation-v1',productionTextureOrigin:'texture-memory-top-left',
      productionTextureYAxisDirection:'down',presentationUvTransform:'fullscreen-framebuffer-top-left-to-texture-top-left',presentationVerticalFlipApplied:false,
      captureReadbackRowOrder:'copyTextureToBuffer-texture-memory-top-to-bottom',pngEncoderRowOrder:'canvas-image-data-top-to-bottom',
      canonicalPresentationOrientation:'production-texture-top-left-y-down',savedPngOrientation:'production-texture-top-left-y-down',
      captureVerticalFlipApplied:false,orientationMismatchClassification:'none'}});
}
