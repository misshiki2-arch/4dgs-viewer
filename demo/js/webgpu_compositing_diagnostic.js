// Small, explicit diagnostic only. No production device/resource input, no UI
// wiring or automatic invocation. Uses the exact production WGSL, not a copy.
import { buildAlignedCompositorWgsl, executeBoundedProductionTileSort } from './webgpu_bounded_tile_sort_and_compositor.js';
import { PRODUCTION_DEPTH_ORDER, PRODUCTION_SORT_ALGORITHM } from './common_4dgs_record_contracts.js';
import { validateCompositingResources } from './common_4dgs_compositing_policy.js';
import { PRODUCTION_TILE_EXECUTION_PLAN_WORD as WORD,
  PRODUCTION_TILE_EXECUTION_PLAN_STATUS as STATUS,
  PRODUCTION_TILE_EXECUTION_PLAN_MAGIC as MAGIC }
  from './common_4dgs_production_tile_execution_plan_contracts.js';

function check(ok, reason) { if(!ok) throw new Error('compositing-diagnostic:'+reason); }
export function validateCompositingDiagnosticInput(input) {
  const sorted = input?.schema === 'small-compositing-diagnostic-v2';
  check(input && JSON.stringify(Object.keys(input).sort())===JSON.stringify(
    ['schema','executionAuthorized','width','height','records','chunkSize','modes',
      ...(sorted?['tiles','tileCols','tileRows']:[])].sort()),'fields');
  check((sorted||input.schema==='small-compositing-diagnostic-v1') && input.executionAuthorized===true,'authorization');
  for(const value of [input.width,input.height]) check(Number.isSafeInteger(value)&&value>=2&&value<=32,'viewport');
  check(Array.isArray(input.records)&&input.records.length>=(sorted?0:1)&&input.records.length<=128,'record-count');
  check(input.records.every(row=>Array.isArray(row)&&row.length===12&&row.every(x=>Number.isFinite(x)&&Number.isFinite(Math.fround(x)))),'record-layout');
  check(Number.isSafeInteger(input.chunkSize)&&input.chunkSize>0&&input.chunkSize<=64,'chunk-size');
  check(Array.isArray(input.modes)&&input.modes.length>0&&input.modes.length<=4&&
    input.modes.every(x=>x==='NEW-OFF'||x==='NEW-ON'),'mode');
  if(sorted) {
    // Bounded fixture grid, no claim of general resize support. Input tiles are
    // scatter order, never sorted/expected references supplied to the GPU.
    check([input.tileCols,input.tileRows].every(x=>Number.isInteger(x)&&x>=1&&x<=2)&&
      input.width===16*input.tileCols&&input.height===16*input.tileRows,'tile-grid');
    check(Array.isArray(input.tiles)&&input.tiles.length===input.tileCols*input.tileRows,'tiles');
    for(const tile of input.tiles) {
      check(Array.isArray(tile)&&tile.length<=128&&new Set(tile).size===tile.length,'tile-references');
      for(const row of tile) {
        check(Number.isInteger(row)&&row>=0&&row<input.records.length,'reference-row');
        const record=input.records[row];
        check(Math.fround(record[6])>0&&record[2]>0,'referenced-depth-radius');
      }
    }
  }
  return JSON.parse(JSON.stringify(input)); // caller cannot mutate an in-flight frame
}

export async function runCompositingDiagnostic(options={}) {
  check(Object.keys(options).every(k=>['input','requestAdapter'].includes(k)),'options-no-production-resource');
  const {input,requestAdapter=()=>globalThis.navigator?.gpu?.requestAdapter()}=options;
  const spec=validateCompositingDiagnosticInput(input);
  const sorted=spec.schema==='small-compositing-diagnostic-v2';
  let device=null;
  const owned=[];
  const result={schema:sorted?'small-compositing-diagnostic-result-v2':'small-compositing-diagnostic-result-v1',status:'blocked',frames:[],
    productionDeviceUsed:false,productionDeviceAcceptedAsInput:false,accepted:false,
    comparisonComplete:false,normalOnAccepted:false,cleanupCompleted:false};
  try {
    const adapter=await requestAdapter();check(adapter&&typeof adapter.requestDevice==='function','adapter');
    device=await adapter.requestDevice({label:'step122-fresh-compositing-diagnostic',
      requiredLimits:{maxStorageBuffersPerShaderStage:8}});
    check(device,'device');validateCompositingResources(device,spec.width,spec.height);
    device.pushErrorScope('validation');
    const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST;
    const buffer=(data,usage,label)=>{
      const b=device.createBuffer({label,size:Math.max(4,data.byteLength),usage,mappedAtCreation:true});owned.push(b);
      new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));b.unmap();return b;
    };
    const n=spec.records.length,pixels=spec.width*spec.height;
    const rows=buffer(new Float32Array(n?spec.records.flat():Array(12).fill(0)),storage,'diagnostic-records');
    const tiles=sorted?spec.tiles:[spec.records.map((_,i)=>i)];
    const tileCols=sorted?spec.tileCols:1,tileRows=sorted?spec.tileRows:1;
    const tableValues=[],refValues=[];let capacity=0,referenceCount=0;
    for(const tile of tiles) {
      const padded=sorted?(tile.length?2**Math.ceil(Math.log2(tile.length)):0):tile.length;
      tableValues.push(capacity,tile.length,padded,84);
      for(const row of tile)refValues.push(row,0,sorted?spec.records[row][3]:n-row,sorted?spec.records[row][6]:n-row);
      for(let i=tile.length;i<padded;i++)refValues.push(0,0,0,0);
      capacity+=padded;referenceCount+=tile.length;
    }
    const table=buffer(new Float32Array(tableValues),storage,'diagnostic-tile');
    const references=buffer(new Float32Array(refValues.length?refValues:[0,0,0,0]),storage,
      sorted?'diagnostic-unsorted-references':'diagnostic-ordered-slots');
    const texture=device.createTexture({label:'diagnostic-output',size:[spec.width,spec.height],format:'rgba8unorm',
      usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC});owned.push(texture);
    const summary=buffer(new Float32Array(44),storage,'diagnostic-summary');
    const params=buffer(new Float32Array([tiles.length,tileCols,tileRows,Math.max(1,capacity),spec.width,spec.height,spec.width,spec.height]),GPUBufferUsage.UNIFORM,'diagnostic-params');
    const accumulation=buffer(new Float32Array(pixels*4),storage,'diagnostic-accumulation');
    const execution=buffer(new Uint32Array([0,spec.chunkSize,n,118]),GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,'diagnostic-chunk');
    const evidence=buffer(new Uint32Array(11),storage,'diagnostic-evidence');
    const planWords=new Uint32Array(16);
    for(const [key,value] of Object.entries({magic:MAGIC,status:STATUS.ready,recordCount:n,tileCount:tiles.length,
      referenceCapacity:Math.max(1,capacity),requiredReferenceCount:referenceCount,requiredPaddedReferenceCapacity:capacity,
      maxReferencesPerTile:Math.max(0,...tiles.map(t=>t.length)),
      scatteredReferenceCount:referenceCount,sortedReferenceCount:sorted?0:referenceCount,compactOffsetTableReady:1,planIdentity:1}))planWords[WORD[key]]=value;
    const plan=buffer(planWords,storage,'diagnostic-plan');
    const done=buffer(new Uint32Array(pixels),storage,'diagnostic-done');
    const resources=[rows,table,references,texture,summary,params,accumulation,execution,evidence,plan,done];
    const layout=device.createBindGroupLayout({entries:resources.map((_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,
      ...(binding===3?{storageTexture:{access:'write-only',format:'rgba8unorm'}}:
        {buffer:{type:[5,7].includes(binding)?'uniform':binding<3?'read-only-storage':'storage'}})}))});
    const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
    const readbacks=[summary,accumulation,done].map((r,i)=>buffer(new Uint8Array([176,pixels*16,pixels*4][i]),
      GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,'diagnostic-readback-'+i));
    const bytesPerRow=Math.ceil(spec.width*4/256)*256;
    const image=buffer(new Uint8Array(bytesPerRow*spec.height),GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,'diagnostic-image-readback');
    for(const mode of spec.modes) {
      device.queue.writeBuffer(evidence,0,new Uint32Array(11));device.queue.writeBuffer(plan,0,planWords);
      let sort=null,sortReadbacks=[];
      if(sorted) {
        sort=executeBoundedProductionTileSort({device,resources:{referenceCapacity:Math.max(1,capacity),
          tileCount:tiles.length,tileTableBuffer:table,referenceListBuffer:references,executionPlanBuffer:plan}});
        owned.push(sort.orderedReferenceBuffer,sort.orderingSummaryBuffer,sort.orderingParamsBuffer,sort.orderingSummaryReadbackBuffer);
        resources[2]=sort.orderedReferenceBuffer;
        sortReadbacks=[buffer(new Uint8Array(sort.orderedReferenceBufferBytes),GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,'diagnostic-order-readback'),
          sort.orderingSummaryReadbackBuffer,
          buffer(new Uint8Array(planWords.byteLength),GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,'diagnostic-plan-readback')];
      }
      const group=device.createBindGroup({layout,entries:resources.map((r,binding)=>({binding,resource:binding===3?r.createView():{buffer:r}}))});
      const shader=device.createShaderModule({label:'same-production-compositing-wgsl',code:buildAlignedCompositorWgsl(mode==='NEW-ON')});
      const pipelines={};
      for(const entryPoint of ['clearProductionAccumulation','compositeReferenceChunk','writeProductionOutput','finalizeSummary'])
        pipelines[entryPoint]=device.createComputePipeline({layout:pipelineLayout,compute:{module:shader,entryPoint}});
      const dispatch=entry=>{
        const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
        pass.setPipeline(pipelines[entry]);pass.setBindGroup(0,group);
        pass.dispatchWorkgroups(...(entry==='finalizeSummary'?[1]:[Math.ceil(spec.width/8),Math.ceil(spec.height/8)]));
        pass.end();device.queue.submit([encoder.finish()]);
      };
      dispatch('clearProductionAccumulation');
      for(let start=0;start<planWords[WORD.maxReferencesPerTile];start+=spec.chunkSize){device.queue.writeBuffer(execution,0,new Uint32Array([start,spec.chunkSize,referenceCount,118]));dispatch('compositeReferenceChunk');}
      dispatch('writeProductionOutput');dispatch('finalizeSummary');
      const encoder=device.createCommandEncoder();
      [summary,accumulation,done].forEach((b,i)=>encoder.copyBufferToBuffer(b,0,readbacks[i],0,[176,pixels*16,pixels*4][i]));
      encoder.copyTextureToBuffer({texture},{buffer:image,bytesPerRow,rowsPerImage:spec.height},{width:spec.width,height:spec.height,depthOrArrayLayers:1});
      if(sorted)[sort.orderedReferenceBuffer,sort.orderingSummaryBuffer,plan].forEach((b,i)=>
        encoder.copyBufferToBuffer(b,0,sortReadbacks[i],0,[sort.orderedReferenceBufferBytes,112,64][i]));
      device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
      const arrays=[];
      for(const [i,b] of [...readbacks,image].entries()){
        await b.mapAsync(GPUMapMode.READ);const bytes=b.getMappedRange().slice(0);b.unmap();
        arrays.push(Array.from(i<2?new Float32Array(bytes):i===2?new Uint32Array(bytes):new Uint8Array(bytes)));
      }
      result.frames.push({mode,summary:arrays[0],accumulation:arrays[1],done:arrays[2],rgbaPadded:arrays[3],bytesPerRow,
        width:spec.width,height:spec.height,coverageUnit:'tile-reference'});
      if(sorted) {
        const observed=[];
        for(const [i,b] of sortReadbacks.entries()) {
          await b.mapAsync(GPUMapMode.READ);const bytes=b.getMappedRange().slice(0);b.unmap();
          observed.push(Array.from(i===0?new Float32Array(bytes):new Uint32Array(bytes)));
        }
        Object.assign(result.frames.at(-1),{orderContract:PRODUCTION_DEPTH_ORDER,sortAlgorithm:PRODUCTION_SORT_ALGORITHM,
          orderedReferences:observed[0],orderingSummary:observed[1],executionPlan:observed[2],
          tileTable:tableValues,productionSortSubmitted:true});
      }
    }
    const error=await device.popErrorScope();check(!error,'gpu-validation');
    result.status='observed-requires-independent-comparison';
  } catch(error) {result.failure={name:error?.name??'Error',message:String(error?.message??error).slice(0,256)};}
  finally {
    // Device remains isolated even on map/dispatch failure; no production cleanup.
    for(const resource of owned.reverse())resource.destroy();
    if(device)device.destroy();result.cleanupCompleted=true;
  }
  return result;
}
