// Independent bounded CPU expectations / saved-result consumer. No GPU calls.
// Explicit future diagnostic execution must be separately authorized.
import { PRODUCTION_DEPTH_ORDER, PRODUCTION_SORT_ALGORITHM } from '../demo/js/common_4dgs_record_contracts.js';
import { PRODUCTION_TILE_EXECUTION_PLAN_WORD as WORD } from '../demo/js/common_4dgs_production_tile_execution_plan_contracts.js';
const need=(ok,why)=>{if(!ok)throw new Error('sort-fixture:'+why);};
const f=Math.fround;
const row=(depth,rgb,alpha)=>[3,3,1,depth,80,0,depth,80,...rgb,alpha];
const input=(records,tiles,chunkSize=1,tileCols=1,tileRows=1)=>({
  schema:'small-compositing-diagnostic-v2',executionAuthorized:false,width:16*tileCols,height:16*tileRows,
  records,tiles,chunkSize,tileCols,tileRows,modes:['NEW-OFF','NEW-ON','NEW-OFF']});

export function sortFixtures() {
  const two=[row(4,[1,0,0],.25),row(1,[0,0,1],.5)];
  const three=[row(3,[1,0,0],.5),row(1,[0,1,0],.25),row(2,[0,0,1],.75)];
  const ties=[row(2,[1,0,0],.5),row(2,[0,1,0],.25),row(2,[0,0,1],.75)];
  return [
    {name:'two-overlap',input:input(two,[[0,1]]),order:[[1,0]],center:[.125,0,.5,.375]},
    {name:'three-padding',input:input(three,[[0,2,1]],2),order:[[1,2,0]],center:[.09375,.25,.5625,.09375]},
    {name:'exact-tie',input:input(ties,[[2,0,1]],2),order:[[0,1,2]],center:[.5,.125,.28125,.09375]},
    {name:'near-not-tie',input:input([row(f(1+2**-23),[1,0,0],.25),two[1]],[[0,1]]),order:[[1,0]],center:[.125,0,.5,.375]},
    {name:'empty',input:input([], [[]]),order:[[]],center:[0,0,0,1]},
    {name:'one',input:input([two[0]],[[0]]),order:[[0]],center:[.25,0,0,.75]},
    {name:'multi-tile-empty-invalid-row',input:input([...three,Array(12).fill(0)],[[2,0,1],[],[1],[0,1]],1,2,2),
      order:[[1,2,0],[],[1],[1,0]],center:[.09375,.25,.5625,.09375]},
    {name:'chunk-64-padding',input:input(Array.from({length:65},(_,i)=>row(65-i,[0,0,.25],1/128)),
      [Array.from({length:65},(_,i)=>i)],64),order:[Array.from({length:65},(_,i)=>64-i)]}
  ];
}

export function expectedSortedFixture(fixture) {
  const {input:s,order}=fixture;
  need(order.length===s.tiles.length,'expected-tiles');
  const table=[],refs=[];let offset=0;
  order.forEach((ids,t)=>{
    need(JSON.stringify([...ids].sort((a,b)=>a-b))===JSON.stringify([...s.tiles[t]].sort((a,b)=>a-b)),'expected-members');
    const padded=ids.length?2**Math.ceil(Math.log2(ids.length)):0;
    table.push(offset,ids.length,padded,84);
    for(const id of ids)refs.push(id,0,f(s.records[id][3]),f(s.records[id][6]));
    for(let j=ids.length;j<padded;j++)refs.push(-1,0,0,0);
    offset+=padded;
  });
  const frames=s.modes.map(mode=>{
    const accumulation=[],done=[],rgba=[];
    for(let y=0;y<s.height;y++)for(let x=0;x<s.width;x++) {
      const ids=order[Math.floor(y/16)*s.tileCols+Math.floor(x/16)];
      let C=[0,0,0],T=1,finished=false;
      // Analytic isolated integer centers; all other weights are exp(-40),
      // far below skip. Not a production renderer or a projection oracle.
      for(const id of ids) {
        const r=s.records[id];
        const alpha=Math.min(f(.99),Math.max(0,f(f(r[11])*(x===3&&y===3?1:f(Math.exp(-40))))));
        if(finished||alpha<f(1/255))continue;
        const next=f(T*f(1-alpha));
        if(mode==='NEW-ON'&&next<f(1e-4)){finished=true;continue;}
        C=C.map((v,c)=>f(v+f(f(f(Math.max(0,r[8+c]))*alpha)*T)));T=next;
      }
      accumulation.push(...C,T);done.push(Number(finished));
      rgba.push(...[...C,f(1-T)].map(v=>Math.round(255*Math.min(1,Math.max(0,v)))));
    }
    const count=order.reduce((n,t)=>n+t.length,0);
    return {mode,accumulation,done,rgba,coverage:count,terminated:done.reduce((a,b)=>a+b,0)};
  });
  if(fixture.center)need(JSON.stringify(frames[0].accumulation.slice((3*s.width+3)*4,(3*s.width+3)*4+4))===JSON.stringify(fixture.center),'analytic-center');
  return {table,refs,frames};
}

export function compareSortedDiagnostic(raw,fixture) {
  const s=fixture.input,e=expectedSortedFixture(fixture);
  need(raw?.schema==='small-compositing-diagnostic-result-v2'&&raw.status==='observed-requires-independent-comparison'&&!raw.failure,'result');
  need(raw.productionDeviceUsed===false&&raw.productionDeviceAcceptedAsInput===false&&raw.cleanupCompleted===true,'ownership');
  need(raw.accepted===false&&raw.normalOnAccepted===false&&raw.comparisonComplete===false,'not-acceptance');
  need(raw.frames.length===s.modes.length,'frame-count');
  let maxError=0;
  raw.frames.forEach((a,j)=>{
    const want=e.frames[j],pixels=s.width*s.height;
    need(a.mode===want.mode&&a.width===s.width&&a.height===s.height&&a.coverageUnit==='tile-reference','frame');
    need(a.orderContract===PRODUCTION_DEPTH_ORDER&&a.sortAlgorithm===PRODUCTION_SORT_ALGORITHM&&a.productionSortSubmitted===true,'sort-identity');
    need(JSON.stringify(a.tileTable)===JSON.stringify(e.table),'table');
    // Empty capacity is one allocation slot but zero logical references.
    need(a.orderedReferences.length===Math.max(4,e.refs.length)&&
      e.refs.every((v,i)=>a.orderedReferences[i]===v),'ordered-references');
    need(a.orderingSummary.length===28&&a.orderingSummary[20]===0&&a.orderingSummary[0]===want.coverage,'sort-validation');
    need(a.executionPlan.length===16&&a.executionPlan[WORD.recordCount]===s.records.length&&
      [WORD.requiredReferenceCount,WORD.scatteredReferenceCount,WORD.sortedReferenceCount,WORD.compositedReferenceCount]
        .every(k=>a.executionPlan[k]===want.coverage),'plan-count');
    need(a.summary.length===44&&[2,3,10,11].every(k=>a.summary[k]===want.coverage)&&a.summary[40]===want.terminated,'coverage');
    need(a.accumulation.length===pixels*4&&a.done.length===pixels&&a.done.every((v,i)=>v===want.done[i]),'state-length-done');
    a.accumulation.forEach((v,i)=>{
      // <=65 terms, RGB<=1: gamma(195)<1.2e-5 for C, gamma(130)<7.8e-6 for T.
      // Absolute margins, only for these exact-center fixtures; no old tolerance changed.
      const error=Math.abs(v-want.accumulation[i]);
      need(Number.isFinite(v)&&error<=(i%4===3?1e-5:2e-5),'C-T');maxError=Math.max(maxError,error);
    });
    need(Number.isInteger(a.bytesPerRow)&&a.bytesPerRow>=s.width*4&&a.bytesPerRow%256===0&&a.rgbaPadded.length===a.bytesPerRow*s.height,'texture');
    for(let y=0;y<s.height;y++)for(let x=0;x<s.width*4;x++) {
      const v=a.rgbaPadded[y*a.bytesPerRow+x],expected=want.rgba[y*s.width*4+x];
      need(Number.isInteger(v)&&v>=0&&v<=255&&Math.abs(v-expected)<=1,'rgba');
      if(expected===0||expected===255)need(v===expected,'rgba-endpoint');
    }
  });
  return {name:fixture.name,status:'fixture-checks-passed-requires-review',frames:raw.frames.length,maxError,
    accepted:false,normalOnAccepted:false};
}
