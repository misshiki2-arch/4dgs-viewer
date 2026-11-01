// Small publication fixture for the Python generator -> VM -> Python CLI test.
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve,join,dirname} from 'node:path';
import {webcrypto} from 'node:crypto';
import {bundleFixture,identity,readySnapshot} from './step21_adopted_bundle_fixture.mjs';
import {loadVerifiedAdoptedBundle} from '../demo/js/common_4dgs_adopted_bundle.js';
import {parseSplat4DV2} from '../demo/js/splat4d_parser_v2.js';
globalThis.crypto??=webcrypto;
const request=JSON.parse(await readFile(process.argv[2],'utf8')),root=resolve(request.output);
assertPath(root);await mkdir(root);
function assertPath(p){if(!['step122-numerical-alignment-fix1','step122-sort-alignment-fix1','step122-normal-on-implement1','step122-normal-on-fix1','step122-normal-on-fix2','step122-normal-on-png-fix1','step122-normal-on-png-fix2'].some(task=>
  p.startsWith('/home/demo/work/reports/webgpu-4dgs-viewer/phase3/step122/'+task+'/')))throw new Error('fixture-destination');}
const f=bundleFixture(()=>{},request.fixed.population.rows),files=f.files;
const encode=x=>new TextEncoder().encode(JSON.stringify(x)).buffer;
const put=(name,bytes)=>{files.set(name,bytes);return identity(bytes);};
const json=name=>JSON.parse(new TextDecoder().decode(files.get(name)));
const bytes=await readFile(request.spl4),spl4=put('bundle/population.splat4d',bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
const manifest=json('bundle/manifest.json'),population=json('bundle/population.json'),verification=json('bundle/verification.json');
population.identity=spl4;manifest.population=population;
for(const ref of manifest.references){
  ref.frame=request.fixed.frame;ref.camera.values=request.fixed.camera.values;
  ref.camera.time_duration=33;ref.camera.time_divisor=1;
  ref.native.scalars[17]=ref.camera.values.height;ref.native.scalars[18]=ref.camera.values.width;
  ref.native.scalars[23]=ref.frame[3];
  ref.native.matrices[13]=ref.camera.values.world_view_transform;
  ref.native.matrices[14]=ref.camera.values.full_proj_transform;
  ref.native.matrices[22]=ref.camera.values.camera_center;
}
manifest.input.selected.representatives=manifest.references.map(x=>x.frame);
verification.selected=manifest.input.selected;
for(const ref of manifest.references)put('bundle/references/'+String(ref.invocation).padStart(2,'0')+'/runtime.json',encode(ref));
for(const [name,value] of Object.entries({'manifest.json':manifest,'population.json':population,'verification.json':verification}))put('bundle/'+name,encode(value));
const members=Object.fromEntries([...files].filter(([p])=>p.startsWith('bundle/')&&p!=='bundle/index.json').map(([p,b])=>[p.slice(7),identity(b)]));
const index=put('bundle/index.json',encode({schema:'adopted-bundle-index-v1',members}));
const receipt=json('receipt.json');receipt.index=index;receipt.verification=members['verification.json'];
const receiptId=put('receipt.json',encode(receipt));
const result=json('result.json');result.receipt=receipt;const resultId=put('result.json',encode(result));
const authority={result:resultId,index,manifest:members['manifest.json'],verification:members['verification.json'],spl4,receipt:receiptId};
const raw=await loadVerifiedAdoptedBundle({authority,selection:request.fixed.selection,readResource:async p=>files.get(p).slice(0),parseArrayBuffer:parseSplat4DV2});
const projection=readySnapshot(raw).projection;
for(const [name,data] of files){const p=join(root,name);await mkdir(dirname(p),{recursive:true});await writeFile(p,new Uint8Array(data),{flag:'wx'});}
await writeFile(join(root,'fixture-authority.json'),JSON.stringify({authority,projection}),{flag:'wx'});
console.log(JSON.stringify({status:'synthetic-publication-verified',members:files.size}));
