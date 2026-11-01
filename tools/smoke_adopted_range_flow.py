"""Small CPU end-to-end entries. Production data/native/HTTP/browser are not run."""
import copy
from contextlib import ExitStack, contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import adopted_range_contract as c
import compare_adopted_range_capture as compare
import make_adopted_range_capture as capture
import prepare_adopted_range_capture as prepare
import render_adopted_range_reference as rr

c.require(sys.executable==c.PYTHON,'test_python')
c.corrected_paths()
import test_adopted_reference as existing
from formal_training_fixture import CPUCase
import staged_comparison_adopted_input as ai
import staged_comparison_checkpoint as cp
import staged_comparison_support as support
import adopted_reference_runtime as rt
import adopted_reference_bundle as bundle


def cli(main, argv):
    with patch.object(sys,'argv',argv),redirect_stdout(io.StringIO()) as output:
        main()
    return output.getvalue()


def save_json(path,value):
    path.write_bytes(c.canonical(value))


def synthetic_resource_sample(*_):
    # CPU fixture data, NOT measured OS resources or capacity acceptance.
    return dict(host_available=10**15,disk_free=10**15,
                rss_current=1,rss_high_water=1,swap=0)


@contextmanager
def fixture_resource_samples(sample_fn=synthetic_resource_sample):
    """Scope only the existing sampler seam; keep real child/monitor semantics."""
    monitor=support.monitor();launch=monitor.monitored_child
    def child(command,**kwargs):
        kwargs.setdefault('sample_fn',sample_fn)
        return launch(command,**kwargs)
    print('CPU_FIXTURE_SAMPLE enter: synthetic resources; not OS capacity evidence')
    try:
        with patch.object(monitor,'monitored_child',side_effect=child):
            yield
    finally:
        print('CPU_FIXTURE_SAMPLE exit: sampler injection scope restored')


class FixtureSampleBoundary(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.root=Path(tmp.name)
        self.addCleanup(tmp.cleanup)
        destination=os.environ.get('STEP122_CPU_EVIDENCE')
        if destination:
            self.addCleanup(lambda:shutil.copytree(self.root,
                Path(destination)/('sampler-'+self.root.name)))

    def child(self,code):
        return support.monitor().monitored_child([c.PYTHON,'-I','-B','-c',code],
            cwd=self.root,env=dict(os.environ),out=self.root,mode='worker',
            filesystem=self.root,interval=.005)

    def test_success_scope_and_exception_restoration(self):
        monitor=support.monitor();original=monitor.monitored_child
        with fixture_resource_samples():
            result=self.child('import time; print("CPU child"); time.sleep(.05)')
            self.assertIsNot(monitor.monitored_child,original)
            self.assertEqual(result['exit_code'],0)
            self.assertIsNone(result['primary'])
            self.assertTrue(result['stop_confirmed'])
            self.assertGreater(result['samples'],0)
            self.assertEqual(result['last_sample'],synthetic_resource_sample())
            support.process_record(self.root,'worker')
        self.assertIs(monitor.monitored_child,original)
        with self.assertRaisesRegex(RuntimeError,'fixture construction failed'):
            with fixture_resource_samples():raise RuntimeError('fixture construction failed')
        self.assertIs(monitor.monitored_child,original)

    def test_nonzero_child_remains_failure(self):
        monitor=support.monitor();original=monitor.monitored_child
        with self.assertRaises(monitor.ChildFailure):
            with fixture_resource_samples():
                self.child('import time; time.sleep(.05); raise SystemExit(7)')
        self.assertIs(monitor.monitored_child,original)
        result=c.read_json(self.root/'worker-exit.json')
        self.assertEqual(result['exit_code'],7)
        self.assertEqual(result['primary']['kind'],'child_exit')
        self.assertTrue(result['stop_confirmed']);self.assertEqual(result['remaining'],[])
        with self.assertRaises(ValueError):support.process_record(self.root,'worker')

    def test_sample_exception_remains_failure(self):
        monitor=support.monitor();original=monitor.monitored_child
        def broken_sample(*_):raise KeyError('synthetic sample failure')
        with self.assertRaises(monitor.ChildFailure):
            with fixture_resource_samples(broken_sample):
                self.child('import time; time.sleep(2)')
        self.assertIs(monitor.monitored_child,original)
        result=c.read_json(self.root/'worker-exit.json')
        self.assertEqual(result['primary']['kind'],'monitor_unavailable')
        self.assertEqual(result['primary']['error_type'],'KeyError')
        self.assertEqual(result['primary']['operation'],'resource_sample')
        self.assertIn('synthetic sample failure',result['primary']['detail'])
        self.assertTrue(result['stop_confirmed']);self.assertEqual(result['remaining'],[])
        with self.assertRaises(ValueError):support.process_record(self.root,'worker')


class DisplayAndPrepare(unittest.TestCase):
    def test_visuals_independent_signed_lsb_and_readback(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'visuals'
            expected=np.array([[[10,20,30],[100,100,100]],[[0,255,128],[8,16,32]]],np.uint8)
            actual=np.array([[[11,19,30],[102,98,100]],[[255,0,128],[8,16,32]]],np.uint8)
            before=(actual.copy(),expected.copy())
            patches=[dict(source_index=3,bounds=[0,0,1,1]),dict(source_index=8,bounds=None)]
            record=compare.save_visuals(root,actual,expected,patches)
            signed=np.array([[[1,-1,0],[2,-2,0]],[[255,-255,0],[0,0,0]]],np.int16)
            self.assertTrue(np.array_equal(np.load(root/'signed-lsb.npy'),signed))
            self.assertTrue(np.array_equal(np.load(root/'absolute-lsb.npy'),np.abs(signed)))
            with Image.open(root/'signed-x8.png') as im:
                self.assertEqual(list(im.getdata()),[(136,120,128),(144,112,128),(255,0,128),(128,128,128)])
            with Image.open(root/'absolute-x8.png') as im:
                self.assertEqual(list(im.getdata()),[(8,8,0,255),(16,16,0,255),(255,255,0,255),(0,0,0,255)])
            with Image.open(root/'absolute-x1.png') as im:
                self.assertEqual(list(im.getdata()),[(1,1,0,255),(2,2,0,255),(255,255,0,255),(0,0,0,255)])
            with Image.open(root/'capture-reference.png') as im:
                self.assertEqual(im.size,(4,2))
                self.assertTrue(np.array_equal(np.asarray(im)[:,:2],actual))
                self.assertTrue(np.array_equal(np.asarray(im)[:,2:],expected))
            self.assertEqual(record['patches'],patches)
            self.assertEqual(record['fixed_gain'],8)
            self.assertTrue(all(np.array_equal(a,b) for a,b in zip(before,(actual,expected))))
            m=compare.evaluate(actual,expected,patches,'NEW-ON')
            self.assertEqual(m['whole']['valuesOver1LSB'],4)
            self.assertEqual(m['patches'][0]['metrics']['valuesOver1LSB'],0)
            self.assertEqual(m['patches'][1]['status'],'outside-or-invisible')
            with self.assertRaises(FileExistsError):compare.save_visuals(root,actual,expected,patches)
            evidence=os.environ.get('STEP122_CPU_EVIDENCE')
            if evidence:
                out=Path(evidence);out.mkdir()
                shutil.copytree(root,out/'synthetic-visuals')
                save_json(out/'visuals.json',record)

    def test_prepare_real_entry_temporary_publication(self):
        # Tiny authority paths are substituted; candidate/member/authorization/
        # source-delivery/publication checks and filesystem operations are real.
        original_args=capture.generator_args()
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);viewer=root/'viewer';prior=root/'prior';reports=root/'reports';run=root/'run'
            for p in (viewer/'demo/js',viewer/'build/demo/js',prior/'publication',reports,run/'bundle'):p.mkdir(parents=True)
            (viewer/'demo/js/a.js').write_text('fixture source')
            (viewer/'build/demo/js/a.js').write_text('fixture source')
            (viewer/'build/demo/step21-validation3').symlink_to(prior)
            for n in ('bundle','receipt.json','result.json','worker-exit.json','reader-exit.json'):
                if n!='bundle':(run/n).write_text('{}')
                (prior/'publication'/n).symlink_to(run/n)
            fixed=dict(authority={},selection={'representatives':[1]},camera={'values':{'width':2,'height':2}})
            inp=dict(authority={},selection=fixed['selection'],baseUrl='/step21-validation3/publication/')
            save_json(prior/'step21-validation3-input.json',inp)
            lineage=prior/'lineage.json';save_json(lineage,{'records':[dict(srcIndex=1,valid=True,screenCenter=[0,0])]})
            for module,name,value in ((c,'VIEWER',viewer),(c,'PRIOR',prior),(c,'REPORTS',reports),
                    (c,'BUNDLE_RUN',run),(capture,'LINEAGE',lineage)):
                stack.enter_context(patch.object(module,name,value))
            stack.enter_context(patch.object(capture,'generator_args',return_value=original_args))
            stack.enter_context(patch.object(c,'fixed_authority',return_value=fixed))
            stack.enter_context(patch.object(c,'support_identity',return_value={'synthetic':c.identity(b'source')}))
            for case in ('unauthorized','member','served','destination','publication','success','link-failure'):
                directory=reports/case;directory.mkdir()
                value=dict(schema='viewer-old-capture-candidate-v1',execution_authorized=False,review_required=True,
                    prefix=capture.PREFIX,mode=dict(schema='viewer-old-source-bound-v1',mode='OLD',math_version=None,
                        executed=False,terminated_pixel_count=None),range_input=dict(fixed=fixed,dependencies=c.support_identity()),
                    input=inp,source_delivery=capture.source_delivery(),patches=compare.patch_plan(c.read_json(lineage),fixed),
                    patch_origin=dict(path=str(lineage),identity=c.file_id(lineage)),generator_argv=original_args,
                    preparation=prepare.plan(directory))
                save_json(directory/'candidate.json',value)
                for n in (capture.NAME+'-input.json',capture.NAME+'-browser.js',capture.NAME+'-capture.js'):
                    (directory/n).write_text('fixture member')
                (directory/'prepare-candidate.sh').write_text(capture.prepare_script(directory))
                files=dict(schema='viewer-capture-candidate-files-v1',executable=False,review_required=True,
                    delivery_verified=False,browser_executed=False,members={p.name:c.file_id(p) for p in directory.iterdir()})
                save_json(directory/'files.json',files)
                authorization=dict(schema='viewer-old-prepare-authorization-v1',execution_authorized=True,
                    candidate=c.file_id(directory/'candidate.json'),files=c.file_id(directory/'files.json'))
                link=viewer/'build/demo/step122-old-candidate';served=viewer/'build/demo/js/a.js'
                if case=='unauthorized':
                    authorization['execution_authorized']=False
                    proc=subprocess.run(['bash',str(directory/'prepare-candidate.sh')],capture_output=True,text=True)
                    self.assertEqual(proc.returncode,1);self.assertIn('authorization required',proc.stdout)
                if case=='member':(directory/(capture.NAME+'-capture.js')).write_text('changed')
                if case=='served':served.write_text('stale')
                if case=='destination':link.symlink_to(root/'nonexistent')
                pub=prior/'publication/receipt.json'
                if case=='publication':pub.unlink();pub.symlink_to(run/'result.json')
                if case=='success':
                    auth=reports/'authorization.json';save_json(auth,authorization)
                    result=cli(prepare.main,['prepare','--directory',str(directory),'--authorization',str(auth)])
                    self.assertEqual(json.loads(result)['status'],'prepared-not-executed')
                    self.assertEqual(os.readlink(link),str(directory))
                    self.assertFalse(c.read_json(directory/'prepare-result.json')['browser_executed'])
                    with self.assertRaisesRegex(ValueError,'repeated_or_extra'):prepare.prepare(directory,authorization)
                    link.unlink()  # synthetic sandbox cleanup only
                elif case=='link-failure':
                    with patch.object(Path,'symlink_to',side_effect=OSError('synthetic link failure')):
                        with self.assertRaises(OSError):prepare.prepare(directory,authorization)
                    self.assertTrue((directory/'prepare-attempt.json').exists())
                    self.assertFalse(c.read_json(directory/'prepare-failure.json')['retry'])
                    self.assertFalse((directory/'prepare-result.json').exists())
                else:
                    with self.subTest(case=case),self.assertRaises(ValueError):prepare.prepare(directory,authorization)
                    self.assertFalse((directory/'prepare-attempt.json').exists())
                    if case=='destination':self.assertEqual(os.readlink(link),str(root/'nonexistent'));link.unlink()
                    else:self.assertFalse(os.path.lexists(link))
                if case=='served':served.write_text('fixture source')
                if case=='publication':pub.unlink();pub.symlink_to(run/'receipt.json')


class FlowCPU(CPUCase):
    initial=existing.ActualCPU.initial
    bundle=existing.ActualCPU.bundle
    provenance_fixture=existing.ActualCPU.provenance_fixture
    final_fixture=existing.ActualCPU.final_fixture

    @contextmanager
    def fixture(self):
        from checkpoint_cuda_acceptance import diagnostic_camera
        import staged_comparison_worker as worker
        # Only construction's worker/reader/verify child samples are synthetic.
        # Restore before reference/candidate/wrapper/consumer work, also on error.
        with fixture_resource_samples():
            _,_,loaded,_,_=self.final_fixture()
        root=self.f.base;config=loaded['config'];ck=Path(config.paths.output_directory)/'checkpoints/completed-12000'
        meta=c.read_json(ck/'metadata.json')
        selected=dict(checkpoint=str(ck),marker=c.file_id(ck/'complete.json'),metadata=c.file_id(ck/'metadata.json'))
        origin=self.provenance_fixture()
        old=dict(schema=ai.SCHEMA,execution_authorized=False,executable=False,review_required=True,selected=selected,
            bridge=dict(current_provenance=origin),output=str(ai.OUTPUT),python=c.PYTHON,
            entry=str(c.CORRECTED/'tests/cuda/adopted_reference_acceptance.py'),resources=support.RESOURCES,
            unverified=['synthetic CPU only'],gate_c_passed=False)
        candidate=root/'adopted.json';save_json(candidate,old)
        rep=['train',1,self.f.ref['frames']['train'][1][0],16.5]
        cam=diagnostic_camera(config,tuple(rep[:3]));record=rt.camera_record(cam)
        oldrun=root/'old-run';oldrun.mkdir()
        save_json(oldrun/'worker-result.json',dict(consumed={p:c.file_id(p) for p in
            (cam.image_path,cam.image_path.replace('/images/','/masks/'))}))
        artifact=root/'artifact';(artifact/'bundle').mkdir(parents=True)
        spl4=artifact/'bundle/population.splat4d'
        bundle.export_spl4(spl4,loaded['state']['capture'],bundle.serializer(ai.SERIALIZER,c.file_id(ai.SERIALIZER)))
        fixed=dict(authority={'spl4':c.file_id(spl4)},population=dict(rows=2,identity=c.file_id(spl4)),
            checkpoint=selected,selection=dict(view=1,start=1,count=1,representatives=[1]),
            camera=record,frame=rep,viewer_projection={'synthetic_projection':True},encoding=bundle.ENCODING)
        prior=root/'prior';prior.mkdir()
        for name in ('browser.js','capture.js','prepare.sh','preparation-evidence.json'):
            shutil.copyfile(c.PRIOR/('step21-validation3-'+name),prior/('step21-validation3-'+name))
        save_json(prior/'step21-validation3-input.json',dict(authority=fixed['authority'],selection=fixed['selection'],
            baseUrl='/step21-validation3/publication/'))
        lineage=prior/'lineage.json';save_json(lineage,dict(records=[dict(srcIndex=1,valid=True,screenCenter=[2,3])]))
        t=self.torch;calls=[]
        def kernel(*args):
            h,w=args[17:19];n=len(args[1]);calls.append((n,h,w))
            return (1,t.full((3,h,w),.5),t.zeros(2,h,w),t.ones(1,h,w),t.ones(1,h,w),
                t.ones(n,dtype=t.int32),t.zeros(1,dtype=t.uint8),t.zeros(1,dtype=t.uint8),t.zeros(1,dtype=t.uint8),
                t.zeros(n,6),t.zeros(n,3),t.empty(0),t.empty(0))
        @contextmanager
        def native_boundary(provenance):
            self.assertEqual(provenance,origin)
            yield self.train,self.f.renderer
        with ExitStack() as stack:
            for module,name,value in ((ai,'CHECKPOINT',ck),(ai,'CANDIDATE',candidate),(ai,'RUN',oldrun),
                    (c,'BUNDLE_RUN',artifact),(c,'REPORTS',root),(c,'PRIOR',prior),(capture,'LINEAGE',lineage)):
                stack.enter_context(patch.object(module,name,value))
            stack.enter_context(patch.object(ai,'selected',return_value=(meta,selected,{})))
            stack.enter_context(patch.object(ai,'bridge',return_value=old['bridge']))
            stack.enter_context(patch.object(c,'fixed_authority',return_value=fixed))
            stack.enter_context(patch.object(worker,'native',native_boundary))
            native=self.f.renderer.diff_gaussian_rasterization._C
            stack.enter_context(patch.object(native,'rasterize_gaussians',kernel,create=True))
            for name,value in (('mem_get_info',(10**15,10**15)),('memory_stats',{'inactive_split_bytes.all.current':0}),
                               ('memory_allocated',0),('memory_reserved',0)):
                stack.enter_context(patch.object(t.cuda,name,return_value=value))
            stack.enter_context(patch.dict(support.resources().memory.__globals__,
                collectors=lambda:{'host_available':lambda:10**15}))
            stack.enter_context(patch.object(support.shutil,'disk_usage',return_value=SimpleNamespace(free=10**15)))
            inp=c.candidate(root/'reference-output')
            yield inp,calls,native
        self.assertFalse(t.cuda.is_initialized())

    def records(self,candidate,png_id):
        # Small historical JSON is only a schema template, not a current image.
        records={n:c.read_json(c.WORK/'json'/('step21_validation3_000118_v16_'+n+'.json')) for n in compare.SUFFIXES}
        b=records['browser_result'];f=candidate['range_input']['fixed'];p=records['png_capture_status']
        ctx=b['inputContext'];ctx.update(external=f['authority'],selection=f['selection'],camera=f['camera']['values'],
            frame=f['frame'],rows=f['population']['rows'])
        snap=b['productionSnapshot'];snap['adoptedBundle']=copy.deepcopy(ctx);b['pngBinding']['context']=copy.deepcopy(ctx)
        snap['projection']=f['viewer_projection'];snap['appliedConfig']['timestamp']=f['frame'][3]
        snap['assetIdentity']['spl4AssetIdentity'].update(sha256=f['population']['identity']['sha256'],
            sizeBytes=f['population']['identity']['size'],recordCount=2)
        snap['productionWorksetIdentity'].update(residentStart=1,residentRecordCount=1,residentEndExclusive=2)
        b['semanticComparison']['orchestrationResult']['coverage'].update(processedRecordCount=1,uniqueSrcIndexCount=1,
            firstSrcIndex=1,lastSrcIndex=1)
        frames=[snap['productionIdentity'][n] for n in ('productionFrameIdentity','presentedFrameIdentity')]
        frames += [p[n] for n in ('presentedFrameIdentity','capturedFrameIdentity')]
        for n in ('webgpu_visible_record_dryrun_compare','webgpu_visible_record_lineage'):
            records[n]['artifactSetIdentity']=candidate['prefix']
            records[n]['artifactProvenance']['capturePrefix']=candidate['prefix']
            frames.append(records[n]['artifactProvenance']['frameIdentity'])
        for frame in frames:
            frame.update(datasetCameraLabel=f['frame'][2],referenceCameraLabel=f['frame'][2],datasetFrameNumber=f['frame'][1],
                datasetTime=f['frame'][3],outputWidth=20,outputHeight=10,frameHash=None)
        name=candidate['prefix']+'_canvas.png'
        p['captureBlobIdentity'].update(fileName=name,sizeBytes=png_id['size'],sha256=png_id['sha256'])
        p['fileName']=name;b['captureStatus']=copy.deepcopy(p)
        b['captureSummary']['png'].update(fileName=name,blobSha256=png_id['sha256'])
        b['comparisonSupport']=dict(schema='viewer-range-capture-support-v1',candidate_identity=c.identity(c.canonical(candidate)),
            delivery=candidate['source_delivery'],mode=dict(candidate['mode'],executed=True),json_members={})
        return records

    def test_actual_cli_reference_to_comparison_and_rejects(self):
        import numpy as np
        from PIL import Image
        with self.fixture() as (inp,calls,_):
            root=self.f.base;candidate=capture.generate(root/'candidate',inp)
            authorized=dict(inp,execution_authorized=True);source=root/'input.json';save_json(source,authorized)
            digest=c.identity(c.canonical(authorized))['sha256']
            text=cli(rr.main,['render',str(source),'--execute','--expect-input-sha256',digest])
            reference=Path(inp['output']);receipt=c.read_json(reference/'receipt.json')
            self.assertEqual(json.loads(text),receipt);self.assertEqual(calls,[(1,10,20)])
            manifest=c.read_json(reference/'reference/manifest.json')
            self.assertEqual(manifest['metadata']['rows'],2)
            self.assertEqual(manifest['mapping']['checked_rows'],1)
            self.assertEqual(manifest['runtime']['native']['tensors']['19']['shape'],[1,16,3])
            self.assertFalse((reference/'.pending-range').exists())
            with self.assertRaisesRegex(ValueError,'exclusive_output'):rr.execute(authorized,digest)
            images=root/'capture';images.mkdir()
            pixels=np.full((10,20,4),127,np.uint8);pixels[:,:,3]=255;pixels[0,0,:3]=[128,126,127]
            png=images/(capture.PREFIX+'_canvas.png');Image.fromarray(pixels,'RGBA').save(png)
            records=self.records(candidate,c.file_id(png))
            for n in compare.SUFFIXES:
                if n!='browser_result':save_json(images/(capture.PREFIX+'_'+n+'.json'),records[n])
            records['browser_result']['comparisonSupport']['json_members']={p.name:c.file_id(p) for p in images.glob('*.json')}
            browser=images/(capture.PREFIX+'_browser_result.json');save_json(browser,records['browser_result'])
            candidate_path=root/'candidate/candidate.json';output=root/'comparison.json'
            args=['compare','--candidate',str(candidate_path),'--captures',str(images),'--reference',str(reference)]
            result_text=cli(compare.main,[*args,'--output',str(output)])
            result=c.read_json(output)
            self.assertEqual(result_text.strip(),'inconclusive');self.assertFalse(result['accepted'])
            self.assertEqual(result['whole']['maxAbsError'],1);self.assertEqual(result['whole']['valuesOver1LSB'],0)
            self.assertEqual(result['visuals']['members'],{p.name:c.file_id(p) for p in (root/'comparison.json.images').iterdir()})
            signed=np.load(root/'comparison.json.images/signed-lsb.npy')
            self.assertTrue(np.array_equal(signed[0,0],[1,-1,0]));self.assertEqual(np.count_nonzero(signed),2)
            before=c.file_id(output)
            with self.assertRaisesRegex(ValueError,'output_exists'):cli(compare.main,[*args,'--output',str(output)])
            self.assertEqual(c.file_id(output),before)
            mutations=[('frame',browser),('source',browser),('png',png),
                       ('json',images/(capture.PREFIX+'_limited_draw_summary.json')),
                       ('reference',reference/'reference/view/render.png'),
                       ('missing',images/(capture.PREFIX+'_webgpu_visible_record_lineage.json'))]
            for kind,path in mutations:
                saved=path.read_bytes()
                if kind in ('frame','source'):
                    bad=c.read_json(browser)
                    if kind=='frame':bad['productionSnapshot']['productionIdentity']['productionFrameIdentity']['generation']+=1
                    else:bad['comparisonSupport']['delivery']={}
                    save_json(browser,bad)
                elif kind=='missing':path.unlink()
                else:path.write_bytes(saved+b'changed')
                failed=root/(kind+'.json')
                with patch.object(compare,'save_visuals',side_effect=AssertionError('downstream reached')):
                    cli(compare.main,[*args,'--output',str(failed)])
                self.assertIn(c.read_json(failed)['status'],('failed','not-established'))
                self.assertFalse((root/(kind+'.json.images')).exists())
                path.write_bytes(saved)
            evidence=os.environ.get('STEP122_CPU_EVIDENCE')
            if evidence:
                out=Path(evidence)/'synthetic-flow';out.mkdir()
                for name in ('comparison.json','comparison.json.images'):
                    p=root/name
                    if p.is_dir():shutil.copytree(p,out/name)
                    else:shutil.copyfile(p,out/name)
                shutil.copytree(reference,out/'reference')
                save_json(out/'scope.json',dict(synthetic=True,production_payload=False,native=False,cuda=False,
                    real_entries=['render_adopted_range_reference.main','compare_adopted_range_capture.main'],
                    native_calls=calls,full_rows=2,selected_rows=1,dimensions=[20,10]))

    def test_preflight_and_native_failure_no_publication(self):
        with self.fixture() as (inp,calls,native):
            authorized=dict(inp,execution_authorized=True);root=Path(inp['output'])
            with self.assertRaisesRegex(ValueError,'authorization'):rr.execute(authorized,'wrong')
            self.assertFalse(root.exists());self.assertEqual(calls,[])
            with patch.object(native,'rasterize_gaussians',side_effect=RuntimeError('injected native failure')):
                with self.assertRaisesRegex(RuntimeError,'injected native'):
                    rr.execute(authorized,c.identity(c.canonical(authorized))['sha256'])
            self.assertEqual(c.read_json(root/'failure.json')['stage'],'render')
            self.assertFalse((root/'receipt.json').exists());self.assertFalse((root/'reference').exists())
            saved=(root/'failure.json').read_bytes()
            with self.assertRaisesRegex(ValueError,'exclusive_output'):
                rr.execute(authorized,c.identity(c.canonical(authorized))['sha256'])
            self.assertEqual(saved,(root/'failure.json').read_bytes())


if __name__=='__main__':unittest.main(verbosity=2)
