"""Synthetic CPU-only fixtures. Real reader/selector/restore/renderer/PNG consumers.

Uses the existing Corrected CPU fixture to isolate native/CUDA seams. Never reads
production tensor/SPL4/image payloads, queries a GPU, or executes a browser.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import adopted_range_contract as c
import render_adopted_range_reference as rr
import compare_adopted_range_capture as compare
import make_adopted_range_capture as capture

c.require(sys.executable == c.PYTHON,'test_python')
c.corrected_paths()
from formal_training_fixture import CPUCase
import test_adopted_reference as existing_cpu
import staged_comparison_checkpoint as cp
import adopted_reference_bundle as bundle
import adopted_reference_runtime as rt
import staged_comparison_adopted_input as ai


class Contracts(unittest.TestCase):
    def test_preflight_subprocess_no_heavy_or_write(self):
        code='''
import sys
sys.path.insert(0,sys.argv[1])
class Deny:
 def find_spec(self,name,*a):
  if name.split('.')[0] in ('torch','numpy','PIL','scene','gaussian_renderer','train'):
   raise AssertionError('heavy_import')
sys.meta_path.insert(0,Deny())
import adopted_range_contract as c
import render_adopted_range_reference as r
for v in ({},{'schema':'wrong'}):
 try:r.execute(v,None)
 except ValueError:pass
 else:raise AssertionError('accepted')
print(sys.executable)
'''
        p=subprocess.run([sys.executable,'-I','-B','-S','-c',code,str(c.VIEWER/'tools')],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(p.stdout.strip(),c.PYTHON)

    def test_preflight_identity_matrix(self):
        fixed={n: {'value':n} for n in ('checkpoint','population','camera','frame','selection','encoding')}
        deps={'required':c.identity(b'source')}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/'absent'; original=root/'original.json';original.write_text('{}')
            v=dict(schema=c.SCHEMA,python=sys.executable,execution_authorized=False,review_required=True,
                   output=str(output),fixed=fixed,dependencies=deps,
                   adopted_candidate=dict(path=str(original),identity=c.file_id(original)))
            with patch.object(c,'REPORTS',root),patch.object(c,'fixed_authority',return_value=fixed),\
                 patch.object(c,'support_identity',return_value=deps),patch.object(ai,'CANDIDATE',original),\
                 patch.object(ai,'verify',return_value='authority-boundary'):
                self.assertEqual(c.preflight(v),'authority-boundary')
                for key in fixed:
                    bad=copy.deepcopy(v);bad['fixed'][key]={}
                    with self.subTest(key=key),self.assertRaisesRegex(ValueError,'fixed_input'):c.preflight(bad)
                for key in ('dependencies','adopted_candidate'):
                    bad=copy.deepcopy(v);bad[key]={}
                    with self.subTest(key=key),self.assertRaises(ValueError):c.preflight(bad)
                with self.assertRaisesRegex(ValueError,'separate_execution'):rr.execute(v,None)
                bad=copy.deepcopy(v);bad['execution_authorized']=True
                with self.assertRaisesRegex(ValueError,'separate_execution'):rr.execute(bad,'wrong')
                output.mkdir()
                with self.assertRaisesRegex(ValueError,'exclusive_output'):c.preflight(v)
                self.assertEqual(list(output.iterdir()),[])

    def test_mode_strict(self):
        old=dict(schema='viewer-old-source-bound-v1',mode='OLD',math_version=None,executed=True,terminated_pixel_count=None)
        self.assertEqual(c.parse_mode(old),old)
        for field,value in (('mode','NEW-OFF'),('math_version','cuda-aligned-composite-v1'),('executed',1)):
            with self.subTest(field=field),self.assertRaises(ValueError):c.parse_mode(dict(old,**{field:value}))
        new=dict(schema='viewer-range-policy-evidence-v1',mode='NEW-ON',math_version='cuda-aligned-composite-v1',
                 executed=True,terminated_pixel_count=7)
        self.assertEqual(c.parse_mode(new),new)
        for value in (dict(new,mode='NEW-OFF'),dict(new,math_version=None),dict(new,schema='unknown'),
                      dict(new,terminated_pixel_count=True),dict(new,extra=0)):
            with self.assertRaises(ValueError):c.parse_mode(value)
        del new['terminated_pixel_count']
        with self.assertRaises(ValueError):c.parse_mode(new)

    def test_frame_optional_hash_and_stale(self):
        fixed=dict(frame=['train',1,'image.png',2.0],camera={'values':{'width':4,'height':3}})
        frame=dict(generation=3,datasetCameraLabel='image.png',datasetFrameNumber=1,datasetTime=2.0,
                   referenceCameraLabel='image.png',outputWidth=4,outputHeight=3,frameHash=None)
        compare.same_frames([frame,dict(frame,frameHash='observed')],fixed)
        for field in compare.FRAME:
            bad=dict(frame);del bad[field]
            with self.subTest(field=field),self.assertRaises(ValueError):compare.same_frames([frame,bad],fixed)
        with self.assertRaisesRegex(ValueError,'stale'):compare.same_frames([frame,dict(frame,generation=2)],fixed)
        with self.assertRaisesRegex(ValueError,'hash_conflict'):
            compare.same_frames([dict(frame,frameHash='a'),frame,dict(frame,frameHash='b')],fixed)

    def test_metrics_independent_known_values(self):
        import numpy as np
        a=np.zeros((2,2,3),np.uint8);b=a.copy();a[0,0]=[1,2,3]
        patches=[dict(source_index=4,bounds=[0,0,1,1]),dict(source_index=5,bounds=None)]
        result=compare.evaluate(a,b,patches,'NEW-ON')
        self.assertEqual(result['status'],'inconclusive');self.assertFalse(result['accepted'])
        m=result['whole'];self.assertEqual(m['mae'],.5);self.assertAlmostEqual(m['rmse'],(14/12)**.5)
        self.assertEqual((m['maxAbsError'],m['valuesOver1LSB'],m['pixelsOver1LSB']),(3,2,1))
        self.assertEqual(result['patches'][0]['metrics']['mae'],2)
        a[:]=1
        self.assertEqual(compare.evaluate(a,b,patches,'NEW-ON')['status'],'candidate-within-1LSB')
        self.assertEqual(compare.evaluate(a,b,patches,'OFF-vs-ON')['status'],'inconclusive')
        self.assertEqual(compare.evaluate(a,b,patches,'OLD',known_violation=True)['status'],'failed')
        self.assertEqual(compare.evaluate(a,b,patches,'OLD',evidence_complete=False)['status'],'not-established')
        with self.assertRaises(ValueError):compare.evaluate(a[:1],b,patches,'OLD')
        with self.assertRaises(ValueError):compare.evaluate(a,b,[dict(source_index=4,bounds=[0,0,5,1])],'OLD')

    def test_patch_plan_prior_fixed_locations(self):
        fixed=dict(selection={'representatives':[5,7]},camera={'values':{'width':20,'height':20}})
        record=dict(records=[dict(srcIndex=7,valid=False,screenCenter=[30,3]),dict(srcIndex=5,valid=True,screenCenter=[10,10])])
        self.assertEqual(compare.patch_plan(record,fixed),[dict(source_index=5,bounds=[2,2,19,19]),dict(source_index=7,bounds=None)])

    def test_capture_json_binding_and_negative_matrix(self):
        # Historical small JSON is a protocol template only. No old PNG is read
        # or passed off as a current capture; source/PNG identities below are
        # independently synthetic and the result is only a parser test.
        records={n:c.read_json(c.WORK/'json'/('step21_validation3_000118_v16_'+n+'.json')) for n in compare.SUFFIXES}
        browser=records['browser_result'];ctx=browser['inputContext']
        fixed=dict(authority=ctx['external'],selection=ctx['selection'],camera={'values':ctx['camera']},frame=ctx['frame'],
                   population=dict(rows=ctx['rows'],identity=ctx['external']['spl4']),
                   viewer_projection=browser['productionSnapshot']['projection'])
        candidate=dict(range_input={'fixed':fixed},prefix=capture.PREFIX,
            mode=dict(schema='viewer-old-source-bound-v1',mode='OLD',math_version=None,executed=False,terminated_pixel_count=None),
            source_delivery={'/js/synthetic.js':c.identity(b'synthetic source')})
        png_identity=c.identity(b'ONLY A SYNTHETIC BLOB IDENTITY; NOT A PNG')
        png=records['png_capture_status'];blob=png['captureBlobIdentity'];name=capture.PREFIX+'_canvas.png'
        blob.update(sizeBytes=png_identity['size'],sha256=png_identity['sha256'],fileName=name)
        png['fileName']=name;browser['captureStatus']=copy.deepcopy(png)
        browser['captureSummary']['png'].update(fileName=name,blobSha256=png_identity['sha256'])
        for n in ('webgpu_visible_record_dryrun_compare','webgpu_visible_record_lineage'):
            records[n]['artifactSetIdentity']=capture.PREFIX
            records[n]['artifactProvenance']['capturePrefix']=capture.PREFIX
        browser['comparisonSupport']=dict(schema='viewer-range-capture-support-v1',candidate_identity=c.identity(c.canonical(candidate)),
            delivery=candidate['source_delivery'],mode=dict(candidate['mode'],executed=True),json_members={})
        self.assertEqual(compare.capture_binding(records,candidate,png_identity)['mode'],'OLD')
        # Preserve the original subset positive above. An outer schema, when
        # present, must agree; the complete-candidate validator remains strict.
        for schema in ('viewer-old-capture-candidate-v1', 'viewer-aligned-capture-candidate-v1', 'unknown', None):
            outer=dict(candidate,schema=schema); bound=copy.deepcopy(records)
            bound['browser_result']['comparisonSupport']['candidate_identity']=c.identity(c.canonical(outer))
            if schema=='viewer-old-capture-candidate-v1':
                self.assertEqual(compare.capture_binding(bound,outer,png_identity)['mode'],'OLD')
            else:
                with self.subTest(schema=schema),self.assertRaisesRegex(ValueError,'legacy_candidate_schema'):
                    compare.capture_binding(bound,outer,png_identity)
        for missing in ('schema','mode','math_version'):
            bad=copy.deepcopy(candidate);del bad['mode'][missing]
            bound=copy.deepcopy(records)
            bound['browser_result']['comparisonSupport']['candidate_identity']=c.identity(c.canonical(bad))
            with self.subTest(missing=missing),self.assertRaises(ValueError):
                compare.capture_binding(bound,bad,png_identity)
        with self.assertRaises(ValueError):capture.validate_candidate(candidate,current=False)
        mutations=[
            lambda r:r.pop('limited_draw_summary'),
            lambda r:r['browser_result']['comparisonSupport'].update(delivery={}),
            lambda r:r['browser_result']['comparisonSupport']['mode'].update(mode='NEW-OFF'),
            lambda r:r['browser_result']['productionSnapshot']['projection'].update(values=[]),
            lambda r:r['browser_result']['inputContext']['selection'].update(start=0),
            lambda r:r['png_capture_status'].update(staleCaptureDetected=True),
            lambda r:r['png_capture_status']['orientationEvidence'].update(captureVerticalFlipApplied=True),
            lambda r:r['png_capture_status']['captureBlobIdentity'].update(sha256=c.identity(b'other')['sha256']),
            lambda r:r['webgpu_visible_record_lineage']['artifactProvenance'].update(requestIdentity='stale'),
            lambda r:r['browser_result'].update(failure={'name':'saved failure'}),
        ]
        for index,mutation in enumerate(mutations):
            bad=copy.deepcopy(records);mutation(bad)
            with self.subTest(index=index),self.assertRaises((ValueError,KeyError)):compare.capture_binding(bad,candidate,png_identity)

    def test_wrapper_and_generator_inheritance(self):
        # Small prior records and source only; no old PNG decode, no deployment.
        args=capture.generator_args()
        original=c.read_json(c.PRIOR/'step21-validation3-preparation-evidence.json')['capture_generator_argv'][3:-2]
        restored=list(args)
        for option in ('--step','--phase-step'):restored[restored.index(option)+1]=original[original.index(option)+1]
        self.assertEqual(restored,original)
        wrapper=capture.wrapper({'fixture':'not authorized'})
        self.assertIn('retryDefaultScene: false',wrapper)
        self.assertLess(wrapper.index('Separate OLD execution authorization required'),wrapper.index('const initial ='))
        self.assertIn('captureFailure = error?.captureFailure',wrapper)
        self.assertIn('comparisonSupport, inputContext',wrapper)
        self.assertIn('candidate.source_delivery',wrapper)


class RangeCPU(CPUCase):
    initial=existing_cpu.ActualCPU.initial
    bundle=existing_cpu.ActualCPU.bundle
    provenance_fixture=existing_cpu.ActualCPU.provenance_fixture
    final_fixture=existing_cpu.ActualCPU.final_fixture
    # Genuine small marker/metadata/state reader; adopted authority is pinned to
    # the fixture, not production; no reader/selector success stub.
    test_strict_adopted_reader=existing_cpu.ActualCPU.test_selected_reader_reuses_real_M_T_and_rejects_wrong_binding
    test_real_camera_and_png=existing_cpu.ActualCPU.test_actual_camera_gt_raw_png_and_independent_png_check
    test_real_renderer_native_observer=existing_cpu.ActualCPU.test_native_observation_actual_renderer_and_camera_copy

    def test_selected_range_and_restore_actual(self):
        _,_,loaded,_,_=self.final_fixture()
        before=copy.deepcopy(loaded['metadata']);original=loaded['state']['capture'];n=len(original[1])
        selection=dict(start=n-1,count=1)
        selected,mapping=rr.select_owned(loaded,selection)
        for slot in rr.SLOTS:
            self.assertTrue(self.torch.equal(selected[slot],original[slot][-1:]))
            self.assertNotEqual(selected[slot].data_ptr(),original[slot][-1:].data_ptr())
        for slot in (0,12,16,17,18):self.assertEqual(selected[slot],original[slot])
        self.assertEqual(loaded['metadata'],before)
        from staged_comparison_state import restore_diagnostic
        full=restore_diagnostic(loaded,'cpu')
        with patch.object(self.torch.optim.Adam,'__init__',side_effect=AssertionError('Adam forbidden')):
            small=rr.restore_range(selected,loaded['config'],'cpu')
        for attr in ('_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity','_t','_scaling_t','_rotation_r',
                     'max_radii2D','xyz_gradient_accum','t_gradient_accum','denom'):
            self.assertTrue(self.torch.equal(getattr(small,attr),getattr(full,attr)[-1:]),attr)
        self.assertIsNone(small.optimizer);self.assertEqual((small.max_sh_degree,small.active_sh_degree),(3,0))
        from checkpoint_cuda_acceptance import diagnostic_camera
        from real_input_cuda_acceptance import render_buffers
        cam=diagnostic_camera(loaded['config'],('train',1,self.f.ref['frames']['train'][1][0]))
        native=self.f.renderer.diff_gaussian_rasterization._C;t=self.torch;observations=[]
        def kernel(*args):
            h,w=args[17:19];rows=len(args[1]);self.assertEqual(rows,1)
            return (1,t.zeros(3,h,w),t.zeros(2,h,w),t.ones(1,h,w),t.ones(1,h,w),
                    t.ones(rows,dtype=t.int32),t.zeros(1,dtype=t.uint8),t.zeros(1,dtype=t.uint8),
                    t.zeros(1,dtype=t.uint8),t.zeros(rows,6),t.zeros(rows,3),t.empty(0),t.empty(0))
        def observed(device,*args,**kwargs):
            with rt.observe_native(native,device,1) as records:
                result=self.f.renderer.render(device,*args,**kwargs)
            observations.extend(records);return result
        with patch.object(native,'rasterize_gaussians',kernel,create=True):
            raw=render_buffers(observed,cam,small,loaded['config'])
        self.assertEqual(len(observations),1);self.assertEqual(raw['radii'].shape,(1,))
        self.assertEqual(observations[0]['scalars']['20'],0)
        self.assertEqual(observations[0]['tensors']['19']['shape'],[1,16,3])
        helper=rr.range_helper()
        for start,count in ((-1,1),(True,1),(0,True),(0,0),(n,1),(0,n+1)):
            with self.subTest(start=start,count=count),self.assertRaises((ValueError,RuntimeError)):
                helper.select_checkpoint_source_range(original,start_inclusive=start,selected_count=count,expected_source_count=n)
        bad=list(original);bad[3]=bad[3][:-1]
        with self.assertRaises((ValueError,RuntimeError)):
            helper.select_checkpoint_source_range(bad,start_inclusive=0,selected_count=1,expected_source_count=n)
        spl4=self.f.base/'small.splat4d'
        serializer=bundle.serializer(ai.SERIALIZER,c.file_id(ai.SERIALIZER))
        bundle.export_spl4(spl4,original,serializer)
        population=dict(rows=n,identity=c.file_id(spl4))
        result=rr.check_selected_spl4(spl4,selected,selection,population)
        self.assertEqual(result['checked_rows'],1)
        wrong=list(selected);wrong[1]=wrong[1]+1
        with self.assertRaisesRegex(ValueError,'correspondence'):rr.check_selected_spl4(spl4,wrong,selection,population)

    def tiny_saved(self):
        t=self.torch
        root=self.f.base/'range-run';root.mkdir();pending=root/'.pending-range';pending.mkdir()
        raw=dict(render=t.full((3,2,3),.5),gt=t.full((3,2,3),.25),alpha=t.full((1,2,3),.5),
                 depth=t.ones(1,2,3),radii=t.ones(2,dtype=t.int32),visibility_filter=t.ones(2,dtype=t.bool))
        fixed=dict(selection=dict(start=5,count=2),camera=dict(values={'width':3,'height':2}),frame=['fixture'])
        record=dict(camera=fixed['camera'],frame=fixed['frame'],native=dict(population_rows=2,call_count=1))
        rt.save_view(pending/'view',raw,record)
        manifest=dict(schema='viewer-range-reference-v1',kind='selected-range-comparison-reference',
                      input=dict(fixed=fixed),metadata={'full_fixture_rows':10},mapping=dict(start=5,count=2,end=7,
                      checked_rows=2,mapping='sourceIndex = rangeStart + localIndex'),runtime=record,environment={'fixture':True},
                      encoding=bundle.ENCODING,members=bundle.members(pending))
        bundle.write_json(pending/'manifest.json',manifest)
        return root,pending,manifest

    def test_save_independent_readback_publication(self):
        root,pending,manifest=self.tiny_saved()
        result=rr.verify_saved(pending,manifest);self.assertFalse(result['image_parity_accepted'])
        receipt=rr.publish(root,manifest)
        self.assertEqual(receipt,c.read_json(root/'receipt.json'))
        self.assertFalse(pending.exists());self.assertTrue((root/'reference').is_dir())
        # Neither claimed output nor confirmed destination may be overwritten.
        with self.assertRaises(FileExistsError):bundle.Claim(root)
        pending.mkdir()
        with self.assertRaises(OSError):bundle.rename_new(pending,root/'reference')

    def test_save_corrupt_and_missing_members(self):
        root,pending,manifest=self.tiny_saved()
        saved=(pending/'view/render.png').read_bytes()
        for mutation in ('missing','extra','changed'):
            if mutation=='missing':(pending/'view/render.png').unlink()
            if mutation=='extra':(pending/'extra').write_text('unexpected')
            if mutation=='changed':(pending/'view/render.png').write_bytes(b'corrupt')
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):rr.verify_saved(pending,manifest)
            if mutation=='extra':(pending/'extra').unlink()
            (pending/'view/render.png').write_bytes(saved)
        bad=copy.deepcopy(manifest);bad['mapping']['start']=4
        bundle.write_json(root/'bad-manifest.json',bad)
        with self.assertRaisesRegex(ValueError,'manifest_binding'):rr.verify_saved(pending,bad)
        bundle.failure(root,'synthetic-stop',ValueError('controlled failure'))
        self.assertEqual(c.read_json(root/'failure.json')['stage'],'synthetic-stop')
        self.assertTrue(pending.exists())
        import numpy as np
        raw={n:np.load(pending/'view'/(n+'.npy')) for n in bundle.RAW}
        for problem in ('shape','dtype','finite'):
            bad={n:v.copy() for n,v in raw.items()}
            if problem=='shape':bad['render']=bad['render'][:1]
            if problem=='dtype':bad['render']=bad['render'].astype(np.float64)
            if problem=='finite':bad['render'][0,0,0]=np.nan
            with self.subTest(problem=problem),self.assertRaises(ValueError):rt.raw_contract(bad,2,2,3)


if __name__=='__main__':unittest.main(verbosity=2)
