"""Synthetic CPU consumer integration. No production payload, GPU or browser."""
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

import adopted_range_contract as c
import aligned_range_capture as aligned
import compare_adopted_range_capture as compare
import make_adopted_range_capture as capture
import render_adopted_range_reference as reference
import smoke_adopted_range_flow as flow


class AlignedCPU(flow.CPUCase):
    initial=flow.FlowCPU.initial
    bundle=flow.FlowCPU.bundle
    provenance_fixture=flow.FlowCPU.provenance_fixture
    final_fixture=flow.FlowCPU.final_fixture
    fixture=flow.FlowCPU.fixture
    records=flow.FlowCPU.records

    def setUp(self):
        super().setUp()
        destination=os.environ.get('STEP122_CPU_EVIDENCE')
        if destination:
            # Preserve both success and failure fixtures before inherited cleanup.
            self.addCleanup(lambda:shutil.copytree(self.f.base,Path(destination)/('flow-'+self.f.base.name)))

    def node(self, label, args):
        root=self.f.base
        with (root/(label+'.stdout.log')).open('xb') as out,(root/(label+'.stderr.log')).open('xb') as err:
            p=subprocess.run(['/home/demo/.nvm/versions/node/v20.20.1/bin/node',*args],stdout=out,stderr=err)
        self.assertEqual(p.returncode,0,(root/(label+'.stderr.log')).read_text())

    def aligned_records(self, candidate, png_identity, generation):
        records=self.records(candidate,png_identity)
        b=records['browser_result']; p=records['png_capture_status']; snap=b['productionSnapshot']
        mode=candidate['mode']['mode']; request='synthetic-'+mode
        policy=dict(schema=aligned.POLICY,mathVersion=aligned.MATH,purpose='validation-only',
            requestedMode=mode,appliedMode=mode,requestIdentity=request,
            candidateIdentity=c.identity(c.canonical(candidate))['sha256'],executed=True,
            onPathExecuted=mode=='NEW-ON',normalOnAccepted=False,accepted=False,offOnComparisonComplete=False,
            terminatedPixelCount=1 if mode=='NEW-ON' else 0,terminationObserved=mode=='NEW-ON',
            evaluatedPixelReferenceCount=None,skippedPixelReferenceCount=None)
        snap['compositingPolicy']=policy
        snap['requestIdentity']['schedulerRequestIdentity']=request
        identity=snap['productionIdentity']
        identity.update(productionGeneration=generation,presentedGeneration=generation)
        p.update(productionOutputGeneration=generation,presentedOutputGeneration=generation,capturedOutputGeneration=generation)
        fresh=b['captureSummary']['captureExpectationContract']['freshIdentity']
        fresh.update(baselineProductionGeneration=generation-1,productionGeneration=generation,
            compositorGeneration=generation,presentedGeneration=generation,
            requestIdentity=request,productionSourceRequestIdentity=request)
        frames=[identity['productionFrameIdentity'],identity['presentedFrameIdentity'],
                p['presentedFrameIdentity'],p['capturedFrameIdentity'],p['requestedStateIdentity']]
        for n in ('webgpu_visible_record_dryrun_compare','webgpu_visible_record_lineage'):
            provenance=records[n]['artifactProvenance']
            provenance.update(requestIdentity=request,productionGeneration=generation)
            frames.append(provenance['frameIdentity'])
        for frame in frames:
            frame.update(generation=generation,compositingMathVersion=aligned.MATH,
                compositingMode=mode,compositingRequestIdentity=request)
        b['captureStatus']=copy.deepcopy(p)
        b['comparisonSupport']['mode']=aligned.policy_evidence(policy,mode,request,generation)
        return records

    def test_actual_new_candidates_binding_cli_and_pair(self):
        import numpy as np
        from PIL import Image
        with self.fixture() as (inp,calls,_):
            root=self.f.base
            request=root/'vm-bundle-request.json'
            flow.save_json(request,dict(output=str(root/'vm-publication'),fixed=inp['fixed'],
                spl4=str(c.BUNDLE_RUN/'bundle/population.splat4d')))
            self.node('bundle',['tools/step122_aligned_vm_bundle.mjs',str(request)])
            verified=c.read_json(root/'vm-publication/fixture-authority.json')
            inp['fixed'].update(authority=verified['authority'],viewer_projection=verified['projection'])
            actual_input=c.read_json(c.PRIOR/'step21-validation3-input.json')
            actual_input['authority']=verified['authority']
            flow.save_json(c.PRIOR/'step21-validation3-input.json',actual_input)
            # All synthetic input bytes must be finalized before provenance is
            # captured, exactly as for a real candidate. Do not relax preflight.
            inp=c.candidate(root/'reference-output')
            old=capture.generate(root/'old-candidate',inp)
            old_path=root/'old-candidate/candidate.json'
            capture.validate_candidate(old,current=False)
            for field in ('schema','mode'):
                bad=copy.deepcopy(old);del bad[field]
                with self.assertRaises(ValueError):capture.validate_candidate(bad,current=False)
            authorized=dict(inp,execution_authorized=True)
            reference.execute(authorized,c.identity(c.canonical(authorized))['sha256'])
            ref=Path(inp['output']);self.assertEqual(calls,[(1,10,20)])
            original_ids=aligned.reference_binding(ref)
            with patch.object(aligned,'OLD',old_path),patch.object(aligned,'REFERENCE',ref):
                generated=root/'new-candidates';aligned.generate(generated)
                candidates=[c.read_json(generated/m/'candidate.json') for m in ('new-off','new-on')]
                for value in candidates:
                    capture.validate_candidate(value,current=True)
                    self.assertEqual(value['range_input'],old['range_input'])
                    self.assertFalse(value['execution_authorized']);self.assertTrue(value['review_required'])
                with self.assertRaisesRegex(ValueError,'exclusive_candidate_directory'):aligned.generate(generated)
                images=root/'captures';images.mkdir()
                for i,value in enumerate(candidates):
                    pixels=np.full((10,20,4),127,np.uint8);pixels[:,:,3]=255
                    pixels[0,0,:3]=[125+i,127,127]  # independently specified synthetic pixels
                    png=images/(value['prefix']+'_canvas.png');Image.fromarray(pixels,'RGBA').save(png)
                    records=self.aligned_records(value,c.file_id(png),10+i)
                    self.assertEqual(compare.capture_binding(records,value,c.file_id(png))['mode'],value['mode']['mode'])
                    # Outer schema cannot be missing, forged, or contradicted in NEW.
                    for schema in ('missing',None,'unknown','viewer-old-capture-candidate-v1'):
                        bad=copy.deepcopy(value)
                        if schema=='missing':del bad['schema']
                        else:bad['schema']=schema
                        altered=copy.deepcopy(records)
                        altered['browser_result']['comparisonSupport']['candidate_identity']=c.identity(c.canonical(bad))
                        with self.subTest(schema=schema),self.assertRaises(ValueError):
                            compare.capture_binding(altered,bad,c.file_id(png))
                    for mutation in (
                        lambda b:b['productionSnapshot']['compositingPolicy'].update(executed=False),
                        lambda b:b['productionSnapshot']['compositingPolicy'].update(appliedMode='OLD'),
                        lambda b:b['productionSnapshot']['compositingPolicy'].update(mathVersion='wrong'),
                        lambda b:b['productionSnapshot']['productionIdentity']['productionFrameIdentity'].update(compositingMode='OLD'),
                        lambda b:b['captureSummary']['captureExpectationContract']['freshIdentity'].update(productionSourceRequestIdentity='stale')):
                        altered=copy.deepcopy(records);mutation(altered['browser_result'])
                        with self.assertRaises(ValueError):compare.capture_binding(altered,value,c.file_id(png))
                    for name in compare.SUFFIXES:
                        if name!='browser_result':flow.save_json(images/(value['prefix']+'_'+name+'.json'),records[name])
                    records['browser_result']['comparisonSupport']['json_members']={
                        value['prefix']+'_'+n+'.json':c.file_id(images/(value['prefix']+'_'+n+'.json'))
                        for n in compare.SUFFIXES if n!='browser_result'}
                    flow.save_json(images/(value['prefix']+'_browser_result.json'),records['browser_result'])
                    output=root/(value['mode']['mode']+'.json')
                    args=['compare','--candidate',str(generated/value['mode']['mode'].lower()/'candidate.json'),
                          '--captures',str(images),'--reference',str(ref),'--output',str(output)]
                    flow.cli(compare.main,args)
                    result=c.read_json(output)
                    self.assertEqual(result['mode'],value['mode']['mode'])
                    self.assertEqual(result['whole']['maxAbsError'],2-i)
                    self.assertFalse(result['accepted'])
                    with self.assertRaisesRegex(ValueError,'output_exists'):flow.cli(compare.main,args)
                output=root/'pair.json'
                flow.cli(compare.main,['compare','--candidate',str(generated/'new-off/candidate.json'),
                    '--on-candidate',str(generated/'new-on/candidate.json'),'--captures',str(images),
                    '--reference',str(ref),'--output',str(output)])
                result=c.read_json(output)
                self.assertTrue(result['comparison_established']);self.assertFalse(result['accepted'])
                self.assertEqual(result['whole']['maxAbsError'],1)
                self.assertEqual(result['status'],'inconclusive')
                signed=np.load(root/'pair.json.images/signed-lsb.npy',allow_pickle=False)
                self.assertEqual(signed[0,0,0],1);self.assertEqual(np.count_nonzero(signed),1)
                bad=copy.deepcopy(candidates[1]);bad['input']['selection']['count']+=1
                with self.assertRaisesRegex(ValueError,'pair_input'):compare.compare_pair(candidates[0],bad,images,ref)
                # Run the actual generated wrapper and capture, save their JSON,
                # then use exactly those saved files in the real offline CLI.
                bridge=root/'vm-bridge.json'
                flow.save_json(bridge,dict(generated=str(generated),publication=str(root/'vm-publication'),
                    input=candidates[0]['input'],pngInputs=str(images),sourceRoot=str(c.VIEWER/'demo')))
                self.node('wrapper',['--experimental-vm-modules','tools/smoke_step122_aligned_wrapper.mjs',str(root/'vm-run'),str(bridge)])
                output_vm=root/'vm-pair.json';captures_vm=root/'vm-run/off-on'
                flow.cli(compare.main,['compare','--candidate',str(generated/'new-off/candidate.json'),
                    '--on-candidate',str(generated/'new-on/candidate.json'),'--captures',str(captures_vm),
                    '--reference',str(ref),'--output',str(output_vm)])
                self.assertTrue(c.read_json(output_vm)['comparison_established'])
                self.assertEqual(c.read_json(output_vm)['whole']['maxAbsError'],1)
                for case in ('missing-json','missing-png','json-bytes','png-bytes','mixed-browser'):
                    broken=root/('negative-'+case);shutil.copytree(captures_vm,broken)
                    prefix=candidates[0]['prefix']
                    if case=='missing-json':(broken/(prefix+'_webgpu_visible_record_lineage.json')).unlink()
                    if case=='missing-png':(broken/(prefix+'_canvas.png')).unlink()
                    if case=='json-bytes':
                        p=broken/(prefix+'_limited_draw_summary.json');p.write_bytes(p.read_bytes()+b' ')
                    if case=='png-bytes':
                        p=broken/(prefix+'_canvas.png');p.write_bytes(p.read_bytes()+b'changed')
                    if case=='mixed-browser':
                        shutil.copyfile(broken/(candidates[1]['prefix']+'_browser_result.json'),broken/(prefix+'_browser_result.json'))
                    with self.subTest(case=case),self.assertRaises((ValueError,FileNotFoundError)):
                        compare.compare_pair(*candidates,broken,ref)
                # Individually valid captures must still be rejected as a pair
                # if they reuse the request or fail to advance the generation.
                for case in ('same-request','same-generation'):
                    broken=root/('negative-'+case);shutil.copytree(captures_vm,broken)
                    off=c.read_json(broken/(candidates[0]['prefix']+'_browser_result.json'))
                    prefix=candidates[1]['prefix']
                    saved={n:c.read_json(broken/(prefix+'_'+n+'.json')) for n in compare.SUFFIXES}
                    a=off['captureSummary']['captureExpectationContract']['freshIdentity']
                    b=saved['browser_result']['captureSummary']['captureExpectationContract']['freshIdentity']
                    previous_request=b['requestIdentity'];previous_generation=b['productionGeneration']
                    generation_keys={'generation','productionGeneration','presentedGeneration','compositorGeneration',
                        'productionOutputGeneration','presentedOutputGeneration','capturedOutputGeneration'}
                    def change(value):
                        if isinstance(value,dict):
                            return {k:(a['productionGeneration'] if case=='same-generation' and k in generation_keys and v==previous_generation
                                else a['baselineProductionGeneration'] if case=='same-generation' and k=='baselineProductionGeneration'
                                else change(v)) for k,v in value.items()}
                        if isinstance(value,list):return [change(v) for v in value]
                        return a['requestIdentity'] if case=='same-request' and value==previous_request else value
                    saved=change(saved)
                    for n in compare.SUFFIXES:
                        if n!='browser_result':flow.save_json(broken/(prefix+'_'+n+'.json'),saved[n])
                    saved['browser_result']['comparisonSupport']['json_members']={
                        prefix+'_'+n+'.json':c.file_id(broken/(prefix+'_'+n+'.json')) for n in compare.SUFFIXES if n!='browser_result'}
                    flow.save_json(broken/(prefix+'_browser_result.json'),saved['browser_result'])
                    compare.compare(candidates[1],broken,ref)
                    with self.subTest(case=case),self.assertRaisesRegex(ValueError,'pair_fresh_request_generation'):
                        compare.compare_pair(*candidates,broken,ref)
                self.assertEqual(aligned.reference_binding(ref),original_ids)


if __name__=='__main__':unittest.main(verbosity=2)
