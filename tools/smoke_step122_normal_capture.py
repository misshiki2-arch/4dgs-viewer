"""Normal-schema saved-file consumer and real CLI, synthetic CPU pixels only."""
import copy
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
import adopted_range_contract as c
import aligned_range_capture as aligned
import normal_range_capture as normal
import make_adopted_range_capture as capture
import compare_adopted_range_capture as compare
import render_adopted_range_reference as reference
import smoke_adopted_range_flow as flow
import smoke_step122_aligned_capture as aligned_test
import summarize_step_json as summary
import prepare_adopted_range_capture as prepare
import make_capture_commands as generator
from verify_capture_command_contract import build_capture_command_contract

def capture_variants(root,candidate):
    """Fixture-local generated negatives and optional immutable before control."""
    expected=normal.capture_expectation(candidate)
    variants={};previous=sys.argv
    try:
        sys.argv=['make_capture_commands.py',*candidate['generator_argv']]
        args=generator.parse_args()
        edits={
            'expected-missing':lambda e:None,
            'expected-type':lambda e:(e['targetIdentity'].update(datasetFrameNumber='2588') or e),
            'expected-dimension':lambda e:(e['targetIdentity'].update(outputWidth=0) or e),
            'expected-scope':lambda e:(e.update(scopeIdentity=c.identity(b'wrong scope')['sha256']) or e),
            'expected-camera':lambda e:(e['targetIdentity'].update(datasetCameraLabel='wrong camera') or e),
        }
        suffix='\nexport { genericProductionCaptureResult, genericProductionPngCaptureStatus, genericProductionPngCaptureResult };\n'
        for name,edit in edits.items():
            path=root/(name+'-capture.js')
            with path.open('x') as f:f.write(generator.build_commands(args,normal_capture=edit(copy.deepcopy(expected)))+suffix)
            variants[name]=str(path)
        before=os.environ.get('STEP122_BEFORE_GENERATOR')
        if before:
            spec=importlib.util.spec_from_file_location('before_capture_generator',before)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            path=root/'before-capture.js'
            with path.open('x') as f:f.write(module.build_commands(args)+suffix)
            variants['before']=str(path)
    finally:sys.argv=previous
    return variants

class NormalCPU(flow.CPUCase):
    initial=flow.FlowCPU.initial
    bundle=flow.FlowCPU.bundle
    provenance_fixture=flow.FlowCPU.provenance_fixture
    final_fixture=flow.FlowCPU.final_fixture
    fixture=flow.FlowCPU.fixture
    records=flow.FlowCPU.records
    aligned_records=aligned_test.AlignedCPU.aligned_records
    node=aligned_test.AlignedCPU.node

    def setUp(self):
        super().setUp()
        destination=os.environ.get('STEP122_CPU_EVIDENCE')
        if destination:
            self.addCleanup(lambda:shutil.copytree(self.f.base,Path(destination)/('normal-'+self.f.base.name)))

    def test_generated_normal_wrapper_and_isolated_prepare(self):
        import numpy as np
        from PIL import Image
        with self.fixture() as (inp,calls,_):
            root=self.f.base
            # Same small verified publication bridge used by aligned consumer.
            request=root/'vm-bundle-request.json'
            flow.save_json(request,dict(output=str(root/'vm-publication'),fixed=inp['fixed'],
                spl4=str(c.BUNDLE_RUN/'bundle/population.splat4d')))
            self.node('bundle',['tools/step122_aligned_vm_bundle.mjs',str(request)])
            verified=c.read_json(root/'vm-publication/fixture-authority.json')
            inp['fixed'].update(authority=verified['authority'],viewer_projection=verified['projection'])
            current=c.read_json(c.PRIOR/'step21-validation3-input.json');current['authority']=verified['authority']
            flow.save_json(c.PRIOR/'step21-validation3-input.json',current)
            # Substitute only the fixed population datum in the synthetic seed
            # template. All generated wrapper/capture/control/binding code runs.
            template=c.PRIOR/'step21-validation3-browser.js'
            text=template.read_text();self.assertEqual(text.count('context.rows === 3248901'),1)
            template.write_text(text.replace('context.rows === 3248901',
                'context.rows === '+str(inp['fixed']['population']['rows'])))
            inp=c.candidate(root/'reference-output');old=capture.generate(root/'old',inp)
            authorized=dict(inp,execution_authorized=True)
            reference.execute(authorized,c.identity(c.canonical(authorized))['sha256'])
            data=dict(id='cpu-normal',input=old['input'],rows=inp['fixed']['population']['rows'],frame=inp['fixed']['frame'],
                mathVersion=aligned.MATH,depthOrder='positive-depth-ascending-source-index-ascending-v2',
                sortAlgorithm='bounded-global-bitonic-stage-v2-depth-ascending-source-index',
                authorization=dict(connectionImplementationAuthorized=True,browserExecutionAuthorized=False))
            scope=dict(profile=data['id'],identity=c.identity(c.canonical(data))['sha256'],data=data)
            with patch.object(aligned,'OLD',root/'old/candidate.json'),patch.object(aligned,'REFERENCE',Path(inp['output'])), \
                 patch.object(normal,'normal_scope',return_value=scope):
                candidate=normal.generate(root/'normal-candidate')
                contract=build_capture_command_contract(root/'normal-candidate'/(normal.NAME+'-capture.js'))
                flow.save_json(root/'normal-api-contract.json',contract)
                self.assertEqual(contract['decision'],'ready',contract)
                expected=normal.capture_expectation(candidate)
                self.assertEqual(expected['targetIdentity'],dict(datasetCameraLabel=inp['fixed']['frame'][2],
                    datasetFrameNumber=inp['fixed']['frame'][1],datasetTime=inp['fixed']['frame'][3],outputWidth=20,outputHeight=10))
                self.assertEqual(expected['candidateSha256'],c.file_id(root/'normal-candidate/candidate.json')['sha256'])
                variants=capture_variants(root,candidate)
                images=root/'normal-png-input';images.mkdir()
                pixels=np.full((10,20,4),127,np.uint8);pixels[:,:,3]=255
                Image.fromarray(pixels,'RGBA').save(images/(normal.PREFIX+'_canvas.png'))
                bridge=root/'normal-vm-bridge.json'
                flow.save_json(bridge,dict(normal=True,generated=str(root/'normal-candidate'),
                    captureVariants=variants,
                    publication=str(root/'vm-publication'),input=candidate['input'],pngInputs=str(images),sourceRoot=str(c.VIEWER/'demo')))
                self.node('normal-wrapper',['--experimental-vm-modules','tools/smoke_step122_aligned_wrapper.mjs',
                    str(root/'normal-vm-run'),str(bridge)])
                output=root/'normal-vm-comparison.json'
                flow.cli(compare.main,['compare','--candidate',str(root/'normal-candidate/candidate.json'),
                    '--captures',str(root/'normal-vm-run/off-on'),'--reference',inp['output'],'--output',str(output)])
                result=c.read_json(output)
                self.assertEqual(result['status'],'candidate-within-1LSB');self.assertEqual(result['whole']['maxAbsError'],0)
                self.assertFalse(result['accepted']);self.assertEqual(len(calls),1)

                # Exercise real normal prepare checks/writes only in an isolated
                # filesystem. Never touch the real build/demo or publication.
                viewer=root/'isolated-viewer';(viewer/'build/demo/js').mkdir(parents=True)
                shutil.copytree(c.VIEWER/'demo/js',viewer/'demo/js')
                for url in candidate['source_delivery']:
                    shutil.copyfile(viewer/'demo'/url.lstrip('/'),viewer/'build/demo'/url.lstrip('/'))
                (viewer/'build/demo/step21-validation3').symlink_to(c.PRIOR)
                (c.PRIOR/'publication').mkdir()
                for n in ('bundle','receipt.json','result.json','worker-exit.json','reader-exit.json'):
                    if n!='bundle' and not (c.BUNDLE_RUN/n).exists():flow.save_json(c.BUNDLE_RUN/n,{})
                    (c.PRIOR/'publication'/n).symlink_to(c.BUNDLE_RUN/n)
                dependencies=c.support_identity()
                dependencies={str(viewer/Path(p).relative_to(c.VIEWER)) if Path(p).is_relative_to(c.VIEWER) else p:v
                    for p,v in dependencies.items()}
                with patch.object(c,'VIEWER',viewer),patch.object(c,'support_identity',return_value=dependencies):
                    directory=root/'prepare-candidate'
                    # A distinct synthetic historical fixture must describe the
                    # isolated publication root too. Keep the earlier VM fixture
                    # intact; do not bypass the historical plan validator.
                    isolated_old=copy.deepcopy(old)
                    isolated_old['preparation']=prepare.plan(root/'isolated-old')
                    isolated_old_path=root/'isolated-old-candidate.json'
                    flow.save_json(isolated_old_path,isolated_old)
                    with patch.object(aligned,'OLD',isolated_old_path):
                        normal.generate(directory)
                        auth=dict(schema='viewer-normal-prepare-authorization-v2',execution_authorized=True,
                            candidate=c.file_id(directory/'candidate.json'),files=c.file_id(directory/'files.json'))
                        result=prepare.prepare(directory,auth)
                        self.assertEqual(c.read_json(directory/'prepare-result.json'),result)
                        self.assertTrue(Path(result['destination']).is_symlink())
                        self.assertEqual(Path(result['destination']).resolve(),directory)
                        self.assertFalse(result['browser_executed']);self.assertFalse(result['build'])
                        with self.assertRaisesRegex(ValueError,'prepare_repeated_or_extra'):prepare.prepare(directory,auth)

    def test_saved_normal_cli_and_negative_matrix(self):
        import numpy as np
        from PIL import Image
        with self.fixture() as (inp,calls,_):
            root=self.f.base;old=capture.generate(root/'old',inp)
            authorized=dict(inp,execution_authorized=True)
            reference.execute(authorized,c.identity(c.canonical(authorized))['sha256'])
            ref=Path(inp['output'])
            data=dict(id='cpu-normal',input=old['input'],rows=inp['fixed']['population']['rows'],frame=inp['fixed']['frame'],
                authorization=dict(connectionImplementationAuthorized=True,browserExecutionAuthorized=False))
            scope=dict(profile=data['id'],identity=c.identity(c.canonical(data))['sha256'],data=data)
            with patch.object(aligned,'OLD',root/'old/candidate.json'),patch.object(aligned,'REFERENCE',ref), \
                 patch.object(normal,'normal_scope',return_value=scope):
                candidate=normal.generate(root/'normal-candidate')
                capture.validate_candidate(candidate)
                text=(root/'normal-candidate'/ (normal.NAME+'-browser.js')).read_text()
                self.assertNotIn('debug.loadAdoptedBundle',text)
                self.assertNotIn('compositingPolicyRequest',text)
                self.assertIn('getNormalStartupEvidence',text)
                images=root/'normal-captures';images.mkdir()
                pixels=np.full((10,20,4),127,np.uint8);pixels[:,:,3]=255
                pixels[0,0,:3]=[128,126,127]  # independently fixed expected signed residual
                png=images/(normal.PREFIX+'_canvas.png');Image.fromarray(pixels,'RGBA').save(png)
                records=self.aligned_records(candidate,c.file_id(png),10)
                b=records['browser_result'];p=b['productionSnapshot']['compositingPolicy'];identity=b['productionSnapshot']['productionIdentity']
                p.update(schema=normal.POLICY,purpose='normal',normalApplicationAuthorized=True,
                    scopeIdentity=scope['identity'],profile=scope['profile'],candidateIdentity=None)
                evidence=b['comparisonSupport'];evidence.update(schema='viewer-normal-capture-support-v2')
                frames=[identity['productionFrameIdentity'],identity['presentedFrameIdentity']]
                frames += [records['png_capture_status'][k] for k in
                           ('presentedFrameIdentity','capturedFrameIdentity','requestedStateIdentity')]
                frames += [records[n]['artifactProvenance']['frameIdentity'] for n in
                           ('webgpu_visible_record_dryrun_compare','webgpu_visible_record_lineage')]
                for f in frames:f.update(compositingPurpose='normal',compositingScopeIdentity=scope['identity'])
                b['captureStatus']=copy.deepcopy(records['png_capture_status'])
                evidence['initial_frame']=dict(schema='adopted-normal-initial-frame-v1',profile=scope['profile'],
                    scopeIdentity=scope['identity'],requestIdentity=p['requestIdentity'],
                    productionIdentity=copy.deepcopy(identity),compositingPolicy=copy.deepcopy(p))
                self.assertEqual(compare.capture_binding(records,candidate,c.file_id(png))['mode'],'NEW-ON')
                for name in compare.SUFFIXES:
                    if name!='browser_result':flow.save_json(images/(normal.PREFIX+'_'+name+'.json'),records[name])
                evidence['json_members']={p.name:c.file_id(p) for p in images.glob('*.json')}
                flow.save_json(images/(normal.PREFIX+'_browser_result.json'),b)
                output=root/'normal-comparison.json'
                flow.cli(compare.main,['compare','--candidate',str(root/'normal-candidate/candidate.json'),
                    '--captures',str(images),'--reference',str(ref),'--output',str(output)])
                result=c.read_json(output)
                self.assertEqual(result['status'],'candidate-within-1LSB',result)
                self.assertEqual(result['whole']['maxAbsError'],1);self.assertEqual(result['whole']['valuesOver1LSB'],0)
                self.assertFalse(result['accepted']);self.assertNotIn('comparison_established',result)
                for name,edit in {
                    'mixed':lambda x:x['comparisonSupport'].update(schema='viewer-range-capture-support-v1'),
                    'unknown':lambda x:x['comparisonSupport'].update(schema='unknown'),
                    'authority':lambda x:x['productionSnapshot']['compositingPolicy'].pop('normalApplicationAuthorized'),
                    'purpose':lambda x:x['productionSnapshot']['productionIdentity']['productionFrameIdentity'].pop('compositingPurpose'),
                    'stale-first':lambda x:x['comparisonSupport']['initial_frame']['productionIdentity'].update(productionGeneration=11),
                    'initial-validation':lambda x:x['comparisonSupport']['initial_frame']['compositingPolicy'].update(purpose='validation-only'),
                    'initial-missing':lambda x:x['comparisonSupport'].pop('initial_frame'),
                    'initial-profile':lambda x:x['comparisonSupport']['initial_frame'].update(profile='wrong-profile'),
                    'initial-scope':lambda x:x['comparisonSupport']['initial_frame'].update(scopeIdentity='wrong-scope'),
                }.items():
                    altered=copy.deepcopy(records);edit(altered['browser_result'])
                    with self.subTest(name=name),self.assertRaises((ValueError,KeyError)):
                        compare.capture_binding(altered,candidate,c.file_id(png))
                contract=dict(compositingPolicy=p,compositingPolicyCapability=dict(version='adopted-normal-compositing-capability-v2',
                    executionReady=True,normalOnAccepted=False,comparisonComplete=False))
                observed=summary.build_step122_compositing_policy_summary({'webgpuTileListCompositorContract':contract})
                self.assertTrue(observed['executionReady']);self.assertFalse(observed['normalOnAccepted'])
                p['terminatedPixelCount']=0;p['terminationObserved']=False
                self.assertTrue(summary.build_step122_compositing_policy_summary({'webgpuTileListCompositorContract':contract})['executionReady'])
                p['terminatedPixelCount']=None
                self.assertFalse(summary.build_step122_compositing_policy_summary({'webgpuTileListCompositorContract':contract})['executionReady'])

class ProductionScope(unittest.TestCase):
    def test_data_only_profile_matches_accepted_authority(self):
        scope=normal.normal_scope();data=scope['data']
        old=c.read_json(aligned.OLD)
        self.assertEqual(data['input'],old['input'])
        self.assertEqual(data['rows'],old['range_input']['fixed']['population']['rows'])
        self.assertEqual(data['frame'],old['range_input']['fixed']['frame'])
        self.assertEqual(data['authorization']['identity'],c.file_id(data['authorization']['instruction']))
        self.assertFalse(data['authorization']['browserExecutionAuthorized'])

if __name__=='__main__':unittest.main(verbosity=2)
