"""Finite normal-entry capture candidate. Generating this grants no execution permission."""
import argparse
import json
import math
import re
import sys
from pathlib import Path
import adopted_range_contract as c
import aligned_range_capture as aligned
from adopted_range_contract import require, canonical, read_json, file_id

SCHEMA = 'viewer-normal-capture-candidate-v2'
POLICY = 'adopted-normal-compositing-policy-v2'
PREFIX = 'step122_normal_on_000118_v16'
NAME = 'step122-normal-on'

def normal_scope():
    text=(c.VIEWER/'demo/js/common_4dgs_normal_adopted_scope.js').read_text()
    data=json.JSONDecoder().raw_decode(text.split('export const NORMAL_ADOPTED_SCOPE = freeze(',1)[1])[0]
    digest=re.search(r"export const NORMAL_SCOPE_IDENTITY = '([a-f0-9]{64})';",text)
    require(digest is not None and c.identity(canonical(data))['sha256']==digest[1], 'scope_data_identity')
    return dict(profile=data['id'],identity=digest[1],data=data)

def generator_args():
    args=aligned.generator_args('NEW-ON')
    args[args.index('--step')+1]=PREFIX
    args[args.index('--phase-step')+1]='viewer-step122-normal-on'
    return args

def preparation_plan(directory):
    from prepare_adopted_range_capture import plan
    result=plan(directory)
    result['destination']=str(c.VIEWER/'build/demo'/NAME)
    result.update(delivery_update_required=True,execution_authorized=False)
    return result

def validate_candidate(value, *, current=True):
    aligned.validate_candidate(value,current=current,_normal=True)
    scope=normal_scope()
    require(value['normal_scope']==scope and value['mode']['mode']=='NEW-ON', 'normal_candidate_scope')
    data=scope['data']
    require(data['input']==value['input'] and data['rows']==value['range_input']['fixed']['population']['rows'] and
            data['frame']==value['range_input']['fixed']['frame'] and
            data['authorization']['connectionImplementationAuthorized'] is True and
            data['authorization']['browserExecutionAuthorized'] is False, 'normal_authority')

def policy_evidence(policy, request, generation, scope):
    require(type(policy) is dict and policy.get('schema')==POLICY and
            policy.get('purpose')=='normal' and policy.get('normalApplicationAuthorized') is True and
            policy.get('scopeIdentity')==scope['identity'] and policy.get('profile')==scope['profile'] and
            policy.get('candidateIdentity') is None, 'normal_policy_authority')
    # Common numeric/execution checks remain exactly the v1 requirements. The
    # caller has already selected the distinct normal schema; no fallback.
    checked=dict(policy,schema=aligned.POLICY,purpose='validation-only')
    return aligned.policy_evidence(checked,'NEW-ON',request,generation)

def initial_evidence(value, candidate, captured_generation):
    c.keys(value,'schema profile scopeIdentity requestIdentity productionIdentity compositingPolicy')
    scope=candidate['normal_scope'];p=value['productionIdentity'];g=p['productionGeneration']
    require(value['schema']=='adopted-normal-initial-frame-v1' and value['profile']==scope['profile'] and
            value['scopeIdentity']==scope['identity'] and type(g) is int and 0<g<=captured_generation and
            p['presentedGeneration']==g, 'normal_initial_frame')
    policy_evidence(value['compositingPolicy'],value['requestIdentity'],g,scope)
    for f in (p['productionFrameIdentity'],p['presentedFrameIdentity']):
        require(f.get('generation')==g and f.get('compositingPurpose')=='normal' and
                f.get('compositingScopeIdentity')==scope['identity'] and
                f.get('compositingMathVersion')==aligned.MATH and f.get('compositingMode')=='NEW-ON' and
                f.get('compositingRequestIdentity')==value['requestIdentity'], 'normal_initial_binding')

def wrapper(value):
    from make_adopted_range_capture import wrapper as base
    text=base(value,name=NAME,prefix=PREFIX)
    text=text.replace("authorization?.mode !== 'OLD'", "authorization?.mode !== 'NEW-ON' || authorization?.purpose !== 'normal-capture'")
    text=text.replace('!debug?.loadAdoptedBundle', '!debug?.getNormalAdoptedScene || !debug?.getNormalStartupEvidence')
    text=text.replace("schema:'viewer-range-capture-support-v1'", "schema:'viewer-normal-capture-support-v2'")
    text=text.replace('mode:{...candidate.mode,executed:true},delivery',
                      'mode:{...candidate.mode},initial_frame:debug.getNormalStartupEvidence(),delivery')
    text=text.replace('raw = await debug.loadAdoptedBundle(input);', '''raw = debug.getNormalAdoptedScene();
    if (!raw || !comparisonSupport.initial_frame) throw new Error('Normal startup not established; no debug load');''')
    anchor="    phase = 'completed-awaiting-user-visual-review';"
    insert='''    const p = snapshot?.compositingPolicy, first = comparisonSupport.initial_frame;
    if (p?.schema !== 'adopted-normal-compositing-policy-v2' || p?.purpose !== 'normal' ||
        p?.normalApplicationAuthorized !== true || p?.scopeIdentity !== candidate.normal_scope.identity ||
        p?.appliedMode !== 'NEW-ON' || p?.executed !== true ||
        first?.compositingPolicy?.purpose !== 'normal' || first?.compositingPolicy?.appliedMode !== 'NEW-ON')
      throw new Error('Normal policy unavailable');
    comparisonSupport.mode = {...candidate.mode,executed:true,terminated_pixel_count:p.terminatedPixelCount};
'''
    require(text.count(anchor)==1,'normal_wrapper_anchor')
    text=text.replace(anchor,insert+anchor)
    require('debug.loadAdoptedBundle' not in text and 'compositingPolicyRequest' not in text,'normal_no_injection')
    return text

def capture_expectation(value):
    """Independent fixed authority, not observed frame/PNG or URL defaults."""
    validate_candidate(value)
    fixed=value['range_input']['fixed'];frame=fixed['frame'];camera=fixed['camera']['values']
    target=dict(datasetCameraLabel=frame[2],datasetFrameNumber=frame[1],datasetTime=frame[3],
                outputWidth=camera['width'],outputHeight=camera['height'])
    require(type(frame[2]) is str and bool(frame[2]) and type(frame[1]) is int and frame[1]>=0 and
            type(frame[3]) in (int,float) and math.isfinite(frame[3]) and
            all(type(target[k]) is int and target[k]>0 for k in ('outputWidth','outputHeight')),
            'normal_capture_target')
    return dict(candidateUrl='/'+NAME+'/candidate.json',candidateSha256=c.identity(canonical(value))['sha256'],
                profile=value['normal_scope']['profile'],scopeIdentity=value['normal_scope']['identity'],targetIdentity=target)


def generate(directory):
    """For separately authorized future preparation; never invoked on real inputs by CPU tests."""
    from make_adopted_range_capture import write, source_delivery
    import make_capture_commands as generator
    directory=c.safe_path(directory)
    require(directory.is_relative_to(c.REPORTS) and not directory.exists(),'exclusive_candidate_directory')
    old=read_json(aligned.OLD)
    value={k:old[k] for k in ('range_input','input','patches','patch_origin')}
    value.update(schema=SCHEMA,execution_authorized=False,review_required=True,prefix=PREFIX,
        mode=dict(schema='viewer-range-policy-evidence-v1',mode='NEW-ON',math_version=aligned.MATH,
                  executed=False,terminated_pixel_count=None),normal_scope=normal_scope(),
        source_delivery=source_delivery(),viewer_dependencies=c.support_identity(),
        cuda_reference=dict(path=str(aligned.REFERENCE),identities=aligned.reference_binding(aligned.REFERENCE)),
        old_candidate=dict(path=str(aligned.OLD),identity=file_id(aligned.OLD)),generator_argv=generator_args(),
        preparation=preparation_plan(directory))
    expected=capture_expectation(value)
    directory.mkdir()
    write(directory/'candidate.json',canonical(value))
    write(directory/(NAME+'-input.json'),canonical(value['input']))
    write(directory/(NAME+'-browser.js'),wrapper(value).encode())
    previous=sys.argv
    try:
        sys.argv=['make_capture_commands.py',*value['generator_argv']]
        capture=generator.build_commands(generator.parse_args(),normal_capture=expected)
    finally:
        sys.argv=previous
    capture+='\nexport { genericProductionCaptureResult, genericProductionPngCaptureStatus, genericProductionPngCaptureResult };\n'
    write(directory/(NAME+'-capture.js'),capture.encode())
    write(directory/'files.json',canonical(dict(schema='viewer-normal-candidate-files-v2',
          members={p.name:file_id(p) for p in directory.iterdir()},
          execution_authorized=False,review_required=True,delivery_update_required=True)))
    return value

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True)
    generate(parser.parse_args().directory)
