"""New Viewer provenance bound separately from immutable CUDA provenance.

Candidates are NOT execution grants. No prepare, browser, GPU or native calls.
"""
import argparse
from pathlib import Path
import sys
import adopted_range_contract as c
from adopted_range_contract import require, read_json, file_id, canonical

MATH = 'cuda-aligned-composite-v1'
SCHEMA = 'viewer-aligned-capture-candidate-v1'
POLICY = 'adopted-range-compositing-policy-v1'
REFERENCE = c.REPORTS/'step122-compare-support-implement2/range-reference-run'
OLD = c.REPORTS/'step122-compare-support-implement2/old-capture-candidate/candidate.json'

def reference_binding(root):
    return {name:file_id(Path(root)/path) for name,path in
        [('receipt','receipt.json'),('manifest','reference/manifest.json'),('verification','verification.json')]}

def validate_candidate(value, *, current=True, _normal=False):
    from make_adopted_range_capture import validate_candidate as old_validate, source_delivery
    c.keys(value, 'schema execution_authorized review_required prefix mode range_input input source_delivery patches patch_origin generator_argv preparation viewer_dependencies cuda_reference old_candidate' +
           (' normal_scope' if _normal else ''))
    require(value['schema']==('viewer-normal-capture-candidate-v2' if _normal else SCHEMA) and value['execution_authorized'] is False and value['review_required'] is True,
        'aligned_candidate')
    mode=c.parse_mode(value['mode'])
    require(mode['schema']=='viewer-range-policy-evidence-v1' and not mode['executed'] and
        mode['terminated_pixel_count'] is None, 'unexecuted_candidate')
    require(value['prefix']==('step122_normal_on_000118_v16' if _normal else
                            'step122_'+mode['mode'].lower().replace('-','_')+'_000118_v16'), 'new_prefix')
    old=value['old_candidate']; c.keys(old,'path identity')
    require(Path(old['path'])==OLD and old['identity']==file_id(OLD), 'old_candidate_identity')
    historical=read_json(OLD); old_validate(historical,current=False)
    for key in ('range_input','input','patches','patch_origin'):
        require(value[key]==historical[key], 'immutable_reference_input:'+key)
    ref=value['cuda_reference']; c.keys(ref,'path identities')
    require(Path(ref['path'])==REFERENCE and ref['identities']==reference_binding(REFERENCE), 'cuda_reference_identity')
    manifest=read_json(REFERENCE/'reference/manifest.json'); receipt=read_json(REFERENCE/'receipt.json')
    require(manifest['input']==dict(historical['range_input'],execution_authorized=True) and
        receipt['manifest']==ref['identities']['manifest'] and receipt['verification']==ref['identities']['verification'],
        'immutable_cuda_provenance')
    if _normal:
        from normal_range_capture import generator_args as normal_args
        expected_args = normal_args()
    else:
        expected_args = generator_args(mode['mode'])
    require(value['generator_argv']==expected_args, 'generator_arguments')
    if _normal:
        from normal_range_capture import preparation_plan
        require(value['preparation']==preparation_plan(value['preparation']['directory']), 'normal_preparation')
    else:
        c.keys(value['preparation'],'directory delivery_update_required execution_authorized')
    require(c.safe_path(value['preparation']['directory']).is_relative_to(c.REPORTS) and
        value['preparation']['delivery_update_required'] is True and value['preparation']['execution_authorized'] is False,
        'unapproved_preparation')
    require(bool(value['source_delivery']) and all(k.startswith('/js/') and set(v)=={'size','sha256'}
        for k,v in value['source_delivery'].items()),'new_delivery_membership')
    # NEW dependencies are never substituted into manifest.input.dependencies.
    for url, identity in value['source_delivery'].items():
        require(value['viewer_dependencies'].get(str(c.VIEWER/'demo'/url.lstrip('/')))==identity, 'viewer_source_binding')
    if current:
        require(value['range_input']['fixed']==c.fixed_authority(), 'fixed_input_changed')
        require(value['viewer_dependencies']==c.support_identity() and value['source_delivery']==source_delivery(),
            'current_viewer_dependencies')

def generator_args(mode):
    from make_adopted_range_capture import generator_args as old_args
    argv=old_args()
    argv[argv.index('--step')+1]='step122_'+mode.lower().replace('-','_')+'_000118_v16'
    argv[argv.index('--phase-step')+1]='viewer-step122-'+mode.lower()
    return argv

def policy_evidence(policy, mode, request, generation):
    """Actual submitted/completed evidence, not the requested label."""
    require(type(policy) is dict and policy.get('schema')==POLICY and policy.get('mathVersion')==MATH and
        policy.get('purpose')=='validation-only' and policy.get('requestedMode')==policy.get('appliedMode')==mode and
        policy.get('requestIdentity')==request and type(request) is str and bool(request) and
        type(generation) is int and generation>0 and policy.get('executed') is True and
        policy.get('onPathExecuted') is (mode=='NEW-ON') and policy.get('normalOnAccepted') is False and
        policy.get('accepted') is False and policy.get('offOnComparisonComplete') is False and
        policy.get('evaluatedPixelReferenceCount') is None and policy.get('skippedPixelReferenceCount') is None,
        'applied_compositing_policy')
    n=policy.get('terminatedPixelCount')
    require(policy.get('terminationObserved') is (type(n) is int and n>0), 'termination_observation')
    return c.parse_mode(dict(schema='viewer-range-policy-evidence-v1', mode=mode, math_version=MATH,
        executed=True,terminated_pixel_count=n))

def wrapper(value):
    from make_adopted_range_capture import wrapper as old_wrapper
    name='step122-'+value['mode']['mode'].lower()
    text=old_wrapper(value,name=name,prefix=value['prefix'])
    text=text.replace('export async function run(authorization) {', 'export async function run(authorization, sharedRaw = null) {')
    text=text.replace('  let raw = null;', '  let raw = sharedRaw;')
    text=text.replace("authorization?.mode !== 'OLD'", "authorization?.mode !== candidate.mode.mode")
    text=text.replace('context.rows === 3248901', 'context.rows === candidate.range_input.fixed.population.rows')
    text=text.replace('mode:{...candidate.mode,executed:true}', 'mode:{...candidate.mode}')
    anchor='    const originalDownload = debug.downloadJsonDebug;'
    insert='''    const originalSchedule = debug.scheduleRender;
    const originalPng = debug.saveCurrentCanvasPng;
    let scheduledRequest = null;
    const policyRequest = {schema:'adopted-range-compositing-policy-v1', mathVersion:'cuda-aligned-composite-v1',
      mode:candidate.mode.mode, authority:candidate.input.authority,
      authorization:{...authorization,schema:'adopted-range-execution-authorization-v1',purpose:'validation-only'}};
    debug.scheduleRender = async options => {
      if (scheduledRequest || options?.forceProductionUpdate !== true) throw new Error('Exactly one fresh policy request required');
      scheduledRequest = await originalSchedule.call(debug, {...options,metadata:{...options.metadata,compositingPolicyRequest:policyRequest}});
      return scheduledRequest;
    };
    debug.saveCurrentCanvasPng = options => originalPng.call(debug,{...options,
      compositingMathVersion:policyRequest.mathVersion,compositingMode:policyRequest.mode,
      compositingRequestIdentity:scheduledRequest?.requestIdentity});
'''
    require(text.count(anchor)==1,'wrapper_schedule_anchor');text=text.replace(anchor,insert+anchor)
    text=text.replace('debug.downloadJsonDebug = originalDownload;','debug.downloadJsonDebug = originalDownload; debug.scheduleRender = originalSchedule; debug.saveCurrentCanvasPng = originalPng;')
    # Reuse the verified scene in ON; do not issue another load/render implicitly.
    text=text.replace('raw = await debug.loadAdoptedBundle(input);','if (raw === null) raw = await debug.loadAdoptedBundle(input);')
    anchor="    phase = 'completed-awaiting-user-visual-review';"
    checked='''    const applied = snapshot?.compositingPolicy;
    const frame = snapshot?.productionIdentity?.productionFrameIdentity;
    if (applied?.schema !== policyRequest.schema || applied?.mathVersion !== policyRequest.mathVersion ||
        applied?.requestedMode !== policyRequest.mode || applied?.appliedMode !== policyRequest.mode ||
        applied?.purpose !== 'validation-only' || applied?.executed !== true ||
        applied?.candidateIdentity !== authorization.candidate_sha256 ||
        applied?.requestIdentity !== scheduledRequest?.requestIdentity ||
        frame?.compositingRequestIdentity !== scheduledRequest?.requestIdentity ||
        frame?.compositingMathVersion !== policyRequest.mathVersion || frame?.compositingMode !== policyRequest.mode ||
        !Number.isSafeInteger(applied?.terminatedPixelCount) || applied.terminatedPixelCount < 0 ||
        applied.terminatedPixelCount > 1280*720 || (policyRequest.mode === 'NEW-OFF' && applied.terminatedPixelCount !== 0))
      throw new Error('Applied numerical policy evidence unavailable');
    comparisonSupport.mode = {...candidate.mode,executed:true,terminated_pixel_count:applied.terminatedPixelCount};
    const expectedMembers = ['webgpu_visible_record_dryrun_compare','webgpu_visible_record_dryrun_capture_status',
      'gpu_candidate_runtime_summary','limited_draw_summary','png_capture_status','webgpu_visible_record_lineage']
      .map(s => candidate.prefix+'_'+s+'.json').sort();
    if (JSON.stringify(Object.keys(comparisonSupport.json_members).sort()) !== JSON.stringify(expectedMembers) ||
        capture.genericProductionPngCaptureStatus?.status !== 'success' ||
        capture.genericProductionPngCaptureStatus?.captureBlobIdentity?.fileName !== candidate.prefix+'_canvas.png')
      throw new Error('Capture byte evidence incomplete');
'''
    require(text.count(anchor)==1,'wrapper_result_anchor');text=text.replace(anchor,checked+anchor)
    text=text.replace('  raw = null;','  const retainedRaw = raw; raw = null;')
    text=text.replace('    await debug.downloadJsonDebug(result, artifact);', '''    const written = await debug.downloadJsonDebug(result, artifact);
    const serialized = new TextEncoder().encode(JSON.stringify(result,null,2));
    if (written?.fileName !== artifact || written?.byteLength !== serialized.byteLength)
      throw new Error('Result download byte evidence unavailable');''')
    text=text.replace('  return text;', '  return {text,raw:retainedRaw,result,summary};')
    return text


def pair_wrapper():
    return b'''// UNAPPROVED. Download byte evidence is not offline saved-file verification.
let started=false;
export async function run(authorizations) {
  if(started) throw new Error('Already started; preserve evidence, no retry'); started=true;
  const off=await (await import('./new-off/step122-new-off-browser.js')).run(authorizations?.off);
  if(!off.summary.executionChecksPassed || !off.summary.resultDownloadRequested) return {off:off.summary,stopped:true};
  const on=await (await import('./new-on/step122-new-on-browser.js')).run(authorizations?.on,off.raw);
  if(!on.summary.executionChecksPassed || !on.summary.resultDownloadRequested) return {off:off.summary,on:on.summary,stopped:true};
  const a=off.result.productionSnapshot, b=on.result.productionSnapshot;
  if(a.productionIdentity.productionGeneration >= b.productionIdentity.productionGeneration ||
     a.requestIdentity.schedulerRequestIdentity === b.requestIdentity.schedulerRequestIdentity)
    throw new Error('Nonfresh ON');
  if(JSON.stringify(a.adoptedBundle)!==JSON.stringify(b.adoptedBundle) ||
     JSON.stringify(a.appliedConfig)!==JSON.stringify(b.appliedConfig) ||
     JSON.stringify(a.projection)!==JSON.stringify(b.projection)) throw new Error('Pair input mismatch');
  return {off:off.summary,on:on.summary,accepted:false,savedFilesVerificationRequired:true};
}
'''

def generate(directory):
    from make_adopted_range_capture import write, source_delivery
    import make_capture_commands as generator
    directory=c.safe_path(directory)
    require(directory.is_relative_to(c.REPORTS) and not directory.exists(), 'exclusive_candidate_directory')
    old=read_json(OLD)
    values=[]
    for mode in ('NEW-OFF','NEW-ON'):
        value={k:old[k] for k in ('range_input','input','patches','patch_origin')}
        value.update(schema=SCHEMA,execution_authorized=False,review_required=True,
            prefix='step122_'+mode.lower().replace('-','_')+'_000118_v16',
            mode=dict(schema='viewer-range-policy-evidence-v1',mode=mode,math_version=MATH,executed=False,terminated_pixel_count=None),
            source_delivery=source_delivery(),viewer_dependencies=c.support_identity(),
            cuda_reference=dict(path=str(REFERENCE),identities=reference_binding(REFERENCE)),
            old_candidate=dict(path=str(OLD),identity=file_id(OLD)),generator_argv=generator_args(mode),
            preparation=dict(directory=str(directory/mode.lower()),delivery_update_required=True,execution_authorized=False))
        validate_candidate(value);values.append(value)
    directory.mkdir()
    for value in values:
        mode=value['mode']['mode'].lower(); dest=directory/mode;dest.mkdir();name='step122-'+mode
        write(dest/'candidate.json',canonical(value));write(dest/(name+'-input.json'),canonical(value['input']))
        write(dest/(name+'-browser.js'),wrapper(value).encode())
        previous=sys.argv
        try:
            sys.argv=['make_capture_commands.py',*value['generator_argv']]
            capture=generator.build_commands(generator.parse_args())
        finally:sys.argv=previous
        capture+='\nexport { genericProductionCaptureResult, genericProductionPngCaptureStatus, genericProductionPngCaptureResult };\n'
        write(dest/(name+'-capture.js'),capture.encode())
        write(dest/'files.json',canonical(dict(members={p.name:file_id(p) for p in dest.iterdir()},execution_authorized=False)))
    write(directory/'run-off-on.js',pair_wrapper())
    write(directory/'files.json',canonical(dict(members={str(p.relative_to(directory)):file_id(p) for p in directory.rglob('*') if p.is_file()},
        execution_authorized=False,review_required=True,delivery_update_required=True)))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True)
    generate(parser.parse_args().directory)
    print('Unapproved OFF/ON candidates saved; source delivery and execution remain unauthorized.')
