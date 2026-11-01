"""Offline comparison: identity gates precede RGB decoding; never accepts a Step."""
import argparse
from pathlib import Path

import adopted_range_contract as c
from adopted_range_contract import require, keys, read_json, file_id, canonical

SUFFIXES = ('browser_result', 'webgpu_visible_record_dryrun_compare',
            'webgpu_visible_record_dryrun_capture_status', 'gpu_candidate_runtime_summary',
            'limited_draw_summary', 'png_capture_status', 'webgpu_visible_record_lineage')
FRAME = ('generation', 'datasetCameraLabel', 'datasetFrameNumber', 'datasetTime',
         'referenceCameraLabel', 'outputWidth', 'outputHeight')


def same_frames(frames, fixed):
    base = frames[0]
    require(type(base) is dict and all(base.get(k) is not None for k in FRAME), 'missing_frame')
    require(type(base['generation']) is int and base['generation'] > 0, 'generation')
    for frame in frames:
        require(type(frame) is dict and all(frame.get(k) == base[k] for k in FRAME), 'stale_frame')
    observed = [f.get('frameHash') for f in frames if f.get('frameHash') is not None]
    require(all(type(h) is str and h for h in observed) and len(set(observed)) <= 1, 'frame_hash_conflict')
    rep = fixed['frame']; cam = fixed['camera']['values']
    require([base['datasetCameraLabel'], base['datasetFrameNumber'], base['datasetTime'],
             base['referenceCameraLabel'], base['outputWidth'], base['outputHeight']] ==
            [rep[2], rep[1], rep[3], rep[2], cam['width'], cam['height']], 'fixed_frame')


def capture_binding(records, candidate, png_identity):
    """Validate saved legacy observations, not just a claimed ready flag."""
    require(set(records) == set(SUFFIXES), 'seven_json')
    browser = records['browser_result']; png = records['png_capture_status']
    require(browser['failure'] is None and browser['captureFailure'] is None, 'recorded_failure')
    require(set(browser['checks']) == {'adoptedInput', 'capture', 'semantic', 'coverage',
                'diagnosticIndependent', 'cleanup'} and all(v is True for v in browser['checks'].values()),
            'browser_checks')
    evidence = browser['comparisonSupport']
    normal = candidate.get('schema') == 'viewer-normal-capture-candidate-v2'
    keys(evidence, 'schema candidate_identity mode delivery json_members' + (' initial_frame' if normal else ''))
    require(evidence['schema'] == ('viewer-normal-capture-support-v2' if normal else 'viewer-range-capture-support-v1') and
            evidence['candidate_identity'] == c.identity(canonical(candidate)) and
            evidence['delivery'] == candidate['source_delivery'], 'source_delivery_binding')
    mode = c.parse_mode(evidence['mode'])
    candidate_mode = c.parse_mode(candidate['mode'])
    require(mode['executed'] is True and mode['mode']==candidate_mode['mode'] and
            mode['schema']==candidate_mode['schema'] and mode['math_version']==candidate_mode['math_version'], 'mode_binding')
    # The legacy helper accepts a subset; complete CLI candidates are validated
    # separately. Dispatch on the strict mode schema, never a missing outer key.
    if mode['schema']=='viewer-old-source-bound-v1':
        require('schema' not in candidate or candidate['schema']=='viewer-old-capture-candidate-v1',
                'legacy_candidate_schema')
        require(mode==dict(candidate_mode,executed=True) and mode['mode']=='OLD', 'legacy_execution')
    else:
        require(candidate.get('schema')==('viewer-normal-capture-candidate-v2' if normal else
                                        'viewer-aligned-capture-candidate-v1') and mode['mode']!='OLD', 'aligned_execution')
    fixed = candidate['range_input']['fixed']; ctx = browser['inputContext']; snap = browser['productionSnapshot']
    require(ctx == snap['adoptedBundle'] == browser['pngBinding']['context'] and
            ctx['external'] == fixed['authority'] and ctx['selection'] == fixed['selection'] and
            ctx['camera'] == fixed['camera']['values'] and ctx['frame'] == fixed['frame'] and
            ctx['activeSh'] == [0,0] and ctx['maximumSh'] == [3,2] and ctx['physicalSh'] == 16 and
            ctx['rows'] == fixed['population']['rows'], 'adopted_binding')
    asset = snap['assetIdentity']['spl4AssetIdentity']; pop = fixed['population']
    require(asset['sha256'] == pop['identity']['sha256'] and asset['sizeBytes'] == pop['identity']['size']
            and asset['recordCount'] == pop['rows'], 'asset_binding')
    selection = fixed['selection']; work = snap['productionWorksetIdentity']
    require([work['residentStart'], work['residentRecordCount'], work['residentEndExclusive']] ==
            [selection['start'], selection['count'], selection['start']+selection['count']], 'range_binding')
    config = snap['appliedConfig']
    require(snap['projection']==fixed['viewer_projection'],'projection_binding')
    require(all(config[k] == v for k,v in dict(timestamp=fixed['frame'][3], scalingModifier=1,
            sigmaScale=1, prefilterVar=0, configuredRenderScale=1, effectiveRenderScale=1).items()), 'scale_binding')
    p = snap['productionIdentity']; generation = p['productionGeneration']
    require(snap['status'] == 'ready' and generation == p['presentedGeneration'] ==
            png['productionOutputGeneration'] == png['presentedOutputGeneration'] ==
            png['capturedOutputGeneration'], 'generation_chain')
    frames = [p['productionFrameIdentity'],p['presentedFrameIdentity'],
              png['presentedFrameIdentity'],png['capturedFrameIdentity']]
    summary = browser['captureSummary']; expectation = summary['captureExpectationContract']
    fresh = expectation['freshIdentity']
    require(fresh['ready'] is True and fresh['generationChainMatches'] is True and
            fresh['frameIdentityMatchesTarget'] is True and type(fresh['baselineProductionGeneration']) is int and
            fresh['baselineProductionGeneration'] < generation and
            fresh['productionGeneration'] == fresh['compositorGeneration'] == fresh['presentedGeneration'] == generation and
            type(fresh['requestIdentity']) is str and bool(fresh['requestIdentity']) and
            fresh['requestIdentity'] == fresh['productionSourceRequestIdentity'], 'request_chain')
    for name in ('webgpu_visible_record_dryrun_compare','webgpu_visible_record_lineage'):
        record = records[name]; provenance = record['artifactProvenance']
        require(record['artifactSetIdentity'] == provenance['capturePrefix'] == candidate['prefix'] and
                provenance['requestIdentity'] == fresh['requestIdentity'] and
                provenance['productionGeneration'] == generation, 'diagnostic_request')
        frames.append(provenance['frameIdentity'])
    same_frames(frames,fixed)
    if mode['mode']!='OLD':
        from aligned_range_capture import policy_evidence, MATH
        policy=snap.get('compositingPolicy')
        if normal:
            from normal_range_capture import policy_evidence as normal_policy, initial_evidence
            require(normal_policy(policy,fresh['requestIdentity'],generation,candidate['normal_scope'])==mode,
                    'normal_policy_binding')
            initial_evidence(evidence['initial_frame'],candidate,generation)
        else:
            require(policy_evidence(policy,mode['mode'],fresh['requestIdentity'],generation)==mode and
                policy['candidateIdentity']==c.identity(canonical(candidate))['sha256'], 'actual_policy_binding')
        for frame in [*frames,png['requestedStateIdentity']]:
            require(frame.get('compositingMathVersion')==MATH and frame.get('compositingMode')==mode['mode'] and
                frame.get('compositingRequestIdentity')==fresh['requestIdentity'], 'policy_frame_chain')
            if normal:
                require(frame.get('compositingPurpose')=='normal' and
                        frame.get('compositingScopeIdentity')==candidate['normal_scope']['identity'], 'normal_frame_chain')
    require(browser['captureStatus'] == png and summary['ready'] is True and
            expectation['ready'] is True and png['status'] == 'success' and
            png['captureMatchesPresentedFrame'] is True and png['captureMatchesRequestedState'] is True and
            png['captureFreshnessKnown'] is True and png['staleCaptureDetected'] is False, 'capture_not_ready')
    health = expectation['runtimeHealth']
    require(health['ready'] is True and all(health[k] is False for k in
            ('webgpuValidationErrorDetected','invalidCommandBufferDetected','queueSubmitFailureDetected','deviceLostDetected')),
            'runtime_failure')
    blob = png['captureBlobIdentity']; require(blob['mimeType'] == 'image/png' and
            dict(size=blob['sizeBytes'],sha256=blob['sha256']) == png_identity and
            blob['fileName'] == png['fileName'] == summary['png']['fileName'] == candidate['prefix']+'_canvas.png' and
            summary['png']['blobSha256'] == blob['sha256'], 'png_bytes')
    orientation = png['orientationEvidence']
    require(orientation['savedPngOrientation'] == 'production-texture-top-left-y-down' and
            orientation['captureVerticalFlipApplied'] is False and orientation['presentationVerticalFlipApplied'] is False and
            orientation['orientationMismatchClassification'] == 'none', 'orientation')
    require(png['captureAlphaPolicy'] == 'force-opaque-to-match-webgpu-canvas-presentation' and
            png['encodedPngPixelEvidence']['decodeCompleted'] is True, 'png_encoding')
    require(records['webgpu_visible_record_dryrun_compare']['comparison']['anyMismatch'] is False, 'diagnostic_mismatch')
    diagnostic_status=records['webgpu_visible_record_dryrun_capture_status']
    require(diagnostic_status['status']=='ok' and diagnostic_status['captureFatalError'] is False and
            diagnostic_status['canonicalDiagnosticSerializationSucceeded'] is True and
            diagnostic_status['detailedLineagePresent'] is True and
            diagnostic_status['detailedLineageSerializationSucceeded'] is True, 'diagnostic_download')
    coverage = browser['semanticComparison']['orchestrationResult']['coverage']
    require(coverage['coverageComplete'] is True and coverage['requestedChunkCount'] == coverage['completedChunkCount'] == 1 and
            coverage['processedRecordCount'] == coverage['uniqueSrcIndexCount'] == selection['count'] and
            coverage['firstSrcIndex'] == selection['start'] and coverage['lastSrcIndex'] == selection['start']+selection['count']-1,
            'coverage')
    return mode


def metrics(actual, expected):
    import numpy as np
    from compare_png import compute_pixel_metrics
    result = compute_pixel_metrics(actual,expected,'rgb')
    signed = actual[:,:,:3].astype(np.int16)-expected[:,:,:3].astype(np.int16)
    diff = np.abs(signed)
    result.update(p95=float(np.percentile(diff,95)),p99=float(np.percentile(diff,99)),
                  valuesOver1LSB=int((diff>1).sum()),pixelsOver1LSB=int((diff>1).any(axis=2).sum()),
                  signedBias=float(signed.mean()),saturatedActualPixels=int((actual[:,:,:3]==255).any(axis=2).sum()))
    return result


def evaluate(actual, expected, patches, mode, *, known_violation=False, evidence_complete=True):
    import numpy as np
    if known_violation:
        return dict(status='failed',accepted=False)
    if not evidence_complete:
        return dict(status='not-established',accepted=False)
    require(actual.dtype == expected.dtype == np.uint8 and actual.shape == expected.shape and
            actual.ndim == 3 and actual.shape[2] in (3,4), 'image_shape_dtype')
    whole = metrics(actual,expected); h,w = actual.shape[:2]; parts=[]
    for patch in patches:
        keys(patch,'source_index bounds')
        bounds=patch['bounds']
        if bounds is None:
            parts.append(dict(patch, status='outside-or-invisible')); continue
        require(type(bounds) is list and len(bounds)==4 and all(type(x) is int for x in bounds), 'patch_type')
        x0,y0,x1,y1=bounds
        require(0<=x0<x1<=w and 0<=y0<y1<=h,'patch_bounds')
        parts.append(dict(patch,metrics=metrics(actual[y0:y1,x0:x1],expected[y0:y1,x0:x1])))
    require(mode in ('OLD','NEW-OFF','NEW-ON','OFF-vs-ON'), 'comparison_mode')
    # OFF-vs-ON is descriptive; never applies the ON-vs-CUDA 1-LSB criterion.
    status = 'candidate-within-1LSB' if mode=='NEW-ON' and whole['maxAbsError']<=1 else 'inconclusive'
    return dict(status=status,accepted=False,whole=whole,patches=parts,mode=mode,
                visual_review_required=True,raw_equivalence_claimed=False)


def save_visuals(directory, actual, expected, patches):
    """Display-only fixed mapping. Raw RGB differences remain exact signed LSBs."""
    import numpy as np
    from PIL import Image
    from compare_png import build_absdiff_image
    directory=c.safe_path(directory)
    directory.mkdir()  # exclusive; leave partial evidence on failure
    signed=actual.astype(np.int16)-expected.astype(np.int16)
    absolute=np.abs(signed).astype(np.uint8)
    gain=8  # fixed before observing a residual; never normalized per image
    arrays={'signed-lsb.npy':signed,'absolute-lsb.npy':absolute}
    pictures={
        'absolute-x1.png':build_absdiff_image(absolute,1),
        'absolute-x8.png':build_absdiff_image(absolute,gain),
        'signed-x8.png':Image.fromarray(np.clip(128+gain*signed,0,255).astype(np.uint8),'RGB'),
        'capture-reference.png':Image.fromarray(np.concatenate((actual,expected),axis=1),'RGB'),
    }
    for name,array in arrays.items():
        with (directory/name).open('xb') as f:np.save(f,array,allow_pickle=False)
        read=np.load(directory/name,allow_pickle=False)
        require(read.dtype==array.dtype and np.array_equal(read,array),'diff_array_readback')
    for name,picture in pictures.items():
        with (directory/name).open('xb') as f:picture.save(f,format='PNG')
        with Image.open(directory/name) as read:
            require(read.mode==picture.mode and np.array_equal(np.asarray(read),np.asarray(picture)),
                    'diff_picture_readback')
    return dict(schema='viewer-range-visuals-v1',sign='capture-minus-reference',channels='RGB',
        units='uint8 LSB',orientation='top-left-y-down',width=int(actual.shape[1]),height=int(actual.shape[0]),
        signed_dtype='int16',absolute_dtype='uint8',fixed_gain=gain,
        signed_display='per RGB channel: clip(128 + 8 * signed LSB, 0, 255); zero=128',
        absolute_display='per RGB channel: clip(gain * absolute LSB, 0, 255); alpha=255',
        side_by_side='capture left, reference right; native scale; width=2*width',
        display_changes_evaluation=False,patches=patches,
        members={p.name:file_id(p) for p in sorted(directory.iterdir())})


def compare(candidate, captures, reference, *, visuals=None):
    from make_adopted_range_capture import validate_candidate
    validate_candidate(candidate, current=False)
    c.corrected_paths()
    import render_adopted_range_reference as rr
    reference=Path(reference); captures=Path(captures)
    receipt=read_json(reference/'receipt.json'); manifest=read_json(reference/'reference/manifest.json')
    require(receipt['manifest']==file_id(reference/'reference/manifest.json') and
            receipt['verification']==file_id(reference/'verification.json'), 'reference_receipt')
    require(rr.verify_saved(reference/'reference',manifest)==read_json(reference/'verification.json'), 'reference_verification')
    require(manifest['input']==dict(candidate['range_input'],execution_authorized=True), 'same_range_input')
    prefix=candidate['prefix']; records={n:read_json(captures/(prefix+'_'+n+'.json')) for n in SUFFIXES}
    inputs={n:file_id(captures/(prefix+'_'+n+'.json')) for n in SUFFIXES}
    png=captures/(prefix+'_canvas.png'); mode=capture_binding(records,candidate,file_id(png))
    png_identity=file_id(png)
    expected_members={prefix+'_'+n+'.json':file_id(captures/(prefix+'_'+n+'.json')) for n in SUFFIXES if n!='browser_result'}
    require(records['browser_result']['comparisonSupport']['json_members']==expected_members,'same_run_json_bytes')
    # Patch centers come from independent expected projections, BEFORE PNG decode.
    patches=candidate['patches']
    import numpy as np
    from PIL import Image
    with Image.open(png) as im:
        require(im.mode=='RGBA' and not im.info.get('icc_profile') and not im.info.get('gamma'), 'viewer_color_space')
        actual=np.asarray(im).copy(); require(bool((actual[:,:,3]==255).all()),'opaque_png')
    with Image.open(reference/'reference/view/render.png') as im:
        require(im.mode=='RGB' and not im.info.get('icc_profile') and not im.info.get('gamma'), 'reference_color_space')
        expected=np.asarray(im).copy()
    h,w=actual.shape[:2]; cam=candidate['range_input']['fixed']['camera']['values']
    require((w,h)==(cam['width'],cam['height']),'viewport')
    result=evaluate(actual[:,:,:3],expected,patches,mode['mode'])
    require(inputs=={n:file_id(captures/(prefix+'_'+n+'.json')) for n in SUFFIXES} and
            png_identity==file_id(png) and
            rr.verify_saved(reference/'reference',manifest)==read_json(reference/'verification.json'),
            'comparison_inputs_changed')
    result['identities']=dict(reference=receipt,inputs=inputs,png=png_identity)
    if visuals is not None:
        result['visuals']=save_visuals(visuals,actual[:,:,:3],expected,patches)
    return result


def compare_pair(off, on, captures, reference, *, visuals=None):
    """Both complete captures must pass the individual CUDA/binding consumer."""
    require(off.get('schema')==on.get('schema')=='viewer-aligned-capture-candidate-v1', 'pair_schema')
    require(c.parse_mode(off['mode'])['mode']=='NEW-OFF' and
            c.parse_mode(on['mode'])['mode']=='NEW-ON', 'pair_modes')
    for field in ('range_input','input','source_delivery','viewer_dependencies',
                  'cuda_reference','old_candidate','patches','patch_origin'):
        require(off[field]==on[field], 'pair_input:'+field)
    require(off['prefix']!=on['prefix'], 'pair_prefix')
    individual=[compare(value,captures,reference) for value in (off,on)]
    snapshots=[read_json(Path(captures)/(value['prefix']+'_browser_result.json')) for value in (off,on)]
    fresh=[b['captureSummary']['captureExpectationContract']['freshIdentity'] for b in snapshots]
    require(fresh[0]['requestIdentity']!=fresh[1]['requestIdentity'] and
            fresh[0]['productionGeneration']<fresh[1]['productionGeneration'], 'pair_fresh_request_generation')
    import numpy as np
    from PIL import Image
    images=[]
    for value,checked in zip((off,on),individual):
        png=Path(captures)/(value['prefix']+'_canvas.png')
        require(file_id(png)==checked['identities']['png'], 'pair_png_changed')
        with Image.open(png) as im: images.append(np.asarray(im.convert('RGB')).copy())
        require(file_id(png)==checked['identities']['png'], 'pair_png_changed')
        require(checked['identities']['inputs']=={n:file_id(Path(captures)/(value['prefix']+'_'+n+'.json'))
                for n in SUFFIXES}, 'pair_records_changed')
    result=evaluate(images[1],images[0],off['patches'],'OFF-vs-ON')
    result.update(comparison_established=True,accepted=False,normal_on_accepted=False,
        sign='NEW-ON-minus-NEW-OFF',individual_cuda=individual,
        identities={'off_candidate':c.identity(canonical(off)), 'on_candidate':c.identity(canonical(on)),
                    'requests':[f['requestIdentity'] for f in fresh],
                    'generations':[f['productionGeneration'] for f in fresh]})
    if visuals is not None:
        result['visuals']=save_visuals(visuals,images[1],images[0],off['patches'])
        result['visuals'].update(sign='NEW-ON-minus-NEW-OFF',
            side_by_side='NEW-ON left, NEW-OFF right; native scale; width=2*width')
    return result


def patch_plan(lineage,fixed):
    """Freeze seven saved projection locations before the new image is available."""
    import math
    reps=fixed['selection']['representatives']; rows=lineage['records']
    require(len(rows)==len(reps) and {r['srcIndex'] for r in rows}==set(reps),'patch_representatives')
    w,h=fixed['camera']['values']['width'],fixed['camera']['values']['height']; result=[]
    for index in reps:
        row=next(r for r in rows if r['srcIndex']==index)
        valid=row['valid']; x,y=row['screenCenter']
        require(type(valid) in (bool,int) and valid in (0,1) and math.isfinite(x) and math.isfinite(y),'patch_projection')
        bounds=None
        if valid and 0<=x<w and 0<=y<h:
            cx,cy=math.floor(x),math.floor(y)
            bounds=[max(0,cx-8),max(0,cy-8),min(w,cx+9),min(h,cy+9)]
        result.append(dict(source_index=index,bounds=bounds))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','captures','reference','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--on-candidate',help='Pair comparison: --candidate must be NEW-OFF; this must be NEW-ON.')
    a=p.parse_args()
    output=c.safe_path(a.output)
    images=output.with_name(output.name+'.images')
    pending=output.with_name(output.name+'.pending-images')
    require(not output.exists() and not images.exists() and not pending.exists(),'comparison_output_exists')
    try:
        if a.on_candidate:
            result=compare_pair(read_json(a.candidate),read_json(a.on_candidate),a.captures,a.reference,visuals=pending)
        else:
            result=compare(read_json(a.candidate),a.captures,a.reference,visuals=pending)
        # Reuse the same no-replace publication primitive as the reference.
        import adopted_reference_bundle as bundle
        bundle.rename_new(pending,images)
        require(result['visuals']['members']=={p.name:file_id(p) for p in images.iterdir()},'visual_publication')
        result['visuals']['directory']=images.name
    except FileNotFoundError as error:result=dict(status='not-established',accepted=False,reason=str(error))
    except (ValueError,KeyError,TypeError) as error:result=dict(status='failed',accepted=False,reason=str(error))
    with output.open('xb') as f:f.write(canonical(result))
    require(read_json(a.output)==result,'comparison_readback')
    print(result['status'])


if __name__=='__main__':main()
