"""One separately authorized selected-range forward; never a trainer/build entry."""
import argparse
from contextlib import closing
import importlib.util
import os
from pathlib import Path
import struct
import sys
from types import FunctionType, SimpleNamespace

import adopted_range_contract as contract
from adopted_range_contract import require, keys, file_id, read_json, canonical

SLOTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15)


def range_helper():
    # Caller must already be inside saved-native guard (or CPU native isolation).
    path = contract.CORRECTED/'tools/render_named_reference_from_ckpt.py'
    spec = importlib.util.spec_from_file_location('_viewer_range_selection', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_owned(loaded, selection):
    from staged_comparison_state import validate_state
    validate_state(loaded['state'], loaded['metadata'], loaded['config'])
    a = loaded['state']['capture']
    sliced, mapping = range_helper().select_checkpoint_source_range(
        a, start_inclusive=selection['start'], selected_count=selection['count'],
        expected_source_count=loaded['metadata']['rows'])
    b = list(sliced)
    for slot in SLOTS:
        b[slot] = b[slot].detach().contiguous().clone()
    # restore(None) never consumes Adam. Do not retain the unselected CPU state.
    b[11] = {}
    return tuple(b), mapping


def restore_range(capture, config, device):
    from scene.gaussian_model import GaussianModel
    from formal_checkpoint_state import model_contract
    model = GaussianModel(3, gaussian_dim=4, time_duration=config.time_derivation.effective_interval,
                          rot_4d=True, force_sh_3d=False, sh_degree_t=2, prefilter_var=-1.0)
    a = list(capture)
    require(len(a) == 19 and a[0] == a[18] == 0 and a[16] is True and a[17] is None,
            'range_model_scalars')
    for slot in SLOTS:
        a[slot] = a[slot].to(device).clone()
    model.restore(tuple(a), training_args=None)
    model.xyz_gradient_accum, model.t_gradient_accum, model.denom = a[8], a[9], a[10]
    model.percent_dense = config.optimization.population.percent_dense
    model._formal_time = config.time_derivation
    model_contract(model, config)
    require(model.optimizer is None and model._features_rest.shape[1:] == (15, 3), 'range_no_adam')
    return model


def check_selected_spl4(path, capture, selection, population):
    """Read all selected rows, not seven representatives or a full SPL4 copy.

    Offsets/activation semantics match adopted_reference_bundle.check_spl4.
    Full-file identity and range read share one checked descriptor.
    """
    import numpy as np
    from formal_checkpoint_identity import open_directory, checked_file, stream_identity
    start, n = selection['start'], selection['count']
    require(type(start) is type(n) is int and start >= 0 and n > 0 and
            start+n <= population['rows'] and len(capture[1]) == n, 'mapping_range')
    p = Path(path); fd = open_directory(str(p.parent))
    try:
        stream, before = checked_file(fd, p.name, population['identity']['size'])
        with stream:
            require(stream_identity(stream,before)==population['identity'],'full_spl4_identity')
            require(os.fstat(stream.fileno()).st_size == 128+260*population['rows'], 'spl4_size')
            header = stream.read(128)
            require(struct.unpack('<4sIIiiBBBB10I', header[:64]) ==
                    (b'SPL4',2,population['rows'],0,0,1,1,1,1,3,4,4,3,3,45,1,1,1,0)
                    and not any(header[64:]), 'spl4_header')
            stream.seek(128+260*start)
            for local in range(0, n, 8192):
                count = min(8192, n-local)
                raw = stream.read(count*260)
                require(len(raw) == count*260, 'spl4_short')
                rows = np.frombuffer(raw, '<f4').reshape(count,65)
                require(np.isfinite(rows).all(), 'spl4_nonfinite')
                for slot, offset, width in ((1,0,3),(5,3,4),(15,7,4),(4,11,3),(2,14,3),
                                            (3,17,45),(6,62,1),(13,63,1),(14,64,1)):
                    expected = capture[slot][local:local+count].detach().cpu().numpy().reshape(count,width)
                    require(rows[:,offset:offset+width].tobytes() == expected.astype('<f4').tobytes(),
                            'selected_spl4_correspondence')
            after = os.fstat(stream.fileno())
            require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) ==
                    (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns), 'spl4_changed')
    finally:
        os.close(fd)
    return dict(start=start, count=n, end=start+n, checked_rows=n,
                mapping='sourceIndex = rangeStart + localIndex')


def resources():
    import staged_comparison_support as s
    import adopted_reference_runtime as rt
    existing = s.resources()
    def clone(fn, **overrides):
        scope = dict(fn.__globals__, **overrides)
        result = FunctionType(fn.__code__, scope, fn.__name__, fn.__defaults__)
        result.__kwdefaults__ = fn.__kwdefaults__
        return result
    # Existing margin/free-space predicates; one range view, not 14 full views.
    memory = clone(existing.memory, demand=rt.incremental_demand)
    def disk_need(kind, n):
        require(kind == 'range', 'range_disk_kind')
        return rt.remaining_disk('view-13', n)
    return SimpleNamespace(memory=memory, disk=clone(existing.disk, disk_need=disk_need))


def png_readback(root, raw):
    import numpy as np
    from PIL import Image
    depth = raw['depth'][0]; lo, hi = depth.min(), depth.max()
    normalized = np.zeros_like(depth) if lo == hi else (depth-lo)/(hi-lo)
    images = dict(render=raw['render'], gt=raw['gt'], alpha=np.repeat(raw['alpha'],3,axis=0),
                  depth=np.repeat(normalized[None],3,axis=0))
    for name, value in images.items():
        expected = (np.clip(value,0,1).transpose(1,2,0)*255).astype(np.uint8)
        with Image.open(Path(root)/(name+'.png')) as image:
            require(image.mode == 'RGB' and np.array_equal(np.asarray(image),expected), 'png_quantization')


def verify_saved(root, manifest):
    """Independent CPU readback: no checkpoint, renderer or native access."""
    import numpy as np
    import adopted_reference_bundle as b
    import adopted_reference_runtime as rt
    root = Path(root)
    keys(manifest, 'schema kind input metadata mapping runtime environment encoding members')
    require(manifest['schema'] == 'viewer-range-reference-v1' and
            manifest['kind'] == 'selected-range-comparison-reference', 'range_schema')
    require(read_json(root/'manifest.json') == manifest, 'manifest_binding')
    expected = {'manifest.json', 'view/runtime.json', *(f'view/{n}.npy' for n in b.RAW),
                *(f'view/{n}.png' for n in b.PNG)}
    found = b.members(root)
    require(set(found) == expected, 'range_members')
    require({k:v for k,v in found.items() if k!='manifest.json'}==manifest['members'],'saved_member_identity')
    inp = manifest['input']; fixed = inp['fixed']; sel = fixed['selection']; n = sel['count']
    require(manifest['mapping'] == dict(start=sel['start'], count=n, end=sel['start']+n,
                checked_rows=n, mapping='sourceIndex = rangeStart + localIndex'), 'saved_mapping')
    record = read_json(root/'view/runtime.json')
    require(record == manifest['runtime'] and record['camera'] == fixed['camera'] and
            record['frame'] == fixed['frame'] and record['native']['population_rows'] == n and
            record['native']['call_count'] == 1, 'saved_runtime')
    h,w = fixed['camera']['values']['height'],fixed['camera']['values']['width']
    raw = {name:np.load(root/'view'/(name+'.npy'),allow_pickle=False) for name in b.RAW}
    rt.raw_contract(raw,n,h,w)
    require(manifest['encoding'] == b.ENCODING, 'encoding')
    png_readback(root/'view',raw)
    return dict(status='verified_requires_review', members=found, mapping=manifest['mapping'],
                image_parity_accepted=False)


def publish(root, manifest):
    import adopted_reference_bundle as b
    root = Path(root); pending = root/'.pending-range'
    verification = verify_saved(pending,manifest)
    b.write_json(root/'verification.json',verification)
    require(read_json(root/'verification.json') == verification, 'verification_readback')
    b.rename_new(pending,root/'reference')
    require(verify_saved(root/'reference',manifest) == verification, 'published_readback')
    receipt = dict(schema='viewer-range-receipt-v1', manifest=file_id(root/'reference/manifest.json'),
                   verification=file_id(root/'verification.json'), status='verified_requires_review')
    b.write_json(root/'receipt.json',receipt)
    return receipt


def execute(value, digest):
    # Every reject here precedes Torch/native/output creation.
    proof = contract.preflight(value,execute=True,expected=digest)
    import adopted_reference_bundle as b
    import adopted_reference_runtime as rt
    import staged_comparison_adopted_input as ai
    import staged_comparison_checkpoint as cp
    root = Path(value['output']); stage = 'claim'
    with closing(b.Claim(root)) as claim:
        try:
            res = resources(); fixed=value['fixed']; sel=fixed['selection']; n=sel['count']
            stage='cpu-load'; res.memory(root,'cpu_load',fixed['population']['rows']);res.disk(root,'range',n)
            loaded=cp.read_adopted_checkpoint(proof)
            require(loaded['marker_identity'] == fixed['checkpoint']['marker'], 'checkpoint_binding')
            import torch
            import numpy as np
            from PIL import __version__ as pillow_version
            from staged_comparison_worker import native
            from real_input_runtime import ConsumedFiles
            from real_input_cuda_acceptance import render_buffers
            from unittest.mock import patch
            old=read_json(ai.CANDIDATE); origin=old['bridge']['current_provenance']
            require(origin['environment'] == dict(executable=sys.executable,python=sys.version,
                    torch=str(torch.__version__),torch_cuda=str(torch.version.cuda)), 'environment')
            environment=dict(origin['environment'],numpy=np.__version__,pillow=pillow_version)
            stage='native'
            with native(origin) as (_,renderer):
                selected,_=select_owned(loaded,sel); meta=loaded['metadata'];c=loaded['config'];del loaded
                stage='selected-spl4'
                spl4=contract.BUNDLE_RUN/'bundle/population.splat4d'
                mapping=check_selected_spl4(spl4,selected,sel,fixed['population'])
                res.memory(root,'load',n)
                with patch.object(torch.optim.Adam,'__init__',side_effect=AssertionError('range_optimizer_forbidden')):
                    model=restore_range(selected,c,'cuda')
                del selected
                consumed=ConsumedFiles();saved_worker=read_json(ai.RUN/'worker-result.json')
                cam,gt=rt.get_view(c,fixed['frame'],consumed,saved_worker)
                require(rt.camera_record(cam) == fixed['camera'], 'camera_binding')
                observations=[]
                def observed(device_cam,*args,**kw):
                    with rt.observe_native(renderer.diff_gaussian_rasterization._C,device_cam,n) as records:
                        result=renderer.render(device_cam,*args,**kw)
                    observations.extend(records);return result
                stage='render';res.memory(root,'render',n)
                raw=render_buffers(observed,cam,model,c);raw['gt']=gt
                require(len(observations) == 1, 'single_forward')
                record=dict(schema='viewer-range-runtime-v1',camera=rt.camera_record(cam),frame=fixed['frame'],
                            native=observations[0],files=consumed.finish())
                stage='save';res.disk(root,'range',n);claim.check()
                pending=root/'.pending-range';pending.mkdir()
                rt.save_view(pending/'view',raw,record)
                del raw,model,cam,gt
            require(value['dependencies'] == contract.support_identity(), 'support_changed_during_run')
            ai.verify(old)
            manifest=dict(schema='viewer-range-reference-v1',kind='selected-range-comparison-reference',
                          input=value,metadata=meta,mapping=mapping,runtime=record,
                          environment=environment,encoding=b.ENCODING,members=b.members(pending))
            b.write_json(pending/'manifest.json',manifest)
            stage='independent-readback-publication'
            return publish(root,manifest)
        except BaseException as error:
            b.failure(root,stage,error)
            raise


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input');ap.add_argument('--execute',action='store_true')
    ap.add_argument('--expect-input-sha256')
    a=ap.parse_args();value=read_json(a.input)
    require(a.execute,'execute_flag_required_no_implicit_run')
    print(canonical(execute(value,a.expect_input_sha256)).decode())


if __name__ == '__main__':
    main()
