"""Lightweight, fixed adopted-B comparison authority. No research imports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

VIEWER = Path(__file__).resolve().parents[1]
WORK = VIEWER.parent
CORRECTED = WORK / '4d-gaussian-splatting_corrected_v1'
REPORTS = WORK / 'reports/webgpu-4dgs-viewer/phase3/step122'
PRIOR = WORK / 'reports/corrected-4dgs/phase5/step21/step21-validation3'
BUNDLE_RUN = WORK / 'reports/corrected-4dgs/phase4/step20/step20-implement1/reference-run'
PYTHON = '/home/demo/miniconda3/envs/4dgs310/bin/python'
SCHEMA = 'viewer-adopted-range-input-v1'
LOCAL = ('adopted_range_contract.py', 'render_adopted_range_reference.py',
         'compare_adopted_range_capture.py', 'make_adopted_range_capture.py',
         'smoke_adopted_range_support.py', 'compare_png.py', 'make_capture_commands.py')
LOCAL += ('prepare_adopted_range_capture.py', 'smoke_adopted_range_flow.py')
LOCAL += ('aligned_range_capture.py',)
LOCAL += ('smoke_step122_aligned_capture.py', 'smoke_step122_aligned_wrapper.mjs',
          'step122_aligned_vm_bundle.mjs', 'summarize_step_json.py',
          'smoke_step122_policy_summary.py', 'smoke_step122_compositing_diagnostic.mjs',
          'smoke_step122_runtime_connection.mjs')
LOCAL += ('step122_sort_fixture.mjs', 'smoke_step122_sort_alignment.mjs')
LOCAL += ('normal_range_capture.py', 'smoke_step122_normal_on.mjs', 'smoke_step122_normal_capture.py')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def keys(value, names):
    require(type(value) is dict and set(value) == set(names.split()), 'fields:' + names)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def identity(data):
    return dict(size=len(data), sha256=hashlib.sha256(data).hexdigest())


def safe_path(path):
    p = Path(path)
    require(p.is_absolute() and '..' not in p.parts, 'absolute_path')
    require(all(not q.is_symlink() for q in (p, *p.parents)), 'symlink')
    return p


def file_id(path):
    p = safe_path(path)
    require(p.is_file(), 'file_missing')
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return dict(size=p.stat().st_size, sha256=h.hexdigest())


def read_json(path):
    p = safe_path(path)
    require(p.stat().st_size <= 32 * 1024**2, 'bounded_json')
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'duplicate_key')
            result[k] = v
        return result
    return json.loads(p.read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda s: require(False, 'nonfinite_json'))


def corrected_paths():
    for p in (CORRECTED, CORRECTED/'tests', CORRECTED/'tests/cuda'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def support_identity():
    # Mandatory membership, not a caller-supplied list that may omit a consumer.
    files = [VIEWER/'tools'/n for n in LOCAL]
    files += [CORRECTED/'tools/render_named_reference_from_ckpt.py']
    files += sorted((VIEWER/'demo/js').glob('*.js'))
    files += [PRIOR/('step21-validation3-' + n) for n in
              ('input.json', 'browser.js', 'capture.js', 'prepare.sh', 'preparation-evidence.json')]
    files += [WORK/'json/step21_validation3_000118_v16_webgpu_visible_record_lineage.json']
    files += [WORK/'json/step21_validation3_000118_v16_browser_result.json']
    return {str(p): file_id(p) for p in files}


def fixed_authority():
    """Only small saved JSON, never checkpoint/SPL4 payload or research PNG."""
    inp = read_json(PRIOR/'step21-validation3-input.json')
    root = BUNDLE_RUN/'bundle'
    for name in ('index', 'manifest', 'verification'):
        require(file_id(root/(name+'.json')) == inp['authority'][name], 'authority:'+name)
    for name in ('receipt', 'result'):
        require(file_id(BUNDLE_RUN/(name+'.json')) == inp['authority'][name], 'authority:'+name)
    manifest = read_json(root/'manifest.json')
    index = read_json(root/'index.json')
    require(index['members']['manifest.json'] == inp['authority']['manifest'] and
            index['members']['population.splat4d'] == inp['authority']['spl4'] ==
            manifest['population']['identity'], 'population_binding')
    require(manifest['schema'] == 'adopted-b-reference-bundle-v1' and
            manifest['population']['rows'] == 3248901 and
            manifest['population']['active_sh'] == [0, 0] and
            manifest['population']['maximum_sh_labels'] == [3, 2] and
            manifest['population']['physical_sh'] == 16, 'adopted_profile')
    sel = inp['selection']
    require(sel['view'] == 4 and sel['start'] == 524288 and sel['count'] == 65536 and
            len(set(sel['representatives'])) == 7, 'fixed_selection')
    ref = manifest['references'][sel['view']]
    saved = read_json(WORK/'json/step21_validation3_000118_v16_browser_result.json')
    require(saved['failure'] is None and saved['inputContext']['external']==inp['authority'] and
            saved['inputContext']['selection']==sel, 'approved_viewer_snapshot')
    require(ref['frame'] == ['train', 2588, 'images/000118_v16.png', 16.6] and
            ref['camera']['values']['width'] == 1280 and ref['camera']['values']['height'] == 720,
            'fixed_camera')
    return dict(authority=inp['authority'], selection=sel, population=manifest['population'],
                checkpoint=manifest['input']['selected'], camera=ref['camera'],
                frame=ref['frame'], encoding=manifest['encoding'], background=[0, 0, 0],
                viewer_projection=saved['productionSnapshot']['projection'],
                scale=dict(renderScale=1, scalingModifier=1, sigmaScale=1, prefilterVar=0))


def output_path(path):
    p = safe_path(path)
    require(p.is_relative_to(REPORTS) and p.parent.is_dir() and not p.exists(), 'exclusive_output')
    return p


def candidate(output):
    require(sys.executable == PYTHON, 'python')
    output_path(output)
    corrected_paths()
    import staged_comparison_adopted_input as ai
    original = read_json(ai.CANDIDATE)
    ai.verify(original, authorized=False)
    return dict(schema=SCHEMA, execution_authorized=False, review_required=True,
                python=PYTHON, output=str(output), fixed=fixed_authority(),
                adopted_candidate=dict(path=str(ai.CANDIDATE), identity=file_id(ai.CANDIDATE)),
                dependencies=support_identity())


def preflight(value, *, execute=False, expected=None):
    keys(value, 'schema execution_authorized review_required python output fixed adopted_candidate dependencies')
    require(value['schema'] == SCHEMA and value['python'] == sys.executable == PYTHON and
            value['review_required'] is True and type(value['execution_authorized']) is bool,
            'input_contract')
    if execute:
        require(value['execution_authorized'] is True and expected == identity(canonical(value))['sha256'],
                'separate_execution_authorization')
    output_path(value['output'])
    require(value['fixed'] == fixed_authority(), 'fixed_input_changed')
    require(value['dependencies'] == support_identity(), 'support_dependencies_changed')
    corrected_paths()
    import staged_comparison_adopted_input as ai
    require(value['adopted_candidate'] == dict(path=str(ai.CANDIDATE), identity=file_id(ai.CANDIDATE)),
            'adopted_candidate_changed')
    proof = ai.verify(read_json(ai.CANDIDATE), authorized=False)
    return proof


def parse_mode(value):
    """Evidence parser only. Never enables a production policy or fallback."""
    keys(value, 'schema mode math_version executed terminated_pixel_count')
    require(type(value['executed']) is bool, 'mode_execution')
    if value['schema'] == 'viewer-old-source-bound-v1':
        require(value['mode'] == 'OLD' and value['math_version'] is None and
                value['terminated_pixel_count'] is None, 'legacy_mode')
    else:
        require(value['schema'] == 'viewer-range-policy-evidence-v1' and
                value['math_version'] == 'cuda-aligned-composite-v1' and
                value['mode'] in ('NEW-OFF', 'NEW-ON'), 'unknown_mode')
        n = value['terminated_pixel_count']
        require((value['executed'] is False and n is None) or
                (value['executed'] is True and type(n) is int and 0 <= n <= 1280*720 and
                 (value['mode'] != 'NEW-OFF' or n == 0)), 'termination_count')
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    ap.add_argument('--candidate', required=True)
    a = ap.parse_args()
    value = candidate(a.output)
    dest = safe_path(a.candidate)
    require(dest.parent.is_relative_to(REPORTS), 'candidate_location')
    with dest.open('xb') as f:
        f.write(canonical(value))
    require(read_json(dest) == value, 'candidate_readback')
    print('Unapproved candidate saved; no checkpoint/native/render execution.')


if __name__ == '__main__':
    main()
