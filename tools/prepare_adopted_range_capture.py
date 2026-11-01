"""Separately authorized OLD publication only; never build, copy JS, or start a server."""
import argparse
import os
from pathlib import Path

import adopted_range_contract as c


def plan(directory):
    return dict(directory=str(directory),destination=str(c.VIEWER/'build/demo/step122-old-candidate'),
        existing_publication={
            str(c.VIEWER/'build/demo/step21-validation3'):str(c.PRIOR),
            **{str(c.PRIOR/'publication'/n):str(c.BUNDLE_RUN/n) for n in
               ('bundle','receipt.json','result.json','worker-exit.json','reader-exit.json')}},
        build=False,copy_source=False,start_server=False)


def prepare(directory, authorization):
    from make_adopted_range_capture import validate_candidate, NAME, write
    directory=c.safe_path(directory)
    value=c.read_json(directory/'candidate.json'); files=c.read_json(directory/'files.json')
    normal=value.get('schema')=='viewer-normal-capture-candidate-v2'
    if normal:
        from normal_range_capture import NAME, preparation_plan
        expected_plan=preparation_plan(directory)
    else:
        expected_plan=plan(directory)
    c.keys(authorization,'schema execution_authorized candidate files')
    c.require(authorization['schema']==('viewer-normal-prepare-authorization-v2' if normal else 'viewer-old-prepare-authorization-v1') and
        authorization['execution_authorized'] is True and
        authorization['candidate']==c.file_id(directory/'candidate.json') and
        authorization['files']==c.file_id(directory/'files.json'),'separate_prepare_authorization')
    validate_candidate(value,current=False)
    c.require(value['preparation']==expected_plan,'prepare_plan')
    # Reference generation may already have claimed its output. Do not require
    # that output absent here; this entry never authorizes or invokes generation.
    c.require(value['range_input']['fixed']==c.fixed_authority() and
              (value['viewer_dependencies'] if normal else value['range_input']['dependencies'])==c.support_identity(),'prepare_source_changed')
    expected={'candidate.json',NAME+'-input.json',NAME+'-browser.js',NAME+'-capture.js','prepare-candidate.sh'}
    if normal:
        expected.remove('prepare-candidate.sh')
        c.keys(files,'schema members execution_authorized review_required delivery_update_required')
        c.require(files['schema']=='viewer-normal-candidate-files-v2' and files['execution_authorized'] is False and
            files['review_required'] is True and files['delivery_update_required'] is True and
            set(files['members'])==expected,'normal_prepare_members')
    else:
        c.keys(files,'schema members executable review_required delivery_verified browser_executed')
        c.require(files['schema']=='viewer-capture-candidate-files-v1' and files['executable'] is False and
            files['review_required'] is True and files['delivery_verified'] is False and
            files['browser_executed'] is False and set(files['members'])==expected,'prepare_members')
    c.require({p.name for p in directory.iterdir()}==expected|{'files.json'},'prepare_repeated_or_extra')
    c.require(files['members']=={n:c.file_id(directory/n) for n in expected},'prepare_files_changed')
    served=c.safe_path(c.VIEWER/'build/demo')
    c.require(served.is_dir(),'served_directory')
    for url,identity in value['source_delivery'].items():
        c.require(url.startswith('/js/') and Path(url).name==url[4:],'served_member')
        c.require(c.file_id(c.VIEWER/'demo'/url.lstrip('/'))==identity and
                  c.file_id(served/url.lstrip('/'))==identity,'source_served_mismatch')
    for link,target in value['preparation']['existing_publication'].items():
        p=Path(link);c.safe_path(p.parent)
        c.require(p.is_symlink() and os.readlink(p)==target and Path(target).exists(),'publication_target')
    destination=Path(value['preparation']['destination']);c.safe_path(destination.parent)
    c.require(not os.path.lexists(destination),'publication_exists')
    # Exclusive attempt is durable evidence, and prohibits an automatic repeat.
    write(directory/'prepare-attempt.json',c.canonical(dict(authorization=authorization,plan=value['preparation'])))
    try:
        destination.symlink_to(directory,target_is_directory=True)
        c.require(destination.is_symlink() and os.readlink(destination)==str(directory),'publication_readback')
        result=dict(schema='viewer-normal-prepare-result-v2' if normal else 'viewer-old-prepare-result-v1',status='prepared-not-executed',
            destination=str(destination),candidate=authorization['candidate'],files=authorization['files'],
            source_and_disk_delivery_equal=True,browser_loaded_source_verified=False,
            browser_executed=False,build=False,server_started=False)
        write(directory/'prepare-result.json',c.canonical(result))
        c.require(c.read_json(directory/'prepare-result.json')==result,'prepare_readback')
        return result
    except BaseException as error:
        write(directory/'prepare-failure.json',c.canonical(dict(status='incomplete',
            error_type=type(error).__name__,retry=False,cleanup=False)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True)
    parser.add_argument('--authorization',required=True)
    args=parser.parse_args()
    c.require(__import__('sys').executable==c.PYTHON,'python')
    print(c.canonical(prepare(args.directory,c.read_json(args.authorization))).decode())


if __name__=='__main__':main()
