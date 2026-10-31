#!/usr/bin/env python3
"""Publish verified Eulerian-shear pilot results without changing earlier campaigns."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from guard import verify, digest, safe_relative, write_json
from reconcile import reconcile
from references import verify_references

PAGE='shear-instability'


def verify_completed(root,source):
    approved=[]
    backup=Path(json.loads((root/'runtime/paths.json').read_text())['backup'])
    for index,case in enumerate(json.loads((root/'campaign.json').read_text())['runs']):
        path=root/'data/status'/case['id']/'completed.json'
        if not path.is_file():continue
        receipt=json.loads(path.read_text())
        if (receipt.get('completed') is not True or receipt.get('persistent_backup_verified') is not True
            or receipt['source_version']!=source['version'] or receipt['id']!=case['id'] or receipt['array_index']!=index):
            raise RuntimeError('Invalid completed experiment receipt')
        saved=backup/'runs'/case['id']
        proof=json.loads((saved/'COMPLETE.json').read_text())
        # Every member was hashed during archive creation. The analysis separately
        # rehashes every diagnostic it consumes on each publication. Native volumes
        # are private and need not be reread for every incremental gallery update.
        if (proof.get('complete') is not True or proof['source_version']!=source['version']
                or proof['receipt_sha256']!=digest(saved/'completion-receipt.json')
                or json.loads((saved/'completion-receipt.json').read_text())!=receipt
                or not (saved/'native-output.tar.gz').is_file()
                or (saved/'native-output.tar.gz').stat().st_size==0):
            raise RuntimeError('Persistent backup proof is missing or differs')
        approved.append(case['id'])
    return approved


def update_landing(html):
    href='experiments/'+PAGE+'/'
    if 'href="'+href+'"' not in html:
        anchor='<div class="nav-links">'
        if anchor not in html:raise RuntimeError('Gallery navigation anchor changed')
        html=html.replace(anchor,anchor+'<a href="'+href+'">Shear instability</a>',1)
    if 'id="shear-instability-campaign"' not in html:
        anchor='<main id="main" class="wrap">'
        if anchor not in html:raise RuntimeError('Gallery main anchor changed')
        html=html.replace(anchor,anchor+'\n<section class="campaign-panel" id="shear-instability-campaign"><p class="eyebrow">Eulerian-shear pilot</p><h2><a href="'+href+'">Does Eulerian shear predict jet disruption?</a></h2><p>A small pilot varies added wind and compares full three-dimensional flow with its transverse-mean evolution. Existing matched runs provide the zero- and strong-wind reference. Signed Eulerian shear, effective beta, roll growth and jet survival are tested together.</p></section>',1)
    return html


def copy_source(root,checkout,source):
    destination=checkout/'scripts'/PAGE
    if destination.exists():shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for name,sha in source['files'].items():
        rel=safe_relative(name);target=destination/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        if digest(root/rel)!=sha:raise RuntimeError('Source changed during publication')
        shutil.copy2(root/rel,target)
    write_json(destination/'source-manifest.json',source)


def confirmed_push(checkout,env):
    """A failed transport can still have accepted the commit; inspect before retry."""
    outcome=subprocess.run(['git','push','git@github.com:npizzo/wave-current-gallery.git','HEAD:main'],cwd=checkout,env=env)
    if outcome.returncode==0:return True
    fetched=subprocess.run(['git','fetch','--quiet','https://github.com/npizzo/wave-current-gallery.git','main'],cwd=checkout,env=env)
    if fetched.returncode:return False
    return subprocess.run(['git','merge-base','--is-ancestor','HEAD','FETCH_HEAD'],cwd=checkout).returncode==0


def publication_receipt(root,source,approved,external,commit):
    value=dict(commit=commit,verified_runs=len(approved),verified_external_references=len(external),
        source_version=source['version'],url='https://npizzo.github.io/wave-current-gallery/experiments/'+PAGE+'/')
    write_json(root/'last-publication.json',value)
    print('Published '+value['url'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gallery',type=Path);a=p.parse_args()
    root=a.root.resolve();source=verify(root);paths=json.loads((root/'runtime/paths.json').read_text())
    gallery=a.gallery.resolve() if a.gallery else Path(paths['publisher'])
    publisher=gallery/'.unity-publisher';env=dict(os.environ)
    env['GIT_SSH_COMMAND']=' '.join(['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
        '-o',shlex.quote('UserKnownHostsFile='+str(publisher/'known_hosts')),'-i',shlex.quote(str(publisher/'id_ed25519'))])
    with (publisher/'publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        reconcile(root);approved=verify_completed(root,source);external=verify_references(root,source)
        with tempfile.TemporaryDirectory(prefix='shear-pilot-render-',dir=publisher) as renderdir:
            render=Path(renderdir);rendered=render/'page';rendered.mkdir()
            approved_file=render/'approved-runs.json';write_json(approved_file,approved)
            subprocess.run([sys.executable,str(root/'analysis/analyze_shear.py'),'--root',str(root),
                '--output',str(rendered),'--approved-runs',str(approved_file)],check=True)
            for filename in ('PROTOCOL.txt','campaign.json','references-manifest.json'):
                shutil.copy2(root/filename,rendered/filename)
            write_json(rendered/'source-manifest.json',source)
            if sum(p.stat().st_size for p in rendered.rglob('*') if p.is_file())>30*1024**2:
                raise RuntimeError('New shear page exceeds its30MiB allowance')
            verify(root)
            for attempt in range(5):
                if attempt:time.sleep(min(2**attempt,20))
                with tempfile.TemporaryDirectory(prefix='shear-pilot-publish-',dir=publisher) as scratch:
                    checkout=Path(scratch)/'repository'
                    cloned=subprocess.run(['git','clone','--quiet','--depth=1','--single-branch','--branch','main',
                        'https://github.com/npizzo/wave-current-gallery.git',str(checkout)],env=env)
                    if cloned.returncode:continue
                    target=checkout/'gallery/experiments'/PAGE
                    if target.exists():shutil.rmtree(target)
                    shutil.copytree(rendered,target)
                    copy_source(root,checkout,source)
                    landing=checkout/'gallery/index.html';landing.write_text(update_landing(landing.read_text()))
                    subprocess.run([sys.executable,str(checkout/'scripts/check_gallery.py'),str(checkout/'gallery')],check=True)
                    verify(root)
                    allowed=['gallery/experiments/'+PAGE,'gallery/index.html','scripts/'+PAGE]
                    subprocess.run(['git','add','--',*allowed],cwd=checkout,check=True)
                    state=subprocess.run(['git','diff','--cached','--quiet'],cwd=checkout).returncode
                    if state==0:
                        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
                        publication_receipt(root,source,approved,external,commit);return
                    if state!=1:raise RuntimeError('Cannot inspect staged publication')
                    changed=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=checkout,text=True).splitlines()
                    if not all(any(n==p or n.startswith(p+'/') for p in allowed) for n in changed):
                        raise RuntimeError('Publication staged an unrelated file')
                    subprocess.run(['git','-c','user.name=Wave-current gallery','-c','user.email=7075590+npizzo@users.noreply.github.com',
                        'commit','-m','Update Eulerian-shear instability pilot'],cwd=checkout,check=True)
                    verify(root)
                    if confirmed_push(checkout,env):
                        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
                        publication_receipt(root,source,approved,external,commit);return
        raise RuntimeError('Publication failed after bounded retries; all campaign outputs are preserved')

if __name__=='__main__':main()
