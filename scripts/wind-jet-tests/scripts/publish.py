#!/usr/bin/env python3
"""Publish verified derived wind-jet results without changing earlier campaigns."""
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
from guard import verify, digest, safe_relative, write_json
from reconcile import reconcile

PAGE='wind-jet-tests'


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
        html=html.replace(anchor,anchor+'<a href="'+href+'">Wind and jets</a>',1)
    if 'id="wind-jet-campaign"' not in html:
        anchor='<main id="main" class="wrap">'
        if anchor not in html:raise RuntimeError('Gallery main anchor changed')
        html=html.replace(anchor,anchor+'\n<section class="campaign-panel" id="wind-jet-campaign"><p class="eyebrow">Wind-forced experiments</p><h2><a href="'+href+'">When does the reduced vorticity equation work, and when do jets grow?</a></h2><p>Sustained wind forcing, forecasts from the instantaneous transverse mean, and paired jet perturbations. The reduction and jet formation are tested separately with criteria fixed before the runs.</p></section>',1)
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


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gallery',type=Path);a=p.parse_args()
    root=a.root.resolve();source=verify(root);paths=json.loads((root/'runtime/paths.json').read_text())
    gallery=a.gallery.resolve() if a.gallery else Path(paths['publisher'])
    publisher=gallery/'.unity-publisher';env=dict(os.environ)
    env['GIT_SSH_COMMAND']=' '.join(['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
        '-o',shlex.quote('UserKnownHostsFile='+str(publisher/'known_hosts')),'-i',shlex.quote(str(publisher/'id_ed25519'))])
    with (publisher/'publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        reconcile(root)
        approved=verify_completed(root,source)
        for attempt in range(3):
            with tempfile.TemporaryDirectory(prefix='wind-jet-publish-',dir=publisher) as scratch:
                checkout=Path(scratch)/'repository'
                subprocess.run(['git','clone','--quiet','--depth=1','--single-branch','--branch','main',
                  'https://github.com/npizzo/wave-current-gallery.git',str(checkout)],check=True,env=env)
                target=checkout/'gallery/experiments'/PAGE
                if target.exists():shutil.rmtree(target)
                target.mkdir(parents=True)
                approved_file=Path(scratch)/'approved-runs.json';write_json(approved_file,approved)
                subprocess.run([sys.executable,str(root/'analysis/analyze_wind.py'),'--root',str(root),
                    '--output',str(target),'--approved-runs',str(approved_file)],check=True)
                shutil.copy2(root/'PROTOCOL.txt',target/'PROTOCOL.txt')
                shutil.copy2(root/'campaign.json',target/'campaign.json')
                write_json(target/'source-manifest.json',source)
                if sum(p.stat().st_size for p in target.rglob('*') if p.is_file())>30*1024**2:
                    raise RuntimeError('New wind page exceeds its 30 MiB allowance')
                copy_source(root,checkout,source)
                landing=checkout/'gallery/index.html';landing.write_text(update_landing(landing.read_text()))
                subprocess.run([sys.executable,str(checkout/'scripts/check_gallery.py'),str(checkout/'gallery')],check=True)
                verify(root)
                allowed=['gallery/experiments/'+PAGE,'gallery/index.html','scripts/'+PAGE]
                subprocess.run(['git','add','--',*allowed],cwd=checkout,check=True)
                state=subprocess.run(['git','diff','--cached','--quiet'],cwd=checkout).returncode
                if state==0:print('Gallery already matches verified campaign state');return
                if state!=1:raise RuntimeError('Cannot inspect staged publication')
                changed=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=checkout,text=True).splitlines()
                if not all(any(n==p or n.startswith(p+'/') for p in allowed) for n in changed):
                    raise RuntimeError('Publication staged an unrelated file')
                subprocess.run(['git','-c','user.name=Wave-current gallery','-c','user.email=7075590+npizzo@users.noreply.github.com',
                  'commit','-m','Update wind-forced closure and jet-formation tests'],cwd=checkout,check=True)
                verify(root)
                if subprocess.run(['git','push','git@github.com:npizzo/wave-current-gallery.git','HEAD:main'],cwd=checkout,env=env).returncode==0:
                    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
                    write_json(root/'last-publication.json',dict(commit=commit,verified_runs=len(approved),source_version=source['version'],
                        url='https://npizzo.github.io/wave-current-gallery/experiments/'+PAGE+'/'))
                    print('Published https://npizzo.github.io/wave-current-gallery/experiments/'+PAGE+'/');return
        raise RuntimeError('Publication failed; all campaign outputs are preserved')

if __name__=='__main__':main()
