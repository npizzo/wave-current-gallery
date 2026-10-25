#!/usr/bin/env python3
"""Verify the complete audit, aggregate it, and publish only small derived products."""
import datetime, fcntl, json, os, shlex, shutil, subprocess, sys, tempfile
from pathlib import Path
from submit import SEEDS,check_campaign,digest,write

def main():
    state,campaign,gallery=map(Path,sys.argv[1:4]); results=state/'results'
    record=json.loads((state/'submission.json').read_text())
    for name,sha in record['source_sha256'].items():
        if digest(state/'code'/name)!=sha: raise RuntimeError('Frozen analysis source changed')
    if check_campaign(campaign)!=record['campaign_fingerprints']: raise RuntimeError('Campaign changed')
    receipts=[]
    for seed in SEEDS:
        folder=results/f'seed-{seed}'; receipt=json.loads((folder/'completed.json').read_text())
        if not receipt.get('completed') or receipt['source_version']!=record['version']: raise RuntimeError('Missing matching seed completion')
        for name,sha in receipt['files'].items():
            if digest(folder/name)!=sha: raise RuntimeError(f'Audit output changed: {folder/name}')
        receipts.append(receipt)
    subprocess.run([sys.executable,str(state/'code/report.py'),'--folder',str(results)],check=True)
    product_names=['ensemble.csv','metrics.csv','summary.json','briefsummary.txt','diagnostic-ratios.png']
    if not all((results/p).is_file() for p in product_names): raise RuntimeError('Missing report product; no publication')
    provenance={'completed':True,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_version':record['version'],'upstream_commit':'9ff503ca9bf4672a30f32c5dcfcc336f0be6faed',
        'seeds':SEEDS,'source_sha256':record['source_sha256'],'seed_receipts':receipts,
        'files':{p:digest(results/p) for p in product_names},
        'scope':'Full-field geometrical corrections to scalar-q transport. No material-conservation, closed-budget or numerical-convergence claim.'}
    write(results/'completed.json',provenance)
    publisher=gallery/'.unity-publisher'
    env=dict(os.environ)
    env['GIT_SSH_COMMAND']=' '.join(['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
        '-o',shlex.quote('UserKnownHostsFile='+str(publisher/'known_hosts')),'-i',shlex.quote(str(publisher/'id_ed25519'))])
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wave-packet vorticity diagnostics</title><style>body{font:17px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 22px;color:#18232e}img{width:100%}a{color:#126090}</style>
<p><a href="../../">Surface waves &amp; turbulence</a></p><h1>Vorticity diagnostics</h1>
<p>Seven matched wave/no-wave pairs, with the full coherent current retained. The analysis tests the three-dimensional corrections to scalar vorticity transport.</p>
<p>Both derivative stencils exclude the same four cells at each vertical boundary. Native full-grid incompressibility is checked separately. Local packet-age bins and all component norms are in the tables.</p>
<img src="diagnostic-ratios.png" alt="Measured divergence and vorticity correction ratios">
<p><a href="metrics.csv">Ratios (CSV)</a> · <a href="ensemble.csv">All term norms (CSV)</a> · <a href="summary.json">Definitions</a> · <a href="METHODS.txt">Methods</a> · <a href="completed.json">Provenance</a></p>
<p>A small geometrical correction does not by itself establish material conservation: saved snapshots do not close a temporal budget or measure implicit numerical dissipation. Two stencil widths test derivative sensitivity, not simulation convergence.</p></html>'''
    with (publisher/'publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for attempt in range(3):
            with tempfile.TemporaryDirectory(prefix='vorticity-publish-',dir=publisher) as scratch:
                checkout=Path(scratch)/'repository'
                subprocess.run(['git','clone','--single-branch','--branch','main','https://github.com/npizzo/wave-current-gallery.git',str(checkout)],check=True,env=env)
                target=checkout/'gallery/audits/vorticity'; target.mkdir(parents=True,exist_ok=True)
                for name in product_names+['completed.json']: shutil.copy2(results/name,target/name)
                shutil.copy2(state/'code/METHODS.txt',target/'METHODS.txt')
                (target/'index.html').write_text(html)
                subprocess.run([sys.executable,str(checkout/'scripts/check_gallery.py'),str(checkout/'gallery')],check=True)
                subprocess.run(['git','add','--','gallery/audits/vorticity'],cwd=checkout,check=True)
                changed=subprocess.run(['git','diff','--cached','--quiet'],cwd=checkout).returncode
                if changed==0: print('Audit already published'); return
                if changed!=1: raise RuntimeError('Cannot inspect publication changes')
                subprocess.run(['git','-c','user.name=Wave-current gallery','-c','user.email=7075590+npizzo@users.noreply.github.com',
                                'commit','-m','Publish packet vorticity balance diagnostics'],cwd=checkout,check=True)
                if subprocess.run(['git','push','git@github.com:npizzo/wave-current-gallery.git','HEAD:main'],cwd=checkout,env=env).returncode==0:
                    print('Published: https://npizzo.github.io/wave-current-gallery/audits/vorticity/'); return
        raise RuntimeError('Publication failed; computed results remain in '+str(results))
if __name__=='__main__': main()
