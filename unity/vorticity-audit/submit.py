#!/usr/bin/env python3
"""Submit only read-only snapshot analysis; freeze its source outside the campaign."""
import argparse, fcntl, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode=True

SEEDS=(1,2,3,4,6,7,8)
def digest(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,obj):
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(obj,indent=2)+'\n'); temp.replace(path)
def check_campaign(root):
    sys.path.insert(0,str(root/'scripts'))
    import guard
    sub=guard.check_submission(root)
    final=guard.read_json(root/'data/analysis/completed.json')
    if not final.get('completed') or final['fingerprints']!=sub['fingerprints']:
        raise RuntimeError('Final campaign analysis is missing or source has changed')
    for index in range(8):
        item=guard.read_json(root/f'data/receipts/array-{index}-completed.json')
        if not item.get('completed') or item['fingerprints']!=sub['fingerprints']:
            raise RuntimeError(f'Array member {index} is not verified complete')
    return sub['fingerprints']
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True,type=Path)
    a=p.parse_args(); campaign=a.campaign.resolve()
    source=Path(__file__).resolve().parent; gallery=source.parents[1]
    if not (gallery/'.git').exists(): raise RuntimeError('Run from the Unity gallery checkout')
    if gallery==campaign or gallery.is_relative_to(campaign): raise RuntimeError('Keep audit outside scientific campaign')
    fingerprints=check_campaign(campaign)
    files={p.name:digest(p) for p in source.iterdir() if p.is_file() and p.suffix in ('.jl','.py','.sbatch','.txt')}
    version=hashlib.sha256(json.dumps({'code':files,'campaign':fingerprints},sort_keys=True).encode()).hexdigest()[:16]
    state=gallery/'.unity-publisher'/('vorticity-audit-'+version); state.mkdir(parents=True,exist_ok=True)
    with (state/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record_path=state/'submission.json'
        if record_path.exists():
            print(record_path.read_text()); print('This audit is already submitted or its submission was interrupted. No duplicate jobs submitted.'); return
        code=state/'code'; code.mkdir(exist_ok=True)
        for name in files: shutil.copy2(source/name,code/name)
        (state/'logs').mkdir(exist_ok=True); (state/'results').mkdir(exist_ok=True)
        if not (gallery/'.unity-publisher/id_ed25519').is_file(): raise RuntimeError('Existing gallery publishing key is missing')
        record={'status':'submitting','campaign':str(campaign),'gallery':str(gallery),'version':version,
                'source_sha256':files,'campaign_fingerprints':fingerprints,'seeds':SEEDS,'jobs':[]}
        write(record_path,record)
        def submit(script,dependency=None):
            command=['sbatch','--parsable',f'--output={state}/logs/%x-%A_%a.out',f'--error={state}/logs/%x-%A_%a.err']
            if dependency: command += [f'--dependency=afterok:{dependency}','--kill-on-invalid-dep=yes']
            command += [str(code/script),str(state),str(campaign),str(gallery)]
            jid=subprocess.check_output(command,cwd=gallery,text=True).strip().split(';')[0]
            if not jid.isdigit(): raise RuntimeError('Unrecognized scheduler response; inspect queue before retrying')
            record['jobs'].append({'job_id':jid,'script':script}); write(record_path,record)
            return jid
        array=submit('audit.sbatch'); report=submit('report.sbatch',array)
        record['status']='submitted'; write(record_path,record)
        print(json.dumps(record,indent=2))
        print(f'Submitted snapshot audit {array} (two CPU tasks at a time), followed by report/publication {report}.')
        print('No simulation was submitted or modified. Results will appear at https://npizzo.github.io/wave-current-gallery/audits/vorticity/')
if __name__=='__main__': main()
