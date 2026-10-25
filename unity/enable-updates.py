"""Attach publishing jobs to an already-submitted scientific campaign."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

TERMINAL = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def atomic(path, data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n'); temp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',required=True,type=Path)
    parser.add_argument('--repository',default='npizzo/wave-current-gallery')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',args.repository): raise ValueError('Expected owner/repository')
    root=Path(__file__).resolve().parents[1]
    campaign=args.campaign.resolve()
    if root==campaign or root.is_relative_to(campaign): raise ValueError('Gallery must be outside the scientific campaign')
    state=root/'.unity-publisher'; state.mkdir(exist_ok=True,mode=0o700)
    (state/'logs').mkdir(exist_ok=True)
    record_path=state/'updates.json'
    with (state/'attach.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if record_path.exists(): raise RuntimeError('Updates already attached or submission interrupted. Inspect .unity-publisher/updates.json and squeue; do not duplicate.')
        submission=json.loads((campaign/'data/submission.json').read_text())
        array,analysis=str(submission['array_job']),str(submission['analysis_job'])
        assert array.isdigit() and analysis.isdigit()
        key=state/'id_ed25519'; hosts=state/'known_hosts'
        if not key.is_file() or not hosts.is_file(): raise RuntimeError('Run bash unity/setup-key.sh and add the public deploy key first.')
        env=dict(os.environ)
        env['GIT_SSH_COMMAND']=' '.join(['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
            '-o',shlex.quote('UserKnownHostsFile='+str(hosts)),'-i',shlex.quote(str(key))])
        subprocess.run(['git','push','--dry-run',f'git@github.com:{args.repository}.git','HEAD:main'],cwd=root,env=env,check=True)
        # Current scheduler state avoids attaching dependencies to already-purged jobs.
        text=subprocess.check_output(['sacct','--array','-X','-n','-P','-j',array+','+analysis,'--format=JobID,State'],text=True)
        states={}
        for line in text.splitlines():
            fields=line.split('|')
            if len(fields)>=2: states[fields[0]]=fields[1].split()[0].rstrip('+')
        parents=[f'{array}_{i}' for i in range(8)]+[analysis]
        missing=[p for p in parents if p not in states]
        if missing: raise RuntimeError('Could not verify scheduler state for '+', '.join(missing)+'. No publishing jobs submitted.')
        record={'campaign_array':array,'campaign_analysis':analysis,'repository':args.repository,'state':'submitting','jobs':[]}
        atomic(record_path,record)
        def submit(parent=None):
            argv=['sbatch','--parsable',f'--output={state}/logs/publish-%j.out',f'--error={state}/logs/publish-%j.err']
            if parent: argv += [f'--dependency=afterany:{parent}','--kill-on-invalid-dep=yes']
            argv += [str(root/'unity/publish.sbatch'),str(root),str(campaign),args.repository]
            job=subprocess.check_output(argv,text=True,cwd=root).strip().split(';')[0]
            if not job.isdigit(): raise RuntimeError('Unrecognized sbatch result; inspect squeue before retrying')
            record['jobs'].append({'job_id':job,'after':parent or 'immediate'})
            atomic(record_path,record)
        submit()
        for parent in parents:
            if states[parent] not in TERMINAL: submit(parent)
        record['state']='attached'; atomic(record_path,record)
        print(json.dumps(record,indent=2))
        print('Gallery updates attached. No simulations were submitted or changed. Safe to log out.')


if __name__=='__main__': main()

