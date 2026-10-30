#!/usr/bin/env python3
"""Idempotent, journalled Slurm submission; ambiguous outcomes never duplicate jobs."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid
from guard import verify, write_json, digest
from inputs import preserve_inputs


def utcnow(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def find_submission(token, since):
    """Inspect both scheduler views; task IDs collapse to their parent array ID."""
    queries = [
        ['squeue','--array','--noheader','--user',os.environ.get('USER','nicholas_pizzo_uri_edu'), '--format=%i|%k'],
        ['sacct','--array','-X','-n','-P','--user',os.environ.get('USER','nicholas_pizzo_uri_edu'),
         '--starttime',since[:10], '--format=JobID%64,Comment%200']]
    found=set(); successes=0
    for command in queries:
        result=subprocess.run(command,text=True,capture_output=True)
        if result.returncode: continue
        successes+=1
        for line in result.stdout.splitlines():
            fields=[v.strip() for v in line.split('|')]
            if len(fields)>=2 and fields[1]==token:
                job=fields[0].split('_')[0]
                if re.fullmatch(r'\d+',job): found.add(job)
    if len(found)>1: raise RuntimeError('Several jobs match one submission token; manual review required')
    return next(iter(found),None), successes


def submit_once(root, label, script, options):
    journal=root/'submission-journal'/f'{label}.json'
    if journal.exists():
        record=json.loads(journal.read_text())
        if record['script']!=script or record['options']!=list(options):
            raise RuntimeError('Submission intent changed for '+label)
        if record.get('job_id'): return str(record['job_id'])
        job,views=find_submission(record['token'],record['created_at'])
        if job:
            record.update(job_id=job,state='recovered_from_slurm',recovered_at=utcnow())
            write_json(journal,record);return job
        if record['state']!='prepared':
            raise RuntimeError(f'Ambiguous submission for {label}; inspected {views} Slurm views but no exact token yet. No duplicate submitted.')
    else:
        record=dict(label=label,script=script,options=list(options),token='winddecay-'+uuid.uuid4().hex,
                    created_at=utcnow(),state='prepared')
        write_json(journal,record)
    record.update(state='attempting',attempted_at=utcnow());write_json(journal,record)
    command=['sbatch','--parsable','--comment='+record['token'],*options,str(root/script)]
    result=subprocess.run(command,cwd=root,text=True,capture_output=True)
    value=result.stdout.strip().split(';')[0]
    if result.returncode!=0 or not re.fullmatch(r'\d+',value):
        record.update(state='ambiguous',returncode=result.returncode,stdout=result.stdout[-2000:],stderr=result.stderr[-4000:])
        write_json(journal,record)
        job,_=find_submission(record['token'],record['created_at'])
        if not job: raise RuntimeError('Submission outcome needs Slurm review: '+label+'; '+result.stderr[-1500:])
        value=job
    record.update(state='submitted',job_id=value,confirmed_at=utcnow());write_json(journal,record)
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--concurrency',type=int,default=2);a=p.parse_args()
    root=a.root.resolve();source=verify(root,require_validation=True)
    if not 1<=a.concurrency<=2: raise ValueError('Campaign permits at most two simultaneous GPUs')
    (root/'logs').mkdir(exist_ok=True)
    paths=json.loads((root/'runtime/paths.json').read_text());backup=Path(paths['backup'])
    backup.mkdir(parents=True,exist_ok=True);source_backup=backup/('source-'+source['version'])
    preserve_inputs(root,backup,source)
    for name,sha in source['files'].items():
        destination=source_backup/name;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() and digest(destination)!=sha: raise RuntimeError('Persistent source backup differs')
        if not destination.exists(): shutil.copy2(root/name,destination)
        if digest(destination)!=sha: raise RuntimeError('Persistent source copy failed verification')
    shutil.copy2(root/'validation/passed.json',source_backup/'validation-passed.json')
    path=root/'submission.json'
    with (root/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        count=len(json.loads((root/'campaign.json').read_text())['runs'])
        if path.exists():
            record=json.loads(path.read_text())
            if record['source_version']!=source['version'] or record['runs']!=count:
                raise RuntimeError('Existing submission belongs to different source or matrix')
            array=record['array']
        else:
            array=submit_once(root,'array','scripts/run.sbatch',[
                f'--array=0-{count-1}%{a.concurrency}','--output=logs/run-%A_%a.out','--error=logs/run-%A_%a.err'])
            record=dict(source_version=source['version'],array=array,runs=count,publishers=[],submitted_at=utcnow(),state='attaching')
            write_json(path,record)
        dependencies=[f'{array}_{i}' for i in range(count)]+['immediate',array]
        for index,dependency in enumerate(dependencies):
            if any(item['after']==dependency for item in record['publishers']):continue
            options=[] if dependency=='immediate' else [f'--dependency=afterany:{dependency}']
            options+=['--output=logs/publish-%j.out','--error=logs/publish-%j.err']
            job=submit_once(root,f'publisher-{index}','scripts/publish.sbatch',options)
            record['publishers'].append({'job':job,'after':dependency});write_json(path,record)
        record.update(state='attached',attached_at=utcnow());write_json(path,record)
        for name in ('campaign.json','source-manifest.json','submission.json','PROTOCOL.txt','source_provenance.toml'):
            shutil.copy2(root/name,backup/name)
        print(json.dumps(record,indent=2))

if __name__=='__main__':main()
